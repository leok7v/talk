#include "transformer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { DIM = 512, LAYERS = 24, HEADS = 8, GROUPS = 2, HEAD = 64, FFN = 768,
       INNER = 768, STATE = 64, SCANS = 24, SCAN = 32, CONV = 4, XBC = 896,
       PROJ = 1688, VOCAB = 32768, CONTEXT = 16384, EOS = 11, IM_END = 228,
       TOOL_CALL = 21, TOOL_END = 22 };

struct layer {
    const float * attn_norm;
    struct matrix q;
    struct matrix k;
    struct matrix v;
    struct matrix out;
    struct matrix ssm_in;
    const float * conv_w;
    const float * conv_b;
    const float * dt_b;
    const float * a;
    const float * d;
    struct matrix ssm_out;
    const float * ffn_norm;
    struct matrix up;
    struct matrix gate;
    struct matrix down;
    float * keys;
    float * values;
    float * state;
    float * conv;
};

struct falcon {
    struct llm llm;
    struct gguf gguf;
    struct layer layers[LAYERS];
    const uint8_t * embed;
    uint32_t embed_type;
    const float * output_norm;
    struct matrix output;
    float rope_rate[HEAD / 2];
    struct chat chat;
    int32_t position;
    int32_t row;
    char * tools;
    float last[DIM];
};

static void layer_open(struct falcon * self, int32_t l) {
    self->layers[l] = (struct layer){
        .attn_norm = gguf_tensor(&self->gguf, "blk.%d.attn_norm.weight", l),
        .q = gguf_matrix(&self->gguf, "blk.%d.attn_q.weight", l),
        .k = gguf_matrix(&self->gguf, "blk.%d.attn_k.weight", l),
        .v = gguf_matrix(&self->gguf, "blk.%d.attn_v.weight", l),
        .out = gguf_matrix(&self->gguf, "blk.%d.attn_output.weight", l),
        .ssm_in = gguf_matrix(&self->gguf, "blk.%d.ssm_in.weight", l),
        .conv_w = gguf_tensor(&self->gguf, "blk.%d.ssm_conv1d.weight", l),
        .conv_b = gguf_tensor(&self->gguf, "blk.%d.ssm_conv1d.bias", l),
        .dt_b = gguf_tensor(&self->gguf, "blk.%d.ssm_dt.bias", l),
        .a = gguf_tensor(&self->gguf, "blk.%d.ssm_a", l),
        .d = gguf_tensor(&self->gguf, "blk.%d.ssm_d", l),
        .ssm_out = gguf_matrix(&self->gguf, "blk.%d.ssm_out.weight", l),
        .ffn_norm = gguf_tensor(&self->gguf, "blk.%d.ffn_norm", l),
        .up = gguf_matrix(&self->gguf, "blk.%d.ffn_up.weight", l),
        .gate = gguf_matrix(&self->gguf, "blk.%d.ffn_gate.weight", l),
        .down = gguf_matrix(&self->gguf, "blk.%d.ffn_down.weight", l),
        .keys = floats((size_t)self->chat.context * GROUPS * HEAD),
        .values = floats((size_t)self->chat.context * GROUPS * HEAD),
        .state = floats((size_t)INNER * STATE),
        .conv = floats((size_t)XBC * (CONV - 1))
    };
    self->chat.track[2 * l] = self->layers[l].keys;
    self->chat.track[2 * l + 1] = self->layers[l].values;
}

static float silu(float x) {
    return x / (1.0f + expf(-x));
}

static float dot(const float * restrict a, const float * restrict b,
                 int32_t n) {
    #pragma clang fp reassociate(on)
    float sum = 0.0f;
    for (int32_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

static void norm(float * y, const float * x, const float * weight,
                 int32_t rows) {
    for (int32_t r = 0; r < rows; r++) {
        const float * in = x + r * DIM;
        const float gain = 1.0f / sqrtf(dot(in, in, DIM) / (float)DIM +
                                        1e-5f);
        for (int32_t i = 0; i < DIM; i++) {
            y[r * DIM + i] = in[i] * gain * weight[i];
        }
    }
}

static void rope(const struct falcon * self, float * x, int32_t heads,
                 int32_t position) {
    for (int32_t i = 0; i < HEAD / 2; i++) {
        const float c = cosf((float)position * self->rope_rate[i]);
        const float s = sinf((float)position * self->rope_rate[i]);
        for (int32_t h = 0; h < heads; h++) {
            float * p = x + h * HEAD;
            const float a = p[i];
            const float b = p[i + HEAD / 2];
            p[i] = a * c - b * s;
            p[i + HEAD / 2] = a * s + b * c;
        }
    }
}

static void attention(struct falcon * self, float * y, const float * x,
                      const struct layer * m, int32_t rows) {
    float * q = floats((size_t)rows * DIM);
    float * mix = floats((size_t)rows * DIM);
    float * k = m->keys + self->row * GROUPS * HEAD;
    const struct attention a = {
        .backend = self->gguf.backend, .keys = m->keys, .values = m->values,
        .heads = HEADS, .groups = GROUPS, .head = HEAD
    };
    gguf_linear(&self->gguf, q, x, &m->q, DIM, DIM, rows);
    gguf_linear(&self->gguf, k, x, &m->k, GROUPS * HEAD, DIM, rows);
    gguf_linear(&self->gguf, m->values + self->row * GROUPS * HEAD,
                x, &m->v, GROUPS * HEAD, DIM, rows);
    for (int32_t r = 0; r < rows; r++) {
        rope(self, q + r * DIM, HEADS, self->position + r);
        rope(self, k + r * GROUPS * HEAD, GROUPS, self->position + r);
    }
    chat_attend(&a, mix, q, self->row, rows);
    gguf_linear(&self->gguf, y, mix, &m->out, DIM, DIM, rows);
    free(q);
    free(mix);
}

static void silus(struct falcon * self, float * x, int32_t n, bool wide) {
    float * e = floats(wide ? (size_t)n : 1);
    for (int32_t i = 0; i < n && wide; i++) {
        e[i] = -x[i];
    }
    if (wide) {
        self->gguf.backend->exp(self->gguf.backend, e, e, n);
    }
    for (int32_t i = 0; i < n; i++) {
        x[i] = wide ? x[i] / (1.0f + e[i]) : silu(x[i]);
    }
    free(e);
}

static void convolve(struct falcon * self, float * xbc,
                     const struct layer * m, bool wide) {
    for (int32_t c = 0; c < XBC; c++) {
        float * past = m->conv + c * (CONV - 1);
        const float * w = m->conv_w + c * CONV;
        const float now = xbc[c];
        xbc[c] = m->conv_b[c] + past[0] * w[0] + past[1] * w[1] +
                 past[2] * w[2] + now * w[3];
        past[0] = past[1];
        past[1] = past[2];
        past[2] = now;
    }
    silus(self, xbc, XBC, wide);
}

static void scan(float * y, const float * xbc, const float * dt,
                 const struct layer * m) {
    const float * b = xbc + INNER;
    const float * c = xbc + INNER + STATE;
    for (int32_t h = 0; h < SCANS; h++) {
        const float raw = dt[h] + m->dt_b[h];
        const float step = raw <= 20.0f ? log1pf(expf(raw)) : raw;
        const float decay = expf(step * m->a[h]);
        for (int32_t i = h * SCAN; i < (h + 1) * SCAN; i++) {
            float * s = m->state + i * STATE;
            const float drive = xbc[i] * step;
            for (int32_t j = 0; j < STATE; j++) {
                s[j] = s[j] * decay + b[j] * drive;
            }
            y[i] = dot(s, c, STATE) + xbc[i] * m->d[h];
        }
    }
}

static void mamba(struct falcon * self, float * y, const float * x,
                  const struct layer * m, int32_t rows) {
    float * proj = floats((size_t)rows * PROJ);
    float * mixed = floats((size_t)rows * INNER);
    gguf_linear(&self->gguf, proj, x, &m->ssm_in, PROJ, DIM, rows);
    for (int32_t r = 0; r < rows; r++) {
        float * p = proj + r * PROJ;
        float * to = mixed + r * INNER;
        convolve(self, p + INNER, m, rows > 1);
        scan(to, p + INNER, p + INNER + XBC, m);
        silus(self, p, INNER, rows > 1);
        for (int32_t i = 0; i < INNER; i++) {
            to[i] *= p[i];
        }
    }
    gguf_linear(&self->gguf, y, mixed, &m->ssm_out, DIM, INNER, rows);
    free(proj);
    free(mixed);
}

static void feed(struct falcon * self, float * x, const struct layer * m,
                 int32_t rows) {
    float * n = floats((size_t)rows * DIM);
    float * up = floats((size_t)rows * FFN);
    float * gate = floats((size_t)rows * FFN);
    float * y = floats((size_t)rows * DIM);
    norm(n, x, m->ffn_norm, rows);
    gguf_linear(&self->gguf, up, n, &m->up, FFN, DIM, rows);
    gguf_linear(&self->gguf, gate, n, &m->gate, FFN, DIM, rows);
    silus(self, gate, rows * FFN, rows > 1);
    for (int32_t i = 0; i < rows * FFN; i++) {
        up[i] *= gate[i];
    }
    gguf_linear(&self->gguf, y, up, &m->down, DIM, FFN, rows);
    for (int32_t i = 0; i < rows * DIM; i++) {
        x[i] += y[i];
    }
    free(n);
    free(up);
    free(gate);
    free(y);
}

static void forward(struct falcon * self, const int32_t * tokens,
                    int32_t rows) {
    const size_t size = (size_t)rows * DIM;
    float * x = floats(4 * size);
    float * n = x + size;
    float * a = n + size;
    float * s = a + size;
    for (int32_t r = 0; r < rows; r++) {
        gguf_row(x + r * DIM, self->embed, tokens[r], DIM, self->embed_type);
    }
    for (int32_t l = 0; l < LAYERS; l++) {
        norm(n, x, self->layers[l].attn_norm, rows);
        attention(self, a, n, &self->layers[l], rows);
        mamba(self, s, n, &self->layers[l], rows);
        for (size_t i = 0; i < size; i++) {
            x[i] += a[i] + s[i];
        }
        feed(self, x, &self->layers[l], rows);
    }
    norm(self->last, x + size - DIM, self->output_norm, 1);
    free(x);
}

static void state_forward(void * engine, const int32_t * tokens,
                          int32_t count, int32_t position, int32_t row) {
    struct falcon * self = engine;
    self->position = position;
    self->row = row;
    forward(self, tokens, count);
}

static void state_shift(void * engine, int32_t keep, int32_t count,
                        int32_t rows) {
    struct falcon * self = engine;
    const size_t row = (size_t)GROUPS * HEAD * sizeof(float);
    for (int32_t t = 0; t < LAYERS * 2; t++) {
        float * at = self->chat.track[t] + (size_t)keep * GROUPS * HEAD;
        memmove(at, at + (size_t)count * GROUPS * HEAD,
                (size_t)(rows - keep - count) * row);
    }
}

static void state_save(void * engine, uint8_t * to) {
    const struct falcon * self = engine;
    const size_t state = (size_t)INNER * STATE * sizeof(float);
    const size_t conv = (size_t)XBC * (CONV - 1) * sizeof(float);
    for (int32_t l = 0; l < LAYERS; l++) {
        memcpy(to + l * (state + conv), self->layers[l].state, state);
        memcpy(to + l * (state + conv) + state, self->layers[l].conv, conv);
    }
    memcpy(to + LAYERS * (state + conv), self->last, sizeof self->last);
}

static void state_load(void * engine, const uint8_t * from) {
    struct falcon * self = engine;
    const size_t state = (size_t)INNER * STATE * sizeof(float);
    const size_t conv = (size_t)XBC * (CONV - 1) * sizeof(float);
    for (int32_t l = 0; l < LAYERS; l++) {
        if (from != NULL) {
            memcpy(self->layers[l].state, from + l * (state + conv), state);
            memcpy(self->layers[l].conv, from + l * (state + conv) + state,
                   conv);
        } else {
            memset(self->layers[l].state, 0, state);
            memset(self->layers[l].conv, 0, conv);
        }
    }
    if (from != NULL) {
        memcpy(self->last, from + LAYERS * (state + conv), sizeof self->last);
    }
}

static const struct format format = {
    .start = " <|begin_of_text|>", .open = " <|im_start|>", .middle = "\n",
    .close = "<|im_end|>\n ", .reply = " <|im_start|>assistant\n",
    .plain = "", .think = ""
};

static const struct format tooled = {
    .start = " <|begin_of_text|>", .open = "<|im_start|>", .middle = "\n",
    .close = "<|im_end|>", .reply = "<|im_start|>assistant\n", .plain = "",
    .think = "", .trim = true
};

static const struct format answered = {
    .start = " <|begin_of_text|>", .open = "<|im_start|>", .middle = "\n",
    .close = "\n<|im_end|>", .reply = "<|im_start|>assistant\n", .plain = "",
    .think = "", .trim = true
};

static const char tools_open[] =
    "You are a Falcon assistant skilled in function calling. You are "
    "helpful, respectful, and concise.\n\n# Tools\n\nYou have access to "
    "the following functions. You MUST use them to answer questions when "
    "needed. For each function call, you MUST return a JSON object inside "
    "<tool_call></tool_call> tags.\n\n<tools>";

static const char tools_close[] =
    "</tools>\n\n# Output Format\n\nYour response MUST follow this format "
    "when making function calls:\n<tool_call>\n[\n  {\"name\": "
    "\"function_name\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": "
    "\"value2\"}},\n  {\"name\": \"another_function\", \"arguments\": "
    "{\"arg\": \"value\"}}\n]\n</tool_call>\nIf no function calls are "
    "needed, respond normally without the tool_call tags.";

static void falcon_reset(struct llm * llm) {
    struct falcon * self = (struct falcon *)llm;
    chat_start(&self->chat, &self->gguf, &format);
    free(self->tools);
    self->tools = NULL;
}

static void tooled_message(struct falcon * self, const char * role,
                           const char * content) {
    struct chat * c = &self->chat;
    const struct gguf * g = &self->gguf;
    const char * from = content + strspn(content, " \t\r\n");
    if (strcmp(role, "system") == 0) {
        chat_begin(c);
        chat_special(c, g, "<|system|>");
        chat_content(c, g, &tooled, content);
        chat_special(c, g, tools_open);
        chat_content(c, g, &tooled, self->tools);
        chat_special(c, g, tools_close);
        chat_want(c, MARK_SYSTEM);
    } else if (strcmp(role, "tool") == 0) {
        chat_begin(c);
        chat_special(c, g, "<|im_start|>assistant\n<tool_response>\n");
        chat_content(c, g, &tooled, content);
        chat_special(c, g, "\n</tool_response><|im_end|>");
    } else if (strcmp(role, "assistant") != 0) {
        chat_message(c, g, &tooled, role, content, NULL);
    } else if (strncmp(from, "<tool_call>", 11) == 0) {
        chat_begin(c);
        chat_special(c, g, from);
    } else {
        chat_message(c, g, &answered, role, content, NULL);
    }
}

static void falcon_message(struct llm * llm, const char * role,
                           const char * content) {
    struct falcon * self = (struct falcon *)llm;
    if (strcmp(role, "tools") == 0 && !self->chat.counting) {
        free(self->tools);
        self->tools = strdup(content);
    } else if (strcmp(role, "tools") != 0 && self->tools != NULL) {
        tooled_message(self, role, content);
    } else if (strcmp(role, "tools") != 0) {
        chat_message(&self->chat, &self->gguf, &format, role, content, NULL);
    }
}

static int32_t falcon_count(struct llm * llm, const char * role,
                            const char * content) {
    struct falcon * self = (struct falcon *)llm;
    self->chat.counting = true;
    self->chat.counted = 0;
    falcon_message(llm, role, content);
    self->chat.counting = false;
    return self->chat.counted;
}

static int32_t falcon_room(struct llm * llm) {
    const struct falcon * self = (const struct falcon *)llm;
    return self->chat.context - self->chat.pending_count;
}

static void falcon_rewind(struct llm * llm, int32_t messages) {
    chat_rewind(&((struct falcon *)llm)->chat, messages);
}

static bool going(const struct chat * c, const struct turn * turn,
                  int32_t count, int32_t token) {
    return count < turn->limit && c->fed_count < c->context && token >= 0 &&
           token != IM_END && token != EOS && token != TOOL_END &&
           (turn->alive == NULL || turn->alive(turn->who));
}

static int32_t reply_text(const struct falcon * self, char * text,
                          int32_t size, int32_t capacity, int32_t token,
                          const struct turn * turn) {
    const int32_t end = token == TOOL_CALL || token == TOOL_END ?
                        gguf_mark(&self->gguf, text, size, capacity, token) :
                        gguf_text(&self->gguf, text, size, capacity, token);
    text[end] = 0;
    if (turn->said != NULL) {
        turn->said(turn->who, text, end);
    }
    return end;
}

static void reply_end(struct falcon * self, const char * text) {
    if (self->tools != NULL) {
        chat_rewind(&self->chat, 1);
        tooled_message(self, "assistant", text);
    } else {
        chat_special(&self->chat, &self->gguf, format.close);
    }
}

static int32_t falcon_reply(struct llm * llm, char * text, int32_t capacity,
                            struct turn * turn) {
    struct falcon * self = (struct falcon *)llm;
    struct chat * c = &self->chat;
    struct sampler * s = malloc(sizeof *s);
    float * logits = floats(VOCAB);
    struct turn verbatim = *turn;
    int32_t size = 0;
    int32_t count = 0;
    int32_t token = 0;
    must(s != NULL, "memory");
    verbatim.temperature = 0.0f;
    verbatim.dry = 0.0f;
    verbatim.presence = 0.0f;
    const struct format * f = self->tools != NULL ? &tooled : &format;
    const bool fits = chat_fit(c, turn, chat_head(c, &self->gguf, f, false));
    chat_reply(c, &self->gguf, f, false);
    c->alive = turn->alive;
    c->who = turn->who;
    if (fits) {
        chat_feed(c);
    }
    const bool fed = fits && c->fed_count == c->pending_count;
    turn->kept = c->kept;
    turn->fed = c->fed_count - c->kept;
    token = fed && sampler_open(s, &self->gguf, c, turn) ? 0 : -1;
    while (going(c, turn, count, token)) {
        gguf_linear(&self->gguf, logits, self->last, &self->output, VOCAB,
                    DIM, 1);
        token = sampler_next(s, logits);
        if (token == TOOL_CALL) {
            s->turn = &verbatim;
            sampler_tools(s, self->tools);
        }
        if (token >= 0 && token != IM_END && token != EOS) {
            size = reply_text(self, text, size, capacity, token, turn);
            chat_took(c, token);
        }
        count++;
    }
    text[size] = 0;
    if (fed) {
        reply_end(self, text);
        c->mode = self->tools != NULL ? 1 : 0;
        chat_commit(c);
    } else {
        chat_rewind(c, 1);
    }
    free(logits);
    free(s);
    return count;
}

static void tools_resume(struct falcon * self) {
    if (self->chat.mode != 0 && self->tools == NULL) {
        self->tools = strdup("");
    }
}

static bool falcon_save(struct llm * llm, const char * path) {
    return chat_save(&((struct falcon *)llm)->chat, path);
}

static bool falcon_take(struct llm * llm, const char * path) {
    struct falcon * self = (struct falcon *)llm;
    const bool taken = chat_load(&self->chat, path);
    if (taken) {
        tools_resume(self);
    }
    return taken;
}

static void falcon_close(struct llm * llm) {
    struct falcon * self = (struct falcon *)llm;
    for (int32_t l = 0; l < LAYERS; l++) {
        free(self->layers[l].keys);
        free(self->layers[l].values);
        free(self->layers[l].state);
        free(self->layers[l].conv);
    }
    gguf_close(&self->gguf);
    chat_close(&self->chat);
    free(self->tools);
    free(self);
}

static void falcon_load(struct falcon * self) {
    struct gguf * g = &self->gguf;
    uint32_t type = 0;
    uint64_t count = 0;
    g->faults += g->vocab_size == VOCAB ? 0 : 1;
    for (int32_t l = 0; l < LAYERS; l++) {
        layer_open(self, l);
    }
    self->embed = gguf_info(g, "token_embd.weight", &type, &count);
    self->embed_type = type;
    g->faults += (type == GGUF_Q8_0 || type == GGUF_Q4_0) &&
                 count == (uint64_t)VOCAB * DIM ? 0 : 1;
    self->output_norm = gguf_tensor(g, "output_norm.weight", 0);
    self->output = gguf_matrix(g, "output.weight", 0);
    for (int32_t i = 0; i < HEAD / 2; i++) {
        self->rope_rate[i] = powf(99999997952.0f,
                                  -(float)(2 * i) / (float)HEAD);
    }
}

static struct falcon * falcon_new(void) {
    struct falcon * self = calloc(1, sizeof *self);
    if (self != NULL) {
        self->llm = (struct llm){
            .name = "falcon", .reset = falcon_reset,
            .message = falcon_message, .count = falcon_count,
            .room = falcon_room, .reply = falcon_reply,
            .rewind = falcon_rewind, .save = falcon_save,
            .load = falcon_take, .close = falcon_close
        };
        self->chat.engine = self;
        self->chat.forward = state_forward;
        self->chat.shift = state_shift;
        self->chat.save = state_save;
        self->chat.load = state_load;
    }
    return self;
}

static struct llm * falcon_open(const char * path, struct backend * backend,
                                const char * const * options) {
    const char * weights = option(options, "weights", "packed");
    const char * wanted = option(options, "context", NULL);
    const int32_t context = wanted != NULL ? atoi(wanted) : CONTEXT;
    const bool packed = strcmp(weights, "packed") == 0;
    const bool valid = (packed || strcmp(weights, "float") == 0) &&
                       context >= 256 && context <= 262144;
    struct store store = {
        .state_size = ((size_t)LAYERS * (INNER * STATE + XBC * (CONV - 1)) +
                       DIM) * sizeof(float),
        .tracks = LAYERS * 2, .width = GROUPS * HEAD,
        .path = option(options, "state", NULL)
    };
    struct falcon * self = valid ? falcon_new() : NULL;
    bool mapped = self != NULL && gguf_open(&self->gguf, path, backend,
                                            packed);
    if (mapped) {
        store.model = self->gguf.size;
        mapped = chat_open(&self->chat, context, &store);
    }
    if (mapped) {
        falcon_load(self);
        chat_resume(&self->chat);
        tools_resume(self);
    }
    const bool ready = mapped && self->gguf.faults == 0;
    if (!ready && self != NULL) {
        falcon_close(&self->llm);
    }
    return ready ? &self->llm : NULL;
}

INSTALL(llm, falcon,
        .pack = "leok7v/Falcon-H1-Tiny-Tool-Calling-90M-GGUF/"
                "Falcon-H1-Tiny-90M-Tool-Calling-Q8_0.gguf",
        .keys = "weights context state", .open = falcon_open)

#ifndef NO_MAIN

int main(int argc, char ** argv) {
    (void)argc;
    return llm_run("falcon", (const char * const *)argv + 1);
}

#endif
