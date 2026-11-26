#include "transformer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { DIM = 1024, LAYERS = 24, HEADS = 8, GROUPS = 2, HEAD = 256, ROT = 64,
       FFN = 3584, VOCAB = 248320, STATE = 128, SCANS = 16, KEY = 2048,
       VALUE = 2048, CONV = 6144, KERNEL = 4, INTERVAL = 4, CONTEXT = 16384,
       END_OF_TEXT = 248044, IM_END = 248046, THINK_END = 248069,
       SCRIPT = 4096 };

static const float eps = 1e-6f;

static const char * const think_script =
    "\"They ask\" S \" I think\" S{1,%d} \"\\n</think>\\n\\n\" %s";

static const char tools_close[] =
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the "
    "following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\n"
    "value_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the "
    "value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner "
    "<function=...></function> block must be nested within "
    "<tool_call></tool_call> XML tags\n- Required parameters MUST be "
    "specified\n- You may provide optional reasoning for your function "
    "call in natural language BEFORE the function call, but NOT after\n"
    "- If there is no function call available, answer the question like "
    "normal with your current knowledge and do not tell the user about "
    "function calls\n</IMPORTANT>";

struct layer {
    const float * attn_norm;
    const float * post_norm;
    struct matrix up;
    struct matrix gate;
    struct matrix down;
    struct matrix qkv;
    struct matrix z;
    struct matrix beta;
    struct matrix alpha;
    struct matrix out;
    const float * conv_w;
    const float * dt;
    const float * a;
    const float * scan_norm;
    struct matrix q;
    struct matrix k;
    struct matrix v;
    const float * q_norm;
    const float * k_norm;
    float * keys;
    float * values;
    float * state;
    float * conv;
};

struct qwen {
    struct llm llm;
    struct gguf gguf;
    struct layer layers[LAYERS];
    const uint8_t * embed;
    const float * output_norm;
    struct matrix output;
    float rope_rate[ROT / 2];
    struct chat chat;
    int32_t position;
    int32_t row;
    char * tools;
    float last[DIM];
};

static bool recurrent(int32_t l) {
    return (l + 1) % INTERVAL != 0;
}

static void layer_open(struct qwen * self, int32_t l) {
    struct gguf * g = &self->gguf;
    struct layer * m = &self->layers[l];
    const size_t cache = (size_t)self->chat.context * GROUPS * HEAD;
    m->attn_norm = gguf_tensor(g, "blk.%d.attn_norm.weight", l);
    m->post_norm = gguf_tensor(g, "blk.%d.post_attention_norm.weight", l);
    m->up = gguf_matrix(g, "blk.%d.ffn_up.weight", l);
    m->gate = gguf_matrix(g, "blk.%d.ffn_gate.weight", l);
    m->down = gguf_matrix(g, "blk.%d.ffn_down.weight", l);
    if (recurrent(l)) {
        m->qkv = gguf_matrix(g, "blk.%d.attn_qkv.weight", l);
        m->z = gguf_matrix(g, "blk.%d.attn_gate.weight", l);
        m->beta = gguf_matrix(g, "blk.%d.ssm_beta.weight", l);
        m->alpha = gguf_matrix(g, "blk.%d.ssm_alpha.weight", l);
        m->out = gguf_matrix(g, "blk.%d.ssm_out.weight", l);
        m->conv_w = gguf_tensor(g, "blk.%d.ssm_conv1d.weight", l);
        m->dt = gguf_tensor(g, "blk.%d.ssm_dt.bias", l);
        m->a = gguf_tensor(g, "blk.%d.ssm_a", l);
        m->scan_norm = gguf_tensor(g, "blk.%d.ssm_norm.weight", l);
        m->state = floats((size_t)SCANS * STATE * STATE);
        m->conv = floats((size_t)CONV * (KERNEL - 1));
    } else {
        m->q = gguf_matrix(g, "blk.%d.attn_q.weight", l);
        m->k = gguf_matrix(g, "blk.%d.attn_k.weight", l);
        m->v = gguf_matrix(g, "blk.%d.attn_v.weight", l);
        m->out = gguf_matrix(g, "blk.%d.attn_output.weight", l);
        m->q_norm = gguf_tensor(g, "blk.%d.attn_q_norm.weight", l);
        m->k_norm = gguf_tensor(g, "blk.%d.attn_k_norm.weight", l);
        m->keys = floats(cache);
        m->values = floats(cache);
    }
}

static float silu(float x) {
    return x / (1.0f + expf(-x));
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static float softplus(float x) {
    return fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
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
                 int32_t rows, int32_t n) {
    for (int32_t r = 0; r < rows; r++) {
        const float * in = x + r * n;
        const float gain = 1.0f / sqrtf(dot(in, in, n) / (float)n + eps);
        for (int32_t i = 0; i < n; i++) {
            y[r * n + i] = in[i] * gain * weight[i];
        }
    }
}

static void unit(float * x, int32_t rows, int32_t n) {
    for (int32_t r = 0; r < rows; r++) {
        float * row = x + r * n;
        const float gain = 1.0f / sqrtf(dot(row, row, n) + eps);
        for (int32_t i = 0; i < n; i++) {
            row[i] *= gain;
        }
    }
}

static void convolve(float * y, const float * x, const struct layer * m) {
    for (int32_t c = 0; c < CONV; c++) {
        const float * w = m->conv_w + c * KERNEL;
        float sum = x[c] * w[KERNEL - 1];
        for (int32_t j = 0; j < KERNEL - 1; j++) {
            sum += m->conv[j * CONV + c] * w[j];
        }
        y[c] = silu(sum);
    }
    memmove(m->conv, m->conv + CONV,
            (size_t)CONV * (KERNEL - 2) * sizeof(float));
    memcpy(m->conv + (KERNEL - 2) * CONV, x, CONV * sizeof(float));
}

static void delta(float * o, const float * q, const float * k,
                  const float * v, float * s, float decay, float beta) {
    float sk[STATE] = { 0 };
    float d[STATE];
    for (int32_t i = 0; i < STATE; i++) {
        float * row = s + i * STATE;
        for (int32_t j = 0; j < STATE; j++) {
            row[j] *= decay;
            sk[j] += row[j] * k[i];
        }
    }
    for (int32_t j = 0; j < STATE; j++) {
        d[j] = beta * (v[j] - sk[j]);
        o[j] = 0.0f;
    }
    for (int32_t i = 0; i < STATE; i++) {
        float * row = s + i * STATE;
        const float qi = q[i] / sqrtf((float)STATE);
        for (int32_t j = 0; j < STATE; j++) {
            row[j] += k[i] * d[j];
            o[j] += row[j] * qi;
        }
    }
}

static void scan_row(float * o, float * conv, const float * mixed,
                     const float * alpha, const float * beta,
                     const struct layer * m) {
    convolve(conv, mixed, m);
    unit(conv, SCANS, STATE);
    unit(conv + KEY, SCANS, STATE);
    for (int32_t h = 0; h < SCANS; h++) {
        delta(o + h * STATE, conv + h * STATE, conv + KEY + h * STATE,
              conv + 2 * KEY + h * STATE, m->state + h * STATE * STATE,
              expf(softplus(alpha[h] + m->dt[h]) * m->a[h]),
              sigmoid(beta[h]));
    }
    norm(o, o, m->scan_norm, SCANS, STATE);
}

static void scan(struct qwen * self, float * y, const float * x,
                 const struct layer * m, int32_t rows) {
    struct gguf * g = &self->gguf;
    float * mixed = floats((size_t)rows * CONV);
    float * conv = floats(CONV);
    float * z = floats((size_t)rows * VALUE);
    float * o = floats((size_t)rows * VALUE);
    float * beta = floats((size_t)rows * SCANS);
    float * alpha = floats((size_t)rows * SCANS);
    gguf_linear(g, mixed, x, &m->qkv, CONV, DIM, rows);
    gguf_linear(g, z, x, &m->z, VALUE, DIM, rows);
    gguf_linear(g, beta, x, &m->beta, SCANS, DIM, rows);
    gguf_linear(g, alpha, x, &m->alpha, SCANS, DIM, rows);
    for (int32_t r = 0; r < rows; r++) {
        scan_row(o + r * VALUE, conv, mixed + r * CONV, alpha + r * SCANS,
                 beta + r * SCANS, m);
    }
    for (int32_t i = 0; i < rows * VALUE; i++) {
        o[i] *= silu(z[i]);
    }
    gguf_linear(g, y, o, &m->out, DIM, VALUE, rows);
    free(mixed);
    free(conv);
    free(z);
    free(o);
    free(beta);
    free(alpha);
}

static void rope(const struct qwen * self, float * x, int32_t heads,
                 int32_t position) {
    for (int32_t i = 0; i < ROT / 2; i++) {
        const float c = cosf((float)position * self->rope_rate[i]);
        const float s = sinf((float)position * self->rope_rate[i]);
        for (int32_t h = 0; h < heads; h++) {
            float * p = x + h * HEAD;
            const float a = p[i];
            const float b = p[i + ROT / 2];
            p[i] = a * c - b * s;
            p[i + ROT / 2] = a * s + b * c;
        }
    }
}

static void attention(struct qwen * self, float * y, const float * x,
                      const struct layer * m, int32_t rows) {
    struct gguf * g = &self->gguf;
    const int32_t wide = HEADS * HEAD;
    float * both = floats((size_t)rows * 2 * wide);
    float * q = floats((size_t)rows * wide);
    float * mix = floats((size_t)rows * wide);
    float * k = m->keys + self->row * GROUPS * HEAD;
    const struct attention a = {
        .backend = g->backend, .keys = m->keys, .values = m->values,
        .heads = HEADS, .groups = GROUPS, .head = HEAD
    };
    gguf_linear(g, both, x, &m->q, 2 * wide, DIM, rows);
    gguf_linear(g, k, x, &m->k, GROUPS * HEAD, DIM, rows);
    gguf_linear(g, m->values + self->row * GROUPS * HEAD, x, &m->v,
                GROUPS * HEAD, DIM, rows);
    for (int32_t j = 0; j < rows * HEADS; j++) {
        memcpy(q + j * HEAD, both + j * 2 * HEAD, HEAD * sizeof(float));
    }
    norm(q, q, m->q_norm, rows * HEADS, HEAD);
    norm(k, k, m->k_norm, rows * GROUPS, HEAD);
    for (int32_t r = 0; r < rows; r++) {
        rope(self, q + r * wide, HEADS, self->position + r);
        rope(self, k + r * GROUPS * HEAD, GROUPS, self->position + r);
    }
    chat_attend(&a, mix, q, self->row, rows);
    for (int32_t j = 0; j < rows * HEADS; j++) {
        for (int32_t i = 0; i < HEAD; i++) {
            mix[j * HEAD + i] *= sigmoid(both[j * 2 * HEAD + HEAD + i]);
        }
    }
    gguf_linear(g, y, mix, &m->out, DIM, wide, rows);
    free(both);
    free(q);
    free(mix);
}

static void feed(struct qwen * self, float * x, const struct layer * m,
                 int32_t rows) {
    float * n = floats((size_t)rows * DIM);
    float * up = floats((size_t)rows * FFN);
    float * gate = floats((size_t)rows * FFN);
    float * y = floats((size_t)rows * DIM);
    norm(n, x, m->post_norm, rows, DIM);
    gguf_linear(&self->gguf, up, n, &m->up, FFN, DIM, rows);
    gguf_linear(&self->gguf, gate, n, &m->gate, FFN, DIM, rows);
    for (int32_t i = 0; i < rows * FFN; i++) {
        up[i] *= silu(gate[i]);
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

static void forward(struct qwen * self, const int32_t * tokens,
                    int32_t rows) {
    const size_t size = (size_t)rows * DIM;
    float * x = floats(3 * size);
    float * n = x + size;
    float * y = n + size;
    for (int32_t r = 0; r < rows; r++) {
        gguf_row(x + r * DIM, self->embed, tokens[r], DIM, GGUF_Q8_0);
    }
    for (int32_t l = 0; l < LAYERS; l++) {
        const struct layer * m = &self->layers[l];
        norm(n, x, m->attn_norm, rows, DIM);
        if (recurrent(l)) {
            scan(self, y, n, m, rows);
        } else {
            attention(self, y, n, m, rows);
        }
        for (size_t i = 0; i < size; i++) {
            x[i] += y[i];
        }
        feed(self, x, m, rows);
    }
    norm(self->last, x + size - DIM, self->output_norm, 1, DIM);
    free(x);
}

static void state_forward(void * engine, const int32_t * tokens,
                          int32_t count, int32_t position, int32_t row) {
    struct qwen * self = engine;
    self->position = position;
    self->row = row;
    forward(self, tokens, count);
}

static void state_shift(void * engine, int32_t keep, int32_t count,
                        int32_t rows) {
    struct qwen * self = engine;
    const size_t row = (size_t)GROUPS * HEAD * sizeof(float);
    for (int32_t l = 0; l < LAYERS; l++) {
        float * k = self->layers[l].keys;
        float * v = self->layers[l].values;
        if (k != NULL) {
            memmove(k + keep * GROUPS * HEAD,
                    k + (keep + count) * GROUPS * HEAD,
                    (size_t)(rows - keep - count) * row);
            memmove(v + keep * GROUPS * HEAD,
                    v + (keep + count) * GROUPS * HEAD,
                    (size_t)(rows - keep - count) * row);
        }
    }
}

static size_t state_bytes(void) {
    return ((size_t)SCANS * STATE * STATE + CONV * (KERNEL - 1)) *
           sizeof(float);
}

static void state_save(void * engine, uint8_t * to) {
    const struct qwen * self = engine;
    const size_t state = (size_t)SCANS * STATE * STATE * sizeof(float);
    uint8_t * at = to;
    for (int32_t l = 0; l < LAYERS; l++) {
        if (recurrent(l)) {
            memcpy(at, self->layers[l].state, state);
            memcpy(at + state, self->layers[l].conv, state_bytes() - state);
            at += state_bytes();
        }
    }
    memcpy(at, self->last, sizeof self->last);
}

static void state_load(void * engine, const uint8_t * from) {
    struct qwen * self = engine;
    const size_t state = (size_t)SCANS * STATE * STATE * sizeof(float);
    const uint8_t * at = from;
    for (int32_t l = 0; l < LAYERS; l++) {
        if (recurrent(l) && from != NULL) {
            memcpy(self->layers[l].state, at, state);
            memcpy(self->layers[l].conv, at + state, state_bytes() - state);
            at += state_bytes();
        } else if (recurrent(l)) {
            memset(self->layers[l].state, 0, state);
            memset(self->layers[l].conv, 0, state_bytes() - state);
        }
    }
    if (from != NULL) {
        memcpy(self->last, at, sizeof self->last);
    }
}

static const struct format format = {
    .start = "", .open = "<|im_start|>", .middle = "\n",
    .close = "<|im_end|>\n", .reply = "<|im_start|>assistant\n",
    .plain = "<think>\n\n</think>\n\n", .think = "<think>\n",
    .tools_open = "# Tools\n\nYou have access to the following "
                  "functions:\n\n<tools>",
    .tools_close = tools_close, .trim = true
};

static void qwen_reset(struct llm * llm) {
    struct qwen * self = (struct qwen *)llm;
    chat_start(&self->chat, &self->gguf, &format);
    free(self->tools);
    self->tools = NULL;
}

static void qwen_message(struct llm * llm, const char * role,
                         const char * content) {
    struct qwen * self = (struct qwen *)llm;
    if (strcmp(role, "tools") != 0) {
        chat_message(&self->chat, &self->gguf, &format, role, content,
                     self->tools);
    } else if (!self->chat.counting) {
        free(self->tools);
        self->tools = strdup(content);
    }
}

static int32_t qwen_count(struct llm * llm, const char * role,
                          const char * content) {
    struct qwen * self = (struct qwen *)llm;
    self->chat.counting = true;
    self->chat.counted = 0;
    qwen_message(llm, role, content);
    self->chat.counting = false;
    return self->chat.counted;
}

static int32_t qwen_room(struct llm * llm) {
    const struct qwen * self = (const struct qwen *)llm;
    return self->chat.context - self->chat.pending_count;
}

static void qwen_rewind(struct llm * llm, int32_t messages) {
    chat_rewind(&((struct qwen *)llm)->chat, messages);
}

static bool going(const struct chat * c, const struct turn * turn,
                  int32_t count, int32_t token) {
    return count < turn->limit && c->fed_count < c->context && token >= 0 &&
           token != IM_END && token != END_OF_TEXT &&
           (turn->alive == NULL || turn->alive(turn->who));
}

static int32_t generate(struct qwen * self, struct turn * turn, char * text,
                        int32_t capacity) {
    struct chat * c = &self->chat;
    struct sampler * s = malloc(sizeof *s);
    float * logits = floats(VOCAB);
    bool thought = turn->think > 0;
    int32_t size = 0;
    int32_t noted = 0;
    int32_t count = 0;
    must(s != NULL, "memory");
    int32_t token = sampler_open(s, &self->gguf, c, turn) ? 0 : -1;
    while (going(c, turn, count, token)) {
        gguf_linear(&self->gguf, logits, self->last, &self->output, VOCAB,
                    DIM, 1);
        token = sampler_next(s, logits);
        if (going(c, turn, count, token) && !thought) {
            size = gguf_text(&self->gguf, text, size, capacity, token);
            text[size] = 0;
            if (turn->said != NULL) {
                turn->said(turn->who, text, size);
            }
        } else if (going(c, turn, count, token) && turn->thought != NULL &&
                   token != THINK_END) {
            noted = gguf_text(&self->gguf, turn->thought, noted,
                              turn->thought_capacity, token);
            turn->thought[noted] = 0;
        }
        if (going(c, turn, count, token)) {
            thought = thought && token != THINK_END;
            chat_took(c, token);
        }
        count++;
    }
    text[size] = 0;
    free(logits);
    free(s);
    return count;
}

static int32_t qwen_reply(struct llm * llm, char * text, int32_t capacity,
                          struct turn * turn) {
    struct qwen * self = (struct qwen *)llm;
    struct chat * c = &self->chat;
    struct turn scripted = *turn;
    char script[SCRIPT];
    const bool fits = chat_fit(c, turn, chat_head(c, &self->gguf, &format,
                                                  turn->think > 0));
    const int32_t reply_at = chat_reply(c, &self->gguf, &format,
                                        turn->think > 0);
    const char * given = turn->script != NULL ? turn->script : "*";
    if (turn->think > 0 && strstr(given, "</think>") == NULL) {
        snprintf(script, sizeof script, think_script, turn->think, given);
        scripted.script = script;
    }
    text[0] = 0;
    if (fits) {
        chat_feed(c);
    }
    turn->kept = c->kept;
    turn->fed = c->fed_count - c->kept;
    const int32_t count = fits ? generate(self, &scripted, text, capacity) :
                          0;
    const size_t lead = strspn(text, " \t\r\n");
    memmove(text, text + lead, strlen(text + lead) + 1);
    c->pending_count = reply_at;
    chat_content(c, &self->gguf, &format, text);
    chat_special(c, &self->gguf, format.close);
    return count;
}

static void qwen_close(struct llm * llm) {
    struct qwen * self = (struct qwen *)llm;
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

static void qwen_load(struct qwen * self) {
    struct gguf * g = &self->gguf;
    const bool packed = g->packed;
    uint32_t type = 0;
    uint64_t count = 0;
    g->digits = 1;
    g->faults += g->vocab_size == VOCAB ? 0 : 1;
    for (int32_t l = 0; l < LAYERS && g->faults == 0; l++) {
        layer_open(self, l);
    }
    self->embed = gguf_info(g, "token_embd.weight", &type, &count);
    g->faults += type == GGUF_Q8_0 && count == (uint64_t)VOCAB * DIM ? 0 : 1;
    self->output_norm = gguf_tensor(g, "output_norm.weight", 0);
    g->packed = true;
    self->output = gguf_matrix(g, "token_embd.weight", 0);
    g->packed = packed;
    for (int32_t i = 0; i < ROT / 2; i++) {
        self->rope_rate[i] = powf(10000000.0f, -(float)(2 * i) / (float)ROT);
    }
}

static struct qwen * qwen_new(void) {
    struct qwen * self = calloc(1, sizeof *self);
    if (self != NULL) {
        self->llm = (struct llm){
            .name = "qwen", .reset = qwen_reset, .message = qwen_message,
            .count = qwen_count, .room = qwen_room, .reply = qwen_reply,
            .rewind = qwen_rewind, .close = qwen_close
        };
        self->chat.engine = self;
        self->chat.forward = state_forward;
        self->chat.shift = state_shift;
        self->chat.save = state_save;
        self->chat.load = state_load;
    }
    return self;
}

static struct llm * qwen_open(const char * path, struct backend * backend,
                              const char * const * options) {
    const char * weights = option(options, "weights", "float");
    const char * wanted = option(options, "context", NULL);
    const int32_t context = wanted != NULL ? atoi(wanted) : CONTEXT;
    const bool packed = strcmp(weights, "packed") == 0;
    const bool valid = (packed || strcmp(weights, "float") == 0) &&
                       context >= 256 && context <= 262144;
    const struct store store = {
        .state_size = (LAYERS - LAYERS / INTERVAL) * state_bytes() +
                      DIM * sizeof(float)
    };
    struct qwen * self = valid ? qwen_new() : NULL;
    const bool mapped = self != NULL &&
                        chat_open(&self->chat, context, &store) &&
                        gguf_open(&self->gguf, path, backend, packed);
    if (mapped) {
        qwen_load(self);
    }
    const bool ready = mapped && self->gguf.faults == 0;
    if (!ready && self != NULL) {
        qwen_close(&self->llm);
    }
    return ready ? &self->llm : NULL;
}

INSTALL(llm, qwen,
        .pack = "leok7v/Qwen3.5-0.8B-Astra-GGUF/"
                "Qwen3.5-0.8B-Astra-Q8_0.gguf",
        .keys = "weights context", .open = qwen_open)

#ifndef NO_MAIN

int main(int argc, char ** argv) {
    (void)argc;
    return llm_run("qwen", (const char * const *)argv + 1);
}

#endif
