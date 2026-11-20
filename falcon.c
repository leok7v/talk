#include "transformer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { DIM = 512, LAYERS = 24, HEADS = 8, GROUPS = 2, HEAD = 64, FFN = 768,
       INNER = 768, STATE = 64, SCANS = 24, SCAN = 32, CONV = 4, XBC = 896,
       PROJ = 1688, VOCAB = 32768, CONTEXT = 16384, EOS = 11, IM_END = 228 };

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
    const float * output_norm;
    struct matrix output;
    float rope_rate[HEAD / 2];
    struct chat chat;
    int32_t position;
    int32_t row;
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
        .keys = m->keys, .values = m->values, .heads = HEADS,
        .groups = GROUPS, .head = HEAD
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

static void convolve(float * xbc, const struct layer * m) {
    for (int32_t c = 0; c < XBC; c++) {
        float * past = m->conv + c * (CONV - 1);
        const float * w = m->conv_w + c * CONV;
        const float now = xbc[c];
        xbc[c] = silu(m->conv_b[c] + past[0] * w[0] + past[1] * w[1] +
                      past[2] * w[2] + now * w[3]);
        past[0] = past[1];
        past[1] = past[2];
        past[2] = now;
    }
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
        convolve(p + INNER, m);
        scan(to, p + INNER, p + INNER + XBC, m);
        for (int32_t i = 0; i < INNER; i++) {
            to[i] *= silu(p[i]);
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

static void forward(struct falcon * self, const int32_t * tokens,
                    int32_t rows) {
    const size_t size = (size_t)rows * DIM;
    float * x = floats(4 * size);
    float * n = x + size;
    float * a = n + size;
    float * s = a + size;
    for (int32_t r = 0; r < rows; r++) {
        gguf_decode(x + r * DIM, self->embed + (size_t)tokens[r] *
                    (DIM / GGUF_BLOCK) * (GGUF_BLOCK + 2), DIM);
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

static void falcon_reset(struct llm * llm) {
    struct falcon * self = (struct falcon *)llm;
    chat_start(&self->chat, &self->gguf, &format);
}

static void falcon_message(struct llm * llm, const char * role,
                           const char * content) {
    struct falcon * self = (struct falcon *)llm;
    if (strcmp(role, "tools") != 0) {
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

static int32_t falcon_reply(struct llm * llm, char * text, int32_t capacity,
                            struct turn * turn) {
    struct falcon * self = (struct falcon *)llm;
    struct chat * c = &self->chat;
    struct sampler * s = malloc(sizeof *s);
    float * logits = floats(VOCAB);
    int32_t size = 0;
    int32_t count = 0;
    int32_t token = 0;
    must(s != NULL, "memory");
    const bool fits = chat_fit(c, turn);
    chat_reply(c, &self->gguf, &format, false);
    if (fits) {
        chat_feed(c);
    }
    turn->kept = c->kept;
    turn->fed = c->fed_count - c->kept;
    token = fits && sampler_open(s, &self->gguf, c, turn) ? 0 : -1;
    while (count < turn->limit && c->fed_count < c->context && token >= 0 &&
           token != IM_END && token != EOS &&
           (turn->alive == NULL || turn->alive(turn->who))) {
        gguf_linear(&self->gguf, logits, self->last, &self->output, VOCAB,
                    DIM, 1);
        token = sampler_next(s, logits);
        if (token >= 0 && token != IM_END && token != EOS) {
            size = gguf_text(&self->gguf, text, size, capacity, token);
            chat_took(c, token);
        }
        count++;
    }
    chat_special(c, &self->gguf, format.close);
    text[size] = 0;
    free(logits);
    free(s);
    return count;
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
    g->faults += type == GGUF_Q8_0 && count == (uint64_t)VOCAB * DIM ? 0 : 1;
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
            .rewind = falcon_rewind, .close = falcon_close
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
    const char * weights = option(options, "weights", "float");
    const char * wanted = option(options, "context", NULL);
    const int32_t context = wanted != NULL ? atoi(wanted) : CONTEXT;
    const bool packed = strcmp(weights, "packed") == 0;
    const bool valid = (packed || strcmp(weights, "float") == 0) &&
                       context >= 256 && context <= 262144;
    const size_t state = ((size_t)LAYERS * (INNER * STATE + XBC * (CONV - 1)) +
                          DIM) * sizeof(float);
    struct falcon * self = valid ? falcon_new() : NULL;
    const bool mapped = self != NULL &&
                        chat_open(&self->chat, context, state) &&
                        gguf_open(&self->gguf, path, backend, packed);
    if (mapped) {
        falcon_load(self);
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
        .keys = "weights context", .open = falcon_open)

#ifndef NO_MAIN

int main(int argc, char ** argv) {
    (void)argc;
    return llm_program("falcon", (const char * const *)argv + 1);
}

#endif
