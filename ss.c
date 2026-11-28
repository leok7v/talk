#include "ss.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { DIM = 128, HEADS = 4, HD = 32, HIDDEN = 384, WIDE = 512, BINS = 65,
       WINDOW = 128, HOP = 64, CORR = 18, EMBED = 512, QUERIES = 7,
       BLOCKS = 6, ENCODERS = 2, DECODERS = 4, OUTS = 27, TAPS = 31,
       NAME = 192, OWNED = 64, CHUNK = 4096, SLICE = 8192 };

static const float rms_epsilon = 1.1920929e-07f;
static const float ln_epsilon = 1e-5f;

struct weight {
    const float * data;
    int32_t format;
    int32_t ldb;
};

struct conv {
    const float * taps;
    const float * bias;
    int32_t out;
    int32_t in;
    int32_t span;
    int32_t format;
    int32_t ldb;
    size_t step;
};

struct convffn {
    const float * norm;
    struct conv up;
    struct conv down;
};

struct attn {
    const float * norm;
    struct weight qkv;
    struct weight out;
    const float * freqs;
};

struct trans {
    struct convffn ffn1;
    struct attn sa;
    struct convffn ffn2;
};

struct ffn {
    const float * norm;
    struct weight up;
    const float * up_bias;
    struct weight down;
    const float * down_bias;
};

struct cross {
    const float * norm;
    struct weight q;
    struct weight kv;
    struct weight out;
};

struct attractor {
    struct cross ca;
    struct attn sa;
    struct ffn ffn;
};

struct separator {
    struct ss ss;
    struct backend * backend;
    const uint8_t * file;
    size_t size;
    const char * header;
    size_t header_size;
    float * owned[OWNED];
    int32_t owned_count;
    int32_t faults;
    struct conv embed1;
    struct conv embed2;
    const float * embed_norm;
    struct trans blocks[BLOCKS][2];
    struct attn cs_sa[DECODERS];
    struct ffn cs_ffn[DECODERS];
    const float * query;
    const float * split_norm;
    const float * split_bias;
    struct attractor dec[2];
    const float * pres_w;
    const float * pres_b;
    const float * fuse_w1;
    const float * fuse_b1;
    struct weight fuse_w2;
    const float * fuse_b2;
    struct weight mask_w;
    const float * mask_b;
    const float * last_norm;
    const float * last_bias;
    float * basis;
    float lowpass[TAPS];
};

struct places {
    float * rope[BLOCKS][2];
    float * bins;
    float * times;
    float * freqs;
};

static const char * const dtypes[] = { "F32", "F16", "Q8_0", "Q4_0",
                                       "Q4_K" };

static const float * find(struct separator * self, const char * prefix,
                          const char * suffix, int32_t format,
                          size_t bytes) {
    char name[NAME];
    char kind[NAME];
    char key[NAME + 8];
    char * next = NULL;
    snprintf(name, sizeof name, "%s%s%s", prefix, prefix[0] != 0 ? "." : "",
             suffix);
    const int size = snprintf(key, sizeof key, "\"%s\":{", name);
    const char * at = memmem(self->header, self->header_size, key,
                             (size_t)size);
    const char * end = at != NULL ?
        memchr(at + size, '}', self->header_size -
               (size_t)(at + size - self->header)) : NULL;
    const char * span = end != NULL ?
        memmem(at, (size_t)(end - at), "\"data_offsets\":[", 16) : NULL;
    const long long from = span != NULL ? strtoll(span + 16, &next, 10) : 0;
    const long long to = span != NULL ? strtoll(next + 1, NULL, 10) : 0;
    const float * data = (const float *)(const void *)
        (self->file + 8 + self->header_size + from);
    const int kind_size = snprintf(kind, sizeof kind, "\"dtype\":\"%s\"",
                                   dtypes[format]);
    const bool fine = span != NULL && (size_t)(to - from) >= bytes &&
                      (size_t)(to - from) < bytes + 16 &&
                      8 + self->header_size + (size_t)to <= self->size &&
                      memmem(at, (size_t)(end - at), kind,
                             (size_t)kind_size) != NULL;
    return fine ? data : NULL;
}

static const float * tensor(struct separator * self, const char * prefix,
                            const char * suffix, size_t count) {
    const float * data = find(self, prefix, suffix, W_F32,
                              count * sizeof(float));
    self->faults += data != NULL ? 0 : 1;
    return data;
}

static struct weight weight_load(struct separator * self,
                                 const char * prefix, const char * suffix,
                                 int32_t rows, int32_t in) {
    struct weight w = { 0 };
    for (int32_t f = W_Q4_K; f >= W_F32; f--) {
        const size_t row = weight_bytes(f, in);
        const float * data = w.data == NULL && row > 0 &&
                             (f == W_F32 || f == W_F16 || in % 32 == 0) ?
                             find(self, prefix, suffix, f, row * rows) : NULL;
        w = data != NULL ? (struct weight){
            .data = data, .format = f,
            .ldb = f == W_F32 ? in : (int32_t)row
        } : w;
    }
    self->faults += w.data != NULL ? 0 : 1;
    return w;
}

static float * take(struct separator * self, size_t count) {
    return self->backend->alloc(self->backend, count);
}

static void drop(struct separator * self, float * p) {
    self->backend->release(self->backend, p);
}

static void sync(struct separator * self) {
    self->backend->sync(self->backend);
}

static float * own(struct separator * self, size_t count) {
    float * p = take(self, count);
    must(self->owned_count < OWNED, "ss owned");
    self->owned[self->owned_count++] = p;
    return p;
}

static struct conv conv_load(struct separator * self, const char * prefix,
                             const char * stem, int32_t out, int32_t in,
                             int32_t span) {
    char name[NAME];
    snprintf(name, sizeof name, "%s.weight", stem);
    const float * w = find(self, prefix, name, W_F32,
                           (size_t)out * in * span * sizeof(float));
    const struct weight packed = w == NULL ?
        weight_load(self, prefix, name, out * span, in) : (struct weight){
            .data = w, .format = W_F32, .ldb = in };
    float * taps = packed.format == W_F32 ?
                   own(self, (size_t)out * in * span) : NULL;
    struct conv c = {
        .taps = taps != NULL ? taps : packed.data, .out = out, .in = in,
        .span = span, .format = packed.format, .ldb = packed.ldb,
        .step = (size_t)out * (packed.format == W_F32 ?
                               (size_t)in * sizeof(float) :
                               (size_t)packed.ldb)
    };
    snprintf(name, sizeof name, "%s.bias", stem);
    c.bias = tensor(self, prefix, name, (size_t)out);
    for (size_t i = 0; taps != NULL && packed.data != NULL &&
         i < (size_t)out * in * (size_t)span; i++) {
        const size_t k = i % (size_t)span;
        const size_t row = i / (size_t)span;
        taps[k * (size_t)out * in + row] = packed.data[i];
    }
    return c;
}

static const float * tap_at(const struct conv * cv, int32_t k) {
    return (const float *)(const void *)
        ((const uint8_t *)(const void *)cv->taps + (size_t)k * cv->step);
}

static struct convffn convffn_load(struct separator * self,
                                   const char * prefix) {
    return (struct convffn){
        .norm = tensor(self, prefix, "layernorm.weight", DIM),
        .up = conv_load(self, prefix, "net.0", 2 * HIDDEN, DIM, 3),
        .down = conv_load(self, prefix, "net.2", DIM, HIDDEN, 3)
    };
}

static struct attn attn_load(struct separator * self, const char * prefix,
                             bool rope) {
    return (struct attn){
        .norm = tensor(self, prefix, "layernorm.weight", DIM),
        .qkv = weight_load(self, prefix, "block.qkv.weight", 3 * DIM, DIM),
        .out = weight_load(self, prefix, "block.aggregate_heads.0.weight",
                           DIM, DIM),
        .freqs = rope ? tensor(self, prefix, "block.rope.freqs", HD / 2) :
                 NULL
    };
}

static struct ffn ffn_load(struct separator * self, const char * prefix) {
    return (struct ffn){
        .norm = tensor(self, prefix, "layernorm.weight", DIM),
        .up = weight_load(self, prefix, "net.0.weight", 2 * WIDE, DIM),
        .up_bias = tensor(self, prefix, "net.0.bias", 2 * WIDE),
        .down = weight_load(self, prefix, "net.2.weight", DIM, WIDE),
        .down_bias = tensor(self, prefix, "net.2.bias", DIM)
    };
}

static struct trans trans_load(struct separator * self, const char * prefix) {
    char name[NAME];
    struct trans t = { 0 };
    snprintf(name, sizeof name, "%s.ffn_1", prefix);
    t.ffn1 = convffn_load(self, name);
    snprintf(name, sizeof name, "%s.sa", prefix);
    t.sa = attn_load(self, name, true);
    snprintf(name, sizeof name, "%s.ffn_2", prefix);
    t.ffn2 = convffn_load(self, name);
    return t;
}

static struct attractor attractor_load(struct separator * self,
                                       const char * prefix) {
    char name[NAME];
    struct attractor a = { 0 };
    snprintf(name, sizeof name, "%s.ca", prefix);
    a.ca = (struct cross){
        .norm = tensor(self, name, "layernorm.weight", DIM),
        .q = weight_load(self, name, "block.q.weight", DIM, DIM),
        .kv = weight_load(self, name, "block.kv.weight", 2 * DIM, DIM),
        .out = weight_load(self, name, "block.aggregate_heads.0.weight", DIM,
                           DIM)
    };
    snprintf(name, sizeof name, "%s.sa", prefix);
    a.sa = attn_load(self, name, false);
    snprintf(name, sizeof name, "%s.ffn", prefix);
    a.ffn = ffn_load(self, name);
    return a;
}

static void blocks_load(struct separator * self) {
    char name[NAME];
    for (int32_t b = 0; b < BLOCKS; b++) {
        const char * side = b < ENCODERS ? "enc_block" : "dec_block";
        const int32_t i = b < ENCODERS ? b : b - ENCODERS;
        snprintf(name, sizeof name, "%s.%d.freq_block.block", side, i);
        self->blocks[b][0] = trans_load(self, name);
        snprintf(name, sizeof name, "%s.%d.time_block.block", side, i);
        self->blocks[b][1] = trans_load(self, name);
    }
    for (int32_t i = 0; i < DECODERS; i++) {
        snprintf(name, sizeof name, "dec_cs.%d.block.block.sa", i);
        self->cs_sa[i] = attn_load(self, name, false);
        snprintf(name, sizeof name, "dec_cs.%d.block.block.ffn", i);
        self->cs_ffn[i] = ffn_load(self, name);
    }
}

static void split_load(struct separator * self) {
    const char * s = "spk_split";
    self->query = tensor(self, s, "spk_query", QUERIES * DIM);
    self->split_norm = tensor(self, s, "layernorm.weight", DIM);
    self->split_bias = tensor(self, s, "layernorm.bias", DIM);
    self->dec[0] = attractor_load(self, "spk_split.dec.net1");
    self->dec[1] = attractor_load(self, "spk_split.dec.net2");
    self->pres_w = tensor(self, s, "pres_linear.weight", DIM);
    self->pres_b = tensor(self, s, "pres_linear.bias", 1);
    self->fuse_w1 = tensor(self, s, "split.fusion.0.weight", WIDE * 2 * DIM);
    self->fuse_b1 = tensor(self, s, "split.fusion.0.bias", WIDE);
    self->fuse_w2 = weight_load(self, s, "split.fusion.2.weight", DIM, WIDE);
    self->fuse_b2 = tensor(self, s, "split.fusion.2.bias", DIM);
}

static void tables_load(struct separator * self) {
    float sum = 0.0f;
    self->basis = own(self, 2 * BINS * WINDOW);
    for (int32_t c = 0; c < 2 * BINS; c++) {
        for (int32_t n = 0; n < WINDOW; n++) {
            const double step = 2.0 * M_PI * n / (double)WINDOW;
            const double w = sqrt(0.5 - 0.5 * cos(step));
            const double a = step * (c % BINS);
            self->basis[c * WINDOW + n] = (float)(c < BINS ?
                w * cos(a) / 8.0 : -w * sin(a) / 8.0);
        }
    }
    for (int32_t j = 0; j < TAPS; j++) {
        const double x = 0.5 * (j - TAPS / 2);
        const double sinc = x == 0.0 ? 1.0 : sin(M_PI * x) / (M_PI * x);
        self->lowpass[j] = (float)(0.5 * sinc *
            (0.5 - 0.5 * cos(2.0 * M_PI * j / (TAPS - 1))));
        sum += self->lowpass[j];
    }
    for (int32_t j = 0; j < TAPS; j++) {
        self->lowpass[j] /= sum;
    }
}

static void weights_load(struct separator * self) {
    self->embed1 = conv_load(self, "encoder", "embed.0", EMBED, CORR, 9);
    self->embed2 = conv_load(self, "encoder", "embed.2", DIM, EMBED, 9);
    self->embed_norm = tensor(self, "encoder", "layer_norm.weight", DIM);
    blocks_load(self);
    split_load(self);
    self->mask_w = weight_load(self, "filter_estim", "mask.net.weight", OUTS,
                               DIM);
    self->mask_b = tensor(self, "filter_estim", "mask.net.bias", OUTS);
    self->last_norm = tensor(self, "", "layernorm.weight", DIM);
    self->last_bias = tensor(self, "", "layernorm.bias", DIM);
    tables_load(self);
}

static void gemm(struct separator * self, const struct gemm_args * g) {
    self->backend->gemm(self->backend, g);
}

static void map(struct separator * self, const struct map_args * m) {
    self->backend->map(self->backend, m);
}

static void flat(struct separator * self, int32_t op, float * y,
                 const float * a, float alpha, size_t count) {
    const size_t rows = count / CHUNK;
    const size_t rest = count % CHUNK;
    const size_t at = rows * CHUNK;
    const struct map_args whole = {
        .y = y, .a = a, .op = op, .alpha = alpha, .rows = (int32_t)rows,
        .cols = CHUNK, .ys = { 0, CHUNK, 1 }, .as = { 0, CHUNK, 1 }
    };
    const struct map_args tail = {
        .y = y + at, .a = a != NULL ? a + at : NULL, .op = op,
        .alpha = alpha, .rows = 1, .cols = (int32_t)rest,
        .ys = { 0, 0, 1 }, .as = { 0, 0, 1 }
    };
    if (rows > 0) {
        map(self, &whole);
    }
    if (rest > 0) {
        map(self, &tail);
    }
}

static void zero(struct separator * self, float * y, size_t count) {
    flat(self, OP_FILL, y, NULL, 0.0f, count);
}

static void spread_rows(struct separator * self, float * y, const float * v,
                        int32_t rows, int32_t cols) {
    const struct map_args m = {
        .y = y, .a = v, .op = v != NULL ? OP_COPY : OP_FILL, .rows = rows,
        .cols = cols, .ys = { 0, cols, 1 }, .as = { 0, 0, 1 }
    };
    map(self, &m);
}

static void linear(struct separator * self, float * y, const float * x,
                   const struct weight * w, const float * bias, int32_t rows,
                   int32_t out, int32_t in) {
    const struct gemm_args g = {
        .a = x, .b = w->data, .c = y, .m = rows, .n = out, .k = in,
        .lda = in, .ldb = w->ldb, .ldc = out, .tb = true,
        .fresh = bias == NULL, .format = w->format
    };
    if (bias != NULL) {
        spread_rows(self, y, bias, rows, out);
    }
    gemm(self, &g);
}

static void rms(struct separator * self, float * y, const float * x,
                const float * w, int32_t rows) {
    const struct norm_args n = {
        .y = y, .x = x, .w = w, .rows = rows, .n = DIM, .eps = rms_epsilon
    };
    self->backend->norm(self->backend, &n);
}

static void layer(struct separator * self, float * y, const float * x,
                  const float * w, const float * b, int32_t rows) {
    const struct norm_args n = {
        .y = y, .x = x, .w = w, .b = b, .rows = rows, .n = DIM,
        .eps = ln_epsilon
    };
    self->backend->norm(self->backend, &n);
}

static void swiglu(struct separator * self, float * y, const float * x,
                   int32_t rows, int32_t half) {
    const struct map_args m = {
        .y = y, .a = x, .b = x + half, .op = OP_SWIGLU, .rows = rows,
        .cols = half, .ys = { 0, half, 1 }, .as = { 0, 2 * half, 1 },
        .bs = { 0, 2 * half, 1 }
    };
    map(self, &m);
}

static void border(struct separator * self, float * padded, int32_t rows,
                   int64_t wide, int32_t in) {
    const struct map_args sides = {
        .y = padded + (1 + wide) * in, .op = OP_FILL, .batch = rows,
        .rows = 2, .cols = in, .ys = { wide * in, (wide - 1) * in, 1 }
    };
    zero(self, padded, (size_t)((1 + wide) * in));
    map(self, &sides);
    zero(self, padded + (1 + (rows + 1) * wide) * in,
         (size_t)((wide + 1) * in));
}

static void tap(struct separator * self, float * y, const float * x,
                const struct conv * cv, int32_t k, int64_t lda, int32_t m,
                bool fresh) {
    const struct gemm_args g = {
        .a = x, .b = tap_at(cv, k), .c = y, .m = m, .n = cv->out,
        .k = cv->in, .lda = (int32_t)lda, .ldb = cv->ldb, .ldc = cv->out,
        .tb = true, .fresh = fresh, .format = cv->format
    };
    gemm(self, &g);
}

static void edge(struct separator * self, float * y, const float * x,
                 int32_t k, const struct conv * cv, int32_t rows,
                 int32_t cols) {
    float * t = take(self, (size_t)(rows - 1) * cv->out);
    const struct map_args fix = {
        .y = y, .a = t, .op = OP_ADD, .alpha = -1.0f, .rows = rows - 1,
        .cols = cv->out, .ys = { 0, (int64_t)cols * cv->out, 1 },
        .as = { 0, cv->out, 1 }
    };
    tap(self, t, x, cv, k, (int64_t)cols * cv->in, rows - 1, true);
    map(self, &fix);
    drop(self, t);
}

static void conv3(struct separator * self, float * y, const float * x,
                  const struct conv * cv, int32_t rows, int32_t cols) {
    const int32_t count = rows * cols;
    spread_rows(self, y, cv->bias, count, cv->out);
    tap(self, y, x, cv, 1, cv->in, count, false);
    tap(self, y + cv->out, x, cv, 0, cv->in, count - 1, false);
    tap(self, y, x + cv->in, cv, 2, cv->in, count - 1, false);
    if (rows > 1) {
        edge(self, y + (size_t)cols * cv->out,
             x + (size_t)(cols - 1) * cv->in, 0, cv, rows, cols);
        edge(self, y + (size_t)(cols - 1) * cv->out,
             x + (size_t)cols * cv->in, 2, cv, rows, cols);
    }
}

static void conv(struct separator * self, float * y, const float * x,
                 const struct conv * cv, int32_t rows, int32_t cols) {
    const int64_t wide = (int64_t)cols + 2;
    const int64_t count = rows * wide;
    const size_t room = (size_t)(((rows + 2) * wide + 2) * cv->in);
    float * padded = take(self, room);
    float * full = take(self, (size_t)(count * cv->out));
    const struct map_args in = {
        .y = padded + (2 + wide) * cv->in, .a = x, .op = OP_COPY,
        .rows = rows, .cols = cols * cv->in, .ys = { 0, wide * cv->in, 1 },
        .as = { 0, (int64_t)cols * cv->in, 1 }
    };
    const struct map_args out = {
        .y = y, .a = full + cv->out, .op = OP_COPY, .rows = rows,
        .cols = cols * cv->out, .ys = { 0, (int64_t)cols * cv->out, 1 },
        .as = { 0, wide * cv->out, 1 }
    };
    border(self, padded, rows, wide, cv->in);
    map(self, &in);
    spread_rows(self, full, cv->bias, (int32_t)count, cv->out);
    for (int32_t k = 0; k < cv->span; k++) {
        const int32_t dr = cv->span == 9 ? k / 3 - 1 : 0;
        const int64_t at = 1 + (1 + dr) * wide + k % 3 - 1;
        const struct gemm_args g = {
            .a = padded + at * cv->in, .b = tap_at(cv, k), .c = full,
            .m = (int32_t)count, .n = cv->out, .k = cv->in, .lda = cv->in,
            .ldb = cv->ldb, .ldc = cv->out, .tb = true, .format = cv->format
        };
        gemm(self, &g);
    }
    map(self, &out);
    drop(self, padded);
    drop(self, full);
}

static void rope(struct separator * self, float * qkv, const float * table,
                 int32_t seqs, int32_t len) {
    for (int32_t side = 0; side < 2; side++) {
        const struct map_args m = {
            .y = qkv + side * DIM, .a = table, .op = OP_ROPE, .batch = seqs,
            .rows = len, .cols = DIM / 2,
            .ys = { (int64_t)len * 3 * DIM, 3 * DIM, 2 },
            .as = { 0, DIM, 2 }
        };
        map(self, &m);
    }
}

static void attend(struct separator * self, float * o, const float * qkv,
                   int32_t seqs, int32_t len, bool causal) {
    const int64_t square = (int64_t)len * len;
    float * s = take(self, (size_t)(seqs * square));
    for (int32_t h = 0; h < HEADS; h++) {
        const struct gemm_args scores = {
            .a = qkv + h * HD, .b = qkv + DIM + h * HD, .c = s, .m = len,
            .n = len, .k = HD, .lda = 3 * DIM, .ldb = 3 * DIM, .ldc = len,
            .tb = true, .batch = seqs, .sa = (int64_t)len * 3 * DIM,
            .sb = (int64_t)len * 3 * DIM, .sc = square, .fresh = true
        };
        const struct gemm_args mixed = {
            .a = s, .b = qkv + 2 * DIM + h * HD, .c = o + h * HD, .m = len,
            .n = HD, .k = len, .lda = len, .ldb = 3 * DIM, .ldc = DIM,
            .batch = seqs, .sa = square, .sb = (int64_t)len * 3 * DIM,
            .sc = (int64_t)len * DIM, .fresh = true
        };
        gemm(self, &scores);
        self->backend->softmax(self->backend, s, seqs * len, len,
                               1.0f / sqrtf((float)HD), causal ? len : 0);
        gemm(self, &mixed);
    }
    drop(self, s);
}

static int32_t sequences(int32_t seqs, int32_t len) {
    const int32_t fit = SLICE / len > 0 ? SLICE / len : 1;
    return seqs < fit ? seqs : fit;
}

static void self_attention(struct separator * self, float * x,
                           const struct attn * a, const float * table,
                           int32_t seqs, int32_t len, bool causal) {
    const int32_t most = sequences(seqs, len);
    const size_t room = (size_t)most * len;
    float * h = take(self, room * DIM);
    float * qkv = take(self, room * 3 * DIM);
    float * o = take(self, room * DIM);
    for (int32_t at = 0; at < seqs; at += most) {
        const int32_t n = seqs - at < most ? seqs - at : most;
        const int32_t rows = n * len;
        float * y = x + (size_t)at * len * DIM;
        rms(self, h, y, a->norm, rows);
        linear(self, qkv, h, &a->qkv, NULL, rows, 3 * DIM, DIM);
        if (table != NULL) {
            rope(self, qkv, table, n, len);
        }
        attend(self, o, qkv, n, len, causal);
        linear(self, h, o, &a->out, NULL, rows, DIM, DIM);
        flat(self, OP_ADD, y, h, 1.0f, (size_t)rows * DIM);
    }
    drop(self, h);
    drop(self, qkv);
    drop(self, o);
}

static void convffn(struct separator * self, float * x,
                    const struct convffn * f, int32_t seqs, int32_t len) {
    const int32_t most = sequences(seqs, len);
    const size_t room = (size_t)most * len;
    float * h = take(self, room * DIM);
    float * a = take(self, room * 2 * HIDDEN);
    float * g = take(self, room * HIDDEN);
    for (int32_t at = 0; at < seqs; at += most) {
        const int32_t n = seqs - at < most ? seqs - at : most;
        const int32_t rows = n * len;
        float * y = x + (size_t)at * len * DIM;
        rms(self, h, y, f->norm, rows);
        conv3(self, a, h, &f->up, n, len);
        swiglu(self, g, a, rows, HIDDEN);
        conv3(self, h, g, &f->down, n, len);
        flat(self, OP_ADD, y, h, 0.5f, (size_t)rows * DIM);
    }
    drop(self, h);
    drop(self, a);
    drop(self, g);
}

static void ffn(struct separator * self, float * x, const struct ffn * f,
                int32_t rows) {
    const int32_t most = rows < SLICE ? rows : SLICE;
    float * h = take(self, (size_t)most * DIM);
    float * a = take(self, (size_t)most * 2 * WIDE);
    float * g = take(self, (size_t)most * WIDE);
    for (int32_t at = 0; at < rows; at += most) {
        const int32_t n = rows - at < most ? rows - at : most;
        float * y = x + (size_t)at * DIM;
        rms(self, h, y, f->norm, n);
        linear(self, a, h, &f->up, f->up_bias, n, 2 * WIDE, DIM);
        swiglu(self, g, a, n, WIDE);
        linear(self, h, g, &f->down, f->down_bias, n, DIM, WIDE);
        flat(self, OP_ADD, y, h, 1.0f, (size_t)n * DIM);
    }
    drop(self, h);
    drop(self, a);
    drop(self, g);
}

static void trans(struct separator * self, float * x, const struct trans * t,
                  const float * table, int32_t seqs, int32_t len) {
    convffn(self, x, &t->ffn1, seqs, len);
    self_attention(self, x, &t->sa, table, seqs, len, false);
    convffn(self, x, &t->ffn2, seqs, len);
}

static void transpose(struct separator * self, float * y, const float * x,
                      int32_t rows, int32_t cols) {
    const struct map_args m = {
        .y = y, .a = x, .op = OP_COPY, .batch = rows, .rows = cols,
        .cols = DIM, .ys = { DIM, (int64_t)rows * DIM, 1 },
        .as = { (int64_t)cols * DIM, DIM, 1 }
    };
    map(self, &m);
}

static void tf_block(struct separator * self, float * x,
                     const struct trans * b, float * const * tables,
                     int32_t frames) {
    float * y = take(self, (size_t)frames * BINS * DIM);
    trans(self, x, &b[0], tables[0], frames, BINS);
    transpose(self, y, x, frames, BINS);
    trans(self, y, &b[1], tables[1], BINS, frames);
    transpose(self, x, y, BINS, frames);
    drop(self, y);
}

static void stft(struct separator * self, float * spec, const float * x,
                 int32_t frames) {
    float * cut = take(self, (size_t)frames * WINDOW);
    const struct gemm_args g = {
        .a = cut, .b = self->basis, .c = spec, .m = frames, .n = 2 * BINS,
        .k = WINDOW, .lda = WINDOW, .ldb = WINDOW, .ldc = 2 * BINS,
        .tb = true, .fresh = true
    };
    const struct map_args m = {
        .y = cut, .a = x, .op = OP_COPY, .rows = frames, .cols = WINDOW,
        .ys = { 0, WINDOW, 1 }, .as = { 0, HOP, 1 }
    };
    map(self, &m);
    gemm(self, &g);
    drop(self, cut);
}

static float bin_re(const float * spec, int32_t frames, int32_t t,
                    int32_t f) {
    const bool inside = t >= 0 && t < frames && f >= 0 && f < BINS;
    return inside ? spec[(size_t)t * 2 * BINS + f] : 0.0f;
}

static float bin_im(const float * spec, int32_t frames, int32_t t,
                    int32_t f) {
    const bool inside = t >= 0 && t < frames && f >= 0 && f < BINS;
    return inside ? spec[(size_t)t * 2 * BINS + BINS + f] : 0.0f;
}

static void correlate(float * y, const float * spec, int32_t frames) {
    float level[BINS];
    for (int32_t f = 0; f < BINS; f++) {
        float sum = 0.0f;
        for (int32_t t = 0; t < frames; t++) {
            const float re = bin_re(spec, frames, t, f);
            const float im = bin_im(spec, frames, t, f);
            sum += re * re + im * im;
        }
        level[f] = sqrtf(sum / (float)frames + 1e-12f);
    }
    for (int32_t r = 0; r < frames * BINS; r++) {
        const int32_t t = r / BINS;
        const int32_t f = r % BINS;
        const float ar = bin_re(spec, frames, t, f) / level[f];
        const float ai = bin_im(spec, frames, t, f) / level[f];
        for (int32_t l = 0; l < 9; l++) {
            const float br = bin_re(spec, frames, t + l % 3 - 1,
                                    f + l / 3 - 1) / level[f];
            const float bi = bin_im(spec, frames, t + l % 3 - 1,
                                    f + l / 3 - 1) / level[f];
            const float scot = sqrtf(ar * ar + ai * ai) *
                               sqrtf(br * br + bi * bi);
            const float d = sqrtf(scot + 1e-12f);
            y[(size_t)r * CORR + 2 * l] = (ar * br + ai * bi) / d;
            y[(size_t)r * CORR + 2 * l + 1] = (ai * br - ar * bi) / d;
        }
    }
}

static void sinusoid(float * v, float at, int32_t channels) {
    const float rate = logf(10000.0f) / (float)(channels / 2 - 1);
    for (int32_t j = 0; j < channels / 2; j++) {
        const float angle = at * expf(-rate * (float)j);
        v[j] = sinf(angle);
        v[channels / 2 + j] = cosf(angle);
    }
}

static float * turns(struct separator * self, const float * freqs,
                     int32_t len) {
    float * t = take(self, (size_t)len * DIM);
    for (int32_t r = 0; r < len; r++) {
        for (int32_t p = 0; p < DIM / 2; p++) {
            const float angle = (float)r * freqs[p % (HD / 2)];
            t[(size_t)r * DIM + 2 * p] = cosf(angle);
            t[(size_t)r * DIM + 2 * p + 1] = sinf(angle);
        }
    }
    return t;
}

static struct places places_make(struct separator * self, int32_t frames) {
    struct places p = {
        .bins = take(self, BINS * DIM),
        .times = take(self, (size_t)frames * DIM / 2),
        .freqs = take(self, BINS * DIM / 2)
    };
    for (int32_t b = 0; b < BLOCKS; b++) {
        p.rope[b][0] = turns(self, self->blocks[b][0].sa.freqs, BINS);
        p.rope[b][1] = turns(self, self->blocks[b][1].sa.freqs, frames);
    }
    for (int32_t f = 0; f < BINS; f++) {
        sinusoid(p.bins + f * DIM, (float)f, DIM);
        sinusoid(p.freqs + f * DIM / 2, (float)f, DIM / 2);
    }
    for (int32_t t = 0; t < frames; t++) {
        sinusoid(p.times + (size_t)t * DIM / 2, (float)t, DIM / 2);
    }
    return p;
}

static void places_drop(struct separator * self, struct places * p) {
    for (int32_t b = 0; b < BLOCKS; b++) {
        drop(self, p->rope[b][0]);
        drop(self, p->rope[b][1]);
    }
    drop(self, p->bins);
    drop(self, p->times);
    drop(self, p->freqs);
}

static void encode(struct separator * self, float * x, const float * spec,
                   const struct places * p, int32_t frames) {
    const size_t rows = (size_t)frames * BINS;
    float * corr = take(self, rows * CORR);
    float * wide = take(self, rows * EMBED);
    const struct map_args bins = {
        .y = x, .a = p->bins, .op = OP_ADD, .alpha = 1.0f, .batch = frames,
        .rows = BINS, .cols = DIM, .ys = { BINS * DIM, DIM, 1 },
        .as = { 0, DIM, 1 }
    };
    sync(self);
    correlate(corr, spec, frames);
    conv(self, wide, corr, &self->embed1, frames, BINS);
    flat(self, OP_SILU, wide, wide, 0.0f, rows * EMBED);
    conv(self, x, wide, &self->embed2, frames, BINS);
    rms(self, x, x, self->embed_norm, (int32_t)rows);
    map(self, &bins);
    for (int32_t b = 0; b < ENCODERS; b++) {
        tf_block(self, x, self->blocks[b], p->rope[b], frames);
    }
    drop(self, corr);
    drop(self, wide);
}

static void cross(struct separator * self, float * q, const float * kv,
                  int32_t rows, const struct cross * c) {
    float * h = take(self, QUERIES * DIM);
    float * qq = take(self, QUERIES * DIM);
    float * pairs = take(self, (size_t)rows * 2 * DIM);
    float * o = take(self, QUERIES * DIM);
    float * s = take(self, (size_t)QUERIES * rows);
    rms(self, h, q, c->norm, QUERIES);
    linear(self, qq, h, &c->q, NULL, QUERIES, DIM, DIM);
    linear(self, pairs, kv, &c->kv, NULL, rows, 2 * DIM, DIM);
    for (int32_t i = 0; i < HEADS; i++) {
        const struct gemm_args scores = {
            .a = qq + i * HD, .b = pairs + i * HD, .c = s, .m = QUERIES,
            .n = rows, .k = HD, .lda = DIM, .ldb = 2 * DIM, .ldc = rows,
            .tb = true, .fresh = true
        };
        const struct gemm_args mixed = {
            .a = s, .b = pairs + DIM + i * HD, .c = o + i * HD, .m = QUERIES,
            .n = HD, .k = rows, .lda = rows, .ldb = 2 * DIM, .ldc = DIM,
            .fresh = true
        };
        gemm(self, &scores);
        self->backend->softmax(self->backend, s, QUERIES, rows,
                               1.0f / sqrtf((float)HD), 0);
        gemm(self, &mixed);
    }
    linear(self, h, o, &c->out, NULL, QUERIES, DIM, DIM);
    flat(self, OP_ADD, q, h, 1.0f, QUERIES * DIM);
    drop(self, h);
    drop(self, qq);
    drop(self, pairs);
    drop(self, o);
    drop(self, s);
}

static float presence(struct separator * self, float * a) {
    float norm = 0.0f;
    float logit = self->pres_b[0];
    for (int32_t c = 0; c < DIM; c++) {
        norm += a[c] * a[c];
    }
    for (int32_t c = 0; c < DIM; c++) {
        a[c] /= sqrtf(norm) + 1e-8f;
        logit += a[c] * self->pres_w[c];
    }
    return logit;
}

static int32_t present(struct separator * self, float * att) {
    float logit[QUERIES];
    float rows[QUERIES * DIM];
    int32_t order[QUERIES];
    int32_t count = 0;
    for (int32_t i = 0; i < QUERIES; i++) {
        logit[i] = presence(self, att + i * DIM);
        order[i] = i;
    }
    for (int32_t i = 2; i < QUERIES; i++) {
        const int32_t pick = order[i];
        int32_t j = i;
        while (j > 1 && logit[order[j - 1]] < logit[pick]) {
            order[j] = order[j - 1];
            j--;
        }
        order[j] = pick;
    }
    memcpy(rows, att, sizeof rows);
    for (int32_t i = 1; i < QUERIES; i++) {
        memcpy(att + i * DIM, rows + order[i] * DIM, DIM * sizeof(float));
        count += 1.0f / (1.0f + expf(-logit[order[i]])) > 0.5f ? 1 : 0;
    }
    return count;
}

static int32_t attract(struct separator * self, float * att, const float * x,
                       const struct places * p, int32_t frames) {
    const int32_t rows = frames * BINS;
    float * kv = take(self, (size_t)rows * DIM);
    const struct map_args times = {
        .y = kv, .a = p->times, .op = OP_ADD, .alpha = 1.0f,
        .batch = frames, .rows = BINS, .cols = DIM / 2,
        .ys = { BINS * DIM, DIM, 1 }, .as = { DIM / 2, 0, 1 }
    };
    const struct map_args freqs = {
        .y = kv + DIM / 2, .a = p->freqs, .op = OP_ADD, .alpha = 1.0f,
        .batch = frames, .rows = BINS, .cols = DIM / 2,
        .ys = { BINS * DIM, DIM, 1 }, .as = { 0, DIM / 2, 1 }
    };
    layer(self, kv, x, self->split_norm, self->split_bias, rows);
    map(self, &times);
    map(self, &freqs);
    flat(self, OP_COPY, att, self->query, 0.0f, QUERIES * DIM);
    for (int32_t d = 0; d < 2; d++) {
        cross(self, att, kv, rows, &self->dec[d].ca);
        self_attention(self, att, &self->dec[d].sa, NULL, 1, QUERIES, true);
        ffn(self, att, &self->dec[d].ffn, QUERIES);
    }
    sync(self);
    drop(self, kv);
    return present(self, att);
}

static void split(struct separator * self, float * y, const float * shared,
                  const float * att, int32_t rows) {
    float * h = take(self, (size_t)rows * WIDE);
    float * a = take(self, WIDE);
    const struct gemm_args g = {
        .a = att, .b = self->fuse_w1 + DIM, .c = a, .m = 1, .n = WIDE,
        .k = DIM, .lda = DIM, .ldb = 2 * DIM, .ldc = WIDE, .tb = true
    };
    const struct map_args add = {
        .y = h, .a = a, .op = OP_ADD, .alpha = 1.0f, .rows = rows,
        .cols = WIDE, .ys = { 0, WIDE, 1 }, .as = { 0, 0, 1 }
    };
    flat(self, OP_COPY, a, self->fuse_b1, 0.0f, WIDE);
    gemm(self, &g);
    flat(self, OP_COPY, h, shared, 0.0f, (size_t)rows * WIDE);
    map(self, &add);
    flat(self, OP_SILU, h, h, 0.0f, (size_t)rows * WIDE);
    linear(self, y, h, &self->fuse_w2, self->fuse_b2, rows, DIM, WIDE);
    layer(self, y, y, self->last_norm, self->last_bias, rows);
    drop(self, h);
    drop(self, a);
}

static void gather(struct separator * self, float * b, float ** xs,
                   int32_t n, int32_t rows, bool back) {
    for (int32_t k = 0; k < n; k++) {
        const struct map_args into = {
            .y = b + k * DIM, .a = xs[k], .op = OP_COPY, .rows = rows,
            .cols = DIM, .ys = { 0, (int64_t)n * DIM, 1 },
            .as = { 0, DIM, 1 }
        };
        const struct map_args from = {
            .y = xs[k], .a = b + k * DIM, .op = OP_COPY, .rows = rows,
            .cols = DIM, .ys = { 0, DIM, 1 },
            .as = { 0, (int64_t)n * DIM, 1 }
        };
        map(self, back ? &from : &into);
    }
}

static void speakers(struct separator * self, float ** xs, int32_t n,
                     int32_t rows, int32_t i) {
    float * b = take(self, (size_t)rows * n * DIM);
    gather(self, b, xs, n, rows, false);
    self_attention(self, b, &self->cs_sa[i], NULL, rows, n, false);
    ffn(self, b, &self->cs_ffn[i], rows * n);
    gather(self, b, xs, n, rows, true);
    drop(self, b);
}

static void filter(float * mixed, const float * m, const float * spec,
                   int32_t frames) {
    for (int32_t r = 0; r < frames * BINS; r++) {
        const int32_t t = r / BINS;
        const int32_t f = r % BINS;
        float re = 0.0f;
        float im = 0.0f;
        for (int32_t l = 0; l < 9; l++) {
            const float * o = m + (size_t)r * OUTS + l * 3;
            const float gain = o[0] > 20.0f ? o[0] : log1pf(expf(o[0]));
            const float wr = gain * tanhf(o[1]);
            const float wi = gain * tanhf(o[2]);
            const float xr = bin_re(spec, frames, t + l % 3 - 1,
                                    f + l / 3 - 1);
            const float xi = bin_im(spec, frames, t + l % 3 - 1,
                                    f + l / 3 - 1);
            re += xr * wr - xi * wi;
            im += xr * wi + xi * wr;
        }
        mixed[(size_t)t * 2 * BINS + f] = re;
        mixed[(size_t)t * 2 * BINS + BINS + f] = im;
    }
}

static void voice(struct separator * self, float * wave, const float * x,
                  const float * spec, int32_t frames) {
    const int32_t rows = frames * BINS;
    float * m = take(self, (size_t)rows * OUTS);
    float * mixed = take(self, (size_t)frames * 2 * BINS);
    float * cut = take(self, (size_t)frames * WINDOW);
    const struct gemm_args g = {
        .a = mixed, .b = self->basis, .c = cut, .m = frames, .n = WINDOW,
        .k = 2 * BINS, .lda = 2 * BINS, .ldb = WINDOW, .ldc = WINDOW,
        .fresh = true
    };
    linear(self, m, x, &self->mask_w, self->mask_b, rows, OUTS, DIM);
    sync(self);
    filter(mixed, m, spec, frames);
    gemm(self, &g);
    sync(self);
    for (int32_t t = 0; t < frames; t++) {
        for (int32_t n = 0; n < WINDOW; n++) {
            wave[t * HOP + n] += cut[(size_t)t * WINDOW + n];
        }
    }
    drop(self, m);
    drop(self, mixed);
    drop(self, cut);
}

static void decode(struct separator * self, float ** xs, int32_t n,
                   const struct places * p, int32_t frames) {
    for (int32_t i = 0; i < DECODERS; i++) {
        for (int32_t k = 0; k < n; k++) {
            tf_block(self, xs[k], self->blocks[ENCODERS + i],
                     p->rope[ENCODERS + i], frames);
        }
        speakers(self, xs, n, frames * BINS, i);
    }
}

static int32_t separate_frames(struct separator * self, const float * x,
                               int32_t frames, float ** out,
                               int32_t capacity) {
    const int32_t rows = frames * BINS;
    float * spec = take(self, (size_t)frames * 2 * BINS);
    float * enc = take(self, (size_t)rows * DIM);
    float * shared = take(self, (size_t)rows * WIDE);
    float * att = take(self, QUERIES * DIM);
    float * xs[QUERIES] = { 0 };
    struct places p = places_make(self, frames);
    const struct gemm_args g = {
        .a = enc, .b = self->fuse_w1, .c = shared, .m = rows, .n = WIDE,
        .k = DIM, .lda = DIM, .ldb = 2 * DIM, .ldc = WIDE, .tb = true,
        .fresh = true
    };
    stft(self, spec, x, frames);
    encode(self, enc, spec, &p, frames);
    const int32_t found = attract(self, att, enc, &p, frames);
    const int32_t n = found < capacity ? found : capacity;
    gemm(self, &g);
    for (int32_t k = 0; k < n; k++) {
        xs[k] = take(self, (size_t)rows * DIM);
        split(self, xs[k], shared, att + (k + 1) * DIM, rows);
    }
    decode(self, xs, n, &p, frames);
    for (int32_t k = 0; k < n; k++) {
        voice(self, out[k], xs[k], spec, frames);
        drop(self, xs[k]);
    }
    places_drop(self, &p);
    drop(self, spec);
    drop(self, enc);
    drop(self, shared);
    drop(self, att);
    return n;
}

static int32_t separate8(struct separator * self, const float * x,
                         int32_t count, float ** out, int32_t capacity) {
    const int32_t frames = count >= WINDOW ? (count - WINDOW) / HOP + 1 : 0;
    float * mix = take(self, (size_t)count + 1);
    int32_t n = 0;
    sync(self);
    memcpy(mix, x, (size_t)count * sizeof(float));
    if (frames > 0) {
        n = separate_frames(self, mix, frames, out, capacity);
    }
    drop(self, mix);
    return n;
}

static void halve(const struct separator * self, float * y, const float * x,
                  int32_t count) {
    for (int32_t i = 0; i < count / 2; i++) {
        float sum = 0.0f;
        for (int32_t j = 0; j < TAPS; j++) {
            const int32_t at = 2 * i + j - TAPS / 2;
            sum += at >= 0 && at < count ? self->lowpass[j] * x[at] : 0.0f;
        }
        y[i] = sum;
    }
}

static void twice(const struct separator * self, float * y, const float * x,
                  int32_t count, float gain) {
    for (int32_t m = 0; m < count; m++) {
        float sum = 0.0f;
        for (int32_t j = (m + TAPS / 2) % 2; j < TAPS; j += 2) {
            const int32_t at = (m + j - TAPS / 2) / 2;
            sum += at >= 0 && at < count / 2 ? self->lowpass[j] * x[at] :
                   0.0f;
        }
        y[m] = 2.0f * gain * sum;
    }
}

static float deviation(const float * x, int32_t count) {
    double mean = 0.0;
    double sum = 0.0;
    for (int32_t i = 0; i < count; i++) {
        mean += x[i];
    }
    mean /= count > 0 ? count : 1;
    for (int32_t i = 0; i < count; i++) {
        sum += (x[i] - mean) * (x[i] - mean);
    }
    return (float)sqrt(sum / (count > 1 ? count - 1 : 1));
}

static int32_t ss_separate(struct ss * handle, const float * samples,
                           int32_t count, float ** voices, int32_t capacity) {
    struct separator * self = (struct separator *)handle;
    const int32_t half = count / 2;
    const int32_t most = capacity < QUERIES - 1 ? capacity : QUERIES - 1;
    float * x = floats((size_t)half + 1);
    float * out[QUERIES] = { 0 };
    halve(self, x, samples, count);
    const float level = deviation(x, half) + 1e-8f;
    for (int32_t i = 0; i < half; i++) {
        x[i] /= level;
    }
    for (int32_t k = 0; k < most; k++) {
        out[k] = floats((size_t)half + 1);
    }
    const int32_t n = separate8(self, x, half, out, most);
    for (int32_t k = 0; k < most; k++) {
        if (k < n) {
            voices[k] = floats((size_t)count + 1);
            twice(self, voices[k], out[k], count, level);
        }
        free(out[k]);
    }
    free(x);
    return n;
}

static void ss_close(struct ss * handle) {
    struct separator * self = (struct separator *)handle;
    for (int32_t i = 0; i < self->owned_count; i++) {
        drop(self, self->owned[i]);
    }
    if (self->file != NULL) {
        self->backend->adopt(self->backend, self->file, 0);
    }
    file_unmap(self->file, self->size);
    free(self);
}

static struct ss * ss_open_pack(const char * path, struct backend * backend,
                                const char * const * options) {
    struct separator * self = calloc(1, sizeof *self);
    (void)options;
    if (self != NULL) {
        self->ss = (struct ss){
            .name = "ss", .separate = ss_separate, .close = ss_close
        };
        self->backend = backend;
        self->file = file_map(path, &self->size);
        self->header_size = self->file != NULL && self->size > 8 ?
                            (size_t)le64(self->file) : 0;
        self->header = (const char *)self->file + 8;
    }
    const bool mapped = self != NULL && self->header_size > 0 &&
                        self->header_size < self->size - 8;
    if (mapped) {
        backend->adopt(backend, self->file, self->size);
        weights_load(self);
        sync(self);
    }
    const bool ready = mapped && self->faults == 0;
    if (!ready && self != NULL) {
        ss_close(&self->ss);
    }
    return ready ? &self->ss : NULL;
}

INSTALL(ss, ss,
        .pack = "leok7v/sr-corrnet-ss/sr-corrnet-ss-2-5spk.safetensors",
        .keys = "", .open = ss_open_pack)

#ifndef NO_MAIN

int main(int argc, char ** argv) {
    (void)argc;
    return ss_run("ss", (const char * const *)argv + 1);
}

#endif
