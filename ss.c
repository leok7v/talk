#include "ss.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { DIM = 128, HEADS = 4, HD = 32, HIDDEN = 384, WIDE = 512, BINS = 65,
       WINDOW = 128, HOP = 64, CORR = 18, EMBED = 512, QUERIES = 7,
       BLOCKS = 6, ENCODERS = 2, DECODERS = 4, OUTS = 27, TAPS = 31,
       NAME = 192, OWNED = 64, SHORT = 16 };

static const float rms_epsilon = 1.1920929e-07f;
static const float ln_epsilon = 1e-5f;

struct conv {
    float * taps;
    const float * bias;
    int32_t out;
    int32_t in;
    int32_t span;
};

struct convffn {
    const float * norm;
    struct conv up;
    struct conv down;
};

struct attn {
    const float * norm;
    const float * qkv;
    const float * out;
    const float * freqs;
};

struct trans {
    struct convffn ffn1;
    struct attn sa;
    struct convffn ffn2;
};

struct ffn {
    const float * norm;
    const float * up;
    const float * up_bias;
    const float * down;
    const float * down_bias;
};

struct cross {
    const float * norm;
    const float * q;
    const float * kv;
    const float * out;
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
    const float * fuse_w2;
    const float * fuse_b2;
    const float * mask_w;
    const float * mask_b;
    const float * last_norm;
    const float * last_bias;
    float basis[2 * BINS * WINDOW];
    float lowpass[TAPS];
};

static const float * tensor(struct separator * self, const char * prefix,
                            const char * suffix, size_t count) {
    char name[NAME];
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
    const bool fine = span != NULL &&
                      (size_t)(to - from) == count * sizeof(float) &&
                      8 + self->header_size + (size_t)to <= self->size &&
                      memmem(at, (size_t)(end - at), "\"dtype\":\"F32\"",
                             13) != NULL;
    self->faults += fine ? 0 : 1;
    return fine ? data : NULL;
}

static float * own(struct separator * self, size_t count) {
    float * p = floats(count);
    must(self->owned_count < OWNED, "ss owned");
    self->owned[self->owned_count++] = p;
    return p;
}

static struct conv conv_load(struct separator * self, const char * prefix,
                             const char * stem, int32_t out, int32_t in,
                             int32_t span) {
    char name[NAME];
    snprintf(name, sizeof name, "%s.weight", stem);
    const float * w = tensor(self, prefix, name, (size_t)out * in * span);
    struct conv c = {
        .taps = own(self, (size_t)out * in * span), .out = out, .in = in,
        .span = span
    };
    snprintf(name, sizeof name, "%s.bias", stem);
    c.bias = tensor(self, prefix, name, (size_t)out);
    for (size_t i = 0; w != NULL && i < (size_t)out * in * (size_t)span;
         i++) {
        const size_t k = i % (size_t)span;
        const size_t row = i / (size_t)span;
        c.taps[k * (size_t)out * in + row] = w[i];
    }
    return c;
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
        .qkv = tensor(self, prefix, "block.qkv.weight", 3 * DIM * DIM),
        .out = tensor(self, prefix, "block.aggregate_heads.0.weight",
                      DIM * DIM),
        .freqs = rope ? tensor(self, prefix, "block.rope.freqs", HD / 2) :
                 NULL
    };
}

static struct ffn ffn_load(struct separator * self, const char * prefix) {
    return (struct ffn){
        .norm = tensor(self, prefix, "layernorm.weight", DIM),
        .up = tensor(self, prefix, "net.0.weight", 2 * WIDE * DIM),
        .up_bias = tensor(self, prefix, "net.0.bias", 2 * WIDE),
        .down = tensor(self, prefix, "net.2.weight", DIM * WIDE),
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
        .q = tensor(self, name, "block.q.weight", DIM * DIM),
        .kv = tensor(self, name, "block.kv.weight", 2 * DIM * DIM),
        .out = tensor(self, name, "block.aggregate_heads.0.weight", DIM * DIM)
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
    self->fuse_w2 = tensor(self, s, "split.fusion.2.weight", DIM * WIDE);
    self->fuse_b2 = tensor(self, s, "split.fusion.2.bias", DIM);
}

static void tables_load(struct separator * self) {
    float sum = 0.0f;
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
    self->mask_w = tensor(self, "filter_estim", "mask.net.weight", OUTS * DIM);
    self->mask_b = tensor(self, "filter_estim", "mask.net.bias", OUTS);
    self->last_norm = tensor(self, "", "layernorm.weight", DIM);
    self->last_bias = tensor(self, "", "layernorm.bias", DIM);
    tables_load(self);
}

static void gemm(struct separator * self, const struct gemm_args * g) {
    self->backend->gemm(self->backend, g);
}

static void linear(struct separator * self, float * y, const float * x,
                   const float * w, const float * bias, int32_t rows,
                   int32_t out, int32_t in) {
    const struct gemm_args g = {
        .a = x, .b = w, .c = y, .m = rows, .n = out, .k = in, .lda = in,
        .ldb = in, .ldc = out, .tb = true
    };
    for (size_t i = 0; i < (size_t)rows * out; i++) {
        y[i] = bias != NULL ? bias[i % (size_t)out] : 0.0f;
    }
    gemm(self, &g);
}

static float silu(float x) {
    return x / (1.0f + expf(-x));
}

static void rms(float * y, const float * x, const float * w, int32_t rows) {
    for (int32_t r = 0; r < rows; r++) {
        const float * in = x + (size_t)r * DIM;
        float * out = y + (size_t)r * DIM;
        float sum = 0.0f;
        for (int32_t i = 0; i < DIM; i++) {
            sum += in[i] * in[i];
        }
        const float gain = 1.0f / sqrtf(sum / (float)DIM + rms_epsilon);
        for (int32_t i = 0; i < DIM; i++) {
            out[i] = in[i] * gain * w[i];
        }
    }
}

static void layer(float * y, const float * x, const float * w,
                  const float * b, int32_t rows) {
    for (int32_t r = 0; r < rows; r++) {
        const float * in = x + (size_t)r * DIM;
        float * out = y + (size_t)r * DIM;
        float mean = 0.0f;
        float spread = 0.0f;
        for (int32_t i = 0; i < DIM; i++) {
            mean += in[i];
        }
        mean /= (float)DIM;
        for (int32_t i = 0; i < DIM; i++) {
            spread += (in[i] - mean) * (in[i] - mean);
        }
        const float gain = 1.0f / sqrtf(spread / (float)DIM + ln_epsilon);
        for (int32_t i = 0; i < DIM; i++) {
            out[i] = (in[i] - mean) * gain * w[i] + b[i];
        }
    }
}

static void swiglu(float * y, const float * x, int32_t rows, int32_t half) {
    for (int32_t r = 0; r < rows; r++) {
        for (int32_t i = 0; i < half; i++) {
            y[(size_t)r * half + i] = x[(size_t)r * 2 * half + i] *
                silu(x[(size_t)r * 2 * half + half + i]);
        }
    }
}

static void shift(float * y, const float * x, int32_t rows, int32_t cols,
                  int32_t width, int32_t dr, int32_t dc) {
    for (int32_t r = 0; r < rows; r++) {
        for (int32_t c = 0; c < cols; c++) {
            const bool inside = r + dr >= 0 && r + dr < rows &&
                                c + dc >= 0 && c + dc < cols;
            float * to = y + ((size_t)r * cols + c) * width;
            if (inside) {
                memcpy(to, x + ((size_t)(r + dr) * cols + c + dc) * width,
                       (size_t)width * sizeof(float));
            } else {
                memset(to, 0, (size_t)width * sizeof(float));
            }
        }
    }
}

static void conv(struct separator * self, float * y, const float * x,
                 const struct conv * cv, int32_t rows, int32_t cols) {
    const size_t count = (size_t)rows * cols;
    float * moved = floats(count * (size_t)cv->in);
    for (size_t i = 0; i < count * (size_t)cv->out; i++) {
        y[i] = cv->bias[i % (size_t)cv->out];
    }
    for (int32_t k = 0; k < cv->span; k++) {
        const struct gemm_args g = {
            .a = moved, .b = cv->taps + (size_t)k * cv->out * cv->in,
            .c = y, .m = (int32_t)count, .n = cv->out, .k = cv->in,
            .lda = cv->in, .ldb = cv->in, .ldc = cv->out, .tb = true
        };
        shift(moved, x, rows, cols, cv->in, cv->span == 9 ? k / 3 - 1 : 0,
              k % 3 - 1);
        gemm(self, &g);
    }
    free(moved);
}

static void rope(float * qkv, const float * freqs, int32_t rows,
                 int32_t len) {
    for (int32_t r = 0; r < rows; r++) {
        const float at = (float)(r % len);
        for (int32_t i = 0; i < DIM; i += 2) {
            const float angle = at * freqs[(i % HD) / 2];
            const float c = cosf(angle);
            const float s = sinf(angle);
            for (int32_t side = 0; side < 2; side++) {
                float * v = qkv + (size_t)r * 3 * DIM + side * DIM + i;
                const float a = v[0];
                v[0] = a * c - v[1] * s;
                v[1] = v[1] * c + a * s;
            }
        }
    }
}

static void soften(float * s, int32_t n, float scale) {
    float top = -INFINITY;
    float sum = 0.0f;
    for (int32_t j = 0; j < n; j++) {
        s[j] *= scale;
        top = fmaxf(top, s[j]);
    }
    for (int32_t j = 0; j < n; j++) {
        s[j] = expf(s[j] - top);
        sum += s[j];
    }
    for (int32_t j = 0; j < n; j++) {
        s[j] /= sum;
    }
}

static void attend_short(float * o, const float * qkv, int32_t len,
                         int32_t h, bool causal) {
    float s[SHORT];
    for (int32_t i = 0; i < len; i++) {
        const float * q = qkv + (size_t)i * 3 * DIM + h * HD;
        const int32_t seen = causal ? i + 1 : len;
        for (int32_t j = 0; j < seen; j++) {
            const float * k = qkv + (size_t)j * 3 * DIM + DIM + h * HD;
            s[j] = 0.0f;
            for (int32_t d = 0; d < HD; d++) {
                s[j] += q[d] * k[d];
            }
        }
        soften(s, seen, 1.0f / sqrtf((float)HD));
        for (int32_t j = 0; j < seen; j++) {
            const float * v = qkv + (size_t)j * 3 * DIM + 2 * DIM + h * HD;
            for (int32_t d = 0; d < HD; d++) {
                o[(size_t)i * DIM + h * HD + d] += s[j] * v[d];
            }
        }
    }
}

static void attend_long(struct separator * self, float * o, const float * qkv,
                        int32_t len, int32_t h, float * s) {
    const struct gemm_args scores = {
        .a = qkv + h * HD, .b = qkv + DIM + h * HD, .c = s, .m = len,
        .n = len, .k = HD, .lda = 3 * DIM, .ldb = 3 * DIM, .ldc = len,
        .tb = true
    };
    const struct gemm_args mixed = {
        .a = s, .b = qkv + 2 * DIM + h * HD, .c = o + h * HD, .m = len,
        .n = HD, .k = len, .lda = len, .ldb = 3 * DIM, .ldc = DIM
    };
    memset(s, 0, (size_t)len * len * sizeof(float));
    gemm(self, &scores);
    for (int32_t i = 0; i < len; i++) {
        soften(s + (size_t)i * len, len, 1.0f / sqrtf((float)HD));
    }
    gemm(self, &mixed);
}

static void self_attention(struct separator * self, float * x,
                           const struct attn * a, int32_t seqs, int32_t len,
                           bool causal) {
    const size_t rows = (size_t)seqs * len;
    float * h = floats(rows * DIM);
    float * qkv = floats(rows * 3 * DIM);
    float * o = floats(rows * DIM);
    float * s = floats(len > SHORT ? (size_t)len * len : 1);
    rms(h, x, a->norm, (int32_t)rows);
    linear(self, qkv, h, a->qkv, NULL, (int32_t)rows, 3 * DIM, DIM);
    if (a->freqs != NULL) {
        rope(qkv, a->freqs, (int32_t)rows, len);
    }
    for (int32_t q = 0; q < seqs * HEADS; q++) {
        const size_t at = (size_t)(q / HEADS) * len;
        if (len > SHORT) {
            attend_long(self, o + at * DIM, qkv + at * 3 * DIM, len,
                        q % HEADS, s);
        } else {
            attend_short(o + at * DIM, qkv + at * 3 * DIM, len, q % HEADS,
                         causal);
        }
    }
    linear(self, h, o, a->out, NULL, (int32_t)rows, DIM, DIM);
    for (size_t i = 0; i < rows * DIM; i++) {
        x[i] += h[i];
    }
    free(h);
    free(qkv);
    free(o);
    free(s);
}

static void convffn(struct separator * self, float * x,
                    const struct convffn * f, int32_t seqs, int32_t len) {
    const size_t rows = (size_t)seqs * len;
    float * h = floats(rows * DIM);
    float * a = floats(rows * 2 * HIDDEN);
    float * g = floats(rows * HIDDEN);
    rms(h, x, f->norm, (int32_t)rows);
    conv(self, a, h, &f->up, seqs, len);
    swiglu(g, a, (int32_t)rows, HIDDEN);
    conv(self, h, g, &f->down, seqs, len);
    for (size_t i = 0; i < rows * DIM; i++) {
        x[i] += 0.5f * h[i];
    }
    free(h);
    free(a);
    free(g);
}

static void ffn(struct separator * self, float * x, const struct ffn * f,
                int32_t rows) {
    float * h = floats((size_t)rows * DIM);
    float * a = floats((size_t)rows * 2 * WIDE);
    float * g = floats((size_t)rows * WIDE);
    rms(h, x, f->norm, rows);
    linear(self, a, h, f->up, f->up_bias, rows, 2 * WIDE, DIM);
    swiglu(g, a, rows, WIDE);
    linear(self, h, g, f->down, f->down_bias, rows, DIM, WIDE);
    for (size_t i = 0; i < (size_t)rows * DIM; i++) {
        x[i] += h[i];
    }
    free(h);
    free(a);
    free(g);
}

static void trans(struct separator * self, float * x, const struct trans * t,
                  int32_t seqs, int32_t len) {
    convffn(self, x, &t->ffn1, seqs, len);
    self_attention(self, x, &t->sa, seqs, len, false);
    convffn(self, x, &t->ffn2, seqs, len);
}

static void transpose(float * y, const float * x, int32_t rows,
                      int32_t cols) {
    for (int32_t r = 0; r < rows; r++) {
        for (int32_t c = 0; c < cols; c++) {
            memcpy(y + ((size_t)c * rows + r) * DIM,
                   x + ((size_t)r * cols + c) * DIM, DIM * sizeof(float));
        }
    }
}

static void tf_block(struct separator * self, float * x,
                     const struct trans * b, int32_t frames) {
    float * y = floats((size_t)frames * BINS * DIM);
    trans(self, x, &b[0], frames, BINS);
    transpose(y, x, frames, BINS);
    trans(self, y, &b[1], BINS, frames);
    transpose(x, y, BINS, frames);
    free(y);
}

static void stft(struct separator * self, float * spec, const float * x,
                 int32_t frames) {
    float * cut = floats((size_t)frames * WINDOW);
    const struct gemm_args g = {
        .a = cut, .b = self->basis, .c = spec, .m = frames, .n = 2 * BINS,
        .k = WINDOW, .lda = WINDOW, .ldb = WINDOW, .ldc = 2 * BINS,
        .tb = true
    };
    for (int32_t t = 0; t < frames; t++) {
        memcpy(cut + (size_t)t * WINDOW, x + (size_t)t * HOP,
               WINDOW * sizeof(float));
    }
    memset(spec, 0, (size_t)frames * 2 * BINS * sizeof(float));
    gemm(self, &g);
    free(cut);
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

static void encode(struct separator * self, float * x, const float * spec,
                   int32_t frames) {
    const size_t rows = (size_t)frames * BINS;
    float * corr = floats(rows * CORR);
    float * wide = floats(rows * EMBED);
    float * wave = floats(DIM);
    correlate(corr, spec, frames);
    conv(self, wide, corr, &self->embed1, frames, BINS);
    for (size_t i = 0; i < rows * EMBED; i++) {
        wide[i] = silu(wide[i]);
    }
    conv(self, x, wide, &self->embed2, frames, BINS);
    rms(x, x, self->embed_norm, (int32_t)rows);
    for (size_t r = 0; r < rows; r++) {
        sinusoid(wave, (float)(r % BINS), DIM);
        for (int32_t c = 0; c < DIM; c++) {
            x[r * DIM + c] += wave[c];
        }
    }
    for (int32_t b = 0; b < ENCODERS; b++) {
        tf_block(self, x, self->blocks[b], frames);
    }
    free(corr);
    free(wide);
    free(wave);
}

static void cross(struct separator * self, float * q, const float * kv,
                  int32_t rows, const struct cross * c) {
    float * h = floats(QUERIES * DIM);
    float * qq = floats(QUERIES * DIM);
    float * pairs = floats((size_t)rows * 2 * DIM);
    float * o = floats(QUERIES * DIM);
    float * s = floats((size_t)rows);
    rms(h, q, c->norm, QUERIES);
    linear(self, qq, h, c->q, NULL, QUERIES, DIM, DIM);
    linear(self, pairs, kv, c->kv, NULL, rows, 2 * DIM, DIM);
    for (int32_t i = 0; i < QUERIES * HEADS; i++) {
        const float * a = qq + (i / HEADS) * DIM + (i % HEADS) * HD;
        float * out = o + (i / HEADS) * DIM + (i % HEADS) * HD;
        for (int32_t j = 0; j < rows; j++) {
            const float * k = pairs + (size_t)j * 2 * DIM + (i % HEADS) * HD;
            s[j] = 0.0f;
            for (int32_t d = 0; d < HD; d++) {
                s[j] += a[d] * k[d];
            }
        }
        soften(s, rows, 1.0f / sqrtf((float)HD));
        for (int32_t j = 0; j < rows; j++) {
            const float * v = pairs + (size_t)j * 2 * DIM + DIM +
                              (i % HEADS) * HD;
            for (int32_t d = 0; d < HD; d++) {
                out[d] += s[j] * v[d];
            }
        }
    }
    linear(self, h, o, c->out, NULL, QUERIES, DIM, DIM);
    for (int32_t i = 0; i < QUERIES * DIM; i++) {
        q[i] += h[i];
    }
    free(h);
    free(qq);
    free(pairs);
    free(o);
    free(s);
}

static int32_t attract(struct separator * self, float * att, const float * x,
                       int32_t frames) {
    const int32_t rows = frames * BINS;
    float * kv = floats((size_t)rows * DIM);
    int32_t count = 0;
    layer(kv, x, self->split_norm, self->split_bias, rows);
    for (int32_t r = 0; r < rows; r++) {
        float wave[DIM];
        sinusoid(wave, (float)(r / BINS), DIM / 2);
        sinusoid(wave + DIM / 2, (float)(r % BINS), DIM / 2);
        for (int32_t c = 0; c < DIM; c++) {
            kv[(size_t)r * DIM + c] += wave[c];
        }
    }
    memcpy(att, self->query, QUERIES * DIM * sizeof(float));
    for (int32_t d = 0; d < 2; d++) {
        cross(self, att, kv, rows, &self->dec[d].ca);
        self_attention(self, att, &self->dec[d].sa, 1, QUERIES, true);
        ffn(self, att, &self->dec[d].ffn, QUERIES);
    }
    for (int32_t i = 0; i < QUERIES; i++) {
        float * a = att + i * DIM;
        float norm = 0.0f;
        float logit = self->pres_b[0];
        for (int32_t c = 0; c < DIM; c++) {
            norm += a[c] * a[c];
        }
        for (int32_t c = 0; c < DIM; c++) {
            a[c] /= sqrtf(norm) + 1e-8f;
            logit += a[c] * self->pres_w[c];
        }
        count += i > 0 && 1.0f / (1.0f + expf(-logit)) > 0.5f ? 1 : 0;
    }
    free(kv);
    return count;
}

static void split(struct separator * self, float * y, const float * shared,
                  const float * att, int32_t rows) {
    float * h = floats((size_t)rows * WIDE);
    float a[WIDE];
    for (int32_t o = 0; o < WIDE; o++) {
        a[o] = self->fuse_b1[o];
        for (int32_t c = 0; c < DIM; c++) {
            a[o] += self->fuse_w1[(size_t)o * 2 * DIM + DIM + c] * att[c];
        }
    }
    for (size_t i = 0; i < (size_t)rows * WIDE; i++) {
        h[i] = silu(shared[i] + a[i % WIDE]);
    }
    linear(self, y, h, self->fuse_w2, self->fuse_b2, rows, DIM, WIDE);
    layer(y, y, self->last_norm, self->last_bias, rows);
    free(h);
}

static void speakers(struct separator * self, float ** xs, int32_t n,
                     int32_t rows, int32_t i) {
    float * b = floats((size_t)rows * n * DIM);
    for (int32_t r = 0; r < rows; r++) {
        for (int32_t k = 0; k < n; k++) {
            memcpy(b + ((size_t)r * n + k) * DIM, xs[k] + (size_t)r * DIM,
                   DIM * sizeof(float));
        }
    }
    self_attention(self, b, &self->cs_sa[i], rows, n, false);
    ffn(self, b, &self->cs_ffn[i], rows * n);
    for (int32_t r = 0; r < rows; r++) {
        for (int32_t k = 0; k < n; k++) {
            memcpy(xs[k] + (size_t)r * DIM, b + ((size_t)r * n + k) * DIM,
                   DIM * sizeof(float));
        }
    }
    free(b);
}

static void voice(struct separator * self, float * wave, const float * x,
                  const float * spec, int32_t frames) {
    const int32_t rows = frames * BINS;
    float * m = floats((size_t)rows * OUTS);
    float * mixed = floats((size_t)frames * 2 * BINS);
    float * cut = floats((size_t)frames * WINDOW);
    const struct gemm_args g = {
        .a = mixed, .b = self->basis, .c = cut, .m = frames, .n = WINDOW,
        .k = 2 * BINS, .lda = 2 * BINS, .ldb = WINDOW, .ldc = WINDOW
    };
    linear(self, m, x, self->mask_w, self->mask_b, rows, OUTS, DIM);
    for (int32_t r = 0; r < rows; r++) {
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
    gemm(self, &g);
    for (int32_t t = 0; t < frames; t++) {
        for (int32_t n = 0; n < WINDOW; n++) {
            wave[t * HOP + n] += cut[(size_t)t * WINDOW + n];
        }
    }
    free(m);
    free(mixed);
    free(cut);
}

static void decode(struct separator * self, float ** xs, int32_t n,
                   int32_t frames) {
    for (int32_t i = 0; i < DECODERS; i++) {
        for (int32_t k = 0; k < n; k++) {
            tf_block(self, xs[k], self->blocks[ENCODERS + i], frames);
        }
        speakers(self, xs, n, frames * BINS, i);
    }
}

static int32_t separate8(struct separator * self, const float * x,
                         int32_t count, float ** out, int32_t capacity) {
    const int32_t frames = count >= WINDOW ? (count - WINDOW) / HOP + 1 : 0;
    const int32_t rows = frames * BINS;
    float * spec = floats((size_t)frames * 2 * BINS);
    float * enc = floats((size_t)rows * DIM);
    float * shared = floats((size_t)rows * WIDE);
    float att[QUERIES * DIM];
    float * xs[QUERIES] = { 0 };
    int32_t n = 0;
    const struct gemm_args g = {
        .a = enc, .b = self->fuse_w1, .c = shared, .m = rows, .n = WIDE,
        .k = DIM, .lda = DIM, .ldb = 2 * DIM, .ldc = WIDE, .tb = true
    };
    if (frames > 0) {
        stft(self, spec, x, frames);
        encode(self, enc, spec, frames);
        n = attract(self, att, enc, frames);
        n = n < capacity ? n : capacity;
        gemm(self, &g);
    }
    for (int32_t k = 0; k < n; k++) {
        xs[k] = floats((size_t)rows * DIM);
        split(self, xs[k], shared, att + (k + 1) * DIM, rows);
    }
    if (n > 0) {
        decode(self, xs, n, frames);
    }
    for (int32_t k = 0; k < n; k++) {
        voice(self, out[k], xs[k], spec, frames);
        free(xs[k]);
    }
    free(spec);
    free(enc);
    free(shared);
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
        free(self->owned[i]);
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
        weights_load(self);
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
    return ss_program("ss", (const char * const *)argv + 1);
}

#endif
