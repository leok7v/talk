#include "asr.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { DIM = 512, HEADS = 8, GROUPS = 2, QK = 48, VD = 64, LANES = 4,
       LAYERS = 8, MELS = 80, BINS = 257, FFT = 512, WIN = 400, HOP = 160,
       STEM = 128, KERNEL = 9, TAPS = 3, RANK = 8, HA = 16, HB = 32,
       GROUP = 128, SLOTS = 18432, TABLES = 4, SUB = 128, SITES = 2,
       ETAPS = 4, DILATION = 3, TEXT = 8192, VOCAB = 8199, TOKENS = 320,
       FRAMES = 376, SAMPLES = 480000, EOS = 1, BOS = 2, GATES = 20,
       SEARCH = 80000, SLICE = 1600, STRIDE = 1280, BEAMS = 8,
       NODES = 8 * 321 + 1 };

static const float steady = 0.7f;

enum { T_EMBED = 0, T_DECODER = 1, DECODER_STRIDE = 27, T_DECODER_LANES = 217,
       T_P1 = 226, T_P2 = 227, T_ENGRAM = 228, T_FINAL = 236,
       T_MANIFEST = 237, T_CROSS = 238, CROSS_STRIDE = 10, T_PE_GATE = 318,
       T_ENCODER = 319, ENCODER_STRIDE = 43, T_ENCODER_LANES = 663,
       T_ENCODER_FINAL = 672, T_STEM = 673, T_MEL = 679, T_TOKENIZER = 680,
       TENSORS = 681 };

enum { HEADER = 196, RECORD = 44, FP16 = 1, FP32 = 2, CQ = 3, RAW = 4,
       TAG = 0x05E12A84, PIECE_NORMAL = 0, PIECE_BYTE = 4 };

struct attention {
    int32_t q;
    int32_t k;
    int32_t v;
    int32_t gate;
    int32_t out;
    const float * q_norm;
    const float * k_norm;
};

struct hadamard {
    const float * d1;
    const float * d2;
    const float * b2;
    const float * d3;
    const float * d4;
    const float * w1a;
    const float * w1b;
    const float * w2a;
    const float * w2b;
    const float * w3a;
    const float * w3b;
    const float * cond_v;
    const float * cond_u;
};

struct lanes {
    const float * a_pre;
    const float * a_post;
    const float * a_res;
    const float * b_pre;
    const float * b_post;
    const float * b_res;
    int32_t phi;
};

struct node {
    int32_t token;
    float raw_q[LAYERS][HEADS * QK];
    float raw_k[LAYERS][GROUPS * QK];
    float raw_v[LAYERS][GROUPS * VD];
    float key[LAYERS][GROUPS * QK];
    float value[LAYERS][GROUPS * VD];
    float engram[SITES][DIM];
    int32_t frame;
    float gain;
};

struct beam {
    int32_t path[TOKENS + 1];
    int32_t length;
    float score;
};

struct candidate {
    int32_t beam;
    int32_t token;
    float score;
};

struct result {
    int32_t token[TOKENS + 1];
    int32_t frame[TOKENS + 2];
    float gain[TOKENS + 1];
    int32_t length;
    float score;
};

struct output {
    char * text;
    int32_t size;
    int32_t capacity;
    int32_t steps;
    struct pieces * pieces;
    const float * first;
    void (* said)(void * who, const char * text, int32_t size);
    void * who;
    int32_t base;
    int32_t written;
    bool more;
};

struct memory {
    float * k;
    float * v;
    int32_t length;
    float spread;
};

struct whistle {
    struct asr asr;
    struct backend * backend;
    bool packed;
    float books[28];
    const uint8_t * file;
    size_t size;
    float * tensors[TENSORS];
    int32_t p1[DIM];
    int32_t p2[DIM];
    const uint8_t * piece[VOCAB];
    int32_t piece_size[VOCAB];
    int32_t piece_type[VOCAB];
    float rope_rate[QK / 2];
    struct node * nodes[NODES];
    int32_t node_count;
    float * view_k;
    float * view_v;
    int32_t beams;
    int32_t depth;
    int32_t layers[LAYERS];
};

static const float none[LANES * DIM];

static float single(const uint8_t * p) {
    const uint32_t bits = le32(p);
    float v = 0.0f;
    memcpy(&v, &bits, sizeof v);
    return v;
}

static void group_decode(struct whistle * self, float * out,
                         const uint8_t * row, int32_t at, int32_t bits,
                         float norm, int32_t count) {
    const float * book = self->books + (bits == 2 ? 0 : 12);
    const float scale = 1.0f / sqrtf((float)GROUP);
    float v[GROUP];
    for (int32_t k = 0; k < GROUP; k++) {
        const int32_t bit = (at * GROUP + k) * bits;
        const int32_t index = row[bit >> 3] >> (bit & 7) & ((1 << bits) - 1);
        v[k] = book[index] * norm;
    }
    hadamard128(v);
    for (int32_t k = 0; k < count; k++) {
        out[k] = v[k] * scale;
    }
}

static void matrix_decode(struct whistle * self, float * out,
                          const uint8_t * blob, int32_t rows, int32_t columns,
                          int32_t bits) {
    const int32_t groups = (columns + GROUP - 1) / GROUP;
    const size_t row_bytes = (size_t)(groups * GROUP * bits / 8);
    const uint8_t * norms = blob + (size_t)rows * row_bytes;
    for (int32_t r = 0; r < rows; r++) {
        for (int32_t g = 0; g < groups; g++) {
            const int32_t left = columns - g * GROUP;
            group_decode(self, out + (size_t)r * columns + g * GROUP,
                         blob + (size_t)r * row_bytes, g, bits,
                         half_float(norms + ((size_t)r * groups + g) * 2),
                         left < GROUP ? left : GROUP);
        }
    }
}

static const uint8_t * record_at(struct whistle * self, int32_t index) {
    return self->file + HEADER + 28 * 4 + (size_t)index * RECORD;
}

static void row_decode(struct whistle * self, float * out, int32_t index,
                       int32_t row) {
    const uint8_t * record = record_at(self, index);
    const int32_t rows = (int32_t)le32(record + 4);
    const int32_t columns = (int32_t)le32(record + 8);
    const int32_t bits = (int32_t)le32(record + 40);
    const int32_t groups = (columns + GROUP - 1) / GROUP;
    const size_t row_bytes = (size_t)(groups * GROUP * bits / 8);
    const uint8_t * blob = self->file + le64(record + 20);
    const uint8_t * norms = blob + (size_t)rows * row_bytes;
    for (int32_t g = 0; g < groups; g++) {
        const int32_t left = columns - g * GROUP;
        group_decode(self, out + g * GROUP, blob + (size_t)row * row_bytes, g,
                     bits, half_float(norms + ((size_t)row * groups + g) * 2),
                     left < GROUP ? left : GROUP);
    }
}

static bool lazy(struct whistle * self, int32_t index) {
    const bool table = index == T_ENGRAM || index == T_ENGRAM + 4;
    const bool block = index >= T_ENCODER && index < T_ENCODER_LANES;
    return table ||
           ((block || self->packed) && record_at(self, index)[0] == CQ);
}

static void tensor_load(struct whistle * self, int32_t index) {
    const uint8_t * record = record_at(self, index);
    const int32_t kind = record[0];
    const uint8_t * blob = self->file + le64(record + 20);
    size_t count = 1;
    for (int32_t d = 0; d < record[1]; d++) {
        count *= le32(record + 4 + d * 4);
    }
    if (kind != RAW) {
        self->tensors[index] = floats(count);
    }
    if (kind == FP16) {
        for (size_t i = 0; i < count; i++) {
            self->tensors[index][i] = half_float(blob + i * 2);
        }
    } else if (kind == FP32) {
        for (size_t i = 0; i < count; i++) {
            self->tensors[index][i] = single(blob + i * 4);
        }
    } else if (kind == CQ) {
        matrix_decode(self, self->tensors[index], blob,
                      (int32_t)le32(record + 4), (int32_t)le32(record + 8),
                      (int32_t)le32(record + 40));
    }
}

static void layer_load(struct whistle * self, int32_t l, bool wanted) {
    const int32_t base = T_ENCODER + l * ENCODER_STRIDE;
    for (int32_t i = base; i < base + ENCODER_STRIDE; i++) {
        if (lazy(self, i) && wanted) {
            tensor_load(self, i);
        } else if (lazy(self, i)) {
            free(self->tensors[i]);
            self->tensors[i] = NULL;
        }
    }
}

static bool pieces_load(struct whistle * self) {
    const uint8_t * record = record_at(self, T_TOKENIZER);
    const uint8_t * p = self->file + le64(record + 20);
    const bool ok = record[0] == RAW && le32(p) == VOCAB;
    p += 24;
    for (int32_t i = 0; ok && i < VOCAB; i++) {
        self->piece_type[i] = p[4];
        self->piece_size[i] = (int32_t)le16(p + 5);
        self->piece[i] = p + 7;
        p += 7 + self->piece_size[i];
    }
    return ok;
}

static void linear(struct whistle * self, float * y, const float * x,
                   const float * w, int32_t rows, int32_t out, int32_t in) {
    const struct gemm_args g = {
        .a = x, .b = w, .c = y, .m = rows, .n = out, .k = in, .lda = in,
        .ldb = in, .ldc = out, .tb = true
    };
    memset(y, 0, (size_t)rows * out * sizeof(float));
    self->backend->gemm(self->backend, &g);
}

static void product(struct whistle * self, float * y, const float * x,
                    int32_t index, int32_t first, int32_t out, int32_t rows) {
    const uint8_t * record = record_at(self, index);
    const int32_t in = (int32_t)le32(record + 8);
    const int32_t bits = (int32_t)le32(record + 40);
    const int32_t groups = (in + GROUP - 1) / GROUP;
    const size_t row_bytes = (size_t)(groups * GROUP * bits / 8);
    const uint8_t * blob = self->file + le64(record + 20);
    const struct packed w = {
        .codes = blob + (size_t)first * row_bytes,
        .norms = blob + (size_t)le32(record + 4) * row_bytes +
                 (size_t)first * groups * 2,
        .book = self->books + (bits == 2 ? 0 : 12), .bits = bits, .out = out,
        .in = in
    };
    if (self->tensors[index] != NULL) {
        linear(self, y, x, self->tensors[index] + (size_t)first * in, rows,
               out, in);
    } else {
        self->backend->packed(self->backend, &w, x, y, rows);
    }
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static float silu(float x) {
    return x * sigmoid(x);
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

static void norm(float * y, const float * x, const float * scale,
                 int32_t rows, int32_t n) {
    for (int32_t r = 0; r < rows; r++) {
        const float * in = x + (size_t)r * n;
        float * out = y + (size_t)r * n;
        const float gain = 1.0f / sqrtf(dot(in, in, n) / (float)n + 1e-6f);
        for (int32_t i = 0; i < n; i++) {
            out[i] = (1.0f + scale[i]) * in[i] * gain;
        }
    }
}

static void rope(struct whistle * self, float * x, int32_t heads, int32_t at) {
    float c[QK / 2];
    float s[QK / 2];
    for (int32_t i = 0; i < QK / 2; i++) {
        c[i] = cosf((float)at * self->rope_rate[i]);
        s[i] = sinf((float)at * self->rope_rate[i]);
    }
    for (int32_t h = 0; h < heads; h++) {
        float * p = x + h * QK;
        for (int32_t i = 0; i < QK / 2; i++) {
            const float a = p[i];
            const float b = p[i + QK / 2];
            p[i] = a * c[i] - b * s[i];
            p[i + QK / 2] = b * c[i] + a * s[i];
        }
    }
}

static void attend(float * out, const float * q, const float * k,
                   const float * v, int32_t count, int32_t groups) {
    float score[FRAMES];
    for (int32_t h = 0; h < HEADS; h++) {
        const int32_t g = h * groups / HEADS;
        float * o = out + h * VD;
        float top = -INFINITY;
        float sum = 0.0f;
        for (int32_t s = 0; s < count; s++) {
            score[s] = dot(q + h * QK, k + ((size_t)s * groups + g) * QK,
                           QK) / sqrtf((float)QK);
            top = fmaxf(top, score[s]);
        }
        for (int32_t s = 0; s < count; s++) {
            score[s] = expf(score[s] - top);
            sum += score[s];
        }
        memset(o, 0, VD * sizeof(float));
        for (int32_t s = 0; s < count; s++) {
            const float * from = v + ((size_t)s * groups + g) * VD;
            const float w = score[s] / sum;
            for (int32_t d = 0; d < VD; d++) {
                o[d] += w * from[d];
            }
        }
    }
}

static void kron(float * out, const float * z, const float * a,
                 const float * b) {
    float t[HA * HB];
    for (int32_t i = 0; i < HA; i++) {
        for (int32_t l = 0; l < HB; l++) {
            float sum = 0.0f;
            for (int32_t j = 0; j < HB; j++) {
                sum += z[i * HB + j] * b[j * HB + l];
            }
            t[i * HB + l] = sum;
        }
    }
    for (int32_t k = 0; k < HA; k++) {
        for (int32_t l = 0; l < HB; l++) {
            float sum = 0.0f;
            for (int32_t i = 0; i < HA; i++) {
                sum += a[i * HA + k] * t[i * HB + l];
            }
            out[k * HB + l] = sum;
        }
    }
}

static void condition(float * cond, const float * x,
                      const struct hadamard * h) {
    float soft[RANK] = { 0 };
    float top = -INFINITY;
    float sum = 0.0f;
    for (int32_t c = 0; c < DIM; c++) {
        for (int32_t j = 0; j < RANK; j++) {
            soft[j] += x[c] * h->cond_v[c * RANK + j];
        }
    }
    for (int32_t j = 0; j < RANK; j++) {
        top = fmaxf(top, soft[j]);
    }
    for (int32_t j = 0; j < RANK; j++) {
        soft[j] = expf(soft[j] - top);
        sum += soft[j];
    }
    for (int32_t i = 0; i < DIM; i++) {
        cond[i] = 1.0f;
    }
    for (int32_t j = 0; j < RANK; j++) {
        for (int32_t i = 0; i < DIM; i++) {
            cond[i] += soft[j] / sum * h->cond_u[j * DIM + i];
        }
    }
}

static void hadamard_row(struct whistle * self, float * out, const float * x,
                         const struct hadamard * h) {
    float cond[DIM];
    float z[DIM];
    float t[DIM];
    condition(cond, x, h);
    for (int32_t i = 0; i < DIM; i++) {
        z[i] = h->d1[i] * x[i];
    }
    kron(t, z, h->w1a, h->w1b);
    for (int32_t i = 0; i < DIM; i++) {
        z[i] = silu(h->d2[i] * cond[i] * t[self->p1[i]] + h->b2[i]);
    }
    kron(t, z, h->w2a, h->w2b);
    for (int32_t i = 0; i < DIM; i++) {
        z[i] = h->d3[i] * t[self->p2[i]];
    }
    kron(t, z, h->w3a, h->w3b);
    for (int32_t i = 0; i < DIM; i++) {
        out[i] = h->d4[i] * t[i];
    }
}

static void hadamard_add(struct whistle * self, float * x, const float * scale,
                         const struct hadamard * h, int32_t rows,
                         float share) {
    float in[DIM];
    float out[DIM];
    for (int32_t r = 0; r < rows; r++) {
        float * row = x + (size_t)r * DIM;
        norm(in, row, scale, 1, DIM);
        hadamard_row(self, out, in, h);
        for (int32_t i = 0; i < DIM; i++) {
            row[i] += share * out[i];
        }
    }
}

static struct hadamard hadamard_at(struct whistle * self, int32_t base) {
    return (struct hadamard){
        .d1 = self->tensors[base], .d2 = self->tensors[base + 1],
        .b2 = self->tensors[base + 2], .d3 = self->tensors[base + 3],
        .d4 = self->tensors[base + 4], .w1a = self->tensors[base + 5],
        .w1b = self->tensors[base + 6], .w2a = self->tensors[base + 7],
        .w2b = self->tensors[base + 8], .w3a = self->tensors[base + 9],
        .w3b = self->tensors[base + 10], .cond_v = self->tensors[base + 11],
        .cond_u = self->tensors[base + 12]
    };
}

static struct lanes lanes_at(struct whistle * self, int32_t base) {
    return (struct lanes){
        .a_pre = self->tensors[base], .a_post = self->tensors[base + 1],
        .a_res = self->tensors[base + 2], .b_pre = self->tensors[base + 3],
        .b_post = self->tensors[base + 4], .b_res = self->tensors[base + 5],
        .phi = base + 6
    };
}

static float log_sum(const float * v, int32_t step) {
    float top = -INFINITY;
    float sum = 0.0f;
    for (int32_t i = 0; i < LANES; i++) {
        top = fmaxf(top, v[i * step]);
    }
    for (int32_t i = 0; i < LANES; i++) {
        sum += expf(v[i * step] - top);
    }
    return top + logf(sum);
}

static void sinkhorn(float * k) {
    for (int32_t round = 0; round < 20; round++) {
        for (int32_t i = 0; i < LANES; i++) {
            const float total = log_sum(k + i * LANES, 1);
            for (int32_t j = 0; j < LANES; j++) {
                k[i * LANES + j] -= total;
            }
        }
        for (int32_t j = 0; j < LANES; j++) {
            const float total = log_sum(k + j, LANES);
            for (int32_t i = 0; i < LANES; i++) {
                k[i * LANES + j] -= total;
            }
        }
    }
    for (int32_t i = 0; i < LANES * LANES; i++) {
        k[i] = expf(k[i]);
    }
}

static void lanes_gates(const struct lanes * m, int32_t l, int32_t own,
                        float * u, float * gate, const float * x,
                        const float * mix) {
    memset(u, 0, DIM * sizeof(float));
    for (int32_t j = 0; j < LANES; j++) {
        const float pre = sigmoid(m->a_pre[l] * mix[j] +
                                  m->b_pre[l * LANES + j] +
                                  (j == own ? 4.0f : -4.0f));
        gate[j] = 2.0f * sigmoid(m->a_post[l] * mix[LANES + j] +
                                 m->b_post[l * LANES + j] +
                                 (j == own ? 0.0f : -4.0f));
        for (int32_t i = 0; i < DIM; i++) {
            u[i] += pre * x[j * DIM + i];
        }
    }
    for (int32_t i = 0; i < LANES * LANES; i++) {
        gate[LANES + i] = m->a_res[l] * mix[2 * LANES + i] +
                          m->b_res[l * LANES * LANES + i];
    }
    sinkhorn(gate + LANES);
}

static void lanes_enter(struct whistle * self, const struct lanes * m,
                        int32_t l, int32_t own, const float * stream,
                        float * u, float * gate, int32_t rows) {
    const int32_t wide = LANES * DIM;
    const int32_t mixes = 2 * LANES + LANES * LANES;
    float * unit = floats((size_t)rows * wide);
    float * mix = floats((size_t)rows * mixes);
    float * part = floats((size_t)rows * LANES * LANES);
    norm(unit, stream, none, rows, wide);
    for (int32_t k = 0; k < 3; k++) {
        const int32_t n = k == 2 ? LANES * LANES : LANES;
        product(self, part, unit, m->phi + k, l * n, n, rows);
        for (int32_t r = 0; r < rows; r++) {
            memcpy(mix + (size_t)r * mixes + k * LANES,
                   part + (size_t)r * n, (size_t)n * sizeof(float));
        }
    }
    for (int32_t r = 0; r < rows; r++) {
        lanes_gates(m, l, own, u + (size_t)r * DIM,
                    gate + (size_t)r * GATES, stream + (size_t)r * wide,
                    mix + (size_t)r * mixes);
    }
    free(unit);
    free(mix);
    free(part);
}

static void lanes_leave(float * stream, const float * u, const float * y,
                        const float * gate, int32_t rows) {
    float old[LANES * DIM];
    for (int32_t r = 0; r < rows; r++) {
        float * x = stream + (size_t)r * LANES * DIM;
        const float * g = gate + (size_t)r * GATES;
        memcpy(old, x, sizeof old);
        for (int32_t i = 0; i < LANES; i++) {
            for (int32_t d = 0; d < DIM; d++) {
                x[i * DIM + d] = g[i] * (y[(size_t)r * DIM + d] -
                                         u[(size_t)r * DIM + d]);
            }
            for (int32_t j = 0; j < LANES; j++) {
                const float w = g[LANES + i * LANES + j];
                for (int32_t d = 0; d < DIM; d++) {
                    x[i * DIM + d] += w * old[j * DIM + d];
                }
            }
        }
    }
}

static void lanes_mean(float * y, const float * stream, const float * scale,
                       int32_t rows) {
    for (int32_t r = 0; r < rows; r++) {
        const float * x = stream + (size_t)r * LANES * DIM;
        float * out = y + (size_t)r * DIM;
        for (int32_t d = 0; d < DIM; d++) {
            out[d] = (x[d] + x[DIM + d] + x[2 * DIM + d] + x[3 * DIM + d]) /
                     (float)LANES;
        }
    }
    norm(y, y, scale, rows, DIM);
}

static void gated(struct whistle * self, float * y, const float * mix,
                  const float * x, const struct attention * a, int32_t rows) {
    float * gate = floats((size_t)rows * DIM);
    float * open = floats((size_t)rows * DIM);
    product(self, gate, x, a->gate, 0, DIM, rows);
    for (size_t i = 0; i < (size_t)rows * DIM; i++) {
        open[i] = mix[i] * sigmoid(gate[i]);
    }
    product(self, y, open, a->out, 0, DIM, rows);
    free(gate);
    free(open);
}

static void fft(float * re, float * im) {
    for (int32_t i = 1, j = 0; i < FFT; i++) {
        int32_t bit = FFT >> 1;
        while ((j & bit) != 0) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            const float a = re[i];
            const float b = im[i];
            re[i] = re[j];
            im[i] = im[j];
            re[j] = a;
            im[j] = b;
        }
    }
    for (int32_t size = 2; size <= FFT; size *= 2) {
        const float step = -2.0f * (float)M_PI / (float)size;
        for (int32_t i = 0; i < FFT; i += size) {
            for (int32_t k = 0; k < size / 2; k++) {
                const float c = cosf(step * (float)k);
                const float s = sinf(step * (float)k);
                const int32_t a = i + k;
                const int32_t b = a + size / 2;
                const float x = re[b] * c - im[b] * s;
                const float y = re[b] * s + im[b] * c;
                re[b] = re[a] - x;
                im[b] = im[a] - y;
                re[a] += x;
                im[a] += y;
            }
        }
    }
}

static float spectrum(struct whistle * self, float * mel, const float * wave,
                      int32_t count, int32_t frame) {
    const float * bank = self->tensors[T_MEL];
    float energy = 0.0f;
    float re[FFT] = { 0 };
    float im[FFT] = { 0 };
    for (int32_t n = 0; n < WIN; n++) {
        const int32_t at = frame * HOP - WIN / 2 + n;
        const float window = 0.5f - 0.5f * cosf(2.0f * (float)M_PI *
                                                (float)n / (float)(WIN - 1));
        re[n] = at >= 0 && at < count ? wave[at] * window : 0.0f;
    }
    fft(re, im);
    for (int32_t m = 0; m < MELS; m++) {
        mel[m] = 0.0f;
    }
    for (int32_t b = 0; b < BINS; b++) {
        const float power = re[b] * re[b] + im[b] * im[b];
        energy += power;
        for (int32_t m = 0; m < MELS; m++) {
            mel[m] += power * bank[b * MELS + m];
        }
    }
    for (int32_t m = 0; m < MELS; m++) {
        mel[m] = logf(mel[m] + 0x1p-24f);
    }
    return logf(energy + 0x1p-24f);
}

static float deviation(const float * v, int32_t count) {
    float mean = 0.0f;
    float spread = 0.0f;
    for (int32_t i = 0; i < count; i++) {
        mean += v[i];
    }
    mean /= (float)(count > 1 ? count : 1);
    for (int32_t i = 0; i < count; i++) {
        spread += (v[i] - mean) * (v[i] - mean);
    }
    return sqrtf(spread / (float)(count > 2 ? count - 1 : 1));
}

static float features(struct whistle * self, float * x, const float * wave,
                      int32_t count, int32_t rows, int32_t length) {
    float * energy = floats((size_t)length + 1);
    for (int32_t f = 0; f < length; f++) {
        energy[f] = spectrum(self, x + (size_t)f * MELS, wave, count, f);
    }
    const float spread = deviation(energy, length);
    free(energy);
    for (int32_t m = 0; m < MELS; m++) {
        float mean = 0.0f;
        float wide = 0.0f;
        for (int32_t f = 0; f < length; f++) {
            mean += x[(size_t)f * MELS + m];
        }
        mean /= (float)(length > 1 ? length : 1);
        for (int32_t f = 0; f < length; f++) {
            const float d = x[(size_t)f * MELS + m] - mean;
            wide += d * d;
        }
        wide = sqrtf(wide / (float)(length > 2 ? length - 1 : 1));
        for (int32_t f = 0; f < rows; f++) {
            const float v = x[(size_t)f * MELS + m];
            x[(size_t)f * MELS + m] = f < length ?
                (v - mean) / (wide + 1e-5f) : 0.0f;
        }
    }
    return spread;
}

static float tap(const float * x, int32_t rows, int32_t columns,
                 int32_t depth, int32_t r, int32_t c, int32_t d) {
    return r >= 0 && r < rows && c >= 0 && c < columns ?
           x[((size_t)r * columns + c) * depth + d] : 0.0f;
}

static void stem_first(struct whistle * self, float * y, const float * x,
                       int32_t rows, int32_t length) {
    const float * w = self->tensors[T_STEM];
    for (int32_t t = 0; t < length; t++) {
        for (int32_t f = 0; f < MELS / 2; f++) {
            float * out = y + ((size_t)t * (MELS / 2) + f) * STEM;
            for (int32_t k = 0; k < 9; k++) {
                const float v = tap(x, rows, MELS, 1, 2 * t + k / 3 - 1,
                                    2 * f + k % 3 - 1, 0);
                for (int32_t c = 0; c < STEM; c++) {
                    out[c] += w[k * STEM + c] * v;
                }
            }
        }
    }
}

static void stem_down(struct whistle * self, float * y, const float * x,
                      const float * w, int32_t pw, int32_t rows,
                      int32_t columns, int32_t length) {
    float * mid = floats((size_t)length * (columns / 2) * STEM);
    for (int32_t t = 0; t < length; t++) {
        for (int32_t f = 0; f < columns / 2; f++) {
            float * out = mid + ((size_t)t * (columns / 2) + f) * STEM;
            for (int32_t k = 0; k < 9; k++) {
                for (int32_t c = 0; c < STEM; c++) {
                    out[c] += w[k * STEM + c] *
                              silu(tap(x, rows, columns, STEM,
                                       2 * t + k / 3 - 1, 2 * f + k % 3 - 1,
                                       c));
                }
            }
        }
    }
    product(self, y, mid, pw, 0, STEM, length * (columns / 2));
    free(mid);
}

static int32_t stem(struct whistle * self, float * y, const float * mel,
                    int32_t rows, int32_t length) {
    const int32_t r1 = (rows - 1) / 2 + 1;
    const int32_t r2 = (r1 - 1) / 2 + 1;
    const int32_t r3 = (r2 - 1) / 2 + 1;
    const int32_t l1 = (length + 1) / 2;
    const int32_t l2 = (l1 + 1) / 2;
    const int32_t l3 = (l2 + 1) / 2;
    float * a = floats((size_t)r1 * 40 * STEM);
    float * b = floats((size_t)r2 * 20 * STEM);
    float * c = floats((size_t)r3 * 10 * STEM);
    float * flat = floats((size_t)r3 * 10 * STEM);
    stem_first(self, a, mel, rows, l1);
    stem_down(self, b, a, self->tensors[T_STEM + 1], T_STEM + 2, r1, 40, l2);
    stem_down(self, c, b, self->tensors[T_STEM + 3], T_STEM + 4, r2, 20, l3);
    for (int32_t t = 0; t < l3; t++) {
        for (int32_t f = 0; f < 10; f++) {
            for (int32_t d = 0; d < STEM; d++) {
                flat[((size_t)t * STEM + d) * 10 + f] =
                    silu(c[((size_t)t * 10 + f) * STEM + d]);
            }
        }
    }
    product(self, y, flat, T_STEM + 5, 0, DIM, l3);
    free(a);
    free(b);
    free(c);
    free(flat);
    return l3;
}

static void encoder_attention(struct whistle * self, float * x, int32_t base,
                              int32_t rows, int32_t length) {
    const struct attention a = {
        .q = base + 6, .k = base + 7, .v = base + 8, .gate = base + 9,
        .out = base + 10, .q_norm = self->tensors[base + 11],
        .k_norm = self->tensors[base + 12]
    };
    float * h = floats((size_t)rows * DIM);
    float * q = floats((size_t)rows * HEADS * QK);
    float * k = floats((size_t)rows * GROUPS * QK);
    float * v = floats((size_t)rows * GROUPS * VD);
    float * mix = floats((size_t)rows * DIM);
    norm(h, x, self->tensors[base + 1], rows, DIM);
    product(self, q, h, a.q, 0, HEADS * QK, rows);
    product(self, k, h, a.k, 0, GROUPS * QK, rows);
    product(self, v, h, a.v, 0, GROUPS * VD, rows);
    norm(q, q, a.q_norm, rows * HEADS, QK);
    norm(k, k, a.k_norm, rows * GROUPS, QK);
    for (int32_t t = 0; t < rows; t++) {
        rope(self, q + (size_t)t * HEADS * QK, HEADS, t);
        rope(self, k + (size_t)t * GROUPS * QK, GROUPS, t);
    }
    for (int32_t t = 0; t < rows; t++) {
        attend(mix + (size_t)t * DIM, q + (size_t)t * HEADS * QK, k, v,
               length, GROUPS);
    }
    gated(self, h, mix, h, &a, rows);
    norm(h, h, self->tensors[base + 2], rows, DIM);
    for (size_t i = 0; i < (size_t)rows * DIM; i++) {
        x[i] += sigmoid(self->tensors[base + 13][0]) * h[i];
    }
    free(h);
    free(q);
    free(k);
    free(v);
    free(mix);
}

static void encoder_conv(struct whistle * self, float * x, int32_t base,
                         int32_t rows, int32_t length) {
    const float * dw = self->tensors[base + 16];
    float * h = floats((size_t)rows * DIM);
    float * wide = floats((size_t)rows * 2 * DIM);
    float * g = floats((size_t)rows * DIM);
    norm(h, x, self->tensors[base + 3], rows, DIM);
    product(self, wide, h, base + 14, 0, 2 * DIM, rows);
    for (int32_t t = 0; t < length; t++) {
        for (int32_t d = 0; d < DIM; d++) {
            g[(size_t)t * DIM + d] = wide[(size_t)t * 2 * DIM + d] *
                sigmoid(wide[(size_t)t * 2 * DIM + DIM + d]);
        }
    }
    memset(h, 0, (size_t)rows * DIM * sizeof(float));
    for (int32_t t = 0; t < rows; t++) {
        for (int32_t k = 0; k < KERNEL; k++) {
            for (int32_t d = 0; d < DIM; d++) {
                h[(size_t)t * DIM + d] += dw[k * DIM + d] *
                    tap(g, rows, 1, DIM, t + k - KERNEL / 2, 0, d);
            }
        }
    }
    norm(h, h, self->tensors[base + 4], rows, DIM);
    for (size_t i = 0; i < (size_t)rows * DIM; i++) {
        h[i] = silu(h[i]);
    }
    product(self, g, h, base + 15, 0, DIM, rows);
    for (size_t i = 0; i < (size_t)rows * DIM; i++) {
        x[i] += g[i];
    }
    free(h);
    free(wide);
    free(g);
}

static void encoder_block(struct whistle * self, float * x, int32_t l,
                          int32_t rows, int32_t length) {
    const int32_t base = T_ENCODER + l * ENCODER_STRIDE;
    const struct hadamard first = hadamard_at(self, base + 17);
    const struct hadamard last = hadamard_at(self, base + 30);
    layer_load(self, l, true);
    hadamard_add(self, x, self->tensors[base], &first, rows, 0.5f);
    encoder_attention(self, x, base, rows, length);
    encoder_conv(self, x, base, rows, length);
    hadamard_add(self, x, self->tensors[base + 5], &last, rows, 0.5f);
    layer_load(self, l, false);
}

static void encoder(struct whistle * self, float * y, const float * x,
                    int32_t rows, int32_t length) {
    const struct lanes m = lanes_at(self, T_ENCODER_LANES);
    float * stream = floats((size_t)rows * LANES * DIM);
    float * u = floats((size_t)rows * DIM);
    float * out = floats((size_t)rows * DIM);
    float * gate = floats((size_t)rows * GATES);
    for (int32_t r = 0; r < rows; r++) {
        for (int32_t j = 0; j < LANES; j++) {
            memcpy(stream + ((size_t)r * LANES + j) * DIM,
                   x + (size_t)r * DIM, DIM * sizeof(float));
        }
    }
    for (int32_t l = 0; l < LAYERS; l++) {
        lanes_enter(self, &m, l, l % LANES, stream, u, gate, rows);
        memcpy(out, u, (size_t)rows * DIM * sizeof(float));
        encoder_block(self, out, l, rows, length);
        lanes_leave(stream, u, out, gate, rows);
    }
    lanes_mean(y, stream, self->tensors[T_ENCODER_FINAL], rows);
    free(stream);
    free(u);
    free(out);
    free(gate);
}

static struct memory memory_open(struct whistle * self, const float * wave,
                                 int32_t count) {
    const int32_t frames = 1 + count / HOP;
    const int32_t all = ((((frames - 1) / 2 + 1) - 1) / 2 + 1 - 1) / 2 + 1;
    const int32_t rows = all + all % 2;
    const float gate = sigmoid(self->tensors[T_PE_GATE][0]);
    float * mel = floats((size_t)frames * MELS);
    float * x = floats((size_t)rows * DIM);
    float * enc = floats((size_t)rows * DIM);
    struct memory memory = {
        .k = floats((size_t)LAYERS * rows * HEADS * QK),
        .v = floats((size_t)LAYERS * rows * HEADS * VD)
    };
    memory.spread = features(self, mel, wave, count, frames, count / HOP);
    memory.length = stem(self, x, mel, frames, count / HOP);
    if (memory.spread >= steady) {
        encoder(self, enc, x, rows, memory.length);
    }
    for (int32_t t = 0; t < rows; t++) {
        for (int32_t i = 0; i < DIM / 2; i++) {
            const float angle = (float)t *
                expf(-logf(10000.0f) / (float)(DIM / 2 - 1) * (float)i);
            enc[(size_t)t * DIM + i] += gate * sinf(angle);
            enc[(size_t)t * DIM + DIM / 2 + i] += gate * cosf(angle);
        }
    }
    for (int32_t i = 0; i < self->depth; i++) {
        const int32_t l = self->layers[i];
        const int32_t base = T_CROSS + l * CROSS_STRIDE;
        float * k = memory.k + (size_t)l * rows * HEADS * QK;
        product(self, k, enc, base + 2, 0, HEADS * QK, rows);
        norm(k, k, self->tensors[base + 7], rows * HEADS, QK);
        product(self, memory.v + (size_t)l * rows * HEADS * VD, enc, base + 3,
                0, HEADS * VD, rows);
    }
    memory.length = memory.length > 0 ? memory.length : 1;
    free(mel);
    free(x);
    free(enc);
    return memory;
}

static int32_t node_new(struct whistle * self, int32_t token, float gain) {
    struct node * node = calloc(1, sizeof *node);
    must(node != NULL && self->node_count < NODES, "memory");
    node->token = token;
    node->gain = gain;
    self->nodes[self->node_count] = node;
    return self->node_count++;
}

static struct node * node_at(const struct whistle * self,
                             const struct beam * b, int32_t position) {
    return self->nodes[b->path[position]];
}

static uint32_t engram_slot(const struct whistle * self,
                            const struct beam * b, int32_t at,
                            int32_t table) {
    const int32_t order = table / 2 + 2;
    uint32_t acc = 0x9E3779B9u * (uint32_t)(table + 1);
    for (int32_t j = 0; j < order; j++) {
        const uint32_t token = at - j >= 0 ?
                               (uint32_t)node_at(self, b, at - j)->token : 0;
        acc = (acc ^ token) * 0x01000193u;
    }
    acc ^= acc >> 15;
    return acc % SLOTS;
}

static void engram(struct whistle * self, const struct beam * b, float * key,
                   float * value) {
    const int32_t at = b->length - 1;
    struct node * now = node_at(self, b, at);
    float e[DIM] = { 0 };
    for (int32_t s = 0; s < SITES; s++) {
        const int32_t base = T_ENGRAM + s * 4;
        for (int32_t table = 0; table < TABLES; table++) {
            const uint32_t row = (uint32_t)table * SLOTS +
                                 engram_slot(self, b, at, table);
            if (at >= table / 2 + 1) {
                row_decode(self, e + table * SUB, base, (int32_t)row);
            }
        }
        product(self, key + s * DIM, e, base + 1, 0, DIM, 1);
        product(self, now->engram[s], e, base + 2, 0, DIM, 1);
        memset(value + s * DIM, 0, DIM * sizeof(float));
        for (int32_t j = 0; j < ETAPS && at - j * DILATION >= 0; j++) {
            const struct node * old = node_at(self, b, at - j * DILATION);
            const float * past = old->engram[s];
            for (int32_t d = 0; d < DIM; d++) {
                value[s * DIM + d] += self->tensors[base + 3][j * DIM + d] *
                                      past[d];
            }
        }
    }
}

static void recall(float * x, const float * key, const float * value) {
    float a[DIM];
    float b[DIM];
    norm(a, x, none, 1, DIM);
    norm(b, key, none, 1, DIM);
    const float alpha = sigmoid(dot(a, b, DIM) / sqrtf((float)DIM));
    for (int32_t d = 0; d < DIM; d++) {
        x[d] += alpha * value[d];
    }
}

static void tapped(float * y, const float * const * past, const float * taps,
                   int32_t width) {
    memset(y, 0, (size_t)width * sizeof(float));
    for (int32_t j = 0; j < TAPS && past[j] != NULL; j++) {
        for (int32_t d = 0; d < width; d++) {
            y[d] += taps[j * width + d] * past[j][d];
        }
    }
}

static void path_view(struct whistle * self, const struct beam * b,
                      int32_t l) {
    for (int32_t p = 0; p < b->length; p++) {
        const struct node * node = node_at(self, b, p);
        memcpy(self->view_k + (size_t)p * GROUPS * QK, node->key[l],
               sizeof node->key[l]);
        memcpy(self->view_v + (size_t)p * GROUPS * VD, node->value[l],
               sizeof node->value[l]);
    }
}

static void attend_own(struct whistle * self, float * mix, int32_t l,
                       const struct beam * b, const float * raw) {
    const int32_t base = T_DECODER + l * DECODER_STRIDE;
    const int32_t at = b->length - 1;
    struct node * now = node_at(self, b, at);
    const float * past_q[TAPS];
    const float * past_k[TAPS];
    const float * past_v[TAPS];
    float q[HEADS * QK];
    memcpy(now->raw_q[l], raw, sizeof now->raw_q[l]);
    memcpy(now->raw_k[l], raw + HEADS * QK, sizeof now->raw_k[l]);
    memcpy(now->raw_v[l], raw + (HEADS + GROUPS) * QK, sizeof now->raw_v[l]);
    for (int32_t j = 0; j < TAPS; j++) {
        const struct node * node = at - j >= 0 ? node_at(self, b, at - j) :
                                   NULL;
        past_q[j] = node != NULL ? node->raw_q[l] : NULL;
        past_k[j] = node != NULL ? node->raw_k[l] : NULL;
        past_v[j] = node != NULL ? node->raw_v[l] : NULL;
    }
    tapped(q, past_q, self->tensors[base + 4], HEADS * QK);
    tapped(now->key[l], past_k, self->tensors[base + 5], GROUPS * QK);
    tapped(now->value[l], past_v, self->tensors[base + 6], GROUPS * VD);
    norm(q, q, self->tensors[base + 7], HEADS, QK);
    norm(now->key[l], now->key[l], self->tensors[base + 8], GROUPS, QK);
    rope(self, q, HEADS, at);
    rope(self, now->key[l], GROUPS, at);
    path_view(self, b, l);
    attend(mix, q, self->view_k, self->view_v, at + 1, GROUPS);
}

static void decoder_attention(struct whistle * self, float * x, int32_t l,
                              const struct beam * beams, int32_t count) {
    const int32_t base = T_DECODER + l * DECODER_STRIDE;
    const int32_t raw = (HEADS + GROUPS) * QK + GROUPS * VD;
    const struct attention a = {
        .gate = base + 9, .out = base + 10
    };
    float * h = floats((size_t)count * DIM);
    float * q = floats((size_t)count * HEADS * QK);
    float * k = floats((size_t)count * GROUPS * QK);
    float * v = floats((size_t)count * GROUPS * VD);
    float * all = floats((size_t)count * raw);
    float * mix = floats((size_t)count * DIM);
    norm(h, x, self->tensors[base], count, DIM);
    product(self, q, h, base + 1, 0, HEADS * QK, count);
    product(self, k, h, base + 2, 0, GROUPS * QK, count);
    product(self, v, h, base + 3, 0, GROUPS * VD, count);
    for (int32_t b = 0; b < count; b++) {
        float * row = all + (size_t)b * raw;
        memcpy(row, q + (size_t)b * HEADS * QK, HEADS * QK * sizeof(float));
        memcpy(row + HEADS * QK, k + (size_t)b * GROUPS * QK,
               GROUPS * QK * sizeof(float));
        memcpy(row + (HEADS + GROUPS) * QK, v + (size_t)b * GROUPS * VD,
               GROUPS * VD * sizeof(float));
        attend_own(self, mix + (size_t)b * DIM, l, &beams[b], row);
    }
    gated(self, h, mix, h, &a, count);
    norm(h, h, self->tensors[base + 11], count, DIM);
    for (size_t i = 0; i < (size_t)count * DIM; i++) {
        x[i] += sigmoid(self->tensors[base + 12][0]) * h[i];
    }
    free(h);
    free(q);
    free(k);
    free(v);
    free(all);
    free(mix);
}

static void weights(float * score, float * total, int32_t frames) {
    float top = -INFINITY;
    float sum = 0.0f;
    for (int32_t s = 0; s < frames; s++) {
        top = fmaxf(top, score[s]);
    }
    for (int32_t s = 0; s < frames; s++) {
        score[s] = expf((score[s] - top) / sqrtf((float)QK));
        sum += score[s];
    }
    for (int32_t s = 0; s < frames; s++) {
        score[s] /= sum;
        total[s] += score[s];
    }
}

static int32_t peak(const float * total, int32_t frames) {
    int32_t pick = 0;
    for (int32_t s = 1; s < frames; s++) {
        pick = total[s] > total[pick] ? s : pick;
    }
    return pick;
}

static void attend_memory(struct whistle * self, float * mix, const float * q,
                          const float * k, const float * v, int32_t frames,
                          int32_t count, float * total) {
    float * score = floats((size_t)count * frames);
    memset(mix, 0, (size_t)count * DIM * sizeof(float));
    for (int32_t h = 0; h < HEADS; h++) {
        const struct gemm_args scores = {
            .a = q + h * QK, .b = k + h * QK, .c = score, .m = count,
            .n = frames, .k = QK, .lda = HEADS * QK, .ldb = HEADS * QK,
            .ldc = frames, .tb = true
        };
        const struct gemm_args mixed = {
            .a = score, .b = v + h * VD, .c = mix + h * VD, .m = count,
            .n = VD, .k = frames, .lda = frames, .ldb = HEADS * VD,
            .ldc = DIM
        };
        memset(score, 0, (size_t)count * frames * sizeof(float));
        self->backend->gemm(self->backend, &scores);
        for (int32_t b = 0; b < count; b++) {
            weights(score + (size_t)b * frames, total + (size_t)b * frames,
                    frames);
        }
        self->backend->gemm(self->backend, &mixed);
    }
    free(score);
}

static void decoder_cross(struct whistle * self, float * x, int32_t l,
                          const struct memory * m, int32_t rows,
                          const struct beam * beams, int32_t count) {
    const int32_t base = T_CROSS + l * CROSS_STRIDE;
    const struct attention a = {
        .gate = base + 4, .out = base + 5
    };
    const float * k = m->k + (size_t)l * rows * HEADS * QK;
    const float * v = m->v + (size_t)l * rows * HEADS * VD;
    float * h = floats((size_t)count * DIM);
    float * q = floats((size_t)count * HEADS * QK);
    float * mix = floats((size_t)count * DIM);
    float * total = floats((size_t)count * m->length);
    norm(h, x, self->tensors[base], count, DIM);
    product(self, q, h, base + 1, 0, HEADS * QK, count);
    norm(q, q, self->tensors[base + 6], count * HEADS, QK);
    attend_memory(self, mix, q, k, v, m->length, count, total);
    for (int32_t b = 0; b < count && l == LAYERS - 1; b++) {
        node_at(self, &beams[b], beams[b].length - 1)->frame =
            peak(total + (size_t)b * m->length, m->length);
    }
    gated(self, h, mix, h, &a, count);
    norm(h, h, self->tensors[base + 8], count, DIM);
    for (size_t i = 0; i < (size_t)count * DIM; i++) {
        x[i] += sigmoid(self->tensors[base + 9][0]) * h[i];
    }
    free(h);
    free(q);
    free(mix);
    free(total);
}

static void decoder_enter(struct whistle * self, const struct beam * b,
                          float * stream, float * key, float * value) {
    float embed[DIM] = { 0 };
    row_decode(self, embed, T_EMBED, node_at(self, b, b->length - 1)->token);
    for (int32_t i = 0; i < LANES * DIM; i++) {
        stream[i] = embed[i % DIM] * sqrtf((float)DIM);
    }
    engram(self, b, key, value);
}

static void decoder_step(struct whistle * self, float * logits,
                         const struct memory * m, int32_t rows,
                         const struct beam * beams, int32_t count) {
    const struct lanes mix = lanes_at(self, T_DECODER_LANES);
    const int32_t sites = SITES * DIM;
    float * stream = floats((size_t)count * LANES * DIM);
    float * u = floats((size_t)count * DIM);
    float * out = floats((size_t)count * DIM);
    float * gate = floats((size_t)count * GATES);
    float * key = floats((size_t)count * sites);
    float * value = floats((size_t)count * sites);
    for (int32_t b = 0; b < count; b++) {
        decoder_enter(self, &beams[b], stream + (size_t)b * LANES * DIM,
                      key + (size_t)b * sites, value + (size_t)b * sites);
    }
    for (int32_t i = 0; i < self->depth; i++) {
        const int32_t l = self->layers[i];
        const int32_t base = T_DECODER + l * DECODER_STRIDE;
        const struct hadamard h = hadamard_at(self, base + 14);
        lanes_enter(self, &mix, l, i % LANES, stream, u, gate, count);
        memcpy(out, u, (size_t)count * DIM * sizeof(float));
        for (int32_t b = 0; b < count && l % 4 == 3; b++) {
            recall(out + (size_t)b * DIM,
                   key + (size_t)b * sites + l / 4 * DIM,
                   value + (size_t)b * sites + l / 4 * DIM);
        }
        decoder_attention(self, out, l, beams, count);
        decoder_cross(self, out, l, m, rows, beams, count);
        hadamard_add(self, out, self->tensors[base + 13], &h, count, 1.0f);
        lanes_leave(stream, u, out, gate, count);
    }
    lanes_mean(out, stream, self->tensors[T_FINAL], count);
    product(self, logits, out, T_EMBED, 0, VOCAB, count);
    free(stream);
    free(u);
    free(out);
    free(gate);
    free(key);
    free(value);
}

static int32_t best(const float * logits, int32_t from, int32_t to) {
    int32_t pick = from;
    for (int32_t i = from + 1; i < to; i++) {
        pick = logits[i] > logits[pick] ? i : pick;
    }
    return pick;
}

static int32_t language_tag(const char * language) {
    const char * const codes[] = { "en", "de", "fr", "es", "it", "nl", "pl" };
    int32_t tag = 0;
    for (int32_t i = 0; i < VOCAB - TEXT; i++) {
        tag = strcmp(language, codes[i]) == 0 ? TEXT + i : tag;
    }
    return tag;
}

static int32_t piece_write(struct whistle * self, char * text, int32_t used,
                           int32_t capacity, int32_t token) {
    const uint8_t * p = self->piece[token];
    const int32_t n = self->piece_size[token];
    int32_t at = used;
    if (self->piece_type[token] == PIECE_BYTE && at + 1 < capacity) {
        text[at++] = (char)strtol((const char *)p + 3, NULL, 16);
    } else if (self->piece_type[token] == PIECE_NORMAL) {
        for (int32_t i = 0; i < n && at + 1 < capacity; i++) {
            const bool space = i + 2 < n && p[i] == 0xE2 &&
                               p[i + 1] == 0x96 && p[i + 2] == 0x81;
            if (!space) {
                text[at++] = (char)p[i];
            } else if (at > 0 && text[at - 1] != ' ') {
                text[at++] = ' ';
            }
            i += space ? 2 : 0;
        }
    }
    return at;
}

static float log_total(const float * logits, int32_t from, int32_t to) {
    float top = -INFINITY;
    float sum = 0.0f;
    for (int32_t i = from; i < to; i++) {
        top = fmaxf(top, logits[i]);
    }
    for (int32_t i = from; i < to; i++) {
        sum += expf(logits[i] - top);
    }
    return top + logf(sum);
}

static int32_t candidates(struct candidate * list, int32_t count,
                          float * logits, const struct beam * b,
                          int32_t index, int32_t forced, int32_t width) {
    const bool first = b->length == 1;
    const int32_t from = first ? TEXT : 0;
    const int32_t to = first ? VOCAB : TEXT;
    const float total = log_total(logits, from, to);
    int32_t made = count;
    if (first && forced != 0) {
        list[made++] = (struct candidate){
            .beam = index, .token = forced, .score = b->score
        };
    }
    for (int32_t w = 0; w < width && !(first && forced != 0); w++) {
        const int32_t pick = best(logits, from, to);
        list[made++] = (struct candidate){
            .beam = index, .token = pick,
            .score = b->score + logits[pick] - total
        };
        logits[pick] = -INFINITY;
    }
    return made;
}

static void order(struct candidate * list, int32_t count) {
    for (int32_t i = 1; i < count; i++) {
        const struct candidate c = list[i];
        int32_t j = i;
        while (j > 0 && list[j - 1].score < c.score) {
            list[j] = list[j - 1];
            j--;
        }
        list[j] = c;
    }
}

static void path_take(const struct whistle * self, const struct beam * b,
                      int32_t length, struct result * r) {
    r->length = length;
    r->frame[0] = 0;
    for (int32_t p = 0; p < length; p++) {
        r->token[p] = node_at(self, b, p)->token;
        r->gain[p] = node_at(self, b, p)->gain;
        r->frame[p + 1] = node_at(self, b, p)->frame;
    }
}

static void keep(const struct whistle * self, const struct beam * b,
                 float score, struct result * best) {
    if (score > best->score) {
        best->score = score;
        path_take(self, b, b->length, best);
    }
}

static int32_t advance(struct whistle * self, const struct beam * cur,
                       struct beam * next, const struct candidate * list,
                       int32_t take, struct result * best) {
    int32_t live = 0;
    for (int32_t c = 0; c < take; c++) {
        const struct beam * from = &cur[list[c].beam];
        if (list[c].token == EOS) {
            keep(self, from, list[c].score / (float)from->length, best);
        } else {
            next[live] = *from;
            next[live].path[from->length] =
                node_new(self, list[c].token, list[c].score - from->score);
            next[live].length = from->length + 1;
            next[live].score = list[c].score;
            live++;
        }
    }
    return live;
}

static void stream(struct whistle * self, const struct beam * live,
                   int32_t count, const struct result * best,
                   struct output * out);

static int32_t search(struct whistle * self, const struct memory * m,
                      int32_t rows, int32_t forced, struct result * best,
                      struct output * out) {
    struct beam * beams = calloc(2 * BEAMS, sizeof(struct beam));
    struct candidate list[BEAMS * BEAMS];
    float * logits = floats((size_t)BEAMS * VOCAB);
    int32_t flip = 0;
    int32_t count = 1;
    int32_t width = self->beams;
    int32_t steps = 0;
    must(beams != NULL, "memory");
    beams[0].path[0] = node_new(self, BOS, 0.0f);
    beams[0].length = 1;
    while (count > 0 && width > 0 && beams[flip * BEAMS].length <= TOKENS) {
        struct beam * cur = beams + flip * BEAMS;
        int32_t made = 0;
        decoder_step(self, logits, m, rows, cur, count);
        for (int32_t i = 0; i < count; i++) {
            made = candidates(list, made, logits + (size_t)i * VOCAB, &cur[i],
                              i, forced, width);
        }
        steps += count;
        order(list, made);
        const int32_t take = made < width ? made : width;
        count = advance(self, cur, beams + (1 - flip) * BEAMS, list, take,
                        best);
        width -= take - count;
        flip = 1 - flip;
        stream(self, beams + flip * BEAMS, count, best, out);
    }
    for (int32_t i = 0; i < count; i++) {
        const struct beam * b = &beams[flip * BEAMS + i];
        keep(self, b, b->score / (float)b->length, best);
    }
    free(beams);
    free(logits);
    return steps;
}

static bool piece_ends(const struct whistle * self, int32_t token) {
    const int32_t n = self->piece_size[token];
    const uint8_t last = n > 0 ? self->piece[token][n - 1] : 0;
    return self->piece_type[token] == PIECE_NORMAL &&
           (last == '.' || last == '!' || last == '?');
}

static bool piece_starts(const struct whistle * self, int32_t token) {
    const uint8_t * p = self->piece[token];
    return self->piece_size[token] > 2 && p[0] == 0xE2 && p[1] == 0x96 &&
           p[2] == 0x81;
}

static int32_t sentence_end(const struct whistle * self,
                            const struct result * best) {
    int32_t stop = 0;
    for (int32_t i = 2; i + 1 < best->length; i++) {
        const bool end = piece_ends(self, best->token[i]) &&
                         piece_starts(self, best->token[i + 1]) &&
                         best->frame[i + 1] > best->frame[i - 1];
        stop = end ? i : stop;
    }
    return stop;
}

static void piece_mark(struct output * out, int32_t sample, float score) {
    struct pieces * p = out->pieces;
    if (p != NULL && p->count < p->capacity) {
        p->list[p->count++] = (struct piece){
            .offset = out->size, .sample = sample, .score = score
        };
    }
}

static void emit(struct whistle * self, struct output * out,
                 const struct result * r, int32_t end) {
    for (int32_t i = out->written; i < end; i++) {
        piece_mark(out, out->base + r->frame[i] * STRIDE, r->gain[i]);
        out->size = piece_write(self, out->text, out->size, out->capacity,
                                r->token[i]);
    }
    if (end > out->written && out->said != NULL) {
        out->text[out->size] = 0;
        out->said(out->who, out->text, out->size);
    }
    out->written = end > out->written ? end : out->written;
}

static int32_t same_start(const struct whistle * self, const struct beam * a,
                          const struct beam * b, int32_t length) {
    int32_t same = 0;
    while (same < length && same < b->length &&
           node_at(self, a, same)->token == node_at(self, b, same)->token) {
        same++;
    }
    return same;
}

static int32_t agreed(const struct whistle * self, const struct beam * live,
                      int32_t count, const struct result * best) {
    int32_t length = count > 0 ? live[0].length : 0;
    for (int32_t i = 1; i < count; i++) {
        length = same_start(self, &live[0], &live[i], length);
    }
    for (int32_t p = 0; best->score > -INFINITY && p < length; p++) {
        length = p < best->length &&
                 best->token[p] == node_at(self, &live[0], p)->token ?
                 length : p;
    }
    return length;
}

static int32_t ready(const struct whistle * self, const struct result * r,
                     const struct output * out) {
    const int32_t stop = out->more ? sentence_end(self, r) : 0;
    int32_t word = out->written;
    for (int32_t i = out->written + 1; i < r->length; i++) {
        word = piece_starts(self, r->token[i]) ? i : word;
    }
    return out->more ? stop + 1 : word;
}

static void stream(struct whistle * self, const struct beam * live,
                   int32_t count, const struct result * best,
                   struct output * out) {
    const int32_t length = out->said != NULL ?
                           agreed(self, live, count, best) : 0;
    if (length > out->written + 1) {
        struct result r = { 0 };
        path_take(self, &live[0], length, &r);
        emit(self, out, &r, ready(self, &r, out));
    }
}

static int32_t clip(struct whistle * self, const float * samples,
                    int32_t count, int32_t forced, bool more,
                    struct output * out) {
    const int32_t frames = 1 + count / HOP;
    const int32_t all = ((((frames - 1) / 2 + 1) - 1) / 2 + 1 - 1) / 2 + 1;
    struct memory m = memory_open(self, samples, count);
    struct result best = { .score = -INFINITY };
    out->base = (int32_t)(samples - out->first);
    out->written = 2;
    out->more = more;
    if (m.spread >= steady) {
        out->steps += search(self, &m, all + all % 2, forced, &best, out);
    }
    const int32_t stop = more ? sentence_end(self, &best) : 0;
    emit(self, out, &best, stop > 0 ? stop + 1 : best.length);
    for (int32_t i = 0; i < self->node_count; i++) {
        free(self->nodes[i]);
    }
    self->node_count = 0;
    free(m.k);
    free(m.v);
    return stop > 0 ?
           (best.frame[stop - 1] + best.frame[stop + 1]) * STRIDE / 2 : count;
}

static int32_t window_end(const float * samples, int32_t at,
                          int32_t count) {
    const int32_t last = at + SAMPLES - SLICE;
    int32_t end = count - at > SAMPLES ? at + SAMPLES : count;
    float least = INFINITY;
    for (int32_t i = at + SAMPLES - SEARCH;
         count - at > SAMPLES && i <= last; i += HOP) {
        const float energy = dot(samples + i, samples + i, SLICE);
        if (energy < least) {
            least = energy;
            end = i + SLICE / 2;
        }
    }
    return end;
}

static int32_t whistle_transcribe(struct asr * asr, const float * samples,
                                  int32_t count, const char * language,
                                  char * text, int32_t capacity,
                                  struct pieces * pieces,
                                  void (* said)(void * who, const char * text,
                                                int32_t size),
                                  void * who) {
    struct whistle * self = (struct whistle *)asr;
    const int32_t forced = language_tag(language != NULL ? language : "");
    struct output out = {
        .text = text, .capacity = capacity, .pieces = pieces,
        .first = samples, .said = said, .who = who
    };
    int32_t at = 0;
    if (pieces != NULL) {
        pieces->count = 0;
    }
    while (at < count) {
        const int32_t end = window_end(samples, at, count);
        at += clip(self, samples + at, end - at, forced, end < count, &out);
        if (at < count && out.size > 0 && out.size + 1 < capacity) {
            text[out.size++] = ' ';
        }
    }
    text[out.size] = 0;
    return out.steps;
}

static void whistle_close(struct asr * asr) {
    struct whistle * self = (struct whistle *)asr;
    for (int32_t i = 0; i < TENSORS; i++) {
        free(self->tensors[i]);
    }
    free(self->view_k);
    free(self->view_v);
    file_unmap(self->file, self->size);
    free(self);
}

static bool whistle_valid(const struct whistle * self) {
    const uint8_t * f = self->file;
    const bool head = f != NULL && self->size > HEADER && le32(f) == TAG &&
                      le32(f + 4) == TENSORS && le32(f + 8) == 28 &&
                      le32(f + 20) == VOCAB && le32(f + 28) == DIM;
    int32_t faults = head ? 0 : 1;
    for (int32_t i = 0; head && i < TENSORS; i++) {
        const uint8_t * record = f + HEADER + 28 * 4 + (size_t)i * RECORD;
        const uint32_t bits = le32(record + 40);
        faults += record[0] == CQ && bits != 2 && bits != 4 ? 1 : 0;
    }
    if (faults == 0) {
        const uint8_t * record = f + HEADER + 28 * 4 +
                                 (size_t)T_MANIFEST * RECORD;
        faults += single(f + le64(record + 20)) == 3246.0f ? 0 : 1;
    }
    return faults == 0;
}

static void whistle_load(struct whistle * self) {
    const int32_t rank[LAYERS] = { 0, 4, 5, 2, 6, 3, 7, 1 };
    int32_t used = 0;
    for (int32_t l = 0; l < LAYERS; l++) {
        if (rank[l] < self->depth) {
            self->layers[used++] = l;
        }
    }
    for (int32_t i = 0; i < 28; i++) {
        self->books[i] = single(self->file + HEADER + i * 4);
    }
    for (int32_t i = 0; i < TENSORS; i++) {
        if (!lazy(self, i)) {
            tensor_load(self, i);
        }
    }
    for (int32_t i = 0; i < DIM; i++) {
        self->p1[i] = (int32_t)self->tensors[T_P1][i];
        self->p2[i] = (int32_t)self->tensors[T_P2][i];
    }
    for (int32_t i = 0; i < QK / 2; i++) {
        self->rope_rate[i] = powf(100000.0f, -(float)(2 * i) / (float)QK);
    }
    self->view_k = floats((size_t)(TOKENS + 1) * GROUPS * QK);
    self->view_v = floats((size_t)(TOKENS + 1) * GROUPS * VD);
}

static struct asr * whistle_open(const char * path, struct backend * backend,
                                 const char * const * options) {
    const char * weights = option(options, "weights", "float");
    const bool packed = strcmp(weights, "packed") == 0;
    struct whistle * self = calloc(1, sizeof *self);
    if (self != NULL) {
        self->asr = (struct asr){
            .name = "whistle", .transcribe = whistle_transcribe,
            .close = whistle_close
        };
        self->backend = backend;
        self->packed = packed;
        self->beams = atoi(option(options, "beam", "5"));
        self->depth = atoi(option(options, "depth", "8"));
        self->file = file_map(path, &self->size);
    }
    const bool ready = self != NULL && self->beams >= 1 &&
                       self->beams <= BEAMS && self->depth >= 2 &&
                       self->depth <= LAYERS &&
                       (packed || strcmp(weights, "float") == 0) &&
                       whistle_valid(self) && pieces_load(self);
    if (ready) {
        whistle_load(self);
    } else if (self != NULL) {
        whistle_close(&self->asr);
    }
    return ready ? &self->asr : NULL;
}

INSTALL(asr, whistle, .pack = "leok7v/whistle/whistle.cact",
        .keys = "weights beam depth", .open = whistle_open)

#ifndef NO_MAIN

int main(int argc, char ** argv) {
    (void)argc;
    return asr_program("whistle", (const char * const *)argv + 1);
}

#endif
