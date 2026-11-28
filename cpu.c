#include "backend.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

enum { GROUP = 128 };

static float cpu_dot(const float * restrict a, const float * restrict b,
                     int32_t k) {
    #pragma clang fp reassociate(on)
    float sum = 0.0f;
    for (int32_t p = 0; p < k; p++) {
        sum += a[p] * b[p];
    }
    return sum;
}

static void cpu_rows(const struct gemm_args * g) {
    for (int32_t i = 0; i < g->m; i++) {
        const float * a = g->a + (size_t)i * g->lda;
        float * c = g->c + (size_t)i * g->ldc;
        for (int32_t j = 0; j < g->n; j++) {
            c[j] += cpu_dot(a, g->b + (size_t)j * g->ldb, g->k);
        }
    }
}

static void cpu_axpy(const struct gemm_args * g) {
    const size_t ars = g->ta ? 1 : (size_t)g->lda;
    const size_t acs = g->ta ? (size_t)g->lda : 1;
    const size_t brs = g->tb ? 1 : (size_t)g->ldb;
    const size_t bcs = g->tb ? (size_t)g->ldb : 1;
    for (int32_t i = 0; i < g->m; i++) {
        float * restrict c = g->c + (size_t)i * g->ldc;
        for (int32_t p = 0; p < g->k; p++) {
            const float v = g->a[i * ars + p * acs];
            const float * b = g->b + p * brs;
            for (int32_t j = 0; j < g->n; j++) {
                c[j] += v * b[j * bcs];
            }
        }
    }
}

size_t weight_bytes(int32_t format, int32_t k) {
    const size_t sizes[] = {
        (size_t)k * 4, (size_t)k * 2, (size_t)k / 32 * 34,
        (size_t)k / 32 * 18, (size_t)k / 256 * 144
    };
    return sizes[format];
}

static float unit(const uint8_t * p) {
    _Float16 h;
    memcpy(&h, p, sizeof h);
    return (float)h;
}

static void widen_q4_k(float * restrict out, const uint8_t * restrict b) {
    const uint8_t * s = b + 4;
    for (int32_t j = 0; j < 8; j++) {
        const int32_t sc = j < 4 ? s[j] & 63 :
                           (s[j + 4] & 15) | (s[j - 4] >> 6) << 4;
        const int32_t mn = j < 4 ? s[j + 4] & 63 :
                           (s[j + 4] >> 4) | (s[j] >> 6) << 4;
        const float scale = unit(b) * (float)sc;
        const float less = unit(b + 2) * (float)mn;
        const uint8_t * p = b + 16 + 32 * (j / 2);
        const int32_t shift = j % 2 != 0 ? 4 : 0;
        for (int32_t i = 0; i < 32; i++) {
            out[j * 32 + i] = scale * (float)(p[i] >> shift & 15) - less;
        }
    }
}

static void widen_row(int32_t format, float * restrict out,
                      const uint8_t * restrict row, int32_t k) {
    if (format == W_F32) {
        memcpy(out, row, (size_t)k * sizeof(float));
    }
    for (int32_t i = 0; format == W_F16 && i < k; i++) {
        out[i] = unit(row + i * 2);
    }
    for (int32_t b = 0; format == W_Q8_0 && b < k / 32; b++) {
        const uint8_t * q = row + b * 34;
        const float scale = unit(q);
        for (int32_t i = 0; i < 32; i++) {
            out[b * 32 + i] = scale * (float)(int8_t)q[2 + i];
        }
    }
    for (int32_t b = 0; format == W_Q4_0 && b < k / 32; b++) {
        const uint8_t * q = row + b * 18;
        const float scale = unit(q);
        for (int32_t i = 0; i < 16; i++) {
            out[b * 32 + i] = scale * (float)((q[2 + i] & 15) - 8);
        }
        for (int32_t i = 0; i < 16; i++) {
            out[b * 32 + 16 + i] = scale * (float)((q[2 + i] >> 4) - 8);
        }
    }
    for (int32_t b = 0; format == W_Q4_K && b < k / 256; b++) {
        widen_q4_k(out + b * 256, row + b * 144);
    }
}

float * cpu_widen(const struct gemm_args * g) {
    float * wide = floats((size_t)g->n * g->k);
    const uint8_t * bytes = (const uint8_t *)(const void *)g->b;
    for (int32_t o = 0; o < g->n; o++) {
        widen_row(g->format, wide + (size_t)o * g->k,
                  bytes + (size_t)o * g->ldb, g->k);
    }
    return wide;
}

void cpu_gemm_one(const struct gemm_args * g, int32_t index) {
    struct gemm_args one = *g;
    one.a = g->a + index * g->sa;
    one.b = g->b + index * g->sb;
    one.c = g->c + index * g->sc;
    for (int32_t i = 0; g->fresh && i < g->m; i++) {
        memset(one.c + (size_t)i * g->ldc, 0, (size_t)g->n * sizeof(float));
    }
    if (one.tb && !one.ta) {
        cpu_rows(&one);
    } else {
        cpu_axpy(&one);
    }
}

struct gemm_args cpu_plain(const struct gemm_args * g, const float * wide) {
    struct gemm_args p = *g;
    p.b = wide != NULL ? wide : g->b;
    p.ldb = wide != NULL ? g->k : g->ldb;
    p.format = W_F32;
    if (g->tc) {
        p = (struct gemm_args){
            .a = p.b, .b = p.a, .c = p.c, .m = p.n, .n = p.m, .k = p.k,
            .lda = p.ldb, .ldb = p.lda, .ldc = p.ldc, .ta = !p.tb,
            .tb = !p.ta, .fresh = p.fresh, .batch = p.batch, .sa = p.sb,
            .sb = p.sa, .sc = p.sc
        };
    }
    return p;
}

static void cpu_gemm(struct backend * backend, const struct gemm_args * g) {
    float * wide = g->format != W_F32 ? cpu_widen(g) : NULL;
    const struct gemm_args plain = cpu_plain(g, wide);
    (void)backend;
    for (int32_t i = 0; i < (g->batch > 1 ? g->batch : 1); i++) {
        cpu_gemm_one(&plain, i);
    }
    free(wide);
}

void hadamard128(float * v) {
    for (int32_t h = 1; h < GROUP; h *= 2) {
        for (int32_t i = 0; i < GROUP; i += 2 * h) {
            for (int32_t j = i; j < i + h; j++) {
                const float a = v[j];
                const float b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
    }
}

static float * turned(const float * x, int32_t rows, int32_t in,
                      int32_t wide) {
    const float scale = 1.0f / sqrtf((float)GROUP);
    float * t = floats((size_t)rows * wide);
    for (int32_t r = 0; r < rows; r++) {
        float * row = t + (size_t)r * wide;
        memcpy(row, x + (size_t)r * in, (size_t)in * sizeof(float));
        for (int32_t g = 0; g < wide; g += GROUP) {
            hadamard128(row + g);
        }
        for (int32_t i = 0; i < wide; i++) {
            row[i] *= scale;
        }
    }
    return t;
}

static void group_values(float * v, const uint8_t * codes,
                         const float * lut, int32_t bits) {
    const int32_t per = 8 / bits;
    for (int32_t i = 0; i < GROUP / per; i++) {
        memcpy(v + i * per, lut + codes[i] * per,
               (size_t)per * sizeof(float));
    }
}

void cpu_packed(struct backend * backend, const struct packed * w,
                const float * x, float * y, int32_t rows) {
    const int32_t per = 8 / w->bits;
    const int32_t groups = (w->in + GROUP - 1) / GROUP;
    const int32_t wide = groups * GROUP;
    const size_t group_bytes = (size_t)(GROUP / per);
    float * t = turned(x, rows, w->in, wide);
    float lut[256 * 4];
    float v[GROUP];
    (void)backend;
    for (int32_t b = 0; b < 256 * per; b++) {
        lut[b] = w->book[b / per >> (b % per * w->bits) &
                         ((1 << w->bits) - 1)];
    }
    memset(y, 0, (size_t)rows * w->out * sizeof(float));
    for (int32_t r = 0; r < w->out; r++) {
        for (int32_t g = 0; g < groups; g++) {
            const size_t at = (size_t)r * groups + g;
            const float norm = half_float(w->norms + at * 2);
            group_values(v, w->codes + at * group_bytes, lut, w->bits);
            for (int32_t i = 0; i < rows; i++) {
                y[(size_t)i * w->out + r] +=
                    norm * cpu_dot(v, t + (size_t)i * wide + g * GROUP,
                                   GROUP);
            }
        }
    }
    free(t);
}

static float block_dot(const uint8_t * restrict q,
                       const float * restrict x) {
    #pragma clang fp reassociate(on)
    float sum = 0.0f;
    #pragma clang loop unroll(disable) vectorize(enable)
    for (int32_t i = 0; i < BLOCK; i++) {
        sum += (float)(int8_t)q[i] * x[i];
    }
    return sum;
}

static float block_dot4(const uint8_t * restrict q,
                        const float * restrict x) {
    #pragma clang fp reassociate(on)
    float sum = 0.0f;
    #pragma clang loop unroll(disable) vectorize(enable)
    for (int32_t i = 0; i < BLOCK / 2; i++) {
        sum += (float)((int32_t)(q[i] & 15) - 8) * x[i] +
               (float)((int32_t)(q[i] >> 4) - 8) * x[i + BLOCK / 2];
    }
    return sum;
}

void cpu_blocks_part(const struct blocks * w, const float * x, float * y,
                     int32_t first, int32_t last) {
    const int32_t count = w->in / BLOCK;
    const size_t step = 2 + (size_t)BLOCK * w->bits / 8;
    for (int32_t o = first; o < last; o++) {
        float sum = 0.0f;
        for (int32_t b = 0; b < count; b++) {
            const size_t at = (size_t)o * count + b;
            const uint8_t * q = w->codes + at * step + 2;
            sum += w->scale[at] * (w->bits == 4 ?
                                   block_dot4(q, x + b * BLOCK) :
                                   block_dot(q, x + b * BLOCK));
        }
        y[o] = sum;
    }
}

void cpu_blocks(struct backend * backend, const struct blocks * w,
                const float * x, float * y) {
    (void)backend;
    cpu_blocks_part(w, x, y, 0, w->out);
}

static void cpu_exp(struct backend * backend, float * y, const float * x,
                    int32_t n) {
    (void)backend;
    for (int32_t i = 0; i < n; i++) {
        y[i] = expf(x[i]);
    }
}

static float gelu_one(float x) {
    const float z = x / 1.4142135f;
    const float e = expf(-z * z);
    const float t = 1.0f / (1.0f + 0.3275911f * fabsf(z));
    const float tail = ((((1.061405429f * t - 1.453152027f) * t +
                          1.421413741f) * t - 0.284496736f) * t +
                        0.254829592f) * t * e;
    return x * (copysignf(1.0f - tail, x) + 1.0f) * 0.5f;
}

static void map_more(const struct map_args * m, float * y, const float * a,
                     const float * b) {
    const int64_t ys = m->ys[2];
    const int64_t as = m->as[2];
    const int64_t bs = m->bs[2];
    switch (m->op) {
    case OP_MUL:
        for (int32_t j = 0; j < m->cols; j++) {
            y[j * ys] = a[j * as] * b[j * bs];
        }
        break;
    case OP_FMA:
        for (int32_t j = 0; j < m->cols; j++) {
            y[j * ys] += a[j * as] * b[j * bs];
        }
        break;
    case OP_SCALE:
        for (int32_t j = 0; j < m->cols; j++) {
            y[j * ys] = m->alpha * a[j * as];
        }
        break;
    case OP_GELU:
        for (int32_t j = 0; j < m->cols; j++) {
            y[j * ys] = gelu_one(a[j * as]);
        }
        break;
    case OP_PRELU:
        for (int32_t j = 0; j < m->cols; j++) {
            y[j * ys] = a[j * as] < 0.0f ? a[j * as] * m->alpha : a[j * as];
        }
        break;
    case OP_TANH:
        for (int32_t j = 0; j < m->cols; j++) {
            y[j * ys] = tanhf(a[j * as]);
        }
        break;
    default:
        break;
    }
}

static void map_row(const struct map_args * m, float * y, const float * a,
                    const float * b) {
    const int64_t ys = m->ys[2];
    const int64_t as = m->as[2];
    const int64_t bs = m->bs[2];
    switch (m->op) {
    case OP_FILL:
        for (int32_t j = 0; j < m->cols; j++) {
            y[j * ys] = m->alpha;
        }
        break;
    case OP_COPY:
        for (int32_t j = 0; j < m->cols; j++) {
            y[j * ys] = a[j * as];
        }
        break;
    case OP_ADD:
        for (int32_t j = 0; j < m->cols; j++) {
            y[j * ys] += m->alpha * a[j * as];
        }
        break;
    case OP_SILU:
        for (int32_t j = 0; j < m->cols; j++) {
            y[j * ys] = a[j * as] / (1.0f + expf(-a[j * as]));
        }
        break;
    case OP_SWIGLU:
        for (int32_t j = 0; j < m->cols; j++) {
            y[j * ys] = a[j * as] * b[j * bs] / (1.0f + expf(-b[j * bs]));
        }
        break;
    case OP_ROPE:
        for (int32_t j = 0; j < m->cols; j++) {
            float * v = y + j * ys;
            const float * cs = a + j * as;
            const int64_t q = m->pair > 0 ? m->pair : 1;
            const float u = v[0];
            v[0] = u * cs[0] - v[q] * cs[1];
            v[q] = v[q] * cs[0] + u * cs[1];
        }
        break;
    default:
        map_more(m, y, a, b);
        break;
    }
}

void cpu_map_part(const struct map_args * m, int64_t first, int64_t last) {
    for (int64_t i = first; i < last; i++) {
        const int64_t t = i / m->rows;
        const int64_t r = i % m->rows;
        map_row(m, m->y + t * m->ys[0] + r * m->ys[1],
                m->a != NULL ? m->a + t * m->as[0] + r * m->as[1] : NULL,
                m->b != NULL ? m->b + t * m->bs[0] + r * m->bs[1] : NULL);
    }
}

static void norm_columns(const struct norm_args * n, int64_t first,
                         int64_t last) {
    const int64_t count = last - first;
    float * mean = floats((size_t)count);
    float * gain = floats((size_t)count);
    for (int32_t i = 0; n->b != NULL && i < n->n; i++) {
        const float * x = n->x + i * n->step + first;
        for (int64_t t = 0; t < count; t++) {
            mean[t] += x[t];
        }
    }
    for (int64_t t = 0; t < count; t++) {
        mean[t] /= (float)n->n;
    }
    for (int32_t i = 0; i < n->n; i++) {
        const float * x = n->x + i * n->step + first;
        for (int64_t t = 0; t < count; t++) {
            gain[t] += (x[t] - mean[t]) * (x[t] - mean[t]);
        }
    }
    for (int64_t t = 0; t < count; t++) {
        gain[t] = 1.0f / sqrtf(gain[t] / (float)n->n + n->eps);
    }
    for (int32_t i = 0; i < n->n; i++) {
        const float * x = n->x + i * n->step + first;
        float * y = n->y + i * n->step + first;
        const float bias = n->b != NULL ? n->b[i] : 0.0f;
        for (int64_t t = 0; t < count; t++) {
            y[t] = (x[t] - mean[t]) * gain[t] * n->w[i] + bias;
        }
    }
    free(mean);
    free(gain);
}

static void norm_rows(const struct norm_args * n, int64_t first,
                      int64_t last) {
    const int64_t ld = n->ld > 0 ? n->ld : n->n;
    const int64_t step = n->step > 0 ? n->step : 1;
    for (int64_t r = first; r < last; r++) {
        const float * x = n->x + r * ld;
        float * y = n->y + r * ld;
        float mean = 0.0f;
        float spread = 0.0f;
        for (int32_t i = 0; n->b != NULL && i < n->n; i++) {
            mean += x[i * step];
        }
        mean /= (float)n->n;
        for (int32_t i = 0; i < n->n; i++) {
            spread += (x[i * step] - mean) * (x[i * step] - mean);
        }
        const float gain = 1.0f / sqrtf(spread / (float)n->n + n->eps);
        for (int32_t i = 0; i < n->n; i++) {
            y[i * step] = (x[i * step] - mean) * gain * n->w[i] +
                          (n->b != NULL ? n->b[i] : 0.0f);
        }
    }
}

void cpu_norm_part(const struct norm_args * n, int64_t first, int64_t last) {
    if (n->step > 1 && n->ld == 1) {
        norm_columns(n, first, last);
    } else {
        norm_rows(n, first, last);
    }
}

void cpu_softmax_part(struct backend * backend, float * x, int32_t n,
                      float scale, int32_t causal, int64_t first,
                      int64_t last) {
    for (int64_t i = first; i < last; i++) {
        float * s = x + i * n;
        const int32_t seen = causal > 0 ? (int32_t)(i % causal) + 1 : n;
        float top = -INFINITY;
        float sum = 0.0f;
        for (int32_t j = 0; j < seen; j++) {
            s[j] *= scale;
            top = fmaxf(top, s[j]);
        }
        for (int32_t j = 0; j < seen; j++) {
            s[j] -= top;
        }
        backend->exp(backend, s, s, seen);
        for (int32_t j = 0; j < seen; j++) {
            sum += s[j];
        }
        for (int32_t j = 0; j < n; j++) {
            s[j] = j < seen ? s[j] / sum : 0.0f;
        }
    }
}

static void cpu_map(struct backend * backend, const struct map_args * m) {
    (void)backend;
    cpu_map_part(m, 0, (int64_t)(m->batch > 1 ? m->batch : 1) * m->rows);
}

static void cpu_norm(struct backend * backend, const struct norm_args * n) {
    (void)backend;
    cpu_norm_part(n, 0, n->rows);
}

static void cpu_softmax(struct backend * backend, float * x, int32_t rows,
                        int32_t n, float scale, int32_t causal) {
    cpu_softmax_part(backend, x, n, scale, causal, 0, rows);
}

static float * cpu_alloc(struct backend * backend, size_t count) {
    (void)backend;
    return floats(count);
}

static void cpu_adopt(struct backend * backend, const void * data,
                      size_t bytes) {
    (void)backend;
    (void)data;
    (void)bytes;
}

static void cpu_release(struct backend * backend, float * p) {
    (void)backend;
    free(p);
}

static void cpu_sync(struct backend * backend) {
    (void)backend;
}

void cpu_ops(struct backend * backend) {
    backend->map = cpu_map;
    backend->norm = cpu_norm;
    backend->softmax = cpu_softmax;
    backend->alloc = cpu_alloc;
    backend->adopt = cpu_adopt;
    backend->release = cpu_release;
    backend->sync = cpu_sync;
}

static void cpu_close(struct backend * backend) {
    free(backend);
}

static struct backend * cpu_open(const char * const * options) {
    struct backend * backend = malloc(sizeof *backend);
    (void)options;
    if (backend != NULL) {
        *backend = (struct backend){
            .name = "cpu", .gemm = cpu_gemm, .packed = cpu_packed,
            .blocks = cpu_blocks, .exp = cpu_exp, .close = cpu_close
        };
        cpu_ops(backend);
    }
    return backend;
}

INSTALL(backend, cpu, .keys = "", .open = cpu_open)
