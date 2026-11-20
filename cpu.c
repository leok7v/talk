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

static void cpu_gemm(struct backend * backend, const struct gemm_args * g) {
    (void)backend;
    if (g->tb && !g->ta) {
        cpu_rows(g);
    } else {
        cpu_axpy(g);
    }
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
    }
    return backend;
}

INSTALL(backend, cpu, .keys = "", .open = cpu_open)
