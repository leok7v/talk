#include "backend.h"
#include <stdlib.h>
#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#else
#include <cblas.h>
#include <math.h>
enum { THREADS = 8 };
void omp_set_num_threads(int count);
int omp_get_num_threads(void);
int omp_get_thread_num(void);
void GOMP_parallel(void (* run)(void *), void * data, unsigned threads,
                   unsigned flags);
#endif

enum { PARTS = 16, GRAIN = 1 << 17, SMALL = 1 << 15 };

struct spread {
    void (* run)(const void * context, int64_t first, int64_t last);
    const void * context;
    int64_t count;
    int32_t parts;
};

struct soft {
    struct backend * backend;
    float * x;
    int32_t n;
    float scale;
    int32_t causal;
};

struct split {
    const struct blocks * w;
    const float * x;
    float * y;
    int32_t parts;
};

static void blas_part(void * context, size_t part) {
    const struct split * s = context;
    const int64_t out = s->w->out;
    const int64_t at = (int64_t)part;
    cpu_blocks_part(s->w, s->x, s->y, (int32_t)(out * at / s->parts),
                    (int32_t)(out * (at + 1) / s->parts));
}

#ifndef __APPLE__
static void blas_team(void * context) {
    struct split s = *(const struct split *)context;
    s.parts = omp_get_num_threads();
    blas_part(&s, (size_t)omp_get_thread_num());
}
#endif

static void spread_part(void * context, size_t part) {
    const struct spread * s = context;
    const int64_t at = (int64_t)part;
    s->run(s->context, s->count * at / s->parts,
           s->count * (at + 1) / s->parts);
}

#ifndef __APPLE__
static void spread_team(void * context) {
    struct spread s = *(const struct spread *)context;
    s.parts = omp_get_num_threads();
    spread_part(&s, (size_t)omp_get_thread_num());
}
#endif

static void spread(void (* run)(const void * context, int64_t first,
                                int64_t last),
                   const void * context, int64_t count, int64_t cost) {
    const int64_t grains = count * cost / GRAIN;
    const int64_t most = count < PARTS ? count : PARTS;
    struct spread s = {
        .run = run, .context = context, .count = count,
        .parts = grains < 2 ? 1 : (int32_t)(grains < most ? grains : most)
    };
    if (s.parts <= 1) {
        run(context, 0, count);
    } else {
#ifdef __APPLE__
        dispatch_apply_f((size_t)s.parts,
                         dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,
                                                   0), &s, spread_part);
#else
        GOMP_parallel(spread_team, &s, (unsigned)s.parts, 0);
#endif
    }
}

static void blas_blocks(struct backend * backend, const struct blocks * w,
                        const float * x, float * y) {
    const int64_t grains = (int64_t)w->out * w->in / GRAIN;
    struct split s = {
        .w = w, .x = x, .y = y,
        .parts = grains < 2 ? 1 : grains < PARTS ? (int32_t)grains : PARTS
    };
    (void)backend;
    if (s.parts == 1) {
        blas_part(&s, 0);
    } else {
#ifdef __APPLE__
        dispatch_apply_f((size_t)s.parts,
                         dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,
                                                   0), &s, blas_part);
#else
        GOMP_parallel(blas_team, &s, (unsigned)s.parts, 0);
#endif
    }
}

static void blas_one(const struct gemm_args * g, int32_t index) {
    if ((int64_t)g->m * g->n * g->k < SMALL) {
        cpu_gemm_one(g, index);
    } else {
        cblas_sgemm(CblasRowMajor, g->ta ? CblasTrans : CblasNoTrans,
                    g->tb ? CblasTrans : CblasNoTrans, g->m, g->n, g->k,
                    1.0f, g->a + index * g->sa, g->lda, g->b + index * g->sb,
                    g->ldb, g->fresh ? 0.0f : 1.0f, g->c + index * g->sc,
                    g->ldc);
    }
}

static void gemm_run(const void * context, int64_t first, int64_t last) {
    for (int64_t i = first; i < last; i++) {
        blas_one(context, (int32_t)i);
    }
}

static void blas_gemm(struct backend * backend, const struct gemm_args * g) {
    const int32_t batch = g->batch > 1 ? g->batch : 1;
    float * wide = g->format != W_F32 ? cpu_widen(g) : NULL;
    const struct gemm_args plain = cpu_plain(g, wide);
    (void)backend;
    spread(gemm_run, &plain, batch, (int64_t)g->m * g->n * g->k);
    free(wide);
}

static void map_run(const void * context, int64_t first, int64_t last) {
    cpu_map_part(context, first, last);
}

static void blas_map(struct backend * backend, const struct map_args * m) {
    (void)backend;
    const bool heavy = m->op == OP_SILU || m->op == OP_SWIGLU ||
                       m->op == OP_GELU || m->op == OP_TANH;
    spread(map_run, m, (int64_t)(m->batch > 1 ? m->batch : 1) * m->rows,
           (int64_t)m->cols * (heavy ? 16 : 2));
}

static void norm_run(const void * context, int64_t first, int64_t last) {
    cpu_norm_part(context, first, last);
}

static void blas_norm(struct backend * backend, const struct norm_args * n) {
    (void)backend;
    spread(norm_run, n, n->rows, 4 * (int64_t)n->n);
}

static void soft_run(const void * context, int64_t first, int64_t last) {
    const struct soft * s = context;
    cpu_softmax_part(s->backend, s->x, s->n, s->scale, s->causal, first,
                     last);
}

static void blas_softmax(struct backend * backend, float * x, int32_t rows,
                         int32_t n, float scale, int32_t causal) {
    const struct soft s = {
        .backend = backend, .x = x, .n = n, .scale = scale, .causal = causal
    };
    spread(soft_run, &s, rows, 8 * (int64_t)n);
}

static void blas_exp(struct backend * backend, float * y, const float * x,
                     int32_t n) {
    (void)backend;
#ifdef __APPLE__
    vvexpf(y, x, &n);
#else
    for (int32_t i = 0; i < n; i++) {
        y[i] = expf(x[i]);
    }
#endif
}

static void blas_close(struct backend * backend) {
    free(backend);
}

static struct backend * blas_open(const char * const * options) {
    struct backend * backend = malloc(sizeof *backend);
    (void)options;
#ifndef __APPLE__
    if (getenv("OMP_NUM_THREADS") == NULL) {
        openblas_set_num_threads(THREADS);
        omp_set_num_threads(THREADS);
    }
#endif
    if (backend != NULL) {
        *backend = (struct backend){
            .name = "blas", .gemm = blas_gemm, .packed = cpu_packed,
            .blocks = blas_blocks, .exp = blas_exp, .close = blas_close
        };
        cpu_ops(backend);
        backend->map = blas_map;
        backend->norm = blas_norm;
        backend->softmax = blas_softmax;
    }
    return backend;
}

INSTALL(backend, blas, .keys = "", .open = blas_open)
