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

enum { PARTS = 16, GRAIN = 1 << 17 };

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

static void blas_gemm(struct backend * backend, const struct gemm_args * g) {
    (void)backend;
    cblas_sgemm(CblasRowMajor, g->ta ? CblasTrans : CblasNoTrans,
                g->tb ? CblasTrans : CblasNoTrans, g->m, g->n, g->k, 1.0f,
                g->a, g->lda, g->b, g->ldb, 1.0f, g->c, g->ldc);
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
    }
    return backend;
}

INSTALL(backend, blas, .keys = "", .open = blas_open)
