#include "backend.h"
#include <stdlib.h>
#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#include <math.h>
enum { THREADS = 8 };
void omp_set_num_threads(int count);
#endif

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
            .blocks = cpu_blocks, .exp = blas_exp, .close = blas_close
        };
    }
    return backend;
}

INSTALL(backend, blas, .keys = "", .open = blas_open)
