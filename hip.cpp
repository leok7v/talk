extern "C" {
#include "backend.h"
}
#include <hip/hip_runtime.h>
#include <stdlib.h>

enum { WIDE = 32768, REGIONS = 4, THREADS = 256 };

struct region {
    const uint8_t * codes;
    int8_t * weights;
    float * scale;
    float * x;
    float * y;
};

struct hip {
    struct backend backend;
    void (* rest)(struct backend * backend, const struct blocks * w,
                  const float * x, float * y);
    struct region regions[REGIONS];
};

__global__ void product(const int8_t * w, const float * scale,
                        const float * x, float * y, int32_t out,
                        int32_t in) {
    const int32_t o = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    const int32_t count = in / BLOCK;
    if (o < out) {
        const int8_t * p = w + (size_t)o * count * (BLOCK + 2) + 2;
        float sum = 0.0f;
        for (int32_t b = 0; b < count; b++) {
            float part = 0.0f;
            for (int32_t i = 0; i < BLOCK; i++) {
                part += (float)p[i] * x[b * BLOCK + i];
            }
            sum += scale[(size_t)o * count + b] * part;
            p += BLOCK + 2;
        }
        y[o] = sum;
    }
}

static void region_free(struct region * r) {
    (void)hipFree(r->weights);
    (void)hipFree(r->scale);
    (void)hipFree(r->x);
    (void)hipFree(r->y);
    *r = region{};
}

static void region_make(struct region * r, const struct blocks * w) {
    const size_t count = (size_t)w->out * (w->in / BLOCK);
    const hipMemcpyKind up = hipMemcpyHostToDevice;
    const bool made =
        hipMalloc(&r->weights, count * (BLOCK + 2)) == hipSuccess &&
        hipMalloc(&r->scale, count * sizeof(float)) == hipSuccess &&
        hipMalloc(&r->x, (size_t)w->in * sizeof(float)) == hipSuccess &&
        hipMalloc(&r->y, (size_t)w->out * sizeof(float)) == hipSuccess &&
        hipMemcpy(r->weights, w->codes, count * (BLOCK + 2), up) ==
            hipSuccess &&
        hipMemcpy(r->scale, w->scale, count * sizeof(float), up) ==
            hipSuccess;
    if (made) {
        r->codes = w->codes;
    } else {
        region_free(r);
    }
}

static struct region * region_of(struct hip * h, const struct blocks * w) {
    int32_t i = 0;
    while (i < REGIONS && h->regions[i].codes != NULL &&
           h->regions[i].codes != w->codes) {
        i++;
    }
    if (i < REGIONS && h->regions[i].codes == NULL) {
        region_make(&h->regions[i], w);
    }
    return i < REGIONS && h->regions[i].codes != NULL ? &h->regions[i] :
           NULL;
}

static bool launch(const struct region * r, const struct blocks * w) {
    product<<<dim3((uint32_t)(w->out + THREADS - 1) / THREADS),
              dim3(THREADS)>>>(r->weights, r->scale, r->x, r->y, w->out,
                               w->in);
    return hipGetLastError() == hipSuccess;
}

static void hip_blocks(struct backend * backend, const struct blocks * w,
                       const float * x, float * y) {
    struct hip * h = (struct hip *)backend;
    const struct region * r = w->out >= WIDE && w->bits == 8 ?
                              region_of(h, w) : NULL;
    const bool done = r != NULL &&
        hipMemcpy(r->x, x, (size_t)w->in * sizeof(float),
                  hipMemcpyHostToDevice) == hipSuccess &&
        launch(r, w) &&
        hipMemcpy(y, r->y, (size_t)w->out * sizeof(float),
                  hipMemcpyDeviceToHost) == hipSuccess;
    if (!done) {
        h->rest(backend, w, x, y);
    }
}

static void hip_close(struct backend * backend) {
    struct hip * h = (struct hip *)backend;
    for (int32_t i = 0; i < REGIONS; i++) {
        region_free(&h->regions[i]);
    }
    free(h);
}

[[maybe_unused]]
static struct backend * hip_open(const char * const * options) {
    struct hip * h = (struct hip *)calloc(1, sizeof *h);
    struct backend * blas = backend_open("blas", options);
    if (h != NULL && blas != NULL) {
        h->backend = *blas;
        h->rest = blas->blocks;
        h->backend.name = "hip";
        h->backend.blocks = hip_blocks;
        h->backend.close = hip_close;
    }
    if (blas != NULL) {
        blas->close(blas);
    }
    if (h != NULL && blas == NULL) {
        hip_close(&h->backend);
        h = NULL;
    }
    return h != NULL ? &h->backend : NULL;
}

#ifndef __HIP_DEVICE_COMPILE__
extern "C" {
extern const struct backend_entry hip_backend;
const struct backend_entry hip_backend = {
    .name = "hip", .keys = "", .open = hip_open
};
}
#endif
