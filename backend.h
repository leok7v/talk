#pragma once
#include "roles.h"

struct gemm_args {
    const float * a;
    const float * b;
    float * c;
    int32_t m;
    int32_t n;
    int32_t k;
    int32_t lda;
    int32_t ldb;
    int32_t ldc;
    bool ta;
    bool tb;
};

struct packed {
    const uint8_t * codes;
    const uint8_t * norms;
    const float * book;
    int32_t bits;
    int32_t out;
    int32_t in;
};

enum { BLOCK = 32 };

struct blocks {
    const uint8_t * codes;
    const float * scale;
    int32_t out;
    int32_t in;
    int32_t bits;
};

struct backend {
    const char * name;
    void (* gemm)(struct backend * backend, const struct gemm_args * g);
    void (* packed)(struct backend * backend, const struct packed * w,
                    const float * x, float * y, int32_t rows);
    void (* blocks)(struct backend * backend, const struct blocks * w,
                    const float * x, float * y);
    void (* exp)(struct backend * backend, float * y, const float * x,
                 int32_t n);
    void (* close)(struct backend * backend);
};

struct backend_entry {
    const char * name;
    const char * keys;
    struct backend * (* open)(const char * const * options);
};

struct backend * backend_open(const char * name,
                              const char * const * options);
void hadamard128(float * v);
void cpu_packed(struct backend * backend, const struct packed * w,
                const float * x, float * y, int32_t rows);
void cpu_blocks(struct backend * backend, const struct blocks * w,
                const float * x, float * y);
void cpu_blocks_part(const struct blocks * w, const float * x, float * y,
                     int32_t first, int32_t last);
