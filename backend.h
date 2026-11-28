#ifndef BACKEND_H
#define BACKEND_H
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
    bool tc;
    bool fresh;
    int32_t format;
    int32_t batch;
    int64_t sa;
    int64_t sb;
    int64_t sc;
};

enum { W_F32, W_F16, W_Q8_0, W_Q4_0, W_Q4_K };

enum { OP_FILL, OP_COPY, OP_ADD, OP_SILU, OP_SWIGLU, OP_ROPE, OP_MUL,
       OP_FMA, OP_SCALE, OP_GELU, OP_PRELU, OP_TANH };

struct map_args {
    float * y;
    const float * a;
    const float * b;
    int32_t op;
    float alpha;
    int32_t batch;
    int32_t rows;
    int32_t cols;
    int64_t ys[3];
    int64_t as[3];
    int64_t bs[3];
    int64_t pair;
};

struct norm_args {
    float * y;
    const float * x;
    const float * w;
    const float * b;
    int32_t rows;
    int32_t n;
    float eps;
    int64_t ld;
    int64_t step;
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
    bool queued;
    void (* gemm)(struct backend * backend, const struct gemm_args * g);
    void (* packed)(struct backend * backend, const struct packed * w,
                    const float * x, float * y, int32_t rows);
    void (* blocks)(struct backend * backend, const struct blocks * w,
                    const float * x, float * y);
    void (* exp)(struct backend * backend, float * y, const float * x,
                 int32_t n);
    void (* map)(struct backend * backend, const struct map_args * m);
    void (* norm)(struct backend * backend, const struct norm_args * n);
    void (* softmax)(struct backend * backend, float * x, int32_t rows,
                     int32_t n, float scale, int32_t causal);
    float * (* alloc)(struct backend * backend, size_t count);
    void (* adopt)(struct backend * backend, const void * data,
                   size_t bytes);
    void (* release)(struct backend * backend, float * p);
    void (* sync)(struct backend * backend);
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
void cpu_gemm_one(const struct gemm_args * g, int32_t index);
float * cpu_widen(const struct gemm_args * g);
struct gemm_args cpu_plain(const struct gemm_args * g, const float * wide);
size_t weight_bytes(int32_t format, int32_t k);
void cpu_map_part(const struct map_args * m, int64_t first, int64_t last);
void cpu_norm_part(const struct norm_args * n, int64_t first, int64_t last);
void cpu_softmax_part(struct backend * backend, float * x, int32_t n,
                      float scale, int32_t causal, int64_t first,
                      int64_t last);
void cpu_ops(struct backend * backend);

#endif
