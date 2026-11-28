#include "backend.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

enum { WIDE = 32768, REGIONS = 4, SPANS = 4096, BATCH = 64, PAGE = 16384,
       LANES = 256, SPLIT = 4, FEW = 16, DEEP = 1024 };

#define METAL(...) #__VA_ARGS__
static const char * const products[] = {
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct tile { uint m; uint n; uint k; uint ldx; uint ldw; uint ldc;\n"
    "              uint ta; uint tc; uint fresh; uint parts; uint slab; };\n"
    "float unit(device const uchar * b) {\n"
    "    return float(as_type<half>(ushort(b[0] | (b[1] << 8))));\n"
    "}\n"
    "void run_f32(device const uchar * row, uint q, thread float * v) {\n"
    "    device const float * p = (device const float *)row + q;\n"
    "    for (uint i = 0; i < 16; i++) { v[i] = p[i]; }\n"
    "}\n"
    "void run_f16(device const uchar * row, uint q, thread float * v) {\n"
    "    device const half * p = (device const half *)row + q;\n"
    "    for (uint i = 0; i < 16; i++) { v[i] = float(p[i]); }\n"
    "}\n"
    "void run_q8_0(device const uchar * row, uint q, thread float * v) {\n"
    "    device const uchar * b = row + (q / 32) * 34;\n"
    "    device const char * c = (device const char *)(b + 2 + q % 32);\n"
    "    for (uint i = 0; i < 16; i++) { v[i] = unit(b) * float(c[i]); }\n"
    "}\n"
    "void run_q4_0(device const uchar * row, uint q, thread float * v) {\n"
    "    device const uchar * b = row + (q / 32) * 18;\n"
    "    const uint high = q % 32 != 0 ? 4 : 0;\n"
    "    for (uint i = 0; i < 16; i++) {\n"
    "        v[i] = unit(b) * (float((b[2 + i] >> high) & 15) - 8.0f);\n"
    "    }\n"
    "}\n"
    "void run_q4_k(device const uchar * row, uint q, thread float * v) {\n"
    "    device const uchar * b = row + (q / 256) * 144;\n"
    "    device const uchar * s = b + 4;\n"
    "    const uint j = q % 256 / 32;\n"
    "    const uint sc = j < 4 ? s[j] & 63 :\n"
    "                    (s[j + 4] & 15) | ((s[j - 4] >> 6) << 4);\n"
    "    const uint mn = j < 4 ? s[j + 4] & 63 :\n"
    "                    (s[j + 4] >> 4) | ((s[j] >> 6) << 4);\n"
    "    const uint high = (j & 1) != 0 ? 4 : 0;\n"
    "    device const uchar * p = b + 16 + 32 * (j / 2) + q % 32;\n"
    "    for (uint i = 0; i < 16; i++) {\n"
    "        v[i] = unit(b) * float(sc) * float((p[i] >> high) & 15) -\n"
    "               unit(b + 2) * float(mn);\n"
    "    }\n"
    "}\n"
    "#define FOUR(F) F(0) F(1) F(2) F(3)\n"
    "#define TILE(F) \\\n"
    "    F(0, 0, 0) F(1, 0, 1) F(2, 0, 2) F(3, 0, 3) \\\n"
    "    F(4, 1, 0) F(5, 1, 1) F(6, 1, 2) F(7, 1, 3) \\\n"
    "    F(8, 2, 0) F(9, 2, 1) F(10, 2, 2) F(11, 2, 3) \\\n"
    "    F(12, 3, 0) F(13, 3, 1) F(14, 3, 2) F(15, 3, 3)\n"
    "#define ROWS(i) simdgroup_float8x8 a##i; \\\n"
    "    const uint y##i = min(row + 8 * i, g.m - 8);\n"
    "#define COLS(j) simdgroup_float8x8 b##j; \\\n"
    "    const uint x##j = min(col + 8 * j, g.n - 8);\n"
    "#define SUMS(i, r, o) simdgroup_float8x8 s##i = \\\n"
    "    make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n"
    "#define FROM_C(i, r, o) if (g.tc != 0) { \\\n"
    "    simdgroup_load(s##i, D, g.ldc, ulong2(y##r, x##o), true); \\\n"
    "} else { simdgroup_load(s##i, D, g.ldc, ulong2(x##o, y##r)); }\n"
    "#define TO_C(i, r, o) if (g.tc != 0) { \\\n"
    "    simdgroup_store(s##i, D, g.ldc, ulong2(y##r, x##o), true); \\\n"
    "} else { simdgroup_store(s##i, D, g.ldc, ulong2(x##o, y##r)); }\n"
    "#define ADD(i, r, o) simdgroup_multiply_accumulate(s##i, a##r, b##o, s##"
    "i);\n"
    "#define LOAD_X(i) if (g.ta != 0) { \\\n"
    "    simdgroup_load(a##i, X, g.ldx, ulong2(y##i, k), true); \\\n"
    "} else { simdgroup_load(a##i, X, g.ldx, ulong2(k, y##i)); }\n"
    "#define LOAD_W(j) simdgroup_load(b##j, w, 64, ulong2(x##j - base, k - k0"
    "));\n"
    "#define PRODUCT(NAME, RUN)                                              "
    "   \\\n"
    "    kernel void NAME(device const float * X [[buffer(0)]],              "
    "   \\\n"
    "                     device const uchar * W [[buffer(1)]],              "
    "   \\\n"
    "                     device float * C [[buffer(2)]],                    "
    "   \\\n"
    "                     constant tile & g [[buffer(3)]],                   "
    "   \\\n"
    "                     uint3 group [[threadgroup_position_in_grid]],      "
    "   \\\n"
    "                     ushort sg [[simdgroup_index_in_threadgroup]],      "
    "   \\\n"
    "                     ushort t [[thread_index_in_threadgroup]]) {        "
    "   \\\n"
    "        threadgroup float w[32 * 64];                                   "
    "   \\\n",
    "        const uint base = min(group.x * 64, g.n - 8);                   "
    "   \\\n"
    "        const uint row = group.y * 64 + (sg / 2) * 32;                  "
    "   \\\n"
    "        const uint col = group.x * 64 + (sg % 2) * 32;                  "
    "   \\\n"
    "        const uint live = row < g.m && col < g.n ? 1 : 0;               "
    "   \\\n"
    "        const uint mine = t / 2;                                        "
    "   \\\n"
    "        const uint part = (t % 2) * 16;                                 "
    "   \\\n"
    "        device float * D = C + ulong(group.z) * g.slab;                 "
    "   \\\n"
    "        const uint first = group.z * (g.k / g.parts);                   "
    "   \\\n"
    "        const uint last = first + g.k / g.parts;                        "
    "   \\\n"
    "        FOUR(ROWS) FOUR(COLS) TILE(SUMS)                                "
    "   \\\n"
    "        if (live != 0 && g.fresh == 0) {                                "
    "   \\\n"
    "            TILE(FROM_C)                                                "
    "   \\\n"
    "        }                                                               "
    "   \\\n"
    "        for (uint k0 = first; k0 < last; k0 += 32) {                    "
    "   \\\n"
    "            float v[16];                                                "
    "   \\\n"
    "            for (uint i = 0; i < 16; i++) {                             "
    "   \\\n"
    "                v[i] = 0.0f;                                            "
    "   \\\n"
    "            }                                                           "
    "   \\\n"
    "            if (base + mine < g.n) {                                    "
    "   \\\n"
    "                RUN(W + ulong(base + mine) * g.ldw, k0 + part, v);      "
    "   \\\n"
    "            }                                                           "
    "   \\\n"
    "            for (uint i = 0; i < 16; i++) {                             "
    "   \\\n"
    "                w[(part + i) * 64 + mine] = v[i];                       "
    "   \\\n"
    "            }                                                           "
    "   \\\n"
    "            threadgroup_barrier(mem_flags::mem_threadgroup);            "
    "   \\\n"
    "            for (uint k = k0; live != 0 && k < k0 + 32; k += 8) {       "
    "   \\\n"
    "                FOUR(LOAD_X) FOUR(LOAD_W) TILE(ADD)                     "
    "   \\\n"
    "            }                                                           "
    "   \\\n"
    "            threadgroup_barrier(mem_flags::mem_threadgroup);            "
    "   \\\n"
    "        }                                                               "
    "   \\\n"
    "        if (live != 0) {                                                "
    "   \\\n"
    "            TILE(TO_C)                                                  "
    "   \\\n"
    "        }                                                               "
    "   \\\n"
    "    }\n"
    "PRODUCT(product_f32, run_f32)\n"
    "PRODUCT(product_f16, run_f16)\n"
    "PRODUCT(product_q8_0, run_q8_0)\n"
    "PRODUCT(product_q4_0, run_q4_0)\n"
    "PRODUCT(product_q4_k, run_q4_k)\n",
    NULL
};

static const char * const product_names[] = {
    "product_f32", "product_f16", "product_q8_0", "product_q4_0",
    "product_q4_k"
};

enum { FORMATS = sizeof product_names / sizeof product_names[0] };

static const char * const source = METAL(
    using namespace metal;
    struct shape {
        int ys0; int ys1; int ys2;
        int as0; int as1; int as2;
        int bs0; int bs1; int bs2;
        int pair; int op; float alpha;
    };
    struct rule { uint n; float eps; uint center; long ld; long step; };
    float gelu_one(float x) {
        const float z = x / 1.4142135f;
        const float t = 1.0f / (1.0f + 0.3275911f * fabs(z));
        const float tail = ((((1.061405429f * t - 1.453152027f) * t +
                              1.421413741f) * t - 0.284496736f) * t +
                            0.254829592f) * t * exp(-z * z);
        return x * (copysign(1.0f - tail, x) + 1.0f) * 0.5f;
    }
    struct soft { uint n; float scale; uint causal; };
    kernel void blocks(device const char * w [[buffer(0)]],
                       device const float * x [[buffer(1)]],
                       device float * y [[buffer(2)]],
                       constant uint & in [[buffer(3)]],
                       uint o [[thread_position_in_grid]]) {
        const uint count = in / 32;
        ulong p = ulong(o) * count * 34;
        float sum = 0.0f;
        for (uint b = 0; b < count; b++) {
            const float scale = as_type<half>(ushort(
                ushort(uchar(w[p])) | ushort(uchar(w[p + 1])) << 8));
            float4 part = float4(0.0f);
            for (uint i = 0; i < 32; i += 4) {
                const float4 q = float4(float(w[p + 2 + i]),
                                        float(w[p + 3 + i]),
                                        float(w[p + 4 + i]),
                                        float(w[p + 5 + i]));
                const device float * v = x + b * 32 + i;
                part += q * float4(v[0], v[1], v[2], v[3]);
            }
            sum += scale * (part.x + part.y + part.z + part.w);
            p += 34;
        }
        y[o] = sum;
    }
    kernel void map(device float * y [[buffer(0)]],
                    device const float * a [[buffer(1)]],
                    device const float * b [[buffer(2)]],
                    constant shape & s [[buffer(3)]],
                    uint3 at [[thread_position_in_grid]]) {
        const int3 p = int3(at);
        const int iy = p.z * s.ys0 + p.y * s.ys1 + p.x * s.ys2;
        const int ia = p.z * s.as0 + p.y * s.as1 + p.x * s.as2;
        const int ib = p.z * s.bs0 + p.y * s.bs1 + p.x * s.bs2;
        if (s.op == 0) {
            y[iy] = s.alpha;
        } else if (s.op == 1) {
            y[iy] = a[ia];
        } else if (s.op == 2) {
            y[iy] += s.alpha * a[ia];
        } else if (s.op == 3) {
            y[iy] = a[ia] / (1.0f + exp(-a[ia]));
        } else if (s.op == 4) {
            y[iy] = a[ia] * b[ib] / (1.0f + exp(-b[ib]));
        } else if (s.op == 5) {
            const int q = s.pair > 0 ? s.pair : 1;
            const float u = y[iy];
            const float v = y[iy + q];
            y[iy] = u * a[ia] - v * a[ia + 1];
            y[iy + q] = v * a[ia] + u * a[ia + 1];
        } else if (s.op == 6) {
            y[iy] = a[ia] * b[ib];
        } else if (s.op == 7) {
            y[iy] += a[ia] * b[ib];
        } else if (s.op == 8) {
            y[iy] = s.alpha * a[ia];
        } else if (s.op == 9) {
            y[iy] = gelu_one(a[ia]);
        } else if (s.op == 10) {
            y[iy] = a[ia] < 0.0f ? a[ia] * s.alpha : a[ia];
        } else if (s.op == 11) {
            y[iy] = tanh(a[ia]);
        }
    }
    kernel void norm(device float * y [[buffer(0)]],
                     device const float * x [[buffer(1)]],
                     device const float * w [[buffer(2)]],
                     device const float * b [[buffer(3)]],
                     constant rule & r [[buffer(4)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_simdgroup]]) {
        device const float * in = x + long(row) * r.ld;
        device float * out = y + long(row) * r.ld;
        float mean = 0.0f;
        float spread = 0.0f;
        for (uint i = lane; r.center != 0 && i < r.n; i += 32) {
            mean += in[i * r.step];
        }
        mean = simd_sum(mean) / float(r.n);
        for (uint i = lane; i < r.n; i += 32) {
            spread += (in[i * r.step] - mean) * (in[i * r.step] - mean);
        }
        const float gain = 1.0f / sqrt(simd_sum(spread) / float(r.n) + r.eps);
        for (uint i = lane; i < r.n; i += 32) {
            out[i * r.step] = (in[i * r.step] - mean) * gain * w[i] +
                              (r.center != 0 ? b[i] : 0.0f);
        }
    }
    kernel void soften(device float * x [[buffer(0)]],
                       constant soft & s [[buffer(1)]],
                       uint row [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
        device float * v = x + ulong(row) * s.n;
        const uint seen = s.causal > 0 ? row % s.causal + 1 : s.n;
        float top = -INFINITY;
        float sum = 0.0f;
        for (uint j = lane; j < seen; j += 32) {
            top = max(top, v[j] * s.scale);
        }
        top = simd_max(top);
        for (uint j = lane; j < seen; j += 32) {
            v[j] = exp(v[j] * s.scale - top);
            sum += v[j];
        }
        sum = simd_sum(sum);
        for (uint j = lane; j < s.n; j += 32) {
            v[j] = j < seen ? v[j] / sum : 0.0f;
        }
    }
);

struct shape {
    int32_t ys[3];
    int32_t as[3];
    int32_t bs[3];
    int32_t pair;
    int32_t op;
    float alpha;
};

struct rule {
    uint32_t n;
    float eps;
    uint32_t center;
    int64_t ld;
    int64_t step;
};

struct soft {
    uint32_t n;
    float scale;
    uint32_t causal;
};

struct tile {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t ldx;
    uint32_t ldw;
    uint32_t ldc;
    uint32_t ta;
    uint32_t tc;
    uint32_t fresh;
    uint32_t parts;
    uint32_t slab;
};

struct region {
    const uint8_t * codes;
    id<MTLBuffer> weights;
    id<MTLBuffer> x;
    id<MTLBuffer> y;
};

struct span {
    uintptr_t base;
    size_t bytes;
    id<MTLBuffer> buffer;
    bool idle;
};

struct metal {
    struct backend backend;
    struct backend cpu;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> state;
    id<MTLComputePipelineState> maps;
    id<MTLComputePipelineState> norms;
    id<MTLComputePipelineState> softs;
    id<MTLComputePipelineState> tiles[FORMATS];
    bool own;
    id<MTLCommandBuffer> commands;
    id<MTLCommandBuffer> last;
    id<MTLComputeCommandEncoder> encoder;
    int32_t ops;
    struct region regions[REGIONS];
    struct span spans[SPANS];
    int32_t count;
};

static void region_make(struct metal * m, struct region * r,
                        const struct blocks * w) {
    const uintptr_t page = (uintptr_t)getpagesize();
    const uintptr_t at = (uintptr_t)w->codes;
    const size_t bytes = (size_t)w->out * (w->in / BLOCK) * (BLOCK + 2);
    const uintptr_t first = at / page * page;
    const uintptr_t last = (at + bytes + page - 1) / page * page;
    r->weights = [m->device
        newBufferWithBytesNoCopy:(void *)first length:last - first
                         options:MTLResourceStorageModeShared deallocator:nil];
    r->x = [m->device newBufferWithLength:(size_t)w->in * sizeof(float)
                                  options:MTLResourceStorageModeShared];
    r->y = [m->device newBufferWithLength:(size_t)w->out * sizeof(float)
                                  options:MTLResourceStorageModeShared];
    if (r->weights != nil && r->x != nil && r->y != nil) {
        r->codes = w->codes;
    } else {
        *r = (struct region){ 0 };
    }
}

static struct region * region_of(struct metal * m, const struct blocks * w) {
    int32_t i = 0;
    while (i < REGIONS && m->regions[i].codes != NULL &&
           m->regions[i].codes != w->codes) {
        i++;
    }
    if (i < REGIONS && m->regions[i].codes == NULL) {
        region_make(m, &m->regions[i], w);
    }
    return i < REGIONS && m->regions[i].codes != NULL ? &m->regions[i] :
           NULL;
}

static void metal_sync(struct backend * backend) {
    struct metal * m = (struct metal *)backend;
    @autoreleasepool {
        [m->encoder endEncoding];
        m->encoder = nil;
        [m->commands commit];
        m->last = m->commands != nil ? m->commands : m->last;
        m->commands = nil;
        m->ops = 0;
        [m->last waitUntilCompleted];
        m->last = nil;
    }
}

static void metal_blocks(struct backend * backend, const struct blocks * w,
                         const float * x, float * y) {
    struct metal * m = (struct metal *)backend;
    struct region * r = w->out >= WIDE && w->bits == 8 ? region_of(m, w) :
                        NULL;
    const uint32_t in = (uint32_t)w->in;
    @autoreleasepool {
        if (r != NULL) {
            const NSUInteger wide = m->state.threadExecutionWidth;
            id<MTLCommandBuffer> commands = [m->queue commandBuffer];
            id<MTLComputeCommandEncoder> e = [commands computeCommandEncoder];
            memcpy(r->x.contents, x, (size_t)w->in * sizeof(float));
            [e setComputePipelineState:m->state];
            [e setBuffer:r->weights
                  offset:(uintptr_t)w->codes % (uintptr_t)getpagesize()
                 atIndex:0];
            [e setBuffer:r->x offset:0 atIndex:1];
            [e setBuffer:r->y offset:0 atIndex:2];
            [e setBytes:&in length:sizeof in atIndex:3];
            [e dispatchThreads:MTLSizeMake((NSUInteger)w->out, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(wide, 1, 1)];
            [e endEncoding];
            [commands commit];
            [commands waitUntilCompleted];
            memcpy(y, r->y.contents, (size_t)w->out * sizeof(float));
        } else {
            m->cpu.blocks(&m->cpu, w, x, y);
        }
    }
}

static int32_t span_of(const struct metal * m, const void * p) {
    const uintptr_t at = (uintptr_t)p;
    int32_t i = 0;
    while (i < m->count && (at < m->spans[i].base ||
                            at >= m->spans[i].base + m->spans[i].bytes)) {
        i++;
    }
    return p != NULL && i < m->count ? i : -1;
}

static bool known(const struct metal * m, const void * p) {
    return p == NULL || span_of(m, p) >= 0;
}

static void span_add(struct metal * m, id<MTLBuffer> buffer, uintptr_t base,
                     size_t bytes) {
    if (buffer != nil && m->count < SPANS) {
        m->spans[m->count++] = (struct span){
            .base = base, .bytes = bytes, .buffer = buffer
        };
    }
}

static void span_trim(struct metal * m, size_t need) {
    int32_t i = 0;
    while (i < m->count) {
        const struct span * s = &m->spans[i];
        if (s->idle && s->bytes < need && s->bytes >= need / 2) {
            m->spans[i] = m->spans[m->count - 1];
            m->spans[m->count - 1] = (struct span){ 0 };
            m->count--;
        } else {
            i++;
        }
    }
}

static float * metal_alloc(struct backend * backend, size_t count) {
    struct metal * m = (struct metal *)backend;
    const size_t need = ((count > 0 ? count : 1) * sizeof(float) + PAGE - 1) /
                        PAGE * PAGE;
    int32_t best = -1;
    for (int32_t i = 0; i < m->count; i++) {
        const struct span * s = &m->spans[i];
        const bool fits = s->idle && s->bytes >= need && s->bytes <= 2 * need;
        best = fits && (best < 0 || s->bytes < m->spans[best].bytes) ? i :
               best;
    }
    if (best < 0) {
        @autoreleasepool {
            span_trim(m, need);
            id<MTLBuffer> buffer = [m->device newBufferWithLength:need
                                       options:MTLResourceStorageModeShared];
            span_add(m, buffer, (uintptr_t)buffer.contents, need);
            best = buffer != nil && m->count > 0 &&
                   m->spans[m->count - 1].buffer == buffer ? m->count - 1 :
                   -1;
        }
    }
    if (best >= 0) {
        m->spans[best].idle = false;
    }
    return best >= 0 ? (float *)m->spans[best].base : floats(count);
}

static void metal_release(struct backend * backend, float * p) {
    struct metal * m = (struct metal *)backend;
    const int32_t i = span_of(m, p);
    if (i >= 0) {
        m->spans[i].idle = true;
    } else {
        free(p);
    }
}

static void metal_adopt(struct backend * backend, const void * data,
                        size_t bytes) {
    struct metal * m = (struct metal *)backend;
    const uintptr_t page = (uintptr_t)getpagesize();
    const uintptr_t first = (uintptr_t)data / page * page;
    const uintptr_t last = ((uintptr_t)data + bytes + page - 1) / page * page;
    const int32_t held = span_of(m, data);
    if (bytes > 0) {
        id<MTLBuffer> buffer = [m->device
            newBufferWithBytesNoCopy:(void *)first length:last - first
                             options:MTLResourceStorageModeShared
                         deallocator:nil];
        span_add(m, buffer, first, last - first);
    } else if (held >= 0) {
        metal_sync(backend);
        m->spans[held] = m->spans[m->count - 1];
        m->spans[m->count - 1] = (struct span){ 0 };
        m->count--;
    }
}

static id<MTLComputeCommandEncoder> encoder(struct metal * m) {
    if (m->commands == nil) {
        m->commands = [m->queue commandBuffer];
    }
    if (m->encoder == nil) {
        m->encoder = [m->commands computeCommandEncoder];
    }
    return m->encoder;
}

static void bind(struct metal * m, const void * p, const void * other,
                 NSUInteger index) {
    const void * at = p != NULL ? p : other;
    const struct span * s = &m->spans[span_of(m, at)];
    [m->encoder setBuffer:s->buffer offset:(uintptr_t)at - s->base
                  atIndex:index];
}

static void queued(struct metal * m) {
    m->ops++;
    if (m->ops >= BATCH) {
        [m->encoder endEncoding];
        m->encoder = nil;
        [m->commands commit];
        m->last = m->commands;
        m->commands = nil;
        m->ops = 0;
    }
}

static void metal_map(struct backend * backend, const struct map_args * a) {
    struct metal * m = (struct metal *)backend;
    const int32_t batch = a->batch > 1 ? a->batch : 1;
    const bool mine = known(m, a->y) && known(m, a->a) && known(m, a->b);
    struct shape s = {
        .pair = (int32_t)a->pair, .op = a->op, .alpha = a->alpha
    };
    for (int32_t i = 0; i < 3; i++) {
        s.ys[i] = (int32_t)a->ys[i];
        s.as[i] = (int32_t)a->as[i];
        s.bs[i] = (int32_t)a->bs[i];
    }
    @autoreleasepool {
        if (!mine) {
            metal_sync(backend);
            m->cpu.map(&m->cpu, a);
        } else if ((int64_t)batch * a->rows * a->cols > 0) {
            id<MTLComputeCommandEncoder> e = encoder(m);
            [e setComputePipelineState:m->maps];
            bind(m, a->y, NULL, 0);
            bind(m, a->a, a->y, 1);
            bind(m, a->b, a->y, 2);
            [e setBytes:&s length:sizeof s atIndex:3];
            [e dispatchThreads:MTLSizeMake((NSUInteger)a->cols,
                                           (NSUInteger)a->rows,
                                           (NSUInteger)batch)
                threadsPerThreadgroup:MTLSizeMake(a->cols < LANES ?
                                                  (NSUInteger)a->cols : LANES,
                                                  1, 1)];
            queued(m);
        }
    }
}

static void metal_norm(struct backend * backend, const struct norm_args * n) {
    struct metal * m = (struct metal *)backend;
    const bool mine = known(m, n->y) && known(m, n->x) && known(m, n->w) &&
                      known(m, n->b);
    const struct rule r = {
        .n = (uint32_t)n->n, .eps = n->eps, .center = n->b != NULL ? 1 : 0,
        .ld = n->ld > 0 ? n->ld : n->n, .step = n->step > 0 ? n->step : 1
    };
    @autoreleasepool {
        if (!mine) {
            metal_sync(backend);
            m->cpu.norm(&m->cpu, n);
        } else if (n->rows > 0) {
            id<MTLComputeCommandEncoder> e = encoder(m);
            [e setComputePipelineState:m->norms];
            bind(m, n->y, NULL, 0);
            bind(m, n->x, NULL, 1);
            bind(m, n->w, NULL, 2);
            bind(m, n->b, n->w, 3);
            [e setBytes:&r length:sizeof r atIndex:4];
            [e dispatchThreadgroups:MTLSizeMake((NSUInteger)n->rows, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            queued(m);
        }
    }
}

static void metal_softmax(struct backend * backend, float * x, int32_t rows,
                          int32_t n, float scale, int32_t causal) {
    struct metal * m = (struct metal *)backend;
    const struct soft s = {
        .n = (uint32_t)n, .scale = scale, .causal = (uint32_t)causal
    };
    @autoreleasepool {
        if (!known(m, x)) {
            metal_sync(backend);
            m->cpu.softmax(&m->cpu, x, rows, n, scale, causal);
        } else if (rows > 0) {
            id<MTLComputeCommandEncoder> e = encoder(m);
            [e setComputePipelineState:m->softs];
            bind(m, x, NULL, 0);
            [e setBytes:&s length:sizeof s atIndex:1];
            [e dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            queued(m);
        }
    }
}

static MPSMatrix * matrix(struct metal * m, const float * p, int32_t rows,
                          int32_t cols, int32_t ld, int32_t batch,
                          int64_t stride) {
    const struct span * s = &m->spans[span_of(m, p)];
    const size_t row = (size_t)ld * sizeof(float);
    MPSMatrixDescriptor * d = [MPSMatrixDescriptor
        matrixDescriptorWithRows:(NSUInteger)rows columns:(NSUInteger)cols
                        matrices:(NSUInteger)batch rowBytes:row
                     matrixBytes:batch > 1 ? (size_t)stride * sizeof(float) :
                                 (size_t)rows * row
                        dataType:MPSDataTypeFloat32];
    return [[MPSMatrix alloc] initWithBuffer:s->buffer
                                      offset:(uintptr_t)p - s->base
                                  descriptor:d];
}

static bool tiled(const struct metal * m, const struct gemm_args * g) {
    return (g->format != W_F32 || g->tc || m->own) && g->tb &&
           g->batch <= 1 &&
           g->k % 32 == 0 && g->m >= 8 && g->n >= 8 &&
           (g->format != W_Q4_K || g->k % 256 == 0);
}

static void metal_tile(struct metal * m, const struct gemm_args * g,
                       int32_t parts) {
    const struct tile t = {
        .m = (uint32_t)g->m, .n = (uint32_t)g->n, .k = (uint32_t)g->k,
        .ldx = (uint32_t)g->lda,
        .ldw = (uint32_t)(g->format == W_F32 ? g->ldb * 4 : g->ldb),
        .ldc = (uint32_t)g->ldc, .ta = g->ta, .tc = g->tc,
        .fresh = g->fresh, .parts = (uint32_t)parts,
        .slab = (uint32_t)((g->tc ? g->n : g->m) * g->ldc)
    };
    id<MTLComputeCommandEncoder> e = encoder(m);
    [e setComputePipelineState:m->tiles[g->format]];
    bind(m, g->a, NULL, 0);
    bind(m, g->b, NULL, 1);
    bind(m, g->c, NULL, 2);
    [e setBytes:&t length:sizeof t atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake((NSUInteger)(g->n + 63) / 64,
                                        (NSUInteger)(g->m + 63) / 64,
                                        (NSUInteger)parts)
      threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    queued(m);
}

static bool torn(const struct gemm_args * g) {
    return !g->fresh && ((g->n % 64 > 0 && g->n % 64 < 8) ||
                         (g->m % 64 > 0 && g->m % 64 < 8));
}

static int32_t parts_of(const struct gemm_args * g) {
    const int64_t groups = (int64_t)((g->n + 63) / 64) * ((g->m + 63) / 64);
    const int32_t unit = g->format == W_Q4_K ? 256 : 32;
    return groups <= FEW && g->k >= DEEP && g->k % (SPLIT * unit) == 0 ?
           SPLIT : 1;
}

static void metal_tile_apart(struct metal * m, const struct gemm_args * g,
                             int32_t parts) {
    const int32_t rows = g->tc ? g->n : g->m;
    const int32_t cols = g->tc ? g->m : g->n;
    const size_t slab = (size_t)rows * cols;
    float * product = metal_alloc(&m->backend, slab * (size_t)parts);
    struct gemm_args fresh = *g;
    fresh.c = product;
    fresh.ldc = cols;
    fresh.fresh = true;
    metal_tile(m, &fresh, parts);
    for (int32_t p = 0; p < parts; p++) {
        const struct map_args sum = {
            .y = g->c, .a = product + (size_t)p * slab,
            .op = p == 0 && g->fresh ? OP_COPY : OP_ADD, .alpha = 1.0f,
            .rows = rows, .cols = cols, .ys = { 0, g->ldc, 1 },
            .as = { 0, cols, 1 }
        };
        metal_map(&m->backend, &sum);
    }
    metal_release(&m->backend, product);
}

static void metal_gemm(struct backend * backend, const struct gemm_args * g) {
    struct metal * m = (struct metal *)backend;
    const int32_t batch = g->batch > 1 ? g->batch : 1;
    const bool mine = known(m, g->a) && known(m, g->b) && known(m, g->c) &&
                      ((g->format == W_F32 && !g->tc) || tiled(m, g));
    @autoreleasepool {
        if (!mine) {
            metal_sync(backend);
            m->cpu.gemm(&m->cpu, g);
        } else if (tiled(m, g) && (torn(g) || parts_of(g) > 1)) {
            metal_tile_apart(m, g, parts_of(g));
        } else if (tiled(m, g)) {
            metal_tile(m, g, 1);
        } else if ((int64_t)g->m * g->n * g->k > 0) {
            MPSMatrix * a = g->ta ?
                matrix(m, g->a, g->k, g->m, g->lda, batch, g->sa) :
                matrix(m, g->a, g->m, g->k, g->lda, batch, g->sa);
            MPSMatrix * b = g->tb ?
                matrix(m, g->b, g->n, g->k, g->ldb, batch, g->sb) :
                matrix(m, g->b, g->k, g->n, g->ldb, batch, g->sb);
            MPSMatrix * c = matrix(m, g->c, g->m, g->n, g->ldc, batch,
                                   g->sc);
            MPSMatrixMultiplication * mm = [[MPSMatrixMultiplication alloc]
                initWithDevice:m->device transposeLeft:g->ta
                transposeRight:g->tb resultRows:(NSUInteger)g->m
                 resultColumns:(NSUInteger)g->n
               interiorColumns:(NSUInteger)g->k alpha:1.0
                          beta:g->fresh ? 0.0 : 1.0];
            mm.batchSize = (NSUInteger)batch;
            encoder(m);
            [m->encoder endEncoding];
            m->encoder = nil;
            [mm encodeToCommandBuffer:m->commands leftMatrix:a
                          rightMatrix:b resultMatrix:c];
            queued(m);
        }
    }
}

static void metal_close(struct backend * backend) {
    struct metal * m = (struct metal *)backend;
    @autoreleasepool {
        metal_sync(backend);
        *m = (struct metal){ 0 };
    }
    free(m);
}

static id<MTLComputePipelineState> pipeline(struct metal * m,
                                            id<MTLLibrary> library,
                                            NSString * name) {
    return [m->device newComputePipelineStateWithFunction:
            [library newFunctionWithName:name] error:nil];
}

static struct backend * metal_open(const char * const * options) {
    struct metal * m = calloc(1, sizeof *m);
    struct backend * blas = backend_open("blas", options);
    if (m != NULL && blas != NULL) {
        m->backend = *blas;
        m->cpu = *blas;
        m->backend.name = "metal";
        m->backend.queued = true;
        m->backend.blocks = metal_blocks;
        m->backend.gemm = metal_gemm;
        m->backend.map = metal_map;
        m->backend.norm = metal_norm;
        m->backend.softmax = metal_softmax;
        m->backend.alloc = metal_alloc;
        m->backend.adopt = metal_adopt;
        m->backend.release = metal_release;
        m->backend.sync = metal_sync;
        m->backend.close = metal_close;
        m->device = MTLCreateSystemDefaultDevice();
        m->queue = [m->device newCommandQueue];
        id<MTLLibrary> library = [m->device newLibraryWithSource:@(source)
                                                         options:nil
                                                           error:nil];
        m->state = pipeline(m, library, @"blocks");
        m->maps = pipeline(m, library, @"map");
        m->norms = pipeline(m, library, @"norm");
        m->softs = pipeline(m, library, @"soften");
        NSMutableString * text = [NSMutableString string];
        for (int32_t i = 0; products[i] != NULL; i++) {
            [text appendString:@(products[i])];
        }
        id<MTLLibrary> more = [m->device newLibraryWithSource:text options:nil
                                                        error:nil];
        for (int32_t f = 0; f < FORMATS; f++) {
            m->tiles[f] = pipeline(m, more, @(product_names[f]));
        }
        m->own = strcmp(option(options, "kernel", "mps"), "own") == 0;
    }
    if (blas != NULL) {
        blas->close(blas);
    }
    if (m != NULL && (m->state == nil || m->maps == nil || m->norms == nil ||
                      m->softs == nil || m->tiles[FORMATS - 1] == nil)) {
        metal_close(&m->backend);
        m = NULL;
    }
    return m != NULL ? &m->backend : NULL;
}

INSTALL(backend, metal, .keys = "kernel", .open = metal_open)
