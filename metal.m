#include "backend.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#import <Metal/Metal.h>

enum { WIDE = 32768, REGIONS = 4 };

#define METAL(...) #__VA_ARGS__

static const char * const source = METAL(
    using namespace metal;
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
);

struct region {
    const uint8_t * codes;
    id<MTLBuffer> weights;
    id<MTLBuffer> x;
    id<MTLBuffer> y;
};

struct metal {
    struct backend backend;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> state;
    struct region regions[REGIONS];
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

static void metal_blocks(struct backend * backend, const struct blocks * w,
                         const float * x, float * y) {
    struct metal * m = (struct metal *)backend;
    struct region * r = w->out >= WIDE ? region_of(m, w) : NULL;
    const uint32_t in = (uint32_t)w->in;
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
        cpu_blocks(backend, w, x, y);
    }
}

static void metal_close(struct backend * backend) {
    struct metal * m = (struct metal *)backend;
    *m = (struct metal){ 0 };
    free(m);
}

static struct backend * metal_open(const char * const * options) {
    struct metal * m = calloc(1, sizeof *m);
    struct backend * blas = backend_open("blas", options);
    if (m != NULL && blas != NULL) {
        m->backend = *blas;
        m->backend.name = "metal";
        m->backend.blocks = metal_blocks;
        m->backend.close = metal_close;
        m->device = MTLCreateSystemDefaultDevice();
        m->queue = [m->device newCommandQueue];
        id<MTLLibrary> library = [m->device newLibraryWithSource:@(source)
                                                         options:nil
                                                           error:nil];
        m->state = [m->device newComputePipelineStateWithFunction:
                    [library newFunctionWithName:@"blocks"] error:nil];
    }
    if (blas != NULL) {
        blas->close(blas);
    }
    if (m != NULL && m->state == nil) {
        metal_close(&m->backend);
        m = NULL;
    }
    return m != NULL ? &m->backend : NULL;
}

INSTALL(backend, metal, .keys = "", .open = metal_open)
