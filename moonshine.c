#include "asr.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { HIDDEN = 288, HEADS = 8, HEAD = 36, PAIRS = 16, WIDE = 1152,
       LAYERS = 6, VOCAB = 32768, PIECES = 32000, TOKENS = 448,
       RATE = 16000, SECONDS = 30, NAME = 96, BOS = 1, EOS = 2 };

struct attention {
    const float * q;
    const float * k;
    const float * v;
    const float * o;
};

struct layer {
    const float * norm1;
    const float * norm2;
    const float * norm3;
    struct attention self;
    struct attention cross;
    const float * fc1;
    const float * fc1_bias;
    const float * fc2;
    const float * fc2_bias;
    float * keys;
    float * values;
};

struct moonshine {
    struct asr asr;
    struct backend * backend;
    const uint8_t * file;
    size_t size;
    const char * header;
    size_t header_size;
    const uint8_t * vocabulary;
    size_t vocabulary_size;
    const float * conv1;
    const float * conv2;
    const float * conv2_bias;
    const float * conv3;
    const float * conv3_bias;
    const float * group_gain;
    const float * group_bias;
    const float * encoder_norm;
    const float * decoder_norm;
    const float * embed;
    struct layer encoder[LAYERS];
    struct layer decoder[LAYERS];
    char * pool;
    int32_t piece_at[VOCAB];
    int32_t piece_size[VOCAB];
    float rate[PAIRS];
    int32_t faults;
};

static const float * tensor(struct moonshine * self, const char * format,
                            int32_t layer, size_t count) {
    char name[NAME];
    char key[NAME + 8];
    char * next = NULL;
    snprintf(name, sizeof name, format, layer);
    const int size = snprintf(key, sizeof key, "\"%s\":{", name);
    const char * at = memmem(self->header, self->header_size, key,
                             (size_t)size);
    const char * end = at != NULL ?
        memchr(at + size, '}', self->header_size -
               (size_t)(at + size - self->header)) : NULL;
    const char * span = end != NULL ?
        memmem(at, (size_t)(end - at), "\"data_offsets\":[", 16) : NULL;
    const long long from = span != NULL ? strtoll(span + 16, &next, 10) : 0;
    const long long to = span != NULL ? strtoll(next + 1, NULL, 10) : 0;
    const float * data = (const float *)(const void *)
        (self->file + 8 + self->header_size + from);
    const bool fine = span != NULL &&
                      (size_t)(to - from) == count * sizeof(float) &&
                      ((uintptr_t)data & 3) == 0 &&
                      memmem(at, (size_t)(end - at), "\"dtype\":\"F32\"",
                             13) != NULL;
    self->faults += fine ? 0 : 1;
    return fine ? data : NULL;
}

static struct attention attention_at(struct moonshine * self,
                                     const char * side, const char * kind,
                                     int32_t layer) {
    const size_t count = (size_t)HIDDEN * HIDDEN;
    char format[NAME];
    struct attention a = { 0 };
    snprintf(format, sizeof format, "model.%s.layers.%%d.%s.q_proj.weight",
             side, kind);
    a.q = tensor(self, format, layer, count);
    snprintf(format, sizeof format, "model.%s.layers.%%d.%s.k_proj.weight",
             side, kind);
    a.k = tensor(self, format, layer, count);
    snprintf(format, sizeof format, "model.%s.layers.%%d.%s.v_proj.weight",
             side, kind);
    a.v = tensor(self, format, layer, count);
    snprintf(format, sizeof format, "model.%s.layers.%%d.%s.o_proj.weight",
             side, kind);
    a.o = tensor(self, format, layer, count);
    return a;
}

static void encoder_open(struct moonshine * self, int32_t l) {
    const char * base = "model.encoder.layers.%d.";
    char format[NAME];
    struct layer * layer = &self->encoder[l];
    snprintf(format, sizeof format, "%sinput_layernorm.weight", base);
    layer->norm1 = tensor(self, format, l, HIDDEN);
    snprintf(format, sizeof format, "%spost_attention_layernorm.weight",
             base);
    layer->norm2 = tensor(self, format, l, HIDDEN);
    layer->self = attention_at(self, "encoder", "self_attn", l);
    snprintf(format, sizeof format, "%smlp.fc1.weight", base);
    layer->fc1 = tensor(self, format, l, (size_t)WIDE * HIDDEN);
    snprintf(format, sizeof format, "%smlp.fc1.bias", base);
    layer->fc1_bias = tensor(self, format, l, WIDE);
    snprintf(format, sizeof format, "%smlp.fc2.weight", base);
    layer->fc2 = tensor(self, format, l, (size_t)HIDDEN * WIDE);
    snprintf(format, sizeof format, "%smlp.fc2.bias", base);
    layer->fc2_bias = tensor(self, format, l, HIDDEN);
}

static void decoder_open(struct moonshine * self, int32_t l) {
    const char * base = "model.decoder.layers.%d.";
    char format[NAME];
    struct layer * layer = &self->decoder[l];
    snprintf(format, sizeof format, "%sinput_layernorm.weight", base);
    layer->norm1 = tensor(self, format, l, HIDDEN);
    snprintf(format, sizeof format, "%spost_attention_layernorm.weight",
             base);
    layer->norm2 = tensor(self, format, l, HIDDEN);
    snprintf(format, sizeof format, "%sfinal_layernorm.weight", base);
    layer->norm3 = tensor(self, format, l, HIDDEN);
    layer->self = attention_at(self, "decoder", "self_attn", l);
    layer->cross = attention_at(self, "decoder", "encoder_attn", l);
    snprintf(format, sizeof format, "%smlp.fc1.weight", base);
    layer->fc1 = tensor(self, format, l, (size_t)2 * WIDE * HIDDEN);
    snprintf(format, sizeof format, "%smlp.fc1.bias", base);
    layer->fc1_bias = tensor(self, format, l, 2 * WIDE);
    snprintf(format, sizeof format, "%smlp.fc2.weight", base);
    layer->fc2 = tensor(self, format, l, (size_t)HIDDEN * WIDE);
    snprintf(format, sizeof format, "%smlp.fc2.bias", base);
    layer->fc2_bias = tensor(self, format, l, HIDDEN);
    layer->keys = floats((size_t)(TOKENS + 1) * HIDDEN);
    layer->values = floats((size_t)(TOKENS + 1) * HIDDEN);
}

static const char * blank(const char * p, const char * end) {
    const char * at = p;
    while (at < end && (*at == ' ' || *at == '\n' || *at == '\r' ||
                        *at == '\t')) {
        at++;
    }
    return at;
}

static const char * after(const char * p, const char * end, char c) {
    const char * at = blank(p, end);
    return at < end && *at == c ? blank(at + 1, end) : end;
}

static void pieces_load(struct moonshine * self) {
    const char * text = (const char *)self->vocabulary;
    const char * end = text + self->vocabulary_size;
    const char * key = memmem(text, self->vocabulary_size, "\"vocab\"", 7);
    const char * p = key != NULL ? after(after(key + 7, end, ':'), end, '{') :
                     end;
    int32_t used = 0;
    int32_t loaded = 0;
    while (p < end && *p == '"') {
        const char * close = json_close(p + 1, end);
        char * tail = NULL;
        const char * value = close != NULL ? after(close + 1, end, ':') : end;
        const long id = value < end ? strtol(value, &tail, 10) : -1;
        if (id >= 0 && id < VOCAB &&
            json_unescape(p + 1, close, self->pool + used,
                          self->vocabulary_size - (size_t)used)) {
            self->piece_at[id] = used;
            self->piece_size[id] = (int32_t)strlen(self->pool + used);
            used += self->piece_size[id] + 1;
            loaded++;
        }
        p = tail != NULL ? blank(tail, end) : end;
        p = p < end && *p == ',' ? blank(p + 1, end) : p;
    }
    self->faults += loaded >= PIECES ? 0 : 1;
}

static void linear(struct moonshine * self, float * y, const float * x,
                   const float * w, const float * bias, int32_t rows,
                   int32_t out, int32_t in) {
    const struct gemm_args g = {
        .a = x, .b = w, .c = y, .m = rows, .n = out, .k = in, .lda = in,
        .ldb = in, .ldc = out, .tb = true
    };
    for (int32_t r = 0; r < rows; r++) {
        for (int32_t o = 0; o < out; o++) {
            y[(size_t)r * out + o] = bias != NULL ? bias[o] : 0.0f;
        }
    }
    self->backend->gemm(self->backend, &g);
}

static float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x / 1.4142135f));
}

static float dot(const float * restrict a, const float * restrict b,
                 int32_t n) {
    #pragma clang fp reassociate(on)
    float sum = 0.0f;
    for (int32_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

static void layer_norm(float * y, const float * x, const float * gain,
                       int32_t rows) {
    for (int32_t r = 0; r < rows; r++) {
        const float * in = x + (size_t)r * HIDDEN;
        float * out = y + (size_t)r * HIDDEN;
        float mean = 0.0f;
        float spread = 0.0f;
        for (int32_t i = 0; i < HIDDEN; i++) {
            mean += in[i];
        }
        mean /= (float)HIDDEN;
        for (int32_t i = 0; i < HIDDEN; i++) {
            spread += (in[i] - mean) * (in[i] - mean);
        }
        const float scale = 1.0f / sqrtf(spread / (float)HIDDEN + 1e-5f);
        for (int32_t i = 0; i < HIDDEN; i++) {
            out[i] = (in[i] - mean) * scale * gain[i];
        }
    }
}

static void group_norm(const struct moonshine * self, float * x,
                       int32_t rows) {
    const size_t n = (size_t)rows * HIDDEN;
    float mean = 0.0f;
    float spread = 0.0f;
    for (size_t i = 0; i < n; i++) {
        mean += x[i];
    }
    mean /= (float)n;
    for (size_t i = 0; i < n; i++) {
        spread += (x[i] - mean) * (x[i] - mean);
    }
    const float scale = 1.0f / sqrtf(spread / (float)n + 1e-5f);
    for (size_t i = 0; i < n; i++) {
        x[i] = (x[i] - mean) * scale * self->group_gain[i % HIDDEN] +
               self->group_bias[i % HIDDEN];
    }
}

static void rotate(const struct moonshine * self, float * x,
                   int32_t position) {
    for (int32_t j = 0; j < PAIRS; j++) {
        const float c = cosf((float)position * self->rate[j]);
        const float s = sinf((float)position * self->rate[j]);
        for (int32_t h = 0; h < HEADS; h++) {
            float * p = x + h * HEAD + 2 * j;
            const float a = p[0];
            p[0] = a * c - p[1] * s;
            p[1] = a * s + p[1] * c;
        }
    }
}

static void attend(float * out, const float * q, const float * keys,
                   const float * values, int32_t count) {
    float * score = floats((size_t)count);
    for (int32_t h = 0; h < HEADS; h++) {
        float * o = out + h * HEAD;
        float top = -INFINITY;
        float sum = 0.0f;
        for (int32_t s = 0; s < count; s++) {
            score[s] = dot(q + h * HEAD, keys + (size_t)s * HIDDEN + h * HEAD,
                           HEAD) / sqrtf((float)HEAD);
            top = fmaxf(top, score[s]);
        }
        for (int32_t s = 0; s < count; s++) {
            score[s] = expf(score[s] - top);
            sum += score[s];
        }
        memset(o, 0, HEAD * sizeof(float));
        for (int32_t s = 0; s < count; s++) {
            const float * from = values + (size_t)s * HIDDEN + h * HEAD;
            const float w = score[s] / sum;
            for (int32_t d = 0; d < HEAD; d++) {
                o[d] += w * from[d];
            }
        }
    }
    free(score);
}

static int32_t convolve(struct moonshine * self, float * y, const float * x,
                        const float * w, const float * bias, int32_t rows,
                        int32_t in, int32_t out, int32_t taps,
                        int32_t stride) {
    const int32_t made = (rows - taps) / stride + 1;
    float * patch = floats((size_t)made * in * taps);
    for (int32_t t = 0; t < made; t++) {
        for (int32_t i = 0; i < in; i++) {
            for (int32_t j = 0; j < taps; j++) {
                patch[((size_t)t * in + i) * taps + j] =
                    x[((size_t)t * stride + j) * in + i];
            }
        }
    }
    linear(self, y, patch, w, bias, made, out, in * taps);
    free(patch);
    return made;
}

static float * front(struct moonshine * self, const float * wave, int32_t n,
                     int32_t * frames) {
    const int32_t m1 = (n - 127) / 64 + 1;
    const int32_t m2 = (m1 - 7) / 3 + 1;
    const int32_t m3 = (m2 - 3) / 2 + 1;
    float * a = floats((size_t)m1 * HIDDEN);
    float * b = floats((size_t)m2 * 2 * HIDDEN);
    float * x = floats((size_t)m3 * HIDDEN);
    convolve(self, a, wave, self->conv1, NULL, n, 1, HIDDEN, 127, 64);
    for (size_t i = 0; i < (size_t)m1 * HIDDEN; i++) {
        a[i] = tanhf(a[i]);
    }
    group_norm(self, a, m1);
    convolve(self, b, a, self->conv2, self->conv2_bias, m1, HIDDEN,
             2 * HIDDEN, 7, 3);
    for (size_t i = 0; i < (size_t)m2 * 2 * HIDDEN; i++) {
        b[i] = gelu(b[i]);
    }
    convolve(self, x, b, self->conv3, self->conv3_bias, m2, 2 * HIDDEN,
             HIDDEN, 3, 2);
    for (size_t i = 0; i < (size_t)m3 * HIDDEN; i++) {
        x[i] = gelu(x[i]);
    }
    free(a);
    free(b);
    *frames = m3;
    return x;
}

static void encoder_layer(struct moonshine * self, float * x,
                          const struct layer * l, int32_t rows) {
    const size_t n = (size_t)rows * HIDDEN;
    float * a = floats(n);
    float * q = floats(n);
    float * k = floats(n);
    float * v = floats(n);
    float * o = floats(n);
    float * h = floats((size_t)rows * WIDE);
    layer_norm(a, x, l->norm1, rows);
    linear(self, q, a, l->self.q, NULL, rows, HIDDEN, HIDDEN);
    linear(self, k, a, l->self.k, NULL, rows, HIDDEN, HIDDEN);
    linear(self, v, a, l->self.v, NULL, rows, HIDDEN, HIDDEN);
    for (int32_t t = 0; t < rows; t++) {
        rotate(self, q + (size_t)t * HIDDEN, t);
        rotate(self, k + (size_t)t * HIDDEN, t);
    }
    for (int32_t t = 0; t < rows; t++) {
        attend(a + (size_t)t * HIDDEN, q + (size_t)t * HIDDEN, k, v, rows);
    }
    linear(self, o, a, l->self.o, NULL, rows, HIDDEN, HIDDEN);
    for (size_t i = 0; i < n; i++) {
        x[i] += o[i];
    }
    layer_norm(a, x, l->norm2, rows);
    linear(self, h, a, l->fc1, l->fc1_bias, rows, WIDE, HIDDEN);
    for (size_t i = 0; i < (size_t)rows * WIDE; i++) {
        h[i] = gelu(h[i]);
    }
    linear(self, o, h, l->fc2, l->fc2_bias, rows, HIDDEN, WIDE);
    for (size_t i = 0; i < n; i++) {
        x[i] += o[i];
    }
    free(a);
    free(q);
    free(k);
    free(v);
    free(o);
    free(h);
}

static void add(float * x, const float * o) {
    for (int32_t i = 0; i < HIDDEN; i++) {
        x[i] += o[i];
    }
}

static void decoder_layer(struct moonshine * self, float * x,
                          const struct layer * l, int32_t at,
                          const float * memory_k, const float * memory_v,
                          int32_t frames) {
    float * key = l->keys + (size_t)at * HIDDEN;
    float a[HIDDEN];
    float q[HIDDEN];
    float mix[HIDDEN];
    float o[HIDDEN];
    float u[2 * WIDE];
    layer_norm(a, x, l->norm1, 1);
    linear(self, q, a, l->self.q, NULL, 1, HIDDEN, HIDDEN);
    linear(self, key, a, l->self.k, NULL, 1, HIDDEN, HIDDEN);
    linear(self, l->values + (size_t)at * HIDDEN, a, l->self.v, NULL, 1,
           HIDDEN, HIDDEN);
    rotate(self, q, at);
    rotate(self, key, at);
    attend(mix, q, l->keys, l->values, at + 1);
    linear(self, o, mix, l->self.o, NULL, 1, HIDDEN, HIDDEN);
    add(x, o);
    layer_norm(a, x, l->norm2, 1);
    linear(self, q, a, l->cross.q, NULL, 1, HIDDEN, HIDDEN);
    attend(mix, q, memory_k, memory_v, frames);
    linear(self, o, mix, l->cross.o, NULL, 1, HIDDEN, HIDDEN);
    add(x, o);
    layer_norm(a, x, l->norm3, 1);
    linear(self, u, a, l->fc1, l->fc1_bias, 1, 2 * WIDE, HIDDEN);
    for (int32_t i = 0; i < WIDE; i++) {
        u[i] *= u[WIDE + i] / (1.0f + expf(-u[WIDE + i]));
    }
    linear(self, o, u, l->fc2, l->fc2_bias, 1, HIDDEN, WIDE);
    add(x, o);
}

static int32_t best(const float * logits) {
    int32_t pick = 0;
    for (int32_t i = 1; i < VOCAB; i++) {
        pick = logits[i] > logits[pick] ? i : pick;
    }
    return pick;
}

static bool word_mark(const uint8_t * p, int32_t i, int32_t n) {
    return i + 2 < n && p[i] == 0xE2 && p[i + 1] == 0x96 && p[i + 2] == 0x81;
}

static int32_t piece_write(const struct moonshine * self, char * text,
                           int32_t used, int32_t capacity, int32_t token) {
    const uint8_t * p = (const uint8_t *)self->pool + self->piece_at[token];
    const int32_t n = token > EOS ? self->piece_size[token] : 0;
    const bool byte = n == 6 && p[0] == '<' && p[1] == '0' && p[2] == 'x';
    int32_t at = used;
    if (byte && at + 1 < capacity) {
        text[at++] = (char)strtol((const char *)p + 3, NULL, 16);
    }
    for (int32_t i = 0; !byte && i < n && at + 1 < capacity; i++) {
        const bool space = word_mark(p, i, n);
        if (!space) {
            text[at++] = (char)p[i];
        } else if (at > 0) {
            text[at++] = ' ';
        }
        i += space ? 2 : 0;
    }
    return at;
}

static int32_t decode(struct moonshine * self, const float * memory,
                      int32_t frames, int32_t limit, char * text,
                      int32_t capacity) {
    const size_t n = (size_t)frames * HIDDEN;
    float * memory_k = floats(LAYERS * n);
    float * memory_v = floats(LAYERS * n);
    float * logits = floats(VOCAB);
    float x[HIDDEN];
    int32_t token = BOS;
    int32_t count = 0;
    int32_t size = 0;
    for (int32_t l = 0; l < LAYERS; l++) {
        linear(self, memory_k + l * n, memory, self->decoder[l].cross.k,
               NULL, frames, HIDDEN, HIDDEN);
        linear(self, memory_v + l * n, memory, self->decoder[l].cross.v,
               NULL, frames, HIDDEN, HIDDEN);
    }
    while (count < limit && token != EOS) {
        memcpy(x, self->embed + (size_t)token * HIDDEN, sizeof x);
        for (int32_t l = 0; l < LAYERS; l++) {
            decoder_layer(self, x, &self->decoder[l], count,
                          memory_k + l * n, memory_v + l * n, frames);
        }
        layer_norm(x, x, self->decoder_norm, 1);
        linear(self, logits, x, self->embed, NULL, 1, VOCAB, HIDDEN);
        token = best(logits);
        size = piece_write(self, text, size, capacity, token);
        count += token != EOS ? 1 : 0;
    }
    text[size] = 0;
    free(memory_k);
    free(memory_v);
    free(logits);
    return count;
}

static int32_t moonshine_transcribe(struct asr * asr, const float * samples,
                                    int32_t count, const char * language,
                                    char * text, int32_t capacity) {
    struct moonshine * self = (struct moonshine *)asr;
    const int32_t n = count < RATE * SECONDS ? count : RATE * SECONDS;
    const int32_t wanted = n / RATE * 13 / 2 + 8;
    int32_t frames = 0;
    int32_t made = 0;
    (void)language;
    text[0] = 0;
    if (n >= RATE / 10) {
        float * x = front(self, samples, n, &frames);
        for (int32_t l = 0; l < LAYERS; l++) {
            encoder_layer(self, x, &self->encoder[l], frames);
        }
        layer_norm(x, x, self->encoder_norm, frames);
        made = decode(self, x, frames, wanted < TOKENS ? wanted : TOKENS,
                      text, capacity);
        free(x);
    }
    return made;
}

static void moonshine_close(struct asr * asr) {
    struct moonshine * self = (struct moonshine *)asr;
    for (int32_t l = 0; l < LAYERS; l++) {
        free(self->decoder[l].keys);
        free(self->decoder[l].values);
    }
    free(self->pool);
    file_unmap(self->vocabulary, self->vocabulary_size);
    file_unmap(self->file, self->size);
    free(self);
}

static void moonshine_load(struct moonshine * self) {
    const size_t square = (size_t)HIDDEN * HIDDEN;
    self->conv1 = tensor(self, "model.encoder.conv1.weight", 0, 127 * HIDDEN);
    self->group_gain = tensor(self, "model.encoder.groupnorm.weight", 0,
                              HIDDEN);
    self->group_bias = tensor(self, "model.encoder.groupnorm.bias", 0,
                              HIDDEN);
    self->conv2 = tensor(self, "model.encoder.conv2.weight", 0,
                         2 * square * 7);
    self->conv2_bias = tensor(self, "model.encoder.conv2.bias", 0,
                              2 * HIDDEN);
    self->conv3 = tensor(self, "model.encoder.conv3.weight", 0,
                         2 * square * 3);
    self->conv3_bias = tensor(self, "model.encoder.conv3.bias", 0, HIDDEN);
    self->encoder_norm = tensor(self, "model.encoder.layer_norm.weight", 0,
                                HIDDEN);
    self->decoder_norm = tensor(self, "model.decoder.norm.weight", 0, HIDDEN);
    self->embed = tensor(self, "model.decoder.embed_tokens.weight", 0,
                         (size_t)VOCAB * HIDDEN);
    for (int32_t l = 0; l < LAYERS; l++) {
        encoder_open(self, l);
        decoder_open(self, l);
    }
    for (int32_t j = 0; j < PAIRS; j++) {
        self->rate[j] = 1.0f / powf(10000.0f, (float)(2 * j) / 32.0f);
    }
    pieces_load(self);
}

static bool moonshine_map(struct moonshine * self, const char * path) {
    const char * tail = "tokenizer.json";
    const char * slash = strrchr(path, '/');
    const int folder = slash != NULL ? (int)(slash - path) + 1 : 0;
    char side[512];
    snprintf(side, sizeof side, "%.*s%s", folder, path, tail);
    self->file = file_map(path, &self->size);
    self->vocabulary = file_map(side, &self->vocabulary_size);
    self->header_size = self->file != NULL && self->size > 8 ?
                        (size_t)le64(self->file) : 0;
    self->header = (const char *)self->file + 8;
    self->pool = self->vocabulary != NULL ?
                 malloc(self->vocabulary_size + 1) : NULL;
    return self->pool != NULL && self->header_size > 0 &&
           self->header_size < self->size - 8;
}

static struct asr * moonshine_open(const char * path,
                                   struct backend * backend,
                                   const char * const * options) {
    struct moonshine * self = calloc(1, sizeof *self);
    (void)options;
    if (self != NULL) {
        self->asr = (struct asr){
            .name = "moonshine", .transcribe = moonshine_transcribe,
            .close = moonshine_close
        };
        self->backend = backend;
    }
    const bool mapped = self != NULL && moonshine_map(self, path);
    if (mapped) {
        moonshine_load(self);
    }
    const bool ready = mapped && self->faults == 0;
    if (!ready && self != NULL) {
        moonshine_close(&self->asr);
    }
    return ready ? &self->asr : NULL;
}

INSTALL(asr, moonshine,
        .pack = "leok7v/moonshine-tiny/model.safetensors", .keys = "",
        .open = moonshine_open)

#ifndef NO_MAIN

int main(int argc, char ** argv) {
    (void)argc;
    return asr_program("moonshine", (const char * const *)argv + 1);
}

#endif
