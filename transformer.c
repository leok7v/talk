#include "transformer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { WORD = 256, NAME = 64, STRING = 8, ARRAY = 9 };

struct span {
    int32_t at;
    int32_t size;
};

struct ids {
    int32_t * out;
    int32_t capacity;
    int32_t count;
};

static uint32_t hash(const uint8_t * key, int32_t size) {
    uint32_t h = 2166136261u;
    for (int32_t i = 0; i < size; i++) {
        h = (h ^ key[i]) * 16777619u;
    }
    return h;
}

static int32_t table_slot(const struct table * t, const uint8_t * key,
                          int32_t size) {
    uint32_t slot = hash(key, size) % t->slots;
    while (t->key[slot] != NULL && (t->size[slot] != size ||
           memcmp(t->key[slot], key, (size_t)size) != 0)) {
        slot = (slot + 1) % t->slots;
    }
    return (int32_t)slot;
}

static int32_t table_find(const struct table * t, const uint8_t * key,
                          int32_t size) {
    const int32_t slot = table_slot(t, key, size);
    return t->key[slot] != NULL ? t->value[slot] : -1;
}

static void table_put(struct table * t, const uint8_t * key, int32_t size,
                      int32_t value) {
    const int32_t slot = table_slot(t, key, size);
    if (t->key[slot] == NULL) {
        t->key[slot] = key;
        t->size[slot] = size;
        t->value[slot] = value;
    }
}

static struct table table_open(uint64_t count) {
    uint32_t slots = 1024;
    while (slots < 2 * count + 2) {
        slots *= 2;
    }
    struct table t = {
        .key = calloc(slots, sizeof(uint8_t *)),
        .size = calloc(slots, sizeof(int32_t)),
        .value = calloc(slots, sizeof(int32_t)), .slots = slots
    };
    must(t.key != NULL && t.size != NULL && t.value != NULL, "memory");
    return t;
}

static void table_close(struct table * t) {
    free(t->key);
    free(t->size);
    free(t->value);
}

static bool key_is(const uint8_t * key, uint64_t size, const char * name) {
    return size == strlen(name) && memcmp(key, name, size) == 0;
}

static void value_skip(struct gguf * g, uint32_t type);

static void array_skip(struct gguf * g) {
    const uint32_t type = le32(g->cursor);
    const uint64_t count = le64(g->cursor + 4);
    g->cursor += 12;
    for (uint64_t i = 0; i < count; i++) {
        value_skip(g, type);
    }
}

static void value_skip(struct gguf * g, uint32_t type) {
    const int32_t sizes[] = { 1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8 };
    if (type == STRING) {
        g->cursor += 8 + le64(g->cursor);
    } else if (type == ARRAY) {
        array_skip(g);
    } else {
        g->cursor += sizes[type];
    }
}

static void tokens_read(struct gguf * g) {
    const uint64_t count = le64(g->cursor + 4);
    const uint8_t * p = g->cursor + 12;
    g->vocab_size = (int32_t)count;
    g->vocab = table_open(count);
    g->token_text = calloc(count + 1, sizeof(uint8_t *));
    g->token_size = calloc(count + 1, sizeof(int32_t));
    must(g->token_text != NULL && g->token_size != NULL, "memory");
    for (int32_t i = 0; i < g->vocab_size; i++) {
        g->token_size[i] = (int32_t)le64(p);
        g->token_text[i] = p + 8;
        table_put(&g->vocab, g->token_text[i], g->token_size[i], i);
        p += 8 + g->token_size[i];
    }
}

static void merges_read(struct gguf * g) {
    const uint64_t count = le64(g->cursor + 4);
    const uint8_t * p = g->cursor + 12;
    g->merges = table_open(count);
    for (uint64_t i = 0; i < count; i++) {
        table_put(&g->merges, p + 8, (int32_t)le64(p), (int32_t)i);
        p += 8 + le64(p);
    }
}

static void types_read(struct gguf * g) {
    const uint64_t count = le64(g->cursor + 4);
    g->token_type = calloc(count + 1, sizeof(int32_t));
    must(g->token_type != NULL, "memory");
    for (uint64_t i = 0; i < count; i++) {
        g->token_type[i] = (int32_t)le32(g->cursor + 12 + i * 4);
    }
}

static void metadata_read(struct gguf * g, uint64_t count) {
    for (uint64_t i = 0; i < count; i++) {
        const uint64_t size = le64(g->cursor);
        const uint8_t * key = g->cursor + 8;
        const uint32_t type = le32(key + size);
        g->cursor = key + size + 4;
        if (key_is(key, size, "tokenizer.ggml.tokens")) {
            tokens_read(g);
        } else if (key_is(key, size, "tokenizer.ggml.merges")) {
            merges_read(g);
        } else if (key_is(key, size, "tokenizer.ggml.token_type")) {
            types_read(g);
        } else if (key_is(key, size, "general.alignment")) {
            g->alignment = le32(g->cursor);
        }
        value_skip(g, type);
    }
}

const uint8_t * gguf_info(struct gguf * g, const char * name, uint32_t * type,
                          uint64_t * count) {
    const uint8_t * p = g->infos;
    const uint8_t * found = NULL;
    for (uint64_t i = 0; i < g->tensor_count; i++) {
        const uint64_t size = le64(p);
        const uint8_t * key = p + 8;
        const uint32_t dims = le32(key + size);
        const uint8_t * tail = key + size + 4 + dims * 8;
        if (key_is(key, size, name)) {
            *count = 1;
            for (uint32_t d = 0; d < dims; d++) {
                *count *= le64(key + size + 4 + d * 8);
            }
            *type = le32(tail);
            found = g->data + le64(tail + 4);
        }
        p = tail + 12;
    }
    g->faults += found != NULL ? 0 : 1;
    return found;
}

static size_t block_bytes(uint32_t type) {
    return type == GGUF_Q4_0 ? 2 + GGUF_BLOCK / 2 : 2 + GGUF_BLOCK;
}

static void block_decode(float * out, const uint8_t * block, uint32_t type) {
    const float scale = half_float(block);
    const uint8_t * q = block + 2;
    for (int32_t i = 0; i < GGUF_BLOCK && type == GGUF_Q8_0; i++) {
        out[i] = scale * (float)(int8_t)q[i];
    }
    for (int32_t i = 0; i < GGUF_BLOCK / 2 && type == GGUF_Q4_0; i++) {
        out[i] = scale * (float)((int32_t)(q[i] & 15) - 8);
        out[i + GGUF_BLOCK / 2] = scale * (float)((int32_t)(q[i] >> 4) - 8);
    }
}

void gguf_decode(float * out, const uint8_t * p, uint64_t count,
                 uint32_t type) {
    for (uint64_t b = 0; b < count / GGUF_BLOCK; b++) {
        block_decode(out + b * GGUF_BLOCK, p + b * block_bytes(type), type);
    }
}

void gguf_row(float * out, const uint8_t * p, int32_t row, int32_t width,
              uint32_t type) {
    gguf_decode(out, p + (size_t)row * (width / GGUF_BLOCK) *
                block_bytes(type), (uint64_t)width, type);
}

static void own(struct gguf * g, float * values) {
    must(g->owned_count < GGUF_OWNED, "tensors");
    g->owned[g->owned_count++] = values;
}

const float * gguf_tensor(struct gguf * g, const char * format,
                          int32_t layer) {
    char name[NAME];
    uint32_t type = 0;
    uint64_t count = 0;
    const float * values = NULL;
    snprintf(name, sizeof name, format, layer);
    const uint8_t * p = gguf_info(g, name, &type, &count);
    if (p != NULL && type == GGUF_F32) {
        values = (const float *)(const void *)p;
    } else if (p != NULL && (type == GGUF_Q8_0 || type == GGUF_Q4_0)) {
        float * decoded = floats(count);
        gguf_decode(decoded, p, count, type);
        values = decoded;
        own(g, decoded);
    } else {
        g->faults++;
    }
    return values;
}

struct matrix gguf_matrix(struct gguf * g, const char * format,
                          int32_t layer) {
    char name[NAME];
    uint32_t type = 0;
    uint64_t count = 0;
    struct matrix m = { 0 };
    snprintf(name, sizeof name, format, layer);
    const uint8_t * p = gguf_info(g, name, &type, &count);
    const bool fine = p != NULL && (type == GGUF_Q8_0 || type == GGUF_Q4_0);
    float * scale = floats(fine ? count / GGUF_BLOCK : 1);
    own(g, scale);
    g->faults += fine ? 0 : 1;
    if (fine) {
        for (uint64_t b = 0; b < count / GGUF_BLOCK; b++) {
            scale[b] = half_float(p + b * block_bytes(type));
        }
        m.blocks = p;
        m.scale = scale;
        m.bits = type == GGUF_Q4_0 ? 4 : 8;
    }
    if (fine && !g->packed) {
        float * values = floats(count);
        own(g, values);
        gguf_decode(values, p, count, type);
        m.values = values;
    }
    return m;
}

void gguf_linear(struct gguf * g, float * y, const float * x,
                 const struct matrix * w, int32_t out, int32_t in,
                 int32_t rows) {
    const struct gemm_args a = {
        .a = x, .b = w->values, .c = y, .m = rows, .n = out, .k = in,
        .lda = in, .ldb = in, .ldc = out, .tb = true
    };
    const struct blocks b = {
        .codes = w->blocks, .scale = w->scale, .out = out, .in = in,
        .bits = w->bits
    };
    if (w->values != NULL && rows > 1) {
        memset(y, 0, (size_t)rows * out * sizeof(float));
        g->backend->gemm(g->backend, &a);
    } else {
        for (int32_t r = 0; r < rows; r++) {
            g->backend->blocks(g->backend, &b, x + (size_t)r * in,
                               y + (size_t)r * out);
        }
    }
}

static bool letter(uint8_t c) {
    return (c | 32) >= 'a' && (c | 32) <= 'z' ? true : c >= 0x80;
}

static bool digit(uint8_t c) {
    return c >= '0' && c <= '9';
}

static bool space(uint8_t c) {
    return c == ' ' || (c >= '\t' && c <= '\r');
}

static bool line(uint8_t c) {
    return c == '\n' || c == '\r';
}

static bool mark(uint8_t c) {
    return c != 0 && !letter(c) && !digit(c) && !space(c);
}

static int32_t run(const char * s, int32_t at, bool (* is)(uint8_t)) {
    int32_t end = at;
    while (s[end] != 0 && is((uint8_t)s[end])) {
        end++;
    }
    return end;
}

static int32_t quote_end(const char * s, int32_t at) {
    const char a = (char)(s[at + 1] | 32);
    const char b = s[at + 1] != 0 ? (char)(s[at + 2] | 32) : 0;
    int32_t end = 0;
    if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') ||
        (a == 'l' && b == 'l')) {
        end = at + 3;
    } else if (a == 's' || a == 't' || a == 'm' || a == 'd') {
        end = at + 2;
    }
    return end;
}

static int32_t blank_end(const char * s, int32_t at) {
    const int32_t all = run(s, at, space);
    int32_t end = all;
    int32_t lines = 0;
    for (int32_t i = at; i < all; i++) {
        lines = line((uint8_t)s[i]) ? i + 1 : lines;
    }
    if (lines != 0) {
        end = lines;
    } else if (s[all] != 0 && all - at > 1) {
        end = all - 1;
    }
    return end;
}

static int32_t word_end(const char * s, int32_t at, int32_t digits) {
    const uint8_t c = (uint8_t)s[at];
    const uint8_t next = (uint8_t)s[at + 1];
    int32_t end = 0;
    if (c == '\'' && quote_end(s, at) != 0) {
        end = quote_end(s, at);
    } else if (letter(c)) {
        end = run(s, at, letter);
    } else if (!digit(c) && !line(c) && letter(next)) {
        end = run(s, at + 1, letter);
    } else if (digit(c)) {
        end = run(s, at, digit);
        end = end - at > digits ? at + digits : end;
    } else if (mark(c) || (c == ' ' && mark(next))) {
        end = run(s, run(s, at + (c == ' ' ? 1 : 0), mark), line);
    } else {
        end = blank_end(s, at);
    }
    return end > at ? end : at + 1;
}

static int32_t byte_write(uint8_t * out, uint8_t c) {
    const bool plain = (c >= 33 && c <= 126) || (c >= 161 && c != 173);
    uint32_t code = c;
    int32_t size = 1;
    if (!plain) {
        code = 256 + (c <= 32 ? c : c <= 160 ? c - 127 + 33 : 67);
    }
    if (code < 128) {
        out[0] = (uint8_t)code;
    } else {
        out[0] = (uint8_t)(0xC0 | code >> 6);
        out[1] = (uint8_t)(0x80 | (code & 63));
        size = 2;
    }
    return size;
}

static int32_t pair_rank(const struct gguf * g, const uint8_t * text,
                         struct span a, struct span b) {
    uint8_t key[2 * WORD * 2 + 1];
    memcpy(key, text + a.at, (size_t)a.size);
    key[a.size] = ' ';
    memcpy(key + a.size + 1, text + b.at, (size_t)b.size);
    return table_find(&g->merges, key, a.size + 1 + b.size);
}

static int32_t pair_best(const struct gguf * g, const uint8_t * text,
                         const struct span * parts, int32_t count,
                         int32_t * rank) {
    int32_t best = -1;
    *rank = -1;
    for (int32_t i = 0; i + 1 < count; i++) {
        const int32_t r = pair_rank(g, text, parts[i], parts[i + 1]);
        if (r >= 0 && (best < 0 || r < *rank)) {
            best = i;
            *rank = r;
        }
    }
    return best;
}

static int32_t pair_merge(const struct gguf * g, const uint8_t * text,
                          struct span * parts, int32_t count, int32_t rank) {
    int32_t kept = 0;
    int32_t i = 0;
    while (i < count) {
        const bool join = i + 1 < count &&
                          pair_rank(g, text, parts[i], parts[i + 1]) == rank;
        parts[kept] = parts[i];
        parts[kept].size += join ? parts[i + 1].size : 0;
        kept++;
        i += join ? 2 : 1;
    }
    return kept;
}

static void emit(struct ids * to, int32_t id) {
    if (to->count < to->capacity) {
        to->out[to->count] = id;
    }
    to->count++;
}

static void word_encode(const struct gguf * g, const char * s, int32_t size,
                        struct ids * to) {
    uint8_t text[WORD * 2];
    struct span parts[WORD];
    const int32_t n = size < WORD ? size : WORD;
    int32_t count = 0;
    int32_t used = 0;
    int32_t rank = 0;
    for (int32_t i = 0; i < n; i++) {
        const int32_t bytes = byte_write(text + used, (uint8_t)s[i]);
        parts[count++] = (struct span){ .at = used, .size = bytes };
        used += bytes;
    }
    const int32_t whole = table_find(&g->vocab, text, used);
    int32_t pick = whole >= 0 ? -1 : pair_best(g, text, parts, count, &rank);
    while (pick >= 0) {
        count = pair_merge(g, text, parts, count, rank);
        pick = pair_best(g, text, parts, count, &rank);
    }
    if (whole >= 0) {
        emit(to, whole);
    }
    for (int32_t i = 0; i < count && whole < 0; i++) {
        const int32_t id = table_find(&g->vocab, text + parts[i].at,
                                      parts[i].size);
        emit(to, id >= 0 ? id : 0);
    }
}

static int32_t special_at(const struct gguf * g, const char * p,
                          int32_t * size) {
    const char * close = p[0] == '<' ? memchr(p, '>', strnlen(p, 40)) : NULL;
    const int32_t n = close != NULL ? (int32_t)(close - p) + 1 : 0;
    const int32_t id = n > 0 ? table_find(&g->vocab, (const uint8_t *)p, n) :
                       -1;
    const bool special = id >= 0 && (g->token_type[id] == 3 ||
                                     g->token_type[id] == 4);
    *size = n;
    return special ? id : -1;
}

static int32_t special_next(const struct gguf * g, const char * s,
                            int32_t at) {
    int32_t size = 0;
    int32_t next = at;
    while (s[next] != 0 && special_at(g, s + next, &size) < 0) {
        next++;
    }
    return next;
}

static int32_t encode(const struct gguf * g, const char * s, int32_t * out,
                      int32_t capacity, bool special) {
    struct ids to = { .out = out, .capacity = capacity };
    int32_t at = 0;
    while (s[at] != 0) {
        const int32_t stop = special ? special_next(g, s, at) :
                             (int32_t)strlen(s);
        int32_t size = 0;
        while (at < stop) {
            const int32_t end = word_end(s, at, g->digits);
            const int32_t cut = end < stop ? end : stop;
            word_encode(g, s + at, cut - at, &to);
            at = cut;
        }
        if (s[at] != 0) {
            emit(&to, special_at(g, s + at, &size));
            at += size;
        }
    }
    return to.count < capacity ? to.count : capacity;
}

int32_t gguf_encode(const struct gguf * g, const char * s, int32_t * out,
                    int32_t capacity) {
    return encode(g, s, out, capacity, false);
}

int32_t gguf_special(const struct gguf * g, const char * s, int32_t * out,
                     int32_t capacity) {
    return encode(g, s, out, capacity, true);
}

int32_t gguf_mark(const struct gguf * g, char * text, int32_t used,
                  int32_t capacity, int32_t token) {
    int32_t at = used;
    for (int32_t i = 0; i < g->token_size[token] && at + 1 < capacity; i++) {
        text[at++] = (char)g->token_text[token][i];
    }
    return at;
}

int32_t gguf_text(const struct gguf * g, char * text, int32_t used,
                  int32_t capacity, int32_t token) {
    const uint8_t * p = g->token_text[token];
    int32_t at = used;
    int32_t i = 0;
    while (g->token_type[token] == 1 && i < g->token_size[token] &&
           at + 1 < capacity) {
        const bool wide = p[i] >= 0xC0 && i + 1 < g->token_size[token];
        const uint32_t code = wide ? (uint32_t)(p[i] & 31) << 6 |
                              (p[i + 1] & 63) : p[i];
        uint8_t c = (uint8_t)code;
        if (code >= 256) {
            c = (uint8_t)(code - 256);
            c = c <= 32 ? c : c == 67 ? 173 : (uint8_t)(c - 33 + 127);
        }
        text[at++] = (char)c;
        i += wide ? 2 : 1;
    }
    return g->token_type[token] == 4 ?
           gguf_mark(g, text, at, capacity, token) : at;
}

bool gguf_open(struct gguf * g, const char * path, struct backend * backend,
               bool packed) {
    g->backend = backend;
    g->packed = packed;
    g->digits = 3;
    g->file = file_map(path, &g->size);
    const bool mapped = g->file != NULL && g->size > 24 &&
                        memcmp(g->file, "GGUF", 4) == 0;
    if (mapped) {
        g->tensor_count = le64(g->file + 8);
        g->cursor = g->file + 24;
        g->alignment = 32;
        metadata_read(g, le64(g->file + 16));
        g->infos = g->cursor;
        for (uint64_t i = 0; i < g->tensor_count; i++) {
            g->cursor += 8 + le64(g->cursor);
            g->cursor += 4 + le32(g->cursor) * 8 + 12;
        }
        g->data = g->file + ((uint64_t)(g->cursor - g->file) + g->alignment -
                             1) / g->alignment * g->alignment;
    }
    return mapped && g->token_text != NULL && g->token_type != NULL &&
           g->merges.key != NULL;
}

void gguf_close(struct gguf * g) {
    for (int32_t i = 0; i < g->owned_count; i++) {
        free(g->owned[i]);
    }
    table_close(&g->vocab);
    table_close(&g->merges);
    free(g->token_text);
    free(g->token_size);
    free(g->token_type);
    file_unmap(g->file, g->size);
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

static void attend(const struct attention * a, float * o, const float * q,
                   int32_t group, int32_t count, float * score) {
    const int32_t step = a->groups * a->head;
    const float * keys = a->keys + group * a->head;
    const float * values = a->values + group * a->head;
    float top = -INFINITY;
    float sum = 0.0f;
    for (int32_t s = 0; s < count; s++) {
        score[s] = dot(q, keys + s * step, a->head) / sqrtf((float)a->head);
        top = fmaxf(top, score[s]);
    }
    for (int32_t s = 0; s < count; s++) {
        score[s] = expf(score[s] - top);
        sum += score[s];
    }
    memset(o, 0, (size_t)a->head * sizeof(float));
    for (int32_t s = 0; s < count; s++) {
        const float * from = values + s * step;
        const float w = score[s] / sum;
        for (int32_t d = 0; d < a->head; d++) {
            o[d] += w * from[d];
        }
    }
}

enum { LANES = 16 };

static float peak(const float * x, int32_t n) {
    float lanes[LANES];
    float top = -INFINITY;
    for (int32_t j = 0; j < LANES; j++) {
        lanes[j] = -INFINITY;
    }
    for (int32_t i = 0; i + LANES <= n; i += LANES) {
        for (int32_t j = 0; j < LANES; j++) {
            lanes[j] = x[i + j] > lanes[j] ? x[i + j] : lanes[j];
        }
    }
    for (int32_t i = n / LANES * LANES; i < n; i++) {
        top = x[i] > top ? x[i] : top;
    }
    for (int32_t j = 0; j < LANES; j++) {
        top = lanes[j] > top ? lanes[j] : top;
    }
    return top;
}

static float total(const float * x, int32_t n) {
    #pragma clang fp reassociate(on)
    float sum = 0.0f;
    for (int32_t i = 0; i < n; i++) {
        sum += x[i];
    }
    return sum;
}

static void soften(const struct attention * a, float * score, int32_t first,
                   int32_t rows) {
    const int32_t count = first + rows;
    const float scale = 1.0f / sqrtf((float)a->head);
    for (int32_t r = 0; r < rows; r++) {
        float * row = score + (size_t)r * count;
        const float top = peak(row, first + r + 1);
        for (int32_t s = 0; s <= first + r; s++) {
            row[s] = (row[s] - top) * scale;
        }
        for (int32_t s = first + r + 1; s < count; s++) {
            row[s] = -INFINITY;
        }
    }
    a->backend->exp(a->backend, score, score, rows * count);
    for (int32_t r = 0; r < rows; r++) {
        float * row = score + (size_t)r * count;
        const float gain = 1.0f / total(row, first + r + 1);
        for (int32_t s = 0; s <= first + r; s++) {
            row[s] *= gain;
        }
    }
}

static void attend_rows(const struct attention * a, float * mix,
                        const float * q, int32_t first, int32_t rows,
                        float * score) {
    const int32_t count = first + rows;
    const int32_t wide = a->heads * a->head;
    for (int32_t h = 0; h < a->heads; h++) {
        const int32_t at = h * a->groups / a->heads * a->head;
        const struct gemm_args scores = {
            .a = q + h * a->head, .b = a->keys + at, .c = score, .m = rows,
            .n = count, .k = a->head, .lda = wide,
            .ldb = a->groups * a->head, .ldc = count, .tb = true
        };
        const struct gemm_args mixed = {
            .a = score, .b = a->values + at, .c = mix + h * a->head,
            .m = rows, .n = a->head, .k = count, .lda = count,
            .ldb = a->groups * a->head, .ldc = wide
        };
        memset(score, 0, (size_t)rows * count * sizeof(float));
        a->backend->gemm(a->backend, &scores);
        soften(a, score, first, rows);
        for (int32_t r = 0; r < rows; r++) {
            memset(mix + (size_t)r * wide + h * a->head, 0,
                   (size_t)a->head * sizeof(float));
        }
        a->backend->gemm(a->backend, &mixed);
    }
}

void chat_attend(const struct attention * a, float * mix, const float * q,
                 int32_t first, int32_t rows) {
    float * score = floats((size_t)(first + rows) * (rows > 1 ? rows : 1));
    if (rows > 1) {
        attend_rows(a, mix, q, first, rows, score);
    }
    for (int32_t i = 0; i < a->heads && rows == 1; i++) {
        attend(a, mix + i * a->head, q + i * a->head,
               i * a->groups / a->heads, first + 1, score);
    }
    free(score);
}

static const uint32_t stored_magic = 0x54414c4b;

static size_t row_bytes(const struct chat * c) {
    return (size_t)c->store.tracks * c->store.width * sizeof(float);
}

static size_t chat_place(struct chat * c, uint8_t * base) {
    const size_t fed = CHAT_HEAD + (size_t)c->capacity * sizeof(int32_t);
    const size_t starts = fed + (size_t)c->context * sizeof(int32_t);
    const size_t slot = starts + CHAT_STARTS * sizeof(int32_t);
    if (base != NULL) {
        c->stored = (struct stored *)(void *)base;
        c->pending = (int32_t *)(void *)(base + CHAT_HEAD);
        c->fed = (int32_t *)(void *)(base + fed);
        c->starts = (int32_t *)(void *)(base + starts);
        c->slot = base + slot;
    }
    return (slot + c->store.state_size + CHAT_HEAD - 1) / CHAT_HEAD *
           CHAT_HEAD;
}

static size_t file_bytes(const struct chat * c) {
    return c->head + (size_t)c->context * row_bytes(c);
}

static bool stored_fits(const struct chat * c, const struct stored * h) {
    return h->magic == stored_magic && h->clean == 1 &&
           h->model == c->store.model && h->context == c->context &&
           h->state_size == c->store.state_size &&
           h->tracks == c->store.tracks && h->width == c->store.width;
}

static void stored_take(struct chat * c) {
    const struct stored * h = c->stored;
    c->pending_count = h->pending_count;
    c->fed_count = h->fed_count;
    c->start_count = h->start_count;
    c->dropped = h->dropped;
    c->mode = h->mode;
    for (int32_t i = 0; i < CHAT_MARKS; i++) {
        c->marks[i].position = -1;
        c->marks[i].wanted = h->wanted[i];
    }
}

static void rows_move(struct chat * c, uint8_t * file, int32_t first,
                      int32_t last, bool out) {
    const size_t wide = (size_t)c->store.width * sizeof(float);
    for (int32_t i = first * c->store.tracks; i < last * c->store.tracks;
         i++) {
        const int32_t t = i % c->store.tracks;
        uint8_t * at = file + c->head + (size_t)i * wide;
        float * live = c->track[t] + (size_t)(i / c->store.tracks) *
                       c->store.width;
        memcpy(out ? (void *)at : (void *)live,
               out ? (const void *)live : (const void *)at, wide);
    }
}

bool chat_open(struct chat * c, int32_t context, const struct store * s) {
    int32_t made = 0;
    c->context = context;
    c->capacity = 2 * context;
    c->store = *s;
    c->head = chat_place(c, NULL);
    uint8_t * base = s->path != NULL ? file_shared(s->path, file_bytes(c)) :
                     calloc(1, c->head);
    c->scratch = calloc((size_t)c->capacity, sizeof(int32_t));
    for (int32_t i = 0; i < CHAT_MARKS; i++) {
        c->marks[i] = (struct mark){
            .position = -1, .wanted = -1, .state = malloc(s->state_size)
        };
        made += c->marks[i].state != NULL ? 1 : 0;
    }
    if (base != NULL) {
        chat_place(c, base);
    }
    if (base != NULL && stored_fits(c, c->stored)) {
        stored_take(c);
    } else if (base != NULL) {
        *c->stored = (struct stored){
            .magic = stored_magic, .clean = 1, .model = s->model,
            .state_size = s->state_size, .tracks = s->tracks,
            .width = s->width, .context = context, .wanted = { -1, -1, -1 }
        };
    }
    return made == CHAT_MARKS && c->scratch != NULL && base != NULL &&
           s->tracks <= CHAT_TRACKS;
}

void chat_resume(struct chat * c) {
    if (c->fed_count > 0) {
        rows_move(c, (uint8_t *)c->stored, 0, c->fed_count, false);
        c->load(c->engine, c->slot);
    }
    c->synced = c->fed_count;
}

void chat_commit(struct chat * c) {
    struct stored * h = c->stored;
    h->clean = 0;
    c->save(c->engine, c->slot);
    if (c->store.path != NULL && c->synced < c->fed_count) {
        rows_move(c, (uint8_t *)h, c->synced, c->fed_count, true);
    }
    c->synced = c->fed_count;
    h->pending_count = c->pending_count;
    h->fed_count = c->fed_count;
    h->start_count = c->start_count;
    h->dropped = c->dropped;
    h->mode = c->mode;
    for (int32_t i = 0; i < CHAT_MARKS; i++) {
        h->wanted[i] = c->marks[i].wanted;
    }
    h->clean = 1;
}

bool chat_save(struct chat * c, const char * path) {
    uint8_t * to = file_shared(path, file_bytes(c));
    chat_commit(c);
    if (to != NULL) {
        memcpy(to, c->stored, c->head);
        rows_move(c, to, 0, c->fed_count, true);
        file_unmap(to, file_bytes(c));
    }
    return to != NULL;
}

bool chat_load(struct chat * c, const char * path) {
    size_t size = 0;
    const uint8_t * from = file_map(path, &size);
    const struct stored * h = (const struct stored *)(const void *)from;
    const bool fine = from != NULL && size >= file_bytes(c) &&
                      stored_fits(c, h);
    if (fine) {
        memcpy(c->stored, from, c->head);
        stored_take(c);
        rows_move(c, (uint8_t *)(uintptr_t)from, 0, c->fed_count, false);
        c->load(c->engine, c->slot);
        c->synced = 0;
    }
    file_unmap(from, size);
    return fine;
}

void chat_close(struct chat * c) {
    for (int32_t i = 0; i < CHAT_MARKS; i++) {
        free(c->marks[i].state);
    }
    free(c->scratch);
    if (c->store.path != NULL) {
        file_unmap((const uint8_t *)c->stored, file_bytes(c));
    } else {
        free(c->stored);
    }
}

void chat_clear(struct chat * c) {
    c->overflow = false;
    c->pending_count = 0;
    c->start_count = 0;
}

static void stored_guard(struct chat * c) {
    const struct stored * h = c->stored;
    if (c->pending_count < h->pending_count ||
        c->fed_count < h->fed_count || c->start_count < h->start_count) {
        c->stored->clean = 0;
    }
}

void chat_token(struct chat * c, int32_t token) {
    if (c->counting) {
        c->counted++;
    } else if (c->pending_count < c->capacity) {
        stored_guard(c);
        c->pending[c->pending_count++] = token;
    } else {
        c->overflow = true;
    }
}

void chat_text(struct chat * c, const struct gguf * g, const char * s) {
    const int32_t count = gguf_encode(g, s, c->scratch, c->capacity);
    for (int32_t i = 0; i < count; i++) {
        chat_token(c, c->scratch[i]);
    }
}

void chat_special(struct chat * c, const struct gguf * g, const char * s) {
    const int32_t count = gguf_special(g, s, c->scratch, c->capacity);
    for (int32_t i = 0; i < count; i++) {
        chat_token(c, c->scratch[i]);
    }
}

void chat_begin(struct chat * c) {
    if (!c->counting && c->start_count < CHAT_STARTS) {
        stored_guard(c);
        c->starts[c->start_count++] = c->pending_count;
    }
}

void chat_want(struct chat * c, int32_t slot) {
    if (!c->counting) {
        c->marks[slot].wanted = c->pending_count;
    }
}

static void marks_save(struct chat * c) {
    for (int32_t i = 0; i < CHAT_MARKS; i++) {
        struct mark * m = &c->marks[i];
        if (m->wanted == c->fed_count && m->position != c->fed_count) {
            c->save(c->engine, m->state);
            m->position = c->fed_count;
            m->dropped = c->dropped;
        }
    }
}

static void marks_restore(struct chat * c, int32_t same) {
    const struct mark * best = NULL;
    for (int32_t i = 0; i < CHAT_MARKS; i++) {
        const struct mark * m = &c->marks[i];
        const bool usable = m->position >= 0 && m->position <= same;
        best = usable && (best == NULL || m->position > best->position) ?
               m : best;
    }
    c->load(c->engine, best != NULL ? best->state : NULL);
    c->fed_count = best != NULL ? best->position : 0;
    c->synced = c->synced < c->fed_count ? c->synced : c->fed_count;
    c->dropped = best != NULL ? best->dropped : 0;
    for (int32_t i = 0; i < CHAT_MARKS; i++) {
        struct mark * m = &c->marks[i];
        m->position = m->position > c->fed_count ? -1 : m->position;
    }
}

static int32_t feed_run(const struct chat * c) {
    int32_t end = c->pending_count - c->fed_count < CHAT_ROWS ?
                  c->pending_count : c->fed_count + CHAT_ROWS;
    for (int32_t i = 0; i < CHAT_MARKS; i++) {
        const int32_t wanted = c->marks[i].wanted;
        end = wanted > c->fed_count && wanted < end ? wanted : end;
    }
    return end - c->fed_count;
}

void chat_feed(struct chat * c) {
    int32_t same = 0;
    while (same < c->fed_count && same < c->pending_count &&
           c->fed[same] == c->pending[same]) {
        same++;
    }
    if (same < c->fed_count) {
        marks_restore(c, same);
    }
    c->kept = c->fed_count;
    while (c->fed_count < c->pending_count &&
           (c->alive == NULL || c->alive(c->who))) {
        const int32_t count = feed_run(c);
        marks_save(c);
        stored_guard(c);
        c->forward(c->engine, c->pending + c->fed_count, count,
                   c->fed_count + c->dropped, c->fed_count);
        memcpy(c->fed + c->fed_count, c->pending + c->fed_count,
               (size_t)count * sizeof(int32_t));
        c->fed_count += count;
    }
    marks_save(c);
}

void chat_took(struct chat * c, int32_t token) {
    stored_guard(c);
    c->forward(c->engine, &token, 1, c->fed_count + c->dropped,
               c->fed_count);
    c->fed[c->fed_count++] = token;
    if (c->pending_count < c->capacity) {
        c->pending[c->pending_count++] = token;
    }
}

void chat_rewind(struct chat * c, int32_t messages) {
    const int32_t kept = c->start_count > messages ?
                         c->start_count - messages : 0;
    if (c->start_count > 0) {
        c->pending_count = c->starts[kept];
        c->start_count = kept;
    }
}

void chat_start(struct chat * c, const struct gguf * g,
                const struct format * f) {
    chat_clear(c);
    chat_special(c, g, f->start);
    chat_want(c, MARK_SYSTEM);
}

void chat_content(struct chat * c, const struct gguf * g,
                  const struct format * f, const char * content) {
    const char * from = content + (f->trim ? strspn(content, " \t\r\n") : 0);
    size_t size = strlen(from);
    while (f->trim && size > 0 &&
           strchr(" \t\r\n", from[size - 1]) != NULL) {
        size--;
    }
    char * text = strndup(from, size);
    must(text != NULL, "memory");
    chat_text(c, g, text);
    free(text);
}

void chat_message(struct chat * c, const struct gguf * g,
                  const struct format * f, const char * role,
                  const char * content, const char * tools) {
    const bool system = strcmp(role, "system") == 0;
    const bool empty = content[strspn(content, " \t\r\n")] == 0;
    chat_begin(c);
    if (strcmp(role, "user") == 0) {
        chat_want(c, MARK_USER);
    }
    chat_special(c, g, f->open);
    chat_text(c, g, role);
    chat_special(c, g, f->middle);
    if (system && tools != NULL && f->tools_open != NULL) {
        chat_special(c, g, f->tools_open);
        chat_text(c, g, "\n");
        chat_content(c, g, f, tools);
        chat_special(c, g, f->tools_close);
        chat_text(c, g, empty ? "" : "\n\n");
    }
    chat_content(c, g, f, content);
    chat_special(c, g, f->close);
    if (system) {
        chat_want(c, MARK_SYSTEM);
    }
}

int32_t chat_reply(struct chat * c, const struct gguf * g,
                   const struct format * f, bool think) {
    chat_begin(c);
    chat_special(c, g, f->reply);
    chat_want(c, MARK_REPLY);
    const int32_t at = c->pending_count;
    chat_special(c, g, think ? f->think : f->plain);
    return at;
}

static int32_t cut_message(const struct chat * c, int32_t keep, int32_t need,
                           int32_t * first) {
    const int32_t goal = c->context - c->context / 4;
    int32_t from = 0;
    while (from < c->start_count && c->starts[from] < keep) {
        from++;
    }
    int32_t to = from;
    while (to + 2 < c->start_count &&
           c->pending_count - (c->starts[to] - keep) + need > goal) {
        to++;
    }
    *first = from;
    return to;
}

static bool fed_through(const struct chat * c, int32_t end) {
    int32_t same = 0;
    while (same < end && same < c->fed_count &&
           c->fed[same] == c->pending[same]) {
        same++;
    }
    return same == end;
}

static void fed_cut(struct chat * c, int32_t keep, int32_t count) {
    c->stored->clean = 0;
    c->shift(c->engine, keep, count, c->fed_count);
    c->synced = c->synced < keep ? c->synced : keep;
    memmove(c->fed + keep, c->fed + keep + count,
            (size_t)(c->fed_count - keep - count) * sizeof(int32_t));
    c->fed_count -= count;
    c->dropped += count;
    for (int32_t i = 0; i < CHAT_MARKS; i++) {
        struct mark * m = &c->marks[i];
        const bool inside = m->position > keep && m->position < keep + count;
        const bool later = m->position >= keep + count;
        m->dropped += later ? count : 0;
        m->position = inside ? -1 : later ? m->position - count :
                      m->position;
    }
}

static void pending_cut(struct chat * c, int32_t keep, int32_t count,
                        int32_t first, int32_t last) {
    c->stored->clean = 0;
    memmove(c->pending + keep, c->pending + keep + count,
            (size_t)(c->pending_count - keep - count) * sizeof(int32_t));
    c->pending_count -= count;
    for (int32_t i = last; i < c->start_count; i++) {
        c->starts[first + i - last] = c->starts[i] - count;
    }
    c->start_count -= last - first;
    for (int32_t i = 0; i < CHAT_MARKS; i++) {
        struct mark * m = &c->marks[i];
        m->wanted = m->wanted >= keep + count ? m->wanted - count : m->wanted;
    }
}

int32_t chat_head(struct chat * c, const struct gguf * g,
                  const struct format * f, bool think) {
    c->counting = true;
    c->counted = 0;
    chat_reply(c, g, f, think);
    c->counting = false;
    return c->counted;
}

bool chat_fit(struct chat * c, struct turn * turn, int32_t head) {
    const struct mark * system = &c->marks[MARK_SYSTEM];
    const int32_t keep = system->wanted > 0 ? system->wanted : 0;
    const int32_t need = turn->limit + head;
    const bool tight = c->pending_count + need > c->context;
    int32_t first = 0;
    const int32_t last = tight ? cut_message(c, keep, need, &first) : 0;
    const int32_t count = tight && last > first ? c->starts[last] - keep : 0;
    const bool soft = turn->shift == SHIFT_SOFT && count > 0 &&
                      system->position == keep &&
                      fed_through(c, keep + count);
    turn->dropped = 0;
    turn->full = c->overflow ||
                 (tight && (turn->shift == SHIFT_NONE || count == 0));
    if (tight && !turn->full && count > 0) {
        if (soft) {
            fed_cut(c, keep, count);
        }
        pending_cut(c, keep, count, first, last);
        turn->dropped = last - first;
    }
    return !turn->full;
}
