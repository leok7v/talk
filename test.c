#include "transformer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { RUNS = 16, TOKENS = 317, OUT = 32768, IN = 64, ROWS = 3 };

static int32_t checks;
static int32_t faults;

static void expect(bool ok, const char * what) {
    checks++;
    if (!ok) {
        faults++;
        printf("test: FAIL %s\n", what);
    }
}

static bool near(const float * a, const float * b, int32_t n) {
    float worst = 0.0f;
    for (int32_t i = 0; i < n; i++) {
        worst = fmaxf(worst, fabsf(a[i] - b[i]) / (1.0f + fabsf(b[i])));
    }
    return worst < 1e-4f;
}

static void fill(float * p, int32_t n, uint32_t seed) {
    uint32_t state = seed;
    for (int32_t i = 0; i < n; i++) {
        state = state * 1664525u + 1013904223u;
        p[i] = (float)(state >> 8) / 8388608.0f - 1.0f;
    }
}

static void options_test(void) {
    const char * const words[] = { "llm=qwen", "tokens=7", "llm=falcon",
                                   NULL };
    const char * const bad[] = { "llm=qwen", "nope=1", "word", NULL };
    expect(strcmp(option(words, "llm", "x"), "falcon") == 0, "option last");
    expect(strcmp(option(words, "gemm", "x"), "x") == 0, "option fallback");
    expect(option(NULL, "gemm", NULL) == NULL, "option of no list");
    expect(option_listed("llm gemm", "gemm", 4), "option_listed");
    expect(!option_listed("llm gemm", "gem", 3), "option_listed prefix");
    expect(option_unknown(words, "llm tokens") == NULL, "all known");
    expect(option_unknown(bad, "llm") == bad[1], "option_unknown");
}

static void guarded_fine(void * context) {
    *(int32_t *)context += 1;
    must(true, "nothing");
}

static void guarded_fault(void * context) {
    *(int32_t *)context += 1;
    must(false, "the inner call");
    *(int32_t *)context += 100;
}

static void guarded_both(void * context) {
    expect(guarded(guarded_fault, context) != NULL, "guarded inside");
    must(false, "the outer call");
}

static void guarded_test(void) {
    int32_t calls = 0;
    const char * what = guarded(guarded_fine, &calls);
    expect(what == NULL && calls == 1, "guarded with no fault");
    what = guarded(guarded_fault, &calls);
    expect(what != NULL && strcmp(what, "the inner call") == 0 && calls == 2,
           "guarded returns the fault");
    what = guarded(guarded_both, &calls);
    expect(what != NULL && strcmp(what, "the outer call") == 0 && calls == 3,
           "guarded in guarded");
    expect(guarded(guarded_fine, &calls) == NULL, "guarded after a fault");
}

static void turn_test(void) {
    const char * const words[] = { "tokens=7", "temperature=0", "shift=soft",
                                   "script=S", NULL };
    struct turn turn = { 0 };
    turn_read(&turn, words);
    expect(turn.limit == 7 && turn.temperature == 0.0f, "turn_read");
    expect(turn.shift == SHIFT_SOFT && turn.top_k == 20, "turn defaults");
    expect(strcmp(turn.script, "S") == 0, "turn script");
    expect(turn_shift("exact") == SHIFT_EXACT && turn_shift("x") == SHIFT_NONE,
           "turn_shift");
}

static void bytes_test(void) {
    const uint8_t b[] = { 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08 };
    const uint8_t one[] = { 0x00, 0x3C };
    const uint8_t minus[] = { 0x00, 0xC0 };
    const uint8_t small[] = { 0x01, 0x00 };
    const uint8_t huge[] = { 0x00, 0x7C };
    expect(le16(b) == 0x0201 && le32(b) == 0x04030201, "le16 and le32");
    expect(le64(b) == 0x0807060504030201ull, "le64");
    expect(half_float(one) == 1.0f && half_float(minus) == -2.0f, "half");
    expect(half_float(small) == ldexpf(1.0f, -24), "half subnormal");
    expect(isinf(half_float(huge)), "half infinity");
}

static void json_test(void) {
    const char text[] = "a\\n\\\"b\\u00e9\\ud83d\\ude00\" tail";
    const char * close = json_close(text, text + strlen(text));
    const char open[] = "abc";
    char out[32];
    char tight[4];
    expect(close != NULL && close[1] == ' ', "json_close");
    expect(json_close(open, open + 3) == NULL, "json_close open");
    expect(close != NULL && json_unescape(text, close, out, sizeof out) &&
           strcmp(out, "a\n\"b\xC3\xA9\xF0\x9F\x98\x80") == 0,
           "json_unescape");
    expect(close != NULL && !json_unescape(text, close, tight, sizeof tight),
           "json_unescape full");
}

static void wav_head(uint8_t * w, int32_t channels, uint32_t rate,
                     uint32_t bytes) {
    memcpy(w, "RIFF____WAVEfmt ", 16);
    memset(w + 16, 0, 28);
    w[16] = 16;
    w[20] = 1;
    w[22] = (uint8_t)channels;
    w[24] = (uint8_t)rate;
    w[25] = (uint8_t)(rate >> 8);
    w[34] = 16;
    memcpy(w + 36, "data", 4);
    w[40] = (uint8_t)bytes;
}

static void wav_test(void) {
    uint8_t w[44 + 16] = { 0 };
    float * samples = NULL;
    wav_head(w, 1, 16000, 8);
    w[44 + 3] = 0x40;
    w[44 + 5] = 0xC0;
    expect(wav_samples(w, 52, &samples) == 4, "wav count");
    expect(samples != NULL && samples[0] == 0.0f && samples[1] == 0.5f &&
           samples[2] == -0.5f, "wav values");
    free(samples);
    wav_head(w, 2, 32000, 16);
    expect(wav_samples(w, 60, &samples) == 2, "wav stereo at 32000");
    free(samples);
    expect(wav_samples((const uint8_t *)"not a wav file", 14, &samples) == 0 &&
           samples == NULL, "wav bad");
}

static void path_test(void) {
    char path[256];
    char tight[8];
    expect(model_path(path, sizeof path, "no/such.file") &&
           strstr(path, "/huggingface.co/no/such.file") != NULL,
           "model_path home");
    expect(!model_path(tight, sizeof tight, "no/such.file"),
           "model_path full");
}

static void gemm_test(struct backend * backend) {
    enum { M = 5, N = 37, K = 96 };
    float * a = floats(M * K);
    float * b = floats(N * K);
    float * c = floats(M * N);
    float * want = floats(M * N);
    const struct gemm_args g = {
        .a = a, .b = b, .c = c, .m = M, .n = N, .k = K, .lda = K, .ldb = K,
        .ldc = N, .tb = true
    };
    fill(a, M * K, 1);
    fill(b, N * K, 2);
    for (int32_t i = 0; i < M * N; i++) {
        for (int32_t p = 0; p < K; p++) {
            want[i] += a[i / N * K + p] * b[i % N * K + p];
        }
    }
    backend->gemm(backend, &g);
    expect(near(c, want, M * N), backend->name);
    backend->exp(backend, c, want, M * N);
    for (int32_t i = 0; i < M * N; i++) {
        want[i] = expf(want[i]);
    }
    expect(near(c, want, M * N), backend->name);
    free(a);
    free(b);
    free(c);
    free(want);
}

static void blocks_test(struct backend * backend, int32_t out,
                        int32_t bits) {
    const size_t page = (size_t)getpagesize();
    const size_t count = (size_t)out * (IN / BLOCK);
    const size_t step = 2 + (size_t)BLOCK * bits / 8;
    const uint32_t type = bits == 4 ? GGUF_Q4_0 : GGUF_Q8_0;
    const size_t bytes = (count * step + page - 1) / page * page;
    uint8_t * codes = NULL;
    float * scale = floats(count);
    float * row = floats(IN);
    float * x = floats(IN);
    float * y = floats((size_t)out);
    float * want = floats((size_t)out);
    must(posix_memalign((void **)&codes, page, bytes) == 0, "memory");
    for (size_t i = 0; i < bytes; i++) {
        codes[i] = (uint8_t)(i * 7 + i / 34);
    }
    for (size_t i = 0; i < count; i++) {
        codes[i * step + 1] = 0x38;
        scale[i] = half_float(codes + i * step);
    }
    const struct blocks w = { .codes = codes, .scale = scale, .out = out,
                              .in = IN, .bits = bits };
    fill(x, IN, 3);
    for (int32_t o = 0; o < out; o++) {
        gguf_row(row, codes, o, IN, type);
        for (int32_t i = 0; i < IN; i++) {
            want[o] += row[i] * x[i];
        }
    }
    backend->blocks(backend, &w, x, y);
    expect(near(y, want, out), backend->name);
    free(codes);
    free(scale);
    free(row);
    free(x);
    free(y);
    free(want);
}

static void backends_test(void) {
    const char * const names[] = { "cpu", "blas", "metal", "hip" };
    for (int32_t i = 0; i < 4; i++) {
        struct backend * backend = backend_open(names[i], NULL);
        if (backend != NULL) {
            gemm_test(backend);
            blocks_test(backend, 100, 8);
            blocks_test(backend, OUT, 8);
            blocks_test(backend, 100, 4);
            blocks_test(backend, OUT, 4);
            backend->close(backend);
        } else {
            printf("test: no backend %s here\n", names[i]);
        }
    }
}

struct fake {
    int32_t runs;
    int32_t count[RUNS];
    int32_t row[RUNS];
    int32_t next;
    int32_t saves;
    int32_t loads;
    bool ordered;
};

static void fake_forward(void * engine, const int32_t * tokens, int32_t count,
                         int32_t position, int32_t row) {
    struct fake * f = engine;
    f->ordered = f->ordered && row == f->next && position == row &&
                 f->runs < RUNS && tokens[0] == row;
    if (f->runs < RUNS) {
        f->count[f->runs] = count;
        f->row[f->runs] = row;
    }
    f->runs++;
    f->next = row + count;
}

static void fake_shift(void * engine, int32_t keep, int32_t count,
                       int32_t rows) {
    (void)engine;
    (void)keep;
    (void)count;
    (void)rows;
}

static void fake_save(void * engine, uint8_t * to) {
    struct fake * f = engine;
    memcpy(to, &f->next, sizeof f->next);
    f->saves++;
}

static void fake_load(void * engine, const uint8_t * from) {
    struct fake * f = engine;
    f->next = 0;
    if (from != NULL) {
        memcpy(&f->next, from, sizeof f->next);
    }
    f->loads++;
}

static void pending_make(struct chat * c) {
    const int32_t marks[] = { 10, 310, 315 };
    chat_clear(c);
    for (int32_t i = 0; i < TOKENS; i++) {
        for (int32_t m = 0; m < CHAT_MARKS; m++) {
            if (marks[m] == i) {
                chat_want(c, m);
            }
        }
        chat_token(c, i);
    }
}

static float fake_rows[2][3 * 1024];

static struct chat * fake_chat(struct chat * c, struct fake * f) {
    *c = (struct chat){
        .engine = f, .forward = fake_forward, .shift = fake_shift,
        .save = fake_save, .load = fake_load,
        .track = { fake_rows[0], fake_rows[1] }
    };
    memset(fake_rows, 0, sizeof fake_rows);
    return c;
}

static void stored_test(const struct store * store) {
    static struct chat c;
    struct fake f = { .ordered = true };
    struct store other = *store;
    must(chat_open(fake_chat(&c, &f), 1024, store), "chat");
    chat_resume(&c);
    expect(c.fed_count == TOKENS && c.pending_count == TOKENS &&
           f.loads == 1 && f.next == TOKENS &&
           c.track[1][3 * TOKENS - 1] == 3.5f,
           "a mapped chat comes back");
    chat_token(&c, 2000);
    chat_feed(&c);
    expect(f.runs == 1 && f.row[0] == TOKENS && f.count[0] == 1,
           "a chat that came back feeds only the new token");
    chat_close(&c);
    must(chat_open(fake_chat(&c, &f), 1024, store), "chat");
    expect(c.fed_count == TOKENS && c.pending_count == TOKENS,
           "a turn with no commit leaves the file at the last commit");
    chat_resume(&c);
    c.pending_count = 5;
    chat_token(&c, 3000);
    chat_close(&c);
    must(chat_open(fake_chat(&c, &f), 1024, store), "chat");
    expect(c.fed_count == 0 && c.pending_count == 0,
           "a turn that writes over the commit starts empty");
    chat_close(&c);
    other.path = NULL;
    must(chat_open(fake_chat(&c, &f), 1024, &other), "chat");
    expect(chat_load(&c, "test.copy") && c.fed_count == TOKENS &&
           c.track[1][3 * TOKENS - 1] == 3.5f && f.next == TOKENS,
           "chat_load takes a saved chat");
    other.model = 8;
    chat_close(&c);
    must(chat_open(fake_chat(&c, &f), 1024, &other), "chat");
    expect(!chat_load(&c, "test.copy") && c.fed_count == 0,
           "chat_load refuses the chat of another model");
    chat_close(&c);
    remove("test.state");
    remove("test.copy");
}

static bool fake_alive(void * who) {
    int32_t * left = who;
    *left -= 1;
    return *left >= 0;
}

static void stop_test(void) {
    static struct chat c;
    struct fake f = { .ordered = true };
    const struct store store = { .state_size = sizeof f.next };
    int32_t left = 1;
    must(chat_open(fake_chat(&c, &f), 1024, &store), "chat");
    pending_make(&c);
    c.alive = fake_alive;
    c.who = &left;
    chat_feed(&c);
    expect(f.runs == 1 && c.fed_count == 10, "a feed stops between runs");
    left = 100;
    chat_feed(&c);
    expect(f.ordered && f.runs == 5 && c.fed_count == TOKENS,
           "a feed that stopped goes on from its place");
    chat_close(&c);
}

static void feed_test(void) {
    static struct chat c;
    struct fake f = { .ordered = true };
    const struct store store = {
        .state_size = sizeof f.next, .tracks = 2, .width = 3, .model = 7,
        .path = "test.state"
    };
    const int32_t want[] = { 10, 256, 44, 5, 2 };
    bool same = true;
    fake_chat(&c, &f);
    remove("test.state");
    must(chat_open(&c, 1024, &store), "chat");
    pending_make(&c);
    chat_feed(&c);
    for (int32_t i = 0; i < 5; i++) {
        same = same && f.count[i] == want[i];
    }
    expect(f.runs == 5 && same, "feed runs end at the marks");
    expect(f.ordered && c.fed_count == TOKENS, "feed order");
    expect(f.saves == 3, "feed saves 3 checkpoints");
    chat_feed(&c);
    expect(f.runs == 5 && c.kept == TOKENS, "feed of nothing new");
    pending_make(&c);
    c.pending[312] = 1000;
    chat_feed(&c);
    expect(f.loads == 1 && f.runs == 7 && f.row[5] == 310 &&
           f.count[5] == 5 && f.count[6] == 2, "feed again from a mark");
    expect(f.ordered && c.fed_count == TOKENS && c.kept == 310,
           "feed order after a load");
    c.track[1][3 * TOKENS - 1] = 3.5f;
    c.synced = 0;
    chat_commit(&c);
    expect(chat_save(&c, "test.copy"), "chat_save");
    chat_close(&c);
    stored_test(&store);
}

static void attend_test(void) {
    enum { HEADS = 4, GROUPS = 2, HEAD = 8, FIRST = 5 };
    const int32_t wide = HEADS * HEAD;
    const int32_t step = GROUPS * HEAD;
    float * keys = floats((FIRST + ROWS) * step);
    float * values = floats((FIRST + ROWS) * step);
    float * q = floats(ROWS * wide);
    float * mix = floats(ROWS * wide);
    float * want = floats(ROWS * wide);
    float score[FIRST + ROWS];
    const struct attention a = { .keys = keys, .values = values,
                                 .heads = HEADS, .groups = GROUPS,
                                 .head = HEAD };
    fill(keys, (FIRST + ROWS) * step, 4);
    fill(values, (FIRST + ROWS) * step, 5);
    fill(q, ROWS * wide, 6);
    for (int32_t i = 0; i < ROWS * HEADS; i++) {
        const int32_t g = i % HEADS * GROUPS / HEADS * HEAD;
        const int32_t count = FIRST + i / HEADS + 1;
        float sum = 0.0f;
        for (int32_t s = 0; s < count; s++) {
            score[s] = 0.0f;
            for (int32_t d = 0; d < HEAD; d++) {
                score[s] += q[i * HEAD + d] * keys[s * step + g + d];
            }
            score[s] = expf(score[s] / sqrtf((float)HEAD));
            sum += score[s];
        }
        for (int32_t s = 0; s < count; s++) {
            for (int32_t d = 0; d < HEAD; d++) {
                want[i * HEAD + d] += score[s] / sum *
                                      values[s * step + g + d];
            }
        }
    }
    chat_attend(&a, mix, q, FIRST, ROWS);
    expect(near(mix, want, ROWS * wide), "chat_attend");
    free(keys);
    free(values);
    free(q);
    free(mix);
    free(want);
}

int main(void) {
    options_test();
    guarded_test();
    turn_test();
    bytes_test();
    json_test();
    wav_test();
    path_test();
    backends_test();
    feed_test();
    stop_test();
    attend_test();
    printf("test: %s, %d checks, %d faults\n", faults == 0 ? "PASS" : "FAIL",
           checks, faults);
    return faults == 0 ? 0 : 1;
}
