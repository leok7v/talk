#include "talk.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { TEXT = 16384, PROMPT = 512, REPLY = 128, SHAPES = 32 };

static const char * const sentence =
    "The river ran past the old mill, and the children of the village "
    "came there each summer to swim and to tell long stories. ";

struct shape {
    const char * kind;
    int32_t out;
    int32_t in;
    int32_t calls;
    double flops;
    double spent;
};

struct timed {
    struct backend backend;
    struct backend * inner;
    struct shape shapes[SHAPES];
    int32_t count;
};

static void timed_add(struct timed * t, const char * kind, int32_t out,
                      int32_t in, int32_t rows, double spent) {
    int32_t i = 0;
    while (i < t->count && (t->shapes[i].kind != kind ||
           t->shapes[i].out != out || t->shapes[i].in != in)) {
        i++;
    }
    if (i == t->count && t->count < SHAPES) {
        t->shapes[i] = (struct shape){ .kind = kind, .out = out, .in = in };
        t->count++;
    }
    if (i < t->count) {
        t->shapes[i].calls++;
        t->shapes[i].flops += 2.0 * rows * out * in;
        t->shapes[i].spent += spent;
    }
}

static void timed_gemm(struct backend * backend, const struct gemm_args * g) {
    struct timed * t = (struct timed *)backend;
    const double started = seconds();
    t->inner->gemm(t->inner, g);
    timed_add(t, "gemm", g->n, g->k, g->m, seconds() - started);
}

static void timed_blocks(struct backend * backend, const struct blocks * w,
                         const float * x, float * y) {
    struct timed * t = (struct timed *)backend;
    const double started = seconds();
    t->inner->blocks(t->inner, w, x, y);
    timed_add(t, "blocks", w->out, w->in, 1, seconds() - started);
}

static void timed_exp(struct backend * backend, float * y, const float * x,
                      int32_t n) {
    struct timed * t = (struct timed *)backend;
    t->inner->exp(t->inner, y, x, n);
}

static void timed_close(struct backend * backend) {
    (void)backend;
}

static void timed_print(struct timed * t, const char * what, double wall) {
    double sum = 0.0;
    for (int32_t i = 0; i < t->count; i++) {
        sum += t->shapes[i].spent;
    }
    printf("  %s: wall %.3f s, products %.3f s %.0f%%, the rest %.3f s "
           "%.0f%%\n", what, wall, sum, 100.0 * sum / wall, wall - sum,
           100.0 * (wall - sum) / wall);
    for (int32_t i = 0; i < t->count; i++) {
        const struct shape * s = &t->shapes[i];
        printf("    %-6s %6d x %-5d %6d calls %7.3f s %5.1f%% %7.1f "
               "GFLOPS\n", s->kind, s->out, s->in, s->calls, s->spent,
               100.0 * s->spent / wall, s->flops / s->spent / 1e9);
    }
    t->count = 0;
}

static int32_t prompt_of(struct llm * llm, const char * text) {
    llm->reset(llm);
    const int32_t room = llm->room(llm);
    llm->message(llm, "user", text);
    return room - llm->room(llm);
}

static void text_make(struct llm * llm, char * text, const char * lead) {
    const char * ask = "Continue this story for many pages.";
    int32_t words = 0;
    int32_t tokens = 0;
    while (tokens < PROMPT - 4 && words < 2000) {
        size_t used = (size_t)snprintf(text, TEXT, "%s ", lead);
        const char * at = sentence;
        words++;
        for (int32_t w = 0; w < words; w++) {
            const size_t n = strcspn(at, " ") + 1;
            used += (size_t)snprintf(text + used, TEXT - used, "%.*s",
                                     (int)n, at);
            at = at[n] != 0 ? at + n : sentence;
        }
        snprintf(text + used, TEXT - used, "\n%s", ask);
        tokens = prompt_of(llm, text);
    }
}

struct run {
    double spent;
    int32_t fed;
    int32_t count;
};

static struct run feed(struct llm * llm, const char * const * options,
                       const char * text, int32_t limit) {
    static char answer[TEXT];
    struct turn turn = { 0 };
    turn_read(&turn, options);
    turn.limit = limit;
    llm->reset(llm);
    llm->message(llm, "user", text);
    const double started = seconds();
    const int32_t count = llm->reply(llm, answer, sizeof answer, &turn);
    return (struct run){ .spent = seconds() - started, .fed = turn.fed,
                         .count = count };
}

int main(int argc, char ** argv) {
    const char * const * options = (const char * const *)argv + 1;
    static struct timed timed;
    static char cold[TEXT];
    static char warm[TEXT];
    char keys[256];
    (void)argc;
    snprintf(keys, sizeof keys, "llm gemm profile%s", turn_keys);
    options_must(options, keys);
    const char * name = option(options, "llm", NULL);
    const bool profile = option(options, "profile", NULL) != NULL;
    const double opened = seconds();
    struct backend * backend = backend_open(option(options, "gemm", NULL),
                                            options);
    must(backend != NULL, "gemm");
    timed.backend = (struct backend){
        .name = backend->name, .gemm = timed_gemm, .packed = backend->packed,
        .blocks = timed_blocks, .exp = timed_exp, .close = timed_close
    };
    timed.inner = backend;
    struct llm * llm = llm_open(name, NULL, profile ? &timed.backend :
                                backend, options);
    must(llm != NULL, "llm");
    const double ready = seconds() - opened;
    text_make(llm, cold, "Once");
    text_make(llm, warm, "Then");
    const struct run first = feed(llm, options, cold, 0);
    timed.count = 0;
    const struct run pp = feed(llm, options, warm, 0);
    if (profile) {
        timed_print(&timed, "pp", pp.spent);
    }
    const struct run tg = feed(llm, options, warm, REPLY);
    if (profile) {
        timed_print(&timed, "tg", tg.spent);
    }
    printf("%-7s %-5s %-6s open %5.2f s | pp %d: %7.1f t/s (first pass "
           "%d: %7.1f) | tg %d: %7.1f t/s, %d fed again\n",
           llm->name, backend->name, option(options, "weights", "float"),
           ready, pp.fed, pp.fed / pp.spent, first.fed,
           first.fed / first.spent, tg.count, tg.count / tg.spent, tg.fed);
    llm->close(llm);
    backend->close(backend);
    return 0;
}
