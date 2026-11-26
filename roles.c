#include "talk.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { PATH = 512, PIECES = 1024 };

#ifndef BACKEND_LIST
#define BACKEND_LIST
#endif
#ifndef ASR_LIST
#define ASR_LIST
#endif
#ifndef LLM_LIST
#define LLM_LIST
#endif
#ifndef TTS_LIST
#define TTS_LIST
#endif

#define LIST(kind, engines)                                               \
    static const struct kind##_entry * const kind##_list[] = {            \
        engines NULL                                                      \
    };

#define FIND(kind)                                                        \
    static const struct kind##_entry * kind##_find(const char * name) {   \
        const struct kind##_entry * found = NULL;                         \
        for (int32_t i = 0; kind##_list[i] != NULL; i++) {                \
            const bool match = name == NULL ? i == 0 :                    \
                strcmp(kind##_list[i]->name, name) == 0;                  \
            found = match ? kind##_list[i] : found;                       \
        }                                                                 \
        return found;                                                     \
    }

#define OPEN(kind)                                                        \
    struct kind * kind##_open(const char * name, const char * path,       \
                              struct backend * backend,                   \
                              const char * const * options) {             \
        const struct kind##_entry * entry = kind##_find(name);            \
        struct backend * used = backend != NULL ? backend : shared();     \
        char pack[PATH];                                                  \
        const bool ready = entry != NULL && used != NULL &&               \
            (path != NULL || model_path(pack, sizeof pack, entry->pack)); \
        return ready ? entry->open(path != NULL ? path : pack, used,      \
                                   options) : NULL;                       \
    }

#define KEYS(kind)                                                        \
    for (int32_t i = 0; kind##_list[i] != NULL; i++) {                    \
        keys_add(keys, size, kind##_list[i]->keys);                       \
    }

#define X(engine) extern const struct backend_entry engine##_backend;
BACKEND_LIST
#undef X
#define X(engine) extern const struct asr_entry engine##_asr;
ASR_LIST
#undef X
#define X(engine) extern const struct llm_entry engine##_llm;
LLM_LIST
#undef X
#define X(engine) extern const struct tts_entry engine##_tts;
TTS_LIST
#undef X

#define X(engine) &engine##_backend,
LIST(backend, BACKEND_LIST)
#undef X
#define X(engine) &engine##_asr,
LIST(asr, ASR_LIST)
#undef X
#define X(engine) &engine##_llm,
LIST(llm, LLM_LIST)
#undef X
#define X(engine) &engine##_tts,
LIST(tts, TTS_LIST)
#undef X

FIND(backend)
FIND(asr)
FIND(llm)
FIND(tts)

struct backend * backend_open(const char * name,
                              const char * const * options) {
    const struct backend_entry * entry = backend_find(name);
    return entry != NULL ? entry->open(options) : NULL;
}

static struct backend * shared(void) {
    static struct backend * backend;
    if (backend == NULL) {
        backend = backend_open(NULL, NULL);
    }
    return backend;
}

OPEN(asr)
OPEN(llm)
OPEN(tts)

static void keys_add(char * keys, size_t size, const char * more) {
    const char * at = more;
    while (*at != 0) {
        const size_t n = strcspn(at, " ");
        const size_t used = strlen(keys);
        if (n > 0 && !option_listed(keys, at, n)) {
            snprintf(keys + used, size - used, " %.*s", (int)n, at);
        }
        at += n + strspn(at + n, " ");
    }
}

void roles_keys(char * keys, size_t size) {
    keys[0] = 0;
    KEYS(backend)
    KEYS(asr)
    KEYS(llm)
    KEYS(tts)
}

void options_must(const char * const * options, const char * own) {
    char known[PATH];
    char keys[PATH];
    roles_keys(keys, sizeof keys);
    snprintf(known, sizeof known, "%s%s", own, keys);
    const char * unknown = option_unknown(options, known);
    if (unknown != NULL) {
        fprintf(stderr, "talk: unknown option %s\ntalk: known options: "
                "%s\n", unknown, known);
        exit(1);
    }
}

const char * asr_listed(int32_t index) {
    int32_t count = 0;
    while (asr_list[count] != NULL) {
        count++;
    }
    return index >= 0 && index < count ? asr_list[index]->name : NULL;
}

const char * llm_listed(int32_t index) {
    int32_t count = 0;
    while (llm_list[count] != NULL) {
        count++;
    }
    return index >= 0 && index < count ? llm_list[index]->name : NULL;
}

struct heard {
    int32_t size;
    double started;
};

static void hear(void * who, const char * text, int32_t size) {
    struct heard * h = who;
    fprintf(stderr, "%7.2f s +%.*s\n", seconds() - h->started,
            (int)(size - h->size), text + h->size);
    h->size = size;
}

static void pieces_print(const char * text, const struct pieces * pieces) {
    for (int32_t i = 0; i < pieces->count; i++) {
        const struct piece * p = &pieces->list[i];
        const int32_t end = i + 1 < pieces->count ? p[1].offset :
                            (int32_t)strlen(text);
        printf("%7.2f s %7.2f %.*s\n", (double)p->sample / 16000.0,
               (double)p->score, end - p->offset, text + p->offset);
    }
}

int asr_program(const char * name, const char * const * options) {
    char text[8192];
    size_t size = 0;
    float * samples = NULL;
    struct piece list[PIECES];
    struct pieces pieces = { .list = list, .capacity = PIECES };
    struct heard heard = { 0 };
    options_must(options, "wav language gemm model pieces stream");
    const uint8_t * data = file_map(option(options, "wav", "tmp/hello.wav"),
                                    &size);
    const int32_t count = data != NULL ? wav_samples(data, size, &samples) :
                          0;
    must(count > 0, "wav");
    const double opened = seconds();
    struct backend * backend = backend_open(option(options, "gemm", NULL),
                                            options);
    must(backend != NULL, "gemm");
    struct asr * asr = asr_open(name, option(options, "model", NULL), backend,
                                options);
    must(asr != NULL, name);
    const double started = seconds();
    heard.started = started;
    const int32_t steps = asr->transcribe(asr, samples, count,
                                          option(options, "language", ""),
                                          text, sizeof text,
                                          option(options, "pieces", NULL) !=
                                          NULL ? &pieces : NULL,
                                          option(options, "stream", NULL) !=
                                          NULL ? hear : NULL, &heard);
    printf("%s\n", text);
    pieces_print(text, &pieces);
    fprintf(stderr, "audio %.2f s, open %.2f s, transcribe %.2f s, "
            "%d steps\n", (double)count / 16000.0, started - opened,
            seconds() - started, steps);
    asr->close(asr);
    backend->close(backend);
    free(samples);
    return 0;
}

int llm_program(const char * name, const char * const * options) {
    char answer[8192];
    char thought[8192] = "";
    char keys[256];
    struct turn turn = {
        .thought = thought, .thought_capacity = sizeof thought
    };
    snprintf(keys, sizeof keys, "text system tools gemm model%s", turn_keys);
    options_must(options, keys);
    turn_read(&turn, options);
    const char * system = option(options, "system", "");
    const char * tools = option(options, "tools", "");
    const double opened = seconds();
    struct backend * backend = backend_open(option(options, "gemm", NULL),
                                            options);
    must(backend != NULL, "gemm");
    struct llm * llm = llm_open(name, option(options, "model", NULL), backend,
                                options);
    must(llm != NULL, name);
    const double started = seconds();
    const int32_t room = llm->room(llm);
    llm->reset(llm);
    if (tools[0] != 0) {
        llm->message(llm, "tools", tools);
    }
    if (system[0] != 0 || tools[0] != 0) {
        llm->message(llm, "system", system);
    }
    llm->message(llm, "user", option(options, "text", "Hello, who are you?"));
    const int32_t prompt = room - llm->room(llm);
    const int32_t count = llm->reply(llm, answer, sizeof answer, &turn);
    if (thought[0] != 0) {
        fprintf(stderr, "thought: %s\n", thought);
    }
    printf("%s\n", answer);
    fprintf(stderr, "open %.2f s, %d prompt and %d reply tokens in %.2f s, "
            "%.1f tokens per second\n", started - opened, prompt, count,
            seconds() - started,
            (double)(prompt + count) / (seconds() - started));
    llm->close(llm);
    backend->close(backend);
    return 0;
}
