#include "talk.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { PATH = 512, PIECES = 1024, VOICES = 8 };

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
#ifndef SS_LIST
#define SS_LIST
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
#define X(engine) extern const struct ss_entry engine##_ss;
SS_LIST
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
#define X(engine) &engine##_ss,
LIST(ss, SS_LIST)
#undef X

FIND(backend)
FIND(asr)
FIND(llm)
FIND(tts)
FIND(ss)

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
OPEN(ss)

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
    KEYS(ss)
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

int asr_run(const char * name, const char * const * options) {
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

int ss_run(const char * name, const char * const * options) {
    char path[PATH];
    size_t size = 0;
    float * samples = NULL;
    float * voices[VOICES] = { 0 };
    options_must(options, "wav out gemm model");
    const char * out = option(options, "out", "tmp/ss");
    const uint8_t * data = file_map(option(options, "wav", "tmp/hello.wav"),
                                    &size);
    const int32_t count = data != NULL ? wav_samples(data, size, &samples) :
                          0;
    must(count > 0, "wav");
    struct backend * backend = backend_open(option(options, "gemm", NULL),
                                            options);
    must(backend != NULL, "gemm");
    struct ss * ss = ss_open(name, option(options, "model", NULL), backend,
                             options);
    must(ss != NULL, name);
    const double started = seconds();
    const int32_t made = ss->separate(ss, samples, count, voices, VOICES);
    const double spent = seconds() - started;
    for (int32_t k = 0; k < made; k++) {
        snprintf(path, sizeof path, "%s.%d.wav", out, k + 1);
        must(wav_write(path, voices[k], count, 16000), path);
        printf("%s\n", path);
        free(voices[k]);
    }
    fprintf(stderr, "audio %.2f s, %d voices, separate %.2f s\n",
            (double)count / 16000.0, made, spent);
    ss->close(ss);
    backend->close(backend);
    free(samples);
    return 0;
}

struct sound {
    float * samples;
    int32_t count;
    double started;
    double first;
};

static void listen(void * who, const float * samples, int32_t count) {
    struct sound * s = who;
    s->first = s->count == 0 ? seconds() - s->started : s->first;
    s->samples = realloc(s->samples,
                         (size_t)(s->count + count) * sizeof(float));
    must(s->samples != NULL, "memory");
    memcpy(s->samples + s->count, samples, (size_t)count * sizeof(float));
    s->count += count;
}

static bool bytes_write(const char * path, const uint8_t * data,
                        size_t size) {
    FILE * sink = fopen(path, "wb");
    const bool done = sink != NULL && fwrite(data, 1, size, sink) == size;
    if (sink != NULL) {
        fclose(sink);
    }
    return done;
}

int tts_run(const char * name, const char * const * options) {
    size_t bytes = 0;
    struct sound sound = { 0 };
    options_must(options, "text voice out stream gemm model");
    const char * text = option(options, "text", "Hello world.");
    const char * voice = option(options, "voice", "F1");
    const char * out = option(options, "out", "tmp/speak.wav");
    const bool streamed = option(options, "stream", NULL) != NULL;
    struct backend * backend = backend_open(option(options, "gemm", NULL),
                                            options);
    must(backend != NULL, "gemm");
    struct tts * tts = tts_open(name, option(options, "model", NULL), backend,
                                options);
    must(tts != NULL, name);
    sound.started = seconds();
    uint8_t * file = streamed ? NULL : tts->wave(tts, text, voice, &bytes);
    const int32_t made = streamed ?
        tts->stream(tts, text, voice, listen, &sound) :
        file != NULL ? (int32_t)((bytes - 44) / 2) : -1;
    const double spent = seconds() - sound.started;
    must(made > 0, "speech");
    must(streamed ? wav_write(out, sound.samples, sound.count, tts->rate) :
         bytes_write(out, file, bytes), out);
    fprintf(stderr, "%s: %.2f s of audio in %.2f s, the first audio after "
            "%.2f s\n", out, (double)made / (double)tts->rate, spent,
            streamed ? sound.first : spent);
    free(file);
    free(sound.samples);
    tts->close(tts);
    backend->close(backend);
    return 0;
}

int llm_run(const char * name, const char * const * options) {
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
