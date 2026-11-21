#ifndef ASR_H
#define ASR_H
#include "backend.h"

struct piece {
    int32_t offset;
    int32_t sample;
    float score;
};

struct pieces {
    struct piece * list;
    int32_t capacity;
    int32_t count;
};

struct asr {
    const char * name;
    int32_t (* transcribe)(struct asr * asr, const float * samples,
                           int32_t count, const char * language,
                           char * text, int32_t capacity,
                           struct pieces * pieces);
    void (* close)(struct asr * asr);
};

struct asr_entry {
    const char * name;
    const char * pack;
    const char * keys;
    struct asr * (* open)(const char * path, struct backend * backend,
                          const char * const * options);
};

struct asr * asr_open(const char * name, const char * path,
                      struct backend * backend,
                      const char * const * options);
const char * asr_listed(int32_t index);
int asr_program(const char * name, const char * const * options);

#endif
