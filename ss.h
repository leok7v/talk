#ifndef SS_H
#define SS_H
#include "backend.h"

struct ss {
    const char * name;
    int32_t (* separate)(struct ss * ss, const float * samples,
                         int32_t count, float ** voices, int32_t capacity);
    void (* close)(struct ss * ss);
};

struct ss_entry {
    const char * name;
    const char * pack;
    const char * keys;
    struct ss * (* open)(const char * path, struct backend * backend,
                         const char * const * options);
};

struct ss * ss_open(const char * name, const char * path,
                    struct backend * backend, const char * const * options);
int ss_program(const char * name, const char * const * options);

#endif
