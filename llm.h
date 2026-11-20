#pragma once
#include "backend.h"

struct turn {
    int32_t limit;
    float temperature;
    int32_t top_k;
    float top_p;
    float presence;
    float dry;
    uint64_t seed;
    const char * script;
    const char * ends;
    const char * banned;
    bool latin;
    int32_t think;
    char * thought;
    int32_t thought_capacity;
    bool (* alive)(void * who);
    void * who;
    int32_t shift;
    int32_t kept;
    int32_t fed;
    int32_t dropped;
    bool full;
};

enum { SHIFT_NONE = 0, SHIFT_SOFT = 1, SHIFT_EXACT = 2 };

extern const char * const turn_keys;
void turn_read(struct turn * turn, const char * const * options);
int32_t turn_shift(const char * word);

struct llm {
    const char * name;
    void (* reset)(struct llm * llm);
    void (* message)(struct llm * llm, const char * role,
                     const char * content);
    int32_t (* count)(struct llm * llm, const char * role,
                      const char * content);
    int32_t (* room)(struct llm * llm);
    int32_t (* reply)(struct llm * llm, char * text, int32_t capacity,
                      struct turn * turn);
    void (* rewind)(struct llm * llm, int32_t messages);
    void (* close)(struct llm * llm);
};

struct llm_entry {
    const char * name;
    const char * pack;
    const char * keys;
    struct llm * (* open)(const char * path, struct backend * backend,
                          const char * const * options);
};

struct llm * llm_open(const char * name, const char * path,
                      struct backend * backend,
                      const char * const * options);
const char * llm_listed(int32_t index);
int llm_program(const char * name, const char * const * options);
