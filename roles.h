#pragma once
#include "utils.h"

#define INSTALL(kind, engine, ...)                                        \
    const struct kind##_entry engine##_##kind = {                         \
        .name = #engine, __VA_ARGS__                                      \
    };

void roles_keys(char * keys, size_t size);
void options_must(const char * const * options, const char * own);
