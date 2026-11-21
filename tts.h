#ifndef TTS_H
#define TTS_H
#include "backend.h"

struct tts {
    const char * name;
    uint8_t * (* wave)(struct tts * tts, const char * text,
                       const char * voice, size_t * bytes);
    void (* close)(struct tts * tts);
};

struct tts_entry {
    const char * name;
    const char * pack;
    const char * keys;
    struct tts * (* open)(const char * path, struct backend * backend,
                          const char * const * options);
};

struct tts * tts_open(const char * name, const char * path,
                      struct backend * backend,
                      const char * const * options);

#endif
