#ifndef TTS_H
#define TTS_H
#include "backend.h"

struct tts {
    const char * name;
    int32_t rate;
    uint8_t * (* wave)(struct tts * tts, const char * text,
                       const char * voice, size_t * bytes);
    int32_t (* stream)(struct tts * tts, const char * text,
                       const char * voice,
                       void (* heard)(void * who, const float * samples,
                                      int32_t count),
                       void * who);
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
