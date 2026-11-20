#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

void must(bool ok, const char * what);
const char * guarded(void (* run)(void * context), void * context);
const char * option(const char * const * options, const char * key,
                    const char * fallback);
bool option_listed(const char * known, const char * word, size_t n);
const char * option_unknown(const char * const * options,
                            const char * known);
double seconds(void);
const uint8_t * file_map(const char * path, size_t * size);
void file_unmap(const uint8_t * data, size_t size);
bool model_path(char * path, size_t size, const char * pack);
float * floats(size_t count);
uint32_t le16(const uint8_t * p);
uint32_t le32(const uint8_t * p);
uint64_t le64(const uint8_t * p);
float half_float(const uint8_t * p);
bool json_unescape(const char * begin, const char * end, char * out,
                   size_t capacity);
const char * json_close(const char * begin, const char * end);
int32_t wav_samples(const uint8_t * data, size_t size, float ** samples);
