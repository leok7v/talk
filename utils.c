#include "llm.h"
#include <fcntl.h>
#include <math.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

enum { RATE = 16000 };

static _Thread_local jmp_buf * trap;
static _Thread_local const char * failed;

void fail(const char * what) {
    fprintf(stderr, "talk: %s failed\n", what);
    failed = what;
    if (trap != NULL) {
        longjmp(*trap, 1);
    }
    exit(1);
}

const char * guarded(void (* run)(void * context), void * context) {
    jmp_buf * const outer = trap;
    jmp_buf here;
    failed = NULL;
    if (setjmp(here) == 0) {
        trap = &here;
        run(context);
    }
    trap = outer;
    return failed;
}

const char * option(const char * const * options, const char * key,
                    const char * fallback) {
    const char * value = fallback;
    const size_t n = strlen(key);
    for (int32_t i = 0; options != NULL && options[i] != NULL; i++) {
        if (strncmp(options[i], key, n) == 0 && options[i][n] == '=') {
            value = options[i] + n + 1;
        }
    }
    return value;
}

bool option_listed(const char * known, const char * word, size_t n) {
    const char * at = known;
    bool found = false;
    while (*at != 0) {
        const size_t size = strcspn(at, " ");
        found = found || (size == n && strncmp(at, word, n) == 0);
        at += size + strspn(at + size, " ");
    }
    return found;
}

const char * option_unknown(const char * const * options,
                            const char * known) {
    const char * unknown = NULL;
    for (int32_t i = 0; options != NULL && options[i] != NULL; i++) {
        const char * equals = strchr(options[i], '=');
        const bool fine = equals != NULL &&
                          option_listed(known, options[i],
                                 (size_t)(equals - options[i]));
        unknown = unknown == NULL && !fine ? options[i] : unknown;
    }
    return unknown;
}

double seconds(void) {
    struct timespec t = { 0 };
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

const uint8_t * file_map(const char * path, size_t * size) {
    struct stat info = { 0 };
    const int fd = open(path, O_RDONLY);
    void * data = MAP_FAILED;
    if (fd >= 0 && fstat(fd, &info) == 0) {
        data = mmap(NULL, (size_t)info.st_size, PROT_READ, MAP_PRIVATE, fd,
                    0);
    }
    if (fd >= 0) {
        close(fd);
    }
    *size = data != MAP_FAILED ? (size_t)info.st_size : 0;
    return data != MAP_FAILED ? data : NULL;
}

uint8_t * file_shared(const char * path, size_t size) {
    const int fd = open(path, O_RDWR | O_CREAT, 0600);
    void * data = MAP_FAILED;
    if (fd >= 0 && ftruncate(fd, (off_t)size) == 0) {
        data = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    }
    if (fd >= 0) {
        close(fd);
    }
    return data != MAP_FAILED ? data : NULL;
}

void file_unmap(const uint8_t * data, size_t size) {
    if (data != NULL) {
        munmap((void *)(uintptr_t)data, size);
    }
}

bool model_path(char * path, size_t size, const char * pack) {
    const char * home = getenv("HOME");
    const bool here = snprintf(path, size, "models/%s", pack) < (int)size &&
                      access(path, R_OK) == 0;
    return here || (home != NULL &&
           snprintf(path, size, "%s/huggingface.co/%s", home, pack) <
           (int)size);
}

float * floats(size_t count) {
    float * p = calloc(count > 0 ? count : 1, sizeof(float));
    must(p != NULL, "memory");
    return p;
}

uint32_t le16(const uint8_t * p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8;
}

uint32_t le32(const uint8_t * p) {
    return le16(p) | le16(p + 2) << 16;
}

uint64_t le64(const uint8_t * p) {
    return (uint64_t)le32(p) | (uint64_t)le32(p + 4) << 32;
}

float half_float(const uint8_t * p) {
    const uint32_t h = le16(p);
    const int32_t exponent = (int32_t)(h >> 10 & 31);
    const float mantissa = (float)(h & 1023);
    float v = 0.0f;
    if (exponent == 0) {
        v = ldexpf(mantissa, -24);
    } else if (exponent == 31) {
        v = mantissa == 0.0f ? INFINITY : NAN;
    } else {
        v = ldexpf(mantissa + 1024.0f, exponent - 25);
    }
    return (h >> 15) != 0 ? -v : v;
}

static int hex_value(char c) {
    int value = -1;
    if ('0' <= c && c <= '9') {
        value = c - '0';
    } else if ('a' <= c && c <= 'f') {
        value = c - 'a' + 10;
    } else if ('A' <= c && c <= 'F') {
        value = c - 'A' + 10;
    }
    return value;
}

static size_t json_utf8(char * out, unsigned cp) {
    size_t size = 0;
    if (cp < 0x80) {
        out[0] = (char)cp;
        size = 1;
    } else if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        size = 2;
    } else if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        size = 3;
    } else {
        out[0] = (char)(0xF0 | (cp >> 18));
        out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[3] = (char)(0x80 | (cp & 0x3F));
        size = 4;
    }
    return size;
}

static unsigned json_codepoint(const char * p, int * valid) {
    unsigned cp = 0;
    int digits = 0;
    for (int i = 1; i <= 4; i++) {
        const int digit = hex_value(p[i]);
        cp = cp * 16 + (unsigned)(digit < 0 ? 0 : digit);
        digits += digit < 0 ? 0 : 1;
    }
    *valid = digits == 4;
    return cp;
}

static unsigned json_low(const char * next, const char * end) {
    const bool present = next + 5 < end && next[0] == '\\' &&
                         next[1] == 'u';
    int valid = 0;
    const unsigned low = present ? json_codepoint(next + 1, &valid) : 0;
    return valid && low >= 0xDC00 && low <= 0xDFFF ? low : 0;
}

static bool json_put_unicode(const char * p, const char * end,
                             char * out, size_t capacity,
                             size_t * used, size_t * take) {
    int valid = 0;
    char piece[4];
    unsigned cp = p + 5 < end ? json_codepoint(p + 1, &valid) : 0;
    const bool high = valid && cp >= 0xD800 && cp <= 0xDBFF;
    const unsigned low = high ? json_low(p + 6, end) : 0;
    if (low != 0) {
        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
    }
    const size_t size = valid ? json_utf8(piece, cp) : 0;
    const bool fits = valid && *used + size + 1 <= capacity;
    *take = 0;
    if (fits) {
        memcpy(out + *used, piece, size);
        *used += size;
        *take = low != 0 ? 12 : 6;
    }
    return fits;
}

static bool json_put_escape(const char * p, char * out,
                            size_t capacity, size_t * used) {
    const char c = *(p + 1);
    const char plain = c == 'n' ? '\n' : c == 'r' ? '\r' : c == 't' ? '\t' :
                       c == 'b' ? '\b' : c == 'f' ? '\f' : c;
    const bool fits = *used + 2 <= capacity;
    if (fits) {
        out[(*used)++] = plain;
    }
    return fits;
}

static size_t json_step(const char * p, const char * end, char * out,
                        size_t capacity, size_t * used) {
    size_t take = 0;
    if (*p == '\\' && p + 1 < end && *(p + 1) == 'u') {
        json_put_unicode(p, end, out, capacity, used, &take);
    } else if (*p == '\\' && p + 1 < end) {
        take = json_put_escape(p, out, capacity, used) ? 2 : 0;
    } else if (*used + 2 <= capacity) {
        out[(*used)++] = *p;
        take = 1;
    }
    return take;
}

bool json_unescape(const char * begin, const char * end,
                          char * out, size_t capacity) {
    const char * p = begin;
    size_t used = 0;
    size_t take = 1;
    while (take > 0 && p < end) {
        take = json_step(p, end, out, capacity, &used);
        p += take;
    }
    if (p >= end) {
        out[used] = 0;
    }
    return p >= end;
}

const char * json_close(const char * begin,
                               const char * end) {
    const char * close = begin;
    while (close < end && *close != '"') {
        if (*close == '\\') {
            close++;
        }
        close++;
    }
    const char * found = NULL;
    if (close < end) {
        found = close;
    }
    return found;
}

const char * json_skip(const char * value, const char * end) {
    const char * at = value;
    int32_t depth = 0;
    do {
        const char * close = at < end && *at == '"' ?
                             json_close(at + 1, end) : NULL;
        if (at < end && *at == '"') {
            at = close != NULL ? close + 1 : end;
        } else if (at < end) {
            depth += *at == '{' || *at == '[' ? 1 :
                     *at == '}' || *at == ']' ? -1 : 0;
            at++;
        }
    } while (at < end && (depth > 0 ||
             (depth == 0 && strchr(",}] \t\r\n", *at) == NULL)));
    return at;
}

const char * json_after(const char * close, const char * end) {
    const char * colon = close != NULL ?
                         close + 1 + strspn(close + 1, " \t\r\n") : end;
    const char * value = colon < end && *colon == ':' ?
                         colon + 1 + strspn(colon + 1, " \t\r\n") : end;
    return value < end ? value : NULL;
}

const char * json_member(const char * object, const char * end,
                         const char * key) {
    const size_t n = strlen(key);
    const char * at = object != NULL && object < end && *object == '{' ?
                      object + 1 : end;
    const char * found = NULL;
    while (found == NULL && at < end && *at != '}') {
        const char * close = *at == '"' ? json_close(at + 1, end) : NULL;
        const char * value = json_after(close, end);
        if (value != NULL && (size_t)(close - at - 1) == n &&
            memcmp(at + 1, key, n) == 0) {
            found = value;
        }
        at = value != NULL ? json_skip(value, end) : at + 1;
    }
    return found;
}

static size_t wav_chunk(const uint8_t * data, size_t size, const char * id,
                        size_t * length) {
    size_t at = 12;
    size_t found = 0;
    while (found == 0 && at + 8 <= size) {
        const size_t n = le32(data + at + 4);
        if (memcmp(data + at, id, 4) == 0) {
            found = at + 8;
            *length = n < size - found ? n : size - found;
        }
        at += 8 + n + (n & 1);
    }
    return found;
}

static float wav_frame(const uint8_t * p, int32_t channels) {
    int32_t sum = 0;
    for (int32_t c = 0; c < channels; c++) {
        sum += (int16_t)(p[c * 2] | p[c * 2 + 1] << 8);
    }
    return (float)sum / (32768.0f * (float)channels);
}

int32_t wav_samples(const uint8_t * data, size_t size, float ** samples) {
    size_t head = 0;
    size_t bytes = 0;
    int32_t count = 0;
    *samples = NULL;
    if (size >= 12 && memcmp(data, "RIFF", 4) == 0 &&
        memcmp(data + 8, "WAVE", 4) == 0) {
        const size_t format = wav_chunk(data, size, "fmt ", &head);
        const size_t body = wav_chunk(data, size, "data", &bytes);
        if (format != 0 && body != 0 && head >= 16 &&
            (data[format] | data[format + 1] << 8) == 1 &&
            (data[format + 14] | data[format + 15] << 8) == 16) {
            const int32_t channels = data[format + 2] | data[format + 3] << 8;
            const double rate = (double)le32(data + format + 4);
            const size_t frames = bytes / (size_t)(channels * 2);
            count = (int32_t)((double)frames * (double)RATE / rate);
            *samples = floats((size_t)count);
            for (int32_t i = 0; i < count && frames > 0; i++) {
                const double at = (double)i * rate / (double)RATE;
                const size_t a = (size_t)at < frames - 1 ? (size_t)at :
                                 frames - 1;
                const size_t b = a + 1 < frames ? a + 1 : a;
                const float mix = (float)(at - (double)a);
                const size_t step = (size_t)channels * 2;
                (*samples)[i] =
                    wav_frame(data + body + a * step, channels) * (1 - mix) +
                    wav_frame(data + body + b * step, channels) * mix;
            }
        }
    }
    return count;
}

int32_t turn_shift(const char * word) {
    return strcmp(word, "soft") == 0 ? SHIFT_SOFT :
           strcmp(word, "exact") == 0 ? SHIFT_EXACT : SHIFT_NONE;
}

const char * const turn_keys =
    " tokens temperature top_k top_p presence dry seed script think shift"
    " ends banned latin";

void turn_read(struct turn * turn, const char * const * options) {
    turn->limit = atoi(option(options, "tokens", "256"));
    turn->temperature = strtof(option(options, "temperature", "0.6"), NULL);
    turn->top_k = atoi(option(options, "top_k", "20"));
    turn->top_p = strtof(option(options, "top_p", "0.95"), NULL);
    turn->presence = strtof(option(options, "presence", "0"), NULL);
    turn->dry = strtof(option(options, "dry", "0.8"), NULL);
    turn->seed = strtoull(option(options, "seed", "0"), NULL, 10);
    turn->script = option(options, "script", "*");
    turn->ends = option(options, "ends", NULL);
    turn->banned = option(options, "banned", NULL);
    turn->latin = atoi(option(options, "latin", "0")) != 0;
    turn->think = atoi(option(options, "think", "0"));
    turn->shift = turn_shift(option(options, "shift", "none"));
}
