#pragma once
#include "llm.h"

enum { GGUF_OWNED = 1024, GGUF_BLOCK = 32, GGUF_F32 = 0, GGUF_Q4_0 = 2,
       GGUF_Q8_0 = 8,
       CHAT_MARKS = 3, CHAT_STARTS = 4096, MARK_SYSTEM = 0, MARK_USER = 1,
       MARK_REPLY = 2, SAMPLER_STEPS = 32, SAMPLER_TOKENS = 1024,
       SAMPLER_CHOICES = 8, SAMPLER_TERMS = 8, SAMPLER_TEXT = 256,
       SAMPLER_BANNED = 32, SAMPLER_MARK = 5, CHAT_ROWS = 256,
       CHAT_HEAD = 4096, CHAT_TRACKS = 64,
       STEP_END = 0, STEP_FIXED = 1, STEP_SENTENCES = 2, STEP_CHOICE = 3,
       STEP_FREE = 4, STEP_TEXT = 5, STEP_NUMBER = 6 };

struct matrix {
    const float * values;
    const uint8_t * blocks;
    const float * scale;
    int32_t bits;
};

struct table {
    const uint8_t ** key;
    int32_t * size;
    int32_t * value;
    uint32_t slots;
};

struct gguf {
    struct backend * backend;
    const uint8_t * file;
    size_t size;
    const uint8_t * cursor;
    const uint8_t * infos;
    const uint8_t * data;
    uint64_t tensor_count;
    uint64_t alignment;
    int32_t vocab_size;
    const uint8_t ** token_text;
    int32_t * token_size;
    int32_t * token_type;
    struct table vocab;
    struct table merges;
    bool packed;
    int32_t digits;
    float * owned[GGUF_OWNED];
    int32_t owned_count;
    int32_t faults;
};

struct mark {
    int32_t position;
    int32_t wanted;
    int32_t dropped;
    uint8_t * state;
};

struct store {
    size_t state_size;
    int32_t tracks;
    int32_t width;
    uint64_t model;
    const char * path;
};

struct stored {
    uint32_t magic;
    uint32_t clean;
    uint64_t model;
    uint64_t state_size;
    int32_t tracks;
    int32_t width;
    int32_t context;
    int32_t pending_count;
    int32_t fed_count;
    int32_t start_count;
    int32_t dropped;
    int32_t mode;
    int32_t wanted[CHAT_MARKS];
};

struct chat {
    void * engine;
    void (* forward)(void * engine, const int32_t * tokens, int32_t count,
                     int32_t position, int32_t row);
    void (* shift)(void * engine, int32_t keep, int32_t count,
                   int32_t rows);
    void (* save)(void * engine, uint8_t * to);
    void (* load)(void * engine, const uint8_t * from);
    bool (* alive)(void * who);
    void * who;
    int32_t context;
    int32_t capacity;
    bool overflow;
    int32_t * scratch;
    int32_t * pending;
    int32_t * fed;
    int32_t pending_count;
    int32_t fed_count;
    int32_t dropped;
    int32_t counted;
    int32_t kept;
    bool counting;
    int32_t * starts;
    int32_t start_count;
    struct mark marks[CHAT_MARKS];
    struct store store;
    struct stored * stored;
    uint8_t * slot;
    float * track[CHAT_TRACKS];
    size_t head;
    int32_t synced;
    int32_t mode;
};

struct attention {
    const float * keys;
    const float * values;
    int32_t heads;
    int32_t groups;
    int32_t head;
};

struct format {
    const char * start;
    const char * open;
    const char * middle;
    const char * close;
    const char * reply;
    const char * plain;
    const char * think;
    const char * tools_open;
    const char * tools_close;
    bool trim;
};

void chat_attend(const struct attention * a, float * mix, const float * q,
                 int32_t first, int32_t rows);
bool chat_open(struct chat * c, int32_t context, const struct store * s);
void chat_resume(struct chat * c);
void chat_commit(struct chat * c);
bool chat_save(struct chat * c, const char * path);
bool chat_load(struct chat * c, const char * path);
void chat_close(struct chat * c);
void chat_clear(struct chat * c);
void chat_token(struct chat * c, int32_t token);
void chat_text(struct chat * c, const struct gguf * g, const char * s);
void chat_special(struct chat * c, const struct gguf * g, const char * s);
void chat_begin(struct chat * c);
void chat_want(struct chat * c, int32_t slot);
void chat_feed(struct chat * c);
void chat_took(struct chat * c, int32_t token);
void chat_rewind(struct chat * c, int32_t messages);
bool chat_fit(struct chat * c, struct turn * turn);
void chat_start(struct chat * c, const struct gguf * g,
                const struct format * f);
void chat_message(struct chat * c, const struct gguf * g,
                  const struct format * f, const char * role,
                  const char * content, const char * tools);
int32_t chat_reply(struct chat * c, const struct gguf * g,
                   const struct format * f, bool think);
void chat_content(struct chat * c, const struct gguf * g,
                  const struct format * f, const char * content);

struct piece {
    int32_t at;
    int32_t count;
};

struct step {
    int32_t kind;
    int32_t low;
    int32_t high;
    int32_t choices;
    bool joined;
    struct piece plain;
    struct piece joint[SAMPLER_TERMS];
    struct piece choice[SAMPLER_CHOICES];
    char text[SAMPLER_TEXT];
};

struct sampler {
    const struct gguf * g;
    const struct chat * c;
    const struct turn * turn;
    uint64_t random;
    struct step steps[SAMPLER_STEPS + 2];
    int32_t step_count;
    int32_t pool[SAMPLER_TOKENS];
    int32_t pool_count;
    int32_t at;
    int32_t index;
    struct piece current;
    int32_t sentences;
    int32_t length;
    int32_t limbo;
    int32_t start;
    bool live[SAMPLER_CHOICES];
    char ends[SAMPLER_TERMS][SAMPLER_MARK];
    int32_t end_count;
    char banned[SAMPLER_BANNED][SAMPLER_MARK];
    int32_t banned_count;
    char tail[8];
    int32_t tail_size;
    char last;
    bool joiner;
    bool faulty;
    const char * tools;
    int32_t chosen;
};

bool sampler_open(struct sampler * s, const struct gguf * g,
                  const struct chat * c, const struct turn * turn);
int32_t sampler_next(struct sampler * s, float * logits);
void sampler_tools(struct sampler * s, const char * tools);

bool gguf_open(struct gguf * g, const char * path, struct backend * backend,
               bool packed);
void gguf_close(struct gguf * g);
const uint8_t * gguf_info(struct gguf * g, const char * name, uint32_t * type,
                          uint64_t * count);
void gguf_decode(float * out, const uint8_t * p, uint64_t count,
                 uint32_t type);
void gguf_row(float * out, const uint8_t * p, int32_t row, int32_t width,
              uint32_t type);
const float * gguf_tensor(struct gguf * g, const char * format, int32_t layer);
struct matrix gguf_matrix(struct gguf * g, const char * format, int32_t layer);
void gguf_linear(struct gguf * g, float * y, const float * x,
                 const struct matrix * w, int32_t out, int32_t in,
                 int32_t rows);
int32_t gguf_encode(const struct gguf * g, const char * s, int32_t * out,
                    int32_t capacity);
int32_t gguf_special(const struct gguf * g, const char * s, int32_t * out,
                     int32_t capacity);
int32_t gguf_text(const struct gguf * g, char * text, int32_t used,
                  int32_t capacity, int32_t token);
int32_t gguf_mark(const struct gguf * g, char * text, int32_t used,
                  int32_t capacity, int32_t token);
