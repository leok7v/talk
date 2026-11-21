#include "transformer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { TOP = 64, PIECE = 64, RECENT = 64, MATCH = 50, LONG = 48,
       INVALID = 0, NORMAL = 1, PLAIN_END = 2, JOINT_START = 3,
       JOINT_NEXT = 4 };

static const char * const default_ends =
    ".!?\xE3\x80\x82\xEF\xBC\x81\xEF\xBC\x9F";

static const char * const default_banned =
    "\n\r*#()[]:;\"`\xE2\x80\x94\xE2\x80\x93\xE2\x80\x9C\xE2\x80\x9D"
    "\xEF\xBC\x9A\xEF\xBC\x9B";

struct view {
    int32_t token;
    int32_t term;
    int32_t joint;
    bool text;
    bool ends;
    bool clean;
};

struct pick {
    int32_t token;
    int32_t kind;
    int32_t term;
};

static double uniform(struct sampler * s) {
    s->random ^= s->random << 13;
    s->random ^= s->random >> 7;
    s->random ^= s->random << 17;
    return (double)(s->random >> 11) / 9007199254740992.0;
}

static struct piece encode(struct sampler * s, const char * text) {
    const int32_t room = SAMPLER_TOKENS - s->pool_count;
    const int32_t count = gguf_special(s->g, text, s->pool + s->pool_count,
                                       room);
    const struct piece p = { .at = s->pool_count, .count = count };
    s->faulty = s->faulty || count >= room || count == 0;
    s->pool_count += count;
    return p;
}

static const char * literal(struct sampler * s, const char * p, char * out,
                            size_t capacity) {
    const char * end = p[0] == '"' ? json_close(p + 1, p + strlen(p)) : NULL;
    s->faulty = s->faulty || end == NULL ||
                !json_unescape(p + 1, end, out, capacity);
    out[s->faulty ? 0 : strlen(out)] = 0;
    return end != NULL ? end + 1 : p + strlen(p);
}

static const char * blank(const char * p) {
    return p + strspn(p, " \t\r\n");
}

static const char * sentences(struct sampler * s, const char * p,
                              struct step * step) {
    char * next = NULL;
    step->kind = STEP_SENTENCES;
    step->low = 1;
    step->high = 1;
    if (p[1] == '{') {
        step->low = (int32_t)strtol(p + 2, &next, 10);
        step->high = *next == ',' ? (int32_t)strtol(next + 1, &next, 10) :
                     step->low;
        s->faulty = s->faulty || *next != '}';
    }
    s->faulty = s->faulty || step->low < 1 || step->high < step->low;
    return next != NULL && *next == '}' ? next + 1 : p + 1;
}

static const char * choice(struct sampler * s, const char * p,
                           struct step * step) {
    char text[SAMPLER_TEXT];
    const char * at = blank(p + 1);
    step->kind = STEP_CHOICE;
    while (*at == '"' && step->choices < SAMPLER_CHOICES) {
        at = blank(literal(s, at, text, sizeof text));
        step->choice[step->choices++] = encode(s, text);
        at = *at == '|' ? blank(at + 1) : at;
    }
    s->faulty = s->faulty || *at != ')' || step->choices == 0;
    return *at == ')' ? at + 1 : at + strlen(at);
}

static const char * fixed(struct sampler * s, const char * p,
                          struct step * step) {
    const char * end = literal(s, p, step->text, sizeof step->text);
    step->kind = STEP_FIXED;
    step->plain = encode(s, step->text);
    return end;
}

static void joints(struct sampler * s) {
    char joint[SAMPLER_TEXT + 8];
    for (int32_t i = 1; i < s->step_count; i++) {
        struct step * step = &s->steps[i];
        const bool after = step->kind == STEP_FIXED &&
                           s->steps[i - 1].kind == STEP_SENTENCES;
        for (int32_t t = 0; after && t < s->end_count; t++) {
            snprintf(joint, sizeof joint, "%s%s", s->ends[t], step->text);
            step->joint[t] = encode(s, joint);
        }
        step->joined = after;
    }
}

static bool ended(const struct sampler * s) {
    int32_t open = 0;
    for (int32_t i = 0; i < s->step_count; i++) {
        const bool value = s->steps[i].kind == STEP_TEXT ||
                           s->steps[i].kind == STEP_NUMBER;
        const bool closed = i + 1 < s->step_count &&
                            s->steps[i + 1].kind == STEP_FIXED;
        open += value && !closed ? 1 : 0;
    }
    return open == 0;
}

static void parse(struct sampler * s, const char * script) {
    const char * p = blank(script);
    while (*p != 0 && !s->faulty && s->step_count < SAMPLER_STEPS) {
        struct step * step = &s->steps[s->step_count++];
        if (*p == '"') {
            p = fixed(s, p, step);
        } else if (*p == 'S') {
            p = sentences(s, p, step);
        } else if (*p == '(') {
            p = choice(s, p, step);
        } else if (*p == '*' || *p == 'T' || *p == 'N') {
            step->kind = *p == '*' ? STEP_FREE : *p == 'T' ? STEP_TEXT :
                         STEP_NUMBER;
            p++;
        } else {
            s->faulty = true;
        }
        p = blank(p);
    }
    s->faulty = s->faulty || *p != 0 || s->step_count == 0 || !ended(s);
    joints(s);
}

static void enter(struct sampler * s) {
    const struct step * step = &s->steps[s->at];
    s->index = 0;
    s->sentences = 0;
    s->length = 0;
    s->limbo = -1;
    if (s->at < s->step_count && step->kind == STEP_FIXED) {
        s->current = step->plain;
    }
    for (int32_t i = 0; i < SAMPLER_CHOICES; i++) {
        s->live[i] = true;
    }
}

static int32_t marks_read(char marks[][SAMPLER_MARK], int32_t limit,
                          const char * text) {
    int32_t count = 0;
    const char * p = text;
    while (*p != 0 && count < limit) {
        const uint8_t lead = (uint8_t)*p;
        const size_t wide = lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 :
                            lead >= 0xC0 ? 2 : 1;
        const size_t size = strnlen(p, wide);
        memcpy(marks[count], p, size);
        marks[count++][size] = 0;
        p += size;
    }
    return count;
}

bool sampler_open(struct sampler * s, const struct gguf * g,
                  const struct chat * c, const struct turn * turn) {
    const char * script = turn->script != NULL && turn->script[0] != 0 ?
                          turn->script : "*";
    memset(s, 0, sizeof *s);
    s->g = g;
    s->c = c;
    s->turn = turn;
    s->start = c->fed_count;
    s->random = turn->seed != 0 ? turn->seed :
                (uint64_t)(seconds() * 1e6) | 1;
    s->end_count = marks_read(s->ends, SAMPLER_TERMS, turn->ends != NULL ?
                              turn->ends : default_ends);
    s->banned_count = marks_read(s->banned, SAMPLER_BANNED,
                                 turn->banned != NULL ? turn->banned :
                                 default_banned);
    parse(s, script);
    enter(s);
    return !s->faulty;
}

static int32_t partial(const char * bytes, int32_t size) {
    int32_t back = 0;
    while (back < 3 && back < size &&
           ((uint8_t)bytes[size - 1 - back] & 0xC0) == 0x80) {
        back++;
    }
    const uint8_t lead = back < size ? (uint8_t)bytes[size - 1 - back] : 0;
    const int32_t need = lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 :
                         lead >= 0xC0 ? 2 : 1;
    return back < size && need > back + 1 ? back + 1 : 0;
}

static int32_t compose(const struct sampler * s, char * out, int32_t token) {
    memcpy(out, s->tail, (size_t)s->tail_size);
    return gguf_text(s->g, out, s->tail_size, PIECE, token);
}

static void tail_push(struct sampler * s, int32_t token) {
    char bytes[PIECE];
    const int32_t size = compose(s, bytes, token);
    const int32_t left = partial(bytes, size);
    const bool joiner = size - left >= 3 &&
                        memcmp(bytes + size - left - 3, "\xE2\x80\x8D",
                               3) == 0;
    s->joiner = size > s->tail_size ? joiner : s->joiner;
    s->last = size > left ? bytes[size - left - 1] : s->last;
    memcpy(s->tail, bytes + size - left, (size_t)left);
    s->tail_size = left;
}

static bool has(const char * bytes, int32_t size, const char * word) {
    return memmem(bytes, (size_t)size, word, strlen(word)) != NULL;
}

static int32_t ending(const struct sampler * s, const char * bytes,
                      int32_t size) {
    int32_t found = -1;
    for (int32_t t = 0; t < s->end_count; t++) {
        const int32_t n = (int32_t)strlen(s->ends[t]);
        const bool last = size >= n &&
                          memcmp(bytes + size - n, s->ends[t], (size_t)n) == 0;
        found = last ? t : found;
    }
    return found;
}

static bool eastern(const char * bytes, int32_t size) {
    int32_t found = 0;
    for (int32_t i = 0; i + 2 < size; i++) {
        const uint8_t a = (uint8_t)bytes[i];
        const uint8_t b = (uint8_t)bytes[i + 1];
        const uint32_t code = a >= 0xE0 && a < 0xF0 ?
            (uint32_t)(a & 15) << 12 | (uint32_t)(b & 63) << 6 |
            ((uint8_t)bytes[i + 2] & 63) : 0;
        found += (code >= 0x2E80 && code <= 0xD7FF) ||
                 (code >= 0xFF00 && code <= 0xFFEF) ? 1 : 0;
    }
    return found > 0;
}

static bool clean(const struct sampler * s, const char * bytes,
                  int32_t size) {
    int32_t faults = s->turn->latin && eastern(bytes, size) ? 1 : 0;
    for (int32_t i = 0; i < s->banned_count; i++) {
        faults += has(bytes, size, s->banned[i]) ? 1 : 0;
    }
    for (int32_t t = 0; t < s->end_count; t++) {
        faults += has(bytes, size, s->ends[t]) ? 1 : 0;
    }
    return faults == 0;
}

static bool joined(const struct sampler * s) {
    return s->at + 1 < s->step_count && s->steps[s->at + 1].joined;
}

static int32_t joint_first(const struct sampler * s, int32_t token) {
    const struct step * next = &s->steps[s->at + 1];
    int32_t found = -1;
    for (int32_t t = 0; joined(s) && t < s->end_count; t++) {
        found = s->pool[next->joint[t].at] == token ? t : found;
    }
    return found;
}

static struct view look(const struct sampler * s, int32_t token) {
    char bytes[PIECE];
    const int32_t size = compose(s, bytes, token);
    const bool text = size > s->tail_size;
    const int32_t term = text ? ending(s, bytes, size) : -1;
    const int32_t body = term >= 0 ? size - (int32_t)strlen(s->ends[term]) :
                         size;
    const char before = body > s->tail_size ? bytes[body - 1] : s->last;
    const bool number = before >= '0' && before <= '9' && s->length < LONG;
    return (struct view){
        .token = token, .term = term, .joint = joint_first(s, token),
        .text = text, .clean = clean(s, bytes, body),
        .ends = term >= 0 && s->length > 0 && !number
    };
}

static bool enough(const struct sampler * s) {
    return s->sentences + 1 >= s->steps[s->at].low;
}

static bool limbo_next(const struct sampler * s, int32_t token) {
    const struct step * next = &s->steps[s->at + 1];
    return s->limbo >= 0 && s->sentences >= s->steps[s->at].low &&
           token == s->pool[next->joint[s->limbo].at + 1];
}

static bool plain_end(const struct sampler * s, const struct view * v) {
    const bool more = s->sentences + 1 < s->steps[s->at].high;
    return v->ends && v->clean && (more || !joined(s));
}

static bool joint_start(const struct sampler * s, const struct view * v) {
    const bool whole = v->term < 0 || (v->ends && v->clean);
    return v->joint >= 0 && s->length > 0 && enough(s) && whole;
}

static bool normal(const struct sampler * s, const struct view * v) {
    const bool forced = s->length >= LONG && s->tail_size == 0 && !s->joiner;
    return v->term < 0 && v->text && v->clean && !forced;
}

static struct pick sentence_kind(const struct sampler * s, int32_t token) {
    const struct view v = look(s, token);
    struct pick p = { .token = token, .kind = INVALID, .term = -1 };
    if (limbo_next(s, token)) {
        p.kind = JOINT_NEXT;
    } else if (plain_end(s, &v)) {
        p.kind = PLAIN_END;
        p.term = v.joint == v.term && enough(s) ? v.term : -1;
    } else if (joint_start(s, &v)) {
        p.kind = JOINT_START;
        p.term = v.joint;
    } else if (normal(s, &v)) {
        p.kind = NORMAL;
    }
    return p;
}

static struct pick choice_kind(const struct sampler * s, int32_t token) {
    const struct step * step = &s->steps[s->at];
    struct pick p = { .token = token, .kind = INVALID, .term = -1 };
    for (int32_t i = 0; i < step->choices; i++) {
        const struct piece * c = &step->choice[i];
        if (s->live[i] && s->index < c->count &&
            s->pool[c->at + s->index] == token) {
            p.kind = NORMAL;
        }
    }
    return p;
}

static struct pick value_kind(const struct sampler * s, int32_t token) {
    const struct step * next = &s->steps[s->at + 1];
    const bool number = s->steps[s->at].kind == STEP_NUMBER;
    struct pick p = { .token = token, .kind = INVALID, .term = -1 };
    char bytes[PIECE];
    const int32_t size = gguf_text(s->g, bytes, 0, PIECE, token);
    int32_t faults = size == 0 ? 1 : 0;
    for (int32_t i = 0; i < size; i++) {
        const char c = bytes[i];
        const bool digit = c != 0 && strchr("0123456789.-", c) != NULL;
        const bool plain = c != '"' && c != '\\' && (uint8_t)c >= 0x20;
        faults += (number ? digit : plain) ? 0 : 1;
    }
    if (token == s->pool[next->plain.at] && s->length > 0) {
        p.kind = JOINT_START;
    } else if (faults == 0 && s->length < LONG) {
        p.kind = NORMAL;
    }
    return p;
}

static struct pick kind_of(const struct sampler * s, int32_t token) {
    const int32_t kind = s->steps[s->at].kind;
    struct pick p = { .token = token, .kind = NORMAL, .term = -1 };
    if (kind == STEP_SENTENCES) {
        p = sentence_kind(s, token);
    } else if (kind == STEP_CHOICE) {
        p = choice_kind(s, token);
    } else if (kind == STEP_TEXT || kind == STEP_NUMBER) {
        p = value_kind(s, token);
    }
    return p;
}

static void presence(const struct sampler * s, float * logits) {
    const int32_t n = s->c->fed_count;
    const int32_t from = n - RECENT > s->start ? n - RECENT : s->start;
    for (int32_t i = from; i < n && s->turn->presence != 0.0f; i++) {
        logits[s->c->fed[i]] -= s->turn->presence;
    }
}

static bool breaker(const struct sampler * s, int32_t token) {
    char bytes[PIECE];
    const int32_t size = gguf_text(s->g, bytes, 0, PIECE, token);
    return size == 0 || memchr(bytes, '\n', (size_t)size) != NULL;
}

static void dry(const struct sampler * s, float * logits) {
    const int32_t * fed = s->c->fed;
    const int32_t n = s->c->fed_count;
    for (int32_t i = 0; i + 1 < n && s->turn->dry != 0.0f; i++) {
        int32_t match = 0;
        while (match < MATCH && match <= i &&
               fed[i - match] == fed[n - 1 - match] &&
               !breaker(s, fed[i - match])) {
            match++;
        }
        if (match >= 2) {
            const float cost = s->turn->dry * powf(1.75f, (float)(match - 2));
            const int32_t token = fed[i + 1];
            const float kept = logits[token] - cost;
            logits[token] = kept < logits[token] ? kept : logits[token];
        }
    }
}

static int32_t top(const float * logits, int32_t vocab, int32_t * ids,
                   int32_t wanted) {
    int32_t count = 0;
    for (int32_t i = 0; i < vocab; i++) {
        if (count < wanted || logits[i] > logits[ids[count - 1]]) {
            int32_t j = count < wanted ? count++ : count - 1;
            while (j > 0 && logits[ids[j - 1]] < logits[i]) {
                ids[j] = ids[j - 1];
                j--;
            }
            ids[j] = i;
        }
    }
    return count;
}

static struct pick widest(const struct sampler * s, const float * logits) {
    struct pick best = { .token = -1, .kind = INVALID, .term = -1 };
    for (int32_t i = 0; i < s->g->vocab_size; i++) {
        if (best.token < 0 || logits[i] > logits[best.token]) {
            const struct pick p = kind_of(s, i);
            best = p.kind != INVALID ? p : best;
        }
    }
    return best;
}

static double weigh(const struct sampler * s, const float * logits,
                    const int32_t * ids, int32_t i) {
    const double heat = s->turn->temperature > 0.0f ?
                        (double)s->turn->temperature : 1.0;
    return exp((double)(logits[ids[i]] - logits[ids[0]]) / heat);
}

static int32_t choose(struct sampler * s, const double * weight,
                      int32_t made, double total) {
    double at = s->turn->temperature > 0.0f ? uniform(s) * total : 0.0;
    int32_t chosen = 0;
    while (chosen + 1 < made && at >= weight[chosen]) {
        at -= weight[chosen++];
    }
    return chosen;
}

static struct pick draw(struct sampler * s, const float * logits) {
    const bool greedy = s->turn->temperature <= 0.0f;
    const int32_t wanted = !greedy && s->turn->top_k > 0 &&
                           s->turn->top_k < TOP ? s->turn->top_k : TOP;
    const double nucleus = greedy || s->turn->top_p <= 0.0f ? 1.0 :
                           (double)s->turn->top_p;
    int32_t ids[TOP];
    struct pick valid[TOP];
    double weight[TOP];
    double total = 0.0;
    double mass = 0.0;
    double seen = 0.0;
    int32_t made = 0;
    const int32_t count = top(logits, s->g->vocab_size, ids, wanted);
    for (int32_t i = 0; i < count; i++) {
        mass += weigh(s, logits, ids, i);
    }
    for (int32_t i = 0; i < count; i++) {
        const struct pick p = kind_of(s, ids[i]);
        if (p.kind != INVALID && (seen < nucleus * mass || made == 0)) {
            valid[made] = p;
            weight[made] = weigh(s, logits, ids, i);
            total += weight[made++];
        }
        seen += weigh(s, logits, ids, i);
    }
    return made > 0 ? valid[choose(s, weight, made, total)] :
           widest(s, logits);
}

static void advance(struct sampler * s) {
    s->at++;
    enter(s);
}

static void fixed_from(struct sampler * s, struct piece piece,
                       int32_t index) {
    advance(s);
    s->current = piece;
    s->index = index;
    if (s->index >= s->current.count) {
        advance(s);
    }
}

static void sentence_took(struct sampler * s, struct pick p) {
    const struct step * step = &s->steps[s->at];
    const struct step * next = &s->steps[s->at + 1];
    if (p.kind == NORMAL) {
        s->limbo = -1;
        s->length++;
    } else if (p.kind == PLAIN_END) {
        s->sentences++;
        s->length = 0;
        s->limbo = p.term >= 0 && next->joint[p.term].count > 1 ? p.term : -1;
        if (s->sentences >= step->high) {
            advance(s);
        }
    } else if (p.kind == JOINT_START) {
        fixed_from(s, next->joint[p.term], 1);
    } else if (p.kind == JOINT_NEXT) {
        fixed_from(s, next->joint[s->limbo], 2);
    }
}

static void choice_took(struct sampler * s, int32_t token) {
    const struct step * step = &s->steps[s->at];
    bool complete = false;
    for (int32_t i = 0; i < step->choices; i++) {
        const struct piece * c = &step->choice[i];
        s->live[i] = s->live[i] && s->index < c->count &&
                     s->pool[c->at + s->index] == token;
        complete = complete || (s->live[i] && s->index + 1 == c->count);
        s->chosen = s->live[i] && s->index + 1 == c->count ? i : s->chosen;
    }
    s->index++;
    if (complete) {
        advance(s);
    }
}

static int32_t fixed_next(struct sampler * s) {
    const int32_t token = s->pool[s->current.at + s->index++];
    if (s->index >= s->current.count) {
        advance(s);
    }
    return token;
}

static struct step * step_new(struct sampler * s, int32_t kind) {
    struct step * step = &s->steps[s->step_count];
    s->faulty = s->faulty || s->step_count >= SAMPLER_STEPS;
    s->step_count += s->step_count < SAMPLER_STEPS ? 1 : 0;
    memset(step, 0, sizeof *step);
    step->kind = kind;
    return step;
}

static void step_fixed(struct sampler * s, const char * text) {
    struct step * step = step_new(s, STEP_FIXED);
    snprintf(step->text, sizeof step->text, "%s", text);
    step->plain = encode(s, step->text);
}

static void step_choice(struct sampler * s, struct step * step,
                        const char * text) {
    if (step->choices < SAMPLER_CHOICES) {
        step->choice[step->choices++] = encode(s, text);
    }
}

static bool json_string(const char * value, const char * end, char * out,
                        size_t capacity) {
    const char * close = value != NULL && value < end && *value == '"' ?
                         json_close(value + 1, end) : NULL;
    out[0] = 0;
    return close != NULL && json_unescape(value + 1, close, out, capacity);
}

static const char * tool_at(const char * tools, int32_t index) {
    const char * end = tools + strlen(tools);
    const char * at = strchr(tools, '[');
    const char * found = NULL;
    int32_t seen = 0;
    at = at != NULL ? at + 1 : end;
    while (found == NULL && at < end && *at != ']') {
        const char * inner = *at == '{' ? json_member(at, end, "function") :
                             NULL;
        if (*at == '{' && seen++ == index) {
            found = inner != NULL ? inner : at;
        }
        at = *at == '{' ? json_skip(at, end) : at + 1;
    }
    return found;
}

static void steps_clear(struct sampler * s) {
    s->step_count = 0;
    s->pool_count = 0;
    s->at = 0;
}

static bool wanted(const char * list, const char * end, const char * key) {
    char quoted[SAMPLER_TEXT / 2 + 4];
    const char * last = list != NULL ? json_skip(list, end) : end;
    const char * found = NULL;
    snprintf(quoted, sizeof quoted, "\"%s\"", key);
    found = list != NULL ? strstr(list, quoted) : NULL;
    return list == NULL || (found != NULL && found < last);
}

static const char * argument(struct sampler * s, const char * at,
                             const char * end, char * text,
                             const char * list) {
    char key[SAMPLER_TEXT / 2];
    char type[16];
    const char * object = json_after(json_close(at + 1, end), end);
    const size_t used = strlen(text);
    json_string(at, end, key, sizeof key);
    json_string(json_member(object, end, "type"), end, type, sizeof type);
    const bool number = strcmp(type, "integer") == 0 ||
                        strcmp(type, "number") == 0;
    const bool flag = strcmp(type, "boolean") == 0;
    if (wanted(list, end, key)) {
        snprintf(text + used, SAMPLER_TEXT - used, "%s\"%s\": %s",
                 s->step_count > 0 ? ", " : "", key,
                 number || flag ? "" : "\"");
        step_fixed(s, text);
        struct step * step = step_new(s, flag ? STEP_CHOICE : number ?
                                      STEP_NUMBER : STEP_TEXT);
        if (flag) {
            step_choice(s, step, "true");
            step_choice(s, step, "false");
        }
        snprintf(text, SAMPLER_TEXT, "%s", number || flag ? "" : "\"");
    }
    return object != NULL ? json_skip(object, end) : end;
}

static void arguments(struct sampler * s) {
    const char * tool = tool_at(s->tools, s->chosen);
    const char * end = s->tools + strlen(s->tools);
    const char * shape = json_member(tool, end, "parameters");
    const char * all = json_member(shape, end, "properties");
    const char * list = json_member(shape, end, "required");
    const char * at = all != NULL ? all + 1 : end;
    char text[SAMPLER_TEXT] = "\", \"arguments\": {";
    steps_clear(s);
    s->tools = NULL;
    while (at < end && *at != '}') {
        at = *at == '"' ? argument(s, at, end, text, list) : at + 1;
    }
    snprintf(text + strlen(text), sizeof text - strlen(text),
             "}}\n]\n</tool_call>");
    step_fixed(s, text);
    enter(s);
}

void sampler_tools(struct sampler * s, const char * tools) {
    const bool open = s->at >= s->step_count ||
                      s->steps[s->at].kind == STEP_FREE;
    char name[SAMPLER_TEXT / 2];
    if (open && tools != NULL && tool_at(tools, 0) != NULL) {
        const char * end = tools + strlen(tools);
        steps_clear(s);
        step_fixed(s, "\n[\n  {\"name\": \"");
        struct step * names = step_new(s, STEP_CHOICE);
        for (int32_t i = 0; i < SAMPLER_CHOICES; i++) {
            const char * tool = tool_at(tools, i);
            if (json_string(json_member(tool, end, "name"), end, name,
                            sizeof name)) {
                step_choice(s, names, name);
            }
        }
        s->tools = tools;
        enter(s);
    }
}

static void value_took(struct sampler * s, struct pick p) {
    if (p.kind == JOINT_START) {
        fixed_from(s, s->steps[s->at + 1].plain, 1);
    } else {
        s->length++;
    }
}

int32_t sampler_next(struct sampler * s, float * logits) {
    if (s->at >= s->step_count && s->tools != NULL) {
        arguments(s);
    }
    const int32_t kind = s->at < s->step_count ? s->steps[s->at].kind :
                         STEP_END;
    int32_t token = -1;
    if (kind == STEP_FIXED) {
        token = fixed_next(s);
    } else if (kind != STEP_END) {
        presence(s, logits);
        dry(s, logits);
        const struct pick p = draw(s, logits);
        token = p.kind != INVALID ? p.token : -1;
        if (token >= 0 && kind == STEP_SENTENCES) {
            sentence_took(s, p);
        } else if (token >= 0 && kind == STEP_CHOICE) {
            choice_took(s, token);
        } else if (token >= 0 && (kind == STEP_TEXT || kind == STEP_NUMBER)) {
            value_took(s, p);
        }
    }
    if (token >= 0) {
        tail_push(s, token);
    }
    return token;
}
