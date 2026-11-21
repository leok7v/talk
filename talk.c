#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include "talk.h"

enum { port = 8081, backlog = 16,
       body_limit = 10 << 20, drain_limit = 64 << 20,
       head_limit = 8192, chunk = 16384,
       text_limit = 8192,
       path_limit = 256, method_limit = 16, engines_limit = 8 };

static const char * const system_prompt =
    "You are a helpful voice assistant. Answer in one or two short "
    "spoken sentences.";
static pthread_mutex_t asr_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t llm_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t tts_lock = PTHREAD_MUTEX_INITIALIZER;
static struct asr * engines[engines_limit];
static int32_t lead;
static struct backend * backend;
static const char * const * settings;
static struct llm * llm;
static struct llm * chats[engines_limit];
static int32_t chat_lead;
static struct tts * tts;
static const char * save;
static struct turn defaults;

struct bytes {
    char * data;
    size_t length;
    size_t capacity;
};

struct page {
    char * data;
    size_t size;
};

struct request {
    char method[method_limit];
    char path[path_limit];
    char query[path_limit];
    long length;
    struct bytes body;
};

static struct page page;

static void log_request(const char * method, const char * path,
                        int status, long long bytes) {
    fprintf(stderr, "%s %s -> %d (%lld bytes)\n", method, path,
            status, bytes);
}

static bool bytes_push(struct bytes * b, const void * data,
                       size_t size) {
    size_t grown = b->capacity == 0 ? 4096 : b->capacity;
    while (grown < b->length + size) {
        grown *= 2;
    }
    char * next = grown > b->capacity ? realloc(b->data, grown) : NULL;
    if (next != NULL) {
        b->data = next;
        b->capacity = grown;
    }
    const bool fits = b->length + size <= b->capacity;
    if (fits) {
        memcpy(b->data + b->length, data, size);
        b->length += size;
    }
    return fits;
}

static bool send_all(int fd, const void * data, size_t size) {
    size_t sent = 0;
    ssize_t n = 1;
    while (sent < size && n > 0) {
        n = send(fd, (const char *)data + sent, size - sent, 0);
        sent += n > 0 ? (size_t)n : 0;
    }
    return sent == size;
}

static const char * status_text(int status) {
    const char * text = "Error";
    switch (status) {
        case 200: text = "OK"; break;
        case 400: text = "Bad Request"; break;
        case 404: text = "Not Found"; break;
        case 409: text = "Conflict"; break;
        case 405: text = "Method Not Allowed"; break;
        case 413: text = "Payload Too Large"; break;
        case 500: text = "Internal Server Error"; break;
        case 502: text = "Bad Gateway"; break;
        case 503: text = "Service Unavailable"; break;
        default: break;
    }
    return text;
}

static bool send_json(int fd, int status, const char * json) {
    char head[256];
    const int size = snprintf(head, sizeof head,
                              "HTTP/1.1 %d %s\r\n"
                              "Content-Type: application/json\r\n"
                              "Content-Length: %zu\r\n"
                              "Connection: close\r\n\r\n",
                              status, status_text(status), strlen(json));
    return size > 0 && send_all(fd, head, (size_t)size) &&
           send_all(fd, json, strlen(json));
}

static bool send_json_error(int fd, int status,
                            const char * message) {
    char json[256];
    const int size = snprintf(json, sizeof json,
                              "{\"error\": \"%s\"}", message);
    return size > 0 && (size_t)size < sizeof json &&
           send_json(fd, status, json);
}

static const char * json_space(const char * p, const char * end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' ||
           *p == '\r')) {
        p++;
    }
    return p;
}

static const char * json_value(const char * at, const char * end) {
    const char * p = json_space(at, end);
    const char * value = NULL;
    if (p < end && *p == ':') {
        p = json_space(p + 1, end);
        if (p < end && *p == '"') {
            value = p + 1;
        }
    }
    return value;
}

static bool json_get(const char * text, size_t length,
                     const char * key, char * out, size_t capacity) {
    const char * finish = text + length;
    const size_t size = strlen(key);
    const char * at = NULL;
    char needle[64];
    if (size + 2 < sizeof needle) {
        needle[0] = '"';
        memcpy(needle + 1, key, size);
        needle[size + 1] = '"';
        at = memmem(text, length, needle, size + 2);
    }
    const char * begin = at != NULL ? json_value(at + size + 2, finish) :
                         NULL;
    const char * close = begin != NULL ? json_close(begin, finish) : NULL;
    return close != NULL && json_unescape(begin, close, out, capacity);
}

static void json_escape(struct bytes * out, const char * text) {
    for (const unsigned char * p = (const unsigned char *)text;
         *p != 0; p++) {
        char piece[8];
        size_t size = 0;
        if (*p == '"') {
            memcpy(piece, "\\\"", 2);
            size = 2;
        } else if (*p == '\\') {
            memcpy(piece, "\\\\", 2);
            size = 2;
        } else if (*p == '\n') {
            memcpy(piece, "\\n", 2);
            size = 2;
        } else if (*p == '\r') {
            memcpy(piece, "\\r", 2);
            size = 2;
        } else if (*p == '\t') {
            memcpy(piece, "\\t", 2);
            size = 2;
        } else if (*p < 0x20) {
            size = (size_t)snprintf(piece, sizeof piece,
                                    "\\u%04x", *p);
        } else {
            piece[0] = (char)*p;
            size = 1;
        }
        bytes_push(out, piece, size);
    }
}

static bool history_span(const char * body, size_t length,
                         const char ** begin, const char ** finish) {
    const char * limit = body + length;
    const char * key = memmem(body, length, "\"history\"", 9);
    const char * open = key != NULL ?
                        memchr(key, '[', (size_t)(limit - key)) : NULL;
    const char * p = open != NULL ? open + 1 : limit;
    int depth = open != NULL ? 1 : -1;
    char quote = 0;
    while (p < limit && depth > 0) {
        if (quote != 0 && *p == '\\') {
            p++;
        } else if (quote != 0) {
            quote = *p == quote ? 0 : quote;
        } else if (*p == '"') {
            quote = '"';
        } else if (*p == '[') {
            depth++;
        } else if (*p == ']') {
            depth--;
        }
        p++;
    }
    if (depth == 0) {
        *begin = open + 1;
        *finish = p - 1;
    }
    return depth == 0;
}

static long read_head(int fd, struct bytes * head) {
    long end = -1;
    ssize_t n = 1;
    while (end < 0 && n > 0 && head->length < head_limit) {
        const char * mark = memmem(head->data, head->length,
                                   "\r\n\r\n", 4);
        if (mark != NULL) {
            end = (long)(mark - head->data) + 4;
        } else {
            char part[1024];
            n = read(fd, part, sizeof part);
            n = n > 0 && bytes_push(head, part, (size_t)n) ? n : 0;
        }
    }
    return end;
}

static long header_length(const char * head) {
    const char * at = strcasestr(head, "content-length:");
    long value = -1;
    if (at != NULL) {
        value = strtol(at + 15, NULL, 10);
    }
    return value;
}

static void drain_body(int fd, size_t size) {
    char part[chunk];
    size_t left = size;
    ssize_t n = 1;
    while (left > 0 && n > 0) {
        n = read(fd, part, left < sizeof part ? left : sizeof part);
        left -= n > 0 ? (size_t)n : 0;
    }
}

static bool read_body(int fd, struct bytes * body, size_t size) {
    char part[chunk];
    size_t left = size;
    ssize_t n = 1;
    while (left > 0 && n > 0) {
        n = read(fd, part, left < sizeof part ? left : sizeof part);
        n = n > 0 && bytes_push(body, part, (size_t)n) ? n : 0;
        left -= (size_t)n;
    }
    return left == 0;
}

static int request_body(int fd, struct request * request,
                        const struct bytes * head, size_t end, size_t raw) {
    const long declared = header_length(head->data);
    const size_t length = declared > 0 ? (size_t)declared : 0;
    const size_t buffered = raw - end;
    const size_t have = buffered < length ? buffered : length;
    const size_t rest = length - have;
    int status = 400;
    request->length = (long)length;
    if (length > body_limit) {
        drain_body(fd, rest < drain_limit ? rest : drain_limit);
        status = 413;
    } else if ((have == 0 ||
                bytes_push(&request->body, head->data + end, have)) &&
               read_body(fd, &request->body, rest)) {
        status = 200;
    }
    return status;
}

static int request_read(int fd, struct request * request) {
    struct bytes head = { 0 };
    const long end = read_head(fd, &head);
    const size_t raw = head.length;
    int status = 400;
    if (end >= 0 && bytes_push(&head, "", 1) &&
        sscanf(head.data, "%15s %255s", request->method,
               request->path) == 2) {
        char * mark = strchr(request->path, '?');
        if (mark != NULL) {
            snprintf(request->query, sizeof request->query, "%s", mark + 1);
            *mark = 0;
        }
        status = request_body(fd, request, &head, (size_t)end, raw);
    }
    free(head.data);
    return status;
}

static void trim_end(char * text) {
    size_t length = strlen(text);
    while (length > 0 &&
           (text[length - 1] == ' ' || text[length - 1] == '\n' ||
            text[length - 1] == '\r' ||
            text[length - 1] == '\t')) {
        text[--length] = 0;
    }
}

static int handle_root(int fd) {
    char head[256];
    const int size = snprintf(head, sizeof head,
                              "HTTP/1.1 200 OK\r\n"
                              "Content-Type: text/html; charset=utf-8\r\n"
                              "Content-Length: %zu\r\n"
                              "Connection: close\r\n\r\n", page.size);
    return size > 0 && send_all(fd, head, (size_t)size) &&
           send_all(fd, page.data, page.size) ? 200 : 500;
}

static int handle_health(int fd) {
    return send_json(fd, 200, "{\"ok\":true}") ? 200 : 500;
}

static bool voice_ok(const char * voice) {
    const char last = voice[0] == 'F' ? '8' : '9';
    return (voice[0] == 'F' || voice[0] == 'M') && voice[1] >= '1' &&
           voice[1] <= last && voice[2] == 0;
}

static bool send_wave(int fd, const uint8_t * wave, size_t bytes) {
    char head[256];
    const int size = snprintf(head, sizeof head,
                              "HTTP/1.1 200 OK\r\n"
                              "Content-Type: audio/wav\r\n"
                              "Content-Length: %zu\r\n"
                              "Connection: close\r\n\r\n", bytes);
    return size > 0 && send_all(fd, head, (size_t)size) &&
           send_all(fd, wave, bytes);
}

static int handle_speak(int fd, const struct request * request) {
    char text[text_limit];
    char voice[16];
    int status = 400;
    const bool have_text = json_get(request->body.data,
                                    request->body.length, "text", text,
                                    sizeof text) && text[0] != 0;
    if (!json_get(request->body.data, request->body.length, "voice", voice,
                  sizeof voice) || voice[0] == 0) {
        snprintf(voice, sizeof voice, "F1");
    }
    if (!have_text) {
        send_json_error(fd, 400, "missing text");
    } else if (!voice_ok(voice)) {
        send_json_error(fd, 400, "unknown voice");
    } else {
        size_t bytes = 0;
        pthread_mutex_lock(&tts_lock);
        uint8_t * wave = tts->wave(tts, text, voice, &bytes);
        pthread_mutex_unlock(&tts_lock);
        if (wave == NULL) {
            status = 502;
            send_json_error(fd, 502, "speech backend failed");
        } else {
            status = send_wave(fd, wave, bytes) ? 200 : 500;
        }
        free(wave);
    }
    return status;
}

static bool query_get(const char * query, const char * key, char * out,
                      size_t capacity) {
    const size_t size = strlen(key);
    const char * at = query;
    const char * found = NULL;
    while (found == NULL && at != NULL) {
        const char * next = strchr(at, '&');
        found = strncmp(at, key, size) == 0 && at[size] == '=' ?
                at + size + 1 : NULL;
        at = next != NULL ? next + 1 : NULL;
    }
    out[0] = 0;
    if (found != NULL) {
        snprintf(out, capacity, "%.*s", (int)strcspn(found, "&"), found);
    }
    return found != NULL;
}

static void upload_save(const struct request * request) {
    char tag[32];
    char path[path_limit];
    query_get(request->query, "tag", tag, sizeof tag);
    const bool plain = strspn(tag, "abcdefghijklmnopqrstuvwxyz0123456789") ==
                       strlen(tag);
    snprintf(path, sizeof path, "%s%s%s", save != NULL ? save : "",
             plain && tag[0] != 0 ? "." : "", plain ? tag : "");
    FILE * sink = save != NULL ? fopen(path, "wb") : NULL;
    if (sink != NULL) {
        fwrite(request->body.data, 1, request->body.length, sink);
        fclose(sink);
    }
}

static int32_t engine_index(const char * name) {
    int32_t found = -1;
    for (int32_t i = 0; asr_listed(i) != NULL; i++) {
        found = strcmp(asr_listed(i), name) == 0 ? i : found;
    }
    return found;
}

static struct asr * engine_at(int32_t index) {
    const bool listed = index >= 0 && index < engines_limit &&
                        asr_listed(index) != NULL;
    if (listed && engines[index] == NULL) {
        engines[index] = asr_open(asr_listed(index), NULL, backend,
                                  settings);
    }
    return listed ? engines[index] : NULL;
}

static int handle_engines(int fd) {
    struct bytes json = { 0 };
    bytes_push(&json, "{\"asr\":[", 8);
    for (int32_t i = 0; asr_listed(i) != NULL; i++) {
        bytes_push(&json, i > 0 ? ",\"" : "\"", i > 0 ? 2 : 1);
        bytes_push(&json, asr_listed(i), strlen(asr_listed(i)));
        bytes_push(&json, "\"", 1);
    }
    bytes_push(&json, "],\"default\":\"", 13);
    bytes_push(&json, asr_listed(lead), strlen(asr_listed(lead)));
    bytes_push(&json, "\",\"llm\":[", 9);
    for (int32_t i = 0; llm_listed(i) != NULL; i++) {
        bytes_push(&json, i > 0 ? ",\"" : "\"", i > 0 ? 2 : 1);
        bytes_push(&json, llm_listed(i), strlen(llm_listed(i)));
        bytes_push(&json, "\"", 1);
    }
    bytes_push(&json, "],\"llm_default\":\"", 17);
    bytes_push(&json, llm_listed(chat_lead), strlen(llm_listed(chat_lead)));
    bytes_push(&json, "\"}", 2);
    const int status = bytes_push(&json, "", 1) &&
                       send_json(fd, 200, json.data) ? 200 : 500;
    free(json.data);
    return status;
}

static bool transcribe_with(struct bytes * json, int32_t index,
                            const float * samples, int32_t count,
                            const char * language, char * main) {
    char transcript[text_limit];
    char tail[48];
    pthread_mutex_lock(&asr_lock);
    struct asr * asr = engine_at(index);
    const double started = seconds();
    if (asr != NULL) {
        asr->transcribe(asr, samples, count, language, transcript,
                        sizeof transcript, NULL);
    }
    pthread_mutex_unlock(&asr_lock);
    if (asr != NULL) {
        snprintf(tail, sizeof tail, "\",\"seconds\":%.2f}",
                 seconds() - started);
        const char * open = ",{\"asr\":\"";
        const size_t first = json->data[json->length - 1] == '[' ? 1 : 0;
        bytes_push(json, open + first, 9 - first);
        bytes_push(json, asr->name, strlen(asr->name));
        bytes_push(json, "\",\"text\":\"", 10);
        json_escape(json, transcript);
        bytes_push(json, tail, strlen(tail));
    }
    if (asr != NULL && main != NULL) {
        snprintf(main, text_limit, "%s", transcript);
    }
    return asr != NULL;
}

static int handle_transcribe(int fd,
                             const struct request * request) {
    float * samples = NULL;
    const int32_t count = wav_samples((const uint8_t *)request->body.data,
                                      request->body.length, &samples);
    struct bytes json = { 0 };
    char language[8];
    char name[32];
    char main[text_limit] = "";
    int status = 400;
    int32_t made = 0;
    upload_save(request);
    query_get(request->query, "language", language, sizeof language);
    if (!query_get(request->query, "asr", name, sizeof name)) {
        snprintf(name, sizeof name, "%s", asr_listed(lead));
    }
    const bool all = strcmp(name, "all") == 0;
    const int32_t pick = all ? lead : engine_index(name);
    bytes_push(&json, "{\"texts\":[", 10);
    for (int32_t i = 0; count > 0 && asr_listed(i) != NULL; i++) {
        if (all || i == pick) {
            made += transcribe_with(&json, i, samples, count, language,
                                    i == pick ? main : NULL) ? 1 : 0;
        }
    }
    if (count <= 0) {
        send_json_error(fd, 400, "not a wav file");
    } else if (made == 0) {
        send_json_error(fd, 400, "unknown asr engine");
    } else {
        bytes_push(&json, "],\"text\":\"", 10);
        json_escape(&json, main);
        bytes_push(&json, "\"}", 2);
        status = bytes_push(&json, "", 1) &&
                 send_json(fd, 200, json.data) ? 200 : 500;
    }
    free(json.data);
    free(samples);
    return status;
}

static const char * object_end(const char * from, const char * finish) {
    const char * p = from;
    const char * end = NULL;
    int depth = 0;
    char quote = 0;
    while (end == NULL && p < finish) {
        if (quote != 0 && *p == '\\') {
            p++;
        } else if (quote != 0) {
            quote = *p == quote ? 0 : quote;
        } else if (*p == '"') {
            quote = '"';
        } else if (*p == '{') {
            depth++;
        } else if (*p == '}') {
            depth--;
            end = depth == 0 ? p + 1 : NULL;
        }
        p++;
    }
    return end != NULL ? end : finish;
}

static void history_walk(const char * body, size_t length) {
    const char * at = NULL;
    const char * finish = NULL;
    char role[16];
    char content[text_limit];
    if (history_span(body, length, &at, &finish)) {
        while (at < finish) {
            const char * open = memchr(at, '{', (size_t)(finish - at));
            const char * end = open != NULL ? object_end(open, finish) :
                               finish;
            if (open != NULL &&
                json_get(open, (size_t)(end - open), "role", role,
                         sizeof role) &&
                json_get(open, (size_t)(end - open), "content", content,
                         sizeof content)) {
                llm->message(llm, role, content);
            }
            at = end;
        }
    }
}

static bool client_alive(void * who) {
    char byte = 0;
    const ssize_t n = recv(*(int *)who, &byte, 1, MSG_PEEK | MSG_DONTWAIT);
    return n > 0 || (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK));
}

static void prompt_declare(const struct request * request) {
    llm->reset(llm);
    llm->message(llm, "system", system_prompt);
    history_walk(request->body.data, request->body.length);
}

struct words {
    char script[text_limit];
    char ends[64];
    char banned[128];
};

static void turn_request(struct turn * turn, const struct request * request,
                         struct words * w) {
    char word[32];
    if (query_get(request->query, "temperature", word, sizeof word)) {
        turn->temperature = strtof(word, NULL);
    }
    if (query_get(request->query, "top_k", word, sizeof word)) {
        turn->top_k = atoi(word);
    }
    if (query_get(request->query, "top_p", word, sizeof word)) {
        turn->top_p = strtof(word, NULL);
    }
    if (query_get(request->query, "presence", word, sizeof word)) {
        turn->presence = strtof(word, NULL);
    }
    if (query_get(request->query, "dry", word, sizeof word)) {
        turn->dry = strtof(word, NULL);
    }
    if (query_get(request->query, "latin", word, sizeof word)) {
        turn->latin = atoi(word) != 0;
    }
    if (query_get(request->query, "shift", word, sizeof word)) {
        turn->shift = turn_shift(word);
    }
    if (query_get(request->query, "think", word, sizeof word)) {
        turn->think = atoi(word);
    }
    if (query_get(request->query, "seed", word, sizeof word)) {
        turn->seed = strtoull(word, NULL, 10);
    }
    const char * body = request->body.data;
    const size_t length = request->body.length;
    if (json_get(body, length, "script", w->script, sizeof w->script)) {
        turn->script = w->script;
    }
    if (json_get(body, length, "ends", w->ends, sizeof w->ends)) {
        turn->ends = w->ends;
    }
    if (json_get(body, length, "banned", w->banned, sizeof w->banned)) {
        turn->banned = w->banned;
    }
}

static int32_t chat_index(const char * name) {
    int32_t found = -1;
    for (int32_t i = 0; llm_listed(i) != NULL; i++) {
        found = strcmp(llm_listed(i), name) == 0 ? i : found;
    }
    return found;
}

static struct llm * chat_at(int32_t index) {
    const bool listed = index >= 0 && index < engines_limit &&
                        llm_listed(index) != NULL;
    if (listed && chats[index] == NULL) {
        chats[index] = llm_open(llm_listed(index), NULL, backend, settings);
    }
    return listed ? chats[index] : NULL;
}

static int32_t chat_turn(const struct request * request, const char * text,
                         char * answer, struct turn * turn) {
    char word[32];
    const int32_t index = query_get(request->query, "llm", word, sizeof word) ?
                          chat_index(word) : chat_lead;
    int32_t made = -1;
    pthread_mutex_lock(&llm_lock);
    llm = chat_at(index);
    if (llm != NULL) {
        if (query_get(request->query, "rewind", word, sizeof word)) {
            llm->rewind(llm, atoi(word));
        }
        if (!query_get(request->query, "append", word, sizeof word)) {
            prompt_declare(request);
        }
        llm->message(llm, "user", text);
        made = llm->reply(llm, answer, text_limit, turn);
    }
    pthread_mutex_unlock(&llm_lock);
    return made;
}

static bool chat_answer(int fd, const char * answer,
                        const struct turn * turn) {
    struct bytes json = { 0 };
    char tail[48];
    snprintf(tail, sizeof tail, "\",\"dropped\":%d}", turn->dropped);
    bytes_push(&json, "{\"text\":\"", 9);
    json_escape(&json, answer);
    bytes_push(&json, "\",\"thought\":\"", 13);
    json_escape(&json, turn->thought);
    bytes_push(&json, tail, strlen(tail));
    const bool sent = bytes_push(&json, "", 1) &&
                      send_json(fd, 200, json.data);
    free(json.data);
    return sent;
}

static int handle_chat(int fd, const struct request * request) {
    struct turn turn = defaults;
    char text[text_limit];
    char answer[text_limit] = "";
    struct words words;
    char thought[text_limit] = "";
    int status = 400;
    if (json_get(request->body.data, request->body.length, "text", text,
                 sizeof text) && text[0] != 0) {
        turn_request(&turn, request, &words);
        turn.alive = client_alive;
        turn.who = &fd;
        turn.thought = thought;
        turn.thought_capacity = sizeof thought;
        const int32_t made = chat_turn(request, text, answer, &turn);
        fprintf(stderr, "chat: kept %d, fed %d, reply %d, dropped %d\n",
                turn.kept, turn.fed, made, turn.dropped);
        trim_end(answer);
        if (made < 0) {
            send_json_error(fd, 400, "unknown llm engine");
        } else if (turn.full) {
            status = 409;
            send_json_error(fd, 409, "context full");
        } else if (answer[0] == 0) {
            status = 502;
            send_json_error(fd, 502, "chat backend failed");
        } else {
            status = chat_answer(fd, answer, &turn) ? 200 : 500;
        }
    } else {
        send_json_error(fd, 400, "missing text");
    }
    return status;
}

static int route(int fd, const struct request * request) {
    const bool get = strcmp(request->method, "GET") == 0;
    const bool post = strcmp(request->method, "POST") == 0;
    const char * path = request->path;
    int status = 404;
    if (!get && !post) {
        status = 405;
        send_json_error(fd, 405, "method not allowed");
    } else if (get && strcmp(path, "/") == 0) {
        status = handle_root(fd);
    } else if (get && strcmp(path, "/health") == 0) {
        status = handle_health(fd);
    } else if (get && strcmp(path, "/api/engines") == 0) {
        status = handle_engines(fd);
    } else if (post && strcmp(path, "/api/speak") == 0) {
        status = handle_speak(fd, request);
    } else if (post && strcmp(path, "/api/transcribe") == 0) {
        status = handle_transcribe(fd, request);
    } else if (post && strcmp(path, "/api/chat") == 0) {
        status = handle_chat(fd, request);
    } else {
        send_json_error(fd, 404, "not found");
    }
    return status;
}

static int serve(int fd) {
    struct request request = { 0 };
    int status = request_read(fd, &request);
    if (status == 200) {
        status = route(fd, &request);
    } else if (status == 413) {
        send_json_error(fd, 413, "request body too large");
    } else {
        send_json_error(fd, 400, "bad request");
    }
    log_request(request.method, request.path, status,
                (long long)request.body.length);
    free(request.body.data);
    return status;
}

static void * serve_client(void * arg) {
    int fd = *(int *)arg;
    free(arg);
    serve(fd);
    close(fd);
    return NULL;
}

static struct page page_load(const char * path) {
    struct page loaded = { 0 };
    FILE * file = fopen(path, "rb");
    must(file != NULL && fseek(file, 0, SEEK_END) == 0, path);
    const long size = ftell(file);
    must(size >= 0 && fseek(file, 0, SEEK_SET) == 0, path);
    loaded.data = malloc((size_t)size + 1);
    must(loaded.data != NULL, "memory");
    must(fread(loaded.data, 1, (size_t)size, file) == (size_t)size, path);
    fclose(file);
    loaded.data[size] = 0;
    loaded.size = (size_t)size;
    return loaded;
}

static void client_start(int listener) {
    int * client = malloc(sizeof(int));
    pthread_t thread;
    must(client != NULL, "memory");
    *client = accept(listener, NULL, NULL);
    if (*client < 0) {
        free(client);
    } else if (pthread_create(&thread, NULL, serve_client, client) == 0) {
        pthread_detach(thread);
    } else {
        serve_client(client);
    }
}

static int serve_forever(const char * const * options) {
    const int listener = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr = {
        .sin_family = AF_INET, .sin_port = htons(port),
        .sin_addr.s_addr = htonl(INADDR_LOOPBACK)
    };
    int reuse = 1;
    signal(SIGPIPE, SIG_IGN);
    page = page_load(option(options, "page", "index.html"));
    must(listener >= 0, "socket");
    setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof reuse);
    must(bind(listener, (struct sockaddr *)&addr, sizeof addr) == 0 &&
         listen(listener, backlog) == 0, "bind 127.0.0.1:8081");
    fprintf(stderr, "talk: listening on 127.0.0.1:%d\n", port);
    while (listener >= 0) {
        client_start(listener);
    }
    return 0;
}

int main(int argc, char ** argv) {
    const char * const * options = (const char * const *)argv + 1;
    (void)argc;
    char keys[256];
    snprintf(keys, sizeof keys, "page save gemm asr llm tts%s", turn_keys);
    options_must(options, keys);
    turn_read(&defaults, options);
    save = option(options, "save", NULL);
    const char * named = option(options, "asr", NULL);
    backend = backend_open(option(options, "gemm", NULL), options);
    must(backend != NULL, "gemm");
    settings = options;
    lead = named != NULL ? engine_index(named) : 0;
    must(engine_at(lead) != NULL, "asr");
    const char * chat = option(options, "llm", NULL);
    chat_lead = chat != NULL ? chat_index(chat) : 0;
    llm = chat_at(chat_lead);
    must(llm != NULL, "llm");
    tts = tts_open(option(options, "tts", NULL), NULL, backend, options);
    must(tts != NULL, "tts");
    fprintf(stderr, "talk: %s, %s, %s on %s\n", asr_listed(lead), llm->name,
            tts->name, backend->name);
    return serve_forever(options);
}
