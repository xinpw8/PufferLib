#ifndef OSRS_LAB_H
#define OSRS_LAB_H

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    ENCOUNTER_LAB_OPTIONAL_INT_UNSET = 0,
    ENCOUNTER_LAB_OPTIONAL_INT_SET,
} EncounterLabOptionalIntKind;

typedef struct {
    EncounterLabOptionalIntKind kind;
    int value;
} EncounterLabOptionalInt;

static inline EncounterLabOptionalInt encounter_lab_optional_int_unset(void) {
    return (EncounterLabOptionalInt){ .kind = ENCOUNTER_LAB_OPTIONAL_INT_UNSET };
}

static inline EncounterLabOptionalInt encounter_lab_optional_int_set(int value) {
    return (EncounterLabOptionalInt){
        .kind = ENCOUNTER_LAB_OPTIONAL_INT_SET,
        .value = value,
    };
}

typedef struct {
    char* data;
    size_t len;
    size_t cap;
    const char* owner_label;
} EncounterLabString;

static inline void encounter_lab_abort(const char* owner_label, const char* fmt, ...) {
    va_list args;
    fprintf(stderr, "%s: ", owner_label);
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    abort();
}

static inline void encounter_lab_string_init(
    EncounterLabString* out, const char* owner_label
) {
    out->len = 0;
    out->cap = 4096;
    out->owner_label = owner_label;
    out->data = (char*)malloc(out->cap);
    if (!out->data) encounter_lab_abort(owner_label, "out of memory");
    out->data[0] = '\0';
}

static inline void encounter_lab_string_reserve(EncounterLabString* out, size_t need) {
    if (need <= out->cap) return;
    size_t next = out->cap;
    while (next < need) {
        if (next > SIZE_MAX / 2)
            encounter_lab_abort(out->owner_label, "json output too large");
        next *= 2;
    }
    char* data = (char*)realloc(out->data, next);
    if (!data) encounter_lab_abort(out->owner_label, "out of memory");
    out->data = data;
    out->cap = next;
}

static inline void encounter_lab_string_append(EncounterLabString* out, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    va_list copy;
    va_copy(copy, args);
    int needed = vsnprintf(NULL, 0, fmt, copy);
    va_end(copy);
    if (needed < 0) encounter_lab_abort(out->owner_label, "json formatting failed");
    encounter_lab_string_reserve(out, out->len + (size_t)needed + 1);
    int written = vsnprintf(out->data + out->len, out->cap - out->len, fmt, args);
    va_end(args);
    if (written != needed)
        encounter_lab_abort(out->owner_label, "json formatting length mismatch");
    out->len += (size_t)written;
}

static inline int encounter_lab_parse_int_value(
    const char* owner_label, const char* value
) {
    char* end = NULL;
    errno = 0;
    long parsed = strtol(value, &end, 10);
    if (errno != 0 || !end || *end != '\0' ||
            parsed < INT32_MIN || parsed > INT32_MAX) {
        encounter_lab_abort(owner_label, "invalid integer %s", value);
    }
    return (int)parsed;
}

static inline uint32_t encounter_lab_parse_seed_value(
    const char* owner_label, const char* value
) {
    char* end = NULL;
    errno = 0;
    unsigned long parsed = strtoul(value, &end, 10);
    if (errno != 0 || !end || *end != '\0' || parsed > UINT32_MAX)
        encounter_lab_abort(owner_label, "invalid seed %s", value);
    return (uint32_t)parsed;
}

static inline EncounterLabOptionalInt encounter_lab_parse_optional_full_int(
    const char* owner_label, const char* value
) {
    if (strcmp(value, "full") == 0) return encounter_lab_optional_int_unset();
    return encounter_lab_optional_int_set(
        encounter_lab_parse_int_value(owner_label, value));
}

static inline char* encounter_lab_trim(char* s) {
    while (*s && isspace((unsigned char)*s)) s++;
    char* end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) end--;
    *end = '\0';
    return s;
}

static inline void encounter_lab_parse_key_value(
    const char* owner_label, char* token, const char** key, const char** value
) {
    char* eq = strchr(token, '=');
    if (!eq || eq == token || eq[1] == '\0')
        encounter_lab_abort(owner_label, "expected key=value token, got %s", token);
    *eq = '\0';
    *key = token;
    *value = eq + 1;
}

static inline char* encounter_lab_next_token(const char* owner_label, char** cursor) {
    if (!cursor || !*cursor) encounter_lab_abort(owner_label, "null token cursor");
    char* start = *cursor + strspn(*cursor, " \t\r\n");
    if (*start == '\0') {
        *cursor = start;
        return NULL;
    }
    char* end = start + strcspn(start, " \t\r\n");
    if (*end != '\0') {
        *end = '\0';
        *cursor = end + 1;
    } else {
        *cursor = end;
    }
    return start;
}

typedef struct {
    char* buffer;
    char* cursor;
    const char* command;
} EncounterLabLine;

static inline EncounterLabLine encounter_lab_line_begin(
    const char* owner_label, const char* line
) {
    if (!line) encounter_lab_abort(owner_label, "null script line");
    size_t len = strlen(line);
    char* buffer = (char*)malloc(len + 1);
    if (!buffer) encounter_lab_abort(owner_label, "out of memory");
    memcpy(buffer, line, len + 1);
    char* text = encounter_lab_trim(buffer);
    EncounterLabLine out = { .buffer = buffer, .cursor = text, .command = NULL };
    if (*text == '\0' || *text == '#') return out;
    out.command = encounter_lab_next_token(owner_label, &out.cursor);
    return out;
}

typedef struct {
    const char* name;
    int kind;
} EncounterLabCommandAlias;

static inline int encounter_lab_lookup_command_kind(
    const char* owner_label,
    const char* command,
    const EncounterLabCommandAlias* aliases,
    size_t alias_count
) {
    for (size_t i = 0; i < alias_count; i++) {
        if (strcmp(command, aliases[i].name) == 0) return aliases[i].kind;
    }
    encounter_lab_abort(owner_label, "unknown script command %s", command);
    return 0;
}

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t state_size;
    uint32_t reserved;
} EncounterSnapshotFrame;

static inline void encounter_snapshot_write_frame(
    void* snapshot,
    size_t snapshot_size,
    uint32_t magic,
    uint32_t version,
    size_t state_size
) {
    memset(snapshot, 0, snapshot_size);
    EncounterSnapshotFrame* frame = (EncounterSnapshotFrame*)snapshot;
    frame->magic = magic;
    frame->version = version;
    frame->state_size = (uint32_t)state_size;
}

static inline const EncounterSnapshotFrame* encounter_snapshot_validate_frame(
    const char* owner_label,
    const void* data,
    size_t actual_size,
    size_t expected_size,
    uint32_t magic,
    uint32_t version,
    size_t state_size
) {
    if (actual_size != expected_size) {
        fprintf(stderr, "%s: bad snapshot size %zu (expected %zu)\n",
            owner_label, actual_size, expected_size);
        abort();
    }
    const EncounterSnapshotFrame* frame = (const EncounterSnapshotFrame*)data;
    if (frame->magic != magic || frame->version != version) {
        fprintf(stderr, "%s: bad magic/version (got 0x%08x v%u, want 0x%08x v%u)\n",
            owner_label, frame->magic, frame->version, magic, version);
        abort();
    }
    if (frame->state_size != state_size) {
        fprintf(stderr, "%s: state size mismatch (got %u, want %zu)\n",
            owner_label, frame->state_size, state_size);
        abort();
    }
    return frame;
}

static inline void encounter_snapshot_copy_state_to(
    void* snapshot, size_t state_offset, const void* state, size_t state_size
) {
    memcpy((uint8_t*)snapshot + state_offset, state, state_size);
}

static inline void encounter_snapshot_copy_state_from(
    void* state, const void* snapshot, size_t state_offset, size_t state_size
) {
    memcpy(state, (const uint8_t*)snapshot + state_offset, state_size);
}

static inline void encounter_write_terminal_status_text(
    int episode_over,
    int winner,
    int win_outcome,
    const char* win_text,
    const char* killed_by_name,
    char* out,
    size_t cap
) {
    if (cap == 0) return;
    out[0] = '\0';
    if (!episode_over) return;
    if (winner == win_outcome) {
        snprintf(out, cap, "%s", win_text);
        return;
    }
    snprintf(out, cap, "Killed by %s", killed_by_name);
}

typedef int (*EncounterLabApplyLineAllocJsonFn)(
    void* lab_state, const char* line, char** out_json);

static inline int encounter_apply_lab_command_dump_wrapper(
    void* lab_state,
    const char* line,
    int dump_result,
    EncounterLabApplyLineAllocJsonFn apply_line_alloc_json
) {
    char* json = NULL;
    int result = apply_line_alloc_json(lab_state, line, &json);
    if (result == dump_result && json) {
        printf("%s\n", json);
        fflush(stdout);
    }
    free(json);
    return result == dump_result ? 1 : 0;
}

#endif
