#pragma once

#include <ctype.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char* key;
    char* val;
} PufConfigEntry;

typedef struct {
    char* name;
    PufConfigEntry* items;
    int len;
    int cap;
} PufConfig;

typedef struct {
    PufConfig* sections;
    int len;
    int cap;
} PufConfigFile;

static char* puf_config_strdup(const char* s) {
    size_t n = strlen(s) + 1;
    char* out = (char*)malloc(n);
    if (!out) {
        perror("malloc");
        exit(1);
    }
    memcpy(out, s, n);
    return out;
}

static char* puf_config_trim(char* s) {
    while (isspace((unsigned char)*s)) {
        s++;
    }

    char* e = s + strlen(s);
    while (e > s && isspace((unsigned char)e[-1])) {
        *--e = 0;
    }
    return s;
}

static int puf_config_streq_ci(const char* a, const char* b) {
    while (*a && *b) {
        if (tolower((unsigned char)*a++) != tolower((unsigned char)*b++)) {
            return 0;
        }
    }
    return *a == *b;
}

static void puf_config_strip_comment(char* s) {
    char* prev = 0;
    char quote = 0;
    for (; *s; s++) {
        if ((*s == '\'' || *s == '"') && (!prev || *prev != '\\')) {
            quote = quote == *s ? 0 : quote ? quote : *s;
        } else if ((*s == '#' || *s == ';') && !quote) {
            *s = 0;
            return;
        }
        prev = s;
    }
}

static void puf_config_strip_quotes(char* s) {
    size_t n = strlen(s);
    if (n >= 2 && ((s[0] == '\'' && s[n-1] == '\'') || (s[0] == '"' && s[n-1] == '"'))) {
        memmove(s, s + 1, n - 2);
        s[n - 2] = 0;
    }
}

static PufConfig* puf_config_section(PufConfigFile* f, const char* name) {
    for (int i = 0; i < f->len; i++) {
        if (strcmp(f->sections[i].name, name) == 0) {
            return &f->sections[i];
        }
    }
    return 0;
}

static PufConfig* puf_config_get_section(PufConfigFile* f, const char* name) {
    PufConfig* cfg = puf_config_section(f, name);
    if (cfg) {
        return cfg;
    }

    if (f->len == f->cap) {
        f->cap = f->cap ? 2 * f->cap : 16;
        f->sections = (PufConfig*)realloc(f->sections, (size_t)f->cap * sizeof(PufConfig));
        if (!f->sections) {
            perror("realloc");
            exit(1);
        }
    }

    cfg = &f->sections[f->len++];
    memset(cfg, 0, sizeof(*cfg));
    cfg->name = puf_config_strdup(name);
    return cfg;
}

static PufConfigEntry* puf_config_entry(PufConfig* cfg, const char* key) {
    for (int i = 0; i < cfg->len; i++) {
        if (strcmp(cfg->items[i].key, key) == 0) {
            return &cfg->items[i];
        }
    }
    return 0;
}

static void puf_config_put(PufConfig* cfg, const char* key, const char* val) {
    PufConfigEntry* e = puf_config_entry(cfg, key);
    if (!e) {
        if (cfg->len == cfg->cap) {
            cfg->cap = cfg->cap ? 2 * cfg->cap : 32;
            cfg->items = (PufConfigEntry*)realloc(cfg->items, (size_t)cfg->cap * sizeof(PufConfigEntry));
            if (!cfg->items) {
                perror("realloc");
                exit(1);
            }
        }
        e = &cfg->items[cfg->len++];
        e->key = puf_config_strdup(key);
        e->val = 0;
    } else {
        free(e->val);
    }
    e->val = puf_config_strdup(val);
}

static const char* puf_config_get(PufConfig* cfg, const char* key) {
    PufConfigEntry* e = puf_config_entry(cfg, key);
    return e ? e->val : 0;
}

static const char* puf_config_str(PufConfig* cfg, const char* key) {
    const char* out = puf_config_get(cfg, key);
    if (!out) {
        fprintf(stderr, "config error: [%s] %s missing\n", cfg->name, key);
        exit(1);
    }
    return out;
}

static int puf_config_parse_val(const char* raw, double* out) {
    if (puf_config_streq_ci(raw, "true") || puf_config_streq_ci(raw, "yes")) {
        *out = 1.0;
        return 1;
    }
    if (puf_config_streq_ci(raw, "false") || puf_config_streq_ci(raw, "no")) {
        *out = 0.0;
        return 1;
    }

    char buf[256];
    size_t j = 0;
    for (size_t i = 0; raw[i] && j + 1 < sizeof(buf); i++) {
        if (raw[i] != '_' && !isspace((unsigned char)raw[i])) {
            buf[j++] = raw[i];
        }
    }
    buf[j] = 0;

    char* end = 0;
    double v = strtod(buf, &end);
    if (!buf[0] || !end || *end) {
        return 0;
    }

    *out = v;
    return 1;
}

static double puf_config_val(PufConfig* cfg, const char* key) {
    const char* raw = puf_config_str(cfg, key);
    double val = 0;
    if (!puf_config_parse_val(raw, &val)) {
        fprintf(stderr, "config error: [%s] %s expected number, got \"%s\"\n",
            cfg->name, key, raw);
        exit(1);
    }
    return val;
}

static int puf_config_parse_kv(PufConfig* cfg, char* s, const char* src, int lineno) {
    char* eq = strchr(s, '=');
    if (!eq) {
        fprintf(stderr, "%s:%d: expected key=value\n", src, lineno);
        return 0;
    }

    *eq = 0;
    char* key = puf_config_trim(s);
    char* val = puf_config_trim(eq + 1);
    puf_config_strip_quotes(val);
    if (!*key) {
        fprintf(stderr, "%s:%d: empty key\n", src, lineno);
        return 0;
    }

    puf_config_put(cfg, key, val);
    return 1;
}

static int puf_config_load_file(PufConfigFile* f, const char* path, bool required) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        if (required) {
            fprintf(stderr, "could not open %s: %s\n", path, strerror(errno));
        }
        return !required;
    }

    PufConfig* cur = puf_config_get_section(f, "base");
    char line[2048];
    for (int n = 1; fgets(line, sizeof(line), fp); n++) {
        if (!strchr(line, '\n') && !feof(fp)) {
            fprintf(stderr, "%s:%d: line too long\n", path, n);
            fclose(fp);
            return 0;
        }

        puf_config_strip_comment(line);
        char* s = puf_config_trim(line);
        if (!*s) {
            continue;
        }

        size_t len = strlen(s);
        if (s[0] == '[' && len >= 3 && s[len - 1] == ']') {
            s[len - 1] = 0;
            cur = puf_config_get_section(f, puf_config_trim(s + 1));
            continue;
        }

        if (!puf_config_parse_kv(cur, s, path, n)) {
            fclose(fp);
            return 0;
        }
    }

    fclose(fp);
    return 1;
}

static int puf_config_apply_cli(PufConfigFile* f, const char* arg, int idx) {
    char* tmp = puf_config_strdup(arg);
    char* s = tmp;
    while (*s == '-') {
        s++;
    }

    char* eq = strchr(s, '=');
    const char* value = "true";
    if (eq) {
        *eq = 0;
        value = eq + 1;
    }

    char* dot = strchr(s, '.');
    const char* section = "base";
    char* key = s;
    if (dot) {
        *dot = 0;
        section = s;
        key = dot + 1;
    }

    for (char* p = key; *p; p++) {
        if (*p == '-') {
            *p = '_';
        }
    }

    if (!*key) {
        fprintf(stderr, "argv:%d: empty key\n", idx);
        free(tmp);
        return 0;
    }

    puf_config_put(puf_config_get_section(f, section), key, value);
    free(tmp);
    return 1;
}

static void puf_config_load_env(PufConfigFile* f, const char* env_name, int argc, char** argv) {
    if (!puf_config_load_file(f, "config/default.ini", true)) {
        exit(1);
    }

    if (strcmp(env_name, "default") != 0) {
        char path[1024];
        snprintf(path, sizeof(path), "config/%s.ini", env_name);
        if (!puf_config_load_file(f, path, true)) {
            exit(1);
        }
    }

    PufConfig* base = puf_config_get_section(f, "base");
    puf_config_put(base, "env_name", env_name);

    for (int i = 0; i < argc; i++) {
        if (!puf_config_apply_cli(f, argv[i], i)) {
            exit(1);
        }
    }
}

static void puf_config_validate_train(PufConfigFile* f) {
    PufConfig* train = puf_config_get_section(f, "train");
    PufConfig* vec = puf_config_get_section(f, "vec");
    long minibatch = (long)puf_config_val(train, "minibatch_size");
    long horizon = (long)puf_config_val(train, "horizon");
    long agents = (long)puf_config_val(vec, "total_agents");

    if (minibatch % horizon != 0) {
        fprintf(stderr, "config error: [train] minibatch_size must be divisible by horizon\n");
        exit(1);
    }
    if (minibatch > horizon * agents) {
        fprintf(stderr, "config error: [train] minibatch_size > horizon * total_agents\n");
        exit(1);
    }
}

static void puf_config_copy(PufConfigFile* dst, PufConfigFile* src) {
    memset(dst, 0, sizeof(*dst));
    for (int i = 0; i < src->len; i++) {
        PufConfig* src_sec = &src->sections[i];
        PufConfig* dst_sec = puf_config_get_section(dst, src_sec->name);
        for (int j = 0; j < src_sec->len; j++) {
            puf_config_put(dst_sec, src_sec->items[j].key, src_sec->items[j].val);
        }
    }
}

static void puf_config_free(PufConfigFile* f) {
    for (int i = 0; i < f->len; i++) {
        PufConfig* cfg = &f->sections[i];
        free(cfg->name);
        for (int j = 0; j < cfg->len; j++) {
            free(cfg->items[j].key);
            free(cfg->items[j].val);
        }
        free(cfg->items);
    }
    free(f->sections);
    memset(f, 0, sizeof(*f));
}
