#pragma once

#include <ctype.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dict.h"

static char* puf_config_strdup(const char* s) {
    return dict_strdup(s);
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
    if (n >= 2 && ((s[0] == '\'' && s[n - 1] == '\'') || (s[0] == '"' && s[n - 1] == '"'))) {
        memmove(s, s + 1, n - 2);
        s[n - 2] = 0;
    }
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

static const char* puf_config_get(Dict* cfg, const char* key) {
    DictItem* item = dict_get_unsafe(cfg, key);
    return item ? item->str : NULL;
}

static const char* puf_config_str(Dict* cfg, const char* key) {
    const char* out = puf_config_get(cfg, key);
    if (!out) {
        fprintf(stderr, "config error: %s missing\n", key);
        exit(1);
    }
    return out;
}

static void puf_config_put(Dict* cfg, const char* key, const char* val) {
    dict_set_str(cfg, key, val);
    double parsed = 0;
    if (puf_config_parse_val(val, &parsed)) {
        dict_get(cfg, key)->value = parsed;
    }
}

static double puf_config_val(Dict* cfg, const char* key) {
    const char* raw = puf_config_str(cfg, key);
    double val = 0;
    if (!puf_config_parse_val(raw, &val)) {
        fprintf(stderr, "config error: %s expected number, got \"%s\"\n", key, raw);
        exit(1);
    }
    return val;
}

static void puf_config_join_key(char* out, size_t out_size,
        const char* section, const char* key) {
    if (strcmp(section, "base") == 0 && strchr(key, '.')) {
        snprintf(out, out_size, "%s", key);
    } else {
        snprintf(out, out_size, "%s.%s", section, key);
    }
}

static int puf_config_parse_kv(Dict* cfg, const char* section,
        char* s, const char* src, int lineno) {
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

    char full_key[512];
    puf_config_join_key(full_key, sizeof(full_key), section, key);
    puf_config_put(cfg, full_key, val);
    return 1;
}

static int puf_config_load_file(Dict* cfg, const char* path, bool required) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        if (required) {
            fprintf(stderr, "could not open %s: %s\n", path, strerror(errno));
        }
        return !required;
    }

    char section[256] = "base";
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
            snprintf(section, sizeof(section), "%s", puf_config_trim(s + 1));
            continue;
        }

        if (!puf_config_parse_kv(cfg, section, s, path, n)) {
            fclose(fp);
            return 0;
        }
    }

    fclose(fp);
    return 1;
}

static int puf_config_apply_cli(Dict* cfg, const char* arg, int idx) {
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

    for (char* p = s; *p; p++) {
        if (*p == '-') {
            *p = '_';
        }
    }

    if (!*s) {
        fprintf(stderr, "argv:%d: empty key\n", idx);
        free(tmp);
        return 0;
    }

    char full_key[512];
    if (strchr(s, '.')) {
        snprintf(full_key, sizeof(full_key), "%s", s);
    } else {
        snprintf(full_key, sizeof(full_key), "base.%s", s);
    }

    puf_config_put(cfg, full_key, value);
    free(tmp);
    return 1;
}

static void puf_config_load_env(Dict* cfg, const char* env_name, int argc, char** argv) {
    if (!puf_config_load_file(cfg, "config/default.ini", true)) {
        exit(1);
    }

    if (strcmp(env_name, "default") != 0) {
        char path[1024];
        snprintf(path, sizeof(path), "config/%s.ini", env_name);
        if (!puf_config_load_file(cfg, path, true)) {
            exit(1);
        }
    }

    puf_config_put(cfg, "base.env_name", env_name);
    for (int i = 0; i < argc; i++) {
        if (!puf_config_apply_cli(cfg, argv[i], i)) {
            exit(1);
        }
    }
}

static void puf_config_validate_train(Dict* cfg) {
    long minibatch = (long)puf_config_val(cfg, "train.minibatch_size");
    long horizon = (long)puf_config_val(cfg, "train.horizon");
    long agents = (long)puf_config_val(cfg, "vec.total_agents");

    if (minibatch % horizon != 0) {
        fprintf(stderr, "config error: train.minibatch_size must be divisible by train.horizon\n");
        exit(1);
    }
    if (minibatch > horizon * agents) {
        fprintf(stderr, "config error: train.minibatch_size > train.horizon * vec.total_agents\n");
        exit(1);
    }
}

static void puf_config_copy(Dict* dst, Dict* src) {
    dict_copy(dst, src);
}

static void puf_config_free(Dict* cfg) {
    dict_clear(cfg);
}
