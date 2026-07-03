#pragma once

#include <ctype.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dict.h"

typedef struct {
    Dict base;
    Dict vec;
    Dict selfplay;
    Dict env;
    Dict policy;
    Dict train;
    Dict sweep;
    Dict sweep_space;
    Dict torch;
} Config;

static DictItem* puf_config_find(Dict* dict, const char* key) {
    for (int i = 0; i < dict->size; i++) {
        if (strcmp(dict->items[i].key, key) == 0) {
            return &dict->items[i];
        }
    }
    return NULL;
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

static Dict* puf_config_section(Config* cfg, const char* section) {
    if (strcmp(section, "base") == 0) {
        return &cfg->base;
    }
    if (strcmp(section, "vec") == 0) {
        return &cfg->vec;
    }
    if (strcmp(section, "selfplay") == 0) {
        return &cfg->selfplay;
    }
    if (strcmp(section, "env") == 0) {
        return &cfg->env;
    }
    if (strcmp(section, "policy") == 0) {
        return &cfg->policy;
    }
    if (strcmp(section, "train") == 0) {
        return &cfg->train;
    }
    if (strcmp(section, "sweep") == 0) {
        return &cfg->sweep;
    }
    if (strncmp(section, "sweep.", 6) == 0) {
        return &cfg->sweep_space;
    }
    if (strcmp(section, "torch") == 0) {
        return &cfg->torch;
    }

    fprintf(stderr, "config error: unknown section [%s]\n", section);
    exit(1);
}

static Dict* puf_config_key(Config* cfg, const char* full_key,
        char* key_out, size_t key_out_size) {
    const char* dot = strchr(full_key, '.');
    if (!dot) {
        fprintf(stderr, "config error: expected section.key, got %s\n", full_key);
        exit(1);
    }

    char section[64];
    snprintf(section, sizeof(section), "%.*s", (int)(dot - full_key), full_key);
    if (strcmp(section, "sweep") == 0 && strchr(dot + 1, '.')) {
        snprintf(key_out, key_out_size, "%s", dot + 1);
        return &cfg->sweep_space;
    }

    snprintf(key_out, key_out_size, "%s", dot + 1);
    return puf_config_section(cfg, section);
}

static const char* puf_config_get(Config* cfg, const char* full_key) {
    char key[PUF_DICT_MAX_KEY];
    Dict* section = puf_config_key(cfg, full_key, key, sizeof(key));
    DictItem* item = puf_config_find(section, key);
    return item && item->str ? item->str : NULL;
}

static const char* puf_config_str(Config* cfg, const char* full_key) {
    char key[PUF_DICT_MAX_KEY];
    Dict* section = puf_config_key(cfg, full_key, key, sizeof(key));
    return dict_get_str(section, key);
}

static void puf_config_put(Config* cfg, const char* full_key, const char* val) {
    char key[PUF_DICT_MAX_KEY];
    Dict* section = puf_config_key(cfg, full_key, key, sizeof(key));
    dict_set_str(section, key, val);
    double parsed = 0;
    if (puf_config_parse_val(val, &parsed)) {
        DictItem* item = puf_config_find(section, key);
        item->value = parsed;
    }
}

static double puf_config_val(Config* cfg, const char* full_key) {
    char key[PUF_DICT_MAX_KEY];
    Dict* section = puf_config_key(cfg, full_key, key, sizeof(key));
    DictItem* item = puf_config_find(section, key);
    if (!item) {
        fprintf(stderr, "config error: %s missing\n", full_key);
        exit(1);
    }
    if (!item->str) {
        return item->value;
    }

    double val = 0;
    if (!puf_config_parse_val(item->str, &val)) {
        fprintf(stderr, "config error: %s expected number, got \"%s\"\n", full_key, item->str);
        exit(1);
    }
    return val;
}

static void puf_config_put_section(Config* cfg, const char* section_name,
        const char* key, const char* val) {
    char full_key[512];
    if (strncmp(section_name, "sweep.", 6) == 0) {
        snprintf(full_key, sizeof(full_key), "%s.%s", section_name, key);
    } else {
        snprintf(full_key, sizeof(full_key), "%s.%s", section_name, key);
    }
    puf_config_put(cfg, full_key, val);
}

static int puf_config_parse_kv(Config* cfg, const char* section,
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

    puf_config_put_section(cfg, section, key, val);
    return 1;
}

static int puf_config_load_file(Config* cfg, const char* path, bool required) {
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
            puf_config_section(cfg, section);
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

static int puf_config_apply_cli(Config* cfg, const char* arg, int idx) {
    char tmp[2048];
    if (strlen(arg) >= sizeof(tmp)) {
        fprintf(stderr, "argv:%d: argument too long\n", idx);
        return 0;
    }
    snprintf(tmp, sizeof(tmp), "%s", arg);
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
        return 0;
    }

    char full_key[PUF_DICT_MAX_KEY * 2];
    if (strchr(s, '.')) {
        if (strlen(s) >= sizeof(full_key)) {
            fprintf(stderr, "argv:%d: key too long\n", idx);
            return 0;
        }
        snprintf(full_key, sizeof(full_key), "%s", s);
    } else {
        if (strlen(s) + 5 >= sizeof(full_key)) {
            fprintf(stderr, "argv:%d: key too long\n", idx);
            return 0;
        }
        snprintf(full_key, sizeof(full_key), "base.%s", s);
    }
    puf_config_put(cfg, full_key, value);
    return 1;
}

static void puf_config_load_env(Config* cfg, const char* env_name, int argc, char** argv) {
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

static void puf_config_validate_train(Config* cfg) {
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

static void puf_config_copy(Config* dst, Config* src) {
    memcpy(dst, src, sizeof(*dst));
    Dict* dicts[] = {
        &dst->base, &dst->vec, &dst->selfplay, &dst->env, &dst->policy,
        &dst->train, &dst->sweep, &dst->sweep_space, &dst->torch,
    };
    for (int d = 0; d < 9; d++) {
        for (int i = 0; i < dicts[d]->size; i++) {
            if (dicts[d]->items[i].str) {
                dicts[d]->items[i].str = dicts[d]->items[i].str_buf;
            }
        }
    }
}

static int puf_config_count(Config* cfg) {
    return cfg->base.size + cfg->vec.size + cfg->selfplay.size + cfg->env.size
        + cfg->policy.size + cfg->train.size + cfg->sweep.size
        + cfg->sweep_space.size + cfg->torch.size;
}

typedef void (*PufConfigEachFn)(const char* full_key, DictItem* item, void* ctx);

static void puf_config_each_dict(const char* prefix, Dict* dict,
        PufConfigEachFn fn, void* ctx) {
    char full_key[PUF_DICT_MAX_KEY * 2];
    for (int i = 0; i < dict->size; i++) {
        snprintf(full_key, sizeof(full_key), "%s.%s", prefix, dict->items[i].key);
        fn(full_key, &dict->items[i], ctx);
    }
}

static void puf_config_each(Config* cfg, PufConfigEachFn fn, void* ctx) {
    puf_config_each_dict("base", &cfg->base, fn, ctx);
    puf_config_each_dict("vec", &cfg->vec, fn, ctx);
    puf_config_each_dict("selfplay", &cfg->selfplay, fn, ctx);
    puf_config_each_dict("env", &cfg->env, fn, ctx);
    puf_config_each_dict("policy", &cfg->policy, fn, ctx);
    puf_config_each_dict("train", &cfg->train, fn, ctx);
    puf_config_each_dict("sweep", &cfg->sweep, fn, ctx);
    puf_config_each_dict("sweep", &cfg->sweep_space, fn, ctx);
    puf_config_each_dict("torch", &cfg->torch, fn, ctx);
}

static void puf_config_free(Config* cfg) {
    memset(cfg, 0, sizeof(*cfg));
}
