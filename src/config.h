#pragma once

#include <ctype.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dict.h"

#define PUF_MAX_SWEEP_PARAMS 64

typedef struct {
    char key[64];
    char distribution[32];
    double min;
    double max;
    double scale;
    double mean;
    int has_mean;
    int is_integer;
} SweepParam;

typedef struct {
    Dict* sections;
    int num_sections;
    Dict env;
    SweepParam sweep_params[PUF_MAX_SWEEP_PARAMS];
    int num_sweep_params;
} Config;

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
    if (puf_config_streq_ci(raw, "true")) {
        *out = 1.0;
        return 1;
    }
    if (puf_config_streq_ci(raw, "false")) {
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

static Dict* puf_config_find_section(Config* cfg, const char* section) {
    for (int i = 0; i < cfg->num_sections; i++) {
        if (strcmp(cfg->sections[i].name, section) == 0) {
            return &cfg->sections[i];
        }
    }
    return NULL;
}

static Dict* puf_config_section(Config* cfg, const char* section) {
    Dict* found = puf_config_find_section(cfg, section);
    if (found) {
        return found;
    }
    cfg->sections = (Dict*)realloc(cfg->sections, (size_t)(cfg->num_sections + 1) * sizeof(Dict));
    if (!cfg->sections) {
        perror("realloc");
        exit(1);
    }

    Dict* dict = &cfg->sections[cfg->num_sections++];
    memset(dict, 0, sizeof(*dict));
    dict->name = dict_strdup(section);
    return dict;
}

static SweepParam* puf_config_sweep_param(Config* cfg, const char* key) {
    for (int i = 0; i < cfg->num_sweep_params; i++) {
        if (strcmp(cfg->sweep_params[i].key, key) == 0) {
            return &cfg->sweep_params[i];
        }
    }
    if (cfg->num_sweep_params >= PUF_MAX_SWEEP_PARAMS) {
        fprintf(stderr, "config error: too many sweep params\n");
        exit(1);
    }

    SweepParam* param = &cfg->sweep_params[cfg->num_sweep_params++];
    memset(param, 0, sizeof(*param));
    snprintf(param->key, sizeof(param->key), "%s", key);
    return param;
}

static void puf_config_set_sweep_param(Config* cfg, const char* section,
        const char* key, const char* raw) {
    const char* param_key = section + 6;
    SweepParam* param = puf_config_sweep_param(cfg, param_key);
    double value = 0;

    if (strcmp(key, "distribution") == 0) {
        snprintf(param->distribution, sizeof(param->distribution), "%s", raw);
    } else if (strcmp(key, "min") == 0) {
        if (!puf_config_parse_val(raw, &value)) {
            fprintf(stderr, "config error: invalid sweep min %s\n", raw);
            exit(1);
        }
        param->min = value;
    } else if (strcmp(key, "max") == 0) {
        if (!puf_config_parse_val(raw, &value)) {
            fprintf(stderr, "config error: invalid sweep max %s\n", raw);
            exit(1);
        }
        param->max = value;
    } else if (strcmp(key, "scale") == 0) {
        if (puf_config_parse_val(raw, &value)) {
            param->scale = value;
        } else if (strcmp(raw, "auto") == 0) {
            param->scale = 0.5;
        } else if (strcmp(raw, "time") == 0) {
            param->scale = -1;
        } else {
            fprintf(stderr, "config error: invalid sweep scale %s\n", raw);
            exit(1);
        }
    } else if (strcmp(key, "mean") == 0) {
        if (!puf_config_parse_val(raw, &value)) {
            fprintf(stderr, "config error: invalid sweep mean %s\n", raw);
            exit(1);
        }
        param->mean = value;
        param->has_mean = 1;
    }
}

static void puf_config_put_section(Config* cfg, const char* section,
        const char* key, const char* val, int add) {
    Dict* dict = puf_config_find_section(cfg, section);
    if (!dict) {
        if (!add) {
            fprintf(stderr, "config error: missing section [%s]\n", section);
            exit(1);
        }
        dict = puf_config_section(cfg, section);
    }
    DictItem* existing = dict_find(dict, key);
    if (!add && !existing) {
        fprintf(stderr, "config error: missing key [%s] %s\n", section, key);
        exit(1);
    }
    dict_set_str(dict, key, val);

    double parsed = 0;
    if (puf_config_parse_val(val, &parsed)) {
        dict_find(dict, key)->value = parsed;
    }

    if (strcmp(section, "env") == 0) {
        if (!add && !dict_find(&cfg->env, key)) {
            fprintf(stderr, "config error: missing env key %s\n", key);
            exit(1);
        }
        dict_set_str(&cfg->env, key, val);
        if (puf_config_parse_val(val, &parsed)) {
            dict_find(&cfg->env, key)->value = parsed;
        }
    }
    if (strncmp(section, "sweep.", 6) == 0) {
        puf_config_set_sweep_param(cfg, section, key, val);
    }
}

static Dict* puf_config_get_section(Config* cfg, const char* section) {
    for (int i = 0; i < cfg->num_sections; i++) {
        if (strcmp(cfg->sections[i].name, section) == 0) {
            return &cfg->sections[i];
        }
    }
    fprintf(stderr, "config error: missing section [%s]\n", section);
    exit(1);
}

static DictItem* puf_config_item(Config* cfg, const char* section, const char* key) {
    Dict* dict = puf_config_get_section(cfg, section);
    DictItem* item = dict_find(dict, key);
    if (!item) {
        fprintf(stderr, "config error: missing key [%s] %s\n", section, key);
        exit(1);
    }
    return item;
}

static double puf_config_get(Config* cfg, const char* section, const char* key) {
    return puf_config_item(cfg, section, key)->value;
}

static const char* puf_config_str(Config* cfg, const char* section, const char* key) {
    DictItem* item = puf_config_item(cfg, section, key);
    if (!item->str) {
        fprintf(stderr, "config error: missing string [%s] %s\n", section, key);
        exit(1);
    }
    return item->str;
}

static void puf_config_split_key(const char* full_key,
        char* section, size_t section_size, char* key, size_t key_size) {
    const char* dot = strchr(full_key, '.');
    if (!dot) {
        fprintf(stderr, "config error: expected section.key, got %s\n", full_key);
        exit(1);
    }

    if (strcmp(full_key, "sweep") == 0) {
        fprintf(stderr, "config error: invalid sweep key %s\n", full_key);
        exit(1);
    }
    if (strncmp(full_key, "sweep.", 6) == 0 && strchr(dot + 1, '.')) {
        const char* last_dot = strrchr(full_key, '.');
        snprintf(section, section_size, "%.*s", (int)(last_dot - full_key), full_key);
        snprintf(key, key_size, "%s", last_dot + 1);
    } else {
        snprintf(section, section_size, "%.*s", (int)(dot - full_key), full_key);
        snprintf(key, key_size, "%s", dot + 1);
    }
}

static void puf_config_add(Config* cfg, const char* full_key, const char* val) {
    char section[128];
    char key[PUF_DICT_MAX_KEY];
    puf_config_split_key(full_key, section, sizeof(section), key, sizeof(key));
    puf_config_put_section(cfg, section, key, val, 1);
}

static void puf_config_put(Config* cfg, const char* full_key, const char* val) {
    char section[128];
    char key[PUF_DICT_MAX_KEY];
    puf_config_split_key(full_key, section, sizeof(section), key, sizeof(key));
    puf_config_put_section(cfg, section, key, val, 0);
}

static void puf_config_parse_kv(Config* cfg, const char* section,
        char* s, const char* src, int lineno) {
    char* eq = strchr(s, '=');
    if (!eq) {
        fprintf(stderr, "%s:%d: expected key=value\n", src, lineno);
        exit(1);
    }

    *eq = 0;
    char* key = puf_config_trim(s);
    char* val = puf_config_trim(eq + 1);
    puf_config_strip_quotes(val);
    if (!*key) {
        fprintf(stderr, "%s:%d: empty key\n", src, lineno);
        exit(1);
    }

    Dict* dict = puf_config_section(cfg, section);
    if (dict_find(dict, key)) {
        puf_config_put_section(cfg, section, key, val, 0);
    } else {
        puf_config_put_section(cfg, section, key, val, 1);
    }
}

static void puf_config_load_file(Config* cfg, const char* path) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "could not open %s: %s\n", path, strerror(errno));
        exit(1);
    }

    char section[256] = "base";
    char line[2048];
    for (int n = 1; fgets(line, sizeof(line), fp); n++) {
        if (!strchr(line, '\n') && !feof(fp)) {
            fprintf(stderr, "%s:%d: line too long\n", path, n);
            fclose(fp);
            exit(1);
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

        puf_config_parse_kv(cfg, section, s, path, n);
    }

    fclose(fp);
}

static void puf_config_apply_cli(Config* cfg, const char* arg, int idx) {
    char tmp[2048];
    if (strlen(arg) >= sizeof(tmp)) {
        fprintf(stderr, "argv:%d: argument too long\n", idx);
        exit(1);
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
        exit(1);
    }

    char full_key[PUF_DICT_MAX_KEY * 2];
    if (strchr(s, '.')) {
        if (strlen(s) >= sizeof(full_key)) {
            fprintf(stderr, "argv:%d: key too long\n", idx);
            exit(1);
        }
        snprintf(full_key, sizeof(full_key), "%s", s);
    } else {
        if (strlen(s) + 5 >= sizeof(full_key)) {
            fprintf(stderr, "argv:%d: key too long\n", idx);
            exit(1);
        }
        snprintf(full_key, sizeof(full_key), "base.%s", s);
    }
    puf_config_put(cfg, full_key, value);
}

static void puf_config_load_env(Config* cfg, const char* env_name, int argc, char** argv) {
    puf_config_load_file(cfg, "config/default.ini");

    if (strcmp(env_name, "default") != 0) {
        char path[1024];
        snprintf(path, sizeof(path), "config/%s.ini", env_name);
        puf_config_load_file(cfg, path);
    }

    puf_config_put_section(cfg, "base", "env_name", env_name, 0);
    for (int i = 0; i < argc; i++) {
        puf_config_apply_cli(cfg, argv[i], i);
    }
}

static void puf_config_validate_train(Config* cfg) {
    int minibatch_size = (int)puf_config_get(cfg, "train", "minibatch_size");
    int horizon = (int)puf_config_get(cfg, "train", "horizon");
    int total_agents = (int)puf_config_get(cfg, "vec", "total_agents");
    if (minibatch_size % horizon != 0) {
        fprintf(stderr, "config error: train.minibatch_size must be divisible by train.horizon\n");
        exit(1);
    }
    if (minibatch_size > horizon * total_agents) {
        fprintf(stderr, "config error: train.minibatch_size > train.horizon * vec.total_agents\n");
        exit(1);
    }
}

static void puf_config_copy(Config* dst, Config* src) {
    *dst = *src;
    dst->sections = NULL;
    dst->num_sections = 0;
    memset(&dst->env, 0, sizeof(dst->env));
    if (src->num_sections) {
        dst->sections = (Dict*)calloc((size_t)src->num_sections, sizeof(Dict));
        if (!dst->sections) {
            perror("calloc");
            exit(1);
        }
        dst->num_sections = src->num_sections;
    }
    for (int i = 0; i < src->num_sections; i++) {
        dict_copy(&dst->sections[i], &src->sections[i]);
    }
    dict_copy(&dst->env, &src->env);
}

static void puf_config_free(Config* cfg) {
    for (int i = 0; i < cfg->num_sections; i++) {
        dict_clear(&cfg->sections[i]);
    }
    free(cfg->sections);
    dict_clear(&cfg->env);
    memset(cfg, 0, sizeof(*cfg));
}
