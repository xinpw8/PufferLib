#pragma once

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ini.h"

typedef struct {
    Ini ini;
    Dict env;
} Config;

typedef struct {
    int horizon;
    int total_agents;
    int num_buffers;
    int num_atns;
    int hidden_size;
    int num_layers;
    float lr;
    float min_lr_ratio;
    bool anneal_lr;
    float beta1;
    float beta2;
    float eps;
    int minibatch_size;
    float replay_ratio;
    long total_timesteps;
    float max_grad_norm;
    float clip_coef;
    float vf_clip_coef;
    float vf_coef;
    float ent_coef;
    float min_ent_coef_ratio;
    bool anneal_ent_coef;
    float gamma;
    float gae_lambda;
    float vtrace_rho_clip;
    float vtrace_c_clip;
    float prio_alpha;
    float prio_beta0;
    bool reset_state;
    int cudagraphs;
    bool profile;
    int rank;
    int world_size;
    int gpu_id;
    int num_threads;
    int seed;
} HypersT;

static void puf_config_set_raw(Config* cfg, Dict* dict, const char* key,
        const char* raw, int must_exist) {
    if (must_exist && !dict_find(dict, key)) {
        fprintf(stderr, "config error: missing key [%s] %s\n", dict->name, key);
        exit(1);
    }

    puf_ini_set(dict, key, raw);
    if (strcmp(dict->name, "env") == 0) {
        if (must_exist && !dict_find(&cfg->env, key)) {
            fprintf(stderr, "config error: missing env key %s\n", key);
            exit(1);
        }
        puf_ini_set(&cfg->env, key, raw);
    }
}

static double puf_config_get(Config* cfg, const char* section, const char* key) {
    Dict* dict = puf_ini_section(&cfg->ini, section, 0);
    DictItem* item = dict_find(dict, key);
    if (!item) {
        fprintf(stderr, "config error: missing key [%s] %s\n", section, key);
        exit(1);
    }
    return item->value;
}

static inline int puf_config_int(Config* cfg, const char* section, const char* key) {
    return (int)puf_config_get(cfg, section, key);
}

static inline long puf_config_long(Config* cfg, const char* section, const char* key) {
    return (long)puf_config_get(cfg, section, key);
}

static inline float puf_config_float(Config* cfg, const char* section, const char* key) {
    return (float)puf_config_get(cfg, section, key);
}

static const char* puf_config_str(Config* cfg, const char* section, const char* key) {
    Dict* dict = puf_ini_section(&cfg->ini, section, 0);
    DictItem* item = dict_find(dict, key);
    if (!item || !item->str) {
        fprintf(stderr, "config error: missing string [%s] %s\n", section, key);
        exit(1);
    }
    return item->str;
}

static inline void puf_config_validate_train(Config* cfg) {
    int minibatch_size = puf_config_int(cfg, "train", "minibatch_size");
    int horizon = puf_config_int(cfg, "train", "horizon");
    int total_agents = puf_config_int(cfg, "vec", "total_agents");
    if (minibatch_size % horizon != 0) {
        fprintf(stderr, "config error: train.minibatch_size must be divisible by train.horizon\n");
        exit(1);
    }
    if (minibatch_size > horizon * total_agents) {
        fprintf(stderr, "config error: train.minibatch_size > train.horizon * vec.total_agents\n");
        exit(1);
    }
}

static inline HypersT puf_config_to_hypers(Config* cfg, int rank, int world_size, int gpu_id) {
    puf_config_validate_train(cfg);
    HypersT h = {0};
    h.total_agents = puf_config_int(cfg, "vec", "total_agents");
    h.num_buffers = puf_config_int(cfg, "vec", "num_buffers");
    h.num_threads = puf_config_int(cfg, "vec", "num_threads");
    h.horizon = puf_config_int(cfg, "train", "horizon");
    h.hidden_size = puf_config_int(cfg, "policy", "hidden_size");
    h.num_layers = puf_config_int(cfg, "policy", "num_layers");
    h.lr = puf_config_float(cfg, "train", "learning_rate");
    h.min_lr_ratio = puf_config_float(cfg, "train", "min_lr_ratio");
    h.anneal_lr = puf_config_int(cfg, "train", "anneal_lr");
    h.beta1 = puf_config_float(cfg, "train", "beta1");
    h.beta2 = puf_config_float(cfg, "train", "beta2");
    h.eps = puf_config_float(cfg, "train", "eps");
    h.minibatch_size = puf_config_int(cfg, "train", "minibatch_size");
    h.replay_ratio = puf_config_float(cfg, "train", "replay_ratio");
    h.total_timesteps = puf_config_long(cfg, "train", "total_timesteps");
    h.max_grad_norm = puf_config_float(cfg, "train", "max_grad_norm");
    h.clip_coef = puf_config_float(cfg, "train", "clip_coef");
    h.vf_clip_coef = puf_config_float(cfg, "train", "vf_clip_coef");
    h.vf_coef = puf_config_float(cfg, "train", "vf_coef");
    h.ent_coef = puf_config_float(cfg, "train", "ent_coef");
    h.min_ent_coef_ratio = puf_config_float(cfg, "train", "min_ent_coef_ratio");
    h.anneal_ent_coef = puf_config_int(cfg, "train", "anneal_ent_coef");
    h.gamma = puf_config_float(cfg, "train", "gamma");
    h.gae_lambda = puf_config_float(cfg, "train", "gae_lambda");
    h.vtrace_rho_clip = puf_config_float(cfg, "train", "vtrace_rho_clip");
    h.vtrace_c_clip = puf_config_float(cfg, "train", "vtrace_c_clip");
    h.prio_alpha = puf_config_float(cfg, "train", "prio_alpha");
    h.prio_beta0 = puf_config_float(cfg, "train", "prio_beta0");
    h.reset_state = puf_config_int(cfg, "base", "reset_state");
    h.cudagraphs = puf_config_int(cfg, "base", "cudagraphs");
    h.profile = puf_config_int(cfg, "base", "profile");
    h.rank = rank;
    h.world_size = world_size;
    h.gpu_id = gpu_id;
    h.seed = puf_config_int(cfg, "base", "seed");
    return h;
}

static void puf_config_put(Config* cfg, const char* full_key, const char* raw) {
    const char* dot = strchr(full_key, '.');
    if (!dot) {
        fprintf(stderr, "config error: expected section.key, got %s\n", full_key);
        exit(1);
    }

    const char* split = dot;
    if (strncmp(full_key, "sweep.", 6) == 0 && strchr(dot + 1, '.')) {
        split = strrchr(full_key, '.');
    }

    char section[128];
    char key[PUF_DICT_MAX_KEY];
    snprintf(section, sizeof(section), "%.*s", (int)(split - full_key), full_key);
    snprintf(key, sizeof(key), "%s", split + 1);
    puf_config_set_raw(cfg, puf_ini_section(&cfg->ini, section, 0), key, raw, 1);
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

    char full_key[4096];
    if (strchr(s, '.')) {
        snprintf(full_key, sizeof(full_key), "%s", s);
    } else {
        snprintf(full_key, sizeof(full_key), "base.%s", s);
    }
    puf_config_put(cfg, full_key, value);
}

static void puf_config_load_env(Config* cfg, const char* env_name, int argc, char** argv) {
    puf_ini_load_file(&cfg->ini, "config/default.ini");

    if (strcmp(env_name, "default") != 0) {
        char path[1024];
        snprintf(path, sizeof(path), "config/%s.ini", env_name);
        puf_ini_load_file(&cfg->ini, path);
    }

    dict_clear(&cfg->env);
    dict_copy(&cfg->env, puf_ini_section(&cfg->ini, "env", 0));
    puf_config_set_raw(cfg, puf_ini_section(&cfg->ini, "base", 0), "env_name", env_name, 1);
    for (int i = 0; i < argc; i++) {
        puf_config_apply_cli(cfg, argv[i], i);
    }
}

static inline void puf_config_copy(Config* dst, Config* src) {
    memset(dst, 0, sizeof(*dst));
    if (src->ini.num_sections) {
        dst->ini.sections = (Dict*)calloc((size_t)src->ini.num_sections, sizeof(Dict));
        if (!dst->ini.sections) {
            perror("calloc");
            exit(1);
        }
        dst->ini.num_sections = src->ini.num_sections;
    }
    for (int i = 0; i < src->ini.num_sections; i++) {
        dict_copy(&dst->ini.sections[i], &src->ini.sections[i]);
    }
    dict_copy(&dst->env, &src->env);
}

static void puf_config_free(Config* cfg) {
    for (int i = 0; i < cfg->ini.num_sections; i++) {
        dict_clear(&cfg->ini.sections[i]);
    }
    free(cfg->ini.sections);
    dict_clear(&cfg->env);
    memset(cfg, 0, sizeof(*cfg));
}
