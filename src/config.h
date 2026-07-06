#pragma once

#include <stdbool.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ini.h"

#define TRAIN_RESULT_MAX_POINTS 64

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

static void puf_config_assert(int ok, const char* fmt, ...) {
    if (ok) {
        return;
    }

    va_list args;
    fprintf(stderr, "config error: ");
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    exit(1);
}

static void puf_config_set_raw(Config* cfg, Dict* dict, const char* key,
        const char* raw, int must_exist) {
    puf_config_assert(!must_exist || dict_find(dict, key),
        "missing key [%s] %s", dict->name, key);

    puf_ini_set(dict, key, raw);
    if (strcmp(dict->name, "env") == 0) {
        puf_config_assert(!must_exist || dict_find(&cfg->env, key),
            "missing env key %s", key);
        puf_ini_set(&cfg->env, key, raw);
    }
}

static double puf_config_get(Config* cfg, const char* section, const char* key) {
    Dict* dict = puf_ini_section(&cfg->ini, section, 0);
    DictItem* item = dict_find(dict, key);
    puf_config_assert(item != NULL, "missing key [%s] %s", section, key);
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
    puf_config_assert(item != NULL && item->str != NULL,
        "missing string [%s] %s", section, key);
    return item->str;
}

static float puf_config_sweep_num(Dict* dict, const char* key) {
    const char* raw = dict_get_str(dict, key);
    double value = 0;
    puf_config_assert(puf_ini_parse_val(raw, &value),
        "invalid numeric field [%s] %s = %s", dict->name, key, raw);
    return (float)value;
}

static int puf_config_sweep_space_type(Dict* dict, int* is_integer) {
    const char* dist = dict_get_str(dict, "distribution");
    *is_integer = 0;
    if (strcmp(dist, "uniform") == 0) {
        return 0;
    }
    if (strcmp(dist, "int_uniform") == 0) {
        *is_integer = 1;
        return 0;
    }
    if (strcmp(dist, "uniform_pow2") == 0) {
        *is_integer = 1;
        return 2;
    }
    if (strcmp(dist, "log_normal") == 0) {
        return 1;
    }
    if (strcmp(dist, "logit_normal") == 0) {
        return 3;
    }

    puf_config_assert(0, "invalid sweep distribution [%s] %s", dict->name, dist);
    return 0;
}

static inline void puf_config_validate(Config* cfg) {
    int minibatch_size = puf_config_int(cfg, "train", "minibatch_size");
    int horizon = puf_config_int(cfg, "train", "horizon");
    int total_agents = puf_config_int(cfg, "vec", "total_agents");
    int train_gpus = puf_config_int(cfg, "train", "gpus");
    puf_config_assert(train_gpus >= 1, "train.gpus must be >= 1");
    puf_config_assert(minibatch_size % horizon == 0,
        "train.minibatch_size must be divisible by train.horizon");
    puf_config_assert(minibatch_size <= horizon * total_agents,
        "train.minibatch_size > train.horizon * vec.total_agents");

    int league = puf_config_int(cfg, "sweep", "league");
    const char* metric = puf_config_str(cfg, "sweep", "metric");
    puf_config_assert(league || strcmp(metric, "score") == 0,
        "native sweep currently scores env/score, got env/%s", metric);

    const char* metric_dist = puf_config_str(cfg, "sweep", "metric_distribution");
    puf_config_assert(strcmp(metric_dist, "linear") == 0 || strcmp(metric_dist, "logit") == 0,
        "sweep.metric_distribution must be linear or logit");

    const char* goal = puf_config_str(cfg, "sweep", "goal");
    puf_config_assert(strcmp(goal, "maximize") == 0 || strcmp(goal, "minimize") == 0,
        "sweep.goal must be maximize or minimize");

    int max_runs = puf_config_int(cfg, "sweep", "max_runs");
    int downsample = puf_config_int(cfg, "sweep", "downsample");
    int sweep_gpus = puf_config_int(cfg, "sweep", "gpus");
    puf_config_assert(max_runs >= 1, "sweep.max_runs must be >= 1");
    puf_config_assert(downsample >= 1 && downsample <= TRAIN_RESULT_MAX_POINTS,
        "sweep.downsample must be in [1, %d]", TRAIN_RESULT_MAX_POINTS);
    puf_config_assert(sweep_gpus >= 0, "sweep.gpus must be >= 0");
    puf_config_assert(sweep_gpus == 0 || sweep_gpus >= train_gpus + league,
        "sweep.gpus must be >= train.gpus%s", league ? " + 1 for league sweeps" : "");
    puf_config_assert(puf_config_float(cfg, "sweep", "max_suggestion_cost") > 0,
        "sweep.max_suggestion_cost must be > 0");

    float q = puf_config_float(cfg, "sweep", "early_stop_quantile");
    puf_config_assert(q > 0 && q < 1, "sweep.early_stop_quantile must be in (0, 1)");
    puf_config_assert(!league || strcmp(puf_config_str(cfg, "base", "env_name"), "robocode") == 0,
        "league sweep currently requires robocode");

    for (int i = 0; i < cfg->ini.num_sections; i++) {
        Dict* dict = &cfg->ini.sections[i];
        if (strncmp(dict->name, "sweep.", 6) != 0) {
            continue;
        }

        const char* sweep_key = dict->name + 6;
        const char* dot = strrchr(sweep_key, '.');
        puf_config_assert(dot && dot != sweep_key && dot[1],
            "expected section [sweep.<section>.<key>]");

        int is_integer = 0;
        puf_config_sweep_space_type(dict, &is_integer);

        float min_v = puf_config_sweep_num(dict, "min");
        float max_v = puf_config_sweep_num(dict, "max");
        puf_config_assert(max_v > min_v, "[%s] max must be greater than min", dict->name);

        const char* scale = dict_get_str(dict, "scale");
        if (strcmp(scale, "time") == 0) {
            puf_config_assert(min_v > 0 && max_v > 0,
                "[%s] scale=time requires positive min/max", dict->name);
        } else if (strcmp(scale, "auto") != 0) {
            puf_config_sweep_num(dict, "scale");
        }
    }
}

static inline HypersT puf_config_to_hypers(Config* cfg, int rank, int world_size, int gpu_id) {
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
    puf_config_assert(dot != NULL, "expected section.key, got %s", full_key);

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
    puf_config_assert(strlen(arg) < sizeof(tmp), "argv:%d: argument too long", idx);
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
    puf_config_assert(*s, "argv:%d: empty key", idx);

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
    puf_config_validate(cfg);
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
