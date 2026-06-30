#pragma once

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"

typedef struct {
    char section[64];
    char key[64];
    char path[128];
    Space space;
} SweepParam;

static SpaceType sweep_space_type(const char* dist, int* is_integer) {
    *is_integer = 0;
    if (strcmp(dist, "uniform") == 0) {
        return SPACE_LINEAR;
    }
    if (strcmp(dist, "int_uniform") == 0) {
        *is_integer = 1;
        return SPACE_LINEAR;
    }
    if (strcmp(dist, "uniform_pow2") == 0) {
        *is_integer = 1;
        return SPACE_POW2;
    }
    if (strcmp(dist, "log_normal") == 0) {
        return SPACE_LOG;
    }
    if (strcmp(dist, "logit_normal") == 0) {
        return SPACE_LOGIT;
    }

    fprintf(stderr, "sweep error: invalid distribution %s\n", dist);
    exit(1);
}

static float sweep_scale(PufConfig* section, SpaceType type, float min_v, float max_v) {
    const char* raw = puf_config_str(section, "scale");
    if (strcmp(raw, "auto") == 0) {
        return 0.5f;
    }
    if (strcmp(raw, "time") == 0) {
        return 1.0f / (log2f(max_v) - log2f(min_v));
    }

    double val = 0;
    if (!puf_config_parse_val(raw, &val)) {
        fprintf(stderr, "sweep error: invalid scale %s\n", raw);
        exit(1);
    }
    return (float)val;
}

static Hyperparameters* sweep_hypers_create(PufConfigFile* cfg,
        SweepParam** params_out, int* num_out) {
    SweepParam* params = (SweepParam*)calloc((size_t)cfg->len, sizeof(SweepParam));
    Space* spaces = (Space*)calloc((size_t)cfg->len, sizeof(Space));
    int n = 0;
    int cost_idx = -1;

    for (int i = 0; i < cfg->len; i++) {
        PufConfig* section = &cfg->sections[i];
        if (strncmp(section->name, "sweep.", 6) != 0) {
            continue;
        }
        if (!puf_config_get(section, "distribution")) {
            continue;
        }

        const char* path = section->name + 6;
        const char* dot = strrchr(path, '.');
        if (!dot) {
            fprintf(stderr, "sweep error: expected section sweep.<section>.<key>\n");
            exit(1);
        }

        int section_len = (int)(dot - path);
        snprintf(params[n].section, sizeof(params[n].section), "%.*s", section_len, path);
        snprintf(params[n].key, sizeof(params[n].key), "%s", dot + 1);
        snprintf(params[n].path, sizeof(params[n].path), "%s/%s",
            params[n].section, params[n].key);

        int is_integer = 0;
        SpaceType type = sweep_space_type(puf_config_str(section, "distribution"), &is_integer);
        float min_v = (float)puf_config_val(section, "min");
        float max_v = (float)puf_config_val(section, "max");
        float scale = sweep_scale(section, type, min_v, max_v);
        space_init(&params[n].space, type, min_v, max_v, scale, is_integer);
        spaces[n] = params[n].space;

        if (strcmp(params[n].path, "train/total_timesteps") == 0) {
            cost_idx = n;
        }
        n++;
    }

    if (n == 0) {
        fprintf(stderr, "sweep error: no sweep parameter sections found\n");
        exit(1);
    }

    PufConfig* sweep = puf_config_get_section(cfg, "sweep");
    int direction = strcmp(puf_config_str(sweep, "goal"), "minimize") == 0 ? -1 : 1;
    *params_out = params;
    *num_out = n;
    return hyperparameters_create(spaces, n, cost_idx, direction);
}

static void sweep_apply(PufConfigFile* cfg, SweepParam* params, int num_params,
        const float* sample) {
    for (int i = 0; i < num_params; i++) {
        float val = space_unnormalize(&params[i].space, sample[i]);
        char buf[64];
        snprintf(buf, sizeof(buf), "%.9g", val);
        PufConfig* section = puf_config_get_section(cfg, params[i].section);
        puf_config_put(section, params[i].key, buf);
    }
}

static int native_num_gpus(void) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count < 1) {
        fprintf(stderr, "sweep error: no CUDA devices available\n");
        exit(1);
    }
    return count;
}

static void validate_sweep_support(PufConfigFile* cfg) {
    PufConfig* sweep = puf_config_get_section(cfg, "sweep");
    PufConfig* train = puf_config_get_section(cfg, "train");

    if (puf_config_val(sweep, "league") != 0) {
        fprintf(stderr, "sweep error: native league sweeps are not ported yet\n");
        exit(1);
    }

    const char* metric = puf_config_str(sweep, "metric");
    if (strcmp(metric, "score") != 0) {
        fprintf(stderr, "sweep error: native sweep currently scores env/score, got env/%s\n", metric);
        exit(1);
    }

    int downsample = (int)puf_config_val(sweep, "downsample");
    if (downsample != 1) {
        fprintf(stderr, "sweep error: native sweep currently observes final metrics only; set [sweep] downsample=1\n");
        exit(1);
    }

    int train_gpus = (int)puf_config_val(train, "gpus");
    int sweep_gpus = (int)puf_config_val(sweep, "gpus");
    if (sweep_gpus == 0) {
        sweep_gpus = native_num_gpus();
    }
    if (sweep_gpus != train_gpus) {
        fprintf(stderr,
            "sweep error: native sweep currently runs one trial at a time; set [sweep] gpus equal to [train] gpus\n");
        exit(1);
    }
}
