#pragma once

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"

typedef struct {
    float score;
    float cost;
    float steps;
} TrainResult;

typedef TrainResult (*TrainFn)(Dict* cfg);

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

static float sweep_scale(Dict* cfg, const char* prefix, SpaceType type, float min_v, float max_v) {
    char key[1024];
    snprintf(key, sizeof(key), "%s.scale", prefix);
    const char* raw = puf_config_str(cfg, key);
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

static Hyperparameters* sweep_hypers_create(Dict* cfg,
        SweepParam** params_out, int* num_out) {
    SweepParam* params = (SweepParam*)calloc((size_t)cfg->size, sizeof(SweepParam));
    Space* spaces = (Space*)calloc((size_t)cfg->size, sizeof(Space));
    int n = 0;
    int cost_idx = -1;

    for (int i = 0; i < cfg->size; i++) {
        const char* key = cfg->items[i].key;
        const char* suffix = ".distribution";
        size_t suffix_len = strlen(suffix);
        size_t key_len = strlen(key);
        if (strncmp(key, "sweep.", 6) != 0 || key_len <= suffix_len) {
            continue;
        }
        if (strcmp(key + key_len - suffix_len, suffix) != 0) {
            continue;
        }

        char prefix[512];
        snprintf(prefix, sizeof(prefix), "%.*s", (int)(key_len - suffix_len), key);
        const char* path = prefix + 6;
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
        char min_key[1024];
        char max_key[1024];
        snprintf(min_key, sizeof(min_key), "%s.min", prefix);
        snprintf(max_key, sizeof(max_key), "%s.max", prefix);
        SpaceType type = sweep_space_type(puf_config_str(cfg, key), &is_integer);
        float min_v = (float)puf_config_val(cfg, min_key);
        float max_v = (float)puf_config_val(cfg, max_key);
        float scale = sweep_scale(cfg, prefix, type, min_v, max_v);
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

    int direction = strcmp(puf_config_str(cfg, "sweep.goal"), "minimize") == 0 ? -1 : 1;
    *params_out = params;
    *num_out = n;
    return hyperparameters_create(spaces, n, cost_idx, direction);
}

static void sweep_apply(Dict* cfg, SweepParam* params, int num_params,
        const float* sample) {
    for (int i = 0; i < num_params; i++) {
        float val = space_unnormalize(&params[i].space, sample[i]);
        char buf[64];
        snprintf(buf, sizeof(buf), "%.9g", val);
        char key[256];
        snprintf(key, sizeof(key), "%s.%s", params[i].section, params[i].key);
        puf_config_put(cfg, key, buf);
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

static void validate_sweep_support(Dict* cfg) {
    if (puf_config_val(cfg, "sweep.league") != 0) {
        fprintf(stderr, "sweep error: native league sweeps are not ported yet\n");
        exit(1);
    }

    const char* metric = puf_config_str(cfg, "sweep.metric");
    if (strcmp(metric, "score") != 0) {
        fprintf(stderr, "sweep error: native sweep currently scores env/score, got env/%s\n", metric);
        exit(1);
    }

    int downsample = (int)puf_config_val(cfg, "sweep.downsample");
    if (downsample != 1) {
        fprintf(stderr, "sweep error: native sweep currently observes final metrics only; set [sweep] downsample=1\n");
        exit(1);
    }

    int train_gpus = (int)puf_config_val(cfg, "train.gpus");
    int sweep_gpus = (int)puf_config_val(cfg, "sweep.gpus");
    if (sweep_gpus == 0) {
        sweep_gpus = native_num_gpus();
    }
    if (sweep_gpus != train_gpus) {
        fprintf(stderr,
            "sweep error: native sweep currently runs one trial at a time; set [sweep] gpus equal to [train] gpus\n");
        exit(1);
    }
}

static void run_sweep(Dict* cfg, TrainFn train) {
    validate_sweep_support(cfg);
    SweepParam* params = NULL;
    int num_params = 0;
    Hyperparameters* hypers = sweep_hypers_create(cfg, &params, &num_params);

    int max_runs = (int)puf_config_val(cfg, "sweep.max_runs");
    int downsample = (int)puf_config_val(cfg, "sweep.downsample");
    int prune_pareto = (int)puf_config_val(cfg, "sweep.prune_pareto");
    int use_logit = strcmp(puf_config_str(cfg, "sweep.metric_distribution"), "logit") == 0;
    float max_cost = (float)puf_config_val(cfg, "sweep.max_suggestion_cost");
    float early_stop_quantile = (float)puf_config_val(cfg, "sweep.early_stop_quantile");
    int success_cap = max_runs * downsample * 2;
    if (success_cap < 8192) {
        success_cap = 8192;
    }

    ProteinSweep* protein = protein_sweep_create(hypers,
        10, 256, 50, 0.001f, 50, 750, 4096,
        downsample == 1, prune_pareto, use_logit,
        1.0f, max_cost, 0.1f, -0.8f, early_stop_quantile,
        success_cap, 1024, 5, 73ULL);

    float* sample = (float*)calloc((size_t)num_params, sizeof(float));
    for (int run = 0; run < max_runs; run++) {
        ProteinSweepInfo info = protein_sweep_suggest(protein, sample, NAN);

        Dict trial = {0};
        puf_config_copy(&trial, cfg);
        sweep_apply(&trial, params, num_params, sample);

        char run_id[64];
        snprintf(run_id, sizeof(run_id), "sweep_%ld_%04d", (long)(1000.0 * wall_clock()), run);
        puf_config_put(&trial, "base.run_id", run_id);
        puf_config_validate_train(&trial);

        TrainResult result = train(&trial);
        protein_sweep_observe(protein, sample, result.score, result.cost, 0);
        printf("sweep run=%d score=%.4f cost=%.2f steps=%.0f random=%d gp_obs=%d pareto=%d\n",
            run, result.score, result.cost, result.steps,
            info.is_random, info.n_gp_obs, info.n_pareto);

        puf_config_free(&trial);
    }

    free(sample);
    free(params);
    protein_sweep_destroy(protein);
}
