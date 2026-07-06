#pragma once

#include <math.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include "checkpoint.h"

#define TRAIN_RESULT_MAX_POINTS 64

typedef struct {
    float score;
    float cost;
    float steps;
    int points;
    char checkpoint_path[4096];
    float scores[TRAIN_RESULT_MAX_POINTS];
    float costs[TRAIN_RESULT_MAX_POINTS];
    float step_points[TRAIN_RESULT_MAX_POINTS];
} TrainResult;

extern char** environ;

typedef struct {
    int run;
    int random;
    int gp_obs;
    int pareto;
    int fd;
    pid_t pid;
    char run_id[128];
    float* sample;
    TrainResult result;
} SweepJob;

typedef struct {
    char section[64];
    char key[64];
    char path[128];
    Space space;
} SweepRuntimeParam;

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

static float sweep_num(Dict* dict, const char* key) {
    const char* raw = dict_get_str(dict, key);
    double value = 0;
    if (!puf_ini_parse_val(raw, &value)) {
        fprintf(stderr, "sweep error: invalid numeric field [%s] %s = %s\n",
            dict->name, key, raw);
        exit(1);
    }
    return (float)value;
}

static float sweep_scale(Dict* dict, float min_v, float max_v) {
    const char* raw = dict_get_str(dict, "scale");
    if (strcmp(raw, "auto") == 0) {
        return 0.5f;
    }
    if (strcmp(raw, "time") == 0) {
        return 1.0f / (log2f(max_v) - log2f(min_v));
    }
    return sweep_num(dict, "scale");
}

static Hyperparameters* sweep_hypers_create(Config* cfg,
        SweepRuntimeParam** params_out, int* num_out) {
    SweepRuntimeParam* params = (SweepRuntimeParam*)calloc((size_t)cfg->ini.num_sections,
        sizeof(SweepRuntimeParam));
    Space* spaces = (Space*)calloc((size_t)cfg->ini.num_sections, sizeof(Space));
    int n = 0;
    int cost_idx = -1;

    for (int i = 0; i < cfg->ini.num_sections; i++) {
        Dict* dict = &cfg->ini.sections[i];
        if (strncmp(dict->name, "sweep.", 6) != 0) {
            continue;
        }

        const char* sweep_key = dict->name + 6;
        const char* dot = strrchr(sweep_key, '.');
        if (!dot) {
            fprintf(stderr, "sweep error: expected section sweep.<section>.<key>\n");
            exit(1);
        }

        int section_len = (int)(dot - sweep_key);
        snprintf(params[n].section, sizeof(params[n].section), "%.*s", section_len, sweep_key);
        snprintf(params[n].key, sizeof(params[n].key), "%s", dot + 1);
        snprintf(params[n].path, sizeof(params[n].path), "%s/%s",
            params[n].section, params[n].key);

        int is_integer = 0;
        SpaceType type = sweep_space_type(dict_get_str(dict, "distribution"), &is_integer);
        float min_v = sweep_num(dict, "min");
        float max_v = sweep_num(dict, "max");
        float scale = sweep_scale(dict, min_v, max_v);
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

    int direction = strcmp(puf_config_str(cfg, "sweep", "goal"), "minimize") == 0 ? -1 : 1;
    *params_out = params;
    *num_out = n;
    return hyperparameters_create(spaces, n, cost_idx, direction);
}

static void sweep_apply(Config* cfg, SweepRuntimeParam* params, int num_params,
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

static void validate_sweep_support(Config* cfg) {
    int league = (int)puf_config_get(cfg, "sweep", "league");
    const char* metric = puf_config_str(cfg, "sweep", "metric");
    if (!league && strcmp(metric, "score") != 0) {
        fprintf(stderr, "sweep error: native sweep currently scores env/score, got env/%s\n", metric);
        exit(1);
    }

    int train_gpus = (int)puf_config_get(cfg, "train", "gpus");
    if (train_gpus < 1) {
        fprintf(stderr, "sweep error: train.gpus must be >= 1\n");
        exit(1);
    }

    int total_gpus = native_num_gpus();
    int sweep_gpus = (int)puf_config_get(cfg, "sweep", "gpus");
    if (sweep_gpus == 0) {
        sweep_gpus = total_gpus;
    }
    int needed_gpus = league ? train_gpus + 1 : train_gpus;
    if (sweep_gpus < needed_gpus) {
        fprintf(stderr, "sweep error: sweep.gpus must be >= train.gpus\n");
        exit(1);
    }
    if (sweep_gpus > total_gpus) {
        fprintf(stderr, "sweep error: sweep.gpus=%d but only %d CUDA devices are visible\n",
            sweep_gpus, total_gpus);
        exit(1);
    }
}

static int sweep_read_result(int fd, TrainResult* out) {
    char* dst = (char*)out;
    size_t need = sizeof(*out);
    while (need > 0) {
        ssize_t n = read(fd, dst, need);
        if (n < 0) {
            return 0;
        }
        if (n == 0) {
            return 0;
        }
        dst += n;
        need -= (size_t)n;
    }
    return 1;
}

static char* sweep_arg_kv(const char* full_key, DictItem* item) {
    char val[128];
    const char* src = item->str;
    if (!src) {
        snprintf(val, sizeof(val), "%.17g", item->value);
        src = val;
    }

    size_t n = strlen(full_key) + strlen(src) + 2;
    char* out = (char*)malloc(n);
    if (!out) {
        perror("malloc");
        exit(1);
    }
    snprintf(out, n, "%s=%s", full_key, src);
    return out;
}

static void sweep_free_argv(char** argv, int argc) {
    for (int i = 3; i < argc; i++) {
        free(argv[i]);
    }
    free(argv);
}

static int sweep_config_count(Config* cfg) {
    int count = 0;
    for (int s = 0; s < cfg->ini.num_sections; s++) {
        count += cfg->ini.sections[s].size;
    }
    return count;
}

static int sweep_fill_args(Config* cfg, char** argv, int idx) {
    char full_key[PUF_DICT_MAX_KEY * 2];
    for (int s = 0; s < cfg->ini.num_sections; s++) {
        Dict* dict = &cfg->ini.sections[s];
        for (int i = 0; i < dict->size; i++) {
            snprintf(full_key, sizeof(full_key), "%s.%s", dict->name, dict->items[i].key);
            argv[idx++] = sweep_arg_kv(full_key, &dict->items[i]);
        }
    }
    return idx;
}

static SweepJob sweep_start_job(Config* cfg, const char* exe_path,
        SweepRuntimeParam* params, int num_params, const float* sample,
        ProteinSweepInfo info, int run, int gpu_offset, int league) {
    SweepJob job = {0};
    job.run = run;
    job.random = info.is_random;
    job.gp_obs = info.n_gp_obs;
    job.pareto = info.n_pareto;
    job.sample = (float*)calloc((size_t)num_params, sizeof(float));
    memcpy(job.sample, sample, (size_t)num_params * sizeof(float));

    int pipefd[2];
    if (pipe(pipefd) != 0) {
        perror("pipe");
        exit(1);
    }

    Config trial = {0};
    puf_config_copy(&trial, cfg);
    sweep_apply(&trial, params, num_params, sample);
    char offset[32];
    if (league) {
        puf_config_put(&trial, "selfplay.enabled", "1");
        puf_config_put(&trial, "env.num_agents", "2");
        puf_config_put(&trial, "env.num_bots", "0");
        puf_config_put(&trial, "vec.num_frozen_banks", "1");
        puf_config_put(&trial, "vec.frozen_bank_pct", "0.1");
        snprintf(offset, sizeof(offset), "%d",
            (int)puf_config_get(&trial, "policy", "hidden_size"));
        puf_config_put(&trial, "vec.frozen_bank_hidden_size", offset);
        snprintf(offset, sizeof(offset), "%d",
            (int)puf_config_get(&trial, "policy", "num_layers"));
        puf_config_put(&trial, "vec.frozen_bank_num_layers", offset);
    }

    char run_id[64];
    snprintf(run_id, sizeof(run_id), "sweep_%ld_%04d",
        (long)(1000.0 * wall_clock()), run);
    snprintf(job.run_id, sizeof(job.run_id), "%s", run_id);
    puf_config_put(&trial, "base.run_id", run_id);

    snprintf(offset, sizeof(offset), "%d", gpu_offset);
    puf_config_put(&trial, "base.gpu_offset", offset);

    char result_fd[32];
    snprintf(result_fd, sizeof(result_fd), "%d", pipefd[1]);
    puf_config_put(&trial, "base.result_fd", result_fd);
    puf_config_validate_train(&trial);

    int argc = sweep_config_count(&trial) + 4;
    char** argv = (char**)calloc((size_t)argc, sizeof(char*));
    if (!argv) {
        perror("calloc");
        exit(1);
    }
    argv[0] = (char*)exe_path;
    argv[1] = (char*)"train";
    argv[2] = (char*)puf_config_str(&trial, "base", "env_name");
    sweep_fill_args(&trial, argv, 3);
    argv[argc - 1] = NULL;

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addclose(&actions, pipefd[0]);
    posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO,
        "/dev/null", O_WRONLY, 0);
    int err = posix_spawnp(&job.pid, exe_path, &actions, NULL, argv, environ);
    posix_spawn_file_actions_destroy(&actions);
    sweep_free_argv(argv, argc - 1);
    puf_config_free(&trial);
    if (err != 0) {
        fprintf(stderr, "posix_spawn failed: %s\n", strerror(err));
        exit(1);
    }

    close(pipefd[1]);
    job.fd = pipefd[0];
    return job;
}

static void sweep_wait_job(ProteinSweep* protein, SweepJob* job,
        int league, const char* league_state_path) {
    int ok = sweep_read_result(job->fd, &job->result);
    close(job->fd);

    int status = 0;
    if (waitpid(job->pid, &status, 0) < 0) {
        perror("waitpid");
        exit(1);
    }
    if (!ok || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "sweep worker run=%d failed\n", job->run);
        exit(1);
    }

    if (league) {
        if (!job->result.checkpoint_path[0]) {
            fprintf(stderr, "league trial run=%d did not produce a checkpoint\n", job->run);
            exit(1);
        }
        league_register_player(league_state_path, job->run_id,
            job->result.checkpoint_path, job->result.cost);
        float elo = league_player_elo(league_state_path, job->run_id);
        protein_sweep_observe(protein, job->sample, elo, job->result.cost, 0);
        job->result.score = elo;
    } else {
        int points = job->result.points > 0 ? job->result.points : 1;
        for (int i = 0; i < points; i++) {
            protein_sweep_observe(protein, job->sample,
                job->result.scores[i], job->result.costs[i], 0);
        }
    }
    printf("sweep run=%d score=%.4f cost=%.2f steps=%.0f random=%d gp_obs=%d pareto=%d\n",
        job->run, job->result.score, job->result.cost, job->result.steps,
        job->random, job->gp_obs, job->pareto);
    free(job->sample);
}

static void sweep_state_path(Config* cfg, char* out, size_t out_size) {
    const char* configured = puf_config_str(cfg, "sweep", "league_state_path");
    if (configured && configured[0]) {
        snprintf(out, out_size, "%s", configured);
        return;
    }

    char dir[2048];
    snprintf(dir, sizeof(dir), "%s/%s",
        puf_config_str(cfg, "base", "log_dir"),
        puf_config_str(cfg, "base", "env_name"));
    mkdir_p(dir);
    snprintf(out, out_size, "%s/%ld_league.txt",
        dir, (long)(1000.0 * wall_clock()));
    puf_config_put(cfg, "sweep.league_state_path", out);
}

static pid_t sweep_start_match_worker(Config* cfg, const char* exe_path,
        const char* state_path, int gpu_id) {
    Config worker = {0};
    puf_config_copy(&worker, cfg);
    puf_config_put(&worker, "sweep.league_state_path", state_path);
    puf_config_put(&worker, "selfplay.enabled", "0");

    char offset[32];
    snprintf(offset, sizeof(offset), "%d", gpu_id);
    puf_config_put(&worker, "base.gpu_offset", offset);

    int argc = sweep_config_count(&worker) + 4;
    char** argv = (char**)calloc((size_t)argc, sizeof(char*));
    argv[0] = (char*)exe_path;
    argv[1] = (char*)"league_match_worker";
    argv[2] = (char*)puf_config_str(&worker, "base", "env_name");
    sweep_fill_args(&worker, argv, 3);
    argv[argc - 1] = NULL;

    pid_t pid = 0;
    int err = posix_spawnp(&pid, exe_path, NULL, NULL, argv, environ);
    sweep_free_argv(argv, argc - 1);
    puf_config_free(&worker);
    if (err != 0) {
        fprintf(stderr, "posix_spawn match worker failed: %s\n", strerror(err));
        exit(1);
    }
    return pid;
}

void run_sweep(Config* cfg, const char* exe_path) {
    validate_sweep_support(cfg);
    int league = (int)puf_config_get(cfg, "sweep", "league");
    SweepRuntimeParam* params = NULL;
    int num_params = 0;
    Hyperparameters* hypers = sweep_hypers_create(cfg, &params, &num_params);

    int max_runs = (int)puf_config_get(cfg, "sweep", "max_runs");
    int downsample = (int)puf_config_get(cfg, "sweep", "downsample");
    int prune_pareto = (int)puf_config_get(cfg, "sweep", "prune_pareto");
    int use_logit = strcmp(puf_config_str(cfg, "sweep", "metric_distribution"), "logit") == 0;
    float max_cost = (float)puf_config_get(cfg, "sweep", "max_suggestion_cost");
    float early_stop_quantile = (float)puf_config_get(cfg, "sweep", "early_stop_quantile");
    int success_cap = max_runs * downsample * 2;
    if (success_cap < 8192) {
        success_cap = 8192;
    }

    int total_gpus = native_num_gpus();
    int sweep_gpus = (int)puf_config_get(cfg, "sweep", "gpus");
    int train_gpus = (int)puf_config_get(cfg, "train", "gpus");
    if (sweep_gpus == 0) {
        sweep_gpus = total_gpus;
    }
    int use_gpu = (int)puf_config_get(cfg, "sweep", "use_gpu");
    if (use_gpu) {
        cudaSetDevice(sweep_gpus - 1);
    }
    char league_state_path[LEAGUE_PATH_MAX] = {0};
    pid_t match_pid = 0;
    int train_gpu_count = sweep_gpus;
    if (league) {
        if (strcmp(puf_config_str(cfg, "base", "env_name"), "robocode") != 0) {
            fprintf(stderr, "league sweep currently requires robocode\n");
            exit(1);
        }
        train_gpu_count = sweep_gpus - 1;
        sweep_state_path(cfg, league_state_path, sizeof(league_state_path));
        match_pid = sweep_start_match_worker(cfg, exe_path,
            league_state_path, sweep_gpus - 1);
    }

    int parallel = train_gpu_count / train_gpus;
    if (parallel < 1) {
        parallel = 1;
    }

    ProteinSweep* protein = protein_sweep_create(hypers,
        10, 256, 50, 0.001f, 50, 750, 4096,
        downsample == 1, prune_pareto, use_logit,
        1.0f, max_cost, 0.1f, -0.8f, early_stop_quantile,
        success_cap, 1024, 5, 73ULL);

    float* sample = (float*)calloc((size_t)num_params, sizeof(float));
    SweepJob* jobs = (SweepJob*)calloc((size_t)parallel, sizeof(SweepJob));
    for (int run = 0; run < max_runs;) {
        int batch = max_runs - run;
        if (batch > parallel) {
            batch = parallel;
        }

        for (int i = 0; i < batch; i++) {
            ProteinSweepInfo info = protein_sweep_suggest(protein, sample, NAN);
            jobs[i] = sweep_start_job(cfg, exe_path, params, num_params, sample,
                info, run + i, i * train_gpus, league);
        }
        for (int i = 0; i < batch; i++) {
            sweep_wait_job(protein, &jobs[i], league, league_state_path);
        }
        run += batch;
    }

    if (match_pid > 0) {
        kill(match_pid, SIGTERM);
        waitpid(match_pid, NULL, 0);
    }
    free(jobs);
    free(sample);
    free(params);
    protein_sweep_destroy(protein);
}
