#pragma once

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "../vendor/cJSON.h"
#include "config.h"

static void mkdir_p(const char* path) {
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char* p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            if (mkdir(tmp, 0777) != 0 && errno != EEXIST) {
                fprintf(stderr, "failed to create directory %s: %s\n", tmp, strerror(errno));
                exit(1);
            }
            *p = '/';
        }
    }
    if (mkdir(tmp, 0777) != 0 && errno != EEXIST) {
        fprintf(stderr, "failed to create directory %s: %s\n", tmp, strerror(errno));
        exit(1);
    }
}

static void dict_set_copy(Dict* dict, const char* key, double val) {
    dict_set(dict, key, val);
}

static void log_util(PuffeRL* p, Dict* out) {
    nvmlUtilization_t util;
    nvmlDeviceGetUtilizationRates(p->nvml_device, &util);
    dict_set_copy(out, "util/gpu_percent", (double)util.gpu);

    size_t cuda_free;
    size_t cuda_total;
    cudaMemGetInfo(&cuda_free, &cuda_total);
    dict_set_copy(out, "util/vram_used_gb",
        (double)(cuda_total - cuda_free) / (1024.0 * 1024.0 * 1024.0));
    dict_set_copy(out, "util/vram_total_gb",
        (double)cuda_total / (1024.0 * 1024.0 * 1024.0));

    long rss_kb = 0;
    FILE* status = fopen("/proc/self/status", "r");
    if (status) {
        char line[256];
        while (fgets(line, sizeof(line), status)) {
            if (sscanf(line, "VmRSS: %ld", &rss_kb) == 1) {
                break;
            }
        }
        fclose(status);
    }
    dict_set_copy(out, "util/cpu_mem_gb", (double)rss_kb / (1024.0 * 1024.0));
}

static void trainer_log(PuffeRL* p, Dict* out) {
    long global_step = p->global_step;
    double now = wall_clock();
    double dt = now - p->last_log_time;
    long sps = dt > 0 ? (long)((global_step - p->last_log_step) / dt) : 0;
    p->last_log_time = now;
    p->last_log_step = global_step;

    dict_set_copy(out, "SPS", (double)sps * p->hypers.world_size);
    dict_set_copy(out, "agent_steps", (double)global_step * p->hypers.world_size);
    dict_set_copy(out, "uptime", now - p->start_time);
    dict_set_copy(out, "epoch", (double)p->epoch);

    Dict* env_out = log_environments_impl(*p);
    for (int i = 0; i < env_out->size; i++) {
        char key[256];
        snprintf(key, sizeof(key), "env/%s", env_out->items[i].key);
        dict_set_copy(out, key, env_out->items[i].value);
    }
    free(env_out->items);
    free(env_out);

    float losses_host[NUM_LOSSES];
    cudaMemcpy(losses_host, p->losses_puf.data, sizeof(losses_host), cudaMemcpyDeviceToHost);
    float loss_n = losses_host[LOSS_N];
    if (loss_n > 0) {
        float inv_n = 1.0f / loss_n;
        dict_set_copy(out, "loss/policy", losses_host[LOSS_PG] * inv_n);
        dict_set_copy(out, "loss/value", losses_host[LOSS_VF] * inv_n);
        dict_set_copy(out, "loss/entropy", losses_host[LOSS_ENT] * inv_n);
        dict_set_copy(out, "loss/total", losses_host[LOSS_TOTAL] * inv_n);
        dict_set_copy(out, "loss/old_kl", losses_host[LOSS_OLD_APPROX_KL] * inv_n);
        dict_set_copy(out, "loss/kl", losses_host[LOSS_APPROX_KL] * inv_n);
        dict_set_copy(out, "loss/clipfrac", losses_host[LOSS_CLIPFRAC] * inv_n);
    }
    cudaMemset(p->losses_puf.data, 0, numel(p->losses_puf.shape) * sizeof(float));

    log_util(p, out);

    float train_total = 0;
    for (int i = 0; i < NUM_PROF; i++) {
        float sec = p->profile.accum[i] / 1000.0f;
        char key[256];
        snprintf(key, sizeof(key), "perf/%s", PROF_NAMES[i]);
        dict_set_copy(out, key, sec);
        if (i >= PROF_TRAIN_MISC) {
            train_total += sec;
        }
    }
    dict_set_copy(out, "perf/train", train_total);
    memset(p->profile.accum, 0, sizeof(p->profile.accum));
}

static void trainer_eval_log(PuffeRL* p, Dict* out) {
    double now = wall_clock();
    p->last_log_time = now;
    p->last_log_step = p->global_step;
    log_util(p, out);

    Dict* env_out = create_dict(64);
    static_vec_eval_log(p->vec, env_out);
    for (int i = 0; i < env_out->size; i++) {
        char key[256];
        snprintf(key, sizeof(key), "env/%s", env_out->items[i].key);
        dict_set_copy(out, key, env_out->items[i].value);
    }
    free(env_out->items);
    free(env_out);
}

static void save_weights(PuffeRL* p, const char* path) {
    int64_t nbytes = numel(p->master_weights.shape) * sizeof(float);
    char* buf = (char*)malloc(nbytes);
    cudaMemcpy(buf, p->master_weights.data, nbytes, cudaMemcpyDeviceToHost);
    FILE* fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "failed to open %s for writing\n", path);
        free(buf);
        exit(1);
    }
    if (fwrite(buf, 1, nbytes, fp) != (size_t)nbytes) {
        fprintf(stderr, "failed to write weights to %s\n", path);
        fclose(fp);
        free(buf);
        exit(1);
    }
    fclose(fp);
    free(buf);
}

static void load_weights(PuffeRL* p, const char* path) {
    int64_t nbytes = numel(p->master_weights.shape) * sizeof(float);
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "failed to open %s for reading\n", path);
        exit(1);
    }
    char* buf = (char*)malloc(nbytes);
    size_t nread = fread(buf, 1, nbytes, fp);
    fclose(fp);
    if ((int64_t)nread != nbytes) {
        fprintf(stderr, "failed to read weights from %s\n", path);
        free(buf);
        exit(1);
    }
    cudaMemcpy(p->master_weights.data, buf, nbytes, cudaMemcpyHostToDevice);
    free(buf);
    if (USE_BF16) {
        int n = numel(p->param_puf.shape);
        cast<<<grid_size(n), BLOCK_SIZE, 0, p->default_stream>>>(
            p->param_puf.data, p->master_weights.data, n);
    }
}

static int has_suffix(const char* s, const char* suffix) {
    size_t n = strlen(s);
    size_t m = strlen(suffix);
    return n >= m && strcmp(s + n - m, suffix) == 0;
}

static void find_latest_checkpoint(const char* dir, char* out, size_t out_size, time_t* best_time) {
    DIR* dp = opendir(dir);
    if (!dp) {
        return;
    }

    struct dirent* ent = NULL;
    while ((ent = readdir(dp))) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
            continue;
        }

        char path[4096];
        snprintf(path, sizeof(path), "%s/%s", dir, ent->d_name);

        struct stat st;
        if (stat(path, &st) != 0) {
            continue;
        }

        if (S_ISDIR(st.st_mode)) {
            find_latest_checkpoint(path, out, out_size, best_time);
        } else if (S_ISREG(st.st_mode) && has_suffix(path, ".bin") && st.st_ctime >= *best_time) {
            *best_time = st.st_ctime;
            snprintf(out, out_size, "%s", path);
        }
    }

    closedir(dp);
}

static const char* resolve_load_model_path(PufConfigFile* cfg, char* out, size_t out_size) {
    PufConfig* base = puf_config_get_section(cfg, "base");
    const char* load_path = puf_config_get(base, "load_model_path");
    if (!load_path || strcmp(load_path, "None") == 0) {
        return NULL;
    }

    if (strcmp(load_path, "latest") != 0) {
        return load_path;
    }

    char root[2048];
    snprintf(root, sizeof(root), "%s/%s",
        puf_config_str(base, "checkpoint_dir"), puf_config_str(base, "env_name"));

    out[0] = 0;
    time_t best_time = 0;
    find_latest_checkpoint(root, out, out_size, &best_time);
    if (!out[0]) {
        fprintf(stderr, "no .bin checkpoints found in %s\n", root);
        exit(1);
    }
    return out;
}

static cJSON* puf_log_json_value(const char* raw) {
    if (puf_config_streq_ci(raw, "none")) {
        return cJSON_CreateNull();
    }
    if (puf_config_streq_ci(raw, "true")) {
        return cJSON_CreateTrue();
    }
    if (puf_config_streq_ci(raw, "false")) {
        return cJSON_CreateFalse();
    }

    char buf[256];
    size_t j = 0;
    int has_float = 0;
    for (size_t i = 0; raw[i] && j + 1 < sizeof(buf); i++) {
        if (raw[i] == '_' || isspace((unsigned char)raw[i])) {
            continue;
        }
        if (raw[i] == '.' || raw[i] == 'e' || raw[i] == 'E') {
            has_float = 1;
        }
        buf[j++] = raw[i];
    }
    buf[j] = 0;

    if (buf[0]) {
        char* end = 0;
        if (has_float) {
            double v = strtod(buf, &end);
            if (end && !*end) {
                return cJSON_CreateNumber(v);
            }
        } else {
            long long v = strtoll(buf, &end, 10);
            if (end && !*end) {
                return cJSON_CreateNumber((double)v);
            }
        }
    }

    return cJSON_CreateString(raw);
}

static cJSON* puf_log_object_child(cJSON* obj, const char* key) {
    cJSON* child = cJSON_GetObjectItemCaseSensitive(obj, key);
    if (child) {
        return child;
    }

    child = cJSON_CreateObject();
    cJSON_AddItemToObject(obj, key, child);
    return child;
}

static void puf_log_add_section(cJSON* root, PufConfig* cfg) {
    cJSON* section = root;
    char name[256];
    snprintf(name, sizeof(name), "%s", cfg->name);

    if (strcmp(cfg->name, "base") != 0) {
        char* start = name;
        for (;;) {
            char* dot = strchr(start, '.');
            if (dot) {
                *dot = 0;
            }

            section = puf_log_object_child(section, start);
            if (!dot) {
                break;
            }
            start = dot + 1;
        }
    }

    for (int i = 0; i < cfg->len; i++) {
        cJSON_AddItemToObject(section, cfg->items[i].key, puf_log_json_value(cfg->items[i].val));
    }
}

static void puf_log_add_metrics(cJSON* root, Dict* log) {
    cJSON* metrics = cJSON_CreateObject();
    cJSON_AddItemToObject(root, "metrics", metrics);

    for (int i = 0; i < log->size; i++) {
        cJSON* values = cJSON_CreateArray();
        cJSON_AddItemToArray(values, cJSON_CreateNumber(log->items[i].value));
        cJSON_AddItemToObject(metrics, log->items[i].key, values);
    }
}

static void puf_log_write_json(const char* path, PufConfigFile* cfg, Dict* log) {
    cJSON* root = cJSON_CreateObject();
    for (int i = 0; i < cfg->len; i++) {
        puf_log_add_section(root, &cfg->sections[i]);
    }
    puf_log_add_metrics(root, log);

    char* json = cJSON_Print(root);
    FILE* fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "failed to write log %s\n", path);
        exit(1);
    }
    fputs(json, fp);
    fputc('\n', fp);
    fclose(fp);

    cJSON_free(json);
    cJSON_Delete(root);
}
