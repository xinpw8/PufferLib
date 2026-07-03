#pragma once

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"
#include "checkpoint.h"

static void log_util(PuffeRL* p, Dict* out) {
    nvmlUtilization_t util;
    nvmlDeviceGetUtilizationRates(p->nvml_device, &util);
    dict_set(out, "util/gpu_percent", (double)util.gpu);

    size_t cuda_free;
    size_t cuda_total;
    cudaMemGetInfo(&cuda_free, &cuda_total);
    dict_set(out, "util/vram_used_gb",
        (double)(cuda_total - cuda_free) / (1024.0 * 1024.0 * 1024.0));
    dict_set(out, "util/vram_total_gb",
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
    dict_set(out, "util/cpu_mem_gb", (double)rss_kb / (1024.0 * 1024.0));
}

static void puf_log_env(Dict* out, Dict* env_out) {
    for (int i = 0; i < env_out->size; i++) {
        char key[256];
        snprintf(key, sizeof(key), "env/%s", env_out->items[i].key);
        dict_set(out, key, env_out->items[i].value);
    }
}

static void trainer_log(PuffeRL* p, Dict* out) {
    long global_step = p->global_step;
    double now = wall_clock();
    double dt = now - p->last_log_time;
    long sps = dt > 0 ? (long)((global_step - p->last_log_step) / dt) : 0;
    p->last_log_time = now;
    p->last_log_step = global_step;

    dict_set(out, "SPS", (double)sps * p->hypers.world_size);
    dict_set(out, "agent_steps", (double)global_step * p->hypers.world_size);
    dict_set(out, "uptime", now - p->start_time);
    dict_set(out, "epoch", (double)p->epoch);

    Dict env_out = {0};
    log_environments_impl(*p, &env_out);
    puf_log_env(out, &env_out);

    float losses_host[NUM_LOSSES];
    cudaMemcpy(losses_host, p->losses_puf.data, sizeof(losses_host), cudaMemcpyDeviceToHost);
    float loss_n = losses_host[LOSS_N];
    if (loss_n > 0) {
        float inv_n = 1.0f / loss_n;
        dict_set(out, "loss/policy", losses_host[LOSS_PG] * inv_n);
        dict_set(out, "loss/value", losses_host[LOSS_VF] * inv_n);
        dict_set(out, "loss/entropy", losses_host[LOSS_ENT] * inv_n);
        dict_set(out, "loss/total", losses_host[LOSS_TOTAL] * inv_n);
        dict_set(out, "loss/old_kl", losses_host[LOSS_OLD_APPROX_KL] * inv_n);
        dict_set(out, "loss/kl", losses_host[LOSS_APPROX_KL] * inv_n);
        dict_set(out, "loss/clipfrac", losses_host[LOSS_CLIPFRAC] * inv_n);
    }
    cudaMemset(p->losses_puf.data, 0, numel(p->losses_puf.shape) * sizeof(float));

    log_util(p, out);

    float train_total = 0;
    for (int i = 0; i < NUM_PROF; i++) {
        float sec = p->profile.accum[i] / 1000.0f;
        char key[256];
        snprintf(key, sizeof(key), "perf/%s", PROF_NAMES[i]);
        dict_set(out, key, sec);
        if (i >= PROF_TRAIN_MISC) {
            train_total += sec;
        }
    }
    dict_set(out, "perf/train", train_total);
    memset(p->profile.accum, 0, sizeof(p->profile.accum));
}

static void trainer_eval_log(PuffeRL* p, Dict* out) {
    double now = wall_clock();
    p->last_log_time = now;
    p->last_log_step = p->global_step;
    log_util(p, out);

    Dict env_out = {0};
    vec_log(p->vec, &env_out, 0);
    puf_log_env(out, &env_out);
}

typedef struct {
    Dict* items;
    int size;
    int capacity;
} PufLogHistory;

static DictItem* puf_log_find(Dict* dict, const char* key) {
    for (int i = 0; i < dict->size; i++) {
        if (strcmp(dict->items[i].key, key) == 0) {
            return &dict->items[i];
        }
    }
    return NULL;
}

static void puf_log_update(Dict* dst, Dict* src) {
    for (int i = 0; i < src->size; i++) {
        DictItem* item = &src->items[i];
        if (item->str) {
            dict_set_str(dst, item->key, item->str);
            puf_log_find(dst, item->key)->value = item->value;
        } else {
            dict_set(dst, item->key, item->value);
        }
    }
}

static void puf_log_history_add(PufLogHistory* history, Dict* log) {
    if (history->size == history->capacity) {
        history->capacity = history->capacity ? 2 * history->capacity : 64;
        history->items = (Dict*)realloc(history->items, (size_t)history->capacity * sizeof(Dict));
        if (!history->items) {
            perror("realloc");
            exit(1);
        }
    }

    dict_copy(&history->items[history->size], log);
    history->size++;
}

static void puf_log_history_free(PufLogHistory* history) {
    for (int i = 0; i < history->size; i++) {
        dict_clear(&history->items[i]);
    }
    free(history->items);
    memset(history, 0, sizeof(*history));
}

static void puf_log_collect_keys(PufLogHistory* history, Dict* keys) {
    for (int i = 0; i < history->size; i++) {
        Dict* log = &history->items[i];
        for (int j = 0; j < log->size; j++) {
            if (!puf_log_find(keys, log->items[j].key)) {
                dict_set(keys, log->items[j].key, 0);
            }
        }
    }
}

static double puf_log_reduce(double* vals, int n, double fallback) {
    if (n == 0) {
        return fallback;
    }

    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += vals[i];
    }
    return sum / n;
}

static void puf_log_write_metric(FILE* fp, const char* key, double* values, int n) {
    fprintf(fp, "%s = ", key);
    for (int i = 0; i < n; i++) {
        if (i > 0) {
            fputc(',', fp);
        }
        fprintf(fp, "%.17g", values[i]);
    }
    fputc('\n', fp);
}

static void puf_log_write_config(FILE* fp, Config* cfg) {
    fprintf(fp, "\n[config]\n");
    for (int s = 0; s < cfg->num_sections; s++) {
        Dict* dict = &cfg->sections[s];
        for (int i = 0; i < dict->size; i++) {
            DictItem* item = &dict->items[i];
            if (item->str) {
                fprintf(fp, "%s.%s = %s\n", dict->name, item->key, item->str);
            } else {
                fprintf(fp, "%s.%s = %.17g\n", dict->name, item->key, item->value);
            }
        }
    }
}

static void puf_log_write(const char* path, Config* cfg, PufLogHistory* history) {
    if (history->size == 0) {
        fprintf(stderr, "cannot write empty log history\n");
        exit(1);
    }

    FILE* fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "failed to write log %s\n", path);
        exit(1);
    }

    fprintf(fp, "# PufferLib log v1\n");
    puf_log_write_config(fp, cfg);
    fprintf(fp, "\n[metrics]\n");

    int downsample = (int)puf_config_get(cfg, "sweep", "downsample");
    Dict keys = {0};
    puf_log_collect_keys(history, &keys);
    int points = downsample <= 1 ? 1 : downsample;
    double* out = (double*)calloc((size_t)points, sizeof(double));
    double* bin = (double*)calloc((size_t)history->size, sizeof(double));
    double final_steps = dict_get(&history->items[history->size - 1], "agent_steps");

    for (int k = 0; k < keys.size; k++) {
        const char* key = keys.items[k].key;
        if (strncmp(key, "loss/", 5) == 0) {
            continue;
        }

        double first_value = 0;
        for (int i = 0; i < history->size; i++) {
            DictItem* item = puf_log_find(&history->items[i], key);
            if (item) {
                first_value = item->value;
                break;
            }
        }

        if (points == 1) {
            DictItem* item = puf_log_find(&history->items[history->size - 1], key);
            out[0] = item ? item->value : first_value;
            puf_log_write_metric(fp, key, out, points);
            continue;
        }

        int out_idx = 0;
        int bin_n = 0;
        double fallback = first_value;
        double next_bin = final_steps / (points - 1);
        for (int i = 0; i < history->size; i++) {
            Dict* log = &history->items[i];
            DictItem* item = puf_log_find(log, key);
            if (item) {
                bin[bin_n++] = item->value;
            }

            double steps = dict_get(log, "agent_steps");
            if (steps < next_bin || out_idx >= points - 1) {
                continue;
            }

            double reduced = puf_log_reduce(bin, bin_n, fallback);
            out[out_idx++] = reduced;
            fallback = reduced;
            bin_n = 0;
            next_bin += final_steps / (points - 1);
        }

        DictItem* final_item = puf_log_find(&history->items[history->size - 1], key);
        out[points - 1] = final_item ? final_item->value : puf_log_reduce(bin, bin_n, fallback);
        while (out_idx < points - 1) {
            out[out_idx++] = fallback;
        }
        puf_log_write_metric(fp, key, out, points);
    }

    free(bin);
    free(out);
    fclose(fp);
}
