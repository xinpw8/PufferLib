#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "config.h"
#include "table.h"

static const char* EXTRA_KEYS[] = {
    "train/learning_rate",
    "train/ent_coef",
    "train/gamma",
    "train/gae_lambda",
    "train/vtrace_rho_clip",
    "train/vtrace_c_clip",
    "train/clip_coef",
    "train/vf_clip_coef",
    "train/vf_coef",
    "train/max_grad_norm",
    "train/beta1",
    "train/beta2",
    "train/eps",
    "train/prio_alpha",
    "train/prio_beta0",
    "train/horizon",
    "train/replay_ratio",
    "train/minibatch_size",
    "policy/hidden_size",
    "policy/num_layers",
    "vec/total_agents",
    "train/total_timesteps",
};

static int has_suffix(const char* s, const char* suffix) {
    size_t n = strlen(s);
    size_t m = strlen(suffix);
    return n >= m && strcmp(s + n - m, suffix) == 0;
}

static int is_dir(const char* path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static void key_to_cache(char* out, size_t out_size, const char* section, const char* key) {
    if (strcmp(section, "config") == 0) {
        if (strncmp(key, "base.", 5) == 0) {
            snprintf(out, out_size, "%s", key + 5);
        } else {
            snprintf(out, out_size, "%s", key);
        }
    } else if (strcmp(section, "base") == 0) {
        snprintf(out, out_size, "%s", key);
    } else {
        snprintf(out, out_size, "%s/%s", section, key);
    }

    for (char* p = out; *p; p++) {
        if (*p == '.') {
            *p = '/';
        }
    }
}

static int parse_list(char* raw, float** out, int* len) {
    int cap = 16;
    int n = 0;
    float* vals = (float*)calloc((size_t)cap, sizeof(float));
    if (!vals) {
        perror("calloc");
        exit(1);
    }

    char* p = raw;
    while (*p) {
        while (*p == ',' || isspace((unsigned char)*p)) {
            p++;
        }
        if (!*p) {
            break;
        }

        char* end = NULL;
        float v = strtof(p, &end);
        if (end == p) {
            free(vals);
            return 0;
        }
        if (n == cap) {
            cap *= 2;
            vals = (float*)realloc(vals, (size_t)cap * sizeof(float));
            if (!vals) {
                perror("realloc");
                exit(1);
            }
        }
        vals[n++] = v;
        p = end;
    }

    *out = vals;
    *len = n;
    return n > 0;
}

static int load_ini_log(const char* path, Dict* scalars, Table* metrics) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        return 0;
    }

    char section[256] = "base";
    char line[8192];
    while (fgets(line, sizeof(line), fp)) {
        puf_config_strip_comment(line);
        char* s = puf_config_trim(line);
        if (!*s) {
            continue;
        }

        size_t len = strlen(s);
        if (s[0] == '[' && len > 2 && s[len - 1] == ']') {
            s[len - 1] = 0;
            snprintf(section, sizeof(section), "%s", puf_config_trim(s + 1));
            continue;
        }

        char* eq = strchr(s, '=');
        if (!eq) {
            fclose(fp);
            return 0;
        }
        *eq = 0;
        char* key = puf_config_trim(s);
        char* val = puf_config_trim(eq + 1);
        puf_config_strip_quotes(val);

        if (strcmp(section, "metrics") == 0) {
            if (strstr(key, "loss")) {
                continue;
            }
            float* values = NULL;
            int n = 0;
            if (!parse_list(val, &values, &n)) {
                fclose(fp);
                return 0;
            }
            if (metrics->rows == 0) {
                table_resize_rows(metrics, n);
            } else if (metrics->rows != n) {
                free(values);
                fclose(fp);
                return 0;
            }
            int col = table_ensure_col(metrics, key);
            for (int r = 0; r < n; r++) {
                table_set(metrics, r, col, values[r]);
            }
            free(values);
        } else {
            double value = 0;
            if (!puf_config_parse_val(val, &value)) {
                continue;
            }
            char full[512];
            key_to_cache(full, sizeof(full), section, key);
            dict_set(scalars, full, value);
        }
    }

    fclose(fp);
    return metrics->rows > 0;
}

static void table_copy_rows(Table* dst, int dst_row, Table* src) {
    for (int c = 0; c < src->cols; c++) {
        int out_col = table_ensure_col(dst, src->labels[c]);
        for (int r = 0; r < src->rows; r++) {
            table_set(dst, dst_row + r, out_col, table_get(src, r, c));
        }
    }
}

static void table_fill_scalar(Table* dst, int row, int rows, const char* key, float value) {
    int col = table_ensure_col(dst, key);
    for (int r = 0; r < rows; r++) {
        table_set(dst, row + r, col, value);
    }
}

static void load_env(const char* env, int full_dataset, Table* out) {
    char dir[1024];
    snprintf(dir, sizeof(dir), "logs/%s", env);
    struct dirent** ents = NULL;
    int nents = scandir(dir, &ents, NULL, alphasort);
    if (nents < 0) {
        return;
    }

    for (int i = 0; i < nents; i++) {
        if (ents[i]->d_name[0] == '.') {
            free(ents[i]);
            continue;
        }

        char path[2048];
        snprintf(path, sizeof(path), "%s/%s", dir, ents[i]->d_name);
        free(ents[i]);
        if (is_dir(path) || (!has_suffix(path, ".ini") && !has_suffix(path, ".log"))) {
            continue;
        }

        Dict scalars = {0};
        Table metrics = {0};
        if (!load_ini_log(path, &scalars, &metrics)) {
            dict_clear(&scalars);
            table_free(&metrics);
            continue;
        }

        int start = out->rows;
        table_resize_rows(out, out->rows + metrics.rows);
        table_copy_rows(out, start, &metrics);
        for (int s = 0; s < scalars.size; s++) {
            DictItem* item = &scalars.items[s];
            table_fill_scalar(out, start, metrics.rows, item->key, (float)item->value);
        }
        for (int k = 0; k < (int)(sizeof(EXTRA_KEYS) / sizeof(EXTRA_KEYS[0])); k++) {
            if (!dict_find(&scalars, EXTRA_KEYS[k])) {
                table_fill_scalar(out, start, metrics.rows, EXTRA_KEYS[k], 0);
            }
        }

        dict_clear(&scalars);
        table_free(&metrics);
    }
    free(ents);

    int steps_col = table_col(out, "agent_steps");
    int total_steps_col = table_col(out, "train/total_timesteps");
    for (int r = 0; r < out->rows; r++) {
        if (steps_col >= 0) {
            table_set(out, r, steps_col, table_get(out, r, steps_col) / 1e6f);
        }
        if (total_steps_col >= 0) {
            table_set(out, r, total_steps_col, table_get(out, r, total_steps_col) / 1e6f);
        }
    }

    int tsne1 = table_ensure_col(out, "tsne1");
    int tsne2 = table_ensure_col(out, "tsne2");
    for (int r = 0; r < out->rows; r++) {
        table_set(out, r, tsne1, (float)(r % 997) / 997.0f);
        table_set(out, r, tsne2, (float)((r * 37) % 991) / 991.0f);
    }

    if (full_dataset || steps_col < 0) {
        return;
    }

    int cost_col = table_col(out, "uptime");
    int score_col = table_col(out, "env/score");
    if (cost_col < 0 || score_col < 0) {
        return;
    }

    int n = out->rows;
    unsigned char* keep = (unsigned char*)calloc((size_t)n, 1);
    for (int i = 0; i < n; i++) {
        keep[i] = 1;
        for (int j = 0; j < n; j++) {
            if (table_get(out, j, score_col) >= table_get(out, i, score_col) &&
                    table_get(out, j, cost_col) < table_get(out, i, cost_col) &&
                    table_get(out, j, steps_col) < table_get(out, i, steps_col)) {
                keep[i] = 0;
                break;
            }
        }
    }

    int w = 0;
    for (int r = 0; r < n; r++) {
        if (!keep[r]) {
            continue;
        }
        for (int c = 0; c < out->cols; c++) {
            table_set(out, w, c, table_get(out, r, c));
        }
        w++;
    }
    out->rows = w;
    free(keep);
}

static void write_env(FILE* fp, const char* env, Table* table) {
    if (table->rows == 0) {
        return;
    }

    fprintf(fp, "\n[%s]\n", env);
    for (int c = 0; c < table->cols; c++) {
        fprintf(fp, "%s = ", table->labels[c]);
        for (int r = 0; r < table->rows; r++) {
            if (r > 0) {
                fputc(',', fp);
            }
            fprintf(fp, "%.6g", table_get(table, r, c));
        }
        fputc('\n', fp);
    }
}

int main(int argc, char** argv) {
    int full_dataset = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--full") == 0) {
            full_dataset = 1;
        }
    }

    if (is_dir("resources/constellation") == 0) {
        mkdir("resources/constellation", 0777);
    }

    FILE* fp = fopen("resources/constellation/experiments.ini", "w");
    if (!fp) {
        perror("resources/constellation/experiments.ini");
        return 1;
    }
    fprintf(fp, "# PufferLib constellation cache v1\n");

    struct dirent** ents = NULL;
    int nents = scandir("logs", &ents, NULL, alphasort);
    if (nents > 0) {
        for (int i = 0; i < nents; i++) {
            if (ents[i]->d_name[0] == '.') {
                free(ents[i]);
                continue;
            }

            char path[1024];
            snprintf(path, sizeof(path), "logs/%s", ents[i]->d_name);
            if (is_dir(path)) {
                Table table = {0};
                snprintf(table.name, sizeof(table.name), "%s", ents[i]->d_name);
                load_env(ents[i]->d_name, full_dataset, &table);
                write_env(fp, ents[i]->d_name, &table);
                table_free(&table);
            }
            free(ents[i]);
        }
        free(ents);
    }

    fclose(fp);
    return 0;
}
