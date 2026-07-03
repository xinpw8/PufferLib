#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

typedef struct {
    char* key;
    double* data;
    int n;
    int cap;
} Series;

typedef struct {
    Series* items;
    int n;
    int cap;
} Table;

typedef struct {
    char* key;
    double value;
} Scalar;

typedef struct {
    Scalar* scalars;
    int num_scalars;
    int cap_scalars;
    Table metrics;
} Experiment;

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

static char* xstrdup(const char* s) {
    size_t n = strlen(s) + 1;
    char* out = (char*)malloc(n);
    if (!out) {
        perror("malloc");
        exit(1);
    }
    memcpy(out, s, n);
    return out;
}

static char* trim(char* s) {
    while (isspace((unsigned char)*s)) {
        s++;
    }
    char* e = s + strlen(s);
    while (e > s && isspace((unsigned char)e[-1])) {
        *--e = 0;
    }
    return s;
}

static void strip_comment(char* s) {
    int quote = 0;
    for (char* p = s; *p; p++) {
        if ((*p == '\'' || *p == '"') && (p == s || p[-1] != '\\')) {
            quote = quote == *p ? 0 : quote ? quote : *p;
        }
        if ((*p == '#' || *p == ';') && !quote) {
            *p = 0;
            return;
        }
    }
}

static int parse_num(const char* raw, double* out) {
    char buf[256];
    int j = 0;
    for (int i = 0; raw[i] && j + 1 < (int)sizeof(buf); i++) {
        if (raw[i] != '_' && !isspace((unsigned char)raw[i])) {
            buf[j++] = raw[i];
        }
    }
    buf[j] = 0;
    if (strcmp(buf, "True") == 0 || strcmp(buf, "true") == 0) {
        *out = 1;
        return 1;
    }
    if (strcmp(buf, "False") == 0 || strcmp(buf, "false") == 0) {
        *out = 0;
        return 1;
    }

    char* end = NULL;
    double v = strtod(buf, &end);
    if (!buf[0] || !end || *end) {
        return 0;
    }
    *out = v;
    return 1;
}

static Series* table_get(Table* t, const char* key) {
    for (int i = 0; i < t->n; i++) {
        if (strcmp(t->items[i].key, key) == 0) {
            return &t->items[i];
        }
    }
    if (t->n == t->cap) {
        t->cap = t->cap ? 2 * t->cap : 32;
        t->items = (Series*)realloc(t->items, (size_t)t->cap * sizeof(Series));
        if (!t->items) {
            perror("realloc");
            exit(1);
        }
    }
    Series* s = &t->items[t->n++];
    memset(s, 0, sizeof(*s));
    s->key = xstrdup(key);
    return s;
}

static Series* table_find(Table* t, const char* key) {
    for (int i = 0; i < t->n; i++) {
        if (strcmp(t->items[i].key, key) == 0) {
            return &t->items[i];
        }
    }
    return NULL;
}

static void series_push(Series* s, double v) {
    if (s->n == s->cap) {
        s->cap = s->cap ? 2 * s->cap : 64;
        s->data = (double*)realloc(s->data, (size_t)s->cap * sizeof(double));
        if (!s->data) {
            perror("realloc");
            exit(1);
        }
    }
    s->data[s->n++] = v;
}

static void table_add_values(Table* t, const char* key, double* values, int n) {
    Series* s = table_get(t, key);
    for (int i = 0; i < n; i++) {
        series_push(s, values[i]);
    }
}

static void table_add_const(Table* t, const char* key, double value, int n) {
    Series* s = table_get(t, key);
    for (int i = 0; i < n; i++) {
        series_push(s, value);
    }
}

static void exp_set_scalar(Experiment* e, const char* key, double value) {
    for (int i = 0; i < e->num_scalars; i++) {
        if (strcmp(e->scalars[i].key, key) == 0) {
            e->scalars[i].value = value;
            return;
        }
    }
    if (e->num_scalars == e->cap_scalars) {
        e->cap_scalars = e->cap_scalars ? 2 * e->cap_scalars : 64;
        e->scalars = (Scalar*)realloc(e->scalars,
            (size_t)e->cap_scalars * sizeof(Scalar));
        if (!e->scalars) {
            perror("realloc");
            exit(1);
        }
    }
    e->scalars[e->num_scalars].key = xstrdup(key);
    e->scalars[e->num_scalars].value = value;
    e->num_scalars++;
}

static int exp_get_scalar(Experiment* e, const char* key, double* out) {
    for (int i = 0; i < e->num_scalars; i++) {
        if (strcmp(e->scalars[i].key, key) == 0) {
            *out = e->scalars[i].value;
            return 1;
        }
    }
    return 0;
}

static void free_table(Table* t) {
    for (int i = 0; i < t->n; i++) {
        free(t->items[i].key);
        free(t->items[i].data);
    }
    free(t->items);
    memset(t, 0, sizeof(*t));
}

static void free_exp(Experiment* e) {
    for (int i = 0; i < e->num_scalars; i++) {
        free(e->scalars[i].key);
    }
    free(e->scalars);
    free_table(&e->metrics);
    memset(e, 0, sizeof(*e));
}

static int load_ini_log(const char* path, Experiment* e) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        return 0;
    }
    char section[256] = "base";
    char line[8192];
    while (fgets(line, sizeof(line), fp)) {
        strip_comment(line);
        char* s = trim(line);
        if (!*s) {
            continue;
        }
        size_t len = strlen(s);
        if (s[0] == '[' && len > 2 && s[len - 1] == ']') {
            s[len - 1] = 0;
            snprintf(section, sizeof(section), "%s", trim(s + 1));
            continue;
        }
        char* eq = strchr(s, '=');
        if (!eq) {
            fclose(fp);
            return 0;
        }
        *eq = 0;
        char* key = trim(s);
        char* val = trim(eq + 1);

        if (strcmp(section, "metrics") == 0) {
            if (strstr(key, "loss")) {
                continue;
            }
            Series* series = table_get(&e->metrics, key);
            char* p = val;
            while (*p) {
                char* end = NULL;
                double x = strtod(p, &end);
                if (end == p) {
                    break;
                }
                series_push(series, x);
                p = end;
                while (*p == ',' || isspace((unsigned char)*p)) {
                    p++;
                }
            }
        } else {
            double x = 0;
            if (!parse_num(val, &x)) {
                continue;
            }
            char full[512];
            if (strcmp(section, "base") == 0) {
                snprintf(full, sizeof(full), "%s", key);
            } else {
                snprintf(full, sizeof(full), "%s/%s", section, key);
            }
            exp_set_scalar(e, full, x);
        }
    }
    fclose(fp);
    return 1;
}

static int valid_experiment(Experiment* e) {
    Series* steps = table_find(&e->metrics, "agent_steps");
    if (!steps || steps->n == 0) {
        return 0;
    }
    int n = steps->n;
    for (int i = 0; i < e->metrics.n; i++) {
        Series* s = &e->metrics.items[i];
        if (s->n != n) {
            return 0;
        }
        for (int j = 0; j < s->n; j++) {
            if (isnan(s->data[j])) {
                return 0;
            }
        }
    }
    return 1;
}

static int has_suffix(const char* s, const char* suffix) {
    size_t n = strlen(s);
    size_t m = strlen(suffix);
    return n >= m && strcmp(s + n - m, suffix) == 0;
}

static int is_dir(const char* path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
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

        Experiment exp = {0};
        if (!load_ini_log(path, &exp) || !valid_experiment(&exp)) {
            free_exp(&exp);
            continue;
        }
        int rows = table_find(&exp.metrics, "agent_steps")->n;
        for (int m = 0; m < exp.metrics.n; m++) {
            table_add_values(out, exp.metrics.items[m].key,
                exp.metrics.items[m].data, rows);
        }
        for (int s = 0; s < exp.num_scalars; s++) {
            table_add_const(out, exp.scalars[s].key, exp.scalars[s].value, rows);
        }
        for (int k = 0; k < (int)(sizeof(EXTRA_KEYS) / sizeof(EXTRA_KEYS[0])); k++) {
            double v = 0;
            if (!exp_get_scalar(&exp, EXTRA_KEYS[k], &v)) {
                table_add_const(out, EXTRA_KEYS[k], 0, rows);
            }
        }
        free_exp(&exp);
    }
    free(ents);

    Series* steps = table_find(out, "agent_steps");
    Series* total_steps = table_find(out, "train/total_timesteps");
    if (steps) {
        for (int i = 0; i < steps->n; i++) {
            steps->data[i] /= 1e6;
        }
    }
    if (total_steps) {
        for (int i = 0; i < total_steps->n; i++) {
            total_steps->data[i] /= 1e6;
        }
    }

    if (steps) {
        Series* tsne1 = table_get(out, "tsne1");
        Series* tsne2 = table_get(out, "tsne2");
        for (int i = 0; i < steps->n; i++) {
            series_push(tsne1, (double)(i % 997) / 997.0);
            series_push(tsne2, (double)((i * 37) % 991) / 991.0);
        }
    }

    if (full_dataset || !steps) {
        return;
    }

    Series* cost = table_find(out, "uptime");
    Series* score = table_find(out, "env/score");
    if (!cost || !score || cost->n != steps->n || score->n != steps->n) {
        return;
    }

    int n = steps->n;
    unsigned char* keep = (unsigned char*)calloc((size_t)n, 1);
    for (int i = 0; i < n; i++) {
        keep[i] = 1;
        for (int j = 0; j < n; j++) {
            if (score->data[j] >= score->data[i] &&
                    cost->data[j] < cost->data[i] &&
                    steps->data[j] < steps->data[i]) {
                keep[i] = 0;
                break;
            }
        }
    }
    for (int k = 0; k < out->n; k++) {
        Series* s = &out->items[k];
        if (s->n != n) {
            continue;
        }
        int w = 0;
        for (int i = 0; i < n; i++) {
            if (keep[i]) {
                s->data[w++] = s->data[i];
            }
        }
        s->n = w;
    }
    free(keep);
}

static void write_env(FILE* fp, const char* env, Table* t) {
    Series* steps = table_find(t, "agent_steps");
    if (!steps || steps->n == 0) {
        return;
    }
    fprintf(fp, "\n[%s]\n", env);
    for (int i = 0; i < t->n; i++) {
        Series* s = &t->items[i];
        if (s->n != steps->n) {
            continue;
        }
        fprintf(fp, "%s = ", s->key);
        for (int j = 0; j < s->n; j++) {
            if (j > 0) {
                fputc(',', fp);
            }
            fprintf(fp, "%.6g", s->data[j]);
        }
        fprintf(fp, "\n");
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
                Table t = {0};
                load_env(ents[i]->d_name, full_dataset, &t);
                write_env(fp, ents[i]->d_name, &t);
                free_table(&t);
            }
            free(ents[i]);
        }
        free(ents);
    }

    fclose(fp);
    return 0;
}
