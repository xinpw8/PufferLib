#pragma once

#include <dirent.h>
#include <errno.h>
#include <unistd.h>

#include "checkpoint.h"

#define SELFPLAY_MAX_BANKS 8
#define SELFPLAY_MAX_POOL 1024
#define SELFPLAY_PATH_MAX 8192

typedef struct {
    char path[SELFPLAY_PATH_MAX];
} SelfplayEntry;

typedef struct {
    char cur_path[SELFPLAY_PATH_MAX];
    char pending_path[SELFPLAY_PATH_MAX];
    double hist_score;
    double hist_n;
    long opp_started_step;
    int epoch_armed;
    int num_hist_envs;
    float last_winrate_at_swap;
    int last_epochs_to_align;
} SelfplayBank;

typedef struct {
    int enabled;
    int artifact_owner;
    int world_size;
    int num_banks;
    int max_size;
    long snapshot_interval;
    long opp_timeout_steps;
    long last_snapshot_step;
    unsigned int rng;
    char pool_dir[SELFPLAY_PATH_MAX];
    char state_path[SELFPLAY_PATH_MAX];
    SelfplayEntry pool[SELFPLAY_MAX_POOL];
    int pool_size;
    SelfplayBank banks[SELFPLAY_MAX_BANKS];
} Selfplay;

static int selfplay_has_suffix(const char* s, const char* suffix) {
    size_t n = strlen(s);
    size_t m = strlen(suffix);
    return n >= m && strcmp(s + n - m, suffix) == 0;
}

static void selfplay_join(char* out, size_t out_size, const char* dir, const char* name) {
    int n = snprintf(out, out_size, "%s/%s", dir, name);
    if (n < 0 || (size_t)n >= out_size) {
        fprintf(stderr, "selfplay path too long: %s/%s\n", dir, name);
        exit(1);
    }
}

static void selfplay_checkpoint_path(char* out, size_t out_size, const char* dir, long step) {
    char name[64];
    snprintf(name, sizeof(name), "%016ld.bin", step);
    selfplay_join(out, out_size, dir, name);
}

static void selfplay_add_pool(Selfplay* sp, const char* path) {
    for (int i = 0; i < sp->pool_size; i++) {
        if (strcmp(sp->pool[i].path, path) == 0) {
            return;
        }
    }
    if (sp->pool_size >= SELFPLAY_MAX_POOL) {
        fprintf(stderr, "selfplay pool exceeds SELFPLAY_MAX_POOL\n");
        exit(1);
    }
    snprintf(sp->pool[sp->pool_size++].path, sizeof(sp->pool[0].path), "%s", path);
}

static void selfplay_load_pool_spec(Selfplay* sp, const char* spec) {
    if (!spec || spec[0] == 0) {
        return;
    }

    struct stat st;
    if (stat(spec, &st) != 0) {
        return;
    }
    if (S_ISREG(st.st_mode)) {
        if (selfplay_has_suffix(spec, ".bin")) {
            selfplay_add_pool(sp, spec);
        }
        return;
    }
    if (!S_ISDIR(st.st_mode)) {
        return;
    }

    DIR* dp = opendir(spec);
    if (!dp) {
        return;
    }
    struct dirent* ent = NULL;
    while ((ent = readdir(dp))) {
        if (!selfplay_has_suffix(ent->d_name, ".bin")) {
            continue;
        }
        char path[SELFPLAY_PATH_MAX];
        selfplay_join(path, sizeof(path), spec, ent->d_name);
        selfplay_add_pool(sp, path);
    }
    closedir(dp);
}

static void selfplay_evict(Selfplay* sp) {
    if (sp->pool_size <= sp->max_size) {
        return;
    }
    int half = sp->pool_size / 2;
    int out = 0;
    for (int i = 0; i < half; i += 2) {
        sp->pool[out++] = sp->pool[i];
    }
    for (int i = half; i < sp->pool_size; i++) {
        sp->pool[out++] = sp->pool[i];
    }
    sp->pool_size = out;
}

static const char* selfplay_sample(Selfplay* sp) {
    if (sp->pool_size == 0) {
        fprintf(stderr, "selfplay opponent pool is empty\n");
        exit(1);
    }
    int idx = (int)(rand_r(&sp->rng) % (unsigned int)sp->pool_size);
    return sp->pool[idx].path;
}

static void selfplay_publish(Selfplay* sp) {
    char tmp[SELFPLAY_PATH_MAX + 64];
    snprintf(tmp, sizeof(tmp), "%s.tmp.%d", sp->state_path, getpid());
    FILE* fp = fopen(tmp, "w");
    if (!fp) {
        fprintf(stderr, "failed to write selfplay state %s\n", tmp);
        exit(1);
    }
    fprintf(fp, "%d %d\n", sp->num_banks, sp->pool_size);
    for (int b = 0; b < sp->num_banks; b++) {
        fprintf(fp, "%s\n", sp->banks[b].cur_path);
    }
    for (int i = 0; i < sp->pool_size; i++) {
        fprintf(fp, "%s\n", sp->pool[i].path);
    }
    fclose(fp);
    if (rename(tmp, sp->state_path) != 0) {
        fprintf(stderr, "failed to publish selfplay state %s\n", sp->state_path);
        exit(1);
    }
}

static int selfplay_read_state(Selfplay* sp, int load_banks) {
    FILE* fp = fopen(sp->state_path, "r");
    if (!fp) {
        return 0;
    }

    int num_banks = 0;
    int pool_size = 0;
    if (fscanf(fp, "%d %d\n", &num_banks, &pool_size) != 2) {
        fclose(fp);
        return 0;
    }
    if (num_banks != sp->num_banks || pool_size < 0 || pool_size > SELFPLAY_MAX_POOL) {
        fclose(fp);
        return 0;
    }
    for (int b = 0; b < num_banks; b++) {
        char line[SELFPLAY_PATH_MAX];
        if (!fgets(line, sizeof(line), fp)) {
            fclose(fp);
            return 0;
        }
        line[strcspn(line, "\r\n")] = 0;
        if (load_banks) {
            snprintf(sp->banks[b].cur_path, sizeof(sp->banks[b].cur_path), "%s", line);
        }
    }
    sp->pool_size = 0;
    for (int i = 0; i < pool_size; i++) {
        char line[SELFPLAY_PATH_MAX];
        if (!fgets(line, sizeof(line), fp)) {
            fclose(fp);
            return 0;
        }
        line[strcspn(line, "\r\n")] = 0;
        selfplay_add_pool(sp, line);
    }
    fclose(fp);
    return 1;
}

static void selfplay_wait_state(Selfplay* sp) {
    for (;;) {
        if (selfplay_read_state(sp, 1)) {
            return;
        }
        usleep(50000);
    }
}

static int selfplay_count_aligned(PuffeRL* p, int tag, int reset) {
    Env* envs = (Env*)p->vec->envs;
    int count = 0;
    for (int i = 0; i < p->vec->size; i++) {
        if (envs[i].tag == tag && envs[i].boundary_reached) {
            count++;
        }
    }
    if (reset) {
        for (int i = 0; i < p->vec->size; i++) {
            if (envs[i].tag == tag) {
                envs[i].boundary_reached = 0;
            }
        }
    }
    return count;
}

static void selfplay_init_bank_counts(Selfplay* sp, PuffeRL* p) {
    Env* envs = (Env*)p->vec->envs;
    for (int b = 0; b < sp->num_banks; b++) {
        sp->banks[b].num_hist_envs = 0;
    }
    for (int i = 0; i < p->vec->size; i++) {
        int tag = envs[i].tag;
        if (tag > 0 && tag <= sp->num_banks) {
            sp->banks[tag - 1].num_hist_envs++;
        }
    }
}

static void selfplay_init(Selfplay* sp, Config* cfg, PuffeRL* p,
        const char* run_id, int artifact_owner, int world_size) {
    memset(sp, 0, sizeof(*sp));
    sp->enabled = (int)puf_config_get(cfg, "selfplay", "enabled");
    if (!sp->enabled) {
        return;
    }

    sp->artifact_owner = artifact_owner;
    sp->world_size = world_size;
    sp->num_banks = p->num_frozen_banks;
    if (sp->num_banks <= 0 || sp->num_banks > SELFPLAY_MAX_BANKS) {
        fprintf(stderr, "selfplay requires 1..%d frozen banks\n", SELFPLAY_MAX_BANKS);
        exit(1);
    }
    sp->max_size = (int)puf_config_get(cfg, "selfplay", "max_size");
    sp->snapshot_interval = (long)puf_config_get(cfg, "selfplay", "snapshot_interval");
    sp->opp_timeout_steps = (long)puf_config_get(cfg, "selfplay", "opp_timeout_steps");
    sp->rng = (unsigned int)puf_config_get(cfg, "selfplay", "seed") + (unsigned int)p->hypers.rank;
    long current_step = p->global_step * world_size;
    sp->last_snapshot_step = current_step;
    selfplay_init_bank_counts(sp, p);

    snprintf(sp->pool_dir, sizeof(sp->pool_dir), "%s/%s/%s/pool",
        puf_config_str(cfg, "base", "checkpoint_dir"),
        puf_config_str(cfg, "base", "env_name"), run_id);
    selfplay_join(sp->state_path, sizeof(sp->state_path), sp->pool_dir, "shared_opponents.txt");

    if (artifact_owner) {
        mkdir_p(sp->pool_dir);
        const char* spec = puf_config_str(cfg, "selfplay", "opponent_pool");
        selfplay_load_pool_spec(sp, spec);
        if (spec[0] != 0 && sp->pool_size == 0) {
            fprintf(stderr, "selfplay.opponent_pool resolved no .bin files: %s\n", spec);
            exit(1);
        }
        if (sp->pool_size == 0) {
            char path[SELFPLAY_PATH_MAX];
            selfplay_checkpoint_path(path, sizeof(path), sp->pool_dir, p->global_step);
            puf_save_weights(p, path);
            selfplay_add_pool(sp, path);
        }
        for (int b = 0; b < sp->num_banks; b++) {
            const char* path = selfplay_sample(sp);
            snprintf(sp->banks[b].cur_path, sizeof(sp->banks[b].cur_path), "%s", path);
            pufferl_load_frozen_bank(p, b, path);
            sp->banks[b].opp_started_step = current_step;
        }
        selfplay_publish(sp);
    } else {
        selfplay_wait_state(sp);
        for (int b = 0; b < sp->num_banks; b++) {
            pufferl_load_frozen_bank(p, b, sp->banks[b].cur_path);
            sp->banks[b].opp_started_step = current_step;
        }
    }
}

static void selfplay_log(Selfplay* sp, Dict* log) {
    if (!sp->enabled) {
        return;
    }
    dict_set(log, "pool/size", sp->pool_size);
    dict_set(log, "pool/num_banks", sp->num_banks);
    double total_score = 0;
    double total_n = 0;
    for (int b = 0; b < sp->num_banks; b++) {
        SelfplayBank* bank = &sp->banks[b];
        char key[128];
        snprintf(key, sizeof(key), "pool/winrate_at_swap_bank_%d", b);
        dict_set(log, key, bank->last_winrate_at_swap);
        snprintf(key, sizeof(key), "pool/epochs_to_align_bank_%d", b);
        dict_set(log, key, bank->last_epochs_to_align);
        if (bank->hist_n > 0) {
            double wr = bank->hist_score / bank->hist_n;
            snprintf(key, sizeof(key), "pool/winrate_bank_%d", b);
            dict_set(log, key, wr);
            snprintf(key, sizeof(key), "env/historical_winrate_bank_%d", b);
            dict_set(log, key, wr);
        }
        total_score += bank->hist_score;
        total_n += bank->hist_n;
    }
    if (total_n > 0) {
        double wr = total_score / total_n;
        dict_set(log, "pool/winrate", wr);
        dict_set(log, "env/historical_winrate", wr);
    }
}

static void selfplay_step(Selfplay* sp, PuffeRL* p, Dict* log, int epoch) {
    if (!sp->enabled) {
        return;
    }

    if (!sp->artifact_owner) {
        selfplay_read_state(sp, 0);
    }

    double n_window = puf_log_get_or(log, "env/n", 0);
    long current_step = p->global_step * sp->world_size;
    for (int b = 0; b < sp->num_banks; b++) {
        SelfplayBank* bank = &sp->banks[b];
        char key[128];
        snprintf(key, sizeof(key), "env/hist_score_bank_%d", b);
        double score = puf_log_get_or(log, key, 0) * n_window;
        snprintf(key, sizeof(key), "env/hist_n_bank_%d", b);
        double n = puf_log_get_or(log, key, 0) * n_window;
        if (n > 0) {
            bank->hist_score += score;
            bank->hist_n += n;
        }
    }

    int pool_changed = 0;
    if (sp->artifact_owner && sp->snapshot_interval > 0 &&
            current_step - sp->last_snapshot_step >= sp->snapshot_interval) {
        char path[SELFPLAY_PATH_MAX];
        selfplay_checkpoint_path(path, sizeof(path), sp->pool_dir, p->global_step);
        puf_save_weights(p, path);
        selfplay_add_pool(sp, path);
        selfplay_evict(sp);
        sp->last_snapshot_step = current_step;
        pool_changed = 1;
    }

    int opponent_changed = 0;
    for (int b = 0; b < sp->num_banks; b++) {
        SelfplayBank* bank = &sp->banks[b];
        int timed_out = sp->opp_timeout_steps > 0 &&
            current_step - bank->opp_started_step >= sp->opp_timeout_steps;
        int tag = b + 1;
        if (bank->pending_path[0]) {
            if (selfplay_count_aligned(p, tag, 0) >= bank->num_hist_envs) {
                pufferl_load_frozen_bank(p, b, bank->pending_path);
                selfplay_count_aligned(p, tag, 1);
                snprintf(bank->cur_path, sizeof(bank->cur_path), "%s", bank->pending_path);
                bank->pending_path[0] = 0;
                bank->hist_score = 0;
                bank->hist_n = 0;
                bank->opp_started_step = current_step;
                bank->last_epochs_to_align = epoch - bank->epoch_armed;
                opponent_changed = 1;
            }
        } else if (timed_out) {
            const char* path = selfplay_sample(sp);
            snprintf(bank->pending_path, sizeof(bank->pending_path), "%s", path);
            bank->epoch_armed = epoch;
            bank->last_winrate_at_swap = bank->hist_n > 0 ?
                (float)(bank->hist_score / bank->hist_n) : 0.0f;
            selfplay_count_aligned(p, tag, 1);
        }
    }

    if (sp->artifact_owner && (pool_changed || opponent_changed)) {
        selfplay_publish(sp);
    }
    selfplay_log(sp, log);
}
