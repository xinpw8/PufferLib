#pragma once

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
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

#define LEAGUE_MAX_PLAYERS 2048
#define LEAGUE_MAX_MATCHES 8192
#define LEAGUE_ID_MAX 128
#define LEAGUE_PATH_MAX 4096

typedef struct {
    char id[LEAGUE_ID_MAX];
    char path[LEAGUE_PATH_MAX];
    float elo;
    float cost;
    int games;
    int matches;
} LeaguePlayer;

typedef struct {
    char a[LEAGUE_ID_MAX];
    char b[LEAGUE_ID_MAX];
    int games;
    float score;
    float draw;
} LeagueMatch;

typedef struct {
    LeaguePlayer players[LEAGUE_MAX_PLAYERS];
    LeagueMatch matches[LEAGUE_MAX_MATCHES];
    int num_players;
    int num_matches;
} LeagueState;

static void league_lock_path(char* out, size_t out_size, const char* path) {
    snprintf(out, out_size, "%s.lock", path);
}

static int league_lock(const char* path) {
    char lock_path[LEAGUE_PATH_MAX];
    league_lock_path(lock_path, sizeof(lock_path), path);
    int fd = open(lock_path, O_CREAT | O_RDWR, 0666);
    if (fd < 0) {
        perror("open league lock");
        exit(1);
    }
    if (flock(fd, LOCK_EX) != 0) {
        perror("flock");
        exit(1);
    }
    return fd;
}

static void league_unlock(int fd) {
    flock(fd, LOCK_UN);
    close(fd);
}

static int league_player_index(LeagueState* st, const char* id) {
    for (int i = 0; i < st->num_players; i++) {
        if (strcmp(st->players[i].id, id) == 0) {
            return i;
        }
    }
    return -1;
}

static void league_load_unlocked(const char* path, LeagueState* st) {
    memset(st, 0, sizeof(*st));
    FILE* fp = fopen(path, "r");
    if (!fp) {
        return;
    }

    char type[32];
    while (fscanf(fp, "%31s", type) == 1) {
        if (strcmp(type, "PLAYER") == 0) {
            if (st->num_players >= LEAGUE_MAX_PLAYERS) {
                fprintf(stderr, "league player cap exceeded\n");
                exit(1);
            }
            LeaguePlayer* p = &st->players[st->num_players++];
            if (fscanf(fp, "%127s %4095s %f %f %d %d",
                    p->id, p->path, &p->elo, &p->cost,
                    &p->games, &p->matches) != 6) {
                fprintf(stderr, "malformed league PLAYER row in %s\n", path);
                exit(1);
            }
        } else if (strcmp(type, "MATCH") == 0) {
            if (st->num_matches >= LEAGUE_MAX_MATCHES) {
                fprintf(stderr, "league match cap exceeded\n");
                exit(1);
            }
            LeagueMatch* m = &st->matches[st->num_matches++];
            if (fscanf(fp, "%127s %127s %d %f %f",
                    m->a, m->b, &m->games, &m->score, &m->draw) != 5) {
                fprintf(stderr, "malformed league MATCH row in %s\n", path);
                exit(1);
            }
        } else {
            char line[4096];
            if (!fgets(line, sizeof(line), fp)) {
                break;
            }
        }
    }
    fclose(fp);
}

static void league_write_unlocked(const char* path, LeagueState* st) {
    char tmp[LEAGUE_PATH_MAX];
    snprintf(tmp, sizeof(tmp), "%s.tmp.%d", path, getpid());
    FILE* fp = fopen(tmp, "w");
    if (!fp) {
        fprintf(stderr, "failed to write league state %s\n", tmp);
        exit(1);
    }
    fprintf(fp, "# PufferLib native league v1\n");
    for (int i = 0; i < st->num_players; i++) {
        LeaguePlayer* p = &st->players[i];
        fprintf(fp, "PLAYER %s %s %.9g %.9g %d %d\n",
            p->id, p->path, p->elo, p->cost, p->games, p->matches);
    }
    for (int i = 0; i < st->num_matches; i++) {
        LeagueMatch* m = &st->matches[i];
        fprintf(fp, "MATCH %s %s %d %.9g %.9g\n",
            m->a, m->b, m->games, m->score, m->draw);
    }
    fclose(fp);
    if (rename(tmp, path) != 0) {
        fprintf(stderr, "failed to publish league state %s\n", path);
        exit(1);
    }
}

static void league_recompute(LeagueState* st) {
    for (int i = 0; i < st->num_players; i++) {
        st->players[i].elo = 0;
        st->players[i].games = 0;
        st->players[i].matches = 0;
    }
    for (int iter = 0; iter < 100; iter++) {
        for (int i = 0; i < st->num_matches; i++) {
            LeagueMatch* m = &st->matches[i];
            int ai = league_player_index(st, m->a);
            int bi = league_player_index(st, m->b);
            if (ai < 0 || bi < 0 || ai == bi || m->games <= 0) {
                continue;
            }
            float ea = 1.0f / (1.0f + powf(10.0f,
                (st->players[bi].elo - st->players[ai].elo) / 400.0f));
            float delta = 0.02f * (float)m->games * (m->score - ea);
            st->players[ai].elo += delta;
            st->players[bi].elo -= delta;
        }
    }
    for (int i = 0; i < st->num_matches; i++) {
        LeagueMatch* m = &st->matches[i];
        int ai = league_player_index(st, m->a);
        int bi = league_player_index(st, m->b);
        if (ai >= 0) {
            st->players[ai].games += m->games;
            st->players[ai].matches++;
        }
        if (bi >= 0) {
            st->players[bi].games += m->games;
            st->players[bi].matches++;
        }
    }
}

static void league_register_player(const char* path, const char* id,
        const char* checkpoint, float cost) {
    int lock = league_lock(path);
    LeagueState st;
    league_load_unlocked(path, &st);
    int idx = league_player_index(&st, id);
    if (idx < 0) {
        if (st.num_players >= LEAGUE_MAX_PLAYERS) {
            fprintf(stderr, "league player cap exceeded\n");
            exit(1);
        }
        idx = st.num_players++;
    }
    LeaguePlayer* p = &st.players[idx];
    snprintf(p->id, sizeof(p->id), "%s", id);
    snprintf(p->path, sizeof(p->path), "%s", checkpoint);
    p->cost = cost;
    league_recompute(&st);
    league_write_unlocked(path, &st);
    league_unlock(lock);
}

static float league_player_elo(const char* path, const char* id) {
    int lock = league_lock(path);
    LeagueState st;
    league_load_unlocked(path, &st);
    league_recompute(&st);
    int idx = league_player_index(&st, id);
    float elo = idx >= 0 ? st.players[idx].elo : 0;
    league_unlock(lock);
    return elo;
}

static void league_record_match(const char* path, const char* a, const char* b,
        int games, float score, float draw) {
    int lock = league_lock(path);
    LeagueState st;
    league_load_unlocked(path, &st);
    if (st.num_matches >= LEAGUE_MAX_MATCHES) {
        fprintf(stderr, "league match cap exceeded\n");
        exit(1);
    }
    LeagueMatch* m = &st.matches[st.num_matches++];
    snprintf(m->a, sizeof(m->a), "%s", a);
    snprintf(m->b, sizeof(m->b), "%s", b);
    m->games = games;
    m->score = score;
    m->draw = draw;
    league_recompute(&st);
    league_write_unlocked(path, &st);
    league_unlock(lock);
}

static int league_choose_pair(const char* path, LeaguePlayer* a, LeaguePlayer* b,
        unsigned int* rng) {
    int lock = league_lock(path);
    LeagueState st;
    league_load_unlocked(path, &st);
    int n = st.num_players;
    if (n < 2) {
        league_unlock(lock);
        return 0;
    }
    int ai = (int)(rand_r(rng) % (unsigned int)n);
    int bi = ai;
    for (int tries = 0; tries < 32 && bi == ai; tries++) {
        bi = (int)(rand_r(rng) % (unsigned int)n);
    }
    if (bi == ai) {
        bi = (ai + 1) % n;
    }
    *a = st.players[ai];
    *b = st.players[bi];
    league_unlock(lock);
    return 1;
}

static const char* resolve_checkpoint_key(Config* cfg, const char* key,
        char* out, size_t out_size) {
    const char* load_path = NULL;
    if (strcmp(key, "load_model_path") == 0) {
        load_path = puf_config_str(cfg, "base", "load_model_path");
    } else if (strcmp(key, "load_enemy_model_path") == 0) {
        load_path = puf_config_str(cfg, "base", "load_enemy_model_path");
    }
    if (!load_path || strcmp(load_path, "None") == 0) {
        return NULL;
    }
    if (strcmp(load_path, "latest") != 0) {
        return load_path;
    }

    char root[2048];
    snprintf(root, sizeof(root), "%s/%s",
        puf_config_str(cfg, "base", "checkpoint_dir"),
        puf_config_str(cfg, "base", "env_name"));
    out[0] = 0;
    time_t best_time = 0;
    puf_find_latest_checkpoint(root, out, out_size, &best_time);
    if (!out[0]) {
        fprintf(stderr, "no .bin checkpoints found in %s\n", root);
        exit(1);
    }
    return out;
}

static void load_primary_if_configured(PuffeRL* pufferl, Config* cfg) {
    char resolved_path[4096];
    const char* load_path = resolve_checkpoint_key(cfg,
        "load_model_path", resolved_path, sizeof(resolved_path));
    if (load_path) {
        puf_load_weights(pufferl, load_path);
        printf("Loaded weights from %s\n", load_path);
    }
}

void run_eval_bot(Config* cfg, TrainContext* ctx) {
    long num_games = puf_config_long(cfg, "base", "num_games");
    if (!num_games) {
        num_games = puf_config_long(cfg, "base", "eval_episodes");
    }
    long burnin_games = puf_config_long(cfg, "base", "burnin_games");
    long eval_agents = puf_config_long(cfg, "base", "eval_agents");
    if (num_games <= 0 || burnin_games < 0) {
        fprintf(stderr, "eval_bot requires positive num_games and nonnegative burnin_games\n");
        exit(1);
    }
    if (eval_agents <= 0) {
        eval_agents = num_games / 8;
        if (eval_agents < 1024) {
            eval_agents = 1024;
        }
        if (eval_agents > 4096) {
            eval_agents = 4096;
        }
        if (eval_agents > num_games && num_games >= 1024) {
            eval_agents = num_games;
        }
    } else if (eval_agents > num_games) {
        eval_agents = num_games;
    }
    eval_agents += (-eval_agents) % 2;

    char buf[64];
    puf_config_put(cfg, "base.reset_state", "0");
    puf_config_put(cfg, "train.horizon", "1");
    puf_config_put(cfg, "vec.num_buffers", "2");
    snprintf(buf, sizeof(buf), "%ld", eval_agents);
    puf_config_put(cfg, "vec.total_agents", buf);
    puf_config_put(cfg, "vec.num_frozen_banks", "0");
    puf_config_put(cfg, "vec.frozen_bank_pct", "0");
    puf_config_put(cfg, "selfplay.enabled", "0");
    puf_config_put(cfg, "env.dr", "0");
    puf_config_put(cfg, "env.num_agents", "1");
    puf_config_put(cfg, "env.num_bots", "1");

    PuffeRL* pufferl = create_trainer(cfg, ctx);
    load_primary_if_configured(pufferl, cfg);

    Dict baseline = {0};
    int has_baseline = 0;
    long baseline_n = 0;
    for (;;) {
        rollouts(pufferl);
        Dict log = {0};
        trainer_eval_log(pufferl, &log);
        long n = (long)puf_log_get_or(&log, "env/n", 0);
        if (burnin_games > 0 && !has_baseline && n >= burnin_games) {
            baseline = log;
            has_baseline = 1;
            baseline_n = n;
            printf("\rbot_eval_burnin=%ld/%ld", n, burnin_games);
            continue;
        }

        double scored_n = n - baseline_n;
        double score = puf_log_get_or(&log, "env/score", 0);
        double perf = puf_log_get_or(&log, "env/perf", 0);
        if (has_baseline && scored_n > 0) {
            double base_n = (double)baseline_n;
            double cur_n = (double)n;
            score = (score * cur_n - puf_log_get_or(&baseline, "env/score", 0) * base_n) / scored_n;
            perf = (perf * cur_n - puf_log_get_or(&baseline, "env/perf", 0) * base_n) / scored_n;
        }
        printf("\rbot_eval=%.0f/%ld  perf=%.4f  score=%.3f",
            scored_n, num_games, perf, score);
        if ((n - baseline_n) >= num_games && (!burnin_games || has_baseline)) {
            break;
        }
    }
    printf("\n");
    close_trainer(pufferl);
}

void run_match_eval(Config* cfg, TrainContext* ctx, int verbose,
        float* score_out, float* draw_out, int* games_out) {
    long num_games = puf_config_long(cfg, "base", "num_games");
    if (!num_games) {
        num_games = puf_config_long(cfg, "base", "eval_episodes");
    }
    long eval_agents = puf_config_long(cfg, "base", "eval_agents");
    if (!eval_agents) {
        eval_agents = puf_config_long(cfg, "sweep", "league_match_eval_agents");
    }
    if (eval_agents <= 0) {
        eval_agents = 8192;
    }
    eval_agents += (-eval_agents) % 4;

    char a_path_buf[4096];
    char b_path_buf[4096];
    const char* a_path = resolve_checkpoint_key(cfg,
        "load_model_path", a_path_buf, sizeof(a_path_buf));
    const char* b_path = resolve_checkpoint_key(cfg,
        "load_enemy_model_path", b_path_buf, sizeof(b_path_buf));
    if (!a_path || !b_path) {
        fprintf(stderr, "match requires base.load_model_path and base.load_enemy_model_path\n");
        exit(1);
    }

    char buf[64];
    puf_config_put(cfg, "base.reset_state", "0");
    puf_config_put(cfg, "train.horizon", "1");
    puf_config_put(cfg, "vec.num_buffers", "2");
    snprintf(buf, sizeof(buf), "%ld", eval_agents);
    puf_config_put(cfg, "vec.total_agents", buf);
    puf_config_put(cfg, "vec.num_frozen_banks", "1");
    puf_config_put(cfg, "vec.frozen_bank_pct", "1");
    puf_config_put(cfg, "selfplay.enabled", "0");
    puf_config_put(cfg, "env.dr", "0");
    puf_config_put(cfg, "env.num_agents", "2");
    puf_config_put(cfg, "env.num_bots", "0");

    PuffeRL* pufferl = create_trainer(cfg, ctx);
    puf_load_weights(pufferl, a_path);
    pufferl_load_frozen_bank(pufferl, 0, b_path);

    for (;;) {
        rollouts(pufferl);
        Dict log = {0};
        trainer_eval_log(pufferl, &log);
        long n = (long)puf_log_get_or(&log, "env/n", 0);
        double a = puf_log_get_or(&log, "env/slot_0_score", 0);
        double b = puf_log_get_or(&log, "env/slot_1_score", 0);
        double draw = puf_log_get_or(&log, "env/draw_rate", 0);
        if (verbose) {
            printf("\rgames=%ld/%ld  A=%.3f  B=%.3f  draw=%.3f",
                n, num_games, a, b, draw);
        }
        if (n >= num_games) {
            *score_out = (float)a;
            *draw_out = (float)draw;
            *games_out = (int)n;
            break;
        }
    }
    if (verbose) {
        printf("\n");
    }
    close_trainer(pufferl);
}

void run_match(Config* cfg, TrainContext* ctx) {
    float score = 0;
    float draw = 0;
    int games = 0;
    run_match_eval(cfg, ctx, 1, &score, &draw, &games);
}

void run_league_match_worker(Config* cfg, TrainContext* ctx) {
    const char* state_path = puf_config_str(cfg, "sweep", "league_state_path");
    long games = puf_config_long(cfg, "base", "num_games");
    if (!games) {
        games = puf_config_long(cfg, "sweep", "league_match_games");
    }
    unsigned int rng = (unsigned int)puf_config_int(cfg, "base", "seed") + 1009U;

    for (;;) {
        LeaguePlayer a;
        LeaguePlayer b;
        if (!league_choose_pair(state_path, &a, &b, &rng)) {
            usleep(500000);
            continue;
        }

        char buf[64];
        puf_config_put(cfg, "base.load_model_path", a.path);
        puf_config_put(cfg, "base.load_enemy_model_path", b.path);
        snprintf(buf, sizeof(buf), "%ld", games);
        puf_config_put(cfg, "base.num_games", buf);

        float score = 0;
        float draw = 0;
        int n = 0;
        run_match_eval(cfg, ctx, 0, &score, &draw, &n);
        league_record_match(state_path, a.id, b.id, n, score, draw);
        printf("league_match %s vs %s games=%d score=%.4f draw=%.4f\n",
            a.id, b.id, n, score, draw);
    }
}
