#pragma once

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <unistd.h>

#include "checkpoint.h"

#define LEAGUE_ID_MAX 128
#define LEAGUE_PATH_MAX 4096

typedef struct {
    char id[LEAGUE_ID_MAX];
    char path[LEAGUE_PATH_MAX];
    float elo;
} LeaguePlayer;

typedef struct {
    char a[LEAGUE_ID_MAX];
    char b[LEAGUE_ID_MAX];
    int games;
    float score;
    float draw;
} LeagueMatch;

typedef struct {
    LeaguePlayer* players;
    LeagueMatch* matches;
    int num_players;
    int num_matches;
} LeagueState;

static int league_lock(const char* path) {
    char lock_path[LEAGUE_PATH_MAX];
    snprintf(lock_path, sizeof(lock_path), "%s.lock", path);
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

static void league_free(LeagueState* st) {
    free(st->players);
    free(st->matches);
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
            st->players = (LeaguePlayer*)realloc(st->players,
                (size_t)(st->num_players + 1) * sizeof(*st->players));
            LeaguePlayer* p = &st->players[st->num_players++];
            int n = fscanf(fp, "%127s %4095s", p->id, p->path);
            (void)n;
        } else if (strcmp(type, "MATCH") == 0) {
            st->matches = (LeagueMatch*)realloc(st->matches,
                (size_t)(st->num_matches + 1) * sizeof(*st->matches));
            LeagueMatch* m = &st->matches[st->num_matches++];
            int n = fscanf(fp, "%127s %127s %d %f %f",
                m->a, m->b, &m->games, &m->score, &m->draw);
            (void)n;
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
        fprintf(fp, "PLAYER %s %s\n", p->id, p->path);
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
}

static float league_register_player(const char* path, const char* id,
        const char* checkpoint) {
    int lock = league_lock(path);
    LeagueState st;
    league_load_unlocked(path, &st);
    int idx = league_player_index(&st, id);
    if (idx < 0) {
        st.players = (LeaguePlayer*)realloc(st.players,
            (size_t)(st.num_players + 1) * sizeof(*st.players));
        idx = st.num_players++;
    }
    LeaguePlayer* p = &st.players[idx];
    snprintf(p->id, sizeof(p->id), "%s", id);
    snprintf(p->path, sizeof(p->path), "%s", checkpoint);
    league_recompute(&st);
    float elo = st.players[idx].elo;
    league_write_unlocked(path, &st);
    league_free(&st);
    league_unlock(lock);
    return elo;
}

static void league_record_match(const char* path, const char* a, const char* b,
        int games, float score, float draw) {
    int lock = league_lock(path);
    LeagueState st;
    league_load_unlocked(path, &st);
    st.matches = (LeagueMatch*)realloc(st.matches,
        (size_t)(st.num_matches + 1) * sizeof(*st.matches));
    LeagueMatch* m = &st.matches[st.num_matches++];
    snprintf(m->a, sizeof(m->a), "%s", a);
    snprintf(m->b, sizeof(m->b), "%s", b);
    m->games = games;
    m->score = score;
    m->draw = draw;
    league_write_unlocked(path, &st);
    league_free(&st);
    league_unlock(lock);
}

static int league_choose_pair(const char* path, LeaguePlayer* a, LeaguePlayer* b,
        unsigned int* rng) {
    int lock = league_lock(path);
    LeagueState st;
    league_load_unlocked(path, &st);
    int n = st.num_players;
    if (n < 2) {
        league_free(&st);
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
    league_free(&st);
    league_unlock(lock);
    return 1;
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

        EvalResult result = run_eval(cfg, ctx, EVAL_MATCH, 0);
        league_record_match(state_path, a.id, b.id,
            result.games, result.score, result.draw);
        printf("league_match %s vs %s games=%d score=%.4f draw=%.4f\n",
            a.id, b.id, result.games, result.score, result.draw);
    }
}
