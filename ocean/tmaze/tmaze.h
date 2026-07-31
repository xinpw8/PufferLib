#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "raylib.h"
#include "pufferenv.h"

#define ACT_SIZES {3}
#define OBS_SIZE 4
#define NUM_ATNS 1
typedef unsigned char obs_t;

const unsigned char FORWARD = 0;
const unsigned char RIGHT = 1;
const unsigned char LEFT = 2;

const unsigned char EMPTY = 1;
const unsigned char WALL = 0;

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))

struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
};

struct Env {
    Log log;
    Agent agents[1];
    int tag;
    int boundary_reached;
    int num_agents;
    int size;
    int tick;
    unsigned char state;
    unsigned char starting_state;
    Texture2D puffer;
    unsigned int rng;
};
typedef Env TMaze;

void free_allocated(TMaze* env) {
    free(env->agents[0].observations);
    free(env->agents[0].actions);
    free(env->agents[0].rewards);
    free(env->agents[0].terminals);
}

void add_log(TMaze* env) {
    env->log.perf += (env->agents[0].rewards[0] + 1) / 2;
    env->log.score += env->agents[0].rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->agents[0].rewards[0];
    env->log.n++;
}

void puf_reset(TMaze* env) {
    memset(((obs_t*)env->agents[0].observations), WALL, 4 * sizeof(obs_t));
    env->starting_state = (rand_r(&env->rng) % 2) + 2; // 2 or 3
    ((obs_t*)env->agents[0].observations)[0] = env->starting_state;
    ((obs_t*)env->agents[0].observations)[1] = EMPTY;
    env->tick = 0;
    env->state = 0;
}

void compute_observations(TMaze* env) {
    if (env->state == env->size - 1) {
        ((obs_t*)env->agents[0].observations)[0] = EMPTY;
        ((obs_t*)env->agents[0].observations)[1] = WALL;
        ((obs_t*)env->agents[0].observations)[2] = EMPTY;
        ((obs_t*)env->agents[0].observations)[3] = EMPTY;
        return;
    }
    ((obs_t*)env->agents[0].observations)[0] = EMPTY;
    ((obs_t*)env->agents[0].observations)[1] = EMPTY;
    ((obs_t*)env->agents[0].observations)[2] = WALL;
    ((obs_t*)env->agents[0].observations)[3] = WALL;
}

void puf_step(TMaze* env) {
    env->tick += 1;
    env->agents[0].terminals[0] = 0;
    env->agents[0].rewards[0] = 0;

    int action = (int)env->agents[0].actions[0];

    if (env->state == env->size - 1) {
        const int left_reward = (env->starting_state == 2) ? 1 : -1;
        const int right_reward = (env->starting_state == 3) ? 1 : -1;

        if (action == LEFT || action == RIGHT) {
            env->agents[0].rewards[0] = (action == LEFT) ? left_reward : right_reward;
            env->agents[0].terminals[0] = 1;
            add_log(env);
            puf_reset(env);
        }
    } else {
        if (action == FORWARD) {
            env->state += 1;
            compute_observations(env);
        }
    }
}

void puf_render(TMaze* env) {
    int px = MAX(8, 1024.0 / env->size);

    if (!IsWindowReady()) {
        InitWindow(px * env->size, px * 5, "PufferLib TMaze MDP");
        SetTargetFPS(4);
        env->puffer = LoadTexture("resources/shared/puffers_128.png");
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    int agent_pos = env->state;
    for (int i = 0; i < env->size; i++) {
        Color color =
            (i == agent_pos) ? (Color){0, 255, 255, 255} :
            (i == 0 && env->starting_state == 2) ? (Color){255, 0, 0, 255} :
            (i == 0 && env->starting_state == 3) ? (Color){0, 255, 0, 255} :
            (Color){224, 224, 224, 255};

        if (i == agent_pos) {
            int starting_sprite_x = 0;
            Rectangle source_rect = (Rectangle){(float)starting_sprite_x, 0, 128, 128};
            Rectangle dest_rect = (Rectangle){(float)(i * px), (float)(2 * px), (float)px, (float)px};
            DrawTexturePro(env->puffer, source_rect, dest_rect, (Vector2){0, 0}, 0, color);
        } else {
            DrawRectangle(i * px, 2 * px, px, px, color);
        }
    }
    DrawRectangle((env->size - 1) * px, 1 * px, px, px, (Color){255, 0, 0, 255});
    DrawRectangle((env->size - 1) * px, 3 * px, px, px, (Color){0, 255, 0, 255});

    char score_text[32];
    snprintf(score_text, sizeof(score_text), "Score: %f", env->agents[0].rewards[0]);
    DrawText(score_text, env->size * px - 180, 10, 32, (Color){255, 255, 255, 255});

    EndDrawing();
}

void puf_close(TMaze* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->size = dict_get(kwargs, "size");
    env->agents[0].policy = 0;
    env->agents[0].action_mask = NULL;
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}

