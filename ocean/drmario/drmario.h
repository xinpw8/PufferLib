/* Squared: a sample single-agent grid env.
 * Use this as a tutorial and template for your first env.
 * See the Target env for a slightly more complex example.
 * Star PufferLib on GitHub to support. It really, really helps!
 */

#include <stdlib.h>
#include <string.h>
#include "raylib.h"

const unsigned char NOOP = 0;
const unsigned char DOWN = 1;
const unsigned char UP = 2;
const unsigned char LEFT = 3;
const unsigned char RIGHT = 4;

const unsigned char EMPTY = 0;
const unsigned char AGENT = 1;
const unsigned char TARGET = 2;

// Required struct. Only use floats!
typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported in binding.c
    float n; // Required as the last field
} Log;

// Required that you have some struct for your env

typedef struct {
    Client *client;
    Log log;

    float *observations;
    float *actions;
    float *rewards;
    float *terminals;
    int dim_obs;

    int n_rows;
    int n_cols;
    int *grid;

    int cap_color_a;
    int cap_color_b;
    int cap_orient;
    int cap_row;
    int cap_col;

    int tick;
    int tick_fall;
    int ticks_per_fall;

    int viruses_remaining;
    int n_init_viruses;

    float episode_return;
    int viruses_cleared;
    int atn_count_soft_drop;
    int atn_count_hard_drop;
    int atn_count_rotate;

    unsigned int rng;
} DrMario;


void add_log(Squared* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
    env->log.score += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.n++;
}

// Required function
void c_reset(DrMario* env) {
    int tiles = env->size*env->size;
    memset(env->observations, 0, tiles*sizeof(unsigned char));
    env->observations[tiles/2] = AGENT;
    env->r = env->size/2;
    env->c = env->size/2;
    env->tick = 0;
    int target_idx = 0; // Deterministic for testing
    do {
        target_idx = rand_r(&env->rng) % tiles;
    } while (target_idx == tiles/2);
    env->observations[target_idx] = TARGET;
}

// Required function
void c_step(Squared* env) {
    env->tick += 1;

    int action = (int)env->actions[0];
    env->terminals[0] = 0;
    env->rewards[0] = 0;

    env->observations[env->r*env->size + env->c] = EMPTY;

    if (action == DOWN) {
        env->r += 1;
    } else if (action == RIGHT) {
        env->c += 1;
    } else if (action == UP) {
        env->r -= 1;
    } else if (action == LEFT) {
        env->c -= 1;
    }

    if (env->tick > 3*env->size
            || env->r < 0
            || env->c < 0
            || env->r >= env->size
            || env->c >= env->size) {
        env->terminals[0] = 1;
        env->rewards[0] = -1.0;
        add_log(env);
        c_reset(env);
        return;
    }

    int pos = env->r*env->size + env->c;
    if (env->observations[pos] == TARGET) {
        env->terminals[0] = 1;
        env->rewards[0] = 1.0;
        add_log(env);
        c_reset(env);
        return;
    }

    env->observations[pos] = AGENT;
}

// Required function. Should handle creating the client on first call
void c_render(Squared* env) {
    if (!IsWindowReady()) {
        InitWindow(64*env->size, 64*env->size, "PufferLib Squared");
        SetTargetFPS(5);
    }

    // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    int px = 64;
    for (int i = 0; i < env->size; i++) {
        for (int j = 0; j < env->size; j++) {
            int tex = env->observations[i*env->size + j];
            if (tex == EMPTY) {
                continue;
            }
            Color color = (tex == AGENT) ? (Color){0, 187, 187, 255} : (Color){187, 0, 0, 255};
            DrawRectangle(j*px, i*px, px, px, color);
        }
    }

    EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Squared* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
