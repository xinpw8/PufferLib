#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include "raylib.h"

#define LENGTH 128
#define NUM_DIGITS 9
#define PASSWORD_SEED 42u
#define CORRECT_REWARD 1.0f
#define WRONG_REWARD -1.0f

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float success;
    float n;
} Log;

typedef struct State {
    int pos;
    int tick;
    float episode_return;
} State;

typedef struct Password {
    Log log;
    unsigned char* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    State state;
    unsigned char passcode[LENGTH];
    unsigned int rng;
} Password;

static const Color PASSWORD_BG = {6, 24, 24, 255};
static const Color PASSWORD_EMPTY = {40, 64, 64, 255};
static const Color PASSWORD_FILLED = {0, 187, 187, 255};
static const Color PASSWORD_WHITE = {241, 241, 241, 255};

static inline unsigned int password_lcg(unsigned int* rng) {
    *rng = 1664525u * (*rng) + 1013904223u;
    return *rng;
}

void update_observations(Password* env) {
    for (int i = 0; i < LENGTH; i++) {
        env->observations[i] = i < env->state.pos ? env->passcode[i] : 0;
    }
}

void refresh_state(Password* env) {
    if (env->state.pos < 0) {
        env->state.pos = 0;
    } else if (env->state.pos > LENGTH) {
        env->state.pos = LENGTH;
    }
    update_observations(env);
}

void init(Password* env) {
    unsigned int rng = PASSWORD_SEED;
    for (int i = 0; i < LENGTH; i++) {
        env->passcode[i] = (unsigned char)(1 + (password_lcg(&rng) % NUM_DIGITS));
    }
}

void add_log(Password* env) {
    float solved = (float)env->state.pos;
    env->log.perf += solved / (float)LENGTH;
    env->log.score += solved;
    env->log.episode_return += env->state.episode_return;
    env->log.episode_length += (float)env->state.tick;
    env->log.success += env->state.pos == LENGTH ? 1.0f : 0.0f;
    env->log.n += 1.0f;
}

void c_reset(Password* env) {
    env->state.pos = 0;
    env->state.tick = 0;
    env->state.episode_return = 0.0f;
    update_observations(env);
}

void c_step(Password* env) {
    int guess = (int)env->actions[0] + 1;
    int correct = env->state.pos < LENGTH && guess == env->passcode[env->state.pos];

    env->state.tick += 1;
    env->terminals[0] = 0.0f;

    if (correct) {
        env->state.pos += 1;
        env->rewards[0] = CORRECT_REWARD;
        update_observations(env);

        if (env->state.pos == LENGTH) {
            env->terminals[0] = 1.0f;
        }
    } else {
        env->rewards[0] = WRONG_REWARD;
        env->terminals[0] = 1.0f;
    }

    env->state.episode_return += env->rewards[0];
    if (env->terminals[0]) {
        add_log(env);
        c_reset(env);
    }
}

void c_render(Password* env) {
    const int px = 28;
    const int gap = 4;
    const int width = LENGTH * (px + gap) + gap;
    const int height = 120;

    if (!IsWindowReady()) {
        InitWindow(width, height, "PufferLib Password");
        SetTargetFPS(10);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(PASSWORD_BG);

    for (int i = 0; i < LENGTH; i++) {
        int x = gap + i * (px + gap);
        unsigned char obs = env->observations[i];
        Color color = obs == 0 ? PASSWORD_EMPTY : PASSWORD_FILLED;
        DrawRectangle(x, 32, px, px, color);
        if (obs != 0) {
            char text[4];
            snprintf(text, sizeof(text), "%d", obs);
            DrawText(text, x + 8, 36, 20, PASSWORD_WHITE);
        }
    }

    char status[64];
    snprintf(status, sizeof(status), "Progress: %d/%d", env->state.pos, LENGTH);
    DrawText(status, gap, 78, 20, PASSWORD_WHITE);

    EndDrawing();
}

void c_close(Password* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
