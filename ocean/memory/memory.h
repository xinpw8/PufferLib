#pragma once
#include <stdlib.h>
#include <string.h>
#include "raylib.h"
#include "pufferenv.h"

#define ACT_SIZES {2}
#define OBS_SIZE 1
#define NUM_ATNS 1

#if defined(from_float) && !defined(PRECISION_FLOAT)
typedef precision_t obs_t;
#else
typedef float obs_t;
#endif

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

struct Log {
    float score;
    float n;
};

struct Env {
    Log log;
    Agent agents[1];
    int tag;
    int boundary_reached;
    int num_agents;
    int length;
    int goal;
    int tick;
    unsigned int rng;
};
// Avoid alias "Memory" — conflicts with CUDA/system types when included via pufferl.cu.

void puf_reset(Env* env) {
    env->goal = (rand_r(&env->rng) % 2 == 0) ? -1 : 1;
    ((obs_t*)env->agents[0].observations)[0] = (obs_t)env->goal;
    env->tick = 0;
}

void puf_step(Env* env) {
    env->agents[0].rewards[0] = 0;
    env->agents[0].terminals[0] = 0;
    ((obs_t*)env->agents[0].observations)[0] = 0;
    env->tick++;

    if (env->tick < env->length) {
        return;
    }

    float val = 0.0f;
    if ((int)env->agents[0].actions[0] == 0 && env->goal == -1) {
        val = 1.0f;
    }
    if ((int)env->agents[0].actions[0] == 1 && env->goal == 1) {
        val = 1.0f;
    }

    puf_reset(env);
    env->agents[0].rewards[0] = val;
    env->agents[0].terminals[0] = 1;
    env->log.score += val;
    env->log.n += 1;
}

void puf_render(Env* env) {
    if (!IsWindowReady()) {
        InitWindow(960, 480, "PufferLib Memory");
        SetTargetFPS(5);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    DrawRectangle(0, 0, 480, 480, (env->goal == -1 ? PUFF_CYAN : PUFF_RED));
    DrawRectangle(480, 0, 480, 480, (env->agents[0].rewards[0] == 0 ? PUFF_RED : GREEN));
    DrawText(TextFormat("Tick %.0d. Simon says...", env->tick), 20, 20, 20, PUFF_WHITE);
    EndDrawing();
}

void puf_close(Env* env) {
    (void)env;
    if (IsWindowReady()) {
        CloseWindow();
    }
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->length = dict_get(kwargs, "length");
    env->agents[0].policy = 0;
    env->agents[0].action_mask = NULL;
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
}

