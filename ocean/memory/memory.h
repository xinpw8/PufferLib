#include <stdlib.h>
#include <string.h>
#include "raylib.h"
#if defined(from_float) && !defined(PRECISION_FLOAT)
typedef precision_t obs_t;
#else
typedef float obs_t;
#endif
#include "pufferenv.h"

#define ACT_SIZES {2}
#define OBS_SIZE 1
#define NUM_ATNS 1

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

struct Log {
    float perf;
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

static void memory_obs(Env* env, float v) {
    obs_t* obs = env->agents[0].observations;
#if defined(from_float) && !defined(PRECISION_FLOAT)
    obs[0] = from_float(v);
#else
    obs[0] = (obs_t)v;
#endif
}

void puf_reset(Env* env) {
    env->goal = (rand_r(&env->rng) % 2 == 0) ? -1 : 1;
    memory_obs(env, (float)env->goal);
    env->tick = 0;
}

// Hold Left Shift + A/D or arrows.
static void memory_human_controls(Env *env) {
    if (!IsWindowReady() || !IsKeyDown(KEY_LEFT_SHIFT)) {
        return;
    }
    if (IsKeyDown(KEY_A) || IsKeyDown(KEY_LEFT)) {
        env->agents[0].actions[0] = 0;
    } else if (IsKeyDown(KEY_D) || IsKeyDown(KEY_RIGHT)) {
        env->agents[0].actions[0] = 1;
    } else {
        env->agents[0].actions[0] = -1;
    }
}

void puf_step(Env* env) {
    memory_human_controls(env);
    env->agents[0].rewards[0] = 0;
    env->agents[0].terminals[0] = 0;
    memory_obs(env, 0.0f);
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
    env->log.perf += val;
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

    memory_human_controls(env);

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    DrawRectangle(0, 0, 480, 480, (env->goal == -1 ? PUFF_CYAN : PUFF_RED));
    DrawRectangle(480, 0, 480, 480, (env->agents[0].rewards[0] == 0 ? PUFF_RED : GREEN));
    DrawText(TextFormat("Tick %.0d. Simon says...", env->tick), 20, 20, 20, PUFF_WHITE);
    DrawText("[Shift] A/D or arrows", 20, 48, 16, PUFF_WHITE);
    EndDrawing();
    puf_web_vsync();
}

void puf_close(Env* env) {
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
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "n", log->n);
}
