/* Squared Continuous: continuous action version of squared.
 * 2 continuous action dimensions: vertical and horizontal.
 * Actions are clamped to [-1, 1] and thresholded at 0.25 magnitude.
 */

#include <stdlib.h>
#include <string.h>
#include "raylib.h"
typedef unsigned char obs_t;
#include "pufferenv.h"

#define ACT_SIZES {1, 1}
#define OBS_SIZE 121
#define NUM_ATNS 2
#define SQUARED_CONTINUOUS_FRAMES 12

const unsigned char EMPTY = 0;
const unsigned char AGENT = 1;
const unsigned char TARGET = 2;

// Required struct. Only use floats!
struct Log {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported in binding.c
    float n; // Required as the last field
};

// Required that you have some struct for your env
struct Env {
    Log log; // Required field. Env binding code uses this to aggregate logs
    Agent agents[1];
    int tag;
    int boundary_reached;
    int num_agents;
    int size;
    int tick;
    int r;
    int c;
    unsigned int rng;
};
typedef Env Squared;

void add_log(Squared* env) {
    env->log.perf += (env->agents[0].rewards[0] > 0) ? 1 : 0;
    env->log.score += env->agents[0].rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->agents[0].rewards[0];
    env->log.n++;
}

// Required function
void puf_reset(Squared* env) {
    obs_t* obs = env->agents[0].observations;
    int tiles = env->size*env->size;
    memset(obs, 0, tiles*sizeof(unsigned char));
    obs[tiles/2] = AGENT;
    env->r = env->size/2;
    env->c = env->size/2;
    env->tick = 0;
    int target_idx = 0; // Deterministic for testing
    do {
        target_idx = rand_r(&env->rng) % tiles;
    } while (target_idx == tiles/2);
    obs[target_idx] = TARGET;
}

// Clamp value to [-1, 1]
static inline float clamp_action(float x) {
    return x < -1.0f ? -1.0f : (x > 1.0f ? 1.0f : x);
}

// Hold Left Shift + WASD/arrows.
static void squared_continuous_human_controls(Squared *env) {
    if (!IsWindowReady() || !IsKeyDown(KEY_LEFT_SHIFT)) {
        return;
    }
    env->agents[0].actions[0] = 0.0f;
    env->agents[0].actions[1] = 0.0f;
    if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) {
        env->agents[0].actions[0] = -1.0f;
    }
    if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) {
        env->agents[0].actions[0] = 1.0f;
    }
    if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
        env->agents[0].actions[1] = -1.0f;
    }
    if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
        env->agents[0].actions[1] = 1.0f;
    }
}

// Required function
void puf_step(Squared* env) {
    obs_t* obs = env->agents[0].observations;
    env->tick += 1;

    // Continuous actions: clamp to [-1, 1] then threshold to discrete move
    // action[0]: vertical (positive = down, negative = up)
    // action[1]: horizontal (positive = right, negative = left)
    float vert = clamp_action(env->agents[0].actions[0]);
    float horiz = clamp_action(env->agents[0].actions[1]);
    env->agents[0].terminals[0] = 0;
    env->agents[0].rewards[0] = 0;

    obs[env->r*env->size + env->c] = EMPTY;

    // Threshold at 0.25 magnitude to determine direction (allows stationary)
    if (vert > 0.25) {
        env->r += 1;  // DOWN
    } else if (vert < -0.25) {
        env->r -= 1;  // UP
    }
    if (horiz > 0.25) {
        env->c += 1;  // RIGHT
    } else if (horiz < -0.25) {
        env->c -= 1;  // LEFT
    }

    if (env->tick > 3*env->size
            || env->r < 0
            || env->c < 0
            || env->r >= env->size
            || env->c >= env->size) {
        env->agents[0].terminals[0] = 1;
        env->agents[0].rewards[0] = -1.0;
        add_log(env);
        puf_reset(env);
        return;
    }

    int pos = env->r*env->size + env->c;
    if (obs[pos] == TARGET) {
        env->agents[0].terminals[0] = 1;
        env->agents[0].rewards[0] = 1.0;
        add_log(env);
        puf_reset(env);
        return;
    }

    obs[pos] = AGENT;
}

// Required function. Should handle creating the client on first call
void puf_render(Squared* env) {
    obs_t* obs = env->agents[0].observations;
    if (!IsWindowReady()) {
        InitWindow(64*env->size, 64*env->size, "PufferLib Squared Continuous");
        SetTargetFPS(60);
    }

    int px = 64;
    for (int f = 0; f < SQUARED_CONTINUOUS_FRAMES; f++) {
        if (IsKeyDown(KEY_ESCAPE)) {
            exit(0);
        }
        squared_continuous_human_controls(env);
        BeginDrawing();
        ClearBackground((Color){6, 24, 24, 255});
        for (int i = 0; i < env->size; i++) {
            for (int j = 0; j < env->size; j++) {
                int tex = obs[i * env->size + j];
                if (tex == EMPTY) {
                    continue;
                }
                Color color = (tex == AGENT)
                    ? (Color){0, 187, 187, 255}
                    : (Color){187, 0, 0, 255};
                DrawRectangle(j * px, i * px, px, px, color);
            }
        }
        EndDrawing();
        puf_web_vsync();
    }
}

// Required function. Should clean up anything you allocated
// Do not free observations, actions, rewards, terminals
void puf_close(Squared* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}

// --- Native trainer (pufferl) API ---
void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n", log->n);
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->size = dict_get(kwargs, "size");
    env->agents[0].action_mask = NULL;
    env->agents[0].policy = 0;
}
