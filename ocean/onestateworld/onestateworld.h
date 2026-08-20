#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "raylib.h"
typedef unsigned char obs_t;
#include "pufferenv.h"

#define ACT_SIZES {2}
#define OBS_SIZE 1
#define NUM_ATNS 1

// Marsaglia polar method from https://en.wikipedia.org/wiki/Marsaglia_polar_method
static double gaussian_sample(double mean, double variance) {
    static int hasSpare = 0;
    static double spare;

    if (hasSpare) {
        hasSpare = 0;
        return mean + sqrt(variance) * spare;
    }

    hasSpare = 1;
    double u, v, s;
    do {
        u = (rand() / ((double)RAND_MAX)) * 2.0 - 1.0;
        v = (rand() / ((double)RAND_MAX)) * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1 || s == 0);

    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return mean + sqrt(variance) * (u * s);
}

const unsigned char LEFT = 0;
const unsigned char RIGHT = 1;

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
    int tick;
    float var_right;
    float mean_right;
    float mean_left;
    Texture2D puffer;
    unsigned int rng;
};
typedef Env World;

void add_log(World* env) {
    env->log.perf += env->agents[0].rewards[0];
    env->log.score += env->agents[0].rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->agents[0].rewards[0];
    env->log.n++;
}

void puf_reset(World* env) {
    obs_t* obs = env->agents[0].observations;
    obs[0] = 0;
    env->tick = 0;
}

// Hold Left Shift + A/D or arrows.
static void onestateworld_human_controls(World *env) {
    if (!IsWindowReady() || !IsKeyDown(KEY_LEFT_SHIFT)) {
        return;
    }
    env->agents[0].actions[0] = 0;
    if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
        env->agents[0].actions[0] = LEFT;
    }
    if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
        env->agents[0].actions[0] = RIGHT;
    }
}

void puf_step(World* env) {
    env->tick += 1;
    env->agents[0].terminals[0] = 0;
    env->agents[0].rewards[0] = 0;

    int action = (int)env->agents[0].actions[0];

    if (action == LEFT) {
        env->agents[0].rewards[0] = (float)tanh(gaussian_sample(env->mean_left, 0));
    } else {
        env->agents[0].rewards[0] = (float)tanh(gaussian_sample(env->mean_right, env->var_right));
    }

    if (env->tick >= 1000) {
        env->agents[0].terminals[0] = 1;
        add_log(env);
        puf_reset(env);
    }
}

void puf_render(World* env) {
    int px = 64;

    if (!IsWindowReady()) {
        InitWindow(px * 5, px * 5, "PufferLib OneStateWorld");
        SetTargetFPS(1);
        env->puffer = LoadTexture("resources/shared/puffers_128.png");
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    onestateworld_human_controls(env);

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    Color color = (Color){255, 255, 255, 255};
    Rectangle source_rect = (Rectangle){0, 0, 128, 128};
    Rectangle dest_rect = (Rectangle){(float)(2 * px), (float)(2 * px), (float)px, (float)px};
    DrawTexturePro(env->puffer, source_rect, dest_rect, (Vector2){0, 0}, 0, color);

    char score_text[32];
    snprintf(score_text, sizeof(score_text), "R: %.4f", env->agents[0].rewards[0]);
    if ((int)env->agents[0].actions[0] == LEFT) {
        DrawText(score_text, 0, (int)(2.5 * px), 28, (Color){255, 255, 255, 255});
    } else {
        DrawText(score_text, 3 * px, (int)(2.5 * px), 28, (Color){255, 255, 255, 255});
    }

    EndDrawing();
    puf_web_vsync();
}

void puf_close(World* env) {
    if (IsWindowReady()) {
        UnloadTexture(env->puffer);
        CloseWindow();
    }
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->var_right = dict_get(kwargs, "var_right");
    env->mean_right = dict_get(kwargs, "mean_right");
    env->mean_left = dict_get(kwargs, "mean_left");
    env->agents[0].policy = 0;
    env->agents[0].action_mask = NULL;
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n", log->n);
}

