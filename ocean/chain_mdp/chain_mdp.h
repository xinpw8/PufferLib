#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "raylib.h"
typedef unsigned char obs_t;
#include "pufferenv.h"

#define ACT_SIZES {2}
#define OBS_SIZE 1
#define NUM_ATNS 1

const unsigned char LEFT = 0;
const unsigned char RIGHT = 1;

const float STATE1_REWARD = 0.001;
const float STATEN_REWARD = 1.0;

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
    Texture2D puffer;
    unsigned int rng;
};
typedef Env Chain;

void add_log(Chain* env) {
    env->log.perf += (env->agents[0].rewards[0] == STATEN_REWARD) ? 1 : 0;
    env->log.score += env->agents[0].rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->agents[0].rewards[0];
    env->log.n++;
}

void puf_reset(Chain* env) {
    unsigned char* obs = env->agents[0].observations;
    obs[0] = 1;
    env->state = 1;
    env->tick = 0;
}

// Hold Left Shift + A/D or arrows.
static void chain_mdp_human_controls(Chain *env) {
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

void puf_step(Chain* env) {
    unsigned char* obs = env->agents[0].observations;
    chain_mdp_human_controls(env);
    env->tick += 1;
    env->agents[0].terminals[0] = 0;
    env->agents[0].rewards[0] = 0;

    int action = (int)env->agents[0].actions[0];
    action = action * 2 - 1; // Map 0,1 to -1,1
    env->state = MIN(MAX(env->state + action, 0), env->size - 1);
    obs[0] = env->state;

    if (env->state == 0) {
        env->agents[0].rewards[0] = STATE1_REWARD;
    } else if (env->state == env->size - 1) {
        env->agents[0].rewards[0] = STATEN_REWARD;
    }

    if (env->tick == env->size + 9) {
        env->agents[0].terminals[0] = 1;
        add_log(env);
        puf_reset(env);
        return;
    }
}

void puf_render(Chain* env) {
    unsigned char* obs = env->agents[0].observations;
    int px = MAX(8, 1024.0 / env->size);

    if (!IsWindowReady()) {
        InitWindow(px * env->size, px * 5, "PufferLib Chain MDP");
        SetTargetFPS(4);
        env->puffer = LoadTexture("resources/shared/puffers_128.png");
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    chain_mdp_human_controls(env);

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    int agent_pos = obs[0];
    for (int i = 0; i < env->size; i++) {
        Color color =
            (i == agent_pos) ? (Color){0, 255, 255, 255} :
            (i == 0) ? (Color){204, 204, 0, 255} :
            (i == env->size - 1) ? (Color){255, 255, 51, 255} :
            (Color){224, 224, 224, 255};

        if (i == agent_pos) {
            int starting_sprite_x = 0;
            if (env->agents[0].actions[0] == 0) {
                starting_sprite_x = 128;
            }
            Rectangle source_rect = (Rectangle){(float)starting_sprite_x, 0, 128, 128};
            Rectangle dest_rect = (Rectangle){(float)(i * px), (float)(2 * px), (float)px, (float)px};
            DrawTexturePro(env->puffer, source_rect, dest_rect, (Vector2){0, 0}, 0, color);
        } else {
            DrawRectangle(i * px, 2 * px, px, px, color);
        }
    }

    EndDrawing();
    puf_web_vsync();
}

void puf_close(Chain* env) {
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
    dict_set(out, "n", log->n);
}

