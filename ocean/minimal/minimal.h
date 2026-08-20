// A sample multiagent coordination env. Star PufferLib on GitHub to support!
// Don't one-line structs/fns/ifs/vars in PRs. This fits in a screenshot.
#include <stdlib.h>
#include <math.h>
typedef float obs_t;
#include "pufferenv.h"

#define AGENTS 8
#define TARGETS 8
#define ACT_SIZES {9, 5}
#define OBS_SIZE (2 + 4*(AGENTS + TARGETS))
#define NUM_ATNS 2

typedef Env Minimal;

#ifdef PUFFERCPU_EVAL_MAIN
#define PUF_MINIMAL_NET 1
#include "minimal_net.h"
#endif

const int WIDTH = 1080, HEIGHT = 720, COOLDOWN = 30, TYPES = 4;
const float SPEED = 20.0f, MIN_TICKS = COOLDOWN*AGENTS/(float)TARGETS;
float clip(float val, float min, float max) { return fmaxf(fminf(val, max), min); }

// Required struct. Floats only, n last
struct Log { float perf, score, n; };
typedef struct { float x, y, heading, speed, type, ticks, cooldown; } Entity;
struct Env {
    Log log; int num_agents; unsigned int rng; // Required
    Agent agents[AGENTS]; int tag, boundary_reached; // Required
    Entity entities[AGENTS + TARGETS]; Texture2D sprites;
}; // Required: An env struct

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = AGENTS;
    for (int i=0; i<AGENTS; i++) {
        env->agents[i].policy = 0;
        env->agents[i].action_mask = NULL;
    }
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "n", log->n);
}

void compute_observations(Env* env) {
    for (int a=0; a<AGENTS ; a++) {
        int idx = 0; obs_t* obs = env->agents[a].observations;
        Entity* agent = &env->entities[a];
        obs[idx++] = agent->heading / (2*PI);
        obs[idx++] = agent->speed / SPEED;
        for (int o=0; o<AGENTS + TARGETS; o++) {
            Entity* other = &env->entities[o];
            obs[idx++] = (other->x - agent->x) / WIDTH;
            obs[idx++] = (other->y - agent->y) / HEIGHT;
            obs[idx++] = other->cooldown / COOLDOWN;
            obs[idx++] = other->type == agent->type ? 1 : 0;
        }
    }
}

void puf_reset(Env* env) {
    for (int i=0; i<AGENTS+TARGETS; i++) {
        Entity* entity = &env->entities[i];
        entity->x = 16 + rand_r(&env->rng)%(WIDTH-16);
        entity->y = 16 + rand_r(&env->rng)%(HEIGHT-16);
        entity->type = i % TYPES;
        entity->ticks = 0;
    }
    compute_observations(env);
}

void puf_step(Env* env) {
    for (int i=0; i<AGENTS; i++) {
        Entity* agent = &env->entities[i];
        float* actions = env->agents[i].actions;
        agent->ticks += 1;
        agent->heading += (actions[0] - 4.0f)/12.0f;
        if (agent->heading < -PI) agent->heading += 2*PI;
        if (agent->heading > PI) agent->heading -= 2*PI;
        float speed = agent->speed;
        agent->speed = clip(speed + (actions[1] - 2.0f), 0.0f, SPEED);
        agent->x = clip(agent->x + speed*cosf(agent->heading), 16, WIDTH-16);
        agent->y = clip(agent->y + speed*sinf(agent->heading), 16, HEIGHT-16);
        for (int t=0; t<TARGETS; t++) {
            Entity* target = &env->entities[AGENTS + t];
            if (target->cooldown > 0 || target->type != agent->type
                || fabsf(target->x - agent->x) > 32
                || fabsf(target->y - agent->y) > 32) continue;
            target->cooldown = COOLDOWN;
            if (rand_r(&env->rng) % 10 == 0) {
                target->x = 16 + rand_r(&env->rng)%(WIDTH-16);
                target->y = 16 + rand_r(&env->rng)%(HEIGHT-16);
            }
            env->agents[i].rewards[0] = 1.0f;
            env->log.perf += clip(MIN_TICKS/agent->ticks, 0.0f, 1.0f);
            env->log.score -= agent->ticks;
            env->log.n++;
            agent->type = ((int)agent->type + 1) % TYPES;
            agent->ticks = 0;
            break;
        }
    }
    for (int t=0; t<TARGETS; t++) {
        Entity* target = &env->entities[AGENTS + t];
        target->cooldown = fmaxf(target->cooldown - 1, 0);
    }
    compute_observations(env);
}

void puf_render(Env* env) {
    if (!IsWindowReady()) {
        InitWindow(WIDTH, HEIGHT, "PufferLib Env"); SetTargetFPS(30);
        env->sprites = LoadTexture("resources/shared/puffers.png");
    }
    if (IsKeyDown(KEY_ESCAPE)) exit(0);
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    for (int i=0; i<AGENTS+TARGETS; i++) {
        Entity* entity = &env->entities[i];
        int sz = i < AGENTS ? 32 : 64, y = i < AGENTS ? 576 : 512;
        if (i < AGENTS && (entity->heading < -PI/2 || entity->heading > PI/2)) y += 32;
        DrawTexturePro(env->sprites,
            (Rectangle){sz*entity->type, (float)y, (float)sz, (float)sz},
            (Rectangle){entity->x - sz/2, entity->y - sz/2, (float)sz, (float)sz},
            (Vector2){0, 0}, 0, entity->cooldown > 0 ? DARKGRAY: WHITE
        );
    }
    EndDrawing();
    puf_web_vsync();
}

void puf_close(Env* env) {
    if (IsWindowReady()) {
        UnloadTexture(env->sprites);
        CloseWindow();
    }
}
