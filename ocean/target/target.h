/* Target: a sample multiagent env about puffers eating stars.
 * Use this as a tutorial and template for your own multiagent envs.
 * We suggest starting with the Squared env for a simpler intro.
 * Star PufferLib on GitHub to support. It really, really helps!
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "raylib.h"
typedef float obs_t;
#include "pufferenv.h"

#define ACT_SIZES {9, 5}
#define OBS_SIZE 28  // num_goals*2 + num_agents*2 + 4 = 4*2 + 8*2 + 4
#define NUM_ATNS 2
#define MAX_AGENTS 8
#define MAX_GOALS 4

typedef Env Target;

// Required struct. Only use floats!
struct Log {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    float n; // Required as the last field
};

typedef struct {
    Texture2D puffer;
    Texture2D star;
} Client;

// Game entity (was Agent)
typedef struct {
    float x;
    float y;
    float heading;
    float speed;
    int ticks_since_reward;
} Entity;

typedef struct {
    float x;
    float y;
} Goal;

struct Env {
    Log log;
    Client* client;
    Agent agents[MAX_AGENTS];
    Entity* entities;
    Goal* goals;
    int width;
    int height;
    int num_agents;
    int num_goals;
    int tag;
    int boundary_reached;
    unsigned int rng;
};

void init(Target* env) {
    env->entities = (Entity*)calloc(env->num_agents, sizeof(Entity));
    env->goals = (Goal*)calloc(env->num_goals, sizeof(Goal));
}

void update_goals(Target* env) {
    for (int a = 0; a < env->num_agents; a++) {
        Entity* agent = &env->entities[a];
        for (int g = 0; g < env->num_goals; g++) {
            Goal* goal = &env->goals[g];
            float dx = goal->x - agent->x;
            float dy = goal->y - agent->y;
            float dist = sqrtf(dx * dx + dy * dy);
            if (dist > 64) {
                continue;
            }
            goal->x = rand_r(&env->rng) % env->width;
            goal->y = rand_r(&env->rng) % env->height;
            env->agents[a].rewards[0] = 1.0f;
            env->log.score += 1.0f;
            env->log.episode_length += agent->ticks_since_reward;
            env->log.perf += fmaxf(0.0f, 1.0f - 0.01f * agent->ticks_since_reward);
            agent->ticks_since_reward = 0;
            env->log.episode_return += 1.0f;
            env->log.n++;
        }
    }
}

void compute_observations(Target* env) {
    for (int a = 0; a < env->num_agents; a++) {
        Entity* agent = &env->entities[a];
        obs_t* obs = env->agents[a].observations;
        int obs_idx = 0;
        for (int g = 0; g < env->num_goals; g++) {
            Goal* goal = &env->goals[g];
            obs[obs_idx++] = (goal->x - agent->x) / env->width;
            obs[obs_idx++] = (goal->y - agent->y) / env->height;
        }
        for (int b = 0; b < env->num_agents; b++) {
            Entity* other = &env->entities[b];
            obs[obs_idx++] = (other->x - agent->x) / env->width;
            obs[obs_idx++] = (other->y - agent->y) / env->height;
        }
        obs[obs_idx++] = agent->heading / (2 * PI);
        obs[obs_idx++] = env->agents[a].rewards[0];
        obs[obs_idx++] = agent->x / env->width;
        obs[obs_idx++] = agent->y / env->height;
    }
}

void puf_reset(Target* env) {
    for (int i = 0; i < env->num_agents; i++) {
        env->entities[i].x = rand_r(&env->rng) % env->width;
        env->entities[i].y = rand_r(&env->rng) % env->height;
        env->entities[i].ticks_since_reward = 0;
    }
    for (int i = 0; i < env->num_goals; i++) {
        env->goals[i].x = rand_r(&env->rng) % env->width;
        env->goals[i].y = rand_r(&env->rng) % env->height;
    }
    compute_observations(env);
}

static inline float clipf(float val, float min, float max) {
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

void puf_step(Target* env) {
    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].rewards[0] = 0;
        Entity* agent = &env->entities[i];
        float* actions = env->agents[i].actions;
        agent->ticks_since_reward += 1;

        agent->heading += (actions[0] - 4.0f) / 12.0f;
        agent->heading = clipf(agent->heading, 0, 2 * PI);

        agent->speed += actions[1] - 2.0f;
        agent->speed = clipf(agent->speed, -20.0f, 20.0f);

        agent->x += agent->speed * cosf(agent->heading);
        agent->x = clipf(agent->x, 0, env->width);

        agent->y += agent->speed * sinf(agent->heading);
        agent->y = clipf(agent->y, 0, env->height);

        if (agent->ticks_since_reward % 512 == 0) {
            agent->x = rand_r(&env->rng) % env->width;
            agent->y = rand_r(&env->rng) % env->height;
        }
    }
    update_goals(env);
    compute_observations(env);
}

// Hold Left Shift + WASD/arrows for agent 0 turn/accel.
static void target_human_controls(Target* env) {
    if (!IsWindowReady() || !IsKeyDown(KEY_LEFT_SHIFT)) {
        return;
    }
    env->agents[0].actions[0] = 4.0f;
    env->agents[0].actions[1] = 2.0f;
    if (IsKeyDown(KEY_A) || IsKeyDown(KEY_LEFT)) {
        env->agents[0].actions[0] = 0.0f;
    }
    if (IsKeyDown(KEY_D) || IsKeyDown(KEY_RIGHT)) {
        env->agents[0].actions[0] = 8.0f;
    }
    if (IsKeyDown(KEY_S) || IsKeyDown(KEY_DOWN)) {
        env->agents[0].actions[1] = 0.0f;
    }
    if (IsKeyDown(KEY_W) || IsKeyDown(KEY_UP)) {
        env->agents[0].actions[1] = 4.0f;
    }
}

void puf_render(Target* env) {
    if (env->client == NULL) {
        InitWindow(1080, 720, "PufferLib Target");
        SetTargetFPS(60);
        env->client = (Client*)calloc(1, sizeof(Client));
        env->client->puffer = LoadTexture("resources/shared/puffers_128.png");
        env->client->star = LoadTexture("resources/shared/star.png");
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    target_human_controls(env);

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    for (int i = 0; i < env->num_goals; i++) {
        Goal* goal = &env->goals[i];
        DrawTexture(env->client->star, (int)goal->x, (int)goal->y, WHITE);
    }

    for (int i = 0; i < env->num_agents; i++) {
        Entity* agent = &env->entities[i];
        float heading = agent->heading;
        DrawTexturePro(
            env->client->puffer,
            (Rectangle){
                (heading < PI / 2 || heading > 3 * PI / 2) ? 128 : 0,
                0, 128, 128,
            },
            (Rectangle){
                agent->x,
                agent->y,
                128,
                128
            },
            (Vector2){0, 0},
            0,
            WHITE
        );
    }

    EndDrawing();
    puf_web_vsync();
}

void puf_close(Target* env) {
    free(env->entities);
    free(env->goals);
    if (env->client != NULL) {
        Client* client = env->client;
        UnloadTexture(client->puffer);
        UnloadTexture(client->star);
        CloseWindow();
        free(client);
    }
}

void puf_init(Env* env, Dict* kwargs) {
    env->width = 952;
    env->height = 592;
    env->num_agents = MAX_AGENTS;
    env->num_goals = MAX_GOALS;
    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].policy = 0;
        env->agents[i].action_mask = NULL;
    }
    init(env);
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n", log->n);
}

