/* Convert: a sample multiagent env about puffers eating stars.
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
#define OBS_SIZE 28
#define NUM_ATNS 2
#define MAX_AGENTS 1024

typedef Env Convert;

struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
};

typedef struct {
    Texture2D sprites;
} Client;

// Game entity (was Agent; renamed to avoid conflict with pufferenv Agent)
typedef struct {
    float x;
    float y;
    float heading;
    float speed;
    int item;
    int episode_length;
} Entity;

typedef struct {
    float x;
    float y;
    float heading;
    int item;
} Factory;

struct Env {
    Log log;
    Client* client;
    Agent agents[MAX_AGENTS];
    Entity* entities;
    Factory* factories;
    int num_agents;
    int tag;
    int boundary_reached;
    int width;
    int height;
    int num_factories;
    int num_resources;
    unsigned int rng;
};

void init(Convert* env) {
    env->entities = (Entity*)calloc(env->num_agents, sizeof(Entity));
    env->factories = (Factory*)calloc(env->num_factories, sizeof(Factory));
}

void compute_observations(Convert* env) {
    for (int a = 0; a < env->num_agents; a++) {
        Entity* agent = &env->entities[a];
        float* obs = env->agents[a].observations;
        int obs_idx = 0;
        float dists[env->num_resources];
        for (int i = 0; i < env->num_resources; i++) {
            dists[i] = 999999;
        }
        for (int f = 0; f < env->num_factories; f++) {
            Factory* factory = &env->factories[f];
            float dx = factory->x - agent->x;
            float dy = factory->y - agent->y;
            float dd = dx * dx + dy * dy;
            int type = f % env->num_resources;
            if (dd < dists[type]) {
                dists[type] = dd;
                obs[obs_idx + 2 * type] = dx / env->width;
                obs[obs_idx + 2 * type + 1] = dy / env->height;
            }
        }
        obs_idx += 2 * env->num_resources;
        obs[obs_idx++] = agent->heading / (2 * PI);
        obs[obs_idx++] = env->agents[a].rewards[0];
        obs[obs_idx++] = agent->x / env->width;
        obs[obs_idx++] = agent->y / env->height;
        memset(&obs[obs_idx], 0, env->num_resources * sizeof(float));
        obs[obs_idx + agent->item] = 1.0f;
    }
}

void puf_reset(Convert* env) {
    for (int i = 0; i < env->num_agents; i++) {
        env->entities[i].x = 16 + rand_r(&env->rng) % (env->width - 16);
        env->entities[i].y = 16 + rand_r(&env->rng) % (env->height - 16);
        env->entities[i].item = rand_r(&env->rng) % env->num_resources;
        env->entities[i].episode_length = 0;
    }
    for (int i = 0; i < env->num_factories; i++) {
        env->factories[i].x = 16 + rand_r(&env->rng) % (env->width - 16);
        env->factories[i].y = 16 + rand_r(&env->rng) % (env->height - 16);
        env->factories[i].item = i % env->num_resources;
        env->factories[i].heading = (rand_r(&env->rng) % 360) * PI / 180.0f;
    }
    compute_observations(env);
}

float clip(float val, float min, float max) {
    if (val < min) {
        return min;
    } else if (val > max) {
        return max;
    }
    return val;
}

void puf_step(Convert* env) {
    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].terminals[0] = 0;
        env->agents[i].rewards[0] = 0;
        Entity* agent = &env->entities[i];
        float* actions = env->agents[i].actions;
        agent->episode_length += 1;

        agent->heading += (actions[0] - 4.0f) / 12.0f;
        while (agent->heading < 0) {
            agent->heading += 2 * PI;
        }
        while (agent->heading >= 2 * PI) {
            agent->heading -= 2 * PI;
        }

        agent->speed += 1.0f * (actions[1] - 2.0f);
        agent->speed = clip(agent->speed, -20.0f, 20.0f);

        agent->x += agent->speed * cosf(agent->heading);
        agent->x = clip(agent->x, 16, env->width - 16);

        agent->y += agent->speed * sinf(agent->heading);
        agent->y = clip(agent->y, 16, env->height - 16);

        // Fixed 1/1024 shuffle so web (512 agents) matches train rate.
        if (rand_r(&env->rng) % 1024 == 0) {
            env->entities[i].x = rand_r(&env->rng) % env->width;
            env->entities[i].y = rand_r(&env->rng) % env->height;
        }

        for (int f = 0; f < env->num_factories; f++) {
            Factory* factory = &env->factories[f];
            float dx = (factory->x - agent->x);
            float dy = (factory->y - agent->y);
            float dist = sqrtf(dx * dx + dy * dy);
            if (dist > 32) {
                continue;
            }
            if (factory->item == agent->item) {
                agent->item = (agent->item + 1) % env->num_resources;
                float elen = (float)agent->episode_length;
                if (elen < 1.0f) {
                    elen = 1.0f;
                }
                // 80-step convert is "solved" (straight-line ~20-40 on this map).
                env->log.perf += fminf(1.0f, 80.0f / elen);
                env->log.score += 1000.0f / elen;
                env->log.episode_length += elen;
                env->log.n++;
                env->agents[i].rewards[0] = 1.0f;
                agent->episode_length = 0;
            }
        }
    }
    for (int f = 0; f < env->num_factories; f++) {
        Factory* factory = &env->factories[f];
        factory->x += 2.0f * cosf(factory->heading);
        factory->y += 2.0f * sinf(factory->heading);

        float factory_x = clip(factory->x, 16, env->width - 16);
        float factory_y = clip(factory->y, 16, env->height - 16);

        if (factory_x != factory->x || factory_y != factory->y) {
            factory->heading = (rand_r(&env->rng) % 360) * PI / 180.0f;
            factory->x = factory_x;
            factory->y = factory_y;
        }
    }
    compute_observations(env);
}

void puf_render(Convert* env) {
    if (env->client == NULL) {
        InitWindow(env->width, env->height, "PufferLib Convert");
        SetTargetFPS(30);
        env->client = (Client*)calloc(1, sizeof(Client));
        env->client->sprites = LoadTexture("resources/shared/puffers.png");
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    for (int f = 0; f < env->num_factories; f++) {
        Factory* factory = &env->factories[f];
        DrawTexturePro(
            env->client->sprites,
            (Rectangle){
                64 * factory->item, 512, 64, 64,
            },
            (Rectangle){
                factory->x - 32,
                factory->y - 32,
                64,
                64
            },
            (Vector2){0, 0},
            0,
            WHITE
        );
    }

    for (int i = 0; i < env->num_agents; i++) {
        Entity* agent = &env->entities[i];
        float heading = agent->heading;
        int y = 576;
        if (heading < PI / 2 || heading > 3 * PI / 2) {
            y += 32;
        }
        DrawTexturePro(
            env->client->sprites,
            (Rectangle){
                32 * agent->item, y, 32, 32,
            },
            (Rectangle){
                agent->x - 16,
                agent->y - 16,
                32,
                32
            },
            (Vector2){0, 0},
            0,
            WHITE
        );
    }

    EndDrawing();
    puf_web_vsync();
}

void puf_close(Convert* env) {
    free(env->entities);
    free(env->factories);
    if (env->client != NULL) {
        Client* client = env->client;
        UnloadTexture(client->sprites);
        CloseWindow();
        free(client);
    }
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = dict_get(kwargs, "num_agents");
    env->width = dict_get(kwargs, "width");
    env->height = dict_get(kwargs, "height");
    env->num_factories = dict_get(kwargs, "num_factories");
    env->num_resources = dict_get(kwargs, "num_resources");
    if (env->num_agents > MAX_AGENTS) {
        fprintf(stderr, "convert: num_agents %d > MAX_AGENTS %d\n",
            env->num_agents, MAX_AGENTS);
        exit(1);
    }
    if (3 * env->num_resources + 4 != OBS_SIZE) {
        fprintf(stderr,
            "convert: num_resources=%d implies obs size %d, but OBS_SIZE=%d\n",
            env->num_resources, 3 * env->num_resources + 4, OBS_SIZE);
        exit(1);
    }
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

