// Originally made by Sam Turner and Finlay Sanders, 2025.
// Included in pufferlib under the original project's MIT license.
// https://github.com/tensaur/drone

#pragma once

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "dronelib.h"
#include "pufferenv.h"

#define HORIZON 1024
#define ACT_SIZES {1, 1, 1, 1}
#define OBS_SIZE DRONE_OBS_SIZE
#define NUM_ATNS 4
#define MAX_DRONES 128

typedef Env DroneEnv;
#if defined(from_float) && !defined(PRECISION_FLOAT)
typedef precision_t obs_t;
#else
typedef float obs_t;
#endif

typedef struct {
    float dist;
    float prev_dist;
    float vel;
    float omega;
} StepCache;

#define MAX_TASK_LOG_ENTRIES 16

typedef struct Log Log;
struct Log {
    float score;
    float perf;
    float episode_return;
    float episode_length;
    float task[MAX_TASK_LOG_ENTRIES];
    float n;
};

static inline void log_task_add(Log* log, int idx, float value) {
    if (idx < 0 || idx >= MAX_TASK_LOG_ENTRIES) return;
    log->task[idx] += value;
}

typedef struct Client Client;

typedef struct {
    const char* name;
    const char* log_keys[MAX_TASK_LOG_ENTRIES];
    int num_log_keys;

    void (*init)(DroneEnv* env);
    void (*close)(DroneEnv* env);
    void (*env_reset)(DroneEnv* env);
    void (*reset)(DroneEnv* env, Drone* agent, int idx);
    float (*reward)(DroneEnv* env, Drone* agent, int idx, StepCache* cache);
    bool (*done)(DroneEnv* env, Drone* agent, int idx, StepCache* cache);

    void (*log)(DroneEnv* env, Drone* agent, int idx, Log* log, StepCache* cache);
    void (*render)(DroneEnv* env, Client* client);
} Task;

struct Env {
    Agent agents[MAX_DRONES];
    int num_agents;
    int tag;
    int boundary_reached;
    unsigned int rng;

    int tick;
    Drone* drones;
    Log log;

    const Task* task;
    void* task_config;
    void* task_state;

    Client* client;
};

void compute_observations(DroneEnv* env) {
    for (int i = 0; i < env->num_agents; i++)
        compute_drone_observations(&env->drones[i], (float*)env->agents[i].observations);
}

void reset_agent_base(Drone* agent, unsigned int* rng) {
    Target* target = agent->target;
    memset(agent, 0, sizeof(Drone));
    agent->target = target;
    init_drone(agent, rng, 0.05f);
}

void init(DroneEnv* env) {
    env->drones = (Drone*)calloc(env->num_agents, sizeof(Drone));
    for (int i = 0; i < env->num_agents; i++)
        env->drones[i].target = (Target*)calloc(1, sizeof(Target));
    env->log = (Log){0};
    env->tick = 0;
}

void add_log(DroneEnv* env, int idx, StepCache* cache) {
    Drone* agent = &env->drones[idx];
    env->log.episode_return += agent->episode_return;
    env->log.episode_length += agent->episode_length;
    env->log.n += 1.0f;

    if (env->task->log) env->task->log(env, agent, idx, &env->log, cache);
}

void puf_reset(DroneEnv* env) {
    if (env->task->env_reset) env->task->env_reset(env);

    for (int i = 0; i < env->num_agents; i++) {
        Drone* agent = &env->drones[i];
        reset_agent_base(agent, &env->rng);
        env->task->reset(env, agent, i);
        agent->prev_pos = agent->state.pos;
    }

    compute_observations(env);
}

void puf_step(DroneEnv* env) {
    env->tick = (env->tick + 1) % HORIZON;

    for (int i = 0; i < env->num_agents; i++) {
        Drone* agent = &env->drones[i];

        agent->prev_pos = agent->state.pos;
        move_drone(agent, env->agents[i].actions);
        agent->episode_length++;

        StepCache cache;
        cache.prev_dist = norm3(sub3(agent->target->pos, agent->prev_pos));
        cache.dist = norm3(sub3(agent->target->pos, agent->state.pos));
        cache.vel = norm3(agent->state.vel);
        cache.omega = norm3(agent->state.omega);

        float reward = env->task->reward(env, agent, i, &cache);
        bool done = env->task->done(env, agent, i, &cache);

        agent->episode_return += reward;
        env->agents[i].rewards[0] = reward;
        env->agents[i].terminals[0] = done ? 1.0f : 0.0f;

        if (done) {
            add_log(env, i, &cache);
            reset_agent_base(agent, &env->rng);
            env->task->reset(env, agent, i);
            agent->prev_pos = agent->state.pos;
        }
    }

    compute_observations(env);
}

void c_close_client(Client* client);

void puf_close(DroneEnv* env) {
    if (env->task != NULL && env->task->close != NULL) env->task->close(env);

    for (int i = 0; i < env->num_agents; i++)
        free(env->drones[i].target);
    free(env->drones);

    if (env->client != NULL) c_close_client(env->client);
}
#include "render.h"
#include "task_hover.h"
#include "task_race.h"

static const Task* DRONE_LOG_TASK = NULL;

static void drone_hover_config(DroneEnv* env, Dict* kwargs) {
    HoverConfig* cfg = (HoverConfig*)calloc(1, sizeof(HoverConfig));
    cfg->target_dist = dict_get(kwargs, "hover_target_dist");
    cfg->hover_dist = dict_get(kwargs, "hover_dist");
    cfg->hover_omega = dict_get(kwargs, "hover_omega");
    cfg->hover_vel = dict_get(kwargs, "hover_vel");
    cfg->alpha_dist = dict_get(kwargs, "alpha_dist");
    cfg->alpha_hover = dict_get(kwargs, "alpha_hover");
    cfg->alpha_shaping = dict_get(kwargs, "alpha_shaping");
    cfg->alpha_omega = dict_get(kwargs, "alpha_omega");
    env->task_config = cfg;
}

static void drone_race_config(DroneEnv* env, Dict* kwargs) {
    RaceConfig* cfg = (RaceConfig*)calloc(1, sizeof(RaceConfig));
    cfg->max_rings = dict_get(kwargs, "max_rings");
    cfg->ring_reward = dict_get(kwargs, "ring_reward");
    cfg->collision_penalty = dict_get(kwargs, "collision_penalty");
    cfg->time_penalty = dict_get(kwargs, "time_penalty");
    cfg->alpha_dist = dict_get(kwargs, "alpha_dist");
    env->task_config = cfg;
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = dict_get(kwargs, "num_drones");
    if (env->num_agents > MAX_DRONES) {
        fprintf(stderr, "drone: num_drones too large\n");
        exit(1);
    }
    int task = dict_get(kwargs, "task");
    if (task == 1) {
        env->task = &TASK_RACE;
        drone_race_config(env, kwargs);
    } else {
        env->task = &TASK_HOVER;
        drone_hover_config(env, kwargs);
    }
    env->task->init(env);
    if (DRONE_LOG_TASK != NULL && DRONE_LOG_TASK != env->task) {
        fprintf(stderr, "drone: multi-task mix not supported in log\n");
        exit(1);
    }
    DRONE_LOG_TASK = env->task;
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
    if (DRONE_LOG_TASK != NULL) {
        for (int i = 0; i < DRONE_LOG_TASK->num_log_keys; i++)
            dict_set(out, DRONE_LOG_TASK->log_keys[i], log->task[i]);
    }
}

