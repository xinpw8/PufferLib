#pragma once

#include "task_hover.h"
#include "task_race.h"


const char* task_name(TaskType task) {
    switch (task) {
        case TASK_HOVER: return "hover";
        case TASK_RACE: return "race";
        case TASK_SPHERE: return "sphere";
        case TASK_CUBE: return "cube";
        case TASK_FLAG: return "flag";
    }
    return "?";
}

int task_horizon(DroneEnv* env) {
    switch (env->task) {
        case TASK_RACE: return ((RaceConfig*)env->task_config)->horizon;
        default: return ((HoverConfig*)env->task_config)->horizon;
    }
}

void task_init(DroneEnv* env) {
    switch (env->task) {
        case TASK_RACE: race_init(env); break;
        default: hover_init(env); break;
    }
}

void task_close(DroneEnv* env) {
    switch (env->task) {
        case TASK_RACE: race_close(env); break;
        default: hover_close(env); break;
    }
}

void task_env_reset(DroneEnv* env) {
    switch (env->task) {
        case TASK_RACE: race_env_reset(env); break;
        default: break;
    }
}

void task_reset(DroneEnv* env, Drone* agent, int idx) {
    switch (env->task) {
        case TASK_HOVER: hover_reset(env, agent, idx); break;
        case TASK_SPHERE: sphere_reset(env, agent, idx); break;
        case TASK_CUBE: cube_reset(env, agent, idx); break;
        case TASK_FLAG: flag_reset(env, agent, idx); break;
        case TASK_RACE: race_reset(env, agent, idx); break;
    }
}

float task_reward(DroneEnv* env, Drone* agent, int idx, StepCache* cache) {
    switch (env->task) {
        case TASK_RACE: return race_reward(env, agent, idx, cache);
        default: return hover_reward(env, agent, idx, cache);
    }
}

bool task_done(DroneEnv* env, Drone* agent, int idx, StepCache* cache) {
    switch (env->task) {
        case TASK_RACE: return race_done(env, agent, idx, cache);
        default: return hover_done(env, agent, idx, cache);
    }
}

void task_log(DroneEnv* env, Drone* agent, int idx, Log* log, StepCache* cache) {
    switch (env->task) {
        case TASK_RACE: race_log(env, agent, idx, log, cache); break;
        default: hover_log(env, agent, idx, log, cache); break;
    }
}
