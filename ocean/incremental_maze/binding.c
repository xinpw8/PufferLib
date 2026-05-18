#include "incremental_maze.h"
#define OBS_SIZE 121
#define NUM_ATNS 1
#define ACT_SIZES {5}
#define OBS_TENSOR_T ByteTensor
#define PUFFER_STATE_T State
#define PUFFER_STATE_SIZE ((int)sizeof(State))
#define PUFFER_STATE_REFRESH(env) compute_observations(env)

#define MY_VEC_INIT
#define MY_VEC_CLOSE
#define Env Grid
#include "vecenv.h"

Env* my_vec_init(int* num_envs_out, int* buffer_env_starts, int* buffer_env_counts,
                 Dict* vec_kwargs, Dict* env_kwargs) {
    (void)env_kwargs;
    int total_agents = (int)dict_get(vec_kwargs, "total_agents")->value;
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers")->value;
    int agents_per_buffer = total_agents / num_buffers;
    int num_envs = total_agents;
    int num_levels = INCREMENTAL_NUM_LEVELS;

    // Generate maze levels (shared across all envs)
    State* levels = calloc(num_levels, sizeof(State));

    for (int i = 0; i < num_levels; i++) {
        int sz = INCREMENTAL_MIN_SIZE + 2*i;
        State* level = &levels[i];
        level->width = sz;
        level->height = sz;

        create_maze_level(level, 0.5f, 0);
    }

    // Allocate all environments
    Env* envs = (Env*)calloc(num_envs, sizeof(Env));

    int buf = 0;
    int buf_agents = 0;
    buffer_env_starts[0] = 0;
    buffer_env_counts[0] = 0;

    for (int i = 0; i < num_envs; i++) {
        Env* env = &envs[i];
        env->num_levels = num_levels;
        env->num_agents = 1;
        env->levels = levels;
        env->rng = 0;

        buf_agents += env->num_agents;
        buffer_env_counts[buf]++;
        if (buf_agents >= agents_per_buffer && buf < num_buffers - 1) {
            buf++;
            buffer_env_starts[buf] = i + 1;
            buffer_env_counts[buf] = 0;
            buf_agents = 0;
        }
    }

    *num_envs_out = num_envs;
    return envs;
}

void my_vec_close(Env* envs) {
    free(envs[0].levels);
}

void my_init(Env* env, Dict* kwargs) {
    (void)kwargs;
    env->num_levels = INCREMENTAL_NUM_LEVELS;
    env->num_agents = 1;
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
