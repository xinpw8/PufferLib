#include "rek.h"

#define OBS_SIZE REK_OBS_SIZE
#define NUM_ATNS REK_NUM_ACTUATORS
#define ACT_SIZES REK_ACT_SIZES
#define OBS_TENSOR_T FloatTensor

#define Env Rek
#define MY_VEC_INIT
#define MY_VEC_CLOSE
#include "vecenv.h"

Env* my_vec_init(
        int* num_envs_out,
        int* buffer_env_starts,
        int* buffer_env_counts,
        Dict* vec_kwargs,
        Dict* env_kwargs) {
    int total_agents = (int)dict_get(vec_kwargs, "total_agents")->value;
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers")->value;
    if (total_agents <= 0 || num_buffers <= 0 || total_agents % num_buffers != 0) {
        rek_fail("total_agents must be positive and divisible by num_buffers");
    }

    Env* envs = (Env*)calloc((size_t)total_agents, sizeof(Env));
    if (envs == NULL) rek_fail("vector environment allocation failed");

    mjModel* shared_model = rek_load_model();
    int max_steps = (int)dict_get(env_kwargs, "max_steps")->value;
    int envs_per_buffer = total_agents / num_buffers;
    for (int buffer = 0; buffer < num_buffers; buffer++) {
        buffer_env_starts[buffer] = buffer * envs_per_buffer;
        buffer_env_counts[buffer] = envs_per_buffer;
    }

    for (int i = 0; i < total_agents; i++) {
        envs[i].num_agents = 1;
        envs[i].rng = (unsigned int)i;
        envs[i].max_steps = max_steps;
        rek_init_with_model(&envs[i], shared_model, 0);
    }

    *num_envs_out = total_agents;
    return envs;
}

void my_vec_close(Env* envs) {
    if (envs != NULL && envs[0].model != NULL) {
        mj_deleteModel(envs[0].model);
        envs[0].model = NULL;
    }
}

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->max_steps = (int)dict_get(kwargs, "max_steps")->value;
    rek_init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "root_height", log->root_height);
    dict_set(out, "max_abs_qvel", log->max_abs_qvel);
    dict_set(out, "invalid_termination", log->invalid_termination);
    dict_set(out, "timeout", log->timeout);
    dict_set(out, "n", log->n);
}
