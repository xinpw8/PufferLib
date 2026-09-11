#include "rek_fight.h"

#define OBS_SIZE REK_FIGHT_OBS_SIZE
#define NUM_ATNS REK_FIGHT_NUM_ACTIONS
#define ACT_SIZES REK_FIGHT_ACT_SIZES
#define OBS_TENSOR_T FloatTensor

#define Env RekFight
#define MY_VEC_INIT
#define MY_VEC_CLOSE
#include "vecenv.h"

static void rek_fight_load_kwargs(Env* env, Dict* kwargs) {
    env->num_agents = REK_FIGHT_NUM_AGENTS;
    env->max_steps = (int)dict_get(kwargs, "max_steps")->value;
    env->fall_height = (float)dict_get(kwargs, "fall_height")->value;
    env->fall_up_z = (float)dict_get(kwargs, "fall_up_z")->value;
    env->root_stabilizer_scale = (float)dict_get(
        kwargs, "root_stabilizer_scale"
    )->value;
}

Env* my_vec_init(
        int* num_envs_out,
        int* buffer_env_starts,
        int* buffer_env_counts,
        Dict* vec_kwargs,
        Dict* env_kwargs) {
    int total_agents = (int)dict_get(vec_kwargs, "total_agents")->value;
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers")->value;
    int physical_envs = 0;
    int envs_per_buffer = 0;
    if (!rek_fight_partition_layout(
                total_agents,
                num_buffers,
                &physical_envs,
                &envs_per_buffer)) {
        rek_fight_fail(
            "total_agents must be positive and even, with whole matches per buffer"
        );
    }

    Env* envs = (Env*)calloc((size_t)physical_envs, sizeof(Env));
    if (envs == NULL) rek_fight_fail("vector environment allocation failed");

    mjModel* shared_model = rek_fight_load_model();
    for (int buffer = 0; buffer < num_buffers; buffer++) {
        buffer_env_starts[buffer] = buffer * envs_per_buffer;
        buffer_env_counts[buffer] = envs_per_buffer;
    }

    for (int i = 0; i < physical_envs; i++) {
        envs[i].rng = (unsigned int)i;
        rek_fight_load_kwargs(&envs[i], env_kwargs);
        rek_fight_init_with_model(&envs[i], shared_model, 0);
        envs[i].shared_model_env_count = physical_envs;
    }

    *num_envs_out = physical_envs;
    return envs;
}

void my_vec_close(Env* envs) {
    rek_fight_close_shared_model(envs);
}

void my_init(Env* env, Dict* kwargs) {
    rek_fight_load_kwargs(env, kwargs);
    rek_fight_init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "hits", log->hits);
    dict_set(out, "falls", log->falls);
    dict_set(out, "invalid_termination", log->invalid_termination);
    dict_set(out, "timeout", log->timeout);
    dict_set(out, "n", log->n);
}
