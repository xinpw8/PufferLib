#include "rek_sandbox.h"

#define OBS_SIZE REK_SANDBOX_OBS_SIZE
#define NUM_ATNS REK_SANDBOX_NUM_ACTIONS
#define ACT_SIZES REK_SANDBOX_ACT_SIZES
#define OBS_TENSOR_T FloatTensor

#define Env RekSandbox
#define MY_VEC_INIT
#define MY_VEC_CLOSE
#include "vecenv.h"

static void rek_sandbox_load_kwargs(Env* env, Dict* kwargs) {
    env->num_agents = REK_SANDBOX_NUM_AGENTS;
    env->max_steps = (int)dict_get(kwargs, "max_steps")->value;
    env->action_scale = (float)dict_get(kwargs, "action_scale")->value;
    env->action_clip = (float)dict_get(kwargs, "action_clip")->value;
    env->dummy_amplitude = (float)dict_get(kwargs, "dummy_amplitude")->value;
    env->dummy_frequency_hz = (float)dict_get(kwargs, "dummy_frequency_hz")->value;
    env->fall_height = (float)dict_get(kwargs, "fall_height")->value;
    env->fall_up_z = (float)dict_get(kwargs, "fall_up_z")->value;
    env->upright_reward_weight = (float)dict_get(
        kwargs, "upright_reward_weight"
    )->value;
    env->height_reward_weight = (float)dict_get(
        kwargs, "height_reward_weight"
    )->value;
    env->action_cost_weight = (float)dict_get(
        kwargs, "action_cost_weight"
    )->value;
    env->fall_penalty = (float)dict_get(kwargs, "fall_penalty")->value;
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
    if (total_agents <= 0 || num_buffers <= 0
            || (total_agents % num_buffers) != 0) {
        rek_sandbox_fail(
            "total_agents must be positive and divisible by num_buffers"
        );
    }

    Env* envs = (Env*)calloc((size_t)total_agents, sizeof(Env));
    if (envs == NULL) rek_sandbox_fail("vector environment allocation failed");

    mjModel* shared_model = rek_sandbox_load_model();
    int envs_per_buffer = total_agents / num_buffers;
    for (int buffer = 0; buffer < num_buffers; buffer++) {
        buffer_env_starts[buffer] = buffer * envs_per_buffer;
        buffer_env_counts[buffer] = envs_per_buffer;
    }

    for (int i = 0; i < total_agents; i++) {
        envs[i].rng = (unsigned int)i;
        rek_sandbox_load_kwargs(&envs[i], env_kwargs);
        rek_sandbox_init_with_model(&envs[i], shared_model, 0);
        envs[i].shared_model_env_count = total_agents;
        if (!rek_sandbox_config_is_valid(&envs[i])) {
            rek_sandbox_fail("invalid vector curriculum configuration");
        }
    }

    *num_envs_out = total_agents;
    return envs;
}

void my_vec_close(Env* envs) {
    rek_sandbox_close_shared_model(envs);
}

void my_init(Env* env, Dict* kwargs) {
    rek_sandbox_load_kwargs(env, kwargs);
    rek_sandbox_init(env);
    if (!rek_sandbox_config_is_valid(env)) {
        rek_sandbox_fail("invalid curriculum configuration");
    }
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "learner_root_height", log->learner_root_height);
    dict_set(out, "learner_root_up_z", log->learner_root_up_z);
    dict_set(out, "action_rms", log->action_rms);
    dict_set(out, "dummy_resets", log->dummy_resets);
    dict_set(out, "learner_fall", log->learner_fall);
    dict_set(out, "invalid_termination", log->invalid_termination);
    dict_set(out, "timeout", log->timeout);
    dict_set(out, "n", log->n);
}
