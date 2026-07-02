#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config.h"
#include "checkpoint.h"
#include "tensor.h"

#define PUFFER_VECENV_INCLUDE
#include ENV_HEADER
#undef PUFFER_VECENV_INCLUDE

#include "puffernet.h"

static const char* model_path(Dict* cfg, const char* env_name,
        char* out, size_t out_size) {
    const char* path = puf_checkpoint_path(cfg, out, out_size);
    if (path) {
        return path;
    }

    snprintf(out, out_size, "resources/%s/%s_weights.bin", env_name, env_name);
    return out;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s ENV [section.key=value ...]\n", argv[0]);
        return 1;
    }

    const char* env_name = argv[1];
    Dict cfg = {0};
    puf_config_load_env(&cfg, env_name, argc - 2, argv + 2);

    OBS_TENSOR_T obs_type;
    if (sizeof(*obs_type.data) != sizeof(float)) {
        fprintf(stderr, "cpu eval currently requires FloatTensor observations\n");
        return 1;
    }

    char path_buf[1024];
    const char* path = model_path(&cfg, env_name, path_buf, sizeof(path_buf));
    Weights* weights = load_weights(path);
    if (!weights) {
        return 1;
    }

    int act_sizes[] = ACT_SIZES;
    int num_actions = (int)(sizeof(act_sizes) / sizeof(act_sizes[0]));

    int hidden_size = (int)puf_config_val(&cfg, "policy.hidden_size");
    int num_layers = (int)puf_config_val(&cfg, "policy.num_layers");

    Dict* env_kwargs = dict_copy_prefix(&cfg, "env.");
    Env env = {0};
    env.rng = 0;
    my_init(&env, env_kwargs);

    float observations[env.num_agents * OBS_SIZE];
    float actions[env.num_agents * NUM_ATNS];
    float rewards[env.num_agents];
    float terminals[env.num_agents];
    memset(observations, 0, sizeof(observations));
    memset(actions, 0, sizeof(actions));
    memset(rewards, 0, sizeof(rewards));
    memset(terminals, 0, sizeof(terminals));
    env.observations = (void*)observations;
    env.actions = actions;
    env.rewards = rewards;
    env.terminals = terminals;
    c_reset(&env);

    PufferNet* net = make_puffernet(weights, env.num_agents, OBS_SIZE,
        hidden_size, num_layers, act_sizes, num_actions);

    int frame = 0;
    c_render(&env);
    while (!WindowShouldClose()) {
        if (frame % 4 == 0) {
            forward_puffernet(net, observations, actions);
        }
        frame = (frame + 1) % 4;
        c_step(&env);
        c_render(&env);
    }

    c_close(&env);
    dict_free(env_kwargs);
    free_puffernet(net);
    free(weights);
    puf_config_free(&cfg);
    return 0;
}
