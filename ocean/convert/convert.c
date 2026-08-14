#include "convert.h"
#include "puffercpu.h"

int main() {
    Convert env = {
        .width = 1920,
        .height = 1080,
        .num_agents = 1024,
        .num_factories = 32,
        .num_resources = 8,
    };
    init(&env);

    int num_obs = 2*env.num_resources + 4 + env.num_resources;
    env.agents[0].observations = calloc(env.num_agents*num_obs, sizeof(float));
    env.agents[0].actions = calloc(2*env.num_agents, sizeof(int));
    env.agents[0].rewards = calloc(env.num_agents, sizeof(float));
    env.agents[0].terminals = calloc(env.num_agents, sizeof(unsigned char));

    Weights* weights = load_weights("resources/convert/convert_weights.bin");
    int logit_sizes[2] = {9, 5};
    LinearLSTM* net = make_linearlstm(weights, env.num_agents, num_obs, logit_sizes, 2);

    puf_reset(&env);
    puf_render(&env);

    while (!WindowShouldClose()) {
        for (int i=0; i<env.num_agents; i++) {
            env.agents[0].actions[2*i] = rand() % 9;
            env.agents[0].actions[2*i + 1] = rand() % 5;
        }

        forward_linearlstm(net, env.agents[0].observations, env.agents[0].actions);
        compute_observations(&env);
        puf_step(&env);
        puf_render(&env);
    }

    free_linearlstm(net);
    free(env.agents[0].observations);
    free(env.agents[0].actions);
    free(env.agents[0].rewards);
    free(env.agents[0].terminals);
    puf_close(&env);
}

