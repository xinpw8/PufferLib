/* Pure C demo file for Convert. Build it with:
 * bash scripts/build_ocean.sh convert local (debug)
 * bash scripts/build_ocean.sh convert fast
 * We suggest building and debugging your env in pure C first. You
 * get faster builds and better error messages
 */
#include "convert.h"

/* Puffernet is our lightweight cpu inference library that
 * lets you load basic PyTorch model architectures so that
 * you can run them in pure C or on the web via WASM
 */
#include "puffernet.h"

int main() {
    // Weights are exported by running puffer export
    //Weights* weights = load_weights("resources/puffer_convert_weights.bin", 137743);

    //int logit_sizes[2] = {9, 5};
    //LinearLSTM* net = make_linearlstm(weights, num_agents, num_obs, logit_sizes, 2);

    Convert env = {
        .width = 1980,
        .height = 1020,
        .num_agents = 1024,
        .num_factories = 32,
        .num_resources = 8,
    };
    init(&env);

    // Allocate these manually since they aren't being passed from Python
    env.observations = calloc(env.num_agents*(2*env.num_resources + 4 + env.num_resources), sizeof(float));
    env.actions = calloc(2*env.num_agents, sizeof(int));
    env.rewards = calloc(env.num_agents, sizeof(float));
    env.terminals = calloc(env.num_agents, sizeof(unsigned char));

    // Always call reset and render first
    c_reset(&env);
    c_render(&env);

    // while(True) will break web builds
    while (!WindowShouldClose()) {
        for (int i=0; i<env.num_agents; i++) {
            env.actions[2*i] = rand() % 9;
            env.actions[2*i + 1] = rand() % 5;
        }

        //forward_linearlstm(net, env.observations, env.actions);
        compute_observations(&env);
        c_step(&env);
        c_render(&env);
    }

    // Try to clean up after yourself
    //free_linearlstm(net);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}

