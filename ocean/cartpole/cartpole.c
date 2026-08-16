// local compile/eval implemented for discrete actions only
// eval with python demo.py --mode eval --env puffer_cartpole --eval-mode-path <path to model>

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "cartpole.h"
#include "puffercpu.h"

#define OBSERVATIONS_SIZE 4
#define ACTIONS_SIZE 2
#define CONTINUOUS 0

const char* WEIGHTS_PATH = "resources/cartpole/cartpole_weights.bin";

void demo() {
    Weights* weights = load_weights(WEIGHTS_PATH);
    
    int logit_sizes[1] = {ACTIONS_SIZE};
    PufferNet* net = make_puffernet(weights, 1, OBSERVATIONS_SIZE, 32, 2, logit_sizes, 1);
    
    Cartpole env = {
        .continuous = CONTINUOUS,
        .cart_mass = 1.0f,
        .pole_mass = 0.1f,
        .pole_length = 0.5f,
        .gravity = 9.8f,
        .force_mag = 10.0f,
        .tau = 0.02f,
    };
    env.agents[0].observations = (float*)calloc(OBSERVATIONS_SIZE, sizeof(float));
    env.agents[0].actions = (float*)calloc(1, sizeof(float));
    env.agents[0].rewards = (float*)calloc(1, sizeof(float));
    env.agents[0].terminals = (float*)calloc(1, sizeof(float));
    puf_reset(&env);
    puf_render(&env);

    while (!WindowShouldClose()) {
        forward_puffernet(net, env.agents[0].observations, env.agents[0].actions);
        env.agents[0].actions[0] = (env.agents[0].actions[0] > 0.5f) ? 1.0f : -1.0f;
        puf_step(&env);
        puf_render(&env);

        if (env.agents[0].terminals[0] > 0.5f) {
            puf_reset(&env);
        }
    }

    free_puffernet(net);
    free(weights);
    free(env.agents[0].observations);
    free(env.agents[0].actions);
    free(env.agents[0].rewards);
    free(env.agents[0].terminals);
}

int main() {
    srand(time(NULL));
    demo();
    return 0;
}
