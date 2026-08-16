#include "laser_puzzle.h"
#include "puffercpu.h"

#define WEIGHTS_PATH "resources/laser_puzzle/laser_puzzle_weights.bin"

static void copy_observations(float* out, const unsigned char* in) {
    for (int i = 0; i < LASER_PUZZLE_OBS_SIZE; i++) {
        out[i] = (float)in[i];
    }
}

int demo() {
    Weights* weights = load_weights(WEIGHTS_PATH);
    if (weights == NULL) {
        return 1;
    }

    int logit_sizes[1] = {NUM_ACTIONS};
    PufferNet* net = make_puffernet(
        weights, 1, LASER_PUZZLE_OBS_SIZE, 128, 2, logit_sizes, 1);
    float net_obs[LASER_PUZZLE_OBS_SIZE] = {0};

    LaserPuzzle env = {0};
    unsigned char observations[LASER_PUZZLE_OBS_SIZE] = {0};
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    env.agents[0].observations = observations;
    env.agents[0].actions = actions;
    env.agents[0].rewards = rewards;
    env.agents[0].terminals = terminals;

    puf_init(&env, NULL);
    puf_reset(&env);
    env.client = make_client();

    while (!WindowShouldClose()) {
        copy_observations(net_obs, env.agents[0].observations);
        forward_puffernet(net, net_obs, env.agents[0].actions);
        puf_step(&env);
        puf_render(&env);
    }

    free_puffernet(net);
    free(weights);

    // call closing procedures
    puf_close(&env);
    return 0;
}

int main() {
    demo();
    return 0;
}
