#include "blastar.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "puffercpu.h"

const char* WEIGHTS_PATH = "resources/blastar/blastar_weights.bin";
#define OBSERVATIONS_SIZE 10
#define ACTIONS_SIZE 6
#define NUM_WEIGHTS 134407

int demo() {
    Weights* weights = load_weights(WEIGHTS_PATH);
    int logit_sizes[1] = {ACTIONS_SIZE};
    LinearLSTM* net = make_linearlstm(weights, 1, OBSERVATIONS_SIZE, logit_sizes, 1);
    Blastar env = {
        .num_obs = OBSERVATIONS_SIZE,
    };
    env.agents[0].observations = calloc(env.num_obs, sizeof(float));
    env.agents[0].actions = (float*)calloc(1, sizeof(float));
    env.agents[0].rewards = (float*)calloc(1, sizeof(float));
    env.agents[0].terminals = (float*)calloc(1, sizeof(float));
    env.num_agents = 1;
    init(&env, env.num_obs);
    Client* client = make_client(&env);
    unsigned int seed = 12345;
    srand(seed);
    puf_reset(&env);
    int running = 1;
    while (running) {
        forward_linearlstm(net, env.agents[0].observations, env.agents[0].actions);
        puf_step(&env);
        puf_render(&env);
        if (WindowShouldClose() || env.game_over) {
            running = 0;
        }
    }
    free_linearlstm(net);
    free(weights);
    close_client(client);
    free(env.agents[0].observations);
    free(env.agents[0].actions);
    free(env.agents[0].rewards);
    free(env.agents[0].terminals);
    return 0;
}

void perftest(float test_time) {
    Blastar env = {
        .num_obs = OBSERVATIONS_SIZE,
    };
    env.agents[0].observations = calloc(env.num_obs, sizeof(float));
    env.agents[0].actions = (float*)calloc(1, sizeof(float));
    env.agents[0].rewards = (float*)calloc(1, sizeof(float));
    env.agents[0].terminals = (float*)calloc(1, sizeof(float));
    env.num_agents = 1;
    init(&env, env.num_obs);
    unsigned int seed = 12345;
    srand(seed);
    puf_reset(&env);
    int start = time(NULL);
    int steps = 0;
    while (time(NULL) - start < test_time) {
        env.agents[0].actions[0] = rand() % ACTIONS_SIZE;  // Random actions
        puf_step(&env);
        steps++;
    }
    int end = time(NULL);
    printf("Steps per second: %f\n", steps / (float)(end - start));
    free(env.agents[0].observations);
    free(env.agents[0].actions);
    free(env.agents[0].rewards);
    free(env.agents[0].terminals);
}

int main() {
    demo();
    // perftest(10.0f);
    return 0;
}
