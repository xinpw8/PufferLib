// puffer_enduro.c

#define MAX_ENEMIES 10

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>
#include "enduro.h"
#include "raylib.h"
#include "puffercpu.h"

int demo() {
    Weights* weights = load_weights("resources/enduro/enduro_weights.bin");
    int logit_sizes[1] = {9};
    PufferNet* net = make_puffernet(weights, 1, 68, 128, 2, logit_sizes, 1);

    Enduro env = {
        .max_enemies = MAX_ENEMIES,
        .obs_size = OBSERVATIONS_MAX_SIZE
    };

    env.agents[0].observations = (float*)calloc(env.obs_size, sizeof(float));
    env.agents[0].actions = (float*)calloc(1, sizeof(float));
    env.agents[0].rewards = (float*)calloc(1, sizeof(float));
    env.agents[0].terminals = (float*)calloc(1, sizeof(float));
    env.num_agents = 1;

    init(&env);
    puf_reset(&env);
    puf_render(&env);

    while (!WindowShouldClose()) {
        forward_puffernet(net, env.agents[0].observations, env.agents[0].actions);
        puf_step(&env);
        puf_render(&env);
    }

    free_puffernet(net);
    free(weights);
    free(env.agents[0].observations);
    free(env.agents[0].actions);
    free(env.agents[0].rewards);
    free(env.agents[0].terminals);
    puf_close(&env);
    return 0;
}

int main() {
   demo();
   return 0;
}
