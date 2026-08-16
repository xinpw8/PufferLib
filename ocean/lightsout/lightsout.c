#include <stdio.h>
#include <time.h>
#include "lightsout.h"

int demo() {
    srand((unsigned)time(NULL));
    LightsOut env = {
        .grid_size = 5,
        .cell_size = 100,
        .max_steps = 100,
        .scramble_prob = 0.15f,
        .rng = (unsigned)time(NULL),
    };
    env.agents[0].observations = (unsigned char*)calloc(
        env.grid_size * env.grid_size, sizeof(unsigned char));
    env.agents[0].actions = (float*)calloc(1, sizeof(float));
    env.agents[0].rewards = (float*)calloc(1, sizeof(float));
    env.agents[0].terminals = (float*)calloc(1, sizeof(float));

    puf_reset(&env);
    env.client = make_client(env.cell_size, env.grid_size);

    while (!WindowShouldClose()) {
        puf_step(&env);
        puf_render(&env);
    }

    free(env.agents[0].observations);
    free(env.agents[0].actions);
    free(env.agents[0].rewards);
    free(env.agents[0].terminals);
    puf_close(&env);
    return 0;
}

int main(void) {
    demo();
    return 0;
}
