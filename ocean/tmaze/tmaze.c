#include "tmaze.h"

int main() {
    TMaze env = {.size = 8};
    env.agents[0].observations = (unsigned char*)calloc(OBS_SIZE, sizeof(unsigned char));
    env.agents[0].actions = (float*)calloc(1, sizeof(float));
    env.agents[0].rewards = (float*)calloc(1, sizeof(float));
    env.agents[0].terminals = (float*)calloc(1, sizeof(float));

    puf_reset(&env);
    puf_render(&env);
    while (!WindowShouldClose()) {
        puf_step(&env);
        puf_render(&env);
    }
    free(env.agents[0].observations);
    free(env.agents[0].actions);
    free(env.agents[0].rewards);
    free(env.agents[0].terminals);
}

