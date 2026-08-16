#include <stdlib.h>
#include <time.h>
#include "boss_fight.h"

int main() {
    srand(time(NULL));
    BossFight env = {0};
    env.rng = (unsigned int)time(NULL);
    env.agents[0].observations = (float*)calloc(OBS_SIZE, sizeof(float));
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
    puf_close(&env);
    return 0;
}
