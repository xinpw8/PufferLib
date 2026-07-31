#include "memory.h"

int main() {
    Memory env = {.length = 16};
    env.agents[0].observations = (float*)calloc(1, sizeof(unsigned char));
    env.agents[0].actions = (int*)calloc(1, sizeof(int));
    env.agents[0].rewards = (float*)calloc(1, sizeof(float));
    env.agents[0].terminals = (unsigned char*)calloc(1, sizeof(unsigned char));

    puf_reset(&env);
    puf_render(&env);
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (IsKeyDown(KEY_A) || IsKeyDown(KEY_LEFT)) {
                env.agents[0].actions[0] = 0;
            } else if (IsKeyDown(KEY_D) || IsKeyDown(KEY_RIGHT)) {
                env.agents[0].actions[0] = 1;
            } else {
                env.agents[0].actions[0] = -1;
            }
        } else {
            env.agents[0].actions[0] = rand() % 2;
        }
        puf_step(&env);
        puf_render(&env);
    }
    free(env.agents[0].observations);
    free(env.agents[0].actions);
    free(env.agents[0].rewards);
    free(env.agents[0].terminals);
    puf_close(&env);
}

