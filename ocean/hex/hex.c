#include "hex.h"
#include <math.h>
#include <stdio.h>
#include <time.h>

static void bind_demo_buffers(Hex* env) {
    env->num_agents = 1;
    env->agents[0].observations = (float*)calloc(OBS_SIZE, sizeof(float));
    env->agents[0].actions = (float*)calloc(NUM_ATNS, sizeof(float));
    env->agents[0].rewards = (float*)calloc(1, sizeof(float));
    env->agents[0].terminals = (float*)calloc(1, sizeof(float));
    init(env);
}

static void free_demo_buffers(Hex* env) {
    free(env->agents[0].observations);
    free(env->agents[0].actions);
    free(env->agents[0].rewards);
    free(env->agents[0].terminals);
}

void demo() {
    Hex env = {0};
    bind_demo_buffers(&env);
    puf_reset(&env);
    puf_render(&env);
    env.random_opponent = false;

    while (!WindowShouldClose()) {
        puf_step(&env);
        puf_render(&env);
    }

    free_demo_buffers(&env);
    puf_close(&env);
}

void speed_test() {
    Hex env = {0};
    bind_demo_buffers(&env);
    puf_reset(&env);
    clock_t start = clock();

    int num_steps = 1000000;
    for(int i = 0; i < num_steps; i++) {
        env.agents[0].actions[0] = compute_legal_move(&env);
        puf_step(&env);
    }
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time for %d steps: %.2f seconds\n", num_steps, elapsed);
    printf("SPS: %.2fM\n", num_steps / elapsed / 1e6);

    free_demo_buffers(&env);
    puf_close(&env);
}

int main() {
    demo();
    // speed_test();
    return 0;
}
