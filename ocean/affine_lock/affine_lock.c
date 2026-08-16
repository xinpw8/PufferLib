#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "affine_lock.h"

static AffineLock* g_env = NULL;

static void demo_cleanup(void) {
    if (g_env == NULL) {
        return;
    }
    free(g_env->agents[0].observations);
    free(g_env->agents[0].actions);
    free(g_env->agents[0].rewards);
    free(g_env->agents[0].terminals);
    puf_close(g_env);
    g_env = NULL;
}

int main(void) {
    AffineLock env;
    memset(&env, 0, sizeof(env));
    g_env = &env;
    atexit(demo_cleanup);

    env.agents[0].observations = calloc(AFFINE_LOCK_OBS_SIZE, sizeof(float));
    env.agents[0].actions = (float*)calloc(AFFINE_LOCK_NUM_ATNS, sizeof(float));
    env.agents[0].rewards = (float*)calloc(1, sizeof(float));
    env.agents[0].terminals = (float*)calloc(1, sizeof(float));
    if (env.agents[0].observations == NULL || env.agents[0].actions == NULL ||
            env.agents[0].rewards == NULL || env.agents[0].terminals == NULL) {
        fprintf(stderr, "failed to allocate affine_lock demo buffers\n");
        return 1;
    }

    Dict kwargs = {0};
    dict_set(&kwargs, "start_depth", 2);
    dict_set(&kwargs, "max_depth", 16);
    dict_set(&kwargs, "step_grace", 2);
    dict_set(&kwargs, "seed", (double)(unsigned int)time(NULL));
    puf_init(&env, &kwargs);
    dict_clear(&kwargs);

    puf_reset(&env);
    puf_render(&env);

    while (!WindowShouldClose()) {
        puf_step(&env);
        puf_render(&env);
    }

    demo_cleanup();
    return 0;
}
