// puffer_enduro.c

#define MAX_ENEMIES 10

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>
#include "enduro.h"
#include "raylib.h"
#include "puffernet.h"
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

#ifndef RESOURCE_PREFIX
#ifdef __EMSCRIPTEN__
#define RESOURCE_PREFIX ""
#else
#define RESOURCE_PREFIX "pufferlib/"
#endif
#endif

void get_input(Enduro *env) {
    if ((IsKeyDown(KEY_DOWN) && IsKeyDown(KEY_RIGHT)) ||
        (IsKeyDown(KEY_S)    && IsKeyDown(KEY_D)))
        env->actions[0] = ACTION_DOWNRIGHT;
    else if ((IsKeyDown(KEY_DOWN) && IsKeyDown(KEY_LEFT)) ||
             (IsKeyDown(KEY_S)    && IsKeyDown(KEY_A)))
        env->actions[0] = ACTION_DOWNLEFT;
    else if (IsKeyDown(KEY_SPACE) &&
             (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)))
        env->actions[0] = ACTION_RIGHTFIRE;
    else if (IsKeyDown(KEY_SPACE) &&
             (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)))
        env->actions[0] = ACTION_LEFTFIRE;
    else if (IsKeyDown(KEY_SPACE))
        env->actions[0] = ACTION_FIRE;
    else if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S))
        env->actions[0] = ACTION_DOWN;
    else if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A))
        env->actions[0] = ACTION_LEFT;
    else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D))
        env->actions[0] = ACTION_RIGHT;
    else
        env->actions[0] = ACTION_NOOP;
}

static Enduro env = {0};
static LinearLSTM *net = NULL;

static void step_frame(void *unused) {
    if (IsKeyDown(KEY_LEFT_SHIFT))
        get_input(&env);
    else
        forward_linearlstm(net, env.observations, env.actions);

    c_step(&env);
    c_render(&env);
}

#ifndef __EMSCRIPTEN__
void perftest(float test_time) {
    Enduro perf_env = {
        .num_envs = 1,
        .max_enemies = MAX_ENEMIES,
        .obs_size = OBSERVATIONS_MAX_SIZE
    };

    allocate(&perf_env);
    init(&perf_env);
    c_reset(&perf_env);

    int start = clock();
    int steps = 0;
    while (clock() - start < test_time * CLOCKS_PER_SEC) {
        perf_env.actions[0] = rand() % 9;
        c_step(&perf_env);
        steps++;
    }

    printf("Enduro: %d steps in %.2f seconds (%.2f steps/s)\n", steps, test_time, steps / test_time);
    free_allocated(&perf_env);
}
#endif

int main() {
    Weights *w = load_weights(RESOURCE_PREFIX "resources/enduro/enduro_weights.bin", 142218);
    int ls[1] = {9};
    net = make_linearlstm(w, 1, 68, ls, 1);

    env.num_envs    = 1;
    env.max_enemies = MAX_ENEMIES;
    env.obs_size    = OBSERVATIONS_MAX_SIZE;

    allocate(&env);
    init(&env);
    c_reset(&env);

#ifdef __EMSCRIPTEN__
    emscripten_set_main_loop_arg(step_frame, NULL, 0, 1);
#else
    c_render(&env);
    while (!WindowShouldClose()) {
        step_frame(NULL);
    }

    perftest(10.0f);

    free_linearlstm(net);
    free(w);
    free_allocated(&env);
#endif

    return 0;
}