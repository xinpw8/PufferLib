// enduro.c

#define MAX_ENEMIES 10

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>
#include "enduro.h"
#include "raylib.h"
#include "puffernet.h"

void get_input(Enduro* env) {
        if ((IsKeyDown(KEY_DOWN) && IsKeyDown(KEY_RIGHT)) || (IsKeyDown(KEY_S) && IsKeyDown(KEY_D))) {
            env->actions[0] = ACTION_DOWNRIGHT; // Decelerate and move right
        } else if ((IsKeyDown(KEY_DOWN) && IsKeyDown(KEY_LEFT)) || (IsKeyDown(KEY_S) && IsKeyDown(KEY_A))) {
            env->actions[0] = ACTION_DOWNLEFT; // Decelerate and move left
        } else if (IsKeyDown(KEY_SPACE) && (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D))) {
            env->actions[0] = ACTION_RIGHTFIRE; // Accelerate and move right
        } else if (IsKeyDown(KEY_SPACE) && (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A))) {
            env->actions[0] = ACTION_LEFTFIRE; // Accelerate and move left   
        } else if (IsKeyDown(KEY_SPACE)) {
            env->actions[0] = ACTION_FIRE; // Accelerate
        } else if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) {
            env->actions[0] = ACTION_DOWN; // Decelerate
        } else if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
            env->actions[0] = ACTION_LEFT; // Move left
        } else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
            env->actions[0] = ACTION_RIGHT; // Move right
        } else {
            env->actions[0] = ACTION_NOOP; // No action
        }
}

int demo() {
    // Weights* weights = load_weights("resources/enduro/enduro_weights.bin", 142218);
    // LinearLSTM* net = make_linearlstm(weights, 1, 68, 9);

    Weights* weights = load_weights("resources/enduro/enduro_w.bin", 10122);
    Default* net = make_default(weights, 1, 68, 128, 9);

    Enduro env = {
        .num_envs = 1,
        .max_enemies = MAX_ENEMIES,
        .obs_size = OBSERVATIONS_MAX_SIZE
    };

    allocate(&env);
    GameState* client = make_client(&env);
    unsigned int seed = 0;
    init(&env, seed, 0);
    reset(&env);

    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            get_input(&env);
        } else {
            forward_default(net, env.observations, env.actions);
        }

        c_step(&env);
        c_render(client, &env);
    }

    free_default(net);
    free(weights);
    close_client(client, &env);
    free_allocated(&env);
    return 0;
}

void perftest(float test_time) {
    Enduro env = {
        .num_envs = 1,
        .max_enemies = MAX_ENEMIES,
        .obs_size = OBSERVATIONS_MAX_SIZE
    };

    allocate(&env);

    unsigned int seed = 12345;
    init(&env, seed, 0);
    reset(&env);

    int start = time(NULL);
    int i = 0;
    while (time(NULL) - start < test_time) {
        env.actions[0] = rand()%9;
        c_step(&env);
        i++;
    }

    int end = time(NULL);
    printf("SPS: %f\n", i / (float)(end - start));
    free_allocated(&env);
}

int main() {
   demo();
//    perftest(20.0f);
   return 0;
}
