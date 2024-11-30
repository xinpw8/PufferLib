// puffer_enduro.c

# define MAX_ENEMIES 10

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include "puffer_enduro.h"
#include "raylib.h"
#include <time.h>

int demo() {
    Enduro env;

    env.num_envs = 1;
    env.max_enemies = MAX_ENEMIES;
    env.obs_size = OBSERVATIONS_MAX_SIZE;

    allocate(&env);

    Client* client = make_client(&env);

    unsigned int seed = 12345;
    init(&env, seed, 0);
    reset(&env);
    initRaylib();

    loadTextures(&client->gameState);

    int running = 1;
    while (running) {
        handleEvents(&running, &env);
        c_step(&env);
        c_render(client, &env);

        if (WindowShouldClose()) {
            running = 0;
        }
    }

    close_client(client, &env);
    cleanup(&client->gameState);
    free_allocated(&env);

    return 0;
}

void perftest(float test_time) {
    Enduro env;

    env.num_envs = 1;
    env.max_enemies = MAX_ENEMIES;
    env.obs_size = OBSERVATIONS_MAX_SIZE;

    allocate(&env);

    unsigned int seed = 12345;
    init(&env, seed, 0);
    reset(&env);

    int start = time(NULL);
    int i = 0;
    int running = 1;
    while (time(NULL) - start < test_time) {
        handleEvents(&running, &env);
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
//    perftest(30.0f);
   return 0;
}

