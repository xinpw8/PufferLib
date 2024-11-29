// TODO: Vanishing point for road should terminate in single pixel, not 2
// TODO: Make spawning closer to original (more clusters of cars, faster spawns at higher speeds)
// TODO: Ascertain original atari scoring logic and implement (differs from reward)

// TODO: Make sure it trains
// TODO: Engineer good policy

// :CURRENT TASKS - MUST DO:
// 1. remove atari logo
// 4. Render init creates 2 windows?

// :CURRENT TASKS - NICE TO DO:
// 1. consistent naming scheme for functions and vars
// 4. utilize inverted conditionals when possible
// 5. make spritesheet; load via DrawTexture Rectancle source vs 
// 6. individual sprites


// puffer_enduro.c

# define MAX_ENEMIES 10

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include "puffer_enduro.h"
#include "raylib.h"
#include <time.h>
// #include <gperftools/profiler.h>

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

    // debugging for eval
    // Clear log file
    FILE* log_file = fopen("collision_log.txt", "w");
    if (!log_file) {
        fprintf(stderr, "Error clearing log file.\n");
        exit(1);
    }
    fclose(log_file);

    log_file = fopen("collision_log.txt", "a");
    if (!log_file) {
        fprintf(stderr, "Error opening log file for appending.\n");
        exit(1);
    }

    int running = 1;
    while (running) {
        handleEvents(&running, &env);
        c_step(&env);

        for (int i = 0; i < env.max_enemies; i++) {
            Car* car = &env.enemyCars[i];

            if (check_collision(&env, car)) {
                printf("Collision detected with car %d in lane %d at x=%.2f, y=%.2f\n",
                       i, car->lane, car->x, car->y);
                log_collision(log_file, &env, car, i);
            }
        }

        c_render(client, &env);

        if (WindowShouldClose()) {
            running = 0;
        }
    }

    fclose(log_file);
    close_client(client, &env);
    cleanup(&client->gameState);
    free_allocated(&env);

    return 0;
}

void perftest(float test_time) {
    Enduro env;

    // initialize necessary variables
    env.num_envs = 1;
    env.max_enemies = MAX_ENEMIES;
    env.obs_size = OBSERVATIONS_MAX_SIZE;

    allocate(&env);

    unsigned int seed = 12345;
    init(&env, seed, 0); // initialize environment variables
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

