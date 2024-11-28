// TODO: Vanishing point for road should terminate in single pixel, not 2
// TODO: Make spawning closer to original (more clusters of cars, faster spawns at higher speeds)
// TODO: Ascertain original atari scoring logic and implement (differs from reward)

// TODO: Make sure it trains
// TODO: Engineer good policy

// :CURRENT TASKS - MUST DO:
// 1. remove atari logo
// 2. troubleshoot calling add_log() q step()
// 3. Compile warning for setup.py full compile (new)
// In file included from pufferlib/environments/ocean/puffer_enduro/cy_puffer_enduro.c:1272:
// pufferlib/environments/ocean/puffer_enduro/puffer_enduro.h:1747:29: note: expected ‘Client *’ but argument is of type ‘Enduro *’
//  1747 | Client* make_client(Client* client)
// 4. Render init creates 2 windows?
// 5. Add perf test

// :CURRENT TASKS - NICE TO DO:
// 1. consistent naming scheme for functions and vars
// 2. remove debug prints
// 3. remove commented out code
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

    // initialize necessary variables
    env.num_envs = 1;
    env.max_enemies = MAX_ENEMIES;
    env.obs_size = OBSERVATIONS_MAX_SIZE;

    // profilerstart("puffer_enduro.prof");

    allocate(&env);

    Client* client = make_client(&env);

    unsigned int seed = 12345;
    init(&env, seed, 0); // initialize environment variables
    reset(&env);
    initRaylib();

    loadTextures(&client->gameState);

    int running = 1;
    while (running) {
        handleEvents(&running, &env);
        c_step(&env);
        printf("reward: %f\n", env.rewards[0]);
        c_render(client, &env);

        if (WindowShouldClose()) {
            running = 0;
        }
    }

    close_client(client, &env);
    cleanup(&client->gameState);
    free_allocated(&env);

    // profilerstop();

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
   //demo();
   perftest(30.0f);
   return 0;
}

