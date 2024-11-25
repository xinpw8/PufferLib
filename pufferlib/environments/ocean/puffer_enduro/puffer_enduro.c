// TODO: Vanishing point for road should terminate in single pixel, not 2
// TODO: Make spawning closer to original (more clusters of cars, faster spawns at higher speeds)
// TODO: Ascertain original atari scoring logic and implement (differs from reward)

// TODO: Make sure it trains
// TODO: Engineer good policy

// :CURRENT TASK:
// Separated rendering and environment logic; player car speed
// seems to somehow be locked to enemy car speed. That is, there is
// no difference in the speeds, so player cannot ever pass enemies.
// Also, the vanishing point curve transitions are suddenly jumping
// when env->speed is low. When env->speed is high, the curve 
// transitions are smooth, like they should be.
// Also, currently, enemy cars seem to wiggle with the road...

// puffer_enduro.c

# define MAX_ENEMIES 10

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include "puffer_enduro.h"
#include "raylib.h"
// #include <gperftools/profiler.h>

int main() {
    Enduro env;

    // Initialize necessary variables
    env.num_envs = 1;
    env.max_enemies = MAX_ENEMIES;
    env.obs_size = OBSERVATIONS_MAX_SIZE;

    // ProfilerStart("puffer_enduro.prof");

    allocate(&env);

    Client* client = NULL;
    client = make_client(client);

    init(&env); // Initialize environment variables
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

    // ProfilerStop();

    return 0;
}

