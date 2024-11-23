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

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include "puffer_enduro.h"
#include "raylib.h"
#include <gperftools/profiler.h>

int main() {
    ProfilerStart("puffer_enduro.prof");
    Enduro env;
    init(&env);
    allocate(&env);
    initRaylib();
    Client* client = NULL;
    client = make_client(client);
    loadTextures(&client->gameState);
    reset(&env);
    int running = 1;

    while (running) {
        handleEvents(&running, &env);
        c_step(&env);
        c_render(client, &env);
        if (WindowShouldClose()) {
            running = 0;
        }
    }

    cleanup(&client->gameState);
    free_allocated(&env);
    close_client(client, &env);
    ProfilerStop();

    return 0;
}
