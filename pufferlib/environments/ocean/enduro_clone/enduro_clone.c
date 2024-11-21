// TODO: Vanishing point for road should terminate in single pixel, not 2
// TODO: Make spawning closer to original (more clusters of cars, faster spawns at higher speeds)
// TODO: Ascertain original atari scoring logic and implement (differs from reward)

// TODO: Make sure it trains
// TODO: Engineer good policy

// :CURRENT TASK:
// TODO: Implement RL in pufferlib

// enduro_clone.c

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include "enduro_clone.h"
#include "raylib.h"

#define SCREEN_WIDTH 160
#define SCREEN_HEIGHT 210
#define CAR_WIDTH 16
#define CAR_HEIGHT 11
#define MAX_ENEMIES 10
#define INITIAL_CARS_TO_PASS 200
#define MIN_SPEED -2.5f
#define MAX_SPEED 7.5f


int main() {
    Enduro env = {
        .width = SCREEN_WIDTH,
        .height = SCREEN_HEIGHT,
        .car_width = CAR_WIDTH,
        .car_height = CAR_HEIGHT,
        .max_enemies = MAX_ENEMIES,
        .initial_cars_to_pass = INITIAL_CARS_TO_PASS,
        .min_speed = MIN_SPEED,
        .max_speed = MAX_SPEED,
    };

    allocate(&env);
    initRaylib();
    Client* client = make_client(&env);
    loadTextures(&env.gameState);
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

    cleanup(&env.gameState);
    free_allocated(&env);
    close_client(client, &env);
}
