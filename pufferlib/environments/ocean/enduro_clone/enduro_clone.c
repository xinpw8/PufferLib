// enduro_clone.c
#include <stdio.h>
#include <stdlib.h>
#include "enduro_clone.h"
#include "raylib.h"  // Including raylib for window management and rendering

#define MAX_SPEED 10.0f
#define MIN_SPEED -1.0f

int main() {
    // Initialize the Enduro environment
    Enduro env = {
        .width = SCREEN_WIDTH,
        .height = SCREEN_HEIGHT,
        .hud_height = HUD_HEIGHT,
        .car_width = CAR_WIDTH,
        .car_height = CAR_HEIGHT,
        .max_enemies = MAX_ENEMIES,
        .crash_noop_duration = CRASH_NOOP_DURATION,
        .day_length = DAY_LENGTH,
        .initial_cars_to_pass = INITIAL_CARS_TO_PASS,
        .min_speed = MIN_SPEED,
        .max_speed = MAX_SPEED,
    };

    // Allocate resources and initialize the environment
    allocate(&env);

    // Initialize the rendering client
    Client* client = make_client(&env);
    reset(&env);

    // Main game loop
    while (!WindowShouldClose()) {
        // Update actions based on input (user can control the player)
        env.actions = 0;  // Default action is noop
        if (IsKeyDown(KEY_LEFT))  env.actions = 1;  // Move left
        if (IsKeyDown(KEY_RIGHT)) env.actions = 2;  // Move right
        if (IsKeyDown(KEY_UP))    env.actions = 3;  // Speed up
        if (IsKeyDown(KEY_DOWN))  env.actions = 4;  // Slow down

        step(&env);
        render(client, &env);
    }

    // Clean up resources after the game loop ends
    close_client(client);
    free_allocated(&env);

    return 0;
}
