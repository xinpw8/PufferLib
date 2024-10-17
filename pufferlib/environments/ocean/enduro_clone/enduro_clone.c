// enduro_clone.c
#include <stdio.h>
#include <stdlib.h>
#include "enduro_clone.h"
#include "raylib.h"  // Including raylib for window management and rendering

// // Function to clear logs by truncating the file
// void clear_logs(const char* filename) {
//     FILE* file = fopen(filename, "w");  // Open in write mode to truncate
//     if (file != NULL) {
//         fclose(file);  // Close immediately after clearing
//     }
// }

int main() {
    // // Clear logs at the start of the game
    // clear_logs("game_debug.log");

    // printf("Starting game loop...\n"); fflush(stdout);

    // Initialize the Enduro environment
    Enduro env = {
        .width = SCREEN_WIDTH,
        .height = SCREEN_HEIGHT,
        .hud_height = HUD_HEIGHT,
        .car_width = CAR_WIDTH,
        .car_height = CAR_HEIGHT,
        .player_x = SCREEN_WIDTH / 2 - CAR_WIDTH / 2,
        .player_y = SCREEN_HEIGHT - CAR_HEIGHT - HUD_HEIGHT,
        .speed = 1.0f,
        .max_speed = 10.0f,
        .min_speed = -1.0f,
        .score = 0,
        .carsToPass = INITIAL_CARS_TO_PASS,
        .day = 1,
        .day_length = DAY_LENGTH,
        .step_count = 0,
        .numEnemies = 0,
        .collision_cooldown = 0,
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
