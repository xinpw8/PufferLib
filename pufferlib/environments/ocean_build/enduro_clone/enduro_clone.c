#include "enduro_clone.h"

int main() {
    printf("Starting game loop...\n"); fflush(stdout);
    Enduro env = {
        .width = 160,
        .height = 210,
        .player_x = 80,
        .player_y = 180,
        .speed = 1.0f,
        .max_speed = 10.0f,
        .min_speed = -1.0f,
        .score = 0,
        .carsToPass = 5,  // Starting number of cars to pass
        .day = 1,
        .day_length = 2000,
        .step_count = 0,
        .numEnemies = 0,
        .collision_cooldown = 0,
    };
    allocate(&env, env.width, env.height, env.hud_height, env.car_width,
    env.car_height, env.max_enemies, env.crash_noop_duration, env.day_length,
    env.initial_cars_to_pass, env.min_speed, env.max_speed);  // Initialize memory and variables


    Client* client = make_client(&env);
    reset(&env);

    while (!WindowShouldClose()) {
        printf("Calling step()...\n"); fflush(stdout);  // Debug print

        // Update actions based on input
        env.actions[0] = 0;
        if (IsKeyDown(KEY_LEFT))  env.actions[0] = 1;
        if (IsKeyDown(KEY_RIGHT)) env.actions[0] = 2;
        if (IsKeyDown(KEY_UP))    env.actions[0] = 3;
        if (IsKeyDown(KEY_DOWN))  env.actions[0] = 4;

        step(&env);  // This should call your step() function

        printf("Called step()\n"); fflush(stdout);  // Debug print

        render(client, &env);
    }

    close_client(client);
    free_allocated(&env);

    return 0;
}