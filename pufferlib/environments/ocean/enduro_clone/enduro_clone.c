#include "enduro_clone.h"

int main() {
    Enduro env = {
        .width = SCREEN_WIDTH,
        .height = SCREEN_HEIGHT,
        .frameskip = 4,
        .max_speed = 10.0f,
        .min_speed = 2.0f,
        .carsToPass = INITIAL_CARS_TO_PASS,
        .day_length = DAY_LENGTH,
        .step_count = 0,
        .numEnemies = 0,
        .collision_cooldown = 0,
    };
    allocate(&env);

    Client* client = make_client(&env);

    reset(&env);

    while (!WindowShouldClose()) {
        // User can control the car with arrow keys
        env.actions[0] = 0;
        if (IsKeyDown(KEY_LEFT))  env.actions[0] = 1;
        if (IsKeyDown(KEY_RIGHT)) env.actions[0] = 2;
        if (IsKeyDown(KEY_UP))    env.actions[0] = 3;
        if (IsKeyDown(KEY_DOWN))  env.actions[0] = 4;

        step(&env);
        render(client, &env);
    }
    close_client(client);
    free_allocated(&env);
}
