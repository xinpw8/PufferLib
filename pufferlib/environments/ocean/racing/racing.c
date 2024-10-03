// racing.c
#include "racing.h"

int main() {
    CEnduro env = {
        .width = 500,
        .height = 640,
        .player_width = 40,
        .player_height = 60,
        .other_car_width = 30,
        .other_car_height = 60,
        .player_speed = 3.0f,
        .base_car_speed = 2.0f,
        .max_player_speed = 10.0f,
        .min_player_speed = 1.0f,
        .speed_increment = 0.5f,
        .max_score = 100,
        .frameskip = 4,
    };

    allocate(&env);
    Client* client = make_client(&env);
    reset(&env);

    while (!WindowShouldClose()) {
        env.actions[0] = 0;
        if (IsKeyDown(KEY_LEFT)) env.actions[0] = 0;
        if (IsKeyDown(KEY_RIGHT)) env.actions[0] = 2;
        if (IsKeyDown(KEY_UP)) env.actions[0] = 1;

        step(&env);
        render(client, &env);
    }

    close_client(client);
    free_allocated(&env);
}
