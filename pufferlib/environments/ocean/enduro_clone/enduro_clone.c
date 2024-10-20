// enduro_clone.c

// TODO: Too much space in road to sneak by enemies
    // TODO: Collision drift doesn't work. Make it work.
    // TODO: Add collisions with the road edges
    // TODO: Add scenery
    // TODO: Add day/night cycle + pre-dawn
// TODO: Ascertain day/night cycle logic from original
    // TODO: Add winter/summer cycle
// TODO: Ascertain winter/summer cycle logic from original
    // TODO: Improve road wiggle effect
// TODO: Ascertain road wiggle effect logic from original

// TODO: Compute/determine enemy car spawn frequency for each day num
// TODO: Ascertain exact atari enemy car spawn frequency per each day
// TODO: Ascertain lane choice logic per atari original
// TODO: Twisting road logic
// TODO: Twisting road render
// TODO: Implement RL
// TODO: Make sure it trains
// TODO: Engineer good policy
// TODO: Add original player car sprite/color
// TODO: Add original enemy car sprites/colors
// TODO: Ascertain original atari scoring logic and implement (differs from reward)



#include <stdio.h>
#include <stdlib.h>
#include "enduro_clone.h"
#include "raylib.h"

#define MAX_SPEED 3.0f
#define MIN_SPEED -1.5f

int main() {
    Enduro env = {
        .width = SCREEN_WIDTH,
        .height = SCREEN_HEIGHT,
        .car_width = CAR_WIDTH,
        .car_height = CAR_HEIGHT,
        .max_enemies = MAX_ENEMIES,
        .crash_noop_duration = CRASH_NOOP_DURATION,
        .day_length = DAY_LENGTH,
        .initial_cars_to_pass = INITIAL_CARS_TO_PASS,
        .min_speed = MIN_SPEED,
        .max_speed = MAX_SPEED,
    };

    allocate(&env);

    Client* client = make_client(&env);
    reset(&env);

    while (!WindowShouldClose()) {
        env.actions = 0;  // Default action is noop
        if (IsKeyDown(KEY_LEFT))  env.actions = 1;  // Move left
        if (IsKeyDown(KEY_RIGHT)) env.actions = 2;  // Move right
        if (IsKeyDown(KEY_UP))    env.actions = 3;  // Speed up
        if (IsKeyDown(KEY_DOWN))  env.actions = 4;  // Slow down

        step(&env);
        render(client, &env);
    }

    close_client(client);
    free_allocated(&env);

    return 0;
}
