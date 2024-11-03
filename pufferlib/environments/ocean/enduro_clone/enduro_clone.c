// TODO: Collision drift doesn't work. Make it work.
// TODO: Add collisions with the road edges
// TODO: Add scenery
// TODO: Add day/night cycle + pre-dawn
// TODO: Ascertain day/night cycle logic from original
// TODO: Add winter/summer cycle
// TODO: Ascertain winter/summer cycle logic from original
// TODO: Improve road wiggle effect
// TODO: Ascertain road wiggle effect logic from original
// TODO: Fog mode
// TODO: Reduced handling on snow
// TODO: Road vanishing point moves based on where player car is on road
// TODO: Compute/determine enemy car spawn frequency for each day num
// TODO: Ascertain exact atari enemy car spawn frequency per each day
// TODO: Ascertain lane choice logic per atari original
// TODO: Implement RL
// TODO: Make sure it trains
// TODO: Engineer good policy
// TODO: Add original player car sprite/color
// TODO: Add original enemy car sprites/colors
// TODO: Add enemy car tail lights for night
// TODO: Ascertain original atari scoring logic and implement (differs from reward)
// TODO: Slow speed player wheel animation
// TODO: Add road bump-out effect


// Leanke's TODOs:
// TODO: Combine 2 fns for leanke
// TODO: reduce line count

// thing 1: when the player moves left or right, the entire road effectively moves the opposite direction. so, when the player's car has moved all the way to the left on the road, the vanishing point of the road x position has moved to 110. and when player car is all the way right, the vanishing point has moved to x=62. this happens without respect to the road going straight or curving. if the road is straight, it stays going straight. 

// thing 2: the lines representing the sides of the road are 3 colors y=52-90 RGB 74 74 74, 91-105 RGB  111, 111, 111, 106-155  RGB 170, 170, 170. these sections of the road lines always stay these colors.

// the vanishing point x should be at 10 when the road curves left and the player's car is all the way to the right of the road,
//  and the vanishing point x should be at 158 when then road curves right and the player's car is all the way to the left of the road. 
// right now, the vanishing point x values are at 36 when curving left and 124 when curving right


// enduro_clone.c

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h> // For NULL
#include "enduro_clone.h"

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
    // Initialize raylib
    initRaylib();
    // Create client
    Client* client = make_client(&env);
    // Load textures
    loadTextures(&env.gameState);
    reset(&env);
    int running = 1;
    // Initialize RoadDirection variable
    RoadDirection roadDirection = ROAD_STRAIGHT;

    // Main game loop
    while (running) {
        // Handle events
        handleEvents(&running, &env);
        // Step the environment
        step(&env);

        // Update roadDirection based on env.current_curve_direction
        if (env.current_curve_direction == -1) {
            roadDirection = ROAD_TURN_LEFT;
        } else if (env.current_curve_direction == 1) {
            roadDirection = ROAD_TURN_RIGHT;
        } else {
            roadDirection = ROAD_STRAIGHT;
        }

        // Update game state
        updateBackground(&env.gameState, env.day % 16); // Ensure day cycles correctly
        updateMountains(&env.gameState, roadDirection);
        // Update victory effects
        updateVictoryEffects(&env.gameState);
        // Update score
        updateScore(&env.gameState);
        // Render everything
        render(client, &env);
        // Check if the window should close
        if (WindowShouldClose()) {
            running = 0;
        }
    }

    // Cleanup
    cleanup(&env.gameState);
    close_client(client);
    free_allocated(&env);
    return 0;
}
