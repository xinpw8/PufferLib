// enduro_clone.c

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h> // For NULL
#include "enduro_clone.h"


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


// enduro_clone.c

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

    // Initialize SDL and create window and renderer
    SDL_Window* window = NULL;
    SDL_Renderer* renderer = NULL;

    if (initSDL(&window, &renderer) != 0) {
        return -1;
    }

    // Create client
    Client* client = make_client(&env, renderer);

    // Load textures
    GameState gameState = {0};
    loadTextures(renderer, &gameState);

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
        updateBackground(&gameState, env.day % 16); // Ensure day cycles correctly
        updateMountains(&gameState, roadDirection);

        // Update victory effects
        updateVictoryEffects(&gameState);

        // Update score
        updateScore(&gameState);

        // Clear screen
        SDL_RenderClear(renderer);

        // Render background and mountains
        renderBackground(renderer, &gameState);
        renderMountains(renderer, &gameState);

        // Render scoreboard
        renderScoreboard(renderer, &gameState);

        // Render the rest of the game (player car, enemy cars, road sides)
        render(client, &env);

        // Present renderer
        SDL_RenderPresent(renderer);

        // Delay to control frame rate
        SDL_Delay(16); // Approximately 60 FPS
    }

    // Cleanup
    cleanup(window, renderer, &gameState);

    close_client(client);
    free_allocated(&env);

    return 0;
}
