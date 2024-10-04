// racing.h

#ifndef RACING_H  // Check if RACING_H is not defined
#define RACING_H  // Define RACING_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include "raylib.h"
#include <assert.h>


// Action definitions
#define ACTION_NOOP     0
#define ACTION_ACCEL    1
#define ACTION_DECEL    2
#define ACTION_LEFT     3
#define ACTION_RIGHT    4

// Define screen dimensions for rendering
#define TOTAL_SCREEN_WIDTH 160
#define TOTAL_SCREEN_HEIGHT 210
#define ACTION_SCREEN_X_START 8
#define ACTION_SCREEN_Y_START 0
#define ACTION_SCREEN_WIDTH 152  // from x=8 to x=160
#define ACTION_SCREEN_HEIGHT 155 // from y=0 to y=155

#define SCOREBOARD_X_START 48
#define SCOREBOARD_Y_START 161
#define SCOREBOARD_WIDTH 64  // from x=48 to x=112
#define SCOREBOARD_HEIGHT 30 // from y=161 to y=191

#define CARS_LEFT_X_START 72
#define CARS_LEFT_Y_START 179
#define CARS_LEFT_WIDTH 32  // from x=72 to x=104
#define CARS_LEFT_HEIGHT 9  // from y=179 to y=188

#define DAY_X_START 56
#define DAY_Y_START 179
#define DAY_WIDTH 8    // from x=56 to x=64
#define DAY_HEIGHT 9   // from y=179 to y=188

#define NUM_ENEMY_CARS 1
#define ROAD_WIDTH 90.0f
#define CAR_WIDTH 16.0f
#define PLAYER_CAR_LENGTH 11.0f
#define ENEMY_CAR_LENGTH 11.0f
#define MAX_SPEED 100.0f
#define MIN_SPEED -10.0f
#define SPEED_INCREMENT 5.0f
#define MAX_Y_POSITION ACTION_SCREEN_HEIGHT + ENEMY_CAR_LENGTH // Max Y for enemy cars
#define MIN_Y_POSITION 0.0f // Min Y for enemy cars (spawn just above the screen)
#define MIN_DISTANCE_BETWEEN_CARS 40.0f  // Minimum Y distance between adjacent enemy cars

#define THROTTLE_MAX 120 // Maximum allowed throttle
#define MY_YELLOW  (Color){ 255, 255, 0, 255 }  // RGBA for yellow
#define MY_RED     (Color){ 255, 0, 0, 255 }    // RGBA for red
#define DAY_COLOR (Color){ 255, 255, 255, 255 }  // RGBA for day color (white)

#define PASS_THRESHOLD ACTION_SCREEN_HEIGHT  // Distance for passed cars to disappear

// Forward declarations
typedef struct CEnduro CEnduro;
typedef struct Client Client;

// Function declarations
extern void init_env(CEnduro *env);
extern CEnduro* allocate(void);
extern void free_initialized(CEnduro *env);
extern void free_allocated(CEnduro *env);
extern void compute_observations(CEnduro *env);
extern void spawn_enemy_cars(CEnduro *env);
extern bool step(CEnduro *env, int action);
extern void apply_weather_conditions(CEnduro *env);
extern void reset_env(CEnduro *env);
extern Client *make_client(CEnduro *env);
extern void render(Client *client, CEnduro *env);
extern void render_hud(CEnduro *env);
extern void close_client(Client *client);


#endif // End of RACING_H