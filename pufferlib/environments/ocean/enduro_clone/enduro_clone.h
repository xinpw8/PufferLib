// enduro_clone.h

#ifndef ENDURO_CLONE_H
#define ENDURO_CLONE_H

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <stddef.h>
#include <string.h>
#include "raylib.h"

// Constant defs
#define TARGET_FPS 60 // Used to calculate wiggle spawn frequency
#define LOG_BUFFER_SIZE 1024
#define SCREEN_WIDTH 160
#define SCREEN_HEIGHT 210
#define PLAYABLE_AREA_TOP 0
#define PLAYABLE_AREA_BOTTOM 154
#define PLAYABLE_AREA_LEFT 8
#define PLAYABLE_AREA_RIGHT 160
#define ACTION_HEIGHT (PLAYABLE_AREA_BOTTOM - PLAYABLE_AREA_TOP)
#define CAR_WIDTH 16
#define CAR_HEIGHT 11
#define MAX_ENEMIES 10
#define CRASH_NOOP_DURATION 120 // 60 // How long controls are disabled after car v car collision
#define INITIAL_CARS_TO_PASS 2 // 200
#define VANISHING_POINT_X 86 // 110
#define VANISHING_POINT_Y 52
#define VANISHING_POINT_TRANSITION_SPEED 1.0f
#define CURVE_TRANSITION_SPEED 0.05f // 0.02f
#define LOGICAL_VANISHING_Y (VANISHING_POINT_Y + 12)  // Separate logical vanishing point for cars disappearing
#define INITIAL_PLAYER_X  86.0f // ((PLAYABLE_AREA_RIGHT - PLAYABLE_AREA_LEFT)/2 + CAR_WIDTH/2)
#define PLAYER_MAX_Y (ACTION_HEIGHT - CAR_HEIGHT) // Max y is car length from bottom
#define PLAYER_MIN_Y (ACTION_HEIGHT - CAR_HEIGHT - 9) // Min y is 2 car lengths from bottom
#define ACCELERATION_RATE 0.05f
#define DECELERATION_RATE 0.1f
#define MIN_SPEED -2.5f
#define MAX_SPEED 4.5f // 2.5f
// Curve constants
#define CURVE_STRAIGHT 0
#define CURVE_LEFT -1
#define CURVE_RIGHT 1
#define NUM_LANES 3
// Rendering constants
// Number of digits in the scoreboard
#define SCORE_DIGITS 5
#define DAY_DIGITS   1
#define CARS_DIGITS  4
// Digit dimensions
#define DIGIT_WIDTH 8
#define DIGIT_HEIGHT 9
// Magic numbers - don't change
#define PLAYER_MIN_X 65.5f
#define PLAYER_MAX_X 91.5f
#define ROAD_LEFT_OFFSET 50.0f
#define ROAD_RIGHT_OFFSET 51.0f
#define VANISHING_POINT_X_LEFT 110.0f
#define VANISHING_POINT_X_RIGHT 62.0f
#define PLAYABLE_AREA_BOTTOM 154
#define VANISHING_POINT_Y 52
#define CURVE_VANISHING_POINT_SHIFT 55.0f
#define CURVE_PLAYER_SHIFT_FACTOR 0.025f // Moves player car towards outside edge of curves
// Constants for wiggle effect timing and amplitude
#define WIGGLE_AMPLITUDE 10.0f // 8.0f              // Maximum 'bump-in' offset in pixels
#define WIGGLE_SPEED 10.1f // 10.1f                 // Speed at which the wiggle moves down the screen
#define WIGGLE_LENGTH 26.0f // 26.0f                // Vertical length of the wiggle effect

// Log structs
typedef struct Log {
    float episode_return;
    float episode_length;
    float score;
} Log;

typedef struct LogBuffer {
    Log* logs;
    int length;
    int idx;
} LogBuffer;

// Car struct for enemy cars
typedef struct Car {
    int lane;   // Lane index: 0=left lane, 1=mid, 2=right lane
    float y;    // Current y position
    int passed; // Flag to indicate if car has been passed by player
    int colorIndex; // Car color idx (0-5)
} Car;

typedef enum {
    GAME_STAGE_DAY_START,
    GAME_STAGE_NIGHT,
    GAME_STAGE_GRASS_AFTER_SNOW,
    // Add more stages as needed
} GameStage;

// Rendering struct
typedef struct GameState {
    Texture2D backgroundTextures[16]; // 16 different backgrounds for time of day
    Texture2D digitTextures[10];      // Textures for digits 0-9
    Texture2D carDigitTexture;        // Texture for the "CAR" digit
    Texture2D mountainTextures[16];   // Mountain textures corresponding to backgrounds
    Texture2D levelCompleteFlagLeftTexture;  // Texture for left flag
    Texture2D levelCompleteFlagRightTexture; // Texture for right flag
    Texture2D greenDigitTextures[10];        // Textures for green digits
    Texture2D yellowDigitTextures[10];       // Textures for yellow digits
    Texture2D playerCarLeftTreadTexture;
    Texture2D playerCarRightTreadTexture;
    Texture2D enemyCarTextures[6][2]; // [color][tread] 6 colors, 2 treads (left, right)
    Texture2D enemyCarNightTailLightsTexture;
    Texture2D enemyCarNightFogTailLightsTexture;
    // For car animation
    float carAnimationTimer;
    float carAnimationInterval;
    bool showLeftTread;
    int currentBackgroundIndex;
    int previousBackgroundIndex;
    int score;
    int day;
    int carsToPass;
    float mountainPosition; // Position of the mountain texture
    bool victoryAchieved;   // Flag to indicate victory condition
    // Background state vars
    // float elapsedTime;       // Total elapsed time in seconds
    int backgroundIndex;     // Current background index
    // Transition times in seconds
    float backgroundTransitionTimes[16];
    // Variables for alternating flags
    int flagTimer;
    bool showLeftFlag;
    // Variables for scrolling yellow digits
    float yellowDigitOffset; // Offset for scrolling effect
    int yellowDigitCurrent;  // Current yellow digit being displayed
    int yellowDigitNext;     // Next yellow digit to scroll in
    // Variables for scrolling digits
    float scoreDigitOffsets[SCORE_DIGITS];   // Offset for scrolling effect for each digit
    int scoreDigitCurrents[SCORE_DIGITS];    // Current digit being displayed for each position
    int scoreDigitNexts[SCORE_DIGITS];       // Next digit to scroll in for each position
    bool scoreDigitScrolling[SCORE_DIGITS];  // Scrolling state for each digit
    int scoreTimer; // Timer to control score increment
    int victoryDisplayTimer; // To track how long the victory flags have been displayed
} GameState;

// Game environment struct
typedef struct Enduro {
    float* observations;
    int32_t actions;  // int32_t
    float* rewards;
    unsigned char* terminals;
    unsigned char* truncateds;
    LogBuffer* log_buffer;
    Log log;
    float width;
    float height;
    float hud_height;
    float car_width;
    float car_height;
    int max_enemies;
    float crash_noop_duration;
    float elapsedTime;
    int initial_cars_to_pass;
    float min_speed;
    float max_speed;
    float player_x;
    float player_y;
    float speed;
    // ints
    int score;
    int day;
    int lane;
    int step_count;
    int numEnemies;
    int carsToPass;
    float collision_cooldown;
    float collision_invulnerability_timer; // Disables collisions during/after crash
    int drift_direction; // Which way player car drifts whilst noops after crash w/ other car
    float action_height;
    Car enemyCars[MAX_ENEMIES];
    float initial_player_x;
    float road_scroll_offset;
    // Road curve variables
    int current_curve_direction; // 1: Right, -1: Left, 0: Straight
    float current_curve_factor;
    float target_curve_factor;
    float target_vanishing_point_x;     // Next curve direction vanishing point
    float current_vanishing_point_x;    // Current interpolated vanishing point
    float base_target_vanishing_point_x; // Target for the base vanishing point
    float vanishing_point_x;
    float base_vanishing_point_x;
    float t_p;
    // Roadside wiggle effect
    float wiggle_y;            // Current y position of the wiggle
    float wiggle_speed;        // Speed at which the wiggle moves down the screen
    float wiggle_length;       // Vertical length of the wiggle effect
    float wiggle_amplitude;    // How far into road wiggle extends
    bool wiggle_active;        // Whether the wiggle is active
    // Player car acceleration
    int currentGear;
    float gearSpeedThresholds[4]; // Speeds at which gear changes occur
    float gearAccelerationRates[4]; // Acceleration rates per gear
    float gearTimings[4]; // Time durations per gear
    float gearElapsedTime; // Time spent in current gear
    // Enemy spawning
    GameStage currentStage;    // Enemy spawn timer
    float enemySpawnTimer;
    float enemySpawnInterval; // Spawn interval based on current stage
    float elapsedStageTime;   // Time elapsed in current stage

    // Logging
    float last_road_left;
    float last_road_right;
    float totalAccelerationTime; // Debug accel

    // Mountain rendering
    float parallaxFactor;
    // Game state
    GameState gameState;
} Enduro;

// Action enumeration
typedef enum {
    ACTION_NOOP = 0,
    ACTION_FIRE = 1,
    ACTION_RIGHT = 2,
    ACTION_LEFT = 3,
    ACTION_DOWN = 4,
    ACTION_DOWNRIGHT = 5,
    ACTION_DOWNLEFT = 6,
    ACTION_RIGHTFIRE = 7,
    ACTION_LEFTFIRE = 8,
} Action;

// Client struct
typedef struct Client {
    float width;
    float height;
    Color player_color;
    Color enemy_color;
    Color road_color;
} Client;

// Prototypes
// LogBuffer functions
LogBuffer* allocate_logbuffer(int size);
void free_logbuffer(LogBuffer* buffer);
void add_log(LogBuffer* logs, Log* log);
Log aggregate_and_clear(LogBuffer* logs);

// Environment functions
void init(Enduro* env);
void allocate(Enduro* env);
void free_allocated(Enduro* env);
void reset(Enduro* env);
void reset_round(Enduro* env);
void step(Enduro* env);

// Client functions
Client* make_client(Enduro* env);
void close_client(Client* client);

// Event handling
void handleEvents(int* running, Enduro* env);

// Rendering functions
void render(Client* client, Enduro* env);
void render_car(Client* client, Enduro* env);

// GameState functions
void initRaylib();
void loadTextures(GameState* gameState);
void cleanup(GameState* gameState);
void updateBackground(GameState* gameState, Enduro* env);
void updateCarAnimation(GameState* gameState, Enduro* env);
void renderBackground(GameState* gameState);
void renderScoreboard(GameState* gameState);
void updateMountains(GameState* gameState, Enduro* env);
void renderMountains(GameState* gameState, Enduro* env);
void updateVictoryEffects(GameState* gameState);
void updateScore(GameState* gameState);

// Additional function prototypes
float road_edge_x(Enduro* env, float y, float offset, bool left);
float car_x_in_lane(Enduro* env, int lane, float y);
int calculate_which_half_of_lane(Enduro* env);
void update_road_curve(Enduro* env);
void update_vanishing_point(Enduro* env, float offset);

// Function defs
// LogBuffer functions
LogBuffer* allocate_logbuffer(int size) {
    LogBuffer* logs = (LogBuffer*)calloc(1, sizeof(LogBuffer));
    logs->logs = (Log*)calloc(size, sizeof(Log));
    logs->length = size;
    logs->idx = 0;
    return logs;
}

void free_logbuffer(LogBuffer* buffer) {
    free(buffer->logs);
    free(buffer);
}

void add_log(LogBuffer* logs, Log* log) {
    if (logs->idx == logs->length) {
        return;
    }
    logs->logs[logs->idx] = *log;
    logs->idx += 1;
    printf("Log: %f, %f, %f\n", log->episode_return, log->episode_length, log->score);
}

Log aggregate_and_clear(LogBuffer* logs) {
    Log log = {0};
    if (logs->idx == 0) {
        return log;
    }
    for (int i = 0; i < logs->idx; i++) {
        log.episode_return += logs->logs[i].episode_return;
        log.episode_length += logs->logs[i].episode_length;
        log.score += logs->logs[i].score;
    }
    log.episode_return /= logs->idx;
    log.episode_length /= logs->idx;
    log.score /= logs->idx;
    logs->idx = 0;
    return log;
}

void init(Enduro* env) {
    env->step_count = 0;
    env->score = 0;
    env->numEnemies = 0;
    env->collision_cooldown = 0.0;
    env->action_height = ACTION_HEIGHT;
    env->elapsedTime = 0.0f;

    // Enemy spawning
    env->currentStage = GAME_STAGE_DAY_START;
    env->enemySpawnTimer = 0.0f;
    env->enemySpawnInterval = 0.8777f; // Spawn interval in seconds

    // Logging
    env->totalAccelerationTime = 0.0f; // Debug accel

    // Set vanishing point to center (86) explicitly
    env->base_vanishing_point_x = 86.0f;
    env->current_vanishing_point_x = 86.0f;
    env->target_vanishing_point_x = 86.0f;
    env->vanishing_point_x = 86.0f;  // Start centered at 86

    // Set player position to center (86) explicitly
    env->initial_player_x = 86.0f;
    env->player_x = env->initial_player_x;
    printf("Init - vanishing_point_x: %f, player_x: %f\n", env->vanishing_point_x, env->player_x); // Debug output

    env->player_y = PLAYER_MAX_Y;
    env->speed = env->min_speed;
    env->initial_cars_to_pass = INITIAL_CARS_TO_PASS;
    env->carsToPass = env->initial_cars_to_pass;
    env->day = 1;

    // Reset curve factors
    env->current_curve_direction = CURVE_STRAIGHT;
    env->current_curve_factor = 0.0f;
    env->target_curve_factor = 0.0f;

    // Wiggle effect initialization
    env->wiggle_y = VANISHING_POINT_Y;
    env->wiggle_speed = WIGGLE_SPEED;
    env->wiggle_length = WIGGLE_LENGTH;
    env->wiggle_amplitude = WIGGLE_AMPLITUDE;
    env->wiggle_active = true;

    // Sync with GameState
    env->gameState.carsToPass = env->carsToPass;
    env->gameState.victoryAchieved = false;

    // Player car acceleration
        env->currentGear = 0;
    env->gearElapsedTime = 0.0f;

    // Define gear timings
    env->gearTimings[0] = 4.0f;
    env->gearTimings[1] = 2.5f;
    env->gearTimings[2] = 3.25f;
    env->gearTimings[3] = 1.5f;

    // Calculate speed thresholds and acceleration rates
    float totalSpeedRange = env->max_speed - env->min_speed;
    float totalTime = 0.0f;
    for (int i = 0; i < 4; i++) {
        totalTime += env->gearTimings[i];
    }

    float cumulativeSpeed = env->min_speed;
    for (int i = 0; i < 4; i++) {
        float gearTime = env->gearTimings[i];
        float gearSpeedIncrement = totalSpeedRange * (gearTime / totalTime);
        env->gearSpeedThresholds[i] = cumulativeSpeed + gearSpeedIncrement;
        env->gearAccelerationRates[i] = gearSpeedIncrement / (gearTime * TARGET_FPS); // per frame
        cumulativeSpeed = env->gearSpeedThresholds[i];
    }
}

void allocate(Enduro* env) {
    init(env);
    env->observations = (float*)calloc(8 + 2 * MAX_ENEMIES, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->truncateds = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->log_buffer = allocate_logbuffer(LOG_BUFFER_SIZE);
    srand(time(NULL));
}

void free_allocated(Enduro* env) {
    free(env->observations);
    free(env->rewards);
    free(env->terminals);
    free(env->truncateds);
    free_logbuffer(env->log_buffer);
}

void reset_round(Enduro* env) {
    env->player_x = env->initial_player_x;
    env->player_y = PLAYER_MAX_Y;
    env->speed = env->min_speed;
    env->numEnemies = 0;
    env->step_count = 0;
    env->collision_cooldown = 0;
    env->road_scroll_offset = 0.0f;
}

// Reset all init vars
void reset(Enduro* env) {
// Environment functions
    env->step_count = 0;
    env->score = 0;
    env->numEnemies = 0;
    env->collision_cooldown = 0.0;
    env->action_height = ACTION_HEIGHT;
    // Initial x value of center of player car
    env->initial_player_x = INITIAL_PLAYER_X;
    env->player_x = env->initial_player_x;
    env->base_vanishing_point_x = VANISHING_POINT_X;
    env->current_vanishing_point_x = VANISHING_POINT_X;
    env->target_vanishing_point_x = VANISHING_POINT_X;
    env->vanishing_point_x = VANISHING_POINT_X;

    // Initialize player y position
    env->player_y = PLAYER_MAX_Y;
    // Speed-related fields
    env->min_speed = MIN_SPEED;
    env->max_speed = MAX_SPEED;
    env->speed = env->min_speed;
    env->currentGear = 0;
    env->gearElapsedTime = 0.0f;
    // Enemy spawning
    env->elapsedStageTime = 0.0f;
    env->initial_cars_to_pass = INITIAL_CARS_TO_PASS;
    env->carsToPass = env->initial_cars_to_pass;
    printf("initial cars to pass, carstopass, INITIAL_CARS_TO_PASS: %d, %d, %d\n", env->initial_cars_to_pass, env->carsToPass, INITIAL_CARS_TO_PASS);
    env->day = 1;
    for (int i = 0; i < MAX_ENEMIES; i++) {
        env->enemyCars[i].lane = 0;
        env->enemyCars[i].y = 0.0;
        env->enemyCars[i].passed = 0;
    }
    env->road_scroll_offset = 0.0f;
    if (env->log_buffer != NULL) {
        env->log_buffer->idx = 0;
    }
    env->log.episode_return = 0.0;
    env->log.episode_length = 0.0;
    env->log.score = 0.0;
    env->current_curve_direction = 0;
    // Initialize curve variables
    env->current_curve_factor = 0.0f;
    env->target_curve_factor = 0.0f;
    // Wiggle effect initialization
    env->wiggle_y = VANISHING_POINT_Y;  // Start at the vanishing point
    env->wiggle_speed = WIGGLE_SPEED;           // Adjust as needed (doesn't matter??)
    env->wiggle_length = WIGGLE_LENGTH; // PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y; // playable area   50.0f;  // Vertical size of the wiggle
    env->wiggle_amplitude = WIGGLE_AMPLITUDE;       // Maximum 'bump-in' offset in pixels
    env->wiggle_active = true;
    // Synchronize carsToPass with GameState
    env->gameState.carsToPass = env->carsToPass;
    env->gameState.victoryAchieved = false; // Initialize victory condition
    // Reset rewards and logs
    env->rewards[0] = 0;
    add_log(env->log_buffer, &env->log);
    printf("Game reset for day %d with %d cars to pass.\n", env->day, env->carsToPass);
}


bool check_collision(Enduro* env, Car* car) {
    // Compute the scale factor based on vanishing point reference
    float depth = (car->y - VANISHING_POINT_Y) / (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y);
    float scale = fmax(0.1f, 0.9f * depth);
    float car_width = CAR_WIDTH * scale;
    float car_height = CAR_HEIGHT * scale;
    // Compute car x position
    float car_center_x = car_x_in_lane(env, car->lane, car->y);
    float car_x = car_center_x - car_width / 2.0f;
    return !(env->player_x > car_x + car_width ||
             env->player_x + CAR_WIDTH < car_x ||
             env->player_y > car->y + car_height ||
             env->player_y + CAR_HEIGHT < car->y);
}

// Determines which of the 3 lanes the player's car is in
int get_player_lane(Enduro* env) {
    float player_center_x = env->player_x + CAR_WIDTH / 2.0f;
    float offset = (env->player_x - env->initial_player_x) * 0.5f;
    float left_edge = road_edge_x(env, env->player_y, offset, true);
    float right_edge = road_edge_x(env, env->player_y, offset, false);
    float lane_width = (right_edge - left_edge) / 3.0f;
    // printf("Player center x: %.2f, left edge: %.2f, right edge: %.2f, lane width: %.2f\n", player_center_x, left_edge, right_edge, lane_width);
    // printf("player_center_x - left_edge/lane_width=%.2f\n", (player_center_x - left_edge) /lane_width);
    env->lane = (int)((player_center_x - left_edge) / lane_width);
        // printf("Player lane: %d\n", lane);
    if (env->lane < 0) env->lane = 0;
    if (env->lane > 2) env->lane = 2;

    return env->lane;
}

int calculate_which_half_of_lane(Enduro* env) {
    // Player in the middle lane, choose based on player’s half
    int lane = 1;
    float player_center_x = env->player_x + CAR_WIDTH / 2.0f;
    float road_center_x = (road_edge_x(env, env->player_y, 0, true) +
                            road_edge_x(env, env->player_y, 0, false)) / 2.0f;
    if (player_center_x < road_center_x) {
        lane = 2;
    } else {
        lane = 0;
    }
    return lane;
}

void add_enemy_car(Enduro* env) {
    if (env->numEnemies >= MAX_ENEMIES) return;

    int player_lane = get_player_lane(env);
    int lane_to_avoid = player_lane;
    int drift_lane = -1;

    // Determine the lane the player will be in due to drift during collision cooldown
    if (env->collision_cooldown > 0) {
        if (env->drift_direction == -1) {
            // Player is drifting left
            if (player_lane > 0) drift_lane = player_lane - 1;
        } else if (env->drift_direction == 1) {
            // Player is drifting right
            if (player_lane < NUM_LANES - 1) drift_lane = player_lane + 1;
        }
    }

    int possible_lanes[NUM_LANES];
    int num_possible_lanes = 0;

    if (env->speed < 0) {
        // Player is moving backward
        // Enemy cars should never spawn in the middle lane
        for (int i = 0; i < NUM_LANES; i++) {
            if (i != 1 && i != lane_to_avoid && i != drift_lane) {
                possible_lanes[num_possible_lanes++] = i;
            }
        }
    } else {
        // Player is moving forward
        for (int i = 0; i < NUM_LANES; i++) {
            if (i != drift_lane) {
                possible_lanes[num_possible_lanes++] = i;
            }
        }
    }

    // If no possible lanes, do not spawn a car (rare)
    if (num_possible_lanes == 0) {
        return;
    }

    // Randomly select a lane from possible_lanes
    int random_index = rand() % num_possible_lanes;
    int lane = possible_lanes[random_index];

    // Configure enemy car properties
    Car car = { .lane = lane, .passed = false, .colorIndex = rand() % 6 };
    if (env->speed > 0) {
        // Spawn cars slightly below the vanishing point when moving forward
        car.y = VANISHING_POINT_Y + 10.0f;
    } else {
        // Spawn cars off-screen at the bottom when moving backward
        car.y = PLAYABLE_AREA_BOTTOM + CAR_HEIGHT;
    }

    // Ensure minimum spacing between cars in the same lane
    float depth = (car.y - VANISHING_POINT_Y) / (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y);
    float scale = fmax(0.1f, 0.9f * depth);
    float scaled_car_length = CAR_HEIGHT * scale;
    float min_spacing = 3.0f * scaled_car_length;

    for (int i = 0; i < env->numEnemies; i++) {
        Car* existing_car = &env->enemyCars[i];
        if (existing_car->lane == lane) {
            float y_distance = fabs(existing_car->y - car.y);
            if (y_distance < min_spacing) {
                // Too close, do not spawn this car
                printf("Not spawning car due to spacing in same lane (distance %.2f, min spacing %.2f)\n", y_distance, min_spacing);
                return;
            }
        }
    }

    // Ensure not occupying all lanes within vertical range
    float min_vertical_range = 6.0f * CAR_HEIGHT; // 6 car lengths

    // Count number of lanes occupied within min_vertical_range of car.y
    int lanes_occupied = 0;
    bool lane_occupied[NUM_LANES] = { false };

    for (int i = 0; i < env->numEnemies; i++) {
        Car* existing_car = &env->enemyCars[i];
        float y_distance = fabs(existing_car->y - car.y);
        if (y_distance < min_vertical_range) {
            lane_occupied[existing_car->lane] = true;
        }
    }

    // Count occupied lanes
    for (int i = 0; i < NUM_LANES; i++) {
        if (lane_occupied[i]) {
            lanes_occupied++;
        }
    }

    if (lanes_occupied >= NUM_LANES - 1 && !lane_occupied[lane]) {
        // Do not spawn if this car would cause all lanes to be occupied within the vertical range
        printf("Not spawning car because it would occupy all lanes within vertical range.\n");
        return;
    }

    // Add the enemy car to the environment
    env->enemyCars[env->numEnemies++] = car;
}

// Adjust base vanishing point with offset during curves
void update_vanishing_point(Enduro* env, float offset) {
    env->vanishing_point_x = env->base_vanishing_point_x + offset;
    printf("Vanishing point updated to %.2f\n", env->vanishing_point_x);
}

void step(Enduro* env) {
    env->elapsedTime += (1.0f / TARGET_FPS); // Increment elapsed time by frame duration
    updateBackground(&env->gameState, env);

    update_road_curve(env);
    env->log.episode_length += 1;
    env->terminals[0] = 0;
    env->road_scroll_offset += env->speed;

    // Update enemy cars positions
    env->elapsedStageTime += (1.0f / TARGET_FPS); // Time elapsed in stage
    for (int i = 0; i < env->numEnemies; i++) {
        Car* car = &env->enemyCars[i];
        car->y += 0.5 * env->speed;
    }

    // Calculate road edges
    float road_left = road_edge_x(env, env->player_y, 0, true);
    float road_right = road_edge_x(env, env->player_y, 0, false) - CAR_WIDTH;
    env->last_road_left = road_left;
    env->last_road_right = road_right;

    // Player movement logic
    if (env->collision_cooldown <= 0) {
        int act = env->actions;
        switch (act) {
            case ACTION_NOOP:
                break;
            case ACTION_FIRE:
                // Accelerate based on current gear
                if (env->speed < env->max_speed) {
                    // Update gear if necessary
                    if (env->speed >= env->gearSpeedThresholds[env->currentGear] && env->currentGear < 3) {
                        env->currentGear++;
                        env->gearElapsedTime = 0.0f;
                        // printf("Gear shifted up to %d at time %.2f seconds\n", env->currentGear + 1, env->totalAccelerationTime);
                    }
                    // Accelerate
                    env->speed += env->gearAccelerationRates[env->currentGear];
                    if (env->speed > env->gearSpeedThresholds[env->currentGear]) {
                        env->speed = env->gearSpeedThresholds[env->currentGear];
                    }
                    // Log speed and gear
                    // printf("Time: %.2f s, Speed: %.2f, Gear: %d\n", env->totalAccelerationTime, env->speed, env->currentGear + 1);

                    // Update total acceleration time
                    env->totalAccelerationTime += (1.0f / TARGET_FPS);
                }
                break;
            case ACTION_RIGHT:
                env->player_x += 0.5f;
                if (env->player_x > road_right) env->player_x = road_right;
                break;
            case ACTION_LEFT:
                env->player_x -= 0.5f;
                if (env->player_x < road_left) env->player_x = road_left;
                break;
            case ACTION_DOWN:
                if (env->speed > env->min_speed) env->speed -= DECELERATION_RATE; // Decelerate
                break;
            case ACTION_DOWNRIGHT:
                if (env->speed > env->min_speed) env->speed -= DECELERATION_RATE;
                env->player_x += 0.5f;
                if (env->player_x > road_right) env->player_x = road_right;
                break;
            case ACTION_DOWNLEFT:
                if (env->speed > env->min_speed) env->speed -= DECELERATION_RATE;
                env->player_x -= 0.5f;
                if (env->player_x < road_left) env->player_x = road_left;
                break;
            case ACTION_RIGHTFIRE:
                if (env->speed < env->max_speed) env->speed += ACCELERATION_RATE;
                env->player_x += 0.5f;
                if (env->player_x > road_right) env->player_x = road_right;
                break;
            case ACTION_LEFTFIRE:
                if (env->speed < env->max_speed) env->speed += ACCELERATION_RATE;
                env->player_x -= 0.5f;
                if (env->player_x < road_left) env->player_x = road_left;
                break;
        }
    } else {
    env->collision_cooldown -= 1;

    // Collision cooldown debugging print
    if (((int)round(env->collision_cooldown)) % 5 == 0) {
    printf("Collision cooldown: %.2f\n", env->collision_cooldown);
    }

    // Apply drift based on initial side
    // Drift left if on right, right if on left
    if (env->drift_direction == 0) { // drift_direction is 0 when noop starts
        env->drift_direction = (env->player_x > (road_left + road_right) / 2) ? -1 : 1;
    }
    env->player_x += env->drift_direction * 0.25f;
    }

    // Adjust player's x position based on the current curve
    float curve_shift = -env->current_curve_factor * CURVE_PLAYER_SHIFT_FACTOR * abs(env->speed);
    env->player_x += curve_shift;

    // Clamp player x position to within road edges
    if (env->player_x < road_left) env->player_x = road_left;
    if (env->player_x > road_right) env->player_x = road_right;

    // Update player's horizontal position ratio, t_p
    float t_p = (env->player_x - PLAYER_MIN_X) / (PLAYER_MAX_X - PLAYER_MIN_X);
    t_p = fmaxf(0.0f, fminf(1.0f, t_p));
    env->t_p = t_p;

    // Base vanishing point based on player's horizontal movement (without curve)
    env->base_vanishing_point_x = VANISHING_POINT_X_LEFT - t_p * (VANISHING_POINT_X_LEFT - VANISHING_POINT_X_RIGHT);

    // Adjust vanishing point based on current curve
    float curve_vanishing_point_shift = env->current_curve_direction * CURVE_VANISHING_POINT_SHIFT;
    env->vanishing_point_x = env->base_vanishing_point_x + curve_vanishing_point_shift;

    if (env->wiggle_active) {
        // Define wiggle period range
        float min_wiggle_period = 5.8f; // Slow wiggle period at min speed
        float max_wiggle_period = 0.3f; // Fast wiggle period at max speed
        // Apply non-linear scaling for wiggle period using square root
        float speed_normalized = (env->speed - env->min_speed) / (env->max_speed - env->min_speed);
        speed_normalized = fmaxf(0.0f, fminf(1.0f, speed_normalized));  // Clamp between 0 and 1
        // Adjust wiggle period with non-linear scale
        float current_wiggle_period = min_wiggle_period - powf(speed_normalized, 0.25) * (min_wiggle_period - max_wiggle_period);
        // Calculate wiggle speed based on adjusted wiggle period
        env->wiggle_speed = (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y) / (current_wiggle_period * TARGET_FPS);
        // Update wiggle position
        env->wiggle_y += env->wiggle_speed;
        // Reset wiggle when it reaches the bottom
        if (env->wiggle_y > PLAYABLE_AREA_BOTTOM) {
            env->wiggle_y = VANISHING_POINT_Y;
        }
        // printf("Speed: %.2f, Normalized Speed: %.2f, Wiggle Period: %.2f, Wiggle Speed: %.2f\n", 
            // env->speed, speed_normalized, current_wiggle_period, env->wiggle_speed);
    }

    // Update player y position based on speed
    env->player_y = PLAYER_MAX_Y - (env->speed - env->min_speed) / (env->max_speed - env->min_speed) * (PLAYER_MAX_Y - PLAYER_MIN_Y);
    // Clamp player_y
    if (env->player_y < PLAYER_MIN_Y) env->player_y = PLAYER_MIN_Y;
    if (env->player_y > PLAYER_MAX_Y) env->player_y = PLAYER_MAX_Y;

    // Enemy car logic
    for (int i = 0; i < env->numEnemies; i++) {
        Car* car = &env->enemyCars[i];

        // Check for passing logic **only if not in collision cooldown**
        if (env->speed > 0 && car->y > env->player_y + CAR_HEIGHT && !car->passed && env->collision_cooldown <= 0) {
            // Mark car as passed and decrement carsToPass
            env->carsToPass--;
            if (env->carsToPass < 0) env->carsToPass = 0; // Stops at 0; resets to 300 on new day
            car->passed = true;
            env->score += 10;
            env->rewards[0] += 1;
            // Passing debug
            // printf("Car passed at y = %.2f. Remaining cars to pass: %d\n", car->y, env->carsToPass);
        }

        if (env->collision_cooldown > 0) {
            // Drifting logic
            if (env->collision_cooldown <= 0) {
                // Start invulnerability timer (1 second)
                env->collision_invulnerability_timer = TARGET_FPS * 1.0f; // 1 second in frames
                printf("Invulnerability timer started\n");
            }
        } else if (env->collision_invulnerability_timer > 0) {
            env->collision_invulnerability_timer -= 1;
        }

        // Check for and handle collisions between player and enemy cars
        if (env->collision_cooldown <= 0 && env->collision_invulnerability_timer <= 0) {
            if (check_collision(env, car)) {
                env->speed = env->min_speed = 1 + MIN_SPEED;
                env->collision_cooldown = CRASH_NOOP_DURATION;
                env->drift_direction = 0; // Reset drift direction
            }
        }

        // Remove off-screen cars that move below the screen
        if (car->y > PLAYABLE_AREA_BOTTOM + CAR_HEIGHT * 5) {
            // Remove car from array if it moves below the screen
            for (int j = i; j < env->numEnemies - 1; j++) {
                env->enemyCars[j] = env->enemyCars[j + 1];
            }
            env->numEnemies--;
            i--;
            // printf("Removing car %d as it went below screen\n", i);
        }

        // Remove cars that reach or surpass the logical vanishing point if moving up (player speed negative)
        if (env->speed < 0 && car->y <= LOGICAL_VANISHING_Y) {
            // Remove car from array if it reaches the logical vanishing point if moving down (player speed positive)
            for (int j = i; j < env->numEnemies - 1; j++) {
                env->enemyCars[j] = env->enemyCars[j + 1];
            }
            env->numEnemies--;
            i--;
            // printf("Removing car %d as it reached the logical vanishing point at LOGICAL_VANISHING_Y = %.2f\n", i, (float)LOGICAL_VANISHING_Y);
        }
    }

    // Calculate enemy spawn interval based on player speed and day
    float min_spawn_interval = 0.8777f; // Minimum spawn interval (from current code)
    float max_spawn_interval;
    int dayIndex = env->day - 1;
    float maxSpawnIntervals[] = {0.6667f, 0.3614f, 0.5405f};
    int numMaxSpawnIntervals = sizeof(maxSpawnIntervals) / sizeof(maxSpawnIntervals[0]);

    if (dayIndex < numMaxSpawnIntervals) {
        max_spawn_interval = maxSpawnIntervals[dayIndex];
    } else {
        // For days beyond the observed, decrease max_spawn_interval further
        max_spawn_interval = maxSpawnIntervals[numMaxSpawnIntervals - 1] - (dayIndex - numMaxSpawnIntervals + 1) * 0.05f;
        if (max_spawn_interval < 0.2f) {
            max_spawn_interval = 0.2f; // Do not go below 0.2 seconds
        }
    }

    // Ensure min_spawn_interval is greater than or equal to max_spawn_interval
    if (min_spawn_interval < max_spawn_interval) {
        min_spawn_interval = max_spawn_interval;
    }

    // Calculate speed factor
    float speed_factor = (env->speed - env->min_speed) / (env->max_speed - env->min_speed);
    if (speed_factor < 0.0f) speed_factor = 0.0f;
    if (speed_factor > 1.0f) speed_factor = 1.0f;

    // Interpolate between min and max spawn intervals
    env->enemySpawnInterval = min_spawn_interval - speed_factor * (min_spawn_interval - max_spawn_interval);


    // Update enemy spawn timer
    env->enemySpawnTimer += (1.0f / TARGET_FPS);
    if (env->enemySpawnTimer >= env->enemySpawnInterval) {
        env->enemySpawnTimer -= env->enemySpawnInterval;
        if (env->numEnemies < MAX_ENEMIES) {
            add_enemy_car(env);
            // Log enemy spawning
            // printf("Enemy car spawned at time %.2f seconds\n", env->elapsedTime);
        }
    }

    // Update Victory Effects
    updateVictoryEffects(&env->gameState);

    // Handle victory condition
    if (env->carsToPass <= 0 && !env->gameState.victoryAchieved) {
        // Set victory condition
        env->gameState.victoryAchieved = true;
        printf("Victory achieved! Continuing until day ends.\n");
    }

    // Handle day transition when background cycles back to 0
    if (env->gameState.currentBackgroundIndex == 0 && env->gameState.previousBackgroundIndex == 15) {
        // Background cycled back to 0
        if (env->gameState.victoryAchieved) {
            // Player has achieved victory, start new day
            env->day++;
            env->carsToPass = 300; // Always 300 after the first day
            env->gameState.victoryAchieved = false;
            printf("Starting day %d with %d cars to pass.\n", env->day, env->carsToPass);
        } else {
            // Player failed to pass required cars, reset environment
            env->terminals[0] = 1; // Signal termination
            reset(env); // Reset the game
            printf("Day %d failed. Resetting game.\n", env->day);
            return;
        }
    }

    // Synchronize carsToPass between Enduro and GameState
    env->gameState.carsToPass = env->carsToPass;

    env->log.episode_return += env->rewards[0];
    env->step_count++;
    env->log.score = env->score;
}

// When to curve road and how to curve it, including dense smooth transitions
// An ugly dense function but it is necessary
void update_road_curve(Enduro* env) {
    static int current_curve_stage = 0;
    static int steps_in_current_stage = 0;
    
    // Map speed to the scale between 0.5 and 3.5
    float speed_scale = 0.5f + ((abs(env->speed) / env->max_speed) * (3.5f - 0.5f));

    // printf("speed=%.2f, speed_scale=%.2f, max_speed=%.2f\n", env->speed, speed_scale, env->max_speed);

    float vanishing_point_transition_speed = VANISHING_POINT_TRANSITION_SPEED + speed_scale; 
    // Steps to curve L, R, go straight for
    int step_thresholds[] = {350, 350, 350, 350, 350, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150};
    int curve_directions[] = {0, -1, 0, 1, 0, -1, 0, 1, -1, 1, 1, 1, 1, 1, 1, 1};    
    steps_in_current_stage++;
    if (steps_in_current_stage >= step_thresholds[current_curve_stage]) {
        env->target_curve_factor = (float)curve_directions[current_curve_stage];
        steps_in_current_stage = 0;
        current_curve_stage = (current_curve_stage + 1) % (sizeof(step_thresholds) / sizeof(int));
    }
    if (env->current_curve_factor < env->target_curve_factor) {
        env->current_curve_factor = fminf(env->current_curve_factor + CURVE_TRANSITION_SPEED, env->target_curve_factor);
    } else if (env->current_curve_factor > env->target_curve_factor) {
        env->current_curve_factor = fmaxf(env->current_curve_factor - CURVE_TRANSITION_SPEED, env->target_curve_factor);
    }
    env->current_curve_direction = fabsf(env->current_curve_factor) < 0.1f ? CURVE_STRAIGHT 
                             : (env->current_curve_factor > 0) ? CURVE_RIGHT : CURVE_LEFT;
    // Move the vanishing point gradually
    env->base_target_vanishing_point_x = VANISHING_POINT_X_LEFT - env->t_p * (VANISHING_POINT_X_LEFT - VANISHING_POINT_X_RIGHT);
    float target_shift = env->current_curve_direction * CURVE_VANISHING_POINT_SHIFT;
    env->target_vanishing_point_x = env->base_target_vanishing_point_x + target_shift;

    if (env->current_vanishing_point_x < env->target_vanishing_point_x) {
        env->current_vanishing_point_x = fminf(env->current_vanishing_point_x + vanishing_point_transition_speed, env->target_vanishing_point_x);
    } else if (env->current_vanishing_point_x > env->target_vanishing_point_x) {
        env->current_vanishing_point_x = fmaxf(env->current_vanishing_point_x - vanishing_point_transition_speed, env->target_vanishing_point_x);
    }
    env->vanishing_point_x = env->current_vanishing_point_x;
}

// B(t) = (1−t)^2 * P0​+2(1−t) * t * P1​+t^2 * P2​, t∈[0,1]
// Quadratic bezier curve helper function
float quadratic_bezier(float bottom_x, float control_x, float top_x, float t) {
    float one_minus_t = 1.0f - t;
    return one_minus_t * one_minus_t * bottom_x + 
           2.0f * one_minus_t * t * control_x + 
           t * t * top_x;
}

// Computes the edges of the road. Use for both L and R. 
// Lots of magic numbers to replicate as exactly as possible
// original Atari 2600 Enduro road rendering.
float road_edge_x(Enduro* env, float y, float offset, bool left) {
    float t = (PLAYABLE_AREA_BOTTOM - y) / (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y);
    float base_offset = left ? -ROAD_LEFT_OFFSET : ROAD_RIGHT_OFFSET;
    float bottom_x = env->base_vanishing_point_x + base_offset + offset;
    float top_x = env->current_vanishing_point_x + offset;    
    float edge_x;
    if (fabsf(env->current_curve_factor) < 0.01f) {
        // Straight road
        edge_x = bottom_x + t * (top_x - bottom_x);
    } else {
        // Adjust curve offset based on curve direction
        float curve_offset = (env->current_curve_factor > 0 ? -30.0f : 30.0f) * fabsf(env->current_curve_factor);
        float control_x = bottom_x + (top_x - bottom_x) * 0.5f + curve_offset;
        // Calculate edge using Bézier curve for proper curvature
        edge_x = quadratic_bezier(bottom_x, control_x, top_x, t);
    }

    // Wiggle effect
    float wiggle_offset = 0.0f;
    if (env->wiggle_active && y >= env->wiggle_y && y <= env->wiggle_y + env->wiggle_length) {
        float t_wiggle = (y - env->wiggle_y) / env->wiggle_length; // Ranges from 0 to 1
        // Trapezoidal wave calculation
        if (t_wiggle < 0.15f) {
            // Connection to road edge
            wiggle_offset = env->wiggle_amplitude * (t_wiggle / 0.15f);
        } else if (t_wiggle < 0.87f) {
            // Flat top of wiggle
            wiggle_offset = env->wiggle_amplitude;
        } else {
            // Reconnection to road edge
            wiggle_offset = env->wiggle_amplitude * ((1.0f - t_wiggle) / 0.13f);
        }
        // Wiggle towards road center
        wiggle_offset *= (left ? 1.0f : -1.0f);
        // Scale wiggle offset based on y position, starting at 0.03f at the vanishing point
        float depth = (y - VANISHING_POINT_Y) / (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y);
        float scale = 0.03f + (depth * depth); 
        if (scale > 0.3f) {
            scale = 0.3f;
        }
        wiggle_offset *= scale;
    }
    // Apply the wiggle offset
    edge_x += wiggle_offset;
    return edge_x;
}

// Computer x position of car in a given lane
float car_x_in_lane(Enduro* env, int lane, float y) {
    // Set offset to 0 to ensure enemy cars align with the road rendering
    float offset = 0.0f;
    float left_edge = road_edge_x(env, y, offset, true);
    float right_edge = road_edge_x(env, y, offset, false);
    float lane_width = (right_edge - left_edge) / NUM_LANES;
    return left_edge + lane_width * (lane + 0.5f);
}

// Client functions
Client* make_client(Enduro* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = env->width;
    client->height = env->height;
    client->player_color = (Color){255, 255, 255, 255}; // WHITE
    client->enemy_color = (Color){255, 0, 0, 255};      // RED
    client->road_color = (Color){0, 100, 0, 255};       // DARKGREEN
    return client;
}

void close_client(Client* client) {
    free(client);
}

// Render car
void render_car(Client* client, Enduro* env) {
    GameState* gameState = &env->gameState;
    Texture2D carTexture;
    // Choose the texture based on the flag
    if (gameState->showLeftTread) {
        carTexture = gameState->playerCarLeftTreadTexture;
    } else {
        carTexture = gameState->playerCarRightTreadTexture;
    }
    // Draw the texture at the player's position
    DrawTexture(carTexture, (int)env->player_x, (int)env->player_y, WHITE);
}

void handleEvents(int* running, Enduro* env) {
    env->actions = ACTION_NOOP;
    if (WindowShouldClose()) {
        *running = 0;
    }
    bool left = IsKeyDown(KEY_LEFT);
    bool right = IsKeyDown(KEY_RIGHT);
    bool down = IsKeyDown(KEY_DOWN);
    bool fire = IsKeyDown(KEY_SPACE); // Fire key
    if (fire) {
        if (right) {
            env->actions = ACTION_RIGHTFIRE;
        } else if (left) {
            env->actions = ACTION_LEFTFIRE;
        } else {
            env->actions = ACTION_FIRE;
        }
    } else if (down) {
        if (right) {
            env->actions = ACTION_DOWNRIGHT;
        } else if (left) {
            env->actions = ACTION_DOWNLEFT;
        } else {
            env->actions = ACTION_DOWN;
        }
    } else if (right) {
        env->actions = ACTION_RIGHT;
    } else if (left) {
        env->actions = ACTION_LEFT;
    } else {
        env->actions = ACTION_NOOP;
    }
}

void initRaylib() {
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Enduro Port Framework");
    SetTargetFPS(60);
}

void loadTextures(GameState* gameState) {
    // Load background and mountain textures for different times of day per og enduro
    char backgroundFile[40];
    char mountainFile[40];
    for (int i = 0; i < 16; ++i) {
        snprintf(backgroundFile, sizeof(backgroundFile), "resources/enduro_clone/%d_bg.png", i);
        gameState->backgroundTextures[i] = LoadTexture(backgroundFile);
        printf("Loaded background image: %s\n", backgroundFile);
        snprintf(mountainFile, sizeof(mountainFile), "resources/enduro_clone/%d_mtns.png", i);
        gameState->mountainTextures[i] = LoadTexture(mountainFile);
        printf("Loaded mountain image: %s\n", mountainFile);
    }
    // Load digit textures 0-9
    char filename[100];
    for (int i = 0; i < 10; i++) {
        snprintf(filename, sizeof(filename), "resources/enduro_clone/digits_%d.png", i);
        gameState->digitTextures[i] = LoadTexture(filename);
        printf("Loaded digit image: %s\n", filename);
    }
    // Load the "CAR" digit texture
    gameState->carDigitTexture = LoadTexture("resources/enduro_clone/digits_car.png");
    printf("Loaded digit image: digits_car.png\n");
    // Load level complete flag textures
    gameState->levelCompleteFlagLeftTexture = LoadTexture("resources/enduro_clone/level_complete_flag_left.png");
    printf("Loaded image: level_complete_flag_left.png\n");
    gameState->levelCompleteFlagRightTexture = LoadTexture("resources/enduro_clone/level_complete_flag_right.png");
    printf("Loaded image: level_complete_flag_right.png\n");
    // Load green digits
    for (int i = 0; i < 10; ++i) {
        snprintf(filename, sizeof(filename), "resources/enduro_clone/green_digits_%d.png", i);
        gameState->greenDigitTextures[i] = LoadTexture(filename);
        printf("Loaded image: %s\n", filename);
    }
    // Load yellow digits
    for (int i = 0; i < 10; ++i) {
        snprintf(filename, sizeof(filename), "resources/enduro_clone/yellow_digits_%d.png", i);
        gameState->yellowDigitTextures[i] = LoadTexture(filename);
        printf("Loaded image: %s\n", filename);
    }
    gameState->playerCarLeftTreadTexture = LoadTexture("resources/enduro_clone/player_car_left_tread.png");
    printf("Loaded image: player_car_left_tread.png\n");
    gameState->playerCarRightTreadTexture = LoadTexture("resources/enduro_clone/player_car_right_tread.png");
    printf("Loaded image: player_car_right_tread.png\n");

    // // Transition times (seconds) from og enduro
    // gameState->backgroundTransitionTimes[0] = 20.0f;
    // gameState->backgroundTransitionTimes[1] = 40.0f;
    // gameState->backgroundTransitionTimes[2] = 60.0f;
    // gameState->backgroundTransitionTimes[3] = 100.0f;
    // gameState->backgroundTransitionTimes[4] = 108.0f;
    // gameState->backgroundTransitionTimes[5] = 114.0f;
    // gameState->backgroundTransitionTimes[6] = 116.0f;
    // gameState->backgroundTransitionTimes[7] = 120.0f;
    // gameState->backgroundTransitionTimes[8] = 124.0f;
    // gameState->backgroundTransitionTimes[9] = 130.0f;
    // gameState->backgroundTransitionTimes[10] = 134.0f;
    // gameState->backgroundTransitionTimes[11] = 138.0f;
    // gameState->backgroundTransitionTimes[12] = 170.0f;
    // gameState->backgroundTransitionTimes[13] = 198.0f;
    // gameState->backgroundTransitionTimes[14] = 214.0f;
    // gameState->backgroundTransitionTimes[15] = 232.0f; // Last transition

// TESTING ONLY - Set cumulative transition times
gameState->backgroundTransitionTimes[0] = 2.0f;   // Transition to background 1 at 2 seconds
gameState->backgroundTransitionTimes[1] = 3.0f;   // Transition to background 2 at 4 seconds
gameState->backgroundTransitionTimes[2] = 4.0f;   // Transition to background 3 at 6 seconds
gameState->backgroundTransitionTimes[3] = 5.0f;   // Transition to background 4 at 8 seconds
gameState->backgroundTransitionTimes[4] = 6.0f;  // Transition to background 5 at 10 seconds
gameState->backgroundTransitionTimes[5] = 7.0f;  // Transition to background 6 at 12 seconds
gameState->backgroundTransitionTimes[6] = 8.0f;  // Transition to background 7 at 14 seconds
gameState->backgroundTransitionTimes[7] = 9.0f;  // Transition to background 8 at 16 seconds
gameState->backgroundTransitionTimes[8] = 10.0f;  // Transition to background 9 at 18 seconds
gameState->backgroundTransitionTimes[9] = 11.0f;  // Transition to background 10 at 20 seconds
gameState->backgroundTransitionTimes[10] = 12.0f; // Transition to background 11 at 22 seconds
gameState->backgroundTransitionTimes[11] = 13.0f; // Transition to background 12 at 24 seconds
// For the next three backgrounds, use longer durations as per your requirement
gameState->backgroundTransitionTimes[12] = 14.0f; // Transition to background 13 at 36 seconds (12-second duration)
gameState->backgroundTransitionTimes[13] = 15.0f; // Transition to background 14 at 48 seconds (12-second duration)
gameState->backgroundTransitionTimes[14] = 16.0f; // Transition to background 15 at 60 seconds (12-second duration)
gameState->backgroundTransitionTimes[15] = 17.0f; // Transition to background 0 at 62 seconds (loop back)


    // Load enemy car textures for each color and tread variant
    gameState->enemyCarTextures[0][0] = LoadTexture("resources/enduro_clone/enemy_car_blue_left_tread.png");
    gameState->enemyCarTextures[0][1] = LoadTexture("resources/enduro_clone/enemy_car_blue_right_tread.png");
    gameState->enemyCarTextures[1][0] = LoadTexture("resources/enduro_clone/enemy_car_gold_left_tread.png");
    gameState->enemyCarTextures[1][1] = LoadTexture("resources/enduro_clone/enemy_car_gold_right_tread.png");
    gameState->enemyCarTextures[2][0] = LoadTexture("resources/enduro_clone/enemy_car_pink_left_tread.png");
    gameState->enemyCarTextures[2][1] = LoadTexture("resources/enduro_clone/enemy_car_pink_right_tread.png");
    gameState->enemyCarTextures[3][0] = LoadTexture("resources/enduro_clone/enemy_car_salmon_left_tread.png");
    gameState->enemyCarTextures[3][1] = LoadTexture("resources/enduro_clone/enemy_car_salmon_right_tread.png");
    gameState->enemyCarTextures[4][0] = LoadTexture("resources/enduro_clone/enemy_car_teal_left_tread.png");
    gameState->enemyCarTextures[4][1] = LoadTexture("resources/enduro_clone/enemy_car_teal_right_tread.png");
    gameState->enemyCarTextures[5][0] = LoadTexture("resources/enduro_clone/enemy_car_yellow_left_tread.png");
    gameState->enemyCarTextures[5][1] = LoadTexture("resources/enduro_clone/enemy_car_yellow_right_tread.png");

    printf("Loaded enemy car images for all colors and tread animations\n");

    // Load enemy car night tail lights textures
    gameState->enemyCarNightTailLightsTexture = LoadTexture("resources/enduro_clone/enemy_car_night_tail_lights.png");
    printf("Loaded image: enemy_car_night_tail_lights.png\n");

    // Load enemy car night fog tail lights texture
    gameState->enemyCarNightFogTailLightsTexture = LoadTexture("resources/enduro_clone/enemy_car_night_fog_tail_lights.png");
    printf("Loaded image: enemy_car_night_fog_tail_lights.png\n");

    // Initialize elapsed time and background index
    // gameState->elapsedTime = 0.0f;
    gameState->backgroundIndex = 0;
    // Initialize animation variables
    gameState->carAnimationTimer = 0.0f;
    gameState->carAnimationInterval = 0.05f; // Initial interval, will be updated based on speed
    gameState->showLeftTread = true;
    // Initialize other game state variables
    gameState->currentBackgroundIndex = 0;
    gameState->score = 0;
    gameState->day = 1;
    gameState->carsToPass = 10; // 200
    gameState->mountainPosition = 0.0f;
}

void cleanup(GameState* gameState) {
    // Unload background and mountain textures
    for (int i = 0; i < 16; ++i) {
        UnloadTexture(gameState->backgroundTextures[i]);
        UnloadTexture(gameState->mountainTextures[i]);
    }
    // Unload digit textures
    for (int i = 0; i < 10; ++i) {
        UnloadTexture(gameState->digitTextures[i]);
        UnloadTexture(gameState->greenDigitTextures[i]);
        UnloadTexture(gameState->yellowDigitTextures[i]);
    }
    // Unload "CAR" digit and flag textures
    UnloadTexture(gameState->carDigitTexture);
    UnloadTexture(gameState->levelCompleteFlagLeftTexture);
    UnloadTexture(gameState->levelCompleteFlagRightTexture);
    // Unload enemy car textures
    for (int color = 0; color < 6; color++) {
        for (int tread = 0; tread < 2; tread++) {
            UnloadTexture(gameState->enemyCarTextures[color][tread]);
        }
    }
    UnloadTexture(gameState->enemyCarNightTailLightsTexture);
    UnloadTexture(gameState->enemyCarNightFogTailLightsTexture);
    // Unload player car textures
    UnloadTexture(gameState->playerCarLeftTreadTexture);
    UnloadTexture(gameState->playerCarRightTreadTexture);
    CloseWindow();
}

void updateCarAnimation(GameState* gameState, Enduro* env) {
    // Update the animation interval based on the player's speed
    // Faster speed means faster alternation
    float minInterval = 0.005f;  // Minimum interval at max speed
    float maxInterval = 0.075f;  // Maximum interval at min speed
    float speedRatio = (env->speed - env->min_speed) / (env->max_speed - env->min_speed);
    gameState->carAnimationInterval = maxInterval - (maxInterval - minInterval) * speedRatio;

    // Update the animation timer
    gameState->carAnimationTimer += GetFrameTime(); // Time since last frame

    if (gameState->carAnimationTimer >= gameState->carAnimationInterval) {
        gameState->carAnimationTimer = 0.0f;
        gameState->showLeftTread = !gameState->showLeftTread; // Switch texture
    }
}

void updateScore(GameState* gameState) {
    // Increase the score every 30 frames (~0.5 seconds at 60 FPS)
    gameState->scoreTimer++;
    if (gameState->scoreTimer >= 30) {
        gameState->scoreTimer = 0;
        gameState->score += 1;
        if (gameState->score > 99999) { // Max score based on SCORE_DIGITS
            gameState->score = 0;
        }
        // Determine which digits have changed and start scrolling them
        int tempScore = gameState->score;
        for (int i = SCORE_DIGITS - 1; i >= 0; i--) {
            int newDigit = tempScore % 10;
            tempScore /= 10;
            if (newDigit != gameState->scoreDigitCurrents[i]) {
                gameState->scoreDigitNexts[i] = newDigit;
                gameState->scoreDigitOffsets[i] = 0.0f;
                gameState->scoreDigitScrolling[i] = true;
            }
        }
    }
    // Update scrolling digits
    for (int i = 0; i < SCORE_DIGITS; i++) {
        if (gameState->scoreDigitScrolling[i]) {
            gameState->scoreDigitOffsets[i] += 0.5f; // Scroll speed
            if (gameState->scoreDigitOffsets[i] >= DIGIT_HEIGHT) {
                gameState->scoreDigitOffsets[i] = 0.0f;
                gameState->scoreDigitCurrents[i] = gameState->scoreDigitNexts[i];
                gameState->scoreDigitScrolling[i] = false; // Stop scrolling
            }
        }
    }
}

void updateBackground(GameState* gameState, Enduro* env) {
    // Use env->elapsedTime instead of gameState->elapsedTime
    float elapsedTime = env->elapsedTime;

    // Total duration of the cycle
    float totalDuration = gameState->backgroundTransitionTimes[15];
    // If elapsed time exceeds total duration, reset it
    if (elapsedTime >= totalDuration) {
        elapsedTime -= totalDuration;
        env->elapsedTime = elapsedTime; // Reset elapsed time in env
        gameState->backgroundIndex = 0;
    }

    // Update previous background index before changing it
    gameState->previousBackgroundIndex = gameState->currentBackgroundIndex;

    // Determine the current background index
    while (gameState->backgroundIndex < 15 &&
           elapsedTime >= gameState->backgroundTransitionTimes[gameState->backgroundIndex]) {
        gameState->backgroundIndex++;
    }
    gameState->currentBackgroundIndex = gameState->backgroundIndex % 16;

    // Logging for verification
    // printf("Elapsed Time: %.2f s, Background Index: %d\n", elapsedTime, gameState->currentBackgroundIndex);
}

void renderBackground(GameState* gameState) {
    Texture2D bgTexture = gameState->backgroundTextures[gameState->currentBackgroundIndex];
    if (bgTexture.id != 0) {
        // Render background
        DrawTexture(bgTexture, 0, 0, WHITE);
    }
}

void renderScoreboard(GameState* gameState) {
    // Positions and sizes
    int digitWidth = DIGIT_WIDTH;
    int digitHeight = DIGIT_HEIGHT;
    // Convert bottom-left coordinates to top-left origin
    int scoreStartX = 56 + digitWidth;
    int scoreStartY = 173 - digitHeight;
    int dayX = 56;
    int dayY = 188 - digitHeight;
    int carsX = 72;
    int carsY = 188 - digitHeight;

    // Render score with scrolling effect
    for (int i = 0; i < SCORE_DIGITS; ++i) {
        int digitX = scoreStartX + i * digitWidth;
        Texture2D currentDigitTexture;
        Texture2D nextDigitTexture;

        if (i == SCORE_DIGITS - 1) {
            // Use yellow digits for the last digit
            currentDigitTexture = gameState->yellowDigitTextures[gameState->scoreDigitCurrents[i]];
            nextDigitTexture = gameState->yellowDigitTextures[gameState->scoreDigitNexts[i]];
        } else {
            // Use regular digits
            currentDigitTexture = gameState->digitTextures[gameState->scoreDigitCurrents[i]];
            nextDigitTexture = gameState->digitTextures[gameState->scoreDigitNexts[i]];
        }

        if (gameState->scoreDigitScrolling[i]) {
            // Scrolling effect for this digit
            float offset = gameState->scoreDigitOffsets[i];
            // Render current digit moving up
            Rectangle srcRectCurrent = { 0, 0, digitWidth, digitHeight - (int)offset };
            Rectangle destRectCurrent = { digitX, scoreStartY + (int)offset, digitWidth, digitHeight - (int)offset };
            DrawTextureRec(currentDigitTexture, srcRectCurrent, (Vector2){ destRectCurrent.x, destRectCurrent.y }, WHITE);
            // Render next digit coming up from below
            Rectangle srcRectNext = { 0, digitHeight - (int)offset, digitWidth, (int)offset };
            Rectangle destRectNext = { digitX, scoreStartY, digitWidth, (int)offset };
            DrawTextureRec(nextDigitTexture, srcRectNext, (Vector2){ destRectNext.x, destRectNext.y }, WHITE);
        } else {
            // No scrolling, render the current digit normally
            DrawTexture(currentDigitTexture, digitX, scoreStartY, WHITE);
        }
    }

    // Render day number
    int day = gameState->day % 10;
    int dayTextureIndex = day;
    if (gameState->victoryAchieved) {
        // Green day digits during victory
        Texture2D greenDigitTexture = gameState->greenDigitTextures[dayTextureIndex];
        DrawTexture(greenDigitTexture, dayX, dayY, WHITE);
    } else {
        // Use normal digits
        Texture2D digitTexture = gameState->digitTextures[dayTextureIndex];
        DrawTexture(digitTexture, dayX, dayY, WHITE);
    }

    // Render "CAR" digit or flags for cars to pass
    if (gameState->victoryAchieved) {
        // Alternate between level_complete_flag_left and level_complete_flag_right
        Texture2D flagTexture = gameState->showLeftFlag ? gameState->levelCompleteFlagLeftTexture : gameState->levelCompleteFlagRightTexture;
        Rectangle destRect = { carsX, carsY, digitWidth * 4, digitHeight };
        DrawTextureEx(flagTexture, (Vector2){ destRect.x, destRect.y }, 0.0f, 1.0f, WHITE);
    } else {
        // Render "CAR" label
        DrawTexture(gameState->carDigitTexture, carsX, carsY, WHITE);
        // Render the remaining digits for cars to pass
        int cars = gameState->carsToPass;
        for (int i = 1; i < CARS_DIGITS; ++i) {
            int divisor = (int)pow(10, CARS_DIGITS - i - 1);
            int digit = (cars / divisor) % 10;
            int digitX = carsX + i * (digitWidth + 1); // Add spacing between digits
            DrawTexture(gameState->digitTextures[digit], digitX, carsY, WHITE);
        }
    }
}

void updateVictoryEffects(GameState* gameState) {
    if (gameState->victoryAchieved) {
        // Update flag timer
        gameState->flagTimer++;
        if (gameState->flagTimer >= 240) { // Switch every 30 frames (~0.5 sec at 60 FPS)
            gameState->flagTimer = 0;
            gameState->showLeftFlag = !gameState->showLeftFlag;
        }
        // Update victory display timer
        gameState->victoryDisplayTimer++;
        if (gameState->victoryDisplayTimer >= 540) { // Display flags for 180 frames (~3 seconds)
            // Reset victory display timer
            gameState->victoryDisplayTimer = 0;
            // Trigger day transition in the step function
            // This is handled in the step function to avoid mixing responsibilities
            printf("Victory display completed.\n");
        }
    }
}

void updateMountains(GameState* gameState, Enduro* env) {
    // Mountain scrolling
    float baseSpeed = 0.0f;
    float curveStrength = fabsf(env->current_curve_factor);
    float speedMultiplier = 1.0f; // Scroll speed
    float scrollSpeed = baseSpeed + curveStrength * speedMultiplier;

    int mountainWidth = gameState->mountainTextures[0].width;

    if (env->current_curve_direction == 1) { // Turning left
        gameState->mountainPosition += scrollSpeed;
        if (gameState->mountainPosition >= mountainWidth) {
            gameState->mountainPosition -= mountainWidth;
        }
    } else if (env->current_curve_direction == -1) { // Turning right
        gameState->mountainPosition -= scrollSpeed;
        if (gameState->mountainPosition <= -mountainWidth) {
            gameState->mountainPosition += mountainWidth;
        }
    }
    // If the road is straight, the mountains don't move
}

void renderMountains(GameState* gameState, Enduro* env) {
    Texture2D mountainTexture = gameState->mountainTextures[gameState->currentBackgroundIndex];
    if (mountainTexture.id != 0) {
        int mountainWidth = mountainTexture.width;
        int mountainY = 45; // y position per original environment

        // Calculate the player's offset from the center
        float playerCenterX = (PLAYER_MIN_X + PLAYER_MAX_X) / 2.0f;
        float playerOffset = env->player_x - playerCenterX;

        // Apply a parallax factor to make the mountains move with the player
        float parallaxFactor = 0.5f; // Adjust as needed
        float adjustedOffset = -playerOffset * parallaxFactor;

        // Base mountain X position including mountainPosition
        float mountainX = -gameState->mountainPosition + adjustedOffset;

        // Use BeginScissorMode to prevent mountains from rendering over the left 8-pixel black space
        BeginScissorMode(PLAYABLE_AREA_LEFT, 0, SCREEN_WIDTH - PLAYABLE_AREA_LEFT, SCREEN_HEIGHT);

        // Draw the mountain textures multiple times to cover the screen width
        for (int x = (int)mountainX; x < SCREEN_WIDTH; x += mountainWidth) {
            DrawTexture(mountainTexture, x, mountainY, WHITE);
        }
        // Draw additional mountain textures to the left if necessary
        for (int x = (int)mountainX - mountainWidth; x > -mountainWidth; x -= mountainWidth) {
            DrawTexture(mountainTexture, x, mountainY, WHITE);
        }

        // End scissor mode
        EndScissorMode();
    }
}


void render(Client* client, Enduro* env) {
    BeginDrawing();
    ClearBackground(BLACK);
    BeginBlendMode(BLEND_ALPHA);

    // Render background
    renderBackground(&env->gameState);

    // Update and render mountains
    updateMountains(&env->gameState, env);
    renderMountains(&env->gameState, env);
    
    int bgIndex = env->gameState.currentBackgroundIndex;
    bool isNightFogStage = (bgIndex == 13);
    bool isNightStage = (bgIndex == 12 || bgIndex == 13 || bgIndex == 14);

    // // Set clipping rectangle to the playable area
    // Rectangle clipRect = { PLAYABLE_AREA_LEFT, VANISHING_POINT_Y, PLAYABLE_AREA_RIGHT - PLAYABLE_AREA_LEFT, PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y };

    // During night fog stage, clip rendering to y >= 92
    float clipStartY = isNightFogStage ? 92.0f : VANISHING_POINT_Y;
    float clipHeight = PLAYABLE_AREA_BOTTOM - clipStartY;
    Rectangle clipRect = { PLAYABLE_AREA_LEFT, clipStartY, PLAYABLE_AREA_RIGHT - PLAYABLE_AREA_LEFT, clipHeight };
    BeginScissorMode(clipRect.x, clipRect.y, clipRect.width, clipRect.height);

    // Render road edges w/ draw vector
    // During night fog stage, start from y=92
    float roadStartY = isNightFogStage ? 92.0f : VANISHING_POINT_Y;
    Vector2 previousLeftPoint = {0}, previousRightPoint = {0};
    bool firstPoint = true;

    for (float y = roadStartY; y <= PLAYABLE_AREA_BOTTOM; y += 0.75f) {
        float adjusted_y = (env->speed < 0) ? y : y + fmod(env->road_scroll_offset, 0.75f);
        if (adjusted_y > PLAYABLE_AREA_BOTTOM) continue;

        float left_edge = road_edge_x(env, adjusted_y, 0, true);
        float right_edge = road_edge_x(env, adjusted_y, 0, false);

        Color roadColor;
        if (adjusted_y >= 52 && adjusted_y < 91) {
            roadColor = (Color){74, 74, 74, 255};
        } else if (adjusted_y >= 91 && adjusted_y < 106) {
            roadColor = (Color){111, 111, 111, 255};
        } else if (adjusted_y >= 106 && adjusted_y <= 154) {
            roadColor = (Color){170, 170, 170, 255};
        } else {
            roadColor = WHITE; // Default color if needed
        }

        Vector2 currentLeftPoint = {left_edge, adjusted_y};
        Vector2 currentRightPoint = {right_edge, adjusted_y};

        if (!firstPoint) {
            DrawLineV(previousLeftPoint, currentLeftPoint, roadColor);
            DrawLineV(previousRightPoint, currentRightPoint, roadColor);
        }

        previousLeftPoint = currentLeftPoint;
        previousRightPoint = currentRightPoint;
        firstPoint = false;
    }

    // Render enemy cars scaled stages for distance/closeness effect
    for (int i = 0; i < env->numEnemies; i++) {
        Car* car = &env->enemyCars[i];
        
        if (isNightFogStage && car->y < 92.0f) {
            continue; // Don't render cars in fog
        }

        // Determine the car scale based on the seven-stage progression
        float car_scale;
        if (car->y <= 68.0f) car_scale = 2.0f / 20.0f;        // Stage 1
        else if (car->y <= 74.0f) car_scale = 4.0f / 20.0f;   // Stage 2
        else if (car->y <= 86.0f) car_scale = 6.0f / 20.0f;   // Stage 3
        else if (car->y <= 100.0f) car_scale = 8.0f / 20.0f;  // Stage 4
        else if (car->y <= 110.0f) car_scale = 12.0f / 20.0f;  // Stage 5
        else if (car->y <= 120.0f) car_scale = 14.0f / 20.0f; // Stage 6
        else if (car->y <= 135.0f) car_scale = 16.0f / 20.0f; // Stage 7
        else car_scale = 1.0f;                                // Normal size

        // Select the correct texture based on the car's color and current tread
        // Night stages use tail light textures; 13 is night fog
        Texture2D carTexture;
        if (isNightStage) {
            if (bgIndex == 13) {
                carTexture = env->gameState.enemyCarNightFogTailLightsTexture;
            } else {
                carTexture = env->gameState.enemyCarNightTailLightsTexture;
            }
        } else {
            // Use existing enemy car textures
            int colorIndex = car->colorIndex;
            int treadIndex = env->gameState.showLeftTread ? 0 : 1;
            carTexture = env->gameState.enemyCarTextures[colorIndex][treadIndex];
        }

        // Compute car x position in its lane
        float car_center_x = car_x_in_lane(env, car->lane, car->y);
        float car_x = car_center_x - (carTexture.width * car_scale) / 2.0f;
        float car_y = car->y - (carTexture.height * car_scale) / 2.0f;

        // Draw the scaled car texture
        DrawTextureEx(carTexture, (Vector2){car_x, car_y}, 0.0f, car_scale, WHITE);
    }

    // Render player car (no scaling since it's at the bottom)
    render_car(client, env);

    // Remove clipping
    EndScissorMode();
    EndBlendMode();

    // Render scoreboard
    renderScoreboard(&env->gameState);

    EndDrawing();
}

#endif