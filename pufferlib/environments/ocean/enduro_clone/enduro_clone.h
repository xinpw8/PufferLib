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
#include <stddef.h> // For NULL
#include <string.h>
#include "raylib.h"

// Constant definitions
#define TARGET_FPS 60 // Used to calculate wiggle spawn frequency
#define LOG_BUFFER_SIZE 1024
#define SCREEN_WIDTH 160
#define SCREEN_HEIGHT 210 // Corrected to 210 as per your specifications
#define PLAYABLE_AREA_TOP 0
#define PLAYABLE_AREA_BOTTOM 154
#define PLAYABLE_AREA_LEFT 8
#define PLAYABLE_AREA_RIGHT 160
#define ACTION_HEIGHT (PLAYABLE_AREA_BOTTOM - PLAYABLE_AREA_TOP)
#define CAR_WIDTH 16
#define CAR_HEIGHT 11
#define MAX_ENEMIES 10
#define CRASH_NOOP_DURATION 60
#define DAY_LENGTH 200000
#define INITIAL_CARS_TO_PASS 5
#define TOP_SPAWN_OFFSET 12.0f // Cars spawn/disappear 12 pixels from top
#define ROAD_LEFT_EDGE_X 60
#define ROAD_RIGHT_EDGE_X 161
#define VANISHING_POINT_X 110
#define VANISHING_POINT_Y 52
#define LOGICAL_VANISHING_Y (VANISHING_POINT_Y + 12)  // Separate logical vanishing point for cars disappearing
#define INITIAL_PLAYER_X ((ROAD_LEFT_EDGE_X + ROAD_RIGHT_EDGE_X)/2 - CAR_WIDTH/2)
#define PLAYER_MAX_Y (ACTION_HEIGHT - CAR_HEIGHT) // Max y is car length from bottom
#define PLAYER_MIN_Y (ACTION_HEIGHT - 2 * CAR_HEIGHT) // Min y is 2 car lengths from bottom
#define ACCELERATION_RATE 0.05f
#define DECELERATION_RATE 0.1f
#define FRICTION 0.95f
#define MIN_SPEED -2.5f
#define MAX_SPEED 3.0f
#define CAR_PIXELS_COUNT 120
#define CAR_PIXEL_HEIGHT 11
#define CAR_PIXEL_WIDTH 16
#define CURVE_FREQUENCY 0.05f
#define CURVE_AMPLITUDE 30.0f
#define ROAD_BASE_WIDTH (ROAD_RIGHT_EDGE_X - ROAD_LEFT_EDGE_X)
#define NUM_LANES 3
// Rendering constants
// Number of digits in the scoreboard
#define SCORE_DIGITS 5
#define DAY_DIGITS   1
#define CARS_DIGITS  4
#define SCOREBOARD_START_Y 165 // Adjust this to fit within the window dimensions
#define SCOREBOARD_DIGIT_HEIGHT 9
// Digit dimensions
#define DIGIT_WIDTH 8
#define DIGIT_HEIGHT 9
// Redef section for testing
#define PLAYER_MIN_X 65.5f
#define PLAYER_MAX_X 91.5f
#define ROAD_LEFT_OFFSET 50.0f
#define ROAD_RIGHT_OFFSET 51.0f
#define VANISHING_POINT_X_LEFT 110.0f
#define VANISHING_POINT_X_RIGHT 62.0f
#define PLAYABLE_AREA_BOTTOM 154
#define VANISHING_POINT_Y 52
#define CURVE_VANISHING_POINT_SHIFT 55.0f  // Adjust as needed
// Constants for wiggle effect timing and amplitude
#define MIN_WIGGLE_FREQUENCY (1.0f / 5.0f) // 5 seconds per cycle at min speed
#define MAX_WIGGLE_FREQUENCY (1.0f / 0.3f) // 0.3 seconds per cycle at max speed
#define WIGGLE_AMPLITUDE 8.0f              // Maximum 'bump-in' offset in pixels

// Test cases and validation framework
typedef struct {
    float player_x;
    int curve_direction;  // -1: left, 0: straight, 1: right
    char* description;
} TestCase;

TestCase test_cases[] = {
    {PLAYER_MIN_X, 0, "Player far left, straight road"},
    {PLAYER_MAX_X, 0, "Player far right, straight road"},
    {PLAYER_MIN_X, 1, "Player far left, right curve"},
    {PLAYER_MAX_X, 1, "Player far right, right curve"},
    {PLAYER_MIN_X, -1, "Player far left, left curve"},
    {PLAYER_MAX_X, -1, "Player far right, left curve"},
};

// Log structures
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

// Car structure for enemy cars
typedef struct Car {
    int lane;   // Lane index: 0, 1, or 2
    float y;    // Current y position
    int passed; // Flag to indicate if car has been passed by player
    int colorIndex; // Index for car's color (0-5)
} Car;

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
    float elapsedTime;       // Total elapsed time in seconds
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

// Game environment structure
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
    float day_length;
    int initial_cars_to_pass;
    float min_speed;
    float max_speed;
    float player_x;
    float player_y;
    float speed;
    // ints
    int score;
    int day;
    int step_count;
    int numEnemies;
    int carsToPass;
    float collision_cooldown;
    float action_height;
    Car enemyCars[MAX_ENEMIES];
    float initial_player_x;
    int last_lr_action; // 0: none, 1: left, 2: right
    float road_scroll_offset;
    int car_pixels;
    int current_curve_direction; // 1: Right, -1: Left, 0: Straight
    // Variables for adjusting left edge control point when curving right
    float left_curve_p1_x;
    float left_curve_p1_x_increment;
    float left_curve_p1_x_min;
    float left_curve_p1_x_max;
    float vanishing_point_x;
    float base_vanishing_point_x;
    float t_p;
    // Roadside wiggle effect
    float wiggle_y;            // Current y position of the wiggle
    float wiggle_speed;        // Speed at which the wiggle moves down the screen
    float wiggle_length;       // Vertical length of the wiggle effect
    float wiggle_amplitude;    // How far into road wiggle extends
    bool wiggle_active;        // Whether the wiggle is active
    // Logging
    float P0_x_left_edge;
    float P2_x_left_edge;
    float P0_x_right_edge;
    float P2_x_right_edge;
    float last_road_left;
    float last_road_right;
    // Current background colors
    Color currentSkyColors[5];
    int currentSkyColorCount;
    Color currentMountainColor;
    Color currentGrassColor;
    // Mountain rendering
    float parallaxFactor;
    // Victory flag display timer
    int victoryFlagTimer;
    // Game state
    GameState gameState;
} Enduro;

// Client structure
typedef struct Client {
    float width;
    float height;
    Color player_color;
    Color enemy_color;
    Color road_color;
} Client;

// Enumeration for road direction
typedef enum {
    ROAD_STRAIGHT,
    ROAD_TURN_LEFT,
    ROAD_TURN_RIGHT
} RoadDirection;

// Function prototypes
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
void updateBackground(GameState* gameState);
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
void update_road_curve(Enduro* env);
void update_vanishing_point(Enduro* env, float offset);
void test_road_edges(Enduro* env);

// Function definitions
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

// Environment functions
void init(Enduro* env) {
    env->step_count = 0;
    env->score = 0;
    env->numEnemies = 0;
    env->collision_cooldown = 0.0;
    env->action_height = ACTION_HEIGHT;
    // Initialize vanishing point and player position
    env->initial_player_x = INITIAL_PLAYER_X;
    env->player_x = env->initial_player_x;
    env->vanishing_point_x = VANISHING_POINT_X;
    // Initialize player y position
    env->player_y = PLAYER_MAX_Y;
    // Speed-related fields
    env->min_speed = MIN_SPEED;
    env->max_speed = MAX_SPEED;
    env->speed = env->min_speed;
    env->carsToPass = env->initial_cars_to_pass;
    env->day = 1;
    for (int i = 0; i < MAX_ENEMIES; i++) {
        env->enemyCars[i].lane = 0;
        env->enemyCars[i].y = 0.0;
        env->enemyCars[i].passed = 0;
    }
    env->last_lr_action = 0;
    env->road_scroll_offset = 0.0f;
    if (env->log_buffer != NULL) {
        env->log_buffer->idx = 0;
    }
    env->log.episode_return = 0.0;
    env->log.episode_length = 0.0;
    env->log.score = 0.0;
    env->car_pixels = CAR_PIXELS_COUNT;
    env->current_curve_direction = 0;
    env->vanishing_point_x = VANISHING_POINT_X;
    // Initialize variables for left edge control point adjustment
    env->left_curve_p1_x = -20.0f;             // Starting offset
    env->left_curve_p1_x_increment = 0.5f;   // Increment per step
    env->left_curve_p1_x_min = -20.0f;         // Minimum offset
    env->left_curve_p1_x_max = 160.0f;       // Maximum offset
    env->victoryFlagTimer = 0;
    test_road_edges(env);
    // Wiggle effect initialization
    env->wiggle_y = VANISHING_POINT_Y;  // Start at the vanishing point
    env->wiggle_speed = 10.1f;           // Adjust as needed (doesn't matter??)
    env->wiggle_length = 26.0f; // PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y; // playable area   50.0f;  // Vertical size of the wiggle
    env->wiggle_amplitude = 6.0f;       // Maximum 'bump-in' offset in pixels
    env->wiggle_active = true;
    // Synchronize carsToPass with GameState
    env->gameState.carsToPass = env->carsToPass;
    env->gameState.victoryAchieved = false; // Initialize victory condition
}

void allocate(Enduro* env) {
    init(env);
    env->observations = (float*)calloc(8 + 2 * MAX_ENEMIES, sizeof(float));
    if (env->observations == NULL) {
        printf("[ERROR] Memory allocation for env->observations failed!\n");
        return;
    }
    env->rewards = (float*)calloc(1, sizeof(float));
    if (env->rewards == NULL) {
        printf("[ERROR] Memory allocation for env->rewards failed!\n");
        free(env->observations);
        return;
    }
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
    env->last_lr_action = 0;
    env->road_scroll_offset = 0.0f;
}

void reset(Enduro* env) {
    // Reset player position
    env->player_x = env->initial_player_x;
    env->player_y = PLAYER_MAX_Y;
    // Reset speed
    env->speed = env->min_speed;
    // Reset road scroll offset
    env->road_scroll_offset = 0.0f;
    // Reset enemy cars
    env->numEnemies = 0;
    // Reset collision cooldown
    env->collision_cooldown = 0;
    // Reset passed flags for enemy cars
    for (int i = 0; i < MAX_ENEMIES; i++) {
        env->enemyCars[i].passed = 0;
    }
    // Reset carsToPass in Enduro and GameState to current day
    env->carsToPass = env->day * 10 + 10; // Based on current day
    env->gameState.carsToPass = env->carsToPass;
    // Reset victory condition
    env->gameState.victoryAchieved = false;
    // Reset rewards and logs
    env->rewards[0] = 0;
    add_log(env->log_buffer, &env->log);
    printf("Game reset for day %d with %d cars to pass.\n", env->day, env->carsToPass);
}


bool check_collision(Enduro* env, Car* car) {
    // Compute the scale factor based on vanishing point reference
    float depth = (car->y - VANISHING_POINT_Y) / (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y);
    float scale = fmax(0.1f, 0.9f * depth);
    // Compute car dimensions
    float car_width = CAR_WIDTH * scale;
    float car_height = CAR_HEIGHT * scale;
    // Compute car x position
    float car_center_x = car_x_in_lane(env, car->lane, car->y);
    float car_x = car_center_x - car_width / 2.0f;
    // Check collision
    return !(env->player_x > car_x + car_width ||
             env->player_x + CAR_WIDTH < car_x ||
             env->player_y > car->y + car_height ||
             env->player_y + CAR_HEIGHT < car->y);
}

int get_player_lane(Enduro* env) {
    float player_center_x = env->player_x + CAR_WIDTH / 2.0f;
    float offset = (env->player_x - env->initial_player_x) * 0.5f;
    float left_edge = road_edge_x(env, env->player_y, offset, true);
    float right_edge = road_edge_x(env, env->player_y, offset, false);
    float lane_width = (right_edge - left_edge) / 3.0f;
    int lane = (int)((player_center_x - left_edge) / lane_width);
    if (lane < 0) lane = 0;
    if (lane > 2) lane = 2;
    return lane;
}

void add_enemy_car(Enduro* env) {
    if (env->numEnemies >= MAX_ENEMIES) return;
    int lane = rand() % NUM_LANES; // Use NUM_LANES for consistency
    if (env->speed < 0) {
        int player_lane = get_player_lane(env);
        // Avoid spawning in the player's lane
        while (lane == player_lane) {
            lane = rand() % NUM_LANES;
        }
    }
    Car car = { .lane = lane, .passed = false, .colorIndex = rand() % 6 };
    if (env->speed > 0) {
        // Spawn cars slightly below the vanishing point
        car.y = VANISHING_POINT_Y + 10.0f;
    } else {
        // Spawn cars behind the player when moving backward
        car.y = PLAYABLE_AREA_BOTTOM + CAR_HEIGHT; // Spawn off-screen at the bottom
    }
    // printf("Adding enemy car at y = %.2f, lane = %d, x = %.2f\n", car.y, car.lane, car_x);
    env->enemyCars[env->numEnemies++] = car;
}

// Adjust base vanishing point with offset during curves
void update_vanishing_point(Enduro* env, float offset) {
    env->vanishing_point_x = env->base_vanishing_point_x + offset;
}

void step(Enduro* env) {
    if (env == NULL) {
        printf("[ERROR] env is NULL! Aborting step.\n");
        return;
    }

    update_road_curve(env);
    env->log.episode_length += 1;
    env->terminals[0] = 0;

    // Update road scroll offset
    env->road_scroll_offset += env->speed;

    // Update enemy cars positions
    for (int i = 0; i < env->numEnemies; i++) {
        Car* car = &env->enemyCars[i];
        car->y += env->speed;
    }

    // Calculate road edges
    float road_left = road_edge_x(env, env->player_y, 0, true);
    float road_right = road_edge_x(env, env->player_y, 0, false) - CAR_WIDTH;
    env->last_road_left = road_left;
    env->last_road_right = road_right;

    // Player movement logic
    if (env->collision_cooldown <= 0) {
        int act = env->actions;
        if (act == 1) {  // Move left
            env->player_x -= 0.5f;
            if (env->player_x < road_left) env->player_x = road_left;
            env->last_lr_action = 1;
        } else if (act == 2) {  // Move right
            env->player_x += 0.5f;
            if (env->player_x > road_right) env->player_x = road_right;
            env->last_lr_action = 2;
        }
        if (act == 3 && env->speed < env->max_speed) env->speed += ACCELERATION_RATE; // Accelerate
        if (act == 4 && env->speed > env->min_speed) env->speed -= DECELERATION_RATE; // Decelerate
    } else {
        env->collision_cooldown -= 1;
        if (env->last_lr_action == 1) env->player_x -= 25;
        if (env->last_lr_action == 2) env->player_x += 25;
        env->speed *= FRICTION;
        env->speed -= 0.305f * DECELERATION_RATE;
        if (env->speed < env->min_speed) env->speed = env->min_speed;
    }
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

    // Update wiggle position
    if (env->wiggle_active) {
        // Calculate wiggle period based on speed
        float min_wiggle_period = 5.8f;    // 5.8 seconds at minimum speed
        float max_wiggle_period = 0.3f;    // 0.3 seconds at maximum speed
        // Normalize speed between 0 (min_speed) and 1 (max_speed)
        float speed_normalized = (env->speed - env->min_speed) / (env->max_speed - env->min_speed);
        speed_normalized = fmaxf(0.0f, fminf(1.0f, speed_normalized));
        // Calculate current wiggle period
        float current_wiggle_period = min_wiggle_period - (min_wiggle_period - max_wiggle_period) * speed_normalized;
        // Calculate wiggle speed (pixels per frame)
        env->wiggle_speed = (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y) / (current_wiggle_period * TARGET_FPS);
        // Update wiggle position
        env->wiggle_y += env->wiggle_speed;
        // Reset wiggle when it reaches the bottom
        if (env->wiggle_y > PLAYABLE_AREA_BOTTOM) {
            env->wiggle_y = VANISHING_POINT_Y;
        }
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
            if (env->carsToPass < 0) env->carsToPass = 0; // Ensure it doesn’t go negative
            car->passed = true;
            env->score += 10;
            env->rewards[0] += 1;

            // Debugging output to confirm passing is detected
            printf("Car passed at y = %.2f. Remaining cars to pass: %d\n", car->y, env->carsToPass);
        }

        // Check for collisions between the player and the car
        if (check_collision(env, car)) {
            env->speed = env->min_speed = MIN_SPEED;
            env->collision_cooldown = CRASH_NOOP_DURATION;
            // Adjust player position if collision occurred
            if (env->last_lr_action == 1) {
                env->player_x -= 10;
                if (env->player_x < road_left) env->player_x = road_left;
            } else if (env->last_lr_action == 2) {
                env->player_x += 10;
                if (env->player_x > road_right) env->player_x = road_right;
            }
            env->last_lr_action = 0;
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

        // Remove cars that reach or surpass the logical vanishing point if moving up (speed negative)
        if (env->speed < 0 && car->y <= LOGICAL_VANISHING_Y) {
            // Remove car from array if it reaches the logical vanishing point while moving backward
            for (int j = i; j < env->numEnemies - 1; j++) {
                env->enemyCars[j] = env->enemyCars[j + 1];
            }
            env->numEnemies--;
            i--;
            // printf("Removing car %d as it reached the logical vanishing point at LOGICAL_VANISHING_Y = %.2f\n", i, (float)LOGICAL_VANISHING_Y);
        }
    }

    // Adjust enemy car spawn frequency based on day number
    if (env->numEnemies < MAX_ENEMIES && rand() % 200 < 1) { // Adjusted spawn rate
        add_enemy_car(env);
    }

    // Handle victory condition
    if (env->carsToPass <= 0 && !env->gameState.victoryAchieved) {
        // Set victory condition
        env->gameState.victoryAchieved = true;
        printf("Victory achieved! Preparing for next day.\n");
    }

    // Update Victory Effects
    updateVictoryEffects(&env->gameState);

    // Handle day transition after victory display timer completes
    if (env->gameState.victoryAchieved && env->gameState.victoryDisplayTimer == 0 && env->step_count % 60 == 0) { // Example condition to reset after some frames
        // Increment day
        env->day++;
        // Set new carsToPass based on the new day
        env->carsToPass = env->day * 10 + 10; // Example: Day 1 = 20 cars, Day 2 = 30 cars, etc.
        // Optionally increase speed for the new day
        env->speed += 0.1f;
        // Reset victory state
        env->gameState.victoryAchieved = false;
        // Reset rewards and logs
        env->rewards[0] = 0;
        add_log(env->log_buffer, &env->log);
        printf("New day %d started with %d cars to pass.\n", env->day, env->carsToPass);
    }

    // Handle loss condition
    if (env->step_count >= env->day_length && env->carsToPass > 0) {
        // Player failed to pass required cars within day_length
        env->terminals[0] = 1; // Signal termination
        reset(env);              // Reset the game state without incrementing day
        printf("Day %d failed. Resetting game without incrementing day.\n", env->day);
        return;
    }

    // Synchronize carsToPass between Enduro and GameState
    env->gameState.carsToPass = env->carsToPass;

    env->log.episode_return += env->rewards[0];
    env->step_count++;
    env->log.score = env->score;
}

// void step(Enduro* env) {
//     if (env == NULL) {
//         printf("[ERROR] env is NULL! Aborting step.\n");
//         return;
//     }

//     update_road_curve(env);
//     env->log.episode_length += 1;
//     env->terminals[0] = 0;
//     // Update road scroll offset
//     env->road_scroll_offset += env->speed;
//     // Update enemy cars even during collision cooldown
//     for (int i = 0; i < env->numEnemies; i++) {
//         Car* car = &env->enemyCars[i];
//         // Update enemy car position
//         car->y += env->speed;
//         // printf("Updating car %d position to y = %.2f\n", i, car->y);
//     }
//     // // Calculate the offset based on player left right movement
//     float road_left = road_edge_x(env, env->player_y, 0, true);
//     float road_right = road_edge_x(env, env->player_y, 0, false) - CAR_WIDTH;
//     env->last_road_left = road_left;
//     env->last_road_right = road_right;
//     // Player movement logic
//     if (env->collision_cooldown <= 0) {
//         int act = env->actions;
//         if (act == 1) {  // Move left
//             env->player_x -= 0.5;
//             if (env->player_x < road_left) env->player_x = road_left;
//             env->last_lr_action = 1;
//         } else if (act == 2) {  // Move right
//             env->player_x += 0.5;
//             if (env->player_x > road_right) env->player_x = road_right;
//             env->last_lr_action = 2;
//         }
//         if (act == 3 && env->speed < env->max_speed) env->speed += ACCELERATION_RATE; // Accelerate
//         if (act == 4 && env->speed > env->min_speed) env->speed -= DECELERATION_RATE; // Decelerate
//     } else {
//         env->collision_cooldown -= 1;
//         if (env->last_lr_action == 1) env->player_x -= 25;
//         if (env->last_lr_action == 2) env->player_x += 25;
//         env->speed *= FRICTION;
//         env->speed -= 0.305 * DECELERATION_RATE;
//         if (env->speed < env->min_speed) env->speed = env->min_speed;
//     }
//     if (env->player_x < road_left) env->player_x = road_left;
//     if (env->player_x > road_right) env->player_x = road_right;
//     // Update player's horizontal position ratio, t_p
//     float t_p = (env->player_x - PLAYER_MIN_X) / (PLAYER_MAX_X - PLAYER_MIN_X);
//     t_p = fmaxf(0.0f, fminf(1.0f, t_p));
//     env->t_p = t_p;
//     // Base vanishing point based on player's horizontal movement (without curve)
//     env->base_vanishing_point_x = VANISHING_POINT_X_LEFT - env->t_p * (VANISHING_POINT_X_LEFT - VANISHING_POINT_X_RIGHT);
//     // Adjust vanishing point based on current curve
//     float curve_vanishing_point_shift = env->current_curve_direction * CURVE_VANISHING_POINT_SHIFT;
//     env->vanishing_point_x = env->base_vanishing_point_x + curve_vanishing_point_shift;

//     // Update wiggle position
//     if (env->wiggle_active) {
//         // Calculate wiggle period based on speed
//         float min_wiggle_period = 5.8f;    // 6.25 // At minimum speed, wiggle occurs every 5 seconds
//         float max_wiggle_period = 0.3f;    // At maximum speed, wiggle occurs every 0.3 seconds
//         // Normalize speed between 0 (min_speed) and 1 (max_speed)
//         float speed_normalized = (env->speed - env->min_speed) / (env->max_speed - env->min_speed);
//         speed_normalized = fmaxf(0.0f, fminf(1.0f, speed_normalized));
//         // Calculate current wiggle period
//         float current_wiggle_period = min_wiggle_period - (min_wiggle_period - max_wiggle_period) * speed_normalized;
//         // Calculate wiggle speed (pixels per frame)
//         env->wiggle_speed = (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y) / (current_wiggle_period * TARGET_FPS);
//         // Update wiggle position
//         env->wiggle_y += env->wiggle_speed;
//         // Reset wiggle when it reaches the bottom
//         if (env->wiggle_y > PLAYABLE_AREA_BOTTOM) {
//             env->wiggle_y = VANISHING_POINT_Y;
//         }
//     }

//     // Update player y position based on speed
//     env->player_y = PLAYER_MAX_Y - (env->speed - env->min_speed) / (env->max_speed - env->min_speed) * (PLAYER_MAX_Y - PLAYER_MIN_Y);
//     // Clamp player_y
//     if (env->player_y < PLAYER_MIN_Y) env->player_y = PLAYER_MIN_Y;
//     if (env->player_y > PLAYER_MAX_Y) env->player_y = PLAYER_MAX_Y;
//     // printf("Player position y = %.2f, speed = %.2f\n", env->player_y, env->speed);
//     // Enemy car logic
//     for (int i = 0; i < env->numEnemies; i++) {
//         Car* car = &env->enemyCars[i];

//         // Check for passing logic
//         if (env->speed > 0 && car->y > env->player_y + CAR_HEIGHT && !car->passed) {
//             // Mark car as passed and increment the score
//             env->carsToPass--;
//             if (env->carsToPass < 0) env->carsToPass = 0; // Ensure it doesn’t go negative
//             car->passed = true;
//             env->score += 10;
//             env->rewards[0] += 1;
            
//             // Debugging output to confirm passing is detected
//             printf("Car passed at y = %.2f. Remaining cars to pass: %d\n", car->y, env->carsToPass);
//         }

//         // Check for collisions between the player and the car (remaining unchanged)
//         if (check_collision(env, car)) {
//             env->speed = env->min_speed = MIN_SPEED;
//             env->collision_cooldown = CRASH_NOOP_DURATION;
//             // Adjust player position if collision occurred
//             if (env->last_lr_action == 1) {
//                 env->player_x -= 10;
//                 if (env->player_x < road_left) env->player_x = road_left;
//             } else if (env->last_lr_action == 2) {
//                 env->player_x += 10;
//                 if (env->player_x > road_right) env->player_x = road_right;
//             }
//             env->last_lr_action = 0;
//         }
//         // Remove off-screen cars that move below the screen
//         if (car->y > PLAYABLE_AREA_BOTTOM + CAR_HEIGHT * 5) {
//             // Remove car from array if it moves below the screen
//             for (int j = i; j < env->numEnemies - 1; j++) {
//                 env->enemyCars[j] = env->enemyCars[j + 1];
//             }
//             env->numEnemies--;
//             i--;
//             // printf("Removing car %d as it went below screen\n", i);
//         }
//         // Remove cars that reach or surpass the logical vanishing point if moving up (speed negative)
//         if (env->speed < 0 && car->y <= LOGICAL_VANISHING_Y) {
//             // Remove car from array if it reaches the logical vanishing point while moving backward
//             for (int j = i; j < env->numEnemies - 1; j++) {
//                 env->enemyCars[j] = env->enemyCars[j + 1];
//             }
//             env->numEnemies--;
//             i--;
//             // printf("Removing car %d as it reached the logical vanishing point at LOGICAL_VANISHING_Y = %.2f\n", i, (float)LOGICAL_VANISHING_Y);
//         }
//     }

//     // Adjust enemy car spawn frequency based on day number
//     if (env->numEnemies < MAX_ENEMIES && rand() % 200 < 1) {
//         add_enemy_car(env);
//     }

//     // Handle day completion logic
//     if (env->carsToPass <= 0 && !env->gameState.victoryAchieved) {
//         // Set victory condition
//         env->gameState.victoryAchieved = true;
//         printf("Victory achieved! Preparing for next day.\n");
//     }

//     // Update Victory Effects
//     updateVictoryEffects(&env->gameState);

//     // After victory display timer completes, reset for next day
//     if (!env->gameState.victoryAchieved && env->gameState.victoryDisplayTimer == 0 && env->carsToPass == 0) {
//         // Increment day
//         env->day++;
//         // Set new carsToPass based on the new day
//         env->carsToPass = env->day * 10 + 10; // Example: Day 1 = 20 cars, Day 2 = 30 cars, etc.
//         // Optionally increase speed for the new day
//         env->speed += 0.1f;
//         printf("New day %d started with %d cars to pass.\n", env->day, env->carsToPass);
//         // Reset log and rewards
//         add_log(env->log_buffer, &env->log);
//         env->rewards[0] = 0;
//     }

//     // Synchronize carsToPass between Enduro and GameState
//     env->gameState.carsToPass = env->carsToPass;

//     env->log.episode_return += env->rewards[0];
//     env->step_count++;
//     env->log.score = env->score;
// }

void update_road_curve(Enduro* env) {
    static int current_curve_stage = 0;
    static int steps_in_current_stage = 0;
    // Define the number of steps for each curve stage
    int step_thresholds[] = {
        1, 150, 150, 150, 150,150, 150, 150, 150, 150,150, 150, 150, 150, 150, // <--test values. actual values: 250, 1000, 125, 250, 500, 800, 600, 200, 1100, 1200, 1000, 400, 200
    };
    int curve_directions[] = {
        // 1,1,1,1,1,1,1,1,1,1,1,1,1
        // -1,1,-1,1,-1,1,-1,1,-1,1,-1,1
       0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    //    1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 1, -1, 1, -1, 1, -1, 1  // -1: Left, 1: Right, 0: Straight
    };
    steps_in_current_stage++;
    if (steps_in_current_stage >= step_thresholds[current_curve_stage]) {
        env->current_curve_direction = curve_directions[current_curve_stage];
        // Reset step counter and move to the next stage
        steps_in_current_stage = 0;
        current_curve_stage = (current_curve_stage + 1) % (sizeof(step_thresholds) / sizeof(int));
    }
}

// B(t) = (1−t)^2 * P0​+2(1−t) * t * P1​+t^2 * P2​, t∈[0,1]
// Quadratic bezier curve helper function
float quadratic_bezier(float bottom_x, float control_x, float top_x, float t) {
    float one_minus_t = 1.0f - t;
    return one_minus_t * one_minus_t * bottom_x + 
           2.0f * one_minus_t * t * control_x + 
           t * t * top_x;
}

float road_edge_x(Enduro* env, float y, float offset, bool left) {
    float t = (PLAYABLE_AREA_BOTTOM - y) / (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y);
    // Determine base offset for left or right edge
    float base_offset = left ? -ROAD_LEFT_OFFSET : ROAD_RIGHT_OFFSET;
    float bottom_x = env->base_vanishing_point_x + base_offset + offset;
    float top_x = env->vanishing_point_x + offset;
    // Calculate edge_x (either linear interpolation or Bezier curve)
    float edge_x;
    if (env->current_curve_direction == 0) {
        edge_x = bottom_x + t * (top_x - bottom_x);
    } else {
        // Calculate curve amount and control point for curves
        float curve_amount = (env->current_curve_direction == -1 ? 30.0f : -30.0f);
        float control_x = bottom_x + (top_x - bottom_x) * 0.5f + curve_amount;
        // Quadratic Bezier curve calculation
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

// Test cases for road edge rendering
// Helper function to print road edges for a given y position
void print_road_edges(Enduro* env, float y, const char* y_position_desc) {
    float left = road_edge_x(env, y, 0, true);
    float right = road_edge_x(env, y, 0, false);
    printf("%s: Left=%.2f, Right=%.2f, Width=%.2f\n", 
           y_position_desc, left, right, right - left);
}
void test_road_edges(Enduro* env) {
    printf("\n=== Road Edge Rendering Tests ===\n");    
    float test_y_positions[] = {
        PLAYABLE_AREA_BOTTOM,
        PLAYABLE_AREA_BOTTOM - (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y) * 0.25f,
        PLAYABLE_AREA_BOTTOM - (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y) * 0.5f,
        PLAYABLE_AREA_BOTTOM - (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y) * 0.75f,
        VANISHING_POINT_Y
    };    
    for (int i = 0; i < sizeof(test_cases)/sizeof(TestCase); i++) {
        printf("\nTest Case: %s\n", test_cases[i].description);
        printf("Player X: %.2f, Curve Direction: %d\n", 
               test_cases[i].player_x, test_cases[i].curve_direction);        
        // Set up environment for test
        env->player_x = test_cases[i].player_x;
        env->current_curve_direction = test_cases[i].curve_direction;        
        // Calculate t_p and update vanishing point
        float t_p = (env->player_x - PLAYER_MIN_X) / (PLAYER_MAX_X - PLAYER_MIN_X);
        t_p = fmaxf(0.0f, fminf(1.0f, t_p));
        env->t_p = t_p;        
        // Update vanishing point
        env->base_vanishing_point_x = VANISHING_POINT_X_LEFT - 
            t_p * (VANISHING_POINT_X_LEFT - VANISHING_POINT_X_RIGHT);
        env->vanishing_point_x = env->base_vanishing_point_x + 
            env->current_curve_direction * CURVE_VANISHING_POINT_SHIFT;        
        // Test road edges at different y positions
        for (int j = 0; j < 5; j++) {
            char desc[64];
            sprintf(desc, "Y=%.2f (%.0f%% from bottom)", 
                    test_y_positions[j], 
                    (1.0f - (test_y_positions[j] - VANISHING_POINT_Y) / 
                     (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y)) * 100);
            print_road_edges(env, test_y_positions[j], desc);
        }        
        // Validate road width at bottom
        float bottom_left = road_edge_x(env, PLAYABLE_AREA_BOTTOM, 0, true);
        float bottom_right = road_edge_x(env, PLAYABLE_AREA_BOTTOM, 0, false);
        float road_width = bottom_right - bottom_left;
        printf("\nRoad width at bottom: %.2f (should be ~%.2f)\n", 
               road_width, ROAD_LEFT_OFFSET + ROAD_RIGHT_OFFSET);        
        // Validate road edges meet at vanishing point
        float top_left = road_edge_x(env, VANISHING_POINT_Y, 0, true);
        float top_right = road_edge_x(env, VANISHING_POINT_Y, 0, false);
        printf("Road edge difference at vanishing point: %.2f (should be ~0)\n", 
               fabsf(top_right - top_left));
    }
}

// car_x_in_lane
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

// Event handling
void handleEvents(int* running, Enduro* env) {
    env->actions = 0; // Default action is noop
    if (WindowShouldClose()) {
        *running = 0;
    }
    if (IsKeyDown(KEY_LEFT)) {
        env->actions = 1; // Move left
    } else if (IsKeyDown(KEY_RIGHT)) {
        env->actions = 2; // Move right
    }
    if (IsKeyDown(KEY_UP)) {
        env->actions = 3; // Speed up
    } else if (IsKeyDown(KEY_DOWN)) {
        env->actions = 4; // Slow down
    }
}

void initRaylib() {
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Enduro Port Framework");
    SetTargetFPS(60);
}

void loadTextures(GameState* gameState) {
    // Load background and mountain textures for different times of day
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

    // Initialize the transition times (in seconds)
    gameState->backgroundTransitionTimes[0] = 20.0f;
    gameState->backgroundTransitionTimes[1] = 40.0f;
    gameState->backgroundTransitionTimes[2] = 60.0f;
    gameState->backgroundTransitionTimes[3] = 100.0f;
    gameState->backgroundTransitionTimes[4] = 108.0f;
    gameState->backgroundTransitionTimes[5] = 114.0f;
    gameState->backgroundTransitionTimes[6] = 116.0f;
    gameState->backgroundTransitionTimes[7] = 120.0f;
    gameState->backgroundTransitionTimes[8] = 124.0f;
    gameState->backgroundTransitionTimes[9] = 130.0f;
    gameState->backgroundTransitionTimes[10] = 134.0f;
    gameState->backgroundTransitionTimes[11] = 138.0f;
    gameState->backgroundTransitionTimes[12] = 170.0f;
    gameState->backgroundTransitionTimes[13] = 198.0f;
    gameState->backgroundTransitionTimes[14] = 214.0f;
    gameState->backgroundTransitionTimes[15] = 232.0f; // Last transition

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

    // Initialize animation variables
    gameState->carAnimationTimer = 0.0f;
    gameState->carAnimationInterval = 0.5f; // Moderate animation speed
    gameState->showLeftTread = true;

    // Initialize elapsed time and background index
    gameState->elapsedTime = 0.0f;
    gameState->backgroundIndex = 0;

    // Initialize animation variables
    gameState->carAnimationTimer = 0.0f;
    gameState->carAnimationInterval = 1.0f; // Initial interval, will be updated based on speed
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
    // Unload player car textures
    UnloadTexture(gameState->playerCarLeftTreadTexture);
    UnloadTexture(gameState->playerCarRightTreadTexture);
    CloseWindow();
}

void updateCarAnimation(GameState* gameState, Enduro* env) {
    // Update the animation interval based on the player's speed
    // Faster speed means faster alternation
    float minInterval = 0.1f;  // Minimum interval at max speed
    float maxInterval = 1.0f;  // Maximum interval at min speed
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
            gameState->scoreDigitOffsets[i] += 0.5f; // Adjust scroll speed as needed
            if (gameState->scoreDigitOffsets[i] >= DIGIT_HEIGHT) {
                gameState->scoreDigitOffsets[i] = 0.0f;
                gameState->scoreDigitCurrents[i] = gameState->scoreDigitNexts[i];
                gameState->scoreDigitScrolling[i] = false; // Stop scrolling
            }
        }
    }
}

void updateBackground(GameState* gameState) {
    // Update elapsed time
    gameState->elapsedTime += GetFrameTime(); // Time since last frame in seconds

    // Total duration of the cycle
    float totalDuration = gameState->backgroundTransitionTimes[15];

    // If elapsed time exceeds total duration, reset it
    if (gameState->elapsedTime >= totalDuration) {
        gameState->elapsedTime -= totalDuration;
        gameState->backgroundIndex = 0;
    }

    // Determine the current background index
    while (gameState->backgroundIndex < 15 &&
           gameState->elapsedTime >= gameState->backgroundTransitionTimes[gameState->backgroundIndex]) {
        gameState->backgroundIndex++;
    }

    gameState->currentBackgroundIndex = gameState->backgroundIndex % 16; // Ensure index is within 0-15

    // No need for victory condition check here unless it's tied to background index
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


// void renderScoreboard(GameState* gameState) {
//     // Positions and sizes
//     int digitWidth = DIGIT_WIDTH;
//     int digitHeight = DIGIT_HEIGHT;
//     // Convert bottom-left coordinates to top-left origin
//     int scoreStartX = 56 + digitWidth;
//     int scoreStartY = 173 - digitHeight;
//     int dayX = 56;
//     int dayY = 188 - digitHeight;
//     int carsX = 72;
//     int carsY = 188 - digitHeight;

//     // Render score with scrolling effect
//     for (int i = 0; i < SCORE_DIGITS; ++i) {
//         int digitX = scoreStartX + i * digitWidth;
//         Texture2D currentDigitTexture;
//         Texture2D nextDigitTexture;

//         if (i == SCORE_DIGITS - 1) {
//             // Use yellow digits for the last digit
//             currentDigitTexture = gameState->yellowDigitTextures[gameState->scoreDigitCurrents[i]];
//             nextDigitTexture = gameState->yellowDigitTextures[gameState->scoreDigitNexts[i]];
//         } else {
//             // Use regular digits
//             currentDigitTexture = gameState->digitTextures[gameState->scoreDigitCurrents[i]];
//             nextDigitTexture = gameState->digitTextures[gameState->scoreDigitNexts[i]];
//         }

//         if (gameState->scoreDigitScrolling[i]) {
//             // Scrolling effect for this digit
//             float offset = gameState->scoreDigitOffsets[i];
//             // Render current digit moving up
//             Rectangle srcRectCurrent = { 0, 0, digitWidth, digitHeight - (int)offset };
//             Rectangle destRectCurrent = { digitX, scoreStartY + (int)offset, digitWidth, digitHeight - (int)offset };
//             DrawTextureRec(currentDigitTexture, srcRectCurrent, (Vector2){ destRectCurrent.x, destRectCurrent.y }, WHITE);
//             // Render next digit coming up from below
//             Rectangle srcRectNext = { 0, digitHeight - (int)offset, digitWidth, (int)offset };
//             Rectangle destRectNext = { digitX, scoreStartY, digitWidth, (int)offset };
//             DrawTextureRec(nextDigitTexture, srcRectNext, (Vector2){ destRectNext.x, destRectNext.y }, WHITE);
//         } else {
//             // No scrolling, render the current digit normally
//             DrawTexture(currentDigitTexture, digitX, scoreStartY, WHITE);
//         }
//     }

//     // Render day number
//     int day = gameState->day % 10; // Probably max 10 days possible
//     int dayTextureIndex = day;
//     if (gameState->victoryAchieved) {
//         // Green day digits during victory
//         Texture2D greenDigitTexture = gameState->greenDigitTextures[dayTextureIndex];
//         DrawTexture(greenDigitTexture, dayX, dayY, WHITE);
//     } else {
//         // Use normal digits
//         Texture2D digitTexture = gameState->digitTextures[dayTextureIndex];
//         DrawTexture(digitTexture, dayX, dayY, WHITE);
//     }
//     // Render "CAR" digit or flags for cars to pass
//     if (gameState->victoryAchieved) {
//         // Alternate between level_complete_flag_left and level_complete_flag_right
//         Texture2D flagTexture = gameState->showLeftFlag ? gameState->levelCompleteFlagLeftTexture : gameState->levelCompleteFlagRightTexture;
//         Rectangle destRect = { carsX, carsY, digitWidth * 4, digitHeight };
//         DrawTextureEx(flagTexture, (Vector2){ destRect.x, destRect.y }, 0.0f, 1.0f, WHITE);
//     } else {
//         // Render "CAR" digit for the first position in cars to pass
//         DrawTexture(gameState->carDigitTexture, carsX, carsY, WHITE);
//         // Render the remaining digits for cars to pass
//         int cars = gameState->carsToPass;
//         for (int i = 1; i < CARS_DIGITS; ++i) {
//             int digit = (cars / (int)pow(10, CARS_DIGITS - i - 1)) % 10;
//             int digitX = carsX + i * digitWidth + i * 1;
//             DrawTexture(gameState->digitTextures[digit], digitX, carsY, WHITE);
//         }
//     }
// }

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
    // Adjust the mountain position based on the road's curve direction
    float speed = 1.0f; // Adjust the speed as needed
    int mountainWidth = gameState->mountainTextures[0].width;

    if (env->current_curve_direction == -1) { // Turning left
        gameState->mountainPosition += speed;
        if (gameState->mountainPosition >= mountainWidth) {
            gameState->mountainPosition -= mountainWidth;
        }
    } else if (env->current_curve_direction == 1) { // Turning right
        gameState->mountainPosition -= speed;
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
        int mountainY = 45; // Adjust as needed

        // Calculate the player's offset from the center
        float playerCenterX = (PLAYER_MIN_X + PLAYER_MAX_X) / 2.0f;
        float playerOffset = env->player_x - playerCenterX;

        // Apply a parallax factor to make the mountains move with the player
        float parallaxFactor = 2.0f; // Adjust as needed for the desired effect
        float adjustedOffset = -playerOffset * parallaxFactor;

        // Base mountain X position (centered)
        int baseMountainX = SCREEN_WIDTH / 2 - mountainWidth / 2;

        // Adjust mountain X position
        int mountainX = (int)(baseMountainX + adjustedOffset);
        int mountainYPosition = mountainY; // Keep Y position consistent

        // Draw the mountain texture once
        DrawTexture(mountainTexture, mountainX, mountainYPosition, WHITE);

        // Handle wrapping if necessary
        if (mountainX > SCREEN_WIDTH) {
            // If the mountain has moved entirely off the right edge, wrap it around to the left
            DrawTexture(mountainTexture, mountainX - mountainWidth, mountainYPosition, WHITE);
        } else if (mountainX + mountainWidth < 0) {
            // If the mountain has moved entirely off the left edge, wrap it around to the right
            DrawTexture(mountainTexture, mountainX + mountainWidth, mountainYPosition, WHITE);
        }
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

    // Set clipping rectangle to the playable area
    Rectangle clipRect = { PLAYABLE_AREA_LEFT, VANISHING_POINT_Y, PLAYABLE_AREA_RIGHT - PLAYABLE_AREA_LEFT, PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y };
    BeginScissorMode(clipRect.x, clipRect.y, clipRect.width, clipRect.height);

    // Render road edges
    for (float y = VANISHING_POINT_Y; y <= PLAYABLE_AREA_BOTTOM; y += 0.75f) {
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

        DrawPixel((int)left_edge, (int)adjusted_y, roadColor);
        DrawPixel((int)right_edge, (int)adjusted_y, roadColor);
    }

    // Render enemy cars with specified scaling stages
    for (int i = 0; i < env->numEnemies; i++) {
        Car* car = &env->enemyCars[i];

        // Determine the car scale based on the seven-stage progression
        float car_scale;
        if (car->y <= 68.0f) car_scale = 2.0f / 16.0f;        // Stage 1
        else if (car->y <= 74.0f) car_scale = 4.0f / 16.0f;   // Stage 2
        else if (car->y <= 86.0f) car_scale = 4.0f / 12.0f;   // Stage 3
        else if (car->y <= 100.0f) car_scale = 6.0f / 12.0f;  // Stage 4
        else if (car->y <= 110.0f) car_scale = 8.0f / 12.0f;  // Stage 5
        else if (car->y <= 120.0f) car_scale = 10.0f / 12.0f; // Stage 6
        else if (car->y <= 135.0f) car_scale = 12.0f / 12.0f; // Stage 7
        else car_scale = 1.0f;                                // Normal size

        // Select the correct texture based on the car's color and current tread
        int colorIndex = car->colorIndex; // Use the colorIndex assigned at spawn
        int treadIndex = env->gameState.showLeftTread ? 0 : 1;
        Texture2D carTexture = env->gameState.enemyCarTextures[colorIndex][treadIndex];

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
