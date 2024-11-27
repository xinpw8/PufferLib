// puffer_enduro.h

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
#include <limits.h>
#include <string.h>
#include "raylib.h"

// Debug prints
#ifdef DEBUG
    #define DEBUG_PRINT(...) printf(__VA_ARGS__)
    #define DEBUG_FPRINT(stream, ...) fprintf((stream), __VA_ARGS__)

#else
    #define DEBUG_PRINT(...) // No-op
    #define DEBUG_FPRINT(stream, ...) // No-op

#endif

// Constant defs
#define MAX_ENEMIES 10
#define OBSERVATIONS_MAX_SIZE (6 + 2 * MAX_ENEMIES + 3 + 1)
#define TARGET_FPS 60 // Used to calculate wiggle spawn frequency
#define LOG_BUFFER_SIZE 4096
#define SCREEN_WIDTH 160
#define SCREEN_HEIGHT 210
#define PLAYABLE_AREA_TOP 0
#define PLAYABLE_AREA_BOTTOM 154
#define PLAYABLE_AREA_LEFT 8
#define PLAYABLE_AREA_RIGHT 160
#define ACTION_HEIGHT (PLAYABLE_AREA_BOTTOM - PLAYABLE_AREA_TOP)
#define CAR_WIDTH 16
#define CAR_HEIGHT 11
#define CRASH_NOOP_DURATION_CAR_VS_CAR 90 // 60 // How long controls are disabled after car v car collision
#define CRASH_NOOP_DURATION_CAR_VS_ROAD 20 // How long controls are disabled after car v road edge collision
#define INITIAL_CARS_TO_PASS 200
#define VANISHING_POINT_X 86 // 110
#define VANISHING_POINT_Y 52
#define VANISHING_POINT_TRANSITION_SPEED 1.0f
#define CURVE_TRANSITION_SPEED 0.05f // 0.02f
#define LOGICAL_VANISHING_Y (VANISHING_POINT_Y + 12)  // Separate logical vanishing point for cars disappearing
#define INITIAL_PLAYER_X 77.0f // 86.0f // ((PLAYABLE_AREA_RIGHT - PLAYABLE_AREA_LEFT)/2 + CAR_WIDTH/2)
#define PLAYER_MAX_Y (ACTION_HEIGHT - CAR_HEIGHT) // Max y is car length from bottom
#define PLAYER_MIN_Y (ACTION_HEIGHT - CAR_HEIGHT - 9) // Min y is 2 car lengths from bottom
#define ACCELERATION_RATE 0.2f // 0.05f
#define DECELERATION_RATE 0.001f
#define MIN_SPEED -2.5f
#define MAX_SPEED 7.5f // 5.5f
#define ENEMY_CAR_SPEED 0.1f 
// Times of day logic
#define NUM_BACKGROUND_TRANSITIONS 16
// Seconds spent in each time of day
static const float BACKGROUND_TRANSITION_TIMES[] = {
    20.0f, 40.0f, 60.0f, 100.0f, 108.0f, 114.0f, 116.0f, 120.0f,
    124.0f, 130.0f, 134.0f, 138.0f, 170.0f, 198.0f, 214.0f, 232.0f
};

// static const float BACKGROUND_TRANSITION_TIMES[] = {
//     2.0f, 4.0f, 6.0f, 10.0f, 10.8f, 11.0f, 11.6f, 12.0f,
//     12.4f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f
// };

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

#define CHECK_AND_INCREMENT_INDEX(index, size) \
    do { \
        if ((index) >= (size)) { \
            DEBUG_FPRINT(stderr, "Error: obs_index %d out of bounds! (size = %d)\n", index, size); \
            exit(EXIT_FAILURE); \
        } \
        index++; \
    } while (0)


// Log structs
typedef struct Log {
    float episode_return;
    float episode_length;
    float score;
    float reward;
    float stay_on_road_reward;
    float passed_cars;
    float passed_by_enemy;
    int cars_to_pass;
    float days_completed;
    float days_failed;
    float collisions_player_vs_car;
    float collisions_player_vs_road;
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
    float last_y; // y post last step
    int passed; // Flag to indicate if car has been passed by player
    int colorIndex; // Car color idx (0-5)
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
    Texture2D enemyCarNightTailLightsTexture;
    Texture2D enemyCarNightFogTailLightsTexture;
    // For car animation
    float carAnimationTimer;
    float carAnimationInterval;
    unsigned char showLeftTread;
    float mountainPosition; // Position of the mountain texture
    // Variables for alternating flags
    unsigned char victoryAchieved;
    int flagTimer;
    unsigned char showLeftFlag; // true shows left flag, false shows right flag
    int victoryDisplayTimer;    // Timer for how long victory effects have been displayed
    // Variables for scrolling yellow digits
    float yellowDigitOffset; // Offset for scrolling effect
    int yellowDigitCurrent;  // Current yellow digit being displayed
    int yellowDigitNext;     // Next yellow digit to scroll in
    // Variables for scrolling digits
    float scoreDigitOffsets[SCORE_DIGITS];   // Offset for scrolling effect for each digit
    int scoreDigitCurrents[SCORE_DIGITS];    // Current digit being displayed for each position
    int scoreDigitNexts[SCORE_DIGITS];       // Next digit to scroll in for each position
    unsigned char scoreDigitScrolling[SCORE_DIGITS];  // Scrolling state for each digit
    int scoreTimer; // Timer to control score increment
    int day;
    int carsLeftGameState;
    int score; // Score for scoreboard rendering
    // Background state vars
    float backgroundTransitionTimes[16];
    int backgroundIndex;
    int currentBackgroundIndex;
    int previousBackgroundIndex;
    float elapsedTime;
    // Variable needed from Enduro to maintain separation
    float speed;
    float min_speed;
    float max_speed;
    int current_curve_direction;
    float current_curve_factor;
    float player_x;
    float player_y;
    float initial_player_x;
    float vanishing_point_x;
    float t_p;
    unsigned char dayCompleted;
} GameState;

// Game environment struct
typedef struct Enduro {
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    unsigned char* truncateds;
    LogBuffer* log_buffer;
    Log log;
    size_t obs_size;
    int num_envs;
    float width;
    float height;
    float car_width;
    float car_height;
    int max_enemies;
    float elapsedTimeEnv;
    int initial_cars_to_pass;
    float min_speed;
    float max_speed;
    float player_x;
    float player_y;
    float speed;
    int score;
    int day;
    int lane;
    int step_count;
    int numEnemies;
    int carsToPass;
    float collision_cooldown_car_vs_car; // Timer for car vs car collisions
    float collision_cooldown_car_vs_road; // Timer for car vs road edge collisions
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
    float current_step_threshold;
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
    unsigned char wiggle_active;        // Whether the wiggle is active
    // Player car acceleration
    int currentGear;
    float gearSpeedThresholds[4]; // Speeds at which gear changes occur
    float gearAccelerationRates[4]; // Acceleration rates per gear
    float gearTimings[4]; // Time durations per gear
    float gearElapsedTime; // Time spent in current gear
    // Enemy spawning
    float enemySpawnTimer;
    float enemySpawnInterval; // Spawn interval based on current stage

    // Enemy movement speed
    float enemySpeed;

    // Day completed (victory) variables
    unsigned char dayCompleted;

    // Logging
    float last_road_left;
    float last_road_right;
    int closest_edge_lane;
    int last_spawned_lane;
    float totalAccelerationTime;

    // Rendering
    float parallaxFactor;

    // Variables for time of day
    float dayTransitionTimes[NUM_BACKGROUND_TRANSITIONS];
    int dayTimeIndex;
    int currentDayTimeIndex;
    int previousDayTimeIndex;

    // RNG
    unsigned int rng_state;
    unsigned int index;
    int reset_count;
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
    GameState gameState;
} Client;

// Prototypes
// RNG
unsigned int xorshift32(unsigned int *state);
// LogBuffer functions
LogBuffer* allocate_logbuffer(int size);
void free_logbuffer(LogBuffer* buffer);
void add_log(LogBuffer* logs, Log* log);
Log aggregate_and_clear(LogBuffer* logs);

// Environment functions
void allocate(Enduro* env);
void init(Enduro* env, int seed, int env_index);
void free_allocated(Enduro* env);
// void reset_round(Enduro* env);
void reset(Enduro* env);
unsigned char check_collision(Enduro* env, Car* car);
int get_player_lane(Enduro* env);
float get_car_scale(float y);
void add_enemy_car(Enduro* env);
void update_time_of_day(Enduro* env);
void update_vanishing_point(Enduro* env, float offset);
void accelerate(Enduro* env);
void compute_enemy_car_rewards(Enduro* env);
void c_step(Enduro* env);
void update_road_curve(Enduro* env);
float quadratic_bezier(float bottom_x, float control_x, float top_x, float t);
float road_edge_x(Enduro* env, float y, float offset, unsigned char left);
float car_x_in_lane(Enduro* env, int lane, float y);
void compute_observations(Enduro* env);

// Client functions
Client* make_client(Enduro* env);
void close_client(Client* client, Enduro* env);
void render_car(Client* client, GameState* gameState);
void handleEvents(int* running, Enduro* env);

// Debugging
void debug_enduro_allocation(Enduro* env);

// GameState rendering functions
void initRaylib();
void loadTextures(GameState* gameState);
void updateCarAnimation(GameState* gameState);
void updateScoreboard(GameState* gameState);
void updateBackground(GameState* gameState);
void renderBackground(GameState* gameState);
void renderScoreboard(GameState* gameState);
void updateMountains(GameState* gameState);
void renderMountains(GameState* gameState);
void updateVictoryEffects(GameState* gameState);
void c_render(Client* client, Enduro* env);
void cleanup(GameState* gameState);

// Function defs
// RNG
unsigned int xorshift32(unsigned int *state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

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
    // printf("Log: %f, %f, %f\n", log->episode_return, log->episode_length, log->score);
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
        log.reward += logs->logs[i].reward;
        log.stay_on_road_reward += logs->logs[i].stay_on_road_reward;
        log.passed_cars += logs->logs[i].passed_cars;
        log.passed_by_enemy += logs->logs[i].passed_by_enemy;
        log.cars_to_pass += logs->logs[i].cars_to_pass;
        log.days_completed += logs->logs[i].days_completed;
        log.days_failed += logs->logs[i].days_failed;
        log.collisions_player_vs_car += logs->logs[i].collisions_player_vs_car;
        log.collisions_player_vs_road += logs->logs[i].collisions_player_vs_road;
    }
    log.episode_return /= logs->idx;
    log.episode_length /= logs->idx;
    log.score /= logs->idx;
    log.reward /= logs->idx;
    log.stay_on_road_reward /= logs->idx;
    log.passed_cars /= logs->idx;
    log.passed_by_enemy /= logs->idx;
    log.cars_to_pass /= logs->idx;
    log.days_completed /= logs->idx;
    log.days_failed /= logs->idx;
    log.collisions_player_vs_car /= logs->idx;
    log.collisions_player_vs_road /= logs->idx;
    logs->idx = 0;
    return log;
}

void init(Enduro* env, int seed, int env_index) {
    env->index = env_index;
    env->rng_state = seed;

    if (seed == 10) { // Activate with seed==0
        // Start the environment at the beginning of the day
        env->elapsedTimeEnv = 0.0f;
        env->currentDayTimeIndex = 0;
        env->previousDayTimeIndex = NUM_BACKGROUND_TRANSITIONS - 1;
    } else {
        // Randomize elapsed time within the day's total duration
        float total_day_duration = BACKGROUND_TRANSITION_TIMES[NUM_BACKGROUND_TRANSITIONS - 1];
        env->elapsedTimeEnv = ((float)xorshift32(&env->rng_state) / UINT32_MAX) * total_day_duration;

        // Determine the current time index
        env->currentDayTimeIndex = 0;
        for (int i = 0; i < NUM_BACKGROUND_TRANSITIONS - 1; i++) {
            if (env->elapsedTimeEnv >= env->dayTransitionTimes[i] &&
                env->elapsedTimeEnv < env->dayTransitionTimes[i + 1]) {
                env->currentDayTimeIndex = i;
                break;
            }
        }

        // Handle the last interval
        if (env->elapsedTimeEnv >= BACKGROUND_TRANSITION_TIMES[NUM_BACKGROUND_TRANSITIONS - 1]) {
            env->currentDayTimeIndex = NUM_BACKGROUND_TRANSITIONS - 1;
        }
    }

    if (env->index % 100 == 0) {
    printf("Environment #%d state after init: elapsedTimeEnv=%f, currentDayTimeIndex=%d\n",
           env_index, env->elapsedTimeEnv, env->currentDayTimeIndex);
    }

    if (!env->observations || !env->actions || !env->rewards || !env->terminals || !env->truncateds) {
        DEBUG_FPRINT(stderr, "Error: Attempting to initialize with unallocated pointers\n");
        exit(EXIT_FAILURE);
    }

    env->numEnemies = 0;
    for (int i = 0; i < MAX_ENEMIES; i++) {
        env->enemyCars[i].lane = -1; // Default invalid lane
        env->enemyCars[i].y = 0.0f;
        env->enemyCars[i].passed = 0;
    }

    env->obs_size = OBSERVATIONS_MAX_SIZE; // Adding missing time_of_day and carsToPass


if (env->obs_size < 0 || env->obs_size > INT_MAX / sizeof(float)) {
    DEBUG_FPRINT(stderr, "Error: obs_size overflow in init\n");
    exit(EXIT_FAILURE);
}
    env->max_enemies = MAX_ENEMIES;
    env->score = 0;
    env->numEnemies = 0;
    env->player_x = INITIAL_PLAYER_X;
    // printf("init: player_x set to %f\n", env->player_x);
    env->player_y = PLAYER_MAX_Y;
    env->speed = MIN_SPEED;
    env->carsToPass = INITIAL_CARS_TO_PASS;
    env->width = SCREEN_WIDTH;
    env->height = SCREEN_HEIGHT;
    env->car_width = CAR_WIDTH;
    env->car_height = CAR_HEIGHT;

    memcpy(env->dayTransitionTimes, BACKGROUND_TRANSITION_TIMES, sizeof(BACKGROUND_TRANSITION_TIMES));
    
    env->step_count = 0;
    env->collision_cooldown_car_vs_car = 0.0f;
    DEBUG_PRINT("collision_cooldown_car_vs_car IMMEDIATELY AFTER INIT IN INIT = %f at line %i\n", env->collision_cooldown_car_vs_car, __LINE__);

    env->collision_cooldown_car_vs_road = 0.0f;
    env->action_height = ACTION_HEIGHT;
    env->elapsedTimeEnv = 0.0f;
    env->enemySpawnTimer = 0.0f;
    env->enemySpawnInterval = 0.8777f;
    env->last_spawned_lane = -1;
    env->closest_edge_lane = -1;
    env->totalAccelerationTime = 0.0f;
    env->base_vanishing_point_x = 86.0f;
    env->current_vanishing_point_x = 86.0f;
    env->target_vanishing_point_x = 86.0f;
    env->vanishing_point_x = 86.0f;
    env->initial_player_x = INITIAL_PLAYER_X;
    env->player_y = PLAYER_MAX_Y;
    env->min_speed = MIN_SPEED;
    env->enemySpeed = ENEMY_CAR_SPEED;
    env->max_speed = MAX_SPEED;
    env->initial_cars_to_pass = INITIAL_CARS_TO_PASS;
    env->day = 1;
    env->current_curve_direction = CURVE_STRAIGHT;
    env->current_curve_factor = 0.0f;
    env->target_curve_factor = 0.0f;
    env->current_step_threshold = 0.0f;
    env->wiggle_y = VANISHING_POINT_Y;
    env->wiggle_speed = WIGGLE_SPEED;
    env->wiggle_length = WIGGLE_LENGTH;
    env->wiggle_amplitude = WIGGLE_AMPLITUDE;
    env->wiggle_active = true;
    env->currentGear = 0;
    env->gearElapsedTime = 0.0f;
    env->gearTimings[0] = 4.0f;
    env->gearTimings[1] = 2.5f;
    env->gearTimings[2] = 3.25f;
    env->gearTimings[3] = 1.5f;
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
        env->gearAccelerationRates[i] = gearSpeedIncrement / (gearTime * TARGET_FPS);
        cumulativeSpeed = env->gearSpeedThresholds[i];
    }

    // // Initialize background indices
    // env->dayTimeIndex = 0;

    // env->currentDayTimeIndex = 0;
    // env->previousDayTimeIndex = 15;

    // Randomize the initial time of day for each environment
    float total_day_duration = BACKGROUND_TRANSITION_TIMES[15];
    env->elapsedTimeEnv = ((float)rand_r(&env->rng_state) / RAND_MAX) * total_day_duration;
    env->currentDayTimeIndex = 0;
    env->dayTimeIndex = 0;
    env->previousDayTimeIndex = 0;
    // printf("init: elapsedTimeEnv set to %f\n", env->elapsedTimeEnv);

    // Advance currentDayTimeIndex to match randomized elapsedTimeEnv
    for (int i = 0; i < NUM_BACKGROUND_TRANSITIONS; i++) {
        if (env->elapsedTimeEnv >= env->dayTransitionTimes[i]) {
            env->currentDayTimeIndex = i;
        } else {
            break;
        }
    }
    env->previousDayTimeIndex = (env->currentDayTimeIndex > 0) ? env->currentDayTimeIndex - 1 : NUM_BACKGROUND_TRANSITIONS - 1;


    env->terminals[0] = 0;
    env->truncateds[0] = 0;
    // Debugging
    // Reset rewards and logs
    env->rewards[0] = 0.0f;
    env->log.episode_return = 0;
    env->log.episode_length = 0;
    env->log.score = 0;
    env->log.reward = 0;
    env->log.stay_on_road_reward = 0;
    env->log.passed_cars = 0;
    env->log.passed_by_enemy = 0;
    env->log.cars_to_pass = INITIAL_CARS_TO_PASS;
    // env->log.days_completed = 0;
    // env->log.days_failed = 0;
    env->log.collisions_player_vs_car = 0;
    env->log.collisions_player_vs_road = 0;
    }

void allocate(Enduro* env) {
    if (!env || env->num_envs <= 0) {
        DEBUG_FPRINT(stderr, "Error: Invalid environment or num_envs\n");
        exit(EXIT_FAILURE);
    }

    env->rewards = (float*)malloc(sizeof(float) * env->num_envs);
    if (!env->rewards) {
        DEBUG_FPRINT(stderr, "Failed to allocate memory for rewards\n");
        exit(EXIT_FAILURE);
    }
    memset(env->rewards, 0, sizeof(float) * env->num_envs);

    if (env->obs_size == 0 || env->obs_size > SIZE_MAX / sizeof(float)) {
        DEBUG_FPRINT(stderr, "Error: obs_size is invalid or too large (%zu)\n", env->obs_size);
        exit(EXIT_FAILURE);
    }

env->observations = (float*)calloc(env->obs_size, sizeof(float));
if (!env->observations) {
    DEBUG_FPRINT(stderr, "Error: Failed to allocate observations array\n");
    exit(EXIT_FAILURE);
}

DEBUG_PRINT("Debug: Allocating observations array with size = %zu bytes\n", env->obs_size * sizeof(float));

    env->actions = (int*)calloc(1, sizeof(int));
    if (!env->actions) {
        DEBUG_FPRINT(stderr, "Failed to allocate memory for actions\n");
        exit(EXIT_FAILURE);
    }

    env->rewards = (float*)calloc(1, sizeof(float));
    if (!env->rewards) {
        DEBUG_FPRINT(stderr, "Failed to allocate memory for rewards\n");
        exit(EXIT_FAILURE);
    }

    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    if (!env->terminals) {
        DEBUG_FPRINT(stderr, "Failed to allocate memory for terminals\n");
        exit(EXIT_FAILURE);
    }

    env->truncateds = (unsigned char*)calloc(1, sizeof(unsigned char));
    if (!env->truncateds) {
        DEBUG_FPRINT(stderr, "Failed to allocate memory for truncateds\n");
        exit(EXIT_FAILURE);
    }

    env->log_buffer = allocate_logbuffer(LOG_BUFFER_SIZE);
    if (!env->log_buffer) {
        DEBUG_FPRINT(stderr, "Failed to allocate memory for log buffer\n");
        exit(EXIT_FAILURE);
    }
}

void free_allocated(Enduro* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env->truncateds);
    free_logbuffer(env->log_buffer);

}


// void reset_round(Enduro* env) {
//     env->player_x = env->initial_player_x;
//     env->player_y = PLAYER_MAX_Y;
//     env->speed = env->min_speed;
//     env->numEnemies = 0;
//     env->step_count = 0;
//     env->collision_cooldown_car_vs_car = 0;
//     env->collision_cooldown_car_vs_road = 0;
//     env->road_scroll_offset = 0.0f;
// }

// Reset all init vars
void reset(Enduro* env) {
    // Debug at top of reset
    DEBUG_PRINT("line 630: calling debug_enduro_allocation at top of reset()\n");
    debug_enduro_allocation(env);

    // // No random after first reset
    // int reset_seed = (env->reset_count == 0) ? xorshift32(&env->rng_state) : 0;

    int reset_seed = xorshift32(&env->rng_state);

    // Reset environment with the appropriate seed
    init(env, reset_seed, env->index);

    // Debug at bottom of reset
    DEBUG_PRINT("line 648: calling debug_enduro_allocation at bottom of reset()\n");
    debug_enduro_allocation(env);

    env->reset_count += 1;
}


unsigned char check_collision(Enduro* env, Car* car) {
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
    env->lane = (int)((player_center_x - left_edge) / lane_width);
    if (env->lane < 0) env->lane = 0;
    if (env->lane > 2) env->lane = 2;
    return env->lane;
}

float get_car_scale(float y) {
    float depth = (y - VANISHING_POINT_Y) / (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y);
    return fmaxf(0.1f, 0.9f * depth);
}

void add_enemy_car(Enduro* env) {
    if (env->numEnemies >= MAX_ENEMIES) return;

    int player_lane = get_player_lane(env);
    int furthest_lane;
    int possible_lanes[NUM_LANES];
    int num_possible_lanes = 0;

    // Determine the furthest lane from the player
    if (player_lane == 0) {
        furthest_lane = 2;
    } else if (player_lane == 2) {
        furthest_lane = 0;
    } else {
        // Player is in the middle lane
        // Decide based on player's position relative to the road center
        float player_center_x = env->player_x + CAR_WIDTH / 2.0f;
        float road_center_x = (road_edge_x(env, env->player_y, 0, true) +
                            road_edge_x(env, env->player_y, 0, false)) / 2.0f;
        if (player_center_x < road_center_x) {
            furthest_lane = 2; // Player is on the left side, choose rightmost lane
        } else {
            furthest_lane = 0; // Player is on the right side, choose leftmost lane
        }
    }

    if (env->speed <= 0) {
        // Only spawn in the lane furthest from the player
        possible_lanes[num_possible_lanes++] = furthest_lane;
    } else {
        for (int i = 0; i < NUM_LANES; i++) {
            possible_lanes[num_possible_lanes++] = i;
        }
    }

    if (num_possible_lanes == 0) {
        return; // Rare
    }

    // Randomly select a lane
    int lane = possible_lanes[rand() % num_possible_lanes];
    // Preferentially spawn in the last_spawned_lane 30% of the time
    if (rand() % 100 < 60 && env->last_spawned_lane != -1) {
        lane = env->last_spawned_lane;
    }
    env->last_spawned_lane = lane;
    Car car = { .lane = lane, .passed = false, .colorIndex = rand() % 6 };
    car.y = (env->speed > 0) ? VANISHING_POINT_Y + 10.0f : PLAYABLE_AREA_BOTTOM + CAR_HEIGHT;
    // Ensure minimum spacing between cars in the same lane
    float depth = (car.y - VANISHING_POINT_Y) / (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y);
    float scale = fmax(0.1f, 0.9f * depth + 0.1f);
    float scaled_car_length = CAR_HEIGHT * scale;
    // Randomize min spacing between 1.0f and 6.0f car lengths
    float dynamic_spacing_factor = rand() % 6 + 0.5f;
    float min_spacing = dynamic_spacing_factor * scaled_car_length;
    for (int i = 0; i < env->numEnemies; i++) {
        Car* existing_car = &env->enemyCars[i];
        if (existing_car->lane == car.lane) {
            float y_distance = fabs(existing_car->y - car.y);
            if (y_distance < min_spacing) {
                // Too close, do not spawn this car
                // DEBUG_PRINT("Not spawning car due to spacing in same lane (distance %.2f, min spacing %.2f)\n", y_distance, min_spacing);
                return;
            }
        }
    }
    // Ensure not occupying all lanes within vertical range of 6 car lengths
    float min_vertical_range = 6.0f * CAR_HEIGHT;
    int lanes_occupied = 0;
    unsigned char lane_occupied[NUM_LANES] = { false };
    for (int i = 0; i < env->numEnemies; i++) {
        Car* existing_car = &env->enemyCars[i];
        float y_distance = fabs(existing_car->y - car.y);
        if (y_distance < min_vertical_range) {
            lane_occupied[existing_car->lane] = true;
        }
    }
    for (int i = 0; i < NUM_LANES; i++) {
        if (lane_occupied[i]) lanes_occupied++;
    }
    if (lanes_occupied >= NUM_LANES - 1 && !lane_occupied[lane]) {
        return;
    }
    env->enemyCars[env->numEnemies++] = car;
}

// Adjust base vanishing point with offset during curves
void update_vanishing_point(Enduro* env, float offset) {
    env->vanishing_point_x = env->base_vanishing_point_x + offset;
}

void update_time_of_day(Enduro* env) {
    float elapsedTime = env->elapsedTimeEnv;
    float totalDuration = env->dayTransitionTimes[15];

    if (elapsedTime >= totalDuration) {
        elapsedTime -= totalDuration;
        env->elapsedTimeEnv = elapsedTime; // Reset elapsed time
        env->dayTimeIndex = 0;
    }

    env->previousDayTimeIndex = env->currentDayTimeIndex;

    while (env->dayTimeIndex < 15 &&
           elapsedTime >= env->dayTransitionTimes[env->dayTimeIndex]) {
        env->dayTimeIndex++;
    }
    env->currentDayTimeIndex = env->dayTimeIndex % 16;
}

void validate_speed(Enduro* env) {
    if (env->speed < env->min_speed || env->speed > env->max_speed) {
        // printf("Speed out of range: %f (min: %f, max: %f)\n", env->speed, env->min_speed, env->max_speed);
        env->speed = fmaxf(env->min_speed, fminf(env->speed, env->max_speed)); // Clamp speed to valid range
    }
}

void validate_gear(Enduro* env) {
    if (env->currentGear < 0 || env->currentGear > 3) {
        // printf("Invalid gear: %d. Resetting to 0.\n", env->currentGear);
        env->currentGear = 0;
    }
}

void accelerate(Enduro* env) {
    validate_speed(env);
    validate_gear(env);

    if (env->speed < env->max_speed) {
        // Gear transition
        if (env->speed >= env->gearSpeedThresholds[env->currentGear] && env->currentGear < 3) {
            env->currentGear++;
            env->gearElapsedTime = 0.0f;
        }

        // Calculate new speed
        float accel = env->gearAccelerationRates[env->currentGear];
        float multiplier = (env->currentGear == 0) ? 4.0f : 2.0f;
        env->speed += accel * multiplier;

        // Clamp speed
        validate_speed(env);

        // Cap speed to gear threshold
        if (env->speed > env->gearSpeedThresholds[env->currentGear]) {
            env->speed = env->gearSpeedThresholds[env->currentGear];
        }
    }
    validate_speed(env);
}

void compute_enemy_car_rewards(Enduro* env) {
    for (int i = 0; i < env->numEnemies; i++) {
        if (i >= env->max_enemies) {
            DEBUG_FPRINT(stderr, "Error: Accessing enemyCars[%d] out of bounds\n", i);
            exit(EXIT_FAILURE);
        }

        Car* car = &env->enemyCars[i];
        if (car == NULL) {
            DEBUG_FPRINT(stderr, "Error: car pointer is NULL for enemyCars[%d]\n", i);
            exit(EXIT_FAILURE);
        }

        // Calculate the horizontal and vertical distance between the player and the enemy car
        // float car_x = car_x_in_lane(env, car->lane, car->y);
        // float horizontal_distance = fabs(car_x - env->player_x);
        float vertical_distance = fabs(car->y - env->player_y);

        // Manhattan distance
        // float current_distance = sqrt(horizontal_distance * horizontal_distance + vertical_distance * vertical_distance);
        float current_distance = sqrt(vertical_distance * vertical_distance);

        // Compute reward for approaching enemy cars
        if (car->last_y <= env->player_y) {
            float previous_distance = fabs(car->last_y - env->player_y);
            float distance_change = previous_distance - current_distance;

            // printf("Distance change: %.2f\n", distance_change);
            // Reward for getting closer
            if (distance_change > 0 && car->y < env->player_y) { // Enemy car is in front
                env->rewards[0] += distance_change * 0.0001f; // Scaled reward for reducing distance
                env->log.reward += distance_change * 0.0001f;
                // printf("Reward for getting closer: %.2f\n", distance_change * 0.0001f);
            }

            // Bonus reward for passing the enemy car
            if (car->y > env->player_y && !car->passed) { // Enemy car is now behind
                env->rewards[0] += 0.005f; // Fixed reward for passing
                env->log.reward += 0.005f;
                car->passed = 1; // Mark car as passed (log-only)
                // printf("Reward for passing enemy car\n");
            }
        }
    }
}

void c_step(Enduro* env) {  
env->rewards[0] = 0.00000000000f;

// Debug at top of step
DEBUG_PRINT("line 806: calling debug enduro allocation at top of step()\n");
debug_enduro_allocation(env);


    env->elapsedTimeEnv += (1.0f / TARGET_FPS);

    // Update time of day
    update_time_of_day(env);

    update_road_curve(env);
    env->log.episode_length += 1;
    env->terminals[0] = 0;
    env->road_scroll_offset += env->speed;

// Update enemy car positions
for (int i = 0; i < env->numEnemies; i++) {
    if (i >= env->max_enemies) {
    DEBUG_FPRINT(stderr, "Error: Accessing enemyCars[%d] out of bounds\n", i);
    exit(EXIT_FAILURE);
}

Car* car = &env->enemyCars[i];
compute_enemy_car_rewards(env);

    float movement_speed;
    float relative_speed;

    if (env->speed > 0) {
        // Gradually increase the speed difference between player and enemy cars
        float speed_range = MAX_SPEED - MIN_SPEED;
        float normalized_speed = (env->speed - MIN_SPEED) / speed_range; // Normalize between 0 and 1
        normalized_speed = fminf(fmaxf(normalized_speed, 0.0f), 1.0f);  // Clamp between 0 and 1
        relative_speed = env->enemySpeed - (normalized_speed * (env->enemySpeed - MIN_SPEED));
        movement_speed = -(relative_speed + (MIN_SPEED * 0.35f)); // Enemies move backward relative to player
        // DEBUG_PRINT("Movement speed POSITIVE: %.2f\n", movement_speed);
    } else if (env->speed < 0) {
        // Enemy cars move forward relative to the player
        // Handle negative speed (already smooth)
        movement_speed = env->speed * 0.75f; 
        // DEBUG_PRINT("Movement speed NEGATIVE: %.2f\n", movement_speed);
    } else {
        // Neutral case (stationary)
        movement_speed = 0.0f;
        // DEBUG_PRINT("Movement speed NEUTRAL: %.2f\n", movement_speed);
    }

    // Update car position
    car->y += movement_speed;

    // Debugging: Print out the final enemy speed
    // DEBUG_PRINT("Enemy car %d final position Y: %.2f\n", i, car->y);
}

    // // Update enemy car positions
    // for (int i = 0; i < env->numEnemies; i++) {
    //     Car* car = &env->enemyCars[i];
    //     // Compute movement speed adjusted for scaling
    //     float scale = get_car_scale(car->y);
    //     float movement_speed = env->speed * scale * 0.75f;
    //     // Update car position
    //     car->y += movement_speed;
    // }

    // Calculate road edges
    float road_left = road_edge_x(env, env->player_y, 0, true);
    float road_right = road_edge_x(env, env->player_y, 0, false) - CAR_WIDTH;
    // DEBUG_PRINT("Player X: %.2f, Road Left: %.2f, Road Right: %.2f\n", env->player_x, road_left, road_right);

// Reward for staying on the road and going faster
if (env->collision_invulnerability_timer <= 0.0f) {
    if (env->player_x > road_left && env->player_x < road_right && env->speed > 0) {
        env->rewards[0] += 0.0010000000f * env->speed;
        env->log.stay_on_road_reward += 0.001f * env->speed;
    }
}

    env->last_road_left = road_left;
    env->last_road_right = road_right;

    // Reduced handling on snow
    unsigned char isSnowStage = (env->currentDayTimeIndex == 3);
    float movement_amount = 0.5f; // Default
    if (isSnowStage) {
        movement_amount = 0.3f; // Snow
    }
    
    // DEBUG_PRINT("Before movement: player_x = %.2f\n", env->player_x);
    // Player movement logic == action space (Discrete[9])
    if (env->collision_cooldown_car_vs_car <= 0 && env->collision_cooldown_car_vs_road <= 0) {
        int act = env->actions[0];
        switch (act) {
            case ACTION_NOOP:
                break;
            case ACTION_FIRE:
                accelerate(env);
                break;
            case ACTION_RIGHT:
                env->player_x += movement_amount;
                if (env->player_x > road_right) env->player_x = road_right;
                break;
            case ACTION_LEFT:
                env->player_x -= movement_amount;
                if (env->player_x < road_left) env->player_x = road_left;
                break;
            case ACTION_DOWN:
                if (env->speed > env->min_speed) env->speed -= DECELERATION_RATE;
                break;
            case ACTION_DOWNRIGHT:
                if (env->speed > env->min_speed) env->speed -= DECELERATION_RATE;
                env->player_x += movement_amount;
                if (env->player_x > road_right) env->player_x = road_right;
                break;
            case ACTION_DOWNLEFT:
                if (env->speed > env->min_speed) env->speed -= DECELERATION_RATE;
                env->player_x -= movement_amount;
                if (env->player_x < road_left) env->player_x = road_left;
                break;
            case ACTION_RIGHTFIRE:
                accelerate(env);
                env->player_x += movement_amount;
                if (env->player_x > road_right) env->player_x = road_right;
                break;
            case ACTION_LEFTFIRE:
                accelerate(env);
                env->player_x -= movement_amount;
                if (env->player_x < road_left) env->player_x = road_left;
                break;
        }
    } else {

    if (env->collision_cooldown_car_vs_car > 0) {
    env->collision_cooldown_car_vs_car -= 1;
    }
    if (env->collision_cooldown_car_vs_road > 0) {
    env->collision_cooldown_car_vs_road -= 1;
    }

    // Drift towards furthest road edge
    if (env->drift_direction == 0) { // drift_direction is 0 when noop starts
        env->drift_direction = (env->player_x > (road_left + road_right) / 2) ? -1 : 1;
        // Remove enemy cars in middle lane and lane player is drifting towards
        // only if they are behind the player (y > player_y) to avoid crashes
        for (int i = 0; i < env->numEnemies; i++) {
            if (i >= env->max_enemies) {
    DEBUG_FPRINT(stderr, "Error: Accessing enemyCars[%d] out of bounds\n", i);
    exit(EXIT_FAILURE);
}

Car* car = &env->enemyCars[i];

if (car == NULL) {
    DEBUG_FPRINT(stderr, "Error: car pointer is NULL for enemyCars[%d]\n", i);
    exit(EXIT_FAILURE);
}
            if ((car->lane == 1 || car->lane == env->lane + env->drift_direction) && (car->y > env->player_y)) {
                for (int j = i; j < env->numEnemies - 1; j++) {
                    env->enemyCars[j] = env->enemyCars[j + 1];
                }
                env->numEnemies--;
                i--;
            }
        }
    }
    // Drift distance per step
    if (env->collision_cooldown_car_vs_road > 0) {
        env->player_x += env->drift_direction * 0.12f;
    } else {
    env->player_x += env->drift_direction * 0.25f;
    }
    }

// DEBUG_PRINT("After action movement: player_x = %.2f\n", env->player_x);
    // Road curve/vanishing point movement logic
    // Adjust player's x position based on the current curve
    float curve_shift = -env->current_curve_factor * CURVE_PLAYER_SHIFT_FACTOR * fabs(env->speed);
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

    // After curve shift
// DEBUG_PRINT("Curve shift: %.2f\n", curve_shift);
// DEBUG_PRINT("After curve shift: player_x = %.2f\n", env->player_x);
    
    // Wiggle logic
    if (env->wiggle_active) {
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
    }

    // Player car moves forward slightly according to speed
    // Update player y position based on speed
    env->player_y = PLAYER_MAX_Y - (env->speed - env->min_speed) / (env->max_speed - env->min_speed) * (PLAYER_MAX_Y - PLAYER_MIN_Y);
    // Clamp player_y to measured range
    if (env->player_y < PLAYER_MIN_Y) env->player_y = PLAYER_MIN_Y;
    if (env->player_y > PLAYER_MAX_Y) env->player_y = PLAYER_MAX_Y;

    // Check for and handle collisions between player and road edges
    if (env->player_x <= road_left || env->player_x >= road_right) {
        env->log.collisions_player_vs_road++;
        env->rewards[0] -= 0.0001f;
        env->speed = fmaxf((env->speed - 1.25f), MIN_SPEED);
        env->collision_cooldown_car_vs_road = CRASH_NOOP_DURATION_CAR_VS_ROAD;
        env->drift_direction = 0; // Reset drift direction, has priority over car collisions
        env->player_x = fmaxf(road_left + 1, fminf(road_right - 1, env->player_x));        
    }

    // Enemy car logic
    for (int i = 0; i < env->numEnemies; i++) {
        if (i >= env->max_enemies) {
    DEBUG_FPRINT(stderr, "Error: Accessing enemyCars[%d] out of bounds\n", i);
    exit(EXIT_FAILURE);
}

Car* car = &env->enemyCars[i];

if (car == NULL) {
    DEBUG_FPRINT(stderr, "Error: car pointer is NULL for enemyCars[%d]\n", i);
    exit(EXIT_FAILURE);
}

        // Remove off-screen cars that move below the screen
        if (car->y > PLAYABLE_AREA_BOTTOM + CAR_HEIGHT * 5) {
            // Remove car from array if it moves below the screen
            for (int j = i; j < env->numEnemies - 1; j++) {
                env->enemyCars[j] = env->enemyCars[j + 1];
            }
            env->numEnemies--;
            i--;
            continue;
        }

        // Remove cars that reach or surpass the logical vanishing point if moving up (player speed negative)
        if (env->speed < 0 && car->y <= LOGICAL_VANISHING_Y) {
            // Remove car from array if it reaches the logical vanishing point if moving down (player speed positive)
            for (int j = i; j < env->numEnemies - 1; j++) {
                env->enemyCars[j] = env->enemyCars[j + 1];
            }
            env->numEnemies--;
            i--;
            continue;
        }
        
        // If the car is behind the player and speed ≤ 0, move it to the furthest lane
        if (env->speed <= 0 && car->y >= env->player_y + CAR_HEIGHT) {
            // Determine the furthest lane
            int furthest_lane;
            int player_lane = get_player_lane(env);
            if (player_lane == 0) {
                furthest_lane = 2;
                continue;
            } else if (player_lane == 2) {
                furthest_lane = 0;
                continue;
            } else {
                // Player is in the middle lane
                // Decide based on player's position relative to the road center
                float player_center_x = env->player_x + CAR_WIDTH / 2.0f;
                float road_center_x = (road_edge_x(env, env->player_y, 0, true) +
                                    road_edge_x(env, env->player_y, 0, false)) / 2.0f;
                if (player_center_x < road_center_x) {
                    furthest_lane = 2; // Player is on the left side
                    continue;
                } else {
                    furthest_lane = 0; // Player is on the right side
                    continue;
                }
            }
            car->lane = furthest_lane;
            continue;
        }

        // Check for passing logic **only if not on collision cooldown**
        if (env->speed > 0 && car->last_y < env->player_y + CAR_HEIGHT && car->y >= env->player_y + CAR_HEIGHT && env->collision_cooldown_car_vs_car <= 0 && env->collision_cooldown_car_vs_road <= 0) {
            if (env->carsToPass > 0) {
                env->carsToPass -= 1;
            }
            if (!car->passed) {
            env->log.passed_cars += 1;
            // env->score += 1;
            env->rewards[0] += 1.0f; // Car passed reward
            }
            car->passed = true;

        }

        // Check if an enemy car passes the player
        if (env->speed < 0 && car->last_y > env->player_y  + CAR_HEIGHT && car->y <= env->player_y + CAR_HEIGHT ) {
            int maxCarsToPass = (env->day == 1) ? 200 : 300; // Set max based on day
            if (env->carsToPass == maxCarsToPass) {
                // Do nothing, but log the event
                env->log.passed_by_enemy += 1.0f;
            } else {
                env->carsToPass += 1;
                env->log.passed_by_enemy += 1.0f;
                env->rewards[0] -= 0.01f;
                // env->score -= 5;
                // env->rewards[0] -= 5;
            }
            
            // Passing debug
            // DEBUG_PRINT("Car passed player at y = %.2f. Remaining cars to pass: %d\n", car->y, env->carsToPass);
        }

        car->last_y = car->y;
        // if (env->carsToPass < env->score) {
        //     env->rewards[0] += 0.1f / env->carsToPass;
        // }

        // Ignore collisions for 1 second to avoid edge-case chain collisions
        if (env->collision_cooldown_car_vs_car > 0) {
            if (env->collision_invulnerability_timer <= 0) {
                env->collision_invulnerability_timer = TARGET_FPS * 1.0f;
            }
        } else if (env->collision_invulnerability_timer > 0) {
            env->collision_invulnerability_timer -= 1;
        }

        // Check for and handle collisions between player and enemy cars
        if (env->collision_cooldown_car_vs_car <= 0 && env->collision_invulnerability_timer <= 0) {
            if (check_collision(env, car)) {
                env->log.collisions_player_vs_car++;
                // env->rewards[0] -= 50.0f;
                env->speed = 1 + MIN_SPEED;
                env->collision_cooldown_car_vs_car = CRASH_NOOP_DURATION_CAR_VS_CAR;
                env->drift_direction = 0; // Reset drift direction
            }
        }
    }

    // Calculate enemy spawn interval based on player speed and day
    // Values measured from Enduro Atari 2600 gameplay
    float min_spawn_interval = 0.8777f; // Minimum spawn interval
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

    // Calculate speed factor for enemy spawning
    float speed_factor = (env->speed - env->min_speed) / (env->max_speed - env->min_speed);
    if (speed_factor < 0.0f) speed_factor = 0.0f;
    if (speed_factor > 1.0f) speed_factor = 1.0f;

    // Interpolate between min and max spawn intervals to scale to player speed
    env->enemySpawnInterval = min_spawn_interval - speed_factor * (min_spawn_interval - max_spawn_interval);

    // Update enemy spawn timer
    env->enemySpawnTimer += (1.0f / TARGET_FPS);
    if (env->enemySpawnTimer >= env->enemySpawnInterval) {
        env->enemySpawnTimer -= env->enemySpawnInterval;
        if (env->numEnemies < MAX_ENEMIES) {
            add_enemy_car(env);
            // Enemy spawn debugging print
            // DEBUG_PRINT("Enemy car spawned at time %.2f seconds\n", env->elapsedTime);
        }
    }

    // Day completed logic
    if (env->carsToPass <= 0 && !env->dayCompleted) {
        env->dayCompleted = true;
    }

    // env->score = env->carsToPass;

    // Handle day transition when background cycles back to 0
    if (env->currentDayTimeIndex == 0 && env->previousDayTimeIndex == 15) {
        // Background cycled back to 0
        if (env->dayCompleted) {
            env->log.days_completed += 1;
            env->day += 1;
            env->rewards[0] += 10.0f;
            env->carsToPass = 300; // Always 300 after the first day
            env->dayCompleted = false;
            add_log(env->log_buffer, &env->log);
            compute_observations(env); // Call compute_observations to log
        
        } else {
            // Player failed to pass required cars, reset environment
            env->log.days_failed += 1.0f;
            env->day = 1;
            // env->rewards[0] -= 10.0f;
            env->terminals[0] = false;
            add_log(env->log_buffer, &env->log);
            compute_observations(env); // Call compute_observations before reset to log
            reset(env);
            return;
        }
    }

    // Reward player for moving forward
    // compute speed-based reward for speed > MIN_SPEED
    // float speed_reward = fabs((env->speed - MIN_SPEED));
    // env->rewards[0] += speed_reward;

    env->log.reward = env->rewards[0];
    env->log.episode_return = env->rewards[0];
    env->step_count++;
    env->score += env->rewards[0];
    env->log.score = env->score;
    int local_cars_to_pass = env->carsToPass;
    env->log.cars_to_pass = (int)local_cars_to_pass;

    // // Debugging print to check all variables before computing observations
    // DEBUG_PRINT("line 1232: calling debug_enduro_allocation at bottom of step() before compute_observations\n");
    // debug_enduro_allocation(env);
    compute_observations(env);
    // add_log(env->log_buffer, &env->log);

    // // Early stopping if not passing cars (stop on snow stage)
    // if (isSnowStage && env->rewards[0] < 1) {
    //     env->terminals[0] = 1;
    //     reset(env);
    // }

    // Early stopping if episode_length > 6000 and no cars have been passed
    if (env->log.episode_length > 6000 && env->log.passed_cars == 0) {
        env->terminals[0] = 1;
        reset(env);
    }

    // printf("stay_on_road_reward: %f, passed_by_enemy: %f, cars_to_pass: %d\n",
    //         env->log.stay_on_road_reward,
    //         env->log.passed_by_enemy,
    //         env->log.cars_to_pass);

}

void debug_enduro_allocation(Enduro* env) {
    DEBUG_PRINT("Memory layout debugging:\n");
    DEBUG_PRINT("Enduro struct size: %zu bytes\n", sizeof(Enduro));
    DEBUG_PRINT("Observation array size: %zu\n", env->obs_size);
    DEBUG_PRINT("Max enemies: %d\n", env->max_enemies);
    DEBUG_PRINT("Enemy cars array size: %zu bytes\n", sizeof(Car) * env->max_enemies);
    DEBUG_PRINT("Day transition times array size: %zu bytes\n", sizeof(float) * 16);
    
    // Validate critical pointers
    DEBUG_PRINT("\nPointer validation:\n");
    DEBUG_PRINT("observations ptr: %p\n", (void*)env->observations);
    DEBUG_PRINT("enemyCars ptr: %p\n", (void*)env->enemyCars);
    DEBUG_PRINT("dayTransitionTimes ptr: %p\n", (void*)env->dayTransitionTimes);
    
    // Verify array bounds
    DEBUG_PRINT("\nArray bounds check:\n");
    DEBUG_PRINT("Last observation index accessible: %zu\n", env->obs_size - 1);
    DEBUG_PRINT("Last enemy car index accessible: %d\n", env->max_enemies - 1);
    DEBUG_PRINT("Last day transition time index accessible: 15\n");

    // Validate observation array
    DEBUG_PRINT("\nObservation array validation:\n");
    for (int i = 0; i < (int)env->obs_size; i++) {
        if (i >= 0 && i < (int)env->obs_size) {
            DEBUG_PRINT("observations[%d]: %.2f\n", i, env->observations[i]);
        } else {
            DEBUG_PRINT("observations[%d]: Out of bounds\n", i);
        }
    }
    
    // Validate enemy cars array
    DEBUG_PRINT("\nEnemy cars array validation:\n");
    for (int i = 0; i < env->max_enemies && i < MAX_ENEMIES; i++) {
        if (i >= 0 && i < env->max_enemies && i < MAX_ENEMIES) {
            DEBUG_PRINT("enemyCars[%d]: y = %.2f, lane = %d, colorIndex = %d\n", i, car->y, car->lane, car->colorIndex);
        } else {
            DEBUG_PRINT("enemyCars[%d]: Out of bounds\n", i);
        }
    }
    // Validate day transition times array
    DEBUG_PRINT("\nDay transition times array validation:\n");
    for (int i = 0; i < 16; i++) {
        if (i >= 0 && i < 16) {
            DEBUG_PRINT("dayTransitionTimes[%d]: %.2f\n", i, env->dayTransitionTimes[i]);
        } else {
            DEBUG_PRINT("dayTransitionTimes[%d]: Out of bounds\n", i);
        }
    }
    // Validate log buffer
    DEBUG_PRINT("\nLog buffer validation:\n");
    DEBUG_PRINT("Log buffer size: %zu bytes\n", sizeof(env->log_buffer));
    DEBUG_PRINT("Log buffer length: %d\n", env->log_buffer->length);
DEBUG_PRINT("Current log index: %d\n", env->log_buffer->idx);

    // Validate log struct
    DEBUG_PRINT("\nLog struct validation:\n");
    DEBUG_PRINT("Log struct size: %zu bytes\n", sizeof(env->log));
    DEBUG_PRINT("Log struct contents: episode_length = %f, reward = %.2f, score = %f\n", env->log.episode_length, env->log.reward, env->log.score);
    // Validate terminals array
    DEBUG_PRINT("\nTerminals array validation:\n");
    DEBUG_PRINT("Terminals array size: %zu bytes\n", sizeof(env->terminals));
    DEBUG_PRINT("Terminals array contents: %d\n", env->terminals[0]);
    // Validate rewards array
    DEBUG_PRINT("\nRewards array validation:\n");
    DEBUG_PRINT("Rewards array size: %zu bytes\n", sizeof(env->rewards));
    DEBUG_PRINT("Rewards array contents: %.2f\n", env->rewards[0]);
    // Validate actions array
    DEBUG_PRINT("\nActions array validation:\n");
    DEBUG_PRINT("Actions array size: %zu bytes\n", sizeof(env->actions));
    DEBUG_PRINT("Actions array contents: %d\n", env->actions[0]);
    // Validate speed
    DEBUG_PRINT("\nSpeed validation:\n");
    DEBUG_PRINT("Speed: %.2f\n", env->speed);
    // Validate player position
    DEBUG_PRINT("\nPlayer position validation:\n");
    DEBUG_PRINT("Player position: x = %.2f, y = %.2f\n", env->player_x, env->player_y);
    // Validate road edges
    DEBUG_PRINT("\nRoad edges validation:\n");
    DEBUG_PRINT("Road edges: left = %.2f, right = %.2f\n", env->last_road_left, env->last_road_right);
    // Validate vanishing point
    DEBUG_PRINT("\nVanishing point validation:\n");
    DEBUG_PRINT("Vanishing point: x = %.2f\n", env->vanishing_point_x);
    // Validate curve factor
    DEBUG_PRINT("\nCurve factor validation:\n");
    DEBUG_PRINT("Curve factor: %.2f\n", env->current_curve_factor);
    // Validate wiggle
    DEBUG_PRINT("\nWiggle validation:\n");
    DEBUG_PRINT("Wiggle active: %d\n", env->wiggle_active);
    DEBUG_PRINT("Wiggle speed: %.2f\n", env->wiggle_speed);
    DEBUG_PRINT("Wiggle y: %.2f\n", env->wiggle_y);
    // Validate day
    DEBUG_PRINT("\nDay validation:\n");
    DEBUG_PRINT("Day: %d\n", env->day);
    // Validate cars to pass
    DEBUG_PRINT("\nCars to pass validation:\n");
    DEBUG_PRINT("Cars to pass: %d\n", env->carsToPass);
    // Validate log struct
    DEBUG_PRINT("\nLog struct validation:\n");
    DEBUG_PRINT("Log struct size: %zu bytes\n", sizeof(env->log));
    DEBUG_PRINT("Log struct contents: episode_length = %f, reward = %.2f, score = %f\n", env->log.episode_length, env->log.reward, env->log.score);
    // Validate step count
    DEBUG_PRINT("\nStep count validation:\n");
    DEBUG_PRINT("Step count: %d\n", env->step_count);
    // Validate elapsed time
    DEBUG_PRINT("\nElapsed time validation:\n");
    DEBUG_PRINT("Elapsed time: %.2f\n", env->elapsedTimeEnv);
    // Validate enemy spawn interval
    DEBUG_PRINT("\nEnemy spawn interval validation:\n");
    DEBUG_PRINT("Enemy spawn interval: %.2f\n", env->enemySpawnInterval);
    // Validate enemy spawn timer
    DEBUG_PRINT("\nEnemy spawn timer validation:\n");
    DEBUG_PRINT("Enemy spawn timer: %.2f\n", env->enemySpawnTimer);
    // Validate collision cooldowns
    DEBUG_PRINT("\nCollision cooldowns validation:\n");
    DEBUG_PRINT("Car vs. car cooldown: %f\n", env->collision_cooldown_car_vs_car);
    DEBUG_PRINT("Car vs. road cooldown: %f\n", env->collision_cooldown_car_vs_road);
    // Validate drift direction
    DEBUG_PRINT("\nDrift direction validation:\n");
    DEBUG_PRINT("Drift direction: %d\n", env->drift_direction);
    // Validate collision invulnerability timer
    DEBUG_PRINT("\nCollision invulnerability timer validation:\n");
    DEBUG_PRINT("Collision invulnerability timer: %.2f\n", env->collision_invulnerability_timer);

}

void compute_observations(Enduro* env) {

    // // Victory or terminal condition logging
    // // Debugging only
    // if (env->carsToPass > 0) {
    //     env->carsToPass -= 1;
    // }

// Debug at top of compute_observations
DEBUG_PRINT("line 1369: calling debug_enduro_allocation at start of compute_observations\n");
debug_enduro_allocation(env);


    float* obs = env->observations;
    int obs_index = 0;

    // Validate obs_size
    const int expected_size = OBSERVATIONS_MAX_SIZE;
    if ((int)env->obs_size != expected_size) {  // Cast to match types
        DEBUG_FPRINT(stderr, "Error: obs_size mismatch! Expected %d, got %zu\n", expected_size, env->obs_size);
        exit(EXIT_FAILURE);
    }

    // Define bounds check
    #define CHECK_BOUNDS(index)                                          \
        if ((index) >= (int)env->obs_size) {                       \
            DEBUG_FPRINT(stderr, "Error: obs_index %d out of bounds on line %d!\n", index, __LINE__); \
            exit(EXIT_FAILURE);                                          \
        }

DEBUG_PRINT("Debug: enemyCars pointer before indexing to array in compute_observations = %p\n", (void*)env->enemyCars);
DEBUG_PRINT("Debug: max_enemies before indexing to array in compute_observations = %d\n", env->max_enemies);

DEBUG_PRINT("initial obs_index: %d\n", obs_index);

    // Player position and speed
    // idx 1
    obs[obs_index++] = (env->player_x - PLAYER_MIN_X) / (PLAYER_MAX_X - PLAYER_MIN_X);
    CHECK_BOUNDS(obs_index);
    DEBUG_PRINT("obs_index after player_x: %d\n", obs_index);
    // idx 2
    obs[obs_index++] = (env->player_y - PLAYER_MIN_Y) / (PLAYER_MAX_Y - PLAYER_MIN_Y);
    CHECK_BOUNDS(obs_index);
    DEBUG_PRINT("obs_index after player_y: %d\n", obs_index);
    // idx 3
    obs[obs_index++] = (env->speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED);
    CHECK_BOUNDS(obs_index);
    DEBUG_PRINT("obs_index after speed: %d\n", obs_index);

    // Road edges
    // idx 4
    obs[obs_index++] = (road_edge_x(env, env->player_y, 0, true) - PLAYABLE_AREA_LEFT) /
                       (PLAYABLE_AREA_RIGHT - PLAYABLE_AREA_LEFT);
    CHECK_BOUNDS(obs_index);
    DEBUG_PRINT("obs_index after road_edge_x: %d\n", obs_index);
    // idx 5
    obs[obs_index++] = (road_edge_x(env, env->player_y, 0, false) - PLAYABLE_AREA_LEFT) /
                       (PLAYABLE_AREA_RIGHT - PLAYABLE_AREA_LEFT);
    CHECK_BOUNDS(obs_index);
    DEBUG_PRINT("obs_index after road_edge_x: %d\n", obs_index);

    // Player lane
    // idx 6
    obs[obs_index++] = (float)get_player_lane(env) / (NUM_LANES - 1);
    CHECK_BOUNDS(obs_index);
    DEBUG_PRINT("obs_index after player_lane: %d\n", obs_index);

    // Enemy cars
    // idx 7 - idx 26 (2 * max_enemies)
    for (int i = 0; i < env->max_enemies; i++) {
        if (i >= env->max_enemies) {
    DEBUG_FPRINT(stderr, "Error: Accessing enemyCars[%d] out of bounds\n", i);
    exit(EXIT_FAILURE);
}

Car* car = &env->enemyCars[i];

if (car == NULL) {
    DEBUG_FPRINT(stderr, "Error: car pointer is NULL for enemyCars[%d]\n", i);
    exit(EXIT_FAILURE);
}
        if (car->y > 0 && car->y < env->height) {
            float car_x = car_x_in_lane(env, car->lane, car->y);
            obs[obs_index++] = (car_x - env->player_x + PLAYABLE_AREA_RIGHT - PLAYABLE_AREA_LEFT) /
                               (2 * (PLAYABLE_AREA_RIGHT - PLAYABLE_AREA_LEFT));
            CHECK_BOUNDS(obs_index);
            DEBUG_PRINT("obs_index after car_x: %d\n", obs_index);
            obs[obs_index++] = (car->y - env->player_y + env->height) / (2 * env->height);
            CHECK_BOUNDS(obs_index);
            DEBUG_PRINT("obs_index after car_y: %d\n", obs_index);
        } else {
            obs[obs_index++] = 0.5f;
            CHECK_BOUNDS(obs_index);
            DEBUG_PRINT("obs_index after car_x: %d\n", obs_index);
            obs[obs_index++] = 0.5f;
            CHECK_BOUNDS(obs_index);
            DEBUG_PRINT("obs_index after car_y: %d\n", obs_index);
        }
    }

DEBUG_PRINT("Debug: enemyCars pointer after indexing to array in compute_observations = %p\n", (void*)env->enemyCars);
DEBUG_PRINT("Debug: max_enemies after indexing to array in compute_observations = %d\n", env->max_enemies);
DEBUG_PRINT("obs_index after enemyCars: %d\n", obs_index);

    // Curve direction
    // idx 27
    obs[obs_index++] = (float)(env->current_curve_direction + 1) / 2.0f;
    CHECK_BOUNDS(obs_index);
    DEBUG_PRINT("obs_index after curve_direction: %d\n", obs_index);

    // Time of day
    // idx 28
    float total_day_length = env->dayTransitionTimes[15];
    obs[obs_index++] = fmodf(env->elapsedTimeEnv, total_day_length) / total_day_length;
    CHECK_BOUNDS(obs_index);
    DEBUG_PRINT("obs_index after time_of_day: %d\n", obs_index);

    // Cars to pass
    // idx 29
    DEBUG_PRINT("Debug: carsToPass = %d\n", env->carsToPass);
    DEBUG_PRINT("Debug: initial_cars_to_pass = %d\n", env->initial_cars_to_pass);
    DEBUG_PRINT("Debug: carsToPass / initial_cars_to_pass = %f\n", (float)env->carsToPass / env->initial_cars_to_pass);
    obs[obs_index++] = (float)env->carsToPass / env->initial_cars_to_pass;
    CHECK_BOUNDS(obs_index);
    DEBUG_PRINT("obs_index after cars_to_pass (should be 29): %d\n", obs_index);

    // Verify obs_index
    // if (obs_index != (int)env->obs_size) {
    //     DEBUG_FPRINT(stderr, "Error: Final obs_index %d does not match obs_size %zu\n", obs_index, env->obs_size);
    //     exit(EXIT_FAILURE);
    // }

if (obs_index >= (int)env->obs_size) {
    DEBUG_FPRINT(stderr, "Error: obs_index %d exceeds observation array size %zu\n", obs_index, env->obs_size);
    exit(EXIT_FAILURE);
}

// Debug at end of compute_observations
DEBUG_PRINT("line 1479: calling debug_enduro_allocation at end of compute_observations\n");
debug_enduro_allocation(env);

}

// When to curve road and how to curve it, including dense smooth transitions
// An ugly, dense function, but it is necessary
void update_road_curve(Enduro* env) {
    static int current_curve_stage = 0;
    static int steps_in_current_stage = 0;
    
    // Map speed to the scale between 0.5 and 3.5
    float speed_scale = 0.5f + ((fabs(env->speed) / env->max_speed) * (MAX_SPEED - MIN_SPEED));
    float vanishing_point_transition_speed = VANISHING_POINT_TRANSITION_SPEED + speed_scale; 

    // Steps to curve L, R, go straight for
    int step_thresholds[] = {350};
    int curve_directions[] = {0, -1, 0, 1};

    // Determine sizes of step_thresholds and curve_directions
    size_t step_thresholds_size = sizeof(step_thresholds) / sizeof(step_thresholds[0]);
    size_t curve_directions_size = sizeof(curve_directions) / sizeof(curve_directions[0]);

    // Find the maximum size
    size_t max_size = (step_thresholds_size > curve_directions_size) ? step_thresholds_size : curve_directions_size;

    // Adjust arrays dynamically
    int adjusted_step_thresholds[max_size];
    int adjusted_curve_directions[max_size];

    for (size_t i = 0; i < max_size; i++) {
        adjusted_step_thresholds[i] = step_thresholds[i % step_thresholds_size];
        adjusted_curve_directions[i] = curve_directions[i % curve_directions_size];
    }

    // Use adjusted arrays for current calculations
    env->current_step_threshold = adjusted_step_thresholds[current_curve_stage % max_size];
    steps_in_current_stage++;

    if (steps_in_current_stage >= adjusted_step_thresholds[current_curve_stage]) {
        env->target_curve_factor = (float)adjusted_curve_directions[current_curve_stage % max_size];
        steps_in_current_stage = 0;
        current_curve_stage = (current_curve_stage + 1) % max_size;
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
float road_edge_x(Enduro* env, float y, float offset, unsigned char left) {
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

// Computes x position of car in a given lane
float car_x_in_lane(Enduro* env, int lane, float y) {
    // Set offset to 0 to ensure enemy cars align with the road rendering
    float offset = 0.0f;
    float left_edge = road_edge_x(env, y, offset, true);
    float right_edge = road_edge_x(env, y, offset, false);
    float lane_width = (right_edge - left_edge) / NUM_LANES;
    return left_edge + lane_width * (lane + 0.5f);
}

// Make the make_client function handle all rendering code. 
// GameState struct should be moved to the client struct
// All rendering logic should be completely separate from env logic
Client* make_client(Enduro* env) {
    // Allocate memory for Client
    Client* client = (Client*)malloc(sizeof(Client));
    if (!client) {
        fprintf(stderr, "Failed to allocate memory for Client\n");
        return NULL;
    }

    // Initialize Client using information from Enduro
    client->width = env->width;  // Example: use env dimensions for rendering
    client->height = env->height;
    
    // Perform additional initialization if necessary
    initRaylib();  // Assume Raylib is being used for rendering setup

    return client;
}



void close_client(Client* client, Enduro* env) {
    if (client != NULL) {
        cleanup(&client->gameState); // Free gameState resources
        CloseWindow(); // Close Raylib window
        free(client);
        client = NULL; // Avoid use-after-free
    }
}


// void render_car(Client* client, Enduro* env) {
//     GameState* gameState = &client->gameState;
//     Texture2D carTexture;
//     if (gameState->showLeftTread) {
//         carTexture = gameState->playerCarLeftTreadTexture;
//     } else {
//         carTexture = gameState->playerCarRightTreadTexture;
//     }
//     DrawTexture(carTexture, (int)env->player_x, (int)env->player_y, WHITE);
// }

void render_car(Client* client, GameState* gameState) {
    Texture2D carTexture = gameState->showLeftTread ? gameState->playerCarLeftTreadTexture : gameState->playerCarRightTreadTexture;
    DrawTexture(carTexture, (int)gameState->player_x, (int)gameState->player_y, WHITE);
}

void handleEvents(int* running, Enduro* env) {
    *env->actions = ACTION_NOOP;
    if (WindowShouldClose()) {
        *running = 0;
    }
    unsigned char left = IsKeyDown(KEY_LEFT);
    unsigned char right = IsKeyDown(KEY_RIGHT);
    unsigned char down = IsKeyDown(KEY_DOWN);
    unsigned char fire = IsKeyDown(KEY_SPACE); // Fire key
    if (fire) {
        if (right) {
            *env->actions = ACTION_RIGHTFIRE;
        } else if (left) {
            *env->actions = ACTION_LEFTFIRE;
        } else {
            *env->actions = ACTION_FIRE;
        }
    } else if (down) {
        if (right) {
            *env->actions = ACTION_DOWNRIGHT;
        } else if (left) {
            *env->actions = ACTION_DOWNLEFT;
        } else {
            *env->actions = ACTION_DOWN;
        }
    } else if (right) {
        *env->actions = ACTION_RIGHT;
    } else if (left) {
        *env->actions = ACTION_LEFT;
    } else {
        *env->actions = ACTION_NOOP;
    }
    // DEBUG_PRINT("Action set to: %d\n", *env->actions);

}

void initRaylib() {
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Enduro Port Framework");
    SetTargetFPS(60);
}

// void performance_test() {
//     // Setup test environment
//     Enduro test_env = {
//         .width = SCREEN_WIDTH,
//         .height = SCREEN_HEIGHT,
//         .car_width = CAR_WIDTH,
//         .car_height = CAR_HEIGHT,
//         .max_enemies = MAX_ENEMIES,
//         .initial_cars_to_pass = INITIAL_CARS_TO_PASS,
//         .min_speed = MIN_SPEED,
//         .max_speed = MAX_SPEED,
//     };
    
//     allocate(&test_env);
//     reset(&test_env);

//     clock_t start = clock();
    
//     // Run core game operations
//     for (int i = 0; i < 1000000; i++) {
//         // Simulate typical game step
//         test_env.speed += 0.1f;
//         if (test_env.speed > test_env.max_speed) 
//             test_env.speed = test_env.min_speed;
            
//         // Update road curve
//         update_road_curve(&test_env);
        
//         // Update car positions
//         for (int j = 0; j < test_env.numEnemies; j++) {
//             test_env.enemyCars[j].y += test_env.speed;
//             if (test_env.enemyCars[j].y > test_env.height) {
//                 test_env.enemyCars[j].y = 0;
//             }
//         }
        
//         // Check collisions
//         for (int j = 0; j < test_env.numEnemies; j++) {
//             check_collision(&test_env, &test_env.enemyCars[j]);
//         }
        
//         // Compute observations
//         compute_observations(&test_env);
//     }
    
//     clock_t end = clock();
//     double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
//     DEBUG_PRINT("Performance test completed in %f seconds\n", time_spent);
//     DEBUG_PRINT("Average FPS: %f\n", 1000000.0 / (time_spent * 60));
    
//     // Cleanup
//     free_allocated(&test_env);
// }


void loadTextures(GameState* gameState) {
    // Initialize animation variables
    gameState->carAnimationTimer = 0.0f;
    gameState->carAnimationInterval = 0.05f; // Initial interval, will be updated based on speed
    gameState->showLeftTread = true;
    gameState->mountainPosition = 0.0f;

    // Initialize victory effect variables
    gameState->showLeftFlag = true;
    gameState->flagTimer = 0;
    gameState->victoryDisplayTimer = 0;
    gameState->victoryAchieved = false;

    // Initialize scoreboard variables
    gameState->score = 0;
    gameState->scoreTimer = 0;
    gameState->carsLeftGameState = 0;
    gameState->day = 1;

    // Initialize score digits arrays
    for (int i = 0; i < SCORE_DIGITS; i++) {
        gameState->scoreDigitCurrents[i] = 0;
        gameState->scoreDigitNexts[i] = 0;
        gameState->scoreDigitOffsets[i] = 0.0f;
        gameState->scoreDigitScrolling[i] = false;
    }

    // Initialize other necessary variables
    gameState->elapsedTime = 0.0f;
    gameState->currentBackgroundIndex = 0;
    gameState->backgroundIndex = 0;
    gameState->previousBackgroundIndex = 0;

    // Load background and mountain textures for different times of day per og enduro
    char backgroundFile[40];
    char mountainFile[40];
    for (int i = 0; i < 16; ++i) {
        snprintf(backgroundFile, sizeof(backgroundFile), "resources/puffer_enduro/%d_bg.png", i);
        gameState->backgroundTextures[i] = LoadTexture(backgroundFile);
        snprintf(mountainFile, sizeof(mountainFile), "resources/puffer_enduro/%d_mtns.png", i);
        gameState->mountainTextures[i] = LoadTexture(mountainFile);
    }
    // Load digit textures 0-9
    char filename[100];
    for (int i = 0; i < 10; i++) {
        snprintf(filename, sizeof(filename), "resources/puffer_enduro/digits_%d.png", i);
        gameState->digitTextures[i] = LoadTexture(filename);
    }
    // Load the "car" digit texture
    gameState->carDigitTexture = LoadTexture("resources/puffer_enduro/digits_car.png");
    DEBUG_PRINT("Loaded digit image: digits_car.png\n");
    // Load level complete flag textures
    gameState->levelCompleteFlagLeftTexture = LoadTexture("resources/puffer_enduro/level_complete_flag_left.png");
    DEBUG_PRINT("Loaded image: level_complete_flag_left.png\n");
    gameState->levelCompleteFlagRightTexture = LoadTexture("resources/puffer_enduro/level_complete_flag_right.png");
    DEBUG_PRINT("Loaded image: level_complete_flag_right.png\n");
    // Load green digits for completed days
    for (int i = 0; i < 10; ++i) {
        snprintf(filename, sizeof(filename), "resources/puffer_enduro/green_digits_%d.png", i);
        gameState->greenDigitTextures[i] = LoadTexture(filename);
    }
    // Load yellow digits for scoreboard numbers
    for (int i = 0; i < 10; ++i) {
        snprintf(filename, sizeof(filename), "resources/puffer_enduro/yellow_digits_%d.png", i);
        gameState->yellowDigitTextures[i] = LoadTexture(filename);
    }
    gameState->playerCarLeftTreadTexture = LoadTexture("resources/puffer_enduro/player_car_left_tread.png");
    gameState->playerCarRightTreadTexture = LoadTexture("resources/puffer_enduro/player_car_right_tread.png");

// // TESTING ONLY
// env->backgroundTransitionTimes[0] = 2.0f;   // Transition to background 1 at 2 seconds
// env->backgroundTransitionTimes[1] = 3.0f;   // Transition to background 2 at 4 seconds
// env->backgroundTransitionTimes[2] = 4.0f;   // Transition to background 3 at 6 seconds
// env->backgroundTransitionTimes[3] = 15.0f;   // Transition to background 4 at 8 seconds
// env->backgroundTransitionTimes[4] = 16.0f;  // Transition to background 5 at 10 seconds
// env->backgroundTransitionTimes[5] = 17.0f;  // Transition to background 6 at 12 seconds
// env->backgroundTransitionTimes[6] = 18.0f;  // Transition to background 7 at 14 seconds
// env->backgroundTransitionTimes[7] = 19.0f;  // Transition to background 8 at 16 seconds
// env->backgroundTransitionTimes[8] = 20.0f;  // Transition to background 9 at 18 seconds
// env->backgroundTransitionTimes[9] = 21.0f;  // Transition to background 10 at 20 seconds
// env->backgroundTransitionTimes[10] = 22.0f; // Transition to background 11 at 22 seconds
// env->backgroundTransitionTimes[11] = 23.0f; // Transition to background 12 at 24 seconds
// env->backgroundTransitionTimes[12] = 24.0f; // Transition to background 13 at 36 seconds (12-second duration)
// env->backgroundTransitionTimes[13] = 25.0f; // Transition to background 14 at 48 seconds (12-second duration)
// env->backgroundTransitionTimes[14] = 26.0f; // Transition to background 15 at 60 seconds (12-second duration)
// env->backgroundTransitionTimes[15] = 27.0f; // Transition to background 0 at 62 seconds (loop back)

    // Load enemy car textures for each color and tread variant
    gameState->enemyCarTextures[0][0] = LoadTexture("resources/puffer_enduro/enemy_car_blue_left_tread.png");
    gameState->enemyCarTextures[0][1] = LoadTexture("resources/puffer_enduro/enemy_car_blue_right_tread.png");
    gameState->enemyCarTextures[1][0] = LoadTexture("resources/puffer_enduro/enemy_car_gold_left_tread.png");
    gameState->enemyCarTextures[1][1] = LoadTexture("resources/puffer_enduro/enemy_car_gold_right_tread.png");
    gameState->enemyCarTextures[2][0] = LoadTexture("resources/puffer_enduro/enemy_car_pink_left_tread.png");
    gameState->enemyCarTextures[2][1] = LoadTexture("resources/puffer_enduro/enemy_car_pink_right_tread.png");
    gameState->enemyCarTextures[3][0] = LoadTexture("resources/puffer_enduro/enemy_car_salmon_left_tread.png");
    gameState->enemyCarTextures[3][1] = LoadTexture("resources/puffer_enduro/enemy_car_salmon_right_tread.png");
    gameState->enemyCarTextures[4][0] = LoadTexture("resources/puffer_enduro/enemy_car_teal_left_tread.png");
    gameState->enemyCarTextures[4][1] = LoadTexture("resources/puffer_enduro/enemy_car_teal_right_tread.png");
    gameState->enemyCarTextures[5][0] = LoadTexture("resources/puffer_enduro/enemy_car_yellow_left_tread.png");
    gameState->enemyCarTextures[5][1] = LoadTexture("resources/puffer_enduro/enemy_car_yellow_right_tread.png");


    // Load enemy car night tail lights textures
    gameState->enemyCarNightTailLightsTexture = LoadTexture("resources/puffer_enduro/enemy_car_night_tail_lights.png");

    // Load enemy car night fog tail lights texture
    gameState->enemyCarNightFogTailLightsTexture = LoadTexture("resources/puffer_enduro/enemy_car_night_fog_tail_lights.png");

    // Initialize animation variables
    gameState->carAnimationTimer = 0.0f;
    gameState->carAnimationInterval = 0.05f; // Initial interval, will be updated based on speed
    gameState->showLeftTread = true;
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
    // Unload "car" digit and flag textures
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

void updateCarAnimation(GameState* gameState) {
    // Update the animation interval based on the player's speed
    // Faster speed means faster alternation
    float minInterval = 0.005f;  // Minimum interval at max speed
    float maxInterval = 0.075f;  // Maximum interval at min speed

    float speedRatio = (gameState->speed - gameState->min_speed) / (gameState->max_speed - gameState->min_speed);
    gameState->carAnimationInterval = maxInterval - (maxInterval - minInterval) * speedRatio;

    // Update the animation timer
    gameState->carAnimationTimer += GetFrameTime(); // Time since last frame

    if (gameState->carAnimationTimer >= gameState->carAnimationInterval) {
        gameState->carAnimationTimer = 0.0f;
        gameState->showLeftTread = !gameState->showLeftTread; // Switch texture
    }
}

void updateScoreboard(GameState* gameState) {
    // Increase the score every 30 frames (~0.5 seconds at 60 FPS)
    gameState->scoreTimer++;
    if (gameState->scoreTimer >= 30) {
        gameState->scoreTimer = 0;
        // env->score += 1;
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

void updateBackground(GameState* gameState) {
    float elapsedTime = gameState->elapsedTime;
    float totalDuration = gameState->backgroundTransitionTimes[15];

    if (elapsedTime >= totalDuration) {
        elapsedTime -= totalDuration;
        gameState->elapsedTime = elapsedTime; // Reset elapsed time in env
        gameState->backgroundIndex = 0;
    }

    gameState->previousBackgroundIndex = gameState->currentBackgroundIndex;

    while (gameState->backgroundIndex < 15 &&
           elapsedTime >= gameState->backgroundTransitionTimes[gameState->backgroundIndex]) {
        gameState->backgroundIndex++;
    }
    gameState->currentBackgroundIndex = gameState->backgroundIndex % 16;
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
    // Pass dayCompleted condition from Enduro to GameState
    if (gameState->dayCompleted) {
        gameState->victoryAchieved = true;
    }
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
        // DEBUG_PRINT("flag direction: %d\n", gameState->showLeftFlag);
        // Alternate between level_complete_flag_left and level_complete_flag_right
        Texture2D flagTexture = gameState->showLeftFlag ? gameState->levelCompleteFlagLeftTexture : gameState->levelCompleteFlagRightTexture;
        Rectangle destRect = { carsX, carsY, digitWidth * 4, digitHeight };
        DrawTextureEx(flagTexture, (Vector2){ destRect.x, destRect.y }, 0.0f, 1.0f, WHITE);
    } else {
        // Render "CAR" label
        DrawTexture(gameState->carDigitTexture, carsX, carsY, WHITE);
        // Render the remaining digits for cars to pass
        int cars = gameState->carsLeftGameState;
        if (cars < 0) cars = 0; // Ensure cars is not negative
        for (int i = 1; i < CARS_DIGITS; ++i) {
            int divisor = (int)pow(10, CARS_DIGITS - i - 1);
            int digit = (cars / divisor) % 10;
            if (digit < 0 || digit > 9) digit = 0; // Clamp digit to valid range
            int digitX = carsX + i * (digitWidth + 1); // Add spacing between digits
            DrawTexture(gameState->digitTextures[digit], digitX, carsY, WHITE);
        }
    }
}

// Triggers the day completed 'victory' display
// Solely for flapping flag visual effect, no game logic
void updateVictoryEffects(GameState* gameState) {
    if (gameState->victoryAchieved) {
        // Dancing flags effect
        gameState->flagTimer++;
        // DEBUG_PRINT("flag timer: %d\n", gameState->flagTimer);
        // Modulo triggers flag direction change
        // Flag renders in that direction until next change
        if (gameState->flagTimer % 50 == 0) {
            gameState->showLeftFlag = !gameState->showLeftFlag;
            // DEBUG_PRINT("flag should have switched directions. direciton: %d\n", gameState->showLeftFlag);
        }
        gameState->victoryDisplayTimer++;
        if (gameState->victoryDisplayTimer >= 10) { // 540
            gameState->victoryDisplayTimer = 0;
        }
    }
}

void updateMountains(GameState* gameState) {
    // Mountain scrolling effect when road is curving
    float baseSpeed = 0.0f;
    float curveStrength = fabsf(gameState->current_curve_factor);
    float speedMultiplier = 1.0f; // Scroll speed
    float scrollSpeed = baseSpeed + curveStrength * speedMultiplier;
    int mountainWidth = gameState->mountainTextures[0].width;
    if (gameState->current_curve_direction == 1) { // Turning left
        gameState->mountainPosition += scrollSpeed;
        if (gameState->mountainPosition >= mountainWidth) {
            gameState->mountainPosition -= mountainWidth;
        }
    } else if (gameState->current_curve_direction == -1) { // Turning right
        gameState->mountainPosition -= scrollSpeed;
        if (gameState->mountainPosition <= -mountainWidth) {
            gameState->mountainPosition += mountainWidth;
        }
    }
}

void renderMountains(GameState* gameState) {
    Texture2D mountainTexture = gameState->mountainTextures[gameState->currentBackgroundIndex];
    if (mountainTexture.id != 0) {
        int mountainWidth = mountainTexture.width;
        int mountainY = 45; // y position per original game
        float playerCenterX = (PLAYER_MIN_X + PLAYER_MAX_X) / 2.0f;
        float playerOffset = gameState->player_x - playerCenterX;
        float parallaxFactor = 0.5f;
        float adjustedOffset = -playerOffset * parallaxFactor;
        float mountainX = -gameState->mountainPosition + adjustedOffset;
        BeginScissorMode(PLAYABLE_AREA_LEFT, 0, SCREEN_WIDTH - PLAYABLE_AREA_LEFT, SCREEN_HEIGHT);
        for (int x = (int)mountainX; x < SCREEN_WIDTH; x += mountainWidth) {
            DrawTexture(mountainTexture, x, mountainY, WHITE);
        }
        for (int x = (int)mountainX - mountainWidth; x > -mountainWidth; x -= mountainWidth) {
            DrawTexture(mountainTexture, x, mountainY, WHITE);
        }
        EndScissorMode();
    }
}

void c_render(Client* client, Enduro* env) {
    GameState* gameState = &client->gameState;

    // Copy necessary values from env to gameState
    gameState->speed = env->speed;
    gameState->min_speed = env->min_speed;
    gameState->max_speed = env->max_speed;
    gameState->current_curve_direction = env->current_curve_direction;
    gameState->current_curve_factor = env->current_curve_factor;
    gameState->player_x = env->player_x;
    gameState->player_y = env->player_y;
    gameState->initial_player_x = env->initial_player_x;
    gameState->vanishing_point_x = env->vanishing_point_x;
    gameState->t_p = env->t_p;
    gameState->dayCompleted = env->dayCompleted;
    gameState->currentBackgroundIndex = env->currentDayTimeIndex;
    gameState->carsLeftGameState = env->carsToPass;
    gameState->day = env->day;
    gameState->elapsedTime = env->elapsedTimeEnv;

    BeginDrawing();
    ClearBackground(BLACK);
    BeginBlendMode(BLEND_ALPHA);

    renderBackground(gameState);
    updateCarAnimation(gameState);
    updateMountains(gameState);
    renderMountains(gameState);
    
    int bgIndex = gameState->currentBackgroundIndex;
    unsigned char isNightFogStage = (bgIndex == 13);
    unsigned char isNightStage = (bgIndex == 12 || bgIndex == 13 || bgIndex == 14);

    // During night fog stage, clip rendering to y >= 92
    float clipStartY = isNightFogStage ? 92.0f : VANISHING_POINT_Y;
    float clipHeight = PLAYABLE_AREA_BOTTOM - clipStartY;
    Rectangle clipRect = { PLAYABLE_AREA_LEFT, clipStartY, PLAYABLE_AREA_RIGHT - PLAYABLE_AREA_LEFT, clipHeight };
    BeginScissorMode(clipRect.x, clipRect.y, clipRect.width, clipRect.height);

    // Render road edges w/ gl lines for original look
    // During night fog stage, start from y=92
    float roadStartY = isNightFogStage ? 92.0f : VANISHING_POINT_Y;
    Vector2 previousLeftPoint = {0}, previousRightPoint = {0};
    unsigned char firstPoint = true;

    for (float y = roadStartY; y <= PLAYABLE_AREA_BOTTOM; y += 0.75f) {
        float adjusted_y = (env->speed < 0) ? y : y + fmod(env->road_scroll_offset, 0.75f);
        if (adjusted_y > PLAYABLE_AREA_BOTTOM) continue;

        float left_edge = road_edge_x(env, adjusted_y, 0, true);
        float right_edge = road_edge_x(env, adjusted_y, 0, false);

        // Road is multiple shades of gray based on y position
        Color roadColor;
        if (adjusted_y >= 52 && adjusted_y < 91) {
            roadColor = (Color){74, 74, 74, 255};
        } else if (adjusted_y >= 91 && adjusted_y < 106) {
            roadColor = (Color){111, 111, 111, 255};
        } else if (adjusted_y >= 106 && adjusted_y <= 154) {
            roadColor = (Color){170, 170, 170, 255};
        } else {
            roadColor = WHITE;
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
    unsigned char skipFogCars = isNightFogStage;
    for (int i = 0; i < env->numEnemies; i++) {
        if (i >= env->max_enemies) {
    DEBUG_FPRINT(stderr, "Error: Accessing enemyCars[%d] out of bounds\n", i);
    exit(EXIT_FAILURE);
}

Car* car = &env->enemyCars[i];

if (car == NULL) {
    DEBUG_FPRINT(stderr, "Error: car pointer is NULL for enemyCars[%d]\n", i);
    exit(EXIT_FAILURE);
}
        
        // Don't render cars in fog
        if (skipFogCars && car->y < 92.0f) {
            continue;
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
        Texture2D carTexture;
        if (isNightStage) {
            carTexture = (bgIndex == 13) ? gameState->enemyCarNightFogTailLightsTexture : gameState->enemyCarNightTailLightsTexture;
        } else {
            int colorIndex = car->colorIndex;
            int treadIndex = gameState->showLeftTread ? 0 : 1;
            carTexture = gameState->enemyCarTextures[colorIndex][treadIndex];
        }
        // Compute car coords
        float car_center_x = car_x_in_lane(env, car->lane, car->y);
        float car_x = car_center_x - (carTexture.width * car_scale) / 2.0f;
        float car_y = car->y - (carTexture.height * car_scale) / 2.0f;

        DrawTextureEx(carTexture, (Vector2){car_x, car_y}, 0.0f, car_scale, WHITE);
    }

    updateCarAnimation(gameState);
    render_car(client, gameState); // Unscaled player car

    EndScissorMode();
    EndBlendMode();
    updateVictoryEffects(gameState);

    // Update GameState data from environment data;
    client->gameState.victoryAchieved = env->dayCompleted;

    // Do pure rendering operations
    updateScoreboard(gameState);
    renderScoreboard(gameState);
    EndDrawing();
}
