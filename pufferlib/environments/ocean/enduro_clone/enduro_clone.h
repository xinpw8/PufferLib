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
#include <time.h>
#include <string.h>

// Constant definitions
#define TARGET_FPS 60
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
#define MAX_ENEMIES 10
#define CRASH_NOOP_DURATION_CAR_VS_CAR 90
#define CRASH_NOOP_DURATION_CAR_VS_ROAD 20
#define INITIAL_CARS_TO_PASS 200
#define VANISHING_POINT_X 86
#define VANISHING_POINT_Y 52
#define VANISHING_POINT_TRANSITION_SPEED 1.0f
#define CURVE_TRANSITION_SPEED 0.05f
#define LOGICAL_VANISHING_Y (VANISHING_POINT_Y + 12)
#define INITIAL_PLAYER_X 86.0f
#define PLAYER_MAX_Y (ACTION_HEIGHT - CAR_HEIGHT)
#define PLAYER_MIN_Y (ACTION_HEIGHT - CAR_HEIGHT - 9)
#define ACCELERATION_RATE 0.2f
#define DECELERATION_RATE 0.1f
#define MIN_SPEED -2.5f
#define MAX_SPEED 7.5f
#define ENEMY_CAR_SPEED 0.1f
#define CURVE_STRAIGHT 0
#define CURVE_LEFT -1
#define CURVE_RIGHT 1
#define NUM_LANES 3
#define PLAYER_MIN_X 65.5f
#define PLAYER_MAX_X 91.5f
#define ROAD_LEFT_OFFSET 50.0f
#define ROAD_RIGHT_OFFSET 51.0f
#define VANISHING_POINT_X_LEFT 110.0f
#define VANISHING_POINT_X_RIGHT 62.0f
#define CURVE_VANISHING_POINT_SHIFT 55.0f
#define CURVE_PLAYER_SHIFT_FACTOR 0.025f
#define WIGGLE_AMPLITUDE 10.0f
#define WIGGLE_SPEED 10.1f
#define WIGGLE_LENGTH 26.0f

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
    int colorIndex; // Car color index (0-5)
} Car;

// Game environment struct
typedef struct Enduro {
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    unsigned char* truncateds;
    LogBuffer* log_buffer;
    Log log;
    float width;
    float height;
    float car_width;
    float car_height;
    int max_enemies;
    float elapsedTime;
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
    float collision_cooldown_car_vs_car;
    float collision_cooldown_car_vs_road;
    float collision_invulnerability_timer;
    int drift_direction;
    float action_height;
    Car enemyCars[MAX_ENEMIES];
    float initial_player_x;
    float road_scroll_offset;
    int current_curve_direction;
    float current_curve_factor;
    float target_curve_factor;
    float target_vanishing_point_x;
    float current_vanishing_point_x;
    float base_target_vanishing_point_x;
    float vanishing_point_x;
    float base_vanishing_point_x;
    float t_p;
    float wiggle_y;
    float wiggle_speed;
    float wiggle_length;
    float wiggle_amplitude;
    unsigned char wiggle_active;
    int currentGear;
    float gearSpeedThresholds[4];
    float gearAccelerationRates[4];
    float gearTimings[4];
    float gearElapsedTime;
    int currentStage;
    float enemySpawnTimer;
    float enemySpawnInterval;
    float last_road_left;
    float last_road_right;
    int closest_edge_lane;
    int last_spawned_lane;
    float totalAccelerationTime;
    float parallaxFactor;

    // Victory condition variables
    unsigned char victoryAchieved;       // Flag to indicate victory condition
    int flagTimer;              // Timer for alternating flags
    unsigned char showLeftFlag;          // Indicator for which flag to show
    int victoryDisplayTimer;    // Timer for how long victory effects have been displayed
    // Background state variables
    float backgroundTransitionTimes[16];
    int backgroundIndex;           // Index used for tracking background transitions
    int currentBackgroundIndex;    // Current background index
    int previousBackgroundIndex;   // Previous background index
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

// Prototypes
LogBuffer* allocate_logbuffer(int size);
void free_logbuffer(LogBuffer* buffer);
void add_log(LogBuffer* logs, Log* log);
Log aggregate_and_clear(LogBuffer* logs);
void init(Enduro* env);
void allocate(Enduro* env);
void free_allocated(Enduro* env);
void reset(Enduro* env);
unsigned char check_collision(Enduro* env, Car* car);
int get_player_lane(Enduro* env);
float get_car_scale(float y);
void add_enemy_car(Enduro* env);
void update_vanishing_point(Enduro* env, float offset);
void accelerate(Enduro* env);
void steppy(Enduro* env);
void update_road_curve(Enduro* env);
float quadratic_bezier(float bottom_x, float control_x, float top_x, float t);
float road_edge_x(Enduro* env, float y, float offset, unsigned char left);
float car_x_in_lane(Enduro* env, int lane, float y);
void updateVictoryEffects(Enduro* env);
void updateBackground(Enduro* env);
void compute_observations(Enduro* env);

// Function definitions
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
    }
    log.episode_return /= logs->idx;
    log.episode_length /= logs->idx;
    log.score /= logs->idx;
    logs->idx = 0;
    return log;
}

void init(Enduro* env) {
    env->max_enemies = MAX_ENEMIES;
    env->step_count = 0;
    env->score = 0;
    env->numEnemies = 0;
    env->collision_cooldown_car_vs_car = 0.0f;
    env->collision_cooldown_car_vs_road = 0.0f;
    env->action_height = ACTION_HEIGHT;
    env->elapsedTime = 0.0f;
    env->currentStage = 0;
    env->enemySpawnTimer = 0.0f;
    env->enemySpawnInterval = 0.8777f;
    env->last_spawned_lane = -1;
    env->closest_edge_lane = -1;
    env->totalAccelerationTime = 0.0f;
    env->base_vanishing_point_x = 86.0f;
    env->current_vanishing_point_x = 86.0f;
    env->target_vanishing_point_x = 86.0f;
    env->vanishing_point_x = 86.0f;
    env->initial_player_x = 86.0f;
    env->player_x = env->initial_player_x;
    env->player_y = PLAYER_MAX_Y;
    env->speed = env->min_speed;
    env->initial_cars_to_pass = INITIAL_CARS_TO_PASS;
    env->carsToPass = env->initial_cars_to_pass;
    env->day = 1;
    env->current_curve_direction = CURVE_STRAIGHT;
    env->current_curve_factor = 0.0f;
    env->target_curve_factor = 0.0f;
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
    // Initialize background transition times
    env->backgroundTransitionTimes[0] = 20.0f;
    env->backgroundTransitionTimes[1] = 40.0f;
    env->backgroundTransitionTimes[2] = 60.0f;
    env->backgroundTransitionTimes[3] = 100.0f;
    env->backgroundTransitionTimes[4] = 108.0f;
    env->backgroundTransitionTimes[5] = 114.0f;
    env->backgroundTransitionTimes[6] = 116.0f;
    env->backgroundTransitionTimes[7] = 120.0f;
    env->backgroundTransitionTimes[8] = 124.0f;
    env->backgroundTransitionTimes[9] = 130.0f;
    env->backgroundTransitionTimes[10] = 134.0f;
    env->backgroundTransitionTimes[11] = 138.0f;
    env->backgroundTransitionTimes[12] = 170.0f;
    env->backgroundTransitionTimes[13] = 198.0f;
    env->backgroundTransitionTimes[14] = 214.0f;
    env->backgroundTransitionTimes[15] = 232.0f;

    // Initialize victory condition variables
    env->victoryAchieved = false;
    env->flagTimer = 0;
    env->showLeftFlag = true;
    env->victoryDisplayTimer = 0;

    // Initialize background indices
    env->backgroundIndex = 0;
    env->currentBackgroundIndex = 0;
    env->previousBackgroundIndex = 15;

    env->initial_cars_to_pass = INITIAL_CARS_TO_PASS;
    env->carsToPass = env->initial_cars_to_pass;
}

void allocate(Enduro* env) {
    int obs_size = 6 + 2 * env->max_enemies + 3;
    env->observations = (float*)calloc(obs_size, sizeof(float));
    env->actions = (int*)calloc(1, sizeof(int));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->truncateds = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->log_buffer = allocate_logbuffer(LOG_BUFFER_SIZE);
    printf("sizes of each array: %d,\n", obs_size);
}

void free_allocated(Enduro* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env->truncateds);
    free_logbuffer(env->log_buffer);
}

void reset(Enduro* env) {
    env->max_enemies = MAX_ENEMIES;
    env->step_count = 0;
    env->score = 0;
    env->numEnemies = 0;
    env->collision_cooldown_car_vs_car = 0.0f;
    env->collision_cooldown_car_vs_road = 0.0f;
    env->action_height = ACTION_HEIGHT;
    env->elapsedTime = 0.0f;
    env->currentStage = 0;
    env->enemySpawnTimer = 0.0f;
    env->enemySpawnInterval = 0.8777f;
    env->last_spawned_lane = -1;
    env->closest_edge_lane = -1;
    env->totalAccelerationTime = 0.0f;
    env->base_vanishing_point_x = 86.0f;
    env->current_vanishing_point_x = 86.0f;
    env->target_vanishing_point_x = 86.0f;
    env->vanishing_point_x = 86.0f;
    env->initial_player_x = 86.0f;
    env->player_x = env->initial_player_x;
    env->player_y = PLAYER_MAX_Y;
    env->speed = env->min_speed;
    env->initial_cars_to_pass = INITIAL_CARS_TO_PASS;
    env->carsToPass = env->initial_cars_to_pass;
    env->day = 1;
    env->current_curve_direction = CURVE_STRAIGHT;
    env->current_curve_factor = 0.0f;
    env->target_curve_factor = 0.0f;
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
    // Initialize background transition times
    env->backgroundTransitionTimes[0] = 20.0f;
    env->backgroundTransitionTimes[1] = 40.0f;
    env->backgroundTransitionTimes[2] = 60.0f;
    env->backgroundTransitionTimes[3] = 100.0f;
    env->backgroundTransitionTimes[4] = 108.0f;
    env->backgroundTransitionTimes[5] = 114.0f;
    env->backgroundTransitionTimes[6] = 116.0f;
    env->backgroundTransitionTimes[7] = 120.0f;
    env->backgroundTransitionTimes[8] = 124.0f;
    env->backgroundTransitionTimes[9] = 130.0f;
    env->backgroundTransitionTimes[10] = 134.0f;
    env->backgroundTransitionTimes[11] = 138.0f;
    env->backgroundTransitionTimes[12] = 170.0f;
    env->backgroundTransitionTimes[13] = 198.0f;
    env->backgroundTransitionTimes[14] = 214.0f;
    env->backgroundTransitionTimes[15] = 232.0f;

    // Initialize victory condition variables
    env->victoryAchieved = false;
    env->flagTimer = 0;
    env->showLeftFlag = true;
    env->victoryDisplayTimer = 0;

    // Initialize background indices
    env->backgroundIndex = 0;
    env->currentBackgroundIndex = 0;
    env->previousBackgroundIndex = 15;

    env->initial_cars_to_pass = INITIAL_CARS_TO_PASS;
    env->carsToPass = env->initial_cars_to_pass;

    env->rewards[0] = 0;
    env->log.episode_return = 0;
    env->log.episode_length = 0;
    env->log.score = 0;
    add_log(env->log_buffer, &env->log);
}

unsigned char check_collision(Enduro* env, Car* car) {
    float depth = (car->y - VANISHING_POINT_Y) / (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y);
    float scale = fmax(0.1f, 0.9f * depth);
    float car_width = CAR_WIDTH * scale;
    float car_height = CAR_HEIGHT * scale;
    float car_center_x = car_x_in_lane(env, car->lane, car->y);
    float car_x = car_center_x - car_width / 2.0f;
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
        float player_center_x = env->player_x + CAR_WIDTH / 2.0f;
        float road_center_x = (road_edge_x(env, env->player_y, 0, true) +
                            road_edge_x(env, env->player_y, 0, false)) / 2.0f;
        if (player_center_x < road_center_x) {
            furthest_lane = 2;
        } else {
            furthest_lane = 0;
        }
    }

    if (env->speed <= 0) {
        possible_lanes[num_possible_lanes++] = furthest_lane;
    } else {
        for (int i = 0; i < NUM_LANES; i++) {
            possible_lanes[num_possible_lanes++] = i;
        }
    }

    if (num_possible_lanes == 0) {
        return;
    }

    int lane = possible_lanes[rand() % num_possible_lanes];

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
    float dynamic_spacing_factor = rand() % 6 + 0.5f;
    float min_spacing = dynamic_spacing_factor * scaled_car_length;
    for (int i = 0; i < env->numEnemies; i++) {
        Car* existing_car = &env->enemyCars[i];
        if (existing_car->lane == car.lane) {
            float y_distance = fabs(existing_car->y - car.y);
            if (y_distance < min_spacing) {
                return;
            }
        }
    }

    // Ensure not occupying all lanes within vertical range
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

void update_vanishing_point(Enduro* env, float offset) {
    env->vanishing_point_x = env->base_vanishing_point_x + offset;
}

void accelerate(Enduro* env) {
    if (env->speed < env->max_speed) {
        if (env->speed >= env->gearSpeedThresholds[env->currentGear] && env->currentGear < 3) {
            env->currentGear++;
            env->gearElapsedTime = 0.0f;
        }

        env->speed += env->gearAccelerationRates[env->currentGear] * 2.0f;
        if (env->speed > env->gearSpeedThresholds[env->currentGear]) {
            env->speed = env->gearSpeedThresholds[env->currentGear];
        }
        env->totalAccelerationTime += (1.0f / TARGET_FPS);
    }
}

void steppy(Enduro* env) {
    // Increment elapsed time by frame duration
    // Used for rendering but incremented in step()
    env->elapsedTime += (1.0f / TARGET_FPS);
    updateBackground(env);

    update_road_curve(env);
    env->log.episode_length += 1;
    env->terminals[0] = 0;
    env->road_scroll_offset += env->speed;

    // Update enemy car positions
    for (int i = 0; i < env->numEnemies; i++) {
        Car* car = &env->enemyCars[i];
        // Compute movement speed adjusted for scaling
        float scale = get_car_scale(car->y);
        float movement_speed = env->speed * scale * 0.75f;
        // Update car position
        car->y += movement_speed;
    }

    // Calculate road edges
    float road_left = road_edge_x(env, env->player_y, 0, true);
    float road_right = road_edge_x(env, env->player_y, 0, false) - CAR_WIDTH;
    env->last_road_left = road_left;
    env->last_road_right = road_right;

    // Reduced handling on snow
    unsigned char isSnowStage = (env->currentBackgroundIndex == 3);
    float movement_amount = 0.5f; // Default
    if (isSnowStage) {
        movement_amount = 0.3f; // Snow
    }
    
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

    // if (env->drift_direction != 0 && env->collision_cooldown_car_vs_car > 0 && env->collision_cooldown_car_vs_road <= 0) {
    //     env->collision_cooldown_car_vs_road = env->collision_cooldown_car_vs_car;
    // }
    if (env->collision_cooldown_car_vs_car > 0) {
    env->collision_cooldown_car_vs_car -= 1;
    }
    if (env->collision_cooldown_car_vs_road > 0) {
    env->collision_cooldown_car_vs_road -= 1;
    }

    // Collision cooldown debugging print
    if (((int)round(env->collision_cooldown_car_vs_car)) % 5 == 0 && env->collision_cooldown_car_vs_car > 0) {
    // printf("Collision cooldown vs car: %.2f\n", env->collision_cooldown_car_vs_car);
    }
    if (((int)round(env->collision_cooldown_car_vs_road)) % 5 == 0 && env->collision_cooldown_car_vs_road > 0) {
    // printf("Collision cooldown car vs road: %.2f\n", env->collision_cooldown_car_vs_road);
    }

    // Drift towards furthest road edge
    if (env->drift_direction == 0) { // drift_direction is 0 when noop starts
        env->drift_direction = (env->player_x > (road_left + road_right) / 2) ? -1 : 1;
        // Remove enemy cars in middle lane and lane player is drifting towards
        // only if they are behind the player (y > player_y) to avoid crashes
        for (int i = 0; i < env->numEnemies; i++) {
            Car* car = &env->enemyCars[i];
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

    // Road curve/vanishing point movement logic
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
        // printf("Speed: %.2f, Normalized Speed: %.2f, Wiggle Period: %.2f, Wiggle Speed: %.2f\n", 
            // env->speed, speed_normalized, current_wiggle_period, env->wiggle_speed);
    }

    // Player car moves forward slightly according to speed
    // Update player y position based on speed
    env->player_y = PLAYER_MAX_Y - (env->speed - env->min_speed) / (env->max_speed - env->min_speed) * (PLAYER_MAX_Y - PLAYER_MIN_Y);
    // Clamp player_y to measured range
    if (env->player_y < PLAYER_MIN_Y) env->player_y = PLAYER_MIN_Y;
    if (env->player_y > PLAYER_MAX_Y) env->player_y = PLAYER_MAX_Y;

    // Check for and handle collisions between player and road edges
    if (env->player_x <= road_left || env->player_x >= road_right) {
        env->speed = fmaxf((env->speed - 1.0f), MIN_SPEED);
        env->collision_cooldown_car_vs_road = CRASH_NOOP_DURATION_CAR_VS_ROAD;
        env->drift_direction = 0; // Reset drift direction, has priority over car collisions
        env->player_x = fmaxf(road_left + 1, fminf(road_right - 1, env->player_x));
        // printf("edge left: %.2f, edge right: %.2f\n", road_left, road_right);
        // printf("Player x: %.2f, Player y: %.2f, Speed: %.2f\n", env->player_x, env->player_y, env->speed);
    }

    // Enemy car logic
    for (int i = 0; i < env->numEnemies; i++) {
        Car* car = &env->enemyCars[i];

        // Check for passing logic **only if not on collision cooldown**
        if (env->speed > 0 && car->y > env->player_y + CAR_HEIGHT && !car->passed && env->collision_cooldown_car_vs_car <= 0 && env->collision_cooldown_car_vs_road <= 0) {
            // Mark car as passed and decrement carsToPass
            env->carsToPass--;
            add_log(env->log_buffer, &env->log);
            if (env->carsToPass < 0) env->carsToPass = 0; // Stops at 0; resets to 300 on new day
            car->passed = true;
            env->score += 1;
            env->rewards[0] += 1;
            env->log.episode_return += 1;
            // Passing debug
            // printf("Car passed at y = %.2f. Remaining cars to pass: %d\n", car->y, env->carsToPass);
        }

        // Ignore collisions for 1 second to avoid edge-case chain collisions
        if (env->collision_cooldown_car_vs_car > 0) {
            if (env->collision_invulnerability_timer <= 0) {
                env->collision_invulnerability_timer = TARGET_FPS * 1.0f;
                // printf("Invulnerability timer started\n");
            }
        } else if (env->collision_invulnerability_timer > 0) {
            env->collision_invulnerability_timer -= 1;
        }

        // Check for and handle collisions between player and enemy cars
        if (env->collision_cooldown_car_vs_car <= 0 && env->collision_invulnerability_timer <= 0) {
            if (check_collision(env, car)) {
                env->speed = 1 + MIN_SPEED;
                env->collision_cooldown_car_vs_car = CRASH_NOOP_DURATION_CAR_VS_CAR;
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

        // If the car is behind the player and speed ≤ 0, move it to the furthest lane
        if (env->speed <= 0 && car->y >= env->player_y + CAR_HEIGHT) {
            // Determine the furthest lane
            int furthest_lane;
            int player_lane = get_player_lane(env);
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
                    furthest_lane = 2; // Player is on the left side
                } else {
                    furthest_lane = 0; // Player is on the right side
                }
            }
            car->lane = furthest_lane;
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

    // Calculate speed factor
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
            // printf("Enemy car spawned at time %.2f seconds\n", env->elapsedTime);
        }
    }

    updateVictoryEffects(env);

    // Day completed logic
    if (env->carsToPass <= 0 && !env->victoryAchieved) {
        env->victoryAchieved = true;
        printf("Day complete! Continue racing until day ends.\n");
    }

    // Handle day transition when background cycles back to 0
    if (env->currentBackgroundIndex == 0 && env->previousBackgroundIndex == 15) {
        // Background cycled back to 0
        if (env->victoryAchieved) {
            // Player has achieved victory, start new day
            env->day++;
            env->carsToPass = 300; // Always 300 after the first day
            env->victoryAchieved = false;
            printf("Starting day %d with %d cars to pass.\n", env->day, env->carsToPass);
        } else {
            // Player failed to pass required cars, reset environment
            env->terminals[0] = 1; // Signal termination
            reset(env); // Reset the game
            printf("Day %d failed. Resetting game.\n", env->day);
            return;
        }
    }

    env->step_count++;
    env->log.score = env->score;
    compute_observations(env);
    add_log(env->log_buffer, &env->log);

    // // Enemy car position debugging print
    // printf("Positions for all enemy cars: ");
    // for (int i = 0; i < env->numEnemies; i++) {
    //     printf("Enemy %d: y = %f ", i, env->enemyCars[i].y);
    // }
    // printf("\n");

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
    // int curve_directions[] = {0, -1, 0, 1, 0, -1, 0, 1, -1, 1, 1, 1, 1, 1, 1, 1};   
        int curve_directions[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};   
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

float car_x_in_lane(Enduro* env, int lane, float y) {
    float offset = 0.0f;
    float left_edge = road_edge_x(env, y, offset, true);
    float right_edge = road_edge_x(env, y, offset, false);
    float lane_width = (right_edge - left_edge) / NUM_LANES;
    return left_edge + lane_width * (lane + 0.5f);
}

void updateVictoryEffects(Enduro* env) {
    if (env->victoryAchieved) {
        // Dancing flags effect
        env->flagTimer++;
        if (env->flagTimer >= 240) { // Switch every 240 frames (~4 seconds)
            env->flagTimer = 0;
            env->showLeftFlag = !env->showLeftFlag;
        }
        // Update victory display timer
        env->victoryDisplayTimer++;
        if (env->victoryDisplayTimer >= 540) { // Display flags for 540 frames (~9 seconds)
            // Reset victory display timer
            env->victoryDisplayTimer = 0;
        }
    }
}

void updateBackground(Enduro* env) {
    float elapsedTime = env->elapsedTime;

    // Total duration of the cycle
    float totalDuration = env->backgroundTransitionTimes[15];

    // If elapsed time exceeds total duration, reset it
    if (elapsedTime >= totalDuration) {
        elapsedTime -= totalDuration;
        env->elapsedTime = elapsedTime; // Reset elapsed time in env
        env->backgroundIndex = 0;
    }

    // Update previous background index before changing it
    env->previousBackgroundIndex = env->currentBackgroundIndex;

    // Determine the current background index
    while (env->backgroundIndex < 15 &&
           elapsedTime >= env->backgroundTransitionTimes[env->backgroundIndex]) {
        env->backgroundIndex++;
    }
    env->currentBackgroundIndex = env->backgroundIndex % 16;

    // Logging for verification (optional)
    // printf("Elapsed Time: %.2f s, Background Index: %d\n", elapsedTime, env->currentBackgroundIndex);
}

void compute_observations(Enduro* env) {
    float* obs = env->observations;

    // Normalize player's x position
    float player_x_norm = (env->player_x - PLAYER_MIN_X) / (PLAYER_MAX_X - PLAYER_MIN_X);
    obs[0] = player_x_norm;

    // Normalize player's y position
    float player_y_norm = (env->player_y - PLAYER_MIN_Y) / (PLAYER_MAX_Y - PLAYER_MIN_Y);
    obs[1] = player_y_norm;

    // Normalize player's speed
    float speed_norm = (env->speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED);
    obs[2] = speed_norm;

    // Compute road edges at player's y position
    float road_left = road_edge_x(env, env->player_y, 0, true);
    float road_right = road_edge_x(env, env->player_y, 0, false);

    // Normalize road edges
    float road_left_norm = (road_left - PLAYABLE_AREA_LEFT) / (PLAYABLE_AREA_RIGHT - PLAYABLE_AREA_LEFT);
    float road_right_norm = (road_right - PLAYABLE_AREA_LEFT) / (PLAYABLE_AREA_RIGHT - PLAYABLE_AREA_LEFT);
    obs[3] = road_left_norm;
    obs[4] = road_right_norm;

    // Normalize player's lane (assuming NUM_LANES = 3)
    int player_lane = get_player_lane(env);
    float player_lane_norm = (float)player_lane / (NUM_LANES - 1);
    obs[5] = player_lane_norm;

    // Initialize index for enemy car observations
    int idx = 6;

    // For each enemy car, compute normalized relative positions
    // idx 6-15
    for (int i = 0; i < env->max_enemies; i++) {
        Car* car = &env->enemyCars[i];

        // Check if the enemy car is active
        if (car->y > 0 && car->y < env->height) {
            // Compute enemy car's x position
            float car_x = car_x_in_lane(env, car->lane, car->y);

            // Compute relative positions
            float relative_x = car_x - env->player_x;
            float relative_y = car->y - env->player_y;

            // Normalize relative positions
            float max_relative_x = PLAYABLE_AREA_RIGHT - PLAYABLE_AREA_LEFT;
            float relative_x_norm = (relative_x + max_relative_x) / (2 * max_relative_x);
            float max_relative_y = env->height;
            float relative_y_norm = (relative_y + max_relative_y) / (2 * max_relative_y);

            obs[idx++] = relative_x_norm;
            obs[idx++] = relative_y_norm;
        } else {
            // If the enemy car is not active, fill with default values
            obs[idx++] = 0.0f;
            obs[idx++] = 0.0f;
        }
    }

    // Add current curve direction to observations
    // idx 16
    float curve_direction_norm = (float)(env->current_curve_direction + 1) / 2.0f;
    obs[idx++] = curve_direction_norm;

    // Compute normalized time of day
    // Total day length is the last background transition time
    // idx 17-31
    float total_day_length = env->backgroundTransitionTimes[15];
    float time_of_day = fmodf(env->elapsedTime, total_day_length);
    float time_of_day_norm = time_of_day / total_day_length;
    obs[idx++] = time_of_day_norm;

    // Add normalized carsToPass
    // idx 32
    float carsToPass_norm = (float)env->carsToPass / (float)env->initial_cars_to_pass;
    obs[idx++] = carsToPass_norm;

    // Compute the expected number of observations
    int obs_size = 6 + 2 * env->max_enemies + 3;

    if (idx != obs_size) {
        printf("Error: Expected idx to be %d but got %d\n", obs_size, idx);
    }

    // // After computing observations, print them
    // printf("Observations for environment:\n");
    // for (int i = 0; i < obs_size; i++) {
    //     printf("%f ", env->observations[i]);
    // }
    // printf("\n");
}

#endif // ENDURO_CLONE_H