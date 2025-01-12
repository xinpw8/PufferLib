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
#include <float.h>
#include "raylib.h"

#define MAX_ENEMIES 10
#define OBSERVATIONS_MAX_SIZE (8 + (5 * MAX_ENEMIES) + 9 + 1) // fyi
#define TARGET_FPS 60
#define LOG_BUFFER_SIZE 4096
#define SCREEN_WIDTH 152
#define SCREEN_HEIGHT 210
#define PLAYABLE_AREA_TOP 0
#define PLAYABLE_AREA_BOTTOM 154
#define PLAYABLE_AREA_LEFT 0
#define PLAYABLE_AREA_RIGHT 152
#define ACTION_HEIGHT (PLAYABLE_AREA_BOTTOM - PLAYABLE_AREA_TOP)
#define CAR_WIDTH 16
#define CAR_HEIGHT 11
#define CRASH_NOOP_DURATION_CAR_VS_CAR 90  // How long controls are disabled after car v car collision
#define CRASH_NOOP_DURATION_CAR_VS_ROAD 20 // How long controls are disabled after car v road edge collision
#define INITIAL_CARS_TO_PASS 200
#define VANISHING_POINT_Y 52
#define VANISH_START_POINT 86.0f
#define MAX_DISTANCE (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y)
#define VANISHING_POINT_TRANSITION_SPEED 1.0f
#define CURVE_TRANSITION_SPEED 0.05f
#define LOGICAL_VANISHING_Y (VANISHING_POINT_Y + 12)  // Separate logical vanishing point for cars disappearing
#define PLAYER_MAX_Y (ACTION_HEIGHT - CAR_HEIGHT)     // Max y is car length from bottom
#define PLAYER_MIN_Y (ACTION_HEIGHT - CAR_HEIGHT - 9) // Min y is ~2 car lengths from bottom
#define DECELERATION_RATE 0.01f
#define MIN_SPEED -2.5f
#define MAX_SPEED 7.5f
#define ENEMY_CAR_SPEED 0.1f
// Constants for spawn interval configuration
#define NUM_MAX_SPAWN_INTERVALS 3
static const float MAX_SPAWN_INTERVALS[] = {0.5f, 0.25f, 0.4f};
static const float MIN_SPAWN_INTERVAL = 0.5f;
static const float SPAWN_SCALING_FACTOR = 1.5f;
static const float DAILY_INTERVAL_REDUCTION = 0.1f;
static const float MIN_POSSIBLE_INTERVAL = 0.1f;
// Times of day logic
#define NUM_BACKGROUND_TRANSITIONS 16
// Seconds spent in each time of day
static const float BACKGROUND_TRANSITION_TIMES[] = {
    20.0f, 40.0f, 60.0f, 100.0f, 108.0f, 114.0f, 116.0f, 120.0f,
    124.0f, 130.0f, 134.0f, 138.0f, 170.0f, 198.0f, 214.0f, 232.0f
};
// Curve constants
#define CURVE_STRAIGHT 0
#define CURVE_LEFT -1
#define CURVE_RIGHT 1
#define NUM_LANES 3
#define CURVE_VANISHING_POINT_SHIFT 55.0f
#define CURVE_PLAYER_SHIFT_FACTOR 0.025f // Moves player car towards outside edge of curves
// Curve wiggle effect timing and amplitude
#define WIGGLE_AMPLITUDE 10.0f  // Maximum 'bump-in' offset in pixels
#define WIGGLE_SPEED 10.1f      // Speed at which the wiggle moves down the screen
#define WIGGLE_LENGTH 26.0f     // Vertical length of the wiggle effect
// Rendering constants
#define SCORE_DIGITS 5
#define CARS_DIGITS  4
#define DIGIT_WIDTH 8
#define DIGIT_HEIGHT 9
#define INITIAL_PLAYER_X 69.0f // Adjusted from 77.0f
#define PLAYER_MIN_X 57.5f     // Adjusted from 65.5f
#define PLAYER_MAX_X 83.5f     // Adjusted from 91.5f
#define VANISHING_POINT_X_LEFT 102.0f // Adjusted from 110.0f
#define VANISHING_POINT_X_RIGHT 54.0f // Adjusted from 62.0f
#define ROAD_LEFT_OFFSET 46.0f  // Adjusted from 50.0f
#define ROAD_RIGHT_OFFSET 47.0f // Adjusted from 51.0f

typedef struct Log Log;
struct Log {
    float episode_return;
    float episode_length;
    float score;
    float reward;
    float step_rew_car_passed_no_crash;
    float crashed_penalty;
    float passed_cars;
    float passed_by_enemy;
    int cars_to_pass;
    float days_completed;
    float days_failed;
    float collisions_player_vs_car;
    float collisions_player_vs_road;
};

typedef struct LogBuffer LogBuffer;
struct LogBuffer {
    Log* logs;
    int length;
    int idx;
};

typedef struct Car {
    int lane;       // Lane index: 0=left lane, 1=mid, 2=right lane
    float x;        // Current x position
    float y;        // Current y position
    float last_x;   // x post last step
    float last_y;   // y post last step
    int passed;     // Flag to indicate if car has been passed by player
    int color_index; // Car color idx (0-5)
} Car;

typedef struct GameState {
    Texture2D spritesheet;
    RenderTexture2D render_target; // for scaling up render
    // Indices into asset_map[] for various assets
    int background_indices[16];
    int mountain_indices[16];
    int digit_indices[11];         // 0-9 and "CAR" digit
    int green_digit_indices[10];    // Green digits 0-9
    int yellow_digit_indices[10];   // Yellow digits 0-9
    // Enemy car indices: [color][tread]
    int enemy_car_indices[6][2];
    int enemy_car_night_taillights_index;
    int enemy_car_night_fog_taillights_index;
    int player_car_left_tread_index;  // Animates player car tire treads
    int player_car_right_tread_index;
    // Flag indices
    int level_complete_flag_left_index;
    int level_complete_flag_right_index;
    // For car animation
    float car_animation_timer;
    float car_animation_interval;
    unsigned char show_left_tread;
    float mountain_pos; // Position of the mountain texture
    // Variables for alternating flags
    unsigned char victory_achieved;
    int flag_timer;
    unsigned char show_left_flag; // true shows left flag, false shows right flag
    int victory_display_timer;    // Timer for how long victory effects have been displayed
    // Variables for scrolling digits    
    float score_digit_offsets[SCORE_DIGITS];    // Offset for scrolling effect for each digit
    int score_digit_currents[SCORE_DIGITS];     // Current digit being displayed for each position
    int score_digit_nexts[SCORE_DIGITS];        // Next digit to scroll in for each position
    unsigned char score_digit_scrolling[SCORE_DIGITS]; // Scrolling state for each digit
    int score_timer; // Timer to control score increment
} GameState;

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
    int max_enemies;
    float elapsed_time_env;
    float min_speed;
    float player_x;
    float player_y;
    float speed;
    float score;
    int day;
    int lane;
    int step_count;
    int num_enemies;
    int cars_to_pass;
    float collision_cooldown_car_vs_car;  // Timer for car vs car collisions
    float collision_cooldown_car_vs_road; // Timer for car vs road edge collisions
    int drift_direction; // Which way player car drifts whilst nooped after crash vs car
    Car enemy_cars[MAX_ENEMIES];
    float road_scroll_offset;
    // Road curve variables
    int current_curve_stage;
    int steps_in_current_stage;
    int current_curve_direction; // 1: Right, -1: Left, 0: Straight
    float current_curve_factor;
    float target_curve_factor;
    float current_step_threshold;
    float target_vanishing_point_x;     
    float current_vanishing_point_x;    
    float base_target_vanishing_point_x;
    float vanishing_point_x;
    float base_vanishing_point_x;
    float t_p;
    // Roadside wiggle effect
    float wiggle_y;            
    float wiggle_speed;        
    float wiggle_length;       
    float wiggle_amplitude;    
    unsigned char wiggle_active; 
    // Player car acceleration
    int current_gear;
    float gear_speed_thresholds[4]; 
    float gear_acceleration_thresholds[4];
    // Enemy spawning
    float enemy_spawn_timer;
    float enemy_spawn_interval; // Spawn interval based on current stage
    unsigned char day_completed; 
    float day_transition_times[NUM_BACKGROUND_TRANSITIONS];
    // Logging
    float last_road_left;
    float last_road_right;
    int last_spawned_lane;
    int current_day_time_index;
    int previous_day_time_index;
    int day_time_index;
    unsigned int rng_state;
    unsigned int index;
    int reset_count;
    // Rewards
    unsigned char car_passed_no_crash_active; 
    float step_rew_car_passed_no_crash; 
    float crashed_penalty;
} Enduro;

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

Rectangle asset_map[] = {
    (Rectangle){ 328, 15, 152, 210 },  // 0_bg
    (Rectangle){ 480, 15, 152, 210 },  // 1_bg
    (Rectangle){ 632, 15, 152, 210 },  // 2_bg
    (Rectangle){ 784, 15, 152, 210 },  // 3_bg
    (Rectangle){ 0, 225, 152, 210 },   // 4_bg
    (Rectangle){ 152, 225, 152, 210 }, // 5_bg
    (Rectangle){ 304, 225, 152, 210 }, // 6_bg
    (Rectangle){ 456, 225, 152, 210 }, // 7_bg
    (Rectangle){ 608, 225, 152, 210 }, // 8_bg
    (Rectangle){ 760, 225, 152, 210 }, // 9_bg
    (Rectangle){ 0, 435, 152, 210 },   // 10_bg
    (Rectangle){ 152, 435, 152, 210 }, // 11_bg
    (Rectangle){ 304, 435, 152, 210 }, // 12_bg
    (Rectangle){ 456, 435, 152, 210 }, // 13_bg
    (Rectangle){ 608, 435, 152, 210 }, // 14_bg
    (Rectangle){ 760, 435, 152, 210 }, // 15_bg
    (Rectangle){ 0, 0, 100, 6 },       // 0_mtns
    (Rectangle){ 100, 0, 100, 6 },     // 1_mtns
    (Rectangle){ 200, 0, 100, 6 },     // 2_mtns
    (Rectangle){ 300, 0, 100, 6 },     // 3_mtns
    (Rectangle){ 400, 0, 100, 6 },     // 4_mtns
    (Rectangle){ 500, 0, 100, 6 },     // 5_mtns
    (Rectangle){ 600, 0, 100, 6 },     // 6_mtns
    (Rectangle){ 700, 0, 100, 6 },     // 7_mtns
    (Rectangle){ 800, 0, 100, 6 },     // 8_mtns
    (Rectangle){ 0, 6, 100, 6 },       // 9_mtns
    (Rectangle){ 100, 6, 100, 6 },     // 10_mtns
    (Rectangle){ 200, 6, 100, 6 },     // 11_mtns
    (Rectangle){ 300, 6, 100, 6 },     // 12_mtns
    (Rectangle){ 400, 6, 100, 6 },     // 13_mtns
    (Rectangle){ 500, 6, 100, 6 },     // 14_mtns
    (Rectangle){ 600, 6, 100, 6 },     // 15_mtns
    (Rectangle){ 700, 6, 8, 9 },       // digits_0
    (Rectangle){ 708, 6, 8, 9 },       // digits_1
    (Rectangle){ 716, 6, 8, 9 },       // digits_2
    (Rectangle){ 724, 6, 8, 9 },       // digits_3
    (Rectangle){ 732, 6, 8, 9 },       // digits_4
    (Rectangle){ 740, 6, 8, 9 },       // digits_5
    (Rectangle){ 748, 6, 8, 9 },       // digits_6
    (Rectangle){ 756, 6, 8, 9 },       // digits_7
    (Rectangle){ 764, 6, 8, 9 },       // digits_8
    (Rectangle){ 772, 6, 8, 9 },       // digits_9
    (Rectangle){ 780, 6, 8, 9 },       // digits_car
    (Rectangle){ 788, 6, 8, 9 },       // green_digits_0
    (Rectangle){ 796, 6, 8, 9 },       // green_digits_1
    (Rectangle){ 804, 6, 8, 9 },       // green_digits_2
    (Rectangle){ 812, 6, 8, 9 },       // green_digits_3
    (Rectangle){ 820, 6, 8, 9 },       // green_digits_4
    (Rectangle){ 828, 6, 8, 9 },       // green_digits_5
    (Rectangle){ 836, 6, 8, 9 },       // green_digits_6
    (Rectangle){ 844, 6, 8, 9 },       // green_digits_7
    (Rectangle){ 852, 6, 8, 9 },       // green_digits_8
    (Rectangle){ 860, 6, 8, 9 },       // green_digits_9
    (Rectangle){ 932, 6, 8, 9 },       // yellow_digits_0
    (Rectangle){ 0, 15, 8, 9 },        // yellow_digits_1
    (Rectangle){ 8, 15, 8, 9 },        // yellow_digits_2
    (Rectangle){ 16, 15, 8, 9 },       // yellow_digits_3
    (Rectangle){ 24, 15, 8, 9 },       // yellow_digits_4
    (Rectangle){ 32, 15, 8, 9 },       // yellow_digits_5
    (Rectangle){ 40, 15, 8, 9 },       // yellow_digits_6
    (Rectangle){ 48, 15, 8, 9 },       // yellow_digits_7
    (Rectangle){ 56, 15, 8, 9 },       // yellow_digits_8
    (Rectangle){ 64, 15, 8, 9 },       // yellow_digits_9
    (Rectangle){ 72, 15, 16, 11 },     // enemy_car_blue_left_tread
    (Rectangle){ 88, 15, 16, 11 },     // enemy_car_blue_right_tread
    (Rectangle){ 104, 15, 16, 11 },    // enemy_car_gold_left_tread
    (Rectangle){ 120, 15, 16, 11 },    // enemy_car_gold_right_tread
    (Rectangle){ 168, 15, 16, 11 },    // enemy_car_pink_left_tread
    (Rectangle){ 184, 15, 16, 11 },    // enemy_car_pink_right_tread
    (Rectangle){ 200, 15, 16, 11 },    // enemy_car_salmon_left_tread
    (Rectangle){ 216, 15, 16, 11 },    // enemy_car_salmon_right_tread
    (Rectangle){ 232, 15, 16, 11 },    // enemy_car_teal_left_tread
    (Rectangle){ 248, 15, 16, 11 },    // enemy_car_teal_right_tread
    (Rectangle){ 264, 15, 16, 11 },    // enemy_car_yellow_left_tread
    (Rectangle){ 280, 15, 16, 11 },    // enemy_car_yellow_right_tread
    (Rectangle){ 136, 15, 16, 11 },    // enemy_car_night_fog_tail_lights
    (Rectangle){ 152, 15, 16, 11 },    // enemy_car_night_tail_lights
    (Rectangle){ 296, 15, 16, 11 },    // player_car_left_tread
    (Rectangle){ 312, 15, 16, 11 },    // player_car_right_tread
    (Rectangle){ 900, 6, 32, 9 },      // level_complete_flag_right
    (Rectangle){ 868, 6, 32, 9 },      // level_complete_flag_left
};

unsigned int xorshift32(unsigned int *state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static void remove_enemy_car(Enduro* env, int i) { // Prune ith enemy car
    for (int j = i; j < env->num_enemies - 1; j++) {
        env->enemy_cars[j] = env->enemy_cars[j + 1];
    }
    env->num_enemies--;
}

static int get_furthest_lane(const Enduro* env) { // Get furthest lane from player
    if (env->lane == 0) {
        return 2;
    } else if (env->lane == 2) {
        return 0;
    } else {
        float player_center_x = env->player_x + CAR_WIDTH / 2.0f;
        float road_center_x   = (env->last_road_left + env->last_road_right) / 2.0f;
        return (player_center_x < road_center_x) ? 2 : 0;
    }
}

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

void add_log(LogBuffer* logs, const Log* log) {
    if (logs->idx == logs->length) {
        return;
    }
    logs->logs[logs->idx] = *log;
    logs->idx += 1;
}

Log aggregate_and_clear(LogBuffer* logs) {
    Log log = {0};
    if (logs->idx == 0) {
        return log;
    }
    for (int i = 0; i < logs->idx; i++) {
        log.episode_return               += logs->logs[i].episode_return /= logs->idx;
        log.episode_length               += logs->logs[i].episode_length /= logs->idx;
        log.score                        += logs->logs[i].score /= logs->idx;
        log.reward                       += logs->logs[i].reward /= logs->idx;
        log.step_rew_car_passed_no_crash += logs->logs[i].step_rew_car_passed_no_crash /= logs->idx;
        log.crashed_penalty              += logs->logs[i].crashed_penalty /= logs->idx;
        log.passed_cars                  += logs->logs[i].passed_cars /= logs->idx;
        log.passed_by_enemy              += logs->logs[i].passed_by_enemy /= logs->idx;
        log.cars_to_pass                 += logs->logs[i].cars_to_pass /= logs->idx;
        log.days_completed               += logs->logs[i].days_completed /= logs->idx;
        log.days_failed                  += logs->logs[i].days_failed /= logs->idx;
        log.collisions_player_vs_car     += logs->logs[i].collisions_player_vs_car /= logs->idx;
        log.collisions_player_vs_road    += logs->logs[i].collisions_player_vs_road /= logs->idx;
    }
    logs->idx = 0;
    return log;
}

void allocate(Enduro* env) {
    env->observations = (float*)calloc(env->obs_size, sizeof(float));
    env->actions      = (int*)calloc(1, sizeof(int));
    env->rewards      = (float*)calloc(1, sizeof(float));
    env->terminals    = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->truncateds   = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->log_buffer   = allocate_logbuffer(LOG_BUFFER_SIZE);
}

void free_allocated(Enduro* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env->truncateds);
    free_logbuffer(env->log_buffer);
}

void init(Enduro* env, int seed, int env_index) {
    env->index = env_index;
    env->rng_state = seed;

    if (seed == 0) { // Activate with seed==0
        // Start the environment at the beginning of the day
        env->rng_state = 0;
        env->elapsed_time_env = 0.0f;
        env->current_day_time_index = 0;
        env->previous_day_time_index = NUM_BACKGROUND_TRANSITIONS;
    } else if (env->reset_count == 0) {
        // Randomize elapsed time within the day's total duration
        float total_day_duration = BACKGROUND_TRANSITION_TIMES[NUM_BACKGROUND_TRANSITIONS - 1];
        env->elapsed_time_env = ((float)xorshift32(&env->rng_state) / (float)UINT32_MAX) * total_day_duration;

        // Determine the current time index
        env->current_day_time_index = 0;
        for (int i = 0; i < NUM_BACKGROUND_TRANSITIONS - 1; i++) {
            if (env->elapsed_time_env >= env->day_transition_times[i] &&
                env->elapsed_time_env < env->day_transition_times[i + 1]) {
                env->current_day_time_index = i;
                break;
            }
        }

        // Handle the last interval
        if (env->elapsed_time_env >= BACKGROUND_TRANSITION_TIMES[NUM_BACKGROUND_TRANSITIONS - 1]) {
            env->current_day_time_index = NUM_BACKGROUND_TRANSITIONS - 1;
        }
    }

    env->num_enemies = 0;
    for (int i = 0; i < MAX_ENEMIES; i++) {
        env->enemy_cars[i].lane = -1; // Default invalid lane
        env->enemy_cars[i].y = 0.0f;
        env->enemy_cars[i].passed = 0;
    }

    env->score = 0;
    env->speed = MIN_SPEED;

    memcpy(env->day_transition_times, BACKGROUND_TRANSITION_TIMES, sizeof(BACKGROUND_TRANSITION_TIMES));
    
    env->step_count = 0;
    env->collision_cooldown_car_vs_car = 0.0f;
    env->collision_cooldown_car_vs_road = 0.0f;
    env->elapsed_time_env = 0.0f;
    env->enemy_spawn_timer = 0.0f;
    env->enemy_spawn_interval = 0.8777f;
    env->last_spawned_lane = -1;
    env->base_vanishing_point_x = VANISH_START_POINT;
    env->current_vanishing_point_x = VANISH_START_POINT;
    env->target_vanishing_point_x = VANISH_START_POINT;
    env->vanishing_point_x = VANISH_START_POINT;
    env->day = 1;
    env->drift_direction = 0; // Means in noop, but only if crashed state
    env->crashed_penalty = 0.0f;
    env->car_passed_no_crash_active = 0;
    env->step_rew_car_passed_no_crash = 0.0f;
    env->current_curve_stage = 0; // 0: straight
    env->steps_in_current_stage = 0;
    env->current_curve_direction = CURVE_STRAIGHT;
    env->current_curve_factor = 0.0f;
    env->target_curve_factor = 0.0f;
    env->current_step_threshold = 0.0f;
    env->wiggle_y = VANISHING_POINT_Y;
    env->wiggle_speed = WIGGLE_SPEED;
    env->wiggle_length = WIGGLE_LENGTH;
    env->wiggle_amplitude = WIGGLE_AMPLITUDE;
    env->wiggle_active = true;

    // Randomize the initial time of day for each environment
    if (env->rng_state == 0) {
        env->elapsed_time_env = 0;
        env->current_day_time_index = 0;
        env->day_time_index = 0;
        env->previous_day_time_index = 0;
    } else {
        float total_day_duration = BACKGROUND_TRANSITION_TIMES[15];
        env->elapsed_time_env = ((float)rand() / (float)RAND_MAX) * total_day_duration;
        env->current_day_time_index = 0;
        env->day_time_index = 0;
        env->previous_day_time_index = 0;

        // Advance current_day_time_index to match randomized elapsed_time_env
        for (int i = 0; i < NUM_BACKGROUND_TRANSITIONS; i++) {
            if (env->elapsed_time_env >= env->day_transition_times[i]) {
                env->current_day_time_index = i;
            } else {
                break;
            }
        }

        env->previous_day_time_index = (env->current_day_time_index > 0) ? env->current_day_time_index - 1 : NUM_BACKGROUND_TRANSITIONS - 1;
    }

    env->player_x               = INITIAL_PLAYER_X;
    env->player_y               = PLAYER_MAX_Y;
    env->speed                  = MIN_SPEED;
    env->cars_to_pass           = INITIAL_CARS_TO_PASS;
    env->current_curve_direction= CURVE_STRAIGHT;
    float gearTimings[4]        = {4.0f, 2.5f, 3.25f, 1.5f};

    float totalSpeedRange = MAX_SPEED - MIN_SPEED;
    float totalTime = 0.0f;
    for (int i = 0; i < 4; i++) {
        totalTime += gearTimings[i];
    }
    float cumulativeSpeed = MIN_SPEED;
    for (int i = 0; i < 4; i++) {
        float gearTime           = gearTimings[i];
        float gearSpeedIncrement = totalSpeedRange * (gearTime / totalTime);
        env->gear_speed_thresholds[i]   = cumulativeSpeed + gearSpeedIncrement;
        env->gear_acceleration_thresholds[i] = gearSpeedIncrement / (gearTime * TARGET_FPS);
        cumulativeSpeed               = env->gear_speed_thresholds[i];
    }

    // Randomize the initial time of day for each environment
    if (env->rng_state == 0) {
        env->elapsed_time_env = 0;
        env->current_day_time_index = 0;
        env->day_time_index = 0;
        env->previous_day_time_index = 0;
    } else {
        float total_day_duration = BACKGROUND_TRANSITION_TIMES[15];
        env->elapsed_time_env = ((float)rand() / (float)RAND_MAX) * total_day_duration;
        env->current_day_time_index = 0;
        env->day_time_index = 0;
        env->previous_day_time_index = 0;

        // Advance current_day_time_index to match randomized elapsed_time_env
        for (int i = 0; i < NUM_BACKGROUND_TRANSITIONS; i++) {
            if (env->elapsed_time_env >= env->day_transition_times[i]) {
                env->current_day_time_index = i;
            } else {
                break;
            }
        }
        env->previous_day_time_index = (env->current_day_time_index > 0) ? env->current_day_time_index - 1 : NUM_BACKGROUND_TRANSITIONS - 1;
    }
    env->terminals[0] = 0;
    env->truncateds[0] = 0;
    env->rewards[0] = 0.0f;
    env->log.cars_to_pass = INITIAL_CARS_TO_PASS;
}

void reset(Enduro* env) {
    // No random after first reset
    int reset_seed = (env->reset_count == 0) ? xorshift32(&env->rng_state) : 0;
    init(env, reset_seed, env->index);
    env->reset_count += 1;
}

// Quadratic bezier curve helper function
// B(t) = (1−t)^2 * P0 + 2(1−t)*t * P1 + t^2 * P2,  t∈[0,1]
float quadratic_bezier(float bottom_x, float control_x, float top_x, float t) {
    float one_minus_t = 1.0f - t;
    return one_minus_t*one_minus_t*bottom_x 
         + 2.0f*one_minus_t*t*control_x 
         + t*t*top_x;
}

float road_edge_x(const Enduro* env, float y, float offset, unsigned char left) {
    float t = (PLAYABLE_AREA_BOTTOM - y) / (MAX_DISTANCE);
    float base_offset = left ? -ROAD_LEFT_OFFSET : ROAD_RIGHT_OFFSET;
    float bottom_x = env->base_vanishing_point_x + base_offset + offset;
    float top_x    = env->current_vanishing_point_x + offset;
    float edge_x;
    if (fabsf(env->current_curve_factor) < 0.01f) {
        // Straight road interpolation
        edge_x = bottom_x + t * (top_x - bottom_x);
    } else {
        // Slight curve
        float curve_offset = (env->current_curve_factor > 0 ? -30.0f : 30.0f) * fabsf(env->current_curve_factor);
        float control_x    = bottom_x + (top_x - bottom_x) * 0.5f + curve_offset;
        edge_x = quadratic_bezier(bottom_x, control_x, top_x, t);
    }
    // Wiggle effect
    float wiggle_offset = 0.0f;
    if (env->wiggle_active && y >= env->wiggle_y && y <= env->wiggle_y + env->wiggle_length) {
        float t_wiggle = (y - env->wiggle_y) / env->wiggle_length;
        if (t_wiggle < 0.15f) {
            wiggle_offset = env->wiggle_amplitude * (t_wiggle / 0.15f);
        } else if (t_wiggle < 0.87f) {
            wiggle_offset = env->wiggle_amplitude;
        } else {
            wiggle_offset = env->wiggle_amplitude * ((1.0f - t_wiggle) / 0.13f);
        }
        wiggle_offset *= (left ? 1.0f : -1.0f);
        float depth = (y - VANISHING_POINT_Y) / (MAX_DISTANCE);
        float scale = 0.03f + (depth * depth);
        if (scale > 0.3f) {
            scale = 0.3f;
        }
        wiggle_offset *= scale;
    }
    edge_x += wiggle_offset;
    return edge_x;
}

float car_x_in_lane(const Enduro* env, int lane, float y) {
    float offset    = 0.0f;
    float left_edge = road_edge_x(env, y, offset, true);
    float right_edge= road_edge_x(env, y, offset, false);
    float lane_width= (right_edge - left_edge) / NUM_LANES;
    return left_edge + lane_width * ((float)lane + 0.5f);
}

unsigned char check_collision(const Enduro* env, const Car* car) {
    float depth = (car->y - VANISHING_POINT_Y) / (MAX_DISTANCE);
    float scale = fmaxf(0.1f, 0.9f * depth);
    float car_width  = CAR_WIDTH  * scale;
    float car_height = CAR_HEIGHT * scale;
    float car_center_x = car_x_in_lane(env, car->lane, car->y);
    float car_x        = car_center_x - car_width / 2.0f;
    return !(env->player_x > car_x + car_width
          || env->player_x + CAR_WIDTH < car_x
          || env->player_y > car->y + car_height
          || env->player_y + CAR_HEIGHT < car->y);
}

int get_player_lane(Enduro* env) {
    float player_center_x = env->player_x + CAR_WIDTH / 2.0f;
    float offset = (env->player_x - INITIAL_PLAYER_X) * 0.5f;
    float left_edge  = road_edge_x(env, env->player_y, offset, true);
    float right_edge = road_edge_x(env, env->player_y, offset, false);
    float lane_width = (right_edge - left_edge) / 3.0f;
    env->lane = (int)((player_center_x - left_edge) / lane_width);
    if (env->lane < 0)   env->lane = 0;
    if (env->lane > 2)   env->lane = 2;
    return env->lane;
}

float get_car_scale(float y) {
    float depth = (y - VANISHING_POINT_Y) / (MAX_DISTANCE);
    return fmaxf(0.1f, 0.9f * depth);
}

static void computeNearestCarInfo(const Enduro* env, 
                                  float nearest_car_distance[NUM_LANES],
                                  bool  is_lane_empty[NUM_LANES]) {
    for (int l = 0; l < NUM_LANES; l++) {
        nearest_car_distance[l] = MAX_DISTANCE;
        is_lane_empty[l]        = true;
    }
    for (int i = 0; i < env->num_enemies; i++) {
        const Car* car = &env->enemy_cars[i];
        if (car->lane >= 0 && car->lane < NUM_LANES && car->y < env->player_y) {
            float distance = env->player_y - car->y;
            if (distance < nearest_car_distance[car->lane]) {
                nearest_car_distance[car->lane] = distance;
                is_lane_empty[car->lane]        = false;
            }
        }
    }
}

void add_enemy_car(Enduro* env) {
    if (env->num_enemies >= MAX_ENEMIES) {
        return;
    }
    int player_lane = get_player_lane(env);
    int possible_lanes[NUM_LANES];
    int num_possible_lanes = 0;
    int furthest_lane;
    if (player_lane == 0)       furthest_lane = 2;
    else if (player_lane == 2)  furthest_lane = 0;
    else {
        float player_center_x = env->player_x + CAR_WIDTH / 2.0f;
        float road_center_x   = (road_edge_x(env, env->player_y, 0, true) 
                               + road_edge_x(env, env->player_y, 0, false)) / 2.0f;
        furthest_lane = (player_center_x < road_center_x) ? 2 : 0;
    }
    if (env->speed <= 0.0f) {
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
    int lane = possible_lanes[rand() % num_possible_lanes]; 
    if (rand() % 100 < 60 && env->last_spawned_lane != -1) {
        lane = env->last_spawned_lane;
    }
    env->last_spawned_lane = lane;
    Car car = {
        .lane       = lane,
        .x          = car_x_in_lane(env, lane, VANISHING_POINT_Y),
        .y          = (env->speed > 0.0f) ? VANISHING_POINT_Y + 10.0f : SCREEN_HEIGHT,
        .last_x     = car_x_in_lane(env, lane, VANISHING_POINT_Y),
        .last_y     = VANISHING_POINT_Y,
        .passed     = false,
        .color_index = rand() % 6
    };
    float depth = (car.y - VANISHING_POINT_Y) / (MAX_DISTANCE);
    float scale = fmaxf(0.1f, 0.9f * depth + 0.1f);
    float scaled_car_length = CAR_HEIGHT * scale;
    float dynamic_spacing_factor = ((float)rand() / (float)RAND_MAX) * 6.0f + 0.5f;
    float min_spacing = dynamic_spacing_factor * scaled_car_length;
    for (int i = 0; i < env->num_enemies; i++) {
        const Car* existing_car = &env->enemy_cars[i];
        if (existing_car->lane != car.lane) {
            continue;
        }
        float y_distance = (float)fabs(existing_car->y - car.y);
        if (y_distance < min_spacing) {
            return; // Too close, do not spawn
        }
    }
    float min_vertical_range = 6.0f * CAR_HEIGHT;
    int lanes_occupied = 0;
    unsigned char lane_occupied[NUM_LANES] = { false };
    for (int i = 0; i < env->num_enemies; i++) {
        const Car* existing_car = &env->enemy_cars[i];
        float y_distance = fabsf(existing_car->y - car.y);
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
    env->enemy_cars[env->num_enemies++] = car;
}

void update_time_of_day(Enduro* env) {
    float elapsed_time = env->elapsed_time_env;
    float total_duration = env->day_transition_times[15];

    if (elapsed_time >= total_duration) {
        elapsed_time -= total_duration;
        env->elapsed_time_env = elapsed_time; // Reset elapsed time
        env->day_time_index = 0;
    }

    env->previous_day_time_index = env->current_day_time_index;

    while (env->day_time_index < 15 &&
           elapsed_time >= env->day_transition_times[env->day_time_index]) {
        env->day_time_index++;
    }
    env->current_day_time_index = env->day_time_index % 16;
}


void clamp_speed(Enduro* env) {
    if (env->speed < MIN_SPEED || env->speed > MAX_SPEED) {
        env->speed = fmaxf(MIN_SPEED, fminf(env->speed, MAX_SPEED));
    }
}

void clamp_gear(Enduro* env) {
    if (env->current_gear < 0 || env->current_gear > 3) {
        env->current_gear = 0;
    }
}

void accelerate(Enduro* env) {
    clamp_speed(env);
    clamp_gear(env);
    if (env->speed < MAX_SPEED) {
        if (env->speed >= env->gear_speed_thresholds[env->current_gear] && env->current_gear < 3) {
            env->current_gear++;
        }
        float accel      = env->gear_acceleration_thresholds[env->current_gear];
        float multiplier = (env->current_gear == 0) ? 4.0f : 2.0f;
        env->speed      += accel * multiplier;
        clamp_speed(env);
        if (env->speed > env->gear_speed_thresholds[env->current_gear]) {
            env->speed = env->gear_speed_thresholds[env->current_gear];
        }
    }
    clamp_speed(env);
}

void update_road_curve(Enduro* env) {
    int* current_curve_stage     = &env->current_curve_stage;
    int* steps_in_current_stage  = &env->steps_in_current_stage;
    float speed_scale            = 0.5f + ((fabsf(env->speed) / MAX_SPEED) * (MAX_SPEED - MIN_SPEED));
    float vanishing_point_transition_speed = VANISHING_POINT_TRANSITION_SPEED + speed_scale;
    int step_thresholds[3];
    int curve_directions[3];
    int last_direction = 0;
    for (int i = 0; i < 3; i++) {
        step_thresholds[i]   = 1500 + rand() % 3801;
        int direction_choices[] = {-1, 0, 1};
        int next_direction;
        do {
            next_direction = direction_choices[rand() % 3];
        } while ((last_direction == -1 && next_direction == 1) 
              || (last_direction == 1 && next_direction == -1));
        curve_directions[i] = next_direction;
        last_direction      = next_direction;
    }
    env->current_step_threshold = (float)step_thresholds[*current_curve_stage % 3];
    (*steps_in_current_stage)++;
    if (*steps_in_current_stage >= step_thresholds[*current_curve_stage % 3]) {
        env->target_curve_factor = (float)curve_directions[*current_curve_stage % 3];
        *steps_in_current_stage  = 0;
        *current_curve_stage     = (*current_curve_stage + 1) % 3;
    }
    size_t step_thresholds_size   = sizeof(step_thresholds) / sizeof(step_thresholds[0]);
    size_t curve_directions_size  = sizeof(curve_directions) / sizeof(curve_directions[0]);
    size_t max_size = (step_thresholds_size > curve_directions_size) 
                      ? step_thresholds_size : curve_directions_size;
    int adjusted_step_thresholds[max_size];
    int adjusted_curve_directions[max_size];
    for (size_t i = 0; i < max_size; i++) {
        adjusted_step_thresholds[i]   = step_thresholds[i % step_thresholds_size];
        adjusted_curve_directions[i]  = curve_directions[i % curve_directions_size];
    }
    env->current_step_threshold = (float)adjusted_step_thresholds[*current_curve_stage % max_size];
    (*steps_in_current_stage)++;
    if (*steps_in_current_stage >= adjusted_step_thresholds[*current_curve_stage]) {
        env->target_curve_factor = (float)adjusted_curve_directions[*current_curve_stage % max_size];
        *steps_in_current_stage  = 0;
        *current_curve_stage     = (*current_curve_stage + 1) % max_size;
    }
    if (env->current_curve_factor < env->target_curve_factor) {
        env->current_curve_factor = fminf(env->current_curve_factor + CURVE_TRANSITION_SPEED, env->target_curve_factor);
    } else if (env->current_curve_factor > env->target_curve_factor) {
        env->current_curve_factor = fmaxf(env->current_curve_factor - CURVE_TRANSITION_SPEED, env->target_curve_factor);
    }
    if (fabsf(env->current_curve_factor) < 0.1f) {
        env->current_curve_direction = CURVE_STRAIGHT;
    } else {
        env->current_curve_direction = (env->current_curve_factor > 0) ? CURVE_RIGHT : CURVE_LEFT;
    }
    env->base_target_vanishing_point_x = VANISHING_POINT_X_LEFT - env->t_p * (VANISHING_POINT_X_LEFT - VANISHING_POINT_X_RIGHT);
    float target_shift = (float)env->current_curve_direction * CURVE_VANISHING_POINT_SHIFT;
    env->target_vanishing_point_x = env->base_target_vanishing_point_x + target_shift;
    if (env->current_vanishing_point_x < env->target_vanishing_point_x) {
        env->current_vanishing_point_x = fminf(env->current_vanishing_point_x + vanishing_point_transition_speed,
                                               env->target_vanishing_point_x);
    } else if (env->current_vanishing_point_x > env->target_vanishing_point_x) {
        env->current_vanishing_point_x = fmaxf(env->current_vanishing_point_x - vanishing_point_transition_speed,
                                               env->target_vanishing_point_x);
    }
    env->vanishing_point_x = env->current_vanishing_point_x;
}

void compute_observations(Enduro* env) {
    float* obs = env->observations;
    int obs_index = 0;
    float player_x_norm = (env->player_x - env->last_road_left) / 
                          (env->last_road_right - env->last_road_left);
    float player_y_norm = (PLAYER_MAX_Y - env->player_y) /
                          (PLAYER_MAX_Y - PLAYER_MIN_Y);
    // Player position and speed
    obs[obs_index++] = player_x_norm;
    obs[obs_index++] = player_y_norm;
    obs[obs_index++] = (env->speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED);
    float road_left  = road_edge_x(env, env->player_y, 0, true);
    float road_right = road_edge_x(env, env->player_y, 0, false) - CAR_WIDTH;
    // Road edges and last road edges
    obs[obs_index++] = (road_left  - PLAYABLE_AREA_LEFT) / (PLAYABLE_AREA_RIGHT - PLAYABLE_AREA_LEFT);
    obs[obs_index++] = (road_right - PLAYABLE_AREA_LEFT) / (PLAYABLE_AREA_RIGHT - PLAYABLE_AREA_LEFT);
    obs[obs_index++] = (env->last_road_left  - PLAYABLE_AREA_LEFT) / (PLAYABLE_AREA_RIGHT - PLAYABLE_AREA_LEFT);
    obs[obs_index++] = (env->last_road_right - PLAYABLE_AREA_LEFT) / (PLAYABLE_AREA_RIGHT - PLAYABLE_AREA_LEFT);
    // Player lane number
    obs[obs_index++] = (float)get_player_lane(env) / (NUM_LANES - 1);
    // Enemy cars => 5 floats per car
    for (int i = 0; i < MAX_ENEMIES; i++) {
        const Car* car = &env->enemy_cars[i];
        if (car->y > VANISHING_POINT_Y && car->y < PLAYABLE_AREA_BOTTOM) {
            float buffer_x = CAR_WIDTH * 0.5f;
            float buffer_y = CAR_HEIGHT * 0.5f;
            float car_x_norm = ((car->x - buffer_x) - env->last_road_left) / 
                               (env->last_road_right - env->last_road_left);
            car_x_norm = fmaxf(0.0f, fminf(1.0f, car_x_norm));
            float car_y_norm = (PLAYABLE_AREA_BOTTOM - (car->y - buffer_y)) /
                               (MAX_DISTANCE);
            car_y_norm = fmaxf(0.0f, fminf(1.0f, car_y_norm));
            float delta_x_norm = (car->last_x - car->x) /
                                 (env->last_road_right - env->last_road_left);
            float delta_y_norm = (car->last_y - car->y) /
                                 (MAX_DISTANCE);
            int is_same_lane = (car->lane == env->lane);
            obs[obs_index++] = car_x_norm;
            obs[obs_index++] = car_y_norm;
            obs[obs_index++] = delta_x_norm;
            obs[obs_index++] = delta_y_norm;
            obs[obs_index++] = (float)is_same_lane;
        } else {
            // Default
            obs[obs_index++] = 0.5f;
            obs[obs_index++] = 0.5f;
            obs[obs_index++] = 0.0f;
            obs[obs_index++] = 0.0f;
            obs[obs_index++] = 0.0f;
        }
    }
    // Curve direction
    obs[obs_index++] = (float)(env->current_curve_direction + 1) / 2.0f;
    // Observation for player's drift due to road curvature
    float drift_magnitude = env->current_curve_factor * CURVE_PLAYER_SHIFT_FACTOR * (float)fabs(env->speed);
    float drift_direction = (env->current_curve_factor > 0) ? 1.0f : -1.0f;
    float max_drift_magnitude = CURVE_PLAYER_SHIFT_FACTOR * MAX_SPEED;
    float normalized_drift_magnitude = (float)fabs(drift_magnitude) / max_drift_magnitude;
    obs[obs_index++] = drift_direction;
    obs[obs_index++] = normalized_drift_magnitude;
    obs[obs_index++] = env->current_curve_factor;
    // Time of day
    float total_day_length = BACKGROUND_TRANSITION_TIMES[15];
    obs[obs_index++] = fmodf(env->elapsed_time_env, total_day_length) / total_day_length;
    // Cars to pass
    obs[obs_index++] = (float)env->cars_to_pass / (float)INITIAL_CARS_TO_PASS;
    // Nearest enemy car distances in each lane
    float nearest_car_distance[NUM_LANES];
    bool is_lane_empty[NUM_LANES];
    computeNearestCarInfo(env, nearest_car_distance, is_lane_empty);
    for (int l = 0; l < NUM_LANES; l++) {
        float normalized_distance = is_lane_empty[l] 
                                    ? 1.0f 
                                    : nearest_car_distance[l] / MAX_DISTANCE;
        obs[obs_index++] = normalized_distance;
    }
}

static float clamp_spawn_interval(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

float calculate_enemy_spawn_interval(const Enduro* env) {
    float max_spawn_interval;
    int day_index = env->day - 1;
    if (day_index == 0) {
        max_spawn_interval = MAX_SPAWN_INTERVALS[0];
    } else {
        float base_interval = MAX_SPAWN_INTERVALS[NUM_MAX_SPAWN_INTERVALS - 1];
        float reduction     = (float)(day_index - NUM_MAX_SPAWN_INTERVALS + 1) * DAILY_INTERVAL_REDUCTION;
        max_spawn_interval  = clamp_spawn_interval(base_interval - reduction, MIN_POSSIBLE_INTERVAL, base_interval);
    }
    max_spawn_interval = fmaxf(max_spawn_interval, MIN_SPAWN_INTERVAL);
    float speed_range = MAX_SPEED - MIN_SPEED;
    float speed_factor= speed_range > 0.0f 
                        ? (env->speed - MIN_SPEED) / speed_range 
                        : 0.0f;
    speed_factor = clamp_spawn_interval(speed_factor, 0.0f, 1.0f);
    float interval_range = max_spawn_interval - MIN_SPAWN_INTERVAL;
    return MIN_SPAWN_INTERVAL + (1.0f - speed_factor) * interval_range * SPAWN_SCALING_FACTOR;
}

void c_step(Enduro* env) {
    env->rewards[0] = 0.0f;
    env->elapsed_time_env += (1.0f / TARGET_FPS);
    update_time_of_day(env);
    update_road_curve(env);
    env->log.episode_length += 1;
    env->terminals[0] = 0;
    env->road_scroll_offset += env->speed;
    // Update enemy car positions
    for (int i = 0; i < env->num_enemies; i++) {
        Car* car = &env->enemy_cars[i];
        float scale = get_car_scale(car->y);
        float movement_speed = env->speed * scale * 0.75f;
        car->y += movement_speed;
    }
    float road_left  = road_edge_x(env, env->player_y, 0, true);
    float road_right = road_edge_x(env, env->player_y, 0, false) - CAR_WIDTH;
    env->last_road_left  = road_left;
    env->last_road_right = road_right;
    unsigned char is_snow_stage = (env->current_day_time_index == 3);
    float movement_amount = 0.5f;
    // Player movement logic
    if (env->collision_cooldown_car_vs_car <= 0 && env->collision_cooldown_car_vs_road <= 0) {
        env->crashed_penalty = 0.0f;
        int act = env->actions[0];
        movement_amount = (((env->speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)) + 0.3f);
        if (is_snow_stage) {
            movement_amount *= 0.6f;
        }
        switch (act) {
            default:
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
                if (env->speed > MIN_SPEED) env->speed -= DECELERATION_RATE;
                break;
            case ACTION_DOWNRIGHT:
                if (env->speed > MIN_SPEED) env->speed -= DECELERATION_RATE;
                env->player_x += movement_amount;
                if (env->player_x > road_right) env->player_x = road_right;
                break;
            case ACTION_DOWNLEFT:
                if (env->speed > MIN_SPEED) env->speed -= DECELERATION_RATE;
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
        // Crash cooldown
        if (env->collision_cooldown_car_vs_car > 0) {
            env->collision_cooldown_car_vs_car -= 1;
            env->crashed_penalty = -0.01f;
        }
        if (env->collision_cooldown_car_vs_road > 0) {
            env->collision_cooldown_car_vs_road -= 1;
            env->crashed_penalty = -0.01f;
        }
        // Drift if crashed
        if (env->drift_direction == 0) {
            env->drift_direction = 
                (env->player_x > (road_left + road_right) / 2) ? -1 : 1;
            // Remove enemy cars in middle lane + lane player is drifting towards 
            // behind the player
            for (int i = 0; i < env->num_enemies; i++) {
                const Car* car = &env->enemy_cars[i];
                if ((car->lane == 1 || car->lane == env->lane + env->drift_direction) 
                     && (car->y > env->player_y)) 
                {
                    remove_enemy_car(env, i);
                    i--;
                }
            }
        }
        if (env->collision_cooldown_car_vs_road > 0) {
            env->player_x += (float)env->drift_direction * 0.12f;
        } else {
            env->player_x += (float)env->drift_direction * 0.25f;
        }
    }
    env->lane = get_player_lane(env);
    float nearest_car_distance[NUM_LANES];
    bool is_lane_empty[NUM_LANES];
    computeNearestCarInfo(env, nearest_car_distance, is_lane_empty);
    float reward_amount = 0.05f;
    if (is_lane_empty[env->lane]) {
        env->rewards[0] += reward_amount;
    }
    // Road curve shift
    float curve_shift = -env->current_curve_factor 
                        * CURVE_PLAYER_SHIFT_FACTOR 
                        * (float)fabs(env->speed);
    env->player_x += curve_shift;
    if (env->player_x < road_left)  env->player_x = road_left;
    if (env->player_x > road_right) env->player_x = road_right;

    float t_p = (env->player_x - PLAYER_MIN_X) / (PLAYER_MAX_X - PLAYER_MIN_X);
    t_p = fmaxf(0.0f, fminf(1.0f, t_p));
    env->t_p = t_p;
    env->base_vanishing_point_x = 
        VANISHING_POINT_X_LEFT 
        - t_p * (VANISHING_POINT_X_LEFT - VANISHING_POINT_X_RIGHT);
    float curve_vanishing_point_shift = (float)env->current_curve_direction 
                                        * CURVE_VANISHING_POINT_SHIFT;
    env->vanishing_point_x = env->base_vanishing_point_x 
                             + curve_vanishing_point_shift;
    // Wiggle logic
    if (env->wiggle_active) {
        float min_wiggle_period = 5.8f; 
        float max_wiggle_period = 0.3f;
        float speed_normalized  = (env->speed - MIN_SPEED) 
                                  / (MAX_SPEED - MIN_SPEED);
        speed_normalized = fmaxf(0.0f, fminf(1.0f, speed_normalized));
        float current_wiggle_period = min_wiggle_period 
             - powf(speed_normalized, 0.25f) * (min_wiggle_period - max_wiggle_period);
        env->wiggle_speed = (MAX_DISTANCE) 
                            / (current_wiggle_period * TARGET_FPS);
        env->wiggle_y += env->wiggle_speed;
        if (env->wiggle_y > PLAYABLE_AREA_BOTTOM) {
            env->wiggle_y = VANISHING_POINT_Y;
        }
    }
    // Player y based on speed
    env->player_y = PLAYER_MAX_Y - (env->speed - MIN_SPEED) 
                    / (MAX_SPEED - MIN_SPEED) 
                    * (PLAYER_MAX_Y - PLAYER_MIN_Y);

    if (env->player_y < PLAYER_MIN_Y) env->player_y = PLAYER_MIN_Y;
    if (env->player_y > PLAYER_MAX_Y) env->player_y = PLAYER_MAX_Y;
    // Collision with road edges
    if (env->player_x <= road_left || env->player_x >= road_right) {
        env->log.collisions_player_vs_road++;
        env->rewards[0] -= 0.5f;
        env->speed = fmaxf((env->speed - 1.25f), MIN_SPEED);
        env->collision_cooldown_car_vs_road = CRASH_NOOP_DURATION_CAR_VS_ROAD;
        env->drift_direction = 0;
        env->player_x = fmaxf(road_left + 1, fminf(road_right - 1, env->player_x));
    }
    // Enemy logic
    for (int i = 0; i < env->num_enemies; i++) {
        Car* car = &env->enemy_cars[i];
        if (car->y > SCREEN_HEIGHT * 2) {
            remove_enemy_car(env, i);
            i--;
            continue;
        }
        if (env->speed < 0 && car->y <= LOGICAL_VANISHING_Y) {
            remove_enemy_car(env, i);
            i--;
            continue;
        }
        if (env->speed <= 0 && car->y >= env->player_y + CAR_HEIGHT * 2.0f) {
            int furthest_lane = get_furthest_lane(env);
            car->lane = furthest_lane;
            continue;
        }
        // Passing logic
        if (env->speed > 0 && car->last_y < env->player_y + CAR_HEIGHT
            && car->y >= env->player_y + CAR_HEIGHT
            && env->collision_cooldown_car_vs_car <= 0
            && env->collision_cooldown_car_vs_road <= 0) 
        {
            if (env->cars_to_pass > 0) {
                env->cars_to_pass -= 1;
            }
            if (!car->passed) {
                env->log.passed_cars += 1;
                env->rewards[0] += 1.0f;
                env->car_passed_no_crash_active = 1;
                env->step_rew_car_passed_no_crash += 0.001f;
            }
            car->passed = true;
        } else if (env->speed < 0 && car->last_y > env->player_y && car->y <= env->player_y) {
            int maxCarsToPass = (env->day == 1) ? 200 : 300;
            if (env->cars_to_pass == maxCarsToPass) {
                env->log.passed_by_enemy += 1.0f;
            } else {
                env->cars_to_pass += 1;
                env->log.passed_by_enemy += 1.0f;
                env->rewards[0] -= 0.1f;
            }
        }
        car->last_y = car->y; // Preserve for compute_observations
        car->last_x = car->x;
        // Collision with player
        if (env->collision_cooldown_car_vs_car <= 0 && check_collision(env, car)) {
            env->log.collisions_player_vs_car++;
            env->rewards[0] -= 0.5f;
            env->speed = 1 + MIN_SPEED;
            env->collision_cooldown_car_vs_car = CRASH_NOOP_DURATION_CAR_VS_CAR;
            env->drift_direction = 0;
            env->car_passed_no_crash_active = 0;
            env->step_rew_car_passed_no_crash = 0.0f;
        }
    }
    // Spawn logic
    env->enemy_spawn_interval = calculate_enemy_spawn_interval(env);
    env->enemy_spawn_timer   += (1.0f / TARGET_FPS);
    if (env->enemy_spawn_timer >= env->enemy_spawn_interval) {
        env->enemy_spawn_timer -= env->enemy_spawn_interval;
        if (env->num_enemies < MAX_ENEMIES) {
            float clump_probability = fminf((env->speed - MIN_SPEED)
                / (MAX_SPEED - MIN_SPEED), 1.0f);
            int num_to_spawn = 1;
            if (((float)rand() / (float)RAND_MAX) < clump_probability) {
                num_to_spawn = 1 + rand() % 2; // up to 3 cars
            }
            int occupied_lanes[NUM_LANES] = {0};
            for (int i = 0; i < num_to_spawn && env->num_enemies < MAX_ENEMIES; i++) {
                int lane;
                do {
                    lane = rand() % NUM_LANES;
                } while (occupied_lanes[lane]);
                occupied_lanes[lane] = 1;
                int previous_num_enemies = env->num_enemies;
                add_enemy_car(env);
                if (env->num_enemies > previous_num_enemies) {
                    Car* new_car = &env->enemy_cars[env->num_enemies - 1];
                    new_car->lane = lane;
                    new_car->y -= (float)i * (CAR_HEIGHT * 3);
                }
            }
        }
    }        
    // Day completed logic
    if (env->cars_to_pass <= 0 && !env->day_completed) {
        env->day_completed = true;
    }
    // Handle day transition when background cycles back to 0
    if (env->current_day_time_index == 0 && env->previous_day_time_index == 15) {
        // Background cycled back to 0
        if (env->day_completed) {
            env->log.days_completed += 1;
            env->day += 1;
            env->rewards[0] += 1.0f;
            env->cars_to_pass = 300; // Always 300 after the first day
            env->day_completed = false;
            add_log(env->log_buffer, &env->log);                    
        } else {
            // Player failed to pass required cars, soft-reset environment
            env->log.days_failed += 1.0f;
            env->terminals[0] = true;
            add_log(env->log_buffer, &env->log);
            compute_observations(env); // Call compute_observations before reset to log
            reset(env);
            return;
        }
    }
    // Reward each step after a car is passed until a collision occurs. 
    // Then, no rewards per step until next car is passed.
    if (env->car_passed_no_crash_active) {
        env->rewards[0] += env->step_rew_car_passed_no_crash;
    }
    env->rewards[0] += env->crashed_penalty;
    env->log.crashed_penalty = env->crashed_penalty;
    env->log.step_rew_car_passed_no_crash = env->step_rew_car_passed_no_crash;
    env->log.reward = env->rewards[0];
    env->log.episode_return = env->rewards[0];
    env->step_count++;
    float normalized_speed = (env->speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED);
    float score_increment = normalized_speed * 0.015f + 0.01f;
    env->score += score_increment;
    env->log.score = env->score;
    env->log.cars_to_pass = env->cars_to_pass;
    compute_observations(env);
}

void init_raylib(GameState* gameState) {
    InitWindow(SCREEN_WIDTH * 2, SCREEN_HEIGHT * 2, "puffer_enduro");
    SetTargetFPS(60);
    gameState->render_target = LoadRenderTexture(SCREEN_WIDTH, SCREEN_HEIGHT);
}

void load_textures(GameState* gameState, Enduro* env) {
    gameState->car_animation_timer  = 0.0f;
    gameState->car_animation_interval = 0.05f;
    gameState->show_left_tread      = true;
    gameState->mountain_pos   = 0.0f;
    gameState->show_left_flag       = true;
    gameState->flag_timer          = 0;
    gameState->victory_display_timer= 0;
    gameState->victory_achieved    = false;
    env->score                    = 0;
    gameState->score_timer         = 0;
    env->day                      = 1;
    for (int i = 0; i < SCORE_DIGITS; i++) {
        gameState->score_digit_currents[i]  = 0;
        gameState->score_digit_nexts[i]     = 0;
        gameState->score_digit_offsets[i]   = 0.0f;
        gameState->score_digit_scrolling[i] = false;
    }
    env->elapsed_time_env = 0.0f;
    gameState->spritesheet = LoadTexture("resources/enduro/enduro_spritesheet.png");
    for (int i = 0; i < 16; ++i) {
        gameState->background_indices[i] = i;          // 0..15 => backgrounds
        gameState->mountain_indices[i]   = 16 + i;     // 16..31 => mountains
    }
    for (int i = 0; i < 10; ++i) {
        gameState->digit_indices[i] = 32 + i; 
    }
    gameState->digit_indices[10] = 42; // "CAR"

    for (int i = 0; i < 10; ++i) {
        gameState->green_digit_indices[i] = 43 + i;
    }
    for (int i = 0; i < 10; ++i) {
        gameState->yellow_digit_indices[i] = 53 + i;
    }
    int base_enemy_car_index = 63;
    for (int color = 0; color < 6; ++color) {
        for (int tread = 0; tread < 2; ++tread) {
            gameState->enemy_car_indices[color][tread] = 
                base_enemy_car_index + color*2 + tread;
        }
    }
    gameState->enemy_car_night_fog_taillights_index = 75;
    gameState->enemy_car_night_taillights_index    = 76;
    gameState->player_car_left_tread_index         = 77;
    gameState->player_car_right_tread_index        = 78;
    gameState->level_complete_flag_right_index     = 79;
    gameState->level_complete_flag_left_index      = 80;
    gameState->car_animation_timer               = 0.0f;
    gameState->car_animation_interval            = 0.05f;
    gameState->show_left_tread                   = true;
    gameState->mountain_pos                = 0.0f;
}

void cleanup(const GameState* gameState) {
    UnloadRenderTexture(gameState->render_target);
    UnloadTexture(gameState->spritesheet);
}

GameState* make_client(Enduro* env) {
    GameState* client = (GameState*)malloc(sizeof(GameState));
    init_raylib(client);
    load_textures(client, env);
    return client;
}

void close_client(GameState* client) {
    cleanup(client);
    CloseWindow();
    free(client);
    }

void render_car(const GameState* gameState, const Enduro* env) {
    int car_asset_index = gameState->show_left_tread 
        ? gameState->player_car_left_tread_index 
        : gameState->player_car_right_tread_index;
    Rectangle src_rect = asset_map[car_asset_index];
    Vector2 position = { env->player_x, env->player_y };
    DrawTextureRec(gameState->spritesheet, src_rect, position, WHITE);
}

// Animates the cars' tire treads
void update_car_animation(GameState* gameState, const Enduro* env) {
    float min_interval = 0.005f;
    float max_interval = 0.075f;
    float speed_ratio  = (env->speed - MIN_SPEED) 
                        / (MAX_SPEED - MIN_SPEED);
    gameState->car_animation_interval = 
        max_interval - (max_interval - min_interval) * speed_ratio;
    gameState->car_animation_timer += (float)GetFrameTime();
    if (gameState->car_animation_timer >= gameState->car_animation_interval) {
        gameState->car_animation_timer = 0.0f;
        gameState->show_left_tread = !gameState->show_left_tread;
    }
}

void update_scoreboard(GameState* gameState, Enduro* env) {
    float normalized_speed = fminf(fmaxf(env->speed, 1.0f), 2.0f);
    int frame_interval = (int)(30 / normalized_speed);
    gameState->score_timer++;
    if (gameState->score_timer >= frame_interval) {
        gameState->score_timer = 0;
        env->score += 1;
        if (env->score > 99999) {
            env->score = 0;
        }
        int temp_score = env->score;
        for (int i = SCORE_DIGITS - 1; i >= 0; i--) {
            int new_digit = temp_score % 10;
            temp_score /= 10;
            if (new_digit != gameState->score_digit_currents[i]) {
                gameState->score_digit_nexts[i] = new_digit;
                gameState->score_digit_offsets[i] = 0.0f;
                gameState->score_digit_scrolling[i] = true;
            }
        }
    }
    float scroll_speed = 0.55f * normalized_speed;
    for (int i = 0; i < SCORE_DIGITS; i++) {
        if (gameState->score_digit_scrolling[i]) {
            gameState->score_digit_offsets[i] += scroll_speed;
            if (gameState->score_digit_offsets[i] >= DIGIT_HEIGHT) {
                gameState->score_digit_offsets[i] = 0.0f;
                gameState->score_digit_currents[i] = gameState->score_digit_nexts[i];
                gameState->score_digit_scrolling[i] = false;
            }
        }
    }
}

void render_background(GameState* gameState, Enduro* env) {
    int bg_index = gameState->background_indices[env->current_day_time_index];
    Rectangle src_rect = asset_map[bg_index];
    DrawTextureRec(gameState->spritesheet, src_rect, (Vector2){0, 0}, WHITE);
}

void render_scoreboard(GameState* gameState, Enduro* env) {
    int digit_width  = DIGIT_WIDTH;
    int digit_height = DIGIT_HEIGHT;
    int score_start_x = 56 + digit_width - 8;
    int score_start_y = 173 - digit_height;
    int day_x  = 56 - 8;
    int day_y  = 188 - digit_height;
    int cars_x = 72 - 8;
    int cars_y = 188 - digit_height;
    for (int i = 0; i < SCORE_DIGITS; ++i) { // Draw scrolling score digits
        int digit_x = score_start_x + i * digit_width;
        int current_digit_index = gameState->score_digit_currents[i];
        int next_digit_index    = gameState->score_digit_nexts[i];
        int current_asset_index;
        int next_asset_index;
        if (i == SCORE_DIGITS - 1) {
            current_asset_index = gameState->yellow_digit_indices[current_digit_index];
            next_asset_index    = gameState->yellow_digit_indices[next_digit_index];
        } else {
            current_asset_index = gameState->digit_indices[current_digit_index];
            next_asset_index    = gameState->digit_indices[next_digit_index];
        }
        Rectangle src_rect_current_full = asset_map[current_asset_index];
        Rectangle src_rect_next_full    = asset_map[next_asset_index];
        if (gameState->score_digit_scrolling[i]) {
            float offset = gameState->score_digit_offsets[i];
            Rectangle src_rect_current = src_rect_current_full;
            src_rect_current.height = digit_height - (int)offset;
            Rectangle dest_rect_current = { 
                digit_x, (float)(score_start_y + (int)offset), 
                (float)digit_width, (float)(digit_height - (int)offset) 
            };
            DrawTexturePro(
                gameState->spritesheet,
                src_rect_current,
                dest_rect_current,
                (Vector2){ 0, 0 },
                0.0f,
                WHITE
            );
            Rectangle src_rect_next = src_rect_next_full;
            src_rect_next.y      += (digit_height - (int)offset);
            src_rect_next.height  = (int)offset;
            Rectangle dest_rect_next = {
                digit_x, (float)score_start_y, 
                (float)digit_width, (float)((int)offset)
            };
            DrawTexturePro(
                gameState->spritesheet,
                src_rect_next,
                dest_rect_next,
                (Vector2){ 0, 0 },
                0.0f,
                WHITE
            );
        } else {
            Rectangle src_rect = asset_map[current_asset_index];
            Vector2 position = { (float)digit_x, (float)score_start_y };
            DrawTextureRec(gameState->spritesheet, src_rect, position, WHITE);
        }
    }
    int day = env->day % 10; // Draw day digit
    int day_texture_index = day;
    if (env->day_completed) {
        gameState->victory_achieved = true;
    }
    Rectangle day_src_rect;
    if (gameState->victory_achieved) {
        int asset_index = gameState->green_digit_indices[day_texture_index];
        day_src_rect = asset_map[asset_index];
    } else {
        int asset_index = gameState->digit_indices[day_texture_index];
        day_src_rect = asset_map[asset_index];
    }
    Vector2 day_position = { (float)day_x, (float)day_y };
    DrawTextureRec(gameState->spritesheet, day_src_rect, day_position, WHITE);
    if (gameState->victory_achieved) {
        int flag_asset_index = gameState->show_left_flag 
            ? gameState->level_complete_flag_left_index 
            : gameState->level_complete_flag_right_index;
        Rectangle flag_src_rect = asset_map[flag_asset_index];
        Rectangle dest_rect = {
            (float)cars_x, (float)cars_y, 
            flag_src_rect.width, flag_src_rect.height
        };
        DrawTexturePro(
            gameState->spritesheet,
            flag_src_rect,
            dest_rect,
            (Vector2){0, 0},
            0.0f,
            WHITE
        );
    } else {
        int car_asset_index = gameState->digit_indices[10]; // "CAR"
        Rectangle car_src_rect = asset_map[car_asset_index];
        Vector2 car_position = { (float)cars_x, (float)cars_y };
        DrawTextureRec(gameState->spritesheet, car_src_rect, car_position, WHITE);
        int cars = env->cars_to_pass;
        if (cars < 0) cars = 0;
        for (int i = 1; i < CARS_DIGITS; ++i) {
            int divisor = (int)pow(10, CARS_DIGITS - i - 1);
            int digit = (cars / divisor) % 10;
            if (digit < 0 || digit > 9) digit = 0;
            int digit_x = cars_x + i * (digit_width + 1);
            int asset_index = gameState->digit_indices[digit];
            Rectangle src_rect = asset_map[asset_index];
            Vector2 position = { (float)digit_x, (float)cars_y };
            DrawTextureRec(gameState->spritesheet, src_rect, position, WHITE);
        }
    }
}

void update_victory_effects(GameState* gameState) {
    if (!gameState->victory_achieved) {
        return;
    }
    gameState->flag_timer++;
    if (gameState->flag_timer % 50 == 0) {
        gameState->show_left_flag = !gameState->show_left_flag;
    }
    gameState->victory_display_timer++;
    if (gameState->victory_display_timer >= 10) {
        gameState->victory_display_timer = 0;
    }
}

void update_mountains(GameState* gameState, Enduro* env) {
    float base_speed = 0.0f;
    float curve_strength = fabsf(env->current_curve_factor);
    float speed_multiplier = 1.0f;
    float scroll_speed = base_speed + curve_strength * speed_multiplier;
    int mountain_index = gameState->mountain_indices[0];
    int mountain_width = asset_map[mountain_index].width;
    if (env->current_curve_direction == 1) { 
        gameState->mountain_pos += scroll_speed;
        if (gameState->mountain_pos >= (float)mountain_width) {
            gameState->mountain_pos -= (float)mountain_width;
        }
    } else if (env->current_curve_direction == -1) {
        gameState->mountain_pos -= scroll_speed;
        if (gameState->mountain_pos <= -(float)mountain_width) {
            gameState->mountain_pos += (float)mountain_width;
        }
    }
}

void render_mountains(GameState* gameState, Enduro* env) {
    int mountain_index = gameState->mountain_indices[env->current_day_time_index];
    Rectangle src_rect = asset_map[mountain_index];
    int mountain_width = src_rect.width;
    int mountain_y = 45;
    float player_center_x = (PLAYER_MIN_X + PLAYER_MAX_X) / 2.0f;
    float player_offset  = env->player_x - player_center_x;
    float parallax_factor = 0.5f;
    float adjusted_offset = -player_offset * parallax_factor;
    float mountain_x     = -gameState->mountain_pos + adjusted_offset;
    BeginScissorMode(PLAYABLE_AREA_LEFT, 0, SCREEN_WIDTH - PLAYABLE_AREA_LEFT, SCREEN_HEIGHT);
    for (int x = (int)mountain_x; x < SCREEN_WIDTH; x += mountain_width) {
        DrawTextureRec(gameState->spritesheet, src_rect, (Vector2){ (float)x, (float)mountain_y }, WHITE);
    }
    for (int x = (int)mountain_x - mountain_width; x > -mountain_width; x -= mountain_width) {
        DrawTextureRec(gameState->spritesheet, src_rect, (Vector2){ (float)x, (float)mountain_y }, WHITE);
    }
    EndScissorMode();
}

static bool should_render_car_in_fog(float car_y, bool is_night_fog_stage) {
    return !is_night_fog_stage || car_y >= 92.0f;
}

static int get_car_texture_index(GameState* gameState, bool is_night_stage, int bg_index, Car* car) {
    if (is_night_stage) {
        return (bg_index == 13) 
            ? gameState->enemy_car_night_fog_taillights_index
            : gameState->enemy_car_night_taillights_index;
    }
    int tread_index = gameState->show_left_tread ? 0 : 1;
    return gameState->enemy_car_indices[car->color_index][tread_index];
}

void render_enemy_cars(GameState* gameState, Enduro* env) {
    int bg_index      = env->current_day_time_index;
    bool is_night_stage = (bg_index == 12 || bg_index == 13 || bg_index == 14);
    bool is_night_fog_stage = (bg_index == 13);
    float clip_start_y = is_night_fog_stage ? 92.0f : VANISHING_POINT_Y;
    float clip_height = PLAYABLE_AREA_BOTTOM - clip_start_y;
    Rectangle clip_rect = {
        (float)PLAYABLE_AREA_LEFT, 
        clip_start_y, 
        (float)(PLAYABLE_AREA_RIGHT - PLAYABLE_AREA_LEFT), 
        clip_height
    };
    BeginScissorMode(clip_rect.x, clip_rect.y, clip_rect.width, clip_rect.height);
    for (int i = 0; i < env->num_enemies; i++) {
        Car* car = &env->enemy_cars[i];
        if (!should_render_car_in_fog(car->y, is_night_fog_stage)) {
            continue;
        }
        float car_scale = get_car_scale(car->y);
        int car_asset_index = get_car_texture_index(gameState, is_night_stage, bg_index, car);
        Rectangle src_rect = asset_map[car_asset_index];
        float car_center_x = car_x_in_lane(env, car->lane, car->y);
        float car_x = car_center_x - (src_rect.width * car_scale) / 2.0f;
        float car_y = car->y - (src_rect.height * car_scale) / 2.0f;
        DrawTexturePro(
            gameState->spritesheet,
            src_rect,
            (Rectangle){ car_x, car_y, src_rect.width * car_scale, src_rect.height * car_scale },
            (Vector2){ 0, 0 },
            0.0f,
            WHITE
        );
    }
    EndScissorMode();
}

static Color get_road_color(float y) {
    if (y >= 52 && y < 91) {
        return (Color){74, 74, 74, 255};
    } else if (y >= 91 && y < 106) {
        return (Color){111, 111, 111, 255};
    } else if (y >= 106 && y <= 154) {
        return (Color){170, 170, 170, 255};
    }
    return WHITE;
}

void render_road(GameState* gameState, Enduro* env) {
    int bg_index = env->current_day_time_index;
    bool is_night_fog_stage = (bg_index == 13);
    float road_start_y = is_night_fog_stage ? 92.0f : VANISHING_POINT_Y;
    Vector2 previous_left_point = {0};
    Vector2 previous_right_point= {0};
    bool first_point = true;
    float road_edges[2][PLAYABLE_AREA_BOTTOM + 1];  // [left/right][y]
    for (float y = road_start_y; y <= PLAYABLE_AREA_BOTTOM; y += 0.75f) {
        float adjusted_y = (env->speed < 0) ? y : y + (float)fmod(env->road_scroll_offset, 0.75f);
        if (adjusted_y > PLAYABLE_AREA_BOTTOM) continue;
        road_edges[0][(int)y] = road_edge_x(env, adjusted_y, 0, true);
        road_edges[1][(int)y] = road_edge_x(env, adjusted_y, 0, false);
    }
    for (float y = road_start_y; y <= PLAYABLE_AREA_BOTTOM; y += 0.75f) {
        float adjusted_y = (env->speed < 0) ? y : y + (float)fmod(env->road_scroll_offset, 0.75f);
        if (adjusted_y > PLAYABLE_AREA_BOTTOM) continue;
        Color road_color = get_road_color(adjusted_y);
        Vector2 current_left_point  = { road_edges[0][(int)y], adjusted_y };
        Vector2 current_right_point = { road_edges[1][(int)y], adjusted_y };
        if (!first_point) {
            DrawLineV(previous_left_point,  current_left_point,  road_color);
            DrawLineV(previous_right_point, current_right_point, road_color);
        }
        previous_left_point  = current_left_point;
        previous_right_point = current_right_point;
        first_point = false;
    }
}

void render(GameState* client, Enduro* env) {
    BeginTextureMode(client->render_target);
    ClearBackground(BLACK);
    BeginBlendMode(BLEND_ALPHA);

    render_background(client, env);
    update_car_animation(client, env);
    update_mountains(client, env);
    render_mountains(client, env);
    render_road(client, env);
    render_enemy_cars(client, env);
    render_car(client, env);
    update_victory_effects(client);
    update_scoreboard(client, env);
    render_scoreboard(client, env);

    EndBlendMode();
    EndTextureMode();

    BeginDrawing();
    ClearBackground(BLACK);
    DrawTexturePro(
        client->render_target.texture,
        (Rectangle){ 
            0, 
            0, 
            (float)client->render_target.texture.width, 
            -(float)client->render_target.texture.height 
        },
        (Rectangle){ 
            0, 
            0, 
            (float)SCREEN_WIDTH * 2, 
            (float)SCREEN_HEIGHT * 2 
        },
        (Vector2){ 0, 0 },
        0.0f,
        WHITE
    );
    EndDrawing();
}
