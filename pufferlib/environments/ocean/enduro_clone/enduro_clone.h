// enduro_clone.h

#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include "raylib.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#define LOG_BUFFER_SIZE 1024
#define SCREEN_WIDTH 160
#define SCREEN_HEIGHT 208

#define PLAYABLE_AREA_TOP 0
#define PLAYABLE_AREA_BOTTOM 154
#define PLAYABLE_AREA_LEFT 8
#define PLAYABLE_AREA_RIGHT 160

#define ACTION_HEIGHT (PLAYABLE_AREA_BOTTOM - PLAYABLE_AREA_TOP)

#define CAR_WIDTH 16
#define CAR_HEIGHT 11
#define MAX_ENEMIES 10
#define CRASH_NOOP_DURATION 60
#define DAY_LENGTH 2000
#define INITIAL_CARS_TO_PASS 5
#define TOP_SPAWN_OFFSET 12.0f // Cars spawn/disappear 12 pixels from top

#define ROAD_LEFT_EDGE_X 26
#define ROAD_RIGHT_EDGE_X 127
#define VANISHING_POINT_Y 52 // 52
#define LOGICAL_VANISHING_Y VANISHING_POINT_Y + 12  // Separate logical vanishing point for cars disappearing

#define VANISHING_POINT_X 80 // Initial vanishing point x when going straight

#define INITIAL_PLAYER_X ((ROAD_LEFT_EDGE_X + ROAD_RIGHT_EDGE_X)/2 - CAR_WIDTH/2)

#define PLAYER_MAX_Y (ACTION_HEIGHT - CAR_HEIGHT) // Max y is carlengthfrom bottom
#define PLAYER_MIN_Y (ACTION_HEIGHT - 2 * CAR_HEIGHT) // Min y is 2 carlengths from bottom

#define ACCELERATION_RATE 0.05f
#define DECELERATION_RATE 0.1f
#define FRICTION 0.95f
#define MIN_SPEED -1.5f
#define CAR_PIXELS_COUNT 120
#define CAR_PIXEL_HEIGHT 11
#define CAR_PIXEL_WIDTH 16 // 16

#define CURVE_FREQUENCY 0.05f
#define CURVE_AMPLITUDE 30.0f
#define ROAD_BASE_WIDTH ROAD_RIGHT_EDGE_X - ROAD_LEFT_EDGE_X
#define NUM_LANES 3


typedef struct Log Log;
struct Log {
    float episode_return;
    float episode_length;
    float score;
};

typedef struct LogBuffer LogBuffer;
struct LogBuffer {
    Log* logs;
    int length;
    int idx;
};

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

// Structure for color transitions
typedef struct {
    int step;       // Step count at which the transition occurs
    Color color;    // The color at that step
} ColorTransition;

// Structure for sky transitions (supports gradients)
typedef struct {
    int step;
    Color colors[5]; // Up to 5 colors for gradient transitions
    int numColors;
} SkyTransition;

// Car structure for enemy cars
typedef struct Car Car;
struct Car {
    int lane;   // Lane index: 0, 1, or 2
    float y;    // Current y position
    int passed;
};

// Define the transitions for sky, mountains, and grass
#define NUM_SKY_TRANSITIONS 10
#define NUM_MOUNTAIN_TRANSITIONS 10
#define NUM_GRASS_TRANSITIONS 10

SkyTransition skyTransitions[NUM_SKY_TRANSITIONS];
ColorTransition mountainTransitions[NUM_MOUNTAIN_TRANSITIONS];
ColorTransition grassTransitions[NUM_GRASS_TRANSITIONS];

// Game environment structure
typedef struct Enduro Enduro;
struct Enduro {
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

    float vanishing_point_x;
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

    // Current background colors
    Color currentSkyColors[5];
    int currentSkyColorCount;
    Color currentMountainColor;
    Color currentGrassColor;

    // Victory flag display timer
    int victoryFlagTimer;

};

// Client structure for rendering and input handling
typedef struct Client Client;
struct Client {
    float width;
    float height;
    Color player_color;
    Color enemy_color;
    Color road_color;
};

Client* make_client(Enduro* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = env->width;
    client->height = env->height;
    client->player_color = WHITE;
    client->enemy_color = RED; // Changed to RED for better visibility
    client->road_color = DARKGREEN;

    InitWindow(client->width, client->height, "Enduro Clone");
    SetTargetFPS(60); // Adjust as needed
    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}


static inline Color MyColorLerp(Color c1, Color c2, float t) {
    Color result;
    result.r = (unsigned char)(c1.r + t * (c2.r - c1.r));
    result.g = (unsigned char)(c1.g + t * (c2.g - c1.g));
    result.b = (unsigned char)(c1.b + t * (c2.b - c1.b));
    result.a = (unsigned char)(c1.a + t * (c2.a - c1.a));
    return result;
}

// Update the init_color_transitions function
void init_color_transitions() {
    // Initialize sky transitions with all provided timings and colors

    // Note: Convert timings from minutes:seconds to steps (assuming 60 FPS)
    // For example, 0:20 (20 seconds) is 20 * 60 = 1200 steps

    // Sky colors provided in HSV, converted to RGB
    // sky_color_1_blue (156, 149, 110), (44, 75, 189)
    skyTransitions[0] = (SkyTransition){0, {{44, 75, 189, 255}}, 1}; // 0:00

    // sky_color_1_blue continues at 0:20 and 0:40
    skyTransitions[1] = (SkyTransition){1200, {{44, 75, 189, 255}}, 1}; // 0:20
    skyTransitions[2] = (SkyTransition){2400, {{44, 75, 189, 255}}, 1}; // 0:40

    // sky_color_snow (assuming white for snow)
    skyTransitions[3] = (SkyTransition){3600, {{236, 236, 236, 255}}, 1}; // 1:00

    // sky_color_2_dark_blue (157, 184, 92), (23, 34, 173)
    skyTransitions[4] = (SkyTransition){6000, {{23, 34, 173, 255}}, 1}; // 1:40

    // sky_color_purp_dark_blue ((185, 141, 91), (112, 40, 153)), ((161, 155, 96), (40, 36, 168))
    // At 1:48
    skyTransitions[5] = (SkyTransition){6480, {{143, 49, 103, 255}, {44, 75, 189, 255}}, 2};

    // At 1:54 - transition to pinkish
    skyTransitions[6] = (SkyTransition){6840, {{143, 49, 103, 255}, {112, 40, 153, 255}, {44, 75, 189, 255}}, 3};

    // At 1:56 - transition to orange-brown to red-orange to pinkish
    skyTransitions[7] = (SkyTransition){6960, {{117, 49, 17, 255}, {179, 42, 25, 255}, {143, 49, 103, 255}, {112, 40, 153, 255}}, 4};

    // Continue adding transitions as per the provided data...

    // For the rest of the sky transitions, you can fill them similarly based on the colors and timings provided.

    // Mountain transitions
    mountainTransitions[0] = (ColorTransition){0, (Color){134, 134, 29, 255}}; // mount_color_1_gold at 0:00
    mountainTransitions[1] = (ColorTransition){2400, (Color){213, 214, 209, 255}}; // mount_color_2_whitish at 0:40
    mountainTransitions[2] = (ColorTransition){6000, (Color){3, 59, 0, 255}}; // mount_color_3_less_dark_green at 1:40
    mountainTransitions[3] = (ColorTransition){8280, (Color){142, 142, 142, 255}}; // mount_color_4_gray at 2:18
    mountainTransitions[4] = (ColorTransition){10200, (Color){0, 0, 0, 0}}; // No mountains during fog at 2:50
    mountainTransitions[5] = (ColorTransition){11880, (Color){142, 142, 142, 255}}; // mount_color_4_gray at 3:18
    mountainTransitions[6] = (ColorTransition){12840, (Color){187, 92, 37, 255}}; // New mountain color at 3:34
    mountainTransitions[7] = (ColorTransition){13920, (Color){140, 128, 24, 255}}; // darkish_gold at 3:52
    mountainTransitions[8] = (ColorTransition){14880, (Color){134, 134, 29, 255}}; // mount_color_1_gold at 4:08

    // Grass transitions
    grassTransitions[0] = (ColorTransition){0, (Color){0, 68, 0, 255}}; // grass_color_1_green at 0:00
    grassTransitions[1] = (ColorTransition){3600, (Color){236, 236, 236, 255}}; // snow_color_1_white at 1:00
    grassTransitions[2] = (ColorTransition){6000, (Color){19, 52, 0, 255}}; // grass_color_2_dark_green at 1:40
    grassTransitions[3] = (ColorTransition){6480, (Color){48, 53, 0, 255}}; // grass_color_3_army_green at 1:48
    grassTransitions[4] = (ColorTransition){8280, (Color){0, 0, 0, 255}}; // grass_color_4_black at 2:18
    grassTransitions[5] = (ColorTransition){10200, (Color){75, 75, 75, 255}}; // grass_color_5_gray at 2:50
    grassTransitions[6] = (ColorTransition){11880, (Color){0, 0, 0, 255}}; // grass_color_4_black at 3:18
    grassTransitions[7] = (ColorTransition){13920, (Color){19, 52, 0, 255}}; // grass_color_2_dark_green at 3:52
    grassTransitions[8] = (ColorTransition){14880, (Color){0, 68, 0, 255}}; // grass_color_1_green at 4:08
}

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
    env->min_speed = -1.5f;
    env->max_speed = 3.0f;
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

    // Initialize color transitions
    init_color_transitions();

    // Set initial colors
    env->currentSkyColors[0] = skyTransitions[0].colors[0];
    env->currentSkyColorCount = skyTransitions[0].numColors;
    env->currentMountainColor = mountainTransitions[0].color;
    env->currentGrassColor = grassTransitions[0].color;

    env->victoryFlagTimer = 0;
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


// Pixel representation of the car, stored as coordinate pairs
static const int car_pixels[CAR_PIXELS_COUNT][2] = {
    {77, 147}, {77, 149}, {77, 151}, {77, 153},
    {78, 147}, {78, 149}, {78, 151}, {78, 153},
    {79, 144}, {79, 145}, {79, 146}, {79, 148}, {79, 150}, {79, 152}, {79, 154},
    {80, 144}, {80, 145}, {80, 146}, {80, 148}, {80, 150}, {80, 152}, {80, 154},
    {81, 145}, {81, 146}, {81, 148}, {81, 149}, {81, 150}, {81, 151}, {81, 152}, {81, 153},
    {82, 145}, {82, 146}, {82, 148}, {82, 149}, {82, 150}, {82, 151}, {82, 152}, {82, 153},
    {83, 144}, {83, 145}, {83, 146}, {83, 147}, {83, 148}, {83, 149}, {83, 150}, {83, 151},
    {83, 152}, {83, 153}, {83, 154},
    {84, 144}, {84, 145}, {84, 146}, {84, 147}, {84, 148}, {84, 149}, {84, 150}, {84, 151},
    {84, 152}, {84, 153}, {84, 154},
    {85, 144}, {85, 145}, {85, 146}, {85, 147}, {85, 148}, {85, 149}, {85, 150}, {85, 151},
    {85, 152}, {85, 153}, {85, 154},
    {86, 144}, {86, 145}, {86, 146}, {86, 147}, {86, 148}, {86, 149}, {86, 150}, {86, 151},
    {86, 152}, {86, 153}, {86, 154},
    {87, 145}, {87, 146}, {87, 148}, {87, 149}, {87, 150}, {87, 151}, {87, 152}, {87, 153},
    {88, 145}, {88, 146}, {88, 148}, {88, 149}, {88, 150}, {88, 151}, {88, 152}, {88, 153},
    {89, 144}, {89, 145}, {89, 146}, {89, 147}, {89, 149}, {89, 151}, {89, 153},
    {90, 144}, {90, 145}, {90, 146}, {90, 147}, {90, 149}, {90, 151}, {90, 153},
    {91, 148}, {91, 150}, {91, 152}, {91, 154},
    {92, 148}, {92, 150}, {92, 152}, {92, 154}
};

// Calculate curve offset
float calculate_curve_offset(Enduro* env) {
    return env->current_curve_direction * CURVE_AMPLITUDE;
}

void update_road_curve(Enduro* env) {
    static int current_curve_stage = 0;
    static int steps_in_current_stage = 0;

    // Define the number of steps for each curve stage
    int step_thresholds[] = {
        100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100 // <--test values. actual values: 250, 1000, 125, 250, 500, 800, 600, 200, 1100, 1200, 1000, 400, 200
    };

    int curve_directions[] = {
        1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1  // -1: Left, 1: Right, 0: Straight
    };

    steps_in_current_stage++;

    if (steps_in_current_stage >= step_thresholds[current_curve_stage]) {
        env->current_curve_direction = curve_directions[current_curve_stage];

        // Reset step counter and move to the next stage
        steps_in_current_stage = 0;
        current_curve_stage = (current_curve_stage + 1) % (sizeof(step_thresholds) / sizeof(int));
    }
}


// B(t)=(1−t)2P0​+2(1−t)tP1​+t2P2​,t∈[0,1]
// Quadratic bezier curve helper function
float quadratic_bezier(float p0, float p1, float p2, float t) {
    float one_minus_t = 1.0f - t;
    return one_minus_t * one_minus_t * p0 + 
           2.0f * one_minus_t * t * p1 + 
           t * t * p2;
}

float road_left_edge_x(Enduro* env, float y) {
    float t = (PLAYABLE_AREA_BOTTOM - y) / (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y);

    float x;

    if (env->current_curve_direction == -1) { // Left curve
        // Left edge Bezier curve points
        float P0_x = ROAD_LEFT_EDGE_X;   // Start point at bottom
        float P1_x = 81;                 // Control point x
        float P2_x = 40;                 // End point at horizon

        x = (1 - t)*(1 - t)*P0_x + 2*(1 - t)*t*P1_x + t*t*P2_x;
    } else if (env->current_curve_direction == 1) { // Right curve
        // Left edge Bezier curve points for right curve
        float P0_x = SCREEN_WIDTH - ROAD_RIGHT_EDGE_X;                      // Start point at bottom
        float P1_x = SCREEN_WIDTH - P0_x - 100; // 100 is magic #; (P0_x - env->left_curve_p1_x) Control point x (adjusted over time) for testing
        float P2_x = SCREEN_WIDTH - 40;                     // End point at horizon

        printf("left_curve_p1_x only: %.2f\n", env->left_curve_p1_x);
        printf("P0_x = %.2f, P1_x = %.2f, P2_x = %.2f\n", P0_x, P1_x, P2_x);



        x = (1 - t)*(1 - t)*P0_x + 2*(1 - t)*t*P1_x + t*t*P2_x;
    } else { // Straight road
        // Linear interpolation for straight road
        float P0_x = ROAD_LEFT_EDGE_X;
        float P2_x = VANISHING_POINT_X;

        x = P0_x + (P2_x - P0_x) * t;
    }

    return x;
}


float road_right_edge_x(Enduro* env, float y) {
    float t = (PLAYABLE_AREA_BOTTOM - y) / (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y);

    float x;
    if (env->current_curve_direction == -1) { // Left curve
        // Right edge Bezier curve points
        float P0_x = ROAD_RIGHT_EDGE_X;  // (127, 177)
        float P1_x = 135;                // Control point x (135, 92)
        float P2_x = 40;                 // End point x (40, 68)

        x = (1 - t)*(1 - t)*P0_x + 2*(1 - t)*t*P1_x + t*t*P2_x;
    } else if (env->current_curve_direction == 1) { // Right curve
        // Mirror the right curve across the center line
        float P0_x = SCREEN_WIDTH - ROAD_LEFT_EDGE_X; // Mirrored start point
        float P1_x = SCREEN_WIDTH - 69; // 69 is magic #; Control point SCREEN_WIDTH - env->left_curve_p1_x (testing)
        float P2_x = SCREEN_WIDTH - 40;               // Horizon endpoint mirrored

        x = (1 - t)*(1 - t)*P0_x + 2*(1 - t)*t*P1_x + t*t*P2_x;
    } else { // Straight road
        // Linear interpolation for straight road
        float P0_x = ROAD_RIGHT_EDGE_X;
        float P2_x = VANISHING_POINT_X;

        x = P0_x + (P2_x - P0_x) * t;
    }

    return x;
}


// Update car_x_in_lane function
float car_x_in_lane(Enduro* env, int lane, float y) {
    float left_edge = road_left_edge_x(env, y);
    float right_edge = road_right_edge_x(env, y);
    float lane_width = (right_edge - left_edge) / NUM_LANES;
    return left_edge + lane_width * (lane + 0.5f);
}

void compute_observations(Enduro* env) {
    env->observations[0] = env->player_x / SCREEN_WIDTH;
    env->observations[1] = env->player_y / ACTION_HEIGHT;
    env->observations[2] = env->speed / env->max_speed;
    env->observations[3] = env->carsToPass;
    env->observations[4] = env->day;
    env->observations[5] = env->numEnemies;
    env->observations[6] = env->collision_cooldown;
    env->observations[7] = env->score;

    int obs_idx = 8;
    for (int i = 0; i < MAX_ENEMIES; i++) {
        env->observations[obs_idx++] = env->enemyCars[i].lane / 3.0f;
        env->observations[obs_idx++] = env->enemyCars[i].y / ACTION_HEIGHT;
    }
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
    env->log = (Log){0};
    reset_round(env);
    compute_observations(env);
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
    float left_edge = road_left_edge_x(env, env->player_y);
    float right_edge = road_right_edge_x(env, env->player_y);
    float lane_width = (right_edge - left_edge) / 3.0f;

    int lane = (int)((player_center_x - left_edge) / lane_width);
    if (lane < 0) lane = 0;
    if (lane > 2) lane = 2;
    return lane;
}

void add_enemy_car(Enduro* env) {
    if (env->numEnemies >= MAX_ENEMIES) return;
    int lane = rand() % 3;

    if (env->speed < 0) {
        int player_lane = get_player_lane(env);
        // Avoid spawning in the player's lane
        while (lane == player_lane) {
            lane = rand() % 3;
        }
    }

    Car car = { .lane = lane, .passed = false };

    // Adjust car spawning location based on player speed
    if (env->speed > 0) {
        // Spawn cars at the vanishing point when moving forward
        car.y = VANISHING_POINT_Y;
        printf("Spawning car at VANISHING_POINT_Y = %.2f\n", (float)VANISHING_POINT_Y);
    } else {
        // Spawn cars behind the player when moving backward
        car.y = ACTION_HEIGHT;  // Spawn at the bottom edge
        printf("Spawning car at ACTION_HEIGHT = %.2f\n", (float)ACTION_HEIGHT);
    }

    env->enemyCars[env->numEnemies++] = car;
}

// Update the update_background_colors function to use MyColorLerp
void update_background_colors(Enduro* env) {
    int step = env->step_count;

    // Update sky colors with interpolation
    for (int i = 0; i < NUM_SKY_TRANSITIONS - 1; i++) {
        if (step >= skyTransitions[i].step && step < skyTransitions[i + 1].step) {
            float t = (float)(step - skyTransitions[i].step) / (skyTransitions[i + 1].step - skyTransitions[i].step);
            for (int j = 0; j < skyTransitions[i].numColors; j++) {
                Color c1 = skyTransitions[i].colors[j];
                Color c2 = skyTransitions[i + 1].colors[j];
                env->currentSkyColors[j] = MyColorLerp(c1, c2, t);
            }
            printf("Current sky transition index: %d\n", i);

            env->currentSkyColorCount = skyTransitions[i].numColors;
            break;
        }
    }

    // Update mountain color with interpolation
    for (int i = 0; i < NUM_MOUNTAIN_TRANSITIONS - 1; i++) {
        if (step >= mountainTransitions[i].step && step < mountainTransitions[i + 1].step) {
            float t = (float)(step - mountainTransitions[i].step) / (mountainTransitions[i + 1].step - mountainTransitions[i].step);
            env->currentMountainColor = MyColorLerp(mountainTransitions[i].color, mountainTransitions[i + 1].color, t);
            break;
        }
    }

    // Update grass color with interpolation
    for (int i = 0; i < NUM_GRASS_TRANSITIONS - 1; i++) {
        if (step >= grassTransitions[i].step && step < grassTransitions[i + 1].step) {
            float t = (float)(step - grassTransitions[i].step) / (grassTransitions[i + 1].step - grassTransitions[i].step);
            env->currentGrassColor = MyColorLerp(grassTransitions[i].color, grassTransitions[i + 1].color, t);
            break;
        }
    }
}

// Update rendering functions to use the updated colors
void render_sky(Client* client, Enduro* env) {
    if (env->currentSkyColorCount == 1) {
        // Solid sky color
        DrawRectangle(0, 0, SCREEN_WIDTH, VANISHING_POINT_Y, env->currentSkyColors[0]);
    } else {
        // Gradient sky
        for (int y = 0; y < VANISHING_POINT_Y; y++) {
            float t = (float)y / VANISHING_POINT_Y;
            Color color = env->currentSkyColors[0];
            for (int i = 1; i < env->currentSkyColorCount; i++) {
                color = MyColorLerp(color, env->currentSkyColors[i], t);
            }
            DrawLine(0, y, SCREEN_WIDTH, y, color);
        }
    }
}

void step(Enduro* env) {
    if (env == NULL) {
        printf("[ERROR] env is NULL! Aborting step.\n");
        return;
    }

    update_road_curve(env);
    update_background_colors(env);

        // Adjust left_curve_p1_x when road is curving right
    if (env->current_curve_direction == 1) {
        env->left_curve_p1_x += env->left_curve_p1_x_increment;
        if (env->left_curve_p1_x > env->left_curve_p1_x_max) {
            env->left_curve_p1_x = env->left_curve_p1_x_min;
        }}
    // } else {
    //     env->left_curve_p1_x = 0.0f;  // Reset when not curving right
    // }

    env->log.episode_length += 1;
    env->terminals[0] = 0;

    // Update road scroll offset
    env->road_scroll_offset += env->speed;

    // Update enemy cars even during collision cooldown
    for (int i = 0; i < env->numEnemies; i++) {
        Car* car = &env->enemyCars[i];
        // Update enemy car position
        car->y += env->speed;
        printf("Updating car %d position to y = %.2f\n", i, car->y);
    }

    // Limit player's x position based on road edges at player's y position
    float road_left = road_left_edge_x(env, env->player_y);
    float road_right = road_right_edge_x(env, env->player_y);

    // Player movement logic
    if (env->collision_cooldown <= 0) {
        int act = env->actions;
        if (act == 1) {  // Move left
            env->player_x -= 1;
            if (env->player_x < road_left) env->player_x = road_left;
            env->last_lr_action = 1;
        } else if (act == 2) {  // Move right
            env->player_x += 1;
            if (env->player_x > road_right - CAR_WIDTH) env->player_x = road_right - CAR_WIDTH;
            env->last_lr_action = 2;
        }
        if (act == 3 && env->speed < env->max_speed) env->speed += ACCELERATION_RATE;
        if (act == 4 && env->speed > env->min_speed) env->speed -= DECELERATION_RATE;
    } else {
        env->collision_cooldown -= 1;
        if (env->last_lr_action == 1) env->player_x -= 25;
        if (env->last_lr_action == 2) env->player_x += 25;
        env->speed *= FRICTION;
        env->speed -= 0.305 * DECELERATION_RATE;
        if (env->speed < env->min_speed) env->speed = env->min_speed;
    }

    if (env->player_x < road_left) env->player_x = road_left;
    if (env->player_x > road_right - CAR_WIDTH) env->player_x = road_right - CAR_WIDTH;

    // Update vanishing point based on player's horizontal movement
    env->vanishing_point_x = VANISHING_POINT_X - (env->player_x - env->initial_player_x);

    // Update player y position based on speed
    env->player_y = PLAYER_MAX_Y - (env->speed - env->min_speed) / (env->max_speed - env->min_speed) * (PLAYER_MAX_Y - PLAYER_MIN_Y);
    // Clamp player_y
    if (env->player_y < PLAYER_MIN_Y) env->player_y = PLAYER_MIN_Y;
    if (env->player_y > PLAYER_MAX_Y) env->player_y = PLAYER_MAX_Y;

    printf("Player position y = %.2f, speed = %.2f\n", env->player_y, env->speed);

    // Enemy car logic
    for (int i = 0; i < env->numEnemies; i++) {
        Car* car = &env->enemyCars[i];

        // Check for passing logic
        if (env->speed > 0 && car->y > env->player_y + CAR_HEIGHT && !car->passed) {
            env->carsToPass--;
            if (env->carsToPass < 0) env->carsToPass = 0;
            car->passed = true;
            env->score += 10;
            env->rewards[0] += 1;
        }

        // Check for collisions between the player and the car
        if (check_collision(env, car)) {
            env->speed = env->min_speed = MIN_SPEED;
            env->collision_cooldown = CRASH_NOOP_DURATION;

            // Drift the player's car if there is a collision
            if (env->last_lr_action == 1) {  // Last action was left
                env->player_x -= 10;
                if (env->player_x < road_left) env->player_x = road_left;
            } else if (env->last_lr_action == 2) {  // Last action was right
                env->player_x += 10;
                if (env->player_x > road_right - CAR_WIDTH) env->player_x = road_right - CAR_WIDTH;
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
            printf("Removing car %d as it went below screen\n", i);
        }

        // Remove cars that reach or surpass the logical vanishing point if moving up (speed negative)
        if (env->speed < 0 && car->y <= LOGICAL_VANISHING_Y) {
            // Remove car from array if it reaches the logical vanishing point while moving backward
            for (int j = i; j < env->numEnemies - 1; j++) {
                env->enemyCars[j] = env->enemyCars[j + 1];
            }
            env->numEnemies--;
            i--;
            printf("Removing car %d as it reached the logical vanishing point at LOGICAL_VANISHING_Y = %.2f\n", i, (float)LOGICAL_VANISHING_Y);
        }

    }

    // Adjust enemy car spawn frequency based on day number
    if (env->numEnemies < MAX_ENEMIES && rand() % 100 < 1) {
        add_enemy_car(env);
    }

    // Handle day completion logic
    if (env->carsToPass <= 0) {
        env->day++;
        env->carsToPass = env->day * 10 + 10;
        env->speed += 0.1f;

        add_log(env->log_buffer, &env->log);
        env->rewards[0] = 0;
    } else if (env->step_count >= env->day_length) {
        if (env->carsToPass > 0) {
            env->terminals[0] = 1;
            add_log(env->log_buffer, &env->log);
            reset(env);
            return;
        }
    }

    env->log.episode_return += env->rewards[0];
    env->step_count++;
    env->log.score = env->score;
}



void render_borders(Client* client) {
    // Left black vertical bar (8 pixels wide)
    DrawRectangle(0, 0, PLAYABLE_AREA_LEFT, SCREEN_HEIGHT, BLACK);

    // Bottom horizontal black bar (55 pixels high)
    DrawRectangle(0, PLAYABLE_AREA_BOTTOM, SCREEN_WIDTH, SCREEN_HEIGHT - PLAYABLE_AREA_BOTTOM, BLACK);
}

void render_scoreboard(Client* client, Enduro* env) {
    // Draw scoreboard as a red rectangle
    DrawRectangle(PLAYABLE_AREA_LEFT, PLAYABLE_AREA_BOTTOM, SCREEN_WIDTH - PLAYABLE_AREA_LEFT, SCREEN_HEIGHT - PLAYABLE_AREA_BOTTOM, RED);

    // Render the score information within the scoreboard
    DrawText(TextFormat("Score: %05i", env->score), 10, PLAYABLE_AREA_BOTTOM + 10, 10, WHITE);
    DrawText(TextFormat("Cars to Pass: %03i", env->carsToPass), 10, PLAYABLE_AREA_BOTTOM + 25, 10, WHITE);
    DrawText(TextFormat("Day: %i", env->day), 10, PLAYABLE_AREA_BOTTOM + 40, 10, WHITE);
    DrawText(TextFormat("Speed: %.2f", env->speed), 10, PLAYABLE_AREA_BOTTOM + 55, 10, WHITE);
    DrawText(TextFormat("Step: %i", env->step_count), 10, PLAYABLE_AREA_BOTTOM + 70, 10, WHITE);

    
    if (env->victoryFlagTimer > 0) {
        DrawText("Victory!", SCREEN_WIDTH / 2 - 30, PLAYABLE_AREA_BOTTOM + 85, 20, GREEN);
        env->victoryFlagTimer--;
    }
}


void render_car(Client* client, Enduro* env) {
    // Render the player car based on dynamic player_x and player_y, centered correctly
    for (int i = 0; i < CAR_PIXELS_COUNT; i++) {
        int dx = car_pixels[i][0];
        int dy = car_pixels[i][1];

        // Calculate the actual position based on player's current coordinates
        int pixel_x = (int)(env->player_x + (dx - 77));
        int pixel_y = (int)(env->player_y + (dy - 144));

        // Draw the car's pixel at the calculated position
        DrawPixel(pixel_x, pixel_y, client->player_color);
    }
}


// Sky vs ground dimensions:
// Sky: 0 to 266
// Ground: 267 to 613

// Background, day mountains:
// mount_color_1_gold (40, 155, 77), (134, 134, 29) (at 20 seconds)
// mount_color_2_whitish (48, 14, 199), (213, 214, 209) (at 40 seconds)
// mount_color_3_less_dark_green (78, 240, 28), (3, 59, 0) (at 1:40) (at 1:48) (at 1:54) (at 1:56)
// Background, night mountains:
// mount_color_4_gray (160, 0, 134), (142, 142, 142) (at 2:18 through 2:50)
// at 2:50, there are no mountains because this is fog weather.
// at 3:18, it is back to mount_color_4_gray.
// at 3:34, mountains are color (15, 161, 105), (187, 92, 37).
// at 3:52, mountains are color darkish_gold (36, 170, 77), (140, 128, 24).
// the cycle repeats from here, starting at 4:08 with mount_color_1_gold.
// at 4:28, mountains are mount_color_2_whitish again, etc.


// Background, sky:
// sky_color_1_blue (156, 149, 110), (44, 75, 189) (at 20 seconds) (at 40 seconds) (at 60 seconds)
// sky_color_2_dark_blue (157, 184, 92), (23, 34, 173) (at 1:40)
// sky_color_purp_dark_blue ((185, 141, 91), (112, 40, 153)), ((161, 155, 96), (40, 36, 168)) (at 1:48) (in order from horizon to top: purple, blue; purple fades into blue)
// 1:54 - sky now transitions from pinkish (217, 118, 90), (143, 49, 103) to purp to dark blue
// 1:56 - sky now transitions from orange-brown (13, 179, 63), (117, 49, 17) to red-orange (4, 181, 96), (179, 42, 25) to pinkish to purp to dark blue
// 2:00 - sky now transitions from lighter-orange-brown (11, 185, 95), (179, 66, 23) to red-orange to pinkish to purp to violet (183, 144, 94), (107, 40, 159). mountains are now almost-black (80, 240, 2), (0, 5, 0).
// 2:04 - sky now transitions from dark-pastel-orange (12, 163, 109), (195, 86, 37) to lighter-orange-brown to red-orange to pinkish.
// 2:10 - sky now transitions from brown-gold (19, 140, 95), (159,99,42) to (14, 170, 103), (187, 88, 32) to (12, 198, 90), (175, 63, 17) to (4, 176, 99), (182, 42, 28).
// 2:14 - sky now transitions from (40, 131, 75), (123, 123, 36) to (31, 151, 87), (150, 124, 34) to (26, 173, 80), (147, 104, 24) to (17, 163, 102), (182, 96, 35) to (13, 197, 89), (173, 66, 17).
// 2:18 - sky is solid gray (160, 0, 71), (75, 75, 75). this continues until 2:50, when the sky disappears due to fog weather.
// 3:18 - sky is again solid gray.
// 3:34 - sky is lighter gray (120, 1, 104), (110, 111, 111).
// 3:52 - sky is again sky_color_2_dark_blue.
// the cycle repeats from here, starting at 4:08 with sky_color_1_blue.


// Background, grass:
// grass_color_1_green (80, 240, 32), (0, 68, 0) (daytime grass color) (at 20 seconds) (at 40 seconds)
// snow_color_1_white (160, 0, 222), (236, 236, 236) (at 60 seconds)
// grass_color_2_dark_green (65, 240, 24), (19, 52, 0) (at 1:40) (at 1:48) (at 1:54) (at 1:56)
// grass_color_3_army_green (44, 240, 25), (48, 53, 0) (at 1:48 through 2:14)
// grass_color_4_black (160, 0, 0), (0, 0, 0) (at 2:18 through 2:50)
// grass_color_5_gray (160, 0, 71), (75, 75, 75) (at 2:50)
// grass_color_4_black again appears at 3:18
// at 3:52, grass_color_2_dark_green appears again.
// at 4:08, grass_color_1_green appears again, and the cycle repeats.


// mountain dimensions: there are only 2 mountains. these px values are scaled to 1016 width 613 height. they should be rescaled to 152 width 154 height.
// mountain_1 is a ziggarut-style step-mountain that is 38 px high, with the base on the horizon at 266 px on screen.
// each step going up is as follows: 264, 260, 258, 254, 250, 247 (top of plateau). steps going down follow the same pattern.
// mountain_1 starts on left at 34 px and spans to 247 px at its base.
// from L to R, steps ascend: 34-47 is the first step, 47-60 is the second step, 60-73 is the third step, 73-86 is the fourth step, 86-99 is the fifth step, 99-112 is the sixth step.
// from 99-127 is the plateau. there are 3 descending steps from 127-140, 140-153, with a 'valley' from 153-180. 
// then, there is 1 step up to a mini-plateau from 180-207, then steps down from 207-220, then from 220-233, and finally from 233-246.

// mountain_2 starts at 514 px on the horizon and spans to 701 px at its base. it is 17 px high. the top of its plateau is at 249 px.
// 528, 541, 581, and 594 are the starts of the steps. 594-621 is the plateau. 621-662, 662-675, 675-688 are the steps down, with the last step from 688-701.

// sky color transitions are from 266-260-254-249-244. Above 244 (from 244-0) is always a solid sky color. 244 is always the highest point of different sky colors.


// Close tail light color: (212, 108, 134), (193, 92, 163)
// Far tail light color: (239, 145, 140), (213, 85, 88)

// Color palette (HSV), (RGB):
// Enemy cars: (colors seem to be randomly distributed)
// teal (108, 99, 105), (66, 158, 130)
// goldenrod (40, 141, 96), (162, 162, 42)
// gold-gray (29, 117, 103), (162, 134, 56)
// perriwinkle (145, 123, 122), (66, 114, 194)

// Road palette:
// Road boundaries closest to horizon:
// slate (160, 0, 70), (74, 74, 74)
// Road boundaries middle:
// gray (160, 0, 104), (111, 111, 111)
// Road boundaries furthest from horizon:
// lightish gray (160, 0, 181), (192, 192, 192)

// HUD:
// Background:
// darkish red (0, 175, 91), (167, 26, 26)
// Text backlight:
// Brown-yellow (25, 126, 120), (195, 144, 61)
// Text:
// Sat black(==black, but still) (160, 0, 0), (0, 0, 0)
// Victory flag:
// Dark Pastel Green (82, 85, 117), (80, 168, 84)


// Implement the render_mountains function
void render_mountains(Client* client, Enduro* env) {
    // Don't render mountains during foggy weather
    if (env->currentMountainColor.a == 0 || (env->currentGrassColor.r == 0 && env->currentGrassColor.g == 0 && env->currentGrassColor.b == 0)) {
        return;
    }

    // Mountain dimensions scaled to your game's screen size
    // Simplified mountain rendering using triangles

    // Mountain 1
    Vector2 m1_base_left = {PLAYABLE_AREA_LEFT, VANISHING_POINT_Y};
    Vector2 m1_peak = {SCREEN_WIDTH / 4, VANISHING_POINT_Y - 38};
    Vector2 m1_base_right = {SCREEN_WIDTH / 2, VANISHING_POINT_Y};

    DrawTriangle(m1_base_left, m1_peak, m1_base_right, env->currentMountainColor);

    // Mountain 2
    Vector2 m2_base_left = {SCREEN_WIDTH / 2, VANISHING_POINT_Y};
    Vector2 m2_peak = {3 * SCREEN_WIDTH / 4, VANISHING_POINT_Y - 17};
    Vector2 m2_base_right = {SCREEN_WIDTH - PLAYABLE_AREA_LEFT, VANISHING_POINT_Y};

    DrawTriangle(m2_base_left, m2_peak, m2_base_right, env->currentMountainColor);
}


void render_grass(Client* client, Enduro* env) {
    DrawRectangle(0, VANISHING_POINT_Y, SCREEN_WIDTH, PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y, env->currentGrassColor);
}


void render(Client* client, Enduro* env) {
    BeginDrawing();

    printf("Step count: %d\n", env->step_count);


    // Render sky
    render_sky(client, env);

    // Render mountains
    render_mountains(client, env);

    // Render grass
    render_grass(client, env);
    
    // Draw the darker sky
    // ClearBackground((Color){50, 50, 150, 255});

    // Draw grass (sides of the road)
    // DrawRectangle(0, VANISHING_POINT_Y, SCREEN_WIDTH, PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y, DARKGREEN);

    // Draw the playable area boundary
    BeginScissorMode(PLAYABLE_AREA_LEFT, VANISHING_POINT_Y, PLAYABLE_AREA_RIGHT - PLAYABLE_AREA_LEFT, PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y);

    
    // Render road
    for (float y = VANISHING_POINT_Y; y <= PLAYABLE_AREA_BOTTOM; y++) {
        float left_edge = road_left_edge_x(env, y);
        float right_edge = road_right_edge_x(env, y);
        DrawLine(left_edge, y, right_edge, y, GRAY);
    }

    // Road edge lines
    for (float y = VANISHING_POINT_Y; y <= PLAYABLE_AREA_BOTTOM; y += 0.75f) {
        float adjusted_y = (env->speed < 0) ? y : y + fmod(env->road_scroll_offset, 0.75f);
        if (adjusted_y > PLAYABLE_AREA_BOTTOM) continue;

        float left_edge = road_left_edge_x(env, adjusted_y);
        float right_edge = road_right_edge_x(env, adjusted_y);

        // Draw left edge line
        DrawPixel(left_edge, adjusted_y, WHITE);

        // Draw right edge line
        DrawPixel(right_edge, adjusted_y, WHITE);
    }



    // Render enemy cars with scaling
    for (int i = 0; i < env->numEnemies; i++) {
        Car* car = &env->enemyCars[i];

        // Compute the scale factor based on y position
        float depth = (car->y - VANISHING_POINT_Y) / (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y);
        float scale = 0.1f + 0.9f * depth; // Scale ranges from 0.1 to 1.0

        // Compute car x position in its lane
        float car_center_x = car_x_in_lane(env, car->lane, car->y);
        float car_x = car_center_x - (CAR_PIXEL_WIDTH * scale) / 2.0f;

        // Draw the scaled car pixels
        for (int j = 0; j < CAR_PIXELS_COUNT; j++) {
            int dx = car_pixels[j][0] - 77; // Centering the x-offset
            int dy = car_pixels[j][1] - 144; // Centering the y-offset

            // Compute the actual pixel position with scaling
            float pixel_x = car_x + dx * scale;
            float pixel_y = car->y + dy * scale;

            // Draw the pixel for the car
            DrawPixel((int)pixel_x, (int)pixel_y, client->enemy_color);
        }
    }


    // Render player car (no scaling since it's at the bottom)
    render_car(client, env);

    EndScissorMode();

    // Render borders
    render_borders(client);
    render_scoreboard(client, env);

    // // Render HUD env data
    // DrawText(TextFormat("Score: %05i", env->score), 10, PLAYABLE_AREA_BOTTOM + 10, 10, WHITE);
    // DrawText(TextFormat("Cars to Pass: %03i", env->carsToPass), 10, PLAYABLE_AREA_BOTTOM + 25, 10, WHITE);
    // DrawText(TextFormat("Day: %i", env->day), 10, PLAYABLE_AREA_BOTTOM + 40, 10, WHITE);
    // DrawText(TextFormat("Speed: %.2f", env->speed), 10, PLAYABLE_AREA_BOTTOM + 55, 10, WHITE);
    // DrawText(TextFormat("Step: %i", env->step_count), 10, PLAYABLE_AREA_BOTTOM + 70, 10, WHITE);
    // // Box around HUD
    // DrawRectangleLines(0, PLAYABLE_AREA_BOTTOM, SCREEN_WIDTH, SCREEN_HEIGHT - PLAYABLE_AREA_BOTTOM, WHITE);

    EndDrawing();
}

