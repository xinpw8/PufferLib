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

// Car structure for enemy cars
typedef struct Car Car;
struct Car {
    int lane;   // Lane index: 0, 1, or 2
    float y;    // Current y position
    int passed;
};

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


float road_left_edge_x(Enduro* env, float y) {
    float x1 = ROAD_LEFT_EDGE_X;
    float y1 = PLAYABLE_AREA_BOTTOM;
    float vp_x = env->vanishing_point_x;
    float vp_y = VANISHING_POINT_Y;  // Keep this for perspective calculations
    if (y >= y1) return x1;
    return vp_x + (x1 - vp_x) * (y - vp_y) / (y1 - vp_y);
}

float road_right_edge_x(Enduro* env, float y) {
    float x2 = ROAD_RIGHT_EDGE_X;
    float y1 = PLAYABLE_AREA_BOTTOM;
    float vp_x = env->vanishing_point_x;
    float vp_y = VANISHING_POINT_Y;  // Keep this for perspective calculations
    if (y >= y1) return x2;
    return vp_x + (x2 - vp_x) * (y - vp_y) / (y1 - vp_y);
}


// float road_left_edge_x(Enduro* env, float y) {
//     float x1 = ROAD_LEFT_EDGE_X;
//     float y1 = PLAYABLE_AREA_BOTTOM;
//     float vp_x = env->vanishing_point_x;
//     float vp_y = VANISHING_POINT_Y;
//     if (y >= y1) return x1;
//     return vp_x + (x1 - vp_x) * (y - vp_y) / (y1 - vp_y);
// }

// float road_right_edge_x(Enduro* env, float y) {
//     float x2 = ROAD_RIGHT_EDGE_X;
//     float y1 = PLAYABLE_AREA_BOTTOM;
//     float vp_x = env->vanishing_point_x;
//     float vp_y = VANISHING_POINT_Y;
//     if (y >= y1) return x2;
//     return vp_x + (x2 - vp_x) * (y - vp_y) / (y1 - vp_y);
// }

float car_x_in_lane(Enduro* env, int lane, float y) {
    float left_edge = road_left_edge_x(env, y);
    float right_edge = road_right_edge_x(env, y);
    float lane_width = (right_edge - left_edge) / 3.0f;
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


// // Update the check_collision function
// bool check_collision(Enduro* env, Car* car) {
//     // Compute the scale factor
//     float depth = (car->y - VANISHING_POINT_Y) / (PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y);
//     float scale = 0.1f + 0.9f * depth;

//     // Compute car dimensions
//     float car_width = CAR_WIDTH * scale;
//     float car_height = CAR_HEIGHT * scale;

//     // Compute car x position
//     float car_center_x = car_x_in_lane(env, car->lane, car->y);
//     float car_x = car_center_x - car_width / 2.0f;

//     // Check collision
//     return !(env->player_x > car_x + car_width ||
//              env->player_x + CAR_WIDTH < car_x ||
//              env->player_y > car->y + car_height ||
//              env->player_y + CAR_HEIGHT < car->y);
// }

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


void step(Enduro* env) {
    if (env == NULL) {
        printf("[ERROR] env is NULL! Aborting step.\n");
        return;
    }

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




// Color pallette (HSV), (RGB):
// Enemy cars:
// teal (108, 99, 105), (66, 158, 130)
// goldenrod (40, 141, 96), (162, 162, 42)
// gold-gray (29, 117, 103), (162, 134, 56)
// perriwinkle (145, 123, 122), (66, 114, 194)

// Grass and Road:
// forest green (80, 240, 32), (0, 68, 0)
// Road boundaries closest to horizon:
// slate (160, 0, 70), (74, 74, 74)
// Road boundaries middle:
// gray (160, 0, 104), (111, 111, 111)
// Road boundaries furthest from horizon:
// lightish gray (160, 0, 181), (192, 192, 192)

// Background, day mountains:
// gold (40, 155, 77), (134, 134, 29)

// HUD:
// Background:
// darkish red (0, 175, 91), (167, 26, 26)
// Text backlight:
// Brown-yellow (25, 126, 120), (195, 144, 61)
// Text:
// Sat black(==black, but still) (160, 0, 0), (0, 0, 0)
// Victory flag:
// Dark Pastel Green (82, 85, 117), (80, 168, 84)


void render(Client* client, Enduro* env) {
    BeginDrawing();

    // Draw the darker sky
    ClearBackground((Color){50, 50, 150, 255});

    // Draw grass (sides of the road)
    DrawRectangle(0, VANISHING_POINT_Y, SCREEN_WIDTH, PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y, DARKGREEN);

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

