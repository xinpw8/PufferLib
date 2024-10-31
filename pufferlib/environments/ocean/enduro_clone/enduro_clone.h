// enduro_clone.h

#ifndef ENDURO_CLONE_H
#define ENDURO_CLONE_H

#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <stddef.h> // For NULL
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <string.h>

// Define constants
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
#define DAY_LENGTH 2000
#define INITIAL_CARS_TO_PASS 5
#define TOP_SPAWN_OFFSET 12.0f // Cars spawn/disappear 12 pixels from top

#define ROAD_LEFT_EDGE_X 26
#define ROAD_RIGHT_EDGE_X 127
#define VANISHING_POINT_Y 52
#define LOGICAL_VANISHING_Y (VANISHING_POINT_Y + 12)  // Separate logical vanishing point for cars disappearing

#define VANISHING_POINT_X 80 // Initial vanishing point x when going straight

#define INITIAL_PLAYER_X ((ROAD_LEFT_EDGE_X + ROAD_RIGHT_EDGE_X)/2 - CAR_WIDTH/2)

#define PLAYER_MAX_Y (ACTION_HEIGHT - CAR_HEIGHT) // Max y is car length from bottom
#define PLAYER_MIN_Y (ACTION_HEIGHT - 2 * CAR_HEIGHT) // Min y is 2 car lengths from bottom

#define ACCELERATION_RATE 0.05f
#define DECELERATION_RATE 0.1f
#define FRICTION 0.95f
#define MIN_SPEED -1.5f
#define MAX_SPEED 3.0f
#define CAR_PIXELS_COUNT 120
#define CAR_PIXEL_HEIGHT 11
#define CAR_PIXEL_WIDTH 16

#define CURVE_FREQUENCY 0.05f
#define CURVE_AMPLITUDE 30.0f

#define SMOOTH_CURVE_TRANSITION_RATE 0.02f  // Controls curve transition speed
#define MAX_CURVE_CONTROL_OFFSET 50.0f      // Maximum offset for control point to create swooping effect


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

// Define Color struct
typedef struct Color {
    Uint8 r;
    Uint8 g;
    Uint8 b;
    Uint8 a;
} Color;

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
    int passed;
} Car;

// Structure to hold game state related to rendering
typedef struct GameState {
    SDL_Texture* backgroundTextures[16]; // 16 different backgrounds for time of day
    SDL_Texture* digitTextures[10];      // Textures for digits 0-9
    SDL_Texture* carDigitTexture;        // Texture for the "CAR" digit
    SDL_Texture* mountainTextures[16];   // Mountain textures corresponding to backgrounds

    SDL_Texture* levelCompleteFlagLeftTexture;  // Texture for left flag
    SDL_Texture* levelCompleteFlagRightTexture; // Texture for right flag
    SDL_Texture* greenDigitTextures[10];        // Textures for green digits
    SDL_Texture* yellowDigitTextures[10];       // Textures for yellow digits

    int currentBackgroundIndex;
    int previousBackgroundIndex;
    int score;
    int day;
    int carsToPass;
    float mountainPosition; // Position of the mountain texture

    bool victoryAchieved;   // Flag to indicate victory condition

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

    // Gamestate
    GameState gameState;

} Enduro;

// Client structure
typedef struct Client {
    float width;
    float height;
    Color player_color;
    Color enemy_color;
    Color road_color;

    SDL_Renderer* renderer; // Add renderer to Client
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
Client* make_client(Enduro* env, SDL_Renderer* renderer);
void close_client(Client* client);

// Event handling
void handleEvents(int* running, Enduro* env);

// Rendering functions
void render(Client* client, Enduro* env);
// void render_borders(Client* client);
void render_car(Client* client, Enduro* env);

// GameState functions
int initSDL(SDL_Window** window, SDL_Renderer** renderer);
void loadTextures(SDL_Renderer* renderer, GameState* gameState);
void cleanup(SDL_Window* window, SDL_Renderer* renderer, GameState* gameState);
void updateBackground(GameState* gameState, int timeOfDay);
void renderBackground(SDL_Renderer* renderer, GameState* gameState);
void renderScoreboard(SDL_Renderer* renderer, GameState* gameState);
void updateMountains(GameState* gameState, RoadDirection direction);
void renderMountains(SDL_Renderer* renderer, GameState* gameState);
void updateVictoryEffects(GameState* gameState);
void updateScore(GameState* gameState);

// Additional function prototypes
float road_left_edge_x(Enduro* env, float y);
float road_right_edge_x(Enduro* env, float y);
float car_x_in_lane(Enduro* env, int lane, float y);
void update_road_curve(Enduro* env);

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
    env->log = (Log){0};
    reset_round(env);
    // compute_observations(env); // Implement if needed
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

void step(Enduro* env) {
    if (env == NULL) {
        printf("[ERROR] env is NULL! Aborting step.\n");
        return;
    }

    update_road_curve(env);
    // update_background_colors(env);

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
        // printf("Updating car %d position to y = %.2f\n", i, car->y);
    }

    // Limit player's x position based on road edges at player's y position
    float road_left = road_left_edge_x(env, env->player_y);
    float road_right = road_right_edge_x(env, env->player_y);

    // Player movement logic
    if (env->collision_cooldown <= 0) {
        int act = env->actions;
        if (act == 1) {  // Move left
            env->player_x -= 2;
            if (env->player_x < road_left) env->player_x = road_left;
            env->last_lr_action = 1;
        } else if (act == 2) {  // Move right
            env->player_x += 2;
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


void update_road_curve(Enduro* env) {
    static int current_curve_stage = 0;
    static int steps_in_current_stage = 0;

    // Define the number of steps for each curve stage
    int step_thresholds[] = {
        150, 150, 150, 150, 150,150, 150, 150, 150, 150,150, 150, 150, 150, 150, // <--test values. actual values: 250, 1000, 125, 250, 500, 800, 600, 200, 1100, 1200, 1000, 400, 200
    };

    int curve_directions[] = {
        1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 1, -1, 1, -1, 1, -1, 1  // -1: Left, 1: Right, 0: Straight
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
           1.5f * one_minus_t * t * p1 + 
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
        x = quadratic_bezier(P0_x, P1_x, P2_x, t);
    } else if (env->current_curve_direction == 1) { // Right curve
        // Left edge Bezier curve points for right curve
        float P0_x = ROAD_LEFT_EDGE_X;                      // Start point at bottom
        float P1_x = ROAD_LEFT_EDGE_X + 30;                 // Reduced curve control point
        float P2_x = SCREEN_WIDTH - 40;                     // End point at horizon
        x = quadratic_bezier(P0_x, P1_x, P2_x, t);
    } else { // Straight road
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
        float P0_x = ROAD_RIGHT_EDGE_X;  
        float P1_x = 135;                // Control point x
        float P2_x = 40;                 // End point at horizon
        x = quadratic_bezier(P0_x, P1_x, P2_x, t);
    } else if (env->current_curve_direction == 1) { // Right curve
        // Right edge Bezier curve points
        float P0_x = ROAD_RIGHT_EDGE_X;
        float P1_x = SCREEN_WIDTH - 45;  // Keep original control point
        float P2_x = SCREEN_WIDTH - 40;  // End point at horizon
        x = quadratic_bezier(P0_x, P1_x, P2_x, t);
    } else { // Straight road
        float P0_x = ROAD_RIGHT_EDGE_X;
        float P2_x = VANISHING_POINT_X;
        x = P0_x + (P2_x - P0_x) * t;
    }
    return x;
}

// car_x_in_lane
float car_x_in_lane(Enduro* env, int lane, float y) {
    float left_edge = road_left_edge_x(env, y);
    float right_edge = road_right_edge_x(env, y);
    float lane_width = (right_edge - left_edge) / NUM_LANES;
    return left_edge + lane_width * (lane + 0.5f);
}


// Client functions
Client* make_client(Enduro* env, SDL_Renderer* renderer) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = env->width;
    client->height = env->height;
    client->player_color = (Color){255, 255, 255, 255}; // WHITE
    client->enemy_color = (Color){255, 0, 0, 255};      // RED
    client->road_color = (Color){0, 100, 0, 255};       // DARKGREEN

    client->renderer = renderer;

    return client;
}

void close_client(Client* client) {
    // Close SDL renderer if needed
    if (client->renderer) {
        SDL_DestroyRenderer(client->renderer);
    }
    free(client);
}

// Render car
void render_car(Client* client, Enduro* env) {
    SDL_Renderer* renderer = client->renderer;

    // Set the draw color to the player's color
    SDL_SetRenderDrawColor(renderer, client->player_color.r, client->player_color.g, client->player_color.b, client->player_color.a);

    // Render the player car based on dynamic player_x and player_y, centered correctly
    for (int i = 0; i < CAR_PIXELS_COUNT; i++) {
        int dx = car_pixels[i][0];
        int dy = car_pixels[i][1];

        // Calculate the actual position based on player's current coordinates
        int pixel_x = (int)(env->player_x + (dx - 77));
        int pixel_y = (int)(env->player_y + (dy - 144));

        // Draw the car's pixel at the calculated position
        SDL_RenderDrawPoint(renderer, pixel_x, pixel_y);
    }
}

// Event handling
void handleEvents(int* running, Enduro* env) {
    SDL_Event event;
    env->actions = 0; // Default action is noop
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) {
            *running = 0;
        } else if (event.type == SDL_KEYDOWN) {
            switch (event.key.keysym.sym) {
                case SDLK_LEFT:
                    env->actions = 1; // Move left
                    break;
                case SDLK_RIGHT:
                    env->actions = 2; // Move right
                    break;
                case SDLK_UP:
                    env->actions = 3; // Speed up
                    break;
                case SDLK_DOWN:
                    env->actions = 4; // Slow down
                    break;
                default:
                    env->actions = 0; // No action
                    break;
            }
        }
    }
}


int initSDL(SDL_Window** window, SDL_Renderer** renderer) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
        return -1;
    }

    // Create window with exact background size
    *window = SDL_CreateWindow("Enduro Port Framework",
                               SDL_WINDOWPOS_UNDEFINED,
                               SDL_WINDOWPOS_UNDEFINED,
                               SCREEN_WIDTH,
                               SCREEN_HEIGHT,
                               SDL_WINDOW_SHOWN);

    if (*window == NULL) {
        printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
        return -1;
    }

    // Create renderer without any scaling
    *renderer = SDL_CreateRenderer(*window, -1, SDL_RENDERER_ACCELERATED);

    if (*renderer == NULL) {
        printf("Renderer could not be created! SDL_Error: %s\n", SDL_GetError());
        return -1;
    }

    // Initialize PNG loading
    if (!(IMG_Init(IMG_INIT_PNG) & IMG_INIT_PNG)) {
        printf("SDL_image could not initialize! SDL_image Error: %s\n", IMG_GetError());
        return -1;
    }

    return 0;
}

void loadTextures(SDL_Renderer* renderer, GameState* gameState) {
    // Load background and mountain textures for different times of day
    char backgroundFile[40];
    char mountainFile[40];

    for (int i = 0; i < 16; ++i) {
        snprintf(backgroundFile, sizeof(backgroundFile), "resources/enduro_clone/%d_bg.png", i);
        SDL_Surface* bgSurface = IMG_Load(backgroundFile);
        if (!bgSurface) {
            printf("Failed to load background image %s! SDL_image Error: %s\n", backgroundFile, IMG_GetError());
            continue;
        }
        gameState->backgroundTextures[i] = SDL_CreateTextureFromSurface(renderer, bgSurface);
        SDL_FreeSurface(bgSurface);
        printf("Loaded background image: %s\n", backgroundFile);

        snprintf(mountainFile, sizeof(mountainFile), "resources/enduro_clone/%d_mtns.png", i);
        SDL_Surface* mtnSurface = IMG_Load(mountainFile);
        if (!mtnSurface) {
            printf("Failed to load mountain image %s! SDL_image Error: %s\n", mountainFile, IMG_GetError());
            continue;
        }
        gameState->mountainTextures[i] = SDL_CreateTextureFromSurface(renderer, mtnSurface);
        SDL_FreeSurface(mtnSurface);
        printf("Loaded mountain image: %s\n", mountainFile);
    }

    // Load digit textures 0-9
    char filename[100];
    for (int i = 0; i < 10; i++) {
        snprintf(filename, sizeof(filename), "resources/enduro_clone/digits_%d.png", i);
        SDL_Surface* tempSurface = IMG_Load(filename);
        if (!tempSurface) {
            fprintf(stderr, "Unable to load image %s! SDL Error: %s\n", filename, SDL_GetError());
            continue;
        }
        gameState->digitTextures[i] = SDL_CreateTextureFromSurface(renderer, tempSurface);
        SDL_FreeSurface(tempSurface);
        if (!gameState->digitTextures[i]) {
            fprintf(stderr, "Unable to create texture from %s! SDL Error: %s\n", filename, SDL_GetError());
        }
    }

    // Load the "CAR" digit texture
    SDL_Surface* carSurface = IMG_Load("resources/enduro_clone/digits_car.png");
    if (!carSurface) {
        printf("Failed to load digit image digits_car.png! SDL_image Error: %s\n", IMG_GetError());
    } else {
        gameState->carDigitTexture = SDL_CreateTextureFromSurface(renderer, carSurface);
        SDL_FreeSurface(carSurface);
        printf("Loaded digit image: digits_car.png\n");
    }

    // Load level complete flag textures
    SDL_Surface* flagLeftSurface = IMG_Load("resources/enduro_clone/level_complete_flag_left.png");
    if (!flagLeftSurface) {
        printf("Failed to load image level_complete_flag_left.png! SDL_image Error: %s\n", IMG_GetError());
    } else {
        gameState->levelCompleteFlagLeftTexture = SDL_CreateTextureFromSurface(renderer, flagLeftSurface);
        SDL_FreeSurface(flagLeftSurface);
        printf("Loaded image: level_complete_flag_left.png\n");
    }

    SDL_Surface* flagRightSurface = IMG_Load("resources/enduro_clone/level_complete_flag_right.png");
    if (!flagRightSurface) {
        printf("Failed to load image level_complete_flag_right.png! SDL_image Error: %s\n", IMG_GetError());
    } else {
        gameState->levelCompleteFlagRightTexture = SDL_CreateTextureFromSurface(renderer, flagRightSurface);
        SDL_FreeSurface(flagRightSurface);
        printf("Loaded image: level_complete_flag_right.png\n");
    }

    // Load green digits
    for (int i = 0; i < 10; ++i) {
        snprintf(filename, sizeof(filename), "resources/enduro_clone/green_digits_%d.png", i);
        SDL_Surface* surface = IMG_Load(filename);
        if (!surface) {
            printf("Failed to load image %s! SDL_image Error: %s\n", filename, IMG_GetError());
        } else {
            gameState->greenDigitTextures[i] = SDL_CreateTextureFromSurface(renderer, surface);
            SDL_FreeSurface(surface);
            printf("Loaded image: %s\n", filename);
        }
    }

    // Load yellow digits
    for (int i = 0; i < 10; ++i) {
        snprintf(filename, sizeof(filename), "resources/enduro_clone/yellow_digits_%d.png", i);
        SDL_Surface* surface = IMG_Load(filename);
        if (!surface) {
            printf("Failed to load image %s! SDL_image Error: %s\n", filename, IMG_GetError());
        } else {
            gameState->yellowDigitTextures[i] = SDL_CreateTextureFromSurface(renderer, surface);
            SDL_FreeSurface(surface);
            printf("Loaded image: %s\n", filename);
        }
    }

    // Initialize other game state variables
    gameState->currentBackgroundIndex = 0;
    gameState->score = 0;
    gameState->day = 1;
    gameState->carsToPass = 200;
    gameState->mountainPosition = 0.0f;
}

void cleanup(SDL_Window* window, SDL_Renderer* renderer, GameState* gameState) {
    // Destroy textures
    for (int i = 0; i < 16; ++i) {
        if (gameState->backgroundTextures[i]) {
            SDL_DestroyTexture(gameState->backgroundTextures[i]);
        }
        if (gameState->mountainTextures[i]) {
            SDL_DestroyTexture(gameState->mountainTextures[i]);
        }
    }

    for (int i = 0; i < 10; ++i) {
        if (gameState->digitTextures[i]) {
            SDL_DestroyTexture(gameState->digitTextures[i]);
        }
        if (gameState->greenDigitTextures[i]) {
            SDL_DestroyTexture(gameState->greenDigitTextures[i]);
        }
        if (gameState->yellowDigitTextures[i]) {
            SDL_DestroyTexture(gameState->yellowDigitTextures[i]);
        }
    }

    if (gameState->carDigitTexture) {
        SDL_DestroyTexture(gameState->carDigitTexture);
    }

    if (gameState->levelCompleteFlagLeftTexture) {
        SDL_DestroyTexture(gameState->levelCompleteFlagLeftTexture);
    }

    if (gameState->levelCompleteFlagRightTexture) {
        SDL_DestroyTexture(gameState->levelCompleteFlagRightTexture);
    }

    // Destroy renderer and window
    if (renderer) SDL_DestroyRenderer(renderer);
    if (window) SDL_DestroyWindow(window);

    // Quit SDL subsystems
    IMG_Quit();
    SDL_Quit();
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

void updateBackground(GameState* gameState, int timeOfDay) {
    gameState->previousBackgroundIndex = gameState->currentBackgroundIndex;
    gameState->currentBackgroundIndex = timeOfDay % 16;

    // Print the background and mountain images whenever they are displayed
    printf("Background image displayed: %d_bg.png\n", gameState->currentBackgroundIndex);
    printf("Mountain image displayed: %d_mtns.png\n", gameState->currentBackgroundIndex);

    // Check for victory condition
    if (!gameState->victoryAchieved && gameState->currentBackgroundIndex == 0 && gameState->previousBackgroundIndex == 15) {
        // Victory condition achieved
        gameState->victoryAchieved = true;
        gameState->carsToPass = 0; // Set cars to pass to 0
        printf("Victory achieved!\n");
    }
}


void renderBackground(SDL_Renderer* renderer, GameState* gameState) {
    SDL_Texture* bgTexture = gameState->backgroundTextures[gameState->currentBackgroundIndex];
    if (bgTexture) {
        // Render background at its native size without scaling
        SDL_RenderCopy(renderer, bgTexture, NULL, NULL);
    }
}

void renderScoreboard(SDL_Renderer* renderer, GameState* gameState) {
    // Positions and sizes
    int digitWidth = DIGIT_WIDTH;
    int digitHeight = DIGIT_HEIGHT;

    // Convert bottom-left coordinates to SDL coordinates (top-left origin)
    int scoreStartX = 56 + digitWidth;
    int scoreStartY = 173 - digitHeight;
    int dayX = 56;
    int dayY = 188 - digitHeight;
    int carsX = 72;
    int carsY = 188 - digitHeight;

    char scoreStr[6]; // Enough for the score plus null terminator
    sprintf(scoreStr, "%05d", gameState->score); // Format score with leading zeros

    // Render score with scrolling effect
    for (int i = 0; i < SCORE_DIGITS; ++i) {
        int digitX = scoreStartX + i * digitWidth;
        SDL_Rect destRect = { digitX, scoreStartY, digitWidth, digitHeight };

        SDL_Texture* currentDigitTexture = gameState->digitTextures[gameState->scoreDigitCurrents[i]];
        SDL_Texture* nextDigitTexture = gameState->digitTextures[gameState->scoreDigitNexts[i]];

        if (gameState->scoreDigitScrolling[i]) {
            // Scrolling effect for this digit
            float offset = gameState->scoreDigitOffsets[i];

            // Render current digit moving up
            SDL_Rect srcRectCurrent = { 0, 0, digitWidth, digitHeight - (int)offset };
            SDL_Rect destRectCurrent = { digitX, scoreStartY + (int)offset, digitWidth, digitHeight - (int)offset };
            SDL_RenderCopy(renderer, currentDigitTexture, &srcRectCurrent, &destRectCurrent);

            // Render next digit coming up from below
            SDL_Rect srcRectNext = { 0, digitHeight - (int)offset, digitWidth, (int)offset };
            SDL_Rect destRectNext = { digitX, scoreStartY, digitWidth, (int)offset };
            SDL_RenderCopy(renderer, nextDigitTexture, &srcRectNext, &destRectNext);

            printf("Rendering scrolling score digit: digits_%d.png and digits_%d.png at position (%d, %d)\n",
                   gameState->scoreDigitCurrents[i], gameState->scoreDigitNexts[i], destRect.x, destRect.y);
        } else {
            // No scrolling, render the current digit normally
            SDL_RenderCopy(renderer, currentDigitTexture, NULL, &destRect);
            printf("Rendering score digit: digits_%d.png at position (%d, %d)\n",
                   gameState->scoreDigitCurrents[i], destRect.x, destRect.y);
        }
    }

    // Render day number
    int day = gameState->day % 10;
    SDL_Rect dayRect = { dayX, dayY, digitWidth, digitHeight };

    if (gameState->victoryAchieved) {
        // Use green digits during victory
        SDL_Texture* greenDigitTexture = gameState->greenDigitTextures[day];
        SDL_RenderCopy(renderer, greenDigitTexture, NULL, &dayRect);
        printf("Rendering day digit: green_digits_%d.png at position (%d, %d)\n", day, dayRect.x, dayRect.y);
    } else {
        // Use normal digits
        SDL_RenderCopy(renderer, gameState->digitTextures[day], NULL, &dayRect);
        printf("Rendering day digit: digits_%d.png at position (%d, %d)\n", day, dayRect.x, dayRect.y);
    }

    // Render "CAR" digit or flags for cars to pass
    if (gameState->victoryAchieved) {
        // Alternate between level_complete_flag_left and level_complete_flag_right
        SDL_Texture* flagTexture = gameState->showLeftFlag ? gameState->levelCompleteFlagLeftTexture : gameState->levelCompleteFlagRightTexture;
        SDL_Rect flagRect = { carsX, carsY, digitWidth * 4, digitHeight };
        SDL_RenderCopy(renderer, flagTexture, NULL, &flagRect);
        printf("Rendering level complete flag: %s at position (%d, %d)\n",
               gameState->showLeftFlag ? "level_complete_flag_left.png" : "level_complete_flag_right.png", flagRect.x, flagRect.y);
    } else {
        // Render "CAR" digit for the first position in cars to pass
        SDL_Rect carDestRect = { carsX, carsY, digitWidth, digitHeight };
        SDL_RenderCopy(renderer, gameState->carDigitTexture, NULL, &carDestRect);
        printf("Rendering cars to pass digit: digits_car.png at position (%d, %d)\n", carDestRect.x, carDestRect.y);

        // Render the remaining digits for cars to pass
        int cars = gameState->carsToPass;
        for (int i = 1; i < CARS_DIGITS; ++i) {
            int digit = (cars / (int)pow(10, CARS_DIGITS - i - 1)) % 10;
            SDL_Rect destRect = { carsX + i * digitWidth + i * 1, carsY, digitWidth, digitHeight };
            SDL_RenderCopy(renderer, gameState->digitTextures[digit], NULL, &destRect);
            printf("Rendering cars to pass digit: digits_%d.png at position (%d, %d)\n", digit, destRect.x, destRect.y);
        }
    }
}


void updateVictoryEffects(GameState* gameState) {
    if (gameState->victoryAchieved) {
        // Update flag timer
        gameState->flagTimer++;
        if (gameState->flagTimer >= 30) { // Switch every 30 frames (~0.5 sec at 60 FPS)
            gameState->flagTimer = 0;
            gameState->showLeftFlag = !gameState->showLeftFlag;
        }

        // Update victory display timer
        gameState->victoryDisplayTimer++;
        if (gameState->victoryDisplayTimer >= 180) { // Display flags for 180 frames (~3 seconds)
            // Reset victory state
            gameState->victoryAchieved = false;
            gameState->victoryDisplayTimer = 0;

            // Increment day
            gameState->day += 1;
            if (gameState->day > 9) { // Assuming single-digit day numbers
                gameState->day = 1;
            }

            // Reset cars to pass for the new day
            gameState->carsToPass = 200; // Or set according to your game logic

            // Reset flags
            gameState->flagTimer = 0;
            gameState->showLeftFlag = true;

            printf("Starting new day: %d\n", gameState->day);
        }
    }
}


void updateMountains(GameState* gameState, RoadDirection direction) {
    // Adjust the mountain position based on the road direction
    float speed = 1.0f; // Adjust the speed as needed
    int mountainWidth = 100;

    if (direction == ROAD_TURN_LEFT) {
        gameState->mountainPosition += speed;
        if (gameState->mountainPosition >= mountainWidth) {
            gameState->mountainPosition -= mountainWidth;
        }
    } else if (direction == ROAD_TURN_RIGHT) {
        gameState->mountainPosition -= speed;
        if (gameState->mountainPosition <= -mountainWidth) {
            gameState->mountainPosition += mountainWidth;
        }
    }
    // If the road is straight, the mountains don't move
}

void renderMountains(SDL_Renderer* renderer, GameState* gameState) {
    SDL_Texture* mountainTexture = gameState->mountainTextures[gameState->currentBackgroundIndex];
    if (mountainTexture) {
        int mountainWidth = 100;
        int mountainHeight = 6;
        int mountainX = (int)gameState->mountainPosition + 37;
        int mountainY = 45; // Corrected Y-coordinate

        SDL_Rect destRect1 = { mountainX, mountainY, mountainWidth, mountainHeight };
        SDL_RenderCopy(renderer, mountainTexture, NULL, &destRect1);

        // Handle wrapping
        if (mountainX > SCREEN_WIDTH - mountainWidth) {
            SDL_Rect destRect2 = { mountainX - mountainWidth, mountainY, mountainWidth, mountainHeight };
            SDL_RenderCopy(renderer, mountainTexture, NULL, &destRect2);
        } else if (mountainX < 0) {
            SDL_Rect destRect2 = { mountainX + mountainWidth, mountainY, mountainWidth, mountainHeight };
            SDL_RenderCopy(renderer, mountainTexture, NULL, &destRect2);
        }
    }
}


// Inside the render function in enduro_clone.h
void render(Client* client, Enduro* env) {
    SDL_Renderer* renderer = client->renderer;

    // Set clipping rectangle to the playable area
    SDL_Rect clipRect = { PLAYABLE_AREA_LEFT, VANISHING_POINT_Y, PLAYABLE_AREA_RIGHT - PLAYABLE_AREA_LEFT, PLAYABLE_AREA_BOTTOM - VANISHING_POINT_Y };
    SDL_RenderSetClipRect(renderer, &clipRect);

    // // Render road
    // SDL_SetRenderDrawColor(renderer, 128, 128, 128, 255); // Gray color
    // for (float y = VANISHING_POINT_Y; y <= PLAYABLE_AREA_BOTTOM; y++) {
    //     float left_edge = road_left_edge_x(env, y);
    //     float right_edge = road_right_edge_x(env, y);
    //     SDL_RenderDrawLine(renderer, (int)left_edge, (int)y, (int)right_edge, (int)y);
    // }

    // Road edge lines
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255); // White color
    for (float y = VANISHING_POINT_Y; y <= PLAYABLE_AREA_BOTTOM; y += 0.75f) {
        float adjusted_y = (env->speed < 0) ? y : y + fmod(env->road_scroll_offset, 0.75f);
        if (adjusted_y > PLAYABLE_AREA_BOTTOM) continue;

        float left_edge = road_left_edge_x(env, adjusted_y);
        float right_edge = road_right_edge_x(env, adjusted_y);

        SDL_RenderDrawPoint(renderer, (int)left_edge, (int)adjusted_y);
        SDL_RenderDrawPoint(renderer, (int)right_edge, (int)adjusted_y);
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
        SDL_SetRenderDrawColor(renderer, client->enemy_color.r, client->enemy_color.g, client->enemy_color.b, client->enemy_color.a);
        for (int j = 0; j < CAR_PIXELS_COUNT; j++) {
            int dx = car_pixels[j][0] - 77; // Centering the x-offset
            int dy = car_pixels[j][1] - 144; // Centering the y-offset

            // Compute the actual pixel position with scaling
            float pixel_x = car_x + dx * scale;
            float pixel_y = car->y + dy * scale;

            SDL_RenderDrawPoint(renderer, (int)pixel_x, (int)pixel_y);
        }
    }

    // Render player car (no scaling since it's at the bottom)
    render_car(client, env);

    // Remove clipping
    SDL_RenderSetClipRect(renderer, NULL);

    // // Render borders
    // render_borders(client);

    // Render scoreboard with the correct GameState pointer
    renderScoreboard(renderer, &env->gameState);
}


#endif