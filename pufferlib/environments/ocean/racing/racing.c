// #include <stdlib.h>
// #include <stdio.h>
// #include <math.h>
// #include <stdbool.h>
// #include <string.h>
// #include "raylib.h"

// // Define screen dimensions for rendering
// #define TOTAL_SCREEN_WIDTH 160
// #define TOTAL_SCREEN_HEIGHT 210
// #define ACTION_SCREEN_X_START 8
// #define ACTION_SCREEN_Y_START 0
// #define ACTION_SCREEN_WIDTH 152  // from x=8 to x=160
// #define ACTION_SCREEN_HEIGHT 155 // from y=0 to y=155

// #define SCOREBOARD_X_START 48
// #define SCOREBOARD_Y_START 161
// #define SCOREBOARD_WIDTH 64  // from x=48 to x=112
// #define SCOREBOARD_HEIGHT 30 // from y=161 to y=191

// #define CARS_LEFT_X_START 72
// #define CARS_LEFT_Y_START 179
// #define CARS_LEFT_WIDTH 32  // from x=72 to x=104
// #define CARS_LEFT_HEIGHT 9  // from y=179 to y=188

// #define DAY_X_START 56
// #define DAY_Y_START 179
// #define DAY_WIDTH 8    // from x=56 to x=64
// #define DAY_HEIGHT 9   // from y=179 to y=188

// #define NUM_ENEMY_CARS 1
// #define ROAD_WIDTH 90.0f
// #define CAR_WIDTH 16.0f
// #define PLAYER_CAR_LENGTH 11.0f
// #define ENEMY_CAR_LENGTH 11.0f
// #define DESPAWN_DISTANCE 33.0f
// #define MAX_SPEED 100.0f
// #define MIN_SPEED -10.0f
// #define SPEED_INCREMENT 5.0f
// #define MAX_Y_POSITION ACTION_SCREEN_HEIGHT + ENEMY_CAR_LENGTH + DESPAWN_DISTANCE // Max Y for enemy cars
// #define MIN_Y_POSITION 0.0f // Min Y for enemy cars (spawn just above the screen)
// #define MIN_DISTANCE_BETWEEN_CARS 40.0f  // Minimum Y distance between adjacent enemy cars

// #define THROTTLE_MAX 120 // Maximum allowed throttle
// #define MY_YELLOW  (Color){ 255, 255, 0, 255 }  // RGBA for yellow
// #define MY_RED     (Color){ 255, 0, 0, 255 }    // RGBA for red
// #define DAY_COLOR (Color){ 255, 255, 255, 255 }  // RGBA for day color (white)

// typedef struct {
//     float rear_bumper_y;
//     int lane;
//     bool active;
// } EnemyCar;

// typedef struct {
//     float front_bumper_y;
//     float left_distance_to_edge;
//     float right_distance_to_edge;
//     float speed;

//     EnemyCar enemy_cars[NUM_ENEMY_CARS];
//     int dayNumber;
//     int carsRemainingLo;
//     int carsRemainingHi;
//     int throttleValue;
//     float reward;
//     bool done;
//     float *observations;
//     unsigned char *actions;
//     float *rewards;
//     unsigned char *dones;
// } EnduroEnv;

// // Action definitions
// #define ACTION_NOOP     0
// #define ACTION_ACCEL    1
// #define ACTION_DECEL    2
// #define ACTION_LEFT     3
// #define ACTION_RIGHT    4

// // Function prototypes
// void init_env(EnduroEnv *env);
// void reset_env(EnduroEnv *env);
// void step_env(EnduroEnv *env);
// void compute_observations(EnduroEnv *env);
// void free_env(EnduroEnv *env);
// void render_env(EnduroEnv *env);
// void render_hud(EnduroEnv *env);
// void render_background(EnduroEnv *env);
// void check_collision(EnduroEnv *env);



// void init_env(EnduroEnv *env) {
//     env->front_bumper_y = 0.0f;
//     env->left_distance_to_edge = (ROAD_WIDTH - CAR_WIDTH) / 2.0f;
//     env->right_distance_to_edge = (ROAD_WIDTH - CAR_WIDTH) / 2.0f;
//     env->speed = MIN_SPEED;

//     for (int i = 0; i < NUM_ENEMY_CARS; i++) {
//         env->enemy_cars[i].rear_bumper_y = rand() % (int)(MAX_Y_POSITION) + 100.0f;
//         env->enemy_cars[i].lane = rand() % 3; // Lanes: 0, 1, 2
//         env->enemy_cars[i].active = true;
//     }

//     env->reward = 0.0f;
//     env->done = false;
// }

// void reset_env(EnduroEnv *env) {
//     init_env(env);
//     compute_observations(env);
//     env->dones[0] = 0;
//     env->rewards[0] = 0.0f;
// }

// // Function to enforce enemy car spawn rules
// void spawn_enemy_cars(EnduroEnv *env) {
//     int num_active_cars = rand() % (NUM_ENEMY_CARS - 2) + 1;  // Allow 1-3 enemy cars

//     bool lanes[3] = { false, false, false };  // Track which lanes are occupied

//     for (int i = 0; i < NUM_ENEMY_CARS; i++) {
//         if (i < num_active_cars) {
//             int lane;
//             do {
//                 lane = rand() % 3;  // Randomly pick a lane
//             } while (lanes[lane]);  // Ensure no duplicates in the same lane

//             // Assign lane to car
//             env->enemy_cars[i].lane = lane;
//             lanes[lane] = true;  // Mark lane as occupied

//             // Assign vertical position and ensure minimum distance between adjacent lane cars
//             bool valid_position;
//             float rear_bumper_y;
//             do {
//                 rear_bumper_y = rand() % (int)(MAX_Y_POSITION) + 100.0f;
//                 valid_position = true;

//                 // Ensure cars in adjacent lanes have enough vertical space
//                 for (int j = 0; j < i; j++) {
//                     if (abs(env->enemy_cars[j].lane - lane) == 1 &&  // Adjacent lane check
//                         fabs(env->enemy_cars[j].rear_bumper_y - rear_bumper_y) < MIN_DISTANCE_BETWEEN_CARS) {
//                         valid_position = false;
//                         break;
//                     }
//                 }
//             } while (!valid_position);

//             env->enemy_cars[i].rear_bumper_y = rear_bumper_y;
//             env->enemy_cars[i].active = true;
//         } else {
//             env->enemy_cars[i].active = false;  // Deactivate extra cars
//         }
//     }
// }

// void step_env(EnduroEnv *env) {
//     // Read user input for controlling the car
//     if (IsKeyDown(KEY_UP)) {
//         env->actions[0] = ACTION_ACCEL;
//     } else if (IsKeyDown(KEY_DOWN)) {
//         env->actions[0] = ACTION_DECEL;
//     } else if (IsKeyDown(KEY_LEFT)) {
//         env->actions[0] = ACTION_LEFT;
//     } else if (IsKeyDown(KEY_RIGHT)) {
//         env->actions[0] = ACTION_RIGHT;
//     } else {
//         env->actions[0] = ACTION_NOOP;
//     }

//     // Update player's speed based on action
//     if (env->actions[0] == ACTION_ACCEL) {
//         env->speed += SPEED_INCREMENT;
//         if (env->speed > MAX_SPEED)
//             env->speed = MAX_SPEED;
//     } else if (env->actions[0] == ACTION_DECEL) {
//         env->speed -= SPEED_INCREMENT;
//         if (env->speed < MIN_SPEED)
//             env->speed = MIN_SPEED;
//     }

//     // Scale lateral movement speed with acceleration (max lateral movement = 5 pixels at max speed)
//     float lateral_speed = 1 + (env->speed / MAX_SPEED * 5);  // Scale between 1 and 5 pixels

//     // Update player's lateral position based on action
//     if (env->actions[0] == ACTION_LEFT) {
//         env->left_distance_to_edge -= lateral_speed;  // Move left, scaled by speed
//         env->right_distance_to_edge += lateral_speed;
//         if (env->left_distance_to_edge < 0) {
//             env->left_distance_to_edge = 0;  // Prevent moving past the left edge
//             env->right_distance_to_edge = ROAD_WIDTH - CAR_WIDTH;
//         }
//     } else if (env->actions[0] == ACTION_RIGHT) {
//         env->left_distance_to_edge += lateral_speed;  // Move right, scaled by speed
//         env->right_distance_to_edge -= lateral_speed;
//         if (env->right_distance_to_edge < 0) {
//             env->right_distance_to_edge = 0;  // Prevent moving past the right edge
//             env->left_distance_to_edge = ROAD_WIDTH - CAR_WIDTH;
//         }
//     }

//     // Update enemy cars
//     for (int i = 0; i < NUM_ENEMY_CARS; i++) {
//         if (env->enemy_cars[i].active) {
//             // Move enemy cars relative to player's speed
//             env->enemy_cars[i].rear_bumper_y -= env->speed * 0.1f;

//             // If enemy car passes behind the player, reset it
//             if (env->enemy_cars[i].rear_bumper_y < MIN_Y_POSITION) {
//                 env->enemy_cars[i].rear_bumper_y = MAX_Y_POSITION + rand() % 500;
//                 env->enemy_cars[i].lane = rand() % 3;
//             }

//             // Collision detection
//             float enemy_lane_center = (env->enemy_cars[i].lane + 0.5f) * (ROAD_WIDTH / 3.0f);
//             float enemy_left_edge = enemy_lane_center - (CAR_WIDTH / 2);
//             float enemy_right_edge = enemy_lane_center + (CAR_WIDTH / 2);
//             float player_left_edge = env->left_distance_to_edge;
//             float player_right_edge = ROAD_WIDTH - env->right_distance_to_edge;

//             bool lateral_overlap = !(player_right_edge <= enemy_left_edge || player_left_edge >= enemy_right_edge);
//             float enemy_front_bumper_y = env->enemy_cars[i].rear_bumper_y + ENEMY_CAR_LENGTH;
//             bool longitudinal_overlap = !(env->front_bumper_y + PLAYER_CAR_LENGTH <= env->enemy_cars[i].rear_bumper_y || env->front_bumper_y >= enemy_front_bumper_y);

//             if (lateral_overlap && longitudinal_overlap) {
//                 env->reward = -10.0f;
//                 env->rewards[0] = env->reward;
//                 env->done = true;
//                 env->dones[0] = 1;
//                 compute_observations(env);
//                 return;
//             }
//         }
//     }

//     // Update reward for moving forward
//     env->reward = env->speed * 0.01f;
//     env->rewards[0] = env->reward;

//     // Update observations
//     compute_observations(env);
// }



// void compute_observations(EnduroEnv *env) {
//     int obs_index = 0;
//     env->observations[obs_index++] = env->front_bumper_y;
//     env->observations[obs_index++] = env->left_distance_to_edge;
//     env->observations[obs_index++] = env->right_distance_to_edge;

//     for (int i = 0; i < NUM_ENEMY_CARS; i++) {
//         env->observations[obs_index++] = env->enemy_cars[i].rear_bumper_y;
//         env->observations[obs_index++] = (float)env->enemy_cars[i].lane;
//     }
// }

// void free_env(EnduroEnv *env) {
//     free(env->observations);
//     free(env->actions);
//     free(env->rewards);
//     free(env->dones);
// }

// void render_env(EnduroEnv *env) {
//     BeginDrawing();
    
//     // Define a dark green background color
//     Color darkGreen = { 0, 100, 0, 255 }; // RGB(0, 100, 0)
    
//     // Set the background color to dark green
//     ClearBackground(darkGreen);

//     // Draw road (centered, with side margins)
//     float lane_width = ROAD_WIDTH / 3.0f;
//     float road_center_x = ACTION_SCREEN_WIDTH / 2.0f;
//     float road_left_edge = road_center_x - ROAD_WIDTH / 2.0f;

//     // Draw the road in gray
//     DrawRectangle(road_left_edge, 0, ROAD_WIDTH, ACTION_SCREEN_HEIGHT, GRAY);
    
//     // Draw lane dividers
//     DrawLine(road_left_edge + lane_width, 0, road_left_edge + lane_width, ACTION_SCREEN_HEIGHT, WHITE);
//     DrawLine(road_left_edge + 2 * lane_width, 0, road_left_edge + 2 * lane_width, ACTION_SCREEN_HEIGHT, WHITE);
    
//     // Draw the player's car
//     float player_car_x = road_left_edge + env->left_distance_to_edge;
//     float player_car_y = ACTION_SCREEN_HEIGHT - 25; // Position near the bottom of the screen
    
//     DrawRectangle(player_car_x, player_car_y, CAR_WIDTH, PLAYER_CAR_LENGTH, BLUE);

//     // Draw enemy cars
//     for (int i = 0; i < NUM_ENEMY_CARS; i++) {
//         if (env->enemy_cars[i].active) {
//             float enemy_lane_center = road_left_edge + (env->enemy_cars[i].lane + 0.5f) * lane_width;
//             float enemy_car_x = enemy_lane_center - CAR_WIDTH / 2.0f;
//             float enemy_car_y = player_car_y - env->enemy_cars[i].rear_bumper_y;

//             // Render the enemy car
//             DrawRectangle(enemy_car_x, enemy_car_y, CAR_WIDTH, ENEMY_CAR_LENGTH, RED);
//         }
//     }

//     // render_background(env);
//     render_hud(env);

//     EndDrawing();
// }



// int main() {
//     InitWindow(ACTION_SCREEN_WIDTH, ACTION_SCREEN_HEIGHT, "Enduro Racing");
//     SetTargetFPS(60);

//     EnduroEnv env;
//     int obs_size = 3 + NUM_ENEMY_CARS * 2;
//     env.observations = (float *)malloc(sizeof(float) * obs_size);
//     env.actions = (unsigned char *)malloc(sizeof(unsigned char) * 1);
//     env.rewards = (float *)malloc(sizeof(float) * 1);
//     env.dones = (unsigned char *)malloc(sizeof(unsigned char) * 1);

//     reset_env(&env);

//     while (!WindowShouldClose()) {
//         env.actions[0] = rand() % 5;
//         step_env(&env);
//         render_env(&env);

//         if (env.dones[0]) {
//             reset_env(&env);
//         }
//     }

//     CloseWindow();
//     free_env(&env);

//     return 0;
// }

// void render_background(EnduroEnv *env) {
//     // Draw sky
//     DrawRectangle(ACTION_SCREEN_X_START, ACTION_SCREEN_Y_START, ACTION_SCREEN_WIDTH, 52, BLUE);
//     // Draw grass
//     DrawRectangle(ACTION_SCREEN_X_START, 52, ACTION_SCREEN_WIDTH, 103, DARKGREEN);
// }

// void render_hud(EnduroEnv *env) {
//     // Draw the scoreboard background
//     DrawRectangle(SCOREBOARD_X_START, SCOREBOARD_Y_START, SCOREBOARD_WIDTH, SCOREBOARD_HEIGHT, RED);

//     // Render the score
//     char score[6];
//     snprintf(score, sizeof(score), "%05d", env->throttleValue);  // Using throttle as a score placeholder
//     DrawText(score, 56, 162, 10, BLACK);

//     // Render cars left
//     char cars_left[5];
//     int totalCarsRemaining = (env->carsRemainingHi << 8) | env->carsRemainingLo;
//     snprintf(cars_left, sizeof(cars_left), "%d", totalCarsRemaining);
//     DrawText(cars_left, CARS_LEFT_X_START, CARS_LEFT_Y_START, 10, BLACK);

//     // Render current day
//     char day[2];
//     snprintf(day, sizeof(day), "%d", env->dayNumber);
//     DrawText(day, DAY_X_START, DAY_Y_START, 10, BLACK);
// }


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include "raylib.h"

// Define screen dimensions for rendering
#define TOTAL_SCREEN_WIDTH 160
#define TOTAL_SCREEN_HEIGHT 210
#define ACTION_SCREEN_X_START 8
#define ACTION_SCREEN_Y_START 0
#define ACTION_SCREEN_WIDTH 152  // from x=8 to x=160
#define ACTION_SCREEN_HEIGHT 155 // from y=0 to y=155
#define SCREEN_HEIGHT ACTION_SCREEN_HEIGHT // Set SCREEN_HEIGHT to be the same as ACTION_SCREEN_HEIGHT

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
#define MIN_SPEED -100.0f
#define SPEED_INCREMENT 5.0f
#define MAX_Y_POSITION ACTION_SCREEN_HEIGHT + ENEMY_CAR_LENGTH + ACTION_SCREEN_HEIGHT // Max Y for enemy cars
#define MIN_Y_POSITION 0.0f // Min Y for enemy cars (spawn just above the screen)
#define MIN_DISTANCE_BETWEEN_CARS 40.0f  // Minimum Y distance between adjacent enemy cars

#define THROTTLE_MAX 120 // Maximum allowed throttle
#define MY_YELLOW  (Color){ 255, 255, 0, 255 }  // RGBA for yellow
#define MY_RED     (Color){ 255, 0, 0, 255 }    // RGBA for red
#define DAY_COLOR (Color){ 255, 255, 255, 255 }  // RGBA for day color (white)

#define PASS_THRESHOLD ACTION_SCREEN_HEIGHT  // Distance for passed cars to disappear

typedef struct {
    float rear_bumper_y;
    int lane;
    float speed;  // Add speed for enemy cars
    bool active;
} EnemyCar;

typedef struct {
    float front_bumper_y;
    float left_distance_to_edge;
    float right_distance_to_edge;
    float speed;  // Add speed for player car

    EnemyCar enemy_cars[NUM_ENEMY_CARS];
    int dayNumber;
    int carsRemainingLo;
    int carsRemainingHi;
    int throttleValue;
    float reward;
    bool done;
    float *observations;
    unsigned char *actions;
    float *rewards;
    unsigned char *dones;
} EnduroEnv;

// Action definitions
#define ACTION_NOOP     0
#define ACTION_ACCEL    1
#define ACTION_DECEL    2
#define ACTION_LEFT     3
#define ACTION_RIGHT    4

// Function prototypes
void init_env(EnduroEnv *env);
void reset_env(EnduroEnv *env);
void step_env(EnduroEnv *env);
void compute_observations(EnduroEnv *env);
void free_env(EnduroEnv *env);
void render_env(EnduroEnv *env);
void render_hud(EnduroEnv *env);
void render_background(EnduroEnv *env);
void check_collision(EnduroEnv *env);
void apply_weather_conditions(EnduroEnv *env);  // Placeholder for future weather

// Function Definitions

void init_env(EnduroEnv *env) {
    env->front_bumper_y = 0.0f;
    env->left_distance_to_edge = (ROAD_WIDTH - CAR_WIDTH) / 2.0f;
    env->right_distance_to_edge = (ROAD_WIDTH - CAR_WIDTH) / 2.0f;
    env->speed = MIN_SPEED;  // Initialize player's speed

    for (int i = 0; i < NUM_ENEMY_CARS; i++) {
        env->enemy_cars[i].rear_bumper_y = rand() % (int)(MAX_Y_POSITION) - 100.0f;
        env->enemy_cars[i].lane = rand() % 3;  // Lanes: 0, 1, 2
        env->enemy_cars[i].speed = 1.0f + rand() % 10;  // Random speed for each enemy car
        env->enemy_cars[i].active = true;
    }

    env->reward = 0.0f;
    env->done = false;
    env->dayNumber = 0;  // Initialize the day number
    env->carsRemainingLo = 200;  // Initialize Cars Passed counter for Day 1
}

void reset_env(EnduroEnv *env) {
    init_env(env);
    compute_observations(env);
    env->dones[0] = 0;
    env->rewards[0] = 0.0f;
}



void step_env(EnduroEnv* env) {
    // Handle player control using arrow keys
    if (IsKeyDown(KEY_UP)) {
        env->speed = fmin(env->speed + SPEED_INCREMENT, MAX_SPEED);
    }
    if (IsKeyDown(KEY_DOWN)) {
        env->speed = fmax(env->speed - SPEED_INCREMENT, MIN_SPEED);
    }
    if (IsKeyDown(KEY_LEFT)) {
        env->left_distance_to_edge = fmax(env->left_distance_to_edge - 1.0f, 0);
        env->right_distance_to_edge = ROAD_WIDTH - CAR_WIDTH - env->left_distance_to_edge;
    }
    if (IsKeyDown(KEY_RIGHT)) {
        env->left_distance_to_edge = fmin(env->left_distance_to_edge + 1.0f, ROAD_WIDTH - CAR_WIDTH);
        env->right_distance_to_edge = ROAD_WIDTH - CAR_WIDTH - env->left_distance_to_edge;
    }

    // Update enemy cars
    for (int i = 0; i < NUM_ENEMY_CARS; i++) {
        if (env->enemy_cars[i].active) {
            env->enemy_cars[i].rear_bumper_y += env->enemy_cars[i].speed - env->speed * 0.1f;
            if (env->enemy_cars[i].rear_bumper_y > SCREEN_HEIGHT) {
                // Logic to respawn enemy car
                env->enemy_cars[i].rear_bumper_y = -ENEMY_CAR_LENGTH;
                env->enemy_cars[i].lane = rand() % 3;
                printf("Enemy car %d passed and respawned.\n", i);
            }

            // Collision detection
            float enemy_center_x = env->left_distance_to_edge + (env->enemy_cars[i].lane + 0.5f) * (ROAD_WIDTH / 3);
            float player_center_x = env->left_distance_to_edge + CAR_WIDTH / 2;
            bool lateral_overlap = fabs(enemy_center_x - player_center_x) < CAR_WIDTH;
            bool longitudinal_overlap = fabs(env->front_bumper_y - env->enemy_cars[i].rear_bumper_y) < PLAYER_CAR_LENGTH;

            if (lateral_overlap && longitudinal_overlap) {
                printf("Collision detected with enemy car %d at position Y: %.2f\n", i, env->enemy_cars[i].rear_bumper_y);
                env->speed = MIN_SPEED; // Reset speed on collision
                env->reward -= 10.0f; // Penalize collision
            }
        }
    }

    // Update reward based on current speed
    env->reward += env->speed * 0.01f;
    // Ensure player's car doesn't exit the screen
    env->front_bumper_y = fmax(0, fmin(env->front_bumper_y, SCREEN_HEIGHT - PLAYER_CAR_LENGTH));
}


void compute_observations(EnduroEnv *env) {
    int obs_index = 0;
    env->observations[obs_index++] = env->front_bumper_y;
    env->observations[obs_index++] = env->left_distance_to_edge;
    env->observations[obs_index++] = env->right_distance_to_edge;

    for (int i = 0; i < NUM_ENEMY_CARS; i++) {
        env->observations[obs_index++] = env->enemy_cars[i].rear_bumper_y;
        env->observations[obs_index++] = (float)env->enemy_cars[i].lane;
    }

    env->observations[obs_index++] = (float)(env->dayNumber);
    env->observations[obs_index++] = (float)((env->carsRemainingHi << 8) | env->carsRemainingLo);
}

void free_env(EnduroEnv *env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->dones);
}

void render_env(EnduroEnv *env) {
    BeginDrawing();

    // Define a dark green background color
    Color darkGreen = { 0, 100, 0, 255 };  // RGB(0, 100, 0)
    ClearBackground(darkGreen);

    // Draw the road in gray
    float lane_width = ROAD_WIDTH / 3.0f;
    float road_center_x = ACTION_SCREEN_WIDTH / 2.0f;
    float road_left_edge = road_center_x - ROAD_WIDTH / 2.0f;
    DrawRectangle(road_left_edge, 0, ROAD_WIDTH, ACTION_SCREEN_HEIGHT, GRAY);

    // Draw lane dividers
    DrawLine(road_left_edge + lane_width, 0, road_left_edge + lane_width, ACTION_SCREEN_HEIGHT, WHITE);
    DrawLine(road_left_edge + 2 * lane_width, 0, road_left_edge + 2 * lane_width, ACTION_SCREEN_HEIGHT, WHITE);

    // Draw the player's car
    float player_car_x = road_left_edge + env->left_distance_to_edge;
    float player_car_y = ACTION_SCREEN_HEIGHT - PLAYER_CAR_LENGTH;  // Player car stays at the bottom
    DrawRectangle(player_car_x, player_car_y, CAR_WIDTH, PLAYER_CAR_LENGTH, BLUE);

    // Draw enemy cars
    for (int i = 0; i < NUM_ENEMY_CARS; i++) {
        if (env->enemy_cars[i].active) {
            float enemy_lane_center = road_left_edge + (env->enemy_cars[i].lane + 0.5f) * lane_width;
            float enemy_car_x = enemy_lane_center - CAR_WIDTH / 2.0f;
            float enemy_car_y = env->enemy_cars[i].rear_bumper_y;

            // Render the enemy car directly from its current position (without modifying it)
            DrawRectangle(enemy_car_x, enemy_car_y, CAR_WIDTH, ENEMY_CAR_LENGTH, RED);
        }
    }

    render_hud(env);  // Render HUD elements like cars passed and day number

    EndDrawing();
}

void render_hud(EnduroEnv *env) {
    // Draw the scoreboard background
    DrawRectangle(SCOREBOARD_X_START, SCOREBOARD_Y_START, SCOREBOARD_WIDTH, SCOREBOARD_HEIGHT, RED);

    // Render the score
    char score[6];
    snprintf(score, sizeof(score), "%05d", env->throttleValue);  // Using throttle as a score placeholder
    DrawText(score, 56, 162, 10, BLACK);

    // Render cars left
    char cars_left[5];
    int totalCarsRemaining = (env->carsRemainingHi << 8) | env->carsRemainingLo;
    snprintf(cars_left, sizeof(cars_left), "%d", totalCarsRemaining);
    DrawText(cars_left, CARS_LEFT_X_START, CARS_LEFT_Y_START, 10, BLACK);

    // Render current day
    char day[2];
    snprintf(day, sizeof(day), "%d", env->dayNumber);
    DrawText(day, DAY_X_START, DAY_Y_START, 10, BLACK);
}

int main() {
    InitWindow(ACTION_SCREEN_WIDTH, ACTION_SCREEN_HEIGHT, "Enduro Racing");
    SetTargetFPS(60);

    EnduroEnv env;
    int obs_size = 3 + NUM_ENEMY_CARS * 2 + 2;  // Adjusted for additional observations
    env.observations = (float *)malloc(sizeof(float) * obs_size);
    env.actions = (unsigned char *)malloc(sizeof(unsigned char) * 1);
    env.rewards = (float *)malloc(sizeof(float) * 1);
    env.dones = (unsigned char *)malloc(sizeof(unsigned char) * 1);

    reset_env(&env);

    while (!WindowShouldClose()) {
        env.actions[0] = rand() % 5;
        step_env(&env);
        render_env(&env);

        if (env.dones[0]) {
            reset_env(&env);
        }
    }

    CloseWindow();
    free_env(&env);

    return 0;
}

void render_background(EnduroEnv *env) {
    // Draw sky
    DrawRectangle(ACTION_SCREEN_X_START, ACTION_SCREEN_Y_START, ACTION_SCREEN_WIDTH, 52, BLUE);
    // Draw grass
    DrawRectangle(ACTION_SCREEN_X_START, 52, ACTION_SCREEN_WIDTH, 103, DARKGREEN);
}
