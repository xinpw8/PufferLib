// enduro_clone.h
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include "raylib.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#define LOG_BUFFER_SIZE 1024
#define SCREEN_WIDTH 160
#define SCREEN_HEIGHT 210
#define HUD_HEIGHT 55
#define CAR_WIDTH 10
#define CAR_HEIGHT 10
#define MAX_ENEMIES 10
#define CRASH_NOOP_DURATION 60
#define DAY_LENGTH 2000
#define INITIAL_CARS_TO_PASS 5
#define ACTION_HEIGHT (SCREEN_HEIGHT - HUD_HEIGHT)

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
    int actions;  // Assuming actions is an int
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

    // New fields for road boundaries and lanes
    float road_left;
    float road_right;
    float road_width;
    float lane_centers[3];
    float screen_center_x;
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
    client->enemy_color = GREEN;
    client->road_color = DARKGREEN;

    InitWindow(client->width, client->height, "Enduro Clone");
    SetTargetFPS(60); // Adjust as needed
    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

// Initialize the Enduro environment
void init(Enduro* env) {
    // Initialize step-related fields
    env->step_count = 0;
    env->score = 0;
    env->numEnemies = 0;
    env->collision_cooldown = 0.0;
    
    // Calculate the action height (playable area height)
    env->action_height = env->height - env->hud_height;

    env->screen_center_x = SCREEN_WIDTH / 2.0f;

    // Road boundaries
    env->road_left = env->width / 3.0f;
    env->road_right = 2.0f * env->width / 3.0f;
    env->road_width = env->road_right - env->road_left;

    // Compute lane centers at the bottom (initial positions)
    float lane_width = env->road_width / 3.0f;
    for (int i = 0; i < 3; i++) {
        env->lane_centers[i] = env->road_left + (i + 0.5f) * lane_width;
    }

    // Initialize player car position (centered in middle lane)
    env->player_x = env->lane_centers[1] - CAR_WIDTH / 2.0f;  // Middle lane
    env->player_y = env->action_height - env->car_height - 10;  // Moved up by 10 pixels

    // Initialize speed-related fields
    env->min_speed = 1.0f;
    env->max_speed = 5.0f;
    env->speed = env->min_speed;
    
    // Set cars to pass and start at day 1
    env->carsToPass = env->initial_cars_to_pass;
    env->day = 1;

    // Initialize all enemy cars (their initial positions and state)
    for (int i = 0; i < MAX_ENEMIES; i++) {
        env->enemyCars[i].lane = 0;
        env->enemyCars[i].y = 0.0;
        env->enemyCars[i].passed = 0;  // Reset passed flag
    }

    // Initialize the log buffer (this should be already allocated by the caller)
    if (env->log_buffer != NULL) {
        env->log_buffer->idx = 0;  // Reset log buffer index
    }

    // Initialize log (this resets episode-specific values)
    env->log.episode_return = 0.0;
    env->log.episode_length = 0.0;
    env->log.score = 0.0;
}


// Allocate memory for Enduro environment
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

    printf("[DEBUG] Memory successfully allocated for observations, actions, and rewards\n");
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->truncateds = (unsigned char*)calloc(1, sizeof(unsigned char));
    
    env->log_buffer = allocate_logbuffer(LOG_BUFFER_SIZE);

    // Initialize random seed
    srand(time(NULL));
}

// Free the allocated memory for Enduro environment
void free_allocated(Enduro* env) {
    free(env->observations);
    // free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env->truncateds);
    free_logbuffer(env->log_buffer);
}

float road_left_edge_x(Enduro* env, float y) {
    return env->road_left + ((SCREEN_WIDTH / 2 - env->road_left) * (ACTION_HEIGHT - y) / ACTION_HEIGHT);
}

float road_right_edge_x(Enduro* env, float y) {
    return env->road_right + ((SCREEN_WIDTH / 2 - env->road_right) * (ACTION_HEIGHT - y) / ACTION_HEIGHT);
}

float lane_center_x(Enduro* env, int lane_index, float y) {
    float left_edge = road_left_edge_x(env, y);
    float right_edge = road_right_edge_x(env, y);
    float lane_width = (right_edge - left_edge) / 3.0f;
    return left_edge + lane_width * (lane_index + 0.5f);
}


void add_enemy_car(Enduro* env) {
    if (env->numEnemies >= MAX_ENEMIES) return;
    int lane = rand() % 3;
    Car car = { .lane = lane, .y = 0.0f, .passed = false };
    env->enemyCars[env->numEnemies++] = car;
}


void print_observations_to_file(Enduro* env, const char* filename) {
    // Open the file in append mode ("a")
    FILE* file = fopen(filename, "a");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        perror("fopen");
        return;
    }
    int obs_idx = 8;
    for (int i = 0; i < MAX_ENEMIES; i++) {
        float car_x = lane_center_x(env, env->enemyCars[i].lane, env->enemyCars[i].y);
        fprintf(file, "%f ", car_x / SCREEN_WIDTH);
        fprintf(file, "%f ", env->enemyCars[i].y / ACTION_HEIGHT);
    }
    fprintf(file, "\n");
    fclose(file);
}

// Compute game state observations
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
        // Compute the car's x position dynamically
        float car_x = lane_center_x(env, env->enemyCars[i].lane, env->enemyCars[i].y);
        env->observations[obs_idx++] = car_x / SCREEN_WIDTH;
        env->observations[obs_idx++] = env->enemyCars[i].y / ACTION_HEIGHT;
    }
}


// Reset the round in the Enduro environment
void reset_round(Enduro* env) {
    // Initialize player car position (centered in middle lane)
    env->player_x = env->lane_centers[1] - CAR_WIDTH / 2.0f;  // Middle lane
    env->player_y = env->action_height - env->car_height - 10;  // Moved up by 10 pixels

    env->score = 0;
    env->carsToPass = INITIAL_CARS_TO_PASS;
    env->speed = env->min_speed;
    env->numEnemies = 0;
    env->step_count = 0;
    env->collision_cooldown = 0;
}

// Reset the entire environment
void reset(Enduro* env) {
    env->log = (Log){0};
    reset_round(env);
    compute_observations(env);
}


// Check collision between player and enemy car
bool check_collision(Enduro* env, Car* car) {
    // Calculate car's x position
    float scale = (car->y + CAR_HEIGHT) / (ACTION_HEIGHT + CAR_HEIGHT);    float car_width_scaled = CAR_WIDTH * scale;
    float car_x = lane_center_x(env, car->lane, car->y) - car_width_scaled / 2.0f;
    return !(env->player_x > car_x + CAR_WIDTH || 
             env->player_x + CAR_WIDTH < car_x || 
             env->player_y > car->y + CAR_HEIGHT || 
             env->player_y + CAR_HEIGHT < car->y);
}

// Helper function to write logs to a file
void log_to_file(const char* filename, const char* format, ...) {
    FILE* file = fopen(filename, "a");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        perror("fopen");
        return;
    }

    va_list args;
    va_start(args, format);
    vfprintf(file, format, args);
    va_end(args);

    fflush(file);  // Ensure logs are flushed immediately
    fclose(file);
}

// debugging step function
void step(Enduro* env) {
    // Basic check to ensure 'env' is not NULL
    if (env == NULL) {
        printf("[ERROR] env is NULL! Aborting step.\n");
        return;
    }

    // Log the beginning of the step
    printf("[DEBUG] Stepping: player_x = %f, player_y = %f, speed = %f, numEnemies = %d\n", 
           env->player_x, env->player_y, env->speed, env->numEnemies);

    // Check if `observations` array is properly initialized
    if (env->observations == NULL) {
        printf("[ERROR] env->observations is NULL!\n");
        return;
    }
    
    // Check if `rewards` array is properly initialized
    if (env->rewards == NULL) {
        printf("[ERROR] env->rewards is NULL!\n");
        return;
    }
    
    // Check if `enemyCars` array is properly initialized and the number of enemies is valid
    if (env->enemyCars == NULL && env->numEnemies > 0) {
        printf("[ERROR] env->enemyCars is NULL but numEnemies > 0!\n");
        return;
    }
    if (env->numEnemies > MAX_ENEMIES) {
        printf("[ERROR] env->numEnemies (%d) exceeds MAX_ENEMIES (%d)! Aborting step.\n", env->numEnemies, MAX_ENEMIES);
        return;
    }

    // Log step start
    log_to_file("game_debug.log", "Observations[0]: %f\n", env->observations[0]);


    env->log.episode_length += 1;
    env->terminals[0] = 0;

    // Handle cooldown if active
    if (env->collision_cooldown > 0) {
        env->collision_cooldown -= 1;
        // log_to_file(log_filename, "Collision Cooldown: %f\n", env->collision_cooldown);
    }

    // Player movement logic
    if (env->collision_cooldown == 0) {
        int act = env->actions;  // Assuming 'actions' is an 'int'
        printf("[DEBUG] Player action: %d\n", act);  // Log player's action
        if (act == 1) {  // Move left
            env->player_x -= 5;
            if (env->player_x < env->road_left) env->player_x = env->road_left;
        }
        if (act == 2) {  // Move right
            env->player_x += 5;
            if (env->player_x > env->road_right - CAR_WIDTH) env->player_x = env->road_right - CAR_WIDTH;
        }
        if (act == 3 && env->speed < env->max_speed) env->speed += 0.1f;
        if (act == 4 && env->speed > env->min_speed) env->speed -= 0.1f;
    }

    // Log player's position and action
    printf("[DEBUG] actions: %d, player_x: %f, player_y: %f\n", env->actions, env->player_x, env->player_y);

    // Ensure the number of enemies does not exceed the limit
    if (env->numEnemies >= MAX_ENEMIES) {
        printf("[DEBUG] Enemy car limit reached: %d enemies.\n", env->numEnemies);
        return;
    }

    // Enemy car logic and collision handling
    for (int i = 0; i < env->numEnemies; i++) {
        Car* car = &env->enemyCars[i];

        // Compute car's x position
        float car_x = lane_center_x(env, car->lane, car->y);

        printf("[DEBUG] Processing enemy car %d: x = %f, y = %f\n", i, car_x, car->y);

        // Update enemy car position
        car->y += env->speed;

        // Check if the car has been passed by the player
        if (car->y > env->player_y && !car->passed) {
            env->carsToPass--;
            car->passed = true;
            env->score++;
            env->rewards[0] += 1;
        }

        // Check for collisions between the player and the car
        if (check_collision(env, car)) {
            env->speed = env->min_speed;
            env->collision_cooldown = CRASH_NOOP_DURATION;
            printf("[DEBUG] Collision detected with car %d. Cooldown: %f\n", i, env->collision_cooldown);
        }

        // Remove off-screen cars
        if (car->y > ACTION_HEIGHT) {
            printf("[DEBUG] Car %d went off-screen. Removing car.\n", i);
            for (int j = i; j < env->numEnemies - 1; j++) {
                env->enemyCars[j] = env->enemyCars[j + 1];
            }
            env->numEnemies--;
            i--;  // Adjust loop index to account for removed car
        }
    }

    // Add new enemy cars based on random conditions (currently 1% chance)
    if (env->numEnemies < MAX_ENEMIES && rand() % 100 < 1) {
        add_enemy_car(env);
        // log_to_file(log_filename, "New enemy car added. Total Enemies: %d\n", env->numEnemies);
        printf("[DEBUG] New enemy car added. Total enemies: %d\n", env->numEnemies);
    }

    // Handle day completion logic
    if (env->carsToPass <= 0) {
        env->day++;
        env->carsToPass = env->day * 10 + 10;
        env->speed += 0.1f;

        // Log day completion
        // log_to_file(log_filename, "Day %d completed! New Cars to Pass: %d\n", env->day, env->carsToPass);
        add_log(env->log_buffer, &env->log);

        // Reset rewards after day completes
        env->rewards[0] = 0;
    } else if (env->step_count >= env->day_length) {
        if (env->carsToPass > 0) {
            env->terminals[0] = 1;  // Mark game as over
            // log_to_file(log_filename, "Game Over! Resetting...\n");
            add_log(env->log_buffer, &env->log);
            reset(env);
            return;
        }
    }

    // Accumulate rewards for the episode
    env->log.episode_return += env->rewards[0];

    // Compute and log observations
    // print_observations_to_file(env, "observations.log");
    printf("[DEBUG] Step %d complete. Rewards: %f\n", env->step_count, env->rewards[0]);
    
    // Log final rewards for this step
    // log_to_file(log_filename, "Final Rewards for Step %d: %f\n", env->step_count, env->rewards[0]);

    // Prepare for the next step
    env->step_count++;
    env->log.score = env->score;
    // log_to_file(log_filename, "--- Step %d Complete ---\n", env->step_count);
}

void render(Client* client, Enduro* env) {
    BeginDrawing();
    ClearBackground(client->road_color);

    // Render road (simplified perspective)
    DrawTriangle((Vector2){ env->road_left, ACTION_HEIGHT }, 
                 (Vector2){ env->road_right, ACTION_HEIGHT }, 
                 (Vector2){ SCREEN_WIDTH / 2.0f, 0 }, GRAY);

    // Optionally, draw lane lines for visual aid
    float lane_width = env->road_width / 3.0f;
    for (int i = 1; i < 3; i++) {
        float lane_x1 = env->road_left + i * lane_width;
        // Draw lines with perspective (from bottom to top center)
        DrawLine(lane_x1, ACTION_HEIGHT, SCREEN_WIDTH / 2.0f, 0, WHITE);
    }

    // Render enemy cars with size adjustment
    for (int i = 0; i < env->numEnemies; i++) {
        Car* car = &env->enemyCars[i];

        // Calculate scaling factor based on y position
        float scale = (car->y + CAR_HEIGHT) / (ACTION_HEIGHT + CAR_HEIGHT);
        if (scale < 0.2f) scale = 0.2f; // Prevent cars from becoming too small

        // Adjust the size of the car
        float car_width_scaled = CAR_WIDTH * scale;
        float car_height_scaled = CAR_HEIGHT * scale;

        // Calculate car's x position
        float car_x = lane_center_x(env, car->lane, car->y) - car_width_scaled / 2.0f;
        
        // Adjust x and y positions to keep the car centered in the lane
        float car_x_adjusted = car_x + (CAR_WIDTH - car_width_scaled) / 2.0f;
        float car_y_adjusted = car->y;

        // Draw the enemy car
        DrawRectangle(car_x_adjusted, car_y_adjusted, car_width_scaled, car_height_scaled, client->enemy_color);
    }

    // Render player car (keep size constant)
    DrawRectangle(env->player_x, env->player_y, CAR_WIDTH, CAR_HEIGHT, client->player_color);

    // Render HUD env data
    DrawText(TextFormat("Score: %05i", env->score), 56, ACTION_HEIGHT + 10, 10, WHITE);
    DrawText(TextFormat("Cars to Pass: %03i", env->carsToPass), 56, ACTION_HEIGHT + 25, 10, WHITE);
    DrawText(TextFormat("Day: %i", env->day), 56, ACTION_HEIGHT + 40, 10, WHITE);
    DrawText(TextFormat("Speed: %.2f", env->speed), 56, ACTION_HEIGHT + 55, 10, WHITE);
    DrawText(TextFormat("Step: %i", env->step_count), 56, ACTION_HEIGHT + 70, 10, WHITE);
    // Box around HUD
    DrawRectangleLines(0, ACTION_HEIGHT, SCREEN_WIDTH, HUD_HEIGHT, WHITE);

    EndDrawing();
}














// // enduro_clone.h
// #include <stdlib.h>
// #include <stdbool.h>
// #include <math.h>
// #include <stdio.h>
// #include <unistd.h>
// #include "raylib.h"

// #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// #define LOG_BUFFER_SIZE 1024
// #define SCREEN_WIDTH 160
// #define SCREEN_HEIGHT 210
// #define HUD_HEIGHT 55
// #define CAR_WIDTH 10
// #define CAR_HEIGHT 10
// #define MAX_ENEMIES 10
// #define CRASH_NOOP_DURATION 60
// #define DAY_LENGTH 2000
// #define INITIAL_CARS_TO_PASS 5
// #define ACTION_HEIGHT (SCREEN_HEIGHT - HUD_HEIGHT)

// typedef struct Log Log;
// struct Log {
//     float episode_return;
//     float episode_length;
//     float score;
// };

// typedef struct LogBuffer LogBuffer;
// struct LogBuffer {
//     Log* logs;
//     int length;
//     int idx;
// };

// LogBuffer* allocate_logbuffer(int size) {
//     LogBuffer* logs = (LogBuffer*)calloc(1, sizeof(LogBuffer));
//     logs->logs = (Log*)calloc(size, sizeof(Log));
//     logs->length = size;
//     logs->idx = 0;
//     return logs;
// }

// void free_logbuffer(LogBuffer* buffer) {
//     free(buffer->logs);
//     free(buffer);
// }

// void add_log(LogBuffer* logs, Log* log) {
//     if (logs->idx == logs->length) {
//         return;
//     }
//     logs->logs[logs->idx] = *log;
//     logs->idx += 1;
//     printf("Log: %f, %f, %f\n", log->episode_return, log->episode_length, log->score);
// }

// Log aggregate_and_clear(LogBuffer* logs) {
//     Log log = {0};
//     if (logs->idx == 0) {
//         return log;
//     }
//     for (int i = 0; i < logs->idx; i++) {
//         log.episode_return += logs->logs[i].episode_return;
//         log.episode_length += logs->logs[i].episode_length;
//         log.score += logs->logs[i].score;
//     }
//     log.episode_return /= logs->idx;
//     log.episode_length /= logs->idx;
//     log.score /= logs->idx;
//     logs->idx = 0;
//     return log;
// }

// // Car structure for enemy cars
// typedef struct Car Car;
// struct Car {
//     float x, y;
//     int passed;
// };

// // Game environment structure
// typedef struct Enduro Enduro;
// struct Enduro {
//     float* observations;
//     int* actions;
//     float* rewards;
//     unsigned char* terminals;
//     LogBuffer* log_buffer;
//     Log log;

//     float width;
//     float height;
//     float hud_height;
//     float car_width;
//     float car_height;
//     int max_enemies;
//     float crash_noop_duration;
//     float day_length;
//     int initial_cars_to_pass;
//     float min_speed;
//     float max_speed;

//     float player_x;
//     float player_y;
//     float speed;
    
//     // ints
//     int score;
//     int day;
//     int step_count;
//     int numEnemies;
//     int carsToPass;
    
//     float collision_cooldown;
//     float action_height;

//     Car enemyCars[MAX_ENEMIES];

// };

// // Client structure for rendering and input handling
// typedef struct Client Client;
// struct Client {
//     float width;
//     float height;
//     Color player_color;
//     Color enemy_color;
//     Color road_color;
// };

// Client* make_client(Enduro* env) {
//     Client* client = (Client*)calloc(1, sizeof(Client));
//     client->width = env->width;
//     client->height = env->height;
//     client->player_color = WHITE;
//     client->enemy_color = GREEN;
//     client->road_color = DARKGREEN;

//     InitWindow(client->width, client->height, "Enduro Clone");
//     SetTargetFPS(60); // Adjust as needed
//     return client;
// }

// void close_client(Client* client) {
//     CloseWindow();
//     free(client);
// }

// // Initialize the Enduro environment
// void init(Enduro* env) {
//     // Initialize step-related fields
//     env->step_count = 0;
//     env->score = 0;
//     env->numEnemies = 0;
//     env->collision_cooldown = 0.0;
    
//     // Calculate the action height (playable area height)
//     env->action_height = env->height - env->hud_height;

//     // Initialize player car position (centered horizontally, near the bottom vertically)
//     env->player_x = env->width / 2 - env->car_width / 2;
//     env->player_y = env->action_height - env->car_height;

//     // Initialize speed-related fields
//     env->speed = env->min_speed;
    
//     // Set cars to pass and start at day 1
//     env->carsToPass = env->initial_cars_to_pass;
//     env->day = 1;

//     // Initialize all enemy cars (their initial positions and state)
//     for (int i = 0; i < MAX_ENEMIES; i++) {
//         env->enemyCars[i].x = 0.0;
//         env->enemyCars[i].y = 0.0;
//         env->enemyCars[i].passed = 0;  // Reset passed flag
//     }

//     // Initialize the log buffer (this should be already allocated by the caller)
//     if (env->log_buffer != NULL) {
//         env->log_buffer->idx = 0;  // Reset log buffer index
//     }

//     // Initialize log (this resets episode-specific values)
//     env->log.episode_return = 0.0;
//     env->log.episode_length = 0.0;
//     env->log.score = 0.0;
// }

// // Allocate memory for Enduro environment
// void allocate(Enduro* env) {
//     init(env);
//     env->observations = (float*)calloc(8 + 2 * MAX_ENEMIES, sizeof(float));
//     if (env->observations == NULL) {
//         printf("[ERROR] Memory allocation for env->observations failed!\n");
//         return;
//     }
//     env->actions = (int*)calloc(5, sizeof(int));
//     if (env->actions == NULL) {
//         printf("[ERROR] Memory allocation for env->actions failed!\n");
//         free(env->observations);  // Free previously allocated memory before returning
//         return;
//     }
//     env->rewards = (float*)calloc(1, sizeof(float));
//     if (env->rewards == NULL) {
//         printf("[ERROR] Memory allocation for env->rewards failed!\n");
//         free(env->observations);
//         free(env->actions);
//         return;
//     }

//     printf("[DEBUG] Memory successfully allocated for observations, actions, and rewards\n");
//     env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    
//     env->log_buffer = allocate_logbuffer(LOG_BUFFER_SIZE);
// }

// // Free the allocated memory for Enduro environment
// void free_allocated(Enduro* env) {
//     free(env->observations);
//     free(env->actions);
//     free(env->rewards);
//     free(env->terminals);
//     free_logbuffer(env->log_buffer);
// }

// void print_observations_to_file(Enduro* env, const char* filename) {
//     // Open the file in append mode ("a")
//     FILE* file = fopen(filename, "a");
//     if (file == NULL) {
//         fprintf(stderr, "Error opening file: %s\n", filename);
//         perror("fopen");
//         return;
//     }
// }


// // Add new enemy cars for player to pass
// void add_enemy_car(Enduro* env) {
//     if (env->numEnemies >= MAX_ENEMIES) return;
//     Car car = { .x = rand() % (SCREEN_WIDTH - CAR_WIDTH), .y = -CAR_HEIGHT, .passed = false };
//     env->enemyCars[env->numEnemies++] = car;
// }


// // Compute game state observations
// void compute_observations(Enduro* env) {
//     env->observations[0] = env->player_x / SCREEN_WIDTH;
//     env->observations[1] = env->player_y / ACTION_HEIGHT;
//     env->observations[2] = env->speed / env->max_speed;
//     env->observations[3] = env->carsToPass;
//     env->observations[4] = env->day;
//     env->observations[5] = env->numEnemies;
//     env->observations[6] = env->collision_cooldown;
//     env->observations[7] = env->score;

//     int obs_idx = 8;
//     for (int i = 0; i < MAX_ENEMIES; i++) {
//         env->observations[obs_idx++] = env->enemyCars[i].x / SCREEN_WIDTH;
//         env->observations[obs_idx++] = env->enemyCars[i].y / ACTION_HEIGHT;
//     }

// }


// // Reset the round in the Enduro environment
// void reset_round(Enduro* env) {
//     env->player_x = SCREEN_WIDTH / 2 - CAR_WIDTH / 2;
//     env->player_y = ACTION_HEIGHT - CAR_HEIGHT;
//     env->score = 0;
//     env->carsToPass = INITIAL_CARS_TO_PASS;
//     env->speed = 1.0;
//     env->numEnemies = 0;
//     env->step_count = 0;
//     env->collision_cooldown = 0;
// }

// // Reset the entire environment
// void reset(Enduro* env) {
//     env->log = (Log){0};
//     reset_round(env);
//     // compute_observations(env);
// }


// // Check collision between player and enemy car
// bool check_collision(Enduro* env, Car* car) {
//     return !(env->player_x > car->x + CAR_WIDTH || 
//              env->player_x + CAR_WIDTH < car->x || 
//              env->player_y > car->y + CAR_HEIGHT || 
//              env->player_y + CAR_HEIGHT < car->y);
// }


// // Helper function to write logs to a file
// void log_to_file(const char* filename, const char* format, ...) {
//     FILE* file = fopen(filename, "a");
//     if (file == NULL) {
//         fprintf(stderr, "Error opening file: %s\n", filename);
//         perror("fopen");
//         return;
//     }

//     va_list args;
//     va_start(args, format);
//     vfprintf(file, format, args);
//     va_end(args);

//     fclose(file);
// }


// // debugging step function
// void step(Enduro* env) {

//     // Basic check to ensure 'env' is not NULL
//     if (env == NULL) {
//         printf("[ERROR] env is NULL! Aborting step.\n");
//         return;
//     }

//     // Log the beginning of the step
//     printf("[DEBUG] Stepping: player_x = %f, player_y = %f, speed = %f, numEnemies = %d\n", 
//            env->player_x, env->player_y, env->speed, env->numEnemies);

//     // Check if `observations` array is properly initialized
//     if (env->observations == NULL) {
//         printf("[ERROR] env->observations is NULL!\n");
//         return;
//     }
    
//     // Check if `rewards` array is properly initialized
//     if (env->rewards == NULL) {
//         printf("[ERROR] env->rewards is NULL!\n");
//         return;
//     }
    
//     // Check if `actions` array is properly initialized
//     if (env->actions == NULL) {
//         printf("[ERROR] env->actions is NULL!\n");
//         return;
//     }

//     // Check if `enemyCars` array is properly initialized and the number of enemies is valid
//     if (env->enemyCars == NULL && env->numEnemies > 0) {
//         printf("[ERROR] env->enemyCars is NULL but numEnemies > 0!\n");
//         return;
//     }
//     if (env->numEnemies > MAX_ENEMIES) {
//         printf("[ERROR] env->numEnemies (%d) exceeds MAX_ENEMIES (%d)! Aborting step.\n", env->numEnemies, MAX_ENEMIES);
//         return;
//     }

//     // Log step start
//     log_to_file("game_debug.log", "Observations[0]: %f\n", env->observations[0]);

//     const char* log_filename = "game_debug.log";
//     // log_to_file(log_filename, "\n--- Step %d Start ---\n", env->step_count);
//     // log_to_file(log_filename, "Episode Length: %f, Reward: %f, Terminal: %d\n",
//                 env->log.episode_length, env->rewards[0], env->terminals[0]);

//     env->log.episode_length += 1;
//     env->terminals[0] = 0;

//     // Handle cooldown if active
//     if (env->collision_cooldown > 0) {
//         env->collision_cooldown -= 1;
//         // log_to_file(log_filename, "Collision Cooldown: %f\n", env->collision_cooldown);
//     }

//     // Player movement logic
//     if (env->collision_cooldown == 0) {
//         int act = env->actions[0];
//         printf("[DEBUG] Player action: %d\n", act);  // Log player's action
//         if (act == 1 && env->player_x > 0) env->player_x -= 5;
//         if (act == 2 && env->player_x < SCREEN_WIDTH - CAR_WIDTH) env->player_x += 5;
//         if (act == 3 && env->speed < env->max_speed) env->speed += 0.1f;
//         if (act == 4 && env->speed > env->min_speed) env->speed -= 0.1f;
//     }

//     // Log player's position and action
//     printf("[DEBUG] Actions[0]: %d, player_x: %f, player_y: %f\n", env->actions[0], env->player_x, env->player_y);

//     // Ensure the number of enemies does not exceed the limit
//     if (env->numEnemies >= MAX_ENEMIES) {
//         printf("[DEBUG] Enemy car limit reached: %d enemies.\n", env->numEnemies);
//         return;
//     }

//     // Enemy car logic and collision handling
//     for (int i = 0; i < env->numEnemies; i++) {
//         Car* car = &env->enemyCars[i];
//         printf("[DEBUG] Processing enemy car %d: x = %f, y = %f\n", i, car->x, car->y);

//         // Update enemy car position
//         car->y += env->speed;

//         // Check if the car has been passed by the player
//         if (car->y > env->player_y && !car->passed) {
//             env->carsToPass--;
//             car->passed = true;
//             env->score++;
//             env->rewards[0] += 1;

//             // Log car being passed
//             // log_to_file(log_filename, 
//                 "Car %d passed! Score: %d, Rewards: %f, Cars Left to Pass: %d\n", 
//                 i, env->score, env->rewards[0], env->carsToPass);
//         }

//         // Check for collisions between the player and the car
//         if (check_collision(env, car)) {
//             env->speed = env->min_speed;
//             env->collision_cooldown = CRASH_NOOP_DURATION;
//             printf("[DEBUG] Collision detected with car %d. Cooldown: %f\n", i, env->collision_cooldown);
//             // log_to_file(log_filename, "Collision detected! Cooldown: %f\n", env->collision_cooldown);
//         }

//         // Remove off-screen cars
//         if (car->y > ACTION_HEIGHT) {
//             printf("[DEBUG] Car %d went off-screen. Removing car.\n", i);
//             for (int j = i; j < env->numEnemies - 1; j++) {
//                 env->enemyCars[j] = env->enemyCars[j + 1];
//             }
//             env->numEnemies--;
//             i--;  // Adjust loop index to account for removed car
//         }
//     }

//     // Add new enemy cars based on random conditions
//     if (env->numEnemies < MAX_ENEMIES && rand() % 100 < 2) {
//         add_enemy_car(env);
//         // log_to_file(log_filename, "New enemy car added. Total Enemies: %d\n", env->numEnemies);
//         printf("[DEBUG] New enemy car added. Total enemies: %d\n", env->numEnemies);
//     }

//     // Handle day completion logic
//     if (env->carsToPass <= 0) {
//         env->day++;
//         env->carsToPass = env->day * 10 + 10;
//         env->speed += 0.1f;

//         // Log day completion
//         // log_to_file(log_filename, "Day %d completed! New Cars to Pass: %d\n", env->day, env->carsToPass);
//         add_log(env->log_buffer, &env->log);

//         // Reset rewards after day completes
//         env->rewards[0] = 0;
//     } else if (env->step_count >= env->day_length) {
//         if (env->carsToPass > 0) {
//             env->terminals[0] = 1;  // Mark game as over
//             // log_to_file(log_filename, "Game Over! Resetting...\n");
//             add_log(env->log_buffer, &env->log);
//             reset(env);
//             return;
//         }
//     }

//     // Accumulate rewards for the episode
//     env->log.episode_return += env->rewards[0];

//     // Compute and log observations
//     print_observations_to_file(env, "observations.log");
//     printf("[DEBUG] Step %d complete. Rewards: %f\n", env->step_count, env->rewards[0]);
    
//     // Log final rewards for this step
//     // log_to_file(log_filename, "Final Rewards for Step %d: %f\n", env->step_count, env->rewards[0]);

//     // Prepare for the next step
//     env->step_count++;
//     env->log.score = env->score;
//     // log_to_file(log_filename, "--- Step %d Complete ---\n", env->step_count);
// }



// // void step(Enduro* env) {

// //     // debugging
// //     if (env == NULL) {
// //         printf("env is NULL!\n");
// //         return;
// //     }
// //     printf("Stepping: player_x = %f, player_y = %f\n", env->player_x, env->player_y);


// //     // debugging - Make sure enemyCars is properly accessed
// //     if (env->numEnemies > 0 && env->enemyCars != NULL) {
// //         for (int i = 0; i < env->numEnemies; i++) {
// //             printf("Enemy car %d: x = %f, y = %f\n", i, env->enemyCars[i].x, env->enemyCars[i].y);
// //         }
// //     }
// //     // Log step start
// //     log_to_file("game_debug.log", "Observations[0]: %f\n", env->observations[0]);


// //     const char* log_filename = "game_debug.log";
// //     // log_to_file(log_filename, "\n--- Step %d Start ---\n", env->step_count);
// //     // log_to_file(log_filename, "Episode Length: %f, Reward: %f, Terminal: %d\n",
// //                 env->log.episode_length, env->rewards[0], env->terminals[0]);


// //     env->log.episode_length += 1;
// //     env->terminals[0] = 0;

// //     // Handle cooldown if active
// //     if (env->collision_cooldown > 0) {
// //         env->collision_cooldown -= 1;
// //         // log_to_file(log_filename, "Collision Cooldown: %d\n", env->collision_cooldown);
// //     }

// //     // Player movement logic
// //     if (env->collision_cooldown == 0) {
// //         int act = env->actions[0];
// //         if (act == 1 && env->player_x > 0) env->player_x -= 5;
// //         if (act == 2 && env->player_x < SCREEN_WIDTH - CAR_WIDTH) env->player_x += 5;
// //         if (act == 3 && env->speed < env->max_speed) env->speed += 0.1f;
// //         if (act == 4 && env->speed > env->min_speed) env->speed -= 0.1f;
// //     }

// //     printf("Actions[0]: %d\n", env->actions[0]);

// //     if (env->numEnemies >= MAX_ENEMIES) {
// //         printf("Enemy car limit reached.\n");
// //         return;
// //     }


// //     // Enemy car logic and collision handling
// //     for (int i = 0; i < env->numEnemies; i++) {
// //         Car* car = &env->enemyCars[i];
// //         car->y += env->speed;

// //         if (car->y > env->player_y && !car->passed) {
// //             env->carsToPass--;
// //             car->passed = true;
// //             env->score++;
// //             env->rewards[0] += 1;

// //             // Log when a car is passed
// //             // log_to_file(log_filename, 
// //                 "Car %d passed! Score: %d, Rewards: %f, Cars Left to Pass: %d\n", 
// //                 i, env->score, env->rewards[0], env->carsToPass);
// //         }

// //         if (check_collision(env, car)) {
// //             env->speed = env->min_speed;
// //             env->collision_cooldown = CRASH_NOOP_DURATION;
// //             // log_to_file(log_filename, "Collision detected! Cooldown: %d\n", env->collision_cooldown);
// //         }

// //         // Remove off-screen cars
// //         if (car->y > ACTION_HEIGHT) {
// //             for (int j = i; j < env->numEnemies - 1; j++) {
// //                 env->enemyCars[j] = env->enemyCars[j + 1];
// //             }
// //             env->numEnemies--;
// //             i--;
// //         }
// //     }

// //     if (env->numEnemies < MAX_ENEMIES && rand() % 100 < 2) {
// //         add_enemy_car(env);
// //         // log_to_file(log_filename, "New enemy car added. Total Enemies: %d\n", env->numEnemies);
// //     }


// //     // Handle day completion
// //     if (env->carsToPass <= 0) {
// //         env->day++;
// //         env->carsToPass = env->day * 10 + 10;
// //         env->speed += 0.1f;

// //         // Log day completion and reset
// //         // log_to_file(log_filename, "Day %d completed! New Cars to Pass: %d\n", env->day, env->carsToPass);
// //         add_log(env->log_buffer, &env->log);

// //         env->rewards[0] = 0;  // Reset rewards after a day completes
// //     } else if (env->step_count >= env->day_length) {
// //         if (env->carsToPass > 0) {
// //             env->terminals[0] = 1;  // Mark the game as over
// //             // log_to_file(log_filename, "Game Over! Resetting...\n");
// //             add_log(env->log_buffer, &env->log);
// //             reset(env);
// //             return;
// //         }
// //     }

// //     // Accumulate rewards for the episode
// //     env->log.episode_return += env->rewards[0];

// //     // compute_observations(env);
// //     print_observations_to_file(env, "observations.log");

// //     // Log the final rewards for this step
// //     // log_to_file(log_filename, "Final Rewards for Step %d: %f\n", env->step_count, env->rewards[0]);

// //     // Prepare for the next step
// //     env->step_count++;
// //     env->log.score = env->score;

// //     // log_to_file(log_filename, "--- Step %d Complete ---\n", env->step_count);
// // }

// // Rendering function
// void render(Client* client, Enduro* env) {
//     BeginDrawing();
//     ClearBackground(client->road_color);

//     // Render road (simplified perspective)
//     DrawTriangle((Vector2){ SCREEN_WIDTH / 3, ACTION_HEIGHT }, 
//                  (Vector2){ 2 * SCREEN_WIDTH / 3, ACTION_HEIGHT }, 
//                  (Vector2){ SCREEN_WIDTH / 2, 0 }, GRAY);

//     // Render player car
//     DrawRectangle(env->player_x, env->player_y, CAR_WIDTH, CAR_HEIGHT, client->player_color);

//     // Render enemy cars
//     for (int i = 0; i < env->numEnemies; i++) {
//         Car* car = &env->enemyCars[i];
//         DrawRectangle(car->x, car->y, CAR_WIDTH, CAR_HEIGHT, client->enemy_color);
//     }

//     // Render HUD env data
//     DrawText(TextFormat("Score: %05i", env->score), 56, ACTION_HEIGHT + 10, 10, WHITE);
//     DrawText(TextFormat("Cars to Pass: %03i", env->carsToPass), 56, ACTION_HEIGHT + 25, 10, WHITE);
//     DrawText(TextFormat("Day: %i", env->day), 56, ACTION_HEIGHT + 40, 10, WHITE);
//     DrawText(TextFormat("Speed: %.2f", env->speed), 56, ACTION_HEIGHT + 55, 10, WHITE);
//     DrawText(TextFormat("Step: %i", env->step_count), 56, ACTION_HEIGHT + 70, 10, WHITE);
//     // Box around HUD
//     DrawRectangleLines(0, ACTION_HEIGHT, SCREEN_WIDTH, HUD_HEIGHT, WHITE);

//     EndDrawing();
// }