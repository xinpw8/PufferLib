#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>
#include <unistd.h>

#include "raylib.h"

#define LOG_BUFFER_SIZE 1024
#define SCREEN_WIDTH 160
#define SCREEN_HEIGHT 210
#define HUD_HEIGHT 55
#define ACTION_HEIGHT (SCREEN_HEIGHT - HUD_HEIGHT)
#define CAR_WIDTH 10
#define CAR_HEIGHT 10
#define MAX_ENEMIES 10
#define CRASH_NOOP_DURATION 60
#define DAY_LENGTH 2000
#define INITIAL_CARS_TO_PASS 5

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
struct Car{
    float x, y;
    bool passed;
};

// Game environment structure
typedef struct Enduro Enduro;
struct Enduro {
    // RL
    float* observations;
    unsigned char* actions;
    float* rewards;
    unsigned char* terminals;
    LogBuffer* log_buffer;
    Log log;

    // Game state
    float player_x;        // X position of the player
    float player_y;        // Y position of the player
    float speed;           // Current speed of the player
    float max_speed;       // Maximum speed
    float min_speed;       // Minimum speed
    int score;             // Player score
    int carsToPass;        // Number of cars left to pass
    int day;               // Current day
    int day_length;        // Total number of steps in a day
    int step_count;        // Count of steps taken so far
    int numEnemies;        // Number of enemy cars on the screen
    int collision_cooldown; // Cooldown after a collision
    int crash_noop_duration; // Duration of no-op after a crash
    Car enemyCars[MAX_ENEMIES];

    // Screen and gameplay parameters
    int width;            // Screen width
    int height;           // Screen height
    int hud_height;       // HUD height
    int car_width;        // Enemy car width
    int car_height;       // Enemy car height
    int max_enemies;        // Maximum number of enemies
    int initial_cars_to_pass; // Initial number of cars to pass
    float action_height;      // Height of the action area
};

void init(Enduro* env, int width, int height, int hud_height, int car_width, int car_height,
          int max_enemies, int crash_noop_duration, int day_length, int initial_cars_to_pass,
          float min_speed, float max_speed) {
    env->width = width;
    env->height = height;
    env->hud_height = hud_height;
    env->car_width = car_width;
    env->car_height = car_height;
    env->max_enemies = max_enemies;
    env->crash_noop_duration = crash_noop_duration;
    env->day_length = day_length;
    env->initial_cars_to_pass = initial_cars_to_pass;
    env->min_speed = min_speed;
    env->max_speed = max_speed;

    // Game start values
    env->action_height = env->height - env->hud_height;
    env->player_x = env->width / 2 - env->car_width / 2;
    env->player_y = env->action_height - env->car_height;
    env->speed = 1.0f;
    env->score = 0;
    env->carsToPass = env->initial_cars_to_pass; // 200;
    env->day = 1;
    env->day_length = env->day_length;
    env->step_count = 0;
    env->numEnemies = 0;
    env->collision_cooldown = 0;
    
    // Allocate memory for actions
    env->actions = (unsigned char*)calloc(5, sizeof(unsigned char)); // Assuming 5 possible actions
    if (env->actions == NULL) {
        fprintf(stderr, "Failed to allocate memory for actions\n");
        exit(1);
    }

    // Initialize actions to 0
    for (int i = 0; i < 5; i++) {
        env->actions[i] = 0;
    }

    printf(".h init function: actions[0]: %d\n", env->actions[0]);
}

void allocate(Enduro* env, int width, int height, int hud_height, int car_width, int car_height,
              int max_enemies, int crash_noop_duration, int day_length, int initial_cars_to_pass,
              float min_speed, float max_speed) {
    
    // Free previously allocated memory if any
    if (env->actions != NULL) {
        free(env->actions);
    }

    init(env, width, height, hud_height, car_width, car_height, max_enemies, crash_noop_duration,
         day_length, initial_cars_to_pass, min_speed, max_speed);
         
    env->observations = (float*)calloc(8 + 2 * MAX_ENEMIES, sizeof(float));
    env->actions = (unsigned char*)calloc(5, sizeof(unsigned char)); // 5 actions: noop, left, right, speed up, slow down
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->log_buffer = allocate_logbuffer(LOG_BUFFER_SIZE);
}

void free_allocated(Enduro* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free_logbuffer(env->log_buffer);
}

// Add new enemy cars for player to pass
void add_enemy_car(Enduro* env) {
    if (env->numEnemies >= MAX_ENEMIES) return;
    Car car = { .x = rand() % (SCREEN_WIDTH - CAR_WIDTH), .y = -CAR_HEIGHT, .passed = false };
    env->enemyCars[env->numEnemies++] = car;
}

void compute_observations(Enduro* env) {
    // Normalize and set player state
    env->observations[0] = env->player_x / SCREEN_WIDTH;  // Player's X position normalized
    env->observations[1] = env->player_y / ACTION_HEIGHT; // Player's Y position normalized
    env->observations[2] = env->speed / env->max_speed;   // Speed normalized
    env->observations[3] = env->carsToPass;               // Cars left to pass
    env->observations[4] = env->day;                      // Current day
    env->observations[5] = env->numEnemies;               // Number of active enemies
    env->observations[6] = env->collision_cooldown;       // Collision cooldown time
    env->observations[7] = env->score;                    // Current score

    // Enemy car positions (X, Y) for each active enemy
    int obs_idx = 8;  // Start after basic info
    for (int i = 0; i < MAX_ENEMIES; i++) {
        env->observations[obs_idx++] = env->enemyCars[i].x / SCREEN_WIDTH;  // X position normalized
        env->observations[obs_idx++] = env->enemyCars[i].y / ACTION_HEIGHT; // Y position normalized
    }
}

void print_observations_to_file(Enduro* env, const char* filename) {
    // Open the file in append mode ("a")
    FILE* file = fopen(filename, "a");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        perror("fopen");
        return;
    }

    // Call compute_observations to update observations
    compute_observations(env);

    // Write observations to the file along with everything else
    fprintf(file, "Player Position (X, Y): (%f, %f)\n", env->player_x, env->player_y);
    fprintf(file, "Speed: %f\n", env->speed);
    fprintf(file, "Cars to Pass: %d\n", env->carsToPass);
    fprintf(file, "Day: %d\n", env->day);
    fprintf(file, "Number of Enemies: %d\n", env->numEnemies);
    fprintf(file, "Collision Cooldown: %d\n", env->collision_cooldown);
    fprintf(file, "Score: %d\n", env->score);
    
    fprintf(file, "Observations (Step: %d):\n", env->step_count);
    for (int i = 0; i < 28; i++) {
        fprintf(file, "Observation[%d]: %f\n", i, env->observations[i]);
    }
    fprintf(file, "\n");

    // Close the file
    fclose(file);
}


void reset_round(Enduro* env) {
    env->player_x = SCREEN_WIDTH / 2 - CAR_WIDTH / 2;
    env->player_y = ACTION_HEIGHT - CAR_HEIGHT;
    env->score = 0;
    env->carsToPass = INITIAL_CARS_TO_PASS;
    env->speed = 1.0;
    env->numEnemies = 0;
    env->day = 1;
    env->day_length = DAY_LENGTH;
    env->step_count = 0;
    env->collision_cooldown = 0;
}

void reset(Enduro* env) {
    env->log = (Log){0};
    reset_round(env);
    compute_observations(env);
}

// Check collision between player and enemy car
bool check_collision(Enduro* env, Car* car) {
    return !(env->player_x > car->x + CAR_WIDTH || 
             env->player_x + CAR_WIDTH < car->x || 
             env->player_y > car->y + CAR_HEIGHT || 
             env->player_y + CAR_HEIGHT < car->y);
}

// Step function to progress the game by one frame
void step(Enduro* env) {
    printf(".h top of step: episode_length, rewards, terminals=%f, %f, %d\n",             env->log.episode_length, env->rewards[0], env->terminals[0]);
    
    env->log.episode_length += 1;
    env->rewards[0] = 0;
    env->terminals[0] = 0;

    // Decrease the cooldown if it is active
    if (env->collision_cooldown > 0) {
        env->collision_cooldown -= 1;
    }

    // Only allow player movement if there is no collision cooldown
    if (env->collision_cooldown == 0) {
        // Player movement logic (actions: 0 = stay, 1 = left, 2 = right, 3 = speed up, 4 = slow down)
        unsigned char act = env->actions[0];  // Remove the pointer, just use the value directly
        if (act == 1 && env->player_x > 0) env->player_x -= 5;
        if (act == 2 && env->player_x < SCREEN_WIDTH - CAR_WIDTH) env->player_x += 5;
        if (act == 3 && env->speed < env->max_speed) env->speed += 0.1f;
        if (act == 4 && env->speed > env->min_speed) env->speed -= 0.1f;  // Changed condition to use min_speed
    }

    // Enemy car movement logic
    for (int i = 0; i < env->numEnemies; i++) {
        Car* car = &env->enemyCars[i];
        car->y += env->speed;

        // Check for collision with player
        if (check_collision(env, car)) {
            env->speed = env->min_speed;  // Set speed to minimum if collision occurs
            env->collision_cooldown = CRASH_NOOP_DURATION; // Prevent input for CRASH_NOOP_DURATION steps after collision
        
        
        }

        // Check if player has passed a car
        if (car->y > env->player_y && !car->passed) {
            env->carsToPass--;
            car->passed = true;
            env->score += 1; // Increase score for passing a car (displayed on HUD)
            env->rewards[0] += 1; // Reward for passing a car
            env->log.episode_return += 1;
        }

        // Remove cars that move off the screen
        if (car->y > ACTION_HEIGHT) {
            for (int j = i; j < env->numEnemies - 1; j++) {
                env->enemyCars[j] = env->enemyCars[j + 1];
            }
            env->numEnemies--;
            i--;
        }
    }

    // Add new enemy cars
    if (rand() % 100 < 2) add_enemy_car(env);

    // Day progression logic
    printf("carsToPass: %d\n", env->carsToPass);
    if (env->carsToPass <= 0) {
        // Player successfully passed enough cars
        
        printf("Day: %d\n", env->day);
        printf("if cond true\n");
        env->day++;
        env->carsToPass = env->day * 10 + 10;
        env->speed += 0.1f;
        add_log(env->log_buffer, &env->log);
    } else if (env->step_count >= env->day_length) {
        // If carsToPass > 0 when the day ends, terminate the environment
        printf("elseif cond true\n");
        
        if (env->carsToPass > 0) {
            printf("terminal condition true\n");
            env->terminals[0] = 1;  // Terminate the environment
            add_log(env->log_buffer, &env->log);
            reset(env);
            return;
        }
    }

    env->step_count++;
    env->log.score = env->score;
    env->log.episode_return += env->rewards[0];

    compute_observations(env);
    // Debugging observations
    // for (int i = 0; i < 28; i++) {
    //     printf("Observation[%d]: %f\n", i, env->observations[i]);
    // }
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        printf("Current working directory: %s\n", cwd);
    } else {
        perror("getcwd() error");
    }
    printf("Writing observations to file...\n");
    print_observations_to_file(env, "observations.log");


}

// Client for rendering and input
typedef struct Client {
    int width;
    int height;
    Color player_color;
    Color enemy_color;
    Color road_color;
} Client;

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

// Rendering function
void render(Client* client, Enduro* env) {
    BeginDrawing();
    ClearBackground(client->road_color);

    // Render road (simplified perspective)
    DrawTriangle((Vector2){ SCREEN_WIDTH / 3, ACTION_HEIGHT }, 
                 (Vector2){ 2 * SCREEN_WIDTH / 3, ACTION_HEIGHT }, 
                 (Vector2){ SCREEN_WIDTH / 2, 0 }, GRAY);

    // Render player car
    DrawRectangle(env->player_x, env->player_y, CAR_WIDTH, CAR_HEIGHT, client->player_color);

    // Render enemy cars
    for (int i = 0; i < env->numEnemies; i++) {
        Car* car = &env->enemyCars[i];
        DrawRectangle(car->x, car->y, CAR_WIDTH, CAR_HEIGHT, client->enemy_color);
    }

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

// int main(void) {
//     Enduro env;
//     allocate(&env);

//     Client* client = make_client(&env);

//     while (!WindowShouldClose()) {
//         // Update game (step)
//         unsigned char action = 0;
//         // noop = 0, left = 1, right = 2, up = 3, down = 4
        
//         if (IsKeyDown(KEY_LEFT)) action = 1;
//         if (IsKeyDown(KEY_RIGHT)) action = 2;
//         if (IsKeyDown(KEY_UP)) action = 3;
//         if (IsKeyDown(KEY_DOWN)) action = 4;

//         env.actions[0] = action;

//         step(&env);

//         // Render the game
//         render(client, &env);
//     }

//     // Clean up
//     close_client(client);
//     free_allocated(&env);
//     return 0;
// }
