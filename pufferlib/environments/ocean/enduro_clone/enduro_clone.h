#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
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
    //printf("Log: %f, %f, %f\n", log->episode_return, log->episode_length, log->score);
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
typedef struct Car {
    float x, y;
    bool passed;
} Car;

// Game environment structure
typedef struct Enduro {
    float* observations;
    unsigned int* actions;
    float* rewards;
    unsigned char* terminals;
    LogBuffer* log_buffer;
    Log log;

    float player_x;
    float player_y;
    float speed;
    float max_speed;
    float min_speed;
    int score;
    int carsToPass;
    int day;
    int day_length;
    int step_count;
    Car enemyCars[MAX_ENEMIES];
    int numEnemies;

    float width;
    float height;
    int frameskip;

    int collision_cooldown;  // Cooldown after a collision (disables input)
} Enduro;

void init(Enduro* env) {
    env->width = SCREEN_WIDTH;
    env->height = SCREEN_HEIGHT;

    // Game start values
    env->player_x = env->width / 2 - CAR_WIDTH / 2;
    env->player_y = ACTION_HEIGHT - CAR_HEIGHT;
    env->speed = 1.0f;
    env->max_speed = 10.0f;
    env->min_speed = -1.0f;
    env->score = 0;
    env->carsToPass = INITIAL_CARS_TO_PASS; // 200;
    env->day = 1;
    env->day_length = DAY_LENGTH;
    env->step_count = 0;
    env->numEnemies = 0;
    env->collision_cooldown = 0;
}

void allocate(Enduro* env) {
    init(env);
    env->observations = (float*)calloc(8, sizeof(float));
    env->actions = (unsigned int*)calloc(5, sizeof(unsigned int)); // noop = 0, left = 1, right = 2, up = 3, down = 4
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
}

// Compute observations for RL agent
void compute_observations(Enduro* env) {
    env->observations[0] = env->player_x / SCREEN_WIDTH;
    env->observations[1] = env->carsToPass;
    env->observations[2] = env->speed;
    env->observations[3] = env->day;
    env->observations[4] = env->numEnemies;
    env->observations[5] = env->day_length;
    // Further observations can be added, such as enemy car positions
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
    env->log.episode_length += 1;
    env->rewards[0] = 0;
    env->terminals[0] = 0;

    // Decrease the cooldown if it is active
    if (env->collision_cooldown > 0) {
        env->collision_cooldown -= 1;
    }

    // Only allow player movement if there is no collision cooldown
    if (env->collision_cooldown == 0) {
        // Player movement logic (actions: 0 = stay, 1 = left, 2 = right)
        unsigned int act = env->actions[0];
        if (act == 1 && env->player_x > 0) env->player_x -= 5;
        if (act == 2 && env->player_x < SCREEN_WIDTH - CAR_WIDTH) env->player_x += 5;
        if (act == 3 && env->speed < env->max_speed) env->speed += 0.1f;
        if (act == 4 && env->speed > 0) env->speed -= 0.1f;
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
    
    if (env->carsToPass <= 0) {
        // Player successfully passed enough cars
        env->day++;
        env->carsToPass = env->day * 10 + 10;
        env->speed += 0.1f;
    } else if (env->step_count >= env->day_length) {
        // If carsToPass > 0 when the day ends, terminate the environment
        if (env->carsToPass > 0) {
            env->terminals[0] = 1;  // Terminate the environment
            reset(env);
        }
    }

    env->step_count += 1;

    compute_observations(env);
}

// Client for rendering and input
typedef struct Client {
    float width;
    float height;
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
//         unsigned int action = 0;
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
