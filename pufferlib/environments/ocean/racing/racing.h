// racing.h
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "raylib.h"

#define NUM_LANES 3
#define MAX_CARS 100
#define NUM_OBSERVATIONS 7
#define NUM_ACTIONS 3

typedef struct CEnduro CEnduro;
struct CEnduro {
    float* observations;
    unsigned int* actions;
    float* rewards;
    unsigned char* terminals;
    float* player_x_y;
    float* other_cars_x_y;
    int* other_cars_active;
    unsigned int* score_day;
    float width;
    float height;
    float player_width;
    float player_height;
    float other_car_width;
    float other_car_height;
    float player_speed;
    float base_car_speed;
    float max_player_speed;
    float min_player_speed;
    float speed_increment;
    unsigned int max_score;
    float min_player_x;
    float max_player_x;
    float road_left_edge;  // Added field
    int num_other_cars;
    int cars_to_pass;
    int cars_passed;
    int frame_count;
    int car_spawn_interval;
    float score;
    int tick;
    int frameskip;
};

typedef struct Client Client;
struct Client {
    float width;
    float height;
    float road_width;
    float road_left_edge;
    Color background_color;
    Color road_color;
    Color player_color;
    Color other_car_color;
};

// Function declarations
void init(CEnduro* env);
void reset(CEnduro* env);
void step(CEnduro* env);
Client* make_client(CEnduro* env);
void close_client(Client* client);
void render(Client* client, CEnduro* env);


void init(CEnduro* env) {
    env->tick = 0;
    env->frame_count = 0;
    env->cars_passed = 0;
    env->player_speed = 3.0f;
    env->score = 0;
    
    env->road_left_edge = (env->width - (env->width * 0.8f)) / 2;
    env->min_player_x = env->road_left_edge;
    env->max_player_x = env->road_left_edge + env->width - env->player_width;
    
    env->num_other_cars = 0;
    env->cars_to_pass = 10;  // Initial cars to pass
    env->car_spawn_interval = 120;
    
    srand(time(NULL));
}

void allocate(CEnduro* env) {
    init(env);
    env->observations = (float*)calloc(NUM_OBSERVATIONS, sizeof(float));
    env->actions = (unsigned int*)calloc(1, sizeof(unsigned int));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->player_x_y = (float*)calloc(2, sizeof(float));
    env->other_cars_x_y = (float*)calloc(2 * MAX_CARS, sizeof(float));
    env->other_cars_active = (int*)calloc(MAX_CARS, sizeof(int));
    env->score_day = (unsigned int*)calloc(2, sizeof(unsigned int));
}

void free_allocated(CEnduro* env) {
    if (env->observations) free(env->observations);
    if (env->actions) free(env->actions);
    if (env->rewards) free(env->rewards);
    if (env->terminals) free(env->terminals);
    if (env->player_x_y) free(env->player_x_y);
    if (env->other_cars_x_y) free(env->other_cars_x_y);
    if (env->other_cars_active) free(env->other_cars_active);
    if (env->score_day) free(env->score_day);
}


// void free_allocated(CEnduro* env) {
//     free(env->observations);
//     free(env->actions);
//     free(env->rewards);
//     free(env->terminals);
//     free(env->player_x_y);
//     free(env->other_cars_x_y);
//     free(env->other_cars_active);
//     free(env->score_day);
// }

void compute_observations(CEnduro* env) {
    float lane_width = env->width / NUM_LANES;
    int player_lane = (int)((env->player_x_y[0] - env->road_left_edge) / lane_width);
    
    env->observations[0] = (float)player_lane / (NUM_LANES - 1);
    env->observations[1] = env->player_speed / env->max_player_speed;
    
    // Initialize nearest car distances
    float nearest_cars[NUM_LANES] = {1.0f, 1.0f, 1.0f};
    
    for (int i = 0; i < env->num_other_cars; i++) {
        if (env->other_cars_active[i]) {
            int car_lane = (int)((env->other_cars_x_y[2*i] - env->road_left_edge) / lane_width);
            float distance = (env->other_cars_x_y[2*i+1] - env->player_x_y[1]) / env->height;
            if (distance < nearest_cars[car_lane] && distance > 0) {
                nearest_cars[car_lane] = distance;
            }
        }
    }
    
    env->observations[2] = nearest_cars[0];
    env->observations[3] = nearest_cars[1];
    env->observations[4] = nearest_cars[2];
    env->observations[5] = (float)env->cars_to_pass / env->max_score;
    env->observations[6] = (float)env->score_day[1] / env->max_score;
}

void reset_round(CEnduro* env) {
    float lane_width = env->width / NUM_LANES;
    env->player_x_y[0] = env->road_left_edge + lane_width + (lane_width - env->player_width) / 2;
    env->player_x_y[1] = env->height - env->player_height - 10;
    env->player_speed = 3.0f;
    
    env->num_other_cars = 0;
    for (int i = 0; i < MAX_CARS; i++) {
        env->other_cars_active[i] = 0;
    }
    
    env->frame_count = 0;
    env->cars_passed = 0;
}

void reset(CEnduro* env) {
    reset_round(env);
    env->score_day[0] = 0;  // score
    env->score_day[1] = 1;  // day
    env->cars_to_pass = 10;  // Initial cars to pass for day 1
    env->car_spawn_interval = 120;
    compute_observations(env);
}

void step(CEnduro* env) {
    env->tick += 1;
    env->rewards[0] = 0;
    env->terminals[0] = 0;
    
    unsigned int act = env->actions[0];
    float lane_width = env->width / NUM_LANES;
    
    for (int i = 0; i < env->frameskip; i++) {
        // Move player
        if (act == 0) {  // Move left
            env->player_x_y[0] -= lane_width;
        } else if (act == 2) {  // Move right
            env->player_x_y[0] += lane_width;
        }
        
        // Adjust speed
        if (act == 1) {  // Speed up
            env->player_speed += env->speed_increment;
        } else {  // Slow down slightly if not speeding up
            env->player_speed -= env->speed_increment * 0.5f;
        }
        
        // Clamp player position and speed
        env->player_x_y[0] = fmaxf(env->min_player_x, fminf(env->player_x_y[0], env->max_player_x));
        env->player_speed = fmaxf(env->min_player_speed, fminf(env->player_speed, env->max_player_speed));
        
        // Move other cars
        for (int j = 0; j < env->num_other_cars; j++) {
            if (env->other_cars_active[j]) {
                env->other_cars_x_y[2*j+1] += env->base_car_speed + env->player_speed;
                
                // Check for collision
                if (fabs(env->other_cars_x_y[2*j+1] - env->player_x_y[1]) < env->player_height &&
                    fabs(env->other_cars_x_y[2*j] - env->player_x_y[0]) < env->player_width) {
                    env->player_speed = env->min_player_speed;
                    env->rewards[0] -= 10.0f;
                }
                
                // Remove car if off screen
                if (env->other_cars_x_y[2*j+1] > env->height) {
                    env->other_cars_active[j] = 0;
                } else if (env->other_cars_x_y[2*j+1] > env->player_x_y[1] && !env->other_cars_active[j]) {
                    env->cars_passed++;
                    env->cars_to_pass--;
                    env->other_cars_active[j] = 1;
                    env->rewards[0] += 1.0f;
                }
            }
        }
        
        // Spawn new cars
        env->frame_count++;
        if (env->frame_count % env->car_spawn_interval == 0) {
            if (env->num_other_cars < MAX_CARS) {
                int lane = rand() % NUM_LANES;
                env->other_cars_x_y[2 * env->num_other_cars] = env->road_left_edge + lane_width * lane + (lane_width - env->other_car_width) / 2;
                env->other_cars_x_y[2 * env->num_other_cars + 1] = -env->other_car_height;
                env->other_cars_active[env->num_other_cars] = 1;
                env->num_other_cars++;  // Only increment after ensuring space
            }

        }
        
        // Update score
        env->score += env->player_speed * 0.1f;
        env->score_day[0] = (unsigned int)env->score;
        
        // Check for day completion
        if (env->cars_to_pass <= 0) {
            env->score_day[1]++;  // New day
            env->cars_to_pass = 10 + (env->score_day[1] - 1) * 5;  // Increase cars to pass each day
            env->car_spawn_interval = (int)(120 * pow(0.9, env->score_day[1] - 1));  // Decrease spawn interval each day
            env->rewards[0] += 50.0f;  // Bonus for completing a day
            reset_round(env);
        }
        
        compute_observations(env);
    }
    
    // Check for game over condition
    if (env->score_day[1] > env->max_score) {
        env->terminals[0] = 1;
        reset(env);
    }
}

Client* make_client(CEnduro* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = env->width;
    client->height = env->height;
    client->road_width = env->width * 0.8f;
    client->road_left_edge = (env->width - client->road_width) / 2;
    client->background_color = RAYWHITE;
    client->road_color = DARKGRAY;
    client->player_color = RED;
    client->other_car_color = BLUE;

    InitWindow(env->width, env->height, "Enduro RL Environment");
    SetTargetFPS(60 / env->frameskip);

    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

void render(Client* client, CEnduro* env) {
    if (WindowShouldClose()) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(client->background_color);

    // Draw road
    DrawRectangle(client->road_left_edge, 0, client->road_width, client->height, client->road_color);
    
    // Draw lane dividers
    for (int i = 1; i < NUM_LANES; i++) {
        float x = client->road_left_edge + (client->road_width / NUM_LANES) * i;
        DrawLine(x, 0, x, client->height, WHITE);
    }

    // Draw player car
    DrawRectangle(env->player_x_y[0], env->player_x_y[1], env->player_width, env->player_height, client->player_color);

    // Draw other cars
    for (int i = 0; i < env->num_other_cars; i++) {
        if (env->other_cars_active[i]) {
            DrawRectangle(env->other_cars_x_y[2*i], env->other_cars_x_y[2*i+1], env->other_car_width, env->other_car_height, client->other_car_color);
        }
    }

    // Draw game information
    DrawText(TextFormat("Score: %d", env->score_day[0]), 10, 10, 20, BLACK);
    DrawText(TextFormat("Day: %d", env->score_day[1]), 10, 40, 20, BLACK);
    DrawText(TextFormat("Cars Passed: %d", env->cars_passed), 10, 70, 20, BLACK);
    DrawText(TextFormat("Cars Left: %d", env->cars_to_pass), 10, 100, 20, BLACK);
    DrawText(TextFormat("Speed: %.1f", env->player_speed), 10, 130, 20, BLACK);

    EndDrawing();
}