#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include "raylib.h"

#define GRAVITY 9.8f
#define MASSCART 1.0f
#define MASSPOLE 0.1f
#define TOTAL_MASS (MASSPOLE + MASSCART)
#define LENGTH 0.5f // half pole length
#define POLEMASS_LENGTH (MASSPOLE * LENGTH)
#define FORCE_MAG 10.0f
#define TAU 0.02f // timestep duration

#define X_THRESHOLD 2.4f
#define THETA_THRESHOLD_RADIANS (12 * 2 * M_PI / 360)
#define MAX_STEPS 200
#define WIDTH 600
#define HEIGHT 800
#define SCALE 100

typedef struct Log {
    float episode_return;
    float episode_length;
    int x_threshold_termination;
    int pole_angle_termination;
    int max_steps_termination;
    int n;
} Log;

typedef struct Client {
} Client;

typedef struct CartPole {
    float* observations;
    float* actions;
    float* rewards;
    unsigned char* terminals;
    unsigned char* truncations;
    Log log;
    Client* client;
    float x;
    float x_dot;
    float theta;
    float theta_dot;
    int tick;
    int is_continuous;
} CartPole;

void add_log(CartPole* env) {
    env->log.episode_return += env->tick;
    env->log.episode_length = env->tick;
    env->log.x_threshold_termination += (env->x < -X_THRESHOLD || env->x > X_THRESHOLD);
    env->log.pole_angle_termination += (env->theta < -THETA_THRESHOLD_RADIANS || env->theta > THETA_THRESHOLD_RADIANS);
    env->log.max_steps_termination += (env->tick >= MAX_STEPS);
    env->log.n += 1;
}

void init(CartPole* env) {
    if (env == NULL) {
        return;
    }
    env->tick = 0;
}

void free_initialized(CartPole* env) {
}

void allocate(CartPole* env) {
    if (env == NULL) {
        return;
    }
    
    init(env);
    
    env->observations = (float*)calloc(4, sizeof(float));
    if (env->observations == NULL) {
        return;
    }
    
    env->actions = (float*)calloc(1, sizeof(float));
    if (env->actions == NULL) {
        free(env->observations);
        return;
    }
    
    env->rewards = (float*)calloc(1, sizeof(float));
    if (env->rewards == NULL) {
        free(env->observations);
        free(env->actions);
        return;
    }
    
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    if (env->terminals == NULL) {
        free(env->observations);
        free(env->actions);
        free(env->rewards);
        return;
    }
    
    env->truncations = (unsigned char*)calloc(1, sizeof(unsigned char));
    if (env->truncations == NULL) {
        free(env->observations);
        free(env->actions);
        free(env->rewards);
        free(env->terminals);
        return;
    }
}

void free_allocated(CartPole* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env->truncations);
    free_initialized(env);
}

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

Client* make_client(CartPole* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    InitWindow(WIDTH, HEIGHT, "puffer cartpole");
    SetTargetFPS(60);
    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

void c_render(CartPole* env) {
    if (IsKeyDown(KEY_ESCAPE))
        exit(0);
    if (IsKeyPressed(KEY_TAB))
        ToggleFullscreen();

    if (env->client == NULL) {
        env->client = make_client(env);
    }

    Client* client = env->client;
    (void)client; // Suppress unused variable warning

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    DrawLine(0, HEIGHT / 2, WIDTH, HEIGHT / 2, PUFF_CYAN);
    float cart_x = WIDTH / 2 + env->x * SCALE;
    float cart_y = HEIGHT / 2;
    DrawRectangle((int)(cart_x - 20), (int)(cart_y - 10), 40, 20, PUFF_CYAN);
    float pole_length = 2.0f * 0.5f * SCALE;
    float pole_x2 = cart_x + sinf(env->theta) * pole_length;
    float pole_y2 = cart_y - cosf(env->theta) * pole_length;
    DrawLineEx((Vector2){cart_x, cart_y}, (Vector2){pole_x2, pole_y2}, 5, PUFF_RED);
    DrawText(TextFormat("Steps: %i", env->tick), 10, 10, 20, PUFF_WHITE);
    DrawText(TextFormat("Cart Position: %.2f", env->x), 10, 40, 20, PUFF_WHITE);
    DrawText(TextFormat("Pole Angle: %.2f", env->theta * 180.0f / M_PI), 10, 70, 20, PUFF_WHITE);
    EndDrawing();
}

void compute_observations(CartPole* env) {
    if (env == NULL) {
        return;
    }
    
    if (env->observations == NULL) {
        return;
    }
    
    env->observations[0] = env->x;
    env->observations[1] = env->x_dot;
    env->observations[2] = env->theta;
    env->observations[3] = env->theta_dot;
}

void c_reset(CartPole* env) {
    if (env == NULL) {
        return;
    }
    
    env->x = ((float)rand() / (float)RAND_MAX) * 0.08f - 0.04f;
    env->x_dot = ((float)rand() / (float)RAND_MAX) * 0.08f - 0.04f;
    env->theta = ((float)rand() / (float)RAND_MAX) * 0.08f - 0.04f;
    env->theta_dot = ((float)rand() / (float)RAND_MAX) * 0.08f - 0.04f;
    env->tick = 0;
    
    compute_observations(env);
}

void c_step(CartPole* env) {  
    float force = 0.0;
    if (env->is_continuous) {
        force = env->actions[0] * FORCE_MAG;
    } else {
        force = (env->actions[0] > 0.5f) ? FORCE_MAG : -FORCE_MAG; 
    }

    float costheta = cosf(env->theta);
    float sintheta = sinf(env->theta);

    float temp = (force + POLEMASS_LENGTH * env->theta_dot * env->theta_dot * sintheta) / TOTAL_MASS;
    float thetaacc = (GRAVITY * sintheta - costheta * temp) / 
                     (LENGTH * (4.0f / 3.0f - MASSPOLE * costheta * costheta / TOTAL_MASS));
    float xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS;

    env->x += TAU * env->x_dot;
    env->x_dot += TAU * xacc;
    env->theta += TAU * env->theta_dot;
    env->theta_dot += TAU * thetaacc;

    bool done = env->x < -X_THRESHOLD || env->x > X_THRESHOLD ||
                env->theta < -THETA_THRESHOLD_RADIANS || env->theta > THETA_THRESHOLD_RADIANS ||
                env->tick >= MAX_STEPS;

    env->rewards[0] = done ? 0.0f : 1.0f;
    env->terminals[0] = done ? 1 : 0;

    env->tick += 1;

    if (done) {
        add_log(env);
        c_reset(env);
    }

    compute_observations(env);
}
