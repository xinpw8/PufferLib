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
#define SCALE 100 // scaling for rendering

typedef struct Log {
    float episode_return;
    float episode_length;
    int x_threshold_termination;
    int pole_angle_termination;
    int max_steps_termination;
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
    if (buffer) {
        free(buffer->logs);
        free(buffer);
    }
}

void add_log(LogBuffer* logs, Log* log) {
    if (logs->idx == logs->length) {
        return;
    }
    logs->logs[logs->idx] = *log;
    logs->idx++;
}

Log aggregate_and_clear(LogBuffer* logs) {
    Log log = {0};
    if (logs->idx == 0) {
        return log;
    }
    for (int i = 0; i < logs->idx; i++) {
        log.episode_return += logs->logs[i].episode_return;
        log.episode_length += logs->logs[i].episode_length;
        log.x_threshold_termination += logs->logs[i].x_threshold_termination;
        log.pole_angle_termination += logs->logs[i].pole_angle_termination;
        log.max_steps_termination += logs->logs[i].max_steps_termination;
    }
    log.episode_return /= logs->idx;
    log.episode_length /= logs->idx;
    log.x_threshold_termination /= logs->idx;
    log.pole_angle_termination /= logs->idx;
    log.max_steps_termination /= logs->idx;
    logs->idx = 0;
    return log;
}

typedef struct CartPole {
    float* observations;      // [x, x_dot, theta, theta_dot]
    float* actions;             // float for cont support. action: 0 (L) or 1 (R)
    float* rewards;
    unsigned char* dones;
    LogBuffer* log_buffer;
    Log log;

    // Environment state variables
    float x;         // cart position
    float x_dot;     // cart velocity
    float theta;     // pole angle
    float theta_dot; // pole angular velocity

    int steps_beyond_done;  // -1 means not done yet, 0 means just done, >0 means stepping after done
    int steps;              // step counter for current episode

    // Control parameters
    int continuous; // set in cartpole.py
} CartPole;

typedef struct Client {
} Client;

void init(CartPole* env) {
    env->steps = 0;
    env->steps_beyond_done = -1;
    if (!env->log_buffer)
        env->log_buffer = allocate_logbuffer(1024);
}

void free_initialized(CartPole* env) {
    if (env->log_buffer) {
        free_logbuffer(env->log_buffer);
        env->log_buffer = NULL;
    }
}

void allocate(CartPole* env) {
    init(env);
    env->observations = (float*)calloc(4, sizeof(float));
    env->actions = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->dones = (unsigned char*)calloc(1, sizeof(unsigned char));
    if (!env->log_buffer)
        env->log_buffer = allocate_logbuffer(1024);
}

void free_allocated(CartPole* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->dones);
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

void c_render(Client* client, CartPole* env) {
    if (IsKeyDown(KEY_ESCAPE))
        exit(0);
    if (IsKeyPressed(KEY_TAB))
        ToggleFullscreen();

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    // Draw track: a horizontal line through the middle
    DrawLine(0, HEIGHT / 2, WIDTH, HEIGHT / 2, PUFF_CYAN);

    // Calculate cart position in pixels (centered)
    float cart_x = WIDTH / 2 + env->x * SCALE;
    float cart_y = HEIGHT / 2;

    // Draw cart as a rectangle (40x20)
    DrawRectangle((int)(cart_x - 20), (int)(cart_y - 10), 40, 20, PUFF_CYAN);

    // Draw pole as a red line. Pole length = 2 * 0.5 scaled.
    float pole_length = 2.0f * 0.5f * SCALE;
    float pole_x2 = cart_x + sinf(env->theta) * pole_length;
    float pole_y2 = cart_y - cosf(env->theta) * pole_length;
    DrawLineEx((Vector2){cart_x, cart_y}, (Vector2){pole_x2, pole_y2}, 5, PUFF_RED);

    // Draw info text
    DrawText(TextFormat("Steps: %i", env->steps), 10, 10, 20, PUFF_WHITE);
    DrawText(TextFormat("Cart Position: %.2f", env->x), 10, 40, 20, PUFF_WHITE);
    DrawText(TextFormat("Pole Angle: %.2f", env->theta * 180.0f / M_PI), 10, 70, 20, PUFF_WHITE);

    EndDrawing();
}

void compute_observations(CartPole* env) {
    env->observations[0] = env->x;
    env->observations[1] = env->x_dot;
    env->observations[2] = env->theta;
    env->observations[3] = env->theta_dot;
}

void c_reset(CartPole* env) {
    env->x = ((float)rand() / (float)RAND_MAX) * 0.08f - 0.04f;
    env->x_dot = ((float)rand() / (float)RAND_MAX) * 0.08f - 0.04f;
    env->theta = ((float)rand() / (float)RAND_MAX) * 0.08f - 0.04f;
    env->theta_dot = ((float)rand() / (float)RAND_MAX) * 0.08f - 0.04f;
    env->steps = 0;

    compute_observations(env);
}

void c_step(CartPole* env) {
    float force = 0.0;
    if (env->continuous) {
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
                env->steps >= MAX_STEPS;

    env->rewards[0] = done ? 0.0f : 1.0f;
    env->dones[0] = done ? 1 : 0;

    env->steps += 1;

    if (done) {
        env->log.episode_return += env->steps;
        env->log.episode_length = env->steps;
        env->log.x_threshold_termination += (env->x < -X_THRESHOLD || env->x > X_THRESHOLD);
        env->log.pole_angle_termination += (env->theta < -THETA_THRESHOLD_RADIANS || env->theta > THETA_THRESHOLD_RADIANS);
        env->log.max_steps_termination += (env->steps >= MAX_STEPS);

        add_log(env->log_buffer, &env->log);
        c_reset(env);
        memset(&env->log, 0, sizeof(Log));
    }

    compute_observations(env);
}
