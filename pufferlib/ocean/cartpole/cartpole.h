#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "raylib.h"

#define TICK_RATE 1.0f/60.0f

#define LOG_BUFFER_SIZE 1024

// Constants specific to CartPole
#define GRAVITY 9.8f
#define CART_MASS 1.0f
#define POLE_MASS 0.1f
#define POLE_HALF_LENGTH 0.5f  // Half the pole's length
#define FORCE_MAG 10.0f
#define TAU 0.02f  // Seconds between state updates

// Angle thresholds in radians
#define THETA_THRESHOLD_RADIANS (12.0f * M_PI / 180.0f)  // 12 degrees
#define X_THRESHOLD 2.4f
#define SCREEN_WIDTH 600
#define SCREEN_HEIGHT 800

#define MAX_STEPS 200

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
    free(buffer->logs);
    free(buffer);
}

void add_log(LogBuffer* logs, Log* log) {
    if (logs->idx == logs->length) {
        return;
    }
    logs->logs[logs->idx] = *log;
    logs->idx += 1;
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
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* dones;
    LogBuffer* log_buffer;
    Log log;
    
    // State variables
    float x;         // Cart position
    float x_dot;     // Cart velocity
    float theta;     // Pole angle
    float theta_dot; // Pole angular velocity
    
    // Environment parameters
    int steps;
    int width;       // For rendering
    int height;      // For rendering
    int frameskip;
    int continuous;  // Whether actions are continuous or discrete
    int num_obs;
} CartPole;

void init(CartPole* env) {
    // Initialize window dimensions to match constants
    env->width = SCREEN_WIDTH;
    env->height = SCREEN_HEIGHT;
    
    // Set default values for other parameters
    env->frameskip = 1;
    env->continuous = 0;  // Discrete actions by default
    env->steps = 0;
}

void allocate(CartPole* env) {
    init(env);  // Initialize all parameters first
    env->observations = (float*)calloc(4, sizeof(float));  // 4 observations
    env->actions = (int*)calloc(1, sizeof(int));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->dones = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->log_buffer = allocate_logbuffer(LOG_BUFFER_SIZE);
}

void free_initialized(CartPole* env) {
    free_logbuffer(env->log_buffer);
}

void free_allocated(CartPole* env) {
    free(env->actions);
    free(env->observations);
    free(env->dones);
    free(env->rewards);
    free_initialized(env);
}

void compute_observations(CartPole* env) {
    env->observations[0] = env->x;             // Cart position
    env->observations[1] = env->x_dot;          // Cart velocity
    env->observations[2] = env->theta;        // Pole angle
    env->observations[3] = env->theta_dot;    // Pole angular velocity
}

void c_reset(CartPole* env) {
    env->log = (Log){0};
    env->steps = 0;
    
    // Reset state with small random values as per Gym implementation
    env->x = ((float)rand() / (float)RAND_MAX) * 0.1f - 0.05f;
    env->x_dot = ((float)rand() / (float)RAND_MAX) * 0.1f - 0.05f;
    env->theta = ((float)rand() / (float)RAND_MAX) * 0.1f - 0.05f;
    env->theta_dot = ((float)rand() / (float)RAND_MAX) * 0.1f - 0.05f;
    
    compute_observations(env);
}

bool is_done(CartPole* env) {
    // Episode is done if:
    // 1. Cart position is more than X_THRESHOLD
    // 2. Pole angle is more than THETA_THRESHOLD_RADIANS
    // 3. Episode length exceeds max_steps
    if (env->x < -X_THRESHOLD || env->x > X_THRESHOLD) {
        env->log.x_threshold_termination += 1;
        return true;
    }
    if (env->theta < -THETA_THRESHOLD_RADIANS || env->theta > THETA_THRESHOLD_RADIANS) {
        env->log.pole_angle_termination += 1;
        return true;
    }
    if (env->steps >= MAX_STEPS) {
        env->log.max_steps_termination += 1;
        return true;
    }
    return false;
}

void c_step(CartPole* env) {

    float force = env->actions[0] ? FORCE_MAG : -FORCE_MAG;

    // Physics calculations following the equations from the CartPole definition
    float costheta = cosf(env->theta);
    float sintheta = sinf(env->theta);
    
    float temp = (force + POLE_MASS * POLE_HALF_LENGTH * env->theta_dot * env->theta_dot * sintheta) / (CART_MASS + POLE_MASS);
    float thetaacc = (GRAVITY * sintheta - costheta * temp) / 
                     (POLE_HALF_LENGTH * (4.0f/3.0f - POLE_MASS * costheta * costheta / (CART_MASS + POLE_MASS)));
    float xacc = temp - POLE_MASS * POLE_HALF_LENGTH * thetaacc * costheta / (CART_MASS + POLE_MASS);
    
    // Update the state using Euler integration
    env->x += TAU * env->x_dot;
    env->x_dot += TAU * xacc;
    env->theta += TAU * env->theta_dot;
    env->theta_dot += TAU * thetaacc;
    
    
    bool done = is_done(env);
    env->rewards[0] = done ? 0.0f : 1.0f;
    env->dones[0] = done ? 1 : 0;    
    // env->dones[0] = 0;
    // env->rewards[0] = 0.0f;

    
    // // Apply a reward of 1 for each step
    // env->rewards[0] = 1.0f;
    // env->log.episode_return += 1.0f;
    env->steps += 1;

    
    if (done) {
        env->dones[0] = 1;
        env->log.episode_return += env->steps;
        env->log.episode_length = env->steps;
        add_log(env->log_buffer, &env->log);
        c_reset(env);
        memset(&env->log, 0, sizeof(Log));  // Reset log after add_log
    }

    compute_observations(env);
}

typedef struct Client {
    float width;
    float height;
    float scale;  // Scaling factor for rendering
} Client;

Client* make_client(CartPole* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = env->width;
    client->height = env->height;
    client->scale = 100.0f;  // Scale for rendering the cartpole

    // Window is properly initialized with dimensions from env
    InitWindow(env->width, env->height, "PufferLib Ray CartPole");
    SetTargetFPS(60);

    return client;
}

void c_render(Client* client, CartPole* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    if (IsKeyPressed(KEY_TAB)) {
        ToggleFullscreen();
    }

    BeginDrawing();
    ClearBackground((Color){230, 230, 230, 255});

    // Draw track
    DrawLine(0, client->height/2, client->width, client->height/2, BLACK);
    
    // Calculate cart position in pixels
    float cart_x = client->width/2 + env->x * client->scale;
    float cart_y = client->height/2;
    
    // Draw cart
    DrawRectangle(cart_x - 20, cart_y - 10, 40, 20, BLACK);
    
    // Draw pole
    float pole_x2 = cart_x + sinf(env->theta) * POLE_HALF_LENGTH * 2 * client->scale;
    float pole_y2 = cart_y - cosf(env->theta) * POLE_HALF_LENGTH * 2 * client->scale;
    DrawLineEx((Vector2){cart_x, cart_y}, (Vector2){pole_x2, pole_y2}, 5, RED);
    
    // Draw info
    DrawText(TextFormat("Steps: %i", env->steps), 10, 10, 20, BLACK);
    DrawText(TextFormat("Cart Position: %.2f", env->x), 10, 40, 20, BLACK);
    DrawText(TextFormat("Pole Angle: %.2f", env->theta * 180 / M_PI), 10, 70, 20, BLACK);

    EndDrawing();
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}