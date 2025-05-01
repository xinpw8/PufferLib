#ifndef BOIDS_H
#define BOIDS_H

#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include "raylib.h"
#include <math.h>
#include <string.h>
#include <limits.h>

#define TOP_MARGIN 50
#define BOTTOM_MARGIN 50
#define LEFT_MARGIN 50
#define RIGHT_MARGIN 50
#define VELOCITY_CAP 3
#define MARGIN_TURN_FACTOR 0.2f
#define VISUAL_RANGE 20
#define VISUAL_RANGE_SQUARED (VISUAL_RANGE * VISUAL_RANGE)
#define PROTECTED_RANGE 2
#define PROTECTED_RANGE_SQUARED (PROTECTED_RANGE * PROTECTED_RANGE)
#define CENTERING_FACTOR 0.0005f
#define AVOID_FACTOR 0.05f
#define MATCHING_FACTOR 0.05f
#define MAX_AVOID_DISTANCE_SQUARED (PROTECTED_RANGE_SQUARED * AVOID_FACTOR)
#define MAX_AVG_POSITION_SQUARED  (VISUAL_RANGE_SQUARED * CENTERING_FACTOR)
#define MAX_AVG_VELOCITY_SQUARED  (VELOCITY_CAP * 4 * MATCHING_FACTOR)
#define WIDTH 800
#define HEIGHT 600
#define BOID_WIDTH 32
#define BOID_HEIGHT 32

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    int n;
} Log;

typedef struct {
    float x;
    float y;
} Velocity;

typedef struct {
    float x;
    float y;
    Velocity velocity;
} Boid;

typedef struct Client Client;
typedef struct {
    // an array of shape (num_boids, 4) with the 4 values correspoinding to (x, y, velocity x, velocity y)
    float* observations;
    // an array of shape (num_boids, 2) with the 2 values correspoinding to (velocity x, velocity y)
    float* actions;
    // an array of shape (num_boids, 1) with the reward for each boid
    float* rewards;
    unsigned char* terminals;
    unsigned char* truncations;
    Boid* boids;
    unsigned int num_boids;
    int max_reward;
    int min_reward;
    int max_steps;
    int tick;
    Log log;
    Log* boid_logs;
    Client* client;
} Boids;

static void add_log(Boids *env, unsigned int i) {
    env->log.perf           += env->boid_logs[i].perf;
    env->log.score          += env->boid_logs[i].score;
    env->log.episode_return += env->boid_logs[i].episode_return;
    env->log.n              += 1.0f;

    /* clear per-boid log for next episode */
    env->boid_logs[i] = (Log){0};
}

static inline float flmax(float a, float b) { return a > b ? a : b; }
static inline float flmin(float a, float b) { return a > b ? b : a; }
static inline float flclip(float x,float lo,float hi){return flmin(hi,flmax(lo,x));}
static inline float rndf(float lo,float hi){return lo + (float)rand()/(float)RAND_MAX*(hi-lo);}

static void respawn_boid(Boids *env, unsigned int i) {
    env->boids[i].x = rndf(LEFT_MARGIN,  WIDTH  - RIGHT_MARGIN);
    env->boids[i].y = rndf(BOTTOM_MARGIN,HEIGHT - TOP_MARGIN);
    env->boids[i].velocity.x = 0;
    env->boids[i].velocity.y = 0;
    env->boid_logs[i]       = (Log){0};
}

void init(Boids *env) {
    /* dynamic allocs - ONLY allocate C-specific data */
    env->boids        = calloc(env->num_boids,     sizeof(Boid));
    env->boid_logs    = calloc(env->num_boids,     sizeof(Log));
    env->log          = (Log){0};
    env->tick         = 0;
    env->max_steps    = 1000;

    /* positions & velocities */
    for (unsigned i = 0; i < env->num_boids; ++i) {
        env->boids[i].x = rndf(LEFT_MARGIN,  WIDTH  - RIGHT_MARGIN);
        env->boids[i].y = rndf(BOTTOM_MARGIN,HEIGHT - TOP_MARGIN);
        env->boids[i].velocity.x = 0;
        env->boids[i].velocity.y = 0;
    }

    /* reward bounds for min-max normalisation */
    env->max_reward = 0;
    env->min_reward = -flmax(MAX_AVOID_DISTANCE_SQUARED * env->num_boids,
                             MAX_AVG_POSITION_SQUARED) - 2*MARGIN_TURN_FACTOR;
}

void c_free_env_specific(Boids* env) {
    free(env->boids);
    free(env->boid_logs);
}

static void compute_observations(Boids *env) {
    // Observation buffer shape is (num_boids, 4)
    for (unsigned i = 0; i < env->num_boids; ++i) {
        unsigned base_index = i * 4; // Boid 'i' data starts here
        env->observations[base_index + 0] = env->boids[i].x;
        env->observations[base_index + 1] = env->boids[i].y;
        env->observations[base_index + 2] = env->boids[i].velocity.x;
        env->observations[base_index + 3] = env->boids[i].velocity.y;
    }
}

void c_reset(Boids *env) {
    env->log = (Log){0};
    env->tick = 0;
    env->terminals[0] = 0;
    env->truncations[0] = 0;
    for (unsigned i = 0; i < env->num_boids; ++i)
        respawn_boid(env, i);
    compute_observations(env);
}

void c_step(Boids *env) {
    env->tick++;
    env->terminals[0] = 0;
    env->truncations[0] = 0;
    float total_reward = 0.0f;
    bool terminated = false;

    for (unsigned i = 0; i < env->num_boids; ++i) {
        /* ----- apply action ----- */
        Boid *b = &env->boids[i];
        float action_vx = env->actions[i * 2 + 0];
        float action_vy = env->actions[i * 2 + 1];

        b->velocity.x += flclip(action_vx, -VELOCITY_CAP, VELOCITY_CAP);
        b->velocity.y += flclip(action_vy, -VELOCITY_CAP, VELOCITY_CAP);

        b->x = flclip(b->x + b->velocity.x, 0, WIDTH  - BOID_WIDTH);
        b->y = flclip(b->y + b->velocity.y, 0, HEIGHT - BOID_HEIGHT);

        /* ----- reward calculation (per boid) ----- */
        float r = 0.0f;
        unsigned visual_cnt = 0;
        float vx_sum = 0, vy_sum = 0, x_sum = 0, y_sum = 0;

        for (unsigned j = 0; j < env->num_boids; ++j) {
            if (i == j) continue;
            Boid ob = env->boids[j];
            float dx = b->x - ob.x;
            float dy = b->y - ob.y;
            float dist2 = dx*dx + dy*dy;

            if (dist2 < PROTECTED_RANGE_SQUARED) {
                r -= (PROTECTED_RANGE_SQUARED - dist2) * AVOID_FACTOR;
            } else if (dist2 < VISUAL_RANGE_SQUARED) {
                x_sum += ob.x; y_sum += ob.y;
                vx_sum += ob.velocity.x; vy_sum += ob.velocity.y;
                ++visual_cnt;
            }
        }

        if (visual_cnt) {
            float x_avg  = x_sum  / visual_cnt;
            float y_avg  = y_sum  / visual_cnt;
            float vx_avg = vx_sum / visual_cnt;
            float vy_avg = vy_sum / visual_cnt;

            r -= fabsf(vx_avg - b->velocity.x) * MATCHING_FACTOR;
            r -= fabsf(vy_avg - b->velocity.y) * MATCHING_FACTOR;
            r -= fabsf(x_avg  - b->x) * CENTERING_FACTOR;
            r -= fabsf(y_avg  - b->y) * CENTERING_FACTOR;
        }

        if (b->y < TOP_MARGIN || b->y > HEIGHT - BOTTOM_MARGIN) r -= MARGIN_TURN_FACTOR;
        if (b->x < LEFT_MARGIN || b->x > WIDTH  - RIGHT_MARGIN) r -= MARGIN_TURN_FACTOR;

        float current_boid_reward = 2.0f * (r - env->min_reward) / (env->max_reward - env->min_reward) - 1.0f;
        total_reward += current_boid_reward;

        /* ----- per-boid log update ----- */
        env->boid_logs[i].episode_return += current_boid_reward;
        env->boid_logs[i].episode_length += 1.0f;

        /* ----- termination check (per boid) ----- */
        if (current_boid_reward <= -0.99f) {
            terminated = true;
            env->boid_logs[i].score = env->boid_logs[i].episode_return;
            env->boid_logs[i].perf  = (env->boid_logs[i].score/env->boid_logs[i].episode_length + 1.0f)*0.5f;
            add_log(env, i);
            respawn_boid(env, i);
        }
    }

    /* ----- Environment Level Updates ----- */
    env->rewards[0] = (env->num_boids > 0) ? total_reward / env->num_boids : 0.0f;

    if (terminated || env->tick >= env->max_steps) {
        env->terminals[0] = 1;
        if (!terminated && env->tick >= env->max_steps) {
            env->truncations[0] = 1;
            env->terminals[0] = 0;
        }
    } else if (env->tick >= env->max_steps) {
        env->truncations[0] = 1;
    }

    compute_observations(env);
}

typedef struct Client Client;
struct Client {
    float width;
    float height;
    Texture2D boid_texture;
};

Client* make_client(Boids* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    if (!client) {
        fprintf(stderr, "Failed to allocate client memory\n");
        return NULL;
    }
    
    client->width = WIDTH;
    client->height = HEIGHT;
    
    InitWindow(WIDTH, HEIGHT, "PufferLib Boids");
    SetTargetFPS(60);
    
    if (!IsWindowReady()) {
        fprintf(stderr, "Window failed to initialize\n");
        free(client);
        return NULL;
    }
    
    char texturePath[PATH_MAX] = {0};

    const char* candidatePaths[] = {
        "./resources/puffers_128.png",
        "./pufferlib/resources/puffers_128.png",
        "./pufferlib/pufferlib/resources/puffers_128.png"
    };

    int found = 0;
    for (size_t i = 0; i < sizeof(candidatePaths)/sizeof(candidatePaths[0]); i++) {
        if (FileExists(candidatePaths[i])) {
            strncpy(texturePath, candidatePaths[i], PATH_MAX - 1);
            found = 1;
            break;
        }
    }

    if (!found) {
        TraceLog(LOG_ERROR, "Failed to find puffers_128.png from current directory.");
        CloseWindow();
        free(client);
        return NULL;
    }

    client->boid_texture = LoadTexture(texturePath);
    if (client->boid_texture.id == 0) {
        TraceLog(LOG_ERROR, "Failed to load texture: %s", texturePath);
        CloseWindow();
        free(client);
        return NULL;
    }
    
    TraceLog(LOG_INFO, "Resource path resolution: %s", texturePath);
    TraceLog(LOG_INFO, "Window dimensions: %d x %d", GetScreenWidth(), GetScreenHeight());
    TraceLog(LOG_INFO, "Starting Boids demo with %d boids. Press ESC to exit.", env->num_boids);

    return client;
}

void c_close_client(Client* client) {
    UnloadTexture(client->boid_texture);
    CloseWindow();
    free(client);
}

void c_render(Boids* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
        if (env->client == NULL) {
            fprintf(stderr, "Failed to initialize client for rendering\n");
            return;
        }
    }
    Client* client = env->client;
    
    if (!WindowShouldClose() && IsWindowReady()) {
        if (IsKeyDown(KEY_ESCAPE)) {
            exit(0);
        }

        BeginDrawing();
        ClearBackground((Color){6, 24, 24, 255});

        for (unsigned int indx = 0; indx < env->num_boids; indx++) {
            DrawTexturePro(
                client->boid_texture,
                (Rectangle){
                    (env->boids[indx].velocity.x > 0) ? 0 : 128,
                    0,
                    128,
                    128,
                },
                (Rectangle){
                    env->boids[indx].x,
                    env->boids[indx].y,
                    BOID_WIDTH,
                    BOID_HEIGHT
                },
                (Vector2){0, 0},
                0,
                WHITE
            );
        }

        EndDrawing();
    } else {
        TraceLog(LOG_WARNING, "Window is not ready or should close");
    }
}

#endif // BOIDS_H