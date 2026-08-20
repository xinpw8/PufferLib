#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <limits.h>

#include "raylib.h"
typedef float obs_t;
#include "pufferenv.h"

#define TOP_MARGIN 50
#define BOTTOM_MARGIN 50
#define LEFT_MARGIN 50
#define RIGHT_MARGIN 50
#define VELOCITY_CAP 5
#define VISUAL_RANGE 20
#define PROTECTED_RANGE 100
#define WIDTH 1080
#define HEIGHT 720
#define BOID_WIDTH 32
#define BOID_HEIGHT 32
#define BOID_TEXTURE_PATH "./resources/puffers_128.png"

// Per-agent obs: relative (x, y, vx, vy) for each of MAX_BOIDS neighbors
// Config: num_boids = 64 in config/boids.ini
#define MAX_BOIDS 64
#define ACT_SIZES {1, 1}
#define OBS_SIZE (MAX_BOIDS * 4)
#define NUM_ATNS 2

struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
};

typedef struct {
    float x;
    float y;
} Velocity;

// Game entity (not pufferenv Agent)
typedef struct {
    float x;
    float y;
    Velocity velocity;
} BoidEntity;

typedef struct Client Client;

struct Env {
    Log log;
    Agent agents[MAX_BOIDS];
    int tag;
    int boundary_reached;
    int num_agents;
    BoidEntity* boids;
    unsigned int num_boids;
    float margin_turn_factor;
    float centering_factor;
    float avoid_factor;
    float matching_factor;
    unsigned tick;
    Log* boid_logs;
    unsigned report_interval;
    Client* client;
    unsigned int rng;
};
typedef Env Boids;

static inline float flmax(float a, float b) { return a > b ? a : b; }
static inline float flmin(float a, float b) { return a > b ? b : a; }
static inline float flclip(float x, float lo, float hi) { return flmin(hi, flmax(lo, x)); }
static inline float rndf(float lo, float hi, unsigned int* rng) {
    return lo + (float)rand_r(rng) / (float)RAND_MAX * (hi - lo);
}

static void respawn_boid(Boids* env, unsigned int i) {
    env->boids[i].x = rndf(LEFT_MARGIN, WIDTH - RIGHT_MARGIN, &env->rng);
    env->boids[i].y = rndf(BOTTOM_MARGIN, HEIGHT - TOP_MARGIN, &env->rng);
    env->boids[i].velocity.x = 0;
    env->boids[i].velocity.y = 0;
    env->boid_logs[i] = (Log){0};
}

void init(Boids* env) {
    env->boids = (BoidEntity*)calloc(env->num_boids, sizeof(BoidEntity));
    env->boid_logs = (Log*)calloc(env->num_boids, sizeof(Log));
    env->log = (Log){0};
    env->tick = 0;

    for (unsigned current_indx = 0; current_indx < env->num_boids; current_indx++) {
        env->boids[current_indx].x = rndf(LEFT_MARGIN, WIDTH - RIGHT_MARGIN, &env->rng);
        env->boids[current_indx].y = rndf(BOTTOM_MARGIN, HEIGHT - TOP_MARGIN, &env->rng);
        env->boids[current_indx].velocity.x = 0;
        env->boids[current_indx].velocity.y = 0;
    }
}

static void compute_observations(Boids* env) {
    for (unsigned i = 0; i < env->num_boids; i++) {
        float* obs = env->agents[i].observations;
        int idx = 0;
        for (unsigned j = 0; j < env->num_boids; j++) {
            obs[idx++] = (env->boids[j].x - env->boids[i].x) / WIDTH;
            obs[idx++] = (env->boids[j].y - env->boids[i].y) / HEIGHT;
            obs[idx++] = (env->boids[j].velocity.x - env->boids[i].velocity.x) / VELOCITY_CAP;
            obs[idx++] = (env->boids[j].velocity.y - env->boids[i].velocity.y) / VELOCITY_CAP;
        }
        // Pad remaining slots when num_boids < MAX_BOIDS
        while (idx < OBS_SIZE) {
            obs[idx++] = 0;
        }
    }
}

void puf_reset(Boids* env) {
    env->tick = 0;
    for (unsigned boid_indx = 0; boid_indx < env->num_boids; boid_indx++) {
        respawn_boid(env, boid_indx);
    }
    compute_observations(env);
}

void puf_step(Boids* env) {
    BoidEntity* current_boid;
    BoidEntity observed_boid;
    float vis_vx_sum, vis_vy_sum, vis_x_sum, vis_y_sum, vis_x_avg, vis_y_avg, vis_vx_avg, vis_vy_avg;
    float diff_x, diff_y, dist, protected_dist_sum, current_boid_reward;
    unsigned visual_count, protected_count;
    bool manual_control = IsWindowReady() && IsKeyDown(KEY_LEFT_SHIFT);
    float mouse_x = IsWindowReady() ? (float)GetMouseX() : 0;
    float mouse_y = IsWindowReady() ? (float)GetMouseY() : 0;

    env->tick++;
    env->log.score = 0;
    for (unsigned current_indx = 0; current_indx < env->num_boids; current_indx++) {
        current_boid = &env->boids[current_indx];
        float* actions = env->agents[current_indx].actions;
        if (manual_control) {
            current_boid->velocity.x = flclip(current_boid->velocity.x + (mouse_x - current_boid->x), -VELOCITY_CAP, VELOCITY_CAP);
            current_boid->velocity.y = flclip(current_boid->velocity.y + (mouse_y - current_boid->y), -VELOCITY_CAP, VELOCITY_CAP);
        } else if (actions != NULL) {
            current_boid->velocity.x = flclip(current_boid->velocity.x + 2 * actions[0], -VELOCITY_CAP, VELOCITY_CAP);
            current_boid->velocity.y = flclip(current_boid->velocity.y + 2 * actions[1], -VELOCITY_CAP, VELOCITY_CAP);
        }
        current_boid->x = flclip(current_boid->x + current_boid->velocity.x, 0, WIDTH - BOID_WIDTH);
        current_boid->y = flclip(current_boid->y + current_boid->velocity.y, 0, HEIGHT - BOID_HEIGHT);

        current_boid_reward = 0.0f;
        protected_dist_sum = 0.0f;
        protected_count = 0;
        visual_count = 0;
        vis_vx_sum = 0.0f;
        vis_vy_sum = 0.0f;
        vis_x_sum = 0.0f;
        vis_y_sum = 0.0f;
        for (unsigned observed_indx = 0; observed_indx < env->num_boids; observed_indx++) {
            if (current_indx == observed_indx)
                continue;
            observed_boid = env->boids[observed_indx];
            diff_x = current_boid->x - observed_boid.x;
            diff_y = current_boid->y - observed_boid.y;
            dist = sqrtf(diff_x * diff_x + diff_y * diff_y);
            if (dist < PROTECTED_RANGE) {
                protected_dist_sum += (PROTECTED_RANGE - dist);
                protected_count++;
            } else if (dist < VISUAL_RANGE) {
                vis_x_sum += observed_boid.x;
                vis_y_sum += observed_boid.y;
                vis_vx_sum += observed_boid.velocity.x;
                vis_vy_sum += observed_boid.velocity.y;
                visual_count++;
            }
        }
        if (protected_count > 0) {
            current_boid_reward -= flclip(protected_count / 5.0f, 0.0f, 1.0f) * env->avoid_factor;
        }
        if (visual_count) {
            vis_x_avg = vis_x_sum / visual_count;
            vis_y_avg = vis_y_sum / visual_count;
            vis_vx_avg = vis_vx_sum / visual_count;
            vis_vy_avg = vis_vy_sum / visual_count;

            current_boid_reward -= fabsf(vis_vx_avg - current_boid->velocity.x) * env->matching_factor;
            current_boid_reward -= fabsf(vis_vy_avg - current_boid->velocity.y) * env->matching_factor;
            current_boid_reward -= fabsf(vis_x_avg - current_boid->x) * env->centering_factor;
            current_boid_reward -= fabsf(vis_y_avg - current_boid->y) * env->centering_factor;
        }
        if (current_boid->y < TOP_MARGIN || current_boid->y > HEIGHT - BOTTOM_MARGIN) {
            current_boid_reward -= env->margin_turn_factor;
        } else {
            current_boid_reward += env->margin_turn_factor;
        }
        if (current_boid->x < LEFT_MARGIN || current_boid->x > WIDTH - RIGHT_MARGIN) {
            current_boid_reward -= env->margin_turn_factor;
        } else {
            current_boid_reward += env->margin_turn_factor;
        }

        if (env->agents[current_indx].rewards != NULL) {
            env->agents[current_indx].rewards[0] = current_boid_reward / 2.0f;
        }
        if (env->agents[current_indx].terminals != NULL) {
            env->agents[current_indx].terminals[0] = 0;
        }

        if (env->tick == env->report_interval) {
            env->log.score += current_boid_reward / 2.0f;
            env->log.n += 1.0f;
            env->tick = 0;
        }
    }

    compute_observations(env);
}

struct Client {
    float width;
    float height;
    Texture2D boid_texture;
};

void close_client(Client* client) {
    UnloadTexture(client->boid_texture);
    CloseWindow();
    free(client);
}

void puf_close(Boids* env) {
    free(env->boids);
    free(env->boid_logs);
    if (env->client != NULL) {
        close_client(env->client);
        env->client = NULL;
    }
}

Client* make_client(Boids* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));

    client->width = WIDTH;
    client->height = HEIGHT;

    InitWindow(WIDTH, HEIGHT, "PufferLib Boids");
    SetTargetFPS(60);

    if (!IsWindowReady()) {
        TraceLog(LOG_ERROR, "Window failed to initialize\n");
        free(client);
        return NULL;
    }

    client->boid_texture = LoadTexture(BOID_TEXTURE_PATH);
    if (client->boid_texture.id == 0) {
        TraceLog(LOG_ERROR, "Failed to load texture: %s", BOID_TEXTURE_PATH);
        close_client(client);
        return NULL;
    }

    return client;
}

void puf_render(Boids* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    for (unsigned boid_indx = 0; boid_indx < env->num_boids; boid_indx++) {
        DrawTexturePro(
            env->client->boid_texture,
            (Rectangle){
                (env->boids[boid_indx].velocity.x > 0) ? 0.0f : 128.0f,
                0,
                128,
                128,
            },
            (Rectangle){
                env->boids[boid_indx].x,
                env->boids[boid_indx].y,
                BOID_WIDTH,
                BOID_HEIGHT},
            (Vector2){0, 0},
            0,
            WHITE);
    }

    EndDrawing();
    puf_web_vsync();
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_boids = dict_get(kwargs, "num_boids");
    env->num_agents = (int)env->num_boids;
    if (env->num_boids > MAX_BOIDS) {
        fprintf(stderr, "boids: num_boids %u > MAX_BOIDS %d\n", env->num_boids, MAX_BOIDS);
        exit(1);
    }
    env->report_interval = dict_get(kwargs, "report_interval");
    env->margin_turn_factor = dict_get(kwargs, "margin_turn_factor");
    env->centering_factor = dict_get(kwargs, "centering_factor");
    env->avoid_factor = dict_get(kwargs, "avoid_factor");
    env->matching_factor = dict_get(kwargs, "matching_factor");
    for (unsigned i = 0; i < env->num_boids; i++) {
        env->agents[i].policy = 0;
        env->agents[i].action_mask = NULL;
    }
    init(env);
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n", log->n);
}

