#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>

#include "raylib.h"

#define BOIDS_OBS_SIZE 24

#define BOID_WIDTH 16.0f
#define BOID_HEIGHT 16.0f
#define TOP_MARGIN 50.0f
#define BOTTOM_MARGIN 50.0f
#define LEFT_MARGIN 50.0f
#define RIGHT_MARGIN 50.0f
#define WALL_TURN_RANGE 95.0f
#define MIN_SPEED 1.4f
#define CRUISE_SPEED 3.0f
#define MAX_SPEED 4.2f
#define TURN_ACCEL 0.45f
#define EDGE_BOUNCE 0.55f
#define VISUAL_RANGE 240.0f
#define PROTECTED_RANGE (2.0f * BOID_WIDTH)
#define WIDTH 1080
#define HEIGHT 720
#define BOID_TEXTURE_PATH "./resources/shared/puffers_128.png"
#define EPS 1e-6f

#define ACTION_REWARD_SCALE 0.08f
#define SPEED_REWARD_SCALE 0.06f
#define MARGIN_REWARD_SCALE 0.035f
#define COHESION_REWARD_SCALE 0.080f
#define SEPARATION_REWARD_SCALE 0.100f
#define ALIGNMENT_REWARD_SCALE 0.060f

typedef struct {
    float score;
    float n;
    float t_margin_turn_reward;
    float t_cohesion_reward;
    float t_separation_reward;
    float t_alignment_reward;
    float t_speed_reward;
    float t_action_reward;
    float avg_speed;
    float avg_visual_count;
    float avg_protected_count;
} Log;

typedef struct {
    float x;
    float y;
} Vec2;

typedef Vec2 Velocity;

typedef struct {
    float x;
    float y;
    Velocity velocity;
} Boid;

typedef struct {
    Vec2 margin;
    Vec2 cohesion;
    Vec2 separation;
    Vec2 alignment;
    Vec2 desired;
    Vec2 centroid_delta;
    Vec2 avg_velocity;
    float visual_count;
    float protected_count;
    float nearest_dist;
    float separation_pressure;
} FlockSignal;

typedef struct {
    float width;
    float height;
    Texture2D boid_texture;
    unsigned frame_count;
    const char* screenshot_path;
    int screenshot_at;
    int max_render_frames;
} Client;

typedef struct {
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    Boid* boids;
    Boid* prev_boids;
    unsigned num_agents;
    float margin_turn_factor;
    float cohesion_factor;
    float separation_factor;
    float alignment_factor;
    unsigned tick;
    Log log;
    unsigned report_interval;
    Client* client;
    unsigned rng;
} Boids;

static inline float flmax(float a, float b) { return a > b ? a : b; }
static inline float flmin(float a, float b) { return a < b ? a : b; }
static inline float flclip(float x, float lo, float hi) { return flmin(hi, flmax(lo, x)); }
static inline float vec_len(Vec2 v) { return sqrtf(v.x*v.x + v.y*v.y); }
static inline Vec2 vec_add(Vec2 a, Vec2 b) { return (Vec2){a.x + b.x, a.y + b.y}; }
static inline Vec2 vec_sub(Vec2 a, Vec2 b) { return (Vec2){a.x - b.x, a.y - b.y}; }
static inline Vec2 vec_scale(Vec2 v, float s) { return (Vec2){v.x * s, v.y * s}; }
static inline float vec_dot(Vec2 a, Vec2 b) { return a.x*b.x + a.y*b.y; }

static inline Vec2 vec_limit(Vec2 v, float max_len) {
    float len = vec_len(v);
    if (len <= max_len || len <= EPS) return v;
    return vec_scale(v, max_len / len);
}

static inline Vec2 vec_norm(Vec2 v) {
    float len = vec_len(v);
    if (len <= EPS) return (Vec2){0.0f, 0.0f};
    return vec_scale(v, 1.0f / len);
}

static inline unsigned rng_next(Boids* env) {
    env->rng = 1664525u * env->rng + 1013904223u;
    return env->rng;
}

static inline float rndf(Boids* env, float lo, float hi) {
    float unit = (float)(rng_next(env) >> 8) * (1.0f / 16777215.0f);
    return lo + unit * (hi - lo);
}

static inline Vec2 boid_pos(const Boid* boid) {
    return (Vec2){boid->x, boid->y};
}

static inline void limit_boid_speed(Boid* boid) {
    float speed = vec_len(boid->velocity);
    if (speed < EPS) {
        boid->velocity.x = CRUISE_SPEED;
        boid->velocity.y = 0.0f;
        return;
    }

    if (speed < MIN_SPEED) {
        float scale = MIN_SPEED / speed;
        boid->velocity.x *= scale;
        boid->velocity.y *= scale;
    } else if (speed > MAX_SPEED) {
        float scale = MAX_SPEED / speed;
        boid->velocity.x *= scale;
        boid->velocity.y *= scale;
    }
}

static inline float action_component(float action) {
    int discrete = (int)floorf(action + 0.5f);
    if (discrete < 0) discrete = 0;
    if (discrete > 2) discrete = 2;
    return (float)(discrete - 1);
}

static inline float target_component(float steering) {
    if (steering > 0.18f) return 1.0f;
    if (steering < -0.18f) return -1.0f;
    return 0.0f;
}

static void spawn_boid(Boids* env, unsigned int i) {
    float angle = rndf(env, 0.0f, 6.28318530718f);
    float speed = rndf(env, MIN_SPEED, MAX_SPEED);
    env->boids[i].x = rndf(env, LEFT_MARGIN, WIDTH - RIGHT_MARGIN - BOID_WIDTH);
    env->boids[i].y = rndf(env, TOP_MARGIN, HEIGHT - BOTTOM_MARGIN - BOID_HEIGHT);
    env->boids[i].velocity.x = cosf(angle) * speed;
    env->boids[i].velocity.y = sinf(angle) * speed;
}

void init(Boids* env) {
    if (env->num_agents < 1) {
        printf("ERROR: num_agents must be bigger than 0\n");
        exit(1);
    }
    if (env->report_interval < 1) {
        printf("ERROR: report_interval must be bigger than 0\n");
        exit(1);
    }
    if (env->rng == 0) {
        env->rng = 1;
    }

    env->boids = (Boid*)calloc(env->num_agents, sizeof(Boid));
    env->prev_boids = (Boid*)calloc(env->num_agents, sizeof(Boid));
    if (env->boids == NULL || env->prev_boids == NULL) {
        printf("ERROR: failed to allocate boids\n");
        exit(1);
    }

    env->log = (Log){0};
    env->tick = 0;

    for (unsigned idx = 0; idx < env->num_agents; idx++) {
        spawn_boid(env, idx);
    }
}

static FlockSignal compute_flock_signal(const Boids* env, const Boid* boids, unsigned i) {
    FlockSignal signal = {0};
    const Boid* current = &boids[i];
    signal.nearest_dist = VISUAL_RANGE;

    const float max_x = WIDTH - BOID_WIDTH;
    const float max_y = HEIGHT - BOID_HEIGHT;
    float left_turn = LEFT_MARGIN + WALL_TURN_RANGE;
    float right_turn = max_x - RIGHT_MARGIN - WALL_TURN_RANGE;
    float top_turn = TOP_MARGIN + WALL_TURN_RANGE;
    float bottom_turn = max_y - BOTTOM_MARGIN - WALL_TURN_RANGE;

    if (current->x < left_turn) {
        signal.margin.x += (left_turn - current->x) / WALL_TURN_RANGE;
    } else if (current->x > right_turn) {
        signal.margin.x -= (current->x - right_turn) / WALL_TURN_RANGE;
    }
    if (current->y < top_turn) {
        signal.margin.y += (top_turn - current->y) / WALL_TURN_RANGE;
    } else if (current->y > bottom_turn) {
        signal.margin.y -= (current->y - bottom_turn) / WALL_TURN_RANGE;
    }
    signal.margin = vec_limit(signal.margin, 1.0f);

    for (unsigned j = 0; j < env->num_agents; j++) {
        if (i == j) continue;

        Vec2 delta = vec_sub(boid_pos(&boids[j]), boid_pos(current));
        float dist = vec_len(delta);
        if (dist < signal.nearest_dist) {
            signal.nearest_dist = dist;
        }

        if (dist < PROTECTED_RANGE) {
            float closeness = (PROTECTED_RANGE - dist) / PROTECTED_RANGE;
            signal.protected_count += 1.0f;
            signal.separation_pressure += closeness;

            if (dist > EPS) {
                signal.separation = vec_add(signal.separation,
                    vec_scale(delta, -closeness / dist));
            } else {
                float side = (j > i) ? -1.0f : 1.0f;
                signal.separation.x += side * closeness;
            }
        } else if (dist < VISUAL_RANGE) {
            signal.visual_count += 1.0f;
            signal.centroid_delta = vec_add(signal.centroid_delta, delta);
            signal.avg_velocity = vec_add(signal.avg_velocity, boids[j].velocity);
        }
    }

    if (signal.protected_count > 0.0f) {
        signal.separation = vec_limit(vec_scale(signal.separation,
            1.0f / signal.protected_count), 1.0f);
        signal.separation_pressure /= signal.protected_count;
    }

    if (signal.visual_count > 0.0f) {
        signal.centroid_delta = vec_scale(signal.centroid_delta, 1.0f / signal.visual_count);
        signal.avg_velocity = vec_scale(signal.avg_velocity, 1.0f / signal.visual_count);
        signal.cohesion = vec_limit(vec_scale(signal.centroid_delta, 1.0f / VISUAL_RANGE), 1.0f);
        signal.alignment = vec_limit(vec_scale(
            vec_sub(signal.avg_velocity, current->velocity), 1.0f / MAX_SPEED), 1.0f);
    } else {
        signal.nearest_dist = VISUAL_RANGE;
    }

    signal.desired = vec_add(signal.desired, vec_scale(signal.margin, env->margin_turn_factor));
    signal.desired = vec_add(signal.desired, vec_scale(signal.cohesion, env->cohesion_factor));
    signal.desired = vec_add(signal.desired, vec_scale(signal.separation, env->separation_factor));
    signal.desired = vec_add(signal.desired, vec_scale(signal.alignment, env->alignment_factor));
    signal.desired = vec_limit(signal.desired, 1.0f);

    return signal;
}

static void compute_observations(Boids* env) {
    float denom = (env->num_agents > 1) ? (float)(env->num_agents - 1) : 1.0f;
    const float max_x = WIDTH - BOID_WIDTH;
    const float max_y = HEIGHT - BOID_HEIGHT;

    for (unsigned i = 0; i < env->num_agents; i++) {
        const Boid* boid = &env->boids[i];
        FlockSignal signal = compute_flock_signal(env, env->boids, i);
        float speed = vec_len(boid->velocity);
        float* obs = &env->observations[i * BOIDS_OBS_SIZE];

        obs[0] = 2.0f * (boid->x / max_x) - 1.0f;
        obs[1] = 2.0f * (boid->y / max_y) - 1.0f;
        obs[2] = boid->velocity.x / MAX_SPEED;
        obs[3] = boid->velocity.y / MAX_SPEED;
        obs[4] = speed / MAX_SPEED;
        obs[5] = signal.margin.x;
        obs[6] = signal.margin.y;
        obs[7] = signal.cohesion.x;
        obs[8] = signal.cohesion.y;
        obs[9] = signal.separation.x;
        obs[10] = signal.separation.y;
        obs[11] = signal.alignment.x;
        obs[12] = signal.alignment.y;
        obs[13] = signal.desired.x;
        obs[14] = signal.desired.y;
        obs[15] = signal.visual_count / denom;
        obs[16] = signal.protected_count / denom;
        obs[17] = flclip(signal.nearest_dist / VISUAL_RANGE, 0.0f, 1.0f);
        obs[18] = flclip(signal.centroid_delta.x / VISUAL_RANGE, -1.0f, 1.0f);
        obs[19] = flclip(signal.centroid_delta.y / VISUAL_RANGE, -1.0f, 1.0f);
        obs[20] = signal.avg_velocity.x / MAX_SPEED;
        obs[21] = signal.avg_velocity.y / MAX_SPEED;
        obs[22] = vec_len(signal.desired);
        obs[23] = (speed - CRUISE_SPEED) / (MAX_SPEED - MIN_SPEED);
    }
}

static void apply_action(Boid* boid, float ax, float ay) {
    boid->velocity.x += ax * TURN_ACCEL;
    boid->velocity.y += ay * TURN_ACCEL;
    limit_boid_speed(boid);

    boid->x += boid->velocity.x;
    boid->y += boid->velocity.y;

    const float max_x = WIDTH - BOID_WIDTH;
    const float max_y = HEIGHT - BOID_HEIGHT;

    if (boid->x < 0.0f) {
        boid->x = 0.0f;
        if (boid->velocity.x < 0.0f) boid->velocity.x = -boid->velocity.x * EDGE_BOUNCE;
    } else if (boid->x > max_x) {
        boid->x = max_x;
        if (boid->velocity.x > 0.0f) boid->velocity.x = -boid->velocity.x * EDGE_BOUNCE;
    }

    if (boid->y < 0.0f) {
        boid->y = 0.0f;
        if (boid->velocity.y < 0.0f) boid->velocity.y = -boid->velocity.y * EDGE_BOUNCE;
    } else if (boid->y > max_y) {
        boid->y = max_y;
        if (boid->velocity.y > 0.0f) boid->velocity.y = -boid->velocity.y * EDGE_BOUNCE;
    }

    limit_boid_speed(boid);
}

void c_reset(Boids* env) {
    env->log = (Log){0};
    env->tick = 0;
    for (unsigned boid_idx = 0; boid_idx < env->num_agents; boid_idx++) {
        spawn_boid(env, boid_idx);
        if (env->terminals != NULL) env->terminals[boid_idx] = 0.0f;
        if (env->rewards != NULL) env->rewards[boid_idx] = 0.0f;
    }
    compute_observations(env);
}

static float compute_margin_reward(const Boid* boid, float factor) {
    const float max_x = WIDTH - BOID_WIDTH;
    const float max_y = HEIGHT - BOID_HEIGHT;
    float violation = 0.0f;

    if (boid->x < LEFT_MARGIN) violation += (LEFT_MARGIN - boid->x) / LEFT_MARGIN;
    if (boid->x > max_x - RIGHT_MARGIN) violation += (boid->x - (max_x - RIGHT_MARGIN)) / RIGHT_MARGIN;
    if (boid->y < TOP_MARGIN) violation += (TOP_MARGIN - boid->y) / TOP_MARGIN;
    if (boid->y > max_y - BOTTOM_MARGIN) violation += (boid->y - (max_y - BOTTOM_MARGIN)) / BOTTOM_MARGIN;

    if (violation <= 0.0f) {
        return MARGIN_REWARD_SCALE * factor;
    }
    return -3.0f * MARGIN_REWARD_SCALE * factor * flclip(violation, 0.0f, 1.5f);
}

void c_step(Boids* env) {
    bool manual_control = env->client != NULL && IsWindowReady() && IsKeyDown(KEY_LEFT_SHIFT);
    float mouse_x = manual_control ? (float)GetMouseX() : 0.0f;
    float mouse_y = manual_control ? (float)GetMouseY() : 0.0f;

    env->tick++;
    env->log = (Log){0};
    memcpy(env->prev_boids, env->boids, env->num_agents * sizeof(Boid));

    for (unsigned i = 0; i < env->num_agents; i++) {
        FlockSignal signal = compute_flock_signal(env, env->prev_boids, i);
        float ax = action_component(env->actions[i*2]);
        float ay = action_component(env->actions[i*2 + 1]);

        if (manual_control) {
            Vec2 to_mouse = vec_norm((Vec2){mouse_x - env->prev_boids[i].x, mouse_y - env->prev_boids[i].y});
            ax = to_mouse.x;
            ay = to_mouse.y;
        }

        float target_x = target_component(signal.desired.x);
        float target_y = target_component(signal.desired.y);
        float action_error = fabsf(ax - target_x) + fabsf(ay - target_y);
        float action_reward = ACTION_REWARD_SCALE * (1.0f - 0.5f * action_error);

        env->rewards[i] = action_reward;
        if (env->terminals != NULL) env->terminals[i] = 0.0f;
        apply_action(&env->boids[i], ax, ay);
    }

    float denom = (env->num_agents > 1) ? (float)(env->num_agents - 1) : 1.0f;
    for (unsigned i = 0; i < env->num_agents; i++) {
        Boid* boid = &env->boids[i];
        FlockSignal signal = compute_flock_signal(env, env->boids, i);
        float speed = vec_len(boid->velocity);

        float margin_reward = compute_margin_reward(boid, env->margin_turn_factor);

        float cohesion_reward = -COHESION_REWARD_SCALE * env->cohesion_factor;
        if (signal.visual_count > 0.0f) {
            float centroid_dist = vec_len(signal.centroid_delta);
            float cohesion_quality = 1.0f - flclip(centroid_dist / VISUAL_RANGE, 0.0f, 1.0f);
            cohesion_reward = COHESION_REWARD_SCALE * env->cohesion_factor
                * (0.35f + 0.65f * cohesion_quality);
        }

        float separation_reward = 0.0f;
        if (signal.protected_count > 0.0f) {
            separation_reward = -SEPARATION_REWARD_SCALE * env->separation_factor
                * (1.0f + 2.0f * signal.separation_pressure * signal.separation_pressure);
        } else if (signal.visual_count > 0.0f) {
            separation_reward = 0.35f * SEPARATION_REWARD_SCALE * env->separation_factor;
        }

        float alignment_reward = 0.0f;
        if (signal.visual_count > 0.0f) {
            float avg_speed = vec_len(signal.avg_velocity);
            float denom_speed = flmax(speed * avg_speed, EPS);
            float cos_align = flclip(vec_dot(boid->velocity, signal.avg_velocity) / denom_speed, -1.0f, 1.0f);
            alignment_reward = ALIGNMENT_REWARD_SCALE * env->alignment_factor * cos_align;
        }

        float speed_quality = 1.0f - flclip(fabsf(speed - CRUISE_SPEED) / (MAX_SPEED - MIN_SPEED), 0.0f, 1.0f);
        float speed_reward = SPEED_REWARD_SCALE * speed_quality;

        env->rewards[i] += margin_reward + cohesion_reward + separation_reward
            + alignment_reward + speed_reward;

        if (env->tick == env->report_interval) {
            env->log.score += env->rewards[i];
            env->log.t_margin_turn_reward += margin_reward;
            env->log.t_cohesion_reward += cohesion_reward;
            env->log.t_separation_reward += separation_reward;
            env->log.t_alignment_reward += alignment_reward;
            env->log.t_speed_reward += speed_reward;
            env->log.t_action_reward += env->rewards[i] - margin_reward - cohesion_reward
                - separation_reward - alignment_reward - speed_reward;
            env->log.avg_speed += speed;
            env->log.avg_visual_count += signal.visual_count / denom;
            env->log.avg_protected_count += signal.protected_count / denom;
            env->log.n += 1.0f;
        }
    }

    if (env->tick == env->report_interval) env->tick = 0;
    compute_observations(env);
}

void c_close_client(Client* client) {
    if (client->boid_texture.id != 0) {
        UnloadTexture(client->boid_texture);
    }
    if (IsWindowReady()) {
        CloseWindow();
    }
    free(client);
}

void c_close(Boids* env) {
    free(env->boids);
    free(env->prev_boids);
    if (env->client != NULL) {
        c_close_client(env->client);
    }
}

Client* make_client(Boids* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));

    client->width = WIDTH;
    client->height = HEIGHT;
    client->screenshot_path = getenv("BOIDS_SCREENSHOT");
    if (client->screenshot_path != NULL && client->screenshot_path[0] == '\0') {
        client->screenshot_path = NULL;
    }
    client->screenshot_at = getenv("BOIDS_SHOT_AT") != NULL ? atoi(getenv("BOIDS_SHOT_AT")) : 60;
    if (client->screenshot_at < 1) client->screenshot_at = 1;
    client->max_render_frames = getenv("BOIDS_MAX_RENDER_FRAMES") != NULL
        ? atoi(getenv("BOIDS_MAX_RENDER_FRAMES")) : 0;

#if defined(__linux__)
    if (getenv("DISPLAY") == NULL && getenv("WAYLAND_DISPLAY") == NULL) {
        TraceLog(LOG_WARNING, "No display is available for raylib rendering");
        free(client);
        return NULL;
    }
#endif

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
        c_close_client(client);
        return NULL;
    }

    return client;
}

void c_render(Boids* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
        if (env->client == NULL) {
            TraceLog(LOG_ERROR, "Failed to initialize client for rendering\n");
            if (getenv("BOIDS_MAX_RENDER_FRAMES") != NULL) exit(1);
            return;
        }
    }
    if (WindowShouldClose() || !IsWindowReady()) {
        TraceLog(LOG_WARNING, "Window is not ready or should close");
        return;
    }
    if (IsKeyDown(KEY_ESCAPE)) exit(0);

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    for (unsigned boid_idx = 0; boid_idx < env->num_agents; boid_idx++) {
        Boid* boid = &env->boids[boid_idx];
        float rotation = atan2f(boid->velocity.y, boid->velocity.x) * 57.295779513f;
        DrawTexturePro(
            env->client->boid_texture,
            (Rectangle){0.0f, 0.0f, 128.0f, 128.0f},
            (Rectangle){
                boid->x + BOID_WIDTH * 0.5f,
                boid->y + BOID_HEIGHT * 0.5f,
                BOID_WIDTH,
                BOID_HEIGHT
            },
            (Vector2){BOID_WIDTH * 0.5f, BOID_HEIGHT * 0.5f},
            rotation,
            WHITE
        );
    }
    EndDrawing();

    Client* client = env->client;
    client->frame_count++;
    if (client->screenshot_path != NULL
            && client->frame_count == (unsigned)client->screenshot_at) {
        TakeScreenshot(client->screenshot_path);
        TraceLog(LOG_INFO, "Wrote screenshot: %s", client->screenshot_path);
    }
    if (client->max_render_frames > 0
            && client->frame_count >= (unsigned)client->max_render_frames) {
        c_close_client(client);
        env->client = NULL;
        exit(0);
    }
}
