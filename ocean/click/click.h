/* Click: An environment to train an RL agent to click targets fast!
The agent has to move the mouse from one target to the next and then click.
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "raylib.h"
typedef float obs_t;
#include "pufferenv.h"

#define MAX_TARGETS 5
#define NUM_START_TARGETS 5
#define OBS_SIZE (3 + 3 * MAX_TARGETS + 2)
#define ACTION_SIZE 3
#define NUM_ATNS 3
#define ACT_SIZES {5, 5, 2}
#define TARGET_RADIUS_MIN 1
#define TARGET_RADIUS_MAX 10

#define CONTINUOUS 0
#define NUM_BINS 5
static float DELTA_X[NUM_BINS] = {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f};
static float DELTA_Y[NUM_BINS] = {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f};
static const float STATUS[2] = {0, 1};

const Color PUFF_CYAN = (Color){0, 187, 187, 255};

struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
};

typedef struct {
    float width;
    float height;
    Texture2D puffer;
    bool puffer_loaded;
    int frame;
} Client;

typedef struct {
    float x;
    float y;
    float status;
} Cursor;

typedef struct {
    float x;
    float y;
    float radius;
    float spawn_time;
} Target;

struct Env {
    Log log;
    Agent agents[1];
    int tag;
    int boundary_reached;
    int num_agents;
    Client* client;
    Cursor cursor;
    Target targets[MAX_TARGETS];
    int width;
    int height;
    int target_spawn_duration;
    unsigned int rng;
    int tick;
    int episode_length;
    int episode_return;
    int targets_hit;
    int targets_total;
    int lives;
    int prev_click;
    Vector2 prev_mouse;
    int human_input;
};
typedef Env ClickEnv;

static float spawn_radius(ClickEnv* env) {
    int span = TARGET_RADIUS_MAX - TARGET_RADIUS_MIN + 1;
    return (float)(TARGET_RADIUS_MIN + (int)(rand_r(&env->rng) % span));
}

void add_log(ClickEnv* env) {
    float hits = (float)env->targets_hit;
    float total = (float)env->targets_total;
    if (total < 1.0f) total = 1.0f;
    env->log.episode_length += env->tick;
    env->log.episode_return += hits;
    env->log.score += hits;
    env->log.perf += hits / total;
    env->log.n++;
}

// Hold Left Shift + mouse. Absolute cursor + click.
static void click_human_controls(ClickEnv *env) {
    if (!IsWindowReady() || !IsKeyDown(KEY_LEFT_SHIFT)) {
        env->human_input = 0;
        return;
    }
    Vector2 mouse = GetMousePosition();
    env->prev_mouse = mouse;
    env->agents[0].actions[0] = mouse.x;
    env->agents[0].actions[1] = mouse.y;
    env->agents[0].actions[2] = IsMouseButtonPressed(MOUSE_LEFT_BUTTON) ? 1.0f : 0.0f;
    env->human_input = 1;
}

void init(ClickEnv* env) {
    env->tick = 0;
}

static void draw_ocean(int w, int h, int frame) {
    Color top    = (Color){ 30, 130, 170, 255 };
    Color bottom = (Color){  3,  35,  70, 255 };
    DrawRectangleGradientV(0, 0, w, h, top, bottom);

    for (int b = 0; b < 5; b++) {
        float baseY = h * (0.1f + 0.18f * b);
        for (int x = 0; x < w; x += 8) {
            float yy = baseY + sinf((x * 0.03f) + frame * 0.05f + b) * 6.0f;
            DrawRectangle(x, (int)yy, 8, 2, (Color){255, 255, 255, 18});
        }
    }

    for (int i = 0; i < 40; i++) {
        float bx = (i * 137) % w;
        float speed = 0.5f + (i % 5) * 0.25f;
        float by = h - fmodf(frame * speed + i * 53, (float)h);
        float br = 1.0f + (i % 3);
        DrawCircleLines((int)bx, (int)by, br, (Color){255, 255, 255, 40});
    }
}

void compute_observations(ClickEnv* env) {
    float* observations = env->agents[0].observations;
    observations[0] = env->cursor.x / env->width;
    observations[1] = env->cursor.y / env->height;
    observations[2] = env->cursor.status;

    int obs_idx = 3;
    for (int i = 0; i < MAX_TARGETS; i++) {
        if (env->targets[i].spawn_time >= 0) {
            observations[obs_idx]     = (env->targets[i].x - env->cursor.x) / env->width;
            observations[obs_idx + 1] = (env->targets[i].y - env->cursor.y) / env->height;
            observations[obs_idx + 2] = fminf(env->targets[i].radius / TARGET_RADIUS_MAX, 1.0f);
        } else {
            observations[obs_idx]     = 0.0f;
            observations[obs_idx + 1] = 0.0f;
            observations[obs_idx + 2] = 0.0f;
        }
        obs_idx += 3;
    }
    // OBS_SIZE reserved +2 (old width/height). Must write them: 5.0 obs is bf16
    // and the last two channels were leftover host garbage / NaNs.
    observations[obs_idx]     = (float)env->lives / 3.0f;
    observations[obs_idx + 1] = (env->episode_length > 0)
        ? (float)env->tick / (float)env->episode_length : 0.0f;
}

void puf_reset(ClickEnv* env) {
    env->tick = 0;
    env->episode_return = 0;
    env->lives = 3;
    env->targets_hit = 0;
    env->targets_total = NUM_START_TARGETS;
    env->prev_click = 0;
    env->prev_mouse = (Vector2){ -1.0f, -1.0f };

    for (int i = 0; i < MAX_TARGETS; i++) {
        env->targets[i].spawn_time = -1;
        env->targets[i].radius = 0;
    }
    for (int i = 0; i < NUM_START_TARGETS; i++) {
        env->targets[i].x = rand_r(&env->rng) % env->width;
        env->targets[i].y = rand_r(&env->rng) % env->height;
        env->targets[i].radius = spawn_radius(env);
        env->targets[i].spawn_time = 0;
    }

    env->cursor.x = env->width / 2 + rand_r(&env->rng) % 21 - 10;
    env->cursor.y = env->height / 2 + rand_r(&env->rng) % 21 - 10;
    env->cursor.status = 0;

    compute_observations(env);
}

static inline float clipf(float val, float min, float max) {
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

void puf_step(ClickEnv* env) {
    click_human_controls(env);
    float* actions = env->agents[0].actions;
    env->agents[0].rewards[0] = 0.0f;
    env->agents[0].terminals[0] = 0.0f;

    if (env->human_input) {
        env->cursor.x = actions[0];
        env->cursor.y = actions[1];
        env->cursor.status = actions[2];
    } else {
        if (CONTINUOUS) {
            env->cursor.x += actions[0];
            env->cursor.y += actions[1];
            env->cursor.status = actions[2];
        } else {
            env->cursor.x += DELTA_X[(int)actions[0]];
            env->cursor.y += DELTA_Y[(int)actions[1]];
            env->cursor.status = actions[2];
        }
    }

    env->cursor.x = clipf(env->cursor.x, 0, env->width);
    env->cursor.y = clipf(env->cursor.y, 0, env->height);

    // Edge-trigger matches the human demo (IsMouseButtonPressed) and the
    // discrete click head. Holding action==1 is one press, not a stream
    // of clicks.
    int want_click = (env->cursor.status >= 0.5f);
    int click_status = (want_click && !env->prev_click);
    env->prev_click = want_click;

    for (int i = 0; i < MAX_TARGETS; i++) {
        if (env->targets[i].spawn_time >= 0) {
            env->targets[i].spawn_time += 1;
            if (env->targets[i].spawn_time > env->target_spawn_duration) {
                env->targets[i].spawn_time = -1;
            }
        }
        if (env->targets[i].spawn_time == -1 && rand_r(&env->rng) % 100 < 20) {
            env->targets[i].x = rand_r(&env->rng) % env->width;
            env->targets[i].y = rand_r(&env->rng) % env->height;
            env->targets[i].radius = spawn_radius(env);
            env->targets[i].spawn_time = 0;
            env->targets_total += 1;
        }
    }

    // 4.0 "Agent learns!" : +1 on hit, empty clicks are free. The later
    // lives / -1 misclick penalty made "never click" the only stable policy.
    if (click_status == 1) {
        for (int j = 0; j < MAX_TARGETS; j++) {
            if (env->targets[j].spawn_time >= 0) {
                float dist_x = env->cursor.x - env->targets[j].x;
                float dist_y = env->cursor.y - env->targets[j].y;
                float distance = sqrt(dist_x * dist_x + dist_y * dist_y);
                if (distance <= env->targets[j].radius) {
                    env->targets_hit += 1;
                    env->agents[0].rewards[0] += 1.0f;
                    env->targets[j].spawn_time = -1;
                }
            }
        }
    }

    for (int i = 0; i < MAX_TARGETS; i++) {
        if (env->targets[i].spawn_time >= 0) {
            env->targets[i].radius += 1;
        }
    }

    if (env->tick >= env->episode_length) {
        env->agents[0].terminals[0] = 1;
        add_log(env);
        puf_reset(env);
    } else {
        env->agents[0].terminals[0] = 0;
    }

    env->tick += 1;

    compute_observations(env);
}

void puf_render(ClickEnv* env) {
    if (IsKeyDown(KEY_ESCAPE))
        exit(0);
    if (IsKeyPressed(KEY_TAB))
        ToggleFullscreen();
    click_human_controls(env);

    if (env->client == NULL) {
        InitWindow(env->width, env->height, "Click environment");
        SetTargetFPS(60);
        env->client = (Client*)malloc(sizeof(Client));
        env->client->width  = env->width;
        env->client->height = env->height;
        env->client->frame  = 0;
        env->client->puffer = LoadTexture("resources/shared/puffers_128.png");
        env->client->puffer_loaded = (env->client->puffer.id != 0);

        if (env->client->puffer_loaded) {
            SetTextureFilter(env->client->puffer, TEXTURE_FILTER_BILINEAR);
        }
    }

    Client* client = env->client;
    client->frame++;

    BeginDrawing();

    draw_ocean(env->width, env->height, client->frame);

    for (int i = 0; i < MAX_TARGETS; i++) {
        if (env->targets[i].spawn_time >= 0) {
            float r = env->targets[i].radius;

            if (client->puffer_loaded) {
                float cell = 128.0f;
                float drawSz = 2.0f * r * 1.4f;
                Rectangle src = {0, 0, cell, cell};
                Rectangle dest = {
                    env->targets[i].x - drawSz * 0.5f,
                    env->targets[i].y - drawSz * 0.5f,
                    drawSz,
                    drawSz
                };
                DrawTexturePro(client->puffer, src, dest, (Vector2){0, 0}, 0.0f, WHITE);
            }
        }
    }

    Color c = (env->cursor.status == 1) ? YELLOW : WHITE;
    float x = env->cursor.x;
    float y = env->cursor.y;
    Vector2 tip   = { x,      y      };
    Vector2 left  = { x,      y + 16 };
    Vector2 notch = { x + 4,  y + 12 };
    Vector2 right = { x + 11, y + 11 };
    DrawTriangle(tip, left, right, c);
    DrawTriangle(left, notch, right, c);
    DrawTriangleLines(tip, left, right, BLACK);
    DrawTriangleLines(left, notch, right, BLACK);

    DrawText(TextFormat("Timestep: %d", env->tick), 10, 10, 20, RAYWHITE);
    DrawText(TextFormat("Lives: %d", env->lives), 10, 35, 20, RAYWHITE);
    DrawText(TextFormat("Targets hit: %d", env->targets_hit), 200, 10, 20,
             (Color){180, 255, 180, 255});

    EndDrawing();
    puf_web_vsync();
}

void puf_close(ClickEnv* env) {
    if (env->client != NULL) {
        Client* client = env->client;
        if (client->puffer_loaded) {
            UnloadTexture(client->puffer);
        }
        CloseWindow();
        free(client);
        env->client = NULL;
    }
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n", log->n);
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->width = 800;
    env->height = 600;
    env->target_spawn_duration = dict_get(kwargs, "target_spawn_duration");
    env->episode_length = dict_get(kwargs, "episode_length");
    env->agents[0].action_mask = NULL;
    env->agents[0].policy = 0;
    // Trainer assigns rng = env_index before puf_init. rand_r(0) is defined
    // but keep a non-zero state so env 0 is not a special case.
    if (env->rng == 0) {
        env->rng = 1u + (unsigned int)env->tag;
    }
    init(env);
}
