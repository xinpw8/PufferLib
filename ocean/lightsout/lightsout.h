#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "raylib.h"
typedef unsigned char obs_t;
#include "pufferenv.h"

#define GRID_SIZE 5
#define ACT_SIZES {GRID_SIZE * GRID_SIZE}
#define OBS_SIZE (GRID_SIZE * GRID_SIZE)
#define NUM_ATNS 1
#define PUF_STEPS_PER_SEC 4

// Only use floats.
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float scramble_p;
    float n; // Required as the last field.
};

typedef struct Client {
    int cell_size;
    int cursor_row;
    int cursor_col;
} Client;

struct Env {
    Log log;                     // Required field.
    Agent agents[1];
    int tag;
    int boundary_reached;
    int grid_size;
    int cell_size;
    int max_steps;
    int step_count;
    int lights_on;
    int prev_action;
    int last_action;
    float episode_return;
    float ema;
    float score_ema;
    float scramble_prob;
    unsigned char* grid;
    Client* client;
    int num_agents;
    int observation_size;
    unsigned int rng;
};
typedef Env LightsOut;

void step_grid(LightsOut* env, int idx) {
    if (idx < 0 || idx >= env->grid_size * env->grid_size) return;
    int row = idx/env->grid_size;
    int col = idx%env->grid_size;
    
    static const int dirs[5][2] = {{0,0}, {1,0}, {0,1}, {-1,0}, {0,-1}};
    for (int i = 0; i < 5; i++) {
        int dr = dirs[i][0];
        int dc = dirs[i][1];
        int r = row + dr;
        int c = col + dc;
        if (r >= 0 && r < env->grid_size && c >= 0 && c < env->grid_size) {
            int offset = r*env->grid_size + c;
            unsigned char old = env->grid[offset];
            env->grid[offset] = (unsigned char)!old;
            env->lights_on += old ? -1 : 1;
        }
    }
}

void init_lightsout(LightsOut* env) {
    int n = env->grid_size * env->grid_size;
    if (env->grid == NULL) {
        env->grid = (unsigned char*)calloc(n, sizeof(unsigned char));
    } else {
        memset(env->grid, 0, n * sizeof(unsigned char));
    }

    if (env->ema > 0.7f && env->score_ema > 0.0f) {
        env->scramble_prob = fminf(0.5f, env->scramble_prob + 0.01f);
    } else if (env->ema < 0.3f) {
        env->scramble_prob = fmaxf(0.15f, env->scramble_prob - 0.01f);
    }

    env->step_count = 0;
    env->lights_on = 0;
    env->prev_action = -1;
    env->last_action = -1;
    env->episode_return = 0.0f;

    for (int i = 0; i < n; i++) {
        float u = (float)rand_r(&env->rng) / (float)RAND_MAX;
        if (u < env->scramble_prob) {
            step_grid(env, i);
        }
    }
}

void puf_close(LightsOut* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
    free(env->grid);
    free(env->client);
}

void compute_observations(LightsOut* env) {
    obs_t* obs = env->agents[0].observations;
    for (int i = 0; i < env->grid_size * env->grid_size; i++) {
        obs[i] = env->grid[i];
    }
}

void puf_reset(LightsOut* env) {
    env->agents[0].rewards[0] = 0.0f;
    env->agents[0].terminals[0] = 0.0f;
    init_lightsout(env);
    compute_observations(env);
}

// Hold Left Shift + WASD to move the cursor, space to flip.
static int lightsout_human_controls(LightsOut *env) {
    if (!IsWindowReady() || !IsKeyDown(KEY_LEFT_SHIFT)) {
        return 0;
    }
    if (env->client == NULL) {
        return 0;
    }
    Client* client = env->client;
    if (IsKeyPressed(KEY_UP) || IsKeyPressed(KEY_W)) {
        client->cursor_row = (client->cursor_row - 1 + env->grid_size)
            % env->grid_size;
    }
    if (IsKeyPressed(KEY_DOWN) || IsKeyPressed(KEY_S)) {
        client->cursor_row = (client->cursor_row + 1) % env->grid_size;
    }
    if (IsKeyPressed(KEY_LEFT) || IsKeyPressed(KEY_A)) {
        client->cursor_col = (client->cursor_col - 1 + env->grid_size)
            % env->grid_size;
    }
    if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_D)) {
        client->cursor_col = (client->cursor_col + 1) % env->grid_size;
    }
    if (IsKeyPressed(KEY_SPACE)) {
        env->agents[0].actions[0] = (float)(
            client->cursor_row * env->grid_size + client->cursor_col);
        return 1;
    }
    return -1;
}

void puf_step(LightsOut* env) {
    if (lightsout_human_controls(env) < 0) {
        return;
    }
    int num_cells = env->grid_size * env->grid_size;
    int atn = (int)env->agents[0].actions[0];
    env->agents[0].terminals[0] = 0.0f;

    float reward = -0.02 * (36.0 / (env->grid_size * env->grid_size)); // Base step penalty.
    int prev_on = env->lights_on;
    if (atn < 0 || atn >= num_cells) {
        reward -= 0.5f; // Invalid action penalty.
    } else {
        if (atn == env->last_action) {
            reward -= 0.03f; // Penalty for pressing the same cell twice in a row.
        } else if (atn == env->prev_action) {
            reward -= 0.02f; // Penalty for 2-step loop (A,B,A).
        }
        if (env->client != NULL) {
            env->client->cursor_row = atn / env->grid_size;
            env->client->cursor_col = atn % env->grid_size;
        }
        step_grid(env, atn);
        env->prev_action = env->last_action;
        env->last_action = atn;
        int next_on = env->lights_on;
        reward += 0.005f * (float)(prev_on - next_on); // Dense shaping: improve when lights decrease.
    }
    env->step_count += 1;

    if (env->lights_on == 0) {
        reward = 2.0f; // Solved reward.
        env->ema = 0.85f * env->ema + 0.15f; // Update EMA of steps to solve.
        env->agents[0].terminals[0] = 1.0f;
    } else if (env->step_count >= env->max_steps) {
        reward -= 0.5f; // Timeout penalty during training.
        env->ema = 0.85f * env->ema; // Decay EMA since we failed to solve.
        env->agents[0].terminals[0] = 1.0f;
    }

    env->agents[0].rewards[0] = reward;
    env->episode_return += reward;

    if (env->agents[0].terminals[0] > 0.0f) {
        env->log.episode_return += env->episode_return;
        env->log.episode_length += (float)env->step_count;
        env->log.n += 1.0f;
        env->log.perf += (env->lights_on == 0) ? 1.0f : 0.0f;
        env->log.score += env->episode_return;
        env->log.scramble_p += env->scramble_prob;

        env->score_ema = 0.9f * env->score_ema + 0.1f * env->episode_return;
        init_lightsout(env);
    }

    compute_observations(env);
}

// Raylib client
static const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
static const Color PUFF_CYAN = (Color){0, 187, 187, 255};
static const Color PUFF_WHITE = (Color){241, 241, 241, 241};
static const Color LIGHT_OFF = (Color){0, 52, 52, 255};

Client* make_client(int cell_size, int grid_size) {
    Client* client= (Client*)malloc(sizeof(Client));
    client->cell_size = cell_size;
    client->cursor_row = 0;
    client->cursor_col = 0;
    InitWindow(grid_size*cell_size, grid_size*cell_size, "PufferLib LightsOut");
    SetTargetFPS(60);
    return client;
}

void puf_render(LightsOut* env) {
    if (env->client == NULL) {
        env->client = make_client(env->cell_size, env->grid_size);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    lightsout_human_controls(env);
    
    Client* client = env->client;
    
    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    int sz = client->cell_size;
    float gap = sz * 0.08f;
    float inset = gap * 0.5f;
    float tile = sz - gap;
    float roundness = 0.16f;
    int segs = 8;
    for (int y = 0; y < env->grid_size; y++) {
        for (int x = 0; x < env->grid_size; x++){
            int on = env->grid[y*env->grid_size + x];
            Rectangle rec = {x * sz + inset, y * sz + inset, tile, tile};
            DrawRectangleRounded(rec, roundness, segs, on ? PUFF_CYAN : LIGHT_OFF);
        }
    }
    Rectangle cursor = {
        client->cursor_col * sz + inset,
        client->cursor_row * sz + inset,
        tile,
        tile
    };
    DrawRectangleRoundedLinesEx(cursor, roundness, segs, 3.0f, PUFF_WHITE);

    if (env->agents[0].terminals[0] > 0.0f) {
        const char* msg = "Solved";
        int font_size = 32;
        int text_w = MeasureText(msg, font_size);
        int screen_w = GetScreenWidth();
        int screen_h = GetScreenHeight();

        DrawRectangle(0, 0, screen_w, screen_h, (Color){0, 0, 0, 120}); // dim overlay
        DrawText(msg, (screen_w - text_w) / 2, (screen_h - font_size) / 2, font_size, PUFF_WHITE);
    }

    EndDrawing();
    puf_web_vsync();
}

// --- Native trainer (pufferl) API ---
void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "scramble_p", log->scramble_p);
    dict_set(out, "n", log->n);
}

void puf_init(Env* env, Dict* kwargs) {
    env->grid_size = GRID_SIZE;
    env->cell_size = 640 / GRID_SIZE;
    if (640 % GRID_SIZE != 0) env->cell_size++; // ceil
    env->max_steps = dict_get(kwargs, "max_steps");
    env->observation_size = OBS_SIZE;
    env->num_agents = 1;
    env->ema = 0.5f;
    env->score_ema = 0.0f;
    env->scramble_prob = 0.15f;
    env->agents[0].action_mask = NULL;
    env->agents[0].policy = 0;
    init_lightsout(env);
}

