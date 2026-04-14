#include <stdlib.h>
#include <string.h>
#include "raylib.h"
#include <stdbool.h>
#include <time.h>

#define SQUARE_SIZE 32
#define INITIAL_TICKS_PER_FALL 6
#define ORIENT_HORIZONTAL 0
#define ORIENT_VERTICAL 1
#define ACTION_NO_OP 0
#define ACTION_LEFT 1
#define ACTION_RIGHT 2
#define ACTION_ROTATE 3
#define ACTION_SOFT_DROP 4
#define ACTION_HARD_DROP 5

// Required struct. Only use floats!
typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported in binding.c
    float viruses_cleared;

    float n; // Required as the last field
} Log;

// Required that you have some struct for your env
typedef struct {
    int total_rows;
    int total_columns;
} Client;

typedef struct {
    Client *client;
    Log log;

    float *observations;
    float *actions;
    float *rewards;
    float *terminals;
    int dim_obs;

    int n_rows;
    int n_cols;
    int *grid;

    int cap_color_a;
    int cap_color_b;
    int cap_orient;
    int cap_row;
    int cap_col;

    int tick;
    int tick_fall;
    int ticks_per_fall;

    int score;

    int viruses_remaining;
    int n_init_viruses;

    float episode_return;
    int viruses_cleared;
    int atn_count_soft_drop;
    int atn_count_hard_drop;
    int atn_count_rotate;

    unsigned int rng;
} DrMario;


void init(DrMario *env)
{
    env->grid=(int*)calloc(env->n_rows*env->n_cols,sizeof(int));
    if(env->grid==NULL)
    {
        exit(1);
    }

}

void allocate(DrMario *env)
{
        init(env);
        env->dim_obs = env->n_rows * env->n_cols + 5; // grid + capsule info
        env->observations = (float *)calloc(env->dim_obs, sizeof(float));
        if(env->observations == NULL)
        {
            exit(1);
        }
        env->actions = (float *)calloc(1, sizeof(float));
        if(env->actions == NULL)
        {
            exit(1);
        }
        env->rewards = (float *)calloc(1, sizeof(float));
        if(env->rewards == NULL)
        {
            exit(1);
        }
        env->terminals = (float *)calloc(1, sizeof(float));
        if(env->terminals == NULL)
        {
            exit(1);
        }
}

void c_close(DrMario *env)
{
    free(env->grid);
     if(IsWindowReady())
     {
        CloseWindow();
     }
}

void free_allocated(DrMario *env) {
	free(env->actions);
	free(env->observations);
	free(env->terminals);
	free(env->rewards);
	c_close(env);
}


void add_log(DrMario *env) {
    env->log.perf += env->viruses_cleared / (float)env->n_init_viruses;
    env->log.score += env->score;
    env->log.episode_length += env->tick;
    env->log.episode_return += env->episode_return;
    env->log.viruses_cleared += env->viruses_cleared;
    env->log.n++;
}

void place_viruses(DrMario *env) {
    for (int i = 0; i < env->n_init_viruses; i++) {
        int r = rand_r(&env->rng) % 8 + 8;
        int c = rand_r(&env->rng) % env->n_cols;
        int color = rand_r(&env->rng) % 3 + 1;
        env->grid[r * env->n_cols + c] = -color;
    }
}

void spawn_capsule(DrMario *env) {
    env->cap_color_a = rand_r(&env->rng) % 3 + 1;
    env->cap_color_b = rand_r(&env->rng) % 3 + 1;
    env->cap_orient  = ORIENT_HORIZONTAL;
    env->cap_row     = 0;
    env->cap_col     = env->n_cols / 2;
    env->tick_fall   = 0;
}

void c_reset(DrMario *env) {
    memset(env->grid, 0, env->n_rows * env->n_cols * sizeof(int));
    env->score = 0;
    env->tick = 0;
    env->tick_fall = 0;
    env->ticks_per_fall = INITIAL_TICKS_PER_FALL;
    env->viruses_remaining = env->n_init_viruses;
    env->episode_return = 0;
    env->viruses_cleared = 0;
    env->atn_count_soft_drop = 0;
    env->atn_count_hard_drop = 0;
    env->atn_count_rotate = 0;
    place_viruses(env);
    spawn_capsule(env);
}

void c_step(DrMario *env) {
    env->tick += 1;
    env->terminals[0] = 0;
    env->rewards[0] = 0;
    // movement logic tomorrow
}

void c_render(DrMario *env) {
    if (!IsWindowReady()) {
        InitWindow(SQUARE_SIZE * env->n_cols, SQUARE_SIZE * env->n_rows, "Dr Mario");
        SetTargetFPS(30);
    }
    if (IsKeyDown(KEY_ESCAPE)) exit(0);

    BeginDrawing();
    ClearBackground(BLACK);

    for (int r = 0; r < env->n_rows; r++) {
        for (int c = 0; c < env->n_cols; c++) {
            int cell = env->grid[r * env->n_cols + c];
            int x = c * SQUARE_SIZE;
            int y = r * SQUARE_SIZE;
            if (cell == 0) continue;

            Color color;
            if      (cell == 1 || cell == -1) color = RED;
            else if (cell == 2 || cell == -2) color = BLUE;
            else                              color = YELLOW;

            if (cell < 0) {
                DrawCircle(x + SQUARE_SIZE/2, y + SQUARE_SIZE/2, 
                          SQUARE_SIZE/2 - 2, color);
            } else {
                DrawRectangle(x + 2, y + 2, 
                             SQUARE_SIZE - 4, SQUARE_SIZE - 4, color);
            }
        }
    }

    // draw active capsule
    int x = env->cap_col * SQUARE_SIZE;
    int y = env->cap_row * SQUARE_SIZE;
    Color ca = (env->cap_color_a == 1) ? RED : (env->cap_color_a == 2) ? BLUE : YELLOW;
    Color cb = (env->cap_color_b == 1) ? RED : (env->cap_color_b == 2) ? BLUE : YELLOW;

    DrawRectangle(x + 2, y + 2, SQUARE_SIZE - 4, SQUARE_SIZE - 4, ca);
    if (env->cap_orient == ORIENT_HORIZONTAL) {
        DrawRectangle(x + SQUARE_SIZE + 2, y + 2, SQUARE_SIZE - 4, SQUARE_SIZE - 4, cb);
    } else {
        DrawRectangle(x + 2, y + SQUARE_SIZE + 2, SQUARE_SIZE - 4, SQUARE_SIZE - 4, cb);
    }

    EndDrawing();
}
