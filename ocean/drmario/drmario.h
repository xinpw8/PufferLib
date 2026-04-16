
#include <stdlib.h>
#include <string.h>
#include "raylib.h"
#include <stdbool.h>
#include <time.h>

#define SQUARE_SIZE 32
#define INITIAL_TICKS_PER_FALL 6

#define ROTATION_0 0
#define ROTATION_90 1
#define ROTATION_180 2
#define ROTATION_270 3

#define ACTION_NO_OP 0
#define ACTION_LEFT 1
#define ACTION_RIGHT 2
#define ACTION_DOWN 3
#define ACTION_ROTATE_LEFT 4
#define ACTION_ROTATE_RIGHT 5
#define ACTION_DROP 6

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
    int cap_row_1;
    int cap_col_1;
    int cap_row_2;
    int cap_col_2;
    bool cal_colliding_left;
    bool cal_colliding_right;
    bool cal_colliding_down; 
    bool cal_colliding_up;

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
        env->dim_obs = env->n_rows * env->n_cols + 5; 
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
    env->viruses_remaining = 0;
    int placed = 0;
    int attempts = 0;
    
    while (placed < env->n_init_viruses && attempts < 1000) {
        attempts++;
        int r = (rand_r(&env->rng) % 8) + 8;
        int c = rand_r(&env->rng) % env->n_cols;
        int idx = r * env->n_cols + c;
        
        // Scollision
        if (env->grid[idx] != 0) continue;
        
        int color = (rand_r(&env->rng) % 3) + 1;
        env->grid[idx] = -color;
        placed++;
        env->viruses_remaining++;
    }
}

void spawn_capsule(DrMario *env) {
    env->cap_color_a = rand_r(&env->rng) % 3 + 1;
    env->cap_color_b = rand_r(&env->rng) % 3 + 1;
    env->cap_orient  = ROTATION_0;
    env->cap_row_1     = -1;
    env->cap_col_1     = env->n_cols / 2;
    env->cap_row_2     = env->cap_row_1;
    env->cap_col_2     = env->cap_col_1 + 1;
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

void get_collisions(DrMario* env){
    env->cal_colliding_left = false;
    env->cal_colliding_right = false;
    env->cal_colliding_down = false;
    env->cal_colliding_up = false;

    if(env->grid[(env->cap_row_1+1) * env->n_cols + env->cap_col_1] != 0
        || env->grid[(env->cap_row_2+1) * env->n_cols + env->cap_col_2] != 0
        || env->cap_row_1 == env->n_rows - 1
        || env->cap_row_2 == env->n_rows - 1) {
        env->cal_colliding_down = true;
    }

    if(env->grid[(env->cap_row_1-1) * env->n_cols + env->cap_col_1] != 0
        || env->grid[(env->cap_row_2-1) * env->n_cols + env->cap_col_2] != 0) {
        env->cal_colliding_up = true;
    }

    if(env->grid[env->cap_row_1 * env->n_cols + env->cap_col_1 + 1] != 0
        || env->grid[env->cap_row_2 * env->n_cols + env->cap_col_2 + 1] != 0
        || env->cap_col_1 == env->n_cols - 1
        || env->cap_col_2 == env->n_cols - 1) {
        env->cal_colliding_right = true;
    }

    if(env->grid[env->cap_row_1 * env->n_cols + env->cap_col_1 - 1] != 0
        || env->grid[env->cap_row_2 * env->n_cols + env->cap_col_2 - 1] != 0
        || env->cap_col_1 == 0
        || env->cap_col_2 == 0) {
        env->cal_colliding_left = true;
    }
}

void rotate_cap(DrMario* env){
    int old_orient = env->cap_orient;
    int old_cap_row_2 = env->cap_row_2;
    int old_cap_col_2 = env->cap_col_2;

    if(env->actions[0] == ACTION_ROTATE_LEFT) {
        env->cap_orient = (env->cap_orient + 1) % 4;
    } else if(env->actions[0] == ACTION_ROTATE_RIGHT) {
        env->cap_orient = (env->cap_orient + 3) % 4;
    }

    env->cap_row_2 = env->cap_row_1;
    if(env->cap_orient == ROTATION_90) {
        env->cap_row_2 -= 1;
    } else if(env->cap_orient == ROTATION_270) {
        env->cap_row_2 += 1;
    }

    env->cap_col_2 = env->cap_col_1;
    if(env->cap_orient == ROTATION_0) {
        env->cap_col_2 += 1;
    } else if(env->cap_orient == ROTATION_180) {
        env->cap_col_2 -= 1;
    }

    if(env->grid[env->cap_row_2 * env->n_cols + env->cap_col_2] != 0) {
        env->cap_orient = old_orient;
        env->cap_row_2 = old_cap_row_2;
        env->cap_col_2 = old_cap_col_2;
    }
}

void move_cap(DrMario* env){
    env->tick_fall += 1;
    if(env->tick_fall >= env->ticks_per_fall)
    {
        env->tick_fall = 0;
        if(!env->cal_colliding_down) {
            env->cap_row_1 += 1;
        }
    }

    if(env->actions[0] == ACTION_LEFT && !env->cal_colliding_left) {
        env->cap_col_1 -= 1;
    } else if(env->actions[0] == ACTION_RIGHT && !env->cal_colliding_right) {
        env->cap_col_1 += 1;
    } else if(env->actions[0] == ACTION_DOWN && !env->cal_colliding_down) {
        env->cap_row_1 += 1;
        env->atn_count_soft_drop += 1;
    }
}

bool clear_lines(DrMario* env) {
    bool *to_clear = (bool*)calloc(env->n_rows * env->n_cols, sizeof(bool));
    if (!to_clear) return false;

    for (int r = 0; r < env->n_rows; r++) {
        int c = 0;
        while (c < env->n_cols) {
            int cell = env->grid[r * env->n_cols + c];
            if (cell == 0) { c++; continue; }
            int color = abs(cell);
            int run_end = c + 1;
            while (run_end < env->n_cols && abs(env->grid[r * env->n_cols + run_end]) == color)
                run_end++;
            if (run_end - c >= 4)
                for (int k = c; k < run_end; k++) to_clear[r * env->n_cols + k] = true;
            c = run_end;
        }
    }

    for (int col = 0; col < env->n_cols; col++) {
        int r = 0;
        while (r < env->n_rows) {
            int cell = env->grid[r * env->n_cols + col];
            if (cell == 0) { r++; continue; }
            int color = abs(cell);
            int run_end = r + 1;
            while (run_end < env->n_rows && abs(env->grid[run_end * env->n_cols + col]) == color)
                run_end++;
            if (run_end - r >= 4)
                for (int k = r; k < run_end; k++) to_clear[k * env->n_cols + col] = true;
            r = run_end;
        }
    }

    bool any_cleared = false;
    for (int i = 0; i < env->n_rows * env->n_cols; i++) {
        if (to_clear[i]) { any_cleared = true; break; }
    }

    if (!any_cleared) {
        free(to_clear);
        return false;
    }

    for (int i = 0; i < env->n_rows * env->n_cols; i++) {
        if (to_clear[i]) {
            if (env->grid[i] < 0) {
                env->viruses_remaining--;
                env->viruses_cleared++;
            }
            env->grid[i] = 0;
        }
    }
    free(to_clear);

    bool falling = true;
    while (falling) {
        falling = false;
        for (int r = env->n_rows - 2; r >= 0; r--) {
            for (int c = 0; c < env->n_cols; c++) {
                int cell = env->grid[r * env->n_cols + c];
                if (cell <= 0) continue;
                if (env->grid[(r + 1) * env->n_cols + c] != 0) continue;
                env->grid[(r + 1) * env->n_cols + c] = cell;
                env->grid[r * env->n_cols + c] = 0;
                falling = true;
            }
        }
    }

    clear_lines(env);
    return true;
}

void spawn_new_cap(DrMario* env) {
    if(env->cal_colliding_down) {
        env->grid[env->cap_row_1 * env->n_cols + env->cap_col_1] = env->cap_color_a;
        env->grid[env->cap_row_2 * env->n_cols + env->cap_col_2] = env->cap_color_b;
        clear_lines(env);
        spawn_capsule(env);
    }
}

void end_game_check(DrMario* env) {
    if(env->viruses_remaining <= 0) {
        env->terminals[0] = 1;
        env->rewards[0] = 1;
        c_reset(env);
    }

    if(env->cal_colliding_down && (env->cap_row_1 <= 0 || env->cap_row_2 <= 0)) {
        env->terminals[0] = 1;
        env->rewards[0] = -1;
        c_reset(env);
    }
}

void c_step(DrMario *env) {
    env->tick += 1;
    env->terminals[0] = 0;
    env->rewards[0] = 0;

    get_collisions(env);

    move_cap(env);

    rotate_cap(env);

    end_game_check(env);

    spawn_new_cap(env);
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
    int x1 = env->cap_col_1 * SQUARE_SIZE;
    int y1 = env->cap_row_1 * SQUARE_SIZE;
    int x2 = env->cap_col_2 * SQUARE_SIZE;
    int y2 = env->cap_row_2 * SQUARE_SIZE;
    Color ca = (env->cap_color_a == 1) ? RED : (env->cap_color_a == 2) ? BLUE : YELLOW;
    Color cb = (env->cap_color_b == 1) ? RED : (env->cap_color_b == 2) ? BLUE : YELLOW;

    DrawRectangle(x1 + 2, y1 + 2, SQUARE_SIZE - 4, SQUARE_SIZE - 4, ca);
    DrawRectangle(x2 + 2, y2 + 2, SQUARE_SIZE - 4, SQUARE_SIZE - 4, cb);

    EndDrawing();
}
