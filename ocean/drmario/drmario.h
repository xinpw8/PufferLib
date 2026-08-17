#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"
#include <stdbool.h>
typedef float obs_t;
#include "pufferenv.h"

#define ACT_SIZES {7}
// Default 16x8 board, 3 planes + 12 scalars. Must match [env] n_rows/n_cols.
#define N_SCALAR_OBS 12
#define N_OBS_PLANES 3
#define OBS_SIZE 396
#define NUM_ATNS 1

#define SQUARE_SIZE 32
#define TICKS_PER_FALL 6
#define MAX_TICKS 4096
#define MAX_BOARD_CELLS 512

#define SCORE_SOFT_DROP 0.0f
#define SCORE_HARD_DROP 0.0f
#define SCORE_ROTATE 0.0f
#define SCORE_KILL_VIRUS 1000.0f
#define SCORE_PLACE_NEXT_TO_SAME_COLOR 10.0f
#define SCORE_NO_LINE_CLEARS -10.0f
#define SCORE_CLEAR_LINE 500.0f

#define REWARD_SOFT_DROP 0.0f
#define REWARD_HARD_DROP 0.0f
#define REWARD_ROTATE 0.0f
#define REWARD_KILL_VIRUS 1.0f
#define REWARD_PLACE_NEXT_TO_SAME_COLOR 0.01f
#define REWARD_NO_LINE_CLEARS -0.01f
#define REWARD_CLEAR_LINE 0.5f
#define REWARD_HEIGHT 0.0f

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

struct Log {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    float viruses_cleared;
    float n; // Required as the last field
};

// Required that you have some struct for your env
typedef struct {
    int total_rows;
    int total_columns;
} Client;

struct Env {
    Client *client;
    Log log;
    Agent agents[1];
    int tag;
    int boundary_reached;

    int dim_obs;

    int num_agents;

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

    bool cap_colliding_left;
    bool cap_colliding_right;
    bool cap_colliding_down; 
    bool cap_colliding_up;

    float cap_colliding_color_hor_1;
    float cap_colliding_color_hor_2;
    float cap_colliding_color_ver_1;
    float cap_colliding_color_ver_2;

    int tick;
    int tick_fall;
    int ticks_per_fall;

    int score;
    int stage;

    int viruses_remaining;
    int n_init_viruses;

    float episode_return;
    int viruses_cleared;

    int viruses_cleared_step;
    int lines_cleared_step;

    int atn_count_soft_drop;
    int atn_count_hard_drop;
    int atn_count_rotate;

    unsigned int rng;
};
typedef Env DrMario;

static inline int drm_cells(DrMario *env) {
    return env->n_rows * env->n_cols;
}

// Above the bottle is empty; walls and the floor are solid.
static inline bool drm_solid(DrMario *env, int r, int c) {
    if (r < 0) {
        return false;
    }
    if (r >= env->n_rows || c < 0 || c >= env->n_cols) {
        return true;
    }
    return env->grid[r * env->n_cols + c] != 0;
}

static inline bool drm_in_board(DrMario *env, int r, int c) {
    return r >= 0 && r < env->n_rows && c >= 0 && c < env->n_cols;
}

void init(DrMario *env) {
    env->grid = (int*)calloc((size_t)drm_cells(env), sizeof(int));
    assert(env->grid != NULL);
}

void puf_close(DrMario *env) {
    free(env->grid);
    if (IsWindowReady()) {
       CloseWindow();
    }
}

void add_log(DrMario *env) {
    env->log.perf += env->viruses_cleared / (float)env->n_init_viruses;
    env->log.score += env->score;
    env->log.episode_length += env->tick;
    env->log.episode_return += env->episode_return;
    env->log.viruses_cleared += env->viruses_cleared;
    env->log.n++;
}

void compute_observations(DrMario *env) {
    int cells = drm_cells(env);
    float* obs = env->agents[0].observations;
    
    float* plane_occupied = obs;
    float* plane_viruses = obs + cells;
    float* plane_colors = obs + 2 * cells;

    for (int i = 0; i < cells; i++) {
        int cell = env->grid[i];
        plane_occupied[i] = cell != 0 ? 1.0f : 0.0f;
        plane_viruses[i] = cell < 0 ? 1.0f : 0.0f;
        plane_colors[i] = cell != 0 ? abs(cell) / 3.0f : 0.0f;
    }

    int r1 = env->cap_row_1, c1 = env->cap_col_1;
    int r2 = env->cap_row_2, c2 = env->cap_col_2;
    if (drm_in_board(env, r1, c1)) {
        int i = r1 * env->n_cols + c1;
        plane_occupied[i] = 1.0f;
        plane_viruses[i] = 0.0f;
        plane_colors[i] = env->cap_color_a / 3.0f;
    }
    if (drm_in_board(env, r2, c2)) {
        int i = r2 * env->n_cols + c2;
        plane_occupied[i] = 1.0f;
        plane_viruses[i] = 0.0f;
        plane_colors[i] = env->cap_color_b / 3.0f;
    }

    int off = cells * N_OBS_PLANES;
    float safe_r1 = (r1 < 0) ? 0.0f : r1 / (float)(env->n_rows - 1);
    float safe_r2 = (r2 < 0) ? 0.0f : r2 / (float)(env->n_rows - 1);
    obs[off + 0] = env->cap_color_a / 3.0f;
    obs[off + 1] = env->cap_color_b / 3.0f;
    obs[off + 2] = env->cap_orient / 3.0f;
    obs[off + 3] = safe_r1;
    obs[off + 4] = c1 / (float)(env->n_cols - 1);
    obs[off + 5] = safe_r2;
    obs[off + 6] = c2 / (float)(env->n_cols - 1);
    obs[off + 7] = env->viruses_remaining / (float)env->n_init_viruses;
    obs[off + 8] = env->viruses_cleared_step / (float)env->n_init_viruses;
    obs[off + 9] = env->lines_cleared_step / 4.0f;
    obs[off + 10] = env->score / 10000.0f;
    obs[off + 11] = env->tick / (float)MAX_TICKS;
}

void place_viruses(DrMario *env) {
    env->viruses_remaining = 0;
    int placed = 0;
    int attempts = 0;
    
    int lo = env->n_rows > 8 ? env->n_rows - 8 : 0;
    int span = env->n_rows - lo;
    if (span < 1) {
        span = 1;
    }
    while (placed < env->n_init_viruses && attempts < 1000) {
        attempts++;
        int r = lo + (int)(rand_r(&env->rng) % (unsigned)span);
        int c = rand_r(&env->rng) % env->n_cols;
        int idx = r * env->n_cols + c;
        
        if (env->grid[idx] != 0) {
            continue;
        }
        
        int color = (rand_r(&env->rng) % 3) + 1;
        env->grid[idx] = -color;
        placed++;
        env->viruses_remaining++;
    }
}

void spawn_capsule(DrMario *env) {
    env->cap_color_a = rand_r(&env->rng) % 3 + 1;
    env->cap_color_b = rand_r(&env->rng) % 3 + 1;
    env->cap_orient = ROTATION_0;
    env->cap_row_1 = -1;
    env->cap_col_1 = env->n_cols / 2;
    env->cap_row_2 = env->cap_row_1;
    env->cap_col_2 = env->cap_col_1 + 1;
    env->tick_fall = 0;
}

void puf_reset(DrMario *env) {
    memset(env->grid, 0, (size_t)drm_cells(env) * sizeof(int));
    env->score = 0;
    env->tick = 0;
    env->tick_fall = 0;

    env->ticks_per_fall = TICKS_PER_FALL;
    env->viruses_remaining = env->n_init_viruses;

    env->episode_return = 0;
    env->viruses_cleared = 0;
    env->viruses_cleared_step = 0;
    env->lines_cleared_step = 0;
    env->atn_count_soft_drop = 0;
    env->atn_count_hard_drop = 0;
    env->atn_count_rotate = 0;
    env->cap_colliding_left = false;
    env->cap_colliding_right = false;
    env->cap_colliding_down = false;
    env->cap_colliding_up = false;
    // Do not zero rewards/terminals: last-step signal is still live.
    place_viruses(env);
    spawn_capsule(env);
    compute_observations(env);
}

void get_collisions(DrMario* env) {
    env->cap_colliding_left = drm_solid(env, env->cap_row_1, env->cap_col_1 - 1)
                           || drm_solid(env, env->cap_row_2, env->cap_col_2 - 1);
    env->cap_colliding_right = drm_solid(env, env->cap_row_1, env->cap_col_1 + 1)
                            || drm_solid(env, env->cap_row_2, env->cap_col_2 + 1);
    env->cap_colliding_down = drm_solid(env, env->cap_row_1 + 1, env->cap_col_1)
                           || drm_solid(env, env->cap_row_2 + 1, env->cap_col_2);
    env->cap_colliding_up = drm_solid(env, env->cap_row_1 - 1, env->cap_col_1)
                         || drm_solid(env, env->cap_row_2 - 1, env->cap_col_2);
}

static int drm_color_run(DrMario* env, int r, int c, int dr, int dc,
        int color, int skip_r, int skip_c) {
    int n = 0;
    r += dr;
    c += dc;
    while (drm_in_board(env, r, c)) {
        if (r == skip_r && c == skip_c) {
            r += dr;
            c += dc;
            continue;
        }
        if (abs(env->grid[r * env->n_cols + c]) != color) {
            break;
        }
        n++;
        r += dr;
        c += dc;
    }
    return n;
}

void get_color_collisions(DrMario* env) {
    env->cap_colliding_color_hor_1 = 0.0f;
    env->cap_colliding_color_ver_1 = 0.0f;
    env->cap_colliding_color_hor_2 = 0.0f;
    env->cap_colliding_color_ver_2 = 0.0f;

    int color1 = abs(env->cap_color_a);
    int color2 = abs(env->cap_color_b);
    env->cap_colliding_color_ver_1 = (float)(
        drm_color_run(env, env->cap_row_1, env->cap_col_1, 1, 0, color1,
            env->cap_row_2, env->cap_col_2)
        + drm_color_run(env, env->cap_row_1, env->cap_col_1, -1, 0, color1,
            env->cap_row_2, env->cap_col_2));
    env->cap_colliding_color_hor_1 = (float)(
        drm_color_run(env, env->cap_row_1, env->cap_col_1, 0, 1, color1,
            env->cap_row_2, env->cap_col_2)
        + drm_color_run(env, env->cap_row_1, env->cap_col_1, 0, -1, color1,
            env->cap_row_2, env->cap_col_2));
    env->cap_colliding_color_ver_2 = (float)(
        drm_color_run(env, env->cap_row_2, env->cap_col_2, 1, 0, color2,
            env->cap_row_1, env->cap_col_1)
        + drm_color_run(env, env->cap_row_2, env->cap_col_2, -1, 0, color2,
            env->cap_row_1, env->cap_col_1));
    env->cap_colliding_color_hor_2 = (float)(
        drm_color_run(env, env->cap_row_2, env->cap_col_2, 0, 1, color2,
            env->cap_row_1, env->cap_col_1)
        + drm_color_run(env, env->cap_row_2, env->cap_col_2, 0, -1, color2,
            env->cap_row_1, env->cap_col_1));
}

void rotate_cap(DrMario* env) {
    int action = (int)env->agents[0].actions[0];
    int new_orient;
    if (action == ACTION_ROTATE_LEFT) {
        new_orient = (env->cap_orient + 1) % 4;
    } else if (action == ACTION_ROTATE_RIGHT) {
        new_orient = (env->cap_orient + 3) % 4;
    } else {
        return;
    }

    int nr2 = env->cap_row_1;
    int nc2 = env->cap_col_1;
    if (new_orient == ROTATION_90) {
        nr2 -= 1;
    } else if (new_orient == ROTATION_270) {
        nr2 += 1;
    }
    if (new_orient == ROTATION_0) {
        nc2 += 1;
    } else if (new_orient == ROTATION_180) {
        nc2 -= 1;
    }

    // Second half may sit above the bottle while entering. Floor/walls/occupancy block.
    if (nc2 < 0 || nc2 >= env->n_cols || nr2 >= env->n_rows) {
        return;
    }
    if (nr2 >= 0 && env->grid[nr2 * env->n_cols + nc2] != 0) {
        return;
    }

    env->cap_orient = new_orient;
    env->cap_row_2 = nr2;
    env->cap_col_2 = nc2;
    env->atn_count_rotate += 1;
    env->score += SCORE_ROTATE;
    env->agents[0].rewards[0] += REWARD_ROTATE;
}

void move_cap(DrMario* env) {
    int action = (int)env->agents[0].actions[0];

    if (action == ACTION_LEFT && !env->cap_colliding_left) {
        env->cap_col_1 -= 1;
        env->cap_col_2 -= 1;
    } else if (action == ACTION_RIGHT && !env->cap_colliding_right) {
        env->cap_col_1 += 1;
        env->cap_col_2 += 1;
    } else if (action == ACTION_DOWN && !env->cap_colliding_down) {
        env->cap_row_1 += 1;
        env->cap_row_2 += 1;
        env->atn_count_soft_drop += 1;
        env->score += SCORE_SOFT_DROP;
        env->agents[0].rewards[0] += REWARD_SOFT_DROP;
    } else if (action == ACTION_DROP && !env->cap_colliding_down) {
        env->atn_count_hard_drop += 1;
        env->score += SCORE_HARD_DROP;
        env->agents[0].rewards[0] += REWARD_HARD_DROP;
        do {
            env->cap_row_1 += 1;
            env->cap_row_2 += 1;
            get_collisions(env);
        } while (!env->cap_colliding_down);
    }

    env->tick_fall += 1;
    if (env->tick_fall >= env->ticks_per_fall) {
        env->tick_fall = 0;
        get_collisions(env);
        if (!env->cap_colliding_down) {
            env->cap_row_1 += 1;
            env->cap_row_2 += 1;
        }
    }
}

static void apply_gravity(DrMario* env) {
    bool falling = true;
    int guard = drm_cells(env);
    while (falling && guard-- > 0) {
        falling = false;
        for (int r = env->n_rows - 2; r >= 0; r--) {
            for (int c = 0; c < env->n_cols; c++) {
                int idx = r * env->n_cols + c;
                int below = idx + env->n_cols;
                if (env->grid[idx] > 0 && env->grid[below] == 0) {
                    env->grid[below] = env->grid[idx];
                    env->grid[idx] = 0;
                    falling = true;
                }
            }
        }
    }
}

void clear_lines(DrMario* env) {
    int cells = drm_cells(env);
    unsigned char to_clear[MAX_BOARD_CELLS];
    if (cells > MAX_BOARD_CELLS) {
        return;
    }

    env->lines_cleared_step = 0;
    env->viruses_cleared_step = 0;

    for (int pass = 0; pass < cells; pass++) {
        memset(to_clear, 0, (size_t)cells);

        for (int r = 0; r < env->n_rows; r++) {
            for (int c = 0; c < env->n_cols; c++) {
                int cell = env->grid[r * env->n_cols + c];
                if (cell == 0) {
                    continue;
                }
                int color = abs(cell);
                int c_end = c + 1;
                while (c_end < env->n_cols
                        && abs(env->grid[r * env->n_cols + c_end]) == color) {
                    c_end += 1;
                }
                if (c_end - c >= 4) {
                    env->lines_cleared_step++;
                    for (int k = c; k < c_end; k++) {
                        to_clear[r * env->n_cols + k] = 1;
                    }
                }
                c = c_end - 1;
            }
        }

        for (int c = 0; c < env->n_cols; c++) {
            for (int r = 0; r < env->n_rows; r++) {
                int cell = env->grid[r * env->n_cols + c];
                if (cell == 0) {
                    continue;
                }
                int color = abs(cell);
                int r_end = r + 1;
                while (r_end < env->n_rows
                        && abs(env->grid[r_end * env->n_cols + c]) == color) {
                    r_end += 1;
                }
                if (r_end - r >= 4) {
                    env->lines_cleared_step++;
                    for (int k = r; k < r_end; k++) {
                        to_clear[k * env->n_cols + c] = 1;
                    }
                }
                r = r_end - 1;
            }
        }

        bool any_cleared = false;
        for (int k = 0; k < cells; k++) {
            if (!to_clear[k]) {
                continue;
            }
            any_cleared = true;
            if (env->grid[k] < 0) {
                env->viruses_remaining--;
                env->viruses_cleared++;
                env->viruses_cleared_step++;
            }
            env->grid[k] = 0;
        }
        if (!any_cleared) {
            break;
        }
        apply_gravity(env);
    }
}

void spawn_new_cap(DrMario* env) {
    if (!env->cap_colliding_down) {
        return;
    }
    // Still above the bottle and blocked: spawn lock. Do not write OOB.
    if (!drm_in_board(env, env->cap_row_1, env->cap_col_1)
            || !drm_in_board(env, env->cap_row_2, env->cap_col_2)) {
        return;
    }

    env->grid[env->cap_row_1 * env->n_cols + env->cap_col_1] = env->cap_color_a;
    env->grid[env->cap_row_2 * env->n_cols + env->cap_col_2] = env->cap_color_b;

    int row = env->cap_row_1 > env->cap_row_2 ? env->cap_row_1 : env->cap_row_2;
    env->agents[0].rewards[0] += row * REWARD_HEIGHT;

    get_color_collisions(env);

    int color_collisions = 0;
    if (env->cap_colliding_color_hor_1 >= 2) {
        color_collisions += (int)env->cap_colliding_color_hor_1;
    }
    if (env->cap_colliding_color_hor_2 >= 2) {
        color_collisions += (int)env->cap_colliding_color_hor_2;
    }
    if (env->cap_colliding_color_ver_1 >= 2) {
        color_collisions += (int)env->cap_colliding_color_ver_1;
    }
    if (env->cap_colliding_color_ver_2 >= 2) {
        color_collisions += (int)env->cap_colliding_color_ver_2;
    }

    if (color_collisions > 0) {
        env->score += color_collisions * SCORE_PLACE_NEXT_TO_SAME_COLOR;
        env->agents[0].rewards[0] += color_collisions * REWARD_PLACE_NEXT_TO_SAME_COLOR;
    }

    clear_lines(env);
    if (env->viruses_cleared_step > 0) {
        env->agents[0].rewards[0] += env->viruses_cleared_step * REWARD_KILL_VIRUS;
        env->score += env->viruses_cleared_step * SCORE_KILL_VIRUS;
    }

    if (env->lines_cleared_step > 0) {
        env->agents[0].rewards[0] += env->lines_cleared_step * REWARD_CLEAR_LINE;
        env->score += env->lines_cleared_step * SCORE_CLEAR_LINE;
    }

    if (env->lines_cleared_step == 0 && env->viruses_cleared_step == 0) {
        env->agents[0].rewards[0] += REWARD_NO_LINE_CLEARS;
        env->score += SCORE_NO_LINE_CLEARS;
    }

    spawn_capsule(env);
    get_collisions(env);
}

static void finish_episode(DrMario* env) {
    env->agents[0].terminals[0] = 1;
    env->episode_return += env->agents[0].rewards[0];
    add_log(env);
    puf_reset(env);
}

void end_game_check(DrMario* env) {
    if (env->viruses_remaining <= 0) {
        float speed_bonus = 1.0f / (1.0f + env->tick * 0.001f);
        env->agents[0].rewards[0] += 1.0f + speed_bonus;
        finish_episode(env);
        return;
    }

    // Game over when the next capsule cannot enter, not when a pill touches row 0.
    bool spawn_locked = env->cap_colliding_down
        && (env->cap_row_1 < 0 || env->cap_row_2 < 0);
    if (spawn_locked) {
        float fraction_remaining = env->viruses_remaining / (float)env->n_init_viruses;
        env->agents[0].rewards[0] -= 1.0f + fraction_remaining * 0.5f;
        finish_episode(env);
        return;
    }

    if (env->tick >= MAX_TICKS) {
        env->agents[0].terminals[0] = 1;
        finish_episode(env);
    }
}

// Hold Left Shift + A/D/S, Z/X rotate, space drop.
// Skip the step when Shift is held and no action this frame.
static int drmario_human_controls(DrMario *env) {
    if (!IsWindowReady() || !IsKeyDown(KEY_LEFT_SHIFT)) {
        return 0;
    }
    if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
        env->agents[0].actions[0] = ACTION_LEFT;
        return 1;
    }
    if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
        env->agents[0].actions[0] = ACTION_RIGHT;
        return 1;
    }
    if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) {
        env->agents[0].actions[0] = ACTION_DOWN;
        return 1;
    }
    if (IsKeyPressed(KEY_Z)) {
        env->agents[0].actions[0] = ACTION_ROTATE_LEFT;
        return 1;
    }
    if (IsKeyPressed(KEY_X)) {
        env->agents[0].actions[0] = ACTION_ROTATE_RIGHT;
        return 1;
    }
    if (IsKeyPressed(KEY_SPACE)) {
        env->agents[0].actions[0] = ACTION_DROP;
        return 1;
    }
    return -1;
}

void puf_step(DrMario *env) {
    if (drmario_human_controls(env) < 0) {
        return;
    }
    env->tick += 1;
    env->agents[0].terminals[0] = 0;
    env->agents[0].rewards[0] = 0;

    env->lines_cleared_step = 0;
    env->viruses_cleared_step = 0;

    get_collisions(env);

    if (!env->cap_colliding_down) {
        rotate_cap(env);
        get_collisions(env);
    }

    move_cap(env);
    get_collisions(env);
    spawn_new_cap(env);
    end_game_check(env);

    // finish_episode already folded this step's reward into the logged return.
    if (env->agents[0].terminals[0] == 0) {
        env->episode_return += env->agents[0].rewards[0];
    }

    compute_observations(env);
}

void puf_render(DrMario *env) {
    if (!IsWindowReady()) {
        InitWindow(SQUARE_SIZE*env->n_cols, SQUARE_SIZE*env->n_rows, "Dr Mario");
        SetTargetFPS(30);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    drmario_human_controls(env);

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    for (int r = 0; r < env->n_rows; r++) {
        for (int c = 0; c < env->n_cols; c++) {
            int cell = env->grid[r*env->n_cols + c];
            int x = c*SQUARE_SIZE;
            int y = r*SQUARE_SIZE;
            if (cell == 0) {
                continue;
            }

            Color color;
            if (cell == 1 || cell == -1) {
                color = RED;
            } else if (cell == 2 || cell == -2) {
                color = BLUE;
            } else {
                color = YELLOW;
            }

            if (cell < 0) {
                DrawCircle(x + SQUARE_SIZE/2, y + SQUARE_SIZE/2, 
                          SQUARE_SIZE/2 - 2, color);
            } else {
                DrawRectangle(x + 2, y + 2, 
                             SQUARE_SIZE - 4, SQUARE_SIZE - 4, color);
            }
        }
    }

    int x1 = env->cap_col_1*SQUARE_SIZE;
    int y1 = env->cap_row_1*SQUARE_SIZE;
    int x2 = env->cap_col_2*SQUARE_SIZE;
    int y2 = env->cap_row_2*SQUARE_SIZE;
    Color ca = (env->cap_color_a == 1) ? RED : (env->cap_color_a == 2) ? BLUE : YELLOW;
    Color cb = (env->cap_color_b == 1) ? RED : (env->cap_color_b == 2) ? BLUE : YELLOW;

    DrawRectangle(x1 + 2, y1 + 2, SQUARE_SIZE - 4, SQUARE_SIZE - 4, ca);
    DrawRectangle(x2 + 2, y2 + 2, SQUARE_SIZE - 4, SQUARE_SIZE - 4, cb);

    DrawText(TextFormat("Viruses: %d", env->viruses_remaining), 4, 4, 14, WHITE);
    DrawText(TextFormat("Score: %d", env->score), 4, 20, 14, WHITE);

    EndDrawing();
    puf_web_vsync();
}

// --- Native trainer (pufferl) API ---
void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "viruses_cleared", log->viruses_cleared);
    dict_set(out, "n", log->n);
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->n_rows = dict_get(kwargs, "n_rows");
    env->n_cols = dict_get(kwargs, "n_cols");
    env->n_init_viruses = dict_get(kwargs, "n_init_viruses");
    env->dim_obs = drm_cells(env) * N_OBS_PLANES + N_SCALAR_OBS;
    assert(env->dim_obs == OBS_SIZE);
    env->ticks_per_fall = TICKS_PER_FALL;
    env->agents[0].action_mask = NULL;
    env->agents[0].policy = 0;
    init(env);
}
