#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define ROWS 6
#define COLS 7
#define WIN_COND 4
#define BOARD_SIZE (ROWS * COLS)

typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
};

typedef struct Client Client;
typedef struct CConnect4 CConnect4;
struct CConnect4 {
    float* observations;
    double* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    Log log;
    Client* client;

    uint64_t player_pieces;
    uint64_t env_pieces;
    int heights[COLS];
    int num_moves;
    float cumulative_reward;

    int selfplay;
    int current_player;  // 0 = player 1 (first mover), 1 = player 2
};

// Bottom position bitmask for column-major bitboard (col 0 bits 0-5, col 1 bits 7-12, etc.)
static inline uint64_t bottom_mask(int col) {
    return 1ULL << (col * (ROWS + 1));
}

static inline uint64_t column_mask(int col) {
    return ((1ULL << ROWS) - 1) << (col * (ROWS + 1));
}

static inline bool can_play(const CConnect4* env, int col) {
    return (col >= 0 && col < COLS) &&
           !((env->player_pieces | env->env_pieces) & (1ULL << (env->heights[col])));
}

static inline void play_move(uint64_t* pieces, int* heights, int col) {
    *pieces |= 1ULL << heights[col];
    heights[col]++;
}

static inline bool check_win(uint64_t pieces) {
    // Horizontal
    uint64_t m = pieces & (pieces >> (ROWS + 1));
    if (m & (m >> (2 * (ROWS + 1)))) return true;
    // Vertical
    m = pieces & (pieces >> 1);
    if (m & (m >> 2)) return true;
    // Diagonal /
    m = pieces & (pieces >> ROWS);
    if (m & (m >> (2 * ROWS))) return true;
    // Diagonal backslash
    m = pieces & (pieces >> (ROWS + 2));
    if (m & (m >> (2 * (ROWS + 2)))) return true;
    return false;
}

static inline bool board_full(const CConnect4* env) {
    return env->num_moves >= BOARD_SIZE;
}

// Write observation for a player: own pieces = 1.0, opponent pieces = -1.0
static void write_obs(float* obs, uint64_t own_pieces, uint64_t opp_pieces) {
    for (int col = 0; col < COLS; col++) {
        for (int row = 0; row < ROWS; row++) {
            int idx = col * ROWS + row;
            uint64_t bit = 1ULL << (col * (ROWS + 1) + row);
            if (own_pieces & bit) obs[idx] = 1.0f;
            else if (opp_pieces & bit) obs[idx] = -1.0f;
            else obs[idx] = 0.0f;
        }
    }
}

static void compute_observation(CConnect4* env) {
    if (env->selfplay) {
        write_obs(env->observations, env->player_pieces, env->env_pieces);
        write_obs(env->observations + BOARD_SIZE, env->env_pieces, env->player_pieces);
    } else {
        write_obs(env->observations, env->player_pieces, env->env_pieces);
        memset(env->observations + BOARD_SIZE, 0, BOARD_SIZE * sizeof(float));
    }
}

static void finish_game(CConnect4* env, float reward) {
    env->rewards[0] = reward;
    env->terminals[0] = 1.0f;
    env->cumulative_reward += reward;
}

// ---- Negamax AI (for non-selfplay mode) ----

static int negamax(uint64_t player, uint64_t opponent, int heights[COLS],
                   int depth, int alpha, int beta, int num_moves) {
    if (num_moves >= BOARD_SIZE) return 0;
    // Check if current player can win immediately
    for (int col = 0; col < COLS; col++) {
        if ((player | opponent) & (1ULL << heights[col])) continue;
        if (heights[col] % (ROWS + 1) >= ROWS) continue;
        uint64_t next = player | (1ULL << heights[col]);
        if (check_win(next)) return (BOARD_SIZE + 1 - num_moves) / 2;
    }
    int max_score = (BOARD_SIZE - 1 - num_moves) / 2;
    if (beta > max_score) {
        beta = max_score;
        if (alpha >= beta) return beta;
    }
    if (depth <= 0) return 0;
    for (int col = 0; col < COLS; col++) {
        if ((player | opponent) & (1ULL << heights[col])) continue;
        if (heights[col] % (ROWS + 1) >= ROWS) continue;
        int h[COLS];
        memcpy(h, heights, sizeof(h));
        uint64_t next_player = player | (1ULL << h[col]);
        h[col]++;
        int score = -negamax(opponent, next_player, h, depth - 1, -beta, -alpha, num_moves + 1);
        if (score >= beta) return score;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

static int compute_env_move(CConnect4* env) {
    // Opening book
    if (env->num_moves == 1) {
        if (can_play(env, 3)) return 3;
    }
    if (env->num_moves == 2) {
        uint64_t center_bottom = 1ULL << (3 * (ROWS + 1));
        if (env->player_pieces & center_bottom) return 3;
    }

    int best_col = -1;
    int best_score = -1000;
    for (int col = 0; col < COLS; col++) {
        if (!can_play(env, col)) continue;
        int h[COLS];
        memcpy(h, env->heights, sizeof(h));
        uint64_t next = env->env_pieces | (1ULL << h[col]);
        h[col]++;
        if (check_win(next)) return col;
        int score = -negamax(env->player_pieces, next, h, 3, -1000, 1000, env->num_moves + 1);
        if (score > best_score || best_col < 0) {
            best_score = score;
            best_col = col;
        }
    }
    return best_col;
}

// ---- Core env functions ----

static void init(CConnect4* env) {
    env->player_pieces = 0;
    env->env_pieces = 0;
    for (int col = 0; col < COLS; col++)
        env->heights[col] = col * (ROWS + 1);
    env->num_moves = 0;
    env->cumulative_reward = 0.0f;
    env->current_player = 0;
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
    compute_observation(env);
}

void c_close(CConnect4* env) {}

static void c_reset(CConnect4* env) {
    env->log.score += env->cumulative_reward;
    env->log.episode_return += env->cumulative_reward;
    env->log.episode_length += (float)env->num_moves;
    env->log.n += 1.0f;
    init(env);
}

static void c_step(CConnect4* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;

    if (env->selfplay) {
        // In selfplay mode: current_player alternates
        // actions[0] = player 1 action, actions[1] = player 2 action
        int col = (int)env->actions[env->current_player];

        uint64_t* current_pieces = (env->current_player == 0) ? &env->player_pieces : &env->env_pieces;
        uint64_t* other_pieces = (env->current_player == 0) ? &env->env_pieces : &env->player_pieces;

        if (!can_play(env, col)) {
            // Invalid move: pick first valid column
            for (int c = 0; c < COLS; c++) {
                if (can_play(env, c)) { col = c; break; }
            }
        }

        play_move(current_pieces, env->heights, col);
        env->num_moves++;

        if (check_win(*current_pieces)) {
            float r = (env->current_player == 0) ? 1.0f : -1.0f;
            finish_game(env, r);
            compute_observation(env);
            c_reset(env);
            return;
        }

        if (board_full(env)) {
            finish_game(env, 0.0f);
            compute_observation(env);
            c_reset(env);
            return;
        }

        env->current_player = 1 - env->current_player;
        compute_observation(env);
    } else {
        // Original single-agent mode vs scripted opponent
        int col = (int)env->actions[0];
        if (!can_play(env, col)) {
            finish_game(env, -1.0f);
            compute_observation(env);
            c_reset(env);
            return;
        }

        play_move(&env->player_pieces, env->heights, col);
        env->num_moves++;

        if (check_win(env->player_pieces)) {
            finish_game(env, 1.0f);
            compute_observation(env);
            c_reset(env);
            return;
        }

        if (board_full(env)) {
            finish_game(env, 0.0f);
            compute_observation(env);
            c_reset(env);
            return;
        }

        int env_col = compute_env_move(env);
        play_move(&env->env_pieces, env->heights, env_col);
        env->num_moves++;

        if (check_win(env->env_pieces)) {
            finish_game(env, -1.0f);
            compute_observation(env);
            c_reset(env);
            return;
        }

        if (board_full(env)) {
            finish_game(env, 0.0f);
            compute_observation(env);
            c_reset(env);
            return;
        }

        compute_observation(env);
    }
}

// ---- Raylib Rendering ----
#ifdef RAYLIB_H

#define CELL_SIZE 80
#define PADDING 10
#define CIRCLE_RADIUS 30
#define WINDOW_WIDTH (COLS * CELL_SIZE + 2 * PADDING)
#define WINDOW_HEIGHT (ROWS * CELL_SIZE + 2 * PADDING + CELL_SIZE)

typedef struct GameRenderer {
    Texture2D puffer_red;
    Texture2D puffer_yellow;
} GameRenderer;

static void init_renderer(GameRenderer* renderer) {
    Image img_red = LoadImage("resources/puffer_red.png");
    Image img_yellow = LoadImage("resources/puffer_yellow.png");
    ImageResize(&img_red, CIRCLE_RADIUS * 2, CIRCLE_RADIUS * 2);
    ImageResize(&img_yellow, CIRCLE_RADIUS * 2, CIRCLE_RADIUS * 2);
    renderer->puffer_red = LoadTextureFromImage(img_red);
    renderer->puffer_yellow = LoadTextureFromImage(img_yellow);
    UnloadImage(img_red);
    UnloadImage(img_yellow);
}

static void render(GameRenderer* renderer, CConnect4* env) {
    BeginDrawing();
    ClearBackground(BLUE);
    for (int col = 0; col < COLS; col++) {
        for (int row = 0; row < ROWS; row++) {
            int x = PADDING + col * CELL_SIZE + CELL_SIZE / 2;
            int y = PADDING + (ROWS - 1 - row) * CELL_SIZE + CELL_SIZE / 2 + CELL_SIZE;
            uint64_t bit = 1ULL << (col * (ROWS + 1) + row);
            if (env->player_pieces & bit) {
                DrawTexture(renderer->puffer_red, x - CIRCLE_RADIUS, y - CIRCLE_RADIUS, WHITE);
            } else if (env->env_pieces & bit) {
                DrawTexture(renderer->puffer_yellow, x - CIRCLE_RADIUS, y - CIRCLE_RADIUS, WHITE);
            } else {
                DrawCircle(x, y, CIRCLE_RADIUS, WHITE);
            }
        }
    }
    EndDrawing();
}

static void close_renderer(GameRenderer* renderer) {
    UnloadTexture(renderer->puffer_red);
    UnloadTexture(renderer->puffer_yellow);
}

#endif
