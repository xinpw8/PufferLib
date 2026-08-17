#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include "raylib.h"
typedef float obs_t;
#include "pufferenv.h"

#define ACT_SIZES {104}
#define OBS_SIZE 35
#define NUM_ATNS 1
#define PUF_STEPS_PER_SEC 2

#define NUM_POINTS 24
#define NUM_CHECKERS 15
#define NUM_ACTIONS 104
#define BAR_POSITION 0
#define BG_WHITE 0
#define BG_BLACK 1
#define WHITE_DIRECTION -1
#define BLACK_DIRECTION 1
#define MAX_STEPS 5000

#define WIN_W 1000
#define WIN_H 640

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

// Log is a flat float struct. n is the episode count used by log_accum.
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float win_rate;
    float black_win_rate;
    float avg_moves_per_turn;
    float hit_rate;
    float checkers_home;
    float checkers_off;
    float n;
};

// 5.0 packing: Log, Agent[], tag, boundary_reached, then env fields. rng is set
// by the trainer before puf_init.
struct Env {
    Log log;
    Agent agents[1];
    int tag;
    int boundary_reached;
    int num_agents;
    int have_window;

    int8_t board[NUM_POINTS + 1];
    int8_t bar[2];
    int8_t off[2];
    int8_t dice[4];
    int8_t num_dice;
    int8_t dice_used;
    bool dice_available[4];
    int8_t current_player;
    bool must_enter_from_bar;

    int tick;
    float episode_return;
    int moves_this_episode;
    int hits_this_episode;
    int turns_this_episode;
    float opponent_random_prob;

    unsigned int rng;

    int pending_reset;
};
typedef Env Backgammon;

static int bg_randi(Backgammon* env, int lo, int hi) {
    return lo + (int)(rand_r(&env->rng) % (unsigned int)(hi - lo + 1));
}

static float bg_randf(Backgammon* env) {
    return (float)rand_r(&env->rng) / (float)RAND_MAX;
}

void roll_dice(Backgammon* env) {
    env->dice[0] = (int8_t)bg_randi(env, 1, 6);
    env->dice[1] = (int8_t)bg_randi(env, 1, 6);
    if (env->dice[0] == env->dice[1]) {
        env->dice[2] = env->dice[0];
        env->dice[3] = env->dice[0];
        env->num_dice = 4;
    } else {
        env->dice[2] = 0;
        env->dice[3] = 0;
        env->num_dice = 2;
    }
    env->dice_used = 0;
    for (int i = 0; i < 4; i++) {
        env->dice_available[i] = i < env->num_dice;
    }
}

int get_direction(int player) {
    return player == BG_WHITE ? WHITE_DIRECTION : BLACK_DIRECTION;
}

bool in_home_board(int player, int point) {
    if (player == BG_WHITE) {
        return point >= 1 && point <= 6;
    }
    return point >= 19 && point <= 24;
}

bool can_bear_off(Backgammon* env, int player) {
    for (int i = 1; i <= NUM_POINTS; i++) {
        if (player == BG_WHITE && env->board[i] > 0 && !in_home_board(BG_WHITE, i)) {
            return false;
        }
        if (player == BG_BLACK && env->board[i] < 0 && !in_home_board(BG_BLACK, i)) {
            return false;
        }
    }
    return env->bar[player] == 0;
}

bool is_dst_available(Backgammon* env, int position, int player) {
    int8_t val = env->board[position];
    if (val == 0) {
        return true;
    }
    if (player == BG_WHITE) {
        return val > 0 || val == -1;
    }
    return val < 0 || val == 1;
}

bool is_legal_move(Backgammon* env, int from, int die_index) {
    int8_t cp = env->current_player;
    int8_t die_value = env->dice[die_index];
    int direction = get_direction(cp);

    if (die_index < 0 || die_index >= 4 || !env->dice_available[die_index]) {
        return false;
    }

    if (env->bar[cp] > 0) {
        if (from != 0) {
            return false;
        }
        int entry = cp == BG_WHITE ? NUM_POINTS + 1 - die_value : die_value;
        return is_dst_available(env, entry, cp);
    }

    if (from < 1 || from > NUM_POINTS) {
        return false;
    }
    int8_t fvalue = env->board[from];
    if ((cp == BG_WHITE && fvalue <= 0) || (cp == BG_BLACK && fvalue >= 0)) {
        return false;
    }

    int dst = from + (die_value * direction);
    if (cp == BG_WHITE && dst < 1) {
        return can_bear_off(env, cp);
    }
    if (cp == BG_BLACK && dst > NUM_POINTS) {
        return can_bear_off(env, cp);
    }
    if (dst < 1 || dst > NUM_POINTS) {
        return false;
    }
    return is_dst_available(env, dst, cp);
}

bool has_legal_moves(Backgammon* env) {
    for (int i = 0; i <= NUM_POINTS; i++) {
        for (int d = 0; d < env->num_dice; d++) {
            if (is_legal_move(env, i, d)) return true;
        }
    }
    return false;
}

void compute_action_mask(Backgammon* env) {
    unsigned char* mask = env->agents[0].action_mask;
    if (mask == NULL) return;
    memset(mask, 0, NUM_ACTIONS);
    int n = 0;
    for (int from = 0; from <= NUM_POINTS; from++) {
        for (int d = 0; d < 4; d++) {
            if (is_legal_move(env, from, d)) {
                mask[from * 4 + d] = 1;
                n++;
            }
        }
    }
    // Sampler requires at least one valid bit. If the player is blocked, keep
    // the original illegal-move path (penalty, then opponent).
    if (n == 0) {
        memset(mask, 1, NUM_ACTIONS);
    }
}

void compute_observations(Backgammon* env) {
    float* obs = env->agents[0].observations;
    int cur = 0;
    for (int i = 1; i <= NUM_POINTS; i++) {
        obs[cur++] = (float)(env->board[i] / (float)NUM_CHECKERS);
    }
    obs[cur++] = (float)((float)env->bar[BG_WHITE] / NUM_CHECKERS);
    obs[cur++] = (float)((float)env->bar[BG_BLACK] / NUM_CHECKERS);
    obs[cur++] = (float)((float)env->off[BG_WHITE] / NUM_CHECKERS);
    obs[cur++] = (float)((float)env->off[BG_BLACK] / NUM_CHECKERS);
    for (int i = 0; i < 4; i++) {
        obs[cur++] = env->dice_available[i] ? (float)((float)env->dice[i] / 6.0f) : (float)0.0f;
    }
    obs[cur++] = (float)env->current_player;
    obs[cur++] = can_bear_off(env, BG_WHITE) ? (float)1.0f : (float)0.0f;
    obs[cur++] = can_bear_off(env, BG_BLACK) ? (float)1.0f : (float)0.0f;
    compute_action_mask(env);
}

static void land_checker(Backgammon* env, int dst, int8_t cp) {
    int blot = (cp == BG_WHITE) ? -1 : 1;
    if (env->board[dst] == blot) {
        env->bar[cp ^ 1]++;
        env->board[dst] = 0;
        env->hits_this_episode++;
    }
    env->board[dst] += (cp == BG_WHITE) ? 1 : -1;
}

void make_move(Backgammon* env, int from, int die_index) {
    int8_t cp = env->current_player;
    int8_t die_value = env->dice[die_index];
    int direction = get_direction(cp);

    env->dice_available[die_index] = false;
    env->dice_used++;
    env->moves_this_episode++;

    if (from == 0) {
        env->bar[cp]--;
        int dst = (cp == BG_WHITE) ? (NUM_POINTS + 1 - die_value) : die_value;
        land_checker(env, dst, cp);
        return;
    }

    env->board[from] += (cp == BG_WHITE) ? -1 : 1;
    int dst = from + (die_value * direction);
    if ((cp == BG_WHITE && dst < 1) || (cp == BG_BLACK && dst > NUM_POINTS)) {
        env->off[cp]++;
        return;
    }
    land_checker(env, dst, cp);
}

bool check_win(Backgammon* env, int player) {
    return env->off[player] == NUM_CHECKERS;
}

int score_move(Backgammon* env, int from, int die_index) {
    int die_value = env->dice[die_index];
    int score = 0;
    if (from == 0) {
        score += 100;
        int dst = die_value;
        if (env->board[dst] == 1) score += 50;
        return score;
    }
    int dst = from + die_value;
    if (dst > NUM_POINTS && can_bear_off(env, BG_BLACK)) {
        score += 200;
        return score;
    }
    if (dst > NUM_POINTS) return -1000;
    if (env->board[dst] == 1) score += 80;
    if (env->board[dst] == -1) score += 30;
    score += dst;
    if (env->board[from] == -1) score += 20;
    if (env->board[from] == -2 && from >= 1 && from <= 6) score -= 10;
    return score;
}

static int pick_legal_move(Backgammon* env, int* from_out, int* die_out, int prefer_heuristic) {
    int legal_moves[NUM_ACTIONS][2];
    int num_legal = 0;
    int best_score = -10000;
    int best_from = -1;
    int best_die = -1;
    for (int from = 0; from <= NUM_POINTS; from++) {
        for (int d = 0; d < env->num_dice; d++) {
            if (!is_legal_move(env, from, d)) {
                continue;
            }
            legal_moves[num_legal][0] = from;
            legal_moves[num_legal][1] = d;
            num_legal++;
            if (prefer_heuristic) {
                int score = score_move(env, from, d);
                if (score > best_score) {
                    best_score = score;
                    best_from = from;
                    best_die = d;
                }
            }
        }
    }
    if (num_legal == 0) {
        return 0;
    }
    if (prefer_heuristic && best_from >= 0) {
        *from_out = best_from;
        *die_out = best_die;
        return 1;
    }
    int pick = bg_randi(env, 0, num_legal - 1);
    *from_out = legal_moves[pick][0];
    *die_out = legal_moves[pick][1];
    return 1;
}

static int play_one_black_move(Backgammon* env) {
    int chosen_from = -1;
    int chosen_die = -1;
    int heuristic = bg_randf(env) >= env->opponent_random_prob;
    if (!pick_legal_move(env, &chosen_from, &chosen_die, heuristic)) {
        return 0;
    }
    make_move(env, chosen_from, chosen_die);
    return 1;
}

void opponent_move(Backgammon* env) {
    env->current_player = BG_BLACK;
    roll_dice(env);
    env->turns_this_episode++;

    while (play_one_black_move(env)) {
        if (check_win(env, BG_BLACK)) return;
    }

    env->current_player = BG_WHITE;
    roll_dice(env);
}

void add_log(Backgammon* env) {
    env->log.episode_return += env->episode_return;
    env->log.score += env->episode_return;
    env->log.episode_length += env->tick;
    float white_win = check_win(env, BG_WHITE) ? 1.0f : 0.0f;
    env->log.win_rate += white_win;
    env->log.perf += white_win;
    env->log.black_win_rate += check_win(env, BG_BLACK) ? 1.0f : 0.0f;
    if (env->turns_this_episode > 0) {
        env->log.avg_moves_per_turn +=
            (float)env->moves_this_episode / (float)env->turns_this_episode;
    }
    if (env->moves_this_episode > 0) {
        env->log.hit_rate += (float)env->hits_this_episode / (float)env->moves_this_episode;
    }
    int home_count = 0;
    for (int i = 1; i <= 6; i++) {
        if (env->board[i] > 0) home_count += env->board[i];
    }
    env->log.checkers_home += home_count;
    env->log.checkers_off += env->off[BG_WHITE];
    env->log.n += 1;
}

static void reset_board(Backgammon* env) {
    memset(env->board, 0, sizeof(env->board));
    env->bar[BG_WHITE] = 0;
    env->bar[BG_BLACK] = 0;
    env->off[BG_WHITE] = 0;
    env->off[BG_BLACK] = 0;
    env->tick = 0;
    env->episode_return = 0.0f;
    env->moves_this_episode = 0;
    env->hits_this_episode = 0;
    env->turns_this_episode = 0;
    env->must_enter_from_bar = false;
    env->pending_reset = 0;

    env->board[24] = 2;
    env->board[13] = 5;
    env->board[8] = 3;
    env->board[6] = 5;
    env->board[1] = -2;
    env->board[12] = -5;
    env->board[17] = -3;
    env->board[19] = -5;

    // Agent is always White. Random first player: if Black wins the opening,
    // they take a full turn so White is not stuck on an illegal-only mask.
    if (bg_randi(env, 0, 1) == 0) {
        env->current_player = BG_WHITE;
        roll_dice(env);
    } else {
        opponent_move(env);
    }
    compute_observations(env);
}

void puf_reset(Backgammon* env) {
    env->agents[0].rewards[0] = 0.0f;
    env->agents[0].terminals[0] = 0.0f;
    reset_board(env);
}

static void finish_game(Backgammon* env, float reward) {
    env->episode_return += reward;
    env->agents[0].rewards[0] = reward;
    env->agents[0].terminals[0] = 1.0f;
    add_log(env);
    if (env->have_window) {
        env->pending_reset = 1;
    } else {
        reset_board(env);
    }
}

void puf_step(Backgammon* env) {
    if (env->pending_reset) {
        reset_board(env);
    }
    env->tick += 1;
    env->agents[0].rewards[0] = 0.0f;
    env->agents[0].terminals[0] = 0.0f;

    if (env->current_player != BG_WHITE) {
        opponent_move(env);
        if (check_win(env, BG_BLACK)) {
            finish_game(env, -1.0f);
            return;
        }
    }

    int action = (int)env->agents[0].actions[0];
    if (action < 0 || action >= NUM_ACTIONS) {
        action = 0;
    }
    int from = action / 4;
    int die_index = action % 4;

    float reward = 0.0f;
    int old_off = env->off[BG_WHITE];
    int old_bar_opponent = env->bar[BG_BLACK];

    if (!is_legal_move(env, from, die_index)
            && !pick_legal_move(env, &from, &die_index, 0)) {
        reward -= 0.1f;
        opponent_move(env);
        env->episode_return += reward;
        env->agents[0].rewards[0] = reward;
        if (check_win(env, BG_BLACK)) {
            finish_game(env, -1.0f);
            return;
        }
        compute_observations(env);
        return;
    }

    make_move(env, from, die_index);
    if (env->off[BG_WHITE] > old_off) {
        reward += 0.05f;
    }
    if (env->bar[BG_BLACK] > old_bar_opponent) {
        reward += 0.02f;
    }

    if (check_win(env, BG_WHITE)) {
        finish_game(env, 1.0f);
        return;
    }
    if (env->dice_used >= env->num_dice || !has_legal_moves(env)) {
        opponent_move(env);
    }
    if (check_win(env, BG_BLACK)) {
        finish_game(env, -1.0f);
        return;
    }
    if (env->tick >= MAX_STEPS) {
        finish_game(env, reward);
        return;
    }

    env->episode_return += reward;
    env->agents[0].rewards[0] = reward;
    compute_observations(env);
}

static int point_index_at(int col, int top) {
    // col 0..5 left of bar, 6..11 right of bar. Top: 13-24, bottom: 12-1.
    if (top) {
        return col < 6 ? 13 + col : 19 + (col - 6);
    }
    return col < 6 ? 12 - col : 6 - (col - 6);
}

void puf_render(Backgammon* env) {
    if (!env->have_window) {
        env->have_window = 1;
        if (!IsWindowReady()) {
            InitWindow(WIN_W, WIN_H, "PufferLib Backgammon");
            SetTargetFPS(60);
        }
    }
    if (IsKeyDown(KEY_ESCAPE)) exit(0);
    if (IsKeyPressed(KEY_TAB)) ToggleFullscreen();

    const int board_x = 40;
    const int board_y = 50;
    const int board_w = 820;
    const int board_h = 540;
    const int bar_w = 52;
    const int point_w = (board_w - bar_w) / 12;
    const int point_h = 230;
    const float radius = point_w * 0.42f;

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    DrawRectangle(board_x, board_y, board_w, board_h, (Color){20, 60, 60, 255});
    DrawRectangle(board_x + 6 * point_w, board_y, bar_w, board_h, (Color){10, 32, 32, 255});

    for (int col = 0; col < 12; col++) {
        int x = board_x + col * point_w + (col >= 6 ? bar_w : 0);
        Color even = (Color){0, 110, 110, 255};
        Color odd = (Color){140, 30, 30, 255};
        Color top_c = (col % 2 == 0) ? even : odd;
        Color bot_c = (col % 2 == 0) ? odd : even;
        DrawTriangle(
            (Vector2){(float)x, (float)board_y},
            (Vector2){(float)(x + point_w), (float)board_y},
            (Vector2){(float)(x + point_w / 2), (float)(board_y + point_h)},
            top_c);
        DrawTriangle(
            (Vector2){(float)x, (float)(board_y + board_h)},
            (Vector2){(float)(x + point_w / 2), (float)(board_y + board_h - point_h)},
            (Vector2){(float)(x + point_w), (float)(board_y + board_h)},
            bot_c);
    }

    for (int col = 0; col < 12; col++) {
        for (int top = 1; top >= 0; top--) {
            int pt = point_index_at(col, top);
            int count = env->board[pt];
            Color color = count > 0 ? PUFF_CYAN : PUFF_RED;
            int n = count > 0 ? count : -count;
            int x = board_x + col * point_w + (col >= 6 ? bar_w : 0) + point_w / 2;
            for (int i = 0; i < n && i < 15; i++) {
                int y;
                if (top) {
                    y = board_y + 16 + (int)(i * (radius * 1.7f));
                } else {
                    y = board_y + board_h - 16 - (int)(i * (radius * 1.7f));
                }
                DrawCircle(x, y, radius, color);
                DrawCircleLines(x, y, radius, PUFF_WHITE);
            }
        }
    }

    int bar_cx = board_x + 6 * point_w + bar_w / 2;
    for (int i = 0; i < env->bar[BG_WHITE]; i++) {
        DrawCircle(bar_cx, board_y + board_h / 2 + 20 + i * 18, 10, PUFF_CYAN);
    }
    for (int i = 0; i < env->bar[BG_BLACK]; i++) {
        DrawCircle(bar_cx, board_y + board_h / 2 - 20 - i * 18, 10, PUFF_RED);
    }

    int tray_x = board_x + board_w + 20;
    DrawRectangle(tray_x, board_y, 100, board_h, (Color){14, 40, 40, 255});
    DrawText("OFF", tray_x + 28, board_y + 8, 20, PUFF_WHITE);
    for (int i = 0; i < env->off[BG_WHITE] && i < 15; i++) {
        DrawCircle(tray_x + 30, board_y + board_h - 20 - i * 16, 8, PUFF_CYAN);
    }
    for (int i = 0; i < env->off[BG_BLACK] && i < 15; i++) {
        DrawCircle(tray_x + 70, board_y + 40 + i * 16, 8, PUFF_RED);
    }

    int die_y = board_y + board_h / 2 - 18;
    int shown = 0;
    for (int i = 0; i < 4; i++) {
        if (!env->dice_available[i] || env->dice[i] <= 0) continue;
        int dx = bar_cx - 16;
        int dy = die_y + shown * 38;
        DrawRectangle(dx, dy, 32, 32, PUFF_WHITE);
        DrawText(TextFormat("%d", env->dice[i]), dx + 10, dy + 6, 20, PUFF_BACKGROUND);
        shown++;
    }

    DrawText(TextFormat("Step %d  White off %d  Black off %d",
        env->tick, env->off[BG_WHITE], env->off[BG_BLACK]),
        40, 12, 22, PUFF_WHITE);
    EndDrawing();
    puf_web_vsync();
    if (env->pending_reset) {
        reset_board(env);
    }
}

void puf_close(Backgammon* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
    env->have_window = 0;
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "win_rate", log->win_rate);
    dict_set(out, "black_win_rate", log->black_win_rate);
    dict_set(out, "avg_moves_per_turn", log->avg_moves_per_turn);
    dict_set(out, "hit_rate", log->hit_rate);
    dict_set(out, "checkers_home", log->checkers_home);
    dict_set(out, "checkers_off", log->checkers_off);
    dict_set(out, "n", log->n);
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->have_window = 0;
    env->opponent_random_prob = dict_get(kwargs, "opponent_random_prob");
    env->agents[0].action_mask = NULL;
    env->agents[0].policy = 0;
    memset(&env->log, 0, sizeof(Log));
}
