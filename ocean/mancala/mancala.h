// Mancala (Kalah, Empty Capture variant). Single-agent: P0 is the policy,
// P1 is a scripted random-valid opponent (or an external driver when
// external_opponent=1). See _design.md for rule and refactor notes.

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "raylib.h"

#define NUM_PITS 6
#define INIT_STONES 4
#define BOARD_SIZE 14
#define P0_PITS_START 0
#define P0_STORE 6
#define P1_PITS_START 7
#define P1_STORE 13
#define TOTAL_STONES (NUM_PITS * 2 * INIT_STONES)  // 48
#define OBS_DIM 15  // 14 board cells + current_player flag

static const float PLAYER_WIN  =  1.0f;
static const float PLAYER_LOSS = -1.0f;
static const float DRAW        =  0.0f;
static const unsigned char DONE = 1;
static const unsigned char NOT_DONE = 0;

typedef struct Log {
    float perf;             // P0 win rate (1 on win, averaged across episodes)
    float score;
    float margin;           // (P0_store - P1_store) / TOTAL_STONES
    float captures;
    float extra_turns;
    float invalid_moves;
    float episode_return;
    float episode_length;
    float n;                // count — MUST be last (pufferlib aggregation)
} Log;

typedef struct Client Client;
typedef struct CMancala CMancala;
static void close_client(Client* client);  // forward — defined with the renderer
struct CMancala {
    // pufferlib required fields
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    Log log;
    Client* client;

    unsigned int rng;        // per-env RNG, seeded by my_vec_init

    int board[BOARD_SIZE];
    int tick;
    int current_player;      // 0 = P0 next, 1 = P1 next; reset to 0 by c_reset
    int external_opponent;   // 0 = c_step auto-plays random P1 (training default)
    int pre_sweep_board[BOARD_SIZE];   // snapshot before terminal sweep

    int ep_captures;
    int ep_extra_turns;
    int ep_invalid;
};

static void allocate_cmancala(CMancala* env) {
    env->observations = (float*)calloc(OBS_DIM, sizeof(float));
    env->actions      = (float*)calloc(1, sizeof(float));
    env->rewards      = (float*)calloc(1, sizeof(float));
    env->terminals    = (float*)calloc(1, sizeof(float));
}

static void free_allocated_cmancala(CMancala* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
}

static void init(CMancala* env) {
    env->log = (Log){0};
    env->tick = 0;
}

static void c_close(CMancala* env) {
    if (env->client) {
        close_client(env->client);
        env->client = NULL;
    }
}

static void add_log(CMancala* env) {
    float r = env->rewards[0];
    env->log.perf           += (r == PLAYER_WIN) ? 1.0f : 0.0f;
    env->log.score          += r;
    env->log.margin         += (env->board[P0_STORE] - env->board[P1_STORE])
                               / (float)TOTAL_STONES;
    env->log.captures       += (float)env->ep_captures;
    env->log.extra_turns    += (float)env->ep_extra_turns;
    env->log.invalid_moves  += (float)env->ep_invalid;
    env->log.episode_return += r;
    env->log.n              += 1.0f;
}

static inline int store_of(int player)     { return (player == 0) ? P0_STORE : P1_STORE; }
static inline int opp_store_of(int player) { return (player == 0) ? P1_STORE : P0_STORE; }
static inline int pit_start_of(int player) { return (player == 0) ? P0_PITS_START : P1_PITS_START; }

static inline int is_own_pit(int player, int idx) {
    int start = pit_start_of(player);
    return idx >= start && idx < start + NUM_PITS;
}

static int side_empty(const int* board, int player) {
    int start = pit_start_of(player);
    for (int i = 0; i < NUM_PITS; i++) {
        if (board[start + i] != 0) return 0;
    }
    return 1;
}

static int game_over(const int* board) {
    return side_empty(board, 0) || side_empty(board, 1);
}

static void sweep_remaining(int* board) {
    for (int p = 0; p < 2; p++) {
        int start = pit_start_of(p);
        int total = 0;
        for (int i = 0; i < NUM_PITS; i++) {
            total += board[start + i];
            board[start + i] = 0;
        }
        board[store_of(p)] += total;
    }
}

// Sow from player's local pit; returns absolute index where the last stone lands.
static int sow(CMancala* env, int player, int pit_local) {
    int idx = pit_start_of(player) + pit_local;
    int stones = env->board[idx];
    env->board[idx] = 0;
    int skip = opp_store_of(player);
    int last = idx;
    while (stones > 0) {
        idx = (idx + 1) % BOARD_SIZE;
        if (idx == skip) continue;
        env->board[idx] += 1;
        stones -= 1;
        last = idx;
    }
    return last;
}

// Empty Capture variant: last stone in own previously-empty pit always moves
// to the player's store; opposite pit's stones go too if any.
static int apply_capture(CMancala* env, int player, int last_idx) {
    if (!is_own_pit(player, last_idx)) return 0;
    if (env->board[last_idx] != 1) return 0;
    int opposite = 12 - last_idx;
    int taken = 0;
    if (opposite >= 0 && opposite < BOARD_SIZE
        && opposite != P0_STORE && opposite != P1_STORE) {
        taken = env->board[opposite];
        env->board[opposite] = 0;
    }
    env->board[store_of(player)] += taken + 1;
    env->board[last_idx] = 0;
    return taken + 1;
}

static inline int is_extra_turn(int player, int last_idx) {
    return last_idx == store_of(player);
}

// Random valid pit for the scripted P1 opponent.
static int scripted_opponent_move(CMancala* env) {
    int valid[NUM_PITS];
    int n = 0;
    int start = pit_start_of(1);
    for (int i = 0; i < NUM_PITS; i++) {
        if (env->board[start + i] > 0) valid[n++] = i;
    }
    return valid[rand_r(&env->rng) % n];
}

// ---------------------------------------------------------------------------
// Observation / reward / step
// ---------------------------------------------------------------------------

static void compute_observation(CMancala* env) {
    // From P0's perspective: own pits, own store, opp pits, opp store, then a
    // current-player flag (always 0 from the agent's POV with this design;
    // kept so OBS_DIM stays stable if a self-play variant is added later).
    float inv = 1.0f / (float)TOTAL_STONES;
    int o = 0;
    for (int i = 0; i < NUM_PITS; i++) env->observations[o++] = env->board[P0_PITS_START + i] * inv;
    env->observations[o++] = env->board[P0_STORE] * inv;
    for (int i = 0; i < NUM_PITS; i++) env->observations[o++] = env->board[P1_PITS_START + i] * inv;
    env->observations[o++] = env->board[P1_STORE] * inv;
    env->observations[o++] = 0.0f;
}

static void c_reset(CMancala* env);  // forward decl for finish_game

static float terminal_reward(const CMancala* env) {
    int diff = env->board[P0_STORE] - env->board[P1_STORE];
    if (diff > 0) return PLAYER_WIN;
    if (diff < 0) return PLAYER_LOSS;
    return DRAW;
}

static void finish_game(CMancala* env, float r) {
    env->rewards[0] = r;
    env->terminals[0] = DONE;
    add_log(env);
    // Skip auto-reset under external_opponent=1 so the caller can inspect
    // the terminal board (incl. pre_sweep_board) before resetting itself.
    if (env->external_opponent == 0) c_reset(env);
}

static void c_reset(CMancala* env) {
    for (int i = 0; i < BOARD_SIZE; i++) env->board[i] = 0;
    for (int p = 0; p < 2; p++) {
        int start = pit_start_of(p);
        for (int i = 0; i < NUM_PITS; i++) env->board[start + i] = INIT_STONES;
    }
    env->ep_captures = 0;
    env->ep_extra_turns = 0;
    env->ep_invalid = 0;
    env->current_player = 0;
    for (int i = 0; i < BOARD_SIZE; i++) env->pre_sweep_board[i] = 0;
    // external_opponent is caller-managed; don't touch.
    // env->log is wiped by the framework after aggregation; don't reset here.
    env->tick = 0;
    compute_observation(env);
}

// One player's move per call. env->rewards[0] is always P0's terminal reward.
// Under external_opponent=0 (training default), an active P0 move that flips
// to P1 inlines P1's scripted-random chain before returning — equivalent to
// the pre-refactor "one c_step = one P0 action" contract.
static void c_step(CMancala* env) {
    env->terminals[0] = NOT_DONE;
    env->rewards[0] = 0.0f;
    env->log.episode_length += 1.0f;
    env->tick += 1;

    int p = env->current_player;
    int act = (int)env->actions[0];
    if (act < 0 || act >= NUM_PITS || env->board[pit_start_of(p) + act] == 0) {
        if (p == 0) env->ep_invalid += 1;
        finish_game(env, (p == 0) ? PLAYER_LOSS : PLAYER_WIN);
        return;
    }

    int last = sow(env, p, act);
    int captured = apply_capture(env, p, last);
    int extra = is_extra_turn(p, last);
    if (p == 0) {
        env->ep_captures += (captured > 0) ? 1 : 0;
        env->ep_extra_turns += extra;
    }

    if (game_over(env->board)) {
        memcpy(env->pre_sweep_board, env->board, sizeof(env->pre_sweep_board));
        sweep_remaining(env->board);
        finish_game(env, terminal_reward(env));
        return;
    }

    if (extra) {
        compute_observation(env);
        return;
    }

    env->current_player = 1 - p;

    // Training-mode auto-P1: verbatim port of the pre-refactor loop. Don't
    // perturb without re-running parity (see _design.md).
    if (env->external_opponent == 0 && env->current_player == 1) {
        while (1) {
            int omove = scripted_opponent_move(env);
            int olast = sow(env, 1, omove);
            apply_capture(env, 1, olast);
            int oextra = is_extra_turn(1, olast);
            if (game_over(env->board)) {
                memcpy(env->pre_sweep_board, env->board, sizeof(env->pre_sweep_board));
                sweep_remaining(env->board);
                finish_game(env, terminal_reward(env));
                return;
            }
            if (!oextra) break;
        }
        env->current_player = 0;
    }

    compute_observation(env);
}

// Render (raylib): puffer aesthetic, P0 cyan on top, P1 red on bottom with
// pit indices mirrored so opposite pits align vertically.

static const Color PUFF_RED        = (Color){187,   0,   0, 255};
static const Color PUFF_CYAN       = (Color){  0, 187, 187, 255};
static const Color PUFF_WHITE      = (Color){241, 241, 241, 255};
static const Color PUFF_BACKGROUND = (Color){  6,  24,  24, 255};
static const Color PIT_FILL        = (Color){  0,  38,  38, 255};

#define MANCALA_WIDTH   1024
#define MANCALA_HEIGHT   440
#define MARGIN_X          32
#define MARGIN_Y          64
#define PIT_RADIUS        38
#define STORE_W           86

struct Client {
    int width;
    int height;
    Font font_big;
    Font font_small;
    bool fonts_loaded;
};

static Client* make_client(void) {
    Client* c = (Client*)calloc(1, sizeof(Client));
    c->width = MANCALA_WIDTH;
    c->height = MANCALA_HEIGHT;
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(c->width, c->height, "PufferLib Mancala");
    SetTargetFPS(30);
    c->font_big   = LoadFontEx("resources/shared/JetBrainsMono-SemiBold.ttf", 36, NULL, 95);
    c->font_small = LoadFontEx("resources/shared/JetBrainsMono-Regular.ttf",  14, NULL, 95);
    c->fonts_loaded = true;
    return c;
}

static void close_client(Client* client) {
    if (client->fonts_loaded) {
        UnloadFont(client->font_big);
        UnloadFont(client->font_small);
    }
    CloseWindow();
    free(client);
}

static void draw_text_centered(Font f, const char* s, int cx, int cy, int sz, Color c) {
    Vector2 m = MeasureTextEx(f, s, (float)sz, 0);
    DrawTextEx(f, s, (Vector2){cx - m.x / 2.0f, cy - m.y / 2.0f}, (float)sz, 0, c);
}

static void draw_pit(Font f, int cx, int cy, int half, int stones, Color rim) {
    Rectangle r = {cx - half, cy - half, 2 * half, 2 * half};
    DrawRectangleRounded(r, 0.40f, 8, PIT_FILL);
    Color edge = (stones == 0) ? Fade(rim, 0.25f) : rim;
    DrawRectangleRoundedLines(r, 0.40f, 8, edge);
    char buf[8];
    snprintf(buf, sizeof(buf), "%d", stones);
    Color txt = (stones == 0) ? Fade(PUFF_WHITE, 0.30f) : PUFF_WHITE;
    draw_text_centered(f, buf, cx, cy, 28, txt);
}

static void draw_store(Font f_big, Font f_small, int x, int y, int w, int h,
                       int stones, const char* label, Color rim) {
    Rectangle r = {x, y, w, h};
    DrawRectangleRounded(r, 0.30f, 12, PIT_FILL);
    DrawRectangleRoundedLines(r, 0.30f, 12, rim);
    draw_text_centered(f_small, label, x + w / 2, y + 18, 14, rim);
    char buf[8];
    snprintf(buf, sizeof(buf), "%d", stones);
    draw_text_centered(f_big, buf, x + w / 2, y + h / 2, 44, PUFF_WHITE);
}

typedef struct { int store_h, pit_pitch, pits_left, row_y_top, row_y_bot; } Layout;
static Layout layout_for(const Client* cli) {
    Layout L;
    L.store_h    = cli->height - 2 * MARGIN_Y;
    int area_w   = cli->width - 2 * MARGIN_X - 2 * STORE_W - 24;
    L.pit_pitch  = area_w / NUM_PITS;
    L.pits_left  = MARGIN_X + STORE_W + 12 + L.pit_pitch / 2;
    L.row_y_top  = MARGIN_Y + 56;
    L.row_y_bot  = cli->height - MARGIN_Y - 56;
    return L;
}

// World coords of a pit/store: P0 top right→left, P1 bottom left→right.
static void pit_center(const Client* cli, int board_idx, int* out_x, int* out_y) {
    Layout L = layout_for(cli);
    if (board_idx == P0_STORE) {
        *out_x = MARGIN_X + STORE_W / 2;
        *out_y = MARGIN_Y + L.store_h / 2;
    } else if (board_idx == P1_STORE) {
        *out_x = cli->width - MARGIN_X - STORE_W / 2;
        *out_y = MARGIN_Y + L.store_h / 2;
    } else if (board_idx >= P0_PITS_START && board_idx < P0_PITS_START + NUM_PITS) {
        int local = board_idx - P0_PITS_START;
        *out_x = L.pits_left + (NUM_PITS - 1 - local) * L.pit_pitch;
        *out_y = L.row_y_top;
    } else {
        int local = board_idx - P1_PITS_START;
        *out_x = L.pits_left + local * L.pit_pitch;
        *out_y = L.row_y_bot;
    }
}

// Draw the static board (header, frame, stores, pits, labels) reading counts
// from `board`. Caller manages BeginDrawing/EndDrawing and any overlays.
// active_alpha modulates the active player's store rim (1.0 = flat).
static void c_draw(CMancala* env, const int* board, float active_alpha) {
    Client* cli = env->client;
    Layout L = layout_for(cli);

    char header[160];
    snprintf(header, sizeof(header),
             "MANCALA   tick %-4d   P0 %2d  vs  %2d P1",
             env->tick, board[P0_STORE], board[P1_STORE]);
    DrawTextEx(cli->font_small, header, (Vector2){MARGIN_X, 22},
               14, 0, Fade(PUFF_WHITE, 0.7f));

    Rectangle frame = {MARGIN_X - 6, MARGIN_Y - 6,
                       cli->width - 2 * (MARGIN_X - 6),
                       cli->height - 2 * MARGIN_Y + 12};
    DrawRectangleRoundedLines(frame, 0.04f, 8, Fade(PUFF_WHITE, 0.08f));

    Color p0_rim = Fade(PUFF_CYAN, env->current_player == 0 ? active_alpha : 0.45f);
    Color p1_rim = Fade(PUFF_RED,  env->current_player == 1 ? active_alpha : 0.45f);
    draw_store(cli->font_big, cli->font_small,
               MARGIN_X, MARGIN_Y, STORE_W, L.store_h,
               board[P0_STORE], "P0", p0_rim);
    draw_store(cli->font_big, cli->font_small,
               cli->width - MARGIN_X - STORE_W, MARGIN_Y, STORE_W, L.store_h,
               board[P1_STORE], "P1", p1_rim);

    for (int i = 0; i < NUM_PITS; i++) {
        int cx = L.pits_left + (NUM_PITS - 1 - i) * L.pit_pitch;
        draw_pit(cli->font_big, cx, L.row_y_top, PIT_RADIUS,
                 board[P0_PITS_START + i], PUFF_CYAN);
        char lbl[4]; snprintf(lbl, sizeof(lbl), "%d", i);
        draw_text_centered(cli->font_small, lbl,
                           cx, L.row_y_top - PIT_RADIUS - 14, 12,
                           Fade(PUFF_CYAN, 0.55f));
    }
    for (int i = 0; i < NUM_PITS; i++) {
        int cx = L.pits_left + i * L.pit_pitch;
        draw_pit(cli->font_big, cx, L.row_y_bot, PIT_RADIUS,
                 board[P1_PITS_START + i], PUFF_RED);
        char lbl[4]; snprintf(lbl, sizeof(lbl), "%d", i + 1);
        draw_text_centered(cli->font_small, lbl,
                           cx, L.row_y_bot + PIT_RADIUS + 14, 12,
                           Fade(PUFF_RED, 0.55f));
    }
}

// Framework's vec_render hook. The standalone calls c_draw directly so it
// can use a different display board (animation) and stack overlays.
__attribute__((unused))
static void c_render(CMancala* env) {
    if (env->client == NULL) env->client = make_client();
    if (WindowShouldClose() || IsKeyDown(KEY_ESCAPE)) {
        close_client(env->client);
        env->client = NULL;
        return;
    }
    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    c_draw(env, env->board, 1.0f);
    EndDrawing();
}
