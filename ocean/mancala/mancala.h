// Mancala (Kalah variant) — single-agent, turn-based, scripted opponent.
//
// Layout:                Index:  0  1  2  3  4  5   6   7  8  9 10 11 12  13
//                                ────P0 pits──────  P0  ────P1 pits──────  P1
//                                                  store                   store
// Sowing direction: counter-clockwise = increasing index, modulo 14, skipping
// the OPPONENT's store. Capture rule: "Empty Capture" variant — if the last
// stone lands in the player's OWN previously-empty pit, that stone always
// moves to the player's store, AND any stones in the opposite pit are taken
// too. (Strictly-standard Kalah only captures when the opposite is non-empty;
// the Empty Capture variant always at least claims the just-sown stone.)
// Game ends when one side empties; the other side sweeps remaining stones
// into its store. Winner = larger store.
//
// Agent always plays as P0; P1 is an internal random-valid opponent.

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
    float perf;             // 1 if last episode was a P0 win, else 0 (averaged → win rate)
    float score;            // signed terminal reward (+1/0/-1)
    float margin;           // (P0_store - P1_store) / TOTAL_STONES at end
    float captures;         // # captures by P0 in episode
    float extra_turns;      // # extra turns earned by P0 in episode
    float invalid_moves;    // # illegal-pit attempts by P0 in episode
    float episode_return;
    float episode_length;
    float n;                // count — MUST be the last field (pufferlib aggregation convention)
} Log;

typedef struct Client Client;
typedef struct CMancala CMancala;
static void close_client(Client* client);  // forward — defined with the renderer
struct CMancala {
    // Pufferlib required fields (order matches connect4.h / breakout.h)
    float* observations;
    float* actions;          // single discrete action: pit index 0..5
    float* rewards;
    float* terminals;
    int num_agents;
    Log log;
    Client* client;

    // Per-env RNG (seeded by my_vec_init in vecenv.h; thread-safe via rand_r)
    unsigned int rng;

    // Game state
    int board[BOARD_SIZE];
    int tick;

    // Whose move c_step will play next: 0 (P0/agent) or 1 (P1/opponent).
    // Reset to 0 by c_reset; flipped inside c_step when no extra turn is earned.
    // Under external_opponent=0 (training default), c_step auto-cycles P1's
    // random move(s) before returning, so this is always 0 at function entry/exit.
    int current_player;

    // 0 = scripted random P1 is auto-played inside c_step (training behavior).
    // 1 = caller supplies P1's action via env->actions[0]; c_step returns after
    //     a single player's move so the caller can interleave another agent
    //     (human, second policy, etc.). Set by the caller AFTER init/c_reset;
    //     not touched by c_reset.
    int external_opponent;

    // Snapshot of env->board immediately before sweep_remaining is applied at
    // the end of an episode. Lets the human-play loop animate the sweep with
    // real per-pit stone counts. Populated by c_step on terminal steps; junk
    // value otherwise. Zero-init by c_reset for cleanliness.
    int pre_sweep_board[BOARD_SIZE];

    // Per-episode counters (folded into Log on episode end). All track P0
    // (the agent) only; P1's actions do not update these regardless of mode.
    int ep_captures;
    int ep_extra_turns;
    int ep_invalid;

    // Trace of opponent (P1) moves taken during the most recent c_step. Used
    // by the standalone playback to show the chain of P1 moves between two
    // consecutive P0 actions (P1 can take multiple turns when its sown stone
    // lands in its own store). Sized at 64 — well above the longest chain
    // I've been able to construct (17 from a contrived [6,5,4,3,2,1] board)
    // and effectively unbounded for a uniform-random opponent, which rarely
    // chains past 5.
    int p1_last_pits[64];
    int p1_last_was_extra[64];
    int p1_last_was_capture[64];
    int p1_last_count;
};

// ---------------------------------------------------------------------------
// Allocation / lifecycle
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------

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
    // episode_length is incremented per step in c_step.
}

// ---------------------------------------------------------------------------
// Game-logic helpers
// ---------------------------------------------------------------------------

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
    // Whichever side still has stones sweeps them into its own store.
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

// Sow stones from player's local pit (0..NUM_PITS-1).
// Returns the absolute board index where the LAST stone landed (0..13).
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

// Capture rule — "Empty Capture" variant. If the last stone lands in the
// player's own pit that was empty before this move (so it now holds exactly
// 1), move that stone to the player's store. If the opposite pit is non-
// empty, take those stones too. Returns the number of stones moved into the
// store (>=1 on capture, 0 if not a capture situation). 12 - last_idx is the
// mirror across the board: P0 pit i ↔ P1 pit (5-i) by index, since pits sit
// symmetrically on either side of the two stores.
static int apply_capture(CMancala* env, int player, int last_idx) {
    if (!is_own_pit(player, last_idx)) return 0;
    if (env->board[last_idx] != 1) return 0;          // landed in empty pit (now 1)
    int opposite = 12 - last_idx;                     // mirror across the board
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

// Random valid pit for player. Caller must ensure player has at least one
// non-empty pit (i.e. game not over). Returns local pit index in [0, NUM_PITS).
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
    // Canonical pufferlib pattern is to reset in the same c_step. We honour
    // that for training (external_opponent=0). Under external_opponent=1, the
    // caller (e.g. the human-play loop) wants to inspect the post-sweep board
    // and the pre_sweep_board snapshot before the next episode starts, so the
    // caller owns the explicit c_reset.
    if (env->external_opponent == 0) c_reset(env);
}

static void c_reset(CMancala* env) {
    // c_reset only resets game state. The framework handles terminals/rewards;
    // c_step zeroes them at its top. This lets finish_game set terminal=DONE
    // and reward=±1, then call c_reset, without those getting clobbered.
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
    // external_opponent is caller-managed and intentionally NOT touched here.
    // NOTE: don't reset env->log here. The framework (static_vec_log) wipes it
    // after aggregation. Resetting here would erase add_log's contribution and
    // make episodes invisible to the aggregator.
    env->tick = 0;
    compute_observation(env);
}

// c_step processes ONE player's move per call. The active player is
// env->current_player. Reward semantics are unchanged: env->rewards[0] is
// always P0's terminal reward (+1 win, -1 loss, 0 draw); that's what the
// learning agent sees regardless of who holds env->current_player.
//
// Under env->external_opponent == 0 (the training default), if the active
// player's move flips control to P1 without ending the game, c_step inlines
// P1's scripted-random move(s) before returning — so to a training caller,
// each c_step still "takes one P0 action and yields the next P0 observation",
// byte-for-byte equivalent to the pre-refactor implementation. RNG sequence,
// log counters, and trace buffers are populated identically.
//
// Under env->external_opponent == 1, c_step returns after the active player's
// chain ends. The caller flips between P0 and P1 actions explicitly. The
// caller is also responsible for c_reset after a terminal step (finish_game
// skips it in this mode, so the caller can read env->board / pre_sweep_board
// to render the final state).
static void c_step(CMancala* env) {
    env->terminals[0] = NOT_DONE;
    env->rewards[0] = 0.0f;
    env->log.episode_length += 1.0f;
    env->tick += 1;

    int p = env->current_player;

    // The opponent trace is meaningful only for P1 chains played inline by
    // c_step (external_opponent=0). Reset it whenever P0 starts a fresh move
    // so a single P0 c_step boundary still corresponds to one trace slice.
    if (p == 0) env->p1_last_count = 0;

    // Validate active player's action.
    int act = (int)env->actions[0];
    if (act < 0 || act >= NUM_PITS || env->board[pit_start_of(p) + act] == 0) {
        // Per-episode invalid counter tracks P0 only — that's the agent the
        // log is about. P1 illegals are still a forfeit but not logged.
        if (p == 0) env->ep_invalid += 1;
        // Reward is P0's outcome: P0 forfeits → P0 LOSS; P1 forfeits → P0 WIN.
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
        // Same player goes again on the next c_step.
        compute_observation(env);
        return;
    }

    env->current_player = 1 - p;

    // Training-mode shortcut: auto-run P1's scripted random chain inline so
    // each c_step still consumes exactly one P0 action. The body below is a
    // verbatim port of the pre-refactor P1 loop; do not perturb it without
    // re-running the parity check.
    if (env->external_opponent == 0 && env->current_player == 1) {
        while (1) {
            int omove = scripted_opponent_move(env);
            int olast = sow(env, 1, omove);
            int ocaptured = apply_capture(env, 1, olast);
            int oextra = is_extra_turn(1, olast);

            if (env->p1_last_count < (int)(sizeof(env->p1_last_pits)/sizeof(env->p1_last_pits[0]))) {
                int slot = env->p1_last_count;
                env->p1_last_pits[slot]        = omove;
                env->p1_last_was_extra[slot]   = oextra;
                env->p1_last_was_capture[slot] = (ocaptured > 0);
                env->p1_last_count++;
            }

            if (game_over(env->board)) {
                memcpy(env->pre_sweep_board, env->board, sizeof(env->pre_sweep_board));
                sweep_remaining(env->board);
                finish_game(env, terminal_reward(env));
                return;
            }
            if (!oextra) break;
        }
        env->current_player = 0;  // P1 chain ended; back to P0
    }

    compute_observation(env);
}

// ---------------------------------------------------------------------------
// Render (raylib) — puffer aesthetic: dark teal background, JetBrainsMono
// text, rounded-rect pits + stores. P0 (agent) cyan along the bottom, P1
// red along the top with pit indices mirrored so opposites line up
// vertically (a captured pair sits in the same column).
// ---------------------------------------------------------------------------

static const Color PUFF_RED        = (Color){187,   0,   0, 255};
static const Color PUFF_CYAN       = (Color){  0, 187, 187, 255};
static const Color PUFF_WHITE      = (Color){241, 241, 241, 255};
static const Color PUFF_BACKGROUND = (Color){  6,  24,  24, 255};
static const Color PIT_FILL        = (Color){  0,  38,  38, 255};

#define MANCALA_WIDTH   1024
#define MANCALA_HEIGHT   440
#define MARGIN_X          32
#define MARGIN_Y          64
#define PIT_RADIUS        38   // half-side of each pit's rounded square
#define STORE_W           86

struct Client {
    int width;
    int height;
    Font font_big;     // JetBrainsMono SemiBold @ 36 — counts and store labels
    Font font_small;   // JetBrainsMono Regular @ 14 — header, pit indices
    bool fonts_loaded;
};

static Client* make_client(void) {
    Client* c = (Client*)calloc(1, sizeof(Client));
    c->width = MANCALA_WIDTH;
    c->height = MANCALA_HEIGHT;
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(c->width, c->height, "PufferLib Mancala");
    SetTargetFPS(30);
    // LoadFontEx falls back to the default font if the file is missing, so it's
    // safe to call unconditionally — but the puffer-style aesthetic depends on
    // JetBrainsMono being present in resources/shared/.
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

// Used by the framework's vec_render hook; the standalone --fast build
// renders via human_render in mancala.c instead.
__attribute__((unused))
static void c_render(CMancala* env) {
    if (env->client == NULL) env->client = make_client();
    if (WindowShouldClose() || IsKeyDown(KEY_ESCAPE)) {
        close_client(env->client);
        env->client = NULL;
        return;
    }
    Client* cli = env->client;

    int store_h = cli->height - 2 * MARGIN_Y;
    int pit_area_w = cli->width - 2 * MARGIN_X - 2 * STORE_W - 24;
    int pit_pitch = pit_area_w / NUM_PITS;
    int pits_left = MARGIN_X + STORE_W + 12 + pit_pitch / 2;
    int row_y_top = MARGIN_Y + 56;
    int row_y_bot = cli->height - MARGIN_Y - 56;

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    // Header — monospace status line. Tick first, then P1 / P0 score with the
    // visual layout (P1 sits on the left of the board so it's listed first).
    char header[160];
    snprintf(header, sizeof(header),
             "MANCALA   tick %-4d   P1 %2d  vs  %2d P0",
             env->tick, env->board[P1_STORE], env->board[P0_STORE]);
    DrawTextEx(cli->font_small, header, (Vector2){MARGIN_X, 22},
               14, 0, Fade(PUFF_WHITE, 0.7f));

    // Subtle outer frame around the play area.
    Rectangle frame = {MARGIN_X - 6, MARGIN_Y - 6,
                       cli->width - 2 * (MARGIN_X - 6),
                       cli->height - 2 * MARGIN_Y + 12};
    DrawRectangleRoundedLines(frame, 0.04f, 8, Fade(PUFF_WHITE, 0.08f));

    // Stores: P1 (red) on the left, P0 (cyan) on the right.
    draw_store(cli->font_big, cli->font_small,
               MARGIN_X, MARGIN_Y, STORE_W, store_h,
               env->board[P1_STORE], "P1", PUFF_RED);
    draw_store(cli->font_big, cli->font_small,
               cli->width - MARGIN_X - STORE_W, MARGIN_Y, STORE_W, store_h,
               env->board[P0_STORE], "P0", PUFF_CYAN);

    // P0 pits — bottom row, indices 0..5 left-to-right (agent's view).
    for (int i = 0; i < NUM_PITS; i++) {
        int cx = pits_left + i * pit_pitch;
        draw_pit(cli->font_big, cx, row_y_bot, PIT_RADIUS,
                 env->board[P0_PITS_START + i], PUFF_CYAN);
        char lbl[4]; snprintf(lbl, sizeof(lbl), "%d", i);
        draw_text_centered(cli->font_small, lbl,
                           cx, row_y_bot + PIT_RADIUS + 14,
                           12, Fade(PUFF_CYAN, 0.55f));
    }
    // P1 pits — top row, drawn right-to-left so capture-opposites line up.
    for (int i = 0; i < NUM_PITS; i++) {
        int cx = pits_left + (NUM_PITS - 1 - i) * pit_pitch;
        draw_pit(cli->font_big, cx, row_y_top, PIT_RADIUS,
                 env->board[P1_PITS_START + i], PUFF_RED);
        char lbl[4]; snprintf(lbl, sizeof(lbl), "%d", i);
        draw_text_centered(cli->font_small, lbl,
                           cx, row_y_top - PIT_RADIUS - 14,
                           12, Fade(PUFF_RED, 0.55f));
    }

    EndDrawing();
}
