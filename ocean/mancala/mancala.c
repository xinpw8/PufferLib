#include "mancala.h"
#include "puffernet.h"
#include <time.h>

// Pick a random valid action for the agent — used by the perf test.
static int random_p0_action(CMancala* env) {
    int valid[NUM_PITS];
    int n = 0;
    for (int i = 0; i < NUM_PITS; i++) {
        if (env->board[P0_PITS_START + i] > 0) valid[n++] = i;
    }
    // If no valid action exists (shouldn't happen mid-episode because the
    // env would have terminated), fall back to 0 — terminal step will reset.
    return n > 0 ? valid[rand() % n] : 0;
}

static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void performance_test(void) {
    const double test_time_s = 5.0;
    CMancala env = {0};
    allocate_cmancala(&env);
    init(&env);
    c_reset(&env);

    double start = monotonic_seconds();
    long steps = 0;
    while (monotonic_seconds() - start < test_time_s) {
        env.actions[0] = (float)random_p0_action(&env);
        c_step(&env);
        steps++;
    }
    double elapsed = monotonic_seconds() - start;
    printf("Mancala SPS: %.2fM over %.3f s (%ld steps)\n",
           (double)steps / elapsed / 1e6, elapsed, steps);

    printf("Final board:");
    for (int i = 0; i < BOARD_SIZE; i++) printf(" %d", env.board[i]);
    printf("\nEpisodes (log.n): %.0f, win rate (log.perf/n): %.3f\n",
           env.log.n, env.log.n > 0 ? env.log.perf / env.log.n : 0.0f);

    free_allocated_cmancala(&env);
}

// --- Rule-sanity checks (deterministic) ---------------------------------

static int assert_eq(const char* tag, int got, int want) {
    if (got != want) {
        printf("FAIL %s: got=%d want=%d\n", tag, got, want);
        return 0;
    }
    printf("OK   %s\n", tag);
    return 1;
}

static int rules_check(void) {
    int ok = 1;
    CMancala env = {0};
    allocate_cmancala(&env);
    init(&env);
    c_reset(&env);

    // 1) Initial layout: 4 stones in each pit, 0 in each store.
    for (int i = 0; i < NUM_PITS; i++) {
        ok &= assert_eq("init P0 pit", env.board[P0_PITS_START + i], INIT_STONES);
        ok &= assert_eq("init P1 pit", env.board[P1_PITS_START + i], INIT_STONES);
    }
    ok &= assert_eq("init P0 store", env.board[P0_STORE], 0);
    ok &= assert_eq("init P1 store", env.board[P1_STORE], 0);

    // 2) Extra-turn rule: P0 plays pit 2 (4 stones) -> sows into 3,4,5,6.
    //    Last lands in P0_STORE (index 6) => extra turn, opponent does NOT move.
    int saved_p1[NUM_PITS];
    for (int i = 0; i < NUM_PITS; i++) saved_p1[i] = env.board[P1_PITS_START + i];
    env.actions[0] = 2.0f;
    c_step(&env);
    ok &= assert_eq("extra-turn P0 store", env.board[P0_STORE], 1);
    ok &= assert_eq("extra-turn P0 pit 2 emptied", env.board[2], 0);
    ok &= assert_eq("extra-turn terminal not set", env.terminals[0], NOT_DONE);
    int p1_changed = 0;
    for (int i = 0; i < NUM_PITS; i++) {
        if (env.board[P1_PITS_START + i] != saved_p1[i]) p1_changed = 1;
    }
    ok &= assert_eq("extra-turn opponent did NOT move", p1_changed, 0);

    // 3a/3b drive c_step under default external_opponent=0, so P1's auto-play
    // runs immediately after P0's capture. Under Empty Capture, P1's single
    // sown stone can also self-capture, which can drain P1's side and trigger
    // game-end + auto-reset, hiding the post-step board. We use external mode
    // to inspect the board strictly after P0's move.

    // 3a) Capture with non-empty opposite: P0 plays pit 0 with 1 stone,
    //     last stone lands in empty pit 1, opposite (12-1=11) has 5 stones.
    //     Empty Capture variant matches strict-standard here: 1 + 5 = 6.
    c_reset(&env);
    env.external_opponent = 1;
    for (int i = 0; i < BOARD_SIZE; i++) env.board[i] = 0;
    env.board[P0_PITS_START + 0] = 1;
    env.board[P0_PITS_START + 1] = 0;
    env.board[12 - 1]            = 5;
    env.board[P0_PITS_START + 5] = 1;        // keep P0 side alive
    env.board[P1_PITS_START + 0] = 1;        // keep P1 side alive
    env.terminals[0] = NOT_DONE;
    env.actions[0] = 0.0f;
    c_step(&env);
    ok &= assert_eq("capture P0 store == 6", env.board[P0_STORE], 6);
    ok &= assert_eq("capture landing pit emptied", env.board[P0_PITS_START + 1], 0);
    ok &= assert_eq("capture opposite emptied", env.board[12 - 1], 0);
    env.external_opponent = 0;               // restore default for subsequent tests

    // 3b) Empty Capture: P0 plays pit 0 with 1 stone, last lands in empty
    //     pit 1, opposite is also empty. Strict-standard would leave the
    //     stone stranded; this variant moves it to P0's store.
    c_reset(&env);
    env.external_opponent = 1;
    for (int i = 0; i < BOARD_SIZE; i++) env.board[i] = 0;
    env.board[P0_PITS_START + 0] = 1;
    env.board[P0_PITS_START + 1] = 0;
    env.board[12 - 1]            = 0;        // opposite empty
    env.board[P0_PITS_START + 5] = 1;
    env.board[P1_PITS_START + 0] = 1;
    env.terminals[0] = NOT_DONE;
    env.actions[0] = 0.0f;
    c_step(&env);
    ok &= assert_eq("empty-cap P0 store == 1",  env.board[P0_STORE], 1);
    ok &= assert_eq("empty-cap landing empty",  env.board[P0_PITS_START + 1], 0);
    ok &= assert_eq("empty-cap opposite stays empty", env.board[12 - 1], 0);
    env.external_opponent = 0;

    // 4) Game-end + sweep. Test sweep_remaining directly (since c_step's
    //    finish_game now auto-resets the board before we can inspect it).
    int sboard[BOARD_SIZE] = {0};
    sboard[P0_STORE] = 1;          // simulating: P0's sown stone landed in own store
    sboard[P1_PITS_START + 0] = 3;
    sboard[P1_PITS_START + 4] = 2;
    sweep_remaining(sboard);
    ok &= assert_eq("sweep P0 store", sboard[P0_STORE], 1);
    ok &= assert_eq("sweep P1 store (3+2)", sboard[P1_STORE], 5);
    for (int i = 0; i < NUM_PITS; i++) {
        ok &= assert_eq("sweep P0 pit empty", sboard[P0_PITS_START + i], 0);
        ok &= assert_eq("sweep P1 pit empty", sboard[P1_PITS_START + i], 0);
    }
    // Drive the same scenario through c_step and verify terminal+reward.
    c_reset(&env);
    for (int i = 0; i < BOARD_SIZE; i++) env.board[i] = 0;
    env.board[P0_PITS_START + 5] = 1;
    env.board[P1_PITS_START + 0] = 3;
    env.board[P1_PITS_START + 4] = 2;
    env.terminals[0] = NOT_DONE;
    env.actions[0] = 5.0f;
    c_step(&env);
    ok &= assert_eq("end terminal set", env.terminals[0], DONE);
    ok &= assert_eq("end reward = LOSS", (int)env.rewards[0], (int)PLAYER_LOSS);

    // 5) Illegal action: empty pit -> instant loss + terminal.
    c_reset(&env);
    env.board[P0_PITS_START + 0] = 0;          // make pit 0 empty
    env.terminals[0] = NOT_DONE;
    env.actions[0] = 0.0f;
    c_step(&env);
    ok &= assert_eq("illegal terminal set", env.terminals[0], DONE);
    ok &= assert_eq("illegal reward = LOSS", (int)env.rewards[0], (int)PLAYER_LOSS);

    free_allocated_cmancala(&env);
    printf(ok ? "RULES: ALL PASS\n" : "RULES: FAILURES ABOVE\n");
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Rule sanity for the turn-by-turn refactor (external_opponent=1 path).
// These exist alongside rules_check and verify that the refactor's new
// behaviors are correct: current_player tracking, no auto-reset on terminal,
// pre_sweep_board snapshot, P1 action handling.
// ---------------------------------------------------------------------------
static int rules_check_external_opponent(void) {
    int ok = 1;
    CMancala env = {0};
    allocate_cmancala(&env);
    init(&env);
    c_reset(&env);
    env.external_opponent = 1;

    // 1) Initial turn-tracking state.
    ok &= assert_eq("ext: initial current_player", env.current_player, 0);

    // 2) Extra turn keeps current_player == 0 and does NOT advance opponent.
    //    From standard start, P0 plays pit 2 (4 stones) -> last lands in store.
    int saved_p1[NUM_PITS];
    for (int i = 0; i < NUM_PITS; i++) saved_p1[i] = env.board[P1_PITS_START + i];
    env.actions[0] = 2.0f;
    c_step(&env);
    ok &= assert_eq("ext: extra-turn keeps current_player 0", env.current_player, 0);
    ok &= assert_eq("ext: extra-turn terminal not set", env.terminals[0], NOT_DONE);
    int p1_changed = 0;
    for (int i = 0; i < NUM_PITS; i++) {
        if (env.board[P1_PITS_START + i] != saved_p1[i]) p1_changed = 1;
    }
    ok &= assert_eq("ext: extra-turn opponent untouched", p1_changed, 0);

    // 3) No-extra move flips current_player to 1, P1 has NOT auto-played.
    c_reset(&env);
    env.external_opponent = 1;
    for (int i = 0; i < NUM_PITS; i++) saved_p1[i] = env.board[P1_PITS_START + i];
    env.actions[0] = 0.0f;  // pit 0 has 4 stones; lands in pit 4 (not store, not capture)
    c_step(&env);
    ok &= assert_eq("ext: no-extra flips to current_player 1", env.current_player, 1);
    p1_changed = 0;
    for (int i = 0; i < NUM_PITS; i++) {
        if (env.board[P1_PITS_START + i] != saved_p1[i]) p1_changed = 1;
    }
    ok &= assert_eq("ext: P1 has not auto-played yet", p1_changed, 0);

    // 4) P1 action advances and flips back to current_player 0 if no extra.
    //    P1 pit 0 has 4 stones now; sows to its pits 1..4 (not store).
    env.actions[0] = 0.0f;
    c_step(&env);
    ok &= assert_eq("ext: P1 no-extra flips to current_player 0", env.current_player, 0);

    // 5) Terminal step does NOT auto-reset under external_opponent=1.
    //    Hand-craft the same end-game scenario as rules_check #4.
    c_reset(&env);
    env.external_opponent = 1;
    for (int i = 0; i < BOARD_SIZE; i++) env.board[i] = 0;
    env.board[P0_PITS_START + 5] = 1;
    env.board[P1_PITS_START + 0] = 3;
    env.board[P1_PITS_START + 4] = 2;
    env.terminals[0] = NOT_DONE;
    env.actions[0] = 5.0f;
    c_step(&env);
    ok &= assert_eq("ext: terminal set", env.terminals[0], DONE);
    ok &= assert_eq("ext: post-sweep P0 store == 1", env.board[P0_STORE], 1);
    ok &= assert_eq("ext: post-sweep P1 store == 5", env.board[P1_STORE], 5);
    // pre_sweep_board snapshot was taken BEFORE sweep_remaining: P0 pit 5 was
    // emptied (stone went to store), P1 still had 3 in pit 0 and 2 in pit 4.
    ok &= assert_eq("ext: pre_sweep P0 store == 1",  env.pre_sweep_board[P0_STORE], 1);
    ok &= assert_eq("ext: pre_sweep P1 store == 0",  env.pre_sweep_board[P1_STORE], 0);
    ok &= assert_eq("ext: pre_sweep P1 pit 0 == 3",  env.pre_sweep_board[P1_PITS_START + 0], 3);
    ok &= assert_eq("ext: pre_sweep P1 pit 4 == 2",  env.pre_sweep_board[P1_PITS_START + 4], 2);

    // 6) Caller-driven c_reset works after terminal under external mode.
    env.terminals[0] = NOT_DONE;
    env.rewards[0] = 0.0f;
    c_reset(&env);
    ok &= assert_eq("ext: post-reset current_player", env.current_player, 0);
    ok &= assert_eq("ext: post-reset P0 pit 0 == 4", env.board[P0_PITS_START + 0], INIT_STONES);

    // 7) external_opponent flag survives c_reset.
    ok &= assert_eq("ext: external_opponent preserved", env.external_opponent, 1);

    // 8) P1 illegal move terminates the episode with reward = WIN (P0's view).
    c_reset(&env);
    env.external_opponent = 1;
    env.actions[0] = 0.0f;
    c_step(&env);  // P0 pit 0 — flips to P1's turn
    if (env.current_player != 1) {
        printf("FAIL ext: setup for P1-illegal expected current_player=1, got %d\n", env.current_player);
        ok = 0;
    } else {
        // Force P1 pit 5 to be empty, then have P1 play it.
        env.board[P1_PITS_START + 5] = 0;
        env.actions[0] = 5.0f;
        c_step(&env);
        ok &= assert_eq("ext: P1 illegal terminal set", env.terminals[0], DONE);
        ok &= assert_eq("ext: P1 illegal reward = WIN (P0 view)",
                        (int)env.rewards[0], (int)PLAYER_WIN);
    }

    free_allocated_cmancala(&env);
    printf(ok ? "RULES (external): ALL PASS\n" : "RULES (external): FAILURES ABOVE\n");
    return ok ? 0 : 1;
}

static const int MANCALA_HIDDEN_SIZE = 128;
static const int MANCALA_NUM_LAYERS  = 4;

static long file_size_bytes(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fclose(f);
    return n;
}


// ---------------------------------------------------------------------------
// Human vs trained policy with animated raylib graphics.
//
// Layout: P1 (red, top row) is the human seat — clickable / 1-6 keys.
// P0 (cyan, bottom row) is the trained model.
// Each c_step under env.external_opponent=1 plays a single player's move; the
// human-play loop alternates P0 (forward_puffernet) and P1 (input) and
// animates each move stone-by-stone before settling to the post-state.
//
// Layout (human sits at the bottom, AI faces from the top):
//
//   ┌───┬───────────────────────────────┬───┐
//   │ P0│  [ 5 ][ 4 ][ 3 ][ 2 ][ 1 ][ 0 ]│   │   ← P0 (AI) pits, drawn right→left
//   │ S │                                │ P1│     so capture-opposites align
//   │   │  [ 0 ][ 1 ][ 2 ][ 3 ][ 4 ][ 5 ]│ S │   ← P1 (human) pits, left→right
//   └───┴───────────────────────────────┴───┘
//
// Number keys 1-6 map left→right to the bottom row's local indices 0..5
// (key 1 = local 0, key 6 = local 5). Labels under the bottom row read 1-6
// in screen order.
// ---------------------------------------------------------------------------

#include <math.h>

#define HUMAN_TARGET_FPS 60
#define HUMAN_AI_THINK_SECS 0.30f      // pause before AI's forward pass
#define HUMAN_PER_STONE_SOW 0.13f      // sow stone in flight
#define HUMAN_PER_STONE_CAP 0.07f      // capture stone in flight
#define HUMAN_PER_STONE_SWP 0.05f      // sweep stone in flight
#define HUMAN_ANIM_MAX_STEPS 96
#define HUMAN_ANIM_MAX_PHASES 4

typedef enum {
    HS_AI_THINKING = 0,
    HS_INPUT,
    HS_ANIM,
    HS_GAME_OVER,
} HumanState;

typedef enum {
    APH_NONE = 0,
    APH_SOW,
    APH_CAPTURE,
    APH_SWEEP,
} AnimPhaseKind;

typedef struct {
    int source;     // board index 0..13
    int target;     // board index 0..13
    AnimPhaseKind phase;  // phase this step belongs to (for per-step delay)
} AnimStep;

typedef struct {
    AnimStep steps[HUMAN_ANIM_MAX_STEPS];
    int n;                 // total queued steps
    int i;                 // index of step currently in flight
    float t;               // 0..1 progress on step i
    int lifted;            // 1 if step i's source has been visually emptied
    int display_board[BOARD_SIZE];   // mutable: stones land here as anim plays

    // Per-move metadata, used for the HUD status line.
    int move_player;       // 0 or 1
    int move_pit;          // local 0..5
    int captured;          // 0 or 1 (whether this move captured)
    int finished_game;     // 0 or 1 (whether this move ended the game)
} Anim;

// Pit-center geometry: P0 (AI) on top drawn right→left, P1 (human) on
// bottom drawn left→right. P0_STORE on the left, P1_STORE on the right.
static void pit_world_xy(const Client* cli, int board_idx, int* out_x, int* out_y) {
    int store_h = cli->height - 2 * MARGIN_Y;
    int pit_area_w = cli->width - 2 * MARGIN_X - 2 * STORE_W - 24;
    int pit_pitch = pit_area_w / NUM_PITS;
    int pits_left = MARGIN_X + STORE_W + 12 + pit_pitch / 2;
    int row_y_top = MARGIN_Y + 56;
    int row_y_bot = cli->height - MARGIN_Y - 56;

    if (board_idx == P0_STORE) {
        *out_x = MARGIN_X + STORE_W / 2;            // left side (AI store)
        *out_y = MARGIN_Y + store_h / 2;
    } else if (board_idx == P1_STORE) {
        *out_x = cli->width - MARGIN_X - STORE_W / 2;  // right side (human store)
        *out_y = MARGIN_Y + store_h / 2;
    } else if (board_idx >= P0_PITS_START && board_idx < P0_PITS_START + NUM_PITS) {
        // P0 (AI) pit — top row, drawn right→left so opposites align.
        int local = board_idx - P0_PITS_START;
        *out_x = pits_left + (NUM_PITS - 1 - local) * pit_pitch;
        *out_y = row_y_top;
    } else {
        // P1 (human) pit — bottom row, drawn left→right.
        int local = board_idx - P1_PITS_START;
        *out_x = pits_left + local * pit_pitch;
        *out_y = row_y_bot;
    }
}

// Hover detection — returns local P1 pit index 0..5, or -1.
static int hovered_p1_pit(const Client* cli, Vector2 mouse) {
    for (int i = 0; i < NUM_PITS; i++) {
        int cx, cy;
        pit_world_xy(cli, P1_PITS_START + i, &cx, &cy);
        float dx = mouse.x - cx, dy = mouse.y - cy;
        if (dx * dx + dy * dy <= (float)(PIT_RADIUS * PIT_RADIUS)) return i;
    }
    return -1;
}

// Compute the per-stone landing path for a sow without mutating any board —
// just walks the standard sowing rule (skip opponent store) starting from
// in_board. Fills out_path[0..stones-1]; returns the number of stones sown.
static int compute_sow_path(const int* in_board, int player, int local_pit,
                             int* out_path) {
    int idx = pit_start_of(player) + local_pit;
    int stones = in_board[idx];
    int skip = opp_store_of(player);
    int k = 0;
    while (k < stones) {
        idx = (idx + 1) % BOARD_SIZE;
        if (idx == skip) continue;
        out_path[k++] = idx;
    }
    return stones;
}

// Predict the capture outcome of (player, local_pit) starting from in_board,
// reusing mancala.h's sow + apply_capture on a temp env to avoid duplicating
// game logic. Returns the per-stone path (for animation) plus capture metadata.
static void shadow_sow_capture(const int* in_board, int player, int local_pit,
                                int* out_path, int* out_path_len,
                                int* out_last_idx,
                                int* out_did_capture, int* out_capture_taken) {
    *out_path_len = compute_sow_path(in_board, player, local_pit, out_path);
    *out_last_idx = (*out_path_len > 0)
        ? out_path[*out_path_len - 1]
        : pit_start_of(player) + local_pit;  // empty source — no movement

    CMancala tmp = {0};
    memcpy(tmp.board, in_board, sizeof(tmp.board));
    sow(&tmp, player, local_pit);
    int captured = apply_capture(&tmp, player, *out_last_idx);
    *out_did_capture   = (captured > 0) ? 1 : 0;
    *out_capture_taken = (captured > 0) ? captured - 1 : 0;
}

// Construct the animation queue for one move, given pre-step board, the action,
// the player, and the env's post-step state. Handles SOW + optional CAPTURE +
// optional SWEEP (when the move ended the game).
static void anim_build(Anim* a, const int* pre_board, int player, int local_pit,
                       const CMancala* env) {
    a->n = 0;
    a->i = 0;
    a->t = 0.0f;
    a->lifted = 0;
    memcpy(a->display_board, pre_board, BOARD_SIZE * sizeof(int));
    a->move_player = player;
    a->move_pit = local_pit;
    a->captured = 0;
    a->finished_game = (env->terminals[0] == DONE) ? 1 : 0;

    int path[BOARD_SIZE], path_len = 0;
    int last_idx = 0, did_cap = 0, cap_taken = 0;
    shadow_sow_capture(pre_board, player, local_pit,
                       path, &path_len, &last_idx, &did_cap, &cap_taken);
    a->captured = did_cap;

    int source = pit_start_of(player) + local_pit;

    // SOW phase: each stone leaves the source pit, lands at path[k].
    for (int k = 0; k < path_len; k++) {
        if (a->n >= HUMAN_ANIM_MAX_STEPS) break;
        a->steps[a->n].source = source;
        a->steps[a->n].target = path[k];
        a->steps[a->n].phase  = APH_SOW;
        a->n++;
    }

    // CAPTURE phase: the just-sown stone (1) at last_idx and the cap_taken
    // stones at the opposite pit fly to the player's store.
    if (did_cap) {
        int store = store_of(player);
        int opposite = 12 - last_idx;
        if (a->n < HUMAN_ANIM_MAX_STEPS) {
            a->steps[a->n].source = last_idx;
            a->steps[a->n].target = store;
            a->steps[a->n].phase  = APH_CAPTURE;
            a->n++;
        }
        for (int k = 0; k < cap_taken && a->n < HUMAN_ANIM_MAX_STEPS; k++) {
            a->steps[a->n].source = opposite;
            a->steps[a->n].target = store;
            a->steps[a->n].phase  = APH_CAPTURE;
            a->n++;
        }
    }

    // SWEEP phase: read pre_sweep_board (set by c_step on terminal) for source
    // distribution; targets are the side's own store. env->board is the final
    // post-sweep state, used implicitly to validate the queue.
    if (a->finished_game) {
        for (int p = 0; p < 2; p++) {
            int start = pit_start_of(p);
            for (int i = 0; i < NUM_PITS; i++) {
                int idx = start + i;
                int n_left = env->pre_sweep_board[idx];
                int store = store_of(p);
                for (int s = 0; s < n_left && a->n < HUMAN_ANIM_MAX_STEPS; s++) {
                    a->steps[a->n].source = idx;
                    a->steps[a->n].target = store;
                    a->steps[a->n].phase  = APH_SWEEP;
                    a->n++;
                }
            }
        }
    }
}

// Per-frame tick. Returns 1 when the animation has fully completed.
static int anim_tick(Anim* a, float dt) {
    while (a->i < a->n) {
        AnimPhaseKind ph = a->steps[a->i].phase;
        float per = (ph == APH_SOW)     ? HUMAN_PER_STONE_SOW
                  : (ph == APH_CAPTURE) ? HUMAN_PER_STONE_CAP
                  :                       HUMAN_PER_STONE_SWP;
        // Lift the source pit's stone exactly once per step (a t==0 check
        // would re-lift if dt is 0 on the entry frame).
        if (!a->lifted) {
            int src = a->steps[a->i].source;
            if (a->display_board[src] > 0) a->display_board[src] -= 1;
            a->lifted = 1;
        }
        a->t += dt / per;
        if (a->t < 1.0f) return 0;
        a->display_board[a->steps[a->i].target] += 1;
        a->i++;
        a->t = 0.0f;
        a->lifted = 0;
    }
    return 1;
}

static void anim_skip_to_end(Anim* a) {
    while (a->i < a->n) {
        if (!a->lifted) {
            int src = a->steps[a->i].source;
            if (a->display_board[src] > 0) a->display_board[src] -= 1;
        }
        a->display_board[a->steps[a->i].target] += 1;
        a->i++;
        a->t = 0.0f;
        a->lifted = 0;
    }
}

// Smoothstep easing for the in-flight stone.
static float smooth01(float t) {
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    return t * t * (3.0f - 2.0f * t);
}

// Render the human-mode UI based on Anim.display_board and surrounding state.
// Mirrors c_render's geometry exactly so animation and static frames align.
static void human_render(const CMancala* env, const Anim* a, HumanState state,
                         int hover_pit, int wins, int losses, int draws,
                         int last_ai_move, float store_pulse_t) {
    Client* cli = env->client;
    int store_h = cli->height - 2 * MARGIN_Y;
    int pit_area_w = cli->width - 2 * MARGIN_X - 2 * STORE_W - 24;
    int pit_pitch = pit_area_w / NUM_PITS;
    int pits_left = MARGIN_X + STORE_W + 12 + pit_pitch / 2;
    int row_y_top = MARGIN_Y + 56;
    int row_y_bot = cli->height - MARGIN_Y - 56;

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    // Top header strip. Order matches the on-screen left→right order of the
    // store columns (P0 store on the left, P1 store on the right).
    char header[160];
    snprintf(header, sizeof(header),
             "MANCALA   tick %-4d   P0 %2d  vs  %2d P1",
             env->tick,
             a->display_board[P0_STORE], a->display_board[P1_STORE]);
    DrawTextEx(cli->font_small, header, (Vector2){MARGIN_X, 22},
               14, 0, Fade(PUFF_WHITE, 0.7f));

    // Top-right tally.
    char tally[64];
    snprintf(tally, sizeof(tally), "GAMES   YOU %d  AI %d  DRAW %d",
             wins, losses, draws);
    Vector2 tm = MeasureTextEx(cli->font_small, tally, 14, 0);
    DrawTextEx(cli->font_small, tally,
               (Vector2){cli->width - MARGIN_X - tm.x, 22},
               14, 0, Fade(PUFF_WHITE, 0.7f));

    // Frame.
    Rectangle frame = {MARGIN_X - 6, MARGIN_Y - 6,
                       cli->width - 2 * (MARGIN_X - 6),
                       cli->height - 2 * MARGIN_Y + 12};
    DrawRectangleRoundedLines(frame, 0.04f, 8, Fade(PUFF_WHITE, 0.08f));

    // Active-side glow on stores. Pulse alpha 0.6→1.0 over ~1s.
    float pulse = 0.6f + 0.4f * (0.5f + 0.5f * sinf(store_pulse_t * 3.14159f));
    int active_player = -1;
    if (state == HS_AI_THINKING || (state == HS_ANIM && a->move_player == 0)) active_player = 0;
    else if (state == HS_INPUT || (state == HS_ANIM && a->move_player == 1)) active_player = 1;

    Color p1_rim = (active_player == 1) ? Fade(PUFF_RED,  pulse) : Fade(PUFF_RED,  0.45f);
    Color p0_rim = (active_player == 0) ? Fade(PUFF_CYAN, pulse) : Fade(PUFF_CYAN, 0.45f);

    // P0 store on the left, P1 store on the right (matches new layout).
    draw_store(cli->font_big, cli->font_small,
               MARGIN_X, MARGIN_Y, STORE_W, store_h,
               a->display_board[P0_STORE], "P0", p0_rim);
    draw_store(cli->font_big, cli->font_small,
               cli->width - MARGIN_X - STORE_W, MARGIN_Y, STORE_W, store_h,
               a->display_board[P1_STORE], "P1", p1_rim);

    // P0 (AI) pits — top row, drawn right→left so capture-opposites align
    // vertically with the human's pits below.
    for (int i = 0; i < NUM_PITS; i++) {
        int cx = pits_left + (NUM_PITS - 1 - i) * pit_pitch;
        Color rim = PUFF_CYAN;
        // Highlight last AI move briefly during sow animation.
        if (state == HS_ANIM && a->move_player == 0 && a->move_pit == i
            && a->steps[a->i < a->n ? a->i : a->n - 1].phase == APH_SOW) {
            rim = (Color){80, 220, 220, 255};
        }
        draw_pit(cli->font_big, cx, row_y_top, PIT_RADIUS,
                 a->display_board[P0_PITS_START + i], rim);
        char lbl[4]; snprintf(lbl, sizeof(lbl), "%d", i);
        draw_text_centered(cli->font_small, lbl,
                           cx, row_y_top - PIT_RADIUS - 14,
                           12, Fade(PUFF_CYAN, 0.55f));
    }

    // P1 (human) pits — bottom row, drawn left→right (visual L→R = local 0..5).
    // Number keys 1-6 map directly: key (i+1) → local pit i.
    for (int i = 0; i < NUM_PITS; i++) {
        int cx = pits_left + i * pit_pitch;
        Color rim = PUFF_RED;
        int stones = a->display_board[P1_PITS_START + i];
        if (state == HS_INPUT && hover_pit == i && stones > 0) {
            rim = (Color){255, 100, 100, 255};
        }
        draw_pit(cli->font_big, cx, row_y_bot, PIT_RADIUS, stones, rim);
        char lbl[4]; snprintf(lbl, sizeof(lbl), "%d", i + 1);
        draw_text_centered(cli->font_small, lbl,
                           cx, row_y_bot + PIT_RADIUS + 14,
                           12, Fade(PUFF_RED, 0.55f));
    }

    // In-flight stone during animation.
    if (state == HS_ANIM && a->i < a->n) {
        int sx, sy, tx, ty;
        pit_world_xy(cli, a->steps[a->i].source, &sx, &sy);
        pit_world_xy(cli, a->steps[a->i].target, &tx, &ty);
        float u = smooth01(a->t);
        float x = (1.0f - u) * sx + u * tx;
        float y = (1.0f - u) * sy + u * ty;
        // Slight arc — arcs feel more like a hand sowing stones.
        float arc = -18.0f * sinf(u * 3.14159f);
        DrawCircle((int)x, (int)(y + arc), 7, PUFF_WHITE);
        DrawCircleLines((int)x, (int)(y + arc), 8, Fade(PUFF_WHITE, 0.4f));
    }

    // Bottom status strip.
    char status[160] = {0};
    Color status_color = PUFF_WHITE;
    switch (state) {
        case HS_AI_THINKING:
            snprintf(status, sizeof(status), "AI THINKING…");
            status_color = PUFF_RED;
            break;
        case HS_INPUT:
            snprintf(status, sizeof(status),
                     "YOUR TURN — click a pit or press 1-6 (esc to quit)");
            status_color = PUFF_CYAN;
            break;
        case HS_ANIM:
            if (a->move_player == 0) {
                snprintf(status, sizeof(status), "AI played pit %d%s%s",
                         last_ai_move,
                         a->captured       ? "  ← CAPTURE" : "",
                         a->finished_game  ? "  ← GAME END" : "");
                status_color = Fade(PUFF_RED, 0.85f);
            } else {
                snprintf(status, sizeof(status), "You played pit %d%s%s   (space to skip anim)",
                         a->move_pit + 1,   // visual key number = local + 1
                         a->captured       ? "  ← CAPTURE" : "",
                         a->finished_game  ? "  ← GAME END" : "");
                status_color = Fade(PUFF_CYAN, 0.85f);
            }
            break;
        case HS_GAME_OVER:
            // Status hidden under modal; leave empty.
            break;
    }
    if (status[0]) {
        draw_text_centered(cli->font_small, status,
                           cli->width / 2, cli->height - 20,
                           14, status_color);
    }

    // Game-over modal.
    if (state == HS_GAME_OVER) {
        DrawRectangle(0, 0, cli->width, cli->height, Fade(BLACK, 0.65f));
        int mw = 520, mh = 240;
        int mx = (cli->width - mw) / 2;
        int my = (cli->height - mh) / 2;
        Rectangle m = {mx, my, mw, mh};
        DrawRectangleRounded(m, 0.10f, 12, PIT_FILL);
        DrawRectangleRoundedLines(m, 0.10f, 12, Fade(PUFF_WHITE, 0.4f));

        const char* big;
        Color big_color;
        float r = env->rewards[0];
        if (r > 0.5f) {
            // P0 (the AI) wins. Human plays P1 → human loses.
            big = "AI WINS"; big_color = PUFF_RED;
        } else if (r < -0.5f) {
            // P0 loses, human (P1) wins.
            big = "YOU WIN!"; big_color = PUFF_CYAN;
        } else {
            big = "DRAW"; big_color = PUFF_WHITE;
        }
        draw_text_centered(cli->font_big, big, mx + mw / 2, my + 70, 44, big_color);

        char sub[96];
        snprintf(sub, sizeof(sub), "Final  P0 %d  —  %d P1   (margin %+d)",
                 a->display_board[P0_STORE], a->display_board[P1_STORE],
                 a->display_board[P1_STORE] - a->display_board[P0_STORE]);
        draw_text_centered(cli->font_small, sub,
                           mx + mw / 2, my + 130, 16, Fade(PUFF_WHITE, 0.85f));

        const char* foot = "Press R to play again  ·  ESC to quit";
        draw_text_centered(cli->font_small, foot,
                           mx + mw / 2, my + 190, 14, Fade(PUFF_WHITE, 0.5f));
    }

    EndDrawing();
}

// If set via env var MANCALA_HUMAN_SCREENSHOT=path.png, capture the initial
// frame and exit. Used to verify render layout without an interactive session.
static const char* human_play_screenshot_path(void) {
    const char* p = getenv("MANCALA_HUMAN_SCREENSHOT");
    return (p && p[0]) ? p : NULL;
}

static int human_play(const char* weights_path) {
    long bytes = file_size_bytes(weights_path);
    if (bytes < 0) {
        fprintf(stderr, "Could not open weights file: %s\n", weights_path);
        return 1;
    }
    printf("Loading %ld bytes from %s\n", bytes, weights_path);
    Weights* w = load_weights(weights_path);

    int logit_sizes[1] = {NUM_PITS};
    PufferNet* net = make_puffernet(w, /*num_agents=*/1, /*input_dim=*/OBS_DIM,
                                    MANCALA_HIDDEN_SIZE, MANCALA_NUM_LAYERS,
                                    logit_sizes, /*num_actions=*/1);

    CMancala env = {0};
    allocate_cmancala(&env);
    init(&env);
    env.rng = (unsigned int)time(NULL);
    c_reset(&env);
    env.external_opponent = 1;

    env.client = make_client();
    SetTargetFPS(HUMAN_TARGET_FPS);

    HumanState state = HS_AI_THINKING;
    int wins = 0, losses = 0, draws = 0;
    float ai_timer = 0.0f;
    float store_pulse_t = 0.0f;
    int last_ai_move = -1;
    Anim anim = {0};
    memcpy(anim.display_board, env.board, sizeof(anim.display_board));

    int pre_board[BOARD_SIZE];

    const char* screenshot_path = human_play_screenshot_path();
    int screenshot_frame = 0;

    while (!WindowShouldClose() && !IsKeyPressed(KEY_ESCAPE)) {
        float dt = GetFrameTime();
        if (dt > 0.10f) dt = 0.10f;
        store_pulse_t += dt;

        Vector2 mouse = GetMousePosition();
        int hover_pit = (state == HS_INPUT) ? hovered_p1_pit(env.client, mouse) : -1;

        // ---- update ----
        switch (state) {
        case HS_AI_THINKING: {
            ai_timer += dt;
            if (ai_timer >= HUMAN_AI_THINK_SECS) {
                ai_timer = 0.0f;
                memcpy(pre_board, env.board, sizeof(pre_board));
                float net_action[1] = {0};
                forward_puffernet(net, env.observations, net_action);
                int act = (int)net_action[0];
                if (act < 0 || act >= NUM_PITS) act = 0;
                last_ai_move = act;
                env.actions[0] = (float)act;
                c_step(&env);
                anim_build(&anim, pre_board, 0, act, &env);
                state = HS_ANIM;
            }
            break;
        }
        case HS_INPUT: {
            int chosen = -1;
            if (hover_pit >= 0
                && env.board[P1_PITS_START + hover_pit] > 0
                && IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
                chosen = hover_pit;
            }
            int keys[6] = {KEY_ONE, KEY_TWO, KEY_THREE, KEY_FOUR, KEY_FIVE, KEY_SIX};
            for (int k = 0; k < 6 && chosen < 0; k++) {
                if (IsKeyPressed(keys[k])) {
                    // Bottom row is drawn left-to-right in local indices, so
                    // key (k+1) maps directly to local pit k.
                    int local = k;
                    if (env.board[P1_PITS_START + local] > 0) chosen = local;
                }
            }
            if (chosen >= 0) {
                memcpy(pre_board, env.board, sizeof(pre_board));
                env.actions[0] = (float)chosen;
                c_step(&env);
                anim_build(&anim, pre_board, 1, chosen, &env);
                state = HS_ANIM;
            }
            break;
        }
        case HS_ANIM: {
            if (IsKeyPressed(KEY_SPACE)) anim_skip_to_end(&anim);
            int done = anim_tick(&anim, dt);
            if (done) {
                if (env.terminals[0] == DONE) {
                    float r = env.rewards[0];
                    if      (r > 0.5f)  losses++;   // P0 won → human (P1) lost
                    else if (r < -0.5f) wins++;     // P0 lost → human won
                    else                draws++;
                    state = HS_GAME_OVER;
                } else if (env.current_player == 0) {
                    state = HS_AI_THINKING;
                    ai_timer = 0.0f;
                } else {
                    state = HS_INPUT;
                }
            }
            break;
        }
        case HS_GAME_OVER: {
            if (IsKeyPressed(KEY_R)) {
                env.terminals[0] = NOT_DONE;
                env.rewards[0] = 0.0f;
                c_reset(&env);
                memcpy(anim.display_board, env.board, sizeof(anim.display_board));
                anim.n = 0; anim.i = 0; anim.t = 0.0f; anim.lifted = 0;
                last_ai_move = -1;
                state = HS_AI_THINKING;
                ai_timer = 0.0f;
            }
            break;
        }
        }

        // ---- render ----
        human_render(&env, &anim, state, hover_pit, wins, losses, draws,
                     last_ai_move, store_pulse_t);

        if (screenshot_path) {
            // Need a few frames so raylib's front buffer is populated.
            if (++screenshot_frame == 5) {
                TakeScreenshot(screenshot_path);
                printf("Wrote screenshot to %s\n", screenshot_path);
                break;
            }
        }
    }

    c_close(&env);
    free_allocated_cmancala(&env);
    // puffernet's load_weights/make_puffernet don't expose matching free_*
    // for every sub-allocation; freeing here triggered an asan bad-free.
    // Process exit reclaims the memory.
    (void)net; (void)w;
    return 0;
}

// Subcommand dispatch. Usage:
//   ./mancala                       rules_check + perf
//   ./mancala p                     perf only (random actions, headless)
//   ./mancala human <weights.bin>   interactive raylib play vs trained policy
int main(int argc, char** argv) {
    srand(42);
    const char* cmd = (argc > 1) ? argv[1] : "";
    if (strcmp(cmd, "p") == 0) { performance_test(); return 0; }
    if (argc >= 3 && strcmp(cmd, "human") == 0) return human_play(argv[2]);
    int rc = rules_check();
    if (rc != 0) return rc;
    rc = rules_check_external_opponent();
    if (rc != 0) return rc;
    performance_test();
    return 0;
}
