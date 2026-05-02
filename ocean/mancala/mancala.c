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

// Hover detection — returns local P1 pit index 0..5, or -1.
static int hovered_p1_pit(const Client* cli, Vector2 mouse) {
    for (int i = 0; i < NUM_PITS; i++) {
        int cx, cy;
        pit_center(cli, P1_PITS_START + i, &cx, &cy);
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
    float pulse = 0.6f + 0.4f * (0.5f + 0.5f * sinf(store_pulse_t * 3.14159f));

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    c_draw((CMancala*)env, a->display_board, pulse);

    // Top-right tally
    char tally[64];
    snprintf(tally, sizeof(tally), "GAMES   YOU %d  AI %d  DRAW %d",
             wins, losses, draws);
    Vector2 tm = MeasureTextEx(cli->font_small, tally, 14, 0);
    DrawTextEx(cli->font_small, tally,
               (Vector2){cli->width - MARGIN_X - tm.x, 22},
               14, 0, Fade(PUFF_WHITE, 0.7f));

    // Hover ring on the human's row
    if (state == HS_INPUT && hover_pit >= 0
        && a->display_board[P1_PITS_START + hover_pit] > 0) {
        int cx, cy; pit_center(cli, P1_PITS_START + hover_pit, &cx, &cy);
        Rectangle r = {cx - PIT_RADIUS, cy - PIT_RADIUS, 2*PIT_RADIUS, 2*PIT_RADIUS};
        DrawRectangleRoundedLines(r, 0.40f, 8, (Color){255, 100, 100, 255});
    }

    // Highlight the AI's source pit during sow animation
    if (state == HS_ANIM && a->move_player == 0 && a->i < a->n
        && a->steps[a->i].phase == APH_SOW) {
        int cx, cy; pit_center(cli, P0_PITS_START + a->move_pit, &cx, &cy);
        Rectangle r = {cx - PIT_RADIUS, cy - PIT_RADIUS, 2*PIT_RADIUS, 2*PIT_RADIUS};
        DrawRectangleRoundedLines(r, 0.40f, 8, (Color){80, 220, 220, 255});
    }

    // In-flight stone
    if (state == HS_ANIM && a->i < a->n) {
        int sx, sy, tx, ty;
        pit_center(cli, a->steps[a->i].source, &sx, &sy);
        pit_center(cli, a->steps[a->i].target, &tx, &ty);
        float u = smooth01(a->t);
        float x = (1.0f - u) * sx + u * tx;
        float y = (1.0f - u) * sy + u * ty;
        float arc = -18.0f * sinf(u * 3.14159f);  // arcs feel like a hand sowing
        DrawCircle((int)x, (int)(y + arc), 7, PUFF_WHITE);
        DrawCircleLines((int)x, (int)(y + arc), 8, Fade(PUFF_WHITE, 0.4f));
    }

    // Status strip
    char status[160] = {0};
    Color status_color = PUFF_WHITE;
    if (state == HS_AI_THINKING) {
        snprintf(status, sizeof(status), "AI THINKING…");
        status_color = PUFF_RED;
    } else if (state == HS_INPUT) {
        snprintf(status, sizeof(status),
                 "YOUR TURN — click a pit or press 1-6 (esc to quit)");
        status_color = PUFF_CYAN;
    } else if (state == HS_ANIM) {
        const char* tail = a->captured      ? "  ← CAPTURE"
                         : a->finished_game ? "  ← GAME END" : "";
        if (a->move_player == 0) {
            snprintf(status, sizeof(status), "AI played pit %d%s",
                     last_ai_move, tail);
            status_color = Fade(PUFF_RED, 0.85f);
        } else {
            snprintf(status, sizeof(status),
                     "You played pit %d%s   (space to skip anim)",
                     a->move_pit + 1, tail);
            status_color = Fade(PUFF_CYAN, 0.85f);
        }
    }
    if (status[0]) {
        draw_text_centered(cli->font_small, status,
                           cli->width / 2, cli->height - 20, 14, status_color);
    }

    if (state == HS_GAME_OVER) {
        DrawRectangle(0, 0, cli->width, cli->height, Fade(BLACK, 0.65f));
        int mw = 520, mh = 240;
        int mx = (cli->width - mw) / 2;
        int my = (cli->height - mh) / 2;
        Rectangle m = {mx, my, mw, mh};
        DrawRectangleRounded(m, 0.10f, 12, PIT_FILL);
        DrawRectangleRoundedLines(m, 0.10f, 12, Fade(PUFF_WHITE, 0.4f));

        // Reward is from P0's POV. Human is P1, so flip the messaging.
        float r = env->rewards[0];
        const char* big = (r >  0.5f) ? "AI WINS"
                        : (r < -0.5f) ? "YOU WIN!" : "DRAW";
        Color big_color = (r >  0.5f) ? PUFF_RED
                        : (r < -0.5f) ? PUFF_CYAN : PUFF_WHITE;
        draw_text_centered(cli->font_big, big, mx + mw/2, my + 70, 44, big_color);

        char sub[96];
        snprintf(sub, sizeof(sub), "Final  P0 %d  —  %d P1   (margin %+d)",
                 a->display_board[P0_STORE], a->display_board[P1_STORE],
                 a->display_board[P1_STORE] - a->display_board[P0_STORE]);
        draw_text_centered(cli->font_small, sub,
                           mx + mw/2, my + 130, 16, Fade(PUFF_WHITE, 0.85f));
        draw_text_centered(cli->font_small,
                           "Press R to play again  ·  ESC to quit",
                           mx + mw/2, my + 190, 14, Fade(PUFF_WHITE, 0.5f));
    }

    EndDrawing();
}

// MANCALA_SCREENSHOT=path.png captures the initial frame and exits — used to
// verify render layout without an interactive session.
static const char* screenshot_path(void) {
    const char* p = getenv("MANCALA_SCREENSHOT");
    return (p && p[0]) ? p : NULL;
}

static int demo(void) {
    const char* weights_path = "resources/mancala/mancala_weights.bin";
    long bytes = file_size_bytes(weights_path);
    if (bytes < 0) {
        fprintf(stderr, "Could not open %s\n", weights_path);
        return 1;
    }
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

    const char* shot_path = screenshot_path();
    int shot_frame = 0;

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

        if (shot_path && ++shot_frame == 5) {
            TakeScreenshot(shot_path);
            printf("Wrote screenshot to %s\n", shot_path);
            break;
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

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "p") == 0) {
        performance_test();
        return 0;
    }
    return demo();
}
