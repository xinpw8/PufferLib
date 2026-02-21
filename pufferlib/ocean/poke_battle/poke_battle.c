// poke_battle.c - Standalone PokeBattle demo with game modes
// Watch: PufferAI auto-plays vs random bot with speed controls
// Play: Click opponent button to play human vs AI (PufferAI/MCTS/Random)
// Build: ./scripts/build_ocean.sh poke_battle fast
// Run:   ./poke_battle          (GUI demo)
//        ./poke_battle --perf   (headless SPS benchmark)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "poke_battle.h"
#include "puffernet.h"

// ============================================================================
// Custom PokeBattle Neural Network
// ============================================================================
//
// Architecture (from torch.py PokeBattle + PokeBattleLSTM):
//   Obs (140) → slice into components:
//     my_active+side[0:25], my_moves[25:53], my_team[53:77],
//     opp_active+side[77:102], opp_team[102:126], battle_info[126:130],
//     action_mask[130:140]
//
//   active_encoder (SHARED): Linear(25→64) + ReLU
//   move_encoder:            Linear(28→64) + ReLU
//   team_encoder (SHARED):   Linear(24→48) + ReLU
//   context_encoder:         Linear(4→16) + ReLU
//
//   Concat: [my_active(64)|opp_active(64)|moves(64)|my_team(48)|opp_team(48)|context(16)] = 304
//
//   combine.0: Linear(304→256) + ReLU
//   combine.2: Linear(256→256) + ReLU
//   LSTM:      LSTMCell(256→256)
//   actor:     Linear(256→10) + action_mask → softmax → sample
//   value_fn:  Linear(256→1)
//
// Total weights: 677,835 floats

#define NUM_WEIGHTS 677835

typedef struct PokeBattleNet PokeBattleNet;
struct PokeBattleNet {
    // Encoders (shared encoders used for both my/opp)
    Linear* active_encoder;   // 25 → 64 (shared)
    ReLU*   active_relu;      // 64
    Linear* move_encoder;     // 28 → 64
    ReLU*   move_relu;        // 64
    Linear* team_encoder;     // 24 → 48 (shared)
    ReLU*   team_relu;        // 48
    Linear* context_encoder;  // 4 → 16
    ReLU*   context_relu;     // 16

    // Combine
    Linear* combine0;         // 304 → 256
    ReLU*   combine0_relu;    // 256
    Linear* combine1;         // 256 → 256
    ReLU*   combine1_relu;    // 256

    // Heads
    Linear* actor;            // 256 → 10
    Linear* value_fn;         // 256 → 1
    LSTM*   lstm;             // 256 → 256

    // Buffers for concat and encoder outputs
    float concat_buf[304];
    float logits[NUM_ACTIONS];
};

PokeBattleNet* make_poke_battle_net(Weights* weights) {
    PokeBattleNet* net = calloc(1, sizeof(PokeBattleNet));

    // Weight consumption order must match export_weights.py PARAM_ORDER:
    // 1. active_encoder (Linear 25→64)
    net->active_encoder = make_linear(weights, 1, 25, 64);
    net->active_relu = make_relu(1, 64);

    // 2. move_encoder (Linear 28→64)
    net->move_encoder = make_linear(weights, 1, 28, 64);
    net->move_relu = make_relu(1, 64);

    // 3. team_encoder (Linear 24→48)
    net->team_encoder = make_linear(weights, 1, 24, 48);
    net->team_relu = make_relu(1, 48);

    // 4. context_encoder (Linear 4→16)
    net->context_encoder = make_linear(weights, 1, 4, 16);
    net->context_relu = make_relu(1, 16);

    // 5. combine.0 (Linear 304→256)
    net->combine0 = make_linear(weights, 1, 304, 256);
    net->combine0_relu = make_relu(1, 256);

    // 6. combine.2 (Linear 256→256)
    net->combine1 = make_linear(weights, 1, 256, 256);
    net->combine1_relu = make_relu(1, 256);

    // 7. actor (Linear 256→10)
    net->actor = make_linear(weights, 1, 256, NUM_ACTIONS);

    // 8. value_fn (Linear 256→1)
    net->value_fn = make_linear(weights, 1, 256, 1);

    // 9. LSTM (256→256)
    net->lstm = make_lstm(weights, 1, 256, 256);

    return net;
}

void free_poke_battle_net(PokeBattleNet* net) {
    free(net->active_encoder);
    free(net->active_relu);
    free(net->move_encoder);
    free(net->move_relu);
    free(net->team_encoder);
    free(net->team_relu);
    free(net->context_encoder);
    free(net->context_relu);
    free(net->combine0);
    free(net->combine0_relu);
    free(net->combine1);
    free(net->combine1_relu);
    free(net->actor);
    free(net->value_fn);
    free(net->lstm);
    free(net);
}

void reset_lstm_state(PokeBattleNet* net) {
    int hidden_size = net->lstm->hidden_size;
    memset(net->lstm->state_h, 0, hidden_size * sizeof(float));
    memset(net->lstm->state_c, 0, hidden_size * sizeof(float));
}

void forward_poke_battle_net(PokeBattleNet* net, float* obs, int* action) {
    // Encode my_active+side [0:25] → 64
    linear(net->active_encoder, &obs[0]);
    relu(net->active_relu, net->active_encoder->output);
    memcpy(&net->concat_buf[0], net->active_relu->output, 64 * sizeof(float));

    // Encode opp_active+side [77:102] → 64 (same shared encoder)
    linear(net->active_encoder, &obs[77]);
    relu(net->active_relu, net->active_encoder->output);
    memcpy(&net->concat_buf[64], net->active_relu->output, 64 * sizeof(float));

    // Encode my_moves [25:53] → 64
    linear(net->move_encoder, &obs[25]);
    relu(net->move_relu, net->move_encoder->output);
    memcpy(&net->concat_buf[128], net->move_relu->output, 64 * sizeof(float));

    // Encode my_team [53:77] → 48
    linear(net->team_encoder, &obs[53]);
    relu(net->team_relu, net->team_encoder->output);
    memcpy(&net->concat_buf[192], net->team_relu->output, 48 * sizeof(float));

    // Encode opp_team [102:126] → 48 (same shared encoder)
    linear(net->team_encoder, &obs[102]);
    relu(net->team_relu, net->team_encoder->output);
    memcpy(&net->concat_buf[240], net->team_relu->output, 48 * sizeof(float));

    // Encode battle_info [126:130] → 16
    linear(net->context_encoder, &obs[126]);
    relu(net->context_relu, net->context_encoder->output);
    memcpy(&net->concat_buf[288], net->context_relu->output, 16 * sizeof(float));

    // Combine: 304 → 256 → 256
    linear(net->combine0, net->concat_buf);
    relu(net->combine0_relu, net->combine0->output);
    linear(net->combine1, net->combine0_relu->output);
    relu(net->combine1_relu, net->combine1->output);

    // LSTM
    lstm(net->lstm, net->combine1_relu->output);

    // Actor head
    linear(net->actor, net->lstm->state_h);

    // Apply action mask from obs[130:140]
    float* mask = &obs[130];
    for (int i = 0; i < NUM_ACTIONS; i++) {
        net->logits[i] = net->actor->output[i];
        if (mask[i] == 0.0f) {
            net->logits[i] = -1e9f;
        }
    }

    // Sample action via softmax
    int logit_sizes[1] = {NUM_ACTIONS};
    _softmax_multidiscrete(net->logits, action, 1, logit_sizes, 1);
}

// ============================================================================
// Raylib GUI Demo
// ============================================================================
//
// Modes:
//   MODE_WATCH: PufferAI (P1) auto-plays vs a randomly-chosen bot (P2).
//               Speed controls adjust turn delay. Three "Play vs" buttons
//               let the user start a game as human (P1) against a chosen AI.
//   MODE_PLAY:  Human plays P1 via mouse. P2 is the selected opponent.
//               Game result → click → back to watch mode with a new random bot.

void demo() {
    // Two nets share weight data but have separate LSTM states.
    // net_p1: drives P1 in watch mode.
    // net_p2: drives P2 when the human plays vs PufferAI.
    Weights* weights = load_weights("pufferlib/resources/poke_battle/poke_battle_weights.bin", NUM_WEIGHTS);
    PokeBattleNet* net_p1 = make_poke_battle_net(weights);
    weights->idx = 0;
    PokeBattleNet* net_p2 = make_poke_battle_net(weights);

    // Env with arrays sized for 2 agents (needed when PufferAI is P2)
    PokeBattle env;
    memset(&env, 0, sizeof(PokeBattle));

    float obs[OBS_SIZE * 2] = {0};
    int actions[2] = {0};
    float rewards[2] = {0};
    unsigned char terminals[2] = {0};

    env.observations = obs;
    env.actions = actions;
    env.rewards = rewards;
    env.terminals = terminals;
    env.selfplay = 0;
    env.auto_reset = 0;
    env.seed = (unsigned long long)time(NULL);

    // Game mode: 0 = watch, 1 = play
    int mode = 0;
    int waiting_for_restart = 0;

    // Speed control (watch mode)
    int speed = 1;
    const int speed_delays[] = {120, 60, 15, 1};
    const char* speed_names[] = {"Slow", "Normal", "Fast", "Instant"};
    int frame_counter = 0;

    // Play mode opponent: 0 = PufferAI, 1 = Foul Play (MCTS), 2 = Random
    int play_opponent = 0;

    // Win/loss/draw records
    int watch_wins = 0, watch_losses = 0, watch_draws = 0;
    int play_wins = 0, play_losses = 0, play_draws = 0;

    // Watch mode P2 bot (re-rolled each game)
    int watch_bot = rand() % 3;
    const char* bot_names[] = {"Random", "Heuristic", "MCTS"};

    // Start first watch game
    env.num_agents = 1;
    env.bot_mode = watch_bot;
    init(&env);
    c_reset(&env);
    reset_lstm_state(net_p1);

    env.client = make_client(&env);
    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        Client* client = env.client;
        Vector2 mouse = GetMousePosition();

        if (mode == 0) {
            // ============================================================
            // WATCH MODE: PufferAI (P1) auto-plays vs bot (P2)
            // Hold SHIFT to take over P1 with mouse controls.
            // ============================================================

            int human_override = IsKeyDown(KEY_LEFT_SHIFT) && !waiting_for_restart;

            // Set player labels
            snprintf(client->p2_label, 32, "P2: %s", bot_names[watch_bot]);
            if (human_override)
                snprintf(client->p1_label, 32, "P1: You (SHIFT)");
            else
                snprintf(client->p1_label, 32, "P1: PufferAI");

            if (human_override) {
                // --- SHIFT held: human controls P1 via c_render ---
                env.mouse_action = -1;
                c_render(&env);

                if (env.mouse_action >= 0 && env.mouse_action < NUM_ACTIONS) {
                    env.actions[0] = env.mouse_action;
                    env.mouse_action = -1;
                    c_step(&env);

                    if (env.terminals[0]) {
                        if (env.last_result > 0) watch_wins++;
                        else if (env.last_result < 0) watch_losses++;
                        else watch_draws++;

                        client->show_result = 1;
                        if (env.last_result > 0)
                            snprintf(client->result_text, 64, "PufferAI wins!");
                        else if (env.last_result < 0)
                            snprintf(client->result_text, 64, "%s wins!", bot_names[watch_bot]);
                        else
                            snprintf(client->result_text, 64, "Draw!");
                        log_add(client, client->result_text, (Color){0xFF, 0xCC, 0x00, 0xFF});
                        waiting_for_restart = 1;
                    }
                    frame_counter = 0;
                }
            } else {
                // --- Normal watch: auto-step + custom panel ---

                // Auto-step (only when game is active)
                if (!waiting_for_restart) {
                    frame_counter++;
                    if (frame_counter >= speed_delays[speed]) {
                        frame_counter = 0;

                        int action = 0;
                        forward_poke_battle_net(net_p1, env.observations, &action);
                        env.actions[0] = action;
                        c_step(&env);
                        update_battle_log(client, &env);

                        if (env.terminals[0]) {
                            if (env.last_result > 0) watch_wins++;
                            else if (env.last_result < 0) watch_losses++;
                            else watch_draws++;

                            client->show_result = 1;
                            if (env.last_result > 0)
                                snprintf(client->result_text, 64, "PufferAI wins!");
                            else if (env.last_result < 0)
                                snprintf(client->result_text, 64, "%s wins!", bot_names[watch_bot]);
                            else
                                snprintf(client->result_text, 64, "Draw!");
                            log_add(client, client->result_text, (Color){0xFF, 0xCC, 0x00, 0xFF});
                            waiting_for_restart = 1;
                        }
                    }
                }

                // --- Collect hover state for buttons ---
                int h_slower = 0, h_faster = 0;
                int h_opp[3] = {0, 0, 0};

                // Speed button rects
                int sy = PANEL_Y + 50;
                Rectangle r_slower = {100, (float)sy, 40, 28};
                Rectangle r_faster = {260, (float)sy, 40, 28};
                h_slower = CheckCollisionPointRec(mouse, r_slower);
                h_faster = CheckCollisionPointRec(mouse, r_faster);

                // Opponent button rects
                int opp_y = PANEL_Y + 100;
                int bw = 130, bh = 36, gap = 12, bx0 = 130;
                Rectangle r_opp[3];
                for (int i = 0; i < 3; i++) {
                    r_opp[i] = (Rectangle){(float)(bx0 + i * (bw + gap)), (float)opp_y, (float)bw, (float)bh};
                    h_opp[i] = CheckCollisionPointRec(mouse, r_opp[i]);
                }

                // --- Draw ---
                BeginDrawing();
                ClearBackground(CLR_PANEL_BG);
                draw_battle_field(client, &env);
                draw_result_overlay(client);

                // Watch control panel
                DrawRectangle(0, PANEL_Y, BATTLE_W, WIN_H - PANEL_Y, CLR_PANEL_BG);

                // Row 1: matchup + record
                {
                    char buf[128];
                    snprintf(buf, 128, "PufferAI vs %s", bot_names[watch_bot]);
                    DrawText(buf, 20, PANEL_Y + 12, 18, WHITE);
                    snprintf(buf, 128, "W: %d   L: %d   D: %d",
                             watch_wins, watch_losses, watch_draws);
                    DrawText(buf, 380, PANEL_Y + 14, 16, (Color){0xBB, 0xBB, 0xBB, 0xFF});
                }

                // Row 2: speed controls
                {
                    DrawText("Speed:", 20, sy + 6, 16, (Color){0xBB, 0xBB, 0xBB, 0xFF});

                    // [<<]
                    DrawRectangleRounded(r_slower, 0.3f, 4,
                        h_slower ? (Color){0x55, 0x66, 0x77, 0xFF} : (Color){0x44, 0x55, 0x66, 0xFF});
                    int tw_s = MeasureText("<<", 16);
                    DrawText("<<", 100 + (40 - tw_s) / 2, sy + 6, 16, WHITE);

                    // Speed name
                    int nw = MeasureText(speed_names[speed], 16);
                    DrawText(speed_names[speed], 160 + (80 - nw) / 2, sy + 6, 16, WHITE);

                    // [>>]
                    DrawRectangleRounded(r_faster, 0.3f, 4,
                        h_faster ? (Color){0x55, 0x66, 0x77, 0xFF} : (Color){0x44, 0x55, 0x66, 0xFF});
                    int tw_f = MeasureText(">>", 16);
                    DrawText(">>", 260 + (40 - tw_f) / 2, sy + 6, 16, WHITE);
                }

                // Row 3: opponent buttons
                {
                    DrawText("Play vs:", 20, opp_y + 10, 16, (Color){0xBB, 0xBB, 0xBB, 0xFF});

                    const char* opp_names[] = {"PufferAI", "Foul Play", "Random"};
                    const Color opp_colors[] = {
                        {0x44, 0x88, 0xCC, 0xFF},
                        {0x88, 0x44, 0xCC, 0xFF},
                        {0x44, 0xAA, 0x66, 0xFF},
                    };

                    for (int i = 0; i < 3; i++) {
                        Color bg = opp_colors[i];
                        if (h_opp[i]) {
                            bg.r = (unsigned char)(bg.r + 30 > 255 ? 255 : bg.r + 30);
                            bg.g = (unsigned char)(bg.g + 30 > 255 ? 255 : bg.g + 30);
                            bg.b = (unsigned char)(bg.b + 30 > 255 ? 255 : bg.b + 30);
                        }
                        DrawRectangleRounded(r_opp[i], 0.3f, 4, bg);
                        if (h_opp[i])
                            DrawRectangleRoundedLinesEx(r_opp[i], 0.3f, 4, 2, WHITE);

                        int tw = MeasureText(opp_names[i], 16);
                        int bx = bx0 + i * (bw + gap);
                        DrawText(opp_names[i], bx + (bw - tw) / 2, opp_y + 10, 16, WHITE);
                    }
                }

                // Row 4: hint text
                if (waiting_for_restart) {
                    DrawText("Click to watch another, or pick an opponent!",
                             20, PANEL_Y + 160, 14, (Color){0xBB, 0xBB, 0xBB, 0xFF});
                } else {
                    DrawText("Hold SHIFT to take over P1",
                             20, PANEL_Y + 160, 14, (Color){0x88, 0x99, 0xAA, 0xFF});
                }

                draw_battle_log(client, LOG_X, 0, LOG_W, WIN_H);

                // --- Handle clicks (must be BEFORE EndDrawing so the poll
                //     clears the pressed state before the next frame's c_render) ---
                if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                    int handled = 0;

                    // Opponent buttons → switch to play mode
                    for (int i = 0; i < 3; i++) {
                        if (h_opp[i]) {
                            play_opponent = i;
                            mode = 1;
                            waiting_for_restart = 0;
                            client->show_result = 0;
                            if (play_opponent == 0) {
                                env.num_agents = 2;
                            } else {
                                env.num_agents = 1;
                                env.bot_mode = (play_opponent == 1) ? BOT_MCTS : BOT_RANDOM;
                            }
                            c_reset(&env);
                            reset_lstm_state(net_p2);
                            reset_client_state(client, &env);
                            play_wins = 0;
                            play_losses = 0;
                            play_draws = 0;
                            frame_counter = 0;
                            handled = 1;
                            break;
                        }
                    }

                    // Speed buttons
                    if (!handled && h_slower && speed > 0) {
                        speed--;
                        handled = 1;
                    }
                    if (!handled && h_faster && speed < 3) {
                        speed++;
                        handled = 1;
                    }

                    // Click anywhere else → restart watch (only during result)
                    if (!handled && waiting_for_restart) {
                        waiting_for_restart = 0;
                        client->show_result = 0;
                        watch_bot = rand() % 3;
                        env.num_agents = 1;
                        env.bot_mode = watch_bot;
                        c_reset(&env);
                        reset_lstm_state(net_p1);
                        reset_client_state(client, &env);
                        frame_counter = 0;
                    }
                }

                EndDrawing();
            }

        } else {
            // ============================================================
            // PLAY MODE: Human (P1) vs selected opponent (P2)
            // ============================================================

            const char* opp_label = play_opponent == 0 ? "PufferAI" :
                                    play_opponent == 1 ? "Foul Play" : "Random";
            snprintf(client->p1_label, 32, "P1: You");
            snprintf(client->p2_label, 32, "P2: %s", opp_label);

            if (waiting_for_restart) {
                c_render(&env);
                if (env.mouse_action == -2) {
                    // Back to watch mode
                    mode = 0;
                    waiting_for_restart = 0;
                    env.mouse_action = -1;
                    client->show_result = 0;
                    watch_bot = rand() % 3;
                    env.num_agents = 1;
                    env.bot_mode = watch_bot;
                    c_reset(&env);
                    reset_lstm_state(net_p1);
                    reset_client_state(client, &env);
                    frame_counter = 0;
                }
                continue;
            }

            env.mouse_action = -1;
            c_render(&env);

            if (env.mouse_action >= 0 && env.mouse_action < NUM_ACTIONS) {
                env.actions[0] = env.mouse_action;
                env.mouse_action = -1;

                // PufferAI opponent: neural net provides P2 action
                if (play_opponent == 0) {
                    int p2_action = 0;
                    forward_poke_battle_net(net_p2, &obs[OBS_SIZE], &p2_action);
                    env.actions[1] = p2_action;
                }

                c_step(&env);

                // Auto-resolve P2-only forced switches for PufferAI opponent
                if (play_opponent == 0) {
                    while (env.battle.mode == 2 && !env.terminals[0]) {
                        update_battle_log(client, &env);
                        int p2_action = 0;
                        forward_poke_battle_net(net_p2, &obs[OBS_SIZE], &p2_action);
                        env.actions[0] = 0;
                        env.actions[1] = p2_action;
                        c_step(&env);
                    }
                }

                if (env.terminals[0]) {
                    if (env.last_result > 0) play_wins++;
                    else if (env.last_result < 0) play_losses++;
                    else play_draws++;
                    waiting_for_restart = 1;
                }
            }
        }
    }

    printf("\nWatch Record: %d W / %d L / %d D\n", watch_wins, watch_losses, watch_draws);
    printf("Play Record:  %d W / %d L / %d D\n", play_wins, play_losses, play_draws);
    c_close(&env);
    free_poke_battle_net(net_p1);
    free_poke_battle_net(net_p2);
    free(weights);
}

// ============================================================================
// Performance Test (headless)
// ============================================================================

void test_performance(int seconds) {
    PokeBattle env;
    memset(&env, 0, sizeof(PokeBattle));

    float obs[OBS_SIZE] = {0};
    int actions[1] = {0};
    float rewards[1] = {0};
    unsigned char terminals[1] = {0};

    env.observations = obs;
    env.actions = actions;
    env.rewards = rewards;
    env.terminals = terminals;
    env.num_agents = 1;
    env.selfplay = 0;
    env.bot_mode = BOT_HEURISTIC;
    env.seed = 42;

    init(&env);
    c_reset(&env);

    time_t start = time(NULL);
    long steps = 0;

    while (time(NULL) - start < seconds) {
        env.actions[0] = pb_rand_int(NUM_ACTIONS);
        c_step(&env);
        steps++;
    }

    time_t elapsed = time(NULL) - start;
    if (elapsed > 0) {
        printf("Performance: %ld steps in %ld seconds = %ld SPS\n",
               steps, elapsed, steps / elapsed);
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    srand((unsigned int)time(NULL));

    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        int seconds = 10;
        if (argc > 2) seconds = atoi(argv[2]);
        test_performance(seconds);
    } else {
        demo();
    }
    return 0;
}
