#include <stdio.h>
#include <time.h>

#include "rek.h"
#include "puffernet.h"
#include "render.h"

// Defaults mirror config/rek.ini so the standalone binary behaves like a
// training env without parsing the ini.
static Rek make_env(int num_agents, int num_bots) {
    Rek env = {
        .num_agents = num_agents,
        .num_bots = num_bots,
        .round_frames = (int)(60.0f * REK_TICK_HZ),
        .arena_radius = 3.0f,
        .body_radius = 0.28f,
        .move_speed = 1.4f,
        .guard_speed_mult = 0.5f,
        .accel = 0.35f,
        .friction = 0.82f,
        .turn_rate = 0.18f,
        .balance_decay = 0.02f,
        .guard_balance_mult = 0.35f,
        .hitstun_frames = 6,
        .getup_frames = 45,
        .reward_hit = 0.1f,
        .reward_hit_taken = -0.1f,
        .reward_down = -0.3f,
        .reward_down_dealt = 0.3f,
        .reward_win = 1.0f,
        .reward_guard = 0.0f,
        .dr = 1.0f,
        .rng = 42,
    };
    return env;
}

// Throughput check. This is the number the >10M SPS target is measured
// against; multiply by the core count for the OMP-parallel figure.
static void performance_test(void) {
    const long test_seconds = 10;
    Rek env = make_env(2, 0);
    allocate_env(&env);
    c_reset(&env);

    long start = time(NULL);
    long steps = 0;
    while (time(NULL) - start < test_seconds) {
        for (int rep = 0; rep < 1000; rep++) {
            for (int s = 0; s < env.num_agents; s++) {
                float* a = env.actions + s * REK_NUM_ATNS;
                a[0] = (float)(rek_rand(&env.rng) % NUM_MOVE_DIRS);
                a[1] = (float)(rek_rand(&env.rng) % (uint32_t)NUM_MOVE_DEFS);
                a[2] = (float)(rek_rand(&env.rng) % 2u);
            }
            c_step(&env);
            steps++;
        }
    }
    long elapsed = time(NULL) - start;
    if (elapsed <= 0) elapsed = 1;
    printf("single-core SPS: %ld  (agent-steps/s: %ld)\n",
        steps / elapsed, steps * env.num_agents / elapsed);

    c_close(&env);
    free_allocated(&env);
}

// Human at slot 0 against the scripted bot. Mirrors REK's scheme: WASD walks,
// the number row fires set moves, shift guards.
static void demo(void) {
    Rek env = make_env(1, 1);
    allocate_env(&env);
    c_reset(&env);

    Weights* weights = NULL;
    PufferNet* net = NULL;
    FILE* f = fopen("resources/rek/rek_weights.bin", "rb");
    if (f != NULL) {
        fclose(f);
        weights = load_weights("resources/rek/rek_weights.bin");
        int logit_sizes[REK_NUM_ATNS] = {NUM_MOVE_DIRS, NUM_MOVE_DEFS, 2};
        net = make_puffernet(weights, env.num_agents, REK_OBS_SIZE, 128, 4,
            logit_sizes, REK_NUM_ATNS);
        printf("Loaded policy from resources/rek/rek_weights.bin\n");
    } else {
        printf("No checkpoint at resources/rek/rek_weights.bin — keyboard control.\n");
        printf("WASD moves, 1-%d fire set moves, LEFT SHIFT guards.\n", NUM_MOVE_DEFS - 1);
    }

    c_render(&env);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_ESCAPE)) break;

        if (net != NULL) {
            forward_puffernet(net, env.observations, env.actions);
        } else {
            int fwd = 0, side = 0;
            if (IsKeyDown(KEY_W)) fwd += 1;
            if (IsKeyDown(KEY_S)) fwd -= 1;
            if (IsKeyDown(KEY_D)) side += 1;
            if (IsKeyDown(KEY_A)) side -= 1;

            // Map the WASD pair onto the 9-way head: 0 neutral, then clockwise
            // from straight ahead, matching rek_move_dir's table.
            int dir = 0;
            if (fwd > 0 && side == 0) dir = 1;
            else if (fwd > 0 && side > 0) dir = 2;
            else if (fwd == 0 && side > 0) dir = 3;
            else if (fwd < 0 && side > 0) dir = 4;
            else if (fwd < 0 && side == 0) dir = 5;
            else if (fwd < 0 && side < 0) dir = 6;
            else if (fwd == 0 && side < 0) dir = 7;
            else if (fwd > 0 && side < 0) dir = 8;

            int move = 0;
            for (int m = 1; m < NUM_MOVE_DEFS && m <= 9; m++) {
                if (IsKeyPressed(KEY_ZERO + m)) { move = m; break; }
            }

            env.actions[0] = (float)dir;
            env.actions[1] = (float)move;
            env.actions[2] = IsKeyDown(KEY_LEFT_SHIFT) ? 1.0f : 0.0f;
        }

        c_step(&env);
        c_render(&env);
    }

    if (net != NULL) {
        free_puffernet(net);
        free(weights);
    }
    close_client(env.client);
    env.client = NULL;
    c_close(&env);
    free_allocated(&env);
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--bench") == 0) {
        performance_test();
        return 0;
    }
    demo();
    return 0;
}
