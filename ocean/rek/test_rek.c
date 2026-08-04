// Headless checks for the REK combat env: match rules, and the step rate the
// >10M SPS target is measured against. No raylib, no GPU — builds and runs
// anywhere, which is the point.
//
//   clang -O2 -I./src -I./ocean/rek ocean/rek/test_rek.c -lm -o test_rek && ./test_rek

#include <assert.h>
#include <stdio.h>
#include <time.h>

#include "rek.h"

static int failures = 0;

#define CHECK(cond, msg, ...) do { \
    if (!(cond)) { \
        printf("  FAIL: " msg "\n", ##__VA_ARGS__); \
        failures++; \
    } \
} while (0)

static Rek make_test_env(int num_agents, int num_bots) {
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
        .dr = 0.0f,   // deterministic for the rule checks
        .rng = 12345,
    };
    return env;
}

static void set_action(Rek* env, int slot, int dir, int move, int guard) {
    float* a = env->actions + slot * REK_NUM_ATNS;
    a[0] = (float)dir;
    a[1] = (float)move;
    a[2] = (float)guard;
}

// Put the two fighters at a fixed separation, facing each other, mid-round.
static void stage(Rek* env, float gap) {
    env->fighters[0].x = -gap * 0.5f; env->fighters[0].z = 0.0f; env->fighters[0].yaw = 0.0f;
    env->fighters[1].x =  gap * 0.5f; env->fighters[1].z = 0.0f; env->fighters[1].yaw = (float)M_PI;
    env->fighters[0].vx = env->fighters[0].vz = 0.0f;
    env->fighters[1].vx = env->fighters[1].vz = 0.0f;
}

static void test_clean_hit_scores(void) {
    printf("clean hit scores a point\n");
    Rek env = make_test_env(2, 0);
    allocate_env(&env);
    c_reset(&env);
    stage(&env, 0.75f);

    int before = env.fighters[0].hits;
    // Jab, then idle through its full envelope.
    set_action(&env, 0, 0, 1, 0);
    set_action(&env, 1, 0, 0, 0);
    c_step(&env);
    for (int i = 0; i < rek_move_total(1) + 2; i++) {
        set_action(&env, 0, 0, 0, 0);
        c_step(&env);
    }

    CHECK(env.fighters[0].hits == before + 1,
        "expected exactly 1 hit landed, got %d", env.fighters[0].hits);
    CHECK(env.fighters[0].moves_whiffed == 0,
        "connecting move counted as a whiff");
    CHECK(rek_score(&env.fighters[0]) == 1,
        "score should be 1 after one clean hit, got %d", rek_score(&env.fighters[0]));

    c_close(&env);
    free_allocated(&env);
}

static void test_out_of_range_whiffs(void) {
    printf("out-of-range move whiffs\n");
    Rek env = make_test_env(2, 0);
    allocate_env(&env);
    c_reset(&env);
    stage(&env, 2.5f);   // far outside jab reach

    set_action(&env, 0, 0, 1, 0);
    set_action(&env, 1, 0, 0, 0);
    c_step(&env);
    for (int i = 0; i < rek_move_total(1) + 2; i++) {
        set_action(&env, 0, 0, 0, 0);
        c_step(&env);
    }

    CHECK(env.fighters[0].hits == 0, "landed a hit at 2.5 m, got %d", env.fighters[0].hits);
    CHECK(env.fighters[0].moves_whiffed == 1,
        "whiff not recorded, got %d", env.fighters[0].moves_whiffed);

    c_close(&env);
    free_allocated(&env);
}

static void test_guard_blocks_score(void) {
    printf("guard blocks the scoreboard but not the balance pressure\n");
    Rek env = make_test_env(2, 0);
    allocate_env(&env);
    c_reset(&env);
    stage(&env, 0.75f);

    set_action(&env, 0, 0, 1, 0);
    set_action(&env, 1, 0, 0, 1);   // slot 1 guards
    c_step(&env);
    for (int i = 0; i < rek_move_total(1) + 2; i++) {
        set_action(&env, 0, 0, 0, 0);
        set_action(&env, 1, 0, 0, 1);
        c_step(&env);
    }

    CHECK(env.fighters[0].hits == 0,
        "guarded hit should not score, got %d", env.fighters[0].hits);

    c_close(&env);
    free_allocated(&env);
}

// Heaviest-hitting move in whatever table is compiled in. Derived rather than
// hardcoded so these checks survive tools/extract_rek.py replacing the roster.
static int heaviest_move(void) {
    int best = 1;
    for (int m = 1; m < NUM_MOVE_DEFS; m++) {
        if (REK_MOVE_TABLE[m].balance_impact > REK_MOVE_TABLE[best].balance_impact) best = m;
    }
    return best;
}

// Land one heavy hit from `attacker` on an opponent whose balance is already
// loaded, so the hit itself is what tips them over. Drives the real hit path
// instead of writing a down straight into the fighter state.
static void land_finishing_hit(Rek* env, int attacker) {
    int victim = 1 - attacker;
    int mv = heaviest_move();
    const MoveDef* m = &REK_MOVE_TABLE[mv];

    stage(env, m->reach);
    env->fighters[victim].down_timer = 0;
    env->fighters[victim].stun = 0;

    // Pre-load balance so the impact clears the threshold even after the
    // passive decay accrued over the move's startup frames.
    float decay_before_hit = env->balance_decay * (float)(m->startup + 2);
    float need = 1.0f - m->balance_impact + decay_before_hit + 0.02f;
    if (need > 0.99f) need = 0.99f;
    if (need < 0.0f) need = 0.0f;
    env->fighters[victim].balance = need;

    set_action(env, attacker, 0, mv, 0);
    set_action(env, victim, 0, 0, 0);
    c_step(env);
    for (int i = 0; i < rek_move_total(mv) + 2; i++) {
        if (*env->terminal_ptr[0] != 0.0f) return;   // match ended on this down
        set_action(env, attacker, 0, 0, 0);
        set_action(env, victim, 0, 0, 0);
        c_step(env);
    }
}

static void test_down_costs_the_faller_a_point(void) {
    printf("a down costs the faller a point\n");
    Rek env = make_test_env(2, 0);
    allocate_env(&env);
    c_reset(&env);

    int score_before = rek_score(&env.fighters[1]);
    land_finishing_hit(&env, 0);

    CHECK(env.fighters[1].downs == 1, "expected 1 down, got %d", env.fighters[1].downs);
    CHECK(rek_score(&env.fighters[1]) == score_before - 1,
        "down should cost the faller 1 point: %d -> %d",
        score_before, rek_score(&env.fighters[1]));
    CHECK(env.fighters[1].down_timer > 0, "downed fighter should be on the floor");
    CHECK(env.fighters[1].balance == 0.0f, "balance should reset on a down");

    c_close(&env);
    free_allocated(&env);
}

static void test_three_downs_ends_the_match(void) {
    printf("3 downs ends the match\n");
    Rek env = make_test_env(2, 0);
    allocate_env(&env);
    c_reset(&env);

    for (int d = 0; d < REK_DOWNS_TO_LOSE; d++) {
        land_finishing_hit(&env, 0);

        if (d < REK_DOWNS_TO_LOSE - 1) {
            CHECK(env.fighters[1].downs == d + 1,
                "expected %d down(s), got %d", d + 1, env.fighters[1].downs);
            CHECK(*env.terminal_ptr[0] == 0.0f,
                "match ended early after %d down(s)", d + 1);
        }
    }

    CHECK(*env.terminal_ptr[0] == 1.0f, "match did not terminate on the 3rd down");
    CHECK(*env.terminal_ptr[1] == 1.0f, "slot 1 terminal not set");
    // Slot 0 won, so its terminal reward is the positive win payout.
    CHECK(*env.reward_ptr[0] > 0.0f,
        "winner should get a positive terminal reward, got %f", *env.reward_ptr[0]);
    CHECK(*env.reward_ptr[1] < 0.0f,
        "loser should get a negative terminal reward, got %f", *env.reward_ptr[1]);
    // c_reset ran inside end_episode, so the round is fresh.
    CHECK(env.tick == 0, "tick should reset after the match, got %d", env.tick);
    CHECK(env.fighters[1].downs == 0, "downs should reset after the match");

    c_close(&env);
    free_allocated(&env);
}

static void test_round_clock_awards_most_hits(void) {
    printf("round clock awards the win to the most hits\n");
    Rek env = make_test_env(2, 0);
    allocate_env(&env);
    c_reset(&env);

    env.fighters[0].hits = 5;
    env.fighters[1].hits = 2;
    env.tick = env.round_frames - 1;
    set_action(&env, 0, 0, 0, 0);
    set_action(&env, 1, 0, 0, 0);
    c_step(&env);

    CHECK(*env.terminal_ptr[0] == 1.0f, "round did not end on the clock");
    CHECK(*env.reward_ptr[0] > 0.0f, "slot 0 led on hits and should win");
    CHECK(*env.reward_ptr[1] < 0.0f, "slot 1 trailed on hits and should lose");

    c_close(&env);
    free_allocated(&env);
}

static void test_obs_size_matches_layout(void) {
    printf("observation layout is self-consistent\n");
    CHECK(REK_OBS_SIZE == 2 * (REK_SCALARS_PER_FIGHTER + NUM_MOVE_DEFS)
            + REK_RELATIVE_FEATURES + REK_CLOCK_FEATURES,
        "REK_OBS_SIZE does not match its parts");

    Rek env = make_test_env(2, 0);
    allocate_env(&env);
    c_reset(&env);
    set_action(&env, 0, 1, 0, 0);
    set_action(&env, 1, 1, 0, 0);
    c_step(&env);

    // Every slot's observation must be finite — a NaN here poisons the rollout.
    for (int s = 0; s < env.num_agents; s++) {
        for (int i = 0; i < REK_OBS_SIZE; i++) {
            float v = env.obs_ptr[s][i];
            CHECK(v == v && v > -1e6f && v < 1e6f,
                "obs[%d][%d] not finite: %f", s, i, v);
        }
    }

    c_close(&env);
    free_allocated(&env);
}

static void test_bot_mode_runs(void) {
    printf("single-agent vs scripted bot runs a full round\n");
    Rek env = make_test_env(1, 1);
    allocate_env(&env);
    c_reset(&env);

    for (int i = 0; i < 4000; i++) {
        set_action(&env, 0, (int)(rek_rand(&env.rng) % NUM_MOVE_DIRS),
            (int)(rek_rand(&env.rng) % (uint32_t)NUM_MOVE_DEFS),
            (int)(rek_rand(&env.rng) % 2u));
        c_step(&env);
    }
    CHECK(env.log.n > 0.0f, "no episodes completed in 4000 steps");

    c_close(&env);
    free_allocated(&env);
}

static void benchmark(void) {
    const double test_seconds = 5.0;
    Rek env = make_test_env(2, 0);
    env.dr = 1.0f;
    allocate_env(&env);
    c_reset(&env);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    long steps = 0;
    double elapsed = 0.0;
    while (elapsed < test_seconds) {
        for (int rep = 0; rep < 10000; rep++) {
            for (int s = 0; s < env.num_agents; s++) {
                float* a = env.actions + s * REK_NUM_ATNS;
                a[0] = (float)(rek_rand(&env.rng) % NUM_MOVE_DIRS);
                a[1] = (float)(rek_rand(&env.rng) % (uint32_t)NUM_MOVE_DEFS);
                a[2] = (float)(rek_rand(&env.rng) % 2u);
            }
            c_step(&env);
            steps++;
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        elapsed = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
    }

    double sps = steps / elapsed;
    printf("\nsingle-core env steps/s : %.2f M\n", sps / 1e6);
    printf("single-core agent-steps/s: %.2f M\n", sps * env.num_agents / 1e6);
    printf("obs size: %d floats, move table: %d entries\n", REK_OBS_SIZE, NUM_MOVE_DEFS);

    c_close(&env);
    free_allocated(&env);
}

int main(int argc, char** argv) {
    printf("REK env checks\n\n");
    test_clean_hit_scores();
    test_out_of_range_whiffs();
    test_guard_blocks_score();
    test_down_costs_the_faller_a_point();
    test_three_downs_ends_the_match();
    test_round_clock_awards_most_hits();
    test_obs_size_matches_layout();
    test_bot_mode_runs();

    if (failures == 0) {
        printf("\nall checks passed\n");
    } else {
        printf("\n%d check(s) failed\n", failures);
    }

    if (argc > 1 && strcmp(argv[1], "--bench") == 0) {
        benchmark();
    }
    return failures == 0 ? 0 : 1;
}
