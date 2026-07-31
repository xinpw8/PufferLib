#include <stdint.h>
#include <stdio.h>

#include "ocean/osrs_inferno/replay_best.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond) do { \
    if (!(cond)) { \
        printf("FAIL at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define RUN(name) do { \
    printf("  %-50s", #name); \
    name(); \
    printf("PASS\n"); \
    tests_passed++; \
} while(0)

static void test_full_run_prefers_higher_wave_then_fewer_ticks_then_lower_seed(void) {
    InfernoReplayBest best = inferno_replay_best_initial();
    ASSERT(inferno_replay_is_better(&best, 0, 10, 500, 1200, 9));
    inferno_replay_best_apply(&best, 10, 500, 1200, 9);

    ASSERT(inferno_replay_is_better(&best, 0, 11, 900, 1200, 8));
    ASSERT(inferno_replay_is_better(&best, 0, 10, 400, 1200, 8));
    ASSERT(inferno_replay_is_better(&best, 0, 10, 500, 1200, 8));
    ASSERT(!inferno_replay_is_better(&best, 0, 10, 500, 1200, 10));
}

static void test_partial_run_prefers_lower_zuk_hp_then_kill_ticks_then_lower_seed(void) {
    InfernoReplayBest best = inferno_replay_best_initial();
    inferno_replay_best_apply(&best, 68, 700, 300, 9);

    ASSERT(inferno_replay_is_better(&best, 68, 68, 900, 200, 8));
    ASSERT(inferno_replay_is_better(&best, 68, 68, 900, 300, 8));
    ASSERT(!inferno_replay_is_better(&best, 68, 68, 900, 600, 8));

    inferno_replay_best_apply(&best, 68, 700, 0, 9);
    ASSERT(inferno_replay_is_better(&best, 68, 68, 600, 0, 10));
    ASSERT(inferno_replay_is_better(&best, 68, 68, 700, 0, 8));
    ASSERT(!inferno_replay_is_better(&best, 68, 68, 700, 0, 10));
}

int main(void) {
    printf("inferno replay best tests\n\n");
    RUN(test_full_run_prefers_higher_wave_then_fewer_ticks_then_lower_seed);
    RUN(test_partial_run_prefers_lower_zuk_hp_then_kill_ticks_then_lower_seed);
    printf("%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
