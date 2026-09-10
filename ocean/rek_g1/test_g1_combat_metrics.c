#include "g1_combat_metrics.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t checks;

#define CHECK(condition) do { \
    checks += 1u; \
    if (!(condition)) { \
        (void)fprintf( \
            stderr, "check failed at %s:%d: %s\n", \
            __FILE__, __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

static void near(float actual, float expected) {
    CHECK(isfinite(actual));
    CHECK(fabsf(actual - expected) <= 1e-6f);
}

static RekG1FightState active_round(void) {
    RekG1FightState fight;
    memset(&fight, 0, sizeof(fight));
    fight.phase = REK_G1_FIGHT_ROUND_ACTIVE;
    fight.current_round_number = 1u;
    fight.round_duration_seconds = 120.0f;
    fight.time_remaining_seconds = 120.0f;
    fight.round_result = REK_G1_ROUND_IN_PROGRESS;
    fight.round_winner_index = -1;
    fight.fight_result = REK_G1_FIGHT_IN_PROGRESS;
    fight.fight_winner_index = -1;
    return fight;
}

static void test_accumulator_persists_until_terminal(void) {
    RekG1CombatMetricAccumulator accumulator = {0};
    RekG1CombatMetricLog completed;
    memset(&completed, 0xA5, sizeof(completed));
    RekG1FightState fight = active_round();

    CHECK(rek_g1_combat_metrics_record_step(
        &accumulator, &fight, 2u, 3u, 0.02f, 0u, &completed));
    CHECK(completed.n == 0.0f);
    CHECK(accumulator.semantic_steps == 1u);
    CHECK(accumulator.scored_hits == 2u);
    CHECK(accumulator.attributed_contacts == 3u);

    memset(&completed, 0xA5, sizeof(completed));
    CHECK(rek_g1_combat_metrics_record_step(
        &accumulator, &fight, 1u, 4u, 0.02f, 0u, &completed));
    CHECK(completed.n == 0.0f);
    CHECK(accumulator.semantic_steps == 2u);
    CHECK(accumulator.scored_hits == 3u);
    CHECK(accumulator.attributed_contacts == 7u);
    near((float)accumulator.elapsed_seconds, 0.04f);
}

static void test_explicit_vector_reset_discards_partial_round(void) {
    RekG1CombatMetricAccumulator accumulator = {0};
    RekG1CombatMetricLog completed = {0};
    RekG1FightState fight = active_round();
    CHECK(rek_g1_combat_metrics_record_step(
        &accumulator, &fight, 3u, 5u, 0.02f, 0u, &completed));
    rek_g1_combat_metric_accumulator_reset(&accumulator);
    CHECK(accumulator.semantic_steps == 0u);
    CHECK(accumulator.scored_hits == 0u);
    CHECK(accumulator.attributed_contacts == 0u);
    CHECK(accumulator.elapsed_seconds == 0.0);
}

static void test_terminal_flush_keeps_hits_distinct_from_points(void) {
    RekG1CombatMetricAccumulator accumulator = {0};
    RekG1CombatMetricLog completed = {0};
    RekG1FightState fight = active_round();

    CHECK(rek_g1_combat_metrics_record_step(
        &accumulator, &fight, 1u, 2u, 0.02f, 0u, &completed));
    fight.phase = REK_G1_FIGHT_BETWEEN_ROUNDS;
    fight.round_result = REK_G1_ROUND_WON_BY_KO;
    fight.round_winner_index = 0;
    fight.knockout_occurred = 1u;
    fight.clean_hits[0] = 7;
    fight.clean_hits[1] = 2;
    fight.falls[0] = 1u;
    fight.falls[1] = 3u;
    CHECK(rek_g1_combat_metrics_record_step(
        &accumulator, &fight, 2u, 5u, 0.02f, 1u, &completed));

    near(completed.n, 1.0f);
    near(completed.side0_round_win_rate, 1.0f);
    near(completed.side1_round_win_rate, 0.0f);
    near(completed.ko_round_rate, 1.0f);
    near(completed.side0_points_per_round, 7.0f);
    near(completed.side1_points_per_round, 2.0f);
    near(completed.side0_falls_per_round, 1.0f);
    near(completed.side1_falls_per_round, 3.0f);
    near(completed.scored_hits_per_round, 3.0f);
    near(completed.attributed_contacts_per_round, 7.0f);
    near(completed.elapsed_seconds_per_round, 0.04f);
    near(completed.semantic_steps_per_round, 2.0f);
    CHECK(completed.scored_hits_per_round
        != completed.side0_points_per_round);
    CHECK(accumulator.semantic_steps == 0u);
    CHECK(accumulator.scored_hits == 0u);
    CHECK(accumulator.attributed_contacts == 0u);
    CHECK(accumulator.elapsed_seconds == 0.0);
}

static RekG1CombatMetricLog complete_one_step(
        RekG1FightState fight) {
    RekG1CombatMetricAccumulator accumulator = {0};
    RekG1CombatMetricLog completed = {0};
    CHECK(rek_g1_combat_metrics_record_step(
        &accumulator, &fight, 0u, 0u, 0.02f, 1u, &completed));
    return completed;
}

static void test_tie_result_redo_result_and_redo_round_are_separate(void) {
    RekG1FightState fight = active_round();
    fight.phase = REK_G1_FIGHT_BETWEEN_ROUNDS;
    fight.round_result = REK_G1_ROUND_TIE;
    RekG1CombatMetricLog tie = complete_one_step(fight);
    near(tie.round_tie_rate, 1.0f);
    near(tie.round_redo_result_rate, 0.0f);
    near(tie.redo_round_rate, 0.0f);

    fight.round_result = REK_G1_ROUND_WON_BY_POINTS;
    fight.round_winner_index = 1;
    fight.current_round_is_redo = 1u;
    RekG1CombatMetricLog completed_redo = complete_one_step(fight);
    near(completed_redo.side1_round_win_rate, 1.0f);
    near(completed_redo.round_tie_rate, 0.0f);
    near(completed_redo.round_redo_result_rate, 0.0f);
    near(completed_redo.redo_round_rate, 1.0f);

    fight.round_result = REK_G1_ROUND_REDO;
    fight.round_winner_index = -1;
    RekG1CombatMetricLog redo_result = complete_one_step(fight);
    near(redo_result.side0_round_win_rate, 0.0f);
    near(redo_result.side1_round_win_rate, 0.0f);
    near(redo_result.round_tie_rate, 0.0f);
    near(redo_result.round_redo_result_rate, 1.0f);
    near(redo_result.redo_round_rate, 1.0f);
}

static void test_double_ko_is_a_ko_tie(void) {
    RekG1FightState fight = active_round();
    fight.phase = REK_G1_FIGHT_BETWEEN_ROUNDS;
    fight.round_result = REK_G1_ROUND_TIE;
    fight.knockout_occurred = 1u;
    RekG1CombatMetricLog completed = complete_one_step(fight);
    near(completed.round_tie_rate, 1.0f);
    near(completed.ko_round_rate, 1.0f);
    near(completed.side0_round_win_rate, 0.0f);
    near(completed.side1_round_win_rate, 0.0f);
}

static void test_multiple_arenas_merge_as_completed_round_sums(void) {
    RekG1FightState side0 = active_round();
    side0.phase = REK_G1_FIGHT_BETWEEN_ROUNDS;
    side0.round_result = REK_G1_ROUND_WON_BY_POINTS;
    side0.round_winner_index = 0;
    side0.clean_hits[0] = 4;
    side0.clean_hits[1] = 1;

    RekG1FightState side1 = side0;
    side1.round_winner_index = 1;
    side1.clean_hits[0] = 2;
    side1.clean_hits[1] = 6;

    RekG1CombatMetricLog first = complete_one_step(side0);
    RekG1CombatMetricLog second = complete_one_step(side1);
    RekG1CombatMetricLog aggregate = {0};
    CHECK(rek_g1_combat_metrics_merge_completed(&aggregate, &first));
    CHECK(rek_g1_combat_metrics_merge_completed(&aggregate, &second));
    near(aggregate.n, 2.0f);
    near(aggregate.side0_round_win_rate, 1.0f);
    near(aggregate.side1_round_win_rate, 1.0f);
    near(aggregate.side0_points_per_round, 6.0f);
    near(aggregate.side1_points_per_round, 7.0f);
}

int main(void) {
    test_accumulator_persists_until_terminal();
    test_explicit_vector_reset_discards_partial_round();
    test_terminal_flush_keeps_hits_distinct_from_points();
    test_tie_result_redo_result_and_redo_round_are_separate();
    test_double_ko_is_a_ko_tie();
    test_multiple_arenas_merge_as_completed_round_sums();
    (void)printf("g1 combat metric checks: %zu\n", checks);
    return 0;
}
