#pragma once

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "g1_fight_state.h"

/*
 * One accumulator belongs to one two-fighter arena. Tick-only contact counts
 * are retained until that arena emits its round terminal. The completed log is
 * then published once, from the arena's side-0 row, so StaticVec's averaging
 * denominator counts rounds rather than fighter rows.
 */
typedef struct RekG1CombatMetricAccumulator {
    uint64_t semantic_steps;
    uint64_t scored_hits;
    uint64_t attributed_contacts;
    double elapsed_seconds;
} RekG1CombatMetricAccumulator;

/*
 * Every field is a sum over completed rounds until StaticVec consumes it.
 * StaticVec divides each field by n, so the exported names describe averages
 * or rates. Keep this structure float-only for StaticVec log aggregation.
 */
typedef struct RekG1CombatMetricLog {
    float side0_round_win_rate;
    float side1_round_win_rate;
    float round_tie_rate;
    float round_redo_result_rate;
    float redo_round_rate;
    float ko_round_rate;
    float side0_points_per_round;
    float side1_points_per_round;
    float side0_falls_per_round;
    float side1_falls_per_round;
    float scored_hits_per_round;
    float attributed_contacts_per_round;
    float elapsed_seconds_per_round;
    float semantic_steps_per_round;
    float n;
} RekG1CombatMetricLog;

enum { REK_G1_COMBAT_METRIC_LOG_FLOATS = 15 };

_Static_assert(
    sizeof(RekG1CombatMetricLog)
        == REK_G1_COMBAT_METRIC_LOG_FLOATS * sizeof(float),
    "combat metric logs must remain a packed float array");

static inline void rek_g1_combat_metric_accumulator_reset(
        RekG1CombatMetricAccumulator* accumulator) {
    if (accumulator != NULL) memset(accumulator, 0, sizeof(*accumulator));
}

static inline int rek_g1_combat_metric_u64_add(
        uint64_t* target,
        uint64_t value) {
    if (target == NULL || *target > UINT64_MAX - value) return 0;
    *target += value;
    return 1;
}

static inline int rek_g1_combat_metric_float_from_u64(
        uint64_t value,
        float* output) {
    if (output == NULL || (double)value > (double)FLT_MAX) return 0;
    *output = (float)value;
    return isfinite(*output);
}

static inline int rek_g1_combat_metric_terminal_result_valid(
        const RekG1FightState* fight) {
    if (fight == NULL
            || (fight->current_round_is_redo != 0u
                && fight->current_round_is_redo != 1u)
            || (fight->knockout_occurred != 0u
                && fight->knockout_occurred != 1u)
            || fight->clean_hits[0] < 0
            || fight->clean_hits[1] < 0) {
        return 0;
    }
    if (fight->round_result == REK_G1_ROUND_WON_BY_POINTS
            || fight->round_result == REK_G1_ROUND_WON_BY_KO) {
        return fight->round_winner_index == 0
            || fight->round_winner_index == 1;
    }
    if (fight->round_result == REK_G1_ROUND_TIE
            || fight->round_result == REK_G1_ROUND_REDO) {
        return fight->round_winner_index == -1;
    }
    return 0;
}

/*
 * Records one successful 50 Hz semantic step. scored_hits is the number of
 * contacts whose complete scoring gate accepted, not awarded points. The
 * attribution count is independently measured and may differ. On a terminal,
 * completed_out receives one round and the accumulator starts empty again.
 */
static inline int rek_g1_combat_metrics_record_step(
        RekG1CombatMetricAccumulator* accumulator,
        const RekG1FightState* fight,
        uint32_t scored_hits,
        uint32_t attributed_contacts,
        float elapsed_seconds,
        uint8_t terminal,
        RekG1CombatMetricLog* completed_out) {
    if (accumulator == NULL || fight == NULL || completed_out == NULL
            || (terminal != 0u && terminal != 1u)
            || !isfinite(elapsed_seconds) || elapsed_seconds < 0.0f
            || accumulator->semantic_steps == UINT64_MAX
            || !rek_g1_combat_metric_u64_add(
                &accumulator->scored_hits, scored_hits)
            || !rek_g1_combat_metric_u64_add(
                &accumulator->attributed_contacts, attributed_contacts)) {
        return 0;
    }
    accumulator->semantic_steps += 1u;
    accumulator->elapsed_seconds += (double)elapsed_seconds;
    if (!isfinite(accumulator->elapsed_seconds)
            || accumulator->elapsed_seconds > (double)FLT_MAX) {
        return 0;
    }

    memset(completed_out, 0, sizeof(*completed_out));
    if (!terminal) return 1;
    if (!rek_g1_combat_metric_terminal_result_valid(fight)
            || !rek_g1_combat_metric_float_from_u64(
                accumulator->scored_hits,
                &completed_out->scored_hits_per_round)
            || !rek_g1_combat_metric_float_from_u64(
                accumulator->attributed_contacts,
                &completed_out->attributed_contacts_per_round)
            || !rek_g1_combat_metric_float_from_u64(
                accumulator->semantic_steps,
                &completed_out->semantic_steps_per_round)) {
        return 0;
    }

    completed_out->side0_round_win_rate =
        fight->round_winner_index == 0 ? 1.0f : 0.0f;
    completed_out->side1_round_win_rate =
        fight->round_winner_index == 1 ? 1.0f : 0.0f;
    completed_out->round_tie_rate =
        fight->round_result == REK_G1_ROUND_TIE ? 1.0f : 0.0f;
    completed_out->round_redo_result_rate =
        fight->round_result == REK_G1_ROUND_REDO ? 1.0f : 0.0f;
    completed_out->redo_round_rate =
        fight->current_round_is_redo ? 1.0f : 0.0f;
    completed_out->ko_round_rate = fight->knockout_occurred ? 1.0f : 0.0f;
    completed_out->side0_points_per_round = (float)fight->clean_hits[0];
    completed_out->side1_points_per_round = (float)fight->clean_hits[1];
    completed_out->side0_falls_per_round = (float)fight->falls[0];
    completed_out->side1_falls_per_round = (float)fight->falls[1];
    completed_out->elapsed_seconds_per_round =
        (float)accumulator->elapsed_seconds;
    completed_out->n = 1.0f;
    rek_g1_combat_metric_accumulator_reset(accumulator);
    return 1;
}

static inline int rek_g1_combat_metric_add_float(
        float* target,
        float value) {
    if (target == NULL || !isfinite(*target) || !isfinite(value)) return 0;
    const float sum = *target + value;
    if (!isfinite(sum)) return 0;
    *target = sum;
    return 1;
}

static inline int rek_g1_combat_metrics_merge_completed(
        RekG1CombatMetricLog* target,
        const RekG1CombatMetricLog* completed) {
    if (target == NULL || completed == NULL || completed->n != 1.0f) return 0;
    const size_t field_count = sizeof(*target) / sizeof(float);
    float* target_fields = (float*)target;
    const float* completed_fields = (const float*)completed;
    for (size_t field = 0u; field < field_count; field++) {
        if (!rek_g1_combat_metric_add_float(
                &target_fields[field], completed_fields[field])) {
            return 0;
        }
    }
    return 1;
}
