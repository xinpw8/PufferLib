#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "semantic_action.h"

static int assertions;

static void require(int condition, const char* name) {
    assertions += 1;
    if (!condition) {
        fprintf(stderr, "FAIL %s\n", name);
        exit(1);
    }
}

static void require_close(float actual, float expected, const char* name) {
    require(fabsf(actual - expected) <= 1e-6f, name);
}

static uint8_t held_code(uint8_t held) {
    uint8_t code = UINT8_MAX;
    require(rek_g1_semantic_encode_held(held, &code) == REK_G1_SEMANTIC_OK,
        "fixture_mask_encodes");
    return code;
}

int main(void) {
    const RekG1InputTiming timing_50_hz_half_second = {
        .elapsed_seconds = 0.02f,
        .yaw_ramp_seconds = 0.5f,
    };
    RekG1HeldInputState state = {0};
    float prior_yaw = 0.0f;

    for (int tick = 0; tick < 25; tick++) {
        RekG1InputDecision decision = rek_g1_apply_input_frame(
            &state,
            (RekG1InputFrame){
                .held = REK_G1_HELD_FORWARD | REK_G1_HELD_YAW_LEFT,
            },
            timing_50_hz_half_second,
            1,
            0);
        require(decision.status == REK_G1_INPUT_ACCEPTED,
            "half_second_ramp_tick_accepted");
        require(decision.forward == 1 && decision.strafe == 0,
            "translation_is_immediate_while_yaw_ramps");
        require(decision.desired_yaw == 1,
            "left_yaw_remains_desired_held_input");
        require(decision.yaw + 1e-7f >= prior_yaw,
            "left_yaw_ramp_is_monotonic");
        require_close(decision.yaw, (float)(tick + 1) / 25.0f,
            "half_second_ramp_value_matches_elapsed_ratio");
        prior_yaw = decision.yaw;
    }
    require_close(state.yaw_ramp, 1.0f,
        "half_second_at_50_hz_reaches_full_yaw");
    RekG1InputDecision decision = rek_g1_apply_input_frame(
        &state,
        (RekG1InputFrame){
            .held = REK_G1_HELD_FORWARD | REK_G1_HELD_YAW_LEFT,
        },
        timing_50_hz_half_second,
        1,
        0);
    require_close(decision.yaw, 1.0f, "yaw_ramp_clamps_at_one");

    decision = rek_g1_apply_input_frame(
        &state,
        (RekG1InputFrame){
            .held = REK_G1_HELD_FORWARD | REK_G1_HELD_YAW_RIGHT,
        },
        timing_50_hz_half_second,
        1,
        0);
    require(decision.desired_yaw == -1 && state.yaw_sign == -1,
        "sign_reversal_selects_new_desired_sign");
    require_close(decision.yaw_ramp, 0.04f,
        "sign_reversal_resets_before_same_tick_accumulation");
    require_close(decision.yaw, -0.04f,
        "sign_reversal_does_not_reuse_old_ramp_magnitude");
    decision = rek_g1_apply_input_frame(
        &state,
        (RekG1InputFrame){
            .held = REK_G1_HELD_FORWARD | REK_G1_HELD_YAW_RIGHT,
        },
        timing_50_hz_half_second,
        1,
        0);
    require_close(decision.yaw, -0.08f,
        "reversed_sign_accumulates_monotonically");

    decision = rek_g1_apply_input_frame(
        &state,
        (RekG1InputFrame){.held = REK_G1_HELD_FORWARD},
        timing_50_hz_half_second,
        1,
        0);
    require(decision.desired_yaw == 0 && state.yaw_sign == 0,
        "yaw_release_clears_desired_sign");
    require_close(decision.yaw_ramp, 0.0f,
        "yaw_release_resets_ramp_magnitude");
    require_close(decision.yaw, 0.0f, "yaw_release_outputs_zero");
    require(decision.forward == 1, "yaw_release_does_not_delay_forward_hold");

    const uint8_t cardinal_masks[] = {
        REK_G1_HELD_FORWARD,
        REK_G1_HELD_BACKWARD,
        REK_G1_HELD_STRAFE_LEFT,
        REK_G1_HELD_STRAFE_RIGHT,
    };
    const int8_t expected_forward[] = {1, -1, 0, 0};
    const int8_t expected_strafe[] = {0, 0, 1, -1};
    for (int item = 0; item < 4; item++) {
        state = (RekG1HeldInputState){0};
        decision = rek_g1_apply_input_frame(
            &state,
            (RekG1InputFrame){
                .held = cardinal_masks[item] | REK_G1_HELD_YAW_LEFT,
            },
            timing_50_hz_half_second,
            1,
            0);
        require(decision.forward == expected_forward[item] &&
                decision.strafe == expected_strafe[item],
            "every_cardinal_axis_is_immediate_with_yaw");
        require_close(decision.yaw, 0.04f,
            "every_cardinal_axis_overlaps_ramping_yaw");
    }

    RekG1SemanticScheduler scheduler;
    rek_g1_semantic_reset(&scheduler);
    RekG1SemanticCommand command = {
        .kind = REK_G1_SEMANTIC_LOCOMOTION,
        .held_code = held_code(REK_G1_HELD_YAW_LEFT),
        .duration_ticks = 10,
        .kick_registry_index = REK_G1_SEMANTIC_KICK_NONE,
    };
    require(rek_g1_semantic_start(&scheduler, command, 4) ==
        REK_G1_SEMANTIC_OK, "pre_kick_yaw_segment_starts");
    for (int tick = 0; tick < 10; tick++) {
        RekG1SemanticTick result = rek_g1_semantic_tick(
            &scheduler, timing_50_hz_half_second, 1, 0);
        require(result.status == REK_G1_SEMANTIC_OK,
            "pre_kick_yaw_tick_accepted");
    }
    require_close(scheduler.input_state.yaw_ramp, 0.4f,
        "pre_kick_ramp_progress_is_known");

    command = (RekG1SemanticCommand){
        .kind = REK_G1_SEMANTIC_KICK,
        .held_code = held_code(REK_G1_HELD_YAW_LEFT),
        .duration_ticks = 3,
        .kick_registry_index = 0,
    };
    require(rek_g1_semantic_start(&scheduler, command, 4) ==
        REK_G1_SEMANTIC_OK, "yaw_kick_segment_starts");
    for (int tick = 0; tick < 3; tick++) {
        RekG1SemanticTick result = rek_g1_semantic_tick(
            &scheduler, timing_50_hz_half_second, 1, tick > 0);
        require(result.kick_active, "accepted_kick_stays_active");
        require(result.input.desired_yaw == 1,
            "accepted_kick_retains_desired_yaw");
        require_close(result.input.yaw, 0.0f,
            "accepted_kick_suppresses_effective_yaw");
        require(result.input.yaw_suppressed_for_attack,
            "accepted_kick_reports_yaw_suppression");
    }
    require_close(scheduler.input_state.yaw_ramp, 0.52f,
        "accepted_kick_preserves_and_advances_yaw_ramp_state");

    command = (RekG1SemanticCommand){
        .kind = REK_G1_SEMANTIC_LOCOMOTION,
        .held_code = held_code(REK_G1_HELD_YAW_LEFT),
        .duration_ticks = 1,
        .kick_registry_index = REK_G1_SEMANTIC_KICK_NONE,
    };
    require(rek_g1_semantic_start(&scheduler, command, 4) ==
        REK_G1_SEMANTIC_OK, "post_kick_yaw_segment_starts");
    RekG1SemanticTick post_kick = rek_g1_semantic_tick(
        &scheduler, timing_50_hz_half_second, 1, 0);
    require_close(post_kick.input.yaw, 0.56f,
        "post_kick_yaw_resumes_from_preserved_ramp_state");

    state = (RekG1HeldInputState){0};
    decision = rek_g1_apply_input_frame(
        &state,
        (RekG1InputFrame){
            .held = REK_G1_HELD_FORWARD | REK_G1_HELD_YAW_LEFT,
            .attack_edge = 1,
        },
        timing_50_hz_half_second,
        1,
        0);
    require(decision.attack_gate == REK_G1_ATTACK_BLOCKED_TRANSLATION_HELD,
        "translation_blocks_attack_edge");
    require(!decision.yaw_suppressed_for_attack,
        "blocked_attack_does_not_suppress_yaw");
    require_close(decision.yaw, 0.04f,
        "blocked_attack_keeps_normal_yaw_ramp_output");
    require(decision.blocked_attack_retention_unknown,
        "blocked_attack_retention_remains_explicitly_unknown");

    state = (RekG1HeldInputState){0};
    decision = rek_g1_apply_input_frame(
        &state,
        (RekG1InputFrame){.held = REK_G1_HELD_YAW_LEFT},
        timing_50_hz_half_second,
        1,
        0);
    require_close(decision.yaw, 0.04f,
        "invalid_parameter_fixture_establishes_state");
    const RekG1HeldInputState before_invalid = state;
    const RekG1InputTiming invalid_timings[] = {
        {.elapsed_seconds = 0.0f, .yaw_ramp_seconds = 0.5f},
        {.elapsed_seconds = -0.02f, .yaw_ramp_seconds = 0.5f},
        {.elapsed_seconds = NAN, .yaw_ramp_seconds = 0.5f},
        {.elapsed_seconds = INFINITY, .yaw_ramp_seconds = 0.5f},
        {.elapsed_seconds = 0.02f, .yaw_ramp_seconds = 0.0f},
        {.elapsed_seconds = 0.02f, .yaw_ramp_seconds = -0.5f},
        {.elapsed_seconds = 0.02f, .yaw_ramp_seconds = NAN},
        {.elapsed_seconds = 0.02f, .yaw_ramp_seconds = INFINITY},
    };
    for (int item = 0; item < 8; item++) {
        decision = rek_g1_apply_input_frame(
            &state,
            (RekG1InputFrame){.held = REK_G1_HELD_YAW_RIGHT},
            invalid_timings[item],
            1,
            0);
        RekG1InputStatus expected = item < 4 ?
            REK_G1_INPUT_REJECTED_INVALID_ELAPSED_SECONDS :
            REK_G1_INPUT_REJECTED_INVALID_YAW_RAMP_SECONDS;
        require(decision.status == expected,
            "invalid_or_nonfinite_timing_is_rejected");
        require(state.held == before_invalid.held &&
                state.yaw_sign == before_invalid.yaw_sign,
            "invalid_timing_does_not_mutate_discrete_state");
        require_close(state.yaw_ramp, before_invalid.yaw_ramp,
            "invalid_timing_does_not_mutate_ramp_state");
    }

    rek_g1_semantic_reset(&scheduler);
    command = (RekG1SemanticCommand){
        .kind = REK_G1_SEMANTIC_LOCOMOTION,
        .held_code = held_code(REK_G1_HELD_YAW_LEFT),
        .duration_ticks = 2,
        .kick_registry_index = REK_G1_SEMANTIC_KICK_NONE,
    };
    require(rek_g1_semantic_start(&scheduler, command, 4) ==
        REK_G1_SEMANTIC_OK, "invalid_timing_segment_starts");
    RekG1SemanticTick invalid_tick = rek_g1_semantic_tick(
        &scheduler,
        (RekG1InputTiming){
            .elapsed_seconds = 0.0f,
            .yaw_ramp_seconds = 0.5f,
        },
        1,
        0);
    require(invalid_tick.status == REK_G1_SEMANTIC_INVALID_INPUT_TIMING,
        "semantic_scheduler_rejects_invalid_timing");
    require(scheduler.active && scheduler.first_tick &&
            scheduler.remaining_ticks == 2,
        "invalid_timing_does_not_consume_semantic_tick");

    printf(
        "PASS g1_yaw_ramp assertions=%d tick_hz=50 ramp_seconds=0.5\n",
        assertions);
    return 0;
}
