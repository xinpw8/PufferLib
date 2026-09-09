#include "g1_fall_state.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int assertions;

static void require(int condition, const char* name) {
    assertions += 1;
    if (!condition) {
        fprintf(stderr, "FAIL %s\n", name);
        exit(1);
    }
}

static uint32_t float_bits(float value) {
    uint32_t bits = 0u;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static RekG1FallSample upright_sample(void) {
    RekG1FallSample sample = {
        .tracking_active = 1u,
        .tilt_degrees = 0.0f,
        .pelvis_height_ratio = 1.0f,
        .both_feet_off_floor = 0u,
        .has_foot_body_contact = 1u,
        .distinct_nonfoot_body_contact_count = 0u,
        .fixed_delta_seconds = 0.02f,
        .can_get_up = 0u,
    };
    return sample;
}

static RekG1FallState initialized_state(void) {
    RekG1FallState state;
    require(rek_g1_fall_state_init(
        &REK_G1_FALL_CONFIG_F84F1874, &state) == REK_G1_FALL_OK,
        "state_init");
    return state;
}

static RekG1FallState falling_state(void) {
    RekG1FallState state = initialized_state();
    state.phase = REK_G1_FALL_FALLING;
    return state;
}

static uint32_t step_ok(
    RekG1FallState* state,
    const RekG1FallSample* sample
) {
    RekG1FallStepResult result;
    require(rek_g1_fall_state_step(
        &REK_G1_FALL_CONFIG_F84F1874, state, sample, &result)
        == REK_G1_FALL_OK, "step_ok");
    *state = result.next_state;
    return result.events;
}

static RekG1FallSample fast_fallen_sample(void) {
    RekG1FallSample sample = upright_sample();
    sample.tilt_degrees = nextafterf(
        REK_G1_FALL_CONFIG_F84F1874.fallen_tilt_degrees, INFINITY);
    sample.pelvis_height_ratio = nextafterf(
        REK_G1_FALL_CONFIG_F84F1874.fallen_height_ratio, -INFINITY);
    sample.both_feet_off_floor = 1u;
    sample.has_foot_body_contact = 0u;
    sample.distinct_nonfoot_body_contact_count = 3u;
    sample.fixed_delta_seconds =
        REK_G1_FALL_CONFIG_F84F1874.fallen_hold_fast_seconds;
    return sample;
}

static RekG1FallState become_fallen(uint8_t can_get_up) {
    RekG1FallState state = falling_state();
    RekG1FallSample sample = fast_fallen_sample();
    sample.can_get_up = can_get_up;
    require(step_ok(&state, &sample) == REK_G1_FALL_EVENT_BECAME_FALLEN,
        "become_fallen_event");
    require(state.phase == REK_G1_FALL_FALLEN, "become_fallen_phase");
    return state;
}

static void test_config(void) {
    const RekG1FallConfig* config = &REK_G1_FALL_CONFIG_F84F1874;
    require(strcmp(REK_G1_FALL_BUILD_FINGERPRINT,
        "f84f187491e3b5cd73493de379ed972c5580b60d63f33956e396e6dec28b1659")
        == 0, "build_fingerprint");
    require(float_bits(config->falling_tilt_degrees) == 0x42280000u,
        "falling_tilt_bits");
    require(float_bits(config->falling_height_ratio) == 0x3f19999au,
        "falling_height_bits");
    require(float_bits(config->fallen_tilt_degrees) == 0x428a999au,
        "fallen_tilt_bits");
    require(float_bits(config->fallen_height_ratio) == 0x3ecccccdu,
        "fallen_height_bits");
    require(config->fallen_contact_points == 3u, "fallen_contacts");
    require(float_bits(config->fallen_hold_fast_seconds) == 0x3e19999au,
        "fast_hold_bits");
    require(float_bits(config->fallen_hold_slow_seconds) == 0x3f000000u,
        "slow_hold_bits");
    require(float_bits(config->fallen_reset_timeout_seconds) == 0x40400000u,
        "reset_timeout_bits");
    require(float_bits(config->reset_grace_seconds) == 0x3f000000u,
        "reset_grace_bits");
    require(float_bits(config->fight_spawn_reset_grace_seconds)
        == 0x40000000u, "fight_spawn_reset_grace_bits");
}

static void test_upright_start_strictness(void) {
    RekG1FallState state = initialized_state();
    RekG1FallSample sample = upright_sample();

    sample.tilt_degrees = REK_G1_FALL_CONFIG_F84F1874.falling_tilt_degrees;
    require(step_ok(&state, &sample) == REK_G1_FALL_EVENT_NONE,
        "tilt_equal_does_not_start");
    require(state.phase == REK_G1_FALL_UPRIGHT, "tilt_equal_upright");

    sample.tilt_degrees = nextafterf(sample.tilt_degrees, INFINITY);
    require(step_ok(&state, &sample) == REK_G1_FALL_EVENT_FALLING_STARTED,
        "tilt_above_starts");
    require(state.phase == REK_G1_FALL_FALLING, "tilt_above_phase");
    require(state.fallen_hold_seconds == 0.0f,
        "start_tick_does_not_accrue_hold");

    state = initialized_state();
    sample = upright_sample();
    sample.both_feet_off_floor = 1u;
    sample.pelvis_height_ratio =
        REK_G1_FALL_CONFIG_F84F1874.falling_height_ratio;
    require(step_ok(&state, &sample) == REK_G1_FALL_EVENT_NONE,
        "height_equal_does_not_start");
    sample.pelvis_height_ratio = nextafterf(
        sample.pelvis_height_ratio, -INFINITY);
    require(step_ok(&state, &sample) == REK_G1_FALL_EVENT_FALLING_STARTED,
        "height_below_with_feet_off_starts");

    state = initialized_state();
    sample.both_feet_off_floor = 0u;
    require(step_ok(&state, &sample) == REK_G1_FALL_EVENT_NONE,
        "height_below_with_foot_contact_does_not_start");
}

static void test_falling_clear_strictness(void) {
    RekG1FallSample sample = upright_sample();
    RekG1FallState state = falling_state();
    state.fallen_hold_seconds = 0.25f;
    sample.tilt_degrees = REK_G1_FALL_CONFIG_F84F1874.falling_tilt_degrees;
    require(step_ok(&state, &sample) == REK_G1_FALL_EVENT_FALLING_CLEARED,
        "falling_clears_at_tilt_boundary");
    require(state.phase == REK_G1_FALL_UPRIGHT, "clear_phase");
    require(state.fallen_hold_seconds == 0.0f, "clear_resets_hold");

    state = falling_state();
    sample.tilt_degrees = nextafterf(
        REK_G1_FALL_CONFIG_F84F1874.falling_tilt_degrees, INFINITY);
    require(step_ok(&state, &sample) == REK_G1_FALL_EVENT_NONE,
        "tilt_above_does_not_clear");
    require(state.phase == REK_G1_FALL_FALLING, "tilt_above_remains_falling");

    state = falling_state();
    sample.tilt_degrees = 0.0f;
    sample.both_feet_off_floor = 1u;
    require(step_ok(&state, &sample) == REK_G1_FALL_EVENT_NONE,
        "feet_off_does_not_clear");
    require(state.phase == REK_G1_FALL_FALLING, "feet_off_remains_falling");
}

static void test_tracking_active_qualifiers(void) {
    RekG1FallSample sample = fast_fallen_sample();
    RekG1FallState state = falling_state();

    sample.fixed_delta_seconds = nextafterf(
        REK_G1_FALL_CONFIG_F84F1874.fallen_hold_fast_seconds, 0.0f);
    require(step_ok(&state, &sample) == REK_G1_FALL_EVENT_NONE,
        "fast_hold_just_below_not_fallen");
    require(state.phase == REK_G1_FALL_FALLING, "fast_below_phase");

    state = falling_state();
    sample.fixed_delta_seconds =
        REK_G1_FALL_CONFIG_F84F1874.fallen_hold_fast_seconds;
    require(step_ok(&state, &sample) == REK_G1_FALL_EVENT_BECAME_FALLEN,
        "fast_hold_equal_fallen");

    state = falling_state();
    sample = fast_fallen_sample();
    sample.has_foot_body_contact = 1u;
    sample.fixed_delta_seconds = 0.25f;
    require(step_ok(&state, &sample) == REK_G1_FALL_EVENT_NONE,
        "foot_contact_three_bodies_uses_slow_first_half");
    require(state.fallen_hold_seconds == 0.25f, "slow_first_half_timer");
    require(step_ok(&state, &sample) == REK_G1_FALL_EVENT_BECAME_FALLEN,
        "foot_contact_three_bodies_slow_boundary");

    state = falling_state();
    sample = fast_fallen_sample();
    sample.distinct_nonfoot_body_contact_count = 1u;
    sample.fixed_delta_seconds = 0.25f;
    require(step_ok(&state, &sample) == REK_G1_FALL_EVENT_NONE,
        "one_nonfoot_no_foot_uses_slow_first_half");
    require(step_ok(&state, &sample) == REK_G1_FALL_EVENT_BECAME_FALLEN,
        "one_nonfoot_no_foot_slow_boundary");

    state = falling_state();
    state.fallen_hold_seconds = 0.25f;
    sample = fast_fallen_sample();
    sample.distinct_nonfoot_body_contact_count = 0u;
    require(step_ok(&state, &sample) == REK_G1_FALL_EVENT_NONE,
        "tracking_active_no_contact_not_fallen");
    require(state.fallen_hold_seconds == 0.0f,
        "tracking_active_no_contact_resets_timer");
    require(state.phase == REK_G1_FALL_FALLING,
        "tracking_active_does_not_fall_through_to_tilt_only");

    state = falling_state();
    state.fallen_hold_seconds = 0.25f;
    sample = fast_fallen_sample();
    sample.pelvis_height_ratio =
        REK_G1_FALL_CONFIG_F84F1874.fallen_height_ratio;
    require(step_ok(&state, &sample) == REK_G1_FALL_EVENT_NONE,
        "fallen_height_equal_not_low");
    require(state.fallen_hold_seconds == 0.0f,
        "height_boundary_resets_timer");

    state = falling_state();
    state.fallen_hold_seconds = 0.25f;
    sample = fast_fallen_sample();
    sample.tilt_degrees = REK_G1_FALL_CONFIG_F84F1874.fallen_tilt_degrees;
    require(step_ok(&state, &sample) == REK_G1_FALL_EVENT_NONE,
        "fallen_tilt_equal_not_tilted");
    require(state.fallen_hold_seconds == 0.0f,
        "tilt_boundary_resets_timer");
}

static void test_tracking_inactive_qualifiers(void) {
    RekG1FallState state = falling_state();
    RekG1FallSample sample = upright_sample();
    sample.tracking_active = 0u;
    sample.tilt_degrees = REK_G1_FALL_CONFIG_F84F1874.fallen_tilt_degrees;
    sample.pelvis_height_ratio = 0.0f;
    sample.has_foot_body_contact = 0u;
    sample.distinct_nonfoot_body_contact_count = UINT32_MAX;
    sample.fixed_delta_seconds = 0.25f;
    state.fallen_hold_seconds = 0.25f;
    require(step_ok(&state, &sample) == REK_G1_FALL_EVENT_NONE,
        "inactive_tilt_equal_not_fallen");
    require(state.fallen_hold_seconds == 0.0f,
        "inactive_tilt_equal_resets_timer");

    sample.tilt_degrees = nextafterf(sample.tilt_degrees, INFINITY);
    require(step_ok(&state, &sample) == REK_G1_FALL_EVENT_NONE,
        "inactive_slow_first_half");
    require(step_ok(&state, &sample) == REK_G1_FALL_EVENT_BECAME_FALLEN,
        "inactive_slow_boundary");
}

static void test_recovery_and_reset_timeout(void) {
    RekG1FallSample sample = upright_sample();
    sample.fixed_delta_seconds = 1.0f;
    RekG1FallState state = become_fallen(0u);
    require(state.recovery_armed == 0u, "unarmed_sampled");
    require(step_ok(&state, &sample) == REK_G1_FALL_EVENT_NONE,
        "reset_timeout_second_one");
    require(step_ok(&state, &sample) == REK_G1_FALL_EVENT_NONE,
        "reset_timeout_second_two");
    require(step_ok(&state, &sample) == REK_G1_FALL_EVENT_RESET_TIMEOUT_DUE,
        "reset_timeout_exact_boundary");
    require(state.fallen_timer_seconds
        == REK_G1_FALL_CONFIG_F84F1874.fallen_reset_timeout_seconds,
        "reset_timeout_rearmed");
    require(state.fallen_elapsed_seconds == 3.0f, "fallen_elapsed_three");
    sample.fixed_delta_seconds = 3.0f;
    require(step_ok(&state, &sample) == REK_G1_FALL_EVENT_RESET_TIMEOUT_DUE,
        "unhandled_timeout_repeats_after_rearm");

    state = become_fallen(1u);
    require(state.recovery_armed == 1u, "armed_sampled");
    sample.fixed_delta_seconds = 3.0f;
    require(step_ok(&state, &sample) == REK_G1_FALL_EVENT_NONE,
        "armed_recovery_bypasses_timeout");
    require(state.fallen_timer_seconds
        == REK_G1_FALL_CONFIG_F84F1874.fallen_reset_timeout_seconds,
        "armed_timeout_unchanged");
    require(state.fallen_elapsed_seconds == 3.0f,
        "armed_fallen_elapsed_advances");
}

static void test_reset_grace(void) {
    RekG1FallState fallen = become_fallen(0u);
    RekG1FallState state;
    require(rek_g1_fall_state_apply_reset_after_fall(
        &REK_G1_FALL_CONFIG_F84F1874, &fallen, &state) == REK_G1_FALL_OK,
        "apply_reset");
    require(state.phase == REK_G1_FALL_UPRIGHT, "reset_upright");
    require(state.reset_grace_remaining_seconds
        == REK_G1_FALL_CONFIG_F84F1874.reset_grace_seconds,
        "reset_grace_loaded");
    require(state.recovery_armed == 0u, "reset_clears_recovery");

    RekG1FallSample sample = upright_sample();
    sample.tilt_degrees = nextafterf(
        REK_G1_FALL_CONFIG_F84F1874.falling_tilt_degrees, INFINITY);
    sample.fixed_delta_seconds = 0.5f;
    require(step_ok(&state, &sample) == REK_G1_FALL_EVENT_NONE,
        "positive_grace_consumes_whole_tick");
    require(state.reset_grace_remaining_seconds == 0.0f,
        "grace_exactly_zero");
    require(state.phase == REK_G1_FALL_UPRIGHT, "grace_keeps_upright");
    require(step_ok(&state, &sample) == REK_G1_FALL_EVENT_FALLING_STARTED,
        "detection_resumes_tick_after_grace_zero");

    fallen = become_fallen(0u);
    require(rek_g1_fall_state_apply_reset_after_fall(
        &REK_G1_FALL_CONFIG_F84F1874, &fallen, &state) == REK_G1_FALL_OK,
        "apply_reset_for_grace_overshoot");
    sample.fixed_delta_seconds = 0.75f;
    require(step_ok(&state, &sample) == REK_G1_FALL_EVENT_NONE,
        "grace_overshoot_still_consumes_whole_tick");
    require(state.reset_grace_remaining_seconds == -0.25f,
        "grace_overshoot_not_clamped");
    require(step_ok(&state, &sample) == REK_G1_FALL_EVENT_FALLING_STARTED,
        "detection_resumes_after_grace_overshoot");

    RekG1FallState unchanged = state;
    RekG1FallState output = state;
    require(rek_g1_fall_state_apply_reset_after_fall(
        &REK_G1_FALL_CONFIG_F84F1874, &state, &output)
        == REK_G1_FALL_STATE_INVALID, "reset_requires_fallen");
    require(memcmp(&output, &unchanged, sizeof(output)) == 0,
        "invalid_reset_transactional");
}

static void test_fight_spawn_reset_grace(void) {
    RekG1FallState states[3] = {
        initialized_state(),
        falling_state(),
        become_fallen(0u),
    };
    for (size_t index = 0u; index < 3u; index++) {
        RekG1FallState next;
        require(rek_g1_fall_state_apply_fight_spawn_reset(
            &REK_G1_FALL_CONFIG_F84F1874, &states[index], &next)
            == REK_G1_FALL_OK, "fight_spawn_reset_accepts_phase");
        require(next.phase == REK_G1_FALL_UPRIGHT,
            "fight_spawn_reset_upright");
        require(float_bits(next.reset_grace_remaining_seconds)
            == 0x40000000u, "fight_spawn_reset_grace_exact");
        require(next.fallen_hold_seconds == 0.0f
            && next.fallen_elapsed_seconds == 0.0f,
            "fight_spawn_reset_clears_fall_timers");
        require(next.fallen_timer_seconds
            == REK_G1_FALL_CONFIG_F84F1874.fallen_reset_timeout_seconds,
            "fight_spawn_reset_rearms_timeout");
        require(next.recovery_armed == 0u,
            "fight_spawn_reset_clears_recovery");
    }

    RekG1FallState invalid = initialized_state();
    invalid.phase = (RekG1FallPhase)99;
    RekG1FallState output;
    memset(&output, 0xa5, sizeof(output));
    RekG1FallState sentinel = output;
    require(rek_g1_fall_state_apply_fight_spawn_reset(
        &REK_G1_FALL_CONFIG_F84F1874, &invalid, &output)
        == REK_G1_FALL_STATE_INVALID,
        "fight_spawn_reset_rejects_invalid_state");
    require(memcmp(&output, &sentinel, sizeof(output)) == 0,
        "fight_spawn_reset_invalid_transactional");
}

static void test_invalid_inputs_transactional(void) {
    RekG1FallState state = initialized_state();
    RekG1FallSample sample = upright_sample();
    RekG1FallStepResult result;
    memset(&result, 0xa5, sizeof(result));
    RekG1FallStepResult sentinel = result;

    sample.tracking_active = 2u;
    require(rek_g1_fall_state_step(&REK_G1_FALL_CONFIG_F84F1874,
        &state, &sample, &result) == REK_G1_FALL_INPUT_INVALID,
        "invalid_flag_rejected");
    require(memcmp(&result, &sentinel, sizeof(result)) == 0,
        "invalid_flag_transactional");

    sample = upright_sample();
    sample.tilt_degrees = NAN;
    require(rek_g1_fall_state_step(&REK_G1_FALL_CONFIG_F84F1874,
        &state, &sample, &result) == REK_G1_FALL_NON_FINITE,
        "nan_rejected");
    require(memcmp(&result, &sentinel, sizeof(result)) == 0,
        "nan_transactional");

    sample = upright_sample();
    sample.fixed_delta_seconds = 0.0f;
    require(rek_g1_fall_state_step(&REK_G1_FALL_CONFIG_F84F1874,
        &state, &sample, &result) == REK_G1_FALL_INPUT_INVALID,
        "zero_delta_rejected");

    RekG1FallConfig bad_config = REK_G1_FALL_CONFIG_F84F1874;
    bad_config.fallen_hold_fast_seconds = INFINITY;
    require(rek_g1_fall_state_step(&bad_config, &state, &sample, &result)
        == REK_G1_FALL_NON_FINITE, "nonfinite_config_rejected_first");

    bad_config = REK_G1_FALL_CONFIG_F84F1874;
    bad_config.fallen_contact_points = 0u;
    require(rek_g1_fall_state_init(&bad_config, &state)
        == REK_G1_FALL_CONFIG_INVALID, "invalid_config_rejected");

    require(rek_g1_fall_state_step(NULL, &state, &sample, &result)
        == REK_G1_FALL_NULL_ARGUMENT, "null_config_rejected");
}

int main(void) {
    test_config();
    test_upright_start_strictness();
    test_falling_clear_strictness();
    test_tracking_active_qualifiers();
    test_tracking_inactive_qualifiers();
    test_recovery_and_reset_timeout();
    test_reset_grace();
    test_fight_spawn_reset_grace();
    test_invalid_inputs_transactional();
    printf("G1 fall state passed: assertions=%d\n", assertions);
    return 0;
}
