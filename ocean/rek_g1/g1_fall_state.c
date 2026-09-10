#include "g1_fall_state.h"

#include <math.h>
#include <stddef.h>

REK_G1_CONSTANT const RekG1FallConfig REK_G1_FALL_CONFIG_F84F1874 = {
    .falling_tilt_degrees = 42.0f,
    .falling_height_ratio = 0.6000000238418579f,
    .fallen_tilt_degrees = 69.30000305175781f,
    .fallen_height_ratio = 0.4000000059604645f,
    .fallen_contact_points = 3u,
    .fallen_hold_fast_seconds = 0.15000000596046448f,
    .fallen_hold_slow_seconds = 0.5f,
    .fallen_reset_timeout_seconds = 3.0f,
    .reset_grace_seconds = 0.5f,
    .fight_spawn_reset_grace_seconds = 2.0f,
};

static REK_G1_FN int valid_flag(uint8_t value) {
    return value == 0u || value == 1u;
}

static REK_G1_FN RekG1FallStatus validate_config(const RekG1FallConfig* config) {
    if (!isfinite(config->falling_tilt_degrees)
            || !isfinite(config->falling_height_ratio)
            || !isfinite(config->fallen_tilt_degrees)
            || !isfinite(config->fallen_height_ratio)
            || !isfinite(config->fallen_hold_fast_seconds)
            || !isfinite(config->fallen_hold_slow_seconds)
            || !isfinite(config->fallen_reset_timeout_seconds)
            || !isfinite(config->reset_grace_seconds)
            || !isfinite(config->fight_spawn_reset_grace_seconds)) {
        return REK_G1_FALL_NON_FINITE;
    }
    if (config->falling_tilt_degrees < 0.0f
            || config->falling_height_ratio < 0.0f
            || config->fallen_tilt_degrees < 0.0f
            || config->fallen_height_ratio < 0.0f
            || config->fallen_contact_points == 0u
            || config->fallen_hold_fast_seconds <= 0.0f
            || config->fallen_hold_slow_seconds <= 0.0f
            || config->fallen_reset_timeout_seconds <= 0.0f
            || config->reset_grace_seconds < 0.0f
            || config->fight_spawn_reset_grace_seconds < 0.0f) {
        return REK_G1_FALL_CONFIG_INVALID;
    }
    return REK_G1_FALL_OK;
}

static REK_G1_FN RekG1FallStatus validate_state(const RekG1FallState* state) {
    if (state->phase != REK_G1_FALL_UPRIGHT
            && state->phase != REK_G1_FALL_FALLING
            && state->phase != REK_G1_FALL_FALLEN) {
        return REK_G1_FALL_STATE_INVALID;
    }
    if (!isfinite(state->fallen_hold_seconds)
            || !isfinite(state->fallen_elapsed_seconds)
            || !isfinite(state->fallen_timer_seconds)
            || !isfinite(state->reset_grace_remaining_seconds)) {
        return REK_G1_FALL_NON_FINITE;
    }
    if (state->fallen_hold_seconds < 0.0f
            || state->fallen_elapsed_seconds < 0.0f
            || state->fallen_timer_seconds < 0.0f
            || !valid_flag(state->recovery_armed)) {
        return REK_G1_FALL_STATE_INVALID;
    }
    return REK_G1_FALL_OK;
}

static REK_G1_FN RekG1FallStatus validate_sample(const RekG1FallSample* sample) {
    if (!isfinite(sample->tilt_degrees)
            || !isfinite(sample->pelvis_height_ratio)
            || !isfinite(sample->fixed_delta_seconds)) {
        return REK_G1_FALL_NON_FINITE;
    }
    if (!valid_flag(sample->tracking_active)
            || !valid_flag(sample->both_feet_off_floor)
            || !valid_flag(sample->has_foot_body_contact)
            || !valid_flag(sample->can_get_up)
            || sample->fixed_delta_seconds <= 0.0f) {
        return REK_G1_FALL_INPUT_INVALID;
    }
    return REK_G1_FALL_OK;
}

static REK_G1_FN RekG1FallStatus validate_common(
    const RekG1FallConfig* config,
    const RekG1FallState* state
) {
    RekG1FallStatus status = validate_config(config);
    if (status != REK_G1_FALL_OK) {
        return status;
    }
    return validate_state(state);
}

REK_G1_FN RekG1FallStatus rek_g1_fall_state_init(
    const RekG1FallConfig* config,
    RekG1FallState* state
) {
    if (config == NULL || state == NULL) {
        return REK_G1_FALL_NULL_ARGUMENT;
    }
    RekG1FallStatus status = validate_config(config);
    if (status != REK_G1_FALL_OK) {
        return status;
    }
    RekG1FallState initialized = {
        .phase = REK_G1_FALL_UPRIGHT,
        .fallen_hold_seconds = 0.0f,
        .fallen_elapsed_seconds = 0.0f,
        .fallen_timer_seconds = config->fallen_reset_timeout_seconds,
        .reset_grace_remaining_seconds = 0.0f,
        .recovery_armed = 0u,
    };
    *state = initialized;
    return REK_G1_FALL_OK;
}

static REK_G1_FN int update_fallen_qualifiers(
    const RekG1FallConfig* config,
    RekG1FallState* state,
    const RekG1FallSample* sample
) {
    float hold_seconds = config->fallen_hold_slow_seconds;
    if (sample->tracking_active) {
        const int tilted = sample->tilt_degrees > config->fallen_tilt_degrees;
        const int low = tilted
            && sample->pelvis_height_ratio < config->fallen_height_ratio;
        const int no_foot_body_contact = !sample->has_foot_body_contact
            && sample->distinct_nonfoot_body_contact_count >= 1u;
        const int enough_nonfoot_contacts =
            sample->distinct_nonfoot_body_contact_count
                >= config->fallen_contact_points;
        const int contact_qualifier =
            no_foot_body_contact || enough_nonfoot_contacts;
        if (!(low && contact_qualifier)) {
            state->fallen_hold_seconds = 0.0f;
            return 0;
        }
        if (no_foot_body_contact && enough_nonfoot_contacts) {
            hold_seconds = config->fallen_hold_fast_seconds;
        }
    } else {
        if (sample->tilt_degrees <= config->fallen_tilt_degrees) {
            state->fallen_hold_seconds = 0.0f;
            return 0;
        }
    }

    state->fallen_hold_seconds += sample->fixed_delta_seconds;
    return state->fallen_hold_seconds >= hold_seconds;
}

static REK_G1_FN void become_fallen(
    const RekG1FallConfig* config,
    RekG1FallState* state,
    const RekG1FallSample* sample
) {
    state->phase = REK_G1_FALL_FALLEN;
    state->fallen_timer_seconds = config->fallen_reset_timeout_seconds;
    state->fallen_elapsed_seconds = 0.0f;
    state->fallen_hold_seconds = 0.0f;
    state->recovery_armed = sample->can_get_up;
}

REK_G1_FN RekG1FallStatus rek_g1_fall_state_step(
    const RekG1FallConfig* config,
    const RekG1FallState* state,
    const RekG1FallSample* sample,
    RekG1FallStepResult* result
) {
    if (config == NULL || state == NULL || sample == NULL || result == NULL) {
        return REK_G1_FALL_NULL_ARGUMENT;
    }
    RekG1FallStatus status = validate_common(config, state);
    if (status != REK_G1_FALL_OK) {
        return status;
    }
    status = validate_sample(sample);
    if (status != REK_G1_FALL_OK) {
        return status;
    }

    RekG1FallStepResult next = {
        .next_state = *state,
        .events = REK_G1_FALL_EVENT_NONE,
    };

    if (next.next_state.phase == REK_G1_FALL_FALLEN) {
        next.next_state.fallen_elapsed_seconds += sample->fixed_delta_seconds;
        if (!next.next_state.recovery_armed) {
            next.next_state.fallen_timer_seconds -= sample->fixed_delta_seconds;
            if (next.next_state.fallen_timer_seconds <= 0.0f) {
                next.next_state.fallen_timer_seconds =
                    config->fallen_reset_timeout_seconds;
                next.events |= REK_G1_FALL_EVENT_RESET_TIMEOUT_DUE;
            }
        }
    } else if (next.next_state.reset_grace_remaining_seconds > 0.0f) {
        next.next_state.reset_grace_remaining_seconds -=
            sample->fixed_delta_seconds;
    } else if (next.next_state.phase == REK_G1_FALL_UPRIGHT) {
        const int tilt_start =
            sample->tilt_degrees > config->falling_tilt_degrees;
        const int height_start = sample->both_feet_off_floor
            && sample->pelvis_height_ratio < config->falling_height_ratio;
        if (tilt_start || height_start) {
            next.next_state.phase = REK_G1_FALL_FALLING;
            next.next_state.fallen_hold_seconds = 0.0f;
            next.events |= REK_G1_FALL_EVENT_FALLING_STARTED;
        }
    } else if (update_fallen_qualifiers(
            config, &next.next_state, sample)) {
        become_fallen(config, &next.next_state, sample);
        next.events |= REK_G1_FALL_EVENT_BECAME_FALLEN;
    } else if (sample->tilt_degrees <= config->falling_tilt_degrees
            && !sample->both_feet_off_floor) {
        next.next_state.phase = REK_G1_FALL_UPRIGHT;
        next.next_state.fallen_hold_seconds = 0.0f;
        next.events |= REK_G1_FALL_EVENT_FALLING_CLEARED;
    }

    if (!isfinite(next.next_state.fallen_hold_seconds)
            || !isfinite(next.next_state.fallen_elapsed_seconds)
            || !isfinite(next.next_state.fallen_timer_seconds)
            || !isfinite(next.next_state.reset_grace_remaining_seconds)) {
        return REK_G1_FALL_NON_FINITE;
    }
    *result = next;
    return REK_G1_FALL_OK;
}

REK_G1_FN RekG1FallStatus rek_g1_fall_state_apply_reset_after_fall(
    const RekG1FallConfig* config,
    const RekG1FallState* state,
    RekG1FallState* next_state
) {
    if (config == NULL || state == NULL || next_state == NULL) {
        return REK_G1_FALL_NULL_ARGUMENT;
    }
    RekG1FallStatus status = validate_common(config, state);
    if (status != REK_G1_FALL_OK) {
        return status;
    }
    if (state->phase != REK_G1_FALL_FALLEN) {
        return REK_G1_FALL_STATE_INVALID;
    }

    RekG1FallState reset = {
        .phase = REK_G1_FALL_UPRIGHT,
        .fallen_hold_seconds = 0.0f,
        .fallen_elapsed_seconds = 0.0f,
        .fallen_timer_seconds = config->fallen_reset_timeout_seconds,
        .reset_grace_remaining_seconds = config->reset_grace_seconds,
        .recovery_armed = 0u,
    };
    *next_state = reset;
    return REK_G1_FALL_OK;
}

REK_G1_FN RekG1FallStatus rek_g1_fall_state_apply_fight_spawn_reset(
    const RekG1FallConfig* config,
    const RekG1FallState* state,
    RekG1FallState* next_state
) {
    if (config == NULL || state == NULL || next_state == NULL) {
        return REK_G1_FALL_NULL_ARGUMENT;
    }
    RekG1FallStatus status = validate_common(config, state);
    if (status != REK_G1_FALL_OK) {
        return status;
    }

    RekG1FallState reset = {
        .phase = REK_G1_FALL_UPRIGHT,
        .fallen_hold_seconds = 0.0f,
        .fallen_elapsed_seconds = 0.0f,
        .fallen_timer_seconds = config->fallen_reset_timeout_seconds,
        .reset_grace_remaining_seconds =
            config->fight_spawn_reset_grace_seconds,
        .recovery_armed = 0u,
    };
    *next_state = reset;
    return REK_G1_FALL_OK;
}
