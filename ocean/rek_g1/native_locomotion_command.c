#include "native_locomotion_command.h"

#include <math.h>

static int command_finite(RekG1NativeVelocityCommand command) {
    return isfinite(command.forward)
        && isfinite(command.strafe)
        && isfinite(command.yaw);
}

static int config_valid(const RekG1NativeCommandConfig* config) {
    return config != NULL
        && isfinite(config->locomotion_speed_scale)
        && isfinite(config->command_yaw_rate_scale)
        && isfinite(config->heading_yaw_rate_scale)
        && config->controller_rate_hz > 0;
}

static int boolean_valid(uint8_t value) {
    return value <= 1u;
}

static int locomotion_route_valid(RekG1NativeRouteId route_id) {
    return route_id >= REK_G1_NATIVE_FORWARD
        && route_id <= REK_G1_NATIVE_TURN_RIGHT;
}

static int route_slot_valid(RekG1NativeRouteId route_id) {
    return route_id >= REK_G1_NATIVE_IDLE
        && route_id <= REK_G1_NATIVE_TURN_RIGHT;
}

static int locomotion_config_valid(
        const RekG1NativeLocomotionConfig* config) {
    return config != NULL
        && isfinite(config->settle_linear_speed)
        && isfinite(config->settle_yaw_rate)
        && isfinite(config->stop_brake_rate)
        && config->settle_linear_speed >= 0.0f
        && config->settle_yaw_rate >= 0.0f
        && config->stop_brake_rate >= 0.0f
        && boolean_valid(config->transition_settle);
}

static int locomotion_state_valid(
        const RekG1NativeLocomotionState* state) {
    if (state == NULL
            || !command_finite(state->stop_brake_command)
            || !command_finite(state->last_driven_command)
            || !boolean_valid(state->locomotion_active)
            || !boolean_valid(state->transition_settling)
            || !boolean_valid(state->stop_braking)
            || !boolean_valid(state->has_momentum)
            || !route_slot_valid(state->current_route_id)
            || !route_slot_valid(state->transition_from_route_id)
            || !route_slot_valid(state->momentum_route_id)) {
        return 0;
    }
    if (state->locomotion_active
            && !locomotion_route_valid(state->current_route_id)) {
        return 0;
    }
    if (state->transition_settling
            && !locomotion_route_valid(state->transition_from_route_id)) {
        return 0;
    }
    if (state->has_momentum
            && !locomotion_route_valid(state->momentum_route_id)) {
        return 0;
    }
    if ((state->locomotion_active && state->transition_settling)
            || (state->locomotion_active && state->has_momentum)
            || (state->transition_settling && state->has_momentum)
            || (state->transition_settling && state->stop_braking)) {
        return 0;
    }
    return 1;
}

static RekG1NativeCommandStatus vector_magnitude(
        RekG1NativeVelocityCommand value,
        float* magnitude) {
    if (magnitude == NULL) return REK_G1_NATIVE_COMMAND_NULL_ARGUMENT;
    if (!command_finite(value)) return REK_G1_NATIVE_COMMAND_NON_FINITE;

    const float squared = value.forward * value.forward
        + value.strafe * value.strafe
        + value.yaw * value.yaw;
    if (!isfinite(squared) || squared < 0.0f) {
        return REK_G1_NATIVE_COMMAND_NON_FINITE;
    }
    const float result = (float)sqrt((double)squared);
    if (!isfinite(result)) return REK_G1_NATIVE_COMMAND_NON_FINITE;
    *magnitude = result;
    return REK_G1_NATIVE_COMMAND_OK;
}

/* UnityEngine.Vector3.MoveTowards(current, Vector3.zero, max_delta). */
static RekG1NativeCommandStatus move_towards_zero(
        RekG1NativeVelocityCommand current,
        float max_delta,
        RekG1NativeVelocityCommand* next) {
    if (next == NULL) return REK_G1_NATIVE_COMMAND_NULL_ARGUMENT;
    if (!command_finite(current) || !isfinite(max_delta)) {
        return REK_G1_NATIVE_COMMAND_NON_FINITE;
    }
    if (max_delta < 0.0f) return REK_G1_NATIVE_COMMAND_TIMING_INVALID;

    const float delta_forward = -current.forward;
    const float delta_strafe = -current.strafe;
    const float delta_yaw = -current.yaw;
    const float squared = delta_forward * delta_forward
        + delta_strafe * delta_strafe
        + delta_yaw * delta_yaw;
    if (!isfinite(squared) || squared < 0.0f) {
        return REK_G1_NATIVE_COMMAND_NON_FINITE;
    }
    const float max_delta_squared = max_delta * max_delta;
    if (squared == 0.0f || max_delta_squared >= squared) {
        *next = (RekG1NativeVelocityCommand){0};
        return REK_G1_NATIVE_COMMAND_OK;
    }

    const float distance = (float)sqrt((double)squared);
    if (!isfinite(distance) || distance <= 0.0f) {
        return REK_G1_NATIVE_COMMAND_NON_FINITE;
    }
    const float scale = max_delta / distance;
    const RekG1NativeVelocityCommand result = {
        .forward = current.forward + delta_forward * scale,
        .strafe = current.strafe + delta_strafe * scale,
        .yaw = current.yaw + delta_yaw * scale,
    };
    if (!command_finite(result)) return REK_G1_NATIVE_COMMAND_NON_FINITE;
    *next = result;
    return REK_G1_NATIVE_COMMAND_OK;
}

RekG1NativeCommandStatus rek_g1_native_select_locomotion_route(
        RekG1NativeVelocityCommand command,
        RekG1NativeRouteSelection* selection) {
    if (selection == NULL) return REK_G1_NATIVE_COMMAND_NULL_ARGUMENT;
    if (!command_finite(command)) return REK_G1_NATIVE_COMMAND_NON_FINITE;

    const float absolute_forward = fabsf(command.forward);
    const float absolute_strafe = fabsf(command.strafe);
    const float absolute_yaw = fabsf(command.yaw);
    const float maximum = fmaxf(
        absolute_forward,
        fmaxf(absolute_strafe, absolute_yaw));
    if (maximum < REK_G1_NATIVE_COMMAND_EPSILON) {
        *selection = (RekG1NativeRouteSelection){
            .route_id = REK_G1_NATIVE_IDLE,
            .locomotion_active = 0,
        };
        return REK_G1_NATIVE_COMMAND_OK;
    }

    RekG1NativeRouteId route_id;
    if (absolute_forward >= REK_G1_NATIVE_COMMAND_EPSILON
            || absolute_strafe >= REK_G1_NATIVE_COMMAND_EPSILON) {
        if (absolute_forward >= absolute_strafe) {
            route_id = command.forward < 0.0f
                ? REK_G1_NATIVE_BACKWARD
                : REK_G1_NATIVE_FORWARD;
        } else {
            route_id = command.strafe < 0.0f
                ? REK_G1_NATIVE_STRAFE_RIGHT
                : REK_G1_NATIVE_STRAFE_LEFT;
        }
    } else {
        route_id = command.yaw < 0.0f
            ? REK_G1_NATIVE_TURN_RIGHT
            : REK_G1_NATIVE_TURN_LEFT;
    }
    *selection = (RekG1NativeRouteSelection){
        .route_id = route_id,
        .locomotion_active = 1,
    };
    return REK_G1_NATIVE_COMMAND_OK;
}

RekG1NativeCommandStatus rek_g1_native_playback_update(
        RekG1NativeVelocityCommand command,
        const RekG1NativeCommandConfig* config,
        RekG1NativePlaybackUpdate* update) {
    if (update == NULL || config == NULL) {
        return REK_G1_NATIVE_COMMAND_NULL_ARGUMENT;
    }
    if (!command_finite(command)) return REK_G1_NATIVE_COMMAND_NON_FINITE;
    if (!config_valid(config)) return REK_G1_NATIVE_COMMAND_CONFIG_INVALID;

    if (config->locomotion_speed_scale == 0.0f) {
        *update = (RekG1NativePlaybackUpdate){0};
        return REK_G1_NATIVE_COMMAND_OK;
    }
    const float squared = command.forward * command.forward
        + command.strafe * command.strafe
        + command.yaw * command.yaw;
    const float magnitude = sqrtf(squared);
    if (!isfinite(squared) || !isfinite(magnitude)) {
        return REK_G1_NATIVE_COMMAND_NON_FINITE;
    }
    *update = (RekG1NativePlaybackUpdate){
        .command_magnitude = magnitude,
        .scale = magnitude > REK_G1_NATIVE_COMMAND_EPSILON
            ? magnitude * config->locomotion_speed_scale
            : 1.0f,
        .apply = 1,
    };
    if (!isfinite(update->scale)) return REK_G1_NATIVE_COMMAND_NON_FINITE;
    return REK_G1_NATIVE_COMMAND_OK;
}

RekG1NativeCommandStatus rek_g1_native_heading_update(
        RekG1NativeVelocityCommand command,
        const RekG1NativeCommandConfig* config,
        float heading_clip_ownership,
        float consumed_clip_heading_delta_radians,
        float forgiveness_delta_radians,
        RekG1NativeHeadingUpdate* update) {
    if (update == NULL || config == NULL) {
        return REK_G1_NATIVE_COMMAND_NULL_ARGUMENT;
    }
    if (!command_finite(command)
            || !isfinite(heading_clip_ownership)
            || !isfinite(consumed_clip_heading_delta_radians)
            || !isfinite(forgiveness_delta_radians)) {
        return REK_G1_NATIVE_COMMAND_NON_FINITE;
    }
    if (!config_valid(config)) return REK_G1_NATIVE_COMMAND_CONFIG_INVALID;
    if (heading_clip_ownership < 0.0f || heading_clip_ownership > 1.0f) {
        return REK_G1_NATIVE_COMMAND_HEADING_OWNERSHIP_INVALID;
    }

    const float clip_delta = consumed_clip_heading_delta_radians
        * config->heading_yaw_rate_scale;
    const float command_delta = (
        config->command_yaw_rate_scale * command.yaw
        / (float)config->controller_rate_hz)
        * (1.0f - heading_clip_ownership);
    const float total_delta = clip_delta + command_delta
        + forgiveness_delta_radians;
    if (!isfinite(clip_delta) || !isfinite(command_delta)
            || !isfinite(total_delta)) {
        return REK_G1_NATIVE_COMMAND_NON_FINITE;
    }
    *update = (RekG1NativeHeadingUpdate){
        .clip_delta_radians = clip_delta,
        .command_delta_radians = command_delta,
        .forgiveness_delta_radians = forgiveness_delta_radians,
        .total_delta_radians = total_delta,
    };
    return REK_G1_NATIVE_COMMAND_OK;
}

RekG1NativeCommandStatus rek_g1_native_transition_settled(
        RekG1NativeRouteId outgoing_route_id,
        const RekG1NativeLocomotionConfig* config,
        const RekG1NativeBaseVelocitySample* base_velocity,
        uint8_t* settled) {
    if (config == NULL || base_velocity == NULL || settled == NULL) {
        return REK_G1_NATIVE_COMMAND_NULL_ARGUMENT;
    }
    if (!locomotion_config_valid(config)) {
        return REK_G1_NATIVE_COMMAND_CONFIG_INVALID;
    }
    if (!locomotion_route_valid(outgoing_route_id)) {
        return REK_G1_NATIVE_COMMAND_STATE_INVALID;
    }
    if (!boolean_valid(base_velocity->available)) {
        return REK_G1_NATIVE_COMMAND_INPUT_INVALID;
    }
    if (!base_velocity->available) {
        return REK_G1_NATIVE_COMMAND_MEASUREMENT_UNAVAILABLE;
    }
    if (!command_finite(base_velocity->angular_velocity_local)
            || !command_finite(base_velocity->linear_velocity_local)) {
        return REK_G1_NATIVE_COMMAND_NON_FINITE;
    }

    float component;
    float threshold;
    switch (outgoing_route_id) {
        case REK_G1_NATIVE_FORWARD:
        case REK_G1_NATIVE_BACKWARD:
            component = base_velocity->linear_velocity_local.forward;
            threshold = config->settle_linear_speed;
            break;
        case REK_G1_NATIVE_STRAFE_LEFT:
        case REK_G1_NATIVE_STRAFE_RIGHT:
            component = base_velocity->linear_velocity_local.strafe;
            threshold = config->settle_linear_speed;
            break;
        case REK_G1_NATIVE_TURN_LEFT:
        case REK_G1_NATIVE_TURN_RIGHT:
            component = base_velocity->angular_velocity_local.yaw;
            threshold = config->settle_yaw_rate;
            break;
        default:
            return REK_G1_NATIVE_COMMAND_STATE_INVALID;
    }
    *settled = threshold > fabsf(component) ? 1u : 0u;
    return REK_G1_NATIVE_COMMAND_OK;
}

static RekG1NativeCommandStatus check_transition(
        RekG1NativeRouteId outgoing_route_id,
        const RekG1NativeLocomotionConfig* config,
        const RekG1NativeLocomotionStepInput* input,
        RekG1NativeLocomotionStepResult* result,
        uint8_t* settled) {
    uint8_t local_settled = 0u;
    const RekG1NativeCommandStatus status =
        rek_g1_native_transition_settled(
            outgoing_route_id,
            config,
            &input->base_velocity,
            &local_settled);
    if (status != REK_G1_NATIVE_COMMAND_OK) return status;
    result->transition_check_performed = 1u;
    result->transition_settled = local_settled;
    *settled = local_settled;
    return REK_G1_NATIVE_COMMAND_OK;
}

static void emit_idle(RekG1NativeLocomotionStepResult* result) {
    result->event = REK_G1_NATIVE_LOCOMOTION_EVENT_PLAY_IDLE;
    result->event_route_id = REK_G1_NATIVE_IDLE;
}

static RekG1NativeCommandStatus update_stop_brake(
        const RekG1NativeLocomotionConfig* config,
        const RekG1NativeLocomotionStepInput* input,
        RekG1NativeLocomotionStepResult* result) {
    RekG1NativeLocomotionState* state = &result->next_state;

    if (state->locomotion_active && config->stop_brake_rate > 0.0f) {
        if (!state->stop_braking) {
            state->stop_brake_command = state->last_driven_command;
            state->stop_braking = 1u;
        }

        const float max_delta = input->delta_seconds
            * config->stop_brake_rate;
        if (!isfinite(max_delta)) return REK_G1_NATIVE_COMMAND_NON_FINITE;
        RekG1NativeVelocityCommand next_brake = {0};
        RekG1NativeCommandStatus status = move_towards_zero(
            state->stop_brake_command,
            max_delta,
            &next_brake);
        if (status != REK_G1_NATIVE_COMMAND_OK) return status;
        state->stop_brake_command = next_brake;

        float magnitude = 0.0f;
        status = vector_magnitude(state->stop_brake_command, &magnitude);
        if (status != REK_G1_NATIVE_COMMAND_OK) return status;
        if (magnitude > REK_G1_NATIVE_STOP_BRAKE_DONE_MAGNITUDE) {
            result->effective_velocity = state->stop_brake_command;
            result->velocity_write = 1u;
            return REK_G1_NATIVE_COMMAND_OK;
        }
        state->stop_braking = 0u;
    }

    if (state->locomotion_active) {
        state->momentum_route_id = state->current_route_id;
        state->has_momentum = 1u;
    }
    if (state->locomotion_active || input->composer_busy) {
        emit_idle(result);
        state->locomotion_active = 0u;
    }
    return REK_G1_NATIVE_COMMAND_OK;
}

RekG1NativeCommandStatus rek_g1_native_locomotion_step(
        const RekG1NativeLocomotionState* state,
        const RekG1NativeLocomotionConfig* config,
        const RekG1NativeLocomotionStepInput* input,
        RekG1NativeLocomotionStepResult* result) {
    if (state == NULL || config == NULL || input == NULL || result == NULL) {
        return REK_G1_NATIVE_COMMAND_NULL_ARGUMENT;
    }
    if (!locomotion_state_valid(state)) {
        return REK_G1_NATIVE_COMMAND_STATE_INVALID;
    }
    if (!locomotion_config_valid(config)) {
        return REK_G1_NATIVE_COMMAND_CONFIG_INVALID;
    }
    if (!command_finite(input->command)) {
        return REK_G1_NATIVE_COMMAND_NON_FINITE;
    }
    if (!isfinite(input->delta_seconds)) {
        return REK_G1_NATIVE_COMMAND_NON_FINITE;
    }
    if (input->delta_seconds < 0.0f) {
        return REK_G1_NATIVE_COMMAND_TIMING_INVALID;
    }
    if (!boolean_valid(input->restrict_yaw)
            || !boolean_valid(input->composer_action_playing)
            || !boolean_valid(input->composer_busy)
            || !boolean_valid(input->selected_route_playable)
            || !boolean_valid(input->base_velocity.available)) {
        return REK_G1_NATIVE_COMMAND_INPUT_INVALID;
    }
    if (input->base_velocity.available
            && (!command_finite(input->base_velocity.angular_velocity_local)
                || !command_finite(
                    input->base_velocity.linear_velocity_local))) {
        return REK_G1_NATIVE_COMMAND_NON_FINITE;
    }

    RekG1NativeLocomotionStepResult local = {
        .next_state = *state,
        .effective_velocity = input->command,
        .selected_route_id = REK_G1_NATIVE_IDLE,
        .event_route_id = REK_G1_NATIVE_IDLE,
        .event = REK_G1_NATIVE_LOCOMOTION_EVENT_NONE,
        .velocity_write = 0u,
        .transition_check_performed = 0u,
        .transition_settled = 0u,
    };

    if (input->restrict_yaw && input->composer_action_playing
            && fabsf(local.effective_velocity.yaw)
                >= REK_G1_NATIVE_COMMAND_EPSILON) {
        local.effective_velocity.yaw = 0.0f;
        local.velocity_write = 1u;
    }
    if (input->composer_action_playing) {
        *result = local;
        return REK_G1_NATIVE_COMMAND_OK;
    }

    RekG1NativeRouteSelection selection = {0};
    RekG1NativeCommandStatus status = rek_g1_native_select_locomotion_route(
        local.effective_velocity,
        &selection);
    if (status != REK_G1_NATIVE_COMMAND_OK) return status;
    local.selected_route_id = selection.route_id;

    if (!selection.locomotion_active) {
        local.next_state.transition_settling = 0u;
        status = update_stop_brake(config, input, &local);
        if (status != REK_G1_NATIVE_COMMAND_OK) return status;
        *result = local;
        return REK_G1_NATIVE_COMMAND_OK;
    }

    local.next_state.stop_braking = 0u;
    local.next_state.last_driven_command = local.effective_velocity;

    if (local.next_state.transition_settling) {
        uint8_t settled = 0u;
        status = check_transition(
            local.next_state.transition_from_route_id,
            config,
            input,
            &local,
            &settled);
        if (status != REK_G1_NATIVE_COMMAND_OK) return status;
        if (!settled) {
            local.effective_velocity = (RekG1NativeVelocityCommand){0};
            local.velocity_write = 1u;
            *result = local;
            return REK_G1_NATIVE_COMMAND_OK;
        }
        local.next_state.transition_settling = 0u;
    } else if (config->transition_settle) {
        RekG1NativeRouteId outgoing_route_id = REK_G1_NATIVE_IDLE;
        uint8_t needs_transition_check = 0u;
        if (local.next_state.locomotion_active) {
            if (selection.route_id == local.next_state.current_route_id) {
                *result = local;
                return REK_G1_NATIVE_COMMAND_OK;
            }
            outgoing_route_id = local.next_state.current_route_id;
            needs_transition_check = 1u;
        } else if (local.next_state.has_momentum
                && selection.route_id
                    != local.next_state.momentum_route_id) {
            outgoing_route_id = local.next_state.momentum_route_id;
            needs_transition_check = 1u;
        }

        if (needs_transition_check) {
            uint8_t settled = 0u;
            status = check_transition(
                outgoing_route_id,
                config,
                input,
                &local,
                &settled);
            if (status != REK_G1_NATIVE_COMMAND_OK) return status;
            if (!settled) {
                const uint8_t was_active =
                    local.next_state.locomotion_active;
                local.next_state.locomotion_active = 0u;
                local.next_state.transition_settling = 1u;
                local.next_state.transition_from_route_id =
                    outgoing_route_id;
                local.next_state.has_momentum = 0u;
                local.effective_velocity =
                    (RekG1NativeVelocityCommand){0};
                local.velocity_write = 1u;
                if (was_active) emit_idle(&local);
                *result = local;
                return REK_G1_NATIVE_COMMAND_OK;
            }
        }
    }

    if (local.next_state.locomotion_active
            && selection.route_id == local.next_state.current_route_id) {
        *result = local;
        return REK_G1_NATIVE_COMMAND_OK;
    }
    if (!input->selected_route_playable) {
        *result = local;
        return REK_G1_NATIVE_COMMAND_OK;
    }

    local.event = REK_G1_NATIVE_LOCOMOTION_EVENT_PLAY_ROUTE;
    local.event_route_id = selection.route_id;
    local.next_state.current_route_id = selection.route_id;
    local.next_state.locomotion_active = 1u;
    local.next_state.has_momentum = 0u;
    *result = local;
    return REK_G1_NATIVE_COMMAND_OK;
}
