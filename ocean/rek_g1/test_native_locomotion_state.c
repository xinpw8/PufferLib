#include "native_locomotion_command.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int checks;
static int failures;

#define CHECK(condition) do { \
    checks++; \
    if (!(condition)) { \
        failures++; \
        fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #condition); \
    } \
} while (0)

static int near(float actual, float expected) {
    return fabsf(actual - expected) <= 1e-6f;
}

static int velocity_equal(
        RekG1NativeVelocityCommand actual,
        RekG1NativeVelocityCommand expected) {
    return near(actual.forward, expected.forward)
        && near(actual.strafe, expected.strafe)
        && near(actual.yaw, expected.yaw);
}

static RekG1NativeLocomotionConfig test_config(void) {
    return (RekG1NativeLocomotionConfig){
        .settle_linear_speed = 0.03f,
        .settle_yaw_rate = 0.04f,
        .stop_brake_rate = 2.0f,
        .transition_settle = 1u,
    };
}

static RekG1NativeLocomotionStepInput test_input(
        RekG1NativeVelocityCommand command) {
    return (RekG1NativeLocomotionStepInput){
        .command = command,
        .base_velocity = {
            .angular_velocity_local = {0},
            .linear_velocity_local = {0},
            .available = 1u,
        },
        .delta_seconds = 0.02f,
        .restrict_yaw = 0u,
        .composer_action_playing = 0u,
        .composer_busy = 0u,
        .selected_route_playable = 1u,
    };
}

static RekG1NativeLocomotionState active_state(
        RekG1NativeRouteId route_id) {
    return (RekG1NativeLocomotionState){
        .current_route_id = route_id,
        .locomotion_active = 1u,
    };
}

static RekG1NativeCommandStatus step(
        const RekG1NativeLocomotionState* state,
        const RekG1NativeLocomotionConfig* config,
        const RekG1NativeLocomotionStepInput* input,
        RekG1NativeLocomotionStepResult* result) {
    memset(result, 0, sizeof(*result));
    return rek_g1_native_locomotion_step(state, config, input, result);
}

static void test_transition_settled_components(void) {
    RekG1NativeLocomotionConfig config = test_config();
    RekG1NativeBaseVelocitySample sample = {.available = 1u};
    uint8_t settled = 7u;

    sample.linear_velocity_local.forward = 0.029f;
    CHECK(rek_g1_native_transition_settled(
        REK_G1_NATIVE_FORWARD, &config, &sample, &settled)
        == REK_G1_NATIVE_COMMAND_OK);
    CHECK(settled == 1u);
    sample.linear_velocity_local.forward = -0.03f;
    CHECK(rek_g1_native_transition_settled(
        REK_G1_NATIVE_BACKWARD, &config, &sample, &settled)
        == REK_G1_NATIVE_COMMAND_OK);
    CHECK(settled == 0u);

    sample = (RekG1NativeBaseVelocitySample){.available = 1u};
    sample.linear_velocity_local.strafe = -0.039f;
    config.settle_linear_speed = 0.04f;
    CHECK(rek_g1_native_transition_settled(
        REK_G1_NATIVE_STRAFE_LEFT, &config, &sample, &settled)
        == REK_G1_NATIVE_COMMAND_OK);
    CHECK(settled == 1u);
    sample.linear_velocity_local.strafe = 0.04f;
    CHECK(rek_g1_native_transition_settled(
        REK_G1_NATIVE_STRAFE_RIGHT, &config, &sample, &settled)
        == REK_G1_NATIVE_COMMAND_OK);
    CHECK(settled == 0u);

    sample = (RekG1NativeBaseVelocitySample){.available = 1u};
    sample.angular_velocity_local.yaw = 0.039f;
    CHECK(rek_g1_native_transition_settled(
        REK_G1_NATIVE_TURN_LEFT, &config, &sample, &settled)
        == REK_G1_NATIVE_COMMAND_OK);
    CHECK(settled == 1u);
    sample.angular_velocity_local.yaw = -0.04f;
    CHECK(rek_g1_native_transition_settled(
        REK_G1_NATIVE_TURN_RIGHT, &config, &sample, &settled)
        == REK_G1_NATIVE_COMMAND_OK);
    CHECK(settled == 0u);

    sample.available = 0u;
    settled = 19u;
    CHECK(rek_g1_native_transition_settled(
        REK_G1_NATIVE_FORWARD, &config, &sample, &settled)
        == REK_G1_NATIVE_COMMAND_MEASUREMENT_UNAVAILABLE);
    CHECK(settled == 19u);
    sample.available = 1u;
    CHECK(rek_g1_native_transition_settled(
        REK_G1_NATIVE_IDLE, &config, &sample, &settled)
        == REK_G1_NATIVE_COMMAND_STATE_INVALID);
}

static void test_route_start_hold_and_direct_change(void) {
    RekG1NativeLocomotionConfig config = test_config();
    config.transition_settle = 0u;
    RekG1NativeLocomotionState state = {0};
    RekG1NativeLocomotionStepInput input = test_input(
        (RekG1NativeVelocityCommand){.forward = 1.0f});
    RekG1NativeLocomotionStepResult result;

    CHECK(step(&state, &config, &input, &result)
        == REK_G1_NATIVE_COMMAND_OK);
    CHECK(result.event == REK_G1_NATIVE_LOCOMOTION_EVENT_PLAY_ROUTE);
    CHECK(result.event_route_id == REK_G1_NATIVE_FORWARD);
    CHECK(result.selected_route_id == REK_G1_NATIVE_FORWARD);
    CHECK(result.next_state.locomotion_active == 1u);
    CHECK(result.next_state.current_route_id == REK_G1_NATIVE_FORWARD);
    CHECK(velocity_equal(
        result.next_state.last_driven_command, input.command));
    CHECK(result.velocity_write == 0u);

    state = result.next_state;
    input.command = (RekG1NativeVelocityCommand){
        .forward = 0.7f,
        .yaw = -0.5f,
    };
    CHECK(step(&state, &config, &input, &result)
        == REK_G1_NATIVE_COMMAND_OK);
    CHECK(result.event == REK_G1_NATIVE_LOCOMOTION_EVENT_NONE);
    CHECK(result.next_state.current_route_id == REK_G1_NATIVE_FORWARD);
    CHECK(velocity_equal(result.effective_velocity, input.command));
    CHECK(velocity_equal(
        result.next_state.last_driven_command, input.command));

    state = result.next_state;
    input.command = (RekG1NativeVelocityCommand){.forward = -1.0f};
    CHECK(step(&state, &config, &input, &result)
        == REK_G1_NATIVE_COMMAND_OK);
    CHECK(result.event == REK_G1_NATIVE_LOCOMOTION_EVENT_PLAY_ROUTE);
    CHECK(result.event_route_id == REK_G1_NATIVE_BACKWARD);
    CHECK(result.next_state.current_route_id == REK_G1_NATIVE_BACKWARD);

    state = active_state(REK_G1_NATIVE_FORWARD);
    input.selected_route_playable = 0u;
    CHECK(step(&state, &config, &input, &result)
        == REK_G1_NATIVE_COMMAND_OK);
    CHECK(result.event == REK_G1_NATIVE_LOCOMOTION_EVENT_NONE);
    CHECK(result.next_state.current_route_id == REK_G1_NATIVE_FORWARD);
    CHECK(result.next_state.locomotion_active == 1u);
    CHECK(velocity_equal(
        result.next_state.last_driven_command, input.command));
}

static void test_active_transition_settle(void) {
    RekG1NativeLocomotionConfig config = test_config();
    RekG1NativeLocomotionState state = active_state(REK_G1_NATIVE_FORWARD);
    RekG1NativeLocomotionStepInput input = test_input(
        (RekG1NativeVelocityCommand){.forward = -1.0f});
    RekG1NativeLocomotionStepResult result;

    input.base_velocity.linear_velocity_local.forward = 0.02f;
    CHECK(step(&state, &config, &input, &result)
        == REK_G1_NATIVE_COMMAND_OK);
    CHECK(result.transition_check_performed == 1u);
    CHECK(result.transition_settled == 1u);
    CHECK(result.event == REK_G1_NATIVE_LOCOMOTION_EVENT_PLAY_ROUTE);
    CHECK(result.event_route_id == REK_G1_NATIVE_BACKWARD);
    CHECK(result.next_state.current_route_id == REK_G1_NATIVE_BACKWARD);

    state = active_state(REK_G1_NATIVE_FORWARD);
    input.base_velocity.linear_velocity_local.forward = 0.03f;
    CHECK(step(&state, &config, &input, &result)
        == REK_G1_NATIVE_COMMAND_OK);
    CHECK(result.transition_check_performed == 1u);
    CHECK(result.transition_settled == 0u);
    CHECK(result.event == REK_G1_NATIVE_LOCOMOTION_EVENT_PLAY_IDLE);
    CHECK(result.velocity_write == 1u);
    CHECK(velocity_equal(
        result.effective_velocity, (RekG1NativeVelocityCommand){0}));
    CHECK(result.next_state.locomotion_active == 0u);
    CHECK(result.next_state.transition_settling == 1u);
    CHECK(result.next_state.transition_from_route_id
        == REK_G1_NATIVE_FORWARD);
    CHECK(result.next_state.has_momentum == 0u);

    state = result.next_state;
    input.base_velocity.linear_velocity_local.forward = -0.031f;
    CHECK(step(&state, &config, &input, &result)
        == REK_G1_NATIVE_COMMAND_OK);
    CHECK(result.transition_check_performed == 1u);
    CHECK(result.transition_settled == 0u);
    CHECK(result.event == REK_G1_NATIVE_LOCOMOTION_EVENT_NONE);
    CHECK(result.velocity_write == 1u);
    CHECK(result.next_state.transition_settling == 1u);

    state = result.next_state;
    input.command = (RekG1NativeVelocityCommand){.strafe = 1.0f};
    input.base_velocity.linear_velocity_local.forward = 0.0f;
    CHECK(step(&state, &config, &input, &result)
        == REK_G1_NATIVE_COMMAND_OK);
    CHECK(result.transition_check_performed == 1u);
    CHECK(result.transition_settled == 1u);
    CHECK(result.event == REK_G1_NATIVE_LOCOMOTION_EVENT_PLAY_ROUTE);
    CHECK(result.event_route_id == REK_G1_NATIVE_STRAFE_LEFT);
    CHECK(result.next_state.transition_settling == 0u);
    CHECK(result.next_state.locomotion_active == 1u);
}

static void test_momentum_transition_gate(void) {
    RekG1NativeLocomotionConfig config = test_config();
    RekG1NativeLocomotionState state = {
        .current_route_id = REK_G1_NATIVE_FORWARD,
        .momentum_route_id = REK_G1_NATIVE_FORWARD,
        .has_momentum = 1u,
    };
    RekG1NativeLocomotionStepInput input = test_input(
        (RekG1NativeVelocityCommand){.forward = 1.0f});
    RekG1NativeLocomotionStepResult result;

    input.base_velocity.available = 0u;
    CHECK(step(&state, &config, &input, &result)
        == REK_G1_NATIVE_COMMAND_OK);
    CHECK(result.transition_check_performed == 0u);
    CHECK(result.event == REK_G1_NATIVE_LOCOMOTION_EVENT_PLAY_ROUTE);
    CHECK(result.next_state.has_momentum == 0u);

    state = (RekG1NativeLocomotionState){
        .current_route_id = REK_G1_NATIVE_FORWARD,
        .momentum_route_id = REK_G1_NATIVE_FORWARD,
        .has_momentum = 1u,
    };
    input = test_input(
        (RekG1NativeVelocityCommand){.forward = -1.0f});
    input.base_velocity.linear_velocity_local.forward = 0.2f;
    CHECK(step(&state, &config, &input, &result)
        == REK_G1_NATIVE_COMMAND_OK);
    CHECK(result.transition_check_performed == 1u);
    CHECK(result.transition_settled == 0u);
    CHECK(result.event == REK_G1_NATIVE_LOCOMOTION_EVENT_NONE);
    CHECK(result.next_state.transition_settling == 1u);
    CHECK(result.next_state.transition_from_route_id
        == REK_G1_NATIVE_FORWARD);
    CHECK(result.next_state.has_momentum == 0u);

    state = (RekG1NativeLocomotionState){0};
    input.base_velocity.available = 0u;
    CHECK(step(&state, &config, &input, &result)
        == REK_G1_NATIVE_COMMAND_OK);
    CHECK(result.transition_check_performed == 0u);
    CHECK(result.event == REK_G1_NATIVE_LOCOMOTION_EVENT_PLAY_ROUTE);
    CHECK(result.event_route_id == REK_G1_NATIVE_BACKWARD);
}

static void test_action_gate(void) {
    RekG1NativeLocomotionConfig config = test_config();
    RekG1NativeLocomotionState state = active_state(REK_G1_NATIVE_FORWARD);
    state.last_driven_command =
        (RekG1NativeVelocityCommand){.forward = 0.4f};
    RekG1NativeLocomotionStepInput input = test_input(
        (RekG1NativeVelocityCommand){
            .forward = 0.5f,
            .strafe = 0.2f,
            .yaw = REK_G1_NATIVE_COMMAND_EPSILON,
        });
    RekG1NativeLocomotionStepResult result;
    input.restrict_yaw = 1u;
    input.composer_action_playing = 1u;

    CHECK(step(&state, &config, &input, &result)
        == REK_G1_NATIVE_COMMAND_OK);
    CHECK(result.velocity_write == 1u);
    CHECK(result.effective_velocity.forward == 0.5f);
    CHECK(result.effective_velocity.strafe == 0.2f);
    CHECK(result.effective_velocity.yaw == 0.0f);
    CHECK(result.event == REK_G1_NATIVE_LOCOMOTION_EVENT_NONE);
    CHECK(result.next_state.last_driven_command.forward == 0.4f);

    input.command.yaw = 0.0009f;
    CHECK(step(&state, &config, &input, &result)
        == REK_G1_NATIVE_COMMAND_OK);
    CHECK(result.velocity_write == 0u);
    CHECK(result.effective_velocity.yaw == 0.0009f);

    input.composer_action_playing = 0u;
    input.command.yaw = 1.0f;
    CHECK(step(&state, &config, &input, &result)
        == REK_G1_NATIVE_COMMAND_OK);
    CHECK(result.effective_velocity.yaw == 1.0f);
}

static void test_stop_brake(void) {
    RekG1NativeLocomotionConfig config = test_config();
    RekG1NativeLocomotionState state = active_state(REK_G1_NATIVE_FORWARD);
    state.last_driven_command =
        (RekG1NativeVelocityCommand){.forward = 1.0f};
    RekG1NativeLocomotionStepInput input = test_input(
        (RekG1NativeVelocityCommand){0});
    RekG1NativeLocomotionStepResult result;

    CHECK(step(&state, &config, &input, &result)
        == REK_G1_NATIVE_COMMAND_OK);
    CHECK(result.event == REK_G1_NATIVE_LOCOMOTION_EVENT_NONE);
    CHECK(result.velocity_write == 1u);
    CHECK(near(result.effective_velocity.forward, 0.96f));
    CHECK(result.next_state.stop_braking == 1u);
    CHECK(result.next_state.locomotion_active == 1u);

    state = result.next_state;
    CHECK(step(&state, &config, &input, &result)
        == REK_G1_NATIVE_COMMAND_OK);
    CHECK(near(result.effective_velocity.forward, 0.92f));
    CHECK(result.next_state.stop_braking == 1u);

    state = active_state(REK_G1_NATIVE_FORWARD);
    state.last_driven_command =
        (RekG1NativeVelocityCommand){.forward = 1.0f};
    input.delta_seconds = 0.5f;
    CHECK(step(&state, &config, &input, &result)
        == REK_G1_NATIVE_COMMAND_OK);
    CHECK(result.event == REK_G1_NATIVE_LOCOMOTION_EVENT_PLAY_IDLE);
    CHECK(result.velocity_write == 0u);
    CHECK(result.next_state.locomotion_active == 0u);
    CHECK(result.next_state.stop_braking == 0u);
    CHECK(result.next_state.has_momentum == 1u);
    CHECK(result.next_state.momentum_route_id == REK_G1_NATIVE_FORWARD);

    state = active_state(REK_G1_NATIVE_FORWARD);
    state.last_driven_command =
        (RekG1NativeVelocityCommand){.forward = 0.1f};
    input.delta_seconds = 0.025f;
    CHECK(step(&state, &config, &input, &result)
        == REK_G1_NATIVE_COMMAND_OK);
    CHECK(result.event == REK_G1_NATIVE_LOCOMOTION_EVENT_PLAY_IDLE);
    CHECK(result.velocity_write == 0u);

    state = active_state(REK_G1_NATIVE_FORWARD);
    state.last_driven_command =
        (RekG1NativeVelocityCommand){.forward = 0.1001f};
    CHECK(step(&state, &config, &input, &result)
        == REK_G1_NATIVE_COMMAND_OK);
    CHECK(result.event == REK_G1_NATIVE_LOCOMOTION_EVENT_NONE);
    CHECK(result.velocity_write == 1u);
    CHECK(result.effective_velocity.forward
        > REK_G1_NATIVE_STOP_BRAKE_DONE_MAGNITUDE);

    state = active_state(REK_G1_NATIVE_FORWARD);
    state.last_driven_command = (RekG1NativeVelocityCommand){
        .forward = 3.0f,
        .strafe = 4.0f,
    };
    config.stop_brake_rate = 10.0f;
    input.delta_seconds = 0.1f;
    CHECK(step(&state, &config, &input, &result)
        == REK_G1_NATIVE_COMMAND_OK);
    CHECK(near(result.effective_velocity.forward, 2.4f));
    CHECK(near(result.effective_velocity.strafe, 3.2f));
    CHECK(near(result.effective_velocity.yaw, 0.0f));
}

static void test_stop_completion_and_idle_busy(void) {
    RekG1NativeLocomotionConfig config = test_config();
    config.stop_brake_rate = 0.0f;
    RekG1NativeLocomotionState state = active_state(REK_G1_NATIVE_TURN_LEFT);
    state.last_driven_command =
        (RekG1NativeVelocityCommand){.yaw = 1.0f};
    RekG1NativeLocomotionStepInput input = test_input(
        (RekG1NativeVelocityCommand){0});
    RekG1NativeLocomotionStepResult result;

    CHECK(step(&state, &config, &input, &result)
        == REK_G1_NATIVE_COMMAND_OK);
    CHECK(result.event == REK_G1_NATIVE_LOCOMOTION_EVENT_PLAY_IDLE);
    CHECK(result.next_state.locomotion_active == 0u);
    CHECK(result.next_state.has_momentum == 1u);
    CHECK(result.next_state.momentum_route_id == REK_G1_NATIVE_TURN_LEFT);

    state = (RekG1NativeLocomotionState){0};
    input.composer_busy = 1u;
    CHECK(step(&state, &config, &input, &result)
        == REK_G1_NATIVE_COMMAND_OK);
    CHECK(result.event == REK_G1_NATIVE_LOCOMOTION_EVENT_PLAY_IDLE);

    input.composer_busy = 0u;
    CHECK(step(&state, &config, &input, &result)
        == REK_G1_NATIVE_COMMAND_OK);
    CHECK(result.event == REK_G1_NATIVE_LOCOMOTION_EVENT_NONE);

    state = (RekG1NativeLocomotionState){
        .transition_from_route_id = REK_G1_NATIVE_FORWARD,
        .transition_settling = 1u,
    };
    CHECK(step(&state, &config, &input, &result)
        == REK_G1_NATIVE_COMMAND_OK);
    CHECK(result.next_state.transition_settling == 0u);
    CHECK(result.transition_check_performed == 0u);
}

static void test_fail_closed_and_transactional(void) {
    RekG1NativeLocomotionConfig config = test_config();
    RekG1NativeLocomotionState state = active_state(REK_G1_NATIVE_FORWARD);
    RekG1NativeLocomotionStepInput input = test_input(
        (RekG1NativeVelocityCommand){.forward = -1.0f});
    RekG1NativeLocomotionStepResult result;
    RekG1NativeLocomotionStepResult before;

#define CHECK_ERROR_UNCHANGED(expected_status) do { \
    memset(&result, 0xA5, sizeof(result)); \
    before = result; \
    CHECK(rek_g1_native_locomotion_step(&state, &config, &input, &result) \
        == (expected_status)); \
    CHECK(memcmp(&result, &before, sizeof(result)) == 0); \
} while (0)

    input.base_velocity.available = 0u;
    CHECK_ERROR_UNCHANGED(REK_G1_NATIVE_COMMAND_MEASUREMENT_UNAVAILABLE);

    config.transition_settle = 0u;
    CHECK(step(&state, &config, &input, &result)
        == REK_G1_NATIVE_COMMAND_OK);
    config = test_config();
    input = test_input(
        (RekG1NativeVelocityCommand){.forward = -1.0f});

    input.delta_seconds = -0.01f;
    CHECK_ERROR_UNCHANGED(REK_G1_NATIVE_COMMAND_TIMING_INVALID);
    input.delta_seconds = NAN;
    CHECK_ERROR_UNCHANGED(REK_G1_NATIVE_COMMAND_NON_FINITE);
    input = test_input(
        (RekG1NativeVelocityCommand){.forward = -1.0f});
    input.restrict_yaw = 2u;
    CHECK_ERROR_UNCHANGED(REK_G1_NATIVE_COMMAND_INPUT_INVALID);
    input = test_input(
        (RekG1NativeVelocityCommand){.forward = NAN});
    CHECK_ERROR_UNCHANGED(REK_G1_NATIVE_COMMAND_NON_FINITE);
    input = test_input(
        (RekG1NativeVelocityCommand){.forward = -1.0f});
    input.base_velocity.linear_velocity_local.forward = NAN;
    CHECK_ERROR_UNCHANGED(REK_G1_NATIVE_COMMAND_NON_FINITE);

    input = test_input(
        (RekG1NativeVelocityCommand){.forward = -1.0f});
    state.locomotion_active = 2u;
    CHECK_ERROR_UNCHANGED(REK_G1_NATIVE_COMMAND_STATE_INVALID);
    state = active_state(REK_G1_NATIVE_IDLE);
    CHECK_ERROR_UNCHANGED(REK_G1_NATIVE_COMMAND_STATE_INVALID);
    state = active_state(REK_G1_NATIVE_FORWARD);
    state.has_momentum = 1u;
    state.momentum_route_id = REK_G1_NATIVE_FORWARD;
    CHECK_ERROR_UNCHANGED(REK_G1_NATIVE_COMMAND_STATE_INVALID);
    state = active_state(REK_G1_NATIVE_FORWARD);
    state.last_driven_command.forward = NAN;
    CHECK_ERROR_UNCHANGED(REK_G1_NATIVE_COMMAND_STATE_INVALID);
    state = (RekG1NativeLocomotionState){0};
    state.current_route_id = (RekG1NativeRouteId)99;
    CHECK_ERROR_UNCHANGED(REK_G1_NATIVE_COMMAND_STATE_INVALID);

    state = active_state(REK_G1_NATIVE_FORWARD);
    config.stop_brake_rate = -1.0f;
    CHECK_ERROR_UNCHANGED(REK_G1_NATIVE_COMMAND_CONFIG_INVALID);
    config = test_config();
    config.settle_linear_speed = NAN;
    CHECK_ERROR_UNCHANGED(REK_G1_NATIVE_COMMAND_CONFIG_INVALID);
    config = test_config();
    config.transition_settle = 2u;
    CHECK_ERROR_UNCHANGED(REK_G1_NATIVE_COMMAND_CONFIG_INVALID);

    config = test_config();
    input = test_input((RekG1NativeVelocityCommand){0});
    state = active_state(REK_G1_NATIVE_FORWARD);
    state.last_driven_command.forward = 3.0e38f;
    CHECK_ERROR_UNCHANGED(REK_G1_NATIVE_COMMAND_NON_FINITE);

    CHECK(rek_g1_native_locomotion_step(NULL, &config, &input, &result)
        == REK_G1_NATIVE_COMMAND_NULL_ARGUMENT);
    CHECK(rek_g1_native_locomotion_step(&state, NULL, &input, &result)
        == REK_G1_NATIVE_COMMAND_NULL_ARGUMENT);
    CHECK(rek_g1_native_locomotion_step(&state, &config, NULL, &result)
        == REK_G1_NATIVE_COMMAND_NULL_ARGUMENT);
    CHECK(rek_g1_native_locomotion_step(&state, &config, &input, NULL)
        == REK_G1_NATIVE_COMMAND_NULL_ARGUMENT);

#undef CHECK_ERROR_UNCHANGED
}

int main(void) {
    test_transition_settled_components();
    test_route_start_hold_and_direct_change();
    test_active_transition_settle();
    test_momentum_transition_gate();
    test_action_gate();
    test_stop_brake();
    test_stop_completion_and_idle_busy();
    test_fail_closed_and_transactional();
    printf(
        "native locomotion state tests: %d checks, %d failures\n",
        checks,
        failures);
    return failures == 0 ? 0 : 1;
}
