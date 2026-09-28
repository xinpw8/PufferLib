#include "native_locomotion_command.h"

#include <math.h>
#include <stdio.h>

static int checks;
static int failures;

#define CHECK(condition) do { \
    checks++; \
    if (!(condition)) { \
        failures++; \
        fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #condition); \
    } \
} while (0)

static void check_route(
        RekG1NativeVelocityCommand command,
        RekG1NativeRouteId expected,
        int active) {
    RekG1NativeRouteSelection selection = {0};
    CHECK(rek_g1_native_select_locomotion_route(command, &selection)
        == REK_G1_NATIVE_COMMAND_OK);
    CHECK(selection.route_id == expected);
    CHECK(selection.locomotion_active == active);
}

static void test_route_selection(void) {
    check_route((RekG1NativeVelocityCommand){0}, REK_G1_NATIVE_IDLE, 0);
    check_route(
        (RekG1NativeVelocityCommand){.forward = 0.000999f},
        REK_G1_NATIVE_IDLE,
        0);
    check_route(
        (RekG1NativeVelocityCommand){.forward = 0.001f},
        REK_G1_NATIVE_FORWARD,
        1);
    check_route(
        (RekG1NativeVelocityCommand){.forward = -1.0f},
        REK_G1_NATIVE_BACKWARD,
        1);
    check_route(
        (RekG1NativeVelocityCommand){.strafe = 1.0f},
        REK_G1_NATIVE_STRAFE_LEFT,
        1);
    check_route(
        (RekG1NativeVelocityCommand){.strafe = -1.0f},
        REK_G1_NATIVE_STRAFE_RIGHT,
        1);
    check_route(
        (RekG1NativeVelocityCommand){.yaw = 1.0f},
        REK_G1_NATIVE_TURN_LEFT,
        1);
    check_route(
        (RekG1NativeVelocityCommand){.yaw = -1.0f},
        REK_G1_NATIVE_TURN_RIGHT,
        1);

    /* Planar motion wins over yaw. Equal planar magnitudes prefer forward. */
    check_route(
        (RekG1NativeVelocityCommand){.forward = 0.25f, .yaw = -1.0f},
        REK_G1_NATIVE_FORWARD,
        1);
    check_route(
        (RekG1NativeVelocityCommand){.strafe = -0.25f, .yaw = 1.0f},
        REK_G1_NATIVE_STRAFE_RIGHT,
        1);
    check_route(
        (RekG1NativeVelocityCommand){.forward = -1.0f, .strafe = 1.0f},
        REK_G1_NATIVE_BACKWARD,
        1);
    check_route(
        (RekG1NativeVelocityCommand){.forward = 0.25f, .strafe = 0.5f},
        REK_G1_NATIVE_STRAFE_LEFT,
        1);
}

static RekG1NativeCommandConfig config(void) {
    return (RekG1NativeCommandConfig){
        .locomotion_speed_scale = 1.0f,
        .command_yaw_rate_scale = 1.0f,
        .heading_yaw_rate_scale = 1.0f,
        .controller_rate_hz = 50,
    };
}

static void test_playback_update(void) {
    RekG1NativeCommandConfig value = config();
    RekG1NativePlaybackUpdate update = {0};
    CHECK(rek_g1_native_playback_update(
        (RekG1NativeVelocityCommand){0}, &value, &update)
        == REK_G1_NATIVE_COMMAND_OK);
    CHECK(update.apply == 1);
    CHECK(update.command_magnitude == 0.0f);
    CHECK(update.scale == 1.0f);

    CHECK(rek_g1_native_playback_update(
        (RekG1NativeVelocityCommand){.forward = 1.0f, .yaw = 1.0f},
        &value,
        &update) == REK_G1_NATIVE_COMMAND_OK);
    CHECK(update.apply == 1);
    CHECK(update.command_magnitude == sqrtf(2.0f));
    CHECK(update.scale == sqrtf(2.0f));

    value.locomotion_speed_scale = 0.5f;
    CHECK(rek_g1_native_playback_update(
        (RekG1NativeVelocityCommand){.strafe = -1.0f},
        &value,
        &update) == REK_G1_NATIVE_COMMAND_OK);
    CHECK(update.scale == 0.5f);

    value.locomotion_speed_scale = 0.0f;
    update = (RekG1NativePlaybackUpdate){.apply = 1, .scale = 17.0f};
    CHECK(rek_g1_native_playback_update(
        (RekG1NativeVelocityCommand){.forward = 1.0f},
        &value,
        &update) == REK_G1_NATIVE_COMMAND_OK);
    CHECK(update.apply == 0);
    CHECK(update.scale == 0.0f);
}

static void test_heading_update(void) {
    RekG1NativeCommandConfig value = config();
    RekG1NativeHeadingUpdate update = {0};
    CHECK(rek_g1_native_heading_update(
        (RekG1NativeVelocityCommand){.yaw = 1.0f},
        &value,
        0.0f,
        0.03f,
        0.01f,
        &update) == REK_G1_NATIVE_COMMAND_OK);
    CHECK(fabsf(update.clip_delta_radians - 0.03f) < 1e-7f);
    CHECK(fabsf(update.command_delta_radians - 0.02f) < 1e-7f);
    CHECK(fabsf(update.total_delta_radians - 0.06f) < 1e-7f);

    CHECK(rek_g1_native_heading_update(
        (RekG1NativeVelocityCommand){.yaw = -1.0f},
        &value,
        1.0f,
        0.03f,
        0.0f,
        &update) == REK_G1_NATIVE_COMMAND_OK);
    CHECK(update.command_delta_radians == 0.0f);
    CHECK(fabsf(update.total_delta_radians - 0.03f) < 1e-7f);
}

static void test_fail_closed(void) {
    RekG1NativeCommandConfig value = config();
    RekG1NativeRouteSelection selection = {0};
    RekG1NativePlaybackUpdate playback = {0};
    RekG1NativeHeadingUpdate heading = {0};
    CHECK(rek_g1_native_select_locomotion_route(
        (RekG1NativeVelocityCommand){.forward = NAN}, &selection)
        == REK_G1_NATIVE_COMMAND_NON_FINITE);
    CHECK(rek_g1_native_select_locomotion_route(
        (RekG1NativeVelocityCommand){0}, NULL)
        == REK_G1_NATIVE_COMMAND_NULL_ARGUMENT);
    value.controller_rate_hz = 0;
    CHECK(rek_g1_native_playback_update(
        (RekG1NativeVelocityCommand){0}, &value, &playback)
        == REK_G1_NATIVE_COMMAND_CONFIG_INVALID);
    value = config();
    CHECK(rek_g1_native_heading_update(
        (RekG1NativeVelocityCommand){0},
        &value,
        1.01f,
        0.0f,
        0.0f,
        &heading) == REK_G1_NATIVE_COMMAND_HEADING_OWNERSHIP_INVALID);
    CHECK(rek_g1_native_heading_update(
        (RekG1NativeVelocityCommand){0},
        &value,
        0.0f,
        NAN,
        0.0f,
        &heading) == REK_G1_NATIVE_COMMAND_NON_FINITE);
}

int main(void) {
    test_route_selection();
    test_playback_update();
    test_heading_update();
    test_fail_closed();
    printf("native locomotion command tests: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
