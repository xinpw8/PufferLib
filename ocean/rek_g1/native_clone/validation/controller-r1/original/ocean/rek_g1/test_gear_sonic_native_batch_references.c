#include "gear_sonic_native_batch.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define TEST_BATCH_SIZE 2
#define TEST_MOTION_FRAMES 67

static int checks_run = 0;
static int checks_failed = 0;

#define CHECK(condition) do { \
    checks_run++; \
    if (!(condition)) { \
        fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); \
        checks_failed++; \
    } \
} while (0)

int gear_sonic_ort_encode(
        GearSonicOrtBatch* batch,
        float* observations,
        float* tokens,
        char* error,
        size_t error_capacity) {
    for (size_t row = 0; row < batch->batch_size; row++) {
        for (size_t column = 0;
                column < GEAR_SONIC_ENCODER_OUTPUT_WIDTH;
                column++) {
            size_t source = (column * 23 + 7) % GEAR_SONIC_ENCODER_INPUT_WIDTH;
            tokens[row * GEAR_SONIC_ENCODER_OUTPUT_WIDTH + column]
                = observations[row * GEAR_SONIC_ENCODER_INPUT_WIDTH + source]
                + (float)column * 0.003125f;
        }
    }
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    return 1;
}

int gear_sonic_ort_decode(
        GearSonicOrtBatch* batch,
        float* observations,
        float* actions,
        char* error,
        size_t error_capacity) {
    for (size_t row = 0; row < batch->batch_size; row++) {
        for (size_t column = 0;
                column < GEAR_SONIC_DECODER_OUTPUT_WIDTH;
                column++) {
            const size_t source = (column * 31 + 11)
                % GEAR_SONIC_DECODER_INPUT_WIDTH;
            actions[row * GEAR_SONIC_DECODER_OUTPUT_WIDTH + column]
                = observations[row * GEAR_SONIC_DECODER_INPUT_WIDTH + source]
                * 0.25f + (float)((int)column - 14) * 9.0f;
        }
    }
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    return 1;
}

static void fill_motion(
        float positions[TEST_MOTION_FRAMES * GEAR_SONIC_ACTION_DIM],
        float rotations[TEST_MOTION_FRAMES * 4]) {
    for (size_t frame = 0; frame < TEST_MOTION_FRAMES; frame++) {
        for (size_t joint = 0; joint < GEAR_SONIC_ACTION_DIM; joint++) {
            positions[frame * GEAR_SONIC_ACTION_DIM + joint]
                = (float)((int)(frame * 37 + joint * 13) - 900) * 0.0007f;
        }
        rotations[frame * 4 + 0] = (float)frame * 0.0001f;
        rotations[frame * 4 + 1] = (float)frame * -0.0002f;
        rotations[frame * 4 + 2] = (float)frame * 0.0003f;
        rotations[frame * 4 + 3] = 1.0f;
    }
}

static void fill_state(
        double base_quaternion[TEST_BATCH_SIZE * 4],
        double angular_velocity[TEST_BATCH_SIZE * 3],
        double joint_position[TEST_BATCH_SIZE * GEAR_SONIC_ACTION_DIM],
        double joint_velocity[TEST_BATCH_SIZE * GEAR_SONIC_ACTION_DIM],
        double heading_delta[TEST_BATCH_SIZE * 4],
        int tick) {
    for (size_t row = 0; row < TEST_BATCH_SIZE; row++) {
        base_quaternion[row * 4 + 0] = 1.0;
        base_quaternion[row * 4 + 1] = 0.001 * (double)(row + 1);
        base_quaternion[row * 4 + 2] = -0.002 * (double)(tick + 1);
        base_quaternion[row * 4 + 3] = 0.003 * (double)(row + tick + 1);
        heading_delta[row * 4 + 0] = 1.0;
        heading_delta[row * 4 + 1] = -0.004 * (double)(row + 1);
        heading_delta[row * 4 + 2] = 0.005 * (double)(tick + 1);
        heading_delta[row * 4 + 3] = 0.006 * (double)(row + tick + 1);
        for (size_t axis = 0; axis < 3; axis++) {
            angular_velocity[row * 3 + axis]
                = 0.01 * (double)((int)(row * 3 + axis) - tick);
        }
        for (size_t joint = 0; joint < GEAR_SONIC_ACTION_DIM; joint++) {
            joint_position[row * GEAR_SONIC_ACTION_DIM + joint]
                = 0.02 * (double)((int)joint - 8) + 0.001 * (double)tick;
            joint_velocity[row * GEAR_SONIC_ACTION_DIM + joint]
                = -0.03 * (double)((int)joint - 5) + 0.002 * (double)row;
        }
    }
}

static void compose_references(
        const float positions[TEST_MOTION_FRAMES * GEAR_SONIC_ACTION_DIM],
        const float rotations[TEST_MOTION_FRAMES * 4],
        const size_t current_frames[TEST_BATCH_SIZE],
        float reference_positions[
            TEST_BATCH_SIZE * GEAR_SONIC_HISTORY_FRAMES
            * GEAR_SONIC_ACTION_DIM],
        float reference_next_positions[
            TEST_BATCH_SIZE * GEAR_SONIC_HISTORY_FRAMES
            * GEAR_SONIC_ACTION_DIM],
        float reference_rotations[
            TEST_BATCH_SIZE * GEAR_SONIC_HISTORY_FRAMES * 4]) {
    for (size_t row = 0; row < TEST_BATCH_SIZE; row++) {
        for (size_t future = 0; future < GEAR_SONIC_HISTORY_FRAMES; future++) {
            const size_t frame = (current_frames[row] + future * 5)
                % TEST_MOTION_FRAMES;
            const size_t next = (frame + 1) % TEST_MOTION_FRAMES;
            const size_t reference_row = row * GEAR_SONIC_HISTORY_FRAMES + future;
            memcpy(
                reference_positions
                    + reference_row * GEAR_SONIC_ACTION_DIM,
                positions + frame * GEAR_SONIC_ACTION_DIM,
                GEAR_SONIC_ACTION_DIM * sizeof(float));
            memcpy(
                reference_next_positions
                    + reference_row * GEAR_SONIC_ACTION_DIM,
                positions + next * GEAR_SONIC_ACTION_DIM,
                GEAR_SONIC_ACTION_DIM * sizeof(float));
            memcpy(
                reference_rotations + reference_row * 4,
                rotations + frame * 4,
                4 * sizeof(float));
        }
    }
}

static void check_batch_equal(
        const GearSonicNativeBatch* expected,
        const GearSonicNativeBatch* actual) {
    const size_t batch = expected->batch_size;
    const size_t history_rows = batch * GEAR_SONIC_HISTORY_FRAMES;
    CHECK(actual->batch_size == batch);
    CHECK(memcmp(
        expected->encoder_observations,
        actual->encoder_observations,
        batch * GEAR_SONIC_ENCODER_INPUT_WIDTH * sizeof(float)) == 0);
    CHECK(memcmp(
        expected->tokens,
        actual->tokens,
        batch * GEAR_SONIC_ENCODER_OUTPUT_WIDTH * sizeof(float)) == 0);
    CHECK(memcmp(
        expected->decoder_observations,
        actual->decoder_observations,
        batch * GEAR_SONIC_DECODER_INPUT_WIDTH * sizeof(float)) == 0);
    CHECK(memcmp(
        expected->actions_policy,
        actual->actions_policy,
        batch * GEAR_SONIC_ACTION_DIM * sizeof(float)) == 0);
    CHECK(memcmp(
        expected->clipped_actions_policy,
        actual->clipped_actions_policy,
        batch * GEAR_SONIC_ACTION_DIM * sizeof(float)) == 0);
    CHECK(memcmp(
        expected->targets_mujoco,
        actual->targets_mujoco,
        batch * GEAR_SONIC_ACTION_DIM * sizeof(float)) == 0);
    CHECK(memcmp(
        expected->history_base_quaternion_wxyz,
        actual->history_base_quaternion_wxyz,
        history_rows * 4 * sizeof(float)) == 0);
    CHECK(memcmp(
        expected->history_base_angular_velocity,
        actual->history_base_angular_velocity,
        history_rows * 3 * sizeof(float)) == 0);
    CHECK(memcmp(
        expected->history_joint_position_policy,
        actual->history_joint_position_policy,
        history_rows * GEAR_SONIC_ACTION_DIM * sizeof(float)) == 0);
    CHECK(memcmp(
        expected->history_joint_velocity_policy,
        actual->history_joint_velocity_policy,
        history_rows * GEAR_SONIC_ACTION_DIM * sizeof(float)) == 0);
    CHECK(memcmp(
        expected->history_last_action_policy,
        actual->history_last_action_policy,
        history_rows * GEAR_SONIC_ACTION_DIM * sizeof(float)) == 0);
    CHECK(memcmp(
        expected->history_count,
        actual->history_count,
        batch * sizeof(uint8_t)) == 0);
    CHECK(memcmp(
        expected->history_head,
        actual->history_head,
        batch * sizeof(uint8_t)) == 0);
    CHECK(actual->failed == expected->failed);
}

int main(void) {
    float positions[TEST_MOTION_FRAMES * GEAR_SONIC_ACTION_DIM];
    float rotations[TEST_MOTION_FRAMES * 4];
    double base_quaternion[TEST_BATCH_SIZE * 4];
    double angular_velocity[TEST_BATCH_SIZE * 3];
    double joint_position[TEST_BATCH_SIZE * GEAR_SONIC_ACTION_DIM];
    double joint_velocity[TEST_BATCH_SIZE * GEAR_SONIC_ACTION_DIM];
    double heading_delta[TEST_BATCH_SIZE * 4];
    size_t current_frames[TEST_BATCH_SIZE] = {3, 29};
    float reference_positions[
        TEST_BATCH_SIZE * GEAR_SONIC_HISTORY_FRAMES * GEAR_SONIC_ACTION_DIM];
    float reference_next_positions[
        TEST_BATCH_SIZE * GEAR_SONIC_HISTORY_FRAMES * GEAR_SONIC_ACTION_DIM];
    float reference_rotations[
        TEST_BATCH_SIZE * GEAR_SONIC_HISTORY_FRAMES * 4];
    char error[256];

    fill_motion(positions, rotations);
    fill_state(
        base_quaternion,
        angular_velocity,
        joint_position,
        joint_velocity,
        heading_delta,
        0);
    compose_references(
        positions,
        rotations,
        current_frames,
        reference_positions,
        reference_next_positions,
        reference_rotations);

    const GearSonicNativeMotion motion = {
        .dof_position_mujoco = positions,
        .root_position_m = NULL,
        .root_rotation_xyzw = rotations,
        .frames = TEST_MOTION_FRAMES,
        .loop = 1,
    };
    const GearSonicNativeReferenceInput references = {
        .dof_position_mujoco = reference_positions,
        .dof_next_position_mujoco = reference_next_positions,
        .root_rotation_xyzw = reference_rotations,
    };
    const GearSonicNativeStateInput legacy_state = {
        .base_quaternion_wxyz = base_quaternion,
        .base_angular_velocity_local = angular_velocity,
        .joint_position_mujoco = joint_position,
        .joint_velocity_mujoco = joint_velocity,
        .heading_delta_wxyz = heading_delta,
        .reference_frames = current_frames,
    };
    const GearSonicNativeStateInput reference_state = {
        .base_quaternion_wxyz = base_quaternion,
        .base_angular_velocity_local = angular_velocity,
        .joint_position_mujoco = joint_position,
        .joint_velocity_mujoco = joint_velocity,
        .heading_delta_wxyz = heading_delta,
        .reference_frames = NULL,
    };
    GearSonicOrtBatch ort = {.batch_size = TEST_BATCH_SIZE};
    GearSonicNativeBatch legacy_batch;
    GearSonicNativeBatch reference_batch;
    CHECK(gear_sonic_native_batch_open(
        &legacy_batch, TEST_BATCH_SIZE, error, sizeof(error)) == 1);
    CHECK(gear_sonic_native_batch_open(
        &reference_batch, TEST_BATCH_SIZE, error, sizeof(error)) == 1);
    CHECK(gear_sonic_native_batch_step(
        &legacy_batch, &ort, &motion, &legacy_state, error, sizeof(error)) == 1);
    CHECK(gear_sonic_native_batch_step_references(
        &reference_batch,
        &ort,
        &references,
        &reference_state,
        error,
        sizeof(error)) == 1);
    CHECK(error[0] == '\0');
    check_batch_equal(&legacy_batch, &reference_batch);
    CHECK(memcmp(
        reference_batch.encoder_observations,
        reference_batch.encoder_observations + GEAR_SONIC_ENCODER_INPUT_WIDTH,
        GEAR_SONIC_ENCODER_INPUT_WIDTH * sizeof(float)) != 0);
    for (size_t row = 0; row < TEST_BATCH_SIZE; row++) {
        for (size_t column = 0; column < 4u; column++) {
            CHECK(reference_batch.encoder_observations[
                row * GEAR_SONIC_ENCODER_INPUT_WIDTH + column] == 0.0f);
        }
    }

    current_frames[0] = 11;
    current_frames[1] = 47;
    fill_state(
        base_quaternion,
        angular_velocity,
        joint_position,
        joint_velocity,
        heading_delta,
        1);
    compose_references(
        positions,
        rotations,
        current_frames,
        reference_positions,
        reference_next_positions,
        reference_rotations);
    CHECK(gear_sonic_native_batch_step(
        &legacy_batch, &ort, &motion, &legacy_state, error, sizeof(error)) == 1);
    CHECK(gear_sonic_native_batch_step_references(
        &reference_batch,
        &ort,
        &references,
        &reference_state,
        error,
        sizeof(error)) == 1);
    check_batch_equal(&legacy_batch, &reference_batch);

    GearSonicNativeBatch failure_batch;
    CHECK(gear_sonic_native_batch_open(
        &failure_batch, TEST_BATCH_SIZE, error, sizeof(error)) == 1);
    GearSonicOrtBatch wrong_ort = {.batch_size = 1};
    CHECK(gear_sonic_native_batch_step_references(
        &failure_batch,
        &wrong_ort,
        &references,
        &reference_state,
        error,
        sizeof(error)) == 0);
    CHECK(failure_batch.failed == 0);
    CHECK(strstr(error, "invalid or failed state") != NULL);
    CHECK(gear_sonic_native_batch_step_references(
        &failure_batch,
        &ort,
        NULL,
        &reference_state,
        error,
        sizeof(error)) == 0);
    CHECK(failure_batch.failed == 0);

    GearSonicNativeBatch incomplete_batch = {
        .batch_size = 1,
        .initialized = 1,
    };
    GearSonicOrtBatch single_ort = {.batch_size = 1};
    CHECK(gear_sonic_native_batch_step_references(
        &incomplete_batch,
        &single_ort,
        &references,
        &reference_state,
        error,
        sizeof(error)) == 0);
    CHECK(incomplete_batch.failed == 0);
    CHECK(strstr(error, "invalid or failed state") != NULL);

    float dummy_float = 0.0f;
    double dummy_double = 0.0;
    uint8_t dummy_byte = 0;
    const size_t oversized_batch_size =
        SIZE_MAX
        / (GEAR_SONIC_HISTORY_FRAMES * GEAR_SONIC_ACTION_DIM * sizeof(float))
        + 1;
    GearSonicNativeBatch oversized_batch = {
        .batch_size = oversized_batch_size,
        .encoder_observations = &dummy_float,
        .tokens = &dummy_float,
        .decoder_observations = &dummy_float,
        .actions_policy = &dummy_float,
        .clipped_actions_policy = &dummy_float,
        .targets_mujoco = &dummy_float,
        .history_base_quaternion_wxyz = &dummy_float,
        .history_base_angular_velocity = &dummy_float,
        .history_joint_position_policy = &dummy_float,
        .history_joint_velocity_policy = &dummy_float,
        .history_last_action_policy = &dummy_float,
        .history_count = &dummy_byte,
        .history_head = &dummy_byte,
        .initialized = 1,
    };
    GearSonicOrtBatch oversized_ort = {.batch_size = oversized_batch_size};
    const GearSonicNativeReferenceInput oversized_references = {
        .dof_position_mujoco = &dummy_float,
        .dof_next_position_mujoco = &dummy_float,
        .root_rotation_xyzw = &dummy_float,
    };
    const GearSonicNativeStateInput oversized_state = {
        .base_quaternion_wxyz = &dummy_double,
        .base_angular_velocity_local = &dummy_double,
        .joint_position_mujoco = &dummy_double,
        .joint_velocity_mujoco = &dummy_double,
        .heading_delta_wxyz = &dummy_double,
        .reference_frames = NULL,
    };
    CHECK(gear_sonic_native_batch_step_references(
        &oversized_batch,
        &oversized_ort,
        &oversized_references,
        &oversized_state,
        error,
        sizeof(error)) == 0);
    CHECK(oversized_batch.failed == 1);
    CHECK(strstr(error, "size overflow") != NULL);

    const float saved = reference_positions[17];
    reference_positions[17] = NAN;
    CHECK(gear_sonic_native_batch_step_references(
        &failure_batch,
        &ort,
        &references,
        &reference_state,
        error,
        sizeof(error)) == 0);
    CHECK(failure_batch.failed == 1);
    CHECK(strstr(error, "non-finite input") != NULL);
    reference_positions[17] = saved;
    CHECK(gear_sonic_native_batch_step_references(
        &failure_batch,
        &ort,
        &references,
        &reference_state,
        error,
        sizeof(error)) == 0);
    CHECK(strstr(error, "invalid or failed state") != NULL);

    gear_sonic_native_batch_close(&failure_batch);
    gear_sonic_native_batch_close(&reference_batch);
    gear_sonic_native_batch_close(&legacy_batch);
    printf("native reference tests: %d checks, %d failures\n", checks_run, checks_failed);
    return checks_failed == 0 ? 0 : 1;
}
