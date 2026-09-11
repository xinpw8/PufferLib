#include "gear_sonic_native_batch.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GEAR_SONIC_CONTROL_DT 0.02
#define GEAR_SONIC_ACTION_CLIP 100.0f

static const uint8_t GEAR_SONIC_ISAACLAB_TO_MUJOCO[GEAR_SONIC_ACTION_DIM] = {
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
    11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
};

static const uint8_t GEAR_SONIC_MUJOCO_TO_ISAACLAB[GEAR_SONIC_ACTION_DIM] = {
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
    16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
};

static const float GEAR_SONIC_DEFAULT_ANGLES_MUJOCO[GEAR_SONIC_ACTION_DIM] = {
    -0.312f, 0.0f, 0.0f, 0.669f, -0.363f, 0.0f,
    -0.312f, 0.0f, 0.0f, 0.669f, -0.363f, 0.0f,
    0.0f, 0.0f, 0.0f,
    0.2f, 0.2f, 0.0f, 0.6f, 0.0f, 0.0f, 0.0f,
    0.2f, -0.2f, 0.0f, 0.6f, 0.0f, 0.0f, 0.0f,
};

static const float GEAR_SONIC_ACTION_SCALE_MUJOCO[GEAR_SONIC_ACTION_DIM] = {
    0.350661f, 0.350661f, 0.547546f, 0.350661f, 0.438577f, 0.438577f,
    0.350661f, 0.350661f, 0.547546f, 0.350661f, 0.438577f, 0.438577f,
    0.547546f, 0.438577f, 0.438577f,
    0.438577f, 0.438577f, 0.438577f, 0.438577f, 0.438577f, 0.074501f, 0.074501f,
    0.438577f, 0.438577f, 0.438577f, 0.438577f, 0.438577f, 0.074501f, 0.074501f,
};

static void set_error(
        char* error, size_t capacity, const char* operation, const char* detail) {
    if (error == NULL || capacity == 0) return;
    snprintf(error, capacity, "%s: %s", operation, detail);
    error[capacity - 1] = '\0';
}

static int checked_product(size_t left, size_t right, size_t* result) {
    if (left != 0 && right > SIZE_MAX / left) return 0;
    *result = left * right;
    return 1;
}

static int byte_count_fits(size_t count, size_t element_size) {
    return element_size != 0 && count <= SIZE_MAX / element_size;
}

static int allocate_floats(float** destination, size_t count) {
    if (count > SIZE_MAX / sizeof(float)) return 0;
    *destination = (float*)calloc(count, sizeof(float));
    return *destination != NULL;
}

static int finite_doubles(const double* values, size_t count) {
    if (values == NULL) return 0;
    for (size_t index = 0; index < count; index++) {
        if (!isfinite(values[index])) return 0;
    }
    return 1;
}

static int finite_floats(const float* values, size_t count) {
    if (values == NULL) return 0;
    for (size_t index = 0; index < count; index++) {
        if (!isfinite(values[index])) return 0;
    }
    return 1;
}

static int batch_buffers_ready(const GearSonicNativeBatch* batch) {
    return batch != NULL
        && batch->initialized
        && batch->batch_size > 0
        && batch->encoder_observations != NULL
        && batch->tokens != NULL
        && batch->decoder_observations != NULL
        && batch->actions_policy != NULL
        && batch->clipped_actions_policy != NULL
        && batch->targets_mujoco != NULL
        && batch->history_base_quaternion_wxyz != NULL
        && batch->history_base_angular_velocity != NULL
        && batch->history_joint_position_policy != NULL
        && batch->history_joint_velocity_policy != NULL
        && batch->history_last_action_policy != NULL
        && batch->history_count != NULL
        && batch->history_head != NULL;
}

static void quaternion_conjugate(const double source[4], double destination[4]) {
    destination[0] = source[0];
    destination[1] = -source[1];
    destination[2] = -source[2];
    destination[3] = -source[3];
}

static void quaternion_multiply(
        const double left[4], const double right[4], double destination[4]) {
    const double lw = left[0];
    const double lx = left[1];
    const double ly = left[2];
    const double lz = left[3];
    const double rw = right[0];
    const double rx = right[1];
    const double ry = right[2];
    const double rz = right[3];
    destination[0] = lw * rw - lx * rx - ly * ry - lz * rz;
    destination[1] = lw * rx + lx * rw + ly * rz - lz * ry;
    destination[2] = lw * ry - lx * rz + ly * rw + lz * rx;
    destination[3] = lw * rz + lx * ry - ly * rx + lz * rw;
}

static void rotation_first_two_columns(const double q[4], float output[6]) {
    const double w = q[0];
    const double x = q[1];
    const double y = q[2];
    const double z = q[3];
    output[0] = (float)(1.0 - 2.0 * (y * y + z * z));
    output[1] = (float)(2.0 * (x * y - z * w));
    output[2] = (float)(2.0 * (x * y + z * w));
    output[3] = (float)(1.0 - 2.0 * (x * x + z * z));
    output[4] = (float)(2.0 * (x * z - y * w));
    output[5] = (float)(2.0 * (y * z + x * w));
}

static void projected_gravity(const float quaternion[4], float output[3]) {
    double q[4] = {
        (double)quaternion[0],
        -(double)quaternion[1],
        -(double)quaternion[2],
        -(double)quaternion[3],
    };
    const double vector[3] = {0.0, 0.0, -1.0};
    const double qw = q[0];
    const double qx = q[1];
    const double qy = q[2];
    const double qz = q[3];
    const double scale = 2.0 * qw * qw - 1.0;
    const double cross[3] = {
        qy * vector[2] - qz * vector[1],
        qz * vector[0] - qx * vector[2],
        qx * vector[1] - qy * vector[0],
    };
    const double dot = qx * vector[0] + qy * vector[1] + qz * vector[2];
    output[0] = (float)(vector[0] * scale + cross[0] * qw * 2.0 + qx * dot * 2.0);
    output[1] = (float)(vector[1] * scale + cross[1] * qw * 2.0 + qy * dot * 2.0);
    output[2] = (float)(vector[2] * scale + cross[2] * qw * 2.0 + qz * dot * 2.0);
}

void gear_sonic_native_batch_close(GearSonicNativeBatch* batch) {
    if (batch == NULL) return;
    free(batch->history_head);
    free(batch->history_count);
    free(batch->history_last_action_policy);
    free(batch->history_joint_velocity_policy);
    free(batch->history_joint_position_policy);
    free(batch->history_base_angular_velocity);
    free(batch->history_base_quaternion_wxyz);
    free(batch->targets_mujoco);
    free(batch->clipped_actions_policy);
    free(batch->actions_policy);
    free(batch->decoder_observations);
    free(batch->tokens);
    free(batch->encoder_observations);
    memset(batch, 0, sizeof(*batch));
}

int gear_sonic_native_batch_open(
        GearSonicNativeBatch* batch,
        size_t batch_size,
        char* error,
        size_t error_capacity) {
    if (batch == NULL || batch_size == 0) {
        set_error(error, error_capacity, "open native batch", "invalid argument");
        return 0;
    }
    memset(batch, 0, sizeof(*batch));
    batch->batch_size = batch_size;
    size_t encoder_count = 0;
    size_t token_count = 0;
    size_t decoder_count = 0;
    size_t action_count = 0;
    size_t history_rows = 0;
    size_t history_quaternion_count = 0;
    size_t history_angular_count = 0;
    size_t history_action_count = 0;
    if (!checked_product(batch_size, GEAR_SONIC_ENCODER_INPUT_WIDTH, &encoder_count)
            || !checked_product(batch_size, GEAR_SONIC_ENCODER_OUTPUT_WIDTH, &token_count)
            || !checked_product(batch_size, GEAR_SONIC_DECODER_INPUT_WIDTH, &decoder_count)
            || !checked_product(batch_size, GEAR_SONIC_ACTION_DIM, &action_count)
            || !checked_product(batch_size, GEAR_SONIC_HISTORY_FRAMES, &history_rows)
            || !checked_product(history_rows, 4, &history_quaternion_count)
            || !checked_product(history_rows, 3, &history_angular_count)
            || !checked_product(
                history_rows, GEAR_SONIC_ACTION_DIM, &history_action_count)
            || !allocate_floats(&batch->encoder_observations, encoder_count)
            || !allocate_floats(&batch->tokens, token_count)
            || !allocate_floats(&batch->decoder_observations, decoder_count)
            || !allocate_floats(&batch->actions_policy, action_count)
            || !allocate_floats(&batch->clipped_actions_policy, action_count)
            || !allocate_floats(&batch->targets_mujoco, action_count)
            || !allocate_floats(
                &batch->history_base_quaternion_wxyz, history_quaternion_count)
            || !allocate_floats(
                &batch->history_base_angular_velocity, history_angular_count)
            || !allocate_floats(
                &batch->history_joint_position_policy, history_action_count)
            || !allocate_floats(
                &batch->history_joint_velocity_policy, history_action_count)
            || !allocate_floats(
                &batch->history_last_action_policy, history_action_count)) {
        set_error(error, error_capacity, "open native batch", "allocation overflow or failure");
        gear_sonic_native_batch_close(batch);
        return 0;
    }
    batch->history_count = (uint8_t*)calloc(batch_size, sizeof(uint8_t));
    batch->history_head = (uint8_t*)calloc(batch_size, sizeof(uint8_t));
    if (batch->history_count == NULL || batch->history_head == NULL) {
        set_error(error, error_capacity, "open native batch", "allocation failure");
        gear_sonic_native_batch_close(batch);
        return 0;
    }
    batch->initialized = 1;
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    return 1;
}

void gear_sonic_native_batch_reset(GearSonicNativeBatch* batch) {
    if (!batch_buffers_ready(batch)) return;
    size_t history_rows = batch->batch_size * GEAR_SONIC_HISTORY_FRAMES;
    memset(
        batch->history_base_quaternion_wxyz,
        0,
        history_rows * 4 * sizeof(float));
    memset(
        batch->history_base_angular_velocity,
        0,
        history_rows * 3 * sizeof(float));
    memset(
        batch->history_joint_position_policy,
        0,
        history_rows * GEAR_SONIC_ACTION_DIM * sizeof(float));
    memset(
        batch->history_joint_velocity_policy,
        0,
        history_rows * GEAR_SONIC_ACTION_DIM * sizeof(float));
    memset(
        batch->history_last_action_policy,
        0,
        history_rows * GEAR_SONIC_ACTION_DIM * sizeof(float));
    memset(
        batch->clipped_actions_policy,
        0,
        batch->batch_size * GEAR_SONIC_ACTION_DIM * sizeof(float));
    memset(batch->history_count, 0, batch->batch_size * sizeof(uint8_t));
    memset(batch->history_head, 0, batch->batch_size * sizeof(uint8_t));
    batch->failed = 0;
}

static size_t history_offset(size_t row, size_t slot, size_t width) {
    return (row * GEAR_SONIC_HISTORY_FRAMES + slot) * width;
}

static void append_history(
        GearSonicNativeBatch* batch,
        size_t row,
        const GearSonicNativeStateInput* state) {
    uint8_t count = batch->history_count[row];
    uint8_t head = batch->history_head[row];
    uint8_t slot = count < GEAR_SONIC_HISTORY_FRAMES
        ? (uint8_t)((head + count) % GEAR_SONIC_HISTORY_FRAMES)
        : head;
    if (count < GEAR_SONIC_HISTORY_FRAMES) {
        batch->history_count[row] = (uint8_t)(count + 1);
    } else {
        batch->history_head[row] = (uint8_t)((head + 1) % GEAR_SONIC_HISTORY_FRAMES);
    }

    const double* base_quaternion = state->base_quaternion_wxyz + row * 4;
    const double* base_angular_velocity = state->base_angular_velocity_local + row * 3;
    const double* joint_position = state->joint_position_mujoco
        + row * GEAR_SONIC_ACTION_DIM;
    const double* joint_velocity = state->joint_velocity_mujoco
        + row * GEAR_SONIC_ACTION_DIM;
    const float* last_action = batch->clipped_actions_policy
        + row * GEAR_SONIC_ACTION_DIM;
    float* history_quaternion = batch->history_base_quaternion_wxyz
        + history_offset(row, slot, 4);
    float* history_angular = batch->history_base_angular_velocity
        + history_offset(row, slot, 3);
    float* history_position = batch->history_joint_position_policy
        + history_offset(row, slot, GEAR_SONIC_ACTION_DIM);
    float* history_velocity = batch->history_joint_velocity_policy
        + history_offset(row, slot, GEAR_SONIC_ACTION_DIM);
    float* history_action = batch->history_last_action_policy
        + history_offset(row, slot, GEAR_SONIC_ACTION_DIM);
    for (size_t index = 0; index < 4; index++) {
        history_quaternion[index] = (float)base_quaternion[index];
    }
    for (size_t index = 0; index < 3; index++) {
        history_angular[index] = (float)base_angular_velocity[index];
    }
    for (size_t policy_index = 0;
            policy_index < GEAR_SONIC_ACTION_DIM;
            policy_index++) {
        size_t mujoco_index = GEAR_SONIC_MUJOCO_TO_ISAACLAB[policy_index];
        history_position[policy_index] = (float)(
            joint_position[mujoco_index]
            - (double)GEAR_SONIC_DEFAULT_ANGLES_MUJOCO[mujoco_index]);
        history_velocity[policy_index] = (float)joint_velocity[mujoco_index];
        history_action[policy_index] = last_action[policy_index];
    }
}

static size_t resolve_frame(
        const GearSonicNativeMotion* motion, size_t current, size_t ahead) {
    size_t raw = current + ahead;
    if (raw < current) return motion->loop ? raw % motion->frames : motion->frames - 1;
    if (motion->loop) return raw % motion->frames;
    return raw < motion->frames ? raw : motion->frames - 1;
}

static void write_encoder_future(
        float* observation,
        size_t future,
        const float* position,
        const float* next_position,
        const float* root_xyzw,
        const double base_conjugate[4],
        const double heading_delta[4]) {
    for (size_t policy_index = 0;
            policy_index < GEAR_SONIC_ACTION_DIM;
            policy_index++) {
        size_t mujoco_index = GEAR_SONIC_MUJOCO_TO_ISAACLAB[policy_index];
        observation[4 + future * GEAR_SONIC_ACTION_DIM + policy_index]
            = position[mujoco_index];
        observation[294 + future * GEAR_SONIC_ACTION_DIM + policy_index]
            = (float)(((double)next_position[mujoco_index]
                - (double)position[mujoco_index]) / GEAR_SONIC_CONTROL_DT);
    }

    const double reference_wxyz[4] = {
        (double)root_xyzw[3],
        (double)root_xyzw[0],
        (double)root_xyzw[1],
        (double)root_xyzw[2],
    };
    double aligned[4];
    double relative[4];
    quaternion_multiply(heading_delta, reference_wxyz, aligned);
    quaternion_multiply(base_conjugate, aligned, relative);
    rotation_first_two_columns(relative, observation + 601 + future * 6);
}

static void build_encoder_row(
        GearSonicNativeBatch* batch,
        size_t row,
        const GearSonicNativeMotion* motion,
        const GearSonicNativeStateInput* state) {
    float* observation = batch->encoder_observations
        + row * GEAR_SONIC_ENCODER_INPUT_WIDTH;
    memset(observation, 0, GEAR_SONIC_ENCODER_INPUT_WIDTH * sizeof(float));
    /* Encoder offsets 0..3 store a scalar mode ID plus zero padding. G1 is 0. */
    size_t current = state->reference_frames[row];
    const double* base = state->base_quaternion_wxyz + row * 4;
    const double* heading_delta = state->heading_delta_wxyz + row * 4;
    double base_conjugate[4];
    quaternion_conjugate(base, base_conjugate);
    for (size_t future = 0; future < GEAR_SONIC_HISTORY_FRAMES; future++) {
        size_t frame = resolve_frame(motion, current, future * 5);
        size_t next = resolve_frame(motion, frame, 1);
        const float* position = motion->dof_position_mujoco
            + frame * GEAR_SONIC_ACTION_DIM;
        const float* next_position = motion->dof_position_mujoco
            + next * GEAR_SONIC_ACTION_DIM;
        const float* root_xyzw = motion->root_rotation_xyzw + frame * 4;
        write_encoder_future(
            observation,
            future,
            position,
            next_position,
            root_xyzw,
            base_conjugate,
            heading_delta);
    }
}

static void build_encoder_reference_row(
        GearSonicNativeBatch* batch,
        size_t row,
        const GearSonicNativeReferenceInput* references,
        const GearSonicNativeStateInput* state) {
    float* observation = batch->encoder_observations
        + row * GEAR_SONIC_ENCODER_INPUT_WIDTH;
    memset(observation, 0, GEAR_SONIC_ENCODER_INPUT_WIDTH * sizeof(float));
    /* Encoder offsets 0..3 store a scalar mode ID plus zero padding. G1 is 0. */
    const double* base = state->base_quaternion_wxyz + row * 4;
    const double* heading_delta = state->heading_delta_wxyz + row * 4;
    double base_conjugate[4];
    quaternion_conjugate(base, base_conjugate);
    for (size_t future = 0; future < GEAR_SONIC_HISTORY_FRAMES; future++) {
        size_t reference_row = row * GEAR_SONIC_HISTORY_FRAMES + future;
        const float* position = references->dof_position_mujoco
            + reference_row * GEAR_SONIC_ACTION_DIM;
        const float* next_position = references->dof_next_position_mujoco
            + reference_row * GEAR_SONIC_ACTION_DIM;
        const float* root_xyzw = references->root_rotation_xyzw
            + reference_row * 4;
        write_encoder_future(
            observation,
            future,
            position,
            next_position,
            root_xyzw,
            base_conjugate,
            heading_delta);
    }
}

static void build_decoder_row(GearSonicNativeBatch* batch, size_t row) {
    float* observation = batch->decoder_observations
        + row * GEAR_SONIC_DECODER_INPUT_WIDTH;
    memset(observation, 0, GEAR_SONIC_DECODER_INPUT_WIDTH * sizeof(float));
    memcpy(
        observation,
        batch->tokens + row * GEAR_SONIC_ENCODER_OUTPUT_WIDTH,
        GEAR_SONIC_ENCODER_OUTPUT_WIDTH * sizeof(float));
    size_t count = batch->history_count[row];
    size_t head = batch->history_head[row];
    size_t left_padding = GEAR_SONIC_HISTORY_FRAMES - count;
    for (size_t chronological = 0; chronological < count; chronological++) {
        size_t slot = (head + chronological) % GEAR_SONIC_HISTORY_FRAMES;
        size_t output_slot = left_padding + chronological;
        const float* quaternion = batch->history_base_quaternion_wxyz
            + history_offset(row, slot, 4);
        const float* angular = batch->history_base_angular_velocity
            + history_offset(row, slot, 3);
        const float* position = batch->history_joint_position_policy
            + history_offset(row, slot, GEAR_SONIC_ACTION_DIM);
        const float* velocity = batch->history_joint_velocity_policy
            + history_offset(row, slot, GEAR_SONIC_ACTION_DIM);
        const float* action = batch->history_last_action_policy
            + history_offset(row, slot, GEAR_SONIC_ACTION_DIM);
        memcpy(observation + 64 + output_slot * 3, angular, 3 * sizeof(float));
        memcpy(
            observation + 94 + output_slot * GEAR_SONIC_ACTION_DIM,
            position,
            GEAR_SONIC_ACTION_DIM * sizeof(float));
        memcpy(
            observation + 384 + output_slot * GEAR_SONIC_ACTION_DIM,
            velocity,
            GEAR_SONIC_ACTION_DIM * sizeof(float));
        memcpy(
            observation + 674 + output_slot * GEAR_SONIC_ACTION_DIM,
            action,
            GEAR_SONIC_ACTION_DIM * sizeof(float));
        projected_gravity(quaternion, observation + 964 + output_slot * 3);
    }
}

static void transform_actions(GearSonicNativeBatch* batch) {
    for (size_t row = 0; row < batch->batch_size; row++) {
        const float* raw = batch->actions_policy + row * GEAR_SONIC_ACTION_DIM;
        float* clipped = batch->clipped_actions_policy
            + row * GEAR_SONIC_ACTION_DIM;
        float* targets = batch->targets_mujoco + row * GEAR_SONIC_ACTION_DIM;
        for (size_t policy_index = 0;
                policy_index < GEAR_SONIC_ACTION_DIM;
                policy_index++) {
            float value = raw[policy_index];
            if (value < -GEAR_SONIC_ACTION_CLIP) value = -GEAR_SONIC_ACTION_CLIP;
            if (value > GEAR_SONIC_ACTION_CLIP) value = GEAR_SONIC_ACTION_CLIP;
            clipped[policy_index] = value;
        }
        for (size_t mujoco_index = 0;
                mujoco_index < GEAR_SONIC_ACTION_DIM;
                mujoco_index++) {
            size_t policy_index = GEAR_SONIC_ISAACLAB_TO_MUJOCO[mujoco_index];
            targets[mujoco_index] = GEAR_SONIC_DEFAULT_ANGLES_MUJOCO[mujoco_index]
                + clipped[policy_index] * GEAR_SONIC_ACTION_SCALE_MUJOCO[mujoco_index];
        }
    }
}

static int finish_batch_step(
        GearSonicNativeBatch* batch,
        GearSonicOrtBatch* ort,
        char* error,
        size_t error_capacity) {
    if (!gear_sonic_ort_encode(
            ort,
            batch->encoder_observations,
            batch->tokens,
            error,
            error_capacity)) {
        batch->failed = 1;
        return 0;
    }
    for (size_t row = 0; row < batch->batch_size; row++) {
        build_decoder_row(batch, row);
    }
    if (!gear_sonic_ort_decode(
            ort,
            batch->decoder_observations,
            batch->actions_policy,
            error,
            error_capacity)) {
        batch->failed = 1;
        return 0;
    }
    transform_actions(batch);
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    return 1;
}

int gear_sonic_native_batch_step(
        GearSonicNativeBatch* batch,
        GearSonicOrtBatch* ort,
        const GearSonicNativeMotion* motion,
        const GearSonicNativeStateInput* state,
        char* error,
        size_t error_capacity) {
    if (!batch_buffers_ready(batch)
            || ort == NULL || motion == NULL || state == NULL
            || batch->failed
            || ort->batch_size != batch->batch_size
            || motion->frames == 0 || (motion->loop != 0 && motion->loop != 1)
            || motion->dof_position_mujoco == NULL
            || motion->root_rotation_xyzw == NULL
            || state->reference_frames == NULL) {
        set_error(error, error_capacity, "native batch step", "invalid or failed state");
        return 0;
    }
    size_t batch_size = batch->batch_size;
    size_t action_count = 0;
    size_t quaternion_count = 0;
    size_t angular_count = 0;
    size_t motion_position_count = 0;
    size_t motion_rotation_count = 0;
    if (!checked_product(
            batch_size, GEAR_SONIC_ACTION_DIM, &action_count)
            || !checked_product(batch_size, 4, &quaternion_count)
            || !checked_product(batch_size, 3, &angular_count)
            || !checked_product(
            motion->frames, GEAR_SONIC_ACTION_DIM, &motion_position_count)
            || !checked_product(motion->frames, 4, &motion_rotation_count)
            || !byte_count_fits(action_count, sizeof(double))
            || !byte_count_fits(quaternion_count, sizeof(double))
            || !byte_count_fits(angular_count, sizeof(double))
            || !byte_count_fits(motion_position_count, sizeof(float))
            || !byte_count_fits(motion_rotation_count, sizeof(float))) {
        set_error(error, error_capacity, "native batch step", "size overflow");
        batch->failed = 1;
        return 0;
    }
    if (!finite_floats(
                motion->dof_position_mujoco, motion_position_count)
            || !finite_floats(motion->root_rotation_xyzw, motion_rotation_count)
            || !finite_doubles(state->base_quaternion_wxyz, quaternion_count)
            || !finite_doubles(state->base_angular_velocity_local, angular_count)
            || !finite_doubles(state->joint_position_mujoco, action_count)
            || !finite_doubles(state->joint_velocity_mujoco, action_count)
            || !finite_doubles(state->heading_delta_wxyz, quaternion_count)) {
        set_error(error, error_capacity, "native batch step", "non-finite input");
        batch->failed = 1;
        return 0;
    }
    for (size_t row = 0; row < batch_size; row++) {
        if (state->reference_frames[row] >= motion->frames) {
            set_error(error, error_capacity, "native batch step", "reference frame out of range");
            batch->failed = 1;
            return 0;
        }
        append_history(batch, row, state);
        build_encoder_row(batch, row, motion, state);
    }
    return finish_batch_step(batch, ort, error, error_capacity);
}

int gear_sonic_native_batch_step_references(
        GearSonicNativeBatch* batch,
        GearSonicOrtBatch* ort,
        const GearSonicNativeReferenceInput* references,
        const GearSonicNativeStateInput* state,
        char* error,
        size_t error_capacity) {
    if (!batch_buffers_ready(batch)
            || ort == NULL || references == NULL || state == NULL
            || batch->failed
            || ort->batch_size != batch->batch_size
            || references->dof_position_mujoco == NULL
            || references->dof_next_position_mujoco == NULL
            || references->root_rotation_xyzw == NULL
            || state->base_quaternion_wxyz == NULL
            || state->base_angular_velocity_local == NULL
            || state->joint_position_mujoco == NULL
            || state->joint_velocity_mujoco == NULL
            || state->heading_delta_wxyz == NULL) {
        set_error(
            error,
            error_capacity,
            "native batch reference step",
            "invalid or failed state");
        return 0;
    }

    const size_t batch_size = batch->batch_size;
    size_t reference_rows = 0;
    size_t reference_position_count = 0;
    size_t reference_rotation_count = 0;
    size_t quaternion_count = 0;
    size_t angular_count = 0;
    size_t action_count = 0;
    if (!checked_product(
            batch_size, GEAR_SONIC_HISTORY_FRAMES, &reference_rows)
            || !checked_product(
                reference_rows,
                GEAR_SONIC_ACTION_DIM,
                &reference_position_count)
            || !checked_product(reference_rows, 4, &reference_rotation_count)
            || !checked_product(batch_size, 4, &quaternion_count)
            || !checked_product(batch_size, 3, &angular_count)
            || !checked_product(
                batch_size, GEAR_SONIC_ACTION_DIM, &action_count)
            || !byte_count_fits(reference_position_count, sizeof(float))
            || !byte_count_fits(reference_rotation_count, sizeof(float))
            || !byte_count_fits(quaternion_count, sizeof(double))
            || !byte_count_fits(angular_count, sizeof(double))
            || !byte_count_fits(action_count, sizeof(double))) {
        set_error(
            error,
            error_capacity,
            "native batch reference step",
            "size overflow");
        batch->failed = 1;
        return 0;
    }

    if (!finite_floats(
            references->dof_position_mujoco,
            reference_position_count)
            || !finite_floats(
                references->dof_next_position_mujoco,
                reference_position_count)
            || !finite_floats(
                references->root_rotation_xyzw,
                reference_rotation_count)
            || !finite_doubles(
                state->base_quaternion_wxyz,
                quaternion_count)
            || !finite_doubles(
                state->base_angular_velocity_local,
                angular_count)
            || !finite_doubles(state->joint_position_mujoco, action_count)
            || !finite_doubles(state->joint_velocity_mujoco, action_count)
            || !finite_doubles(state->heading_delta_wxyz, quaternion_count)) {
        set_error(
            error,
            error_capacity,
            "native batch reference step",
            "non-finite input");
        batch->failed = 1;
        return 0;
    }

    for (size_t row = 0; row < batch_size; row++) {
        append_history(batch, row, state);
        build_encoder_reference_row(batch, row, references, state);
    }
    return finish_batch_step(batch, ort, error, error_capacity);
}
