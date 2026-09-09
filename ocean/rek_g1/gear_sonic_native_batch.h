#pragma once

#include <stddef.h>
#include <stdint.h>

#include "gear_sonic_ort.h"

#define GEAR_SONIC_HISTORY_FRAMES 10
#define GEAR_SONIC_ACTION_DIM 29

typedef struct GearSonicNativeMotion {
    const float* dof_position_mujoco;
    const float* root_position_m;
    const float* root_rotation_xyzw;
    size_t frames;
    int loop;
} GearSonicNativeMotion;

typedef struct GearSonicNativeStateInput {
    const double* base_quaternion_wxyz;
    const double* base_angular_velocity_local;
    const double* joint_position_mujoco;
    const double* joint_velocity_mujoco;
    const double* heading_delta_wxyz;
    const size_t* reference_frames;
} GearSonicNativeStateInput;

typedef struct GearSonicNativeReferenceInput {
    // Row-major [batch_size, GEAR_SONIC_HISTORY_FRAMES, 29]. The next-position
    // array supplies frame+1 for the same future sample so the native boundary
    // retains the pinned double-precision finite-difference operation.
    const float* dof_position_mujoco;
    const float* dof_next_position_mujoco;
    // Row-major [batch_size, GEAR_SONIC_HISTORY_FRAMES, 4], xyzw.
    const float* root_rotation_xyzw;
} GearSonicNativeReferenceInput;

typedef struct GearSonicNativeBatch {
    size_t batch_size;
    float* encoder_observations;
    float* tokens;
    float* decoder_observations;
    float* actions_policy;
    float* clipped_actions_policy;
    float* targets_mujoco;
    float* history_base_quaternion_wxyz;
    float* history_base_angular_velocity;
    float* history_joint_position_policy;
    float* history_joint_velocity_policy;
    float* history_last_action_policy;
    uint8_t* history_count;
    uint8_t* history_head;
    uint8_t initialized;
    int failed;
} GearSonicNativeBatch;

int gear_sonic_native_batch_open(
    GearSonicNativeBatch* batch,
    size_t batch_size,
    char* error,
    size_t error_capacity
);

void gear_sonic_native_batch_reset(GearSonicNativeBatch* batch);

int gear_sonic_native_batch_step(
    GearSonicNativeBatch* batch,
    GearSonicOrtBatch* ort,
    const GearSonicNativeMotion* motion,
    const GearSonicNativeStateInput* state,
    char* error,
    size_t error_capacity
);

// Run one controller tick from caller-composed, per-row reference windows.
// This is the semantic-motion boundary for vector runtimes. It performs the
// same history, observation, inference, and target transform as
// gear_sonic_native_batch_step. state->reference_frames is not read.
// Argument, size, ORT-batch, and initialization preflight rejections happen
// before state mutation and are retryable. Overflow, non-finite input, and
// inference failures latch batch->failed until gear_sonic_native_batch_reset.
int gear_sonic_native_batch_step_references(
    GearSonicNativeBatch* batch,
    GearSonicOrtBatch* ort,
    const GearSonicNativeReferenceInput* references,
    const GearSonicNativeStateInput* state,
    char* error,
    size_t error_capacity
);

void gear_sonic_native_batch_close(GearSonicNativeBatch* batch);
