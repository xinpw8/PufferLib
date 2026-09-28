#pragma once

#include <stddef.h>
#include <stdint.h>

#include <mujoco/mujoco.h>

#include "gear_sonic_native_batch.h"

typedef struct GearSonicNativeMujocoVector {
    mjModel* model;
    mjData** data;
    size_t batch_size;
    int physics_workers;
    int joint_ids[GEAR_SONIC_ACTION_DIM];
    int qpos_addresses[GEAR_SONIC_ACTION_DIM];
    int qvel_addresses[GEAR_SONIC_ACTION_DIM];
    int actuator_ids[GEAR_SONIC_ACTION_DIM];
    int root_qpos_address;
    int root_qvel_address;
    int root_body_id;
    GearSonicNativeMotion motion;
    GearSonicOrtBatch ort;
    GearSonicNativeBatch controller;
    double* base_quaternion_wxyz;
    double* base_angular_velocity_local;
    double* joint_position_mujoco;
    double* joint_velocity_mujoco;
    double* heading_delta_wxyz;
    size_t* reference_frames;
    uint64_t* policy_ticks;
    float* command_lpf_state_mujoco;
    uint8_t* command_lpf_initialized;
    int failed;
} GearSonicNativeMujocoVector;

int gear_sonic_native_mujoco_open(
    GearSonicNativeMujocoVector* vector,
    const char* model_path,
    const char* encoder_path,
    const char* decoder_path,
    GearSonicNativeMotion motion,
    size_t batch_size,
    int physics_workers,
    char* error,
    size_t error_capacity
);

int gear_sonic_native_mujoco_reset(
    GearSonicNativeMujocoVector* vector,
    char* error,
    size_t error_capacity
);

int gear_sonic_native_mujoco_step(
    GearSonicNativeMujocoVector* vector,
    char* error,
    size_t error_capacity
);

void gear_sonic_native_mujoco_close(GearSonicNativeMujocoVector* vector);
