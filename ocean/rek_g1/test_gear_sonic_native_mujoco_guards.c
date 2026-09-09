#include "gear_sonic_native_mujoco.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int require(int condition, const char* message) {
    if (condition) return 1;
    fprintf(stderr, "%s\n", message);
    return 0;
}

int main(void) {
    char error[256] = {0};
    GearSonicNativeMujocoVector vector = {0};
    float one_float = 0.0f;
    GearSonicNativeMotion oversized_motion = {
        .dof_position_mujoco = &one_float,
        .root_position_m = &one_float,
        .root_rotation_xyzw = &one_float,
        .frames = SIZE_MAX / GEAR_SONIC_ACTION_DIM,
        .loop = 0,
    };
    if (!require(
            !gear_sonic_native_mujoco_open(
                &vector,
                "unused-model",
                "unused-encoder",
                "unused-decoder",
                oversized_motion,
                1,
                1,
                error,
                sizeof(error)),
            "byte-overflow motion unexpectedly passed validation")
            || !require(
                strstr(error, "invalid motion arrays") != NULL,
                "byte-overflow motion returned the wrong error")) {
        return 1;
    }
    if (!require(
            !gear_sonic_native_mujoco_reset(&vector, error, sizeof(error)),
            "zero vector reset unexpectedly succeeded")
            || !require(vector.failed, "zero vector reset did not mark failure")
            || !require(
                strstr(error, "uninitialized vector") != NULL,
                "zero vector reset returned the wrong error")) {
        return 1;
    }
    gear_sonic_native_mujoco_close(&vector);
    gear_sonic_native_mujoco_close(&vector);

    memset(&vector, 0, sizeof(vector));
    vector.model = (mjModel*)(uintptr_t)1;
    if (!require(
            !gear_sonic_native_mujoco_step(&vector, error, sizeof(error)),
            "incomplete vector step unexpectedly succeeded")
            || !require(vector.failed, "incomplete vector step did not mark failure")
            || !require(
                strstr(error, "incomplete vector state") != NULL,
                "incomplete vector step returned the wrong error")) {
        vector.model = NULL;
        return 1;
    }
    vector.model = NULL;
    gear_sonic_native_mujoco_close(&vector);

    memset(&vector, 0, sizeof(vector));
    vector.model = (mjModel*)(uintptr_t)1;
    mjData* fake_data = (mjData*)(uintptr_t)1;
    vector.data = &fake_data;
    vector.batch_size = 1;
    float motion_position[GEAR_SONIC_ACTION_DIM] = {0};
    float root_position[3] = {0};
    float root_rotation[4] = {0, 0, 0, 1};
    vector.motion.frames = 1;
    vector.motion.dof_position_mujoco = motion_position;
    vector.motion.root_position_m = root_position;
    vector.motion.root_rotation_xyzw = root_rotation;
    vector.controller.batch_size = 1;
    vector.ort.batch_size = 1;
    vector.base_quaternion_wxyz = (double*)(uintptr_t)1;
    vector.base_angular_velocity_local = (double*)(uintptr_t)1;
    vector.joint_position_mujoco = (double*)(uintptr_t)1;
    vector.joint_velocity_mujoco = (double*)(uintptr_t)1;
    vector.heading_delta_wxyz = (double*)(uintptr_t)1;
    vector.reference_frames = (size_t*)(uintptr_t)1;
    vector.policy_ticks = (uint64_t*)(uintptr_t)1;
    vector.command_lpf_state_mujoco = (float*)(uintptr_t)1;
    vector.command_lpf_initialized = (uint8_t*)(uintptr_t)1;
    if (!require(
            !gear_sonic_native_mujoco_reset(&vector, error, sizeof(error)),
            "partial controller reset unexpectedly succeeded")
            || !require(vector.failed, "partial controller reset did not mark failure")
            || !require(
                strstr(error, "uninitialized vector") != NULL,
                "partial controller reset returned the wrong error")) {
        return 1;
    }

    memset(&vector, 0, sizeof(vector));
    vector.model = (mjModel*)(uintptr_t)1;
    vector.data = &fake_data;
    vector.batch_size = 1;
    vector.motion.frames = 1;
    vector.motion.dof_position_mujoco = motion_position;
    vector.motion.root_position_m = root_position;
    vector.motion.root_rotation_xyzw = root_rotation;
    vector.controller.batch_size = 1;
    vector.ort.batch_size = 1;
    vector.base_quaternion_wxyz = (double*)(uintptr_t)1;
    vector.base_angular_velocity_local = (double*)(uintptr_t)1;
    vector.joint_position_mujoco = (double*)(uintptr_t)1;
    vector.joint_velocity_mujoco = (double*)(uintptr_t)1;
    vector.heading_delta_wxyz = (double*)(uintptr_t)1;
    vector.reference_frames = (size_t*)(uintptr_t)1;
    vector.policy_ticks = (uint64_t*)(uintptr_t)1;
    vector.command_lpf_state_mujoco = (float*)(uintptr_t)1;
    vector.command_lpf_initialized = (uint8_t*)(uintptr_t)1;
    if (!require(
            !gear_sonic_native_mujoco_step(&vector, error, sizeof(error)),
            "partial controller step unexpectedly succeeded")
            || !require(vector.failed, "partial controller step did not mark failure")
            || !require(
                strstr(error, "incomplete vector state") != NULL,
                "partial controller step returned the wrong error")) {
        return 1;
    }
    return 0;
}
