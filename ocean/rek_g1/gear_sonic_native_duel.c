#include "gear_sonic_native_duel.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GEAR_SONIC_DUEL_PHYSICS_DT 0.002
#define GEAR_SONIC_DUEL_PHYSICS_STEPS_PER_CONTROL 10
#define GEAR_SONIC_DUEL_COMMAND_LPF_INTERVAL 2
#define GEAR_SONIC_DUEL_COMMAND_LPF_ALPHA 0.55686271190643310546875f
#define GEAR_SONIC_DUEL_SPAWN_TOLERANCE 1e-12
#define GEAR_SONIC_DUEL_LIFECYCLE_TOKEN UINT64_C(0x52454b473144554c)

typedef struct GearSonicDuelDampenedControllerSnapshot {
    float actions_policy[GEAR_SONIC_ACTION_DIM];
    float clipped_actions_policy[GEAR_SONIC_ACTION_DIM];
    float targets_mujoco[GEAR_SONIC_ACTION_DIM];
    float history_base_quaternion_wxyz[GEAR_SONIC_HISTORY_FRAMES * 4];
    float history_base_angular_velocity[GEAR_SONIC_HISTORY_FRAMES * 3];
    float history_joint_position_policy[
        GEAR_SONIC_HISTORY_FRAMES * GEAR_SONIC_ACTION_DIM];
    float history_joint_velocity_policy[
        GEAR_SONIC_HISTORY_FRAMES * GEAR_SONIC_ACTION_DIM];
    float history_last_action_policy[
        GEAR_SONIC_HISTORY_FRAMES * GEAR_SONIC_ACTION_DIM];
    uint8_t history_count;
    uint8_t history_head;
} GearSonicDuelDampenedControllerSnapshot;

static const char* const GEAR_SONIC_DUEL_ROLE_PREFIX[GEAR_SONIC_DUEL_FIGHTERS] = {
    "player__",
    "opponent__",
};

static const char* const GEAR_SONIC_DUEL_ACTUATOR_NAMES[GEAR_SONIC_ACTION_DIM] = {
    "left_hip_pitch_3206",
    "left_hip_roll_3164",
    "left_hip_yaw_2984",
    "left_knee_3031",
    "left_ankle_pitch_3234",
    "left_ankle_roll_2975",
    "right_hip_pitch_3479",
    "right_hip_roll_3421",
    "right_hip_yaw_2949",
    "right_knee_3114",
    "right_ankle_pitch_3119",
    "right_ankle_roll_3089",
    "waist_yaw_3343",
    "waist_roll_3272",
    "waist_pitch_3466",
    "left_shoulder_pitch_3146",
    "left_shoulder_roll_3230",
    "left_shoulder_yaw_3150",
    "left_elbow_3483",
    "left_wrist_roll_3213",
    "left_wrist_pitch_3488",
    "left_wrist_yaw_3009",
    "right_shoulder_pitch_2971",
    "right_shoulder_roll_3289",
    "right_shoulder_yaw_2966",
    "right_elbow_3486",
    "right_wrist_roll_3478",
    "right_wrist_pitch_2990",
    "right_wrist_yaw_2973",
};

static const char* const GEAR_SONIC_DUEL_JOINT_NAMES[GEAR_SONIC_ACTION_DIM] = {
    "joint__left_hip_pitch_joint_3047",
    "joint__left_hip_roll_joint_3248",
    "joint__left_hip_yaw_joint_3267",
    "joint__left_knee_joint_3137",
    "joint__left_ankle_pitch_joint_2982",
    "joint__left_ankle_roll_joint_2905",
    "joint__right_hip_pitch_joint_3298",
    "joint__right_hip_roll_joint_3059",
    "joint__right_hip_yaw_joint_3071",
    "joint__right_knee_joint_3412",
    "joint__right_ankle_pitch_joint_3312",
    "joint__right_ankle_roll_joint_3474",
    "joint__waist_yaw_joint_3441",
    "joint__waist_roll_joint_3341",
    "joint__waist_pitch_joint_3233",
    "joint__left_shoulder_pitch_joint_3340",
    "joint__left_shoulder_roll_joint_3184",
    "joint__left_shoulder_yaw_joint_2923",
    "joint__left_elbow_joint_3144",
    "joint__left_wrist_roll_joint_3260",
    "joint__left_wrist_pitch_joint_3007",
    "joint__left_wrist_yaw_joint_3398",
    "joint__right_shoulder_pitch_joint_3242",
    "joint__right_shoulder_roll_joint_3044",
    "joint__right_shoulder_yaw_joint_3176",
    "joint__right_elbow_joint_3407",
    "joint__right_wrist_roll_joint_3378",
    "joint__right_wrist_pitch_joint_3437",
    "joint__right_wrist_yaw_joint_3226",
};

static const double GEAR_SONIC_DUEL_SPAWN_QPOS[GEAR_SONIC_DUEL_FIGHTERS][7] = {
    {
        -0.9000000357627869,
        0.0,
        0.8029999826103449,
        0.9999999999998863,
        0.0,
        0.0,
        -4.7683710135965193e-07,
    },
    {
        0.8999999761581421,
        0.0,
        0.8029999826103449,
        -1.1026858146572184e-06,
        0.0,
        0.0,
        0.999999999999392,
    },
};

static const double GEAR_SONIC_DUEL_KP_MUJOCO[GEAR_SONIC_ACTION_DIM] = {
    99.098428, 99.098428, 40.179238, 99.098428, 28.501246, 28.501246,
    99.098428, 99.098428, 40.179238, 99.098428, 28.501246, 28.501246,
    40.179238, 28.501246, 28.501246,
    14.250623, 14.250623, 14.250623, 14.250623, 14.250623, 16.778327, 16.778327,
    14.250623, 14.250623, 14.250623, 14.250623, 14.250623, 16.778327, 16.778327,
};

/*
 * These actuator tables configure the native staging runtime. They are not
 * current-Steam numeric authority. Dampening below retains the live configured
 * values, rather than treating this table as recovered REK gain evidence.
 */

static const double GEAR_SONIC_DUEL_KD_MUJOCO[GEAR_SONIC_ACTION_DIM] = {
    6.308802, 6.308802, 2.55789, 6.308802, 1.814446, 1.814446,
    6.308802, 6.308802, 2.55789, 6.308802, 1.814446, 1.814446,
    2.55789, 1.814446, 1.814446,
    0.907223, 0.907223, 0.907223, 0.907223, 0.907223, 1.068142, 1.068142,
    0.907223, 0.907223, 0.907223, 0.907223, 0.907223, 1.068142, 1.068142,
};

static const double GEAR_SONIC_DUEL_FORCE_LIMIT_MUJOCO[GEAR_SONIC_ACTION_DIM] = {
    139.0, 139.0, 88.0, 139.0, 25.0, 25.0,
    139.0, 139.0, 88.0, 139.0, 25.0, 25.0,
    88.0, 25.0, 25.0,
    25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
    25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
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

static int finite_float_array(const float* values, size_t count) {
    if (values == NULL) return 0;
    for (size_t index = 0; index < count; index++) {
        if (!isfinite(values[index])) return 0;
    }
    return 1;
}

static int unit_xyzw_quaternion_array(const float* values, size_t count) {
    if (values == NULL) return 0;
    for (size_t index = 0; index < count; index++) {
        const float* quaternion = values + index * 4;
        double norm_squared = 0.0;
        for (size_t axis = 0; axis < 4; axis++) {
            const double component = (double)quaternion[axis];
            norm_squared += component * component;
        }
        if (!isfinite(norm_squared) || fabs(sqrt(norm_squared) - 1.0) > 1e-4) {
            return 0;
        }
    }
    return 1;
}

static int joined_name(
        char destination[160], const char* prefix, const char* source) {
    int written = snprintf(destination, 160, "%s%s", prefix, source);
    return written > 0 && written < 160;
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

static void heading_quaternion(const double q[4], double result[4]) {
    const double w = q[0];
    const double x = q[1];
    const double y = q[2];
    const double z = q[3];
    const double yaw = atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z));
    result[0] = cos(0.5 * yaw);
    result[1] = 0.0;
    result[2] = 0.0;
    result[3] = sin(0.5 * yaw);
}

static size_t robot_index(size_t arena_index, size_t fighter_index) {
    return arena_index * GEAR_SONIC_DUEL_FIGHTERS + fighter_index;
}

typedef struct LegacyPostStepObserverAdapter {
    GearSonicNativeDuelPostStepObserver observer;
    void* context;
} LegacyPostStepObserverAdapter;

static int adapt_legacy_post_step_observer(
        void* opaque,
        const GearSonicNativeDuelPostStepObservation* observation,
        GearSonicNativeDuelPostStepDirective* directive) {
    LegacyPostStepObserverAdapter* adapter
        = (LegacyPostStepObserverAdapter*)opaque;
    if (adapter == NULL || adapter->observer == NULL || directive == NULL
            || !adapter->observer(adapter->context, observation)) {
        return 0;
    }
    *directive = GEAR_SONIC_DUEL_POST_STEP_CONTINUE;
    return 1;
}

static int validate_motion(
        const GearSonicNativeMotion* motion,
        char* error,
        size_t error_capacity) {
    size_t dof_count = 0;
    size_t root_position_count = 0;
    size_t root_rotation_count = 0;
    if (motion == NULL || motion->frames == 0
            || (motion->loop != 0 && motion->loop != 1)
            || motion->dof_position_mujoco == NULL
            || motion->root_position_m == NULL
            || motion->root_rotation_xyzw == NULL
            || !checked_product(
                motion->frames, GEAR_SONIC_ACTION_DIM, &dof_count)
            || !checked_product(motion->frames, 3, &root_position_count)
            || !checked_product(motion->frames, 4, &root_rotation_count)
            || !finite_float_array(motion->dof_position_mujoco, dof_count)
            || !finite_float_array(motion->root_position_m, root_position_count)
            || !finite_float_array(motion->root_rotation_xyzw, root_rotation_count)
            || !unit_xyzw_quaternion_array(
                motion->root_rotation_xyzw, motion->frames)) {
        set_error(error, error_capacity, "validate duel motion", "invalid motion arrays");
        return 0;
    }
    return 1;
}

static void configure_actuator(mjModel* model, int actuator_id, size_t index) {
    model->actuator_dyntype[actuator_id] = mjDYN_NONE;
    model->actuator_gaintype[actuator_id] = mjGAIN_FIXED;
    model->actuator_biastype[actuator_id] = mjBIAS_AFFINE;
    mjtNum* gain = model->actuator_gainprm + actuator_id * mjNGAIN;
    mjtNum* bias = model->actuator_biasprm + actuator_id * mjNBIAS;
    mju_zero(gain, mjNGAIN);
    mju_zero(bias, mjNBIAS);
    gain[0] = GEAR_SONIC_DUEL_KP_MUJOCO[index];
    bias[1] = -GEAR_SONIC_DUEL_KP_MUJOCO[index];
    bias[2] = -GEAR_SONIC_DUEL_KD_MUJOCO[index];
    model->actuator_ctrllimited[actuator_id] = 0;
    model->actuator_forcelimited[actuator_id] = 1;
    model->actuator_forcerange[2 * actuator_id]
        = -GEAR_SONIC_DUEL_FORCE_LIMIT_MUJOCO[index];
    model->actuator_forcerange[2 * actuator_id + 1]
        = GEAR_SONIC_DUEL_FORCE_LIMIT_MUJOCO[index];
}

static int validate_spawn_prefix(
        const mjModel* model,
        const GearSonicDuelFighterMap* map,
        size_t fighter_index,
        char* error,
        size_t error_capacity) {
    for (size_t axis = 0; axis < 7; axis++) {
        const double observed = model->qpos0[map->root_qpos_address + (int)axis];
        const double expected = GEAR_SONIC_DUEL_SPAWN_QPOS[fighter_index][axis];
        if (!isfinite(observed)
                || fabs(observed - expected) > GEAR_SONIC_DUEL_SPAWN_TOLERANCE) {
            set_error(
                error,
                error_capacity,
                "configure duel model",
                "build-pinned spawn qpos0 prefix mismatch");
            return 0;
        }
    }
    return 1;
}

static int configure_fighter(
        GearSonicNativeDuelVector* vector,
        size_t fighter_index,
        char* error,
        size_t error_capacity) {
    mjModel* model = vector->model;
    GearSonicDuelFighterMap* map = &vector->fighters[fighter_index];
    const char* prefix = GEAR_SONIC_DUEL_ROLE_PREFIX[fighter_index];
    char name[160];
    if (!joined_name(name, prefix, "joint__floating_base_joint_3081")) {
        set_error(error, error_capacity, "configure duel model", "free-joint name overflow");
        return 0;
    }
    const int free_joint_id = mj_name2id(model, mjOBJ_JOINT, name);
    if (!joined_name(name, prefix, "pelvis_3266")) {
        set_error(error, error_capacity, "configure duel model", "root-body name overflow");
        return 0;
    }
    const int root_body_id = mj_name2id(model, mjOBJ_BODY, name);
    const int expected_free_joint_id = (int)(fighter_index * 30);
    const int expected_root_qpos = (int)(fighter_index * 36);
    const int expected_root_qvel = (int)(fighter_index * 35);
    if (free_joint_id != expected_free_joint_id || root_body_id <= 0
            || model->jnt_type[free_joint_id] != mjJNT_FREE
            || model->jnt_bodyid[free_joint_id] != root_body_id
            || model->jnt_qposadr[free_joint_id] != expected_root_qpos
            || model->jnt_dofadr[free_joint_id] != expected_root_qvel) {
        set_error(error, error_capacity, "configure duel model", "free-joint mapping mismatch");
        return 0;
    }
    map->root_qpos_address = expected_root_qpos;
    map->root_qvel_address = expected_root_qvel;
    map->root_body_id = root_body_id;

    uint8_t seen_joint[60] = {0};
    uint8_t seen_actuator[GEAR_SONIC_DUEL_CONTROL_DIM] = {0};
    uint8_t seen_qpos[GEAR_SONIC_DUEL_QPOS_DIM] = {0};
    uint8_t seen_qvel[GEAR_SONIC_DUEL_QVEL_DIM] = {0};
    for (int axis = 0; axis < 7; axis++) {
        seen_qpos[map->root_qpos_address + axis] = 1;
    }
    for (int axis = 0; axis < 6; axis++) {
        seen_qvel[map->root_qvel_address + axis] = 1;
    }
    for (size_t index = 0; index < GEAR_SONIC_ACTION_DIM; index++) {
        if (!joined_name(name, prefix, GEAR_SONIC_DUEL_ACTUATOR_NAMES[index])) {
            set_error(error, error_capacity, "configure duel model", "actuator name overflow");
            return 0;
        }
        const int actuator_id = mj_name2id(model, mjOBJ_ACTUATOR, name);
        if (!joined_name(name, prefix, GEAR_SONIC_DUEL_JOINT_NAMES[index])) {
            set_error(error, error_capacity, "configure duel model", "joint name overflow");
            return 0;
        }
        const int joint_id = mj_name2id(model, mjOBJ_JOINT, name);
        const int expected_actuator_id
            = (int)(fighter_index * GEAR_SONIC_ACTION_DIM + index);
        const int qpos_address = joint_id >= 0 && joint_id < model->njnt
            ? model->jnt_qposadr[joint_id]
            : -1;
        const int qvel_address = joint_id >= 0 && joint_id < model->njnt
            ? model->jnt_dofadr[joint_id]
            : -1;
        int gear_matches = actuator_id >= 0
            && model->actuator_gear[actuator_id * 6] == 1.0;
        for (int gear_index = 1; gear_index < 6 && gear_matches; gear_index++) {
            gear_matches = model->actuator_gear[actuator_id * 6 + gear_index] == 0.0;
        }
        if (joint_id < 0 || joint_id >= model->njnt
                || actuator_id != expected_actuator_id
                || seen_joint[joint_id] || seen_actuator[actuator_id]
                || model->jnt_type[joint_id] != mjJNT_HINGE
                || model->actuator_trntype[actuator_id] != mjTRN_JOINT
                || model->actuator_trnid[2 * actuator_id] != joint_id
                || model->actuator_trnid[2 * actuator_id + 1] != -1
                || qpos_address < 0 || qpos_address >= model->nq
                || qvel_address < 0 || qvel_address >= model->nv
                || seen_qpos[qpos_address] || seen_qvel[qvel_address]
                || !gear_matches) {
            set_error(error, error_capacity, "configure duel model", "named actuator mapping mismatch");
            return 0;
        }
        seen_joint[joint_id] = 1;
        seen_actuator[actuator_id] = 1;
        seen_qpos[qpos_address] = 1;
        seen_qvel[qvel_address] = 1;
        map->joint_ids[index] = joint_id;
        map->qpos_addresses[index] = qpos_address;
        map->qvel_addresses[index] = qvel_address;
        map->actuator_ids[index] = actuator_id;
        configure_actuator(model, actuator_id, index);
    }
    return validate_spawn_prefix(model, map, fighter_index, error, error_capacity);
}

static int configure_model(
        GearSonicNativeDuelVector* vector,
        char* error,
        size_t error_capacity) {
    mjModel* model = vector->model;
    if (model->nbody != 63 || model->njnt != 60
            || model->nq != GEAR_SONIC_DUEL_QPOS_DIM
            || model->nv != GEAR_SONIC_DUEL_QVEL_DIM
            || model->nu != GEAR_SONIC_DUEL_CONTROL_DIM
            || model->ngeom != 91) {
        set_error(error, error_capacity, "configure duel model", "dimension mismatch");
        return 0;
    }
    if (!configure_fighter(vector, GEAR_SONIC_DUEL_PLAYER, error, error_capacity)
            || !configure_fighter(
                vector, GEAR_SONIC_DUEL_OPPONENT, error, error_capacity)) {
        return 0;
    }
    for (size_t player_index = 0;
            player_index < GEAR_SONIC_ACTION_DIM;
            player_index++) {
        for (size_t opponent_index = 0;
                opponent_index < GEAR_SONIC_ACTION_DIM;
                opponent_index++) {
            if (vector->fighters[GEAR_SONIC_DUEL_PLAYER].joint_ids[player_index]
                        == vector->fighters[GEAR_SONIC_DUEL_OPPONENT]
                            .joint_ids[opponent_index]
                    || vector->fighters[GEAR_SONIC_DUEL_PLAYER]
                            .qpos_addresses[player_index]
                        == vector->fighters[GEAR_SONIC_DUEL_OPPONENT]
                            .qpos_addresses[opponent_index]
                    || vector->fighters[GEAR_SONIC_DUEL_PLAYER]
                            .qvel_addresses[player_index]
                        == vector->fighters[GEAR_SONIC_DUEL_OPPONENT]
                            .qvel_addresses[opponent_index]
                    || vector->fighters[GEAR_SONIC_DUEL_PLAYER]
                            .actuator_ids[player_index]
                        == vector->fighters[GEAR_SONIC_DUEL_OPPONENT]
                            .actuator_ids[opponent_index]) {
                set_error(error, error_capacity, "configure duel model", "fighter maps overlap");
                return 0;
            }
        }
    }
    model->opt.timestep = GEAR_SONIC_DUEL_PHYSICS_DT;
    vector->spawn_prefixes_verified = 1;
    return 1;
}

void gear_sonic_native_duel_close(GearSonicNativeDuelVector* vector) {
    if (vector == NULL) return;
    if (vector->lifecycle_token == 0u) return;
    if (vector->lifecycle_token != GEAR_SONIC_DUEL_LIFECYCLE_TOKEN) return;
    free(vector->reset_complete_not_before_time);
    free(vector->reset_completed_in_step_arenas);
    free(vector->reset_pending_arenas);
    free(vector->resetting_rows);
    free(vector->dampened_controller_snapshots);
    free(vector->dampened_force_limit_mujoco);
    free(vector->dampened_kd_mujoco);
    free(vector->dampened_kp_mujoco);
    free(vector->dampened_control_target_mujoco);
    free(vector->dampened_rows);
    free(vector->command_lpf_initialized);
    free(vector->command_lpf_state_mujoco);
    free(vector->motion_ticks);
    free(vector->policy_ticks);
    free(vector->reference_frames);
    free(vector->heading_delta_wxyz);
    free(vector->joint_velocity_mujoco);
    free(vector->joint_position_mujoco);
    free(vector->base_angular_velocity_local);
    free(vector->base_quaternion_wxyz);
    gear_sonic_native_batch_close(&vector->controller);
    gear_sonic_ort_close(&vector->ort);
    if (vector->data != NULL) {
        for (size_t index = 0; index < vector->arena_count; index++) {
            if (vector->data[index] != NULL) mj_deleteData(vector->data[index]);
        }
    }
    free(vector->data);
    if (vector->model != NULL) mj_deleteModel(vector->model);
    memset(vector, 0, sizeof(*vector));
}

static int allocate_buffers(
        GearSonicNativeDuelVector* vector,
        char* error,
        size_t error_capacity) {
    size_t action_count = 0;
    if (!checked_product(vector->robot_count, GEAR_SONIC_ACTION_DIM, &action_count)
            || vector->robot_count > SIZE_MAX / 4
            || vector->robot_count > SIZE_MAX / 3
            || action_count > SIZE_MAX / sizeof(double)
            || action_count > SIZE_MAX / sizeof(float)
            || vector->arena_count > SIZE_MAX / sizeof(mjData*)) {
        set_error(error, error_capacity, "allocate duel vector", "size overflow");
        return 0;
    }
    vector->data = (mjData**)calloc(vector->arena_count, sizeof(mjData*));
    vector->base_quaternion_wxyz
        = (double*)calloc(vector->robot_count * 4, sizeof(double));
    vector->base_angular_velocity_local
        = (double*)calloc(vector->robot_count * 3, sizeof(double));
    vector->joint_position_mujoco = (double*)calloc(action_count, sizeof(double));
    vector->joint_velocity_mujoco = (double*)calloc(action_count, sizeof(double));
    vector->heading_delta_wxyz
        = (double*)calloc(vector->robot_count * 4, sizeof(double));
    vector->reference_frames = (size_t*)calloc(vector->robot_count, sizeof(size_t));
    vector->policy_ticks = (uint64_t*)calloc(vector->robot_count, sizeof(uint64_t));
    vector->motion_ticks = (uint64_t*)calloc(vector->robot_count, sizeof(uint64_t));
    vector->command_lpf_state_mujoco = (float*)calloc(action_count, sizeof(float));
    vector->command_lpf_initialized
        = (uint8_t*)calloc(vector->robot_count, sizeof(uint8_t));
    vector->dampened_rows = (uint8_t*)calloc(vector->robot_count, sizeof(uint8_t));
    vector->dampened_control_target_mujoco
        = (float*)calloc(action_count, sizeof(float));
    vector->dampened_kp_mujoco = (float*)calloc(action_count, sizeof(float));
    vector->dampened_kd_mujoco = (float*)calloc(action_count, sizeof(float));
    vector->dampened_force_limit_mujoco
        = (float*)calloc(action_count, sizeof(float));
    vector->dampened_controller_snapshots = calloc(
        vector->robot_count,
        sizeof(GearSonicDuelDampenedControllerSnapshot));
    vector->resetting_rows
        = (uint8_t*)calloc(vector->robot_count, sizeof(uint8_t));
    vector->reset_pending_arenas
        = (uint8_t*)calloc(vector->arena_count, sizeof(uint8_t));
    vector->reset_completed_in_step_arenas
        = (uint8_t*)calloc(vector->arena_count, sizeof(uint8_t));
    vector->reset_complete_not_before_time
        = (double*)calloc(vector->arena_count, sizeof(double));
    if (vector->data == NULL || vector->base_quaternion_wxyz == NULL
            || vector->base_angular_velocity_local == NULL
            || vector->joint_position_mujoco == NULL
            || vector->joint_velocity_mujoco == NULL
            || vector->heading_delta_wxyz == NULL
            || vector->reference_frames == NULL || vector->policy_ticks == NULL
            || vector->motion_ticks == NULL
            || vector->command_lpf_state_mujoco == NULL
            || vector->command_lpf_initialized == NULL
            || vector->dampened_rows == NULL
            || vector->dampened_control_target_mujoco == NULL
            || vector->dampened_kp_mujoco == NULL
            || vector->dampened_kd_mujoco == NULL
            || vector->dampened_force_limit_mujoco == NULL
            || vector->dampened_controller_snapshots == NULL
            || vector->resetting_rows == NULL
            || vector->reset_pending_arenas == NULL
            || vector->reset_completed_in_step_arenas == NULL
            || vector->reset_complete_not_before_time == NULL) {
        set_error(error, error_capacity, "allocate duel vector", "allocation failure");
        return 0;
    }
    return 1;
}

static int gear_sonic_native_duel_open_internal(
        GearSonicNativeDuelVector* vector,
        const char* model_path,
        const void* model_xml_data,
        size_t model_xml_byte_count,
        const char* encoder_path,
        const char* decoder_path,
        const void* encoder_data,
        size_t encoder_byte_count,
        const void* decoder_data,
        size_t decoder_byte_count,
        int from_memory,
        GearSonicNativeMotion fixed_motion,
        size_t arena_count,
        int physics_workers,
        char* error,
        size_t error_capacity) {
    if (vector == NULL) {
        set_error(error, error_capacity, "open duel vector", "null vector");
        return 0;
    }
    if (vector->lifecycle_token != 0u) {
        set_error(
            error,
            error_capacity,
            "open duel vector",
            "vector is already initialized");
        return 0;
    }
    size_t robot_count = 0;
    if ((from_memory
                ? model_xml_data == NULL || model_xml_byte_count == 0u
                    || model_xml_byte_count > (size_t)INT_MAX
                    || encoder_data == NULL || encoder_byte_count == 0u
                    || decoder_data == NULL || decoder_byte_count == 0u
                : model_path == NULL || encoder_path == NULL
                    || decoder_path == NULL)
            || arena_count == 0 || physics_workers < 1
            || !checked_product(
                arena_count, GEAR_SONIC_DUEL_FIGHTERS, &robot_count)
            || !validate_motion(&fixed_motion, error, error_capacity)) {
        if (error == NULL || error_capacity == 0 || error[0] == '\0') {
            set_error(error, error_capacity, "open duel vector", "invalid argument");
        }
        return 0;
    }
    memset(vector, 0, sizeof(*vector));
    vector->lifecycle_token = GEAR_SONIC_DUEL_LIFECYCLE_TOKEN;
    vector->arena_count = arena_count;
    vector->robot_count = robot_count;
    vector->physics_workers = physics_workers;
    vector->fixed_motion = fixed_motion;
    char mujoco_error[1024] = {0};
    if (from_memory) {
        static const char MODEL_VFS_NAME[] = "model.two_fighter_arena.xml";
        mjVFS vfs;
        mj_defaultVFS(&vfs);
        const int vfs_status = mj_addBufferVFS(
            &vfs,
            MODEL_VFS_NAME,
            model_xml_data,
            (int)model_xml_byte_count);
        if (vfs_status == 0) {
            vector->model = mj_loadXML(
                MODEL_VFS_NAME,
                &vfs,
                mujoco_error,
                (int)sizeof(mujoco_error));
        }
        mj_deleteVFS(&vfs);
        if (vfs_status != 0) {
            set_error(error, error_capacity,
                "load duel MuJoCo model", "add model XML to VFS failed");
            gear_sonic_native_duel_close(vector);
            return 0;
        }
    } else {
        vector->model = mj_loadXML(
            model_path, NULL, mujoco_error, (int)sizeof(mujoco_error));
    }
    if (vector->model == NULL) {
        set_error(error, error_capacity, "load duel MuJoCo model", mujoco_error);
        gear_sonic_native_duel_close(vector);
        return 0;
    }
    if (!configure_model(vector, error, error_capacity)
            || !allocate_buffers(vector, error, error_capacity)
            || !(from_memory
                ? gear_sonic_ort_open_from_memory(
                    &vector->ort,
                    encoder_data,
                    encoder_byte_count,
                    decoder_data,
                    decoder_byte_count,
                    robot_count,
                    error,
                    error_capacity)
                : gear_sonic_ort_open(
                    &vector->ort,
                    encoder_path,
                    decoder_path,
                    robot_count,
                    error,
                    error_capacity))
            || !gear_sonic_native_batch_open(
                &vector->controller, robot_count, error, error_capacity)) {
        gear_sonic_native_duel_close(vector);
        return 0;
    }
    for (size_t arena_index = 0; arena_index < arena_count; arena_index++) {
        vector->data[arena_index] = mj_makeData(vector->model);
        if (vector->data[arena_index] == NULL) {
            set_error(error, error_capacity, "open duel vector", "mj_makeData failed");
            gear_sonic_native_duel_close(vector);
            return 0;
        }
    }
    if (!gear_sonic_native_duel_reset(vector, error, error_capacity)) {
        gear_sonic_native_duel_close(vector);
        return 0;
    }
    return 1;
}

int gear_sonic_native_duel_open(
        GearSonicNativeDuelVector* vector,
        const char* model_path,
        const char* encoder_path,
        const char* decoder_path,
        GearSonicNativeMotion fixed_motion,
        size_t arena_count,
        int physics_workers,
        char* error,
        size_t error_capacity) {
    return gear_sonic_native_duel_open_internal(
        vector,
        model_path,
        NULL,
        0u,
        encoder_path,
        decoder_path,
        NULL,
        0u,
        NULL,
        0u,
        0,
        fixed_motion,
        arena_count,
        physics_workers,
        error,
        error_capacity);
}

int gear_sonic_native_duel_open_from_memory(
        GearSonicNativeDuelVector* vector,
        const void* model_xml_data,
        size_t model_xml_byte_count,
        const void* encoder_data,
        size_t encoder_byte_count,
        const void* decoder_data,
        size_t decoder_byte_count,
        GearSonicNativeMotion fixed_motion,
        size_t arena_count,
        int physics_workers,
        char* error,
        size_t error_capacity) {
    return gear_sonic_native_duel_open_internal(
        vector,
        NULL,
        model_xml_data,
        model_xml_byte_count,
        NULL,
        NULL,
        encoder_data,
        encoder_byte_count,
        decoder_data,
        decoder_byte_count,
        1,
        fixed_motion,
        arena_count,
        physics_workers,
        error,
        error_capacity);
}

static int validate_live_spawn_prefixes(
        GearSonicNativeDuelVector* vector,
        mjData* data,
        char* error,
        size_t error_capacity) {
    for (size_t fighter_index = 0;
            fighter_index < GEAR_SONIC_DUEL_FIGHTERS;
            fighter_index++) {
        GearSonicDuelFighterMap* map = &vector->fighters[fighter_index];
        for (size_t axis = 0; axis < 7; axis++) {
            const double observed = data->qpos[map->root_qpos_address + (int)axis];
            const double expected = GEAR_SONIC_DUEL_SPAWN_QPOS[fighter_index][axis];
            if (!isfinite(observed)
                    || fabs(observed - expected) > GEAR_SONIC_DUEL_SPAWN_TOLERANCE) {
                set_error(error, error_capacity, "reset duel vector", "live spawn prefix mismatch");
                return 0;
            }
        }
    }
    return 1;
}

static int reset_arena_data_and_heading(
        GearSonicNativeDuelVector* vector,
        size_t arena_index,
        int preserve_time,
        char* error,
        size_t error_capacity) {
    mjData* data = vector->data[arena_index];
    if (data == NULL) {
        set_error(error, error_capacity, "reset duel arena", "missing arena data");
        return 0;
    }
    const mjtNum preserved_time = data->time;
    if (preserve_time && !isfinite(preserved_time)) {
        set_error(error, error_capacity, "reset duel arena", "non-finite arena time");
        return 0;
    }
    mj_resetData(vector->model, data);
    if (preserve_time) data->time = preserved_time;
    if (!validate_live_spawn_prefixes(vector, data, error, error_capacity)) {
        return 0;
    }
    for (size_t fighter_index = 0;
            fighter_index < GEAR_SONIC_DUEL_FIGHTERS;
            fighter_index++) {
        GearSonicDuelFighterMap* map = &vector->fighters[fighter_index];
        for (size_t index = 0; index < GEAR_SONIC_ACTION_DIM; index++) {
            float value = vector->fixed_motion.dof_position_mujoco[index];
            const int joint_id = map->joint_ids[index];
            if (vector->model->jnt_limited[joint_id]) {
                const double lower = vector->model->jnt_range[2 * joint_id];
                const double upper = vector->model->jnt_range[2 * joint_id + 1];
                if ((double)value < lower) value = (float)lower;
                if ((double)value > upper) value = (float)upper;
            }
            data->qpos[map->qpos_addresses[index]] = (double)value;
        }
    }
    mju_zero(data->qvel, vector->model->nv);
    mju_zero(data->ctrl, vector->model->nu);
    mj_forward(vector->model, data);
    if (preserve_time) data->time = preserved_time;
    if (!validate_live_spawn_prefixes(vector, data, error, error_capacity)) {
        return 0;
    }

    const float* root_xyzw = vector->fixed_motion.root_rotation_xyzw;
    const double reference_wxyz[4] = {
        (double)root_xyzw[3],
        (double)root_xyzw[0],
        (double)root_xyzw[1],
        (double)root_xyzw[2],
    };
    for (size_t fighter_index = 0;
            fighter_index < GEAR_SONIC_DUEL_FIGHTERS;
            fighter_index++) {
        const size_t row = robot_index(arena_index, fighter_index);
        const GearSonicDuelFighterMap* map = &vector->fighters[fighter_index];
        double base_heading[4];
        double reference_heading[4];
        double reference_heading_inverse[4];
        heading_quaternion(data->qpos + map->root_qpos_address + 3, base_heading);
        heading_quaternion(reference_wxyz, reference_heading);
        quaternion_conjugate(reference_heading, reference_heading_inverse);
        quaternion_multiply(
            base_heading,
            reference_heading_inverse,
            vector->heading_delta_wxyz + row * 4);
    }
    return 1;
}

static void reset_controller_row_history(
        GearSonicNativeBatch* controller, size_t row) {
    const size_t history_row = row * GEAR_SONIC_HISTORY_FRAMES;
    memset(
        controller->history_base_quaternion_wxyz + history_row * 4,
        0,
        GEAR_SONIC_HISTORY_FRAMES * 4 * sizeof(float));
    memset(
        controller->history_base_angular_velocity + history_row * 3,
        0,
        GEAR_SONIC_HISTORY_FRAMES * 3 * sizeof(float));
    memset(
        controller->history_joint_position_policy
            + history_row * GEAR_SONIC_ACTION_DIM,
        0,
        GEAR_SONIC_HISTORY_FRAMES * GEAR_SONIC_ACTION_DIM * sizeof(float));
    memset(
        controller->history_joint_velocity_policy
            + history_row * GEAR_SONIC_ACTION_DIM,
        0,
        GEAR_SONIC_HISTORY_FRAMES * GEAR_SONIC_ACTION_DIM * sizeof(float));
    memset(
        controller->history_last_action_policy
            + history_row * GEAR_SONIC_ACTION_DIM,
        0,
        GEAR_SONIC_HISTORY_FRAMES * GEAR_SONIC_ACTION_DIM * sizeof(float));
    memset(
        controller->clipped_actions_policy + row * GEAR_SONIC_ACTION_DIM,
        0,
        GEAR_SONIC_ACTION_DIM * sizeof(float));
    memset(
        controller->actions_policy + row * GEAR_SONIC_ACTION_DIM,
        0,
        GEAR_SONIC_ACTION_DIM * sizeof(float));
    memset(
        controller->targets_mujoco + row * GEAR_SONIC_ACTION_DIM,
        0,
        GEAR_SONIC_ACTION_DIM * sizeof(float));
    controller->history_count[row] = 0;
    controller->history_head[row] = 0;
}

static void capture_dampened_controller_row(
        GearSonicNativeDuelVector* vector, size_t row) {
    GearSonicNativeBatch* controller = &vector->controller;
    GearSonicDuelDampenedControllerSnapshot* snapshots
        = (GearSonicDuelDampenedControllerSnapshot*)
            vector->dampened_controller_snapshots;
    GearSonicDuelDampenedControllerSnapshot* snapshot = snapshots + row;
    const size_t action_offset = row * GEAR_SONIC_ACTION_DIM;
    const size_t history_offset = row * GEAR_SONIC_HISTORY_FRAMES;
    memcpy(
        snapshot->actions_policy,
        controller->actions_policy + action_offset,
        sizeof(snapshot->actions_policy));
    memcpy(
        snapshot->clipped_actions_policy,
        controller->clipped_actions_policy + action_offset,
        sizeof(snapshot->clipped_actions_policy));
    memcpy(
        snapshot->targets_mujoco,
        controller->targets_mujoco + action_offset,
        sizeof(snapshot->targets_mujoco));
    memcpy(
        snapshot->history_base_quaternion_wxyz,
        controller->history_base_quaternion_wxyz + history_offset * 4,
        sizeof(snapshot->history_base_quaternion_wxyz));
    memcpy(
        snapshot->history_base_angular_velocity,
        controller->history_base_angular_velocity + history_offset * 3,
        sizeof(snapshot->history_base_angular_velocity));
    memcpy(
        snapshot->history_joint_position_policy,
        controller->history_joint_position_policy
            + history_offset * GEAR_SONIC_ACTION_DIM,
        sizeof(snapshot->history_joint_position_policy));
    memcpy(
        snapshot->history_joint_velocity_policy,
        controller->history_joint_velocity_policy
            + history_offset * GEAR_SONIC_ACTION_DIM,
        sizeof(snapshot->history_joint_velocity_policy));
    memcpy(
        snapshot->history_last_action_policy,
        controller->history_last_action_policy
            + history_offset * GEAR_SONIC_ACTION_DIM,
        sizeof(snapshot->history_last_action_policy));
    snapshot->history_count = controller->history_count[row];
    snapshot->history_head = controller->history_head[row];
}

static void restore_dampened_controller_row(
        GearSonicNativeDuelVector* vector, size_t row) {
    GearSonicNativeBatch* controller = &vector->controller;
    const GearSonicDuelDampenedControllerSnapshot* snapshots
        = (const GearSonicDuelDampenedControllerSnapshot*)
            vector->dampened_controller_snapshots;
    const GearSonicDuelDampenedControllerSnapshot* snapshot = snapshots + row;
    const size_t action_offset = row * GEAR_SONIC_ACTION_DIM;
    const size_t history_offset = row * GEAR_SONIC_HISTORY_FRAMES;
    memcpy(
        controller->actions_policy + action_offset,
        snapshot->actions_policy,
        sizeof(snapshot->actions_policy));
    memcpy(
        controller->clipped_actions_policy + action_offset,
        snapshot->clipped_actions_policy,
        sizeof(snapshot->clipped_actions_policy));
    memcpy(
        controller->targets_mujoco + action_offset,
        snapshot->targets_mujoco,
        sizeof(snapshot->targets_mujoco));
    memcpy(
        controller->history_base_quaternion_wxyz + history_offset * 4,
        snapshot->history_base_quaternion_wxyz,
        sizeof(snapshot->history_base_quaternion_wxyz));
    memcpy(
        controller->history_base_angular_velocity + history_offset * 3,
        snapshot->history_base_angular_velocity,
        sizeof(snapshot->history_base_angular_velocity));
    memcpy(
        controller->history_joint_position_policy
            + history_offset * GEAR_SONIC_ACTION_DIM,
        snapshot->history_joint_position_policy,
        sizeof(snapshot->history_joint_position_policy));
    memcpy(
        controller->history_joint_velocity_policy
            + history_offset * GEAR_SONIC_ACTION_DIM,
        snapshot->history_joint_velocity_policy,
        sizeof(snapshot->history_joint_velocity_policy));
    memcpy(
        controller->history_last_action_policy
            + history_offset * GEAR_SONIC_ACTION_DIM,
        snapshot->history_last_action_policy,
        sizeof(snapshot->history_last_action_policy));
    controller->history_count[row] = snapshot->history_count;
    controller->history_head[row] = snapshot->history_head;
}

static int row_policy_suspended(
        const GearSonicNativeDuelVector* vector, size_t row) {
    return vector->dampened_rows[row] || vector->resetting_rows[row];
}

static void clear_dampened_gain_storage(
        GearSonicNativeDuelVector* vector, size_t row) {
    const size_t offset = row * GEAR_SONIC_ACTION_DIM;
    vector->dampened_rows[row] = 0;
    memset(
        vector->dampened_kp_mujoco + offset,
        0,
        GEAR_SONIC_ACTION_DIM * sizeof(float));
    memset(
        vector->dampened_kd_mujoco + offset,
        0,
        GEAR_SONIC_ACTION_DIM * sizeof(float));
    memset(
        vector->dampened_force_limit_mujoco + offset,
        0,
        GEAR_SONIC_ACTION_DIM * sizeof(float));
}

static void clear_suspended_row_storage(
        GearSonicNativeDuelVector* vector, size_t row) {
    const size_t offset = row * GEAR_SONIC_ACTION_DIM;
    GearSonicDuelDampenedControllerSnapshot* snapshots
        = (GearSonicDuelDampenedControllerSnapshot*)
            vector->dampened_controller_snapshots;
    memset(
        vector->dampened_control_target_mujoco + offset,
        0,
        GEAR_SONIC_ACTION_DIM * sizeof(float));
    memset(snapshots + row, 0, sizeof(*snapshots));
}

static void clear_all_row_suspension(
        GearSonicNativeDuelVector* vector, size_t row) {
    clear_dampened_gain_storage(vector, row);
    vector->resetting_rows[row] = 0;
    clear_suspended_row_storage(vector, row);
}

int gear_sonic_native_duel_set_row_dampened(
        GearSonicNativeDuelVector* vector,
        size_t row,
        int dampened,
        char* error,
        size_t error_capacity) {
    GearSonicNativeBatch* controller
        = vector == NULL ? NULL : &vector->controller;
    if (vector == NULL || vector->model == NULL || vector->data == NULL
            || vector->failed || row >= vector->robot_count
            || (dampened != 0 && dampened != 1)
            || vector->robot_count == 0
            || vector->robot_count
                != vector->arena_count * GEAR_SONIC_DUEL_FIGHTERS
            || vector->dampened_rows == NULL
            || vector->dampened_control_target_mujoco == NULL
            || vector->dampened_kp_mujoco == NULL
            || vector->dampened_kd_mujoco == NULL
            || vector->dampened_force_limit_mujoco == NULL
            || vector->dampened_controller_snapshots == NULL
            || vector->resetting_rows == NULL
            || controller->batch_size != vector->robot_count
            || controller->actions_policy == NULL
            || controller->clipped_actions_policy == NULL
            || controller->targets_mujoco == NULL
            || controller->history_base_quaternion_wxyz == NULL
            || controller->history_base_angular_velocity == NULL
            || controller->history_joint_position_policy == NULL
            || controller->history_joint_velocity_policy == NULL
            || controller->history_last_action_policy == NULL
            || controller->history_count == NULL
            || controller->history_head == NULL) {
        set_error(
            error,
            error_capacity,
            "set duel row dampened",
            "invalid or incomplete state");
        if (vector != NULL) vector->failed = 1;
        return 0;
    }
    if ((int)vector->dampened_rows[row] == dampened) {
        if (error != NULL && error_capacity > 0) error[0] = '\0';
        return 1;
    }
    if (!dampened) {
        clear_dampened_gain_storage(vector, row);
        if (!vector->resetting_rows[row]) {
            clear_suspended_row_storage(vector, row);
        }
        if (error != NULL && error_capacity > 0) error[0] = '\0';
        return 1;
    }
    if (vector->resetting_rows[row]) {
        set_error(
            error,
            error_capacity,
            "set duel row dampened",
            "cannot enter dampening during a pending arena reset");
        vector->failed = 1;
        return 0;
    }

    const size_t arena_index = row / GEAR_SONIC_DUEL_FIGHTERS;
    const size_t fighter_index = row % GEAR_SONIC_DUEL_FIGHTERS;
    mjData* data = vector->data[arena_index];
    const GearSonicDuelFighterMap* map = &vector->fighters[fighter_index];
    float target[GEAR_SONIC_ACTION_DIM];
    float retained_kp[GEAR_SONIC_ACTION_DIM];
    float retained_kd[GEAR_SONIC_ACTION_DIM];
    float retained_force[GEAR_SONIC_ACTION_DIM];
    if (data == NULL) {
        set_error(error, error_capacity, "set duel row dampened", "missing arena data");
        vector->failed = 1;
        return 0;
    }
    for (size_t index = 0; index < GEAR_SONIC_ACTION_DIM; index++) {
        const int actuator_id = map->actuator_ids[index];
        const mjtNum* gain
            = vector->model->actuator_gainprm + actuator_id * mjNGAIN;
        const mjtNum* bias
            = vector->model->actuator_biasprm + actuator_id * mjNBIAS;
        const double force_lower
            = vector->model->actuator_forcerange[2 * actuator_id];
        const double force_upper
            = vector->model->actuator_forcerange[2 * actuator_id + 1];
        const float live_kp = (float)gain[0];
        const float live_kd = (float)(-bias[2]);
        const float live_force = (float)force_upper;
        target[index] = (float)data->ctrl[actuator_id];
        if (vector->model->actuator_dyntype[actuator_id] != mjDYN_NONE
                || vector->model->actuator_gaintype[actuator_id] != mjGAIN_FIXED
                || vector->model->actuator_biastype[actuator_id] != mjBIAS_AFFINE
                || !vector->model->actuator_forcelimited[actuator_id]
                || !isfinite(gain[0]) || gain[0] <= 0.0
                || !isfinite(bias[0]) || !isfinite(bias[1])
                || !isfinite(bias[2])
                || bias[0] != 0.0 || bias[1] != -gain[0]
                || !isfinite(force_lower) || !isfinite(force_upper)
                || force_lower != -force_upper || force_upper <= 0.0
                || !isfinite(target[index])
                || !isfinite(live_kp) || live_kp <= 0.0f
                || !isfinite(live_kd) || live_kd < 0.0f
                || !isfinite(live_force) || live_force <= 0.0f) {
            set_error(
                error,
                error_capacity,
                "set duel row dampened",
                "live staged actuator is not a finite symmetric position drive");
            vector->failed = 1;
            return 0;
        }
        retained_kp[index]
            = live_kp * GEAR_SONIC_DUEL_DAMPEN_RETENTION;
        retained_kd[index]
            = live_kd * GEAR_SONIC_DUEL_DAMPEN_RETENTION;
        retained_force[index]
            = live_force * GEAR_SONIC_DUEL_DAMPEN_RETENTION;
    }

    const size_t offset = row * GEAR_SONIC_ACTION_DIM;
    memcpy(
        vector->dampened_control_target_mujoco + offset,
        target,
        sizeof(target));
    memcpy(vector->dampened_kp_mujoco + offset, retained_kp, sizeof(retained_kp));
    memcpy(vector->dampened_kd_mujoco + offset, retained_kd, sizeof(retained_kd));
    memcpy(
        vector->dampened_force_limit_mujoco + offset,
        retained_force,
        sizeof(retained_force));
    capture_dampened_controller_row(vector, row);
    vector->dampened_rows[row] = 1;
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    return 1;
}

static int arena_reset_surface_ready(
        const GearSonicNativeDuelVector* vector, size_t arena_index) {
    if (vector == NULL || vector->model == NULL || vector->data == NULL
            || vector->failed || vector->arena_count == 0
            || arena_index >= vector->arena_count
            || vector->arena_count > SIZE_MAX / GEAR_SONIC_DUEL_FIGHTERS
            || vector->robot_count
                != vector->arena_count * GEAR_SONIC_DUEL_FIGHTERS
            || vector->data[arena_index] == NULL
            || !vector->spawn_prefixes_verified
            || vector->heading_delta_wxyz == NULL
            || vector->policy_ticks == NULL || vector->motion_ticks == NULL
            || vector->reference_frames == NULL
            || vector->command_lpf_state_mujoco == NULL
            || vector->command_lpf_initialized == NULL
            || vector->dampened_rows == NULL
            || vector->dampened_control_target_mujoco == NULL
            || vector->dampened_kp_mujoco == NULL
            || vector->dampened_kd_mujoco == NULL
            || vector->dampened_force_limit_mujoco == NULL
            || vector->dampened_controller_snapshots == NULL
            || vector->resetting_rows == NULL
            || vector->reset_pending_arenas == NULL
            || vector->reset_completed_in_step_arenas == NULL
            || vector->reset_complete_not_before_time == NULL
            || vector->controller.batch_size != vector->robot_count
            || !vector->controller.initialized || vector->controller.failed
            || vector->controller.actions_policy == NULL
            || vector->controller.clipped_actions_policy == NULL
            || vector->controller.targets_mujoco == NULL
            || vector->controller.history_base_quaternion_wxyz == NULL
            || vector->controller.history_base_angular_velocity == NULL
            || vector->controller.history_joint_position_policy == NULL
            || vector->controller.history_joint_velocity_policy == NULL
            || vector->controller.history_last_action_policy == NULL
            || vector->controller.history_count == NULL
            || vector->controller.history_head == NULL
            || !validate_motion(&vector->fixed_motion, NULL, 0)) {
        return 0;
    }
    return 1;
}

static int finite_arena_state(
        const GearSonicNativeDuelVector* vector, const mjData* data) {
    for (int index = 0; index < vector->model->nq; index++) {
        if (!isfinite(data->qpos[index])) return 0;
    }
    for (int index = 0; index < vector->model->nv; index++) {
        if (!isfinite(data->qvel[index])) return 0;
    }
    return isfinite(data->time);
}

static int restore_arena_root_spawns(
        GearSonicNativeDuelVector* vector,
        size_t arena_index,
        char* error,
        size_t error_capacity) {
    mjData* data = vector->data[arena_index];
    if (!finite_arena_state(vector, data)) {
        set_error(error, error_capacity, "restore duel arena roots", "non-finite state");
        return 0;
    }
    const mjtNum preserved_time = data->time;
    for (size_t fighter_index = 0;
            fighter_index < GEAR_SONIC_DUEL_FIGHTERS;
            fighter_index++) {
        const GearSonicDuelFighterMap* map = &vector->fighters[fighter_index];
        memcpy(
            data->qpos + map->root_qpos_address,
            vector->model->qpos0 + map->root_qpos_address,
            7 * sizeof(double));
    }
    mj_forward(vector->model, data);
    data->time = preserved_time;
    if (!finite_arena_state(vector, data)
            || !validate_live_spawn_prefixes(
                vector, data, error, error_capacity)) {
        if (error == NULL || error_capacity == 0 || error[0] == '\0') {
            set_error(
                error,
                error_capacity,
                "restore duel arena roots",
                "root restore produced invalid state");
        }
        return 0;
    }
    return 1;
}

static void refresh_arena_heading_from_fixed_motion(
        GearSonicNativeDuelVector* vector, size_t arena_index) {
    const float* root_xyzw = vector->fixed_motion.root_rotation_xyzw;
    const double reference_wxyz[4] = {
        (double)root_xyzw[3],
        (double)root_xyzw[0],
        (double)root_xyzw[1],
        (double)root_xyzw[2],
    };
    mjData* data = vector->data[arena_index];
    for (size_t fighter_index = 0;
            fighter_index < GEAR_SONIC_DUEL_FIGHTERS;
            fighter_index++) {
        const size_t row = robot_index(arena_index, fighter_index);
        const GearSonicDuelFighterMap* map = &vector->fighters[fighter_index];
        double base_heading[4];
        double reference_heading[4];
        double reference_heading_inverse[4];
        heading_quaternion(data->qpos + map->root_qpos_address + 3, base_heading);
        heading_quaternion(reference_wxyz, reference_heading);
        quaternion_conjugate(reference_heading, reference_heading_inverse);
        quaternion_multiply(
            base_heading,
            reference_heading_inverse,
            vector->heading_delta_wxyz + row * 4);
    }
}

int gear_sonic_native_duel_begin_arena_reset(
        GearSonicNativeDuelVector* vector,
        size_t arena_index,
        char* error,
        size_t error_capacity) {
    if (!arena_reset_surface_ready(vector, arena_index)) {
        set_error(
            error,
            error_capacity,
            "begin deferred duel arena reset",
            "invalid or incomplete state");
        if (vector != NULL) vector->failed = 1;
        return 0;
    }
    if (vector->reset_pending_arenas[arena_index]) {
        set_error(
            error,
            error_capacity,
            "begin deferred duel arena reset",
            "arena reset is already pending");
        vector->failed = 1;
        return 0;
    }
    mjData* data = vector->data[arena_index];
    const double complete_not_before
        = (double)data->time + GEAR_SONIC_DUEL_PHYSICS_DT;
    if (!finite_arena_state(vector, data) || !isfinite(complete_not_before)) {
        set_error(
            error,
            error_capacity,
            "begin deferred duel arena reset",
            "non-finite arena state or completion boundary");
        vector->failed = 1;
        return 0;
    }
    for (size_t fighter_index = 0;
            fighter_index < GEAR_SONIC_DUEL_FIGHTERS;
            fighter_index++) {
        const size_t row = robot_index(arena_index, fighter_index);
        if (row_policy_suspended(vector, row)) continue;
        const GearSonicDuelFighterMap* map = &vector->fighters[fighter_index];
        float target[GEAR_SONIC_ACTION_DIM];
        for (size_t index = 0; index < GEAR_SONIC_ACTION_DIM; index++) {
            target[index] = (float)data->ctrl[map->actuator_ids[index]];
            if (!isfinite(target[index])) {
                set_error(
                    error,
                    error_capacity,
                    "begin deferred duel arena reset",
                    "non-finite live joint target");
                vector->failed = 1;
                return 0;
            }
        }
    }
    for (size_t fighter_index = 0;
            fighter_index < GEAR_SONIC_DUEL_FIGHTERS;
            fighter_index++) {
        const size_t row = robot_index(arena_index, fighter_index);
        if (row_policy_suspended(vector, row)) continue;
        const GearSonicDuelFighterMap* map = &vector->fighters[fighter_index];
        float* target = vector->dampened_control_target_mujoco
            + row * GEAR_SONIC_ACTION_DIM;
        for (size_t index = 0; index < GEAR_SONIC_ACTION_DIM; index++) {
            target[index] = (float)data->ctrl[map->actuator_ids[index]];
        }
        capture_dampened_controller_row(vector, row);
    }
    if (!restore_arena_root_spawns(
            vector, arena_index, error, error_capacity)) {
        vector->failed = 1;
        return 0;
    }
    for (size_t fighter_index = 0;
            fighter_index < GEAR_SONIC_DUEL_FIGHTERS;
            fighter_index++) {
        vector->resetting_rows[robot_index(arena_index, fighter_index)] = 1;
    }
    vector->reset_pending_arenas[arena_index] = 1;
    vector->reset_completed_in_step_arenas[arena_index] = 0;
    vector->reset_complete_not_before_time[arena_index] = complete_not_before;
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    return 1;
}

int gear_sonic_native_duel_complete_arena_reset(
        GearSonicNativeDuelVector* vector,
        size_t arena_index,
        char* error,
        size_t error_capacity) {
    if (!arena_reset_surface_ready(vector, arena_index)) {
        set_error(
            error,
            error_capacity,
            "complete deferred duel arena reset",
            "invalid or incomplete state");
        if (vector != NULL) vector->failed = 1;
        return 0;
    }
    if (!vector->reset_pending_arenas[arena_index]) {
        set_error(
            error,
            error_capacity,
            "complete deferred duel arena reset",
            "no pending reset");
        return 0;
    }
    mjData* data = vector->data[arena_index];
    if (!finite_arena_state(vector, data)) {
        set_error(
            error,
            error_capacity,
            "complete deferred duel arena reset",
            "non-finite arena state");
        vector->failed = 1;
        return 0;
    }
    if ((double)data->time
            < vector->reset_complete_not_before_time[arena_index]) {
        set_error(
            error,
            error_capacity,
            "complete deferred duel arena reset",
            "the next 2 ms fixed boundary has not completed");
        return 0;
    }
    for (size_t fighter_index = 0;
            fighter_index < GEAR_SONIC_DUEL_FIGHTERS;
            fighter_index++) {
        const GearSonicDuelFighterMap* map = &vector->fighters[fighter_index];
        for (size_t index = 0; index < GEAR_SONIC_ACTION_DIM; index++) {
            data->qpos[map->qpos_addresses[index]] = 0.0;
            data->qvel[map->qvel_addresses[index]] = 0.0;
        }
    }
    mju_zero(data->ctrl, vector->model->nu);
    if (!restore_arena_root_spawns(
            vector, arena_index, error, error_capacity)) {
        vector->failed = 1;
        return 0;
    }
    refresh_arena_heading_from_fixed_motion(vector, arena_index);
    for (size_t fighter_index = 0;
            fighter_index < GEAR_SONIC_DUEL_FIGHTERS;
            fighter_index++) {
        const size_t row = robot_index(arena_index, fighter_index);
        vector->policy_ticks[row] = 0;
        vector->motion_ticks[row] = 0;
        vector->reference_frames[row] = 0;
        memset(
            vector->command_lpf_state_mujoco + row * GEAR_SONIC_ACTION_DIM,
            0,
            GEAR_SONIC_ACTION_DIM * sizeof(float));
        vector->command_lpf_initialized[row] = 0;
        reset_controller_row_history(&vector->controller, row);
        clear_all_row_suspension(vector, row);
    }
    vector->reset_pending_arenas[arena_index] = 0;
    vector->reset_completed_in_step_arenas[arena_index] = 1;
    vector->reset_complete_not_before_time[arena_index] = 0.0;
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    return 1;
}

int gear_sonic_native_duel_reset(
        GearSonicNativeDuelVector* vector,
        char* error,
        size_t error_capacity) {
    if (vector == NULL || vector->model == NULL || vector->data == NULL
            || vector->arena_count == 0
            || vector->arena_count > SIZE_MAX / GEAR_SONIC_DUEL_FIGHTERS
            || vector->robot_count != vector->arena_count * GEAR_SONIC_DUEL_FIGHTERS
            || !vector->spawn_prefixes_verified
            || vector->controller.batch_size != vector->robot_count
            || vector->ort.batch_size != vector->robot_count
            || vector->heading_delta_wxyz == NULL
            || vector->policy_ticks == NULL || vector->motion_ticks == NULL
            || vector->reference_frames == NULL
            || vector->command_lpf_state_mujoco == NULL
            || vector->command_lpf_initialized == NULL
            || vector->dampened_rows == NULL
            || vector->dampened_control_target_mujoco == NULL
            || vector->dampened_kp_mujoco == NULL
            || vector->dampened_kd_mujoco == NULL
            || vector->dampened_force_limit_mujoco == NULL
            || vector->dampened_controller_snapshots == NULL
            || vector->resetting_rows == NULL
            || vector->reset_pending_arenas == NULL
            || vector->reset_completed_in_step_arenas == NULL
            || vector->reset_complete_not_before_time == NULL
            || !validate_motion(&vector->fixed_motion, error, error_capacity)) {
        if (error == NULL || error_capacity == 0 || error[0] == '\0') {
            set_error(error, error_capacity, "reset duel vector", "uninitialized vector");
        }
        if (vector != NULL) vector->failed = 1;
        return 0;
    }
    for (size_t fighter_index = 0;
            fighter_index < GEAR_SONIC_DUEL_FIGHTERS;
            fighter_index++) {
        if (!validate_spawn_prefix(
                vector->model,
                &vector->fighters[fighter_index],
                fighter_index,
                error,
                error_capacity)) {
            vector->failed = 1;
            return 0;
        }
    }
    for (size_t arena_index = 0; arena_index < vector->arena_count; arena_index++) {
        if (!reset_arena_data_and_heading(
                vector, arena_index, 0, error, error_capacity)) {
            vector->failed = 1;
            return 0;
        }
    }
    memset(vector->policy_ticks, 0, vector->robot_count * sizeof(uint64_t));
    memset(vector->motion_ticks, 0, vector->robot_count * sizeof(uint64_t));
    memset(vector->reference_frames, 0, vector->robot_count * sizeof(size_t));
    memset(
        vector->command_lpf_state_mujoco,
        0,
        vector->robot_count * GEAR_SONIC_ACTION_DIM * sizeof(float));
    memset(
        vector->command_lpf_initialized,
        0,
        vector->robot_count * sizeof(uint8_t));
    memset(vector->dampened_rows, 0, vector->robot_count * sizeof(uint8_t));
    memset(
        vector->dampened_control_target_mujoco,
        0,
        vector->robot_count * GEAR_SONIC_ACTION_DIM * sizeof(float));
    memset(
        vector->dampened_kp_mujoco,
        0,
        vector->robot_count * GEAR_SONIC_ACTION_DIM * sizeof(float));
    memset(
        vector->dampened_kd_mujoco,
        0,
        vector->robot_count * GEAR_SONIC_ACTION_DIM * sizeof(float));
    memset(
        vector->dampened_force_limit_mujoco,
        0,
        vector->robot_count * GEAR_SONIC_ACTION_DIM * sizeof(float));
    memset(
        vector->dampened_controller_snapshots,
        0,
        vector->robot_count
            * sizeof(GearSonicDuelDampenedControllerSnapshot));
    memset(vector->resetting_rows, 0, vector->robot_count * sizeof(uint8_t));
    memset(
        vector->reset_pending_arenas,
        0,
        vector->arena_count * sizeof(uint8_t));
    memset(
        vector->reset_completed_in_step_arenas,
        0,
        vector->arena_count * sizeof(uint8_t));
    memset(
        vector->reset_complete_not_before_time,
        0,
        vector->arena_count * sizeof(double));
    gear_sonic_native_batch_reset(&vector->controller);
    vector->failed = 0;
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    return 1;
}

int gear_sonic_native_duel_reset_arena_immediate(
        GearSonicNativeDuelVector* vector,
        size_t arena_index,
        char* error,
        size_t error_capacity) {
    GearSonicNativeBatch* controller
        = vector == NULL ? NULL : &vector->controller;
    if (vector == NULL || vector->model == NULL || vector->data == NULL
            || vector->failed || arena_index >= vector->arena_count
            || vector->arena_count == 0
            || vector->arena_count > SIZE_MAX / GEAR_SONIC_DUEL_FIGHTERS
            || vector->robot_count != vector->arena_count * GEAR_SONIC_DUEL_FIGHTERS
            || !vector->spawn_prefixes_verified
            || controller->batch_size != vector->robot_count
            || !controller->initialized || controller->failed
            || vector->ort.batch_size != vector->robot_count
            || vector->heading_delta_wxyz == NULL
            || vector->policy_ticks == NULL || vector->motion_ticks == NULL
            || vector->reference_frames == NULL
            || vector->command_lpf_state_mujoco == NULL
            || vector->command_lpf_initialized == NULL
            || vector->dampened_rows == NULL
            || vector->dampened_control_target_mujoco == NULL
            || vector->dampened_kp_mujoco == NULL
            || vector->dampened_kd_mujoco == NULL
            || vector->dampened_force_limit_mujoco == NULL
            || vector->dampened_controller_snapshots == NULL
            || vector->resetting_rows == NULL
            || vector->reset_pending_arenas == NULL
            || vector->reset_completed_in_step_arenas == NULL
            || vector->reset_complete_not_before_time == NULL
            || controller->history_base_quaternion_wxyz == NULL
            || controller->history_base_angular_velocity == NULL
            || controller->history_joint_position_policy == NULL
            || controller->history_joint_velocity_policy == NULL
            || controller->history_last_action_policy == NULL
            || controller->actions_policy == NULL
            || controller->clipped_actions_policy == NULL
            || controller->targets_mujoco == NULL
            || controller->history_count == NULL || controller->history_head == NULL
            || !validate_motion(&vector->fixed_motion, error, error_capacity)) {
        if (error == NULL || error_capacity == 0 || error[0] == '\0') {
            set_error(error, error_capacity, "reset duel arena", "invalid or incomplete state");
        }
        if (vector != NULL) vector->failed = 1;
        return 0;
    }
    for (size_t fighter_index = 0;
            fighter_index < GEAR_SONIC_DUEL_FIGHTERS;
            fighter_index++) {
        if (!validate_spawn_prefix(
                vector->model,
                &vector->fighters[fighter_index],
                fighter_index,
                error,
                error_capacity)) {
            vector->failed = 1;
            return 0;
        }
    }
    if (!reset_arena_data_and_heading(
            vector, arena_index, 1, error, error_capacity)) {
        vector->failed = 1;
        return 0;
    }
    for (size_t fighter_index = 0;
            fighter_index < GEAR_SONIC_DUEL_FIGHTERS;
            fighter_index++) {
        const size_t row = robot_index(arena_index, fighter_index);
        vector->policy_ticks[row] = 0;
        vector->motion_ticks[row] = 0;
        vector->reference_frames[row] = 0;
        memset(
            vector->command_lpf_state_mujoco + row * GEAR_SONIC_ACTION_DIM,
            0,
            GEAR_SONIC_ACTION_DIM * sizeof(float));
        vector->command_lpf_initialized[row] = 0;
        reset_controller_row_history(controller, row);
        clear_all_row_suspension(vector, row);
    }
    vector->reset_pending_arenas[arena_index] = 0;
    vector->reset_completed_in_step_arenas[arena_index] = 1;
    vector->reset_complete_not_before_time[arena_index] = 0.0;
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    return 1;
}

static int gather_state(
        GearSonicNativeDuelVector* vector,
        char* error,
        size_t error_capacity) {
    for (size_t arena_index = 0; arena_index < vector->arena_count; arena_index++) {
        mjData* data = vector->data[arena_index];
        for (int index = 0; index < vector->model->nq; index++) {
            if (!isfinite(data->qpos[index])) {
                set_error(error, error_capacity, "gather duel state", "non-finite qpos");
                return 0;
            }
        }
        for (int index = 0; index < vector->model->nv; index++) {
            if (!isfinite(data->qvel[index])) {
                set_error(error, error_capacity, "gather duel state", "non-finite qvel");
                return 0;
            }
        }
        for (size_t fighter_index = 0;
                fighter_index < GEAR_SONIC_DUEL_FIGHTERS;
                fighter_index++) {
            const size_t row = robot_index(arena_index, fighter_index);
            const GearSonicDuelFighterMap* map = &vector->fighters[fighter_index];
            double velocity[6];
            mj_objectVelocity(
                vector->model,
                data,
                mjOBJ_BODY,
                map->root_body_id,
                velocity,
                1);
            for (size_t axis = 0; axis < 4; axis++) {
                vector->base_quaternion_wxyz[row * 4 + axis]
                    = data->qpos[map->root_qpos_address + 3 + (int)axis];
            }
            for (size_t axis = 0; axis < 3; axis++) {
                vector->base_angular_velocity_local[row * 3 + axis] = velocity[axis];
            }
            for (size_t index = 0; index < GEAR_SONIC_ACTION_DIM; index++) {
                const size_t offset = row * GEAR_SONIC_ACTION_DIM + index;
                vector->joint_position_mujoco[offset]
                    = data->qpos[map->qpos_addresses[index]];
                vector->joint_velocity_mujoco[offset]
                    = data->qvel[map->qvel_addresses[index]];
            }
            if (!row_policy_suspended(vector, row)) {
                const uint64_t tick = vector->motion_ticks[row];
                vector->reference_frames[row] = vector->fixed_motion.loop
                    ? (size_t)(tick % vector->fixed_motion.frames)
                    : tick < vector->fixed_motion.frames
                        ? (size_t)tick
                        : vector->fixed_motion.frames - 1;
            }
        }
    }
    return 1;
}

static void update_filtered_target(
        GearSonicNativeDuelVector* vector, size_t row, int substep) {
    if (row_policy_suspended(vector, row)) return;
    const float* targets
        = vector->controller.targets_mujoco + row * GEAR_SONIC_ACTION_DIM;
    float* filtered
        = vector->command_lpf_state_mujoco + row * GEAR_SONIC_ACTION_DIM;
    if (substep % GEAR_SONIC_DUEL_COMMAND_LPF_INTERVAL != 0) return;
    if (!vector->command_lpf_initialized[row]) {
        memcpy(filtered, targets, GEAR_SONIC_ACTION_DIM * sizeof(float));
        vector->command_lpf_initialized[row] = 1;
        return;
    }
    for (size_t index = 0; index < GEAR_SONIC_ACTION_DIM; index++) {
        float delta = targets[index] - filtered[index];
        delta *= GEAR_SONIC_DUEL_COMMAND_LPF_ALPHA;
        filtered[index] += delta;
    }
}

static int prepare_controls(
        GearSonicNativeDuelVector* vector,
        size_t arena_index,
        int substep,
        char* error,
        size_t error_capacity) {
    mjData* data = vector->data[arena_index];
    mju_zero(data->ctrl, vector->model->nu);
    for (size_t fighter_index = 0;
            fighter_index < GEAR_SONIC_DUEL_FIGHTERS;
            fighter_index++) {
        const size_t row = robot_index(arena_index, fighter_index);
        const GearSonicDuelFighterMap* map = &vector->fighters[fighter_index];
        update_filtered_target(vector, row, substep);
        if (vector->dampened_rows[row]) {
            const size_t offset = row * GEAR_SONIC_ACTION_DIM;
            for (size_t index = 0; index < GEAR_SONIC_ACTION_DIM; index++) {
                const int actuator_id = map->actuator_ids[index];
                const int qpos_address = map->qpos_addresses[index];
                const int qvel_address = map->qvel_addresses[index];
                const double kp
                    = (double)vector->dampened_kp_mujoco[offset + index];
                const double kd
                    = (double)vector->dampened_kd_mujoco[offset + index];
                const double force_limit = (double)
                    vector->dampened_force_limit_mujoco[offset + index];
                const double target = (double)
                    vector->dampened_control_target_mujoco[offset + index];
                const double position = data->qpos[qpos_address];
                const double velocity = data->qvel[qvel_address];
                double force = kp * (target - position) - kd * velocity;
                if (force < -force_limit) force = -force_limit;
                if (force > force_limit) force = force_limit;
                const mjtNum* gain
                    = vector->model->actuator_gainprm + actuator_id * mjNGAIN;
                const mjtNum* bias
                    = vector->model->actuator_biasprm + actuator_id * mjNBIAS;
                const double base_bias = bias[0]
                    + bias[1] * position
                    + bias[2] * velocity;
                const double control = (force - base_bias) / gain[0];
                if (!isfinite(force) || !isfinite(control)) {
                    set_error(
                        error,
                        error_capacity,
                        "prepare dampened duel controls",
                        "non-finite retained drive");
                    return 0;
                }
                data->ctrl[actuator_id] = control;
            }
            continue;
        }
        if (vector->resetting_rows[row]) {
            const float* target = vector->dampened_control_target_mujoco
                + row * GEAR_SONIC_ACTION_DIM;
            for (size_t index = 0; index < GEAR_SONIC_ACTION_DIM; index++) {
                data->ctrl[map->actuator_ids[index]] = (double)target[index];
            }
            continue;
        }
        const float* filtered
            = vector->command_lpf_state_mujoco + row * GEAR_SONIC_ACTION_DIM;
        for (size_t index = 0; index < GEAR_SONIC_ACTION_DIM; index++) {
            float value = filtered[index];
            const int joint_id = map->joint_ids[index];
            if (vector->model->jnt_limited[joint_id]) {
                const float lower = (float)vector->model->jnt_range[2 * joint_id];
                const float upper = (float)vector->model->jnt_range[2 * joint_id + 1];
                if (value < lower) value = lower;
                if (value > upper) value = upper;
            }
            data->ctrl[map->actuator_ids[index]] = (double)value;
        }
    }
    return 1;
}

static int step_with_references(
        GearSonicNativeDuelVector* vector,
        const GearSonicNativeReferenceInput* references,
        GearSonicNativeDuelPostStepDirectiveCallback callback,
        void* callback_context,
        char* error,
        size_t error_capacity) {
    if (vector == NULL || vector->model == NULL || vector->failed
            || vector->data == NULL || vector->arena_count == 0
            || vector->arena_count > SIZE_MAX / GEAR_SONIC_DUEL_FIGHTERS
            || vector->robot_count != vector->arena_count * GEAR_SONIC_DUEL_FIGHTERS
            || vector->controller.batch_size != vector->robot_count
            || vector->ort.batch_size != vector->robot_count
            || vector->base_quaternion_wxyz == NULL
            || vector->base_angular_velocity_local == NULL
            || vector->joint_position_mujoco == NULL
            || vector->joint_velocity_mujoco == NULL
            || vector->heading_delta_wxyz == NULL
            || vector->policy_ticks == NULL || vector->motion_ticks == NULL
            || vector->reference_frames == NULL
            || vector->command_lpf_state_mujoco == NULL
            || vector->command_lpf_initialized == NULL
            || vector->dampened_rows == NULL
            || vector->dampened_control_target_mujoco == NULL
            || vector->dampened_kp_mujoco == NULL
            || vector->dampened_kd_mujoco == NULL
            || vector->dampened_force_limit_mujoco == NULL
            || vector->dampened_controller_snapshots == NULL
            || vector->resetting_rows == NULL
            || vector->reset_pending_arenas == NULL
            || vector->reset_completed_in_step_arenas == NULL
            || vector->reset_complete_not_before_time == NULL) {
        set_error(error, error_capacity, "step duel vector", "invalid or incomplete state");
        if (vector != NULL) vector->failed = 1;
        return 0;
    }
    for (size_t arena_index = 0; arena_index < vector->arena_count; arena_index++) {
        if (vector->data[arena_index] == NULL) {
            set_error(error, error_capacity, "step duel vector", "missing arena data");
            vector->failed = 1;
            return 0;
        }
    }
    memset(
        vector->reset_completed_in_step_arenas,
        0,
        vector->arena_count * sizeof(uint8_t));
    if (!gather_state(vector, error, error_capacity)) {
        vector->failed = 1;
        return 0;
    }
    GearSonicNativeStateInput input = {
        .base_quaternion_wxyz = vector->base_quaternion_wxyz,
        .base_angular_velocity_local = vector->base_angular_velocity_local,
        .joint_position_mujoco = vector->joint_position_mujoco,
        .joint_velocity_mujoco = vector->joint_velocity_mujoco,
        .heading_delta_wxyz = vector->heading_delta_wxyz,
        .reference_frames = vector->reference_frames,
    };
    const int controller_ok = references == NULL
        ? gear_sonic_native_batch_step(
            &vector->controller,
            &vector->ort,
            &vector->fixed_motion,
            &input,
            error,
            error_capacity)
        : gear_sonic_native_batch_step_references(
            &vector->controller,
            &vector->ort,
            references,
            &input,
            error,
            error_capacity);
    if (!controller_ok) {
        vector->failed = 1;
        return 0;
    }
    for (size_t row = 0; row < vector->robot_count; row++) {
        if (row_policy_suspended(vector, row)) {
            restore_dampened_controller_row(vector, row);
        }
    }
    for (int substep = 0;
            substep < GEAR_SONIC_DUEL_PHYSICS_STEPS_PER_CONTROL;
            substep++) {
        for (size_t arena_index = 0;
                arena_index < vector->arena_count;
                arena_index++) {
            if (!prepare_controls(
                    vector,
                    arena_index,
                    substep,
                    error,
                    error_capacity)) {
                vector->failed = 1;
                return 0;
            }
        }
        #pragma omp parallel for schedule(static) num_threads(vector->physics_workers)
        for (size_t arena_index = 0;
                arena_index < vector->arena_count;
                arena_index++) {
            mj_step(vector->model, vector->data[arena_index]);
        }
        if (callback != NULL) {
            for (size_t arena_index = 0;
                    arena_index < vector->arena_count;
                    arena_index++) {
                const GearSonicNativeDuelPostStepObservation observation = {
                    .vector = vector,
                    .model = vector->model,
                    .data = vector->data[arena_index],
                    .arena_index = arena_index,
                    .physics_substep_index = (uint32_t)substep,
                    .physics_dt_seconds = GEAR_SONIC_DUEL_PHYSICS_DT,
                };
                GearSonicNativeDuelPostStepDirective directive
                    = (GearSonicNativeDuelPostStepDirective)-1;
                if (!callback(callback_context, &observation, &directive)) {
                    char detail[160];
                    (void)snprintf(
                        detail,
                        sizeof(detail),
                        "post-step callback rejected arena %zu substep %u",
                        arena_index,
                        (unsigned int)observation.physics_substep_index);
                    set_error(error, error_capacity, "step duel vector", detail);
                    vector->failed = 1;
                    return 0;
                }
                if (directive == GEAR_SONIC_DUEL_POST_STEP_CONTINUE) continue;
                if (directive
                        == GEAR_SONIC_DUEL_POST_STEP_RESET_ARENA_IMMEDIATE) {
                    if (!gear_sonic_native_duel_reset_arena_immediate(
                            vector,
                            arena_index,
                            error,
                            error_capacity)) {
                        return 0;
                    }
                    continue;
                }
                char detail[160];
                (void)snprintf(
                    detail,
                    sizeof(detail),
                    "invalid post-step directive for arena %zu substep %u",
                    arena_index,
                    (unsigned int)observation.physics_substep_index);
                set_error(error, error_capacity, "step duel vector", detail);
                vector->failed = 1;
                return 0;
            }
        }
    }
    for (size_t arena_index = 0; arena_index < vector->arena_count; arena_index++) {
        mjData* data = vector->data[arena_index];
        for (int index = 0; index < vector->model->nq; index++) {
            if (!isfinite(data->qpos[index])) {
                set_error(error, error_capacity, "step duel vector", "non-finite qpos");
                vector->failed = 1;
                return 0;
            }
        }
        for (int index = 0; index < vector->model->nv; index++) {
            if (!isfinite(data->qvel[index])) {
                set_error(error, error_capacity, "step duel vector", "non-finite qvel");
                vector->failed = 1;
                return 0;
            }
        }
    }
    for (size_t row = 0; row < vector->robot_count; row++) {
        const size_t arena_index = row / GEAR_SONIC_DUEL_FIGHTERS;
        if (vector->reset_completed_in_step_arenas[arena_index]
                || row_policy_suspended(vector, row)) {
            continue;
        }
        vector->policy_ticks[row] += 1;
        vector->motion_ticks[row] += 1;
    }
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    return 1;
}

int gear_sonic_native_duel_step_fixed(
        GearSonicNativeDuelVector* vector,
        char* error,
        size_t error_capacity) {
    return gear_sonic_native_duel_step_fixed_with_post_step_observer(
        vector, NULL, NULL, error, error_capacity);
}

int gear_sonic_native_duel_step_fixed_with_post_step_observer(
        GearSonicNativeDuelVector* vector,
        GearSonicNativeDuelPostStepObserver observer,
        void* observer_context,
        char* error,
        size_t error_capacity) {
    LegacyPostStepObserverAdapter adapter = {
        .observer = observer,
        .context = observer_context,
    };
    return step_with_references(
        vector,
        NULL,
        observer == NULL ? NULL : adapt_legacy_post_step_observer,
        observer == NULL ? NULL : &adapter,
        error,
        error_capacity);
}

int gear_sonic_native_duel_step_references(
        GearSonicNativeDuelVector* vector,
        const GearSonicNativeReferenceInput* references,
        char* error,
        size_t error_capacity) {
    return gear_sonic_native_duel_step_references_with_post_step_observer(
        vector, references, NULL, NULL, error, error_capacity);
}

int gear_sonic_native_duel_step_references_with_post_step_observer(
        GearSonicNativeDuelVector* vector,
        const GearSonicNativeReferenceInput* references,
        GearSonicNativeDuelPostStepObserver observer,
        void* observer_context,
        char* error,
        size_t error_capacity) {
    LegacyPostStepObserverAdapter adapter = {
        .observer = observer,
        .context = observer_context,
    };
    return gear_sonic_native_duel_step_references_with_post_step_directive(
        vector,
        references,
        observer == NULL ? NULL : adapt_legacy_post_step_observer,
        observer == NULL ? NULL : &adapter,
        error,
        error_capacity);
}

int gear_sonic_native_duel_step_references_with_post_step_directive(
        GearSonicNativeDuelVector* vector,
        const GearSonicNativeReferenceInput* references,
        GearSonicNativeDuelPostStepDirectiveCallback callback,
        void* callback_context,
        char* error,
        size_t error_capacity) {
    size_t rotation_rows = 0;
    if (vector == NULL || references == NULL
            || references->root_rotation_xyzw == NULL
            || !checked_product(
                vector->robot_count,
                GEAR_SONIC_HISTORY_FRAMES,
                &rotation_rows)
            || rotation_rows > SIZE_MAX / 4) {
        set_error(error, error_capacity, "step duel references", "invalid reference windows");
        if (vector != NULL) vector->failed = 1;
        return 0;
    }
    for (size_t row = 0; row < rotation_rows; row++) {
        const float* quaternion = references->root_rotation_xyzw + row * 4;
        double norm_squared = 0.0;
        for (size_t axis = 0; axis < 4; axis++) {
            const double component = (double)quaternion[axis];
            norm_squared += component * component;
        }
        if (!isfinite(norm_squared) || fabs(sqrt(norm_squared) - 1.0) > 1e-4) {
            set_error(
                error,
                error_capacity,
                "step duel references",
                "reference rotation is not a finite unit quaternion");
            vector->failed = 1;
            return 0;
        }
    }
    return step_with_references(
        vector,
        references,
        callback,
        callback_context,
        error,
        error_capacity);
}

int gear_sonic_native_duel_step_semantic_unavailable(
        GearSonicNativeDuelVector* vector,
        const uint32_t* opaque_command_ids,
        size_t command_count,
        char* error,
        size_t error_capacity) {
    (void)opaque_command_ids;
    (void)command_count;
    set_error(
        error,
        error_capacity,
        "step duel semantic commands",
        "semantic motion routing is not wired; physics was not advanced");
    if (vector != NULL) vector->failed = 1;
    return 0;
}
