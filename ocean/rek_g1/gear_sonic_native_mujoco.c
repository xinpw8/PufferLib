#include "gear_sonic_native_mujoco.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GEAR_SONIC_PHYSICS_DT 0.002
#define GEAR_SONIC_PHYSICS_STEPS_PER_CONTROL 10
#define GEAR_SONIC_COMMAND_LPF_INTERVAL 2
#define GEAR_SONIC_COMMAND_LPF_ALPHA 0.55686271190643310546875f

static const double GEAR_SONIC_KP_MUJOCO[GEAR_SONIC_ACTION_DIM] = {
    99.098428, 99.098428, 40.179238, 99.098428, 28.501246, 28.501246,
    99.098428, 99.098428, 40.179238, 99.098428, 28.501246, 28.501246,
    40.179238, 28.501246, 28.501246,
    14.250623, 14.250623, 14.250623, 14.250623, 14.250623, 16.778327, 16.778327,
    14.250623, 14.250623, 14.250623, 14.250623, 14.250623, 16.778327, 16.778327,
};

static const double GEAR_SONIC_KD_MUJOCO[GEAR_SONIC_ACTION_DIM] = {
    6.308802, 6.308802, 2.55789, 6.308802, 1.814446, 1.814446,
    6.308802, 6.308802, 2.55789, 6.308802, 1.814446, 1.814446,
    2.55789, 1.814446, 1.814446,
    0.907223, 0.907223, 0.907223, 0.907223, 0.907223, 1.068142, 1.068142,
    0.907223, 0.907223, 0.907223, 0.907223, 0.907223, 1.068142, 1.068142,
};

static const double GEAR_SONIC_FORCE_LIMIT_MUJOCO[GEAR_SONIC_ACTION_DIM] = {
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
            double component = (double)quaternion[axis];
            norm_squared += component * component;
        }
        if (!isfinite(norm_squared) || fabs(sqrt(norm_squared) - 1.0) > 1e-4) {
            return 0;
        }
    }
    return 1;
}

static int motion_is_valid(const GearSonicNativeMotion* motion) {
    size_t action_count = 0;
    size_t position_count = 0;
    size_t rotation_count = 0;
    return motion != NULL && motion->frames != 0
        && (motion->loop == 0 || motion->loop == 1)
        && checked_product(motion->frames, GEAR_SONIC_ACTION_DIM, &action_count)
        && checked_product(motion->frames, 3, &position_count)
        && checked_product(motion->frames, 4, &rotation_count)
        && action_count <= SIZE_MAX / sizeof(float)
        && position_count <= SIZE_MAX / sizeof(float)
        && rotation_count <= SIZE_MAX / sizeof(float)
        && finite_float_array(motion->dof_position_mujoco, action_count)
        && finite_float_array(motion->root_position_m, position_count)
        && finite_float_array(motion->root_rotation_xyzw, rotation_count)
        && unit_xyzw_quaternion_array(motion->root_rotation_xyzw, motion->frames);
}

static int controller_is_ready(
        const GearSonicNativeBatch* controller, size_t batch_size) {
    return controller != NULL && controller->batch_size == batch_size
        && controller->encoder_observations != NULL
        && controller->tokens != NULL
        && controller->decoder_observations != NULL
        && controller->actions_policy != NULL
        && controller->clipped_actions_policy != NULL
        && controller->targets_mujoco != NULL
        && controller->history_base_quaternion_wxyz != NULL
        && controller->history_base_angular_velocity != NULL
        && controller->history_joint_position_policy != NULL
        && controller->history_joint_velocity_policy != NULL
        && controller->history_last_action_policy != NULL
        && controller->history_count != NULL
        && controller->history_head != NULL;
}

static int ort_is_ready(const GearSonicOrtBatch* ort, size_t batch_size) {
    return ort != NULL && ort->batch_size == batch_size
        && ort->api != NULL && ort->environment != NULL
        && ort->session_options != NULL && ort->encoder != NULL
        && ort->decoder != NULL && ort->cpu_memory != NULL;
}

static int vector_buffers_are_ready(
        const GearSonicNativeMujocoVector* vector) {
    return vector->base_quaternion_wxyz != NULL
        && vector->base_angular_velocity_local != NULL
        && vector->joint_position_mujoco != NULL
        && vector->joint_velocity_mujoco != NULL
        && vector->heading_delta_wxyz != NULL
        && vector->reference_frames != NULL
        && vector->policy_ticks != NULL
        && vector->command_lpf_state_mujoco != NULL
        && vector->command_lpf_initialized != NULL;
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

static void configure_actuator(
        mjModel* model, int actuator_id, size_t index) {
    model->actuator_dyntype[actuator_id] = mjDYN_NONE;
    model->actuator_gaintype[actuator_id] = mjGAIN_FIXED;
    model->actuator_biastype[actuator_id] = mjBIAS_AFFINE;
    mjtNum* gain = model->actuator_gainprm + actuator_id * mjNGAIN;
    mjtNum* bias = model->actuator_biasprm + actuator_id * mjNBIAS;
    mju_zero(gain, mjNGAIN);
    mju_zero(bias, mjNBIAS);
    gain[0] = GEAR_SONIC_KP_MUJOCO[index];
    bias[1] = -GEAR_SONIC_KP_MUJOCO[index];
    bias[2] = -GEAR_SONIC_KD_MUJOCO[index];
    model->actuator_ctrllimited[actuator_id] = 0;
    model->actuator_forcelimited[actuator_id] = 1;
    model->actuator_forcerange[2 * actuator_id] = -GEAR_SONIC_FORCE_LIMIT_MUJOCO[index];
    model->actuator_forcerange[2 * actuator_id + 1] = GEAR_SONIC_FORCE_LIMIT_MUJOCO[index];
}

static int configure_model(
        GearSonicNativeMujocoVector* vector,
        char* error,
        size_t error_capacity) {
    mjModel* model = vector->model;
    if (model->nbody != 31 || model->njnt != 30 || model->nq != 36
            || model->nv != 35 || model->nu != 29 || model->ngeom != 54) {
        set_error(error, error_capacity, "configure model", "dimension mismatch");
        return 0;
    }
    if (model->jnt_type[0] != mjJNT_FREE) {
        set_error(error, error_capacity, "configure model", "joint zero is not free");
        return 0;
    }
    vector->root_qpos_address = model->jnt_qposadr[0];
    vector->root_qvel_address = model->jnt_dofadr[0];
    vector->root_body_id = model->jnt_bodyid[0];
    if (vector->root_qpos_address != 0 || vector->root_qvel_address != 0
            || vector->root_body_id <= 0) {
        set_error(error, error_capacity, "configure model", "free-joint mapping mismatch");
        return 0;
    }
    uint8_t joint_seen[30] = {0};
    for (size_t index = 0; index < GEAR_SONIC_ACTION_DIM; index++) {
        int actuator_id = (int)index;
        int joint_id = model->actuator_trnid[2 * actuator_id];
        int expected_joint_id = actuator_id + 1;
        int gear_matches = model->actuator_gear[actuator_id * 6] == 1.0;
        for (int gear_index = 1; gear_index < 6; gear_index++) {
            gear_matches = gear_matches
                && model->actuator_gear[actuator_id * 6 + gear_index] == 0.0;
        }
        if (joint_id != expected_joint_id || joint_seen[joint_id]
                || model->jnt_type[joint_id] != mjJNT_HINGE
                || model->actuator_trntype[actuator_id] != mjTRN_JOINT
                || model->actuator_trnid[2 * actuator_id + 1] != -1
                || model->jnt_qposadr[joint_id] != 7 + actuator_id
                || model->jnt_dofadr[joint_id] != 6 + actuator_id
                || !gear_matches) {
            set_error(error, error_capacity, "configure model", "actuator mapping mismatch");
            return 0;
        }
        joint_seen[joint_id] = 1;
        vector->joint_ids[index] = joint_id;
        vector->qpos_addresses[index] = model->jnt_qposadr[joint_id];
        vector->qvel_addresses[index] = model->jnt_dofadr[joint_id];
        vector->actuator_ids[index] = actuator_id;
        configure_actuator(model, actuator_id, index);
    }
    model->opt.timestep = GEAR_SONIC_PHYSICS_DT;
    return 1;
}

void gear_sonic_native_mujoco_close(GearSonicNativeMujocoVector* vector) {
    if (vector == NULL) return;
    free(vector->command_lpf_initialized);
    free(vector->command_lpf_state_mujoco);
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
        for (size_t index = 0; index < vector->batch_size; index++) {
            if (vector->data[index] != NULL) mj_deleteData(vector->data[index]);
        }
    }
    free(vector->data);
    if (vector->model != NULL) mj_deleteModel(vector->model);
    memset(vector, 0, sizeof(*vector));
}

static int allocate_vector_buffers(
        GearSonicNativeMujocoVector* vector,
        char* error,
        size_t error_capacity) {
    size_t action_count = 0;
    if (!checked_product(vector->batch_size, GEAR_SONIC_ACTION_DIM, &action_count)
            || vector->batch_size > SIZE_MAX / 4
            || vector->batch_size > SIZE_MAX / 3
            || action_count > SIZE_MAX / sizeof(double)
            || action_count > SIZE_MAX / sizeof(float)) {
        set_error(error, error_capacity, "allocate vector", "size overflow");
        return 0;
    }
    vector->data = (mjData**)calloc(vector->batch_size, sizeof(mjData*));
    vector->base_quaternion_wxyz = (double*)calloc(
        vector->batch_size * 4, sizeof(double));
    vector->base_angular_velocity_local = (double*)calloc(
        vector->batch_size * 3, sizeof(double));
    vector->joint_position_mujoco = (double*)calloc(action_count, sizeof(double));
    vector->joint_velocity_mujoco = (double*)calloc(action_count, sizeof(double));
    vector->heading_delta_wxyz = (double*)calloc(
        vector->batch_size * 4, sizeof(double));
    vector->reference_frames = (size_t*)calloc(vector->batch_size, sizeof(size_t));
    vector->policy_ticks = (uint64_t*)calloc(vector->batch_size, sizeof(uint64_t));
    vector->command_lpf_state_mujoco = (float*)calloc(action_count, sizeof(float));
    vector->command_lpf_initialized = (uint8_t*)calloc(
        vector->batch_size, sizeof(uint8_t));
    if (vector->data == NULL || vector->base_quaternion_wxyz == NULL
            || vector->base_angular_velocity_local == NULL
            || vector->joint_position_mujoco == NULL
            || vector->joint_velocity_mujoco == NULL
            || vector->heading_delta_wxyz == NULL
            || vector->reference_frames == NULL || vector->policy_ticks == NULL
            || vector->command_lpf_state_mujoco == NULL
            || vector->command_lpf_initialized == NULL) {
        set_error(error, error_capacity, "allocate vector", "allocation failure");
        return 0;
    }
    return 1;
}

int gear_sonic_native_mujoco_open(
        GearSonicNativeMujocoVector* vector,
        const char* model_path,
        const char* encoder_path,
        const char* decoder_path,
        GearSonicNativeMotion motion,
        size_t batch_size,
        int physics_workers,
        char* error,
        size_t error_capacity) {
    if (vector == NULL || model_path == NULL || encoder_path == NULL
            || decoder_path == NULL || batch_size == 0
            || physics_workers < 1 || motion.frames == 0
            || motion.dof_position_mujoco == NULL || motion.root_position_m == NULL
            || motion.root_rotation_xyzw == NULL
            || (motion.loop != 0 && motion.loop != 1)) {
        set_error(error, error_capacity, "open MuJoCo vector", "invalid argument");
        return 0;
    }
    if (!motion_is_valid(&motion)) {
        set_error(error, error_capacity, "open MuJoCo vector", "invalid motion arrays");
        return 0;
    }
    memset(vector, 0, sizeof(*vector));
    vector->batch_size = batch_size;
    vector->physics_workers = physics_workers;
    vector->motion = motion;
    char mujoco_error[1024] = {0};
    vector->model = mj_loadXML(
        model_path, NULL, mujoco_error, (int)sizeof(mujoco_error));
    if (vector->model == NULL) {
        set_error(error, error_capacity, "load MuJoCo model", mujoco_error);
        gear_sonic_native_mujoco_close(vector);
        return 0;
    }
    if (!configure_model(vector, error, error_capacity)
            || !allocate_vector_buffers(vector, error, error_capacity)
            || !gear_sonic_ort_open(
                &vector->ort,
                encoder_path,
                decoder_path,
                batch_size,
                error,
                error_capacity)
            || !gear_sonic_native_batch_open(
                &vector->controller, batch_size, error, error_capacity)) {
        gear_sonic_native_mujoco_close(vector);
        return 0;
    }
    for (size_t index = 0; index < batch_size; index++) {
        vector->data[index] = mj_makeData(vector->model);
        if (vector->data[index] == NULL) {
            set_error(error, error_capacity, "open MuJoCo vector", "mj_makeData failed");
            gear_sonic_native_mujoco_close(vector);
            return 0;
        }
    }
    if (!gear_sonic_native_mujoco_reset(vector, error, error_capacity)) {
        gear_sonic_native_mujoco_close(vector);
        return 0;
    }
    return 1;
}

int gear_sonic_native_mujoco_reset(
        GearSonicNativeMujocoVector* vector,
        char* error,
        size_t error_capacity) {
    if (vector == NULL || vector->model == NULL || vector->data == NULL
            || vector->batch_size == 0 || !motion_is_valid(&vector->motion)
            || !controller_is_ready(&vector->controller, vector->batch_size)
            || !ort_is_ready(&vector->ort, vector->batch_size)
            || !vector_buffers_are_ready(vector)) {
        set_error(error, error_capacity, "reset MuJoCo vector", "uninitialized vector");
        if (vector != NULL) vector->failed = 1;
        return 0;
    }
    for (size_t env_index = 0; env_index < vector->batch_size; env_index++) {
        if (vector->data[env_index] == NULL) {
            set_error(error, error_capacity, "reset MuJoCo vector", "missing environment data");
            vector->failed = 1;
            return 0;
        }
    }
    const float* root_position = vector->motion.root_position_m;
    const float* root_xyzw = vector->motion.root_rotation_xyzw;
    const double reference_wxyz[4] = {
        (double)root_xyzw[3],
        (double)root_xyzw[0],
        (double)root_xyzw[1],
        (double)root_xyzw[2],
    };
    for (size_t env_index = 0; env_index < vector->batch_size; env_index++) {
        mjData* data = vector->data[env_index];
        mj_resetData(vector->model, data);
        int root = vector->root_qpos_address;
        for (size_t axis = 0; axis < 3; axis++) {
            data->qpos[root + (int)axis] = (double)root_position[axis];
        }
        for (size_t axis = 0; axis < 4; axis++) {
            data->qpos[root + 3 + (int)axis] = reference_wxyz[axis];
        }
        for (size_t index = 0; index < GEAR_SONIC_ACTION_DIM; index++) {
            int joint_id = vector->joint_ids[index];
            float value = vector->motion.dof_position_mujoco[index];
            if (vector->model->jnt_limited[joint_id]) {
                double lower = vector->model->jnt_range[2 * joint_id];
                double upper = vector->model->jnt_range[2 * joint_id + 1];
                if ((double)value < lower) value = (float)lower;
                if ((double)value > upper) value = (float)upper;
            }
            data->qpos[vector->qpos_addresses[index]] = (double)value;
        }
        mju_zero(data->qvel, vector->model->nv);
        mju_zero(data->ctrl, vector->model->nu);
        mj_forward(vector->model, data);
        double base_heading[4];
        double reference_heading[4];
        double reference_heading_inverse[4];
        heading_quaternion(data->qpos + root + 3, base_heading);
        heading_quaternion(reference_wxyz, reference_heading);
        quaternion_conjugate(reference_heading, reference_heading_inverse);
        quaternion_multiply(
            base_heading,
            reference_heading_inverse,
            vector->heading_delta_wxyz + env_index * 4);
    }
    memset(vector->policy_ticks, 0, vector->batch_size * sizeof(uint64_t));
    memset(vector->reference_frames, 0, vector->batch_size * sizeof(size_t));
    memset(
        vector->command_lpf_state_mujoco,
        0,
        vector->batch_size * GEAR_SONIC_ACTION_DIM * sizeof(float));
    memset(
        vector->command_lpf_initialized,
        0,
        vector->batch_size * sizeof(uint8_t));
    gear_sonic_native_batch_reset(&vector->controller);
    vector->failed = 0;
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    return 1;
}

static int gather_state(
        GearSonicNativeMujocoVector* vector,
        char* error,
        size_t error_capacity) {
    for (size_t env_index = 0; env_index < vector->batch_size; env_index++) {
        mjData* data = vector->data[env_index];
        double velocity[6];
        mj_objectVelocity(
            vector->model,
            data,
            mjOBJ_BODY,
            vector->root_body_id,
            velocity,
            1);
        for (size_t axis = 0; axis < 4; axis++) {
            vector->base_quaternion_wxyz[env_index * 4 + axis]
                = data->qpos[vector->root_qpos_address + 3 + (int)axis];
        }
        for (size_t axis = 0; axis < 3; axis++) {
            vector->base_angular_velocity_local[env_index * 3 + axis]
                = velocity[axis];
        }
        for (size_t index = 0; index < GEAR_SONIC_ACTION_DIM; index++) {
            size_t offset = env_index * GEAR_SONIC_ACTION_DIM + index;
            vector->joint_position_mujoco[offset]
                = data->qpos[vector->qpos_addresses[index]];
            vector->joint_velocity_mujoco[offset]
                = data->qvel[vector->qvel_addresses[index]];
        }
        uint64_t tick = vector->policy_ticks[env_index];
        vector->reference_frames[env_index] = vector->motion.loop
            ? (size_t)(tick % vector->motion.frames)
            : tick < vector->motion.frames ? (size_t)tick : vector->motion.frames - 1;
        for (int index = 0; index < vector->model->nq; index++) {
            if (!isfinite(data->qpos[index])) {
                set_error(error, error_capacity, "gather MuJoCo state", "non-finite qpos");
                return 0;
            }
        }
        for (int index = 0; index < vector->model->nv; index++) {
            if (!isfinite(data->qvel[index])) {
                set_error(error, error_capacity, "gather MuJoCo state", "non-finite qvel");
                return 0;
            }
        }
    }
    return 1;
}

static void prepare_controls(
        GearSonicNativeMujocoVector* vector, size_t env_index, int substep) {
    const float* targets = vector->controller.targets_mujoco
        + env_index * GEAR_SONIC_ACTION_DIM;
    float* filtered = vector->command_lpf_state_mujoco
        + env_index * GEAR_SONIC_ACTION_DIM;
    if (substep % GEAR_SONIC_COMMAND_LPF_INTERVAL == 0) {
        if (!vector->command_lpf_initialized[env_index]) {
            memcpy(filtered, targets, GEAR_SONIC_ACTION_DIM * sizeof(float));
            vector->command_lpf_initialized[env_index] = 1;
        } else {
            for (size_t index = 0; index < GEAR_SONIC_ACTION_DIM; index++) {
                float delta = targets[index] - filtered[index];
                delta *= GEAR_SONIC_COMMAND_LPF_ALPHA;
                filtered[index] += delta;
            }
        }
    }
    mjData* data = vector->data[env_index];
    mju_zero(data->ctrl, vector->model->nu);
    for (size_t index = 0; index < GEAR_SONIC_ACTION_DIM; index++) {
        int joint_id = vector->joint_ids[index];
        float value = filtered[index];
        if (vector->model->jnt_limited[joint_id]) {
            float lower = (float)vector->model->jnt_range[2 * joint_id];
            float upper = (float)vector->model->jnt_range[2 * joint_id + 1];
            if (value < lower) value = lower;
            if (value > upper) value = upper;
        }
        data->ctrl[vector->actuator_ids[index]] = (double)value;
    }
}

int gear_sonic_native_mujoco_step(
        GearSonicNativeMujocoVector* vector,
        char* error,
        size_t error_capacity) {
    if (vector == NULL || vector->model == NULL || vector->failed) {
        set_error(error, error_capacity, "step MuJoCo vector", "invalid or failed vector");
        return 0;
    }
    if (vector->data == NULL || vector->batch_size == 0
            || !motion_is_valid(&vector->motion)
            || !controller_is_ready(&vector->controller, vector->batch_size)
            || vector->controller.failed
            || !ort_is_ready(&vector->ort, vector->batch_size)
            || !vector_buffers_are_ready(vector)) {
        set_error(error, error_capacity, "step MuJoCo vector", "incomplete vector state");
        vector->failed = 1;
        return 0;
    }
    for (size_t env_index = 0; env_index < vector->batch_size; env_index++) {
        if (vector->data[env_index] == NULL) {
            set_error(error, error_capacity, "step MuJoCo vector", "missing environment data");
            vector->failed = 1;
            return 0;
        }
    }
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
    if (!gear_sonic_native_batch_step(
            &vector->controller,
            &vector->ort,
            &vector->motion,
            &input,
            error,
            error_capacity)) {
        vector->failed = 1;
        return 0;
    }
    for (int substep = 0; substep < GEAR_SONIC_PHYSICS_STEPS_PER_CONTROL; substep++) {
        for (size_t env_index = 0; env_index < vector->batch_size; env_index++) {
            prepare_controls(vector, env_index, substep);
        }
        #pragma omp parallel for schedule(static) num_threads(vector->physics_workers)
        for (size_t env_index = 0; env_index < vector->batch_size; env_index++) {
            mj_step(vector->model, vector->data[env_index]);
        }
    }
    for (size_t env_index = 0; env_index < vector->batch_size; env_index++) {
        mjData* data = vector->data[env_index];
        for (int index = 0; index < vector->model->nq; index++) {
            if (!isfinite(data->qpos[index])) {
                set_error(error, error_capacity, "step MuJoCo vector", "non-finite qpos");
                vector->failed = 1;
                return 0;
            }
        }
        for (int index = 0; index < vector->model->nv; index++) {
            if (!isfinite(data->qvel[index])) {
                set_error(error, error_capacity, "step MuJoCo vector", "non-finite qvel");
                vector->failed = 1;
                return 0;
            }
        }
        vector->policy_ticks[env_index] += 1;
    }
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    return 1;
}
