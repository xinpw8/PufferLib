#include "g1_fall_mujoco.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* const FLOOR_GEOM = "arena_Collider_Floor_Rektagon";
static const double RADIANS_TO_DEGREES =
    57.295779513082320876798154814105;
enum {
    BUILD_F84F1874_ARENA_BODY_COUNT = 63,
    BUILD_F84F1874_ARENA_GEOM_COUNT = 91,
};
static const char* const ROOT_BODIES[GEAR_SONIC_DUEL_FIGHTERS] = {
    "player__pelvis_3266",
    "opponent__pelvis_3266",
};
static const char* const LEFT_FOOT_BODIES[GEAR_SONIC_DUEL_FIGHTERS] = {
    "player__left_ankle_roll_link_3045",
    "opponent__left_ankle_roll_link_3045",
};
static const char* const RIGHT_FOOT_BODIES[GEAR_SONIC_DUEL_FIGHTERS] = {
    "player__right_ankle_roll_link_3090",
    "opponent__right_ankle_roll_link_3090",
};

static void set_error(
        char* error,
        size_t error_capacity,
        const char* operation,
        const char* detail) {
    if (error == NULL || error_capacity == 0u) return;
    if (operation == NULL) operation = "measure G1 fall state";
    if (detail == NULL) detail = "unknown failure";
    (void)snprintf(error, error_capacity, "%s: %s", operation, detail);
    error[error_capacity - 1u] = '\0';
}

static RekG1FallMujocoStatus fail(
        RekG1FallMujocoAdapter* adapter,
        RekG1FallMujocoStatus status,
        char* error,
        size_t error_capacity,
        const char* operation,
        const char* detail) {
    if (adapter != NULL) adapter->last_status = status;
    set_error(error, error_capacity, operation, detail);
    return status;
}

static int finite_values(const mjtNum* values, size_t count) {
    if (values == NULL) return 0;
    for (size_t index = 0u; index < count; index++) {
        if (!isfinite((double)values[index])) return 0;
    }
    return 1;
}

static int body_descends_from(
        const mjModel* model, int body_id, int ancestor_id) {
    if (model == NULL || body_id <= 0 || body_id >= model->nbody
            || ancestor_id <= 0 || ancestor_id >= model->nbody) {
        return 0;
    }
    for (int current = body_id;
            current > 0;
            current = model->body_parentid[current]) {
        if (current == ancestor_id) return 1;
    }
    return 0;
}

static int exact_name(
        const mjModel* model,
        mjtObj object_type,
        int object_id,
        const char* expected) {
    if (model == NULL || object_id < 0 || expected == NULL) return 0;
    const char* actual = mj_id2name(model, object_type, object_id);
    return actual != NULL && strcmp(actual, expected) == 0;
}

static int duel_shape_valid(const GearSonicNativeDuelVector* duel) {
    if (duel == NULL || duel->model == NULL || duel->data == NULL
            || duel->failed || !duel->spawn_prefixes_verified
            || duel->arena_count == 0u
            || duel->arena_count > SIZE_MAX / GEAR_SONIC_DUEL_FIGHTERS
            || duel->robot_count
                != duel->arena_count * GEAR_SONIC_DUEL_FIGHTERS
            || duel->model->nbody != BUILD_F84F1874_ARENA_BODY_COUNT
            || duel->model->ngeom != BUILD_F84F1874_ARENA_GEOM_COUNT
            || duel->model->nq != GEAR_SONIC_DUEL_QPOS_DIM
            || duel->model->nv != GEAR_SONIC_DUEL_QVEL_DIM
            || duel->model->nu != GEAR_SONIC_DUEL_CONTROL_DIM) {
        return 0;
    }
    for (size_t arena = 0u; arena < duel->arena_count; arena++) {
        if (duel->data[arena] == NULL) return 0;
    }
    return 1;
}

static int root_map_valid(
        const mjModel* model,
        const GearSonicDuelFighterMap* map,
        int root_body_id) {
    if (model == NULL || map == NULL || root_body_id <= 0
            || root_body_id >= model->nbody
            || map->root_body_id != root_body_id
            || model->body_jntnum[root_body_id] < 1) {
        return 0;
    }
    const int joint_id = model->body_jntadr[root_body_id];
    return joint_id >= 0 && joint_id < model->njnt
        && model->jnt_type[joint_id] == mjJNT_FREE
        && model->jnt_qposadr[joint_id] == map->root_qpos_address
        && model->jnt_dofadr[joint_id] == map->root_qvel_address
        && map->root_qpos_address >= 0
        && map->root_qpos_address + 7 <= model->nq;
}

static int checked_array_bytes(
        size_t count, size_t element_size, size_t* bytes) {
    if (bytes == NULL || count == 0u || element_size == 0u
            || count > SIZE_MAX / element_size) {
        return 0;
    }
    *bytes = count * element_size;
    return 1;
}

static int map_body_owners(RekG1FallMujocoAdapter* adapter) {
    const mjModel* model = adapter->duel->model;
    for (int body_id = 0; body_id < model->nbody; body_id++) {
        const int player = body_descends_from(
            model, body_id, adapter->root_body_ids[GEAR_SONIC_DUEL_PLAYER]);
        const int opponent = body_descends_from(
            model, body_id, adapter->root_body_ids[GEAR_SONIC_DUEL_OPPONENT]);
        if (player && opponent) return 0;
        adapter->body_owner[body_id] = player
            ? (int8_t)GEAR_SONIC_DUEL_PLAYER
            : opponent
                ? (int8_t)GEAR_SONIC_DUEL_OPPONENT
                : (int8_t)-1;
    }
    return adapter->body_owner[
            adapter->left_foot_body_ids[GEAR_SONIC_DUEL_PLAYER]]
            == (int8_t)GEAR_SONIC_DUEL_PLAYER
        && adapter->body_owner[
            adapter->right_foot_body_ids[GEAR_SONIC_DUEL_PLAYER]]
            == (int8_t)GEAR_SONIC_DUEL_PLAYER
        && adapter->body_owner[
            adapter->left_foot_body_ids[GEAR_SONIC_DUEL_OPPONENT]]
            == (int8_t)GEAR_SONIC_DUEL_OPPONENT
        && adapter->body_owner[
            adapter->right_foot_body_ids[GEAR_SONIC_DUEL_OPPONENT]]
            == (int8_t)GEAR_SONIC_DUEL_OPPONENT;
}

const char* rek_g1_fall_mujoco_status_string(
        RekG1FallMujocoStatus status) {
    switch (status) {
        case REK_G1_FALL_MUJOCO_OK: return "ok";
        case REK_G1_FALL_MUJOCO_NULL_ARGUMENT: return "null argument";
        case REK_G1_FALL_MUJOCO_INVALID_DUEL: return "invalid duel";
        case REK_G1_FALL_MUJOCO_MAPPING_MISSING: return "mapping missing";
        case REK_G1_FALL_MUJOCO_MAPPING_MISMATCH: return "mapping mismatch";
        case REK_G1_FALL_MUJOCO_ALLOCATION_FAILED: return "allocation failed";
        case REK_G1_FALL_MUJOCO_NOT_CALIBRATED: return "not calibrated";
        case REK_G1_FALL_MUJOCO_NON_FINITE: return "non-finite measurement";
        case REK_G1_FALL_MUJOCO_MEASUREMENT_INVALID:
            return "invalid measurement";
        default: return "unknown MuJoCo fall status";
    }
}

RekG1FallMujocoStatus rek_g1_fall_mujoco_open(
        RekG1FallMujocoAdapter* adapter,
        GearSonicNativeDuelVector* duel,
        char* error,
        size_t error_capacity) {
    if (adapter == NULL || duel == NULL) {
        set_error(error, error_capacity, "open fall adapter", "null argument");
        return REK_G1_FALL_MUJOCO_NULL_ARGUMENT;
    }
    memset(adapter, 0, sizeof(*adapter));
    if (!duel_shape_valid(duel)) {
        return fail(
            adapter, REK_G1_FALL_MUJOCO_INVALID_DUEL,
            error, error_capacity, "open fall adapter",
            "duel is not the exact two-fighter model");
    }
    adapter->duel = duel;
    adapter->robot_count = duel->robot_count;
    adapter->body_count = (size_t)duel->model->nbody;
    adapter->floor_geom_id = mj_name2id(
        duel->model, mjOBJ_GEOM, FLOOR_GEOM);
    for (size_t fighter = 0u;
            fighter < GEAR_SONIC_DUEL_FIGHTERS;
            fighter++) {
        adapter->root_body_ids[fighter] = mj_name2id(
            duel->model, mjOBJ_BODY, ROOT_BODIES[fighter]);
        adapter->left_foot_body_ids[fighter] = mj_name2id(
            duel->model, mjOBJ_BODY, LEFT_FOOT_BODIES[fighter]);
        adapter->right_foot_body_ids[fighter] = mj_name2id(
            duel->model, mjOBJ_BODY, RIGHT_FOOT_BODIES[fighter]);
        if (adapter->root_body_ids[fighter] < 0
                || adapter->left_foot_body_ids[fighter] < 0
                || adapter->right_foot_body_ids[fighter] < 0) {
            return fail(
                adapter, REK_G1_FALL_MUJOCO_MAPPING_MISSING,
                error, error_capacity, "open fall adapter",
                "required G1 body name is missing");
        }
    }
    if (adapter->floor_geom_id < 0) {
        return fail(
            adapter, REK_G1_FALL_MUJOCO_MAPPING_MISSING,
            error, error_capacity, "open fall adapter",
            "exact arena floor geom is missing");
    }
    const mjModel* model = duel->model;
    if (!exact_name(model, mjOBJ_GEOM, adapter->floor_geom_id, FLOOR_GEOM)
            || model->geom_type[adapter->floor_geom_id] != mjGEOM_BOX
            || model->geom_bodyid[adapter->floor_geom_id] != 0) {
        return fail(
            adapter, REK_G1_FALL_MUJOCO_MAPPING_MISMATCH,
            error, error_capacity, "open fall adapter",
            "floor geom is not the exact fixed world box");
    }
    for (size_t fighter = 0u;
            fighter < GEAR_SONIC_DUEL_FIGHTERS;
            fighter++) {
        const int root = adapter->root_body_ids[fighter];
        const int left = adapter->left_foot_body_ids[fighter];
        const int right = adapter->right_foot_body_ids[fighter];
        if (!exact_name(model, mjOBJ_BODY, root, ROOT_BODIES[fighter])
                || !exact_name(
                    model, mjOBJ_BODY, left, LEFT_FOOT_BODIES[fighter])
                || !exact_name(
                    model, mjOBJ_BODY, right, RIGHT_FOOT_BODIES[fighter])
                || !root_map_valid(model, &duel->fighters[fighter], root)
                || root == left || root == right || left == right
                || !body_descends_from(model, left, root)
                || !body_descends_from(model, right, root)) {
            return fail(
                adapter, REK_G1_FALL_MUJOCO_MAPPING_MISMATCH,
                error, error_capacity, "open fall adapter",
                "G1 root or foot body mapping does not match the duel map");
        }
    }

    size_t owner_bytes = 0u;
    size_t seen_bytes = 0u;
    size_t calibration_bytes = 0u;
    if (!checked_array_bytes(
            adapter->body_count, sizeof(*adapter->body_owner), &owner_bytes)
            || !checked_array_bytes(
                adapter->body_count,
                sizeof(*adapter->contact_seen),
                &seen_bytes)
            || !checked_array_bytes(
                adapter->robot_count,
                sizeof(*adapter->calibrations),
                &calibration_bytes)) {
        return fail(
            adapter, REK_G1_FALL_MUJOCO_ALLOCATION_FAILED,
            error, error_capacity, "open fall adapter",
            "scratch allocation size overflow");
    }
    adapter->body_owner = malloc(owner_bytes);
    adapter->contact_seen = malloc(seen_bytes);
    adapter->calibrations = calloc(1u, calibration_bytes);
    if (adapter->body_owner == NULL || adapter->contact_seen == NULL
            || adapter->calibrations == NULL) {
        rek_g1_fall_mujoco_close(adapter);
        set_error(error, error_capacity, "open fall adapter", "allocation failed");
        adapter->last_status = REK_G1_FALL_MUJOCO_ALLOCATION_FAILED;
        return adapter->last_status;
    }
    if (!map_body_owners(adapter)) {
        rek_g1_fall_mujoco_close(adapter);
        set_error(
            error, error_capacity, "open fall adapter",
            "fighter body trees overlap or foot ownership is invalid");
        adapter->last_status = REK_G1_FALL_MUJOCO_MAPPING_MISMATCH;
        return adapter->last_status;
    }
    adapter->initialized = 1u;
    adapter->ready = 1u;
    adapter->last_status = REK_G1_FALL_MUJOCO_OK;
    if (error != NULL && error_capacity > 0u) error[0] = '\0';
    return adapter->last_status;
}

static int row_arena_fighter(
        const RekG1FallMujocoAdapter* adapter,
        size_t row,
        size_t* arena,
        size_t* fighter) {
    if (adapter == NULL || arena == NULL || fighter == NULL
            || row >= adapter->robot_count) {
        return 0;
    }
    *arena = row / GEAR_SONIC_DUEL_FIGHTERS;
    *fighter = row % GEAR_SONIC_DUEL_FIGHTERS;
    return *arena < adapter->duel->arena_count;
}

static int measure_floor_height(
        const RekG1FallMujocoAdapter* adapter,
        const mjData* data,
        double* floor_height) {
    if (adapter == NULL || data == NULL || floor_height == NULL
            || adapter->floor_geom_id < 0
            || adapter->floor_geom_id >= adapter->duel->model->ngeom) {
        return 0;
    }
    const int geom_id = adapter->floor_geom_id;
    const mjtNum* position = data->geom_xpos + 3 * geom_id;
    const mjtNum* matrix = data->geom_xmat + 9 * geom_id;
    const mjtNum* size = adapter->duel->model->geom_size + 3 * geom_id;
    if (!finite_values(position, 3u) || !finite_values(matrix, 9u)
            || !finite_values(size, 3u)
            || (double)size[0] <= 0.0 || (double)size[1] <= 0.0
            || (double)size[2] <= 0.0) {
        return 0;
    }
    /* The recovered arena contract has one horizontal, fixed floor box. */
    const double horizontal_tolerance = 1e-10;
    if (fabs((double)matrix[6]) > horizontal_tolerance
            || fabs((double)matrix[7]) > horizontal_tolerance
            || fabs(fabs((double)matrix[8]) - 1.0) > horizontal_tolerance) {
        return 0;
    }
    const double measured = (double)position[2]
        + fabs((double)matrix[8]) * (double)size[2];
    if (!isfinite(measured)) return 0;
    *floor_height = measured;
    return 1;
}

static int normalize_quaternion(
        const mjtNum* source_wxyz, double output_wxyz[4]) {
    if (!finite_values(source_wxyz, 4u) || output_wxyz == NULL) return 0;
    double norm_squared = 0.0;
    for (size_t axis = 0u; axis < 4u; axis++) {
        const double value = (double)source_wxyz[axis];
        norm_squared += value * value;
    }
    if (!isfinite(norm_squared) || norm_squared <= DBL_MIN) return 0;
    const double inverse_norm = 1.0 / sqrt(norm_squared);
    if (!isfinite(inverse_norm)) return 0;
    for (size_t axis = 0u; axis < 4u; axis++) {
        output_wxyz[axis] = (double)source_wxyz[axis] * inverse_norm;
    }
    return 1;
}

static void rotate_vector_by_unit_quaternion(
        const double quaternion_wxyz[4],
        const double input[3],
        double output[3]) {
    const double w = quaternion_wxyz[0];
    const double x = quaternion_wxyz[1];
    const double y = quaternion_wxyz[2];
    const double z = quaternion_wxyz[3];
    const double tx = 2.0 * (y * input[2] - z * input[1]);
    const double ty = 2.0 * (z * input[0] - x * input[2]);
    const double tz = 2.0 * (x * input[1] - y * input[0]);
    output[0] = input[0] + w * tx + (y * tz - z * ty);
    output[1] = input[1] + w * ty + (z * tx - x * tz);
    output[2] = input[2] + w * tz + (x * ty - y * tx);
}

static int calibrate_row(
        RekG1FallMujocoAdapter* adapter,
        size_t row,
        RekG1FallMujocoCalibration* calibration) {
    size_t arena = 0u;
    size_t fighter = 0u;
    if (calibration == NULL
            || !row_arena_fighter(adapter, row, &arena, &fighter)) {
        return 0;
    }
    const GearSonicDuelFighterMap* map = &adapter->duel->fighters[fighter];
    const mjData* data = adapter->duel->data[arena];
    double quaternion[4];
    double floor_height = 0.0;
    const mjtNum* source_quaternion =
        data->qpos + map->root_qpos_address + 3;
    const mjtNum* pelvis = data->xpos + 3 * map->root_body_id;
    if (!normalize_quaternion(source_quaternion, quaternion)
            || !measure_floor_height(adapter, data, &floor_height)
            || !finite_values(pelvis, 3u)) {
        return 0;
    }
    const double inverse[4] = {
        quaternion[0],
        -quaternion[1],
        -quaternion[2],
        -quaternion[3],
    };
    const double world_up[3] = {0.0, 0.0, 1.0};
    RekG1FallMujocoCalibration measured = {0};
    rotate_vector_by_unit_quaternion(
        inverse, world_up, measured.upright_up_local);
    measured.standing_pelvis_height = (double)pelvis[2] - floor_height;
    measured.reset_floor_height = floor_height;
    if (!isfinite(measured.standing_pelvis_height)
            || measured.standing_pelvis_height <= DBL_EPSILON
            || !isfinite(measured.upright_up_local[0])
            || !isfinite(measured.upright_up_local[1])
            || !isfinite(measured.upright_up_local[2])) {
        return 0;
    }
    measured.calibrated = 1u;
    *calibration = measured;
    return 1;
}

RekG1FallMujocoStatus rek_g1_fall_mujoco_calibrate_reset(
        RekG1FallMujocoAdapter* adapter,
        char* error,
        size_t error_capacity) {
    if (adapter == NULL) {
        set_error(error, error_capacity, "calibrate fall adapter", "null argument");
        return REK_G1_FALL_MUJOCO_NULL_ARGUMENT;
    }
    if (!adapter->ready || adapter->duel == NULL
            || !duel_shape_valid(adapter->duel)
            || adapter->calibrations == NULL) {
        return fail(
            adapter, REK_G1_FALL_MUJOCO_INVALID_DUEL,
            error, error_capacity, "calibrate fall adapter",
            "adapter or duel is not ready");
    }
    for (size_t row = 0u; row < adapter->robot_count; row++) {
        adapter->calibrations[row].calibrated = 0u;
    }
    for (size_t row = 0u; row < adapter->robot_count; row++) {
        if (!calibrate_row(adapter, row, &adapter->calibrations[row])) {
            for (size_t clear = 0u;
                    clear < adapter->robot_count;
                    clear++) {
                adapter->calibrations[clear].calibrated = 0u;
            }
            return fail(
                adapter, REK_G1_FALL_MUJOCO_MEASUREMENT_INVALID,
                error, error_capacity, "calibrate fall adapter",
                "reset upright pose, pelvis height, or floor is unavailable");
        }
    }
    adapter->last_status = REK_G1_FALL_MUJOCO_OK;
    if (error != NULL && error_capacity > 0u) error[0] = '\0';
    return adapter->last_status;
}

static int sample_contacts(
        RekG1FallMujocoAdapter* adapter,
        const mjData* data,
        size_t fighter,
        uint8_t* left_contact,
        uint8_t* right_contact,
        uint32_t* distinct_nonfoot_count) {
    if (adapter == NULL || data == NULL || left_contact == NULL
            || right_contact == NULL || distinct_nonfoot_count == NULL
            || fighter >= GEAR_SONIC_DUEL_FIGHTERS
            || data->ncon < 0
            || (data->ncon > 0 && data->contact == NULL)
            || adapter->contact_seen == NULL) {
        return 0;
    }
    memset(adapter->contact_seen, 0, adapter->body_count);
    uint32_t count = 0u;
    uint8_t left = 0u;
    uint8_t right = 0u;
    const mjModel* model = adapter->duel->model;
    for (int contact_index = 0; contact_index < data->ncon; contact_index++) {
        const mjContact* contact = &data->contact[contact_index];
        const int geom0 = contact->geom[0];
        const int geom1 = contact->geom[1];
        if (geom0 < 0 || geom0 >= model->ngeom
                || geom1 < 0 || geom1 >= model->ngeom) {
            return 0;
        }
        int other_geom = -1;
        if (geom0 == adapter->floor_geom_id
                && geom1 != adapter->floor_geom_id) {
            other_geom = geom1;
        } else if (geom1 == adapter->floor_geom_id
                && geom0 != adapter->floor_geom_id) {
            other_geom = geom0;
        } else {
            continue;
        }
        const int body_id = model->geom_bodyid[other_geom];
        if (body_id <= 0 || (size_t)body_id >= adapter->body_count
                || adapter->body_owner[body_id] != (int8_t)fighter) {
            continue;
        }
        if (body_id == adapter->left_foot_body_ids[fighter]) {
            left = 1u;
        } else if (body_id == adapter->right_foot_body_ids[fighter]) {
            right = 1u;
        } else if (!adapter->contact_seen[body_id]) {
            if (count == UINT32_MAX) return 0;
            adapter->contact_seen[body_id] = 1u;
            count++;
        }
    }
    *left_contact = left;
    *right_contact = right;
    *distinct_nonfoot_count = count;
    return 1;
}

RekG1FallMujocoStatus rek_g1_fall_mujoco_sample(
        RekG1FallMujocoAdapter* adapter,
        size_t robot_row,
        float fixed_delta_seconds,
        uint8_t can_get_up,
        RekG1FallMujocoMeasurement* output,
        char* error,
        size_t error_capacity) {
    if (adapter == NULL || output == NULL) {
        set_error(error, error_capacity, "sample fall adapter", "null argument");
        return REK_G1_FALL_MUJOCO_NULL_ARGUMENT;
    }
    if (!adapter->ready || adapter->duel == NULL
            || !duel_shape_valid(adapter->duel)) {
        return fail(
            adapter, REK_G1_FALL_MUJOCO_INVALID_DUEL,
            error, error_capacity, "sample fall adapter",
            "adapter or duel is not ready");
    }
    if (robot_row >= adapter->robot_count
            || !isfinite(fixed_delta_seconds)
            || fixed_delta_seconds <= 0.0f
            || can_get_up > 1u) {
        return fail(
            adapter, REK_G1_FALL_MUJOCO_MEASUREMENT_INVALID,
            error, error_capacity, "sample fall adapter",
            "row, fixed delta, or CanGetUp fact is invalid");
    }
    const RekG1FallMujocoCalibration* calibration =
        &adapter->calibrations[robot_row];
    if (!calibration->calibrated) {
        return fail(
            adapter, REK_G1_FALL_MUJOCO_NOT_CALIBRATED,
            error, error_capacity, "sample fall adapter",
            "reset calibration is unavailable");
    }
    size_t arena = 0u;
    size_t fighter = 0u;
    if (!row_arena_fighter(adapter, robot_row, &arena, &fighter)) {
        return fail(
            adapter, REK_G1_FALL_MUJOCO_MEASUREMENT_INVALID,
            error, error_capacity, "sample fall adapter", "row mapping failed");
    }
    const GearSonicDuelFighterMap* map = &adapter->duel->fighters[fighter];
    const mjData* data = adapter->duel->data[arena];
    const mjtNum* source_quaternion =
        data->qpos + map->root_qpos_address + 3;
    const mjtNum* pelvis = data->xpos + 3 * map->root_body_id;
    double quaternion[4];
    double world_up[3];
    double floor_height = 0.0;
    uint8_t left_contact = 0u;
    uint8_t right_contact = 0u;
    uint32_t nonfoot_count = 0u;
    if (!normalize_quaternion(source_quaternion, quaternion)
            || !finite_values(pelvis, 3u)
            || !measure_floor_height(adapter, data, &floor_height)
            || !sample_contacts(
                adapter,
                data,
                fighter,
                &left_contact,
                &right_contact,
                &nonfoot_count)) {
        return fail(
            adapter, REK_G1_FALL_MUJOCO_NON_FINITE,
            error, error_capacity, "sample fall adapter",
            "root pose, floor, or contact facts are unavailable");
    }
    rotate_vector_by_unit_quaternion(
        quaternion, calibration->upright_up_local, world_up);
    double up_norm_squared = 0.0;
    for (size_t axis = 0u; axis < 3u; axis++) {
        up_norm_squared += world_up[axis] * world_up[axis];
    }
    if (!isfinite(up_norm_squared) || up_norm_squared <= DBL_MIN
            || !isfinite(calibration->standing_pelvis_height)
            || calibration->standing_pelvis_height <= DBL_EPSILON) {
        return fail(
            adapter, REK_G1_FALL_MUJOCO_NON_FINITE,
            error, error_capacity, "sample fall adapter",
            "upright calibration is invalid");
    }
    double up_dot = world_up[2] / sqrt(up_norm_squared);
    if (up_dot < -1.0) up_dot = -1.0;
    if (up_dot > 1.0) up_dot = 1.0;
    const double tilt_degrees = acos(up_dot) * RADIANS_TO_DEGREES;
    const double height_ratio = ((double)pelvis[2] - floor_height)
        / calibration->standing_pelvis_height;
    const float tilt_f32 = (float)tilt_degrees;
    const float ratio_f32 = (float)height_ratio;
    const float floor_f32 = (float)floor_height;
    const float standing_f32 = (float)calibration->standing_pelvis_height;
    if (!isfinite(tilt_degrees) || !isfinite(height_ratio)
            || !isfinite(tilt_f32) || !isfinite(ratio_f32)
            || !isfinite(floor_f32) || !isfinite(standing_f32)) {
        return fail(
            adapter, REK_G1_FALL_MUJOCO_NON_FINITE,
            error, error_capacity, "sample fall adapter",
            "derived fall measurement is non-finite");
    }
    const uint8_t has_foot_contact = left_contact || right_contact;
    RekG1FallMujocoMeasurement measured = {
        .fall_sample = {
            .tracking_active = 1u,
            .tilt_degrees = tilt_f32,
            .pelvis_height_ratio = ratio_f32,
            .both_feet_off_floor = has_foot_contact ? 0u : 1u,
            .has_foot_body_contact = has_foot_contact,
            .distinct_nonfoot_body_contact_count = nonfoot_count,
            .fixed_delta_seconds = fixed_delta_seconds,
            .can_get_up = can_get_up,
        },
        .left_foot_body_contact = left_contact,
        .right_foot_body_contact = right_contact,
        .floor_height = floor_f32,
        .standing_pelvis_height = standing_f32,
    };
    *output = measured;
    adapter->last_status = REK_G1_FALL_MUJOCO_OK;
    if (error != NULL && error_capacity > 0u) error[0] = '\0';
    return adapter->last_status;
}

void rek_g1_fall_mujoco_close(RekG1FallMujocoAdapter* adapter) {
    if (adapter == NULL) return;
    free(adapter->calibrations);
    free(adapter->contact_seen);
    free(adapter->body_owner);
    memset(adapter, 0, sizeof(*adapter));
}
