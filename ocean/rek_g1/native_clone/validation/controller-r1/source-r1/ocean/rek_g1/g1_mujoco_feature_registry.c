#include "g1_mujoco_feature_registry.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* const PLAYER_ROOT_BODY = "player__pelvis_3266";
static const char* const PLAYER_LEFT_ANKLE_ROLL_BODY =
    "player__left_ankle_roll_link_3045";
static const char* const PLAYER_RIGHT_ANKLE_ROLL_BODY =
    "player__right_ankle_roll_link_3090";
static const char* const PLAYER_JOINT_NAMES[GEAR_SONIC_ACTION_DIM] = {
    "player__joint__left_hip_pitch_joint_3047",
    "player__joint__left_hip_roll_joint_3248",
    "player__joint__left_hip_yaw_joint_3267",
    "player__joint__left_knee_joint_3137",
    "player__joint__left_ankle_pitch_joint_2982",
    "player__joint__left_ankle_roll_joint_2905",
    "player__joint__right_hip_pitch_joint_3298",
    "player__joint__right_hip_roll_joint_3059",
    "player__joint__right_hip_yaw_joint_3071",
    "player__joint__right_knee_joint_3412",
    "player__joint__right_ankle_pitch_joint_3312",
    "player__joint__right_ankle_roll_joint_3474",
    "player__joint__waist_yaw_joint_3441",
    "player__joint__waist_roll_joint_3341",
    "player__joint__waist_pitch_joint_3233",
    "player__joint__left_shoulder_pitch_joint_3340",
    "player__joint__left_shoulder_roll_joint_3184",
    "player__joint__left_shoulder_yaw_joint_2923",
    "player__joint__left_elbow_joint_3144",
    "player__joint__left_wrist_roll_joint_3260",
    "player__joint__left_wrist_pitch_joint_3007",
    "player__joint__left_wrist_yaw_joint_3398",
    "player__joint__right_shoulder_pitch_joint_3242",
    "player__joint__right_shoulder_roll_joint_3044",
    "player__joint__right_shoulder_yaw_joint_3176",
    "player__joint__right_elbow_joint_3407",
    "player__joint__right_wrist_roll_joint_3378",
    "player__joint__right_wrist_pitch_joint_3437",
    "player__joint__right_wrist_yaw_joint_3226",
};

static void set_error(
        char* error,
        size_t error_capacity,
        const char* operation,
        const char* detail) {
    if (error == NULL || error_capacity == 0u) return;
    if (operation == NULL) operation = "build MuJoCo foot features";
    if (detail == NULL) detail = "unknown failure";
    (void)snprintf(error, error_capacity, "%s: %s", operation, detail);
    error[error_capacity - 1u] = '\0';
}

static int checked_add(size_t a, size_t b, size_t* result) {
    if (result == NULL || b > SIZE_MAX - a) return 0;
    *result = a + b;
    return 1;
}

static int checked_product(size_t a, size_t b, size_t* result) {
    if (result == NULL || (a != 0u && b > SIZE_MAX / a)) return 0;
    *result = a * b;
    return 1;
}

static int finite_float_values(const float* values, size_t count) {
    if (values == NULL) return 0;
    for (size_t index = 0; index < count; index++) {
        if (!isfinite(values[index])) return 0;
    }
    return 1;
}

static int finite_mjt_values(const mjtNum* values, size_t count) {
    if (values == NULL) return 0;
    for (size_t index = 0; index < count; index++) {
        if (!isfinite((double)values[index])) return 0;
    }
    return 1;
}

static double f64_add(double a, double b) {
    volatile double result = a + b;
    return result;
}

static double f64_sub(double a, double b) {
    volatile double result = a - b;
    return result;
}

static double f64_mul(double a, double b) {
    volatile double result = a * b;
    return result;
}

static float f32_from_f64(double value) {
    volatile float result = (float)value;
    return result;
}

static int body_descends_from(
        const mjModel* model, int body_id, int ancestor_id) {
    if (model == NULL || body_id <= 0 || body_id >= model->nbody
            || ancestor_id <= 0 || ancestor_id >= model->nbody) {
        return 0;
    }
    for (int current = body_id; current > 0; current = model->body_parentid[current]) {
        if (current == ancestor_id) return 1;
    }
    return 0;
}

static int exact_body_name(
        const mjModel* model, int body_id, const char* expected) {
    if (model == NULL || body_id <= 0 || body_id >= model->nbody
            || expected == NULL) {
        return 0;
    }
    const char* actual = mj_id2name(model, mjOBJ_BODY, body_id);
    return actual != NULL && strcmp(actual, expected) == 0;
}

static RekG1MujocoFeatureRegistryStatus validate_duel_and_map(
        GearSonicNativeDuelVector* duel,
        int* root_body_id,
        int* left_body_id,
        int* right_body_id,
        char* error,
        size_t error_capacity) {
    if (duel == NULL || root_body_id == NULL || left_body_id == NULL
            || right_body_id == NULL) {
        set_error(error, error_capacity, "validate duel", "null argument");
        return REK_G1_MUJOCO_FEATURE_REGISTRY_NULL_ARGUMENT;
    }
    const mjModel* model = duel->model;
    if (model == NULL || duel->failed || !duel->spawn_prefixes_verified
            || duel->data == NULL || duel->arena_count == 0u
            || duel->arena_count > SIZE_MAX / GEAR_SONIC_DUEL_FIGHTERS
            || duel->robot_count != duel->arena_count * GEAR_SONIC_DUEL_FIGHTERS
            || model->nq != GEAR_SONIC_DUEL_QPOS_DIM
            || model->nv != GEAR_SONIC_DUEL_QVEL_DIM
            || model->nu != GEAR_SONIC_DUEL_CONTROL_DIM) {
        set_error(error, error_capacity, "validate duel", "duel is not an opened exact two-fighter model");
        return REK_G1_MUJOCO_FEATURE_REGISTRY_INVALID_DUEL;
    }
    const GearSonicDuelFighterMap* map =
        &duel->fighters[GEAR_SONIC_DUEL_PLAYER];
    const int root = mj_name2id(model, mjOBJ_BODY, PLAYER_ROOT_BODY);
    const int left = mj_name2id(
        model, mjOBJ_BODY, PLAYER_LEFT_ANKLE_ROLL_BODY);
    const int right = mj_name2id(
        model, mjOBJ_BODY, PLAYER_RIGHT_ANKLE_ROLL_BODY);
    if (root < 0 || left < 0 || right < 0) {
        set_error(error, error_capacity, "validate duel", "player kinematic body name is missing");
        return REK_G1_MUJOCO_FEATURE_REGISTRY_MAPPING_MISSING;
    }
    if (root != map->root_body_id
            || !exact_body_name(model, map->root_body_id, PLAYER_ROOT_BODY)
            || root == left || root == right || left == right
            || !body_descends_from(model, left, root)
            || !body_descends_from(model, right, root)) {
        set_error(error, error_capacity, "validate duel", "player body mapping mismatch");
        return REK_G1_MUJOCO_FEATURE_REGISTRY_MAPPING_MISMATCH;
    }
    if (model->body_jntnum[root] < 1) {
        set_error(error, error_capacity, "validate duel", "player root has no free joint");
        return REK_G1_MUJOCO_FEATURE_REGISTRY_MAPPING_MISMATCH;
    }
    const int root_joint = model->body_jntadr[root];
    if (root_joint < 0 || root_joint >= model->njnt
            || model->jnt_type[root_joint] != mjJNT_FREE
            || model->jnt_qposadr[root_joint] != map->root_qpos_address
            || model->jnt_dofadr[root_joint] != map->root_qvel_address
            || map->root_qpos_address < 0
            || map->root_qpos_address + 7 > model->nq) {
        set_error(error, error_capacity, "validate duel", "player free-joint mapping mismatch");
        return REK_G1_MUJOCO_FEATURE_REGISTRY_MAPPING_MISMATCH;
    }
    uint8_t seen_qpos[GEAR_SONIC_DUEL_QPOS_DIM] = {0};
    for (size_t index = 0; index < GEAR_SONIC_ACTION_DIM; index++) {
        const int joint_id = map->joint_ids[index];
        const int qpos_address = map->qpos_addresses[index];
        const int expected_joint_id = mj_name2id(
            model, mjOBJ_JOINT, PLAYER_JOINT_NAMES[index]);
        if (expected_joint_id < 0) {
            set_error(error, error_capacity, "validate duel", "player joint name is missing");
            return REK_G1_MUJOCO_FEATURE_REGISTRY_MAPPING_MISSING;
        }
        if (joint_id != expected_joint_id
                || joint_id < 0 || joint_id >= model->njnt
                || qpos_address < 0 || qpos_address >= model->nq
                || model->jnt_type[joint_id] != mjJNT_HINGE
                || model->jnt_qposadr[joint_id] != qpos_address
                || seen_qpos[qpos_address]) {
            set_error(error, error_capacity, "validate duel", "player 29-DOF mapping mismatch");
            return REK_G1_MUJOCO_FEATURE_REGISTRY_MAPPING_MISMATCH;
        }
        seen_qpos[qpos_address] = 1u;
    }
    *root_body_id = root;
    *left_body_id = left;
    *right_body_id = right;
    return REK_G1_MUJOCO_FEATURE_REGISTRY_OK;
}

static int same_clip_storage(
        const SonicMotionComposerNativeClip* clip,
        const RekG1SemanticClipStorage* storage) {
    return clip != NULL && storage != NULL
        && clip->dof_position_mujoco == storage->dof_position_mujoco
        && clip->root_quaternion_wxyz == storage->root_quaternion_wxyz
        && clip->frame_count == storage->frame_count
        && clip->dof_position_count
            == storage->frame_count * SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT
        && clip->root_quaternion_count == storage->frame_count * 4u;
}

static const RekG1SemanticClipStorage* storage_by_path_id(
        const RekG1SemanticAssets* assets, int32_t npz_path_id) {
    if (assets == NULL) return NULL;
    for (size_t index = 0;
            index < REK_G1_MUJOCO_FEATURE_CLIP_COUNT;
            index++) {
        if (assets->clips[index].npz_path_id == npz_path_id) {
            return &assets->clips[index];
        }
    }
    return NULL;
}

static RekG1MujocoFeatureRegistryStatus validate_assets_and_build_views(
        RekG1MujocoFeatureRegistry* registry,
        const RekG1SemanticAssets* assets,
        char* error,
        size_t error_capacity) {
    if (registry == NULL || assets == NULL) {
        set_error(error, error_capacity, "validate feature assets", "null argument");
        return REK_G1_MUJOCO_FEATURE_REGISTRY_NULL_ARGUMENT;
    }
    if (!assets->loaded) {
        set_error(error, error_capacity, "validate feature assets", "semantic assets are not loaded");
        return REK_G1_MUJOCO_FEATURE_REGISTRY_INVALID_ASSETS;
    }
    const RekG1NativeMotionRouteTable* route_table =
        rek_g1_native_static_motion_routes();
    if (!rek_g1_native_validate_static_motion_routes(route_table)) {
        set_error(error, error_capacity, "validate feature assets", "static route table is invalid");
        return REK_G1_MUJOCO_FEATURE_REGISTRY_INVALID_ASSETS;
    }
    for (size_t route_index = 0;
            route_index < REK_G1_STATIC_ROUTE_COUNT;
            route_index++) {
        const RekG1NativeMotionRoute* expected =
            &route_table->routes[route_index];
        const RekG1SemanticDuelRouteAsset* actual =
            &assets->route_assets[route_index];
        const RekG1SemanticClipStorage* storage = storage_by_path_id(
            assets, expected->npz_path_id);
        if ((size_t)actual->route_id != route_index || storage == NULL
                || !same_clip_storage(&actual->clip, storage)
                || actual->clip.frame_count != expected->asset_frames
                || actual->clip.fps != expected->asset_fps) {
            set_error(error, error_capacity, "validate feature assets", "route asset contract mismatch");
            return REK_G1_MUJOCO_FEATURE_REGISTRY_INVALID_ASSETS;
        }
    }
    uint8_t route_covered[REK_G1_STATIC_ROUTE_COUNT] = {0};
    size_t total_frames = 0u;
    for (size_t clip_index = 0;
            clip_index < REK_G1_MUJOCO_FEATURE_CLIP_COUNT;
            clip_index++) {
        const RekG1SemanticClipStorage* storage = &assets->clips[clip_index];
        if (storage->dof_position_mujoco == NULL
                || storage->root_quaternion_wxyz == NULL
                || storage->frame_count == 0u
                || storage->frame_count > SIZE_MAX / 4u
                || storage->frame_count
                    > SIZE_MAX / SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT) {
            set_error(error, error_capacity, "validate feature assets", "clip storage is invalid");
            return REK_G1_MUJOCO_FEATURE_REGISTRY_INVALID_ASSETS;
        }
        for (size_t prior = 0; prior < clip_index; prior++) {
            if (storage->dof_position_mujoco
                        == assets->clips[prior].dof_position_mujoco
                    || storage->root_quaternion_wxyz
                        == assets->clips[prior].root_quaternion_wxyz
                    || storage->npz_path_id == assets->clips[prior].npz_path_id) {
                set_error(error, error_capacity, "validate feature assets", "clip identities are not unique");
                return REK_G1_MUJOCO_FEATURE_REGISTRY_INVALID_ASSETS;
            }
        }
        const SonicMotionComposerNativeClip* selected = NULL;
        for (size_t route_index = 0;
                route_index < REK_G1_STATIC_ROUTE_COUNT;
                route_index++) {
            const RekG1SemanticDuelRouteAsset* route =
                &assets->route_assets[route_index];
            if ((size_t)route->route_id != route_index) {
                set_error(error, error_capacity, "validate feature assets", "route identity mismatch");
                return REK_G1_MUJOCO_FEATURE_REGISTRY_INVALID_ASSETS;
            }
            if (same_clip_storage(&route->clip, storage)) {
                if (!isfinite(route->clip.fps) || route->clip.fps <= 0.0f) {
                    set_error(error, error_capacity, "validate feature assets", "clip fps is invalid");
                    return REK_G1_MUJOCO_FEATURE_REGISTRY_NON_FINITE;
                }
                route_covered[route_index] = 1u;
                if (selected == NULL) selected = &route->clip;
            }
        }
        if (selected == NULL) {
            set_error(error, error_capacity, "validate feature assets", "unique clip has no semantic route");
            return REK_G1_MUJOCO_FEATURE_REGISTRY_INVALID_ASSETS;
        }
        const size_t dof_count = storage->frame_count
            * SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT;
        const size_t root_count = storage->frame_count * 4u;
        if (!finite_float_values(storage->dof_position_mujoco, dof_count)
                || !finite_float_values(
                    storage->root_quaternion_wxyz, root_count)) {
            set_error(error, error_capacity, "validate feature assets", "clip contains non-finite values");
            return REK_G1_MUJOCO_FEATURE_REGISTRY_NON_FINITE;
        }
        registry->clip_views[clip_index] = *selected;
        if (!checked_add(total_frames, storage->frame_count, &total_frames)) {
            set_error(error, error_capacity, "validate feature assets", "frame count overflow");
            return REK_G1_MUJOCO_FEATURE_REGISTRY_SIZE_OVERFLOW;
        }
    }
    for (size_t route_index = 0;
            route_index < REK_G1_STATIC_ROUTE_COUNT;
            route_index++) {
        if (!route_covered[route_index]) {
            set_error(error, error_capacity, "validate feature assets", "semantic route is not covered by a unique clip");
            return REK_G1_MUJOCO_FEATURE_REGISTRY_INVALID_ASSETS;
        }
    }
    if (!checked_product(
            total_frames,
            SONIC_MOTION_ENTRY_MATCHER_NATIVE_FEATURE_WIDTH,
            &registry->total_feature_count)) {
        set_error(error, error_capacity, "validate feature assets", "feature count overflow");
        return REK_G1_MUJOCO_FEATURE_REGISTRY_SIZE_OVERFLOW;
    }
    registry->total_frame_count = total_frames;
    return REK_G1_MUJOCO_FEATURE_REGISTRY_OK;
}

const char* rek_g1_mujoco_feature_registry_status_string(
        RekG1MujocoFeatureRegistryStatus status) {
    switch (status) {
        case REK_G1_MUJOCO_FEATURE_REGISTRY_OK: return "ok";
        case REK_G1_MUJOCO_FEATURE_REGISTRY_NULL_ARGUMENT: return "null argument";
        case REK_G1_MUJOCO_FEATURE_REGISTRY_INVALID_DUEL: return "invalid duel";
        case REK_G1_MUJOCO_FEATURE_REGISTRY_INVALID_ASSETS: return "invalid assets";
        case REK_G1_MUJOCO_FEATURE_REGISTRY_MAPPING_MISSING: return "kinematic mapping missing";
        case REK_G1_MUJOCO_FEATURE_REGISTRY_MAPPING_MISMATCH: return "kinematic mapping mismatch";
        case REK_G1_MUJOCO_FEATURE_REGISTRY_SIZE_OVERFLOW: return "size overflow";
        case REK_G1_MUJOCO_FEATURE_REGISTRY_ALLOCATION_FAILED: return "allocation failed";
        case REK_G1_MUJOCO_FEATURE_REGISTRY_NON_FINITE: return "non-finite value";
        case REK_G1_MUJOCO_FEATURE_REGISTRY_MATCHER_FAILED: return "motion matcher failed";
        case REK_G1_MUJOCO_FEATURE_REGISTRY_BACKEND_FAILED: return "delegated math backend failed";
        case REK_G1_MUJOCO_FEATURE_REGISTRY_NOT_READY: return "registry not ready";
        default: return "unknown feature registry status";
    }
}

int rek_g1_mujoco_feature_registry_sample(
        void* context,
        const float dof_position_mujoco[
            SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT],
        float output[SONIC_MOTION_ENTRY_MATCHER_NATIVE_FEATURE_WIDTH]) {
    RekG1MujocoFeatureRegistry* registry =
        (RekG1MujocoFeatureRegistry*)context;
    if (output != NULL) {
        memset(
            output,
            0,
            SONIC_MOTION_ENTRY_MATCHER_NATIVE_FEATURE_WIDTH * sizeof(float));
    }
    if (registry == NULL || dof_position_mujoco == NULL || output == NULL) {
        if (registry != NULL) {
            registry->last_status =
                REK_G1_MUJOCO_FEATURE_REGISTRY_NULL_ARGUMENT;
        }
        return 0;
    }
    if (!registry->initialized || registry->duel == NULL
            || registry->duel->model == NULL || registry->scratch == NULL
            || registry->root_body_id <= 0
            || registry->left_ankle_roll_body_id <= 0
            || registry->right_ankle_roll_body_id <= 0) {
        registry->last_status = REK_G1_MUJOCO_FEATURE_REGISTRY_NOT_READY;
        return 0;
    }
    if (!finite_float_values(
            dof_position_mujoco,
            SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT)) {
        registry->last_status = REK_G1_MUJOCO_FEATURE_REGISTRY_NON_FINITE;
        return 0;
    }

    const mjModel* model = registry->duel->model;
    const GearSonicDuelFighterMap* map =
        &registry->duel->fighters[GEAR_SONIC_DUEL_PLAYER];
    mjData* data = registry->scratch;
    mj_resetData(model, data);
    if (!finite_mjt_values(data->qpos, (size_t)model->nq)) {
        registry->last_status = REK_G1_MUJOCO_FEATURE_REGISTRY_NON_FINITE;
        return 0;
    }
    for (size_t index = 0;
            index < SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT;
            index++) {
        const int address = map->qpos_addresses[index];
        if (address < 0 || address >= model->nq) {
            registry->last_status =
                REK_G1_MUJOCO_FEATURE_REGISTRY_MAPPING_MISMATCH;
            return 0;
        }
        data->qpos[address] = (mjtNum)dof_position_mujoco[index];
    }

    mj_kinematics(model, data);
    const mjtNum* root = data->xpos + 3 * registry->root_body_id;
    const mjtNum* root_matrix = data->xmat + 9 * registry->root_body_id;
    const int foot_ids[2] = {
        registry->left_ankle_roll_body_id,
        registry->right_ankle_roll_body_id,
    };
    if (!finite_mjt_values(root, 3u)
            || !finite_mjt_values(root_matrix, 9u)) {
        registry->last_status = REK_G1_MUJOCO_FEATURE_REGISTRY_NON_FINITE;
        return 0;
    }
    for (size_t foot_index = 0; foot_index < 2u; foot_index++) {
        const mjtNum* foot = data->xpos + 3 * foot_ids[foot_index];
        if (!finite_mjt_values(foot, 3u)) {
            registry->last_status = REK_G1_MUJOCO_FEATURE_REGISTRY_NON_FINITE;
            return 0;
        }
        double delta[3];
        double local_mujoco[3];
        for (size_t axis = 0; axis < 3u; axis++) {
            delta[axis] = f64_sub((double)foot[axis], (double)root[axis]);
        }
        /* xmat maps local to world. Unity InverseTransformPoint is xmat^T. */
        for (size_t local_axis = 0; local_axis < 3u; local_axis++) {
            double value = f64_mul(
                (double)root_matrix[local_axis], delta[0]);
            value = f64_add(
                value,
                f64_mul((double)root_matrix[3u + local_axis], delta[1]));
            value = f64_add(
                value,
                f64_mul((double)root_matrix[6u + local_axis], delta[2]));
            local_mujoco[local_axis] = value;
        }
        if (!isfinite(local_mujoco[0]) || !isfinite(local_mujoco[1])
                || !isfinite(local_mujoco[2])) {
            registry->last_status = REK_G1_MUJOCO_FEATURE_REGISTRY_NON_FINITE;
            return 0;
        }
        const size_t output_offset = foot_index * 3u;
        /* Current runner arrays are Unity local xyz. MuJoCo is x,z,y. */
        output[output_offset] = f32_from_f64(local_mujoco[0]);
        output[output_offset + 1u] = f32_from_f64(local_mujoco[2]);
        output[output_offset + 2u] = f32_from_f64(local_mujoco[1]);
    }
    if (!finite_float_values(
            output, SONIC_MOTION_ENTRY_MATCHER_NATIVE_FEATURE_WIDTH)) {
        memset(
            output,
            0,
            SONIC_MOTION_ENTRY_MATCHER_NATIVE_FEATURE_WIDTH * sizeof(float));
        registry->last_status = REK_G1_MUJOCO_FEATURE_REGISTRY_NON_FINITE;
        return 0;
    }
    registry->last_status = REK_G1_MUJOCO_FEATURE_REGISTRY_OK;
    return 1;
}

static int adapter_slerp(
        void* context,
        const float a_wxyz[4],
        const float b_wxyz[4],
        float t,
        float output_wxyz[4]) {
    RekG1MujocoFeatureRegistry* registry = context;
    if (registry == NULL || !registry->ready
            || registry->delegated_backends.quaternion_slerp == NULL
            || !registry->delegated_backends.quaternion_slerp(
                registry->delegated_backends.context,
                a_wxyz,
                b_wxyz,
                t,
                output_wxyz)) {
        if (registry != NULL) {
            registry->last_status =
                REK_G1_MUJOCO_FEATURE_REGISTRY_BACKEND_FAILED;
        }
        return 0;
    }
    return 1;
}

static int adapter_atan2(
        void* context,
        float numerator,
        float denominator,
        float* output) {
    RekG1MujocoFeatureRegistry* registry = context;
    if (registry == NULL || !registry->ready
            || registry->delegated_backends.atan2_f == NULL
            || !registry->delegated_backends.atan2_f(
                registry->delegated_backends.context,
                numerator,
                denominator,
                output)) {
        if (registry != NULL) {
            registry->last_status =
                REK_G1_MUJOCO_FEATURE_REGISTRY_BACKEND_FAILED;
        }
        return 0;
    }
    return 1;
}

static int adapter_sin_cos(
        void* context,
        float angle,
        float* sine,
        float* cosine) {
    RekG1MujocoFeatureRegistry* registry = context;
    if (registry == NULL || !registry->ready
            || registry->delegated_backends.sin_cos_f == NULL
            || !registry->delegated_backends.sin_cos_f(
                registry->delegated_backends.context,
                angle,
                sine,
                cosine)) {
        if (registry != NULL) {
            registry->last_status =
                REK_G1_MUJOCO_FEATURE_REGISTRY_BACKEND_FAILED;
        }
        return 0;
    }
    return 1;
}

static int adapter_loop_match(
        void* context,
        const SonicMotionComposerNativeLayer* target,
        const SonicMotionComposerNativeLayer* outgoing,
        float* matched_cursor) {
    RekG1MujocoFeatureRegistry* registry = context;
    if (registry == NULL || !registry->ready) {
        if (registry != NULL) {
            registry->last_status =
                REK_G1_MUJOCO_FEATURE_REGISTRY_NOT_READY;
        }
        return 0;
    }
    if (!sonic_motion_entry_matcher_native_callback(
            &registry->matcher, target, outgoing, matched_cursor)) {
        registry->last_status = REK_G1_MUJOCO_FEATURE_REGISTRY_MATCHER_FAILED;
        return 0;
    }
    registry->last_status = REK_G1_MUJOCO_FEATURE_REGISTRY_OK;
    return 1;
}

RekG1MujocoFeatureRegistryStatus
rek_g1_mujoco_feature_registry_bind_backends(
        RekG1MujocoFeatureRegistry* registry,
        SonicMotionComposerNativeBackends* backends,
        char* error,
        size_t error_capacity) {
    if (registry == NULL || backends == NULL) {
        set_error(error, error_capacity, "bind feature matcher", "null argument");
        return REK_G1_MUJOCO_FEATURE_REGISTRY_NULL_ARGUMENT;
    }
    if (!registry->ready) {
        set_error(error, error_capacity, "bind feature matcher", "registry is not ready");
        registry->last_status = REK_G1_MUJOCO_FEATURE_REGISTRY_NOT_READY;
        return registry->last_status;
    }
    if (registry->bound) {
        set_error(error, error_capacity, "bind feature matcher", "registry is already bound");
        registry->last_status = REK_G1_MUJOCO_FEATURE_REGISTRY_INVALID_DUEL;
        return registry->last_status;
    }
    if (backends->quaternion_slerp == NULL || backends->atan2_f == NULL
            || backends->sin_cos_f == NULL) {
        set_error(error, error_capacity, "bind feature matcher", "math backend is incomplete");
        registry->last_status = REK_G1_MUJOCO_FEATURE_REGISTRY_INVALID_DUEL;
        return registry->last_status;
    }
    if (backends->loop_entry_matcher != NULL) {
        set_error(error, error_capacity, "bind feature matcher", "loop matcher is already installed");
        registry->last_status = REK_G1_MUJOCO_FEATURE_REGISTRY_INVALID_DUEL;
        return registry->last_status;
    }
    registry->delegated_backends = *backends;
    SonicMotionComposerNativeBackends result = {
        .quaternion_slerp = adapter_slerp,
        .atan2_f = adapter_atan2,
        .sin_cos_f = adapter_sin_cos,
        .loop_entry_matcher = adapter_loop_match,
        .context = registry,
    };
    *backends = result;
    registry->bound = 1u;
    registry->last_status = REK_G1_MUJOCO_FEATURE_REGISTRY_OK;
    return registry->last_status;
}

RekG1MujocoFeatureRegistryStatus rek_g1_mujoco_feature_registry_open(
        RekG1MujocoFeatureRegistry* registry,
        GearSonicNativeDuelVector* duel,
        const RekG1SemanticAssets* assets,
        char* error,
        size_t error_capacity) {
    RekG1MujocoFeatureRegistryStatus status;
    if (registry == NULL || duel == NULL || assets == NULL) {
        set_error(error, error_capacity, "open feature registry", "null argument");
        return REK_G1_MUJOCO_FEATURE_REGISTRY_NULL_ARGUMENT;
    }
    memset(registry, 0, sizeof(*registry));
    status = validate_duel_and_map(
        duel,
        &registry->root_body_id,
        &registry->left_ankle_roll_body_id,
        &registry->right_ankle_roll_body_id,
        error,
        error_capacity);
    if (status != REK_G1_MUJOCO_FEATURE_REGISTRY_OK) goto fail;
    status = validate_assets_and_build_views(
        registry, assets, error, error_capacity);
    if (status != REK_G1_MUJOCO_FEATURE_REGISTRY_OK) goto fail;
    if (registry->total_feature_count == 0u
            || registry->total_feature_count > SIZE_MAX / sizeof(float)) {
        set_error(error, error_capacity, "open feature registry", "feature allocation size overflow");
        status = REK_G1_MUJOCO_FEATURE_REGISTRY_SIZE_OVERFLOW;
        goto fail;
    }
    registry->root_local_foot_xyz = calloc(
        registry->total_feature_count, sizeof(float));
    registry->scratch = mj_makeData(duel->model);
    if (registry->root_local_foot_xyz == NULL || registry->scratch == NULL) {
        set_error(error, error_capacity, "open feature registry", "scratch allocation failed");
        status = REK_G1_MUJOCO_FEATURE_REGISTRY_ALLOCATION_FAILED;
        goto fail;
    }
    registry->duel = duel;
    registry->assets = assets;
    registry->initialized = 1u;
    SonicMotionEntryMatcherNativeStatus matcher_status =
        sonic_motion_entry_matcher_native_init(
            &registry->matcher,
            (int32_t)REK_G1_SEMANTIC_DUEL_CONTROLLER_RATE_HZ,
            registry->slots,
            REK_G1_MUJOCO_FEATURE_CLIP_COUNT);
    if (matcher_status != SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK) {
        set_error(
            error,
            error_capacity,
            "open feature registry",
            sonic_motion_entry_matcher_native_status_string(matcher_status));
        status = REK_G1_MUJOCO_FEATURE_REGISTRY_MATCHER_FAILED;
        goto fail;
    }
    size_t feature_offset = 0u;
    for (size_t clip_index = 0;
            clip_index < REK_G1_MUJOCO_FEATURE_CLIP_COUNT;
            clip_index++) {
        const SonicMotionComposerNativeClip* clip =
            &registry->clip_views[clip_index];
        size_t clip_feature_count = 0u;
        if (!checked_product(
                clip->frame_count,
                SONIC_MOTION_ENTRY_MATCHER_NATIVE_FEATURE_WIDTH,
                &clip_feature_count)
                || feature_offset > registry->total_feature_count
                || clip_feature_count
                    > registry->total_feature_count - feature_offset) {
            set_error(error, error_capacity, "open feature registry", "feature offset overflow");
            status = REK_G1_MUJOCO_FEATURE_REGISTRY_SIZE_OVERFLOW;
            goto fail;
        }
        registry->feature_offsets[clip_index] = feature_offset;
        matcher_status = sonic_motion_entry_matcher_native_bake_features(
            clip,
            rek_g1_mujoco_feature_registry_sample,
            registry,
            registry->root_local_foot_xyz + feature_offset,
            clip_feature_count);
        if (matcher_status != SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK) {
            const char* detail =
                registry->last_status != REK_G1_MUJOCO_FEATURE_REGISTRY_OK
                ? rek_g1_mujoco_feature_registry_status_string(
                    registry->last_status)
                : sonic_motion_entry_matcher_native_status_string(
                    matcher_status);
            set_error(error, error_capacity, "bake feature clip", detail);
            status = registry->last_status
                    != REK_G1_MUJOCO_FEATURE_REGISTRY_OK
                ? registry->last_status
                : REK_G1_MUJOCO_FEATURE_REGISTRY_MATCHER_FAILED;
            goto fail;
        }
        matcher_status = sonic_motion_entry_matcher_native_register(
            &registry->matcher,
            clip,
            registry->root_local_foot_xyz + feature_offset,
            clip_feature_count);
        if (matcher_status != SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK) {
            set_error(
                error,
                error_capacity,
                "register feature clip",
                sonic_motion_entry_matcher_native_status_string(
                    matcher_status));
            status = REK_G1_MUJOCO_FEATURE_REGISTRY_MATCHER_FAILED;
            goto fail;
        }
        feature_offset += clip_feature_count;
    }
    if (feature_offset != registry->total_feature_count
            || registry->matcher.slot_count
                != REK_G1_MUJOCO_FEATURE_CLIP_COUNT) {
        set_error(error, error_capacity, "open feature registry", "incomplete feature registry");
        status = REK_G1_MUJOCO_FEATURE_REGISTRY_MATCHER_FAILED;
        goto fail;
    }
    registry->ready = 1u;
    registry->last_status = REK_G1_MUJOCO_FEATURE_REGISTRY_OK;
    return registry->last_status;

fail:
    rek_g1_mujoco_feature_registry_close(registry);
    registry->last_status = status;
    return status;
}

void rek_g1_mujoco_feature_registry_close(
        RekG1MujocoFeatureRegistry* registry) {
    if (registry == NULL) return;
    if (registry->scratch != NULL) mj_deleteData(registry->scratch);
    free(registry->root_local_foot_xyz);
    memset(registry, 0, sizeof(*registry));
}
