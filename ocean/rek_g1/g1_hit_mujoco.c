#include "g1_hit_mujoco.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REK_G1_HIT_MUJOCO_PHYSICS_DT 0.002

typedef struct RekG1PinnedBodyDescriptor {
    const char* suffix;
    RekG1BodyZone zone;
    RekG1BodyPartType striker_part;
    int8_t striker_side;
    int8_t striker_slot;
    uint8_t expected_geom_count;
} RekG1PinnedBodyDescriptor;

static const char* const FIGHTER_PREFIXES[GEAR_SONIC_DUEL_FIGHTERS] = {
    "player__",
    "opponent__",
};

static const RekG1PinnedBodyDescriptor PINNED_BODIES
        [REK_G1_HIT_MUJOCO_FIGHTER_BODY_COUNT] = {
    {"pelvis_3266", REK_G1_BODY_ZONE_PELVIS,
        REK_G1_BODY_PART_NONE, -1, -1, 1u},
    {"left_hip_pitch_link_3457", REK_G1_BODY_ZONE_LEFT_HIP,
        REK_G1_BODY_PART_NONE, -1, -1, 1u},
    {"left_hip_roll_link_3425", REK_G1_BODY_ZONE_LEFT_HIP,
        REK_G1_BODY_PART_NONE, -1, -1, 1u},
    {"left_hip_yaw_link_2943", REK_G1_BODY_ZONE_LEFT_HIP,
        REK_G1_BODY_PART_NONE, -1, -1, 1u},
    {"left_knee_link_3106", REK_G1_BODY_ZONE_LEFT_KNEE,
        REK_G1_BODY_PART_SHIN, 0, REK_G1_HIT_MUJOCO_LEFT_SHIN_SLOT, 1u},
    {"left_ankle_pitch_link_3033", REK_G1_BODY_ZONE_LEFT_ANKLE,
        REK_G1_BODY_PART_NONE, -1, -1, 1u},
    {"left_ankle_roll_link_3045", REK_G1_BODY_ZONE_LEFT_ANKLE,
        REK_G1_BODY_PART_FOOT, 0, REK_G1_HIT_MUJOCO_LEFT_FOOT_SLOT, 4u},
    {"right_hip_pitch_link_3469", REK_G1_BODY_ZONE_RIGHT_HIP,
        REK_G1_BODY_PART_NONE, -1, -1, 1u},
    {"right_hip_roll_link_3345", REK_G1_BODY_ZONE_RIGHT_HIP,
        REK_G1_BODY_PART_NONE, -1, -1, 1u},
    {"right_hip_yaw_link_3191", REK_G1_BODY_ZONE_RIGHT_HIP,
        REK_G1_BODY_PART_NONE, -1, -1, 1u},
    {"right_knee_link_3429", REK_G1_BODY_ZONE_RIGHT_KNEE,
        REK_G1_BODY_PART_SHIN, 1, REK_G1_HIT_MUJOCO_RIGHT_SHIN_SLOT, 1u},
    {"right_ankle_pitch_link_3173", REK_G1_BODY_ZONE_RIGHT_ANKLE,
        REK_G1_BODY_PART_NONE, -1, -1, 1u},
    {"right_ankle_roll_link_3090", REK_G1_BODY_ZONE_RIGHT_ANKLE,
        REK_G1_BODY_PART_FOOT, 1, REK_G1_HIT_MUJOCO_RIGHT_FOOT_SLOT, 4u},
    {"waist_yaw_link_3359", REK_G1_BODY_ZONE_UNKNOWN,
        REK_G1_BODY_PART_NONE, -1, -1, 1u},
    {"waist_roll_link_2894", REK_G1_BODY_ZONE_UNKNOWN,
        REK_G1_BODY_PART_NONE, -1, -1, 1u},
    {"torso_link_3347", REK_G1_BODY_ZONE_TORSO,
        REK_G1_BODY_PART_NONE, -1, -1, 2u},
    {"left_shoulder_pitch_link_3161", REK_G1_BODY_ZONE_LEFT_SHOULDER,
        REK_G1_BODY_PART_NONE, -1, -1, 1u},
    {"left_shoulder_roll_link_3360", REK_G1_BODY_ZONE_LEFT_SHOULDER,
        REK_G1_BODY_PART_NONE, -1, -1, 1u},
    {"left_shoulder_yaw_link_3168", REK_G1_BODY_ZONE_LEFT_SHOULDER,
        REK_G1_BODY_PART_NONE, -1, -1, 1u},
    {"left_elbow_link_3284", REK_G1_BODY_ZONE_LEFT_ELBOW,
        REK_G1_BODY_PART_NONE, -1, -1, 1u},
    {"left_wrist_roll_link_3032", REK_G1_BODY_ZONE_LEFT_WRIST,
        REK_G1_BODY_PART_NONE, -1, -1, 1u},
    {"left_wrist_pitch_link_2914", REK_G1_BODY_ZONE_LEFT_WRIST,
        REK_G1_BODY_PART_NONE, -1, -1, 1u},
    {"left_wrist_yaw_link_3467", REK_G1_BODY_ZONE_LEFT_WRIST,
        REK_G1_BODY_PART_HAND, 0, REK_G1_HIT_MUJOCO_LEFT_HAND_SLOT, 1u},
    {"right_shoulder_pitch_link_3391", REK_G1_BODY_ZONE_RIGHT_SHOULDER,
        REK_G1_BODY_PART_NONE, -1, -1, 1u},
    {"right_shoulder_roll_link_3426", REK_G1_BODY_ZONE_RIGHT_SHOULDER,
        REK_G1_BODY_PART_NONE, -1, -1, 1u},
    {"right_shoulder_yaw_link_3107", REK_G1_BODY_ZONE_RIGHT_SHOULDER,
        REK_G1_BODY_PART_NONE, -1, -1, 1u},
    {"right_elbow_link_3322", REK_G1_BODY_ZONE_RIGHT_ELBOW,
        REK_G1_BODY_PART_NONE, -1, -1, 1u},
    {"right_wrist_roll_link_3331", REK_G1_BODY_ZONE_RIGHT_WRIST,
        REK_G1_BODY_PART_NONE, -1, -1, 1u},
    {"right_wrist_pitch_link_3282", REK_G1_BODY_ZONE_RIGHT_WRIST,
        REK_G1_BODY_PART_NONE, -1, -1, 1u},
    {"right_wrist_yaw_link_3293", REK_G1_BODY_ZONE_RIGHT_WRIST,
        REK_G1_BODY_PART_HAND, 1, REK_G1_HIT_MUJOCO_RIGHT_HAND_SLOT, 1u},
};

static void set_error(
        char* error,
        size_t error_capacity,
        const char* operation,
        const char* detail) {
    if (error == NULL || error_capacity == 0u) return;
    if (operation == NULL) operation = "measure G1 hit contacts";
    if (detail == NULL) detail = "unknown failure";
    (void)snprintf(error, error_capacity, "%s: %s", operation, detail);
    error[error_capacity - 1u] = '\0';
}

static RekG1HitMujocoStatus fail(
        RekG1HitMujocoAdapter* adapter,
        RekG1HitMujocoStatus status,
        char* error,
        size_t error_capacity,
        const char* operation,
        const char* detail) {
    if (adapter != NULL) adapter->last_status = status;
    set_error(error, error_capacity, operation, detail);
    return status;
}

static int checked_product(
        size_t left, size_t right, size_t* product) {
    if (product == NULL || left == 0u || right == 0u
            || left > SIZE_MAX / right) {
        return 0;
    }
    *product = left * right;
    return 1;
}

static int finite_mjt(const mjtNum* values, size_t count) {
    if (values == NULL) return 0;
    for (size_t index = 0u; index < count; index++) {
        if (!isfinite((double)values[index])) return 0;
    }
    return 1;
}

static int finite_f32(const float values[3]) {
    return values != NULL && isfinite(values[0])
        && isfinite(values[1]) && isfinite(values[2]);
}

static int binary_flag(uint8_t value) {
    return value == 0u || value == 1u;
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

static int exact_body_name(
        const mjModel* model, int object_id, const char* expected) {
    if (model == NULL || object_id < 0 || expected == NULL) return 0;
    const char* actual = mj_id2name(model, (int)mjOBJ_BODY, object_id);
    return actual != NULL && strcmp(actual, expected) == 0;
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
        && map->root_qpos_address + 7 <= model->nq
        && map->root_qvel_address >= 0
        && map->root_qvel_address + 6 <= model->nv;
}

static int duel_shape_valid(const GearSonicNativeDuelVector* duel) {
    if (duel == NULL || duel->model == NULL || duel->data == NULL
            || duel->failed || !duel->spawn_prefixes_verified
            || duel->arena_count == 0u
            || duel->arena_count > SIZE_MAX / GEAR_SONIC_DUEL_FIGHTERS
            || duel->robot_count
                != duel->arena_count * GEAR_SONIC_DUEL_FIGHTERS
            || duel->model->nbody
                != (int)REK_G1_HIT_MUJOCO_MODEL_BODY_COUNT
            || duel->model->ngeom
                != (int)REK_G1_HIT_MUJOCO_MODEL_GEOM_COUNT
            || duel->model->nq != GEAR_SONIC_DUEL_QPOS_DIM
            || duel->model->nv != GEAR_SONIC_DUEL_QVEL_DIM
            || duel->model->nu != GEAR_SONIC_DUEL_CONTROL_DIM
            || !isfinite((double)duel->model->opt.timestep)
            || fabs((double)duel->model->opt.timestep
                - REK_G1_HIT_MUJOCO_PHYSICS_DT) > 1e-15) {
        return 0;
    }
    for (size_t arena = 0u; arena < duel->arena_count; arena++) {
        if (duel->data[arena] == NULL) return 0;
    }
    return 1;
}

static int count_body_geoms(const mjModel* model, int body_id) {
    if (model == NULL || body_id <= 0 || body_id >= model->nbody) return -1;
    int count = 0;
    for (int geom_id = 0; geom_id < model->ngeom; geom_id++) {
        if (model->geom_bodyid[geom_id] == body_id) count++;
    }
    return count;
}

static RekG1HitMujocoStatus map_exact_bodies(
        RekG1HitMujocoAdapter* adapter,
        char* error,
        size_t error_capacity) {
    const mjModel* model = adapter->duel->model;
    uint8_t* descriptor_seen = calloc(
        adapter->body_count, sizeof(*descriptor_seen));
    if (descriptor_seen == NULL) {
        return fail(
            adapter, REK_G1_HIT_MUJOCO_ALLOCATION_FAILED,
            error, error_capacity, "map hit bodies", "allocation failed");
    }
    memset(adapter->body_owner, -1,
        adapter->body_count * sizeof(*adapter->body_owner));
    memset(adapter->striker_side, -1,
        adapter->body_count * sizeof(*adapter->striker_side));
    memset(adapter->striker_slot, -1,
        adapter->body_count * sizeof(*adapter->striker_slot));

    int roots[GEAR_SONIC_DUEL_FIGHTERS];
    for (size_t fighter = 0u;
            fighter < GEAR_SONIC_DUEL_FIGHTERS;
            fighter++) {
        char root_name[96];
        const int written = snprintf(
            root_name, sizeof(root_name), "%s%s",
            FIGHTER_PREFIXES[fighter],
            PINNED_BODIES[REK_G1_HIT_MUJOCO_PELVIS_BODY].suffix);
        if (written <= 0 || (size_t)written >= sizeof(root_name)) {
            free(descriptor_seen);
            return fail(
                adapter, REK_G1_HIT_MUJOCO_MAPPING_MISMATCH,
                error, error_capacity, "map hit bodies",
                "root body name construction failed");
        }
        roots[fighter] = mj_name2id(model, mjOBJ_BODY, root_name);
        if (roots[fighter] < 0) {
            free(descriptor_seen);
            return fail(
                adapter, REK_G1_HIT_MUJOCO_MAPPING_MISSING,
                error, error_capacity, "map hit bodies",
                "pinned fighter root is missing");
        }
        if (!exact_body_name(model, roots[fighter], root_name)
                || !root_map_valid(
                    model, &adapter->duel->fighters[fighter], roots[fighter])) {
            free(descriptor_seen);
            return fail(
                adapter, REK_G1_HIT_MUJOCO_MAPPING_MISMATCH,
                error, error_capacity, "map hit bodies",
                "fighter root does not match the opened duel map");
        }
    }
    if (roots[0] == roots[1]) {
        free(descriptor_seen);
        return fail(
            adapter, REK_G1_HIT_MUJOCO_MAPPING_MISMATCH,
            error, error_capacity, "map hit bodies",
            "fighter roots overlap");
    }

    size_t owner_counts[GEAR_SONIC_DUEL_FIGHTERS] = {0u, 0u};
    for (int body_id = 1; body_id < model->nbody; body_id++) {
        const int player = body_descends_from(model, body_id, roots[0]);
        const int opponent = body_descends_from(model, body_id, roots[1]);
        if (player && opponent) {
            free(descriptor_seen);
            return fail(
                adapter, REK_G1_HIT_MUJOCO_MAPPING_MISMATCH,
                error, error_capacity, "map hit bodies",
                "fighter body trees overlap");
        }
        if (player || opponent) {
            const size_t owner = player ? 0u : 1u;
            adapter->body_owner[body_id] = (int8_t)owner;
            owner_counts[owner]++;
        }
    }
    if (owner_counts[0] != REK_G1_HIT_MUJOCO_FIGHTER_BODY_COUNT
            || owner_counts[1] != REK_G1_HIT_MUJOCO_FIGHTER_BODY_COUNT) {
        free(descriptor_seen);
        return fail(
            adapter, REK_G1_HIT_MUJOCO_MAPPING_MISMATCH,
            error, error_capacity, "map hit bodies",
            "fighter body-tree size differs from the pinned G1 model");
    }

    uint8_t slot_seen[GEAR_SONIC_DUEL_FIGHTERS]
        [REK_G1_HIT_STRIKER_BODY_SLOTS] = {{0u}};
    for (size_t fighter = 0u;
            fighter < GEAR_SONIC_DUEL_FIGHTERS;
            fighter++) {
        for (size_t index = 0u;
                index < REK_G1_HIT_MUJOCO_FIGHTER_BODY_COUNT;
                index++) {
            const RekG1PinnedBodyDescriptor* descriptor =
                &PINNED_BODIES[index];
            char name[128];
            const int written = snprintf(
                name, sizeof(name), "%s%s",
                FIGHTER_PREFIXES[fighter], descriptor->suffix);
            if (written <= 0 || (size_t)written >= sizeof(name)) {
                free(descriptor_seen);
                return fail(
                    adapter, REK_G1_HIT_MUJOCO_MAPPING_MISMATCH,
                    error, error_capacity, "map hit bodies",
                    "body name construction failed");
            }
            const int body_id = mj_name2id(model, mjOBJ_BODY, name);
            if (body_id < 0) {
                free(descriptor_seen);
                return fail(
                    adapter, REK_G1_HIT_MUJOCO_MAPPING_MISSING,
                    error, error_capacity, "map hit bodies",
                    "pinned G1 body name is missing");
            }
            if ((size_t)body_id >= adapter->body_count
                    || descriptor_seen[body_id]
                    || adapter->body_owner[body_id] != (int8_t)fighter
                    || !exact_body_name(model, body_id, name)
                    || count_body_geoms(model, body_id)
                        != (int)descriptor->expected_geom_count) {
                free(descriptor_seen);
                return fail(
                    adapter, REK_G1_HIT_MUJOCO_MAPPING_MISMATCH,
                    error, error_capacity, "map hit bodies",
                    "G1 body identity, ownership, or geom count differs");
            }
            descriptor_seen[body_id] = 1u;
            adapter->body_ids[fighter][index] = body_id;
            adapter->body_zone[body_id] = descriptor->zone;
            adapter->striker_part[body_id] = descriptor->striker_part;
            adapter->striker_side[body_id] = descriptor->striker_side;
            adapter->striker_slot[body_id] = descriptor->striker_slot;
            if (descriptor->striker_slot >= 0) {
                const size_t slot = (size_t)descriptor->striker_slot;
                if (slot >= REK_G1_HIT_STRIKER_BODY_SLOTS
                        || slot_seen[fighter][slot]) {
                    free(descriptor_seen);
                    return fail(
                        adapter, REK_G1_HIT_MUJOCO_MAPPING_MISMATCH,
                        error, error_capacity, "map hit bodies",
                        "striker slot mapping is not one-to-one");
                }
                slot_seen[fighter][slot] = 1u;
            } else if (descriptor->striker_part != REK_G1_BODY_PART_NONE
                    || descriptor->striker_side != -1) {
                free(descriptor_seen);
                return fail(
                    adapter, REK_G1_HIT_MUJOCO_MAPPING_MISMATCH,
                    error, error_capacity, "map hit bodies",
                    "untagged body has partial striker metadata");
            }
        }
    }
    for (int body_id = 1; body_id < model->nbody; body_id++) {
        if (adapter->body_owner[body_id] >= 0
                && !descriptor_seen[body_id]) {
            free(descriptor_seen);
            return fail(
                adapter, REK_G1_HIT_MUJOCO_MAPPING_MISMATCH,
                error, error_capacity, "map hit bodies",
                "fighter descendant lacks a pinned body descriptor");
        }
    }
    for (size_t fighter = 0u;
            fighter < GEAR_SONIC_DUEL_FIGHTERS;
            fighter++) {
        for (size_t slot = 0u;
                slot < REK_G1_HIT_STRIKER_BODY_SLOTS;
                slot++) {
            if (!slot_seen[fighter][slot]) {
                free(descriptor_seen);
                return fail(
                    adapter, REK_G1_HIT_MUJOCO_MAPPING_MISMATCH,
                    error, error_capacity, "map hit bodies",
                    "a pinned striker slot is missing");
            }
        }
    }
    free(descriptor_seen);
    return REK_G1_HIT_MUJOCO_OK;
}

const char* rek_g1_hit_mujoco_status_string(
        RekG1HitMujocoStatus status) {
    switch (status) {
        case REK_G1_HIT_MUJOCO_OK: return "ok";
        case REK_G1_HIT_MUJOCO_NULL_ARGUMENT: return "null argument";
        case REK_G1_HIT_MUJOCO_INVALID_DUEL: return "invalid duel";
        case REK_G1_HIT_MUJOCO_MAPPING_MISSING: return "mapping missing";
        case REK_G1_HIT_MUJOCO_MAPPING_MISMATCH: return "mapping mismatch";
        case REK_G1_HIT_MUJOCO_ALLOCATION_FAILED:
            return "allocation failed";
        case REK_G1_HIT_MUJOCO_OBSERVATION_INVALID:
            return "invalid observation";
        case REK_G1_HIT_MUJOCO_SUBSTEP_SEQUENCE_INVALID:
            return "invalid substep sequence";
        case REK_G1_HIT_MUJOCO_NON_FINITE:
            return "non-finite measurement";
        case REK_G1_HIT_MUJOCO_CAPACITY_INSUFFICIENT:
            return "candidate capacity insufficient";
        case REK_G1_HIT_MUJOCO_CALLER_FACTS_INVALID:
            return "caller facts invalid";
        default: return "unknown MuJoCo hit status";
    }
}

RekG1HitMujocoStatus rek_g1_hit_mujoco_open(
        RekG1HitMujocoAdapter* adapter,
        GearSonicNativeDuelVector* duel,
        char* error,
        size_t error_capacity) {
    if (adapter == NULL || duel == NULL) {
        set_error(error, error_capacity, "open hit adapter", "null argument");
        return REK_G1_HIT_MUJOCO_NULL_ARGUMENT;
    }
    memset(adapter, 0, sizeof(*adapter));
    if (!duel_shape_valid(duel)) {
        return fail(
            adapter, REK_G1_HIT_MUJOCO_INVALID_DUEL,
            error, error_capacity, "open hit adapter",
            "duel is not the exact build-pinned two-fighter model");
    }
    adapter->duel = duel;
    adapter->arena_count = duel->arena_count;
    adapter->body_count = (size_t)duel->model->nbody;
    adapter->geom_count = (size_t)duel->model->ngeom;

    size_t previous_count = 0u;
    size_t scratch_capacity = 0u;
    size_t candidate_bytes = 0u;
    size_t body_zone_bytes = 0u;
    size_t striker_part_bytes = 0u;
    if (!checked_product(
            adapter->geom_count, adapter->geom_count, &adapter->pair_span)
            || !checked_product(
                adapter->arena_count, adapter->pair_span, &previous_count)
            || !checked_product(
                adapter->pair_span, 2u, &scratch_capacity)
            || !checked_product(
                scratch_capacity,
                sizeof(*adapter->candidate_scratch),
                &candidate_bytes)
            || !checked_product(
                adapter->body_count,
                sizeof(*adapter->body_zone),
                &body_zone_bytes)
            || !checked_product(
                adapter->body_count,
                sizeof(*adapter->striker_part),
                &striker_part_bytes)) {
        return fail(
            adapter, REK_G1_HIT_MUJOCO_ALLOCATION_FAILED,
            error, error_capacity, "open hit adapter",
            "scratch allocation size overflow");
    }
    adapter->candidate_scratch_capacity = scratch_capacity;
    adapter->body_owner = malloc(
        adapter->body_count * sizeof(*adapter->body_owner));
    adapter->body_zone = calloc(1u, body_zone_bytes);
    adapter->striker_part = calloc(1u, striker_part_bytes);
    adapter->striker_side = malloc(
        adapter->body_count * sizeof(*adapter->striker_side));
    adapter->striker_slot = malloc(
        adapter->body_count * sizeof(*adapter->striker_slot));
    adapter->previous_pairs = calloc(
        previous_count, sizeof(*adapter->previous_pairs));
    adapter->current_pairs = calloc(
        adapter->pair_span, sizeof(*adapter->current_pairs));
    adapter->expected_substep = calloc(
        adapter->arena_count, sizeof(*adapter->expected_substep));
    adapter->candidate_scratch = malloc(candidate_bytes);
    if (adapter->body_owner == NULL || adapter->body_zone == NULL
            || adapter->striker_part == NULL
            || adapter->striker_side == NULL
            || adapter->striker_slot == NULL
            || adapter->previous_pairs == NULL
            || adapter->current_pairs == NULL
            || adapter->expected_substep == NULL
            || adapter->candidate_scratch == NULL) {
        rek_g1_hit_mujoco_close(adapter);
        set_error(error, error_capacity, "open hit adapter", "allocation failed");
        adapter->last_status = REK_G1_HIT_MUJOCO_ALLOCATION_FAILED;
        return adapter->last_status;
    }
    RekG1HitMujocoStatus status = map_exact_bodies(
        adapter, error, error_capacity);
    if (status != REK_G1_HIT_MUJOCO_OK) {
        rek_g1_hit_mujoco_close(adapter);
        adapter->last_status = status;
        return status;
    }
    adapter->initialized = 1u;
    adapter->ready = 1u;
    adapter->last_status = REK_G1_HIT_MUJOCO_OK;
    if (error != NULL && error_capacity > 0u) error[0] = '\0';
    return adapter->last_status;
}

static int adapter_ready(const RekG1HitMujocoAdapter* adapter) {
    return adapter != NULL && adapter->ready && adapter->initialized
        && adapter->duel != NULL && duel_shape_valid(adapter->duel)
        && adapter->arena_count == adapter->duel->arena_count
        && adapter->body_count == (size_t)adapter->duel->model->nbody
        && adapter->geom_count == (size_t)adapter->duel->model->ngeom
        && adapter->pair_span == adapter->geom_count * adapter->geom_count
        && adapter->body_owner != NULL && adapter->body_zone != NULL
        && adapter->striker_part != NULL && adapter->striker_side != NULL
        && adapter->striker_slot != NULL
        && adapter->previous_pairs != NULL
        && adapter->current_pairs != NULL
        && adapter->expected_substep != NULL
        && adapter->candidate_scratch != NULL;
}

RekG1HitMujocoStatus rek_g1_hit_mujoco_reset(
        RekG1HitMujocoAdapter* adapter,
        char* error,
        size_t error_capacity) {
    if (adapter == NULL) {
        set_error(error, error_capacity, "reset hit adapter", "null argument");
        return REK_G1_HIT_MUJOCO_NULL_ARGUMENT;
    }
    if (!adapter_ready(adapter)) {
        return fail(
            adapter, REK_G1_HIT_MUJOCO_INVALID_DUEL,
            error, error_capacity, "reset hit adapter",
            "adapter or duel is not ready");
    }
    memset(adapter->previous_pairs, 0,
        adapter->arena_count * adapter->pair_span
            * sizeof(*adapter->previous_pairs));
    memset(adapter->current_pairs, 0,
        adapter->pair_span * sizeof(*adapter->current_pairs));
    memset(adapter->expected_substep, 0,
        adapter->arena_count * sizeof(*adapter->expected_substep));
    adapter->last_status = REK_G1_HIT_MUJOCO_OK;
    if (error != NULL && error_capacity > 0u) error[0] = '\0';
    return adapter->last_status;
}

RekG1HitMujocoStatus rek_g1_hit_mujoco_clear_arena_contacts(
        RekG1HitMujocoAdapter* adapter,
        size_t arena_index,
        char* error,
        size_t error_capacity) {
    if (adapter == NULL) {
        set_error(
            error, error_capacity,
            "clear arena hit contacts", "null argument");
        return REK_G1_HIT_MUJOCO_NULL_ARGUMENT;
    }
    if (!adapter_ready(adapter) || arena_index >= adapter->arena_count) {
        return fail(
            adapter, REK_G1_HIT_MUJOCO_INVALID_DUEL,
            error, error_capacity, "clear arena hit contacts",
            "adapter is not ready or arena index is invalid");
    }
    memset(
        adapter->previous_pairs + arena_index * adapter->pair_span,
        0,
        adapter->pair_span * sizeof(*adapter->previous_pairs));
    adapter->last_status = REK_G1_HIT_MUJOCO_OK;
    if (error != NULL && error_capacity > 0u) error[0] = '\0';
    return adapter->last_status;
}

static size_t sorted_pair_index(
        size_t geom_count, int geom0, int geom1) {
    const size_t first = (size_t)(geom0 < geom1 ? geom0 : geom1);
    const size_t second = (size_t)(geom0 < geom1 ? geom1 : geom0);
    return first * geom_count + second;
}

static int valid_contact_geometry(const mjContact* contact) {
    return contact != NULL
        && isfinite((double)contact->dist)
        && finite_mjt(contact->pos, 3u)
        && finite_mjt(contact->frame, 9u);
}

static RekG1HitMujocoStatus append_directed_candidate(
        RekG1HitMujocoAdapter* adapter,
        const GearSonicNativeDuelPostStepObservation* observation,
        int striker_geom,
        int target_geom,
        size_t* candidate_count,
        char* error,
        size_t error_capacity) {
    const mjModel* model = observation->model;
    const mjData* data = observation->data;
    const int striker_body = model->geom_bodyid[striker_geom];
    const int target_body = model->geom_bodyid[target_geom];
    if (striker_body < 0 || (size_t)striker_body >= adapter->body_count
            || target_body < 0
            || (size_t)target_body >= adapter->body_count) {
        return fail(
            adapter, REK_G1_HIT_MUJOCO_MAPPING_MISMATCH,
            error, error_capacity, "scan hit contacts",
            "contact geom resolves to an invalid body");
    }
    const int8_t striker_owner = adapter->body_owner[striker_body];
    const int8_t target_owner = adapter->body_owner[target_body];
    const int8_t slot = adapter->striker_slot[striker_body];
    if (slot < 0 || striker_owner < 0 || target_owner < 0
            || striker_owner == target_owner) {
        return REK_G1_HIT_MUJOCO_OK;
    }
    if (striker_owner >= (int8_t)GEAR_SONIC_DUEL_FIGHTERS
            || target_owner >= (int8_t)GEAR_SONIC_DUEL_FIGHTERS
            || slot >= (int8_t)REK_G1_HIT_STRIKER_BODY_SLOTS
            || (adapter->striker_part[striker_body]
                    != REK_G1_BODY_PART_HAND
                && adapter->striker_part[striker_body]
                    != REK_G1_BODY_PART_FOOT
                && adapter->striker_part[striker_body]
                    != REK_G1_BODY_PART_SHIN)
            || (adapter->striker_side[striker_body] != 0
                && adapter->striker_side[striker_body] != 1)
            || (int)adapter->body_zone[target_body]
                < (int)REK_G1_BODY_ZONE_UNKNOWN
            || adapter->body_zone[target_body]
                > REK_G1_BODY_ZONE_RIGHT_ANKLE) {
        return fail(
            adapter, REK_G1_HIT_MUJOCO_MAPPING_MISMATCH,
            error, error_capacity, "scan hit contacts",
            "cached striker or target mapping is invalid");
    }
    if (*candidate_count >= adapter->candidate_scratch_capacity) {
        return fail(
            adapter, REK_G1_HIT_MUJOCO_ALLOCATION_FAILED,
            error, error_capacity, "scan hit contacts",
            "internal candidate capacity exhausted");
    }

    mjtNum striker_velocity[6];
    mjtNum target_velocity[6];
    mj_objectVelocity(
        model, data, mjOBJ_BODY, striker_body, striker_velocity, 0);
    mj_objectVelocity(
        model, data, mjOBJ_BODY, target_body, target_velocity, 0);
    const mjtNum* striker_position = data->xpos + 3 * striker_body;
    const mjtNum* target_position = data->xpos + 3 * target_body;
    if (!finite_mjt(striker_position, 3u)
            || !finite_mjt(target_position, 3u)
            || !finite_mjt(striker_velocity, 6u)
            || !finite_mjt(target_velocity, 6u)) {
        return fail(
            adapter, REK_G1_HIT_MUJOCO_NON_FINITE,
            error, error_capacity, "scan hit contacts",
            "body position or world velocity is non-finite");
    }

    RekG1HitMujocoCandidate candidate = {
        .arena_index = observation->arena_index,
        .physics_substep_index = observation->physics_substep_index,
        .striker_geom_id = striker_geom,
        .target_geom_id = target_geom,
        .striker_body_id = striker_body,
        .target_body_id = target_body,
        .striker_fighter = (uint32_t)striker_owner,
        .target_fighter = (uint32_t)target_owner,
        .striker_body_slot = (uint32_t)slot,
        .striker_part = adapter->striker_part[striker_body],
        .striker_side = (RekG1HandSide)adapter->striker_side[striker_body],
        .target_zone = adapter->body_zone[target_body],
    };
    float norm_squared = 0.0f;
    for (size_t axis = 0u; axis < 3u; axis++) {
        candidate.striker_body_position_world[axis] =
            (float)striker_position[axis];
        candidate.target_body_position_world[axis] =
            (float)target_position[axis];
        candidate.striker_body_linear_velocity_world[axis] =
            (float)striker_velocity[3u + axis];
        candidate.target_body_linear_velocity_world[axis] =
            (float)target_velocity[3u + axis];
        const float relative =
            candidate.striker_body_linear_velocity_world[axis]
            - candidate.target_body_linear_velocity_world[axis];
        norm_squared += relative * relative;
    }
    candidate.relative_speed_mps = sqrtf(norm_squared);
    if (!finite_f32(candidate.striker_body_position_world)
            || !finite_f32(candidate.target_body_position_world)
            || !finite_f32(candidate.striker_body_linear_velocity_world)
            || !finite_f32(candidate.target_body_linear_velocity_world)
            || !isfinite(norm_squared)
            || !isfinite(candidate.relative_speed_mps)) {
        return fail(
            adapter, REK_G1_HIT_MUJOCO_NON_FINITE,
            error, error_capacity, "scan hit contacts",
            "float contact measurement is non-finite");
    }
    adapter->candidate_scratch[*candidate_count] = candidate;
    *candidate_count += 1u;
    return REK_G1_HIT_MUJOCO_OK;
}

RekG1HitMujocoStatus rek_g1_hit_mujoco_scan_substep(
        RekG1HitMujocoAdapter* adapter,
        const GearSonicNativeDuelPostStepObservation* observation,
        RekG1HitMujocoCandidate* candidates,
        size_t candidate_capacity,
        size_t* candidate_count,
        char* error,
        size_t error_capacity) {
    if (adapter == NULL || observation == NULL || candidate_count == NULL) {
        set_error(error, error_capacity, "scan hit contacts", "null argument");
        return REK_G1_HIT_MUJOCO_NULL_ARGUMENT;
    }
    if (!adapter_ready(adapter)) {
        return fail(
            adapter, REK_G1_HIT_MUJOCO_INVALID_DUEL,
            error, error_capacity, "scan hit contacts",
            "adapter or duel is not ready");
    }
    if ((candidate_capacity > 0u && candidates == NULL)
            || observation->vector != adapter->duel
            || observation->model != adapter->duel->model
            || observation->arena_index >= adapter->arena_count
            || observation->data
                != adapter->duel->data[observation->arena_index]
            || observation->physics_substep_index
                >= REK_G1_HIT_MUJOCO_PHYSICS_SUBSTEPS
            || !isfinite(observation->physics_dt_seconds)
            || fabs(observation->physics_dt_seconds
                - REK_G1_HIT_MUJOCO_PHYSICS_DT) > 1e-15
            || !isfinite((double)observation->data->time)
            || observation->data->ncon < 0
            || (observation->data->ncon > 0
                && observation->data->contact == NULL)) {
        return fail(
            adapter, REK_G1_HIT_MUJOCO_OBSERVATION_INVALID,
            error, error_capacity, "scan hit contacts",
            "post-step observation does not match the opened duel");
    }
    if (adapter->expected_substep[observation->arena_index]
            != (uint8_t)observation->physics_substep_index) {
        return fail(
            adapter, REK_G1_HIT_MUJOCO_SUBSTEP_SEQUENCE_INVALID,
            error, error_capacity, "scan hit contacts",
            "caller skipped or repeated a physics substep");
    }

    memset(adapter->current_pairs, 0,
        adapter->pair_span * sizeof(*adapter->current_pairs));
    const uint8_t* previous = adapter->previous_pairs
        + observation->arena_index * adapter->pair_span;
    size_t measured_count = 0u;
    for (int contact_index = 0;
            contact_index < observation->data->ncon;
            contact_index++) {
        const mjContact* contact =
            &observation->data->contact[contact_index];
        const int geom0 = contact->geom[0];
        const int geom1 = contact->geom[1];
        if (geom0 < 0 || (size_t)geom0 >= adapter->geom_count
                || geom1 < 0 || (size_t)geom1 >= adapter->geom_count
                || geom0 == geom1) {
            return fail(
                adapter, REK_G1_HIT_MUJOCO_MAPPING_MISMATCH,
                error, error_capacity, "scan hit contacts",
                "contact contains an invalid geom pair");
        }
        if (!valid_contact_geometry(contact)) {
            return fail(
                adapter, REK_G1_HIT_MUJOCO_NON_FINITE,
                error, error_capacity, "scan hit contacts",
                "contact geometry is non-finite");
        }
        const size_t pair = sorted_pair_index(
            adapter->geom_count, geom0, geom1);
        if (pair >= adapter->pair_span) {
            return fail(
                adapter, REK_G1_HIT_MUJOCO_MAPPING_MISMATCH,
                error, error_capacity, "scan hit contacts",
                "sorted geom-pair identity overflowed");
        }
        if (adapter->current_pairs[pair]) continue;
        adapter->current_pairs[pair] = 1u;
        if (previous[pair]) continue;

        RekG1HitMujocoStatus status = append_directed_candidate(
            adapter,
            observation,
            geom0,
            geom1,
            &measured_count,
            error,
            error_capacity);
        if (status != REK_G1_HIT_MUJOCO_OK) return status;
        status = append_directed_candidate(
            adapter,
            observation,
            geom1,
            geom0,
            &measured_count,
            error,
            error_capacity);
        if (status != REK_G1_HIT_MUJOCO_OK) return status;
    }
    if (measured_count > candidate_capacity) {
        return fail(
            adapter, REK_G1_HIT_MUJOCO_CAPACITY_INSUFFICIENT,
            error, error_capacity, "scan hit contacts",
            "caller candidate buffer is too small");
    }
    if (measured_count > 0u) {
        memcpy(
            candidates,
            adapter->candidate_scratch,
            measured_count * sizeof(*candidates));
    }
    uint8_t* committed = adapter->previous_pairs
        + observation->arena_index * adapter->pair_span;
    memcpy(committed, adapter->current_pairs,
        adapter->pair_span * sizeof(*committed));
    adapter->expected_substep[observation->arena_index] =
        observation->physics_substep_index + 1u
            == REK_G1_HIT_MUJOCO_PHYSICS_SUBSTEPS
        ? 0u
        : (uint8_t)(observation->physics_substep_index + 1u);
    *candidate_count = measured_count;
    adapter->last_status = REK_G1_HIT_MUJOCO_OK;
    if (error != NULL && error_capacity > 0u) error[0] = '\0';
    return adapter->last_status;
}

static int candidate_valid(const RekG1HitMujocoCandidate* candidate) {
    if (candidate == NULL
            || candidate->striker_geom_id < 0
            || candidate->target_geom_id < 0
            || candidate->striker_geom_id == candidate->target_geom_id
            || candidate->striker_body_id <= 0
            || candidate->target_body_id <= 0
            || candidate->striker_fighter >= REK_G1_HIT_FIGHTERS
            || candidate->target_fighter >= REK_G1_HIT_FIGHTERS
            || candidate->striker_fighter == candidate->target_fighter
            || candidate->striker_body_slot
                >= REK_G1_HIT_STRIKER_BODY_SLOTS
            || (candidate->striker_part != REK_G1_BODY_PART_HAND
                && candidate->striker_part != REK_G1_BODY_PART_FOOT
                && candidate->striker_part != REK_G1_BODY_PART_SHIN)
            || ((int)candidate->striker_side != 0
                && (int)candidate->striker_side != 1)
            || (int)candidate->target_zone
                < (int)REK_G1_BODY_ZONE_UNKNOWN
            || candidate->target_zone > REK_G1_BODY_ZONE_RIGHT_ANKLE
            || !finite_f32(candidate->striker_body_position_world)
            || !finite_f32(candidate->target_body_position_world)
            || !finite_f32(candidate->striker_body_linear_velocity_world)
            || !finite_f32(candidate->target_body_linear_velocity_world)
            || !isfinite(candidate->relative_speed_mps)
            || candidate->relative_speed_mps < 0.0f) {
        return 0;
    }
    return 1;
}

RekG1HitMujocoStatus rek_g1_hit_mujoco_candidate_to_contact(
        const RekG1HitMujocoCandidate* candidate,
        const RekG1HitMujocoCallerFacts* facts,
        RekG1HitContact* contact,
        char* error,
        size_t error_capacity) {
    if (candidate == NULL || facts == NULL || contact == NULL) {
        set_error(
            error, error_capacity, "assemble hit contact", "null argument");
        return REK_G1_HIT_MUJOCO_NULL_ARGUMENT;
    }
    if (!candidate_valid(candidate)
            || !isfinite(facts->time_seconds)
            || facts->time_seconds < 0.0f
            || !binary_flag(facts->round_active)
            || !binary_flag(facts->fighter_upright[0])
            || !binary_flag(facts->fighter_upright[1])
            || !binary_flag(facts->fighter_standing[0])
            || !binary_flag(facts->fighter_standing[1])) {
        set_error(
            error, error_capacity, "assemble hit contact",
            "candidate or explicit caller facts are invalid");
        return REK_G1_HIT_MUJOCO_CALLER_FACTS_INVALID;
    }
    RekG1HitContact measured = {
        .strike_intent = facts->strike_intent,
        .relative_speed_mps = candidate->relative_speed_mps,
        .time_seconds = facts->time_seconds,
        .striker_part = candidate->striker_part,
        .striker_side = candidate->striker_side,
        .target_zone = candidate->target_zone,
        .striker_fighter = candidate->striker_fighter,
        .target_fighter = candidate->target_fighter,
        .striker_body_slot = candidate->striker_body_slot,
        .is_enter = 1u,
        .round_active = facts->round_active,
        .striker_upright =
            facts->fighter_upright[candidate->striker_fighter],
        .target_upright =
            facts->fighter_upright[candidate->target_fighter],
        .target_standing =
            facts->fighter_standing[candidate->target_fighter],
    };
    memcpy(
        measured.striker_body_position_world,
        candidate->striker_body_position_world,
        sizeof(measured.striker_body_position_world));
    memcpy(
        measured.target_body_position_world,
        candidate->target_body_position_world,
        sizeof(measured.target_body_position_world));
    memcpy(
        measured.striker_body_linear_velocity_world,
        candidate->striker_body_linear_velocity_world,
        sizeof(measured.striker_body_linear_velocity_world));
    memcpy(
        measured.target_body_linear_velocity_world,
        candidate->target_body_linear_velocity_world,
        sizeof(measured.target_body_linear_velocity_world));
    *contact = measured;
    if (error != NULL && error_capacity > 0u) error[0] = '\0';
    return REK_G1_HIT_MUJOCO_OK;
}

void rek_g1_hit_mujoco_close(RekG1HitMujocoAdapter* adapter) {
    if (adapter == NULL) return;
    free(adapter->candidate_scratch);
    free(adapter->expected_substep);
    free(adapter->current_pairs);
    free(adapter->previous_pairs);
    free(adapter->striker_slot);
    free(adapter->striker_side);
    free(adapter->striker_part);
    free(adapter->body_zone);
    free(adapter->body_owner);
    memset(adapter, 0, sizeof(*adapter));
}
