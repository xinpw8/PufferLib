#include "g1_fall_mujoco.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int assertions;

static void require(int condition, const char* name) {
    assertions++;
    if (!condition) {
        fprintf(stderr, "FAIL %s\n", name);
        exit(1);
    }
}

static int close_double(double actual, double expected, double tolerance) {
    return isfinite(actual) && isfinite(expected)
        && fabs(actual - expected) <= tolerance;
}

static int root_qpos_address(const mjModel* model, int body_id) {
    if (model == NULL || body_id <= 0 || body_id >= model->nbody
            || model->body_jntnum[body_id] < 1) {
        return -1;
    }
    const int joint_id = model->body_jntadr[body_id];
    if (joint_id < 0 || joint_id >= model->njnt
            || model->jnt_type[joint_id] != mjJNT_FREE) {
        return -1;
    }
    return model->jnt_qposadr[joint_id];
}

static int root_qvel_address(const mjModel* model, int body_id) {
    if (model == NULL || body_id <= 0 || body_id >= model->nbody
            || model->body_jntnum[body_id] < 1) {
        return -1;
    }
    const int joint_id = model->body_jntadr[body_id];
    if (joint_id < 0 || joint_id >= model->njnt
            || model->jnt_type[joint_id] != mjJNT_FREE) {
        return -1;
    }
    return model->jnt_dofadr[joint_id];
}

static GearSonicNativeDuelVector make_duel(
        mjModel* model, mjData** data) {
    GearSonicNativeDuelVector duel = {
        .model = model,
        .data = data,
        .arena_count = 1u,
        .robot_count = GEAR_SONIC_DUEL_FIGHTERS,
        .spawn_prefixes_verified = 1,
    };
    const char* const roots[GEAR_SONIC_DUEL_FIGHTERS] = {
        "player__pelvis_3266",
        "opponent__pelvis_3266",
    };
    for (size_t fighter = 0u;
            fighter < GEAR_SONIC_DUEL_FIGHTERS;
            fighter++) {
        const int body_id = mj_name2id(model, mjOBJ_BODY, roots[fighter]);
        duel.fighters[fighter].root_body_id = body_id;
        duel.fighters[fighter].root_qpos_address =
            root_qpos_address(model, body_id);
        duel.fighters[fighter].root_qvel_address =
            root_qvel_address(model, body_id);
    }
    return duel;
}

static void quaternion_multiply(
        const double left[4],
        const double right[4],
        double output[4]) {
    output[0] = left[0] * right[0] - left[1] * right[1]
        - left[2] * right[2] - left[3] * right[3];
    output[1] = left[0] * right[1] + left[1] * right[0]
        + left[2] * right[3] - left[3] * right[2];
    output[2] = left[0] * right[2] - left[1] * right[3]
        + left[2] * right[0] + left[3] * right[1];
    output[3] = left[0] * right[3] + left[1] * right[2]
        - left[2] * right[1] + left[3] * right[0];
}

typedef struct ContactFacts {
    uint8_t left;
    uint8_t right;
    uint32_t nonfoot;
} ContactFacts;

static ContactFacts manual_contact_facts(
        const RekG1FallMujocoAdapter* adapter,
        const mjData* data,
        size_t fighter) {
    ContactFacts result = {0};
    uint8_t* seen = calloc(adapter->body_count, sizeof(*seen));
    require(seen != NULL, "manual_seen_allocated");
    const mjModel* model = adapter->duel->model;
    for (int index = 0; index < data->ncon; index++) {
        const int geom0 = data->contact[index].geom[0];
        const int geom1 = data->contact[index].geom[1];
        int other = -1;
        if (geom0 == adapter->floor_geom_id
                && geom1 != adapter->floor_geom_id) {
            other = geom1;
        } else if (geom1 == adapter->floor_geom_id
                && geom0 != adapter->floor_geom_id) {
            other = geom0;
        }
        if (other < 0) continue;
        const int body = model->geom_bodyid[other];
        if (body <= 0 || (size_t)body >= adapter->body_count
                || adapter->body_owner[body] != (int8_t)fighter) {
            continue;
        }
        if (body == adapter->left_foot_body_ids[fighter]) {
            result.left = 1u;
        } else if (body == adapter->right_foot_body_ids[fighter]) {
            result.right = 1u;
        } else if (!seen[body]) {
            seen[body] = 1u;
            result.nonfoot++;
        }
    }
    free(seen);
    return result;
}

static void restore_root(
        const mjModel* model,
        mjData* data,
        int qpos_address,
        const double root_qpos[7]) {
    memcpy(data->qpos + qpos_address, root_qpos, 7u * sizeof(double));
    mj_forward(model, data);
}

static int first_geom_for_body(const mjModel* model, int body_id) {
    if (model == NULL || body_id < 0 || body_id >= model->nbody) return -1;
    for (int geom_id = 0; geom_id < model->ngeom; geom_id++) {
        if (model->geom_bodyid[geom_id] == body_id) return geom_id;
    }
    return -1;
}

static void add_test_contact(
        const mjModel* model,
        mjData* data,
        int floor_geom_id,
        int other_geom_id) {
    mjContact contact = {0};
    contact.geom[0] = floor_geom_id;
    contact.geom[1] = other_geom_id;
    require(mj_addContact(model, data, &contact) == 0,
        "synthetic_contact_added");
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s TWO_FIGHTER_MODEL_XML\n", argv[0]);
        return 64;
    }
    char error[1024] = {0};
    mjModel* model = mj_loadXML(argv[1], NULL, error, sizeof(error));
    if (model == NULL) {
        fprintf(stderr, "model load failed: %s\n", error);
        return 1;
    }
    mjData* arena = mj_makeData(model);
    require(arena != NULL, "arena_data_allocated");
    mjData* arenas[1] = {arena};
    mj_resetData(model, arena);
    mj_forward(model, arena);
    GearSonicNativeDuelVector duel = make_duel(model, arenas);
    require(duel.fighters[0].root_body_id >= 0, "player_root_resolved");
    require(duel.fighters[1].root_body_id >= 0, "opponent_root_resolved");

    RekG1FallMujocoAdapter invalid = {0};
    const int saved_opponent_root = duel.fighters[1].root_body_id;
    duel.fighters[1].root_body_id = duel.fighters[0].root_body_id;
    require(rek_g1_fall_mujoco_open(
        &invalid, &duel, error, sizeof(error))
        == REK_G1_FALL_MUJOCO_MAPPING_MISMATCH,
        "wrong_duel_root_rejected");
    rek_g1_fall_mujoco_close(&invalid);
    duel.fighters[1].root_body_id = saved_opponent_root;

    const int floor_id = mj_name2id(
        model, mjOBJ_GEOM, "arena_Collider_Floor_Rektagon");
    require(floor_id >= 0, "floor_resolved");
    const int saved_floor_type = model->geom_type[floor_id];
    model->geom_type[floor_id] = mjGEOM_SPHERE;
    require(rek_g1_fall_mujoco_open(
        &invalid, &duel, error, sizeof(error))
        == REK_G1_FALL_MUJOCO_MAPPING_MISMATCH,
        "non_box_floor_rejected");
    rek_g1_fall_mujoco_close(&invalid);
    model->geom_type[floor_id] = saved_floor_type;

    RekG1FallMujocoAdapter adapter = {0};
    require(rek_g1_fall_mujoco_open(
        &adapter, &duel, error, sizeof(error)) == REK_G1_FALL_MUJOCO_OK,
        "adapter_open");
    require(adapter.ready == 1u, "adapter_ready");
    require(adapter.floor_geom_id == floor_id, "exact_floor_cached");
    require(adapter.body_count == (size_t)model->nbody, "body_count_cached");

    RekG1FallMujocoMeasurement untouched;
    memset(&untouched, 0x5a, sizeof(untouched));
    RekG1FallMujocoMeasurement sentinel = untouched;
    require(rek_g1_fall_mujoco_sample(
        &adapter, 0u, 0.02f, 0u, &untouched, error, sizeof(error))
        == REK_G1_FALL_MUJOCO_NOT_CALIBRATED,
        "sample_before_calibration_rejected");
    require(memcmp(&untouched, &sentinel, sizeof(untouched)) == 0,
        "failed_sample_does_not_modify_output");

    require(rek_g1_fall_mujoco_calibrate_reset(
        &adapter, error, sizeof(error)) == REK_G1_FALL_MUJOCO_OK,
        "reset_calibration");
    for (size_t row = 0u; row < GEAR_SONIC_DUEL_FIGHTERS; row++) {
        RekG1FallMujocoMeasurement measured = {0};
        require(rek_g1_fall_mujoco_sample(
            &adapter,
            row,
            0.02f,
            (uint8_t)row,
            &measured,
            error,
            sizeof(error)) == REK_G1_FALL_MUJOCO_OK,
            "reset_sample");
        require(measured.fall_sample.tracking_active == 1u,
            "tracking_active");
        require(close_double(measured.fall_sample.tilt_degrees, 0.0, 1e-4),
            "reset_tilt_zero");
        require(close_double(
            measured.fall_sample.pelvis_height_ratio, 1.0, 1e-6),
            "reset_height_ratio_one");
        require(measured.fall_sample.can_get_up == (uint8_t)row,
            "can_get_up_is_caller_fact");
        require(measured.fall_sample.fixed_delta_seconds == 0.02f,
            "fixed_delta_passthrough");
        require(measured.floor_height > 0.009f
                && measured.floor_height < 0.011f,
            "floor_top_measured");
        require(measured.standing_pelvis_height > 0.79f
                && measured.standing_pelvis_height < 0.80f,
            "standing_pelvis_height_measured");
    }

    const int player_left_geom = first_geom_for_body(
        model, adapter.left_foot_body_ids[GEAR_SONIC_DUEL_PLAYER]);
    const int player_right_geom = first_geom_for_body(
        model, adapter.right_foot_body_ids[GEAR_SONIC_DUEL_PLAYER]);
    const int opponent_right_geom = first_geom_for_body(
        model, adapter.right_foot_body_ids[GEAR_SONIC_DUEL_OPPONENT]);
    int player_nonfoot_geoms[2] = {-1, -1};
    size_t player_nonfoot_geom_count = 0u;
    for (int body_id = 1;
            body_id < model->nbody && player_nonfoot_geom_count < 2u;
            body_id++) {
        if (adapter.body_owner[body_id]
                    != (int8_t)GEAR_SONIC_DUEL_PLAYER
                || body_id
                    == adapter.left_foot_body_ids[GEAR_SONIC_DUEL_PLAYER]
                || body_id
                    == adapter.right_foot_body_ids[GEAR_SONIC_DUEL_PLAYER]) {
            continue;
        }
        const int geom_id = first_geom_for_body(model, body_id);
        if (geom_id >= 0) {
            player_nonfoot_geoms[player_nonfoot_geom_count++] = geom_id;
        }
    }
    require(player_left_geom >= 0 && player_right_geom >= 0
            && opponent_right_geom >= 0
            && player_nonfoot_geom_count == 2u,
        "synthetic_contact_geoms_resolved");
    arena->ncon = 0;
    add_test_contact(model, arena, floor_id, player_left_geom);
    add_test_contact(model, arena, floor_id, player_left_geom);
    add_test_contact(model, arena, floor_id, player_right_geom);
    add_test_contact(model, arena, floor_id, player_nonfoot_geoms[0]);
    add_test_contact(model, arena, floor_id, player_nonfoot_geoms[0]);
    add_test_contact(model, arena, floor_id, player_nonfoot_geoms[1]);
    add_test_contact(model, arena, floor_id, opponent_right_geom);
    RekG1FallMujocoMeasurement synthetic_player = {0};
    require(rek_g1_fall_mujoco_sample(
        &adapter,
        0u,
        0.02f,
        0u,
        &synthetic_player,
        error,
        sizeof(error)) == REK_G1_FALL_MUJOCO_OK,
        "synthetic_player_contact_sample");
    require(synthetic_player.left_foot_body_contact == 1u,
        "synthetic_left_foot_detected");
    require(synthetic_player.right_foot_body_contact == 1u,
        "synthetic_right_foot_detected");
    require(synthetic_player.fall_sample.distinct_nonfoot_body_contact_count
            == 2u,
        "synthetic_nonfoot_bodies_deduplicated");
    RekG1FallMujocoMeasurement synthetic_opponent = {0};
    require(rek_g1_fall_mujoco_sample(
        &adapter,
        1u,
        0.02f,
        0u,
        &synthetic_opponent,
        error,
        sizeof(error)) == REK_G1_FALL_MUJOCO_OK,
        "synthetic_opponent_contact_sample");
    require(synthetic_opponent.left_foot_body_contact == 0u,
        "player_contacts_ignored_for_opponent_left");
    require(synthetic_opponent.right_foot_body_contact == 1u,
        "opponent_right_foot_detected");
    require(synthetic_opponent.fall_sample.distinct_nonfoot_body_contact_count
            == 0u,
        "player_nonfoot_contacts_ignored_for_opponent");
    mj_forward(model, arena);

    const int player_qpos = duel.fighters[0].root_qpos_address;
    double player_reset[7];
    memcpy(player_reset, arena->qpos + player_qpos, sizeof(player_reset));
    const RekG1FallMujocoCalibration player_calibration =
        adapter.calibrations[0];
    const double half_sqrt = sqrt(0.5);
    const double world_roll_90[4] = {half_sqrt, half_sqrt, 0.0, 0.0};
    double rotated[4];
    quaternion_multiply(world_roll_90, player_reset + 3, rotated);
    arena->qpos[player_qpos + 2] = player_calibration.reset_floor_height
        + 0.5 * player_calibration.standing_pelvis_height;
    memcpy(arena->qpos + player_qpos + 3, rotated, sizeof(rotated));
    mj_forward(model, arena);
    RekG1FallMujocoMeasurement tilted = {0};
    require(rek_g1_fall_mujoco_sample(
        &adapter, 0u, 0.02f, 0u, &tilted, error, sizeof(error))
        == REK_G1_FALL_MUJOCO_OK,
        "tilted_sample");
    require(close_double(tilted.fall_sample.tilt_degrees, 90.0, 1e-3),
        "reset_calibrated_world_tilt");
    require(close_double(
        tilted.fall_sample.pelvis_height_ratio, 0.5, 1e-5),
        "measured_height_ratio_half");

    restore_root(model, arena, player_qpos, player_reset);
    int saw_foot_contact = 0;
    for (int millimetres = 2; millimetres <= 30; millimetres += 2) {
        arena->qpos[player_qpos + 2] = player_reset[2]
            - (double)millimetres / 1000.0;
        mj_forward(model, arena);
        ContactFacts expected = manual_contact_facts(&adapter, arena, 0u);
        if (expected.left || expected.right) {
            RekG1FallMujocoMeasurement contact = {0};
            require(rek_g1_fall_mujoco_sample(
                &adapter, 0u, 0.02f, 0u, &contact, error, sizeof(error))
                == REK_G1_FALL_MUJOCO_OK,
                "foot_contact_sample");
            require(contact.left_foot_body_contact == expected.left,
                "left_foot_contact_exact_body");
            require(contact.right_foot_body_contact == expected.right,
                "right_foot_contact_exact_body");
            require(contact.fall_sample.has_foot_body_contact == 1u,
                "aggregate_foot_contact");
            require(contact.fall_sample.both_feet_off_floor == 0u,
                "both_feet_off_false_with_contact");
            require(contact.fall_sample.distinct_nonfoot_body_contact_count
                    == expected.nonfoot,
                "nonfoot_contacts_deduplicated");
            saw_foot_contact = 1;
            break;
        }
    }
    require(saw_foot_contact, "physical_foot_contact_generated");

    restore_root(model, arena, player_qpos, player_reset);
    int saw_nonfoot_contact = 0;
    for (int centimetres = 10; centimetres <= 50; centimetres += 2) {
        arena->qpos[player_qpos + 2] = (double)centimetres / 100.0;
        memcpy(arena->qpos + player_qpos + 3, rotated, sizeof(rotated));
        mj_forward(model, arena);
        ContactFacts expected = manual_contact_facts(&adapter, arena, 0u);
        if (expected.nonfoot > 0u) {
            RekG1FallMujocoMeasurement contact = {0};
            require(rek_g1_fall_mujoco_sample(
                &adapter, 0u, 0.02f, 0u, &contact, error, sizeof(error))
                == REK_G1_FALL_MUJOCO_OK,
                "nonfoot_contact_sample");
            require(contact.fall_sample.distinct_nonfoot_body_contact_count
                    == expected.nonfoot,
                "distinct_nonfoot_body_count_matches_contacts");
            require(contact.left_foot_body_contact == expected.left,
                "nonfoot_case_left_matches");
            require(contact.right_foot_body_contact == expected.right,
                "nonfoot_case_right_matches");
            saw_nonfoot_contact = 1;
            break;
        }
    }
    require(saw_nonfoot_contact, "physical_nonfoot_contact_generated");

    restore_root(model, arena, player_qpos, player_reset);
    arena->qpos[player_qpos + 3] = 0.0;
    arena->qpos[player_qpos + 4] = 0.0;
    arena->qpos[player_qpos + 5] = 0.0;
    arena->qpos[player_qpos + 6] = 0.0;
    RekG1FallMujocoMeasurement invalid_quaternion = {0};
    require(rek_g1_fall_mujoco_sample(
        &adapter,
        0u,
        0.02f,
        0u,
        &invalid_quaternion,
        error,
        sizeof(error)) == REK_G1_FALL_MUJOCO_NON_FINITE,
        "zero_quaternion_fails_closed");

    restore_root(model, arena, player_qpos, player_reset);
    arena->qpos[player_qpos + 2] = player_calibration.reset_floor_height;
    mj_forward(model, arena);
    require(rek_g1_fall_mujoco_calibrate_reset(
        &adapter, error, sizeof(error))
        == REK_G1_FALL_MUJOCO_MEASUREMENT_INVALID,
        "zero_standing_height_rejected");
    require(adapter.calibrations[0].calibrated == 0u,
        "failed_calibration_clears_player");
    require(adapter.calibrations[1].calibrated == 0u,
        "failed_calibration_clears_opponent");

    restore_root(model, arena, player_qpos, player_reset);
    require(rek_g1_fall_mujoco_calibrate_reset(
        &adapter, error, sizeof(error)) == REK_G1_FALL_MUJOCO_OK,
        "recalibration_after_restore");
    require(rek_g1_fall_mujoco_sample(
        &adapter, 2u, 0.02f, 0u, &untouched, error, sizeof(error))
        == REK_G1_FALL_MUJOCO_MEASUREMENT_INVALID,
        "out_of_range_row_rejected");
    require(rek_g1_fall_mujoco_sample(
        &adapter, 0u, 0.0f, 0u, &untouched, error, sizeof(error))
        == REK_G1_FALL_MUJOCO_MEASUREMENT_INVALID,
        "zero_delta_rejected");
    require(rek_g1_fall_mujoco_sample(
        &adapter, 0u, 0.02f, 2u, &untouched, error, sizeof(error))
        == REK_G1_FALL_MUJOCO_MEASUREMENT_INVALID,
        "invalid_can_get_up_rejected");

    rek_g1_fall_mujoco_close(&adapter);
    require(adapter.ready == 0u && adapter.duel == NULL,
        "adapter_close_clears_state");
    mj_deleteData(arena);
    mj_deleteModel(model);
    printf("G1 MuJoCo fall adapter passed: assertions=%d\n", assertions);
    return 0;
}
