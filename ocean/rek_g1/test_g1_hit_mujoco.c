#include "g1_hit_mujoco.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_ARENAS 2u
#define TEST_CANDIDATE_CAPACITY 8u

static size_t assertion_count = 0u;

#define CHECK(condition) do { \
    assertion_count += 1u; \
    if (!(condition)) { \
        (void)fprintf( \
            stderr, "check failed at %s:%d: %s\n", \
            __FILE__, __LINE__, #condition); \
        goto cleanup; \
    } \
} while (0)

static const char* const BODY_SUFFIXES
        [REK_G1_HIT_MUJOCO_FIGHTER_BODY_COUNT] = {
    "pelvis_3266",
    "left_hip_pitch_link_3457",
    "left_hip_roll_link_3425",
    "left_hip_yaw_link_2943",
    "left_knee_link_3106",
    "left_ankle_pitch_link_3033",
    "left_ankle_roll_link_3045",
    "right_hip_pitch_link_3469",
    "right_hip_roll_link_3345",
    "right_hip_yaw_link_3191",
    "right_knee_link_3429",
    "right_ankle_pitch_link_3173",
    "right_ankle_roll_link_3090",
    "waist_yaw_link_3359",
    "waist_roll_link_2894",
    "torso_link_3347",
    "left_shoulder_pitch_link_3161",
    "left_shoulder_roll_link_3360",
    "left_shoulder_yaw_link_3168",
    "left_elbow_link_3284",
    "left_wrist_roll_link_3032",
    "left_wrist_pitch_link_2914",
    "left_wrist_yaw_link_3467",
    "right_shoulder_pitch_link_3391",
    "right_shoulder_roll_link_3426",
    "right_shoulder_yaw_link_3107",
    "right_elbow_link_3322",
    "right_wrist_roll_link_3331",
    "right_wrist_pitch_link_3282",
    "right_wrist_yaw_link_3293",
};

static const RekG1BodyZone EXPECTED_ZONES
        [REK_G1_HIT_MUJOCO_FIGHTER_BODY_COUNT] = {
    REK_G1_BODY_ZONE_PELVIS,
    REK_G1_BODY_ZONE_LEFT_HIP,
    REK_G1_BODY_ZONE_LEFT_HIP,
    REK_G1_BODY_ZONE_LEFT_HIP,
    REK_G1_BODY_ZONE_LEFT_KNEE,
    REK_G1_BODY_ZONE_LEFT_ANKLE,
    REK_G1_BODY_ZONE_LEFT_ANKLE,
    REK_G1_BODY_ZONE_RIGHT_HIP,
    REK_G1_BODY_ZONE_RIGHT_HIP,
    REK_G1_BODY_ZONE_RIGHT_HIP,
    REK_G1_BODY_ZONE_RIGHT_KNEE,
    REK_G1_BODY_ZONE_RIGHT_ANKLE,
    REK_G1_BODY_ZONE_RIGHT_ANKLE,
    REK_G1_BODY_ZONE_UNKNOWN,
    REK_G1_BODY_ZONE_UNKNOWN,
    REK_G1_BODY_ZONE_TORSO,
    REK_G1_BODY_ZONE_LEFT_SHOULDER,
    REK_G1_BODY_ZONE_LEFT_SHOULDER,
    REK_G1_BODY_ZONE_LEFT_SHOULDER,
    REK_G1_BODY_ZONE_LEFT_ELBOW,
    REK_G1_BODY_ZONE_LEFT_WRIST,
    REK_G1_BODY_ZONE_LEFT_WRIST,
    REK_G1_BODY_ZONE_LEFT_WRIST,
    REK_G1_BODY_ZONE_RIGHT_SHOULDER,
    REK_G1_BODY_ZONE_RIGHT_SHOULDER,
    REK_G1_BODY_ZONE_RIGHT_SHOULDER,
    REK_G1_BODY_ZONE_RIGHT_ELBOW,
    REK_G1_BODY_ZONE_RIGHT_WRIST,
    REK_G1_BODY_ZONE_RIGHT_WRIST,
    REK_G1_BODY_ZONE_RIGHT_WRIST,
};

static const RekG1BodyPartType EXPECTED_PARTS
        [REK_G1_HIT_MUJOCO_FIGHTER_BODY_COUNT] = {
    REK_G1_BODY_PART_NONE,
    REK_G1_BODY_PART_NONE,
    REK_G1_BODY_PART_NONE,
    REK_G1_BODY_PART_NONE,
    REK_G1_BODY_PART_SHIN,
    REK_G1_BODY_PART_NONE,
    REK_G1_BODY_PART_FOOT,
    REK_G1_BODY_PART_NONE,
    REK_G1_BODY_PART_NONE,
    REK_G1_BODY_PART_NONE,
    REK_G1_BODY_PART_SHIN,
    REK_G1_BODY_PART_NONE,
    REK_G1_BODY_PART_FOOT,
    REK_G1_BODY_PART_NONE,
    REK_G1_BODY_PART_NONE,
    REK_G1_BODY_PART_NONE,
    REK_G1_BODY_PART_NONE,
    REK_G1_BODY_PART_NONE,
    REK_G1_BODY_PART_NONE,
    REK_G1_BODY_PART_NONE,
    REK_G1_BODY_PART_NONE,
    REK_G1_BODY_PART_NONE,
    REK_G1_BODY_PART_HAND,
    REK_G1_BODY_PART_NONE,
    REK_G1_BODY_PART_NONE,
    REK_G1_BODY_PART_NONE,
    REK_G1_BODY_PART_NONE,
    REK_G1_BODY_PART_NONE,
    REK_G1_BODY_PART_NONE,
    REK_G1_BODY_PART_HAND,
};

static const int EXPECTED_SIDES
        [REK_G1_HIT_MUJOCO_FIGHTER_BODY_COUNT] = {
    -1, -1, -1, -1, 0, -1, 0, -1, -1, -1,
    1, -1, 1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, 0, -1, -1, -1, -1, -1, -1, 1,
};

static const int EXPECTED_SLOTS
        [REK_G1_HIT_MUJOCO_FIGHTER_BODY_COUNT] = {
    -1, -1, -1, -1, REK_G1_HIT_MUJOCO_LEFT_SHIN_SLOT,
    -1, REK_G1_HIT_MUJOCO_LEFT_FOOT_SLOT,
    -1, -1, -1, REK_G1_HIT_MUJOCO_RIGHT_SHIN_SLOT,
    -1, REK_G1_HIT_MUJOCO_RIGHT_FOOT_SLOT,
    -1, -1, -1, -1, -1, -1, -1,
    -1, -1, REK_G1_HIT_MUJOCO_LEFT_HAND_SLOT,
    -1, -1, -1, -1, -1, -1,
    REK_G1_HIT_MUJOCO_RIGHT_HAND_SLOT,
};

static const int EXPECTED_GEOM_COUNTS
        [REK_G1_HIT_MUJOCO_FIGHTER_BODY_COUNT] = {
    1, 1, 1, 1, 1, 1, 4, 1, 1, 1,
    1, 1, 4, 1, 1, 2, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
};

static int configure_root_map(
        const mjModel* model,
        const char* root_name,
        GearSonicDuelFighterMap* map) {
    if (model == NULL || root_name == NULL || map == NULL) return 0;
    const int body_id = mj_name2id(model, mjOBJ_BODY, root_name);
    if (body_id <= 0 || body_id >= model->nbody
            || model->body_jntnum[body_id] < 1) {
        return 0;
    }
    const int joint_id = model->body_jntadr[body_id];
    if (joint_id < 0 || joint_id >= model->njnt
            || model->jnt_type[joint_id] != mjJNT_FREE) {
        return 0;
    }
    map->root_body_id = body_id;
    map->root_qpos_address = model->jnt_qposadr[joint_id];
    map->root_qvel_address = model->jnt_dofadr[joint_id];
    return 1;
}

static int count_body_geoms(const mjModel* model, int body_id) {
    if (model == NULL || body_id <= 0 || body_id >= model->nbody) return -1;
    int count = 0;
    for (int geom = 0; geom < model->ngeom; geom++) {
        if (model->geom_bodyid[geom] == body_id) count++;
    }
    return count;
}

static int first_body_geom(const mjModel* model, int body_id) {
    if (model == NULL || body_id <= 0 || body_id >= model->nbody) return -1;
    for (int geom = 0; geom < model->ngeom; geom++) {
        if (model->geom_bodyid[geom] == body_id) return geom;
    }
    return -1;
}

static mjContact make_contact(int geom0, int geom1) {
    mjContact contact;
    memset(&contact, 0, sizeof(contact));
    contact.dist = -0.001;
    contact.pos[0] = 0.25;
    contact.pos[1] = -0.5;
    contact.pos[2] = 1.0;
    contact.frame[0] = 1.0;
    contact.frame[4] = 1.0;
    contact.frame[8] = 1.0;
    contact.geom[0] = geom0;
    contact.geom[1] = geom1;
    contact.dim = 3;
    contact.efc_address = -1;
    return contact;
}

static int replace_contacts(
        const mjModel* model,
        mjData* data,
        const mjContact* contacts,
        size_t count) {
    if (model == NULL || data == NULL
            || (count > 0u && contacts == NULL)) {
        return 0;
    }
    data->ncon = 0;
    for (size_t index = 0u; index < count; index++) {
        if (mj_addContact(model, data, &contacts[index]) != 0) return 0;
    }
    return data->ncon == (int)count;
}

static GearSonicNativeDuelPostStepObservation observation_for(
        const GearSonicNativeDuelVector* duel,
        size_t arena,
        uint32_t substep) {
    GearSonicNativeDuelPostStepObservation observation = {
        .vector = duel,
        .model = duel->model,
        .data = duel->data[arena],
        .arena_index = arena,
        .physics_substep_index = substep,
        .physics_dt_seconds = 0.002,
    };
    return observation;
}

static int close_f32(float actual, float expected, float tolerance) {
    return isfinite(actual) && isfinite(expected)
        && isfinite(tolerance) && tolerance >= 0.0f
        && fabsf(actual - expected) <= tolerance;
}

static int bytes_are(const void* value, size_t size, unsigned char byte) {
    if (value == NULL) return 0;
    const unsigned char* bytes = value;
    for (size_t index = 0u; index < size; index++) {
        if (bytes[index] != byte) return 0;
    }
    return 1;
}

int main(int argc, char** argv) {
    int exit_code = EXIT_FAILURE;
    mjModel* model = NULL;
    mjData* arena_data[TEST_ARENAS] = {NULL, NULL};
    GearSonicNativeDuelVector duel;
    RekG1HitMujocoAdapter adapter;
    memset(&duel, 0, sizeof(duel));
    memset(&adapter, 0, sizeof(adapter));

    CHECK(argc == 2);
    char xml_error[1024] = {0};
    model = mj_loadXML(argv[1], NULL, xml_error, (int)sizeof(xml_error));
    CHECK(model != NULL);
    /* gear_sonic_native_duel_open pins the loaded model to each 2 ms step. */
    model->opt.timestep = 0.002;
    CHECK(model->nbody == (int)REK_G1_HIT_MUJOCO_MODEL_BODY_COUNT);
    CHECK(model->ngeom == (int)REK_G1_HIT_MUJOCO_MODEL_GEOM_COUNT);
    CHECK(model->nq == GEAR_SONIC_DUEL_QPOS_DIM);
    CHECK(model->nv == GEAR_SONIC_DUEL_QVEL_DIM);
    CHECK(model->nu == GEAR_SONIC_DUEL_CONTROL_DIM);
    CHECK(fabs((double)model->opt.timestep - 0.002) <= 1e-15);

    for (size_t arena = 0u; arena < TEST_ARENAS; arena++) {
        arena_data[arena] = mj_makeData(model);
        CHECK(arena_data[arena] != NULL);
    }
    duel.model = model;
    duel.data = arena_data;
    duel.arena_count = TEST_ARENAS;
    duel.robot_count = TEST_ARENAS * GEAR_SONIC_DUEL_FIGHTERS;
    duel.spawn_prefixes_verified = 1;
    CHECK(configure_root_map(
        model, "player__pelvis_3266", &duel.fighters[0]));
    CHECK(configure_root_map(
        model, "opponent__pelvis_3266", &duel.fighters[1]));

    for (size_t arena = 0u; arena < TEST_ARENAS; arena++) {
        mj_resetData(model, arena_data[arena]);
        memset(
            arena_data[arena]->qvel,
            0,
            (size_t)model->nv * sizeof(*arena_data[arena]->qvel));
        const int player_qvel = duel.fighters[0].root_qvel_address;
        const int opponent_qvel = duel.fighters[1].root_qvel_address;
        arena_data[arena]->qvel[player_qvel + 0] = 1.0;
        arena_data[arena]->qvel[player_qvel + 1] = 2.0;
        arena_data[arena]->qvel[player_qvel + 2] = 3.0;
        arena_data[arena]->qvel[opponent_qvel + 0] = 4.0;
        arena_data[arena]->qvel[opponent_qvel + 1] = 6.0;
        arena_data[arena]->qvel[opponent_qvel + 2] = 3.0;
        mj_forward(model, arena_data[arena]);
    }

    char error[512] = {0};
    CHECK(rek_g1_hit_mujoco_open(
        &adapter, &duel, error, sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    CHECK(adapter.ready == 1u);
    CHECK(adapter.initialized == 1u);
    CHECK(adapter.arena_count == TEST_ARENAS);
    CHECK(adapter.body_count == (size_t)model->nbody);
    CHECK(adapter.geom_count == (size_t)model->ngeom);
    CHECK(adapter.geom_zone != NULL);
    CHECK(adapter.pair_span
        == (size_t)model->ngeom * (size_t)model->ngeom);

    const char* const prefixes[GEAR_SONIC_DUEL_FIGHTERS] = {
        "player__", "opponent__",
    };
    for (size_t fighter = 0u;
            fighter < GEAR_SONIC_DUEL_FIGHTERS;
            fighter++) {
        for (size_t body_index = 0u;
                body_index < REK_G1_HIT_MUJOCO_FIGHTER_BODY_COUNT;
                body_index++) {
            char name[128];
            const int written = snprintf(
                name, sizeof(name), "%s%s",
                prefixes[fighter], BODY_SUFFIXES[body_index]);
            CHECK(written > 0);
            CHECK((size_t)written < sizeof(name));
            const int body_id = mj_name2id(model, mjOBJ_BODY, name);
            CHECK(body_id > 0);
            CHECK(adapter.body_ids[fighter][body_index] == body_id);
            CHECK(adapter.body_owner[body_id] == (int8_t)fighter);
            CHECK(adapter.body_zone[body_id] == EXPECTED_ZONES[body_index]);
            CHECK(adapter.striker_part[body_id]
                == EXPECTED_PARTS[body_index]);
            CHECK((int)adapter.striker_side[body_id]
                == EXPECTED_SIDES[body_index]);
            CHECK((int)adapter.striker_slot[body_id]
                == EXPECTED_SLOTS[body_index]);
            CHECK(count_body_geoms(model, body_id)
                == EXPECTED_GEOM_COUNTS[body_index]);
        }
    }
    CHECK(adapter.striker_side[
        adapter.body_ids[0][REK_G1_HIT_MUJOCO_LEFT_WRIST_YAW_BODY]] == 0);
    CHECK(adapter.striker_side[
        adapter.body_ids[0][REK_G1_HIT_MUJOCO_RIGHT_WRIST_YAW_BODY]] == 1);

    GearSonicNativeDuelVector bad_duel = duel;
    bad_duel.fighters[0].root_body_id += 1;
    RekG1HitMujocoAdapter rejected_adapter;
    memset(&rejected_adapter, 0x5a, sizeof(rejected_adapter));
    CHECK(rek_g1_hit_mujoco_open(
        &rejected_adapter,
        &bad_duel,
        error,
        sizeof(error)) == REK_G1_HIT_MUJOCO_MAPPING_MISMATCH);
    CHECK(rejected_adapter.ready == 0u);
    CHECK(rejected_adapter.body_owner == NULL);
    CHECK(rejected_adapter.geom_zone == NULL);
    rek_g1_hit_mujoco_close(&rejected_adapter);

    const int player_left_wrist = first_body_geom(
        model,
        adapter.body_ids[0][REK_G1_HIT_MUJOCO_LEFT_WRIST_YAW_BODY]);
    const int player_left_knee = first_body_geom(
        model,
        adapter.body_ids[0][REK_G1_HIT_MUJOCO_LEFT_KNEE_BODY]);
    const int opponent_head = mj_name2id(
        model, mjOBJ_GEOM, "opponent__mjgeom_3064");
    const int opponent_torso = mj_name2id(
        model, mjOBJ_GEOM, "opponent__mjgeom_3285");
    const int player_pelvis = first_body_geom(
        model,
        adapter.body_ids[0][REK_G1_HIT_MUJOCO_PELVIS_BODY]);
    const int opponent_right_ankle = first_body_geom(
        model,
        adapter.body_ids[1][REK_G1_HIT_MUJOCO_RIGHT_ANKLE_ROLL_BODY]);
    const int opponent_right_knee = first_body_geom(
        model,
        adapter.body_ids[1][REK_G1_HIT_MUJOCO_RIGHT_KNEE_BODY]);
    const int player_left_ankle = first_body_geom(
        model,
        adapter.body_ids[0][REK_G1_HIT_MUJOCO_LEFT_ANKLE_ROLL_BODY]);
    CHECK(player_left_wrist >= 0);
    CHECK(player_left_knee >= 0);
    CHECK(opponent_head >= 0);
    CHECK(opponent_torso >= 0);
    CHECK(player_pelvis >= 0);
    CHECK(opponent_right_ankle >= 0);
    CHECK(opponent_right_knee >= 0);
    CHECK(player_left_ankle >= 0);
    CHECK(model->geom_bodyid[opponent_head]
        == adapter.body_ids[1][REK_G1_HIT_MUJOCO_TORSO_BODY]);
    CHECK(model->geom_bodyid[opponent_torso]
        == adapter.body_ids[1][REK_G1_HIT_MUJOCO_TORSO_BODY]);
    CHECK(adapter.geom_zone[opponent_head] == REK_G1_BODY_ZONE_HEAD);
    CHECK(adapter.geom_zone[opponent_torso] == REK_G1_BODY_ZONE_TORSO);

    {
        const mjContact head_zone_contacts[4] = {
            make_contact(player_left_wrist, opponent_head),
            make_contact(player_left_ankle, opponent_head),
            make_contact(player_left_knee, opponent_head),
            make_contact(player_left_wrist, opponent_torso),
        };
        CHECK(replace_contacts(model, arena_data[1], head_zone_contacts, 4u));
        GearSonicNativeDuelPostStepObservation head_zone_observation =
            observation_for(&duel, 1u, 0u);
        RekG1HitMujocoCandidate head_zone_candidates[4];
        size_t head_zone_count = 0u;
        CHECK(rek_g1_hit_mujoco_scan_substep(
            &adapter,
            &head_zone_observation,
            head_zone_candidates,
            4u,
            &head_zone_count,
            error,
            sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
        CHECK(head_zone_count == 4u);
        const RekG1BodyPartType expected_head_parts[3] = {
            REK_G1_BODY_PART_HAND,
            REK_G1_BODY_PART_FOOT,
            REK_G1_BODY_PART_SHIN,
        };
        for (size_t index = 0u; index < 3u; index++) {
            CHECK(head_zone_candidates[index].striker_part
                == expected_head_parts[index]);
            CHECK(head_zone_candidates[index].target_geom_id == opponent_head);
            CHECK(head_zone_candidates[index].target_zone
                == REK_G1_BODY_ZONE_HEAD);
        }
        CHECK(head_zone_candidates[3].striker_part
            == REK_G1_BODY_PART_HAND);
        CHECK(head_zone_candidates[3].target_geom_id == opponent_torso);
        CHECK(head_zone_candidates[3].target_zone
            == REK_G1_BODY_ZONE_TORSO);
        CHECK(rek_g1_hit_mujoco_reset(
            &adapter, error, sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    }

    mjContact four_contacts[4] = {
        make_contact(player_left_wrist, opponent_torso),
        make_contact(player_pelvis, opponent_right_ankle),
        make_contact(opponent_right_knee, player_left_ankle),
        make_contact(opponent_torso, player_left_wrist),
    };
    CHECK(replace_contacts(model, arena_data[0], four_contacts, 4u));
    GearSonicNativeDuelPostStepObservation observation =
        observation_for(&duel, 0u, 0u);
    RekG1HitMujocoCandidate candidates[TEST_CANDIDATE_CAPACITY];
    memset(candidates, 0xa5, sizeof(candidates));
    size_t candidate_count = 777u;
    CHECK(rek_g1_hit_mujoco_scan_substep(
        &adapter,
        &observation,
        candidates,
        3u,
        &candidate_count,
        error,
        sizeof(error)) == REK_G1_HIT_MUJOCO_CAPACITY_INSUFFICIENT);
    CHECK(candidate_count == 777u);
    CHECK(bytes_are(candidates, sizeof(candidates), 0xa5u));
    CHECK(adapter.expected_substep[0] == 0u);

    CHECK(rek_g1_hit_mujoco_scan_substep(
        &adapter,
        &observation,
        candidates,
        TEST_CANDIDATE_CAPACITY,
        &candidate_count,
        error,
        sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    CHECK(candidate_count == 4u);

    const int expected_striker_geoms[4] = {
        player_left_wrist,
        opponent_right_ankle,
        opponent_right_knee,
        player_left_ankle,
    };
    const int expected_target_geoms[4] = {
        opponent_torso,
        player_pelvis,
        player_left_ankle,
        opponent_right_knee,
    };
    const uint32_t expected_striker_fighters[4] = {0u, 1u, 1u, 0u};
    const uint32_t expected_target_fighters[4] = {1u, 0u, 0u, 1u};
    const RekG1BodyPartType expected_candidate_parts[4] = {
        REK_G1_BODY_PART_HAND,
        REK_G1_BODY_PART_FOOT,
        REK_G1_BODY_PART_SHIN,
        REK_G1_BODY_PART_FOOT,
    };
    const int expected_candidate_sides[4] = {0, 1, 1, 0};
    const uint32_t expected_candidate_slots[4] = {
        REK_G1_HIT_MUJOCO_LEFT_HAND_SLOT,
        REK_G1_HIT_MUJOCO_RIGHT_FOOT_SLOT,
        REK_G1_HIT_MUJOCO_RIGHT_SHIN_SLOT,
        REK_G1_HIT_MUJOCO_LEFT_FOOT_SLOT,
    };
    const RekG1BodyZone expected_target_zones[4] = {
        REK_G1_BODY_ZONE_TORSO,
        REK_G1_BODY_ZONE_PELVIS,
        REK_G1_BODY_ZONE_LEFT_ANKLE,
        REK_G1_BODY_ZONE_RIGHT_KNEE,
    };
    for (size_t index = 0u; index < 4u; index++) {
        CHECK(candidates[index].arena_index == 0u);
        CHECK(candidates[index].physics_substep_index == 0u);
        CHECK(candidates[index].striker_geom_id
            == expected_striker_geoms[index]);
        CHECK(candidates[index].target_geom_id
            == expected_target_geoms[index]);
        CHECK(candidates[index].striker_fighter
            == expected_striker_fighters[index]);
        CHECK(candidates[index].target_fighter
            == expected_target_fighters[index]);
        CHECK(candidates[index].striker_part
            == expected_candidate_parts[index]);
        CHECK((int)candidates[index].striker_side
            == expected_candidate_sides[index]);
        CHECK(candidates[index].striker_body_slot
            == expected_candidate_slots[index]);
        CHECK(candidates[index].target_zone
            == expected_target_zones[index]);
        CHECK(close_f32(candidates[index].relative_speed_mps, 5.0f, 1e-5f));
        const float* striker_velocity =
            candidates[index].striker_body_linear_velocity_world;
        const float* target_velocity =
            candidates[index].target_body_linear_velocity_world;
        const uint32_t striker = expected_striker_fighters[index];
        CHECK(close_f32(
            striker_velocity[0], striker == 0u ? 1.0f : 4.0f, 1e-5f));
        CHECK(close_f32(
            striker_velocity[1], striker == 0u ? 2.0f : 6.0f, 1e-5f));
        CHECK(close_f32(striker_velocity[2], 3.0f, 1e-5f));
        CHECK(close_f32(
            target_velocity[0], striker == 0u ? 4.0f : 1.0f, 1e-5f));
        CHECK(close_f32(
            target_velocity[1], striker == 0u ? 6.0f : 2.0f, 1e-5f));
        CHECK(close_f32(target_velocity[2], 3.0f, 1e-5f));
    }

    mjContact reversed_contacts[3] = {
        make_contact(opponent_torso, player_left_wrist),
        make_contact(opponent_right_ankle, player_pelvis),
        make_contact(player_left_ankle, opponent_right_knee),
    };
    CHECK(replace_contacts(model, arena_data[0], reversed_contacts, 3u));
    observation = observation_for(&duel, 0u, 1u);
    candidate_count = 991u;
    CHECK(rek_g1_hit_mujoco_scan_substep(
        &adapter,
        &observation,
        candidates,
        TEST_CANDIDATE_CAPACITY,
        &candidate_count,
        error,
        sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    CHECK(candidate_count == 0u);

    CHECK(replace_contacts(model, arena_data[1], four_contacts, 1u));
    observation = observation_for(&duel, 1u, 0u);
    CHECK(rek_g1_hit_mujoco_scan_substep(
        &adapter,
        &observation,
        candidates,
        TEST_CANDIDATE_CAPACITY,
        &candidate_count,
        error,
        sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    CHECK(candidate_count == 1u);
    CHECK(candidates[0].arena_index == 1u);
    CHECK(candidates[0].striker_geom_id == player_left_wrist);

    CHECK(replace_contacts(model, arena_data[0], NULL, 0u));
    observation = observation_for(&duel, 0u, 2u);
    CHECK(rek_g1_hit_mujoco_scan_substep(
        &adapter,
        &observation,
        candidates,
        TEST_CANDIDATE_CAPACITY,
        &candidate_count,
        error,
        sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    CHECK(candidate_count == 0u);

    CHECK(replace_contacts(model, arena_data[0], reversed_contacts, 1u));
    observation = observation_for(&duel, 0u, 3u);
    CHECK(rek_g1_hit_mujoco_scan_substep(
        &adapter,
        &observation,
        candidates,
        TEST_CANDIDATE_CAPACITY,
        &candidate_count,
        error,
        sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    CHECK(candidate_count == 1u);
    CHECK(candidates[0].striker_geom_id == player_left_wrist);
    CHECK(candidates[0].target_geom_id == opponent_torso);

    CHECK(replace_contacts(model, arena_data[0], NULL, 0u));
    observation = observation_for(&duel, 0u, 5u);
    candidate_count = 992u;
    CHECK(rek_g1_hit_mujoco_scan_substep(
        &adapter,
        &observation,
        candidates,
        TEST_CANDIDATE_CAPACITY,
        &candidate_count,
        error,
        sizeof(error)) == REK_G1_HIT_MUJOCO_SUBSTEP_SEQUENCE_INVALID);
    CHECK(candidate_count == 992u);
    CHECK(adapter.expected_substep[0] == 4u);
    observation = observation_for(&duel, 0u, 4u);
    CHECK(rek_g1_hit_mujoco_scan_substep(
        &adapter,
        &observation,
        candidates,
        TEST_CANDIDATE_CAPACITY,
        &candidate_count,
        error,
        sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    CHECK(candidate_count == 0u);

    mjContact nonfinite_contact =
        make_contact(player_left_wrist, opponent_torso);
    nonfinite_contact.pos[0] = NAN;
    CHECK(replace_contacts(model, arena_data[0], &nonfinite_contact, 1u));
    observation = observation_for(&duel, 0u, 5u);
    memset(candidates, 0x3c, sizeof(candidates));
    candidate_count = 993u;
    CHECK(rek_g1_hit_mujoco_scan_substep(
        &adapter,
        &observation,
        candidates,
        TEST_CANDIDATE_CAPACITY,
        &candidate_count,
        error,
        sizeof(error)) == REK_G1_HIT_MUJOCO_NON_FINITE);
    CHECK(candidate_count == 993u);
    CHECK(bytes_are(candidates, sizeof(candidates), 0x3cu));
    CHECK(adapter.expected_substep[0] == 5u);
    nonfinite_contact.pos[0] = 0.25;
    CHECK(replace_contacts(model, arena_data[0], &nonfinite_contact, 1u));
    CHECK(rek_g1_hit_mujoco_scan_substep(
        &adapter,
        &observation,
        candidates,
        TEST_CANDIDATE_CAPACITY,
        &candidate_count,
        error,
        sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    CHECK(candidate_count == 1u);

    CHECK(replace_contacts(model, arena_data[0], NULL, 0u));
    observation = observation_for(&duel, 0u, 6u);
    CHECK(rek_g1_hit_mujoco_scan_substep(
        &adapter,
        &observation,
        candidates,
        TEST_CANDIDATE_CAPACITY,
        &candidate_count,
        error,
        sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    CHECK(candidate_count == 0u);

    nonfinite_contact = make_contact(player_left_wrist, opponent_torso);
    CHECK(replace_contacts(model, arena_data[0], &nonfinite_contact, 1u));
    observation = observation_for(&duel, 0u, 7u);
    const int player_left_wrist_body =
        adapter.body_ids[0][REK_G1_HIT_MUJOCO_LEFT_WRIST_YAW_BODY];
    const size_t xpos_index = (size_t)(3 * player_left_wrist_body);
    const mjtNum saved_xpos = arena_data[0]->xpos[xpos_index];
    arena_data[0]->xpos[xpos_index] = NAN;
    candidate_count = 994u;
    CHECK(rek_g1_hit_mujoco_scan_substep(
        &adapter,
        &observation,
        candidates,
        TEST_CANDIDATE_CAPACITY,
        &candidate_count,
        error,
        sizeof(error)) == REK_G1_HIT_MUJOCO_NON_FINITE);
    CHECK(candidate_count == 994u);
    CHECK(adapter.expected_substep[0] == 7u);
    arena_data[0]->xpos[xpos_index] = saved_xpos;
    CHECK(rek_g1_hit_mujoco_scan_substep(
        &adapter,
        &observation,
        candidates,
        TEST_CANDIDATE_CAPACITY,
        &candidate_count,
        error,
        sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    CHECK(candidate_count == 1u);

    CHECK(replace_contacts(model, arena_data[0], NULL, 0u));
    observation = observation_for(&duel, 0u, 8u);
    CHECK(rek_g1_hit_mujoco_scan_substep(
        &adapter, &observation, candidates, TEST_CANDIDATE_CAPACITY,
        &candidate_count, error, sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    CHECK(candidate_count == 0u);
    observation = observation_for(&duel, 0u, 9u);
    CHECK(rek_g1_hit_mujoco_scan_substep(
        &adapter, &observation, candidates, TEST_CANDIDATE_CAPACITY,
        &candidate_count, error, sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    CHECK(candidate_count == 0u);
    CHECK(adapter.expected_substep[0] == 0u);

    nonfinite_contact = make_contact(player_left_wrist, opponent_torso);
    CHECK(replace_contacts(model, arena_data[0], &nonfinite_contact, 1u));
    observation = observation_for(&duel, 0u, 0u);
    CHECK(rek_g1_hit_mujoco_scan_substep(
        &adapter, &observation, candidates, TEST_CANDIDATE_CAPACITY,
        &candidate_count, error, sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    CHECK(candidate_count == 1u);
    CHECK(adapter.expected_substep[0] == 1u);
    CHECK(rek_g1_hit_mujoco_clear_arena_contacts(
        &adapter, 0u, error, sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    CHECK(adapter.expected_substep[0] == 1u);
    size_t retained_pairs = 0u;
    for (size_t pair = 0u; pair < adapter.pair_span; pair++) {
        retained_pairs += adapter.previous_pairs[pair];
    }
    CHECK(retained_pairs == 0u);
    observation = observation_for(&duel, 0u, 1u);
    CHECK(rek_g1_hit_mujoco_scan_substep(
        &adapter, &observation, candidates, TEST_CANDIDATE_CAPACITY,
        &candidate_count, error, sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    CHECK(candidate_count == 1u);
    CHECK(adapter.expected_substep[0] == 2u);
    CHECK(rek_g1_hit_mujoco_clear_arena_contacts(
        &adapter, TEST_ARENAS, error, sizeof(error))
        == REK_G1_HIT_MUJOCO_INVALID_DUEL);
    CHECK(rek_g1_hit_mujoco_reset(
        &adapter, error, sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    CHECK(adapter.expected_substep[0] == 0u);
    CHECK(adapter.expected_substep[1] == 0u);
    observation = observation_for(&duel, 0u, 0u);
    CHECK(rek_g1_hit_mujoco_scan_substep(
        &adapter, &observation, candidates, TEST_CANDIDATE_CAPACITY,
        &candidate_count, error, sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    CHECK(candidate_count == 1u);

    RekG1ImpactEvent events[1] = {{
        .impact_time_seconds = 0.3f,
        .lead_time_seconds = 0.1f,
        .release_time_seconds = 0.2f,
        .limb = REK_G1_AIM_LIMB_LEFT_UPPER_BODY,
    }};
    RekG1HitMujocoCallerFacts facts = {
        .strike_intent = {
            .impact_events = events,
            .impact_event_count = 1u,
            .clip_cursor_frames = 12.5f,
            .clip_fps = 50.0f,
            .move_id = 73,
            .action_playing = 1u,
            .layer_active = 1u,
            .layer_loop = 0u,
        },
        .time_seconds = 17.25f,
        .round_active = 1u,
        .fighter_upright = {1u, 0u},
        .fighter_standing = {1u, 0u},
    };
    RekG1HitContact measured_contact;
    memset(&measured_contact, 0x77, sizeof(measured_contact));
    CHECK(rek_g1_hit_mujoco_candidate_to_contact(
        &candidates[0],
        &facts,
        &measured_contact,
        error,
        sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    CHECK(measured_contact.strike_intent.impact_events == events);
    CHECK(measured_contact.strike_intent.impact_event_count == 1u);
    CHECK(measured_contact.strike_intent.move_id == 73);
    CHECK(measured_contact.time_seconds == 17.25f);
    CHECK(measured_contact.is_enter == 1u);
    CHECK(measured_contact.round_active == 1u);
    CHECK(measured_contact.striker_upright == 1u);
    CHECK(measured_contact.target_upright == 0u);
    CHECK(measured_contact.target_standing == 0u);
    CHECK((int)measured_contact.striker_side == 0);
    CHECK(measured_contact.striker_part == REK_G1_BODY_PART_HAND);
    CHECK(measured_contact.target_zone == REK_G1_BODY_ZONE_TORSO);
    CHECK(close_f32(measured_contact.relative_speed_mps, 5.0f, 1e-5f));

    RekG1HitContact unchanged_contact;
    memset(&unchanged_contact, 0x66, sizeof(unchanged_contact));
    unsigned char unchanged_bytes[sizeof(unchanged_contact)];
    memcpy(unchanged_bytes, &unchanged_contact, sizeof(unchanged_bytes));
    facts.round_active = 2u;
    CHECK(rek_g1_hit_mujoco_candidate_to_contact(
        &candidates[0],
        &facts,
        &unchanged_contact,
        error,
        sizeof(error)) == REK_G1_HIT_MUJOCO_CALLER_FACTS_INVALID);
    CHECK(memcmp(
        &unchanged_contact,
        unchanged_bytes,
        sizeof(unchanged_contact)) == 0);
    facts.round_active = 1u;
    RekG1HitMujocoCandidate bad_candidate = candidates[0];
    bad_candidate.relative_speed_mps = NAN;
    CHECK(rek_g1_hit_mujoco_candidate_to_contact(
        &bad_candidate,
        &facts,
        &unchanged_contact,
        error,
        sizeof(error)) == REK_G1_HIT_MUJOCO_CALLER_FACTS_INVALID);
    CHECK(memcmp(
        &unchanged_contact,
        unchanged_bytes,
        sizeof(unchanged_contact)) == 0);

    const RekG1ImpactEvent move_7_event = {
        .impact_time_seconds = 1.0f,
        .lead_time_seconds = 0.2f,
        .release_time_seconds = 0.5f,
        .limb = REK_G1_AIM_LIMB_LEFT_LOWER_BODY,
    };
    RekG1HitMujocoCallerFacts move_7_facts = {
        .strike_intent = {
            .impact_events = &move_7_event,
            .impact_event_count = 1u,
            .clip_fps = 50.0f,
            .move_id = 7,
            .action_playing = 1u,
            .layer_active = 1u,
            .layer_loop = 0u,
        },
        .round_active = 1u,
        .fighter_upright = {1u, 1u},
        .fighter_standing = {1u, 1u},
    };
    const RekG1HitDetectorConfig move_7_config =
        rek_g1_current_build_hit_detector_config();
    RekG1HitDetectorState move_7_state;
    RekG1HitResult move_7_result;
    mjContact move_7_contact =
        make_contact(player_left_ankle, opponent_torso);
    CHECK(rek_g1_hit_mujoco_reset(
        &adapter, error, sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    CHECK(replace_contacts(model, arena_data[0], &move_7_contact, 1u));

    observation = observation_for(&duel, 0u, 0u);
    CHECK(rek_g1_hit_mujoco_scan_substep(
        &adapter, &observation, candidates, TEST_CANDIDATE_CAPACITY,
        &candidate_count, error, sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    CHECK(candidate_count == 1u);
    CHECK(candidates[0].striker_part == REK_G1_BODY_PART_FOOT);
    CHECK(candidates[0].striker_side == REK_G1_HAND_LEFT);
    CHECK(candidates[0].target_zone == REK_G1_BODY_ZONE_TORSO);
    move_7_facts.strike_intent.clip_cursor_frames = 42.0f;
    move_7_facts.time_seconds = 42.0f / 50.0f;
    CHECK(rek_g1_hit_mujoco_candidate_to_contact(
        &candidates[0], &move_7_facts, &measured_contact,
        error, sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    rek_g1_hit_detector_reset(&move_7_state);
    CHECK(rek_g1_hit_detector_process(
        &move_7_state, &move_7_config, &measured_contact, &move_7_result));
    CHECK(move_7_result.score_accepted == 0u);

    move_7_facts.strike_intent.clip_cursor_frames = 43.0f;
    move_7_facts.time_seconds = 43.0f / 50.0f;
    observation = observation_for(&duel, 0u, 1u);
    CHECK(rek_g1_hit_mujoco_scan_substep(
        &adapter, &observation, candidates, TEST_CANDIDATE_CAPACITY,
        &candidate_count, error, sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    CHECK(candidate_count == 0u);

    CHECK(replace_contacts(model, arena_data[0], NULL, 0u));
    observation = observation_for(&duel, 0u, 2u);
    CHECK(rek_g1_hit_mujoco_scan_substep(
        &adapter, &observation, candidates, TEST_CANDIDATE_CAPACITY,
        &candidate_count, error, sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    CHECK(candidate_count == 0u);
    CHECK(replace_contacts(model, arena_data[0], &move_7_contact, 1u));
    observation = observation_for(&duel, 0u, 3u);
    CHECK(rek_g1_hit_mujoco_scan_substep(
        &adapter, &observation, candidates, TEST_CANDIDATE_CAPACITY,
        &candidate_count, error, sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    CHECK(candidate_count == 1u);
    CHECK(rek_g1_hit_mujoco_candidate_to_contact(
        &candidates[0], &move_7_facts, &measured_contact,
        error, sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    CHECK(rek_g1_hit_detector_process(
        &move_7_state, &move_7_config, &measured_contact, &move_7_result));
    CHECK(move_7_result.score_accepted == 1u);
    CHECK(move_7_result.points_awarded == 2.0f);

    CHECK(replace_contacts(model, arena_data[0], NULL, 0u));
    observation = observation_for(&duel, 0u, 4u);
    CHECK(rek_g1_hit_mujoco_scan_substep(
        &adapter, &observation, candidates, TEST_CANDIDATE_CAPACITY,
        &candidate_count, error, sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    CHECK(candidate_count == 0u);
    CHECK(replace_contacts(model, arena_data[0], &move_7_contact, 1u));
    observation = observation_for(&duel, 0u, 5u);
    CHECK(rek_g1_hit_mujoco_scan_substep(
        &adapter, &observation, candidates, TEST_CANDIDATE_CAPACITY,
        &candidate_count, error, sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    CHECK(candidate_count == 1u);
    move_7_facts.strike_intent.clip_cursor_frames = 67.0f;
    move_7_facts.time_seconds = 67.0f / 50.0f;
    CHECK(rek_g1_hit_mujoco_candidate_to_contact(
        &candidates[0], &move_7_facts, &measured_contact,
        error, sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    rek_g1_hit_detector_reset(&move_7_state);
    CHECK(rek_g1_hit_detector_process(
        &move_7_state, &move_7_config, &measured_contact, &move_7_result));
    CHECK(move_7_result.score_accepted == 1u);

    CHECK(replace_contacts(model, arena_data[0], NULL, 0u));
    observation = observation_for(&duel, 0u, 6u);
    CHECK(rek_g1_hit_mujoco_scan_substep(
        &adapter, &observation, candidates, TEST_CANDIDATE_CAPACITY,
        &candidate_count, error, sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    CHECK(candidate_count == 0u);
    CHECK(replace_contacts(model, arena_data[0], &move_7_contact, 1u));
    observation = observation_for(&duel, 0u, 7u);
    CHECK(rek_g1_hit_mujoco_scan_substep(
        &adapter, &observation, candidates, TEST_CANDIDATE_CAPACITY,
        &candidate_count, error, sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    CHECK(candidate_count == 1u);
    move_7_facts.strike_intent.clip_cursor_frames = 68.0f;
    move_7_facts.time_seconds = 68.0f / 50.0f;
    CHECK(rek_g1_hit_mujoco_candidate_to_contact(
        &candidates[0], &move_7_facts, &measured_contact,
        error, sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    rek_g1_hit_detector_reset(&move_7_state);
    CHECK(rek_g1_hit_detector_process(
        &move_7_state, &move_7_config, &measured_contact, &move_7_result));
    CHECK(move_7_result.score_accepted == 0u);

    CHECK(rek_g1_hit_mujoco_clear_arena_contacts(
        &adapter, 0u, error, sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    CHECK(adapter.expected_substep[0] == 8u);
    observation = observation_for(&duel, 0u, 8u);
    CHECK(rek_g1_hit_mujoco_scan_substep(
        &adapter, &observation, candidates, TEST_CANDIDATE_CAPACITY,
        &candidate_count, error, sizeof(error)) == REK_G1_HIT_MUJOCO_OK);
    CHECK(candidate_count == 1u);

    CHECK(strcmp(
        rek_g1_hit_mujoco_status_string(REK_G1_HIT_MUJOCO_OK),
        "ok") == 0);
    exit_code = EXIT_SUCCESS;
    (void)printf(
        "{\"status\":\"ok\",\"assertions\":%zu,"
        "\"model_bodies\":%zu,\"model_geoms\":%zu,"
        "\"first_enter_candidates\":4,"
        "\"relative_speed_mps\":5.0,"
        "\"hand_side_values\":{\"left\":0,\"right\":1}}\n",
        assertion_count,
        (size_t)model->nbody,
        (size_t)model->ngeom);

cleanup:
    rek_g1_hit_mujoco_close(&adapter);
    for (size_t arena = 0u; arena < TEST_ARENAS; arena++) {
        if (arena_data[arena] != NULL) mj_deleteData(arena_data[arena]);
    }
    if (model != NULL) mj_deleteModel(model);
    return exit_code;
}
