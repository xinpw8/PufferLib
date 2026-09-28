#include "g1_mujoco_feature_registry.h"
#include "sonic_motion_composer_libm_candidate.h"

#include <math.h>
#include <openssl/evp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int assertions;
static const char* const EXPECTED_FEATURE_SHA256 =
    "a1ac27c3e8331e9b383e676f78c4721c9b609f6b4daccc7254c8424057ecdc09";

static void require(int condition, const char* name) {
    assertions += 1;
    if (!condition) {
        fprintf(stderr, "FAIL %s\n", name);
        exit(1);
    }
}

static int sha256_hex(
        const void* data, size_t bytes, char output[65]) {
    static const char HEX[] = "0123456789abcdef";
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_length = 0u;
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    if (context == NULL || data == NULL || output == NULL
            || EVP_DigestInit_ex(context, EVP_sha256(), NULL) != 1
            || EVP_DigestUpdate(context, data, bytes) != 1
            || EVP_DigestFinal_ex(context, digest, &digest_length) != 1
            || digest_length != 32u) {
        EVP_MD_CTX_free(context);
        return 0;
    }
    EVP_MD_CTX_free(context);
    for (size_t index = 0; index < 32u; index++) {
        output[index * 2u] = HEX[digest[index] >> 4u];
        output[index * 2u + 1u] = HEX[digest[index] & 15u];
    }
    output[64] = '\0';
    return 1;
}

static SonicMotionComposerNativeLayer route_layer(
        const RekG1SemanticAssets* assets,
        const RekG1NativeMotionRoute* route) {
    SonicMotionComposerNativeLayer layer = {0};
    layer.clip = assets->route_assets[(size_t)route->id].clip;
    const int32_t resolved_end = route->end_frame < 0
        ? (int32_t)(layer.clip.frame_count - 1u)
        : route->end_frame;
    layer.config = (SonicMotionComposerNativeConfig){
        .mirror = route->mirror,
        .loop = route->loop,
        .playback_speed = route->playback_speed,
        .start_frame = route->start_frame,
        .end_frame = route->end_frame,
        .blend_in_seconds = route->blend_in_seconds,
        .blend_out_seconds = route->blend_out_seconds,
        .yaw_blend = route->yaw_blend,
    };
    layer.speed = route->playback_speed;
    layer.per_tick = route->asset_fps * route->playback_speed
        / (float)REK_G1_SEMANTIC_DUEL_CONTROLLER_RATE_HZ;
    layer.cursor = (float)route->start_frame;
    layer.start_frame = route->start_frame;
    layer.end_frame = resolved_end;
    layer.has_clip = 1;
    layer.has_config = 1;
    layer.active = 1;
    return layer;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s ASSET_DIR ENCODER DECODER\n", argv[0]);
        return 64;
    }
    char error[1024] = {0};
    RekG1SemanticAssets assets = {0};
    RekG1SemanticAssetsStatus asset_status = rek_g1_semantic_assets_load(
        &assets,
        argv[1],
        REK_G1_PINNED_COMPOSITOR_MOVE_DURATION_TICKS,
        error,
        sizeof(error));
    if (asset_status != REK_G1_SEMANTIC_ASSETS_OK) {
        fprintf(stderr, "asset load failed: %s: %s\n",
            rek_g1_semantic_assets_status_string(asset_status), error);
        return 1;
    }

    GearSonicNativeDuelVector duel = {0};
    if (!gear_sonic_native_duel_open(
            &duel,
            assets.model_path,
            argv[2],
            argv[3],
            assets.fixed_idle,
            32u,
            1,
            error,
            sizeof(error))) {
        fprintf(stderr, "duel open failed: %s\n", error);
        rek_g1_semantic_assets_close(&assets);
        return 1;
    }
    require(duel.robot_count == 64u, "duel_batch64");
    require(duel.data[0] != NULL, "duel_live_data");
    double qpos_before[GEAR_SONIC_DUEL_QPOS_DIM];
    double qvel_before[GEAR_SONIC_DUEL_QVEL_DIM];
    memcpy(qpos_before, duel.data[0]->qpos, sizeof(qpos_before));
    memcpy(qvel_before, duel.data[0]->qvel, sizeof(qvel_before));
    const double time_before = duel.data[0]->time;

    RekG1MujocoFeatureRegistry invalid = {0};
    RekG1SemanticAssets unloaded = assets;
    unloaded.loaded = 0u;
    require(rek_g1_mujoco_feature_registry_open(
        &invalid, &duel, &unloaded, error, sizeof(error))
        == REK_G1_MUJOCO_FEATURE_REGISTRY_INVALID_ASSETS,
        "unloaded_assets_rejected");
    rek_g1_mujoco_feature_registry_close(&invalid);

    const int saved_root_body =
        duel.fighters[GEAR_SONIC_DUEL_PLAYER].root_body_id;
    duel.fighters[GEAR_SONIC_DUEL_PLAYER].root_body_id = -1;
    require(rek_g1_mujoco_feature_registry_open(
        &invalid, &duel, &assets, error, sizeof(error))
        == REK_G1_MUJOCO_FEATURE_REGISTRY_MAPPING_MISMATCH,
        "bad_root_mapping_rejected");
    rek_g1_mujoco_feature_registry_close(&invalid);
    duel.fighters[GEAR_SONIC_DUEL_PLAYER].root_body_id = saved_root_body;

    const int saved_joint_zero =
        duel.fighters[GEAR_SONIC_DUEL_PLAYER].joint_ids[0];
    const int saved_joint_one =
        duel.fighters[GEAR_SONIC_DUEL_PLAYER].joint_ids[1];
    const int saved_qpos_zero =
        duel.fighters[GEAR_SONIC_DUEL_PLAYER].qpos_addresses[0];
    const int saved_qpos_one =
        duel.fighters[GEAR_SONIC_DUEL_PLAYER].qpos_addresses[1];
    duel.fighters[GEAR_SONIC_DUEL_PLAYER].joint_ids[0] = saved_joint_one;
    duel.fighters[GEAR_SONIC_DUEL_PLAYER].joint_ids[1] = saved_joint_zero;
    duel.fighters[GEAR_SONIC_DUEL_PLAYER].qpos_addresses[0] = saved_qpos_one;
    duel.fighters[GEAR_SONIC_DUEL_PLAYER].qpos_addresses[1] = saved_qpos_zero;
    require(rek_g1_mujoco_feature_registry_open(
        &invalid, &duel, &assets, error, sizeof(error))
        == REK_G1_MUJOCO_FEATURE_REGISTRY_MAPPING_MISMATCH,
        "permuted_joint_mapping_rejected");
    rek_g1_mujoco_feature_registry_close(&invalid);
    duel.fighters[GEAR_SONIC_DUEL_PLAYER].joint_ids[0] = saved_joint_zero;
    duel.fighters[GEAR_SONIC_DUEL_PLAYER].joint_ids[1] = saved_joint_one;
    duel.fighters[GEAR_SONIC_DUEL_PLAYER].qpos_addresses[0] = saved_qpos_zero;
    duel.fighters[GEAR_SONIC_DUEL_PLAYER].qpos_addresses[1] = saved_qpos_one;

    RekG1MujocoFeatureRegistry first = {0};
    RekG1MujocoFeatureRegistryStatus status =
        rek_g1_mujoco_feature_registry_open(
            &first, &duel, &assets, error, sizeof(error));
    if (status != REK_G1_MUJOCO_FEATURE_REGISTRY_OK) {
        fprintf(stderr, "feature registry open failed: %s: %s\n",
            rek_g1_mujoco_feature_registry_status_string(status), error);
        gear_sonic_native_duel_close(&duel);
        rek_g1_semantic_assets_close(&assets);
        return 1;
    }
    require(first.ready == 1u, "registry_ready");
    require(first.matcher.slot_count == REK_G1_MUJOCO_FEATURE_CLIP_COUNT,
        "all_unique_clips_registered");
    require(first.total_frame_count == 1704u, "all_frames_baked");
    require(first.total_feature_count == 10224u, "six_values_per_frame");
    require(first.scratch != duel.data[0], "scratch_not_live_data");
    require(memcmp(qpos_before, duel.data[0]->qpos, sizeof(qpos_before)) == 0,
        "live_qpos_unchanged");
    require(memcmp(qvel_before, duel.data[0]->qvel, sizeof(qvel_before)) == 0,
        "live_qvel_unchanged");
    require(duel.data[0]->time == time_before, "live_time_unchanged");
    for (size_t index = 0; index < first.total_feature_count; index++) {
        require(isfinite(first.root_local_foot_xyz[index]),
            "baked_feature_finite");
    }
    for (size_t clip_index = 0;
            clip_index < REK_G1_MUJOCO_FEATURE_CLIP_COUNT;
            clip_index++) {
        const SonicMotionEntryMatcherNativeFeatureSlot* slot =
            &first.slots[clip_index];
        require(slot->registered == 1, "slot_registered");
        require(slot->root_local_foot_xyz
            == first.root_local_foot_xyz + first.feature_offsets[clip_index],
            "slot_feature_identity");
        require(slot->clip_frame_count == assets.clips[clip_index].frame_count,
            "slot_frame_identity");
    }

    float sampled[SONIC_MOTION_ENTRY_MATCHER_NATIVE_FEATURE_WIDTH];
    require(rek_g1_mujoco_feature_registry_sample(
        &first, first.clip_views[0].dof_position_mujoco, sampled),
        "direct_sample");
    require(memcmp(
        sampled,
        first.root_local_foot_xyz + first.feature_offsets[0],
        sizeof(sampled)) == 0,
        "direct_sample_matches_bake");
    require(sampled[1] < -0.5f && fabsf(sampled[2]) < 0.5f,
        "unity_axis_order_vertical_is_y");

    float invalid_dof[SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT];
    memcpy(invalid_dof, first.clip_views[0].dof_position_mujoco,
        sizeof(invalid_dof));
    invalid_dof[4] = NAN;
    require(!rek_g1_mujoco_feature_registry_sample(
        &first, invalid_dof, sampled), "nonfinite_dof_rejected");
    require(first.last_status == REK_G1_MUJOCO_FEATURE_REGISTRY_NON_FINITE,
        "nonfinite_dof_status");

    RekG1MujocoFeatureRegistry second = {0};
    status = rek_g1_mujoco_feature_registry_open(
        &second, &duel, &assets, error, sizeof(error));
    require(status == REK_G1_MUJOCO_FEATURE_REGISTRY_OK,
        "second_registry_open");
    require(second.total_feature_count == first.total_feature_count,
        "repeat_feature_count");
    require(memcmp(
        first.root_local_foot_xyz,
        second.root_local_foot_xyz,
        first.total_feature_count * sizeof(float)) == 0,
        "repeat_bake_bitwise_deterministic");

    SonicMotionComposerNativeBackends backends = {
        .quaternion_slerp =
            sonic_motion_composer_libm_candidate_quaternion_slerp,
        .atan2_f = sonic_motion_composer_libm_candidate_atan2_f,
        .sin_cos_f = sonic_motion_composer_libm_candidate_sin_cos_f,
    };
    status = rek_g1_mujoco_feature_registry_bind_backends(
        &first, &backends, error, sizeof(error));
    require(status == REK_G1_MUJOCO_FEATURE_REGISTRY_OK,
        "bind_backends");
    require(backends.context == &first, "aggregate_backend_context");
    require(backends.loop_entry_matcher != NULL, "loop_matcher_installed");
    float identity[4];
    const float unit[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    require(backends.quaternion_slerp(
        backends.context, unit, unit, 0.5f, identity),
        "delegated_slerp");
    require(memcmp(identity, unit, sizeof(unit)) == 0,
        "delegated_slerp_value");

    const RekG1NativeMotionRouteTable* route_table =
        rek_g1_native_static_motion_routes();
    const RekG1NativeMotionRoute* idle_route = rek_g1_native_route_by_id(
        route_table, REK_G1_NATIVE_IDLE);
    const RekG1NativeMotionRoute* forward_route = rek_g1_native_route_by_id(
        route_table, REK_G1_NATIVE_FORWARD);
    require(idle_route != NULL && forward_route != NULL,
        "loop_match_routes");
    SonicMotionComposerNativeLayer outgoing = route_layer(
        &assets, idle_route);
    SonicMotionComposerNativeLayer target = route_layer(
        &assets, forward_route);
    float matched_cursor = NAN;
    const int loop_match_ok = backends.loop_entry_matcher(
        backends.context, &target, &outgoing, &matched_cursor);
    if (!loop_match_ok) {
        fprintf(stderr, "loop match status: registry=%s matcher=%s\n",
            rek_g1_mujoco_feature_registry_status_string(first.last_status),
            sonic_motion_entry_matcher_native_status_string(
                first.matcher.last_status));
    }
    require(loop_match_ok, "real_feature_loop_match");
    require(isfinite(matched_cursor)
        && matched_cursor >= (float)target.start_frame
        && matched_cursor <= (float)target.end_frame,
        "matched_cursor_in_route");

    char digest[65];
    require(sha256_hex(
        first.root_local_foot_xyz,
        first.total_feature_count * sizeof(float),
        digest),
        "feature_sha256");
    if (strcmp(digest, EXPECTED_FEATURE_SHA256) != 0) {
        fprintf(stderr, "feature digest mismatch: actual=%s expected=%s\n",
            digest, EXPECTED_FEATURE_SHA256);
    }
    require(strcmp(digest, EXPECTED_FEATURE_SHA256) == 0,
        "feature_sha256_regression");
    printf(
        "{\"schema\":\"rek.g1_mujoco_feature_registry_test.v1\","
        "\"assertions\":%d,\"clips\":%u,\"frames\":%zu,"
        "\"features\":%zu,\"feature_sha256\":\"%s\","
        "\"first_feature_unity_xyz\":[%.9g,%.9g,%.9g,%.9g,%.9g,%.9g],"
        "\"idle_to_forward_cursor\":%.9g}\n",
        assertions,
        (unsigned)REK_G1_MUJOCO_FEATURE_CLIP_COUNT,
        first.total_frame_count,
        first.total_feature_count,
        digest,
        first.root_local_foot_xyz[0],
        first.root_local_foot_xyz[1],
        first.root_local_foot_xyz[2],
        first.root_local_foot_xyz[3],
        first.root_local_foot_xyz[4],
        first.root_local_foot_xyz[5],
        matched_cursor);

    rek_g1_mujoco_feature_registry_close(&second);
    rek_g1_mujoco_feature_registry_close(&first);
    gear_sonic_native_duel_close(&duel);
    rek_g1_semantic_assets_close(&assets);
    return 0;
}
