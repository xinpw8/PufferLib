#include "g1_semantic_assets.h"
#include "sonic_motion_composer_libm_candidate.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int assertions;

static void require(int condition, const char* name) {
    assertions += 1;
    if (!condition) {
        fprintf(stderr, "FAIL %s\n", name);
        exit(1);
    }
}

static uint32_t parse_ticks(const char* text) {
    char* end = NULL;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    require(errno == 0 && end != text && *end == '\0'
        && value > 0ul && value <= UINT32_MAX, "parse_duration");
    return (uint32_t)value;
}

static float heading_wxyz(const float* quaternion) {
    const float w = quaternion[0];
    const float x = quaternion[1];
    const float y = quaternion[2];
    const float z = quaternion[3];
    return atan2f(
        2.0f * (x * y + z * w),
        1.0f - 2.0f * (y * y + z * z));
}

static float heading_xyzw(const float* quaternion) {
    const float wxyz[4] = {
        quaternion[3], quaternion[0], quaternion[1], quaternion[2],
    };
    return heading_wxyz(wxyz);
}

static SonicMotionComposerNativeConfig route_config(
        const RekG1NativeMotionRoute* route) {
    return (SonicMotionComposerNativeConfig){
        .mirror = route->mirror,
        .loop = route->loop,
        .playback_speed = route->playback_speed,
        .start_frame = route->start_frame,
        .end_frame = route->end_frame,
        .blend_in_seconds = route->blend_in_seconds,
        .blend_out_seconds = route->blend_out_seconds,
        .yaw_blend = route->yaw_blend,
    };
}

static uint32_t measure_composer_completion_ticks(
        const RekG1SemanticDuelRouteAsset* asset,
        const RekG1NativeMotionRoute* route) {
    SonicMotionComposerNativeBackends backends = {
        .quaternion_slerp =
            sonic_motion_composer_libm_candidate_quaternion_slerp,
        .atan2_f = sonic_motion_composer_libm_candidate_atan2_f,
        .sin_cos_f = sonic_motion_composer_libm_candidate_sin_cos_f,
    };
    SonicMotionComposerNative composer;
    SonicMotionComposerNativeConfig config = route_config(route);
    require(sonic_motion_composer_native_init(&composer, 50, &backends)
        == SONIC_MOTION_COMPOSER_NATIVE_OK, "duration_composer_init");
    require(sonic_motion_composer_native_play_action_immediate(
        &composer, &asset->clip, &config)
        == SONIC_MOTION_COMPOSER_NATIVE_OK, "duration_play_action");
    require(composer.action_playing == 1, "duration_action_started");
    uint32_t ticks = 0u;
    while (composer.action_playing) {
        SonicMotionComposerNativeAdvanceResult advance;
        require(sonic_motion_composer_native_advance(&composer, &advance)
            == SONIC_MOTION_COMPOSER_NATIVE_OK, "duration_advance");
        ticks += 1u;
        require(ticks < 10000u, "duration_bounded");
    }
    require(composer.current_layer.active == 1,
        "duration_terminal_layer_remains_active");
    require(composer.current_layer.config.loop == 0,
        "duration_terminal_layer_nonloop");
    return ticks;
}

int main(int argc, char** argv) {
    if (argc != 6) {
        fprintf(stderr,
            "usage: %s ASSET_DIR MOVE6_TICKS MOVE7_TICKS MOVE8_TICKS MOVE9_TICKS\n",
            argv[0]);
        return 2;
    }
    uint32_t durations[REK_G1_REQUIRED_KICK_COUNT];
    for (size_t index = 0; index < REK_G1_REQUIRED_KICK_COUNT; index++) {
        durations[index] = parse_ticks(argv[index + 2]);
    }

    RekG1SemanticAssets assets;
    char error[512] = {0};
    RekG1SemanticAssetsStatus status = rek_g1_semantic_assets_load(
        &assets, argv[1], durations, error, sizeof(error));
    if (status != REK_G1_SEMANTIC_ASSETS_OK) {
        fprintf(stderr, "load failed: status=%s error=%s\n",
            rek_g1_semantic_assets_status_string(status), error);
        return 1;
    }
    require(assets.loaded == 1u, "loaded_flag");
    require(assets.fixed_idle.frames == 39u, "idle_frames");
    require(assets.fixed_idle.loop == 1, "idle_loop");
    require(assets.fixed_idle.dof_position_mujoco
        == assets.route_assets[REK_G1_NATIVE_IDLE].clip.dof_position_mujoco,
        "idle_dof_alias");
    require(assets.route_assets[REK_G1_NATIVE_FORWARD].clip.dof_position_mujoco
        == assets.route_assets[REK_G1_NATIVE_BACKWARD].clip.dof_position_mujoco,
        "walk_route_alias");
    require(assets.route_assets[REK_G1_NATIVE_STRAFE_LEFT].clip.dof_position_mujoco
        == assets.route_assets[REK_G1_NATIVE_STRAFE_RIGHT].clip.dof_position_mujoco,
        "strafe_route_alias");
    require(assets.route_assets[REK_G1_NATIVE_TURN_LEFT].clip.dof_position_mujoco
        == assets.route_assets[REK_G1_NATIVE_TURN_RIGHT].clip.dof_position_mujoco,
        "turn_route_alias");
    for (size_t index = 0; index < REK_G1_STATIC_ROUTE_COUNT; index++) {
        require(assets.route_assets[index].route_id == (RekG1NativeRouteId)index,
            "route_id_order");
        const RekG1NativeMotionRoute* route = rek_g1_native_route_by_id(
            rek_g1_native_static_motion_routes(), (RekG1NativeRouteId)index);
        require(route != NULL, "route_exists");
        require(assets.route_assets[index].clip.frame_count == route->asset_frames,
            "route_frame_count");
        require(assets.route_assets[index].clip.fps == route->asset_fps,
            "route_fps");
        require(fabsf(heading_wxyz(
            assets.route_assets[index].clip.root_quaternion_wxyz)) <= 2.0e-6f,
            "clip_frame_zero_heading_normalized");
        if (route->kind == REK_G1_NATIVE_ROUTE_KICK) {
            require(assets.route_assets[index].configured_compositor_duration_ticks
                == durations[route->runtime_move_index - 6u],
                "kick_duration_identity");
            require(measure_composer_completion_ticks(
                &assets.route_assets[index], route)
                == durations[route->runtime_move_index - 6u],
                "kick_duration_matches_composer_terminal");
        } else {
            require(assets.route_assets[index].configured_compositor_duration_ticks == 0u,
                "nonkick_duration_zero");
        }
    }
    require(fabsf(heading_xyzw(assets.fixed_idle.root_rotation_xyzw))
        <= 2.0e-6f, "fixed_idle_frame_zero_heading_normalized");
    require(strstr(assets.model_path, "model.two_fighter_arena.xml") != NULL,
        "model_path");
    rek_g1_semantic_assets_close(&assets);
    require(assets.loaded == 0u, "close_clears_state");

    uint32_t invalid_durations[REK_G1_REQUIRED_KICK_COUNT] = {
        durations[0], durations[1], 0u, durations[3],
    };
    memset(error, 0, sizeof(error));
    status = rek_g1_semantic_assets_load(
        &assets, argv[1], invalid_durations, error, sizeof(error));
    require(status == REK_G1_SEMANTIC_ASSETS_CONTENT_INVALID,
        "missing_duration_rejected");
    require(assets.loaded == 0u, "missing_duration_not_loaded");

    printf(
        "G1 semantic asset loader passed: assertions=%d manifest_sha256=%s\n",
        assertions,
        REK_G1_SEMANTIC_ASSET_MANIFEST_SHA256);
    return 0;
}
