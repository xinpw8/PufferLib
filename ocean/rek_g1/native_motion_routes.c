#include "native_motion_routes.h"

#include <math.h>
#include <string.h>

#define NO_RUNTIME_MOVE UINT16_MAX

static const RekG1NativeMotionRoute STATIC_ROUTES[REK_G1_STATIC_ROUTE_COUNT] = {
    {
        REK_G1_NATIVE_IDLE, REK_G1_NATIVE_ROUTE_IDLE, 2702, 377,
        50.0f, 1.0f, 39, 0, -1, 1.0f, 0.5139999985694885f,
        0.0f, 0, 1, NO_RUNTIME_MOVE,
    },
    {
        REK_G1_NATIVE_FORWARD, REK_G1_NATIVE_ROUTE_TRANSLATION, 2720, 370,
        50.0f, 1.0f, 36, 0, -1, 1.0f, 1.0f,
        0.0f, 0, 1, NO_RUNTIME_MOVE,
    },
    {
        REK_G1_NATIVE_BACKWARD, REK_G1_NATIVE_ROUTE_TRANSLATION, 2721, 370,
        50.0f, -1.0f, 36, 0, -1, 1.0f, 1.0f,
        0.0f, 0, 1, NO_RUNTIME_MOVE,
    },
    {
        REK_G1_NATIVE_STRAFE_LEFT, REK_G1_NATIVE_ROUTE_TRANSLATION, 2711, 388,
        50.0f, 1.0f, 35, 0, -1,
        0.09399999678134918f, 0.29499998688697815f,
        0.0f, 0, 1, NO_RUNTIME_MOVE,
    },
    {
        REK_G1_NATIVE_STRAFE_RIGHT, REK_G1_NATIVE_ROUTE_TRANSLATION, 2716, 388,
        50.0f, -1.0f, 35, 0, -1,
        0.09399999678134918f, 0.29499998688697815f,
        0.0f, 0, 1, NO_RUNTIME_MOVE,
    },
    {
        REK_G1_NATIVE_TURN_LEFT, REK_G1_NATIVE_ROUTE_TURN, 2719, 375,
        50.0f, -1.0f, 47, 0, -1,
        0.06499999761581421f, 0.07100000232458115f,
        1.0f, 0, 1, NO_RUNTIME_MOVE,
    },
    {
        REK_G1_NATIVE_TURN_RIGHT, REK_G1_NATIVE_ROUTE_TURN, 2718, 375,
        50.0f, 1.0f, 47, 0, -1,
        0.06499999761581421f, 0.07100000232458115f,
        1.0f, 0, 1, NO_RUNTIME_MOVE,
    },
    {
        REK_G1_NATIVE_KICK_MOVE_6_LEFT_SIDE, REK_G1_NATIVE_ROUTE_KICK, 2710, 392,
        50.0f, 1.0f, 158, 0, -1,
        0.035999998450279236f, 0.6899999976158142f,
        0.0f, 0, 0, 6,
    },
    {
        REK_G1_NATIVE_KICK_MOVE_7_LEFT_FRONT, REK_G1_NATIVE_ROUTE_KICK, 2703, 371,
        50.0f, 1.0f, 146, 0, -1,
        0.035999998450279236f, 0.6899999976158142f,
        1.0f, 0, 0, 7,
    },
    {
        REK_G1_NATIVE_KICK_MOVE_8_RIGHT_SIDE, REK_G1_NATIVE_ROUTE_KICK, 2715, 372,
        50.0f, 1.0f, 159, 0, -1,
        0.035999998450279236f, 0.6899999976158142f,
        0.0f, 0, 0, 8,
    },
    {
        REK_G1_NATIVE_KICK_MOVE_9_RIGHT_KNEE, REK_G1_NATIVE_ROUTE_KICK, 2714, 380,
        50.0f, 1.0f, 140, 0, -1,
        0.05999999865889549f, 0.550000011920929f,
        0.0f, 0, 0, 9,
    },
};

static const RekG1NativeMotionRouteTable STATIC_TABLE = {
    .build_fingerprint = REK_G1_STATIC_ROUTE_BUILD_FINGERPRINT,
    .source_probe_sha256 = REK_G1_STATIC_ROUTE_PROBE_SHA256,
    .robot_config_path_id = REK_G1_STATIC_ROBOT_CONFIG_PATH_ID,
    .routes = STATIC_ROUTES,
    .count = REK_G1_STATIC_ROUTE_COUNT,
};

const RekG1NativeMotionRouteTable* rek_g1_native_static_motion_routes(void) {
    return &STATIC_TABLE;
}

int rek_g1_native_validate_static_motion_routes(
        const RekG1NativeMotionRouteTable* table) {
    if (table == NULL || table->build_fingerprint == NULL
            || table->source_probe_sha256 == NULL
            || table->routes == NULL
            || strcmp(
                table->build_fingerprint,
                REK_G1_STATIC_ROUTE_BUILD_FINGERPRINT) != 0
            || strcmp(
                table->source_probe_sha256,
                REK_G1_STATIC_ROUTE_PROBE_SHA256) != 0
            || table->robot_config_path_id !=
                REK_G1_STATIC_ROBOT_CONFIG_PATH_ID
            || table->count != REK_G1_STATIC_ROUTE_COUNT) {
        return 0;
    }
    for (size_t index = 0; index < REK_G1_STATIC_ROUTE_COUNT; index++) {
        const RekG1NativeMotionRoute* actual = &table->routes[index];
        const RekG1NativeMotionRoute* expected = &STATIC_ROUTES[index];
        if (actual->id != expected->id || actual->kind != expected->kind
                || actual->mocap_clip_config_path_id !=
                    expected->mocap_clip_config_path_id
                || actual->npz_path_id != expected->npz_path_id
                || actual->asset_fps != expected->asset_fps
                || actual->playback_speed != expected->playback_speed
                || actual->asset_frames != expected->asset_frames
                || actual->start_frame != expected->start_frame
                || actual->end_frame != expected->end_frame
                || actual->blend_in_seconds != expected->blend_in_seconds
                || actual->blend_out_seconds != expected->blend_out_seconds
                || actual->yaw_blend != expected->yaw_blend
                || actual->mirror != expected->mirror
                || actual->loop != expected->loop
                || actual->runtime_move_index != expected->runtime_move_index
                || !isfinite(actual->asset_fps)
                || !isfinite(actual->playback_speed)
                || !isfinite(actual->blend_in_seconds)
                || !isfinite(actual->blend_out_seconds)
                || !isfinite(actual->yaw_blend)
                || actual->asset_fps <= 0.0f
                || actual->playback_speed == 0.0f
                || actual->mirror > 1u || actual->loop > 1u) {
            return 0;
        }
    }
    return 1;
}

const RekG1NativeMotionRoute* rek_g1_native_route_by_id(
        const RekG1NativeMotionRouteTable* table,
        RekG1NativeRouteId id) {
    if (!rek_g1_native_validate_static_motion_routes(table)
            || id < REK_G1_NATIVE_IDLE
            || id > REK_G1_NATIVE_KICK_MOVE_9_RIGHT_KNEE) {
        return NULL;
    }
    return &table->routes[(size_t)id];
}

const RekG1NativeMotionRoute* rek_g1_native_kick_route(
        const RekG1NativeMotionRouteTable* table,
        uint16_t runtime_move_index) {
    if (!rek_g1_native_validate_static_motion_routes(table)) return NULL;
    for (size_t index = 0; index < table->count; index++) {
        if (table->routes[index].kind == REK_G1_NATIVE_ROUTE_KICK
                && table->routes[index].runtime_move_index ==
                    runtime_move_index) {
            return &table->routes[index];
        }
    }
    return NULL;
}
