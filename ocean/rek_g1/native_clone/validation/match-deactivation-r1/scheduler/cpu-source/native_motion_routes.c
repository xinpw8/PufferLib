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
        REK_G1_NATIVE_MOVE_6_LEFT_SIDE, REK_G1_NATIVE_ROUTE_DISCRETE_MOVE,
        2710, 392,
        50.0f, 1.0f, 158, 0, -1,
        0.035999998450279236f, 0.6899999976158142f,
        0.0f, 0, 0, 6,
    },
    {
        REK_G1_NATIVE_MOVE_7_LEFT_FRONT, REK_G1_NATIVE_ROUTE_DISCRETE_MOVE,
        2703, 371,
        50.0f, 1.0f, 146, 0, -1,
        0.035999998450279236f, 0.6899999976158142f,
        1.0f, 0, 0, 7,
    },
    {
        REK_G1_NATIVE_MOVE_8_RIGHT_SIDE, REK_G1_NATIVE_ROUTE_DISCRETE_MOVE,
        2715, 372,
        50.0f, 1.0f, 159, 0, -1,
        0.035999998450279236f, 0.6899999976158142f,
        0.0f, 0, 0, 8,
    },
    {
        REK_G1_NATIVE_MOVE_9_RIGHT_KNEE, REK_G1_NATIVE_ROUTE_DISCRETE_MOVE,
        2714, 380,
        50.0f, 1.0f, 140, 0, -1,
        0.05999999865889549f, 0.550000011920929f,
        0.0f, 0, 0, 9,
    },
    {
        REK_G1_NATIVE_MOVE_0_LEFT_HOOK, REK_G1_NATIVE_ROUTE_DISCRETE_MOVE,
        2704, 381, 50.0f, 1.0f, 36, 0, -1,
        0.0f, 0.8479999899864197f, 0.0f, 0, 0, 0,
    },
    {
        REK_G1_NATIVE_MOVE_1_LEFT_JAB, REK_G1_NATIVE_ROUTE_DISCRETE_MOVE,
        2706, 379, 50.0f, 1.2599999904632568f, 34, 0, -1,
        0.0f, 0.8870000243186951f, 0.0f, 0, 0, 1,
    },
    {
        REK_G1_NATIVE_MOVE_2_DOUBLE_UPPERCUT,
        REK_G1_NATIVE_ROUTE_DISCRETE_MOVE,
        2701, 374, 50.0f, 1.8200000524520874f, 57, 0, -1,
        0.0f, 0.7480000257492065f, 0.0f, 0, 0, 2,
    },
    {
        REK_G1_NATIVE_MOVE_3_RIGHT_HOOK, REK_G1_NATIVE_ROUTE_DISCRETE_MOVE,
        2712, 373, 50.0f, 1.0f, 46, 0, -1,
        0.0f, 0.6460000276565552f, 0.0f, 0, 0, 3,
    },
    {
        REK_G1_NATIVE_MOVE_4_RIGHT_JAB, REK_G1_NATIVE_ROUTE_DISCRETE_MOVE,
        2713, 387, 50.0f, 1.25f, 40, 0, -1,
        0.0f, 0.781000018119812f, 0.0f, 0, 0, 4,
    },
    {
        REK_G1_NATIVE_MOVE_5_LEFT_JAB_RIGHT_UPPERCUT,
        REK_G1_NATIVE_ROUTE_DISCRETE_MOVE,
        2707, 376, 50.0f, 1.0f, 46, 0, -1,
        0.0f, 0.8240000009536743f, 0.0f, 0, 0, 5,
    },
    {
        REK_G1_NATIVE_MOVE_10_SIX_PUNCH,
        REK_G1_NATIVE_ROUTE_DISCRETE_MOVE,
        2698, 378, 50.0f, 1.1200000047683716f, 151, 0, -1,
        0.10000000149011612f, 0.10000000149011612f,
        0.0f, 0, 0, 10,
    },
    {
        REK_G1_NATIVE_MOVE_11_RUN_AND_PUNCH,
        REK_G1_NATIVE_ROUTE_DISCRETE_MOVE,
        2717, 383, 50.0f, 1.0f, 139, 0, -1,
        0.2750000059604645f, 0.8500000238418579f,
        0.0f, 0, 0, 11,
    },
    {
        REK_G1_NATIVE_MOVE_12_LEFT_RIGHT_JAB,
        REK_G1_NATIVE_ROUTE_DISCRETE_MOVE,
        2709, 386, 50.0f, 1.0f, 74, 0, -1,
        0.10000000149011612f, 0.10000000149011612f,
        0.0f, 0, 0, 12,
    },
    {
        REK_G1_NATIVE_MOVE_13_LEFT_RIGHT_HOOK,
        REK_G1_NATIVE_ROUTE_DISCRETE_MOVE,
        2708, 384, 50.0f, 1.0f, 76, 0, -1,
        0.10000000149011612f, 0.10000000149011612f,
        0.0f, 0, 0, 13,
    },
    {
        REK_G1_NATIVE_MOVE_14_LEFT_HOOK_RIGHT_JAB,
        REK_G1_NATIVE_ROUTE_DISCRETE_MOVE,
        2705, 385, 50.0f, 1.0f, 69, 0, -1,
        0.10000000149011612f, 0.10000000149011612f,
        0.0f, 0, 0, 14,
    },
    {
        REK_G1_NATIVE_MOVE_15_DOUBLE_HOOK,
        REK_G1_NATIVE_ROUTE_DISCRETE_MOVE,
        2700, 390, 50.0f, 1.0f, 72, 0, -1,
        0.0f, 0.4699999988079071f, 0.0f, 0, 0, 15,
    },
    {
        REK_G1_NATIVE_MOVE_16_BUTT_SMACK_EMOTE,
        REK_G1_NATIVE_ROUTE_DISCRETE_MOVE,
        2699, 382, 50.0f, 1.0f, 104, 0, -1,
        0.03999999910593033f, 0.9819999933242798f,
        0.0f, 0, 0, 16,
    },
};

static const RekG1NativeMotionRouteTable STATIC_TABLE = {
    .build_fingerprint = REK_G1_STATIC_ROUTE_BUILD_FINGERPRINT,
    .source_probe_sha256 = REK_G1_STATIC_ROUTE_PROBE_SHA256,
    .robot_config_path_id = REK_G1_STATIC_ROBOT_CONFIG_PATH_ID,
    .routes = STATIC_ROUTES,
    .count = REK_G1_STATIC_ROUTE_COUNT,
};

static int float_bits_equal(float left, float right) {
    uint32_t left_bits = 0u;
    uint32_t right_bits = 0u;
    memcpy(&left_bits, &left, sizeof(left_bits));
    memcpy(&right_bits, &right, sizeof(right_bits));
    return left_bits == right_bits;
}

const RekG1NativeMotionRouteTable* rek_g1_native_static_motion_routes(void) {
    return &STATIC_TABLE;
}

int rek_g1_native_validate_static_motion_routes(
        const RekG1NativeMotionRouteTable* table) {
    if (table == &STATIC_TABLE) return 1;
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
                || !float_bits_equal(actual->asset_fps, expected->asset_fps)
                || !float_bits_equal(
                    actual->playback_speed, expected->playback_speed)
                || actual->asset_frames != expected->asset_frames
                || actual->start_frame != expected->start_frame
                || actual->end_frame != expected->end_frame
                || !float_bits_equal(
                    actual->blend_in_seconds, expected->blend_in_seconds)
                || !float_bits_equal(
                    actual->blend_out_seconds, expected->blend_out_seconds)
                || !float_bits_equal(actual->yaw_blend, expected->yaw_blend)
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
            || id > REK_G1_NATIVE_MOVE_16_BUTT_SMACK_EMOTE) {
        return NULL;
    }
    return &table->routes[(size_t)id];
}

const RekG1NativeMotionRoute* rek_g1_native_discrete_move_route(
        const RekG1NativeMotionRouteTable* table,
        uint16_t runtime_move_index) {
    if (!rek_g1_native_validate_static_motion_routes(table)) return NULL;
    for (size_t index = 0; index < table->count; index++) {
        if (table->routes[index].kind == REK_G1_NATIVE_ROUTE_DISCRETE_MOVE
                && table->routes[index].runtime_move_index ==
                    runtime_move_index) {
            return &table->routes[index];
        }
    }
    return NULL;
}

const RekG1NativeMotionRoute* rek_g1_native_kick_route(
        const RekG1NativeMotionRouteTable* table,
        uint16_t runtime_move_index) {
    if (runtime_move_index < 6u || runtime_move_index > 9u) return NULL;
    return rek_g1_native_discrete_move_route(table, runtime_move_index);
}
