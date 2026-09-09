#pragma once

#include <stddef.h>
#include <stdint.h>

// Build-pinned static routing metadata for the current public-family G1
// candidate. These identities come from RobotConfig path 2722 and its exact
// MocapClipConfig/NPZ references in mujoco_asset_probe_v8.json. They describe
// candidate asset routing only. Asset identity does not establish runtime
// selection order, transition behavior, completion timing, or REK parity.

#define REK_G1_STATIC_ROUTE_BUILD_FINGERPRINT \
    "f84f187491e3b5cd73493de379ed972c5580b60d63f33956e396e6dec28b1659"
#define REK_G1_STATIC_ROUTE_PROBE_SHA256 \
    "b132eb19cb7b223a87ee3885c16e521e82d7e99006c09ed63e8cc899ad057686"

enum {
    REK_G1_STATIC_ROBOT_CONFIG_PATH_ID = 2722,
    REK_G1_STATIC_ROUTE_COUNT = 11,
};

typedef enum RekG1NativeRouteKind {
    REK_G1_NATIVE_ROUTE_IDLE = 0,
    REK_G1_NATIVE_ROUTE_TRANSLATION = 1,
    REK_G1_NATIVE_ROUTE_TURN = 2,
    REK_G1_NATIVE_ROUTE_KICK = 3,
} RekG1NativeRouteKind;

typedef enum RekG1NativeRouteId {
    REK_G1_NATIVE_IDLE = 0,
    REK_G1_NATIVE_FORWARD = 1,
    REK_G1_NATIVE_BACKWARD = 2,
    REK_G1_NATIVE_STRAFE_LEFT = 3,
    REK_G1_NATIVE_STRAFE_RIGHT = 4,
    REK_G1_NATIVE_TURN_LEFT = 5,
    REK_G1_NATIVE_TURN_RIGHT = 6,
    REK_G1_NATIVE_KICK_MOVE_6_LEFT_SIDE = 7,
    REK_G1_NATIVE_KICK_MOVE_7_LEFT_FRONT = 8,
    REK_G1_NATIVE_KICK_MOVE_8_RIGHT_SIDE = 9,
    REK_G1_NATIVE_KICK_MOVE_9_RIGHT_KNEE = 10,
} RekG1NativeRouteId;

typedef struct RekG1NativeMotionRoute {
    RekG1NativeRouteId id;
    RekG1NativeRouteKind kind;
    int32_t mocap_clip_config_path_id;
    int32_t npz_path_id;
    float asset_fps;
    float playback_speed;
    // NPZ frame count is asset metadata. It is not an observed action duration.
    uint32_t asset_frames;
    int32_t start_frame;
    int32_t end_frame;
    float blend_in_seconds;
    float blend_out_seconds;
    float yaw_blend;
    uint8_t mirror;
    uint8_t loop;
    // UINT16_MAX for non-kick routes. Kicks use exact runtime move indices 6..9.
    uint16_t runtime_move_index;
} RekG1NativeMotionRoute;

typedef struct RekG1NativeMotionRouteTable {
    const char* build_fingerprint;
    const char* source_probe_sha256;
    int32_t robot_config_path_id;
    const RekG1NativeMotionRoute* routes;
    size_t count;
} RekG1NativeMotionRouteTable;

const RekG1NativeMotionRouteTable* rek_g1_native_static_motion_routes(void);

int rek_g1_native_validate_static_motion_routes(
    const RekG1NativeMotionRouteTable* table);

const RekG1NativeMotionRoute* rek_g1_native_route_by_id(
    const RekG1NativeMotionRouteTable* table,
    RekG1NativeRouteId id);

const RekG1NativeMotionRoute* rek_g1_native_kick_route(
    const RekG1NativeMotionRouteTable* table,
    uint16_t runtime_move_index);
