#pragma once

#include <stddef.h>
#include <stdint.h>

#include "semantic_duel_runtime.h"

#define REK_G1_SEMANTIC_ASSET_MANIFEST_SHA256 \
    "b305abade35ffc01ccf284a5d81d88fa9c2bcd0204d8e7b98b5778289197e0b6"

enum {
    REK_G1_SEMANTIC_UNIQUE_CLIP_COUNT = 8,
    REK_G1_SEMANTIC_ASSET_PATH_CAPACITY = 4096,
};

typedef enum RekG1SemanticAssetsStatus {
    REK_G1_SEMANTIC_ASSETS_OK = 0,
    REK_G1_SEMANTIC_ASSETS_NULL_ARGUMENT = 1,
    REK_G1_SEMANTIC_ASSETS_PATH_INVALID = 2,
    REK_G1_SEMANTIC_ASSETS_FILE_INVALID = 3,
    REK_G1_SEMANTIC_ASSETS_HASH_MISMATCH = 4,
    REK_G1_SEMANTIC_ASSETS_ALLOCATION_FAILED = 5,
    REK_G1_SEMANTIC_ASSETS_CONTENT_INVALID = 6,
    REK_G1_SEMANTIC_ASSETS_ROUTE_MISMATCH = 7,
    REK_G1_SEMANTIC_ASSETS_UNSUPPORTED_ENDIAN = 8,
} RekG1SemanticAssetsStatus;

typedef struct RekG1SemanticClipStorage {
    int32_t npz_path_id;
    size_t frame_count;
    float* dof_position_mujoco;
    float* root_quaternion_wxyz;
} RekG1SemanticClipStorage;

/*
 * Owns only decoded binary32 arrays exported by
 * export_g1_semantic_duel_assets.py. The source NPZ files and proprietary game
 * binaries are never linked into the Puffer extension. Every opened file is
 * exact-size and SHA-256 checked before its address is exposed to the runtime.
 */
typedef struct RekG1SemanticAssets {
    char root[REK_G1_SEMANTIC_ASSET_PATH_CAPACITY];
    char model_path[REK_G1_SEMANTIC_ASSET_PATH_CAPACITY];
    RekG1SemanticClipStorage clips[REK_G1_SEMANTIC_UNIQUE_CLIP_COUNT];
    float* idle_root_position_m;
    float* idle_root_rotation_xyzw;
    RekG1SemanticDuelRouteAsset route_assets[REK_G1_STATIC_ROUTE_COUNT];
    GearSonicNativeMotion fixed_idle;
    uint8_t loaded;
} RekG1SemanticAssets;

const char* rek_g1_semantic_assets_status_string(
    RekG1SemanticAssetsStatus status);

/*
 * configured_compositor_kick_duration_ticks is indexed by runtime move id
 * minus six. The loader requires four nonzero traversal lengths and never
 * derives one from NPZ frame count. These values describe scheduler and
 * compositor traversal, not measured physical completion.
 */
RekG1SemanticAssetsStatus rek_g1_semantic_assets_load(
    RekG1SemanticAssets* assets,
    const char* root,
    const uint32_t configured_compositor_kick_duration_ticks[
        REK_G1_REQUIRED_KICK_COUNT],
    char* error,
    size_t error_capacity);

void rek_g1_semantic_assets_close(RekG1SemanticAssets* assets);
