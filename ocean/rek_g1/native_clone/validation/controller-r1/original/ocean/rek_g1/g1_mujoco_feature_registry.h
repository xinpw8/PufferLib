#pragma once

#include <stddef.h>
#include <stdint.h>

#include "g1_semantic_assets.h"
#include "sonic_motion_entry_matcher_native.h"

enum {
    REK_G1_MUJOCO_FEATURE_CLIP_COUNT = REK_G1_SEMANTIC_UNIQUE_CLIP_COUNT,
};

typedef enum RekG1MujocoFeatureRegistryStatus {
    REK_G1_MUJOCO_FEATURE_REGISTRY_OK = 0,
    REK_G1_MUJOCO_FEATURE_REGISTRY_NULL_ARGUMENT = 1,
    REK_G1_MUJOCO_FEATURE_REGISTRY_INVALID_DUEL = 2,
    REK_G1_MUJOCO_FEATURE_REGISTRY_INVALID_ASSETS = 3,
    REK_G1_MUJOCO_FEATURE_REGISTRY_MAPPING_MISSING = 4,
    REK_G1_MUJOCO_FEATURE_REGISTRY_MAPPING_MISMATCH = 5,
    REK_G1_MUJOCO_FEATURE_REGISTRY_SIZE_OVERFLOW = 6,
    REK_G1_MUJOCO_FEATURE_REGISTRY_ALLOCATION_FAILED = 7,
    REK_G1_MUJOCO_FEATURE_REGISTRY_NON_FINITE = 8,
    REK_G1_MUJOCO_FEATURE_REGISTRY_MATCHER_FAILED = 9,
    REK_G1_MUJOCO_FEATURE_REGISTRY_BACKEND_FAILED = 10,
    REK_G1_MUJOCO_FEATURE_REGISTRY_NOT_READY = 11,
} RekG1MujocoFeatureRegistryStatus;

/*
 * Owns the scratch MuJoCo state and the derived six-float foot-feature rows.
 * The model and all clip arrays remain caller-owned. One registry may be
 * shared by every composer in a semantic duel because current batch routing
 * invokes composer callbacks serially.
 */
typedef struct RekG1MujocoFeatureRegistry {
    GearSonicNativeDuelVector* duel;
    const RekG1SemanticAssets* assets;
    mjData* scratch;
    int root_body_id;
    int left_ankle_roll_body_id;
    int right_ankle_roll_body_id;
    SonicMotionEntryMatcherNative matcher;
    SonicMotionEntryMatcherNativeFeatureSlot
        slots[REK_G1_MUJOCO_FEATURE_CLIP_COUNT];
    SonicMotionComposerNativeClip
        clip_views[REK_G1_MUJOCO_FEATURE_CLIP_COUNT];
    size_t feature_offsets[REK_G1_MUJOCO_FEATURE_CLIP_COUNT];
    float* root_local_foot_xyz;
    size_t total_feature_count;
    size_t total_frame_count;
    SonicMotionComposerNativeBackends delegated_backends;
    RekG1MujocoFeatureRegistryStatus last_status;
    uint8_t initialized;
    uint8_t ready;
    uint8_t bound;
} RekG1MujocoFeatureRegistry;

const char* rek_g1_mujoco_feature_registry_status_string(
    RekG1MujocoFeatureRegistryStatus status);

/*
 * Validate the exact two-fighter map, create one scratch mjData, pose every
 * frame of all eight caller-owned clip identities, bake Unity-order root-local
 * ankle-roll coordinates, and register every identity with the native matcher.
 * This never reads or mutates any live arena mjData.
 */
RekG1MujocoFeatureRegistryStatus rek_g1_mujoco_feature_registry_open(
    RekG1MujocoFeatureRegistry* registry,
    GearSonicNativeDuelVector* duel,
    const RekG1SemanticAssets* assets,
    char* error,
    size_t error_capacity);

/*
 * Public sampler boundary used during the bake. The output order is current
 * runner order:
 *   left Unity-local x,y,z, right Unity-local x,y,z.
 */
int rek_g1_mujoco_feature_registry_sample(
    void* context,
    const float dof_position_mujoco[SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT],
    float root_local_foot_xyz[
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_FEATURE_WIDTH]);

/*
 * Install an aggregate backend context into a semantic-duel config backend.
 * Existing quaternion/atan2/sin-cos callbacks and their context are retained
 * behind adapters. An already-installed loop matcher is rejected rather than
 * silently replaced.
 */
RekG1MujocoFeatureRegistryStatus
rek_g1_mujoco_feature_registry_bind_backends(
    RekG1MujocoFeatureRegistry* registry,
    SonicMotionComposerNativeBackends* backends,
    char* error,
    size_t error_capacity);

void rek_g1_mujoco_feature_registry_close(
    RekG1MujocoFeatureRegistry* registry);
