#pragma once

#include "sonic_motion_composer_native.h"

#include <stddef.h>
#include <stdint.h>

#define SONIC_MOTION_ENTRY_MATCHER_NATIVE_FEATURE_WIDTH 6

typedef enum SonicMotionEntryMatcherNativeStatus {
    SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK = 0,
    SONIC_MOTION_ENTRY_MATCHER_NATIVE_INVALID_ARGUMENT = 1,
    SONIC_MOTION_ENTRY_MATCHER_NATIVE_SIZE_OVERFLOW = 2,
    SONIC_MOTION_ENTRY_MATCHER_NATIVE_NON_FINITE = 3,
    SONIC_MOTION_ENTRY_MATCHER_NATIVE_OUT_OF_RANGE = 4,
    SONIC_MOTION_ENTRY_MATCHER_NATIVE_UNSUPPORTED_FLOAT_ENVIRONMENT = 5,
    SONIC_MOTION_ENTRY_MATCHER_NATIVE_FEATURES_NOT_REGISTERED = 6,
    SONIC_MOTION_ENTRY_MATCHER_NATIVE_REGISTRY_FULL = 7,
    SONIC_MOTION_ENTRY_MATCHER_NATIVE_KINEMATICS_FAILURE = 8
} SonicMotionEntryMatcherNativeStatus;

typedef struct SonicMotionEntryMatcherNativeFootFeature {
    float left_xyz[3];
    float right_xyz[3];
} SonicMotionEntryMatcherNativeFootFeature;

/*
 * One slot corresponds to the current-build footFeatures dictionary entry for
 * one decoded NPZ clip.  The clip and feature arrays remain caller-owned.
 * Feature rows are [left.x, left.y, left.z, right.x, right.y, right.z].
 */
typedef struct SonicMotionEntryMatcherNativeFeatureSlot {
    const float* clip_dof_position_identity;
    const float* clip_root_quaternion_identity;
    size_t clip_frame_count;
    const float* root_local_foot_xyz;
    size_t root_local_foot_xyz_count;
    int registered;
} SonicMotionEntryMatcherNativeFeatureSlot;

typedef struct SonicMotionEntryMatcherNativeDiagnostics {
    float outgoing_feature_cursor;
    float transition_center_ticks;
    int32_t target_best_frame;
    float best_squared_distance;
    int valid;
} SonicMotionEntryMatcherNativeDiagnostics;

/*
 * slots points to caller-owned writable storage.  One matcher is intended per
 * composer when callbacks can execute concurrently because last_status and
 * diagnostics are updated by the callback.
 */
typedef struct SonicMotionEntryMatcherNative {
    SonicMotionEntryMatcherNativeFeatureSlot* slots;
    size_t slot_capacity;
    size_t slot_count;
    int32_t controller_rate_hz;
    SonicMotionEntryMatcherNativeStatus last_status;
    SonicMotionEntryMatcherNativeDiagnostics diagnostics;
} SonicMotionEntryMatcherNative;

/*
 * The current runner obtains each feature row by posing one decoded clip row,
 * synchronizing kinematics, and sampling both ankle-roll body positions in the
 * root body's local frame.  This callback is that explicit runtime boundary.
 */
typedef int (*SonicMotionEntryMatcherNativeKinematicsSampler)(
    void* context,
    const float dof_position_mujoco[
        SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT],
    float root_local_foot_xyz[
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_FEATURE_WIDTH]
);

const char* sonic_motion_entry_matcher_native_status_string(
    SonicMotionEntryMatcherNativeStatus status
);

SonicMotionEntryMatcherNativeStatus sonic_motion_entry_matcher_native_init(
    SonicMotionEntryMatcherNative* matcher,
    int32_t controller_rate_hz,
    SonicMotionEntryMatcherNativeFeatureSlot* slots,
    size_t slot_capacity
);

/* Exact current-build MakeFeat mirror and left/right exchange. */
SonicMotionEntryMatcherNativeStatus sonic_motion_entry_matcher_native_make_feature(
    const float raw_root_local_foot_xyz[
        SONIC_MOTION_ENTRY_MATCHER_NATIVE_FEATURE_WIDTH],
    int mirror,
    SonicMotionEntryMatcherNativeFootFeature* output
);

/* Exact clamped integer sampler used by MatchEntryCursor. */
SonicMotionEntryMatcherNativeStatus sonic_motion_entry_matcher_native_sample_at(
    const float* root_local_foot_xyz,
    size_t frame_count,
    int32_t frame,
    int mirror,
    SonicMotionEntryMatcherNativeFootFeature* output
);

/* Exact clamped binary32 linear sampler used for the outgoing feature. */
SonicMotionEntryMatcherNativeStatus sonic_motion_entry_matcher_native_sample_lerp(
    const float* root_local_foot_xyz,
    size_t frame_count,
    float cursor,
    int mirror,
    SonicMotionEntryMatcherNativeFootFeature* output
);

/*
 * Exposes the C equivalent of GetClipIlDofPos without decoding or copying.
 * Returned storage remains owned by clip.
 */
SonicMotionEntryMatcherNativeStatus
sonic_motion_entry_matcher_native_clip_dof_positions(
    const SonicMotionComposerNativeClip* clip,
    const float** dof_position_mujoco,
    size_t* frame_count,
    size_t* row_width
);

/*
 * Heap-free current-runner feature bake.  It deliberately requires an exact
 * kinematics sampler because the Unity Transform/MuJoCo runtime boundary is
 * not inferred from joint samples.  Rows written before a callback failure
 * remain written, matching the imperative current-runner bake order.
 */
SonicMotionEntryMatcherNativeStatus sonic_motion_entry_matcher_native_bake_features(
    const SonicMotionComposerNativeClip* clip,
    SonicMotionEntryMatcherNativeKinematicsSampler sampler,
    void* sampler_context,
    float* output_root_local_foot_xyz,
    size_t output_count
);

/* Current-build RegisterFootFeatures dictionary replacement semantics. */
SonicMotionEntryMatcherNativeStatus sonic_motion_entry_matcher_native_register(
    SonicMotionEntryMatcherNative* matcher,
    const SonicMotionComposerNativeClip* clip,
    const float* root_local_foot_xyz,
    size_t root_local_foot_xyz_count
);

/*
 * Exact MatchEntryCursor arithmetic for registered feature arrays.  Unlike the
 * game UI, missing features fail closed instead of selecting an authored entry
 * cursor, so an unknown kinematic dependency cannot become simulator data.
 */
SonicMotionEntryMatcherNativeStatus sonic_motion_entry_matcher_native_match(
    SonicMotionEntryMatcherNative* matcher,
    const SonicMotionComposerNativeLayer* target,
    const SonicMotionComposerNativeLayer* outgoing,
    float* matched_cursor
);

/* Drop-in SonicMotionComposerNativeLoopEntryMatcher callback. */
int sonic_motion_entry_matcher_native_callback(
    void* context,
    const SonicMotionComposerNativeLayer* target,
    const SonicMotionComposerNativeLayer* outgoing,
    float* matched_cursor
);
