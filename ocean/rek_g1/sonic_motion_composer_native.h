#pragma once

#include "g1_cuda_qualifiers.h"

#include <stddef.h>
#include <stdint.h>

#define SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT 29
#define SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS 10

typedef enum SonicMotionComposerNativeStatus {
    SONIC_MOTION_COMPOSER_NATIVE_OK = 0,
    SONIC_MOTION_COMPOSER_NATIVE_INVALID_ARGUMENT = 1,
    SONIC_MOTION_COMPOSER_NATIVE_SIZE_OVERFLOW = 2,
    SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE = 3,
    SONIC_MOTION_COMPOSER_NATIVE_OUT_OF_RANGE = 4,
    SONIC_MOTION_COMPOSER_NATIVE_UNSUPPORTED_QUATERNION_SLERP = 5,
    SONIC_MOTION_COMPOSER_NATIVE_UNSUPPORTED_ATAN2 = 6,
    SONIC_MOTION_COMPOSER_NATIVE_UNSUPPORTED_SIN_COS = 7,
    SONIC_MOTION_COMPOSER_NATIVE_UNSUPPORTED_LOOP_ENTRY_MATCHER = 8,
    SONIC_MOTION_COMPOSER_NATIVE_UNSUPPORTED_MIRROR_TABLE = 9,
    SONIC_MOTION_COMPOSER_NATIVE_UNSUPPORTED_FLOAT_ENVIRONMENT = 10,
    SONIC_MOTION_COMPOSER_NATIVE_BACKEND_FAILURE = 11
} SonicMotionComposerNativeStatus;

/*
 * The sample arrays are caller-owned and must remain readable and unchanged
 * while a layer containing this view is active. Joint rows are in MuJoCo order.
 * Root rows are WXYZ because that is the recovered composer order. frame_count
 * must be at most 16,777,216 so every accepted frame boundary is binary32 exact.
 */
typedef struct SonicMotionComposerNativeClip {
    const float* dof_position_mujoco;
    const float* root_quaternion_wxyz;
    size_t dof_position_count;
    size_t root_quaternion_count;
    size_t frame_count;
    float fps;
} SonicMotionComposerNativeClip;

/* All fields are required. Negative frame bounds follow the recovered clamps. */
typedef struct SonicMotionComposerNativeConfig {
    int mirror;
    int loop;
    float playback_speed;
    int32_t start_frame;
    int32_t end_frame;
    float blend_in_seconds;
    float blend_out_seconds;
    float yaw_blend;
} SonicMotionComposerNativeConfig;

/*
 * Tables are caller-owned. Each array must contain exactly 29 entries when a
 * mirrored active layer is sampled. negate entries must be zero or one.
 */
typedef struct SonicMotionComposerNativeMirrorTable {
    const uint32_t* source_indices;
    const uint8_t* negate;
    size_t source_index_count;
    size_t negate_count;
} SonicMotionComposerNativeMirrorTable;

typedef struct SonicMotionComposerNativeResolvedFrames {
    int32_t f0;
    int32_t f1;
    float t;
} SonicMotionComposerNativeResolvedFrames;

typedef struct SonicMotionComposerNativeLayer {
    SonicMotionComposerNativeClip clip;
    SonicMotionComposerNativeConfig config;
    float speed;
    float per_tick;
    float cursor;
    int32_t start_frame;
    int32_t end_frame;
    float prev_heading;
    float last_heading_delta;
    int has_clip;
    int has_config;
    int active;
    int heading_valid;
    int heading_resync;
} SonicMotionComposerNativeLayer;

/*
 * These callbacks are the only route to runtime operations not recovered as
 * static native code. A callback returns nonzero on success and writes one
 * finite binary32 result. No mathematical fallback is selected by this API.
 */
typedef int (*SonicMotionComposerNativeQuaternionSlerp)(
    void* context,
    const float a_wxyz[4],
    const float b_wxyz[4],
    float t,
    float output_wxyz[4]
);

typedef int (*SonicMotionComposerNativeAtan2F)(
    void* context,
    float numerator,
    float denominator,
    float* output
);

typedef int (*SonicMotionComposerNativeSinCosF)(
    void* context,
    float angle,
    float* sine,
    float* cosine
);

typedef int (*SonicMotionComposerNativeLoopEntryMatcher)(
    void* context,
    const SonicMotionComposerNativeLayer* target,
    const SonicMotionComposerNativeLayer* outgoing,
    float* matched_cursor
);

typedef struct SonicMotionComposerNativeBackends {
    SonicMotionComposerNativeQuaternionSlerp quaternion_slerp;
    SonicMotionComposerNativeAtan2F atan2_f;
    SonicMotionComposerNativeSinCosF sin_cos_f;
    SonicMotionComposerNativeLoopEntryMatcher loop_entry_matcher;
    void* context;
} SonicMotionComposerNativeBackends;

typedef struct SonicMotionComposerNative {
    int32_t controller_rate_hz;
    SonicMotionComposerNativeLayer current_layer;
    SonicMotionComposerNativeLayer from_layer;
    SonicMotionComposerNativeBackends backends;
    int32_t xt;
    int32_t w_in;
    int32_t w_out;
    int32_t w_total;
    int action_playing;
    int32_t action_move_id;
    float pending_heading_delta;
} SonicMotionComposerNative;

typedef struct SonicMotionComposerNativeLayerAdvanceResult {
    int wrapped;
    int completed;
} SonicMotionComposerNativeLayerAdvanceResult;

typedef struct SonicMotionComposerNativeAdvanceResult {
    SonicMotionComposerNativeLayerAdvanceResult current;
    SonicMotionComposerNativeLayerAdvanceResult outgoing;
    float weight_current;
} SonicMotionComposerNativeAdvanceResult;

/*
 * Both schedules are explicit controller-tick offsets from the current layer
 * cursor. root_rotation_xyzw uses current_offsets. The next-position array uses
 * next_offsets and is not inferred from current_offsets.
 */
typedef struct SonicMotionComposerNativeReferenceTiming {
    int32_t current_offsets[SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS];
    int32_t next_offsets[SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS];
} SonicMotionComposerNativeReferenceTiming;

/*
 * These pointers and shapes match the three arrays consumed by
 * GearSonicNativeReferenceInput for one fighter. The root output is XYZW.
 * Buffers must not overlap each other or active clip sample arrays. No output
 * buffer is changed when composition fails.
 */
typedef struct SonicMotionComposerNativeReferenceOutput {
    float* dof_position_mujoco;
    float* dof_next_position_mujoco;
    float* root_rotation_xyzw;
    size_t dof_position_capacity;
    size_t dof_next_position_capacity;
    size_t root_rotation_capacity;
} SonicMotionComposerNativeReferenceOutput;

REK_G1_FN const char* sonic_motion_composer_native_status_string(
    SonicMotionComposerNativeStatus status
);

/*
 * The module requires the default round-to-nearest floating-point environment
 * and compilation without fast-math transformations. Explicit volatile stores
 * preserve the binary32 operation boundaries pinned by the Python contract.
 */

REK_G1_FN SonicMotionComposerNativeStatus sonic_motion_composer_native_init(
    SonicMotionComposerNative* composer,
    int32_t controller_rate_hz,
    const SonicMotionComposerNativeBackends* backends
);

REK_G1_FN SonicMotionComposerNativeStatus sonic_motion_composer_native_install_layer(
    /* Must point to a zero-initialized or previously valid layer. */
    SonicMotionComposerNativeLayer* layer,
    const SonicMotionComposerNativeClip* clip,
    const SonicMotionComposerNativeConfig* config,
    int32_t controller_rate_hz
);

REK_G1_FN SonicMotionComposerNativeStatus sonic_motion_composer_native_resolve_frames(
    const SonicMotionComposerNativeLayer* layer,
    int32_t frames_ahead,
    SonicMotionComposerNativeResolvedFrames* result
);

REK_G1_FN SonicMotionComposerNativeStatus sonic_motion_composer_native_weight_current(
    float tt,
    int32_t w_in,
    int32_t w_out,
    float* result
);

REK_G1_FN SonicMotionComposerNativeStatus sonic_motion_composer_native_xfade_at(
    int32_t xt,
    int32_t frames_ahead,
    int32_t w_in,
    int32_t w_out,
    float* result
);

REK_G1_FN SonicMotionComposerNativeStatus sonic_motion_composer_native_play_action(
    SonicMotionComposerNative* composer,
    const SonicMotionComposerNativeClip* clip,
    const SonicMotionComposerNativeConfig* config
);

REK_G1_FN SonicMotionComposerNativeStatus sonic_motion_composer_native_play_action_immediate(
    SonicMotionComposerNative* composer,
    const SonicMotionComposerNativeClip* clip,
    const SonicMotionComposerNativeConfig* config
);

REK_G1_FN SonicMotionComposerNativeStatus sonic_motion_composer_native_cancel_action(
    SonicMotionComposerNative* composer
);

REK_G1_FN SonicMotionComposerNativeStatus sonic_motion_composer_native_set_locomotion_speed(
    SonicMotionComposerNative* composer,
    float scale
);

REK_G1_FN SonicMotionComposerNativeStatus sonic_motion_composer_native_advance(
    SonicMotionComposerNative* composer,
    SonicMotionComposerNativeAdvanceResult* result
);

REK_G1_FN SonicMotionComposerNativeStatus sonic_motion_composer_native_consume_heading_delta(
    SonicMotionComposerNative* composer,
    float* result
);

REK_G1_FN SonicMotionComposerNativeStatus sonic_motion_composer_native_heading_clip_ownership(
    const SonicMotionComposerNative* composer,
    float* result
);

REK_G1_FN SonicMotionComposerNativeStatus sonic_motion_composer_native_build_reference_rows(
    const SonicMotionComposerNative* composer,
    const SonicMotionComposerNativeReferenceTiming* timing,
    const SonicMotionComposerNativeMirrorTable* mirror_table,
    SonicMotionComposerNativeReferenceOutput* output
);
