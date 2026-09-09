#include "sonic_motion_entry_matcher_native.h"

#include <float.h>
#include <fenv.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

_Static_assert(sizeof(float) == 4, "binary32 float is required");
_Static_assert(FLT_RADIX == 2, "binary floating point is required");
_Static_assert(FLT_MANT_DIG == 24, "binary32 precision is required");
_Static_assert(FLT_MAX_EXP == 128, "binary32 exponent range is required");

enum {
    SONIC_MOTION_ENTRY_MATCHER_NATIVE_MAX_EXACT_FRAME_COUNT = 16777216
};

static float f32_add(float a, float b) {
    volatile float result = a + b;
    return result;
}

static float f32_sub(float a, float b) {
    volatile float result = a - b;
    return result;
}

static float f32_mul(float a, float b) {
    volatile float result = a * b;
    return result;
}

static float f32_div(float a, float b) {
    volatile float result = a / b;
    return result;
}

static float f32_from_i32(int32_t value) {
    volatile float result = (float)value;
    return result;
}

static int valid_flag(int value) {
    return value == 0 || value == 1;
}

static int checked_product(size_t a, size_t b, size_t* result) {
    if (result == NULL || (a != 0 && b > SIZE_MAX / a)) {
        return 0;
    }
    *result = a * b;
    return 1;
}

static int finite_floats(const float* values, size_t count) {
    if (values == NULL) {
        return 0;
    }
    for (size_t index = 0; index < count; index++) {
        if (!isfinite(values[index])) {
            return 0;
        }
    }
    return 1;
}

static SonicMotionEntryMatcherNativeStatus validate_float_environment(void) {
    return fegetround() == FE_TONEAREST
        ? SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK
        : SONIC_MOTION_ENTRY_MATCHER_NATIVE_UNSUPPORTED_FLOAT_ENVIRONMENT;
}

static SonicMotionEntryMatcherNativeStatus validate_clip_identity(
        const SonicMotionComposerNativeClip* clip,
        int validate_dof_values) {
    size_t expected_dof_count = 0;
    size_t expected_root_count = 0;
    if (clip == NULL || clip->dof_position_mujoco == NULL
            || clip->root_quaternion_wxyz == NULL || clip->frame_count == 0) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_INVALID_ARGUMENT;
    }
    if (clip->frame_count
            > (size_t)SONIC_MOTION_ENTRY_MATCHER_NATIVE_MAX_EXACT_FRAME_COUNT) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_OUT_OF_RANGE;
    }
    if (!isfinite(clip->fps) || clip->fps <= 0.0f) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_NON_FINITE;
    }
    if (!checked_product(
            clip->frame_count,
            SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT,
            &expected_dof_count)
            || !checked_product(clip->frame_count, 4, &expected_root_count)) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_SIZE_OVERFLOW;
    }
    if (clip->dof_position_count != expected_dof_count
            || clip->root_quaternion_count != expected_root_count) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_INVALID_ARGUMENT;
    }
    if (validate_dof_values
            && !finite_floats(clip->dof_position_mujoco, expected_dof_count)) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_NON_FINITE;
    }
    return SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK;
}

static SonicMotionEntryMatcherNativeStatus validate_layer(
        const SonicMotionComposerNativeLayer* layer,
        int require_active) {
    SonicMotionEntryMatcherNativeStatus status;
    if (layer == NULL || !valid_flag(layer->active)
            || !valid_flag(layer->has_clip) || !valid_flag(layer->has_config)
            || !valid_flag(layer->config.mirror)
            || !valid_flag(layer->config.loop)) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_INVALID_ARGUMENT;
    }
    if ((require_active && !layer->active) || !layer->has_clip
            || !layer->has_config) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_INVALID_ARGUMENT;
    }
    status = validate_clip_identity(&layer->clip, 0);
    if (status != SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK) {
        return status;
    }
    if (layer->start_frame < 0 || layer->end_frame < layer->start_frame
            || (size_t)layer->end_frame >= layer->clip.frame_count) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_OUT_OF_RANGE;
    }
    if (!isfinite(layer->cursor) || !isfinite(layer->per_tick)
            || !isfinite(layer->config.blend_in_seconds)
            || !isfinite(layer->config.blend_out_seconds)) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_NON_FINITE;
    }
    return SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK;
}

static SonicMotionEntryMatcherNativeStatus blend_frames(
        int32_t controller_rate_hz,
        float seconds,
        int32_t* result) {
    float product;
    double lower;
    double fraction;
    double rounded;
    if (result == NULL || controller_rate_hz <= 0) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_INVALID_ARGUMENT;
    }
    if (!isfinite(seconds)) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_NON_FINITE;
    }
    product = f32_mul(f32_from_i32(controller_rate_hz), seconds);
    if (!isfinite(product)) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_OUT_OF_RANGE;
    }
    lower = floor((double)product);
    fraction = (double)product - lower;
    if (fraction < 0.5) {
        rounded = lower;
    } else if (fraction > 0.5) {
        rounded = lower + 1.0;
    } else if (fmod(lower, 2.0) == 0.0) {
        rounded = lower;
    } else {
        rounded = lower + 1.0;
    }
    if (rounded < (double)INT32_MIN || rounded > (double)INT32_MAX) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_OUT_OF_RANGE;
    }
    *result = rounded < 1.0 ? 1 : (int32_t)rounded;
    return SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK;
}

static int same_clip_identity(
        const SonicMotionEntryMatcherNativeFeatureSlot* slot,
        const SonicMotionComposerNativeClip* clip) {
    return slot->registered
        && slot->clip_dof_position_identity == clip->dof_position_mujoco
        && slot->clip_root_quaternion_identity == clip->root_quaternion_wxyz
        && slot->clip_frame_count == clip->frame_count;
}

static SonicMotionEntryMatcherNativeFeatureSlot* find_slot(
        SonicMotionEntryMatcherNative* matcher,
        const SonicMotionComposerNativeClip* clip) {
    for (size_t index = 0; index < matcher->slot_count; index++) {
        if (same_clip_identity(&matcher->slots[index], clip)) {
            return &matcher->slots[index];
        }
    }
    return NULL;
}

static SonicMotionEntryMatcherNativeStatus wrap_cursor(
        const SonicMotionComposerNativeLayer* layer,
        float cursor,
        float* result) {
    int32_t span;
    float start;
    float end;
    if (layer == NULL || result == NULL) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_INVALID_ARGUMENT;
    }
    if (!isfinite(cursor)) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_NON_FINITE;
    }
    start = f32_from_i32(layer->start_frame);
    end = f32_from_i32(layer->end_frame);
    if (!layer->config.loop) {
        if (cursor < start) {
            *result = start;
        } else if (cursor > end) {
            *result = end;
        } else {
            *result = cursor;
        }
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK;
    }
    if ((int64_t)layer->end_frame - (int64_t)layer->start_frame + 1
            > INT32_MAX) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_OUT_OF_RANGE;
    }
    span = layer->end_frame - layer->start_frame + 1;
    if (span <= 0) {
        *result = start;
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK;
    }
    {
        float span_f = f32_from_i32(span);
        float delta = f32_sub(cursor, start);
        float remainder = (float)fmod((double)delta, (double)span_f);
        if (!isfinite(remainder)) {
            return SONIC_MOTION_ENTRY_MATCHER_NATIVE_NON_FINITE;
        }
        if (remainder < 0.0f) {
            remainder = f32_add(remainder, span_f);
        }
        *result = f32_add(start, remainder);
    }
    return SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK;
}

static SonicMotionEntryMatcherNativeStatus squared_distance(
        const SonicMotionEntryMatcherNativeFootFeature* a,
        const SonicMotionEntryMatcherNativeFootFeature* b,
        float* result) {
    float d_lfx;
    float d_lfy;
    float d_lfz;
    float d_rfx;
    float d_rfy;
    float d_rfz;
    float left;
    float right;
    if (a == NULL || b == NULL || result == NULL) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_INVALID_ARGUMENT;
    }
    d_lfx = f32_sub(a->left_xyz[0], b->left_xyz[0]);
    d_lfy = f32_sub(a->left_xyz[1], b->left_xyz[1]);
    d_lfz = f32_sub(a->left_xyz[2], b->left_xyz[2]);
    d_rfx = f32_sub(a->right_xyz[0], b->right_xyz[0]);
    d_rfy = f32_sub(a->right_xyz[1], b->right_xyz[1]);
    d_rfz = f32_sub(a->right_xyz[2], b->right_xyz[2]);

    left = f32_add(f32_mul(d_lfy, d_lfy), f32_mul(d_lfx, d_lfx));
    left = f32_add(left, f32_mul(d_lfz, d_lfz));
    right = f32_add(f32_mul(d_rfy, d_rfy), f32_mul(d_rfx, d_rfx));
    right = f32_add(right, f32_mul(d_rfz, d_rfz));
    *result = f32_add(right, left);
    return SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK;
}

const char* sonic_motion_entry_matcher_native_status_string(
        SonicMotionEntryMatcherNativeStatus status) {
    switch (status) {
        case SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK:
            return "ok";
        case SONIC_MOTION_ENTRY_MATCHER_NATIVE_INVALID_ARGUMENT:
            return "invalid argument";
        case SONIC_MOTION_ENTRY_MATCHER_NATIVE_SIZE_OVERFLOW:
            return "size overflow";
        case SONIC_MOTION_ENTRY_MATCHER_NATIVE_NON_FINITE:
            return "non-finite value";
        case SONIC_MOTION_ENTRY_MATCHER_NATIVE_OUT_OF_RANGE:
            return "out of range";
        case SONIC_MOTION_ENTRY_MATCHER_NATIVE_UNSUPPORTED_FLOAT_ENVIRONMENT:
            return "round-to-nearest floating-point environment is required";
        case SONIC_MOTION_ENTRY_MATCHER_NATIVE_FEATURES_NOT_REGISTERED:
            return "clip foot features are not registered";
        case SONIC_MOTION_ENTRY_MATCHER_NATIVE_REGISTRY_FULL:
            return "feature registry is full";
        case SONIC_MOTION_ENTRY_MATCHER_NATIVE_KINEMATICS_FAILURE:
            return "kinematics sampler failed";
        default:
            return "unknown status";
    }
}

SonicMotionEntryMatcherNativeStatus sonic_motion_entry_matcher_native_init(
        SonicMotionEntryMatcherNative* matcher,
        int32_t controller_rate_hz,
        SonicMotionEntryMatcherNativeFeatureSlot* slots,
        size_t slot_capacity) {
    SonicMotionEntryMatcherNativeStatus status = validate_float_environment();
    if (status != SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK) {
        return status;
    }
    if (matcher == NULL || controller_rate_hz <= 0
            || (slot_capacity != 0 && slots == NULL)) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_INVALID_ARGUMENT;
    }
    if (slot_capacity > SIZE_MAX / sizeof(*slots)) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_SIZE_OVERFLOW;
    }
    if (slot_capacity != 0) {
        memset(slots, 0, slot_capacity * sizeof(*slots));
    }
    memset(matcher, 0, sizeof(*matcher));
    matcher->slots = slots;
    matcher->slot_capacity = slot_capacity;
    matcher->controller_rate_hz = controller_rate_hz;
    matcher->last_status = SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK;
    return SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK;
}

SonicMotionEntryMatcherNativeStatus sonic_motion_entry_matcher_native_make_feature(
        const float raw[SONIC_MOTION_ENTRY_MATCHER_NATIVE_FEATURE_WIDTH],
        int mirror,
        SonicMotionEntryMatcherNativeFootFeature* output) {
    SonicMotionEntryMatcherNativeFootFeature result;
    if (raw == NULL || output == NULL || !valid_flag(mirror)) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_INVALID_ARGUMENT;
    }
    if (!finite_floats(raw, SONIC_MOTION_ENTRY_MATCHER_NATIVE_FEATURE_WIDTH)) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_NON_FINITE;
    }
    if (mirror) {
        result.left_xyz[0] = -raw[3];
        result.left_xyz[1] = raw[4];
        result.left_xyz[2] = raw[5];
        result.right_xyz[0] = -raw[0];
        result.right_xyz[1] = raw[1];
        result.right_xyz[2] = raw[2];
    } else {
        memcpy(result.left_xyz, raw, 3 * sizeof(float));
        memcpy(result.right_xyz, raw + 3, 3 * sizeof(float));
    }
    *output = result;
    return SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK;
}

SonicMotionEntryMatcherNativeStatus sonic_motion_entry_matcher_native_sample_at(
        const float* features,
        size_t frame_count,
        int32_t frame,
        int mirror,
        SonicMotionEntryMatcherNativeFootFeature* output) {
    size_t index;
    if (features == NULL || output == NULL || frame_count == 0
            || frame_count
                > (size_t)SONIC_MOTION_ENTRY_MATCHER_NATIVE_MAX_EXACT_FRAME_COUNT
            || !valid_flag(mirror)) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_INVALID_ARGUMENT;
    }
    if (frame < 0) {
        index = 0;
    } else if ((size_t)frame >= frame_count) {
        index = frame_count - 1;
    } else {
        index = (size_t)frame;
    }
    return sonic_motion_entry_matcher_native_make_feature(
        features + index * SONIC_MOTION_ENTRY_MATCHER_NATIVE_FEATURE_WIDTH,
        mirror,
        output);
}

SonicMotionEntryMatcherNativeStatus sonic_motion_entry_matcher_native_sample_lerp(
        const float* features,
        size_t frame_count,
        float cursor,
        int mirror,
        SonicMotionEntryMatcherNativeFootFeature* output) {
    SonicMotionEntryMatcherNativeFootFeature a;
    SonicMotionEntryMatcherNativeFootFeature b;
    double floored;
    int32_t f0;
    int32_t f1;
    float t;
    SonicMotionEntryMatcherNativeStatus status;
    if (features == NULL || output == NULL || frame_count == 0
            || frame_count
                > (size_t)SONIC_MOTION_ENTRY_MATCHER_NATIVE_MAX_EXACT_FRAME_COUNT
            || !valid_flag(mirror)) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_INVALID_ARGUMENT;
    }
    if (!isfinite(cursor)) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_NON_FINITE;
    }
    floored = floor((double)cursor);
    if (floored < (double)INT32_MIN || floored > (double)INT32_MAX) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_OUT_OF_RANGE;
    }
    f0 = (int32_t)floored;
    if (f0 < 0) {
        f0 = 0;
    } else if ((size_t)f0 >= frame_count) {
        f0 = (int32_t)(frame_count - 1);
    }
    f1 = (size_t)f0 + 1 < frame_count ? f0 + 1 : f0;
    t = f32_sub(cursor, f32_from_i32((int32_t)floored));
    if (t < 0.0f) {
        t = 0.0f;
    } else if (t > 1.0f) {
        t = 1.0f;
    }
    status = sonic_motion_entry_matcher_native_sample_at(
        features, frame_count, f0, mirror, &a);
    if (status != SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK) {
        return status;
    }
    status = sonic_motion_entry_matcher_native_sample_at(
        features, frame_count, f1, mirror, &b);
    if (status != SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK) {
        return status;
    }
    for (size_t axis = 0; axis < 3; axis++) {
        output->left_xyz[axis] = f32_add(
            a.left_xyz[axis],
            f32_mul(f32_sub(b.left_xyz[axis], a.left_xyz[axis]), t));
        output->right_xyz[axis] = f32_add(
            a.right_xyz[axis],
            f32_mul(f32_sub(b.right_xyz[axis], a.right_xyz[axis]), t));
    }
    return SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK;
}

SonicMotionEntryMatcherNativeStatus
sonic_motion_entry_matcher_native_clip_dof_positions(
        const SonicMotionComposerNativeClip* clip,
        const float** dof_position_mujoco,
        size_t* frame_count,
        size_t* row_width) {
    SonicMotionEntryMatcherNativeStatus status;
    if (dof_position_mujoco == NULL || frame_count == NULL
            || row_width == NULL) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_INVALID_ARGUMENT;
    }
    status = validate_clip_identity(clip, 0);
    if (status != SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK) {
        return status;
    }
    *dof_position_mujoco = clip->dof_position_mujoco;
    *frame_count = clip->frame_count;
    *row_width = SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT;
    return SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK;
}

SonicMotionEntryMatcherNativeStatus sonic_motion_entry_matcher_native_bake_features(
        const SonicMotionComposerNativeClip* clip,
        SonicMotionEntryMatcherNativeKinematicsSampler sampler,
        void* sampler_context,
        float* output,
        size_t output_count) {
    size_t expected_count = 0;
    SonicMotionEntryMatcherNativeStatus status;
    status = validate_float_environment();
    if (status != SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK) {
        return status;
    }
    status = validate_clip_identity(clip, 1);
    if (status != SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK) {
        return status;
    }
    if (sampler == NULL || output == NULL) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_INVALID_ARGUMENT;
    }
    if (!checked_product(
            clip->frame_count,
            SONIC_MOTION_ENTRY_MATCHER_NATIVE_FEATURE_WIDTH,
            &expected_count)) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_SIZE_OVERFLOW;
    }
    if (output_count != expected_count) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_INVALID_ARGUMENT;
    }
    for (size_t frame = 0; frame < clip->frame_count; frame++) {
        float feature[SONIC_MOTION_ENTRY_MATCHER_NATIVE_FEATURE_WIDTH];
        const float* dof = clip->dof_position_mujoco
            + frame * SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT;
        if (!sampler(sampler_context, dof, feature)) {
            return SONIC_MOTION_ENTRY_MATCHER_NATIVE_KINEMATICS_FAILURE;
        }
        if (!finite_floats(
                feature,
                SONIC_MOTION_ENTRY_MATCHER_NATIVE_FEATURE_WIDTH)) {
            return SONIC_MOTION_ENTRY_MATCHER_NATIVE_NON_FINITE;
        }
        memcpy(
            output + frame * SONIC_MOTION_ENTRY_MATCHER_NATIVE_FEATURE_WIDTH,
            feature,
            sizeof(feature));
    }
    return SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK;
}

SonicMotionEntryMatcherNativeStatus sonic_motion_entry_matcher_native_register(
        SonicMotionEntryMatcherNative* matcher,
        const SonicMotionComposerNativeClip* clip,
        const float* features,
        size_t feature_count) {
    size_t expected_count = 0;
    SonicMotionEntryMatcherNativeFeatureSlot* slot;
    SonicMotionEntryMatcherNativeStatus status = validate_float_environment();
    if (status != SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK) {
        return status;
    }
    if (matcher == NULL || matcher->controller_rate_hz <= 0
            || matcher->slot_count > matcher->slot_capacity
            || (matcher->slot_capacity != 0 && matcher->slots == NULL)) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_INVALID_ARGUMENT;
    }
    status = validate_clip_identity(clip, 0);
    if (status != SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK) {
        return status;
    }
    if (!checked_product(
            clip->frame_count,
            SONIC_MOTION_ENTRY_MATCHER_NATIVE_FEATURE_WIDTH,
            &expected_count)) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_SIZE_OVERFLOW;
    }
    if (features == NULL || feature_count != expected_count) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_INVALID_ARGUMENT;
    }
    if (!finite_floats(features, feature_count)) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_NON_FINITE;
    }
    slot = find_slot(matcher, clip);
    if (slot == NULL) {
        if (matcher->slot_count >= matcher->slot_capacity) {
            return SONIC_MOTION_ENTRY_MATCHER_NATIVE_REGISTRY_FULL;
        }
        slot = &matcher->slots[matcher->slot_count++];
    }
    slot->clip_dof_position_identity = clip->dof_position_mujoco;
    slot->clip_root_quaternion_identity = clip->root_quaternion_wxyz;
    slot->clip_frame_count = clip->frame_count;
    slot->root_local_foot_xyz = features;
    slot->root_local_foot_xyz_count = feature_count;
    slot->registered = 1;
    return SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK;
}

SonicMotionEntryMatcherNativeStatus sonic_motion_entry_matcher_native_match(
        SonicMotionEntryMatcherNative* matcher,
        const SonicMotionComposerNativeLayer* target,
        const SonicMotionComposerNativeLayer* outgoing,
        float* matched_cursor) {
    SonicMotionEntryMatcherNativeFeatureSlot* target_slot;
    SonicMotionEntryMatcherNativeFeatureSlot* outgoing_slot;
    SonicMotionEntryMatcherNativeFootFeature outgoing_feature;
    int32_t w_in;
    int32_t w_out;
    int32_t width_sum;
    float transition_center;
    float outgoing_cursor;
    float best_distance = FLT_MAX;
    int32_t best_frame;
    float entry;
    SonicMotionEntryMatcherNativeStatus status;

    if (matcher == NULL || matched_cursor == NULL
            || matcher->controller_rate_hz <= 0
            || matcher->slot_count > matcher->slot_capacity
            || (matcher->slot_capacity != 0 && matcher->slots == NULL)) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_INVALID_ARGUMENT;
    }
    memset(&matcher->diagnostics, 0, sizeof(matcher->diagnostics));
    status = validate_float_environment();
    if (status != SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK) {
        return status;
    }
    status = validate_layer(target, 1);
    if (status != SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK) {
        return status;
    }
    status = validate_layer(outgoing, 1);
    if (status != SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK) {
        return status;
    }
    target_slot = find_slot(matcher, &target->clip);
    outgoing_slot = find_slot(matcher, &outgoing->clip);
    if (target_slot == NULL || outgoing_slot == NULL) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_FEATURES_NOT_REGISTERED;
    }
    if (target_slot->root_local_foot_xyz_count
                != target_slot->clip_frame_count
                    * SONIC_MOTION_ENTRY_MATCHER_NATIVE_FEATURE_WIDTH
            || outgoing_slot->root_local_foot_xyz_count
                != outgoing_slot->clip_frame_count
                    * SONIC_MOTION_ENTRY_MATCHER_NATIVE_FEATURE_WIDTH) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_INVALID_ARGUMENT;
    }
    status = blend_frames(
        matcher->controller_rate_hz,
        target->config.blend_in_seconds,
        &w_in);
    if (status != SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK) {
        return status;
    }
    status = blend_frames(
        matcher->controller_rate_hz,
        outgoing->config.blend_out_seconds,
        &w_out);
    if (status != SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK) {
        return status;
    }
    if (w_in > INT32_MAX - w_out) {
        return SONIC_MOTION_ENTRY_MATCHER_NATIVE_OUT_OF_RANGE;
    }
    width_sum = w_out + w_in;
    transition_center = f32_div(
        f32_mul(f32_from_i32(w_out), f32_from_i32(w_in)),
        f32_from_i32(width_sum));
    outgoing_cursor = f32_add(
        outgoing->cursor,
        f32_mul(outgoing->per_tick, transition_center));
    status = wrap_cursor(outgoing, outgoing_cursor, &outgoing_cursor);
    if (status != SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK) {
        return status;
    }
    status = sonic_motion_entry_matcher_native_sample_lerp(
        outgoing_slot->root_local_foot_xyz,
        outgoing_slot->clip_frame_count,
        outgoing_cursor,
        outgoing->config.mirror,
        &outgoing_feature);
    if (status != SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK) {
        return status;
    }

    best_frame = target->start_frame;
    for (int32_t frame = target->start_frame; ; frame++) {
        SonicMotionEntryMatcherNativeFootFeature candidate;
        float distance;
        status = sonic_motion_entry_matcher_native_sample_at(
            target_slot->root_local_foot_xyz,
            target_slot->clip_frame_count,
            frame,
            target->config.mirror,
            &candidate);
        if (status != SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK) {
            return status;
        }
        status = squared_distance(&candidate, &outgoing_feature, &distance);
        if (status != SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK) {
            return status;
        }
        if (best_distance > distance) {
            best_distance = distance;
            best_frame = frame;
        }
        if (frame == target->end_frame) {
            break;
        }
    }

    entry = f32_sub(
        f32_from_i32(best_frame),
        f32_mul(transition_center, target->per_tick));
    status = wrap_cursor(target, entry, &entry);
    if (status != SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK) {
        return status;
    }
    matcher->diagnostics.outgoing_feature_cursor = outgoing_cursor;
    matcher->diagnostics.transition_center_ticks = transition_center;
    matcher->diagnostics.target_best_frame = best_frame;
    matcher->diagnostics.best_squared_distance = best_distance;
    matcher->diagnostics.valid = 1;
    *matched_cursor = entry;
    return SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK;
}

int sonic_motion_entry_matcher_native_callback(
        void* context,
        const SonicMotionComposerNativeLayer* target,
        const SonicMotionComposerNativeLayer* outgoing,
        float* matched_cursor) {
    SonicMotionEntryMatcherNative* matcher
        = (SonicMotionEntryMatcherNative*)context;
    if (matcher == NULL) {
        return 0;
    }
    matcher->last_status = sonic_motion_entry_matcher_native_match(
        matcher, target, outgoing, matched_cursor);
    return matcher->last_status == SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK;
}
