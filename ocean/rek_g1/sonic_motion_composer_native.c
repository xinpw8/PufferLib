#include "sonic_motion_composer_native.h"

#include <float.h>
#include <fenv.h>
#include <limits.h>
#include <math.h>
#include <string.h>

_Static_assert(sizeof(float) == 4, "binary32 float is required");
_Static_assert(FLT_RADIX == 2, "binary floating point is required");
_Static_assert(FLT_MANT_DIG == 24, "binary32 precision is required");
_Static_assert(FLT_MAX_EXP == 128, "binary32 exponent range is required");

enum {
    SONIC_MOTION_COMPOSER_NATIVE_MAX_EXACT_FRAME_COUNT = 16777216
};

static const float MIN_ABS_PLAYBACK_SPEED = 0.01f;
static const float MIN_LOCOMOTION_SCALE = 0.05f;
static const float HALF = 0.5f;
static const float PI_F32 = 3.1415927f;
static const float NEG_PI_F32 = -3.1415927f;
static const float TWO_PI_F32 = 6.2831855f;
static const float NEG_TWO_PI_F32 = -6.2831855f;

typedef struct SonicMotionComposerNativePose {
    float joint_positions[SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT];
    float root_quaternion_wxyz[4];
} SonicMotionComposerNativePose;

typedef struct SonicMotionComposerNativeRootHeading {
    float heading;
    int seam;
} SonicMotionComposerNativeRootHeading;

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

static int valid_flag(int value) {
    return value == 0 || value == 1;
}

static SonicMotionComposerNativeStatus validate_float_environment(void) {
    return fegetround() == FE_TONEAREST
        ? SONIC_MOTION_COMPOSER_NATIVE_OK
        : SONIC_MOTION_COMPOSER_NATIVE_UNSUPPORTED_FLOAT_ENVIRONMENT;
}

static int32_t i32_add(int32_t a, int32_t b) {
    uint32_t sum = (uint32_t)a + (uint32_t)b;
    if (sum <= (uint32_t)INT32_MAX) {
        return (int32_t)sum;
    }
    return (int32_t)((int64_t)sum - INT64_C(4294967296));
}

static SonicMotionComposerNativeStatus clamp01(float value, float* result) {
    if (result == NULL) {
        return SONIC_MOTION_COMPOSER_NATIVE_INVALID_ARGUMENT;
    }
    if (!isfinite(value)) {
        return SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE;
    }
    if (value < 0.0f) {
        *result = 0.0f;
    } else if (value > 1.0f) {
        *result = 1.0f;
    } else {
        *result = value;
    }
    return SONIC_MOTION_COMPOSER_NATIVE_OK;
}

static SonicMotionComposerNativeStatus lerp_f32(
        float a,
        float b,
        float t,
        float* result) {
    float clamped = 0.0f;
    SonicMotionComposerNativeStatus status = clamp01(t, &clamped);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    if (!isfinite(a) || !isfinite(b)) {
        return SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE;
    }
    *result = f32_add(a, f32_mul(f32_sub(b, a), clamped));
    if (!isfinite(*result)) {
        return SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE;
    }
    return SONIC_MOTION_COMPOSER_NATIVE_OK;
}

static SonicMotionComposerNativeStatus validate_clip_shape(
        const SonicMotionComposerNativeClip* clip,
        int validate_values) {
    size_t expected_dofs = 0;
    size_t expected_roots = 0;
    if (clip == NULL || clip->dof_position_mujoco == NULL
            || clip->root_quaternion_wxyz == NULL || clip->frame_count == 0) {
        return SONIC_MOTION_COMPOSER_NATIVE_INVALID_ARGUMENT;
    }
    if (clip->frame_count
            > (size_t)SONIC_MOTION_COMPOSER_NATIVE_MAX_EXACT_FRAME_COUNT) {
        return SONIC_MOTION_COMPOSER_NATIVE_OUT_OF_RANGE;
    }
    if (!isfinite(clip->fps) || clip->fps <= 0.0f) {
        return SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE;
    }
    if (!checked_product(
            clip->frame_count,
            SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT,
            &expected_dofs)
            || !checked_product(clip->frame_count, 4, &expected_roots)) {
        return SONIC_MOTION_COMPOSER_NATIVE_SIZE_OVERFLOW;
    }
    if (clip->dof_position_count != expected_dofs
            || clip->root_quaternion_count != expected_roots) {
        return SONIC_MOTION_COMPOSER_NATIVE_INVALID_ARGUMENT;
    }
    if (validate_values
            && (!finite_floats(clip->dof_position_mujoco, expected_dofs)
                || !finite_floats(
                    clip->root_quaternion_wxyz,
                    expected_roots))) {
        return SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE;
    }
    return SONIC_MOTION_COMPOSER_NATIVE_OK;
}

static SonicMotionComposerNativeStatus validate_config(
        const SonicMotionComposerNativeConfig* config) {
    if (config == NULL || !valid_flag(config->mirror)
            || !valid_flag(config->loop)) {
        return SONIC_MOTION_COMPOSER_NATIVE_INVALID_ARGUMENT;
    }
    if (!isfinite(config->playback_speed)
            || !isfinite(config->blend_in_seconds)
            || !isfinite(config->blend_out_seconds)
            || !isfinite(config->yaw_blend)) {
        return SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE;
    }
    return SONIC_MOTION_COMPOSER_NATIVE_OK;
}

static SonicMotionComposerNativeStatus validate_active_layer(
        const SonicMotionComposerNativeLayer* layer) {
    SonicMotionComposerNativeStatus status;
    if (layer == NULL || !valid_flag(layer->has_clip)
            || !valid_flag(layer->has_config) || !valid_flag(layer->active)
            || !valid_flag(layer->heading_valid)
            || !valid_flag(layer->heading_resync)) {
        return SONIC_MOTION_COMPOSER_NATIVE_INVALID_ARGUMENT;
    }
    if (!layer->active) {
        return SONIC_MOTION_COMPOSER_NATIVE_OK;
    }
    if (!layer->has_clip || !layer->has_config) {
        return SONIC_MOTION_COMPOSER_NATIVE_INVALID_ARGUMENT;
    }
    status = validate_clip_shape(&layer->clip, 0);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    status = validate_config(&layer->config);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    if (layer->start_frame < 0 || layer->end_frame < layer->start_frame
            || (size_t)layer->end_frame >= layer->clip.frame_count) {
        return SONIC_MOTION_COMPOSER_NATIVE_OUT_OF_RANGE;
    }
    if (!isfinite(layer->speed) || !isfinite(layer->per_tick)
            || !isfinite(layer->cursor) || !isfinite(layer->prev_heading)
            || !isfinite(layer->last_heading_delta)) {
        return SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE;
    }
    return SONIC_MOTION_COMPOSER_NATIVE_OK;
}

static SonicMotionComposerNativeStatus validate_resolvable_layer(
        const SonicMotionComposerNativeLayer* layer) {
    SonicMotionComposerNativeStatus status;
    if (layer == NULL || !valid_flag(layer->has_clip)
            || !valid_flag(layer->has_config) || !valid_flag(layer->active)
            || !valid_flag(layer->heading_valid)
            || !valid_flag(layer->heading_resync)
            || !layer->has_clip || !layer->has_config) {
        return SONIC_MOTION_COMPOSER_NATIVE_INVALID_ARGUMENT;
    }
    status = validate_clip_shape(&layer->clip, 0);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    status = validate_config(&layer->config);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    if (layer->start_frame < 0 || layer->end_frame < layer->start_frame
            || (size_t)layer->end_frame >= layer->clip.frame_count) {
        return SONIC_MOTION_COMPOSER_NATIVE_OUT_OF_RANGE;
    }
    if (!isfinite(layer->speed) || !isfinite(layer->per_tick)
            || !isfinite(layer->cursor) || !isfinite(layer->prev_heading)
            || !isfinite(layer->last_heading_delta)) {
        return SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE;
    }
    return SONIC_MOTION_COMPOSER_NATIVE_OK;
}

static SonicMotionComposerNativeStatus validate_composer(
        const SonicMotionComposerNative* composer) {
    SonicMotionComposerNativeStatus status;
    status = validate_float_environment();
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    if (composer == NULL || composer->controller_rate_hz <= 0
            || composer->w_in <= 0 || composer->w_out <= 0
            || composer->w_total <= 0 || !valid_flag(composer->action_playing)) {
        return SONIC_MOTION_COMPOSER_NATIVE_INVALID_ARGUMENT;
    }
    if (!isfinite(composer->pending_heading_delta)) {
        return SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE;
    }
    status = validate_active_layer(&composer->current_layer);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    return validate_active_layer(&composer->from_layer);
}

static SonicMotionComposerNativeStatus sanitize_playback_speed(
        float speed,
        float* result) {
    float magnitude;
    if (result == NULL) {
        return SONIC_MOTION_COMPOSER_NATIVE_INVALID_ARGUMENT;
    }
    if (!isfinite(speed)) {
        return SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE;
    }
    magnitude = fabsf(speed);
    if (magnitude < MIN_ABS_PLAYBACK_SPEED) {
        magnitude = MIN_ABS_PLAYBACK_SPEED;
    }
    *result = speed < 0.0f ? -magnitude : magnitude;
    return SONIC_MOTION_COMPOSER_NATIVE_OK;
}

static float entry_cursor(const SonicMotionComposerNativeLayer* layer) {
    return layer->per_tick >= 0.0f
        ? (float)layer->start_frame
        : (float)layer->end_frame;
}

static SonicMotionComposerNativeStatus wrap_loop_cursor(
        SonicMotionComposerNativeLayer* layer,
        int* wrapped) {
    int32_t span;
    float span_f;
    float upper_exclusive;
    if (layer == NULL || wrapped == NULL) {
        return SONIC_MOTION_COMPOSER_NATIVE_INVALID_ARGUMENT;
    }
    span = layer->end_frame - layer->start_frame + 1;
    if (span <= 0) {
        *wrapped = 0;
        return SONIC_MOTION_COMPOSER_NATIVE_OK;
    }
    *wrapped = 0;
    span_f = (float)span;
    upper_exclusive = (float)(layer->end_frame + 1);
    while (layer->cursor >= upper_exclusive) {
        float next = f32_sub(layer->cursor, span_f);
        if (next == layer->cursor) {
            return SONIC_MOTION_COMPOSER_NATIVE_OUT_OF_RANGE;
        }
        layer->cursor = next;
        *wrapped = 1;
    }
    while (layer->cursor < (float)layer->start_frame) {
        float next = f32_add(layer->cursor, span_f);
        if (next == layer->cursor) {
            return SONIC_MOTION_COMPOSER_NATIVE_OUT_OF_RANGE;
        }
        layer->cursor = next;
        *wrapped = 1;
    }
    return SONIC_MOTION_COMPOSER_NATIVE_OK;
}

static SonicMotionComposerNativeStatus wrapped_cursor(
        const SonicMotionComposerNativeLayer* layer,
        float cursor,
        float* result) {
    int32_t span;
    float span_f;
    float delta;
    float remainder;
    if (layer == NULL || result == NULL) {
        return SONIC_MOTION_COMPOSER_NATIVE_INVALID_ARGUMENT;
    }
    span = layer->end_frame - layer->start_frame + 1;
    if (span <= 0) {
        *result = (float)layer->start_frame;
        return SONIC_MOTION_COMPOSER_NATIVE_OK;
    }
    span_f = (float)span;
    delta = f32_sub(cursor, (float)layer->start_frame);
    remainder = (float)fmod((double)delta, (double)span_f);
    if (!isfinite(remainder)) {
        return SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE;
    }
    if (remainder < 0.0f) {
        remainder = f32_add(remainder, span_f);
    }
    *result = f32_add((float)layer->start_frame, remainder);
    return SONIC_MOTION_COMPOSER_NATIVE_OK;
}

static SonicMotionComposerNativeStatus resolve_frames_impl(
        const SonicMotionComposerNativeLayer* layer,
        int32_t frames_ahead,
        SonicMotionComposerNativeResolvedFrames* result) {
    float ahead = f32_mul((float)frames_ahead, layer->per_tick);
    float cursor = f32_add(layer->cursor, ahead);
    int32_t f0;
    int32_t f1;
    SonicMotionComposerNativeStatus status;
    if (!isfinite(cursor)) {
        return SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE;
    }
    if (layer->config.loop) {
        status = wrapped_cursor(layer, cursor, &cursor);
        if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
            return status;
        }
        f0 = (int32_t)floor((double)cursor);
        f1 = f0 + 1 <= layer->end_frame
            ? f0 + 1
            : layer->start_frame;
    } else {
        if (cursor < (float)layer->start_frame) {
            cursor = (float)layer->start_frame;
        } else if (cursor > (float)layer->end_frame) {
            cursor = (float)layer->end_frame;
        }
        f0 = (int32_t)floor((double)cursor);
        f1 = f0 + 1 <= layer->end_frame ? f0 + 1 : layer->end_frame;
    }
    result->f0 = f0;
    result->f1 = f1;
    result->t = f32_sub(cursor, (float)f0);
    return SONIC_MOTION_COMPOSER_NATIVE_OK;
}

static SonicMotionComposerNativeStatus advance_layer(
        SonicMotionComposerNativeLayer* layer,
        SonicMotionComposerNativeLayerAdvanceResult* result) {
    SonicMotionComposerNativeStatus status;
    if (layer == NULL || result == NULL) {
        return SONIC_MOTION_COMPOSER_NATIVE_INVALID_ARGUMENT;
    }
    result->wrapped = 0;
    result->completed = 0;
    if (!layer->active) {
        return SONIC_MOTION_COMPOSER_NATIVE_OK;
    }
    layer->cursor = f32_add(layer->cursor, layer->per_tick);
    if (!isfinite(layer->cursor)) {
        return SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE;
    }
    if (layer->config.loop) {
        status = wrap_loop_cursor(layer, &result->wrapped);
        return status;
    }
    if (layer->cursor < (float)layer->start_frame) {
        layer->cursor = (float)layer->start_frame;
    } else if (layer->cursor > (float)layer->end_frame) {
        layer->cursor = (float)layer->end_frame;
    }
    if (layer->per_tick >= 0.0f) {
        result->completed = !(layer->cursor < (float)layer->end_frame);
    } else {
        result->completed = !((float)layer->start_frame < layer->cursor);
    }
    return SONIC_MOTION_COMPOSER_NATIVE_OK;
}

static void copy_layer(
        const SonicMotionComposerNativeLayer* source,
        SonicMotionComposerNativeLayer* destination) {
    *destination = *source;
    destination->active = 1;
}

static SonicMotionComposerNativeStatus blend_frames(
        int32_t controller_rate_hz,
        float seconds,
        int32_t* result) {
    float product;
    double lower;
    double fraction;
    double rounded;
    if (result == NULL || controller_rate_hz <= 0) {
        return SONIC_MOTION_COMPOSER_NATIVE_INVALID_ARGUMENT;
    }
    if (!isfinite(seconds)) {
        return SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE;
    }
    product = f32_mul((float)controller_rate_hz, seconds);
    if (!isfinite(product)) {
        return SONIC_MOTION_COMPOSER_NATIVE_OUT_OF_RANGE;
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
        return SONIC_MOTION_COMPOSER_NATIVE_OUT_OF_RANGE;
    }
    *result = rounded < 1.0 ? 1 : (int32_t)rounded;
    return SONIC_MOTION_COMPOSER_NATIVE_OK;
}

static SonicMotionComposerNativeStatus weight_current_impl(
        float tt,
        int32_t w_in,
        int32_t w_out,
        float* result) {
    float outgoing_progress;
    float incoming_progress;
    float denominator;
    SonicMotionComposerNativeStatus status;
    status = clamp01(f32_div(tt, (float)w_out), &outgoing_progress);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    status = clamp01(f32_div(tt, (float)w_in), &incoming_progress);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    denominator = f32_add(f32_sub(1.0f, outgoing_progress), incoming_progress);
    if (denominator <= 0.0f) {
        *result = 1.0f;
    } else {
        *result = f32_div(incoming_progress, denominator);
    }
    return isfinite(*result)
        ? SONIC_MOTION_COMPOSER_NATIVE_OK
        : SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE;
}

static SonicMotionComposerNativeStatus xfade_at_impl(
        int32_t xt,
        int32_t frames_ahead,
        int32_t w_in,
        int32_t w_out,
        float* result) {
    float tt = (float)i32_add(xt, frames_ahead);
    return weight_current_impl(tt, w_in, w_out, result);
}

static SonicMotionComposerNativeStatus validate_mirror_table(
        const SonicMotionComposerNativeMirrorTable* table) {
    if (table == NULL || table->source_indices == NULL || table->negate == NULL) {
        return SONIC_MOTION_COMPOSER_NATIVE_UNSUPPORTED_MIRROR_TABLE;
    }
    if (table->source_index_count != SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT
            || table->negate_count != SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT) {
        return SONIC_MOTION_COMPOSER_NATIVE_INVALID_ARGUMENT;
    }
    for (size_t index = 0;
            index < SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT;
            index++) {
        if (table->source_indices[index]
                >= SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT
                || table->negate[index] > 1U) {
            return SONIC_MOTION_COMPOSER_NATIVE_OUT_OF_RANGE;
        }
    }
    return SONIC_MOTION_COMPOSER_NATIVE_OK;
}

static SonicMotionComposerNativeStatus slerp_wxyz(
        const SonicMotionComposerNative* composer,
        const float a[4],
        const float b[4],
        float t,
        float output[4]) {
    if (composer->backends.quaternion_slerp == NULL) {
        return SONIC_MOTION_COMPOSER_NATIVE_UNSUPPORTED_QUATERNION_SLERP;
    }
    if (!finite_floats(a, 4) || !finite_floats(b, 4) || !isfinite(t)) {
        return SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE;
    }
    if (!composer->backends.quaternion_slerp(
            composer->backends.context,
            a,
            b,
            t,
            output)) {
        return SONIC_MOTION_COMPOSER_NATIVE_BACKEND_FAILURE;
    }
    if (!finite_floats(output, 4)) {
        return SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE;
    }
    return SONIC_MOTION_COMPOSER_NATIVE_OK;
}

static SonicMotionComposerNativeStatus calc_heading_wxyz(
        const SonicMotionComposerNative* composer,
        const float quaternion[4],
        float* result) {
    float w;
    float x;
    float y;
    float z;
    float cross_sum;
    float numerator;
    float sum_squares;
    float denominator;
    if (result == NULL) {
        return SONIC_MOTION_COMPOSER_NATIVE_INVALID_ARGUMENT;
    }
    if (composer->backends.atan2_f == NULL) {
        return SONIC_MOTION_COMPOSER_NATIVE_UNSUPPORTED_ATAN2;
    }
    if (!finite_floats(quaternion, 4)) {
        return SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE;
    }
    w = quaternion[0];
    x = quaternion[1];
    y = quaternion[2];
    z = quaternion[3];
    cross_sum = f32_add(f32_mul(y, x), f32_mul(z, w));
    numerator = f32_add(cross_sum, cross_sum);
    sum_squares = f32_add(f32_mul(z, z), f32_mul(y, y));
    denominator = f32_sub(1.0f, f32_add(sum_squares, sum_squares));
    if (!composer->backends.atan2_f(
            composer->backends.context,
            numerator,
            denominator,
            result)) {
        return SONIC_MOTION_COMPOSER_NATIVE_BACKEND_FAILURE;
    }
    if (!isfinite(*result)) {
        return SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE;
    }
    return SONIC_MOTION_COMPOSER_NATIVE_OK;
}

static SonicMotionComposerNativeStatus sample_root_wxyz(
        const SonicMotionComposerNative* composer,
        const SonicMotionComposerNativeLayer* layer,
        const SonicMotionComposerNativeResolvedFrames* frames,
        float output[4]) {
    const float* a = layer->clip.root_quaternion_wxyz
        + (size_t)frames->f0 * 4;
    const float* b = layer->clip.root_quaternion_wxyz
        + (size_t)frames->f1 * 4;
    SonicMotionComposerNativeStatus status = slerp_wxyz(
        composer,
        a,
        b,
        frames->t,
        output);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    if (layer->config.mirror) {
        output[1] = -output[1];
        output[3] = -output[3];
    }
    return SONIC_MOTION_COMPOSER_NATIVE_OK;
}

static SonicMotionComposerNativeStatus remove_sampled_yaw(
        const SonicMotionComposerNative* composer,
        const SonicMotionComposerNativeLayer* layer,
        float root[4]) {
    float heading;
    float half_angle;
    float sine;
    float cosine;
    float w;
    float x;
    float y;
    float z;
    SonicMotionComposerNativeStatus status;
    if (layer->config.yaw_blend <= 0.0f) {
        return SONIC_MOTION_COMPOSER_NATIVE_OK;
    }
    status = calc_heading_wxyz(composer, root, &heading);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    if (composer->backends.sin_cos_f == NULL) {
        return SONIC_MOTION_COMPOSER_NATIVE_UNSUPPORTED_SIN_COS;
    }
    half_angle = f32_mul(
        f32_mul(-heading, layer->config.yaw_blend),
        HALF);
    if (!composer->backends.sin_cos_f(
            composer->backends.context,
            half_angle,
            &sine,
            &cosine)) {
        return SONIC_MOTION_COMPOSER_NATIVE_BACKEND_FAILURE;
    }
    if (!isfinite(sine) || !isfinite(cosine)) {
        return SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE;
    }
    w = root[0];
    x = root[1];
    y = root[2];
    z = root[3];
    root[0] = f32_sub(f32_mul(w, cosine), f32_mul(z, sine));
    root[1] = f32_sub(f32_mul(x, cosine), f32_mul(y, sine));
    root[2] = f32_add(f32_mul(x, sine), f32_mul(y, cosine));
    root[3] = f32_add(f32_mul(z, cosine), f32_mul(w, sine));
    if (!finite_floats(root, 4)) {
        return SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE;
    }
    return SONIC_MOTION_COMPOSER_NATIVE_OK;
}

static SonicMotionComposerNativeStatus sample_layer(
        const SonicMotionComposerNative* composer,
        const SonicMotionComposerNativeLayer* layer,
        int32_t frames_ahead,
        const SonicMotionComposerNativeMirrorTable* mirror_table,
        SonicMotionComposerNativePose* result) {
    SonicMotionComposerNativeResolvedFrames frames;
    const float* row0;
    const float* row1;
    SonicMotionComposerNativeStatus status;
    status = resolve_frames_impl(layer, frames_ahead, &frames);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    row0 = layer->clip.dof_position_mujoco
        + (size_t)frames.f0 * SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT;
    row1 = layer->clip.dof_position_mujoco
        + (size_t)frames.f1 * SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT;
    for (size_t output_index = 0;
            output_index < SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT;
            output_index++) {
        size_t source_index = output_index;
        if (layer->config.mirror) {
            source_index = mirror_table->source_indices[output_index];
        }
        status = lerp_f32(
            row0[source_index],
            row1[source_index],
            frames.t,
            &result->joint_positions[output_index]);
        if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
            return status;
        }
        if (layer->config.mirror && mirror_table->negate[output_index]) {
            result->joint_positions[output_index]
                = -result->joint_positions[output_index];
        }
    }
    status = sample_root_wxyz(composer, layer, &frames, result->root_quaternion_wxyz);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    return remove_sampled_yaw(composer, layer, result->root_quaternion_wxyz);
}

static SonicMotionComposerNativeStatus get_reference_frame(
        const SonicMotionComposerNative* composer,
        int32_t frames_ahead,
        const SonicMotionComposerNativeMirrorTable* mirror_table,
        SonicMotionComposerNativePose* result) {
    SonicMotionComposerNativePose current;
    SonicMotionComposerNativePose outgoing;
    float weight;
    SonicMotionComposerNativeStatus status;
    if (!composer->current_layer.active) {
        memset(result->joint_positions, 0, sizeof(result->joint_positions));
        result->root_quaternion_wxyz[0] = 1.0f;
        result->root_quaternion_wxyz[1] = 0.0f;
        result->root_quaternion_wxyz[2] = 0.0f;
        result->root_quaternion_wxyz[3] = 0.0f;
        return SONIC_MOTION_COMPOSER_NATIVE_OK;
    }
    status = sample_layer(
        composer,
        &composer->current_layer,
        frames_ahead,
        mirror_table,
        &current);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    if (!composer->from_layer.active) {
        *result = current;
        return SONIC_MOTION_COMPOSER_NATIVE_OK;
    }
    status = sample_layer(
        composer,
        &composer->from_layer,
        frames_ahead,
        mirror_table,
        &outgoing);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    status = xfade_at_impl(
        composer->xt,
        frames_ahead,
        composer->w_in,
        composer->w_out,
        &weight);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    for (size_t index = 0;
            index < SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT;
            index++) {
        status = lerp_f32(
            outgoing.joint_positions[index],
            current.joint_positions[index],
            weight,
            &result->joint_positions[index]);
        if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
            return status;
        }
    }
    return slerp_wxyz(
        composer,
        outgoing.root_quaternion_wxyz,
        current.root_quaternion_wxyz,
        weight,
        result->root_quaternion_wxyz);
}

static SonicMotionComposerNativeStatus wrap_pi(float angle, float* result) {
    if (result == NULL) {
        return SONIC_MOTION_COMPOSER_NATIVE_INVALID_ARGUMENT;
    }
    if (!isfinite(angle)) {
        return SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE;
    }
    while (angle > PI_F32) {
        float next = f32_add(angle, NEG_TWO_PI_F32);
        if (next == angle) {
            return SONIC_MOTION_COMPOSER_NATIVE_OUT_OF_RANGE;
        }
        angle = next;
    }
    while (angle < NEG_PI_F32) {
        float next = f32_add(angle, TWO_PI_F32);
        if (next == angle) {
            return SONIC_MOTION_COMPOSER_NATIVE_OUT_OF_RANGE;
        }
        angle = next;
    }
    *result = angle;
    return SONIC_MOTION_COMPOSER_NATIVE_OK;
}

static SonicMotionComposerNativeStatus layer_root_heading(
        const SonicMotionComposerNative* composer,
        const SonicMotionComposerNativeLayer* layer,
        SonicMotionComposerNativeRootHeading* result) {
    SonicMotionComposerNativeResolvedFrames frames;
    float root[4];
    SonicMotionComposerNativeStatus status;
    status = resolve_frames_impl(layer, 0, &frames);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    status = sample_root_wxyz(composer, layer, &frames, root);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    status = calc_heading_wxyz(composer, root, &result->heading);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    result->seam = layer->config.loop && frames.f1 < frames.f0;
    return SONIC_MOTION_COMPOSER_NATIVE_OK;
}

static SonicMotionComposerNativeStatus layer_heading_contribution(
        const SonicMotionComposerNative* composer,
        SonicMotionComposerNativeLayer* layer,
        int wrapped,
        float* result) {
    SonicMotionComposerNativeRootHeading sampled;
    float delta = 0.0f;
    SonicMotionComposerNativeStatus status;
    if (!layer->has_config) {
        layer->heading_valid = 0;
        *result = 0.0f;
        return SONIC_MOTION_COMPOSER_NATIVE_OK;
    }
    if (!isfinite(layer->config.yaw_blend)) {
        return SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE;
    }
    if (layer->config.yaw_blend <= 0.0f) {
        layer->heading_valid = 0;
        *result = 0.0f;
        return SONIC_MOTION_COMPOSER_NATIVE_OK;
    }
    status = layer_root_heading(composer, layer, &sampled);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    if (layer->heading_valid) {
        if (sampled.seam || wrapped) {
            layer->heading_resync = 1;
            delta = layer->last_heading_delta;
        } else if (layer->heading_resync) {
            layer->heading_resync = 0;
            delta = layer->last_heading_delta;
        } else {
            status = wrap_pi(
                f32_sub(sampled.heading, layer->prev_heading),
                &delta);
            if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
                return status;
            }
            layer->last_heading_delta = delta;
        }
    }
    layer->heading_valid = 1;
    layer->prev_heading = sampled.heading;
    *result = f32_mul(delta, layer->config.yaw_blend);
    if (!isfinite(*result)) {
        return SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE;
    }
    return SONIC_MOTION_COMPOSER_NATIVE_OK;
}

static SonicMotionComposerNativeStatus require_heading_backends(
        const SonicMotionComposerNative* composer) {
    const SonicMotionComposerNativeLayer* layers[2];
    size_t layer_count = 1;
    layers[0] = &composer->current_layer;
    if (composer->from_layer.active
            && i32_add(composer->xt, 1) < composer->w_total) {
        layers[layer_count++] = &composer->from_layer;
    }
    for (size_t index = 0; index < layer_count; index++) {
        const SonicMotionComposerNativeLayer* layer = layers[index];
        if (!layer->active || !layer->has_config
                || layer->config.yaw_blend <= 0.0f) {
            continue;
        }
        if (composer->backends.quaternion_slerp == NULL) {
            return SONIC_MOTION_COMPOSER_NATIVE_UNSUPPORTED_QUATERNION_SLERP;
        }
        if (composer->backends.atan2_f == NULL) {
            return SONIC_MOTION_COMPOSER_NATIVE_UNSUPPORTED_ATAN2;
        }
    }
    return SONIC_MOTION_COMPOSER_NATIVE_OK;
}

static SonicMotionComposerNativeStatus yaw_ownership(
        const SonicMotionComposerNativeLayer* layer,
        float* result) {
    float nonnegative;
    if (!layer->active || !layer->has_config) {
        *result = 0.0f;
        return SONIC_MOTION_COMPOSER_NATIVE_OK;
    }
    if (!isfinite(layer->config.yaw_blend)) {
        return SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE;
    }
    nonnegative = layer->config.yaw_blend < 0.0f
        ? 0.0f
        : layer->config.yaw_blend;
    return clamp01(nonnegative, result);
}

const char* sonic_motion_composer_native_status_string(
        SonicMotionComposerNativeStatus status) {
    switch (status) {
        case SONIC_MOTION_COMPOSER_NATIVE_OK:
            return "ok";
        case SONIC_MOTION_COMPOSER_NATIVE_INVALID_ARGUMENT:
            return "invalid argument";
        case SONIC_MOTION_COMPOSER_NATIVE_SIZE_OVERFLOW:
            return "size overflow";
        case SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE:
            return "non-finite input or result";
        case SONIC_MOTION_COMPOSER_NATIVE_OUT_OF_RANGE:
            return "unsupported or invalid numeric range";
        case SONIC_MOTION_COMPOSER_NATIVE_UNSUPPORTED_QUATERNION_SLERP:
            return "exact quaternion slerp backend is required";
        case SONIC_MOTION_COMPOSER_NATIVE_UNSUPPORTED_ATAN2:
            return "exact atan2 backend is required";
        case SONIC_MOTION_COMPOSER_NATIVE_UNSUPPORTED_SIN_COS:
            return "exact sin/cos backend is required";
        case SONIC_MOTION_COMPOSER_NATIVE_UNSUPPORTED_LOOP_ENTRY_MATCHER:
            return "exact active-source loop entry matcher is required";
        case SONIC_MOTION_COMPOSER_NATIVE_UNSUPPORTED_MIRROR_TABLE:
            return "exact mirror table is required";
        case SONIC_MOTION_COMPOSER_NATIVE_UNSUPPORTED_FLOAT_ENVIRONMENT:
            return "round-to-nearest floating-point environment is required";
        case SONIC_MOTION_COMPOSER_NATIVE_BACKEND_FAILURE:
            return "injected backend failed";
        default:
            return "unknown status";
    }
}

SonicMotionComposerNativeStatus sonic_motion_composer_native_init(
        SonicMotionComposerNative* composer,
        int32_t controller_rate_hz,
        const SonicMotionComposerNativeBackends* backends) {
    if (composer == NULL || controller_rate_hz <= 0) {
        return SONIC_MOTION_COMPOSER_NATIVE_INVALID_ARGUMENT;
    }
    if (validate_float_environment() != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return SONIC_MOTION_COMPOSER_NATIVE_UNSUPPORTED_FLOAT_ENVIRONMENT;
    }
    memset(composer, 0, sizeof(*composer));
    composer->controller_rate_hz = controller_rate_hz;
    composer->w_in = 1;
    composer->w_out = 1;
    composer->w_total = 1;
    if (backends != NULL) {
        composer->backends = *backends;
    }
    return SONIC_MOTION_COMPOSER_NATIVE_OK;
}

SonicMotionComposerNativeStatus sonic_motion_composer_native_install_layer(
        SonicMotionComposerNativeLayer* layer,
        const SonicMotionComposerNativeClip* clip,
        const SonicMotionComposerNativeConfig* config,
        int32_t controller_rate_hz) {
    int32_t last_frame;
    int32_t clamped_start;
    int32_t requested_end;
    int32_t clamped_end;
    float native_speed;
    float fps_over_rate;
    SonicMotionComposerNativeLayer next;
    SonicMotionComposerNativeStatus status;
    if (layer == NULL || controller_rate_hz <= 0) {
        return SONIC_MOTION_COMPOSER_NATIVE_INVALID_ARGUMENT;
    }
    status = validate_float_environment();
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    status = validate_clip_shape(clip, 1);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    status = validate_config(config);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    status = sanitize_playback_speed(config->playback_speed, &native_speed);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    last_frame = (int32_t)(clip->frame_count - 1);
    clamped_start = config->start_frame < 0 ? 0 : config->start_frame;
    if (clamped_start > last_frame) {
        clamped_start = last_frame;
    }
    requested_end = config->end_frame < 0
        ? last_frame
        : config->end_frame;
    if (requested_end < clamped_start) {
        clamped_end = clamped_start;
    } else if (requested_end > last_frame) {
        clamped_end = last_frame;
    } else {
        clamped_end = requested_end;
    }
    next = *layer;
    next.clip = *clip;
    next.config = *config;
    next.speed = native_speed;
    fps_over_rate = f32_div(clip->fps, (float)controller_rate_hz);
    next.per_tick = f32_mul(fps_over_rate, native_speed);
    if (!isfinite(next.per_tick)) {
        return SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE;
    }
    next.start_frame = clamped_start;
    next.end_frame = clamped_end;
    next.has_clip = 1;
    next.has_config = 1;
    next.active = 1;
    next.heading_valid = 0;
    next.heading_resync = 0;
    next.cursor = entry_cursor(&next);
    *layer = next;
    return SONIC_MOTION_COMPOSER_NATIVE_OK;
}

SonicMotionComposerNativeStatus sonic_motion_composer_native_resolve_frames(
        const SonicMotionComposerNativeLayer* layer,
        int32_t frames_ahead,
        SonicMotionComposerNativeResolvedFrames* result) {
    SonicMotionComposerNativeStatus status;
    if (result == NULL) {
        return SONIC_MOTION_COMPOSER_NATIVE_INVALID_ARGUMENT;
    }
    status = validate_float_environment();
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    status = validate_resolvable_layer(layer);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    return resolve_frames_impl(layer, frames_ahead, result);
}

SonicMotionComposerNativeStatus sonic_motion_composer_native_weight_current(
        float tt,
        int32_t w_in,
        int32_t w_out,
        float* result) {
    SonicMotionComposerNativeStatus status;
    if (result == NULL || w_in <= 0 || w_out <= 0) {
        return SONIC_MOTION_COMPOSER_NATIVE_INVALID_ARGUMENT;
    }
    status = validate_float_environment();
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    if (!isfinite(tt)) {
        return SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE;
    }
    return weight_current_impl(tt, w_in, w_out, result);
}

SonicMotionComposerNativeStatus sonic_motion_composer_native_xfade_at(
        int32_t xt,
        int32_t frames_ahead,
        int32_t w_in,
        int32_t w_out,
        float* result) {
    SonicMotionComposerNativeStatus status;
    if (result == NULL || w_in <= 0 || w_out <= 0) {
        return SONIC_MOTION_COMPOSER_NATIVE_INVALID_ARGUMENT;
    }
    status = validate_float_environment();
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    return xfade_at_impl(xt, frames_ahead, w_in, w_out, result);
}

SonicMotionComposerNativeStatus sonic_motion_composer_native_play_action(
        SonicMotionComposerNative* composer,
        const SonicMotionComposerNativeClip* clip,
        const SonicMotionComposerNativeConfig* config) {
    SonicMotionComposerNative next;
    SonicMotionComposerNativeLayer target;
    int outgoing_active;
    int32_t new_w_in;
    int32_t new_w_out;
    int32_t new_w_total;
    SonicMotionComposerNativeStatus status;
    status = validate_composer(composer);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    status = validate_config(config);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    outgoing_active = composer->current_layer.active;
    if (config->loop && outgoing_active
            && composer->backends.loop_entry_matcher == NULL) {
        return SONIC_MOTION_COMPOSER_NATIVE_UNSUPPORTED_LOOP_ENTRY_MATCHER;
    }
    new_w_in = composer->w_in;
    new_w_out = composer->w_out;
    new_w_total = composer->w_total;
    if (outgoing_active) {
        status = blend_frames(
            composer->controller_rate_hz,
            config->blend_in_seconds,
            &new_w_in);
        if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
            return status;
        }
        status = blend_frames(
            composer->controller_rate_hz,
            composer->current_layer.config.blend_out_seconds,
            &new_w_out);
        if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
            return status;
        }
        new_w_total = new_w_in > new_w_out ? new_w_in : new_w_out;
    }
    target = composer->current_layer;
    status = sonic_motion_composer_native_install_layer(
        &target,
        clip,
        config,
        composer->controller_rate_hz);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    next = *composer;
    if (outgoing_active) {
        copy_layer(&composer->current_layer, &next.from_layer);
    } else {
        next.from_layer.active = 0;
    }
    next.current_layer = target;
    next.xt = 0;
    next.action_playing = !config->loop;
    if (!config->loop) {
        next.action_move_id = i32_add(next.action_move_id, 1);
    }
    next.w_in = new_w_in;
    next.w_out = new_w_out;
    next.w_total = new_w_total;
    if (config->loop && outgoing_active) {
        float matched_cursor = 0.0f;
        if (!next.backends.loop_entry_matcher(
                next.backends.context,
                &next.current_layer,
                &next.from_layer,
                &matched_cursor)) {
            return SONIC_MOTION_COMPOSER_NATIVE_BACKEND_FAILURE;
        }
        if (!isfinite(matched_cursor)) {
            return SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE;
        }
        if (matched_cursor < (float)next.current_layer.start_frame
                || matched_cursor > (float)next.current_layer.end_frame) {
            return SONIC_MOTION_COMPOSER_NATIVE_OUT_OF_RANGE;
        }
        next.current_layer.cursor = matched_cursor;
        next.current_layer.heading_valid = 0;
    } else {
        next.current_layer.cursor = entry_cursor(&next.current_layer);
    }
    next.current_layer.heading_resync = 0;
    *composer = next;
    return SONIC_MOTION_COMPOSER_NATIVE_OK;
}

SonicMotionComposerNativeStatus sonic_motion_composer_native_play_action_immediate(
        SonicMotionComposerNative* composer,
        const SonicMotionComposerNativeClip* clip,
        const SonicMotionComposerNativeConfig* config) {
    SonicMotionComposerNativeStatus status = sonic_motion_composer_native_play_action(
        composer,
        clip,
        config);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    composer->from_layer.active = 0;
    composer->xt = 0;
    return SONIC_MOTION_COMPOSER_NATIVE_OK;
}

SonicMotionComposerNativeStatus sonic_motion_composer_native_cancel_action(
        SonicMotionComposerNative* composer) {
    SonicMotionComposerNativeStatus status = validate_composer(composer);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    composer->from_layer.active = 0;
    composer->action_playing = 0;
    composer->xt = 0;
    composer->current_layer.heading_valid = 0;
    composer->current_layer.heading_resync = 0;
    return SONIC_MOTION_COMPOSER_NATIVE_OK;
}

SonicMotionComposerNativeStatus sonic_motion_composer_native_set_locomotion_speed(
        SonicMotionComposerNative* composer,
        float scale) {
    SonicMotionComposerNativeLayer* layer;
    float fps_over_rate;
    float base;
    float effective_scale;
    SonicMotionComposerNativeStatus status = validate_composer(composer);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    layer = &composer->current_layer;
    if (!layer->active || !layer->config.loop) {
        return SONIC_MOTION_COMPOSER_NATIVE_OK;
    }
    if (!isfinite(scale)) {
        return SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE;
    }
    fps_over_rate = f32_div(
        layer->clip.fps,
        (float)composer->controller_rate_hz);
    base = f32_mul(fps_over_rate, layer->speed);
    effective_scale = scale < MIN_LOCOMOTION_SCALE
        ? MIN_LOCOMOTION_SCALE
        : scale;
    layer->per_tick = f32_mul(base, effective_scale);
    return isfinite(layer->per_tick)
        ? SONIC_MOTION_COMPOSER_NATIVE_OK
        : SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE;
}

SonicMotionComposerNativeStatus sonic_motion_composer_native_advance(
        SonicMotionComposerNative* composer,
        SonicMotionComposerNativeAdvanceResult* result) {
    SonicMotionComposerNative next;
    SonicMotionComposerNativeAdvanceResult local_result;
    SonicMotionComposerNativeStatus status;
    float contribution;
    float outgoing_weight;
    status = validate_composer(composer);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK || result == NULL) {
        return status != SONIC_MOTION_COMPOSER_NATIVE_OK
            ? status
            : SONIC_MOTION_COMPOSER_NATIVE_INVALID_ARGUMENT;
    }
    status = require_heading_backends(composer);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    next = *composer;
    status = advance_layer(&next.current_layer, &local_result.current);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    if (local_result.current.completed) {
        next.action_playing = 0;
    }
    local_result.outgoing.wrapped = 0;
    local_result.outgoing.completed = 0;
    if (next.from_layer.active) {
        status = advance_layer(&next.from_layer, &local_result.outgoing);
        if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
            return status;
        }
        next.xt = i32_add(next.xt, 1);
        if (next.xt >= next.w_total) {
            next.from_layer.active = 0;
        }
    }
    if (next.from_layer.active) {
        status = weight_current_impl(
            (float)next.xt,
            next.w_in,
            next.w_out,
            &local_result.weight_current);
        if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
            return status;
        }
    } else {
        local_result.weight_current = 1.0f;
    }
    if (next.current_layer.active) {
        status = layer_heading_contribution(
            &next,
            &next.current_layer,
            local_result.current.wrapped,
            &contribution);
        if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
            return status;
        }
        next.pending_heading_delta = f32_add(
            next.pending_heading_delta,
            f32_mul(contribution, local_result.weight_current));
    }
    if (next.from_layer.active) {
        status = layer_heading_contribution(
            &next,
            &next.from_layer,
            local_result.outgoing.wrapped,
            &contribution);
        if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
            return status;
        }
        outgoing_weight = f32_sub(1.0f, local_result.weight_current);
        next.pending_heading_delta = f32_add(
            next.pending_heading_delta,
            f32_mul(contribution, outgoing_weight));
    }
    if (!isfinite(next.pending_heading_delta)) {
        return SONIC_MOTION_COMPOSER_NATIVE_NON_FINITE;
    }
    *composer = next;
    *result = local_result;
    return SONIC_MOTION_COMPOSER_NATIVE_OK;
}

SonicMotionComposerNativeStatus sonic_motion_composer_native_consume_heading_delta(
        SonicMotionComposerNative* composer,
        float* result) {
    SonicMotionComposerNativeStatus status = validate_composer(composer);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK || result == NULL) {
        return status != SONIC_MOTION_COMPOSER_NATIVE_OK
            ? status
            : SONIC_MOTION_COMPOSER_NATIVE_INVALID_ARGUMENT;
    }
    *result = composer->pending_heading_delta;
    composer->pending_heading_delta = 0.0f;
    return SONIC_MOTION_COMPOSER_NATIVE_OK;
}

SonicMotionComposerNativeStatus sonic_motion_composer_native_heading_clip_ownership(
        const SonicMotionComposerNative* composer,
        float* result) {
    float weight = 1.0f;
    float current_ownership;
    float outgoing_ownership;
    float outgoing_weight;
    float combined;
    SonicMotionComposerNativeStatus status = validate_composer(composer);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK || result == NULL) {
        return status != SONIC_MOTION_COMPOSER_NATIVE_OK
            ? status
            : SONIC_MOTION_COMPOSER_NATIVE_INVALID_ARGUMENT;
    }
    if (composer->from_layer.active) {
        status = weight_current_impl(
            (float)composer->xt,
            composer->w_in,
            composer->w_out,
            &weight);
        if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
            return status;
        }
    }
    status = yaw_ownership(&composer->current_layer, &current_ownership);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    status = yaw_ownership(&composer->from_layer, &outgoing_ownership);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    current_ownership = f32_mul(weight, current_ownership);
    outgoing_weight = f32_sub(1.0f, weight);
    outgoing_ownership = f32_mul(outgoing_weight, outgoing_ownership);
    combined = f32_add(current_ownership, outgoing_ownership);
    return clamp01(combined, result);
}

SonicMotionComposerNativeStatus sonic_motion_composer_native_build_reference_rows(
        const SonicMotionComposerNative* composer,
        const SonicMotionComposerNativeReferenceTiming* timing,
        const SonicMotionComposerNativeMirrorTable* mirror_table,
        SonicMotionComposerNativeReferenceOutput* output) {
    float current_rows[
        SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS
        * SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT];
    float next_rows[
        SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS
        * SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT];
    float root_rows[SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS * 4];
    SonicMotionComposerNativePose current;
    SonicMotionComposerNativePose next;
    SonicMotionComposerNativeStatus status = validate_composer(composer);
    const size_t dof_values = SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS
        * SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT;
    const size_t root_values = SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS * 4;
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        return status;
    }
    if (timing == NULL || output == NULL
            || output->dof_position_mujoco == NULL
            || output->dof_next_position_mujoco == NULL
            || output->root_rotation_xyzw == NULL
            || output->dof_position_capacity < dof_values
            || output->dof_next_position_capacity < dof_values
            || output->root_rotation_capacity < root_values) {
        return SONIC_MOTION_COMPOSER_NATIVE_INVALID_ARGUMENT;
    }
    if (composer->current_layer.active
            && (composer->current_layer.config.mirror
                || (composer->from_layer.active
                    && composer->from_layer.config.mirror))) {
        status = validate_mirror_table(mirror_table);
        if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
            return status;
        }
    }
    for (size_t row = 0;
            row < SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS;
            row++) {
        status = get_reference_frame(
            composer,
            timing->current_offsets[row],
            mirror_table,
            &current);
        if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
            return status;
        }
        status = get_reference_frame(
            composer,
            timing->next_offsets[row],
            mirror_table,
            &next);
        if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
            return status;
        }
        memcpy(
            current_rows + row * SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT,
            current.joint_positions,
            sizeof(current.joint_positions));
        memcpy(
            next_rows + row * SONIC_MOTION_COMPOSER_NATIVE_DOF_COUNT,
            next.joint_positions,
            sizeof(next.joint_positions));
        root_rows[row * 4 + 0] = current.root_quaternion_wxyz[1];
        root_rows[row * 4 + 1] = current.root_quaternion_wxyz[2];
        root_rows[row * 4 + 2] = current.root_quaternion_wxyz[3];
        root_rows[row * 4 + 3] = current.root_quaternion_wxyz[0];
    }
    memcpy(output->dof_position_mujoco, current_rows, sizeof(current_rows));
    memcpy(output->dof_next_position_mujoco, next_rows, sizeof(next_rows));
    memcpy(output->root_rotation_xyzw, root_rows, sizeof(root_rows));
    return SONIC_MOTION_COMPOSER_NATIVE_OK;
}
