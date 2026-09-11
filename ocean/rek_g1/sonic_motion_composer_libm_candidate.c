#include "sonic_motion_composer_libm_candidate.h"

#include <math.h>
#include <stddef.h>

static REK_G1_FN float f32_add(float a, float b) {
    volatile float result = a + b;
    return result;
}

static REK_G1_FN float f32_mul(float a, float b) {
    volatile float result = a * b;
    return result;
}

static REK_G1_FN int quaternion_finite(const float value[4]) {
    if (value == NULL) return 0;
    for (size_t axis = 0; axis < 4; axis++) {
        if (!isfinite(value[axis])) return 0;
    }
    return 1;
}

static REK_G1_FN int normalize(float value[4]) {
    float squared = 0.0f;
    for (size_t axis = 0; axis < 4; axis++) {
        squared = f32_add(squared, f32_mul(value[axis], value[axis]));
    }
    if (!isfinite(squared) || squared <= 0.0f) return 0;
    const float norm = sqrtf(squared);
    if (!isfinite(norm) || norm <= 0.0f) return 0;
    const float inverse = 1.0f / norm;
    if (!isfinite(inverse)) return 0;
    for (size_t axis = 0; axis < 4; axis++) {
        value[axis] = f32_mul(value[axis], inverse);
    }
    return quaternion_finite(value);
}

REK_G1_FN int sonic_motion_composer_libm_candidate_quaternion_slerp(
        void* context,
        const float a_wxyz[4],
        const float b_wxyz[4],
        float t,
        float output_wxyz[4]) {
    (void)context;
    if (!quaternion_finite(a_wxyz) || !quaternion_finite(b_wxyz)
            || output_wxyz == NULL || !isfinite(t)) {
        return 0;
    }
    float clamped_t = t;
    if (clamped_t < 0.0f) clamped_t = 0.0f;
    if (clamped_t > 1.0f) clamped_t = 1.0f;

    float a[4];
    float b[4];
    for (size_t axis = 0; axis < 4; axis++) {
        a[axis] = a_wxyz[axis];
        b[axis] = b_wxyz[axis];
    }
    if (!normalize(a) || !normalize(b)) return 0;

    float dot = 0.0f;
    for (size_t axis = 0; axis < 4; axis++) {
        dot = f32_add(dot, f32_mul(a[axis], b[axis]));
    }
    if (dot < 0.0f) {
        dot = -dot;
        for (size_t axis = 0; axis < 4; axis++) b[axis] = -b[axis];
    }
    if (dot > 1.0f) dot = 1.0f;

    float result[4];
    if (dot > 0.9995f) {
        for (size_t axis = 0; axis < 4; axis++) {
            result[axis] = f32_add(
                a[axis],
                f32_mul(clamped_t, b[axis] - a[axis]));
        }
    } else {
        const float theta = acosf(dot);
        const float sine_theta = sinf(theta);
        if (!isfinite(theta) || !isfinite(sine_theta)
                || fabsf(sine_theta) <= 1.0e-7f) {
            return 0;
        }
        const float scaled_theta = f32_mul(theta, clamped_t);
        const float sine_scaled = sinf(scaled_theta);
        const float scale_b = sine_scaled / sine_theta;
        const float scale_a = cosf(scaled_theta) - f32_mul(dot, scale_b);
        if (!isfinite(scale_a) || !isfinite(scale_b)) return 0;
        for (size_t axis = 0; axis < 4; axis++) {
            result[axis] = f32_add(
                f32_mul(scale_a, a[axis]),
                f32_mul(scale_b, b[axis]));
        }
    }
    if (!normalize(result)) return 0;
    for (size_t axis = 0; axis < 4; axis++) output_wxyz[axis] = result[axis];
    return 1;
}

REK_G1_FN int sonic_motion_composer_libm_candidate_atan2_f(
        void* context,
        float numerator,
        float denominator,
        float* output) {
    (void)context;
    if (output == NULL || !isfinite(numerator) || !isfinite(denominator)) {
        return 0;
    }
    const float result = atan2f(numerator, denominator);
    if (!isfinite(result)) return 0;
    *output = result;
    return 1;
}

REK_G1_FN int sonic_motion_composer_libm_candidate_sin_cos_f(
        void* context,
        float angle,
        float* sine,
        float* cosine) {
    (void)context;
    if (sine == NULL || cosine == NULL || !isfinite(angle)) return 0;
    const float sin_result = sinf(angle);
    const float cos_result = cosf(angle);
    if (!isfinite(sin_result) || !isfinite(cos_result)) return 0;
    *sine = sin_result;
    *cosine = cos_result;
    return 1;
}
