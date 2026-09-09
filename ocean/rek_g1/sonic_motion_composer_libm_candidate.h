#pragma once

#include "sonic_motion_composer_native.h"

/*
 * Portable libm implementations for the native composer's external math
 * surfaces. These functions make the public-family candidate executable on
 * Linux. They are not evidence that glibc libm is bit-identical to Unity's
 * Windows icalls and therefore do not establish REK trajectory parity.
 *
 * The callback context is ignored. A caller may use it for the separately
 * supplied loop-entry matcher.
 */

int sonic_motion_composer_libm_candidate_quaternion_slerp(
    void* context,
    const float a_wxyz[4],
    const float b_wxyz[4],
    float t,
    float output_wxyz[4]
);

int sonic_motion_composer_libm_candidate_atan2_f(
    void* context,
    float numerator,
    float denominator,
    float* output
);

int sonic_motion_composer_libm_candidate_sin_cos_f(
    void* context,
    float angle,
    float* sine,
    float* cosine
);
