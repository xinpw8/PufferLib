#pragma once
#include "sonic_motion_composer_native.h"
#include <math.h>

/* Recovered SonicPolicyRunner.CalcHeadingMj/YawQuatMj operation order.
 * Transcendental callbacks retain their declared backend provenance. */
REK_G1_FN static inline int rek_g1_heading_angle(
        const SonicMotionComposerNativeBackends* b, const float q[4], float* out) {
    if (!b || !b->atan2_f || !q || !out) return 0;
    for (int i=0; i<4; ++i) if (!isfinite(q[i])) return 0;
    volatile float yx=q[2]*q[1], zw=q[3]*q[0];
    volatile float sum=yx+zw, numerator=2.0f*sum;
    volatile float zz=q[3]*q[3], yy=q[2]*q[2];
    volatile float sq=zz+yy, twice=2.0f*sq, denominator=1.0f-twice;
    return b->atan2_f(b->context,numerator,denominator,out) && isfinite(*out);
}

REK_G1_FN static inline int rek_g1_initial_heading(
        const SonicMotionComposerNativeBackends* b, const float base[4],
        const float reference[4], float out[4]) {
    float base_angle, reference_angle, sine, cosine;
    if (!b || !b->sin_cos_f || !out
            || !rek_g1_heading_angle(b,base,&base_angle)
            || !rek_g1_heading_angle(b,reference,&reference_angle)) return 0;
    volatile float difference=base_angle-reference_angle, half=difference*0.5f;
    if (!b->sin_cos_f(b->context,half,&sine,&cosine)
            || !isfinite(sine) || !isfinite(cosine)) return 0;
    out[0]=cosine; out[1]=0; out[2]=0; out[3]=sine;
    return 1;
}
