#include "sonic_motion_composer_libm_candidate.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int assertions;

static void require(int condition, const char* name) {
    assertions += 1;
    if (!condition) {
        fprintf(stderr, "FAIL %s\n", name);
        exit(1);
    }
}

static float norm(const float value[4]) {
    return sqrtf(
        value[0] * value[0] + value[1] * value[1]
        + value[2] * value[2] + value[3] * value[3]);
}

static void test_endpoints_and_normalization(void) {
    const float identity[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    const float yaw_180[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    float output[4] = {0};
    require(sonic_motion_composer_libm_candidate_quaternion_slerp(
        NULL, identity, yaw_180, 0.0f, output), "slerp_t0_success");
    require(fabsf(output[0] - 1.0f) < 1.0e-6f, "slerp_t0_identity");
    require(sonic_motion_composer_libm_candidate_quaternion_slerp(
        NULL, identity, yaw_180, 0.5f, output), "slerp_half_success");
    require(fabsf(norm(output) - 1.0f) < 1.0e-6f, "slerp_half_unit");
    require(fabsf(fabsf(output[0]) - 0.70710677f) < 1.0e-5f,
        "slerp_half_scalar");
    require(fabsf(fabsf(output[3]) - 0.70710677f) < 1.0e-5f,
        "slerp_half_yaw");
}

static void test_shortest_path_and_clamp(void) {
    const float a[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    const float negative_a[4] = {-1.0f, 0.0f, 0.0f, 0.0f};
    float output[4] = {0};
    require(sonic_motion_composer_libm_candidate_quaternion_slerp(
        NULL, a, negative_a, 0.5f, output), "slerp_antipodal_success");
    require(fabsf(output[0] - 1.0f) < 1.0e-6f, "slerp_antipodal_shortest");
    require(sonic_motion_composer_libm_candidate_quaternion_slerp(
        NULL, a, negative_a, 2.0f, output), "slerp_clamped_success");
    require(fabsf(output[0] - 1.0f) < 1.0e-6f, "slerp_t_clamped");
}

static void test_scalar_backends(void) {
    float value = 0.0f;
    float sine = 0.0f;
    float cosine = 0.0f;
    require(sonic_motion_composer_libm_candidate_atan2_f(
        NULL, 1.0f, 0.0f, &value), "atan2_success");
    require(fabsf(value - 1.57079637f) < 1.0e-6f, "atan2_value");
    require(sonic_motion_composer_libm_candidate_sin_cos_f(
        NULL, 0.25f, &sine, &cosine), "sin_cos_success");
    require(fabsf(sine * sine + cosine * cosine - 1.0f) < 2.0e-6f,
        "sin_cos_identity");
    require(!sonic_motion_composer_libm_candidate_atan2_f(
        NULL, NAN, 1.0f, &value), "atan2_nonfinite_rejected");
    require(!sonic_motion_composer_libm_candidate_sin_cos_f(
        NULL, INFINITY, &sine, &cosine), "sin_cos_nonfinite_rejected");
}

int main(void) {
    test_endpoints_and_normalization();
    test_shortest_path_and_clamp();
    test_scalar_backends();
    printf("libm candidate tests passed: %d assertions\n", assertions);
    return 0;
}
