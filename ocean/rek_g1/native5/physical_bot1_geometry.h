#pragma once
#include <cmath>

#if defined(__CUDACC__)
#define REK_BOT_GEOMETRY_FN __host__ __device__ inline
#else
#define REK_BOT_GEOMETRY_FN inline
#endif

namespace rek5_bot1_physical {
// Recovered AIOpponentController DistanceToOpponent / AngleToOpponent:
// root positions projected onto the floor; signed angle uses projected native
// Robot.Forward. Current G1's serialized -180 degree forwardYawOffset makes
// horizontal root-local +X the forward convention in common MuJoCo XY.
// Inputs are existing physical free-root XYZ/WXYZ, never missing transforms.
// This does not claim bitwise equivalence to Unity vector/quaternion primitives.
REK_BOT_GEOMETRY_FN bool geometry(const float* own, const float* opponent,
        float& distance, float& angle_degrees) {
    for (int k = 0; k < 7; ++k) if (!std::isfinite(own[k])) return false;
    for (int k = 0; k < 3; ++k) if (!std::isfinite(opponent[k])) return false;
    const float dx = opponent[0] - own[0], dy = opponent[1] - own[1];
    const float target_sq = dx * dx + dy * dy;
    float w = own[3], x = own[4], y = own[5], z = own[6];
    const float norm_sq = w*w + x*x + y*y + z*z;
    if (!std::isfinite(target_sq) || !std::isfinite(norm_sq) || norm_sq <= 0) return false;
    const float inv_norm = 1.0f / std::sqrt(norm_sq);
    w *= inv_norm; x *= inv_norm; y *= inv_norm; z *= inv_norm;
    const float fx = 1.0f - 2.0f * (y*y + z*z);
    const float fy = 2.0f * (w*z + x*y);
    const float forward_sq = fx*fx + fy*fy;
    distance = std::sqrt(target_sq);
    // Native guard is squared horizontal magnitude < 0.001, not zero only.
    if (target_sq < 0.001f || forward_sq < 0.001f) {
        angle_degrees = 0;
        return true;
    }
    const float dot = (fx*dx + fy*dy) / std::sqrt(forward_sq * target_sq);
    const float bounded = std::fmin(1.0f, std::fmax(-1.0f, dot));
    const float cross = fy*dx - fx*dy;
    // Native Sign(0) is positive, so exactly collinear behind is +180.
    angle_degrees = std::acos(bounded) * 57.29578f * (cross >= 0 ? 1.0f : -1.0f);
    return std::isfinite(distance) && std::isfinite(angle_degrees);
}
}
#undef REK_BOT_GEOMETRY_FN
