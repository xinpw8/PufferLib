#pragma once
#include "../g1_fall_state.h"
#include <cstdint>

#if defined(__CUDACC__)
#define REK_NORMALIZED_REWARD_FN __host__ __device__ inline
#else
#define REK_NORMALIZED_REWARD_FN inline
#endif

namespace rek5_normalized_reward {
constexpr const char* kMode = "normalized_points_falls_v1";
constexpr int kScalePoints = 100;
constexpr int kOwnFallPenaltyPoints = 1;

struct Result {
    float reward;
    // Components before safety clipping, in normalized reward units.
    float score_reward;
    float fall_reward;
    std::int64_t raw_total;
    bool saturated;
};

// Supply awarded point deltas for this step, including referee awards, and
// the own fighter's recovered detector events for the same step. Do not pass
// score totals, contacts, attack requests, a persistent fallen flag, or a
// counter delta across a round reset. Callers must consume each event once.
// BECAME_FALLEN is confirmed falling -> fallen. FALLING_STARTED may recover;
// RESET_TIMEOUT_DUE may repeat while down. Neither earns another penalty.
// There is no opponent-fall bonus or extra terminal reward.
REK_NORMALIZED_REWARD_FN Result value(int own_awarded_delta,
        int opponent_awarded_delta, std::uint32_t own_fall_events) {
    const std::int64_t score = std::int64_t(own_awarded_delta)
        - std::int64_t(opponent_awarded_delta);
    const int fall = (own_fall_events & REK_G1_FALL_EVENT_BECAME_FALLEN)
        ? kOwnFallPenaltyPoints : 0;
    const std::int64_t raw = score - fall;
    const bool saturated = raw < -kScalePoints || raw > kScalePoints;
    const double bounded = raw < -kScalePoints ? -1.0
        : raw > kScalePoints ? 1.0 : double(raw) / kScalePoints;
    return {float(bounded), float(double(score) / kScalePoints),
        -float(double(fall) / kScalePoints), raw, saturated};
}
}
#undef REK_NORMALIZED_REWARD_FN
