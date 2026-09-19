#ifndef REK_NATIVE5_TASK_LEARNING_CONTRACT_H
#define REK_NATIVE5_TASK_LEARNING_CONTRACT_H

#include <cmath>

#if defined(__CUDACC__)
#define REK_TASK_HD __host__ __device__
#else
#define REK_TASK_HD
#endif

namespace rek5_task_learning {

// These flags refer to the transition that produced a reward. A scheduled
// training batch boundary is not an episode ending.
enum class Boundary { continuing, rollout_cut, episode_terminal, external_truncation };
struct BoundaryContract {
    bool bootstrap_value;
    bool continue_gae_trace;
    bool carry_recurrent_state;
};
REK_TASK_HD inline BoundaryContract boundary_contract(Boundary boundary) {
    switch (boundary) {
        case Boundary::episode_terminal: return {false, false, false};
        case Boundary::external_truncation: return {true, false, false};
        case Boundary::rollout_cut: return {true, false, true};
        default: return {true, true, true};
    }
}

// gamma is reward retention per control tick. gamma*lambda is GAE trace
// retention, so deriving lambda alone from seconds would change that target.
REK_TASK_HD inline double retention(double tick_seconds, double half_life_seconds) {
    return ::exp(-::log(2.0) * tick_seconds / half_life_seconds);
}
REK_TASK_HD inline double trace_lambda(double tick_seconds, double trace_half_life_seconds,
        double discount_half_life_seconds) {
    return retention(tick_seconds, trace_half_life_seconds) /
        retention(tick_seconds, discount_half_life_seconds);
}

struct RewardConfig {
    float point_scale = 1.0f;
    float terminal_win_points = 0.0f; // Explicit opt-in, denominated in game points.
};
struct RewardBreakdown { float scored_points, terminal_outcome, total; };

// Point deltas already contain every actual game award, including a KO if
// the runtime models one. Do not add another KO reward here. The optional
// W/L bonus is an independently declared objective, equal for every win path.
// Call with terminal_transition exactly once, when the round ends.
REK_TASK_HD inline RewardBreakdown reward(const RewardConfig& config,
        float own_point_delta, float opponent_point_delta, bool terminal_transition,
        int own_outcome) {
    const float scored = config.point_scale * (own_point_delta - opponent_point_delta);
    const float outcome = terminal_transition ?
        config.point_scale * float(own_outcome) * config.terminal_win_points : 0.0f;
    return {scored, outcome, scored + outcome};
}

// Expected GAE for one transition, including the final action in a batch.
// A truncation's next_value must come from its final observation, before reset.
REK_TASK_HD inline double advantage(double reward_value, double value, double next_value,
        double next_advantage, double gamma, double lambda, Boundary boundary) {
    const auto contract = boundary_contract(boundary);
    const double delta = reward_value + (contract.bootstrap_value ? gamma * next_value : 0) - value;
    return delta + (contract.continue_gae_trace ? gamma * lambda * next_advantage : 0);
}

} // namespace rek5_task_learning
#undef REK_TASK_HD
#endif
