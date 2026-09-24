#ifndef REK_AUTHENTIC_GAE_H
#define REK_AUTHENTIC_GAE_H
#include <cmath>
#include <stdexcept>
#include <vector>
namespace rek_authentic_gae {
struct Step { double reward, gamma, lambda, old_value; bool terminal_after; };
struct Targets { std::vector<float> advantages, returns; };
inline Targets compute(const std::vector<Step>& sequence) {
    if (sequence.empty() || !sequence.back().terminal_after)
        throw std::runtime_error("GAE requires a closed captured round");
    Targets out; out.advantages.resize(sequence.size()); out.returns.resize(sequence.size());
    double next_advantage = 0;
    for (size_t n = sequence.size(); n > 0; --n) {
        const size_t t = n - 1; const auto& s = sequence[t];
        if (!std::isfinite(s.reward) || !std::isfinite(s.old_value)
                || !std::isfinite(s.gamma) || !std::isfinite(s.lambda)
                || s.gamma <= 0 || s.gamma > 1 || s.lambda < 0 || s.lambda > 1)
            throw std::runtime_error("invalid GAE input");
        const double next_value = s.terminal_after ? 0 : sequence[t + 1].old_value;
        const double delta = s.reward + s.gamma * next_value - s.old_value;
        const double advantage = delta + (s.terminal_after ? 0 : s.gamma * s.lambda * next_advantage);
        const double value_return = s.old_value + advantage;
        if (!std::isfinite(float(advantage)) || !std::isfinite(float(value_return)))
            throw std::runtime_error("nonfinite GAE target");
        out.advantages[t] = float(advantage); out.returns[t] = float(value_return);
        next_advantage = advantage;
    }
    return out;
}
} // namespace rek_authentic_gae
#endif
