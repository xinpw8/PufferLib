#include "task_learning_contract.h"
#include <iostream>
#include <stdexcept>

int main() {
    using namespace rek5_task_learning;
    int checks = 0;
    auto check = [&](bool ok) { ++checks; if (!ok) throw std::runtime_error("check " + std::to_string(checks)); };
    auto near = [&](double a, double b) { check(std::abs(a-b) < 1e-10); };
    const double gamma = retention(.02, 120), lambda = trace_lambda(.02, 6.16, 120);
    near(std::pow(gamma, 6000), .5);
    near(std::pow(gamma * lambda, 308), .5);
    near(std::pow(retention(.04, 120), 3000), .5);
    const auto terminal = boundary_contract(Boundary::episode_terminal);
    check(!terminal.bootstrap_value && !terminal.continue_gae_trace && !terminal.carry_recurrent_state);
    const auto cut = boundary_contract(Boundary::rollout_cut);
    check(cut.bootstrap_value && !cut.continue_gae_trace && cut.carry_recurrent_state);
    const auto truncation = boundary_contract(Boundary::external_truncation);
    check(truncation.bootstrap_value && !truncation.continue_gae_trace && !truncation.carry_recurrent_state);
    near(advantage(5, 2, 100, 100, .9, .8, Boundary::episode_terminal), 3);
    near(advantage(5, 2, 10, 100, .9, .8, Boundary::external_truncation), 12);
    near(advantage(5, 2, 10, 100, .9, .8, Boundary::rollout_cut), 12);
    near(advantage(5, 2, 10, 4, .9, .8, Boundary::continuing), 14.88);
    RewardConfig config{};
    near(reward(config, 2, 1, true, 1).total, 1);
    config.terminal_win_points = 5;
    near(reward(config, 2, 1, false, 1).total, 1);
    near(reward(config, 2, 1, true, 1).total, 6);
    near(reward(config, 2, 1, true, 0).total, 1);
    near(reward(config, 0, 5, true, -1).total, -10);
    near(reward(config, 5, 0, true, 1).total, 10);
    near(reward(config, 0, 0, true, 1).terminal_outcome, 5);
    config.point_scale = .5f;
    near(reward(config, 2, 0, true, 1).total, 3.5);
    std::cout << "task_learning_contract_checks=" << checks << " passed\n";
}
