#pragma once
// CPU emulation of the compact runtime's per-arena step, for tests only.
// The device half of fast_runtime.cu is compiled as host C++ with thin CUDA
// shims and driven on synthetic assets: no GPU, private assets or trainer.
#include <cstdint>
#include <string>
#include <vector>

namespace rek_fast_cpu {
enum LearnerPolicy { RepeatHH=0, ApproachAndHH=1, AlternateKicks=2, Idle=3 };
struct Config {
    int arenas=8, ticks=3000;
    float round_seconds=20;
    uint32_t seed=73;
    LearnerPolicy learner=RepeatHH;
    // Fields below exist only in the modified runtime.
    std::string lite_model;   // empty: lite falls disabled
    bool move_reward=false;
    int reward_move=7;
    float reward_value=.01f;
    std::string dataset_out;  // aggregate with lite_fall_dataset_observe.h and write a dataset
};
struct Totals {
    uint64_t ticks=0, rounds=0, learner_starts[17]{}, opponent_starts[17]{};
    double learner_reward=0, opponent_reward=0;
    uint64_t nonzero_reward_ticks=0, reward_without_target_start=0, target_start_without_reward=0;
    uint64_t falls[2]{}, onsets[2]{}, recoveries[2]{}, counts_started=0, knockouts=0, double_knockouts=0;
    uint64_t knockout_points[2]{}, spawn_resets=0, stuck_cleared_by_reset=0;
    uint64_t hits_while_not_upright=0, nonzero_mask_while_down=0, deferred_round_end_ticks=0;
    uint64_t terminal_during_count=0, invariant_failures=0;
    uint64_t trajectory_hash=1469598103934665603ull;  // FNV-1a over every exported buffer
};
Totals run_original(const Config&);
Totals run_modified(const Config&);
}
