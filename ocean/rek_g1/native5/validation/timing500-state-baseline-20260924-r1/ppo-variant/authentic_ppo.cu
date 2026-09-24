// Reuse the tested Puffer5 recurrent buffers/full-prefix replay, without the BC CLI.
#define main rek_bc_embedded_command_entry
#include "bc_train.cu"
#undef main
#include "puffer5_ppo_fp32.cuh"
#include "authentic_trajectory.h"
#include "owned_yaw_trajectory.h"
#include "authentic_gae.h"
#include "authentic_parity.h"
#include "authentic_distributional_parity.h"
#include "balance8_observation.h"
#include "crossfit_baseline.h"
#include <optional>

namespace {
struct AuthenticData {
    rek_bc::Dataset history;
};
AuthenticData prepare(const rek_authentic::Dataset& data, const rek_authentic::Replay& replay) {
    require(data.rows.size() == replay.rows.size(), "behavior replay row count mismatch");
    AuthenticData out; out.history.feature_mask = data.feature_mask;
    for (size_t i = 0; i < data.rows.size(); ++i) {
        const auto& row = data.rows[i]; const auto& old = replay.rows[i];
        require(row.split == 0 && old.index == i && old.action == row.action, "authentic row/replay identity mismatch");
        require(std::isfinite(old.old_logprob) && std::isfinite(old.old_value), "nonfinite frozen behavior");
        rek_bc::Row h; h.split = row.split; h.sequence = row.sequence; h.reset = row.reset;
        h.time = row.time; h.obs = row.obs; h.support = row.support;
        // The history adapter's label fields are never consumed by PPO.
        h.action = -1; h.weight = 0; out.history.rows.push_back(h);
    }
    rek_bc::validate(out.history);
    return out;
}
struct GaeStep { float reward, gamma, lambda, old_value; uint32_t terminal_after; };
__global__ void authentic_gae_cuda(const GaeStep* steps, const int* bounds,
        int rounds, float* advantages, float* returns, bool complete_mc_zero_baseline) {
    const int round = blockIdx.x * blockDim.x + threadIdx.x;
    if (round >= rounds) return;
    const int begin = bounds[round * 2], end = bounds[round * 2 + 1];
    double next_advantage = 0;
    for (int t = end - 1; t >= begin; --t) {
        const GaeStep s = steps[t];
        const double old_value = complete_mc_zero_baseline ? 0 : s.old_value;
        const double next_value = s.terminal_after || complete_mc_zero_baseline ? 0 : steps[t + 1].old_value;
        const double lambda = complete_mc_zero_baseline ? 1 : s.lambda;
        const double delta = double(s.reward) + double(s.gamma) * next_value - old_value;
        const double advantage = delta + (s.terminal_after ? 0 : double(s.gamma) * lambda * next_advantage);
        advantages[t] = float(advantage); returns[t] = float(old_value + advantage);
        next_advantage = advantage;
    }
}
__global__ void authentic_target_slice(precision_t* adv, precision_t* ret,
        const float* all_adv, const float* all_ret, int begin, int valid, int T) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    adv[t] = from_float(t < valid ? all_adv[begin + t] : 0);
    ret[t] = from_float(t < valid ? all_ret[begin + t] : 0);
}
__global__ void authentic_subtract_state_baseline(float* advantages, const float* returns,
        const float* prediction, int rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) advantages[i] = rek_crossfit_baseline::subtract(returns[i], prediction[i]);
}
struct GaeTargets {
    float *advantages = nullptr, *returns = nullptr;
    GaeTargets(const rek_authentic::Dataset& data, const rek_authentic::Replay& replay, bool complete_mc_zero_baseline,
            const rek_crossfit_baseline::Baseline* state_baseline = nullptr) {
        require(!state_baseline || complete_mc_zero_baseline, "state baseline requires complete MC targets");
        std::vector<GaeStep> input; std::vector<int> bounds;
        for (size_t i = 0; i < data.rows.size(); ++i) {
            const auto& r = data.rows[i]; input.push_back({r.reward, r.gamma, r.lambda, replay.rows[i].old_value, r.terminal_after});
        }
        for (const auto& s : data.sequences) { bounds.push_back(int(s.begin)); bounds.push_back(int(s.end)); }
        GaeStep* device_input = nullptr; int* device_bounds = nullptr;
        ck(cudaMalloc(&device_input, input.size() * sizeof(GaeStep))); ck(cudaMalloc(&device_bounds, bounds.size() * sizeof(int)));
        ck(cudaMalloc(&advantages, input.size() * sizeof(float))); ck(cudaMalloc(&returns, input.size() * sizeof(float)));
        ck(cudaMemcpy(device_input, input.data(), input.size() * sizeof(GaeStep), cudaMemcpyHostToDevice));
        ck(cudaMemcpy(device_bounds, bounds.data(), bounds.size() * sizeof(int), cudaMemcpyHostToDevice));
        authentic_gae_cuda<<<grid_size(data.sequences.size()), BLOCK_SIZE>>>(device_input, device_bounds, data.sequences.size(), advantages, returns, complete_mc_zero_baseline);
        ck(cudaDeviceSynchronize()); cudaFree(device_input); cudaFree(device_bounds);
        // CPU is an independent verification reference, never the training-target producer.
        std::vector<float> actual_adv(input.size()), actual_ret(input.size());
        ck(cudaMemcpy(actual_adv.data(), advantages, input.size() * sizeof(float), cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(actual_ret.data(), returns, input.size() * sizeof(float), cudaMemcpyDeviceToHost));
        double maximum = 0;
        for (const auto& seq : data.sequences) {
            std::vector<rek_authentic_gae::Step> reference;
            for (size_t i = seq.begin; i < seq.end; ++i) {
                const auto& s = input[i]; reference.push_back({s.reward, s.gamma,
                    complete_mc_zero_baseline ? 1 : s.lambda,
                    complete_mc_zero_baseline ? 0 : s.old_value, bool(s.terminal_after)});
            }
            const auto expected = rek_authentic_gae::compute(reference);
            for (size_t i = seq.begin; i < seq.end; ++i) {
                require(std::isfinite(actual_adv[i]) && std::isfinite(actual_ret[i]), "nonfinite CUDA GAE");
                maximum = std::max(maximum, std::abs(double(actual_adv[i]) - expected.advantages[i - seq.begin]));
                maximum = std::max(maximum, std::abs(double(actual_ret[i]) - expected.returns[i - seq.begin]));
            }
        }
        double sum = 0, sumsq = 0;
        for (float x : actual_adv) { sum += x; sumsq += double(x) * x; }
        const double mean = sum / actual_adv.size();
        std::printf("{\"phase\":\"cuda_target_reference_check\",\"target_mode\":\"%s\",\"rows\":%zu,\"max_abs_error\":%.12g,\"training_target_producer\":\"cuda\",\"advantage_mean\":%.12g,\"advantage_std\":%.12g,\"advantage_min\":%.12g,\"advantage_max\":%.12g}\n",
            state_baseline ? "complete_mc_before_state_baseline" : complete_mc_zero_baseline ? "complete_mc_zero_baseline" : "frozen_value_gae", input.size(), maximum,
            mean, std::sqrt(std::max(0.0, sumsq / actual_adv.size() - mean * mean)),
            double(*std::min_element(actual_adv.begin(), actual_adv.end())), double(*std::max_element(actual_adv.begin(), actual_adv.end())));
        require(maximum <= 1e-6, "CUDA GAE differs from CPU reference");
        if (state_baseline) {
            require(state_baseline->prediction.size() == input.size() && state_baseline->returns.size() == input.size()
                && state_baseline->residual.size() == input.size(), "baseline row count changed");
            for (size_t i = 0; i < input.size(); ++i)
                require(actual_ret[i] == state_baseline->returns[i], "CUDA MC return differs from pinned baseline return");
            float* prediction = nullptr;
            ck(cudaMalloc(&prediction, input.size() * sizeof(float)));
            ck(cudaMemcpy(prediction, state_baseline->prediction.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
            authentic_subtract_state_baseline<<<grid_size(input.size()), BLOCK_SIZE>>>(advantages, returns, prediction, int(input.size()));
            ck(cudaGetLastError()); ck(cudaDeviceSynchronize()); cudaFree(prediction);
            std::vector<float> after_adv(input.size()), after_ret(input.size());
            ck(cudaMemcpy(after_adv.data(), advantages, input.size() * sizeof(float), cudaMemcpyDeviceToHost));
            ck(cudaMemcpy(after_ret.data(), returns, input.size() * sizeof(float), cudaMemcpyDeviceToHost));
            double residual_sum = 0, residual_sumsq = 0; size_t eligible = 0;
            for (size_t i = 0; i < input.size(); ++i) {
                require(std::isfinite(after_adv[i]) && after_adv[i] == state_baseline->residual[i]
                    && after_ret[i] == actual_ret[i], "CUDA baseline subtraction/reference or unchanged return failure");
                if (data.rows[i].policy_weight == 1) {
                    ++eligible; residual_sum += after_adv[i]; residual_sumsq += double(after_adv[i]) * after_adv[i];
                }
            }
            require(eligible > 0, "no eligible baseline rows");
            const double residual_mean = residual_sum / eligible;
            std::printf("{\"phase\":\"cuda_cross_fitted_baseline_reference_check\",\"target_mode\":\"complete_mc_cross_fitted_state_baseline\",\"rows\":%zu,\"eligible_rows\":%zu,\"max_abs_error\":0,\"returns_unchanged\":true,\"baseline_sha256\":\"%s\",\"protocol_sha256\":\"%s\",\"advantage_normalization\":false,\"eligible_advantage_mean\":%.12g,\"eligible_advantage_mse\":%.12g,\"eligible_advantage_variance\":%.12g}\n",
                input.size(), eligible, state_baseline->sha256.c_str(), state_baseline->protocol_sha256.c_str(), residual_mean,
                residual_sumsq / eligible, std::max(0., residual_sumsq / eligible - residual_mean * residual_mean));
        }
    }
    ~GaeTargets() { cudaFree(advantages); cudaFree(returns); }
};
void cross_fitted_baseline_gpu_test() {
    rek_authentic::Dataset data; data.rows.resize(2); data.sequences.push_back({0,2,0,0});
    data.rows[0].reward=1; data.rows[0].gamma=.5f; data.rows[0].lambda=1; data.rows[0].policy_weight=1;
    data.rows[1].reward=2; data.rows[1].gamma=.5f; data.rows[1].lambda=1; data.rows[1].terminal_after=1;
    rek_authentic::Replay replay; replay.rows.resize(2); replay.rows[0].old_value=3; replay.rows[1].old_value=-4;
    GaeTargets zero(data,replay,true); std::vector<float> old_adv(2),old_ret(2);
    ck(cudaMemcpy(old_adv.data(),zero.advantages,2*sizeof(float),cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(old_ret.data(),zero.returns,2*sizeof(float),cudaMemcpyDeviceToHost));
    require(old_adv==std::vector<float>({2,2}) && old_ret==old_adv, "unchanged zero-target CUDA self-test failed");
    rek_crossfit_baseline::Baseline baseline; baseline.returns={2,2}; baseline.prediction={.25f,.5f}; baseline.residual={1.75f,1.5f};
    GaeTargets residual(data,replay,true,&baseline);
    std::vector<float> after_adv(2),after_ret(2);
    ck(cudaMemcpy(after_adv.data(),residual.advantages,2*sizeof(float),cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(after_ret.data(),residual.returns,2*sizeof(float),cudaMemcpyDeviceToHost));
    require(after_adv==baseline.residual && after_ret==old_ret, "cross-fitted CUDA self-test failed");
    std::printf("{\"phase\":\"cross_fitted_baseline_gpu_self_test\",\"passed\":true,\"zero_path_unchanged\":true,\"excluded_terminal_reward_retained\":true,\"optimizer_updates\":0}\n");
}
__global__ void zero_excluded_ppo_gradients(float* actor, float* critic,
        const float* policy_weight, const float* value_weight, int T) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    for (int a = 0; a < A; ++a) actor[t * A + a] *= policy_weight[t];
    critic[t] *= value_weight[t];
}
struct PpoWork {
    int T;
    Allocator alloc{};
    PPOBufs buffers{};
    TrainGraph graph{};
    Float old_logprobs{}, old_values{}, actions{}, policy_weight{}, value_weight{};
    Prec old_rounded{}, advantages{}, returns{}, legal{}, importance{}, current_value{};
    int* act_sizes = nullptr;
    PpoWork(int horizon, float entropy) : T(horizon) {
        register_ppo_buffers(buffers, &alloc, 1, T, A, false);
        old_logprobs = {.shape = {1, T}}; actions = {.shape = {1, T, 1}};
        policy_weight = {.shape = {1, T}}; value_weight = {.shape = {1, T}};
        old_rounded = {.shape = {1, T}}; advantages = {.shape = {1, T}};
        old_values = {.shape = {1, T}}; returns = {.shape = {1, T}};
        legal = {.shape = {1, T, A}}; importance = {.shape = {1, T}}; current_value = {.shape = {1, T}};
        for (Float* t : {&old_logprobs, &old_values, &actions, &policy_weight, &value_weight}) alloc_register(&alloc, t);
        for (Prec* t : {&old_rounded, &advantages, &returns, &legal, &importance, &current_value}) alloc_register(&alloc, t);
        alloc_create(&alloc);
        ck(cudaMalloc(&act_sizes, sizeof(int))); const int classes = A;
        ck(cudaMemcpy(act_sizes, &classes, sizeof(int), cudaMemcpyHostToDevice));
        ck(cudaMemcpy(buffers.ent_coef, &entropy, sizeof(float), cudaMemcpyHostToDevice));
    }
    ~PpoWork() { cudaDeviceSynchronize(); cudaFree(act_sizes); cudaFree(buffers.ent_coef); cudaFree(alloc.mem); free(alloc.regs); }
    void load(const rek_authentic::Dataset& data, const rek_authentic::Replay& replay,
            const GaeTargets& targets, size_t begin, size_t end) {
        std::vector<float> lp(T, 0), values(T, 0), acts(T, 0), pw(T, 0), vw(T, 0);
        std::vector<precision_t> rounded(T, from_float(0)), mask(T * A, from_float(1));
        for (size_t i = begin; i < end; ++i) {
            const auto& row = data.rows[i]; const auto& old = replay.rows[i]; const int t = int(i - begin);
            lp[t] = old.old_logprob; acts[t] = float(row.action); pw[t] = row.policy_weight; vw[t] = row.value_weight;
            rounded[t] = from_float(old.old_logprob); values[t] = old.old_value;
            for (int a = 0; a < A; ++a) mask[t * A + a] = from_float(row.support[a]);
        }
        auto copy_float = [&](Float dst, const auto& host) { ck(cudaMemcpy(dst.data, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice)); };
        auto copy_prec = [&](Prec dst, const auto& host) { ck(cudaMemcpy(dst.data, host.data(), host.size() * sizeof(precision_t), cudaMemcpyHostToDevice)); };
        copy_float(old_logprobs, lp); copy_float(actions, acts); copy_float(policy_weight, pw); copy_float(value_weight, vw);
        copy_float(old_values, values); copy_prec(old_rounded, rounded); copy_prec(legal, mask);
        authentic_target_slice<<<grid_size(T), BLOCK_SIZE>>>(advantages.data, returns.data,
            targets.advantages, targets.returns, int(begin), int(end - begin), T);
    }
    void cache(Prec logits) {
        logits.shape[0] = 1; logits.shape[1] = T; logits.shape[2] = A + 1;
        // Only the unused importance buffer uses the native BF16 old-logp argument.
        // The actual PPO subtraction below reads immutable FP32 behavior logp.
        cache_imp_and_v<<<grid_size(T), BLOCK_SIZE>>>(logits, actions.data, old_rounded.data,
            legal.data, Prec{}, act_sizes, importance.data, current_value.data,
            buffers.grad_logits.data, buffers.grad_values.data);
        ck(cudaGetLastError());
    }
    void loss(Prec logits, float clip, float vf_clip, float vf_coef) {
        PPOGraphArgsFp32 g = {.imp = importance.data, .actions = actions.data,
            .old_logprobs = old_logprobs.data, .advantages = advantages.data,
            .values = old_values.data, .returns = returns.data};
        PPOKernelArgs a = {.grad_logits = buffers.grad_logits.data, .grad_logstd = nullptr,
            .grad_values_pred = buffers.grad_values.data, .logits = logits.data,
            .logstd = nullptr, .values_pred = logits.data + A, .act_sizes = act_sizes,
            .action_mask = legal.data, .num_atns = 1, .clip_coef = clip,
            .vf_clip_coef = vf_clip, .vf_coef = vf_coef, .ent_coef = buffers.ent_coef,
            .T_seq = T, .A_total = A, .N = 1, .is_continuous = false};
        ppo_loss_compute_fp32<<<(T + PPO_THREADS - 1) / PPO_THREADS, PPO_THREADS>>>(buffers.ppo_partials.data, a, g);
        zero_excluded_ppo_gradients<<<grid_size(T), BLOCK_SIZE>>>(buffers.grad_logits.data,
            buffers.grad_values.data, policy_weight.data, value_weight.data, T);
        ck(cudaGetLastError());
    }
};
struct Parity {
    double max_logit_error = 0, max_value_error = 0, max_logp_error = 0, max_ratio_error = 0;
    size_t worst_row = 0, rows = 0;
};
std::pair<double, double> sequential_teacher_probe(Trainer& engine, const rek_authentic::Dataset& data,
        const rek_authentic::Replay& replay) {
    // Diagnostic only: existing Puffer one-step forward, no new backward/cache path.
    Allocator alloc{};
    Activations rollout = arch_reg_rollout(&engine.arch, engine.w, &alloc, 1);
    Prec obs = {.shape = {1, O}}, state = {.shape = {L, 1, H}};
    alloc_register(&alloc, &obs); alloc_register(&alloc, &state); alloc_create(&alloc);
    double max_logit = 0, max_value = 0;
    std::vector<precision_t> input(O), output(A + 1);
    for (size_t i = 0; i < data.rows.size(); ++i) {
        const auto& row = data.rows[i];
        if (row.reset) ck(cudaMemset(state.data, 0, L * H * sizeof(precision_t)));
        for (int j = 0; j < O; ++j) input[j] = from_float(row.obs[j]);
        ck(cudaMemcpy(obs.data, input.data(), O * sizeof(precision_t), cudaMemcpyHostToDevice));
        Prec z = arch_forward(&engine.arch, engine.w, rollout, obs, state, 0);
        ck(cudaMemcpy(output.data(), z.data, output.size() * sizeof(precision_t), cudaMemcpyDeviceToHost));
        for (int j = 0; j < A; ++j) max_logit = std::max(max_logit, std::abs(double(to_float(output[j])) - replay.rows[i].logits[j]));
        max_value = std::max(max_value, std::abs(double(to_float(output[A])) - replay.rows[i].old_value));
    }
    auto* network = static_cast<MinGRUActivations*>(rollout.network);
    free(network->combined); free(rollout.encoder); free(rollout.decoder); free(rollout.network);
    cudaFree(alloc.mem); free(alloc.regs);
    std::printf("{\"phase\":\"puffer_sequential_teacher_diagnostic\",\"rows\":%zu,\"max_logit_abs_error\":%.12g,\"max_value_abs_error\":%.12g,\"optimizer_updates\":0}\n", data.rows.size(), max_logit, max_value);
    std::fflush(stdout);
    return {max_logit, max_value};
}
Parity parity(Trainer& engine, PpoWork& work, const rek_authentic::Dataset& data,
        const rek_authentic::Replay& replay, const AuthenticData& prepared, const GaeTargets& targets,
        float clip, bool initial = true, bool allow_bf16_batch = false, bool allow_distributional = false) {
    Parity result;
    std::vector<double> ratios, deviations;
    std::vector<float> target_values(data.rows.size());
    std::vector<float> target_advantages(data.rows.size());
    ck(cudaMemcpy(target_values.data(), targets.returns, target_values.size() * sizeof(float), cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(target_advantages.data(), targets.advantages, target_advantages.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double legal_kl = 0, maximum_kl = 0, old_value_mse = 0, new_value_mse = 0;
    double mean_old_value = 0, mean_new_value = 0, mean_target = 0, approx_kl = 0;
    size_t clipped = 0;
    double absolute_advantage=0,absolute_surrogate_error=0;
    for (const auto& seq : data.sequences) {
        engine.reset();
        for (size_t p = seq.begin; p < seq.end; p += engine.T) {
            const size_t end = std::min(p + engine.T, seq.end);
            work.load(data, replay, targets, p, end);
            Prec logits = engine.forward(prepared.history, p, end, false); work.cache(logits);
            std::vector<precision_t> z(engine.T * (A + 1)); std::vector<float> lp(engine.T);
            ck(cudaMemcpy(z.data(), logits.data, z.size() * sizeof(precision_t), cudaMemcpyDeviceToHost));
            ck(cudaMemcpy(lp.data(), work.buffers.grad_values.data, lp.size() * sizeof(float), cudaMemcpyDeviceToHost));
            for (size_t i = p; i < end; ++i) {
                const int t = int(i - p); const auto& old = replay.rows[i];
                for (int a = 0; a < A; ++a) result.max_logit_error = std::max(result.max_logit_error,
                    std::abs(double(to_float(z[t * (A + 1) + a])) - old.logits[a]));
                result.max_value_error = std::max(result.max_value_error, std::abs(double(to_float(z[t * (A + 1) + A])) - old.old_value));
                const double difference = double(lp[t]) - old.old_logprob;
                require(std::isfinite(difference), "nonfinite initial behavior ratio");
                result.max_logp_error = std::max(result.max_logp_error, std::abs(difference));
                const double error = std::abs(std::exp(difference) - 1);
                if (error > result.max_ratio_error) { result.max_ratio_error = error; result.worst_row = i; }
                const double ratio = std::exp(difference);
                const double weighted_advantage=data.rows[i].policy_weight*std::abs(double(target_advantages[i]));
                absolute_advantage+=weighted_advantage; absolute_surrogate_error+=weighted_advantage*error;
                ratios.push_back(ratio); deviations.push_back(error);
                clipped += error > clip; approx_kl += ratio - 1 - difference;
                double max_old = -INFINITY, max_new = -INFINITY, sum_old = 0, sum_new = 0;
                for (int a = 0; a < A; ++a) if (data.rows[i].support[a]) {
                    max_old = std::max(max_old, double(old.logits[a]));
                    max_new = std::max(max_new, double(to_float(z[t * (A + 1) + a])));
                }
                for (int a = 0; a < A; ++a) if (data.rows[i].support[a]) {
                    sum_old += std::exp(old.logits[a] - max_old);
                    sum_new += std::exp(double(to_float(z[t * (A + 1) + a])) - max_new);
                }
                const double lse_old = max_old + std::log(sum_old), lse_new = max_new + std::log(sum_new);
                double kl = 0;
                for (int a = 0; a < A; ++a) if (data.rows[i].support[a]) {
                    const double old_lp = old.logits[a] - lse_old;
                    kl += std::exp(old_lp) * (old_lp - double(to_float(z[t * (A + 1) + a])) + lse_new);
                }
                require(std::isfinite(kl) && kl>=-1e-12,"invalid full legal-distribution KL");
                kl=std::max(0.0,kl); legal_kl += kl; maximum_kl = std::max(maximum_kl, kl);
                const double value = to_float(z[t * (A + 1) + A]), target = target_values[i];
                old_value_mse += (old.old_value - target) * (old.old_value - target);
                new_value_mse += (value - target) * (value - target);
                mean_old_value += old.old_value; mean_new_value += value; mean_target += target;
                ++result.rows;
            }
            engine.carry();
        }
    }
    std::sort(ratios.begin(), ratios.end()); std::sort(deviations.begin(), deviations.end());
    const auto q = [](const auto& xs, double p) { return xs[size_t((xs.size() - 1) * p)]; };
    std::printf("{\"phase\":\"%s\",\"rows\":%zu,\"max_logit_abs_error\":%.12g,\"max_value_abs_error\":%.12g,\"max_logp_abs_error\":%.12g,\"max_ratio_abs_error\":%.12g,\"worst_ratio_row\":%zu,\"ratio_tolerance\":0.0001}\n",
        initial ? "initial_teacher_train_parity" : "post_update_frozen_behavior", result.rows, result.max_logit_error,
        result.max_value_error, result.max_logp_error, result.max_ratio_error, result.worst_row);
    std::printf("{\"phase\":\"%s_distribution_diagnostics\",\"rows\":%zu,\"mean_legal_kl_old_to_current\":%.12g,\"max_legal_kl\":%.12g,\"chosen_approx_kl\":%.12g,\"ratio_min\":%.12g,\"ratio_p50\":%.12g,\"ratio_p95\":%.12g,\"ratio_max\":%.12g,\"absolute_ratio_error_p50\":%.12g,\"absolute_ratio_error_p95\":%.12g,\"clipped_fraction\":%.12g,\"mean_old_value\":%.12g,\"mean_current_value\":%.12g,\"mean_target_return\":%.12g,\"old_value_mse_to_return\":%.12g,\"current_value_mse_to_return\":%.12g}\n",
        initial ? "initial" : "post_update", result.rows, legal_kl / result.rows, maximum_kl, approx_kl / result.rows,
        ratios.front(), q(ratios, .5), q(ratios, .95), ratios.back(), q(deviations, .5), q(deviations, .95),
        double(clipped) / result.rows, mean_old_value / result.rows, mean_new_value / result.rows,
        mean_target / result.rows, old_value_mse / result.rows, new_value_mse / result.rows);
    std::fflush(stdout);
    if (initial) {
        rek_authentic_parity::Evidence evidence{result.max_logit_error, result.max_value_error,
            result.max_ratio_error, 0, 0, double(clipped) / result.rows, clip};
        if (allow_bf16_batch || allow_distributional || !rek_authentic_parity::accepts(evidence, false)) {
            const auto exact = sequential_teacher_probe(engine, data, replay);
            evidence.sequential_logit_error = exact.first; evidence.sequential_value_error = exact.second;
        }
        const double relative_surrogate=absolute_advantage>0 ? absolute_surrogate_error/absolute_advantage : 0;
        const rek_authentic_parity::DistributionalEvidence distributional{evidence,legal_kl/result.rows,maximum_kl,relative_surrogate};
        const bool accepted = allow_distributional ? rek_authentic_parity::accepts_distributional(distributional)
            : rek_authentic_parity::accepts(evidence, allow_bf16_batch);
        if(allow_distributional) std::printf("{\"phase\":\"distributional_acceptance_budget\",\"mean_legal_kl\":%.12g,\"mean_legal_kl_limit\":%.12g,\"maximum_legal_kl\":%.12g,\"maximum_legal_kl_limit\":%.12g,\"relative_absolute_surrogate_error\":%.12g,\"relative_absolute_surrogate_error_limit\":%.12g,\"mean_absolute_surrogate_error\":%.12g,\"exact_sequential_required\":true,\"zero_initial_clipping_required\":true,\"true_behavior_logprobs_unchanged\":true}\n",
            distributional.mean_legal_kl,rek_authentic_parity::kMeanLegalKlLimit,maximum_kl,rek_authentic_parity::kMaximumLegalKlLimit,
            relative_surrogate,rek_authentic_parity::kRelativeSurrogateLimit,absolute_surrogate_error/result.rows);
        std::printf("{\"phase\":\"initial_numerical_acceptance\",\"mode\":\"%s\",\"accepted\":%s,\"max_chosen_ratio_error_limit\":%.12g,\"exact_batch_parity_claimed\":false}\n",
            allow_distributional ? "explicit_distributional_bf16_approximation" : allow_bf16_batch ? "explicit_bounded_bf16_batch_approximation" : "strict_default",
            accepted ? "true" : "false", allow_distributional ? clip : allow_bf16_batch ? .1 * clip : 1e-4);
        std::fflush(stdout);
        require(accepted, "initial teacher/train numerical acceptance failed; no optimizer update permitted");
    }
    return result;
}
void optimize_epoch(Trainer& engine, PpoWork& work, const rek_authentic::Dataset& data,
        const rek_authentic::Replay& replay, const AuthenticData& prepared,
        const GaeTargets& targets, float clip, float vf_clip, float vf_coef, int epoch) {
    double policy_loss = 0, value_mse = 0, approximate_kl = 0, clip_fraction = 0, policy_weight = 0, value_weight = 0;
    for (const auto& seq : data.sequences) for (size_t p = seq.begin; p < seq.end; p += engine.T) {
        const size_t end = std::min(p + engine.T, seq.end);
        engine.burn_in(prepared.history, seq.begin, p);
        work.load(data, replay, targets, p, end);
        Prec logits = engine.forward(prepared.history, p, end, false); work.cache(logits);
        std::vector<float> lp(engine.T); std::vector<precision_t> z(engine.T * (A + 1));
        ck(cudaMemcpy(lp.data(), work.buffers.grad_values.data, lp.size() * sizeof(float), cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(z.data(), logits.data, z.size() * sizeof(precision_t), cudaMemcpyDeviceToHost));
        std::vector<float> adv_host(end - p), ret_host(end - p);
        ck(cudaMemcpy(adv_host.data(), targets.advantages + p, adv_host.size() * sizeof(float), cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(ret_host.data(), targets.returns + p, ret_host.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = p; i < end; ++i) {
            const int t = int(i - p); const auto& row = data.rows[i];
            const double logratio = double(lp[t]) - replay.rows[i].old_logprob, ratio = std::exp(logratio);
            require(std::isfinite(ratio), "nonfinite PPO ratio");
            const double adv = adv_host[t], clipped = std::clamp(ratio, 1.0 - clip, 1.0 + clip);
            policy_loss += row.policy_weight * std::max(-adv * ratio, -adv * clipped);
            approximate_kl += row.policy_weight * (ratio - 1 - logratio);
            clip_fraction += row.policy_weight * (std::abs(ratio - 1) > clip);
            policy_weight += row.policy_weight;
            const double value_error = double(to_float(z[t * (A + 1) + A])) - ret_host[t];
            value_mse += row.value_weight * value_error * value_error; value_weight += row.value_weight;
        }
        work.loss(logits, clip, vf_clip, vf_coef);
        ck(cudaMemset(engine.grads.mem, 0, engine.grads.total_bytes));
        auto* a = static_cast<MinGRUActivations*>(engine.train.network);
        ck(cudaMemset(a->grad_next_state.data, 0, H * sizeof(precision_t)));
        arch_backward(&engine.arch, engine.w, engine.train, work.buffers.grad_logits, Float{}, work.buffers.grad_values, 0);
        ck(cudaDeviceSynchronize());
        muon_step(&engine.optimizer, engine.master, engine.flat_grads, 1, 0); engine.refresh();
        ck(cudaDeviceSynchronize()); ++engine.updates;
    }
    std::printf("{\"phase\":\"epoch\",\"epoch\":%d,\"updates\":%d,\"policy_loss\":%.9g,\"value_mse\":%.9g,\"approximate_kl\":%.9g,\"clip_fraction\":%.9g,\"policy_rows\":%.9g,\"value_rows\":%.9g,\"behavior_logprobs_unchanged\":true}\n",
        epoch, engine.updates, policy_loss / policy_weight, value_mse / value_weight,
        approximate_kl / policy_weight, clip_fraction / policy_weight, policy_weight, value_weight);
    std::fflush(stdout);
}
float number(const char* text, float low, float high) {
    char* end = nullptr; const float n = std::strtof(text, &end);
    require(end != text && *end == 0 && std::isfinite(n) && n >= low && n <= high, "invalid numeric argument");
    return n;
}
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--cross-fitted-baseline-gpu-self-test") { cross_fitted_baseline_gpu_test(); return 0; }
        require(argc >= 13 && argc <= 19, "Usage: authentic-ppo DATA REPLAY INITIAL SHA256 NEW_OUTPUT EPOCHS LR HORIZON CLIP VF_CLIP VF_COEF ENT_COEF [--allow-bounded-bf16-batch|--allow-distributional-bf16-batch] [--targets=complete-mc-zero-baseline|--targets=complete-mc-cross-fitted-state-baseline] [--observation-schema=SCHEMA] [--state-baseline=PATH --state-baseline-sha256=SHA --state-baseline-protocol-sha256=SHA]");
        bool allow_bf16_batch = false, allow_distributional = false, complete_mc_zero_baseline = false, owned_yaw = false, balance8 = false;
        bool cross_fitted_baseline = false;
        std::string baseline_path, baseline_sha, baseline_protocol;
        for (int i = 13; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--allow-bounded-bf16-batch" && !allow_bf16_batch) allow_bf16_batch = true;
            else if (option == "--allow-distributional-bf16-batch" && !allow_distributional) allow_distributional = true;
            else if (option == "--targets=complete-mc-zero-baseline" && !complete_mc_zero_baseline) complete_mc_zero_baseline = true;
            else if (option == "--targets=complete-mc-cross-fitted-state-baseline" && !cross_fitted_baseline) cross_fitted_baseline = true;
            else if (option.rfind("--state-baseline=",0)==0 && baseline_path.empty()) baseline_path=option.substr(17);
            else if (option.rfind("--state-baseline-sha256=",0)==0 && baseline_sha.empty()) baseline_sha=option.substr(24);
            else if (option.rfind("--state-baseline-protocol-sha256=",0)==0 && baseline_protocol.empty()) baseline_protocol=option.substr(33);
            else if (option == std::string("--observation-schema=") + rek_owned_yaw::kSchema && !owned_yaw) owned_yaw = true;
            else if (option == std::string("--observation-schema=") + rek_balance8::kSchema && !balance8) balance8 = true;
            else require(false, "unknown or duplicate optional mode");
        }
        require(!(allow_bf16_batch && allow_distributional),"numerical approximation modes are mutually exclusive");
        require(!(complete_mc_zero_baseline && cross_fitted_baseline), "target modes are mutually exclusive");
        require(cross_fitted_baseline ? !baseline_path.empty() && !baseline_sha.empty() && !baseline_protocol.empty()
            : baseline_path.empty() && baseline_sha.empty() && baseline_protocol.empty(), "baseline options require explicit cross-fitted target mode and all three pins");
        const bool complete_mc = complete_mc_zero_baseline || cross_fitted_baseline;
        const auto data = owned_yaw ? rek_owned_yaw_trajectory::load(argv[1]) : rek_authentic::load(argv[1]);
        require(!(owned_yaw&&balance8) && (data.format_version!=4 || balance8) && (!balance8 || data.format_version>=3),"explicit identity-bound balance8 schema required");
        const auto replay = owned_yaw ? rek_owned_yaw_trajectory::decode_replay(rek_authentic::read_file(argv[2]), data)
            : rek_authentic::load_replay(argv[2], data);
        require((data.format_version==4?data.derived_teacher_sha256:replay.checkpoint_sha256) == argv[4], "initial checkpoint identity mismatch");
        const auto initial = read_checkpoint(argv[3], argv[4]); const auto prepared = prepare(data, replay);
        const float epoch_number = number(argv[6], 0, 2), horizon_number = number(argv[8], 4, 256);
        require(std::floor(epoch_number) == epoch_number && std::floor(horizon_number) == horizon_number, "epochs/horizon must be integers");
        const int epochs = int(epoch_number), horizon = int(horizon_number);
        const float lr = number(argv[7], 1e-8f, 0.001f), clip = number(argv[9], .001f, .4f);
        const float vf_clip = number(argv[10], .001f, 10), vf_coef = number(argv[11], 0, 10), entropy = number(argv[12], 0, .1f);
        require(!complete_mc || vf_coef == 0, "complete MC targets require vf_coef=0");
        require(data.format_version!=5 || complete_mc,"score-delta five-second experiment requires complete MC targets");
        require(!cross_fitted_baseline || (balance8 && data.format_version==5), "cross-fitted baseline requires explicit V5 balance8 schema");
        std::optional<rek_crossfit_baseline::Baseline> state_baseline;
        if (cross_fitted_baseline) state_baseline=rek_crossfit_baseline::load(baseline_path.c_str(),data,baseline_sha,baseline_protocol);
        struct stat st{};
        require(stat(argv[5], &st) != 0 && errno == ENOENT, "output exists or cannot be inspected");
        for (int e = 1; e <= epochs; ++e) {
            const std::string path = std::string(argv[5]) + ".epoch-" + std::to_string(e) + ".bin";
            require(stat(path.c_str(), &st) != 0 && errno == ENOENT, "epoch output exists or cannot be inspected");
        }
        if (cross_fitted_baseline) cross_fitted_baseline_gpu_test();
        Trainer engine(initial, horizon, lr); PpoWork work(horizon, entropy);
        GaeTargets targets(data, replay, complete_mc, state_baseline ? &*state_baseline : nullptr);
        if (owned_yaw) std::printf("{\"observation_schema\":\"%s\",\"trajectory_format\":\"REKRL002\",\"replay_format\":\"REKBR002\"}\n", rek_owned_yaw::kSchema);
        if (balance8 && data.format_version==3) std::printf("{\"observation_schema\":\"%s\",\"trajectory_format\":\"REKRL003\",\"replay_format\":\"REKBR003\",\"teacher_was_recorded_worker\":true}\n",rek_balance8::kSchema);
        if (balance8 && data.format_version==5) std::printf("{\"observation_schema\":\"%s\",\"trajectory_format\":\"REKRL005\",\"replay_format\":\"REKBR005\",\"teacher_was_recorded_worker\":true,\"reward_contract\":\"received-score-delta-div5\",\"discount_half_life_seconds\":5,\"terminal_bonus\":0,\"score_potential\":false}\n",rek_balance8::kSchema);
        if (balance8 && data.format_version==4) std::printf("{\"observation_schema\":\"%s\",\"trajectory_format\":\"REKRL004\",\"replay_format\":\"REKBR004\",\"derived_initial_teacher_sha256\":\"%s\",\"original_dataset_sha256\":\"%s\",\"original_replay_sha256\":\"%s\",\"derived_teacher_was_recorded_worker\":false}\n",rek_balance8::kSchema,data.derived_teacher_sha256.c_str(),data.original_dataset_sha256.c_str(),data.original_replay_sha256.c_str());
        std::printf("{\"schema\":\"rek.authentic_trajectory_ppo.v1\",\"rows\":%zu,\"rounds\":%zu,\"epochs\":%d,\"horizon\":%d,\"learning_rate\":%.9g,\"clip\":%.9g,\"vf_clip\":%.9g,\"vf_coef\":%.9g,\"entropy\":%.9g,\"old_logprob_dtype\":\"fp32\",\"behavior_checkpoint_sha256\":\"%s\",\"environment_stepping\":false,\"heldout\":false}\n",
            data.rows.size(), data.sequences.size(), epochs, horizon, lr, clip, vf_clip, vf_coef, entropy, replay.checkpoint_sha256.c_str());
        std::printf("{\"old_value_dtype\":\"fp32\",\"target_producer\":\"cuda_fp64_recurrence_fp32_output\",\"loss_advantage_return_dtype\":\"bf16_native\",\"advantage_normalization\":\"none_pinned_native_convention\"}\n");
        const auto replay_bytes = rek_authentic::read_file(argv[2]);
        std::printf("{\"dataset_sha256\":\"%s\",\"behavior_replay_sha256\":\"%s\",\"seed\":%llu}\n", data.digest.c_str(),
            rek_authentic::sha256(replay_bytes.data(), replay_bytes.size()).c_str(), static_cast<unsigned long long>(replay.seed));
        parity(engine, work, data, replay, prepared, targets, clip, true, allow_bf16_batch, allow_distributional);
        for (int e = 1; e <= epochs; ++e) {
            optimize_epoch(engine, work, data, replay, prepared, targets, clip, vf_clip, vf_coef, e);
            parity(engine, work, data, replay, prepared, targets, clip, false);
            const auto checkpoint = engine.checkpoint();
            const std::string path = std::string(argv[5]) + ".epoch-" + std::to_string(e) + ".bin";
            write_checkpoint(path.c_str(), checkpoint);
            std::printf("{\"phase\":\"epoch_saved\",\"epoch\":%d,\"sha256\":\"%s\"}\n", e, digest(checkpoint.data(), checkpoint.size() * sizeof(float)).c_str());
        }
        const auto checkpoint = engine.checkpoint(); write_checkpoint(argv[5], checkpoint);
        std::printf("{\"phase\":\"saved\",\"sha256\":\"%s\"}\n", digest(checkpoint.data(), checkpoint.size() * sizeof(float)).c_str());
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "authentic_ppo_error: %s\n", e.what()); return 2; }
}
