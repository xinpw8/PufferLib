// Native command-intent BC, using the prepared PufferLib5 kernels and optimizer.
// No simulator, game bridge, Python, or alternate checkpoint architecture.
#include "puffer5_bc_core.cuh"
#include "bc_dataset.h"
#include <openssl/sha.h>
#include <algorithm>
#include <string>

namespace {
using rek_bc::require;
constexpr int O = rek_bc::OBS, A = rek_bc::ACTIONS, H = 256, L = 2;
constexpr int PARAMS = O * H + (A + 1) * H + L * 3 * H * H;
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
std::string digest(const void* data, size_t size) {
    unsigned char sum[SHA256_DIGEST_LENGTH]; SHA256(static_cast<const unsigned char*>(data), size, sum);
    char result[65]; for (int i = 0; i < 32; ++i) std::snprintf(result + 2 * i, 3, "%02x", sum[i]);
    return result;
}
std::vector<float> read_checkpoint(const char* path, const char* expected) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    require(bool(f) && f.tellg() == PARAMS * int64_t(sizeof(float)), "checkpoint shape/size mismatch");
    std::vector<float> out(PARAMS); f.seekg(0); f.read(reinterpret_cast<char*>(out.data()), PARAMS * sizeof(float));
    require(bool(f), "checkpoint read failed");
    require(digest(out.data(), out.size() * sizeof(float)) == expected, "checkpoint SHA256 mismatch");
    for (float v : out) require(std::isfinite(v), "nonfinite checkpoint");
    return out;
}
void write_checkpoint(const char* path, const std::vector<float>& weights) {
    for (float v : weights) require(std::isfinite(v), "nonfinite updated checkpoint");
    // Same flat FP32 order as puf_save_weights, with exclusive publication.
    FILE* f = std::fopen(path, "wbx"); require(f != nullptr, "output exists or cannot be created");
    const size_t n = std::fwrite(weights.data(), sizeof(float), weights.size(), f);
    const int closed = std::fclose(f);
    require(n == weights.size() && closed == 0, "checkpoint write failed");
}

// One thread per timestep, deterministic row loss and masked softmax derivative.
// Zero-weight/burn-in rows contribute no direct gradient. They still affect RNN.
__global__ void bc_loss(const precision_t* logits, const float* support,
        const int* targets, const float* weights, float denominator,
        float* gradients, float* metrics, int T) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    float* g = gradients + t * A;
    for (int a = 0; a < A; ++a) g[a] = 0;
    for (int k = 0; k < 3; ++k) metrics[t * 3 + k] = 0;
    if (weights[t] == 0) return;
    float maximum = -INFINITY; int best = -1;
    for (int a = 0; a < A; ++a) if (support[t * A + a] > 0) {
        const float z = to_float(logits[t * (A + 1) + a]);
        if (z > maximum) { maximum = z; best = a; }
    }
    float total = 0;
    for (int a = 0; a < A; ++a) if (support[t * A + a] > 0)
        total += expf(to_float(logits[t * (A + 1) + a]) - maximum);
    const float target_logit = to_float(logits[t * (A + 1) + targets[t]]);
    metrics[t * 3] = weights[t] * (logf(total) + maximum - target_logit);
    metrics[t * 3 + 1] = weights[t] * (best == targets[t]);
    metrics[t * 3 + 2] = weights[t];
    for (int a = 0; a < A; ++a) if (support[t * A + a] > 0) {
        const float p = expf(to_float(logits[t * (A + 1) + a]) - maximum) / total;
        g[a] = weights[t] / denominator * (p - (a == targets[t]));
    }
}
struct Metrics {
    double loss = 0, correct = 0, weight = 0;
    std::array<double, A> action_loss{}, action_correct{}, action_weight{};
    std::array<double, 6> category_loss{}, category_correct{}, category_weight{};
    void add(const std::vector<float>& rows) {
        for (size_t i = 0; i < rows.size(); i += 3) {
            require(std::isfinite(rows[i]) && std::isfinite(rows[i + 1]) && std::isfinite(rows[i + 2]), "nonfinite loss");
            loss += rows[i]; correct += rows[i + 1]; weight += rows[i + 2];
        }
    }
    double ce() const { return weight ? loss / weight : 0; }
};
int category(int action) {
    // Native vocabulary: 16..19 kicks/knee, 20..31 punches, 32 emote.
    return action == 0 ? 0 : action == 1 ? 1 : action < 16 ? 2 : action < 20 ? 3 : action < 32 ? 4 : 5;
}
struct TrainingNormalization {
    size_t rows = 0;
    double weight = 0;
    float denominator = 1;
};
TrainingNormalization normalization(const rek_bc::Dataset& d, int horizon) {
    TrainingNormalization out;
    for (const auto& r : d.rows) if (r.split == 0) { ++out.rows; out.weight += r.weight; }
    if (out.weight > 0) {
        require(out.rows > 0, "training weight without rows");
        out.denominator = float(out.weight / double(out.rows) * horizon);
        require(std::isfinite(out.denominator) && out.denominator > 0, "invalid global training normalization");
    }
    return out;
}
struct Trainer {
    int T;
    Arch arch{}; Weights w{}; Activations train{};
    Allocator params{}, grads{}, acts{};
    Muon optimizer{};
    Float master{}, grad_logits{}, grad_value{};
    Prec state{}, observations{}, terminals{}, flat_params{}, flat_grads{};
    float *support = nullptr, *weights = nullptr, *metrics = nullptr;
    int* targets = nullptr;
    int updates = 0;
    Trainer(const std::vector<float>& initial, int horizon, float lr) : T(horizon) {
        require(T >= 4 && T <= 256 && T % 4 == 0, "horizon must be a multiple of4 in[4,256]");
        if (!g_cublas_handle) { cublas_init_handle(); ck(cudaDeviceSynchronize()); }
        arch = build_arch("rek_native5", O, H, L, A, false, T);
        w = weights_create(&arch, &params);
        train = arch_reg_train(&arch, w, &acts, &grads, T);
        require(params.total_elems == PARAMS && grads.total_elems == PARAMS, "Puffer5 parameter layout mismatch");
        require(params.total_bytes == PARAMS * sizeof(precision_t)
            && grads.total_bytes == PARAMS * sizeof(precision_t), "unexpected allocator padding");
        state = {.shape = {L, 1, H}}; observations = {.shape = {1, T, O}}; terminals = {.shape = {1, T}};
        master = {.shape = {PARAMS}}; grad_logits = {.shape = {1, T, A}}; grad_value = {.shape = {1, T}};
        alloc_register(&acts, &state); alloc_register(&acts, &observations); alloc_register(&acts, &terminals);
        alloc_register(&acts, &master); alloc_register(&acts, &grad_logits); alloc_register(&acts, &grad_value);
        muon_init(&optimizer, &params, 0.95, &acts);
        alloc_create(&params); alloc_create(&grads); alloc_create(&acts);
        flat_params = {.data = static_cast<precision_t*>(params.mem), .shape = {PARAMS}};
        flat_grads = {.data = static_cast<precision_t*>(grads.mem), .shape = {PARAMS}};
        ck(cudaMalloc(&support, T * A * sizeof(float))); ck(cudaMalloc(&weights, T * sizeof(float)));
        ck(cudaMalloc(&targets, T * sizeof(int))); ck(cudaMalloc(&metrics, T * 3 * sizeof(float)));
        ck(cudaMemcpy(master.data, initial.data(), PARAMS * sizeof(float), cudaMemcpyHostToDevice));
        ck(cudaMemcpy(optimizer.lr, &lr, sizeof(lr), cudaMemcpyHostToDevice));
        refresh(); ck(cudaDeviceSynchronize());
    }
    Trainer(const Trainer&) = delete;
    ~Trainer() {
        cudaDeviceSynchronize();
        auto* n = static_cast<MinGRUWeights*>(w.network); free(n->weights);
        free(w.encoder); free(w.decoder); free(w.network);
        auto* a = static_cast<MinGRUActivations*>(train.network);
        free(a->saved_inputs); free(a->scan_bufs); free(a->combined_bufs); free(a->wgrad_scratch);
        free(train.encoder); free(train.decoder); free(train.network);
        for (Allocator* p : {&params, &grads, &acts}) { cudaFree(p->mem); free(p->regs); }
        for (float* p : {support, weights, metrics, optimizer.lr, optimizer.grad_norm, optimizer.ns_norm, optimizer.norm_partials}) cudaFree(p);
        cudaFree(targets);
    }
    void refresh() { cast<<<grid_size(PARAMS), BLOCK_SIZE>>>(flat_params.data, master.data, PARAMS); }
    void reset() { ck(cudaMemset(state.data, 0, L * H * sizeof(precision_t))); }
    Prec forward(const rek_bc::Dataset& d, size_t begin, size_t end, bool labels) {
        require(begin < end && end <= d.rows.size() && end - begin <= size_t(T), "invalid chunk");
        std::vector<precision_t> obs(T * O, from_float(0)), term(T, from_float(1));
        std::vector<float> mask(T * A, 1), label_weights(T, 0);
        std::vector<int> actions(T, 0);
        for (size_t i = begin; i < end; ++i) {
            const auto& row = d.rows[i]; const int t = int(i - begin);
            for (int j = 0; j < O; ++j) obs[t * O + j] = from_float(d.feature_mask[j] ? row.obs[j] : 0);
            term[t] = from_float(float(row.reset));
            if (labels && row.weight > 0) {
                label_weights[t] = row.weight; actions[t] = row.action;
                std::copy(row.support.begin(), row.support.end(), mask.begin() + t * A);
            }
        }
        ck(cudaMemcpy(observations.data, obs.data(), obs.size() * sizeof(precision_t), cudaMemcpyHostToDevice));
        ck(cudaMemcpy(terminals.data, term.data(), term.size() * sizeof(precision_t), cudaMemcpyHostToDevice));
        ck(cudaMemcpy(support, mask.data(), mask.size() * sizeof(float), cudaMemcpyHostToDevice));
        ck(cudaMemcpy(weights, label_weights.data(), label_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
        ck(cudaMemcpy(targets, actions.data(), actions.size() * sizeof(int), cudaMemcpyHostToDevice));
        Prec x = observations;
        Prec h = arch.encoder.forward(w.encoder, train.encoder, *puf_squeeze(&x, 0), 0);
        h = arch.network.forward_train(w.network, *puf_unsqueeze(&h, 0, 1, T), state, terminals, train.network, 0, 0);
        Prec out = arch.decoder.forward(w.decoder, train.decoder, *puf_squeeze(&h, 0), 0);
        ck(cudaGetLastError());
        return out;
    }
    void carry() {
        auto* a = static_cast<MinGRUActivations*>(train.network);
        for (int layer = 0; layer < L; ++layer)
            ck(cudaMemcpy(state.data + layer * H, a->scan_bufs[layer].next_state.data,
                H * sizeof(precision_t), cudaMemcpyDeviceToDevice));
    }
    Metrics loss(Prec logits, float denominator, std::vector<float>* readback = nullptr) {
        bc_loss<<<grid_size(T), BLOCK_SIZE>>>(logits.data, support, targets, weights,
            denominator, grad_logits.data, metrics, T);
        std::vector<float> host(T * 3); ck(cudaMemcpy(host.data(), metrics, host.size() * sizeof(float), cudaMemcpyDeviceToHost));
        Metrics result; result.add(host);
        if (readback) *readback = std::move(host);
        return result;
    }
    void burn_in(const rek_bc::Dataset& d, size_t sequence_begin, size_t chunk_begin) {
        reset();
        for (size_t p = sequence_begin; p < chunk_begin; p += T) {
            // Prefix chunks are full and labels never enter network inputs.
            forward(d, p, p + T, false); carry();
        }
    }
    void epoch(const rek_bc::Dataset& d) {
        const auto norm = normalization(d, T);
        for (const auto& s : d.sequences) if (s.split == 0) {
            for (size_t p = s.begin; p < s.end; p += T) {
                const size_t end = std::min(p + T, s.end);
                float total = 0; for (size_t i = p; i < end; ++i) total += d.rows[i].weight;
                if (!total) continue;
                // Recompute all preceding state with current weights, never stale pre-update state.
                burn_in(d, s.begin, p);
                // Fixed epoch-wide scale, including short chunks. A row's coefficient
                // cannot depend on nearby labels or heldout data.
                Prec logits = forward(d, p, end, true); loss(logits, norm.denominator);
                ck(cudaMemset(grads.mem, 0, grads.total_bytes));
                ck(cudaMemset(grad_value.data, 0, T * sizeof(float)));
                auto* a = static_cast<MinGRUActivations*>(train.network);
                ck(cudaMemset(a->grad_next_state.data, 0, H * sizeof(precision_t)));
                arch_backward(&arch, w, train, grad_logits, Float{}, grad_value, 0);
                ck(cudaDeviceSynchronize());
                muon_step(&optimizer, master, flat_grads, 1.0f, 0); refresh();
                ck(cudaDeviceSynchronize()); ++updates;
            }
        }
    }
    Metrics evaluate(const rek_bc::Dataset& d, uint32_t split) {
        Metrics result;
        for (const auto& s : d.sequences) if (s.split == split) {
            reset();
            for (size_t p = s.begin; p < s.end; p += T) {
                const size_t end = std::min(p + T, s.end);
                Prec logits = forward(d, p, end, true);
                std::vector<float> rows;
                auto m = loss(logits, 1, &rows);
                result.loss += m.loss; result.correct += m.correct; result.weight += m.weight;
                std::vector<precision_t> host_logits(T * (A + 1));
                ck(cudaMemcpy(host_logits.data(), logits.data, host_logits.size() * sizeof(precision_t), cudaMemcpyDeviceToHost));
                for (size_t i = p; i < end; ++i) if (d.rows[i].weight > 0) {
                    const auto& row = d.rows[i]; const int t = int(i - p), target = row.action, group = category(target);
                    result.action_loss[target] += rows[t * 3];
                    result.action_correct[target] += rows[t * 3 + 1]; result.action_weight[target] += row.weight;
                    double maximum = -INFINITY, all = 0, within = 0; int best = -1;
                    for (int c = 0; c < A; ++c) if (row.support[c]) {
                        const double z = to_float(host_logits[t * (A + 1) + c]);
                        if (z > maximum) { maximum = z; best = c; }
                    }
                    for (int c = 0; c < A; ++c) if (row.support[c]) {
                        const double probability = std::exp(double(to_float(host_logits[t * (A + 1) + c])) - maximum);
                        all += probability; if (category(c) == group) within += probability;
                    }
                    result.category_loss[group] += row.weight * std::log(all / within);
                    result.category_correct[group] += row.weight * (category(best) == group);
                    result.category_weight[group] += row.weight;
                }
                carry();
            }
        }
        return result;
    }
    std::vector<float> checkpoint() {
        std::vector<float> result(PARAMS);
        ck(cudaMemcpy(result.data(), master.data, PARAMS * sizeof(float), cudaMemcpyDeviceToHost));
        for (float v : result) require(std::isfinite(v), "nonfinite trained weights");
        return result;
    }
};
void report(const char* phase, int epoch, const Metrics& train, const Metrics& heldout, int updates) {
    std::printf("{\"phase\":\"%s\",\"epoch\":%d,\"updates\":%d,\"train_ce\":%.9g,\"train_accuracy\":%.9g,\"train_weight\":%.9g,\"heldout_ce\":%.9g,\"heldout_accuracy\":%.9g,\"heldout_weight\":%.9g}\n",
        phase, epoch, updates, train.ce(), train.weight ? train.correct / train.weight : 0,
        train.weight, heldout.ce(), heldout.weight ? heldout.correct / heldout.weight : 0, heldout.weight);
    std::fflush(stdout);
}
void report_details(const char* split, int epoch, const Metrics& heldout) {
    std::printf("{\"phase\":\"%s_details\",\"epoch\":%d,\"actions\":[", split, epoch);
    for (int action = 0; action < A; ++action) {
        if (action) std::printf(",");
        std::printf("{\"action\":%d,\"weight\":%.9g,", action, heldout.action_weight[action]);
        if (heldout.action_weight[action]) std::printf("\"ce\":%.9g,\"recall\":%.9g}", heldout.action_loss[action] / heldout.action_weight[action], heldout.action_correct[action] / heldout.action_weight[action]);
        else std::printf("\"ce\":null,\"recall\":null}");
    }
    const char* names[] = {"hold", "neutral_command", "movement", "kick_or_knee", "punch", "emote"};
    std::printf("],\"categories\":[");
    for (int c = 0; c < 6; ++c) {
        if (c) std::printf(",");
        std::printf("{\"category\":\"%s\",\"weight\":%.9g,", names[c], heldout.category_weight[c]);
        if (heldout.category_weight[c]) std::printf("\"ce\":%.9g,\"argmax_category_recall\":%.9g}", heldout.category_loss[c] / heldout.category_weight[c], heldout.category_correct[c] / heldout.category_weight[c]);
        else std::printf("\"ce\":null,\"argmax_category_recall\":null}");
    }
    std::printf("]}\n"); std::fflush(stdout);
}
void self_test(const std::vector<float>& initial) {
    rek_bc::Dataset d; d.feature_mask.fill(1); d.feature_mask[222] = 0;
    for (int s = 0; s < 2; ++s) for (int t = 0; t < 32; ++t) {
        rek_bc::Row r; r.sequence = s; r.split = s; r.reset = t == 0; r.time = t * 0.02;
        r.support.fill(1); r.support[32] = 0;
        for (int j = 0; j < O; ++j) r.obs[j] = 0.08f * std::sin(j * 0.13f + t * 0.02f);
        if (t >= 16) { r.action = 3; r.weight = 1; }
        d.rows.push_back(r);
    }
    rek_bc::validate(d);
    Trainer a(initial, 16, 0.001f);
    auto read_logits = [&](Prec logits) {
        std::vector<precision_t> out(a.T * (A + 1));
        ck(cudaMemcpy(out.data(), logits.data, out.size() * sizeof(precision_t), cudaMemcpyDeviceToHost));
        return out;
    };
    auto identical = [](const auto& left, const auto& right) {
        return left.size() == right.size() && std::memcmp(left.data(), right.data(), left.size() * sizeof(left[0])) == 0;
    };
    a.reset(); a.forward(d, 0, 16, false); a.carry();
    const auto continued = read_logits(a.forward(d, 16, 32, true));
    a.burn_in(d, 0, 16);
    const auto replayed = read_logits(a.forward(d, 16, 32, true));
    require(identical(continued, replayed), "full-prefix replay changed recurrent chronology");
    a.reset();
    const auto omitted_prefix = read_logits(a.forward(d, 16, 32, true));
    require(!identical(continued, omitted_prefix), "unlabeled prefix had no recurrent effect");
    a.reset(); const auto clean_start = read_logits(a.forward(d, 0, 16, true));
    a.carry(); const auto reset_start = read_logits(a.forward(d, 0, 16, false));
    require(identical(clean_start, reset_start), "reset-before-observation failed or labels entered inputs");
    a.burn_in(d, 0, 16); const Prec loss_logits = a.forward(d, 16, 32, true);
    const auto actual_logits = read_logits(loss_logits); const auto actual_loss = a.loss(loss_logits, 16);
    std::vector<float> actual_grad(16 * A);
    ck(cudaMemcpy(actual_grad.data(), a.grad_logits.data, actual_grad.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double reference_loss = 0, maximum_gradient_error = 0;
    for (int t = 0; t < 16; ++t) {
        double maximum = -INFINITY, total = 0;
        for (int c = 0; c < 32; ++c) maximum = std::max(maximum, double(to_float(actual_logits[t * (A + 1) + c])));
        for (int c = 0; c < 32; ++c) total += std::exp(double(to_float(actual_logits[t * (A + 1) + c])) - maximum);
        reference_loss += std::log(total) + maximum - double(to_float(actual_logits[t * (A + 1) + 3]));
        for (int c = 0; c < A; ++c) {
            const double expected = c == 32 ? 0 : (std::exp(double(to_float(actual_logits[t * (A + 1) + c])) - maximum) / total - (c == 3)) / 16;
            maximum_gradient_error = std::max(maximum_gradient_error, std::abs(expected - actual_grad[t * A + c]));
        }
        require(actual_grad[t * A + 32] == 0, "unsupported class received gradient");
    }
    require(std::abs(actual_loss.loss - reference_loss) < 1e-4 && maximum_gradient_error < 1e-7, "masked CE/gradient CPU reference mismatch");
    auto uneven = d; uneven.rows.resize(27);
    for (size_t i = 0; i < uneven.rows.size(); ++i) {
        uneven.rows[i].action = i < 16 ? 1 : 21;
        uneven.rows[i].weight = i < 16 ? 0.4f : 3.0f;
    }
    rek_bc::validate(uneven);
    const auto norm = normalization(uneven, a.T);
    require(norm.rows == 27 && std::abs(norm.denominator - norm.weight / 27 * 16) < 1e-6, "global normalization formula mismatch");
    std::array<double, A> aggregate{}, global_reference{};
    a.reset();
    for (size_t p = 0; p < uneven.rows.size(); p += a.T) {
        const size_t end = std::min(p + a.T, uneven.rows.size());
        Prec z = a.forward(uneven, p, end, true); const auto host_z = read_logits(z);
        a.loss(z, norm.denominator);
        ck(cudaMemcpy(actual_grad.data(), a.grad_logits.data, actual_grad.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (int t = 0; t < a.T; ++t) for (int c = 0; c < A; ++c) aggregate[c] += actual_grad[t * A + c] * double(a.T) / norm.rows;
        for (size_t i = p; i < end; ++i) {
            const int t = int(i - p); double maximum = -INFINITY, total = 0;
            for (int c = 0; c < 32; ++c) maximum = std::max(maximum, double(to_float(host_z[t * (A + 1) + c])));
            for (int c = 0; c < 32; ++c) total += std::exp(double(to_float(host_z[t * (A + 1) + c])) - maximum);
            for (int c = 0; c < 32; ++c) global_reference[c] += uneven.rows[i].weight / norm.weight *
                (std::exp(double(to_float(host_z[t * (A + 1) + c])) - maximum) / total - (c == uneven.rows[i].action));
        }
        a.carry();
    }
    double global_gradient_error = 0;
    for (int c = 0; c < A; ++c) global_gradient_error = std::max(global_gradient_error, std::abs(aggregate[c] - global_reference[c]));
    require(global_gradient_error < 1e-7, "two-chunk aggregate disagrees with global weighted objective");
    const auto before = a.evaluate(d, 0); const auto start = a.checkpoint();
    a.evaluate(d, 1); require(a.checkpoint() == start, "heldout evaluation changed parameters");
    a.epoch(d); const auto one = a.checkpoint();
    {
        auto variant = d;
        for (auto& r : variant.rows) {
            r.obs[222] = 999;
            if (r.split == 1) { r.action = 4; r.weight = 3; for (float& v : r.obs) v += 5; }
        }
        Trainer b(initial, 16, 0.001f); b.epoch(variant);
        require(b.checkpoint() == one, "heldout data or masked feature influenced update");
    }
    for (int i = 1; i < 12; ++i) a.epoch(d);
    const auto after = a.evaluate(d, 0); const auto final = a.checkpoint();
    require(after.ce() < before.ce(), "native supervised loss did not improve");
    const int boundaries[] = {0, O * H, (O + A + 1) * H, PARAMS};
    for (int i = 0; i < 3; ++i) require(!std::equal(initial.begin() + boundaries[i], initial.begin() + boundaries[i + 1], final.begin() + boundaries[i]), "a policy module did not train");
    require(std::equal(initial.begin() + (O + A) * H, initial.begin() + (O + A + 1) * H, final.begin() + (O + A) * H), "BC changed value decoder row");
    auto unlabeled = d; for (auto& r : unlabeled.rows) { r.action = -1; r.weight = 0; }
    const auto saved = a.checkpoint(); const int updates = a.updates; a.epoch(unlabeled);
    require(a.checkpoint() == saved && a.updates == updates, "unlabeled sequence changed optimizer/weights");
    report("self_test_pass", 12, after, a.evaluate(d, 1), a.updates);
    std::printf("{\"initial_ce\":%.9g,\"final_ce\":%.9g,\"heldout_and_mask_update_invariance\":true,\"encoder_decoder_recurrent_updated\":true,\"value_row_preserved\":true,\"unlabeled_update_skipped\":true}\n", before.ce(), after.ce());
    std::printf("{\"full_prefix_replay_identical\":true,\"unlabeled_prefix_affects_rnn\":true,\"reset_before_observation\":true,\"labels_absent_from_features\":true,\"unsupported_gradient_zero\":true,\"cpu_gradient_max_error\":%.9g}\n", maximum_gradient_error);
    std::printf("{\"global_weighted_objective_two_chunks\":true,\"short_final_chunk\":true,\"global_gradient_max_error\":%.9g}\n", global_gradient_error);
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 4 && std::string(argv[1]) == "--self-test") {
            self_test(read_checkpoint(argv[2], argv[3])); return 0;
        }
        require(argc == 8, "Usage: bc-train DATA INITIAL SHA256 NEW_OUTPUT EPOCHS LR HORIZON | --self-test INITIAL SHA256");
        const auto data = rek_bc::load(argv[1]); const auto initial = read_checkpoint(argv[2], argv[3]);
        char* end = nullptr;
        const long epochs = std::strtol(argv[5], &end, 10); require(end != argv[5] && *end == 0 && epochs >= 0 && epochs <= 100, "invalid epochs");
        const float lr = std::strtof(argv[6], &end); require(*end == 0 && std::isfinite(lr) && lr > 0 && lr <= 0.01f, "invalid learning rate");
        const long horizon = std::strtol(argv[7], &end, 10); require(*end == 0 && horizon >= 4 && horizon <= 256, "invalid horizon");
        struct stat st{}; require(stat(argv[4], &st) != 0 && errno == ENOENT, "output exists or cannot be inspected");
        for (int epoch = 1; epoch <= epochs; ++epoch) {
            const std::string path = std::string(argv[4]) + ".epoch-" + std::to_string(epoch) + ".bin";
            require(stat(path.c_str(), &st) != 0 && errno == ENOENT, "epoch output exists or cannot be inspected");
        }
        Trainer trainer(initial, int(horizon), lr);
        auto train = trainer.evaluate(data, 0), heldout = trainer.evaluate(data, 1);
        require(train.weight > 0 && heldout.weight > 0, "both whole-sequence splits need grounded labels");
        double min_dt = INFINITY, max_dt = 0;
        for (size_t i = 1; i < data.rows.size(); ++i) if (!data.rows[i].reset) {
            const double dt = data.rows[i].time - data.rows[i - 1].time;
            min_dt = std::min(min_dt, dt); max_dt = std::max(max_dt, dt);
        }
        require(std::isfinite(min_dt), "dataset has no sequential timesteps");
        std::printf("{\"schema\":\"rek.native_bc.v2\",\"obs\":223,\"actions\":33,\"hidden\":256,\"layers\":2,\"horizon\":%ld,\"burn_in\":\"full_preceding_sequence_current_weights\",\"feature_mask_sha256\":\"%s\",\"input_sha256\":\"%s\",\"rows\":%zu,\"sequences\":%zu}\n",
            horizon, digest(data.feature_mask.data(), data.feature_mask.size()).c_str(), argv[3], data.rows.size(), data.sequences.size());
        std::printf("{\"learning_rate\":%.9g,\"momentum\":0.95,\"max_grad_norm\":1,\"min_dt_seconds\":%.9g,\"max_dt_seconds\":%.9g,\"class_support_semantics\":\"vocabulary_only_not_runtime_legality\"}\n", lr, min_dt, max_dt);
        const auto norm = normalization(data, int(horizon));
        std::printf("{\"loss_normalization\":\"global_training_weight_over_all_training_rows_times_horizon_v2\",\"training_rows\":%zu,\"training_weight\":%.12g,\"fixed_denominator\":%.12g}\n", norm.rows, norm.weight, norm.denominator);
        report("initial", 0, train, heldout, 0);
        report_details("train", 0, train); report_details("heldout", 0, heldout);
        for (int epoch = 1; epoch <= epochs; ++epoch) {
            trainer.epoch(data); train = trainer.evaluate(data, 0); heldout = trainer.evaluate(data, 1);
            report("epoch", epoch, train, heldout, trainer.updates);
            report_details("train", epoch, train); report_details("heldout", epoch, heldout);
            const auto snapshot = trainer.checkpoint();
            const std::string path = std::string(argv[4]) + ".epoch-" + std::to_string(epoch) + ".bin";
            write_checkpoint(path.c_str(), snapshot);
            std::printf("{\"phase\":\"epoch_saved\",\"epoch\":%d,\"sha256\":\"%s\"}\n", epoch, digest(snapshot.data(), snapshot.size() * sizeof(float)).c_str());
        }
        const auto output = trainer.checkpoint(); write_checkpoint(argv[4], output);
        std::printf("{\"phase\":\"saved\",\"sha256\":\"%s\",\"bytes\":%zu}\n", digest(output.data(), output.size() * sizeof(float)).c_str(), output.size() * sizeof(float));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "native_bc_error: %s\n", e.what()); return 2;
    }
}
