// Standalone observational-score probe. No simulator, policy, or trainer integration.
// Every forward pass, training gradient, reduction and SGD update executes on CUDA.
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int Outputs = 2, Hidden = 16, Epochs = 500;
constexpr float LearningRate = .03f, L2 = .01f;
constexpr std::array<unsigned, 3> Seeds{11, 29, 73};
void require(bool ok, const char* reason) { if (!ok) throw std::runtime_error(reason); }
void checked(cudaError_t error) { if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error)); }
template<class T> struct Device {
    T* p = nullptr; size_t n;
    explicit Device(size_t count) : n(count) { checked(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T))); }
    ~Device() { if (p) cudaFree(p); }
    Device(const Device&) = delete; Device& operator=(const Device&) = delete;
    void put(const std::vector<T>& values) { require(values.size() == n, "device_copy_size"); checked(cudaMemcpy(p, values.data(), n * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get() const { std::vector<T> result(n); checked(cudaMemcpy(result.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return result; }
};
struct Spec {
    int d, h, stride, parameters;
    Spec(int dimensions, int hidden, int inputStride) : d(dimensions), h(hidden), stride(inputStride),
        parameters(hidden ? hidden * dimensions + hidden + Outputs * hidden + Outputs : Outputs * dimensions + Outputs) {}
};
__host__ __device__ bool weight_parameter(Spec s, int j) {
    return s.h ? (j < s.h * s.d || (j >= s.h * s.d + s.h && j < s.parameters - Outputs)) : j < Outputs * s.d;
}
__device__ float sigmoid(float z) { return z >= 0 ? 1.f / (1.f + expf(-z)) : expf(z) / (1.f + expf(z)); }
__device__ uint32_t mix(uint32_t x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16); }
__global__ void initialize(float* parameters, Spec s, unsigned seed) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x; if (j >= s.parameters) return;
    float bound = .05f;
    if (s.h) bound = j < s.h * s.d ? sqrtf(6.f / float(s.d + s.h)) : .1f * sqrtf(6.f / float(s.h + Outputs));
    parameters[j] = weight_parameter(s, j) ? (float(mix(seed ^ uint32_t(j + 1) * 0x9e3779b9u) >> 8) / 16777216.f * 2.f - 1.f) * bound : 0.f;
}
__global__ void scale_features(const float* raw, const float* means, const float* scales, float* x, int values, int f, int* status) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= values) return;
    const float value = (raw[i] - means[i % f]) / scales[i % f]; x[i] = value;
    if (!isfinite(value)) atomicExch(status, 1);
}
__device__ void forward_row(const float* x, const float* w, Spec s, float* hidden, float* logits) {
    if (s.h) {
        const int b1 = s.h * s.d, w2 = b1 + s.h, b2 = w2 + Outputs * s.h;
        for (int k = 0; k < s.h; k++) {
            float a = w[b1 + k]; for (int j = 0; j < s.d; j++) a += w[k * s.d + j] * x[j]; hidden[k] = tanhf(a);
        }
        for (int o = 0; o < Outputs; o++) { float z = w[b2 + o]; for (int k = 0; k < s.h; k++) z += w[w2 + o * s.h + k] * hidden[k]; logits[o] = z; }
    } else {
        for (int o = 0; o < Outputs; o++) { float z = w[Outputs * s.d + o]; for (int j = 0; j < s.d; j++) z += w[o * s.d + j] * x[j]; logits[o] = z; }
    }
}
__global__ void predict(const float* x, const float* w, Spec s, int n, float* logits, int* status) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float a[Hidden], z[Outputs]; forward_row(x + i * s.stride, w, s, a, z);
    for (int o = 0; o < Outputs; o++) { logits[i * Outputs + o] = z[o]; if (!isfinite(z[o])) atomicExch(status, 1); }
}
__global__ void sample_gradients(const float* x, const float* y, const float* w, Spec s, int trainN, float* perRow, int* status) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= trainN) return;
    const float* row = x + i * s.stride; float a[Hidden], z[Outputs], delta[Outputs];
    forward_row(row, w, s, a, z);
    for (int o = 0; o < Outputs; o++) { delta[o] = (sigmoid(z[o]) - y[i * Outputs + o]) / float(trainN * Outputs); if (!isfinite(delta[o])) atomicExch(status, 1); }
    float* g = perRow + i * s.parameters;
    if (s.h) {
        const int b1 = s.h * s.d, w2 = b1 + s.h, b2 = w2 + Outputs * s.h;
        for (int o = 0; o < Outputs; o++) { g[b2 + o] = delta[o]; for (int k = 0; k < s.h; k++) g[w2 + o * s.h + k] = delta[o] * a[k]; }
        for (int k = 0; k < s.h; k++) {
            float d = 0; for (int o = 0; o < Outputs; o++) d += delta[o] * w[w2 + o * s.h + k]; d *= 1.f - a[k] * a[k];
            g[b1 + k] = d; for (int j = 0; j < s.d; j++) g[k * s.d + j] = d * row[j];
        }
    } else {
        for (int o = 0; o < Outputs; o++) { g[Outputs * s.d + o] = delta[o]; for (int j = 0; j < s.d; j++) g[o * s.d + j] = delta[o] * row[j]; }
    }
}
__global__ void reduce_gradients(const float* perRow, const float* w, Spec s, int n, float l2, float* gradient, int* status) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x; if (j >= s.parameters) return;
    double sum = 0; for (int i = 0; i < n; i++) sum += perRow[i * s.parameters + j];
    const float g = float(sum) + (weight_parameter(s, j) ? l2 * w[j] : 0.f); gradient[j] = g;
    if (!isfinite(g)) atomicExch(status, 1);
}
__global__ void sgd(float* w, const float* gradient, int count, float lr, int* status) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x; if (j >= count) return;
    w[j] -= lr * gradient[j]; if (!isfinite(w[j])) atomicExch(status, 1);
}
void gradients(Device<float>& x, Device<float>& y, Device<float>& w, Spec s, int n, Device<float>& rows, Device<float>& g, Device<int>& status) {
    sample_gradients<<<(n + 127) / 128, 128>>>(x.p, y.p, w.p, s, n, rows.p, status.p);
    reduce_gradients<<<(s.parameters + 127) / 128, 128>>>(rows.p, w.p, s, n, L2, g.p, status.p); checked(cudaGetLastError());
}
double softplus_loss(double z, double y) { return std::max(z, 0.) - y * z + std::log1p(std::exp(-std::abs(z))); }
// CPU reference evaluation exists only for numerical tests, never for fitting.
double reference(const std::vector<float>& x, const std::vector<float>& y, const std::vector<double>& w, Spec s, int n) {
    double loss = 0;
    for (int i = 0; i < n; i++) {
        std::array<double, Hidden> a{};
        if (s.h) for (int k = 0; k < s.h; k++) { double v = w[s.h * s.d + k]; for (int j = 0; j < s.d; j++) v += w[k * s.d + j] * x[i * s.stride + j]; a[k] = std::tanh(v); }
        for (int o = 0; o < Outputs; o++) {
            double z = w[s.parameters - Outputs + o];
            if (s.h) for (int k = 0; k < s.h; k++) z += w[s.h * s.d + s.h + o * s.h + k] * a[k];
            else for (int j = 0; j < s.d; j++) z += w[o * s.d + j] * x[i * s.stride + j];
            loss += softplus_loss(z, y[i * Outputs + o]) / (n * Outputs);
        }
    }
    for (int j = 0; j < s.parameters; j++) if (weight_parameter(s, j)) loss += .5 * L2 * w[j] * w[j];
    return loss;
}
struct TestResult { int gradientComparisons = 0, updateChecks = 0, scalerComparisons = 0; double maxGradientError = 0, maxScalerError = 0; };
TestResult self_test() {
    constexpr int n = 8, f = 44; TestResult result;
    std::vector<float> x(n * f), raw(n * f), y(n * Outputs), means(f), scales(f);
    for (int j = 0; j < f; j++) { means[j] = .01f * j; scales[j] = .5f + .02f * j; }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < f; j++) { x[i * f + j] = .7f * std::sin(float(13 * i + 3 * j)); raw[i * f + j] = x[i * f + j] * scales[j] + means[j]; }
        y[2 * i] = x[i * f] > 0; y[2 * i + 1] = x[i * f + 1] > 0;
    }
    Device<float> dx(x.size()), draw(raw.size()), dy(y.size()), dm(f), ds(f); Device<int> status(1);
    draw.put(raw); dy.put(y); dm.put(means); ds.put(scales); checked(cudaMemset(status.p, 0, sizeof(int)));
    scale_features<<<(int(x.size()) + 127) / 128, 128>>>(draw.p, dm.p, ds.p, dx.p, int(x.size()), f, status.p); checked(cudaGetLastError());
    const auto scaled = dx.get();
    for (size_t i = 0; i < x.size(); i++) { result.maxScalerError = std::max(result.maxScalerError, double(std::abs(scaled[i] - x[i]))); result.scalerComparisons++; }
    require(result.maxScalerError < 1e-6, "self_test_scaler");
    for (const Spec s : {Spec(5, 0, f), Spec(f, 0, f), Spec(f, Hidden, f)}) {
        Device<float> w(s.parameters), rows(n * s.parameters), g(s.parameters);
        initialize<<<(s.parameters + 127) / 128, 128>>>(w.p, s, 73); checked(cudaGetLastError());
        gradients(dx, dy, w, s, n, rows, g, status); const auto weights = w.get(), analytic = g.get();
        std::vector<double> referenceWeights(weights.begin(), weights.end());
        for (int j = 0; j < s.parameters; j++) {
            constexpr double epsilon = 1e-4; referenceWeights[j] += epsilon; const double plus = reference(scaled, y, referenceWeights, s, n);
            referenceWeights[j] -= 2 * epsilon; const double minus = reference(scaled, y, referenceWeights, s, n); referenceWeights[j] += epsilon;
            const double error = std::abs(analytic[j] - (plus - minus) / (2 * epsilon));
            result.maxGradientError = std::max(result.maxGradientError, error); result.gradientComparisons++;
        }
        const double before = reference(scaled, y, referenceWeights, s, n);
        sgd<<<(s.parameters + 127) / 128, 128>>>(w.p, g.p, s.parameters, LearningRate, status.p); checked(cudaGetLastError());
        const auto afterWeights = w.get(); referenceWeights.assign(afterWeights.begin(), afterWeights.end());
        require(reference(scaled, y, referenceWeights, s, n) < before, "self_test_gpu_sgd_did_not_lower_loss"); result.updateChecks++;
    }
    require(result.maxGradientError < 3e-5, "self_test_gradient_mismatch"); require(status.get()[0] == 0, "self_test_nonfinite"); return result;
}
uint32_t read_u32(std::istream& in) { unsigned char b[4]; in.read(reinterpret_cast<char*>(b), 4); require(bool(in), "dataset_truncated"); return uint32_t(b[0]) | uint32_t(b[1]) << 8 | uint32_t(b[2]) << 16 | uint32_t(b[3]) << 24; }
float read_float(std::istream& in) { const uint32_t bits = read_u32(in); float value; std::memcpy(&value, &bits, 4); require(std::isfinite(value), "dataset_nonfinite"); return value; }
void write_u32(std::ostream& out, uint32_t value) { const unsigned char b[4] = {static_cast<unsigned char>(value), static_cast<unsigned char>(value >> 8), static_cast<unsigned char>(value >> 16), static_cast<unsigned char>(value >> 24)}; out.write(reinterpret_cast<const char*>(b), 4); }
void write_float(std::ostream& out, float value) { uint32_t bits; std::memcpy(&bits, &value, 4); write_u32(out, bits); }
struct Data { int f, trainN, heldoutN; std::vector<float> means, scales, x, y; int n() const { return trainN + heldoutN; } };
Data load_data(const std::filesystem::path& file) {
    std::ifstream in(file, std::ios::binary); require(bool(in), "dataset_open_failed"); char magic[8]; in.read(magic, 8);
    require(bool(in) && std::memcmp(magic, "REKSHP1\0", 8) == 0, "dataset_magic"); require(read_u32(in) == 1, "dataset_version");
    Data d; d.f = int(read_u32(in)); d.trainN = int(read_u32(in)); d.heldoutN = int(read_u32(in));
    require(d.f == 44 && d.trainN > 0 && d.trainN <= 8192 && d.heldoutN > 0 && d.heldoutN <= 8192, "dataset_dimensions");
    const uintmax_t expected = 24 + uintmax_t(2 * d.f + d.n() * (d.f + Outputs)) * 4;
    require(std::filesystem::file_size(file) == expected, "dataset_length");
    d.means.resize(d.f); d.scales.resize(d.f); d.x.resize(d.n() * d.f); d.y.resize(d.n() * Outputs);
    for (float& v : d.means) v = read_float(in);
    for (float& v : d.scales) { v = read_float(in); require(v > 0, "dataset_scale_nonpositive"); }
    for (int i = 0; i < d.n(); i++) {
        for (int j = 0; j < d.f; j++) { const float value = read_float(in); require(std::abs(value) <= 1e6, "dataset_feature_out_of_bounds"); d.x[i * d.f + j] = value; }
        for (int o = 0; o < Outputs; o++) { const float value = read_float(in); require(value == 0 || value == 1, "dataset_label_not_binary"); d.y[i * Outputs + o] = value; }
    }
    return d;
}
struct Metrics { int n = 0, positives = 0; double bce = 0, brier = 0, meanPrediction = 0, auc = 0; bool aucDefined = false; };
Metrics metrics(const std::vector<float>& logits, const std::vector<float>& y, int begin, int n, int output) {
    Metrics m; m.n = n;
    for (int i = begin; i < begin + n; i++) {
        const double z = logits[i * Outputs + output], target = y[i * Outputs + output]; require(std::isfinite(z), "nonfinite_evaluation");
        const double p = z >= 0 ? 1 / (1 + std::exp(-z)) : std::exp(z) / (1 + std::exp(z));
        m.positives += int(target); m.bce += softplus_loss(z, target) / n; m.brier += (p - target) * (p - target) / n; m.meanPrediction += p / n;
    }
    m.aucDefined = m.positives > 0 && m.positives < n;
    if (m.aucDefined) {
        double wins = 0;
        for (int i = begin; i < begin + n; i++) if (y[i * Outputs + output] == 1)
            for (int j = begin; j < begin + n; j++) if (y[j * Outputs + output] == 0)
                wins += logits[i * Outputs + output] > logits[j * Outputs + output] ? 1 : logits[i * Outputs + output] == logits[j * Outputs + output] ? .5 : 0;
        m.auc = wins / (double(m.positives) * (n - m.positives));
    }
    return m;
}
void emit_metric(std::ostream& out, const Metrics& m) {
    out << "{\"n\":" << m.n << ",\"positive_windows\":" << m.positives << ",\"bce\":" << m.bce << ",\"brier\":" << m.brier
        << ",\"mean_prediction\":" << m.meanPrediction << ",\"observed_fraction\":" << double(m.positives) / m.n << ",\"roc_auc\":";
    if (m.aucDefined) out << m.auc; else out << "null"; out << '}';
}
void emit_splits(std::ostream& out, const std::vector<float>& logits, const Data& data) {
    out << "\"train\":["; for (int o = 0; o < Outputs; o++) { if (o) out << ','; emit_metric(out, metrics(logits, data.y, 0, data.trainN, o)); }
    out << "],\"heldout\":["; for (int o = 0; o < Outputs; o++) { if (o) out << ','; emit_metric(out, metrics(logits, data.y, data.trainN, data.heldoutN, o)); } out << ']';
}
void save_weights(const std::filesystem::path& file, const Data& data, Spec s, unsigned seed, const std::vector<float>& values) {
    require(!std::filesystem::exists(file), "checkpoint_exists"); std::ofstream out(file, std::ios::binary); require(bool(out), "checkpoint_open");
    out.write("REKSHPW1", 8); for (uint32_t v : {1u, uint32_t(data.f), uint32_t(s.d), uint32_t(s.h), uint32_t(Outputs), uint32_t(s.parameters), seed, uint32_t(Epochs)}) write_u32(out, v);
    write_float(out, LearningRate); write_float(out, L2); for (float v : data.means) write_float(out, v); for (float v : data.scales) write_float(out, v); for (float v : values) write_float(out, v);
    out.flush(); require(bool(out), "checkpoint_write_failed");
}
void run(const std::filesystem::path& input, const std::filesystem::path& transforms, const std::filesystem::path& output) {
    const Data data = load_data(input); require(std::filesystem::is_regular_file(transforms) && std::filesystem::file_size(transforms) > 0 && std::filesystem::file_size(transforms) <= 1048576, "transform_manifest_invalid");
    require(!std::filesystem::exists(output), "output_directory_exists"); require(std::filesystem::create_directory(output), "output_directory_create_failed");
    std::filesystem::copy_file(transforms, output / "feature-transform-manifest.json");
    const auto tests = self_test();
    Device<float> raw(data.x.size()), x(data.x.size()), y(data.y.size()), means(data.f), scales(data.f), logits(data.n() * Outputs); Device<int> status(1);
    raw.put(data.x); y.put(data.y); means.put(data.means); scales.put(data.scales); checked(cudaMemset(status.p, 0, sizeof(int)));
    scale_features<<<(int(data.x.size()) + 127) / 128, 128>>>(raw.p, means.p, scales.p, x.p, int(data.x.size()), data.f, status.p); checked(cudaGetLastError());
    std::ofstream out(output / "report.json"); require(bool(out), "report_open_failed"); out << std::setprecision(10);
    out << "{\"schema\":\"rek.cuda_observational_score_head_probe.v1\",\"authentic_parity\":false,\"policy_training\":false,\"training_backend\":\"native_cuda_full_batch_sgd\","
        << "\"target_order\":[\"observed_local_strike_points_next_3s_any\",\"observed_opponent_strike_points_next_3s_any\"],"
        << "\"observation_windows_are_independent\":false,\"rounds\":{\"train\":1,\"heldout\":2},\"heldout_selection\":false,"
        << "\"features\":" << data.f << ",\"train_windows\":" << data.trainN << ",\"heldout_windows\":" << data.heldoutN
        << ",\"epochs\":500,\"learning_rate\":0.03,\"l2\":0.01,\"objective\":\"mean_binary_cross_entropy_over_samples_and_two_outputs_plus_half_l2_sum_nonbias_weight_squares\","
        << "\"scaler\":\"supplied_training_round_only_mean_and_population_sd_applied_once_on_cuda\",\"self_test\":{\"passed\":true,\"gradient_comparisons\":" << tests.gradientComparisons
        << ",\"max_gradient_absolute_error\":" << tests.maxGradientError << ",\"gpu_loss_reduction_checks\":" << tests.updateChecks << ",\"scaler_comparisons\":" << tests.scalerComparisons << "},\"constant_train_prior\":{";
    std::vector<float> baseline(data.n() * Outputs);
    for (int o = 0; o < Outputs; o++) { double prior = 0; for (int i = 0; i < data.trainN; i++) prior += data.y[i * Outputs + o] / data.trainN;
        prior = std::max(1e-6, std::min(1 - 1e-6, prior)); for (int i = 0; i < data.n(); i++) baseline[i * Outputs + o] = float(std::log(prior / (1 - prior))); }
    emit_splits(out, baseline, data); out << "},\"runs\":["; bool firstRun = true;
    for (int kind = 0; kind < 3; kind++) for (unsigned seed : Seeds) {
        const char* name = kind == 0 ? "geometry_logistic" : kind == 1 ? "full_logistic" : "full_mlp16";
        const Spec s(kind == 0 ? 5 : data.f, kind == 2 ? Hidden : 0, data.f);
        Device<float> w(s.parameters), rows(data.trainN * s.parameters), gradient(s.parameters);
        initialize<<<(s.parameters + 127) / 128, 128>>>(w.p, s, seed); checked(cudaGetLastError());
        if (!firstRun) out << ','; firstRun = false;
        out << "{\"model\":\"" << name << "\",\"seed\":" << seed << ",\"learned_parameters\":" << s.parameters << ",\"input_features\":" << s.d << ",\"hidden_units\":" << s.h << ",\"snapshots\":[";
        const auto started = std::chrono::steady_clock::now(); bool firstSnapshot = true;
        for (int epoch = 1; epoch <= Epochs; epoch++) {
            gradients(x, y, w, s, data.trainN, rows, gradient, status);
            sgd<<<(s.parameters + 127) / 128, 128>>>(w.p, gradient.p, s.parameters, LearningRate, status.p); checked(cudaGetLastError());
            if (epoch == 100 || epoch == 500) {
                predict<<<(data.n() + 127) / 128, 128>>>(x.p, w.p, s, data.n(), logits.p, status.p); checked(cudaGetLastError());
                const auto hostLogits = logits.get(); require(status.get()[0] == 0, "nonfinite_training_or_prediction");
                if (!firstSnapshot) out << ','; firstSnapshot = false; out << "{\"epoch\":" << epoch << ','; emit_splits(out, hostLogits, data); out << '}';
            }
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        const std::string checkpoint = std::string(name) + "-seed-" + std::to_string(seed) + ".private.bin";
        const auto weights = w.get(); save_weights(output / checkpoint, data, s, seed, weights);
        out << "],\"fit_and_measure_wall_seconds\":" << seconds << ",\"private_weights\":\"" << checkpoint << "\"}";
    }
    out << "],\"limitations\":[\"Two rounds from one human session, with overlapping three-second outcome windows; no independent-sample or causal-hit claim.\","
        << "\"Zero labels mean no observed strike-score receipt in a complete window, not a failed attack.\","
        << "\"No executed-move, server-acceptance, authentic-policy-improvement or policy-SPS inference.\","
        << "\"Fixed seeds and epoch snapshots were not selected using heldout performance.\"]}\n";
    out.flush(); require(bool(out), "report_write_failed"); checked(cudaDeviceSynchronize());
    std::cout << "{\"completed\":true,\"models\":3,\"seeds\":3,\"train_windows\":" << data.trainN << ",\"heldout_windows\":" << data.heldoutN << ",\"all_optimization_cuda\":true}\n";
}
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") { const auto t = self_test(); std::cout << std::setprecision(10)
            << "{\"passed\":true,\"gradient_comparisons\":" << t.gradientComparisons << ",\"max_gradient_absolute_error\":" << t.maxGradientError
            << ",\"gpu_loss_reduction_checks\":" << t.updateChecks << ",\"scaler_comparisons\":" << t.scalerComparisons << ",\"max_scaler_absolute_error\":" << t.maxScalerError << "}\n"; return 0; }
        require(argc == 5 && std::string(argv[1]) == "--run", "usage: score-head-probe --self-test | --run DATA_BIN TRANSFORM_MANIFEST_JSON NEW_PRIVATE_OUTPUT_DIRECTORY");
        run(argv[2], argv[3], argv[4]); return 0;
    } catch (const std::exception& error) { std::cerr << "score_head_probe: " << error.what() << '\n'; return 1; }
}
