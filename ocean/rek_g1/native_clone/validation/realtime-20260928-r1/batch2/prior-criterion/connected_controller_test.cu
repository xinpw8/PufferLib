#include "sonic_controller.cuh"
#include "sonic_onnx_reader.h"
#include <onnxruntime_c_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

// Implementation equivalence only. ORT CPU is not a Unity/server oracle.
// The default inputs reproduce test_sonic_controller.cu's deterministic probe.
namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { cuda_check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
};
struct DeviceBuffer {
    float* data = nullptr;
    explicit DeviceBuffer(std::size_t count) {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(float)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
struct CpuOracle {
    const OrtApi* api = nullptr;
    OrtEnv* environment = nullptr;
    OrtSessionOptions* options = nullptr;
    OrtMemoryInfo* memory = nullptr;
    OrtSession* encoder = nullptr;
    OrtSession* decoder = nullptr;
    void ok(OrtStatus* status) {
        if (!status) return;
        std::string detail = api->GetErrorMessage(status);
        api->ReleaseStatus(status);
        throw std::runtime_error(detail);
    }
    CpuOracle(const char* encoder_path, const char* decoder_path) {
        api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        check(api != nullptr, "ORT C API unavailable");
        ok(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "connected-inference-equivalence", &environment));
        ok(api->CreateSessionOptions(&options));
        ok(api->SetIntraOpNumThreads(options, 1));
        ok(api->SetInterOpNumThreads(options, 1));
        ok(api->SetSessionExecutionMode(options, ORT_SEQUENTIAL));
        ok(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory));
        ok(api->CreateSession(environment, encoder_path, options, &encoder));
        ok(api->CreateSession(environment, decoder_path, options, &decoder));
    }
    ~CpuOracle() {
        if (decoder) api->ReleaseSession(decoder);
        if (encoder) api->ReleaseSession(encoder);
        if (memory) api->ReleaseMemoryInfo(memory);
        if (options) api->ReleaseSessionOptions(options);
        if (environment) api->ReleaseEnv(environment);
    }
    std::vector<float> run(bool encode, std::vector<float>& input, std::size_t batch) {
        const std::int64_t shape[] = {std::int64_t(batch), encode ? 1762 : 994};
        const std::int64_t output_shape[] = {std::int64_t(batch), encode ? 64 : 29};
        check(input.size() == batch * std::size_t(shape[1]), "ORT input size mismatch");
        std::vector<float> output(batch * std::size_t(output_shape[1]));
        OrtValue* in = nullptr;
        OrtValue* out = nullptr;
        ok(api->CreateTensorWithDataAsOrtValue(memory, input.data(), input.size() * sizeof(float),
            shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in));
        auto status = api->CreateTensorWithDataAsOrtValue(memory, output.data(), output.size() * sizeof(float),
            output_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &out);
        if (status) { api->ReleaseValue(in); ok(status); }
        const char* input_name = "obs_dict";
        const char* output_name = encode ? "encoded_tokens" : "action";
        const OrtValue* inputs[] = {in};
        status = api->Run(encode ? encoder : decoder, nullptr, &input_name, inputs, 1,
            &output_name, 1, &out);
        api->ReleaseValue(out);
        api->ReleaseValue(in);
        ok(status);
        return output;
    }
};
float probe(std::size_t row, std::size_t column, std::size_t salt) {
    return (float((row * 131 + column * 17 + salt * 29) % 2003) - 1001.0f) / 317.0f;
}
void finite(const std::vector<float>& values) {
    for (float value : values) check(std::isfinite(value), "nonfinite input or output");
}
std::vector<float> read_f32(const char* path, std::size_t count) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    check(bool(file), "input file open failed");
    check(file.tellg() == std::streamoff(count * sizeof(float)), "input file size mismatch");
    file.seekg(0);
    std::vector<float> values(count);
    file.read(reinterpret_cast<char*>(values.data()), std::streamsize(count * sizeof(float)));
    check(bool(file), "input file read failed");
    finite(values);
    return values;
}
std::vector<float> connected_input(const std::vector<float>& original,
    const std::vector<float>& tokens, std::size_t batch) {
    check(original.size() == batch * 994 && tokens.size() == batch * 64,
        "connected input size mismatch");
    finite(original); finite(tokens);
    auto result = original;
    for (std::size_t row = 0; row < batch; ++row)
        std::copy_n(tokens.data() + row * 64, 64, result.data() + row * 994);
    return result;
}
struct Difference { double maximum = 0; std::size_t unequal = 0, outside = 0; };
Difference difference(const std::vector<float>& a, const std::vector<float>& b,
    double tolerance, std::size_t start = 0, std::size_t count = 0) {
    check(a.size() == b.size(), "comparison size mismatch");
    if (!count) count = a.size();
    check(start <= a.size() && count <= a.size() - start, "comparison range mismatch");
    Difference result;
    for (std::size_t i = start; i < start + count; ++i) {
        check(std::isfinite(a[i]) && std::isfinite(b[i]), "nonfinite controller result");
        const double delta = std::abs(double(a[i]) - double(b[i]));
        result.maximum = std::max(result.maximum, delta);
        result.unequal += delta != 0;
        result.outside += delta > tolerance;
    }
    return result;
}
Difference report(const char* label, const std::vector<float>& a,
    const std::vector<float>& b, double tolerance) {
    const auto d = difference(a, b, tolerance);
    std::printf("{\"kind\":\"comparison\",\"label\":\"%s\",\"values\":%zu,\"max_abs\":%.12g,"
        "\"unequal\":%zu,\"outside_tolerance\":%zu,\"tolerance\":%.12g}\n",
        label, a.size(), d.maximum, d.unequal, d.outside, tolerance);
    return d;
}
std::vector<float> copy_to_host(const float* source, std::size_t count) {
    std::vector<float> result(count);
    cuda_check(cudaMemcpy(result.data(), source, count * sizeof(float), cudaMemcpyDeviceToHost));
    return result;
}
void self_test() {
    std::vector<float> original(2 * 994), tokens(2 * 64);
    for (std::size_t i = 0; i < original.size(); ++i) original[i] = float(i + 1);
    for (std::size_t i = 0; i < tokens.size(); ++i) tokens[i] = -float(i + 1);
    const auto connected = connected_input(original, tokens, 2);
    for (std::size_t row = 0; row < 2; ++row) {
        check(std::equal(connected.begin() + row * 994, connected.begin() + row * 994 + 64,
            tokens.begin() + row * 64), "token placement self-test failed");
        check(std::equal(connected.begin() + row * 994 + 64, connected.begin() + (row + 1) * 994,
            original.begin() + row * 994 + 64), "history preservation self-test failed");
    }
    bool rejected = false;
    try { connected_input(original, std::vector<float>(127), 2); }
    catch (const std::exception&) { rejected = true; }
    check(rejected, "wrong token count accepted");
    auto changed = connected; changed[994 + 17] += .0625f;
    const auto d = difference(connected, changed, 0, 994, 64);
    check(d.unequal == 1 && d.maximum == .0625, "mismatch-row self-test failed");
    std::printf("{\"kind\":\"cpu_self_test\",\"passed\":true,\"cuda_used\":false}\n");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") { self_test(); return 0; }
        check(argc == 4 || (argc == 7 && std::string(argv[4]) == "--inputs"),
            "usage: connected_controller_test BATCH ENCODER.onnx DECODER.onnx [--inputs encoder.f32 decoder.f32]");
        char* end = nullptr;
        const auto parsed = std::strtoull(argv[1], &end, 10);
        check(end && *end == '\0' && parsed > 0 && parsed <= 1024, "batch must be 1..1024");
        const std::size_t batch = std::size_t(parsed);
        sonic_onnx::validate_io(sonic_onnx::load(argv[2]), batch, true);
        sonic_onnx::validate_io(sonic_onnx::load(argv[3]), batch, false);
        std::vector<float> encoder_input(batch * 1762, 0), history_input(batch * 994);
        if (argc == 7) {
            encoder_input = read_f32(argv[5], batch * 1762);
            history_input = read_f32(argv[6], batch * 994);
        } else {
            for (std::size_t row = 0; row < batch; ++row) {
                for (std::size_t c = 0; c < 580; ++c) encoder_input[row * 1762 + 4 + c] = probe(row,c,1);
                for (std::size_t c = 0; c < 60; ++c) encoder_input[row * 1762 + 601 + c] = probe(row,c,2);
                for (std::size_t c = 0; c < 994; ++c) history_input[row * 994 + c] = probe(row,c,3);
            }
        }
        CpuOracle oracle(argv[2], argv[3]);
        auto ort_tokens = oracle.run(true, encoder_input, batch);
        Stream stream;
        char error[1024] = {};
        std::unique_ptr<SonicController, decltype(&sonic_controller_destroy)> controller(
            sonic_controller_create(argv[2], argv[3], batch, stream.value, error, sizeof(error)),
            sonic_controller_destroy);
        check(controller != nullptr, error);
        DeviceBuffer d_encoder(batch * 1762), d_tokens(batch * 64), d_decoder(batch * 994), d_actions(batch * 29);
        cuda_check(cudaMemcpy(d_encoder.data, encoder_input.data(), encoder_input.size() * sizeof(float), cudaMemcpyHostToDevice));
        check(sonic_controller_encode(controller.get(), d_encoder.data, d_tokens.data, error, sizeof(error)), error);
        cuda_check(cudaStreamSynchronize(stream.value));
        auto native_tokens = copy_to_host(d_tokens.data, batch * 64);
        auto from_native = connected_input(history_input, native_tokens, batch);
        auto from_ort = connected_input(history_input, ort_tokens, batch);
        for (std::size_t row = 0; row < batch; ++row)
            check(std::equal(from_native.begin() + row * 994 + 64, from_native.begin() + (row + 1) * 994,
                from_ort.begin() + row * 994 + 64), "decoder histories differ");
        auto decode_native = [&](const std::vector<float>& input) {
            cuda_check(cudaMemcpy(d_decoder.data, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
            check(sonic_controller_decode(controller.get(), d_decoder.data, d_actions.data, error, sizeof(error)), error);
            cuda_check(cudaStreamSynchronize(stream.value));
            return copy_to_host(d_actions.data, batch * 29);
        };
        const auto native_native = decode_native(from_native);
        const auto native_ort = decode_native(from_ort);
        auto ort_native = oracle.run(false, from_native, batch);
        auto ort_ort = oracle.run(false, from_ort, batch);
        std::printf("{\"kind\":\"provenance\",\"scope\":\"inference-equivalence-only\",\"unity_or_server_parity\":false,"
            "\"input_kind\":\"%s\",\"batch\":%zu,\"history_width\":930,\"history_equal\":true,"
            "\"encoder_sha256\":\"%s\",\"decoder_sha256\":\"%s\",\"ort_version\":\"%s\"}\n",
            argc == 7 ? "caller-supplied-f32" : "existing-synthetic-probe", batch,
            sonic_controller_encoder_sha256(controller.get()), sonic_controller_decoder_sha256(controller.get()),
            OrtGetApiBase()->GetVersionString());
        const auto token_diff = report("native_vs_ort_tokens", native_tokens, ort_tokens, 0);
        const auto same_native = report("decoder_same_native_tokens", native_native, ort_native, 2e-5);
        const auto same_ort = report("decoder_same_ort_tokens", native_ort, ort_ort, 2e-5);
        report("token_effect_native_decoder", native_native, native_ort, 0);
        report("token_effect_ort_decoder", ort_native, ort_ort, 0);
        report("connected_native_vs_ort", native_native, ort_ort, 2e-5);
        std::size_t mismatch_rows = 0;
        for (std::size_t row = 0; row < batch; ++row) {
            const auto td = difference(native_tokens, ort_tokens, 0, row * 64, 64);
            if (!td.unequal) continue;
            ++mismatch_rows;
            for (std::size_t token = 0; token < 64; ++token) {
                const auto index = row * 64 + token;
                if (native_tokens[index] != ort_tokens[index])
                    std::printf("{\"kind\":\"quantized_token_mismatch\",\"row\":%zu,\"token\":%zu,\"native\":%.9g,\"ort_cpu\":%.9g}\n",
                        row, token, native_tokens[index], ort_tokens[index]);
            }
            const auto dn = difference(native_native, native_ort, 0, row * 29, 29);
            const auto dc = difference(native_native, ort_ort, 2e-5, row * 29, 29);
            std::printf("{\"kind\":\"mismatch_row_action_effect\",\"row\":%zu,\"token_mismatches\":%zu,"
                "\"native_decoder_action_max_abs\":%.12g,\"native_decoder_actions_unequal\":%zu,"
                "\"connected_action_max_abs\":%.12g,\"connected_actions_outside_tolerance\":%zu}\n",
                row, td.unequal, dn.maximum, dn.unequal, dc.maximum, dc.outside);
        }
        const bool equivalent = token_diff.unequal == 0 && same_native.outside == 0 && same_ort.outside == 0;
        std::printf("{\"kind\":\"result\",\"completed\":true,\"strict_inference_equivalence\":%s,"
            "\"token_mismatch_rows\":%zu,\"unity_or_server_parity\":false,\"training\":false}\n",
            equivalent ? "true" : "false", mismatch_rows);
        return equivalent ? 0 : 1;
    } catch (const std::exception& exception) {
        std::fprintf(stderr, "connected_controller_test_failed: %s\n", exception.what());
        return 2;
    }
}
