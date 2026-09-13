#include "sonic_controller.cuh"
#include "sonic_onnx_reader.h"
#include <onnxruntime_c_api.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>

namespace {
bool comparison_failed = false;
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
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
        ok(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "native5-sonic-oracle", &environment));
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
        const std::int64_t input_shape[] = {std::int64_t(batch), encode ? 1762 : 994};
        const std::int64_t output_shape[] = {std::int64_t(batch), encode ? 64 : 29};
        std::vector<float> output(batch * std::size_t(output_shape[1]));
        OrtValue* in = nullptr;
        OrtValue* out = nullptr;
        ok(api->CreateTensorWithDataAsOrtValue(memory, input.data(), input.size() * sizeof(float),
            input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in));
        ok(api->CreateTensorWithDataAsOrtValue(memory, output.data(), output.size() * sizeof(float),
            output_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &out));
        const char* input_name = "obs_dict";
        const char* output_name = encode ? "encoded_tokens" : "action";
        const OrtValue* inputs[] = {in};
        auto status = api->Run(encode ? encoder : decoder, nullptr, &input_name, inputs, 1,
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

double compare(const std::vector<float>& actual, const std::vector<float>& expected,
    const char* label, double tolerance) {
    check(actual.size() == expected.size(), "comparison size mismatch");
    double maximum = 0;
    std::size_t differing = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        check(std::isfinite(actual[i]) && std::isfinite(expected[i]), "nonfinite controller result");
        auto difference = std::abs(double(actual[i]) - double(expected[i]));
        maximum = std::max(maximum, difference);
        differing += difference > tolerance;
        if (difference > tolerance && differing <= 3)
            std::printf("%s mismatch index=%zu actual=%.9g expected=%.9g\n",label,i,actual[i],expected[i]);
    }
    std::printf("%s max_abs=%.9g outside_tolerance=%zu tolerance=%.9g\n", label, maximum, differing, tolerance);
    comparison_failed = comparison_failed || differing != 0;
    return maximum;
}

std::vector<float> copy_to_host(const float* source, std::size_t count) {
    std::vector<float> result(count);
    cuda_check(cudaMemcpy(result.data(), source, count * sizeof(float), cudaMemcpyDeviceToHost));
    return result;
}
} // namespace

int main(int argc, char** argv) {
    try {
        check(argc == 4 || (argc == 5 && std::string(argv[4]) == "--inspect-only"),
            "usage: test_sonic_controller BATCH ENCODER.onnx DECODER.onnx [--inspect-only]");
        char* end = nullptr;
        const auto parsed = std::strtoull(argv[1], &end, 10);
        check(end && *end == '\0' && parsed > 0 && parsed < 1000000, "invalid test batch");
        const std::size_t batch = std::size_t(parsed);
        auto encoder = sonic_onnx::load(argv[2]);
        auto decoder = sonic_onnx::load(argv[3]);
        sonic_onnx::validate_io(encoder, batch, true);
        sonic_onnx::validate_io(decoder, batch, false);
        bool wrong_batch_rejected = false;
        try { sonic_onnx::validate_io(encoder, batch + 1, true); }
        catch (const std::exception&) { wrong_batch_rejected = true; }
        check(wrong_batch_rejected, "wrong explicit batch was accepted");
        const int encoder_widths[] = {640,2048,1024,512,512,64};
        const int decoder_widths[] = {994,2048,2048,1024,1024,512,512,29};
        std::size_t coefficient_count = 0;
        for (int i = 0; i < 5; ++i) {
            auto name = "module.encoders.g1.module." + std::to_string(i * 2);
            coefficient_count += sonic_onnx::floats(encoder.initializers, name + ".weight",
                {encoder_widths[i+1], encoder_widths[i]}).size();
            coefficient_count += sonic_onnx::floats(encoder.initializers, name + ".bias", {encoder_widths[i+1]}).size();
        }
        for (int i = 0; i < 7; ++i) {
            coefficient_count += sonic_onnx::floats(decoder.initializers,
                "onnx::MatMul_" + std::to_string(136 + i), {decoder_widths[i], decoder_widths[i+1]}).size();
            coefficient_count += sonic_onnx::floats(decoder.initializers,
                "module.decoders.g1_dyn.module." + std::to_string(i * 2) + ".bias", {decoder_widths[i+1]}).size();
        }
        for (int i = 1; i <= 4; ++i)
            coefficient_count += sonic_onnx::floats(encoder.constants, "/quantizer/Constant_" + std::to_string(i), {32}).size();
        // Malformed wire data must fail in host parsing, before any CUDA call.
        bool truncated_rejected = false;
        const unsigned char truncated[] = {0x3a,0xff};
        try { sonic_onnx::Reader reader({truncated,sizeof(truncated)}); sonic_onnx::Field f; reader.next(f); }
        catch (const std::exception&) { truncated_rejected = true; }
        check(truncated_rejected, "truncated protobuf length was accepted");
        std::printf("native_onnx_inspection batch=%zu coefficients=%zu float32_bytes=%zu wrong_batch_rejected=1 truncated_rejected=1\n",
            batch, coefficient_count, coefficient_count * sizeof(float));
        if (argc == 5) return 0;

        CpuOracle oracle(argv[2], argv[3]);
        std::vector<float> encoder_input(batch * 1762, 0), decoder_input(batch * 994), packed(batch * 640);
        for (std::size_t row = 0; row < batch; ++row) {
            for (std::size_t column = 0; column < 580; ++column)
                encoder_input[row * 1762 + 4 + column] = probe(row,column,1);
            for (std::size_t column = 0; column < 60; ++column)
                encoder_input[row * 1762 + 601 + column] = probe(row,column,2);
            for (std::size_t column = 0; column < 994; ++column)
                decoder_input[row * 994 + column] = probe(row,column,3);
            for (std::size_t sample = 0; sample < 10; ++sample) {
                for (std::size_t column = 0; column < 58; ++column)
                    packed[row * 640 + sample * 64 + column] = encoder_input[row * 1762 + 4 + sample * 58 + column];
                for (std::size_t column = 0; column < 6; ++column)
                    packed[row * 640 + sample * 64 + 58 + column] = encoder_input[row * 1762 + 601 + sample * 6 + column];
            }
        }
        auto expected_tokens = oracle.run(true, encoder_input, batch);
        auto expected_actions = oracle.run(false, decoder_input, batch);
        cudaStream_t stream = nullptr;
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        char error[1024] = {};
        std::unique_ptr<SonicController, decltype(&sonic_controller_destroy)> controller(
            sonic_controller_create(argv[2],argv[3],batch,stream,error,sizeof(error)), sonic_controller_destroy);
        check(controller != nullptr, error);
        DeviceBuffer device_encoder(encoder_input.size()), device_decoder(decoder_input.size()), device_packed(packed.size());
        DeviceBuffer device_tokens(batch * 64), device_actions(batch * 29);
        cuda_check(cudaMemcpy(device_encoder.data, encoder_input.data(), encoder_input.size() * 4, cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(device_decoder.data, decoder_input.data(), decoder_input.size() * 4, cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(device_packed.data, packed.data(), packed.size() * 4, cudaMemcpyHostToDevice));
        auto run = [&] {
            check(sonic_controller_encode(controller.get(), device_encoder.data,device_tokens.data,error,sizeof(error)),error);
            check(sonic_controller_decode(controller.get(), device_decoder.data,device_actions.data,error,sizeof(error)),error);
        };
        run();
        cuda_check(cudaStreamSynchronize(stream));
        auto actual_tokens = copy_to_host(device_tokens.data,batch * 64);
        auto actual_actions = copy_to_host(device_actions.data,batch * 29);
        compare(actual_tokens,expected_tokens,"encoder_vs_ort_cpu",2e-5);
        compare(actual_actions,expected_actions,"decoder_vs_ort_cpu",2e-5);
        check(sonic_controller_encode_packed(controller.get(),device_packed.data,device_tokens.data,error,sizeof(error)),error);
        cuda_check(cudaStreamSynchronize(stream));
        compare(copy_to_host(device_tokens.data,batch * 64),actual_tokens,"packed_vs_full_encoder",0);
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        cuda_check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        run();
        cuda_check(cudaStreamEndCapture(stream,&graph));
        cuda_check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
        for (int i = 0; i < 3; ++i) cuda_check(cudaGraphLaunch(executable,stream));
        cuda_check(cudaStreamSynchronize(stream));
        compare(copy_to_host(device_tokens.data,batch * 64),actual_tokens,"graph_vs_eager_encoder",0);
        compare(copy_to_host(device_actions.data,batch * 29),actual_actions,"graph_vs_eager_decoder",0);
        check(!comparison_failed, "native SONIC numerical comparison failed");
        std::printf("native_sonic_pass batch=%zu resident_bytes=%zu ort_version=%s python_used=0 inference_backend=cublas_fp32_pedantic\n",
            batch,sonic_controller_resident_bytes(controller.get()),OrtGetApiBase()->GetVersionString());
        cuda_check(cudaGraphExecDestroy(executable));
        cuda_check(cudaGraphDestroy(graph));
        controller.reset();
        cuda_check(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& exception) {
        std::fprintf(stderr,"native_sonic_test_failed: %s\n",exception.what());
        return 1;
    }
}
