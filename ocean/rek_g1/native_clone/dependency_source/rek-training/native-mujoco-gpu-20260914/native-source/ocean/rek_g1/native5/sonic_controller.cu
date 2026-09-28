#include "sonic_controller.cuh"
#include "sonic_onnx_reader.h"

#include <cublas_v2.h>
#include <openssl/evp.h>
#include <climits>
#include <cstdio>
#include <memory>

namespace {
struct Layer { float* weight = nullptr; float* bias = nullptr; int input = 0, output = 0; };

void set_error(char* error, std::size_t capacity, const char* message) {
    if (error && capacity) {
        std::snprintf(error, capacity, "%s", message);
        error[capacity - 1] = '\0';
    }
}
void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(status));
}
void blas_check(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(
        std::string(operation) + ": cuBLAS status " + std::to_string(int(status)));
}

__global__ void pack_encoder(const float* observation, float* packed, std::size_t count) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += std::size_t(gridDim.x) * blockDim.x) {
        auto row = i / 640;
        auto sample = (i % 640) / 64;
        auto channel = i % 64;
        auto source = channel < 58 ? 4 + sample * 58 + channel : 601 + sample * 6 + channel - 58;
        packed[i] = observation[row * 1762 + source];
    }
}

__global__ void bias_activation(float* values, const float* bias,
    std::size_t count, int width, bool activate) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += std::size_t(gridDim.x) * blockDim.x) {
        float value = bias ? __fadd_rn(values[i], bias[i % width]) : values[i];
        if (activate) {
            float sigmoid = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-value)));
            value = __fmul_rn(value, sigmoid);
        }
        values[i] = value;
    }
}

__global__ void fill_bias(float* values, const float* bias, std::size_t count, int width) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += std::size_t(gridDim.x) * blockDim.x) values[i] = bias[i % width];
}

__global__ void quantize(float* values, const float* constants, std::size_t count) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += std::size_t(gridDim.x) * blockDim.x) {
        auto channel = i % 32;
        float value = tanhf(__fadd_rn(values[i], constants[channel]));
        value = __fmul_rn(value, constants[32 + channel]);
        value = __fsub_rn(value, constants[64 + channel]);
        // Preserve the source's value + (round(value) - value), including its
        // binary32 intermediates. nearbyintf uses ties-to-even on CUDA.
        value = __fadd_rn(value, __fsub_rn(nearbyintf(value), value));
        values[i] = __fdiv_rn(value, constants[96 + channel]);
    }
}

unsigned blocks(std::size_t count) {
    return unsigned(std::min<std::size_t>((count + 255) / 256, 65535));
}
} // namespace

struct SonicController {
    std::size_t batch = 0, bytes = 0;
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    Layer encoder[5], decoder[7];
    float* quantizer = nullptr;
    float* packed = nullptr;
    float* scratch[2] = {nullptr, nullptr};
    char encoder_sha256[65] = {}, decoder_sha256[65] = {};
};

void sonic_controller_destroy(SonicController* controller) {
    if (!controller) return;
    if (controller->blas) cublasDestroy(controller->blas);
    for (auto& layer : controller->encoder) { cudaFree(layer.bias); cudaFree(layer.weight); }
    for (auto& layer : controller->decoder) { cudaFree(layer.bias); cudaFree(layer.weight); }
    cudaFree(controller->quantizer);
    cudaFree(controller->packed);
    cudaFree(controller->scratch[0]);
    cudaFree(controller->scratch[1]);
    delete controller;
}

namespace {
void digest_model(const sonic_onnx::Model& model, char output[65]) {
    unsigned char digest[32];
    unsigned int count=0;
    if (EVP_Digest(model.bytes.data(),model.bytes.size(),digest,&count,EVP_sha256(),nullptr)!=1 || count!=32)
        throw std::runtime_error("SONIC model SHA-256 failed");
    const char* hex="0123456789abcdef";
    for (unsigned i=0;i<32;++i) { output[i*2]=hex[digest[i]>>4];output[i*2+1]=hex[digest[i]&15]; }
    output[64]='\0';
}

void allocate(SonicController& controller, float** pointer, std::size_t count) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(pointer), count * sizeof(float)), "allocate SONIC buffer");
    controller.bytes += count * sizeof(float);
}
void upload(SonicController& controller, float** pointer, const std::vector<float>& values) {
    allocate(controller, pointer, values.size());
    cuda_check(cudaMemcpy(*pointer, values.data(), values.size() * sizeof(float),
        cudaMemcpyHostToDevice), "upload SONIC coefficients");
}

void load_layers(SonicController& controller, const sonic_onnx::Model& model, bool encoder) {
    static const int enc_widths[] = {640, 2048, 1024, 512, 512, 64};
    static const int dec_widths[] = {994, 2048, 2048, 1024, 1024, 512, 512, 29};
    const int* widths = encoder ? enc_widths : dec_widths;
    const int count = encoder ? 5 : 7;
    Layer* layers = encoder ? controller.encoder : controller.decoder;
    for (int i = 0; i < count; ++i) {
        auto& layer = layers[i];
        layer.input = widths[i]; layer.output = widths[i + 1];
        const auto prefix = std::string(encoder ? "module.encoders.g1.module." : "module.decoders.g1_dyn.module.")
            + std::to_string(i * 2);
        auto weight_name = encoder ? prefix + ".weight" : "onnx::MatMul_" + std::to_string(136 + i);
        std::vector<std::int64_t> shape = encoder
            ? std::vector<std::int64_t>{layer.output, layer.input}
            : std::vector<std::int64_t>{layer.input, layer.output};
        upload(controller, &layer.weight, sonic_onnx::floats(model.initializers, weight_name, shape));
        upload(controller, &layer.bias, sonic_onnx::floats(model.initializers, prefix + ".bias", {layer.output}));
    }
}

int run_layers(SonicController* controller, const float* input, float* output,
    bool encoder, char* error, std::size_t capacity) {
    if (!controller || !input || !output) {
        set_error(error, capacity, "SONIC inference received null argument");
        return 0;
    }
    const auto& c = *controller;
    const Layer* layers = encoder ? c.encoder : c.decoder;
    const int count = encoder ? 5 : 7;
    const float alpha = 1, beta = encoder ? 1 : 0;
    for (int i = 0; i < count; ++i) {
        const auto& layer = layers[i];
        float* destination = i == count - 1 ? output : c.scratch[i % 2];
        auto elements = c.batch * std::size_t(layer.output);
        if (encoder) fill_bias<<<blocks(elements),256,0,c.stream>>>(
            destination, layer.bias, elements, layer.output);
        // Row-major [batch,input] is column-major [input,batch]. Encoder
        // weights are [output,input], while decoder weights are [input,output].
        auto status = cublasGemmEx(c.blas, encoder ? CUBLAS_OP_T : CUBLAS_OP_N,
            CUBLAS_OP_N, layer.output, int(c.batch), layer.input, &alpha,
            layer.weight, CUDA_R_32F, encoder ? layer.input : layer.output,
            input, CUDA_R_32F, layer.input, &beta, destination, CUDA_R_32F,
            layer.output, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT);
        if (status != CUBLAS_STATUS_SUCCESS) {
            char detail[128];
            std::snprintf(detail, sizeof(detail), "SONIC %s layer %d cuBLAS status %d",
                encoder ? "encoder" : "decoder", i, int(status));
            set_error(error, capacity, detail);
            return 0;
        }
        bias_activation<<<blocks(elements),256,0,c.stream>>>(destination,
            encoder ? nullptr : layer.bias, elements, layer.output, i != count - 1);
        auto launch = cudaPeekAtLastError();
        if (launch != cudaSuccess) { set_error(error, capacity, cudaGetErrorString(launch)); return 0; }
        input = destination;
    }
    if (encoder) {
        quantize<<<blocks(c.batch * 64),256,0,c.stream>>>(output, c.quantizer, c.batch * 64);
        auto launch = cudaPeekAtLastError();
        if (launch != cudaSuccess) { set_error(error, capacity, cudaGetErrorString(launch)); return 0; }
    }
    set_error(error, capacity, "");
    return 1;
}
} // namespace

SonicController* sonic_controller_create(const char* encoder_path,
    const char* decoder_path, std::size_t batch, cudaStream_t stream,
    char* error, std::size_t capacity) {
    std::unique_ptr<SonicController, decltype(&sonic_controller_destroy)> controller(
        nullptr, sonic_controller_destroy);
    try {
        if (batch == 0 || batch > INT_MAX || batch > std::numeric_limits<std::size_t>::max() / (4096 * sizeof(float)))
            throw std::runtime_error("invalid SONIC batch size");
        auto encoder = sonic_onnx::load(encoder_path);
        auto decoder = sonic_onnx::load(decoder_path);
        sonic_onnx::validate_io(encoder, batch, true);
        sonic_onnx::validate_io(decoder, batch, false);
        // Validate all quantizer constants before any device work.
        std::vector<float> quantizers;
        for (int i = 1; i <= 4; ++i) {
            auto values = sonic_onnx::floats(encoder.constants,
                "/quantizer/Constant_" + std::to_string(i), {32});
            if (i == 4) for (float value : values) if (value == 0)
                throw std::runtime_error("SONIC quantizer divisor is zero");
            quantizers.insert(quantizers.end(), values.begin(), values.end());
        }
        controller.reset(new SonicController);
        controller->batch = batch;
        controller->stream = stream;
        digest_model(encoder,controller->encoder_sha256);
        digest_model(decoder,controller->decoder_sha256);
        blas_check(cublasCreate(&controller->blas), "create SONIC cuBLAS handle");
        blas_check(cublasSetStream(controller->blas, stream), "set SONIC CUDA stream");
        blas_check(cublasSetMathMode(controller->blas, CUBLAS_PEDANTIC_MATH), "disable SONIC TF32");
        blas_check(cublasSetAtomicsMode(controller->blas, CUBLAS_ATOMICS_NOT_ALLOWED), "set SONIC atomics mode");
        load_layers(*controller, encoder, true);
        load_layers(*controller, decoder, false);
        upload(*controller, &controller->quantizer, quantizers);
        allocate(*controller, &controller->packed, batch * 640);
        allocate(*controller, &controller->scratch[0], batch * 2048);
        allocate(*controller, &controller->scratch[1], batch * 2048);
        set_error(error, capacity, "");
        return controller.release();
    } catch (const std::exception& exception) {
        set_error(error, capacity, exception.what());
        return nullptr;
    }
}

int sonic_controller_encode(SonicController* controller, const float* obs,
    float* tokens, char* error, std::size_t capacity) {
    if (!controller || !obs || !tokens) {
        set_error(error, capacity, "SONIC encoder received null argument"); return 0;
    }
    pack_encoder<<<blocks(controller->batch * 640),256,0,controller->stream>>>(
        obs, controller->packed, controller->batch * 640);
    auto status = cudaPeekAtLastError();
    if (status != cudaSuccess) { set_error(error, capacity, cudaGetErrorString(status)); return 0; }
    return run_layers(controller, controller->packed, tokens, true, error, capacity);
}

int sonic_controller_encode_packed(SonicController* controller, const float* packed,
    float* tokens, char* error, std::size_t capacity) {
    return run_layers(controller, packed, tokens, true, error, capacity);
}

int sonic_controller_decode(SonicController* controller, const float* obs,
    float* actions, char* error, std::size_t capacity) {
    return run_layers(controller, obs, actions, false, error, capacity);
}

std::size_t sonic_controller_resident_bytes(const SonicController* controller) {
    return controller ? controller->bytes : 0;
}

int sonic_controller_set_stream(SonicController* controller,cudaStream_t stream,
    char* error,std::size_t capacity) {
    if (!controller) { set_error(error,capacity,"null SONIC controller");return 0; }
    if (controller->stream==stream) { set_error(error,capacity,"");return 1; }
    const auto status=cublasSetStream(controller->blas,stream);
    if (status!=CUBLAS_STATUS_SUCCESS) { set_error(error,capacity,"SONIC cuBLAS stream binding failed");return 0; }
    controller->stream=stream;
    set_error(error,capacity,"");
    return 1;
}
const char* sonic_controller_encoder_sha256(const SonicController* controller) {
    return controller ? controller->encoder_sha256 : nullptr;
}
const char* sonic_controller_decoder_sha256(const SonicController* controller) {
    return controller ? controller->decoder_sha256 : nullptr;
}
