#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

// Fixed G1 branch of the existing SONIC encoder and decoder. All inference
// arguments are contiguous, row-major float32 buffers on the current CUDA
// device. One controller owns one stream and must be called serially.
// Every int-returning sonic_controller_* function returns 1 for success and
// 0 for failure, with error text on failure.
struct SonicController;

// Model loading, validation, allocation, and transfers occur only here.
// Explicit model batch dimensions must equal batch_size. The caller's model
// identity gate must authorize these exported ONNX files before creation.
SonicController* sonic_controller_create(
    const char* encoder_path, const char* decoder_path, std::size_t batch_size,
    cudaStream_t stream, char* error, std::size_t error_capacity);

// Call when the trainer binds a rollout stream, before capture or inference.
// The previous stream's pending work must already be ordered by the caller.
int sonic_controller_set_stream(SonicController* controller, cudaStream_t stream,
    char* error, std::size_t error_capacity);
// SHA-256 of exactly the ONNX bytes used to load the resident coefficients.
const char* sonic_controller_encoder_sha256(const SonicController* controller);
const char* sonic_controller_decoder_sha256(const SonicController* controller);

// obs: [batch,1762], tokens: [batch,64]. Only the pinned G1 mode-zero channels
// obs[4:584] and obs[601:661] are read, matching gpu_controller.py.
int sonic_controller_encode(SonicController* controller, const float* obs,
    float* tokens, char* error, std::size_t error_capacity);

// packed: [batch,640]. For each of ten samples its 64 values are the 58
// consecutive values from obs[4:584], followed by six from obs[601:661].
int sonic_controller_encode_packed(SonicController* controller,
    const float* packed, float* tokens, char* error, std::size_t error_capacity);

// obs: [batch,994], actions: [batch,29]. Action clipping and joint permutation
// belong to the environment, as in the existing controller contract.
int sonic_controller_decode(SonicController* controller, const float* obs,
    float* actions, char* error, std::size_t error_capacity);

std::size_t sonic_controller_resident_bytes(const SonicController* controller);
void sonic_controller_destroy(SonicController* controller);
