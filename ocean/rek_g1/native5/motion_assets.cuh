#pragma once

#include "runtime_api.h"
#include "../g1_semantic_scheduler_cuda.h"
#include <array>
#include <string>
#include <vector>

/* Native ownership of the existing motion assets and semantic scheduler.
 * All exposed data pointers are device addresses, except scheduler_host.
 * Construction validates assets and synchronizes initialization. Step wrappers
 * only enqueue existing kernels. Errors are C++ exceptions for the caller's
 * runtime C ABI to translate into rek_native5_error(). */
class RekNative5Motion {
public:
    explicit RekNative5Motion(const RekNative5Config& config, cudaStream_t stream);
    ~RekNative5Motion();
    RekNative5Motion(const RekNative5Motion&) = delete;
    RekNative5Motion& operator=(const RekNative5Motion&) = delete;

    void reset(const uint8_t* reset_flags, const float* heading_wxyz, cudaStream_t stream);
    void pre(const float* actions, const float* local_velocity,
        const uint8_t* suspended, cudaStream_t stream);
    void post(const float* local_velocity, const int32_t* fall_phase,
        const uint8_t* suspended, const uint8_t* input_reset,
        const uint8_t* reset_event, const uint8_t* terminal, cudaStream_t stream);
    void check_status(cudaStream_t stream) const;

    /* File parsing, hash validation, shape validation, and heading
     * normalization only. Does not initialize CUDA or allocate device memory. */
    static std::string validate_assets(const char* assets_root, const char* features_root);

    size_t count = 0;
    size_t asset_count = 0;
    std::string manifest_sha256;
    std::array<float, 29> idle_positions = {};
    std::array<float, 4> idle_root_xyzw = {};
    RekG1CudaSemanticBuffers scheduler_host = {};
    RekG1CudaSemanticBuffers* scheduler = nullptr;
    RekG1CudaSemanticRow* rows = nullptr;
    SonicMotionComposerNative* composers = nullptr;
    SonicMotionEntryMatcherNative* matchers = nullptr;
    float* positions = nullptr;
    float* next_positions = nullptr;
    float* rotations = nullptr;
    float* heading = nullptr;
    float* observation12 = nullptr;
    uint8_t* masks = nullptr;
    uint8_t* zero_flags = nullptr;
    uint8_t* all_flags = nullptr;

private:
    std::vector<void*> allocations_;
    void* allocate(size_t bytes, bool zero = false);
    void* upload(const void* source, size_t bytes);
    void release() noexcept;
};
