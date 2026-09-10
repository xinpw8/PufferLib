#define REK_G1_CUDA_DEVICE 1
#include "g1_motion_cuda.h"
#include "sonic_motion_composer_libm_candidate.h"

namespace {
constexpr unsigned THREADS = 128;
unsigned blocks_for(size_t count) {
    const size_t blocks = count / THREADS + (count % THREADS != 0);
    return static_cast<unsigned>(blocks > 65535 ? 65535 : blocks);
}

__global__ void init_kernel(SonicMotionComposerNative* composers,
        SonicMotionEntryMatcherNative* matchers,
        SonicMotionEntryMatcherNativeFeatureSlot* slots,
        const RekG1CudaMotionAsset* assets, size_t asset_count, int32_t rate,
        RekG1CudaMotionInitStatus* statuses, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            i < count; i += size_t(blockDim.x) * gridDim.x) {
        auto& status = statuses[i];
        status.composer = SONIC_MOTION_COMPOSER_NATIVE_INVALID_ARGUMENT;
        status.matcher = sonic_motion_entry_matcher_native_init(
            &matchers[i], rate, slots + i * asset_count, asset_count);
        if (status.matcher != SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK) continue;
        for (size_t j = 0; j < asset_count; ++j) {
            status.matcher = sonic_motion_entry_matcher_native_register(
                &matchers[i], &assets[j].clip, assets[j].root_local_foot_xyz,
                assets[j].root_local_foot_xyz_count);
            if (status.matcher != SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK) break;
        }
        if (status.matcher != SONIC_MOTION_ENTRY_MATCHER_NATIVE_OK) continue;
        const SonicMotionComposerNativeBackends backends = {
            sonic_motion_composer_libm_candidate_quaternion_slerp,
            sonic_motion_composer_libm_candidate_atan2_f,
            sonic_motion_composer_libm_candidate_sin_cos_f,
            sonic_motion_entry_matcher_native_callback,
            &matchers[i],
        };
        status.composer = sonic_motion_composer_native_init(
            &composers[i], rate, &backends);
    }
}

__global__ void command_kernel(SonicMotionComposerNative* composers,
        const RekG1CudaComposerCommand* commands,
        SonicMotionComposerNativeStatus* statuses, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            i < count; i += size_t(blockDim.x) * gridDim.x) {
        const auto& command = commands[i];
        switch (command.operation) {
            case REK_G1_CUDA_COMPOSER_NONE:
                statuses[i] = SONIC_MOTION_COMPOSER_NATIVE_OK;
                break;
            case REK_G1_CUDA_COMPOSER_PLAY:
                statuses[i] = sonic_motion_composer_native_play_action(
                    &composers[i], &command.clip, &command.config);
                break;
            case REK_G1_CUDA_COMPOSER_PLAY_IMMEDIATE:
                statuses[i] = sonic_motion_composer_native_play_action_immediate(
                    &composers[i], &command.clip, &command.config);
                break;
            case REK_G1_CUDA_COMPOSER_CANCEL:
                statuses[i] = sonic_motion_composer_native_cancel_action(&composers[i]);
                break;
            case REK_G1_CUDA_COMPOSER_SPEED:
                statuses[i] = sonic_motion_composer_native_set_locomotion_speed(
                    &composers[i], command.scale);
                break;
            default:
                statuses[i] = SONIC_MOTION_COMPOSER_NATIVE_INVALID_ARGUMENT;
        }
    }
}

__global__ void advance_kernel(SonicMotionComposerNative* composers,
        SonicMotionComposerNativeAdvanceResult* results,
        SonicMotionComposerNativeStatus* statuses, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            i < count; i += size_t(blockDim.x) * gridDim.x) {
        statuses[i] = sonic_motion_composer_native_advance(&composers[i], &results[i]);
    }
}

__global__ void reference_kernel(const SonicMotionComposerNative* composers,
        const SonicMotionComposerNativeReferenceTiming* timing,
        const SonicMotionComposerNativeMirrorTable* mirror,
        SonicMotionComposerNativeReferenceOutput* outputs,
        SonicMotionComposerNativeStatus* statuses, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            i < count; i += size_t(blockDim.x) * gridDim.x) {
        statuses[i] = sonic_motion_composer_native_build_reference_rows(
            &composers[i], timing, mirror, &outputs[i]);
    }
}

__global__ void heading_kernel(SonicMotionComposerNative* composers,
        float* deltas, float* ownership,
        SonicMotionComposerNativeStatus* statuses, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            i < count; i += size_t(blockDim.x) * gridDim.x) {
        statuses[i] = sonic_motion_composer_native_heading_clip_ownership(
            &composers[i], &ownership[i]);
        if (statuses[i] != SONIC_MOTION_COMPOSER_NATIVE_OK) continue;
        statuses[i] = sonic_motion_composer_native_consume_heading_delta(
            &composers[i], &deltas[i]);
    }
}

__global__ void locomotion_kernel(const RekG1NativeLocomotionState* states,
        const RekG1NativeLocomotionConfig* config,
        const RekG1NativeLocomotionStepInput* inputs,
        RekG1NativeLocomotionStepResult* results,
        RekG1NativeCommandStatus* statuses, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            i < count; i += size_t(blockDim.x) * gridDim.x) {
        statuses[i] = rek_g1_native_locomotion_step(&states[i], config,
            &inputs[i], &results[i]);
    }
}

__global__ void held_kernel(RekG1HeldInputState* states,
        const RekG1InputFrame* frames, const RekG1InputTiming* timing,
        const int* settled, const int* busy, RekG1InputDecision* decisions,
        size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            i < count; i += size_t(blockDim.x) * gridDim.x) {
        decisions[i] = rek_g1_apply_input_frame(
            &states[i], frames[i], *timing, settled[i], busy[i]);
    }
}
} // namespace

extern "C" cudaError_t rek_g1_cuda_motion_init(
        SonicMotionComposerNative* composers,
        SonicMotionEntryMatcherNative* matchers,
        SonicMotionEntryMatcherNativeFeatureSlot* slots,
        const RekG1CudaMotionAsset* assets, size_t asset_count,
        int32_t controller_rate_hz, RekG1CudaMotionInitStatus* statuses,
        size_t fighter_count, cudaStream_t stream) {
    if (!fighter_count) return cudaSuccess;
    if (!composers || !matchers || !slots || !assets || !statuses || !asset_count
            || asset_count > SIZE_MAX / fighter_count) return cudaErrorInvalidValue;
    init_kernel<<<blocks_for(fighter_count), THREADS, 0, stream>>>(composers,
        matchers, slots, assets, asset_count, controller_rate_hz, statuses, fighter_count);
    return cudaGetLastError();
}

extern "C" cudaError_t rek_g1_cuda_composer_command(
        SonicMotionComposerNative* composers, const RekG1CudaComposerCommand* commands,
        SonicMotionComposerNativeStatus* statuses, size_t count, cudaStream_t stream) {
    if (!count) return cudaSuccess;
    if (!composers || !commands || !statuses) return cudaErrorInvalidValue;
    command_kernel<<<blocks_for(count), THREADS, 0, stream>>>(composers, commands, statuses, count);
    return cudaGetLastError();
}

extern "C" cudaError_t rek_g1_cuda_composer_advance(
        SonicMotionComposerNative* composers, SonicMotionComposerNativeAdvanceResult* results,
        SonicMotionComposerNativeStatus* statuses, size_t count, cudaStream_t stream) {
    if (!count) return cudaSuccess;
    if (!composers || !results || !statuses) return cudaErrorInvalidValue;
    advance_kernel<<<blocks_for(count), THREADS, 0, stream>>>(composers, results, statuses, count);
    return cudaGetLastError();
}

extern "C" cudaError_t rek_g1_cuda_composer_reference(
        const SonicMotionComposerNative* composers,
        const SonicMotionComposerNativeReferenceTiming* timing,
        const SonicMotionComposerNativeMirrorTable* mirror,
        SonicMotionComposerNativeReferenceOutput* outputs,
        SonicMotionComposerNativeStatus* statuses, size_t count, cudaStream_t stream) {
    if (!count) return cudaSuccess;
    if (!composers || !timing || !outputs || !statuses) return cudaErrorInvalidValue;
    reference_kernel<<<blocks_for(count), THREADS, 0, stream>>>(composers, timing,
        mirror, outputs, statuses, count);
    return cudaGetLastError();
}

extern "C" cudaError_t rek_g1_cuda_composer_heading(
        SonicMotionComposerNative* composers, float* deltas, float* ownership,
        SonicMotionComposerNativeStatus* statuses, size_t count, cudaStream_t stream) {
    if (!count) return cudaSuccess;
    if (!composers || !deltas || !ownership || !statuses) return cudaErrorInvalidValue;
    heading_kernel<<<blocks_for(count), THREADS, 0, stream>>>(composers, deltas, ownership, statuses, count);
    return cudaGetLastError();
}

extern "C" cudaError_t rek_g1_cuda_locomotion_step(
        const RekG1NativeLocomotionState* states, const RekG1NativeLocomotionConfig* config,
        const RekG1NativeLocomotionStepInput* inputs, RekG1NativeLocomotionStepResult* results,
        RekG1NativeCommandStatus* statuses, size_t count, cudaStream_t stream) {
    if (!count) return cudaSuccess;
    if (!states || !config || !inputs || !results || !statuses) return cudaErrorInvalidValue;
    locomotion_kernel<<<blocks_for(count), THREADS, 0, stream>>>(states, config, inputs,
        results, statuses, count);
    return cudaGetLastError();
}

extern "C" cudaError_t rek_g1_cuda_held_input(RekG1HeldInputState* states,
        const RekG1InputFrame* frames, const RekG1InputTiming* timing,
        const int* translation_settled, const int* action_busy,
        RekG1InputDecision* decisions, size_t count, cudaStream_t stream) {
    if (!count) return cudaSuccess;
    if (!states || !frames || !timing || !translation_settled || !action_busy
            || !decisions) return cudaErrorInvalidValue;
    held_kernel<<<blocks_for(count), THREADS, 0, stream>>>(states, frames, timing,
        translation_settled, action_busy, decisions, count);
    return cudaGetLastError();
}
