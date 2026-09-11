#pragma once

#include <cuda_runtime_api.h>
#include "held_input.h"
#include "native_locomotion_command.h"
#include "sonic_motion_composer_native.h"
#include "sonic_motion_entry_matcher_native.h"

typedef struct RekG1CudaMotionAsset {
    SonicMotionComposerNativeClip clip;
    const float* root_local_foot_xyz;
    size_t root_local_foot_xyz_count;
} RekG1CudaMotionAsset;

typedef struct RekG1CudaMotionInitStatus {
    SonicMotionComposerNativeStatus composer;
    SonicMotionEntryMatcherNativeStatus matcher;
} RekG1CudaMotionInitStatus;

typedef enum RekG1CudaComposerOperation {
    REK_G1_CUDA_COMPOSER_NONE = 0,
    REK_G1_CUDA_COMPOSER_PLAY = 1,
    REK_G1_CUDA_COMPOSER_PLAY_IMMEDIATE = 2,
    REK_G1_CUDA_COMPOSER_CANCEL = 3,
    REK_G1_CUDA_COMPOSER_SPEED = 4,
} RekG1CudaComposerOperation;

typedef struct RekG1CudaComposerCommand {
    RekG1CudaComposerOperation operation;
    SonicMotionComposerNativeClip clip;
    SonicMotionComposerNativeConfig config;
    float scale;
} RekG1CudaComposerCommand;

/* Every pointer, including nested clip/feature/output pointers, is on device.
 * Initial slot storage has fighter_count * asset_count entries. */
extern "C" cudaError_t rek_g1_cuda_motion_init(
    SonicMotionComposerNative* composers,
    SonicMotionEntryMatcherNative* matchers,
    SonicMotionEntryMatcherNativeFeatureSlot* slots,
    const RekG1CudaMotionAsset* assets,
    size_t asset_count,
    int32_t controller_rate_hz,
    RekG1CudaMotionInitStatus* statuses,
    size_t fighter_count,
    cudaStream_t stream);

extern "C" cudaError_t rek_g1_cuda_composer_command(
    SonicMotionComposerNative* composers,
    const RekG1CudaComposerCommand* commands,
    SonicMotionComposerNativeStatus* statuses,
    size_t fighter_count,
    cudaStream_t stream);

extern "C" cudaError_t rek_g1_cuda_composer_advance(
    SonicMotionComposerNative* composers,
    SonicMotionComposerNativeAdvanceResult* results,
    SonicMotionComposerNativeStatus* statuses,
    size_t fighter_count,
    cudaStream_t stream);

extern "C" cudaError_t rek_g1_cuda_composer_reference(
    const SonicMotionComposerNative* composers,
    const SonicMotionComposerNativeReferenceTiming* timing,
    const SonicMotionComposerNativeMirrorTable* mirror_table,
    SonicMotionComposerNativeReferenceOutput* outputs,
    SonicMotionComposerNativeStatus* statuses,
    size_t fighter_count,
    cudaStream_t stream);

extern "C" cudaError_t rek_g1_cuda_composer_heading(
    SonicMotionComposerNative* composers,
    float* deltas,
    float* ownership,
    SonicMotionComposerNativeStatus* statuses,
    size_t fighter_count,
    cudaStream_t stream);

extern "C" cudaError_t rek_g1_cuda_locomotion_step(
    const RekG1NativeLocomotionState* states,
    const RekG1NativeLocomotionConfig* config,
    const RekG1NativeLocomotionStepInput* inputs,
    RekG1NativeLocomotionStepResult* results,
    RekG1NativeCommandStatus* statuses,
    size_t fighter_count,
    cudaStream_t stream);

extern "C" cudaError_t rek_g1_cuda_held_input(
    RekG1HeldInputState* states,
    const RekG1InputFrame* frames,
    const RekG1InputTiming* timing,
    const int* translation_settled,
    const int* action_busy,
    RekG1InputDecision* decisions,
    size_t fighter_count,
    cudaStream_t stream);
