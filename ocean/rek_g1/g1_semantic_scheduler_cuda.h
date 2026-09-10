#pragma once
#include "g1_motion_cuda.h"
#include "g1_semantic_action_table.h"

typedef struct RekG1CudaSemanticConfig {
    RekG1InputTiming timing;
    RekG1NativeCommandConfig command;
    RekG1NativeLocomotionConfig locomotion;
} RekG1CudaSemanticConfig;

typedef struct RekG1CudaSemanticRow {
    RekG1PufferAdapter adapter;
    RekG1NativeLocomotionState locomotion;
    RekG1NativeVelocityCommand effective_velocity;
    RekG1NativeRouteId active_route_id;
    RekG1SemanticTick semantic;
    uint8_t translation_settled;
    uint8_t action_busy;
    uint8_t recovery_active;
    int status;
} RekG1CudaSemanticRow;

/* Device-only pointers. All arrays are fighter-major. The motion library
 * retains the original C layouts. Statuses are sticky until reset:
 * 100+PufferStatus, 200+NativeCommandStatus, 300+ComposerStatus, 400 protocol. */
typedef struct RekG1CudaSemanticBuffers {
    RekG1CudaSemanticRow* rows;
    SonicMotionComposerNative* composers;
    const RekG1SemanticActionTableStorage* table;
    const RekG1CudaSemanticConfig* config;
    const RekG1CudaComposerCommand* routes;
    const int32_t* route_kinds;
    const int32_t* move_routes;
    const SonicMotionComposerNativeReferenceTiming* reference_timing;
    const SonicMotionComposerNativeMirrorTable* mirror;
    SonicMotionComposerNativeReferenceOutput* references;
    float* heading_wxyz;
    float* observation12;
    uint8_t* masks;
} RekG1CudaSemanticBuffers;

extern "C" cudaError_t rek_g1_cuda_semantic_table_init(
    RekG1SemanticActionTableStorage* table, uint32_t locomotion_ticks,
    const uint32_t* move_durations, int* status, cudaStream_t stream);

extern "C" cudaError_t rek_g1_cuda_semantic_reset(
    const RekG1CudaSemanticBuffers* buffers, const uint8_t* reset,
    const float* heading_wxyz, size_t count, cudaStream_t stream);

/* local_velocity order: angular xyz, linear xyz, as mj_objectVelocity(local=1).
 * Reference rows and headings here feed the controller BEFORE physics. */
extern "C" cudaError_t rek_g1_cuda_semantic_pre(
    const RekG1CudaSemanticBuffers* buffers, const float* actions,
    const float* local_velocity, const uint8_t* suspended,
    size_t count, cudaStream_t stream);

/* Call after physics and its fall/combat transitions. reset_event suppresses
 * composer advancement; completed spawn resets must call semantic_reset first.
 * Native input_reset/terminal rules are applied before the next action mask. */
extern "C" cudaError_t rek_g1_cuda_semantic_post(
    const RekG1CudaSemanticBuffers* buffers, const float* local_velocity,
    const int32_t* fall_phase, const uint8_t* suspended,
    const uint8_t* input_reset, const uint8_t* reset_event,
    const uint8_t* terminal, size_t count, cudaStream_t stream);
