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
    uint8_t direct_native_active;
    int status;
} RekG1CudaSemanticRow;

/* Direct opponent commands share native motion playback, not the categorical
 * held-input scheduler. Velocity magnitudes/signs are supplied verbatim.
 * move_index is the assigned RobotConfig index, -1 for no attempt. ExecuteMove
 * precedes locomotion: rejection_velocity is used on an ordinary rejection;
 * velocity is used otherwise (the Bot1 accepted-phase command is zero).
 * input_recovering is native input-controller recovery, not an inferred fall.
 * cancel_action reproduces CancelPunch: ignore while recovering; cancel a
 * playing action, then PlayIdle. It never changes rigid-body state. */
typedef struct RekG1CudaDirectCommand {
    RekG1NativeVelocityCommand velocity;
    RekG1NativeVelocityCommand rejection_velocity;
    int32_t move_index;
    uint8_t cancel_action;
    uint8_t input_recovering;
} RekG1CudaDirectCommand;

typedef enum RekG1CudaDirectRejection {
    REK_G1_DIRECT_NOT_REJECTED = 0,
    REK_G1_DIRECT_INVALID_MOVE = 1,
    REK_G1_DIRECT_RECOVERING = 2,
    REK_G1_DIRECT_PUNCHING = 3,
} RekG1CudaDirectRejection;

enum { REK_G1_DIRECT_HANDOFF_UNSUPPORTED = 401 };

typedef struct RekG1CudaDirectResult {
    uint8_t move_attempted;
    uint8_t move_accepted;
    uint8_t move_rejected;
    uint8_t cancelled;
    uint8_t suspended; // motor/reference suspension, not an input rejection
    RekG1CudaDirectRejection rejection_reason;
    int32_t applied_route;
    int status;
} RekG1CudaDirectResult;

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

/* All pointers are explicit device arrays, fighter-major. enabled=0 retains
 * categorical semantics; its direct command is not read. Results are written
 * for every row. Mode switches with a playing action or active categorical
 * segment fail with 401 before motion/adapter mutation. Switch at a quiescent
 * boundary or after the existing lifecycle reset; never reset a body/action
 * merely to enable an override. semantic_reset clears the row mode.
 * Direct input dispatch remains active during motor suspension: ExecuteMove,
 * CancelPunch and locomotion writes run, but reference rebuilding and post-step
 * composer/heading advancement remain suspended. */
extern "C" cudaError_t rek_g1_cuda_semantic_pre_direct(
    const RekG1CudaSemanticBuffers* buffers, const float* actions,
    const RekG1CudaDirectCommand* commands, const uint8_t* enabled,
    RekG1CudaDirectResult* results, const float* local_velocity,
    const uint8_t* suspended, size_t count, cudaStream_t stream);

/* Call after physics and its fall/combat transitions. reset_event suppresses
 * composer advancement; completed spawn resets must call semantic_reset first.
 * Native input_reset/terminal rules are applied before the next action mask. */
extern "C" cudaError_t rek_g1_cuda_semantic_post(
    const RekG1CudaSemanticBuffers* buffers, const float* local_velocity,
    const int32_t* fall_phase, const uint8_t* suspended,
    const uint8_t* input_reset, const uint8_t* reset_event,
    const uint8_t* terminal, size_t count, cudaStream_t stream);
