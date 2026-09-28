#pragma once

#include <cuda_runtime_api.h>
#include <stddef.h>
#include <stdint.h>

#include "g1_combat_tick.h"
#include "sonic_motion_composer_native.h"

enum RekG1CudaNativeCombatStatus {
    REK_G1_CUDA_NATIVE_COMBAT_OK = 0,
    REK_G1_CUDA_NATIVE_COMBAT_INPUT_INVALID = 1,
    REK_G1_CUDA_NATIVE_COMBAT_FALL_REJECTED = 2,
    REK_G1_CUDA_NATIVE_COMBAT_CONTACT_REJECTED = 3,
    REK_G1_CUDA_NATIVE_COMBAT_TICK_REJECTED = 4,
    REK_G1_CUDA_NATIVE_COMBAT_RESET_REJECTED = 5,
    REK_G1_CUDA_NATIVE_COMBAT_OVERFLOW = 6,
};

typedef struct RekG1CudaNativeCombatState {
    RekG1CombatArenaState combat;
    RekG1FallState fall[2];
    float reset_complete_not_before_seconds;
    uint8_t reset_pending;
    uint8_t pending_episode_reset;
} RekG1CudaNativeCombatState;

#if defined(__cplusplus)
extern "C" {
#endif

size_t rek_g1_cuda_native_combat_state_size(void);
size_t rek_g1_cuda_native_combat_contact_size(void);
size_t rek_g1_cuda_native_combat_impact_event_size(void);
size_t rek_g1_cuda_native_combat_impact_event_count(void);

/* One-time, same-source upload from g1_strike_catalog.c. */
cudaError_t rek_g1_cuda_native_combat_upload_catalog(
    RekG1ImpactEvent* impact_events,
    int32_t* route_event_offsets,
    int32_t* route_event_counts
);

cudaError_t rek_g1_cuda_native_combat_init(
    RekG1CudaNativeCombatState* states,
    int32_t* statuses,
    size_t arena_count,
    cudaStream_t stream
);

/*
 * Starts one 20 ms controller tick. Arrays use arena or arena*2 row order.
 * A previous ROUND_ENDED is converted to an immediate episode-reset request.
 */
cudaError_t rek_g1_cuda_native_combat_begin_tick(
    RekG1CudaNativeCombatState* states,
    uint32_t* tick_fall_events,
    uint32_t* tick_signals,
    uint32_t* tick_referee_calls,
    int32_t* tick_score_delta,
    uint32_t* tick_attributed_contacts,
    uint32_t* tick_scored_contacts,
    uint8_t* terminals,
    uint8_t* episode_reset,
    uint8_t* input_reset,
    int32_t* statuses,
    size_t arena_count,
    cudaStream_t stream
);

/*
 * Consumes one successfully measured post-mj_step 2 ms state. Candidate order
 * contains valid directed source rows grouped by arena while preserving raw
 * Warp contact-slot and direction order. All pointers reside on device.
 */
cudaError_t rek_g1_cuda_native_combat_post_step(
    RekG1CudaNativeCombatState* states,
    const float* fall_floats,
    const int64_t* fall_integers,
    const uint8_t* fall_valid,
    const int64_t* hit_integers,
    const float* hit_floats,
    const uint8_t* candidate_valid,
    const int64_t* candidate_order,
    const int64_t* candidate_offsets,
    const int64_t* candidate_counts,
    const uint8_t* arena_scan_valid,
    const float* arena_time_seconds,
    const SonicMotionComposerNative* composers,
    const int32_t* active_route_ids,
    const RekG1ImpactEvent* impact_events,
    const int32_t* route_event_offsets,
    const int32_t* route_event_counts,
    RekG1HitContact* packed_contacts,
    uint32_t* tick_fall_events,
    uint32_t* tick_signals,
    uint32_t* tick_referee_calls,
    int32_t* tick_score_delta,
    uint32_t* tick_attributed_contacts,
    uint32_t* tick_scored_contacts,
    uint8_t* terminals,
    uint8_t* input_reset,
    uint8_t* dampened,
    uint8_t* begin_reset,
    uint8_t* complete_reset,
    uint8_t* clear_contacts,
    int32_t* statuses,
    size_t arena_count,
    size_t directed_candidate_capacity,
    cudaStream_t stream
);

/* Opt-in unobserved substep: preserve observe's validation before deferring
 * only its packed outputs. Call observe before consuming packed outputs. */
cudaError_t rek_g1_cuda_native_combat_post_step_deferred(
    RekG1CudaNativeCombatState* states,
    const float* fall_floats,
    const int64_t* fall_integers,
    const uint8_t* fall_valid,
    const int64_t* hit_integers,
    const float* hit_floats,
    const uint8_t* candidate_valid,
    const int64_t* candidate_order,
    const int64_t* candidate_offsets,
    const int64_t* candidate_counts,
    const uint8_t* arena_scan_valid,
    const float* arena_time_seconds,
    const SonicMotionComposerNative* composers,
    const int32_t* active_route_ids,
    const RekG1ImpactEvent* impact_events,
    const int32_t* route_event_offsets,
    const int32_t* route_event_counts,
    RekG1HitContact* packed_contacts,
    uint32_t* tick_fall_events,
    uint32_t* tick_signals,
    uint32_t* tick_referee_calls,
    int32_t* tick_score_delta,
    uint32_t* tick_attributed_contacts,
    uint32_t* tick_scored_contacts,
    uint8_t* terminals,
    uint8_t* input_reset,
    uint8_t* dampened,
    uint8_t* begin_reset,
    uint8_t* complete_reset,
    uint8_t* clear_contacts,
    int32_t* statuses,
    size_t arena_count,
    size_t directed_candidate_capacity,
    cudaStream_t stream
);

/* Pack the existing fall15/fight39 row ABIs and reward/terminal vectors. */
cudaError_t rek_g1_cuda_native_combat_observe(
    const RekG1CudaNativeCombatState* states,
    const float* fall_floats,
    const int64_t* fall_integers,
    const uint8_t* fall_valid,
    const uint32_t* tick_fall_events,
    const uint32_t* tick_signals,
    const uint32_t* tick_referee_calls,
    const int32_t* tick_score_delta,
    const uint32_t* tick_attributed_contacts,
    const uint32_t* tick_scored_contacts,
    const uint8_t* terminals,
    float* fall_observations,
    float* fight_observations,
    float* rewards,
    float* terminal_observations,
    int32_t* statuses,
    size_t arena_count,
    cudaStream_t stream
);

#if defined(__cplusplus)
}
#endif
