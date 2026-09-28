#pragma once

#include <cuda_runtime_api.h>
#include "g1_combat_tick.h"

/*
 * All array arguments and nested contact/impact-event pointers reside on the
 * CUDA device. These launchers enqueue work on stream without synchronizing
 * or transferring arena data. Semantic errors are returned per arena; failed
 * operations preserve their output just as the original functions do.
 */
extern "C" cudaError_t rek_g1_cuda_combat_init(
    RekG1CombatArenaState* states,
    RekG1CombatTickStatus* statuses,
    size_t arena_count,
    cudaStream_t stream);

extern "C" cudaError_t rek_g1_cuda_combat_substep(
    const RekG1CombatArenaState* states,
    const RekG1HitDetectorConfig* config,
    const RekG1CombatSubstepInput* inputs,
    RekG1CombatSubstepResult* results,
    RekG1CombatTickStatus* statuses,
    size_t arena_count,
    cudaStream_t stream);

extern "C" cudaError_t rek_g1_cuda_combat_spawn_reset(
    const RekG1CombatArenaState* states,
    RekG1CombatArenaState* next_states,
    RekG1CombatTickStatus* statuses,
    size_t arena_count,
    cudaStream_t stream);

extern "C" cudaError_t rek_g1_cuda_fall_step(
    const RekG1FallConfig* config,
    const RekG1FallState* states,
    const RekG1FallSample* samples,
    RekG1FallStepResult* results,
    RekG1FallStatus* statuses,
    size_t fighter_count,
    cudaStream_t stream);

extern "C" cudaError_t rek_g1_cuda_hit_process(
    RekG1HitDetectorState* states,
    const RekG1HitDetectorConfig* config,
    const RekG1HitContact* contacts,
    RekG1HitResult* results,
    int* statuses,
    size_t contact_count,
    cudaStream_t stream);
