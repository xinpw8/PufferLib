#define REK_G1_CUDA_DEVICE 1
#include "g1_combat_cuda.h"

namespace {
constexpr unsigned THREADS = 128;

unsigned blocks_for(size_t count) {
    const size_t blocks = count / THREADS + (count % THREADS != 0);
    return static_cast<unsigned>(blocks > 65535 ? 65535 : blocks);
}

__global__ void init_kernel(RekG1CombatArenaState* states,
        RekG1CombatTickStatus* statuses, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            i < count; i += size_t(blockDim.x) * gridDim.x) {
        statuses[i] = rek_g1_combat_arena_init(&states[i]);
    }
}

__global__ void substep_kernel(const RekG1CombatArenaState* states,
        const RekG1HitDetectorConfig* config,
        const RekG1CombatSubstepInput* inputs,
        RekG1CombatSubstepResult* results,
        RekG1CombatTickStatus* statuses, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            i < count; i += size_t(blockDim.x) * gridDim.x) {
        statuses[i] = rek_g1_combat_arena_substep(
            &states[i], config, &inputs[i], &results[i]);
    }
}

__global__ void reset_kernel(const RekG1CombatArenaState* states,
        RekG1CombatArenaState* next_states,
        RekG1CombatTickStatus* statuses, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            i < count; i += size_t(blockDim.x) * gridDim.x) {
        statuses[i] = rek_g1_combat_arena_apply_spawn_reset(
            &states[i], &next_states[i]);
    }
}

__global__ void fall_kernel(const RekG1FallConfig* config,
        const RekG1FallState* states, const RekG1FallSample* samples,
        RekG1FallStepResult* results, RekG1FallStatus* statuses, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            i < count; i += size_t(blockDim.x) * gridDim.x) {
        statuses[i] = rek_g1_fall_state_step(
            config, &states[i], &samples[i], &results[i]);
    }
}

__global__ void hit_kernel(RekG1HitDetectorState* states,
        const RekG1HitDetectorConfig* config, const RekG1HitContact* contacts,
        RekG1HitResult* results, int* statuses, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            i < count; i += size_t(blockDim.x) * gridDim.x) {
        statuses[i] = rek_g1_hit_detector_process(
            &states[i], config, &contacts[i], &results[i]);
    }
}
} // namespace

extern "C" cudaError_t rek_g1_cuda_combat_init(
        RekG1CombatArenaState* states, RekG1CombatTickStatus* statuses,
        size_t arena_count, cudaStream_t stream) {
    if (arena_count == 0) return cudaSuccess;
    if (!states || !statuses) return cudaErrorInvalidValue;
    init_kernel<<<blocks_for(arena_count), THREADS, 0, stream>>>(
        states, statuses, arena_count);
    return cudaGetLastError();
}

extern "C" cudaError_t rek_g1_cuda_combat_substep(
        const RekG1CombatArenaState* states,
        const RekG1HitDetectorConfig* config,
        const RekG1CombatSubstepInput* inputs,
        RekG1CombatSubstepResult* results, RekG1CombatTickStatus* statuses,
        size_t arena_count, cudaStream_t stream) {
    if (arena_count == 0) return cudaSuccess;
    if (!states || !config || !inputs || !results || !statuses) {
        return cudaErrorInvalidValue;
    }
    substep_kernel<<<blocks_for(arena_count), THREADS, 0, stream>>>(
        states, config, inputs, results, statuses, arena_count);
    return cudaGetLastError();
}

extern "C" cudaError_t rek_g1_cuda_combat_spawn_reset(
        const RekG1CombatArenaState* states,
        RekG1CombatArenaState* next_states, RekG1CombatTickStatus* statuses,
        size_t arena_count, cudaStream_t stream) {
    if (arena_count == 0) return cudaSuccess;
    if (!states || !next_states || !statuses) return cudaErrorInvalidValue;
    reset_kernel<<<blocks_for(arena_count), THREADS, 0, stream>>>(
        states, next_states, statuses, arena_count);
    return cudaGetLastError();
}

extern "C" cudaError_t rek_g1_cuda_fall_step(
        const RekG1FallConfig* config,
        const RekG1FallState* states, const RekG1FallSample* samples,
        RekG1FallStepResult* results, RekG1FallStatus* statuses,
        size_t fighter_count, cudaStream_t stream) {
    if (fighter_count == 0) return cudaSuccess;
    if (!config || !states || !samples || !results || !statuses) {
        return cudaErrorInvalidValue;
    }
    fall_kernel<<<blocks_for(fighter_count), THREADS, 0, stream>>>(
        config, states, samples, results, statuses, fighter_count);
    return cudaGetLastError();
}

extern "C" cudaError_t rek_g1_cuda_hit_process(
        RekG1HitDetectorState* states,
        const RekG1HitDetectorConfig* config, const RekG1HitContact* contacts,
        RekG1HitResult* results, int* statuses,
        size_t contact_count, cudaStream_t stream) {
    if (contact_count == 0) return cudaSuccess;
    if (!states || !config || !contacts || !results || !statuses) {
        return cudaErrorInvalidValue;
    }
    hit_kernel<<<blocks_for(contact_count), THREADS, 0, stream>>>(
        states, config, contacts, results, statuses, contact_count);
    return cudaGetLastError();
}
