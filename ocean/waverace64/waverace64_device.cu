#include <cuda_runtime.h>
#include <stdint.h>

typedef float obs_t;

#include "waverace64_cuda_adapter.cuh"
#include "pufferenv.h"
#include "waverace64_cuda_types.cuh"
#include "waverace64_cuda_runtime.cuh"
#include "waverace64_recomp_device.inc"

static_assert(WR64_CUDA_RDRAM_SIZE % sizeof(uint4) == 0,
    "Wave Race RDRAM must be uint4-copy aligned");
static_assert(WR64_CUDA_PAGE_SIZE % sizeof(uint4) == 0,
    "Wave Race reset pages must be uint4-copy aligned");

__device__ static inline uint32_t wr64_cuda_run_frame(
        uint8_t* rdram, WR64DeviceMachine* machine) {
    recomp_context ctx;
    wr64_device_context_init(&ctx, machine);
    ctx.r29 = (gpr)(int64_t)(int32_t)WR64_CUDA_STACK_POINTER;
    func_800922E4(rdram, &ctx);
    func_i1_802C5DF4(rdram, &ctx);
    uint32_t frame = wr64_cuda_u(rdram, WR64_CUDA_FRAME_COUNTER) + 1u;
    wr64_cuda_w(rdram, WR64_CUDA_FRAME_COUNTER, frame);
    return wr64_cuda_u(rdram, WR64_CUDA_ADDR_GAMESTATE);
}

__global__ void wr64_cuda_step_kernel(
        Env* envs, int n, uint8_t* all_rdram, size_t rdram_stride,
        const float* actions, obs_t* observations,
        float* rewards, float* terminals) {
    int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= n) return;
    Env* env = &envs[index];
    if (env->machine.error) asm("trap;");
    if (env->human.paused) {
        rewards[index] = 0.f;
        terminals[index] = 0.f;
        return;
    }
    uint8_t* rdram = all_rdram
        + (size_t)index * rdram_stride;
    const float* selected_actions = env->human.control
        ? env->human.actions
        : actions + (size_t)index * WR64_CUDA_NUM_ATNS;
    for (int action = 0; action < WR64_CUDA_NUM_ATNS; action++) {
        env->last_actions[action] = selected_actions[action];
    }
    WR64CudaStepPrelude prelude = wr64_cuda_adapter_begin_step(
        env, &env->machine, selected_actions);
    int elapsed = env->frameskip > 0 ? env->frameskip : 1;
    uint32_t game_state = wr64_cuda_u(rdram, WR64_CUDA_ADDR_GAMESTATE);
    int first_frame = env->snapshot_boundary_pending ? 1 : 0;
    env->snapshot_boundary_pending = 0;
    for (int frame = first_frame; frame < elapsed; frame++) {
        game_state = wr64_cuda_run_frame(rdram, &env->machine);
        if (env->machine.error) asm("trap;");
    }
    int done = wr64_cuda_adapter_finish_step(env, &env->machine, rdram,
        game_state, elapsed, prelude,
        observations + (size_t)index * WR64_CUDA_OBS_SIZE,
        &rewards[index], &terminals[index]);
    if (env->machine.error) asm("trap;");
    env->needs_reset = done;
}

__global__ void wr64_cuda_reset_kernel(
        Env* envs, int n, uint8_t* all_rdram, size_t rdram_stride,
        const uint8_t* canonical, const uint32_t* variant_offsets,
        const uint16_t* variant_pages, const uint8_t* variant_data,
        const WR64DeviceMachine* variant_machines,
        obs_t* observations, float* rewards, float* terminals,
        int force, int evaluation_mode, int clear_outputs,
        uint8_t* terminal_rdram, WR64CudaTerminal* terminal) {
    int index = (int)blockIdx.x;
    if (index >= n) return;
    Env* env = &envs[index];
    int lane = (int)threadIdx.x;

    if (force) {
        if (lane == 0) {
            env->needs_reset = 1;
            env->evaluation = evaluation_mode;
            if (evaluation_mode) {
                env->curriculum_laps = 3;
                env->curriculum_successes = 0;
                env->wave_episode = 0;
            }
        }
        __syncthreads();
    }
    if (!env->needs_reset) return;

    int capture_terminal = !force && index == 0 && env->evaluation;
    uint8_t* rdram = all_rdram
        + (size_t)index * rdram_stride;
    if (capture_terminal) {
        uint4* terminal_vectors = reinterpret_cast<uint4*>(terminal_rdram);
        const uint4* rdram_vectors = reinterpret_cast<const uint4*>(rdram);
        size_t vector_count = WR64_CUDA_RDRAM_SIZE / sizeof(uint4);
        for (size_t vector = (size_t)lane; vector < vector_count;
                vector += blockDim.x) {
            terminal_vectors[vector] = rdram_vectors[vector];
        }
        __syncthreads();
        if (lane == 0) {
            terminal->machine = env->machine;
            terminal->state = env->state;
            for (int action = 0; action < WR64_CUDA_NUM_ATNS; action++) {
                terminal->actions[action] = env->last_actions[action];
            }
            terminal->valid = 1;
        }
        __syncthreads();
    }

    if (lane == 0) {
        uint32_t variant = env->randomize_waves
            ? wr64_cuda_wave_next_variant(env) : 0u;
        env->reset_variant = variant;
        env->active_wave_variant = variant;
    }
    __syncthreads();
    uint32_t variant = env->reset_variant;
    uint4* rdram_vectors = reinterpret_cast<uint4*>(rdram);
    const uint4* canonical_vectors = reinterpret_cast<const uint4*>(canonical);
    size_t vector_count = WR64_CUDA_RDRAM_SIZE / sizeof(uint4);
    for (size_t vector = (size_t)lane; vector < vector_count;
            vector += blockDim.x) {
        rdram_vectors[vector] = canonical_vectors[vector];
    }
    __syncthreads();

    uint32_t first = variant_offsets[variant];
    uint32_t last = variant_offsets[variant + 1u];
    const size_t vectors_per_page = WR64_CUDA_PAGE_SIZE / sizeof(uint4);
    size_t delta_vectors = (size_t)(last - first) * vectors_per_page;
    for (size_t vector = (size_t)lane; vector < delta_vectors;
            vector += blockDim.x) {
        uint32_t local_page = (uint32_t)(vector / vectors_per_page);
        uint32_t within = (uint32_t)(vector & (vectors_per_page - 1u));
        uint32_t page = variant_pages[first + local_page];
        uint4* destination = reinterpret_cast<uint4*>(
            rdram + (size_t)page * WR64_CUDA_PAGE_SIZE);
        const uint4* source = reinterpret_cast<const uint4*>(
            variant_data + (size_t)(first + local_page)
                * WR64_CUDA_PAGE_SIZE);
        destination[within] = source[within];
    }
    __syncthreads();
    if (lane == 0) {
        env->machine = variant_machines[variant];
        env->machine.error = 0;
        env->machine.indirect_target = 0;
        int valid = wr64_cuda_adapter_reset_after_restore(env, &env->machine,
            rdram, observations + (size_t)index * WR64_CUDA_OBS_SIZE);
        if (!valid || env->machine.error) asm("trap;");
        env->needs_reset = 0;
        env->snapshot_boundary_pending = 1;
        env->episode_id++;
        if (clear_outputs) {
            rewards[index] = 0.f;
            terminals[index] = 0.f;
        }
    }
}
