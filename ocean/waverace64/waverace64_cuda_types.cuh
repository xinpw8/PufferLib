#pragma once

// Shared host/device backend layout. Include this only after obs_t,
// waverace64_cuda_adapter.cuh, and pufferenv.h have been declared.

#define WR64_CUDA_FRAME_COUNTER UINT32_C(0x80151960)
#define WR64_CUDA_STACK_POINTER UINT32_C(0x80153140)
#define WR64_CUDA_RESET_THREADS 256
#define WR64_CUDA_PAGE_SIZE 4096u
#define WR64_CUDA_PAGES (WR64_CUDA_RDRAM_SIZE / WR64_CUDA_PAGE_SIZE)

typedef struct WR64CudaHumanInput {
    int32_t control;
    int32_t paused;
    float actions[WR64_CUDA_NUM_ATNS];
} WR64CudaHumanInput;

struct Env : WR64CudaAdapter {
    Agent agents[1];
    int num_agents;
    int tag;
    int boundary_reached;
    WR64DeviceMachine machine;
    int32_t needs_reset;
    int32_t snapshot_boundary_pending;
    int32_t evaluation;
    uint32_t reset_variant;
    uint32_t episode_id;
    WR64CudaHumanInput human;
    float last_actions[WR64_CUDA_NUM_ATNS];
};

typedef struct WR64CudaTerminal {
    int32_t valid;
    WR64DeviceMachine machine;
    WR64CudaState state;
    float actions[WR64_CUDA_NUM_ATNS];
} WR64CudaTerminal;

static_assert(sizeof(Log) == sizeof(WR64CudaLog),
    "Wave Race CUDA log layout changed");

// These kernels live in a separate translation unit compiled with
// --fmad=false. Keeping the exact cartridge closure out of pufferl.cu leaves
// the policy and optimizer kernels on nvcc's normal fused-math path.
__global__ void wr64_cuda_step_kernel(
    Env* envs, int n, uint8_t* all_rdram, size_t rdram_stride,
    const float* actions, obs_t* observations,
    float* rewards, float* terminals);

__global__ void wr64_cuda_reset_kernel(
    Env* envs, int n, uint8_t* all_rdram, size_t rdram_stride,
    const uint8_t* canonical, const uint32_t* variant_offsets,
    const uint16_t* variant_pages, const uint8_t* variant_data,
    const WR64DeviceMachine* variant_machines,
    obs_t* observations, float* rewards, float* terminals,
    int force, int evaluation_mode, int clear_outputs,
    uint8_t* terminal_rdram, WR64CudaTerminal* terminal);
