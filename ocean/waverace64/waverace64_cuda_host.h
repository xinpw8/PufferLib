#pragma once

// Reset-only host bootstrap and human-eval bridge for the CUDA Wave Race 64
// environment. Training kernels consume the returned immutable bytes directly;
// none of these functions belongs in the rollout hot path.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WR64_CUDA_HOST_ABI_VERSION UINT32_C(1)
#define WR64_CUDA_HOST_RDRAM_SIZE  UINT32_C(0x800000)
#define WR64_CUDA_HOST_MAX_WAVE_VARIANTS 128
#define WR64_CUDA_HOST_ROUTE_NODES 64
#define WR64_CUDA_HOST_ACTIONS      5

typedef struct WR64CudaHost WR64CudaHost;

typedef enum WR64CudaHostStatus {
    WR64_CUDA_HOST_OK = 0,
    WR64_CUDA_HOST_INVALID_ARGUMENT = -1,
    WR64_CUDA_HOST_ALLOCATION_FAILED = -2,
    WR64_CUDA_HOST_RUNTIME_ABI_MISMATCH = -3,
    WR64_CUDA_HOST_ROM_FAILED = -4,
    WR64_CUDA_HOST_BOOT_FAILED = -5,
    WR64_CUDA_HOST_RACE_CONTRACT_FAILED = -6,
    WR64_CUDA_HOST_RENDER_UNAVAILABLE = -7,
} WR64CudaHostStatus;

// Scalar runtime state needed by the generated device root in addition to
// RDRAM. The CUDA backend copies these fields into WR64DeviceMachine.
typedef struct WR64CudaHostMachineState {
    uint64_t ticks;
    uint16_t pad_buttons;
    int8_t pad_stick_x;
    int8_t pad_stick_y;
    int32_t resident_overlay;
    int32_t rounding_mode;
} WR64CudaHostMachineState;

// Pointer-free copy of the adapter bookkeeping. It intentionally mirrors the
// CPU environment's authoritative State fields so render and CPU/CUDA parity
// tests can consume the same values without exposing the private Env layout.
typedef struct WR64CudaHostAdapterState {
    int32_t tick;
    float prev_a;
    float prev_y;
    float prev_b;
    float episode_return;
    float dist_total;
    float progress_total;
    float max_progress;
    float velocity_x;
    float velocity_y;
    float velocity_z;
    float prev_course_progress;
    int32_t prev_node;
    int32_t prev_lap;
    int32_t checkpoints;
    int32_t prev_misses;
    int32_t misses;
    int32_t recovery;
    int32_t success;
    int32_t failed;
    int32_t disqualified;
    int32_t safety_timeout;
    int32_t env_fault;
} WR64CudaHostAdapterState;

// Complete host-produced reset metadata. route_arc and route_pred are the
// checked Sunny Beach topology cache used by the CPU adapter. A CUDA backend
// may upload this once and duplicate it entirely on device.
typedef struct WR64CudaHostSnapshotInfo {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t runtime_abi_version;
    uint32_t wave_seed;
    uint32_t wave_variant;
    int32_t randomize_waves;
    WR64CudaHostMachineState machine;
    WR64CudaHostAdapterState adapter;
    float vertical_origin;
    float route_arc[WR64_CUDA_HOST_ROUTE_NODES];
    int32_t route_pred[WR64_CUDA_HOST_ROUTE_NODES];
    float route_total;
    int32_t route_nodes;
    int32_t route_valid;
} WR64CudaHostSnapshotInfo;

enum {
    // Reset the renderer's camera, wake, and terminal presentation state before
    // displaying this frame.
    WR64_CUDA_HOST_RENDER_NEW_EPISODE = 1u << 0,
    // Capture this copied state as the official terminal frame and hold it
    // until the evaluator presses Enter.
    WR64_CUDA_HOST_RENDER_TERMINAL = 1u << 1,
};

typedef struct WR64CudaHostRenderInput {
    uint32_t flags;
    uint32_t reserved;
    WR64CudaHostMachineState machine;
    WR64CudaHostAdapterState adapter;
    float policy_actions[WR64_CUDA_HOST_ACTIONS];
} WR64CudaHostRenderInput;

typedef struct WR64CudaHostRenderOutput {
    float actions[WR64_CUDA_HOST_ACTIONS];
    int32_t human_control;
    int32_t paused;
    int32_t terminal_ready;
    int32_t window_ready;
    int32_t window_should_close;
} WR64CudaHostRenderOutput;

// The returned context owns only the ROM path and an optional evaluator mirror.
// Boot machines are short-lived and snapshots are written directly into the
// caller's staging buffer, avoiding one retained 8 MiB host copy per variant.
int wr64_cuda_host_create(const char* rom_path, WR64CudaHost** out,
    char* error, size_t error_size);
void wr64_cuda_host_destroy(WR64CudaHost* host);

size_t wr64_cuda_host_rdram_size(void);
uint64_t wr64_cuda_host_wave_ticks(
    uint32_t wave_seed, uint32_t wave_variant);

// Boots the actual US Rev 1 cartridge to the fixed Sunny Beach Time Trial,
// writes its time-zero RDRAM into caller-owned storage, and returns all scalar
// reset metadata. randomize_waves must be zero or one. Fixed mode preserves the
// cartridge runtime's WR_BOOT_OS_TIME; randomized mode derives the boot time
// from wave_seed and a wave_variant in [0, 127]. Calls sharing one host are
// thread-safe and may run in parallel during vector creation. This call is
// initialization-only.
int wr64_cuda_host_boot_snapshot(WR64CudaHost* host,
    uint32_t wave_seed, int32_t randomize_waves, uint32_t wave_variant,
    void* rdram_out, size_t rdram_size,
    WR64CudaHostSnapshotInfo* info_out,
    char* error, size_t error_size);

// Human-eval-only projection. The selected device RDRAM image is copied into a
// private host mirror and passed through the existing state renderer. Human
// controls are returned as the same five discrete action indices expected by
// the policy interface. Training never calls this function.
int wr64_cuda_host_render(WR64CudaHost* host,
    const void* rdram, size_t rdram_size,
    const WR64CudaHostRenderInput* input,
    WR64CudaHostRenderOutput* output,
    char* error, size_t error_size);

// Closes only the evaluator mirror and its window. The bootstrap context and
// ROM path remain valid for subsequent snapshot requests.
void wr64_cuda_host_render_close(WR64CudaHost* host);

#ifdef __cplusplus
}
#endif
