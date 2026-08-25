#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

typedef float obs_t;
#include "ocean/waverace64/waverace64_cuda_adapter.cuh"
#include "pufferenv.h"
#include "ocean/waverace64/waverace64_cuda_types.cuh"
#include "ocean/waverace64/waverace64_cuda_host.h"

/*
Build from the PufferLib repository root after generating the exhaustive
device closure:

  python3 ocean/waverace64/generate_cuda_recomp.py \
    --runtime "$WR64_DIR" --rom "$ROM" \
    --output build/stress/waverace64_recomp_device.inc
  gcc -O2 -std=c11 -fopenmp -ffp-contract=off -I. -Isrc -Ivendor \
    -Iraylib-5.5_linux_aarch64/include -I"$WR64_DIR/runtime" \
    -I"$WR64_DIR/RecompiledFuncs" -c \
    ocean/waverace64/waverace64_cuda_host.c \
    -o build/stress/waverace64_cuda_host.o
  nvcc -O2 --threads 0 -arch=native -std=c++17 \
    -DWR64_SUPPRESS_DISPLAY_WATER_NORMALS=1 \
    --fmad=false --ftz=false --prec-div=true --prec-sqrt=true \
    --diag-suppress=68,177,550 -I. -Ibuild/stress -Isrc -Ivendor \
    -Iraylib-5.5_linux_aarch64/include \
    -c ocean/waverace64/waverace64_device.cu \
    -o build/stress/waverace64_device.o
  nvcc -O2 -arch=native -std=c++17 -I. -Isrc -Ivendor \
    -Iraylib-5.5_linux_aarch64/include \
    -c tests/test_waverace64_cuda_stress.cu \
    -o build/stress/test_waverace64_cuda_stress.o
  nvcc -arch=native build/stress/test_waverace64_cuda_stress.o \
    build/stress/waverace64_device.o \
    build/stress/waverace64_cuda_host.o "$WR64_DIR/libwr64.a" \
    -Xcompiler=-fopenmp -lcuda -lm -lpthread \
    -o build/stress/test_waverace64_cuda_stress

The optional arguments are decisions, environments, and wave variants. The
environment count must be even, and the variant count must be a power of two.
Run WR64_STRESS_VMM_FAULT_PROBE=1 with a small configuration in a separate
process to prove that a write beyond the granularity-rounded guard mapping
faults in the VMM release hole. That intentional probe poisons its CUDA
context, reports the expected error, and exits without entering the stress.
*/

#define WR64_STRESS_DEFAULT_DECISIONS 4096
#define WR64_STRESS_DEFAULT_ENVS 32
#define WR64_STRESS_DEFAULT_VARIANTS 16
#define WR64_STRESS_FRAMESKIP 2
#define WR64_STRESS_GUARD_BYTES (64u * 1024u)
#define WR64_STRESS_RDRAM_STRIDE (UINT64_C(1) << 32)
#define WR64_STRESS_CANARY 0xA5u
#define WR64_STRESS_SEED UINT32_C(0xC001D00D)
#define WR64_STRESS_REWARD_MODE 0

typedef struct WR64StressStats {
    unsigned long long env_steps;
    unsigned long long native_updates;
    unsigned long long boundary_skips;
    unsigned long long autoresets;
    unsigned long long terminals;
    unsigned long long successes;
    unsigned long long failures;
    unsigned long long ordinary_failures;
    unsigned long long disqualifications;
    unsigned long long safety_timeouts;
    unsigned long long env_faults;
    unsigned long long recovery_samples;
    unsigned long long recovery_entries;
    unsigned long long machine_errors;
    unsigned long long indirect_faults;
    unsigned long long nan_observations;
    unsigned long long nan_rewards;
    unsigned long long nan_terminals;
    unsigned long long nan_state;
    unsigned long long invalid_terminals;
    unsigned long long pair_mismatches;
    unsigned long long pair_env_mismatches;
    unsigned long long pair_output_mismatches;
    unsigned long long guard_faults;
} WR64StressStats;

typedef struct WR64StressHash {
    unsigned long long xor_hash;
    unsigned long long sum_hash;
} WR64StressHash;

typedef struct WR64StressPool {
    uint32_t variants;
    uint32_t total_pages;
    std::vector<uint8_t> images;
    std::vector<WR64CudaHostSnapshotInfo> infos;
    std::vector<uint32_t> offsets;
    std::vector<uint16_t> pages;
    std::vector<uint8_t> data;
    std::vector<WR64DeviceMachine> machines;
} WR64StressPool;

typedef struct WR64StressDevice {
    int n;
    uint32_t variants;
    size_t rdram_stride;
    size_t mapped_bytes;
    CUdeviceptr reservation;
    size_t reservation_bytes;
    CUmemGenericAllocationHandle* rdram_handles;
    Env* envs;
    uint8_t* rdram;
    uint8_t* canonical;
    uint32_t* offsets;
    uint16_t* pages;
    uint8_t* data;
    WR64DeviceMachine* machines;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    uint8_t* terminal_rdram;
    WR64CudaTerminal* terminal;
    WR64StressStats* stats;
    uint32_t* variant_seen;
    uint32_t* last_episode;
    int32_t* last_recovery;
    WR64StressHash* hashes;
} WR64StressDevice;

static int g_pass = -1;
static int g_decision = -1;

static void wr64_require(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr,
        "FAIL CUDA pass=%d decision=%d operation=%s error=%s\n",
        g_pass, g_decision, operation, cudaGetErrorString(status));
    std::exit(2);
}

static void wr64_driver_require(CUresult status, const char* operation) {
    if (status == CUDA_SUCCESS) return;
    const char* name = nullptr;
    const char* detail = nullptr;
    (void)cuGetErrorName(status, &name);
    (void)cuGetErrorString(status, &detail);
    std::fprintf(stderr,
        "FAIL CUDA-driver pass=%d decision=%d operation=%s error=%s (%s)\n",
        g_pass, g_decision, operation,
        detail ? detail : "unknown", name ? name : "unknown");
    std::exit(2);
}

static WR64DeviceMachine wr64_machine_from_host(
        const WR64CudaHostMachineState& source) {
    WR64DeviceMachine machine = {};
    machine.ticks = source.ticks;
    machine.pad_buttons = source.pad_buttons;
    machine.pad_stick_x = source.pad_stick_x;
    machine.pad_stick_y = source.pad_stick_y;
    machine.resident_overlay = source.resident_overlay;
    machine.rounding_mode = source.rounding_mode;
    return machine;
}

static bool wr64_route_compatible(const WR64CudaHostSnapshotInfo& a,
        const WR64CudaHostSnapshotInfo& b) {
    return a.route_total == b.route_total
        && a.route_nodes == b.route_nodes
        && a.route_valid == b.route_valid
        && std::memcmp(a.route_arc, b.route_arc, sizeof(a.route_arc)) == 0
        && std::memcmp(a.route_pred, b.route_pred, sizeof(a.route_pred)) == 0;
}

static WR64StressPool wr64_build_pool(
        WR64CudaHost* host, uint32_t variants) {
    WR64StressPool pool = {};
    pool.variants = variants;
    pool.images.resize((size_t)variants * WR64_CUDA_RDRAM_SIZE);
    pool.infos.resize(variants);
    std::vector<int> status(variants);
    std::vector<char> errors((size_t)variants * 256u);

    #pragma omp parallel for schedule(dynamic)
    for (int variant = 0; variant < (int)variants; variant++) {
        status[variant] = wr64_cuda_host_boot_snapshot(host,
            WR64_STRESS_SEED, 1, (uint32_t)variant,
            pool.images.data() + (size_t)variant * WR64_CUDA_RDRAM_SIZE,
            WR64_CUDA_RDRAM_SIZE, &pool.infos[variant],
            errors.data() + (size_t)variant * 256u, 256u);
    }
    for (uint32_t variant = 0; variant < variants; variant++) {
        if (status[variant] != WR64_CUDA_HOST_OK) {
            std::fprintf(stderr, "snapshot variant %u failed (%d): %s\n",
                variant, status[variant],
                errors.data() + (size_t)variant * 256u);
            std::exit(2);
        }
        const WR64CudaHostSnapshotInfo& info = pool.infos[variant];
        if (info.abi_version != WR64_CUDA_HOST_ABI_VERSION
                || info.struct_size != sizeof(info)
                || !info.route_valid
                || info.wave_variant != variant
                || !wr64_route_compatible(pool.infos[0], info)) {
            std::fprintf(stderr,
                "snapshot variant %u returned incompatible metadata\n",
                variant);
            std::exit(2);
        }
    }

    pool.offsets.resize((size_t)variants + 1u);
    for (uint32_t variant = 0; variant < variants; variant++) {
        pool.offsets[variant] = (uint32_t)pool.pages.size();
        const uint8_t* image = pool.images.data()
            + (size_t)variant * WR64_CUDA_RDRAM_SIZE;
        for (uint32_t page = 0; page < WR64_CUDA_PAGES; page++) {
            size_t byte = (size_t)page * WR64_CUDA_PAGE_SIZE;
            if (std::memcmp(pool.images.data() + byte,
                    image + byte, WR64_CUDA_PAGE_SIZE) == 0) continue;
            pool.pages.push_back((uint16_t)page);
            pool.data.insert(pool.data.end(), image + byte,
                image + byte + WR64_CUDA_PAGE_SIZE);
        }
    }
    pool.offsets[variants] = (uint32_t)pool.pages.size();
    pool.total_pages = (uint32_t)pool.pages.size();
    pool.machines.resize(variants);
    for (uint32_t variant = 0; variant < variants; variant++) {
        pool.machines[variant] = wr64_machine_from_host(
            pool.infos[variant].machine);
    }
    return pool;
}

static Env wr64_make_env(const WR64CudaHostSnapshotInfo& canonical,
        uint32_t group, uint32_t variants) {
    Env env = {};
    env.frameskip = WR64_STRESS_FRAMESKIP;
    env.rng = group;
    env.randomize_waves = 1;
    env.wave_seed = WR64_STRESS_SEED;
    env.wave_rng_state = wr64_cuda_wave_stream_seed(
        WR64_STRESS_SEED, group);
    env.wave_boot_variant = group & (variants - 1u);
    env.active_wave_variant = env.wave_boot_variant;
    env.wave_variants = (int32_t)variants;
    env.reward_speed = 0.f;
    env.reward_progress = 1.f;
    env.reward_slip = 0.f;
    env.reward_checkpoint = 0.1f;
    env.reward_miss = 0.5f;
    env.reward_finish = 10.f;
    env.reward_fail = 2.f;
    env.discount = 0.9995f;
    env.reward_mode = WR64_STRESS_REWARD_MODE;
    env.curriculum_start_laps = 3;
    env.curriculum_max_laps = 3;
    env.curriculum_successes_per_lap = 1;
    env.curriculum_laps = 3;
    env.vertical_origin = canonical.vertical_origin;
    std::memcpy(env.route_arc, canonical.route_arc, sizeof(env.route_arc));
    std::memcpy(env.route_pred, canonical.route_pred, sizeof(env.route_pred));
    env.route_total = canonical.route_total;
    env.route_nodes = canonical.route_nodes;
    env.route_valid = canonical.route_valid;
    env.num_agents = 1;
    env.needs_reset = 1;
    return env;
}

static void wr64_alloc_device(WR64StressDevice* device,
        int n, const WR64StressPool& pool) {
    std::memset(device, 0, sizeof(*device));
    device->n = n;
    device->variants = pool.variants;
    device->rdram_stride = (size_t)WR64_STRESS_RDRAM_STRIDE;
    device->mapped_bytes = 0;
    if ((uint64_t)n > (uint64_t)SIZE_MAX / WR64_STRESS_RDRAM_STRIDE) {
        std::fprintf(stderr, "RDRAM VMM reservation size overflow\n");
        std::exit(2);
    }
    wr64_require(cudaFree(nullptr), "initialize CUDA runtime context");
    wr64_driver_require(cuInit(0), "initialize CUDA driver");
    CUcontext context = nullptr;
    wr64_driver_require(cuCtxGetCurrent(&context), "read CUDA context");
    if (context == nullptr) {
        std::fprintf(stderr, "CUDA context is not current\n");
        std::exit(2);
    }
    int runtime_device = 0;
    wr64_require(cudaGetDevice(&runtime_device), "read CUDA device");
    CUdevice cu_device;
    wr64_driver_require(cuDeviceGet(&cu_device, runtime_device),
        "resolve CUDA device");
    int vmm_supported = 0;
    wr64_driver_require(cuDeviceGetAttribute(&vmm_supported,
        CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
        cu_device), "query CUDA VMM support");
    if (!vmm_supported) {
        std::fprintf(stderr, "CUDA VMM is required for the stress gate\n");
        std::exit(2);
    }
    CUmemAllocationProp properties = {};
    properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    properties.location.id = (int)cu_device;
    properties.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    size_t granularity = 0;
    wr64_driver_require(cuMemGetAllocationGranularity(&granularity,
        &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
        "query VMM granularity");
    if (granularity == 0) {
        std::fprintf(stderr, "CUDA VMM returned zero granularity\n");
        std::exit(2);
    }
    size_t desired_mapping = WR64_CUDA_RDRAM_SIZE
        + WR64_STRESS_GUARD_BYTES;
    device->mapped_bytes = ((desired_mapping + granularity - 1)
        / granularity) * granularity;
    if (device->mapped_bytes <= WR64_CUDA_RDRAM_SIZE
            || device->mapped_bytes >= device->rdram_stride) {
        std::fprintf(stderr,
            "invalid guarded RDRAM mapping size %zu at granularity %zu\n",
            device->mapped_bytes, granularity);
        std::exit(2);
    }
    device->reservation_bytes = (size_t)n * device->rdram_stride;
    wr64_driver_require(cuMemAddressReserve(&device->reservation,
        device->reservation_bytes, 0, 0, 0),
        "reserve isolated RDRAM address space");
    device->rdram_handles = (CUmemGenericAllocationHandle*)std::calloc(
        (size_t)n, sizeof(*device->rdram_handles));
    if (!device->rdram_handles) {
        std::fprintf(stderr, "RDRAM VMM handle allocation failed\n");
        std::exit(2);
    }
    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = (int)cu_device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    for (int index = 0; index < n; index++) {
        CUdeviceptr address = device->reservation
            + (CUdeviceptr)(size_t)index * device->rdram_stride;
        wr64_driver_require(cuMemCreate(&device->rdram_handles[index],
            device->mapped_bytes, &properties, 0),
            "allocate guarded RDRAM physical memory");
        wr64_driver_require(cuMemMap(address, device->mapped_bytes, 0,
            device->rdram_handles[index], 0), "map guarded RDRAM");
        wr64_driver_require(cuMemSetAccess(address, device->mapped_bytes,
            &access, 1), "enable guarded RDRAM access");
    }
    device->rdram = (uint8_t*)(uintptr_t)device->reservation;
#define WR64_ALLOC(member, bytes, label) \
    wr64_require(cudaMalloc((void**)&device->member, (bytes)), (label))
    WR64_ALLOC(envs, (size_t)n * sizeof(Env), "allocate envs");
    WR64_ALLOC(canonical, WR64_CUDA_RDRAM_SIZE, "allocate canonical reset");
    WR64_ALLOC(offsets, ((size_t)pool.variants + 1u) * sizeof(uint32_t),
        "allocate reset offsets");
    if (!pool.pages.empty()) {
        WR64_ALLOC(pages, pool.pages.size() * sizeof(uint16_t),
            "allocate reset pages");
        WR64_ALLOC(data, pool.data.size(), "allocate reset data");
    }
    WR64_ALLOC(machines,
        (size_t)pool.variants * sizeof(WR64DeviceMachine),
        "allocate reset machines");
    WR64_ALLOC(observations,
        (size_t)n * WR64_CUDA_OBS_SIZE * sizeof(float),
        "allocate observations");
    WR64_ALLOC(actions, (size_t)n * WR64_CUDA_NUM_ATNS * sizeof(float),
        "allocate actions");
    WR64_ALLOC(rewards, (size_t)n * sizeof(float), "allocate rewards");
    WR64_ALLOC(terminals, (size_t)n * sizeof(float), "allocate terminals");
    WR64_ALLOC(terminal_rdram, WR64_CUDA_RDRAM_SIZE,
        "allocate terminal RDRAM");
    WR64_ALLOC(terminal, sizeof(WR64CudaTerminal),
        "allocate terminal metadata");
    WR64_ALLOC(stats, sizeof(WR64StressStats), "allocate stress stats");
    WR64_ALLOC(variant_seen, (size_t)pool.variants * sizeof(uint32_t),
        "allocate variant coverage");
    WR64_ALLOC(last_episode, (size_t)n * sizeof(uint32_t),
        "allocate episode audit");
    WR64_ALLOC(last_recovery, (size_t)n * sizeof(int32_t),
        "allocate recovery audit");
    WR64_ALLOC(hashes, (size_t)n * sizeof(WR64StressHash),
        "allocate state hashes");
#undef WR64_ALLOC

    wr64_require(cudaMemcpy(device->canonical, pool.images.data(),
        WR64_CUDA_RDRAM_SIZE, cudaMemcpyHostToDevice),
        "upload canonical reset");
    wr64_require(cudaMemcpy(device->offsets, pool.offsets.data(),
        pool.offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice),
        "upload reset offsets");
    if (!pool.pages.empty()) {
        wr64_require(cudaMemcpy(device->pages, pool.pages.data(),
            pool.pages.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
            "upload reset pages");
        wr64_require(cudaMemcpy(device->data, pool.data.data(),
            pool.data.size(), cudaMemcpyHostToDevice),
            "upload reset data");
    }
    wr64_require(cudaMemcpy(device->machines, pool.machines.data(),
        pool.machines.size() * sizeof(WR64DeviceMachine),
        cudaMemcpyHostToDevice), "upload reset machines");
}

static void wr64_free_device(WR64StressDevice* device) {
    cudaFree(device->hashes);
    cudaFree(device->last_recovery);
    cudaFree(device->last_episode);
    cudaFree(device->variant_seen);
    cudaFree(device->stats);
    cudaFree(device->terminal);
    cudaFree(device->terminal_rdram);
    cudaFree(device->terminals);
    cudaFree(device->rewards);
    cudaFree(device->actions);
    cudaFree(device->observations);
    cudaFree(device->machines);
    cudaFree(device->data);
    cudaFree(device->pages);
    cudaFree(device->offsets);
    cudaFree(device->canonical);
    cudaFree(device->envs);
    if (device->reservation != 0) {
        for (int index = 0; index < device->n; index++) {
            CUdeviceptr address = device->reservation
                + (CUdeviceptr)(size_t)index * device->rdram_stride;
            wr64_driver_require(cuMemUnmap(address, device->mapped_bytes),
                "unmap guarded RDRAM");
        }
    }
    if (device->rdram_handles) {
        for (int index = 0; index < device->n; index++) {
            if (device->rdram_handles[index] != 0) {
                wr64_driver_require(
                    cuMemRelease(device->rdram_handles[index]),
                    "release guarded RDRAM");
            }
        }
        std::free(device->rdram_handles);
    }
    if (device->reservation != 0) {
        wr64_driver_require(cuMemAddressFree(device->reservation,
            device->reservation_bytes), "release RDRAM address space");
    }
    std::memset(device, 0, sizeof(*device));
}

__device__ static inline uint32_t wr64_stress_mix32(uint32_t x) {
    x ^= x >> 16;
    x *= UINT32_C(0x7FEB352D);
    x ^= x >> 15;
    x *= UINT32_C(0x846CA68B);
    x ^= x >> 16;
    return x;
}

__device__ static inline unsigned long long wr64_stress_mix64(
        unsigned long long x) {
    x ^= x >> 30;
    x *= UINT64_C(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    x *= UINT64_C(0x94D049BB133111EB);
    x ^= x >> 31;
    return x;
}

__device__ static inline int wr64_stress_nearest_stick(float desired) {
    const int detents[15] = {
        -80, -68, -56, -44, -32, -20, -10, 0,
        10, 20, 32, 44, 56, 68, 80,
    };
    int best = 0;
    float best_error = INFINITY;
    for (int index = 0; index < 15; index++) {
        float error = fabsf(desired - (float)detents[index]);
        if (error < best_error) {
            best = index;
            best_error = error;
        }
    }
    return best;
}

__device__ static inline void wr64_stress_route_actions(
        const float* observations, float* actions, bool wrong_side) {
    const float steer_gain = 134.347687f;
    const float throttle_angle = 0.228220314f;
    const float dampen_angle = 0.242280304f;
    const float high_throttle_angle = 2.70581746f;
    const float pass_scale = wrong_side ? -1.75507069f : 1.75507069f;
    const float curve_near_blend = 0.427550882f;
    const float curve_distance = 484.638611f;
    const float slide_angle = 0.245567679f;
    const float route_total = 29078.811f;

    float center_x = observations[17] * observations[19];
    float center_z = -observations[18] * observations[19];
    float pass_x = observations[24] * observations[26];
    float pass_z = -observations[25] * observations[26];
    float dx = center_x + pass_scale * (pass_x - center_x);
    float dz = center_z + pass_scale * (pass_z - center_z);
    if (observations[30] > 0.5f) {
        float next_x = observations[27] * observations[29];
        float next_z = -observations[28] * observations[29];
        float blend = observations[26] < curve_distance / route_total
            ? curve_near_blend : 0.f;
        dx = dx * (1.f - blend) + next_x * blend;
        dz = dz * (1.f - blend) + next_z * blend;
    }
    float angle = atan2f(dz, dx);
    int steer = (int)lrintf(angle * steer_gain);
    steer = steer > 80 ? 80 : (steer < -80 ? -80 : steer);
    int throttle = fabsf(angle) <= throttle_angle;
    int dampen = fabsf(angle) > dampen_angle;
    int throttle_alias = observations[15] * (float)WR64_CUDA_MAX_STEPS
            >= 59.5f
        && fabsf(angle) > high_throttle_angle;
    int slide = fabsf(angle) > slide_angle;
    actions[0] = (float)wr64_stress_nearest_stick((float)steer);
    actions[1] = 1.f;
    actions[2] = (float)(throttle || throttle_alias);
    actions[3] = (float)dampen;
    actions[4] = (float)slide;
}

__global__ static void wr64_stress_actions(float* actions,
        const float* observations, int n, uint32_t decision,
        int perturb_env) {
    int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= n) return;
    uint32_t group = (uint32_t)index >> 1;
    uint32_t state = wr64_stress_mix32(WR64_STRESS_SEED
        ^ wr64_stress_mix32(decision + UINT32_C(0x9E3779B9))
        ^ wr64_stress_mix32(group + UINT32_C(0xD1B54A35)));
    uint32_t r0 = wr64_stress_mix32(state + UINT32_C(0x11111111));
    uint32_t r1 = wr64_stress_mix32(state + UINT32_C(0x33333333));
    uint32_t r2 = wr64_stress_mix32(state + UINT32_C(0x55555555));
    uint32_t r3 = wr64_stress_mix32(state + UINT32_C(0x77777777));
    uint32_t r4 = wr64_stress_mix32(state + UINT32_C(0x99999999));
    float* action = actions + (size_t)index * WR64_CUDA_NUM_ATNS;
    uint32_t route_mode = group & 7u;
    if (route_mode <= 1u) {
        wr64_stress_route_actions(observations
            + (size_t)index * WR64_CUDA_OBS_SIZE,
            action, route_mode == 1u);
    } else {
        action[0] = (float)(r0 % 15u);
        action[1] = (float)(r1 % 9u);
        action[2] = (float)((r2 % 4u) != 0u);
        action[3] = (float)((r3 % 5u) == 0u);
        action[4] = (float)((r4 % 5u) == 0u);
    }
    if (index == perturb_env) {
        action[0] = (float)(((uint32_t)action[0] + 7u) % 15u);
        action[2] = 1.f - action[2];
    }
}

__global__ static void wr64_stress_pre_step(const Env* envs, int n,
        WR64StressStats* stats) {
    int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= n) return;
    const Env* env = &envs[index];
    int elapsed = env->frameskip > 0 ? env->frameskip : 1;
    int boundary_skip = env->snapshot_boundary_pending ? 1 : 0;
    int updates = elapsed - boundary_skip;
    atomicAdd(&stats->env_steps, 1ull);
    atomicAdd(&stats->native_updates, (unsigned long long)updates);
    atomicAdd(&stats->boundary_skips, (unsigned long long)boundary_skip);
}

__device__ static inline void wr64_stress_check_float(
        float value, unsigned long long* counter) {
    if (!isfinite(value)) atomicAdd(counter, 1ull);
}

__global__ static void wr64_stress_post_step(const Env* envs, int n,
        const float* observations, const float* rewards,
        const float* terminals, int32_t* last_recovery,
        WR64StressStats* stats) {
    int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= n) return;
    const Env* env = &envs[index];
    if (env->machine.error) atomicAdd(&stats->machine_errors, 1ull);
    if (env->machine.indirect_target) {
        atomicAdd(&stats->indirect_faults, 1ull);
    }
    for (int obs = 0; obs < WR64_CUDA_OBS_SIZE; obs++) {
        wr64_stress_check_float(
            observations[(size_t)index * WR64_CUDA_OBS_SIZE + obs],
            &stats->nan_observations);
    }
    wr64_stress_check_float(rewards[index], &stats->nan_rewards);
    wr64_stress_check_float(terminals[index], &stats->nan_terminals);
    if (terminals[index] != 0.f && terminals[index] != 1.f) {
        atomicAdd(&stats->invalid_terminals, 1ull);
    }
    const float state_values[] = {
        env->state.prev_a, env->state.prev_y, env->state.prev_b,
        env->state.episode_return, env->state.dist_total,
        env->state.progress_total, env->state.max_progress,
        env->state.velocity_x, env->state.velocity_y,
        env->state.velocity_z, env->state.prev_course_progress,
    };
    for (unsigned field = 0;
            field < sizeof(state_values) / sizeof(state_values[0]); field++) {
        wr64_stress_check_float(state_values[field], &stats->nan_state);
    }
    const float* log_values = (const float*)&env->log;
    for (unsigned field = 0; field < sizeof(env->log) / sizeof(float);
            field++) {
        wr64_stress_check_float(log_values[field], &stats->nan_state);
    }
    int recovery = env->state.recovery;
    if (recovery) atomicAdd(&stats->recovery_samples, 1ull);
    if (recovery && !last_recovery[index]) {
        atomicAdd(&stats->recovery_entries, 1ull);
    }
    last_recovery[index] = recovery;
    if (terminals[index] == 1.f) {
        atomicAdd(&stats->terminals, 1ull);
        if (env->state.success) atomicAdd(&stats->successes, 1ull);
        if (env->state.failed) atomicAdd(&stats->failures, 1ull);
        if (env->state.failed && !env->state.disqualified
                && !env->state.safety_timeout) {
            atomicAdd(&stats->ordinary_failures, 1ull);
        }
        if (env->state.disqualified) {
            atomicAdd(&stats->disqualifications, 1ull);
        }
        if (env->state.safety_timeout) {
            atomicAdd(&stats->safety_timeouts, 1ull);
        }
        if (env->state.env_fault) atomicAdd(&stats->env_faults, 1ull);
    }
}

__global__ static void wr64_stress_pair_compare(const Env* envs, int n,
        const float* observations, const float* rewards,
        const float* terminals, WR64StressStats* stats) {
    int pair = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int a = pair * 2;
    int b = a + 1;
    if (b >= n) return;
    bool env_mismatch = false;
    const uint8_t* env_a = (const uint8_t*)&envs[a];
    const uint8_t* env_b = (const uint8_t*)&envs[b];
    for (size_t byte = 0; !env_mismatch && byte < sizeof(Env); byte++) {
        env_mismatch = env_a[byte] != env_b[byte];
    }
    const uint32_t* reward_a = (const uint32_t*)&rewards[a];
    const uint32_t* reward_b = (const uint32_t*)&rewards[b];
    const uint32_t* terminal_a = (const uint32_t*)&terminals[a];
    const uint32_t* terminal_b = (const uint32_t*)&terminals[b];
    bool output_mismatch = *reward_a != *reward_b
        || *terminal_a != *terminal_b;
    for (int obs = 0; !output_mismatch && obs < WR64_CUDA_OBS_SIZE; obs++) {
        const uint32_t* obs_a = (const uint32_t*)&observations[
            (size_t)a * WR64_CUDA_OBS_SIZE + obs];
        const uint32_t* obs_b = (const uint32_t*)&observations[
            (size_t)b * WR64_CUDA_OBS_SIZE + obs];
        output_mismatch = *obs_a != *obs_b;
    }
    if (env_mismatch) atomicAdd(&stats->pair_env_mismatches, 1ull);
    if (output_mismatch) atomicAdd(&stats->pair_output_mismatches, 1ull);
    if (env_mismatch || output_mismatch) {
        atomicAdd(&stats->pair_mismatches, 1ull);
    }
}

__global__ static void wr64_stress_post_reset(const Env* envs, int n,
        uint32_t variants, uint32_t* last_episode,
        int32_t* last_recovery, uint32_t* variant_seen,
        WR64StressStats* stats) {
    int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= n) return;
    const Env* env = &envs[index];
    uint32_t episode = env->episode_id;
    if (episode != last_episode[index]) {
        if (last_episode[index] != 0u) {
            atomicAdd(&stats->autoresets, 1ull);
        }
        last_episode[index] = episode;
        last_recovery[index] = env->state.recovery;
        if (env->active_wave_variant < variants) {
            atomicExch(&variant_seen[env->active_wave_variant], 1u);
        } else {
            atomicAdd(&stats->machine_errors, 1ull);
        }
    }
}

__global__ static void wr64_stress_guard_check(const uint8_t* rdram,
        int n, size_t stride, WR64StressStats* stats) {
    size_t guard_byte = (size_t)(blockIdx.x * blockDim.x + threadIdx.x);
    size_t total = (size_t)n * WR64_STRESS_GUARD_BYTES;
    for (; guard_byte < total;
            guard_byte += (size_t)gridDim.x * blockDim.x) {
        int env = (int)(guard_byte / WR64_STRESS_GUARD_BYTES);
        size_t within = guard_byte % WR64_STRESS_GUARD_BYTES;
        size_t address = (size_t)env * stride
            + WR64_CUDA_RDRAM_SIZE + within;
        if (rdram[address] != WR64_STRESS_CANARY) {
            atomicAdd(&stats->guard_faults, 1ull);
        }
    }
}

__global__ static void wr64_stress_vmm_fault_probe(
        uint8_t* rdram, size_t first_unmapped_byte) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        rdram[first_unmapped_byte] = 0x5Au;
    }
}

__global__ static void wr64_stress_hash_kernel(const Env* envs, int n,
        const uint8_t* all_rdram, size_t stride,
        const float* observations, const float* rewards,
        const float* terminals, WR64StressHash* hashes) {
    int env_index = (int)blockIdx.x;
    if (env_index >= n) return;
    int lane = (int)threadIdx.x;
    unsigned long long local_xor = 0;
    unsigned long long local_sum = 0;
    const unsigned long long* words = (const unsigned long long*)(
        all_rdram + (size_t)env_index * stride);
    size_t word_count = WR64_CUDA_RDRAM_SIZE / sizeof(*words);
    for (size_t word = (size_t)lane; word < word_count;
            word += blockDim.x) {
        unsigned long long mixed = wr64_stress_mix64(words[word]
            ^ wr64_stress_mix64((unsigned long long)word
                + UINT64_C(0xA0761D6478BD642F)));
        local_xor ^= mixed;
        local_sum += mixed;
    }
    const uint8_t* env_bytes = (const uint8_t*)&envs[env_index];
    for (size_t byte = (size_t)lane; byte < sizeof(Env);
            byte += blockDim.x) {
        unsigned long long mixed = wr64_stress_mix64(
            (unsigned long long)env_bytes[byte]
            ^ wr64_stress_mix64((unsigned long long)byte
                + UINT64_C(0xE7037ED1A0B428DB)));
        local_xor ^= mixed;
        local_sum += mixed;
    }
    const uint8_t* obs_bytes = (const uint8_t*)(observations
        + (size_t)env_index * WR64_CUDA_OBS_SIZE);
    size_t output_bytes = WR64_CUDA_OBS_SIZE * sizeof(float);
    for (size_t byte = (size_t)lane; byte < output_bytes;
            byte += blockDim.x) {
        unsigned long long mixed = wr64_stress_mix64(
            (unsigned long long)obs_bytes[byte]
            ^ wr64_stress_mix64((unsigned long long)byte
                + UINT64_C(0x8EBC6AF09C88C6E3)));
        local_xor ^= mixed;
        local_sum += mixed;
    }
    if (lane == 0) {
        const uint32_t* reward_bits = (const uint32_t*)&rewards[env_index];
        const uint32_t* terminal_bits = (const uint32_t*)&terminals[env_index];
        unsigned long long mixed = wr64_stress_mix64(
            (unsigned long long)*reward_bits
            | ((unsigned long long)*terminal_bits << 32));
        local_xor ^= mixed;
        local_sum += mixed;
    }
    __shared__ unsigned long long shared_xor[256];
    __shared__ unsigned long long shared_sum[256];
    shared_xor[lane] = local_xor;
    shared_sum[lane] = local_sum;
    __syncthreads();
    for (int width = blockDim.x / 2; width > 0; width >>= 1) {
        if (lane < width) {
            shared_xor[lane] ^= shared_xor[lane + width];
            shared_sum[lane] += shared_sum[lane + width];
        }
        __syncthreads();
    }
    if (lane == 0) {
        hashes[env_index].xor_hash = shared_xor[0];
        hashes[env_index].sum_hash = shared_sum[0];
    }
}

static void wr64_launch_reset(WR64StressDevice* device,
        int force, int clear_outputs) {
    wr64_cuda_reset_kernel<<<device->n, WR64_CUDA_RESET_THREADS>>>(
        device->envs, device->n, device->rdram, device->rdram_stride,
        device->canonical, device->offsets, device->pages, device->data,
        device->machines, device->observations, device->rewards,
        device->terminals, force, 0, clear_outputs,
        device->terminal_rdram, device->terminal);
    wr64_require(cudaGetLastError(), "launch production reset kernel");
}

static void wr64_verify_initial_snapshots(WR64StressDevice* device,
        const WR64StressPool& pool) {
    std::vector<Env> envs(device->n);
    std::vector<uint8_t> image(WR64_CUDA_RDRAM_SIZE);
    wr64_require(cudaMemcpy(envs.data(), device->envs,
        (size_t)device->n * sizeof(Env), cudaMemcpyDeviceToHost),
        "download reset environments");
    for (int index = 0; index < device->n; index += 2) {
        uint32_t variant = envs[index].active_wave_variant;
        if (variant >= pool.variants) {
            std::fprintf(stderr, "invalid active variant %u for env %d\n",
                variant, index);
            std::exit(1);
        }
        wr64_require(cudaMemcpy(image.data(),
            device->rdram + (size_t)index * device->rdram_stride,
            WR64_CUDA_RDRAM_SIZE, cudaMemcpyDeviceToHost),
            "download restored snapshot");
        const uint8_t* expected = pool.images.data()
            + (size_t)variant * WR64_CUDA_RDRAM_SIZE;
        if (std::memcmp(image.data(), expected, WR64_CUDA_RDRAM_SIZE) != 0) {
            size_t byte = 0;
            while (byte < WR64_CUDA_RDRAM_SIZE
                    && image[byte] == expected[byte]) byte++;
            std::fprintf(stderr,
                "FAIL restored snapshot env=%d variant=%u offset=0x%zx\n",
                index, variant, byte);
            std::exit(1);
        }
    }
}

static int wr64_count_variants(const std::vector<uint32_t>& seen) {
    int count = 0;
    for (uint32_t value : seen) count += value != 0;
    return count;
}

static void wr64_diagnose_pairs(const WR64StressDevice* device) {
    std::vector<Env> envs(device->n);
    std::vector<float> observations(
        (size_t)device->n * WR64_CUDA_OBS_SIZE);
    std::vector<float> rewards(device->n);
    std::vector<float> terminals(device->n);
    wr64_require(cudaMemcpy(envs.data(), device->envs,
        envs.size() * sizeof(Env), cudaMemcpyDeviceToHost),
        "diagnose paired environments");
    wr64_require(cudaMemcpy(observations.data(), device->observations,
        observations.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "diagnose paired observations");
    wr64_require(cudaMemcpy(rewards.data(), device->rewards,
        rewards.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "diagnose paired rewards");
    wr64_require(cudaMemcpy(terminals.data(), device->terminals,
        terminals.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "diagnose paired terminals");
    for (int a = 0; a + 1 < device->n; a += 2) {
        int b = a + 1;
        const uint8_t* env_a = (const uint8_t*)&envs[a];
        const uint8_t* env_b = (const uint8_t*)&envs[b];
        size_t env_byte = 0;
        while (env_byte < sizeof(Env)
                && env_a[env_byte] == env_b[env_byte]) env_byte++;
        int obs = 0;
        while (obs < WR64_CUDA_OBS_SIZE) {
            uint32_t bits_a;
            uint32_t bits_b;
            std::memcpy(&bits_a,
                &observations[(size_t)a * WR64_CUDA_OBS_SIZE + obs], 4);
            std::memcpy(&bits_b,
                &observations[(size_t)b * WR64_CUDA_OBS_SIZE + obs], 4);
            if (bits_a != bits_b) break;
            obs++;
        }
        uint32_t reward_a;
        uint32_t reward_b;
        uint32_t terminal_a;
        uint32_t terminal_b;
        std::memcpy(&reward_a, &rewards[a], 4);
        std::memcpy(&reward_b, &rewards[b], 4);
        std::memcpy(&terminal_a, &terminals[a], 4);
        std::memcpy(&terminal_b, &terminals[b], 4);
        if (env_byte == sizeof(Env) && obs == WR64_CUDA_OBS_SIZE
                && reward_a == reward_b && terminal_a == terminal_b) {
            continue;
        }
        std::vector<uint8_t> rdram_a(WR64_CUDA_RDRAM_SIZE);
        std::vector<uint8_t> rdram_b(WR64_CUDA_RDRAM_SIZE);
        wr64_require(cudaMemcpy(rdram_a.data(),
            device->rdram + (size_t)a * device->rdram_stride,
            WR64_CUDA_RDRAM_SIZE, cudaMemcpyDeviceToHost),
            "diagnose first paired RDRAM");
        wr64_require(cudaMemcpy(rdram_b.data(),
            device->rdram + (size_t)b * device->rdram_stride,
            WR64_CUDA_RDRAM_SIZE, cudaMemcpyDeviceToHost),
            "diagnose second paired RDRAM");
        size_t rdram_byte = 0;
        while (rdram_byte < WR64_CUDA_RDRAM_SIZE
                && rdram_a[rdram_byte] == rdram_b[rdram_byte]) rdram_byte++;
        std::fprintf(stderr,
            "PAIR_DIAG envs=%d,%d env_offset=%s%zu obs=%d "
            "reward=[%08x,%08x] terminal=[%08x,%08x] "
            "rdram_offset=%s%zu ticks=[%d,%d] episodes=[%u,%u] "
            "variants=[%u,%u] wave_episodes=[%u,%u]\n",
            a, b, env_byte == sizeof(Env) ? "none:" : "", env_byte,
            obs, reward_a, reward_b, terminal_a, terminal_b,
            rdram_byte == WR64_CUDA_RDRAM_SIZE ? "none:" : "",
            rdram_byte, envs[a].state.tick, envs[b].state.tick,
            envs[a].episode_id, envs[b].episode_id,
            envs[a].active_wave_variant, envs[b].active_wave_variant,
            envs[a].wave_episode, envs[b].wave_episode);
        return;
    }
}

static WR64StressStats wr64_run_pass(WR64StressDevice* device,
        const WR64StressPool& pool, const std::vector<Env>& initial_envs,
        int decisions, int pass, int perturb_env,
        std::vector<WR64StressHash>* hashes_out) {
    g_pass = pass;
    g_decision = -1;
    wr64_require(cudaMemcpy(device->envs, initial_envs.data(),
        initial_envs.size() * sizeof(Env), cudaMemcpyHostToDevice),
        "upload initial environments");
    for (int index = 0; index < device->n; index++) {
        wr64_require(cudaMemset(
            device->rdram + (size_t)index * device->rdram_stride,
            WR64_STRESS_CANARY, device->mapped_bytes),
            "initialize guarded RDRAM mapping");
    }
    wr64_require(cudaMemset(device->observations, 0,
        (size_t)device->n * WR64_CUDA_OBS_SIZE * sizeof(float)),
        "clear observations");
    wr64_require(cudaMemset(device->rewards, 0,
        (size_t)device->n * sizeof(float)), "clear rewards");
    wr64_require(cudaMemset(device->terminals, 0,
        (size_t)device->n * sizeof(float)), "clear terminals");
    wr64_require(cudaMemset(device->terminal, 0,
        sizeof(WR64CudaTerminal)), "clear terminal metadata");
    wr64_require(cudaMemset(device->stats, 0, sizeof(WR64StressStats)),
        "clear stress stats");
    wr64_require(cudaMemset(device->variant_seen, 0,
        (size_t)device->variants * sizeof(uint32_t)),
        "clear variant coverage");
    wr64_require(cudaMemset(device->last_episode, 0,
        (size_t)device->n * sizeof(uint32_t)), "clear episode audit");
    wr64_require(cudaMemset(device->last_recovery, 0,
        (size_t)device->n * sizeof(int32_t)), "clear recovery audit");

    wr64_launch_reset(device, 1, 1);
    int threads = 128;
    int blocks = (device->n + threads - 1) / threads;
    wr64_stress_post_reset<<<blocks, threads>>>(device->envs, device->n,
        device->variants, device->last_episode, device->last_recovery,
        device->variant_seen, device->stats);
    wr64_require(cudaDeviceSynchronize(), "initial production reset");
    if (pass == 0) wr64_verify_initial_snapshots(device, pool);

    for (int decision = 0; decision < decisions; decision++) {
        g_decision = decision;
        wr64_stress_actions<<<blocks, threads>>>(device->actions,
            device->observations, device->n,
            (uint32_t)decision, perturb_env);
        wr64_stress_pre_step<<<blocks, threads>>>(
            device->envs, device->n, device->stats);
        int step_threads = 32;
        int step_blocks = (device->n + step_threads - 1) / step_threads;
        wr64_cuda_step_kernel<<<step_blocks, step_threads>>>(
            device->envs, device->n, device->rdram, device->rdram_stride,
            device->actions, device->observations,
            device->rewards, device->terminals);
        wr64_require(cudaGetLastError(), "launch production step kernel");
        wr64_stress_post_step<<<blocks, threads>>>(device->envs, device->n,
            device->observations, device->rewards, device->terminals,
            device->last_recovery, device->stats);
        if (perturb_env < 0) {
            int pairs = device->n / 2;
            int pair_blocks = (pairs + threads - 1) / threads;
            wr64_stress_pair_compare<<<pair_blocks, threads>>>(
                device->envs, device->n, device->observations,
                device->rewards, device->terminals, device->stats);
        }
        wr64_launch_reset(device, 0, 0);
        wr64_stress_post_reset<<<blocks, threads>>>(device->envs, device->n,
            device->variants, device->last_episode,
            device->last_recovery, device->variant_seen, device->stats);
        wr64_require(cudaGetLastError(), "launch stress audit kernels");
        if ((decision & 15) == 15) {
            wr64_require(cudaDeviceSynchronize(), "stress batch");
        }
        if ((decision + 1) % 512 == 0) {
            std::printf("PROGRESS pass=%d decisions=%d/%d\n",
                pass, decision + 1, decisions);
            std::fflush(stdout);
        }
    }
    wr64_require(cudaDeviceSynchronize(), "complete stress trajectory");

    wr64_stress_guard_check<<<256, 256>>>(device->rdram, device->n,
        device->rdram_stride, device->stats);
    wr64_stress_hash_kernel<<<device->n, 256>>>(device->envs, device->n,
        device->rdram, device->rdram_stride, device->observations,
        device->rewards, device->terminals, device->hashes);
    wr64_require(cudaDeviceSynchronize(), "guard and state hash audit");

    WR64StressStats stats = {};
    std::vector<uint32_t> seen(device->variants);
    hashes_out->resize(device->n);
    wr64_require(cudaMemcpy(&stats, device->stats, sizeof(stats),
        cudaMemcpyDeviceToHost), "download stress stats");
    wr64_require(cudaMemcpy(seen.data(), device->variant_seen,
        seen.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost),
        "download variant coverage");
    wr64_require(cudaMemcpy(hashes_out->data(), device->hashes,
        hashes_out->size() * sizeof(WR64StressHash), cudaMemcpyDeviceToHost),
        "download state hashes");
    int covered = wr64_count_variants(seen);
    if (stats.pair_mismatches != 0 && perturb_env < 0) {
        wr64_diagnose_pairs(device);
    }
    std::printf(
        "PASS%d env_steps=%llu native_updates=%llu terminals=%llu "
        "autoresets=%llu success=%llu failures=%llu ordinary=%llu "
        "dq=%llu timeout=%llu recovery_samples=%llu recovery_entries=%llu "
        "variants=%d/%u\n",
        pass, stats.env_steps, stats.native_updates, stats.terminals,
        stats.autoresets, stats.successes, stats.failures,
        stats.ordinary_failures, stats.disqualifications,
        stats.safety_timeouts, stats.recovery_samples,
        stats.recovery_entries, covered, device->variants);
    if (covered != (int)device->variants) {
        std::fprintf(stderr, "FAIL pass %d covered %d/%u wave variants\n",
            pass, covered, device->variants);
        std::exit(1);
    }
    return stats;
}

static bool wr64_hash_equal(
        const WR64StressHash& a, const WR64StressHash& b) {
    return a.xor_hash == b.xor_hash && a.sum_hash == b.sum_hash;
}

static int wr64_validate_stats(const WR64StressStats& stats,
        int n, int decisions, int pass) {
    int failures = 0;
    unsigned long long expected_steps =
        (unsigned long long)n * (unsigned long long)decisions;
    if (stats.env_steps != expected_steps) {
        std::fprintf(stderr,
            "FAIL pass %d env steps expected=%llu actual=%llu\n",
            pass, expected_steps, stats.env_steps);
        failures++;
    }
    if (stats.native_updates >= stats.env_steps * WR64_STRESS_FRAMESKIP
            || stats.native_updates + stats.boundary_skips
                != stats.env_steps * WR64_STRESS_FRAMESKIP) {
        std::fprintf(stderr,
            "FAIL pass %d native update accounting steps=%llu "
            "updates=%llu boundary_skips=%llu\n",
            pass, stats.env_steps, stats.native_updates,
            stats.boundary_skips);
        failures++;
    }
    if (stats.terminals == 0 || stats.autoresets != stats.terminals
            || stats.failures == 0 || stats.disqualifications == 0
            || stats.recovery_entries == 0) {
        std::fprintf(stderr,
            "FAIL pass %d required dynamics terminals=%llu resets=%llu "
            "failures=%llu dq=%llu recovery=%llu\n",
            pass, stats.terminals, stats.autoresets, stats.failures,
            stats.disqualifications, stats.recovery_entries);
        failures++;
    }
    unsigned long long faults = stats.machine_errors
        + stats.indirect_faults + stats.nan_observations
        + stats.nan_rewards + stats.nan_terminals + stats.nan_state
        + stats.invalid_terminals + stats.env_faults
        + stats.pair_mismatches + stats.guard_faults;
    if (faults != 0) {
        std::fprintf(stderr,
            "FAIL pass %d faults machine=%llu indirect=%llu nan_obs=%llu "
            "nan_reward=%llu nan_terminal=%llu nan_state=%llu "
            "bad_terminal=%llu env_fault=%llu pair=%llu "
            "pair_env=%llu pair_output=%llu guard=%llu\n",
            pass, stats.machine_errors, stats.indirect_faults,
            stats.nan_observations, stats.nan_rewards,
            stats.nan_terminals, stats.nan_state,
            stats.invalid_terminals, stats.env_faults,
            stats.pair_mismatches, stats.pair_env_mismatches,
            stats.pair_output_mismatches, stats.guard_faults);
        failures++;
    }
    return failures;
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) {
        std::fprintf(stderr,
            "usage: %s /path/to/baserom.us.rev1.z64 "
            "[decisions] [envs] [variants]\n", argv[0]);
        return 2;
    }
    int decisions = argc > 2
        ? std::atoi(argv[2]) : WR64_STRESS_DEFAULT_DECISIONS;
    int n = argc > 3 ? std::atoi(argv[3]) : WR64_STRESS_DEFAULT_ENVS;
    uint32_t variants = argc > 4
        ? (uint32_t)std::strtoul(argv[4], nullptr, 10)
        : WR64_STRESS_DEFAULT_VARIANTS;
    if (decisions < 256 || n < 2 || (n & 1)
            || variants < 2 || variants > WR64_CUDA_MAX_WAVE_VARIANTS
            || (variants & (variants - 1u)) != 0u) {
        std::fprintf(stderr,
            "decisions>=256, positive even envs, and power-of-two "
            "variants in [2,128] are required\n");
        return 2;
    }
    wr64_require(cudaDeviceSetLimit(cudaLimitStackSize, 8u * 1024u),
        "set production transition stack limit");
    size_t actual_stack = 0;
    wr64_require(cudaDeviceGetLimit(&actual_stack, cudaLimitStackSize),
        "read production transition stack limit");

    char error[256] = {};
    WR64CudaHost* host = nullptr;
    int status = wr64_cuda_host_create(
        argv[1], &host, error, sizeof(error));
    if (status != WR64_CUDA_HOST_OK) {
        std::fprintf(stderr, "host bootstrap create failed (%d): %s\n",
            status, error);
        return 2;
    }
    WR64StressPool pool = wr64_build_pool(host, variants);
    wr64_cuda_host_destroy(host);
    std::printf(
        "SETUP envs=%d decisions=%d passes=3 variants=%u "
        "delta_pages=%u payload=%zu stack=%zu diagnostic_bounds=0 "
        "vmm_stride=%zu guard=%u\n",
        n, decisions, variants, pool.total_pages, pool.data.size(),
        actual_stack, (size_t)WR64_STRESS_RDRAM_STRIDE,
        WR64_STRESS_GUARD_BYTES);

    std::vector<Env> initial_envs(n);
    std::memset(initial_envs.data(), 0,
        initial_envs.size() * sizeof(Env));
    for (int index = 0; index < n; index++) {
        initial_envs[index] = wr64_make_env(
            pool.infos[0], (uint32_t)index >> 1, variants);
    }
    WR64StressDevice device = {};
    wr64_alloc_device(&device, n, pool);
    const char* fault_probe = std::getenv("WR64_STRESS_VMM_FAULT_PROBE");
    if (fault_probe && std::strcmp(fault_probe, "1") == 0) {
        wr64_stress_vmm_fault_probe<<<1, 1>>>(
            device.rdram, device.mapped_bytes);
        wr64_require(cudaGetLastError(), "launch VMM release-hole probe");
        cudaError_t probe_status = cudaDeviceSynchronize();
        if (probe_status == cudaSuccess) {
            std::fprintf(stderr,
                "FAIL VMM release-hole probe write unexpectedly succeeded\n");
            return 1;
        }
        std::printf(
            "PASS VMM release-hole probe trapped at offset=%zu error=%s\n",
            device.mapped_bytes, cudaGetErrorString(probe_status));
        std::fflush(stdout);
        std::_Exit(0);
    }

    std::vector<WR64StressHash> hash_a;
    std::vector<WR64StressHash> hash_b;
    std::vector<WR64StressHash> hash_isolation;
    WR64StressStats stats_a = wr64_run_pass(
        &device, pool, initial_envs, decisions, 0, -1, &hash_a);
    WR64StressStats stats_b = wr64_run_pass(
        &device, pool, initial_envs, decisions, 1, -1, &hash_b);
    int failures = 0;
    failures += wr64_validate_stats(stats_a, n, decisions, 0);
    failures += wr64_validate_stats(stats_b, n, decisions, 1);
    for (int index = 0; index < n; index++) {
        if (!wr64_hash_equal(hash_a[index], hash_b[index])) {
            std::fprintf(stderr,
                "FAIL deterministic replay env=%d "
                "A=[%016llx,%016llx] B=[%016llx,%016llx]\n",
                index, hash_a[index].xor_hash, hash_a[index].sum_hash,
                hash_b[index].xor_hash, hash_b[index].sum_hash);
            failures++;
        }
        if ((index & 1) == 0
                && !wr64_hash_equal(hash_a[index], hash_a[index + 1])) {
            std::fprintf(stderr,
                "FAIL paired environment mismatch envs=%d,%d\n",
                index, index + 1);
            failures++;
        }
    }
    if (failures == 0) {
        std::puts("PASS deterministic replay and paired-environment hashes");
    }

    WR64StressStats stats_c = wr64_run_pass(
        &device, pool, initial_envs, decisions, 2, 0, &hash_isolation);
    failures += wr64_validate_stats(stats_c, n, decisions, 2);
    if (wr64_hash_equal(hash_a[0], hash_isolation[0])) {
        std::fprintf(stderr,
            "FAIL isolated perturbation did not change target env 0\n");
        failures++;
    }
    for (int index = 1; index < n; index++) {
        if (!wr64_hash_equal(hash_a[index], hash_isolation[index])) {
            std::fprintf(stderr,
                "FAIL cross-env contamination target=0 changed=%d\n", index);
            failures++;
        }
    }
    if (failures == 0) {
        std::puts("PASS one-environment perturbation isolation");
    }

    std::printf(
        "TOTAL env_steps=%llu native_updates=%llu autoresets=%llu "
        "terminals=%llu variants=%u traps=0 indirect_faults=0 "
        "nans=0 cross_env_faults=0\n",
        stats_a.env_steps + stats_b.env_steps + stats_c.env_steps,
        stats_a.native_updates + stats_b.native_updates
            + stats_c.native_updates,
        stats_a.autoresets + stats_b.autoresets + stats_c.autoresets,
        stats_a.terminals + stats_b.terminals + stats_c.terminals,
        variants);
    wr64_free_device(&device);
    if (failures == 0) {
        std::puts("PASS production-kernel Wave Race CUDA stress gate");
    }
    return failures ? 1 : 0;
}
