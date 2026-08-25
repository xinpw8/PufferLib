#ifndef PUFFER_WAVERACE64_GPU_CU
#define PUFFER_WAVERACE64_GPU_CU

#define PUF_BACKEND PUF_GPU

#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef float obs_t;

#include "waverace64_cuda_adapter.cuh"
#include "pufferenv.h"
#include "waverace64_cuda_host.h"
#include "waverace64_cuda_types.cuh"

#define OBS_SIZE WR64_CUDA_OBS_SIZE
#define NUM_ATNS WR64_CUDA_NUM_ATNS
#define ACT_SIZES {15, 9, 2, 2, 2}
#define PUF_STEPS_PER_SEC 10
#define PUF_STEPS_PER_SEC_ENV(env) wr64_cuda_steps_per_second(env)

#define PUFFER_ENV_DISCOUNT_FROM_TRAIN
#define PUFFER_ENV_INTERNAL_FRAMESKIP
#define PUFFER_ENV_UNCLIPPED_REWARDS
#define PUFFER_ENV_EVAL_RESET
#define PUFFER_ENV_EXACT_OUTCOMES
#define PUF_EVAL_RENDER_PAUSED(env) wr64_cuda_eval_paused(env)
#define PUF_EVAL_RENDER_TERMINAL(env) wr64_cuda_eval_terminal(env)
#define PUF_EVAL_POLICY_DURING_HUMAN 1

#define WR64_CUDA_ERROR_BUFFER 256
#define WR64_CUDA_MAX_RESET_THREADS 1024
#define WR64_CUDA_RDRAM_STRIDE (UINT64_C(1) << 32)
static_assert(sizeof(WR64CudaState) == sizeof(WR64CudaHostAdapterState),
    "Wave Race CUDA host/device adapter state changed");

static struct {
    Env* envs;
    int n;
    obs_t* observations;
    float* actions;
    float* rewards;
    float* terminals;
    uint8_t* rdram;
    size_t rdram_stride;
    CUdeviceptr rdram_reservation;
    size_t rdram_reservation_size;
    CUmemGenericAllocationHandle* rdram_handles;
    CUcontext rdram_context;
    uint8_t* reset_canonical;
    uint32_t* variant_offsets;
    uint16_t* variant_pages;
    uint8_t* variant_data;
    WR64DeviceMachine* variant_machines;
    uint32_t variant_count;
    uint32_t variant_page_count;
    cudaStream_t stream;
    WR64CudaHost* host;
    uint8_t* render_rdram;
    uint8_t* terminal_rdram;
    WR64CudaTerminal* terminal;
    uint32_t render_episode_id;
    int render_episode_valid;
    int terminal_presented;
    int render_paused;
    int render_terminal;
    int frameskip;
    int step_block_size;
    int reset_block_size;
} g_wr64_cuda;

static int wr64_cuda_steps_per_second(Env*) {
    int frameskip = g_wr64_cuda.frameskip > 0 ? g_wr64_cuda.frameskip : 2;
    return (int)WR64_CUDA_GAME_UPDATE_HZ / frameskip;
}

static int wr64_cuda_eval_paused(Env*) {
    return g_wr64_cuda.render_paused;
}

static int wr64_cuda_eval_terminal(Env*) {
    return g_wr64_cuda.render_terminal;
}

static void wr64_cuda_fail(const char* operation, cudaError_t status) {
    if (status == cudaSuccess) return;
    fprintf(stderr, "[waverace64 cuda] %s: %s\n",
        operation, cudaGetErrorString(status));
    abort();
}

static void wr64_cuda_driver_fail(const char* operation, CUresult status) {
    if (status == CUDA_SUCCESS) return;
    const char* name = NULL;
    const char* detail = NULL;
    (void)cuGetErrorName(status, &name);
    (void)cuGetErrorString(status, &detail);
    fprintf(stderr, "[waverace64 cuda] %s: %s (%s)\n", operation,
        detail ? detail : "unknown CUDA driver error",
        name ? name : "unknown");
    abort();
}

static void wr64_cuda_allocate_rdram(int n) {
    static_assert(WR64_CUDA_RDRAM_STRIDE > WR64_CUDA_RDRAM_SIZE,
        "Wave Race CUDA RDRAM guard stride must exceed mapped RDRAM");
    if ((uint64_t)n > (uint64_t)SIZE_MAX / WR64_CUDA_RDRAM_STRIDE) {
        fprintf(stderr,
            "[waverace64 cuda] RDRAM virtual reservation size overflow\n");
        abort();
    }

    // Initialize the runtime primary context before mixing runtime and driver
    // APIs, then use that same current context for sparse VMM mappings.
    wr64_cuda_fail("initialize CUDA context", cudaFree(NULL));
    wr64_cuda_driver_fail("initialize CUDA driver", cuInit(0));
    CUcontext context = NULL;
    wr64_cuda_driver_fail("read current CUDA context", cuCtxGetCurrent(&context));
    if (context == NULL) {
        fprintf(stderr, "[waverace64 cuda] CUDA context is not current\n");
        abort();
    }

    int runtime_device = 0;
    wr64_cuda_fail("read CUDA device", cudaGetDevice(&runtime_device));
    CUdevice device;
    wr64_cuda_driver_fail("resolve CUDA device", cuDeviceGet(&device, runtime_device));
    int vmm_supported = 0;
    wr64_cuda_driver_fail("query CUDA VMM support", cuDeviceGetAttribute(
        &vmm_supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
        device));
    if (!vmm_supported) {
        fprintf(stderr,
            "[waverace64 cuda] CUDA virtual memory management is required "
            "for isolated guest RDRAM\n");
        abort();
    }

    CUmemAllocationProp properties = {};
    properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    properties.location.id = (int)device;
    properties.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    size_t granularity = 0;
    wr64_cuda_driver_fail("query CUDA VMM granularity",
        cuMemGetAllocationGranularity(&granularity, &properties,
            CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    if (granularity == 0 || WR64_CUDA_RDRAM_SIZE % granularity != 0) {
        fprintf(stderr,
            "[waverace64 cuda] 8 MiB RDRAM is incompatible with CUDA VMM "
            "granularity %zu\n", granularity);
        abort();
    }

    size_t reservation_size = (size_t)n * (size_t)WR64_CUDA_RDRAM_STRIDE;
    CUdeviceptr reservation = 0;
    wr64_cuda_driver_fail("reserve isolated RDRAM address space",
        cuMemAddressReserve(&reservation, reservation_size, 0, 0, 0));
    CUmemGenericAllocationHandle* handles =
        (CUmemGenericAllocationHandle*)calloc((size_t)n, sizeof(*handles));
    if (handles == NULL) {
        fprintf(stderr,
            "[waverace64 cuda] RDRAM allocation-handle table failed\n");
        abort();
    }

    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = (int)device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    for (int index = 0; index < n; index++) {
        CUdeviceptr address = reservation
            + (CUdeviceptr)(size_t)index * WR64_CUDA_RDRAM_STRIDE;
        wr64_cuda_driver_fail("allocate isolated RDRAM physical memory",
            cuMemCreate(&handles[index], WR64_CUDA_RDRAM_SIZE,
                &properties, 0));
        wr64_cuda_driver_fail("map isolated RDRAM",
            cuMemMap(address, WR64_CUDA_RDRAM_SIZE, 0,
                handles[index], 0));
        wr64_cuda_driver_fail("enable isolated RDRAM access",
            cuMemSetAccess(address, WR64_CUDA_RDRAM_SIZE, &access, 1));
    }

    g_wr64_cuda.rdram = (uint8_t*)(uintptr_t)reservation;
    g_wr64_cuda.rdram_stride = (size_t)WR64_CUDA_RDRAM_STRIDE;
    g_wr64_cuda.rdram_reservation = reservation;
    g_wr64_cuda.rdram_reservation_size = reservation_size;
    g_wr64_cuda.rdram_handles = handles;
    g_wr64_cuda.rdram_context = context;
}

static void wr64_cuda_free_rdram(void) {
    if (g_wr64_cuda.rdram_reservation != 0) {
        for (int index = 0; index < g_wr64_cuda.n; index++) {
            CUdeviceptr address = g_wr64_cuda.rdram_reservation
                + (CUdeviceptr)(size_t)index * g_wr64_cuda.rdram_stride;
            wr64_cuda_driver_fail("unmap isolated RDRAM",
                cuMemUnmap(address, WR64_CUDA_RDRAM_SIZE));
        }
    }
    if (g_wr64_cuda.rdram_handles != NULL) {
        for (int index = 0; index < g_wr64_cuda.n; index++) {
            if (g_wr64_cuda.rdram_handles[index] != 0) {
                wr64_cuda_driver_fail("release isolated RDRAM",
                    cuMemRelease(g_wr64_cuda.rdram_handles[index]));
            }
        }
        free(g_wr64_cuda.rdram_handles);
    }
    if (g_wr64_cuda.rdram_reservation != 0) {
        wr64_cuda_driver_fail("release isolated RDRAM address space",
            cuMemAddressFree(g_wr64_cuda.rdram_reservation,
                g_wr64_cuda.rdram_reservation_size));
    }
    g_wr64_cuda.rdram = NULL;
    g_wr64_cuda.rdram_stride = 0;
    g_wr64_cuda.rdram_reservation = 0;
    g_wr64_cuda.rdram_reservation_size = 0;
    g_wr64_cuda.rdram_handles = NULL;
}

static void wr64_cuda_host_fail(const char* operation,
        int status, const char* detail) {
    if (status == WR64_CUDA_HOST_OK) return;
    fprintf(stderr, "[waverace64 cuda] %s failed (%d): %s\n",
        operation, status, detail ? detail : "unknown error");
    abort();
}

static int wr64_cuda_dict_int(Dict* kwargs, const char* key) {
    return (int)dict_get(kwargs, key);
}

static int wr64_cuda_dict_bool(Dict* kwargs, const char* key) {
    double value = dict_get(kwargs, key);
    if (value != 0.0 && value != 1.0) {
        fprintf(stderr, "[waverace64 cuda] %s must be 0 or 1\n", key);
        abort();
    }
    return (int)value;
}

static unsigned long wr64_cuda_env_ulong(
        const char* key, unsigned long fallback,
        unsigned long minimum, unsigned long maximum) {
    const char* text = getenv(key);
    if (!text || !text[0]) return fallback;
    char* end = NULL;
    unsigned long value = strtoul(text, &end, 10);
    if (!end || *end || value < minimum || value > maximum) {
        fprintf(stderr, "[waverace64 cuda] %s must be in [%lu,%lu]\n",
            key, minimum, maximum);
        abort();
    }
    return value;
}

static uint32_t wr64_cuda_dict_u32(Dict* kwargs, const char* key) {
    double value = dict_get(kwargs, key);
    if (!(value >= 0.0 && value <= 4294967295.0)
            || value != (double)(uint32_t)value) {
        fprintf(stderr, "[waverace64 cuda] %s must be a uint32 integer\n", key);
        abort();
    }
    return (uint32_t)value;
}

static void wr64_cuda_fill_adapter(
        Env* env, Dict* kwargs, unsigned int index, uint32_t variants,
        const WR64CudaHostSnapshotInfo* canonical) {
    memset(env, 0, sizeof(*env));
    env->frameskip = wr64_cuda_dict_int(kwargs, "frameskip");
    if (env->frameskip < 1) env->frameskip = 2;
    env->rng = index;
    env->randomize_waves = wr64_cuda_dict_bool(kwargs, "randomize_waves");
    env->wave_seed = wr64_cuda_dict_u32(kwargs, "wave_seed");
    env->wave_variants = (int32_t)variants;
    env->wave_rng_state = wr64_cuda_wave_stream_seed(env->wave_seed, index);
    env->wave_episode = 0;
    env->wave_boot_variant = index & (variants - 1u);
    env->active_wave_variant = env->wave_boot_variant;

    env->reward_speed = (float)dict_get(kwargs, "reward_speed");
    env->reward_progress = (float)dict_get(kwargs, "reward_progress");
    env->reward_slip = (float)dict_get(kwargs, "reward_slip");
    env->reward_checkpoint = (float)dict_get(kwargs, "reward_checkpoint");
    env->reward_miss = (float)dict_get(kwargs, "reward_miss");
    env->reward_finish = (float)dict_get(kwargs, "reward_finish");
    env->reward_fail = (float)dict_get(kwargs, "reward_fail");
    env->discount = (float)dict_get(kwargs, "discount");
    env->reward_mode = wr64_cuda_dict_int(kwargs, "reward_mode");
    env->curriculum_start_laps = wr64_cuda_dict_int(
        kwargs, "curriculum_start_laps");
    env->curriculum_max_laps = wr64_cuda_dict_int(
        kwargs, "curriculum_max_laps");
    env->curriculum_successes_per_lap = wr64_cuda_dict_int(
        kwargs, "curriculum_successes_per_lap");
    env->curriculum_laps = env->curriculum_start_laps;

    if ((env->randomize_waves != 0 && env->randomize_waves != 1)
            || !(env->discount > 0.f && env->discount <= 1.f)
            || env->reward_mode < 0 || env->reward_mode > 2
            || env->curriculum_start_laps < 1
            || env->curriculum_max_laps > 3
            || env->curriculum_max_laps < env->curriculum_start_laps
            || env->curriculum_successes_per_lap < 1) {
        fprintf(stderr, "[waverace64 cuda] invalid environment configuration\n");
        abort();
    }

    memcpy(env->route_arc, canonical->route_arc, sizeof(env->route_arc));
    memcpy(env->route_pred, canonical->route_pred, sizeof(env->route_pred));
    env->route_total = canonical->route_total;
    env->route_nodes = canonical->route_nodes;
    env->route_valid = canonical->route_valid;
    env->vertical_origin = canonical->vertical_origin;
    env->num_agents = 1;
    env->agents[0].policy = 0;
    env->needs_reset = 1;
}

static WR64DeviceMachine wr64_cuda_machine_from_host(
        const WR64CudaHostMachineState* source) {
    WR64DeviceMachine machine = {};
    machine.ticks = source->ticks;
    machine.pad_buttons = source->pad_buttons;
    machine.pad_stick_x = source->pad_stick_x;
    machine.pad_stick_y = source->pad_stick_y;
    machine.resident_overlay = source->resident_overlay;
    machine.rounding_mode = source->rounding_mode;
    return machine;
}

static int wr64_cuda_route_compatible(
        const WR64CudaHostSnapshotInfo* a,
        const WR64CudaHostSnapshotInfo* b) {
    return a->route_total == b->route_total
        && a->route_nodes == b->route_nodes
        && a->route_valid == b->route_valid
        && memcmp(a->route_arc, b->route_arc, sizeof(a->route_arc)) == 0
        && memcmp(a->route_pred, b->route_pred, sizeof(a->route_pred)) == 0;
}

static uint64_t wr64_cuda_hash_bytes(const uint8_t* data, size_t size) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t index = 0; index < size; index++) {
        hash = (hash ^ data[index]) * UINT64_C(1099511628211);
    }
    return hash;
}

static void wr64_cuda_build_reset_pool(
        Dict* kwargs, uint32_t variant_count,
        WR64CudaHostSnapshotInfo** infos_out) {
    size_t image_bytes = (size_t)variant_count * WR64_CUDA_RDRAM_SIZE;
    uint8_t* images = (uint8_t*)malloc(image_bytes);
    WR64CudaHostSnapshotInfo* infos = (WR64CudaHostSnapshotInfo*)calloc(
        variant_count, sizeof(*infos));
    int* statuses = (int*)calloc(variant_count, sizeof(*statuses));
    char* errors = (char*)calloc(variant_count, WR64_CUDA_ERROR_BUFFER);
    if (!images || !infos || !statuses || !errors) {
        fprintf(stderr, "[waverace64 cuda] reset-pool host allocation failed\n");
        abort();
    }

    uint32_t wave_seed = wr64_cuda_dict_u32(kwargs, "wave_seed");
    int randomize_waves = wr64_cuda_dict_bool(kwargs, "randomize_waves");
    int threads = omp_get_max_threads();
    if (threads < 1) threads = 1;
    if (threads > (int)variant_count) threads = (int)variant_count;
    #pragma omp parallel for schedule(dynamic) num_threads(threads)
    for (int variant = 0; variant < (int)variant_count; variant++) {
        statuses[variant] = wr64_cuda_host_boot_snapshot(g_wr64_cuda.host,
            wave_seed, randomize_waves, (uint32_t)variant,
            images + (size_t)variant * WR64_CUDA_RDRAM_SIZE,
            WR64_CUDA_RDRAM_SIZE, &infos[variant],
            errors + (size_t)variant * WR64_CUDA_ERROR_BUFFER,
            WR64_CUDA_ERROR_BUFFER);
    }
    for (uint32_t variant = 0; variant < variant_count; variant++) {
        wr64_cuda_host_fail("authentic snapshot boot", statuses[variant],
            errors + (size_t)variant * WR64_CUDA_ERROR_BUFFER);
        if (infos[variant].abi_version != WR64_CUDA_HOST_ABI_VERSION
                || infos[variant].struct_size != sizeof(infos[variant])
                || !wr64_cuda_route_compatible(&infos[0], &infos[variant])) {
            fprintf(stderr,
                "[waverace64 cuda] incompatible reset variant %u\n", variant);
            abort();
        }
    }

    const size_t water_offset =
        WR64_CUDA_WATER_GRID & UINT32_C(0x1FFFFFFF);
    const size_t water_size =
        (size_t)WR64_CUDA_WATER_ROWS * WR64_CUDA_WATER_COLS * 4u;
    if (water_offset > WR64_CUDA_RDRAM_SIZE
            || water_size > WR64_CUDA_RDRAM_SIZE - water_offset) {
        fprintf(stderr, "[waverace64 cuda] water field is outside RDRAM\n");
        abort();
    }
    uint64_t rdram_hashes[WR64_CUDA_MAX_WAVE_VARIANTS] = {};
    uint64_t water_hashes[WR64_CUDA_MAX_WAVE_VARIANTS] = {};
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int variant = 0; variant < (int)variant_count; variant++) {
        const uint8_t* image = images
            + (size_t)variant * WR64_CUDA_RDRAM_SIZE;
        rdram_hashes[variant] = wr64_cuda_hash_bytes(
            image, WR64_CUDA_RDRAM_SIZE);
        water_hashes[variant] = wr64_cuda_hash_bytes(
            image + water_offset, water_size);
    }
    for (uint32_t variant = 0; variant < variant_count; variant++) {
        const uint8_t* image = images
            + (size_t)variant * WR64_CUDA_RDRAM_SIZE;
        for (uint32_t previous = 0; previous < variant; previous++) {
            const uint8_t* previous_image = images
                + (size_t)previous * WR64_CUDA_RDRAM_SIZE;
            if (rdram_hashes[previous] == rdram_hashes[variant]
                    && memcmp(previous_image, image,
                        WR64_CUDA_RDRAM_SIZE) == 0) {
                fprintf(stderr,
                    "[waverace64 cuda] reset variants %u and %u "
                    "are identical\n", previous, variant);
                abort();
            }
            if (water_hashes[previous] == water_hashes[variant]
                    && memcmp(previous_image + water_offset,
                        image + water_offset, water_size) == 0) {
                fprintf(stderr,
                    "[waverace64 cuda] reset variants %u and %u have "
                    "identical water fields\n", previous, variant);
                abort();
            }
        }
    }

    uint32_t* offsets = (uint32_t*)calloc(
        (size_t)variant_count + 1, sizeof(*offsets));
    if (!offsets) abort();
    uint32_t total_pages = 0;
    for (uint32_t variant = 0; variant < variant_count; variant++) {
        offsets[variant] = total_pages;
        const uint8_t* image = images
            + (size_t)variant * WR64_CUDA_RDRAM_SIZE;
        for (uint32_t page = 0; page < WR64_CUDA_PAGES; page++) {
            size_t byte = (size_t)page * WR64_CUDA_PAGE_SIZE;
            if (memcmp(images + byte, image + byte, WR64_CUDA_PAGE_SIZE) != 0) {
                total_pages++;
            }
        }
    }
    offsets[variant_count] = total_pages;

    uint16_t* pages = total_pages
        ? (uint16_t*)malloc((size_t)total_pages * sizeof(*pages)) : NULL;
    uint8_t* data = total_pages
        ? (uint8_t*)malloc((size_t)total_pages * WR64_CUDA_PAGE_SIZE) : NULL;
    if (total_pages && (!pages || !data)) abort();
    uint32_t cursor = 0;
    for (uint32_t variant = 0; variant < variant_count; variant++) {
        const uint8_t* image = images
            + (size_t)variant * WR64_CUDA_RDRAM_SIZE;
        for (uint32_t page = 0; page < WR64_CUDA_PAGES; page++) {
            size_t byte = (size_t)page * WR64_CUDA_PAGE_SIZE;
            if (memcmp(images + byte, image + byte, WR64_CUDA_PAGE_SIZE) == 0) {
                continue;
            }
            pages[cursor] = (uint16_t)page;
            memcpy(data + (size_t)cursor * WR64_CUDA_PAGE_SIZE,
                image + byte, WR64_CUDA_PAGE_SIZE);
            cursor++;
        }
    }
    if (cursor != total_pages) {
        fprintf(stderr, "[waverace64 cuda] inconsistent reset delta count\n");
        abort();
    }
    for (uint32_t variant = 0; variant < variant_count; variant++) {
        const uint8_t* image = images
            + (size_t)variant * WR64_CUDA_RDRAM_SIZE;
        uint32_t packed = offsets[variant];
        uint32_t packed_end = offsets[variant + 1u];
        for (uint32_t page = 0; page < WR64_CUDA_PAGES; page++) {
            const uint8_t* reconstructed = images
                + (size_t)page * WR64_CUDA_PAGE_SIZE;
            if (packed < packed_end && pages[packed] == page) {
                reconstructed = data
                    + (size_t)packed * WR64_CUDA_PAGE_SIZE;
                packed++;
            }
            const uint8_t* expected = image
                + (size_t)page * WR64_CUDA_PAGE_SIZE;
            if (memcmp(reconstructed, expected,
                    WR64_CUDA_PAGE_SIZE) != 0) {
                fprintf(stderr,
                    "[waverace64 cuda] reset variant %u failed "
                    "reconstruction at page %u\n", variant, page);
                abort();
            }
        }
        if (packed != packed_end) {
            fprintf(stderr,
                "[waverace64 cuda] reset variant %u has unapplied pages\n",
                variant);
            abort();
        }
    }

    WR64DeviceMachine* machines = (WR64DeviceMachine*)calloc(
        variant_count, sizeof(*machines));
    if (!machines) abort();
    for (uint32_t variant = 0; variant < variant_count; variant++) {
        machines[variant] = wr64_cuda_machine_from_host(&infos[variant].machine);
    }

    wr64_cuda_fail("allocate canonical reset",
        cudaMalloc((void**)&g_wr64_cuda.reset_canonical,
            WR64_CUDA_RDRAM_SIZE));
    wr64_cuda_fail("upload canonical reset",
        cudaMemcpy(g_wr64_cuda.reset_canonical, images,
            WR64_CUDA_RDRAM_SIZE, cudaMemcpyHostToDevice));
    wr64_cuda_fail("allocate variant offsets",
        cudaMalloc((void**)&g_wr64_cuda.variant_offsets,
            ((size_t)variant_count + 1) * sizeof(*offsets)));
    wr64_cuda_fail("upload variant offsets",
        cudaMemcpy(g_wr64_cuda.variant_offsets, offsets,
            ((size_t)variant_count + 1) * sizeof(*offsets),
            cudaMemcpyHostToDevice));
    if (total_pages) {
        wr64_cuda_fail("allocate variant page indices",
            cudaMalloc((void**)&g_wr64_cuda.variant_pages,
                (size_t)total_pages * sizeof(*pages)));
        wr64_cuda_fail("upload variant page indices",
            cudaMemcpy(g_wr64_cuda.variant_pages, pages,
                (size_t)total_pages * sizeof(*pages), cudaMemcpyHostToDevice));
        wr64_cuda_fail("allocate variant page payload",
            cudaMalloc((void**)&g_wr64_cuda.variant_data,
                (size_t)total_pages * WR64_CUDA_PAGE_SIZE));
        wr64_cuda_fail("upload variant page payload",
            cudaMemcpy(g_wr64_cuda.variant_data, data,
                (size_t)total_pages * WR64_CUDA_PAGE_SIZE,
                cudaMemcpyHostToDevice));
    }
    wr64_cuda_fail("allocate variant machine states",
        cudaMalloc((void**)&g_wr64_cuda.variant_machines,
            (size_t)variant_count * sizeof(*machines)));
    wr64_cuda_fail("upload variant machine states",
        cudaMemcpy(g_wr64_cuda.variant_machines, machines,
            (size_t)variant_count * sizeof(*machines),
            cudaMemcpyHostToDevice));

    g_wr64_cuda.variant_count = variant_count;
    g_wr64_cuda.variant_page_count = total_pages;
    fprintf(stderr,
        "[waverace64 cuda] authentic wave variants=%u delta_pages=%u "
        "payload=%zu bytes\n",
        variant_count, total_pages,
        (size_t)total_pages * WR64_CUDA_PAGE_SIZE);

    free(machines);
    free(data);
    free(pages);
    free(offsets);
    free(errors);
    free(statuses);
    free(images);
    *infos_out = infos;
}

static void wr64_cuda_launch_reset(
        int force, int evaluation_mode, int clear_outputs) {
    int threads = force ? WR64_CUDA_RESET_THREADS
        : g_wr64_cuda.reset_block_size;
    wr64_cuda_reset_kernel<<<g_wr64_cuda.n, threads,
        0, g_wr64_cuda.stream>>>(
        g_wr64_cuda.envs, g_wr64_cuda.n, g_wr64_cuda.rdram,
        g_wr64_cuda.rdram_stride,
        g_wr64_cuda.reset_canonical, g_wr64_cuda.variant_offsets,
        g_wr64_cuda.variant_pages, g_wr64_cuda.variant_data,
        g_wr64_cuda.variant_machines,
        g_wr64_cuda.observations, g_wr64_cuda.rewards,
        g_wr64_cuda.terminals, force, evaluation_mode, clear_outputs,
        g_wr64_cuda.terminal_rdram, g_wr64_cuda.terminal);
    wr64_cuda_fail("launch reset kernel", cudaGetLastError());
}

Env* puf_vec_create(int n, Dict* env_kwargs,
        obs_t* observations, float* actions,
        float* rewards, float* terminals) {
    if (n < 1) {
        fprintf(stderr, "[waverace64 cuda] total_agents must be positive\n");
        abort();
    }
    const char* rom_path = dict_get_str(env_kwargs, "rom_path");
    if (!rom_path || !rom_path[0] || strcmp(rom_path, "None") == 0) {
        rom_path = "../baserom.us.rev1.z64";
    }
    char error[WR64_CUDA_ERROR_BUFFER] = {};
    int host_status = wr64_cuda_host_create(
        rom_path, &g_wr64_cuda.host, error, sizeof(error));
    wr64_cuda_host_fail("host bridge create", host_status, error);

    int randomize_waves = wr64_cuda_dict_bool(env_kwargs, "randomize_waves");
    uint32_t configured_variants = wr64_cuda_dict_u32(
        env_kwargs, "wave_variants");
    if (configured_variants < 1
            || configured_variants > WR64_CUDA_MAX_WAVE_VARIANTS
            || (configured_variants & (configured_variants - 1u)) != 0u) {
        fprintf(stderr,
            "[waverace64 cuda] wave_variants must be a power of two in [1,128]\n");
        abort();
    }
    uint32_t variants = randomize_waves ? configured_variants : 1u;
    WR64CudaHostSnapshotInfo* infos = NULL;
    wr64_cuda_build_reset_pool(env_kwargs, variants, &infos);

    Env* host_envs = (Env*)calloc((size_t)n, sizeof(Env));
    if (!host_envs) abort();
    for (int index = 0; index < n; index++) {
        wr64_cuda_fill_adapter(&host_envs[index], env_kwargs,
            (unsigned int)index, variants, &infos[0]);
    }
    free(infos);

    size_t requested_stack = (size_t)wr64_cuda_env_ulong(
        "WR64_CUDA_STACK_KIB", 8u, 1u, 1024u) * 1024u;
    wr64_cuda_fail("set device stack size",
        cudaDeviceSetLimit(cudaLimitStackSize, requested_stack));
    size_t actual_stack = 0;
    wr64_cuda_fail("read device stack size",
        cudaDeviceGetLimit(&actual_stack, cudaLimitStackSize));
    wr64_cuda_fail("allocate environments",
        cudaMalloc((void**)&g_wr64_cuda.envs, (size_t)n * sizeof(Env)));
    wr64_cuda_fail("upload environments",
        cudaMemcpy(g_wr64_cuda.envs, host_envs,
            (size_t)n * sizeof(Env), cudaMemcpyHostToDevice));
    free(host_envs);
    wr64_cuda_allocate_rdram(n);
    wr64_cuda_fail("allocate evaluator terminal RDRAM",
        cudaMalloc((void**)&g_wr64_cuda.terminal_rdram,
            WR64_CUDA_RDRAM_SIZE));
    wr64_cuda_fail("allocate evaluator terminal metadata",
        cudaMalloc((void**)&g_wr64_cuda.terminal, sizeof(WR64CudaTerminal)));
    wr64_cuda_fail("clear evaluator terminal metadata",
        cudaMemset(g_wr64_cuda.terminal, 0, sizeof(WR64CudaTerminal)));
    g_wr64_cuda.render_rdram = (uint8_t*)malloc(WR64_CUDA_RDRAM_SIZE);
    if (!g_wr64_cuda.render_rdram) abort();

    g_wr64_cuda.n = n;
    g_wr64_cuda.observations = observations;
    g_wr64_cuda.actions = actions;
    g_wr64_cuda.rewards = rewards;
    g_wr64_cuda.terminals = terminals;
    g_wr64_cuda.stream = 0;
    g_wr64_cuda.frameskip = wr64_cuda_dict_int(env_kwargs, "frameskip");
    if (g_wr64_cuda.frameskip < 1) g_wr64_cuda.frameskip = 2;
    g_wr64_cuda.step_block_size = (int)wr64_cuda_env_ulong(
        "WR64_CUDA_BLOCK_SIZE", 32u, 1u, 256u);
    if ((g_wr64_cuda.step_block_size
            & (g_wr64_cuda.step_block_size - 1)) != 0) {
        fprintf(stderr,
            "[waverace64 cuda] WR64_CUDA_BLOCK_SIZE must be a power of two\n");
        abort();
    }
    g_wr64_cuda.reset_block_size = (int)wr64_cuda_env_ulong(
        "WR64_CUDA_RESET_BLOCK_SIZE", WR64_CUDA_MAX_RESET_THREADS, 1u,
        WR64_CUDA_MAX_RESET_THREADS);
    if ((g_wr64_cuda.reset_block_size
            & (g_wr64_cuda.reset_block_size - 1)) != 0) {
        fprintf(stderr,
            "[waverace64 cuda] WR64_CUDA_RESET_BLOCK_SIZE must be a power of two\n");
        abort();
    }
    fprintf(stderr,
        "[waverace64 cuda] device-resident envs=%d rdram=%zu bytes "
        "rdram_va=%zu bytes stack=%zu bytes step_block=%d "
        "reset_block=%d\n",
        n, (size_t)n * WR64_CUDA_RDRAM_SIZE,
        g_wr64_cuda.rdram_reservation_size, actual_stack,
        g_wr64_cuda.step_block_size, g_wr64_cuda.reset_block_size);
    return g_wr64_cuda.envs;
}

void puf_bind_stream(cudaStream_t stream) {
    g_wr64_cuda.stream = stream;
}

void puf_init(Env*, Dict*) {
}

void puf_reset(Env*) {
    wr64_cuda_launch_reset(1, 0, 1);
}

void puf_eval_reset(Env*) {
    g_wr64_cuda.render_episode_valid = 0;
    g_wr64_cuda.terminal_presented = 0;
    g_wr64_cuda.render_paused = 0;
    g_wr64_cuda.render_terminal = 0;
    wr64_cuda_fail("clear evaluator terminal",
        cudaMemsetAsync(g_wr64_cuda.terminal, 0,
            sizeof(WR64CudaTerminal), g_wr64_cuda.stream));
    wr64_cuda_launch_reset(1, 1, 1);
}

void puf_step(Env*) {
    int block = g_wr64_cuda.step_block_size;
    int grid = (g_wr64_cuda.n + block - 1) / block;
    wr64_cuda_step_kernel<<<grid, block, 0, g_wr64_cuda.stream>>>(
        g_wr64_cuda.envs, g_wr64_cuda.n, g_wr64_cuda.rdram,
        g_wr64_cuda.rdram_stride,
        g_wr64_cuda.actions, g_wr64_cuda.observations,
        g_wr64_cuda.rewards, g_wr64_cuda.terminals);
    wr64_cuda_fail("launch step kernel", cudaGetLastError());
    wr64_cuda_launch_reset(0, 0, 0);
}

void puf_close(Env*) {
    CUcontext previous_context = NULL;
    wr64_cuda_driver_fail("read teardown CUDA context",
        cuCtxGetCurrent(&previous_context));
    int pushed_context = g_wr64_cuda.rdram_context != NULL
        && previous_context != g_wr64_cuda.rdram_context;
    if (pushed_context) {
        wr64_cuda_driver_fail("push owning CUDA context",
            cuCtxPushCurrent(g_wr64_cuda.rdram_context));
    }
    wr64_cuda_fail("synchronize Wave Race stream",
        cudaStreamSynchronize(g_wr64_cuda.stream));
    wr64_cuda_host_destroy(g_wr64_cuda.host);
    g_wr64_cuda.host = NULL;
    free(g_wr64_cuda.render_rdram);
    wr64_cuda_fail("free evaluator terminal metadata",
        cudaFree(g_wr64_cuda.terminal));
    wr64_cuda_fail("free evaluator terminal RDRAM",
        cudaFree(g_wr64_cuda.terminal_rdram));
    wr64_cuda_fail("free variant machine states",
        cudaFree(g_wr64_cuda.variant_machines));
    wr64_cuda_fail("free variant page payload",
        cudaFree(g_wr64_cuda.variant_data));
    wr64_cuda_fail("free variant page indices",
        cudaFree(g_wr64_cuda.variant_pages));
    wr64_cuda_fail("free variant offsets",
        cudaFree(g_wr64_cuda.variant_offsets));
    wr64_cuda_fail("free canonical reset",
        cudaFree(g_wr64_cuda.reset_canonical));
    wr64_cuda_free_rdram();
    wr64_cuda_fail("free environment state", cudaFree(g_wr64_cuda.envs));
    if (pushed_context) {
        CUcontext popped_context = NULL;
        wr64_cuda_driver_fail("pop owning CUDA context",
            cuCtxPopCurrent(&popped_context));
        if (popped_context != g_wr64_cuda.rdram_context) {
            fprintf(stderr,
                "[waverace64 cuda] teardown CUDA context stack changed\n");
            abort();
        }
    }
    memset(&g_wr64_cuda, 0, sizeof(g_wr64_cuda));
}

static WR64CudaHostMachineState wr64_cuda_machine_to_host(
        const WR64DeviceMachine* source) {
    WR64CudaHostMachineState machine = {};
    machine.ticks = source->ticks;
    machine.pad_buttons = source->pad_buttons;
    machine.pad_stick_x = source->pad_stick_x;
    machine.pad_stick_y = source->pad_stick_y;
    machine.resident_overlay = source->resident_overlay;
    machine.rounding_mode = source->rounding_mode;
    return machine;
}

void puf_render(Env*) {
    wr64_cuda_fail("synchronize evaluator stream",
        cudaStreamSynchronize(g_wr64_cuda.stream));
    Env host_env;
    WR64CudaTerminal terminal = {};
    wr64_cuda_fail("copy evaluator environment",
        cudaMemcpy(&host_env, g_wr64_cuda.envs,
            sizeof(host_env), cudaMemcpyDeviceToHost));
    wr64_cuda_fail("copy evaluator terminal metadata",
        cudaMemcpy(&terminal, g_wr64_cuda.terminal,
            sizeof(terminal), cudaMemcpyDeviceToHost));

    WR64CudaHostRenderInput input = {};
    const uint8_t* source_rdram = g_wr64_cuda.rdram;
    if (terminal.valid && !g_wr64_cuda.terminal_presented) {
        g_wr64_cuda.terminal_presented = 1;
    }
    if (g_wr64_cuda.terminal_presented) {
        source_rdram = g_wr64_cuda.terminal_rdram;
        input.flags |= WR64_CUDA_HOST_RENDER_TERMINAL;
        input.machine = wr64_cuda_machine_to_host(&terminal.machine);
        memcpy(&input.adapter, &terminal.state, sizeof(input.adapter));
        memcpy(input.policy_actions, terminal.actions,
            sizeof(input.policy_actions));
    } else {
        input.machine = wr64_cuda_machine_to_host(&host_env.machine);
        memcpy(&input.adapter, &host_env.state, sizeof(input.adapter));
        memcpy(input.policy_actions, host_env.last_actions,
            sizeof(input.policy_actions));
        if (!g_wr64_cuda.render_episode_valid
                || g_wr64_cuda.render_episode_id != host_env.episode_id) {
            input.flags |= WR64_CUDA_HOST_RENDER_NEW_EPISODE;
            g_wr64_cuda.render_episode_id = host_env.episode_id;
            g_wr64_cuda.render_episode_valid = 1;
        }
    }
    wr64_cuda_fail("copy evaluator RDRAM",
        cudaMemcpy(g_wr64_cuda.render_rdram, source_rdram,
            WR64_CUDA_RDRAM_SIZE, cudaMemcpyDeviceToHost));

    WR64CudaHostRenderOutput output = {};
    char error[WR64_CUDA_ERROR_BUFFER] = {};
    int status = wr64_cuda_host_render(g_wr64_cuda.host,
        g_wr64_cuda.render_rdram, WR64_CUDA_RDRAM_SIZE,
        &input, &output, error, sizeof(error));
    wr64_cuda_host_fail("human evaluator", status, error);
    if (output.window_should_close) exit(0);
    g_wr64_cuda.render_paused = output.paused;
    g_wr64_cuda.render_terminal = output.terminal_ready;

    WR64CudaHumanInput human = {};
    human.control = output.human_control;
    human.paused = output.paused;
    memcpy(human.actions, output.actions, sizeof(human.actions));
    wr64_cuda_fail("upload human evaluator controls",
        cudaMemcpy(&g_wr64_cuda.envs[0].human, &human,
            sizeof(human), cudaMemcpyHostToDevice));
    if (g_wr64_cuda.terminal_presented && !output.terminal_ready) {
        g_wr64_cuda.terminal_presented = 0;
        g_wr64_cuda.render_episode_valid = 0;
        int32_t clear = 0;
        wr64_cuda_fail("clear presented terminal",
            cudaMemcpy(&g_wr64_cuda.terminal->valid, &clear,
                sizeof(clear), cudaMemcpyHostToDevice));
    }
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "distance", log->distance);
    dict_set(out, "checkpoints", log->checkpoints);
    dict_set(out, "misses", log->misses);
    dict_set(out, "success_rate", log->success_rate);
    dict_set(out, "failure_rate", log->failure_rate);
    dict_set(out, "disqualification_rate", log->disqualification_rate);
    dict_set(out, "safety_timeout_rate", log->safety_timeout_rate);
    dict_set(out, "env_fault_rate", log->env_fault_rate);
    dict_set(out, "mean_speed", log->mean_speed);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "target_laps", log->target_laps);
    dict_set(out, "three_lap_success_rate", log->three_lap_success_rate);
    float success_inv = log->success_rate > 0.f
        ? 1.f / log->success_rate : 0.f;
    dict_set(out, "finish_time_ms",
        log->successful_race_time_ms * success_inv);
    dict_set(out, "lap_1_ms", log->successful_lap_1_ms * success_inv);
    dict_set(out, "lap_2_ms", log->successful_lap_2_ms * success_inv);
    dict_set(out, "lap_3_ms", log->successful_lap_3_ms * success_inv);
    dict_set(out, "n", log->n);
}

#endif
