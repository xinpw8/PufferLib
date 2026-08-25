#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef float obs_t;
#include "ocean/waverace64/waverace64_cuda_adapter.cuh"
#include "pufferenv.h"
#include "ocean/waverace64/waverace64_cuda_types.cuh"
#include "ocean/waverace64/waverace64_cuda_host.h"

/*
Build from the PufferLib repository root after generating the device closure:

  python3 ocean/waverace64/generate_cuda_recomp.py \
    --runtime "$WR64_DIR" --rom "$ROM" \
    --output build/waverace64_recomp_device.inc
  gcc -O2 -std=c11 -fopenmp -ffp-contract=off -I. -Isrc -Ivendor \
    -Iraylib-5.5_linux_aarch64/include -I"$WR64_DIR/runtime" \
    -I"$WR64_DIR/RecompiledFuncs" -c \
    ocean/waverace64/waverace64_cuda_host.c \
    -o build/waverace64_cuda_contract_host.o
  gcc -O2 -std=c11 -fopenmp -ffp-contract=off -I. -Isrc -Ivendor \
    -Iraylib-5.5_linux_aarch64/include -I"$WR64_DIR/runtime" \
    -I"$WR64_DIR/RecompiledFuncs" -c \
    tests/waverace64_cuda_contract_bridge.c \
    -o build/waverace64_cuda_contract_bridge.o
  nvcc -O2 --threads 0 -arch=native -std=c++17 \
    -DWR64_SUPPRESS_DISPLAY_WATER_NORMALS=1 \
    --fmad=false --ftz=false --prec-div=true --prec-sqrt=true \
    --diag-suppress=68,177,550 -I. -Ibuild -Isrc -Ivendor \
    -Iraylib-5.5_linux_aarch64/include \
    -c ocean/waverace64/waverace64_device.cu \
    -o build/waverace64_cuda_contract_device.o
  nvcc -O2 -arch=native -std=c++17 -I. -Isrc -Ivendor \
    -Iraylib-5.5_linux_aarch64/include \
    -c tests/test_waverace64_cuda_contract.cu \
    -o build/test_waverace64_cuda_contract.o
  nvcc -arch=native build/test_waverace64_cuda_contract.o \
    build/waverace64_cuda_contract_device.o \
    build/waverace64_cuda_contract_bridge.o \
    build/waverace64_cuda_contract_host.o "$WR64_DIR/libwr64.a" \
    -Xcompiler=-fopenmp -lm -lpthread \
    -o build/test_waverace64_cuda_contract

Run with the pinned US Rev 1 ROM as the sole argument. An optional second
integer overrides frameskip for diagnosis; the contract test defaults to the
production value of two guest updates per Puffer transition.
*/

#define WR64_CONTRACT_LOG_FLOATS 20
#define WR64_CONTRACT_PHYSICS_OFFSET \
    (UINT32_C(0x80192690) - UINT32_C(0x80000000))
#define WR64_CONTRACT_PHYSICS_BYTES (UINT32_C(0x1718) * 4u)
#define WR64_CONTRACT_RIDER_OFFSET \
    (UINT32_C(0x801C2938) - UINT32_C(0x80000000))
#define WR64_CONTRACT_RIDER_BYTES (UINT32_C(0x0378) * 4u)
#define WR64_CONTRACT_HELPER_OFFSET \
    (UINT32_C(0x801C3C60) - UINT32_C(0x80000000))
#define WR64_CONTRACT_HELPER_BYTES UINT32_C(0xE8)
#define WR64_CONTRACT_WATER_OFFSET \
    (UINT32_C(0x80162420) - UINT32_C(0x80000000))
#define WR64_CONTRACT_WATER_BYTES (384u * 128u * 4u)
#define WR64_CONTRACT_GAME_STATE_OFFSET \
    (UINT32_C(0x800DAB24) - UINT32_C(0x80000000))
#define WR64_CONTRACT_ROUTE_TOTAL 29078.811f
#define WR64_CONTRACT_MAX_DECISIONS 4000
#define WR64_CONTRACT_RDRAM_SAMPLE_PERIOD 64

extern "C" {
void* wr64_contract_cpu_create(const char* rom_path, int frameskip);
int wr64_contract_cpu_reset(void* oracle, int clear_outputs);
int wr64_contract_cpu_step(void* oracle,
    const float actions[WR64_CUDA_NUM_ATNS]);
int wr64_contract_cpu_copy_rdram(void* oracle,
    void* destination, size_t destination_size);
int wr64_contract_cpu_copy_result(void* oracle,
    float observations[WR64_CUDA_OBS_SIZE], float* reward, float* terminal,
    WR64CudaHostMachineState* machine,
    WR64CudaHostAdapterState* state,
    float log_values[WR64_CONTRACT_LOG_FLOATS],
    int32_t* curriculum_laps, int32_t* curriculum_successes);
void wr64_contract_cpu_destroy(void* oracle);
}

static_assert(WR64_CUDA_OBS_SIZE == 57,
    "CUDA Wave Race observation contract changed");
static_assert(WR64_CUDA_NUM_ATNS == 5,
    "CUDA Wave Race action contract changed");
static_assert(sizeof(WR64CudaLog)
        == WR64_CONTRACT_LOG_FLOATS * sizeof(float),
    "CUDA Wave Race log is no longer a flat 20-float record");
static_assert(sizeof(WR64CudaState) == sizeof(WR64CudaHostAdapterState),
    "CUDA adapter and host-exported adapter state differ in size");

typedef struct WR64ContractResult {
    float observations[WR64_CUDA_OBS_SIZE];
    float reward;
    float terminal;
    WR64CudaHostMachineState machine;
    WR64CudaHostAdapterState state;
    float log_values[WR64_CONTRACT_LOG_FLOATS];
    int32_t curriculum_laps;
    int32_t curriculum_successes;
    int32_t device_error_before_reset;
    uint32_t indirect_target_before_reset;
} WR64ContractResult;

typedef struct WR64ContractDevice {
    uint8_t* rdram;
    uint8_t* reset_rdram;
    Env* env;
    uint32_t* variant_offsets;
    WR64DeviceMachine* variant_machines;
    float* observations;
    float* actions;
    float* reward;
    float* terminal;
} WR64ContractDevice;

typedef struct WR64FloatAudit {
    uint64_t compared;
    uint64_t non_bitwise;
    float max_absolute_error;
    float max_ulp_error;
} WR64FloatAudit;

static WR64FloatAudit g_float_audit;

static void wr64_cuda_require(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    fprintf(stderr, "CUDA %s failed: %s\n",
        operation, cudaGetErrorString(status));
    exit(2);
}

static uint32_t wr64_float_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float wr64_ulp_size(float value) {
    float magnitude = fabsf(value);
    if (!isfinite(magnitude)) return INFINITY;
    float next = nextafterf(magnitude, INFINITY);
    float ulp = next - magnitude;
    return ulp > 0.f ? ulp : FLT_TRUE_MIN;
}

static int wr64_compare_float(const char* category, const char* field,
        int decision, float expected, float actual, float absolute_floor,
        float maximum_ulps) {
    g_float_audit.compared++;
    if (wr64_float_bits(expected) == wr64_float_bits(actual)) return 0;
    g_float_audit.non_bitwise++;
    float absolute_error = fabsf(expected - actual);
    float ulp = fmaxf(wr64_ulp_size(expected), wr64_ulp_size(actual));
    float ulp_error = absolute_error / ulp;
    if (absolute_error > g_float_audit.max_absolute_error) {
        g_float_audit.max_absolute_error = absolute_error;
    }
    if (ulp_error > g_float_audit.max_ulp_error) {
        g_float_audit.max_ulp_error = ulp_error;
    }
    if (isfinite(expected) && isfinite(actual)
            && (absolute_error <= absolute_floor
                || ulp_error <= maximum_ulps)) return 0;
    fprintf(stderr,
        "FAIL decision=%d %s.%s expected=% .9g [0x%08x] "
        "actual=% .9g [0x%08x] abs=%g ulp=%g\n",
        decision, category, field, expected, wr64_float_bits(expected),
        actual, wr64_float_bits(actual), absolute_error, ulp_error);
    return 1;
}

static int wr64_compare_i32(const char* category, const char* field,
        int decision, int32_t expected, int32_t actual) {
    if (expected == actual) return 0;
    fprintf(stderr,
        "FAIL decision=%d %s.%s expected=%d actual=%d\n",
        decision, category, field, expected, actual);
    return 1;
}

static int wr64_compare_machine(int decision,
        const WR64CudaHostMachineState* expected,
        const WR64CudaHostMachineState* actual) {
    int failures = 0;
#define WR64_MACHINE_FIELD(name) \
    failures += wr64_compare_i32("machine", #name, decision, \
        (int32_t)expected->name, (int32_t)actual->name)
    if (expected->ticks != actual->ticks) {
        fprintf(stderr,
            "FAIL decision=%d machine.ticks expected=%llu actual=%llu\n",
            decision, (unsigned long long)expected->ticks,
            (unsigned long long)actual->ticks);
        failures++;
    }
    WR64_MACHINE_FIELD(pad_buttons);
    WR64_MACHINE_FIELD(pad_stick_x);
    WR64_MACHINE_FIELD(pad_stick_y);
    WR64_MACHINE_FIELD(resident_overlay);
    WR64_MACHINE_FIELD(rounding_mode);
#undef WR64_MACHINE_FIELD
    return failures;
}

static int wr64_compare_state(int decision,
        const WR64CudaHostAdapterState* expected,
        const WR64CudaHostAdapterState* actual) {
    int failures = 0;
#define WR64_STATE_FLOAT(name) \
    failures += wr64_compare_float("state", #name, decision, \
        expected->name, actual->name, 2e-6f, 8.f)
#define WR64_STATE_INT(name) \
    failures += wr64_compare_i32("state", #name, decision, \
        expected->name, actual->name)
    WR64_STATE_INT(tick);
    WR64_STATE_FLOAT(prev_a);
    WR64_STATE_FLOAT(prev_y);
    WR64_STATE_FLOAT(prev_b);
    WR64_STATE_FLOAT(episode_return);
    WR64_STATE_FLOAT(dist_total);
    WR64_STATE_FLOAT(progress_total);
    WR64_STATE_FLOAT(max_progress);
    WR64_STATE_FLOAT(velocity_x);
    WR64_STATE_FLOAT(velocity_y);
    WR64_STATE_FLOAT(velocity_z);
    WR64_STATE_FLOAT(prev_course_progress);
    WR64_STATE_INT(prev_node);
    WR64_STATE_INT(prev_lap);
    WR64_STATE_INT(checkpoints);
    WR64_STATE_INT(prev_misses);
    WR64_STATE_INT(misses);
    WR64_STATE_INT(recovery);
    WR64_STATE_INT(success);
    WR64_STATE_INT(failed);
    WR64_STATE_INT(disqualified);
    WR64_STATE_INT(safety_timeout);
    WR64_STATE_INT(env_fault);
#undef WR64_STATE_INT
#undef WR64_STATE_FLOAT
    return failures;
}

static int wr64_compare_results(const char* phase, int decision,
        const WR64ContractResult* expected,
        const WR64ContractResult* actual) {
    static const char* log_names[WR64_CONTRACT_LOG_FLOATS] = {
        "perf", "score", "distance", "checkpoints", "misses",
        "success_rate", "failure_rate", "disqualification_rate",
        "safety_timeout_rate", "env_fault_rate", "mean_speed",
        "episode_return", "episode_length", "target_laps",
        "three_lap_success_rate", "successful_race_time_ms",
        "successful_lap_1_ms", "successful_lap_2_ms",
        "successful_lap_3_ms", "n"
    };
    int failures = 0;
    for (int index = 0; index < WR64_CUDA_OBS_SIZE; index++) {
        char field[24];
        snprintf(field, sizeof(field), "observation[%d]", index);
        failures += wr64_compare_float(phase, field, decision,
            expected->observations[index], actual->observations[index],
            2e-6f, 8.f);
    }
    failures += wr64_compare_float(phase, "reward", decision,
        expected->reward, actual->reward, 2e-7f, 8.f);
    failures += wr64_compare_float(phase, "terminal", decision,
        expected->terminal, actual->terminal, 0.f, 0.f);
    failures += wr64_compare_machine(
        decision, &expected->machine, &actual->machine);
    failures += wr64_compare_state(
        decision, &expected->state, &actual->state);
    for (int index = 0; index < WR64_CONTRACT_LOG_FLOATS; index++) {
        failures += wr64_compare_float("log", log_names[index], decision,
            expected->log_values[index], actual->log_values[index],
            2e-6f, 16.f);
    }
    failures += wr64_compare_i32("curriculum", "laps", decision,
        expected->curriculum_laps, actual->curriculum_laps);
    failures += wr64_compare_i32("curriculum", "successes", decision,
        expected->curriculum_successes, actual->curriculum_successes);
    if (actual->device_error_before_reset != 0) {
        fprintf(stderr,
            "FAIL decision=%d device_error=%d indirect_target=0x%08x\n",
            decision, actual->device_error_before_reset,
            actual->indirect_target_before_reset);
        failures++;
    }
    return failures;
}

static void wr64_adapter_from_snapshot(WR64CudaAdapter* adapter,
        const WR64CudaHostSnapshotInfo* snapshot, int frameskip) {
    memset(adapter, 0, sizeof(*adapter));
    adapter->frameskip = frameskip;
    adapter->rng = 0;
    adapter->randomize_waves = 0;
    adapter->wave_seed = 42;
    adapter->wave_rng_state = wr64_cuda_wave_stream_seed(42, 0);
    adapter->wave_boot_variant = 0;
    adapter->active_wave_variant = 0;
    adapter->wave_variants = 1;
    adapter->reward_speed = 0.f;
    adapter->reward_progress = 1.f;
    adapter->reward_slip = 0.f;
    adapter->reward_checkpoint = 0.1f;
    adapter->reward_miss = 0.5f;
    adapter->reward_finish = 10.f;
    adapter->reward_fail = 2.f;
    adapter->discount = 0.9995f;
    adapter->reward_mode = 0;
    adapter->curriculum_start_laps = 3;
    adapter->curriculum_max_laps = 3;
    adapter->curriculum_successes_per_lap = 1;
    adapter->curriculum_laps = 3;
    adapter->curriculum_successes = 0;
    adapter->vertical_origin = snapshot->vertical_origin;
    memcpy(adapter->route_arc, snapshot->route_arc,
        sizeof(adapter->route_arc));
    memcpy(adapter->route_pred, snapshot->route_pred,
        sizeof(adapter->route_pred));
    adapter->route_total = snapshot->route_total;
    adapter->route_nodes = snapshot->route_nodes;
    adapter->route_valid = snapshot->route_valid;
}

static WR64DeviceMachine wr64_machine_from_snapshot(
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

static void wr64_contract_device_create(WR64ContractDevice* device,
        const uint8_t* reset_rdram,
        const WR64CudaHostSnapshotInfo* snapshot, int frameskip) {
    memset(device, 0, sizeof(*device));
    Env env = {};
    wr64_adapter_from_snapshot(&env, snapshot, frameskip);
    env.machine = wr64_machine_from_snapshot(&snapshot->machine);
    env.needs_reset = 1;
    WR64DeviceMachine reset_machine = env.machine;
    const uint32_t variant_offsets[2] = {0, 0};

    wr64_cuda_require(cudaMalloc((void**)&device->rdram,
        WR64_CUDA_RDRAM_SIZE), "allocate live RDRAM");
    wr64_cuda_require(cudaMalloc((void**)&device->reset_rdram,
        WR64_CUDA_RDRAM_SIZE), "allocate reset RDRAM");
    wr64_cuda_require(cudaMalloc((void**)&device->env,
        sizeof(*device->env)), "allocate production Env");
    wr64_cuda_require(cudaMalloc((void**)&device->variant_offsets,
        sizeof(variant_offsets)), "allocate variant offsets");
    wr64_cuda_require(cudaMalloc((void**)&device->variant_machines,
        sizeof(reset_machine)), "allocate reset machine");
    wr64_cuda_require(cudaMalloc((void**)&device->observations,
        WR64_CUDA_OBS_SIZE * sizeof(float)), "allocate observations");
    wr64_cuda_require(cudaMalloc((void**)&device->actions,
        WR64_CUDA_NUM_ATNS * sizeof(float)), "allocate actions");
    wr64_cuda_require(cudaMalloc((void**)&device->reward,
        sizeof(float)), "allocate reward");
    wr64_cuda_require(cudaMalloc((void**)&device->terminal,
        sizeof(float)), "allocate terminal");

    wr64_cuda_require(cudaMemcpy(device->reset_rdram, reset_rdram,
        WR64_CUDA_RDRAM_SIZE, cudaMemcpyHostToDevice),
        "upload reset RDRAM");
    wr64_cuda_require(cudaMemcpy(device->env, &env,
        sizeof(env), cudaMemcpyHostToDevice), "upload production Env");
    wr64_cuda_require(cudaMemcpy(device->variant_offsets, variant_offsets,
        sizeof(variant_offsets), cudaMemcpyHostToDevice),
        "upload variant offsets");
    wr64_cuda_require(cudaMemcpy(device->variant_machines, &reset_machine,
        sizeof(reset_machine), cudaMemcpyHostToDevice),
        "upload reset machine");
    wr64_cuda_require(cudaMemset(device->reward, 0, sizeof(float)),
        "clear reward");
    wr64_cuda_require(cudaMemset(device->terminal, 0, sizeof(float)),
        "clear terminal");
}

static int wr64_contract_device_reset(WR64ContractDevice* device,
        int force, int clear_outputs) {
    wr64_cuda_reset_kernel<<<1, WR64_CUDA_RESET_THREADS>>>(
        device->env, 1, device->rdram, WR64_CUDA_RDRAM_SIZE,
        device->reset_rdram,
        device->variant_offsets, NULL, NULL, device->variant_machines,
        device->observations, device->reward, device->terminal,
        force, 0, clear_outputs, NULL, NULL);
    wr64_cuda_require(cudaDeviceSynchronize(), "production reset kernel");
    Env env;
    wr64_cuda_require(cudaMemcpy(&env, device->env, sizeof(env),
        cudaMemcpyDeviceToHost), "download reset Env");
    return env.needs_reset == 0 && env.machine.error == 0;
}

static int wr64_contract_device_step(WR64ContractDevice* device,
        const float actions[WR64_CUDA_NUM_ATNS],
        WR64DeviceMachine* machine_before_reset) {
    wr64_cuda_require(cudaMemcpy(device->actions, actions,
        WR64_CUDA_NUM_ATNS * sizeof(float), cudaMemcpyHostToDevice),
        "upload step actions");
    wr64_cuda_step_kernel<<<1, 1>>>(device->env, 1, device->rdram,
        WR64_CUDA_RDRAM_SIZE, device->actions, device->observations,
        device->reward, device->terminal);
    wr64_cuda_require(cudaDeviceSynchronize(), "production step kernel");
    Env env;
    wr64_cuda_require(cudaMemcpy(&env, device->env, sizeof(env),
        cudaMemcpyDeviceToHost), "download pre-reset Env");
    *machine_before_reset = env.machine;
    return env.needs_reset;
}

static void wr64_contract_device_destroy(WR64ContractDevice* device) {
    cudaFree(device->terminal);
    cudaFree(device->reward);
    cudaFree(device->actions);
    cudaFree(device->observations);
    cudaFree(device->variant_machines);
    cudaFree(device->variant_offsets);
    cudaFree(device->env);
    cudaFree(device->reset_rdram);
    cudaFree(device->rdram);
    memset(device, 0, sizeof(*device));
}

static int wr64_contract_fetch_cpu(
        void* oracle, WR64ContractResult* result) {
    memset(result, 0, sizeof(*result));
    return wr64_contract_cpu_copy_result(oracle,
        result->observations, &result->reward, &result->terminal,
        &result->machine, &result->state, result->log_values,
        &result->curriculum_laps, &result->curriculum_successes);
}

static void wr64_contract_fetch_device(WR64ContractDevice* device,
        const WR64DeviceMachine* machine_before_reset,
        WR64ContractResult* result) {
    Env env;
    memset(result, 0, sizeof(*result));
    wr64_cuda_require(cudaMemcpy(result->observations, device->observations,
        sizeof(result->observations), cudaMemcpyDeviceToHost),
        "download observations");
    wr64_cuda_require(cudaMemcpy(&result->reward, device->reward,
        sizeof(result->reward), cudaMemcpyDeviceToHost),
        "download reward");
    wr64_cuda_require(cudaMemcpy(&result->terminal, device->terminal,
        sizeof(result->terminal), cudaMemcpyDeviceToHost),
        "download terminal");
    wr64_cuda_require(cudaMemcpy(&env, device->env,
        sizeof(env), cudaMemcpyDeviceToHost), "download production Env");

    result->machine.ticks = env.machine.ticks;
    result->machine.pad_buttons = env.machine.pad_buttons;
    result->machine.pad_stick_x = env.machine.pad_stick_x;
    result->machine.pad_stick_y = env.machine.pad_stick_y;
    result->machine.resident_overlay = env.machine.resident_overlay;
    result->machine.rounding_mode = env.machine.rounding_mode;
    memcpy(&result->state, &env.state, sizeof(result->state));
    memcpy(result->log_values, &env.log, sizeof(env.log));
    result->curriculum_laps = env.curriculum_laps;
    result->curriculum_successes = env.curriculum_successes;
    if (machine_before_reset != NULL) {
        result->device_error_before_reset = machine_before_reset->error;
        result->indirect_target_before_reset =
            machine_before_reset->indirect_target;
    } else {
        result->device_error_before_reset = env.machine.error;
        result->indirect_target_before_reset = env.machine.indirect_target;
    }
}

static size_t wr64_first_difference(const uint8_t* expected,
        const uint8_t* actual, size_t offset, size_t size) {
    for (size_t byte = 0; byte < size; byte++) {
        if (expected[offset + byte] != actual[offset + byte]) {
            return offset + byte;
        }
    }
    return SIZE_MAX;
}

static int wr64_compare_region(const char* name, int decision,
        const uint8_t* expected, const uint8_t* actual,
        size_t offset, size_t size) {
    size_t difference = wr64_first_difference(
        expected, actual, offset, size);
    if (difference == SIZE_MAX) return 0;
    fprintf(stderr,
        "FAIL decision=%d RDRAM.%s offset=0x%zx expected=%02x actual=%02x\n",
        decision, name, difference,
        expected[difference], actual[difference]);
    return 1;
}

static int wr64_compare_authoritative_rdram(int decision, void* oracle,
        const WR64ContractDevice* device,
        uint8_t* cpu_rdram, uint8_t* gpu_rdram) {
    if (wr64_contract_cpu_copy_rdram(
            oracle, cpu_rdram, WR64_CUDA_RDRAM_SIZE) != 0) return 1;
    wr64_cuda_require(cudaMemcpy(gpu_rdram + WR64_CONTRACT_PHYSICS_OFFSET,
        device->rdram + WR64_CONTRACT_PHYSICS_OFFSET,
        WR64_CONTRACT_PHYSICS_BYTES, cudaMemcpyDeviceToHost),
        "download physics region");
    wr64_cuda_require(cudaMemcpy(gpu_rdram + WR64_CONTRACT_RIDER_OFFSET,
        device->rdram + WR64_CONTRACT_RIDER_OFFSET,
        WR64_CONTRACT_RIDER_BYTES, cudaMemcpyDeviceToHost),
        "download rider region");
    wr64_cuda_require(cudaMemcpy(gpu_rdram + WR64_CONTRACT_HELPER_OFFSET,
        device->rdram + WR64_CONTRACT_HELPER_OFFSET,
        WR64_CONTRACT_HELPER_BYTES, cudaMemcpyDeviceToHost),
        "download helper region");
    wr64_cuda_require(cudaMemcpy(gpu_rdram + WR64_CONTRACT_WATER_OFFSET,
        device->rdram + WR64_CONTRACT_WATER_OFFSET,
        WR64_CONTRACT_WATER_BYTES, cudaMemcpyDeviceToHost),
        "download water region");
    wr64_cuda_require(cudaMemcpy(gpu_rdram + WR64_CONTRACT_GAME_STATE_OFFSET,
        device->rdram + WR64_CONTRACT_GAME_STATE_OFFSET, sizeof(uint32_t),
        cudaMemcpyDeviceToHost), "download game state");
    int failures = 0;
    failures += wr64_compare_region("physics", decision,
        cpu_rdram, gpu_rdram,
        WR64_CONTRACT_PHYSICS_OFFSET, WR64_CONTRACT_PHYSICS_BYTES);
    // The rider record also contains presentation-only one-shot flags written
    // by the omitted display-list chain. Compare every field consumed by the
    // Puffer contract and official race logic instead of treating those GPU
    // render flags as training state.
    static const uint32_t rider_fields[] = {
        0x0000u, 0x0004u, 0x000Cu, 0x012Cu, 0x0134u, 0x013Cu,
        0x0168u, 0x0178u, 0x017Cu, 0x0180u, 0x019Cu,
        0x02ECu, 0x02F4u,
    };
    static const char* rider_names[] = {
        "rider.lap", "rider.race_position", "rider.node",
        "rider.power", "rider.misses", "rider.disqualified",
        "rider.lap_time", "rider.lap_split_1", "rider.lap_split_2",
        "rider.lap_split_3", "rider.total_time", "rider.ended",
        "rider.finished",
    };
    for (size_t field = 0;
            field < sizeof(rider_fields) / sizeof(rider_fields[0]); field++) {
        failures += wr64_compare_region(rider_names[field], decision,
            cpu_rdram, gpu_rdram,
            WR64_CONTRACT_RIDER_OFFSET + rider_fields[field],
            sizeof(uint32_t));
    }
    failures += wr64_compare_region("helper", decision,
        cpu_rdram, gpu_rdram,
        WR64_CONTRACT_HELPER_OFFSET, WR64_CONTRACT_HELPER_BYTES);
    failures += wr64_compare_region("water", decision,
        cpu_rdram, gpu_rdram,
        WR64_CONTRACT_WATER_OFFSET, WR64_CONTRACT_WATER_BYTES);
    failures += wr64_compare_region("game_state", decision,
        cpu_rdram, gpu_rdram,
        WR64_CONTRACT_GAME_STATE_OFFSET, sizeof(uint32_t));
    return failures;
}

static int wr64_nearest_stick(float desired) {
    static const int8_t detents[15] = {
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

static void wr64_route_controller(const float* observations,
        float actions[WR64_CUDA_NUM_ATNS]) {
    const float steer_gain = 134.347687f;
    const float throttle_angle = 0.228220314f;
    const float dampen_angle = 0.242280304f;
    const float high_throttle_angle = 2.70581746f;
    const float pass_scale = 1.75507069f;
    const float curve_near_blend = 0.427550882f;
    const float curve_far_blend = 0.f;
    const float curve_distance = 484.638611f;
    const float slide_angle = 0.245567679f;

    float center_x = observations[17] * observations[19];
    float center_z = -observations[18] * observations[19];
    float pass_x = observations[24] * observations[26];
    float pass_z = -observations[25] * observations[26];
    float dx = center_x + pass_scale * (pass_x - center_x);
    float dz = center_z + pass_scale * (pass_z - center_z);
    if (observations[30] > 0.5f) {
        float next_x = observations[27] * observations[29];
        float next_z = -observations[28] * observations[29];
        float blend = observations[26]
                < curve_distance / WR64_CONTRACT_ROUTE_TOTAL
            ? curve_near_blend : curve_far_blend;
        dx = dx * (1.f - blend) + next_x * blend;
        dz = dz * (1.f - blend) + next_z * blend;
    }
    float angle = atan2f(dz, dx);
    int steer = (int)lrintf(angle * steer_gain);
    if (steer > 80) steer = 80;
    if (steer < -80) steer = -80;
    int throttle = fabsf(angle) <= throttle_angle;
    int dampen = fabsf(angle) > dampen_angle;
    int throttle_alias = observations[15] * (float)WR64_CUDA_MAX_STEPS
            >= 59.5f
        && high_throttle_angle > 0.f
        && fabsf(angle) > high_throttle_angle;
    int slide = slide_angle > 0.f && fabsf(angle) > slide_angle;
    actions[0] = (float)wr64_nearest_stick((float)steer);
    actions[1] = 1.f;
    actions[2] = (float)(throttle || throttle_alias);
    actions[3] = (float)dampen;
    actions[4] = (float)slide;
}

static int wr64_run_one_step(int decision, const char* phase,
        void* oracle, WR64ContractDevice* device,
        const float actions[WR64_CUDA_NUM_ATNS],
        WR64ContractResult* cpu_result, WR64ContractResult* gpu_result,
        uint8_t* cpu_rdram, uint8_t* gpu_rdram) {
    if (wr64_contract_cpu_step(oracle, actions) != 0) {
        fprintf(stderr, "CPU oracle step failed at decision %d\n", decision);
        return 1;
    }
    WR64DeviceMachine pre_reset_machine = {};
    int gpu_done = wr64_contract_device_step(
        device, actions, &pre_reset_machine);
    if (wr64_contract_fetch_cpu(oracle, cpu_result) != 0) {
        fprintf(stderr, "CPU result fetch failed at decision %d\n", decision);
        return 1;
    }
    int cpu_done = cpu_result->terminal == 1.f;
    if (cpu_done != gpu_done) {
        fprintf(stderr,
            "FAIL decision=%d terminal disagreement CPU=%d CUDA=%d\n",
            decision, cpu_done, gpu_done);
        return 1;
    }
    if (gpu_done && !wr64_contract_device_reset(device, 0, 0)) {
        fprintf(stderr,
            "FAIL decision=%d CUDA autoreset contract rejected\n", decision);
        return 1;
    }
    wr64_contract_fetch_device(device, &pre_reset_machine, gpu_result);
    int failures = wr64_compare_results(
        phase, decision, cpu_result, gpu_result);
    if (!gpu_done && (decision <= 2
            || strcmp(phase, "post-autoreset") == 0
            || decision % WR64_CONTRACT_RDRAM_SAMPLE_PERIOD == 0)) {
        failures += wr64_compare_authoritative_rdram(
            decision, oracle, device, cpu_rdram, gpu_rdram);
    }
    if (gpu_done) {
        failures += wr64_compare_authoritative_rdram(
            decision, oracle, device, cpu_rdram, gpu_rdram);
    }
    return failures;
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr,
            "usage: %s /path/to/baserom.us.rev1.z64 [frameskip]\n", argv[0]);
        return 2;
    }
    int frameskip = argc == 3 ? atoi(argv[2]) : 2;
    if (frameskip < 1 || frameskip > 16) {
        fprintf(stderr, "frameskip must be in [1,16]\n");
        return 2;
    }
    wr64_cuda_require(cudaDeviceSetLimit(
        cudaLimitStackSize, 8u * 1024u), "set device stack limit");

    uint8_t* reset_rdram = (uint8_t*)malloc(WR64_CUDA_RDRAM_SIZE);
    uint8_t* cpu_rdram = (uint8_t*)malloc(WR64_CUDA_RDRAM_SIZE);
    uint8_t* gpu_rdram = (uint8_t*)malloc(WR64_CUDA_RDRAM_SIZE);
    if (reset_rdram == NULL || cpu_rdram == NULL || gpu_rdram == NULL) {
        fprintf(stderr, "cannot allocate contract RDRAM buffers\n");
        return 2;
    }
    memset(gpu_rdram, 0, WR64_CUDA_RDRAM_SIZE);

    char error[256] = {};
    WR64CudaHost* bootstrap = NULL;
    int host_status = wr64_cuda_host_create(
        argv[1], &bootstrap, error, sizeof(error));
    if (host_status != WR64_CUDA_HOST_OK) {
        fprintf(stderr, "host bootstrap create failed: %s\n", error);
        return 2;
    }
    WR64CudaHostSnapshotInfo snapshot = {};
    host_status = wr64_cuda_host_boot_snapshot(bootstrap,
        42, 0, 0, reset_rdram, WR64_CUDA_RDRAM_SIZE,
        &snapshot, error, sizeof(error));
    wr64_cuda_host_destroy(bootstrap);
    if (host_status != WR64_CUDA_HOST_OK) {
        fprintf(stderr, "host bootstrap failed: %s\n", error);
        return 2;
    }
    if (snapshot.abi_version != WR64_CUDA_HOST_ABI_VERSION
            || snapshot.struct_size != sizeof(snapshot)
            || !snapshot.route_valid || snapshot.route_nodes <= 0) {
        fprintf(stderr, "host bootstrap returned invalid contract metadata\n");
        return 2;
    }

    void* oracle = wr64_contract_cpu_create(argv[1], frameskip);
    if (oracle == NULL) {
        fprintf(stderr, "CPU Puffer oracle failed to boot\n");
        return 2;
    }
    if (wr64_contract_cpu_copy_rdram(
            oracle, cpu_rdram, WR64_CUDA_RDRAM_SIZE) != 0) {
        fprintf(stderr, "CPU reset RDRAM fetch failed\n");
        return 2;
    }
    size_t reset_difference = wr64_first_difference(
        cpu_rdram, reset_rdram, 0, WR64_CUDA_RDRAM_SIZE);
    if (reset_difference != SIZE_MAX) {
        fprintf(stderr,
            "FAIL reset RDRAM offset=0x%zx CPU=%02x bootstrap=%02x\n",
            reset_difference, cpu_rdram[reset_difference],
            reset_rdram[reset_difference]);
        return 1;
    }
    printf("PASS authentic reset RDRAM %u bytes\n", WR64_CUDA_RDRAM_SIZE);

    WR64ContractDevice device;
    wr64_contract_device_create(&device, reset_rdram, &snapshot, frameskip);
    if (!wr64_contract_device_reset(&device, 1, 1)) {
        fprintf(stderr, "CUDA reset adapter rejected authentic snapshot\n");
        return 1;
    }

    WR64ContractResult cpu_result = {};
    WR64ContractResult gpu_result = {};
    if (wr64_contract_fetch_cpu(oracle, &cpu_result) != 0) {
        fprintf(stderr, "CPU reset result fetch failed\n");
        return 2;
    }
    wr64_contract_fetch_device(&device, NULL, &gpu_result);
    int failures = wr64_compare_results(
        "reset", 0, &cpu_result, &gpu_result);
    if (failures != 0) return 1;
    puts("PASS reset observations, adapter state, machine, and log contract");

    const float low_actions[WR64_CUDA_NUM_ATNS] = {0, 0, 0, 0, 0};
    const float high_actions[WR64_CUDA_NUM_ATNS] = {14, 8, 1, 1, 1};
    failures += wr64_run_one_step(1, "low-action", oracle, &device,
        low_actions, &cpu_result, &gpu_result, cpu_rdram, gpu_rdram);
    failures += wr64_run_one_step(2, "high-action", oracle, &device,
        high_actions, &cpu_result, &gpu_result, cpu_rdram, gpu_rdram);
    if (failures != 0) return 1;
    printf("PASS five-head action mapping and frameskip=%d transition contract\n",
        frameskip);

    if (wr64_contract_cpu_reset(oracle, 1) != 0
            || !wr64_contract_device_reset(&device, 1, 1)) {
        fprintf(stderr, "pre-race differential reset failed\n");
        return 1;
    }
    if (wr64_contract_fetch_cpu(oracle, &cpu_result) != 0) return 2;
    wr64_contract_fetch_device(&device, NULL, &gpu_result);
    failures += wr64_compare_results(
        "pre-race-reset", 0, &cpu_result, &gpu_result);
    if (failures != 0) return 1;

    int terminal_decision = -1;
    for (int decision = 1;
            decision <= WR64_CONTRACT_MAX_DECISIONS; decision++) {
        float actions[WR64_CUDA_NUM_ATNS];
        wr64_route_controller(cpu_result.observations, actions);
        failures += wr64_run_one_step(decision, "race", oracle, &device,
            actions, &cpu_result, &gpu_result, cpu_rdram, gpu_rdram);
        if (failures != 0) break;
        if (cpu_result.terminal == 1.f) {
            terminal_decision = decision;
            break;
        }
    }
    if (failures == 0 && terminal_decision < 0) {
        fprintf(stderr,
            "FAIL deterministic controller did not reach an official finish "
            "within %d decisions\n", WR64_CONTRACT_MAX_DECISIONS);
        failures++;
    }
    if (failures == 0) {
        if (cpu_result.log_values[5] != 1.f
                || cpu_result.log_values[6] != 0.f
                || cpu_result.log_values[7] != 0.f
                || cpu_result.log_values[8] != 0.f
                || cpu_result.log_values[9] != 0.f
                || cpu_result.log_values[13] != 3.f
                || cpu_result.log_values[14] != 1.f
                || cpu_result.log_values[19] != 1.f) {
            fprintf(stderr,
                "FAIL terminal log is not one successful official "
                "three-lap race\n");
            failures++;
        }
    }
    if (failures == 0) {
        printf("PASS official three-lap finish decisions=%d updates=%.0f "
            "race_ms=%.0f laps=[%.0f,%.0f,%.0f]\n",
            terminal_decision, cpu_result.log_values[12],
            cpu_result.log_values[15], cpu_result.log_values[16],
            cpu_result.log_values[17], cpu_result.log_values[18]);
    }
    for (int post_reset = 1; failures == 0 && post_reset <= 3;
            post_reset++) {
        float actions[WR64_CUDA_NUM_ATNS];
        wr64_route_controller(cpu_result.observations, actions);
        failures += wr64_run_one_step(
            terminal_decision + post_reset, "post-autoreset",
            oracle, &device, actions, &cpu_result, &gpu_result,
            cpu_rdram, gpu_rdram);
        if (cpu_result.terminal != 0.f) {
            fprintf(stderr,
                "FAIL post-autoreset transition %d terminated\n", post_reset);
            failures++;
        }
    }
    if (failures == 0) {
        puts("PASS autoreset and first three transitions of the next episode");
    }
    printf("FLOAT AUDIT compared=%llu non_bitwise=%llu "
        "max_abs=%g max_ulp=%g\n",
        (unsigned long long)g_float_audit.compared,
        (unsigned long long)g_float_audit.non_bitwise,
        g_float_audit.max_absolute_error, g_float_audit.max_ulp_error);

    wr64_contract_device_destroy(&device);
    wr64_contract_cpu_destroy(oracle);
    free(gpu_rdram);
    free(cpu_rdram);
    free(reset_rdram);
    if (failures == 0) {
        puts("PASS complete CPU/CUDA Wave Race Puffer contract differential");
    }
    return failures ? 1 : 0;
}
