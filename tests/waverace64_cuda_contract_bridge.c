#define _GNU_SOURCE

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "ocean/waverace64/waverace64_cuda_host.h"
#include "ocean/waverace64/waverace64.h"

#define WR64_CONTRACT_LOG_FLOATS 20

typedef struct WR64ContractCpu {
    WaveRace64 env;
    Dict kwargs;
    float observations[WR64_OBS_SIZE];
    float actions[NUM_ATNS];
    float reward;
    float terminal;
} WR64ContractCpu;

_Static_assert(WR64_OBS_SIZE == 57,
    "CPU Wave Race observation contract changed");
_Static_assert(NUM_ATNS == 5,
    "CPU Wave Race action contract changed");
_Static_assert(sizeof(Log) == WR64_CONTRACT_LOG_FLOATS * sizeof(float),
    "CPU Wave Race log is no longer a flat 20-float reduction record");
_Static_assert(sizeof(State) == sizeof(WR64CudaHostAdapterState),
    "CPU and CUDA-exported adapter states differ in size");

static void wr64_contract_bind(WR64ContractCpu* oracle) {
    oracle->env.agents[0].observations = oracle->observations;
    oracle->env.agents[0].actions = oracle->actions;
    oracle->env.agents[0].rewards = &oracle->reward;
    oracle->env.agents[0].terminals = &oracle->terminal;
}

static void wr64_contract_set_defaults(WR64ContractCpu* oracle,
        const char* rom_path, int frameskip) {
    Dict* kwargs = &oracle->kwargs;
    dict_set_str(kwargs, "rom_path", rom_path);
    dict_set(kwargs, "frameskip", frameskip);
    dict_set(kwargs, "randomize_waves", 0);
    dict_set(kwargs, "wave_seed", 42);
    dict_set(kwargs, "wave_variants", 1);
    dict_set(kwargs, "reward_speed", 0);
    dict_set(kwargs, "reward_progress", 1);
    dict_set(kwargs, "reward_slip", 0);
    dict_set(kwargs, "reward_checkpoint", 0.1);
    dict_set(kwargs, "reward_miss", 0.5);
    dict_set(kwargs, "reward_finish", 10);
    dict_set(kwargs, "reward_fail", 2);
    dict_set(kwargs, "discount", 0.9995);
    dict_set(kwargs, "reward_mode", 0);
    dict_set(kwargs, "curriculum_start_laps", 3);
    dict_set(kwargs, "curriculum_max_laps", 3);
    dict_set(kwargs, "curriculum_successes_per_lap", 1);
}

void* wr64_contract_cpu_create(const char* rom_path, int frameskip) {
    if (rom_path == NULL || rom_path[0] == '\0' || frameskip < 1) return NULL;
    WR64ContractCpu* oracle = (WR64ContractCpu*)calloc(1, sizeof(*oracle));
    if (oracle == NULL) return NULL;
    wr64_contract_set_defaults(oracle, rom_path, frameskip);
    puf_init(&oracle->env, &oracle->kwargs);
    wr64_contract_bind(oracle);
    puf_reset(&oracle->env);
    return oracle;
}

int wr64_contract_cpu_reset(void* opaque, int clear_outputs) {
    WR64ContractCpu* oracle = (WR64ContractCpu*)opaque;
    if (oracle == NULL) return -1;
    if (clear_outputs) {
        oracle->reward = 0.f;
        oracle->terminal = 0.f;
    }
    puf_reset(&oracle->env);
    return 0;
}

int wr64_contract_cpu_step(void* opaque, const float actions[NUM_ATNS]) {
    WR64ContractCpu* oracle = (WR64ContractCpu*)opaque;
    if (oracle == NULL || actions == NULL) return -1;
    memcpy(oracle->actions, actions, sizeof(oracle->actions));
    puf_step(&oracle->env);
    return 0;
}

int wr64_contract_cpu_copy_rdram(void* opaque,
        void* destination, size_t destination_size) {
    WR64ContractCpu* oracle = (WR64ContractCpu*)opaque;
    if (oracle == NULL || destination == NULL
            || destination_size != WR_RDRAM_SIZE) return -1;
    memcpy(destination, oracle->env.machine.rdram, WR_RDRAM_SIZE);
    return 0;
}

int wr64_contract_cpu_copy_result(void* opaque,
        float observations[WR64_OBS_SIZE], float* reward, float* terminal,
        WR64CudaHostMachineState* machine,
        WR64CudaHostAdapterState* state,
        float log_values[WR64_CONTRACT_LOG_FLOATS],
        int32_t* curriculum_laps, int32_t* curriculum_successes) {
    WR64ContractCpu* oracle = (WR64ContractCpu*)opaque;
    if (oracle == NULL || observations == NULL || reward == NULL
            || terminal == NULL || machine == NULL || state == NULL
            || log_values == NULL || curriculum_laps == NULL
            || curriculum_successes == NULL) return -1;

    memcpy(observations, oracle->observations, sizeof(oracle->observations));
    *reward = oracle->reward;
    *terminal = oracle->terminal;

    memset(machine, 0, sizeof(*machine));
    machine->ticks = oracle->env.machine.ticks;
    machine->pad_buttons = oracle->env.machine.pad_buttons;
    machine->pad_stick_x = oracle->env.machine.pad_stick_x;
    machine->pad_stick_y = oracle->env.machine.pad_stick_y;
    machine->resident_overlay = oracle->env.machine.resident_overlay;
    machine->rounding_mode = 0;

    memcpy(state, &oracle->env.state, sizeof(*state));
    memcpy(log_values, &oracle->env.log, sizeof(oracle->env.log));
    *curriculum_laps = oracle->env.curriculum_laps;
    *curriculum_successes = oracle->env.curriculum_successes;
    return 0;
}

void wr64_contract_cpu_destroy(void* opaque) {
    WR64ContractCpu* oracle = (WR64ContractCpu*)opaque;
    if (oracle == NULL) return;
    puf_close(&oracle->env);
    dict_clear(&oracle->kwargs);
    free(oracle);
}
