#include "waverace64_cuda_host.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// waverace64.h is a header-only CPU Puffer environment and therefore defines
// its public entry points. Namespace those definitions inside this translation
// unit so the CUDA backend can provide the real puf_* symbols without linker
// collisions. Static helpers and the existing renderer remain directly usable.
#define init                    wr64_cuda_host_cpu_init
#define puf_close               wr64_cuda_host_cpu_close
#define add_log                 wr64_cuda_host_cpu_add_log
#define compute_observations    wr64_cuda_host_cpu_compute_observations
#define puffer_state_refresh    wr64_cuda_host_cpu_state_refresh
#define puf_reset               wr64_cuda_host_cpu_reset
#define puf_eval_reset          wr64_cuda_host_cpu_eval_reset
#define puf_step                wr64_cuda_host_cpu_step
#define puf_render              wr64_cuda_host_cpu_render
#define puf_init                wr64_cuda_host_cpu_puf_init
#define my_vec_init             wr64_cuda_host_cpu_vec_init
#define my_vec_close            wr64_cuda_host_cpu_vec_close
#define puf_log                 wr64_cuda_host_cpu_log
#include "waverace64.h"
#undef puf_log
#undef my_vec_close
#undef my_vec_init
#undef puf_init
#undef puf_render
#undef puf_step
#undef puf_eval_reset
#undef puf_reset
#undef puffer_state_refresh
#undef compute_observations
#undef add_log
#undef puf_close
#undef init

#if defined(__cplusplus)
static_assert(WR64_CUDA_HOST_RDRAM_SIZE == WR_RDRAM_SIZE,
    "host bridge RDRAM size drifted from the cartridge runtime");
static_assert(WR64_CUDA_HOST_ROUTE_NODES == WR64_MAX_COURSE_NODES,
    "host bridge route capacity drifted from the CPU adapter");
static_assert(WR64_CUDA_HOST_MAX_WAVE_VARIANTS == WR64_MAX_WAVE_VARIANTS,
    "host bridge wave capacity drifted from the CPU adapter");
#else
_Static_assert(WR64_CUDA_HOST_RDRAM_SIZE == WR_RDRAM_SIZE,
    "host bridge RDRAM size drifted from the cartridge runtime");
_Static_assert(WR64_CUDA_HOST_ROUTE_NODES == WR64_MAX_COURSE_NODES,
    "host bridge route capacity drifted from the CPU adapter");
_Static_assert(WR64_CUDA_HOST_MAX_WAVE_VARIANTS == WR64_MAX_WAVE_VARIANTS,
    "host bridge wave capacity drifted from the CPU adapter");
#endif

struct WR64CudaHost {
    char* rom_path;
#ifdef PUFFER_WAVERACE64_RENDER
    WaveRace64 render_env;
    float render_actions[WR64_CUDA_HOST_ACTIONS];
    int render_machine_ready;
#endif
};

static int wr64_cuda_host_error(int status, char* error, size_t error_size,
        const char* format, ...) {
    if (error != NULL && error_size > 0) {
        va_list args;
        va_start(args, format);
        vsnprintf(error, error_size, format, args);
        va_end(args);
        error[error_size - 1] = '\0';
    }
    return status;
}

static void wr64_cuda_host_clear_error(char* error, size_t error_size) {
    if (error != NULL && error_size > 0) error[0] = '\0';
}

static char* wr64_cuda_host_string_copy(const char* source) {
    size_t length = strlen(source) + 1;
    char* copy = (char*)malloc(length);
    if (copy != NULL) memcpy(copy, source, length);
    return copy;
}

static void wr64_cuda_host_adapter_from_cpu(
        WR64CudaHostAdapterState* destination, const State* source) {
    memset(destination, 0, sizeof(*destination));
    destination->tick = source->tick;
    destination->prev_a = source->prev_a;
    destination->prev_y = source->prev_y;
    destination->prev_b = source->prev_b;
    destination->episode_return = source->episode_return;
    destination->dist_total = source->dist_total;
    destination->progress_total = source->progress_total;
    destination->max_progress = source->max_progress;
    destination->velocity_x = source->velocity_x;
    destination->velocity_y = source->velocity_y;
    destination->velocity_z = source->velocity_z;
    destination->prev_course_progress = source->prev_course_progress;
    destination->prev_node = source->prev_node;
    destination->prev_lap = source->prev_lap;
    destination->checkpoints = source->checkpoints;
    destination->prev_misses = source->prev_misses;
    destination->misses = source->misses;
    destination->recovery = source->recovery;
    destination->success = source->success;
    destination->failed = source->failed;
    destination->disqualified = source->disqualified;
    destination->safety_timeout = source->safety_timeout;
    destination->env_fault = source->env_fault;
}

#ifdef PUFFER_WAVERACE64_RENDER
static void wr64_cuda_host_adapter_to_cpu(
        State* destination, const WR64CudaHostAdapterState* source) {
    memset(destination, 0, sizeof(*destination));
    destination->tick = source->tick;
    destination->prev_a = source->prev_a;
    destination->prev_y = source->prev_y;
    destination->prev_b = source->prev_b;
    destination->episode_return = source->episode_return;
    destination->dist_total = source->dist_total;
    destination->progress_total = source->progress_total;
    destination->max_progress = source->max_progress;
    destination->velocity_x = source->velocity_x;
    destination->velocity_y = source->velocity_y;
    destination->velocity_z = source->velocity_z;
    destination->prev_course_progress = source->prev_course_progress;
    destination->prev_node = source->prev_node;
    destination->prev_lap = source->prev_lap;
    destination->checkpoints = source->checkpoints;
    destination->prev_misses = source->prev_misses;
    destination->misses = source->misses;
    destination->recovery = source->recovery;
    destination->success = source->success;
    destination->failed = source->failed;
    destination->disqualified = source->disqualified;
    destination->safety_timeout = source->safety_timeout;
    destination->env_fault = source->env_fault;
}
#endif

static void wr64_cuda_host_machine_from_cpu(
        WR64CudaHostMachineState* destination, const WRMachine* source) {
    memset(destination, 0, sizeof(*destination));
    destination->ticks = source->ticks;
    destination->pad_buttons = source->pad_buttons;
    destination->pad_stick_x = source->pad_stick_x;
    destination->pad_stick_y = source->pad_stick_y;
    destination->resident_overlay = source->resident_overlay;
    // The supported root starts in the VR4300 default round-to-nearest mode.
    destination->rounding_mode = 0;
}

#ifdef PUFFER_WAVERACE64_RENDER
static void wr64_cuda_host_machine_to_cpu(
        WRMachine* destination, const WR64CudaHostMachineState* source) {
    destination->ticks = source->ticks;
    destination->pad_buttons = source->pad_buttons;
    destination->pad_stick_x = source->pad_stick_x;
    destination->pad_stick_y = source->pad_stick_y;
    destination->resident_overlay = source->resident_overlay;
}
#endif

static int wr64_cuda_host_initialize_adapter(WaveRace64* env) {
    memset(&env->state, 0, sizeof(env->state));
    env->state.prev_lap = wr64_lap(env);
    env->state.checkpoints = 0;
    env->state.prev_misses = wr64_misses(env);
    env->state.recovery = wr64_recovery(env);

    float x;
    float y;
    float z;
    wr64_position(env, &x, &y, &z);
    if (!isfinite(x) || !isfinite(y) || !isfinite(z)) return 0;
    env->vertical_origin = y;
    env->state.prev_node = wr64_node(env);
    env->state.prev_a = x;
    env->state.prev_y = y;
    env->state.prev_b = z;

    float fraction = 0.f;
    return wr64_reset_contract_valid(env, 3)
        && wr64_course_progress(
            env, &env->state.prev_course_progress, &fraction)
        && env->state.recovery == 0
        && !wr64_disqualified(env)
        && !wr64_ended(env)
        && !wr64_finished(env);
}

static void wr64_cuda_host_fill_snapshot_info(
        WR64CudaHostSnapshotInfo* info, const WaveRace64* source,
        uint32_t wave_seed, int32_t randomize_waves,
        uint32_t wave_variant) {
    memset(info, 0, sizeof(*info));
    info->abi_version = WR64_CUDA_HOST_ABI_VERSION;
    info->struct_size = (uint32_t)sizeof(*info);
    info->runtime_abi_version = WR_RUNTIME_ABI_VERSION;
    info->wave_seed = wave_seed;
    info->wave_variant = wave_variant;
    info->randomize_waves = randomize_waves;
    wr64_cuda_host_machine_from_cpu(&info->machine, &source->machine);
    wr64_cuda_host_adapter_from_cpu(&info->adapter, &source->state);
    info->vertical_origin = source->vertical_origin;
    memcpy(info->route_arc, source->route_arc, sizeof(info->route_arc));
    memcpy(info->route_pred, source->route_pred, sizeof(info->route_pred));
    info->route_total = source->route_total;
    info->route_nodes = source->route_nodes;
    info->route_valid = source->route_valid;
}

int wr64_cuda_host_create(const char* rom_path, WR64CudaHost** out,
        char* error, size_t error_size) {
    wr64_cuda_host_clear_error(error, error_size);
    if (out == NULL || rom_path == NULL || rom_path[0] == '\0') {
        return wr64_cuda_host_error(WR64_CUDA_HOST_INVALID_ARGUMENT,
            error, error_size, "rom_path and out are required");
    }
    *out = NULL;
    int abi_status = WR_RUNTIME_ABI_CHECK();
    if (abi_status != WR_RUNTIME_ABI_OK) {
        return wr64_cuda_host_error(WR64_CUDA_HOST_RUNTIME_ABI_MISMATCH,
            error, error_size, "Wave Race runtime ABI mismatch (status=%d)",
            abi_status);
    }

    WR64CudaHost* host = (WR64CudaHost*)calloc(1, sizeof(*host));
    if (host == NULL) {
        return wr64_cuda_host_error(WR64_CUDA_HOST_ALLOCATION_FAILED,
            error, error_size, "host bridge allocation failed");
    }
    host->rom_path = wr64_cuda_host_string_copy(rom_path);
    if (host->rom_path == NULL) {
        free(host);
        return wr64_cuda_host_error(WR64_CUDA_HOST_ALLOCATION_FAILED,
            error, error_size, "ROM path allocation failed");
    }
    *out = host;
    return WR64_CUDA_HOST_OK;
}

void wr64_cuda_host_destroy(WR64CudaHost* host) {
    if (host == NULL) return;
    wr64_cuda_host_render_close(host);
    free(host->rom_path);
    free(host);
}

size_t wr64_cuda_host_rdram_size(void) {
    return (size_t)WR64_CUDA_HOST_RDRAM_SIZE;
}

uint64_t wr64_cuda_host_wave_ticks(
        uint32_t wave_seed, uint32_t wave_variant) {
    return wr64_wave_variant_ticks(wave_seed, wave_variant);
}

int wr64_cuda_host_boot_snapshot(WR64CudaHost* host,
        uint32_t wave_seed, int32_t randomize_waves, uint32_t wave_variant,
        void* rdram_out, size_t rdram_size,
        WR64CudaHostSnapshotInfo* info_out,
        char* error, size_t error_size) {
    wr64_cuda_host_clear_error(error, error_size);
    if (info_out != NULL) memset(info_out, 0, sizeof(*info_out));
    if (host == NULL || rdram_out == NULL || info_out == NULL
            || (randomize_waves != 0 && randomize_waves != 1)
            || (randomize_waves
                && wave_variant >= WR64_CUDA_HOST_MAX_WAVE_VARIANTS)
            || rdram_size != (size_t)WR64_CUDA_HOST_RDRAM_SIZE) {
        return wr64_cuda_host_error(WR64_CUDA_HOST_INVALID_ARGUMENT,
            error, error_size,
            "host, randomize_waves in {0,1}, randomized wave_variant in "
            "[0,127], an exact 8 MiB RDRAM buffer, and info_out are required");
    }

    WaveRace64* env = (WaveRace64*)calloc(1, sizeof(*env));
    if (env == NULL) {
        return wr64_cuda_host_error(WR64_CUDA_HOST_ALLOCATION_FAILED,
            error, error_size, "bootstrap environment allocation failed");
    }
    env->frameskip = 2;
    env->num_agents = 1;
    env->randomize_waves = randomize_waves;
    env->wave_seed = wave_seed;
    env->wave_boot_variant = wave_variant;
    env->active_wave_variant = wave_variant;
    env->curriculum_start_laps = 3;
    env->curriculum_max_laps = 3;
    env->curriculum_laps = 3;
    env->rom_path = host->rom_path;

    int result = WR64_CUDA_HOST_OK;
    uint64_t fp_scope = wr_env_fp_enter();
    if (wr_machine_init(&env->machine, host->rom_path) != 0) {
        result = wr64_cuda_host_error(WR64_CUDA_HOST_ROM_FAILED,
            error, error_size, "cannot open pinned Wave Race ROM: %s",
            host->rom_path);
        goto cleanup;
    }
    if (randomize_waves) {
        env->machine.ticks = wr64_cuda_host_wave_ticks(
            wave_seed, wave_variant);
    }
    wr_current = &env->machine;
    wr_init_overlay_table();
    wr_install_fault_reporter(&env->machine);
    wr_dma_copy(env->machine.rdram, UINT32_C(0x80046800),
        env->machine.rom + 0x1000, 0xA95D0 - 0x1000);
    if (wr_boot_to_race(&env->machine, 8000) < 0) {
        result = wr64_cuda_host_error(WR64_CUDA_HOST_BOOT_FAILED,
            error, error_size,
            "cartridge did not reach Sunny Beach Time Trial");
        goto cleanup;
    }

    env->machine.pad_buttons = 0;
    env->machine.pad_stick_x = 0;
    env->machine.pad_stick_y = 0;
    wr_wr32(env->machine.rdram, WR64_CONFIG_LAPS_ADDR, 3);
    wr_wr32(env->machine.rdram, WR64_TARGET_LAPS_ADDR, 3);
    if (!wr64_build_route(env)
            || !wr64_reset_contract_valid(env, 3)
            || !wr64_cuda_host_initialize_adapter(env)) {
        result = wr64_cuda_host_error(
            WR64_CUDA_HOST_RACE_CONTRACT_FAILED, error, error_size,
            "booted state does not satisfy the fixed Time Trial contract");
        goto cleanup;
    }

    memcpy(rdram_out, env->machine.rdram, WR64_CUDA_HOST_RDRAM_SIZE);
    wr64_cuda_host_fill_snapshot_info(
        info_out, env, wave_seed, randomize_waves, wave_variant);

cleanup:
    if (wr_current == &env->machine) wr_current = NULL;
    if (env->machine.rdram != NULL) wr_machine_free(&env->machine);
    wr_env_fp_leave(fp_scope);
    free(env);
    return result;
}

#ifdef PUFFER_WAVERACE64_RENDER
static int wr64_cuda_host_ensure_render_machine(WR64CudaHost* host,
        char* error, size_t error_size) {
    if (host->render_machine_ready) return WR64_CUDA_HOST_OK;
    if (wr_machine_init(&host->render_env.machine, host->rom_path) != 0) {
        return wr64_cuda_host_error(WR64_CUDA_HOST_ROM_FAILED,
            error, error_size, "cannot initialize evaluator RDRAM mirror");
    }
    host->render_env.frameskip = 2;
    host->render_env.num_agents = 1;
    host->render_env.curriculum_start_laps = 3;
    host->render_env.curriculum_max_laps = 3;
    host->render_env.curriculum_laps = 3;
    host->render_env.rom_path = host->rom_path;
    host->render_env.agents[0].actions = host->render_actions;
    host->render_machine_ready = 1;
    return WR64_CUDA_HOST_OK;
}
#endif

int wr64_cuda_host_render(WR64CudaHost* host,
        const void* rdram, size_t rdram_size,
        const WR64CudaHostRenderInput* input,
        WR64CudaHostRenderOutput* output,
        char* error, size_t error_size) {
    wr64_cuda_host_clear_error(error, error_size);
    if (host == NULL || rdram == NULL || input == NULL || output == NULL
            || rdram_size != (size_t)WR64_CUDA_HOST_RDRAM_SIZE) {
        return wr64_cuda_host_error(WR64_CUDA_HOST_INVALID_ARGUMENT,
            error, error_size,
            "host, copied RDRAM, render input, and render output are required");
    }
    memset(output, 0, sizeof(*output));

#ifndef PUFFER_WAVERACE64_RENDER
    return wr64_cuda_host_error(WR64_CUDA_HOST_RENDER_UNAVAILABLE,
        error, error_size,
        "host bridge was built without PUFFER_WAVERACE64_RENDER");
#else
    int status = wr64_cuda_host_ensure_render_machine(
        host, error, error_size);
    if (status != WR64_CUDA_HOST_OK) return status;

    WaveRace64* env = &host->render_env;
    uint64_t fp_scope = wr_env_fp_enter();
    memcpy(env->machine.rdram, rdram, WR64_CUDA_HOST_RDRAM_SIZE);
    wr64_cuda_host_machine_to_cpu(&env->machine, &input->machine);
    wr64_cuda_host_adapter_to_cpu(&env->state, &input->adapter);
    memcpy(host->render_actions, input->policy_actions,
        sizeof(host->render_actions));
    wr_current = &env->machine;

    if (!wr64_build_route(env)) {
        wr_env_fp_leave(fp_scope);
        return wr64_cuda_host_error(
            WR64_CUDA_HOST_RACE_CONTRACT_FAILED, error, error_size,
            "copied evaluator state has an invalid course route");
    }
    if ((input->flags & WR64_CUDA_HOST_RENDER_NEW_EPISODE) != 0
            && env->client != NULL) {
        wr64_render_reset_episode(env);
    }
    if ((input->flags & WR64_CUDA_HOST_RENDER_TERMINAL) != 0
            && !wr64_render_terminal_ready(env)) {
        wr64_render_capture_terminal(env);
    }
    wr64_render_draw(env);

    memcpy(output->actions, host->render_actions,
        sizeof(output->actions));
    if (env->client != NULL) {
        output->human_control = env->client->human_control;
        output->paused = wr64_render_is_paused(env);
        output->terminal_ready = wr64_render_terminal_ready(env);
    }
    output->window_ready = IsWindowReady() ? 1 : 0;
    output->window_should_close = IsWindowReady() && WindowShouldClose();
    wr_env_fp_leave(fp_scope);
    return WR64_CUDA_HOST_OK;
#endif
}

void wr64_cuda_host_render_close(WR64CudaHost* host) {
    if (host == NULL) return;
#ifdef PUFFER_WAVERACE64_RENDER
    if (host->render_env.client != NULL) {
        wr64_render_close(&host->render_env);
    }
    if (host->render_machine_ready) {
        if (wr_current == &host->render_env.machine) wr_current = NULL;
        wr_machine_free(&host->render_env.machine);
        host->render_machine_ready = 0;
    }
    memset(&host->render_env, 0, sizeof(host->render_env));
    memset(host->render_actions, 0, sizeof(host->render_actions));
#endif
}
