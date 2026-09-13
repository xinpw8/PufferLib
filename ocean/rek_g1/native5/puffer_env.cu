#ifndef REK_NATIVE5_PUFFER_ENV_CU
#define REK_NATIVE5_PUFFER_ENV_CU

#define PUF_BACKEND PUF_GPU
#define PUFFER_ENV_UNCLIPPED_REWARDS
#define PUFFER_ENV_GPU_ACTION_MASK
#define PUFFER_ENV_GPU_ROLLOUT_CHECK

#include <cuda_runtime.h>
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef float obs_t;
#include "pufferenv.h"
#include "runtime_api.h"

#define OBS_SIZE REK_NATIVE5_OBSERVATION_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {REK_NATIVE5_ACTION_COUNT}

struct Log : RekNative5Log {};

struct Env {
    Log log;
    Agent agents[1];
    int num_agents;
    int tag;
    int boundary_reached;
    unsigned int rng;
};

static_assert(sizeof(Log) == sizeof(RekNative5Log), "Log ABI mismatch");
static_assert(offsetof(Env, log) == 0, "Log must start the Env record");

static struct {
    RekNative5Runtime* runtime;
    Env* envs;
    cudaStream_t stream;
} rek_native5_binding;

static void rek_native5_require_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        fprintf(stderr, "REK native 5.0 %s: %s\n", operation,
            cudaGetErrorString(result));
        abort();
    }
}

static void rek_native5_require_runtime(int result, const char* operation) {
    if (result != 0) {
        const char* error = rek_native5_error();
        fprintf(stderr, "REK native 5.0 %s: %s\n", operation,
            error ? error : "runtime failed without an error message");
        abort();
    }
}

static const char* rek_native5_required_path(Dict* kwargs, const char* key) {
    DictItem* item = dict_find(kwargs, key);
    if (item == NULL || item->str == NULL || item->str[0] == '\0'
            || strcmp(item->str, "None") == 0) {
        fprintf(stderr, "REK native 5.0 requires --env.%s=PATH\n", key);
        abort();
    }
    return item->str;
}

Env* puf_vec_create(int n, Dict* kwargs, obs_t* observations,
        float* actions, float* rewards, float* terminals) {
    if (n <= 0 || rek_native5_binding.runtime != NULL) {
        fprintf(stderr, "REK native 5.0 requires a positive, single active batch\n");
        abort();
    }
    RekNative5Config config = {};
    config.abi_version = REK_NATIVE5_RUNTIME_ABI;
    config.model_path = rek_native5_required_path(kwargs, "model_path");
    config.physics_export_path = rek_native5_required_path(kwargs, "physics_export_path");
    config.assets_path = rek_native5_required_path(kwargs, "assets_path");
    config.motion_features_path = rek_native5_required_path(kwargs, "motion_features_path");
    config.controller_encoder_path = rek_native5_required_path(kwargs, "controller_encoder_path");
    config.controller_decoder_path = rek_native5_required_path(kwargs, "controller_decoder_path");
    config.arenas = n;
    DictItem* seed = dict_find(kwargs, "seed");
    config.seed = seed ? (uint32_t)seed->value : 73u;
    DictItem* segment = dict_find(kwargs, "locomotion_segment_ticks");
    config.locomotion_segment_ticks = segment ? (int)segment->value : 1;
    if (config.locomotion_segment_ticks != 1) {
        fprintf(stderr, "REK native 5.0 currently requires locomotion_segment_ticks=1\n");
        abort();
    }
    const uint32_t durations[REK_NATIVE5_MOVE_COUNT] = {
        35, 27, 31, 45, 32, 45, 157, 145, 158, 139, 134, 138, 73, 75, 68, 71, 103
    };
    for (int move = 0; move < REK_NATIVE5_MOVE_COUNT; ++move) {
        char key[64];
        snprintf(key, sizeof(key), "move_duration_%d", move);
        DictItem* item = dict_find(kwargs, key);
        if (item && (!(item->value >= 1.0 && item->value <= (double)UINT32_MAX)
                || item->value != (double)(uint32_t)item->value)) {
            fprintf(stderr, "REK native 5.0 env.%s must be a positive uint32\n", key);
            abort();
        }
        config.move_duration_ticks[move] = item ? (uint32_t)item->value : durations[move];
    }

    Env* host_envs = (Env*)calloc((size_t)n, sizeof(Env));
    if (host_envs == NULL) {
        fprintf(stderr, "REK native 5.0 could not allocate batch metadata\n");
        abort();
    }
    for (int i = 0; i < n; ++i) {
        host_envs[i].num_agents = 1;
        host_envs[i].rng = config.seed + (unsigned int)i;
    }
    Env* envs = NULL;
    rek_native5_require_cuda(cudaMalloc((void**)&envs, (size_t)n * sizeof(Env)),
        "allocate batch metadata");
    rek_native5_require_cuda(cudaMemcpy(envs, host_envs, (size_t)n * sizeof(Env),
        cudaMemcpyHostToDevice), "upload batch metadata");
    free(host_envs);

    RekNative5Buffers buffers = {};
    buffers.observations = observations;
    buffers.actions = actions;
    buffers.rewards = rewards;
    buffers.terminals = terminals;
    buffers.logs = reinterpret_cast<RekNative5Log*>(envs);
    buffers.log_stride_bytes = sizeof(Env);
    RekNative5Runtime* runtime = rek_native5_create(&config, &buffers, 0);
    if (runtime == NULL) {
        cudaFree(envs);
        rek_native5_require_runtime(1, "create runtime");
    }
    rek_native5_binding.runtime = runtime;
    rek_native5_binding.envs = envs;
    rek_native5_binding.stream = 0;
    return envs;
}

void puf_bind_action_mask(uint8_t* action_mask) {
    rek_native5_require_runtime(rek_native5_bind_action_mask(rek_native5_binding.runtime,
        action_mask, rek_native5_binding.stream), "bind action mask");
}

void puf_bind_stream(cudaStream_t stream) {
    rek_native5_binding.stream = stream;
}

void puf_check_rollout(void) {
    rek_native5_require_runtime(rek_native5_check_status(rek_native5_binding.runtime,
        rek_native5_binding.stream), "check completed rollout before training");
}

void puf_init(Env*, Dict*) {
    /* The GPU trainer initializes the entire batch through puf_vec_create. */
}

void puf_reset(Env*) {
    rek_native5_require_runtime(rek_native5_reset(rek_native5_binding.runtime,
        rek_native5_binding.stream), "reset runtime");
}

void puf_step(Env*) {
    rek_native5_require_runtime(rek_native5_step(rek_native5_binding.runtime,
        rek_native5_binding.stream), "step runtime");
}

void puf_close(Env*) {
    rek_native5_require_runtime(rek_native5_check_status(rek_native5_binding.runtime,
        rek_native5_binding.stream), "check final runtime state");
    rek_native5_require_runtime(rek_native5_close(rek_native5_binding.runtime),
        "close runtime");
    rek_native5_require_cuda(cudaFree(rek_native5_binding.envs), "free batch metadata");
    memset(&rek_native5_binding, 0, sizeof(rek_native5_binding));
}

void puf_render(Env*) {
    fprintf(stderr, "REK native 5.0 evaluation requires --headless\n");
    abort();
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "hits", log->hits);
    dict_set(out, "falls", log->falls);
    dict_set(out, "wins", log->wins);
    dict_set(out, "losses", log->losses);
    dict_set(out, "draws", log->draws);
    dict_set(out, "actions_invalid", log->actions_invalid);
    dict_set(out, "finite_failures", log->finite_failures);
    dict_set(out, "n", log->n);
}

#endif
