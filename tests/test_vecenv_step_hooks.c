#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    SYNC_ENVS = 2,
    THREADED_ENVS = 5,
    THREADED_AGENTS = 8,
    THREADED_BUFFERS = 2,
};

typedef struct TestLog {
    float n;
} TestLog;

typedef struct TestEnv {
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;
    TestLog log;
    int fallback_steps;
} TestEnv;

static void c_reset(TestEnv* env) {
    (void)env;
}

void c_step(TestEnv* env) {
    env->fallback_steps += 1;
}

static void c_render(TestEnv* env) {
    (void)env;
}

static void c_close(TestEnv* env) {
    (void)env;
}

#define OBS_SIZE 1
#define NUM_ATNS 1
#define ACT_SIZES {5}
#define OBS_TENSOR_T FloatTensor
#define MY_VEC_STEP test_vec_step
#define MY_VEC_STEP_RANGE test_vec_step_range
#define Env TestEnv
#define Log TestLog
#include "../src/vecenv.h"

static int assertions;
static int sync_hook_calls;
static int sync_gpu_active;
static int sync_copy_stage;
static int sync_failures;
static float* sync_host_actions;
static float* sync_device_actions;
static float* sync_host_observations;
static float* sync_device_observations;
static float sync_expected_actions[SYNC_ENVS];

static atomic_int threaded_failures;
static atomic_int threaded_range_calls;
static atomic_int threaded_copy_stage[THREADED_BUFFERS];
static float* threaded_host_actions;
static float* threaded_device_actions;
static float* threaded_host_observations;
static float* threaded_device_observations;

static void require(int condition, const char* name) {
    assertions += 1;
    if (!condition) {
        fprintf(stderr, "FAIL %s\n", name);
        exit(1);
    }
}

static int copy_buffer(
        const float* destination,
        const float* source,
        const float* destination_base,
        const float* source_base) {
    int agents_per_buffer = THREADED_AGENTS / THREADED_BUFFERS;
    if (destination == destination_base && source == source_base) return 0;
    if (destination == destination_base + agents_per_buffer &&
            source == source_base + agents_per_buffer) return 1;
    return -1;
}

cudaError_t cudaHostAlloc(void** ptr, size_t bytes, unsigned int flags) {
    (void)flags;
    *ptr = malloc(bytes);
    return *ptr == NULL ? 1 : cudaSuccess;
}

cudaError_t cudaMalloc(void** ptr, size_t bytes) {
    *ptr = malloc(bytes);
    return *ptr == NULL ? 1 : cudaSuccess;
}

cudaError_t cudaMemcpy(
        void* destination,
        const void* source,
        size_t bytes,
        cudaMemcpyKind kind) {
    if (sync_gpu_active && destination == sync_host_actions &&
            source == sync_device_actions && kind == cudaMemcpyDeviceToHost) {
        if (sync_copy_stage != 0) sync_failures += 1;
        memmove(destination, source, bytes);
        sync_copy_stage = 1;
        return cudaSuccess;
    }
    if (sync_gpu_active && destination == sync_device_observations &&
            source == sync_host_observations && kind == cudaMemcpyHostToDevice) {
        if (sync_copy_stage != 2) sync_failures += 1;
        memmove(destination, source, bytes);
        sync_copy_stage = 3;
        return cudaSuccess;
    }
    memmove(destination, source, bytes);
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(
        void* destination,
        const void* source,
        size_t bytes,
        cudaMemcpyKind kind,
        cudaStream_t stream) {
    (void)stream;
    int buf = -1;
    if (threaded_host_actions != NULL && kind == cudaMemcpyDeviceToHost) {
        buf = copy_buffer(
            (const float*)destination,
            (const float*)source,
            threaded_host_actions,
            threaded_device_actions);
        if (buf >= 0) {
            int expected = 1;
            if (!atomic_compare_exchange_strong(
                    &threaded_copy_stage[buf], &expected, 2)) {
                atomic_fetch_add(&threaded_failures, 1);
            }
        }
    } else if (threaded_device_observations != NULL &&
            kind == cudaMemcpyHostToDevice) {
        buf = copy_buffer(
            (const float*)destination,
            (const float*)source,
            threaded_device_observations,
            threaded_host_observations);
        if (buf >= 0) {
            int expected = 3;
            if (!atomic_compare_exchange_strong(
                    &threaded_copy_stage[buf], &expected, 4)) {
                atomic_fetch_add(&threaded_failures, 1);
            }
        }
    }
    memmove(destination, source, bytes);
    return cudaSuccess;
}

cudaError_t cudaMemset(void* destination, int value, size_t bytes) {
    memset(destination, value, bytes);
    return cudaSuccess;
}

cudaError_t cudaFree(void* ptr) {
    free(ptr);
    return cudaSuccess;
}

cudaError_t cudaFreeHost(void* ptr) {
    free(ptr);
    return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) {
    (void)device;
    return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize(void) {
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    (void)stream;
    return cudaSuccess;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags) {
    (void)flags;
    *stream = NULL;
    return cudaSuccess;
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
    (void)stream;
    return cudaSuccess;
}

const char* cudaGetErrorString(cudaError_t error) {
    (void)error;
    return "test CUDA stub";
}

void my_init(TestEnv* env, Dict* kwargs) {
    (void)kwargs;
    env->num_agents = 1;
}

void my_log(TestLog* log, Dict* out) {
    (void)log;
    (void)out;
}

void test_vec_step(StaticVec* vec) {
    sync_hook_calls += 1;
    if (sync_gpu_active && sync_copy_stage != 1) sync_failures += 1;
    memset(vec->rewards, 0, (size_t)vec->total_agents * sizeof(float));
    memset(vec->terminals, 0, (size_t)vec->total_agents * sizeof(float));
    for (int i = 0; i < vec->total_agents; i++) {
        if (vec->actions[i] != sync_expected_actions[i]) sync_failures += 1;
        ((float*)vec->observations)[i] = vec->actions[i] + 200.0f;
        vec->rewards[i] = vec->actions[i] + 20.0f;
    }
    if (sync_gpu_active) sync_copy_stage = 2;
}

void test_vec_step_range(
        StaticVec* vec,
        int env_start,
        int env_count,
        int num_workers) {
    int buf = -1;
    if (env_start == 0 && env_count == 2) buf = 0;
    if (env_start == 2 && env_count == 3) buf = 1;
    if (buf < 0 || num_workers != 3) {
        atomic_fetch_add(&threaded_failures, 1);
        return;
    }
    if (atomic_load(&threaded_copy_stage[buf]) != 2) {
        atomic_fetch_add(&threaded_failures, 1);
    }

    int agents_per_buffer = vec->agents_per_buffer;
    int agent_start = buf * agents_per_buffer;
    for (int i = 0; i < agents_per_buffer; i++) {
        int agent = agent_start + i;
        float expected = 1000.0f + (float)agent;
        if (vec->actions[agent] != expected) {
            atomic_fetch_add(&threaded_failures, 1);
        }
        ((float*)vec->observations)[agent] = expected + 300.0f;
        vec->rewards[agent] = expected + 30.0f;
    }
    atomic_fetch_add(&threaded_range_calls, 1);
    atomic_store(&threaded_copy_stage[buf], 3);
}

static void threaded_net_callback(void* ctx, int buf, int t) {
    (void)ctx;
    if (t != 0 || buf < 0 || buf >= THREADED_BUFFERS) {
        atomic_fetch_add(&threaded_failures, 1);
        return;
    }
    int expected = 0;
    if (!atomic_compare_exchange_strong(
            &threaded_copy_stage[buf], &expected, 1)) {
        atomic_fetch_add(&threaded_failures, 1);
    }
    int agents_per_buffer = THREADED_AGENTS / THREADED_BUFFERS;
    int agent_start = buf * agents_per_buffer;
    for (int i = 0; i < agents_per_buffer; i++) {
        int agent = agent_start + i;
        threaded_device_actions[agent] = 1000.0f + (float)agent;
    }
}

static void test_synchronous_hooks(void) {
    TestEnv envs[SYNC_ENVS] = {0};
    float host_observations[SYNC_ENVS] = {-1.0f, -1.0f};
    float host_actions[SYNC_ENVS] = {-1.0f, -1.0f};
    float caller_actions[SYNC_ENVS] = {3.0f, 5.0f};
    float host_rewards[SYNC_ENVS] = {-1.0f, -1.0f};
    float host_terminals[SYNC_ENVS] = {-1.0f, -1.0f};
    float device_observations[SYNC_ENVS] = {-2.0f, -2.0f};
    float device_actions[SYNC_ENVS] = {7.0f, 11.0f};
    float device_rewards[SYNC_ENVS] = {-2.0f, -2.0f};
    float device_terminals[SYNC_ENVS] = {-2.0f, -2.0f};

    StaticVec vec = {
        .envs = envs,
        .size = SYNC_ENVS,
        .total_agents = SYNC_ENVS,
        .buffers = 1,
        .observations = host_observations,
        .actions = host_actions,
        .rewards = host_rewards,
        .terminals = host_terminals,
        .gpu_observations = host_observations,
        .gpu_actions = host_actions,
        .gpu_rewards = host_rewards,
        .gpu_terminals = host_terminals,
    };

    memcpy(host_actions, caller_actions, sizeof(host_actions));
    memcpy(sync_expected_actions, caller_actions, sizeof(caller_actions));
    sync_hook_calls = 0;
    sync_failures = 0;
    sync_gpu_active = 0;
    cpu_vec_step(&vec);
    require(sync_hook_calls == 1, "custom_cpu_calls_vector_hook_once");
    require(sync_failures == 0, "custom_cpu_hook_sees_current_actions");
    for (int i = 0; i < SYNC_ENVS; i++) {
        require(envs[i].fallback_steps == 0,
            "custom_cpu_does_not_call_per_env_fallback");
        require(host_observations[i] == host_actions[i] + 200.0f,
            "custom_cpu_observation_is_current_after_hook");
    }

    for (int i = 0; i < SYNC_ENVS; i++) {
        host_actions[i] = -1.0f;
        host_observations[i] = -1.0f;
        sync_expected_actions[i] = device_actions[i];
    }
    vec.gpu_observations = device_observations;
    vec.gpu_actions = device_actions;
    vec.gpu_rewards = device_rewards;
    vec.gpu_terminals = device_terminals;
    sync_host_actions = host_actions;
    sync_device_actions = device_actions;
    sync_host_observations = host_observations;
    sync_device_observations = device_observations;
    sync_hook_calls = 0;
    sync_copy_stage = 0;
    sync_failures = 0;
    sync_gpu_active = 1;

    gpu_vec_step(&vec);
    sync_gpu_active = 0;
    require(sync_hook_calls == 1, "custom_gpu_calls_vector_hook_once");
    require(sync_failures == 0, "custom_gpu_copy_order_is_action_hook_observation");
    require(sync_copy_stage == 3, "custom_gpu_completed_ordered_observation_copy");
    for (int i = 0; i < SYNC_ENVS; i++) {
        require(envs[i].fallback_steps == 0,
            "custom_gpu_does_not_call_per_env_fallback");
        require(device_observations[i] == device_actions[i] + 200.0f,
            "custom_gpu_observation_copy_follows_hook");
        require(device_rewards[i] == device_actions[i] + 20.0f,
            "custom_gpu_reward_copy_follows_hook");
    }
}

static void test_threaded_range_hooks(void) {
    TestEnv envs[THREADED_ENVS] = {0};
    float host_observations[THREADED_AGENTS] = {0};
    float device_observations[THREADED_AGENTS] = {0};
    float host_actions[THREADED_AGENTS] = {0};
    float device_actions[THREADED_AGENTS] = {0};
    float host_rewards[THREADED_AGENTS] = {0};
    float device_rewards[THREADED_AGENTS] = {0};
    float host_terminals[THREADED_AGENTS] = {0};
    float device_terminals[THREADED_AGENTS] = {0};
    int buffer_env_starts[THREADED_BUFFERS] = {0, 2};
    int buffer_env_counts[THREADED_BUFFERS] = {2, 3};
    cudaStream_t streams[THREADED_BUFFERS] = {NULL, NULL};

    envs[0].num_agents = 1;
    envs[1].num_agents = 3;
    envs[2].num_agents = 1;
    envs[3].num_agents = 1;
    envs[4].num_agents = 2;

    StaticVec vec = {
        .envs = envs,
        .size = THREADED_ENVS,
        .total_agents = THREADED_AGENTS,
        .buffers = THREADED_BUFFERS,
        .agents_per_buffer = THREADED_AGENTS / THREADED_BUFFERS,
        .buffer_env_starts = buffer_env_starts,
        .buffer_env_counts = buffer_env_counts,
        .observations = host_observations,
        .actions = host_actions,
        .rewards = host_rewards,
        .terminals = host_terminals,
        .gpu_observations = device_observations,
        .gpu_actions = device_actions,
        .gpu_rewards = device_rewards,
        .gpu_terminals = device_terminals,
        .streams = streams,
    };

    threaded_host_actions = host_actions;
    threaded_device_actions = device_actions;
    threaded_host_observations = host_observations;
    threaded_device_observations = device_observations;
    atomic_store(&threaded_failures, 0);
    atomic_store(&threaded_range_calls, 0);
    for (int buf = 0; buf < THREADED_BUFFERS; buf++) {
        atomic_store(&threaded_copy_stage[buf], 0);
    }

    create_static_threads(&vec, 6, 1, NULL, threaded_net_callback, NULL);
    static_vec_omp_step(&vec);

    require(atomic_load(&threaded_failures) == 0,
        "threaded_ranges_and_copy_order_are_correct");
    require(atomic_load(&threaded_range_calls) == THREADED_BUFFERS,
        "threaded_hook_runs_once_per_buffer");
    for (int buf = 0; buf < THREADED_BUFFERS; buf++) {
        require(atomic_load(&threaded_copy_stage[buf]) == 4,
            "threaded_observation_copy_follows_range_hook");
    }
    for (int agent = 0; agent < THREADED_AGENTS; agent++) {
        float expected_action = 1000.0f + (float)agent;
        require(host_actions[agent] == expected_action,
            "threaded_action_copy_precedes_range_hook");
        require(device_observations[agent] == expected_action + 300.0f,
            "threaded_range_observation_reaches_device_buffer");
    }
    for (int i = 0; i < THREADED_ENVS; i++) {
        require(envs[i].fallback_steps == 0,
            "threaded_range_does_not_call_per_env_fallback");
    }

    atomic_store(&vec.threading->shutdown, 1);
    for (int buf = 0; buf < THREADED_BUFFERS; buf++) {
        pthread_join(vec.threading->threads[buf], NULL);
    }
    free(vec.threading->buffer_states);
    free(vec.threading->threads);
    free(vec.threading->accum);
    free(vec.threading);
    vec.threading = NULL;
    threaded_host_actions = NULL;
    threaded_device_actions = NULL;
    threaded_host_observations = NULL;
    threaded_device_observations = NULL;
}

int main(void) {
    test_synchronous_hooks();
    test_threaded_range_hooks();
    printf("PASS vecenv_step_hooks assertions=%d\n", assertions);
    return 0;
}
