#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    TEST_ENVS = 2,
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
    int steps;
    float seen_action;
} TestEnv;

static void c_reset(TestEnv* env) {
    (void)env;
}

static void c_step(TestEnv* env) {
    env->steps += 1;
    env->seen_action = env->actions[0];
    env->observations[0] = env->actions[0] + 100.0f;
    env->rewards[0] = env->actions[0] + 10.0f;
    env->terminals[0] = 0.0f;
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
#define Env TestEnv
#define Log TestLog
#include "../src/vecenv.h"

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
    (void)kind;
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
    return cudaMemcpy(destination, source, bytes, kind);
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

static int assertions;

static void require(int condition, const char* name) {
    assertions += 1;
    if (!condition) {
        fprintf(stderr, "FAIL %s\n", name);
        exit(1);
    }
}

static void bind_envs(
        TestEnv envs[TEST_ENVS],
        float observations[TEST_ENVS],
        float actions[TEST_ENVS],
        float rewards[TEST_ENVS],
        float terminals[TEST_ENVS]) {
    for (int i = 0; i < TEST_ENVS; i++) {
        envs[i].observations = &observations[i];
        envs[i].actions = &actions[i];
        envs[i].rewards = &rewards[i];
        envs[i].terminals = &terminals[i];
        envs[i].num_agents = 1;
    }
}

int main(void) {
    TestEnv envs[TEST_ENVS] = {0};
    float host_observations[TEST_ENVS] = {-1.0f, -1.0f};
    float host_actions[TEST_ENVS] = {-1.0f, -1.0f};
    float caller_actions[TEST_ENVS] = {2.0f, 4.0f};
    float host_rewards[TEST_ENVS] = {-1.0f, -1.0f};
    float host_terminals[TEST_ENVS] = {-1.0f, -1.0f};
    float device_observations[TEST_ENVS] = {-2.0f, -2.0f};
    float device_actions[TEST_ENVS] = {7.0f, 9.0f};
    float device_rewards[TEST_ENVS] = {-2.0f, -2.0f};
    float device_terminals[TEST_ENVS] = {-2.0f, -2.0f};

    bind_envs(envs, host_observations, host_actions, host_rewards, host_terminals);
    StaticVec vec = {
        .envs = envs,
        .size = TEST_ENVS,
        .total_agents = TEST_ENVS,
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
    cpu_vec_step(&vec);
    for (int i = 0; i < TEST_ENVS; i++) {
        require(envs[i].steps == 1, "default_cpu_calls_c_step_once_per_env");
        require(envs[i].seen_action == caller_actions[i],
            "default_cpu_action_copy_precedes_c_step");
        require(host_observations[i] == caller_actions[i] + 100.0f,
            "default_cpu_observation_is_current_after_step");
    }

    for (int i = 0; i < TEST_ENVS; i++) {
        envs[i].steps = 0;
        envs[i].seen_action = -1.0f;
        host_actions[i] = -1.0f;
        host_observations[i] = -1.0f;
    }
    vec.gpu_observations = device_observations;
    vec.gpu_actions = device_actions;
    vec.gpu_rewards = device_rewards;
    vec.gpu_terminals = device_terminals;

    gpu_vec_step(&vec);
    for (int i = 0; i < TEST_ENVS; i++) {
        require(envs[i].steps == 1, "default_gpu_calls_c_step_once_per_env");
        require(envs[i].seen_action == device_actions[i],
            "default_gpu_action_copy_precedes_c_step");
        require(device_observations[i] == device_actions[i] + 100.0f,
            "default_gpu_observation_copy_follows_c_step");
        require(device_rewards[i] == device_actions[i] + 10.0f,
            "default_gpu_reward_copy_follows_c_step");
    }

    printf("PASS vecenv_default_step assertions=%d\n", assertions);
    return 0;
}
