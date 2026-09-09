#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    int fallback_resets;
} TestEnv;

void c_reset(TestEnv* env) {
    env->fallback_resets += 1;
}

static void c_step(TestEnv* env) {
    (void)env;
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
#define MY_VEC_RESET test_vec_reset
#define Env TestEnv
#define Log TestLog
#include "../src/vecenv.h"

static int reset_calls;

void my_init(TestEnv* env, Dict* kwargs) {
    (void)kwargs;
    env->num_agents = 1;
}

void my_log(TestLog* log, Dict* out) {
    (void)log;
    (void)out;
}

void test_vec_reset(StaticVec* vec) {
    reset_calls += 1;
    for (int i = 0; i < vec->total_agents; i++) {
        ((float*)vec->observations)[i] = 100.0f + (float)i;
    }
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

static void require(int condition, const char* name) {
    if (!condition) {
        fprintf(stderr, "FAIL %s\n", name);
        exit(1);
    }
}

static void run_reset_case(int gpu) {
    TestEnv envs[2] = {0};
    float host_observations[2] = {-1.0f, -1.0f};
    float device_observations[2] = {-2.0f, -2.0f};
    float host_rewards[2] = {3.0f, 4.0f};
    float device_rewards[2] = {5.0f, 6.0f};
    float host_terminals[2] = {7.0f, 8.0f};
    float device_terminals[2] = {9.0f, 10.0f};
    StaticVec vec = {
        .envs = envs,
        .size = 2,
        .total_agents = 2,
        .buffers = 1,
        .observations = host_observations,
        .rewards = host_rewards,
        .terminals = host_terminals,
        .gpu_observations = device_observations,
        .gpu_rewards = device_rewards,
        .gpu_terminals = device_terminals,
        .gpu = gpu,
    };

    reset_calls = 0;
    static_vec_reset(&vec);
    require(reset_calls == 1, "vector_reset_hook_called_once");
    require(envs[0].fallback_resets == 0 && envs[1].fallback_resets == 0,
        "per_env_reset_bypassed");
    require(host_observations[0] == 100.0f && host_observations[1] == 101.0f,
        "hook_populates_host_observations");
    if (gpu) {
        require(device_observations[0] == 100.0f && device_observations[1] == 101.0f,
            "gpu_reset_copies_hook_observations");
        require(device_rewards[0] == 0.0f && device_rewards[1] == 0.0f,
            "gpu_reset_clears_device_rewards");
        require(device_terminals[0] == 0.0f && device_terminals[1] == 0.0f,
            "gpu_reset_clears_device_terminals");
    } else {
        require(host_rewards[0] == 0.0f && host_rewards[1] == 0.0f,
            "cpu_reset_clears_host_rewards");
        require(host_terminals[0] == 0.0f && host_terminals[1] == 0.0f,
            "cpu_reset_clears_host_terminals");
    }
}

int main(void) {
    run_reset_case(0);
    run_reset_case(1);
    puts("PASS vecenv_reset_hook");
    return 0;
}
