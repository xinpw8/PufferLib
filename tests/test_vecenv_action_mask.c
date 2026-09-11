#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    TEST_AGENTS = 2,
    TEST_ACTIONS = 4,
};

typedef struct TestLog {
    float n;
} TestLog;

typedef struct TestEnv {
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned char* action_mask;
    int num_agents;
    unsigned int rng;
    TestLog log;
    int steps;
} TestEnv;

static void c_reset(TestEnv* env) {
    (void)env;
}

static void c_step(TestEnv* env) {
    int selected = (int)env->actions[0];
    for (int action = 0; action < TEST_ACTIONS; action++) {
        env->action_mask[action] = (unsigned char)(action == selected);
    }
    env->observations[0] = (float)(10 * selected);
    env->rewards[0] = (float)selected;
    env->terminals[0] = 0.0f;
    env->steps += 1;
}

static void c_render(TestEnv* env) {
    (void)env;
}

static void c_close(TestEnv* env) {
    (void)env;
}

#define OBS_SIZE 1
#define NUM_ATNS 1
#define ACT_SIZES {TEST_ACTIONS}
#define OBS_TENSOR_T FloatTensor
#define MY_ACTION_MASK TEST_ACTIONS
#define MY_VEC_CLOSE
#define MY_SHARED
#define MY_SHARED_CLOSE
#define MY_GET
#define MY_PUT
#define Env TestEnv
#define Log TestLog
#include "../src/vecenv.h"

static void* expected_mask_destination;
static const void* expected_mask_source;
static size_t expected_mask_bytes;
static int matching_mask_copies;

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
    if (destination == expected_mask_destination &&
            source == expected_mask_source &&
            bytes == expected_mask_bytes &&
            kind == cudaMemcpyHostToDevice) {
        matching_mask_copies += 1;
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

cudaError_t cudaStreamCreateWithFlags(
        cudaStream_t* stream,
        unsigned int flags) {
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

void my_vec_close(TestEnv* envs) {
    (void)envs;
}

void* my_shared(void* env, Dict* kwargs) {
    (void)env;
    (void)kwargs;
    return NULL;
}

void my_shared_close(void* env) {
    (void)env;
}

void* my_get(void* env, Dict* out) {
    (void)env;
    (void)out;
    return NULL;
}

int my_put(void* env, Dict* kwargs) {
    (void)env;
    (void)kwargs;
    return 0;
}

static int assertions;

static void require(int condition, const char* name) {
    assertions += 1;
    if (!condition) {
        fprintf(stderr, "FAIL %s\n", name);
        exit(1);
    }
}

int main(void) {
    TestEnv envs[TEST_AGENTS] = {0};
    float host_observations[TEST_AGENTS] = {0};
    float device_observations[TEST_AGENTS] = {-1.0f, -1.0f};
    float host_actions[TEST_AGENTS] = {-1.0f, -1.0f};
    float device_actions[TEST_AGENTS] = {1.0f, 3.0f};
    float host_rewards[TEST_AGENTS] = {0};
    float device_rewards[TEST_AGENTS] = {-1.0f, -1.0f};
    float host_terminals[TEST_AGENTS] = {0};
    float device_terminals[TEST_AGENTS] = {-1.0f, -1.0f};
    unsigned char host_mask[TEST_AGENTS * TEST_ACTIONS];
    unsigned char device_mask[TEST_AGENTS * TEST_ACTIONS];
    memset(host_mask, 0xcc, sizeof(host_mask));
    memset(device_mask, 0xee, sizeof(device_mask));

    for (int agent = 0; agent < TEST_AGENTS; agent++) {
        envs[agent].observations = &host_observations[agent];
        envs[agent].actions = &host_actions[agent];
        envs[agent].rewards = &host_rewards[agent];
        envs[agent].terminals = &host_terminals[agent];
        envs[agent].action_mask = &host_mask[agent * TEST_ACTIONS];
        envs[agent].num_agents = 1;
    }

    StaticVec vec = {
        .envs = envs,
        .size = TEST_AGENTS,
        .total_agents = TEST_AGENTS,
        .buffers = 1,
        .observations = host_observations,
        .actions = host_actions,
        .rewards = host_rewards,
        .terminals = host_terminals,
        .action_mask = host_mask,
        .gpu_observations = device_observations,
        .gpu_actions = device_actions,
        .gpu_rewards = device_rewards,
        .gpu_terminals = device_terminals,
        .gpu_action_mask = device_mask,
        .action_mask_size = TEST_ACTIONS,
        .gpu = 1,
    };
    expected_mask_destination = device_mask;
    expected_mask_source = host_mask;
    expected_mask_bytes = sizeof(host_mask);

    gpu_vec_step(&vec);

    require(matching_mask_copies == 1,
        "synchronous_step_uploads_one_complete_action_mask");
    for (int agent = 0; agent < TEST_AGENTS; agent++) {
        require(envs[agent].steps == 1, "each_environment_steps_once");
        require(host_actions[agent] == device_actions[agent],
            "device_action_arrives_before_environment_step");
        for (int action = 0; action < TEST_ACTIONS; action++) {
            unsigned char expected =
                (unsigned char)(action == (int)device_actions[agent]);
            require(host_mask[agent * TEST_ACTIONS + action] == expected,
                "environment_writes_expected_host_mask");
            require(device_mask[agent * TEST_ACTIONS + action] == expected,
                "post_step_mask_is_visible_on_device");
        }
    }

    printf("PASS vecenv_gpu_action_mask assertions=%d bytes=%zu\n",
        assertions, sizeof(host_mask));
    return 0;
}
