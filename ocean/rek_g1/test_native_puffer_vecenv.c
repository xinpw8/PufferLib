#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "native_puffer_vector.h"

enum {
    TEST_VEC_ENVS = 2,
    TEST_VEC_OBS = 4,
    TEST_VEC_CATEGORIES = 20,
};

typedef struct TestVecLog {
    float n;
} TestVecLog;

typedef struct TestVecEnv {
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned char* action_mask;
    int num_agents;
    unsigned int rng;
    TestVecLog log;
    int fallback_steps;
} TestVecEnv;

static void c_reset(TestVecEnv* env) {
    (void)env;
}

static void c_step(TestVecEnv* env) {
    env->fallback_steps += 1;
}

static void c_render(TestVecEnv* env) {
    (void)env;
}

static void c_close(TestVecEnv* env) {
    (void)env;
}

typedef struct StaticVec StaticVec;
static void test_rek_g1_vec_step(StaticVec* vec);

#define OBS_SIZE TEST_VEC_OBS
#define NUM_ATNS REK_G1_PUFFER_ACTION_HEADS
#define ACT_SIZES {TEST_VEC_CATEGORIES}
#define OBS_TENSOR_T FloatTensor
#define MY_ACTION_MASK TEST_VEC_CATEGORIES
#define MY_VEC_STEP test_rek_g1_vec_step
#define MY_VEC_CLOSE
#define MY_SHARED
#define MY_SHARED_CLOSE
#define MY_GET
#define MY_PUT
#define Env TestVecEnv
#define Log TestVecLog
#include "vecenv.h"

typedef struct TestVecRuntime {
    int reset_calls;
    int advance_calls;
    RekG1SemanticTick captured[TEST_VEC_ENVS];
} TestVecRuntime;

static RekG1NativePufferVector native_vector;
static TestVecRuntime runtime;
static int assertions;

static void require(int condition, const char* name) {
    assertions += 1;
    if (!condition) {
        fprintf(stderr, "FAIL %s\n", name);
        exit(1);
    }
}

static RekG1RuntimeFacts test_facts(void) {
    return (RekG1RuntimeFacts){
        .timing = {.elapsed_seconds = 0.02f, .yaw_ramp_seconds = 0.10f},
        .translation_transition_settled = 1,
    };
}

static int test_reset_batch(
        void* context,
        const RekG1NativeMotionRouteTable* routes,
        const RekG1PufferActionTable* action_table,
        size_t environment_count,
        RekG1RuntimeFacts* facts_out,
        void* observations,
        size_t observation_stride_bytes,
        float* rewards,
        float* terminals,
        char* error,
        size_t error_capacity) {
    (void)error;
    (void)error_capacity;
    TestVecRuntime* state = (TestVecRuntime*)context;
    state->reset_calls += 1;
    require(rek_g1_native_validate_static_motion_routes(routes),
        "vec_reset_route_contract");
    require(action_table != NULL && action_table->count == TEST_VEC_CATEGORIES,
        "vec_reset_action_registry");
    require(environment_count == TEST_VEC_ENVS, "vec_reset_batch_size");
    for (size_t index = 0; index < environment_count; index++) {
        facts_out[index] = test_facts();
        float* row = (float*)((unsigned char*)observations
            + index * observation_stride_bytes);
        memset(row, 0, TEST_VEC_OBS * sizeof(float));
        rewards[index] = 0.0f;
        terminals[index] = 0.0f;
    }
    return 1;
}

static int test_advance_batch(
        void* context,
        const RekG1NativeMotionRouteTable* routes,
        const RekG1PufferActionTable* action_table,
        const RekG1SemanticTick* semantics,
        size_t environment_count,
        RekG1RuntimeFacts* next_facts_out,
        void* observations,
        size_t observation_stride_bytes,
        float* rewards,
        float* terminals,
        char* error,
        size_t error_capacity) {
    (void)error;
    (void)error_capacity;
    TestVecRuntime* state = (TestVecRuntime*)context;
    state->advance_calls += 1;
    require(rek_g1_native_validate_static_motion_routes(routes),
        "vec_advance_route_contract");
    require(action_table != NULL && action_table->count == TEST_VEC_CATEGORIES,
        "vec_advance_action_registry");
    require(environment_count == TEST_VEC_ENVS, "vec_advance_batch_size");
    memcpy(state->captured, semantics, sizeof(state->captured));
    for (size_t index = 0; index < environment_count; index++) {
        next_facts_out[index] = test_facts();
        float* row = (float*)((unsigned char*)observations
            + index * observation_stride_bytes);
        row[0] = (float)semantics[index].input.held;
        row[1] = (float)semantics[index].input.forward;
        row[2] = (float)semantics[index].input.strafe;
        row[3] = semantics[index].input.yaw;
        rewards[index] = (float)index + 0.25f;
        terminals[index] = 0.0f;
    }
    return 1;
}

static void test_rek_g1_vec_step(StaticVec* vec) {
    RekG1NativePufferIO io = {
        .actions = vec->actions,
        .action_rows = (size_t)vec->total_agents,
        .action_heads = NUM_ATNS,
        .observations = vec->observations,
        .observation_rows = (size_t)vec->total_agents,
        .observation_stride_bytes = OBS_SIZE * sizeof(float),
        .rewards = vec->rewards,
        .reward_rows = (size_t)vec->total_agents,
        .terminals = vec->terminals,
        .terminal_rows = (size_t)vec->total_agents,
        .action_masks = vec->action_mask,
        .action_mask_rows = (size_t)vec->total_agents,
        .action_mask_stride_bytes = MY_ACTION_MASK,
    };
    char error[128] = {0};
    RekG1NativePufferStatus status = rek_g1_native_puffer_step(
        &native_vector, io, error, sizeof(error));
    if (status != REK_G1_NATIVE_PUFFER_OK) {
        fprintf(stderr, "native G1 vector step failed: status=%d error=%s\n",
            (int)status, error);
        abort();
    }
}

void my_init(TestVecEnv* env, Dict* kwargs) {
    (void)kwargs;
    env->num_agents = 1;
}

void my_log(TestVecLog* log, Dict* out) {
    (void)log;
    (void)out;
}

void my_vec_close(TestVecEnv* envs) {
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

cudaError_t cudaHostAlloc(void** pointer, size_t bytes, unsigned int flags) {
    (void)flags;
    *pointer = malloc(bytes);
    return *pointer == NULL ? 1 : cudaSuccess;
}

cudaError_t cudaMalloc(void** pointer, size_t bytes) {
    *pointer = malloc(bytes);
    return *pointer == NULL ? 1 : cudaSuccess;
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

cudaError_t cudaFree(void* pointer) {
    free(pointer);
    return cudaSuccess;
}

cudaError_t cudaFreeHost(void* pointer) {
    free(pointer);
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

static uint8_t test_held_code(uint8_t held) {
    uint8_t code = 255;
    require(rek_g1_semantic_encode_held(held, &code) == REK_G1_SEMANTIC_OK,
        "vec_table_held_code");
    return code;
}

static RekG1PufferActionTable make_table(
        RekG1PufferCategory* categories,
        uint16_t* kick_indices,
        uint32_t* kick_durations) {
    static const uint8_t required_held[] = {
        0,
        REK_G1_HELD_FORWARD,
        REK_G1_HELD_BACKWARD,
        REK_G1_HELD_STRAFE_LEFT,
        REK_G1_HELD_STRAFE_RIGHT,
        REK_G1_HELD_YAW_LEFT,
        REK_G1_HELD_YAW_RIGHT,
        REK_G1_HELD_FORWARD | REK_G1_HELD_YAW_LEFT,
        REK_G1_HELD_FORWARD | REK_G1_HELD_YAW_RIGHT,
        REK_G1_HELD_BACKWARD | REK_G1_HELD_YAW_LEFT,
        REK_G1_HELD_BACKWARD | REK_G1_HELD_YAW_RIGHT,
        REK_G1_HELD_STRAFE_LEFT | REK_G1_HELD_YAW_LEFT,
        REK_G1_HELD_STRAFE_LEFT | REK_G1_HELD_YAW_RIGHT,
        REK_G1_HELD_STRAFE_RIGHT | REK_G1_HELD_YAW_LEFT,
        REK_G1_HELD_STRAFE_RIGHT | REK_G1_HELD_YAW_RIGHT,
    };
    memset(categories, 0, TEST_VEC_CATEGORIES * sizeof(*categories));
    categories[0].kind = REK_G1_PUFFER_CONTINUE;
    for (size_t index = 0; index < sizeof(required_held); index++) {
        categories[index + 1] = (RekG1PufferCategory){
            .kind = REK_G1_PUFFER_START,
            .command = {
                .kind = REK_G1_SEMANTIC_LOCOMOTION,
                .held_code = test_held_code(required_held[index]),
                .duration_ticks = 2,
                .kick_registry_index = REK_G1_SEMANTIC_KICK_NONE,
            },
        };
    }
    for (uint16_t kick = 0; kick < REK_G1_REQUIRED_KICK_COUNT; kick++) {
        kick_indices[kick] = (uint16_t)(6 + kick);
        kick_durations[kick] = 2;
        categories[16 + kick] = (RekG1PufferCategory){
            .kind = REK_G1_PUFFER_START,
            .command = {
                .kind = REK_G1_SEMANTIC_KICK,
                .held_code = test_held_code(0),
                .duration_ticks = 2,
                .kick_registry_index = kick,
            },
        };
    }
    return (RekG1PufferActionTable){
        .categories = categories,
        .count = TEST_VEC_CATEGORIES,
        .kick_move_indices = kick_indices,
        .kick_duration_ticks = kick_durations,
        .kick_registry_count = REK_G1_REQUIRED_KICK_COUNT,
    };
}

static void reset_native_vector(
        RekG1PufferActionTable* table,
        RekG1NativePufferIO io) {
    memset(&runtime, 0, sizeof(runtime));
    RekG1NativeBatchOps ops = {
        .runtime_facts_abi_version = REK_G1_RUNTIME_FACTS_ABI_VERSION,
        .runtime_facts_size = REK_G1_RUNTIME_FACTS_SIZE,
        .reset = test_reset_batch,
        .advance = test_advance_batch,
    };
    require(rek_g1_native_puffer_open(
        &native_vector,
        TEST_VEC_ENVS,
        table,
        rek_g1_native_static_motion_routes(),
        ops,
        &runtime) == REK_G1_NATIVE_PUFFER_OK,
        "vec_native_open");
    require(rek_g1_native_puffer_reset(
        &native_vector, io, NULL, 0) == REK_G1_NATIVE_PUFFER_OK,
        "vec_native_reset");
}

int main(void) {
    RekG1PufferCategory categories[TEST_VEC_CATEGORIES];
    uint16_t kick_indices[REK_G1_REQUIRED_KICK_COUNT];
    uint32_t kick_durations[REK_G1_REQUIRED_KICK_COUNT];
    RekG1PufferActionTable table = make_table(
        categories, kick_indices, kick_durations);

    TestVecEnv envs[TEST_VEC_ENVS] = {0};
    float observations[TEST_VEC_ENVS][TEST_VEC_OBS] = {{0}};
    float actions[TEST_VEC_ENVS] = {2.0f, 5.0f};
    float rewards[TEST_VEC_ENVS] = {0};
    float terminals[TEST_VEC_ENVS] = {0};
    uint8_t masks[TEST_VEC_ENVS][TEST_VEC_CATEGORIES] = {{0}};
    for (size_t index = 0; index < TEST_VEC_ENVS; index++) {
        envs[index].num_agents = 1;
    }
    StaticVec vec = {
        .envs = envs,
        .size = TEST_VEC_ENVS,
        .total_agents = TEST_VEC_ENVS,
        .buffers = 1,
        .observations = observations,
        .actions = actions,
        .rewards = rewards,
        .terminals = terminals,
        .action_mask = &masks[0][0],
    };
    RekG1NativePufferIO io = {
        .actions = actions,
        .action_rows = TEST_VEC_ENVS,
        .action_heads = NUM_ATNS,
        .observations = observations,
        .observation_rows = TEST_VEC_ENVS,
        .observation_stride_bytes = sizeof(observations[0]),
        .rewards = rewards,
        .reward_rows = TEST_VEC_ENVS,
        .terminals = terminals,
        .terminal_rows = TEST_VEC_ENVS,
        .action_masks = &masks[0][0],
        .action_mask_rows = TEST_VEC_ENVS,
        .action_mask_stride_bytes = sizeof(masks[0]),
    };
    reset_native_vector(&table, io);
    cpu_vec_step(&vec);
    require(runtime.advance_calls == 1,
        "cpu_puffer_step_invokes_one_native_batch");
    require(runtime.captured[0].input.held == REK_G1_HELD_FORWARD,
        "cpu_puffer_forward_preserved");
    require(runtime.captured[1].input.held == REK_G1_HELD_STRAFE_RIGHT,
        "cpu_puffer_strafe_preserved");
    require(envs[0].fallback_steps == 0 && envs[1].fallback_steps == 0,
        "cpu_puffer_skips_per_env_c_step");

    rek_g1_native_puffer_close(&native_vector);
    float device_observations[TEST_VEC_ENVS][TEST_VEC_OBS] = {{0}};
    float device_actions[TEST_VEC_ENVS] = {3.0f, 7.0f};
    float device_rewards[TEST_VEC_ENVS] = {0};
    float device_terminals[TEST_VEC_ENVS] = {0};
    uint8_t device_masks[TEST_VEC_ENVS][TEST_VEC_CATEGORIES] = {{0}};
    memset(actions, 0, sizeof(actions));
    vec.gpu = 1;
    vec.gpu_observations = device_observations;
    vec.gpu_actions = device_actions;
    vec.gpu_rewards = device_rewards;
    vec.gpu_terminals = device_terminals;
    vec.gpu_action_mask = &device_masks[0][0];
    reset_native_vector(&table, io);
    gpu_vec_step(&vec);
    require(runtime.advance_calls == 1,
        "gpu_puffer_step_invokes_one_native_batch");
    require(runtime.captured[0].input.held == REK_G1_HELD_BACKWARD,
        "gpu_action_copy_precedes_native_batch");
    require(runtime.captured[1].input.held == REK_G1_HELD_YAW_RIGHT,
        "gpu_second_action_reaches_native_batch");
    require(device_observations[0][1] == -1.0f,
        "gpu_observation_copy_follows_native_batch");
    require(device_rewards[1] == 1.25f,
        "gpu_reward_copy_follows_native_batch");
    require(device_masks[0][0] == 1 && device_masks[1][0] == 1,
        "gpu_mask_copy_follows_native_batch");
    require(envs[0].fallback_steps == 0 && envs[1].fallback_steps == 0,
        "gpu_puffer_skips_per_env_c_step");

    rek_g1_native_puffer_close(&native_vector);
    printf("native puffer vecenv tests passed: %d assertions\n", assertions);
    return 0;
}
