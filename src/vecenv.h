// vecenv.h - Static vectorized env implementation.

#pragma once

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef __CUDACC__
#include <cuda_runtime_api.h>
#endif

#include "dict.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef ENV_HEADER
#define PUFFER_VECENV_INCLUDE
#include ENV_HEADER
#undef PUFFER_VECENV_INCLUDE
#endif

typedef struct CUstream_st* cudaStream_t;
typedef struct StaticThreading StaticThreading;

#ifdef OBS_SIZE
typedef Env StaticEnv;
#else
typedef void StaticEnv;
#endif

#ifdef OBS_TENSOR_T
typedef OBS_TENSOR_T StaticObsTensor;
#else
typedef void* StaticObsTensor;
#endif

typedef struct StaticVec {
    StaticEnv* envs;
    int size;
    int total_agents;
    int buffers;
    int agents_per_buffer;
    int* buffer_env_starts;
    int* buffer_env_counts;
    StaticObsTensor observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned char* action_mask;  // NULL unless env defines MY_ACTION_MASK
    StaticObsTensor gpu_observations;
    float* gpu_actions;
    float* gpu_rewards;
    float* gpu_terminals;
    unsigned char* gpu_action_mask;  // NULL unless env defines MY_ACTION_MASK
    cudaStream_t* streams;
    StaticThreading* threading;
    int action_mask_size;        // 0 unless env defines MY_ACTION_MASK
    int* agent_perm;
} StaticVec;

typedef void (*net_callback_fn)(void* ctx, int buf, int t);
typedef void (*thread_init_fn)(void* ctx, int buf);

enum EvalProfileIdx {
    EVAL_GPU = 0,   // forward + D2H (everything before env step)
    EVAL_ENV_STEP,  // OMP c_step (pure CPU)
    NUM_EVAL_PROF,
};

StaticVec* create_static_vec(int total_agents, int num_buffers, Dict* vec_kwargs, Dict* env_kwargs);
void static_vec_reset(StaticVec* vec);
void static_vec_close(StaticVec* vec);
void static_vec_log(StaticVec* vec, Dict* out);
void static_vec_eval_log(StaticVec* vec, Dict* out);
void create_static_threads(StaticVec* vec, int num_threads, int horizon,
    void* ctx, net_callback_fn net_callback, thread_init_fn thread_init);
void static_vec_omp_step(StaticVec* vec);
void static_vec_render(StaticVec* vec, int env_id);
void static_vec_read_profile(StaticVec* vec, float out[NUM_EVAL_PROF]);

void static_vec_set_perm(StaticVec* vec, const int* perm);
void static_vec_set_env_tags(StaticVec* vec, const int* tags);
int static_vec_count_aligned(StaticVec* vec, int tag_value, int reset_flags);

// Optional shared state functions
void* my_shared(void* env, Dict* kwargs);
void my_shared_close(void* env);
void* my_get(void* env, Dict* out);
int my_put(void* env, Dict* kwargs);

#ifdef __cplusplus
}
#endif

#ifdef OBS_SIZE

static inline size_t obs_element_size(void) {
    OBS_TENSOR_T t;
    return sizeof(*t.data);
}

static inline void static_obs_set(StaticObsTensor* obs, void* data, int total_agents) {
#ifdef __cplusplus
    obs->data = (decltype(obs->data))data;
#else
    obs->data = data;
#endif
    memset(obs->shape, 0, sizeof(obs->shape));
    obs->shape[0] = total_agents;
    obs->shape[1] = OBS_SIZE;
}

#include <omp.h>
typedef int atomic_int;
static inline int atomic_load(const atomic_int* ptr) {
    return __atomic_load_n(ptr, __ATOMIC_SEQ_CST);
}
static inline void atomic_store(atomic_int* ptr, int value) {
    __atomic_store_n(ptr, value, __ATOMIC_SEQ_CST);
}
#include <pthread.h>
#include <stdbool.h>
#include <time.h>

#define OMP_WAITING 5
#define OMP_RUNNING 6

void my_init(Env* env, Dict* kwargs);
void my_log(Log* log, Dict* out);

#ifdef MY_USES_PERM
void my_setup_perm(StaticVec* vec, Env* env, int slot_base);
#endif


struct StaticThreading {
    atomic_int* buffer_states;
    atomic_int shutdown;
    int num_threads;
    int num_buffers;
    pthread_t* threads;
    float* accum;  // [num_buffers * NUM_EVAL_PROF] per-buffer timing in ms
};

typedef struct StaticOMPArg {
    StaticVec* vec;
    int buf;
    int horizon;
    void* ctx;
    net_callback_fn net_callback;
    thread_init_fn thread_init;
} StaticOMPArg;

static void* static_omp_threadmanager(void* arg) {
    StaticOMPArg* worker_arg = (StaticOMPArg*)arg;
    StaticVec* vec = worker_arg->vec;
    StaticThreading* threading = vec->threading;
    int buf = worker_arg->buf;
    int horizon = worker_arg->horizon;
    void* ctx = worker_arg->ctx;
    net_callback_fn net_callback = worker_arg->net_callback;
    thread_init_fn thread_init = worker_arg->thread_init;

    if (thread_init != NULL) {
        thread_init(ctx, buf);
    }

    int agents_per_buffer = vec->agents_per_buffer;
    int agent_start = buf * agents_per_buffer;
    int env_start = vec->buffer_env_starts[buf];
    int env_count = vec->buffer_env_counts[buf];
    atomic_int* buffer_states = threading->buffer_states;
    int num_workers = threading->num_threads / vec->buffers;
    if (num_workers < 1) {
        num_workers = 1;
    }

    Env* envs = (Env*)vec->envs;

    printf("Num workers: %d\n", num_workers);
    while (true) {
        while (atomic_load(&buffer_states[buf]) != OMP_RUNNING) {
            if (atomic_load(&threading->shutdown)) {
                return NULL;
            }
        }
        cudaStream_t stream = vec->streams[buf];

        float* my_accum = &threading->accum[buf * NUM_EVAL_PROF];
        struct timespec t0, t1;

        for (int t = 0; t < horizon; t++) {
            clock_gettime(CLOCK_MONOTONIC, &t0);
            net_callback(ctx, buf, t);

            cudaMemcpyAsync(
                &vec->actions[agent_start * NUM_ATNS],
                &vec->gpu_actions[agent_start * NUM_ATNS],
                agents_per_buffer * NUM_ATNS * sizeof(float),
                cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            my_accum[EVAL_GPU] += (t1.tv_sec - t0.tv_sec) * 1000.0f + (t1.tv_nsec - t0.tv_nsec) / 1e6f;

            memset(&vec->rewards[agent_start], 0, agents_per_buffer * sizeof(float));
            memset(&vec->terminals[agent_start], 0, agents_per_buffer * sizeof(float));
            clock_gettime(CLOCK_MONOTONIC, &t0);
            #pragma omp parallel for schedule(static) num_threads(num_workers)
            for (int i = env_start; i < env_start + env_count; i++) {
                c_step(&envs[i]);
            }
            clock_gettime(CLOCK_MONOTONIC, &t1);
            my_accum[EVAL_ENV_STEP] += (t1.tv_sec - t0.tv_sec) * 1000.0f + (t1.tv_nsec - t0.tv_nsec) / 1e6f;

            cudaMemcpyAsync(
                (char*)vec->gpu_observations.data + agent_start * OBS_SIZE * obs_element_size(),
                (char*)vec->observations.data + agent_start * OBS_SIZE * obs_element_size(),
                agents_per_buffer * OBS_SIZE * obs_element_size(),
                cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(
                &vec->gpu_rewards[agent_start],
                &vec->rewards[agent_start],
                agents_per_buffer * sizeof(float),
                cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(
                &vec->gpu_terminals[agent_start],
                &vec->terminals[agent_start],
                agents_per_buffer * sizeof(float),
                cudaMemcpyHostToDevice, stream);
#ifdef MY_ACTION_MASK
            cudaMemcpyAsync(
                vec->gpu_action_mask + agent_start * MY_ACTION_MASK,
                vec->action_mask     + agent_start * MY_ACTION_MASK,
                agents_per_buffer * MY_ACTION_MASK * sizeof(unsigned char),
                cudaMemcpyHostToDevice, stream);
#endif
        }
        cudaStreamSynchronize(stream);
        atomic_store(&buffer_states[buf], OMP_WAITING);
    }
}

void static_vec_omp_step(StaticVec* vec) {
    StaticThreading* threading = vec->threading;
    for (int buf = 0; buf < vec->buffers; buf++) {
        atomic_store(&threading->buffer_states[buf], OMP_RUNNING);
    }
    for (int buf = 0; buf < vec->buffers; buf++) {
        while (atomic_load(&threading->buffer_states[buf]) != OMP_WAITING) {}
    }
}

#ifdef MY_VEC_INIT
Env* my_vec_init(int* num_envs_out, int* buffer_env_starts, int* buffer_env_counts,
                 Dict* vec_kwargs, Dict* env_kwargs);
#else
Env* my_vec_init(int* num_envs_out, int* buffer_env_starts, int* buffer_env_counts,
                 Dict* vec_kwargs, Dict* env_kwargs) {

    int total_agents = (int)dict_get(vec_kwargs, "total_agents")->value;
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers")->value;
    int agents_per_buffer = total_agents / num_buffers;

    Env* envs = (Env*)calloc(total_agents, sizeof(Env));

    int num_envs = 0;
    int agents_created = 0;
    while (agents_created < total_agents) {
        srand(num_envs);
        envs[num_envs].rng = num_envs;
        my_init(&envs[num_envs], env_kwargs);
        agents_created += envs[num_envs].num_agents;
        num_envs++;
    }

    envs = (Env*)realloc(envs, num_envs * sizeof(Env));

    int buf = 0;
    int buf_agents = 0;
    buffer_env_starts[0] = 0;
    buffer_env_counts[0] = 0;
    for (int i = 0; i < num_envs; i++) {
        buf_agents += envs[i].num_agents;
        buffer_env_counts[buf]++;
        if (buf_agents >= agents_per_buffer && buf < num_buffers - 1) {
            buf++;
            buffer_env_starts[buf] = i + 1;
            buffer_env_counts[buf] = 0;
            buf_agents = 0;
        }
    }

    *num_envs_out = num_envs;
    return envs;
}
#endif

#ifdef MY_VEC_CLOSE
void my_vec_close(Env* envs);
#else
void my_vec_close(Env* envs) {
    return;
}
#endif

StaticVec* create_static_vec(int total_agents, int num_buffers, Dict* vec_kwargs, Dict* env_kwargs) {
    StaticVec* vec = (StaticVec*)calloc(1, sizeof(StaticVec));
    vec->total_agents = total_agents;
    vec->buffers = num_buffers;
    vec->agents_per_buffer = total_agents / num_buffers;

    vec->buffer_env_starts = (int*)calloc(num_buffers, sizeof(int));
    vec->buffer_env_counts = (int*)calloc(num_buffers, sizeof(int));

    int num_envs = 0;
    vec->envs = my_vec_init(&num_envs, vec->buffer_env_starts, vec->buffer_env_counts,
                            vec_kwargs, env_kwargs);
    vec->size = num_envs;

    size_t obs_elem_size = obs_element_size();
    void* observations = NULL;
    void* gpu_observations = NULL;
    cudaHostAlloc(&observations, total_agents * OBS_SIZE * obs_elem_size, cudaHostAllocPortable);
    cudaHostAlloc((void**)&vec->actions, total_agents * NUM_ATNS * sizeof(float), cudaHostAllocPortable);
    cudaHostAlloc((void**)&vec->rewards, total_agents * sizeof(float), cudaHostAllocPortable);
    cudaHostAlloc((void**)&vec->terminals, total_agents * sizeof(float), cudaHostAllocPortable);

    cudaMalloc(&gpu_observations, total_agents * OBS_SIZE * obs_elem_size);
    cudaMalloc((void**)&vec->gpu_actions, total_agents * NUM_ATNS * sizeof(float));
    cudaMalloc((void**)&vec->gpu_rewards, total_agents * sizeof(float));
    cudaMalloc((void**)&vec->gpu_terminals, total_agents * sizeof(float));

    static_obs_set(&vec->observations, observations, total_agents);
    static_obs_set(&vec->gpu_observations, gpu_observations, total_agents);

    cudaMemset(vec->gpu_observations.data, 0, total_agents * OBS_SIZE * obs_elem_size);
    cudaMemset(vec->gpu_actions, 0, total_agents * NUM_ATNS * sizeof(float));
    cudaMemset(vec->gpu_rewards, 0, total_agents * sizeof(float));
    cudaMemset(vec->gpu_terminals, 0, total_agents * sizeof(float));

#ifdef MY_ACTION_MASK
    vec->action_mask_size = MY_ACTION_MASK;
    size_t mask_bytes = (size_t)total_agents * MY_ACTION_MASK * sizeof(unsigned char);
    cudaHostAlloc((void**)&vec->action_mask, mask_bytes, cudaHostAllocPortable);
    cudaMalloc((void**)&vec->gpu_action_mask, mask_bytes);
    cudaMemset(vec->gpu_action_mask, 0, mask_bytes);
#endif
    vec->streams = (cudaStream_t*)calloc(num_buffers, sizeof(cudaStream_t));

    Env* envs = (Env*)vec->envs;
    for (int buf = 0; buf < num_buffers; buf++) {
        int buf_start = buf * vec->agents_per_buffer;
        int buf_agent = 0;
        int env_start = vec->buffer_env_starts[buf];
        int env_count = vec->buffer_env_counts[buf];

        for (int e = 0; e < env_count; e++) {
            Env* env = &envs[env_start + e];
            int slot = buf_start + buf_agent;
            env->observations = vec->observations.data + slot * OBS_SIZE;
            env->actions = vec->actions + slot * NUM_ATNS;
            env->rewards = vec->rewards + slot;
            env->terminals = vec->terminals + slot;
#ifdef MY_ACTION_MASK
            env->action_mask = vec->action_mask + slot * MY_ACTION_MASK;
#endif
#ifdef MY_USES_PERM
            my_setup_perm(vec, env, slot);
#endif
            buf_agent += env->num_agents;
        }
    }

    return vec;
}

void static_vec_set_perm(StaticVec* vec, const int* perm) {
#ifndef MY_USES_PERM
    (void)vec; (void)perm;
    fprintf(stderr, "static_vec_set_perm: env did not opt in via MY_USES_PERM; ignoring.\n");
    return;
#else
    int N = vec->total_agents;
    if (vec->agent_perm == NULL) {
        vec->agent_perm = (int*)malloc(N * sizeof(int));
    }
    memcpy(vec->agent_perm, perm, N * sizeof(int));

    Env* envs = (Env*)vec->envs;
    for (int buf = 0; buf < vec->buffers; buf++) {
        int buf_start = buf * vec->agents_per_buffer;
        int buf_agent = 0;
        int env_start = vec->buffer_env_starts[buf];
        int env_count = vec->buffer_env_counts[buf];
        for (int e = 0; e < env_count; e++) {
            Env* env = &envs[env_start + e];
            my_setup_perm(vec, env, buf_start + buf_agent);
            buf_agent += env->num_agents;
        }
    }
#endif
}

#ifdef MY_USES_TAGS
void static_vec_set_env_tags(StaticVec* vec, const int* tags) {
    Env* envs = (Env*)vec->envs;
    for (int i = 0; i < vec->size; i++) {
        envs[i].tag = tags[i];
        envs[i].boundary_reached = 0;
    }
}

int static_vec_count_aligned(StaticVec* vec, int tag_value, int reset_flags) {
    Env* envs = (Env*)vec->envs;
    int count = 0;
    for (int i = 0; i < vec->size; i++) {
        if (envs[i].tag == tag_value && envs[i].boundary_reached) {
            count++;
        }
    }
    if (reset_flags) {
        for (int i = 0; i < vec->size; i++) {
            if (envs[i].tag == tag_value) {
                envs[i].boundary_reached = 0;
            }
        }
    }
    return count;
}
#else
void static_vec_set_env_tags(StaticVec* vec, const int* tags) {
    (void)vec; (void)tags;
    fprintf(stderr, "static_vec_set_env_tags: env did not opt in via MY_USES_TAGS; ignoring.\n");
}
int static_vec_count_aligned(StaticVec* vec, int tag_value, int reset_flags) {
    (void)vec; (void)tag_value; (void)reset_flags;
    return 0;
}
#endif

void static_vec_reset(StaticVec* vec) {
    Env* envs = (Env*)vec->envs;
    for (int i = 0; i < vec->size; i++) {
        c_reset(&envs[i]);
    }
    cudaMemcpy(vec->gpu_observations.data, vec->observations.data,
        vec->total_agents * OBS_SIZE * obs_element_size(), cudaMemcpyHostToDevice);
    cudaMemset(vec->gpu_rewards,   0, vec->total_agents * sizeof(float));
    cudaMemset(vec->gpu_terminals, 0, vec->total_agents * sizeof(float));
#ifdef MY_ACTION_MASK
    cudaMemcpy(vec->gpu_action_mask, vec->action_mask,
        (size_t)vec->total_agents * MY_ACTION_MASK * sizeof(unsigned char),
        cudaMemcpyHostToDevice);
#endif
    cudaDeviceSynchronize();
}

void create_static_threads(StaticVec* vec, int num_threads, int horizon,
        void* ctx, net_callback_fn net_callback, thread_init_fn thread_init) {
    vec->threading = (StaticThreading*)calloc(1, sizeof(StaticThreading));
    vec->threading->num_threads = num_threads;
    vec->threading->num_buffers = vec->buffers;
    vec->threading->buffer_states = (atomic_int*)calloc(vec->buffers, sizeof(atomic_int));
    vec->threading->threads = (pthread_t*)calloc(vec->buffers, sizeof(pthread_t));
    vec->threading->accum = (float*)calloc(vec->buffers * NUM_EVAL_PROF, sizeof(float));

    StaticOMPArg* args = (StaticOMPArg*)calloc(vec->buffers, sizeof(StaticOMPArg));
    for (int i = 0; i < vec->buffers; i++) {
        args[i].vec = vec;
        args[i].buf = i;
        args[i].horizon = horizon;
        args[i].ctx = ctx;
        args[i].net_callback = net_callback;
        args[i].thread_init = thread_init;
        pthread_create(&vec->threading->threads[i], NULL, static_omp_threadmanager, &args[i]);
    }
}

void static_vec_close(StaticVec* vec) {
    Env* envs = (Env*)vec->envs;

    if (vec->threading != NULL) {
        atomic_store(&vec->threading->shutdown, 1);
        for (int i = 0; i < vec->buffers; i++) {
            pthread_join(vec->threading->threads[i], NULL);
        }
    }

    for (int i = 0; i < vec->size; i++) {
        Env* env = &envs[i];
        c_close(env);
    }

    my_vec_close(envs);
    free(vec->envs);
    if (vec->threading != NULL) {
        free(vec->threading->buffer_states);
        free(vec->threading->threads);
        free(vec->threading->accum);
        free(vec->threading);
    }
    free(vec->buffer_env_starts);
    free(vec->buffer_env_counts);

    cudaDeviceSynchronize();
    cudaFree(vec->gpu_observations.data);
    cudaFree(vec->gpu_actions);
    cudaFree(vec->gpu_rewards);
    cudaFree(vec->gpu_terminals);
    cudaFreeHost(vec->observations.data);
    cudaFreeHost(vec->actions);
    cudaFreeHost(vec->rewards);
    cudaFreeHost(vec->terminals);
#ifdef MY_ACTION_MASK
    cudaFree(vec->gpu_action_mask);
    cudaFreeHost(vec->action_mask);
#endif

    free(vec->streams);
    if (vec->agent_perm != NULL) {
        free(vec->agent_perm);
    }
    free(vec);
}

static inline float static_vec_aggregate_logs(StaticVec* vec, Log* out) {
    Env* envs = (Env*)vec->envs;
    memset(out, 0, sizeof(Log));
    int num_keys = sizeof(Log) / sizeof(float);
    for (int i = 0; i < vec->size; i++) {
        Env* env = &envs[i];
        if (env->log.n == 0) {
            continue;
        }
        for (int j = 0; j < num_keys; j++) {
            ((float*)out)[j] += ((float*)&env->log)[j];
        }
    }
    float n = out->n;
    if (n == 0.0f) {
        return 0;
    }
    for (int i = 0; i < num_keys; i++) {
        ((float*)out)[i] /= n;
    }
    return n;
}

void static_vec_log(StaticVec* vec, Dict* out) {
    Env* envs = (Env*)vec->envs;
    Log aggregate;
    float n = static_vec_aggregate_logs(vec, &aggregate);
    if (n == 0) {
        return;
    }
    for (int i = 0; i < vec->size; i++) {
        memset(&envs[i].log, 0, sizeof(Log));
    }
    my_log(&aggregate, out);
    dict_set(out, "n", n);
}

void static_vec_eval_log(StaticVec* vec, Dict* out) {
    Log aggregate;
    float n = static_vec_aggregate_logs(vec, &aggregate);
    if (n == 0) {
        return;
    }
    my_log(&aggregate, out);
    dict_set(out, "n", n);
}

void static_vec_read_profile(StaticVec* vec, float out[NUM_EVAL_PROF]) {
    StaticThreading* threading = vec->threading;
    memset(out, 0, NUM_EVAL_PROF * sizeof(float));
    for (int buf = 0; buf < threading->num_buffers; buf++) {
        float* src = &threading->accum[buf * NUM_EVAL_PROF];
        for (int i = 0; i < NUM_EVAL_PROF; i++) {
            out[i] += src[i];
        }
        memset(src, 0, NUM_EVAL_PROF * sizeof(float));
    }
    for (int i = 0; i < NUM_EVAL_PROF; i++) {
        out[i] /= threading->num_buffers;
    }
}

void static_vec_render(StaticVec* vec, int env_id) {
    Env* envs = (Env*)vec->envs;
    c_render(&envs[env_id]);
}

#ifndef MY_SHARED
void* my_shared(void* env, Dict* kwargs) {
    return NULL;
}
#endif

#ifndef MY_SHARED_CLOSE
void my_shared_close(void* env) {}
#endif

#ifndef MY_GET
void* my_get(void* env, Dict* out) {
    return NULL;
}
#endif

#ifndef MY_PUT
int my_put(void* env, Dict* kwargs) {
    return 0;
}
#endif

#endif // OBS_SIZE
