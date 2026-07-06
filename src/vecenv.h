#pragma once

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <omp.h>
#include <pthread.h>

#include "dict.h"

#define PUFFER_VECENV_INCLUDE
#include ENV_HEADER
#undef PUFFER_VECENV_INCLUDE

typedef int atomic_int;

struct PuffeRL;
void pufferl_thread_init(struct PuffeRL* pufferl, int buf);
void pufferl_forward(struct PuffeRL* pufferl, int buf, int t);

typedef struct ObsTensor {
    obs_t* data;
    int64_t shape[8];
} ObsTensor;

static inline int atomic_load(const atomic_int* ptr) {
    return __atomic_load_n(ptr, __ATOMIC_SEQ_CST);
}

static inline void atomic_store(atomic_int* ptr, int value) {
    __atomic_store_n(ptr, value, __ATOMIC_SEQ_CST);
}

enum VecProfileIdx {
    VEC_GPU = 0,
    VEC_ENV_STEP,
    NUM_VEC_PROF,
};

#define OMP_WAITING 5
#define OMP_RUNNING 6

typedef struct VecWorker VecWorker;

typedef struct VecEnv {
    Env* envs;
    int size;
    int total_agents;
    int buffers;
    int agents_per_buffer;
    int* buffer_env_starts;
    int* buffer_env_counts;
    obs_t* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned char* action_mask;
    obs_t* gpu_observations;
    float* gpu_actions;
    float* gpu_rewards;
    float* gpu_terminals;
    unsigned char* gpu_action_mask;
    cudaStream_t* streams;
    atomic_int* buffer_states;
    atomic_int shutdown;
    pthread_t* threads;
    VecWorker* workers;
    float* accum;
    int num_workers;
    int action_mask_size;
    int num_banks;
    int* bank_layout;
} VecEnv;

struct VecWorker {
    VecEnv* vec;
    int buf;
    int horizon;
    struct PuffeRL* pufferl;
};

static void* vec_thread_main(void* arg) {
    VecWorker* worker_arg = (VecWorker*)arg;
    VecEnv* vec = worker_arg->vec;
    int buf = worker_arg->buf;
    int horizon = worker_arg->horizon;
    struct PuffeRL* pufferl = worker_arg->pufferl;

    pufferl_thread_init(pufferl, buf);

    int agents_per_buffer = vec->agents_per_buffer;
    int agent_start = buf * agents_per_buffer;
    int env_start = vec->buffer_env_starts[buf];
    int env_count = vec->buffer_env_counts[buf];

    Env* envs = vec->envs;

    while (true) {
        while (atomic_load(&vec->buffer_states[buf]) != OMP_RUNNING) {
            if (atomic_load(&vec->shutdown)) {
                return NULL;
            }
        }
        cudaStream_t stream = vec->streams[buf];

        float* my_accum = &vec->accum[buf * NUM_VEC_PROF];
        struct timespec t0, t1;

        for (int t = 0; t < horizon; t++) {
            clock_gettime(CLOCK_MONOTONIC, &t0);
            pufferl_forward(pufferl, buf, t);

            cudaMemcpyAsync(
                &vec->actions[agent_start * NUM_ATNS],
                &vec->gpu_actions[agent_start * NUM_ATNS],
                agents_per_buffer * NUM_ATNS * sizeof(float),
                cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            my_accum[VEC_GPU] += (t1.tv_sec - t0.tv_sec) * 1000.0f + (t1.tv_nsec - t0.tv_nsec) / 1e6f;

            memset(&vec->rewards[agent_start], 0, agents_per_buffer * sizeof(float));
            memset(&vec->terminals[agent_start], 0, agents_per_buffer * sizeof(float));
            clock_gettime(CLOCK_MONOTONIC, &t0);
            #pragma omp parallel for schedule(static) num_threads(vec->num_workers)
            for (int i = env_start; i < env_start + env_count; i++) {
                puf_step(&envs[i]);
            }
            clock_gettime(CLOCK_MONOTONIC, &t1);
            my_accum[VEC_ENV_STEP] += (t1.tv_sec - t0.tv_sec) * 1000.0f + (t1.tv_nsec - t0.tv_nsec) / 1e6f;

            cudaMemcpyAsync(
                vec->gpu_observations + (size_t)agent_start * OBS_SIZE,
                vec->observations + (size_t)agent_start * OBS_SIZE,
                (size_t)agents_per_buffer * OBS_SIZE * sizeof(obs_t),
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
            if (vec->action_mask_size > 0) {
                cudaMemcpyAsync(
                    vec->gpu_action_mask + agent_start * vec->action_mask_size,
                    vec->action_mask     + agent_start * vec->action_mask_size,
                    (size_t)agents_per_buffer * vec->action_mask_size * sizeof(unsigned char),
                    cudaMemcpyHostToDevice, stream);
            }
        }
        cudaStreamSynchronize(stream);
        atomic_store(&vec->buffer_states[buf], OMP_WAITING);
    }
}

void vec_step(VecEnv* vec) {
    for (int buf = 0; buf < vec->buffers; buf++) {
        atomic_store(&vec->buffer_states[buf], OMP_RUNNING);
    }
    for (int buf = 0; buf < vec->buffers; buf++) {
        while (atomic_load(&vec->buffer_states[buf]) != OMP_WAITING) {}
    }
}

Env* vec_init_envs(int* num_envs_out, int* buffer_env_starts, int* buffer_env_counts,
                 Dict* vec_kwargs, Dict* env_kwargs) {

    int total_agents = (int)dict_get(vec_kwargs, "total_agents");
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers");
    int agents_per_buffer = total_agents / num_buffers;

    Env* envs = (Env*)calloc(total_agents, sizeof(Env));

    int num_envs = 0;
    int agents_created = 0;
    while (agents_created < total_agents) {
        envs[num_envs].rng = num_envs;
        puf_init(&envs[num_envs], env_kwargs);
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

VecEnv* vec_create(Dict* vec_kwargs, Dict* env_kwargs) {
    int total_agents = (int)dict_get(vec_kwargs, "total_agents");
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers");
    VecEnv* vec = (VecEnv*)calloc(1, sizeof(VecEnv));
    vec->total_agents = total_agents;
    vec->buffers = num_buffers;
    vec->agents_per_buffer = total_agents / num_buffers;
    vec->num_workers = (int)dict_get(vec_kwargs, "num_threads") / num_buffers;
    if (vec->num_workers < 1) {
        vec->num_workers = 1;
    }
    int frozen_banks = (int)dict_get(vec_kwargs, "num_frozen_banks");
    vec->num_banks = frozen_banks + 1;
    vec->bank_layout = (int*)calloc(vec->num_banks + 1, sizeof(int));

    vec->buffer_env_starts = (int*)calloc(num_buffers, sizeof(int));
    vec->buffer_env_counts = (int*)calloc(num_buffers, sizeof(int));

    int num_envs = 0;
    vec->envs = vec_init_envs(&num_envs, vec->buffer_env_starts, vec->buffer_env_counts,
                            vec_kwargs, env_kwargs);
    vec->size = num_envs;

    obs_t* observations = NULL;
    obs_t* gpu_observations = NULL;
    cudaHostAlloc((void**)&observations,
        (size_t)total_agents * OBS_SIZE * sizeof(obs_t), cudaHostAllocPortable);
    cudaHostAlloc((void**)&vec->actions, total_agents * NUM_ATNS * sizeof(float), cudaHostAllocPortable);
    cudaHostAlloc((void**)&vec->rewards, total_agents * sizeof(float), cudaHostAllocPortable);
    cudaHostAlloc((void**)&vec->terminals, total_agents * sizeof(float), cudaHostAllocPortable);

    cudaMalloc((void**)&gpu_observations,
        (size_t)total_agents * OBS_SIZE * sizeof(obs_t));
    cudaMalloc((void**)&vec->gpu_actions, total_agents * NUM_ATNS * sizeof(float));
    cudaMalloc((void**)&vec->gpu_rewards, total_agents * sizeof(float));
    cudaMalloc((void**)&vec->gpu_terminals, total_agents * sizeof(float));

    vec->observations = observations;
    vec->gpu_observations = gpu_observations;

    cudaMemset(vec->gpu_observations, 0,
        (size_t)total_agents * OBS_SIZE * sizeof(obs_t));
    cudaMemset(vec->gpu_actions, 0, total_agents * NUM_ATNS * sizeof(float));
    cudaMemset(vec->gpu_rewards, 0, total_agents * sizeof(float));
    cudaMemset(vec->gpu_terminals, 0, total_agents * sizeof(float));

    vec->action_mask_size = (int)dict_get(vec_kwargs, "action_mask_size");
    if (vec->action_mask_size > 0) {
        size_t mask_bytes = (size_t)total_agents * vec->action_mask_size * sizeof(unsigned char);
        cudaHostAlloc((void**)&vec->action_mask, mask_bytes, cudaHostAllocPortable);
        cudaMalloc((void**)&vec->gpu_action_mask, mask_bytes);
        cudaMemset(vec->gpu_action_mask, 0, mask_bytes);
    }
    vec->streams = (cudaStream_t*)calloc(num_buffers, sizeof(cudaStream_t));

    Env* envs = vec->envs;
    for (int buf = 0; buf < num_buffers; buf++) {
        int buf_start = buf * vec->agents_per_buffer;
        int env_start = vec->buffer_env_starts[buf];
        int env_count = vec->buffer_env_counts[buf];

        float frozen_pct = (float)dict_get(vec_kwargs, "frozen_bank_pct");
        int frozen_envs = (int)(frozen_pct * env_count);
        int frozen_start = env_count - frozen_envs;
        if (frozen_banks == 0 || frozen_pct <= 0.0f) {
            frozen_start = env_count;
        }

        int* counts = (int*)calloc(vec->num_banks, sizeof(int));
        for (int e = 0; e < env_count; e++) {
            Env* env = &envs[env_start + e];
            for (int s = 0; s < env->num_agents; s++) {
                int policy = e < frozen_start ? 0 : env->agents[s].policy;
                if (policy < 0 || policy >= vec->num_banks) {
                    fprintf(stderr, "Agent policy %d outside bank range [0, %d)\n",
                        policy, vec->num_banks);
                    exit(1);
                }
                counts[policy]++;
            }
        }

        int offset = 0;
        for (int b = 0; b < vec->num_banks; b++) {
            if (buf == 0) {
                vec->bank_layout[b] = offset;
            } else if (vec->bank_layout[b] != offset) {
                fprintf(stderr, "Bank layout must match across buffers\n");
                exit(1);
            }
            offset += counts[b];
        }
        if (offset != vec->agents_per_buffer) {
            fprintf(stderr, "Buffer has %d agents, expected %d\n",
                offset, vec->agents_per_buffer);
            exit(1);
        }
        if (buf == 0) {
            vec->bank_layout[vec->num_banks] = offset;
        } else if (vec->bank_layout[vec->num_banks] != offset) {
            fprintf(stderr, "Bank layout must match across buffers\n");
            exit(1);
        }

        int* cursors = (int*)calloc(vec->num_banks, sizeof(int));
        for (int b = 0; b < vec->num_banks; b++) {
            cursors[b] = buf_start + vec->bank_layout[b];
        }
        for (int e = 0; e < env_count; e++) {
            Env* env = &envs[env_start + e];
            int tag = 0;
            for (int s = 0; s < env->num_agents; s++) {
                int policy = e < frozen_start ? 0 : env->agents[s].policy;
                if (policy > tag) {
                    tag = policy;
                }
                int phys = cursors[policy];
                env->agents[s].observations = vec->observations + (size_t)phys * OBS_SIZE;
                env->agents[s].actions = vec->actions + (size_t)phys * NUM_ATNS;
                env->agents[s].rewards = vec->rewards + phys;
                env->agents[s].terminals = vec->terminals + phys;
                if (vec->action_mask_size > 0) {
                    env->agents[s].action_mask =
                        vec->action_mask + (size_t)phys * vec->action_mask_size;
                } else {
                    env->agents[s].action_mask = NULL;
                }
                cursors[policy]++;
            }
            env->tag = tag;
            env->boundary_reached = 0;
        }
        free(cursors);
        free(counts);
    }

    return vec;
}

void vec_reset(VecEnv* vec) {
    Env* envs = vec->envs;
    #pragma omp parallel for schedule(static) num_threads(vec->num_workers)
    for (int i = 0; i < vec->size; i++) {
        puf_reset(&envs[i]);
    }
    cudaMemcpy(vec->gpu_observations, vec->observations,
        (size_t)vec->total_agents * OBS_SIZE * sizeof(obs_t), cudaMemcpyHostToDevice);
    cudaMemset(vec->gpu_rewards,   0, vec->total_agents * sizeof(float));
    cudaMemset(vec->gpu_terminals, 0, vec->total_agents * sizeof(float));
    if (vec->action_mask_size > 0) {
        cudaMemcpy(vec->gpu_action_mask, vec->action_mask,
            (size_t)vec->total_agents * vec->action_mask_size * sizeof(unsigned char),
            cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();
}

void vec_create_threads(VecEnv* vec, int num_threads, int horizon, struct PuffeRL* pufferl) {
    vec->buffer_states = (atomic_int*)calloc(vec->buffers, sizeof(atomic_int));
    vec->threads = (pthread_t*)calloc(vec->buffers, sizeof(pthread_t));
    vec->workers = (VecWorker*)calloc(vec->buffers, sizeof(VecWorker));
    vec->accum = (float*)calloc(vec->buffers * NUM_VEC_PROF, sizeof(float));
    for (int i = 0; i < vec->buffers; i++) {
        vec->workers[i].vec = vec;
        vec->workers[i].buf = i;
        vec->workers[i].horizon = horizon;
        vec->workers[i].pufferl = pufferl;
        pthread_create(&vec->threads[i], NULL, vec_thread_main, &vec->workers[i]);
    }
}

void vec_close(VecEnv* vec) {
    Env* envs = vec->envs;

    if (vec->threads != NULL) {
        atomic_store(&vec->shutdown, 1);
        for (int i = 0; i < vec->buffers; i++) {
            pthread_join(vec->threads[i], NULL);
        }
    }

    for (int i = 0; i < vec->size; i++) {
        Env* env = &envs[i];
        puf_close(env);
    }

    free(vec->envs);
    free(vec->buffer_states);
    free(vec->threads);
    free(vec->workers);
    free(vec->accum);
    free(vec->buffer_env_starts);
    free(vec->buffer_env_counts);
    free(vec->bank_layout);

    cudaDeviceSynchronize();
    cudaFree(vec->gpu_observations);
    cudaFree(vec->gpu_actions);
    cudaFree(vec->gpu_rewards);
    cudaFree(vec->gpu_terminals);
    cudaFreeHost(vec->observations);
    cudaFreeHost(vec->actions);
    cudaFreeHost(vec->rewards);
    cudaFreeHost(vec->terminals);
    if (vec->action_mask_size > 0) {
        cudaFree(vec->gpu_action_mask);
        cudaFreeHost(vec->action_mask);
    }

    free(vec->streams);
    free(vec);
}

void vec_log(VecEnv* vec, Dict* out, int clear) {
    Env* envs = vec->envs;
    Log aggregate;
    memset(&aggregate, 0, sizeof(Log));
    int num_keys = sizeof(Log) / sizeof(float);
    for (int i = 0; i < vec->size; i++) {
        Env* env = &envs[i];
        if (env->log.n == 0) {
            continue;
        }
        for (int j = 0; j < num_keys; j++) {
            ((float*)&aggregate)[j] += ((float*)&env->log)[j];
        }
    }

    float n = aggregate.n;
    if (n == 0) {
        return;
    }
    for (int i = 0; i < num_keys; i++) {
        ((float*)&aggregate)[i] /= n;
    }
    if (clear) {
        for (int i = 0; i < vec->size; i++) {
            memset(&envs[i].log, 0, sizeof(Log));
        }
    }
    puf_log(&aggregate, out);
    dict_set(out, "n", n);
}
