// CUDA
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <nccl.h>
#include <nvml.h>
#include <nvtx3/nvToolsExt.h>

// C standard
#include <cassert>
#include <cmath>
#include <cstdint>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// POSIX / threading
#include <dirent.h>
#include <fcntl.h>
#include <omp.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

// Project
#include "ini.h"

// To investigate: 32f compute? Need to check bf16
#ifdef PRECISION_FLOAT
typedef float precision_t;
constexpr bool USE_BF16 = false;
constexpr cudaDataType_t CUBLAS_PRECISION = CUDA_R_32F;
constexpr cublasComputeType_t CUBLAS_COMPUTE_PRECISION = CUBLAS_COMPUTE_32F;
#define NCCL_PRECISION ncclFloat
#define to_float(x) (x)
#define from_float(x) (x)
#else
typedef __nv_bfloat16 precision_t;
constexpr bool USE_BF16 = true;
constexpr cudaDataType_t CUBLAS_PRECISION = CUDA_R_16BF;
constexpr cublasComputeType_t CUBLAS_COMPUTE_PRECISION = CUBLAS_COMPUTE_32F;
#define NCCL_PRECISION ncclBfloat16
#define to_float(x) __bfloat162float(x)
#define from_float(x) __float2bfloat16(x)
#endif

#define PUF_MAX_DIMS 8
#define BLOCK_SIZE 256
int grid_size(int N) {
    return (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

// Compile-time env: build.sh passes -DENV_HEADER="ocean/<env>/<env>.h"
// (and GPU_ENV_HEADER when a .cu exists). Needs ini (Dict) + grid_size for GPU envs.
#include ENV_HEADER
#ifdef GPU_ENV_HEADER
#include GPU_ENV_HEADER
#endif

typedef struct {
    float* data;
    int64_t shape[PUF_MAX_DIMS];
} FloatTensor;

typedef struct {
    unsigned char* data;
    int64_t shape[PUF_MAX_DIMS];
} ByteTensor;

typedef struct {
    long* data;
    int64_t shape[PUF_MAX_DIMS];
} LongTensor;

typedef struct {
    int* data;
    int64_t shape[PUF_MAX_DIMS];
} IntTensor;

typedef struct {
    precision_t* data;
    int64_t shape[PUF_MAX_DIMS];
} PrecisionTensor;

__host__ __device__ int ndim(int64_t* shape) {
    int n = 0;
    while (n < PUF_MAX_DIMS && shape[n] != 0) {
        n++;
    }
    return n;
}

__host__ __device__ int64_t numel(int64_t* shape) {
    int64_t n = 1;
    for (int i = 0; i < PUF_MAX_DIMS && shape[i] != 0; i++) {
        n *= shape[i];
    }
    return n;
}

int64_t batch_size(int64_t* shape) {
    int n = ndim(shape);
    int64_t b = 1;
    for (int i = 0; i < n - 2; i++) {
        b *= shape[i];
    }
    return b;
}

void puf_squeeze_shape(int64_t* shape, int dim) {
    int n = ndim(shape);
    shape[dim] *= shape[dim + 1];
    for (int i = dim + 1; i < n - 1; i++) {
        shape[i] = shape[i + 1];
    }
    shape[n - 1] = 0;
}

PrecisionTensor* puf_squeeze(PrecisionTensor* t, int dim) {
    puf_squeeze_shape(t->shape, dim);
    return t;
}

FloatTensor* puf_squeeze(FloatTensor* t, int dim) {
    puf_squeeze_shape(t->shape, dim);
    return t;
}

PrecisionTensor* puf_unsqueeze(PrecisionTensor* t, int dim, int64_t d0, int64_t d1) {
    int n = ndim(t->shape);
    assert(n + 1 <= PUF_MAX_DIMS);
    assert(t->shape[dim] == d0 * d1);
    for (int i = n; i > dim; i--) {
        t->shape[i] = t->shape[i - 1];
    }
    t->shape[dim] = d0;
    t->shape[dim + 1] = d1;
    return t;
}

void puf_copy(PrecisionTensor* dst, PrecisionTensor* src, cudaStream_t stream) {
    assert(numel(dst->shape) == numel(src->shape) && "puf_copy: size mismatch");
    cudaMemcpyAsync(dst->data, src->data,
        numel(dst->shape) * sizeof(precision_t),
        cudaMemcpyDeviceToDevice, stream);
}

__global__ void cast(precision_t* dst,
        float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float(src[idx]);
    }
}

#ifndef PRECISION_FLOAT
__global__ void cast(float* dst,
        precision_t* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = to_float(src[idx]);
    }
}
#endif

__global__ void cast(precision_t* dst,
        unsigned char* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float((float)src[idx]);
    }
}

__global__ void cast(unsigned char* dst,
        precision_t* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = to_float(src[idx]);
    }
}

// Fused rew+term cast (two tiny launches were launch-bound).
__global__ void cast_rew_term(
        precision_t* __restrict__ rew_dst, const float* __restrict__ rew_src,
        precision_t* __restrict__ term_dst, const float* __restrict__ term_src,
        int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        rew_dst[idx] = from_float(rew_src[idx]);
        term_dst[idx] = from_float(term_src[idx]);
    }
}

struct AllocEntry {
    void** data_ptr;    // address of the tensor's data field
    int64_t* shape;     // pointer to the tensor's shape array
    int elem_size;      // sizeof element type
};

struct Allocator {
    AllocEntry* regs = nullptr;
    int num_regs = 0;
    void* mem = nullptr;
    long total_elems = 0;
    long total_bytes = 0;
};

void alloc_register_impl(Allocator* alloc, void** data_ptr, int64_t* shape, int elem_size) {
    alloc->regs = (AllocEntry*)realloc(alloc->regs, (alloc->num_regs + 1) * sizeof(AllocEntry));
    alloc->regs[alloc->num_regs++] = {data_ptr, shape, elem_size};
    int64_t n = numel(shape);
    alloc->total_elems += n;
    alloc->total_bytes = (alloc->total_bytes + 15) & ~15;
    alloc->total_bytes += n * elem_size;
}
void alloc_register(Allocator* a, PrecisionTensor* t) {
    alloc_register_impl(a, (void**)&t->data, t->shape, sizeof(precision_t));
}
void alloc_register(Allocator* a, FloatTensor* t) {
    alloc_register_impl(a, (void**)&t->data, t->shape, sizeof(float));
}
void alloc_register(Allocator* a, LongTensor* t) {
    alloc_register_impl(a, (void**)&t->data, t->shape, sizeof(long));
}
void alloc_register(Allocator* a, IntTensor* t) {
    alloc_register_impl(a, (void**)&t->data, t->shape, sizeof(int));
}

cudaError_t alloc_create(Allocator* alloc) {
    if (alloc->total_bytes == 0) {
        return cudaSuccess;
    }
    cudaError_t err = cudaMalloc(&alloc->mem, alloc->total_bytes);
    if (err != cudaSuccess) {
        return err;
    }
    cudaMemset(alloc->mem, 0, alloc->total_bytes);
    long offset = 0;
    for (int i = 0; i < alloc->num_regs; i++) {
        offset = (offset + 15) & ~15;
        *alloc->regs[i].data_ptr = (char*)alloc->mem + offset;
        offset += numel(alloc->regs[i].shape) * alloc->regs[i].elem_size;
    }
    return cudaSuccess;
}

void alloc_free(Allocator* alloc) {
    if (alloc->mem) {
        cudaFree(alloc->mem);
        alloc->mem = nullptr;
    }
    if (alloc->regs) {
        free(alloc->regs);
        alloc->regs = nullptr;
    }
    alloc->num_regs = 0;
    alloc->total_elems = 0;
    alloc->total_bytes = 0;
}

// Policy / optim / GEMM / encoders / weight init.
// Needs tensors, Allocator, batch_size/ndim, grid_size, cast, CUBLAS_PRECISION*.
#include "algo.cu"
#include "protein.cu"

typedef struct {
    int horizon;
    int total_agents;
    int num_buffers;
    int hidden_size;
    int num_layers;
    float lr;
    float min_lr_ratio;
    bool anneal_lr;
    float momentum;
    int minibatch_size;
    float replay_ratio;
    long total_timesteps;
    float max_grad_norm;
    float clip_coef;
    float vf_clip_coef;
    float vf_coef;
    float ent_coef;
    float min_ent_coef_ratio;
    bool anneal_ent_coef;
    float gamma;
    float gae_lambda;
    float vtrace_rho_clip;
    float vtrace_c_clip;
    float prio_alpha;
    float prio_beta0;
    bool async;
    bool reset_state;
    // true when base.cudagraphs >= 0: first real rollout/train use captures.
    bool cudagraphs;
    bool profile;
    int rank;
    int world_size;
    int gpu_id;
    int num_threads;
    int seed;
} HypersT;

// Rank / device context for one process in a multi-GPU train job.
typedef struct {
    int rank;
    int world_size;
    int gpu_id;
    int artifact_owner;
    ncclUniqueId* nccl_id;
} TrainContext;

void puf_die(const char* fmt, ...) {
    va_list args;
    fprintf(stderr, "config error: ");
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    exit(1);
}

int puf_err(char* out, size_t out_size, const char* fmt, ...) {
    if (out && out_size > 0) {
        va_list args;
        va_start(args, fmt);
        vsnprintf(out, out_size, fmt, args);
        va_end(args);
    }
    return 0;
}

// Shared by startup validation and sweep sample soft-check.
static int train_shape_valid(int mb, int horizon, int agents, int gpus,
        char* err, size_t err_size) {
    if (gpus < 1)
        return puf_err(err, err_size, "train.gpus must be >= 1");
    if (horizon <= 0 || mb % horizon != 0)
        return puf_err(err, err_size, "train.minibatch_size must be divisible by train.horizon");
    if (mb > (long)horizon * agents)
        return puf_err(err, err_size, "train.minibatch_size > train.horizon * vec.total_agents");
    return 1;
}

int puf_config_train_valid(Ini* ini, char* err, size_t err_size) {
    return train_shape_valid(
        (int)puf_ini_get(ini, "train", "minibatch_size"),
        (int)puf_ini_get(ini, "train", "horizon"),
        (int)puf_ini_get(ini, "vec", "total_agents"),
        (int)puf_ini_get(ini, "train", "gpus"),
        err, err_size);
}

typedef struct ObsTensor {
    obs_t* data;
    int64_t shape[8];
} ObsTensor;

// Data collected by parallel environment workers. Each worker handles
// a constant subset of agents
struct RolloutBuf {
    PrecisionTensor observations;  // (horizon, agents, input_size)
    // (slots, layers, agents, hidden) when !reset_state; slots = async ? 2 : 1.
    // Default path is carry + async: per-slot states for pipelined horizons.
    PrecisionTensor initial_states;
    PrecisionTensor actions;       // (horizon, agents, num_atns)
    PrecisionTensor values;        // (horizon, agents)
    PrecisionTensor logprobs;      // ...
    PrecisionTensor rewards;
    PrecisionTensor terminals;
    PrecisionTensor ratio;
    PrecisionTensor importance;
    PrecisionTensor action_mask;   // (horizon, agents, mask_size); .data=nullptr when env opts out
};

// Buffers are initialized as raw structs with only shape information. alloc_register
// stores the shape and data pointer. Memory is only allocated after all buffers are registered.
void register_rollout_buffers(RolloutBuf& bufs, Allocator* alloc, int T, int B, int input_size,
        int num_atns, int mask_size) {
    memset(&bufs, 0, sizeof(bufs));
    bufs.observations = {.shape = {T, B, input_size}};
    bufs.actions = {.shape = {T, B, num_atns}};
    bufs.values = {.shape = {T, B}};
    bufs.logprobs = {.shape = {T, B}};
    bufs.rewards = {.shape = {T, B}};
    bufs.terminals = {.shape = {T, B}};
    bufs.ratio = {.shape = {T, B}};
    bufs.importance = {.shape = {T, B}};
    PrecisionTensor* fields[] = {
        &bufs.observations, &bufs.actions, &bufs.values, &bufs.logprobs,
        &bufs.rewards, &bufs.terminals, &bufs.ratio, &bufs.importance,
    };
    for (int i = 0; i < (int)(sizeof(fields) / sizeof(fields[0])); i++) {
        alloc_register(alloc, fields[i]);
    }
    if (mask_size > 0) {
        bufs.action_mask = {.shape = {T, B, mask_size}};
        alloc_register(alloc, &bufs.action_mask);
    }
}

PrecisionTensor puf_time_view(PrecisionTensor p, int start_t, int T) {
    if (p.data == NULL) {
        return p;
    }
    int n = ndim(p.shape);
    if (n == 3) {
        long B = p.shape[1], F = p.shape[2];
        return {.data = p.data + (long)start_t * B * F, .shape = {T, B, F}};
    }
    if (n == 2) {
        long B = p.shape[1];
        return {.data = p.data + (long)start_t * B, .shape = {T, B}};
    }
    return p;
}

RolloutBuf rollout_time_view(RolloutBuf* base, int start_t, int T) {
    RolloutBuf view = *base;
    view.observations = puf_time_view(base->observations, start_t, T);
    view.actions      = puf_time_view(base->actions,      start_t, T);
    view.values       = puf_time_view(base->values,       start_t, T);
    view.logprobs     = puf_time_view(base->logprobs,     start_t, T);
    view.rewards      = puf_time_view(base->rewards,      start_t, T);
    view.terminals    = puf_time_view(base->terminals,    start_t, T);
    view.ratio        = puf_time_view(base->ratio,        start_t, T);
    view.importance   = puf_time_view(base->importance,   start_t, T);
    view.action_mask  = puf_time_view(base->action_mask,  start_t, T);
    return view;
}

enum VecProfileIdx {
    VEC_MODEL = 0,
    VEC_ENV_STEP,
    VEC_COPY,
    NUM_VEC_PROF,
};

// Env batch + host/device buffers. Per-buffer worker threads are started later
// with a PuffeRL* (see vec_create_threads); they are not isolated from the trainer.
struct VecEnv {
    Env* envs;
    int gpu_env;
    void* gpu_envs;
    float* gpu_log;
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
    // Cross-thread flags: only accessed via __atomic_* at the few sites below.
    int* buffer_states;
    int shutdown;
    pthread_t* threads;
    void* thread_args;  // VecThreadArg[buffers]; owned once threads exist
    float* accum;
    int num_workers;
    int action_mask_size;
    int num_banks;
    int* bank_layout;  // per-buffer agent offsets; sole owner (not mirrored on PuffeRL)
};

struct EnvBuf {
    ObsTensor obs;         // (total_agents, obs_size)
    FloatTensor actions;   // (total_agents, num_atns)
    FloatTensor rewards;   // (total_agents,)
    FloatTensor terminals; // (total_agents,)
    ByteTensor action_mask; // (total_agents, mask_size); .data=nullptr when env opts out
};

// Frozen opponent bank (selfplay / match): rollout-only policy + params.
// Same build_policy / weights_create path as primary, but no train acts, grads, or muon.
// Optional different hidden/layers via frozen_bank_hidden_size / frozen_bank_num_layers.
typedef struct {
    Policy policy;
    PolicyWeights weights;
    Allocator params_alloc;
    Allocator acts_alloc;
    PrecisionTensor param_puf;
    FloatTensor master_weights;
    PrecisionTensor* buffer_states;         // [num_buffers]
    PolicyActivations* buffer_activations;  // [num_buffers]
} WeightBank;

enum ProfileIdx {
    PROF_ROLLOUT = 0,
    PROF_EVAL_MODEL,
    PROF_EVAL_ENV,
    PROF_EVAL_COPY,
    PROF_TRAIN_MISC,
    PROF_TRAIN_MODEL,
};

// Index must match ProfileIdx order.
const char* PROF_NAMES[] = {
    "rollout",
    "eval_model",
    "eval_env",
    "eval_copy",
    "train_misc",
    "train_model",
};

typedef struct {
    cudaEvent_t events[5];  // train_impl markers: pre-start, pre-end, misc, model-start, model-end
    cudaEvent_t* rollout_gpu_start;
    cudaEvent_t* rollout_gpu_end;
    cudaEvent_t* rollout_env_end;
    int rollout_horizon;
    float accum[sizeof(PROF_NAMES) / sizeof(PROF_NAMES[0])];
} ProfileT;

typedef struct PuffeRL {
    Policy policy;
    PolicyWeights weights;       // current precision_t weights (structured)
    PolicyWeights actor_weights; // async rollout snapshot; unused when async=0
    PolicyActivations train_activations;
    Allocator params_alloc;
    Allocator actor_params_alloc;
    Allocator grads_alloc;
    Allocator activations_alloc;
    VecEnv* vec;
    Muon muon;
    ncclComm_t nccl_comm;  // NCCL communicator for multi-GPU
    HypersT hypers;
    bool is_continuous;  // True if all action dimensions are continuous (size==1)
    PrecisionTensor* buffer_states;  // Per-buffer states for contiguous access
    PolicyActivations* buffer_activations;  // Per-buffer inference activations
    RolloutBuf rollouts;
    RolloutBuf train_rollouts;  // Pre-allocated transposed copy for train_impl
    EnvBuf env;
    TrainGraph train_buf;
    PrecisionTensor advantages_puf;  // Pre-allocated for train_impl (B, T)
    cudaGraphExec_t* fused_rollout_cudagraphs;  // [slots][horizon][num_buffers]; null if !cudagraphs
    cudaGraphExec_t train_cudagraph;  // null until first-use capture
    cudaStream_t* streams;  // per-buffer raw CUDA streams
    cudaStream_t default_stream;  // main-thread stream (captured once at init)
    cudaStream_t train_stream;    // dedicated learner stream (always non-default)
    IntTensor act_sizes_puf;    // CUDA int32 tensor of action head sizes
    FloatTensor losses_puf;     // (NUM_LOSSES,) f32 accumulator
    PPOBuffersPuf ppo_bufs_puf; // Pre-allocated buffers for ppo_loss_fwd_bwd
    PrioBuffers prio_bufs;      // Pre-allocated buffers for prio_replay
    FloatTensor master_weights;  // fp32 master weights (flat); same buffer as param_puf in fp32 mode
    PrecisionTensor param_puf;
    PrecisionTensor actor_param_puf;
    PrecisionTensor grad_puf;
    LongTensor rng_offset_puf;   // (num_buffers+1,) int64 CUDA device counters
    ProfileT profile;
    nvmlDevice_t nvml_device;
    long epoch;
    long global_step;
    double start_time;
    double last_log_time;
    long last_log_step;
    int rollout_write_slot;
    int async_ready_slot;
    int async_next_slot;
    bool async_bootstrapped;
    ulong seed;
    curandStatePhilox4_32_10_t** rng_states;  // per-buffer persistent RNG states [num_buffers]
    // Optional frozen weight banks for match / selfplay opponents.
    WeightBank* frozen_banks;  // [num_frozen_banks]
    int num_frozen_banks;
    char env_name[64];  // For frozen-bank policy rebuild at create.
} PuffeRL;

// --- Infer path: sample + forward, then vec workers ---
#define profile_begin(tag, enable) do { if (enable) nvtxRangePushA(tag); } while (0)
#define profile_end(enable) do { if (enable) nvtxRangePop(); } while (0)

__global__ void rng_init(curandStatePhilox4_32_10_t* states, uint64_t seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__device__ float masked_logit(precision_t* logits,
        int logits_base, int logits_offset, int offset,
        precision_t* mask, int mask_base) {
    float l = safe_logit(logits, logits_base, logits_offset, offset);
    if (mask != nullptr) {
        float m = to_float(mask[mask_base + logits_offset + offset]);
        if (m == 0.0f) l = -1e4f;
    }
    return l;
}

// Expects action logits and values to be in the same contiguous buffer. See default decoder
__global__ void sample_logits(
        PrecisionTensor dec_out,              // (B, logits_dim + 1 for values)
        PrecisionTensor logstd_puf,           // (1, od) - continuous actions only
        IntTensor act_sizes_puf,              // (num_atns,) action head sizes
        precision_t* actions,    // (B, num_atns)
        precision_t* logprobs,   // (B,)
        precision_t* value_out,  // (B,)
        curandStatePhilox4_32_10_t* rng_states,
        precision_t* action_mask, // (B, A_total) or nullptr
        int mask_stride) {                    // 0 when action_mask is nullptr
    int B = dec_out.shape[0];
    int fused_cols = dec_out.shape[1];
    int num_atns = numel(act_sizes_puf.shape);
    int* act_sizes = act_sizes_puf.data;
    precision_t* logits = dec_out.data;
    int logits_stride = fused_cols;
    int value_stride = fused_cols;
    bool is_continuous = logstd_puf.data != nullptr && numel(logstd_puf.shape) > 0;
    precision_t* logstd = logstd_puf.data;
    precision_t* value = logits + (fused_cols - 1);  // last column

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B) {
        return;
    }

    // Load persistent RNG state (advanced in-place each call)
    curandStatePhilox4_32_10_t state = rng_states[idx];

    int logits_base = idx * logits_stride;
    float total_log_prob = 0.0f;

    if (is_continuous) {
        // Continuous action sampling from Normal(mean, exp(logstd)); logstd is 1D broadcast.
        constexpr float LOG_2PI = 1.8378770664093453f;  // log(2*pi)

        for (int h = 0; h < num_atns; ++h) {
            float mean = safe_continuous_mean(logits, logits_base + h);
            float log_std = safe_continuous_logstd(logstd, h);
            float std = expf(log_std);

            // Sample from N(0,1) and transform: action = mean + std * noise
            float noise = curand_normal(&state);
            float action = finite_or_clamp(mean + std * noise, -1.0e6f, 1.0e6f);

            precision_t stored_action_p = from_float(action);
            float stored_action = to_float(stored_action_p);
            // Log probability: -0.5 * ((action - mean) / std)^2 - 0.5 * log(2*pi) - log(std)
            float normalized = (stored_action - mean) / std;
            float log_prob = -0.5f * normalized * normalized - 0.5f * LOG_2PI - log_std;

            actions[idx * num_atns + h] = stored_action_p;
            total_log_prob += log_prob;
        }
    } else if (num_atns == 1 && action_mask == nullptr && act_sizes[0] <= 8) {
        // Fast path: single discrete head, no mask (breakout A=3). Logits in
        // regs, one curand — general path re-reads logits 2–3× via masked_logit.
        int A = act_sizes[0];
        float logit_reg[8];
        float max_val = -INFINITY;
        for (int a = 0; a < A; ++a) {
            float l = to_float(logits[logits_base + a]);
            if (isnan(l)) {
                l = 0.0f;
            }
            logit_reg[a] = l;
            max_val = fmaxf(max_val, l);
        }
        float sum_exp = 0.0f;
        for (int a = 0; a < A; ++a) {
            sum_exp += expf(logit_reg[a] - max_val);
        }
        float logsumexp = max_val + logf(sum_exp);
        float rand_val = curand_uniform(&state);
        float cumsum = 0.0f;
        int sampled_action = A - 1;
        for (int a = 0; a < A; ++a) {
            cumsum += expf(logit_reg[a] - logsumexp);
            if (rand_val < cumsum) {
                sampled_action = a;
                break;
            }
        }
        actions[idx] = from_float((float)sampled_action);
        total_log_prob = logit_reg[sampled_action] - logsumexp;
    } else {
        // Discrete action sampling (general multi-head / masked)
        int logits_offset = 0;  // offset within row for current action head
        int mask_base = (action_mask != nullptr) ? idx * mask_stride : 0;

        for (int h = 0; h < num_atns; ++h) {
            int A = act_sizes[h];  // size of this action head

            // Step 1: Find max and sum for numerical stability (with nan_to_num)
            float max_val = -INFINITY;
            float sum_exp = 0.0f;
            for (int a = 0; a < A; ++a) {
                float l = masked_logit(logits, logits_base, logits_offset, a, action_mask, mask_base);
                if (l > max_val) {
                    sum_exp *= expf(max_val - l);
                    max_val = l;
                }
                sum_exp += expf(l - max_val);
            }
            float logsumexp = max_val + logf(sum_exp);

            // Step 3: Generate random value for this action head
            float rand_val = curand_uniform(&state);

            // Step 4: Multinomial sampling using inverse CDF
            float cumsum = 0.0f;
            int sampled_action = -1;  // sentinel: no action chosen yet

            for (int a = 0; a < A; ++a) {
                float l = masked_logit(logits, logits_base, logits_offset, a, action_mask, mask_base);
                float prob = expf(l - logsumexp);
                cumsum += prob;
                if (rand_val < cumsum) {
                    sampled_action = a;
                    break;
                }
            }

            // Float rounding can leave cumsum < 1.0; fall back to the last legal action.
            if (sampled_action < 0) {
                sampled_action = A - 1;
                if (action_mask != nullptr) {
                    for (int a = A - 1; a >= 0; --a) {
                        if (to_float(action_mask[mask_base + logits_offset + a]) != 0.0f) {
                            sampled_action = a;
                            break;
                        }
                    }
                }
            }

            // Step 5: Gather log probability of sampled action
            float sampled_logit = masked_logit(logits, logits_base, logits_offset, sampled_action, action_mask, mask_base);
            float log_prob = sampled_logit - logsumexp;

            // Write action for this head
            actions[idx * num_atns + h] = from_float(sampled_action);
            total_log_prob += log_prob;

            // Advance to next action head
            logits_offset += A;
        }
    }

    // Write summed log probability (log of joint probability)
    logprobs[idx] = from_float(total_log_prob);

    // Copy value (fused to avoid separate elementwise kernel for strided->contiguous copy)
    value_out[idx] = value[idx * value_stride];

    // Save RNG state back for next call
    rng_states[idx] = state;
}

// Primary bank state is contiguous from agent 0 in src; only dst_start needed.
__global__ void snapshot_initial_state(PrecisionTensor dst, PrecisionTensor src,
        int dst_start, int count) {
    int L = src.shape[0];
    int H = src.shape[2];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = L * count * H;
    if (idx >= total) {
        return;
    }

    int h = idx % H;
    int rel = (idx / H) % count;
    int layer = idx / (count * H);
    long src_idx = ((long)layer * src.shape[1] + rel) * H + h;
    long dst_idx = ((long)layer * dst.shape[1] + dst_start + rel) * H + h;
    dst.data[dst_idx] = src.data[src_idx];
}

// View slot of full (slots, L, A, H) as (L, A, H).
static PrecisionTensor initial_states_slot(PrecisionTensor full, int slot) {
    int L = (int)full.shape[1];
    int A = (int)full.shape[2];
    int H = (int)full.shape[3];
    long stride = (long)L * A * H;
    return {.data = full.data + (long)slot * stride, .shape = {L, A, H}};
}

// One thread per agent; zeros all layers/H when that agent is terminal.
__global__ void zero_state_on_terminal(PrecisionTensor state, FloatTensor terminals,
        int state_start, int terminal_start, int count) {
    int rel = blockIdx.x * blockDim.x + threadIdx.x;
    if (rel >= count) {
        return;
    }
    if (terminals.data[terminal_start + rel] == 0.0f) {
        return;
    }
    int L = state.shape[0];
    int H = state.shape[2];
    int agents_stride = state.shape[1];
    for (int layer = 0; layer < L; layer++) {
        long base = ((long)layer * agents_stride + state_start + rel) * H;
        for (int h = 0; h < H; h++) {
            state.data[base + h] = from_float(0.0f);
        }
    }
}

// Slice: select dim0 index t, then narrow dim0 from start for count.
// 3D (T, B, F) -> (count, F); 2D (T, B) -> (count,)
PrecisionTensor puf_slice(PrecisionTensor& p, int t, int start, int count) {
    if (ndim(p.shape) == 3) {
        long B = p.shape[1], F = p.shape[2];
        return {.data = p.data + (t*B + start)*F, .shape = {count, F}};
    } else {
        long B = p.shape[1];
        return {.data = p.data + (t*B + start), .shape = {count}};
    }
}

void pufferl_forward(PuffeRL* pufferl, int buf, int t, cudaStream_t stream) {
    HypersT& hypers = pufferl->hypers;
    int graph_slot = hypers.async ? pufferl->rollout_write_slot : 0;
    int graph = (graph_slot * hypers.horizon + t) * hypers.num_buffers + buf;
    profile_begin("fused_rollout", hypers.profile);

    if (pufferl->fused_rollout_cudagraphs != nullptr
            && pufferl->fused_rollout_cudagraphs[graph] != nullptr) {
        assert(cudaGraphLaunch(pufferl->fused_rollout_cudagraphs[graph], stream) == cudaSuccess
                && "cudaGraphLaunch failed");
        profile_end(hypers.profile);
        return;
    }

    bool capturing = hypers.cudagraphs;
    if (capturing) {
        assert(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) == cudaSuccess
                && "cudaStreamBeginCapture failed");
    }

    RolloutBuf rollouts = pufferl->rollouts;
    if (hypers.async) {
        rollouts = rollout_time_view(&pufferl->rollouts,
            pufferl->rollout_write_slot * hypers.horizon, hypers.horizon);
    }
    EnvBuf& env = pufferl->env;
    VecEnv* vec = pufferl->vec;
    int block_size = hypers.total_agents / hypers.num_buffers;
    int start = buf * block_size;
    int* layout = vec->bank_layout;

    // Copy observations, rewards, terminals from GPU env buffers to rollout buffer
    ObsTensor& obs_env = env.obs;
    int n = block_size * obs_env.shape[1];
    PrecisionTensor obs_dst = puf_slice(rollouts.observations, t, start, block_size);
#ifdef OBS_MATCHES_PRECISION
    // Env already wrote precision_t (e.g. breakout bf16) — D2D copy, no float cast.
    cudaMemcpyAsync(obs_dst.data, obs_env.data + (long)start * obs_env.shape[1],
        (size_t)n * sizeof(precision_t), cudaMemcpyDeviceToDevice, stream);
#else
    // obs_t → precision_t (float/uchar → bf16/float)
    cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
        obs_dst.data, obs_env.data + (long)start * obs_env.shape[1], n);
#endif

    PrecisionTensor rew_dst = puf_slice(rollouts.rewards, t, start, block_size);
    PrecisionTensor term_dst = puf_slice(rollouts.terminals, t, start, block_size);
    cast_rew_term<<<grid_size(block_size), BLOCK_SIZE, 0, stream>>>(
        rew_dst.data, env.rewards.data + start,
        term_dst.data, env.terminals.data + start, block_size);

    // Copy action mask from env into rollout buffer (if env opted in)
    PrecisionTensor mask_slice = {};
    int mask_stride = 0;
    if (rollouts.action_mask.data != nullptr) {
        int mask_size = rollouts.action_mask.shape[2];
        mask_stride = mask_size;
        mask_slice = puf_slice(rollouts.action_mask, t, start, block_size);
        int mask_n = block_size * mask_size;
        cast<<<grid_size(mask_n), BLOCK_SIZE, 0, stream>>>(
            mask_slice.data,
            env.action_mask.data + (long)start * mask_size,
            mask_n);
    }

    // Per-bank forward: layout[b]..layout[b+1) within each buffer chunk.
    int num_banks = 1 + pufferl->num_frozen_banks;
    long act_cols = env.actions.shape[1];
    for (int b = 0; b < num_banks; b++) {
        int bank_off = layout[b];
        int bank_end = layout[b + 1];
        int bank_size = bank_end - bank_off;
        if (bank_size == 0) continue;

        Policy* p_bank;
        PolicyWeights* w_bank;
        PolicyActivations* a_bank;
        PrecisionTensor* s_bank;
        if (b == 0) {
            p_bank = &pufferl->policy;
            w_bank = hypers.async ? &pufferl->actor_weights : &pufferl->weights;
            a_bank = &pufferl->buffer_activations[buf];
            s_bank = &pufferl->buffer_states[buf];
        } else {
            WeightBank* fb = &pufferl->frozen_banks[b - 1];
            p_bank = &fb->policy;
            w_bank = &fb->weights;
            a_bank = &fb->buffer_activations[buf];
            s_bank = &fb->buffer_states[buf];
        }

        int sub_start = start + bank_off;
        PrecisionTensor obs_b   = puf_slice(rollouts.observations, t, sub_start, bank_size);
        PrecisionTensor act_b   = puf_slice(rollouts.actions,      t, sub_start, bank_size);
        PrecisionTensor lp_b    = puf_slice(rollouts.logprobs,     t, sub_start, bank_size);
        PrecisionTensor val_b   = puf_slice(rollouts.values,       t, sub_start, bank_size);
        PrecisionTensor mask_b  = {};
        int mask_stride_b = 0;
        if (rollouts.action_mask.data != nullptr) {
            mask_b = puf_slice(rollouts.action_mask, t, sub_start, bank_size);
            mask_stride_b = mask_stride;
        }

        int state_start = (b == 0) ? bank_off : 0;
        zero_state_on_terminal<<<grid_size(bank_size), BLOCK_SIZE, 0, stream>>>(
            *s_bank, env.terminals, state_start, sub_start, bank_size);

        if (b == 0 && t == 0 && rollouts.initial_states.data != nullptr) {
            int state_n = s_bank->shape[0] * bank_size * s_bank->shape[2];
            PrecisionTensor init_slot = initial_states_slot(
                rollouts.initial_states, graph_slot);
            snapshot_initial_state<<<grid_size(state_n), BLOCK_SIZE, 0, stream>>>(
                init_slot, *s_bank, sub_start, bank_size);
        }

        PrecisionTensor dec_puf = policy_forward(p_bank, *w_bank, *a_bank, obs_b, *s_bank, stream);

        PrecisionTensor p_logstd = {};
        DecoderWeights* dw = (DecoderWeights*)w_bank->decoder;
        if (dw->continuous) {
            p_logstd = dw->logstd;
        }

        // Offset RNG by bank_off so banks don't collide on per-buffer rng slots.
        sample_logits<<<grid_size(bank_size), BLOCK_SIZE, 0, stream>>>(
            dec_puf, p_logstd, pufferl->act_sizes_puf,
            act_b.data, lp_b.data, val_b.data,
            pufferl->rng_states[buf] + bank_off,
            mask_b.data, mask_stride_b);

        cast<<<grid_size(numel(act_b.shape)), BLOCK_SIZE, 0, stream>>>(
                env.actions.data + (long)sub_start * act_cols,
                act_b.data, numel(act_b.shape));
    }

    if (capturing) {
        cudaGraph_t _graph;
        cudaError_t err = cudaStreamEndCapture(stream, &_graph);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaStreamEndCapture rollout failed: %s\n", cudaGetErrorString(err));
            abort();
        }
        assert(cudaGraphInstantiate(&pufferl->fused_rollout_cudagraphs[graph], _graph, 0) == cudaSuccess
                && "cudaGraphInstantiate failed");
        assert(cudaGraphDestroy(_graph) == cudaSuccess && "cudaGraphDestroy failed");
        // Capture records without executing; run once so this step has effects.
        assert(cudaGraphLaunch(pufferl->fused_rollout_cudagraphs[graph], stream) == cudaSuccess
                && "cudaGraphLaunch failed");
        cudaDeviceSynchronize();
    }
    profile_end(hypers.profile);
}


// --- VecEnv worker threads + lifecycle (after pufferl_forward; needs PuffeRL) ---
// One arg per buffer thread: trainer + which buffer. vec/horizon come from pufferl.
typedef struct {
    PuffeRL* pufferl;
    int buf;
} VecThreadArg;

#define OMP_WAITING 5
#define OMP_RUNNING 6

void* vec_thread_main(void* arg) {
    VecThreadArg* a = (VecThreadArg*)arg;
    PuffeRL* pufferl = a->pufferl;
    VecEnv* vec = pufferl->vec;
    int buf = a->buf;
    int horizon = pufferl->hypers.horizon;
    cublas_init_handle();

    int agents_per_buffer = vec->agents_per_buffer;
    int agent_start = buf * agents_per_buffer;
    int env_start = vec->buffer_env_starts[buf];
    int env_count = vec->buffer_env_counts[buf];

    Env* envs = vec->envs;
    cudaStream_t stream = pufferl->streams[buf];
    cudaEvent_t model_start, model_end, copy_end, h2d_start, h2d_end;
    cudaEventCreate(&model_start);
    cudaEventCreate(&model_end);
    cudaEventCreate(&copy_end);
    cudaEventCreate(&h2d_start);
    cudaEventCreate(&h2d_end);

    float* my_accum = &vec->accum[buf * NUM_VEC_PROF];
    struct timespec t0, t1;
    float ms = 0.0f;

    int alive = 1;
    while (alive) {
        while (__atomic_load_n(&vec->buffer_states[buf], __ATOMIC_SEQ_CST) != OMP_RUNNING) {
            if (__atomic_load_n(&vec->shutdown, __ATOMIC_SEQ_CST)) {
                alive = 0;
                break;
            }
        }
        if (!alive) {
            break;
        }

        int h2d_pending = 0;

        for (int t = 0; t < horizon; t++) {
            cudaEventRecord(model_start, stream);
            pufferl_forward(pufferl, buf, t, stream);
            cudaEventRecord(model_end, stream);
            cudaMemcpyAsync(
                &vec->actions[agent_start * NUM_ATNS],
                &vec->gpu_actions[agent_start * NUM_ATNS],
                agents_per_buffer * NUM_ATNS * sizeof(float),
                cudaMemcpyDeviceToHost, stream);
            cudaEventRecord(copy_end, stream);
            cudaStreamSynchronize(stream);

            cudaEventElapsedTime(&ms, model_start, model_end);
            my_accum[VEC_MODEL] += ms;
            cudaEventElapsedTime(&ms, model_end, copy_end);
            my_accum[VEC_COPY] += ms;
            if (h2d_pending) {
                cudaEventElapsedTime(&ms, h2d_start, h2d_end);
                my_accum[VEC_COPY] += ms;
                h2d_pending = 0;
            }

            memset(&vec->rewards[agent_start], 0, agents_per_buffer * sizeof(float));
            memset(&vec->terminals[agent_start], 0, agents_per_buffer * sizeof(float));
            clock_gettime(CLOCK_MONOTONIC, &t0);
            #pragma omp parallel for schedule(static) num_threads(vec->num_workers)
            for (int i = env_start; i < env_start + env_count; i++) {
                puf_step(&envs[i]);
            }
            clock_gettime(CLOCK_MONOTONIC, &t1);
            my_accum[VEC_ENV_STEP] += (t1.tv_sec - t0.tv_sec) * 1000.0f + (t1.tv_nsec - t0.tv_nsec) / 1e6f;

            cudaEventRecord(h2d_start, stream);
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
                    vec->action_mask + agent_start * vec->action_mask_size,
                    (size_t)agents_per_buffer * vec->action_mask_size * sizeof(unsigned char),
                    cudaMemcpyHostToDevice, stream);
            }
            cudaEventRecord(h2d_end, stream);
            h2d_pending = 1;
        }
        cudaStreamSynchronize(stream);
        if (h2d_pending) {
            cudaEventElapsedTime(&ms, h2d_start, h2d_end);
            my_accum[VEC_COPY] += ms;
        }
        __atomic_store_n(&vec->buffer_states[buf], OMP_WAITING, __ATOMIC_SEQ_CST);
    }

    cudaEventDestroy(model_start);
    cudaEventDestroy(model_end);
    cudaEventDestroy(copy_end);
    cudaEventDestroy(h2d_start);
    cudaEventDestroy(h2d_end);
    return NULL;
}

Env* vec_init_envs(int* num_envs_out, int* buffer_env_starts, int* buffer_env_counts,
                 Dict* vec_kwargs, Dict* env_kwargs) {

// Note: we need a better way of controlling user-defined fns without macros.
#ifdef MY_VEC_INIT
    return my_vec_init(num_envs_out, buffer_env_starts, buffer_env_counts,
        vec_kwargs, env_kwargs);
#else
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
#endif
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
    int action_mask_size = (int)dict_get(vec_kwargs, "action_mask_size");
    int gpu_env_requested = (int)dict_get(vec_kwargs, "gpu_env");
    vec->num_banks = frozen_banks + 1;
    vec->bank_layout = (int*)calloc(vec->num_banks + 1, sizeof(int));

    vec->buffer_env_starts = (int*)calloc(num_buffers, sizeof(int));
    vec->buffer_env_counts = (int*)calloc(num_buffers, sizeof(int));

    vec->gpu_env = gpu_env_requested;
    int num_envs = 0;
    if (vec->gpu_env) {
        vec->size = total_agents;
        vec->buffer_env_starts[0] = 0;
        vec->buffer_env_counts[0] = total_agents;
#ifdef PUFFER_GPU_ENV_AVAILABLE
        vec->gpu_envs = puf_gpu_env_create(total_agents, env_kwargs);
#endif
    } else {
        vec->envs = vec_init_envs(&num_envs, vec->buffer_env_starts, vec->buffer_env_counts,
                                vec_kwargs, env_kwargs);
        vec->size = num_envs;
    }

    obs_t* observations = NULL;
    obs_t* gpu_observations = NULL;
    if (!vec->gpu_env) {
        cudaHostAlloc((void**)&observations,
            (size_t)total_agents * OBS_SIZE * sizeof(obs_t), cudaHostAllocPortable);
        cudaHostAlloc((void**)&vec->actions, total_agents * NUM_ATNS * sizeof(float), cudaHostAllocPortable);
        cudaHostAlloc((void**)&vec->rewards, total_agents * sizeof(float), cudaHostAllocPortable);
        cudaHostAlloc((void**)&vec->terminals, total_agents * sizeof(float), cudaHostAllocPortable);
    }

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

    vec->action_mask_size = action_mask_size;
    if (vec->gpu_env) {
        cudaMalloc((void**)&vec->gpu_log, sizeof(Log));
    }
    if (vec->action_mask_size > 0) {
        size_t mask_bytes = (size_t)total_agents * vec->action_mask_size * sizeof(unsigned char);
        cudaHostAlloc((void**)&vec->action_mask, mask_bytes, cudaHostAllocPortable);
        cudaMalloc((void**)&vec->gpu_action_mask, mask_bytes);
        cudaMemset(vec->gpu_action_mask, 0, mask_bytes);
    }
    if (vec->gpu_env) {
        vec->bank_layout[0] = 0;
        vec->bank_layout[vec->num_banks] = vec->agents_per_buffer;
        return vec;
    }

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
    if (vec->gpu_env) {
#ifdef PUFFER_GPU_ENV_AVAILABLE
        puf_gpu_env_reset(vec->gpu_envs, vec->gpu_observations, vec->gpu_rewards,
            vec->gpu_terminals, vec->total_agents);
        cudaDeviceSynchronize();
#endif
        return;
    }

    Env* envs = vec->envs;
    #pragma omp parallel for schedule(static) num_threads(vec->num_workers)
    for (int i = 0; i < vec->size; i++) {
        puf_reset(&envs[i]);
    }
    cudaMemcpy(vec->gpu_observations, vec->observations,
        (size_t)vec->total_agents * OBS_SIZE * sizeof(obs_t),
        cudaMemcpyHostToDevice);
    cudaMemset(vec->gpu_rewards,   0, vec->total_agents * sizeof(float));
    cudaMemset(vec->gpu_terminals, 0, vec->total_agents * sizeof(float));
    if (vec->action_mask_size > 0) {
        cudaMemcpy(vec->gpu_action_mask, vec->action_mask,
            (size_t)vec->total_agents * vec->action_mask_size * sizeof(unsigned char),
            cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();
}

void vec_create_threads(PuffeRL* pufferl) {
    VecEnv* vec = pufferl->vec;
    vec->buffer_states = (int*)calloc(vec->buffers, sizeof(int));
    vec->threads = (pthread_t*)calloc(vec->buffers, sizeof(pthread_t));
    VecThreadArg* args = (VecThreadArg*)calloc(vec->buffers, sizeof(VecThreadArg));
    vec->thread_args = args;
    vec->accum = (float*)calloc(vec->buffers * NUM_VEC_PROF, sizeof(float));
    for (int i = 0; i < vec->buffers; i++) {
        args[i].pufferl = pufferl;
        args[i].buf = i;
        pthread_create(&vec->threads[i], NULL, vec_thread_main, &args[i]);
    }
}

void vec_close(VecEnv* vec) {
    if (vec->gpu_env) {
#ifdef PUFFER_GPU_ENV_AVAILABLE
        puf_gpu_env_close(vec->gpu_envs);
#endif
    } else {
        Env* envs = vec->envs;
        __atomic_store_n(&vec->shutdown, 1, __ATOMIC_SEQ_CST);
        for (int i = 0; i < vec->buffers; i++) {
            pthread_join(vec->threads[i], NULL);
        }
        for (int i = 0; i < vec->size; i++) {
            Env* env = &envs[i];
            puf_close(env);
        }
#ifdef MY_VEC_CLOSE
        my_vec_close(envs);
#endif
        free(vec->envs);
    }
    free(vec->buffer_states);
    free(vec->threads);
    free(vec->thread_args);
    free(vec->accum);
    free(vec->buffer_env_starts);
    free(vec->buffer_env_counts);
    free(vec->bank_layout);

    cudaDeviceSynchronize();
    cudaFree(vec->gpu_observations);
    cudaFree(vec->gpu_actions);
    cudaFree(vec->gpu_rewards);
    cudaFree(vec->gpu_terminals);
    if (vec->gpu_env) {
        cudaFree(vec->gpu_log);
    } else {
        cudaFreeHost(vec->observations);
        cudaFreeHost(vec->actions);
        cudaFreeHost(vec->rewards);
        cudaFreeHost(vec->terminals);
    }
    if (vec->action_mask_size > 0) {
        cudaFree(vec->gpu_action_mask);
        cudaFreeHost(vec->action_mask);
    }

    free(vec);
}

void vec_log(VecEnv* vec, Dict* out, int clear) {
    if (vec->gpu_env) {
#ifdef PUFFER_GPU_ENV_AVAILABLE
        puf_gpu_env_log(vec->gpu_envs, vec->total_agents, vec->gpu_log, out, clear);
#endif
        return;
    }

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

// Zero advantages on frozen-bank rows so prio_replay never samples them. Frozen
// rollout rows hold actions/logprobs from the frozen policy; training the
// primary's PPO on them produces garbage ratios and poisoned gradients.
__global__ void zero_frozen_advantages_kernel(precision_t* advantages,
        int agents_per_buffer, int primary_per_buffer, int total_rows, int horizon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_rows * horizon;
    if (idx >= total) {
        return;
    }
    int row = idx / horizon;
    int rel = row % agents_per_buffer;
    if (rel >= primary_per_buffer) {
        advantages[idx] = from_float(0.0f);
    }
}

// Cooperative row copy (int4 when 16-byte aligned).
__device__ void copy_bytes(
        const char* src, char* dst,
        int src_row, int dst_row, int row_bytes) {
    const char* s = src + (int64_t)src_row * row_bytes;
    char* d = dst + (int64_t)dst_row * row_bytes;
    if (((uintptr_t)s & 15) == 0 && ((uintptr_t)d & 15) == 0 && row_bytes >= 16) {
        int n16 = row_bytes >> 4;
        const int4* __restrict__ s4 = reinterpret_cast<const int4*>(s);
        int4* __restrict__ d4 = reinterpret_cast<int4*>(d);
        for (int i = threadIdx.x; i < n16; i += blockDim.x) {
            d4[i] = s4[i];
        }
        for (int i = (n16 << 4) + threadIdx.x; i < row_bytes; i += blockDim.x) {
            d[i] = s[i];
        }
    } else {
        for (int i = threadIdx.x; i < row_bytes; i += blockDim.x) {
            d[i] = s[i];
        }
    }
}

#define SELECT_COPY_THREADS 256
// One block per minibatch segment: copy all base train fields for idx[mb].
// Row byte sizes are host-precomputed. Mask is an optional follow-up kernel.
// When initial_states.data is set (carry path), fold state gather here.
__global__ void select_copy(RolloutBuf rollouts, TrainGraph graph,
        int* idx, precision_t* advantages, float* mb_prio,
        int obs_rb, int act_rb, int lp_rb, int term_rb, int horizon) {
    int mb = blockIdx.x;
    int src_row = idx[mb];

    copy_bytes((const char*)rollouts.observations.data,
        (char*)graph.mb_obs.data, src_row, mb, obs_rb);
    copy_bytes((const char*)rollouts.actions.data,
        (char*)graph.mb_actions.data, src_row, mb, act_rb);
    copy_bytes((const char*)rollouts.logprobs.data,
        (char*)graph.mb_logprobs.data, src_row, mb, lp_rb);

    int srh = (int64_t)src_row * horizon;
    int drh = (int64_t)mb * horizon;
    precision_t* s_values = rollouts.values.data + srh;
    precision_t* s_adv = advantages + srh;
    precision_t* d_values = graph.mb_values.data + drh;
    precision_t* d_adv = graph.mb_advantages.data + drh;
    precision_t* d_returns = graph.mb_returns.data + drh;
    for (int i = threadIdx.x; i < horizon; i += blockDim.x) {
        precision_t val = s_values[i];
        precision_t adv = s_adv[i];
        d_values[i] = val;
        d_adv[i] = adv;
#ifndef PRECISION_FLOAT
        d_returns[i] = __hadd(val, adv);
#else
        d_returns[i] = val + adv;
#endif
    }

    if (threadIdx.x == 0) {
        graph.mb_prio.data[mb] = from_float(mb_prio[mb]);
    }
    copy_bytes((const char*)rollouts.terminals.data,
        (char*)graph.mb_terminals.data, src_row, mb, term_rb);

    if (rollouts.initial_states.data != nullptr) {
        int L = (int)rollouts.initial_states.shape[0];
        int H = (int)rollouts.initial_states.shape[2];
        int total = L * H;
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            int layer = i / H;
            int h = i % H;
            long src_idx = ((long)layer * rollouts.initial_states.shape[1] + src_row) * H + h;
            long dst_idx = ((long)layer * graph.mb_state.shape[1] + mb) * H + h;
            graph.mb_state.data[dst_idx] = rollouts.initial_states.data[src_idx];
        }
    }
}

__global__ void select_copy_mask(const char* src, char* dst, int* idx, int row_bytes) {
    int mb = blockIdx.x;
    copy_bytes(src, dst, idx[mb], mb, row_bytes);
}

// Transpose dims 0,1: [A, B, C] -> [B, A, C]. For 2D, pass C=1.
__global__ void transpose_102(precision_t* dst,
        precision_t* src, int A, int B, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = A * B * C;
    if (idx >= total) {
        return;
    }
    int a = idx / (B * C);
    int rem = idx % (B * C);
    int b = rem / C;
    int c = rem % C;
    dst[b * A * C + a * C + c] = src[idx];
}

// Sparse row scatter: dst[idx[i], :] = src[i, :]. One thread per element.
__global__ void scatter_rows(precision_t* dst, int* idx, const precision_t* src,
        int num_idx, int row_elems) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_idx * row_elems;
    if (i >= total) {
        return;
    }
    int mb = i / row_elems;
    int e = i % row_elems;
    dst[(int64_t)idx[mb] * row_elems + e] = src[i];
}

float cosine_annealing(float lr_base, float lr_min, long t, long T) {
    if (T == 0) {
        return lr_base;
    }
    float ratio = (double)t / (double)T;
    ratio = std::max(0.0f, std::min(1.0f, ratio));
    return lr_min + 0.5f * (lr_base - lr_min) * (1.0f + std::cos(M_PI * ratio));
}

__global__ void fill_precision_kernel(precision_t* dst, precision_t val, int n) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
            idx += blockDim.x * gridDim.x) {
        dst[idx] = val;
    }
}

__global__ void clamp_precision_kernel(precision_t* dst, float lo, float hi, int n) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
            idx += blockDim.x * gridDim.x) {
        float v = to_float(dst[idx]);
        dst[idx] = from_float(fminf(fmaxf(v, lo), hi));
    }
}

void train_impl(PuffeRL& pufferl, RolloutBuf* src_arg) {
    HypersT& hypers = pufferl.hypers;
    RolloutBuf src = src_arg ? *src_arg : pufferl.rollouts;
    cudaStream_t train_stream = pufferl.train_stream;

    cudaEventRecord(pufferl.profile.events[0], train_stream);  // pre-loop start

    // Transpose from rollout layout (T, B, ...) to train layout (B, T, ...)
    RolloutBuf& rollouts = pufferl.train_rollouts;
    PrecisionTensor& advantages_puf = pufferl.advantages_puf;

    int T = src.observations.shape[0];
    int B = src.observations.shape[1];
    int obs_size = (ndim(src.observations.shape) >= 3) ? src.observations.shape[2] : 1;
    int num_atns = (ndim(src.actions.shape) >= 3) ? src.actions.shape[2] : 1;

    transpose_102<<<grid_size(T * B * obs_size), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.observations.data, src.observations.data, T, B, obs_size);
    transpose_102<<<grid_size(T * B * num_atns), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.actions.data, src.actions.data, T, B, num_atns);
    transpose_102<<<grid_size(T * B), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.logprobs.data, src.logprobs.data, T, B, 1);
    transpose_102<<<grid_size(T * B), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.rewards.data, src.rewards.data, T, B, 1);
    transpose_102<<<grid_size(T * B), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.terminals.data, src.terminals.data, T, B, 1);
    transpose_102<<<grid_size(T * B), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.ratio.data, src.ratio.data, T, B, 1);
    transpose_102<<<grid_size(T * B), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.values.data, src.values.data, T, B, 1);
    if (src.action_mask.data != nullptr) {
        int mask_c = src.action_mask.shape[2];
        transpose_102<<<grid_size(T * B * mask_c), BLOCK_SIZE, 0, train_stream>>>(
            rollouts.action_mask.data, src.action_mask.data, T, B, mask_c);
    }

    // We hard-clamp rewards to -1, 1. Our envs are mostly designed to respect this range
    clamp_precision_kernel<<<grid_size(numel(rollouts.rewards.shape)), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.rewards.data, -1.0f, 1.0f, numel(rollouts.rewards.shape));

    // Treat rollout data as on-policy for advantage. PPO clipping still uses
    // behavior logprobs captured by the actor snapshot.
    fill_precision_kernel<<<grid_size(numel(rollouts.ratio.shape)), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.ratio.data, from_float(1.0f), numel(rollouts.ratio.shape));

    int minibatch_size = hypers.minibatch_size;
    int batch_size = hypers.total_agents * hypers.horizon;
    int minibatch_segments = minibatch_size / hypers.horizon;
    float prio_beta0 = hypers.prio_beta0;
    float prio_alpha = hypers.prio_alpha;
    bool anneal_lr = hypers.anneal_lr;
    int current_epoch = pufferl.epoch;

    Muon* muon = &pufferl.muon;
    int total_epochs = hypers.total_timesteps / batch_size;
    if (anneal_lr) {
        float lr_min = hypers.min_lr_ratio * hypers.lr;
        float lr = cosine_annealing(hypers.lr, lr_min, current_epoch, total_epochs);
        cudaMemcpy(muon->lr_ptr, &lr, sizeof(float), cudaMemcpyHostToDevice);
    }

    // Annealed entropy coefficient — same cosine shape as lr. With PG signal
    // alive, the entropy bonus that kept early-training exploratory becomes
    // load-bearing dead weight late in training; cosine-decay frees the policy
    // to commit harder on what it has already learned.
    float current_ent_coef = hypers.ent_coef;
    if (hypers.anneal_ent_coef) {
        float ent_min = hypers.min_ent_coef_ratio * hypers.ent_coef;
        current_ent_coef = cosine_annealing(hypers.ent_coef, ent_min,
                                            current_epoch, total_epochs);
    }
    // Device ptr for ent_coef so CUDA graphs do not bake host by-value.
    cudaMemcpyAsync(pufferl.ppo_bufs_puf.ent_coef.data, &current_ent_coef,
        sizeof(float), cudaMemcpyHostToDevice, train_stream);

    // Annealed priority exponent
    float anneal_beta = prio_beta0 + (1.0f - prio_beta0) * prio_alpha * (float)current_epoch/(float)total_epochs;
    TrainGraph& graph = pufferl.train_buf;
    cudaEventRecord(pufferl.profile.events[1], train_stream);  // pre-loop end

    // Advantage + prio CDF once per train; minibatches only re-sample indices.
    cudaMemsetAsync(advantages_puf.data, 0,
        numel(advantages_puf.shape) * sizeof(precision_t), train_stream);
    profile_begin("compute_advantage", hypers.profile);
    puff_advantage_cuda(rollouts.values, rollouts.rewards, rollouts.terminals,
        rollouts.ratio, advantages_puf, hypers.gamma, hypers.gae_lambda,
        hypers.vtrace_rho_clip, hypers.vtrace_c_clip, train_stream);
    if (pufferl.num_frozen_banks > 0) {
        int apb = hypers.total_agents / hypers.num_buffers;
        int rows = advantages_puf.shape[0];
        int horizon = advantages_puf.shape[1];
        int total = rows * horizon;
        zero_frozen_advantages_kernel<<<grid_size(total), BLOCK_SIZE, 0, train_stream>>>(
            advantages_puf.data, apb, pufferl.vec->bank_layout[1], rows, horizon);
    }
    profile_end(hypers.profile);

    profile_begin("compute_prio", hypers.profile);
    prio_build_cdf_cuda(advantages_puf, prio_alpha, pufferl.prio_bufs, train_stream);
    profile_end(hypers.profile);

    long* train_rng_offset = pufferl.rng_offset_puf.data + hypers.num_buffers;
    int total_minibatches = hypers.replay_ratio * batch_size / hypers.minibatch_size;
    for (int mb = 0; mb < total_minibatches; ++mb) {
        cudaEventRecord(pufferl.profile.events[2], train_stream);  // start of misc (overwritten each iter)

        profile_begin("compute_prio", hypers.profile);
        prio_sample_cuda(minibatch_segments, hypers.total_agents, anneal_beta,
            pufferl.prio_bufs, pufferl.seed, train_rng_offset, train_stream);
        profile_end(hypers.profile);

        profile_begin("train_select_and_copy", hypers.profile);
        RolloutBuf sel_src = rollouts;
        if (hypers.reset_state) {
            cudaMemsetAsync(graph.mb_state.data, 0,
                numel(graph.mb_state.shape) * sizeof(precision_t), train_stream);
            sel_src.initial_states = PrecisionTensor();
        } else if (src.initial_states.data != nullptr) {
            int slot = hypers.async ? pufferl.async_ready_slot : 0;
            sel_src.initial_states = initial_states_slot(src.initial_states, slot);
        } else {
            sel_src.initial_states = PrecisionTensor();
        }
        int mb_segs = pufferl.prio_bufs.idx.shape[0];
        int* sel_idx = pufferl.prio_bufs.idx.data;
        int pe = (int)sizeof(precision_t);
        int obs_rb = (numel(rollouts.observations.shape)
            / rollouts.observations.shape[0]) * pe;
        int act_rb = (numel(rollouts.actions.shape)
            / rollouts.actions.shape[0]) * pe;
        int lp_rb = (numel(rollouts.logprobs.shape)
            / rollouts.logprobs.shape[0]) * pe;
        int term_rb = (numel(rollouts.terminals.shape)
            / rollouts.terminals.shape[0]) * pe;
        int sel_horizon = rollouts.values.shape[1];
        select_copy<<<mb_segs, SELECT_COPY_THREADS, 0, train_stream>>>(
            sel_src, graph, sel_idx, advantages_puf.data, pufferl.prio_bufs.mb_prio.data,
            obs_rb, act_rb, lp_rb, term_rb, sel_horizon);
        if (graph.mb_action_mask.data != nullptr) {
            int mask_rb = (numel(rollouts.action_mask.shape)
                / rollouts.action_mask.shape[0]) * pe;
            select_copy_mask<<<mb_segs, SELECT_COPY_THREADS, 0, train_stream>>>(
                (const char*)rollouts.action_mask.data,
                (char*)graph.mb_action_mask.data, sel_idx, mask_rb);
        }
        profile_end(hypers.profile);

        cudaEventRecord(pufferl.profile.events[3], train_stream);  // end misc / start forward
        profile_begin("train_forward_backward", hypers.profile);
        if (pufferl.train_cudagraph != nullptr) {
            cudaGraphLaunch(pufferl.train_cudagraph, train_stream);
        } else {
            bool capturing = hypers.cudagraphs;
            if (capturing) {
                assert(cudaStreamBeginCapture(train_stream, cudaStreamCaptureModeGlobal) == cudaSuccess
                        && "cudaStreamBeginCapture failed");
            }

            cudaStream_t stream = train_stream;
            PrecisionTensor obs_puf = graph.mb_obs;
            PrecisionTensor state_puf = graph.mb_state;
            PrecisionTensor terminals_puf = graph.mb_terminals;
            PrecisionTensor dec_puf = policy_forward_train(&pufferl.policy, pufferl.weights,
                pufferl.train_activations, obs_puf, state_puf, terminals_puf, stream);
            DecoderWeights* dw_train = (DecoderWeights*)pufferl.weights.decoder;
            PrecisionTensor p_logstd;
            if (dw_train->continuous) {
                p_logstd = dw_train->logstd;
            }

            ppo_loss_fwd_bwd(dec_puf, p_logstd, graph,
                pufferl.act_sizes_puf, pufferl.losses_puf,
                hypers.clip_coef, hypers.vf_clip_coef, hypers.vf_coef,
                pufferl.ppo_bufs_puf.ent_coef.data,
                pufferl.ppo_bufs_puf, pufferl.is_continuous, stream);

            FloatTensor grad_logits_puf = pufferl.ppo_bufs_puf.grad_logits;
            FloatTensor grad_logstd_puf = pufferl.is_continuous ? pufferl.ppo_bufs_puf.grad_logstd : FloatTensor();
            FloatTensor grad_values_puf = pufferl.ppo_bufs_puf.grad_values;
            policy_backward(&pufferl.policy, pufferl.weights, pufferl.train_activations,
                grad_logits_puf, grad_logstd_puf, grad_values_puf, stream);

            if (pufferl.nccl_comm != nullptr && hypers.world_size > 1) {
                ncclAllReduce(pufferl.grad_puf.data, pufferl.grad_puf.data,
                    numel(pufferl.grad_puf.shape), NCCL_PRECISION, ncclAvg,
                    pufferl.nccl_comm, stream);
            }
            muon_step(&pufferl.muon, pufferl.master_weights, pufferl.grad_puf, hypers.max_grad_norm, stream);
            if (USE_BF16) {
                int n = numel(pufferl.param_puf.shape);
                cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
                    pufferl.param_puf.data, pufferl.master_weights.data, n);
            }
            if (capturing) {
                cudaGraph_t _graph;
                cudaError_t err = cudaStreamEndCapture(train_stream, &_graph);
                if (err != cudaSuccess) {
                    fprintf(stderr, "cudaStreamEndCapture train failed: %s\n", cudaGetErrorString(err));
                    abort();
                }
                assert(cudaGraphInstantiate(
                    &pufferl.train_cudagraph, _graph, 0) == cudaSuccess
                    && "cudaGraphInstantiate failed");
                assert(cudaGraphDestroy(_graph) == cudaSuccess && "cudaGraphDestroy failed");
                // Capture records without executing; run once so this step has effects.
                cudaGraphLaunch(pufferl.train_cudagraph, train_stream);
                cudaDeviceSynchronize();
            }
        }
        profile_end(hypers.profile);

        // This version is consistent with PufferLib 3.0. One of the major algorithmic
        // questions remaining is how and when to update value and advantage estimates.
        int num_idx = numel(pufferl.prio_bufs.idx.shape);
        int ratio_row = numel(graph.mb_ratio.shape) / graph.mb_ratio.shape[0];
        int value_row = graph.mb_newvalue.shape[1];
        scatter_rows<<<grid_size(num_idx * ratio_row), BLOCK_SIZE, 0, train_stream>>>(
            rollouts.ratio.data, pufferl.prio_bufs.idx.data,
            graph.mb_ratio.data, num_idx, ratio_row);
        scatter_rows<<<grid_size(num_idx * value_row), BLOCK_SIZE, 0, train_stream>>>(
            rollouts.values.data, pufferl.prio_bufs.idx.data,
            graph.mb_newvalue.data, num_idx, value_row);
        cudaEventRecord(pufferl.profile.events[4], train_stream);  // end forward
    }

    cudaStreamSynchronize(train_stream);

    if (total_minibatches > 0) {
        float ms;
        // Pre-loop setup (transpose, advantage, allocs)
        cudaEventElapsedTime(&ms, pufferl.profile.events[0], pufferl.profile.events[1]);
        pufferl.profile.accum[PROF_TRAIN_MISC] += ms;
        // In-loop misc (last iteration, representative) scaled by count
        cudaEventElapsedTime(&ms, pufferl.profile.events[2], pufferl.profile.events[3]);
        pufferl.profile.accum[PROF_TRAIN_MISC] += ms * total_minibatches;
        // In-loop forward (last iteration, representative) scaled by count
        cudaEventElapsedTime(&ms, pufferl.profile.events[3], pufferl.profile.events[4]);
        pufferl.profile.accum[PROF_TRAIN_MODEL] += ms * total_minibatches;
    }
    pufferl.epoch += 1;
}


// Load frozen bank weights (flat fp32 checkpoint). Safe between rollouts —
// graphs hold the pointer, not a copy of the data.
// --- Checkpoint I/O (load/save weights) ---
void mkdir_p(const char* path) {
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char* p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            if (mkdir(tmp, 0777) != 0 && errno != EEXIST) {
                fprintf(stderr, "failed to create directory %s: %s\n", tmp, strerror(errno));
                exit(1);
            }
            *p = '/';
        }
    }
    if (mkdir(tmp, 0777) != 0 && errno != EEXIST) {
        fprintf(stderr, "failed to create directory %s: %s\n", tmp, strerror(errno));
        exit(1);
    }
}

void puf_find_latest_checkpoint(const char* dir,
        char* out, size_t out_size, time_t* best_time) {
    DIR* dp = opendir(dir);
    if (!dp) {
        return;
    }

    struct dirent* ent = NULL;
    while ((ent = readdir(dp))) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
            continue;
        }

        char path[4096];
        snprintf(path, sizeof(path), "%s/%s", dir, ent->d_name);

        struct stat st;
        if (stat(path, &st) != 0) {
            continue;
        }

        if (S_ISDIR(st.st_mode)) {
            puf_find_latest_checkpoint(path, out, out_size, best_time);
        } else if (S_ISREG(st.st_mode)) {
            size_t plen = strlen(path);
            if (plen < 4 || strcmp(path + plen - 4, ".bin") != 0) continue;
            if (st.st_ctime < *best_time) continue;
            *best_time = st.st_ctime;
            snprintf(out, out_size, "%s", path);
        }
    }

    closedir(dp);
}

const char* puf_checkpoint_path_key(Ini* ini, const char* key,
        char* out, size_t out_size) {
    const char* load_path = puf_ini_get_str(ini, "base", key);
    if (!load_path || strcmp(load_path, "None") == 0) {
        return NULL;
    }

    if (strcmp(load_path, "latest") != 0) {
        return load_path;
    }

    char root[2048];
    snprintf(root, sizeof(root), "%s/%s",
        puf_ini_get_str(ini, "base", "checkpoint_dir"),
        puf_ini_get_str(ini, "base", "env_name"));

    out[0] = 0;
    time_t best_time = 0;
    puf_find_latest_checkpoint(root, out, out_size, &best_time);
    if (!out[0]) {
        fprintf(stderr, "no .bin checkpoints found in %s\n", root);
        exit(1);
    }
    return out;
}

void puf_save_weights(PuffeRL* p, const char* path) {
    int64_t nbytes = numel(p->master_weights.shape) * sizeof(float);
    char* buf = (char*)malloc(nbytes);
    cudaMemcpy(buf, p->master_weights.data, nbytes, cudaMemcpyDeviceToHost);
    char tmp[4096];
    snprintf(tmp, sizeof(tmp), "%s.tmp.%d", path, getpid());
    FILE* fp = fopen(tmp, "wb");
    if (!fp) {
        fprintf(stderr, "failed to open %s for writing\n", tmp);
        free(buf);
        exit(1);
    }
    if (fwrite(buf, 1, nbytes, fp) != (size_t)nbytes) {
        fprintf(stderr, "failed to write weights to %s\n", tmp);
        fclose(fp);
        free(buf);
        exit(1);
    }
    fclose(fp);
    free(buf);
    if (rename(tmp, path) != 0) {
        fprintf(stderr, "failed to publish weights to %s\n", path);
        exit(1);
    }
}

void puf_load_weights_into(FloatTensor dst, PrecisionTensor params,
        cudaStream_t stream, const char* path) {
    int64_t nbytes = numel(dst.shape) * sizeof(float);
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "failed to open %s for reading\n", path);
        exit(1);
    }
    char* buf = (char*)malloc(nbytes);
    size_t nread = fread(buf, 1, nbytes, fp);
    fclose(fp);
    if ((int64_t)nread != nbytes) {
        fprintf(stderr, "failed to read weights from %s\n", path);
        free(buf);
        exit(1);
    }
    cudaMemcpy(dst.data, buf, nbytes, cudaMemcpyHostToDevice);
    free(buf);
    if (USE_BF16) {
        int n = numel(params.shape);
        cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(params.data, dst.data, n);
    }
}

void pufferl_load_frozen_bank(PuffeRL* pufferl, int bank_idx, const char* path) {
    if (bank_idx < 0 || bank_idx >= pufferl->num_frozen_banks) {
        fprintf(stderr, "pufferl_load_frozen_bank: bank_idx %d out of range\n", bank_idx);
        exit(1);
    }
    WeightBank* bank = &pufferl->frozen_banks[bank_idx];
    puf_load_weights_into(bank->master_weights, bank->param_puf,
        pufferl->default_stream, path);
    cudaDeviceSynchronize();
}

// Die on OOM in the train worker. Sweep observes failed workers as bad samples and continues.
void create_allocator_or_die(const char* name, Allocator* alloc) {
    cudaError_t err = alloc_create(alloc);
    if (err != cudaSuccess) {
        fprintf(stderr, "create_pufferl: alloc_create(%s) failed for %ld bytes: %s\n",
            name, alloc->total_bytes, cudaGetErrorString(err));
        exit(1);
    }
}

// Zero primary + frozen per-buffer RNN states (used at reset / reset_state).
void zero_all_buffer_states(PuffeRL* p, cudaStream_t stream) {
    for (int i = 0; i < p->hypers.num_buffers; i++) {
        cudaMemsetAsync(p->buffer_states[i].data, 0,
            numel(p->buffer_states[i].shape) * sizeof(precision_t), stream);
    }
    for (int b = 0; b < p->num_frozen_banks; b++) {
        for (int i = 0; i < p->hypers.num_buffers; i++) {
            PrecisionTensor* st = &p->frozen_banks[b].buffer_states[i];
            cudaMemsetAsync(st->data, 0, numel(st->shape) * sizeof(precision_t), stream);
        }
    }
}

void create_buffer_streams(PuffeRL* pufferl) {
    int n = pufferl->hypers.num_buffers;
    pufferl->streams = (cudaStream_t*)calloc(n, sizeof(cudaStream_t));
    for (int i = 0; i < n; i++) {
        if (pufferl->hypers.async) {
            cudaStreamCreateWithFlags(&pufferl->streams[i], cudaStreamNonBlocking);
        } else {
            cudaStreamCreate(&pufferl->streams[i]);
        }
    }
}

double wall_clock() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Build trainer from ini + rank context. (Former Python create_pufferl + impl.)
PuffeRL* create_pufferl(Ini* ini, TrainContext* ctx) {
    HypersT hypers = {0};
    hypers.total_agents = puf_ini_get(ini, "vec", "total_agents");
    hypers.num_buffers = puf_ini_get(ini, "vec", "num_buffers");
    hypers.num_threads = puf_ini_get(ini, "vec", "num_threads");
    hypers.horizon = puf_ini_get(ini, "train", "horizon");
    hypers.hidden_size = puf_ini_get(ini, "policy", "hidden_size");
    hypers.num_layers = puf_ini_get(ini, "policy", "num_layers");
    hypers.lr = puf_ini_get(ini, "train", "learning_rate");
    hypers.min_lr_ratio = puf_ini_get(ini, "train", "min_lr_ratio");
    hypers.anneal_lr = puf_ini_get(ini, "train", "anneal_lr");
    hypers.momentum = puf_ini_get(ini, "train", "momentum");
    hypers.minibatch_size = puf_ini_get(ini, "train", "minibatch_size");
    hypers.replay_ratio = puf_ini_get(ini, "train", "replay_ratio");
    hypers.total_timesteps = puf_ini_get(ini, "train", "total_timesteps");
    hypers.max_grad_norm = puf_ini_get(ini, "train", "max_grad_norm");
    hypers.clip_coef = puf_ini_get(ini, "train", "clip_coef");
    hypers.vf_clip_coef = puf_ini_get(ini, "train", "vf_clip_coef");
    hypers.vf_coef = puf_ini_get(ini, "train", "vf_coef");
    hypers.ent_coef = puf_ini_get(ini, "train", "ent_coef");
    hypers.min_ent_coef_ratio = puf_ini_get(ini, "train", "min_ent_coef_ratio");
    hypers.anneal_ent_coef = puf_ini_get(ini, "train", "anneal_ent_coef");
    hypers.gamma = puf_ini_get(ini, "train", "gamma");
    hypers.gae_lambda = puf_ini_get(ini, "train", "gae_lambda");
    hypers.vtrace_rho_clip = puf_ini_get(ini, "train", "vtrace_rho_clip");
    hypers.vtrace_c_clip = puf_ini_get(ini, "train", "vtrace_c_clip");
    hypers.prio_alpha = puf_ini_get(ini, "train", "prio_alpha");
    hypers.prio_beta0 = puf_ini_get(ini, "train", "prio_beta0");
    hypers.async = puf_ini_get(ini, "base", "async");
    hypers.reset_state = puf_ini_get(ini, "base", "reset_state");
    hypers.cudagraphs = puf_ini_get(ini, "base", "cudagraphs") >= 0;
    hypers.profile = puf_ini_get(ini, "base", "profile");
    hypers.seed = puf_ini_get(ini, "base", "seed");
    hypers.rank = ctx->rank;
    hypers.world_size = ctx->world_size;
    hypers.gpu_id = ctx->gpu_id;

    Dict vec_kwargs = {0};
    dict_copy(&vec_kwargs, puf_ini_section(ini, "vec", 0));
    Dict* env_kwargs = puf_ini_section(ini, "env", 0);
    ncclUniqueId* nccl_id = ctx->nccl_id;

    PuffeRL* pufferl = new PuffeRL();
    pufferl->hypers = hypers;
    pufferl->nccl_comm = nullptr;
    pufferl->default_stream = 0;
    snprintf(pufferl->env_name, sizeof(pufferl->env_name), "%s", PUFFER_ENV_NAME);

    cudaSetDevice(hypers.gpu_id);
    cublas_init_handle();

    // Multi-GPU: initialize NCCL
    if (hypers.world_size > 1) {
        ncclCommInitRank(&pufferl->nccl_comm, hypers.world_size, *nccl_id, hypers.rank);
        printf("Rank %d/%d: NCCL initialized\n", hypers.rank, hypers.world_size);
    }

    ulong seed = hypers.seed + hypers.rank;
    pufferl->seed = seed;

    // Create environments and bind GPU buffers into EnvBuf views.
    VecEnv* vec = vec_create(&vec_kwargs, env_kwargs);
    pufferl->vec = vec;
    int n_agents = vec->total_agents;
    EnvBuf& env = pufferl->env;
    env.obs = { .data = vec->gpu_observations, .shape = {n_agents, OBS_SIZE} };
    env.actions = { .data = (float*)vec->gpu_actions, .shape = {n_agents, NUM_ATNS} };
    env.rewards = { .data = (float*)vec->gpu_rewards, .shape = {n_agents} };
    env.terminals = { .data = (float*)vec->gpu_terminals, .shape = {n_agents} };
    if (vec->action_mask_size > 0) {
        env.action_mask = { .data = vec->gpu_action_mask,
                            .shape = {n_agents, vec->action_mask_size} };
    } else {
        env.action_mask = { .data = nullptr, .shape = {0} };
    }

    // Sanity check action space
    int num_action_heads = pufferl->env.actions.shape[1];
    int act_sizes[] = ACT_SIZES;
    int act_n = 0;
    int num_continuous = 0;
    int num_discrete = 0;
    for (int i = 0; i < num_action_heads; i++) {
        int val = act_sizes[i];
        if (val == 1) {
            num_continuous++;
        } else {
            num_discrete++;
        }
        act_n += val;
    }
    assert((num_continuous == 0 || num_discrete == 0) &&
        "Mixed continuous/discrete action spaces not supported");
    pufferl->is_continuous = (num_continuous > 0);
    if (pufferl->is_continuous) {
        printf("Detected continuous action space with %d dimensions\n", num_action_heads);
    } else {
        printf("Detected discrete action space with %d heads\n", num_action_heads);
    }

    // Create profiling events
    for (int i = 0; i < (int)(sizeof(pufferl->profile.events) / sizeof(pufferl->profile.events[0])); i++) {
        cudaEventCreate(&pufferl->profile.events[i]);
    }
    pufferl->profile.rollout_horizon = hypers.horizon;
    pufferl->profile.rollout_gpu_start =
        (cudaEvent_t*)calloc((size_t)hypers.horizon, sizeof(cudaEvent_t));
    pufferl->profile.rollout_gpu_end =
        (cudaEvent_t*)calloc((size_t)hypers.horizon, sizeof(cudaEvent_t));
    pufferl->profile.rollout_env_end =
        (cudaEvent_t*)calloc((size_t)hypers.horizon, sizeof(cudaEvent_t));
    for (int i = 0; i < hypers.horizon; i++) {
        cudaEventCreate(&pufferl->profile.rollout_gpu_start[i]);
        cudaEventCreate(&pufferl->profile.rollout_gpu_end[i]);
        cudaEventCreate(&pufferl->profile.rollout_env_end[i]);
    }
    memset(pufferl->profile.accum, 0, sizeof(pufferl->profile.accum));
    nvmlInit();
    nvmlDeviceGetHandleByIndex(hypers.gpu_id, &pufferl->nvml_device);

    // Create policy
    int input_size = pufferl->env.obs.shape[1];
    int hidden_size = hypers.hidden_size;
    int num_layers = hypers.num_layers;
    bool is_continuous = pufferl->is_continuous;
    int decoder_output_size = is_continuous ? num_action_heads : act_n;
    int minibatch_segments = hypers.minibatch_size / hypers.horizon;
    int inf_batch = vec->total_agents / hypers.num_buffers;
    int B_TT = minibatch_segments * hypers.horizon;
    int horizon = hypers.horizon;
    int total_agents = vec->total_agents;
    int batch = total_agents / hypers.num_buffers;
    int num_buffers = hypers.num_buffers;

    pufferl->policy = build_policy(pufferl->env_name, input_size, hidden_size,
        num_layers, decoder_output_size, act_n, is_continuous, hypers.horizon);

    // Dedicated learner stream (always non-default; nonblocking when async).
    if (hypers.async) {
        cudaStreamCreateWithFlags(&pufferl->train_stream, cudaStreamNonBlocking);
    } else {
        cudaStreamCreate(&pufferl->train_stream);
    }

    // Create and allocate params
    Allocator* params = &pufferl->params_alloc;
    Allocator* acts = &pufferl->activations_alloc;
    Allocator* grads = &pufferl->grads_alloc;

    // Buffers for weights, grads, and activations
    pufferl->weights = policy_weights_create(&pufferl->policy, params);
    if (hypers.async) {
        pufferl->actor_weights = policy_weights_create(&pufferl->policy, &pufferl->actor_params_alloc);
    }
    pufferl->train_activations = policy_reg_train(&pufferl->policy, pufferl->weights, acts, grads, B_TT);
    pufferl->buffer_activations = (PolicyActivations*)calloc(num_buffers, sizeof(PolicyActivations));
    pufferl->buffer_states = (PrecisionTensor*)calloc(num_buffers, sizeof(PrecisionTensor));
    for (int i = 0; i < num_buffers; i++) {
        pufferl->buffer_activations[i] = policy_reg_rollout(
            &pufferl->policy, pufferl->weights, acts, inf_batch);
        pufferl->buffer_states[i] = {
            .shape = {num_layers, batch, hidden_size},
        };
        alloc_register(acts, &pufferl->buffer_states[i]);
    }
    int mask_size = pufferl->vec->action_mask_size;
    int rollout_horizon = hypers.async ? 2 * horizon : horizon;
    register_rollout_buffers(pufferl->rollouts,
        acts, rollout_horizon, total_agents, input_size, num_action_heads, mask_size);
    // Carry path: per-slot initial RNN states. reset_state zeros mb_state instead.
    if (!hypers.reset_state) {
        int slots = hypers.async ? 2 : 1;
        pufferl->rollouts.initial_states = {
            .shape = {slots, num_layers, total_agents, hidden_size}};
        alloc_register(acts, &pufferl->rollouts.initial_states);
    }
    register_train_buffers(pufferl->train_buf,
        acts, minibatch_segments, horizon, input_size,
        hidden_size, num_action_heads, num_layers, mask_size);
    register_rollout_buffers(pufferl->train_rollouts,
        acts, total_agents, horizon, input_size, num_action_heads, mask_size);
    register_ppo_buffers(pufferl->ppo_bufs_puf,
        acts, minibatch_segments, hypers.horizon, decoder_output_size, is_continuous);
    register_prio_buffers(pufferl->prio_bufs,
        acts, hypers.total_agents, minibatch_segments);

    // Extra cuda buffers just reuse activ allocator
    pufferl->rng_offset_puf = {.shape = {num_buffers + 1}};
    alloc_register(acts, &pufferl->rng_offset_puf);

    pufferl->act_sizes_puf  = {.shape = {num_action_heads}};
    alloc_register(acts, &pufferl->act_sizes_puf);

    pufferl->losses_puf = {.shape = {NUM_LOSSES}};
    alloc_register(acts, &pufferl->losses_puf);

    pufferl->advantages_puf = {.shape = {total_agents, horizon}};
    alloc_register(acts, &pufferl->advantages_puf);

    muon_init(&pufferl->muon, params, hypers.lr, hypers.momentum, 0.0, acts);

    // All buffers allocated here
    create_allocator_or_die("params", params);
    if (hypers.async) {
        create_allocator_or_die("actor_params", &pufferl->actor_params_alloc);
    }
    create_allocator_or_die("grads", grads);
    create_allocator_or_die("acts", acts);

    pufferl->grad_puf = {.data = (precision_t*)grads->mem, .shape = {grads->total_elems}};
    pufferl->param_puf = {.data = (precision_t*)params->mem, .shape = {params->total_elems}};
    if (hypers.async) {
        pufferl->actor_param_puf = {
            .data = (precision_t*)pufferl->actor_params_alloc.mem,
            .shape = {pufferl->actor_params_alloc.total_elems},
        };
    }

    ulong init_seed = hypers.seed;
    policy_init_weights(&pufferl->policy, pufferl->weights, &init_seed, pufferl->default_stream);
    pufferl->master_weights = {.data = (float*)pufferl->param_puf.data, .shape = {params->total_elems}};
    if (USE_BF16) {
        pufferl->master_weights = {.shape = {params->total_elems}};
        cudaMalloc(&pufferl->master_weights.data, params->total_elems * sizeof(float));
        int n = numel(pufferl->param_puf.shape);
        cast<<<grid_size(n), BLOCK_SIZE, 0, pufferl->default_stream>>>(
            pufferl->master_weights.data, pufferl->param_puf.data, n);
    }
    if (hypers.async) {
        puf_copy(&pufferl->actor_param_puf, &pufferl->param_puf, pufferl->default_stream);
        cudaStreamSynchronize(pufferl->default_stream);
    }

    // Per-buffer persistent RNG states
    int agents_per_buf = total_agents / num_buffers;
    pufferl->rng_states = (curandStatePhilox4_32_10_t**)calloc(num_buffers, sizeof(curandStatePhilox4_32_10_t*));
    for (int i = 0; i < num_buffers; i++) {
        cudaMalloc(&pufferl->rng_states[i], agents_per_buf * sizeof(curandStatePhilox4_32_10_t));
        rng_init<<<grid_size(agents_per_buf), BLOCK_SIZE>>>(
            pufferl->rng_states[i], pufferl->seed + i, agents_per_buf);
    }

    // Post-create initialization
    cudaMemcpy(pufferl->act_sizes_puf.data, act_sizes, num_action_heads * sizeof(int),
        cudaMemcpyHostToDevice);
    cudaMemset(pufferl->losses_puf.data, 0, NUM_LOSSES * sizeof(float));
    float one = 1.0f;
    cudaMemcpy(pufferl->ppo_bufs_puf.grad_loss.data, &one, sizeof(float), cudaMemcpyHostToDevice);
    muon_post_create(&pufferl->muon);

    // Frozen banks (selfplay/match opponents): rollout-only policy.
    // Arch comes from vec.frozen_bank_*; required when banks > 0.
    int num_frozen = vec->num_banks - 1;
    int frozen_hidden = (int)dict_get(&vec_kwargs, "frozen_bank_hidden_size");
    int frozen_layers = (int)dict_get(&vec_kwargs, "frozen_bank_num_layers");
    if (num_frozen > 0 && (frozen_hidden <= 0 || frozen_layers <= 0)) {
        fprintf(stderr,
            "create_pufferl: num_frozen_banks=%d requires frozen_bank_hidden_size and "
            "frozen_bank_num_layers > 0 (got %d, %d)\n",
            num_frozen, frozen_hidden, frozen_layers);
        exit(1);
    }
    pufferl->num_frozen_banks = num_frozen;
    pufferl->frozen_banks = num_frozen
        ? (WeightBank*)calloc(num_frozen, sizeof(WeightBank)) : nullptr;
    for (int b = 0; b < num_frozen; b++) {
        int slice = vec->bank_layout[b + 2] - vec->bank_layout[b + 1];
        if (slice <= 0) {
            fprintf(stderr, "create_pufferl: frozen bank %d has no agents\n", b);
            exit(1);
        }
        WeightBank* bank = &pufferl->frozen_banks[b];
        bank->policy = build_policy(pufferl->env_name, input_size, frozen_hidden,
            frozen_layers, decoder_output_size, act_n, is_continuous, hypers.horizon);
        bank->weights = policy_weights_create(&bank->policy, &bank->params_alloc);
        bank->buffer_activations = (PolicyActivations*)calloc(num_buffers, sizeof(PolicyActivations));
        bank->buffer_states = (PrecisionTensor*)calloc(num_buffers, sizeof(PrecisionTensor));
        for (int i = 0; i < num_buffers; i++) {
            bank->buffer_activations[i] = policy_reg_rollout(
                &bank->policy, bank->weights, &bank->acts_alloc, slice);
            bank->buffer_states[i] = {.shape = {frozen_layers, slice, frozen_hidden}};
            alloc_register(&bank->acts_alloc, &bank->buffer_states[i]);
        }
        create_allocator_or_die("frozen_params", &bank->params_alloc);
        create_allocator_or_die("frozen_acts", &bank->acts_alloc);
        bank->param_puf = {
            .data = (precision_t*)bank->params_alloc.mem,
            .shape = {bank->params_alloc.total_elems},
        };
        if (USE_BF16) {
            bank->master_weights = {.shape = {bank->params_alloc.total_elems}};
            cudaMalloc(&bank->master_weights.data,
                bank->params_alloc.total_elems * sizeof(float));
        } else {
            bank->master_weights = {
                .data = (float*)bank->param_puf.data,
                .shape = {bank->params_alloc.total_elems},
            };
        }
    }

    // CUDA graphs: allocate graph array only; capture on first real use.
    if (hypers.cudagraphs) {
        int rollout_graph_slots = hypers.async ? 2 : 1;
        pufferl->fused_rollout_cudagraphs = (cudaGraphExec_t*)calloc(
            rollout_graph_slots * horizon * num_buffers, sizeof(cudaGraphExec_t));
    }
    create_buffer_streams(pufferl);

    if (!vec->gpu_env) {
        vec_create_threads(pufferl);
    }
    vec_reset(vec);

    if (hypers.profile) {
        cudaDeviceSynchronize();
        cudaProfilerStart();
    }

    double now = wall_clock();
    pufferl->start_time = now;
    pufferl->last_log_time = now;
    pufferl->last_log_step = 0;

    dict_clear(&vec_kwargs);
    return pufferl;
}

void close_pufferl(PuffeRL* p) {
    PuffeRL& pufferl = *p;
    cudaDeviceSynchronize();
    if (pufferl.hypers.profile) {
        cudaProfilerStop();
    }

    if (pufferl.train_cudagraph != nullptr) {
        cudaGraphExecDestroy(pufferl.train_cudagraph);
    }
    if (pufferl.fused_rollout_cudagraphs != nullptr) {
        int slots = pufferl.hypers.async ? 2 : 1;
        int num_graphs = slots * pufferl.hypers.horizon * pufferl.hypers.num_buffers;
        for (int i = 0; i < num_graphs; i++) {
            if (pufferl.fused_rollout_cudagraphs[i] != nullptr) {
                cudaGraphExecDestroy(pufferl.fused_rollout_cudagraphs[i]);
            }
        }
    }

    policy_weights_free(&pufferl.policy, &pufferl.weights);
    if (pufferl.hypers.async) {
        policy_weights_free(&pufferl.policy, &pufferl.actor_weights);
    }
    policy_activations_free(&pufferl.policy, pufferl.train_activations);
    for (int buf = 0; buf < pufferl.hypers.num_buffers; buf++) {
        policy_activations_free(&pufferl.policy, pufferl.buffer_activations[buf]);
    }

    for (int i = 0; i < pufferl.hypers.num_buffers; i++) {
        cudaFree(pufferl.rng_states[i]);
    }
    free(pufferl.rng_states);

    if (USE_BF16) {
        cudaFree(pufferl.master_weights.data);
    }

    alloc_free(&pufferl.params_alloc);
    alloc_free(&pufferl.actor_params_alloc);
    alloc_free(&pufferl.grads_alloc);
    alloc_free(&pufferl.activations_alloc);

    if (pufferl.train_stream) {
        cudaStreamDestroy(pufferl.train_stream);
    }
    for (int i = 0; i < pufferl.hypers.num_buffers; i++) {
        cudaStreamDestroy(pufferl.streams[i]);
    }
    for (int i = 0; i < (int)(sizeof(pufferl.profile.events) / sizeof(pufferl.profile.events[0])); i++) {
        cudaEventDestroy(pufferl.profile.events[i]);
    }
    for (int i = 0; i < pufferl.profile.rollout_horizon; i++) {
        cudaEventDestroy(pufferl.profile.rollout_gpu_start[i]);
        cudaEventDestroy(pufferl.profile.rollout_gpu_end[i]);
        cudaEventDestroy(pufferl.profile.rollout_env_end[i]);
    }
    free(pufferl.profile.rollout_gpu_start);
    free(pufferl.profile.rollout_gpu_end);
    free(pufferl.profile.rollout_env_end);
    nvmlShutdown();

    vec_close(pufferl.vec);

    free(pufferl.buffer_states);
    free(pufferl.buffer_activations);
    free(pufferl.fused_rollout_cudagraphs);
    free(pufferl.streams);

    for (int b = 0; b < pufferl.num_frozen_banks; b++) {
        WeightBank* bank = &pufferl.frozen_banks[b];
        policy_weights_free(&bank->policy, &bank->weights);
        if (bank->buffer_activations) {
            for (int i = 0; i < pufferl.hypers.num_buffers; i++) {
                policy_activations_free(&bank->policy, bank->buffer_activations[i]);
            }
            free(bank->buffer_activations);
        }
        free(bank->buffer_states);
        alloc_free(&bank->params_alloc);
        alloc_free(&bank->acts_alloc);
        if (USE_BF16 && bank->master_weights.data) {
            cudaFree(bank->master_weights.data);
        }
    }
    free(pufferl.frozen_banks);

    if (pufferl.nccl_comm != nullptr) {
        ncclCommDestroy(pufferl.nccl_comm);
    }
    delete p;
}

DictItem* puf_log_find(Dict* dict, const char* key) {
    for (int i = 0; i < dict->size; i++) {
        if (strcmp(dict->items[i].key, key) == 0) {
            return &dict->items[i];
        }
    }
    return NULL;
}

double puf_log_get_or(Dict* dict, const char* key, double fallback) {
    DictItem* item = puf_log_find(dict, key);
    return item ? item->value : fallback;
}

// --- Dashboard: fixed 80-col box, TTY home/clear, compact fallback on tiny terminals ---
static int puf_dashboard_tty = 0;
static int puf_dashboard_last_rows = 0;
static int puf_dashboard_last_cols = 0;
static int puf_dashboard_frame = 0;

#define PUF_DASH_W 80
#define PUF_DASH_BASE_ROWS 12
#define PUF_DASH_MAX_USER_ROWS 15

// Colors no-op when not a TTY.
#define PUF_A  (puf_dashboard_tty ? "\033[96m" : "")
#define PUF_W  (puf_dashboard_tty ? "\033[97m" : "")
#define PUF_G  (puf_dashboard_tty ? "\033[90m" : "")
#define PUF_R  (puf_dashboard_tty ? "\033[0m" : "")

static void dash_eol(void) {
    if (puf_dashboard_tty) printf("\033[K");
    putchar('\n');
}

static void dash_end(void) {
    if (puf_dashboard_tty) printf("\033[J\033[?2026l");
    fflush(stdout);
}

// Right-align into w cols (truncate if long). Unit letters gray when value looks numeric.
static void dash_cell(const char* s, int w) {
    int n = (int)strlen(s);
    if (n > w) n = w;
    int numeric = 0;
    for (int i = 0; i < n; i++) numeric |= (s[i] >= '0' && s[i] <= '9');
    printf("%s%*s", PUF_W, w - n, "");
    for (int i = 0; i < n; i++) {
        int unit = numeric && strchr("%KMBTGdhms", s[i]);
        printf("%s%c", unit ? PUF_G : PUF_W, s[i]);
    }
    printf("%s", PUF_R);
}

static void dash_rule(const char* left, const char* right) {
    printf("%s%s", PUF_W, left);
    for (int i = 0; i < PUF_DASH_W - 2; i++) printf("─");
    printf("%s%s", right, PUF_R);
    dash_eol();
}

// Stats | Perf | Losses  — fixed field widths sum to 80 with borders/spacing.
static void dash_row(const char* a, const char* av,
        const char* b, const char* bt, const char* bp,
        const char* c, const char* cv) {
    printf("%s│%s %s%-9.9s%s ", PUF_W, PUF_R, PUF_A, a, PUF_R);
    dash_cell(av, 13);
    printf("    %s%-12.12s%s ", PUF_A, b, PUF_R);
    dash_cell(bt, 6);
    putchar(' ');
    dash_cell(bp, 4);
    printf("    %s%-10.10s%s ", PUF_A, c, PUF_R);
    dash_cell(cv, 7);
    printf("    %s│%s", PUF_W, PUF_R);
    dash_eol();
}

static void dash_abbrev(char* out, size_t n, double val) {
    const char* suf[] = {"", "K", "M", "B", "T"};
    int i = 0;
    while (val >= 1000.0 && i < 4) { val /= 1000.0; i++; }
    snprintf(out, n, "%.1f%s", val, suf[i]);
}

static void dash_duration(char* out, size_t n, double sec) {
    if (sec < 0) sec = 0;
    if (sec < 1.0) { snprintf(out, n, "%.0fms", sec * 1000.0); return; }
    long s = (long)sec;
    snprintf(out, n, "%ldd %ldh %ldm %lds",
        s / 86400, (s / 3600) % 24, (s / 60) % 60, s % 60);
}

static void dash_duration_ms(char* out, size_t n, double sec) {
    if (sec < 0) sec = 0;
    long ms = (long)(sec * 1000.0 + 0.5);
    if (ms < 1000) snprintf(out, n, "%ldms", ms);
    else if (ms < 60000) snprintf(out, n, "%lds %03ldms", ms / 1000, ms % 1000);
    else if (ms < 3600000)
        snprintf(out, n, "%ldm %02lds %03ldms", ms / 60000, (ms / 1000) % 60, ms % 1000);
    else dash_duration(out, n, sec);
}

static void dash_perf(char* t, size_t tn, char* p, size_t pn, double part, double total) {
    dash_duration(t, tn, part);
    snprintf(p, pn, "%d%%", total > 0 ? (int)(100.0 * part / total) : 0);
}

static void dash_loss(Dict* log, const char* key, char* out, size_t n) {
    out[0] = 0;
    DictItem* item = puf_log_find(log, key);
    if (item) snprintf(out, n, "%.3f", item->value);
}

void puf_dashboard_print(Ini* ini, PuffeRL* p, Dict* log, int epoch) {
    puf_dashboard_tty = isatty(STDOUT_FILENO);
    int term_rows = 1000, term_cols = PUF_DASH_W;
    if (puf_dashboard_tty) {
        struct winsize ws;
        if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0) {
            if (ws.ws_row > 0) term_rows = ws.ws_row;
            if (ws.ws_col > 0) term_cols = ws.ws_col;
        }
    }

    const char* env_name = puf_ini_get_str(ini, "base", "env_name");
    double steps = puf_log_get_or(log, "agent_steps", (double)p->global_step);
    double sps = puf_log_get_or(log, "SPS", 0);
    long configured = (long)puf_ini_get(ini, "train", "total_timesteps");
    long local_batch = (long)p->hypers.total_agents * p->hypers.horizon;
    long local_steps = configured / p->hypers.world_size;
    double target = (double)((local_steps / local_batch) * local_batch * p->hypers.world_size);
    double remain_sec = sps > 0 && target > steps ? (target - steps) / sps : 0;
    double rollout = puf_log_get_or(log, "perf/rollout", 0);
    double train_time = puf_log_get_or(log, "perf/train", 0);
    double perf_total = rollout + train_time;

    char params[32], steps_s[32], sps_s[32], uptime[64], remaining[64], epoch_s[32];
    dash_abbrev(params, sizeof(params), (double)numel(p->master_weights.shape));
    dash_abbrev(steps_s, sizeof(steps_s), steps);
    dash_abbrev(sps_s, sizeof(sps_s), sps);
    dash_duration_ms(uptime, sizeof(uptime), puf_log_get_or(log, "uptime", 0));
    dash_duration(remaining, sizeof(remaining), remain_sec);
    snprintf(epoch_s, sizeof(epoch_s), "%d", epoch);

    if (puf_dashboard_tty) {
        printf("\033[?2026h");
        if (term_rows != puf_dashboard_last_rows || term_cols != puf_dashboard_last_cols)
            printf("\033[H\033[J");
        else
            printf("\033[H");
        puf_dashboard_last_rows = term_rows;
        puf_dashboard_last_cols = term_cols;
    }
    // Tiny terminal: one-line compact summary.
    if (puf_dashboard_tty && (term_cols < PUF_DASH_W || term_rows <= PUF_DASH_BASE_ROWS)) {
        char compact[512];
        snprintf(compact, sizeof(compact),
            "PufferLib 5.0  env=%s  steps=%s  SPS=%s  score=%.3f  epoch=%s  to_go=%s",
            env_name, steps_s, sps_s, puf_log_get_or(log, "env/score", 0), epoch_s, remaining);
        printf("%.*s", term_cols > 1 ? term_cols - 1 : term_cols, compact);
        if (term_rows > 1) dash_eol();
        else if (puf_dashboard_tty) printf("\033[K");
        dash_end();
        return;
    }

    char gpu[16], vram[32], ram[16];
    snprintf(gpu, sizeof(gpu), "%3.0f%%", puf_log_get_or(log, "util/gpu_percent", 0));
    snprintf(vram, sizeof(vram), "%.1f/%.0fG",
        puf_log_get_or(log, "util/vram_used_gb", 0),
        puf_log_get_or(log, "util/vram_total_gb", 0));
    snprintf(ram, sizeof(ram), "%.1fG", puf_log_get_or(log, "util/cpu_mem_gb", 0));

    int fish_span = 18;
    int fish_pos = (fish_span - 3) - (puf_dashboard_frame++ % (fish_span - 2));

    dash_rule("╭", "╮");
    printf("%s│%s %sPufferLib %s5.0%s%*s%s🐡%s%*s%sGPU%s:%s ",
        PUF_W, PUF_R, PUF_A, PUF_W, PUF_R, fish_pos, "", PUF_A, PUF_R,
        fish_span - 2 - fish_pos, "", PUF_A, PUF_G, PUF_R);
    dash_cell(gpu, 4);
    printf("   %sVRAM%s:%s", PUF_A, PUF_G, PUF_R);
    dash_cell(vram, 10);
    printf("    %sRAM%s:%s", PUF_A, PUF_G, PUF_R);
    dash_cell(ram, 6);
    printf("     %s│%s", PUF_W, PUF_R);
    dash_eol();
    printf("%s│%*s│%s", PUF_W, PUF_DASH_W - 2, "", PUF_R);
    dash_eol();

    char et[64], ep[16], emt[64], emp[16], evt[64], evp[16], ct[64], cp[16];
    char tt[64], tp[16], tmt[64], tmp[16], mt[64], mp[16];
    char lp[32], lv[32], le[32], lt[32], lok[32], lkl[32], lcf[32];
    dash_perf(et, sizeof(et), ep, sizeof(ep), rollout, perf_total);
    dash_perf(emt, sizeof(emt), emp, sizeof(emp), puf_log_get_or(log, "perf/eval_model", 0), perf_total);
    dash_perf(evt, sizeof(evt), evp, sizeof(evp), puf_log_get_or(log, "perf/eval_env", 0), perf_total);
    dash_perf(ct, sizeof(ct), cp, sizeof(cp), puf_log_get_or(log, "perf/eval_copy", 0), perf_total);
    dash_perf(tt, sizeof(tt), tp, sizeof(tp), train_time, perf_total);
    dash_perf(tmt, sizeof(tmt), tmp, sizeof(tmp), puf_log_get_or(log, "perf/train_model", 0), perf_total);
    dash_perf(mt, sizeof(mt), mp, sizeof(mp), puf_log_get_or(log, "perf/train_misc", 0), perf_total);
    dash_loss(log, "loss/policy", lp, sizeof(lp));
    dash_loss(log, "loss/value", lv, sizeof(lv));
    dash_loss(log, "loss/entropy", le, sizeof(le));
    dash_loss(log, "loss/total", lt, sizeof(lt));
    dash_loss(log, "loss/old_kl", lok, sizeof(lok));
    dash_loss(log, "loss/kl", lkl, sizeof(lkl));
    dash_loss(log, "loss/clipfrac", lcf, sizeof(lcf));

    dash_row("Env", env_name, "Evaluate", et, ep, "Losses", lt);
    dash_row("Params", params, "  Model", emt, emp, "policy", lp);
    dash_row("Steps", steps_s, "  Env", evt, evp, "value", lv);
    dash_row("SPS", sps_s, "  Copy", ct, cp, "entropy", le);
    dash_row("Epoch", epoch_s, "Train", tt, tp, "old_kl", lok);
    dash_row("Uptime", uptime, "  Model", tmt, tmp, "kl", lkl);
    dash_row("To go", remaining, "  Misc", mt, mp, "clipfrac", lcf);
    printf("%s│%*s│%s", PUF_W, PUF_DASH_W - 2, "", PUF_R);
    dash_eol();

    int user_rows = PUF_DASH_MAX_USER_ROWS;
    if (puf_dashboard_tty) {
        user_rows = term_rows - PUF_DASH_BASE_ROWS - 1;
        if (user_rows < 0) user_rows = 0;
        if (user_rows > PUF_DASH_MAX_USER_ROWS) user_rows = PUF_DASH_MAX_USER_ROWS;
    }

    char pending_key[128];
    double pending_val = 0;
    int pending = 0, n = 0, max_items = 2 * user_rows;
    for (int i = 0; i < log->size && n < max_items; i++) {
        const char* key = log->items[i].key;
        if (strncmp(key, "env/", 4) != 0 || strcmp(key, "env/n") == 0) continue;
        const char* sk = key + 4;
        if (!pending) {
            snprintf(pending_key, sizeof(pending_key), "%s", sk);
            pending_val = log->items[i].value;
            pending = 1;
        } else {
            char ls[32], rs[32];
            snprintf(ls, sizeof(ls), "%.3f", pending_val);
            snprintf(rs, sizeof(rs), "%.3f", log->items[i].value);
            printf("%s│%s %s%-25.25s%s %s%9.9s%s   %s%-25.25s%s %s%9.9s%s    %s│%s",
                PUF_W, PUF_R, PUF_A, pending_key, PUF_R, PUF_W, ls, PUF_R,
                PUF_A, sk, PUF_R, PUF_W, rs, PUF_R, PUF_W, PUF_R);
            dash_eol();
            pending = 0;
        }
        n++;
    }
    if (pending) {
        char ls[32];
        snprintf(ls, sizeof(ls), "%.3f", pending_val);
        printf("%s│%s %s%-25.25s%s %s%9.9s%s   %-25.25s %9.9s    %s│%s",
            PUF_W, PUF_R, PUF_A, pending_key, PUF_R, PUF_W, ls, PUF_R, "", "", PUF_W, PUF_R);
        dash_eol();
    }
    dash_rule("╰", "╯");
    dash_end();
}

void log_util(PuffeRL* p, Dict* out) {
    nvmlUtilization_t util;
    nvmlDeviceGetUtilizationRates(p->nvml_device, &util);
    dict_set(out, "util/gpu_percent", (double)util.gpu);

    size_t cuda_free;
    size_t cuda_total;
    cudaMemGetInfo(&cuda_free, &cuda_total);
    dict_set(out, "util/vram_used_gb",
        (double)(cuda_total - cuda_free) / (1024.0 * 1024.0 * 1024.0));
    dict_set(out, "util/vram_total_gb",
        (double)cuda_total / (1024.0 * 1024.0 * 1024.0));

    long rss_kb = 0;
    FILE* status = fopen("/proc/self/status", "r");
    if (status) {
        char line[256];
        while (fgets(line, sizeof(line), status)) {
            if (sscanf(line, "VmRSS: %ld", &rss_kb) == 1) {
                break;
            }
        }
        fclose(status);
    }
    dict_set(out, "util/cpu_mem_gb", (double)rss_kb / (1024.0 * 1024.0));
}

void puf_log_env(Dict* out, Dict* env_out) {
    for (int i = 0; i < env_out->size; i++) {
        char key[256];
        snprintf(key, sizeof(key), "env/%s", env_out->items[i].key);
        dict_set(out, key, env_out->items[i].value);
    }
}

void trainer_log(PuffeRL* p, Dict* out) {
    long global_step = p->global_step;
    double now = wall_clock();
    double dt = now - p->last_log_time;
    long sps = dt > 0 ? (long)((global_step - p->last_log_step) / dt) : 0;
    p->last_log_time = now;
    p->last_log_step = global_step;

    dict_set(out, "SPS", (double)sps * p->hypers.world_size);
    dict_set(out, "agent_steps", (double)global_step * p->hypers.world_size);
    dict_set(out, "uptime", now - p->start_time);
    dict_set(out, "epoch", (double)p->epoch);

    Dict env_out = {0};
    vec_log(p->vec, &env_out, 1);
    puf_log_env(out, &env_out);

    float losses_host[NUM_LOSSES];
    cudaMemcpy(losses_host, p->losses_puf.data, sizeof(losses_host), cudaMemcpyDeviceToHost);
    float loss_n = losses_host[LOSS_N];
    if (loss_n > 0) {
        float inv_n = 1.0f / loss_n;
        dict_set(out, "loss/policy", losses_host[LOSS_PG] * inv_n);
        dict_set(out, "loss/value", losses_host[LOSS_VF] * inv_n);
        dict_set(out, "loss/entropy", losses_host[LOSS_ENT] * inv_n);
        dict_set(out, "loss/total", losses_host[LOSS_TOTAL] * inv_n);
        dict_set(out, "loss/old_kl", losses_host[LOSS_OLD_APPROX_KL] * inv_n);
        dict_set(out, "loss/kl", losses_host[LOSS_APPROX_KL] * inv_n);
        dict_set(out, "loss/clipfrac", losses_host[LOSS_CLIPFRAC] * inv_n);
    }
    cudaMemset(p->losses_puf.data, 0, numel(p->losses_puf.shape) * sizeof(float));

    log_util(p, out);

    float train_total = 0;
    for (int i = 0; i < (int)(sizeof(PROF_NAMES) / sizeof(PROF_NAMES[0])); i++) {
        float sec = p->profile.accum[i] / 1000.0f;
        char key[256];
        snprintf(key, sizeof(key), "perf/%s", PROF_NAMES[i]);
        dict_set(out, key, sec);
        if (i >= PROF_TRAIN_MISC) {
            train_total += sec;
        }
    }
    dict_set(out, "perf/train", train_total);
    memset(p->profile.accum, 0, sizeof(p->profile.accum));
}

void trainer_eval_log(PuffeRL* p, Dict* out) {
    double now = wall_clock();
    p->last_log_time = now;
    p->last_log_step = p->global_step;
    log_util(p, out);

    Dict env_out = {0};
    vec_log(p->vec, &env_out, 0);
    puf_log_env(out, &env_out);
}

typedef struct {
    Dict* items;
    int size;
    int capacity;
} PufLogHistory;

void puf_log_update(Dict* dst, Dict* src) {
    for (int i = 0; i < src->size; i++) {
        DictItem* item = &src->items[i];
        if (item->str) {
            dict_set_str(dst, item->key, item->str);
            puf_log_find(dst, item->key)->value = item->value;
        } else {
            dict_set(dst, item->key, item->value);
        }
    }
}

void puf_log_history_add(PufLogHistory* history, Dict* log) {
    if (history->size == history->capacity) {
        history->capacity = history->capacity ? 2 * history->capacity : 64;
        history->items = (Dict*)realloc(history->items, (size_t)history->capacity * sizeof(Dict));
        if (!history->items) {
            perror("realloc");
            exit(1);
        }
    }

    dict_copy(&history->items[history->size], log);
    history->size++;
}

void puf_log_history_free(PufLogHistory* history) {
    for (int i = 0; i < history->size; i++) {
        dict_clear(&history->items[i]);
    }
    free(history->items);
    memset(history, 0, sizeof(*history));
}

void puf_log_collect_keys(PufLogHistory* history, Dict* keys) {
    for (int i = 0; i < history->size; i++) {
        Dict* log = &history->items[i];
        for (int j = 0; j < log->size; j++) {
            if (!puf_log_find(keys, log->items[j].key)) {
                dict_set(keys, log->items[j].key, 0);
            }
        }
    }
}

double puf_log_reduce(double* vals, int n, double fallback) {
    if (n == 0) {
        return fallback;
    }

    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += vals[i];
    }
    return sum / n;
}

void puf_log_write_metric(FILE* fp, const char* key, double* values, int n) {
    fprintf(fp, "%s = ", key);
    for (int i = 0; i < n; i++) {
        if (i > 0) {
            fputc(',', fp);
        }
        fprintf(fp, "%.17g", values[i]);
    }
    fputc('\n', fp);
}

void puf_log_write(const char* path, Ini* ini, PufLogHistory* history) {
    if (history->size == 0) {
        fprintf(stderr, "cannot write empty log history\n");
        exit(1);
    }

    FILE* fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "failed to write log %s\n", path);
        exit(1);
    }

    fprintf(fp, "# PufferLib log v1\n");
    puf_ini_write(fp, ini);
    fprintf(fp, "\n[metrics]\n");

    int downsample = (int)puf_ini_get(ini, "sweep", "downsample");
    Dict keys = {0};
    puf_log_collect_keys(history, &keys);
    int points = downsample <= 1 ? 1 : downsample;
    double* out = (double*)calloc((size_t)points, sizeof(double));
    double* bin = (double*)calloc((size_t)history->size, sizeof(double));
    double final_steps = dict_get(&history->items[history->size - 1], "agent_steps");

    for (int k = 0; k < keys.size; k++) {
        const char* key = keys.items[k].key;
        if (strncmp(key, "loss/", 5) == 0) {
            continue;
        }

        double first_value = 0;
        for (int i = 0; i < history->size; i++) {
            DictItem* item = puf_log_find(&history->items[i], key);
            if (item) {
                first_value = item->value;
                break;
            }
        }

        if (points == 1) {
            DictItem* item = puf_log_find(&history->items[history->size - 1], key);
            out[0] = item ? item->value : first_value;
            puf_log_write_metric(fp, key, out, points);
            continue;
        }

        int out_idx = 0;
        int bin_n = 0;
        double fallback = first_value;
        double next_bin = final_steps / (points - 1);
        for (int i = 0; i < history->size; i++) {
            Dict* log = &history->items[i];
            DictItem* item = puf_log_find(log, key);
            if (item) {
                bin[bin_n++] = item->value;
            }

            double steps = dict_get(log, "agent_steps");
            if (steps < next_bin || out_idx >= points - 1) {
                continue;
            }

            double reduced = puf_log_reduce(bin, bin_n, fallback);
            out[out_idx++] = reduced;
            fallback = reduced;
            bin_n = 0;
            next_bin += final_steps / (points - 1);
        }

        DictItem* final_item = puf_log_find(&history->items[history->size - 1], key);
        out[points - 1] = final_item ? final_item->value : puf_log_reduce(bin, bin_n, fallback);
        while (out_idx < points - 1) {
            out[out_idx++] = fallback;
        }
        puf_log_write_metric(fp, key, out, points);
    }

    free(bin);
    free(out);
    fclose(fp);
}

double rollout_start(PuffeRL* p, int slot) {
    p->rollout_write_slot = slot;
    if (p->hypers.async) {
        int64_t n = numel(p->param_puf.shape);
        cudaMemcpyAsync(p->actor_param_puf.data, p->param_puf.data,
            n * sizeof(precision_t), cudaMemcpyDeviceToDevice, p->default_stream);
        cudaStreamSynchronize(p->default_stream);
    }
    if (p->hypers.reset_state) {
        zero_all_buffer_states(p, p->default_stream);
        cudaStreamSynchronize(p->default_stream);
    }

    double t0 = wall_clock();
    if (p->vec->gpu_env) {
#ifdef PUFFER_GPU_ENV_AVAILABLE
        VecEnv* vec = p->vec;
        int count = p->hypers.total_agents;
        for (int t = 0; t < p->hypers.horizon; t++) {
            cudaEventRecord(p->profile.rollout_gpu_start[t], p->streams[0]);
            pufferl_forward(p, 0, t, p->streams[0]);
            cudaEventRecord(p->profile.rollout_gpu_end[t], p->streams[0]);
            puf_gpu_env_step(vec->gpu_envs, vec->gpu_actions, vec->gpu_observations,
                vec->gpu_rewards, vec->gpu_terminals, 0, count, p->streams[0]);
            cudaEventRecord(p->profile.rollout_env_end[t], p->streams[0]);
        }
#else
        assert(0);
#endif
    } else {
        for (int buf = 0; buf < p->vec->buffers; buf++) {
            __atomic_store_n(&p->vec->buffer_states[buf], OMP_RUNNING, __ATOMIC_SEQ_CST);
        }
    }
    return t0;
}

void rollout_finish(PuffeRL* p, double t0) {
    if (p->vec->gpu_env) {
        cudaStreamSynchronize(p->streams[0]);
        float gpu_ms = 0.0f;
        float env_ms = 0.0f;
        for (int t = 0; t < p->hypers.horizon; t++) {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms,
                p->profile.rollout_gpu_start[t], p->profile.rollout_gpu_end[t]);
            gpu_ms += ms;
            cudaEventElapsedTime(&ms,
                p->profile.rollout_gpu_end[t], p->profile.rollout_env_end[t]);
            env_ms += ms;
        }
        p->profile.accum[PROF_EVAL_MODEL] += gpu_ms;
        p->profile.accum[PROF_EVAL_ENV] += env_ms;
        p->profile.accum[PROF_ROLLOUT] += gpu_ms + env_ms;
    } else {
        for (int buf = 0; buf < p->vec->buffers; buf++) {
            while (__atomic_load_n(&p->vec->buffer_states[buf], __ATOMIC_SEQ_CST) != OMP_WAITING) {
            }
        }
        float sec = (float)(wall_clock() - t0);
        p->profile.accum[PROF_ROLLOUT] += sec * 1000.0f;

        float eval_prof[NUM_VEC_PROF] = {0};
        for (int buf = 0; buf < p->vec->buffers; buf++) {
            float* src = &p->vec->accum[buf * NUM_VEC_PROF];
            for (int i = 0; i < NUM_VEC_PROF; i++) {
                eval_prof[i] += src[i];
            }
            memset(src, 0, NUM_VEC_PROF * sizeof(float));
        }
        p->profile.accum[PROF_EVAL_MODEL] += eval_prof[VEC_MODEL] / p->vec->buffers;
        p->profile.accum[PROF_EVAL_ENV] += eval_prof[VEC_ENV_STEP] / p->vec->buffers;
        p->profile.accum[PROF_EVAL_COPY] += eval_prof[VEC_COPY] / p->vec->buffers;
    }
}

void rollouts(PuffeRL* p) {
    double t0 = rollout_start(p, 0);
    rollout_finish(p, t0);
    p->global_step += p->hypers.horizon * p->hypers.total_agents;
}

void async_train_step(PuffeRL* p, int prefetch_next) {
    if (!p->async_bootstrapped) {
        double t0 = rollout_start(p, 0);
        rollout_finish(p, t0);
        p->async_ready_slot = 0;
        p->async_next_slot = 1;
        p->async_bootstrapped = true;
    }

    int ready_slot = p->async_ready_slot;
    int next_slot = p->async_next_slot;
    double t0 = 0.0;
    if (prefetch_next) {
        t0 = rollout_start(p, next_slot);
    }

    p->global_step += p->hypers.horizon * p->hypers.total_agents;
    RolloutBuf train_src = rollout_time_view(&p->rollouts,
        ready_slot * p->hypers.horizon, p->hypers.horizon);
    train_impl(*p, &train_src);

    if (prefetch_next) {
        rollout_finish(p, t0);
        p->async_ready_slot = next_slot;
        p->async_next_slot = ready_slot;
    }
}

typedef struct {
    float score;
    float draw;
    int games;
} EvalResult;

#define TRAIN_RESULT_MAX_POINTS 64
typedef struct {
    float score;
    float cost;
    float steps;
    int points;
    char checkpoint_path[4096];
    float scores[TRAIN_RESULT_MAX_POINTS];
    float costs[TRAIN_RESULT_MAX_POINTS];
    float step_points[TRAIN_RESULT_MAX_POINTS];
} TrainResult;

#define EVAL_RENDER 0
#define EVAL_SCORE 1
#define EVAL_MATCH 2

EvalResult run_eval(Ini* ini, TrainContext* ctx, int mode, int verbose);

#define SELFPLAY_MAX_BANKS 8
#define SELFPLAY_MAX_POOL 1024
#define SELFPLAY_PATH_MAX 4096

typedef struct {
    char pending_path[SELFPLAY_PATH_MAX];
    long opp_started_step;
    int num_envs;
} SelfplayBank;

typedef struct {
    int num_banks;
    int max_size;
    long opp_timeout_steps;
    unsigned int rng;
    char pool[SELFPLAY_MAX_POOL][SELFPLAY_PATH_MAX];
    int pool_size;
    SelfplayBank banks[SELFPLAY_MAX_BANKS];
} Selfplay;

void selfplay_add_checkpoint(Selfplay* sp, const char* path) {
    while (access(path, R_OK) != 0) usleep(50000);
    for (int i = 0; i < sp->pool_size; i++) {
        if (strcmp(sp->pool[i], path) == 0) return;
    }
    if (sp->pool_size >= SELFPLAY_MAX_POOL) {
        fprintf(stderr, "selfplay pool exceeds SELFPLAY_MAX_POOL\n");
        exit(1);
    }
    snprintf(sp->pool[sp->pool_size++], sizeof(sp->pool[0]), "%s", path);
    if (sp->pool_size > sp->max_size) {
        int start = sp->pool_size - sp->max_size;
        memmove(sp->pool, sp->pool + start, (size_t)sp->max_size * sizeof(sp->pool[0]));
        sp->pool_size = sp->max_size;
    }
}

const char* selfplay_sample(Selfplay* sp) {
    if (sp->pool_size == 0) {
        fprintf(stderr, "selfplay opponent pool is empty\n");
        exit(1);
    }
    int idx = (int)(rand_r(&sp->rng) % (unsigned int)sp->pool_size);
    return sp->pool[idx];
}

void selfplay_init(Selfplay* sp, Ini* ini, PuffeRL* p,
        const char* initial_checkpoint) {
    memset(sp, 0, sizeof(*sp));
    sp->num_banks = p->num_frozen_banks;
    if (sp->num_banks <= 0 || sp->num_banks > SELFPLAY_MAX_BANKS) {
        fprintf(stderr, "selfplay requires 1..%d frozen banks\n", SELFPLAY_MAX_BANKS);
        exit(1);
    }
    sp->max_size = (int)puf_ini_get(ini, "selfplay", "max_size");
    sp->opp_timeout_steps = (long)puf_ini_get(ini, "selfplay", "opp_timeout_steps");
    sp->rng = (unsigned int)puf_ini_get(ini, "selfplay", "seed") + (unsigned int)p->hypers.rank;
    long current_step = p->global_step * p->hypers.world_size;

    Env* envs = (Env*)p->vec->envs;
    for (int i = 0; i < p->vec->size; i++) {
        int tag = envs[i].tag;
        if (tag > 0 && tag <= sp->num_banks) sp->banks[tag - 1].num_envs++;
    }

    selfplay_add_checkpoint(sp, initial_checkpoint);
    for (int b = 0; b < sp->num_banks; b++) {
        pufferl_load_frozen_bank(p, b, selfplay_sample(sp));
        sp->banks[b].opp_started_step = current_step;
    }
}

void selfplay_step(Selfplay* sp, PuffeRL* p, Dict* log) {
    long current_step = p->global_step * p->hypers.world_size;
    Env* envs = (Env*)p->vec->envs;
    for (int b = 0; b < sp->num_banks; b++) {
        SelfplayBank* bank = &sp->banks[b];
        int timed_out = sp->opp_timeout_steps > 0 &&
            current_step - bank->opp_started_step >= sp->opp_timeout_steps;
        int tag = b + 1;
        if (bank->pending_path[0]) {
            int aligned = 0;
            for (int i = 0; i < p->vec->size; i++) {
                if (envs[i].tag == tag && envs[i].boundary_reached) aligned++;
            }
            if (aligned >= bank->num_envs) {
                pufferl_load_frozen_bank(p, b, bank->pending_path);
                for (int i = 0; i < p->vec->size; i++) {
                    if (envs[i].tag == tag) envs[i].boundary_reached = 0;
                }
                bank->pending_path[0] = 0;
                bank->opp_started_step = current_step;
            }
        } else if (timed_out) {
            const char* path = selfplay_sample(sp);
            snprintf(bank->pending_path, sizeof(bank->pending_path), "%s", path);
            for (int i = 0; i < p->vec->size; i++) {
                if (envs[i].tag == tag) envs[i].boundary_reached = 0;
            }
        }
    }
    dict_set(log, "pool/size", sp->pool_size);
    dict_set(log, "pool/num_banks", sp->num_banks);
}

// End-of-train selfplay rating: sequential matches vs a small pool of saved
// checkpoints. Metric is mean A winrate (not Elo) — cheap and good enough for
// sweeps. Returns NAN if skipped so the caller can keep the train metric.
float selfplay_pool_eval(Ini* ini, TrainContext* ctx, Selfplay* sp,
        const char* our_ckpt) {
    if (!sp || !our_ckpt || !our_ckpt[0]) return NAN;
    int max_opp = (int)puf_ini_get(ini, "selfplay", "eval_pool_size");
    long games = (long)puf_ini_get(ini, "selfplay", "eval_games");
    if (max_opp <= 0 || games <= 0 || sp->pool_size <= 0) return NAN;

    // Unique opponents from the pool, excluding our own checkpoint path.
    char opp[SELFPLAY_MAX_POOL][SELFPLAY_PATH_MAX];
    int n_opp = 0;
    for (int i = 0; i < sp->pool_size && n_opp < max_opp; i++) {
        if (strcmp(sp->pool[i], our_ckpt) == 0) continue;
        int seen = 0;
        for (int j = 0; j < n_opp; j++) {
            if (strcmp(opp[j], sp->pool[i]) == 0) { seen = 1; break; }
        }
        if (seen) continue;
        snprintf(opp[n_opp++], SELFPLAY_PATH_MAX, "%s", sp->pool[i]);
    }
    if (n_opp == 0) return NAN;

    char games_buf[32];
    snprintf(games_buf, sizeof(games_buf), "%ld", games);
    puf_ini_put(ini, "base.num_games", games_buf);
    puf_ini_put(ini, "base.load_model_path", our_ckpt);

    float sum = 0;
    for (int i = 0; i < n_opp; i++) {
        puf_ini_put(ini, "base.load_enemy_model_path", opp[i]);
        EvalResult r = run_eval(ini, ctx, EVAL_MATCH, 0);
        sum += r.score;
        printf("selfplay_eval vs %s games=%d score=%.4f draw=%.4f\n",
            opp[i], r.games, r.score, r.draw);
    }
    float mean = sum / (float)n_opp;
    printf("selfplay_eval mean_score=%.4f n=%d\n", mean, n_opp);
    return mean;
}

typedef struct {
    char section[64];
    char key[64];
} SweepParam;

// Build SweepSpace + param map from [sweep.<section>.<key>] sections.
// Dies on bad names / dist / min-max-scale. cost_idx set if timesteps is swept.
SweepSpace* puf_config_sweep_space(Ini* ini, SweepParam** params_out) {
    const char* goal = puf_ini_get_str(ini, "sweep", "goal");
    int direction = 1;
    if (strcmp(goal, "minimize") == 0) {
        direction = -1;
    } else if (strcmp(goal, "maximize") != 0) {
        puf_die("sweep.goal must be maximize or minimize");
    }

    SweepSpace* space = sweep_space_create(ini->num_sections, -1, direction);
    SweepParam* params = (SweepParam*)calloc((size_t)ini->num_sections,
        sizeof(SweepParam));
    int n = 0;

    for (int i = 0; i < ini->num_sections; i++) {
        Dict* dict = &ini->sections[i];
        if (strncmp(dict->name, "sweep.", 6) != 0) {
            continue;
        }

        // [sweep.train.lr] -> section=train, key=lr (last dot).
        const char* path = dict->name + 6;
        const char* dot = strrchr(path, '.');
        if (!dot || dot == path || !dot[1]) {
            puf_die("expected section [sweep.<section>.<key>], got [%s]",
                dict->name);
        }
        snprintf(params[n].section, sizeof(params[n].section), "%.*s",
            (int)(dot - path), path);
        snprintf(params[n].key, sizeof(params[n].key), "%s", dot + 1);

        const char* dist = dict_get_str(dict, "distribution");
        SpaceType type;
        int is_integer = 0;
        if (strcmp(dist, "uniform") == 0) {
            type = SPACE_LINEAR;
        } else if (strcmp(dist, "int_uniform") == 0) {
            type = SPACE_LINEAR;
            is_integer = 1;
        } else if (strcmp(dist, "uniform_pow2") == 0) {
            type = SPACE_POW2;
            is_integer = 1;
        } else if (strcmp(dist, "log_normal") == 0) {
            type = SPACE_LOG;
        } else if (strcmp(dist, "logit_normal") == 0) {
            type = SPACE_LOGIT;
        } else {
            puf_die("invalid sweep distribution [%s] %s", dict->name, dist);
        }

        float min_v = (float)dict_get(dict, "min");
        float max_v = (float)dict_get(dict, "max");
        const char* scale_s = dict_get_str(dict, "scale");
        float scale;
        if (strcmp(scale_s, "auto") == 0) {
            scale = 0.5f;
        } else if (strcmp(scale_s, "time") == 0) {
            if (min_v <= 0 || max_v <= 0) {
                puf_die("[%s] scale=time requires positive min/max", dict->name);
            }
            scale = 1.0f / (log2f(max_v) - log2f(min_v));
        } else {
            scale = (float)dict_get(dict, "scale");
        }

        space_init(&space->spaces[n], type, min_v, max_v, scale, is_integer);
        // Budget lever for protein (observed cost is still wallclock).
        if (strcmp(dict->name, "sweep.train.total_timesteps") == 0) {
            space->cost_idx = n;
        }
        n++;
    }

    space->num = n;
    *params_out = params;
    return space;
}

void puf_config_sweep_apply(Ini* ini, SweepParam* params,
        SweepSpace* space, float* sample) {
    for (int i = 0; i < space->num; i++) {
        float val = space_unnormalize(&space->spaces[i], sample[i]);
        char buf[64];
        snprintf(buf, sizeof(buf), "%.9g", val);
        char key[256];
        snprintf(key, sizeof(key), "%s.%s", params[i].section, params[i].key);
        puf_ini_put(ini, key, buf);
    }
}

extern char** environ;

typedef struct {
    int run;
    int random;
    int gp_obs;
    int pareto;
    int fd;
    pid_t pid;
    char run_id[128];
    float* sample;
    TrainResult result;
} SweepJob;

int native_num_gpus(void) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count < 1) {
        fprintf(stderr, "sweep error: no CUDA devices available\n");
        exit(1);
    }
    return count;
}

int sweep_read_result(int fd, TrainResult* out) {
    char* dst = (char*)out;
    size_t need = sizeof(*out);
    while (need > 0) {
        ssize_t n = read(fd, dst, need);
        if (n <= 0) {
            return 0;
        }
        dst += n;
        need -= (size_t)n;
    }
    return 1;
}

char* sweep_arg_kv(const char* full_key, DictItem* item) {
    char val[128];
    const char* src = item->str;
    if (!src) {
        snprintf(val, sizeof(val), "%.17g", item->value);
        src = val;
    }

    size_t n = strlen(full_key) + strlen(src) + 2;
    char* out = (char*)malloc(n);
    if (!out) {
        perror("malloc");
        exit(1);
    }
    snprintf(out, n, "%s=%s", full_key, src);
    return out;
}

void sweep_free_argv(char** argv, int argc) {
    for (int i = 3; i < argc; i++) {
        free(argv[i]);
    }
    free(argv);
}

// Soft-check train combos after overlaying sample onto base keys (no full Ini copy).
int sweep_sample_valid(Ini* ini, SweepParam* params,
        SweepSpace* space, float* sample, char* err, size_t err_size) {
    int mb = (int)puf_ini_get(ini, "train", "minibatch_size");
    int horizon = (int)puf_ini_get(ini, "train", "horizon");
    int agents = (int)puf_ini_get(ini, "vec", "total_agents");
    int gpus = (int)puf_ini_get(ini, "train", "gpus");
    for (int i = 0; i < space->num; i++) {
        float val = space_unnormalize(&space->spaces[i], sample[i]);
        if (strcmp(params[i].section, "train") == 0) {
            if (strcmp(params[i].key, "minibatch_size") == 0) mb = (int)val;
            else if (strcmp(params[i].key, "horizon") == 0) horizon = (int)val;
            else if (strcmp(params[i].key, "gpus") == 0) gpus = (int)val;
        } else if (strcmp(params[i].section, "vec") == 0 &&
                strcmp(params[i].key, "total_agents") == 0) {
            agents = (int)val;
        }
    }
    return train_shape_valid(mb, horizon, agents, gpus, err, err_size);
}

// Flatten ini to argv: exe mode env section.key=value ...
char** sweep_make_argv(Ini* ini, const char* exe_path, const char* mode,
        int* argc_out) {
    int nkeys = 0;
    for (int s = 0; s < ini->num_sections; s++) {
        nkeys += ini->sections[s].size;
    }
    int argc = nkeys + 4;
    char** argv = (char**)calloc((size_t)argc, sizeof(char*));
    if (!argv) {
        perror("calloc");
        exit(1);
    }
    argv[0] = (char*)exe_path;
    argv[1] = (char*)mode;
    argv[2] = (char*)puf_ini_get_str(ini, "base", "env_name");
    int idx = 3;
    char full_key[PUF_DICT_MAX_KEY * 2];
    for (int s = 0; s < ini->num_sections; s++) {
        Dict* dict = &ini->sections[s];
        for (int i = 0; i < dict->size; i++) {
            snprintf(full_key, sizeof(full_key), "%s.%s",
                dict->name, dict->items[i].key);
            argv[idx++] = sweep_arg_kv(full_key, &dict->items[i]);
        }
    }
    argv[argc - 1] = NULL;
    *argc_out = argc;
    return argv;
}

// Spawn a clean `train` process (needed after parent CUDA init for protein).
// Child runs main→launch_train; train.gpus>1 DP-forks inside that process.
// gpu_block_offset: first device of this trial's block (size train.gpus).
// *read_fd receives the TrainResult pipe (caller closes after read).
static pid_t spawn_train_process(Ini* trial, const char* exe_path,
        int gpu_block_offset, int* read_fd) {
    int pipefd[2];
    if (pipe(pipefd) != 0) { perror("pipe"); exit(1); }

    char buf[64];
    snprintf(buf, sizeof(buf), "%d", gpu_block_offset);
    puf_ini_put(trial, "base.gpu_offset", buf);
    snprintf(buf, sizeof(buf), "%d", pipefd[1]);
    puf_ini_put(trial, "base.result_fd", buf);

    int argc = 0;
    char** argv = sweep_make_argv(trial, exe_path, "train", &argc);
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addclose(&actions, pipefd[0]);
    posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
    pid_t pid = 0;
    int err = posix_spawnp(&pid, exe_path, &actions, NULL, argv, environ);
    posix_spawn_file_actions_destroy(&actions);
    sweep_free_argv(argv, argc - 1);
    if (err != 0) {
        fprintf(stderr, "posix_spawn train failed: %s\n", strerror(err));
        exit(1);
    }
    close(pipefd[1]);
    *read_fd = pipefd[0];
    return pid;
}

// One sweep trial: optional hyper overlay + spawn_train_process on a GPU block.
SweepJob sweep_start_job(Ini* ini, const char* exe_path,
        SweepParam* params, SweepSpace* space, float* sample,
        ProteinSweepInfo info, int run, int gpu_block_offset, int apply_sample) {
    SweepJob job = {0};
    job.run = run;
    job.random = info.is_random;
    job.gp_obs = info.n_gp_obs;
    job.pareto = info.n_pareto;
    job.sample = (float*)calloc((size_t)space->num, sizeof(float));
    memcpy(job.sample, sample, (size_t)space->num * sizeof(float));

    Ini trial = {0};
    puf_ini_copy(&trial, ini);
    if (apply_sample) puf_config_sweep_apply(&trial, params, space, sample);
    snprintf(job.run_id, sizeof(job.run_id), "sweep_%ld_%04d",
        (long)(1000.0 * wall_clock()), run);
    puf_ini_put(&trial, "base.run_id", job.run_id);

    job.pid = spawn_train_process(&trial, exe_path, gpu_block_offset, &job.fd);
    puf_ini_free(&trial);
    return job;
}

// Returns 1 on success, 0 if the worker failed. Failures are observed as bad
// samples so the sweep can keep going and fill the slot later.
int sweep_wait_job(ProteinSweep* protein, SweepJob* job, float max_cost) {
    int ok = sweep_read_result(job->fd, &job->result);
    close(job->fd);

    int status = 0;
    int wait_ok = waitpid(job->pid, &status, 0) >= 0;
    if (!wait_ok) perror("waitpid");
    int good = wait_ok && ok && WIFEXITED(status) && WEXITSTATUS(status) == 0;

    if (!good) {
        fprintf(stderr, "sweep worker run=%d failed; marking sample bad\n", job->run);
        protein_sweep_observe(protein, job->sample, NAN, max_cost, 1);
        free(job->sample);
        job->sample = NULL;
        return 0;
    }

    // Non-selfplay: points[] is a downsampled learning curve.
    // Selfplay: points==1 — final pool-eval winrate (or last train metric).
    int points = job->result.points > 0 ? job->result.points : 1;
    for (int i = 0; i < points; i++) {
        protein_sweep_observe(protein, job->sample,
            job->result.scores[i], job->result.costs[i], 0);
    }
    printf("sweep run=%d score=%.4f cost=%.2f steps=%.0f random=%d gp_obs=%d pareto=%d\n",
        job->run, job->result.score, job->result.cost, job->result.steps,
        job->random, job->gp_obs, job->pareto);
    free(job->sample);
    job->sample = NULL;
    return 1;
}

void run_sweep(Ini* ini, const char* exe_path) {
    SweepParam* params = NULL;
    SweepSpace* space = puf_config_sweep_space(ini, &params);

    int max_runs = (int)puf_ini_get(ini, "sweep", "max_runs");
    int downsample = (int)puf_ini_get(ini, "sweep", "downsample");
    int prune_pareto = (int)puf_ini_get(ini, "sweep", "prune_pareto");
    const char* metric_dist = puf_ini_get_str(ini, "sweep", "metric_distribution");
    if (strcmp(metric_dist, "linear") != 0 && strcmp(metric_dist, "logit") != 0) {
        puf_die("sweep.metric_distribution must be linear or logit");
    }
    int use_logit = strcmp(metric_dist, "logit") == 0;
    float max_cost = (float)puf_ini_get(ini, "sweep", "max_suggestion_cost");
    float early_stop_quantile = (float)puf_ini_get(ini, "sweep", "early_stop_quantile");
    if (max_runs < 1) puf_die("sweep.max_runs must be >= 1");
    if (downsample < 1 || downsample > TRAIN_RESULT_MAX_POINTS) {
        puf_die("sweep.downsample must be in [1, %d]", TRAIN_RESULT_MAX_POINTS);
    }
    int success_cap = max_runs * downsample * 2;
    if (success_cap < 8192) success_cap = 8192;

    // GPU packing: each trial is a full train (launch_train) that may itself
    // use train.gpus for NCCL DP. Concurrent trials = sweep_gpus / train_gpus
    // on disjoint blocks [0,W), [W,2W), ...  e.g. 8 GPUs, train.gpus=2 → 4 trials.
    int total_gpus = native_num_gpus();
    int sweep_gpus = (int)puf_ini_get(ini, "sweep", "gpus");
    int train_gpus = (int)puf_ini_get(ini, "train", "gpus");
    if (sweep_gpus < 0) puf_die("sweep.gpus must be >= 0");
    if (sweep_gpus == 0) sweep_gpus = total_gpus;
    if (sweep_gpus > total_gpus) {
        puf_die("sweep.gpus=%d but only %d CUDA devices are visible",
            sweep_gpus, total_gpus);
    }
    if (train_gpus < 1) puf_die("train.gpus must be >= 1");
    if (sweep_gpus < train_gpus) {
        puf_die("sweep.gpus (%d) must be >= train.gpus (%d)", sweep_gpus, train_gpus);
    }
    if ((int)puf_ini_get(ini, "sweep", "use_gpu")) {
        cudaSetDevice(sweep_gpus - 1);
    }

    int parallel = sweep_gpus / train_gpus;
    if (parallel < 1) parallel = 1;
    if (sweep_gpus % train_gpus != 0) {
        fprintf(stderr, "sweep note: %d GPUs not divisible by train.gpus=%d; "
            "using %d concurrent trials (%d GPUs idle)\n",
            sweep_gpus, train_gpus, parallel, sweep_gpus - parallel * train_gpus);
    }

    ProteinSweep* protein = protein_sweep_create(space,
        10, 256, 50, 0.001f, 50, 750, 4096,
        downsample == 1, prune_pareto, use_logit,
        1.0f, max_cost, 0.1f, -0.8f, early_stop_quantile,
        success_cap, 1024, 5, 73ULL);

    float* sample = (float*)calloc((size_t)space->num, sizeof(float));
    SweepJob* jobs = (SweepJob*)calloc((size_t)parallel, sizeof(SweepJob));
    int invalid_samples = 0;
    int failed_workers = 0;
    int next_run_id = 0;
    int completed = 0;
    int used_default = 0;
    while (completed < max_runs) {
        int batch = max_runs - completed;
        if (batch > parallel) batch = parallel;

        for (int i = 0; i < batch; i++) {
            ProteinSweepInfo info = {0};
            char err[256];
            int job_run = next_run_id++;
            int apply_sample = 1;
            if (!used_default) {
                // First launch uses config defaults (not a suggested sample).
                for (int j = 0; j < space->num; j++) {
                    float val = (float)puf_ini_get(ini, params[j].section, params[j].key);
                    float norm = space_normalize(&space->spaces[j], val);
                    if (!isfinite(norm) || norm < -1.0f || norm > 1.0f) {
                        puf_die("default sweep value [%s] %s=%.9g is outside its sweep range",
                            params[j].section, params[j].key, val);
                    }
                    sample[j] = norm;
                }
                if (!sweep_sample_valid(ini, params, space, sample, err, sizeof(err))) {
                    puf_die("default sweep sample is invalid: %s", err);
                }
                used_default = 1;
                apply_sample = 0;
            } else {
                for (;;) {
                    info = protein_sweep_suggest(protein, sample, NAN);
                    if (sweep_sample_valid(ini, params, space, sample, err, sizeof(err))) {
                        break;
                    }
                    protein_sweep_observe(protein, sample, NAN, max_cost, 1);
                    invalid_samples++;
                    if (invalid_samples <= 5 || invalid_samples % 100 == 0) {
                        fprintf(stderr, "sweep skipped invalid sample: %s\n", err);
                    }
                    if (invalid_samples > 10000) {
                        fprintf(stderr, "sweep error: too many invalid samples\n");
                        exit(1);
                    }
                }
            }
            // Trial i owns GPU block [i*train_gpus, (i+1)*train_gpus).
            jobs[i] = sweep_start_job(ini, exe_path, params, space, sample,
                info, job_run, /*gpu_block_offset=*/i * train_gpus, apply_sample);
        }
        for (int i = 0; i < batch; i++) {
            if (sweep_wait_job(protein, &jobs[i], max_cost)) {
                completed++;
            } else {
                failed_workers++;
                if (failed_workers > 10000) {
                    fprintf(stderr, "sweep error: too many failed workers\n");
                    exit(1);
                }
            }
        }
    }

    free(jobs);
    free(sample);
    free(params);
    protein_sweep_destroy(protein);
    sweep_space_destroy(space);
}

// Fill TrainResult for protein. single_point=1: only the final score (selfplay —
// expensive pool eval is the only signal). single_point=0: downsample the
// learning curve so protein can multi-fidelity observe intermediate points.
void train_result_fill(TrainResult* result, PufLogHistory* history,
        Dict* last_log, Ini* ini, const char* target_key, int single_point) {
    result->score = (float)puf_log_get_or(last_log, target_key, 0);
    result->cost = (float)puf_log_get_or(last_log, "uptime", 0);
    result->steps = (float)puf_log_get_or(last_log, "agent_steps", 0);

    int points = 1;
    if (!single_point) {
        points = (int)puf_ini_get(ini, "sweep", "downsample");
        if (points < 1) points = 1;
        if (points > TRAIN_RESULT_MAX_POINTS) points = TRAIN_RESULT_MAX_POINTS;
    }
    result->points = points;

    if (history->size == 0 || points == 1) {
        result->scores[0] = result->score;
        result->costs[0] = result->cost;
        result->step_points[0] = result->steps;
        return;
    }

    float final_steps = (float)puf_log_get_or(&history->items[history->size - 1],
        "agent_steps", result->steps);
    int cursor = 0;
    for (int p = 0; p < points; p++) {
        float target = final_steps * (float)p / (float)(points - 1);
        while (cursor + 1 < history->size &&
                (float)puf_log_get_or(&history->items[cursor], "agent_steps", 0) < target) {
            cursor++;
        }
        Dict* log = &history->items[cursor];
        result->scores[p] = (float)puf_log_get_or(log, target_key, result->score);
        result->costs[p] = (float)puf_log_get_or(log, "uptime", result->cost);
        result->step_points[p] = (float)puf_log_get_or(log, "agent_steps", target);
    }
    result->scores[points - 1] = result->score;
    result->costs[points - 1] = result->cost;
    result->step_points[points - 1] = result->steps;
}

EvalResult run_eval(Ini* ini, TrainContext* ctx, int mode, int verbose) {
    int render = mode == EVAL_RENDER;
    int match = mode == EVAL_MATCH;
    EvalResult result = {0};
    long num_games = puf_ini_get(ini, "base", "num_games");
    if (!num_games) {
        num_games = puf_ini_get(ini, "base", "eval_episodes");
    }
    long burnin_games = puf_ini_get(ini, "base", "burnin_games");
    if (!render && (num_games <= 0 || burnin_games < 0)) {
        fprintf(stderr, "eval requires positive num_games and nonnegative burnin_games\n");
        exit(1);
    }
    if (!render) {
        long eval_agents = puf_ini_get(ini, "base", "eval_agents");
        if (!eval_agents && match) {
            eval_agents = puf_ini_get(ini, "selfplay", "eval_agents");
        }
        if (eval_agents <= 0 && match) {
            eval_agents = 8192;
        } else if (eval_agents <= 0) {
            eval_agents = num_games / 8;
            if (eval_agents < 1024) {
                eval_agents = 1024;
            }
            if (eval_agents > 4096) {
                eval_agents = 4096;
            }
            if (eval_agents > num_games && num_games >= 1024) {
                eval_agents = num_games;
            }
        } else if (eval_agents > num_games) {
            eval_agents = num_games;
        }
        eval_agents += (-eval_agents) % (match ? 4 : 2);

        char agents_buf[64];
        snprintf(agents_buf, sizeof(agents_buf), "%ld", eval_agents);
        puf_ini_put(ini, "vec.num_buffers", "2");
        puf_ini_put(ini, "vec.total_agents", agents_buf);
    }
    if (match) {
        // Enemy bank uses same arch as policy (match loads both checkpoints).
        char hidden_buf[32], layers_buf[32];
        snprintf(hidden_buf, sizeof(hidden_buf), "%d",
            (int)puf_ini_get(ini, "policy", "hidden_size"));
        snprintf(layers_buf, sizeof(layers_buf), "%d",
            (int)puf_ini_get(ini, "policy", "num_layers"));
        puf_ini_put(ini, "vec.num_frozen_banks", "1");
        puf_ini_put(ini, "vec.frozen_bank_pct", "1");
        puf_ini_put(ini, "vec.frozen_bank_hidden_size", hidden_buf);
        puf_ini_put(ini, "vec.frozen_bank_num_layers", layers_buf);
        puf_ini_put(ini, "selfplay.enabled", "0");
        puf_ini_put(ini, "env.dr", "0");
        puf_ini_put(ini, "env.num_agents", "2");
        puf_ini_put(ini, "env.num_bots", "0");
    }
    static const char* const eval_common[][2] = {
        {"base.reset_state", "0"},
        {"train.horizon", "1"},
    };
    puf_ini_put_pairs(ini, eval_common, 2);

    PuffeRL* pufferl = create_pufferl(ini, ctx);
    if (match) {
        char a_path_buf[4096];
        char b_path_buf[4096];
        const char* a_path = puf_checkpoint_path_key(ini,
            "load_model_path", a_path_buf, sizeof(a_path_buf));
        const char* b_path = puf_checkpoint_path_key(ini,
            "load_enemy_model_path", b_path_buf, sizeof(b_path_buf));
        if (!a_path || !b_path) {
            fprintf(stderr, "match requires base.load_model_path and base.load_enemy_model_path\n");
            exit(1);
        }
        puf_load_weights_into(pufferl->master_weights,
            pufferl->param_puf, pufferl->default_stream, a_path);
        pufferl_load_frozen_bank(pufferl, 0, b_path);
    } else {
        char resolved_path[4096];
        const char* load_path = puf_checkpoint_path_key(ini,
            "load_model_path", resolved_path, sizeof(resolved_path));
        if (load_path) {
            puf_load_weights_into(pufferl->master_weights, pufferl->param_puf,
                pufferl->default_stream, load_path);
            printf("Loaded weights from %s\n", load_path);
        }
    }

    Dict baseline = {0};
    long baseline_n = 0;
    for (;;) {
        if (render) {
            puf_render(&pufferl->vec->envs[0]);
        }
        rollouts(pufferl);
        Dict log = {0};
        trainer_eval_log(pufferl, &log);
        if (render) {
            puf_dashboard_print(ini, pufferl, &log, 0);
            continue;
        }

        long n = (long)puf_log_get_or(&log, "env/n", 0);
        if (match) {
            double score = puf_log_get_or(&log, "env/slot_0_score", 0);
            double draw = puf_log_get_or(&log, "env/draw_rate", 0);
            if (verbose) {
                double b = puf_log_get_or(&log, "env/slot_1_score", 0);
                printf("\rgames=%ld/%ld  A=%.3f  B=%.3f  draw=%.3f",
                    n, num_games, score, b, draw);
            }
            if (n >= num_games) {
                result.score = (float)score;
                result.draw = (float)draw;
                result.games = (int)n;
                break;
            }
            continue;
        }

        if (burnin_games > 0 && baseline_n == 0 && n >= burnin_games) {
            baseline = log;
            baseline_n = n;
            if (verbose) {
                printf("\rbot_eval_burnin=%ld/%ld", n, burnin_games);
            }
            continue;
        }

        double scored_n = n - baseline_n;
        double score = puf_log_get_or(&log, "env/score", 0);
        double perf = puf_log_get_or(&log, "env/perf", 0);
        if (baseline_n > 0 && scored_n > 0) {
            double base_n = (double)baseline_n;
            double cur_n = (double)n;
            score = (score * cur_n - puf_log_get_or(&baseline, "env/score", 0) * base_n) / scored_n;
            perf = (perf * cur_n - puf_log_get_or(&baseline, "env/perf", 0) * base_n) / scored_n;
        }
        if (verbose) {
            printf("\rbot_eval=%.0f/%ld  perf=%.4f  score=%.3f",
                scored_n, num_games, perf, score);
        }
        if ((n - baseline_n) >= num_games && (!burnin_games || baseline_n > 0)) {
            result.score = (float)score;
            result.games = (int)scored_n;
            break;
        }
    }
    if (!render && verbose) {
        printf("\n");
    }
    close_pufferl(pufferl);
    return result;
}

TrainResult run_train(Ini* ini, TrainContext* ctx) {
    int use_selfplay = puf_ini_get(ini, "selfplay", "enabled");
    if (!use_selfplay) {
        static const char* const off[][2] = {
            {"vec.num_frozen_banks", "0"},
            {"vec.frozen_bank_pct", "0"},
        };
        puf_ini_put_pairs(ini, off, 2);
    }

    char run_id[64];
    const char* configured_run_id = puf_ini_get_str(ini, "base", "run_id");
    if (!configured_run_id[0] || strcmp(configured_run_id, "None") == 0) {
        snprintf(run_id, sizeof(run_id), "%ld", (long)(1000.0 * wall_clock()));
        puf_ini_put(ini, "base.run_id", run_id);
    } else {
        snprintf(run_id, sizeof(run_id), "%s", configured_run_id);
    }

    char checkpoint_dir[2048];
    char log_dir[2048];
    snprintf(checkpoint_dir, sizeof(checkpoint_dir), "%s/%s/%s",
        puf_ini_get_str(ini, "base", "checkpoint_dir"),
        puf_ini_get_str(ini, "base", "env_name"), run_id);
    snprintf(log_dir, sizeof(log_dir), "%s/%s",
        puf_ini_get_str(ini, "base", "log_dir"),
        puf_ini_get_str(ini, "base", "env_name"));
    if (ctx->artifact_owner) {
        mkdir_p(checkpoint_dir);
        mkdir_p(log_dir);
    }

    PuffeRL* pufferl = create_pufferl(ini, ctx);
    char initial_checkpoint[4096] = {0};
    if (use_selfplay) {
        snprintf(initial_checkpoint, sizeof(initial_checkpoint),
            "%s/%016ld.bin", checkpoint_dir, pufferl->global_step);
        if (ctx->artifact_owner) {
            puf_save_weights(pufferl, initial_checkpoint);
        }
    }

    Selfplay* selfplay = NULL;
    if (use_selfplay) {
        selfplay = (Selfplay*)calloc(1, sizeof(Selfplay));
        selfplay_init(selfplay, ini, pufferl, initial_checkpoint);
    }

    long total_timesteps = puf_ini_get(ini, "train", "total_timesteps");
    long batch_size = (long)puf_ini_get(ini, "vec", "total_agents") *
        (long)puf_ini_get(ini, "train", "horizon");
    long local_timesteps = total_timesteps / ctx->world_size;
    long train_epochs = local_timesteps / batch_size;
    long eval_epochs = train_epochs / 2;
    long checkpoint_interval = puf_ini_get(ini, "base", "checkpoint_interval");
    long eval_episodes = puf_ini_get(ini, "base", "eval_episodes");
    // Sweep objective: bare names → env/<name>; keys with '/' used as-is.
    char target_key[128];
    const char* metric = puf_ini_get_str(ini, "sweep", "metric");
    if (strchr(metric, '/')) {
        snprintf(target_key, sizeof(target_key), "%s", metric);
    } else {
        snprintf(target_key, sizeof(target_key), "env/%s", metric);
    }
    Dict last_log = {0};
    PufLogHistory log_history = {0};
    TrainResult result = {0};
    double last_dashboard_time = 0;

    for (long epoch = 0; epoch < train_epochs + eval_epochs; epoch++) {
        if (epoch < train_epochs && pufferl->hypers.async) {
            async_train_step(pufferl, epoch + 1 < train_epochs);
        } else {
            rollouts(pufferl);
            if (epoch < train_epochs) {
                train_impl(*pufferl, NULL);
            }
        }

        bool is_final = epoch == train_epochs - 1;
        bool interval_save = checkpoint_interval > 0 &&
            (epoch + 1) % checkpoint_interval == 0;
        bool should_save = epoch < train_epochs && (interval_save || is_final);
        char saved_checkpoint[4096] = {0};
        if (should_save) {
            snprintf(saved_checkpoint, sizeof(saved_checkpoint),
                "%s/%016ld.bin", checkpoint_dir, pufferl->global_step);
            if (ctx->artifact_owner) {
                puf_save_weights(pufferl, saved_checkpoint);
                snprintf(result.checkpoint_path, sizeof(result.checkpoint_path),
                    "%s", saved_checkpoint);
            }
        }
        if (selfplay && saved_checkpoint[0]) {
            selfplay_add_checkpoint(selfplay, saved_checkpoint);
        }

        int is_eval = epoch >= train_epochs;
        if (!is_eval && last_log.size &&
                wall_clock() < pufferl->last_log_time + 0.6 && epoch < train_epochs - 1) {
            continue;
        }

        Dict new_log = {0};
        if (is_eval) {
            trainer_eval_log(pufferl, &new_log);
        } else {
            trainer_log(pufferl, &new_log);
        }
        puf_log_update(&last_log, &new_log);
        if (selfplay && !is_eval) {
            selfplay_step(selfplay, pufferl, &last_log);
        }

        int eval_done = is_eval && puf_log_get_or(&last_log, "env/n", 0) > eval_episodes;
        int loop_done = epoch == train_epochs + eval_epochs - 1;
        double now = wall_clock();
        int print_dashboard = !is_eval || eval_done || loop_done || now >= last_dashboard_time + 0.6;
        if (ctx->artifact_owner && print_dashboard) {
            puf_dashboard_print(ini, pufferl, &last_log, (int)pufferl->epoch);
            last_dashboard_time = now;
        }

        // Wait until the objective appears; do not treat negative values as missing.
        if (!dict_find(&last_log, target_key)) {
            continue;
        }
        if (!is_eval) {
            puf_log_history_add(&log_history, &last_log);
        }
        if (eval_done) {
            break;
        }
    }

    // Selfplay: single final observation only (pool eval or last train metric).
    // Normal sweeps: downsampled learning curve for multi-fidelity protein.
    train_result_fill(&result, &log_history, &last_log, ini, target_key,
        /*single_point=*/use_selfplay);
    close_pufferl(pufferl);

    // Selfplay end rating: match final checkpoint vs a small fixed opponent pool.
    // Mean A winrate is the sole protein score when eval_games > 0.
    if (selfplay && result.checkpoint_path[0] && ctx->artifact_owner) {
        float pool_score = selfplay_pool_eval(ini, ctx, selfplay, result.checkpoint_path);
        if (isfinite(pool_score)) {
            result.score = pool_score;
            result.points = 1;
            result.scores[0] = pool_score;
            result.costs[0] = result.cost;
            result.step_points[0] = result.steps;
            dict_set(&last_log, "selfplay/pool_score", pool_score);
        }
    }

    if (ctx->artifact_owner) {
        puf_log_history_add(&log_history, &last_log);
        char log_path[4096];
        snprintf(log_path, sizeof(log_path), "%s/%s.ini", log_dir, run_id);
        puf_log_write(log_path, ini, &log_history);
    }
    puf_log_history_free(&log_history);
    free(selfplay);
    return result;
}

// --- Train launch: one DP job on a GPU block ---
//
// Device map within a block of size world_size starting at gpu_offset:
//   rank 0 (artifact owner) → offset + world_size - 1
//   rank r (r>=1)           → offset + r - 1
// Sweep packing only sets base.gpu_offset; launch_train always does the rest.
// Nested multi-GPU trials are free: each sweep slot is a full train process
// that may fork NCCL ranks inside its block (train.gpus > 1).

static int train_gpu_id(int rank, int world_size, int gpu_offset) {
    int local = (rank == 0) ? world_size - 1 : rank - 1;
    return gpu_offset + local;
}

static void wait_train_ranks(pid_t* pids, int n) {
    for (int i = 0; i < n; i++) {
        int status = 0;
        if (waitpid(pids[i], &status, 0) < 0) {
            fprintf(stderr, "waitpid failed for rank worker %d: %s\n",
                (int)pids[i], strerror(errno));
            exit(1);
        }
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            fprintf(stderr, "train rank worker pid %d failed\n", (int)pids[i]);
            exit(1);
        }
    }
}

// Sole multi-GPU DP launcher. Forks ranks 1..W-1 (before CUDA init), parent is
// rank 0. Reads train.gpus and base.gpu_offset. Optionally writes TrainResult
// to base.result_fd (sweep child).
TrainResult launch_train(Ini* ini) {
    int world_size = (int)puf_ini_get(ini, "train", "gpus");
    if (world_size < 1) {
        fprintf(stderr, "config error: [train] gpus must be >= 1\n");
        exit(1);
    }
    int gpu_offset = (int)puf_ini_get(ini, "base", "gpu_offset");

    ncclUniqueId nccl_id;
    ncclUniqueId* nccl_ptr = NULL;
    if (world_size > 1) {
        ncclGetUniqueId(&nccl_id);
        nccl_ptr = &nccl_id;
    }

    pid_t* pids = NULL;
    int n_workers = world_size - 1;
    if (n_workers > 0) {
        pids = (pid_t*)calloc((size_t)n_workers, sizeof(pid_t));
        for (int rank = world_size - 1; rank >= 1; rank--) {
            pid_t pid = fork();
            if (pid < 0) {
                fprintf(stderr, "fork failed: %s\n", strerror(errno));
                exit(1);
            }
            if (pid == 0) {
                if (!freopen("/dev/null", "w", stdout)) {
                    fprintf(stderr, "failed to silence rank worker stdout\n");
                    exit(1);
                }
                TrainContext child = {
                    .rank = rank,
                    .world_size = world_size,
                    .gpu_id = train_gpu_id(rank, world_size, gpu_offset),
                    .artifact_owner = 0,
                    .nccl_id = nccl_ptr,
                };
                run_train(ini, &child);
                puf_ini_free(ini);
                exit(0);
            }
            pids[rank - 1] = pid;
        }
    }

    TrainContext host = {
        .rank = 0,
        .world_size = world_size,
        .gpu_id = train_gpu_id(0, world_size, gpu_offset),
        .artifact_owner = 1,
        .nccl_id = nccl_ptr,
    };
    TrainResult result = run_train(ini, &host);
    if (n_workers > 0) {
        wait_train_ranks(pids, n_workers);
        free(pids);
    }

    // Sweep parent reads this over the pipe; CLI train ignores (result_fd=0).
    int result_fd = (int)puf_ini_get(ini, "base", "result_fd");
    if (result_fd > 0) {
        if (write(result_fd, &result, sizeof(result)) != (ssize_t)sizeof(result)) {
            fprintf(stderr, "failed to write train result\n");
            exit(1);
        }
        close(result_fd);
    }
    return result;
}

#ifdef PUFFERLIB_BUILD_MAIN
int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
    if (argc < 3) {
        fprintf(stderr, "usage: %s train|eval|eval_bot|match|sweep ENV [section.key=value ...]\n", argv[0]);
        exit(1);
    }

    const char* mode = argv[1];
    const char* env_name = argv[2];
    Ini* ini = (Ini*)calloc(1, sizeof(Ini));
    puf_ini_load_env(ini, env_name, argc - 3, argv + 3);
    char verr[256];
    if (!puf_config_train_valid(ini, verr, sizeof(verr))) {
        puf_die("%s", verr);
    }
    TrainContext ctx = {
        .rank = 0,
        .world_size = 1,
        .gpu_id = 0,
        .artifact_owner = 1,
        .nccl_id = NULL,
    };

    if (strcmp(mode, "train") == 0) {
        launch_train(ini);
    } else if (strcmp(mode, "sweep") == 0) {
        run_sweep(ini, argv[0]);
    } else if (strcmp(mode, "eval") == 0 || strcmp(mode, "eval_bot") == 0) {
        int bot = strcmp(mode, "eval_bot") == 0;
        if (bot) {
            static const char* const bot_puts[][2] = {
                {"vec.num_frozen_banks", "0"},
                {"vec.frozen_bank_pct", "0"},
                {"selfplay.enabled", "0"},
                {"env.dr", "0"},
                {"env.num_agents", "1"},
                {"env.num_bots", "1"},
            };
            puf_ini_put_pairs(ini, bot_puts, 6);
        }
        run_eval(ini, &ctx, bot ? EVAL_SCORE : EVAL_RENDER, 1);
    } else if (strcmp(mode, "match") == 0) {
        run_eval(ini, &ctx, EVAL_MATCH, 1);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        exit(1);
    }

    puf_ini_free(ini);
    free(ini);
    return 0;
}

#endif

