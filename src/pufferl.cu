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
constexpr cublasComputeType_t CUBLAS_COMPUTE = CUBLAS_COMPUTE_32F;
#define NCCL_PRECISION ncclFloat
#define to_float(x) (x)
#define from_float(x) (x)
#else
typedef __nv_bfloat16 precision_t;
constexpr bool USE_BF16 = true;
constexpr cudaDataType_t CUBLAS_PRECISION = CUDA_R_16BF;
constexpr cublasComputeType_t CUBLAS_COMPUTE = CUBLAS_COMPUTE_32F;
#define NCCL_PRECISION ncclBfloat16
#define to_float(x) __bfloat162float(x)
#define from_float(x) __float2bfloat16(x)
#endif

#define PUF_MAX_DIMS 8
#define BLOCK_SIZE 256
int grid_size(int N) {
    return (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

// Exclusive env: -DENV_HEADER=ocean/<env>/<env>.h or .cu (--gpu). Never both.
#include ENV_HEADER

typedef struct {
    float* data;
    int64_t shape[PUF_MAX_DIMS];
} Float;

typedef struct {
    unsigned char* data;
    int64_t shape[PUF_MAX_DIMS];
} Byte;

typedef struct {
    long* data;
    int64_t shape[PUF_MAX_DIMS];
} Long;

typedef struct {
    int* data;
    int64_t shape[PUF_MAX_DIMS];
} Int;

typedef struct {
    precision_t* data;
    int64_t shape[PUF_MAX_DIMS];
} Prec;

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

void squeeze_shape(int64_t* shape, int dim) {
    int n = ndim(shape);
    shape[dim] *= shape[dim + 1];
    for (int i = dim + 1; i < n - 1; i++) {
        shape[i] = shape[i + 1];
    }
    shape[n - 1] = 0;
}

Prec* puf_squeeze(Prec* t, int dim) {
    squeeze_shape(t->shape, dim);
    return t;
}

Float* puf_squeeze(Float* t, int dim) {
    squeeze_shape(t->shape, dim);
    return t;
}

Prec* puf_unsqueeze(Prec* t, int dim, int64_t d0, int64_t d1) {
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

void puf_copy(Prec* dst, Prec* src, cudaStream_t stream) {
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

#ifdef PRECISION_FLOAT
// GPU envs fix obs as bf16; float train needs an explicit widen.
__global__ void cast(precision_t* dst,
        __nv_bfloat16* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __bfloat162float(src[idx]);
    }
}
#else
// Identity overload so obs→rollout cast arm typechecks when obs_t is bf16.
__global__ void cast(precision_t* dst,
        precision_t* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

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

// Fused rew+term cast
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

// Allocates large buffers (params/acts/grads) in contiguous memory.
struct AllocEntry {
    void** data_ptr;
    long* shape;
    int elem_size;
};

struct Allocator {
    AllocEntry* regs;
    int num_regs;
    void* mem;
    long total_elems;
    long total_bytes;
};

void _alloc_register(Allocator* alloc,
        void** data_ptr, long* shape, int elem_size) {
    alloc->regs = (AllocEntry*)realloc(alloc->regs,
        (alloc->num_regs + 1) * sizeof(AllocEntry));
    alloc->regs[alloc->num_regs++] = {data_ptr, shape, elem_size};
    long n = numel(shape);
    alloc->total_elems += n;
    alloc->total_bytes = (alloc->total_bytes + 15) & ~15;
    alloc->total_bytes += n * elem_size;
}
void alloc_register(Allocator* a, Prec* t) {
    _alloc_register(a, (void**)&t->data, t->shape, sizeof(precision_t));
}
void alloc_register(Allocator* a, Float* t) {
    _alloc_register(a, (void**)&t->data, t->shape, sizeof(float));
}
void alloc_register(Allocator* a, Long* t) {
    _alloc_register(a, (void**)&t->data, t->shape, sizeof(long));
}
void alloc_register(Allocator* a, Int* t) {
    _alloc_register(a, (void**)&t->data, t->shape, sizeof(int));
}

void alloc_create(Allocator* alloc) {
    assert(cudaMalloc(&alloc->mem, alloc->total_bytes) == cudaSuccess
        && "alloc_create: cudaMalloc failed");
    cudaMemset(alloc->mem, 0, alloc->total_bytes);
    long offset = 0;
    for (int i = 0; i < alloc->num_regs; i++) {
        offset = (offset + 15) & ~15;
        *alloc->regs[i].data_ptr = (char*)alloc->mem + offset;
        offset += numel(alloc->regs[i].shape) * alloc->regs[i].elem_size;
    }
}

// Algo + sweeps only depend on basic tensor types and utilities
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
    bool reset_every_horizon;
    bool cudagraphs;
    bool profile;
    int rank;
    int world_size;
    int gpu_id;
    int num_threads;
    int seed;
} Hypers;

// Rank / device context for one process in a multi-GPU train job.
typedef struct {
    int rank;
    int world_size;
    int gpu_id;
    int artifact_owner;
    ncclUniqueId* nccl_id;
} TrainContext;

typedef struct ObsTensor {
    obs_t* data;
    int64_t shape[8];
} ObsTensor;

// Each rollout buffer manages a constant subset of environments
// Simulation + inference overlap across buffers
// Note: experimental async path adds an extra slot dimension.
struct RolloutBuf {
    Prec observations;  // (horizon, agents, input_size)
    Prec initial_states;
    Prec actions;       // (horizon, agents, num_atns)
    Prec values;        // (horizon, agents)
    Prec logprobs;      // ...
    Prec rewards;
    Prec terminals;
    Prec ratio;
    Prec importance;
    Prec action_mask;   // (horizon, agents, mask_size)
};

// Buffers are initialized as raw structs with only shape information.
// alloc_register stores the shape and data pointer.
// Memory is only allocated after all buffers are registered.
void register_rollout_buffers(RolloutBuf* bufs, Allocator* alloc,
        int T, int B, int input_size, int num_atns, int mask_size) {
    memset(bufs, 0, sizeof(*bufs));
    bufs->observations = {.shape = {T, B, input_size}};
    bufs->actions      = {.shape = {T, B, num_atns}};
    bufs->values       = {.shape = {T, B}};
    bufs->logprobs     = {.shape = {T, B}};
    bufs->rewards      = {.shape = {T, B}};
    bufs->terminals    = {.shape = {T, B}};
    bufs->ratio        = {.shape = {T, B}};
    bufs->importance   = {.shape = {T, B}};
    bufs->action_mask  = {.shape = {T, B, mask_size}};
    Prec* fields[] = {
        &bufs->observations, &bufs->actions, &bufs->values, &bufs->logprobs,
        &bufs->rewards, &bufs->terminals, &bufs->ratio, &bufs->importance,
        &bufs->action_mask,
    };
    for (int i = 0; i < (int)(sizeof(fields) / sizeof(fields[0])); i++) {
        alloc_register(alloc, fields[i]);
    }
}

// Rank-2 or rank-3 time-major tensor. F==0 means rank-2 (zero-terminated shape);
// stride still multiplies by max(F, 1).
Prec puf_time_view(Prec p, int start_t, int T) {
    long B = p.shape[1];
    long F = p.shape[2];
    long stride_f = F > 1 ? F : 1;
    return {
        .data = p.data + (long)start_t * B * stride_f,
        .shape = {T, B, F},
    };
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

// Env batch. Device IO is always PuffeRL.env (EnvBuf).
// CPU: host pins + workers. GPU: single-buffer device envs (no host pins).
struct VecEnv {
    Env* envs;  // CPU: host array. GPU: device base (batch).
    int size;
    int total_agents;
    int buffers;
    int agents_per_buf;
    int mask_size;
    int num_banks;
    int* bank_layout;
    // CPU pins / workers (unused on GPU)
    int* env_starts;
    int* env_counts;
    obs_t* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned char* action_mask;
    int* worker_state;
    int shutdown;
    pthread_t* threads;
    float* accum;
    int num_workers;
    // GPU log reduce scratch (unused on CPU)
    float* log_scratch;
};

struct EnvBuf {
    ObsTensor obs;    // (total_agents, obs_size)
    Float actions;    // (total_agents, num_atns)
    Float rewards;    // (total_agents,)
    Float terminals;  // (total_agents,)
    Byte action_mask; // (total_agents, mask_size); always allocated
};

// Frozen opponent bank (selfplay / match): rollout-only policy + params.
// Same build_policy / weights_create path as primary, but no train acts, grads, or muon.
// Optional different hidden/layers via frozen_bank_hidden_size / frozen_bank_num_layers.
typedef struct {
    Policy policy;
    PolicyWeights weights;
    Allocator params_alloc;
    Allocator activ_alloc;
    Prec param;
    Float master_weights;
    Prec* buffer_states;         // [num_buffers]
    PolicyActivations* buf_acts;  // [num_buffers]
} WeightBank;

// PROF_* accum; MODEL_*/H2D_* worker+GPU rollout events; TE_* train_impl events.
enum {
    PROF_ROLLOUT,
    PROF_MODEL,
    PROF_ENV,
    PROF_COPY,
    PROF_TRAIN_MISC,
    PROF_TRAIN_MODEL,
    NUM_PROF,
    MODEL_START = 0,
    MODEL_END,
    COPY_END,
    ENV_END = COPY_END,
    H2D_START,
    H2D_END,
    NUM_EV,
    EV_T = ENV_END + 1,
    TE_S = 0,
    TE_E,
    TE_MS,
    TE_ME,
    TE_FE,
    NUM_TE,
};

// Index must match PROF_* order above (first NUM_PROF names).
const char* PROF_NAMES[] = {
    "rollout",
    "eval_model",
    "eval_env",
    "eval_copy",
    "train_misc",
    "train_model",
};

typedef struct {
    cudaEvent_t events[NUM_TE];
    cudaEvent_t* rollout_ev;  // GPU [EV_T * horizon]; null on CPU
    float accum[NUM_PROF];
} Profile;

typedef struct PuffeRL {
    Policy policy;
    PolicyWeights weights;       // current precision_t weights (structured)
    PolicyWeights actor_weights; // async rollout snapshot; unused when async=0
    PolicyActivations train_activs;
    Allocator params_alloc;
    Allocator weight_alloc;
    Allocator grads_alloc;
    Allocator activ_alloc;
    VecEnv* vec;
    Muon muon;
    ncclComm_t nccl_comm;  // NCCL communicator for multi-GPU
    Hypers hypers;
    bool is_continuous;  // True if all action dimensions are continuous (size==1)
    Prec* buffer_states;  // Per-buffer states for contiguous access
    PolicyActivations* buf_acts;  // Per-buffer inference activations
    RolloutBuf rollouts;
    RolloutBuf train_rollouts;  // Pre-allocated transposed copy for train_impl
    EnvBuf env;
    TrainGraph train_buf;
    Prec advantages;  // Pre-allocated for train_impl (B, T)
    cudaGraphExec_t* rollout_graphs;  // [slots][horizon][num_buffers]; null if !cudagraphs
    cudaGraphExec_t train_cudagraph;  // null until first-use capture
    cudaStream_t* streams;  // per-buffer raw CUDA streams
    cudaStream_t default_stream;  // main-thread stream (captured once at init)
    cudaStream_t train_stream;    // dedicated learner stream (always non-default)
    int* act_sizes;        // device ACT_SIZES (NUM_ATNS ints)
    float* losses;         // device loss accumulator (NUM_LOSSES)
    PPOBufs ppo_bufs; // Pre-allocated buffers for ppo_loss_fwd_bwd
    PrioBuffers prio_bufs;      // Pre-allocated buffers for prio_replay
    Float master_weights;  // fp32 master weights (flat); same buffer as param in fp32 mode
    Prec param;
    Prec actor_param;
    Prec grad;
    long* rng_offset;      // device counters (num_buffers+1)
    Profile profile;
    nvmlDevice_t nvml_device;
    long epoch;
    long global_step;
    double start_time;
    double last_log_time;
    long last_log_step;
    int write_slot;
    int async_ready_slot;
    int async_next_slot;
    bool async_boot;
    ulong seed;
    curandStatePhilox4_32_10_t** rng_states;  // per-buffer persistent RNG states [num_buffers]
    // Optional frozen weight banks for match / selfplay opponents.
    WeightBank* frozen_banks;  // [num_frozen_banks]
    int num_frozen_banks;
    char env_name[64];  // For frozen-bank policy rebuild at create.
} PuffeRL;

// --- Infer path: sample + forward, then vec workers ---
static void profile_begin(const char* tag, bool enable) {
    if (enable) {
        nvtxRangePushA(tag);
    }
}

static void profile_end(bool enable) {
    if (enable) {
        nvtxRangePop();
    }
}

__global__ void rng_init(curandStatePhilox4_32_10_t* states, uint64_t seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Action logits and value share one row: [logits..., value]. logstd empty ⇒ discrete.
// Discrete: always-cache logsumexp + inverse-CDF; mask always present (all-ones if env has none).
// Continuous: ignores mask.
__global__ void sample_logits(
        Prec dec_out,              // (B, logits_dim + 1)
        Prec logstd,           // (1, od) continuous only; .data null if discrete
        int* act_sizes,            // (NUM_ATNS,)
        precision_t* actions,                 // (B, num_atns)
        precision_t* logprobs,                // (B,)
        precision_t* value_out,               // (B,)
        curandStatePhilox4_32_10_t* rng_states,
        precision_t* action_mask,             // (B, A_total); always allocated
        int mask_stride,                      // 0 when unused
        const signed char* head_consume,      // (nverbs, num_atns) or NULL
        int hc_stride) {
    int B = dec_out.shape[0];
    int fused_cols = dec_out.shape[1];
    int num_atns = NUM_ATNS;
    precision_t* logits = dec_out.data;
    bool is_continuous = logstd.data != NULL && numel(logstd.shape) > 0;
    precision_t* logstd_data = logstd.data;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B) {
        return;
    }

    curandStatePhilox4_32_10_t state = rng_states[idx];
    int logits_base = idx * fused_cols;
    float total_log_prob = 0.0f;

    if (is_continuous) {
        for (int h = 0; h < num_atns; h++) {
            float mean = safe_continuous_mean(logits, logits_base + h);
            float log_std = safe_continuous_logstd(logstd_data, h);
            float std = expf(log_std);
            float action = finite_or_clamp(
                mean + std * curand_normal(&state), -1.0e6f, 1.0e6f);
            // Round-trip so logprob matches stored precision_t action (bf16).
            precision_t stored_p = from_float(action);
            float lp, ent;
            ppo_continuous_head(mean, log_std, to_float(stored_p), &lp, &ent);
            total_log_prob += lp;
            actions[idx * num_atns + h] = stored_p;
        }
    } else {
        int logits_offset = 0;
        int mask_base = idx * mask_stride;
        for (int h = 0; h < num_atns; h++) {
            int A = act_sizes[h];
            float cache[PPO_MAX_HEAD_A];
            float logsumexp = ppo_discrete_logsumexp(
                logits, logits_base, logits_offset, A, action_mask, mask_base, cache);

            float rand_val = curand_uniform(&state);
            float cumsum = 0.0f;
            int sampled = A - 1;
            for (int a = 0; a < A; a++) {
                cumsum += expf(cache[a] - logsumexp);
                if (rand_val < cumsum) {
                    sampled = a;
                    break;
                }
            }
            actions[idx * num_atns + h] = from_float((float)sampled);
            // consumed-head gating: only heads the sampled verb uses count
            int verb = (int)to_float(actions[idx * num_atns]);
            int used = (head_consume == NULL || h == 0)
                ? 1 : (int)head_consume[verb * hc_stride + h];
            if (used) {
                total_log_prob += cache[sampled] - logsumexp;
            }
            logits_offset += A;
        }
    }

    logprobs[idx] = from_float(total_log_prob);
    value_out[idx] = logits[logits_base + fused_cols - 1];
    rng_states[idx] = state;
}

// Index into (L, agents, H): element-parallel over L*count*H.
// state_row is agent index within the state tensor's agent dim.
static __device__ long state_elem_idx(
        int layer, int agents_stride, int state_row, int h, int H) {
    return ((long)layer * agents_stride + state_row) * H + h;
}

// Copy buffer RNN state into rollout initial_states at t=0 (carry path).
// Primary bank only; src agents are 0..count, dst at dst_start.
__global__ void snapshot_state(Prec dst, Prec src,
        int dst_start, int count) {
    int L = src.shape[0];
    int H = src.shape[2];
    int total = L * count * H;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int h = idx % H;
    int rel = (idx / H) % count;
    int layer = idx / (count * H);
    long src_i = state_elem_idx(layer, (int)src.shape[1], rel, h, H);
    long dst_i = state_elem_idx(layer, (int)dst.shape[1], dst_start + rel, h, H);
    dst.data[dst_i] = src.data[src_i];
}

// View slot of full (slots, L, A, H) as (L, A, H).
static Prec init_slot(Prec full, int slot) {
    int L = (int)full.shape[1];
    int A = (int)full.shape[2];
    int H = (int)full.shape[3];
    long stride = (long)L * A * H;
    return {.data = full.data + (long)slot * stride, .shape = {L, A, H}};
}

// Zero RNN state for agents that just terminated. Same grid as snapshot
// (L*count*H); non-terminal threads return after one terminal load.
__global__ void zero_term_state(Prec state, Float terminals,
        int state_start, int terminal_start, int count) {
    int L = state.shape[0];
    int H = state.shape[2];
    int total = L * count * H;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int h = idx % H;
    int rel = (idx / H) % count;
    int layer = idx / (count * H);
    if (terminals.data[terminal_start + rel] == 0.0f) {
        return;
    }
    long i = state_elem_idx(layer, (int)state.shape[1], state_start + rel, h, H);
    state.data[i] = from_float(0.0f);
}

// Select time t, then agents [start, start+count). Rank-2 has F==0 (zero-term shape);
// stride uses max(F, 1). Out shape {count, F} keeps ndim 1 when F==0.
Prec puf_slice(Prec p, int t, int start, int count) {
    long B = p.shape[1];
    long F = p.shape[2];
    long stride_f = F > 1 ? F : 1;
    return {
        .data = p.data + (long)(t * B + start) * stride_f,
        .shape = {count, F},
    };
}

void pufferl_forward(PuffeRL* pufferl, int buf, int t, cudaStream_t stream) {
    Hypers* hypers = &pufferl->hypers;
    int graph_slot = hypers->async ? pufferl->write_slot : 0;
    int graph = (graph_slot * hypers->horizon + t) * hypers->num_buffers + buf;
    profile_begin("fused_rollout", hypers->profile);

    if (hypers->cudagraphs && pufferl->rollout_graphs[graph] != NULL) {
        assert(cudaGraphLaunch(
            pufferl->rollout_graphs[graph], stream) == cudaSuccess
            && "cudaGraphLaunch failed");
        profile_end(hypers->profile);
        return;
    }

    if (hypers->cudagraphs) {
        assert(cudaStreamBeginCapture(
            stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess
            && "cudaStreamBeginCapture failed");
    }

    RolloutBuf rollouts = pufferl->rollouts;
    if (hypers->async) {
        rollouts = rollout_time_view(&pufferl->rollouts,
            pufferl->write_slot * hypers->horizon, hypers->horizon);
    }
    EnvBuf* env = &pufferl->env;
    VecEnv* vec = pufferl->vec;
    int block_size = hypers->total_agents / hypers->num_buffers;
    int start = buf * block_size;
    int* layout = vec->bank_layout;

    // Copy observations, rewards, terminals from GPU env buffers to rollout buffer
    ObsTensor* obs_env = &env->obs;
    int n = block_size * obs_env->shape[1];
    Prec obs_dst = puf_slice(rollouts.observations, t, start, block_size);
    // Env obs → rollout: D2D if same type, else cast (float/uchar → precision_t).
    if (sizeof(obs_t) == sizeof(precision_t)) {
        cudaMemcpyAsync(obs_dst.data,
            obs_env->data + (long)start * obs_env->shape[1],
            n * sizeof(precision_t), cudaMemcpyDeviceToDevice, stream);
    } else {
        cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
            obs_dst.data, obs_env->data + (long)start * obs_env->shape[1], n);
    }

    Prec rew_dst = puf_slice(rollouts.rewards, t, start, block_size);
    Prec term_dst = puf_slice(rollouts.terminals, t, start, block_size);
    cast_rew_term<<<grid_size(block_size), BLOCK_SIZE, 0, stream>>>(
        rew_dst.data, env->rewards.data + start,
        term_dst.data, env->terminals.data + start, block_size);

    // Mask always allocated (env-written or synthetic all-ones). Continuous ignores it in sample.
    int mask_size = rollouts.action_mask.shape[2];
    int mask_stride = mask_size;
    Prec mask_slice = puf_slice(rollouts.action_mask, t, start, block_size);
    cast<<<grid_size(block_size * mask_size), BLOCK_SIZE, 0, stream>>>(
        mask_slice.data,
        env->action_mask.data + (long)start * mask_size,
        block_size * mask_size);

    // Per-bank forward: layout[b]..layout[b+1) within each buffer chunk.
    int num_banks = 1 + pufferl->num_frozen_banks;
    long act_cols = env->actions.shape[1];
    for (int b = 0; b < num_banks; b++) {
        int bank_off = layout[b];
        int bank_end = layout[b + 1];
        int bank_size = bank_end - bank_off;
        if (bank_size == 0) {
            continue;
        }

        Policy* p_bank;
        PolicyWeights* w_bank;
        PolicyActivations* a_bank;
        Prec* s_bank;
        if (b == 0) {
            p_bank = &pufferl->policy;
            w_bank = hypers->async ? &pufferl->actor_weights : &pufferl->weights;
            a_bank = &pufferl->buf_acts[buf];
            s_bank = &pufferl->buffer_states[buf];
        } else {
            WeightBank* fb = &pufferl->frozen_banks[b - 1];
            p_bank = &fb->policy;
            w_bank = &fb->weights;
            a_bank = &fb->buf_acts[buf];
            s_bank = &fb->buffer_states[buf];
        }

        int sub_start = start + bank_off;
        Prec obs_b  = puf_slice(rollouts.observations, t, sub_start, bank_size);
        Prec act_b  = puf_slice(rollouts.actions,      t, sub_start, bank_size);
        Prec lp_b   = puf_slice(rollouts.logprobs,     t, sub_start, bank_size);
        Prec val_b  = puf_slice(rollouts.values,       t, sub_start, bank_size);
        Prec mask_b = puf_slice(rollouts.action_mask,  t, sub_start, bank_size);

        int state_start = (b == 0) ? bank_off : 0;
        int state_n = (int)s_bank->shape[0] * bank_size * (int)s_bank->shape[2];
        zero_term_state<<<grid_size(state_n), BLOCK_SIZE, 0, stream>>>(
            *s_bank, env->terminals, state_start, sub_start, bank_size);

        // Carry path: snapshot primary buffer state into per-slot initial_states.
        if (b == 0 && t == 0 && rollouts.initial_states.data != NULL) {
            Prec slot_st = init_slot(rollouts.initial_states, graph_slot);
            snapshot_state<<<grid_size(state_n), BLOCK_SIZE, 0, stream>>>(
                slot_st, *s_bank, sub_start, bank_size);
        }

        Prec dec = policy_forward(p_bank, *w_bank, *a_bank, obs_b, *s_bank, stream);

        Prec p_logstd = {};
        DecoderWeights* dw = (DecoderWeights*)w_bank->decoder;
        if (dw->continuous) {
            p_logstd = dw->logstd;
        }

        // Offset RNG by bank_off so banks don't collide on per-buffer rng slots.
        int hc_stride_s = 0;
        const signed char* hc_dev_s = get_head_consume_dev(&hc_stride_s);
        sample_logits<<<grid_size(bank_size), BLOCK_SIZE, 0, stream>>>(
            dec, p_logstd, pufferl->act_sizes,
            act_b.data, lp_b.data, val_b.data,
            pufferl->rng_states[buf] + bank_off,
            mask_b.data, mask_stride, hc_dev_s, hc_stride_s);

        cast<<<grid_size(numel(act_b.shape)), BLOCK_SIZE, 0, stream>>>(
                env->actions.data + (long)sub_start * act_cols,
                act_b.data, numel(act_b.shape));
    }

    if (hypers->cudagraphs) {
        cudaGraph_t _graph;
        assert(cudaStreamEndCapture(stream, &_graph) == cudaSuccess
                && "cudaStreamEndCapture failed");
        assert(cudaGraphInstantiate(&pufferl->rollout_graphs[graph], _graph, 0)
                == cudaSuccess && "cudaGraphInstantiate failed");
        cudaGraphDestroy(_graph);
        // Capture records without executing; run once so this step has effects.
        assert(cudaGraphLaunch(pufferl->rollout_graphs[graph], stream) == cudaSuccess
                && "cudaGraphLaunch failed");
    }
    profile_end(hypers->profile);
}

double wall_clock() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Log is a flat float struct (host + device).
constexpr int LOG_NF = (int)(sizeof(Log) / sizeof(float));

// Shared host/device: fold Log into float acc; optionally clear the source.
__host__ __device__ static inline void log_accum(float* acc, Log* log, int clear) {
    if (log->n != 0.0f) {
        const float* el = (const float*)log;
        for (int j = 0; j < LOG_NF; j++) {
            acc[j] += el[j];
        }
    }
    if (clear) {
        float* el = (float*)log;
        for (int j = 0; j < LOG_NF; j++) {
            el[j] = 0.0f;
        }
    }
}

__global__ void log_reduce(Env* envs,
        float* out, int num_envs, int clear) {
    extern __shared__ float sh[];
    int tid = threadIdx.x;
    float local[LOG_NF] = {};
    for (int i = tid; i < num_envs; i += blockDim.x) {
        log_accum(local, &envs[i].log, clear);
    }
    for (int j = 0; j < LOG_NF; j++) {
        sh[j * blockDim.x + tid] = local[j];
    }
    block_reduce_sum(sh, out, tid, blockDim.x, LOG_NF);
}

static void env_log_sum(VecEnv* vec, Log* out, int clear) {
    if (PUF_BACKEND == PUF_GPU) {
        log_reduce<<<1, 256, LOG_NF * 256 * sizeof(float)>>>(
            vec->envs, vec->log_scratch, vec->size, clear);
        cudaMemcpy(out, vec->log_scratch, sizeof(Log), cudaMemcpyDeviceToHost);
        return;
    }
    float* acc = (float*)out;
    for (int i = 0; i < vec->size; i++) {
        log_accum(acc, &vec->envs[i].log, clear);
    }
}

#if PUF_BACKEND != PUF_GPU
static void puf_bind_stream(cudaStream_t) {}
static Env* puf_vec_create(int, Dict*, obs_t*, float*, float*, float*) {
    return NULL;
}
#endif

static void env_setup(PuffeRL* p, VecEnv* vec,
        Dict* vk, Dict* ek, int frozen_banks) {
    if (PUF_BACKEND == PUF_GPU) {
        assert(vec->buffers == 1 && "GPU env: num_buffers must be 1");
        assert(frozen_banks == 0 && "GPU env: no frozen banks");
        vec->size = vec->total_agents;
        vec->envs = puf_vec_create(vec->total_agents, ek,
            p->env.obs.data, p->env.actions.data,
            p->env.rewards.data, p->env.terminals.data);
        cudaMalloc((void**)&vec->log_scratch, sizeof(Log));
        cudaMemset(vec->log_scratch, 0, sizeof(Log));
        vec->bank_layout[0] = 0;
        vec->bank_layout[1] = vec->agents_per_buf;
        return;
    }
    int total_agents = vec->total_agents;
    int num_buffers = vec->buffers;
    int apb = vec->agents_per_buf;
    vec->num_workers = (int)dict_get(vk, "num_threads") / num_buffers;
    if (vec->num_workers < 1) {
        vec->num_workers = 1;
    }
    vec->env_starts = (int*)calloc(1, num_buffers * sizeof(int));
    vec->env_counts = (int*)calloc(1, num_buffers * sizeof(int));
    int num_envs = 0;
#ifdef MY_VEC_INIT
    vec->envs = my_vec_init(&num_envs, vec->env_starts,
        vec->env_counts, vk, ek);
#else
    Env* envs = (Env*)calloc(total_agents, sizeof(Env));
    int agents_created = 0;
    while (agents_created < total_agents) {
        envs[num_envs].rng = num_envs;
        puf_init(&envs[num_envs], ek);
        agents_created += envs[num_envs].num_agents;
        num_envs++;
    }
    envs = (Env*)realloc(envs, num_envs * sizeof(Env));
    int buf = 0, buf_agents = 0;
    for (int i = 0; i < num_envs; i++) {
        buf_agents += envs[i].num_agents;
        vec->env_counts[buf]++;
        if (buf_agents >= apb && buf < num_buffers - 1) {
            buf++;
            vec->env_starts[buf] = i + 1;
            buf_agents = 0;
        }
    }
    vec->envs = envs;
#endif
    vec->size = num_envs;

    size_t obs_bytes = total_agents * OBS_SIZE * sizeof(obs_t);
    size_t mask_bytes = total_agents * vec->mask_size * sizeof(unsigned char);
    cudaHostAlloc((void**)&vec->observations, obs_bytes, cudaHostAllocPortable);
    cudaHostAlloc((void**)&vec->actions, total_agents * NUM_ATNS * sizeof(float),
        cudaHostAllocPortable);
    cudaHostAlloc((void**)&vec->rewards, total_agents * sizeof(float),
        cudaHostAllocPortable);
    cudaHostAlloc((void**)&vec->terminals, total_agents * sizeof(float),
        cudaHostAllocPortable);
    cudaHostAlloc((void**)&vec->action_mask, mask_bytes, cudaHostAllocPortable);
    memset(vec->action_mask, 1, mask_bytes);

    float frozen_pct = (float)dict_get(vk, "frozen_bank_pct");
    for (int buf = 0; buf < num_buffers; buf++) {
        int buf_start = buf * apb;
        int env_start = vec->env_starts[buf];
        int env_count = vec->env_counts[buf];
        int frozen_start = env_count;
        if (frozen_banks > 0 && frozen_pct > 0.0f) {
            frozen_start = env_count - (int)(frozen_pct * env_count);
        }
        int* counts = (int*)calloc(vec->num_banks, sizeof(int));
        for (int e = 0; e < env_count; e++) {
            Env* eptr = &vec->envs[env_start + e];
            for (int s = 0; s < eptr->num_agents; s++) {
                int policy = e < frozen_start ? 0 : eptr->agents[s].policy;
                assert(policy >= 0 && policy < vec->num_banks);
                counts[policy]++;
            }
        }
        int offset = 0;
        for (int b = 0; b <= vec->num_banks; b++) {
            if (buf == 0) {
                vec->bank_layout[b] = offset;
            } else {
                assert(vec->bank_layout[b] == offset);
            }
            if (b < vec->num_banks) {
                offset += counts[b];
            }
        }
        assert(offset == apb);
        int* cursors = (int*)calloc(vec->num_banks, sizeof(int));
        for (int b = 0; b < vec->num_banks; b++) {
            cursors[b] = buf_start + vec->bank_layout[b];
        }
        for (int e = 0; e < env_count; e++) {
            Env* eptr = &vec->envs[env_start + e];
            int tag = 0;
            for (int s = 0; s < eptr->num_agents; s++) {
                int policy = e < frozen_start ? 0 : eptr->agents[s].policy;
                if (policy > tag) {
                    tag = policy;
                }
                int phys = cursors[policy]++;
                Agent* a = &eptr->agents[s];
                a->observations = vec->observations + (size_t)phys * OBS_SIZE;
                a->actions = vec->actions + (size_t)phys * NUM_ATNS;
                a->rewards = vec->rewards + phys;
                a->terminals = vec->terminals + phys;
                a->action_mask = vec->action_mask + (size_t)phys * vec->mask_size;
            }
            eptr->tag = tag;
            eptr->boundary_reached = 0;
        }
        free(cursors);
        free(counts);
    }
}

static void cpu_upload(PuffeRL* p, int start, int n, cudaStream_t stream) {
    VecEnv* v = p->vec;
    EnvBuf* e = &p->env;
    long mask = v->mask_size;
    cudaMemcpyAsync(e->obs.data + (size_t)start * OBS_SIZE,
        v->observations + (size_t)start * OBS_SIZE,
        n * OBS_SIZE * sizeof(obs_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(e->rewards.data + start, v->rewards + start,
        n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(e->terminals.data + start, v->terminals + start,
        n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(e->action_mask.data + (size_t)start * mask,
        v->action_mask + (size_t)start * mask,
        n * mask * sizeof(unsigned char), cudaMemcpyHostToDevice, stream);
}

// CPU worker handshake. Atomic on worker_state[]; calloc leaves BUF_STARTING.
enum {
    BUF_STARTING,
    BUF_WAITING,
    BUF_RUNNING,
};

typedef struct {
    PuffeRL* pufferl;
    int buf;
} VecThreadArg;

static void* vec_thread_main(void* arg) {
    VecThreadArg* a = (VecThreadArg*)arg;
    PuffeRL* pufferl = a->pufferl;
    VecEnv* vec = pufferl->vec;
    int buf = a->buf;
    int* state = &vec->worker_state[buf];
    int horizon = pufferl->hypers.horizon;
    cublas_init_handle();
    int apb = vec->agents_per_buf;
    int agent_start = buf * apb;
    int env_start = vec->env_starts[buf];
    int env_count = vec->env_counts[buf];
    Env* envs = vec->envs;
    cudaStream_t stream = pufferl->streams[buf];
    cudaEvent_t ev[NUM_EV];
    for (int i = 0; i < NUM_EV; i++) {
        cudaEventCreate(&ev[i]);
    }
    __atomic_store_n(state, BUF_WAITING, __ATOMIC_SEQ_CST);
    float* my_accum = &vec->accum[buf * NUM_PROF];
    struct timespec t0, t1;
    float ms = 0.0f;
    while (true) {
        while (__atomic_load_n(state, __ATOMIC_SEQ_CST) != BUF_RUNNING) {
            if (!__atomic_load_n(&vec->shutdown, __ATOMIC_SEQ_CST)) {
                continue;
            }
            for (int i = 0; i < NUM_EV; i++) {
                cudaEventDestroy(ev[i]);
            }
            return NULL;
        }
        int h2d_pending = 0;
        for (int t = 0; t < horizon; t++) {
            cudaEventRecord(ev[MODEL_START], stream);
            pufferl_forward(pufferl, buf, t, stream);
            cudaEventRecord(ev[MODEL_END], stream);
            cudaMemcpyAsync(
                &vec->actions[agent_start * NUM_ATNS],
                &pufferl->env.actions.data[agent_start * NUM_ATNS],
                apb * NUM_ATNS * sizeof(float),
                cudaMemcpyDeviceToHost, stream);
            cudaEventRecord(ev[COPY_END], stream);
            cudaStreamSynchronize(stream);
            cudaEventElapsedTime(&ms, ev[MODEL_START], ev[MODEL_END]);
            my_accum[PROF_MODEL] += ms;
            cudaEventElapsedTime(&ms, ev[MODEL_END], ev[COPY_END]);
            my_accum[PROF_COPY] += ms;
            if (h2d_pending) {
                cudaEventElapsedTime(&ms, ev[H2D_START], ev[H2D_END]);
                my_accum[PROF_COPY] += ms;
                h2d_pending = 0;
            }
            memset(&vec->rewards[agent_start], 0, apb * sizeof(float));
            memset(&vec->terminals[agent_start], 0, apb * sizeof(float));
            clock_gettime(CLOCK_MONOTONIC, &t0);
            #pragma omp parallel for schedule(static) num_threads(vec->num_workers)
            for (int i = env_start; i < env_start + env_count; i++) {
                puf_step(&envs[i]);
            }
            clock_gettime(CLOCK_MONOTONIC, &t1);
            my_accum[PROF_ENV] += (t1.tv_sec - t0.tv_sec) * 1000.0f
                + (t1.tv_nsec - t0.tv_nsec) / 1e6f;
            cudaEventRecord(ev[H2D_START], stream);
            cpu_upload(pufferl, agent_start, apb, stream);
            cudaEventRecord(ev[H2D_END], stream);
            h2d_pending = 1;
        }
        cudaStreamSynchronize(stream);
        if (h2d_pending) {
            cudaEventElapsedTime(&ms, ev[H2D_START], ev[H2D_END]);
            my_accum[PROF_COPY] += ms;
        }
        __atomic_store_n(state, BUF_WAITING, __ATOMIC_SEQ_CST);
    }
}

static void env_start(PuffeRL* p) {
    if (PUF_BACKEND == PUF_GPU) {
        puf_reset(p->vec->envs);
        cudaDeviceSynchronize();
        return;
    }
    VecEnv* vec = p->vec;
    vec->worker_state = (int*)calloc(1, vec->buffers * sizeof(int));
    vec->threads = (pthread_t*)calloc(1, vec->buffers * sizeof(pthread_t));
    VecThreadArg* args = (VecThreadArg*)calloc(1,
        vec->buffers * sizeof(VecThreadArg));
    vec->accum = (float*)calloc(1, vec->buffers * NUM_PROF * sizeof(float));
    #pragma omp parallel for schedule(static) num_threads(vec->num_workers)
    for (int i = 0; i < vec->size; i++) {
        puf_reset(&vec->envs[i]);
    }
    cpu_upload(p, 0, vec->total_agents, p->default_stream);
    cudaDeviceSynchronize();
    for (int i = 0; i < vec->buffers; i++) {
        args[i].pufferl = p;
        args[i].buf = i;
        pthread_create(&vec->threads[i], NULL, vec_thread_main, &args[i]);
    }
    for (int i = 0; i < vec->buffers; i++) {
        int* state = &vec->worker_state[i];
        while (__atomic_load_n(state, __ATOMIC_SEQ_CST) != BUF_WAITING) {}
    }
}

static void rollout_start(PuffeRL* p) {
    if (PUF_BACKEND == PUF_GPU) {
        cudaStream_t stream = p->streams[0];
        puf_bind_stream(stream);
        cudaEvent_t* ev = p->profile.rollout_ev;
        int H = p->hypers.horizon;
        for (int t = 0; t < H; t++) {
            int base = t * EV_T;
            cudaEventRecord(ev[base + MODEL_START], stream);
            pufferl_forward(p, 0, t, stream);
            cudaEventRecord(ev[base + MODEL_END], stream);
            puf_step(p->vec->envs);
            cudaEventRecord(ev[base + ENV_END], stream);
        }
        return;
    }
    for (int buf = 0; buf < p->vec->buffers; buf++) {
        int* state = &p->vec->worker_state[buf];
        __atomic_store_n(state, BUF_RUNNING, __ATOMIC_SEQ_CST);
    }
}

void rollout_finish(PuffeRL* p, double t0) {
    if (PUF_BACKEND == PUF_GPU) {
        cudaStreamSynchronize(p->streams[0]);
        float model_ms = 0.0f, env_ms = 0.0f, ms;
        cudaEvent_t* ev = p->profile.rollout_ev;
        int H = p->hypers.horizon;
        for (int t = 0; t < H; t++) {
            int base = t * EV_T;
            cudaEventElapsedTime(&ms, ev[base + MODEL_START], ev[base + MODEL_END]);
            model_ms += ms;
            cudaEventElapsedTime(&ms, ev[base + MODEL_END], ev[base + ENV_END]);
            env_ms += ms;
        }
        p->profile.accum[PROF_MODEL] += model_ms;
        p->profile.accum[PROF_ENV] += env_ms;
        p->profile.accum[PROF_ROLLOUT] += model_ms + env_ms;
        return;
    }
    for (int buf = 0; buf < p->vec->buffers; buf++) {
        int* state = &p->vec->worker_state[buf];
        while (__atomic_load_n(state, __ATOMIC_SEQ_CST) != BUF_WAITING) {}
    }
    float sec = (float)(wall_clock() - t0);
    p->profile.accum[PROF_ROLLOUT] += sec * 1000.0f;
    float eval_prof[NUM_PROF] = {0};
    for (int buf = 0; buf < p->vec->buffers; buf++) {
        float* src = &p->vec->accum[buf * NUM_PROF];
        for (int i = 0; i < NUM_PROF; i++) {
            eval_prof[i] += src[i];
        }
        memset(src, 0, NUM_PROF * sizeof(float));
    }
    p->profile.accum[PROF_MODEL] += eval_prof[PROF_MODEL] / p->vec->buffers;
    p->profile.accum[PROF_ENV] += eval_prof[PROF_ENV] / p->vec->buffers;
    p->profile.accum[PROF_COPY] += eval_prof[PROF_COPY] / p->vec->buffers;
}

static void env_close(VecEnv* vec) {
    if (PUF_BACKEND == PUF_GPU) {
        puf_close(vec->envs);
        cudaFree(vec->log_scratch);
        return;
    }
    __atomic_store_n(&vec->shutdown, 1, __ATOMIC_SEQ_CST);
    for (int i = 0; i < vec->buffers; i++) {
        pthread_join(vec->threads[i], NULL);
    }
    for (int i = 0; i < vec->size; i++) {
        puf_close(&vec->envs[i]);
    }
#ifdef MY_VEC_CLOSE
    my_vec_close(vec->envs);
#endif
}

void vec_log(VecEnv* vec, Dict* out, int clear) {
    Log aggregate = {0};
    float* acc = (float*)&aggregate;
    env_log_sum(vec, &aggregate, clear);

    float n = aggregate.n;
    Dict env_out = {0};
    if (n > 0.0f) {
        for (int j = 0; j < LOG_NF; j++) {
            acc[j] /= n;
        }
        puf_log(&aggregate, &env_out);
    }
    dict_set(&env_out, "n", n);
    for (int i = 0; i < env_out.size; i++) {
        char key[256];
        snprintf(key, sizeof(key), "env/%s", env_out.items[i].key);
        dict_set(out, key, env_out.items[i].value);
    }
    dict_clear(&env_out);
}

// Zero advantages on frozen-bank rows so prio_replay never samples them. Frozen
// rollout rows hold actions/logprobs from the frozen policy; training the
// primary's PPO on them produces garbage ratios and poisoned gradients.
__global__ void zero_frozen_advantages_kernel(precision_t* advantages,
        int agents_per_buf, int primary_per_buffer, int total_rows, int horizon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_rows * horizon;
    if (idx >= total) {
        return;
    }
    int row = idx / horizon;
    int rel = row % agents_per_buf;
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

// One block per minibatch segment: copy train fields for idx[mb].
// Row byte sizes are host-precomputed. initial_states gather when data is set.
#define SELECT_COPY_THREADS 256
__global__ void select_copy(RolloutBuf rollouts, TrainGraph graph,
        int* idx, precision_t* advantages, float* mb_prio,
        int obs_rb, int act_rb, int lp_rb, int term_rb, int mask_rb, int horizon) {
    int mb = blockIdx.x;
    int src_row = idx[mb];

    copy_bytes((const char*)rollouts.observations.data,
        (char*)graph.mb_obs.data, src_row, mb, obs_rb);
    copy_bytes((const char*)rollouts.actions.data,
        (char*)graph.mb_actions.data, src_row, mb, act_rb);
    copy_bytes((const char*)rollouts.logprobs.data,
        (char*)graph.mb_logprobs.data, src_row, mb, lp_rb);
    copy_bytes((const char*)rollouts.action_mask.data,
        (char*)graph.mb_action_mask.data, src_row, mb, mask_rb);

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
        d_returns[i] = val + adv;
    }

    if (threadIdx.x == 0) {
        graph.mb_prio.data[mb] = from_float(mb_prio[mb]);
    }
    copy_bytes((const char*)rollouts.terminals.data,
        (char*)graph.mb_terminals.data, src_row, mb, term_rb);

    Prec* init_state = &rollouts.initial_states;
    if (init_state->data != NULL) {
        int L = (int)init_state->shape[0];
        int H = (int)init_state->shape[2];
        int total = L * H;
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            int layer = i / H;
            int h = i % H;
            long src_idx = ((long)layer * init_state->shape[1] + src_row) * H + h;
            long dst_idx = ((long)layer * graph.mb_state.shape[1] + mb) * H + h;
            graph.mb_state.data[dst_idx] = init_state->data[src_idx];
        }
    }
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
__global__ void scatter_rows(precision_t* dst, int* idx,
        const precision_t* src, int num_idx, int row_elems) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_idx * row_elems;
    if (i >= total) {
        return;
    }
    int mb = i / row_elems;
    int e = i % row_elems;
    dst[(int64_t)idx[mb] * row_elems + e] = src[i];
}

// Cosine decay base → min over t in [0, T). Double for t/T (float loses
// precision past 2^24). Caller passes epoch and total train epochs.
float cosine_annealing(float base, float min_v, long t, long T) {
    double u = (double)t / (double)T;
    return min_v + 0.5f * (base - min_v) * (1.0f + (float)cos(M_PI * u));
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

void train_impl(PuffeRL* pufferl, RolloutBuf* src_arg) {
    Hypers* hypers = &pufferl->hypers;
    RolloutBuf src = src_arg ? *src_arg : pufferl->rollouts;
    cudaStream_t train_stream = pufferl->train_stream;
    cudaEventRecord(pufferl->profile.events[TE_S], train_stream);

    // Transpose from rollout layout (T, B, ...) to train layout (B, T, ...)
    RolloutBuf* rollouts = &pufferl->train_rollouts;
    Prec* advantages = &pufferl->advantages;

    // Obs/actions always (T, B, F) from register_rollout_buffers; scalars (T, B).
    int T = src.observations.shape[0];
    int B = src.observations.shape[1];
    int obs_size = (int)src.observations.shape[2];
    int num_atns = (int)src.actions.shape[2];

    transpose_102<<<grid_size(T * B * obs_size), BLOCK_SIZE, 0, train_stream>>>(
        rollouts->observations.data, src.observations.data, T, B, obs_size);
    transpose_102<<<grid_size(T * B * num_atns), BLOCK_SIZE, 0, train_stream>>>(
        rollouts->actions.data, src.actions.data, T, B, num_atns);
    transpose_102<<<grid_size(T * B), BLOCK_SIZE, 0, train_stream>>>(
        rollouts->logprobs.data, src.logprobs.data, T, B, 1);
    transpose_102<<<grid_size(T * B), BLOCK_SIZE, 0, train_stream>>>(
        rollouts->rewards.data, src.rewards.data, T, B, 1);
    transpose_102<<<grid_size(T * B), BLOCK_SIZE, 0, train_stream>>>(
        rollouts->terminals.data, src.terminals.data, T, B, 1);
    transpose_102<<<grid_size(T * B), BLOCK_SIZE, 0, train_stream>>>(
        rollouts->ratio.data, src.ratio.data, T, B, 1);
    transpose_102<<<grid_size(T * B), BLOCK_SIZE, 0, train_stream>>>(
        rollouts->values.data, src.values.data, T, B, 1);
    int mask_c = src.action_mask.shape[2];
    transpose_102<<<grid_size(T * B * mask_c), BLOCK_SIZE, 0, train_stream>>>(
        rollouts->action_mask.data, src.action_mask.data, T, B, mask_c);

    // We hard-clamp rewards to -1, 1. This CAN introduce reward hacks
    clamp_precision_kernel<<<grid_size(
        numel(rollouts->rewards.shape)), BLOCK_SIZE, 0, train_stream>>>(
        rollouts->rewards.data, -1.0f, 1.0f, numel(rollouts->rewards.shape));

    // Treat rollout data as on-policy for advantage. PPO clipping still uses
    // behavior logprobs captured by the actor snapshot.
    fill_precision_kernel<<<grid_size(
        numel(rollouts->ratio.shape)), BLOCK_SIZE, 0, train_stream>>>(
        rollouts->ratio.data, from_float(1.0f), numel(rollouts->ratio.shape));

    int batch_size = hypers->total_agents * hypers->horizon;
    float prio_beta0 = hypers->prio_beta0;
    float prio_alpha = hypers->prio_alpha;
    bool anneal_lr = hypers->anneal_lr;
    int current_epoch = pufferl->epoch;
    Muon* muon = &pufferl->muon;

    // Schedule over this rank's train epochs (same as outer loop), not global
    // total_timesteps/batch — multi-GPU would otherwise only traverse 1/W of the
    // cosine and never reach min_lr / min_ent / full prio beta anneal.
    int total_epochs = hypers->total_timesteps / hypers->world_size / batch_size;
    if (anneal_lr) {
        float lr_min = hypers->min_lr_ratio * hypers->lr;
        float lr = cosine_annealing(hypers->lr, lr_min, current_epoch, total_epochs);
        cudaMemcpy(muon->lr, &lr, sizeof(float), cudaMemcpyHostToDevice);
    }

    // Annealed entropy coefficient — same cosine shape as lr. With PG signal
    // alive, the entropy bonus that kept early-training exploratory becomes
    // load-bearing dead weight late in training; cosine-decay frees the policy
    // to commit harder on what it has already learned.
    float current_ent_coef = hypers->ent_coef;
    if (hypers->anneal_ent_coef) {
        float ent_min = hypers->min_ent_coef_ratio * hypers->ent_coef;
        current_ent_coef = cosine_annealing(
            hypers->ent_coef, ent_min, current_epoch, total_epochs);
    }
    // Device ptr for ent_coef so CUDA graphs do not bake host by-value.
    cudaMemcpyAsync(pufferl->ppo_bufs.ent_coef, &current_ent_coef,
        sizeof(float), cudaMemcpyHostToDevice, train_stream);

    // Annealed priority exponent
    float anneal_beta = prio_beta0 + (1.0f - prio_beta0) * prio_alpha
        * (float)current_epoch / (float)total_epochs;
    TrainGraph* graph = &pufferl->train_buf;
    cudaEventRecord(pufferl->profile.events[TE_E], train_stream);

    // Advantage + prio CDF once per train; minibatches only re-sample indices.
    // 64 threads/block: better occupancy than 256 for this heavy-per-thread scan
    // (bench_puff_advantage launch sweep; breakout 4096×32 ~2× faster).
    constexpr int ADV_THREADS = 64;
    int rows = (int)rollouts->values.shape[0];
    int horizon = (int)rollouts->values.shape[1];
    int grid = (rows + ADV_THREADS - 1) / ADV_THREADS;
    cudaMemsetAsync(advantages->data, 0,
        numel(advantages->shape) * sizeof(precision_t), train_stream);
    profile_begin("compute_advantage", hypers->profile);
    puff_advantage<<<grid, ADV_THREADS, 0, train_stream>>>(
        rollouts->values.data, rollouts->rewards.data, rollouts->terminals.data,
        rollouts->ratio.data, advantages->data,
        hypers->gamma, hypers->gae_lambda,
        hypers->vtrace_rho_clip, hypers->vtrace_c_clip, rows, horizon);
    if (pufferl->num_frozen_banks > 0) {
        int apb = hypers->total_agents / hypers->num_buffers;
        int total = rows * horizon;
        zero_frozen_advantages_kernel<<<
            grid_size(total), BLOCK_SIZE, 0, train_stream>>>(
            advantages->data, apb, pufferl->vec->bank_layout[1], rows, horizon);
    }
    profile_end(hypers->profile);

    profile_begin("compute_prio", hypers->profile);
    prio_build_cdf_cuda(*advantages, prio_alpha, pufferl->prio_bufs, train_stream);
    profile_end(hypers->profile);

    long* train_rng_offset = pufferl->rng_offset + hypers->num_buffers;
    int total_minibatches = hypers->replay_ratio * batch_size / hypers->minibatch_size;
    // Row strides for select_copy: fixed train-buffer layout (agent-major).
    int pe = (int)sizeof(precision_t);
    int obs_rb = (numel(rollouts->observations.shape)
        / rollouts->observations.shape[0]) * pe;
    int act_rb = (numel(rollouts->actions.shape)
        / rollouts->actions.shape[0]) * pe;
    int lp_rb = (numel(rollouts->logprobs.shape)
        / rollouts->logprobs.shape[0]) * pe;
    int term_rb = (numel(rollouts->terminals.shape)
        / rollouts->terminals.shape[0]) * pe;
    int mask_rb = (numel(rollouts->action_mask.shape)
        / rollouts->action_mask.shape[0]) * pe;
    int sel_horizon = (int)rollouts->values.shape[1];
    int mb_segs = (int)pufferl->prio_bufs.idx.shape[0];
    int* sel_idx = pufferl->prio_bufs.idx.data;
    for (int mb = 0; mb < total_minibatches; ++mb) {
        cudaEventRecord(pufferl->profile.events[TE_MS], train_stream);

        profile_begin("compute_prio", hypers->profile);
        prio_sample_cuda(anneal_beta, pufferl->prio_bufs, pufferl->seed,
            train_rng_offset, train_stream);
        profile_end(hypers->profile);

        profile_begin("train_select_and_copy", hypers->profile);
        // reset_every_horizon: train from zeros, no carry buffer. Else: gather from
        // initial_states slot (always allocated when !reset_every_horizon).
        RolloutBuf sel_src = *rollouts;
        if (hypers->reset_every_horizon) {
            cudaMemsetAsync(graph->mb_state.data, 0,
                numel(graph->mb_state.shape) * sizeof(precision_t), train_stream);
            sel_src.initial_states = Prec();
        } else {
            int slot = hypers->async ? pufferl->async_ready_slot : 0;
            sel_src.initial_states = init_slot(src.initial_states, slot);
        }
        select_copy<<<mb_segs, SELECT_COPY_THREADS, 0, train_stream>>>(
            sel_src, *graph, sel_idx, advantages->data,
            pufferl->prio_bufs.mb_prio.data,
            obs_rb, act_rb, lp_rb, term_rb, mask_rb, sel_horizon);
        profile_end(hypers->profile);

        cudaEventRecord(pufferl->profile.events[TE_ME], train_stream);
        profile_begin("train_forward_backward", hypers->profile);
        if (hypers->cudagraphs && pufferl->train_cudagraph != NULL) {
            cudaGraphLaunch(pufferl->train_cudagraph, train_stream);
        } else {
            if (hypers->cudagraphs) {
                assert(cudaStreamBeginCapture(
                    train_stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess
                    && "cudaStreamBeginCapture failed");
            }

            cudaStream_t stream = train_stream;
            Prec obs = graph->mb_obs;
            Prec state = graph->mb_state;
            Prec terminals = graph->mb_terminals;
            Prec dec = policy_forward_train(&pufferl->policy, pufferl->weights,
                pufferl->train_activs, obs, state, terminals, stream);
            DecoderWeights* dw_train = (DecoderWeights*)pufferl->weights.decoder;
            Prec p_logstd;
            if (dw_train->continuous) {
                p_logstd = dw_train->logstd;
            }

            ppo_loss_fwd_bwd(dec, p_logstd, *graph,
                pufferl->act_sizes, pufferl->losses,
                hypers->clip_coef, hypers->vf_clip_coef, hypers->vf_coef,
                pufferl->ppo_bufs.ent_coef,
                pufferl->ppo_bufs, pufferl->is_continuous, stream);

            Float grad_logits = pufferl->ppo_bufs.grad_logits;
            Float grad_logstd = pufferl->is_continuous ? pufferl->ppo_bufs.grad_logstd : Float();
            Float grad_values = pufferl->ppo_bufs.grad_values;
            policy_backward(&pufferl->policy, pufferl->weights, pufferl->train_activs,
                grad_logits, grad_logstd, grad_values, stream);

            if (pufferl->nccl_comm != NULL && hypers->world_size > 1) {
                ncclAllReduce(pufferl->grad.data, pufferl->grad.data,
                    numel(pufferl->grad.shape), NCCL_PRECISION, ncclAvg,
                    pufferl->nccl_comm, stream);
            }
            muon_step(muon, pufferl->master_weights,
                pufferl->grad, hypers->max_grad_norm, stream);
            if (USE_BF16) {
                int n = numel(pufferl->param.shape);
                cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
                    pufferl->param.data, pufferl->master_weights.data, n);
            }
            if (hypers->cudagraphs) {
                cudaGraph_t _graph;
                assert(cudaStreamEndCapture(train_stream, &_graph) == cudaSuccess
                    && "cudaStreamEndCapture failed");
                assert(cudaGraphInstantiate(&pufferl->train_cudagraph, _graph, 0)
                    == cudaSuccess && "cudaGraphInstantiate failed");
                cudaGraphDestroy(_graph);
                // Capture records without executing; run once so this step has effects.
                assert(cudaGraphLaunch(
                    pufferl->train_cudagraph, train_stream) == cudaSuccess
                    && "cudaGraphLaunch failed");
            }
        }
        profile_end(hypers->profile);

        // This version is consistent with PufferLib 3.0. One of the major algorithmic
        // questions remaining is how and when to update value and advantage estimates.
        int num_idx = numel(pufferl->prio_bufs.idx.shape);
        int ratio_row = numel(graph->mb_ratio.shape) / graph->mb_ratio.shape[0];
        int value_row = graph->mb_newvalue.shape[1];
        scatter_rows<<<grid_size(num_idx * ratio_row),
            BLOCK_SIZE, 0, train_stream>>>(
            rollouts->ratio.data, pufferl->prio_bufs.idx.data,
            graph->mb_ratio.data, num_idx, ratio_row);
        scatter_rows<<<grid_size(num_idx * value_row),
            BLOCK_SIZE, 0, train_stream>>>(
            rollouts->values.data, pufferl->prio_bufs.idx.data,
            graph->mb_newvalue.data, num_idx, value_row);
        cudaEventRecord(pufferl->profile.events[TE_FE], train_stream);
    }

    cudaStreamSynchronize(train_stream);

    if (total_minibatches > 0) {
        float ms;
        cudaEvent_t* ev = pufferl->profile.events;
        float* accum = pufferl->profile.accum;
        cudaEventElapsedTime(&ms, ev[TE_S], ev[TE_E]);
        accum[PROF_TRAIN_MISC] += ms;
        cudaEventElapsedTime(&ms, ev[TE_MS], ev[TE_ME]);
        accum[PROF_TRAIN_MISC] += ms * total_minibatches;
        cudaEventElapsedTime(&ms, ev[TE_ME], ev[TE_FE]);
        accum[PROF_TRAIN_MODEL] += ms * total_minibatches;
    }
    pufferl->epoch += 1;
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

void puf_load_weights_into(Float dst, Prec params,
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
    WeightBank* bank = &pufferl->frozen_banks[bank_idx];
    puf_load_weights_into(bank->master_weights, bank->param,
        pufferl->default_stream, path);
    cudaDeviceSynchronize();
}

// fp32 master weights: alias param buffer in float mode; separate fp32 copy in bf16.
// cast_now: copy param→master now (primary after init). Frozen banks load later.
static void master_weights_setup(Float* mw, Prec* param,
        bool cast_now, cudaStream_t stream) {
    long n = numel(param->shape);
    if (USE_BF16) {
        *mw = (Float){.shape = {n}};
        cudaMalloc((void**)&mw->data, n * sizeof(float));
        if (cast_now) {
            cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(mw->data, param->data, n);
        }
    } else {
        *mw = (Float){.data = (float*)param->data, .shape = {n}};
    }
}

PuffeRL* create_pufferl(Ini* ini, TrainContext* ctx) {
    Hypers hypers = {
        .horizon = puf_ini_get_int(ini, "train", "horizon"),
        .total_agents = puf_ini_get_int(ini, "vec", "total_agents"),
        .num_buffers = puf_ini_get_int(ini, "vec", "num_buffers"),
        .hidden_size = puf_ini_get_int(ini, "policy", "hidden_size"),
        .num_layers = puf_ini_get_int(ini, "policy", "num_layers"),
        .lr = puf_ini_get_float(ini, "train", "learning_rate"),
        .min_lr_ratio = puf_ini_get_float(ini, "train", "min_lr_ratio"),
        .anneal_lr = puf_ini_get_int(ini, "train", "anneal_lr") != 0,
        .momentum = puf_ini_get_float(ini, "train", "momentum"),
        .minibatch_size = puf_ini_get_int(ini, "train", "minibatch_size"),
        .replay_ratio = puf_ini_get_float(ini, "train", "replay_ratio"),
        .total_timesteps = puf_ini_get_long(ini, "train", "total_timesteps"),
        .max_grad_norm = puf_ini_get_float(ini, "train", "max_grad_norm"),
        .clip_coef = puf_ini_get_float(ini, "train", "clip_coef"),
        .vf_clip_coef = puf_ini_get_float(ini, "train", "vf_clip_coef"),
        .vf_coef = puf_ini_get_float(ini, "train", "vf_coef"),
        .ent_coef = puf_ini_get_float(ini, "train", "ent_coef"),
        .min_ent_coef_ratio = puf_ini_get_float(ini, "train", "min_ent_coef_ratio"),
        .anneal_ent_coef = puf_ini_get_int(ini, "train", "anneal_ent_coef") != 0,
        .gamma = puf_ini_get_float(ini, "train", "gamma"),
        .gae_lambda = puf_ini_get_float(ini, "train", "gae_lambda"),
        .vtrace_rho_clip = puf_ini_get_float(ini, "train", "vtrace_rho_clip"),
        .vtrace_c_clip = puf_ini_get_float(ini, "train", "vtrace_c_clip"),
        .prio_alpha = puf_ini_get_float(ini, "train", "prio_alpha"),
        .prio_beta0 = puf_ini_get_float(ini, "train", "prio_beta0"),
        .async = puf_ini_get_int(ini, "base", "async") != 0,
        .reset_every_horizon = puf_ini_get_int(ini, "base", "reset_every_horizon") != 0,
        .cudagraphs = puf_ini_get(ini, "base", "cudagraphs") >= 0,
        .profile = puf_ini_get_int(ini, "base", "profile") != 0,
        .rank = ctx->rank,
        .world_size = ctx->world_size,
        .gpu_id = ctx->gpu_id,
        .num_threads = puf_ini_get_int(ini, "vec", "num_threads"),
        .seed = puf_ini_get_int(ini, "base", "seed"),
    };

    Dict vec_kwargs = {0};
    dict_copy(&vec_kwargs, puf_ini_section(ini, "vec", 0));
    Dict* env_kwargs = puf_ini_section(ini, "env", 0);
    ncclUniqueId* nccl_id = ctx->nccl_id;

    PuffeRL* pufferl = (PuffeRL*)calloc(1, sizeof(PuffeRL));
    pufferl->hypers = hypers;
    snprintf(pufferl->env_name, sizeof(pufferl->env_name), "%s", PUFFER_ENV_NAME);

    cudaSetDevice(hypers.gpu_id);
    cublas_init_handle();

    if (hypers.world_size > 1) {
        ncclCommInitRank(&pufferl->nccl_comm, hypers.world_size, *nccl_id, hypers.rank);
        printf("Rank %d/%d: NCCL initialized\n", hypers.rank, hypers.world_size);
    }

    pufferl->seed = (ulong)hypers.seed + hypers.rank;

    // GPU tensors allocated into pufferl->env; vec aliases them.
    int total_agents = (int)dict_get(&vec_kwargs, "total_agents");
    int num_buffers = (int)dict_get(&vec_kwargs, "num_buffers");
    VecEnv* vec = (VecEnv*)calloc(1, sizeof(VecEnv));
    vec->total_agents = total_agents;
    vec->buffers = num_buffers;
    vec->agents_per_buf = total_agents / num_buffers;
    int frozen_banks = (int)dict_get(&vec_kwargs, "num_frozen_banks");

    // Discrete action layout. Continuous dims are size 1. Mask width is act_n.
    int num_action_heads = NUM_ATNS;
    int act_sizes[] = ACT_SIZES;
    int act_n = 0;
    int n_cont = 0;
    int n_disc = 0;
    for (int i = 0; i < num_action_heads; i++) {
        if (act_sizes[i] == 1) {
            n_cont++;
        } else {
            n_disc++;
        }
        act_n += act_sizes[i];
    }
    assert(!(n_cont > 0 && n_disc > 0)
        && "mixed continuous/discrete action spaces not supported");
    bool is_continuous = n_cont > 0;
    pufferl->is_continuous = is_continuous;
    vec->num_banks = frozen_banks + 1;
    vec->bank_layout = (int*)calloc(1, (vec->num_banks + 1) * sizeof(int));
    vec->mask_size = act_n;

    // Device env IO (EnvBuf).
    pufferl->env = {
        .obs =         {.shape = {total_agents, OBS_SIZE}},
        .actions =     {.shape = {total_agents, NUM_ATNS}},
        .rewards =     {.shape = {total_agents}},
        .terminals =   {.shape = {total_agents}},
        .action_mask = {.shape = {total_agents, act_n}},
    };
    EnvBuf* env = &pufferl->env;
    size_t mask_bytes = total_agents * act_n * sizeof(unsigned char);
    cudaMalloc((void**)&env->obs.data, total_agents * OBS_SIZE * sizeof(obs_t));
    cudaMalloc((void**)&env->actions.data, total_agents * NUM_ATNS * sizeof(float));
    cudaMalloc((void**)&env->rewards.data, total_agents * sizeof(float));
    cudaMalloc((void**)&env->terminals.data, total_agents * sizeof(float));
    cudaMalloc((void**)&env->action_mask.data, mask_bytes);
    cudaMemset(env->obs.data, 0, total_agents * OBS_SIZE * sizeof(obs_t));
    cudaMemset(env->actions.data, 0, total_agents * NUM_ATNS * sizeof(float));
    cudaMemset(env->rewards.data, 0, total_agents * sizeof(float));
    cudaMemset(env->terminals.data, 0, total_agents * sizeof(float));
    cudaMemset(env->action_mask.data, 1, mask_bytes);

    env_setup(pufferl, vec, &vec_kwargs, env_kwargs, frozen_banks);
    pufferl->vec = vec;

    for (int i = 0; i < NUM_TE; i++) {
        assert(cudaEventCreate(&pufferl->profile.events[i]) == cudaSuccess);
    }
    if (PUF_BACKEND == PUF_GPU) {
        int H = hypers.horizon;
        pufferl->profile.rollout_ev = (cudaEvent_t*)calloc(
            1, EV_T * H * sizeof(cudaEvent_t));
        for (int i = 0; i < EV_T * H; i++) {
            assert(cudaEventCreate(&pufferl->profile.rollout_ev[i]) == cudaSuccess);
        }
    }
    nvmlInit();
    nvmlDeviceGetHandleByIndex(hypers.gpu_id, &pufferl->nvml_device);

    int input_size = OBS_SIZE;
    int hidden_size = hypers.hidden_size;
    int num_layers = hypers.num_layers;
    int decoder_output_size = is_continuous ? num_action_heads : act_n;
    int minibatch_segments = hypers.minibatch_size / hypers.horizon;
    int inf_batch = total_agents / num_buffers;
    int B_TT = minibatch_segments * hypers.horizon;
    int horizon = hypers.horizon;
    int batch = total_agents / num_buffers;

    pufferl->policy = build_policy(pufferl->env_name, input_size, hidden_size,
        num_layers, decoder_output_size, is_continuous, hypers.horizon);

    // Dedicated learner stream (always non-default; nonblocking when async).
    if (hypers.async) {
        assert(cudaStreamCreateWithFlags(
            &pufferl->train_stream, cudaStreamNonBlocking) == cudaSuccess);
    } else {
        assert(cudaStreamCreate(&pufferl->train_stream) == cudaSuccess);
    }

    // Create and allocate params
    Allocator* params = &pufferl->params_alloc;
    Allocator* acts = &pufferl->activ_alloc;
    Allocator* grads = &pufferl->grads_alloc;

    // Buffers for weights, grads, and activations
    pufferl->weights = policy_weights_create(&pufferl->policy, params);
    if (hypers.async) {
        pufferl->actor_weights = policy_weights_create(
            &pufferl->policy, &pufferl->weight_alloc);
    }
    pufferl->train_activs = policy_reg_train(
        &pufferl->policy, pufferl->weights, acts, grads, B_TT);
    pufferl->buf_acts = (PolicyActivations*)calloc(
        1, num_buffers * sizeof(PolicyActivations));
    pufferl->buffer_states = (Prec*)calloc(1, num_buffers * sizeof(Prec));
    for (int i = 0; i < num_buffers; i++) {
        pufferl->buf_acts[i] = policy_reg_rollout(
            &pufferl->policy, pufferl->weights, acts, inf_batch);
        pufferl->buffer_states[i] = {
            .shape = {num_layers, batch, hidden_size},
        };
        alloc_register(acts, &pufferl->buffer_states[i]);
    }
    int rollout_horizon = hypers.async ? 2 * horizon : horizon;
    register_rollout_buffers(&pufferl->rollouts,
        acts, rollout_horizon, total_agents, input_size, num_action_heads, act_n);
    // Carry path: per-slot initial RNN states. reset_every_horizon zeros mb_state instead.
    if (!hypers.reset_every_horizon) {
        int slots = hypers.async ? 2 : 1;
        pufferl->rollouts.initial_states = {
            .shape = {slots, num_layers, total_agents, hidden_size}};
        alloc_register(acts, &pufferl->rollouts.initial_states);
    }
    register_train_buffers(pufferl->train_buf,
        acts, minibatch_segments, horizon, input_size,
        hidden_size, num_action_heads, num_layers, act_n);
    register_rollout_buffers(&pufferl->train_rollouts,
        acts, total_agents, horizon, input_size, num_action_heads, act_n);
    register_ppo_buffers(pufferl->ppo_bufs, acts, minibatch_segments,
        hypers.horizon, decoder_output_size, is_continuous);
    register_prio_buffers(pufferl->prio_bufs,
        acts, hypers.total_agents, minibatch_segments);

    pufferl->advantages = {.shape = {total_agents, horizon}};
    alloc_register(acts, &pufferl->advantages);

    cudaMalloc((void**)&pufferl->rng_offset, (num_buffers + 1) * sizeof(long));
    cudaMemset(pufferl->rng_offset, 0, (num_buffers + 1) * sizeof(long));
    cudaMalloc((void**)&pufferl->act_sizes, num_action_heads * sizeof(int));
    cudaMalloc((void**)&pufferl->losses, NUM_LOSSES * sizeof(float));

    muon_init(&pufferl->muon, params, hypers.momentum, acts);

    // All buffers allocated here
    alloc_create(params);
    if (hypers.async) {
        alloc_create(&pufferl->weight_alloc);
    }
    alloc_create(grads);
    alloc_create(acts);

    pufferl->grad = {.data = (precision_t*)grads->mem, .shape = {grads->total_elems}};
    pufferl->param = {.data = (precision_t*)params->mem, .shape = {params->total_elems}};
    if (hypers.async) {
        pufferl->actor_param = {
            .data = (precision_t*)pufferl->weight_alloc.mem,
            .shape = {pufferl->weight_alloc.total_elems},
        };
    }

    ulong init_seed = hypers.seed;
    policy_init_weights(&pufferl->policy,
        pufferl->weights, &init_seed, pufferl->default_stream);
    master_weights_setup(&pufferl->master_weights,
        &pufferl->param, true, pufferl->default_stream);
    if (hypers.async) {
        puf_copy(&pufferl->actor_param, &pufferl->param, pufferl->default_stream);
        cudaStreamSynchronize(pufferl->default_stream);
    }

    // Per-buffer persistent RNG states
    int agents_per_buf = total_agents / num_buffers;
    pufferl->rng_states = (curandStatePhilox4_32_10_t**)calloc(
        1, num_buffers * sizeof(curandStatePhilox4_32_10_t*));
    for (int i = 0; i < num_buffers; i++) {
        cudaMalloc((void**)&pufferl->rng_states[i],
            agents_per_buf * sizeof(curandStatePhilox4_32_10_t));
        cudaMemset(pufferl->rng_states[i], 0,
            agents_per_buf * sizeof(curandStatePhilox4_32_10_t));
        rng_init<<<grid_size(agents_per_buf), BLOCK_SIZE>>>(
            pufferl->rng_states[i], pufferl->seed + i, agents_per_buf);
    }

    // Post-create initialization
    cudaMemcpy(pufferl->act_sizes, act_sizes,
        num_action_heads*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(pufferl->losses, 0, NUM_LOSSES * sizeof(float));
    cudaMemcpy(pufferl->muon.lr, &hypers.lr, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(pufferl->muon.mb.data, 0, numel(pufferl->muon.mb.shape) * sizeof(float));

    // Frozen banks (selfplay/match opponents): rollout-only policy.
    // Arch comes from vec.frozen_bank_*; required when banks > 0.
    int num_frozen = vec->num_banks - 1;
    int frozen_hidden = (int)dict_get(&vec_kwargs, "frozen_bank_hidden_size");
    int frozen_layers = (int)dict_get(&vec_kwargs, "frozen_bank_num_layers");
    assert(!(num_frozen > 0 && (frozen_hidden <= 0 || frozen_layers <= 0))
        && "num_frozen_banks requires frozen_bank_hidden_size and frozen_bank_num_layers > 0");
    pufferl->num_frozen_banks = num_frozen;
    if (num_frozen > 0) {
        pufferl->frozen_banks = (WeightBank*)calloc(1,
            num_frozen * sizeof(WeightBank));
    }
    for (int b = 0; b < num_frozen; b++) {
        int slice = vec->bank_layout[b + 2] - vec->bank_layout[b + 1];
        assert(slice > 0 && "frozen bank has no agents");
        WeightBank* bank = &pufferl->frozen_banks[b];
        bank->policy = build_policy(pufferl->env_name, input_size, frozen_hidden,
            frozen_layers, decoder_output_size, is_continuous, hypers.horizon);
        bank->weights = policy_weights_create(&bank->policy, &bank->params_alloc);
        bank->buf_acts = (PolicyActivations*)calloc(1,
            num_buffers * sizeof(PolicyActivations));
        bank->buffer_states = (Prec*)calloc(1, num_buffers * sizeof(Prec));
        for (int i = 0; i < num_buffers; i++) {
            bank->buf_acts[i] = policy_reg_rollout(
                &bank->policy, bank->weights, &bank->activ_alloc, slice);
            bank->buffer_states[i] = {.shape = {frozen_layers, slice, frozen_hidden}};
            alloc_register(&bank->activ_alloc, &bank->buffer_states[i]);
        }
        alloc_create(&bank->params_alloc);
        alloc_create(&bank->activ_alloc);
        bank->param = {
            .data = (precision_t*)bank->params_alloc.mem,
            .shape = {bank->params_alloc.total_elems},
        };
        // Weights loaded later via pufferl_load_frozen_bank; no cast yet.
        master_weights_setup(&bank->master_weights, &bank->param, false,
            pufferl->default_stream);
    }

    // Upload head-consume map before any stream capture (see init_head_consume_map).
    init_head_consume_map();

    // CUDA graphs: allocate graph array only; capture on first real use.
    if (hypers.cudagraphs) {
        int rollout_graph_slots = hypers.async ? 2 : 1;
        pufferl->rollout_graphs = (cudaGraphExec_t*)calloc(1,
            rollout_graph_slots*horizon*num_buffers*sizeof(cudaGraphExec_t));
    }
    pufferl->streams = (cudaStream_t*)calloc(1, num_buffers * sizeof(cudaStream_t));
    for (int i = 0; i < num_buffers; i++) {
        if (hypers.async) {
            assert(cudaStreamCreateWithFlags(
                &pufferl->streams[i], cudaStreamNonBlocking) == cudaSuccess);
        } else {
            assert(cudaStreamCreate(&pufferl->streams[i]) == cudaSuccess);
        }
    }

    env_start(pufferl);

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

// OS reclaims memory/CUDA context on exit. This is the intended design.
// All memory is allocated up front and static across training.
void close_pufferl(PuffeRL* p) {
    cudaDeviceSynchronize();
    if (p->hypers.profile) {
        cudaProfilerStop();
    }
    nvmlShutdown();
    env_close(p->vec);
    cudaDeviceSynchronize();
    if (p->nccl_comm != NULL) {
        ncclCommDestroy(p->nccl_comm);
    }
}

// Dashboard
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
    if (puf_dashboard_tty) {
        printf("\033[K");
    }
    putchar('\n');
}

static void dash_end(void) {
    if (puf_dashboard_tty) {
        printf("\033[J\033[?2026l");
    }
    fflush(stdout);
}

// Right-align into w cols (truncate if long). Gray unit letters
static void dash_cell(const char* s, int w) {
    int n = (int)strlen(s);
    if (n > w) {
        n = w;
    }
    int numeric = 0;
    for (int i = 0; i < n; i++) {
        numeric |= (s[i] >= '0' && s[i] <= '9');
    }
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

static void dash_blank(void) {
    printf("%s│%*s│%s", PUF_W, PUF_DASH_W - 2, "", PUF_R);
    dash_eol();
}

static void dash_abbrev(char* out, size_t n, double val) {
    const char* suf[] = {"", "K", "M", "B", "T"};
    int i = 0;
    while (val >= 1000.0 && i < 4) {
        val /= 1000.0;
        i++;
    }
    snprintf(out, n, "%.1f%s", val, suf[i]);
}

// <1s = ms; sub-hour keeps ms; longer → d/h/m/s.
static void dash_duration(char* out, size_t n, double sec) {
    if (sec < 0) {
        sec = 0;
    }
    long ms = (long)(sec * 1000.0 + 0.5);
    if (ms < 1000) {
        snprintf(out, n, "%ldms", ms);
    } else if (ms < 60000) {
        snprintf(out, n, "%lds %03ldms", ms / 1000, ms % 1000);
    } else if (ms < 3600000) {
        snprintf(out, n, "%ldm %02lds %03ldms",
            ms / 60000, (ms / 1000) % 60, ms % 1000);
    } else {
        long s = ms / 1000;
        snprintf(out, n, "%ldd %ldh %ldm %lds",
            s / 86400, (s / 3600) % 24, (s / 60) % 60, s % 60);
    }
}

// Missing keys keep last train values when merged into last_log.
static double dash_num(Dict* log, const char* key, double fallback) {
    DictItem* it = dict_find(log, key);
    return it ? it->value : fallback;
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
    double steps = dash_num(log, "agent_steps",
        (double)p->global_step * p->hypers.world_size);
    double sps = dash_num(log, "SPS", 0);
    long configured = (long)puf_ini_get(ini, "train", "total_timesteps");
    long local_batch = (long)p->hypers.total_agents * p->hypers.horizon;
    long local_steps = configured / p->hypers.world_size;
    double target = (double)((local_steps / local_batch) * local_batch * p->hypers.world_size);
    double remain_sec = sps > 0 && target > steps ? (target - steps) / sps : 0;
    double rollout = dash_num(log, "perf/rollout", 0);
    double train_time = dash_num(log, "perf/train", 0);
    double perf_total = rollout + train_time;

    char params[32], steps_s[32], sps_s[32], uptime[64], remaining[64], epoch_s[32];
    dash_abbrev(params, sizeof(params), (double)numel(p->master_weights.shape));
    dash_abbrev(steps_s, sizeof(steps_s), steps);
    dash_abbrev(sps_s, sizeof(sps_s), sps);
    double up = dash_num(log, "uptime", wall_clock() - p->start_time);
    dash_duration(uptime, sizeof(uptime), up);
    dash_duration(remaining, sizeof(remaining), remain_sec);
    snprintf(epoch_s, sizeof(epoch_s), "%d", epoch);

    if (puf_dashboard_tty) {
        printf("\033[?2026h");
        if (term_rows != puf_dashboard_last_rows || term_cols != puf_dashboard_last_cols) {
            printf("\033[H\033[J");
        } else {
            printf("\033[H");
        }
        puf_dashboard_last_rows = term_rows;
        puf_dashboard_last_cols = term_cols;
    }
    // Tiny terminal: one-line compact summary.
    if (puf_dashboard_tty && (term_cols < PUF_DASH_W || term_rows <= PUF_DASH_BASE_ROWS)) {
        char compact[512];
        snprintf(compact, sizeof(compact),
            "PufferLib 5.0  env=%s  steps=%s  SPS=%s  score=%.3f  epoch=%s  to_go=%s",
            env_name, steps_s, sps_s, dash_num(log, "env/score", 0), epoch_s, remaining);
        printf("%.*s", term_cols > 1 ? term_cols - 1 : term_cols, compact);
        if (term_rows > 1) {
            dash_eol();
        } else if (puf_dashboard_tty) {
            printf("\033[K");
        }
        dash_end();
        return;
    }

    char gpu[16], vram[32], ram[16];
    snprintf(gpu, sizeof(gpu), "%3.0f%%", dict_get(log, "util/gpu_percent"));
    snprintf(vram, sizeof(vram), "%.1f/%.0fG",
        dict_get(log, "util/vram_used_gb"),
        dict_get(log, "util/vram_total_gb"));
    snprintf(ram, sizeof(ram), "%.1fG", dict_get(log, "util/cpu_mem_gb"));

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
    dash_blank();

    // Stats | Perf | Losses — fixed widths sum to 80 with borders/spacing.
    struct {
        const char* a;
        const char* av;
        const char* b;
        const char* perf_key;  // null → use perf_sec
        double perf_sec;
        const char* c;
        const char* loss_key;
    } rows[] = {
        {"Env", env_name, "Evaluate", NULL, rollout, "Losses", "loss/total"},
        {"Params", params, "  Model", "perf/eval_model", 0, "policy", "loss/policy"},
        {"Steps", steps_s, "  Env", "perf/eval_env", 0, "value", "loss/value"},
        {"SPS", sps_s, "  Copy", "perf/eval_copy", 0, "entropy", "loss/entropy"},
        {"Epoch", epoch_s, "Train", NULL, train_time, "old_kl", "loss/old_kl"},
        {"Uptime", uptime, "  Model", "perf/train_model", 0, "kl", "loss/kl"},
        {"To go", remaining, "  Misc", "perf/train_misc", 0, "clipfrac", "loss/clipfrac"},
    };
    for (int i = 0; i < (int)(sizeof(rows) / sizeof(rows[0])); i++) {
        char bt[64], bp[16], cv[32];
        double sec = rows[i].perf_key
            ? dash_num(log, rows[i].perf_key, 0) : rows[i].perf_sec;
        dash_duration(bt, sizeof(bt), sec);
        snprintf(bp, sizeof(bp), "%d%%",
            perf_total > 0 ? (int)(100.0 * sec / perf_total) : 0);
        DictItem* lit = dict_find(log, rows[i].loss_key);
        if (lit) {
            snprintf(cv, sizeof(cv), "%.3f", lit->value);
        } else {
            cv[0] = 0;
        }
        printf("%s│%s %s%-9.9s%s ", PUF_W, PUF_R, PUF_A, rows[i].a, PUF_R);
        dash_cell(rows[i].av, 13);
        printf("    %s%-12.12s%s ", PUF_A, rows[i].b, PUF_R);
        dash_cell(bt, 6);
        putchar(' ');
        dash_cell(bp, 4);
        printf("    %s%-10.10s%s ", PUF_A, rows[i].c, PUF_R);
        dash_cell(cv, 7);
        printf("    %s│%s", PUF_W, PUF_R);
        dash_eol();
    }
    dash_blank();

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
            PUF_W, PUF_R, PUF_A, pending_key,
            PUF_R, PUF_W, ls, PUF_R, "", "", PUF_W, PUF_R);
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

// Eval only refreshes util + env. Do not write SPS/uptime/perf/losses: merging
// into last_log would overwrite the last train snapshot and zero the dashboard.
void trainer_eval_log(PuffeRL* p, Dict* out) {
    double now = wall_clock();
    p->last_log_time = now;
    p->last_log_step = p->global_step;
    log_util(p, out);
    vec_log(p->vec, out, 0);
}

// One dict copy per train log (+ optional final snapshot). Capacity fixed at
// train start: ≤ train_epochs + 1 final append.
typedef struct {
    Dict* items;
    int size;
    int capacity;
} PufLogHistory;

static void puf_log_history_init(PufLogHistory* h, int capacity) {
    h->size = 0;
    h->capacity = capacity;
    h->items = (Dict*)calloc(capacity, sizeof(Dict));
}

static void puf_log_history_add(PufLogHistory* h, Dict* log) {
    assert(h->size < h->capacity);
    dict_copy(&h->items[h->size], log);
    h->size++;
}

double rollout_start(PuffeRL* p, int slot) {
    p->write_slot = slot;
    if (p->hypers.async) {
        int64_t n = numel(p->param.shape);
        cudaMemcpyAsync(p->actor_param.data, p->param.data,
            n * sizeof(precision_t), cudaMemcpyDeviceToDevice, p->default_stream);
        cudaStreamSynchronize(p->default_stream);
    }
    if (p->hypers.reset_every_horizon) {
        for (int i = 0; i < p->hypers.num_buffers; i++) {
            cudaMemsetAsync(p->buffer_states[i].data, 0,
                numel(p->buffer_states[i].shape) * sizeof(precision_t), p->default_stream);
        }
        for (int b = 0; b < p->num_frozen_banks; b++) {
            for (int i = 0; i < p->hypers.num_buffers; i++) {
                Prec* st = &p->frozen_banks[b].buffer_states[i];
                cudaMemsetAsync(st->data, 0, numel(st->shape) * sizeof(precision_t),
                    p->default_stream);
            }
        }
        cudaStreamSynchronize(p->default_stream);
    }

    double t0 = wall_clock();
    rollout_start(p);
    return t0;
}
void rollouts(PuffeRL* p) {
    double t0 = rollout_start(p, 0);
    rollout_finish(p, t0);
    p->global_step += p->hypers.horizon * p->hypers.total_agents;
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
    float scores[TRAIN_RESULT_MAX_POINTS];
    float costs[TRAIN_RESULT_MAX_POINTS];
    float step_points[TRAIN_RESULT_MAX_POINTS];
} TrainResult;

#define EVAL_RENDER 0
#define EVAL_SCORE 1
#define EVAL_MATCH 2

#define SELFPLAY_MAX_BANKS 8
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
    char (*pool)[SELFPLAY_PATH_MAX];
    int pool_size;
    SelfplayBank banks[SELFPLAY_MAX_BANKS];
} Selfplay;

void selfplay_add_checkpoint(Selfplay* sp, const char* path) {
    while (access(path, R_OK) != 0) usleep(50000);
    for (int i = 0; i < sp->pool_size; i++) {
        if (strcmp(sp->pool[i], path) == 0) return;
    }
    if (sp->pool_size == sp->max_size) {
        memmove(sp->pool, sp->pool + 1,
            (sp->max_size - 1) * sizeof(*sp->pool));
        sp->pool_size--;
    }
    snprintf(sp->pool[sp->pool_size++], sizeof(sp->pool[0]), "%s", path);
}

const char* selfplay_sample(Selfplay* sp) {
    int idx = (int)(rand_r(&sp->rng) % (unsigned int)sp->pool_size);
    return sp->pool[idx];
}

typedef struct {
    char section[64];
    char key[64];
} SweepParam;

extern char** environ;

typedef struct {
    int run;
    int random;
    int gp_obs;
    int pareto;
    int fd;
    pid_t pid;
    float* sample;
    TrainResult result;
} SweepJob;

void run_sweep(Ini* ini, const char* exe_path) {
    // Build SweepSpace + param map from [sweep.<section>.<key>] sections.
    const char* goal = puf_ini_get_str(ini, "sweep", "goal");
    assert((strcmp(goal, "maximize") == 0 || strcmp(goal, "minimize") == 0)
        && "sweep.goal must be maximize or minimize");
    int direction = strcmp(goal, "minimize") == 0 ? -1 : 1;

    SweepParam* params = (SweepParam*)calloc(ini->num_sections, sizeof(SweepParam));
    SweepSpace* space = sweep_space_create(ini->num_sections, -1, direction);
    int n_params = 0;
    for (int i = 0; i < ini->num_sections; i++) {
        Dict* dict = &ini->sections[i];
        if (strncmp(dict->name, "sweep.", 6) != 0) {
            continue;
        }
        const char* path = dict->name + 6;
        const char* dot = strrchr(path, '.');
        assert(dot && dot != path && dot[1]
            && "expected section [sweep.<section>.<key>]");
        snprintf(params[n_params].section, sizeof(params[n_params].section), "%.*s",
            (int)(dot - path), path);
        snprintf(params[n_params].key, sizeof(params[n_params].key), "%s", dot + 1);

        const char* dist = dict_get_str(dict, "distribution");
        SpaceType type = SPACE_LINEAR;
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
            assert(0 && "invalid sweep distribution (use uniform/int_uniform/"
                "uniform_pow2/log_normal/logit_normal)");
        }

        float min_v = (float)dict_get(dict, "min");
        float max_v = (float)dict_get(dict, "max");
        const char* scale_s = dict_get_str(dict, "scale");
        float scale;
        if (strcmp(scale_s, "auto") == 0) {
            scale = 0.5f;
        } else if (strcmp(scale_s, "time") == 0) {
            assert(min_v > 0 && max_v > 0
                && "scale=time requires positive min/max");
            scale = 1.0f / (log2f(max_v) - log2f(min_v));
        } else {
            scale = (float)dict_get(dict, "scale");
        }

        space_init(&space->spaces[n_params], type, min_v, max_v, scale, is_integer);
        if (strcmp(dict->name, "sweep.train.total_timesteps") == 0) {
            space->cost_idx = n_params;
        }
        n_params++;
    }
    space->num = n_params;

    int max_runs = (int)puf_ini_get(ini, "sweep", "max_runs");
    int downsample = (int)puf_ini_get(ini, "sweep", "downsample");
    int prune_pareto = (int)puf_ini_get(ini, "sweep", "prune_pareto");
    const char* metric_dist = puf_ini_get_str(ini, "sweep", "metric_distribution");
    assert((strcmp(metric_dist, "linear") == 0 || strcmp(metric_dist, "logit") == 0)
        && "sweep.metric_distribution must be linear or logit");
    int use_logit = strcmp(metric_dist, "logit") == 0;
    float max_cost = (float)puf_ini_get(ini, "sweep", "max_suggestion_cost");
    float early_stop_quantile = (float)puf_ini_get(ini, "sweep", "early_stop_quantile");
    assert(max_runs >= 1 && "sweep.max_runs must be >= 1");
    assert(downsample >= 1 && downsample <= TRAIN_RESULT_MAX_POINTS
        && "sweep.downsample must be in [1, TRAIN_RESULT_MAX_POINTS]");
    int success_cap = max_runs * downsample * 2;
    if (success_cap < 8192) success_cap = 8192;

    // GPU packing: each trial is a full train (launch_train) that may itself
    // use train.gpus for NCCL DP. Concurrent trials = sweep_gpus / train_gpus
    // on disjoint blocks [0,W), [W,2W), ...  e.g. 8 GPUs, train.gpus=2 → 4 trials.
    int total_gpus = 0;
    cudaError_t err = cudaGetDeviceCount(&total_gpus);
    if (err != cudaSuccess || total_gpus < 1) {
        fprintf(stderr, "sweep error: no CUDA devices available\n");
        exit(1);
    }
    int sweep_gpus = (int)puf_ini_get(ini, "sweep", "gpus");
    int train_gpus = (int)puf_ini_get(ini, "train", "gpus");
    assert(sweep_gpus >= 0 && "sweep.gpus must be >= 0");
    if (sweep_gpus == 0) {
        sweep_gpus = total_gpus;
    }
    assert(sweep_gpus <= total_gpus
        && "sweep.gpus exceeds visible CUDA devices");
    assert(train_gpus >= 1 && "train.gpus must be >= 1");
    assert(sweep_gpus >= train_gpus
        && "sweep.gpus must be >= train.gpus");
    if ((int)puf_ini_get(ini, "sweep", "use_gpu")) {
        cudaSetDevice(sweep_gpus - 1);
    }

    int parallel = sweep_gpus / train_gpus;

    ProteinSweep* protein = protein_sweep_create(space,
        10, 256, 50, 0.001f, 50, 750, 4096,
        downsample == 1, prune_pareto, use_logit,
        1.0f, max_cost, 0.1f, -0.8f, early_stop_quantile,
        success_cap, 1024, 5, 73ULL);

    float* samples = (float*)calloc((parallel + 1) * space->num, sizeof(float));
    float* sample = samples;
    SweepJob* jobs = (SweepJob*)calloc(parallel, sizeof(SweepJob));
    int failed_workers = 0;
    int next_run_id = 0;
    int completed = 0;
    int active = 0;
    // Free-list: slot i owns GPUs [i*train_gpus, (i+1)*train_gpus). Refill as
    // soon as any trial exits (waitpid -1), so finished GPUs are never idle
    // while more runs remain.
    while (completed < max_runs || active > 0) {
        for (int i = 0; i < parallel; i++) {
            if (jobs[i].pid || completed + active >= max_runs) continue;

            ProteinSweepInfo info = {0};
            if (!next_run_id) {
                for (int j = 0; j < space->num; j++) {
                    float val = (float)puf_ini_get(ini, params[j].section, params[j].key);
                    float norm = space_normalize(&space->spaces[j], val);
                    assert(isfinite(norm) && norm >= -1.0f && norm <= 1.0f
                        && "default sweep value outside its sweep range");
                    sample[j] = norm;
                }
            } else {
                // Invalid train shapes die in the worker (launch_train asserts);
                // failed workers are observed as bad samples below.
                info = protein_sweep_suggest(protein, sample, NAN);
            }

            SweepJob job = {0};
            job.run = next_run_id++;
            job.random = info.is_random;
            job.gp_obs = info.n_gp_obs;
            job.pareto = info.n_pareto;
            job.sample = samples + (size_t)(i + 1) * space->num;
            memcpy(job.sample, sample, space->num * sizeof(float));

            if (job.run) {
                for (int p = 0; p < space->num; p++) {
                    float val = space_unnormalize(&space->spaces[p], sample[p]);
                    char buf[64];
                    snprintf(buf, sizeof(buf), "%.9g", val);
                    char key[256];
                    snprintf(key, sizeof(key), "%s.%s", params[p].section, params[p].key);
                    puf_ini_put(ini, key, buf);
                }
            }
            char run_id[128];
            snprintf(run_id, sizeof(run_id), "sweep_%ld_%04d",
                (long)(1000.0 * wall_clock()), job.run);
            puf_ini_put(ini, "base.run_id", run_id);

            // Spawn clean train process (after parent CUDA init for protein).
            // Child runs main→launch_train; train.gpus>1 DP-forks inside that process.
            int pipefd[2];
            assert(pipe(pipefd) == 0);
            char buf[64];
            snprintf(buf, sizeof(buf), "%d", i * train_gpus);
            puf_ini_put(ini, "base.gpu_offset", buf);
            snprintf(buf, sizeof(buf), "%d", pipefd[1]);
            puf_ini_put(ini, "base.result_fd", buf);

            int nkeys = 0;
            for (int s = 0; s < ini->num_sections; s++) {
                nkeys += ini->sections[s].size;
            }
            int argc = nkeys + 4;
            char** argv = (char**)calloc(argc, sizeof(char*));
            argv[0] = (char*)exe_path;
            argv[1] = (char*)"train";
            argv[2] = (char*)puf_ini_get_str(ini, "base", "env_name");
            int idx = 3;
            char full_key[PUF_DICT_MAX_KEY * 2];
            for (int s = 0; s < ini->num_sections; s++) {
                Dict* dict = &ini->sections[s];
                for (int k = 0; k < dict->size; k++) {
                    snprintf(full_key, sizeof(full_key), "%s.%s",
                        dict->name, dict->items[k].key);
                    DictItem* item = &dict->items[k];
                    char val[128];
                    const char* src = item->str;
                    if (!src) {
                        snprintf(val, sizeof(val), "%.17g", item->value);
                        src = val;
                    }
                    size_t n = strlen(full_key) + strlen(src) + 2;
                    char* out = (char*)malloc(n);
                    snprintf(out, n, "%s=%s", full_key, src);
                    argv[idx++] = out;
                }
            }
            argv[argc - 1] = NULL;

            posix_spawn_file_actions_t actions;
            posix_spawn_file_actions_init(&actions);
            posix_spawn_file_actions_addclose(&actions, pipefd[0]);
            posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
            pid_t pid = 0;
            int err = posix_spawnp(&pid, exe_path, &actions, NULL, argv, environ);
            posix_spawn_file_actions_destroy(&actions);
            for (int a = 3; a < argc - 1; a++) {
                free(argv[a]);
            }
            free(argv);
            if (err != 0) {
                fprintf(stderr, "posix_spawn train failed: %s\n", strerror(err));
                exit(1);
            }
            close(pipefd[1]);
            job.fd = pipefd[0];
            job.pid = pid;
            jobs[i] = job;
            active++;
        }
        if (!active) break;

        int status = 0;
        pid_t done = waitpid(-1, &status, 0);
        assert(done > 0 && "sweep waitpid failed");
        int i = 0;
        for (; i < parallel && jobs[i].pid != done; i++) {}
        assert(i < parallel && "waitpid reaped unknown child");
        SweepJob* job = &jobs[i];
        int ok = read(job->fd, &job->result, sizeof(job->result)) == sizeof(job->result);
        close(job->fd);
        job->pid = 0;
        active--;
        int good = ok && WIFEXITED(status) && WEXITSTATUS(status) == 0;

        if (!good) {
            fprintf(stderr, "sweep worker run=%d failed; marking sample bad\n", job->run);
            protein_sweep_observe(protein, job->sample, NAN, max_cost, 1);
            if (++failed_workers > 1000) {
                fprintf(stderr, "sweep error: too many failed workers\n");
                exit(1);
            }
        } else {
            // Non-selfplay: points[] is a downsampled learning curve.
            // Selfplay: points==1 — final pool-eval winrate (or last train metric).
            for (int pi = 0; pi < job->result.points; pi++) {
                protein_sweep_observe(protein, job->sample,
                    job->result.scores[pi], job->result.costs[pi], 0);
            }
            printf("sweep run=%d score=%.4f cost=%.2f steps=%.0f random=%d gp_obs=%d pareto=%d\n",
                job->run, job->result.score, job->result.cost, job->result.steps,
                job->random, job->gp_obs, job->pareto);
            completed++;
        }
    }

    free(jobs);
    free(samples);
    free(params);
    protein_sweep_destroy(protein);
    sweep_space_destroy(space);
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
    puf_ini_put(ini, "base.reset_every_horizon", "0");
    puf_ini_put(ini, "train.horizon", "1");

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
            pufferl->param, pufferl->default_stream, a_path);
        pufferl_load_frozen_bank(pufferl, 0, b_path);
    } else {
        char resolved_path[4096];
        const char* load_path = puf_checkpoint_path_key(ini,
            "load_model_path", resolved_path, sizeof(resolved_path));
        if (load_path) {
            puf_load_weights_into(pufferl->master_weights, pufferl->param,
                pufferl->default_stream, load_path);
            printf("Loaded weights from %s\n", load_path);
        }
    }

    Dict baseline = {0};
    long baseline_n = 0;
    while (true) {
        if (render) {
            puf_render(PUF_BACKEND == PUF_GPU ? NULL : &pufferl->vec->envs[0]);
        }
        rollouts(pufferl);
        Dict log = {0};
        trainer_eval_log(pufferl, &log);
        if (render) {
            puf_dashboard_print(ini, pufferl, &log, 0);
            continue;
        }

        long n = (long)dict_get(&log, "env/n");
        if (match) {
            if (n < num_games) {
                if (verbose && n > 0) {
                    printf("\rgames=%ld/%ld  A=%.3f  B=%.3f  draw=%.3f",
                        n, num_games,
                        dict_get(&log, "env/slot_0_score"),
                        dict_get(&log, "env/slot_1_score"),
                        dict_get(&log, "env/draw_rate"));
                }
                continue;
            }
            result.score = (float)dict_get(&log, "env/slot_0_score");
            result.draw = (float)dict_get(&log, "env/draw_rate");
            result.games = (int)n;
            break;
        }
        if (n == 0) {
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
        double score = dict_get(&log, "env/score");
        double perf = dict_get(&log, "env/perf");
        if (baseline_n > 0 && scored_n > 0) {
            double base_n = (double)baseline_n;
            double cur_n = (double)n;
            score = (score * cur_n - dict_get(&baseline, "env/score") * base_n) / scored_n;
            perf = (perf * cur_n - dict_get(&baseline, "env/perf") * base_n) / scored_n;
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
        puf_ini_put(ini, "vec.num_frozen_banks", "0");
        puf_ini_put(ini, "vec.frozen_bank_pct", "0");
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
    Selfplay selfplay = {0};
    if (use_selfplay) {
        char initial_checkpoint[4096];
        snprintf(initial_checkpoint, sizeof(initial_checkpoint),
            "%s/%016ld.bin", checkpoint_dir, pufferl->global_step);
        if (ctx->artifact_owner) {
            puf_save_weights(pufferl, initial_checkpoint);
        }
        selfplay.num_banks = pufferl->num_frozen_banks;
        if (selfplay.num_banks <= 0 || selfplay.num_banks > SELFPLAY_MAX_BANKS) {
            fprintf(stderr, "selfplay requires 1..%d frozen banks\n", SELFPLAY_MAX_BANKS);
            exit(1);
        }
        selfplay.max_size = (int)puf_ini_get(ini, "selfplay", "max_size");
        assert(selfplay.max_size > 0 && "selfplay.max_size must be positive");
        selfplay.pool = (char (*)[SELFPLAY_PATH_MAX])calloc(
            selfplay.max_size, sizeof(*selfplay.pool));
        selfplay.opp_timeout_steps = (long)puf_ini_get(ini, "selfplay", "opp_timeout_steps");
        selfplay.rng = (unsigned int)puf_ini_get(ini, "selfplay", "seed")
            + (unsigned int)pufferl->hypers.rank;
        long current_step = pufferl->global_step * pufferl->hypers.world_size;

        if (PUF_BACKEND != PUF_GPU) {
            Env* envs = pufferl->vec->envs;
            for (int i = 0; i < pufferl->vec->size; i++) {
                int tag = envs[i].tag;
                if (tag > 0 && tag <= selfplay.num_banks) {
                    selfplay.banks[tag - 1].num_envs++;
                }
            }
        }

        selfplay_add_checkpoint(&selfplay, initial_checkpoint);
        for (int b = 0; b < selfplay.num_banks; b++) {
            pufferl_load_frozen_bank(pufferl, b, selfplay_sample(&selfplay));
            selfplay.banks[b].opp_started_step = current_step;
        }
    }

    long total_timesteps = puf_ini_get(ini, "train", "total_timesteps");
    long batch_size = (long)puf_ini_get(ini, "vec", "total_agents") *
        (long)puf_ini_get(ini, "train", "horizon");
    long local_timesteps = total_timesteps / ctx->world_size;
    long train_epochs = local_timesteps / batch_size;
    long eval_epochs = train_epochs / 2;
    // Optional multiplier for longer post-train eval (noise studies). Default 1.
    double eval_epoch_mult = puf_ini_get(ini, "base", "eval_epoch_mult");
    if (eval_epoch_mult > 1.0) {
        eval_epochs = (long)fmax(1, (double)eval_epochs * eval_epoch_mult);
    }
    long checkpoint_interval = puf_ini_get(ini, "base", "checkpoint_interval");
    long eval_episodes = puf_ini_get(ini, "base", "eval_episodes");
    // Sweep objective: bare names → env/<name>; keys with '/' used as-is.
    char target_key[128];
    const char* metric = puf_ini_get_str(ini, "sweep", "metric");
    snprintf(target_key, sizeof(target_key), "%s%s",
        strchr(metric, '/') ? "" : "env/", metric);
    Dict last_log = {0};
    // At most one history entry per train epoch (+1 final snapshot for log dump).
    PufLogHistory log_history;
    puf_log_history_init(&log_history, (int)train_epochs + 1);
    TrainResult result = {0};
    char final_checkpoint[4096] = {0};
    double last_dashboard_time = 0;

    for (long epoch = 0; epoch < train_epochs + eval_epochs; epoch++) {
        if (epoch < train_epochs && pufferl->hypers.async) {
            int prefetch_next = epoch + 1 < train_epochs;
            if (!pufferl->async_boot) {
                double t0 = rollout_start(pufferl, 0);
                rollout_finish(pufferl, t0);
                pufferl->async_ready_slot = 0;
                pufferl->async_next_slot = 1;
                pufferl->async_boot = true;
            }

            int ready_slot = pufferl->async_ready_slot;
            int next_slot = pufferl->async_next_slot;
            double t0 = 0.0;
            if (prefetch_next) {
                t0 = rollout_start(pufferl, next_slot);
            }

            pufferl->global_step += pufferl->hypers.horizon * pufferl->hypers.total_agents;
            RolloutBuf train_src = rollout_time_view(&pufferl->rollouts,
                ready_slot * pufferl->hypers.horizon, pufferl->hypers.horizon);
            train_impl(pufferl, &train_src);

            if (prefetch_next) {
                rollout_finish(pufferl, t0);
                pufferl->async_ready_slot = next_slot;
                pufferl->async_next_slot = ready_slot;
            }
        } else {
            rollouts(pufferl);
            if (epoch < train_epochs) {
                train_impl(pufferl, NULL);
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
                snprintf(final_checkpoint, sizeof(final_checkpoint),
                    "%s", saved_checkpoint);
            }
        }
        if (use_selfplay && saved_checkpoint[0]) {
            selfplay_add_checkpoint(&selfplay, saved_checkpoint);
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
            long global_step = pufferl->global_step;
            double now = wall_clock();
            double dt = now - pufferl->last_log_time;
            double sps = dt > 0
                ? (double)(global_step - pufferl->last_log_step) / dt * pufferl->hypers.world_size
                : 0;
            pufferl->last_log_time = now;
            pufferl->last_log_step = global_step;

            dict_set(&new_log, "SPS", sps);
            dict_set(&new_log, "agent_steps", (double)global_step * pufferl->hypers.world_size);
            dict_set(&new_log, "uptime", now - pufferl->start_time);
            dict_set(&new_log, "epoch", (double)pufferl->epoch);

            vec_log(pufferl->vec, &new_log, 1);

            float losses_host[NUM_LOSSES];
            cudaMemcpy(losses_host, pufferl->losses, sizeof(losses_host),
                cudaMemcpyDeviceToHost);
            float loss_n = losses_host[LOSS_N];
            float inv_n = loss_n > 0 ? 1.0f / loss_n : 0.0f;
            dict_set(&new_log, "loss/policy", losses_host[LOSS_PG] * inv_n);
            dict_set(&new_log, "loss/value", losses_host[LOSS_VF] * inv_n);
            dict_set(&new_log, "loss/entropy", losses_host[LOSS_ENT] * inv_n);
            dict_set(&new_log, "loss/total", losses_host[LOSS_TOTAL] * inv_n);
            dict_set(&new_log, "loss/old_kl", losses_host[LOSS_OLD_APPROX_KL] * inv_n);
            dict_set(&new_log, "loss/kl", losses_host[LOSS_APPROX_KL] * inv_n);
            dict_set(&new_log, "loss/clipfrac", losses_host[LOSS_CLIPFRAC] * inv_n);
            cudaMemset(pufferl->losses, 0, NUM_LOSSES * sizeof(float));

            log_util(pufferl, &new_log);

            float train_total = 0;
            for (int i = 0; i < (int)(sizeof(PROF_NAMES) / sizeof(PROF_NAMES[0])); i++) {
                float sec = pufferl->profile.accum[i] / 1000.0f;
                char key[256];
                snprintf(key, sizeof(key), "perf/%s", PROF_NAMES[i]);
                dict_set(&new_log, key, sec);
                if (i >= PROF_TRAIN_MISC) {
                    train_total += sec;
                }
            }
            dict_set(&new_log, "perf/train", train_total);
            memset(pufferl->profile.accum, 0, sizeof(pufferl->profile.accum));
        }
        for (int i = 0; i < new_log.size; i++) {
            DictItem* item = &new_log.items[i];
            dict_set(&last_log, item->key, item->value);
        }
        dict_clear(&new_log);
        if (use_selfplay && !is_eval && PUF_BACKEND != PUF_GPU) {
            long current_step = pufferl->global_step * pufferl->hypers.world_size;
            Env* envs = pufferl->vec->envs;
            for (int b = 0; b < selfplay.num_banks; b++) {
                SelfplayBank* bank = &selfplay.banks[b];
                int tag = b + 1;
                if (bank->pending_path[0]) {
                    int aligned = 0;
                    for (int i = 0; i < pufferl->vec->size; i++) {
                        if (envs[i].tag == tag && envs[i].boundary_reached) {
                            aligned++;
                        }
                    }
                    if (aligned >= bank->num_envs) {
                        pufferl_load_frozen_bank(pufferl, b, bank->pending_path);
                        for (int i = 0; i < pufferl->vec->size; i++) {
                            if (envs[i].tag == tag) {
                                envs[i].boundary_reached = 0;
                            }
                        }
                        bank->pending_path[0] = 0;
                        bank->opp_started_step = current_step;
                    }
                } else if (selfplay.opp_timeout_steps > 0 &&
                        current_step - bank->opp_started_step >= selfplay.opp_timeout_steps) {
                    snprintf(bank->pending_path, sizeof(bank->pending_path), "%s",
                        selfplay_sample(&selfplay));
                    for (int i = 0; i < pufferl->vec->size; i++) {
                        if (envs[i].tag == tag) {
                            envs[i].boundary_reached = 0;
                        }
                    }
                }
            }
            dict_set(&last_log, "pool/size", selfplay.pool_size);
            dict_set(&last_log, "pool/num_banks", selfplay.num_banks);
        }

        int eval_done = is_eval && dict_get(&last_log, "env/n") > eval_episodes;
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
    // uptime/agent_steps always from the train log path. target_key only after the
    // train loop has seen at least one objective sample (else leave 0).
    result.cost = (float)dict_get(&last_log, "uptime");
    result.steps = (float)dict_get(&last_log, "agent_steps");
    DictItem* target = dict_find(&last_log, target_key);
    result.score = target ? (float)target->value : 0;

    int points = use_selfplay ? 1 : (int)puf_ini_get(ini, "sweep", "downsample");
    assert(points >= 1 && points <= TRAIN_RESULT_MAX_POINTS
        && "sweep.downsample must be in [1, TRAIN_RESULT_MAX_POINTS]");
    result.points = points;

    if (log_history.size == 0 || points == 1) {
        result.scores[0] = result.score;
        result.costs[0] = result.cost;
        result.step_points[0] = result.steps;
    } else {
        // History entries are only added after target_key appears, so all keys exist.
        float final_steps = (float)dict_get(
            &log_history.items[log_history.size - 1], "agent_steps");
        int cursor = 0;
        for (int p = 0; p < points; p++) {
            float step_target = final_steps * (float)p / (float)(points - 1);
            while (cursor + 1 < log_history.size &&
                    (float)dict_get(&log_history.items[cursor], "agent_steps") < step_target) {
                cursor++;
            }
            Dict* log = &log_history.items[cursor];
            result.scores[p] = (float)dict_get(log, target_key);
            result.costs[p] = (float)dict_get(log, "uptime");
            result.step_points[p] = (float)dict_get(log, "agent_steps");
        }
        result.scores[points - 1] = result.score;
        result.costs[points - 1] = result.cost;
        result.step_points[points - 1] = result.steps;
    }
    close_pufferl(pufferl);

    // Selfplay end rating: match final checkpoint vs a small fixed opponent pool.
    // Mean A winrate is the sole protein score when eval_games > 0.
    if (use_selfplay && final_checkpoint[0] && ctx->artifact_owner) {
        int max_opp = (int)puf_ini_get(ini, "selfplay", "eval_pool_size");
        long games = (long)puf_ini_get(ini, "selfplay", "eval_games");
        if (max_opp > 0 && games > 0) {
            int n_opp = 0;
            char games_buf[32];
            snprintf(games_buf, sizeof(games_buf), "%ld", games);
            puf_ini_put(ini, "base.num_games", games_buf);
            puf_ini_put(ini, "base.load_model_path", final_checkpoint);

            float sum = 0;
            for (int i = 0; i < selfplay.pool_size && n_opp < max_opp; i++) {
                const char* opponent = selfplay.pool[i];
                if (strcmp(opponent, final_checkpoint) == 0) continue;
                puf_ini_put(ini, "base.load_enemy_model_path", opponent);
                EvalResult r = run_eval(ini, ctx, EVAL_MATCH, 0);
                sum += r.score;
                n_opp++;
                printf("selfplay_eval vs %s games=%d score=%.4f draw=%.4f\n",
                    opponent, r.games, r.score, r.draw);
            }
            if (n_opp) {
                float pool_score = sum / n_opp;
                printf("selfplay_eval mean_score=%.4f n=%d\n", pool_score, n_opp);
                result.score = pool_score;
                result.points = 1;
                result.scores[0] = pool_score;
                result.costs[0] = result.cost;
                result.step_points[0] = result.steps;
                dict_set(&last_log, "selfplay/pool_score", pool_score);
            }
        }
    }

    if (ctx->artifact_owner) {
        puf_log_history_add(&log_history, &last_log);
        char log_path[4096];
        snprintf(log_path, sizeof(log_path), "%s/%s.ini", log_dir, run_id);

        FILE* fp = fopen(log_path, "w");
        if (!fp) {
            fprintf(stderr, "failed to write log %s\n", log_path);
            exit(1);
        }

        fprintf(fp, "# PufferLib log v1\n");
        puf_ini_write(fp, ini);
        fprintf(fp, "\n[metrics]\n");

        int downsample = (int)puf_ini_get(ini, "sweep", "downsample");
        Dict keys = {0};
        for (int i = 0; i < log_history.size; i++) {
            Dict* log = &log_history.items[i];
            for (int j = 0; j < log->size; j++) {
                if (!dict_find(&keys, log->items[j].key)) {
                    dict_set(&keys, log->items[j].key, 0);
                }
            }
        }
        int metric_points = downsample <= 1 ? 1 : downsample;
        double* out = (double*)calloc(metric_points, sizeof(double));
        double final_steps = dict_get(
            &log_history.items[log_history.size - 1], "agent_steps");

        for (int k = 0; k < keys.size; k++) {
            const char* key = keys.items[k].key;
            if (strncmp(key, "loss/", 5) == 0) {
                continue;
            }

            // Keys may be sparse across history (e.g. env/* only after first episode).
            double first_value = 0;
            for (int i = 0; i < log_history.size; i++) {
                DictItem* item = dict_find(&log_history.items[i], key);
                if (item) {
                    first_value = item->value;
                    break;
                }
            }

            if (metric_points == 1) {
                DictItem* item = dict_find(&log_history.items[log_history.size - 1], key);
                out[0] = item ? item->value : first_value;
                fprintf(fp, "%s = %.17g\n", key, out[0]);
                continue;
            }

            int out_idx = 0;
            int bin_n = 0;
            double fallback = first_value;
            double bin_sum = 0;
            double next_bin = final_steps / (metric_points - 1);
            for (int i = 0; i < log_history.size; i++) {
                Dict* log = &log_history.items[i];
                DictItem* item = dict_find(log, key);
                if (item) {
                    bin_sum += item->value;
                    bin_n++;
                }

                double steps = dict_get(log, "agent_steps");
                if (steps < next_bin || out_idx >= metric_points - 1) {
                    continue;
                }

                double reduced = bin_n ? bin_sum / bin_n : fallback;
                out[out_idx++] = reduced;
                fallback = reduced;
                bin_n = 0;
                bin_sum = 0;
                next_bin += final_steps / (metric_points - 1);
            }

            DictItem* final_item = dict_find(
                &log_history.items[log_history.size - 1], key);
            out[metric_points - 1] = final_item
                ? final_item->value
                : bin_n ? bin_sum / bin_n : fallback;
            while (out_idx < metric_points - 1) {
                out[out_idx++] = fallback;
            }
            fprintf(fp, "%s = ", key);
            for (int i = 0; i < metric_points; i++) {
                fprintf(fp, "%s%.17g", i ? "," : "", out[i]);
            }
            fputc('\n', fp);
        }

        free(out);
        fclose(fp);
    }
    for (int i = 0; i < log_history.size; i++) {
        dict_clear(&log_history.items[i]);
    }
    free(log_history.items);
    free(selfplay.pool);
    return result;
}

// Fork DP workers before CUDA initialization. Sweep trials occupy contiguous GPU
// blocks; rank 0 owns the last GPU and writes TrainResult to base.result_fd.
TrainResult launch_train(Ini* ini) {
    int mb = (int)puf_ini_get(ini, "train", "minibatch_size");
    int horizon = (int)puf_ini_get(ini, "train", "horizon");
    int agents = (int)puf_ini_get(ini, "vec", "total_agents");
    int world_size = (int)puf_ini_get(ini, "train", "gpus");
    assert(world_size >= 1 && "train.gpus must be >= 1");
    assert(horizon > 0 && mb % horizon == 0
        && "train.minibatch_size must be divisible by train.horizon");
    assert((long)mb <= (long)horizon * agents
        && "train.minibatch_size must be <= train.horizon * vec.total_agents");
    assert(horizon % ADV_VEC_WIDTH == 0
        && "train.horizon must be a multiple of ADV_VEC_WIDTH (4 float / 8 bf16)");

    int gpu_offset = (int)puf_ini_get(ini, "base", "gpu_offset");

    ncclUniqueId nccl_id;
    ncclUniqueId* nccl_ptr = NULL;
    if (world_size > 1) {
        ncclGetUniqueId(&nccl_id);
        nccl_ptr = &nccl_id;
    }

    int n_workers = world_size - 1;
    pid_t* pids = (pid_t*)calloc(n_workers, sizeof(pid_t));
    for (int rank = world_size - 1; rank >= 1; rank--) {
        pid_t pid = fork();
        if (pid < 0) {
            fprintf(stderr, "fork failed: %s\n", strerror(errno));
            exit(1);
        }
        if (pid == 0) {
            assert(freopen("/dev/null", "w", stdout) == stdout);
            TrainContext child = {
                .rank = rank,
                .world_size = world_size,
                .gpu_id = gpu_offset + rank - 1,
                .artifact_owner = 0,
                .nccl_id = nccl_ptr,
            };
            run_train(ini, &child);
            puf_ini_free(ini);
            exit(0);
        }
        pids[rank - 1] = pid;
    }

    TrainContext host = {
        .rank = 0,
        .world_size = world_size,
        .gpu_id = gpu_offset + world_size - 1,
        .artifact_owner = 1,
        .nccl_id = nccl_ptr,
    };
    TrainResult result = run_train(ini, &host);
    for (int i = 0; i < n_workers; i++) {
        int status = 0;
        waitpid(pids[i], &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            fprintf(stderr, "train rank worker pid %d failed\n", (int)pids[i]);
            exit(1);
        }
    }
    free(pids);

    // Sweep parent reads this over the pipe; CLI train ignores (result_fd=0).
    int result_fd = (int)puf_ini_get(ini, "base", "result_fd");
    if (result_fd > 0) {
        assert(write(result_fd, &result, sizeof(result)) == sizeof(result));
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
    Ini ini = {0};
    puf_ini_load_env(&ini, argv[2], argc - 3, argv + 3);
    TrainContext ctx = {.world_size = 1, .artifact_owner = 1};

    if (strcmp(mode, "train") == 0) {
        launch_train(&ini);
    } else if (strcmp(mode, "sweep") == 0) {
        run_sweep(&ini, argv[0]);
    } else if (strcmp(mode, "eval") == 0 || strcmp(mode, "eval_bot") == 0) {
        if (strcmp(mode, "eval_bot") == 0) {
            puf_ini_put(&ini, "vec.num_frozen_banks", "0");
            puf_ini_put(&ini, "vec.frozen_bank_pct", "0");
            puf_ini_put(&ini, "selfplay.enabled", "0");
            puf_ini_put(&ini, "env.dr", "0");
            puf_ini_put(&ini, "env.num_agents", "1");
            puf_ini_put(&ini, "env.num_bots", "1");
        }
        // Headless score eval (use a separate interactive path if you need render).
        run_eval(&ini, &ctx, EVAL_SCORE, 1);
    } else if (strcmp(mode, "match") == 0) {
        run_eval(&ini, &ctx, EVAL_MATCH, 1);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        exit(1);
    }

    puf_ini_free(&ini);
    return 0;
}

#endif

