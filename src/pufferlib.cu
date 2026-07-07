#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include <nvml.h>
#include <nccl.h>

#include <time.h>
#include "config.h"
#include "models.cu"
#include "ocean.cu"
#include "muon.cu"
#include "vecenv.h"

#define _PUFFER_STRINGIFY(x) #x
#define PUFFER_STRINGIFY(x) _PUFFER_STRINGIFY(x)

static double wall_clock() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

enum ProfileIdx {
    PROF_ROLLOUT = 0,
    PROF_EVAL_GPU,
    PROF_EVAL_ENV,
    PROF_TRAIN_MISC,
    PROF_TRAIN_FORWARD,
    NUM_PROF,
};

static const char* PROF_NAMES[NUM_PROF] = {
    "rollout",
    "eval_gpu",
    "eval_env",
    "train_misc",
    "train_forward",
};

#define NUM_TRAIN_EVENTS 5
typedef struct {
    cudaEvent_t events[NUM_TRAIN_EVENTS];
    float accum[NUM_PROF];
} ProfileT;

// Data collected by parallel environment workers. Each worker handles
// a constant subset of agents 
struct RolloutBuf {
    PrecisionTensor observations;  // (horizon, agents, input_size)
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
    bufs = (RolloutBuf){
        .observations = {.shape = {T, B, input_size}},
        .actions      = {.shape = {T, B, num_atns}},
        .values       = {.shape = {T, B}},
        .logprobs     = {.shape = {T, B}},
        .rewards      = {.shape = {T, B}},
        .terminals    = {.shape = {T, B}},
        .ratio        = {.shape = {T, B}},
        .importance   = {.shape = {T, B}},
        .action_mask  = {},
    };
    alloc_register(alloc, &bufs.observations);
    alloc_register(alloc, &bufs.actions);
    alloc_register(alloc, &bufs.values);
    alloc_register(alloc, &bufs.logprobs);
    alloc_register(alloc, &bufs.rewards);
    alloc_register(alloc, &bufs.terminals);
    alloc_register(alloc, &bufs.ratio);
    alloc_register(alloc, &bufs.importance);
    if (mask_size > 0) {
        bufs.action_mask = {.shape = {T, B, mask_size}};
        alloc_register(alloc, &bufs.action_mask);
    }
}

// Train data layout is transposed to (B, T) from rollouts layout (T, B)
// This allows env workers to collect data with contiguous writes and
// training to perform several (though not all) ops in contiguous memory
struct TrainGraph {
    PrecisionTensor mb_state;       // (layers, B, hidden)
    PrecisionTensor mb_obs;         // (B, T, input_size)
    PrecisionTensor mb_actions;     // (B, T, num_atns)
    PrecisionTensor mb_logprobs;    // (B, T)
    PrecisionTensor mb_advantages;  // ...
    PrecisionTensor mb_values;
    PrecisionTensor mb_returns;
    PrecisionTensor mb_ratio;
    PrecisionTensor mb_newvalue;
    PrecisionTensor mb_prio;        // (B,)
    PrecisionTensor mb_action_mask; // (B, T, mask_size); .data=nullptr when disabled
};

void register_train_buffers(TrainGraph& bufs, Allocator* alloc, int B, int T, int input_size,
        int hidden_size, int num_atns, int num_layers, int mask_size) {
    bufs = (TrainGraph){
        .mb_state =         {.shape = {num_layers, B, hidden_size}},
        .mb_obs =           {.shape = {B, T, input_size}},
        .mb_actions =       {.shape = {B, T, num_atns}},
        .mb_logprobs =      {.shape = {B, T}},
        .mb_advantages =    {.shape = {B, T}},
        .mb_values =        {.shape = {B, T}},
        .mb_returns =       {.shape = {B, T}},
        .mb_ratio =         {.shape = {B, T}},
        .mb_newvalue =      {.shape = {B, T}},
        .mb_prio =          {.shape = {B}},
        .mb_action_mask =   {},
    };
    alloc_register(alloc, &bufs.mb_obs);
    alloc_register(alloc, &bufs.mb_state);
    alloc_register(alloc, &bufs.mb_actions);
    alloc_register(alloc, &bufs.mb_logprobs);
    alloc_register(alloc, &bufs.mb_advantages);
    alloc_register(alloc, &bufs.mb_prio);
    alloc_register(alloc, &bufs.mb_values);
    alloc_register(alloc, &bufs.mb_returns);
    alloc_register(alloc, &bufs.mb_ratio);
    alloc_register(alloc, &bufs.mb_newvalue);
    if (mask_size > 0) {
        bufs.mb_action_mask = {.shape = {B, T, mask_size}};
        alloc_register(alloc, &bufs.mb_action_mask);
    }
}

__device__ __forceinline__ float finite_or_clamp(float x, float lo, float hi) {
    if (isnan(x)) {
        return 0.0f;
    }
    if (isinf(x)) {
        return x > 0.0f ? hi : lo;
    }
    return fminf(hi, fmaxf(lo, x));
}

__device__ __forceinline__ float safe_continuous_mean(const precision_t* logits, int idx) {
    return finite_or_clamp(to_float(logits[idx]), -1.0e6f, 1.0e6f);
}

__device__ __forceinline__ float safe_continuous_logstd(const precision_t* logstd, int idx) {
    return finite_or_clamp(to_float(logstd[idx]), -20.0f, 2.0f);
}

#include "loss.cu"

// Prioritized replay over single-epoch data. These kernels are
// the least cleaned because we will likely have a better method in 5.0
struct PrioBuffers {
    FloatTensor prio_probs, cdf, mb_prio;
    IntTensor idx;
};

void register_prio_buffers(PrioBuffers& bufs, Allocator* alloc, int B, int minibatch_segments) {
    bufs = (PrioBuffers){
        .prio_probs = {.shape = {B}},
        .cdf = {.shape = {B}},
        .mb_prio = {.shape = {minibatch_segments}},
        .idx = {.shape = {minibatch_segments}},
    };
    alloc_register(alloc, &bufs.prio_probs);
    alloc_register(alloc, &bufs.cdf);
    alloc_register(alloc, &bufs.idx);
    alloc_register(alloc, &bufs.mb_prio);
}

// Slice: select dim0 index t, then narrow dim0 from start for count.
// 3D (T, B, F) -> (count, F); 2D (T, B) -> (count,)
inline PrecisionTensor puf_slice(PrecisionTensor& p, int t, int start, int count) {
    if (ndim(p.shape) == 3) {
        long B = p.shape[1], F = p.shape[2];
        return {.data = p.data + (t*B + start)*F, .shape = {count, F}};
    } else {
        long B = p.shape[1];
        return {.data = p.data + (t*B + start), .shape = {count}};
    }
}

struct EnvBuf {
    ObsTensor obs;         // (total_agents, obs_size)
    FloatTensor actions;   // (total_agents, num_atns)
    FloatTensor rewards;   // (total_agents,)
    FloatTensor terminals; // (total_agents,)
    ByteTensor action_mask; // (total_agents, mask_size); .data=nullptr when env opts out
};

static int puf_act_sizes[] = ACT_SIZES;

VecEnv* create_environments(Dict* vec_kwargs, Dict* env_kwargs, EnvBuf& env) {
    VecEnv* vec = vec_create(vec_kwargs, env_kwargs);
    int total_agents = vec->total_agents;
    env.obs = { .data = vec->gpu_observations, .shape = {total_agents, OBS_SIZE} };
    env.actions = { .data = (float*)vec->gpu_actions, .shape = {total_agents, NUM_ATNS} };
    env.rewards = { .data = (float*)vec->gpu_rewards, .shape = {total_agents} };
    env.terminals = { .data = (float*)vec->gpu_terminals, .shape = {total_agents} };
    if (vec->action_mask_size > 0) {
        env.action_mask = { .data = vec->gpu_action_mask,
                            .shape = {total_agents, vec->action_mask_size} };
    } else {
        env.action_mask = { .data = nullptr, .shape = {0} };
    }
    return vec;
}

// A frozen weight bank: same shape as the primary, but its own params buffer
// (and per-buffer rollout states/activations). Used for match (eval) and league
// (frozen historical opponents). Not trained; updated only via load.
typedef struct {
    Policy policy;  // Bank-owned Policy; lets banks have different arch than primary.
    PolicyWeights weights;
    Allocator params_alloc;
    Allocator acts_alloc;
    PrecisionTensor param_puf;
    FloatTensor master_weights;
    PrecisionTensor* buffer_states;         // [num_buffers]
    PolicyActivations* buffer_activations;  // [num_buffers]
    int slice_size;  // # agents per buffer this bank owns; sets activation/state batch dim
    int hidden_size;
    int num_layers;
} WeightBank;

typedef struct PuffeRL {
    Policy policy;
    PolicyWeights weights;       // current precision_t weights (structured)
    PolicyActivations train_activations;
    Allocator params_alloc;
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
    cudaGraphExec_t* fused_rollout_cudagraphs;  // [horizon][num_buffers]
    cudaGraphExec_t train_cudagraph;
    cudaStream_t* streams;  // per-buffer raw CUDA streams
    cudaStream_t default_stream;  // main-thread stream (captured once at init)
    IntTensor act_sizes_puf;    // CUDA int32 tensor of action head sizes
    FloatTensor losses_puf;     // (NUM_LOSSES,) f32 accumulator
    PPOBuffersPuf ppo_bufs_puf; // Pre-allocated buffers for ppo_loss_fwd_bwd
    PrioBuffers prio_bufs;      // Pre-allocated buffers for prio_replay
    FloatTensor master_weights;  // fp32 master weights (flat); same buffer as param_puf in fp32 mode
    PrecisionTensor param_puf;
    PrecisionTensor grad_puf;
    LongTensor rng_offset_puf;   // (num_buffers+1,) int64 CUDA device counters
    ProfileT profile;
    nvmlDevice_t nvml_device;
    long epoch;
    long global_step;
    double start_time;
    double last_log_time;
    long last_log_step;
    int train_warmup;
    bool rollout_captured;
    bool train_captured;
    ulong seed;
    curandStatePhilox4_32_10_t** rng_states;  // per-buffer persistent RNG states [num_buffers]
    // Optional frozen weight banks for match / league.
    WeightBank* frozen_banks;  // [num_frozen_banks]
    int num_frozen_banks;
    char env_name[64];  // Kept for post-init bank adds (needs create_custom_encoder).
    // Per-buffer-relative bank layout: bank_layout[b] = first agent within each
    // buffer chunk owned by bank b. Length num_banks+1; ends at agents_per_buffer.
    // Same shape applied to every buffer (each buffer hosts every bank), so each
    // worker thread only writes inside its own physical chunk.
    // Bank 0 = primary (learner). NULL = no layout set (primary owns full chunk).
    int* bank_layout;
} PuffeRL;

#include "checkpoint.h"

void log_environments_impl(PuffeRL& pufferl, Dict* out) {
    vec_log(pufferl.vec, out, 1);
}

inline void profile_begin(const char* tag, bool enable) {
    if (enable) nvtxRangePushA(tag);
}

inline void profile_end(bool enable) {
    if (enable) nvtxRangePop();
}

// Thread-local stream for per-buffer rollout threads.
static thread_local cudaStream_t tl_stream = 0;

void pufferl_thread_init(PuffeRL* pufferl, int buf) {
    tl_stream = pufferl->streams[buf];
}

__global__ void rng_init(curandStatePhilox4_32_10_t* states, uint64_t seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__device__ __forceinline__ float safe_logit(const precision_t* logits,
        int logits_base, int logits_offset, int offset) {
    float l = to_float(logits[logits_base + logits_offset + offset]);
    if (isnan(l)) {
        l = 0.0f;
    }
    if (isinf(l)) {
        l = (l > 0) ? 3.4028e+38f : -3.4028e+38f;
    }
    return l;
}

__device__ __forceinline__ float masked_logit(const precision_t* logits,
        int logits_base, int logits_offset, int offset,
        const precision_t* mask, int mask_base) {
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
        precision_t* __restrict__ actions,    // (B, num_atns)
        precision_t* __restrict__ logprobs,   // (B,)
        precision_t* __restrict__ value_out,  // (B,)
        curandStatePhilox4_32_10_t* __restrict__ rng_states,
        const precision_t* __restrict__ action_mask, // (B, A_total) or nullptr
        int mask_stride) {                    // 0 when action_mask is nullptr
    int B = dec_out.shape[0];
    int fused_cols = dec_out.shape[1];
    int num_atns = numel(act_sizes_puf.shape);
    const int* act_sizes = act_sizes_puf.data;
    const precision_t* logits = dec_out.data;
    int logits_stride = fused_cols;
    int value_stride = fused_cols;
    bool is_continuous = logstd_puf.data != nullptr && numel(logstd_puf.shape) > 0;
    const precision_t* logstd = logstd_puf.data;
    int logstd_stride = is_continuous ? 0 : 0;  // 1D broadcast: stride 0
    const precision_t* value = logits + (fused_cols - 1);  // last column

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B) {
        return;
    }

    // Load persistent RNG state (advanced in-place each call)
    curandStatePhilox4_32_10_t state = rng_states[idx];

    int logits_base = idx * logits_stride;
    float total_log_prob = 0.0f;

    if (is_continuous) {
        // Continuous action sampling from Normal(mean, exp(logstd))
        constexpr float LOG_2PI = 1.8378770664093453f;  // log(2*pi)
        int logstd_base = idx * logstd_stride;  // separate stride for logstd (may be 0 for broadcast)

        for (int h = 0; h < num_atns; ++h) {
            float mean = safe_continuous_mean(logits, logits_base + h);
            float log_std = safe_continuous_logstd(logstd, logstd_base + h);
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
    } else {
        // Discrete action sampling (original multinomial logic)
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

void pufferl_forward(PuffeRL* pufferl, int buf, int t) {
    HypersT& hypers = pufferl->hypers;
    int graph = t * hypers.num_buffers + buf;
    profile_begin("fused_rollout", hypers.profile);

    cudaStream_t current_stream = tl_stream;
    if (pufferl->rollout_captured) {
        assert(cudaGraphLaunch(pufferl->fused_rollout_cudagraphs[graph], current_stream) == cudaSuccess
                && "cudaGraphLaunch failed");
        profile_end(hypers.profile);
        return;
    }

    bool capturing = pufferl->epoch == hypers.cudagraphs;
    if (capturing) {
        assert(cudaStreamBeginCapture(current_stream, cudaStreamCaptureModeGlobal) == cudaSuccess
                && "cudaStreamBeginCapture failed");
    }

    RolloutBuf& rollouts = pufferl->rollouts;
    EnvBuf& env = pufferl->env;
    int block_size = pufferl->vec->total_agents / hypers.num_buffers;
    int start = buf * block_size;
    cudaStream_t stream = current_stream;

    // Copy observations, rewards, terminals from GPU env buffers to rollout buffer
    ObsTensor& obs_env = env.obs;
    int n = block_size * obs_env.shape[1];
    PrecisionTensor obs_dst = puf_slice(rollouts.observations, t, start, block_size);
    cast_dispatch(obs_dst.data, obs_env.data + (long)start*obs_env.shape[1], n, stream);

    PrecisionTensor rew_dst = puf_slice(rollouts.rewards, t, start, block_size);
    n = block_size;
    cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
        rew_dst.data, env.rewards.data + start, n);

    PrecisionTensor term_dst = puf_slice(rollouts.terminals, t, start, block_size);
    cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
        term_dst.data, env.terminals.data + start, n);

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

    // Per-bank policy forward + sampling. Each bank owns a contiguous sub-range
    // [bank_layout[b], bank_layout[b+1]) within every buffer's chunk; layout is
    // per-buffer-relative so each worker writes only inside its own chunk.
    // Cudagraph capture absorbs the extra kernel launches.
    int num_banks = 1 + pufferl->num_frozen_banks;
    long act_cols = env.actions.shape[1];
    for (int b = 0; b < num_banks; b++) {
        int bank_off = pufferl->bank_layout ? pufferl->bank_layout[b] : 0;
        int bank_end = pufferl->bank_layout ? pufferl->bank_layout[b + 1] : block_size;
        int bank_size = bank_end - bank_off;
        if (bank_size == 0) continue;

        Policy* p_bank;
        PolicyWeights* w_bank;
        PolicyActivations* a_bank;
        PrecisionTensor* s_bank;
        if (b == 0) {
            p_bank = &pufferl->policy;
            w_bank = &pufferl->weights;
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
        assert(cudaStreamEndCapture(current_stream, &_graph) == cudaSuccess
                && "cudaStreamEndCapture failed");
        assert(cudaGraphInstantiate(&pufferl->fused_rollout_cudagraphs[graph], _graph, 0) == cudaSuccess
                && "cudaGraphInstantiate failed");
        assert(cudaGraphDestroy(_graph) == cudaSuccess && "cudaGraphDestroy failed");
        cudaDeviceSynchronize();
    }
    profile_end(hypers.profile);
}


#include "advantage.cu"

// Minor copy bandwidth optimizations
__global__ void index_copy(char* __restrict__ dst, const int* __restrict__ idx,
        const char* __restrict__ src, int num_idx, int row_bytes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_idx) {
        int dst_row = idx[i];
        memcpy(dst + (int64_t)dst_row * row_bytes, src + (int64_t)i * row_bytes, row_bytes);
    }
}

inline float cosine_annealing(float lr_base, float lr_min, long t, long T) {
    if (T == 0) return lr_base;
    float ratio = (double )t / (double) T;
    ratio = std::max(0.0f, std::min(1.0f, ratio));
    return lr_min + 0.5f*(lr_base - lr_min)*(1.0f + std::cos(M_PI * ratio));
}

void train_impl(PuffeRL& pufferl) {
    // Update to HypersT& p
    HypersT& hypers = pufferl.hypers;

    cudaEventRecord(pufferl.profile.events[0]);  // pre-loop start
    cudaStream_t train_stream = pufferl.default_stream;

    // Transpose from rollout layout (T, B, ...) to train layout (B, T, ...)
    RolloutBuf& src = pufferl.rollouts;
    RolloutBuf& rollouts = pufferl.train_rollouts;
    PrecisionTensor& advantages_puf = pufferl.advantages_puf;

    int T = src.observations.shape[0], B = src.observations.shape[1];
    int obs_size = (ndim(src.observations.shape) >= 3) ? src.observations.shape[2] : 1;
    int num_atns = (ndim(src.actions.shape) >= 3) ? src.actions.shape[2] : 1;

    transpose_102<<<grid_size(T*B*obs_size), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.observations.data, src.observations.data, T, B, obs_size);
    transpose_102<<<grid_size(T*B*num_atns), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.actions.data, src.actions.data, T, B, num_atns);
    transpose_102<<<grid_size(T*B), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.logprobs.data, src.logprobs.data, T, B, 1);
    transpose_102<<<grid_size(T*B), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.rewards.data, src.rewards.data, T, B, 1);
    transpose_102<<<grid_size(T*B), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.terminals.data, src.terminals.data, T, B, 1);
    transpose_102<<<grid_size(T*B), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.ratio.data, src.ratio.data, T, B, 1);
    transpose_102<<<grid_size(T*B), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.values.data, src.values.data, T, B, 1);
    if (src.action_mask.data != nullptr) {
        int mask_size = src.action_mask.shape[2];
        transpose_102<<<grid_size(T*B*mask_size), BLOCK_SIZE, 0, train_stream>>>(
            rollouts.action_mask.data, src.action_mask.data, T, B, mask_size);
    }

    // We hard-clamp rewards to -1, 1. Our envs are mostly designed to respect this range
    clamp_precision_kernel<<<grid_size(numel(rollouts.rewards.shape)), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.rewards.data, -1.0f, 1.0f, numel(rollouts.rewards.shape));

    // Set importance weights to 1.0
    fill_precision_kernel<<<grid_size(numel(rollouts.ratio.shape)), BLOCK_SIZE, 0, train_stream>>>(
        rollouts.ratio.data, from_float(1.0f), numel(rollouts.ratio.shape));

    // Inline any of these only used once
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

    // Annealed priority exponent
    float anneal_beta = prio_beta0 + (1.0f - prio_beta0) * prio_alpha * (float)current_epoch/(float)total_epochs;
    TrainGraph& graph = pufferl.train_buf;
    cudaEventRecord(pufferl.profile.events[1]);  // pre-loop end

    int total_minibatches = hypers.replay_ratio * batch_size / hypers.minibatch_size;
    for (int mb = 0; mb < total_minibatches; ++mb) {
        cudaEventRecord(pufferl.profile.events[2]);  // start of misc (overwritten each iter)
        puf_zero(&advantages_puf, train_stream);

        profile_begin("compute_advantage", hypers.profile);
        puff_advantage_cuda(rollouts.values, rollouts.rewards, rollouts.terminals,
            rollouts.ratio, advantages_puf, hypers.gamma, hypers.gae_lambda,
            hypers.vtrace_rho_clip, hypers.vtrace_c_clip, train_stream);
        if (pufferl.num_frozen_banks > 0 && pufferl.bank_layout != NULL) {
            int apb = hypers.total_agents / hypers.num_buffers;
            zero_frozen_advantages_cuda(advantages_puf, apb,
                pufferl.bank_layout[1], train_stream);
        }
        profile_end(hypers.profile);

        profile_begin("compute_prio", hypers.profile);
        // Use the training RNG offset slot (last slot, index num_buffers)
        long* train_rng_offset = pufferl.rng_offset_puf.data + hypers.num_buffers;
        prio_replay_cuda(advantages_puf, prio_alpha, minibatch_segments,
            hypers.total_agents, anneal_beta,
            pufferl.prio_bufs, pufferl.seed, train_rng_offset, train_stream);
        profile_end(hypers.profile);

        profile_begin("train_select_and_copy", hypers.profile);
        if (hypers.reset_state) puf_zero(&graph.mb_state, train_stream);
        {
            RolloutBuf sel_src = rollouts;
            sel_src.values = rollouts.values;
            int mb_segs = pufferl.prio_bufs.idx.shape[0];
            int channels = (graph.mb_action_mask.data != nullptr) ? 6 : 5;
            select_copy<<<dim3(mb_segs, channels), SELECT_COPY_THREADS, 0, train_stream>>>(
                sel_src, graph, pufferl.prio_bufs.idx.data,
                advantages_puf.data, pufferl.prio_bufs.mb_prio.data);
        }
        profile_end(hypers.profile);

        cudaEventRecord(pufferl.profile.events[3]);  // end misc / start forward
        profile_begin("train_forward_backward", hypers.profile);
        if (pufferl.train_captured) {
            cudaGraphLaunch(pufferl.train_cudagraph, train_stream);
        } else {
            bool capturing = pufferl.train_warmup == hypers.cudagraphs;
            if (capturing) {
                assert(cudaStreamBeginCapture(train_stream, cudaStreamCaptureModeGlobal) == cudaSuccess
                        && "cudaStreamBeginCapture failed");
            }

            cudaStream_t stream = train_stream;
            PrecisionTensor obs_puf = graph.mb_obs;
            PrecisionTensor state_puf = graph.mb_state;
            PrecisionTensor dec_puf = policy_forward_train(&pufferl.policy, pufferl.weights, pufferl.train_activations, obs_puf, state_puf, stream);
            DecoderWeights* dw_train = (DecoderWeights*)pufferl.weights.decoder;
            PrecisionTensor p_logstd;
            if (dw_train->continuous) {
                p_logstd = dw_train->logstd;
            }

            ppo_loss_fwd_bwd(dec_puf, p_logstd, graph,
                pufferl.act_sizes_puf, pufferl.losses_puf,
                hypers.clip_coef, hypers.vf_clip_coef, hypers.vf_coef, current_ent_coef,
                pufferl.ppo_bufs_puf, pufferl.is_continuous, stream);

            FloatTensor grad_logits_puf = pufferl.ppo_bufs_puf.grad_logits;
            FloatTensor grad_logstd_puf = pufferl.is_continuous ? pufferl.ppo_bufs_puf.grad_logstd : FloatTensor();
            FloatTensor grad_values_puf = pufferl.ppo_bufs_puf.grad_values;
            policy_backward(&pufferl.policy, pufferl.weights, pufferl.train_activations,
                grad_logits_puf, grad_logstd_puf, grad_values_puf, stream);

            muon_step(&pufferl.muon, pufferl.master_weights, pufferl.grad_puf, hypers.max_grad_norm, stream);
            if (USE_BF16) {
                int n = numel(pufferl.param_puf.shape);
                cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
                    pufferl.param_puf.data, pufferl.master_weights.data, n);
            }
            if (capturing) {
                cudaGraph_t _graph;
                assert(cudaStreamEndCapture(train_stream, &_graph) == cudaSuccess
                        && "cudaStreamEndCapture failed");
                assert(cudaGraphInstantiate(&pufferl.train_cudagraph, _graph, 0) == cudaSuccess
                        && "cudaGraphInstantiate failed");
                assert(cudaGraphDestroy(_graph) == cudaSuccess && "cudaGraphDestroy failed");
                cudaDeviceSynchronize();
                pufferl.train_captured = true;
            }
            pufferl.train_warmup++;
        }
        profile_end(hypers.profile);

        // This version is consistent with PufferLib 3.0. One of the major algorithmic
        // questions remaining is how and when to update value and advantage estimates.
        {
            int num_idx = numel(pufferl.prio_bufs.idx.shape);
            int row_bytes = (numel(graph.mb_ratio.shape) / graph.mb_ratio.shape[0]) * sizeof(precision_t);
            index_copy<<<grid_size(num_idx), BLOCK_SIZE, 0, train_stream>>>(
                (char*)rollouts.ratio.data, pufferl.prio_bufs.idx.data,
                (const char*)graph.mb_ratio.data, num_idx, row_bytes);
        }
        {
            int num_idx = numel(pufferl.prio_bufs.idx.shape);
            int row_bytes = graph.mb_newvalue.shape[1] * sizeof(precision_t);
            index_copy<<<grid_size(num_idx), BLOCK_SIZE, 0, train_stream>>>(
                (char*)rollouts.values.data, pufferl.prio_bufs.idx.data,
                (const char*)graph.mb_newvalue.data, num_idx, row_bytes);
        }
        cudaEventRecord(pufferl.profile.events[4]);  // end forward
    }
    pufferl.epoch += 1;

    cudaStreamSynchronize(pufferl.default_stream);

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
        pufferl.profile.accum[PROF_TRAIN_FORWARD] += ms * total_minibatches;
    }

}

// Build a Policy value for a given env + arch. Encoder/decoder algorithms are
// fixed by the env; hidden_size/num_layers/horizon parameterize shape. Policy
// has no heap state so this returns by value; callers store it wherever.
static Policy build_policy(const char* env_name, int input_size, int hidden_size,
                           int num_layers, int decoder_output_size, int act_n,
                           bool is_continuous, int horizon) {
    Encoder encoder = {
        .forward = encoder_forward,
        .backward = encoder_backward,
        .init_weights = encoder_init_weights,
        .reg_params = encoder_reg_params,
        .reg_train = encoder_reg_train,
        .reg_rollout = encoder_reg_rollout,
        .create_weights = encoder_create_weights,
        .free_weights = encoder_free_weights,
        .free_activations = encoder_free_activations,
        .in_dim = input_size, .out_dim = hidden_size,
        .activation_size = sizeof(EncoderActivations),
    };
    create_custom_encoder(env_name, &encoder);
    Decoder decoder = {
        .forward = decoder_forward,
        .backward = decoder_backward,
        .init_weights = decoder_init_weights,
        .reg_params = decoder_reg_params,
        .reg_train = decoder_reg_train,
        .reg_rollout = decoder_reg_rollout,
        .create_weights = decoder_create_weights,
        .free_weights = decoder_free_weights,
        .free_activations = decoder_free_activations,
        .hidden_dim = hidden_size, .output_dim = decoder_output_size, .continuous = is_continuous,
    };
    Network network = {
        .forward = mingru_forward,
        .forward_train = mingru_forward_train,
        .backward = mingru_backward,
        .init_weights = mingru_init_weights,
        .reg_params = mingru_reg_params,
        .reg_train = mingru_reg_train,
        .reg_rollout = mingru_reg_rollout,
        .create_weights = mingru_create_weights,
        .free_weights = mingru_free_weights,
        .free_activations = mingru_free_activations,
        .hidden = hidden_size, .num_layers = num_layers, .horizon = horizon,
    };
    return Policy{
        .encoder = encoder, .decoder = decoder, .network = network,
        .input_dim = input_size, .hidden_dim = hidden_size, .output_dim = decoder_output_size,
        .num_atns = act_n,
    };
}

// Allocate a fresh frozen WeightBank with its own Policy (may differ in
// hidden_size/num_layers from primary). slice_size = how many agents per buffer
// this bank will own. Weights are uninitialized — caller must load before use.
static void weight_bank_create_for_pufferl(WeightBank* bank, PuffeRL* pufferl,
        int slice_size, int hidden_size, int num_layers) {
    int num_buffers = pufferl->hypers.num_buffers;

    // Rebuild arch-varying Policy from env metadata already on pufferl.
    int input_size = pufferl->env.obs.shape[1];
    int num_action_heads = pufferl->env.actions.shape[1];
    int act_n = 0;
    for (int i = 0; i < num_action_heads; i++) act_n += puf_act_sizes[i];
    int decoder_output_size = pufferl->is_continuous ? num_action_heads : act_n;
    bank->policy = build_policy(pufferl->env_name, input_size, hidden_size,
        num_layers, decoder_output_size, act_n, pufferl->is_continuous, pufferl->hypers.horizon);
    bank->hidden_size = hidden_size;
    bank->num_layers = num_layers;

    Allocator* params = &bank->params_alloc;
    Allocator* acts = &bank->acts_alloc;

    bank->slice_size = slice_size;
    bank->weights = policy_weights_create(&bank->policy, params);
    bank->buffer_activations = (PolicyActivations*)calloc(num_buffers, sizeof(PolicyActivations));
    bank->buffer_states = (PrecisionTensor*)calloc(num_buffers, sizeof(PrecisionTensor));
    for (int i = 0; i < num_buffers; i++) {
        bank->buffer_activations[i] = policy_reg_rollout(&bank->policy, bank->weights, acts, slice_size);
        bank->buffer_states[i] = {.shape = {num_layers, slice_size, hidden_size}};
        alloc_register(acts, &bank->buffer_states[i]);
    }

    alloc_create(params);
    alloc_create(acts);

    bank->param_puf = {.data = (precision_t*)params->mem, .shape = {params->total_elems}};
    if (USE_BF16) {
        bank->master_weights = {.shape = {params->total_elems}};
        cudaMalloc(&bank->master_weights.data, params->total_elems * sizeof(float));
    } else {
        bank->master_weights = {.data = (float*)bank->param_puf.data, .shape = {params->total_elems}};
    }
}

// Mirror of weight_bank_create_for_pufferl. Frees the bank's weights, per-buffer
// activations, allocators, and master_weights (BF16 only). Does not free the
// WeightBank struct itself — caller owns that.
static void weight_bank_destroy(WeightBank* bank, PuffeRL* pufferl) {
    int num_buffers = pufferl->hypers.num_buffers;
    policy_weights_free(&bank->policy, &bank->weights);
    if (bank->buffer_activations != NULL) {
        for (int i = 0; i < num_buffers; i++) {
            policy_activations_free(&bank->policy, bank->buffer_activations[i]);
        }
        free(bank->buffer_activations);
    }
    free(bank->buffer_states);
    alloc_free(&bank->params_alloc);
    alloc_free(&bank->acts_alloc);
    if (USE_BF16 && bank->master_weights.data != NULL) {
        cudaFree(bank->master_weights.data);
    }
}

// Append a fresh frozen bank with the given per-buffer slice size. Must be
// called before cudagraph capture.
int pufferl_add_frozen_bank(PuffeRL* pufferl, int slice_size,
        int hidden_size, int num_layers) {
    int idx = pufferl->num_frozen_banks;
    pufferl->frozen_banks = (WeightBank*)realloc(
        pufferl->frozen_banks, (idx + 1) * sizeof(WeightBank));
    memset(&pufferl->frozen_banks[idx], 0, sizeof(WeightBank));
    weight_bank_create_for_pufferl(&pufferl->frozen_banks[idx], pufferl,
        slice_size, hidden_size, num_layers);
    pufferl->num_frozen_banks++;
    return idx;
}

// Load a frozen bank's weights from a file (same format as save_weights — flat fp32).
// Safe to call between rollouts (in-place cudaMemcpy; cudagraphs hold the pointer,
// not a copy of the data).
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

static bool create_allocator_or_report(const char* name, Allocator* alloc) {
    cudaError_t err = alloc_create(alloc);
    if (err == cudaSuccess) {
        return true;
    }

    fprintf(stderr, "create_pufferl: alloc_create(%s) failed for %ld bytes: %s\n",
        name, alloc->total_bytes, cudaGetErrorString(err));
    return false;
}

PuffeRL* create_pufferl_impl(HypersT& hypers, Dict* vec_kwargs,
        Dict* env_kwargs, ncclUniqueId* nccl_id) {
    PuffeRL* pufferl = new PuffeRL();
    pufferl->hypers = hypers;
    pufferl->nccl_comm = nullptr;
    pufferl->default_stream = 0;
    snprintf(pufferl->env_name, sizeof(pufferl->env_name), "%s", PUFFER_STRINGIFY(ENV_NAME));

    cudaSetDevice(hypers.gpu_id);

    // Multi-GPU: initialize NCCL
    if (hypers.world_size > 1) {
        ncclCommInitRank(&pufferl->nccl_comm, hypers.world_size, *nccl_id, hypers.rank);
        printf("Rank %d/%d: NCCL initialized\n", hypers.rank, hypers.world_size);
    }

    ulong seed = hypers.seed + hypers.rank;
    pufferl->seed = seed;

    // Load environment first to get input_size and action info from env
    // Create environments and set up action sizes
    VecEnv* vec = create_environments(vec_kwargs, env_kwargs, pufferl->env);
    pufferl->vec = vec;
    pufferl->bank_layout = vec->bank_layout;

    // Sanity check action space
    int num_action_heads = pufferl->env.actions.shape[1];
    int act_n = 0;
    int num_continuous = 0;
    int num_discrete = 0;
    for (int i = 0; i < num_action_heads; i++) {
        int val = puf_act_sizes[i];
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
    for (int i = 0; i < NUM_TRAIN_EVENTS; i++) {
        cudaEventCreate(&pufferl->profile.events[i]);
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

    // Create and allocate params
    Allocator* params = &pufferl->params_alloc;
    Allocator* acts = &pufferl->activations_alloc;
    Allocator* grads = &pufferl->grads_alloc;

    // Buffers for weights, grads, and activations
    pufferl->weights = policy_weights_create(&pufferl->policy, params);
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
    register_rollout_buffers(pufferl->rollouts,
        acts, horizon, total_agents, input_size, num_action_heads, mask_size);
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

    muon_init(&pufferl->muon, params, hypers.lr, hypers.beta1, hypers.eps, 0.0, acts);
    pufferl->muon.nccl_comm = pufferl->nccl_comm;
    pufferl->muon.world_size = hypers.world_size;

    // All buffers allocated here
    if (!create_allocator_or_report("params", params)) {
        return nullptr;
    }
    if (!create_allocator_or_report("grads", grads)) {
        return nullptr;
    }
    if (!create_allocator_or_report("acts", acts)) {
        return nullptr;
    }

    pufferl->grad_puf = {.data = (precision_t*)grads->mem, .shape = {grads->total_elems}};
    pufferl->param_puf = {.data = (precision_t*)params->mem, .shape = {params->total_elems}};

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

    // Per-buffer persistent RNG states
    int agents_per_buf = total_agents / num_buffers;
    pufferl->rng_states = (curandStatePhilox4_32_10_t**)calloc(num_buffers, sizeof(curandStatePhilox4_32_10_t*));
    for (int i = 0; i < num_buffers; i++) {
        cudaMalloc(&pufferl->rng_states[i], agents_per_buf * sizeof(curandStatePhilox4_32_10_t));
        rng_init<<<grid_size(agents_per_buf), BLOCK_SIZE>>>(
            pufferl->rng_states[i], pufferl->seed + i, agents_per_buf);
    }

    // Post-create initialization
    cudaMemcpy(pufferl->act_sizes_puf.data, puf_act_sizes, num_action_heads * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(pufferl->losses_puf.data, 0, NUM_LOSSES * sizeof(float));
    float one = 1.0f;
    cudaMemcpy(pufferl->ppo_bufs_puf.grad_loss.data, &one, sizeof(float), cudaMemcpyHostToDevice);
    muon_post_create(&pufferl->muon);

    // Set up frozen banks from the layout computed by VecEnv.
    // Must happen before cudagraph capture so graph launch counts are fixed.
    int num_frozen = vec->num_banks - 1;
    int frozen_hidden = hidden_size;
    int frozen_layers = num_layers;
    for (int i = 0; i < vec_kwargs->size; i++) {
        if (strcmp(vec_kwargs->items[i].key, "frozen_bank_hidden_size") == 0) {
            int value = (int)vec_kwargs->items[i].value;
            if (value > 0) {
                frozen_hidden = value;
            }
        }
        if (strcmp(vec_kwargs->items[i].key, "frozen_bank_num_layers") == 0) {
            int value = (int)vec_kwargs->items[i].value;
            if (value > 0) {
                frozen_layers = value;
            }
        }
    }
    if (num_frozen > 0) {
        for (int b = 0; b < num_frozen; b++) {
            int frozen_size = vec->bank_layout[b + 2] - vec->bank_layout[b + 1];
            if (frozen_size <= 0) {
                fprintf(stderr, "create_pufferl: frozen bank %d has no agents\n", b);
                return nullptr;
            }
            pufferl_add_frozen_bank(pufferl, frozen_size, frozen_hidden, frozen_layers);
        }
    }

    // Cudagraph rolluts and entire training step
    if (hypers.cudagraphs >= 0) {
        pufferl->fused_rollout_cudagraphs = (cudaGraphExec_t*)calloc(horizon*num_buffers, sizeof(cudaGraphExec_t));
        pufferl->train_warmup = 0;

        // Snapshot weights + optimizer state before init-time capture
        long wb_bytes = numel(pufferl->master_weights.shape) * sizeof(float);
        void* saved_weights;
        cudaMalloc(&saved_weights, wb_bytes);
        cudaMemcpy(saved_weights, pufferl->master_weights.data, wb_bytes, cudaMemcpyDeviceToDevice);
        void* saved_momentum;
        cudaMalloc(&saved_momentum, wb_bytes);
        cudaMemcpy(saved_momentum, pufferl->muon.mb_puf.data, wb_bytes, cudaMemcpyDeviceToDevice);

        // Create per-buffer streams before capture so graphs are
        // captured and replayed on the same streams.
        pufferl->streams = (cudaStream_t*)calloc(num_buffers, sizeof(cudaStream_t));
        for (int i = 0; i < num_buffers; i++) {
            cudaStreamCreate(&pufferl->streams[i]);
            vec->streams[i] = pufferl->streams[i];
        }

        cudaStream_t saved_default = pufferl->default_stream;
        cudaStream_t saved_tl = tl_stream;
        cudaStream_t warmup_stream;
        cudaStreamCreate(&warmup_stream);
        pufferl->default_stream = warmup_stream;

        for (pufferl->epoch = 0; pufferl->epoch <= hypers.cudagraphs; pufferl->epoch++) {
            for (int i = 0; i < num_buffers * horizon; ++i) {
                int buf = i % num_buffers;
                tl_stream = pufferl->streams[buf];
                pufferl_forward(pufferl, buf, i / num_buffers);
                cudaDeviceSynchronize();
            }
        }
        pufferl->rollout_captured = true;

        tl_stream = warmup_stream;
        for (int i = 0; i <= hypers.cudagraphs; i++) {
            train_impl(*pufferl);
        }

        cudaStreamSynchronize(warmup_stream);
        cudaDeviceSynchronize();
        pufferl->default_stream = saved_default;
        tl_stream = saved_tl;
        cudaStreamDestroy(warmup_stream);

        // Restore weights + optimizer state corrupted by warmup/capture
        cudaMemcpy(pufferl->master_weights.data, saved_weights, wb_bytes, cudaMemcpyDeviceToDevice);
        cudaFree(saved_weights);
        cudaMemcpy(pufferl->muon.mb_puf.data, saved_momentum, wb_bytes, cudaMemcpyDeviceToDevice);
        cudaFree(saved_momentum);
        if (USE_BF16) {
            int n = numel(pufferl->param_puf.shape);
            cast<<<grid_size(n), BLOCK_SIZE, 0, pufferl->default_stream>>>(
                pufferl->param_puf.data, pufferl->master_weights.data, n);
        }

        // Re-init RNG states corrupted by warmup
        for (int i = 0; i < num_buffers; i++) {
            rng_init<<<grid_size(agents_per_buf), BLOCK_SIZE>>>(
                pufferl->rng_states[i], pufferl->seed + i, agents_per_buf);
        }
        cudaDeviceSynchronize();

        pufferl->epoch = 0;
        pufferl->global_step = 0;
    }

    // Create per-buffer streams if not already created by cudagraph path
    if (!pufferl->streams) {
        pufferl->streams = (cudaStream_t*)calloc(num_buffers, sizeof(cudaStream_t));
        for (int i = 0; i < num_buffers; i++) {
            cudaStreamCreate(&pufferl->streams[i]);
            vec->streams[i] = pufferl->streams[i];
        }
    }

    vec_create_threads(vec, hypers.num_threads, horizon, pufferl);
    vec_reset(vec);

    if (hypers.profile) {
        cudaDeviceSynchronize();
        cudaProfilerStart();
    }

    double now = wall_clock();
    pufferl->start_time = now;
    pufferl->last_log_time = now;
    pufferl->last_log_step = 0;

    return pufferl;
}

void close_impl(PuffeRL& pufferl) {
    cudaDeviceSynchronize();
    if (pufferl.hypers.profile) {
        cudaProfilerStop();
    }

    cudaGraphExecDestroy(pufferl.train_cudagraph);
    for (int i = 0; i < pufferl.hypers.horizon * pufferl.hypers.num_buffers; i++) {
        cudaGraphExecDestroy(pufferl.fused_rollout_cudagraphs[i]);
    }

    policy_weights_free(&pufferl.policy, &pufferl.weights);
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
    alloc_free(&pufferl.grads_alloc);
    alloc_free(&pufferl.activations_alloc);

    for (int i = 0; i < pufferl.hypers.num_buffers; i++) {
        cudaStreamDestroy(pufferl.streams[i]);
    }
    for (int i = 0; i < NUM_TRAIN_EVENTS; i++) {
        cudaEventDestroy(pufferl.profile.events[i]);
    }
    nvmlShutdown();

    vec_close(pufferl.vec);

    free(pufferl.buffer_states);
    free(pufferl.buffer_activations);
    free(pufferl.fused_rollout_cudagraphs);
    free(pufferl.streams);

    for (int b = 0; b < pufferl.num_frozen_banks; b++) {
        weight_bank_destroy(&pufferl.frozen_banks[b], &pufferl);
    }
    free(pufferl.frozen_banks);

    if (pufferl.nccl_comm != nullptr) {
        ncclCommDestroy(pufferl.nccl_comm);
    }
}

#ifdef PUFFERLIB_BUILD_MAIN

#include <errno.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "protein.cu"
#include "dashboard.h"

static void log_util(PuffeRL* p, Dict* out) {
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

static void puf_log_env(Dict* out, Dict* env_out) {
    for (int i = 0; i < env_out->size; i++) {
        char key[256];
        snprintf(key, sizeof(key), "env/%s", env_out->items[i].key);
        dict_set(out, key, env_out->items[i].value);
    }
}

static void trainer_log(PuffeRL* p, Dict* out) {
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
    log_environments_impl(*p, &env_out);
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
    for (int i = 0; i < NUM_PROF; i++) {
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

static void trainer_eval_log(PuffeRL* p, Dict* out) {
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

static DictItem* puf_log_find(Dict* dict, const char* key) {
    for (int i = 0; i < dict->size; i++) {
        if (strcmp(dict->items[i].key, key) == 0) {
            return &dict->items[i];
        }
    }
    return NULL;
}

static void puf_log_update(Dict* dst, Dict* src) {
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

static void puf_log_history_add(PufLogHistory* history, Dict* log) {
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

static void puf_log_history_free(PufLogHistory* history) {
    for (int i = 0; i < history->size; i++) {
        dict_clear(&history->items[i]);
    }
    free(history->items);
    memset(history, 0, sizeof(*history));
}

static void puf_log_collect_keys(PufLogHistory* history, Dict* keys) {
    for (int i = 0; i < history->size; i++) {
        Dict* log = &history->items[i];
        for (int j = 0; j < log->size; j++) {
            if (!puf_log_find(keys, log->items[j].key)) {
                dict_set(keys, log->items[j].key, 0);
            }
        }
    }
}

static double puf_log_reduce(double* vals, int n, double fallback) {
    if (n == 0) {
        return fallback;
    }

    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += vals[i];
    }
    return sum / n;
}

static void puf_log_write_metric(FILE* fp, const char* key, double* values, int n) {
    fprintf(fp, "%s = ", key);
    for (int i = 0; i < n; i++) {
        if (i > 0) {
            fputc(',', fp);
        }
        fprintf(fp, "%.17g", values[i]);
    }
    fputc('\n', fp);
}

static void puf_log_write(const char* path, Config* cfg, PufLogHistory* history) {
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
    puf_ini_write(fp, &cfg->ini);
    fprintf(fp, "\n[metrics]\n");

    int downsample = (int)puf_config_get(cfg, "sweep", "downsample");
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

typedef struct {
    int rank;
    int world_size;
    int gpu_id;
    int artifact_owner;
    ncclUniqueId* nccl_id;
} TrainContext;

static PuffeRL* create_trainer(Config* cfg, TrainContext* ctx) {
    HypersT hypers = puf_config_to_hypers(cfg,
        ctx->rank, ctx->world_size, ctx->gpu_id);
    Dict vec = {0};
    dict_copy(&vec, puf_ini_section(&cfg->ini, "vec", 0));
    PuffeRL* pufferl = create_pufferl_impl(hypers, &vec, &cfg->env, ctx->nccl_id);
    dict_clear(&vec);
    if (!pufferl) {
        fprintf(stderr, "create_pufferl_impl failed\n");
        exit(1);
    }
    return pufferl;
}

static void rollouts(PuffeRL* p) {
    if (p->hypers.reset_state) {
        for (int i = 0; i < p->hypers.num_buffers; i++) {
            puf_zero(&p->buffer_states[i], p->default_stream);
        }
        for (int b = 0; b < p->num_frozen_banks; b++) {
            for (int i = 0; i < p->hypers.num_buffers; i++) {
                puf_zero(&p->frozen_banks[b].buffer_states[i], p->default_stream);
            }
        }
    }

    double t0 = wall_clock();
    vec_step(p->vec);
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
    p->profile.accum[PROF_EVAL_GPU] += eval_prof[VEC_GPU] / p->vec->buffers;
    p->profile.accum[PROF_EVAL_ENV] += eval_prof[VEC_ENV_STEP] / p->vec->buffers;
    p->global_step += p->hypers.horizon * p->hypers.total_agents;
}

static void close_trainer(PuffeRL* p) {
    close_impl(*p);
    delete p;
}

typedef struct {
    float score;
    float draw;
    int games;
} EvalResult;

#define EVAL_RENDER 0
#define EVAL_SCORE 1
#define EVAL_MATCH 2

static EvalResult run_eval(Config* cfg, TrainContext* ctx, int mode, int verbose);

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

static void selfplay_add_pool(Selfplay* sp, const char* path) {
    for (int i = 0; i < sp->pool_size; i++) {
        if (strcmp(sp->pool[i], path) == 0) {
            return;
        }
    }
    if (sp->pool_size >= SELFPLAY_MAX_POOL) {
        fprintf(stderr, "selfplay pool exceeds SELFPLAY_MAX_POOL\n");
        exit(1);
    }
    snprintf(sp->pool[sp->pool_size++], sizeof(sp->pool[0]), "%s", path);
}

static void selfplay_evict(Selfplay* sp) {
    if (sp->pool_size <= sp->max_size) {
        return;
    }

    int start = sp->pool_size - sp->max_size;
    memmove(sp->pool, sp->pool + start, (size_t)sp->max_size * sizeof(sp->pool[0]));
    sp->pool_size = sp->max_size;
}

static void selfplay_add_checkpoint(Selfplay* sp, const char* path) {
    while (access(path, R_OK) != 0) {
        usleep(50000);
    }
    selfplay_add_pool(sp, path);
    selfplay_evict(sp);
}

static const char* selfplay_sample(Selfplay* sp) {
    if (sp->pool_size == 0) {
        fprintf(stderr, "selfplay opponent pool is empty\n");
        exit(1);
    }
    int idx = (int)(rand_r(&sp->rng) % (unsigned int)sp->pool_size);
    return sp->pool[idx];
}

static int selfplay_count_aligned(PuffeRL* p, int tag) {
    Env* envs = (Env*)p->vec->envs;
    int count = 0;
    for (int i = 0; i < p->vec->size; i++) {
        if (envs[i].tag == tag && envs[i].boundary_reached) {
            count++;
        }
    }
    return count;
}

static void selfplay_clear_aligned(PuffeRL* p, int tag) {
    Env* envs = (Env*)p->vec->envs;
    for (int i = 0; i < p->vec->size; i++) {
        if (envs[i].tag == tag) {
            envs[i].boundary_reached = 0;
        }
    }
}

static void selfplay_init(Selfplay* sp, Config* cfg, PuffeRL* p,
        const char* initial_checkpoint) {
    memset(sp, 0, sizeof(*sp));
    sp->num_banks = p->num_frozen_banks;
    if (sp->num_banks <= 0 || sp->num_banks > SELFPLAY_MAX_BANKS) {
        fprintf(stderr, "selfplay requires 1..%d frozen banks\n", SELFPLAY_MAX_BANKS);
        exit(1);
    }
    sp->max_size = (int)puf_config_get(cfg, "selfplay", "max_size");
    sp->opp_timeout_steps = (long)puf_config_get(cfg, "selfplay", "opp_timeout_steps");
    sp->rng = (unsigned int)puf_config_get(cfg, "selfplay", "seed") + (unsigned int)p->hypers.rank;
    long current_step = p->global_step * p->hypers.world_size;

    Env* envs = (Env*)p->vec->envs;
    for (int i = 0; i < p->vec->size; i++) {
        int tag = envs[i].tag;
        if (tag > 0 && tag <= sp->num_banks) {
            sp->banks[tag - 1].num_envs++;
        }
    }

    selfplay_add_checkpoint(sp, initial_checkpoint);
    for (int b = 0; b < sp->num_banks; b++) {
        const char* path = selfplay_sample(sp);
        pufferl_load_frozen_bank(p, b, path);
        sp->banks[b].opp_started_step = current_step;
    }
}

static void selfplay_step(Selfplay* sp, PuffeRL* p, Dict* log) {
    long current_step = p->global_step * p->hypers.world_size;
    for (int b = 0; b < sp->num_banks; b++) {
        SelfplayBank* bank = &sp->banks[b];
        int timed_out = sp->opp_timeout_steps > 0 &&
            current_step - bank->opp_started_step >= sp->opp_timeout_steps;
        int tag = b + 1;
        if (bank->pending_path[0]) {
            if (selfplay_count_aligned(p, tag) >= bank->num_envs) {
                pufferl_load_frozen_bank(p, b, bank->pending_path);
                selfplay_clear_aligned(p, tag);
                bank->pending_path[0] = 0;
                bank->opp_started_step = current_step;
            }
        } else if (timed_out) {
            const char* path = selfplay_sample(sp);
            snprintf(bank->pending_path, sizeof(bank->pending_path), "%s", path);
            selfplay_clear_aligned(p, tag);
        }
    }
    dict_set(log, "pool/size", sp->pool_size);
    dict_set(log, "pool/num_banks", sp->num_banks);
}

#include "league.h"
#include "sweep.h"

static float log_value(Dict* log, const char* key, float fallback) {
    for (int i = 0; i < log->size; i++) {
        if (strcmp(log->items[i].key, key) == 0) {
            return (float)log->items[i].value;
        }
    }
    return fallback;
}

static void train_result_fill(TrainResult* result, PufLogHistory* history,
        Dict* last_log, Config* cfg, const char* target_key) {
    result->score = (float)puf_log_get_or(last_log, target_key, 0);
    result->cost = (float)puf_log_get_or(last_log, "uptime", 0);
    result->steps = (float)puf_log_get_or(last_log, "agent_steps", 0);

    int points = puf_config_int(cfg, "sweep", "downsample");
    if (points < 1) {
        points = 1;
    }
    if (points > TRAIN_RESULT_MAX_POINTS) {
        points = TRAIN_RESULT_MAX_POINTS;
    }
    result->points = points;

    if (history->size == 0 || points == 1) {
        result->scores[0] = result->score;
        result->costs[0] = result->cost;
        result->step_points[0] = result->steps;
        return;
    }

    float final_steps = log_value(&history->items[history->size - 1],
        "agent_steps", result->steps);
    int cursor = 0;
    for (int p = 0; p < points; p++) {
        float target = final_steps * (float)p / (float)(points - 1);
        while (cursor + 1 < history->size &&
                log_value(&history->items[cursor], "agent_steps", 0) < target) {
            cursor++;
        }
        Dict* log = &history->items[cursor];
        result->scores[p] = log_value(log, target_key, result->score);
        result->costs[p] = log_value(log, "uptime", result->cost);
        result->step_points[p] = log_value(log, "agent_steps", target);
    }
    result->scores[points - 1] = result->score;
    result->costs[points - 1] = result->cost;
    result->step_points[points - 1] = result->steps;
}

static EvalResult run_eval(Config* cfg, TrainContext* ctx, int mode, int verbose) {
    int render = mode == EVAL_RENDER;
    int match = mode == EVAL_MATCH;
    EvalResult result = {0};
    long num_games = puf_config_long(cfg, "base", "num_games");
    if (!num_games) {
        num_games = puf_config_long(cfg, "base", "eval_episodes");
    }
    long burnin_games = puf_config_long(cfg, "base", "burnin_games");
    if (!render && (num_games <= 0 || burnin_games < 0)) {
        fprintf(stderr, "eval requires positive num_games and nonnegative burnin_games\n");
        exit(1);
    }
    if (!render) {
        long eval_agents = puf_config_long(cfg, "base", "eval_agents");
        if (!eval_agents && match) {
            eval_agents = puf_config_long(cfg, "sweep", "league_match_eval_agents");
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

        char buf[64];
        snprintf(buf, sizeof(buf), "%ld", eval_agents);
        puf_config_put(cfg, "vec.num_buffers", "2");
        puf_config_put(cfg, "vec.total_agents", buf);
    }
    if (match) {
        puf_config_put(cfg, "vec.num_frozen_banks", "1");
        puf_config_put(cfg, "vec.frozen_bank_pct", "1");
        puf_config_put(cfg, "selfplay.enabled", "0");
        puf_config_put(cfg, "env.dr", "0");
        puf_config_put(cfg, "env.num_agents", "2");
        puf_config_put(cfg, "env.num_bots", "0");
    }

    puf_config_put(cfg, "base.reset_state", "0");
    puf_config_put(cfg, "train.horizon", "1");

    PuffeRL* pufferl = create_trainer(cfg, ctx);
    if (match) {
        char a_path_buf[4096];
        char b_path_buf[4096];
        const char* a_path = puf_checkpoint_path_key(cfg,
            "load_model_path", a_path_buf, sizeof(a_path_buf));
        const char* b_path = puf_checkpoint_path_key(cfg,
            "load_enemy_model_path", b_path_buf, sizeof(b_path_buf));
        if (!a_path || !b_path) {
            fprintf(stderr, "match requires base.load_model_path and base.load_enemy_model_path\n");
            exit(1);
        }
        puf_load_weights(pufferl, a_path);
        pufferl_load_frozen_bank(pufferl, 0, b_path);
    } else {
        puf_load_primary_if_configured(pufferl, cfg);
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
            puf_dashboard_print(cfg, pufferl, &log, 0);
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
    close_trainer(pufferl);
    return result;
}

static void train_checkpoint_path(PuffeRL* p, const char* dir,
        char* out, size_t out_size) {
    snprintf(out, out_size, "%s/%016ld.bin", dir, p->global_step);
}

TrainResult run_train(Config* cfg, TrainContext* ctx) {
    int use_selfplay = puf_config_int(cfg, "selfplay", "enabled");
    if (!use_selfplay) {
        puf_config_put(cfg, "vec.num_frozen_banks", "0");
        puf_config_put(cfg, "vec.frozen_bank_pct", "0");
    }

    char run_id[64];
    const char* configured_run_id = puf_config_str(cfg, "base", "run_id");
    if (!configured_run_id[0] || strcmp(configured_run_id, "None") == 0) {
        snprintf(run_id, sizeof(run_id), "%ld", (long)(1000.0 * wall_clock()));
        puf_config_put(cfg, "base.run_id", run_id);
    } else {
        snprintf(run_id, sizeof(run_id), "%s", configured_run_id);
    }

    char checkpoint_dir[2048];
    char log_dir[2048];
    snprintf(checkpoint_dir, sizeof(checkpoint_dir), "%s/%s/%s",
        puf_config_str(cfg, "base", "checkpoint_dir"),
        puf_config_str(cfg, "base", "env_name"), run_id);
    snprintf(log_dir, sizeof(log_dir), "%s/%s",
        puf_config_str(cfg, "base", "log_dir"),
        puf_config_str(cfg, "base", "env_name"));
    if (ctx->artifact_owner) {
        mkdir_p(checkpoint_dir);
        mkdir_p(log_dir);
    }

    PuffeRL* pufferl = create_trainer(cfg, ctx);
    char initial_checkpoint[4096] = {0};
    if (use_selfplay) {
        train_checkpoint_path(pufferl, checkpoint_dir,
            initial_checkpoint, sizeof(initial_checkpoint));
        if (ctx->artifact_owner) {
            puf_save_weights(pufferl, initial_checkpoint);
        }
    }

    Selfplay* selfplay = NULL;
    if (use_selfplay) {
        selfplay = (Selfplay*)calloc(1, sizeof(Selfplay));
        selfplay_init(selfplay, cfg, pufferl, initial_checkpoint);
    }

    long total_timesteps = puf_config_long(cfg, "train", "total_timesteps");
    long batch_size = (long)puf_config_int(cfg, "vec", "total_agents") *
        (long)puf_config_int(cfg, "train", "horizon");
    long local_timesteps = total_timesteps / ctx->world_size;
    long train_epochs = local_timesteps / batch_size;
    long eval_epochs = train_epochs / 2;
    long checkpoint_interval = puf_config_long(cfg, "base", "checkpoint_interval");
    long eval_episodes = puf_config_long(cfg, "base", "eval_episodes");
    const char* target_key = "env/score";
    Dict last_log = {0};
    PufLogHistory log_history = {0};
    TrainResult result = {0};

    for (long epoch = 0; epoch < train_epochs + eval_epochs; epoch++) {
        rollouts(pufferl);
        if (epoch < train_epochs) {
            train_impl(*pufferl);
        }

        bool is_final = epoch == train_epochs - 1;
        bool interval_save = checkpoint_interval > 0 &&
            (epoch + 1) % checkpoint_interval == 0;
        bool should_save = epoch < train_epochs && (interval_save || is_final);
        char saved_checkpoint[4096] = {0};
        if (should_save) {
            train_checkpoint_path(pufferl, checkpoint_dir,
                saved_checkpoint, sizeof(saved_checkpoint));
            if (ctx->artifact_owner) {
                puf_save_weights(pufferl, saved_checkpoint);
                snprintf(result.checkpoint_path, sizeof(result.checkpoint_path),
                    "%s", saved_checkpoint);
            }
        }
        if (selfplay && saved_checkpoint[0]) {
            selfplay_add_checkpoint(selfplay, saved_checkpoint);
        }

        if (wall_clock() < pufferl->last_log_time + 0.6 && epoch < train_epochs - 1) {
            continue;
        }

        Dict new_log = {0};
        if (epoch >= train_epochs) {
            trainer_eval_log(pufferl, &new_log);
        } else {
            trainer_log(pufferl, &new_log);
        }
        puf_log_update(&last_log, &new_log);
        if (selfplay && epoch < train_epochs) {
            selfplay_step(selfplay, pufferl, &last_log);
        }
        if (ctx->artifact_owner) {
            puf_dashboard_print(cfg, pufferl, &last_log, (int)epoch);
        }

        if (puf_log_get_or(&last_log, target_key, -1) < 0) {
            continue;
        }
        if (epoch < train_epochs) {
            puf_log_history_add(&log_history, &last_log);
        }
        if (epoch >= train_epochs && puf_log_get_or(&last_log, "env/n", 0) > eval_episodes) {
            break;
        }
    }

    train_result_fill(&result, &log_history, &last_log, cfg, target_key);
    if (ctx->artifact_owner) {
        puf_log_history_add(&log_history, &last_log);
        char log_path[4096];
        snprintf(log_path, sizeof(log_path), "%s/%s.ini", log_dir, run_id);
        puf_log_write(log_path, cfg, &log_history);
    }
    puf_log_history_free(&log_history);
    free(selfplay);
    close_trainer(pufferl);
    return result;
}

static int gpu_for_rank(int rank, int world_size) {
    if (rank == 0) {
        return world_size - 1;
    }
    return rank - 1;
}

static void wait_children(pid_t* pids, int num_pids) {
    for (int i = 0; i < num_pids; i++) {
        int status = 0;
        if (waitpid(pids[i], &status, 0) < 0) {
            fprintf(stderr, "waitpid failed for child %d: %s\n", (int)pids[i], strerror(errno));
            exit(1);
        }
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            fprintf(stderr, "worker pid %d failed\n", (int)pids[i]);
            exit(1);
        }
    }
}

TrainResult launch_train(Config* cfg) {
    int world_size = puf_config_int(cfg, "train", "gpus");
    if (world_size < 1) {
        fprintf(stderr, "config error: [train] gpus must be >= 1\n");
        exit(1);
    }
    int gpu_offset = puf_config_int(cfg, "base", "gpu_offset");

    ncclUniqueId nccl_id;
    ncclUniqueId* nccl_ptr = NULL;
    if (world_size > 1) {
        ncclGetUniqueId(&nccl_id);
        nccl_ptr = &nccl_id;
    }

    pid_t* pids = (pid_t*)calloc(world_size > 1 ? world_size - 1 : 1, sizeof(pid_t));
    for (int rank = world_size - 1; rank >= 1; rank--) {
        pid_t pid = fork();
        if (pid < 0) {
            fprintf(stderr, "fork failed: %s\n", strerror(errno));
            exit(1);
        }

        if (pid == 0) {
            if (!freopen("/dev/null", "w", stdout)) {
                fprintf(stderr, "failed to redirect child stdout: %s\n", strerror(errno));
                exit(1);
            }
            TrainContext child = {
                .rank = rank,
                .world_size = world_size,
                .gpu_id = gpu_offset + gpu_for_rank(rank, world_size),
                .artifact_owner = 0,
                .nccl_id = nccl_ptr,
            };
            run_train(cfg, &child);
            puf_config_free(cfg);
            exit(0);
        }

        pids[rank - 1] = pid;
    }

    TrainContext host = {
        .rank = 0,
        .world_size = world_size,
        .gpu_id = gpu_offset + gpu_for_rank(0, world_size),
        .artifact_owner = 1,
        .nccl_id = nccl_ptr,
    };
    TrainResult result = run_train(cfg, &host);
    wait_children(pids, world_size - 1);
    free(pids);
    return result;
}

int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
    if (argc < 3) {
        fprintf(stderr, "usage: %s train|eval|eval_bot|match|sweep ENV [section.key=value ...]\n", argv[0]);
        exit(1);
    }

    const char* mode = argv[1];
    const char* env_name = argv[2];
    Config* cfg = (Config*)calloc(1, sizeof(Config));
    puf_config_load_env(cfg, env_name, argc - 3, argv + 3);
    TrainContext ctx = {
        .rank = 0,
        .world_size = 1,
        .gpu_id = 0,
        .artifact_owner = 1,
        .nccl_id = NULL,
    };

    if (strcmp(mode, "train") == 0) {
        TrainResult result = launch_train(cfg);
        int result_fd = puf_config_int(cfg, "base", "result_fd");
        if (result_fd) {
            int fd = result_fd;
            if (write(fd, &result, sizeof(result)) != sizeof(result)) {
                fprintf(stderr, "failed to write train result\n");
                exit(1);
            }
            close(fd);
        }
    } else if (strcmp(mode, "sweep") == 0) {
        run_sweep(cfg, argv[0]);
    } else if (strcmp(mode, "eval") == 0 || strcmp(mode, "eval_bot") == 0) {
        int bot = strcmp(mode, "eval_bot") == 0;
        if (bot) {
            puf_config_put(cfg, "vec.num_frozen_banks", "0");
            puf_config_put(cfg, "vec.frozen_bank_pct", "0");
            puf_config_put(cfg, "selfplay.enabled", "0");
            puf_config_put(cfg, "env.dr", "0");
            puf_config_put(cfg, "env.num_agents", "1");
            puf_config_put(cfg, "env.num_bots", "1");
        }
        run_eval(cfg, &ctx, bot ? EVAL_SCORE : EVAL_RENDER, 1);
    } else if (strcmp(mode, "match") == 0) {
        run_eval(cfg, &ctx, EVAL_MATCH, 1);
    } else if (strcmp(mode, "league_match_worker") == 0) {
        int gpu_offset = puf_config_int(cfg, "base", "gpu_offset");
        if (gpu_offset) {
            ctx.gpu_id = gpu_offset;
        }
        run_league_match_worker(cfg, &ctx);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        exit(1);
    }

    puf_config_free(cfg);
    free(cfg);
    return 0;
}

#endif
