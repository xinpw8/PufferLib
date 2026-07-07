#define PRIO_WARP_SIZE 32
#define PRIO_FULL_MASK 0xffffffff
#define PRIO_BLOCK_SIZE 256
#define PRIO_NUM_WARPS (PRIO_BLOCK_SIZE / PRIO_WARP_SIZE)

__global__ void compute_prio_adv_reduction(
        const precision_t* __restrict__ advantages,
        float* prio_weights, float prio_alpha, int stride) {
    int row = blockIdx.x;
    int tx = threadIdx.x;
    int offset = row * stride;

    float local_sum = 0.0f;
    for (int t = tx; t < stride; t += blockDim.x) {
        local_sum += fabsf(to_float(advantages[offset + t]));
    }

    for (int s = PRIO_WARP_SIZE / 2; s >= 1; s /= 2) {
        local_sum += __shfl_down_sync(PRIO_FULL_MASK, local_sum, s);
    }
    if (tx == 0) {
        float pw = __powf(local_sum, prio_alpha);
        if (isnan(pw) || isinf(pw)) {
            pw = 0.0f;
        }
        prio_weights[row] = pw;
    }
}

__global__ void compute_prio_normalize(float* prio_weights, int length) {
    __shared__ float shmem[PRIO_NUM_WARPS];
    __shared__ float block_sum;

    int tx = threadIdx.x;
    int lane = tx % PRIO_WARP_SIZE;
    int warp_id = tx / PRIO_WARP_SIZE;
    const float eps = 1e-6f;

    float local_sum = 0.0f;
    for (int t = tx; t < length; t += blockDim.x) {
        local_sum += prio_weights[t];
    }
    for (int s = PRIO_WARP_SIZE / 2; s >= 1; s /= 2) {
        local_sum += __shfl_down_sync(PRIO_FULL_MASK, local_sum, s);
    }
    if (lane == 0) {
        shmem[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < PRIO_NUM_WARPS) ? shmem[lane] : 0.0f;
        for (int s = PRIO_NUM_WARPS / 2; s >= 1; s /= 2) {
            val += __shfl_down_sync(PRIO_FULL_MASK, val, s);
        }
        if (tx == 0) {
            block_sum = val + eps;
        }
    }
    __syncthreads();

    for (int t = tx; t < length; t += blockDim.x) {
        prio_weights[t] = (prio_weights[t] + eps) / block_sum;
    }
}

// mb_prio[i] = pow(total_agents * prio_probs[idx[i]], -anneal_beta)
__global__ void compute_prio_imp_weights(
        const int* __restrict__ indices,
        const float* __restrict__ prio_probs,
        float* mb_prio, int total_agents,
        float anneal_beta, int minibatch_segments) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tx < minibatch_segments) {
        float value = prio_probs[indices[tx]] * (float)total_agents;
        mb_prio[tx] = __powf(value, -anneal_beta);
    }
}

__global__ void build_cdf(
    float* __restrict__ cdf, const float* __restrict__ probs, int B) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float cum = 0.0f;
        for (int i = 0; i < B; i++) {
            cum += probs[i];
            cdf[i] = cum;
        }
    }
}

__global__ void advance_rng_offset(int64_t* __restrict__ offset_ptr, int64_t delta) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *offset_ptr += delta;
    }
}

// Multinomial with replacement (uses cuRAND)
__global__ void multinomial_sample(int* __restrict__ out_idx, const float* __restrict__ cdf,
        int B, int num_samples, uint64_t seed, const int64_t* __restrict__ offset_ptr) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_samples) {
        return;
    }

    uint64_t base_off = (uint64_t)(*offset_ptr);
    curandStatePhilox4_32_10_t rng_state;
    curand_init(seed, base_off + tid, 0, &rng_state);
    float u = curand_uniform(&rng_state);

    int lo = 0;
    int hi = B - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cdf[mid] < u) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    out_idx[tid] = lo;
}

// Prioritize high absolute advantage trajectories. This is a form of implicit
// curriculum learning; sweep-found alpha/beta values decide whether it matters.
void prio_replay_cuda(PrecisionTensor& advantages, float prio_alpha,
        int minibatch_segments, int total_agents, float anneal_beta,
        PrioBuffers& bufs, ulong seed, long* offset_ptr, cudaStream_t stream) {
    int B = advantages.shape[0];
    int T = advantages.shape[1];
    compute_prio_adv_reduction<<<B, PRIO_WARP_SIZE, 0, stream>>>(
        advantages.data, bufs.prio_probs.data, prio_alpha, T);
    compute_prio_normalize<<<1, PRIO_BLOCK_SIZE, 0, stream>>>(
        bufs.prio_probs.data, B);
    build_cdf<<<1, 1, 0, stream>>>(bufs.cdf.data, bufs.prio_probs.data, B);
    int threads = 256;
    int blocks = (minibatch_segments + threads - 1) / threads;
    multinomial_sample<<<blocks, threads, 0, stream>>>(
        bufs.idx.data, bufs.cdf.data, B, minibatch_segments, seed, offset_ptr);
    advance_rng_offset<<<1, 1, 0, stream>>>(offset_ptr, (int64_t)minibatch_segments);

    int p3_blocks = (minibatch_segments + PRIO_BLOCK_SIZE - 1) / PRIO_BLOCK_SIZE;
    compute_prio_imp_weights<<<p3_blocks, PRIO_BLOCK_SIZE, 0, stream>>>(
        bufs.idx.data, bufs.prio_probs.data,
        bufs.mb_prio.data, total_agents, anneal_beta, minibatch_segments);
}

// Experience the puffer advantage! Generalized advantage estimation + V-Trace
// importance sampling correction in a single streamlined operation
__device__ void puff_advantage_row_scalar(
        const precision_t* values, const precision_t* rewards, const precision_t* dones,
        const precision_t* importance, precision_t* advantages, float gamma, float lambda,
        float rho_clip, float c_clip, int horizon) {
    float lastpufferlam = 0;
    for (int t = horizon - 2; t >= 0; t--) {
        int t_next = t + 1;
        float nextnonterminal = 1.0f - to_float(dones[t_next]);
        float imp = to_float(importance[t]);
        float rho_t = fminf(imp, rho_clip);
        float c_t = fminf(imp, c_clip);
        float r_nxt = to_float(rewards[t_next]);
        float v = to_float(values[t]);
        float v_nxt = to_float(values[t_next]);
        float delta = rho_t*r_nxt + gamma*v_nxt*nextnonterminal - v;
        lastpufferlam = delta + gamma*lambda*c_t*lastpufferlam*nextnonterminal;
        advantages[t] = from_float(lastpufferlam);
    }
}

// These loading fns just optimize bandwidth for advantage since we call it on all
// the data every minibatch. This should change in 5.0
__device__ __forceinline__ void adv_vec_load(const float* ptr, float* out) {
    float4 v = *reinterpret_cast<const float4*>(ptr);
    out[0] = v.x;
    out[1] = v.y;
    out[2] = v.z;
    out[3] = v.w;
}

__device__ __forceinline__ void adv_vec_load(const __nv_bfloat16* ptr, float* out) {
    uint4 raw = *reinterpret_cast<const uint4*>(ptr);
    const __nv_bfloat16* bf = reinterpret_cast<const __nv_bfloat16*>(&raw);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[i] = __bfloat162float(bf[i]);
    }
}

// Store N floats as precision_t via 128-bit writes (float4 for f32, uint4 for bf16)
__device__ __forceinline__ void adv_vec_store(float* ptr, const float* vals) {
    *reinterpret_cast<float4*>(ptr) = make_float4(vals[0], vals[1], vals[2], vals[3]);
}

__device__ __forceinline__ void adv_vec_store(__nv_bfloat16* ptr, const float* vals) {
    // N=8 for bf16: all 8 elements fit in one uint4 (128 bits)
    __nv_bfloat16 tmp[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        tmp[i] = __float2bfloat16(vals[i]);
    }
    *reinterpret_cast<uint4*>(ptr) = *reinterpret_cast<const uint4*>(tmp);
}

__device__ __forceinline__ void puff_advantage_row_vec(
        const precision_t* values, const precision_t* rewards, const precision_t* dones,
        const precision_t* importance, precision_t* advantages, float gamma, float lambda,
        float rho_clip, float c_clip, int horizon) {
    constexpr int N = 16 / sizeof(precision_t);

    float lastpufferlam = 0.0f;
    int num_chunks = horizon / N;

    float next_value = to_float(values[horizon - 1]);
    float next_done = to_float(dones[horizon - 1]);
    float next_reward = to_float(rewards[horizon - 1]);

    for (int chunk = num_chunks - 1; chunk >= 0; chunk--) {
        int base = chunk * N;

        float v[N];
        float r[N];
        float d[N];
        float imp[N];
        adv_vec_load(values + base, v);
        adv_vec_load(rewards + base, r);
        adv_vec_load(dones + base, d);
        adv_vec_load(importance + base, imp);

        float adv[N] = {0};
        int start_idx = (chunk == num_chunks - 1) ? (N - 2) : (N - 1);

        #pragma unroll
        for (int i = start_idx; i >= 0; i--) {
            float nextnonterminal = 1.0f - next_done;
            float rho_t = fminf(imp[i], rho_clip);
            float c_t = fminf(imp[i], c_clip);
            float delta = rho_t * (next_reward + gamma * next_value * nextnonterminal - v[i]);
            lastpufferlam = delta + gamma * lambda * c_t * lastpufferlam * nextnonterminal;
            adv[i] = lastpufferlam;
            next_value = v[i];
            next_done = d[i];
            next_reward = r[i];
        }

        adv_vec_store(advantages + base, adv);
    }
}

__global__ void puff_advantage(const precision_t* values, const precision_t* rewards,
        const precision_t* dones, const precision_t* importance, precision_t* advantages, float gamma,
        float lambda, float rho_clip, float c_clip, int num_steps, int horizon) {
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= num_steps) {
        return;
    }
    int offset = row*horizon;
    puff_advantage_row_vec(values + offset, rewards + offset, dones + offset,
        importance + offset, advantages + offset, gamma, lambda, rho_clip, c_clip, horizon);
}

__global__ void puff_advantage_scalar(const precision_t* values, const precision_t* rewards,
        const precision_t* dones, const precision_t* importance, precision_t* advantages, float gamma,
        float lambda, float rho_clip, float c_clip, int num_steps, int horizon) {
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= num_steps) {
        return;
    }
    int offset = row*horizon;
    puff_advantage_row_scalar(values + offset, rewards + offset, dones + offset,
        importance + offset, advantages + offset, gamma, lambda, rho_clip, c_clip, horizon);
}

void puff_advantage_cuda(PrecisionTensor& values, PrecisionTensor& rewards,
        PrecisionTensor& dones, PrecisionTensor& importance, PrecisionTensor& advantages,
        float gamma, float lambda, float rho_clip, float c_clip, cudaStream_t stream) {
    int num_steps = values.shape[0];
    int horizon = values.shape[1];
    int blocks = grid_size(num_steps);
    constexpr int N = 16 / sizeof(precision_t);
    auto kernel = (horizon % N == 0) ? puff_advantage : puff_advantage_scalar;
    kernel<<<blocks, 256, 0, stream>>>(
        values.data, rewards.data, dones.data, importance.data,
        advantages.data, gamma, lambda, rho_clip, c_clip, num_steps, horizon);
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

void zero_frozen_advantages_cuda(PrecisionTensor& advantages,
        int agents_per_buffer, int primary_per_buffer, cudaStream_t stream) {
    int total_rows = advantages.shape[0];
    int horizon = advantages.shape[1];
    int total = total_rows * horizon;
    zero_frozen_advantages_kernel<<<grid_size(total), BLOCK_SIZE, 0, stream>>>(
        advantages.data, agents_per_buffer, primary_per_buffer, total_rows, horizon);
}

__device__ __forceinline__ void copy_values_adv_returns(
        const precision_t* __restrict__ src_values, precision_t* __restrict__ dst_values,
        const precision_t* __restrict__ src_advantages, precision_t* __restrict__ dst_advantages,
        precision_t* __restrict__ dst_returns,
        int src_row, int dst_row, int horizon) {
    int srh = (int64_t)src_row * horizon;
    int drh = (int64_t)dst_row * horizon;
    const precision_t* s_values = src_values + srh;
    const precision_t* s_adv = src_advantages + srh;
    precision_t* d_values = dst_values + drh;
    precision_t* d_adv = dst_advantages + drh;
    precision_t* d_returns = dst_returns + drh;
    for (int i = threadIdx.x; i < horizon; i += blockDim.x) {
        precision_t val = s_values[i];
        precision_t adv = s_adv[i];
        d_values[i] = val;
        d_adv[i] = adv;
        d_returns[i] = from_float(to_float(val) + to_float(adv));
    }
}

__global__ void select_copy(RolloutBuf rollouts, TrainGraph graph,
        const int* __restrict__ idx, const precision_t* __restrict__ advantages,
        const float* __restrict__ mb_prio) {
    int mb = blockIdx.x;
    int ch = blockIdx.y;
    int src_row = idx[mb];

    int obs_row_bytes = (numel(rollouts.observations.shape)
        / rollouts.observations.shape[0]) * sizeof(precision_t);
    int act_row_bytes = (numel(rollouts.actions.shape)
        / rollouts.actions.shape[0]) * sizeof(precision_t);
    int lp_row_bytes = (numel(rollouts.logprobs.shape)
        / rollouts.logprobs.shape[0]) * sizeof(precision_t);
    int horizon = rollouts.values.shape[1];

    switch (ch) {
    case 0:
        copy_bytes((const char*)rollouts.observations.data,
            (char*)graph.mb_obs.data, src_row, mb, obs_row_bytes);
        break;
    case 1:
        copy_bytes((const char*)rollouts.actions.data,
            (char*)graph.mb_actions.data, src_row, mb, act_row_bytes);
        break;
    case 2:
        copy_bytes((const char*)rollouts.logprobs.data,
            (char*)graph.mb_logprobs.data, src_row, mb, lp_row_bytes);
        break;
    case 3:
        copy_values_adv_returns(rollouts.values.data, graph.mb_values.data,
            advantages, graph.mb_advantages.data,
            graph.mb_returns.data, src_row, mb, horizon);
        break;
    case 4:
        if (threadIdx.x == 0) {
            graph.mb_prio.data[mb] = from_float(mb_prio[mb]);
        }
        break;
    case 5:
        if (graph.mb_action_mask.data != nullptr) {
            int mask_row_bytes = (numel(rollouts.action_mask.shape)
                / rollouts.action_mask.shape[0]) * sizeof(precision_t);
            copy_bytes((const char*)rollouts.action_mask.data,
                (char*)graph.mb_action_mask.data, src_row, mb, mask_row_bytes);
        }
        break;
    }
}
