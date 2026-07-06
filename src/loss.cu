enum LossIdx {
    LOSS_PG = 0, LOSS_VF = 1, LOSS_ENT = 2, LOSS_TOTAL = 3,
    LOSS_OLD_APPROX_KL = 4, LOSS_APPROX_KL = 5, LOSS_CLIPFRAC = 6,
    LOSS_N = 7, NUM_LOSSES = 8,
};

// PPO buffers + args are quite complex. We do the entire
// forward + backwards pass for the full loss function in one kernel
struct PPOGraphArgs {
    precision_t* out_ratio;
    precision_t* out_newvalue;
    const precision_t* actions;
    const precision_t* old_logprobs;
    const precision_t* advantages;
    const precision_t* prio;
    const precision_t* values;
    const precision_t* returns;
};

struct PPOKernelArgs {
    float* grad_logits;
    float* grad_logstd; // For continuous actions
    float* grad_values_pred;
    const precision_t* logits;
    const precision_t* logstd; // Continuous only
    const precision_t* values_pred;
    const float* adv_mean;
    const float* adv_var;
    const int* act_sizes;
    const precision_t* action_mask; // (N, T, A_total) or nullptr
    int mask_stride_n, mask_stride_t;
    int num_atns;
    float clip_coef, vf_clip_coef, vf_coef, ent_coef;
    int T_seq, A_total, N;
    int logits_stride_n, logits_stride_t, logits_stride_a;
    int values_stride_n, values_stride_t;
    bool is_continuous;
};

struct PPOBuffersPuf {
    FloatTensor loss_output, grad_loss;
    FloatTensor saved_for_bwd;
    FloatTensor grad_logits, grad_values, grad_logstd, adv_scratch;
};

void register_ppo_buffers(PPOBuffersPuf& bufs, Allocator* alloc, int N, int T, int A_total, bool is_continuous) {
    long total = (long)N * T;
    bufs = (PPOBuffersPuf){
        .loss_output = {.shape = {1}},
        .grad_loss = {.shape = {1}},
        .saved_for_bwd = {.shape = {total, 5}},
        .grad_logits = {.shape = {N, T, A_total}},
        .grad_values = {.shape = {N, T, 1}},
        .grad_logstd = {.shape = {N, T, A_total}},
        .adv_scratch = {.shape = {2}},
    };
    alloc_register(alloc, &bufs.loss_output);
    alloc_register(alloc, &bufs.saved_for_bwd);
    alloc_register(alloc, &bufs.grad_loss);
    alloc_register(alloc, &bufs.grad_logits);
    alloc_register(alloc, &bufs.grad_values);
    if (is_continuous) {
        alloc_register(alloc, &bufs.grad_logstd);
    }
    alloc_register(alloc, &bufs.adv_scratch);
}

__device__ __forceinline__ float load_logit_masked(
        const precision_t* __restrict__ logits, int logits_base,
        int logits_stride_a, int logits_offset, int a,
        const precision_t* __restrict__ mask, int mask_base) {
    float l = to_float(logits[logits_base + (logits_offset + a) * logits_stride_a]);
    if (mask != nullptr) {
        float m = to_float(mask[mask_base + logits_offset + a]);
        if (m == 0.0f) {
            l = -1e4f;
            return l;
        }
    }
    return l;
}

__device__ __forceinline__ void ppo_discrete_head(
        const precision_t* __restrict__ logits, int logits_base,
        int logits_stride_a, int logits_offset, int A, int act,
        const precision_t* __restrict__ mask, int mask_base,
        float* out_logsumexp, float* out_entropy, float* out_logp) {
    float max_logit = -INFINITY;
    float sum = 0.0f;
    float act_logit = 0.0f;

    for (int a = 0; a < A; ++a) {
        float l = load_logit_masked(logits, logits_base, logits_stride_a, logits_offset, a, mask, mask_base);
        if (a == act) {
            act_logit = l;
        }
        if (l > max_logit) {
            sum *= __expf(max_logit - l);
            max_logit = l;
        }
        sum += __expf(l - max_logit);
    }
    float logsumexp = max_logit + __logf(sum);

    float ent = 0.0f;
    for (int a = 0; a < A; ++a) {
        float l = load_logit_masked(logits, logits_base, logits_stride_a, logits_offset, a, mask, mask_base);
        float logp = l - logsumexp;
        float p = __expf(logp);
        ent -= p * logp;
    }

    *out_logsumexp = logsumexp;
    *out_entropy = ent;
    *out_logp = act_logit - logsumexp;
}

__device__ __forceinline__ void ppo_continuous_head(
        float mean, float log_std, float action,
        float* out_logp, float* out_entropy) {
    constexpr float HALF_LOG_2PI = 0.9189385332046727f;
    constexpr float HALF_1_PLUS_LOG_2PI = 1.4189385332046727f;
    float std = __expf(log_std);
    float normalized = (action - mean) / std;
    *out_logp = -0.5f * normalized * normalized - HALF_LOG_2PI - log_std;
    *out_entropy = HALF_1_PLUS_LOG_2PI + log_std;
}

__global__ void ppo_loss_compute(
        float* __restrict__ ppo_partials,
        PPOKernelArgs a, PPOGraphArgs g) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int total_elements = a.N * a.T_seq;
    float inv_NT = 1.0f / float(total_elements);

    __shared__ float block_losses[LOSS_N][PPO_THREADS];
    for (int c = 0; c < LOSS_N; c++) {
        block_losses[c][tid] = 0.0f;
    }

    if (idx >= total_elements) {
        goto reduce;
    }

    {
    int n = idx / a.T_seq;
    int t = idx % a.T_seq;
    int nt = n * a.T_seq + t;

    int logits_base = n * a.logits_stride_n + t * a.logits_stride_t;
    int values_idx = n * a.values_stride_n + t * a.values_stride_t;
    int grad_logits_base = nt * a.A_total;

    // Shared computation (used by both forward and backward)

    float old_logp = to_float(g.old_logprobs[nt]);
    float adv = to_float(g.advantages[nt]);
    float w = to_float(g.prio[n]);
    float val = to_float(g.values[nt]);
    float ret = to_float(g.returns[nt]);
    float val_pred = to_float(a.values_pred[values_idx]);
    g.out_newvalue[nt] = from_float(val_pred);

    float adv_std = sqrtf(float(a.adv_var[0]));
    float adv_normalized = (adv - float(a.adv_mean[0])) / (adv_std + 1e-8f);

    // grad_loss is always 1.0 (set in post_create, never changes)
    float dL = inv_NT;
    float d_pg_loss = dL;
    float d_entropy_term = dL * (-a.ent_coef);

    // Value loss (forward) + value gradient (backward)

    float v_error = val_pred - val;
    float v_clipped = val + fmaxf(-a.vf_clip_coef, fminf(a.vf_clip_coef, v_error));
    float v_loss_unclipped = (val_pred - ret) * (val_pred - ret);
    float v_loss_clipped = (v_clipped - ret) * (v_clipped - ret);
    float v_loss = 0.5f * fmaxf(v_loss_unclipped, v_loss_clipped);

    // Value gradient
    bool use_clipped_vf = (v_loss_clipped > v_loss_unclipped);
    float d_val_pred = 0.0f;
    if (use_clipped_vf) {
        if (v_error >= -a.vf_clip_coef && v_error <= a.vf_clip_coef) {
            d_val_pred = v_clipped - ret;
        }
    } else {
        d_val_pred = val_pred - ret;
    }
    a.grad_values_pred[nt] = dL * a.vf_coef * d_val_pred;

    // Policy loss + gradients

    float pg_loss, total_entropy, logratio, ratio;
    float total_log_prob = 0.0f;
    total_entropy = 0.0f;

    // Discrete-only: per-head arrays needed across forward + backward
    float head_logsumexp[MAX_ATN_HEADS];
    float head_entropy[MAX_ATN_HEADS];
    int head_act[MAX_ATN_HEADS];

    int mask_base = (a.action_mask != nullptr)
        ? n * a.mask_stride_n + t * a.mask_stride_t : 0;

    if (!a.is_continuous) {
        int logits_offset = 0;
        for (int h = 0; h < a.num_atns; ++h) {
            int A = a.act_sizes[h];
            int act = static_cast<int>(g.actions[nt * a.num_atns + h]);
            head_act[h] = act;
            float lse, ent, lp;
            ppo_discrete_head(a.logits, logits_base, a.logits_stride_a, logits_offset, A, act,
                              a.action_mask, mask_base, &lse, &ent, &lp);
            head_logsumexp[h] = lse;
            head_entropy[h] = ent;
            total_log_prob += lp;
            total_entropy += ent;
            logits_offset += A;
        }
    } else {
        for (int h = 0; h < a.num_atns; ++h) {
            float mean = safe_continuous_mean(a.logits, logits_base + h * a.logits_stride_a);
            float log_std = safe_continuous_logstd(a.logstd, h);
            float action = finite_or_clamp(float(g.actions[nt * a.num_atns + h]), -1.0e6f, 1.0e6f);
            float lp, ent;
            ppo_continuous_head(mean, log_std, action, &lp, &ent);
            total_log_prob += lp;
            total_entropy += ent;
        }
    }

    // Shared pg loss computation
    logratio = total_log_prob - old_logp;
    ratio = __expf(logratio);
    g.out_ratio[nt] = from_float(ratio);
    float ratio_clipped = fmaxf(1.0f - a.clip_coef, fminf(1.0f + a.clip_coef, ratio));
    float wa = -w * adv_normalized;
    float pg_loss1 = wa * ratio;
    float pg_loss2 = wa * ratio_clipped;
    pg_loss = fmaxf(pg_loss1, pg_loss2);

    float d_ratio = wa * d_pg_loss;
    if (pg_loss2 > pg_loss1) {
        if (ratio <= (1.0f - a.clip_coef) || ratio >= (1.0f + a.clip_coef)) {
            d_ratio = 0.0f;
        }
    }
    float d_new_logp = d_ratio * ratio;

    if (!a.is_continuous) {
        int logits_offset = 0;
        for (int h = 0; h < a.num_atns; ++h) {
            int A = a.act_sizes[h];
            int act = head_act[h];
            float logsumexp = head_logsumexp[h];
            float ent = head_entropy[h];

            for (int j = 0; j < A; ++j) {
                float l = load_logit_masked(a.logits, logits_base, a.logits_stride_a,
                                            logits_offset, j, a.action_mask, mask_base);
                float logp = l - logsumexp;
                float p = __expf(logp);
                float d_logit = (j == act) ? d_new_logp : 0.0f;
                d_logit -= p * d_new_logp;
                d_logit += d_entropy_term * p * (-ent - logp);
                a.grad_logits[grad_logits_base + logits_offset + j] = d_logit;
            }
            logits_offset += A;
        }
    } else {
        for (int h = 0; h < a.num_atns; ++h) {
            float mean = safe_continuous_mean(a.logits, logits_base + h * a.logits_stride_a);
            float log_std = safe_continuous_logstd(a.logstd, h);
            float std = __expf(log_std);
            float var = std * std;
            float action = finite_or_clamp(float(g.actions[nt * a.num_atns + h]), -1.0e6f, 1.0e6f);
            float diff = action - mean;

            a.grad_logits[grad_logits_base + h] = d_new_logp * diff / var;
            a.grad_logstd[nt * a.num_atns + h] = d_new_logp * (diff * diff / var - 1.0f) + d_entropy_term;
        }
    }

    // Forward: loss partials
    float thread_loss = (pg_loss + a.vf_coef * v_loss - a.ent_coef * total_entropy) * inv_NT;
    block_losses[LOSS_PG][tid] = pg_loss * inv_NT;
    block_losses[LOSS_VF][tid] = v_loss * inv_NT;
    block_losses[LOSS_ENT][tid] = total_entropy * inv_NT;
    block_losses[LOSS_TOTAL][tid] = thread_loss;
    block_losses[LOSS_OLD_APPROX_KL][tid] = (-logratio) * inv_NT;
    block_losses[LOSS_APPROX_KL][tid] = ((ratio - 1.0f) - logratio) * inv_NT;
    block_losses[LOSS_CLIPFRAC][tid] = (fabsf(ratio - 1.0f) > a.clip_coef ? 1.0f : 0.0f) * inv_NT;
    } // end if (idx < total_elements)

// Deterministic aggregation
reduce:
    __syncthreads();

    for (int stride = PPO_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (int c = 0; c < LOSS_N; c++) {
                block_losses[c][tid] += block_losses[c][tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        int base = blockIdx.x * (LOSS_N + 1);
        ppo_partials[base] = block_losses[LOSS_TOTAL][0];
        for (int c = 0; c < LOSS_N; c++) {
            ppo_partials[base + 1 + c] = block_losses[c][0];
        }
    }
}

// Deterministic reduction of per-block PPO loss partials + count increment
__global__ void ppo_loss_reduce(
        float* __restrict__ loss,
        float* __restrict__ losses_acc,
        const float* __restrict__ partials,
        int num_blocks) {
    int tid = threadIdx.x;
    if (tid > LOSS_N) {
        return;
    }

    float sum = 0.0f;
    for (int b = 0; b < num_blocks; b++) {
        sum += partials[b * (LOSS_N + 1) + tid];
    }

    if (tid == 0) {
        *loss += sum;
    } else {
        losses_acc[tid - 1] += sum;
    }

    // Fold add_scalar: increment epoch count
    if (tid == 0) {
        losses_acc[LOSS_N] += 1.0f;
    }
}

__global__ void ppo_var_mean(const precision_t* __restrict__ src,
        float* __restrict__ var_out, float* __restrict__ mean_out, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        sum += to_float(src[i]);
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float mean = sdata[0] / (float)n;
    if (tid == 0) {
        *mean_out = mean;
    }
    __syncthreads();
    float ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float d = to_float(src[i]) - mean;
        ss += d * d;
    }
    sdata[tid] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        *var_out = sdata[0] / (float)(n - 1);
    }
}

// This is a huge kernel for a relatively cheap operation. But without this,
// it's death by a thousand cuts with repeated kernel launches. Even graphed, you
// blow up the memory bandwidth.
void ppo_loss_fwd_bwd(
        PrecisionTensor& dec_out,    // (N, T, fused_cols) - fused logits+value from decoder
        PrecisionTensor& logstd,     // continuous logstd or empty
        TrainGraph& graph,
        IntTensor& act_sizes, FloatTensor& losses_acc,
        float clip_coef, float vf_clip_coef, float vf_coef, float ent_coef,
        PPOBuffersPuf& bufs, bool is_continuous,
        cudaStream_t stream) {
    int N = dec_out.shape[0], T = dec_out.shape[1], fused_cols = dec_out.shape[2];
    int A_total = fused_cols - 1;  // last column is value
    int total = N * T;

    // Pointers into fused decoder output
    const precision_t* logits_ptr = dec_out.data;

    float* adv_var_ptr = bufs.adv_scratch.data;
    float* adv_mean_ptr = adv_var_ptr + 1;
    ppo_var_mean<<<1, 256, 0, stream>>>(
        graph.mb_advantages.data, adv_var_ptr, adv_mean_ptr, numel(graph.mb_advantages.shape));

    int ppo_grid = (total + PPO_THREADS - 1) / PPO_THREADS;

    static float* ppo_partials_buf = nullptr;
    static int ppo_partials_capacity = 0;
    int ppo_partials_needed = ppo_grid * (LOSS_N + 1);
    if (!ppo_partials_buf || ppo_partials_needed > ppo_partials_capacity) {
        if (ppo_partials_buf) {
            cudaFree(ppo_partials_buf);
        }
        ppo_partials_capacity = ppo_partials_needed;
        cudaMalloc(&ppo_partials_buf, ppo_partials_capacity * sizeof(float));
    }

    cudaMemsetAsync(bufs.loss_output.data, 0, sizeof(float), stream);

    PPOGraphArgs graph_args = {
        .out_ratio = graph.mb_ratio.data,
        .out_newvalue = graph.mb_newvalue.data,
        .actions = graph.mb_actions.data,
        .old_logprobs = graph.mb_logprobs.data,
        .advantages = graph.mb_advantages.data,
        .prio = graph.mb_prio.data,
        .values = graph.mb_values.data,
        .returns = graph.mb_returns.data,
    };

    bool has_mask = (graph.mb_action_mask.data != nullptr);
    PPOKernelArgs args = {
        .grad_logits = bufs.grad_logits.data,
        .grad_logstd = is_continuous ? bufs.grad_logstd.data : nullptr,
        .grad_values_pred = bufs.grad_values.data,
        .logits = logits_ptr,
        .logstd = is_continuous ? logstd.data : nullptr,
        .values_pred = logits_ptr + A_total,
        .adv_mean = adv_mean_ptr,
        .adv_var = adv_var_ptr,
        .act_sizes = act_sizes.data,
        .action_mask = has_mask ? graph.mb_action_mask.data : nullptr,
        .mask_stride_n = has_mask ? T * A_total : 0,
        .mask_stride_t = has_mask ? A_total : 0,
        .num_atns = (int)numel(act_sizes.shape),
        .clip_coef = clip_coef, .vf_clip_coef = vf_clip_coef,
        .vf_coef = vf_coef, .ent_coef = ent_coef,
        .T_seq = T, .A_total = A_total, .N = N,
        .logits_stride_n = T * fused_cols, .logits_stride_t = fused_cols, .logits_stride_a = 1,
        .values_stride_n = T * fused_cols, .values_stride_t = fused_cols,
        .is_continuous = is_continuous,
    };

    ppo_loss_compute<<<ppo_grid, PPO_THREADS, 0, stream>>>(ppo_partials_buf, args, graph_args);

    ppo_loss_reduce<<<1, LOSS_N + 1, 0, stream>>>(
        bufs.loss_output.data, losses_acc.data, ppo_partials_buf, ppo_grid);
}
