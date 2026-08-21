// NMMO3 CUDA encoder: sparse conv, embedding, concat, projection.
// Included by src/ocean.cu.

#define N3_MAP_H 11
#define N3_MAP_W 15
#define N3_NFEAT 10
#define N3_MAP_SIZE (N3_MAP_H * N3_MAP_W * N3_NFEAT)
#define N3_PLAYER 47
#define N3_REWARD 10
#define N3_EMBED_DIM 32
#define N3_EMBED_VOCAB 128
#define N3_PLAYER_EMBED (N3_PLAYER * N3_EMBED_DIM)
#define N3_C1_IC 59
#define N3_C1_OC 128
#define N3_C1_K 5
#define N3_C1_S 3
#define N3_C1_OH 3
#define N3_C1_OW 4
#define N3_C2_IC 128
#define N3_C2_OC 128
#define N3_C2_K 3
#define N3_C2_OH 1
#define N3_C2_OW 2
#define N3_CONV_FLAT (N3_C2_OC * N3_C2_OH * N3_C2_OW)
#define N3_CONCAT (N3_CONV_FLAT + N3_PLAYER_EMBED + N3_PLAYER + N3_REWARD)
#define N3_C1_COL_W (N3_C1_IC * N3_C1_K * N3_C1_K)
#define N3_C1_SPATIAL (N3_C1_OH * N3_C1_OW)
#define N3_C2_COL_W (N3_C2_IC * N3_C2_K * N3_C2_K)
#define N3_C2_SPATIAL (N3_C2_OH * N3_C2_OW)
#define N3_C2_IN_PLANE (N3_C2_IC * N3_C1_OH * N3_C1_OW)
#define N3_EMBED_N (N3_EMBED_VOCAB * N3_EMBED_DIM)
// 2^24 fixed-point: integer atomics are associative, so embed scatter is
// bit-identical run to run. Quantization (6e-8) is below fp32 ulp here.
#define N3_FXP 16777216.0f

__constant__ int N3_OFFSETS[10] = {0, 4, 8, 25, 30, 33, 38, 43, 48, 55};

// Conv1 im2col from raw map codes. Channel ic is one-hot of feature f at
// (ih, iw); skips the dense 59-channel multihot tensor.
__global__ void n3_c1_im2col_obs(
    const precision_t* __restrict__ obs, precision_t* __restrict__ col,
    int B, int obs_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N3_C1_SPATIAL * N3_C1_COL_W) {
        return;
    }
    int row = idx / N3_C1_COL_W;
    int c = idx % N3_C1_COL_W;
    int b = row / N3_C1_SPATIAL;
    int rem = row % N3_C1_SPATIAL;
    int oh = rem / N3_C1_OW;
    int ow = rem % N3_C1_OW;
    int ic = c / (N3_C1_K * N3_C1_K);
    int kk = c % (N3_C1_K * N3_C1_K);
    int kh = kk / N3_C1_K;
    int kw = kk % N3_C1_K;
    int ih = oh * N3_C1_S + kh;
    int iw = ow * N3_C1_S + kw;
    int f = 0;
#pragma unroll
    for (int i = 1; i < 10; i++) {
        if (ic >= N3_OFFSETS[i]) {
            f = i;
        }
    }
    int val = (int)to_float(obs[(int64_t)b * obs_size
        + (int64_t)(ih * N3_MAP_W + iw) * N3_NFEAT + f]);
    col[idx] = from_float(val == ic - N3_OFFSETS[f] ? 1.0f : 0.0f);
}

__global__ void n3_c2_im2col(
    const precision_t* __restrict__ input, precision_t* __restrict__ col, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N3_C2_SPATIAL * N3_C2_COL_W) {
        return;
    }
    int row = idx / N3_C2_COL_W;
    int c = idx % N3_C2_COL_W;
    int b = row / N3_C2_SPATIAL;
    int ow = row % N3_C2_SPATIAL;
    int ic = c / (N3_C2_K * N3_C2_K);
    int kk = c % (N3_C2_K * N3_C2_K);
    int kh = kk / N3_C2_K;
    int kw = kk % N3_C2_K;
    col[idx] = input[((int64_t)b * N3_C2_IC + ic) * N3_C1_SPATIAL
        + kh * N3_C1_OW + (ow + kw)];
}

__global__ void n3_c2_col2im(
    const precision_t* __restrict__ col, precision_t* __restrict__ grad_input,
    int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N3_C2_IN_PLANE) {
        return;
    }
    int iw = idx % N3_C1_OW;
    int ih = (idx / N3_C1_OW) % N3_C1_OH;
    int ic = (idx / N3_C1_SPATIAL) % N3_C2_IC;
    int b = idx / N3_C2_IN_PLANE;
    float sum = 0.0f;
    int base = b * N3_C2_SPATIAL * N3_C2_COL_W
        + ic * (N3_C2_K * N3_C2_K) + ih * N3_C2_K;
    for (int kw = 0; kw < N3_C2_K; kw++) {
        int ow = iw - kw;
        if (ow >= 0 && ow < N3_C2_OW) {
            sum += to_float(col[base + ow * N3_C2_COL_W + kw]);
        }
    }
    grad_input[idx] = from_float(sum);
}

__global__ void n3_rows_to_nchw(
    const precision_t* __restrict__ src, precision_t* __restrict__ dst,
    int B, int OC, int spatial, int relu) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * OC * spatial) {
        return;
    }
    int b = idx / (OC * spatial);
    int oc = (idx / spatial) % OC;
    int s = idx % spatial;
    float value = to_float(src[(b * spatial + s) * OC + oc]);
    if (relu) {
        value = fmaxf(0.0f, value);
    }
    dst[idx] = from_float(value);
}

__global__ void n3_nchw_to_rows(
    const precision_t* __restrict__ src, precision_t* __restrict__ dst,
    int B, int OC, int spatial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * OC * spatial) {
        return;
    }
    int b = idx / (OC * spatial);
    int oc = (idx / spatial) % OC;
    int s = idx % spatial;
    dst[(b * spatial + s) * OC + oc] = src[idx];
}

__global__ void n3_embedding_kernel(
    precision_t* __restrict__ out, const precision_t* __restrict__ obs,
    const precision_t* __restrict__ embed_w, int B, int obs_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N3_PLAYER) {
        return;
    }
    int b = idx / N3_PLAYER;
    int f = idx % N3_PLAYER;
    int val = (int)to_float(obs[b * obs_size + N3_MAP_SIZE + f]);
    const precision_t* src = embed_w + val * N3_EMBED_DIM;
    precision_t* dst = out + b * N3_PLAYER_EMBED + f * N3_EMBED_DIM;
#ifdef PRECISION_FLOAT
    const float4* src4 = (const float4*)src;
    float4* dst4 = (float4*)dst;
#pragma unroll
    for (int i = 0; i < N3_EMBED_DIM / 4; i++) {
        dst4[i] = src4[i];
    }
#else
    const uint4* src4 = (const uint4*)src;
    uint4* dst4 = (uint4*)dst;
#pragma unroll
    for (int i = 0; i < N3_EMBED_DIM / 8; i++) {
        dst4[i] = src4[i];
    }
#endif
}

__global__ void n3_concat_kernel(
    precision_t* __restrict__ out, const precision_t* __restrict__ conv_flat,
    const precision_t* __restrict__ embed, const precision_t* __restrict__ obs,
    int B, int obs_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N3_CONCAT) {
        return;
    }
    int b = idx / N3_CONCAT;
    int c = idx % N3_CONCAT;
    precision_t val;
    if (c < N3_CONV_FLAT) {
        val = conv_flat[b * N3_CONV_FLAT + c];
    } else if (c < N3_CONV_FLAT + N3_PLAYER_EMBED) {
        val = embed[b * N3_PLAYER_EMBED + (c - N3_CONV_FLAT)];
    } else if (c < N3_CONV_FLAT + N3_PLAYER_EMBED + N3_PLAYER) {
        val = obs[b * obs_size + N3_MAP_SIZE
            + (c - N3_CONV_FLAT - N3_PLAYER_EMBED)];
    } else {
        val = obs[b * obs_size + obs_size - N3_REWARD
            + (c - N3_CONV_FLAT - N3_PLAYER_EMBED - N3_PLAYER)];
    }
    out[idx] = val;
}

__global__ void n3_relu_kernel(precision_t* __restrict__ data, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    data[idx] = from_float(fmaxf(0.0f, to_float(data[idx])));
}

__global__ void n3_relu_backward_kernel(
    precision_t* __restrict__ grad, const precision_t* __restrict__ out,
    int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    if (to_float(out[idx]) <= 0.0f) {
        grad[idx] = from_float(0.0f);
    }
}

__global__ void n3_concat_backward_conv_kernel(
    precision_t* __restrict__ conv_grad,
    const precision_t* __restrict__ concat_grad, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N3_CONV_FLAT) {
        return;
    }
    int b = idx / N3_CONV_FLAT;
    int c = idx % N3_CONV_FLAT;
    conv_grad[b * N3_CONV_FLAT + c] = concat_grad[b * N3_CONCAT + c];
}

// Per-block int64 histogram over the 128-row table, then one global integer
// atomic per table entry. Integer add is associative so scatter order does
// not change bits.
__global__ void n3_embedding_backward_kernel(
    long long* __restrict__ embed_wgrad_i,
    const precision_t* __restrict__ concat_grad,
    const precision_t* __restrict__ obs,
    int B, int obs_size) {
    __shared__ long long acc[N3_EMBED_N];
    for (int i = threadIdx.x; i < N3_EMBED_N; i += blockDim.x) {
        acc[i] = 0;
    }
    __syncthreads();
    int total = B * N3_PLAYER;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
            idx += blockDim.x * gridDim.x) {
        int b = idx / N3_PLAYER;
        int f = idx % N3_PLAYER;
        int val = (int)to_float(obs[b * obs_size + N3_MAP_SIZE + f]);
        const precision_t* g = concat_grad + b * N3_CONCAT + N3_CONV_FLAT
            + f * N3_EMBED_DIM;
#pragma unroll
        for (int d = 0; d < N3_EMBED_DIM; d++) {
            atomicAdd((unsigned long long*)&acc[val * N3_EMBED_DIM + d],
                (unsigned long long)(long long)__float2ll_rn(
                    to_float(g[d]) * N3_FXP));
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < N3_EMBED_N; i += blockDim.x) {
        if (acc[i] != 0) {
            atomicAdd((unsigned long long*)&embed_wgrad_i[i],
                (unsigned long long)acc[i]);
        }
    }
}

__global__ void n3_fxp_to_precision_kernel(
    precision_t* __restrict__ dst, const long long* __restrict__ src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float((float)((double)src[idx] * (1.0 / (double)N3_FXP)));
    }
}

struct NMMO3EncoderWeights {
    Prec conv1_w, conv2_w, embed_w, proj_w;
    int obs_size, hidden;
};

struct NMMO3EncoderActivations {
    Prec conv1_out, conv1_grad, conv1_wgrad;
    Prec conv2_out, conv2_grad, conv2_wgrad;
    Prec col1, mm1, col2, mm2;
    Prec embed_out, concat, out, saved_obs;
    Prec embed_wgrad, proj_wgrad;
    Long embed_wgrad_i;
};

static Prec nmmo3_encoder_forward(
    void* w, void* activations, Prec input, cudaStream_t stream) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)w;
    NMMO3EncoderActivations* a = (NMMO3EncoderActivations*)activations;
    int B = input.shape[0];
    if (a->saved_obs.data) {
        puf_copy(&a->saved_obs, &input, stream);
    }

    int c1_rows = B * N3_C1_SPATIAL;
    n3_c1_im2col_obs<<<grid_size(c1_rows * N3_C1_COL_W), BLOCK_SIZE, 0, stream>>>(
        input.data, a->col1.data, B, ew->obs_size);
    Prec c1_col = {.data = a->col1.data, .shape = {c1_rows, N3_C1_COL_W}};
    Prec c1_mm = {.data = a->mm1.data, .shape = {c1_rows, N3_C1_OC}};
    puf_mm(&c1_col, &ew->conv1_w, &c1_mm, stream);
    n3_rows_to_nchw<<<grid_size(B * N3_C1_OC * N3_C1_SPATIAL), BLOCK_SIZE, 0,
        stream>>>(a->mm1.data, a->conv1_out.data, B, N3_C1_OC, N3_C1_SPATIAL, 1);

    int c2_rows = B * N3_C2_SPATIAL;
    n3_c2_im2col<<<grid_size(c2_rows * N3_C2_COL_W), BLOCK_SIZE, 0, stream>>>(
        a->conv1_out.data, a->col2.data, B);
    Prec c2_col = {.data = a->col2.data, .shape = {c2_rows, N3_C2_COL_W}};
    Prec c2_mm = {.data = a->mm2.data, .shape = {c2_rows, N3_C2_OC}};
    puf_mm(&c2_col, &ew->conv2_w, &c2_mm, stream);
    n3_rows_to_nchw<<<grid_size(B * N3_C2_OC * N3_C2_SPATIAL), BLOCK_SIZE, 0,
        stream>>>(a->mm2.data, a->conv2_out.data, B, N3_C2_OC, N3_C2_SPATIAL, 0);

    n3_embedding_kernel<<<grid_size(B * N3_PLAYER), BLOCK_SIZE, 0, stream>>>(
        a->embed_out.data, input.data, ew->embed_w.data, B, ew->obs_size);
    n3_concat_kernel<<<grid_size(B * N3_CONCAT), BLOCK_SIZE, 0, stream>>>(
        a->concat.data, a->conv2_out.data, a->embed_out.data, input.data, B,
        ew->obs_size);
    puf_mm(&a->concat, &ew->proj_w, &a->out, stream);
    n3_relu_kernel<<<grid_size(B * ew->hidden), BLOCK_SIZE, 0, stream>>>(
        a->out.data, B * ew->hidden);
    return a->out;
}

// col_buf still holds the forward im2col; skip rebuilding it.
static void n3_conv_wgrad(
    precision_t* grad_output, precision_t* wgrad,
    precision_t* col_buf, precision_t* mm_buf,
    int B, int OC, int spatial, int col_cols, cudaStream_t stream) {
    int col_rows = B * spatial;
    n3_nchw_to_rows<<<grid_size(B * OC * spatial), BLOCK_SIZE, 0, stream>>>(
        grad_output, mm_buf, B, OC, spatial);
    Prec mm_t = {.data = mm_buf, .shape = {col_rows, OC}};
    Prec col_t = {.data = col_buf, .shape = {col_rows, col_cols}};
    Prec wg_t = {.data = wgrad, .shape = {OC, col_cols}};
    puf_mm_tn(&mm_t, &col_t, &wg_t, stream);
}

static void nmmo3_encoder_backward(
    void* w, void* activations, Prec grad, cudaStream_t stream) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)w;
    NMMO3EncoderActivations* a = (NMMO3EncoderActivations*)activations;
    int B = grad.shape[0];
    int H = ew->hidden;

    n3_relu_backward_kernel<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(
        grad.data, a->out.data, B * H);
    puf_mm_tn(&grad, &a->concat, &a->proj_wgrad, stream);
    Prec grad_concat = {.data = a->concat.data, .shape = {B, N3_CONCAT}};
    puf_mm_nn(&grad, &ew->proj_w, &grad_concat, stream);
    n3_concat_backward_conv_kernel<<<grid_size(B * N3_CONV_FLAT), BLOCK_SIZE, 0,
        stream>>>(a->conv2_grad.data, grad_concat.data, B);

    n3_conv_wgrad(a->conv2_grad.data, a->conv2_wgrad.data,
        a->col2.data, a->mm2.data, B, N3_C2_OC, N3_C2_SPATIAL, N3_C2_COL_W,
        stream);
    Prec mm2 = {.data = a->mm2.data, .shape = {B * N3_C2_SPATIAL, N3_C2_OC}};
    Prec col2 = {.data = a->col2.data, .shape = {B * N3_C2_SPATIAL, N3_C2_COL_W}};
    puf_mm_nn(&mm2, &ew->conv2_w, &col2, stream);
    n3_c2_col2im<<<grid_size(B * N3_C2_IN_PLANE), BLOCK_SIZE, 0, stream>>>(
        a->col2.data, a->conv1_grad.data, B);

    n3_relu_backward_kernel<<<grid_size(B * N3_C1_OC * N3_C1_SPATIAL),
        BLOCK_SIZE, 0, stream>>>(
        a->conv1_grad.data, a->conv1_out.data, B * N3_C1_OC * N3_C1_SPATIAL);
    n3_conv_wgrad(a->conv1_grad.data, a->conv1_wgrad.data,
        a->col1.data, a->mm1.data, B, N3_C1_OC, N3_C1_SPATIAL, N3_C1_COL_W,
        stream);

    cudaMemsetAsync(a->embed_wgrad_i.data, 0, N3_EMBED_N * sizeof(long), stream);
    int blocks = B * N3_PLAYER / BLOCK_SIZE;
    if (blocks < 1) {
        blocks = 1;
    }
    if (blocks > 256) {
        blocks = 256;
    }
    n3_embedding_backward_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        (long long*)a->embed_wgrad_i.data, grad_concat.data, a->saved_obs.data,
        B, ew->obs_size);
    n3_fxp_to_precision_kernel<<<grid_size(N3_EMBED_N), BLOCK_SIZE, 0, stream>>>(
        a->embed_wgrad.data, (long long*)a->embed_wgrad_i.data, N3_EMBED_N);
}

static void nmmo3_encoder_init_weights(
    void* w, uint64_t* seed, cudaStream_t stream) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)w;
    Prec c1 = {.data = ew->conv1_w.data, .shape = {N3_C1_OC, N3_C1_COL_W}};
    Prec c2 = {.data = ew->conv2_w.data, .shape = {N3_C2_OC, N3_C2_COL_W}};
    Prec proj = {.data = ew->proj_w.data, .shape = {ew->hidden, N3_CONCAT}};
    puf_kaiming_init(&c1, 1.0f, (*seed)++, stream);
    puf_kaiming_init(&c2, 1.0f, (*seed)++, stream);
    puf_normal_init(&ew->embed_w, 1.0f, (*seed)++, stream);
    puf_kaiming_init(&proj, 1.0f, (*seed)++, stream);
}

static void nmmo3_encoder_reg_params(void* w, Allocator* alloc) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)w;
    ew->conv1_w = {.shape = {N3_C1_OC, N3_C1_COL_W}};
    ew->conv2_w = {.shape = {N3_C2_OC, N3_C2_COL_W}};
    ew->embed_w = {.shape = {N3_EMBED_VOCAB, N3_EMBED_DIM}};
    ew->proj_w = {.shape = {ew->hidden, N3_CONCAT}};
    alloc_register(alloc, &ew->conv1_w); alloc_register(alloc, &ew->conv2_w);
    alloc_register(alloc, &ew->embed_w); alloc_register(alloc, &ew->proj_w);
}

static void nmmo3_encoder_reg_train(
    void* w, void* activations, Allocator* acts, Allocator* grads, int B_TT) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)w;
    NMMO3EncoderActivations* a = (NMMO3EncoderActivations*)activations;
    a->conv1_out = {.shape = {B_TT * N3_C1_OC * N3_C1_SPATIAL}};
    a->conv1_grad = {.shape = {B_TT * N3_C1_OC * N3_C1_SPATIAL}};
    a->conv1_wgrad = {.shape = {N3_C1_OC, N3_C1_COL_W}};
    a->col1 = {.shape = {B_TT * N3_C1_SPATIAL, N3_C1_COL_W}};
    a->mm1 = {.shape = {B_TT * N3_C1_SPATIAL, N3_C1_OC}};
    a->conv2_out = {.shape = {B_TT * N3_C2_OC * N3_C2_SPATIAL}};
    a->conv2_grad = {.shape = {B_TT * N3_C2_OC * N3_C2_SPATIAL}};
    a->conv2_wgrad = {.shape = {N3_C2_OC, N3_C2_COL_W}};
    a->col2 = {.shape = {B_TT * N3_C2_SPATIAL, N3_C2_COL_W}};
    a->mm2 = {.shape = {B_TT * N3_C2_SPATIAL, N3_C2_OC}};
    a->embed_out = {.shape = {B_TT, N3_PLAYER_EMBED}};
    a->concat = {.shape = {B_TT, N3_CONCAT}};
    a->out = {.shape = {B_TT, ew->hidden}};
    a->saved_obs = {.shape = {B_TT, ew->obs_size}};
    a->embed_wgrad = {.shape = {N3_EMBED_VOCAB, N3_EMBED_DIM}};
    a->embed_wgrad_i = {.shape = {N3_EMBED_VOCAB, N3_EMBED_DIM}};
    a->proj_wgrad = {.shape = {ew->hidden, N3_CONCAT}};
    alloc_register(acts, &a->conv1_out); alloc_register(acts, &a->conv1_grad);
    alloc_register(grads, &a->conv1_wgrad);
    alloc_register(acts, &a->col1); alloc_register(acts, &a->mm1);
    alloc_register(acts, &a->conv2_out); alloc_register(acts, &a->conv2_grad);
    alloc_register(grads, &a->conv2_wgrad);
    alloc_register(acts, &a->col2); alloc_register(acts, &a->mm2);
    alloc_register(acts, &a->embed_out); alloc_register(acts, &a->concat);
    alloc_register(acts, &a->out); alloc_register(acts, &a->saved_obs);
    alloc_register(grads, &a->embed_wgrad); alloc_register(acts, &a->embed_wgrad_i);
    alloc_register(grads, &a->proj_wgrad);
}

static void nmmo3_encoder_reg_rollout(
    void* w, void* activations, Allocator* alloc, int B) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)w;
    NMMO3EncoderActivations* a = (NMMO3EncoderActivations*)activations;
    a->conv1_out = {.shape = {B * N3_C1_OC * N3_C1_SPATIAL}};
    a->col1 = {.shape = {B * N3_C1_SPATIAL, N3_C1_COL_W}};
    a->mm1 = {.shape = {B * N3_C1_SPATIAL, N3_C1_OC}};
    a->conv2_out = {.shape = {B * N3_C2_OC * N3_C2_SPATIAL}};
    a->col2 = {.shape = {B * N3_C2_SPATIAL, N3_C2_COL_W}};
    a->mm2 = {.shape = {B * N3_C2_SPATIAL, N3_C2_OC}};
    a->embed_out = {.shape = {B, N3_PLAYER_EMBED}};
    a->concat = {.shape = {B, N3_CONCAT}};
    a->out = {.shape = {B, ew->hidden}};
    alloc_register(alloc, &a->conv1_out);
    alloc_register(alloc, &a->col1); alloc_register(alloc, &a->mm1);
    alloc_register(alloc, &a->conv2_out);
    alloc_register(alloc, &a->col2); alloc_register(alloc, &a->mm2);
    alloc_register(alloc, &a->embed_out); alloc_register(alloc, &a->concat);
    alloc_register(alloc, &a->out);
}

static void* nmmo3_encoder_create_weights(void* self) {
    Encoder* e = (Encoder*)self;
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)calloc(
        1, sizeof(NMMO3EncoderWeights));
    ew->obs_size = e->in_dim;
    ew->hidden = e->out_dim;
    return ew;
}

static void create_nmmo3_conv_encoder(Encoder* enc) {
    *enc = Encoder{
        .forward = nmmo3_encoder_forward,
        .backward = nmmo3_encoder_backward,
        .init_weights = nmmo3_encoder_init_weights,
        .reg_params = nmmo3_encoder_reg_params,
        .reg_train = nmmo3_encoder_reg_train,
        .reg_rollout = nmmo3_encoder_reg_rollout,
        .create_weights = nmmo3_encoder_create_weights,
        .in_dim = enc->in_dim, .out_dim = enc->out_dim,
        .activation_size = sizeof(NMMO3EncoderActivations),
    };
}

#ifdef N3_ATTN
// Experimental. NVCC_EXTRA=-DN3_ATTN. Did not beat conv (12.5 vs 14.3).
// Conv1+conv2 spatial stem plus ego-query attention: token 0 is self,
// remaining 15 are entities first then items. Readout Y[0] is
// concatenated onto the 256-d conv flatten.

#define N3A_TILES 165
#define N3A_CX 7
#define N3A_CY 5
#define N3A_CENTER (N3A_CY * N3_MAP_W + N3A_CX)
#define N3A_TN N3_CONV_FLAT
#define N3A_N 16
#define N3A_D 16
#define N3A_F 8
#define N3A_WARPS 8
#define N3A_EK 0
#define N3A_EIT 2
#define N3A_EET 19
#define N3A_EEL 23
#define N3A_EDL 28
#define N3A_EHP 33
#define N3A_EDX 38
#define N3A_EDY 53
#define N3A_ETR 64
#define N3A_TOK_VOCAB 70
#define N3A_TOK_N (N3A_TOK_VOCAB * N3A_D)
#define N3A_LOUD 1.0f
#define N3A_HIT 3
#define N3A_HIT_REP 8
#define N3A_HIT_N (N3A_HIT * N3A_HIT_REP)
#define N3A_RAY 8
#define N3A_AT_OFF N3A_TN
#define N3A_RAY_OFF N3A_HIT_N
#define N3A_POOL_OFF (N3A_RAY_OFF + N3A_RAY)
#define N3A_AT_N (N3A_POOL_OFF + N3A_D)
#define N3A_EM_OFF (N3A_AT_OFF + N3A_AT_N)
#define N3A_PC_OFF (N3A_EM_OFF + N3_PLAYER_EMBED)
#define N3A_RW_OFF (N3A_PC_OFF + N3_PLAYER)
#define N3A_CONCAT (N3A_RW_OFF + N3_REWARD)

__device__ __forceinline__ unsigned n3a_pack(float a, float b) {
    __nv_bfloat162 p = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<unsigned*>(&p);
}
__device__ __forceinline__ void n3a_mma16x8(
        float& d0, float& d1, float& d2, float& d3,
        unsigned a0, unsigned a1, unsigned a2, unsigned a3,
        unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void n3a_load_A(
        unsigned a[4], const precision_t* M, int lda, int lane) {
    int row = lane >> 2, col = (lane & 3) << 1;
    a[0] = n3a_pack(to_float(M[row * lda + col]), to_float(M[row * lda + col + 1]));
    a[1] = n3a_pack(to_float(M[(row + 8) * lda + col]), to_float(M[(row + 8) * lda + col + 1]));
    a[2] = n3a_pack(to_float(M[row * lda + col + 8]), to_float(M[row * lda + col + 9]));
    a[3] = n3a_pack(to_float(M[(row + 8) * lda + col + 8]), to_float(M[(row + 8) * lda + col + 9]));
}
__device__ __forceinline__ void n3a_load_B_At(
        unsigned b[2], const precision_t* M, int lda, int n0, int lane) {
    int ncol = n0 + (lane >> 2), krow = (lane & 3) << 1;
    b[0] = n3a_pack(to_float(M[ncol * lda + krow]), to_float(M[ncol * lda + krow + 1]));
    b[1] = n3a_pack(to_float(M[ncol * lda + krow + 8]), to_float(M[ncol * lda + krow + 9]));
}
__device__ __forceinline__ void n3a_load_B_A(
        unsigned b[2], const precision_t* M, int ldb, int n0, int lane) {
    int ncol = n0 + (lane >> 2), krow = (lane & 3) << 1;
    b[0] = n3a_pack(to_float(M[krow * ldb + ncol]), to_float(M[(krow + 1) * ldb + ncol]));
    b[1] = n3a_pack(to_float(M[(krow + 8) * ldb + ncol]), to_float(M[(krow + 9) * ldb + ncol]));
}
__device__ __forceinline__ void n3a_mma_ABt(
        float c0[4], float c1[4], unsigned a[4], const precision_t* B, int ldb, int lane) {
    unsigned b[2];
    n3a_load_B_At(b, B, ldb, 0, lane);
    n3a_mma16x8(c0[0], c0[1], c0[2], c0[3], a[0], a[1], a[2], a[3], b[0], b[1]);
    n3a_load_B_At(b, B, ldb, 8, lane);
    n3a_mma16x8(c1[0], c1[1], c1[2], c1[3], a[0], a[1], a[2], a[3], b[0], b[1]);
}
__device__ __forceinline__ void n3a_mma_AB(
        float c0[4], float c1[4], unsigned a[4], const precision_t* B, int ldb, int lane) {
    unsigned b[2];
    n3a_load_B_A(b, B, ldb, 0, lane);
    n3a_mma16x8(c0[0], c0[1], c0[2], c0[3], a[0], a[1], a[2], a[3], b[0], b[1]);
    n3a_load_B_A(b, B, ldb, 8, lane);
    n3a_mma16x8(c1[0], c1[1], c1[2], c1[3], a[0], a[1], a[2], a[3], b[0], b[1]);
}
__device__ __forceinline__ void n3a_store_C(
        precision_t* M, int lda, const float c0[4], const float c1[4], int lane) {
    int row = lane >> 2, col = (lane & 3) << 1;
    M[row * lda + col] = from_float(c0[0]);
    M[row * lda + col + 1] = from_float(c0[1]);
    M[(row + 8) * lda + col] = from_float(c0[2]);
    M[(row + 8) * lda + col + 1] = from_float(c0[3]);
    M[row * lda + col + 8] = from_float(c1[0]);
    M[row * lda + col + 9] = from_float(c1[1]);
    M[(row + 8) * lda + col + 8] = from_float(c1[2]);
    M[(row + 8) * lda + col + 9] = from_float(c1[3]);
}
__device__ __forceinline__ void n3a_frags_to_A(unsigned a[4], const float c0[4], const float c1[4]) {
    a[0] = n3a_pack(c0[0], c0[1]); a[1] = n3a_pack(c0[2], c0[3]);
    a[2] = n3a_pack(c1[0], c1[1]); a[3] = n3a_pack(c1[2], c1[3]);
}
__device__ __forceinline__ void n3a_softmax_16(
        float c0[4], float c1[4], int lane, int valid_n, float scale) {
    int col = (lane & 3) << 1;
    c0[0] *= scale; c0[1] *= scale; c0[2] *= scale; c0[3] *= scale;
    c1[0] *= scale; c1[1] *= scale; c1[2] *= scale; c1[3] *= scale;
    if (col >= valid_n) { c0[0] = -1.0e9f; c0[1] = -1.0e9f; c0[2] = -1.0e9f; c0[3] = -1.0e9f; }
    else if (col + 1 >= valid_n) { c0[1] = -1.0e9f; c0[3] = -1.0e9f; }
    if (col + 8 >= valid_n) { c1[0] = -1.0e9f; c1[1] = -1.0e9f; c1[2] = -1.0e9f; c1[3] = -1.0e9f; }
    else if (col + 9 >= valid_n) { c1[1] = -1.0e9f; c1[3] = -1.0e9f; }
    float m0 = fmaxf(fmaxf(c0[0], c0[1]), fmaxf(c1[0], c1[1]));
    float m8 = fmaxf(fmaxf(c0[2], c0[3]), fmaxf(c1[2], c1[3]));
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffff, m0, 1));
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffff, m0, 2));
    m8 = fmaxf(m8, __shfl_xor_sync(0xffffffff, m8, 1));
    m8 = fmaxf(m8, __shfl_xor_sync(0xffffffff, m8, 2));
    c0[0] = expf(c0[0] - m0); c0[1] = expf(c0[1] - m0);
    c1[0] = expf(c1[0] - m0); c1[1] = expf(c1[1] - m0);
    c0[2] = expf(c0[2] - m8); c0[3] = expf(c0[3] - m8);
    c1[2] = expf(c1[2] - m8); c1[3] = expf(c1[3] - m8);
    float s0 = c0[0] + c0[1] + c1[0] + c1[1];
    float s8 = c0[2] + c0[3] + c1[2] + c1[3];
    s0 += __shfl_xor_sync(0xffffffff, s0, 1);
    s0 += __shfl_xor_sync(0xffffffff, s0, 2);
    s8 += __shfl_xor_sync(0xffffffff, s8, 1);
    s8 += __shfl_xor_sync(0xffffffff, s8, 2);
    float i0 = 1.0f / fmaxf(s0, 1e-8f), i8 = 1.0f / fmaxf(s8, 1e-8f);
    c0[0] *= i0; c0[1] *= i0; c1[0] *= i0; c1[1] *= i0;
    c0[2] *= i8; c0[3] *= i8; c1[2] *= i8; c1[3] *= i8;
}
__device__ __forceinline__ void n3a_proj_tile(
        precision_t* dst, const precision_t* X, const precision_t* W, int lane) {
    unsigned a[4];
    n3a_load_A(a, X, N3A_D, lane);
    float c0[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0};
    n3a_mma_ABt(c0, c1, a, W, N3A_D, lane);
    n3a_store_C(dst, N3A_D, c0, c1, lane);
}
__device__ __forceinline__ void n3a_store_residual(
        precision_t* dst, const precision_t* x,
        const float c0[4], const float c1[4], int lane) {
    int row = lane >> 2, col = (lane & 3) << 1;
    dst[row * N3A_D + col] = from_float(c0[0] + to_float(x[row * N3A_D + col]));
    dst[row * N3A_D + col + 1] = from_float(c0[1] + to_float(x[row * N3A_D + col + 1]));
    dst[(row + 8) * N3A_D + col] = from_float(c0[2] + to_float(x[(row + 8) * N3A_D + col]));
    dst[(row + 8) * N3A_D + col + 1] = from_float(c0[3] + to_float(x[(row + 8) * N3A_D + col + 1]));
    dst[row * N3A_D + col + 8] = from_float(c1[0] + to_float(x[row * N3A_D + col + 8]));
    dst[row * N3A_D + col + 9] = from_float(c1[1] + to_float(x[row * N3A_D + col + 9]));
    dst[(row + 8) * N3A_D + col + 8] = from_float(c1[2] + to_float(x[(row + 8) * N3A_D + col + 8]));
    dst[(row + 8) * N3A_D + col + 9] = from_float(c1[3] + to_float(x[(row + 8) * N3A_D + col + 9]));
}
__device__ __forceinline__ void n3a_acc_C(
        precision_t* dst, const float c0[4], const float c1[4], int lane) {
    int row = lane >> 2, col = (lane & 3) << 1;
    dst[row * N3A_D + col] = from_float(to_float(dst[row * N3A_D + col]) + c0[0]);
    dst[row * N3A_D + col + 1] = from_float(to_float(dst[row * N3A_D + col + 1]) + c0[1]);
    dst[(row + 8) * N3A_D + col] = from_float(to_float(dst[(row + 8) * N3A_D + col]) + c0[2]);
    dst[(row + 8) * N3A_D + col + 1] = from_float(to_float(dst[(row + 8) * N3A_D + col + 1]) + c0[3]);
    dst[row * N3A_D + col + 8] = from_float(to_float(dst[row * N3A_D + col + 8]) + c1[0]);
    dst[row * N3A_D + col + 9] = from_float(to_float(dst[row * N3A_D + col + 9]) + c1[1]);
    dst[(row + 8) * N3A_D + col + 8] = from_float(to_float(dst[(row + 8) * N3A_D + col + 8]) + c1[2]);
    dst[(row + 8) * N3A_D + col + 9] = from_float(to_float(dst[(row + 8) * N3A_D + col + 9]) + c1[3]);
}
__device__ __forceinline__ void n3a_attn_layer(
        precision_t* Y, precision_t* Q, precision_t* K, precision_t* V,
        const precision_t* X, const precision_t* wq, const precision_t* wk,
        const precision_t* wv, int lane, int n_valid, float qk_scale) {
    const int D = N3A_D, N = N3A_N;
    n3a_proj_tile(Q, X, wq, lane);
    n3a_proj_tile(K, X, wk, lane);
    n3a_proj_tile(V, X, wv, lane);
    __syncwarp();
    unsigned a[4];
    n3a_load_A(a, Q, D, lane);
    float s0[4] = {0, 0, 0, 0}, s1[4] = {0, 0, 0, 0};
    n3a_mma_ABt(s0, s1, a, K, D, lane);
    n3a_softmax_16(s0, s1, lane, n_valid, qk_scale);
    n3a_frags_to_A(a, s0, s1);
    float y0[4] = {0, 0, 0, 0}, y1[4] = {0, 0, 0, 0};
    n3a_mma_AB(y0, y1, a, V, D, lane);
    n3a_store_residual(Y, X, y0, y1, lane);
    __syncwarp();
    for (int i = lane; i < N * D; i += 32) {
        if ((i >> 4) >= n_valid) Y[i] = from_float(0.0f);
    }
}
__device__ __forceinline__ void n3a_fill_slot(
        precision_t* slot, const precision_t* map, int t, int is_ent) {
    int r = t / N3_MAP_W, c = t - r * N3_MAP_W;
    const precision_t* tile = map + t * 10;
    for (int f = 0; f < N3A_F; f++) slot[f] = from_float(0.0f);
    slot[0] = from_float(1.0f);
    slot[1] = from_float(is_ent ? 1.0f : 0.0f);
    slot[2] = from_float((float)(c - N3A_CX));
    slot[3] = from_float((float)(r - N3A_CY));
    if (is_ent) {
        slot[4] = from_float(to_float(tile[4]));
        slot[5] = from_float(to_float(tile[5]));
        slot[6] = from_float(to_float(tile[6]));
        slot[7] = from_float(to_float(tile[7]));
    } else {
        slot[4] = from_float(to_float(tile[2]));
        slot[5] = from_float(to_float(tile[3]));
    }
}
__device__ __forceinline__ int n3a_clip(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
__device__ __forceinline__ void n3a_tok_embed(
        precision_t* X, const precision_t* e, const precision_t* tok, int lane) {
    const int D = N3A_D, F = N3A_F, N = N3A_N;
    for (int n = 0; n < N; n++) {
        const precision_t* s = e + n * F;
        int valid = to_float(s[0]) > 0.5f;
        int kind = n3a_clip((int)to_float(s[1]), 0, 1);
        int dx = n3a_clip((int)to_float(s[2]), -N3A_CX, N3A_CX);
        int dy = n3a_clip((int)to_float(s[3]), -N3A_CY, N3A_CY);
        int typ = (int)to_float(s[4]);
        int extra = (int)to_float(s[5]);
        int delta = n3a_clip((int)to_float(s[6]), 0, 4);
        int hp = n3a_clip((int)to_float(s[7]), 0, 4);
        for (int d = lane; d < D; d += 32) {
            float acc = 0.0f;
            if (valid) {
                acc += to_float(tok[(N3A_EK + kind) * D + d]);
                acc += to_float(tok[(N3A_EDX + dx + N3A_CX) * D + d]);
                acc += to_float(tok[(N3A_EDY + dy + N3A_CY) * D + d]);
                if (kind) {
                    acc += to_float(tok[(N3A_EET + n3a_clip(typ, 0, 3)) * D + d]);
                    acc += to_float(tok[(N3A_EEL + n3a_clip(extra, 0, 4)) * D + d]);
                    acc += to_float(tok[(N3A_EDL + delta) * D + d]);
                    acc += to_float(tok[(N3A_EHP + hp) * D + d]);
                } else {
                    acc += to_float(tok[(N3A_EIT + n3a_clip(typ, 0, 16)) * D + d]);
                    acc += to_float(tok[(N3A_ETR + n3a_clip(extra, 0, 5)) * D + d]);
                }
            }
            X[n * D + d] = from_float(acc);
        }
    }
}

__global__ void n3a_pack_kernel(
        precision_t* __restrict__ cands,
        const precision_t* __restrict__ obs, int B, int obs_size) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int b = blockIdx.x * N3A_WARPS + warp;
    if (b >= B) return;
    const precision_t* map = obs + (int64_t)b * obs_size;
    precision_t* out = cands + (int64_t)b * N3A_N * N3A_F;
    for (int i = lane; i < N3A_N * N3A_F; i += 32) out[i] = from_float(0.0f);
    unsigned used_e[6] = {0, 0, 0, 0, 0, 0};
    unsigned used_i[6] = {0, 0, 0, 0, 0, 0};
    if (lane == 0) n3a_fill_slot(out, map, N3A_CENTER, 1);
    {
        unsigned bit = 1u << (N3A_CENTER & 31);
        used_e[N3A_CENTER >> 5] |= bit;
    }
    for (int k = 1; k < N3A_N; k++) {
        float best = 1.0e9f;
        int best_t = -1;
        int best_ent = 1;
        for (int t = lane; t < N3A_TILES; t += 32) {
            int r = t / N3_MAP_W;
            int col = t - r * N3_MAP_W;
            float dist = (float)((r - N3A_CY) * (r - N3A_CY)
                + (col - N3A_CX) * (col - N3A_CX));
            int et = (int)to_float(map[t * 10 + 4]);
            unsigned bit = 1u << (t & 31);
            int bucket = t >> 5;
            if (et != 0 && (used_e[bucket] & bit) == 0 && dist < best) {
                best = dist; best_t = t;
            }
        }
        for (int off = 16; off > 0; off >>= 1) {
            float o = __shfl_xor_sync(0xffffffff, best, off);
            int ot = __shfl_xor_sync(0xffffffff, best_t, off);
            if (o < best || (o == best && ot < best_t)) {
                best = o; best_t = ot;
            }
        }
        if (best_t < 0) {
            best = 1.0e9f;
            best_ent = 0;
            for (int t = lane; t < N3A_TILES; t += 32) {
                int r = t / N3_MAP_W;
                int col = t - r * N3_MAP_W;
                float dist = (float)((r - N3A_CY) * (r - N3A_CY)
                    + (col - N3A_CX) * (col - N3A_CX));
                int item = (int)to_float(map[t * 10 + 2]);
                unsigned bit = 1u << (t & 31);
                int bucket = t >> 5;
                if (item != 0 && (used_i[bucket] & bit) == 0 && dist < best) {
                    best = dist; best_t = t;
                }
            }
            for (int off = 16; off > 0; off >>= 1) {
                float o = __shfl_xor_sync(0xffffffff, best, off);
                int ot = __shfl_xor_sync(0xffffffff, best_t, off);
                if (o < best || (o == best && ot < best_t)) {
                    best = o; best_t = ot;
                }
            }
        }
        if (best_t < 0) break;
        unsigned bit = 1u << (best_t & 31);
        int bucket = best_t >> 5;
        if (best_ent) used_e[bucket] |= bit;
        else used_i[bucket] |= bit;
        if (lane == 0) n3a_fill_slot(out + k * N3A_F, map, best_t, best_ent);
    }
}

__global__ void __launch_bounds__(256, 4) n3a_attn_fwd_kernel(
        precision_t* __restrict__ attn, precision_t* __restrict__ x0_out,
        const precision_t* __restrict__ cands, const precision_t* __restrict__ tok,
        const precision_t* __restrict__ wq, const precision_t* __restrict__ wk,
        const precision_t* __restrict__ wv, int B) {
    extern __shared__ char n3a_smem[];
    const int D = N3A_D, N = N3A_N;
    precision_t* s_wq = (precision_t*)n3a_smem;
    precision_t* s_wk = s_wq + D * D;
    precision_t* s_wv = s_wk + D * D;
    precision_t* s_tiles = s_wv + D * D;
    int tid = threadIdx.x;
    int n_w = 3 * D * D;
    for (int i = tid; i < n_w; i += blockDim.x) {
        precision_t v;
        if (i < D * D) v = wq[i];
        else if (i < 2 * D * D) v = wk[i - D * D];
        else v = wv[i - 2 * D * D];
        s_wq[i] = v;
    }
    __syncthreads();
    int warp = tid >> 5, lane = tid & 31;
    int b = blockIdx.x * N3A_WARPS + warp;
    if (b >= B) return;
    precision_t* X = s_tiles + warp * 4 * D * D;
    precision_t* Q = X + D * D;
    precision_t* K = Q + D * D;
    precision_t* V = K + D * D;
    const precision_t* e = cands + (int64_t)b * N * N3A_F;
    int n_valid = 0;
    n3a_tok_embed(X, e, tok, lane);
    __syncwarp();
    if (lane == 0) {
        for (int n = 0; n < N; n++) {
            if (to_float(e[n * N3A_F]) > 0.5f) n_valid++;
        }
    }
    n_valid = __shfl_sync(0xffffffff, n_valid, 0);
    if (n_valid < 1) n_valid = 1;
    __syncwarp();
    if (x0_out) {
        for (int i = lane; i < N * D; i += 32) x0_out[(int64_t)b * N * D + i] = X[i];
    }
    n3a_attn_layer(Q, Q, K, V, X, s_wq, s_wk, s_wv, lane, n_valid, 0.25f);
    __syncwarp();
    precision_t* gout = attn + (int64_t)b * N3A_AT_N;
    for (int i = lane; i < N3A_AT_N; i += 32) gout[i] = from_float(0.0f);
    __syncwarp();
    if (lane == 0) {
        float melee = 0.0f, sword = 0.0f, bow = 0.0f;
        for (int n = 0; n < n_valid; n++) {
            int kind = (int)to_float(e[n * N3A_F + 1]);
            int dx = (int)to_float(e[n * N3A_F + 2]);
            int dy = (int)to_float(e[n * N3A_F + 3]);
            int typ = (int)to_float(e[n * N3A_F + 4]);
            int adx = dx < 0 ? -dx : dx;
            int ady = dy < 0 ? -dy : dy;
            if (kind && typ == 2) {
                int cheb = adx > ady ? adx : ady;
                if (adx + ady == 1) melee = N3A_LOUD;
                if (cheb == 1) sword = N3A_LOUD;
                if ((dx == 0 && ady >= 1 && ady <= 4)
                        || (dy == 0 && adx >= 1 && adx <= 4))
                    bow = N3A_LOUD;
                if (dx == 0 && (ady == 3 || ady == 4))
                    gout[N3A_RAY_OFF + (ady == 4 ? 4 : 0) + (dy > 0 ? 0 : 1)]
                        = from_float(N3A_LOUD);
                if (dy == 0 && (adx == 3 || adx == 4))
                    gout[N3A_RAY_OFF + (adx == 4 ? 4 : 0) + (dx > 0 ? 2 : 3)]
                        = from_float(N3A_LOUD);
            }
        }
        for (int r = 0; r < N3A_HIT_REP; r++) {
            gout[r * N3A_HIT + 0] = from_float(melee);
            gout[r * N3A_HIT + 1] = from_float(sword);
            gout[r * N3A_HIT + 2] = from_float(bow);
        }
    }
    __syncwarp();
    if (lane < D) gout[N3A_POOL_OFF + lane] = Q[lane];
}

__global__ void __launch_bounds__(256, 4) n3a_attn_bwd_kernel(
        precision_t* __restrict__ dq, precision_t* __restrict__ dk,
        precision_t* __restrict__ dv, precision_t* __restrict__ dx,
        const precision_t* __restrict__ Xg, const precision_t* __restrict__ d_attn,
        const precision_t* __restrict__ wq, const precision_t* __restrict__ wk,
        const precision_t* __restrict__ wv, const precision_t* __restrict__ cands,
        int B, float qk_scale) {
    extern __shared__ char n3a_smem[];
    const int D = N3A_D, N = N3A_N;
    precision_t* s_wq = (precision_t*)n3a_smem;
    precision_t* s_wk = s_wq + D * D;
    precision_t* s_wv = s_wk + D * D;
    precision_t* s_tiles = s_wv + D * D;
    int tid = threadIdx.x;
    for (int i = tid; i < D * D; i += blockDim.x) {
        s_wq[i] = wq[i]; s_wk[i] = wk[i]; s_wv[i] = wv[i];
    }
    __syncthreads();
    int warp = tid >> 5, lane = tid & 31;
    int b = blockIdx.x * N3A_WARPS + warp;
    if (b >= B) return;
    precision_t* X = s_tiles + warp * 8 * D * D;
    precision_t* Q = X + D * D;
    precision_t* K = Q + D * D;
    precision_t* V = K + D * D;
    precision_t* P = V + D * D;
    precision_t* dY = P + D * D;
    precision_t* tmp = dY + D * D;
    precision_t* Pt = tmp + D * D;
    const precision_t* Xb = Xg + (int64_t)b * N * D;
    const precision_t* e = cands + (int64_t)b * N * N3A_F;
    int n_valid = 0;
    if (lane == 0) {
        for (int n = 0; n < N; n++) {
            if (to_float(e[n * N3A_F]) > 0.5f) n_valid++;
        }
    }
    n_valid = __shfl_sync(0xffffffff, n_valid, 0);
    if (n_valid < 1) n_valid = 1;
    for (int i = lane; i < D * D; i += 32)
        X[i] = (i < N * D) ? Xb[i] : from_float(0.0f);
    __syncwarp();
    n3a_proj_tile(Q, X, s_wq, lane);
    n3a_proj_tile(K, X, s_wk, lane);
    n3a_proj_tile(V, X, s_wv, lane);
    __syncwarp();
    unsigned a[4];
    n3a_load_A(a, Q, D, lane);
    float s0[4] = {0, 0, 0, 0}, s1[4] = {0, 0, 0, 0};
    n3a_mma_ABt(s0, s1, a, K, D, lane);
    n3a_softmax_16(s0, s1, lane, n_valid, qk_scale);
    n3a_store_C(P, D, s0, s1, lane);
    __syncwarp();
    const precision_t* da = d_attn + (int64_t)b * N * D;
    for (int i = lane; i < D * D; i += 32) {
        int n = i >> 4, d = i & 15;
        dY[i] = from_float((n < n_valid) ? to_float(da[i]) : 0.0f);
        Pt[d * D + n] = P[i];
    }
    __syncwarp();
    n3a_load_A(a, Pt, D, lane);
    float c0[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0};
    n3a_mma_AB(c0, c1, a, dY, D, lane);
    n3a_store_C(tmp, D, c0, c1, lane);
    __syncwarp();
    for (int i = lane; i < N * D; i += 32) dv[(int64_t)b * N * D + i] = tmp[i];
    n3a_load_A(a, dY, D, lane);
    c0[0] = c0[1] = c0[2] = c0[3] = 0;
    c1[0] = c1[1] = c1[2] = c1[3] = 0;
    n3a_mma_ABt(c0, c1, a, V, D, lane);
    int row = lane >> 2, col = (lane & 3) << 1;
    float p0[4], p1[4];
    p0[0] = to_float(P[row * D + col]);
    p0[1] = to_float(P[row * D + col + 1]);
    p0[2] = to_float(P[(row + 8) * D + col]);
    p0[3] = to_float(P[(row + 8) * D + col + 1]);
    p1[0] = to_float(P[row * D + col + 8]);
    p1[1] = to_float(P[row * D + col + 9]);
    p1[2] = to_float(P[(row + 8) * D + col + 8]);
    p1[3] = to_float(P[(row + 8) * D + col + 9]);
    float dot0 = p0[0] * c0[0] + p0[1] * c0[1] + p1[0] * c1[0] + p1[1] * c1[1];
    float dot8 = p0[2] * c0[2] + p0[3] * c0[3] + p1[2] * c1[2] + p1[3] * c1[3];
    dot0 += __shfl_xor_sync(0xffffffff, dot0, 1);
    dot0 += __shfl_xor_sync(0xffffffff, dot0, 2);
    dot8 += __shfl_xor_sync(0xffffffff, dot8, 1);
    dot8 += __shfl_xor_sync(0xffffffff, dot8, 2);
    c0[0] = p0[0] * (c0[0] - dot0); c0[1] = p0[1] * (c0[1] - dot0);
    c1[0] = p1[0] * (c1[0] - dot0); c1[1] = p1[1] * (c1[1] - dot0);
    c0[2] = p0[2] * (c0[2] - dot8); c0[3] = p0[3] * (c0[3] - dot8);
    c1[2] = p1[2] * (c1[2] - dot8); c1[3] = p1[3] * (c1[3] - dot8);
    n3a_store_C(P, D, c0, c1, lane);
    __syncwarp();
    n3a_load_A(a, P, D, lane);
    c0[0] = c0[1] = c0[2] = c0[3] = 0;
    c1[0] = c1[1] = c1[2] = c1[3] = 0;
    n3a_mma_AB(c0, c1, a, K, D, lane);
    n3a_store_C(tmp, D, c0, c1, lane);
    __syncwarp();
    for (int i = lane; i < N * D; i += 32) dq[(int64_t)b * N * D + i] = tmp[i];
    for (int i = lane; i < D * D; i += 32) {
        int r = i >> 4, c = i & 15;
        Pt[c * D + r] = P[i];
    }
    __syncwarp();
    n3a_load_A(a, Pt, D, lane);
    c0[0] = c0[1] = c0[2] = c0[3] = 0;
    c1[0] = c1[1] = c1[2] = c1[3] = 0;
    n3a_mma_AB(c0, c1, a, Q, D, lane);
    n3a_store_C(tmp, D, c0, c1, lane);
    __syncwarp();
    for (int i = lane; i < N * D; i += 32) dk[(int64_t)b * N * D + i] = tmp[i];
    for (int i = lane; i < D * D; i += 32) tmp[i] = dY[i];
    __syncwarp();
    n3a_load_A(a, dq + (int64_t)b * N * D, D, lane);
    c0[0] = c0[1] = c0[2] = c0[3] = 0;
    c1[0] = c1[1] = c1[2] = c1[3] = 0;
    n3a_mma_AB(c0, c1, a, s_wq, D, lane);
    n3a_acc_C(tmp, c0, c1, lane);
    __syncwarp();
    n3a_load_A(a, dk + (int64_t)b * N * D, D, lane);
    c0[0] = c0[1] = c0[2] = c0[3] = 0;
    c1[0] = c1[1] = c1[2] = c1[3] = 0;
    n3a_mma_AB(c0, c1, a, s_wk, D, lane);
    n3a_acc_C(tmp, c0, c1, lane);
    __syncwarp();
    n3a_load_A(a, dv + (int64_t)b * N * D, D, lane);
    c0[0] = c0[1] = c0[2] = c0[3] = 0;
    c1[0] = c1[1] = c1[2] = c1[3] = 0;
    n3a_mma_AB(c0, c1, a, s_wv, D, lane);
    n3a_acc_C(tmp, c0, c1, lane);
    __syncwarp();
    for (int i = lane; i < N * D; i += 32)
        dx[(int64_t)b * N * D + i] = tmp[i];
}

__global__ void n3a_tok_embed_bwd_kernel(
        long long* __restrict__ tok_wgrad_i,
        const precision_t* __restrict__ dx, const precision_t* __restrict__ cands,
        int B) {
    __shared__ long long acc[N3A_TOK_N];
    for (int i = threadIdx.x; i < N3A_TOK_N; i += blockDim.x) acc[i] = 0;
    __syncthreads();
    int total = B * N3A_N;
    const int D = N3A_D, F = N3A_F;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
            idx += blockDim.x * gridDim.x) {
        int b = idx / N3A_N, n = idx % N3A_N;
        const precision_t* s = cands + ((int64_t)b * N3A_N + n) * F;
        if (to_float(s[0]) <= 0.5f) continue;
        int kind = n3a_clip((int)to_float(s[1]), 0, 1);
        int dxv = n3a_clip((int)to_float(s[2]), -N3A_CX, N3A_CX);
        int dyv = n3a_clip((int)to_float(s[3]), -N3A_CY, N3A_CY);
        int typ = (int)to_float(s[4]);
        int extra = (int)to_float(s[5]);
        int delta = n3a_clip((int)to_float(s[6]), 0, 4);
        int hp = n3a_clip((int)to_float(s[7]), 0, 4);
        const precision_t* g = dx + ((int64_t)b * N3A_N + n) * D;
        int ids[8];
        int nids = 0;
        ids[nids++] = N3A_EK + kind;
        ids[nids++] = N3A_EDX + dxv + N3A_CX;
        ids[nids++] = N3A_EDY + dyv + N3A_CY;
        if (kind) {
            ids[nids++] = N3A_EET + n3a_clip(typ, 0, 3);
            ids[nids++] = N3A_EEL + n3a_clip(extra, 0, 4);
            ids[nids++] = N3A_EDL + delta;
            ids[nids++] = N3A_EHP + hp;
        } else {
            ids[nids++] = N3A_EIT + n3a_clip(typ, 0, 16);
            ids[nids++] = N3A_ETR + n3a_clip(extra, 0, 5);
        }
        for (int k = 0; k < nids; k++) {
            int base = ids[k] * D;
            for (int d = 0; d < D; d++) {
                atomicAdd((unsigned long long*)&acc[base + d],
                    (unsigned long long)(long long)__float2ll_rn(
                        to_float(g[d]) * N3_FXP));
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < N3A_TOK_N; i += blockDim.x) {
        if (acc[i] != 0) {
            atomicAdd((unsigned long long*)&tok_wgrad_i[i],
                (unsigned long long)acc[i]);
        }
    }
}

__global__ void n3a_concat_kernel(
        precision_t* __restrict__ out, const precision_t* __restrict__ conv_flat,
        const precision_t* __restrict__ attn, const precision_t* __restrict__ embed,
        const precision_t* __restrict__ obs, int B, int obs_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N3A_CONCAT) return;
    int b = idx / N3A_CONCAT, c = idx - b * N3A_CONCAT;
    precision_t val;
    if (c < N3A_TN) val = conv_flat[b * N3A_TN + c];
    else if (c < N3A_EM_OFF) val = attn[b * N3A_AT_N + (c - N3A_AT_OFF)];
    else if (c < N3A_PC_OFF) val = embed[b * N3_PLAYER_EMBED + (c - N3A_EM_OFF)];
    else if (c < N3A_RW_OFF) {
        val = obs[b * obs_size + N3_MAP_SIZE + (c - N3A_PC_OFF)];
    } else {
        val = obs[b * obs_size + obs_size - N3_REWARD + (c - N3A_RW_OFF)];
    }
    out[idx] = val;
}

__global__ void n3a_scatter_bwd_kernel(
        precision_t* __restrict__ dy, const precision_t* __restrict__ d_attn,
        const precision_t* __restrict__ cands, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tot = B * N3A_N * N3A_D;
    if (idx >= tot) return;
    int d = idx % N3A_D;
    int n = (idx / N3A_D) % N3A_N;
    int b = idx / (N3A_N * N3A_D);
    (void)cands;
    float g = 0.0f;
    if (n == 0) {
        g = to_float(d_attn[(int64_t)b * N3A_AT_N + N3A_POOL_OFF + d]);
    }
    dy[idx] = from_float(g);
}

__global__ void n3a_concat_bwd_kernel(
        precision_t* __restrict__ d_th, precision_t* __restrict__ d_attn,
        const precision_t* __restrict__ d_concat, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N3A_CONCAT) return;
    int b = idx / N3A_CONCAT, c = idx - b * N3A_CONCAT;
    if (c < N3A_TN) d_th[b * N3A_TN + c] = d_concat[idx];
    else if (c < N3A_EM_OFF)
        d_attn[b * N3A_AT_N + (c - N3A_AT_OFF)] = d_concat[idx];
}

__global__ void n3a_embedding_backward_kernel(
        long long* __restrict__ embed_wgrad_i,
        const precision_t* __restrict__ concat_grad,
        const precision_t* __restrict__ obs, int B, int obs_size) {
    __shared__ long long acc[N3_EMBED_N];
    for (int i = threadIdx.x; i < N3_EMBED_N; i += blockDim.x) acc[i] = 0;
    __syncthreads();
    int total = B * N3_PLAYER;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
            idx += blockDim.x * gridDim.x) {
        int b = idx / N3_PLAYER, f = idx % N3_PLAYER;
        int val = (int)to_float(obs[b * obs_size + N3_MAP_SIZE + f]);
        const precision_t* g = concat_grad + b * N3A_CONCAT + N3A_EM_OFF + f * N3_EMBED_DIM;
#pragma unroll
        for (int d = 0; d < N3_EMBED_DIM; d++) {
            atomicAdd((unsigned long long*)&acc[val * N3_EMBED_DIM + d],
                (unsigned long long)(long long)__float2ll_rn(to_float(g[d]) * N3_FXP));
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < N3_EMBED_N; i += blockDim.x) {
        if (acc[i] != 0) {
            atomicAdd((unsigned long long*)&embed_wgrad_i[i], (unsigned long long)acc[i]);
        }
    }
}

struct NMMO3AttnWeights {
    Prec conv1_w, conv2_w, tok_w, wq, wk, wv, embed_w, proj_w;
    int obs_size, hidden;
};
struct NMMO3AttnActs {
    Prec conv1_out, conv1_grad, col1, mm1;
    Prec conv2_out, conv2_grad, col2, mm2;
    Prec cands, x0, attn, embed_out, concat, out, saved_obs;
    Prec dq, dk, dv, dx, d_attn, d_y;
    Prec conv1_wgrad, conv2_wgrad, tok_g, wq_g, wk_g, wv_g, embed_wgrad, proj_wgrad;
    Long embed_wgrad_i, tok_wgrad_i;
};

static size_t n3a_attn_smem() {
    return (size_t)(3 * N3A_D * N3A_D + N3A_WARPS * 4 * N3A_D * N3A_D)
        * sizeof(precision_t);
}
static size_t n3a_bwd_smem() {
    return (size_t)(3 * N3A_D * N3A_D + N3A_WARPS * 8 * N3A_D * N3A_D) * sizeof(precision_t);
}

static Prec n3a_forward(void* w, void* acts, Prec input, cudaStream_t stream) {
    NMMO3AttnWeights* ew = (NMMO3AttnWeights*)w;
    NMMO3AttnActs* a = (NMMO3AttnActs*)acts;
    int B = input.shape[0];
    if (a->saved_obs.data) puf_copy(&a->saved_obs, &input, stream);
    int grid = (B + N3A_WARPS - 1) / N3A_WARPS;
    n3a_pack_kernel<<<grid, N3A_WARPS * 32, 0, stream>>>(
        a->cands.data, input.data, B, ew->obs_size);

    int c1_rows = B * N3_C1_SPATIAL;
    n3_c1_im2col_obs<<<grid_size(c1_rows * N3_C1_COL_W), BLOCK_SIZE, 0, stream>>>(
        input.data, a->col1.data, B, ew->obs_size);
    Prec c1_col = {.data = a->col1.data, .shape = {c1_rows, N3_C1_COL_W}};
    Prec c1_mm = {.data = a->mm1.data, .shape = {c1_rows, N3_C1_OC}};
    puf_mm(&c1_col, &ew->conv1_w, &c1_mm, stream);
    n3_rows_to_nchw<<<grid_size(B * N3_C1_OC * N3_C1_SPATIAL), BLOCK_SIZE, 0,
        stream>>>(a->mm1.data, a->conv1_out.data, B, N3_C1_OC, N3_C1_SPATIAL, 1);

    int c2_rows = B * N3_C2_SPATIAL;
    n3_c2_im2col<<<grid_size(c2_rows * N3_C2_COL_W), BLOCK_SIZE, 0, stream>>>(
        a->conv1_out.data, a->col2.data, B);
    Prec c2_col = {.data = a->col2.data, .shape = {c2_rows, N3_C2_COL_W}};
    Prec c2_mm = {.data = a->mm2.data, .shape = {c2_rows, N3_C2_OC}};
    puf_mm(&c2_col, &ew->conv2_w, &c2_mm, stream);
    n3_rows_to_nchw<<<grid_size(B * N3_C2_OC * N3_C2_SPATIAL), BLOCK_SIZE, 0,
        stream>>>(a->mm2.data, a->conv2_out.data, B, N3_C2_OC, N3_C2_SPATIAL, 0);

    n3a_attn_fwd_kernel<<<grid, N3A_WARPS * 32, n3a_attn_smem(), stream>>>(
        a->attn.data, a->x0.data, a->cands.data, ew->tok_w.data,
        ew->wq.data, ew->wk.data, ew->wv.data, B);
    n3_embedding_kernel<<<grid_size(B * N3_PLAYER), BLOCK_SIZE, 0, stream>>>(
        a->embed_out.data, input.data, ew->embed_w.data, B, ew->obs_size);
    n3a_concat_kernel<<<grid_size(B * N3A_CONCAT), BLOCK_SIZE, 0, stream>>>(
        a->concat.data, a->conv2_out.data, a->attn.data, a->embed_out.data,
        input.data, B, ew->obs_size);
    puf_mm(&a->concat, &ew->proj_w, &a->out, stream);
    n3_relu_kernel<<<grid_size(B * ew->hidden), BLOCK_SIZE, 0, stream>>>(
        a->out.data, B * ew->hidden);
    return a->out;
}

static void n3a_backward(void* w, void* acts, Prec grad, cudaStream_t stream) {
    NMMO3AttnWeights* ew = (NMMO3AttnWeights*)w;
    NMMO3AttnActs* a = (NMMO3AttnActs*)acts;
    int B = grad.shape[0];
    n3_relu_backward_kernel<<<grid_size(B * ew->hidden), BLOCK_SIZE, 0, stream>>>(
        grad.data, a->out.data, B * ew->hidden);
    puf_mm_tn(&grad, &a->concat, &a->proj_wgrad, stream);
    Prec dcat = {.data = a->concat.data, .shape = {B, N3A_CONCAT}};
    puf_mm_nn(&grad, &ew->proj_w, &dcat, stream);
    n3a_concat_bwd_kernel<<<grid_size(B * N3A_CONCAT), BLOCK_SIZE, 0, stream>>>(
        a->conv2_grad.data, a->d_attn.data, a->concat.data, B);
    n3_conv_wgrad(a->conv2_grad.data, a->conv2_wgrad.data,
        a->col2.data, a->mm2.data, B, N3_C2_OC, N3_C2_SPATIAL, N3_C2_COL_W,
        stream);
    Prec mm2 = {.data = a->mm2.data, .shape = {B * N3_C2_SPATIAL, N3_C2_OC}};
    Prec col2 = {.data = a->col2.data, .shape = {B * N3_C2_SPATIAL, N3_C2_COL_W}};
    puf_mm_nn(&mm2, &ew->conv2_w, &col2, stream);
    n3_c2_col2im<<<grid_size(B * N3_C2_IN_PLANE), BLOCK_SIZE, 0, stream>>>(
        a->col2.data, a->conv1_grad.data, B);
    n3_relu_backward_kernel<<<grid_size(B * N3_C1_OC * N3_C1_SPATIAL),
        BLOCK_SIZE, 0, stream>>>(
        a->conv1_grad.data, a->conv1_out.data, B * N3_C1_OC * N3_C1_SPATIAL);
    n3_conv_wgrad(a->conv1_grad.data, a->conv1_wgrad.data,
        a->col1.data, a->mm1.data, B, N3_C1_OC, N3_C1_SPATIAL, N3_C1_COL_W,
        stream);
    cudaMemsetAsync(a->embed_wgrad_i.data, 0,
        N3_EMBED_N * sizeof(long long), stream);
    n3a_embedding_backward_kernel<<<64, BLOCK_SIZE, 0, stream>>>(
        (long long*)a->embed_wgrad_i.data, a->concat.data, a->saved_obs.data,
        B, ew->obs_size);
    n3_fxp_to_precision_kernel<<<grid_size(N3_EMBED_N), BLOCK_SIZE, 0, stream>>>(
        a->embed_wgrad.data, (long long*)a->embed_wgrad_i.data, N3_EMBED_N);
    int grid = (B + N3A_WARPS - 1) / N3A_WARPS;
    Prec dq = {.data = a->dq.data, .shape = {B, N3A_N, N3A_D}};
    Prec dk = {.data = a->dk.data, .shape = {B, N3A_N, N3A_D}};
    Prec dv = {.data = a->dv.data, .shape = {B, N3A_N, N3A_D}};
    Prec x0 = {.data = a->x0.data, .shape = {B, N3A_N, N3A_D}};
    n3a_scatter_bwd_kernel<<<grid_size(B * N3A_N * N3A_D), BLOCK_SIZE, 0, stream>>>(
        a->d_y.data, a->d_attn.data, a->cands.data, B);
    n3a_attn_bwd_kernel<<<grid, N3A_WARPS * 32, n3a_bwd_smem(), stream>>>(
        a->dq.data, a->dk.data, a->dv.data, a->dx.data, a->x0.data, a->d_y.data,
        ew->wq.data, ew->wk.data, ew->wv.data, a->cands.data, B, 0.25f);
    puf_mm_tn(&dq, &x0, &a->wq_g, stream);
    puf_mm_tn(&dk, &x0, &a->wk_g, stream);
    puf_mm_tn(&dv, &x0, &a->wv_g, stream);
    cudaMemsetAsync(a->tok_wgrad_i.data, 0, N3A_TOK_N * sizeof(long long), stream);
    n3a_tok_embed_bwd_kernel<<<64, BLOCK_SIZE, 0, stream>>>(
        (long long*)a->tok_wgrad_i.data, a->dx.data, a->cands.data, B);
    n3_fxp_to_precision_kernel<<<grid_size(N3A_TOK_N), BLOCK_SIZE, 0, stream>>>(
        a->tok_g.data, (long long*)a->tok_wgrad_i.data, N3A_TOK_N);
}

static void n3a_init_weights(void* w, uint64_t* seed, cudaStream_t stream) {
    NMMO3AttnWeights* ew = (NMMO3AttnWeights*)w;
    Prec c1 = {.data = ew->conv1_w.data, .shape = {N3_C1_OC, N3_C1_COL_W}};
    Prec c2 = {.data = ew->conv2_w.data, .shape = {N3_C2_OC, N3_C2_COL_W}};
    Prec tok = {.data = ew->tok_w.data, .shape = {N3A_TOK_VOCAB, N3A_D}};
    Prec wq = {.data = ew->wq.data, .shape = {N3A_D, N3A_D}};
    Prec wk = {.data = ew->wk.data, .shape = {N3A_D, N3A_D}};
    Prec wv = {.data = ew->wv.data, .shape = {N3A_D, N3A_D}};
    Prec proj = {.data = ew->proj_w.data, .shape = {ew->hidden, N3A_CONCAT}};
    puf_kaiming_init(&c1, 1.0f, (*seed)++, stream);
    puf_kaiming_init(&c2, 1.0f, (*seed)++, stream);
    puf_normal_init(&tok, 1.0f / sqrtf(6.0f), (*seed)++, stream);
    puf_kaiming_init(&wq, 1.0f, (*seed)++, stream);
    puf_kaiming_init(&wk, 1.0f, (*seed)++, stream);
    puf_kaiming_init(&wv, 1.0f, (*seed)++, stream);
    puf_normal_init(&ew->embed_w, 1.0f, (*seed)++, stream);
    puf_kaiming_init(&proj, 1.0f, (*seed)++, stream);
}
static void n3a_reg_params(void* w, Allocator* alloc) {
    NMMO3AttnWeights* ew = (NMMO3AttnWeights*)w;
    ew->conv1_w = {.shape = {N3_C1_OC, N3_C1_COL_W}};
    ew->conv2_w = {.shape = {N3_C2_OC, N3_C2_COL_W}};
    ew->tok_w = {.shape = {N3A_TOK_VOCAB, N3A_D}};
    ew->wq = {.shape = {N3A_D, N3A_D}};
    ew->wk = {.shape = {N3A_D, N3A_D}};
    ew->wv = {.shape = {N3A_D, N3A_D}};
    ew->embed_w = {.shape = {N3_EMBED_VOCAB, N3_EMBED_DIM}};
    ew->proj_w = {.shape = {ew->hidden, N3A_CONCAT}};
    alloc_register(alloc, &ew->conv1_w);
    alloc_register(alloc, &ew->conv2_w);
    alloc_register(alloc, &ew->tok_w);
    alloc_register(alloc, &ew->wq);
    alloc_register(alloc, &ew->wk);
    alloc_register(alloc, &ew->wv);
    alloc_register(alloc, &ew->embed_w);
    alloc_register(alloc, &ew->proj_w);
}
static void n3a_reg_train(void* w, void* acts, Allocator* ac, Allocator* gr, int B) {
    NMMO3AttnWeights* ew = (NMMO3AttnWeights*)w;
    NMMO3AttnActs* a = (NMMO3AttnActs*)acts;
    *a = {};
    a->conv1_out = {.shape = {B * N3_C1_OC * N3_C1_SPATIAL}};
    a->conv1_grad = {.shape = {B * N3_C1_OC * N3_C1_SPATIAL}};
    a->col1 = {.shape = {B * N3_C1_SPATIAL, N3_C1_COL_W}};
    a->mm1 = {.shape = {B * N3_C1_SPATIAL, N3_C1_OC}};
    a->conv2_out = {.shape = {B * N3_C2_OC * N3_C2_SPATIAL}};
    a->conv2_grad = {.shape = {B * N3_C2_OC * N3_C2_SPATIAL}};
    a->col2 = {.shape = {B * N3_C2_SPATIAL, N3_C2_COL_W}};
    a->mm2 = {.shape = {B * N3_C2_SPATIAL, N3_C2_OC}};
    a->cands = {.shape = {B, N3A_N, N3A_F}};
    a->x0 = {.shape = {B, N3A_N, N3A_D}};
    a->attn = {.shape = {B, N3A_AT_N}};
    a->embed_out = {.shape = {B, N3_PLAYER_EMBED}};
    a->concat = {.shape = {B, N3A_CONCAT}};
    a->out = {.shape = {B, ew->hidden}};
    a->saved_obs = {.shape = {B, ew->obs_size}};
    a->dq = {.shape = {B, N3A_N, N3A_D}};
    a->dk = {.shape = {B, N3A_N, N3A_D}};
    a->dv = {.shape = {B, N3A_N, N3A_D}};
    a->dx = {.shape = {B, N3A_N, N3A_D}};
    a->d_attn = {.shape = {B, N3A_AT_N}};
    a->d_y = {.shape = {B, N3A_N, N3A_D}};
    alloc_register(ac, &a->conv1_out); alloc_register(ac, &a->conv1_grad);
    alloc_register(ac, &a->col1); alloc_register(ac, &a->mm1);
    alloc_register(ac, &a->conv2_out); alloc_register(ac, &a->conv2_grad);
    alloc_register(ac, &a->col2); alloc_register(ac, &a->mm2);
    alloc_register(ac, &a->cands); alloc_register(ac, &a->x0);
    alloc_register(ac, &a->attn);
    alloc_register(ac, &a->embed_out);
    alloc_register(ac, &a->concat); alloc_register(ac, &a->out);
    alloc_register(ac, &a->saved_obs);
    alloc_register(ac, &a->dq); alloc_register(ac, &a->dk);
    alloc_register(ac, &a->dv); alloc_register(ac, &a->dx);
    alloc_register(ac, &a->d_attn); alloc_register(ac, &a->d_y);
    a->conv1_wgrad = {.shape = {N3_C1_OC, N3_C1_COL_W}};
    a->conv2_wgrad = {.shape = {N3_C2_OC, N3_C2_COL_W}};
    a->tok_g = {.shape = {N3A_TOK_VOCAB, N3A_D}};
    a->tok_wgrad_i = {.shape = {N3A_TOK_VOCAB, N3A_D}};
    a->wq_g = {.shape = {N3A_D, N3A_D}};
    a->wk_g = {.shape = {N3A_D, N3A_D}};
    a->wv_g = {.shape = {N3A_D, N3A_D}};
    a->embed_wgrad = {.shape = {N3_EMBED_VOCAB, N3_EMBED_DIM}};
    a->embed_wgrad_i = {.shape = {N3_EMBED_VOCAB, N3_EMBED_DIM}};
    a->proj_wgrad = {.shape = {ew->hidden, N3A_CONCAT}};
    alloc_register(gr, &a->conv1_wgrad);
    alloc_register(gr, &a->conv2_wgrad);
    alloc_register(gr, &a->tok_g);
    alloc_register(ac, &a->tok_wgrad_i);
    alloc_register(gr, &a->wq_g); alloc_register(gr, &a->wk_g);
    alloc_register(gr, &a->wv_g);
    alloc_register(gr, &a->embed_wgrad);
    alloc_register(ac, &a->embed_wgrad_i); alloc_register(gr, &a->proj_wgrad);
}
static void n3a_reg_rollout(void* w, void* acts, Allocator* alloc, int B) {
    NMMO3AttnWeights* ew = (NMMO3AttnWeights*)w;
    NMMO3AttnActs* a = (NMMO3AttnActs*)acts;
    *a = {};
    a->conv1_out = {.shape = {B * N3_C1_OC * N3_C1_SPATIAL}};
    a->col1 = {.shape = {B * N3_C1_SPATIAL, N3_C1_COL_W}};
    a->mm1 = {.shape = {B * N3_C1_SPATIAL, N3_C1_OC}};
    a->conv2_out = {.shape = {B * N3_C2_OC * N3_C2_SPATIAL}};
    a->col2 = {.shape = {B * N3_C2_SPATIAL, N3_C2_COL_W}};
    a->mm2 = {.shape = {B * N3_C2_SPATIAL, N3_C2_OC}};
    a->cands = {.shape = {B, N3A_N, N3A_F}};
    a->attn = {.shape = {B, N3A_AT_N}};
    a->embed_out = {.shape = {B, N3_PLAYER_EMBED}};
    a->concat = {.shape = {B, N3A_CONCAT}};
    a->out = {.shape = {B, ew->hidden}};
    alloc_register(alloc, &a->conv1_out);
    alloc_register(alloc, &a->col1); alloc_register(alloc, &a->mm1);
    alloc_register(alloc, &a->conv2_out);
    alloc_register(alloc, &a->col2); alloc_register(alloc, &a->mm2);
    alloc_register(alloc, &a->cands); alloc_register(alloc, &a->attn);
    alloc_register(alloc, &a->embed_out); alloc_register(alloc, &a->concat);
    alloc_register(alloc, &a->out);
}
static void* n3a_create_weights(void* self) {
    Encoder* e = (Encoder*)self;
    NMMO3AttnWeights* ew = (NMMO3AttnWeights*)calloc(1, sizeof(NMMO3AttnWeights));
    ew->obs_size = e->in_dim;
    ew->hidden = e->out_dim;
    return ew;
}

static void create_nmmo3_attn_encoder(Encoder* enc) {
    cudaFuncSetAttribute((const void*)n3a_attn_fwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)n3a_attn_smem());
    cudaFuncSetAttribute((const void*)n3a_attn_bwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)n3a_bwd_smem());
    *enc = Encoder{
        .forward = n3a_forward,
        .backward = n3a_backward,
        .init_weights = n3a_init_weights,
        .reg_params = n3a_reg_params,
        .reg_train = n3a_reg_train,
        .reg_rollout = n3a_reg_rollout,
        .create_weights = n3a_create_weights,
        .in_dim = enc->in_dim, .out_dim = enc->out_dim,
        .activation_size = sizeof(NMMO3AttnActs),
    };
}
#endif
