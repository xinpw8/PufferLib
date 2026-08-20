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

static void create_nmmo3_encoder(Encoder* enc) {
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
