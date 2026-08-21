// Minimal entity encoder (points + max pool).
// Included by src/ocean.cu — requires precision_t, Prec, Allocator, puf_mm, etc.

// ---- Minimal entity encoder ----

static constexpr int ME_SELF_DIM = 2;
static constexpr int ME_POINT_DIM = 4;
static constexpr int ME_NUM_POINTS = 16;
static constexpr int ME_ENTITY_IN = ME_SELF_DIM + ME_POINT_DIM;
static constexpr int ME_ENTITY_HIDDEN = 16;
static constexpr int ME_OBS_SIZE = ME_SELF_DIM + ME_NUM_POINTS * ME_POINT_DIM;
static constexpr int ME_BATCH_TILE = 8;
static constexpr int ME_HIDDEN_TILE = 32;
static constexpr int ME_FC_THREADS = ME_BATCH_TILE * ME_HIDDEN_TILE;

struct MinimalEntityEncoderWeights {
    Prec input_w, output_w;
    int obs_size, hidden;
};

struct MinimalEntityEncoderActivations {
    Prec point_input, entity_hidden, out, grad_entity;
    Prec input_wgrad, output_wgrad;
    Int argmax;
};

__global__ void me_materialize_points_kernel(
        precision_t* __restrict__ point_input,
        const precision_t* __restrict__ obs,
        int B, int obs_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * ME_NUM_POINTS * ME_ENTITY_IN;
    if (idx >= total) return;

    int d = idx % ME_ENTITY_IN;
    int point_idx = (idx / ME_ENTITY_IN) % ME_NUM_POINTS;
    int b = idx / (ME_NUM_POINTS * ME_ENTITY_IN);
    int obs_idx = (d < ME_SELF_DIM)
        ? d
        : ME_SELF_DIM + point_idx * ME_POINT_DIM + (d - ME_SELF_DIM);
    point_input[idx] = obs[(int64_t)b * obs_size + obs_idx];
}

__global__ void me_linear_max_kernel(
        precision_t* __restrict__ output,
        int* __restrict__ argmax,
        const precision_t* __restrict__ entity_hidden,
        const precision_t* __restrict__ weight,
        int B, int hidden) {
    extern __shared__ precision_t shared[];
    precision_t* point_tile = shared;
    precision_t* weight_tile = point_tile + ME_BATCH_TILE * ME_NUM_POINTS * ME_ENTITY_HIDDEN;

    int batch_base = blockIdx.x * ME_BATCH_TILE;
    int hidden_base = blockIdx.y * ME_HIDDEN_TILE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * ME_HIDDEN_TILE + tx;
    int b = batch_base + ty;
    int h = hidden_base + tx;

    int point_values = ME_BATCH_TILE * ME_NUM_POINTS * ME_ENTITY_HIDDEN;
    for (int idx = tid; idx < point_values; idx += ME_FC_THREADS) {
        int batch_tile_idx = idx / (ME_NUM_POINTS * ME_ENTITY_HIDDEN);
        int rem = idx - batch_tile_idx * ME_NUM_POINTS * ME_ENTITY_HIDDEN;
        int global_b = batch_base + batch_tile_idx;
        point_tile[idx] = global_b < B
            ? from_float(fmaxf(0.0f, to_float(
                entity_hidden[(int64_t)global_b * ME_NUM_POINTS * ME_ENTITY_HIDDEN + rem])))
            : from_float(0.0f);
    }

    int weight_values = ME_HIDDEN_TILE * ME_ENTITY_HIDDEN;
    for (int idx = tid; idx < weight_values; idx += ME_FC_THREADS) {
        int d = idx / ME_HIDDEN_TILE;
        int tile_h = idx - d * ME_HIDDEN_TILE;
        int global_h = hidden_base + tile_h;
        weight_tile[idx] = global_h < hidden
            ? weight[(int64_t)global_h * ME_ENTITY_HIDDEN + d]
            : from_float(0.0f);
    }
    __syncthreads();

    if (b < B && h < hidden) {
        float max_val = -3.4028234663852886e38f;
        int best_point = 0;
#pragma unroll
        for (int point_idx = 0; point_idx < ME_NUM_POINTS; ++point_idx) {
            const precision_t* point = point_tile
                + ((int64_t)ty * ME_NUM_POINTS + point_idx) * ME_ENTITY_HIDDEN;
            float sum = 0.0f;
#pragma unroll
            for (int d = 0; d < ME_ENTITY_HIDDEN; ++d) {
                sum += to_float(weight_tile[d * ME_HIDDEN_TILE + tx]) * to_float(point[d]);
            }
            if (sum > max_val) {
                max_val = sum;
                best_point = point_idx;
            }
        }
        output[(int64_t)b * hidden + h] = from_float(max_val);
        if (argmax) argmax[(int64_t)b * hidden + h] = best_point;
    }
}

__global__ void me_output_wgrad_kernel(
        precision_t* __restrict__ wgrad,
        const precision_t* __restrict__ grad_out,
        const precision_t* __restrict__ entity_hidden,
        const int* __restrict__ argmax,
        int B, int hidden) {
    int h = blockIdx.x;
    if (h >= hidden) return;

    float sum[ME_ENTITY_HIDDEN];
#pragma unroll
    for (int k = 0; k < ME_ENTITY_HIDDEN; ++k) sum[k] = 0.0f;

    for (int b = threadIdx.x; b < B; b += blockDim.x) {
        int point_idx = argmax[(int64_t)b * hidden + h];
        float g = to_float(grad_out[(int64_t)b * hidden + h]);
        const precision_t* point = entity_hidden
            + ((int64_t)b * ME_NUM_POINTS + point_idx) * ME_ENTITY_HIDDEN;
#pragma unroll
        for (int k = 0; k < ME_ENTITY_HIDDEN; ++k) {
            sum[k] += g * fmaxf(0.0f, to_float(point[k]));
        }
    }

    __shared__ float warp_sums[ME_ENTITY_HIDDEN * 32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int num_warps = (blockDim.x + 31) >> 5;
#pragma unroll
    for (int k = 0; k < ME_ENTITY_HIDDEN; ++k) {
        float s = sum[k];
        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_down_sync(0xffffffff, s, offset);
        if (lane == 0) warp_sums[k * 32 + warp] = s;
    }
    __syncthreads();

    if (warp == 0) {
#pragma unroll
        for (int k = 0; k < ME_ENTITY_HIDDEN; ++k) {
            float s = lane < num_warps ? warp_sums[k * 32 + lane] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1)
                s += __shfl_down_sync(0xffffffff, s, offset);
            if (lane == 0) wgrad[(int64_t)h * ME_ENTITY_HIDDEN + k] = from_float(s);
        }
    }
}

__global__ void me_grad_entity_kernel(
        precision_t* __restrict__ grad_entity,
        const precision_t* __restrict__ grad_out,
        const precision_t* __restrict__ output_w,
        const precision_t* __restrict__ entity_hidden,
        const int* __restrict__ argmax,
        int B, int hidden) {
    int b = blockIdx.x;
    if (b >= B) return;

    __shared__ float accum[ME_NUM_POINTS * ME_ENTITY_HIDDEN];
    __shared__ int arg_s[BLOCK_SIZE];
    __shared__ float grad_s[BLOCK_SIZE];

    for (int idx = threadIdx.x; idx < ME_NUM_POINTS * ME_ENTITY_HIDDEN; idx += blockDim.x) {
        accum[idx] = 0.0f;
    }
    __syncthreads();

    for (int base = 0; base < hidden; base += blockDim.x) {
        int h = base + threadIdx.x;
        if (h < hidden) {
            arg_s[threadIdx.x] = argmax[(int64_t)b * hidden + h];
            grad_s[threadIdx.x] = to_float(grad_out[(int64_t)b * hidden + h]);
        }
        __syncthreads();

        int tile = hidden - base;
        if (tile > blockDim.x) tile = blockDim.x;
        for (int idx = threadIdx.x; idx < tile * ME_ENTITY_HIDDEN; idx += blockDim.x) {
            int j = idx / ME_ENTITY_HIDDEN;
            int k = idx - j * ME_ENTITY_HIDDEN;
            int point_idx = arg_s[j];
            float g = grad_s[j] * to_float(output_w[(int64_t)(base + j) * ME_ENTITY_HIDDEN + k]);
            atomicAdd(&accum[point_idx * ME_ENTITY_HIDDEN + k], g);
        }
        __syncthreads();
    }

    for (int idx = threadIdx.x; idx < ME_NUM_POINTS * ME_ENTITY_HIDDEN; idx += blockDim.x) {
        int point_idx = idx / ME_ENTITY_HIDDEN;
        int k = idx - point_idx * ME_ENTITY_HIDDEN;
        int64_t offset = ((int64_t)b * ME_NUM_POINTS + point_idx) * ME_ENTITY_HIDDEN + k;
        float g = to_float(entity_hidden[offset]) > 0.0f ? accum[idx] : 0.0f;
        grad_entity[offset] = from_float(g);
    }
}

static MinimalEntityEncoderWeights* me_encoder_create(int obs_size, int hidden) {
    assert(obs_size == ME_OBS_SIZE && "minimal entity encoder expects the minimal observation layout");
    MinimalEntityEncoderWeights* ew =
        (MinimalEntityEncoderWeights*)calloc(1, sizeof(MinimalEntityEncoderWeights));
    ew->obs_size = obs_size;
    ew->hidden = hidden;
    return ew;
}

static Prec me_encoder_forward(void* w, void* activations, Prec input, cudaStream_t stream) {
    MinimalEntityEncoderWeights* ew = (MinimalEntityEncoderWeights*)w;
    MinimalEntityEncoderActivations* a = (MinimalEntityEncoderActivations*)activations;
    int B = input.shape[0];

    me_materialize_points_kernel<<<grid_size(B * ME_NUM_POINTS * ME_ENTITY_IN), BLOCK_SIZE, 0, stream>>>(
        a->point_input.data, input.data, B, ew->obs_size);
    Prec point_input = {.data = a->point_input.data, .shape = {B, ME_NUM_POINTS, ME_ENTITY_IN}};
    Prec entity_hidden = {.data = a->entity_hidden.data, .shape = {B, ME_NUM_POINTS, ME_ENTITY_HIDDEN}};
    puf_mm(&point_input, &ew->input_w, &entity_hidden, stream);

    dim3 block(ME_HIDDEN_TILE, ME_BATCH_TILE);
    dim3 grid((B + ME_BATCH_TILE - 1) / ME_BATCH_TILE,
        (ew->hidden + ME_HIDDEN_TILE - 1) / ME_HIDDEN_TILE);
    size_t shared_bytes = (
        (size_t)ME_BATCH_TILE * ME_NUM_POINTS * ME_ENTITY_HIDDEN +
        (size_t)ME_HIDDEN_TILE * ME_ENTITY_HIDDEN) * sizeof(precision_t);
    me_linear_max_kernel<<<grid, block, shared_bytes, stream>>>(
        a->out.data, a->argmax.data, a->entity_hidden.data,
        ew->output_w.data, B, ew->hidden);
    return a->out;
}

static void me_encoder_backward(void* w, void* activations, Prec grad, cudaStream_t stream) {
    MinimalEntityEncoderWeights* ew = (MinimalEntityEncoderWeights*)w;
    MinimalEntityEncoderActivations* a = (MinimalEntityEncoderActivations*)activations;
    int B = grad.shape[0];

    me_output_wgrad_kernel<<<ew->hidden, 256, 0, stream>>>(
        a->output_wgrad.data, grad.data, a->entity_hidden.data, a->argmax.data, B, ew->hidden);

    me_grad_entity_kernel<<<B, BLOCK_SIZE, 0, stream>>>(
        a->grad_entity.data, grad.data, ew->output_w.data,
        a->entity_hidden.data, a->argmax.data, B, ew->hidden);

    puf_mm_tn(&a->grad_entity, &a->point_input, &a->input_wgrad, stream);
}

static void me_encoder_init_weights(void* w, uint64_t* seed, cudaStream_t stream) {
    MinimalEntityEncoderWeights* ew = (MinimalEntityEncoderWeights*)w;
    Prec input_w = {.data = ew->input_w.data, .shape = {ME_ENTITY_HIDDEN, ME_ENTITY_IN}};
    Prec output_w = {.data = ew->output_w.data, .shape = {ew->hidden, ME_ENTITY_HIDDEN}};
    puf_kaiming_init(&input_w, std::sqrt(2.0f), (*seed)++, stream);
    puf_kaiming_init(&output_w, 1.0f, (*seed)++, stream);
}

static void me_encoder_reg_params(void* w, Allocator* alloc) {
    MinimalEntityEncoderWeights* ew = (MinimalEntityEncoderWeights*)w;
    ew->input_w = {.shape = {ME_ENTITY_HIDDEN, ME_ENTITY_IN}};
    ew->output_w = {.shape = {ew->hidden, ME_ENTITY_HIDDEN}};
    alloc_register(alloc, &ew->input_w);
    alloc_register(alloc, &ew->output_w);
}

static void me_encoder_reg_train(void* w, void* activations, Allocator* acts, Allocator* grads, int B_TT) {
    MinimalEntityEncoderWeights* ew = (MinimalEntityEncoderWeights*)w;
    MinimalEntityEncoderActivations* a = (MinimalEntityEncoderActivations*)activations;
    *a = {};
    a->point_input = {.shape = {B_TT, ME_NUM_POINTS, ME_ENTITY_IN}};
    a->entity_hidden = {.shape = {B_TT, ME_NUM_POINTS, ME_ENTITY_HIDDEN}};
    a->out = {.shape = {B_TT, ew->hidden}};
    a->grad_entity = {.shape = {B_TT, ME_NUM_POINTS, ME_ENTITY_HIDDEN}};
    a->argmax = {.shape = {B_TT, ew->hidden}};
    alloc_register(acts, &a->point_input);
    alloc_register(acts, &a->entity_hidden);
    alloc_register(acts, &a->out);
    alloc_register(acts, &a->grad_entity);
    alloc_register(acts, &a->argmax);

    a->input_wgrad = {.shape = {ME_ENTITY_HIDDEN, ME_ENTITY_IN}};
    a->output_wgrad = {.shape = {ew->hidden, ME_ENTITY_HIDDEN}};
    alloc_register(grads, &a->input_wgrad);
    alloc_register(grads, &a->output_wgrad);
}

static void me_encoder_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    MinimalEntityEncoderWeights* ew = (MinimalEntityEncoderWeights*)w;
    MinimalEntityEncoderActivations* a = (MinimalEntityEncoderActivations*)activations;
    *a = {};
    a->point_input = {.shape = {B, ME_NUM_POINTS, ME_ENTITY_IN}};
    a->entity_hidden = {.shape = {B, ME_NUM_POINTS, ME_ENTITY_HIDDEN}};
    a->out = {.shape = {B, ew->hidden}};
    alloc_register(alloc, &a->point_input);
    alloc_register(alloc, &a->entity_hidden);
    alloc_register(alloc, &a->out);
}

static void* me_encoder_create_weights(void* self) {
    Encoder* e = (Encoder*)self;
    return me_encoder_create(e->in_dim, e->out_dim);
}


static void create_minimal_encoder(Encoder* enc) {
    *enc = Encoder{
        .forward = me_encoder_forward,
        .backward = me_encoder_backward,
        .init_weights = me_encoder_init_weights,
        .reg_params = me_encoder_reg_params,
        .reg_train = me_encoder_reg_train,
        .reg_rollout = me_encoder_reg_rollout,
        .create_weights = me_encoder_create_weights,
        .in_dim = enc->in_dim, .out_dim = enc->out_dim,
        .activation_size = sizeof(MinimalEntityEncoderActivations),
    };
}

// Entity-attention encoder. One QKV self-attention layer over N<=64 entities
// at D=16 (PTX m16n8k16 tiles), mean-pool, then Linear D->H via puf_mm.
// Pairwise interactions live in the 16x16 MMA; H scaling is a skinny GEMM.

static constexpr int EA_D = 16;
static constexpr int EA_WARPS = 8;
static constexpr int EA_MAX_N = 64;

struct EntityAttnWeights {
    Prec win, wq, wk, wv, wout;
    int obs_size, hidden, n_entities, f_in, self_dim, point_dim;
};

struct EntityAttnActivations {
    Prec ents, x, pooled, out;
    Prec dq, dk, dv, dx;
    Prec win_g, wq_g, wk_g, wv_g, wout_g;
};

__device__ __forceinline__ unsigned ea_pack(float a, float b) {
    __nv_bfloat162 p = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<unsigned*>(&p);
}

__device__ __forceinline__ void ea_mma16x8(
        float& d0, float& d1, float& d2, float& d3,
        unsigned a0, unsigned a1, unsigned a2, unsigned a3,
        unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}

__device__ __forceinline__ void ea_load_A(
        unsigned a[4], const precision_t* M, int lda, int lane) {
    int row = lane >> 2;
    int col = (lane & 3) << 1;
    a[0] = ea_pack(to_float(M[row * lda + col]), to_float(M[row * lda + col + 1]));
    a[1] = ea_pack(to_float(M[(row + 8) * lda + col]), to_float(M[(row + 8) * lda + col + 1]));
    a[2] = ea_pack(to_float(M[row * lda + col + 8]), to_float(M[row * lda + col + 9]));
    a[3] = ea_pack(to_float(M[(row + 8) * lda + col + 8]), to_float(M[(row + 8) * lda + col + 9]));
}

__device__ __forceinline__ void ea_load_B_At(
        unsigned b[2], const precision_t* M, int lda, int n0, int lane) {
    int ncol = n0 + (lane >> 2);
    int krow = (lane & 3) << 1;
    b[0] = ea_pack(to_float(M[ncol * lda + krow]), to_float(M[ncol * lda + krow + 1]));
    b[1] = ea_pack(to_float(M[ncol * lda + krow + 8]), to_float(M[ncol * lda + krow + 9]));
}

__device__ __forceinline__ void ea_load_B_A(
        unsigned b[2], const precision_t* M, int ldb, int n0, int lane) {
    int ncol = n0 + (lane >> 2);
    int krow = (lane & 3) << 1;
    b[0] = ea_pack(to_float(M[krow * ldb + ncol]), to_float(M[(krow + 1) * ldb + ncol]));
    b[1] = ea_pack(to_float(M[(krow + 8) * ldb + ncol]), to_float(M[(krow + 9) * ldb + ncol]));
}

__device__ __forceinline__ void ea_mma_ABt(
        float c0[4], float c1[4], unsigned a[4], const precision_t* B, int ldb, int lane) {
    unsigned b[2];
    ea_load_B_At(b, B, ldb, 0, lane);
    ea_mma16x8(c0[0], c0[1], c0[2], c0[3], a[0], a[1], a[2], a[3], b[0], b[1]);
    ea_load_B_At(b, B, ldb, 8, lane);
    ea_mma16x8(c1[0], c1[1], c1[2], c1[3], a[0], a[1], a[2], a[3], b[0], b[1]);
}

__device__ __forceinline__ void ea_mma_AB(
        float c0[4], float c1[4], unsigned a[4], const precision_t* B, int ldb, int lane) {
    unsigned b[2];
    ea_load_B_A(b, B, ldb, 0, lane);
    ea_mma16x8(c0[0], c0[1], c0[2], c0[3], a[0], a[1], a[2], a[3], b[0], b[1]);
    ea_load_B_A(b, B, ldb, 8, lane);
    ea_mma16x8(c1[0], c1[1], c1[2], c1[3], a[0], a[1], a[2], a[3], b[0], b[1]);
}

__device__ __forceinline__ void ea_store_C(
        precision_t* M, int lda, const float c0[4], const float c1[4], int lane) {
    int row = lane >> 2;
    int col = (lane & 3) << 1;
    M[row * lda + col] = from_float(c0[0]);
    M[row * lda + col + 1] = from_float(c0[1]);
    M[(row + 8) * lda + col] = from_float(c0[2]);
    M[(row + 8) * lda + col + 1] = from_float(c0[3]);
    M[row * lda + col + 8] = from_float(c1[0]);
    M[row * lda + col + 9] = from_float(c1[1]);
    M[(row + 8) * lda + col + 8] = from_float(c1[2]);
    M[(row + 8) * lda + col + 9] = from_float(c1[3]);
}

__device__ __forceinline__ void ea_frags_to_A(unsigned a[4], const float c0[4], const float c1[4]) {
    a[0] = ea_pack(c0[0], c0[1]);
    a[1] = ea_pack(c0[2], c0[3]);
    a[2] = ea_pack(c1[0], c1[1]);
    a[3] = ea_pack(c1[2], c1[3]);
}

__device__ __forceinline__ void ea_softmax_16(float c0[4], float c1[4], int lane, int valid_n) {
    int col = (lane & 3) << 1;
    float scale = 0.25f;
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
    float i0 = 1.0f / s0, i8 = 1.0f / s8;
    c0[0] *= i0; c0[1] *= i0; c1[0] *= i0; c1[1] *= i0;
    c0[2] *= i8; c0[3] *= i8; c1[2] *= i8; c1[3] *= i8;
}

__device__ __forceinline__ void ea_proj_tile(
        precision_t* dst, const precision_t* X, const precision_t* W, int lane) {
    unsigned a[4];
    ea_load_A(a, X, EA_D, lane);
    float c0[4] = {0, 0, 0, 0};
    float c1[4] = {0, 0, 0, 0};
    ea_mma_ABt(c0, c1, a, W, EA_D, lane);
    ea_store_C(dst, EA_D, c0, c1, lane);
}

__device__ __forceinline__ void ea_embed_from_obs(
        precision_t* X, const precision_t* obs, const precision_t* win,
        int N, int F, int self_dim, int point_dim, int obs_size, int n_pad, int lane) {
    for (int i = lane; i < n_pad * EA_D; i += 32) {
        int n = i >> 4;
        int d = i & 15;
        float acc = 0.0f;
        if (n < N) {
            const precision_t* w = win + d * F;
            for (int f = 0; f < F; f++) {
                int obs_i = (f < self_dim) ? f
                    : self_dim + n * point_dim + (f - self_dim);
                acc += to_float(obs[obs_i]) * to_float(w[f]);
            }
            if (acc < 0.0f) acc = 0.0f;
        }
        X[i] = from_float(acc);
    }
    (void)obs_size;
}

__device__ __forceinline__ void ea_mean_pool(
        precision_t* pooled, const precision_t* Y, int N, int n_pad, int lane) {
    if (lane < EA_D) {
        float s = 0.0f;
        for (int n = 0; n < N; n++) s += to_float(Y[n * EA_D + lane]);
        pooled[lane] = from_float(s / (float)N);
    }
    (void)n_pad;
}

// One warp / sample, N<=16. Writes pooled [D] and optionally X [N,D].
__global__ void __launch_bounds__(256, 4) ea_n16_fwd_kernel(
        precision_t* __restrict__ pooled,
        precision_t* __restrict__ X_out,
        const precision_t* __restrict__ obs,
        const precision_t* __restrict__ win,
        const precision_t* __restrict__ wq,
        const precision_t* __restrict__ wk,
        const precision_t* __restrict__ wv,
        int B, int N, int F, int self_dim, int point_dim, int obs_size) {
    extern __shared__ char ea_smem[];
    const int D = EA_D;
    int win_n = D * F;
    int wd_n = D * D;
    precision_t* s_win = (precision_t*)ea_smem;
    precision_t* s_wq = s_win + win_n;
    precision_t* s_wk = s_wq + wd_n;
    precision_t* s_wv = s_wk + wd_n;
    precision_t* s_tiles = s_wv + wd_n;
    int tid = threadIdx.x;
    int n_w = win_n + 3 * wd_n;
    for (int i = tid; i < n_w; i += blockDim.x) {
        precision_t v;
        if (i < win_n) v = win[i];
        else if (i < win_n + wd_n) v = wq[i - win_n];
        else if (i < win_n + 2 * wd_n) v = wk[i - win_n - wd_n];
        else v = wv[i - win_n - 2 * wd_n];
        s_win[i] = v;
    }
    __syncthreads();

    int warp = tid >> 5;
    int lane = tid & 31;
    int b = blockIdx.x * EA_WARPS + warp;
    if (b >= B) return;

    precision_t* X = s_tiles + warp * 4 * D * D;
    precision_t* Q = X + D * D;
    precision_t* K = Q + D * D;
    precision_t* V = K + D * D;
    ea_embed_from_obs(X, obs + (int64_t)b * obs_size, s_win,
        N, F, self_dim, point_dim, obs_size, D, lane);
    __syncwarp();
    if (X_out) {
        for (int i = lane; i < N * D; i += 32) {
            X_out[(int64_t)b * N * D + i] = X[i];
        }
    }
    ea_proj_tile(Q, X, s_wq, lane);
    ea_proj_tile(K, X, s_wk, lane);
    ea_proj_tile(V, X, s_wv, lane);
    __syncwarp();

    unsigned a[4];
    ea_load_A(a, Q, D, lane);
    float s0[4] = {0, 0, 0, 0};
    float s1[4] = {0, 0, 0, 0};
    ea_mma_ABt(s0, s1, a, K, D, lane);
    ea_softmax_16(s0, s1, lane, N);
    ea_frags_to_A(a, s0, s1);
    float y0[4] = {0, 0, 0, 0};
    float y1[4] = {0, 0, 0, 0};
    ea_mma_AB(y0, y1, a, V, D, lane);

    int row = lane >> 2;
    int col = (lane & 3) << 1;
    auto addst = [&](int r, int c, float v) {
        Q[r * D + c] = from_float(v + to_float(X[r * D + c]));
    };
    addst(row, col, y0[0]);
    addst(row, col + 1, y0[1]);
    addst(row + 8, col, y0[2]);
    addst(row + 8, col + 1, y0[3]);
    addst(row, col + 8, y1[0]);
    addst(row, col + 9, y1[1]);
    addst(row + 8, col + 8, y1[2]);
    addst(row + 8, col + 9, y1[3]);
    __syncwarp();
    ea_mean_pool(pooled + (int64_t)b * D, Q, N, D, lane);
}

__global__ void ea_flash_fwd_kernel(
        precision_t* __restrict__ pooled,
        precision_t* __restrict__ X_out,
        const precision_t* __restrict__ obs,
        const precision_t* __restrict__ win,
        const precision_t* __restrict__ wq,
        const precision_t* __restrict__ wk,
        const precision_t* __restrict__ wv,
        int B, int N, int F, int self_dim, int point_dim, int obs_size) {
    extern __shared__ char ea_smem[];
    const int D = EA_D;
    int n_pad = (N + 15) & ~15;
    int win_n = D * F;
    int wd_n = D * D;
    precision_t* s_win = (precision_t*)ea_smem;
    precision_t* s_wq = s_win + win_n;
    precision_t* s_wk = s_wq + wd_n;
    precision_t* s_wv = s_wk + wd_n;
    precision_t* s_x = s_wv + wd_n;
    precision_t* s_y = s_x + EA_WARPS * n_pad * D;
    precision_t* s_kv = s_y + EA_WARPS * n_pad * D;
    int tid = threadIdx.x;
    int n_w = win_n + 3 * wd_n;
    for (int i = tid; i < n_w; i += blockDim.x) {
        precision_t v;
        if (i < win_n) v = win[i];
        else if (i < win_n + wd_n) v = wq[i - win_n];
        else if (i < win_n + 2 * wd_n) v = wk[i - win_n - wd_n];
        else v = wv[i - win_n - 2 * wd_n];
        s_win[i] = v;
    }
    __syncthreads();

    int warp = tid >> 5;
    int lane = tid & 31;
    int b = blockIdx.x * EA_WARPS + warp;
    if (b >= B) return;
    precision_t* X = s_x + warp * n_pad * D;
    precision_t* Y = s_y + warp * n_pad * D;
    precision_t* Kt_s = s_kv + warp * 2 * D * D;
    precision_t* Vt_s = Kt_s + D * D;
    precision_t* Kt = Kt_s;
    precision_t* Vt = Vt_s;
    ea_embed_from_obs(X, obs + (int64_t)b * obs_size, s_win,
        N, F, self_dim, point_dim, obs_size, n_pad, lane);
    __syncwarp();
    if (X_out) {
        for (int i = lane; i < N * D; i += 32) {
            X_out[(int64_t)b * N * D + i] = X[i];
        }
    }
    int n_tiles = n_pad >> 4;
    for (int qt = 0; qt < n_tiles; qt++) {
        precision_t* Xq = X + qt * D * D;
        unsigned qA[4];
        ea_proj_tile(Kt, Xq, s_wq, lane);
        __syncwarp();
        ea_load_A(qA, Kt, D, lane);
        float o0[4] = {0, 0, 0, 0};
        float o1[4] = {0, 0, 0, 0};
        float m0 = -1.0e9f, m8 = -1.0e9f, l0 = 0.0f, l8 = 0.0f;
        for (int kt = 0; kt < n_tiles; kt++) {
            precision_t* Xk = X + kt * D * D;
            ea_proj_tile(Kt, Xk, s_wk, lane);
            ea_proj_tile(Vt, Xk, s_wv, lane);
            __syncwarp();
            float s0[4] = {0, 0, 0, 0};
            float s1[4] = {0, 0, 0, 0};
            ea_mma_ABt(s0, s1, qA, Kt, D, lane);
            int valid = N - kt * D;
            if (valid > D) valid = D;
            int col = (lane & 3) << 1;
            float scale = 0.25f;
            s0[0] *= scale; s0[1] *= scale; s0[2] *= scale; s0[3] *= scale;
            s1[0] *= scale; s1[1] *= scale; s1[2] *= scale; s1[3] *= scale;
            if (col >= valid) { s0[0] = -1.0e9f; s0[1] = -1.0e9f; s0[2] = -1.0e9f; s0[3] = -1.0e9f; }
            else if (col + 1 >= valid) { s0[1] = -1.0e9f; s0[3] = -1.0e9f; }
            if (col + 8 >= valid) { s1[0] = -1.0e9f; s1[1] = -1.0e9f; s1[2] = -1.0e9f; s1[3] = -1.0e9f; }
            else if (col + 9 >= valid) { s1[1] = -1.0e9f; s1[3] = -1.0e9f; }
            float rowmax0 = fmaxf(fmaxf(s0[0], s0[1]), fmaxf(s1[0], s1[1]));
            float rowmax8 = fmaxf(fmaxf(s0[2], s0[3]), fmaxf(s1[2], s1[3]));
            rowmax0 = fmaxf(rowmax0, __shfl_xor_sync(0xffffffff, rowmax0, 1));
            rowmax0 = fmaxf(rowmax0, __shfl_xor_sync(0xffffffff, rowmax0, 2));
            rowmax8 = fmaxf(rowmax8, __shfl_xor_sync(0xffffffff, rowmax8, 1));
            rowmax8 = fmaxf(rowmax8, __shfl_xor_sync(0xffffffff, rowmax8, 2));
            float m0n = fmaxf(m0, rowmax0);
            float m8n = fmaxf(m8, rowmax8);
            float a0 = expf(m0 - m0n);
            float a8 = expf(m8 - m8n);
            s0[0] = expf(s0[0] - m0n); s0[1] = expf(s0[1] - m0n);
            s1[0] = expf(s1[0] - m0n); s1[1] = expf(s1[1] - m0n);
            s0[2] = expf(s0[2] - m8n); s0[3] = expf(s0[3] - m8n);
            s1[2] = expf(s1[2] - m8n); s1[3] = expf(s1[3] - m8n);
            float t0 = s0[0] + s0[1] + s1[0] + s1[1];
            float t8 = s0[2] + s0[3] + s1[2] + s1[3];
            t0 += __shfl_xor_sync(0xffffffff, t0, 1);
            t0 += __shfl_xor_sync(0xffffffff, t0, 2);
            t8 += __shfl_xor_sync(0xffffffff, t8, 1);
            t8 += __shfl_xor_sync(0xffffffff, t8, 2);
            o0[0] *= a0; o0[1] *= a0; o1[0] *= a0; o1[1] *= a0;
            o0[2] *= a8; o0[3] *= a8; o1[2] *= a8; o1[3] *= a8;
            l0 = l0 * a0 + t0;
            l8 = l8 * a8 + t8;
            m0 = m0n; m8 = m8n;
            unsigned pA[4];
            ea_frags_to_A(pA, s0, s1);
            float p0[4] = {0, 0, 0, 0};
            float p1[4] = {0, 0, 0, 0};
            ea_mma_AB(p0, p1, pA, Vt, D, lane);
            o0[0] += p0[0]; o0[1] += p0[1]; o0[2] += p0[2]; o0[3] += p0[3];
            o1[0] += p1[0]; o1[1] += p1[1]; o1[2] += p1[2]; o1[3] += p1[3];
        }
        float inv0 = 1.0f / l0, inv8 = 1.0f / l8;
        o0[0] *= inv0; o0[1] *= inv0; o1[0] *= inv0; o1[1] *= inv0;
        o0[2] *= inv8; o0[3] *= inv8; o1[2] *= inv8; o1[3] *= inv8;
        int row = lane >> 2;
        int col = (lane & 3) << 1;
        auto yst = [&](int r, int c, float v) {
            Y[(qt * D + r) * D + c] = from_float(v + to_float(Xq[r * D + c]));
        };
        yst(row, col, o0[0]); yst(row, col + 1, o0[1]);
        yst(row + 8, col, o0[2]); yst(row + 8, col + 1, o0[3]);
        yst(row, col + 8, o1[0]); yst(row, col + 9, o1[1]);
        yst(row + 8, col + 8, o1[2]); yst(row + 8, col + 9, o1[3]);
        __syncwarp();
    }
    ea_mean_pool(pooled + (int64_t)b * D, Y, N, n_pad, lane);
}

// N=16 QKV backward. Writes dQ,dK,dV and relu-gated dX (residual included).
__global__ void __launch_bounds__(256, 4) ea_n16_bwd_kernel(
        precision_t* __restrict__ dq,
        precision_t* __restrict__ dk,
        precision_t* __restrict__ dv,
        precision_t* __restrict__ dx,
        const precision_t* __restrict__ Xg,
        const precision_t* __restrict__ d_pooled,
        const precision_t* __restrict__ wq,
        const precision_t* __restrict__ wk,
        const precision_t* __restrict__ wv,
        int B, int N) {
    extern __shared__ char ea_smem[];
    const int D = EA_D;
    precision_t* s_wq = (precision_t*)ea_smem;
    precision_t* s_wk = s_wq + D * D;
    precision_t* s_wv = s_wk + D * D;
    precision_t* s_tiles = s_wv + D * D;
    int tid = threadIdx.x;
    for (int i = tid; i < D * D; i += blockDim.x) {
        s_wq[i] = wq[i];
        s_wk[i] = wk[i];
        s_wv[i] = wv[i];
    }
    __syncthreads();
    int warp = tid >> 5, lane = tid & 31;
    int b = blockIdx.x * EA_WARPS + warp;
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
    for (int i = lane; i < D * D; i += 32)
        X[i] = (i < N * D) ? Xb[i] : from_float(0.0f);
    __syncwarp();
    ea_proj_tile(Q, X, s_wq, lane);
    ea_proj_tile(K, X, s_wk, lane);
    ea_proj_tile(V, X, s_wv, lane);
    __syncwarp();
    unsigned a[4];
    ea_load_A(a, Q, D, lane);
    float s0[4] = {0, 0, 0, 0}, s1[4] = {0, 0, 0, 0};
    ea_mma_ABt(s0, s1, a, K, D, lane);
    ea_softmax_16(s0, s1, lane, N);
    ea_store_C(P, D, s0, s1, lane);
    __syncwarp();
    float invN = 1.0f / (float)N;
    float dp = (lane < D) ? to_float(d_pooled[(int64_t)b * D + lane]) * invN : 0.0f;
    float dy[16];
    #pragma unroll
    for (int d = 0; d < D; d++) dy[d] = __shfl_sync(0xffffffff, dp, d);
    for (int i = lane; i < D * D; i += 32) {
        int n = i >> 4, d = i & 15;
        dY[i] = from_float((n < N) ? dy[d] : 0.0f);
        int r = n, c = d;
        Pt[c * D + r] = P[i];
    }
    __syncwarp();
    ea_load_A(a, Pt, D, lane);
    float c0[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0};
    ea_mma_AB(c0, c1, a, dY, D, lane);
    ea_store_C(tmp, D, c0, c1, lane);
    __syncwarp();
    for (int i = lane; i < N * D; i += 32) dv[(int64_t)b * N * D + i] = tmp[i];
    ea_load_A(a, dY, D, lane);
    c0[0] = c0[1] = c0[2] = c0[3] = 0;
    c1[0] = c1[1] = c1[2] = c1[3] = 0;
    ea_mma_ABt(c0, c1, a, V, D, lane);
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
    ea_store_C(P, D, c0, c1, lane);
    __syncwarp();
    ea_load_A(a, P, D, lane);
    c0[0] = c0[1] = c0[2] = c0[3] = 0;
    c1[0] = c1[1] = c1[2] = c1[3] = 0;
    ea_mma_AB(c0, c1, a, K, D, lane);
    ea_store_C(tmp, D, c0, c1, lane);
    __syncwarp();
    for (int i = lane; i < N * D; i += 32) dq[(int64_t)b * N * D + i] = tmp[i];
    for (int i = lane; i < D * D; i += 32) {
        int r = i >> 4, c = i & 15;
        Pt[c * D + r] = P[i];
    }
    __syncwarp();
    ea_load_A(a, Pt, D, lane);
    c0[0] = c0[1] = c0[2] = c0[3] = 0;
    c1[0] = c1[1] = c1[2] = c1[3] = 0;
    ea_mma_AB(c0, c1, a, Q, D, lane);
    ea_store_C(tmp, D, c0, c1, lane);
    __syncwarp();
    for (int i = lane; i < N * D; i += 32) dk[(int64_t)b * N * D + i] = tmp[i];
    for (int i = lane; i < D * D; i += 32) tmp[i] = dY[i];
    __syncwarp();
    ea_load_A(a, dq + (int64_t)b * N * D, D, lane);
    c0[0] = c0[1] = c0[2] = c0[3] = 0;
    c1[0] = c1[1] = c1[2] = c1[3] = 0;
    ea_mma_AB(c0, c1, a, s_wq, D, lane);
    auto accadd = [&](int rr, int cc, float v) {
        tmp[rr * D + cc] = from_float(to_float(tmp[rr * D + cc]) + v);
    };
    accadd(row, col, c0[0]); accadd(row, col + 1, c0[1]);
    accadd(row + 8, col, c0[2]); accadd(row + 8, col + 1, c0[3]);
    accadd(row, col + 8, c1[0]); accadd(row, col + 9, c1[1]);
    accadd(row + 8, col + 8, c1[2]); accadd(row + 8, col + 9, c1[3]);
    __syncwarp();
    ea_load_A(a, dk + (int64_t)b * N * D, D, lane);
    c0[0] = c0[1] = c0[2] = c0[3] = 0;
    c1[0] = c1[1] = c1[2] = c1[3] = 0;
    ea_mma_AB(c0, c1, a, s_wk, D, lane);
    accadd(row, col, c0[0]); accadd(row, col + 1, c0[1]);
    accadd(row + 8, col, c0[2]); accadd(row + 8, col + 1, c0[3]);
    accadd(row, col + 8, c1[0]); accadd(row, col + 9, c1[1]);
    accadd(row + 8, col + 8, c1[2]); accadd(row + 8, col + 9, c1[3]);
    __syncwarp();
    ea_load_A(a, dv + (int64_t)b * N * D, D, lane);
    c0[0] = c0[1] = c0[2] = c0[3] = 0;
    c1[0] = c1[1] = c1[2] = c1[3] = 0;
    ea_mma_AB(c0, c1, a, s_wv, D, lane);
    accadd(row, col, c0[0]); accadd(row, col + 1, c0[1]);
    accadd(row + 8, col, c0[2]); accadd(row + 8, col + 1, c0[3]);
    accadd(row, col + 8, c1[0]); accadd(row, col + 9, c1[1]);
    accadd(row + 8, col + 8, c1[2]); accadd(row + 8, col + 9, c1[3]);
    __syncwarp();
    for (int i = lane; i < N * D; i += 32) {
        float g = to_float(X[i]) > 0.0f ? to_float(tmp[i]) : 0.0f;
        dx[(int64_t)b * N * D + i] = from_float(g);
    }
}

__global__ void ea_relu_mask_kernel(precision_t* dx, const precision_t* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dx[i] = from_float(to_float(x[i]) > 0.0f ? to_float(dx[i]) : 0.0f);
    }
}

__global__ void ea_add_broadcast_rows(
        precision_t* dx, const precision_t* d_pooled, int B, int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tot = B * N * D;
    if (i >= tot) return;
    int d = i % D;
    int b = i / (N * D);
    float inv = 1.0f / (float)N;
    dx[i] = from_float(to_float(dx[i]) + to_float(d_pooled[b * D + d]) * inv);
}

static size_t ea_n16_smem(int F) {
    return (size_t)(EA_D * F + 3 * EA_D * EA_D + EA_WARPS * 4 * EA_D * EA_D) * sizeof(precision_t);
}

static size_t ea_flash_smem(int N, int F) {
    int n_pad = (N + 15) & ~15;
    return (size_t)(EA_D * F + 3 * EA_D * EA_D
        + EA_WARPS * n_pad * EA_D * 2
        + EA_WARPS * 2 * EA_D * EA_D) * sizeof(precision_t);
}

static size_t ea_bwd_smem() {
    return (size_t)(3 * EA_D * EA_D + EA_WARPS * 8 * EA_D * EA_D) * sizeof(precision_t);
}

static EntityAttnWeights* ea_create(int obs_size, int hidden) {
    EntityAttnWeights* w = (EntityAttnWeights*)calloc(1, sizeof(EntityAttnWeights));
    w->obs_size = obs_size;
    w->hidden = hidden;
    w->n_entities = ME_NUM_POINTS;
    w->self_dim = ME_SELF_DIM;
    w->point_dim = ME_POINT_DIM;
    w->f_in = ME_ENTITY_IN;
    return w;
}

static Prec ea_forward(void* vw, void* vact, Prec input, cudaStream_t stream) {
    EntityAttnWeights* w = (EntityAttnWeights*)vw;
    EntityAttnActivations* a = (EntityAttnActivations*)vact;
    int B = input.shape[0];
    int N = w->n_entities, F = w->f_in;
    if (a->ents.data) {
        me_materialize_points_kernel<<<grid_size(B * N * F), BLOCK_SIZE, 0, stream>>>(
            a->ents.data, input.data, B, w->obs_size);
    }
    precision_t* Xsave = a->x.data;
    int grid = (B + EA_WARPS - 1) / EA_WARPS;
    if (N <= 16) {
        size_t smem = ea_n16_smem(F);
        ea_n16_fwd_kernel<<<grid, EA_WARPS * 32, smem, stream>>>(
            a->pooled.data, Xsave, input.data, w->win.data, w->wq.data, w->wk.data, w->wv.data,
            B, N, F, w->self_dim, w->point_dim, w->obs_size);
    } else {
        size_t smem = ea_flash_smem(N, F);
        if (smem > 48 * 1024) {
            cudaFuncSetAttribute((const void*)ea_flash_fwd_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        }
        ea_flash_fwd_kernel<<<grid, EA_WARPS * 32, smem, stream>>>(
            a->pooled.data, Xsave, input.data, w->win.data, w->wq.data, w->wk.data, w->wv.data,
            B, N, F, w->self_dim, w->point_dim, w->obs_size);
    }
    Prec pooled = {.data = a->pooled.data, .shape = {B, EA_D}};
    Prec out = {.data = a->out.data, .shape = {B, w->hidden}};
    puf_mm(&pooled, &w->wout, &out, stream);
    return a->out;
}

static void ea_backward(void* vw, void* vact, Prec grad, cudaStream_t stream) {
    EntityAttnWeights* w = (EntityAttnWeights*)vw;
    EntityAttnActivations* a = (EntityAttnActivations*)vact;
    int B = grad.shape[0];
    int N = w->n_entities;
    puf_mm_tn(&grad, &a->pooled, &a->wout_g, stream);
    Prec d_pooled = {.data = a->pooled.data, .shape = {B, EA_D}};
    puf_mm_nn(&grad, &w->wout, &d_pooled, stream);

    if (N <= 16 && a->dq.data) {
        int grid = (B + EA_WARPS - 1) / EA_WARPS;
        ea_n16_bwd_kernel<<<grid, EA_WARPS * 32, ea_bwd_smem(), stream>>>(
            a->dq.data, a->dk.data, a->dv.data, a->dx.data,
            a->x.data, a->pooled.data, w->wq.data, w->wk.data, w->wv.data, B, N);
        Prec dq = {.data = a->dq.data, .shape = {B, N, EA_D}};
        Prec dk = {.data = a->dk.data, .shape = {B, N, EA_D}};
        Prec dv = {.data = a->dv.data, .shape = {B, N, EA_D}};
        Prec dx = {.data = a->dx.data, .shape = {B, N, EA_D}};
        Prec x = {.data = a->x.data, .shape = {B, N, EA_D}};
        puf_mm_tn(&dq, &x, &a->wq_g, stream);
        puf_mm_tn(&dk, &x, &a->wk_g, stream);
        puf_mm_tn(&dv, &x, &a->wv_g, stream);
        puf_mm_tn(&dx, &a->ents, &a->win_g, stream);
    }
}

static void ea_init_weights(void* vw, uint64_t* seed, cudaStream_t stream) {
    EntityAttnWeights* w = (EntityAttnWeights*)vw;
    Prec win = {.data = w->win.data, .shape = {EA_D, w->f_in}};
    Prec wq = {.data = w->wq.data, .shape = {EA_D, EA_D}};
    Prec wk = {.data = w->wk.data, .shape = {EA_D, EA_D}};
    Prec wv = {.data = w->wv.data, .shape = {EA_D, EA_D}};
    Prec wout = {.data = w->wout.data, .shape = {w->hidden, EA_D}};
    puf_kaiming_init(&win, sqrtf(2.0f), (*seed)++, stream);
    puf_kaiming_init(&wq, 1.0f, (*seed)++, stream);
    puf_kaiming_init(&wk, 1.0f, (*seed)++, stream);
    puf_kaiming_init(&wv, 1.0f, (*seed)++, stream);
    puf_kaiming_init(&wout, 1.0f, (*seed)++, stream);
}

static void ea_reg_params(void* vw, Allocator* alloc) {
    EntityAttnWeights* w = (EntityAttnWeights*)vw;
    w->win = {.shape = {EA_D, w->f_in}};
    w->wq = {.shape = {EA_D, EA_D}};
    w->wk = {.shape = {EA_D, EA_D}};
    w->wv = {.shape = {EA_D, EA_D}};
    w->wout = {.shape = {w->hidden, EA_D}};
    alloc_register(alloc, &w->win);
    alloc_register(alloc, &w->wq);
    alloc_register(alloc, &w->wk);
    alloc_register(alloc, &w->wv);
    alloc_register(alloc, &w->wout);
}

static void ea_reg_train(void* vw, void* vact, Allocator* acts, Allocator* grads, int B_TT) {
    EntityAttnWeights* w = (EntityAttnWeights*)vw;
    EntityAttnActivations* a = (EntityAttnActivations*)vact;
    *a = {};
    int N = w->n_entities, F = w->f_in;
    a->ents = {.shape = {B_TT, N, F}};
    a->x = {.shape = {B_TT, N, EA_D}};
    a->pooled = {.shape = {B_TT, EA_D}};
    a->out = {.shape = {B_TT, w->hidden}};
    a->dq = {.shape = {B_TT, N, EA_D}};
    a->dk = {.shape = {B_TT, N, EA_D}};
    a->dv = {.shape = {B_TT, N, EA_D}};
    a->dx = {.shape = {B_TT, N, EA_D}};
    alloc_register(acts, &a->ents);
    alloc_register(acts, &a->x);
    alloc_register(acts, &a->pooled);
    alloc_register(acts, &a->out);
    alloc_register(acts, &a->dq);
    alloc_register(acts, &a->dk);
    alloc_register(acts, &a->dv);
    alloc_register(acts, &a->dx);
    a->win_g = {.shape = {EA_D, F}};
    a->wq_g = {.shape = {EA_D, EA_D}};
    a->wk_g = {.shape = {EA_D, EA_D}};
    a->wv_g = {.shape = {EA_D, EA_D}};
    a->wout_g = {.shape = {w->hidden, EA_D}};
    alloc_register(grads, &a->win_g);
    alloc_register(grads, &a->wq_g);
    alloc_register(grads, &a->wk_g);
    alloc_register(grads, &a->wv_g);
    alloc_register(grads, &a->wout_g);
}

static void ea_reg_rollout(void* vw, void* vact, Allocator* alloc, int B) {
    EntityAttnWeights* w = (EntityAttnWeights*)vw;
    EntityAttnActivations* a = (EntityAttnActivations*)vact;
    *a = {};
    a->pooled = {.shape = {B, EA_D}};
    a->out = {.shape = {B, w->hidden}};
    alloc_register(alloc, &a->pooled);
    alloc_register(alloc, &a->out);
}

static void* ea_create_weights(void* self) {
    Encoder* e = (Encoder*)self;
    return ea_create(e->in_dim, e->out_dim);
}

static void create_entity_attn_encoder(Encoder* enc) {
    *enc = Encoder{
        .forward = ea_forward,
        .backward = ea_backward,
        .init_weights = ea_init_weights,
        .reg_params = ea_reg_params,
        .reg_train = ea_reg_train,
        .reg_rollout = ea_reg_rollout,
        .create_weights = ea_create_weights,
        .in_dim = enc->in_dim, .out_dim = enc->out_dim,
        .activation_size = sizeof(EntityAttnActivations),
    };
}

