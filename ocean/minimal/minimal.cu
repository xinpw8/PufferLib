// Minimal entity encoder (points + max pool).
// Included by src/ocean.cu — requires precision_t, PrecisionTensor, Allocator, puf_mm, etc.

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
    PrecisionTensor input_w, output_w;
    int obs_size, hidden;
};

struct MinimalEntityEncoderActivations {
    PrecisionTensor point_input, entity_hidden, out, grad_entity;
    PrecisionTensor input_wgrad, output_wgrad;
    IntTensor argmax;
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

static PrecisionTensor me_encoder_forward(void* w, void* activations, PrecisionTensor input, cudaStream_t stream) {
    MinimalEntityEncoderWeights* ew = (MinimalEntityEncoderWeights*)w;
    MinimalEntityEncoderActivations* a = (MinimalEntityEncoderActivations*)activations;
    int B = input.shape[0];

    me_materialize_points_kernel<<<grid_size(B * ME_NUM_POINTS * ME_ENTITY_IN), BLOCK_SIZE, 0, stream>>>(
        a->point_input.data, input.data, B, ew->obs_size);
    PrecisionTensor point_input = {.data = a->point_input.data, .shape = {B, ME_NUM_POINTS, ME_ENTITY_IN}};
    PrecisionTensor entity_hidden = {.data = a->entity_hidden.data, .shape = {B, ME_NUM_POINTS, ME_ENTITY_HIDDEN}};
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

static void me_encoder_backward(void* w, void* activations, PrecisionTensor grad, cudaStream_t stream) {
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
    PrecisionTensor input_w = {.data = ew->input_w.data, .shape = {ME_ENTITY_HIDDEN, ME_ENTITY_IN}};
    PrecisionTensor output_w = {.data = ew->output_w.data, .shape = {ew->hidden, ME_ENTITY_HIDDEN}};
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
