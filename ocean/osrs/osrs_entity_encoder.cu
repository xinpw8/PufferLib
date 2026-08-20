static constexpr int OSRS_ENTITY_INV_START = 52;
static constexpr int OSRS_ENTITY_INV_NUM_RECORDS = 28;
static constexpr int OSRS_ENTITY_INV_OBS_FEATURES = 1;
static constexpr int OSRS_ENTITY_INV_FEATURES = OSRS_ITEM_OBS_TABLE_COLS;
static constexpr int OSRS_ENTITY_BOTTLENECK = 16;
static constexpr int OSRS_ENTITY_BATCH_TILE = 8;
static constexpr int OSRS_ENTITY_HIDDEN_TILE = 32;
static constexpr int OSRS_ENTITY_FC_THREADS =
    OSRS_ENTITY_BATCH_TILE * OSRS_ENTITY_HIDDEN_TILE;

static_assert(OSRS_ENTITY_INV_START == 52);
static_assert(OSRS_ENTITY_INV_NUM_RECORDS == 28);
static_assert(OSRS_ENTITY_INV_OBS_FEATURES == 1);
static_assert(OSRS_ENTITY_INV_FEATURES == 14);

enum OsrsEntityBranchExpansion {
    OSRS_ENTITY_BRANCH_TYPE_ONEHOT = 0,
    OSRS_ENTITY_BRANCH_ITEM_TABLE,
};

struct OsrsEntityBranchDescriptor {
    int obs_start;
    int num_records;
    int obs_features;
    int type_onehot;
    int type_code_scale;
    OsrsEntityBranchExpansion expansion;
};

struct OsrsEntityEncoderDescriptor {
    const OsrsEntityBranchDescriptor* branches;
    int num_branches;
};

struct OsrsEntityBranchWeights {
    Prec l1_w;
    Prec l2_w;
};

struct OsrsEntityEncoderWeights {
    Prec global_w;
    Prec inv_l1_w;
    Prec inv_l2_w;
    const OsrsEntityEncoderDescriptor* descriptor;
    int obs_size;
    int hidden;
};

struct OsrsEntityBranchActivations {
    Prec flat;
    Prec z1;
    Prec h1;
    Prec grad_z1;
    Int pool_argmax;
    Prec l1_wgrad;
    Prec l2_wgrad;
};

struct OsrsEntityEncoderActivations {
    Prec out;
    Prec saved_obs;
    Prec inv_flat;
    Prec inv_z1;
    Prec inv_h1;
    Prec inv_grad_z1;
    Int inv_pool_argmax;
    Prec global_wgrad;
    Prec inv_l1_wgrad;
    Prec inv_l2_wgrad;
};

static OsrsEntityBranchWeights* osrs_entity_branch_weights(
    OsrsEntityEncoderWeights* weights
) {
    return (OsrsEntityBranchWeights*)(weights + 1);
}

static OsrsEntityBranchActivations* osrs_entity_branch_activations(
    OsrsEntityEncoderActivations* activations
) {
    return (OsrsEntityBranchActivations*)(activations + 1);
}

__device__ __forceinline__ float osrs_entity_gelu_fwd(float x) {
    float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ float osrs_entity_gelu_grad(float x) {
    float x3 = x * x * x;
    float inner = 0.7978845608028654f * (x + 0.044715f * x3);
    float t = tanhf(inner);
    float dinner = 0.7978845608028654f * (1.0f + 3.0f * 0.044715f * x * x);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * dinner;
}

__global__ void osrs_entity_gather_branch(
    precision_t* __restrict__ flat,
    const precision_t* __restrict__ obs,
    int B,
    int obs_size,
    int obs_start,
    int num_records,
    int obs_features,
    int type_onehot,
    int type_code_scale,
    OsrsEntityBranchExpansion expansion
) {
    int features = expansion == OSRS_ENTITY_BRANCH_ITEM_TABLE
        ? OSRS_ITEM_OBS_TABLE_COLS
        : type_onehot + obs_features - 1;
    int record_block = num_records * features;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * record_block;
    if (idx >= total) return;
    int b = idx / record_block;
    int off = idx - b * record_block;
    int record = off / features;
    int feature = off - record * features;
    const precision_t* src = obs + (int64_t)b * obs_size + obs_start
        + record * obs_features;
    int code = (int)lrintf(to_float(src[0]) * (float)type_code_scale);
    if (expansion == OSRS_ENTITY_BRANCH_ITEM_TABLE) {
        assert(code >= 0 && code < OSRS_ITEM_OBS_TABLE_ROWS);
        flat[idx] = from_float(OSRS_ITEM_OBS_TABLE_DEV[code][feature]);
    } else if (feature < type_onehot) {
        assert(code >= 0 && code <= type_onehot);
        flat[idx] = from_float(code == feature + 1 ? 1.0f : 0.0f);
    } else {
        flat[idx] = src[1 + feature - type_onehot];
    }
}

__global__ void osrs_entity_fused_pool_fwd(
    precision_t* __restrict__ out,
    int* __restrict__ argmax,
    precision_t* __restrict__ h1,
    const precision_t* __restrict__ z1,
    const precision_t* __restrict__ record_flat,
    const precision_t* __restrict__ l2_w,
    int B,
    int H,
    int num_records,
    int record_features,
    int active_width
) {
    extern __shared__ float osrs_entity_pool_shared[];
    float* h1_tile = osrs_entity_pool_shared;
    float* mask_tile = h1_tile +
        OSRS_ENTITY_BATCH_TILE * num_records * OSRS_ENTITY_BOTTLENECK;
    float* w_tile = mask_tile + OSRS_ENTITY_BATCH_TILE * num_records;
    int batch_base = blockIdx.x * OSRS_ENTITY_BATCH_TILE;
    int hidden_base = blockIdx.y * OSRS_ENTITY_HIDDEN_TILE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * OSRS_ENTITY_HIDDEN_TILE + tx;
    int h1_values = OSRS_ENTITY_BATCH_TILE * num_records * OSRS_ENTITY_BOTTLENECK;
    for (int idx = tid; idx < h1_values; idx += OSRS_ENTITY_FC_THREADS) {
        int bt = idx / (num_records * OSRS_ENTITY_BOTTLENECK);
        int rem = idx - bt * num_records * OSRS_ENTITY_BOTTLENECK;
        int gb = batch_base + bt;
        float value = gb < B ? osrs_entity_gelu_fwd(to_float(
            z1[(int64_t)gb * num_records * OSRS_ENTITY_BOTTLENECK + rem])) : 0.0f;
        h1_tile[idx] = value;
        if (h1 && blockIdx.y == 0 && gb < B)
            h1[(int64_t)gb * num_records * OSRS_ENTITY_BOTTLENECK + rem] = from_float(value);
    }
    int mask_values = OSRS_ENTITY_BATCH_TILE * num_records;
    for (int idx = tid; idx < mask_values; idx += OSRS_ENTITY_FC_THREADS) {
        int bt = idx / num_records;
        int record = idx - bt * num_records;
        int gb = batch_base + bt;
        float active = 0.0f;
        if (gb < B) {
            const precision_t* rec = record_flat +
                ((int64_t)gb * num_records + record) * record_features;
            for (int feature = 0; feature < active_width; feature++)
                active += to_float(rec[feature]);
        }
        mask_tile[idx] = active;
    }
    int w_values = OSRS_ENTITY_HIDDEN_TILE * OSRS_ENTITY_BOTTLENECK;
    for (int idx = tid; idx < w_values; idx += OSRS_ENTITY_FC_THREADS) {
        int d = idx / OSRS_ENTITY_HIDDEN_TILE;
        int th = idx - d * OSRS_ENTITY_HIDDEN_TILE;
        int gh = hidden_base + th;
        w_tile[idx] = gh < H
            ? to_float(l2_w[(int64_t)gh * OSRS_ENTITY_BOTTLENECK + d]) : 0.0f;
    }
    __syncthreads();
    int b = batch_base + ty;
    int h = hidden_base + tx;
    if (b >= B || h >= H) return;
    float best = -3.4028234663852886e38f;
    int best_record = -1;
    for (int record = 0; record < num_records; record++) {
        if (mask_tile[ty * num_records + record] <= 0.0f) continue;
        const float* hp = h1_tile +
            (ty * num_records + record) * OSRS_ENTITY_BOTTLENECK;
        float sum = 0.0f;
#pragma unroll
        for (int d = 0; d < OSRS_ENTITY_BOTTLENECK; d++)
            sum += w_tile[d * OSRS_ENTITY_HIDDEN_TILE + tx] * hp[d];
        if (sum > best) { best = sum; best_record = record; }
    }
    int64_t out_idx = (int64_t)b * H + h;
    out[out_idx] = from_float(
        to_float(out[out_idx]) + (best_record < 0 ? 0.0f : best));
    if (argmax) argmax[out_idx] = best_record;
}

__global__ void osrs_entity_fused_l2_wgrad(
    precision_t* __restrict__ wgrad,
    const precision_t* __restrict__ grad,
    const precision_t* __restrict__ h1,
    const int* __restrict__ argmax,
    int B,
    int H,
    int num_records
) {
    int h = blockIdx.x;
    if (h >= H) return;
    float sum[OSRS_ENTITY_BOTTLENECK];
#pragma unroll
    for (int k = 0; k < OSRS_ENTITY_BOTTLENECK; k++) sum[k] = 0.0f;
    for (int b = threadIdx.x; b < B; b += blockDim.x) {
        int record = argmax[(int64_t)b * H + h];
        if (record < 0) continue;
        float g = to_float(grad[(int64_t)b * H + h]);
        const precision_t* hp = h1 +
            ((int64_t)b * num_records + record) * OSRS_ENTITY_BOTTLENECK;
#pragma unroll
        for (int k = 0; k < OSRS_ENTITY_BOTTLENECK; k++)
            sum[k] += g * to_float(hp[k]);
    }
    __shared__ float warp_sums[OSRS_ENTITY_BOTTLENECK * 32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int num_warps = (blockDim.x + 31) >> 5;
#pragma unroll
    for (int k = 0; k < OSRS_ENTITY_BOTTLENECK; k++) {
        float value = sum[k];
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffff, value, offset);
        if (lane == 0) warp_sums[k * 32 + warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int k = 0; k < OSRS_ENTITY_BOTTLENECK; k++) {
            float value = lane < num_warps ? warp_sums[k * 32 + lane] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1)
                value += __shfl_down_sync(0xffffffff, value, offset);
            if (lane == 0)
                wgrad[(int64_t)h * OSRS_ENTITY_BOTTLENECK + k] = from_float(value);
        }
    }
}

__global__ void osrs_entity_fused_grad_z1(
    precision_t* __restrict__ grad_z1,
    const precision_t* __restrict__ grad,
    const precision_t* __restrict__ l2_w,
    const precision_t* __restrict__ z1,
    const int* __restrict__ argmax,
    int B,
    int H,
    int num_records
) {
    int b = blockIdx.x;
    if (b >= B) return;
    extern __shared__ float osrs_entity_grad_shared[];
    float* accum = osrs_entity_grad_shared;
    int* arg_s = (int*)(accum + num_records * OSRS_ENTITY_BOTTLENECK);
    float* grad_s = (float*)(arg_s + blockDim.x);
    for (int idx = threadIdx.x;
            idx < num_records * OSRS_ENTITY_BOTTLENECK;
            idx += blockDim.x)
        accum[idx] = 0.0f;
    __syncthreads();
    for (int base = 0; base < H; base += blockDim.x) {
        int h = base + threadIdx.x;
        if (h < H) {
            arg_s[threadIdx.x] = argmax[(int64_t)b * H + h];
            grad_s[threadIdx.x] = to_float(grad[(int64_t)b * H + h]);
        }
        __syncthreads();
        int tile = H - base;
        if (tile > (int)blockDim.x) tile = blockDim.x;
        for (int idx = threadIdx.x;
                idx < num_records * OSRS_ENTITY_BOTTLENECK;
                idx += blockDim.x) {
            int record = idx / OSRS_ENTITY_BOTTLENECK;
            int k = idx - record * OSRS_ENTITY_BOTTLENECK;
            float sum = 0.0f;
            for (int j = 0; j < tile; j++) {
                if (arg_s[j] != record) continue;
                sum += grad_s[j] * to_float(
                    l2_w[(int64_t)(base + j) * OSRS_ENTITY_BOTTLENECK + k]);
            }
            accum[idx] += sum;
        }
        __syncthreads();
    }
    for (int idx = threadIdx.x;
            idx < num_records * OSRS_ENTITY_BOTTLENECK;
            idx += blockDim.x) {
        int64_t out_idx =
            (int64_t)b * num_records * OSRS_ENTITY_BOTTLENECK + idx;
        grad_z1[out_idx] = from_float(
            accum[idx] * osrs_entity_gelu_grad(to_float(z1[out_idx])));
    }
}

static void osrs_entity_launch_fused_fwd(
    precision_t* out,
    int* argmax,
    precision_t* h1,
    const precision_t* z1,
    const precision_t* record_flat,
    const precision_t* l2_w,
    int B,
    int H,
    int num_records,
    int record_features,
    int active_width,
    cudaStream_t stream
) {
    dim3 block(OSRS_ENTITY_HIDDEN_TILE, OSRS_ENTITY_BATCH_TILE);
    dim3 grid(
        (B + OSRS_ENTITY_BATCH_TILE - 1) / OSRS_ENTITY_BATCH_TILE,
        (H + OSRS_ENTITY_HIDDEN_TILE - 1) / OSRS_ENTITY_HIDDEN_TILE);
    size_t shared_bytes = (
        (size_t)OSRS_ENTITY_BATCH_TILE * num_records * OSRS_ENTITY_BOTTLENECK +
        (size_t)OSRS_ENTITY_BATCH_TILE * num_records +
        (size_t)OSRS_ENTITY_HIDDEN_TILE * OSRS_ENTITY_BOTTLENECK) * sizeof(float);
    osrs_entity_fused_pool_fwd<<<grid, block, shared_bytes, stream>>>(
        out, argmax, h1, z1, record_flat, l2_w,
        B, H, num_records, record_features, active_width);
}

static void osrs_entity_launch_fused_bwd(
    precision_t* l2_wgrad,
    precision_t* grad_z1,
    const precision_t* grad,
    const precision_t* l2_w,
    const precision_t* z1,
    const precision_t* h1,
    const int* argmax,
    int B,
    int H,
    int num_records,
    cudaStream_t stream
) {
    osrs_entity_fused_l2_wgrad<<<H, 256, 0, stream>>>(
        l2_wgrad, grad, h1, argmax, B, H, num_records);
    size_t shared_bytes =
        ((size_t)num_records * OSRS_ENTITY_BOTTLENECK + 2 * BLOCK_SIZE) *
        sizeof(float);
    osrs_entity_fused_grad_z1<<<B, BLOCK_SIZE, shared_bytes, stream>>>(
        grad_z1, grad, l2_w, z1, argmax, B, H, num_records);
}

static int osrs_entity_branch_features(const OsrsEntityBranchDescriptor* branch) {
    return branch->expansion == OSRS_ENTITY_BRANCH_ITEM_TABLE
        ? OSRS_ITEM_OBS_TABLE_COLS
        : branch->type_onehot + branch->obs_features - 1;
}

static Prec osrs_entity_encoder_forward(
    void* weights,
    void* activations,
    Prec input,
    cudaStream_t stream
) {
    OsrsEntityEncoderWeights* ew = (OsrsEntityEncoderWeights*)weights;
    OsrsEntityEncoderActivations* a = (OsrsEntityEncoderActivations*)activations;
    int B = input.shape[0];
    int H = ew->hidden;
    if (a->saved_obs.data) puf_copy(&a->saved_obs, &input, stream);
    puf_mm(&input, &ew->global_w, &a->out, stream);

    int inventory_batch = B * OSRS_ENTITY_INV_NUM_RECORDS;
    osrs_entity_gather_branch<<<
        grid_size(inventory_batch * OSRS_ENTITY_INV_FEATURES),
        BLOCK_SIZE, 0, stream>>>(
        a->inv_flat.data, input.data, B, ew->obs_size,
        OSRS_ENTITY_INV_START, OSRS_ENTITY_INV_NUM_RECORDS,
        OSRS_ENTITY_INV_OBS_FEATURES, 0, OSRS_ITEM_OBS_CODE_SCALE,
        OSRS_ENTITY_BRANCH_ITEM_TABLE);
    Prec inventory_2d = {
        .data = a->inv_flat.data,
        .shape = {inventory_batch, OSRS_ENTITY_INV_FEATURES},
    };
    puf_mm(&inventory_2d, &ew->inv_l1_w, &a->inv_z1, stream);
    osrs_entity_launch_fused_fwd(
        a->out.data, a->inv_pool_argmax.data, a->inv_h1.data,
        a->inv_z1.data, a->inv_flat.data, ew->inv_l2_w.data,
        B, H, OSRS_ENTITY_INV_NUM_RECORDS, OSRS_ENTITY_INV_FEATURES, 1, stream);

    for (int branch_idx = 0;
            branch_idx < ew->descriptor->num_branches;
            branch_idx++) {
        const OsrsEntityBranchDescriptor* descriptor =
            &ew->descriptor->branches[branch_idx];
        OsrsEntityBranchWeights* bw =
            &osrs_entity_branch_weights(ew)[branch_idx];
        OsrsEntityBranchActivations* ba =
            &osrs_entity_branch_activations(a)[branch_idx];
        int features = osrs_entity_branch_features(descriptor);
        int branch_batch = B * descriptor->num_records;
        osrs_entity_gather_branch<<<
            grid_size(branch_batch * features), BLOCK_SIZE, 0, stream>>>(
            ba->flat.data, input.data, B, ew->obs_size,
            descriptor->obs_start, descriptor->num_records,
            descriptor->obs_features, descriptor->type_onehot,
            descriptor->type_code_scale, descriptor->expansion);
        Prec branch_2d = {
            .data = ba->flat.data,
            .shape = {branch_batch, features},
        };
        puf_mm(&branch_2d, &bw->l1_w, &ba->z1, stream);
        osrs_entity_launch_fused_fwd(
            a->out.data, ba->pool_argmax.data, ba->h1.data,
            ba->z1.data, ba->flat.data, bw->l2_w.data,
            B, H, descriptor->num_records, features,
            descriptor->expansion == OSRS_ENTITY_BRANCH_ITEM_TABLE
                ? 1 : descriptor->type_onehot,
            stream);
    }
    return a->out;
}

static void osrs_entity_encoder_backward(
    void* weights,
    void* activations,
    Prec grad,
    cudaStream_t stream
) {
    OsrsEntityEncoderWeights* ew = (OsrsEntityEncoderWeights*)weights;
    OsrsEntityEncoderActivations* a = (OsrsEntityEncoderActivations*)activations;
    int B = grad.shape[0];
    int H = ew->hidden;
    puf_mm_tn(&grad, &a->saved_obs, &a->global_wgrad, stream);

    int inventory_batch = B * OSRS_ENTITY_INV_NUM_RECORDS;
    osrs_entity_launch_fused_bwd(
        a->inv_l2_wgrad.data, a->inv_grad_z1.data, grad.data,
        ew->inv_l2_w.data, a->inv_z1.data, a->inv_h1.data,
        a->inv_pool_argmax.data, B, H, OSRS_ENTITY_INV_NUM_RECORDS, stream);
    Prec inventory_2d = {
        .data = a->inv_flat.data,
        .shape = {inventory_batch, OSRS_ENTITY_INV_FEATURES},
    };
    puf_mm_tn(&a->inv_grad_z1, &inventory_2d, &a->inv_l1_wgrad, stream);

    for (int branch_idx = 0;
            branch_idx < ew->descriptor->num_branches;
            branch_idx++) {
        const OsrsEntityBranchDescriptor* descriptor =
            &ew->descriptor->branches[branch_idx];
        OsrsEntityBranchWeights* bw =
            &osrs_entity_branch_weights(ew)[branch_idx];
        OsrsEntityBranchActivations* ba =
            &osrs_entity_branch_activations(a)[branch_idx];
        int features = osrs_entity_branch_features(descriptor);
        int branch_batch = B * descriptor->num_records;
        osrs_entity_launch_fused_bwd(
            ba->l2_wgrad.data, ba->grad_z1.data, grad.data,
            bw->l2_w.data, ba->z1.data, ba->h1.data, ba->pool_argmax.data,
            B, H, descriptor->num_records, stream);
        Prec branch_2d = {
            .data = ba->flat.data,
            .shape = {branch_batch, features},
        };
        puf_mm_tn(&ba->grad_z1, &branch_2d, &ba->l1_wgrad, stream);
    }
}

static void osrs_entity_encoder_init_weights(
    void* weights,
    uint64_t* seed,
    cudaStream_t stream
) {
    OsrsEntityEncoderWeights* ew = (OsrsEntityEncoderWeights*)weights;
    auto init2d = [&](Prec& tensor, int rows, int cols) {
        Prec shaped = {.data = tensor.data, .shape = {rows, cols}};
        puf_kaiming_init(&shaped, sqrtf(2.0f), (*seed)++, stream);
    };
    init2d(ew->global_w, ew->hidden, ew->obs_size);
    init2d(ew->inv_l1_w, OSRS_ENTITY_BOTTLENECK, OSRS_ENTITY_INV_FEATURES);
    init2d(ew->inv_l2_w, ew->hidden, OSRS_ENTITY_BOTTLENECK);
    for (int branch_idx = 0;
            branch_idx < ew->descriptor->num_branches;
            branch_idx++) {
        int features = osrs_entity_branch_features(
            &ew->descriptor->branches[branch_idx]);
        OsrsEntityBranchWeights* bw =
            &osrs_entity_branch_weights(ew)[branch_idx];
        init2d(bw->l1_w, OSRS_ENTITY_BOTTLENECK, features);
        init2d(bw->l2_w, ew->hidden, OSRS_ENTITY_BOTTLENECK);
    }
}

static void osrs_entity_register_param(Allocator* allocator, Prec* tensor) {
    assert(numel(tensor->shape) % 8 == 0);
    alloc_register(allocator, tensor);
}

static void osrs_entity_encoder_reg_params(void* weights, Allocator* allocator) {
    OsrsEntityEncoderWeights* ew = (OsrsEntityEncoderWeights*)weights;
    ew->global_w = {.shape = {ew->hidden, ew->obs_size}};
    osrs_entity_register_param(allocator, &ew->global_w);
    ew->inv_l1_w = {.shape = {OSRS_ENTITY_BOTTLENECK, OSRS_ENTITY_INV_FEATURES}};
    ew->inv_l2_w = {.shape = {ew->hidden, OSRS_ENTITY_BOTTLENECK}};
    osrs_entity_register_param(allocator, &ew->inv_l1_w);
    osrs_entity_register_param(allocator, &ew->inv_l2_w);
    for (int branch_idx = 0;
            branch_idx < ew->descriptor->num_branches;
            branch_idx++) {
        int features = osrs_entity_branch_features(
            &ew->descriptor->branches[branch_idx]);
        OsrsEntityBranchWeights* bw =
            &osrs_entity_branch_weights(ew)[branch_idx];
        bw->l1_w = {.shape = {OSRS_ENTITY_BOTTLENECK, features}};
        bw->l2_w = {.shape = {ew->hidden, OSRS_ENTITY_BOTTLENECK}};
        osrs_entity_register_param(allocator, &bw->l1_w);
        osrs_entity_register_param(allocator, &bw->l2_w);
    }
}

static void osrs_entity_register_branch_train(
    const OsrsEntityBranchDescriptor* descriptor,
    OsrsEntityBranchActivations* branch,
    Allocator* acts,
    Allocator* grads,
    int B,
    int H
) {
    int features = osrs_entity_branch_features(descriptor);
    int record_batch = B * descriptor->num_records;
    branch->flat = {.shape = {record_batch, features}};
    branch->z1 = {.shape = {record_batch, OSRS_ENTITY_BOTTLENECK}};
    branch->h1 = {.shape = {record_batch, OSRS_ENTITY_BOTTLENECK}};
    branch->grad_z1 = {.shape = {record_batch, OSRS_ENTITY_BOTTLENECK}};
    branch->pool_argmax = {.shape = {B, H}};
    alloc_register(acts, &branch->flat);
    alloc_register(acts, &branch->z1);
    alloc_register(acts, &branch->h1);
    alloc_register(acts, &branch->grad_z1);
    alloc_register(acts, &branch->pool_argmax);
    branch->l1_wgrad = {.shape = {OSRS_ENTITY_BOTTLENECK, features}};
    branch->l2_wgrad = {.shape = {H, OSRS_ENTITY_BOTTLENECK}};
    alloc_register(grads, &branch->l1_wgrad);
    alloc_register(grads, &branch->l2_wgrad);
}

static void osrs_entity_encoder_reg_train(
    void* weights,
    void* activations,
    Allocator* acts,
    Allocator* grads,
    int B
) {
    OsrsEntityEncoderWeights* ew = (OsrsEntityEncoderWeights*)weights;
    OsrsEntityEncoderActivations* a = (OsrsEntityEncoderActivations*)activations;
    int H = ew->hidden;
    *a = {};
    a->out = {.shape = {B, H}};
    a->saved_obs = {.shape = {B, ew->obs_size}};
    a->inv_flat = {
        .shape = {B * OSRS_ENTITY_INV_NUM_RECORDS, OSRS_ENTITY_INV_FEATURES},
    };
    a->inv_z1 = {
        .shape = {B * OSRS_ENTITY_INV_NUM_RECORDS, OSRS_ENTITY_BOTTLENECK},
    };
    a->inv_h1 = a->inv_z1;
    a->inv_grad_z1 = a->inv_z1;
    a->inv_pool_argmax = {.shape = {B, H}};
    alloc_register(acts, &a->out);
    alloc_register(acts, &a->saved_obs);
    alloc_register(acts, &a->inv_flat);
    alloc_register(acts, &a->inv_z1);
    alloc_register(acts, &a->inv_h1);
    alloc_register(acts, &a->inv_grad_z1);
    alloc_register(acts, &a->inv_pool_argmax);
    a->global_wgrad = {.shape = {H, ew->obs_size}};
    a->inv_l1_wgrad = {
        .shape = {OSRS_ENTITY_BOTTLENECK, OSRS_ENTITY_INV_FEATURES},
    };
    a->inv_l2_wgrad = {.shape = {H, OSRS_ENTITY_BOTTLENECK}};
    alloc_register(grads, &a->global_wgrad);
    alloc_register(grads, &a->inv_l1_wgrad);
    alloc_register(grads, &a->inv_l2_wgrad);
    for (int branch_idx = 0;
            branch_idx < ew->descriptor->num_branches;
            branch_idx++) {
        osrs_entity_register_branch_train(
            &ew->descriptor->branches[branch_idx],
            &osrs_entity_branch_activations(a)[branch_idx],
            acts, grads, B, H);
    }
}

static void osrs_entity_register_branch_rollout(
    const OsrsEntityBranchDescriptor* descriptor,
    OsrsEntityBranchActivations* branch,
    Allocator* allocator,
    int B
) {
    *branch = {};
    int features = osrs_entity_branch_features(descriptor);
    int record_batch = B * descriptor->num_records;
    branch->flat = {.shape = {record_batch, features}};
    branch->z1 = {.shape = {record_batch, OSRS_ENTITY_BOTTLENECK}};
    alloc_register(allocator, &branch->flat);
    alloc_register(allocator, &branch->z1);
}

static void osrs_entity_encoder_reg_rollout(
    void* weights,
    void* activations,
    Allocator* allocator,
    int B
) {
    OsrsEntityEncoderWeights* ew = (OsrsEntityEncoderWeights*)weights;
    OsrsEntityEncoderActivations* a = (OsrsEntityEncoderActivations*)activations;
    int H = ew->hidden;
    *a = {};
    a->out = {.shape = {B, H}};
    a->inv_flat = {
        .shape = {B * OSRS_ENTITY_INV_NUM_RECORDS, OSRS_ENTITY_INV_FEATURES},
    };
    a->inv_z1 = {
        .shape = {B * OSRS_ENTITY_INV_NUM_RECORDS, OSRS_ENTITY_BOTTLENECK},
    };
    alloc_register(allocator, &a->out);
    alloc_register(allocator, &a->inv_flat);
    alloc_register(allocator, &a->inv_z1);
    for (int branch_idx = 0;
            branch_idx < ew->descriptor->num_branches;
            branch_idx++) {
        osrs_entity_register_branch_rollout(
            &ew->descriptor->branches[branch_idx],
            &osrs_entity_branch_activations(a)[branch_idx],
            allocator, B);
    }
}

template <const OsrsEntityEncoderDescriptor* descriptor>
static void* osrs_entity_encoder_create_weights(void* self) {
    Encoder* encoder = (Encoder*)self;
    size_t weights_size = sizeof(OsrsEntityEncoderWeights) +
        (size_t)descriptor->num_branches * sizeof(OsrsEntityBranchWeights);
    OsrsEntityEncoderWeights* ew =
        (OsrsEntityEncoderWeights*)calloc(1, weights_size);
    ew->descriptor = descriptor;
    ew->obs_size = encoder->in_dim;
    ew->hidden = encoder->out_dim;
    return ew;
}

template <const OsrsEntityEncoderDescriptor* descriptor>
static void create_osrs_entity_encoder(Encoder* encoder) {
    *encoder = Encoder{
        .forward = osrs_entity_encoder_forward,
        .backward = osrs_entity_encoder_backward,
        .init_weights = osrs_entity_encoder_init_weights,
        .reg_params = osrs_entity_encoder_reg_params,
        .reg_train = osrs_entity_encoder_reg_train,
        .reg_rollout = osrs_entity_encoder_reg_rollout,
        .create_weights = osrs_entity_encoder_create_weights<descriptor>,
        .in_dim = encoder->in_dim,
        .out_dim = encoder->out_dim,
        .activation_size = sizeof(OsrsEntityEncoderActivations) +
            (size_t)descriptor->num_branches *
                sizeof(OsrsEntityBranchActivations),
    };
}

static constexpr OsrsEntityBranchDescriptor OSRS_EQUIPMENT_ENTITY_BRANCH[] = {
    {
        .obs_start = 80,
        .num_records = NUM_GEAR_SLOTS,
        .obs_features = 1,
        .type_onehot = 0,
        .type_code_scale = OSRS_ITEM_OBS_CODE_SCALE,
        .expansion = OSRS_ENTITY_BRANCH_ITEM_TABLE,
    },
};

static constexpr OsrsEntityEncoderDescriptor OSRS_EQUIPMENT_ENTITY_DESCRIPTOR = {
    .branches = OSRS_EQUIPMENT_ENTITY_BRANCH,
    .num_branches = 1,
};
#ifdef ZUL_NUM_OBS
static_assert(ZUL_NUM_OBS == 205);
#endif
