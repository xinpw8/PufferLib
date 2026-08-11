// OSRS Colosseum CUDA entity encoder.
// Included by src/ocean.cu — requires precision_t, Prec, Allocator, puf_mm, etc.


// ---- Colosseum entity encoder ----
// Mirrors the observation layout in encounter_colosseum_obs_mask.inc. No shared
// header reaches this TU, so drift is caught only by the _Static_asserts in
// osrs_visual.c; update both when the layout changes.
//

static constexpr int COLO_ENT_OBS_SIZE    = 934;
static constexpr int COLO_ENT_NPC_START   = 130;
static constexpr int COLO_ENT_NUM_NPCS    = 24;
static constexpr int COLO_ENT_FEATS       = 34;
static constexpr int COLO_ENT_TYPE_ONEHOT = 12;
static constexpr int COLO_ENT_BOTTLENECK  = 16;
static constexpr int COLO_ENT_NPC_BLOCK   = COLO_ENT_NUM_NPCS * COLO_ENT_FEATS;
// The observation carries a type CODE per slot (0 empty, type+1 otherwise) where the
// encoder wants a one-hot. colo_ent_gather_npcs expands it, so every kernel downstream --
// including the deterministic backward -- is unchanged.
static constexpr int COLO_ENT_OBS_FEATS =
    1 + (COLO_ENT_FEATS - COLO_ENT_TYPE_ONEHOT);
static constexpr int COLO_ENT_NPC_OBS_BLOCK = COLO_ENT_NUM_NPCS * COLO_ENT_OBS_FEATS;
static constexpr int COLO_ENT_INV_START      = 36;
static constexpr int COLO_ENT_INV_NUM_CELLS  = 28;
static constexpr int COLO_ENT_INV_FEATS      = 15;
static constexpr int COLO_ENT_INV_PRESENT    = 0;
static constexpr int COLO_ENT_INV_IS_ARMOR   = 3;
static constexpr int COLO_ENT_INV_IS_WEAPON  = 4;
static constexpr int COLO_ENT_INV_BOTTLENECK = 16;
static constexpr int COLO_ENT_INV_BLOCK      = COLO_ENT_INV_NUM_CELLS * COLO_ENT_INV_FEATS;
// Each cell observes an item code plus the dynamic equipped and HP-heal fields.
// colo_ent_gather_inv always overwrites equipped and overwrites HP-heal only for non-gear.
static constexpr int COLO_ENT_INV_OBS_FEATS    = 3;
static constexpr int COLO_ENT_INV_OBS_CODE     = 0;
static constexpr int COLO_ENT_INV_OBS_EQUIPPED = 1;
static constexpr int COLO_ENT_INV_OBS_HP_HEAL  = 2;
static_assert(COLO_ENT_INV_FEATS == OSRS_ITEM_OBS_TABLE_COLS,
    "encoder record width is one item table row");

struct ColosseumEntityEncoderWeights {
    Prec global_w;
    Prec entity_l1_w;
    Prec entity_l2_w;
    Prec inv_l1_w;
    Prec inv_l2_w;
    int obs_size, hidden;
};

struct ColosseumEntityEncoderActivations {
    Prec out;
    Prec saved_obs;
    Prec npc_flat;
    Prec entity_z1;
    Prec entity_h1;
    Prec grad_z1;
    Int pool_argmax;
    Prec global_wgrad;
    Prec entity_l1_wgrad;
    Prec entity_l2_wgrad;
    Prec inv_flat;
    Prec inv_z1;
    Prec inv_h1;
    Prec inv_grad_z1;
    Int inv_pool_argmax;
    Prec inv_l1_wgrad;
    Prec inv_l2_wgrad;
};

__global__ void colo_ent_gather_npcs(
    precision_t* __restrict__ npc_flat, const precision_t* __restrict__ obs,
    int B, int obs_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * COLO_ENT_NPC_BLOCK;
    if (idx >= total) return;
    int b = idx / COLO_ENT_NPC_BLOCK;
    int off = idx % COLO_ENT_NPC_BLOCK;
    int rec = off / COLO_ENT_FEATS;
    int f = off - rec * COLO_ENT_FEATS;
    const precision_t* src = obs + (int64_t)b * obs_size + COLO_ENT_NPC_START
        + rec * COLO_ENT_OBS_FEATS;
    if (f < COLO_ENT_TYPE_ONEHOT) {
        int code = (int)lrintf(to_float(src[0]));
        npc_flat[idx] = from_float(code == f + 1 ? 1.0f : 0.0f);
    } else {
        npc_flat[idx] = src[1 + (f - COLO_ENT_TYPE_ONEHOT)];
    }
}

__device__ __forceinline__ float colo_ent_gelu_fwd(float x) {
    float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ float colo_ent_gelu_grad(float x) {
    float x3 = x * x * x;
    float inner = 0.7978845608028654f * (x + 0.044715f * x3);
    float t = tanhf(inner);
    float dinner = 0.7978845608028654f * (1.0f + 3.0f * 0.044715f * x * x);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * dinner;
}

__global__ void colo_ent_gather_inv(
    precision_t* __restrict__ inv_flat, const precision_t* __restrict__ obs,
    int B, int obs_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * COLO_ENT_INV_BLOCK;
    if (idx >= total) return;
    int b = idx / COLO_ENT_INV_BLOCK;
    int off = idx % COLO_ENT_INV_BLOCK;
    int cell = off / COLO_ENT_INV_FEATS;
    int f = off - cell * COLO_ENT_INV_FEATS;
    const precision_t* src = obs + (int64_t)b * obs_size + COLO_ENT_INV_START
        + cell * COLO_ENT_INV_OBS_FEATS;
    int code = (int)lrintf(
        to_float(src[COLO_ENT_INV_OBS_CODE]) * (float)OSRS_ITEM_OBS_CODE_SCALE);
    assert(code >= 0 && code < OSRS_ITEM_OBS_TABLE_ROWS);
    const float* table_row = OSRS_ITEM_OBS_TABLE_DEV[code];
    float v = table_row[f];
    if (f == OSRS_ITEM_OBS_OVERLAY_EQUIPPED)
        v = to_float(src[COLO_ENT_INV_OBS_EQUIPPED]);
    if (f == OSRS_ITEM_OBS_OVERLAY_HP_HEAL) {
        int is_gear = table_row[COLO_ENT_INV_IS_ARMOR] != 0.0f ||
            table_row[COLO_ENT_INV_IS_WEAPON] != 0.0f;
        if (!is_gear)
            v = to_float(src[COLO_ENT_INV_OBS_HP_HEAL]);
    }
    inv_flat[idx] = from_float(v);
}

// ---- fused pool kernels ----
static constexpr int COLO_ENT_BATCH_TILE  = 8;
static constexpr int COLO_ENT_HIDDEN_TILE = 32;
static constexpr int COLO_ENT_FC_THREADS  = COLO_ENT_BATCH_TILE * COLO_ENT_HIDDEN_TILE;

__global__ void colo_ent_fused_pool_fwd(
    precision_t* __restrict__ out, int* __restrict__ argmax,
    precision_t* __restrict__ h1,
    const precision_t* __restrict__ z1, const precision_t* __restrict__ rec_flat,
    const precision_t* __restrict__ l2_w,
    int B, int H, int num_rec, int rec_feats, int active_width) {
    extern __shared__ float colo_ent_sh[];
    float* h1_tile = colo_ent_sh;
    float* mask_tile = h1_tile + COLO_ENT_BATCH_TILE * num_rec * COLO_ENT_BOTTLENECK;
    float* w_tile = mask_tile + COLO_ENT_BATCH_TILE * num_rec;

    int batch_base = blockIdx.x * COLO_ENT_BATCH_TILE;
    int hidden_base = blockIdx.y * COLO_ENT_HIDDEN_TILE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * COLO_ENT_HIDDEN_TILE + tx;

    int h1_values = COLO_ENT_BATCH_TILE * num_rec * COLO_ENT_BOTTLENECK;
    for (int idx = tid; idx < h1_values; idx += COLO_ENT_FC_THREADS) {
        int bt = idx / (num_rec * COLO_ENT_BOTTLENECK);
        int rem = idx - bt * num_rec * COLO_ENT_BOTTLENECK;
        int gb = batch_base + bt;
        float v = gb < B
            ? colo_ent_gelu_fwd(to_float(z1[(int64_t)gb * num_rec * COLO_ENT_BOTTLENECK + rem]))
            : 0.0f;
        h1_tile[idx] = v;
        if (h1 && blockIdx.y == 0 && gb < B)
            h1[(int64_t)gb * num_rec * COLO_ENT_BOTTLENECK + rem] = from_float(v);
    }
    int mask_values = COLO_ENT_BATCH_TILE * num_rec;
    for (int idx = tid; idx < mask_values; idx += COLO_ENT_FC_THREADS) {
        int bt = idx / num_rec;
        int n = idx - bt * num_rec;
        int gb = batch_base + bt;
        float active = 0.0f;
        if (gb < B) {
            const precision_t* rec = rec_flat + ((int64_t)gb * num_rec + n) * rec_feats;
            for (int t = 0; t < active_width; t++) active += to_float(rec[t]);
        }
        mask_tile[idx] = active;
    }
    int w_values = COLO_ENT_HIDDEN_TILE * COLO_ENT_BOTTLENECK;
    for (int idx = tid; idx < w_values; idx += COLO_ENT_FC_THREADS) {
        int d = idx / COLO_ENT_HIDDEN_TILE;
        int th = idx - d * COLO_ENT_HIDDEN_TILE;
        int gh = hidden_base + th;
        w_tile[idx] = gh < H
            ? to_float(l2_w[(int64_t)gh * COLO_ENT_BOTTLENECK + d])
            : 0.0f;
    }
    __syncthreads();

    int b = batch_base + ty;
    int h = hidden_base + tx;
    if (b >= B || h >= H) return;
    float best = -3.4028234663852886e38f;
    int best_n = -1;
    for (int n = 0; n < num_rec; n++) {
        if (mask_tile[ty * num_rec + n] <= 0.0f) continue;
        const float* hp = h1_tile + (ty * num_rec + n) * COLO_ENT_BOTTLENECK;
        float sum = 0.0f;
#pragma unroll
        for (int d = 0; d < COLO_ENT_BOTTLENECK; d++)
            sum += w_tile[d * COLO_ENT_HIDDEN_TILE + tx] * hp[d];
        if (sum > best) { best = sum; best_n = n; }
    }
    int64_t o = (int64_t)b * H + h;
    out[o] = from_float(to_float(out[o]) + (best_n < 0 ? 0.0f : best));
    argmax[o] = best_n;
}

__global__ void colo_ent_fused_l2_wgrad(
    precision_t* __restrict__ wgrad, const precision_t* __restrict__ grad,
    const precision_t* __restrict__ h1, const int* __restrict__ argmax,
    int B, int H, int num_rec) {
    int h = blockIdx.x;
    if (h >= H) return;

    float sum[COLO_ENT_BOTTLENECK];
#pragma unroll
    for (int k = 0; k < COLO_ENT_BOTTLENECK; k++) sum[k] = 0.0f;

    for (int b = threadIdx.x; b < B; b += blockDim.x) {
        int n = argmax[(int64_t)b * H + h];
        if (n < 0) continue;
        float g = to_float(grad[(int64_t)b * H + h]);
        const precision_t* hp = h1 + ((int64_t)b * num_rec + n) * COLO_ENT_BOTTLENECK;
#pragma unroll
        for (int k = 0; k < COLO_ENT_BOTTLENECK; k++)
            sum[k] += g * to_float(hp[k]);
    }

    __shared__ float warp_sums[COLO_ENT_BOTTLENECK * 32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int num_warps = (blockDim.x + 31) >> 5;
#pragma unroll
    for (int k = 0; k < COLO_ENT_BOTTLENECK; k++) {
        float s = sum[k];
        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_down_sync(0xffffffff, s, offset);
        if (lane == 0) warp_sums[k * 32 + warp] = s;
    }
    __syncthreads();

    if (warp == 0) {
#pragma unroll
        for (int k = 0; k < COLO_ENT_BOTTLENECK; k++) {
            float s = lane < num_warps ? warp_sums[k * 32 + lane] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1)
                s += __shfl_down_sync(0xffffffff, s, offset);
            if (lane == 0) wgrad[(int64_t)h * COLO_ENT_BOTTLENECK + k] = from_float(s);
        }
    }
}

__global__ void colo_ent_fused_grad_z1(
    precision_t* __restrict__ grad_z1, const precision_t* __restrict__ grad,
    const precision_t* __restrict__ l2_w, const precision_t* __restrict__ z1,
    const int* __restrict__ argmax, int B, int H, int num_rec) {
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float colo_ent_sh[];
    float* accum = colo_ent_sh;
    int* arg_s = (int*)(accum + num_rec * COLO_ENT_BOTTLENECK);
    float* grad_s = (float*)(arg_s + blockDim.x);

    for (int idx = threadIdx.x; idx < num_rec * COLO_ENT_BOTTLENECK; idx += blockDim.x)
        accum[idx] = 0.0f;
    __syncthreads();

    for (int base = 0; base < H; base += blockDim.x) {
        int h = base + threadIdx.x;
        if (h < H) {
            arg_s[threadIdx.x] = argmax[(int64_t)b * H + h];
            grad_s[threadIdx.x] = to_float(grad[(int64_t)b * H + h]);
        }
        __syncthreads();

        // Gathered, not scattered: one thread owns each (record, k) accumulator and
        // walks j in order, so the summation order is fixed. An atomicAdd scatter
        // here let warp scheduling pick the order, and non-associative float
        // addition then made training irreproducible run to run.
        int tile = H - base;
        if (tile > (int)blockDim.x) tile = blockDim.x;
        for (int idx = threadIdx.x; idx < num_rec * COLO_ENT_BOTTLENECK; idx += blockDim.x) {
            int n = idx / COLO_ENT_BOTTLENECK;
            int k = idx - n * COLO_ENT_BOTTLENECK;
            float sum = 0.0f;
            for (int j = 0; j < tile; j++) {
                if (arg_s[j] != n) continue;
                sum += grad_s[j] * to_float(l2_w[(int64_t)(base + j) * COLO_ENT_BOTTLENECK + k]);
            }
            accum[idx] += sum;
        }
        __syncthreads();
    }

    for (int idx = threadIdx.x; idx < num_rec * COLO_ENT_BOTTLENECK; idx += blockDim.x) {
        int64_t o = (int64_t)b * num_rec * COLO_ENT_BOTTLENECK + idx;
        grad_z1[o] = from_float(accum[idx] * colo_ent_gelu_grad(to_float(z1[o])));
    }
}

static void colo_ent_launch_fused_fwd(
    precision_t* out, int* argmax, precision_t* h1,
    const precision_t* z1, const precision_t* rec_flat,
    const precision_t* l2_w, int B, int H, int num_rec, int rec_feats, int active_width,
    cudaStream_t stream) {
    dim3 block(COLO_ENT_HIDDEN_TILE, COLO_ENT_BATCH_TILE);
    dim3 grid((B + COLO_ENT_BATCH_TILE - 1) / COLO_ENT_BATCH_TILE,
        (H + COLO_ENT_HIDDEN_TILE - 1) / COLO_ENT_HIDDEN_TILE);
    size_t shared_bytes = (
        (size_t)COLO_ENT_BATCH_TILE * num_rec * COLO_ENT_BOTTLENECK +
        (size_t)COLO_ENT_BATCH_TILE * num_rec +
        (size_t)COLO_ENT_HIDDEN_TILE * COLO_ENT_BOTTLENECK) * sizeof(float);
    colo_ent_fused_pool_fwd<<<grid, block, shared_bytes, stream>>>(
        out, argmax, h1, z1, rec_flat, l2_w, B, H, num_rec, rec_feats, active_width);
}

static void colo_ent_launch_fused_bwd(
    precision_t* l2_wgrad, precision_t* grad_z1, const precision_t* grad,
    const precision_t* l2_w, const precision_t* z1, const precision_t* h1,
    const int* argmax, int B, int H, int num_rec, cudaStream_t stream) {
    colo_ent_fused_l2_wgrad<<<H, 256, 0, stream>>>(
        l2_wgrad, grad, h1, argmax, B, H, num_rec);
    size_t shared_bytes =
        ((size_t)num_rec * COLO_ENT_BOTTLENECK + 2 * BLOCK_SIZE) * sizeof(float);
    colo_ent_fused_grad_z1<<<B, BLOCK_SIZE, shared_bytes, stream>>>(
        grad_z1, grad, l2_w, z1, argmax, B, H, num_rec);
}

static Prec colo_entity_encoder_forward(void* w, void* activations, Prec input, cudaStream_t stream) {
    ColosseumEntityEncoderWeights* ew = (ColosseumEntityEncoderWeights*)w;
    ColosseumEntityEncoderActivations* a = (ColosseumEntityEncoderActivations*)activations;
    int B = input.shape[0];
    int H = ew->hidden;
    int NB = B * COLO_ENT_NUM_NPCS;

    if (a->saved_obs.data) puf_copy(&a->saved_obs, &input, stream);

    puf_mm(&input, &ew->global_w, &a->out, stream);

    colo_ent_gather_npcs<<<grid_size(B * COLO_ENT_NPC_BLOCK), BLOCK_SIZE, 0, stream>>>(
        a->npc_flat.data, input.data, B, ew->obs_size);

    Prec npc2d = {.data = a->npc_flat.data, .shape = {NB, COLO_ENT_FEATS}};
    puf_mm(&npc2d, &ew->entity_l1_w, &a->entity_z1, stream);
    colo_ent_launch_fused_fwd(
        a->out.data, a->pool_argmax.data, a->entity_h1.data,
        a->entity_z1.data, a->npc_flat.data,
        ew->entity_l2_w.data, B, H, COLO_ENT_NUM_NPCS, COLO_ENT_FEATS,
        COLO_ENT_TYPE_ONEHOT, stream);

    int IB = B * COLO_ENT_INV_NUM_CELLS;
    colo_ent_gather_inv<<<grid_size(B * COLO_ENT_INV_BLOCK), BLOCK_SIZE, 0, stream>>>(
        a->inv_flat.data, input.data, B, ew->obs_size);
    Prec inv2d = {.data = a->inv_flat.data, .shape = {IB, COLO_ENT_INV_FEATS}};
    puf_mm(&inv2d, &ew->inv_l1_w, &a->inv_z1, stream);
    static_assert(COLO_ENT_INV_PRESENT == 0,
        "fused pool masks the EXPANDED record, so present must be cell-local offset 0 there; "
        "the observation carries the item code at that offset instead");
    colo_ent_launch_fused_fwd(
        a->out.data, a->inv_pool_argmax.data, a->inv_h1.data,
        a->inv_z1.data, a->inv_flat.data,
        ew->inv_l2_w.data, B, H, COLO_ENT_INV_NUM_CELLS, COLO_ENT_INV_FEATS,
        1, stream);
    return a->out;
}

static void colo_entity_encoder_backward(void* w, void* activations, Prec grad, cudaStream_t stream) {
    ColosseumEntityEncoderWeights* ew = (ColosseumEntityEncoderWeights*)w;
    ColosseumEntityEncoderActivations* a = (ColosseumEntityEncoderActivations*)activations;
    int B = grad.shape[0];
    int H = ew->hidden;
    int NB = B * COLO_ENT_NUM_NPCS;

    puf_mm_tn(&grad, &a->saved_obs, &a->global_wgrad, stream);

    colo_ent_launch_fused_bwd(
        a->entity_l2_wgrad.data, a->grad_z1.data, grad.data,
        ew->entity_l2_w.data, a->entity_z1.data, a->entity_h1.data,
        a->pool_argmax.data, B, H, COLO_ENT_NUM_NPCS, stream);
    Prec npc2d = {.data = a->npc_flat.data, .shape = {NB, COLO_ENT_FEATS}};
    puf_mm_tn(&a->grad_z1, &npc2d, &a->entity_l1_wgrad, stream);

    int IB = B * COLO_ENT_INV_NUM_CELLS;
    colo_ent_launch_fused_bwd(
        a->inv_l2_wgrad.data, a->inv_grad_z1.data, grad.data,
        ew->inv_l2_w.data, a->inv_z1.data, a->inv_h1.data,
        a->inv_pool_argmax.data, B, H, COLO_ENT_INV_NUM_CELLS, stream);
    Prec inv2d = {.data = a->inv_flat.data, .shape = {IB, COLO_ENT_INV_FEATS}};
    puf_mm_tn(&a->inv_grad_z1, &inv2d, &a->inv_l1_wgrad, stream);
}

static void colo_entity_encoder_init_weights(void* w, uint64_t* seed, cudaStream_t stream) {
    ColosseumEntityEncoderWeights* ew = (ColosseumEntityEncoderWeights*)w;
    auto init2d = [&](Prec& t, int rows, int cols) {
        Prec wt = {.data = t.data, .shape = {rows, cols}};
        puf_kaiming_init(&wt, sqrtf(2.0f), (*seed)++, stream);
    };
    init2d(ew->global_w, ew->hidden, ew->obs_size);
    init2d(ew->entity_l1_w, COLO_ENT_BOTTLENECK, COLO_ENT_FEATS);
    init2d(ew->entity_l2_w, ew->hidden, COLO_ENT_BOTTLENECK);
    init2d(ew->inv_l1_w, COLO_ENT_INV_BOTTLENECK, COLO_ENT_INV_FEATS);
    init2d(ew->inv_l2_w, ew->hidden, COLO_ENT_INV_BOTTLENECK);
}

static void colo_entity_assert_aligned(int64_t numel, const char* name) {
    if (numel % 8 != 0) {
        fprintf(stderr, "colosseum entity encoder: %s numel %lld not a multiple of 8; "
            "bf16 packing would corrupt weights\n", name, (long long)numel);
        abort();
    }
}

static void colo_entity_encoder_reg_params(void* w, Allocator* alloc) {
    ColosseumEntityEncoderWeights* ew = (ColosseumEntityEncoderWeights*)w;
    ew->global_w    = {.shape = {ew->hidden, ew->obs_size}};
    ew->entity_l1_w = {.shape = {COLO_ENT_BOTTLENECK, COLO_ENT_FEATS}};
    ew->entity_l2_w = {.shape = {ew->hidden, COLO_ENT_BOTTLENECK}};
    colo_entity_assert_aligned(numel(ew->global_w.shape), "global_w");
    colo_entity_assert_aligned(numel(ew->entity_l1_w.shape), "entity_l1_w");
    colo_entity_assert_aligned(numel(ew->entity_l2_w.shape), "entity_l2_w");
    alloc_register(alloc, &ew->global_w);
    alloc_register(alloc, &ew->entity_l1_w);
    alloc_register(alloc, &ew->entity_l2_w);
    ew->inv_l1_w = {.shape = {COLO_ENT_INV_BOTTLENECK, COLO_ENT_INV_FEATS}};
    ew->inv_l2_w = {.shape = {ew->hidden, COLO_ENT_INV_BOTTLENECK}};
    colo_entity_assert_aligned(numel(ew->inv_l1_w.shape), "inv_l1_w");
    colo_entity_assert_aligned(numel(ew->inv_l2_w.shape), "inv_l2_w");
    alloc_register(alloc, &ew->inv_l1_w);
    alloc_register(alloc, &ew->inv_l2_w);
}

static void colo_entity_encoder_reg_train(void* w, void* activations, Allocator* acts, Allocator* grads, int B_TT) {
    ColosseumEntityEncoderWeights* ew = (ColosseumEntityEncoderWeights*)w;
    ColosseumEntityEncoderActivations* a = (ColosseumEntityEncoderActivations*)activations;
    int H = ew->hidden;
    int NB = B_TT * COLO_ENT_NUM_NPCS;
    *a = {};
    a->out        = {.shape = {B_TT, H}};
    a->saved_obs  = {.shape = {B_TT, ew->obs_size}};
    a->npc_flat   = {.shape = {NB, COLO_ENT_FEATS}};
    a->entity_z1  = {.shape = {NB, COLO_ENT_BOTTLENECK}};
    a->entity_h1  = {.shape = {NB, COLO_ENT_BOTTLENECK}};
    a->grad_z1    = {.shape = {NB, COLO_ENT_BOTTLENECK}};
    a->pool_argmax = {.shape = {B_TT, H}};
    alloc_register(acts, &a->out);
    alloc_register(acts, &a->saved_obs);
    alloc_register(acts, &a->npc_flat);
    alloc_register(acts, &a->entity_z1);
    alloc_register(acts, &a->entity_h1);
    alloc_register(acts, &a->grad_z1);
    alloc_register(acts, &a->pool_argmax);
    int IB = B_TT * COLO_ENT_INV_NUM_CELLS;
    a->inv_flat        = {.shape = {IB, COLO_ENT_INV_FEATS}};
    a->inv_z1          = {.shape = {IB, COLO_ENT_INV_BOTTLENECK}};
    a->inv_h1          = {.shape = {IB, COLO_ENT_INV_BOTTLENECK}};
    a->inv_grad_z1     = {.shape = {IB, COLO_ENT_INV_BOTTLENECK}};
    a->inv_pool_argmax = {.shape = {B_TT, H}};
    alloc_register(acts, &a->inv_flat);
    alloc_register(acts, &a->inv_z1);
    alloc_register(acts, &a->inv_h1);
    alloc_register(acts, &a->inv_grad_z1);
    alloc_register(acts, &a->inv_pool_argmax);
    a->global_wgrad    = {.shape = {H, ew->obs_size}};
    a->entity_l1_wgrad = {.shape = {COLO_ENT_BOTTLENECK, COLO_ENT_FEATS}};
    a->entity_l2_wgrad = {.shape = {H, COLO_ENT_BOTTLENECK}};
    alloc_register(grads, &a->global_wgrad);
    alloc_register(grads, &a->entity_l1_wgrad);
    alloc_register(grads, &a->entity_l2_wgrad);
    a->inv_l1_wgrad = {.shape = {COLO_ENT_INV_BOTTLENECK, COLO_ENT_INV_FEATS}};
    a->inv_l2_wgrad = {.shape = {H, COLO_ENT_INV_BOTTLENECK}};
    alloc_register(grads, &a->inv_l1_wgrad);
    alloc_register(grads, &a->inv_l2_wgrad);
}

static void colo_entity_encoder_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    ColosseumEntityEncoderWeights* ew = (ColosseumEntityEncoderWeights*)w;
    ColosseumEntityEncoderActivations* a = (ColosseumEntityEncoderActivations*)activations;
    int H = ew->hidden;
    int NB = B * COLO_ENT_NUM_NPCS;
    a->out        = {.shape = {B, H}};
    a->npc_flat   = {.shape = {NB, COLO_ENT_FEATS}};
    a->entity_z1  = {.shape = {NB, COLO_ENT_BOTTLENECK}};
    a->pool_argmax = {.shape = {B, H}};
    alloc_register(alloc, &a->out);
    alloc_register(alloc, &a->npc_flat);
    alloc_register(alloc, &a->entity_z1);
    alloc_register(alloc, &a->pool_argmax);
    int IB = B * COLO_ENT_INV_NUM_CELLS;
    a->inv_flat        = {.shape = {IB, COLO_ENT_INV_FEATS}};
    a->inv_z1          = {.shape = {IB, COLO_ENT_INV_BOTTLENECK}};
    a->inv_pool_argmax = {.shape = {B, H}};
    alloc_register(alloc, &a->inv_flat);
    alloc_register(alloc, &a->inv_z1);
    alloc_register(alloc, &a->inv_pool_argmax);
}

static void* colo_entity_encoder_create_weights(void* self) {
    Encoder* e = (Encoder*)self;
    ColosseumEntityEncoderWeights* ew =
        (ColosseumEntityEncoderWeights*)calloc(1, sizeof(ColosseumEntityEncoderWeights));
    ew->obs_size = e->in_dim;
    ew->hidden = e->out_dim;
    return ew;
}

// ---- Inferno entity encoder ----
// Mirrors the observation layout in encounter_inferno_obs_mask.inc, whose block
// offsets are the INF_OBS_AFTER_* macros in encounter_inferno_forecast.inc. No
// shared header reaches this TU, so drift is caught only by the _Static_asserts
// in osrs_visual.c; update both when the layout changes.

static void create_osrs_colosseum_encoder(Encoder* enc) {
    *enc = Encoder{
        .forward = colo_entity_encoder_forward,
        .backward = colo_entity_encoder_backward,
        .init_weights = colo_entity_encoder_init_weights,
        .reg_params = colo_entity_encoder_reg_params,
        .reg_train = colo_entity_encoder_reg_train,
        .reg_rollout = colo_entity_encoder_reg_rollout,
        .create_weights = colo_entity_encoder_create_weights,
        .in_dim = enc->in_dim, .out_dim = enc->out_dim,
        .activation_size = sizeof(ColosseumEntityEncoderActivations),
    };
}
