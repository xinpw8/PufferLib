// OSRS Inferno CUDA entity encoder.
// Included by src/ocean.cu — requires precision_t, Prec, Allocator, puf_mm, etc.
// Shares ColosseumEntityEncoderWeights/Activations layout (same weight slots).

static constexpr int INF_ENT_NPC_START       = 54;
static constexpr int INF_ENT_NUM_NPCS        = 14;
static constexpr int INF_ENT_OBS_FEATS       = 13;
static constexpr int INF_ENT_FEATS           = 26;
static constexpr int INF_ENT_TYPE_ONEHOT     = 14;
static constexpr int INF_ENT_TYPE_CODE_SCALE = 16;
static constexpr int INF_ENT_NPC_BLOCK       = INF_ENT_NUM_NPCS * INF_ENT_FEATS;
static constexpr int INF_ENT_INV_START       = 460;
static constexpr int INF_ENT_INV_NUM_CELLS   = 28;
static constexpr int INF_ENT_INV_OBS_FEATS   = 1;
static constexpr int INF_ENT_INV_FEATS       = 15;
static constexpr int INF_ENT_INV_BLOCK       = INF_ENT_INV_NUM_CELLS * INF_ENT_INV_FEATS;
static constexpr int INF_ENT_OBS_SIZE        = 498;

static_assert(INF_ENT_FEATS ==
    INF_ENT_TYPE_ONEHOT + INF_ENT_OBS_FEATS - 1);
static_assert(INF_ENT_INV_FEATS == OSRS_ITEM_OBS_TABLE_COLS);

__global__ void inf_ent_gather_npcs(
    precision_t* __restrict__ npc_flat, const precision_t* __restrict__ obs,
    int B, int obs_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * INF_ENT_NPC_BLOCK;
    if (idx >= total) return;
    int b = idx / INF_ENT_NPC_BLOCK;
    int off = idx % INF_ENT_NPC_BLOCK;
    int rec = off / INF_ENT_FEATS;
    int f = off - rec * INF_ENT_FEATS;
    const precision_t* src = obs + (int64_t)b * obs_size + INF_ENT_NPC_START
        + rec * INF_ENT_OBS_FEATS;
    if (f < INF_ENT_TYPE_ONEHOT) {
        int code = (int)lrintf(
            to_float(src[0]) * (float)INF_ENT_TYPE_CODE_SCALE);
        assert(code >= 0 && code <= INF_ENT_TYPE_ONEHOT);
        npc_flat[idx] = from_float(code == f + 1 ? 1.0f : 0.0f);
    } else {
        npc_flat[idx] = src[1 + f - INF_ENT_TYPE_ONEHOT];
    }
}

__global__ void inf_ent_gather_inv(
    precision_t* __restrict__ inv_flat, const precision_t* __restrict__ obs,
    int B, int obs_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * INF_ENT_INV_BLOCK;
    if (idx >= total) return;
    int b = idx / INF_ENT_INV_BLOCK;
    int off = idx % INF_ENT_INV_BLOCK;
    int cell = off / INF_ENT_INV_FEATS;
    int f = off - cell * INF_ENT_INV_FEATS;
    const precision_t* src = obs + (int64_t)b * obs_size + INF_ENT_INV_START
        + cell * INF_ENT_INV_OBS_FEATS;
    int code = (int)lrintf(
        to_float(src[0]) * (float)OSRS_ITEM_OBS_CODE_SCALE);
    assert(code >= 0 && code < OSRS_ITEM_OBS_TABLE_ROWS);
    inv_flat[idx] = from_float(OSRS_ITEM_OBS_TABLE_DEV[code][f]);
}

static Prec inf_entity_encoder_forward(void* w, void* activations, Prec input, cudaStream_t stream) {
    ColosseumEntityEncoderWeights* ew = (ColosseumEntityEncoderWeights*)w;
    ColosseumEntityEncoderActivations* a = (ColosseumEntityEncoderActivations*)activations;
    int B = input.shape[0];
    int H = ew->hidden;
    int NB = B * INF_ENT_NUM_NPCS;

    if (a->saved_obs.data) puf_copy(&a->saved_obs, &input, stream);

    puf_mm(&input, &ew->global_w, &a->out, stream);

    inf_ent_gather_npcs<<<grid_size(B * INF_ENT_NPC_BLOCK), BLOCK_SIZE, 0, stream>>>(
        a->npc_flat.data, input.data, B, ew->obs_size);
    Prec npc2d = {.data = a->npc_flat.data, .shape = {NB, INF_ENT_FEATS}};
    puf_mm(&npc2d, &ew->entity_l1_w, &a->entity_z1, stream);
    colo_ent_launch_fused_fwd(
        a->out.data, a->pool_argmax.data, a->entity_h1.data,
        a->entity_z1.data, a->npc_flat.data,
        ew->entity_l2_w.data, B, H, INF_ENT_NUM_NPCS, INF_ENT_FEATS,
        INF_ENT_TYPE_ONEHOT, stream);

    int IB = B * INF_ENT_INV_NUM_CELLS;
    inf_ent_gather_inv<<<grid_size(B * INF_ENT_INV_BLOCK), BLOCK_SIZE, 0, stream>>>(
        a->inv_flat.data, input.data, B, ew->obs_size);
    Prec inv2d = {.data = a->inv_flat.data, .shape = {IB, INF_ENT_INV_FEATS}};
    puf_mm(&inv2d, &ew->inv_l1_w, &a->inv_z1, stream);
    colo_ent_launch_fused_fwd(
        a->out.data, a->inv_pool_argmax.data, a->inv_h1.data,
        a->inv_z1.data, a->inv_flat.data,
        ew->inv_l2_w.data, B, H, INF_ENT_INV_NUM_CELLS, INF_ENT_INV_FEATS,
        1, stream);
    return a->out;
}

static void inf_entity_encoder_backward(void* w, void* activations, Prec grad, cudaStream_t stream) {
    ColosseumEntityEncoderWeights* ew = (ColosseumEntityEncoderWeights*)w;
    ColosseumEntityEncoderActivations* a = (ColosseumEntityEncoderActivations*)activations;
    int B = grad.shape[0];
    int H = ew->hidden;
    int NB = B * INF_ENT_NUM_NPCS;

    puf_mm_tn(&grad, &a->saved_obs, &a->global_wgrad, stream);

    colo_ent_launch_fused_bwd(
        a->entity_l2_wgrad.data, a->grad_z1.data, grad.data,
        ew->entity_l2_w.data, a->entity_z1.data, a->entity_h1.data,
        a->pool_argmax.data, B, H, INF_ENT_NUM_NPCS, stream);
    Prec npc2d = {.data = a->npc_flat.data, .shape = {NB, INF_ENT_FEATS}};
    puf_mm_tn(&a->grad_z1, &npc2d, &a->entity_l1_wgrad, stream);

    int IB = B * INF_ENT_INV_NUM_CELLS;
    colo_ent_launch_fused_bwd(
        a->inv_l2_wgrad.data, a->inv_grad_z1.data, grad.data,
        ew->inv_l2_w.data, a->inv_z1.data, a->inv_h1.data,
        a->inv_pool_argmax.data, B, H, INF_ENT_INV_NUM_CELLS, stream);
    Prec inv2d = {.data = a->inv_flat.data, .shape = {IB, INF_ENT_INV_FEATS}};
    puf_mm_tn(&a->inv_grad_z1, &inv2d, &a->inv_l1_wgrad, stream);
}

static void inf_entity_encoder_init_weights(void* w, uint64_t* seed, cudaStream_t stream) {
    ColosseumEntityEncoderWeights* ew = (ColosseumEntityEncoderWeights*)w;
    auto init2d = [&](Prec& t, int rows, int cols) {
        Prec wt = {.data = t.data, .shape = {rows, cols}};
        puf_kaiming_init(&wt, sqrtf(2.0f), (*seed)++, stream);
    };
    init2d(ew->global_w, ew->hidden, ew->obs_size);
    init2d(ew->entity_l1_w, COLO_ENT_BOTTLENECK, INF_ENT_FEATS);
    init2d(ew->entity_l2_w, ew->hidden, COLO_ENT_BOTTLENECK);
    init2d(ew->inv_l1_w, COLO_ENT_INV_BOTTLENECK, INF_ENT_INV_FEATS);
    init2d(ew->inv_l2_w, ew->hidden, COLO_ENT_INV_BOTTLENECK);
}

static void inf_entity_encoder_reg_params(void* w, Allocator* alloc) {
    ColosseumEntityEncoderWeights* ew = (ColosseumEntityEncoderWeights*)w;
    ew->global_w    = {.shape = {ew->hidden, ew->obs_size}};
    ew->entity_l1_w = {.shape = {COLO_ENT_BOTTLENECK, INF_ENT_FEATS}};
    ew->entity_l2_w = {.shape = {ew->hidden, COLO_ENT_BOTTLENECK}};
    colo_entity_assert_aligned(numel(ew->global_w.shape), "global_w");
    colo_entity_assert_aligned(numel(ew->entity_l1_w.shape), "entity_l1_w");
    colo_entity_assert_aligned(numel(ew->entity_l2_w.shape), "entity_l2_w");
    alloc_register(alloc, &ew->global_w);
    alloc_register(alloc, &ew->entity_l1_w);
    alloc_register(alloc, &ew->entity_l2_w);
    ew->inv_l1_w = {.shape = {COLO_ENT_INV_BOTTLENECK, INF_ENT_INV_FEATS}};
    ew->inv_l2_w = {.shape = {ew->hidden, COLO_ENT_INV_BOTTLENECK}};
    colo_entity_assert_aligned(numel(ew->inv_l1_w.shape), "inv_l1_w");
    colo_entity_assert_aligned(numel(ew->inv_l2_w.shape), "inv_l2_w");
    alloc_register(alloc, &ew->inv_l1_w);
    alloc_register(alloc, &ew->inv_l2_w);
}

static void inf_entity_encoder_reg_train(void* w, void* activations, Allocator* acts, Allocator* grads, int B_TT) {
    ColosseumEntityEncoderWeights* ew = (ColosseumEntityEncoderWeights*)w;
    ColosseumEntityEncoderActivations* a = (ColosseumEntityEncoderActivations*)activations;
    int H = ew->hidden;
    int NB = B_TT * INF_ENT_NUM_NPCS;
    *a = {};
    a->out        = {.shape = {B_TT, H}};
    a->saved_obs  = {.shape = {B_TT, ew->obs_size}};
    a->npc_flat   = {.shape = {NB, INF_ENT_FEATS}};
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
    int IB = B_TT * INF_ENT_INV_NUM_CELLS;
    a->inv_flat        = {.shape = {IB, INF_ENT_INV_FEATS}};
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
    a->entity_l1_wgrad = {.shape = {COLO_ENT_BOTTLENECK, INF_ENT_FEATS}};
    a->entity_l2_wgrad = {.shape = {H, COLO_ENT_BOTTLENECK}};
    alloc_register(grads, &a->global_wgrad);
    alloc_register(grads, &a->entity_l1_wgrad);
    alloc_register(grads, &a->entity_l2_wgrad);
    a->inv_l1_wgrad = {.shape = {COLO_ENT_INV_BOTTLENECK, INF_ENT_INV_FEATS}};
    a->inv_l2_wgrad = {.shape = {H, COLO_ENT_INV_BOTTLENECK}};
    alloc_register(grads, &a->inv_l1_wgrad);
    alloc_register(grads, &a->inv_l2_wgrad);
}

static void inf_entity_encoder_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    ColosseumEntityEncoderWeights* ew = (ColosseumEntityEncoderWeights*)w;
    ColosseumEntityEncoderActivations* a = (ColosseumEntityEncoderActivations*)activations;
    int H = ew->hidden;
    int NB = B * INF_ENT_NUM_NPCS;
    a->out        = {.shape = {B, H}};
    a->npc_flat   = {.shape = {NB, INF_ENT_FEATS}};
    a->entity_z1  = {.shape = {NB, COLO_ENT_BOTTLENECK}};
    a->pool_argmax = {.shape = {B, H}};
    alloc_register(alloc, &a->out);
    alloc_register(alloc, &a->npc_flat);
    alloc_register(alloc, &a->entity_z1);
    alloc_register(alloc, &a->pool_argmax);
    int IB = B * INF_ENT_INV_NUM_CELLS;
    a->inv_flat        = {.shape = {IB, INF_ENT_INV_FEATS}};
    a->inv_z1          = {.shape = {IB, COLO_ENT_INV_BOTTLENECK}};
    a->inv_pool_argmax = {.shape = {B, H}};
    alloc_register(alloc, &a->inv_flat);
    alloc_register(alloc, &a->inv_z1);
    alloc_register(alloc, &a->inv_pool_argmax);
}

static void* inf_entity_encoder_create_weights(void* self) {
    Encoder* e = (Encoder*)self;
    if (e->in_dim != INF_ENT_OBS_SIZE) {
        fprintf(stderr, "inferno entity encoder: env obs size %d != encoder slice "
            "expectation %d; INF_ENT_* constants are stale\n",
            e->in_dim, INF_ENT_OBS_SIZE);
        abort();
    }
    ColosseumEntityEncoderWeights* ew =
        (ColosseumEntityEncoderWeights*)calloc(1, sizeof(ColosseumEntityEncoderWeights));
    ew->obs_size = e->in_dim;
    ew->hidden = e->out_dim;
    return ew;
}

static void create_osrs_inferno_encoder(Encoder* enc) {
    *enc = Encoder{
        .forward = inf_entity_encoder_forward,
        .backward = inf_entity_encoder_backward,
        .init_weights = inf_entity_encoder_init_weights,
        .reg_params = inf_entity_encoder_reg_params,
        .reg_train = inf_entity_encoder_reg_train,
        .reg_rollout = inf_entity_encoder_reg_rollout,
        .create_weights = inf_entity_encoder_create_weights,
        .in_dim = enc->in_dim, .out_dim = enc->out_dim,
        .activation_size = sizeof(ColosseumEntityEncoderActivations),
    };
}
