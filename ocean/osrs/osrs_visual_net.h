#pragma once

#include "osrs_colosseum_item_obs_generated.h"

/* 5c puffercpu.h dropped _gelu. Keep a local twin so the viewer encoder stays
   bit-compatible with the CUDA GELU without carrying a core patch. */
static inline void osrs_visual_gelu(float* input, float* output, int size) {
    for (int i = 0; i < size; i++) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    }
}

#define COLO_ENT_INF_NPC_START   130
#define COLO_ENT_INF_NUM_NPCS    24
#define COLO_ENT_INF_FEATS       34
/* obs carries a type CODE per slot; the encoder expands it back to the one-hot */
#define COLO_ENT_INF_OBS_FEATS   23
#define COLO_ENT_INF_TYPE_ONEHOT 12
#define COLO_ENT_INF_BOTTLENECK  16
#define COLO_ENT_INF_INV_START      36
#define COLO_ENT_INF_INV_NUM_CELLS  28
#define COLO_ENT_INF_INV_FEATS      15
/* obs carries an item CODE plus is_equipped and hp_heal; the encoder rebuilds the rest */
#define COLO_ENT_INF_INV_OBS_FEATS  3
#define COLO_ENT_INF_INV_BOTTLENECK 16

/* Host twin of COLO_ITEM_OBS_TABLE_DEV in src/ocean.cu. Separate names on purpose: both
   land in one translation unit when the viewer is built, and a single guarded definition
   would give the CUDA kernel a host array. */
static const float COLO_ITEM_OBS_TABLE
    [COLO_ITEM_OBS_TABLE_ROWS][COLO_ITEM_OBS_TABLE_COLS] = {
#include "osrs_colosseum_item_obs_table.inc"
};

#define INF_ENT_NPC_START   90
#define INF_ENT_NUM_NPCS    37
#define INF_ENT_FEATS       48
#define INF_ENT_TYPE_ONEHOT 14
#define INF_ENT_INV_START     2450
#define INF_ENT_INV_NUM_CELLS 28
#define INF_ENT_INV_FEATS     28

/* How the observation record becomes the encoder record. Every branch materialises the
   expanded record first and then runs one dense dot, which is what the CUDA gathers do. */
typedef enum {
    ENTITY_RECORD_VERBATIM = 0,
    ENTITY_RECORD_TYPE_ONEHOT,
    ENTITY_RECORD_ITEM_TABLE,
} EntityRecordExpansion;

#define ENTITY_RECORD_MAX_FEATS 64

typedef struct EntityPoolBranch EntityPoolBranch;
struct EntityPoolBranch {
    int start;
    int num_recs;
    int feats;
    int obs_feats;
    int type_onehot;
    int bottleneck;
    int mask_prefix;
    EntityRecordExpansion expansion;
    float* l1_w;
    float* l2_w;
    float* z1;
    float* h1;
    float* e;
};

typedef struct EntityEncoder EntityEncoder;
struct EntityEncoder {
    float* output;
    float* global_w;
    int batch_size;
    int input_dim;
    int hidden_dim;
    int num_branches;
    EntityPoolBranch branches[2];
};

static void entity_pool_branch_init(
        EntityPoolBranch* branch, Weights* weights, int hidden_dim,
        int start, int num_recs, int feats, int obs_feats, int type_onehot,
        int bottleneck, int mask_prefix, EntityRecordExpansion expansion) {
    if (feats > ENTITY_RECORD_MAX_FEATS) {
        fprintf(stderr, "entity pool branch: %d features exceeds the expansion scratch\n",
            feats);
        abort();
    }
    branch->start = start;
    branch->num_recs = num_recs;
    branch->feats = feats;
    branch->obs_feats = obs_feats;
    branch->type_onehot = type_onehot;
    branch->bottleneck = bottleneck;
    branch->mask_prefix = mask_prefix;
    branch->expansion = expansion;
    branch->l1_w = get_weights_aligned(weights, bottleneck * feats);
    branch->l2_w = get_weights_aligned(weights, hidden_dim * bottleneck);
    branch->z1 = (float*)calloc((size_t)num_recs * bottleneck, sizeof(float));
    branch->h1 = (float*)calloc((size_t)num_recs * bottleneck, sizeof(float));
    branch->e = (float*)calloc((size_t)num_recs * hidden_dim, sizeof(float));
}

/* Weight reads are sequenced statements in reg_params order (src/ocean.cu):
   global_w, then entity_l1_w/entity_l2_w, then inv_l1_w/inv_l2_w.
   get_weights_aligned advances a shared cursor, so call order IS the .bin layout. */
static EntityEncoder* make_entity_encoder_global(
        Weights* weights, int batch_size, int input_dim, int hidden_dim) {
    size_t out_size = (size_t)batch_size * hidden_dim * sizeof(float);
    EntityEncoder* layer = (EntityEncoder*)calloc(1, sizeof(EntityEncoder) + out_size);
    layer->output = (float*)(layer + 1);
    layer->global_w = get_weights_aligned(weights, hidden_dim * input_dim);
    layer->batch_size = batch_size;
    layer->input_dim = input_dim;
    layer->hidden_dim = hidden_dim;
    return layer;
}

EntityEncoder* make_colosseum_entity_encoder(
        Weights* weights, int batch_size, int input_dim, int hidden_dim, int mode) {
    EntityEncoder* layer = make_entity_encoder_global(
        weights, batch_size, input_dim, hidden_dim);
    entity_pool_branch_init(&layer->branches[0], weights, hidden_dim,
        COLO_ENT_INF_NPC_START, COLO_ENT_INF_NUM_NPCS,
        COLO_ENT_INF_FEATS, COLO_ENT_INF_OBS_FEATS, COLO_ENT_INF_TYPE_ONEHOT,
        COLO_ENT_INF_BOTTLENECK, COLO_ENT_INF_TYPE_ONEHOT,
        ENTITY_RECORD_TYPE_ONEHOT);
    layer->num_branches = 1;
    if (mode >= 2) {
        entity_pool_branch_init(&layer->branches[1], weights, hidden_dim,
            COLO_ENT_INF_INV_START, COLO_ENT_INF_INV_NUM_CELLS,
            COLO_ENT_INF_INV_FEATS, COLO_ENT_INF_INV_OBS_FEATS, 0,
            COLO_ENT_INF_INV_BOTTLENECK, 1, ENTITY_RECORD_ITEM_TABLE);
        layer->num_branches = 2;
    }
    return layer;
}

EntityEncoder* make_inferno_entity_encoder(
        Weights* weights, int batch_size, int input_dim, int hidden_dim, int mode) {
    EntityEncoder* layer = make_entity_encoder_global(
        weights, batch_size, input_dim, hidden_dim);
    entity_pool_branch_init(&layer->branches[0], weights, hidden_dim,
        INF_ENT_NPC_START, INF_ENT_NUM_NPCS,
        INF_ENT_FEATS, INF_ENT_FEATS, 0,
        COLO_ENT_INF_BOTTLENECK, INF_ENT_TYPE_ONEHOT, ENTITY_RECORD_VERBATIM);
    layer->num_branches = 1;
    if (mode >= 2) {
        entity_pool_branch_init(&layer->branches[1], weights, hidden_dim,
            INF_ENT_INV_START, INF_ENT_INV_NUM_CELLS,
            INF_ENT_INV_FEATS, INF_ENT_INV_FEATS, 0,
            COLO_ENT_INF_INV_BOTTLENECK, 1, ENTITY_RECORD_VERBATIM);
        layer->num_branches = 2;
    }
    return layer;
}

/* Mirrors colo_ent_gather_npcs and colo_ent_gather_inv in src/ocean.cu. */
static void entity_expand_record(const EntityPoolBranch* p, const float* rec, float* out) {
    switch (p->expansion) {
        case ENTITY_RECORD_VERBATIM:
            for (int i = 0; i < p->feats; i++) out[i] = rec[i];
            return;
        case ENTITY_RECORD_TYPE_ONEHOT: {
            int code = (int)lrintf(rec[0]);
            for (int i = 0; i < p->type_onehot; i++)
                out[i] = (code == i + 1) ? 1.0f : 0.0f;
            for (int i = 0; i < p->feats - p->type_onehot; i++)
                out[p->type_onehot + i] = rec[1 + i];
            return;
        }
        case ENTITY_RECORD_ITEM_TABLE: {
            int code = (int)lrintf(rec[0] * (float)COLO_ITEM_OBS_CODE_SCALE);
            if (code < 0 || code >= COLO_ITEM_OBS_TABLE_ROWS) {
                fprintf(stderr, "entity pool branch: item code %d out of table\n", code);
                abort();
            }
            for (int i = 0; i < p->feats; i++) out[i] = COLO_ITEM_OBS_TABLE[code][i];
            out[COLO_ITEM_OBS_OVERLAY_EQUIPPED] += rec[1];
            out[COLO_ITEM_OBS_OVERLAY_HP_HEAL] += rec[2];
            return;
        }
    }
}

void entity_encoder_forward(EntityEncoder* layer, float* observations) {
    int H = layer->hidden_dim;
    int IN = layer->input_dim;
    for (int b = 0; b < layer->batch_size; b++) {
        float* obs = observations + (size_t)b * IN;
        float* out = layer->output + (size_t)b * H;

        for (int o = 0; o < H; o++) {
            float sum = 0.0f;
            for (int i = 0; i < IN; i++) sum += obs[i] * layer->global_w[o * IN + i];
            out[o] = sum;
        }

        for (int br = 0; br < layer->num_branches; br++) {
            EntityPoolBranch* p = &layer->branches[br];
            float* recs = obs + p->start;
            float expanded[ENTITY_RECORD_MAX_FEATS];
            for (int n = 0; n < p->num_recs; n++) {
                float* rec = recs + n * p->obs_feats;
                float* z1n = p->z1 + n * p->bottleneck;
                entity_expand_record(p, rec, expanded);
                for (int k = 0; k < p->bottleneck; k++) {
                    const float* w = p->l1_w + k * p->feats;
                    float sum = 0.0f;
                    for (int i = 0; i < p->feats; i++) sum += expanded[i] * w[i];
                    z1n[k] = sum;
                }
            }
            osrs_visual_gelu(p->z1, p->h1, p->num_recs * p->bottleneck);
            for (int n = 0; n < p->num_recs; n++) {
                float* h1n = p->h1 + n * p->bottleneck;
                float* en = p->e + (size_t)n * H;
                for (int o = 0; o < H; o++) {
                    float sum = 0.0f;
                    for (int k = 0; k < p->bottleneck; k++)
                        sum += h1n[k] * p->l2_w[o * p->bottleneck + k];
                    en[o] = sum;
                }
            }
            for (int o = 0; o < H; o++) {
                float best = -INFINITY;
                int best_n = -1;
                for (int n = 0; n < p->num_recs; n++) {
                    float* rec = recs + n * p->obs_feats;
                    /* A coded record is live when its code is nonzero; a verbatim one
                       when its presence prefix is. Both mirror the fused pool's mask. */
                    float mask_sum = 0.0f;
                    if (p->expansion != ENTITY_RECORD_VERBATIM)
                        mask_sum = rec[0] > 0.0f ? 1.0f : 0.0f;
                    else for (int t = 0; t < p->mask_prefix; t++) mask_sum += rec[t];
                    if (mask_sum <= 0.0f) continue;
                    float v = p->e[(size_t)n * H + o];
                    if (v > best) { best = v; best_n = n; }
                }
                out[o] += (best_n < 0) ? 0.0f : best;
            }
        }
    }
}

void free_entity_encoder(EntityEncoder* layer) {
    for (int br = 0; br < layer->num_branches; br++) {
        free(layer->branches[br].z1);
        free(layer->branches[br].h1);
        free(layer->branches[br].e);
    }
    free(layer);
}

typedef struct VisualNet VisualNet;
struct VisualNet {
    int num_agents;
    float* obs;
    Linear* encoder;
    EntityEncoder* entity_encoder;
    MinGRU* mingru;
    Linear* decoder;
    float* log_std;
    int is_continuous;
    int num_actions;
    Multidiscrete* multidiscrete;
};

void visual_net_free(VisualNet* net) {
    free(net->obs);
    if (net->encoder) free(net->encoder);
    if (net->entity_encoder) free_entity_encoder(net->entity_encoder);
    free(net->decoder);
    free_mingru(net->mingru);
    if (net->multidiscrete) free(net->multidiscrete);
    free(net);
}
