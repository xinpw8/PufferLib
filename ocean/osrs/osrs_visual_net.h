#pragma once

#include "osrs_item_obs_generated.h"

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
/* obs carries an item code plus equipped and HP-heal; gear keeps the table's union slot */
#define COLO_ENT_INF_INV_OBS_FEATS  3
#define COLO_ENT_INF_INV_BOTTLENECK 16

/* Host twin of OSRS_ITEM_OBS_TABLE_DEV in src/ocean.cu. */
static const float OSRS_ITEM_OBS_TABLE
    [OSRS_ITEM_OBS_TABLE_ROWS][OSRS_ITEM_OBS_TABLE_COLS] = {
#include "osrs_item_obs_table.inc"
};

#define INF_ENT_OBS_SIZE          498
#define INF_ENT_NPC_START         54
#define INF_ENT_NUM_NPCS          14
#define INF_ENT_OBS_FEATS         13
#define INF_ENT_FEATS             26
#define INF_ENT_TYPE_ONEHOT       14
#define INF_ENT_TYPE_CODE_SCALE   16
#define INF_ENT_INV_START         460
#define INF_ENT_INV_NUM_CELLS     28
#define INF_ENT_INV_OBS_FEATS     1
#define INF_ENT_INV_FEATS         15

typedef enum {
    ENTITY_RECORD_TYPE_ONEHOT = 0,
    ENTITY_RECORD_ITEM_TABLE,
} EntityRecordExpansion;

typedef enum {
    ENTITY_ITEM_OVERLAYS_NONE = 0,
    ENTITY_ITEM_OVERLAYS_EQUIPPED_HP_HEAL,
} EntityItemOverlays;

#define ENTITY_RECORD_MAX_FEATS 64

typedef struct EntityPoolBranch EntityPoolBranch;
struct EntityPoolBranch {
    int start;
    int num_recs;
    int feats;
    int obs_feats;
    int type_onehot;
    int code_scale;
    int bottleneck;
    int active_width;
    EntityRecordExpansion expansion;
    EntityItemOverlays item_overlays;
    float* l1_w;
    float* l2_w;
    float* z1;
    float* h1;
    float* e;
    unsigned char* active;
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
        int code_scale, int bottleneck, int active_width,
        EntityRecordExpansion expansion, EntityItemOverlays item_overlays) {
    if (feats > ENTITY_RECORD_MAX_FEATS || start < 0 || num_recs <= 0 ||
            obs_feats <= 0 || code_scale <= 0 || bottleneck <= 0 ||
            active_width <= 0 || active_width > feats) {
        fprintf(stderr, "entity pool branch: invalid shape or encoding contract\n");
        abort();
    }
    if (expansion == ENTITY_RECORD_TYPE_ONEHOT &&
            (type_onehot <= 0 || feats != type_onehot + obs_feats - 1 ||
             item_overlays != ENTITY_ITEM_OVERLAYS_NONE)) {
        fprintf(stderr, "entity pool branch: stale type-code expansion contract\n");
        abort();
    }
    if (expansion == ENTITY_RECORD_ITEM_TABLE &&
            (type_onehot != 0 || feats != OSRS_ITEM_OBS_TABLE_COLS ||
             (item_overlays == ENTITY_ITEM_OVERLAYS_NONE && obs_feats != 1) ||
             (item_overlays == ENTITY_ITEM_OVERLAYS_EQUIPPED_HP_HEAL &&
              obs_feats != OSRS_INVENTORY_CELL_OBS_FEATURES_CODED))) {
        fprintf(stderr, "entity pool branch: stale item-table expansion contract\n");
        abort();
    }
    branch->start = start;
    branch->num_recs = num_recs;
    branch->feats = feats;
    branch->obs_feats = obs_feats;
    branch->type_onehot = type_onehot;
    branch->code_scale = code_scale;
    branch->bottleneck = bottleneck;
    branch->active_width = active_width;
    branch->expansion = expansion;
    branch->item_overlays = item_overlays;
    branch->l1_w = get_weights_aligned(weights, bottleneck * feats);
    branch->l2_w = get_weights_aligned(weights, hidden_dim * bottleneck);
    branch->z1 = (float*)calloc((size_t)num_recs * bottleneck, sizeof(float));
    branch->h1 = (float*)calloc((size_t)num_recs * bottleneck, sizeof(float));
    branch->e = (float*)calloc((size_t)num_recs * hidden_dim, sizeof(float));
    branch->active = (unsigned char*)calloc((size_t)num_recs, sizeof(unsigned char));
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
        1, COLO_ENT_INF_BOTTLENECK, COLO_ENT_INF_TYPE_ONEHOT,
        ENTITY_RECORD_TYPE_ONEHOT, ENTITY_ITEM_OVERLAYS_NONE);
    layer->num_branches = 1;
    if (mode >= 2) {
        entity_pool_branch_init(&layer->branches[1], weights, hidden_dim,
            COLO_ENT_INF_INV_START, COLO_ENT_INF_INV_NUM_CELLS,
            COLO_ENT_INF_INV_FEATS, COLO_ENT_INF_INV_OBS_FEATS, 0,
            OSRS_ITEM_OBS_CODE_SCALE, COLO_ENT_INF_INV_BOTTLENECK, 1,
            ENTITY_RECORD_ITEM_TABLE, ENTITY_ITEM_OVERLAYS_EQUIPPED_HP_HEAL);
        layer->num_branches = 2;
    }
    return layer;
}

EntityEncoder* make_inferno_entity_encoder(
        Weights* weights, int batch_size, int input_dim, int hidden_dim, int mode) {
    if (input_dim != INF_ENT_OBS_SIZE) {
        fprintf(stderr, "inferno entity encoder: input width %d != %d\n",
            input_dim, INF_ENT_OBS_SIZE);
        abort();
    }
    EntityEncoder* layer = make_entity_encoder_global(
        weights, batch_size, input_dim, hidden_dim);
    entity_pool_branch_init(&layer->branches[0], weights, hidden_dim,
        INF_ENT_NPC_START, INF_ENT_NUM_NPCS,
        INF_ENT_FEATS, INF_ENT_OBS_FEATS, INF_ENT_TYPE_ONEHOT,
        INF_ENT_TYPE_CODE_SCALE, COLO_ENT_INF_BOTTLENECK, INF_ENT_TYPE_ONEHOT,
        ENTITY_RECORD_TYPE_ONEHOT, ENTITY_ITEM_OVERLAYS_NONE);
    layer->num_branches = 1;
    if (mode >= 2) {
        entity_pool_branch_init(&layer->branches[1], weights, hidden_dim,
            INF_ENT_INV_START, INF_ENT_INV_NUM_CELLS,
            INF_ENT_INV_FEATS, INF_ENT_INV_OBS_FEATS, 0,
            OSRS_ITEM_OBS_CODE_SCALE, COLO_ENT_INF_INV_BOTTLENECK, 1,
            ENTITY_RECORD_ITEM_TABLE, ENTITY_ITEM_OVERLAYS_NONE);
        layer->num_branches = 2;
    }
    return layer;
}

static void entity_expand_record(const EntityPoolBranch* p, const float* rec, float* out) {
    switch (p->expansion) {
        case ENTITY_RECORD_TYPE_ONEHOT: {
            int code = (int)lrintf(rec[0] * (float)p->code_scale);
            if (code < 0 || code > p->type_onehot) {
                fprintf(stderr, "entity pool branch: type code %d out of range\n", code);
                abort();
            }
            for (int i = 0; i < p->type_onehot; i++)
                out[i] = (code == i + 1) ? 1.0f : 0.0f;
            for (int i = 0; i < p->feats - p->type_onehot; i++)
                out[p->type_onehot + i] = rec[1 + i];
            return;
        }
        case ENTITY_RECORD_ITEM_TABLE: {
            int code = (int)lrintf(rec[0] * (float)p->code_scale);
            if (code < 0 || code >= OSRS_ITEM_OBS_TABLE_ROWS) {
                fprintf(stderr, "entity pool branch: item code %d out of table\n", code);
                abort();
            }
            for (int i = 0; i < p->feats; i++) out[i] = OSRS_ITEM_OBS_TABLE[code][i];
            if (p->item_overlays == ENTITY_ITEM_OVERLAYS_EQUIPPED_HP_HEAL) {
                int is_gear =
                    OSRS_ITEM_OBS_TABLE[code][OSRS_INVENTORY_CELL_COMPACT_IS_ARMOR] != 0.0f ||
                    OSRS_ITEM_OBS_TABLE[code][OSRS_INVENTORY_CELL_COMPACT_IS_WEAPON] != 0.0f;
                out[OSRS_ITEM_OBS_OVERLAY_EQUIPPED] = rec[1];
                if (!is_gear) out[OSRS_ITEM_OBS_OVERLAY_HP_HEAL] = rec[2];
            }
            return;
        }
    }
    fprintf(stderr, "entity pool branch: unknown record expansion\n");
    abort();
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
                float active_sum = 0.0f;
                for (int i = 0; i < p->active_width; i++) active_sum += expanded[i];
                p->active[n] = active_sum > 0.0f;
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
                    if (!p->active[n]) continue;
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
        free(layer->branches[br].active);
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
