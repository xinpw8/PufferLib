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

#define OSRS_ENT_INV_START        52
#define OSRS_ENT_INV_NUM_RECS     28
#define OSRS_ENT_INV_OBS_FEATS    1
#define OSRS_ENT_ITEM_FEATS       14
#define OSRS_ENT_EQUIPPED_START   80
#define OSRS_ENT_EQUIPPED_NUM_RECS NUM_GEAR_SLOTS
#define OSRS_ENT_EQUIPPED_OBS_FEATS 1
#define OSRS_ENT_ITEM_BOTTLENECK  16

#define COLO_ENT_OBS_SIZE          904
#define COLO_ENT_NPC_START         101
#define COLO_ENT_NPC_NUM_RECS      24
#define COLO_ENT_NPC_OBS_FEATS     23
#define COLO_ENT_NPC_FEATS         34
#define COLO_ENT_NPC_TYPE_ONEHOT   12
#define COLO_ENT_NPC_BOTTLENECK    16
#define COLO_ENT_NPC_TYPE_SCALE    1

#define INF_ENT_OBS_SIZE           530
#define INF_ENT_NPC_START          124
#define INF_ENT_NPC_NUM_RECS       14
#define INF_ENT_NPC_OBS_FEATS      13
#define INF_ENT_NPC_FEATS          26
#define INF_ENT_NPC_TYPE_ONEHOT    14
#define INF_ENT_NPC_BOTTLENECK     16
#define INF_ENT_NPC_TYPE_SCALE     16

static const float OSRS_ITEM_OBS_TABLE
    [OSRS_ITEM_OBS_TABLE_ROWS][OSRS_ITEM_OBS_TABLE_COLS] = {
#include "osrs_item_obs_table.inc"
};

typedef enum {
    ENTITY_RECORD_TYPE_ONEHOT = 0,
    ENTITY_RECORD_ITEM_TABLE,
} EntityRecordExpansion;


#define ENTITY_RECORD_MAX_FEATS 64
#define ENTITY_ENCODER_MAX_BRANCHES 3

typedef struct {
    const char* weight_name;
    int start;
    int num_recs;
    int feats;
    int obs_feats;
    int type_onehot;
    int code_scale;
    int bottleneck;
    int active_width;
    EntityRecordExpansion expansion;
} EntityPoolDescriptor;

typedef struct {
    const char* env_name;
    int obs_size;
    int num_branches;
    EntityPoolDescriptor branches[ENTITY_ENCODER_MAX_BRANCHES];
} EntityEncoderDescriptor;

typedef struct EntityPoolBranch EntityPoolBranch;
struct EntityPoolBranch {
    EntityPoolDescriptor descriptor;
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
    EntityPoolBranch branches[ENTITY_ENCODER_MAX_BRANCHES];
};

static void entity_pool_branch_init(
        EntityPoolBranch* branch, Weights* weights, int hidden_dim,
        const EntityPoolDescriptor* descriptor) {
    if (descriptor->feats > ENTITY_RECORD_MAX_FEATS ||
            descriptor->start < 0 || descriptor->num_recs <= 0 ||
            descriptor->obs_feats <= 0 || descriptor->code_scale <= 0 ||
            descriptor->bottleneck <= 0 || descriptor->active_width <= 0 ||
            descriptor->active_width > descriptor->feats) {
        fprintf(stderr, "entity pool branch: invalid shape or encoding contract\n");
        abort();
    }
    if (descriptor->expansion == ENTITY_RECORD_TYPE_ONEHOT &&
            (descriptor->type_onehot <= 0 ||
             descriptor->feats != descriptor->type_onehot + descriptor->obs_feats - 1)) {
        fprintf(stderr, "entity pool branch: stale type-code expansion contract\n");
        abort();
    }
    if (descriptor->expansion == ENTITY_RECORD_ITEM_TABLE &&
            (descriptor->type_onehot != 0 ||
             descriptor->feats != OSRS_ITEM_OBS_TABLE_COLS ||
             descriptor->obs_feats != 1)) {
        fprintf(stderr, "entity pool branch: stale item-table expansion contract\n");
        abort();
    }
    branch->descriptor = *descriptor;
    branch->l1_w = get_weights_aligned(
        weights, descriptor->bottleneck * descriptor->feats);
    branch->l2_w = get_weights_aligned(
        weights, hidden_dim * descriptor->bottleneck);
    branch->z1 = (float*)calloc(
        (size_t)descriptor->num_recs * descriptor->bottleneck, sizeof(float));
    branch->h1 = (float*)calloc(
        (size_t)descriptor->num_recs * descriptor->bottleneck, sizeof(float));
    branch->e = (float*)calloc(
        (size_t)descriptor->num_recs * hidden_dim, sizeof(float));
    branch->active = (unsigned char*)calloc(
        (size_t)descriptor->num_recs, sizeof(unsigned char));
}

static int entity_pool_descriptor_is_shared_item_branch(
        const EntityPoolDescriptor* descriptor, int start, int num_recs) {
    return descriptor->start == start &&
        descriptor->num_recs == num_recs &&
        descriptor->feats == OSRS_ENT_ITEM_FEATS &&
        descriptor->obs_feats == 1 &&
        descriptor->type_onehot == 0 &&
        descriptor->code_scale == OSRS_ITEM_OBS_CODE_SCALE &&
        descriptor->bottleneck == OSRS_ENT_ITEM_BOTTLENECK &&
        descriptor->active_width == 1 &&
        descriptor->expansion == ENTITY_RECORD_ITEM_TABLE;
}

static EntityEncoder* make_entity_encoder(
        Weights* weights, int batch_size, int input_dim, int hidden_dim,
        const EntityEncoderDescriptor* descriptor) {
    if (!descriptor ||
            (descriptor->obs_size > 0 && descriptor->obs_size != input_dim) ||
            descriptor->num_branches < 2 ||
            descriptor->num_branches > ENTITY_ENCODER_MAX_BRANCHES ||
            !entity_pool_descriptor_is_shared_item_branch(
                &descriptor->branches[0],
                OSRS_ENT_INV_START,
                OSRS_ENT_INV_NUM_RECS) ||
            !entity_pool_descriptor_is_shared_item_branch(
                &descriptor->branches[1],
                OSRS_ENT_EQUIPPED_START,
                OSRS_ENT_EQUIPPED_NUM_RECS)) {
        fprintf(stderr, "entity encoder: stale environment descriptor\n");
        abort();
    }
    for (int i = 0; i < descriptor->num_branches; i++) {
        const EntityPoolDescriptor* branch = &descriptor->branches[i];
        if (branch->start < 0 ||
                branch->start + branch->num_recs * branch->obs_feats >
                    input_dim) {
            fprintf(stderr, "entity encoder: branch exceeds observation width\n");
            abort();
        }
    }
    size_t out_size = (size_t)batch_size * hidden_dim * sizeof(float);
    EntityEncoder* layer =
        (EntityEncoder*)calloc(1, sizeof(EntityEncoder) + out_size);
    layer->output = (float*)(layer + 1);
    layer->global_w = get_weights_aligned(weights, hidden_dim * input_dim);
    layer->batch_size = batch_size;
    layer->input_dim = input_dim;
    layer->hidden_dim = hidden_dim;
    layer->num_branches = descriptor->num_branches;
    for (int i = 0; i < descriptor->num_branches; i++) {
        entity_pool_branch_init(
            &layer->branches[i], weights, hidden_dim, &descriptor->branches[i]);
    }
    return layer;
}

static void entity_expand_record(
        const EntityPoolDescriptor* descriptor, const float* rec, float* out) {
    switch (descriptor->expansion) {
        case ENTITY_RECORD_TYPE_ONEHOT: {
            int code = (int)lrintf(rec[0] * (float)descriptor->code_scale);
            if (code < 0 || code > descriptor->type_onehot) {
                fprintf(stderr, "entity pool branch: type code %d out of range\n", code);
                abort();
            }
            for (int i = 0; i < descriptor->type_onehot; i++)
                out[i] = (code == i + 1) ? 1.0f : 0.0f;
            for (int i = 0; i < descriptor->feats - descriptor->type_onehot; i++)
                out[descriptor->type_onehot + i] = rec[1 + i];
            return;
        }
        case ENTITY_RECORD_ITEM_TABLE: {
            int code = (int)lrintf(rec[0] * (float)descriptor->code_scale);
            if (code < 0 || code >= OSRS_ITEM_OBS_TABLE_ROWS) {
                fprintf(stderr, "entity pool branch: item code %d out of table\n", code);
                abort();
            }
            for (int i = 0; i < descriptor->feats; i++)
                out[i] = OSRS_ITEM_OBS_TABLE[code][i];
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
            const EntityPoolDescriptor* descriptor = &p->descriptor;
            float* recs = obs + descriptor->start;
            float expanded[ENTITY_RECORD_MAX_FEATS];
            for (int n = 0; n < descriptor->num_recs; n++) {
                float* rec = recs + n * descriptor->obs_feats;
                float* z1n = p->z1 + n * descriptor->bottleneck;
                entity_expand_record(descriptor, rec, expanded);
                float active_sum = 0.0f;
                for (int i = 0; i < descriptor->active_width; i++)
                    active_sum += expanded[i];
                p->active[n] = active_sum > 0.0f;
                for (int k = 0; k < descriptor->bottleneck; k++) {
                    const float* w = p->l1_w + k * descriptor->feats;
                    float sum = 0.0f;
                    for (int i = 0; i < descriptor->feats; i++)
                        sum += expanded[i] * w[i];
                    z1n[k] = sum;
                }
            }
            osrs_visual_gelu(
                p->z1, p->h1, descriptor->num_recs * descriptor->bottleneck);
            for (int n = 0; n < descriptor->num_recs; n++) {
                float* h1n = p->h1 + n * descriptor->bottleneck;
                float* en = p->e + (size_t)n * H;
                for (int o = 0; o < H; o++) {
                    float sum = 0.0f;
                    for (int k = 0; k < descriptor->bottleneck; k++)
                        sum += h1n[k] *
                            p->l2_w[o * descriptor->bottleneck + k];
                    en[o] = sum;
                }
            }
            for (int o = 0; o < H; o++) {
                float best = -INFINITY;
                int best_n = -1;
                for (int n = 0; n < descriptor->num_recs; n++) {
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
