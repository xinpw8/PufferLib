#pragma once

#include "osrs_item_obs_generated.h"

static inline float osrs_visual_gelu(float x) {
    return 0.5f * x *
        (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
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

typedef struct {
    const EntityPoolDescriptor* descriptor;
    float* l1_w;
    float* l2_w;
    float* hidden;
} EntityPoolBranch;

typedef struct {
    float* output;
    float* global_w;
    int batch_size;
    int input_dim;
    int hidden_dim;
    int num_branches;
    EntityPoolBranch branches[ENTITY_ENCODER_MAX_BRANCHES];
} EntityEncoder;

static void entity_pool_branch_init(
        EntityPoolBranch* branch, Weights* weights, int hidden_dim,
        const EntityPoolDescriptor* descriptor, float** scratch) {
    branch->descriptor = descriptor;
    branch->l1_w = get_weights_aligned(
        weights, descriptor->bottleneck * descriptor->feats);
    branch->l2_w = get_weights_aligned(
        weights, hidden_dim * descriptor->bottleneck);
    branch->hidden = *scratch;
    *scratch += descriptor->num_recs * descriptor->bottleneck;
}

static EntityEncoder* make_entity_encoder(
        Weights* weights, int batch_size, int input_dim, int hidden_dim,
        const EntityEncoderDescriptor* descriptor) {
    size_t scratch_floats = (size_t)batch_size * hidden_dim;
    for (int i = 0; i < descriptor->num_branches; i++) {
        scratch_floats += (size_t)descriptor->branches[i].num_recs *
            descriptor->branches[i].bottleneck;
    }
    EntityEncoder* layer = (EntityEncoder*)calloc(
        1, sizeof(EntityEncoder) + scratch_floats * sizeof(float));
    float* scratch = (float*)(layer + 1);
    layer->output = scratch;
    scratch += (size_t)batch_size * hidden_dim;
    layer->global_w = get_weights_aligned(weights, hidden_dim * input_dim);
    layer->batch_size = batch_size;
    layer->input_dim = input_dim;
    layer->hidden_dim = hidden_dim;
    layer->num_branches = descriptor->num_branches;
    for (int i = 0; i < descriptor->num_branches; i++) {
        entity_pool_branch_init(
            &layer->branches[i], weights, hidden_dim,
            &descriptor->branches[i], &scratch);
    }
    return layer;
}

static void entity_expand_record(
        const EntityPoolDescriptor* descriptor, const float* rec, float* out) {
    int code = (int)lrintf(rec[0] * (float)descriptor->code_scale);
    if (descriptor->expansion == ENTITY_RECORD_ITEM_TABLE) {
        assert(code >= 0 && code < OSRS_ITEM_OBS_TABLE_ROWS);
        for (int i = 0; i < descriptor->feats; i++)
            out[i] = OSRS_ITEM_OBS_TABLE[code][i];
        return;
    }
    assert(code >= 0 && code <= descriptor->type_onehot);
    for (int i = 0; i < descriptor->type_onehot; i++)
        out[i] = code == i + 1 ? 1.0f : 0.0f;
    for (int i = 0; i < descriptor->feats - descriptor->type_onehot; i++)
        out[descriptor->type_onehot + i] = rec[1 + i];
}

void entity_encoder_forward(EntityEncoder* layer, float* observations) {
    int H = layer->hidden_dim;
    int IN = layer->input_dim;
    for (int b = 0; b < layer->batch_size; b++) {
        float* obs = observations + (size_t)b * IN;
        float* out = layer->output + (size_t)b * H;
        for (int o = 0; o < H; o++) {
            float sum = 0.0f;
            for (int i = 0; i < IN; i++)
                sum += obs[i] * layer->global_w[o * IN + i];
            out[o] = sum;
        }
        for (int br = 0; br < layer->num_branches; br++) {
            EntityPoolBranch* branch = &layer->branches[br];
            const EntityPoolDescriptor* descriptor = branch->descriptor;
            float* recs = obs + descriptor->start;
            unsigned long long active_records = 0;
            float expanded[COLO_ENT_NPC_FEATS];
            for (int n = 0; n < descriptor->num_recs; n++) {
                float* rec = recs + n * descriptor->obs_feats;
                float* hidden = branch->hidden + n * descriptor->bottleneck;
                entity_expand_record(descriptor, rec, expanded);
                float active_sum = 0.0f;
                for (int i = 0; i < descriptor->active_width; i++)
                    active_sum += expanded[i];
                if (active_sum > 0.0f)
                    active_records |= 1ULL << n;
                for (int k = 0; k < descriptor->bottleneck; k++) {
                    const float* weight = branch->l1_w + k * descriptor->feats;
                    float sum = 0.0f;
                    for (int i = 0; i < descriptor->feats; i++)
                        sum += expanded[i] * weight[i];
                    hidden[k] = osrs_visual_gelu(sum);
                }
            }
            for (int o = 0; o < H; o++) {
                float best = -INFINITY;
                int has_active_record = 0;
                for (int n = 0; n < descriptor->num_recs; n++) {
                    if ((active_records & (1ULL << n)) == 0) continue;
                    const float* hidden =
                        branch->hidden + n * descriptor->bottleneck;
                    float sum = 0.0f;
                    for (int k = 0; k < descriptor->bottleneck; k++)
                        sum += hidden[k] *
                            branch->l2_w[o * descriptor->bottleneck + k];
                    if (sum > best) best = sum;
                    has_active_record = 1;
                }
                if (has_active_record) out[o] += best;
            }
        }
    }
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
    free(net->entity_encoder);
    free(net->decoder);
    free_mingru(net->mingru);
    if (net->multidiscrete) free(net->multidiscrete);
    free(net);
}
