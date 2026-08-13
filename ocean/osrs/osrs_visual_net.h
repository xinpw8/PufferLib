#pragma once

#define COLO_ENT_INF_NPC_START   1030
#define COLO_ENT_INF_NUM_NPCS    24
#define COLO_ENT_INF_FEATS       37
#define COLO_ENT_INF_TYPE_ONEHOT 12
#define COLO_ENT_INF_BOTTLENECK  16
#define COLO_ENT_INF_INV_START      48
#define COLO_ENT_INF_INV_NUM_CELLS  28
#define COLO_ENT_INF_INV_FEATS      28
#define COLO_ENT_INF_INV_PRESENT    0
#define COLO_ENT_INF_INV_BOTTLENECK 16

typedef struct ColosseumEntityEncoder ColosseumEntityEncoder;
struct ColosseumEntityEncoder {
    float* output;
    float* global_w;
    float* entity_l1_w;
    float* entity_l2_w;
    float* z1;
    float* h1;
    float* entity_e;
    int batch_size;
    int input_dim;
    int hidden_dim;
    int mode;
    float* inv_l1_w;
    float* inv_l2_w;
    float* inv_z1;
    float* inv_h1;
    float* inv_e;
};

ColosseumEntityEncoder* make_colosseum_entity_encoder(
        Weights* weights, int batch_size, int input_dim, int hidden_dim, int mode) {
    size_t out_size = (size_t)batch_size * hidden_dim * sizeof(float);
    ColosseumEntityEncoder* layer =
        (ColosseumEntityEncoder*)calloc(1, sizeof(ColosseumEntityEncoder) + out_size);
    *layer = (ColosseumEntityEncoder){
        .output = (float*)(layer + 1),
        .global_w = get_weights_aligned(weights, hidden_dim * input_dim),
        .entity_l1_w = get_weights_aligned(weights, COLO_ENT_INF_BOTTLENECK * COLO_ENT_INF_FEATS),
        .entity_l2_w = get_weights_aligned(weights, hidden_dim * COLO_ENT_INF_BOTTLENECK),
        .z1 = (float*)calloc((size_t)COLO_ENT_INF_NUM_NPCS * COLO_ENT_INF_BOTTLENECK, sizeof(float)),
        .h1 = (float*)calloc((size_t)COLO_ENT_INF_NUM_NPCS * COLO_ENT_INF_BOTTLENECK, sizeof(float)),
        .entity_e = (float*)calloc((size_t)COLO_ENT_INF_NUM_NPCS * hidden_dim, sizeof(float)),
        .batch_size = batch_size,
        .input_dim = input_dim,
        .hidden_dim = hidden_dim,
        .mode = mode,
    };
    // mode 2: the inventory-pool weights follow entity_l2 in the .bin (reg_params order).
    // Read them as sequenced statements AFTER the initializer above (C does not specify the
    // initializer's internal evaluation order, so the global/entity reads must complete first).
    if (mode >= 2) {
        layer->inv_l1_w = get_weights_aligned(weights, COLO_ENT_INF_INV_BOTTLENECK * COLO_ENT_INF_INV_FEATS);
        layer->inv_l2_w = get_weights_aligned(weights, hidden_dim * COLO_ENT_INF_INV_BOTTLENECK);
        layer->inv_z1 = (float*)calloc((size_t)COLO_ENT_INF_INV_NUM_CELLS * COLO_ENT_INF_INV_BOTTLENECK, sizeof(float));
        layer->inv_h1 = (float*)calloc((size_t)COLO_ENT_INF_INV_NUM_CELLS * COLO_ENT_INF_INV_BOTTLENECK, sizeof(float));
        layer->inv_e = (float*)calloc((size_t)COLO_ENT_INF_INV_NUM_CELLS * hidden_dim, sizeof(float));
    }
    return layer;
}

void colosseum_entity_encoder(ColosseumEntityEncoder* layer, float* observations) {
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

        float* npcs = obs + COLO_ENT_INF_NPC_START;
        for (int n = 0; n < COLO_ENT_INF_NUM_NPCS; n++) {
            float* rec = npcs + n * COLO_ENT_INF_FEATS;
            float* z1n = layer->z1 + n * COLO_ENT_INF_BOTTLENECK;
            for (int k = 0; k < COLO_ENT_INF_BOTTLENECK; k++) {
                float sum = 0.0f;
                for (int i = 0; i < COLO_ENT_INF_FEATS; i++)
                    sum += rec[i] * layer->entity_l1_w[k * COLO_ENT_INF_FEATS + i];
                z1n[k] = sum;
            }
        }
        _gelu(layer->z1, layer->h1, COLO_ENT_INF_NUM_NPCS * COLO_ENT_INF_BOTTLENECK);
        for (int n = 0; n < COLO_ENT_INF_NUM_NPCS; n++) {
            float* h1n = layer->h1 + n * COLO_ENT_INF_BOTTLENECK;
            float* en = layer->entity_e + (size_t)n * H;
            for (int o = 0; o < H; o++) {
                float sum = 0.0f;
                for (int k = 0; k < COLO_ENT_INF_BOTTLENECK; k++)
                    sum += h1n[k] * layer->entity_l2_w[o * COLO_ENT_INF_BOTTLENECK + k];
                en[o] = sum;
            }
        }

        for (int o = 0; o < H; o++) {
            float best = -INFINITY;
            int best_n = -1;
            for (int n = 0; n < COLO_ENT_INF_NUM_NPCS; n++) {
                float* rec = npcs + n * COLO_ENT_INF_FEATS;
                float type_sum = 0.0f;
                for (int t = 0; t < COLO_ENT_INF_TYPE_ONEHOT; t++) type_sum += rec[t];
                if (type_sum <= 0.0f) continue;
                float v = layer->entity_e[(size_t)n * H + o];
                if (v > best) { best = v; best_n = n; }
            }
            out[o] += (best_n < 0) ? 0.0f : best;
        }

        if (layer->mode >= 2) {
            float* cells = obs + COLO_ENT_INF_INV_START;
            for (int n = 0; n < COLO_ENT_INF_INV_NUM_CELLS; n++) {
                float* rec = cells + n * COLO_ENT_INF_INV_FEATS;
                float* z1n = layer->inv_z1 + n * COLO_ENT_INF_INV_BOTTLENECK;
                for (int k = 0; k < COLO_ENT_INF_INV_BOTTLENECK; k++) {
                    float sum = 0.0f;
                    for (int i = 0; i < COLO_ENT_INF_INV_FEATS; i++)
                        sum += rec[i] * layer->inv_l1_w[k * COLO_ENT_INF_INV_FEATS + i];
                    z1n[k] = sum;
                }
            }
            _gelu(layer->inv_z1, layer->inv_h1, COLO_ENT_INF_INV_NUM_CELLS * COLO_ENT_INF_INV_BOTTLENECK);
            for (int n = 0; n < COLO_ENT_INF_INV_NUM_CELLS; n++) {
                float* h1n = layer->inv_h1 + n * COLO_ENT_INF_INV_BOTTLENECK;
                float* en = layer->inv_e + (size_t)n * H;
                for (int o = 0; o < H; o++) {
                    float sum = 0.0f;
                    for (int k = 0; k < COLO_ENT_INF_INV_BOTTLENECK; k++)
                        sum += h1n[k] * layer->inv_l2_w[o * COLO_ENT_INF_INV_BOTTLENECK + k];
                    en[o] = sum;
                }
            }
            for (int o = 0; o < H; o++) {
                float best = -INFINITY;
                int best_n = -1;
                for (int n = 0; n < COLO_ENT_INF_INV_NUM_CELLS; n++) {
                    float* rec = cells + n * COLO_ENT_INF_INV_FEATS;
                    if (rec[COLO_ENT_INF_INV_PRESENT] <= 0.0f) continue;
                    float v = layer->inv_e[(size_t)n * H + o];
                    if (v > best) { best = v; best_n = n; }
                }
                out[o] += (best_n < 0) ? 0.0f : best;
            }
        }
    }
}

void free_colosseum_entity_encoder(ColosseumEntityEncoder* layer) {
    free(layer->z1);
    free(layer->h1);
    free(layer->entity_e);
    free(layer->inv_z1);
    free(layer->inv_h1);
    free(layer->inv_e);
    free(layer);
}

typedef struct VisualNet VisualNet;
struct VisualNet {
    int num_agents;
    float* obs;
    Linear* encoder;
    ColosseumEntityEncoder* entity_encoder;
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
    if (net->entity_encoder) free_colosseum_entity_encoder(net->entity_encoder);
    free(net->decoder);
    free_mingru(net->mingru);
    if (net->multidiscrete) free(net->multidiscrete);
    free(net);
}
