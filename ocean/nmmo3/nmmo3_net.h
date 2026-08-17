#pragma once
// CPU NMMO3 policy: custom conv/embed encoder + MinGRU + decoder.
// Weight order matches CUDA weights_create: encoder (conv1, conv2, embed,
// proj_w, proj_b), decoder, mingru layers.
// Include puffercpu.c first (Conv2D / Linear / MinGRU).

#define N3_MAP_H 11
#define N3_MAP_W 15
#define N3_NFEAT 10
#define N3_MULTIHOT 59
#define N3_PLAYER 47
#define N3_REWARD 10
#define N3_EMBED_DIM 32
#define N3_EMBED_VOCAB 128
#define N3_C1_OC 128
#define N3_C1_OH 3
#define N3_C1_OW 4
#define N3_C2_OC 128
#define N3_C2_OH 1
#define N3_C2_OW 2
#define N3_CONV_FLAT (N3_C2_OC * N3_C2_OH * N3_C2_OW)
#define N3_CONCAT (N3_CONV_FLAT + N3_PLAYER * N3_EMBED_DIM + N3_PLAYER + N3_REWARD)
#define N3_ATN 26

typedef struct MMONet MMONet;
struct MMONet {
    int num_agents;
    int hidden;
    float* ob_map;
    int* ob_player_discrete;
    float* ob_player_continuous;
    float* ob_reward;
    Conv2D* map_conv1;
    ReLU* map_relu;
    Conv2D* map_conv2;
    Embedding* player_embed;
    float* proj_buffer;
    Linear* proj;
    float* proj_bias;
    ReLU* proj_relu;
    Linear* decoder;
    MinGRU* mingru;
    Multidiscrete* multidiscrete;
};

static inline int nmmo3_weight_count(int hidden, int layers) {
    int n = 0;
    n += N3_C1_OC * N3_MULTIHOT * 5 * 5 + N3_C1_OC;
    n += N3_C2_OC * N3_C1_OC * 3 * 3 + N3_C2_OC;
    n += N3_EMBED_VOCAB * N3_EMBED_DIM;
    n += hidden * N3_CONCAT + hidden;
    n += (N3_ATN + 1) * hidden;
    n += layers * 3 * hidden * hidden;
    return n;
}

static inline void mmonet_add_bias_relu(float* x, const float* bias,
        int batch, int dim) {
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < dim; i++) {
            float v = x[b * dim + i] + bias[i];
            x[b * dim + i] = v > 0.0f ? v : 0.0f;
        }
    }
}

static inline MMONet* init_mmonet_arch(Weights* weights, int num_agents,
        int hidden, int layers) {
    MMONet* net = (MMONet*)calloc(1, sizeof(MMONet));
    net->num_agents = num_agents;
    net->hidden = hidden;
    net->ob_map = (float*)calloc((size_t)num_agents * N3_MULTIHOT * N3_MAP_H * N3_MAP_W,
        sizeof(float));
    net->ob_player_discrete = (int*)calloc((size_t)num_agents * N3_PLAYER, sizeof(int));
    net->ob_player_continuous = (float*)calloc((size_t)num_agents * N3_PLAYER, sizeof(float));
    net->ob_reward = (float*)calloc((size_t)num_agents * N3_REWARD, sizeof(float));
    net->map_conv1 = make_conv2d(weights, num_agents, N3_MAP_W, N3_MAP_H,
        N3_MULTIHOT, N3_C1_OC, 5, 3);
    net->map_relu = make_relu(num_agents, N3_C1_OC * N3_C1_OH * N3_C1_OW);
    net->map_conv2 = make_conv2d(weights, num_agents, N3_C1_OW, N3_C1_OH,
        N3_C1_OC, N3_C2_OC, 3, 1);
    net->player_embed = make_embedding(weights, num_agents * N3_PLAYER,
        N3_EMBED_VOCAB, N3_EMBED_DIM);
    net->proj_buffer = (float*)calloc((size_t)num_agents * N3_CONCAT, sizeof(float));
    net->proj = make_linear(weights, num_agents, N3_CONCAT, hidden);
    net->proj_bias = get_weights_aligned(weights, hidden);
    net->proj_relu = make_relu(num_agents, hidden);
    net->decoder = make_linear(weights, num_agents, hidden, N3_ATN + 1);
    net->mingru = make_mingru(weights, num_agents, hidden, layers);
    int logit_sizes[1] = {N3_ATN};
    net->multidiscrete = make_multidiscrete(num_agents, logit_sizes, 1);
    return net;
}

static inline MMONet* init_mmonet(Weights* weights, int num_agents) {
    return init_mmonet_arch(weights, num_agents, 512, 4);
}

static inline void free_mmonet(MMONet* net) {
    free(net->ob_map);
    free(net->ob_player_discrete);
    free(net->ob_player_continuous);
    free(net->ob_reward);
    free(net->map_conv1);
    free(net->map_relu);
    free(net->map_conv2);
    free(net->player_embed);
    free(net->proj_buffer);
    free(net->proj);
    free(net->proj_relu);
    free(net->decoder);
    free_mingru(net->mingru);
    free(net->multidiscrete);
    free(net);
}

static inline void mmonet_prepare_obs(MMONet* net, unsigned char* observations) {
    memset(net->ob_map, 0,
        (size_t)net->num_agents * N3_MULTIHOT * N3_MAP_H * N3_MAP_W * sizeof(float));
    static const int factors[10] = {4, 4, 17, 5, 3, 5, 5, 5, 7, 4};
    float (*ob_map)[N3_MULTIHOT][N3_MAP_H][N3_MAP_W] =
        (float (*)[N3_MULTIHOT][N3_MAP_H][N3_MAP_W])net->ob_map;
    int stride = N3_MAP_H * N3_MAP_W * N3_NFEAT + N3_PLAYER + N3_REWARD;
    for (int b = 0; b < net->num_agents; b++) {
        int b_offset = b * stride;
        for (int i = 0; i < N3_MAP_H; i++) {
            for (int j = 0; j < N3_MAP_W; j++) {
                int f_offset = 0;
                for (int f = 0; f < N3_NFEAT; f++) {
                    int obs_idx = f_offset + observations[b_offset + i * N3_MAP_W * N3_NFEAT + j * N3_NFEAT + f];
                    ob_map[b][obs_idx][i][j] = 1;
                    f_offset += factors[f];
                }
            }
        }
        for (int i = 0; i < N3_PLAYER; i++) {
            unsigned char ob = observations[b_offset + N3_MAP_H * N3_MAP_W * N3_NFEAT + i];
            net->ob_player_discrete[b * N3_PLAYER + i] = ob;
            net->ob_player_continuous[b * N3_PLAYER + i] = (float)ob;
        }
        for (int i = 0; i < N3_REWARD; i++) {
            net->ob_reward[b * N3_REWARD + i] =
                (float)observations[b_offset + N3_MAP_H * N3_MAP_W * N3_NFEAT + N3_PLAYER + i];
        }
    }
}

static inline void mmonet_encode(MMONet* net, unsigned char* observations,
        float* hidden_out) {
    mmonet_prepare_obs(net, observations);
    conv2d(net->map_conv1, net->ob_map);
    relu(net->map_relu, net->map_conv1->output);
    conv2d(net->map_conv2, net->map_relu->output);
    embedding(net->player_embed, net->ob_player_discrete);

    for (int b = 0; b < net->num_agents; b++) {
        int b_offset = b * N3_CONCAT;
        for (int i = 0; i < N3_CONV_FLAT; i++) {
            net->proj_buffer[b_offset + i] = net->map_conv2->output[b * N3_CONV_FLAT + i];
        }
        b_offset += N3_CONV_FLAT;
        for (int i = 0; i < N3_PLAYER * N3_EMBED_DIM; i++) {
            net->proj_buffer[b_offset + i] =
                net->player_embed->output[b * N3_PLAYER * N3_EMBED_DIM + i];
        }
        b_offset += N3_PLAYER * N3_EMBED_DIM;
        for (int i = 0; i < N3_PLAYER; i++) {
            net->proj_buffer[b_offset + i] = net->ob_player_continuous[b * N3_PLAYER + i];
        }
        b_offset += N3_PLAYER;
        for (int i = 0; i < N3_REWARD; i++) {
            net->proj_buffer[b_offset + i] = net->ob_reward[b * N3_REWARD + i];
        }
    }

    linear(net->proj, net->proj_buffer);
    mmonet_add_bias_relu(net->proj->output, net->proj_bias, net->num_agents, net->hidden);
    memcpy(net->proj_relu->output, net->proj->output,
        (size_t)net->num_agents * net->hidden * sizeof(float));
    if (hidden_out) {
        memcpy(hidden_out, net->proj_relu->output,
            (size_t)net->num_agents * net->hidden * sizeof(float));
    }
}

static inline void forward(MMONet* net, unsigned char* observations,
        float* terminals, float* actions) {
    mingru_zero_term(net->mingru, terminals);
    mmonet_encode(net, observations, NULL);
    mingru(net->mingru, net->proj_relu->output);
    linear(net->decoder, net->mingru->output);
    multidiscrete(net->multidiscrete, net->decoder->output, actions, 0, NULL);
}
