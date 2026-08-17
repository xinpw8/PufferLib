// CPU asteroids policy: entity encoder + MinGRU + decoder.
// Weight order matches CUDA weights_create / reg_params:
//   encoder input_w (16 x 9), encoder output_w (hidden x 16),
//   decoder ((4+1) x hidden), mingru layers (3*hidden x hidden each).
// Forward matches ocean/asteroids/asteroids.cu:
//   materialize 20 entities [self(4) | point_i(5)], Linear 9->16, ReLU,
//   Linear 16->hidden, max-pool over the 20 points, then MinGRU + decoder.
// Include puffercpu.c first (Linear / ReLU / MinGRU).

#define AE_SELF_DIM 4
#define AE_POINT_DIM 5
#define AE_NUM_POINTS 20
#define AE_ENTITY_IN (AE_SELF_DIM + AE_POINT_DIM)
#define AE_ENTITY_HIDDEN 16
#define AE_OBS_SIZE (AE_SELF_DIM + AE_NUM_POINTS * AE_POINT_DIM)
#define AE_ATN 4

#if defined(OBS_SIZE) && (OBS_SIZE != AE_OBS_SIZE)
#error "asteroids entity encoder expects OBS_SIZE 104"
#endif

typedef struct AsteroidsNet AsteroidsNet;
struct AsteroidsNet {
    int num_agents;
    int hidden;
    float* point_input;
    float* encoded;
    Linear* entity_fc;
    ReLU* entity_relu;
    Linear* out_fc;
    Linear* decoder;
    MinGRU* mingru;
    Multidiscrete* multidiscrete;
};

static inline int puf_ae_align8(int n) {
    return (n + 7) & ~7;
}

static inline int asteroids_weight_count(int hidden, int layers) {
    int n = 0;
    n = puf_ae_align8(n + AE_ENTITY_HIDDEN * AE_ENTITY_IN);
    n = puf_ae_align8(n + hidden * AE_ENTITY_HIDDEN);
    n = puf_ae_align8(n + (AE_ATN + 1) * hidden);
    for (int l = 0; l < layers; l++) {
        n = puf_ae_align8(n + 3 * hidden * hidden);
    }
    return n;
}

static inline AsteroidsNet* init_asteroids_net(Weights* weights, int num_agents,
        int hidden, int layers) {
    AsteroidsNet* net = (AsteroidsNet*)calloc(1, sizeof(AsteroidsNet));
    net->num_agents = num_agents;
    net->hidden = hidden;
    net->point_input = (float*)calloc((size_t)num_agents * AE_NUM_POINTS * AE_ENTITY_IN,
        sizeof(float));
    net->encoded = (float*)calloc((size_t)num_agents * hidden, sizeof(float));
    net->entity_fc = make_linear(weights, num_agents * AE_NUM_POINTS,
        AE_ENTITY_IN, AE_ENTITY_HIDDEN);
    net->entity_relu = make_relu(num_agents * AE_NUM_POINTS, AE_ENTITY_HIDDEN);
    net->out_fc = make_linear(weights, num_agents * AE_NUM_POINTS,
        AE_ENTITY_HIDDEN, hidden);
    net->decoder = make_linear(weights, num_agents, hidden, AE_ATN + 1);
    net->mingru = make_mingru(weights, num_agents, hidden, layers);
    int logit_sizes[1] = {AE_ATN};
    net->multidiscrete = make_multidiscrete(num_agents, logit_sizes, 1);
    return net;
}

static inline void free_asteroids_net(AsteroidsNet* net) {
    free(net->point_input);
    free(net->encoded);
    free(net->entity_fc);
    free(net->entity_relu);
    free(net->out_fc);
    free(net->decoder);
    free_mingru(net->mingru);
    free(net->multidiscrete);
    free(net);
}

static inline void asteroids_encode(AsteroidsNet* net, const float* observations,
        float* hidden_out) {
    int B = net->num_agents;
    int H = net->hidden;
    for (int b = 0; b < B; b++) {
        const float* obs = observations + b * AE_OBS_SIZE;
        for (int p = 0; p < AE_NUM_POINTS; p++) {
            float* ent = net->point_input
                + ((size_t)b * AE_NUM_POINTS + p) * AE_ENTITY_IN;
            memcpy(ent, obs, AE_SELF_DIM * sizeof(float));
            memcpy(ent + AE_SELF_DIM,
                obs + AE_SELF_DIM + p * AE_POINT_DIM,
                AE_POINT_DIM * sizeof(float));
        }
    }
    linear(net->entity_fc, net->point_input);
    relu(net->entity_relu, net->entity_fc->output);
    linear(net->out_fc, net->entity_relu->output);
    _max_dim1(net->out_fc->output, net->encoded, B, AE_NUM_POINTS, H);
    if (hidden_out) {
        memcpy(hidden_out, net->encoded, (size_t)B * H * sizeof(float));
    }
}

static inline void forward_asteroids(AsteroidsNet* net, const float* observations,
        float* terminals, float* actions) {
    mingru_zero_term(net->mingru, terminals);
    asteroids_encode(net, observations, NULL);
    mingru(net->mingru, net->encoded);
    linear(net->decoder, net->mingru->output);
    multidiscrete(net->multidiscrete, net->decoder->output, actions, 0, NULL);
}
