#include <time.h>
#include "drmario.h"
#include "puffercpu.h"

void demo() {
    Weights* weights = load_weights("resources/drmario/drmario_weights.bin");
    int logit_sizes[] = {7};
    PufferNet* net = make_puffernet(weights, 1, OBS_SIZE, 256, 2, logit_sizes, 1);

    DrMario env = {0};
    env.n_rows = 16;
    env.n_cols = 8;
    env.n_init_viruses = 14;
    env.rng = (unsigned int)time(NULL);
    init(&env);
    env.agents[0].observations = (float*)calloc(OBS_SIZE, sizeof(float));
    env.agents[0].actions = (float*)calloc(1, sizeof(float));
    env.agents[0].rewards = (float*)calloc(1, sizeof(float));
    env.agents[0].terminals = (float*)calloc(1, sizeof(float));
    puf_reset(&env);

    while (!WindowShouldClose()) {
        if (!IsKeyDown(KEY_LEFT_SHIFT)) {
            forward_puffernet(net, env.agents[0].observations,
                env.agents[0].actions);
        }
        puf_step(&env);
        puf_render(&env);
    }
    free_puffernet(net);
    free(weights);
    free(env.agents[0].observations);
    free(env.agents[0].actions);
    free(env.agents[0].rewards);
    free(env.agents[0].terminals);
    puf_close(&env);
}

int main() {
    demo();
    return 0;
}
