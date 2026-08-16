#include "connect4.h"
#include "puffercpu.h"

void demo() {
    Weights* weights = load_weights("resources/connect4/connect4_weights.bin");
    int logit_sizes[] = {7};
    PufferNet* net = make_puffernet(weights, 1, 42, 256, 1, logit_sizes, 1);

    Connect4 env = {
    };
    env.num_agents = 1;
    env.agents[0].observations = (float*)calloc(42, sizeof(float));
    env.agents[0].actions = (float*)calloc(1, sizeof(float));
    env.agents[0].rewards = (float*)calloc(1, sizeof(float));
    env.agents[0].terminals = (float*)calloc(1, sizeof(float));
    puf_reset(&env);

    env.client = make_client();

    int tick = 0;
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            puf_step(&env);
        } else if (tick % 30 == 0) {
            forward_puffernet(net, env.agents[0].observations, env.agents[0].actions);
            puf_step(&env);
        }
        tick = (tick + 1) % 60;
        puf_render(&env);
    }
    free_puffernet(net);
    free(weights);
    puf_close(&env);
    free(env.agents[0].observations);
    free(env.agents[0].actions);
    free(env.agents[0].rewards);
    free(env.agents[0].terminals);
}

int main() {
    demo();
    return 0;
}
