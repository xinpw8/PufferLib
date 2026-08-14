#include "g2048.h"
#include "puffercpu.h"

void demo() {
    Weights* weights = load_weights("resources/g2048/g2048_weights.bin");
    int logit_sizes[1] = {4};
    PufferNet* net = make_puffernet(weights, 1, 16, 512, 4, logit_sizes, 1);

    Game env = {
        .scaffolding_ratio = 0.0,
    };
    init(&env);

    float observations[16] = {0};
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};

    env.agents[0].observations = observations;
    env.agents[0].actions = actions;
    env.agents[0].rewards = rewards;
    env.agents[0].terminals = terminals;

    puf_reset(&env);
    puf_render(&env);

    while (!WindowShouldClose()) {
        // User can take control
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            bool pressed = false;
            if (IsKeyPressed(KEY_UP) || IsKeyPressed(KEY_W)) { env.agents[0].actions[0] = 0; pressed = true; }
            else if (IsKeyPressed(KEY_DOWN) || IsKeyPressed(KEY_S)) { env.agents[0].actions[0] = 1; pressed = true; }
            else if (IsKeyPressed(KEY_LEFT) || IsKeyPressed(KEY_A)) { env.agents[0].actions[0] = 2; pressed = true; }
            else if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_D)) { env.agents[0].actions[0] = 3; pressed = true; }

            if (pressed) {
                puf_step(&env);
            }
        } else {
            forward_puffernet(net, env.agents[0].observations, env.agents[0].actions);
            puf_step(&env);
        }

        puf_render(&env);
    }

    free_puffernet(net);
    free(weights);
    puf_close(&env);
}

int main() {
    demo();
    return 0;
}
