#include "g2048.h"
#include "puffernet.h"

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

    env.observations = observations;
    env.actions = actions;
    env.rewards = rewards;
    env.terminals = terminals;

    c_reset(&env);
    c_render(&env);

    while (!WindowShouldClose()) {
        // User can take control
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            bool pressed = false;
            if (IsKeyPressed(KEY_UP) || IsKeyPressed(KEY_W)) { env.actions[0] = 0; pressed = true; }
            else if (IsKeyPressed(KEY_DOWN) || IsKeyPressed(KEY_S)) { env.actions[0] = 1; pressed = true; }
            else if (IsKeyPressed(KEY_LEFT) || IsKeyPressed(KEY_A)) { env.actions[0] = 2; pressed = true; }
            else if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_D)) { env.actions[0] = 3; pressed = true; }

            if (pressed) {
                c_step(&env);
            }
        } else {
            forward_puffernet(net, env.observations, env.actions);
            c_step(&env);
        }

        c_render(&env);
    }

    free_puffernet(net);
    free(weights);
    c_close(&env);
}

int main() {
    demo();
    return 0;
}
