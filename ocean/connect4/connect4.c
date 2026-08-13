#include "connect4.h"
#include "puffernet.h"
#include "time.h"

const unsigned char NOOP = 8;

void demo() {
    Weights* weights = load_weights("resources/connect4/connect4_weights.bin");
    int logit_sizes[] = {7};
    PufferNet* net = make_puffernet(weights, 1, 42, 256, 1, logit_sizes, 1);

    Connect4 env = {
    };
    allocate_cconnect4(&env);
    puf_reset(&env);
 
    env.client = make_client();

    int tick = 0;
    while (!WindowShouldClose()) {
        env.agents[0].actions[0] = NOOP;
        // user inputs 1 - 7 key pressed
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if(IsKeyPressed(KEY_ONE)) env.agents[0].actions[0] = 0;
            if(IsKeyPressed(KEY_TWO)) env.agents[0].actions[0] = 1;
            if(IsKeyPressed(KEY_THREE)) env.agents[0].actions[0] = 2;
            if(IsKeyPressed(KEY_FOUR)) env.agents[0].actions[0] = 3;
            if(IsKeyPressed(KEY_FIVE)) env.agents[0].actions[0] = 4;
            if(IsKeyPressed(KEY_SIX)) env.agents[0].actions[0] = 5;
            if(IsKeyPressed(KEY_SEVEN)) env.agents[0].actions[0] = 6;
        } else if (tick % 30 == 0) {
            forward_puffernet(net, env.agents[0].observations, env.agents[0].actions);
        }

        tick = (tick + 1) % 60;
        if (env.agents[0].actions[0] >= 0 && env.agents[0].actions[0] <= 6) {
            puf_step(&env);
        }

        puf_render(&env);
    }
    free_puffernet(net);
    free(weights);
    close_client(env.client);
    free_allocated_cconnect4(&env);
}

int main() {
    demo();
    return 0;
}
