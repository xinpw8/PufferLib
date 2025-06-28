#include <time.h>
#include "tetris.h"
#include "puffernet.h"
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

void demo() {
    Tetris env = {
        .n_rows = 20,
        .n_cols = 10,
        .deck_size=3,
    };
    allocate(&env);
    env.client = make_client(&env);
    c_reset(&env);

    Weights* weights = load_weights("resources/tetris/tetris_weights.bin", 163208);
    int logit_sizes[1] = {7};
    LinearLSTM* net = make_linearlstm(weights, 1, 234, logit_sizes, 1);

    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)){
                env.actions[0] = 1;
            }
            if (IsKeyDown(KEY_RIGHT)  || IsKeyDown(KEY_D)){
                env.actions[0] = 2;
            }
            if (IsKeyPressed(KEY_UP)  || IsKeyDown(KEY_W)) {
                env.actions[0] = 3;
            }
            if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) {
                env.actions[0] = 4;
            }
            if (IsKeyPressed(KEY_SPACE)) {
                env.actions[0] = 5;
            }
            if (IsKeyPressed(KEY_C)) {
                env.actions[0] = 6;
            }
        } else {
            forward_linearlstm(net, env.observations, env.actions);
        }

        c_step(&env);
        env.actions[0] = 0;
        c_render(&env);
    }
    free_linearlstm(net);
    free_allocated(&env);
    close_client(env.client);
}

int main() {
    demo();
}
