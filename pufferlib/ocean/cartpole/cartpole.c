#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cartpole.h"
#include "puffernet.h"

#define NUM_WEIGHTS 132995
#define OBSERVATIONS_SIZE 4
#define ACTIONS_SIZE 0
const char* WEIGHTS_PATH = "/puffertank/pufferlib/pufferlib/resources/cartpole/cartpole_weights.bin";

void get_input(CartPole* env) {
    if (env->continuous) {
        float move = GetMouseWheelMove();
        float clamped_wheel = fmaxf(-1.0f, fminf(1.0f, move));
        env->actions[0] = clamped_wheel;
    } else if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
        env->actions[0] = 0; // left
    } else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
        env->actions[0] = 1; // right
    }
}

void demo() {
    Weights* weights = load_weights(WEIGHTS_PATH, NUM_WEIGHTS);
    LinearLSTM* net = make_linearlstm(weights, 1, OBSERVATIONS_SIZE, ACTIONS_SIZE);
    
    CartPole env = {0};

    allocate(&env);
    Client* client = make_client(&env);
    c_reset(&env);
    
    SetTargetFPS(60);
    
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            get_input(&env);  // Human input
        } else {
            forward_linearlstm(net, env.observations, env.actions);  // AI input
        }
        
        c_step(&env);
        
        BeginDrawing();
        ClearBackground(RAYWHITE);
        c_render(client, &env);
        DrawText("Hold LEFT SHIFT for manual control", 10, 160, 20, DARKGRAY);
        EndDrawing();
        
        if (env.dones[0]) {
            c_reset(&env);
        }
    }
    
    free_linearlstm(net);
    free(weights);
    close_client(client);
    free_allocated(&env);
}

int main() {
    srand(time(NULL));  // random seed
    demo();
    return 0;
}