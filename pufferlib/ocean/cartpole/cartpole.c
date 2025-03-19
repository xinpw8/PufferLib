#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "cartpole.h"
#include "puffernet.h"

#define NUM_WEIGHTS 133123
#define OBSERVATIONS_SIZE 0
#define ACTIONS_SIZE 0
const char* WEIGHTS_PATH = "/puffertank/pufferlib/pufferlib/resources/cartpole_gpt/cartpole_gpt_weights.bin";

void get_input(CartPole* env) {
    if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
        env->actions[0] = 0; // Left
    } else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
        env->actions[0] = 1; // Right
    }
}

void demo() {
    Weights* weights = load_weights(WEIGHTS_PATH, NUM_WEIGHTS);
    LinearLSTM* net = make_linearlstm(weights, 1, OBSERVATIONS_SIZE, ACTIONS_SIZE);
    
    CartPole env = {0};
    env.num_obs = OBSERVATIONS_SIZE;
    
    // Proper initialization sequence
    allocate(&env);
    Client* client = make_client(&env);
    c_reset(&env);
    
    // REMOVE THIS LINE - Window already initialized in make_client
    // InitWindow(env.width, env.height, "CartPole AI Control");
    
    SetTargetFPS(60);
    
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            get_input(&env);  // Human input
        } else {
            forward_linearlstm(net, env.observations, env.actions);  // AI input
        }
        
        c_step(&env);
        printf("ran this");
        
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
    srand(time(NULL));  // Initialize random seed
    demo();
    return 0;
}