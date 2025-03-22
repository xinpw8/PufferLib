#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "cartpole.h"
#include "puffernet.h"

#define NUM_WEIGHTS 132866
#define OBSERVATIONS_SIZE 4
#define ACTIONS_SIZE 1
const char* WEIGHTS_PATH = "/puffertank/pufferlib/pufferlib/resources/cartpole/cartpole_weights.bin";

void demo() {
    Weights* weights = load_weights(WEIGHTS_PATH, NUM_WEIGHTS);
float decoder_logstd[ACTIONS_SIZE] = {-0.3126390f};

    LinearLSTM* net = make_linearlstm_float(weights, 1, OBSERVATIONS_SIZE, ACTIONS_SIZE, ACTION_TYPE_FLOAT);

    CartPole env = {0};
    // env.continuous = 1; // Ensure continuous actions
    allocate(&env);
    Client* client = make_client(&env);
    c_reset(&env);

    SetTargetFPS(60);

    int episode_steps = 0;
    float episode_return = 0.0f;

    while (!WindowShouldClose()) {
        forward_linearlstm_float(net, env.observations, env.actions);

        float mu = net->actor->output[0];
        float sigma = expf(decoder_logstd[0]);

        // Deterministic evaluation action
        float action = tanhf(mu);
        env.actions[0] = action;
        // Take step in environment
        c_step(&env);

        episode_return += env.rewards[0];
        episode_steps++;

        BeginDrawing();
        ClearBackground(RAYWHITE);
        c_render(client, &env);
        DrawText("Evaluating policy...", 10, 160, 20, DARKGRAY);
        EndDrawing();

        if (env.dones[0]) {
            printf("Episode done. Steps: %d, Return: %.2f\n\n", episode_steps, episode_return);
            episode_steps = 0;
            episode_return = 0.0f;
            c_reset(&env);
        }
    }

    free_linearlstm(net);
    free(weights);
    close_client(client);
    free_allocated(&env);
}

int main() {
    srand(time(NULL));
    demo();
    return 0;
}
