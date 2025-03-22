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

// Explicitly declared logstd from the trained model
float decoder_logstd[ACTIONS_SIZE] = {-0.3126390f};

// Function to sample from normal (unused at eval)
float sample_normal(float mu, float sigma) {
    float u1 = rand() / (RAND_MAX + 1.0f);
    float u2 = rand() / (RAND_MAX + 1.0f);
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    return mu + sigma * z0;
}

void demo() {
    Weights* weights = load_weights(WEIGHTS_PATH, NUM_WEIGHTS);
float decoder_logstd[ACTIONS_SIZE] = {-0.3126390f};
    printf("Decoder logstd: %f\n", decoder_logstd); // sanity check
    printf("Encoder first weight: %.5f\n", weights->data[0]);
    printf("Encoder first bias: %.5f\n", weights->data[512]); // index after 128*4 encoder weights
    // etc., check alignment at LSTM weights as well




    LinearLSTM* net = make_linearlstm_float(weights, 1, OBSERVATIONS_SIZE, ACTIONS_SIZE, ACTION_TYPE_FLOAT);

    CartPole env = {0};
    env.continuous = 1; // Ensure continuous actions
    allocate(&env);
    Client* client = make_client(&env);
    c_reset(&env);

    SetTargetFPS(60);

    int episode_steps = 0;
    float episode_return = 0.0f;

    printf("Decoder logstd: %f\n", decoder_logstd[0]);

    while (!WindowShouldClose()) {
        // Network forward pass
        forward_linearlstm_float(net, env.observations, env.actions);

        float mu = net->actor->output[0];
        float sigma = expf(decoder_logstd[0]); // logstd explicitly used

        // Deterministic evaluation action
        float action = tanhf(mu);
        env.actions[0] = action;

        // Debug information
        printf("Obs: [%.4f, %.4f, %.4f, %.4f] | Actor mean: %.4f, sigma: %.4f, action (tanh): %.4f\n",
               env.observations[0], env.observations[1],
               env.observations[2], env.observations[3],
               mu, sigma, action);

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
