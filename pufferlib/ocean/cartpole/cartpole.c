#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "cartpole.h"
#include "puffernet.h"

#define NUM_WEIGHTS 133123
#define OBSERVATIONS_SIZE 4
#define ACTIONS_SIZE 2
#define CONTINUOUS 0
const char* WEIGHTS_PATH = "/puffertank/pufferlib/pufferlib/resources/cartpole/cartpole_weights.bin";


// Map discrete action to force
float movement(int discrete_action, int userControlMode) {
    // For discrete mode, we assume action 0 means left, action 1 means right.
    // When in user control, we override with keyboard input.
    if (userControlMode) {
        // Use keyboard control: default left, unless RIGHT key is pressed.
        return (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) ? 1.0f : -1.0f;
    } else {
        return (discrete_action == 1) ? 1.0f : -1.0f;
    }
}

// Deterministic discrete forward pass for a LSTM policy.
// Instead of softmax sampling, we take the argmax over the actor's output.
void forward_linearlstm_deterministic(LinearLSTM* net, float* observations, int* actions) {
    linear(net->encoder, observations);
    relu(net->relu1, net->encoder->output);
    lstm(net->lstm, net->relu1->output);
    linear(net->actor, net->lstm->state_h);
    // Instead of softmax sampling, use argmax:
    {
        int logit_size = net->actor->output_dim;  // should equal ACTIONS_SIZE, which is 2.
        // We'll assume num_agents == 1.
        float* logits = net->actor->output;
        float max_logit = logits[0];
        int selected = 0;
        for (int i = 1; i < logit_size; i++) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                selected = i;
            }
        }
        actions[0] = selected;
    }
}

void demo() {
    Weights* weights = load_weights(WEIGHTS_PATH, NUM_WEIGHTS);
    LinearLSTM* net = make_linearlstm_float(weights, 1, OBSERVATIONS_SIZE, ACTIONS_SIZE,
        CONTINUOUS ? ACTION_TYPE_FLOAT : ACTION_TYPE_INT);

    CartPole env = {0};
    env.continuous = CONTINUOUS;
    allocate(&env);
    Client* client = make_client(&env);
    c_reset(&env);

    SetTargetFPS(60);
    int episode_steps = 0;
    float episode_return = 0.0f;

    while (!WindowShouldClose()) {
        int userControlMode = IsKeyDown(KEY_LEFT_SHIFT);

        if (!userControlMode) {
            if (CONTINUOUS) {
                forward_linearlstm_float(net, env.observations, env.actions);
                env.actions[0] = tanhf(env.actions[0]);
            } else {
                int discrete_action[ACTIONS_SIZE]; // intermediate buffer for discrete actions
                forward_linearlstm_deterministic(net, env.observations, discrete_action);
                env.actions[0] = movement(discrete_action[0], 0);
            }
        } else {
            env.actions[0] = movement(env.actions[0], userControlMode);
        }

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
