#include "puffernet.h"
#include "breakout.h"
#include <time.h>

#define NUM_WEIGHTS 147972
#define OBSERVATIONS_SIZE 119
#define ACTIONS_SIZE 3
#define CONTINUOUS 0

const char* WEIGHTS_PATH = "/puffertank/pufferlib/pufferlib/resources/breakout/breakout_weights.bin";

void forward_linearlstm_deterministic(LinearLSTM* net, float* observations, int* actions) {
    linear(net->encoder, observations);
    relu(net->relu1, net->encoder->output);
    lstm(net->lstm, net->relu1->output);
    linear(net->actor, net->lstm->state_h);
    {
        int logit_size = net->actor->output_dim;  // equals ACTIONS_SIZE.
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

    Breakout env = {
        .width = 576,
        .height = 330,
        .paddle_width = 62,
        .paddle_height = 8,
        .ball_width = 32,
        .ball_height = 32,
        .brick_width = 32,
        .brick_height = 12,
        .brick_rows = 6,
        .brick_cols = 18,
        .continuous = CONTINUOUS,
        .frameskip = 4,
        .ball_speed = 256
    };
    allocate(&env);
    c_reset(&env);
    Client* client = make_client(&env);

    while (!WindowShouldClose()) {
        int userControlMode = IsKeyDown(KEY_LEFT_SHIFT);
        
        if (!userControlMode) {
            if (CONTINUOUS) {
                forward_linearlstm_float(net, env.observations, env.actions);
                env.actions[0] = tanhf(env.actions[0]);
            } else {
                int discrete_action[ACTIONS_SIZE];
                forward_linearlstm_deterministic(net, env.observations, discrete_action);
                env.actions[0] = discrete_action[0];
            }
        } else {
            // Manual override via keyboard for debugging/testing
            if (CONTINUOUS) {
                float move = GetMouseWheelMove();
                env.actions[0] = fmaxf(-1.0f, fminf(1.0f, move));
            } else {
                env.actions[0] = NOOP;
                if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A))
                    env.actions[0] = LEFT;
                if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D))
                    env.actions[0] = RIGHT;
            }
        }

        c_step(&env);
        c_render(client, &env);
    }

    free_linearlstm(net);
    free(weights);
    free_allocated(&env);
    close_client(client);
}

void performance_test() {
    long test_time = 10;
    Breakout env = {
        .width = 576,
        .height = 330,
        .paddle_width = 62,
        .paddle_height = 8,
        .ball_width = 32,
        .ball_height = 32,
        .brick_width = 32,
        .brick_height = 12,
        .brick_rows = 6,
        .brick_cols = 18,
        .continuous = CONTINUOUS,
        .frameskip = 4,
        .ball_speed = 256
    };
    allocate(&env);
    c_reset(&env);

    long start = time(NULL);
    int i = 0;
    while (time(NULL) - start < test_time) {
        env.actions[0] = rand() % 4;
        c_step(&env);
        i++;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", i / (end - start));
    free_initialized(&env);
}

int main() {
    // Uncomment the following line to run the performance test instead of demo.
    // performance_test();
    demo();
    return 0;
}
