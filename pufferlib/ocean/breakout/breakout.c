#include "puffernet.h"   // This header provides definitions for Weights, LinearLSTM, load_weights, etc.
#include "breakout.h"
#include <time.h>

// Macro definitions for our model and environment dimensions:
#define NUM_WEIGHTS 147715
#define OBSERVATIONS_SIZE 119
#define ACTIONS_SIZE 1

const char* WEIGHTS_PATH = "/puffertank/pufferlib/pufferlib/resources/breakout/breakout_weights.bin";

void demo() {
    // Load the model weights (using puffernet.h's load_weights)
    Weights* weights = load_weights(WEIGHTS_PATH, NUM_WEIGHTS);
// float decoder_logstd[ACTIONS_SIZE] = {-0.8550374f};
    LinearLSTM* net = make_linearlstm_float(weights, 1, OBSERVATIONS_SIZE, ACTIONS_SIZE, ACTION_TYPE_FLOAT);
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
        .continuous = 1,
        .frameskip = 4,
        .ball_speed = 256
    };
    allocate(&env);
    c_reset(&env);

    Client* client = make_client(&env);

    while (!WindowShouldClose()) {
        // Allow user to override the network's action via manual input.
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (env.continuous) {
                float move = GetMouseWheelMove();
                float clamped_wheel = fmaxf(-1.0f, fminf(1.0f, move));
                env.actions[0] = clamped_wheel;
            } else {
                env.actions[0] = 0.0;
                if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A))
                    env.actions[0] = 1;
                if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D))
                    env.actions[0] = 2;
            }
        } else {
            // Otherwise, use the model’s deterministic output (using tanh for evaluation)
            forward_linearlstm_float(net, env.observations, env.actions);
            // Note: the output is passed through tanhf(mu) in make_linearlstm_float.
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
        .continuous = 1,
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
