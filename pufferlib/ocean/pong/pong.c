#include "pong.h"
#include "puffernet.h"
#include <time.h>

#define NUM_WEIGHTS 133507
#define OBSERVATIONS_SIZE 8
#define ACTIONS_SIZE 1
#define CONTINUOUS 1

const char* WEIGHTS_PATH = "/puffertank/pufferlib/pufferlib/resources/pong/pong_weights.bin";

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
    
    Pong env = {
        .width = 500,
        .height = 640,
        .paddle_width = 20,
        .paddle_height = 70,
        .ball_width = 32,
        .ball_height = 32,
        .paddle_speed = 8,
        .ball_initial_speed_x = 10,
        .ball_initial_speed_y = 1,
        .ball_speed_y_increment = 3,
        .ball_max_speed_y = 13,
        .max_score = 21,
        .frameskip = 1,
        .continuous = CONTINUOUS,
    };
    
    allocate(&env);
    Client* client = make_client(&env);
    c_reset(&env);
    
    int episode_steps = 0;
    float episode_return = 0.0f;
    
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
                env.actions[0] = 0;  // NOOP
                if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W))
                    env.actions[0] = 1;  // UP
                if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S))
                    env.actions[0] = 2;  // DOWN
            }
        }
        
        c_step(&env);
        episode_return += env.rewards[0];
        episode_steps++;
        
        c_render(client, &env);
        
        if (env.terminals[0]) {
            printf("Episode done. Steps: %d, Return: %.2f\n\n", episode_steps, episode_return);
            episode_steps = 0;
            episode_return = 0.0f;
            c_reset(&env);
        }
    }
    
    free_linearlstm(net);
    free(weights);
    free_allocated(&env);
    close_client(client);
}

void performance_test() {
    long test_time = 10;
    Pong env = {
        .width = 500,
        .height = 640,
        .paddle_width = 20,
        .paddle_height = 70,
        .ball_width = 32,
        .ball_height = 32,
        .paddle_speed = 8,
        .ball_initial_speed_x = 10,
        .ball_initial_speed_y = 1,
        .ball_speed_y_increment = 3,
        .ball_max_speed_y = 13,
        .max_score = 21,
        .frameskip = 1,
        .continuous = CONTINUOUS,
    };
    
    allocate(&env);
    c_reset(&env);
    
    long start = time(NULL);
    int i = 0;
    while (time(NULL) - start < test_time) {
        env.actions[0] = rand() % ACTIONS_SIZE;
        c_step(&env);
        i++;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", i / (end - start));
    
    free_allocated(&env);
}

int main() {
    srand(time(NULL));
    
    // Uncomment the following line to run the performance test instead of demo.
    // performance_test();
    
    demo();
    return 0;
}