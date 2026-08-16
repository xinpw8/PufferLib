#include <time.h>
#include "breakout.h"
#include "puffercpu.h"

void demo() {
    Weights* weights = load_weights("resources/breakout/breakout_weights.bin");
    int logit_sizes[1] = {3};
    PufferNet* net = make_puffernet(weights, 1, 118, 64, 2, logit_sizes, 1);

    Breakout env = {
        .frameskip = 1,
        .width = 576,
        .height = 330,
        .initial_paddle_width = 62,
        .paddle_width = 62,
        .paddle_height = 8,
        .ball_width = 32,
        .ball_height = 32,
        .brick_width = 32,
        .brick_height = 12,
        .brick_rows = 6,
        .brick_cols = 18,
        .initial_ball_speed = 256,
        .max_ball_speed = 448,
        .paddle_speed = 620,
        .continuous = 0,
    };
    env.agents[0].observations = (float*)calloc(OBS_SIZE, sizeof(float));
    env.agents[0].actions = (float*)calloc(1, sizeof(float));
    env.agents[0].rewards = (float*)calloc(1, sizeof(float));
    env.agents[0].terminals = (float*)calloc(1, sizeof(float));

    env.client = make_client(&env);

    puf_reset(&env);
    int frame = 0;
    SetTargetFPS(60);
    while (!WindowShouldClose()) {
        if (frame % 4 == 0) {
            forward_puffernet(net, env.agents[0].observations, env.agents[0].actions);
        }
        frame = (frame + 1) % 4;
        puf_step(&env);
        puf_render(&env);
    }
    free_puffernet(net);
    free(weights);
    free(env.agents[0].observations);
    free(env.agents[0].actions);
    free(env.agents[0].rewards);
    free(env.agents[0].terminals);
    close_client(env.client);
}

int main() {
    demo();
}
