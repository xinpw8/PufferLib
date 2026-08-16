#include <stdio.h>
#include <time.h>
#include "flappy.h"

int main(void) {
    Flappy env = {
        .width = 420,
        .height = 640,
        .max_steps = 4096,
        .gravity = 0.45f,
        .flap_velocity = -7.5f,
        .pipe_speed = 3.0f,
        .pipe_gap = 190.0f,
        .pipe_width = 58.0f,
        .pipe_spacing = 220.0f,
        .first_pipe_x = 220.0f,
        .bird_x = 96.0f,
        .bird_radius = 14.0f,
        .alive_reward = 0.01f,
        .pass_reward = 1.0f,
        .crash_reward = -1.0f,
        .center_reward = 0.03f,
        .rng = (unsigned int)time(NULL),
    };

    init(&env);
    env.agents[0].observations = (float*)calloc(OBS_SIZE, sizeof(float));
    env.agents[0].actions = (float*)calloc(1, sizeof(float));
    env.agents[0].rewards = (float*)calloc(1, sizeof(float));
    env.agents[0].terminals = (float*)calloc(1, sizeof(float));
    puf_reset(&env);
    puf_render(&env);

    while (!WindowShouldClose()) {
        puf_step(&env);
        puf_render(&env);
    }

    puf_close(&env);
    free(env.agents[0].observations);
    free(env.agents[0].actions);
    free(env.agents[0].rewards);
    free(env.agents[0].terminals);
    return 0;
}
