#include "click.h"

int main() {
    ClickEnv env = {0};
    env.width = 800;
    env.height = 600;
    env.target_spawn_duration = 200;
    env.episode_length = 1000;
    env.rng = 1234;
    env.num_agents = 1;
    env.agents[0].observations = (float*)calloc(OBS_SIZE, sizeof(float));
    env.agents[0].actions = (float*)calloc(NUM_ATNS, sizeof(float));
    env.agents[0].rewards = (float*)calloc(1, sizeof(float));
    env.agents[0].terminals = (float*)calloc(1, sizeof(float));

    init(&env);
    puf_reset(&env);
    puf_render(&env);
    while (!WindowShouldClose()) {
        puf_step(&env);
        puf_render(&env);
    }

    free(env.agents[0].observations);
    free(env.agents[0].actions);
    free(env.agents[0].rewards);
    free(env.agents[0].terminals);
    puf_close(&env);
    return 0;
}
