#include "battle.h"

int main() {
    Battle env = {
        .width = 1980,
        .height = 1020,
        .size_x = 8,
        .size_y = 2,
        .size_z = 8,
        .num_agents = 128,
        .num_armies = 2,
    };
    env.num_units = env.num_agents * env.num_armies;
    init(&env);

    float observations[128 * OBS_SIZE] = {0};
    float actions[128 * NUM_ATNS] = {0};
    float rewards[128] = {0};
    float terminals[128] = {0};
    for (int i = 0; i < env.num_agents; i++) {
        env.agents[i].observations = observations + i * OBS_SIZE;
        env.agents[i].actions = actions + i * NUM_ATNS;
        env.agents[i].rewards = rewards + i;
        env.agents[i].terminals = terminals + i;
        env.agents[i].policy = 0;
        env.agents[i].action_mask = NULL;
    }

    puf_reset(&env);
    while (!WindowShouldClose()) {
        puf_step(&env);
        puf_render(&env);
    }
    puf_close(&env);
    return 0;
}
