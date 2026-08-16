#include "double_pendulum.h"

int main(void) {
    float observations[DP_OBS_SIZE] = {0};
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};

    DoublePendulum env = {
        .num_agents = 1,
        .rng = 1,
        .cart_mass = 1.0f,
        .link1_mass = 0.1f,
        .link2_mass = 0.1f,
        .link1_length = 0.5f,
        .link2_length = 0.5f,
        .gravity = 9.8f,
        .force_mag = 10.0f,
        .dt = 0.02f,
    };

    env.agents[0].observations = observations;
    env.agents[0].actions = actions;
    env.agents[0].rewards = rewards;
    env.agents[0].terminals = terminals;
    init(&env);
    puf_reset(&env);
    puf_render(&env);
    while (!WindowShouldClose()) {
        puf_step(&env);
        puf_render(&env);
    }
    puf_close(&env);
    return 0;
}
