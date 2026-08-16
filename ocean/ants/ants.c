/* Standalone ants demo. Keyboard (V/P/ESC) lives in puf_render.
 * Scripted forage AI is ants_scripted_act — not bound into render.
 */
#include "ants.h"

int main(void) {
    Ants env = {
        .width = 1280,
        .height = 720,
        .num_agents = 64,
        .reward_food_pickup = 0.1f,
        .reward_delivery = 10.0f,
        .rng = 1,
    };

    float observations[MAX_AGENTS * OBS_SIZE];
    float actions[MAX_AGENTS];
    float rewards[MAX_AGENTS];
    float terminals[MAX_AGENTS];
    for (int i = 0; i < env.num_agents; i++) {
        env.agents[i].observations = observations + i * OBS_SIZE;
        env.agents[i].actions = actions + i;
        env.agents[i].rewards = rewards + i;
        env.agents[i].terminals = terminals + i;
        env.agents[i].action_mask = NULL;
        env.agents[i].policy = 0;
    }

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
