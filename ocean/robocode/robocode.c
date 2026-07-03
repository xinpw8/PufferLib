#include "robocode.h"
#include <time.h>

static void bind_agents(Robocode* env, obs_t* observations,
        float* actions, float* rewards, float* terminals) {
    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].observations = observations + i * (EGO_FEATURES + OTHER_FEATURES);
        env->agents[i].actions = actions + i * NUM_ACTIONS;
        env->agents[i].rewards = rewards + i;
        env->agents[i].terminals = terminals + i;
        env->agents[i].action_mask = NULL;
        env->agents[i].policy = 0;
    }
}

void performance_test() {
    long test_time = 10;
    Robocode env = {
        .num_agents = 2,
        .num_bots = 0,
        .width = 800,
        .height = 600,
        .reward_damage = 0.01f,
        .reward_spot = 0.001f,
        .bot_policy = 3,  // BOT_WAVE_SURFER
        .max_ticks = 3000,
        .rng = 42,
    };
    allocate_env(&env);
    obs_t observations[2 * (EGO_FEATURES + OTHER_FEATURES)] = {0};
    float actions[2 * NUM_ACTIONS] = {0};
    float rewards[2] = {0};
    float terminals[2] = {0};
    bind_agents(&env, observations, actions, rewards, terminals);
    puf_reset(&env);

    long start = time(NULL);
    int i = 0;
    while (time(NULL) - start < test_time) {
        float* actions = env.agents[0].actions;
        actions[0] = rand_r(&env.rng) % 4;
        actions[1] = rand_r(&env.rng) % 9;
        actions[2] = rand_r(&env.rng) % 11;
        actions[3] = rand_r(&env.rng) % 11;
        actions[4] = (rand_r(&env.rng) % 6) > 4 ? 1.0f : 0.0f;
        puf_step(&env);
        i++;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", (long)i*env.num_agents / (end - start));
    puf_close(&env);
}

void demo(void) {
    Robocode env = {
        .num_agents = 1,
        .num_bots = 1,
        .reward_damage = 0.01,
        .width = 800,
        .height = 600,
        .max_ticks = 512,
    };
    allocate_env(&env);
    obs_t observations[EGO_FEATURES + OTHER_FEATURES] = {0};
    float actions[NUM_ACTIONS] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    bind_agents(&env, observations, actions, rewards, terminals);
    puf_reset(&env);

    env.client = make_client(&env);
    puf_render(&env);

    while (!WindowShouldClose()) {
        float* actions = env.agents[0].actions;
        actions[0] = 2;
        actions[1] = 4;
        actions[2] = 5;
        actions[3] = 5;
        actions[4] = 0;

        if (IsKeyPressed(KEY_ESCAPE)) break;
        if (IsKeyDown(KEY_W)) actions[0] = 3.0f;
        if (IsKeyDown(KEY_S)) actions[0] = 1.0f;
        if (IsKeyDown(KEY_A)) actions[1] = 3.0f;
        if (IsKeyDown(KEY_D)) actions[1] = 5.0f;
        if (IsKeyDown(KEY_Q)) actions[2] = 4.0f;
        if (IsKeyDown(KEY_E)) actions[2] = 6.0f;
        if (IsKeyDown(KEY_LEFT)) actions[3] = 0.0f;
        if (IsKeyDown(KEY_RIGHT)) actions[3] = 8.0f;
        if (IsKeyDown(KEY_SPACE)) actions[4] = 1.0f;

        puf_step(&env);
        puf_render(&env);
    }
    puf_close(&env);
    CloseWindow();
}

int main() {
    demo();
    //performance_test();
    return 0;
}
