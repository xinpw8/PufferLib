/*
 * Local random-policy smoke demo (./build.sh wef --local && ./wef).
 * Model eval: ./build.sh wef --cpu && ./build_cpu wef
 */

#include <time.h>
#include "wef.h"

int main(void) {
    Dict kwargs = {0};
    dict_set(&kwargs, "num_agents", 4);
    dict_set(&kwargs, "min_arena_width", 70);
    dict_set(&kwargs, "min_arena_height", 70);
    dict_set(&kwargs, "max_arena_width", 70);
    dict_set(&kwargs, "max_arena_height", 70);
    dict_set(&kwargs, "food_distribution", FOOD_RANDOM);
    dict_set(&kwargs, "num_food", 64);
    dict_set(&kwargs, "patch_radius", 6);
    dict_set(&kwargs, "patch_radius_std", 1.5);
    dict_set(&kwargs, "patch_density", 0.001);
    dict_set(&kwargs, "electric_field_radius", 15);
    dict_set(&kwargs, "reflection_wall_range", 100);
    dict_set(&kwargs, "field_fish_range", 100);
    dict_set(&kwargs, "field_food_range", 5);
    dict_set(&kwargs, "episode_length", 512);

    Wef env = {0};
    env.rng = (unsigned int)time(NULL);
    puf_init(&env, &kwargs);
    dict_clear(&kwargs);

    obs_t observations[MAX_AGENTS * OBS_SIZE];
    float actions[MAX_AGENTS * NUM_ATNS];
    float rewards[MAX_AGENTS];
    float terminals[MAX_AGENTS];
    memset(observations, 0, sizeof(observations));
    memset(actions, 0, sizeof(actions));
    memset(rewards, 0, sizeof(rewards));
    memset(terminals, 0, sizeof(terminals));
    for (int i = 0; i < env.num_agents; i++) {
        env.agents[i].observations = observations + i * OBS_SIZE;
        env.agents[i].actions = actions + i * NUM_ATNS;
        env.agents[i].rewards = rewards + i;
        env.agents[i].terminals = terminals + i;
    }
    puf_reset(&env);
    puf_render(&env);

    unsigned int demo_rng = env.rng ^ 0x9e3779b9U;
    float target[MAX_AGENTS][ACTION_SIZE] = {{0}};
    while (!WindowShouldClose()) {
        if (env.tick % 45 == 0) {
            for (int i = 0; i < env.num_agents; i++) {
                target[i][0] =
                    2.0f * (float)rand_r(&demo_rng) / (float)RAND_MAX - 1.0f;
                target[i][1] =
                    2.0f * (float)rand_r(&demo_rng) / (float)RAND_MAX - 1.0f;
                target[i][2] = rand_r(&demo_rng) % 100 < 35 ? 1.0f : -1.0f;
                target[i][3] = rand_r(&demo_rng) % 100 < 8 ? 1.0f : -1.0f;
            }
        }
        for (int i = 0; i < env.num_agents; i++) {
            float* a = env.agents[i].actions;
            a[0] += 0.06f * (target[i][0] - a[0]);
            a[1] += 0.08f * (target[i][1] - a[1]);
            a[2] = target[i][2];
            a[3] = target[i][3];
        }
        puf_step(&env);
        puf_render(&env);
    }
    puf_close(&env);
    return 0;
}
