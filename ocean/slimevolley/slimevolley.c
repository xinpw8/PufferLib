/* Pure C demo file for SlimeVolley. Build it with:
 * bash scripts/build_ocean.sh target local (debug)
 * bash scripts/build_ocean.sh target fast
 * We suggest building and debugging your env in pure C first. You
 * get faster builds and better error messages
 */
#include "slimevolley.h"
#include "puffernet.h"
#include <stdio.h>


void abranti_simple_policy(float* obs, float* action) {
    float x_agent = obs[0];
    float x_ball = obs[4];
    float vx_ball = obs[6];
    float backward = (-23.757145f * x_agent + 23.206863f * x_ball + 0.7943352f * vx_ball) + 1.4617119f;
    float forward = -64.6463748f * backward + 22.4668393f;
    action[0] = forward;
    action[1] = backward;
    action[2] = 1.0f; // always jump
}


void demo() {
    int num_obs = 12;
    int num_actions = 3;
    SlimeVolley env = {.num_agents = 1};
    init(&env);
    env.agents[0].observations = (float*)calloc(env.num_agents*num_obs, sizeof(float));
    env.agents[0].actions = (float*)calloc(num_actions*env.num_agents, sizeof(float));
    env.agents[0].rewards = (float*)calloc(env.num_agents, sizeof(float));
    env.agents[0].terminals = (float*)calloc(env.num_agents, sizeof(float));

    Weights* weights = load_weights("resources/slimevolley/slimevolley_weights.bin");
    int logit_sizes[3] = {2, 2, 2};
    PufferNet* net = make_puffernet(weights, 1, num_obs, 128, 3, logit_sizes, 3);

    // Always call reset and render first
    puf_reset(&env);
    puf_render(&env);

    fprintf(stderr, "num agents: %d\n", env.num_agents);

    while (!WindowShouldClose()) {
        env.agents[0].actions[0] = 0.0f;
        env.agents[0].actions[1] = 0.0f;
        env.agents[0].actions[2] = 0.0f;
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.agents[0].actions[0] = 1.0f;
            if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) env.agents[0].actions[1] = 1.0f;
            if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W) || IsKeyDown(KEY_SPACE)) env.agents[0].actions[2] = 1.0f;
        } else {
            abranti_simple_policy(env.agents[0].observations, env.agents[0].actions);
        }
        puf_step(&env);
        puf_render(&env);
    }

    free_puffernet(net);
    free(weights);
    
    // Try to clean up after yourself
    free(env.agents[0].observations);
    free(env.agents[0].actions);
    free(env.agents[0].rewards);
    free(env.agents[0].terminals);
    puf_close(&env);
}

int main() {
    demo();
    return 0;
}
