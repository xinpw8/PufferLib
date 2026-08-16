#include "backgammon.h"

int main() {
    Backgammon env = {0};
    env.num_agents = 1;
    env.opponent_random_prob = 1.0f;
    env.agents[0].observations = (float*)calloc(OBS_SIZE, sizeof(float));
    env.agents[0].actions = (float*)calloc(1, sizeof(float));
    env.agents[0].rewards = (float*)calloc(1, sizeof(float));
    env.agents[0].terminals = (float*)calloc(1, sizeof(float));
    env.agents[0].action_mask = (unsigned char*)calloc(NUM_ACTIONS, sizeof(unsigned char));
    env.agents[0].policy = 0;

    puf_reset(&env);
    puf_render(&env);
    while (!WindowShouldClose()) {
        int chosen = 0;
        unsigned char* mask = env.agents[0].action_mask;
        int legal[NUM_ACTIONS];
        int n = 0;
        if (mask) {
            for (int i = 0; i < NUM_ACTIONS; i++) {
                if (mask[i]) legal[n++] = i;
            }
        }
        if (n > 0) {
            chosen = legal[rand() % n];
        }
        env.agents[0].actions[0] = (float)chosen;
        puf_step(&env);
        puf_render(&env);
    }

    free(env.agents[0].observations);
    free(env.agents[0].actions);
    free(env.agents[0].rewards);
    free(env.agents[0].terminals);
    free(env.agents[0].action_mask);
    puf_close(&env);
    return 0;
}
