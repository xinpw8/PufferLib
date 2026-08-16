#include <time.h>
#include <unistd.h>

#include "chain_reaction.h"

int main(void) {
    ChainEnv env = {0};
    env.num_agents = 1;
    env.rows = 9;
    env.cols = 6;
    env.max_steps = 132;
    env.opponent_policy = HEURISTIC_OPPONENT;
    env.win_reward = 1.0f;
    env.loss_reward = -1.0f;
    env.invalid_move_reward = -1.0f;
    env.rng = (unsigned int)time(NULL) ^ (unsigned int)getpid();
    init(&env);

    float observation_buf[OBS_SIZE] = {0};
    float action_buf[1] = {0};
    float reward_buf[1] = {0};
    float terminal_buf[1] = {0};
    unsigned char action_mask_buf[MAX_ACTIONS] = {0};
    env.agents[0].observations = observation_buf;
    env.agents[0].actions = action_buf;
    env.agents[0].rewards = reward_buf;
    env.agents[0].terminals = terminal_buf;
    env.agents[0].action_mask = action_mask_buf;
    env.agents[0].policy = 0;
    puf_render(&env);
    puf_reset(&env);

    while (!WindowShouldClose()) {
        puf_step(&env);
        puf_render(&env);
    }
}
