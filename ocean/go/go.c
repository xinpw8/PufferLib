#include <time.h>
#include "go.h"
#include "puffercpu.h"

void demo(int grid_size) {

    Go env = {
        .width = 950,
        .height = 750,
        .grid_size = grid_size,
        .board_width = 600,
        .board_height = 600,
        .grid_square_size = 64,
        .komi = 7.5,
        .reward_move_pass = -0.518441,
        .reward_move_valid = 0,
        .reward_move_invalid = -0.0864746,
        .reward_player_capture = 0.553628,
        .reward_opponent_capture = -0.102283,
        .selfplay = 0,
        .side = 1,
    };

    Weights* weights = load_weights("resources/go/go_weights.bin");
    int logit_sizes[1] = {grid_size * grid_size + 1};
    int obs_size = grid_size * grid_size * 4 + 2;
    PufferNet* net = make_puffernet(weights, 1, obs_size, 512, 1, logit_sizes, 1);
    env.agents[0].observations = (float*)calloc(obs_size, sizeof(float));
    env.agents[0].actions = (float*)calloc(2, sizeof(float));
    env.agents[0].rewards = (float*)calloc(1, sizeof(float));
    env.agents[0].terminals = (float*)calloc(1, sizeof(float));
    puf_reset(&env);
    puf_render(&env);

    int tick = 0;
    while (!WindowShouldClose()) {
        if (tick % 3 == 0) {
            tick = 0;
            forward_puffernet(net, env.agents[0].observations, env.agents[0].actions);
            puf_step(&env);
        }
        tick++;
        puf_render(&env);
    }
    free_puffernet(net);
    free(weights);
    free(env.agents[0].observations);
    free(env.agents[0].actions);
    free(env.agents[0].rewards);
    free(env.agents[0].terminals);
    puf_close(&env);
}

int main() {
    demo(9);
    return 0;
}
