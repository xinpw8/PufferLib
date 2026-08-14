#include <time.h>
#include "trash_pickup.h"
#include "puffercpu.h"

void demo() {
    CTrashPickupEnv env = {
        .grid_size = 20,
        .num_agents = 8,
        .num_trash = 40,
        .num_bins = 2,
        .max_steps = 300,
        .agent_sight_range = 5,
        .do_human_control = true
    };

    Weights* weights = load_weights("resources/trash_pickup/trash_pickup_weights.bin");
    int logit_sizes[1] = {4};
    PufferNet* net = make_puffernet(weights, env.num_agents, 605, 128, 2, logit_sizes, 1);

    allocate(&env);
    puf_reset(&env);
    puf_render(&env);

    int tick = 0;
    while (!WindowShouldClose()) {
        if (tick % 2 == 0) {
            for (int a = 0; a < env.num_agents; a++) {
                obs_t* obs = (obs_t*)env.agents[a].observations;
                for (int e = 0; e < 605; e++) {
                    net->obs[a * 605 + e] = obs[e];
                }
            }
            float actions[8];
            for (int a = 0; a < env.num_agents; a++) {
                actions[a] = 0.0f;
            }
            forward_puffernet(net, net->obs, actions);
            for (int a = 0; a < env.num_agents; a++) {
                env.agents[a].actions[0] = actions[a];
            }

            if (IsKeyDown(KEY_LEFT_SHIFT)) {
                if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) env.agents[0].actions[0] = ACTION_UP;
                if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) env.agents[0].actions[0] = ACTION_LEFT;
                if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.agents[0].actions[0] = ACTION_RIGHT;
                if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) env.agents[0].actions[0] = ACTION_DOWN;
            }

            puf_step(&env);
        }
        tick++;
        puf_render(&env);
    }

    free_puffernet(net);
    free(weights);
    free_allocated(&env);
}

int main() {
    demo();
    return 0;
}
