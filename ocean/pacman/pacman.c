#include <time.h>
#include "pacman.h"
#include "puffercpu.h"

void demo() {
    // printf("OBSERVATIONS_COUNT: %d\n", OBSERVATIONS_COUNT);
    Weights* weights = load_weights("resources/pacman/pacman_weights.bin");
    int logit_sizes[1] = {4};
    PufferNet* net = make_puffernet(weights, 1, OBSERVATIONS_COUNT, 256, 6, logit_sizes, 1);

    PacmanEnv env = {
        .randomize_starting_position = false,
        .min_start_timeout = 0, // randomized ghost delay range
        .max_start_timeout = 49,
        .frightened_time = 35,   // ghost frighten time
        .max_mode_changes = 6,
        .scatter_mode_length = 700,
        .chase_mode_length = 70,
    };
    allocate(&env);
    puf_reset(&env);
 
    Client* client = make_client(&env);
    bool human_control = false;

    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env.agents[0].actions[0] = DOWN;
            if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env.agents[0].actions[0] = UP;
            if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.agents[0].actions[0] = LEFT;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.agents[0].actions[0] = RIGHT;
            human_control = true;
        } else {
            human_control = false;
        }

        if (!human_control) {
            forward_puffernet(net, env.agents[0].observations, env.agents[0].actions);
        }

        puf_step(&env);
        if (env.agents[0].terminals[0] > 0.5f) {
            puf_reset(&env);
        }

        for (int i = 0; i < FRAMES; i++) {
            puf_render(&env);
        }
    }
    free_puffernet(net);
    free(weights);
    free_allocated(&env);
    close_client(client);
}

int main() {
    demo();
    return 0;
}
