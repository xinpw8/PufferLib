#include "snake.h"
#include "puffernet.h"

int main() {
    CSnake env = {
        .num_snakes = 1024,
        .width = 1280,
        .height = 720,
        .max_snake_length = 200,
        .food = 16384,
        .vision = 5,
        .leave_corpse_on_death = true,
        .reward_food = 1.0f,
        .reward_corpse = 0.5f,
        .reward_death = -1.0f,
    };
    allocate_csnake(&env);
    reset(&env);

    Weights* weights = load_weights("resources/snake_weights.bin", 148357);
    LinearLSTM* net = make_linearlstm(weights, env.num_snakes, env.obs_size, 4);
    Client* client = make_client(1, env.width, env.height);

    while (!WindowShouldClose()) {
        // User can take control of the first snake
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env.actions[0] = 0;
            if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env.actions[0] = 1;
            if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[0] = 2;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = 3;
        } else {
            for (int i = 0; i < env.num_snakes*env.obs_size; i++) {
                net->obs[i] = (float)env.observations[i];
            }
            forward_linearlstm(net, net->obs, env.actions);
        }
        step(&env);
        render(client, &env);
    }
    free_linearlstm(net);
    free(weights);
    close_client(client);
    free_csnake(&env);
    return 0;
}

