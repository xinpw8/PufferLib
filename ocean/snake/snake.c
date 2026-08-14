#include <time.h>
#include "snake.h"
#include "puffercpu.h"

int demo() {
    CSnake env = {
        .num_agents = 256,
        .width = 640,
        .height = 360,
        .max_snake_length = 200,
        .food = 4096,
        .vision = 5,
        .leave_corpse_on_death = true,
        .reward_food = 1.0f,
        .reward_corpse = 0.5f,
        .reward_death = -1.0f,
    };
    allocate_csnake(&env);
    puf_reset(&env);

    Weights* weights = load_weights("resources/snake/snake_weights.bin");
    int logit_sizes[] = {4};
    LinearLSTM* net = make_linearlstm(weights, env.num_agents, OBS_SIZE, logit_sizes, 1);
    env.client = make_client(2, env.width, env.height);

    while (!WindowShouldClose()) {
        // User can take control of the first snake
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env.agents[0].actions[0] = 0;
            if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env.agents[0].actions[0] = 1;
            if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.agents[0].actions[0] = 2;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.agents[0].actions[0] = 3;
        } else {
            for (int a = 0; a < env.num_agents; a++) {
                obs_t* src = (obs_t*)env.agents[a].observations;
                float* dst = net->obs + a * OBS_SIZE;
                for (int i = 0; i < OBS_SIZE; i++) {
                    dst[i] = (float)src[i];
                }
            }
            int* actions = (int*)calloc(env.num_agents, sizeof(int));
            forward_linearlstm(net, net->obs, actions);
            for (int i = 0; i < env.num_agents; i++) {
                env.agents[0].actions[i] = actions[i];
            }
            free(actions);
        }
        puf_step(&env);
        puf_render(&env);
    }
    free_linearlstm(net);
    free(weights);
    close_client(env.client);
    free_csnake(&env);
    return 0;
}

void test_performance(float test_time) {
    CSnake env = {
        .num_agents = 1024,
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
    puf_reset(&env);

    int start = time(NULL);
    int i = 0;
    while (time(NULL) - start < test_time) {
        for (int j = 0; j < env.num_agents; j++) {
            env.agents[0].actions[j] = rand()%4;
        }
        puf_step(&env);
        i++;
    }
    int end = time(NULL);
    free_csnake(&env);
    printf("SPS: %f\n", (float)env.num_agents*i / (end - start));
}

int main() {
    demo();
    // test_performance(30);
    return 0;
}
