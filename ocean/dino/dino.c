#include "dino.h"
#include "puffercpu.h"

int main() {
    Weights* weights = load_weights("resources/dino/dino_weights.bin");
    int logit_sizes[1] = {3};
    int obs_size = 5 + 3 * 9;
    PufferNet* net = make_puffernet(weights, 1, obs_size, 512, 1, logit_sizes, 1);

    Dinosaur env = {
        .width = 800,
        .height = 400,
        .speed_init = 6,
        .speed_max = 14,
        .spawn_rate_max = 65,
        .spawn_rate_min = 45,
        .rate_increment_rate = 600,
    };
    env.client = make_client(&env);
    env.agents[0].observations = (float*)calloc(OBS_SIZE, sizeof(float));
    env.agents[0].actions = (float*)calloc(1, sizeof(float));
    env.agents[0].rewards = (float*)calloc(1, sizeof(float));
    env.agents[0].terminals = (float*)calloc(1, sizeof(float));
    
    c_init(&env);
    puf_reset(&env);
    puf_render(&env);

    while (!WindowShouldClose()) {
        forward_puffernet(net, env.agents[0].observations, env.agents[0].actions);
        puf_step(&env);
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