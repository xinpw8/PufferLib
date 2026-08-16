#include "maze.h"
#include "puffercpu.h"

void demo() {
    Weights* weights = load_weights("resources/maze/maze_weights.bin");
    int logit_sizes[1] = {5};
    PufferNet* net = make_puffernet(weights, 1, 121, 512, 5, logit_sizes, 1);

    int num_maps = 64;
    int horizon = 256;
    float speed = 1;
    int vision = 5;
    bool discretize = true;

    Grid* env = (Grid*)calloc(1, sizeof(Grid));
    env->num_agents = 1;
    env->rng = 73;
    env->agents[0].observations = calloc(WINDOW * WINDOW, sizeof(unsigned char));
    env->agents[0].actions = calloc(1, sizeof(float));
    env->agents[0].rewards = calloc(1, sizeof(float));
    env->agents[0].terminals = calloc(1, sizeof(float));

    // Generate maps matching binding.c: random odd sizes, random difficulty
    State* levels = calloc(num_maps, sizeof(State));
    unsigned int map_rng = 42;
    for (int i = 0; i < num_maps; i++) {
        int sz = 5 + (rand_r(&map_rng) % (MAX_SIZE - 5));
        if (sz % 2 == 0) sz -= 1;
        float difficulty = (float)rand_r(&map_rng) / (float)(RAND_MAX);
        State* level = &levels[i];
        level->width = sz;
        level->height = sz;
        create_maze_level(level, difficulty, i);
    }

    env->num_levels = num_maps;
    env->levels = levels;

    puf_reset(env);
    puf_render(env);
    while (!WindowShouldClose()) {
        float obs[121];
        obs_t* src = env->agents[0].observations;
        for (int i = 0; i < 121; i++) {
            obs[i] = src[i];
        }
        forward_puffernet(net, obs, env->agents[0].actions);
        puf_step(env);
        puf_render(env);
    }
    
    free_puffernet(net);
    free(weights);
    free(env->agents[0].observations);
    free(env->agents[0].actions);
    free(env->agents[0].rewards);
    free(env->agents[0].terminals);
    puf_close(env);
    free(levels);
}

int main() {
    demo();
    return 0;
}
