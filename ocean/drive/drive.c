#include <time.h>
#include <unistd.h>
#include "drive.h"
#include "puffercpu.h"

static void bind_demo_buffers(Drive* env) {
    int n = env->active_agent_count;
    if (n < 1) {
        n = 1;
    }
    env->num_agents = n;
    float* observations = (float*)calloc((size_t)n * OBS_SIZE, sizeof(float));
    float* actions = (float*)calloc((size_t)n * NUM_ATNS, sizeof(float));
    float* rewards = (float*)calloc((size_t)n, sizeof(float));
    float* terminals = (float*)calloc((size_t)n, sizeof(float));
    for (int i = 0; i < n; i++) {
        env->agents[i].observations = observations + i * OBS_SIZE;
        env->agents[i].actions = actions + i * NUM_ATNS;
        env->agents[i].rewards = rewards + i;
        env->agents[i].terminals = terminals + i;
        env->agents[i].action_mask = NULL;
        env->agents[i].policy = 0;
    }
}

static void free_demo_buffers(Drive* env) {
    free(env->agents[0].observations);
    free(env->agents[0].actions);
    free(env->agents[0].rewards);
    free(env->agents[0].terminals);
}

void demo() {
    Drive env = {
        .dynamics_model = CLASSIC,
        .human_agent_idx = 0,
        .reward_vehicle_collision = -0.1f,
        .reward_offroad_collision = -0.1f,
	    .map_name = "resources/drive/map_010.bin",
    };
    init(&env);
    bind_demo_buffers(&env);
    puf_reset(&env);
    puf_render(&env);
    Weights* weights = load_weights("resources/drive/drive_weights.bin");
    int logit_sizes[2] = {7, 13};
    PufferNet* net = make_puffernet(weights, env.active_agent_count, OBS_SIZE, 256, 4, logit_sizes, 2);
    while (!WindowShouldClose()) {
        forward_puffernet(net, env.agents[0].observations, env.agents[0].actions);
        puf_step(&env);
        puf_render(&env);
    }

    close_client(env.client);
    free_demo_buffers(&env);
    puf_close(&env);
    free_puffernet(net);
    free(weights);
}

void performance_test() {
    long test_time = 10;
    Drive env = {
        .dynamics_model = CLASSIC,
        .human_agent_idx = 0,
	    .map_name = "resources/drive/map_942.bin",
    };
    init(&env);
    bind_demo_buffers(&env);
    puf_reset(&env);

    Weights* weights = load_weights("resources/drive/drive_weights.bin");
    int logit_sizes[2] = {7, 13};
    PufferNet* net = make_puffernet(weights, env.active_agent_count, OBS_SIZE, 256, 4, logit_sizes, 2);

    long start = time(NULL);
    int i = 0;
    while (time(NULL) - start < test_time) {
        forward_puffernet(net, env.agents[0].observations, env.agents[0].actions);
        puf_step(&env);
        i++;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", (long)(i*env.active_agent_count) / (end - start));
    free_demo_buffers(&env);
    puf_close(&env);
    free_puffernet(net);
    free(weights);
}

int main() {
    demo();
    return 0;
}
