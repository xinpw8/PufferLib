#include "terraform.h"

void allocate(Terraform* env) {
    env->observations = (unsigned char*)calloc(env->num_agents*246, sizeof(unsigned char));
    env->actions = (int*)calloc(3*env->num_agents, sizeof(int));
    env->rewards = (float*)calloc(env->num_agents, sizeof(float));
    env->terminals = (unsigned char*)calloc(env->num_agents, sizeof(unsigned char));
    init(env);
}

void free_allocated(Terraform* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free_initialized(env);
}

void demo() {
    //Weights* weights = load_weights("resources/pong_weights.bin", 133764);
    //LinearLSTM* net = make_linearlstm(weights, 1, 8, 3);

    Terraform env = {.size = 512, .num_agents = 8};
    allocate(&env);

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        for (int i = 0; i < env.num_agents; i++) {
            env.actions[3*i] = 4; //rand() % 5;
            env.actions[3*i + 1] = rand() % 5;
            env.actions[3*i + 2] = rand() % 3;
        }
        env.actions[0] = 2;
        env.actions[1] = 2;
        env.actions[2] = 0;
        if (IsKeyDown(KEY_UP)    || IsKeyPressed(KEY_W)) env.actions[0] = 4;
        if (IsKeyDown(KEY_DOWN)  || IsKeyPressed(KEY_S)) env.actions[0] = 0;
        if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[1] = 4;
        if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[1] = 0;
        if (IsKeyDown(KEY_SPACE)) env.actions[2] = 1;
        if (IsKeyPressed(KEY_LEFT_SHIFT)) {
            env.actions[2] = 2;
        }

        c_step(&env);
        c_render(&env);
    }
    //free_linearlstm(net);
    //free(weights);
    free_allocated(&env);
}

void test_performance(int timeout) {
    Terraform env = {
        .size = 128,
        .num_agents = 8,
    };
    allocate(&env);
    c_reset(&env);

    int start = time(NULL);
    int num_steps = 0;
    while (time(NULL) - start < timeout) {
        for (int i = 0; i < env.num_agents; i++) {
            env.actions[3*i] = rand() % 5;
            env.actions[3*i + 1] = rand() % 5;
            env.actions[3*i + 2] = rand() % 3;
        }

        c_step(&env);
        num_steps++;
    }

    int end = time(NULL);
    float sps = num_steps / (end - start);
    printf("Test Environment SPS: %f\n", sps);
    free_allocated(&env);
}

int main() {
    //test_performance(10);
    demo();
}

