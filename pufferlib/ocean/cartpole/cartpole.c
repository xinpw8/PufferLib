#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cartpole.h"

#define NUM_WEIGHTS 133123
#define OBSERVATIONS_SIZE 4
#define ACTIONS_SIZE 2

const char* WEIGHTS_PATH = "pufferlib/resources/cartpole/cartpole_weights.bin";

typedef struct Policy {
    float* weights;
    int num_weights;
} Policy;

Policy* load_policy(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        perror("Could not load weights");
        exit(1);
    }
    Policy* p = malloc(sizeof(Policy));
    p->weights = malloc(NUM_WEIGHTS * sizeof(float));
    fread(p->weights, sizeof(float), NUM_WEIGHTS, f);
    fclose(f);
    return p;
}

int policy_forward(Policy* policy, float* observations) {
    // Simple dummy policy based on pole angle
    return (observations[2] > 0) ? 1 : 0;
}

void free_policy(Policy* policy) {
    free(policy->weights);
    free(policy);
}

void get_input(CartPole* env) {
    if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
        env->actions[0] = 0; // Left
    } else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
        env->actions[0] = 1; // Right
    }
}

void demo() {
    CartPole env = {
        .frameskip = 1,
        .width = 800,
        .height = 600,
        .max_steps = 500,
        .continuous = 0,
    };

    allocate(&env);
    c_reset(&env);

    Client* client = make_client(&env);
    Policy* policy = load_policy(WEIGHTS_PATH);

    InitWindow(env.width, env.height, "CartPole AI Control");
    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            get_input(&env);  // Manual override
        } else {
            env.actions[0] = policy_forward(policy, env.observations);  // AI action
        }

        c_step(&env);

        BeginDrawing();
        ClearBackground(RAYWHITE);
        c_render(client, &env);
        DrawText("Hold LEFT SHIFT for manual control", 10, 10, 20, DARKGRAY);
        EndDrawing();

        if (env.dones[0]) {
            c_reset(&env);
        }
    }

    free_policy(policy);
    close_client(client);
    free_allocated(&env);
    CloseWindow();
}

int main() {
    demo();
    return 0;
}
