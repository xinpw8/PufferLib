#include "onestateworld.h"

int main() {
    World env = {
        .mean_left = 0.1f,
        .mean_right = 0.5f,
        .var_right = 10.0f
    };
    allocate_World(&env);

    puf_reset(&env);
    puf_render(&env);
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.agents[0].actions[0] = 0;
            if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.agents[0].actions[0] = LEFT;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.agents[0].actions[0] = RIGHT;
        } else {
            env.agents[0].actions[0] = rand() % 2;
        }
        puf_step(&env);
        puf_render(&env);
    }
    free_allocated(&env);
}

