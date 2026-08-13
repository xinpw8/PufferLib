#include "tmaze.h"

int main() {
    TMaze env = {.size = 8};
    allocate_TMaze(&env);

    puf_reset(&env);
    puf_render(&env);
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.agents[0].actions[0] = FORWARD;
            if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.agents[0].actions[0] = LEFT;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.agents[0].actions[0] = RIGHT;

        } else {
            env.agents[0].actions[0] = rand() % 3;
        }
        puf_step(&env);
        puf_render(&env);
    }
    free_allocated(&env);
}

