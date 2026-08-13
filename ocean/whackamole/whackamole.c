#include "whackamole.h"

int main() {
    Whackamole env = {0};
    env.num_agents = 1;
    env.rng = (unsigned int)time(NULL);
    
    env.agents[0].observations = (float*)calloc(TOTAL_CELLS, sizeof(float));
    env.agents[0].actions = (float*)calloc(1, sizeof(float));
    env.agents[0].rewards = (float*)calloc(1, sizeof(float));
    env.agents[0].terminals = (float*)calloc(1, sizeof(float));
    
    puf_reset(&env);
    puf_render(&env);
    
    int frame = 0;
    while (1) {
        frame += 1;
        
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.agents[0].actions[0] = NOOP;
            if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
                Vector2 mouse = GetMousePosition();
                int c = (int)(mouse.x / CELL_SIZE);
                int r = (int)(mouse.y / CELL_SIZE);
                if (r >= 0 && r < GRID_SIZE && c >= 0 && c < GRID_SIZE) {
                    env.agents[0].actions[0] = (float)(r * GRID_SIZE + c);
                }
            }
            if (IsKeyPressed(KEY_R)) puf_reset(&env);
        } else {
            if (frame % 10 == 0) {
                env.agents[0].actions[0] = (float)(rand_r(&env.rng) % TOTAL_CELLS);
            } else {
                env.agents[0].actions[0] = NOOP;
            }
        }
        
        puf_step(&env);
        puf_render(&env);
    }
    
    free(env.agents[0].observations);
    free(env.agents[0].actions);
    free(env.agents[0].rewards);
    free(env.agents[0].terminals);
    puf_close(&env);
    
    return 0;
}