#include "drmario.h" 

int main() {
    DrMario env = {0};
    env.n_rows = 16;
    env.n_cols = 8;
    env.n_init_viruses = 14;
    env.rng = (unsigned int)time(NULL);
    
    allocate(&env);
    puf_reset(&env);
    
    env.agents[0].actions[0] = 0;
    int frame = 0;
    bool processLogic;
    while (1) {
        frame += 1;
        processLogic = true;
        
        if(IsKeyDown(KEY_LEFT_SHIFT)){ 
            processLogic = frame % 3 == 0;

            if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
                env.agents[0].actions[0] = ACTION_LEFT;
            } else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
                env.agents[0].actions[0] = ACTION_RIGHT;
            } else if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) {
                env.agents[0].actions[0] = ACTION_DOWN;
            } else if (IsKeyPressed(KEY_Z)) {
                env.agents[0].actions[0] = ACTION_ROTATE_LEFT;
            } else if (IsKeyPressed(KEY_X)) {
                env.agents[0].actions[0] = ACTION_ROTATE_RIGHT;
            } else if (IsKeyPressed(KEY_SPACE)) {
                env.agents[0].actions[0] = ACTION_DROP;
            }

            if (IsKeyPressed(KEY_R)) puf_reset(&env);
        }
        
        if(processLogic){
            puf_step(&env);
            env.agents[0].actions[0] = 0;
        }
        puf_render(&env);

    }
    
    free_allocated(&env);
    return 0;
}