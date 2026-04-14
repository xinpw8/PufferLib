#include "drmario.h"  // or dr_mario.h depending on your filename

int main() {
    DrMario env = {0};
    env.n_rows = 16;
    env.n_cols = 8;
    env.n_init_viruses = 10;
    env.rng = (unsigned int)time(NULL);
    
    allocate(&env);
    c_reset(&env);
    
    while (1) {
        c_step(&env);
        c_render(&env);
        if (IsKeyPressed(KEY_R)) c_reset(&env);
    }
    
    free_allocated(&env);
    return 0;
}