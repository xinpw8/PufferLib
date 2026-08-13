#include "tactical.h"


int main() {
    Tactical env = {0};
    init_tactical(&env);
    
    env.client = init_client(&env);

    puf_reset(&env);
    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_Q) || IsKeyPressed(KEY_BACKSPACE)) break;
        puf_step(&env);
        puf_render(&env);
    }

    close_client(env.client);
    puf_close(&env);
    
    // free_linearlstm(net);
    // free(weights);
    // free_allocated(&env);
    // close_client(client);
}

