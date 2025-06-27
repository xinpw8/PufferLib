// chess.c
#include "chess.h"

int main() {
    // Demo or performance test
    CChess env = {
        .reward_valid_move = 0.0f,
        .reward_invalid_move = -0.1f,
        .reward_agent_captures_enemy_piece = 0.05f,
        .reward_enemy_captures_agent_piece = -0.05f,
        .reward_win = 1.0f,
        .reward_draw = 0.0f,
        .reward_loss = -1.0f,
    };
    
    allocate(&env);
    init(&env);
    c_reset(&env);
    
    // Simple performance test
    for (int i = 0; i < 1000; i++) {
        env.actions[0] = i % 4674;
        c_step(&env);
        if (env.terminals[0]) {
            c_reset(&env);
        }
    }
    
    c_close(&env);
    free_allocated(&env);
    return 0;
}