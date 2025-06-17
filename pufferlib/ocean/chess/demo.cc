#include "pufferlib/pufferlib/ocean/chess/flat_chess_env.h"
#include <cstdlib>
#include <ctime>

int main() {
    std::srand(std::time(nullptr));

    CChess env{};
    allocate(&env);

    for (int t = 0; t < 50 && !env.terminals[0]; ++t) {
        env.actions[0] = std::rand() % 4096;   // random move
        c_step(&env);
        c_render(&env);
    }

    c_close(&env);
    free_allocated(&env);
    return 0;
}