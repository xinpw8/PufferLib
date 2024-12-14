#include "blastar.h"

int main() {
    BlastarEnv env;
    BlastarRenderer renderer;

    init_blastar(&env);
    init_renderer(&renderer);

    while (!WindowShouldClose() && !env.gameOver) {
        int action = 0;

        // Handle input
        if (IsKeyDown(KEY_LEFT)) action = 1;  // Move left
        if (IsKeyDown(KEY_RIGHT)) action = 2; // Move right
        if (IsKeyDown(KEY_UP)) action = 3;    // Move up
        if (IsKeyDown(KEY_DOWN)) action = 4;  // Move down
        if (IsKeyPressed(KEY_SPACE)) action = 5; // Fire bullet

        // Step the environment
        step_blastar(&env, action);

        // Render the game
        render_blastar(&renderer, &env);
    }

    close_renderer(&renderer);
    return 0;
}
