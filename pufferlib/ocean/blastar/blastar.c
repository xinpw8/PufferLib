#include "blastar_env.h"
#include "blastar_renderer.h"

int main() {
    BlastarEnv env;
    init_blastar(&env);

    allocate_env(&env); 

    BlastarRenderer renderer;
    init_renderer(&renderer);

    while (!WindowShouldClose() && !env.gameOver) {
        int action = 0;
        if (IsKeyDown(KEY_LEFT)) action = 1;
        if (IsKeyDown(KEY_RIGHT)) action = 2;
        if (IsKeyDown(KEY_UP)) action = 3;
        if (IsKeyDown(KEY_DOWN)) action = 4;
        if (IsKeyPressed(KEY_SPACE)) action = 5;

        if (env.actions) env.actions[0] = action;
        c_step(&env);
        render_blastar(&renderer, &env);
    }

    close_renderer(&renderer);
    close_blastar(&env);
    return 0;
}
