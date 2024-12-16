#include "blastar_env.h"
#include "blastar_renderer.h"
#include <assert.h>

int main() {
    BlastarEnv env;
    init_blastar(&env);

    allocate_env(&env); 

    // debug
    assert(env.observations != NULL);
    assert(env.actions != NULL);
    assert(env.rewards != NULL);
    assert(env.terminals != NULL);

    Client* client = make_client(&env);

    while (!WindowShouldClose() && !env.game_over) {
        int action = 0;
        if (IsKeyDown(KEY_LEFT)) action = 1;
        if (IsKeyDown(KEY_RIGHT)) action = 2;
        if (IsKeyDown(KEY_UP)) action = 3;
        if (IsKeyDown(KEY_DOWN)) action = 4;
        if (IsKeyPressed(KEY_SPACE)) action = 5;

        if (env.actions) env.actions[0] = action;
        c_step(&env);
        c_render(client, &env);  // Use the correct render function
    }

    close_client(client);
    free_allocated_env(&env);
    close_blastar(&env);
    return 0;
}
