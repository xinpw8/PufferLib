#include <time.h>
#include "boulder.h"
#include "puffernet.h"

void demo() {
    Boulder env = {
        .width      = 900,
        .height     = 600,
        .num_agents = 2,
    };
    allocate(&env);

    env.client = make_client(&env);

    c_reset(&env);
    int frame = 0;
    SetTargetFPS(60);
    while (!WindowShouldClose()) {
        int kl = IsKeyDown(KEY_LEFT), kr = IsKeyDown(KEY_RIGHT);
        int ku = IsKeyDown(KEY_UP),   kd = IsKeyDown(KEY_DOWN);
        env.actions[0] = (kl || kr || ku || kd) ? 1 : 0;  // throttle
        int dir = 0;
        if (kr && !ku && !kd) dir = 0;  // E
        if (kr &&  ku)        dir = 1;  // NE
        if (ku && !kl && !kr) dir = 2;  // N
        if (kl &&  ku)        dir = 3;  // NW
        if (kl && !ku && !kd) dir = 4;  // W
        if (kl &&  kd)        dir = 5;  // SW
        if (kd && !kl && !kr) dir = 6;  // S
        if (kr &&  kd)        dir = 7;  // SE
        env.actions[1] = dir;

        c_step(&env);
        c_render(&env);
    }
    //free_puffernet(net);
    //free(weights);
    free_allocated(&env);
    close_client(env.client);
}

int main() {
    demo();
}

