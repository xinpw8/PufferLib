#include <time.h>
#include <unistd.h>
#include "tower_climb.h"

void PLG(){
    init_random_levels(4);
}

void demo() {    
    CTowerClimb  env = {
        .reward_climb_row = 0.25,
        .reward_fall_row = -0.1,
        .reward_illegal_move = -0.01,
    };
    init_random_levels(8);

    allocate(&env);
    reset(&env);
    Client* client = make_client(&env);

    int tick = 0;
    while (!WindowShouldClose()) {
        // Camera controls
        env.actions[0] = NOOP;
        if (IsKeyPressed(KEY_UP) || IsKeyPressed(KEY_W)) {
            env.actions[0] = UP;
        }
        if (IsKeyPressed(KEY_LEFT) || IsKeyPressed(KEY_A)) {
            env.actions[0] = LEFT;
        }
        if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_D)) {
            env.actions[0] = RIGHT;
        }
        if (IsKeyPressed(KEY_DOWN) || IsKeyPressed(KEY_S)){
            env.actions[0] = DOWN;
        }
        if (IsKeyPressed(KEY_SPACE)){
            env.actions[0] = GRAB;
        }
        if (IsKeyPressed(KEY_LEFT_SHIFT)){
            env.actions[0] = DROP;
        }
        render(client, &env);
        step(&env);
    }
    close_client(client);
    free_allocated(&env);
}

void performance_test() {
    long test_time = 10;
    CTowerClimb env = {
        .map_choice = 1,
        
    };
    init_random_levels(5);

    allocate(&env);
    reset(&env);

    long start = time(NULL);
    int i = 0;
    while (time(NULL) - start < test_time) {
        env.actions[0] = rand() % 5;
        step(&env);
        i++;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", i / (end - start));
    free_allocated(&env);
}


int main() {
    // demo();
    performance_test();
    // PLG();
    return 0;
}


