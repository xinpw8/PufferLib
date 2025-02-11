#include <time.h>
#include <unistd.h>
#include "tower_climb.h"

void demo() {    
    CTowerClimb* env = allocate();
    int seed = 0;
    init_random_level(env, 5, 25, seed);

    Client* client = make_client(env);

    int tick = 0;
    while (!WindowShouldClose()) {
        // Camera controls
        env->actions[0] = NOOP;
        if (IsKeyPressed(KEY_UP) || IsKeyPressed(KEY_W)) {
            env->actions[0] = UP;
        }
        if (IsKeyPressed(KEY_LEFT) || IsKeyPressed(KEY_A)) {
            env->actions[0] = LEFT;
        }
        if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_D)) {
            env->actions[0] = RIGHT;
        }
        if (IsKeyPressed(KEY_DOWN) || IsKeyPressed(KEY_S)){
            env->actions[0] = DOWN;
        }
        if (IsKeyPressed(KEY_SPACE)){
            env->actions[0] = GRAB;
        }
        if (IsKeyPressed(KEY_LEFT_SHIFT)){
            env->actions[0] = DROP;
        }
        render(client, env);
        int done = 0;
        tick = (tick + 1)%12;

        if (tick % 12 == 0) {
            done = step(env);
        }
        if (done) {
            printf("Done, reward: %f\n", env->rewards[0]);
            seed++;
            c_reset(env);
            init_random_level(env, 5, 25, seed);
        }
    }
    close_client(client);
    free_allocated(env);
}

void performance_test() {
    long test_time = 10;
    CTowerClimb* env = allocate();
    int seed = 0;
    init_random_level(env, 8, 25, seed);


    long start = time(NULL);
    int i = 0;
    while (time(NULL) - start < test_time) {
        env->actions[0] = rand() % 5;
        int done = 0;
        done = step(env);
        if (done) {
            seed++;
            c_reset(env);
            init_random_level(env, 8, 25, seed);
        }
        i++;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", i / (end - start));
    free_allocated(env);
}


int main() {
    demo();
    // performance_test();
    return 0;
}


