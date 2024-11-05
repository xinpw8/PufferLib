#include <time.h>
#include "rware.h"

void demo(int map_choice) {
    CRware env = {
        .width = 704,
        .height = 450,
        .map_choice = map_choice,
        .num_agents = 2,
        .num_requested_shelves = 2
    };
    allocate(&env);
    reset(&env);
 
    Client* client = make_client(env.width, env.height);

    while (!WindowShouldClose()) {
        // User can take control of the agent
        env.actions[0] = 0;
        /* actions using arrow keys to turn left and turn right and space to pickup and drop shelf. forward arrow to move forward*/
        if (IsKeyPressed(KEY_UP)) {
            env.actions[0] = 1;
        }
        if (IsKeyPressed(KEY_LEFT)) {
            env.actions[0] = 2;
        }
        if (IsKeyPressed(KEY_RIGHT)) {
            env.actions[0] = 3;
        }
        if (IsKeyPressed(KEY_SPACE)) {
            env.actions[0] = 4;
        }
        step(&env);
        render(client,&env);
    }
    close_client(client);
    free_allocated(&env);
}

void performance_test() {
    long test_time = 10;
    CRware env = {
        .width = 1000,
        .height = 800,
        .map_choice = 1,
        .num_agents = 2,
        .num_requested_shelves = 2
    };
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
    // demo(1);
    performance_test();
    return 0;
}
