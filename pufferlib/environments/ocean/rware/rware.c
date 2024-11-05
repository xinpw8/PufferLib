#include <time.h>
#include "rware.h"

#define MAP_TINY_WIDTH 640
#define MAP_TINY_HEIGHT 704
#define MAP_SMALL_WIDTH 1280
#define MAP_SMALL_HEIGHT 640
#define MAP_MEDIUM_WIDTH 1280
#define MAP_MEDIUM_HEIGHT 1024

void demo(int map_choice) {
    int width;
    int height;
    if (map_choice == 1) {
        width = MAP_TINY_WIDTH;
        height = MAP_TINY_HEIGHT;
    } else if (map_choice == 2) {
        width = MAP_SMALL_WIDTH;
        height = MAP_SMALL_HEIGHT;
    } else {
        width = MAP_MEDIUM_WIDTH;
        height = MAP_MEDIUM_HEIGHT;
    }
    CRware env = {
        .width = width,
        .height = height,
        .map_choice = map_choice,
        .num_agents = 2,
        .num_requested_shelves = 2,
        .grid_square_size = 64,
        .human_agent_idx = 0
    };
    allocate(&env);
    reset(&env);
 
    Client* client = make_client(env.width, env.height);

    while (!WindowShouldClose()) {
        // User can take control of the agent
        for (int i = 0; i < env.num_agents; i++) {
            env.actions[i] = NOOP;
        }

        // Handle keyboard input only for selected agent
        if (IsKeyPressed(KEY_UP)) {
            env.actions[env.human_agent_idx] = FORWARD;
        }
        if (IsKeyPressed(KEY_LEFT)) {
            env.actions[env.human_agent_idx] = LEFT;
        }
        if (IsKeyPressed(KEY_RIGHT)) {
            env.actions[env.human_agent_idx] = RIGHT;
        }
        if (IsKeyPressed(KEY_SPACE)) {
            env.actions[env.human_agent_idx] = TOGGLE_LOAD;
        }
        // Add agent switching with TAB key
        if (IsKeyPressed(KEY_TAB)) {
            env.actions[env.human_agent_idx] = TOGGLE_AGENT;
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
        .width = 1280,
        .height = 704,
        .map_choice = 2,
        .num_agents = 4,
        .num_requested_shelves = 2
    };
    allocate(&env);
    reset(&env);

    long start = time(NULL);
    int i = 0;
    while (time(NULL) - start < test_time) {
        env.actions[0] = rand() % 6;
        step(&env);
        i++;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", i / (end - start));
    free_allocated(&env);
}

int main() {
    demo(3);
    // performance_test();
    return 0;
}
