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
        .num_agents = 4,
        .num_requested_shelves = 4,
        .grid_square_size = 64,
        .human_agent_idx = 0,
	.reward_type = 2
    };
    allocate(&env);
    reset(&env);
    Client* client = make_client(&env);

    while (!WindowShouldClose()) {
        for (int i = 0; i < env.num_agents; i++) {
            env.actions[i] = rand() % 5;
        }

        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[env.human_agent_idx] = NOOP;

            // Handle keyboard input only for selected agent
            if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) {
                env.actions[env.human_agent_idx] = FORWARD;
            }
            if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
                env.actions[env.human_agent_idx] = LEFT;
            }
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
                env.actions[env.human_agent_idx] = RIGHT;
            }
            if (IsKeyDown(KEY_SPACE) || IsKeyDown(KEY_ENTER)) {
                env.actions[env.human_agent_idx] = TOGGLE_LOAD;
            }
            // Add agent switching with TAB key
            if (IsKeyDown(KEY_TAB)) {
                env.human_agent_idx = (env.human_agent_idx + 1) % env.num_agents;
            }
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
        .num_requested_shelves = 4,
	.reward_type = 2
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
    demo(2);
    //performance_test();
    return 0;
}
