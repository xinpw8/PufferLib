#include <time.h>
#include "trash_pickup.h"

#define INDEX_C(env, x, y) ((y) * (env).grid_size + (x))

// Demo function for visualizing the TrashPickupEnv


void demo(int grid_size, int num_agents, int num_trash, int num_bins) {
    CTrashPickupEnv env;
    env.grid_size = grid_size;
    env.num_agents = num_agents;
    env.num_trash = num_trash;
    env.num_bins = num_bins;

    allocate(&env);

    Client* client = make_client(&env);

    reset(&env);

    while (!WindowShouldClose()) {
        // Random actions for all agents
        for (int i = 0; i < env.num_agents; i++) {
            env.actions[i] = rand() % 4; // 0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT
        }

        // Step the environment and render the grid
        step(&env);
        render(client, &env);
    }

    free_allocated(&env);
    close_client(client);
}



// Performance test function for benchmarking
void performance_test() {
    long test_time = 10; // Test duration in seconds

    CTrashPickupEnv env;
    env.grid_size = 10;
    env.num_agents = 3;
    env.num_trash = 15;
    env.num_bins = 3;
    env.max_steps = 300;
    env.current_step = 0;

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


// Main entry point
int main() {
    // demo(10, 3, 15, 1); // Visual demo
    performance_test(); // Uncomment for benchmarking
    return 0;
}