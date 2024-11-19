#include <time.h>
#include "trash_pickup.h"

// Demo function for visualizing the TrashPickupEnv
/*
void demo(int grid_size, int num_agents, int num_trash, int num_bins) {
    int width, height;

    CTrashPickupEnv env;
    initialize_env(&env, grid_size, num_agents, num_trash, num_bins, 300);

    InitWindow(width, height, "Trash Pickup Demo");
    SetTargetFPS(10);

    while (!WindowShouldClose()) {
        // Random actions for all agents
        for (int i = 0; i < env.num_agents; i++) {
            env.actions[i] = rand() % 4; // 0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT
        }

        // Interactive controls for agent 0
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[0] = -1; // NOOP for agent 0
            if (IsKeyDown(KEY_UP)) env.actions[0] = ACTION_UP;
            if (IsKeyDown(KEY_DOWN)) env.actions[0] = ACTION_DOWN;
            if (IsKeyDown(KEY_LEFT)) env.actions[0] = ACTION_LEFT;
            if (IsKeyDown(KEY_RIGHT)) env.actions[0] = ACTION_RIGHT;
        }

        // Step the environment and render the grid
        step_env(&env);

        BeginDrawing();
        ClearBackground(RAYWHITE);

        for (int x = 0; x < env.grid_size; x++) {
            for (int y = 0; y < env.grid_size; y++) {
                int cell = env->grid[INDEX(env, x, y)];
                Color cell_color;

                if (cell == EMPTY) {
                    cell_color = LIGHTGRAY;
                } else if (cell == TRASH) {
                    cell_color = RED;
                } else if (cell == TRASH_BIN) {
                    cell_color = GREEN;
                } else if (cell == AGENT) {
                    cell_color = BLUE;
                }

                DrawRectangle(
                    x * (width / env.grid_size), 
                    y * (height / env.grid_size),
                    (width / env.grid_size),
                    (height / env.grid_size),
                    cell_color
                );
            }
        }

        EndDrawing();
    }

    CloseWindow();
    free_env(&env);
}
*/


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
    // demo(10, 4, 10, 2); // Visual demo
    performance_test(); // Uncomment for benchmarking
    return 0;
}