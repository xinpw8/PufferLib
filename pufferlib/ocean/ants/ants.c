// for local testing of c code,build with: 
// bash scripts/build_ocean.sh ants local

#define MAX_ANTS_PER_COLONY 100
#define NUM_COLONIES 2

#include <time.h>
#include "ants.h"
#include "puffernet.h"

int demo() {
    // Initialize environment with proper parameters - FOLLOWING SNAKE PATTERN
    AntsEnv env = {
        .num_ants = NUM_COLONIES * MAX_ANTS_PER_COLONY,
        .width = WINDOW_WIDTH,
        .height = WINDOW_HEIGHT,
        .reward_food = 0.1f,
        .reward_delivery = 1.0f,
        .reward_death = -1.0f,
        .cell_size = 1,
    };
    
    // Allocate memory - CRITICAL: USING PROPER ALLOCATION PATTERN
    allocate_ants_env(&env);
    c_reset(&env);

    // Load trained weights if available
    Weights* weights = load_weights("resources/ants_weights.bin", 266501);
    LinearLSTM* net = NULL;
    if (weights) {
        net = make_linearlstm(weights, env.num_ants, env.obs_size, 4);
    }
    
    printf("Environment initialized. Starting render loop...\n");
    printf("Ants: %d, Observation size: %d\n", env.num_ants, env.obs_size);
    
    // Initialize rendering client
    env.client = make_client(1, env.width, env.height);
    
    // Main loop - FOLLOWING SNAKE PATTERN
    while (!WindowShouldClose()) {
        // User can take control with shift key
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            // Control first ant of colony 1 for demo
            env.actions[0] = ACTION_MOVE_FORWARD;
            if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) env.actions[0] = ACTION_TURN_LEFT;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = ACTION_TURN_RIGHT;
            if (IsKeyDown(KEY_SPACE)) env.actions[0] = ACTION_DROP_PHEROMONE;
            
            // Rest of ants act randomly or via neural network
            for (int i = 1; i < env.num_ants; i++) {
                env.actions[i] = rand() % 4;
            }
        // } else if (net) {
        //     // Use neural network for all ants
        //     forward_linearlstm(net, env.observations, env.actions);
        } else {
            // All ants act randomly
            for (int i = 0; i < env.num_ants; i++) {
                env.actions[i] = rand() % 4;
            }
        }
        
        c_step(&env);
        c_render(&env);
        
        // Print stats periodically
        if (env.tick % 1000 == 0 && env.log.n > 0) {
            printf("Tick %d: Episodes completed: %.0f, Avg score: %.2f, Avg return: %.2f\n",
                   env.tick, env.log.n, env.log.score / env.log.n, env.log.episode_return / env.log.n);
        }
    }
    
    printf("Closing environment...\n");
    
    // Clean up - PROPER CLEANUP FOLLOWING SNAKE PATTERN
    if (net) {
        free_linearlstm(net);
    }
    if (weights) {
        free(weights);
    }
    close_client(env.client);
    free_ants_env(&env);
    
    return 0;
}

void test_performance(float test_time) {
    // Performance test environment
    AntsEnv env = {
        .num_ants = 1024,
        .width = 1280,
        .height = 720,
        .reward_food = 0.1f,
        .reward_delivery = 1.0f,
        .reward_death = -1.0f,
        .cell_size = 1,
    };
    
    allocate_ants_env(&env);
    c_reset(&env);
    
    int start = time(NULL);
    int steps = 0;
    
    while (time(NULL) - start < test_time) {
        // Random actions for performance test
        for (int i = 0; i < env.num_ants; i++) {
            env.actions[i] = rand() % 4;
        }
        
        c_step(&env);
        steps++;
        
        // Print intermediate stats
        if (steps % 1000 == 0 && env.log.n > 0) {
            printf("Step %d: Episodes: %.0f, Avg performance: %.4f\n",
                   steps, env.log.n, env.log.perf / env.log.n);
        }
    }
    
    int end = time(NULL);
    float sps = (float)env.num_ants * steps / (end - start);
    printf("Ant Colony Environment SPS: %.0f\n", sps);
    printf("Total ant steps: %.0f\n", sps);
    printf("Episodes completed: %.0f\n", env.log.n);
    if (env.log.n > 0) {
        printf("Average score: %.2f\n", env.log.score / env.log.n);
        printf("Average performance: %.4f\n", env.log.perf / env.log.n);
    }
    
    // Clean up
    free_ants_env(&env);
}

int main() {
    // Initialize random seed
    srand(time(NULL));
    
    printf("Ant Colony Environment Demo\n");
    printf("Controls:\n");
    printf("- Hold SHIFT to control the first ant\n");
    printf("- A/D or LEFT/RIGHT to turn\n");
    printf("- SPACE to drop pheromone\n");
    printf("- ESC to exit\n\n");
    
    demo();
    
    // Uncomment for performance testing
    // printf("\nRunning performance test...\n");
    // test_performance(10);
    
    return 0;
}