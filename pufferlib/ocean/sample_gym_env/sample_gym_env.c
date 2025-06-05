/* sample_gym_env.c
 * Pure C demo file for SampleGymEnv
 * This is ONLY for local compilation and rendering/testing
 * Build it with: gcc -o sample_gym sample_gym_env.c -lraylib -lm
 */

#include "sample_gym_env.h"
#include "puffernet.h"
#include <time.h>
#include <stdlib.h>

int main() {
    Weights* weights = load_weights("resources/sample_gym_env_weights.bin", 279940);
    LinearLSTM* net = make_linearlstm(weights, 1, 121, 5);

    srand(time(NULL)); // Initialize random seed
    
    // Initialize environment with 11x11 grid
    SampleGymEnv env = {.size = 11};
    
    // Allocate memory for environment buffers
    int total_cells = env.size * env.size;
    env.observations = (unsigned char*)calloc(total_cells, sizeof(unsigned char));
    env.actions = (int*)calloc(1, sizeof(int));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    
    // Reset environment to initial state
    c_reset(&env);
    
    printf("SampleGymEnv Demo Started!\n");
    printf("Controls:\n");
    printf("  Hold SHIFT + Arrow keys/WASD for manual control\n");
    printf("  Release SHIFT for random actions\n");
    printf("  ESC to exit\n\n");
    
    // Game loop
    c_render(&env);
    while (!WindowShouldClose()) {
        // Input handling
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            // Manual control mode
            env.actions[0] = NOOP;
            if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env.actions[0] = UP;
            if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env.actions[0] = DOWN;
            if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[0] = LEFT;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = RIGHT;
        } else {
            // Convert unsigned char observations to float buffer
            for (int i = 0; i < total_cells; ++i) {
                net->obs[i] = (float)env.observations[i];
            }
            // Run the policy
            forward_linearlstm(net, net->obs, env.actions);
        }
        
        // Step environment
        c_step(&env);
        
        // Print reward info if significant
        if (env.rewards[0] > 1.0f) {
            printf("Item collected! Reward: %.1f\n", env.rewards[0]);
        } else if (env.rewards[0] < -1.0f) {
            printf("Episode ended. Final reward: %.1f\n", env.rewards[0]);
        }
        
        // Render
        c_render(&env);
        
        // Small delay for better visibility in random mode
        if (!IsKeyDown(KEY_LEFT_SHIFT)) {
            WaitTime(0.1);
        }
    }
    
    // Cleanup
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    
    printf("Demo ended. Final stats:\n");
    printf("  Performance: %.2f\n", env.log.perf);
    printf("  Total score: %.2f\n", env.log.score);
    printf("  Episodes completed: %.0f\n", env.log.n);
    
    return 0;
}