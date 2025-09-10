/* Main file for testing the drone pick and place environment */

#include "drone_pickplace.h"
#include <stdio.h>
#include <time.h>

int main(int argc, char* argv[]) {
    srand(time(NULL));
    
    printf("Drone Pick & Place Environment Demo\n");
    printf("====================================\n");
    printf("Controls:\n");
    printf("  WASD - Move Forward/Back/Left/Right\n");
    printf("  Q/E - Move Up/Down\n");
    printf("  Z/C - Rotate Left/Right\n");
    printf("  SPACE - Close Gripper\n");
    printf("  R - Open Gripper\n");
    printf("  ESC - Exit\n\n");
    
    // Create environment
    DronePickPlace env = {0};
    env.num_drones = 1;
    env.num_objects = 3;
    env.num_targets = 2;
    env.world_size = 2.0f;
    env.max_height = 1.5f;
    env.max_steps = 1000;
    env.debug_mode = 1;  // Enable debug logging for standalone
    
    // Allocate observation, action, reward, and terminal buffers
    int obs_per_drone = 45;  // Fixed observation size
    env.observations = calloc(env.num_drones * obs_per_drone, sizeof(float));
    env.actions = calloc(env.num_drones, sizeof(int));
    env.rewards = calloc(env.num_drones, sizeof(float));
    env.terminals = calloc(env.num_drones, sizeof(unsigned char));
    
    // Initialize environment
    init(&env);
    c_reset(&env);
    
    // Run simulation
    int total_steps = 0;
    int episodes = 0;
    float total_reward = 0;
    
    while (total_steps < 10000) {
        // Keyboard controls for manual testing
        int action = -1;
        
        if (IsKeyDown(KEY_W)) action = 0;  // MOVE_FORWARD
        else if (IsKeyDown(KEY_S)) action = 1;  // MOVE_BACKWARD
        else if (IsKeyDown(KEY_A)) action = 2;  // MOVE_LEFT
        else if (IsKeyDown(KEY_D)) action = 3;  // MOVE_RIGHT
        else if (IsKeyDown(KEY_Q)) action = 4;  // MOVE_UP
        else if (IsKeyDown(KEY_E)) action = 5;  // MOVE_DOWN
        else if (IsKeyDown(KEY_Z)) action = 6;  // ROTATE_LEFT
        else if (IsKeyDown(KEY_C)) action = 7;  // ROTATE_RIGHT
        else if (IsKeyDown(KEY_SPACE)) action = 9;  // GRIPPER_CLOSE
        else if (IsKeyDown(KEY_R)) action = 8;  // GRIPPER_OPEN
        
        // Use random action if no key pressed
        if (action == -1) {
            action = rand() % 10;
        }
        
        for (int i = 0; i < env.num_drones; i++) {
            env.actions[i] = action;
        }
        
        c_step(&env);
        c_render(&env);
        
        // Accumulate rewards
        for (int i = 0; i < env.num_drones; i++) {
            total_reward += env.rewards[i];
        }
        
        // Check for episode end
        if (env.terminals[0]) {
            episodes++;
            printf("Episode %d completed. Steps: %d, Total Reward: %.2f\n", 
                   episodes, env.current_step, total_reward);
            total_reward = 0;
        }
        
        total_steps++;
        
        // Exit on ESC
        if (IsKeyDown(KEY_ESCAPE)) {
            break;
        }
    }
    
    // Print final statistics
    printf("\n====================================\n");
    printf("Simulation Complete!\n");
    printf("Total Episodes: %d\n", episodes);
    printf("Total Steps: %d\n", total_steps);
    if (env.log.n > 0) {
        printf("Average Performance: %.2f\n", env.log.perf / env.log.n);
        printf("Average Score: %.2f\n", env.log.score / env.log.n);
        printf("Grasp Success Rate: %.2f%%\n", 
               (env.log.grasp_success / (env.log.n * env.num_objects)) * 100);
        printf("Placement Success Rate: %.2f%%\n", 
               (env.log.placement_success / (env.log.n * env.num_objects)) * 100);
    }
    
    // Cleanup
    c_close(&env);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    
    return 0;
}