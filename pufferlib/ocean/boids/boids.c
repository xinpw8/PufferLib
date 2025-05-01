// Standalone C demo for Boids environment
// Compile using: ./scripts/build_ocean.sh boids [local|fast]
// Run with: ./boids

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h> // For memset
#include "boids.h" // Include the boids environment header

// --- Demo Configuration ---
#define NUM_BOIDS_DEMO 50   // Number of boids for the standalone demo
#define MAX_STEPS_DEMO 2000 // Max steps per episode in the demo
#define ACTION_SCALE 3.0f   // Corresponds to action space [-3.0, 3.0]

// Dummy action generation: random velocity changes for each boid
void generate_dummy_actions(Boids* env) {
    for (unsigned int i = 0; i < env->num_boids; ++i) {
        // Generate random floats in [-1, 1] range
        float rand_vx = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        float rand_vy = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        
        // Scale to the action space [-ACTION_SCALE, ACTION_SCALE]
        env->actions[i * 2 + 0] = rand_vx * ACTION_SCALE;
        env->actions[i * 2 + 1] = rand_vy * ACTION_SCALE;
    }
}

void demo() {
    // Initialize Boids environment struct
    Boids env = {0}; 
    env.num_boids = NUM_BOIDS_DEMO;
    env.max_steps = MAX_STEPS_DEMO;
    
    // --- Manual Buffer Allocation (for standalone demo only) ---
    // In the Python binding, these pointers are assigned from NumPy arrays.
    // Here, we need to allocate them explicitly.
    size_t obs_size = env.num_boids * 4; // num_boids * (x, y, vx, vy)
    size_t act_size = env.num_boids * 2; // num_boids * (dvx, dvy)
    env.observations = (float*)calloc(obs_size, sizeof(float));
    env.actions = (float*)calloc(act_size, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float)); // Env-level reward
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char)); // Env-level terminal
    env.truncations = (unsigned char*)calloc(1, sizeof(unsigned char)); // Env-level truncation
    
    if (!env.observations || !env.actions || !env.rewards || !env.terminals || !env.truncations) {
        fprintf(stderr, "Failed to allocate memory for demo buffers.\n");
        // Free any successfully allocated buffers before exiting
        free(env.observations); free(env.actions); free(env.rewards); free(env.terminals); free(env.truncations);
        return;
    }
    // -----------------------------------------------------------

    // Initialize Boids C-specific data (boids array, logs)
    init(&env); 
    
    // Initialize rendering client EXPLICITLY before the loop, like Cartpole/Pong
    Client* client = make_client(&env);
    if (client == NULL) {
        fprintf(stderr, "ERROR: Failed to create rendering client during initial setup.\n");
        // Need to free manually allocated buffers and env-specific data before returning
        c_free_env_specific(&env);
        free(env.observations); free(env.actions); free(env.rewards); free(env.terminals); free(env.truncations);
        return; // Exit demo function
    }
    env.client = client; // Assign the created client to the env struct
    
    // Initial reset
    c_reset(&env);

    SetTargetFPS(60);
    int total_steps = 0;
    float total_return = 0.0f;

    printf("Starting Boids demo with %d boids. Press ESC to exit.\n", env.num_boids);

    while (!WindowShouldClose()) { // Raylib function to check if ESC is pressed or window closed
        
        // Generate dummy actions
        generate_dummy_actions(&env);

        // Step the environment
        c_step(&env);
        
        total_return += env.rewards[0]; // Accumulate env-level reward
        total_steps++;

        // Render the current state
        c_render(&env);

        // Check for termination or truncation
        if (env.terminals[0] || env.truncations[0]) {
            printf("Episode finished. Steps: %d, Return: %.2f (%s)\n", 
                   env.tick, total_return, env.terminals[0] ? "Terminated" : "Truncated");
            total_steps = 0;
            total_return = 0.0f;
            
            // Reset the environment for the next episode
            c_reset(&env);
        }
    }

    // Cleanup
    // Use the 'client' variable created before the loop
    if (client) { // Check the local variable, not env.client which might be modified
        c_close_client(client);
    }
    // env.client = NULL; // No longer necessary as client is local scope to demo
    
    c_free_env_specific(&env); // Free Boids-specific C memory (boids array, logs)
    
    // --- Free Manually Allocated Buffers ---
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    free(env.truncations);
    // ----------------------------------------
}

int main() {
    srand(time(NULL)); // Seed random number generator
    demo();
    return 0;
}