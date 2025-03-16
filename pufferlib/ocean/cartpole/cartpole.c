// #include <time.h>
// #include "cartpole.h"

// // Stripped-down demo function that bypasses all neural network components
// void demo() {
//     // Initialize environment with standard parameters
//     CartPole env = {
//         .frameskip = 1,
//         .width = 800,
//         .height = 600,
//         .max_steps = 200,
//         .continuous = 0,
//     };
    
//     // Initialize environment memory structures
//     allocate(&env);
//     c_reset(&env);
 
//     // Initialize rendering window directly
//     InitWindow(800, 600, "CartPole Manual Control");
//     SetTargetFPS(60);

//     // Main simulation loop
//     while (!WindowShouldClose()) {
//         // Manual control input handling
//         env.actions[0] = 0.0;
//         if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) env.actions[0] = 0;
//         if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = 1;
        
//         // Update physics
//         c_step(&env);
        
//         // Render current state
//         BeginDrawing();
//         ClearBackground((Color){230, 230, 230, 255});
        
//         // Draw track
//         DrawLine(0, 300, 800, 300, BLACK);
        
//         // Calculate cart position in pixels
//         float cart_x = 400 + env.x * 100;
//         float cart_y = 300;
        
//         // Draw cart
//         DrawRectangle(cart_x - 20, cart_y - 10, 40, 20, BLACK);
        
//         // Draw pole
//         float pole_x2 = cart_x + sinf(env.theta) * 100;
//         float pole_y2 = cart_y - cosf(env.theta) * 100;
//         DrawLineEx((Vector2){cart_x, cart_y}, (Vector2){pole_x2, pole_y2}, 5, RED);
        
//         // Draw info
//         DrawText(TextFormat("Steps: %i", env.steps), 10, 10, 20, BLACK);
//         DrawText(TextFormat("Cart Position: %.2f", env.x), 10, 40, 20, BLACK);
//         DrawText(TextFormat("Pole Angle: %.2f°", env.theta * 180 / M_PI), 10, 70, 20, BLACK);
//         DrawText("Use LEFT/RIGHT arrow keys to balance", 10, 540, 20, DARKGRAY);
        
//         EndDrawing();
//     }
    
//     // Clean up resources
//     CloseWindow();
//     free_allocated(&env);
// }

// void performance_test() {
//     long test_time = 10;
//     CartPole env = {
//         .frameskip = 1,
//         .width = 800,
//         .height = 600,
//         .max_steps = 200,
//     };
//     allocate(&env);
//     c_reset(&env);

//     long start = time(NULL);
//     int i = 0;
//     while (time(NULL) - start < test_time) {
//         env.actions[0] = rand() % 2;  // CartPole has 2 actions (0 or 1)
//         c_step(&env);
//         i++;
//     }
//     long end = time(NULL);
//     printf("SPS: %ld\n", i / (end - start));
//     free_initialized(&env);
// }

// int main() {
//     //performance_test();
//     demo();
//     return 0;
// }

#include <stdio.h>
#include <stdlib.h>
#include "cartpole.h"

typedef struct Policy {
    // Define policy structure
    float* weights; // Adjust based on actual policy details
    int num_weights;
} Policy;

Policy* load_policy(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        perror("Could not load weights");
        exit(1);
    }
    // Allocate and load your policy weights
    Policy* p = malloc(sizeof(Policy));
    p->weights = malloc(NUM_WEIGHTS * sizeof(float)); // NUM_WEIGHTS: replace with actual number
    fread(p->weights, sizeof(float), NUM_WEIGHTS, f);
    fclose(f);
    return p;
}

int policy_forward(Policy* policy, float* observations) {
    // Dummy inference: just for illustration purposes, always right (1)
    // Replace this with actual inference using loaded weights
    return (observations[2] > 0) ? 1 : 0;
}

void free_policy(Policy* policy) {
    free(policy->weights);
    free(policy);
}

int main() {
    CartPole env = {
        .frameskip = 1,
        .width = 800,
        .height = 600,
        .max_steps = 200,
        .continuous = 0,
    };
    
    allocate(&env);
    c_reset(&env);

    // Declare client correctly
    Client* client = make_client(&env);

    // Load your policy weights
    Policy* policy = load_policy("weights.bin"); // weights.bin is your saved weights file
    
    while (!WindowShouldClose()) {
        float* obs = env.observations;
        int action = policy_forward(policy, obs);
        env.actions[0] = action;

        c_step(&env);
        c_render(client, &env);

        if (env.dones[0]) {
            c_reset(&env);
        }
    }

    free_policy(policy);
    close_client(client);
    free_initialized(&env);
    return 0;
}
