// puffer_enduro.c

# define MAX_ENEMIES 10

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>
#include "enduro.h"
#include "raylib.h"
#include "puffernet.h"

int demo() {
    Weights* weights = load_weights("resources/puffer_enduro/enduro_weights.bin", 140170);
    LinearLSTM* net = make_linearlstm(weights, 1, 52, 9);

    Enduro env;

    env.num_envs = 1;
    env.max_enemies = MAX_ENEMIES;
    env.obs_size = OBSERVATIONS_MAX_SIZE;

    allocate(&env);

    Client* client = make_client(&env);

    unsigned int seed = 0;
    init(&env, seed, 0);
    reset(&env);

    loadTextures(&client->gameState);

    int running = 1;
    while (running) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[0] = 0;
            int fps = GetFPS();
            printf("FPS player: %d\n", fps);
            handleEvents(&running, &env);
        } else {
            forward_linearlstm(net, env.observations, env.actions);
            int fps = GetFPS();
            printf("FPS agent: %d\n", fps);
        }

        c_step(&env);
        c_render(client, &env);

        if (WindowShouldClose()) {
            running = 0;
        }
    }
    free_linearlstm(net);
    free(weights);
    close_client(client, &env);
    cleanup(&client->gameState);
    free_allocated(&env);
    return 0;
}

// int demo() {
//     Weights* weights = load_weights("resources/puffer_enduro/enduro_weights.bin", 140170);
//     LinearLSTM* net = make_linearlstm(weights, 1, 52, 9);

//     Enduro env;

//     env.num_envs = 1;
//     env.max_enemies = MAX_ENEMIES;
//     env.obs_size = OBSERVATIONS_MAX_SIZE;

//     allocate(&env);

//     Client* client = make_client(&env);

//     unsigned int seed = 0;
//     init(&env, seed, 0);
//     reset(&env);

//     loadTextures(&client->gameState);

//     const float targetFrameTime = 1.0f / 60.0f; // Target 60 FPS
//     float accumulator = 0.0f;

//     int running = 1;
//     int playerControl = 0; // 0 = AI, 1 = Player

//     while (running) {
//         float frameTime = GetFrameTime();
//         accumulator += frameTime;

//         // Toggle control mode on SHIFT press
//         if (IsKeyPressed(KEY_LEFT_SHIFT)) {
//             playerControl = !playerControl; // Toggle between 0 (AI) and 1 (Player)
//             printf("Control mode: %s\n", playerControl ? "Player" : "AI");
//         }

//         // Handle player input outside fixed time step loop
//         if (playerControl) {
//             // handleEvents(&running, &env);
//             float start = GetTime();
//             handleEvents(&running, &env);
//             float end = GetTime();
//             printf("Time taken for handleEvents: %.4f ms\n", (end - start) * 1000);
//             float fps = GetFPS();
//             printf("FPS player: %.2f\n", fps);

//         }

//         // Fixed time step for game logic
//         while (accumulator >= targetFrameTime) {
//             if (!playerControl) {
//                 forward_linearlstm(net, env.observations, env.actions); // AI control
//                 float fps = GetFPS();
//                 printf("FPS agent: %.2f\n", fps);
//             }

//             c_step(&env); // Update game logic
//             accumulator -= targetFrameTime;
//         }

//         c_render(client, &env); // Render the frame

//         if (WindowShouldClose()) {
//             running = 0;
//         }
//     }

//     free_linearlstm(net);
//     free(weights);
//     close_client(client, &env);
//     cleanup(&client->gameState);
//     free_allocated(&env);
//     return 0;
// }

void perftest(float test_time) {
    Enduro env;

    env.num_envs = 1;
    env.max_enemies = MAX_ENEMIES;
    env.obs_size = OBSERVATIONS_MAX_SIZE;

    allocate(&env);

    unsigned int seed = 12345;
    init(&env, seed, 0);
    reset(&env);

    int start = time(NULL);
    int i = 0;
    int running = 1;
    while (time(NULL) - start < test_time) {
        handleEvents(&running, &env);
        env.actions[0] = rand()%9;
        c_step(&env);
        i++;
    }

    int end = time(NULL);
    printf("SPS: %f\n", i / (float)(end - start));
    free_allocated(&env);
}


int main() {
   demo();
//    perftest(30.0f);
   return 0;
}

