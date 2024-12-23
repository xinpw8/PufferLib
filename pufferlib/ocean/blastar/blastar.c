// #include "blastar_env.h"
// #include "blastar_renderer.h"
// #include "puffernet.h"
// #include <assert.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <time.h>

// #define OBSERVATIONS_SIZE 27
// #define ACTIONS_SIZE 6

// void get_input(BlastarEnv* env) {
//     if ((IsKeyDown(KEY_DOWN) && IsKeyDown(KEY_RIGHT))) {
//         env->actions[0] = 4; // Move down-right
//     } else if ((IsKeyDown(KEY_DOWN) && IsKeyDown(KEY_LEFT))) {
//         env->actions[0] = 5; // Move down-left
//     } else if (IsKeyDown(KEY_SPACE) && (IsKeyDown(KEY_RIGHT))) {
//         env->actions[0] = 6; // Fire and move right
//     } else if (IsKeyDown(KEY_SPACE) && (IsKeyDown(KEY_LEFT))) {
//         env->actions[0] = 7; // Fire and move left
//     } else if (IsKeyDown(KEY_SPACE)) {
//         env->actions[0] = 3; // Fire
//     } else if (IsKeyDown(KEY_DOWN)) {
//         env->actions[0] = 2; // Move down
//     } else if (IsKeyDown(KEY_LEFT)) {
//         env->actions[0] = 1; // Move left
//     } else if (IsKeyDown(KEY_RIGHT)) {
//         env->actions[0] = 0; // Move right
//     } else {
//         env->actions[0] = 0; // No action
//     }
// }

// int demo() {
//     // Load weights for the AI model
//     Weights* weights = load_weights("resources/blastar/blastar_2.bin", 136583);
//     LinearLSTM* net = make_linearlstm(weights, 1, OBSERVATIONS_SIZE, ACTIONS_SIZE);

//     BlastarEnv env;
//     init_blastar(&env);
//     allocate_env(&env);

//     Client* client = make_client(&env);

//     unsigned int seed = 12345;
//     srand(seed);
//     reset_blastar(&env);

//     int running = 1;
//     while (running) {
//         if (IsKeyDown(KEY_LEFT_SHIFT)) {
//             get_input(&env); // Human input
//         } else {
//             forward_linearlstm(net, env.observations, env.actions); // AI input
//         }

//         c_step(&env);
//         c_render(client, &env);

//         if (WindowShouldClose() || env.game_over) {
//             running = 0;
//         }
//     }

//     free_linearlstm(net);
//     free(weights);
//     close_client(client);
//     free_allocated_env(&env);
//     close_blastar(&env);
//     return 0;
// }

// void perftest(float test_time) {
//     BlastarEnv env;
//     init_blastar(&env);
//     allocate_env(&env);

//     unsigned int seed = 12345;
//     srand(seed);
//     reset_blastar(&env);

//     int start = time(NULL);
//     int steps = 0;
//     while (time(NULL) - start < test_time) {
//         env.actions[0] = rand() % ACTIONS_SIZE; // Random actions
//         c_step(&env);
//         steps++;
//     }

//     int end = time(NULL);
//     printf("Steps per second: %f\n", steps / (float)(end - start));

//     free_allocated_env(&env);
//     close_blastar(&env);
// }

// int main() {
//     demo();
//     // perftest(30.0f);
//     return 0;
// }






#include "blastar_env.h"
#include "blastar_renderer.h"
#include <assert.h>

int main() {
    BlastarEnv env;
    init_blastar(&env);

    allocate_env(&env); 

    Client* client = make_client(&env);

    while (!WindowShouldClose() && !env.game_over) {
        int action = 0;
        if (IsKeyDown(KEY_LEFT)) action = 1;
        if (IsKeyDown(KEY_RIGHT)) action = 2;
        if (IsKeyDown(KEY_UP)) action = 3;
        if (IsKeyDown(KEY_DOWN)) action = 4;
        if (IsKeyPressed(KEY_SPACE)) action = 5;

        if (env.actions) env.actions[0] = action;
        c_step(&env);
        c_render(client, &env);
    }

    close_client(client);
    free_allocated_env(&env);
    close_blastar(&env);
    return 0;
}
