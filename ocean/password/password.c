#include "password.h"

static int key_to_action(void) {
    if (IsKeyPressed(KEY_ONE) || IsKeyPressed(KEY_KP_1)) return 0;
    if (IsKeyPressed(KEY_TWO) || IsKeyPressed(KEY_KP_2)) return 1;
    if (IsKeyPressed(KEY_THREE) || IsKeyPressed(KEY_KP_3)) return 2;
    if (IsKeyPressed(KEY_FOUR) || IsKeyPressed(KEY_KP_4)) return 3;
    if (IsKeyPressed(KEY_FIVE) || IsKeyPressed(KEY_KP_5)) return 4;
    if (IsKeyPressed(KEY_SIX) || IsKeyPressed(KEY_KP_6)) return 5;
    if (IsKeyPressed(KEY_SEVEN) || IsKeyPressed(KEY_KP_7)) return 6;
    if (IsKeyPressed(KEY_EIGHT) || IsKeyPressed(KEY_KP_8)) return 7;
    if (IsKeyPressed(KEY_NINE) || IsKeyPressed(KEY_KP_9)) return 8;
    return -1;
}

void demo(void) {
    Password env;
    memset(&env, 0, sizeof(Password));

    env.observations = (unsigned char*)calloc(LENGTH, sizeof(unsigned char));
    env.actions = (float*)calloc(1, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (float*)calloc(1, sizeof(float));
    env.num_agents = 1;

    init(&env);
    c_reset(&env);

    while (!WindowShouldClose()) {
        int action = key_to_action();
        if (action >= 0) {
            env.actions[0] = (float)action;
            c_step(&env);
        }
        c_render(&env);
    }

    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}

int main(void) {
    demo();
    return 0;
}
