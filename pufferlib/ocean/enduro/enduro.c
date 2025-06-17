// puffer_enduro.c

#define MAX_ENEMIES 10

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>
#include "enduro.h"
#include "raylib.h"
#include "puffernet.h"
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

void get_input(Enduro *env) {
    if ((IsKeyDown(KEY_DOWN) && IsKeyDown(KEY_RIGHT)) ||
        (IsKeyDown(KEY_S)    && IsKeyDown(KEY_D)))
        env->actions[0] = ACTION_DOWNRIGHT;
    else if ((IsKeyDown(KEY_DOWN) && IsKeyDown(KEY_LEFT)) ||
             (IsKeyDown(KEY_S)    && IsKeyDown(KEY_A)))
        env->actions[0] = ACTION_DOWNLEFT;
    else if (IsKeyDown(KEY_SPACE) &&
             (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)))
        env->actions[0] = ACTION_RIGHTFIRE;
    else if (IsKeyDown(KEY_SPACE) &&
             (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)))
        env->actions[0] = ACTION_LEFTFIRE;
    else if (IsKeyDown(KEY_SPACE))
        env->actions[0] = ACTION_FIRE;
    else if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S))
        env->actions[0] = ACTION_DOWN;
    else if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A))
        env->actions[0] = ACTION_LEFT;
    else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D))
        env->actions[0] = ACTION_RIGHT;
    else
        env->actions[0] = ACTION_NOOP;
}

#ifdef __EMSCRIPTEN__

void get_input(Enduro *env);

static Enduro* e = NULL;
static LinearLSTM* n = NULL;

static void loop(void *unused) {
    if (IsKeyDown(KEY_LEFT_SHIFT))
        get_input(e);
    else
        forward_linearlstm(n, e->observations, e->actions);

    c_step(e);
    c_render(e);
}

int main() {
    Weights *w = load_weights("resources/enduro/enduro_weights.bin", 142218);
    int ls[1] = {9};
    n = make_linearlstm(w, 1, 68, ls, 1);

    static Enduro env = {
        .num_envs = 1,
        .max_enemies = MAX_ENEMIES,
        .obs_size = OBSERVATIONS_MAX_SIZE
    };

    allocate(&env);
    init(&env);
    c_reset(&env);

    e = &env;

    emscripten_set_main_loop_arg(loop, NULL, 0, 1);
    return 0;
}
#else

int main() {
    Weights *w  = load_weights("resources/enduro/enduro_weights.bin", 142218);
    int ls[1] = {9};
    LinearLSTM *n = make_linearlstm(w, 1, 68, ls, 1);

    static Enduro env = {
        .num_envs = 1,
        .max_enemies = MAX_ENEMIES,
        .obs_size = OBSERVATIONS_MAX_SIZE
    };

    allocate(&env);
    init(&env);
    c_reset(&env);
    c_render(&env);

    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT))
            get_input(&env);
        else
            forward_linearlstm(n, env.observations, env.actions);

        c_step(&env);
        c_render(&env);
    }

    free_linearlstm(n);
    free(w);
    free_allocated(&env);
    return 0;
}

#endif