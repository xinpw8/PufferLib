// binding.cpp
#pragma once
#include <Python.h> // makes PyObject visible
#include "chess.h"
#define Env CChess

// forward-declare so env_binding.h can use it
static PyObject* env_set_self_play(PyObject* self, PyObject* args);
static PyObject* vec_set_self_play(PyObject* self, PyObject* args);

// Define custom methods for chess module
#define MY_METHODS \
    {"env_set_self_play", env_set_self_play, METH_VARARGS, "Enable self-play mode"}, \
    {"vec_set_self_play", vec_set_self_play, METH_VARARGS, "Enable self-play mode for vector env"}, \
    {NULL, NULL, 0, NULL}

#include "../env_binding.h"

static PyObject* env_set_self_play(PyObject* self, PyObject* args) {
    Env* env = unpack_env(args);
    if (!env) return NULL;
    set_self_play_mode(env, true);
    Py_RETURN_NONE;
}

// Enable self-play for every env in a VecEnv
static PyObject* vec_set_self_play(PyObject* self, PyObject* args) {
    VecEnv* vec = unpack_vecenv(args);
    if (!vec) return NULL;
    for (int i = 0; i < vec->num_envs; ++i)
        set_self_play_mode(vec->envs[i], true);
    Py_RETURN_NONE;
}


static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->reward_valid = unpack(kwargs, "reward_valid");
    env->reward_invalid = unpack(kwargs, "reward_invalid");
    env->reward_agent_captures_enemy_piece = unpack(kwargs, "reward_agent_captures_enemy_piece");
    env->reward_enemy_captures_agent_piece = unpack(kwargs, "reward_enemy_captures_agent_piece");
    env->reward_win = unpack(kwargs, "reward_win");
    env->reward_draw = unpack(kwargs, "reward_draw");
    env->reward_loss = unpack(kwargs, "reward_loss");
    init(env);  // alloc & new ChessContext
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "reward_valid", log->reward_valid);
    assign_to_dict(dict, "reward_invalid", log->reward_invalid);
    assign_to_dict(dict, "reward_agent_captures_enemy_piece", log->reward_agent_captures_enemy_piece);
    assign_to_dict(dict, "reward_enemy_captures_agent_piece", log->reward_enemy_captures_agent_piece);
    assign_to_dict(dict, "reward_win", log->reward_win);
    assign_to_dict(dict, "reward_draw", log->reward_draw);
    assign_to_dict(dict, "reward_loss", log->reward_loss);
    assign_to_dict(dict, "game_won", log->game_won);
    assign_to_dict(dict, "game_lost", log->game_lost);
    assign_to_dict(dict, "game_drawn", log->game_drawn);
    assign_to_dict(dict, "stalemate", log->stalemate);
    assign_to_dict(dict, "insufficient_material", log->insufficient_material);
    assign_to_dict(dict, "threefold_repetition", log->threefold_repetition);
    assign_to_dict(dict, "fifty_move_rule", log->fifty_move_rule);
    assign_to_dict(dict, "max_depth", log->max_depth);
    assign_to_dict(dict, "white_checkmated", log->white_checkmated);
    assign_to_dict(dict, "black_checkmated", log->black_checkmated);
    assign_to_dict(dict, "n", log->n);
    return 0;
}
