// binding.cpp
#include <string>
#include <iostream>
#include "flat_chess_env.h"

#define Env CChess
#include "../env_binding.h" 

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {   
    env->ctx->r_valid = unpack(kwargs, const_cast<char*>("reward_move_valid"));
    env->ctx->r_invalid = unpack(kwargs, const_cast<char*>("reward_move_invalid"));
    env->ctx->r_capture = unpack(kwargs, const_cast<char*>("reward_player_capture"));
    env->ctx->r_captured = unpack(kwargs, const_cast<char*>("reward_opponent_capture"));
    env->ctx->r_win = unpack(kwargs, const_cast<char*>("reward_win"));
    env->ctx->r_draw = unpack(kwargs, const_cast<char*>("reward_draw"));
    env->ctx->r_loss = unpack(kwargs, const_cast<char*>("reward_loss"));
    c_init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, const_cast<char*>("perf"), log->perf);
    assign_to_dict(dict, const_cast<char*>("score"), log->score);
    assign_to_dict(dict, const_cast<char*>("episode_return"), log->episode_return);
    assign_to_dict(dict, const_cast<char*>("episode_length"), log->episode_length);
    assign_to_dict(dict, const_cast<char*>("n"), log->n);
    return 0;
} 