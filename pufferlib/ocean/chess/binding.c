// binding.c
#include "chess.h"
#define Env CChess
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    c_init(env); // alloc & new ChessContext
    env->reward_valid   = unpack(kwargs,"reward_valid");
    env->reward_invalid = unpack(kwargs,"reward_invalid");
    env->reward_capture = unpack(kwargs,"reward_capture");
    env->reward_captured= unpack(kwargs,"reward_captured");
    env->reward_win     = unpack(kwargs,"reward_win");
    env->reward_draw    = unpack(kwargs,"reward_draw");
    env->reward_loss    = unpack(kwargs,"reward_loss");
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "n", log->n);
    return 0;
}