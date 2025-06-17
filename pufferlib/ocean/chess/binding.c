#include "flat_chess_env.h"
#define Env CChess
#include "../env_binding.h"

// Populate env parameters (none for now) and call allocate()
static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    // For Chess we currently do not support runtime parameters; future params
    // can be unpacked from kwargs here.
    allocate(env);
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