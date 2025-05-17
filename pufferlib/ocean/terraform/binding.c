#include "squared.h"

#define Env Squared
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->obs_size = unpack(kwargs, "obs_size");
    env->map_size = unpack(kwargs, "map_size");
    env->num_agents = unpack(kwargs, "num_agents");
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}
