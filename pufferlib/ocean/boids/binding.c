#include "boids.h"

#define Env Boids
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->num_envs = extract_int(kwargs, "num_envs", 1);
    env->num_boids = extract_int(kwargs, "num_boids", 1);
    env->max_steps = extract_int(kwargs, "max_steps", 1000);
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "n", log->n);
    return 0;
}
