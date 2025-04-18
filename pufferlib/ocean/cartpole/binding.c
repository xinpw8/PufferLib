#include "cartpole.h"
#define Env CartPole
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {   
    env->is_continuous = unpack(kwargs, "continuous");
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "x_threshold_termination", log->x_threshold_termination);
    assign_to_dict(dict, "pole_angle_termination", log->pole_angle_termination);
    assign_to_dict(dict, "max_steps_termination", log->max_steps_termination);
    assign_to_dict(dict, "n", log->n);
    return 0;
}
