#include "snake.h"

#define Env CSnake
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {   
    env->width = unpack(kwargs, "width");
    env->height = unpack(kwargs, "height");
    env->num_snakes = unpack(kwargs, "num_snakes");
    env->vision = unpack(kwargs, "vision");
    env->leave_corpse_on_death = unpack(kwargs, "leave_corpse_on_death");
    env->food = unpack(kwargs, "num_food");
    env->reward_food = unpack(kwargs, "reward_food");
    env->reward_corpse = unpack(kwargs, "reward_corpse");
    env->reward_death = unpack(kwargs, "reward_death");
    env->max_snake_length = unpack(kwargs, "max_snake_length");
    env->cell_size = unpack(kwargs, "cell_size");    
    init_csnake(env);
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
