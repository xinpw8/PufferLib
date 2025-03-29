#include <Python.h>
#include "pong.h"
#define ENV_MODULE_NAME "binding"
#define Env Pong
#define step c_step
#define render c_render
#define reset c_reset

static char *kwlist[] = {"width", "height", "paddle_width", "paddle_height",
    "ball_width", "ball_height", "paddle_speed", "ball_initial_speed_x",
    "ball_initial_speed_y", "ball_max_speed_y", "ball_speed_y_increment",
    "max_score", "frameskip", "continuous", NULL
};

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "fffffffffffIii", kwlist,
            &env->width, &env->height, &env->paddle_width,
            &env->paddle_height, &env->ball_width, &env->ball_height,
            &env->paddle_speed, &env->ball_initial_speed_x,
            &env->ball_initial_speed_y, &env->ball_max_speed_y,
            &env->ball_speed_y_increment, &env->max_score,
            &env->frameskip, &env->continuous)) {
        return 1;
    }
    init(env);
    return 0;
}

#include "../env_binding.h"
DEFINE_PYINIT(binding)
