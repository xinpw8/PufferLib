#include <Python.h>
#include "breakout.h"

#define ENV_MODULE_NAME "binding"
#define Env Breakout
#define step c_step
#define render c_render
#define reset c_reset

static char *kwlist[] = {
    "frameskip", "width", "height", "paddle_width", "paddle_height",
    "ball_width", "ball_height", "brick_width", "brick_height",
    "brick_rows", "brick_cols", "continuous",
    NULL
};

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|iiiffiiffiii", kwlist,
        &env->frameskip, &env->width, &env->height,
        &env->paddle_width, &env->paddle_height,
        &env->ball_width, &env->ball_height,
        &env->brick_width, &env->brick_height,
        &env->brick_rows, &env->brick_cols,
        &env->continuous)) {
        return 1;
    }

    init(env);
    return 0;
}

#include "../env_binding.h"
DEFINE_PYINIT(binding)
