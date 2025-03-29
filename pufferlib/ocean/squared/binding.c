#include <Python.h>
#include "squared.h"
#define ENV_MODULE_NAME "binding"
#define Env Squared
#define step c_step
#define render c_render
#define reset c_reset

static char *kwlist[] = {"size", NULL};
static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|l", kwlist, &env->size)) {
        return 1;
    }
    return 0;
}

#include "../env_binding.h"
DEFINE_PYINIT(binding)
