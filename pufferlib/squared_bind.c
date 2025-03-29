#include "ocean/squared/squared.h"
#define ENV_MODULE_NAME "squared_bind"
#define Env Squared
#define step c_step
#define render c_render
#define reset c_reset

#include "env_binding.h"

static PyObject* env_init(PyObject* self, PyObject* args, PyObject* kwargs) {
    Env* env = base_init(self, args, kwargs);
    if (!env) {
        return NULL;
    }
    env->size = 11;
    return PyLong_FromVoidPtr(env);
}

DEFINE_PYINIT(squared_bind)
