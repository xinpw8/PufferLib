#include <Python.h>
#include "enduro.h"

#define ENV_MODULE_NAME "binding"
#define Env Enduro
#define step c_step
#define render c_render
#define reset c_reset

// If you want to parse "seed" and "index" from Python:
static char *kwlist[] = {
    "seed",
    "index",
    NULL
};

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    // local variables to hold the parsed ints
    int seed_val, index_val;

    // parse exactly two integers: seed, index
    if (!PyArg_ParseTupleAndKeywords(
        args, kwargs,
        "ii",         // format: "ii" means we want two ints
        kwlist,
        &seed_val,
        &index_val
    )) {
        return 1; // parse error
    }

    // Now call your environment's init with those values
    init(env, seed_val, index_val);
    return 0; // success
}

// Then pull in the generic "env_binding.h" that calls env_init, etc.
#include "../env_binding.h"

// Provide the PyInit function
DEFINE_PYINIT(binding)
