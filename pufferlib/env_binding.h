#ifndef ENV_BINDING_H
#define ENV_BINDING_H

#include <Python.h>
#include <numpy/arrayobject.h>

// Environment-specific functions (assumed to exist in env_*.h)
//void init(void* env);
//void step(void* env);
//void close(void* env);

static PyObject* env_init(PyObject* self, PyObject* args, PyObject* kwargs);

// Validate a single NumPy array: 1D and contiguous
static int validate_array(PyObject* arg, const char* name, PyArrayObject** array_out) {
    if (!PyObject_TypeCheck(arg, &PyArray_Type)) {
        char msg[64];
        snprintf(msg, sizeof(msg), "%s must be a NumPy array", name);
        PyErr_SetString(PyExc_TypeError, msg);
        return 0;
    }

    PyArrayObject* array = (PyArrayObject*)arg;
    if (PyArray_NDIM(array) != 1 || !PyArray_ISCONTIGUOUS(array)) {
        char msg[64];
        snprintf(msg, sizeof(msg), "%s must be 1D and contiguous", name);
        PyErr_SetString(PyExc_ValueError, msg);
        return 0;
    }

    *array_out = array;
    return 1;
}

static Env* unpack_env(PyObject* args) {
    PyObject* handle_obj;
    if (!PyArg_ParseTuple(args, "O", &handle_obj)) {
        PyErr_SetString(PyExc_ValueError, "Unrecognized arguments");
        return NULL;
    }

    Env* env = (Env*)PyLong_AsVoidPtr(handle_obj);
    if (!env) {
        PyErr_SetString(PyExc_ValueError, "Invalid env handle");
        return NULL;
    }

    return env;
}


// Python function to initialize the environment
static Env* base_init(PyObject* self, PyObject* args, PyObject* kwargs) {
    Env* env = (Env*)calloc(1, sizeof(Env));
    if (!env) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate environment");
        return NULL;
    }

    if (PyTuple_Size(args) < 5) {
        PyErr_SetString(PyExc_TypeError, "init_env requires at least 5 arguments");
        return NULL;
    }

    PyArrayObject* obs_array = NULL;
    if (!validate_array(PyTuple_GetItem(args, 0), "Observations", &obs_array)) return NULL;
    env->observations = PyArray_DATA(obs_array);

    PyArrayObject* act_array = NULL;
    if (!validate_array(PyTuple_GetItem(args, 1), "Actions", &act_array)) return NULL;
    env->actions = PyArray_DATA(act_array);

    PyArrayObject* rew_array = NULL;
    if (!validate_array(PyTuple_GetItem(args, 2), "Rewards", &rew_array)) return NULL;
    env->rewards = PyArray_DATA(rew_array);

    PyArrayObject* term_array = NULL;
    if (!validate_array(PyTuple_GetItem(args, 3), "Terminals", &term_array)) return NULL;
    env->terminals = PyArray_DATA(term_array);

    PyArrayObject* trunc_array = NULL;
    if (!validate_array(PyTuple_GetItem(args, 4), "Truncations", &trunc_array)) return NULL;
    //env->truncations = PyArray_DATA(trunc_array);

    return env;
}

static Env* default_init(PyObject* self, PyObject* args, PyObject* kwargs) {
    Env* env = base_init(self, args, kwargs);
    if (!env) {
        return NULL; }
    init(env);
    return env;
}

// Python function to reset the environment
static PyObject* env_reset(PyObject* self, PyObject* args) {
    Env* env = unpack_env(args);
    if (!env){
        return NULL;
    }
    reset(env);
    Py_RETURN_NONE;
}


// Python function to step the environment
static PyObject* env_step(PyObject* self, PyObject* args) {
    Env* env = unpack_env(args);
    if (!env){
        return NULL;
    }
    step(env);
    Py_RETURN_NONE;
}

// Python function to step the environment
static PyObject* env_render(PyObject* self, PyObject* args) {
    Env* env = unpack_env(args);
    if (!env){
        return NULL;
    }
    render(env);
    Py_RETURN_NONE;
}

// Python function to close the environment
static PyObject* env_close(PyObject* self, PyObject* args) {
    Env* env = unpack_env(args);
    if (!env){
        return NULL;
    }
    //close(env);
    free(env); // TODO: Where should we free?
    Py_RETURN_NONE;
}

typedef struct {
    Env** envs;
    int num_envs;
} VecEnv;

static VecEnv* unpack_vecenv(PyObject* args) {
    PyObject* handle_obj;
    if (!PyArg_ParseTuple(args, "O", &handle_obj)) {
        PyErr_SetString(PyExc_ValueError, "Unrecognized arguments");
        return NULL;
    }

    VecEnv* vec = (VecEnv*)PyLong_AsVoidPtr(handle_obj);
    if (!vec) {
        PyErr_SetString(PyExc_ValueError, "Invalid vec env handle");
        return NULL;
    }

    return vec;
}

// Python function to close the environment
static PyObject* make_vec(PyObject* self, PyObject* args) {
    int num_envs = PyTuple_Size(args);
    if (num_envs == 0) {
        PyErr_SetString(PyExc_TypeError, "make_vec requires at least 1 env id");
        return NULL;
    }

    VecEnv* vec = (VecEnv*)calloc(1, sizeof(VecEnv));
    if (!vec) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate vec env");
        return NULL;
    }

    vec->envs = (Env**)calloc(num_envs, sizeof(Env*));
    if (!vec->envs) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate vec env");
        return NULL;
    }

    vec->num_envs = num_envs;
    for (int i = 0; i < num_envs; i++) {
        PyObject* handle_obj = PyTuple_GetItem(args, i);
        if (!PyObject_TypeCheck(handle_obj, &PyLong_Type)) {
            PyErr_SetString(PyExc_TypeError, "Env ids must be integers. Pass them as separate args with *env_ids, not as a list.");
            return NULL;
        }
        vec->envs[i] = (Env*)PyLong_AsVoidPtr(handle_obj);
    }

    return PyLong_FromVoidPtr(vec);
}

static PyObject* vec_reset(PyObject* self, PyObject* args) {
    VecEnv* vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    for (int i = 0; i < vec->num_envs; i++) {
        reset(vec->envs[i]);
    }
    Py_RETURN_NONE;
}

static PyObject* vec_step(PyObject* self, PyObject* args) {
    VecEnv* vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    for (int i = 0; i < vec->num_envs; i++) {
        step(vec->envs[i]);
    }
    Py_RETURN_NONE;
}

static PyObject* vec_close(PyObject* self, PyObject* args) {
    VecEnv* vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    for (int i = 0; i < vec->num_envs; i++) {
        free(vec->envs[i]);
        //close(vec->envs[i]);
    }
    free(vec->envs);
    free(vec);
    Py_RETURN_NONE;
}

// Method table
static PyMethodDef methods[] = {
    {"env_init", (PyCFunction)env_init, METH_VARARGS | METH_KEYWORDS, "Init environment with observation, action, reward, terminal, truncation arrays"},
    {"env_reset", env_reset, METH_VARARGS, "Reset the environment"},
    {"env_step", env_step, METH_VARARGS, "Step the environment"},
    {"env_render", env_render, METH_VARARGS, "Render the environment"},
    {"env_close", env_close, METH_VARARGS, "Close the environment"},
    {"make_vec", make_vec, METH_VARARGS, "Make a vector of environment handles"},
    {"vec_step", vec_step, METH_VARARGS, "Step the vector of environments"},
    {"vec_close", vec_close, METH_VARARGS, "Close the vector of environments"},
    {"vec_reset", vec_reset, METH_VARARGS, "Reset the vector of environments"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    ENV_MODULE_NAME, // Macro for module name
    NULL,
    -1,
    methods
};

// Macro to define the PyInit function
#define DEFINE_PYINIT(name) \
    PyMODINIT_FUNC PyInit_##name(void) { \
        import_array(); \
        return PyModule_Create(&module); \
    }

#endif
