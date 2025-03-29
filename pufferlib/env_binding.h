#include <numpy/arrayobject.h>

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
static PyObject* env_init(PyObject* self, PyObject* args, PyObject* kwargs) {
    if (PyTuple_Size(args) != 5) {
        PyErr_SetString(PyExc_TypeError, "Environment requires 5 arguments");
        return NULL;
    }

    Env* env = (Env*)calloc(1, sizeof(Env));
    if (!env) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate environment");
        return NULL;
    }

    PyObject* obs = PyTuple_GetItem(args, 0);
    if (!PyObject_TypeCheck(obs, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Observations must be a NumPy array");
        return NULL;
    }
    PyArrayObject* observations = (PyArrayObject*)obs;
    if (!PyArray_ISCONTIGUOUS(observations)) {
        PyErr_SetString(PyExc_ValueError, "Observations must be contiguous");
        return NULL;
    }
    env->observations = PyArray_DATA(observations);

    PyObject* act = PyTuple_GetItem(args, 1);
    if (!PyObject_TypeCheck(act, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Actions must be a NumPy array");
        return NULL;
    }
    PyArrayObject* actions = (PyArrayObject*)act;
    if (!PyArray_ISCONTIGUOUS(actions)) {
        PyErr_SetString(PyExc_ValueError, "Actions must be contiguous");
        return NULL;
    }
    env->actions = PyArray_DATA(actions);

    PyObject* rew = PyTuple_GetItem(args, 2);
    if (!PyObject_TypeCheck(rew, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Rewards must be a NumPy array");
        return NULL;
    }
    PyArrayObject* rewards = (PyArrayObject*)rew;
    if (!PyArray_ISCONTIGUOUS(rewards)) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be contiguous");
        return NULL;
    }
    if (PyArray_NDIM(rewards) != 1) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be 1D");
        return NULL;
    }
    env->rewards = PyArray_DATA(rewards);

    PyObject* term = PyTuple_GetItem(args, 3);
    if (!PyObject_TypeCheck(term, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Terminals must be a NumPy array");
        return NULL;
    }
    PyArrayObject* terminals = (PyArrayObject*)term;
    if (!PyArray_ISCONTIGUOUS(terminals)) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be contiguous");
        return NULL;
    }
    if (PyArray_NDIM(terminals) != 1) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be 1D");
        return NULL;
    }
    env->terminals = PyArray_DATA(terminals);

    PyObject* trunc = PyTuple_GetItem(args, 4);
    if (!PyObject_TypeCheck(trunc, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Truncations must be a NumPy array");
        return NULL;
    }
    PyArrayObject* truncations = (PyArrayObject*)trunc;
    if (!PyArray_ISCONTIGUOUS(truncations)) {
        PyErr_SetString(PyExc_ValueError, "Truncations must be contiguous");
        return NULL;
    }
    if (PyArray_NDIM(truncations) != 1) {
        PyErr_SetString(PyExc_ValueError, "Truncations must be 1D");
        return NULL;
    }
    //env->truncations = PyArray_DATA(truncations);

    PyObject* empty_args = PyTuple_New(0);
    if (my_init(env, empty_args, kwargs)) {
        PyErr_SetString(PyExc_TypeError, "env_init failed");
        return NULL;
    }

    return PyLong_FromVoidPtr(env);
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

static PyObject* init_vec(PyObject* self, PyObject* args, PyObject* kwargs) {
    if (PyTuple_Size(args) != 6) {
        PyErr_SetString(PyExc_TypeError, "init_vec requires 6 arguments");
        return NULL;
    }

    VecEnv* vec = (VecEnv*)calloc(1, sizeof(VecEnv));
    if (!vec) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate vec env");
        return NULL;
    }
    PyObject* num_envs_arg = PyTuple_GetItem(args, 5);
    if (!PyObject_TypeCheck(num_envs_arg, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "num_envs must be an integer");
        return NULL;
    }
    int num_envs = PyLong_AsLong(num_envs_arg);
    if (num_envs <= 0) {
        PyErr_SetString(PyExc_TypeError, "num_envs must be greater than 0");
        return NULL;
    }
    vec->num_envs = num_envs;
    vec->envs = (Env**)calloc(num_envs, sizeof(Env*));
    if (!vec->envs) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate vec env");
        return NULL;
    }

    PyObject* obs = PyTuple_GetItem(args, 0);
    if (!PyObject_TypeCheck(obs, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Observations must be a NumPy array");
        return NULL;
    }
    PyArrayObject* observations = (PyArrayObject*)obs;
    if (!PyArray_ISCONTIGUOUS(observations)) {
        PyErr_SetString(PyExc_ValueError, "Observations must be contiguous");
        return NULL;
    }
    if (PyArray_NDIM(observations) < 2) {
        PyErr_SetString(PyExc_ValueError, "Batched Observations must be at least 2D");
        return NULL;
    }

    PyObject* act = PyTuple_GetItem(args, 1);
    if (!PyObject_TypeCheck(act, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Actions must be a NumPy array");
        return NULL;
    }
    PyArrayObject* actions = (PyArrayObject*)act;
    if (!PyArray_ISCONTIGUOUS(actions)) {
        PyErr_SetString(PyExc_ValueError, "Actions must be contiguous");
        return NULL;
    }

    PyObject* rew = PyTuple_GetItem(args, 2);
    if (!PyObject_TypeCheck(rew, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Rewards must be a NumPy array");
        return NULL;
    }
    PyArrayObject* rewards = (PyArrayObject*)rew;
    if (!PyArray_ISCONTIGUOUS(rewards)) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be contiguous");
        return NULL;
    }
    if (PyArray_NDIM(rewards) != 1) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be 1D");
        return NULL;
    }

    PyObject* term = PyTuple_GetItem(args, 3);
    if (!PyObject_TypeCheck(term, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Terminals must be a NumPy array");
        return NULL;
    }
    PyArrayObject* terminals = (PyArrayObject*)term;
    if (!PyArray_ISCONTIGUOUS(terminals)) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be contiguous");
        return NULL;
    }
    if (PyArray_NDIM(terminals) != 1) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be 1D");
        return NULL;
    }

    PyObject* trunc = PyTuple_GetItem(args, 4);
    if (!PyObject_TypeCheck(trunc, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Truncations must be a NumPy array");
        return NULL;
    }
    PyArrayObject* truncations = (PyArrayObject*)trunc;
    if (!PyArray_ISCONTIGUOUS(truncations)) {
        PyErr_SetString(PyExc_ValueError, "Truncations must be contiguous");
        return NULL;
    }
    if (PyArray_NDIM(truncations) != 1) {
        PyErr_SetString(PyExc_ValueError, "Truncations must be 1D");
        return NULL;
    }

    for (int i = 0; i < num_envs; i++) {
        Env* env = (Env*)calloc(1, sizeof(Env));
        if (!env) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate environment");
            return NULL;
        }
        vec->envs[i] = env;
        env->observations = (void*)((char*)PyArray_DATA(observations) + i*PyArray_STRIDE(observations, 0));
        env->actions = (void*)((char*)PyArray_DATA(actions) + i*PyArray_STRIDE(actions, 0));
        env->rewards = (void*)((char*)PyArray_DATA(rewards) + i*PyArray_STRIDE(rewards, 0));
        env->terminals = (void*)((char*)PyArray_DATA(terminals) + i*PyArray_STRIDE(terminals, 0));
        //env->truncations = (void*)((char*)PyArray_DATA(truncations) + i*PyArray_STRIDE(truncations, 0));
        PyObject* empty_args = PyTuple_New(0);
        if (my_init(env, empty_args, kwargs)) {
            PyErr_SetString(PyExc_TypeError, "env_init failed");
            return NULL;
        }
    }

    return PyLong_FromVoidPtr(vec);
}


// Python function to close the environment
static PyObject* vectorize(PyObject* self, PyObject* args) {
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

static PyObject* vec_render(PyObject* self, PyObject* args) {
    int num_args = PyTuple_Size(args);
    if (num_args != 2) {
        PyErr_SetString(PyExc_TypeError, "vec_render requires 2 arguments");
        return NULL;
    }

    VecEnv* vec = (VecEnv*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
    if (!vec) {
        PyErr_SetString(PyExc_ValueError, "Invalid vec_env handle");
        return NULL;
    }

    PyObject* env_id_arg = PyTuple_GetItem(args, 1);
    if (!PyObject_TypeCheck(env_id_arg, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "env_id must be an integer");
        return NULL;
    }
    int env_id = PyLong_AsLong(env_id_arg);
 
    render(vec->envs[env_id]);
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
    {"vectorize", vectorize, METH_VARARGS, "Make a vector of environment handles"},
    {"init_vec", (PyCFunction)init_vec, METH_VARARGS | METH_KEYWORDS, "Initialize a vector of environments"},
    {"vec_step", vec_step, METH_VARARGS, "Step the vector of environments"},
    {"vec_render", vec_render, METH_VARARGS, "Render the vector of environments"},
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
