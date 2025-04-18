#include <Python.h>
#include <numpy/arrayobject.h>

// Forward declarations for env-specific functions supplied by user
static int my_log(PyObject* dict, Log* log);
static int my_init(Env* env, PyObject* args, PyObject* kwargs);
// static PyObject* my_shared(PyObject* self, PyObject* args, PyObject* kwargs);

// Forward declaration for utility function
static double unpack(PyObject* kwargs, char* key);

static Env* unpack_env(PyObject* args) {
    PyObject* handle_obj = PyTuple_GetItem(args, 0);
    if (!PyObject_TypeCheck(handle_obj, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "env_handle must be an integer");
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
    if (PyTuple_Size(args) != 6) {
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
    env->truncations = PyArray_DATA(truncations);
    
    
    PyObject* seed_arg = PyTuple_GetItem(args, 5);
    if (!PyObject_TypeCheck(seed_arg, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "seed must be an integer");
        return NULL;
    }
    int seed = PyLong_AsLong(seed_arg);
 
    // Assumes each process has the same number of environments
    srand(seed);

    // If kwargs is NULL, create a new dictionary
    if (kwargs == NULL) {
        kwargs = PyDict_New();
    } else {
        Py_INCREF(kwargs);  // We need to increment the reference since we'll be modifying it
    }

    // Add the seed to kwargs
    PyObject* py_seed = PyLong_FromLong(seed);
    if (PyDict_SetItemString(kwargs, "seed", py_seed) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to set seed in kwargs");
        Py_DECREF(py_seed);
        Py_DECREF(kwargs);
        return NULL;
    }
    Py_DECREF(py_seed);

    // Extract the continuous flag (7th argument)
    PyObject* continuous_obj = PyTuple_GetItem(args, 6);
    if (continuous_obj != NULL && PyObject_TypeCheck(continuous_obj, &PyLong_Type)) {
        PyObject* py_continuous = PyLong_FromLong(PyLong_AsLong(continuous_obj));
        if (PyDict_SetItemString(kwargs, "continuous", py_continuous) < 0) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to set continuous in kwargs");
            Py_DECREF(py_continuous);
            Py_DECREF(kwargs);
            return NULL;
        }
        Py_DECREF(py_continuous);
    }

    PyObject* empty_args = PyTuple_New(0);
    if (my_init(env, empty_args, kwargs)) {
        //PyErr_SetString(PyExc_TypeError, "env_init failed");
        Py_DECREF(kwargs);
        return NULL;
    }

    Py_DECREF(kwargs);
    return PyLong_FromVoidPtr(env);
}

// Python function to reset the environment
static PyObject* env_reset(PyObject* self, PyObject* args) {
    Env* env = unpack_env(args);
    if (!env){
        return NULL;
    }
    c_reset(env);
    Py_RETURN_NONE;
}


// Python function to step the environment
static PyObject* env_step(PyObject* self, PyObject* args) {
    Env* env = unpack_env(args);
    if (!env){
        return NULL;
    }
    c_step(env);
    Py_RETURN_NONE;
}

// Python function to step the environment
static PyObject* env_render(PyObject* self, PyObject* args) {
    Env* env = unpack_env(args);
    if (!env){
        return NULL;
    }
    c_render(env);
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

/*
static VecEnv* unpack_vecenv(PyObject* args) {
    fprintf(stderr, "unpack_vecenv: Starting\n");
    
    PyObject* self = PyTuple_GetItem(args, 0);
    if (!self) {
        fprintf(stderr, "unpack_vecenv: Failed to get self from args\n");
        PyErr_SetString(PyExc_TypeError, "Could not get self");
        return NULL;
    }
    fprintf(stderr, "unpack_vecenv: Got self\n");

    PyObject* vec_capsule = PyObject_GetAttrString(self, "vec");
    if (!vec_capsule) {
        fprintf(stderr, "unpack_vecenv: Failed to get vec capsule\n");
        PyErr_SetString(PyExc_TypeError, "No vec capsule");
        return NULL;
    }
    fprintf(stderr, "unpack_vecenv: Got vec capsule\n");

    VecEnv* vec = (VecEnv*)PyCapsule_GetPointer(vec_capsule, "vec");
    if (!vec) {
        fprintf(stderr, "unpack_vecenv: Failed to get vec pointer from capsule\n");
        PyErr_SetString(PyExc_TypeError, "Not a valid vec capsule");
        Py_DECREF(vec_capsule);
        return NULL;
    }
    fprintf(stderr, "unpack_vecenv: Got vec pointer\n");

    Py_DECREF(vec_capsule);
    return vec;
}
*/

// Destructor for the VecEnv capsule
static void vec_destructor(PyObject* capsule) {
    VecEnv* vec = (VecEnv*)PyCapsule_GetPointer(capsule, "vec");
    if (vec == NULL) {
        // Error getting pointer, maybe already freed or wrong capsule?
        // You might want to set a Python error here if appropriate
        // fprintf(stderr, "Error: Could not get VecEnv pointer in destructor\n");
        return;
    }
    if (vec->envs) {
        // Assuming individual envs are freed elsewhere or don't need freeing if part of a larger block
        // If vec_init allocated individual envs, free them here:
        // for (int i = 0; i < vec->num_envs; ++i) {
        //     if (vec->envs[i]) free(vec->envs[i]);
        // }
        free(vec->envs); // Free the array of pointers
    }
    free(vec); // Free the VecEnv struct itself
}

static PyObject* vec_init(PyObject* self, PyObject* args, PyObject* kwargs) {
    if (PyTuple_Size(args) != 7) {
        PyErr_SetString(PyExc_TypeError, "vec_init requires 7 arguments: obs, act, rew, term, trunc, num_envs, is_continuous");
        return NULL;
    }

    VecEnv* vec = (VecEnv*)calloc(1, sizeof(VecEnv));
    if (!vec) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate vec env");
        return NULL;
    }

    PyObject* obs = PyTuple_GetItem(args, 0);
    PyObject* act = PyTuple_GetItem(args, 1);
    PyObject* rew = PyTuple_GetItem(args, 2);
    PyObject* term = PyTuple_GetItem(args, 3);
    PyObject* trunc = PyTuple_GetItem(args, 4);
    
    PyObject* num_envs_obj = PyTuple_GetItem(args, 5);
    if (!PyLong_Check(num_envs_obj)) {
        PyErr_SetString(PyExc_TypeError, "num_envs must be an integer");
        free(vec);
        return NULL;
    }
    vec->num_envs = PyLong_AsLong(num_envs_obj);
    if (vec->num_envs <= 0) {
        PyErr_SetString(PyExc_ValueError, "num_envs must be positive");
        free(vec);
        return NULL;
    }

    PyObject* continuous_obj = PyTuple_GetItem(args, 6);
     if (!PyLong_Check(continuous_obj)) {
        PyErr_SetString(PyExc_TypeError, "is_continuous must be an integer (0 or 1)");
        free(vec);
        return NULL;
    }
    int is_continuous = PyLong_AsLong(continuous_obj);


    // Allocate array of environment pointers
    vec->envs = (Env**)calloc(vec->num_envs, sizeof(Env*));
    if (!vec->envs) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate env pointers");
        free(vec);
        return NULL;
    }

    // Check array types and get pointers
    PyArrayObject* obs_arr = (PyArrayObject*)PyArray_FROM_OTF(obs, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* act_arr = (PyArrayObject*)PyArray_FROM_OTF(act, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY); // Assuming float for continuous, might need adjustment
    PyArrayObject* rew_arr = (PyArrayObject*)PyArray_FROM_OTF(rew, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* term_arr = (PyArrayObject*)PyArray_FROM_OTF(term, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* trunc_arr = (PyArrayObject*)PyArray_FROM_OTF(trunc, NPY_UINT8, NPY_ARRAY_IN_ARRAY);

    if (!obs_arr || !act_arr || !rew_arr || !term_arr || !trunc_arr) {
        Py_XDECREF(obs_arr); Py_XDECREF(act_arr); Py_XDECREF(rew_arr); Py_XDECREF(term_arr); Py_XDECREF(trunc_arr);
        PyErr_SetString(PyExc_TypeError, "Failed to convert input arrays");
        free(vec->envs); free(vec);
        return NULL;
    }

    // Get data pointers and dimensions/strides
    float* obs_data = (float*)PyArray_DATA(obs_arr);
    float* act_data = (float*)PyArray_DATA(act_arr);
    float* rew_data = (float*)PyArray_DATA(rew_arr);
    unsigned char* term_data = (unsigned char*)PyArray_DATA(term_arr);
    unsigned char* trunc_data = (unsigned char*)PyArray_DATA(trunc_arr);

    npy_intp* obs_dims = PyArray_DIMS(obs_arr);
    npy_intp* act_dims = PyArray_DIMS(act_arr);
    // Assuming obs is (num_envs, obs_size), act is (num_envs, act_size) or (num_envs,)
    int obs_size = (PyArray_NDIM(obs_arr) > 1) ? obs_dims[1] : 1; // Handle 1D case?
    int act_size = (PyArray_NDIM(act_arr) > 1) ? act_dims[1] : 1; 


    PyObject* empty_args = PyTuple_New(0);
    PyObject* local_kwargs = PyDict_Copy(kwargs); // Copy kwargs for local modification if needed
    if (!local_kwargs) local_kwargs = PyDict_New(); // Create if NULL

    // Set the 'continuous' flag in kwargs for my_init
    PyObject* py_continuous = PyLong_FromLong(is_continuous);
    if (PyDict_SetItemString(local_kwargs, "continuous", py_continuous) < 0) {
         PyErr_SetString(PyExc_RuntimeError, "Failed to set continuous in kwargs for my_init");
         Py_DECREF(py_continuous); Py_DECREF(local_kwargs); Py_DECREF(empty_args);
         Py_DECREF(obs_arr); Py_DECREF(act_arr); Py_DECREF(rew_arr); Py_DECREF(term_arr); Py_DECREF(trunc_arr);
         free(vec->envs); free(vec);
         return NULL;
    }
    Py_DECREF(py_continuous);


    // Initialize each environment
    for (int i = 0; i < vec->num_envs; ++i) {
        vec->envs[i] = (Env*)calloc(1, sizeof(Env));
        if (!vec->envs[i]) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate single env");
            // Cleanup previously allocated envs
            for (int j = 0; j < i; ++j) free(vec->envs[j]);
            free(vec->envs); free(vec);
             Py_DECREF(local_kwargs); Py_DECREF(empty_args);
            Py_DECREF(obs_arr); Py_DECREF(act_arr); Py_DECREF(rew_arr); Py_DECREF(term_arr); Py_DECREF(trunc_arr);
            return NULL;
        }

        // Assign sliced pointers from NumPy arrays
        vec->envs[i]->observations = obs_data + i * obs_size;
        vec->envs[i]->actions = act_data + i * act_size;
        vec->envs[i]->rewards = rew_data + i;
        vec->envs[i]->terminals = term_data + i;
        vec->envs[i]->truncations = trunc_data + i;
        
        // Call the environment-specific initializer
        if (my_init(vec->envs[i], empty_args, local_kwargs)) {
            PyErr_SetString(PyExc_RuntimeError, "my_init failed for an environment");
            // Cleanup
            for (int j = 0; j <= i; ++j) free(vec->envs[j]);
            free(vec->envs); free(vec);
            Py_DECREF(local_kwargs); Py_DECREF(empty_args);
            Py_DECREF(obs_arr); Py_DECREF(act_arr); Py_DECREF(rew_arr); Py_DECREF(term_arr); Py_DECREF(trunc_arr);
            return NULL;
        }
    }

    Py_DECREF(empty_args);
    Py_DECREF(local_kwargs);
    Py_DECREF(obs_arr); Py_DECREF(act_arr); Py_DECREF(rew_arr); Py_DECREF(term_arr); Py_DECREF(trunc_arr);

    // Return the VecEnv pointer wrapped in a capsule
    PyObject* vec_capsule = PyCapsule_New(vec, "vec", vec_destructor);
    if (!vec_capsule) {
        PyErr_SetString(PyExc_SystemError, "Failed to create vec capsule");
        // Cleanup everything allocated
        for (int i = 0; i < vec->num_envs; ++i) free(vec->envs[i]);
        free(vec->envs); free(vec);
        return NULL;
    }
    
    // fprintf(stderr, "vec_init: Success, returning capsule %p wrapping vec %p\n", (void*)vec_capsule, (void*)vec);
    return vec_capsule;
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
    // fprintf(stderr, "vec_reset: Starting reset\n");
    PyObject* vec_capsule = PyTuple_GetItem(args, 0); // Capsule is the first arg
    if (!PyCapsule_CheckExact(vec_capsule)) {
         PyErr_SetString(PyExc_TypeError, "Argument must be a vec capsule");
         return NULL;
    }
    VecEnv* vec = (VecEnv*)PyCapsule_GetPointer(vec_capsule, "vec");
    if (!vec) {
        // fprintf(stderr, "vec_reset: Invalid vector environment capsule\n");
        // Error already set by PyCapsule_GetPointer
        return NULL;
    }

    for (int i=0; i<vec->num_envs; ++i) {
        // fprintf(stderr, "vec_reset: Resetting environment %d\n", i);
        if (vec->envs[i] == NULL) {
            // fprintf(stderr, "vec_reset: Error - environment %d is NULL\n", i);
            continue;
        }
        
        // Assumes each process has the same number of environments
        // srand(i + seed*vec->num_envs); // Removed: Seed should not be used in reset
        // fprintf(stderr, "vec_reset: Calling c_reset for environment %d\n", i);
        c_reset(vec->envs[i]);
        // fprintf(stderr, "vec_reset: Reset environment %d successful\n", i);
    }
    Py_RETURN_NONE;
}

static PyObject* vec_step(PyObject* self, PyObject* args) {
    PyObject* vec_capsule = PyTuple_GetItem(args, 0); // Capsule is the first arg
    if (!PyCapsule_CheckExact(vec_capsule)) {
         PyErr_SetString(PyExc_TypeError, "Argument must be a vec capsule");
         return NULL;
    }
    VecEnv* vec = (VecEnv*)PyCapsule_GetPointer(vec_capsule, "vec");
     if (!vec) {
        // fprintf(stderr, "vec_step: Invalid vector environment capsule\n");
        return NULL;
    }

    for (int i=0; i<vec->num_envs; ++i) {
        c_step(vec->envs[i]);
    }
    Py_RETURN_NONE;
}

static PyObject* vec_render(PyObject* self, PyObject* args) {
    PyObject* vec_capsule = PyTuple_GetItem(args, 0); // Capsule is the first arg
     if (!PyCapsule_CheckExact(vec_capsule)) {
         PyErr_SetString(PyExc_TypeError, "Argument must be a vec capsule");
         return NULL;
    }
    VecEnv* vec = (VecEnv*)PyCapsule_GetPointer(vec_capsule, "vec");
     if (!vec) {
        // fprintf(stderr, "vec_render: Invalid vector environment capsule\n");
        return NULL;
    }

    PyObject* index_obj = PyTuple_GetItem(args, 1);
    if (!PyLong_Check(index_obj)){
        PyErr_SetString(PyExc_TypeError, "Render index must be an integer");
        return NULL;
    }
    int index = PyLong_AsLong(index_obj);

    if (index < 0 || index >= vec->num_envs) {
        PyErr_SetString(PyExc_IndexError, "Render index out of bounds");
        return NULL;
    }

    c_render(vec->envs[index]);
    Py_RETURN_NONE;
}

static PyObject* vec_close(PyObject* self, PyObject* args) {
    PyObject* vec_capsule = PyTuple_GetItem(args, 0); // Capsule is the first arg
    if (!PyCapsule_CheckExact(vec_capsule)) {
         PyErr_SetString(PyExc_TypeError, "Argument must be a vec capsule");
         return NULL;
    }
    // The capsule destructor (vec_destructor) handles freeing memory.
    // We might optionally invalidate the capsule pointer here if needed,
    // but Python's garbage collection + capsule mechanism should handle it.
    // Forcing destruction isn't standard; usually, it happens when the capsule
    // object goes out of scope in Python and is garbage collected.

    // Explicitly clear the pointer inside the capsule to prevent double free
    // if close is called multiple times before GC.
    // PyCapsule_SetPointer(vec_capsule, NULL); 
    // ^^^ This might be useful if vec_close could be called manually multiple times.
    // However, the destructor should be robust against NULL pointers anyway.
    
    // fprintf(stderr, "vec_close: Called (memory freeing deferred to capsule destructor)\n");

    Py_RETURN_NONE;
}

static PyObject* vec_log(PyObject* self, PyObject* args) {
    PyObject* vec_capsule = PyTuple_GetItem(args, 0); // Capsule is the first arg
    if (!PyCapsule_CheckExact(vec_capsule)) {
         PyErr_SetString(PyExc_TypeError, "Argument must be a vec capsule");
         return NULL;
    }
    VecEnv* vec = (VecEnv*)PyCapsule_GetPointer(vec_capsule, "vec");
     if (!vec) {
        // fprintf(stderr, "vec_log: Invalid vector environment capsule\n");
        return NULL;
    }

    PyObject* log_list = PyList_New(vec->num_envs);
    if (!log_list) {
        return NULL; // Error creating list
    }

    for (int i=0; i<vec->num_envs; ++i) {
        PyObject* log_dict = PyDict_New();
        if (!log_dict) {
            Py_DECREF(log_list); // Clean up list
            return NULL; // Error creating dict
        }

        if (my_log(log_dict, &vec->envs[i]->log)) {
             Py_DECREF(log_dict);
             Py_DECREF(log_list);
             PyErr_SetString(PyExc_RuntimeError, "my_log failed");
             return NULL;
        }

        // Zero out the log struct after reading
        memset(&vec->envs[i]->log, 0, sizeof(Log));

        PyList_SET_ITEM(log_list, i, log_dict); // Steals reference to log_dict
    }

    return log_list;
}

static int assign_to_dict(PyObject* dict, char* key, float value) {
    PyObject* v = PyFloat_FromDouble(value);
    if (v == NULL) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert log value");
        return 1;
    }
    if(PyDict_SetItemString(dict, key, v) < 0) {
        PyErr_SetString(PyExc_TypeError, "Failed to set log value");
        return 1;
    }
    return 0;
}

// Helper function to unpack keyword arguments
static double unpack(PyObject* kwargs, char* key) {
    if (!kwargs || !PyDict_Check(kwargs)) {
        // Return a default value or handle error if kwargs is not a dict
        // fprintf(stderr, "Warning: unpack called with non-dict or NULL kwargs for key '%s'\n", key);
        return 0.0; // Or some other indicator
    }
    PyObject* val = PyDict_GetItemString(kwargs, key);
    if (val == NULL) {
        // Key doesn't exist - this might be intentional for optional args
        // PyErr_Clear(); // Clear any potential error from GetItemString if key not found
        return 0.0; // Return default or indicator
    }
    if (PyLong_Check(val)) {
        PyErr_Clear(); // Clear any previous error before calling AsLong
        long out = PyLong_AsLong(val);
        // Check for overflow if necessary, depending on expected range
        if (PyErr_Occurred()) { // PyLong_AsLong sets an error on overflow
            char error_msg[100];
            snprintf(error_msg, sizeof(error_msg), "Integer value for key '%s' out of range", key);
            PyErr_SetString(PyExc_OverflowError, error_msg);
            return -1.0; // Indicate error
        }
        return (double)out; // Safe cast
    }
    if (PyFloat_Check(val)) {
        return PyFloat_AsDouble(val);
    }

    // Value exists but is not Long or Float
    char error_msg[100];
    snprintf(error_msg, sizeof(error_msg), "Keyword argument '%s' must be an integer or float", key);
    PyErr_SetString(PyExc_TypeError, error_msg);
    return -1.0; // Indicate error
}

// Method table
static PyMethodDef methods[] = {
    {"env_init", (PyCFunction)env_init, METH_VARARGS | METH_KEYWORDS, "Init environment with observation, action, reward, terminal, truncation arrays"},
    {"env_reset", env_reset, METH_VARARGS, "Reset the environment"},
    {"env_step", env_step, METH_VARARGS, "Step the environment"},
    {"env_render", env_render, METH_VARARGS, "Render the environment"},
    {"env_close", env_close, METH_VARARGS, "Close the environment"},
    {"vectorize", vectorize, METH_VARARGS, "Make a vector of environment handles"},
    {"vec_init", (PyCFunction)vec_init, METH_VARARGS | METH_KEYWORDS, "Initialize a vector of environments"},
    {"vec_reset", (PyCFunction)vec_reset, METH_VARARGS, "Reset the vector of environments"},
    {"vec_step", (PyCFunction)vec_step, METH_VARARGS, "Step the vector of environments"},
    {"vec_log", vec_log, METH_VARARGS, "Log the vector of environments"},
    {"vec_render", vec_render, METH_VARARGS, "Render the vector of environments"},
    {"vec_close", vec_close, METH_VARARGS, "Close the vector of environments"},
    // {"shared", (PyCFunction)my_shared, METH_VARARGS | METH_KEYWORDS, "Shared state"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "binding",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit_binding(void) { \
    import_array(); \
    return PyModule_Create(&module); \
}
