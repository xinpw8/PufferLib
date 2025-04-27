#include "snake.h"

#define Env CSnake
#include "../env_binding.h"

// Helper function to extract an int from a Python object, handling lists
static int extract_int(PyObject* kwargs, const char* key, int default_value) {
    PyObject* obj = PyDict_GetItemString(kwargs, key);
    if (obj != NULL) {
        if (PyList_Check(obj)) {
            obj = PyList_GetItem(obj, 0);
        }
        return PyLong_AsLong(obj);
    }
    return default_value;
}

// Helper function to extract a float from a Python object, handling lists
static float extract_float(PyObject* kwargs, const char* key, float default_value) {
    PyObject* obj = PyDict_GetItemString(kwargs, key);
    if (obj != NULL) {
        if (PyList_Check(obj)) {
            obj = PyList_GetItem(obj, 0);
        }
        return PyFloat_AsDouble(obj);
    }
    return default_value;
}

// Helper function to extract a bool from a Python object, handling lists
static int extract_bool(PyObject* kwargs, const char* key, int default_value) {
    PyObject* obj = PyDict_GetItemString(kwargs, key);
    if (obj != NULL) {
        if (PyList_Check(obj)) {
            obj = PyList_GetItem(obj, 0);
        }
        return PyObject_IsTrue(obj);
    }
    return default_value;
}

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    // Get the environment index from kwargs
    int env_index = 0;
    PyObject* env_index_obj = PyDict_GetItemString(kwargs, "env_index");
    if (env_index_obj != NULL) {
        env_index = PyLong_AsLong(env_index_obj);
    }
    
    // Use unpack_with_index to properly handle lists
    env->width = unpack_with_index(kwargs, "width", env_index);
    env->height = unpack_with_index(kwargs, "height", env_index);
    env->num_snakes = unpack_with_index(kwargs, "num_snakes", env_index);
    env->vision = unpack_with_index(kwargs, "vision", env_index);
    env->leave_corpse_on_death = unpack_with_index(kwargs, "leave_corpse_on_death", env_index);
    env->food = unpack_with_index(kwargs, "num_food", env_index);
    env->reward_food = unpack_with_index(kwargs, "reward_food", env_index);
    env->reward_corpse = unpack_with_index(kwargs, "reward_corpse", env_index);
    env->reward_death = unpack_with_index(kwargs, "reward_death", env_index);
    env->max_snake_length = unpack_with_index(kwargs, "max_snake_length", env_index);
    env->cell_size = unpack_with_index(kwargs, "cell_size", env_index);
    
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
