/*
 * Diablo C binding for PufferLib
 *
 * This file provides the Python bindings for the Diablo environment.
 * The game launches as a separate process; this binding handles
 * the shared memory interface.
 */

#include "diablo.h"

#define Env Diablo
#define MY_PUT
#include "../env_binding.h"

/* Helper to get string from kwargs */
static const char* unpack_string(PyObject* kwargs, const char* key) {
    PyObject* val = PyDict_GetItemString(kwargs, key);
    if (val == NULL) {
        return NULL;
    }
    if (!PyUnicode_Check(val)) {
        return NULL;
    }
    return PyUnicode_AsUTF8(val);
}

/* Custom put function to set up mmap after game launches */
static int my_put(Env* env, PyObject* args, PyObject* kwargs) {
    /* Get mmap path and offset from kwargs */
    const char* mmap_path = unpack_string(kwargs, "mmap_path");
    if (mmap_path == NULL) {
        PyErr_SetString(PyExc_TypeError, "mmap_path is required");
        return 1;
    }

    PyObject* base_obj = PyDict_GetItemString(kwargs, "base_offset");
    uint64_t base_offset = 0;
    if (base_obj && PyLong_Check(base_obj)) {
        base_offset = PyLong_AsUnsignedLongLong(base_obj);
    }

    /* Initialize mmap */
    if (init_mmap(env, mmap_path, base_offset) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to mmap shared memory");
        return 1;
    }

    /* Set goal position if provided */
    PyObject* goal_x_obj = PyDict_GetItemString(kwargs, "goal_x");
    PyObject* goal_y_obj = PyDict_GetItemString(kwargs, "goal_y");
    if (goal_x_obj && goal_y_obj) {
        env->goal_x = (int32_t)PyLong_AsLong(goal_x_obj);
        env->goal_y = (int32_t)PyLong_AsLong(goal_y_obj);
    }

    /* Set step mode flag */
    PyObject* step_mode_obj = PyDict_GetItemString(kwargs, "step_mode");
    if (step_mode_obj) {
        env->step_mode = PyObject_IsTrue(step_mode_obj);
    }

    return 0;
}

/* Initialize environment with parameters */
static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    /* Get configuration parameters */
    env->view_radius = (int)unpack(kwargs, "view_radius");
    if (PyErr_Occurred()) {
        env->view_radius = VIEW_RADIUS;  /* Default */
        PyErr_Clear();
    }

    env->max_steps = (int)unpack(kwargs, "max_steps");
    if (PyErr_Occurred()) {
        env->max_steps = 10000;  /* Default */
        PyErr_Clear();
    }

    env->game_ticks_per_step = (int)unpack(kwargs, "game_ticks_per_step");
    if (PyErr_Occurred()) {
        env->game_ticks_per_step = 10;  /* Default */
        PyErr_Clear();
    }

    /* Initialize state */
    env->mmap_base = NULL;
    env->mmap_fd = -1;
    env->mmap_size = 0;
    env->initialized = 0;
    env->ep_return = 0.0f;
    env->ep_len = 0;
    env->goal_x = 0;
    env->goal_y = 0;
    env->step_mode = 1;  /* Default to step mode */

    memset(&env->log, 0, sizeof(Log));

    return 0;
}

/* Export log values to Python dict */
static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "success_rate", log->success_rate);
    return 0;
}
