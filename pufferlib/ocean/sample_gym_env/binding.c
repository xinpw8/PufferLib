/* binding.c
 * Python C API binding for SampleGymEnv
 * This file binds the C environment to Python for use with PufferLib
 */

#include "sample_gym_env.h"

#define Env SampleGymEnv
#include "../env_binding.h"

// Initialize environment with Python arguments
static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    // Extract size parameter from Python kwargs
    // Default to 11 if not specified
    env->size = kwargs ? unpack(kwargs, "size") : 11;
    
    // Validate size parameter
    if (env->size < 5 || env->size > 50) {
        PyErr_SetString(PyExc_ValueError, "Environment size must be between 5 and 50");
        return -1;
    }
    
    return 0;
}

// Export log data back to Python
static int my_log(PyObject* dict, Log* log) {
    // Convert C log struct to Python dictionary
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    
    // Additional custom metrics
    assign_to_dict(dict, "episodes_completed", log->n);
    
    return 0;
}