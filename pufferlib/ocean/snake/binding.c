#include "snake.h"
#define Env CSnake
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    printf("my_init called\n");
    
    long width = unpack(kwargs, "width");
    printf("width = %ld\n", width);
    
    long height = unpack(kwargs, "height");
    printf("height = %ld\n", height);
    
    long num_snakes = unpack(kwargs, "num_snakes");
    printf("num_snakes = %ld\n", num_snakes);
    
    long num_food = unpack(kwargs, "num_food");
    printf("num_food = %ld\n", num_food);
    
    long vision = unpack(kwargs, "vision");
    printf("vision = %ld\n", vision);
    
    // Validate width and height based on vision
    long min_dimension = 2 * vision + 2;
    if (width < min_dimension || height < min_dimension) {
        char error_msg[200];
        snprintf(error_msg, sizeof(error_msg), 
                "Width and height must be at least 2*vision+2. Got width=%ld, height=%ld, vision=%ld, minimum required=%ld", 
                width, height, vision, min_dimension);
        PyErr_SetString(PyExc_ValueError, error_msg);
        return 1;
    }
    
    long max_snake_length = unpack(kwargs, "max_snake_length");
    printf("max_snake_length = %ld\n", max_snake_length);
    
    long leave_corpse_on_death_long = unpack(kwargs, "leave_corpse_on_death");
    printf("leave_corpse_on_death = %ld\n", leave_corpse_on_death_long);
    
    double reward_food = unpack(kwargs, "reward_food");
    printf("reward_food = %f\n", reward_food);
    
    double reward_corpse = unpack(kwargs, "reward_corpse");
    printf("reward_corpse = %f\n", reward_corpse);
    
    double reward_death = unpack(kwargs, "reward_death");
    printf("reward_death = %f\n", reward_death);
    
    double survival_reward = unpack(kwargs, "survival_reward");
    printf("survival_reward = %f\n", survival_reward);

    if (width < 0 || height < 0 || num_snakes < 0 || num_food < 0 || vision < 0 ||
        max_snake_length < 0 || leave_corpse_on_death_long < 0 ||
        reward_food < -1e9 || reward_corpse < -1e9 || reward_death < -1e9 || 
        survival_reward < -1e9) { // unpack returns -1e10 on error
         PyErr_SetString(PyExc_TypeError, "Failed to parse one or more required keyword arguments for Snake init");
         return 1;
    }

    env->width = (int)width;
    env->height = (int)height;
    env->num_snakes = (int)num_snakes;
    env->food = (int)num_food;
    env->vision = (int)vision;
    env->max_snake_length = (int)max_snake_length;
    env->leave_corpse_on_death = (unsigned char)leave_corpse_on_death_long;
    env->reward_food = (float)reward_food;
    env->reward_corpse = (float)reward_corpse;
    env->reward_death = (float)reward_death;
    env->survival_reward = (float)survival_reward;

    printf("About to call allocate_csnake\n");
    allocate_csnake(env);
    printf("About to call c_reset\n");
    c_reset(env);
    printf("my_init completed successfully\n");

    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    if (assign_to_dict(dict, "episode_return", log->episode_return)) return 1;
    if (assign_to_dict(dict, "episode_length", log->episode_length)) return 1;
    if (assign_to_dict(dict, "score", log->score)) return 1;
    return 0; 
}
