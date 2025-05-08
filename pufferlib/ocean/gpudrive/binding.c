#include "gpudrive.h"
#define Env GPUDrive
#define MY_SHARED
#include "../env_binding.h"

static PyObject* my_shared(PyObject* self, PyObject* args, PyObject* kwargs) {
    int num_envs = unpack(kwargs, "num_envs");
    GPUDrive* temp_envs = calloc(num_envs, sizeof(GPUDrive));
    PyObject* agent_offsets = PyList_New(num_envs+1);
    int total_count = 0;
    // getting  agent counts and offsets
    for(int i = 0;i< num_envs;i++) {
        char map_file[100];
        sprintf(map_file, "resources/gpudrive/binaries/map_%03d.bin", i);
        temp_envs[i].entities = load_map_binary(map_file, &temp_envs[i]);
        set_active_agents(&temp_envs[i]);
        PyObject* num = PyLong_FromLong(total_count);
        PyList_SetItem(agent_offsets, i, num);
        //Py_DECREF(num);
        total_count += temp_envs[i].active_agent_count;
    }
    PyObject* num = PyLong_FromLong(total_count);
    PyList_SetItem(agent_offsets, num_envs, num);
    //Py_DECREF(num);
    /*
    for(int i = 0;i<num_envs; i++) {
        for(int j=0;j<temp_envs[i].num_entities;j++) {
            free_entity(&temp_envs[i].entities[j]);
        }
        free(temp_envs[i].entities);
        free(temp_envs[i].active_agent_indices);
        free(temp_envs[i].static_car_indices);
    }
    free(temp_envs);
    */
    return agent_offsets;
}

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->human_agent_idx = unpack(kwargs, "human_agent_idx");
    env->reward_vehicle_collision = unpack(kwargs, "reward_vehicle_collision");
    env->reward_offroad_collision = unpack(kwargs, "reward_offroad_collision");
    int env_id = unpack(kwargs, "env_id");

    char map_file[100];
    sprintf(map_file, "resources/gpudrive/binaries/map_%03d.bin", env_id);
    env->map_name = map_file;
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "offroad_rate", log->offroad_rate);
    assign_to_dict(dict, "collision_rate", log->collision_rate);
    assign_to_dict(dict, "dnf_rate", log->dnf_rate);
    assign_to_dict(dict, "n", log->n);
    assign_to_dict(dict, "completion_rate", log->completion_rate);
    return 0;
}
