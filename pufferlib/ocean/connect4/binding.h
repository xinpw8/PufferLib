#include "connect4.h"

#define OBS_SIZE 84
#define NUM_ATNS 2
#define ACT_SIZES {7, 7}
#define OBS_TYPE FLOAT
#define ACT_TYPE DOUBLE
#define Env CConnect4
#include "env_binding.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    DictItem* sp = dict_get_unsafe(kwargs, "selfplay");
    env->selfplay = sp ? (int)sp->value : 0;
    // Don't call init() here - obs pointers not assigned yet
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n", log->n);
}
