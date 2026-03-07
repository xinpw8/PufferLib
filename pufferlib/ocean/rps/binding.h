#include "rps.h"
#define OBS_SIZE 12
#define NUM_ATNS 2
#define ACT_SIZES {3, 3}
#define OBS_TYPE FLOAT
#define ACT_TYPE DOUBLE

#define Env CRPS
#include "env_binding.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    DictItem* mt = dict_get_unsafe(kwargs, "max_ticks");
    env->max_ticks = mt ? (int)mt->value : 100;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n", log->n);
}
