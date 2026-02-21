#include "slimevolley.h"
// Selfplay mode: doubled obs [learner(12) | opponent(12)], interleaved actions [learner(3) | opponent(3)]
#define OBS_SIZE 24
#define NUM_ATNS 6
#define ACT_SIZES {2, 2, 2, 2, 2, 2}
#define OBS_TYPE FLOAT
#define ACT_TYPE DOUBLE

#define Env SlimeVolley
#include "env_binding.h"

void my_init(Env* env, Dict* kwargs) {
    DictItem* sp = dict_get_unsafe(kwargs, "selfplay");
    env->selfplay = sp ? (int)sp->value : 0;
    env->num_agents = 1;  // Always 1 logical agent with doubled obs layout
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
