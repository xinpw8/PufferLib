#include "password.h"
#define OBS_SIZE LENGTH
#define NUM_ATNS 1
#define ACT_SIZES {NUM_DIGITS}
#define OBS_TENSOR_T ByteTensor
#define PUFFER_HAS_STATE 1
#define PUFFER_STATE_REFRESH(env) refresh_state(env)
#define PUFFER_STATE_SCORE(env) ((env)->state.pos)

#define Env Password
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    (void)kwargs;
    env->num_agents = 1;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "success", log->success);
}
