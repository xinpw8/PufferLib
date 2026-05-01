#include "mancala.h"

#define OBS_SIZE OBS_DIM
#define NUM_ATNS 1
#define ACT_SIZES {NUM_PITS}
#define OBS_TENSOR_T FloatTensor

#define Env CMancala
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    (void)kwargs;
    env->num_agents = 1;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "margin", log->margin);
    dict_set(out, "captures", log->captures);
    dict_set(out, "extra_turns", log->extra_turns);
    dict_set(out, "invalid_moves", log->invalid_moves);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n", log->n);
}
