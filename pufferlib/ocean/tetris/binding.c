#include "tetris.h"

#define Env Tetris
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->n_rows = unpack(kwargs, "n_rows");
    env->n_cols = unpack(kwargs, "n_cols");
    env->deck_size = unpack(kwargs, "deck_size");
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "ep_length", log->ep_length);
    assign_to_dict(dict, "ep_return", log->ep_return);
    assign_to_dict(dict, "avg_combo", log->avg_combo);
    assign_to_dict(dict, "lines_deleted", log->lines_deleted);

    assign_to_dict(dict, "atn_frac_soft_drop", log->atn_frac_soft_drop);
    assign_to_dict(dict, "atn_frac_hard_drop", log->atn_frac_hard_drop);
    assign_to_dict(dict, "atn_frac_rotate", log->atn_frac_rotate);
    return 0;
}