#include <Python.h>
#include "poke_battle.h"

#define Env PokeBattle
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->num_agents = (int)unpack(kwargs, "num_agents");
    env->seed = (unsigned long long)unpack(kwargs, "seed");
    env->selfplay = (int)unpack(kwargs, "selfplay");
    env->learner_side = (int)unpack(kwargs, "learner_side");
    env->bot_mode = (int)unpack(kwargs, "bot_mode");
    env->mcts_iterations = (int)unpack(kwargs, "mcts_iterations");
    env->mcts_depth = (int)unpack(kwargs, "mcts_depth");
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "p1_wins", log->p1_wins);
    assign_to_dict(dict, "p2_wins", log->p2_wins);
    assign_to_dict(dict, "draws", log->draws);
    return 0;
}
