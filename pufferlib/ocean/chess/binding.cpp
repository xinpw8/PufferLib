// binding.c
#include "chess.h"
#define Env CChess
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->reward_valid   = unpack(kwargs,"reward_valid");
    env->reward_invalid = unpack(kwargs,"reward_invalid");
    env->reward_agent_captures_enemy_piece = unpack(kwargs,"reward_agent_captures_enemy_piece");
    env->reward_enemy_captures_agent_piece = unpack(kwargs,"reward_enemy_captures_agent_piece");
    env->reward_win = unpack(kwargs,"reward_win");
    env->reward_draw = unpack(kwargs,"reward_draw");
    env->reward_loss = unpack(kwargs,"reward_loss");
    
    init(env); // alloc & new ChessContext
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "reward_valid", log->reward_valid);
    assign_to_dict(dict, "reward_invalid", log->reward_invalid);
    assign_to_dict(dict, "reward_agent_captures_enemy_piece", log->reward_agent_captures_enemy_piece);
    assign_to_dict(dict, "reward_enemy_captures_agent_piece", log->reward_enemy_captures_agent_piece);
    assign_to_dict(dict, "reward_win", log->reward_win);
    assign_to_dict(dict, "reward_draw", log->reward_draw);
    assign_to_dict(dict, "reward_loss", log->reward_loss);
    assign_to_dict(dict, "game_won", log->game_won);
    assign_to_dict(dict, "game_lost", log->game_lost);
    assign_to_dict(dict, "game_drawn", log->game_drawn);
    assign_to_dict(dict, "n", log->n);
    return 0;
}