// binding.cpp
#pragma once
#include <Python.h> // makes PyObject visible
#include "chess.h"
#include "stockfish_wrapper.h"
#define Env CChess

// forward-declare so env_binding.h can use it
static PyObject* env_set_self_play(PyObject* self, PyObject* args);
static PyObject* vec_set_self_play(PyObject* self, PyObject* args);
static PyObject* env_set_fen(PyObject* self, PyObject* args);
static PyObject* vec_set_fen(PyObject* self, PyObject* args);
static PyObject* vec_enable_stockfish_black(PyObject* self, PyObject* args);

// Define custom methods for chess module
#define MY_METHODS \
    {"env_set_self_play", env_set_self_play, METH_VARARGS, "Enable self-play mode"}, \
    {"vec_set_self_play", vec_set_self_play, METH_VARARGS, "Enable self-play mode for vector env"}, \
    {"env_set_fen", env_set_fen, METH_VARARGS, "Load a FEN into a single env"}, \
    {"vec_set_fen", vec_set_fen, METH_VARARGS, "Load a FEN into every env in a VecEnv"}, \
    {"vec_enable_stockfish_black", vec_enable_stockfish_black, METH_VARARGS, "Enable Stockfish for all environments in a VecEnv"}, \
    {NULL, NULL, 0, NULL}

#include "../env_binding.h"

static PyObject* env_set_self_play(PyObject* self, PyObject* args) {
    Env* env = unpack_env(args);
    if (!env) return NULL;
    set_self_play_mode(env, true);
    Py_RETURN_NONE;
}

// Enable self-play for every env in a VecEnv
static PyObject* vec_set_self_play(PyObject* self, PyObject* args) {
    VecEnv* vec = unpack_vecenv(args);
    if (!vec) return NULL;
    for (int i = 0; i < vec->num_envs; ++i)
        set_self_play_mode(vec->envs[i], true);
    Py_RETURN_NONE;
}

// FEN setters
static PyObject* env_set_fen(PyObject* self, PyObject* args) {
    Env* env = unpack_env(args);
    if (!env) return NULL;

    const char* fen = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
    if (!fen) return NULL;

    c_set_fen(env, fen);
    Py_RETURN_NONE;
}

static PyObject* vec_set_fen(PyObject* self, PyObject* args) {
    VecEnv* vec = unpack_vecenv(args);
    if (!vec) return NULL;

    const char* fen = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
    if (!fen) return NULL;

    // apply to every env in the vector
    for (int i = 0; i < vec->num_envs; ++i)
        c_set_fen(vec->envs[i], fen);

    Py_RETURN_NONE;
}

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->reward_valid = unpack(kwargs, "reward_valid");
    env->reward_invalid = unpack(kwargs, "reward_invalid");
    env->reward_agent_captures_enemy_piece = unpack(kwargs, "reward_agent_captures_enemy_piece");
    env->reward_enemy_captures_agent_piece = unpack(kwargs, "reward_enemy_captures_agent_piece");
    env->reward_win = unpack(kwargs, "reward_win");
    env->reward_draw = unpack(kwargs, "reward_draw");
    env->reward_loss = unpack(kwargs, "reward_loss");
    env->reward_check = unpack(kwargs, "reward_check");
    env->reward_material_diff = unpack(kwargs, "reward_material_diff");
    env->max_depth = (int)unpack(kwargs, "max_depth");
    init(env);    
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_return_white", log->episode_return_white);
    assign_to_dict(dict, "episode_return_black", log->episode_return_black);
    assign_to_dict(dict, "reward_valid", log->reward_valid);
    assign_to_dict(dict, "reward_invalid", log->reward_invalid);
    assign_to_dict(dict, "reward_agent_captures_enemy_piece", log->reward_agent_captures_enemy_piece);
    assign_to_dict(dict, "reward_enemy_captures_agent_piece", log->reward_enemy_captures_agent_piece);
    assign_to_dict(dict, "reward_win", log->reward_win);
    assign_to_dict(dict, "reward_draw", log->reward_draw);
    assign_to_dict(dict, "reward_loss", log->reward_loss);
    assign_to_dict(dict, "reward_win_white", log->reward_win_white);
    assign_to_dict(dict, "reward_win_black", log->reward_win_black);
    assign_to_dict(dict, "reward_loss_white", log->reward_loss_white);
    assign_to_dict(dict, "reward_loss_black", log->reward_loss_black);
    assign_to_dict(dict, "reward_draw_white", log->reward_draw_white);
    assign_to_dict(dict, "reward_draw_black", log->reward_draw_black);
    assign_to_dict(dict, "game_won", log->game_won);
    assign_to_dict(dict, "game_lost", log->game_lost);
    assign_to_dict(dict, "game_drawn", log->game_drawn);
    assign_to_dict(dict, "white_win", log->white_win);
    assign_to_dict(dict, "white_loss", log->white_loss);
    assign_to_dict(dict, "black_win", log->black_win);
    assign_to_dict(dict, "black_loss", log->black_loss);
    assign_to_dict(dict, "stalemate", log->stalemate);
    assign_to_dict(dict, "insufficient_material", log->insufficient_material);
    assign_to_dict(dict, "threefold_repetition", log->threefold_repetition);
    assign_to_dict(dict, "fifty_move_rule", log->fifty_move_rule);
    assign_to_dict(dict, "max_depth", log->max_depth);
    assign_to_dict(dict, "white_checkmated", log->white_checkmated);
    assign_to_dict(dict, "black_checkmated", log->black_checkmated);
    assign_to_dict(dict, "white_moves", log->white_moves);
    assign_to_dict(dict, "black_moves", log->black_moves);
    assign_to_dict(dict, "valid_moves", log->valid_moves);
    assign_to_dict(dict, "invalid_moves", log->invalid_moves);
    assign_to_dict(dict, "invalid_moves_white", log->invalid_moves_white);
    assign_to_dict(dict, "invalid_moves_black", log->invalid_moves_black);
    assign_to_dict(dict, "reward_check", log->reward_check);
    assign_to_dict(dict, "reward_material_diff", log->reward_material_diff);
    assign_to_dict(dict, "stockfish_eval", log->stockfish_eval);
    assign_to_dict(dict, "en_passant_white", log->en_passant_white);
    assign_to_dict(dict, "en_passant_black", log->en_passant_black);
    assign_to_dict(dict, "white_castle_kingside", log->white_castle_kingside);
    assign_to_dict(dict, "white_castle_queenside", log->white_castle_queenside);
    assign_to_dict(dict, "black_castle_kingside", log->black_castle_kingside);
    assign_to_dict(dict, "black_castle_queenside", log->black_castle_queenside);
    assign_to_dict(dict, "white_promotion_count", log->white_promotion_count);
    assign_to_dict(dict, "white_promotion_knight", log->white_promotion_knight);
    assign_to_dict(dict, "white_promotion_bishop", log->white_promotion_bishop);
    assign_to_dict(dict, "white_promotion_rook", log->white_promotion_rook);
    assign_to_dict(dict, "white_promotion_queen", log->white_promotion_queen);
    assign_to_dict(dict, "black_promotion_count", log->black_promotion_count);
    assign_to_dict(dict, "black_promotion_knight", log->black_promotion_knight);
    assign_to_dict(dict, "black_promotion_bishop", log->black_promotion_bishop);
    assign_to_dict(dict, "black_promotion_rook", log->black_promotion_rook);
    assign_to_dict(dict, "black_promotion_queen", log->black_promotion_queen);
    
    assign_to_dict(dict, "n", log->n);
    return 0;
}

// // Provide enable_stockfish_black here so the Python extension has the symbol
// extern "C" void enable_stockfish_black(CChess* env, const char* stockfish_cmd, int elo, int search_ms) {
//     if (!env) return;
//     ChessContext* ctx = (ChessContext*)env->context;
//     if (!ctx) return;

//     if (ctx->sf) {
//         delete ctx->sf;
//         ctx->sf = nullptr;
//     }

//     // Resolve Stockfish binary path
//     const char* cmd = nullptr;

//     // 1) Caller-supplied path takes highest priority
//     if (stockfish_cmd && stockfish_cmd[0]) {
//         cmd = stockfish_cmd;
//     }

//     // 2) Search common bundled locations
//     if (!cmd) {
//         const char* candidates[] = {
//             "pufferlib/Stockfish/src/stockfish",
//             "./pufferlib/Stockfish/src/stockfish",
//             "Stockfish/src/stockfish",
//             "./Stockfish/src/stockfish",
//             "stockfish",
//             nullptr
//         };
//         for (int i = 0; candidates[i]; ++i) {
//             if (access(candidates[i], X_OK) == 0) {
//                 cmd = candidates[i];
//                 break;
//             }
//         }
//     }

//     // 3) Fallback to plain "stockfish" if nothing found
//     if (!cmd) cmd = "stockfish";

//     ctx->sf = new Stockfish(cmd, elo, search_ms);
//     ctx->stockfish_enabled = ctx->sf && ctx->sf->ok();
// }

static PyObject* vec_enable_stockfish_black(PyObject* self, PyObject* args) {
    VecEnv* vec = unpack_vecenv(args);
    if (!vec) return NULL;

    const char* stockfish_cmd = nullptr;
    int elo = 1320;
    int search_ms = 10;
    
    // Parse optional arguments
    if (PyTuple_Size(args) >= 2) {
        PyObject* cmd_obj = PyTuple_GetItem(args, 1);
        if (cmd_obj != Py_None) {
            stockfish_cmd = PyUnicode_AsUTF8(cmd_obj);
            if (!stockfish_cmd) return NULL;
        }
    }
    if (PyTuple_Size(args) >= 3) {
        elo = PyLong_AsLong(PyTuple_GetItem(args, 2));
        if (PyErr_Occurred()) return NULL;
    }
    if (PyTuple_Size(args) >= 4) {
        search_ms = PyLong_AsLong(PyTuple_GetItem(args, 3));
        if (PyErr_Occurred()) return NULL;
    }
    
    for (int i = 0; i < vec->num_envs; ++i) {
        enable_stockfish_black(vec->envs[i], stockfish_cmd, elo, search_ms);
    }

    Py_RETURN_NONE;
}
