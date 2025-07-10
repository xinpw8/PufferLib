// binding.cpp
#pragma once
#include <Python.h> // makes PyObject visible
#include "chess.h"
#include "stockfish_wrapper.h"
#define Env CChess

// forward-declare so env_binding.h can use it
static PyObject* env_set_self_play(PyObject* self, PyObject* args);
static PyObject* vec_set_self_play(PyObject* self, PyObject* args);
static PyObject* env_set_dual_agent_self_play(PyObject* self, PyObject* args);
static PyObject* vec_set_dual_agent_self_play(PyObject* self, PyObject* args);
static PyObject* env_set_fen(PyObject* self, PyObject* args);
static PyObject* vec_set_fen(PyObject* self, PyObject* args);
static PyObject* vec_enable_stockfish_black(PyObject* self, PyObject* args);

// Define custom methods for chess module
#define MY_METHODS \
    {"env_set_self_play", env_set_self_play, METH_VARARGS, "Enable self-play mode"}, \
    {"vec_set_self_play", vec_set_self_play, METH_VARARGS, "Enable self-play mode for vector env"}, \
    {"env_set_dual_agent_self_play", env_set_dual_agent_self_play, METH_VARARGS, "Enable dual agent self-play mode"}, \
    {"vec_set_dual_agent_self_play", vec_set_dual_agent_self_play, METH_VARARGS, "Enable dual agent self-play mode for vector env"}, \
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

// Enable dual agent self-play for a single env
static PyObject* env_set_dual_agent_self_play(PyObject* self, PyObject* args) {
    Env* env = unpack_env(args);
    if (!env) return NULL;
    set_dual_agent_self_play_mode(env, true);
    Py_RETURN_NONE;
}

// Enable dual agent self-play for every env in a VecEnv
static PyObject* vec_set_dual_agent_self_play(PyObject* self, PyObject* args) {
    VecEnv* vec = unpack_vecenv(args);
    if (!vec) return NULL;
    for (int i = 0; i < vec->num_envs; ++i)
        set_dual_agent_self_play_mode(vec->envs[i], true);
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
    env->reward_invalid_white = unpack(kwargs, "reward_invalid_white");
    env->reward_invalid_black = unpack(kwargs, "reward_invalid_black");
    env->reward_agent_captures_enemy_piece = unpack(kwargs, "reward_agent_captures_enemy_piece");
    env->reward_enemy_captures_agent_piece = unpack(kwargs, "reward_enemy_captures_agent_piece");
    env->reward_draw = unpack(kwargs, "reward_draw");
    env->reward_win_white = unpack(kwargs, "reward_win_white");
    env->reward_win_black = unpack(kwargs, "reward_win_black");
    env->reward_loss_white = unpack(kwargs, "reward_loss_white");
    env->reward_loss_black = unpack(kwargs, "reward_loss_black");
    env->reward_check_white = unpack(kwargs, "reward_check_white");
    env->reward_check_black = unpack(kwargs, "reward_check_black");
    env->reward_material_diff_white = unpack(kwargs, "reward_material_diff_white");
    env->reward_material_diff_black = unpack(kwargs, "reward_material_diff_black");
    env->max_depth = (int)unpack(kwargs, "max_depth");
    env->debug_disable_mask = (bool)unpack(kwargs, "debug_disable_mask");  // FIX: Pass debug flag to C++
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
    assign_to_dict(dict, "reward_agent_captures_enemy_piece", log->reward_agent_captures_enemy_piece);
    assign_to_dict(dict, "reward_enemy_captures_agent_piece", log->reward_enemy_captures_agent_piece);
    assign_to_dict(dict, "reward_draw", log->reward_draw);
    assign_to_dict(dict, "reward_win_white", log->reward_win_white);
    assign_to_dict(dict, "reward_win_black", log->reward_win_black);
    assign_to_dict(dict, "reward_loss_white", log->reward_loss_white);
    assign_to_dict(dict, "reward_loss_black", log->reward_loss_black);
    assign_to_dict(dict, "reward_draw_white", log->reward_draw_white);
    assign_to_dict(dict, "reward_draw_black", log->reward_draw_black);
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
    assign_to_dict(dict, "invalid_moves_white", log->invalid_moves_white);
    assign_to_dict(dict, "invalid_moves_black", log->invalid_moves_black);
    assign_to_dict(dict, "reward_check_white", log->reward_check_white);
    assign_to_dict(dict, "reward_check_black", log->reward_check_black);
    assign_to_dict(dict, "reward_material_diff_white", log->reward_material_diff_white);
    assign_to_dict(dict, "reward_material_diff_black", log->reward_material_diff_black);
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
    
    // Game logging fields
    assign_to_dict(dict, "last_move_from", log->last_move_from);
    assign_to_dict(dict, "last_move_to", log->last_move_to);
    assign_to_dict(dict, "last_move_promotion", log->last_move_promotion);
    assign_to_dict(dict, "game_step_logged", log->game_step_logged);
    assign_to_dict(dict, "game_moves_count", log->game_moves_count);
    
    // Add complete game logging fields
    assign_to_dict(dict, "complete_game_move_count", log->complete_game_move_count);
    assign_to_dict(dict, "complete_game_action_0", log->complete_game_action_0);
    assign_to_dict(dict, "complete_game_action_1", log->complete_game_action_1);
    assign_to_dict(dict, "complete_game_action_2", log->complete_game_action_2);
    assign_to_dict(dict, "complete_game_action_3", log->complete_game_action_3);
    assign_to_dict(dict, "complete_game_action_4", log->complete_game_action_4);
    assign_to_dict(dict, "complete_game_action_5", log->complete_game_action_5);
    assign_to_dict(dict, "complete_game_action_6", log->complete_game_action_6);
    assign_to_dict(dict, "complete_game_action_7", log->complete_game_action_7);
    assign_to_dict(dict, "complete_game_action_8", log->complete_game_action_8);
    assign_to_dict(dict, "complete_game_action_9", log->complete_game_action_9);
    assign_to_dict(dict, "complete_game_action_10", log->complete_game_action_10);
    assign_to_dict(dict, "complete_game_action_11", log->complete_game_action_11);
    assign_to_dict(dict, "complete_game_action_12", log->complete_game_action_12);
    assign_to_dict(dict, "complete_game_action_13", log->complete_game_action_13);
    assign_to_dict(dict, "complete_game_action_14", log->complete_game_action_14);
    assign_to_dict(dict, "complete_game_action_15", log->complete_game_action_15);
    assign_to_dict(dict, "complete_game_action_16", log->complete_game_action_16);
    assign_to_dict(dict, "complete_game_action_17", log->complete_game_action_17);
    assign_to_dict(dict, "complete_game_action_18", log->complete_game_action_18);
    assign_to_dict(dict, "complete_game_action_19", log->complete_game_action_19);
    assign_to_dict(dict, "complete_game_action_20", log->complete_game_action_20);
    assign_to_dict(dict, "complete_game_action_21", log->complete_game_action_21);
    assign_to_dict(dict, "complete_game_action_22", log->complete_game_action_22);
    assign_to_dict(dict, "complete_game_action_23", log->complete_game_action_23);
    assign_to_dict(dict, "complete_game_action_24", log->complete_game_action_24);
    assign_to_dict(dict, "complete_game_action_25", log->complete_game_action_25);
    assign_to_dict(dict, "complete_game_action_26", log->complete_game_action_26);
    assign_to_dict(dict, "complete_game_action_27", log->complete_game_action_27);
    assign_to_dict(dict, "complete_game_action_28", log->complete_game_action_28);
    assign_to_dict(dict, "complete_game_action_29", log->complete_game_action_29);
    assign_to_dict(dict, "complete_game_action_30", log->complete_game_action_30);
    assign_to_dict(dict, "complete_game_action_31", log->complete_game_action_31);
    assign_to_dict(dict, "complete_game_action_32", log->complete_game_action_32);
    assign_to_dict(dict, "complete_game_action_33", log->complete_game_action_33);
    assign_to_dict(dict, "complete_game_action_34", log->complete_game_action_34);
    assign_to_dict(dict, "complete_game_action_35", log->complete_game_action_35);
    assign_to_dict(dict, "complete_game_action_36", log->complete_game_action_36);
    assign_to_dict(dict, "complete_game_action_37", log->complete_game_action_37);
    assign_to_dict(dict, "complete_game_action_38", log->complete_game_action_38);
    assign_to_dict(dict, "complete_game_action_39", log->complete_game_action_39);
    assign_to_dict(dict, "complete_game_action_40", log->complete_game_action_40);
    assign_to_dict(dict, "complete_game_action_41", log->complete_game_action_41);
    assign_to_dict(dict, "complete_game_action_42", log->complete_game_action_42);
    assign_to_dict(dict, "complete_game_action_43", log->complete_game_action_43);
    assign_to_dict(dict, "complete_game_action_44", log->complete_game_action_44);
    assign_to_dict(dict, "complete_game_action_45", log->complete_game_action_45);
    assign_to_dict(dict, "complete_game_action_46", log->complete_game_action_46);
    assign_to_dict(dict, "complete_game_action_47", log->complete_game_action_47);
    assign_to_dict(dict, "complete_game_action_48", log->complete_game_action_48);
    assign_to_dict(dict, "complete_game_action_49", log->complete_game_action_49);
    assign_to_dict(dict, "complete_game_action_50", log->complete_game_action_50);
    assign_to_dict(dict, "complete_game_action_51", log->complete_game_action_51);
    assign_to_dict(dict, "complete_game_action_52", log->complete_game_action_52);
    assign_to_dict(dict, "complete_game_action_53", log->complete_game_action_53);
    assign_to_dict(dict, "complete_game_action_54", log->complete_game_action_54);
    assign_to_dict(dict, "complete_game_action_55", log->complete_game_action_55);
    assign_to_dict(dict, "complete_game_action_56", log->complete_game_action_56);
    assign_to_dict(dict, "complete_game_action_57", log->complete_game_action_57);
    assign_to_dict(dict, "complete_game_action_58", log->complete_game_action_58);
    assign_to_dict(dict, "complete_game_action_59", log->complete_game_action_59);
    assign_to_dict(dict, "complete_game_action_60", log->complete_game_action_60);
    assign_to_dict(dict, "complete_game_action_61", log->complete_game_action_61);
    assign_to_dict(dict, "complete_game_action_62", log->complete_game_action_62);
    assign_to_dict(dict, "complete_game_action_63", log->complete_game_action_63);
    assign_to_dict(dict, "complete_game_action_64", log->complete_game_action_64);
    assign_to_dict(dict, "complete_game_action_65", log->complete_game_action_65);
    assign_to_dict(dict, "complete_game_action_66", log->complete_game_action_66);
    assign_to_dict(dict, "complete_game_action_67", log->complete_game_action_67);
    assign_to_dict(dict, "complete_game_action_68", log->complete_game_action_68);
    assign_to_dict(dict, "complete_game_action_69", log->complete_game_action_69);
    assign_to_dict(dict, "complete_game_action_70", log->complete_game_action_70);
    assign_to_dict(dict, "complete_game_action_71", log->complete_game_action_71);
    assign_to_dict(dict, "complete_game_action_72", log->complete_game_action_72);
    assign_to_dict(dict, "complete_game_action_73", log->complete_game_action_73);
    assign_to_dict(dict, "complete_game_action_74", log->complete_game_action_74);
    assign_to_dict(dict, "complete_game_action_75", log->complete_game_action_75);
    assign_to_dict(dict, "complete_game_action_76", log->complete_game_action_76);
    assign_to_dict(dict, "complete_game_action_77", log->complete_game_action_77);
    assign_to_dict(dict, "complete_game_action_78", log->complete_game_action_78);
    assign_to_dict(dict, "complete_game_action_79", log->complete_game_action_79);
    assign_to_dict(dict, "complete_game_action_80", log->complete_game_action_80);
    assign_to_dict(dict, "complete_game_action_81", log->complete_game_action_81);
    assign_to_dict(dict, "complete_game_action_82", log->complete_game_action_82);
    assign_to_dict(dict, "complete_game_action_83", log->complete_game_action_83);
    assign_to_dict(dict, "complete_game_action_84", log->complete_game_action_84);
    assign_to_dict(dict, "complete_game_action_85", log->complete_game_action_85);
    assign_to_dict(dict, "complete_game_action_86", log->complete_game_action_86);
    assign_to_dict(dict, "complete_game_action_87", log->complete_game_action_87);
    assign_to_dict(dict, "complete_game_action_88", log->complete_game_action_88);
    assign_to_dict(dict, "complete_game_action_89", log->complete_game_action_89);
    assign_to_dict(dict, "complete_game_action_90", log->complete_game_action_90);
    assign_to_dict(dict, "complete_game_action_91", log->complete_game_action_91);
    assign_to_dict(dict, "complete_game_action_92", log->complete_game_action_92);
    assign_to_dict(dict, "complete_game_action_93", log->complete_game_action_93);
    assign_to_dict(dict, "complete_game_action_94", log->complete_game_action_94);
    assign_to_dict(dict, "complete_game_action_95", log->complete_game_action_95);
    assign_to_dict(dict, "complete_game_action_96", log->complete_game_action_96);
    assign_to_dict(dict, "complete_game_action_97", log->complete_game_action_97);
    assign_to_dict(dict, "complete_game_action_98", log->complete_game_action_98);
    assign_to_dict(dict, "complete_game_action_99", log->complete_game_action_99);

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
