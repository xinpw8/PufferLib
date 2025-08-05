// // binding.cpp
// #pragma once
// #include <Python.h> // makes PyObject visible
// #include "chess.h"
// #include "stockfish_wrapper.h"
// #define Env CChess

// // forward-declare so env_binding.h can use it
// static PyObject* env_set_self_play(PyObject* self, PyObject* args);
// static PyObject* vec_set_self_play(PyObject* self, PyObject* args);
// static PyObject* env_set_dual_agent_self_play(PyObject* self, PyObject* args);
// static PyObject* vec_set_dual_agent_self_play(PyObject* self, PyObject* args);
// static PyObject* env_set_fen(PyObject* self, PyObject* args);
// static PyObject* vec_set_fen(PyObject* self, PyObject* args);
// static PyObject* vec_enable_stockfish_black(PyObject* self, PyObject* args);
// static PyObject* env_set_puzzle_mode(PyObject* self, PyObject* args);
// static PyObject* vec_set_puzzle_mode(PyObject* self, PyObject* args);
// static PyObject* env_set_puzzle_data(PyObject* self, PyObject* args);
// static PyObject* vec_set_puzzle_data(PyObject* self, PyObject* args);
// static PyObject* env_set_puzzle_difficulty(PyObject* self, PyObject* args);
// static PyObject* vec_set_puzzle_difficulty(PyObject* self, PyObject* args);
// static PyObject* env_set_puzzle_training_params(PyObject* self, PyObject* args);
// static PyObject* vec_set_puzzle_training_params(PyObject* self, PyObject* args);

// // Define custom methods for chess module
// #define MY_METHODS \
//     {"env_set_self_play", env_set_self_play, METH_VARARGS, "Enable self-play mode"}, \
//     {"vec_set_self_play", vec_set_self_play, METH_VARARGS, "Enable self-play mode for vector env"}, \
//     {"env_set_dual_agent_self_play", env_set_dual_agent_self_play, METH_VARARGS, "Enable dual agent self-play mode"}, \
//     {"vec_set_dual_agent_self_play", vec_set_dual_agent_self_play, METH_VARARGS, "Enable dual agent self-play mode for vector env"}, \
//     {"env_set_fen", env_set_fen, METH_VARARGS, "Load a FEN into a single env"}, \
//     {"vec_set_fen", vec_set_fen, METH_VARARGS, "Load a FEN into every env in a VecEnv"}, \
//     {"vec_enable_stockfish_black", vec_enable_stockfish_black, METH_VARARGS, "Enable Stockfish for all environments in a VecEnv"}, \
//     {"env_set_puzzle_mode", env_set_puzzle_mode, METH_VARARGS, "Enable puzzle mode for single env"}, \
//     {"vec_set_puzzle_mode", vec_set_puzzle_mode, METH_VARARGS, "Enable puzzle mode for vector env"}, \
//     {"env_set_puzzle_data", env_set_puzzle_data, METH_VARARGS, "Set puzzle data for single env"}, \
//     {"vec_set_puzzle_data", vec_set_puzzle_data, METH_VARARGS, "Set puzzle data for vector env"}, \
//     {"env_set_puzzle_difficulty", env_set_puzzle_difficulty, METH_VARARGS, "Set puzzle difficulty for single env"}, \
//     {"vec_set_puzzle_difficulty", vec_set_puzzle_difficulty, METH_VARARGS, "Set puzzle difficulty for vector env"}, \
//     {"env_set_puzzle_training_params", env_set_puzzle_training_params, METH_VARARGS, "Set puzzle training params for single env"}, \
//     {"vec_set_puzzle_training_params", vec_set_puzzle_training_params, METH_VARARGS, "Set puzzle training params for vector env"}, \
//     {NULL, NULL, 0, NULL}

// #include "../env_binding.h"

// static PyObject* env_set_self_play(PyObject* self, PyObject* args) {
//     Env* env = unpack_env(args);
//     if (!env) return NULL;
//     set_self_play_mode(env, true);
//     Py_RETURN_NONE;
// }

// // Enable self-play for every env in a VecEnv
// static PyObject* vec_set_self_play(PyObject* self, PyObject* args) {
//     VecEnv* vec = unpack_vecenv(args);
//     if (!vec) return NULL;
//     for (int i = 0; i < vec->num_envs; ++i)
//         set_self_play_mode(vec->envs[i], true);
//     Py_RETURN_NONE;
// }

// // Enable dual agent self-play for a single env
// static PyObject* env_set_dual_agent_self_play(PyObject* self, PyObject* args) {
//     Env* env = unpack_env(args);
//     if (!env) return NULL;
//     set_dual_agent_self_play_mode(env, true);
//     Py_RETURN_NONE;
// }

// // Enable dual agent self-play for every env in a VecEnv
// static PyObject* vec_set_dual_agent_self_play(PyObject* self, PyObject* args) {
//     VecEnv* vec = unpack_vecenv(args);
//     if (!vec) return NULL;
//     for (int i = 0; i < vec->num_envs; ++i)
//         set_dual_agent_self_play_mode(vec->envs[i], true);
//     Py_RETURN_NONE;
// }

// // FEN setters
// static PyObject* env_set_fen(PyObject* self, PyObject* args) {
//     Env* env = unpack_env(args);
//     if (!env) return NULL;

//     const char* fen = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
//     if (!fen) return NULL;

//     c_set_fen(env, fen);
//     Py_RETURN_NONE;
// }

// static PyObject* vec_set_fen(PyObject* self, PyObject* args) {
//     VecEnv* vec = unpack_vecenv(args);
//     if (!vec) return NULL;

//     const char* fen = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
//     if (!fen) return NULL;

//     // apply to every env in the vector
//     for (int i = 0; i < vec->num_envs; ++i)
//         c_set_fen(vec->envs[i], fen);

//     Py_RETURN_NONE;
// }

// static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
//     env->reward_valid = unpack(kwargs, (char*)"reward_valid");
//     env->reward_invalid_white = unpack(kwargs, (char*)"reward_invalid_white");
//     env->reward_invalid_black = unpack(kwargs, (char*)"reward_invalid_black");
//     env->reward_white_captures_enemy_piece = unpack(kwargs, (char*)"reward_white_captures_enemy_piece");
//     env->reward_black_captures_enemy_piece = unpack(kwargs, (char*)"reward_black_captures_enemy_piece");
//     env->reward_draw = unpack(kwargs, (char*)"reward_draw");
//     env->reward_win_white = unpack(kwargs, (char*)"reward_win_white");
//     env->reward_win_black = unpack(kwargs, (char*)"reward_win_black");
//     env->reward_loss_white = unpack(kwargs, (char*)"reward_loss_white");
//     env->reward_loss_black = unpack(kwargs, (char*)"reward_loss_black");
//     env->reward_check_white = unpack(kwargs, (char*)"reward_check_white");
//     env->reward_check_black = unpack(kwargs, (char*)"reward_check_black");
//     env->reward_material_diff_white = unpack(kwargs, (char*)"reward_material_diff_white");
//     env->reward_material_diff_black = unpack(kwargs, (char*)"reward_material_diff_black");
//     // Puzzle rewards
//     env->reward_puzzle_solved = unpack(kwargs, (char*)"reward_puzzle_solved");
//     env->reward_puzzle_failed = unpack(kwargs, (char*)"reward_puzzle_failed");
//     env->reward_correct_move = unpack(kwargs, (char*)"reward_correct_move");
//     env->reward_puzzle_correct_piece = unpack(kwargs, (char*)"reward_puzzle_correct_piece");
//     env->reward_puzzle_closer_to_target = unpack(kwargs, (char*)"reward_puzzle_closer_to_target");
//     env->reward_puzzle_correct_promotion = unpack(kwargs, (char*)"reward_puzzle_correct_promotion");
//     env->max_depth = (int)unpack(kwargs, (char*)"max_depth");
//     env->debug_disable_mask = (bool)unpack(kwargs, (char*)"debug_disable_mask");  // FIX: Pass debug flag to C++
    
//     // Set game logging frequency from config
//     int full_game_logging_frequency = (int)unpack(kwargs, (char*)"full_game_logging_frequency");
    
//     init(env);    
    
//     // Set frequency AFTER init to avoid being wiped by memset
//     if (full_game_logging_frequency > 0) {
//         env->context.game_logging_frequency = full_game_logging_frequency;
//     } else {
//         env->context.game_logging_frequency = 500000; // default
//     }    
//     return 0;
// }

// static int my_log(PyObject* dict, Log* log) {
//     assign_to_dict(dict, "perf", log->perf);
//     assign_to_dict(dict, "score", log->score);
//     assign_to_dict(dict, "episode_length", log->episode_length);
//     assign_to_dict(dict, "episode_return", log->episode_return);
//     assign_to_dict(dict, "episode_return_white", log->episode_return_white);
//     assign_to_dict(dict, "episode_return_black", log->episode_return_black);
//     assign_to_dict(dict, "reward_valid", log->reward_valid);
//     assign_to_dict(dict, "reward_white_captures_enemy_piece", log->reward_white_captures_enemy_piece);
//     assign_to_dict(dict, "reward_black_captures_enemy_piece", log->reward_black_captures_enemy_piece);
//     assign_to_dict(dict, "reward_draw", log->reward_draw);
//     assign_to_dict(dict, "reward_win_white", log->reward_win_white);
//     assign_to_dict(dict, "reward_win_black", log->reward_win_black);
//     assign_to_dict(dict, "reward_loss_white", log->reward_loss_white);
//     assign_to_dict(dict, "reward_loss_black", log->reward_loss_black);
//     assign_to_dict(dict, "reward_draw_white", log->reward_draw_white);
//     assign_to_dict(dict, "reward_draw_black", log->reward_draw_black);
//     assign_to_dict(dict, "game_drawn", log->game_drawn);
//     assign_to_dict(dict, "white_win", log->white_win);
//     assign_to_dict(dict, "white_loss", log->white_loss);
//     assign_to_dict(dict, "black_win", log->black_win);
//     assign_to_dict(dict, "black_loss", log->black_loss);
//     assign_to_dict(dict, "stalemate", log->stalemate);
//     assign_to_dict(dict, "insufficient_material", log->insufficient_material);
//     assign_to_dict(dict, "threefold_repetition", log->threefold_repetition);
//     assign_to_dict(dict, "fifty_move_rule", log->fifty_move_rule);
//     assign_to_dict(dict, "max_depth", log->max_depth);
//     assign_to_dict(dict, "white_checkmated", log->white_checkmated);
//     assign_to_dict(dict, "black_checkmated", log->black_checkmated);
//     assign_to_dict(dict, "white_moves", log->white_moves);
//     assign_to_dict(dict, "black_moves", log->black_moves);
//     assign_to_dict(dict, "valid_moves", log->valid_moves);
//     assign_to_dict(dict, "invalid_moves_white", log->invalid_moves_white);
//     assign_to_dict(dict, "invalid_moves_black", log->invalid_moves_black);
//     assign_to_dict(dict, "reward_check_white", log->reward_check_white);
//     assign_to_dict(dict, "reward_check_black", log->reward_check_black);
//     assign_to_dict(dict, "reward_material_diff_white", log->reward_material_diff_white);
//     assign_to_dict(dict, "reward_material_diff_black", log->reward_material_diff_black);
//     assign_to_dict(dict, "stockfish_eval", log->stockfish_eval);
//     assign_to_dict(dict, "en_passant_white", log->en_passant_white);
//     assign_to_dict(dict, "en_passant_black", log->en_passant_black);
//     assign_to_dict(dict, "white_castle_kingside", log->white_castle_kingside);
//     assign_to_dict(dict, "white_castle_queenside", log->white_castle_queenside);
//     assign_to_dict(dict, "black_castle_kingside", log->black_castle_kingside);
//     assign_to_dict(dict, "black_castle_queenside", log->black_castle_queenside);
//     assign_to_dict(dict, "white_promotion_count", log->white_promotion_count);
//     assign_to_dict(dict, "white_promotion_knight", log->white_promotion_knight);
//     assign_to_dict(dict, "white_promotion_bishop", log->white_promotion_bishop);
//     assign_to_dict(dict, "white_promotion_rook", log->white_promotion_rook);
//     assign_to_dict(dict, "white_promotion_queen", log->white_promotion_queen);
//     assign_to_dict(dict, "black_promotion_count", log->black_promotion_count);
//     assign_to_dict(dict, "black_promotion_knight", log->black_promotion_knight);
//     assign_to_dict(dict, "black_promotion_bishop", log->black_promotion_bishop);
//     assign_to_dict(dict, "black_promotion_rook", log->black_promotion_rook);
//     assign_to_dict(dict, "black_promotion_queen", log->black_promotion_queen);

//     // Game logging fields
//     assign_to_dict(dict, "last_move_from", log->last_move_from);
//     assign_to_dict(dict, "last_move_to", log->last_move_to);
//     assign_to_dict(dict, "last_move_promotion", log->last_move_promotion);
//     assign_to_dict(dict, "game_step_logged", log->game_step_logged);
//     assign_to_dict(dict, "game_moves_count", log->game_moves_count);
    
//     // Add complete game logging fields (only fields that exist in current Log struct)
//     assign_to_dict(dict, "complete_game_move_count", log->complete_game_move_count);
    
//     // Puzzle mode statistics
//     assign_to_dict(dict, "puzzle_solved", log->puzzle_solved);
//     assign_to_dict(dict, "puzzle_attempts", log->puzzle_attempts);
//     assign_to_dict(dict, "puzzle_correct_moves", log->puzzle_correct_moves);
//     assign_to_dict(dict, "puzzle_wrong_moves", log->puzzle_wrong_moves);
//     assign_to_dict(dict, "puzzle_difficulty", log->puzzle_difficulty);
//     assign_to_dict(dict, "puzzle_success_rate", log->puzzle_success_rate);
    
//     // Puzzle rewards
//     assign_to_dict(dict, "reward_puzzle_solved", log->reward_puzzle_solved);
//     assign_to_dict(dict, "reward_puzzle_failed", log->reward_puzzle_failed);
//     assign_to_dict(dict, "reward_puzzle_correct_move", log->reward_puzzle_correct_move);
    
//     // Calculate puzzle success rate here, AFTER aggregation
//     // This ensures we get the correct ratio of total solved / total attempts
//     if (log->puzzle_attempts > 0) {
//         float correct_success_rate = log->puzzle_solved / log->puzzle_attempts;
//         assign_to_dict(dict, "puzzle_success_rate", correct_success_rate);
        
//         // Debug logging to trace stats
//         static int log_counter = 0;
//         log_counter++;
//         if (log_counter % 50 == 0) {  // Every 50 logs
//             printf("[STATS DEBUG] Aggregated log: attempts=%.1f, solved=%.1f, rate=%.3f\n",
//                    log->puzzle_attempts, log->puzzle_solved, correct_success_rate);
//             fflush(stdout);
//         }
//     } else {
//         assign_to_dict(dict, "puzzle_success_rate", 0.0f);
//     }
    
//     // Export individual complete game actions from the context  
//     // Note: Cannot access individual env contexts from aggregated log function
//     // Actions will be exported directly via Python processing instead

//     assign_to_dict(dict, "n", log->n);

//     return 0;
// }

// // // Provide enable_stockfish_black here so the Python extension has the symbol
// // extern "C" void enable_stockfish_black(CChess* env, const char* stockfish_cmd, int elo, int search_ms) {
// //     if (!env) return;
// //     ChessContext* ctx = (ChessContext*)env->context;
// //     if (!ctx) return;

// //     if (ctx->sf) {
// //         delete ctx->sf;
// //         ctx->sf = nullptr;
// //     }

// //     // Resolve Stockfish binary path
// //     const char* cmd = nullptr;

// //     // 1) Caller-supplied path takes highest priority
// //     if (stockfish_cmd && stockfish_cmd[0]) {
// //         cmd = stockfish_cmd;
// //     }

// //     // 2) Search common bundled locations
// //     if (!cmd) {
// //         const char* candidates[] = {
// //             "pufferlib/Stockfish/src/stockfish",
// //             "./pufferlib/Stockfish/src/stockfish",
// //             "Stockfish/src/stockfish",
// //             "./Stockfish/src/stockfish",
// //             "stockfish",
// //             nullptr
// //         };
// //         for (int i = 0; candidates[i]; ++i) {
// //             if (access(candidates[i], X_OK) == 0) {
// //                 cmd = candidates[i];
// //                 break;
// //             }
// //         }
// //     }

// //     // 3) Fallback to plain "stockfish" if nothing found
// //     if (!cmd) cmd = "stockfish";

// //     ctx->sf = new Stockfish(cmd, elo, search_ms);
// //     ctx->stockfish_enabled = ctx->sf && ctx->sf->ok();
// // }

// static PyObject* vec_enable_stockfish_black(PyObject* self, PyObject* args) {
//     VecEnv* vec = unpack_vecenv(args);
//     if (!vec) return NULL;

//     const char* stockfish_cmd = nullptr;
//     int elo = 1320;
//     int search_ms = 10;
    
//     // Parse optional arguments
//     if (PyTuple_Size(args) >= 2) {
//         PyObject* cmd_obj = PyTuple_GetItem(args, 1);
//         if (cmd_obj != Py_None) {
//             stockfish_cmd = PyUnicode_AsUTF8(cmd_obj);
//             if (!stockfish_cmd) return NULL;
//         }
//     }
//     if (PyTuple_Size(args) >= 3) {
//         elo = PyLong_AsLong(PyTuple_GetItem(args, 2));
//         if (PyErr_Occurred()) return NULL;
//     }
//     if (PyTuple_Size(args) >= 4) {
//         search_ms = PyLong_AsLong(PyTuple_GetItem(args, 3));
//         if (PyErr_Occurred()) return NULL;
//     }
    
//     for (int i = 0; i < vec->num_envs; ++i) {
//         enable_stockfish_black(vec->envs[i], stockfish_cmd, elo, search_ms);
//     }

//     Py_RETURN_NONE;
// }

// // Puzzle mode functions
// static PyObject* env_set_puzzle_mode(PyObject* self, PyObject* args) {
//     Env* env = unpack_env(args);
//     if (!env) return NULL;
    
//     PyObject* enabled_obj = PyTuple_GetItem(args, 1);
//     bool enabled = PyObject_IsTrue(enabled_obj);
    
//     set_puzzle_mode(env, enabled);
//     Py_RETURN_NONE;
// }

// static PyObject* vec_set_puzzle_mode(PyObject* self, PyObject* args) {
//     VecEnv* vec = unpack_vecenv(args);
//     if (!vec) return NULL;
    
//     PyObject* enabled_obj = PyTuple_GetItem(args, 1);
//     bool enabled = PyObject_IsTrue(enabled_obj);
    
//     for (int i = 0; i < vec->num_envs; ++i) {
//         set_puzzle_mode(vec->envs[i], enabled);
//     }
//     Py_RETURN_NONE;
// }

// static PyObject* env_set_puzzle_data(PyObject* self, PyObject* args) {
//     Env* env = unpack_env(args);
//     if (!env) return NULL;
    
//     const char* fen = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
//     if (!fen) return NULL;
    
//     PyObject* solution_list = PyTuple_GetItem(args, 2);
//     if (!PyList_Check(solution_list)) {
//         PyErr_SetString(PyExc_TypeError, "Solution must be a list of moves");
//         return NULL;
//     }
    
//     int solution_length = PyList_Size(solution_list);
//     if (solution_length > 10) solution_length = 10; // Limit to array size
    
//     const char* solution_moves[10];
//     for (int i = 0; i < solution_length; i++) {
//         PyObject* move_obj = PyList_GetItem(solution_list, i);
//         solution_moves[i] = PyUnicode_AsUTF8(move_obj);
//         if (!solution_moves[i]) return NULL;
//     }
    
//     set_puzzle_data(env, fen, solution_moves, solution_length);
//     Py_RETURN_NONE;
// }

// static PyObject* vec_set_puzzle_data(PyObject* self, PyObject* args) {
//     VecEnv* vec = unpack_vecenv(args);
//     if (!vec) return NULL;
    
//     const char* fen = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
//     if (!fen) return NULL;
    
//     PyObject* solution_list = PyTuple_GetItem(args, 2);
//     if (!PyList_Check(solution_list)) {
//         PyErr_SetString(PyExc_TypeError, "Solution must be a list of moves");
//         return NULL;
//     }
    
//     int solution_length = PyList_Size(solution_list);
//     if (solution_length > 10) solution_length = 10; // Limit to array size
    
//     const char* solution_moves[10];
//     for (int i = 0; i < solution_length; i++) {
//         PyObject* move_obj = PyList_GetItem(solution_list, i);
//         solution_moves[i] = PyUnicode_AsUTF8(move_obj);
//         if (!solution_moves[i]) return NULL;
//     }
    
//     for (int i = 0; i < vec->num_envs; ++i) {
//         set_puzzle_data(vec->envs[i], fen, solution_moves, solution_length);
//     }
    
//     Py_RETURN_NONE;
// }

// static PyObject* env_set_puzzle_difficulty(PyObject* self, PyObject* args) {
//     Env* env = unpack_env(args);
//     if (!env) return NULL;
    
//     int difficulty = PyLong_AsLong(PyTuple_GetItem(args, 1));
//     set_puzzle_difficulty(env, difficulty);
//     Py_RETURN_NONE;
// }

// static PyObject* vec_set_puzzle_difficulty(PyObject* self, PyObject* args) {
//     VecEnv* vec = unpack_vecenv(args);
//     if (!vec) return NULL;
    
//     int difficulty = PyLong_AsLong(PyTuple_GetItem(args, 1));
    
//     for (int i = 0; i < vec->num_envs; ++i) {
//         set_puzzle_difficulty(vec->envs[i], difficulty);
//     }
//     Py_RETURN_NONE;
// }

// static PyObject* env_set_puzzle_training_params(PyObject* self, PyObject* args) {
//     CChess* env = unpack_env(args);
//     if (!env) return NULL;
    
//     int max_tries = PyLong_AsLong(PyTuple_GetItem(args, 1));
//     float success_threshold = PyFloat_AsDouble(PyTuple_GetItem(args, 2));
    
//     set_puzzle_training_params(env, max_tries, success_threshold);
//     Py_RETURN_NONE;
// }

// static PyObject* vec_set_puzzle_training_params(PyObject* self, PyObject* args) {
//     VecEnv* vec = unpack_vecenv(args);
//     if (!vec) return NULL;
    
//     int max_tries = PyLong_AsLong(PyTuple_GetItem(args, 1));
//     float success_threshold = PyFloat_AsDouble(PyTuple_GetItem(args, 2));
    
//     for (int i = 0; i < vec->num_envs; ++i) {
//         set_puzzle_training_params(vec->envs[i], max_tries, success_threshold);
//     }
//     Py_RETURN_NONE;
// }

#pragma once
#include "chess.h"
#include "stockfish_wrapper.h"
#include <Python.h> // makes PyObject visible
#define Env CChess
// forward-declare so env_binding.h can use it
static PyObject *env_set_self_play(PyObject *self, PyObject *args);
static PyObject *vec_set_self_play(PyObject *self, PyObject *args);
static PyObject *env_set_dual_agent_self_play(PyObject *self, PyObject *args);
static PyObject *vec_set_dual_agent_self_play(PyObject *self, PyObject *args);
static PyObject *env_set_fen(PyObject *self, PyObject *args);
static PyObject *vec_set_fen(PyObject *self, PyObject *args);
static PyObject *vec_enable_stockfish_black(PyObject *self, PyObject *args);
static PyObject *env_set_puzzle_mode(PyObject *self, PyObject *args);
static PyObject *vec_set_puzzle_mode(PyObject *self, PyObject *args);
static PyObject *env_set_puzzle_data(PyObject *self, PyObject *args);
static PyObject *vec_set_puzzle_data(PyObject *self, PyObject *args);
static PyObject *env_set_puzzle_difficulty(PyObject *self, PyObject *args);
static PyObject *vec_set_puzzle_difficulty(PyObject *self, PyObject *args);
static PyObject *env_set_puzzle_training_params(PyObject *self, PyObject *args);
static PyObject *vec_set_puzzle_training_params(PyObject *self, PyObject *args);
// Define custom methods for chess module
#define MY_METHODS                                                             \
  {"env_set_self_play", env_set_self_play, METH_VARARGS,                       \
   "Enable self-play mode"},                                                   \
      {"vec_set_self_play", vec_set_self_play, METH_VARARGS,                   \
       "Enable self-play mode for vector env"},                                \
      {"env_set_dual_agent_self_play", env_set_dual_agent_self_play,           \
       METH_VARARGS, "Enable dual agent self-play mode"},                      \
      {"vec_set_dual_agent_self_play", vec_set_dual_agent_self_play,           \
       METH_VARARGS, "Enable dual agent self-play mode for vector env"},       \
      {"env_set_fen", env_set_fen, METH_VARARGS,                               \
       "Load a FEN into a single env"},                                        \
      {"vec_set_fen", vec_set_fen, METH_VARARGS,                               \
       "Load a FEN into every env in a VecEnv"},                               \
      {"vec_enable_stockfish_black", vec_enable_stockfish_black, METH_VARARGS, \
       "Enable Stockfish for all environments in a VecEnv"},                   \
      {"env_set_puzzle_mode", env_set_puzzle_mode, METH_VARARGS,               \
       "Enable puzzle mode for single env"},                                   \
      {"vec_set_puzzle_mode", vec_set_puzzle_mode, METH_VARARGS,               \
       "Enable puzzle mode for vector env"},                                   \
      {"env_set_puzzle_data", env_set_puzzle_data, METH_VARARGS,               \
       "Set puzzle data for single env"},                                      \
      {"vec_set_puzzle_data", vec_set_puzzle_data, METH_VARARGS,               \
       "Set puzzle data for vector env"},                                      \
      {"env_set_puzzle_difficulty", env_set_puzzle_difficulty, METH_VARARGS,   \
       "Set puzzle difficulty for single env"},                                \
      {"vec_set_puzzle_difficulty", vec_set_puzzle_difficulty, METH_VARARGS,   \
       "Set puzzle difficulty for vector env"},                                \
      {"env_set_puzzle_training_params", env_set_puzzle_training_params,       \
       METH_VARARGS, "Set puzzle training params for single env"},             \
      {"vec_set_puzzle_training_params", vec_set_puzzle_training_params,       \
       METH_VARARGS, "Set puzzle training params for vector env"},             \
  {                                                                            \
    NULL, NULL, 0, NULL                                                        \
  }
#include "../env_binding.h"
static PyObject *env_set_self_play(PyObject *self, PyObject *args) {
  Env *env = unpack_env(args);
  if (!env)
    return NULL;
  set_self_play_mode(env, true);
  Py_RETURN_NONE;
}
// Enable self-play for every env in a VecEnv
static PyObject *vec_set_self_play(PyObject *self, PyObject *args) {
  VecEnv *vec = unpack_vecenv(args);
  if (!vec)
    return NULL;
  for (int i = 0; i < vec->num_envs; ++i)
    set_self_play_mode(vec->envs[i], true);
  Py_RETURN_NONE;
}
// Enable dual agent self-play for a single env
static PyObject *env_set_dual_agent_self_play(PyObject *self, PyObject *args) {
  Env *env = unpack_env(args);
  if (!env)
    return NULL;
  set_dual_agent_self_play_mode(env, true);
  Py_RETURN_NONE;
}
// Enable dual agent self-play for every env in a VecEnv
static PyObject *vec_set_dual_agent_self_play(PyObject *self, PyObject *args) {
  VecEnv *vec = unpack_vecenv(args);
  if (!vec)
    return NULL;
  for (int i = 0; i < vec->num_envs; ++i)
    set_dual_agent_self_play_mode(vec->envs[i], true);
  Py_RETURN_NONE;
}
// FEN setters
static PyObject *env_set_fen(PyObject *self, PyObject *args) {
  Env *env = unpack_env(args);
  if (!env)
    return NULL;
  const char *fen = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
  if (!fen)
    return NULL;
  c_set_fen(env, fen);
  Py_RETURN_NONE;
}
static PyObject *vec_set_fen(PyObject *self, PyObject *args) {
  VecEnv *vec = unpack_vecenv(args);
  if (!vec)
    return NULL;
  const char *fen = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
  if (!fen)
    return NULL;
  // apply to every env in the vector
  for (int i = 0; i < vec->num_envs; ++i)
    c_set_fen(vec->envs[i], fen);
  Py_RETURN_NONE;
}
static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
  env->reward_valid = unpack(kwargs, (char *)"reward_valid");
  env->reward_invalid_white = unpack(kwargs, (char *)"reward_invalid_white");
  env->reward_invalid_black = unpack(kwargs, (char *)"reward_invalid_black");
  env->reward_white_captures_enemy_piece =
      unpack(kwargs, (char *)"reward_white_captures_enemy_piece");
  env->reward_black_captures_enemy_piece =
      unpack(kwargs, (char *)"reward_black_captures_enemy_piece");
  env->reward_draw = unpack(kwargs, (char *)"reward_draw");
  env->reward_win_white = unpack(kwargs, (char *)"reward_win_white");
  env->reward_win_black = unpack(kwargs, (char *)"reward_win_black");
  env->reward_loss_white = unpack(kwargs, (char *)"reward_loss_white");
  env->reward_loss_black = unpack(kwargs, (char *)"reward_loss_black");
  env->reward_check_white = unpack(kwargs, (char *)"reward_check_white");
  env->reward_check_black = unpack(kwargs, (char *)"reward_check_black");
  env->reward_material_diff_white =
      unpack(kwargs, (char *)"reward_material_diff_white");
  env->reward_material_diff_black =
      unpack(kwargs, (char *)"reward_material_diff_black");
  // Puzzle rewards
  env->reward_puzzle_solved = unpack(kwargs, (char *)"reward_puzzle_solved");
  env->reward_puzzle_failed = unpack(kwargs, (char *)"reward_puzzle_failed");
  env->reward_correct_move = unpack(kwargs, (char *)"reward_correct_move");
  env->reward_puzzle_correct_piece =
      unpack(kwargs, (char *)"reward_puzzle_correct_piece");
  env->reward_puzzle_closer_to_target =
      unpack(kwargs, (char *)"reward_puzzle_closer_to_target");
  env->reward_puzzle_correct_promotion =
      unpack(kwargs, (char *)"reward_puzzle_correct_promotion");
  env->max_depth = (int)unpack(kwargs, (char *)"max_depth");
  env->debug_disable_mask = (bool)unpack(
      kwargs, (char *)"debug_disable_mask"); // FIX: Pass debug flag to C++

  // Set game logging frequency from config
  int full_game_logging_frequency =
      (int)unpack(kwargs, (char *)"full_game_logging_frequency");

  init(env);

  // Set frequency AFTER init to avoid being wiped by memset
  if (full_game_logging_frequency > 0) {
    env->context.game_logging_frequency = full_game_logging_frequency;
  } else {
    env->context.game_logging_frequency = 500000; // default
  }
  return 0;
}
static int my_log(PyObject *dict, Log *log) {
  assign_to_dict(dict, "perf", log->perf);
  assign_to_dict(dict, "score", log->score);
  assign_to_dict(dict, "episode_length", log->episode_length);
  assign_to_dict(dict, "episode_return", log->episode_return);
  assign_to_dict(dict, "episode_return_white", log->episode_return_white);
  assign_to_dict(dict, "episode_return_black", log->episode_return_black);
  assign_to_dict(dict, "reward_valid", log->reward_valid);
  assign_to_dict(dict, "reward_white_captures_enemy_piece",
                 log->reward_white_captures_enemy_piece);
  assign_to_dict(dict, "reward_black_captures_enemy_piece",
                 log->reward_black_captures_enemy_piece);
  assign_to_dict(dict, "reward_max_depth_termination",
                 log->reward_max_depth_termination);
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
  assign_to_dict(dict, "reward_material_diff_white",
                 log->reward_material_diff_white);
  assign_to_dict(dict, "reward_material_diff_black",
                 log->reward_material_diff_black);
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
  assign_to_dict(dict, "last_move_from", log->last_move_from);
  assign_to_dict(dict, "last_move_to", log->last_move_to);
  assign_to_dict(dict, "last_move_promotion", log->last_move_promotion);
  assign_to_dict(dict, "game_step_logged", log->game_step_logged);
  assign_to_dict(dict, "game_moves_count", log->game_moves_count);
  assign_to_dict(dict, "complete_game_move_count",
                 log->complete_game_move_count);
  assign_to_dict(dict, "puzzle_solved", log->puzzle_solved);
  assign_to_dict(dict, "puzzle_attempts", log->puzzle_attempts);
  assign_to_dict(dict, "puzzle_correct_moves", log->puzzle_correct_moves);
  assign_to_dict(dict, "puzzle_wrong_moves", log->puzzle_wrong_moves);
  assign_to_dict(dict, "puzzle_difficulty", log->puzzle_difficulty);
  assign_to_dict(dict, "puzzle_success_rate", log->puzzle_success_rate);
  assign_to_dict(dict, "puzzle_current_id", log->puzzle_current_id);
  assign_to_dict(dict, "puzzle_global_attempts", log->puzzle_global_attempts);
  assign_to_dict(dict, "puzzle_global_successes", log->puzzle_global_successes);
  assign_to_dict(dict, "puzzle_avg_samples_to_solve",
                 log->puzzle_avg_samples_to_solve);
  assign_to_dict(dict, "puzzle_avg_time_to_solve",
                 log->puzzle_avg_time_to_solve);
  assign_to_dict(dict, "reward_puzzle_solved", log->reward_puzzle_solved);
  assign_to_dict(dict, "reward_puzzle_failed", log->reward_puzzle_failed);
  assign_to_dict(dict, "reward_puzzle_correct_move",
                 log->reward_puzzle_correct_move);

  // Calculate puzzle success rate here, AFTER aggregation
  // This ensures we get the correct ratio of total solved / total attempts
  if (log->puzzle_attempts > 0) {
    float correct_success_rate = log->puzzle_solved / log->puzzle_attempts;
    assign_to_dict(dict, "puzzle_success_rate", correct_success_rate);

    // Debug logging to trace stats
    static int log_counter = 0;
    log_counter++;
    if (log_counter % 50 == 0) { // Every 50 logs
      printf("[STATS DEBUG] Aggregated log: attempts=%.1f, solved=%.1f, "
             "rate=%.3f\n",
             log->puzzle_attempts, log->puzzle_solved, correct_success_rate);
      fflush(stdout);
    }
  } else {
    assign_to_dict(dict, "puzzle_success_rate", 0.0f);
  }

  // Export individual complete game actions from the context
  // Note: Cannot access individual env contexts from aggregated log function
  // Actions will be exported directly via Python processing instead
  assign_to_dict(dict, "n", log->n);
  return 0;
}
// // Provide enable_stockfish_black here so the Python extension has the symbol
// extern "C" void enable_stockfish_black(CChess* env, const char*
// stockfish_cmd, int elo, int search_ms) {
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
//         for (int i = 0; i < candidates[i]; ++i) {
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
static PyObject *vec_enable_stockfish_black(PyObject *self, PyObject *args) {
  VecEnv *vec = unpack_vecenv(args);
  if (!vec)
    return NULL;
  const char *stockfish_cmd = nullptr;
  int elo = 1320;
  int search_ms = 10;

  // Parse optional arguments
  if (PyTuple_Size(args) >= 2) {
    PyObject *cmd_obj = PyTuple_GetItem(args, 1);
    if (cmd_obj != Py_None) {
      stockfish_cmd = PyUnicode_AsUTF8(cmd_obj);
      if (!stockfish_cmd)
        return NULL;
    }
  }
  if (PyTuple_Size(args) >= 3) {
    elo = PyLong_AsLong(PyTuple_GetItem(args, 2));
    if (PyErr_Occurred())
      return NULL;
  }
  if (PyTuple_Size(args) >= 4) {
    search_ms = PyLong_AsLong(PyTuple_GetItem(args, 3));
    if (PyErr_Occurred())
      return NULL;
  }

  for (int i = 0; i < vec->num_envs; ++i) {
    enable_stockfish_black(vec->envs[i], stockfish_cmd, elo, search_ms);
  }
  Py_RETURN_NONE;
}
// Puzzle mode functions
static PyObject *env_set_puzzle_mode(PyObject *self, PyObject *args) {
  Env *env = unpack_env(args);
  if (!env)
    return NULL;

  PyObject *enabled_obj = PyTuple_GetItem(args, 1);
  bool enabled = PyObject_IsTrue(enabled_obj);

  set_puzzle_mode(env, enabled);
  Py_RETURN_NONE;
}
static PyObject *vec_set_puzzle_mode(PyObject *self, PyObject *args) {
  VecEnv *vec = unpack_vecenv(args);
  if (!vec)
    return NULL;

  PyObject *enabled_obj = PyTuple_GetItem(args, 1);
  bool enabled = PyObject_IsTrue(enabled_obj);

  for (int i = 0; i < vec->num_envs; ++i) {
    set_puzzle_mode(vec->envs[i], enabled);
  }
  Py_RETURN_NONE;
}
static PyObject *env_set_puzzle_data(PyObject *self, PyObject *args) {
  Env *env = unpack_env(args);
  if (!env)
    return NULL;

  const char *fen = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
  if (!fen)
    return NULL;

  PyObject *solution_list = PyTuple_GetItem(args, 2);
  if (!PyList_Check(solution_list)) {
    PyErr_SetString(PyExc_TypeError, "Solution must be a list of moves");
    return NULL;
  }

  int solution_length = PyList_Size(solution_list);
  if (solution_length > 10)
    solution_length = 10; // Limit to array size

  const char *solution_moves[10];
  for (int i = 0; i < solution_length; i++) {
    PyObject *move_obj = PyList_GetItem(solution_list, i);
    solution_moves[i] = PyUnicode_AsUTF8(move_obj);
    if (!solution_moves[i])
      return NULL;
  }

  set_puzzle_data(env, fen, solution_moves, solution_length);
  Py_RETURN_NONE;
}
static PyObject *vec_set_puzzle_data(PyObject *self, PyObject *args) {
  VecEnv *vec = unpack_vecenv(args);
  if (!vec)
    return NULL;

  const char *fen = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
  if (!fen)
    return NULL;

  PyObject *solution_list = PyTuple_GetItem(args, 2);
  if (!PyList_Check(solution_list)) {
    PyErr_SetString(PyExc_TypeError, "Solution must be a list of moves");
    return NULL;
  }

  int solution_length = PyList_Size(solution_list);
  if (solution_length > 10)
    solution_length = 10; // Limit to array size

  const char *solution_moves[10];
  for (int i = 0; i < solution_length; i++) {
    PyObject *move_obj = PyList_GetItem(solution_list, i);
    solution_moves[i] = PyUnicode_AsUTF8(move_obj);
    if (!solution_moves[i])
      return NULL;
  }

  for (int i = 0; i < vec->num_envs; ++i) {
    set_puzzle_data(vec->envs[i], fen, solution_moves, solution_length);
  }

  Py_RETURN_NONE;
}
static PyObject *env_set_puzzle_difficulty(PyObject *self, PyObject *args) {
  Env *env = unpack_env(args);
  if (!env)
    return NULL;

  int difficulty = PyLong_AsLong(PyTuple_GetItem(args, 1));
  set_puzzle_difficulty(env, difficulty);
  Py_RETURN_NONE;
}
static PyObject *vec_set_puzzle_difficulty(PyObject *self, PyObject *args) {
  VecEnv *vec = unpack_vecenv(args);
  if (!vec)
    return NULL;

  int difficulty = PyLong_AsLong(PyTuple_GetItem(args, 1));

  for (int i = 0; i < vec->num_envs; ++i) {
    set_puzzle_difficulty(vec->envs[i], difficulty);
  }
  Py_RETURN_NONE;
}
static PyObject *env_set_puzzle_training_params(PyObject *self,
                                                PyObject *args) {
  CChess *env = unpack_env(args);
  if (!env)
    return NULL;

  int max_tries = PyLong_AsLong(PyTuple_GetItem(args, 1));
  float success_threshold = PyFloat_AsDouble(PyTuple_GetItem(args, 2));

  set_puzzle_training_params(env, max_tries, success_threshold);
  Py_RETURN_NONE;
}
static PyObject *vec_set_puzzle_training_params(PyObject *self,
                                                PyObject *args) {
  VecEnv *vec = unpack_vecenv(args);
  if (!vec)
    return NULL;

  int max_tries = PyLong_AsLong(PyTuple_GetItem(args, 1));
  float success_threshold = PyFloat_AsDouble(PyTuple_GetItem(args, 2));

  for (int i = 0; i < vec->num_envs; ++i) {
    set_puzzle_training_params(vec->envs[i], max_tries, success_threshold);
  }
  Py_RETURN_NONE;
}