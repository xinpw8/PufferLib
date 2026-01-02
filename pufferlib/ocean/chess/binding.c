#include "chess.h"

#define Env Chess
#define MY_SHARED
#include "../env_binding.h"

static PyObject* my_shared(PyObject* self, PyObject* args, PyObject* kwargs) {
    init_bitboards();
    Py_RETURN_NONE;
}

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
    init_bitboards();
    
    env->max_moves = 500;
    env->reward_draw = 0.0f;
    env->reward_invalid_piece = -0.1f;
    env->reward_invalid_move = -0.1f;
    env->reward_valid_piece = 0.0f;
    env->reward_valid_move = 0.0f;
    env->reward_material = 0.0f;
    env->reward_position = 0.0f;
    env->reward_castling = 0.0f;
    env->reward_repetition = 0.0f;
    env->client = NULL;
    env->render_fps = 30;
    env->selfplay = 1;
    env->human_play = 0;
    env->human_color = -1;
    env->fen_curriculum = NULL;
    env->num_fens = 0;
    strcpy(env->starting_fen, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    
    env->log_pgn = 0;
    env->log_pgn_choice_made = 1;
    env->pgn_filename[0] = '\0';
    env->pgn_game_number = 0;
    env->debug_mode = 0;
    env->learner_color = 0; 
    
    if (kwargs != NULL) {
        PyObject* max_moves_obj = PyDict_GetItemString(kwargs, "max_moves");
        if (max_moves_obj != NULL && PyLong_Check(max_moves_obj)) {
            env->max_moves = (int)PyLong_AsLong(max_moves_obj);
        }

        PyObject* reward_draw_obj = PyDict_GetItemString(kwargs, "reward_draw");
        if (reward_draw_obj != NULL && PyFloat_Check(reward_draw_obj)) {
            env->reward_draw = (float)PyFloat_AsDouble(reward_draw_obj);
        } else if (reward_draw_obj != NULL && PyLong_Check(reward_draw_obj)) {
            env->reward_draw = (float)PyLong_AsDouble(reward_draw_obj);
        }

        PyObject* reward_invalid_piece_obj = PyDict_GetItemString(kwargs, "reward_invalid_piece");
        if (reward_invalid_piece_obj != NULL && PyFloat_Check(reward_invalid_piece_obj)) {
            env->reward_invalid_piece = (float)PyFloat_AsDouble(reward_invalid_piece_obj);
        } else if (reward_invalid_piece_obj != NULL && PyLong_Check(reward_invalid_piece_obj)) {
            env->reward_invalid_piece = (float)PyLong_AsDouble(reward_invalid_piece_obj);
        }

        PyObject* reward_invalid_move_obj = PyDict_GetItemString(kwargs, "reward_invalid_move");
        if (reward_invalid_move_obj != NULL && PyFloat_Check(reward_invalid_move_obj)) {
            env->reward_invalid_move = (float)PyFloat_AsDouble(reward_invalid_move_obj);
        } else if (reward_invalid_move_obj != NULL && PyLong_Check(reward_invalid_move_obj)) {
            env->reward_invalid_move = (float)PyLong_AsDouble(reward_invalid_move_obj);
        }

        PyObject* reward_valid_piece_obj = PyDict_GetItemString(kwargs, "reward_valid_piece");
        if (reward_valid_piece_obj != NULL && PyFloat_Check(reward_valid_piece_obj)) {
            env->reward_valid_piece = (float)PyFloat_AsDouble(reward_valid_piece_obj);
        } else if (reward_valid_piece_obj != NULL && PyLong_Check(reward_valid_piece_obj)) {
            env->reward_valid_piece = (float)PyLong_AsDouble(reward_valid_piece_obj);
        }

        PyObject* reward_valid_move_obj = PyDict_GetItemString(kwargs, "reward_valid_move");
        if (reward_valid_move_obj != NULL && PyFloat_Check(reward_valid_move_obj)) {
            env->reward_valid_move = (float)PyFloat_AsDouble(reward_valid_move_obj);
        } else if (reward_valid_move_obj != NULL && PyLong_Check(reward_valid_move_obj)) {
            env->reward_valid_move = (float)PyLong_AsDouble(reward_valid_move_obj);
        }

        PyObject* reward_material_obj = PyDict_GetItemString(kwargs, "reward_material");
        if (reward_material_obj != NULL && PyFloat_Check(reward_material_obj)) {
            env->reward_material = (float)PyFloat_AsDouble(reward_material_obj);
        } else if (reward_material_obj != NULL && PyLong_Check(reward_material_obj)) {
            env->reward_material = (float)PyLong_AsDouble(reward_material_obj);
        }

        PyObject* reward_position_obj = PyDict_GetItemString(kwargs, "reward_position");
        if (reward_position_obj != NULL && PyFloat_Check(reward_position_obj)) {
            env->reward_position = (float)PyFloat_AsDouble(reward_position_obj);
        } else if (reward_position_obj != NULL && PyLong_Check(reward_position_obj)) {
            env->reward_position = (float)PyLong_AsDouble(reward_position_obj);
        }

        PyObject* reward_castling_obj = PyDict_GetItemString(kwargs, "reward_castling");
        if (reward_castling_obj != NULL && PyFloat_Check(reward_castling_obj)) {
            env->reward_castling = (float)PyFloat_AsDouble(reward_castling_obj);
        } else if (reward_castling_obj != NULL && PyLong_Check(reward_castling_obj)) {
            env->reward_castling = (float)PyLong_AsDouble(reward_castling_obj);
        }

        PyObject* reward_repetition_obj = PyDict_GetItemString(kwargs, "reward_repetition");
        if (reward_repetition_obj != NULL && PyFloat_Check(reward_repetition_obj)) {
            env->reward_repetition = (float)PyFloat_AsDouble(reward_repetition_obj);
        } else if (reward_repetition_obj != NULL && PyLong_Check(reward_repetition_obj)) {
            env->reward_repetition = (float)PyLong_AsDouble(reward_repetition_obj);
        }

        PyObject* fps_obj = PyDict_GetItemString(kwargs, "render_fps");
        if (fps_obj != NULL && PyLong_Check(fps_obj)) {
            env->render_fps = (int)PyLong_AsLong(fps_obj);
        }

        PyObject* selfplay_obj = PyDict_GetItemString(kwargs, "selfplay");
        if (selfplay_obj != NULL && PyLong_Check(selfplay_obj)) {
            env->selfplay = (int)PyLong_AsLong(selfplay_obj);
        }

        PyObject* human_obj = PyDict_GetItemString(kwargs, "human_play");
        if (human_obj != NULL && PyLong_Check(human_obj)) {
            env->human_play = (int)PyLong_AsLong(human_obj);
        }

        PyObject* learner_color_obj = PyDict_GetItemString(kwargs, "learner_color");
        if (learner_color_obj != NULL && PyLong_Check(learner_color_obj)) {
            env->learner_color = (int)PyLong_AsLong(learner_color_obj);
        }

        env->enable_50_move_rule = 1;
        PyObject* enable_50_obj = PyDict_GetItemString(kwargs, "enable_50_move_rule");
        if (enable_50_obj != NULL && PyLong_Check(enable_50_obj)) {
            env->enable_50_move_rule = (int)PyLong_AsLong(enable_50_obj);
        }
        
        env->enable_threefold_repetition = 1;
        PyObject* enable_3fold_obj = PyDict_GetItemString(kwargs, "enable_threefold_repetition");
        if (enable_3fold_obj != NULL && PyLong_Check(enable_3fold_obj)) {
            env->enable_threefold_repetition = (int)PyLong_AsLong(enable_3fold_obj);
        }

        env->random_fen = 0;
        PyObject* random_fen_obj = PyDict_GetItemString(kwargs, "random_fen");
        if (random_fen_obj != NULL && PyLong_Check(random_fen_obj)) {
            env->random_fen = (int)PyLong_AsLong(random_fen_obj);
        }
        
        PyObject* fen_obj = PyDict_GetItemString(kwargs, "starting_fen");
        if (fen_obj != NULL && PyUnicode_Check(fen_obj)) {
            const char* fen_str = PyUnicode_AsUTF8(fen_obj);
            if (fen_str != NULL) {
                strncpy(env->starting_fen, fen_str, sizeof(env->starting_fen) - 1);
                env->starting_fen[sizeof(env->starting_fen) - 1] = '\0';
            }
        }
        
        PyObject* debug_obj = PyDict_GetItemString(kwargs, "debug");
        if (debug_obj != NULL && PyLong_Check(debug_obj)) {
            env->debug_mode = (int)PyLong_AsLong(debug_obj);
        }
    }
    
    return 0;
}

static int my_log(PyObject *dict, Log *log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "draw_rate", log->draw_rate);
    assign_to_dict(dict, "timeout_rate", log->timeout_rate);
    assign_to_dict(dict, "chess_moves", log->chess_moves);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "episode_return", log->episode_return);
    
    float avg_invalid_rate = (log->n > 0) ? (log->invalid_action_rate / log->n) : 0.0f;
    assign_to_dict(dict, "invalid_action_rate", avg_invalid_rate);
    
    float avg_material = (log->n > 0) ? (log->material_score / log->n) : 0.0f;
    float avg_positional = (log->n > 0) ? (log->positional_score / log->n) : 0.0f;
    assign_to_dict(dict, "material_score", avg_material);
    assign_to_dict(dict, "positional_score", avg_positional);
    return 0;
}