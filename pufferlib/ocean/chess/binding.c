#include "chess.h"

#define Env Chess
#define MY_SHARED
#include "../env_binding.h"

typedef struct {
    char** fens;
    int num_fens;
} FenCurriculum;

static PyObject* my_shared(PyObject* self, PyObject* args, PyObject* kwargs) {
    init_bitboards();
    
    PyObject* fen_file_obj = PyDict_GetItemString(kwargs, "fen_file");
    if (fen_file_obj == NULL || fen_file_obj == Py_None) {
        Py_RETURN_NONE;
    }
    
    const char* fen_file = PyUnicode_AsUTF8(fen_file_obj);
    if (fen_file == NULL) {
        Py_RETURN_NONE;
    }
    
    FILE* f = fopen(fen_file, "r");
    if (f == NULL) {
        PyErr_Format(PyExc_FileNotFoundError, "Could not open FEN file: %s", fen_file);
        return NULL;
    }
    
    int num_fens = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] != '#' && line[0] != '\n' && line[0] != '\r') {
            num_fens++;
        }
    }
    
    if (num_fens == 0) {
        fclose(f);
        Py_RETURN_NONE;
    }
    
    FenCurriculum* curriculum = malloc(sizeof(FenCurriculum));
    curriculum->fens = malloc(num_fens * sizeof(char*));
    curriculum->num_fens = num_fens;
    
    rewind(f);
    int idx = 0;
    while (fgets(line, sizeof(line), f) && idx < num_fens) {
        if (line[0] != '#' && line[0] != '\n' && line[0] != '\r') {
            size_t len = strlen(line);
            while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) {
                line[--len] = '\0';
            }
            curriculum->fens[idx++] = strdup(line);
        }
    }
    fclose(f);
    
    printf("Loaded %d FENs from %s\n", curriculum->num_fens, fen_file);
    return PyLong_FromVoidPtr(curriculum);
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
    env->random_bot = 0;
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

        PyObject* reward_check = PyDict_GetItemString(kwargs, "reward_check");
        if (reward_check != NULL && PyFloat_Check(reward_check)) {
            env->reward_check = (float)PyFloat_AsDouble(reward_check);
        } else if (reward_check != NULL && PyLong_Check(reward_check)) {
            env->reward_check = (float)PyLong_AsDouble(reward_check);
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

        PyObject* random_bot_obj = PyDict_GetItemString(kwargs, "random_bot");
        if (random_bot_obj != NULL && PyLong_Check(random_bot_obj)) {
            env->random_bot = (int)PyLong_AsLong(random_bot_obj);
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

        env->fen_curric_pct = 0;
        PyObject* fen_curric_pct = PyDict_GetItemString(kwargs, "fen_curric_pct");
        if (fen_curric_pct != NULL && PyFloat_Check(fen_curric_pct)) {
            env->fen_curric_pct = (float)PyFloat_AsDouble(fen_curric_pct);
        } else if (fen_curric_pct != NULL && PyLong_Check(fen_curric_pct)) {
            env->fen_curric_pct = (float)PyLong_AsDouble(fen_curric_pct);
        }
        
        PyObject* curriculum_obj = PyDict_GetItemString(kwargs, "fen_curriculum");
        if (curriculum_obj != NULL && PyLong_Check(curriculum_obj)) {
            FenCurriculum* curriculum = (FenCurriculum*)PyLong_AsVoidPtr(curriculum_obj);
            if (curriculum != NULL) {
                env->fen_curriculum = curriculum->fens;
                env->num_fens = curriculum->num_fens;
            }
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
    assign_to_dict(dict, "invalid_action_rate", log->invalid_action_rate);
    assign_to_dict(dict, "material_score", log->material_score);
    assign_to_dict(dict, "positional_score", log->positional_score);
    assign_to_dict(dict, "white_win_rate", log->white_winrate);
    assign_to_dict(dict, "black_win_rate", log->black_winrate);
    return 0;
}
