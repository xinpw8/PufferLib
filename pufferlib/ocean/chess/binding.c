#include "chess.h"

#define Env Chess
#define MY_SHARED
#include "../env_binding.h"

typedef struct {
    char** fens;
    int num_fens;
    char** fens_dm;
    int num_fens_dm;
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
    curriculum->fens_dm = NULL;
    curriculum->num_fens_dm = 0;

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

    // Load DeepMind FEN curriculum if provided
    PyObject* fen_file_dm_obj = PyDict_GetItemString(kwargs, "fen_file_dm");
    if (fen_file_dm_obj != NULL && fen_file_dm_obj != Py_None) {
        const char* fen_file_dm = PyUnicode_AsUTF8(fen_file_dm_obj);
        if (fen_file_dm != NULL) {
            FILE* f_dm = fopen(fen_file_dm, "r");
            if (f_dm != NULL) {
                int num_fens_dm = 0;
                char line_dm[256];
                while (fgets(line_dm, sizeof(line_dm), f_dm)) {
                    if (line_dm[0] != '#' && line_dm[0] != '\n' && line_dm[0] != '\r') {
                        num_fens_dm++;
                    }
                }
                if (num_fens_dm > 0) {
                    curriculum->fens_dm = malloc(num_fens_dm * sizeof(char*));
                    curriculum->num_fens_dm = num_fens_dm;
                    rewind(f_dm);
                    int dm_idx = 0;
                    while (fgets(line_dm, sizeof(line_dm), f_dm) && dm_idx < num_fens_dm) {
                        if (line_dm[0] != '#' && line_dm[0] != '\n' && line_dm[0] != '\r') {
                            size_t len = strlen(line_dm);
                            while (len > 0 && (line_dm[len-1] == '\n' || line_dm[len-1] == '\r')) {
                                line_dm[--len] = '\0';
                            }
                            curriculum->fens_dm[dm_idx++] = strdup(line_dm);
                        }
                    }
                }
                fclose(f_dm);
                printf("Loaded %d DeepMind FENs from %s\n", curriculum->num_fens_dm, fen_file_dm);
            } else {
                fprintf(stderr, "WARNING: Could not open DeepMind FEN file: %s\n", fen_file_dm);
            }
        }
    }

    return PyLong_FromVoidPtr(curriculum);
}

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
    init_bitboards();
    
    env->max_moves = 500;
    env->reward_draw = 0.0f;
    env->reward_invalid_piece = -0.1f;
    env->reward_invalid_move = -0.1f;
    env->reward_capture_scale = 0.0f;
    env->reward_repetition = 0.0f;
    env->client = NULL;
    env->render_fps = 30;
    env->selfplay = 1;
    env->human_play = 0;
    env->random_bot = 0;
    env->mode = CHESS_MODE_SELFPLAY;
    env->legal_dirty = 1;
    env->human_color = -1;
    env->fen_curriculum = NULL;
    env->num_fens = 0;
    env->fen_curriculum_dm = NULL;
    env->num_fens_dm = 0;
    env->deepmind_fen_pct = 0.0f;
    strcpy(env->starting_fen, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    
    env->log_pgn = 0;
    env->log_pgn_choice_made = 1;
    env->pgn_filename[0] = '\0';
    env->pgn_game_number = 0;
    strcpy(env->last_result, "Game starting...");
    
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

        PyObject* reward_repetition_obj = PyDict_GetItemString(kwargs, "reward_repetition");
        if (reward_repetition_obj != NULL && PyFloat_Check(reward_repetition_obj)) {
            env->reward_repetition = (float)PyFloat_AsDouble(reward_repetition_obj);
        } else if (reward_repetition_obj != NULL && PyLong_Check(reward_repetition_obj)) {
            env->reward_repetition = (float)PyLong_AsDouble(reward_repetition_obj);
        }

        PyObject* reward_capture_scale_obj = PyDict_GetItemString(kwargs, "reward_capture_scale");
        if (reward_capture_scale_obj != NULL && PyFloat_Check(reward_capture_scale_obj)) {
            env->reward_capture_scale = (float)PyFloat_AsDouble(reward_capture_scale_obj);
        } else if (reward_capture_scale_obj != NULL && PyLong_Check(reward_capture_scale_obj)) {
            env->reward_capture_scale = (float)PyLong_AsDouble(reward_capture_scale_obj);
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
                env->fen_curriculum_dm = curriculum->fens_dm;
                env->num_fens_dm = curriculum->num_fens_dm;
            }
        }

        env->deepmind_fen_pct = 0.0f;
        PyObject* dm_pct_obj = PyDict_GetItemString(kwargs, "deepmind_fen_pct");
        if (dm_pct_obj != NULL && PyFloat_Check(dm_pct_obj)) {
            env->deepmind_fen_pct = (float)PyFloat_AsDouble(dm_pct_obj);
        } else if (dm_pct_obj != NULL && PyLong_Check(dm_pct_obj)) {
            env->deepmind_fen_pct = (float)PyLong_AsDouble(dm_pct_obj);
        }
        
        PyObject* fen_obj = PyDict_GetItemString(kwargs, "starting_fen");
        if (fen_obj != NULL && PyUnicode_Check(fen_obj)) {
            const char* fen_str = PyUnicode_AsUTF8(fen_obj);
            if (fen_str != NULL) {
                strncpy(env->starting_fen, fen_str, sizeof(env->starting_fen) - 1);
                env->starting_fen[sizeof(env->starting_fen) - 1] = '\0';
            }
        }
        
    }

    if (env->human_play) {
        if (env->selfplay || env->random_bot) {
            PyErr_SetString(PyExc_ValueError,
                "human_play mode requires selfplay=0 and random_bot=0");
            return -1;
        }
        env->mode = CHESS_MODE_HUMAN;
    } else if (env->selfplay) {
        if (env->random_bot) {
            PyErr_SetString(PyExc_ValueError,
                "selfplay mode requires random_bot=0");
            return -1;
        }
        env->mode = CHESS_MODE_SELFPLAY;
    } else if (env->random_bot) {
        env->mode = CHESS_MODE_RANDOM_BOT;
    } else {
        PyErr_SetString(PyExc_ValueError,
            "invalid mode: one of selfplay=1, human_play=1, or random_bot=1 must be set");
        return -1;
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
    return 0;
}
