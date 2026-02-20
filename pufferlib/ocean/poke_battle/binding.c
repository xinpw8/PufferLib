#include <Python.h>
#include "poke_battle.h"

#define Env PokeBattle
#define MY_GET
#define MY_PUT

// Forward declare so MY_METHODS can reference it
static PyObject* env_render_get_action(PyObject* self, PyObject* args);

#define MY_METHODS \
    {"env_render_get_action", env_render_get_action, METH_VARARGS, "Render and block until human clicks an action"}

#include "../env_binding.h"

static int dict_set_long(PyObject* dict, const char* key, long value) {
    PyObject* obj = PyLong_FromLong(value);
    if (!obj) return -1;
    int rc = PyDict_SetItemString(dict, key, obj);
    Py_DECREF(obj);
    return rc;
}

static int dict_set_str(PyObject* dict, const char* key, const char* value) {
    PyObject* obj = PyUnicode_FromString(value);
    if (!obj) return -1;
    int rc = PyDict_SetItemString(dict, key, obj);
    Py_DECREF(obj);
    return rc;
}

static int dict_set_obj(PyObject* dict, const char* key, PyObject* obj) {
    int rc = PyDict_SetItemString(dict, key, obj);
    Py_DECREF(obj);
    return rc;
}

static const char* status_to_str(StatusCondition s) {
    switch (s) {
        case STATUS_SLEEP: return "slp";
        case STATUS_FREEZE: return "frz";
        case STATUS_BURN: return "brn";
        case STATUS_POISON: return "psn";
        case STATUS_TOXIC: return "tox";
        case STATUS_PARALYSIS: return "par";
        case STATUS_NONE:
        default:
            return "";
    }
}

static const char* safe_species_name(SpeciesID id) {
    if (id <= SPECIES_NONE || id > NUM_SPECIES) return "None";
    return SPECIES_DATA[id].name;
}

static const char* safe_move_name(MoveID id) {
    if (id <= MOVE_NONE || id > NUM_MOVES) return "None";
    return MOVE_DATA[id].name;
}

static PyObject* pack_move(const MoveSlot* move) {
    PyObject* out = PyDict_New();
    if (!out) return NULL;

    if (dict_set_long(out, "id", move->id) < 0 ||
            dict_set_str(out, "name", safe_move_name(move->id)) < 0 ||
            dict_set_long(out, "pp", move->pp) < 0 ||
            dict_set_long(out, "max_pp", move->max_pp) < 0) {
        Py_DECREF(out);
        return NULL;
    }

    return out;
}

static PyObject* pack_pokemon(const Pokemon* p) {
    PyObject* out = PyDict_New();
    if (!out) return NULL;

    if (dict_set_long(out, "species_id", p->species) < 0 ||
            dict_set_str(out, "species", safe_species_name(p->species)) < 0 ||
            dict_set_long(out, "hp", p->hp) < 0 ||
            dict_set_long(out, "max_hp", p->max_hp) < 0 ||
            dict_set_long(out, "is_alive", p->is_alive) < 0 ||
            dict_set_long(out, "status", p->status) < 0 ||
            dict_set_str(out, "status_name", status_to_str(p->status)) < 0 ||
            dict_set_long(out, "type1", p->type1) < 0 ||
            dict_set_long(out, "type2", p->type2) < 0) {
        Py_DECREF(out);
        return NULL;
    }

    PyObject* moves = PyList_New(NUM_MOVE_SLOTS);
    if (!moves) {
        Py_DECREF(out);
        return NULL;
    }

    for (int i = 0; i < NUM_MOVE_SLOTS; i++) {
        PyObject* m = pack_move(&p->moves[i]);
        if (!m) {
            Py_DECREF(moves);
            Py_DECREF(out);
            return NULL;
        }
        PyList_SET_ITEM(moves, i, m);
    }

    if (dict_set_obj(out, "moves", moves) < 0) {
        Py_DECREF(out);
        return NULL;
    }

    return out;
}

static PyObject* pack_player(const Player* player) {
    PyObject* out = PyDict_New();
    if (!out) return NULL;

    if (dict_set_long(out, "active_idx", player->active_idx) < 0 ||
            dict_set_long(out, "alive_count", player->alive_count) < 0 ||
            dict_set_long(out, "atk_stage", player->atk_stage) < 0 ||
            dict_set_long(out, "def_stage", player->def_stage) < 0 ||
            dict_set_long(out, "spc_stage", player->spc_stage) < 0 ||
            dict_set_long(out, "spe_stage", player->spe_stage) < 0 ||
            dict_set_long(out, "accuracy_stage", player->accuracy_stage) < 0 ||
            dict_set_long(out, "evasion_stage", player->evasion_stage) < 0 ||
            dict_set_long(out, "is_confused", player->is_confused) < 0 ||
            dict_set_long(out, "substitute_hp", player->substitute_hp) < 0 ||
            dict_set_long(out, "has_reflect", player->has_reflect) < 0 ||
            dict_set_long(out, "has_light_screen", player->has_light_screen) < 0 ||
            dict_set_long(out, "is_recharging", player->is_recharging) < 0 ||
            dict_set_long(out, "is_trapped", player->is_trapped) < 0) {
        Py_DECREF(out);
        return NULL;
    }

    PyObject* team = PyList_New(NUM_POKEMON);
    if (!team) {
        Py_DECREF(out);
        return NULL;
    }

    for (int i = 0; i < NUM_POKEMON; i++) {
        PyObject* mon = pack_pokemon(&player->team[i]);
        if (!mon) {
            Py_DECREF(team);
            Py_DECREF(out);
            return NULL;
        }
        PyList_SET_ITEM(team, i, mon);
    }

    if (dict_set_obj(out, "team", team) < 0) {
        Py_DECREF(out);
        return NULL;
    }

    return out;
}

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->num_agents = (int)unpack(kwargs, "num_agents");
    env->seed = (unsigned long long)unpack(kwargs, "seed");
    env->selfplay = (int)unpack(kwargs, "selfplay");
    env->learner_side = (int)unpack(kwargs, "learner_side");
    env->bot_mode = (int)unpack(kwargs, "bot_mode");
    env->mcts_iterations = (int)unpack(kwargs, "mcts_iterations");
    env->mcts_depth = (int)unpack(kwargs, "mcts_depth");
    env->auto_reset = (int)unpack(kwargs, "auto_reset");
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

    // Per-species win rates (only emit species that appeared in games)
    for (int i = 1; i <= NUM_SPECIES; i++) {
        if (log->species_games[i] > 0.0f) {
            char key[64];
            snprintf(key, sizeof(key), "wr_%s", SPECIES_DATA[i].name);
            // Convert spaces to underscores for clean metric names
            for (char* p = key; *p; p++) { if (*p == ' ') *p = '_'; }
            assign_to_dict(dict, key, log->species_wins[i] / log->species_games[i]);
        }
    }
    return 0;
}

static PyObject* my_get(PyObject* dict, Env* env) {
    if (dict_set_long(dict, "turn", env->battle.turn) < 0 ||
            dict_set_long(dict, "mode", env->battle.mode) < 0 ||
            dict_set_long(dict, "tick", env->tick) < 0 ||
            dict_set_long(dict, "selfplay", env->selfplay) < 0 ||
            dict_set_long(dict, "learner_side", env->learner_side) < 0 ||
            dict_set_long(dict, "bot_mode", env->bot_mode) < 0 ||
            dict_set_long(dict, "auto_reset", env->auto_reset) < 0 ||
            dict_set_long(dict, "last_p1_action", env->last_p1_action) < 0 ||
            dict_set_long(dict, "last_p2_action", env->last_p2_action) < 0 ||
            dict_set_long(dict, "last_result", env->last_result) < 0 ||
            dict_set_long(dict, "mouse_action", env->mouse_action) < 0 ||
            dict_set_long(dict, "p1_active_idx", env->battle.players[0].active_idx) < 0 ||
            dict_set_long(dict, "p2_active_idx", env->battle.players[1].active_idx) < 0) {
        return NULL;
    }

    PyObject* p1 = pack_player(&env->battle.players[0]);
    PyObject* p2 = pack_player(&env->battle.players[1]);
    if (!p1 || !p2) {
        Py_XDECREF(p1);
        Py_XDECREF(p2);
        return NULL;
    }

    if (dict_set_obj(dict, "p1", p1) < 0 || dict_set_obj(dict, "p2", p2) < 0) {
        return NULL;
    }

    return dict;
}

static int my_put(Env* env, PyObject* args, PyObject* kwargs) {
    (void)args;
    if (!kwargs || kwargs == Py_None) return 0;

    PyObject* seed_obj = PyDict_GetItemString(kwargs, "seed");
    if (seed_obj) {
        if (!PyLong_Check(seed_obj)) {
            PyErr_SetString(PyExc_TypeError, "seed must be an integer");
            return -1;
        }
        unsigned long long seed = PyLong_AsUnsignedLongLong(seed_obj);
        if (PyErr_Occurred()) return -1;
        env->seed = seed;
    }

    PyObject* ep_obj = PyDict_GetItemString(kwargs, "episode_count");
    if (ep_obj) {
        if (!PyLong_Check(ep_obj)) {
            PyErr_SetString(PyExc_TypeError, "episode_count must be an integer");
            return -1;
        }
        unsigned long long episode_count = PyLong_AsUnsignedLongLong(ep_obj);
        if (PyErr_Occurred()) return -1;
        env->episode_count = episode_count;
    }

    return 0;
}

// Render the battle UI and block until the human clicks a valid action.
// Returns the action int (0-9), or -1 if window was closed.
static PyObject* env_render_get_action(PyObject* self, PyObject* args) {
    (void)self;
    Env* env = unpack_env(args);
    if (!env) return NULL;

    env->mouse_action = -1;

    // Release the GIL while we're in the render loop.
    // Loop while mouse_action == -1 (unset). Exits on:
    //   >= 0: valid action clicked
    //   -2:   result overlay clicked (restart signal)
    //   -1:   WindowShouldClose (quit signal)
    Py_BEGIN_ALLOW_THREADS
    while (env->mouse_action == -1) {
        // c_render lazy-inits the Raylib window, so it must run
        // BEFORE WindowShouldClose (which needs an active window).
        c_render(env);
        if (WindowShouldClose()) {
            env->mouse_action = -1;
            break;
        }
    }
    Py_END_ALLOW_THREADS

    return PyLong_FromLong(env->mouse_action);
}
