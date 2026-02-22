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

static int dict_set_double(PyObject* dict, const char* key, double value) {
    PyObject* obj = PyFloat_FromDouble(value);
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

static void sanitize_metric_key(char* key) {
    for (char* p = key; *p; p++) {
        char c = *p;
        int is_alpha = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
        int is_digit = (c >= '0' && c <= '9');
        if (!(is_alpha || is_digit || c == '_')) {
            *p = '_';
        }
    }
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

static double team_builder_recent_winrate_env(const Env* env) {
    if (env->team_builder_recent_count <= 0) return 0.5;
    return (double)env->team_builder_recent_sum / (double)env->team_builder_recent_count;
}

static int team_builder_unique_species_seen_env(const Env* env) {
    int seen = 0;
    for (int i = 0; i < OU_LEGAL_SIZE; i++) {
        SpeciesID sp = OU_LEGAL[i];
        if (env->learner_species_games[sp] > 0.0f) seen++;
    }
    return seen;
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
            dict_set_long(out, "sleep_turns", p->sleep_turns) < 0 ||
            dict_set_long(out, "sleep_source_side", p->sleep_source_side) < 0 ||
            dict_set_long(out, "freeze_source_side", p->freeze_source_side) < 0 ||
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

// Parse a Python list/tuple of NUM_POKEMON ints into a SpeciesID array.
// Returns 0 if key not present (no-op), 1 on success, -1 on error.
static int is_ou_legal_species(long species_id) {
    if (species_id <= 0 || species_id > NUM_SPECIES) return 0;
    for (int i = 0; i < OU_LEGAL_SIZE; i++) {
        if ((long)OU_LEGAL[i] == species_id) return 1;
    }
    return 0;
}

static int parse_team(PyObject* kwargs, const char* key, SpeciesID* out) {
    PyObject* obj = PyDict_GetItemString(kwargs, key);
    if (!obj || obj == Py_None) return 0;

    if (!PySequence_Check(obj)) {
        PyErr_Format(PyExc_TypeError, "%s must be a list or tuple", key);
        return -1;
    }
    Py_ssize_t len = PySequence_Length(obj);
    if (len != NUM_POKEMON) {
        PyErr_Format(PyExc_ValueError, "%s must have exactly %d elements, got %zd",
                     key, NUM_POKEMON, len);
        return -1;
    }
    int used[NUM_SPECIES + 1];
    memset(used, 0, sizeof(used));
    for (int i = 0; i < NUM_POKEMON; i++) {
        PyObject* item = PySequence_GetItem(obj, i);
        if (!item || !PyLong_Check(item)) {
            Py_XDECREF(item);
            PyErr_Format(PyExc_TypeError, "%s[%d] must be an integer", key, i);
            return -1;
        }
        long val = PyLong_AsLong(item);
        Py_DECREF(item);
        if (val < 0 || val > NUM_SPECIES) {
            PyErr_Format(PyExc_ValueError, "%s[%d] = %ld is not a valid species ID (0-%d)",
                         key, i, val, NUM_SPECIES);
            return -1;
        }
        if (!is_ou_legal_species(val)) {
            PyErr_Format(PyExc_ValueError,
                         "%s[%d] = %ld is not an OU-legal Gen 1 species id", key, i, val);
            return -1;
        }
        if (used[val]) {
            PyErr_Format(PyExc_ValueError,
                         "%s violates Species Clause: duplicate species id %ld", key, val);
            return -1;
        }
        used[val] = 1;
        out[i] = (SpeciesID)val;
    }
    return 1;
}

// Parse optional int kwarg into [min_val, max_val].
// Returns 0 if key missing, 1 on success, -1 on error.
static int parse_int_kwarg(PyObject* kwargs, const char* key, int min_val, int max_val, int* out) {
    PyObject* obj = PyDict_GetItemString(kwargs, key);
    if (!obj || obj == Py_None) return 0;
    if (!PyLong_Check(obj)) {
        PyErr_Format(PyExc_TypeError, "%s must be an integer", key);
        return -1;
    }
    long value = PyLong_AsLong(obj);
    if (PyErr_Occurred()) return -1;
    if (value < min_val || value > max_val) {
        PyErr_Format(PyExc_ValueError, "%s must be in [%d, %d], got %ld",
                     key, min_val, max_val, value);
        return -1;
    }
    *out = (int)value;
    return 1;
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
    if (parse_int_kwarg(kwargs, "team_builder_mode",
                        TEAM_BUILDER_DISABLED, TEAM_BUILDER_ADAPTIVE,
                        &env->team_builder_mode) < 0) return -1;
    if (parse_team(kwargs, "p1_team", env->p1_fixed_team) < 0) return -1;
    if (parse_team(kwargs, "p2_team", env->p2_fixed_team) < 0) return -1;
    if (parse_int_kwarg(kwargs, "force_accuracy", -1, 1, &env->force_accuracy) < 0) return -1;
    if (parse_int_kwarg(kwargs, "force_secondary", -1, 1, &env->force_secondary) < 0) return -1;
    if (parse_int_kwarg(kwargs, "enforce_endless_clause", 0, 1, &env->enforce_endless_clause) < 0) return -1;
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
    assign_to_dict(dict, "team_builder_recent_winrate", log->team_builder_recent_winrate);
    assign_to_dict(dict, "team_builder_pool_coverage", log->team_builder_pool_coverage);

    float wr_by_species[NUM_SPECIES + 1] = {0};
    float pick_by_species[NUM_SPECIES + 1] = {0};
    float score_by_species[NUM_SPECIES + 1] = {0};

    // Per-species diagnostics (only emit species that appeared in games)
    // `species_games` is averaged by vec_log, so this is recent pick-rate per episode.
    for (int i = 1; i <= NUM_SPECIES; i++) {
        if (log->species_games[i] > 0.0f) {
            pick_by_species[i] = log->species_games[i];
            wr_by_species[i] = log->species_wins[i] / log->species_games[i];

            char key[64];
            snprintf(key, sizeof(key), "wr_%s", SPECIES_DATA[i].name);
            sanitize_metric_key(key);
            assign_to_dict(dict, key, wr_by_species[i]);

            snprintf(key, sizeof(key), "pick_%s", SPECIES_DATA[i].name);
            sanitize_metric_key(key);
            assign_to_dict(dict, key, pick_by_species[i]);
        }
    }

    // Build a compact "current best team" snapshot from recent builder behavior.
    // Score prioritizes pick-rate, then win-rate, then OU prior base weight.
    for (int i = 0; i < OU_LEGAL_SIZE; i++) {
        SpeciesID sp = OU_LEGAL[i];
        float pick_rate = pick_by_species[sp];
        float wr = (pick_rate > 0.0f) ? wr_by_species[sp] : 0.5f;
        score_by_species[sp] = 4.0f * pick_rate + wr + 0.03f * species_base_weight(sp);
    }

    int used[NUM_SPECIES + 1] = {0};
    float best_mean_wr = 0.0f;
    float best_mean_pick = 0.0f;
    for (int slot = 0; slot < NUM_POKEMON; slot++) {
        SpeciesID best_sp = SPECIES_NONE;
        float best_score = -1.0e30f;
        for (int i = 0; i < OU_LEGAL_SIZE; i++) {
            SpeciesID sp = OU_LEGAL[i];
            if (used[sp]) continue;
            float s = score_by_species[sp];
            if (s > best_score) {
                best_score = s;
                best_sp = sp;
            }
        }
        if (best_sp == SPECIES_NONE) break;
        used[best_sp] = 1;
        float pick_rate = pick_by_species[best_sp];
        float wr = (pick_rate > 0.0f) ? wr_by_species[best_sp] : 0.5f;

        char key[80];
        snprintf(key, sizeof(key), "team_builder_best_species_%d", slot + 1);
        assign_to_dict(dict, key, (float)best_sp);
        snprintf(key, sizeof(key), "team_builder_best_species_%d_pick_rate", slot + 1);
        assign_to_dict(dict, key, pick_rate);
        snprintf(key, sizeof(key), "team_builder_best_species_%d_wr", slot + 1);
        assign_to_dict(dict, key, wr);
        snprintf(key, sizeof(key), "team_builder_best_species_%d_score", slot + 1);
        assign_to_dict(dict, key, best_score);

        best_mean_wr += wr;
        best_mean_pick += pick_rate;
    }
    assign_to_dict(dict, "team_builder_best_team_mean_wr", best_mean_wr / (float)NUM_POKEMON);
    assign_to_dict(dict, "team_builder_best_team_mean_pick_rate", best_mean_pick / (float)NUM_POKEMON);

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
            dict_set_long(dict, "team_builder_mode", env->team_builder_mode) < 0 ||
            dict_set_long(dict, "team_builder_unique_species_seen", team_builder_unique_species_seen_env(env)) < 0 ||
            dict_set_long(dict, "stale_turns", env->stale_turns) < 0 ||
            dict_set_long(dict, "enforce_endless_clause", env->enforce_endless_clause) < 0 ||
            dict_set_long(dict, "force_accuracy", env->force_accuracy) < 0 ||
            dict_set_long(dict, "force_secondary", env->force_secondary) < 0 ||
            dict_set_long(dict, "last_p1_action", env->last_p1_action) < 0 ||
            dict_set_long(dict, "last_p2_action", env->last_p2_action) < 0 ||
            dict_set_long(dict, "last_result", env->last_result) < 0 ||
            dict_set_long(dict, "mouse_action", env->mouse_action) < 0 ||
            dict_set_long(dict, "p1_active_idx", env->battle.players[0].active_idx) < 0 ||
            dict_set_long(dict, "p2_active_idx", env->battle.players[1].active_idx) < 0 ||
            dict_set_str(dict, "ruleset", "[Gen 1] OU") < 0 ||
            dict_set_long(dict, "sleep_clause_mod", 1) < 0 ||
            dict_set_long(dict, "freeze_clause_mod", 1) < 0 ||
            dict_set_long(dict, "species_clause", 1) < 0 ||
            dict_set_long(dict, "ohko_clause", 1) < 0 ||
            dict_set_long(dict, "evasion_moves_clause", 1) < 0 ||
            dict_set_long(dict, "endless_battle_clause", 1) < 0) {
        return NULL;
    }
    if (dict_set_double(dict, "team_builder_recent_winrate", team_builder_recent_winrate_env(env)) < 0 ||
            dict_set_double(dict, "team_builder_pool_coverage",
                            (double)team_builder_unique_species_seen_env(env) / (double)OU_LEGAL_SIZE) < 0) {
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

    int parsed_team_builder_mode = 0;
    int next_team_builder_mode = env->team_builder_mode;
    parsed_team_builder_mode = parse_int_kwarg(kwargs, "team_builder_mode",
                                              TEAM_BUILDER_DISABLED, TEAM_BUILDER_ADAPTIVE,
                                              &next_team_builder_mode);
    if (parsed_team_builder_mode < 0) return -1;
    if (parsed_team_builder_mode == 1 && next_team_builder_mode != env->team_builder_mode) {
        env->team_builder_mode = next_team_builder_mode;
        team_builder_reset_state(env);
    } else if (parsed_team_builder_mode == 1) {
        env->team_builder_mode = next_team_builder_mode;
    }

    if (parse_team(kwargs, "p1_team", env->p1_fixed_team) < 0) return -1;
    if (parse_team(kwargs, "p2_team", env->p2_fixed_team) < 0) return -1;
    if (parse_int_kwarg(kwargs, "force_accuracy", -1, 1, &env->force_accuracy) < 0) return -1;
    if (parse_int_kwarg(kwargs, "force_secondary", -1, 1, &env->force_secondary) < 0) return -1;
    if (parse_int_kwarg(kwargs, "enforce_endless_clause", 0, 1, &env->enforce_endless_clause) < 0) return -1;

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
