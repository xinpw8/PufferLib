#include "chess.h"
#include <libgen.h>

// Selfplay mode: doubled obs [learner(1082) | opponent(1082)], interleaved actions [learner(97) | opponent(97)]
#define OBS_SIZE 2164
#define NUM_ATNS 2
#define ACT_SIZES {97, 97}
#define OBS_TYPE UNSIGNED_CHAR
#define ACT_TYPE DOUBLE

#define Env Chess
#include "env_binding.h"

static int _chess_bitboards_initialized = 0;
static char** _fen_curriculum = NULL;
static int _num_fens = 0;
static int _fens_loaded = 0;

static char** _fen_curriculum_dm = NULL;
static uint16_t* _tutor_moves_dm = NULL;
static int _num_fens_dm = 0;
static int _fens_dm_loaded = 0;
static int _g_color_counter = 0;

#define MY_GET
void* my_get(void* env_ptr, Dict* out) {
    (void)env_ptr;
    dict_set(out, "sf_random_pct", (double)_g_sf_random_pct);
    dict_set(out, "sf_random_pct_f", (double)_g_sf_random_pct_f);
    dict_set(out, "ema_winrate", (double)_g_ema_wr);
    dict_set(out, "annealing_games", (double)_g_annealing_games);
    dict_set(out, "color_counter", (double)_g_color_counter);
    return NULL;
}

#define MY_PUT
int my_put(void* env_ptr, Dict* kwargs) {
    (void)env_ptr;

    DictItem* i = dict_get_unsafe(kwargs, "sf_random_pct");
    if (i) _g_sf_random_pct = (int)i->value;

    i = dict_get_unsafe(kwargs, "sf_random_pct_f");
    if (i) _g_sf_random_pct_f = (float)i->value;

    i = dict_get_unsafe(kwargs, "ema_winrate");
    if (i) _g_ema_wr = (float)i->value;

    i = dict_get_unsafe(kwargs, "annealing_games");
    if (i) _g_annealing_games = (int)i->value;

    i = dict_get_unsafe(kwargs, "color_counter");
    if (i) _g_color_counter = (int)i->value;

    // Keep int/float random pct in sync if only one was provided.
    if (_g_sf_random_pct >= 0 && _g_sf_random_pct_f < 0.0f) {
        _g_sf_random_pct_f = (float)_g_sf_random_pct;
    } else if (_g_sf_random_pct_f >= 0.0f && _g_sf_random_pct < 0) {
        _g_sf_random_pct = (int)roundf(_g_sf_random_pct_f);
    }

    return 0;
}

void load_fen_curriculum(void) {
    if (_fens_loaded) return;
    _fens_loaded = 1;

    // Build path to fens2.txt relative to this source file
    char dir[512];
    strncpy(dir, __FILE__, sizeof(dir) - 1);
    dir[sizeof(dir) - 1] = '\0';
    // dirname modifies in place
    char* d = dirname(dir);
    char path[1024];
    snprintf(path, sizeof(path), "%s/fens2.txt", d);

    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "WARNING: Could not open FEN curriculum at %s\n", path);
        return;
    }

    // Count lines
    int capacity = 16384;
    _fen_curriculum = (char**)malloc(capacity * sizeof(char*));
    _num_fens = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        // Strip trailing newline
        int len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) {
            line[--len] = '\0';
        }
        if (len == 0) continue;
        if (_num_fens >= capacity) {
            capacity *= 2;
            _fen_curriculum = (char**)realloc(_fen_curriculum, capacity * sizeof(char*));
        }
        _fen_curriculum[_num_fens] = (char*)malloc(len + 1);
        strcpy(_fen_curriculum[_num_fens], line);
        _num_fens++;
    }
    fclose(f);
    fprintf(stderr, "Loaded FEN curriculum: %d positions from %s\n", _num_fens, path);
}

void load_fen_curriculum_dm(void) {
    if (_fens_dm_loaded) return;
    _fens_dm_loaded = 1;

    // Build path to fens_deepmind.txt relative to this source file
    char dir[512];
    strncpy(dir, __FILE__, sizeof(dir) - 1);
    dir[sizeof(dir) - 1] = '\0';
    char* d = dirname(dir);
    char path[1024];
    snprintf(path, sizeof(path), "%s/fens_deepmind.txt", d);

    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "WARNING: Could not open DeepMind FEN curriculum at %s\n", path);
        return;
    }

    int capacity = 65536;
    _fen_curriculum_dm = (char**)malloc(capacity * sizeof(char*));
    _num_fens_dm = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        int len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) {
            line[--len] = '\0';
        }
        if (len == 0) continue;
        if (_num_fens_dm >= capacity) {
            capacity *= 2;
            _fen_curriculum_dm = (char**)realloc(_fen_curriculum_dm, capacity * sizeof(char*));
        }
        _fen_curriculum_dm[_num_fens_dm] = (char*)malloc(len + 1);
        strcpy(_fen_curriculum_dm[_num_fens_dm], line);
        _num_fens_dm++;
    }
    fclose(f);
    fprintf(stderr, "Loaded DeepMind FEN curriculum: %d positions from %s\n", _num_fens_dm, path);
}

// Parse UCI move string (e.g. "e2e4", "a7a8q") into packed uint16_t:
// bits [5:0] = from_sq, [11:6] = to_sq, [15:12] = promo piece type (KNIGHT=2..QUEEN=5, 0=none)
static uint16_t parse_uci_to_packed(const char* uci) {
    if (!uci || strlen(uci) < 4) return 0;
    int from_file = uci[0] - 'a';
    int from_rank = uci[1] - '1';
    int to_file = uci[2] - 'a';
    int to_rank = uci[3] - '1';
    if (from_file < 0 || from_file > 7 || from_rank < 0 || from_rank > 7) return 0;
    if (to_file < 0 || to_file > 7 || to_rank < 0 || to_rank > 7) return 0;
    uint16_t from_sq = (uint16_t)(from_rank * 8 + from_file);
    uint16_t to_sq = (uint16_t)(to_rank * 8 + to_file);
    uint16_t promo = 0;
    if (strlen(uci) >= 5) {
        switch (uci[4]) {
            case 'n': promo = KNIGHT; break;
            case 'b': promo = BISHOP; break;
            case 'r': promo = ROOK; break;
            case 'q': promo = QUEEN; break;
            default: break;
        }
    }
    return from_sq | (to_sq << 6) | (promo << 12);
}

void load_fen_curriculum_dm_with_moves(void) {
    if (_fens_dm_loaded) return;
    _fens_dm_loaded = 1;

    // Try fens_moves_deepmind.txt first (FEN<tab>UCI_MOVE per line)
    char dir[512];
    strncpy(dir, __FILE__, sizeof(dir) - 1);
    dir[sizeof(dir) - 1] = '\0';
    char* d = dirname(dir);
    char path[1024];
    snprintf(path, sizeof(path), "%s/fens_moves_deepmind.txt", d);

    FILE* f = fopen(path, "r");
    if (!f) {
        // Fall back to FEN-only file
        fprintf(stderr, "NOTE: fens_moves_deepmind.txt not found, falling back to fens_deepmind.txt\n");
        _fens_dm_loaded = 0;  // Allow load_fen_curriculum_dm to run
        load_fen_curriculum_dm();
        return;
    }

    int capacity = 65536;
    _fen_curriculum_dm = (char**)malloc(capacity * sizeof(char*));
    _tutor_moves_dm = (uint16_t*)malloc(capacity * sizeof(uint16_t));
    _num_fens_dm = 0;
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        int len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) {
            line[--len] = '\0';
        }
        if (len == 0) continue;

        // Split on tab
        char* tab = strchr(line, '\t');
        if (!tab) {
            // No tab — treat as FEN-only line
            if (_num_fens_dm >= capacity) {
                capacity *= 2;
                _fen_curriculum_dm = (char**)realloc(_fen_curriculum_dm, capacity * sizeof(char*));
                _tutor_moves_dm = (uint16_t*)realloc(_tutor_moves_dm, capacity * sizeof(uint16_t));
            }
            _fen_curriculum_dm[_num_fens_dm] = (char*)malloc(len + 1);
            strcpy(_fen_curriculum_dm[_num_fens_dm], line);
            _tutor_moves_dm[_num_fens_dm] = 0;
            _num_fens_dm++;
            continue;
        }

        *tab = '\0';
        const char* fen = line;
        const char* uci = tab + 1;

        if (_num_fens_dm >= capacity) {
            capacity *= 2;
            _fen_curriculum_dm = (char**)realloc(_fen_curriculum_dm, capacity * sizeof(char*));
            _tutor_moves_dm = (uint16_t*)realloc(_tutor_moves_dm, capacity * sizeof(uint16_t));
        }
        int fen_len = strlen(fen);
        _fen_curriculum_dm[_num_fens_dm] = (char*)malloc(fen_len + 1);
        strcpy(_fen_curriculum_dm[_num_fens_dm], fen);
        _tutor_moves_dm[_num_fens_dm] = parse_uci_to_packed(uci);
        _num_fens_dm++;
    }
    fclose(f);
    fprintf(stderr, "Loaded DeepMind FEN+move curriculum: %d positions from %s\n", _num_fens_dm, path);
}

void my_init(Env* env, Dict* kwargs) {
    // Initialize bitboards once (global chess tables)
    if (!_chess_bitboards_initialized) {
        init_bitboards();
        _chess_bitboards_initialized = 1;
    }

    DictItem* sp = dict_get_unsafe(kwargs, "selfplay");
    env->selfplay = sp ? (int)sp->value : 0;

    DictItem* hp = dict_get_unsafe(kwargs, "human_play");
    env->human_play = hp ? (int)hp->value : 0;
    if (env->human_play) {
        env->selfplay = 0;
    }

    DictItem* rb = dict_get_unsafe(kwargs, "random_bot");
    env->random_bot = rb ? (int)rb->value : 0;

    DictItem* sf = dict_get_unsafe(kwargs, "stockfish_bot");
    env->stockfish_bot = sf ? (int)sf->value : 0;

    DictItem* sfls = dict_get_unsafe(kwargs, "stockfish_limit_strength");
    env->stockfish_limit_strength = sfls ? (int)sfls->value : 1;

    DictItem* sfelo = dict_get_unsafe(kwargs, "stockfish_elo");
    env->stockfish_elo = sfelo ? (int)sfelo->value : 2200;

    DictItem* sfms = dict_get_unsafe(kwargs, "stockfish_movetime_ms");
    env->stockfish_movetime_ms = sfms ? (int)sfms->value : 30;

    DictItem* sfd = dict_get_unsafe(kwargs, "stockfish_depth");
    env->stockfish_depth = sfd ? (int)sfd->value : 0;

    DictItem* sfrp = dict_get_unsafe(kwargs, "stockfish_random_pct");
    env->stockfish_random_pct = sfrp ? (int)sfrp->value : 0;
    DictItem* sfqp = dict_get_unsafe(kwargs, "stockfish_query_pct");
    env->stockfish_query_pct = sfqp ? (int)sfqp->value : 100;
    if (env->stockfish_query_pct < 0) env->stockfish_query_pct = 0;
    if (env->stockfish_query_pct > 100) env->stockfish_query_pct = 100;

    if (_g_sf_random_pct < 0) {
        _g_sf_random_pct = env->stockfish_random_pct;
        _g_sf_random_pct_f = (float)env->stockfish_random_pct;
    }

    if (env->stockfish_bot) {
        env->random_bot = 0;
    }

    DictItem* mm = dict_get_unsafe(kwargs, "max_moves");
    env->max_moves = mm ? (int)mm->value : 500;

    DictItem* rfps = dict_get_unsafe(kwargs, "render_fps");
    env->render_fps = rfps ? (int)rfps->value : 30;

    env->num_agents = 1;  // Always 1 logical agent with doubled obs layout
    env->client = NULL;
    env->random_fen = 0;

    DictItem* fcp = dict_get_unsafe(kwargs, "fen_curric_pct");
    env->fen_curric_pct = fcp ? (float)fcp->value : 0.0f;

    if (env->fen_curric_pct > 0.0f) {
        load_fen_curriculum();
        env->fen_curriculum = _fen_curriculum;
        env->num_fens = _num_fens;
    } else {
        env->fen_curriculum = NULL;
        env->num_fens = 0;
    }

    DictItem* dmpct = dict_get_unsafe(kwargs, "deepmind_fen_pct");
    env->deepmind_fen_pct = dmpct ? (float)dmpct->value : 0.0f;

    if (env->deepmind_fen_pct > 0.0f) {
        load_fen_curriculum_dm_with_moves();
        env->fen_curriculum_dm = _fen_curriculum_dm;
        env->num_fens_dm = _num_fens_dm;
        env->tutor_moves_dm = _tutor_moves_dm;
    } else {
        env->fen_curriculum_dm = NULL;
        env->num_fens_dm = 0;
        env->tutor_moves_dm = NULL;
    }
    env->debug_mode = 0;
    env->stockfish_in = NULL;
    env->stockfish_out = NULL;
    env->stockfish_pid = -1;
    env->stockfish_ready = 0;

    // Alternate learner_color across envs.
    env->learner_color = _g_color_counter % 2;
    _g_color_counter++;

    // Prevent early-return bug in c_step (lines 2999-3006)
    env->log_pgn_choice_made = 1;
    DictItem* lpgn = dict_get_unsafe(kwargs, "log_pgn");
    env->log_pgn = lpgn ? (int)lpgn->value : 0;
    if (env->log_pgn) {
        static char _shared_pgn_filename[128] = {0};
        if (_shared_pgn_filename[0] == '\0') {
            snprintf(_shared_pgn_filename, sizeof(_shared_pgn_filename),
                     "eval_%d.pgn", (int)time(NULL));
            printf("PGN logging to: %s\n", _shared_pgn_filename);
        }
        strncpy(env->pgn_filename, _shared_pgn_filename, sizeof(env->pgn_filename) - 1);
    }

    // Reward shaping (configurable via ini)
    DictItem* rd = dict_get_unsafe(kwargs, "reward_draw");
    env->reward_draw = rd ? (float)rd->value : -0.5f;

    DictItem* rip = dict_get_unsafe(kwargs, "reward_invalid_piece");
    env->reward_invalid_piece = rip ? (float)rip->value : -0.01f;

    DictItem* rim = dict_get_unsafe(kwargs, "reward_invalid_move");
    env->reward_invalid_move = rim ? (float)rim->value : -0.01f;

    DictItem* rvp = dict_get_unsafe(kwargs, "reward_valid_piece");
    env->reward_valid_piece = rvp ? (float)rvp->value : 0.0f;

    DictItem* rvm = dict_get_unsafe(kwargs, "reward_valid_move");
    env->reward_valid_move = rvm ? (float)rvm->value : 0.0f;

    DictItem* rmat = dict_get_unsafe(kwargs, "reward_material");
    env->reward_material = rmat ? (float)rmat->value : 0.1f;

    DictItem* rpos = dict_get_unsafe(kwargs, "reward_position");
    env->reward_position = rpos ? (float)rpos->value : 0.0f;

    DictItem* rcast = dict_get_unsafe(kwargs, "reward_castling");
    env->reward_castling = rcast ? (float)rcast->value : 0.0f;

    DictItem* rrep = dict_get_unsafe(kwargs, "reward_repetition");
    env->reward_repetition = rrep ? (float)rrep->value : -0.05f;

    DictItem* rchk = dict_get_unsafe(kwargs, "reward_check");
    env->reward_check = rchk ? (float)rchk->value : 0.01f;

    // Move tutor config
    DictItem* rtp = dict_get_unsafe(kwargs, "reward_tutor_piece");
    env->reward_tutor_piece = rtp ? (float)rtp->value : 0.0f;

    DictItem* rtm = dict_get_unsafe(kwargs, "reward_tutor_move");
    env->reward_tutor_move = rtm ? (float)rtm->value : 0.0f;

    DictItem* rtw = dict_get_unsafe(kwargs, "reward_tutor_wrong");
    env->reward_tutor_wrong = rtw ? (float)rtw->value : 0.0f;

    DictItem* tom = dict_get_unsafe(kwargs, "tutor_only_mode");
    env->tutor_only_mode = tom ? (int)tom->value : 0;

    env->tutor_target = 0;
    env->tutor_phase = 0;

    DictItem* e50 = dict_get_unsafe(kwargs, "enable_50_move_rule");
    env->enable_50_move_rule = e50 ? (int)e50->value : 1;

    DictItem* e3r = dict_get_unsafe(kwargs, "enable_threefold_repetition");
    env->enable_threefold_repetition = e3r ? (int)e3r->value : 0;  // Disabled by default for training

    strcpy(env->starting_fen, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    // Initialize log
    memset(&env->log, 0, sizeof(Log));

    // NOTE: Do NOT call c_reset() here. Observation/action/reward/terminal
    // pointers are not yet assigned (they're set in create_static_vec after
    // my_vec_init returns). static_vec_reset() will call c_reset() for all
    // envs after pointers are assigned.

    // But we do need to set up the starting position (c_reset writes to
    // observations, which is NULL here). Set up position without obs.
    env->tick = 0;
    env->chess_moves = 0;
    env->game_result = 0;
    env->undo_stack_ptr = 0;
    env->invalid_actions_this_episode = 0;
    env->episode_reward = 0.0f;
    env->pgn_move_count = 0;
    env->show_game_end_popup = 0;
    env->opp_in_check = 0;
    env->pick_phase[0] = 0;
    env->pick_phase[1] = 0;
    env->selected_square[0] = 64; // SQ_NONE
    env->selected_square[1] = 64; // SQ_NONE
    env->valid_destinations[0].count = 0;
    env->valid_destinations[1].count = 0;
    memset(env->white_captured, 0, sizeof(env->white_captured));
    memset(env->black_captured, 0, sizeof(env->black_captured));
    pos_set(&env->pos, env->starting_fen);
    generate_legal(&env->pos, &env->legal_moves, env->undo_stack, &env->undo_stack_ptr);
    env->legal_moves_side = env->pos.sideToMove;
    env->legal_moves_key = env->pos.key;

    // Stockfish processes are no longer needed for training - the built-in
    // eval (builtin_select_move) replaces pipe-based Stockfish I/O.
    // stockfish_start() is only called lazily from stockfish_select_move()
    // for evaluation scripts that explicitly need it.
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "draw_rate", log->draw_rate);
    dict_set(out, "timeout_rate", log->timeout_rate);
    dict_set(out, "chess_moves", log->chess_moves);
    dict_set(out, "invalid_action_rate", log->invalid_action_rate);
    dict_set(out, "game_length_score", log->game_length_score);
    dict_set(out, "material_score", log->material_score);
    dict_set(out, "positional_score", log->positional_score);
    dict_set(out, "white_winrate", log->white_winrate);
    dict_set(out, "black_winrate", log->black_winrate);
    dict_set(out, "white_lossrate", log->white_lossrate);
    dict_set(out, "black_lossrate", log->black_lossrate);
    dict_set(out, "draw_by_stalemate", log->draw_by_stalemate);
    dict_set(out, "draw_by_insufficient", log->draw_by_insufficient);
    dict_set(out, "draw_by_50move", log->draw_by_50move);
    dict_set(out, "draw_by_repetition", log->draw_by_repetition);
    dict_set(out, "opponent_winrate", log->opponent_winrate);
    dict_set(out, "stockfish_random_pct", log->stockfish_random_pct);
    dict_set(out, "stockfish_query_pct", log->stockfish_query_pct);
    dict_set(out, "ema_winrate", log->ema_winrate);
    if (log->tutor_total > 0) {
        dict_set(out, "tutor_piece_rate", log->tutor_piece_match / log->tutor_total);
        dict_set(out, "tutor_move_rate", log->tutor_move_match / log->tutor_total);
    }
}
