#include "chess.h"
#include <libgen.h>
#include <unistd.h>

// Enable GPU-batched opponent support in env_binding.c
#define GPU_OPPONENT_SUPPORT 1

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

// Mate-in-N curriculum data
#define MATE_LEVELS 5
static char** _mate_fens[MATE_LEVELS] = {NULL};
static int _mate_fen_counts[MATE_LEVELS] = {0};
static int _mate_fens_loaded = 0;

// Midgame curriculum data (FEN + Stockfish best move)
static char** _fen_curriculum_midgame = NULL;
static uint16_t* _tutor_moves_midgame = NULL;
static int _num_fens_midgame = 0;
static int _fens_midgame_loaded = 0;

// Cap loaded positions to avoid OOM on large datasets.
// 10M positions ≈ 1GB RAM — plenty of diversity for training.
#define MAX_CURRICULUM_POSITIONS 10000000

#define MY_GET
void* my_get(void* env_ptr, Dict* out) {
    (void)env_ptr;
    dict_set(out, "sf_random_pct", (double)_g_sf_random_pct);
    dict_set(out, "sf_random_pct_f", (double)_g_sf_random_pct_f);
    dict_set(out, "ema_winrate", (double)_g_ema_wr);
    dict_set(out, "annealing_games", (double)_g_annealing_games);
    dict_set(out, "color_counter", (double)_g_color_counter);
    dict_set(out, "curriculum_phase", (double)_g_curriculum_phase);
    dict_set(out, "curriculum_ema", (double)_g_curriculum_ema);
    dict_set(out, "curriculum_games", (double)_g_curriculum_games);
    dict_set(out, "curriculum_advances", (double)_g_curriculum_advances);
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

    i = dict_get_unsafe(kwargs, "curriculum_phase");
    if (i) _g_curriculum_phase = (int)i->value;

    i = dict_get_unsafe(kwargs, "curriculum_ema");
    if (i) _g_curriculum_ema = (float)i->value;

    i = dict_get_unsafe(kwargs, "curriculum_games");
    if (i) _g_curriculum_games = (int)i->value;

    i = dict_get_unsafe(kwargs, "curriculum_advances");
    if (i) _g_curriculum_advances = (int)i->value;

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

// Helper: add a FEN+move line to a curriculum array, with reservoir sampling if at capacity.
// Returns 1 if added/replaced, 0 if skipped.
static int _curriculum_add_line(char*** fens, uint16_t** moves, int* count,
                                int max_cap, const char* fen_str, int fen_len,
                                uint16_t packed_move, long total_seen) {
    if (*count < max_cap) {
        (*fens)[*count] = (char*)malloc(fen_len + 1);
        strcpy((*fens)[*count], fen_str);
        (*moves)[*count] = packed_move;
        (*count)++;
        return 1;
    } else {
        // Reservoir sampling: replace random element with probability max_cap/total_seen
        long j = rand() % total_seen;
        if (j < max_cap) {
            free((*fens)[j]);
            (*fens)[j] = (char*)malloc(fen_len + 1);
            strcpy((*fens)[j], fen_str);
            (*moves)[j] = packed_move;
            return 1;
        }
        return 0;
    }
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

    int capacity = MAX_CURRICULUM_POSITIONS;
    _fen_curriculum_dm = (char**)malloc(capacity * sizeof(char*));
    _tutor_moves_dm = (uint16_t*)malloc(capacity * sizeof(uint16_t));
    _num_fens_dm = 0;
    long total_lines = 0;
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        int len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) {
            line[--len] = '\0';
        }
        if (len == 0) continue;
        total_lines++;

        // Split on tab
        char* tab = strchr(line, '\t');
        if (!tab) {
            _curriculum_add_line(&_fen_curriculum_dm, &_tutor_moves_dm,
                                &_num_fens_dm, capacity, line, len, 0, total_lines);
            continue;
        }

        *tab = '\0';
        const char* fen = line;
        const char* uci = tab + 1;
        int fen_len = strlen(fen);
        uint16_t packed = parse_uci_to_packed(uci);
        _curriculum_add_line(&_fen_curriculum_dm, &_tutor_moves_dm,
                            &_num_fens_dm, capacity, fen, fen_len, packed, total_lines);
    }
    fclose(f);
    fprintf(stderr, "Loaded DeepMind FEN+move curriculum: %d positions (sampled from %ld) from %s\n",
            _num_fens_dm, total_lines, path);
}

void load_mate_curriculum(void) {
    if (_mate_fens_loaded) return;
    _mate_fens_loaded = 1;

    char dir[512];
    strncpy(dir, __FILE__, sizeof(dir) - 1);
    dir[sizeof(dir) - 1] = '\0';
    char* d = dirname(dir);

    for (int level = 0; level < MATE_LEVELS; level++) {
        char path[1024];
        // Prefer full puzzle files (100K-800K) over 10K subsets to prevent
        // memorization with 16K agents (each puzzle seen 300x/epoch with 10K vs
        // ~1x/epoch with full set → forces pattern learning, not memorization)
        snprintf(path, sizeof(path), "%s/fens_mate_in_%d.txt", d, level + 1);
        // Fallback to 10k subset if full file doesn't exist
        if (access(path, F_OK) != 0) {
            snprintf(path, sizeof(path), "%s/fens_mate_in_%d_10k.txt", d, level + 1);
        }

        FILE* f = fopen(path, "r");
        if (!f) {
            fprintf(stderr, "WARNING: Could not open mate FEN file at %s\n", path);
            continue;
        }

        int capacity = 16384;
        _mate_fens[level] = (char**)malloc(capacity * sizeof(char*));
        _mate_fen_counts[level] = 0;
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            int len = strlen(line);
            while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) {
                line[--len] = '\0';
            }
            if (len == 0) continue;
            if (_mate_fen_counts[level] >= capacity) {
                capacity *= 2;
                _mate_fens[level] = (char**)realloc(_mate_fens[level], capacity * sizeof(char*));
            }
            _mate_fens[level][_mate_fen_counts[level]] = (char*)malloc(len + 1);
            strcpy(_mate_fens[level][_mate_fen_counts[level]], line);
            _mate_fen_counts[level]++;
        }
        fclose(f);
        fprintf(stderr, "Loaded mate-in-%d curriculum: %d positions from %s\n",
                level + 1, _mate_fen_counts[level], path);
    }
}

void load_midgame_curriculum(void) {
    if (_fens_midgame_loaded) return;
    _fens_midgame_loaded = 1;

    char dir[512];
    strncpy(dir, __FILE__, sizeof(dir) - 1);
    dir[sizeof(dir) - 1] = '\0';
    char* d = dirname(dir);
    char path[1024];
    snprintf(path, sizeof(path), "%s/fens_moves_midgame.txt", d);

    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "WARNING: Could not open midgame FEN file at %s\n", path);
        return;
    }

    int capacity = MAX_CURRICULUM_POSITIONS;
    _fen_curriculum_midgame = (char**)malloc(capacity * sizeof(char*));
    _tutor_moves_midgame = (uint16_t*)malloc(capacity * sizeof(uint16_t));
    _num_fens_midgame = 0;
    long total_lines = 0;
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        int len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) {
            line[--len] = '\0';
        }
        if (len == 0) continue;
        total_lines++;

        char* tab = strchr(line, '\t');
        if (!tab) {
            _curriculum_add_line(&_fen_curriculum_midgame, &_tutor_moves_midgame,
                                &_num_fens_midgame, capacity, line, len, 0, total_lines);
            continue;
        }

        *tab = '\0';
        const char* fen = line;
        const char* uci = tab + 1;
        int fen_len = strlen(fen);
        uint16_t packed = parse_uci_to_packed(uci);
        _curriculum_add_line(&_fen_curriculum_midgame, &_tutor_moves_midgame,
                            &_num_fens_midgame, capacity, fen, fen_len, packed, total_lines);
    }
    fclose(f);
    fprintf(stderr, "Loaded midgame FEN+move curriculum: %d positions (sampled from %ld) from %s\n",
            _num_fens_midgame, total_lines, path);
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
    DictItem* rmate = dict_get_unsafe(kwargs, "reward_mate");
    env->reward_mate = rmate ? (float)rmate->value : 0.0f;
    DictItem* rsyz = dict_get_unsafe(kwargs, "reward_syzygy");
    env->reward_syzygy = rsyz ? (float)rsyz->value : 0.0f;
    if (env->reward_syzygy != 0.0f) {
        const char* syzygy_path = getenv("PUFFER_SYZYGY_PATH");
        if (!syzygy_path) syzygy_path = "/home/spark-advantage/syzygy";
        init_syzygy(syzygy_path);
    }

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

    // Multi-phase curriculum config
    DictItem* mc = dict_get_unsafe(kwargs, "mate_curriculum");
    env->mate_curriculum = mc ? (int)mc->value : 0;

    DictItem* rmf = dict_get_unsafe(kwargs, "reward_mate_fail");
    env->reward_mate_fail = rmf ? (float)rmf->value : -2.0f;

    DictItem* mat = dict_get_unsafe(kwargs, "mate_advance_threshold");
    env->mate_advance_threshold = mat ? (float)mat->value : 0.90f;

    DictItem* rmp = dict_get_unsafe(kwargs, "reward_mate_progress");
    env->reward_mate_progress = rmp ? (float)rmp->value : 0.5f;

    DictItem* mretry = dict_get_unsafe(kwargs, "mate_retry");
    env->mate_retry = mretry ? (int)mretry->value : 1;

    env->mate_retry_idx = -1;      // -1 = no retry pending
    env->mate_retry_level = -1;
    env->mate_current_idx = 0;
    env->mate_current_level = 0;
    env->mate_mix_ratio = 0.0f;

    DictItem* mmf = dict_get_unsafe(kwargs, "mate_mix_floor");
    env->mate_mix_floor = mmf ? (float)mmf->value : 0.65f;

    // Puzzle drill mode
    DictItem* pdm = dict_get_unsafe(kwargs, "puzzle_drill_mode");
    env->puzzle_drill_mode = pdm ? (int)pdm->value : 0;
    DictItem* ppr = dict_get_unsafe(kwargs, "puzzle_piece_reward_0");
    env->puzzle_piece_reward_0 = ppr ? (float)ppr->value : 0.01f;
    DictItem* pdr = dict_get_unsafe(kwargs, "puzzle_dest_reward_0");
    env->puzzle_dest_reward_0 = pdr ? (float)pdr->value : 0.015f;
    DictItem* pri = dict_get_unsafe(kwargs, "puzzle_reward_increment");
    env->puzzle_reward_increment = pri ? (float)pri->value : 0.01f;
    DictItem* pdl = dict_get_unsafe(kwargs, "puzzle_drill_levels");
    env->puzzle_drill_levels = pdl ? (int)pdl->value : 3;
    env->puzzle_reward_accum = 0.0f;
    if (env->puzzle_drill_mode) fprintf(stderr, "PUZZLE DRILL MODE ENABLED (piece=%.3f dest=%.3f inc=%.3f levels=%d)\n", env->puzzle_piece_reward_0, env->puzzle_dest_reward_0, env->puzzle_reward_increment, env->puzzle_drill_levels);
    if (env->puzzle_drill_mode) env->random_bot = 1;
    env->puzzle_move_num = 0;
    env->curriculum_episode_type = 3;
    env->mate_level = 0;
    env->mate_max_moves = 0;

    if (env->mate_curriculum) {
        // Load mate FENs
        load_mate_curriculum();
        env->mate_fens = _mate_fens;
        env->mate_fen_counts = _mate_fen_counts;

        // Load midgame curriculum (for phase 5)
        load_midgame_curriculum();
        env->fen_curriculum_midgame = _fen_curriculum_midgame;
        env->tutor_moves_midgame = _tutor_moves_midgame;
        env->num_fens_midgame = _num_fens_midgame;

        // Ensure endgame data is loaded for phase 6
        if (env->fen_curriculum_dm == NULL) {
            load_fen_curriculum_dm_with_moves();
            env->fen_curriculum_dm = _fen_curriculum_dm;
            env->num_fens_dm = _num_fens_dm;
            env->tutor_moves_dm = _tutor_moves_dm;
        }
    } else {
        env->mate_fens = NULL;
        env->mate_fen_counts = NULL;
        env->fen_curriculum_midgame = NULL;
        env->tutor_moves_midgame = NULL;
        env->num_fens_midgame = 0;
    }

    // GPU-batched opponent mode
    DictItem* gpo = dict_get_unsafe(kwargs, "gpu_opponent");
    env->gpu_opponent = gpo ? (int)gpo->value : 0;
    env->gpu_opponent_pending = 0;

    // Mate-in-1 detection rewards
    DictItem* rmt = dict_get_unsafe(kwargs, "reward_mate_threat");
    env->reward_mate_threat = rmt ? (float)rmt->value : 0.1f;

    DictItem* rmd = dict_get_unsafe(kwargs, "reward_mate_defense");
    env->reward_mate_defense = rmd ? (float)rmd->value : 0.05f;

    DictItem* ram = dict_get_unsafe(kwargs, "reward_allowed_mate");
    env->reward_allowed_mate = ram ? (float)ram->value : -0.1f;

    DictItem* bnc = dict_get_unsafe(kwargs, "builtin_noise_cp");
    env->builtin_noise_cp = bnc ? (int)bnc->value : 150;

    DictItem* bd = dict_get_unsafe(kwargs, "builtin_depth");
    env->builtin_depth = bd ? (int)bd->value : 2;

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

    // Shared Stockfish process pool (one SF process per OMP thread)
    DictItem* sfps = dict_get_unsafe(kwargs, "stockfish_pool_size");
    int stockfish_pool_size = sfps ? (int)sfps->value : 0;  // 0=auto, -1=disabled

    if (env->stockfish_bot && stockfish_pool_size != -1 && !_g_sf_pool_initialized) {
        int pool_sz = (stockfish_pool_size > 0) ? stockfish_pool_size : 128;
        int pool_depth = env->stockfish_depth > 0 ? env->stockfish_depth : 5;
        _g_sf_pool = sf_pool_create(pool_sz, pool_depth);
        _g_sf_pool_initialized = 1;
    }
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
    dict_set(out, "syzygy_probes", log->syzygy_probes);
    dict_set(out, "syzygy_wins", log->syzygy_wins);
    dict_set(out, "syzygy_draws", log->syzygy_draws);
    dict_set(out, "syzygy_reward_total", log->syzygy_reward_total);
    if (log->curriculum_n > 0) {
        dict_set(out, "curriculum_phase", log->curriculum_phase / log->curriculum_n);
        dict_set(out, "curriculum_ema", log->curriculum_ema / log->curriculum_n);
        dict_set(out, "curriculum_success_rate", log->curriculum_success / log->curriculum_n);
        dict_set(out, "curriculum_count", log->curriculum_n);
    }

    // Puzzle drill stats
    if (log->puzzle_n > 0) {
        dict_set(out, "puzzle_attempts", log->puzzle_attempts);
        dict_set(out, "puzzle_solves", log->puzzle_solves);
        dict_set(out, "puzzle_wrong_piece", log->puzzle_wrong_piece);
        dict_set(out, "puzzle_wrong_dest", log->puzzle_wrong_dest);
        dict_set(out, "puzzle_timeouts", log->puzzle_timeouts);
        dict_set(out, "puzzle_piece_acc", log->puzzle_piece_correct / log->puzzle_n);
        dict_set(out, "puzzle_dest_acc", log->puzzle_dest_correct / log->puzzle_n);
        dict_set(out, "puzzle_solve_rate", log->puzzle_solves / log->puzzle_n);
        dict_set(out, "puzzle_reward_total", log->puzzle_reward_total / log->puzzle_n);
        dict_set(out, "puzzle_level", log->puzzle_level / log->puzzle_n);
        dict_set(out, "puzzle_fail_l1", log->puzzle_fail_l1);
        dict_set(out, "puzzle_fail_l2", log->puzzle_fail_l2);
        dict_set(out, "puzzle_fail_l3", log->puzzle_fail_l3);
    }
    // Per-level unique solved counts
    dict_set(out, "puzzle_unique_l1", (float)_puzzle_solved_count[0]);
    dict_set(out, "puzzle_unique_l2", (float)_puzzle_solved_count[1]);
    dict_set(out, "puzzle_unique_l3", (float)_puzzle_solved_count[2]);
}
