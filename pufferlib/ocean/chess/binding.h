#include "chess.h"

// Selfplay mode: doubled obs [learner(1082) | opponent(1082)], interleaved actions [learner(97) | opponent(97)]
#define OBS_SIZE 2164
#define NUM_ATNS 2
#define ACT_SIZES {97, 97}
#define OBS_TYPE UNSIGNED_CHAR
#define ACT_TYPE DOUBLE

#define Env Chess
#include "env_binding.h"

static int _chess_bitboards_initialized = 0;

void my_init(Env* env, Dict* kwargs) {
    // Initialize bitboards once (global chess tables)
    if (!_chess_bitboards_initialized) {
        init_bitboards();
        _chess_bitboards_initialized = 1;
    }

    DictItem* sp = dict_get_unsafe(kwargs, "selfplay");
    env->selfplay = sp ? (int)sp->value : 0;

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

    if (env->stockfish_bot) {
        env->random_bot = 0;
    }

    DictItem* mm = dict_get_unsafe(kwargs, "max_moves");
    env->max_moves = mm ? (int)mm->value : 500;

    env->num_agents = 1;  // Always 1 logical agent with doubled obs layout
    env->human_play = 0;
    env->client = NULL;
    env->fen_curriculum = NULL;
    env->num_fens = 0;
    env->random_fen = 0;
    env->debug_mode = 0;
    env->stockfish_in = NULL;
    env->stockfish_out = NULL;
    env->stockfish_pid = -1;
    env->stockfish_ready = 0;

    // Alternate learner_color across envs (using static counter)
    static int _color_counter = 0;
    env->learner_color = _color_counter % 2;
    _color_counter++;

    // Prevent early-return bug in c_step (lines 2999-3006)
    env->log_pgn_choice_made = 1;
    env->log_pgn = 0;

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

    // Start Stockfish process eagerly during init (before CUDA training loop)
    if (env->stockfish_bot) {
        stockfish_start(env);
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
}
