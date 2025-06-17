#pragma once

// Single-translation-unit Chess environment for PufferLib Ocean
// -------------------------------------------------------------
// We embed DeepMind OpenSpiel's reference chess implementation directly by
// text-including its .cc files, then wrap it with the small C interface that
// every Ocean env provides (allocate/c_step/…).  No external OpenSpiel build
// is required – all sources become part of this header.
//
// Only the wrapper code below is ours (public domain).  The embedded engine
// remains Apache-2.0 as per the upstream license notices inside each file.

// ---------------------------------------------------------------------------
// Upstream OpenSpiel sources
// ---------------------------------------------------------------------------
#include "chess_common.h"
#include "chess_board.h"

#include "chess_common.cc"
#include "chess_board.cc"
#include "chess960_starting_positions.cc"

#ifdef __cplusplus
extern "C" {
#endif

// ---- PufferLib required structs & prototypes --------------------------------

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct CChess CChess;

void allocate(CChess* env);
void free_allocated(CChess* env);
void c_reset(CChess* env);
void c_step(CChess* env);
void c_render(CChess* env);
void c_close(CChess* env);

#ifdef __cplusplus
} // extern "C"
#endif

// ---------------------------------------------------------------------------
// Implementation (C++)
// ---------------------------------------------------------------------------
#ifdef __cplusplus

#include <vector>
#include <cstring>
#include <iostream>
#include <functional>

namespace open_spiel {
// Provide a minimal definition so the linker can resolve it when only the
// chess engine is embedded (we don't pull the full OpenSpiel error handler).
[[noreturn]] inline void SpielFatalError(const std::string& msg) {
    std::cerr << "[OpenSpiel] Fatal error: " << msg << std::endl;
    std::abort();
}
} // namespace open_spiel

struct CChess {
    float* observations = nullptr;
    int*   actions      = nullptr;
    float* rewards      = nullptr;
    unsigned char* terminals = nullptr;

    Log log{};

    open_spiel::chess::ChessBoard board;

    int obs_size = 0;
    int cur_player = 0;
};

static int calc_obs_size() {
    // 8x8 board, one float per square
    return 64;
}

static void write_observation(CChess* env) {
    const int board_size = env->board.BoardSize();
    for (int y = 0; y < board_size; ++y) {
        for (int x = 0; x < board_size; ++x) {
            open_spiel::chess_common::Square sq{static_cast<int8_t>(x), static_cast<int8_t>(y)};
            const auto& piece = env->board.at(sq);
            float code = 0.0f;
            if (piece.type != open_spiel::chess::PieceType::kEmpty) {
                int sign = (piece.color == open_spiel::chess::Color::kWhite) ? 1 : -1;
                code = static_cast<float>(sign * static_cast<int>(piece.type));
            }
            env->observations[y * board_size + x] = code;
        }
    }
}

extern "C" void allocate(CChess* env) {
    env->obs_size    = calc_obs_size();
    env->observations = (float*)calloc(env->obs_size, sizeof(float));
    env->actions      = (int*)calloc(1, sizeof(int));
    env->rewards      = (float*)calloc(1, sizeof(float));
    env->terminals    = (unsigned char*)calloc(1, sizeof(unsigned char));

    env->board  = open_spiel::chess::MakeDefaultBoard();

    write_observation(env);
}

extern "C" void free_allocated(CChess* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    env->observations = nullptr;
    env->actions = nullptr;
    env->rewards = nullptr;
    env->terminals = nullptr;

    // No heap-allocated chess objects inside env->board
}

extern "C" void c_reset(CChess* env) {
    env->board = open_spiel::chess::MakeDefaultBoard();
    env->terminals[0] = 0;
    env->rewards[0] = 0.0f;
    write_observation(env);
}

extern "C" void c_step(CChess* env) {
    int a = env->actions[0];
    // Encode move as (from_square_index << 6) | to_square_index, each 0-63
    if (a < 0 || a >= 4096) {
        env->rewards[0] = -1.0f;
        write_observation(env);
        return;
    }

    int from_idx = (a >> 6) & 63;
    int to_idx   = a & 63;

    open_spiel::chess_common::Square from{static_cast<int8_t>(from_idx % 8), static_cast<int8_t>(from_idx / 8)};
    open_spiel::chess_common::Square to  {static_cast<int8_t>(to_idx  % 8), static_cast<int8_t>(to_idx  / 8)};

    bool legal_found = false;
    open_spiel::chess::Move chosen_move;

    env->board.GenerateLegalMoves([&](const open_spiel::chess::Move& mv) {
        if (!legal_found && mv.from == from && mv.to == to) {
            legal_found = true;
            chosen_move = mv;
            return false; // stop early
        }
        return true; // continue
    });

    if (!legal_found) {
        env->rewards[0] = -1.0f;
        write_observation(env);
        return;
    }

    env->board.ApplyMove(chosen_move);

    if (!env->board.HasLegalMoves()) {
        env->terminals[0] = 1;
        env->rewards[0] = env->board.InCheck() ? 1.0f : 0.0f;
    } else {
        env->terminals[0] = 0;
        env->rewards[0] = 0.0f;
    }

    write_observation(env);
}

extern "C" void c_render(CChess* env) {
    std::cout << env->board.DebugString() << std::endl;
}

extern "C" void c_close(CChess* /*env*/) {}

#endif // __cplusplus 