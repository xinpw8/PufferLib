// flat_chess_env.h – REBUILT FROM SCRATCH
#pragma once

/* -------------------------------------------------------------------------
   PufferLib Ocean – Flat (single-translation-unit) Chess Environment
   -------------------------------------------------------------------------
   This header embeds OpenSpiel's reference chess engine directly (by text
   including its *.cc files) and then exposes the small C interface expected
   by the Ocean runtime.  The wrapper follows the "Go-style" pattern used by
   other Ocean envs: the C-visible Env struct only stores raw C buffers plus a
   pointer to a heap-allocated C++ context that owns all complex state.
   -----------------------------------------------------------------------*/

// ----- Upstream engine ----------------------------------------------------
#include "chess_common.h"
#include "chess_board.h"
// NOTE: these straight-line includes make this header self-contained.
#include "chess_common.cc"
#include "chess_board.cc"
#include "chess960_starting_positions.cc"

#ifdef __cplusplus
extern "C" {
#endif

// -------------------------------------------------------------------------
// 1.  Plain-C PODs passed to/returned from Python
// -------------------------------------------------------------------------

// Minimal logging structure required by env_binding.h
typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;               // must be last
} Log;

// -------------------------------------------------------------------------
// 2.  Hidden C++ context (allocated once per env)
// -------------------------------------------------------------------------
#ifdef __cplusplus
#include <vector>
#include <unordered_map>
#include <random>
#endif

typedef struct ChessContext {
    // Core engine objects
    open_spiel::chess::ChessBoard board;
    std::vector<open_spiel::chess::Move> legal_moves;
    std::unordered_map<uint64_t,int> repetition;   // 3-fold repetition

    // RNG (used for black's random reply)
    std::mt19937 rng;

    // Episode tracking
    int   step   = 0;
    float ep_ret = 0.f;

    // Reward config (filled by Python kwargs)
    float r_valid     = +0.01f;
    float r_invalid   = -0.001f;
    float r_capture   = +0.01f;
    float r_captured  = -0.01f;
    float r_win       = +1.0f;
    float r_draw      =  0.0f;
    float r_loss      = -1.0f;
} ChessContext;

// -------------------------------------------------------------------------
// 3.  C-visible struct (allocated by env_binding.h)
// -------------------------------------------------------------------------

typedef struct CChess {
    // Raw buffers (owned by Python)
    float*          observations;
    int*            actions;
    float*          rewards;
    unsigned char*  terminals;

    Log             log;        // aggregated by PufferLib

    ChessContext*   ctx;        // heap object – created in allocate()
} CChess;

// Observation layout constants
enum { OBS_SIZE = 2560 };     // 768 board + 256 mask + 1536 move-enc

// -------------------------------------------------------------------------
// 4.  Helper (internal) – build observation & legal move list
// -------------------------------------------------------------------------
#ifdef __cplusplus
static void compute_observation(CChess* env)
{
    ChessContext& C = *env->ctx;
    float* obs      = env->observations;
    if(!obs) return;
    std::fill(obs, obs + OBS_SIZE, 0.f);

    // (a) one-hot board   [64 * 12  = 768]
    for(int y=0; y<8; ++y) for(int x=0; x<8; ++x) {
        auto sq    = open_spiel::chess_common::Square{(int8_t)x,(int8_t)y};
        const auto& p = C.board.at(sq);
        if(p.type == open_spiel::chess::PieceType::kEmpty) continue;
        int idx   = y*8 + x;
        int plane = (p.color == open_spiel::chess::Color::kWhite ? 0:6) + int(p.type)-1;
        obs[idx*12 + plane] = 1.f;
    }

    // (b) generate legal moves list (white to play only)
    C.legal_moves.clear();
    if(C.board.ToPlay() == open_spiel::chess::Color::kWhite) {
        C.board.GenerateLegalMoves([&](const open_spiel::chess::Move& mv){
            if(C.legal_moves.size()<256) C.legal_moves.push_back(mv);
            return true;
        });
    }

    // (c) legal-move mask  [offset 768]
    for(size_t i=0;i<C.legal_moves.size();++i) obs[768+i] = 1.f;

    // (d) move encodings   [offset 1024]  each move => 6 floats
    for(size_t i=0;i<C.legal_moves.size() && i<256;++i){
        const auto& m = C.legal_moves[i];
        float* dst = obs + 1024 + i*6;
        dst[0] = m.from.x / 7.f;  dst[1] = m.from.y / 7.f;
        dst[2] = m.to.x   / 7.f;  dst[3] = m.to.y   / 7.f;
        dst[4] = float(int(m.piece.type)-1);
        dst[5] = (m.piece.color==open_spiel::chess::Color::kWhite?0.f:1.f);
    }
}
#endif

// -------------------------------------------------------------------------
// 5.  C API – called from Python
// -------------------------------------------------------------------------

static void c_init(CChess* env) { /* called by binding after kwargs parsed */ }

static void allocate(CChess* env)
{
    env->ctx = new ChessContext();
    env->ctx->rng.seed(std::random_device{}());
    compute_observation(env);
}

static void free_allocated(CChess* env)
{
    delete env->ctx; env->ctx=nullptr;
}

static void c_reset(CChess* env)
{
    ChessContext& C=*env->ctx; C.board = open_spiel::chess::MakeDefaultBoard();
    C.repetition.clear(); C.step=0; C.ep_ret=0.f;
    compute_observation(env);
    env->terminals[0]=0; env->rewards[0]=0.f;
}

// Forward declaration so c_step can call it
static void add_log(CChess* env);

static void c_step(CChess* env)
{
    ChessContext& C=*env->ctx;
    const int act = env->actions[0];
    env->rewards[0]=0.f; env->terminals[0]=0;
    C.step++;

    // ensure legal list current (white to move)
    if(C.legal_moves.empty()) { env->terminals[0]=1; add_log(env); return; }
    int idx = (act>=0? act%int(C.legal_moves.size()):0);
    const auto& myMove = C.legal_moves[idx];
    if(!C.board.IsMoveLegal(myMove)) {
        env->rewards[0]+=C.r_invalid; env->terminals[0]=1; add_log(env); return; }
    if(C.board.at(myMove.to).color!=open_spiel::chess::Color::kEmpty)
        env->rewards[0]+=C.r_capture;
    C.board.ApplyMove(myMove);

    // simple terminal checks (material & legal moves)
    if(!C.board.HasSufficientMaterial()) { env->rewards[0]+=C.r_draw; env->terminals[0]=1; add_log(env); return; }

    // Black random reply
    std::vector<open_spiel::chess::Move> replies;
    C.board.GenerateLegalMoves([&](const open_spiel::chess::Move& mv){ replies.push_back(mv); return true; });
    if(replies.empty()) { env->rewards[0]+=C.r_win; env->terminals[0]=1; add_log(env); return; }
    const auto& opp = replies[C.rng()%replies.size()];
    if(C.board.at(opp.to).color!=open_spiel::chess::Color::kEmpty)
        env->rewards[0]+=C.r_captured;
    C.board.ApplyMove(opp);

    if(!C.board.HasSufficientMaterial()) { env->rewards[0]+=C.r_draw; env->terminals[0]=1; add_log(env); return; }

    // update observation for next white move
    compute_observation(env);
    C.ep_ret+=env->rewards[0];
}

static void c_render(CChess* env)
{
    printf("%s\n", env->ctx->board.DebugString().c_str());
}

static void c_close(CChess* env) { free_allocated(env); }

static void add_log(CChess* env)
{
    ChessContext& C=*env->ctx; env->log.episode_return=C.ep_ret; env->log.episode_length=float(C.step);
    env->log.n=1.f;
    if(env->rewards[0]>0.5f) { env->log.perf=1; env->log.score=1; }
    else if(env->rewards[0]<-0.5f){ env->log.perf=0; env->log.score=-1; } else { env->log.perf=0.5f; env->log.score=0; }
    C.ep_ret=0; C.step=0;
}

#ifdef __cplusplus
} // extern "C"
#endif 