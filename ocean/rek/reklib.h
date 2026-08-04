// Core simulation for the REK G1 combat env: fighter state, the set-move state
// machine, hit resolution, the balance/down model, and REK's round rules.
//
// Deliberately not an articulated-body sim. REK's pilot picks canned moves over
// a self-balancing controller, so joint-level dynamics would model a layer the
// pilot never touches while costing ~20x the budget needed to hit 10M SPS.
// What matters for parity is the move envelope, what it can reach, and what
// puts a robot on the floor.

#pragma once

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include "moves.h"

// Locomotion head: neutral plus the 8 WASD combinations, ego-relative.
#define NUM_MOVE_DIRS 9

// Three discrete heads, one per thing a REK pilot's keyboard can express:
// where to walk, which set move to fire, and whether the guard is up.
#define REK_NUM_ATNS 3

// REK match rules: a down costs the faller a point, 3 downs ends the match.
#define REK_DOWNS_TO_LOSE 3

// Frozen historical opponent banks, matching ocean/robocode's cap.
#define REK_MAX_BANKS 2

// Depth of the action-latency shift register used by domain randomisation.
#define REK_MAX_LATENCY 3

// Feature block sizes. OBS_SIZE in binding.c is derived from these, so adding a
// feature here propagates without touching the binding.
#define REK_SCALARS_PER_FIGHTER 14
#define REK_FIGHTER_FEATURES (REK_SCALARS_PER_FIGHTER + NUM_MOVE_DEFS)
#define REK_RELATIVE_FEATURES 7
#define REK_CLOCK_FEATURES 1
#define REK_OBS_SIZE (2 * REK_FIGHTER_FEATURES + REK_RELATIVE_FEATURES + REK_CLOCK_FEATURES)

typedef struct Log Log;
struct Log {
    float perf;             // normalised round score, 0..1
    float score;            // hits landed minus downs taken
    float episode_return;
    float episode_length;
    float hits_landed;
    float hits_taken;
    float downs;            // times this fighter hit the floor
    float knockouts;        // rounds ended by reaching 3 downs
    float guard_uptime;     // fraction of frames spent guarding
    float whiff_rate;       // moves started that never connected
    // Selfplay-pool accounting, mirroring ocean/robocode. hist_* track results
    // against frozen historical opponents; slot_* let match() read a win rate
    // straight off eval_log.
    float hist_score;
    float hist_n;
    float hist_score_bank[REK_MAX_BANKS];
    float hist_n_bank[REK_MAX_BANKS];
    float slot_0_score;
    float slot_1_score;
    float draw_rate;
    float n;
};

typedef struct {
    float x, z;          // root position on the arena floor, metres
    float vx, vz;        // root velocity, m/s
    float yaw;           // facing, radians
    float balance;       // 0 = planted, >= 1 = on the floor
    int move;            // active move id, 0 = none
    int frame;           // frames elapsed inside the active move
    int move_connected;  // move already scored; stops one swing multi-hitting
    int guard;           // guard raised this frame
    int stun;            // hitstun frames remaining
    int down_timer;      // get-up frames remaining, 0 = standing
    int hits;            // clean hits landed this round
    int downs;           // times floored this round
    int moves_started;   // for whiff-rate logging
    int moves_whiffed;
    int guard_frames;
} Fighter;

// REK's scoreboard: clean hits, minus a point for every time you go down.
static inline int rek_score(const Fighter* f) {
    return f->hits - f->downs;
}

static inline bool rek_committed(const Fighter* f) {
    return f->move != 0;
}

static inline bool rek_actionable(const Fighter* f) {
    return f->down_timer == 0 && f->stun == 0 && !rek_committed(f);
}

// xorshift32. rand_r() is the ocean convention but shows up in profiles at this
// step rate; this is deterministic per env and inlines to a few instructions.
static inline uint32_t rek_rand(uint32_t* s) {
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x ? x : 0x9e3779b9u;
    return *s;
}

static inline float rek_randf(uint32_t* s) {
    return (float)(rek_rand(s) >> 8) * (1.0f / 16777216.0f);
}

static inline float rek_uniform(uint32_t* s, float lo, float hi) {
    return lo + (hi - lo) * rek_randf(s);
}

// Signed angle from `from` to `to`, wrapped to [-pi, pi].
static inline float rek_angle_delta(float from, float to) {
    float d = to - from;
    while (d > (float)M_PI) d -= 2.0f * (float)M_PI;
    while (d < -(float)M_PI) d += 2.0f * (float)M_PI;
    return d;
}

// Ego-relative locomotion. Index 0 is stand still; 1..8 walk the 8 compass
// directions relative to facing, so "forward" is always toward the opponent
// once the lock-on has slewed around.
static inline void rek_move_dir(int action, float* out_fwd, float* out_side) {
    static const float FWD[NUM_MOVE_DIRS]  = {0.0f, 1.0f, 0.707f, 0.0f, -0.707f, -1.0f, -0.707f,  0.0f,  0.707f};
    static const float SIDE[NUM_MOVE_DIRS] = {0.0f, 0.0f, 0.707f, 1.0f,  0.707f,  0.0f, -0.707f, -1.0f, -0.707f};
    if (action < 0 || action >= NUM_MOVE_DIRS) action = 0;
    *out_fwd = FWD[action];
    *out_side = SIDE[action];
}
