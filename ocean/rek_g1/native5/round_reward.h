#pragma once
#include <cmath>

#if defined(__CUDACC__)
#define REK_REWARD_FN __host__ __device__ inline
#else
#define REK_REWARD_FN inline
#endif

// These functions consume awarded scoreboard points, never animation contacts,
// attack requests, inferred hits, or visual fall flags. They do not generate
// points or decide when a round ends.
namespace rek5_round_reward {
enum Mode { PointDifference = 0, RoundOutcome = 1 };

REK_REWARD_FN float potential(int own_points,int opponent_points) {
    const float difference=float(own_points)-float(opponent_points);
    // Five points is one recovered G1 countout award. This bounded potential
    // supplies dense score feedback; it does not change the terminal objective.
    return difference/(5.f+fabsf(difference));
}

REK_REWARD_FN float value(Mode mode,float gamma,int previous_own,int previous_opponent,
        int own_points,int opponent_points,bool round_terminal,int winner,int side) {
    if(mode==PointDifference)
        return float(own_points-previous_own)-float(opponent_points-previous_opponent);
    const float before=potential(previous_own,previous_opponent);
    const float after=round_terminal?0.f:potential(own_points,opponent_points);
    const float outcome=round_terminal&&winner>=0?(winner==side?1.f:-1.f):0.f;
    return outcome+gamma*after-before;
}
}
#undef REK_REWARD_FN
