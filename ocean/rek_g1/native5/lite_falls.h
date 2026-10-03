#pragma once
// Lite falls: a learned, state-conditioned fall model for the compact runtime.
//
// The compact runtime has no articulated balance. This module supplies the
// missing fall events from a model fitted to the physical MuJoCo + SONIC
// runtime (see lite_fall_dataset.cu and fit_lite_falls.py), then applies the
// recovered REK referee exactly:
//   * onset hazard per fighter tick, conditioned on own move/phase, opponent
//     move/phase, distance, closing speed and a received scored hit;
//   * after onset the fighter is falling (not upright): it recovers, becomes
//     fallen quickly, or stays down uncounted ("stuck") before becoming fallen;
//   * every BECAME_FALLEN starts the no-recovery count (G1 cannot get up):
//     3 s, restarted by a second fall (double down);
//   * count expiry awards 5 points to the opponent (5 to each on a double
//     count) and resets both fighters to spawn with the 2 s fall grace. A
//     stuck, uncounted fighter is cleared by that reset without conceding.
//   * hits score only while both fighters are upright (requireBothUpright).
// Classification (slip/knockdown) uses the recovered strike window. With
// CanGetUp false it changes only the referee call, not the points.
#include <stdint.h>
#include <math.h>

#if defined(__CUDACC__)
#define REK_LITE_FN __host__ __device__ inline
#else
#define REK_LITE_FN inline
#endif

namespace rek_lite_falls {

constexpr int kMoves=17, kPhaseBins=4;
constexpr int kIdleBin=kMoves*kPhaseBins, kTranslateBin=kIdleBin+1, kYawBin=kIdleBin+2;
constexpr int kMoveBins=kIdleBin+3;
constexpr int kDistanceBins=5, kClosingBins=3;
constexpr int kOutcomeClasses=3;  // 0 locomotion/idle, 1 hand move, 2 kick/knee
constexpr int kOutcomes=3;        // Recover, FallenQuick, Stuck
constexpr int kQuantiles=9;       // 0, 12.5, ..., 100 percent of the delay
constexpr int kTicksPerSecond=50;
constexpr int kCountTicks=150;       // noRecoveryCountSeconds 3.0
constexpr int kSpawnGraceTicks=100;  // ResetToSpawn post-reset fall grace 2.0 s
constexpr int kKnockoutPoints=5;     // koPoints

enum Phase : int32_t { Upright=0, Falling=1, Fallen=2 };
enum Outcome : int32_t { Recover=0, FallenQuick=1, Stuck=2 };
// Same bit values as RekG1FallEvent in g1_fall_state.h.
enum Event : uint32_t { FallingStarted=1u, FallingCleared=2u, BecameFallen=4u };
enum Classification : int32_t { NoFall=0, Slip=1, Knockdown=2 };
// RefereeCall order from the recovered build, as bit positions.
enum Call : uint32_t { CallSlip=1u<<0, CallKnockdown=1u<<2, CallKnockout=1u<<4,
    CallDoubleKnockdown=1u<<5, CallDoubleKnockout=1u<<6 };

// Native route ids 7..23 are discrete moves in registry order.
REK_LITE_FN int route_move(int route){
    const int moves[17]={6,7,8,9,0,1,2,3,4,5,10,11,12,13,14,15,16};
    return route>=7&&route<24?moves[route-7]:-1;
}
REK_LITE_FN int move_class(int move){return move<0?0:(move>=6&&move<=9)?2:1;}
REK_LITE_FN int move_bin(int move,float progress,bool translating,bool yawing){
    if(move>=0&&move<kMoves){
        int bin=int(progress*kPhaseBins);
        bin=bin<0?0:bin>=kPhaseBins?kPhaseBins-1:bin;
        return move*kPhaseBins+bin;
    }
    return translating?kTranslateBin:yawing?kYawBin:kIdleBin;
}
REK_LITE_FN int bucket(float value,const float* edges,int edge_count){
    int bin=0;
    for(int i=0;i<edge_count;i++)bin+=value>=edges[i];
    return bin;
}

struct Model {
    int32_t enabled;
    float distance_edges[kDistanceBins-1];   // metres between roots
    float closing_edges[kClosingBins-1];     // metres per second, positive closing
    float bias;
    float own_move[kMoveBins], opponent_move[kMoveBins];
    float distance[kDistanceBins], closing[kClosingBins];
    float own_move_distance[kMoveBins*kDistanceBins];
    float opponent_move_distance[kMoveBins*kDistanceBins];
    float struck;                            // received a scored hit this tick
    float outcome_probability[kOutcomeClasses*kOutcomes];
    float delay_quantiles[kOutcomes*kQuantiles]; // ticks from onset to resolution
    // Observation proxies for the detector fields; not used by the dynamics.
    float falling_tilt_degrees, fallen_tilt_degrees;
    float falling_height_ratio, fallen_height_ratio;
};

struct Features {
    int own_bin, opponent_bin, distance_bin, closing_bin;
    int struck;
};

REK_LITE_FN Features features(const Model& m,int own_move,float own_progress,bool own_translating,
        bool own_yawing,int opponent_move,float opponent_progress,bool opponent_translating,
        bool opponent_yawing,float distance,float closing_speed,bool struck){
    Features f;
    f.own_bin=move_bin(own_move,own_progress,own_translating,own_yawing);
    f.opponent_bin=move_bin(opponent_move,opponent_progress,opponent_translating,opponent_yawing);
    f.distance_bin=bucket(distance,m.distance_edges,kDistanceBins-1);
    f.closing_bin=bucket(closing_speed,m.closing_edges,kClosingBins-1);
    f.struck=struck?1:0;
    return f;
}
REK_LITE_FN float logit(const Model& m,const Features& f){
    return m.bias+m.own_move[f.own_bin]+m.opponent_move[f.opponent_bin]
        +m.distance[f.distance_bin]+m.closing[f.closing_bin]
        +m.own_move_distance[f.own_bin*kDistanceBins+f.distance_bin]
        +m.opponent_move_distance[f.opponent_bin*kDistanceBins+f.distance_bin]
        +float(f.struck)*m.struck;
}
REK_LITE_FN float hazard(const Model& m,const Features& f){
    const float z=logit(m,f);
    return z>=0?1.f/(1.f+expf(-z)):expf(z)/(1.f+expf(z));
}
// Inverse CDF through linearly interpolated quantiles. At least one tick.
REK_LITE_FN int sample_ticks(const float* quantiles,float u){
    float x=u*float(kQuantiles-1);
    int i=int(x);if(i>=kQuantiles-1)i=kQuantiles-2;if(i<0)i=0;
    const float t=x-float(i);
    const float ticks=quantiles[i]+t*(quantiles[i+1]-quantiles[i]);
    const int whole=int(ceilf(ticks));
    return whole<1?1:whole;
}
REK_LITE_FN int sample_outcome(const Model& m,int outcome_class,float u){
    const float* p=m.outcome_probability+outcome_class*kOutcomes;
    if(u<p[Recover])return Recover;
    if(u<p[Recover]+p[FallenQuick])return FallenQuick;
    return Stuck;
}
// FightCoordinator knockdown attribution window (knockdownWeakSpeed 1.75,
// knockdownStrongSpeed 6.0, window 1.5 to 5.0 s).
REK_LITE_FN bool knockdown(bool struck_valid,float struck_age_seconds,float struck_speed){
    if(!struck_valid)return false;
    float t=(struck_speed-1.75f)/(6.0f-1.75f);t=t<0?0:t>1?1:t;
    return struck_age_seconds<=1.5f+t*3.5f;
}

struct FighterState {
    int32_t phase, outcome, ticks, resolve_ticks, grace_ticks, classification;
    uint32_t events;   // this tick
};
struct Referee {
    int32_t count_active[2], count_is_slip[2], count_ticks;
    uint32_t calls;    // this tick
};
struct ArenaState {
    FighterState fighter[2];
    Referee referee;
    float previous_distance;
    int32_t has_previous_distance;
};

REK_LITE_FN bool upright(const FighterState& f){return f.phase==Upright;}
REK_LITE_FN bool counting(const Referee& r){return r.count_active[0]||r.count_active[1];}

REK_LITE_FN void reset_to_spawn(ArenaState& s){
    for(int side=0;side<2;side++){
        FighterState& f=s.fighter[side];const uint32_t events=f.events;
        f=FighterState{};f.grace_ticks=kSpawnGraceTicks;f.events=events;
    }
    const uint32_t calls=s.referee.calls;s.referee=Referee{};s.referee.calls=calls;
    s.has_previous_distance=0;
}
REK_LITE_FN void begin_tick(ArenaState& s){
    s.fighter[0].events=s.fighter[1].events=0;s.referee.calls=0;
}
// Advance an existing fall by one tick. Returns true on BECAME_FALLEN.
REK_LITE_FN bool advance_fall(FighterState& f){
    if(f.phase==Upright){if(f.grace_ticks>0)f.grace_ticks--;return false;}
    f.ticks++;
    if(f.phase==Falling&&f.ticks>=f.resolve_ticks){
        if(f.outcome==Recover){f.phase=Upright;f.ticks=0;f.classification=NoFall;f.events|=FallingCleared;return false;}
        f.phase=Fallen;f.ticks=0;f.events|=BecameFallen;return true;
    }
    return false;
}
// Sample a new onset for an upright fighter outside the spawn grace.
REK_LITE_FN bool try_onset(const Model& m,FighterState& f,const Features& x,int own_move,
        bool struck_valid,float struck_age_seconds,float struck_speed,
        float u_onset,float u_outcome,float u_delay){
    if(!m.enabled||f.phase!=Upright||f.grace_ticks>0)return false;
    if(u_onset>=hazard(m,x))return false;
    f.phase=Falling;f.ticks=0;f.events|=FallingStarted;
    f.outcome=sample_outcome(m,move_class(own_move),u_outcome);
    f.resolve_ticks=sample_ticks(m.delay_quantiles+f.outcome*kQuantiles,u_delay);
    f.classification=knockdown(struck_valid,struck_age_seconds,struck_speed)?Knockdown:Slip;
    return true;
}
// OnFighterFallen with CanGetUp false for both fighters.
REK_LITE_FN void on_fallen(ArenaState& s,int side){
    Referee& r=s.referee;
    const bool other=r.count_active[side^1]!=0;
    r.count_active[side]=1;r.count_is_slip[side]=s.fighter[side].classification!=Knockdown;
    r.count_ticks=0;  // first count starts, or a double down restarts the shared count
    r.calls|=other?CallDoubleKnockdown:(r.count_is_slip[side]?CallSlip:CallKnockdown);
}
// RefereeCountRoutine/ResolveCountExpiry. Writes awarded points and returns
// true when the count expired (caller resets both or ends the round).
REK_LITE_FN bool advance_referee(ArenaState& s,int award[2]){
    Referee& r=s.referee;award[0]=award[1]=0;
    if(!counting(r))return false;
    if(++r.count_ticks<kCountTicks)return false;
    const bool both=r.count_active[0]&&r.count_active[1];
    if(both){award[0]=award[1]=kKnockoutPoints;r.calls|=CallDoubleKnockout;}
    else{award[r.count_active[0]?1:0]=kKnockoutPoints;r.calls|=CallKnockout;}
    r.count_active[0]=r.count_active[1]=0;r.count_is_slip[0]=r.count_is_slip[1]=0;r.count_ticks=0;
    return true;
}

// Detector-field proxies in the physical runtime's 15-float fall block layout.
REK_LITE_FN float tilt_degrees(const Model& m,const FighterState& f,float upright_tilt){
    return f.phase==Fallen?m.fallen_tilt_degrees:f.phase==Falling?m.falling_tilt_degrees:upright_tilt;
}
REK_LITE_FN float height_ratio(const Model& m,const FighterState& f,float upright_ratio){
    return f.phase==Fallen?m.fallen_height_ratio:f.phase==Falling?m.falling_height_ratio:upright_ratio;
}

}  // namespace rek_lite_falls
