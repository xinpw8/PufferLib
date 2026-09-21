#pragma once
#include "../g1_hit_detector.h"
#include <math.h>
#include <string.h>

// The recovered apex implementation is compiled verbatim for host and device.
// Namespace isolation avoids changing or exporting the original native symbols.
#undef REK_G1_FN
#if defined(__CUDACC__)
#define REK_G1_FN __host__ __device__ inline
#else
#define REK_G1_FN inline
#endif
namespace rek5_recovered {
#define rek_g1_strike_intent_apex embedded_strike_intent_apex
#define rek_g1_impact_event_ramp_at embedded_impact_event_ramp_at
#include "../g1_hit_detector.c"
#undef rek_g1_strike_intent_apex
#undef rek_g1_impact_event_ramp_at

enum Reject { Speed=1,Apex=2,Cooldown=4,Duplicate=8 };
struct Result { int points;int reject;int apex; };

// Partial recovered acceptance on a compact contact-enter. The caller supplies
// an explicit relative-speed proxy: legacy sphere-center finite difference or
// opt-in body-cvel kinematics. Upright/zone predicates
// are the compact model's existing upright / pelvis-torso-head assumptions.
// No native contact manifold, contact-point velocity or impulse is implied.
REK_G1_FN Result score(RekG1HitDetectorState& state,
        const RekG1HitDetectorConfig& config,const RekG1StrikeIntent& intent,
        int fighter,int limb,float relative_speed,float time_seconds){
    const auto part=limb<2?REK_G1_BODY_PART_FOOT:limb<4?REK_G1_BODY_PART_HAND:REK_G1_BODY_PART_SHIN;
    const auto side=(limb&1)?REK_G1_HAND_RIGHT:REK_G1_HAND_LEFT;
    if(relative_speed<config.speed_threshold_mps)return {0,Speed,-1};
    int32_t apex=-1;float ramp=0;
    if(!embedded_strike_intent_apex(&intent,part,side,config.apex_min_ramp,&apex,&ramp))return {0,Apex,-1};
    if(state.cooldown_seen[fighter][limb]&&time_seconds-state.last_score_time_seconds[fighter][limb]<config.per_body_cooldown_seconds)return {0,Cooldown,apex};
    const uint32_t bit=uint32_t(1)<<(apex>int32_t(REK_G1_HIT_MAX_APEX_BIT)?REK_G1_HIT_MAX_APEX_BIT:uint32_t(apex));
    if(state.scored_move_seen[fighter]&&state.scored_move_id[fighter]==intent.move_id&&(state.scored_apex_mask[fighter]&bit))return {0,Duplicate,apex};
    state.last_score_time_seconds[fighter][limb]=time_seconds;state.cooldown_seen[fighter][limb]=1;
    if(!state.scored_move_seen[fighter]||state.scored_move_id[fighter]!=intent.move_id){state.scored_move_id[fighter]=intent.move_id;state.scored_apex_mask[fighter]=0;state.scored_move_seen[fighter]=1;}
    state.scored_apex_mask[fighter]|=bit;
    return {part==REK_G1_BODY_PART_HAND?1:2,0,apex};
}
}
#undef REK_G1_FN
#define REK_G1_FN
