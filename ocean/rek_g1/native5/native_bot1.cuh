#pragma once
#include <math.h>
#include <stdint.h>

#if defined(__CUDACC__)
#define REK_BOT_FN __host__ __device__ inline
#else
#define REK_BOT_FN inline
#endif

// Recovered client-image high-level AI only. See NATIVE_BOT1_GAP.md for source
// hashes and method RVAs. Cadence, RNG and compact actuator response are explicit
// candidate choices; none establishes authoritative server behavior.
namespace rek5_bot1 {
enum Phase { Inactive=0, Engaging=1, Repositioning=2, Settling=3,
    Attacking=4, Recovering=5, GivingRoom=6 };
constexpr float stop=.4180000126361847f, facing=35.f;
constexpr float initial_delay=.6600000262260437f, max_engage=2.1500000953674316f;
constexpr float min_footwork=.20000000298023224f, settle_time=.30000001192092896f;
constexpr float recovery_time=.257999986410141f, max_punch=3.f;
constexpr float forward_speed=.800000011920929f, yaw_speed=1.5f;
constexpr float reposition_chance=.05000000074505806f, kick_chance=.25f;
constexpr float reposition_min=.35199999809265137f, reposition_max=1.4129999876022339f;
constexpr float reposition_back=.15000000596046448f, reposition_strafe=.4339999854564667f;

struct Command { float forward,strafe,yaw; };
struct State {
    int phase;
    float timer,min_timer;
    Command reposition;
    uint32_t rng;
    unsigned attempts,accepted,rejected,repositions;
};
struct Input {
    float distance,angle_degrees,delta_seconds,time_seconds,round_elapsed;
    bool punching,opponent_down,own_recovery,round_active;
};
struct Decision { int move;bool clear_punching,unsupported_recovery; };
struct Catalog { int primary_limb[17]; };

REK_BOT_FN float clamp(float x,float lo,float hi){return fminf(hi,fmaxf(lo,x));}
// Candidate-private xorshift32. Native Unity RNG state/algorithm is unknown.
struct Random {
    uint32_t& state;
    REK_BOT_FN uint32_t next(){uint32_t x=state;x^=x<<13;x^=x>>17;x^=x<<5;return state=x;}
    REK_BOT_FN float value(){return float(next()>>8)*(1.f/16777216.f);}
    REK_BOT_FN int integer(int bound){
        const uint32_t b=uint32_t(bound),threshold=uint32_t(-b)%b;
        uint32_t x;do{x=next();}while(x<threshold);return int(x%b);
    }
};
REK_BOT_FN void activate(State& s,uint32_t seed){
    s={};s.phase=Settling;s.timer=initial_delay;s.rng=seed?seed:0x6d2b79f5u;
}
REK_BOT_FN void engage(State& s,const Input& i){
    s.phase=Engaging;s.timer=max_engage;
    s.min_timer=fabsf(i.angle_degrees)<facing&&i.distance<=stop+.15f?0.f:min_footwork;
}
// RobotConfig.TryPickMove consumes both full-category and preferred-side
// reservoir draws while scanning assigned order, then favors the side pool.
template<class R> REK_BOT_FN int pick_category(const Catalog& c,int category,int side,R& rng){
    int all=-1,preferred=-1,nall=0,npreferred=0;
    for(int move=0;move<17;move++){
        const int limb=c.primary_limb[move];if(!limb)continue;
        const int cat=limb==3||limb==4?1:0;
        if(cat!=category)continue;
        if(rng.integer(++nall)==0)all=move;
        const int limb_side=limb==1||limb==3?0:1;
        if(limb_side==side&&rng.integer(++npreferred)==0)preferred=move;
    }
    return npreferred?preferred:all;
}
template<class R> REK_BOT_FN int pick_attack(const Catalog& c,float angle,R& rng){
    const int category=rng.value()<kick_chance?1:0,side=angle>=0?1:0;
    int move=pick_category(c,category,side,rng);
    return move<0?pick_category(c,category^1,side,rng):move;
}
template<class R> REK_BOT_FN void reposition(State& s,R& rng){
    const float sign=rng.value()>.5f?1.f:-1.f;
    s.reposition={-reposition_back,-sign*reposition_strafe,-(-.3f+.6f*rng.value())*yaw_speed};
    s.phase=Repositioning;s.timer=reposition_min+(reposition_max-reposition_min)*rng.value();s.repositions++;
}
// Attack execution is supplied by the compact adapter. This preserves the
// recovered success/failure transition without asserting native acceptance.
REK_BOT_FN void attack_result(State& s,const Input& i,bool accepted){
    if(accepted){s.phase=Attacking;s.timer=max_punch;s.accepted++;}
    else{s.rejected++;engage(s,i);}
}
template<class R> REK_BOT_FN Decision update(State& s,const Input& i,const Catalog& c,R& rng){
    Decision out{-1,false,false};s.timer-=i.delta_seconds;
    if(i.own_recovery){out.unsupported_recovery=true;return out;}
    if(i.opponent_down&&s.phase!=GivingRoom&&s.phase!=Attacking){s.phase=GivingRoom;s.timer=1.f;}
    switch(s.phase){
        case Engaging:
            s.min_timer-=i.delta_seconds;
            if(s.min_timer<=0&&fabsf(i.angle_degrees)<facing&&i.distance<=stop+.3f){s.phase=Settling;s.timer=settle_time;}
            else if(s.timer<=0)engage(s,i);
            break;
        case Repositioning:if(s.timer<=0)engage(s,i);break;
        case Settling:
            if(s.timer>0){if(i.distance>stop+.5f&&fabsf(i.angle_degrees)>=facing)engage(s,i);}
            else if(i.distance<stop+.3f||fabsf(i.angle_degrees)<facing){
                if(!i.round_active||i.round_elapsed<initial_delay){engage(s,i);break;}
                out.move=pick_attack(c,i.angle_degrees,rng);s.attempts++;
                if(out.move<0)attack_result(s,i,false);
            }else engage(s,i);
            break;
        case Attacking:
            if(!i.punching||s.timer<=0){out.clear_punching=true;s.phase=Recovering;s.timer=recovery_time;}
            break;
        case Recovering:
            if(s.timer<=0){if(rng.value()<reposition_chance)reposition(s,rng);else engage(s,i);}
            break;
        case GivingRoom:if(i.opponent_down)s.timer=1.f;else if(s.timer<=0)engage(s,i);break;
        default:break;
    }
    return out;
}
REK_BOT_FN float facing_yaw(float angle){
    if(fabsf(angle)<=facing*.5f)return 0;
    return -(angle>0?1.f:-1.f)*clamp(fabsf(angle)/45.f,0,1)*yaw_speed;
}
REK_BOT_FN Command locomotion(const State& s,const Input& i){
    if(i.own_recovery)return {};
    if(s.phase==Repositioning)return s.reposition;
    if(s.phase==Settling)return {0,0,facing_yaw(i.angle_degrees)};
    if(s.phase==GivingRoom)return {i.distance<1.5f?-clamp((1.5f-i.distance)*2.f,0,1)*.25f:0.f,0,facing_yaw(i.angle_degrees)};
    // Native jump table sends Attacking and Recovering to zero velocity.
    if(s.phase!=Engaging)return {};
    Command c{i.distance>stop?clamp((i.distance-stop)/.8f,0,1)*forward_speed:0.f,0,facing_yaw(i.angle_degrees)};
    if(fabsf(i.angle_degrees)<facing&&i.distance<=stop+.3f){
        c.strafe=-sinf(2.f*i.time_seconds)*.15f;
        if(i.distance>stop*.8f)c.forward=forward_speed*.3f;
    }
    return c;
}
}
#undef REK_BOT_FN
