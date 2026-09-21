#include "runtime_api.h"
#include "fast_assets.h"
#include "device_storage.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include "recovered_contact_rules.cuh"
#include "native_bot1.cuh"
#include "rendered_pose_observation.h"
#include "primitive_motion.cuh"
#include "contact_potential_loader.h"
#include "round_reward.h"
#include "normalized_reward.h"
#include "policy_feature_mask.h"
#include "owned_yaw_observation.h"
#include "action_cadence.h"
#include "keyboard_yaw.h"
#include "contact_entry.h"
#include "fast_observable_balance.h"

// Explicit reduced-order candidate. The source clips provide pose and strike
// trajectories; slider motion and temporally sampled contacts are modeling
// choices, not recovered REK dynamics. No full-physics backend is called here.
namespace {
constexpr float DT=.02f, PI=3.14159265358979323846f;
constexpr int WARPS_PER_BLOCK=4, THREADS=32*WARPS_PER_BLOCK;
thread_local std::string error_text;

struct Fighter {
    float x,y,yaw,vx,vy,omega;
    float old_x,old_y,old_yaw;
    float phase,old_phase;
    float last_hit_age,last_hit_speed;
    int route,old_route,held,move_tick,attack_duration,strike_active,last_hit_valid;
    int cooldown[6];
    unsigned contact_latched;
    int points,falls,down;
    int move_instance;
    int bot_controlled;
    rek5_bot1::Command bot_command;
    int observed_old_frame;
    float observed_old_x,observed_old_y,observed_old_yaw;
    float observed_heading,observed_local[2],observed_omega,observed_joint_rate[29];
    rek_keyboard_yaw::State keyboard_yaw;
    rek_contact_entry::State contact_pairs;
};
struct Arena {
    Fighter fighter[2];
    float elapsed,episode_return,episode_hits,invalid;
    float reward[2];
    int tick,reset_wait,dummy_move[2],opponent_mode;
    int delta[2],hit_count,down_event[2];
    unsigned failures;
    RekG1HitDetectorState recovered_hits;
    rek5_bot1::State bot[2];
    int contact_pairs_initialized;
};
struct Parameters {
    FastRoute routes[24];
    int action_to_route[33],qindices[2][29],vindices[2][29];
    unsigned durations[17];
    float initial_qpos[72],spawn[2][2],heading[2],half_extent[2];
    float floor,round_seconds,move_speed,yaw_speed,brake_rate,yaw_ramp,settle_speed;
    float body_radius,hit_speed;
    uint32_t seed;
    int opponent_mode,random_resets;
    float reset_gap_min,reset_gap_max,reset_heading_spread;
    float shaping_weight,shaping_gamma,shaping_target,shaping_bearing_weight;
    rek5_round_reward::Mode reward_mode;
    bool normalized_rewards;
    float reward_gamma;
    rek5_contact_potential::Model contact_potential;
    int recovered_scoring;
    RekG1ImpactEvent impact_events[29];
    int impact_offsets[24],impact_counts[24];
    RekG1HitDetectorConfig recovered_hit_config;
    int recovered_bot;
    rek5_bot1::Catalog bot_catalog;
    int move_to_action[17];
    int rendered_observation;
    int owned_yaw_observation;
    int policy_action_stride=1;
    rek_keyboard_yaw::Mode yaw_command=rek_keyboard_yaw::Mode::LegacyVelocitySlew;
    rek_contact_entry::Mode contact_entry=rek_contact_entry::Mode::LegacyLimbUnion;
    rek_contact_velocity::Mode contact_velocity=rek_contact_velocity::Mode::LegacySphereProxy;
    int primitive_contacts,contact_substeps,strike_limb[12];
};
struct View {
    int arenas;
    const Parameters* p;
    const FastFrame* frames;
    const FastBodyVelocityFrame* body_velocity_frames;
    Arena* state;
    RekNative5RoundResult* rounds;
    RekNative5Buffers out;
    float *raw,*qpos,*qvel,*actions,*rewards,*terminals;
    uint8_t *masks,*learner_masks;
    const float* external;
    const uint8_t* override_rows;
    const uint8_t* policy_feature_mask;
    unsigned* reward_saturations;
    rek_fast_observable::History* observable_history;
    float* observable_observations;
};

__device__ float clampf(float x,float lo,float hi){return fminf(hi,fmaxf(lo,x));}
__device__ float approach(float x,float target,float delta){return x+clampf(target-x,-delta,delta);}
__device__ float angle(float x){return atan2f(sinf(x),cosf(x));}
__device__ bool attacking(const Fighter& f){return f.attack_duration>0;}
__device__ bool translating(const Fighter& f){
    return f.held==2||f.held==3||f.held==4||f.held==5||(f.held>=8&&f.held<=15);
}
__device__ bool settled(const Parameters& p,const Fighter& f){
    return !translating(f)&&f.vx*f.vx+f.vy*f.vy<=p.settle_speed*p.settle_speed;
}
__device__ int frame_index(const Parameters& p,const Fighter& f,bool previous){
    int route=previous?f.old_route:f.route;
    const FastRoute& r=p.routes[route];
    float phase=previous?f.old_phase:f.phase;
    int index=int(phase);
    if(r.loop)index=index%r.count;
    else index=min(max(index,0),r.count-1);
    return r.offset+index;
}
__device__ void root_quaternion(const Fighter& f,const FastFrame& frame,float* q){
    float s,c;sincosf(.5f*f.yaw,&s,&c);
    q[0]=c*frame.root_wxyz[0]-s*frame.root_wxyz[3];
    q[1]=c*frame.root_wxyz[1]-s*frame.root_wxyz[2];
    q[2]=c*frame.root_wxyz[2]+s*frame.root_wxyz[1];
    q[3]=c*frame.root_wxyz[3]+s*frame.root_wxyz[0];
}
__device__ void point(const Fighter& f,const float* local,bool previous,float* xyz){
    float s,c;sincosf(previous?f.old_yaw:f.yaw,&s,&c);
    xyz[0]=(previous?f.old_x:f.x)+c*local[0]-s*local[1];
    xyz[1]=(previous?f.old_y:f.y)+s*local[0]+c*local[1];
    xyz[2]=local[2];
}
__device__ void capture_observed_pose(const Parameters& p,Arena& a){
    for(int side=0;side<2;side++){
        Fighter& f=a.fighter[side];f.observed_old_frame=frame_index(p,f,false);
        f.observed_old_x=f.x;f.observed_old_y=f.y;f.observed_old_yaw=f.yaw;
    }
}
__device__ void update_observed_pose(const View& v,Arena& a){
    for(int side=0;side<2;side++){
        Fighter& f=a.fighter[side];const auto& frame=v.frames[frame_index(*v.p,f,false)];
        const auto& old=v.frames[f.observed_old_frame];float q[4],previous[4];
        root_quaternion(f,frame,q);rek_rendered_pose::compose_root(f.observed_old_yaw,old.root_wxyz,previous);
        f.observed_heading=rek_rendered_pose::heading(q);
        f.observed_omega=rek_rendered_pose::angular_rate(f.observed_heading,rek_rendered_pose::heading(previous),DT);
        rek_rendered_pose::local_velocity(f.x,f.y,f.observed_old_x,f.observed_old_y,f.observed_heading,DT,f.observed_local);
        for(int j=0;j<29;j++)f.observed_joint_rate[j]=rek_rendered_pose::joint_rate(frame.q[j],old.q[j],DT);
    }
}
__device__ void pose_reset(const Parameters& p,Arena& a){
    a.contact_pairs_initialized=0;
    for(int side=0;side<2;side++){
        Fighter& f=a.fighter[side];int points=f.points,falls=f.falls;
        f={};f.points=points;f.falls=falls;
        f.x=f.old_x=p.spawn[side][0];f.y=f.old_y=p.spawn[side][1];
        f.yaw=f.old_yaw=p.heading[side];f.held=1;
    }
}
__device__ uint32_t reset_hash(uint32_t value){
    value^=value>>16;value*=0x7feb352du;value^=value>>15;value*=0x846ca68bu;return value^(value>>16);
}
__device__ float reset_uniform(uint32_t key,uint32_t field){
    return float(reset_hash(key^field)>>8)*(1.f/16777216.f);
}
__device__ void round_reset(const Parameters& p,Arena& a,RekNative5RoundResult& result,bool full,int index){
    if(full){result={};result.round_number=1;}
    else result.round_number++;
    a={};pose_reset(p,a);
    if(p.recovered_bot)for(int side=0;side<2;side++)rek5_bot1::activate(a.bot[side],
        reset_hash(p.seed^reset_hash(uint32_t(index)*2+side+0x7216b539u)^reset_hash(result.round_number)));
    a.opponent_mode=p.opponent_mode;
    if(p.random_resets||p.opponent_mode==4){
        const uint32_t key=reset_hash(p.seed^reset_hash(uint32_t(index)+0x9e3779b9u)^reset_hash(result.round_number+0x85ebca6bu));
        if(p.opponent_mode==4)a.opponent_mode=int(reset_hash(key^0xa511e9b3u)&3u);
        if(p.random_resets){
            const float gap=p.reset_gap_min+(p.reset_gap_max-p.reset_gap_min)*reset_uniform(key,1);
            const float axis=2*PI*reset_uniform(key,2);float sn,cs;sincosf(axis,&sn,&cs);
            const float dx=.5f*gap*cs,dy=.5f*gap*sn;
            // Host validation guarantees a feasible midpoint for every axis.
            // Do not clamp sampled roots, which would silently change the gap.
            const float ex=p.half_extent[0]-p.body_radius-.01f-fabsf(dx);
            const float ey=p.half_extent[1]-p.body_radius-.01f-fabsf(dy);
            const float cx=(2*reset_uniform(key,3)-1)*ex,cy=(2*reset_uniform(key,4)-1)*ey;
            for(int side=0;side<2;side++){
                Fighter& f=a.fighter[side];const float sign=side?1.f:-1.f;
                f.x=f.old_x=cx+sign*dx;f.y=f.old_y=cy+sign*dy;
                f.yaw=f.old_yaw=angle(axis+(side?PI:0)+(2*reset_uniform(key,5+side)-1)*p.reset_heading_spread);
            }
        }
    }
    result.phase=2;result.round_result=0;result.round_winner=-1;
    result.fight_result=0;result.fight_winner=-1;
    result.time_remaining_seconds=p.round_seconds;result.terminal=0;
    result.failure_bits=0;
    for(int s=0;s<2;s++){result.points[s]=0;result.falls[s]=0;}
}
__device__ int action_value(Arena& a,float value){
    if(!isfinite(value)||value<0||value>=33||value!=floorf(value)){
        a.failures|=16;return 0;
    }
    return int(value);
}
__device__ int scripted_action(const Parameters& p,Arena& a,int side){
    const Fighter& f=a.fighter[side];const Fighter& other=a.fighter[side^1];
    if(attacking(f)||a.reset_wait)return 0;
    float dx=other.x-f.x,dy=other.y-f.y;
    float bearing=angle(atan2f(dy,dx)-f.yaw),distance=hypotf(dx,dy);
    if(fabsf(bearing)>.16f)return bearing>0?6:7;
    if(distance>1.05f)return 2;
    if(distance<.55f)return 3;
    if(!settled(p,f))return 1;
    return 16+(a.dummy_move[side]++%16);
}
__device__ int opponent_action(const Parameters& p,Arena& a,int side){
    if(a.opponent_mode==0)return scripted_action(p,a,side);
    if(a.opponent_mode==1)return 1;
    const Fighter& f=a.fighter[side];const Fighter& other=a.fighter[side^1];
    if(attacking(f))return 0;
    const float dx=other.x-f.x,dy=other.y-f.y;
    const float bearing=angle(atan2f(dy,dx)-f.yaw);
    if(fabsf(bearing)>.16f)return bearing>0?6:7;
    if(a.opponent_mode==2)return hypotf(dx,dy)<1.25f?3:1;
    // The strafe target does not attack. Its direction changes every second.
    return (a.tick/50)&1?4:5;
}
__device__ float shaping_potential(const View& v,const Arena& a,int side){
    const Parameters& p=*v.p;
    const Fighter& f=a.fighter[side];const Fighter& other=a.fighter[side^1];
    const float dx=other.x-f.x,dy=other.y-f.y;
    if(p.contact_potential.count){
        float rendered[4];root_quaternion(f,v.frames[frame_index(p,f,false)],rendered);
        const float heading=rek_rendered_pose::heading(rendered);
        // Evidence yaw is atan2(Unity X,Unity Z). The candidate uses
        // atan2(Unity Z,Unity X), so signed relative bearing changes sign.
        const float evidence_bearing=-angle(atan2f(dy,dx)-heading);
        return rek5_contact_potential::potential(p.contact_potential,hypotf(dx,dy),evidence_bearing);
    }
    const float error=fabsf(hypotf(dx,dy)-p.shaping_target);
    const float bearing=fabsf(angle(atan2f(dy,dx)-f.yaw))/PI;
    return -(error/(1+error)+p.shaping_bearing_weight*bearing)/(1+p.shaping_bearing_weight);
}
__device__ void held_command(int category,float& forward,float& strafe,float& yaw){
    forward=strafe=yaw=0;
    switch(category){
        case 2:forward=1;break;case 3:forward=-1;break;
        case 4:strafe=1;break;case 5:strafe=-1;break;
        case 6:yaw=1;break;case 7:yaw=-1;break;
        case 8:forward=1;yaw=1;break;case 9:forward=1;yaw=-1;break;
        case 10:forward=-1;yaw=1;break;case 11:forward=-1;yaw=-1;break;
        case 12:strafe=1;yaw=1;break;case 13:strafe=1;yaw=-1;break;
        case 14:strafe=-1;yaw=1;break;case 15:strafe=-1;yaw=-1;break;
    }
}
template<bool Bot=false> __device__ void advance_fighter(const Parameters& p,Fighter& f,int action,Arena& a,const rek5_bot1::Command* command=nullptr){
    f.bot_controlled=Bot;
    if constexpr(Bot)f.bot_command=*command;
    f.old_x=f.x;f.old_y=f.y;f.old_yaw=f.yaw;f.old_phase=f.phase;f.old_route=f.route;
    f.strike_active=0;
    if(f.last_hit_valid)f.last_hit_age=fminf(120.f,f.last_hit_age+DT);
    for(int k=0;k<6;k++)f.cooldown[k]=max(0,f.cooldown[k]-1);
    if(attacking(f)){
        // Desired yaw/release can change while animation owns actual motion.
        // Attacks and translation commands are discarded; no move queue exists.
        if(action==1||action==6||action==7)f.held=action;
        else if(action)a.invalid+=1;
    }else if(action>=16){
        if(settled(p,f)){
            f.route=p.action_to_route[action];f.phase=0;f.move_tick=0;
            f.move_instance++;
            f.attack_duration=int(p.durations[p.routes[f.route].move]);
            if(f.held!=6&&f.held!=7)f.held=1;
            f.vx=f.vy=f.omega=0;f.contact_latched=0;
            // A new move never sweeps from the previous move's unrelated pose.
            f.old_route=f.route;f.old_phase=0;
        }else{
            // Translation must settle first. This attempted attack is not buffered.
            f.held=1;a.invalid+=1;
        }
    }else if(action>0){f.held=action;}
    if(attacking(f)){
        if constexpr(!Bot)if(p.yaw_command==rek_keyboard_yaw::Mode::KeyboardReset)
            rek_keyboard_yaw::advance(f.keyboard_yaw,0.f,DT,rek_keyboard_yaw::kRampSeconds,1.f);
        f.strike_active=1;
        f.phase=fminf(float(p.routes[f.route].count-1),
            float(f.move_tick+1)*float(p.routes[f.route].count-1)/float(f.attack_duration));
        f.move_tick++;
        if(f.move_tick>=f.attack_duration)f.attack_duration=0;
        return;
    }
    float forward,strafe,yaw;held_command(f.held,forward,strafe,yaw);
    if constexpr(Bot){forward=command->forward;strafe=command->strafe;yaw=command->yaw;}
    float sn,cs;sincosf(f.yaw,&sn,&cs);
    float tx=p.move_speed*(cs*forward-sn*strafe),ty=p.move_speed*(sn*forward+cs*strafe);
    float accel=p.brake_rate*DT;
    f.vx=approach(f.vx,tx,accel);f.vy=approach(f.vy,ty,accel);
    if(!Bot&&p.yaw_command==rek_keyboard_yaw::Mode::KeyboardReset){
        // Candidate physical response: normalized keyboard command times the
        // existing yaw-speed constant. No second, unmeasured actuator lag.
        const float normalized=rek_keyboard_yaw::advance(f.keyboard_yaw,yaw,DT,rek_keyboard_yaw::kRampSeconds,1.f);
        f.omega=normalized*p.yaw_speed;
    }else f.omega=approach(f.omega,yaw*p.yaw_speed,p.yaw_speed*DT/p.yaw_ramp);
    f.x+=f.vx*DT;f.y+=f.vy*DT;f.yaw=angle(f.yaw+f.omega*DT);
    int route=p.action_to_route[f.held];
    if(route!=f.route){f.route=route;f.phase=0;f.old_route=route;f.old_phase=0;}
    else f.phase+=1;
    const FastRoute& r=p.routes[f.route];
    if(r.loop&&f.phase>=r.count)f.phase-=r.count;
    else f.phase=fminf(f.phase,float(r.count-1));
}
__device__ int bot_pose_action(const rek5_bot1::Command& c){
    // Existing canned routes cannot represent continuous blends. Select the
    // dominant translation route, without quantizing root command magnitude.
    if(fabsf(c.forward)>1e-6f||fabsf(c.strafe)>1e-6f)
        return fabsf(c.forward)>=fabsf(c.strafe)?(c.forward>0?2:3):(c.strafe>0?4:5);
    return fabsf(c.yaw)>1e-6f?(c.yaw>0?6:7):1;
}
__device__ void advance_bot(const Parameters& p,Arena& a,int side,const rek5_bot1::Input& input){
    Fighter& f=a.fighter[side];auto& bot=a.bot[side];
    rek5_bot1::Random rng{bot.rng};auto decision=rek5_bot1::update(bot,input,p.bot_catalog,rng);
    if(decision.unsupported_recovery){a.failures|=1024;return;}
    if(decision.clear_punching)f.attack_duration=0;
    if(decision.move>=0){
        // Retain compact settling/animation acceptance, report failures to FSM.
        const bool accepted=!attacking(f)&&settled(p,f);
        rek5_bot1::attack_result(bot,input,accepted);
        auto cmd=rek5_bot1::locomotion(bot,input);
        if(accepted){advance_fighter<true>(p,f,p.move_to_action[decision.move],a,&cmd);return;}
    }
    auto cmd=rek5_bot1::locomotion(bot,input);
    advance_fighter<true>(p,f,bot_pose_action(cmd),a,&cmd);
}
__device__ void confine(const Parameters& p,Fighter& f){
    float x=clampf(f.x,-p.half_extent[0]+p.body_radius,p.half_extent[0]-p.body_radius);
    float y=clampf(f.y,-p.half_extent[1]+p.body_radius,p.half_extent[1]-p.body_radius);
    if(x!=f.x)f.vx=0;if(y!=f.y)f.vy=0;f.x=x;f.y=y;
}
__device__ bool limb_enabled(int route,int limb){
    if(route==7||route==8)return limb==0;
    if(route==9)return limb==1;
    if(route==10)return limb==5;
    if(route==11||route==12)return limb==2;
    if(route==14||route==15)return limb==3;
    return route>=13&&route<=22&&limb>=2&&limb<=3;
}
__device__ float sweep_distance2(const float* from,const float* to){
    float x=to[0]-from[0],y=to[1]-from[1],z=to[2]-from[2];
    float denom=x*x+y*y+z*z;
    float t=denom>1e-12f?clampf(-(from[0]*x+from[1]*y+from[2]*z)/denom,0,1):0;
    x=from[0]+t*x;y=from[1]+t*y;z=from[2]+t*z;return x*x+y*y+z*z;
}
__device__ bool primitive_touch(const Parameters& p,const Fighter& f,const Fighter& enemy,
        const FastFrame& before,const FastFrame& now,const FastFrame& old_target,
        const FastFrame& target,int limb,int zone){
    using namespace rek5_primitive;
    const Shape old_dst=world_shape(old_target.target_shapes[zone],enemy.old_x,enemy.old_y,enemy.old_yaw);
    const Shape dst=world_shape(target.target_shapes[zone],enemy.x,enemy.y,enemy.yaw);
    for(int i=0;i<12;i++)if(p.strike_limb[i]==limb){
        const Shape old_tip=world_shape(before.strike_shapes[i],f.old_x,f.old_y,f.old_yaw);
        const Shape tip=world_shape(now.strike_shapes[i],f.x,f.y,f.yaw);
        if(sampled_overlap(old_tip,tip,old_dst,dst,p.contact_substeps))return true;
    }
    return false;
}
__device__ void seed_contact_pairs(const View& v,Arena& a){
    using namespace rek5_primitive;
    for(int side=0;side<2;side++){
        auto& f=a.fighter[side];const auto& enemy=a.fighter[side^1];
        const auto& frame=v.frames[frame_index(*v.p,f,false)];
        const auto& target=v.frames[frame_index(*v.p,enemy,false)];
        for(int i=0;i<rek_contact_entry::Strikers;i++)for(int j=0;j<rek_contact_entry::Targets;j++)
            rek_contact_entry::update(f.contact_pairs,i*rek_contact_entry::Targets+j,
                overlap(world_shape(frame.strike_shapes[i],f.x,f.y,f.yaw),
                        world_shape(target.target_shapes[j],enemy.x,enemy.y,enemy.yaw)));
    }
    a.contact_pairs_initialized=1;
}
__device__ rek_contact_velocity::Linear body_linear_velocity(const View& v,
        const Fighter& f,int side,int body_slot){
    const int current=frame_index(*v.p,f,false),previous=frame_index(*v.p,f,true);
    return rek_contact_velocity::compose(v.body_velocity_frames[current],side,body_slot,
        f.yaw,f.vx,f.vy,f.omega,current!=previous);
}
__device__ void geom_pair_contacts(const View& v,Arena& a,int side,int* hits,int* points){
    using namespace rek5_primitive;
    const Parameters& p=*v.p;auto& f=a.fighter[side];const auto& enemy=a.fighter[side^1];
    const auto& now=v.frames[frame_index(p,f,false)];const auto& before=v.frames[frame_index(p,f,true)];
    const auto& target=v.frames[frame_index(p,enemy,false)];const auto& old_target=v.frames[frame_index(p,enemy,true)];
    for(int limb=0;limb<6;limb++){
        RekG1StrikeIntent intent{};bool can_score=f.strike_active;
        if(can_score){
            const auto& route=p.routes[f.route];
            intent.impact_events=p.impact_events+p.impact_offsets[f.route];intent.impact_event_count=p.impact_counts[f.route];
            intent.clip_cursor_frames=clampf(route.start_frame+f.phase*route.playback_speed,float(route.start_frame),float(route.end_frame));
            intent.clip_fps=50;intent.move_id=f.move_instance;intent.action_playing=intent.layer_active=1;
            const auto part=limb<2?REK_G1_BODY_PART_FOOT:limb<4?REK_G1_BODY_PART_HAND:REK_G1_BODY_PART_SHIN;
            const auto hand=(limb&1)?REK_G1_HAND_RIGHT:REK_G1_HAND_LEFT;
            int32_t apex=-1;float ramp=0;
            can_score=rek5_recovered::embedded_strike_intent_apex(&intent,part,hand,p.recovered_hit_config.apex_min_ramp,&apex,&ramp);
        }
        float tip[3],old_tip[3];point(f,now.strike_xyz[limb],false,tip);point(f,before.strike_xyz[limb],true,old_tip);
        bool entered=false;float max_relative_speed=0;
        for(int zone=0;zone<rek_contact_entry::Targets;zone++){
            float dst[3],old_dst[3],from[3],to[3];
            point(enemy,target.target_shapes[zone].center,false,dst);point(enemy,old_target.target_shapes[zone].center,true,old_dst);
            for(int k=0;k<3;k++){from[k]=old_tip[k]-old_dst[k];to[k]=tip[k]-dst[k];}
            const float radius=fmaxf(before.strike_radius[limb],now.strike_radius[limb])+
                fmaxf(old_target.target_shape_radius[zone],target.target_shape_radius[zone]);
            const bool broad=sweep_distance2(from,to)<=radius*radius;
            bool intersects=false;
            for(int i=0;i<rek_contact_entry::Strikers;i++)if(p.strike_limb[i]==limb){
                const int pair=i*rek_contact_entry::Targets+zone;
                if(!broad){rek_contact_entry::update(f.contact_pairs,pair,false);continue;}
                if(!can_score){
                    rek_contact_entry::unscored_endpoint(f.contact_pairs,pair,
                        world_shape(now.strike_shapes[i],f.x,f.y,f.yaw),
                        world_shape(target.target_shapes[zone],enemy.x,enemy.y,enemy.yaw));
                    continue;
                }
                const auto result=rek_contact_entry::sample(f.contact_pairs,pair,
                    world_shape(before.strike_shapes[i],f.old_x,f.old_y,f.old_yaw),
                    world_shape(now.strike_shapes[i],f.x,f.y,f.yaw),
                    world_shape(old_target.target_shapes[zone],enemy.old_x,enemy.old_y,enemy.old_yaw),
                    world_shape(target.target_shapes[zone],enemy.x,enemy.y,enemy.yaw),p.contact_substeps);
                if(p.contact_velocity==rek_contact_velocity::Mode::BodyCvel&&result.entered){
                    // Each entered geom pair supplies its own body velocities.
                    // End-of-tick rates approximate every sampled entry in this
                    // tick; persistent/unrelated targets never lend speed.
                    const float speed=rek_contact_velocity::relative_speed(
                        body_linear_velocity(v,f,side,rek_contact_velocity::limb_slot(limb)),
                        body_linear_velocity(v,enemy,side^1,rek_contact_velocity::target_slot(zone)));
                    const auto score=rek5_recovered::score(a.recovered_hits,p.recovered_hit_config,intent,side,limb,speed,a.elapsed);
                    if(score.points){hits[side]++;points[side]+=score.points;auto& other=a.fighter[side^1];other.last_hit_valid=1;other.last_hit_age=0;other.last_hit_speed=speed;}
                }
                entered=entered||result.entered;intersects=intersects||result.overlap_after_start;
            }
            // Preserve the existing per-limb maximum sphere-center velocity
            // proxy, including its aggregation across intersecting targets.
            if(p.contact_velocity==rek_contact_velocity::Mode::LegacySphereProxy&&intersects){float d2=0;for(int k=0;k<3;k++){float d=to[k]-from[k];d2+=d*d;}max_relative_speed=fmaxf(max_relative_speed,sqrtf(d2)/DT);}
        }
        // History above advances even without intent and never resets at move
        // start. Native apex, body cooldown and invocation dedup stay unchanged.
        if(p.contact_velocity==rek_contact_velocity::Mode::BodyCvel||!can_score||!entered)continue;
        const auto result=rek5_recovered::score(a.recovered_hits,p.recovered_hit_config,intent,side,limb,max_relative_speed,a.elapsed);
        if(result.points){hits[side]++;points[side]+=result.points;auto& other=a.fighter[side^1];other.last_hit_valid=1;other.last_hit_age=0;other.last_hit_speed=max_relative_speed;}
    }
}
struct WarpContactScratch {
    std::uint64_t delta[6][2];
    float speed[6];
    unsigned eligible_entry[6];
    unsigned pair_entered[rek_contact_entry::Pairs];
    float pair_speed[rek_contact_entry::Pairs];
};
// Independent limb geometry, followed by the original ordered scorer on lane0.
// Each limb owns disjoint pair bits. XOR deltas merge without atomic updates.
__device__ void geom_pair_contacts_warp(const View& v,Arena& a,int side,int* hits,int* points,
        int lane,WarpContactScratch& scratch){
    using namespace rek5_primitive;
    const Parameters& p=*v.p;auto& f=a.fighter[side];const auto& enemy=a.fighter[side^1];
    const auto& now=v.frames[frame_index(p,f,false)];const auto& before=v.frames[frame_index(p,f,true)];
    const auto& target=v.frames[frame_index(p,enemy,false)];const auto& old_target=v.frames[frame_index(p,enemy,true)];
    if(lane<6){
        const int limb=lane;
        rek_contact_entry::State history=f.contact_pairs;
        RekG1StrikeIntent intent{};bool can_score=f.strike_active;
        if(can_score){
            const auto& route=p.routes[f.route];
            intent.impact_events=p.impact_events+p.impact_offsets[f.route];intent.impact_event_count=p.impact_counts[f.route];
            intent.clip_cursor_frames=clampf(route.start_frame+f.phase*route.playback_speed,float(route.start_frame),float(route.end_frame));
            intent.clip_fps=50;intent.move_id=f.move_instance;intent.action_playing=intent.layer_active=1;
            const auto part=limb<2?REK_G1_BODY_PART_FOOT:limb<4?REK_G1_BODY_PART_HAND:REK_G1_BODY_PART_SHIN;
            const auto hand=(limb&1)?REK_G1_HAND_RIGHT:REK_G1_HAND_LEFT;
            int32_t apex=-1;float ramp=0;
            can_score=rek5_recovered::embedded_strike_intent_apex(&intent,part,hand,p.recovered_hit_config.apex_min_ramp,&apex,&ramp);
        }
        float tip[3],old_tip[3];point(f,now.strike_xyz[limb],false,tip);point(f,before.strike_xyz[limb],true,old_tip);
        bool entered=false;float max_relative_speed=0;
        for(int zone=0;zone<rek_contact_entry::Targets;zone++){
            float dst[3],old_dst[3],from[3],to[3];
            point(enemy,target.target_shapes[zone].center,false,dst);point(enemy,old_target.target_shapes[zone].center,true,old_dst);
            for(int k=0;k<3;k++){from[k]=old_tip[k]-old_dst[k];to[k]=tip[k]-dst[k];}
            const float radius=fmaxf(before.strike_radius[limb],now.strike_radius[limb])+
                fmaxf(old_target.target_shape_radius[zone],target.target_shape_radius[zone]);
            const bool broad=sweep_distance2(from,to)<=radius*radius;
            bool intersects=false;
            for(int i=0;i<rek_contact_entry::Strikers;i++)if(p.strike_limb[i]==limb){
                const int pair=i*rek_contact_entry::Targets+zone;
                if(p.contact_velocity==rek_contact_velocity::Mode::BodyCvel)scratch.pair_entered[pair]=0;
                if(!broad){rek_contact_entry::update(history,pair,false);continue;}
                if(!can_score){
                    rek_contact_entry::unscored_endpoint(history,pair,
                        world_shape(now.strike_shapes[i],f.x,f.y,f.yaw),
                        world_shape(target.target_shapes[zone],enemy.x,enemy.y,enemy.yaw));
                    continue;
                }
                const auto result=rek_contact_entry::sample(history,pair,
                    world_shape(before.strike_shapes[i],f.old_x,f.old_y,f.old_yaw),
                    world_shape(now.strike_shapes[i],f.x,f.y,f.yaw),
                    world_shape(old_target.target_shapes[zone],enemy.old_x,enemy.old_y,enemy.old_yaw),
                    world_shape(target.target_shapes[zone],enemy.x,enemy.y,enemy.yaw),p.contact_substeps);
                if(p.contact_velocity==rek_contact_velocity::Mode::BodyCvel&&result.entered){
                    scratch.pair_entered[pair]=1;
                    scratch.pair_speed[pair]=rek_contact_velocity::relative_speed(
                        body_linear_velocity(v,f,side,rek_contact_velocity::limb_slot(limb)),
                        body_linear_velocity(v,enemy,side^1,rek_contact_velocity::target_slot(zone)));
                }
                entered=entered||result.entered;intersects=intersects||result.overlap_after_start;
            }
            // Preserve the existing per-limb maximum sphere-center velocity
            // proxy, including its aggregation across intersecting targets.
            if(p.contact_velocity==rek_contact_velocity::Mode::LegacySphereProxy&&intersects){float d2=0;for(int k=0;k<3;k++){float d=to[k]-from[k];d2+=d*d;}max_relative_speed=fmaxf(max_relative_speed,sqrtf(d2)/DT);}
        }
        scratch.delta[limb][0]=history.words[0]^f.contact_pairs.words[0];
        scratch.delta[limb][1]=history.words[1]^f.contact_pairs.words[1];
        scratch.speed[limb]=max_relative_speed;
        scratch.eligible_entry[limb]=can_score&&entered;
    }
    __syncwarp();
    if(lane==0){
        for(int limb=0;limb<6;limb++){
            f.contact_pairs.words[0]^=scratch.delta[limb][0];
            f.contact_pairs.words[1]^=scratch.delta[limb][1];
        }
        for(int limb=0;limb<6;limb++)if(scratch.eligible_entry[limb]){
            const auto& route=p.routes[f.route];RekG1StrikeIntent intent{};
            intent.impact_events=p.impact_events+p.impact_offsets[f.route];intent.impact_event_count=p.impact_counts[f.route];
            intent.clip_cursor_frames=clampf(route.start_frame+f.phase*route.playback_speed,float(route.start_frame),float(route.end_frame));
            intent.clip_fps=50;intent.move_id=f.move_instance;intent.action_playing=intent.layer_active=1;
            if(p.contact_velocity==rek_contact_velocity::Mode::BodyCvel){
                // Preserve original limb, target-zone, then striker order. A
                // rejected slow entry cannot borrow a later/persistent speed.
                for(int zone=0;zone<rek_contact_entry::Targets;zone++)
                    for(int i=0;i<rek_contact_entry::Strikers;i++)if(p.strike_limb[i]==limb){
                        const int pair=i*rek_contact_entry::Targets+zone;
                        if(!scratch.pair_entered[pair])continue;
                        const float speed=scratch.pair_speed[pair];
                        const auto result=rek5_recovered::score(a.recovered_hits,p.recovered_hit_config,intent,side,limb,speed,a.elapsed);
                        if(result.points){hits[side]++;points[side]+=result.points;auto& other=a.fighter[side^1];other.last_hit_valid=1;other.last_hit_age=0;other.last_hit_speed=speed;}
                    }
            }else{
            const float max_relative_speed=scratch.speed[limb];
            const auto result=rek5_recovered::score(a.recovered_hits,p.recovered_hit_config,intent,side,limb,max_relative_speed,a.elapsed);
            if(result.points){hits[side]++;points[side]+=result.points;auto& other=a.fighter[side^1];other.last_hit_valid=1;other.last_hit_age=0;other.last_hit_speed=max_relative_speed;}
            }
        }
    }
    __syncwarp();
}
__device__ void strike_contacts(const View& v,Arena& a,int side,int* hits,int* points){
    if(v.p->contact_entry==rek_contact_entry::Mode::GeomPair){geom_pair_contacts(v,a,side,hits,points);return;}
    const Parameters& p=*v.p;Fighter& f=a.fighter[side];Fighter& enemy=a.fighter[side^1];
    if(!f.strike_active||(p.recovered_scoring<2&&f.route==23))return;
    const FastFrame& now=v.frames[frame_index(p,f,false)];
    const FastFrame& before=v.frames[frame_index(p,f,true)];
    const FastFrame& target=v.frames[frame_index(p,enemy,false)];
    const FastFrame& old_target=v.frames[frame_index(p,enemy,true)];
    for(int limb=0;limb<6;limb++){
        if(p.recovered_scoring<2&&!limb_enabled(f.route,limb))continue;
        float tip[3],old_tip[3];point(f,now.strike_xyz[limb],false,tip);point(f,before.strike_xyz[limb],true,old_tip);
        float dx=tip[0]-old_tip[0],dy=tip[1]-old_tip[1],dz=tip[2]-old_tip[2];
        float speed=sqrtf(dx*dx+dy*dy+dz*dz)/DT;
        bool touch=false;float max_relative_speed=0;
        const int target_count=p.primitive_contacts?rek5_native_contact::TargetCount:3;
        for(int zone=0;zone<target_count;zone++){
            float dst[3],old_dst[3],from[3],to[3];
            const float* target_center=p.primitive_contacts?target.target_shapes[zone].center:target.target_xyz[zone];
            const float* old_target_center=p.primitive_contacts?old_target.target_shapes[zone].center:old_target.target_xyz[zone];
            point(enemy,target_center,false,dst);point(enemy,old_target_center,true,old_dst);
            for(int k=0;k<3;k++){from[k]=old_tip[k]-old_dst[k];to[k]=tip[k]-dst[k];}
            // Keep legacy arithmetic and its original three targets unchanged.
            const float radius=p.primitive_contacts?
                fmaxf(before.strike_radius[limb],now.strike_radius[limb])+
                    fmaxf(old_target.target_shape_radius[zone],target.target_shape_radius[zone]):
                now.strike_radius[limb]+target.target_radius[zone];
            bool intersects=sweep_distance2(from,to)<=radius*radius;
            if(intersects&&p.primitive_contacts)intersects=primitive_touch(p,f,enemy,before,now,old_target,target,limb,zone);
            touch=touch||intersects;
            if(p.recovered_scoring&&intersects){float d2=0;for(int k=0;k<3;k++){float d=to[k]-from[k];d2+=d*d;}max_relative_speed=fmaxf(max_relative_speed,sqrtf(d2)/DT);}
        }
        unsigned bit=1u<<limb;
        if(!p.recovered_scoring&&touch&&!(f.contact_latched&bit)&&f.cooldown[limb]==0&&speed>=p.hit_speed){
            hits[side]++;
            points[side]++;
            enemy.last_hit_valid=1;enemy.last_hit_age=0;enemy.last_hit_speed=speed;f.cooldown[limb]=10;
        }else if(p.recovered_scoring&&touch&&!(f.contact_latched&bit)){
            const auto& route=p.routes[f.route];RekG1StrikeIntent intent{};
            intent.impact_events=p.impact_events+p.impact_offsets[f.route];intent.impact_event_count=p.impact_counts[f.route];
            intent.clip_cursor_frames=clampf(route.start_frame+f.phase*route.playback_speed,float(route.start_frame),float(route.end_frame));
            intent.clip_fps=50;intent.move_id=f.move_instance;intent.action_playing=intent.layer_active=1;
            auto result=rek5_recovered::score(a.recovered_hits,p.recovered_hit_config,intent,side,limb,max_relative_speed,a.elapsed);
            if(result.points){hits[side]++;points[side]+=result.points;enemy.last_hit_valid=1;enemy.last_hit_age=0;enemy.last_hit_speed=max_relative_speed;}
        }
        if(touch)f.contact_latched|=bit;else f.contact_latched&=~bit;
    }
}
__device__ void finish_round(const View& v,int index,Arena& a,RekNative5RoundResult& r){
    int winner=a.fighter[0].points==a.fighter[1].points?-1:(a.fighter[0].points>a.fighter[1].points?0:1);
    r.round_result=winner<0?3:1;r.round_winner=winner;
    r.fight_result=winner<0?0:1;r.fight_winner=winner;r.phase=4;r.terminal=1;
    r.completed_rounds++;if(winner<0)r.ties++;else r.wins[winner]++;
    for(int s=0;s<2;s++)r.completed_points[s]+=a.fighter[s].points;
    auto* log=reinterpret_cast<RekNative5Log*>(reinterpret_cast<char*>(v.out.logs)+index*v.out.log_stride_bytes);
    log->score+=a.fighter[0].points;log->episode_return+=a.episode_return;
    log->episode_length+=a.tick;log->hits+=a.episode_hits;log->falls+=a.fighter[0].falls;
    log->wins+=winner==0;log->losses+=winner==1;log->draws+=winner<0;
    log->actions_invalid+=a.invalid;log->n+=1;
}
__device__ void advance_arena(const View& v,int index){
    Arena& a=v.state[index];auto& r=v.rounds[index];const Parameters& p=*v.p;
    if(r.terminal)round_reset(p,a,r,false,index);
    if(p.contact_entry==rek_contact_entry::Mode::GeomPair&&!a.contact_pairs_initialized)seed_contact_pairs(v,a);
    if(p.rendered_observation)capture_observed_pose(p,a);
    a.delta[0]=a.delta[1]=a.hit_count=a.down_event[0]=a.down_event[1]=0;
    a.reward[0]=a.reward[1]=0;
    float previous_potential[2]={};
    if(p.shaping_weight>0)for(int side=0;side<2;side++)previous_potential[side]=shaping_potential(v,a,side);
    int actions[2];bool bot_rows[2]={};
    for(int side=0;side<2;side++){
        int row=index*2+side;
        // Override 2 invokes the configured GPU opponent on either side.
        // Override 1 remains an external action, including frozen opponents.
        int override_value=v.override_rows?v.override_rows[row]:0;
        if(override_value>2){a.failures|=16;override_value=0;}
        bot_rows[side]=p.recovered_bot&&a.opponent_mode==0&&(override_value==2||(override_value==0&&side));
        float value=bot_rows[side]?0.f:override_value==2?float(opponent_action(p,a,side)):
            (override_value==1?v.external[row]:(side?float(opponent_action(p,a,side)):v.out.actions[index]));
        actions[side]=action_value(a,value);v.actions[row]=float(actions[side]);
    }
    if(a.failures){r.failure_bits=a.failures;return;}
    a.tick++;a.elapsed=float(a.tick)*DT;
    rek5_bot1::Input bot_inputs[2]{};
    for(int side=0;side<2;side++)if(bot_rows[side]){
        const Fighter& f=a.fighter[side];const Fighter& other=a.fighter[side^1];
        const float dx=other.x-f.x,dy=other.y-f.y;
        float rendered[4];root_quaternion(f,v.frames[frame_index(p,f,false)],rendered);
        const float heading=rek_rendered_pose::heading(rendered);
        // Native signed angle is positive-right. Native command yaw/strafe
        // are positive-left already, matching the compact command convention.
        bot_inputs[side]={hypotf(dx,dy),-angle(atan2f(dy,dx)-heading)*(180.f/PI),DT,a.elapsed,a.elapsed,
            attacking(f),bool(other.down),bool(f.down),true};
    }
    if(a.reset_wait){
        if(--a.reset_wait==0){pose_reset(p,a);if(p.rendered_observation)capture_observed_pose(p,a);}
    }else{
        for(int s=0;s<2;s++){
            if(bot_rows[s]){
                const unsigned accepted=a.bot[s].accepted;advance_bot(p,a,s,bot_inputs[s]);
                v.actions[index*2+s]=a.bot[s].accepted!=accepted?float(p.move_to_action[p.routes[a.fighter[s].route].move]):0.f;
            }else advance_fighter(p,a.fighter[s],actions[s],a);
        }
        float dx=a.fighter[1].x-a.fighter[0].x,dy=a.fighter[1].y-a.fighter[0].y;
        float distance=hypotf(dx,dy),minimum=2*p.body_radius;
        if(distance<minimum){
            float correction=.5f*(minimum-distance);
            if(distance<1e-6f){dx=1;dy=0;distance=1;}
            for(int s=0;s<2;s++){float sign=s?1.f:-1.f;a.fighter[s].x+=sign*correction*dx/distance;a.fighter[s].y+=sign*correction*dy/distance;}
        }
        for(int s=0;s<2;s++)confine(p,a.fighter[s]);
        int hits[2]={},points[2]={};
        strike_contacts(v,a,0,hits,points);strike_contacts(v,a,1,hits,points);
        for(int s=0;s<2;s++)a.delta[s]=points[s];
        // V3: contact scores are not measured falls. The compact candidate
        // does not integrate balance/fall dynamics, so accumulating two kick
        // hits must not fabricate a knockdown and teleport both fighters.
        // A future knockdown model needs an explicit state/geometry contract.
        for(int s=0;s<2;s++)a.hit_count+=hits[s];
        for(int s=0;s<2;s++)a.fighter[s].points+=a.delta[s];
    }
    if(p.rendered_observation)update_observed_pose(v,a);
    for(int s=0;s<2;s++){
        const Fighter& f=a.fighter[s];
        if(!isfinite(f.x)||!isfinite(f.y)||!isfinite(f.yaw)||!isfinite(f.phase)||f.route<0||f.route>=24)a.failures|=1;
        r.points[s]=f.points;r.falls[s]=f.falls;
    }
    r.time_remaining_seconds=fmaxf(0,p.round_seconds-a.elapsed);r.failure_bits=a.failures;
    bool terminal=a.elapsed>=p.round_seconds;
    const int winner=a.fighter[0].points==a.fighter[1].points?-1:
        (a.fighter[0].points>a.fighter[1].points?0:1);
    for(int side=0;side<2;side++){
        if(p.normalized_rewards){
            // Compact contacts still do not produce a physical fall event.
            const auto reward=rek5_normalized_reward::value(a.delta[side],a.delta[side^1],0);
            a.reward[side]=reward.reward;
            if(reward.saturated)++v.reward_saturations[index];
        }else a.reward[side]=rek5_round_reward::value(p.reward_mode,p.reward_gamma,
            a.fighter[side].points-a.delta[side],a.fighter[side^1].points-a.delta[side^1],
            a.fighter[side].points,a.fighter[side^1].points,terminal,winner,side);
        if(p.shaping_weight>0){
            const float next_potential=terminal?0:shaping_potential(v,a,side);
            a.reward[side]+=p.shaping_weight*rek5_contact_potential::shaping_delta(previous_potential[side],next_potential,terminal,p.shaping_gamma);
        }
    }
    a.episode_return+=a.reward[0];a.episode_hits+=a.hit_count;
    if(terminal&&!a.failures)finish_round(v,index,a,r);
}
__device__ void advance_arena_warp(const View& v,int index,int lane,WarpContactScratch& scratch){
    Arena& a=v.state[index];auto& r=v.rounds[index];const Parameters& p=*v.p;
    float previous_potential[2]={};bool do_contacts=false,input_failed=false;
    if(lane==0){
    if(r.terminal)round_reset(p,a,r,false,index);
    if(p.contact_entry==rek_contact_entry::Mode::GeomPair&&!a.contact_pairs_initialized)seed_contact_pairs(v,a);
    if(p.rendered_observation)capture_observed_pose(p,a);
    a.delta[0]=a.delta[1]=a.hit_count=a.down_event[0]=a.down_event[1]=0;
    a.reward[0]=a.reward[1]=0;
    if(p.shaping_weight>0)for(int side=0;side<2;side++)previous_potential[side]=shaping_potential(v,a,side);
    int actions[2];bool bot_rows[2]={};
    for(int side=0;side<2;side++){
        int row=index*2+side;
        // Override 2 invokes the configured GPU opponent on either side.
        // Override 1 remains an external action, including frozen opponents.
        int override_value=v.override_rows?v.override_rows[row]:0;
        if(override_value>2){a.failures|=16;override_value=0;}
        bot_rows[side]=p.recovered_bot&&a.opponent_mode==0&&(override_value==2||(override_value==0&&side));
        float value=bot_rows[side]?0.f:override_value==2?float(opponent_action(p,a,side)):
            (override_value==1?v.external[row]:(side?float(opponent_action(p,a,side)):v.out.actions[index]));
        actions[side]=action_value(a,value);v.actions[row]=float(actions[side]);
    }
    if(a.failures){r.failure_bits=a.failures;input_failed=true;}else{
    a.tick++;a.elapsed=float(a.tick)*DT;
    rek5_bot1::Input bot_inputs[2]{};
    for(int side=0;side<2;side++)if(bot_rows[side]){
        const Fighter& f=a.fighter[side];const Fighter& other=a.fighter[side^1];
        const float dx=other.x-f.x,dy=other.y-f.y;
        float rendered[4];root_quaternion(f,v.frames[frame_index(p,f,false)],rendered);
        const float heading=rek_rendered_pose::heading(rendered);
        // Native signed angle is positive-right. Native command yaw/strafe
        // are positive-left already, matching the compact command convention.
        bot_inputs[side]={hypotf(dx,dy),-angle(atan2f(dy,dx)-heading)*(180.f/PI),DT,a.elapsed,a.elapsed,
            attacking(f),bool(other.down),bool(f.down),true};
    }
    if(a.reset_wait){
        if(--a.reset_wait==0){pose_reset(p,a);if(p.rendered_observation)capture_observed_pose(p,a);}
    }else{
        for(int s=0;s<2;s++){
            if(bot_rows[s]){
                const unsigned accepted=a.bot[s].accepted;advance_bot(p,a,s,bot_inputs[s]);
                v.actions[index*2+s]=a.bot[s].accepted!=accepted?float(p.move_to_action[p.routes[a.fighter[s].route].move]):0.f;
            }else advance_fighter(p,a.fighter[s],actions[s],a);
        }
        float dx=a.fighter[1].x-a.fighter[0].x,dy=a.fighter[1].y-a.fighter[0].y;
        float distance=hypotf(dx,dy),minimum=2*p.body_radius;
        if(distance<minimum){
            float correction=.5f*(minimum-distance);
            if(distance<1e-6f){dx=1;dy=0;distance=1;}
            for(int s=0;s<2;s++){float sign=s?1.f:-1.f;a.fighter[s].x+=sign*correction*dx/distance;a.fighter[s].y+=sign*correction*dy/distance;}
        }
        for(int s=0;s<2;s++)confine(p,a.fighter[s]);
        do_contacts=true;
    }
    }
    }
    __syncwarp();
    if(__shfl_sync(0xffffffffu,int(input_failed),0))return;
    const bool run_contacts=__shfl_sync(0xffffffffu,int(do_contacts),0);
    if(run_contacts){
        int hits[2]={},points[2]={};
        geom_pair_contacts_warp(v,a,0,hits,points,lane,scratch);
        geom_pair_contacts_warp(v,a,1,hits,points,lane,scratch);
        if(lane==0){
        for(int s=0;s<2;s++)a.delta[s]=points[s];
        // V3: contact scores are not measured falls. The compact candidate
        // does not integrate balance/fall dynamics, so accumulating two kick
        // hits must not fabricate a knockdown and teleport both fighters.
        // A future knockdown model needs an explicit state/geometry contract.
        for(int s=0;s<2;s++)a.hit_count+=hits[s];
        for(int s=0;s<2;s++)a.fighter[s].points+=a.delta[s];
    }
    }
    __syncwarp();
    if(lane==0){
    if(p.rendered_observation)update_observed_pose(v,a);
    for(int s=0;s<2;s++){
        const Fighter& f=a.fighter[s];
        if(!isfinite(f.x)||!isfinite(f.y)||!isfinite(f.yaw)||!isfinite(f.phase)||f.route<0||f.route>=24)a.failures|=1;
        r.points[s]=f.points;r.falls[s]=f.falls;
    }
    r.time_remaining_seconds=fmaxf(0,p.round_seconds-a.elapsed);r.failure_bits=a.failures;
    bool terminal=a.elapsed>=p.round_seconds;
    const int winner=a.fighter[0].points==a.fighter[1].points?-1:
        (a.fighter[0].points>a.fighter[1].points?0:1);
    for(int side=0;side<2;side++){
        if(p.normalized_rewards){
            const auto reward=rek5_normalized_reward::value(a.delta[side],a.delta[side^1],0);
            a.reward[side]=reward.reward;
            if(reward.saturated)++v.reward_saturations[index];
        }else a.reward[side]=rek5_round_reward::value(p.reward_mode,p.reward_gamma,
            a.fighter[side].points-a.delta[side],a.fighter[side^1].points-a.delta[side^1],
            a.fighter[side].points,a.fighter[side^1].points,terminal,winner,side);
        if(p.shaping_weight>0){
            const float next_potential=terminal?0:shaping_potential(v,a,side);
            a.reward[side]+=p.shaping_weight*rek5_contact_potential::shaping_delta(previous_potential[side],next_potential,terminal,p.shaping_gamma);
        }
    }
    a.episode_return+=a.reward[0];a.episode_hits+=a.hit_count;
    if(terminal&&!a.failures)finish_round(v,index,a,r);
    }
}
__device__ float entity_value(const View& v,const Arena& a,int side,int field){
    const Fighter& f=a.fighter[side];const Parameters& p=*v.p;
    const FastFrame& frame=v.frames[frame_index(p,f,false)];
    if(field==0)return f.x;if(field==1)return f.y;if(field==2)return frame.root_z;
    if(field>=3&&field<7){float q[4];root_quaternion(f,frame,q);return q[field-3];}
    if(field==7)return p.rendered_observation?f.observed_local[0]:cosf(f.yaw)*f.vx+sinf(f.yaw)*f.vy;
    if(field==8)return p.rendered_observation?f.observed_local[1]:-sinf(f.yaw)*f.vx+cosf(f.yaw)*f.vy;
    if(field==12)return p.rendered_observation?f.observed_omega:f.omega;
    if(field>=13&&field<42)return frame.q[field-13];
    if(field>=42&&field<71){if(p.rendered_observation)return f.observed_joint_rate[field-42];const FastFrame& old=v.frames[frame_index(p,f,true)];return (frame.q[field-42]-old.q[field-42])/DT;}
    if(field==71)return f.down?1.f:0.f;
    if(field==72)return f.down?90.f:0.f;
    if(field==73)return frame.root_z;
    if(field==77)return 2;
    if(field==79)return f.down?1.f:0.f;
    if(field==80||field==81)return f.down?float(25-a.reset_wait)*DT:0;
    if(field==83)return float(a.reset_wait)*DT;
    if(field==85)return float(a.down_event[side]);
    return 0;
}
__device__ float raw_value(const View& v,int index,int side,int field){
    const Arena& a=v.state[index];const auto& r=v.rounds[index];const Fighter& f=a.fighter[side];
    if(field<172)return entity_value(v,a,field<86?side:side^1,field%86);
    if(field<184){
        int k=field-172;float fw,st,yaw;held_command(f.held,fw,st,yaw);
        if(f.bot_controlled){fw=f.bot_command.forward;st=f.bot_command.strafe;yaw=f.bot_command.yaw;}
        if(k==0)return cosf(.5f*(v.p->rendered_observation?f.observed_heading:f.yaw));if(k==3)return sinf(.5f*(v.p->rendered_observation?f.observed_heading:f.yaw));
        if(k==4)return attacking(f)?0:fw;if(k==5)return attacking(f)?0:st;if(k==6)return attacking(f)?0:yaw;
        if(k==7)return float(f.route);if(k==8)return f.route>0&&f.route<7;
        if(k==9)return !translating(f)&&!settled(*v.p,f);
        if(k==10||k==11)return attacking(f)?1.f:0.f;
        return 0;
    }
    int k=field-184,opponent=side^1;
    switch(k){
        case 3:return !r.terminal&&!f.bot_controlled?rek_owned_yaw::pending_value(v.p->owned_yaw_observation,attacking(f),f.held):0.f;
        case 0:return side;case 1:return r.phase;
        // A training episode is one independent round. The cumulative session
        // counter remains available in diagnostics, never in policy inputs.
        case 2:return 1;
        case 4:return v.p->round_seconds;case 5:return r.time_remaining_seconds;
        case 6:return f.points;case 7:return a.fighter[opponent].points;
        case 8:return f.falls;case 9:return a.fighter[opponent].falls;
        case 10:return r.terminal&&r.round_winner==side;case 11:return r.terminal&&r.round_winner==opponent;
        case 12:return f.last_hit_valid;case 13:return a.fighter[opponent].last_hit_valid;
        case 14:return f.last_hit_age;case 15:return a.fighter[opponent].last_hit_age;
        case 16:return f.last_hit_speed;case 17:return a.fighter[opponent].last_hit_speed;
        case 18:return f.down?2:0;case 19:return a.fighter[opponent].down?2:0;
        case 20:return f.down;case 21:return a.fighter[opponent].down;
        case 24:return a.reset_wait?float(25-a.reset_wait)*DT:0;case 25:return .5f;
        case 26:return r.round_result;case 27:return r.round_winner;case 28:return r.round_result==2;
        case 29:return r.fight_result;case 30:return r.fight_winner;
        case 33:return a.delta[side];case 34:return a.delta[opponent];
        case 35:return a.down_event[side];case 36:return a.down_event[opponent];
        case 37:case 38:return v.p->rendered_observation?rek_rendered_pose::weighted_delta_total(a.delta[side],a.delta[opponent]):a.hit_count;
    }
    return 0;
}
__device__ float scaled_value(const View& v,int index,int side,int field,float raw){
    const Fighter& f=v.state[index].fighter[side];const Fighter& enemy=v.state[index].fighter[side^1];
    if(field==86)return hypotf(enemy.x-f.x,enemy.y-f.y);
    if(field==87)return v.p->rendered_observation?rek_rendered_pose::bearing(f.x,f.y,enemy.x,enemy.y,f.observed_heading):angle(atan2f(enemy.y-f.y,enemy.x-f.x)-f.yaw)/PI;
    if(field==72||field==158)return raw/180;
    if(field==188||field==189)return raw/120;
    return raw;
}
__device__ float policy_value(const View& v,int index,int side,int field,float raw){
    if(v.policy_feature_mask&&!v.policy_feature_mask[field])return 0.f;
    return v.observable_observations?v.observable_observations[index*446+side*223+field]:scaled_value(v,index,side,field,raw);
}
__device__ void pack_observable(const View& v,int index){
    auto& a=v.state[index];auto& r=v.rounds[index];rek_fast_observable::Input input{};
    for(int side=0;side<2;side++){
        const auto& f=a.fighter[side];const auto& frame=v.frames[frame_index(*v.p,f,false)];
        input.root[side][0]=f.x;input.root[side][1]=f.y;input.root[side][2]=frame.root_z;
        root_quaternion(f,frame,input.root[side]+3);input.points[side]=f.points;
    }
    input.round_key=r.round_number;input.sample_seconds=double(a.tick)*.02;
    input.round_duration_seconds=v.p->round_seconds;input.round_remaining_seconds=r.time_remaining_seconds;
    input.round_active=r.phase==2;input.terminal=r.terminal!=0;
    if(rek_fast_observable::project(v.observable_history[index],input,
            v.observable_observations+index*446)!=rek_observable_balance::kOk){
        a.failures|=2048;r.failure_bits|=2048;
    }
}
__device__ void export_arena(const View& v,int index,int lane){
    if(v.observable_observations){if(lane==0)pack_observable(v,index);__syncwarp();}
    const Arena& a=v.state[index];const Parameters& p=*v.p;const auto& r=v.rounds[index];
    for(int side=0;side<2;side++){
        int row=2*index+side;const Fighter& f=a.fighter[side];
        const FastFrame& frame=v.frames[frame_index(p,f,false)];
        const FastFrame& old=v.frames[frame_index(p,f,true)];
        for(int k=lane;k<223;k+=32){
            float value=raw_value(v,index,side,k);v.raw[row*223+k]=value;
            if(side==0)v.out.observations[index*223+k]=policy_value(v,index,0,k,value);
        }
        for(int k=lane;k<33;k+=32){
            bool yaw_update=k==1||k==6||k==7;
            bool allowed=k==0||(!a.reset_wait&&(yaw_update||(!attacking(f)&&(k<16||settled(p,f)))));
            // The recovered Bot1 controller is never cadence-limited, including
            // its diagnostic side-0 override. Episode reset exports tick zero.
            int override_value=v.override_rows?v.override_rows[row]:0;
            bool bot=p.recovered_bot&&a.opponent_mode==0&&(override_value==2||(override_value==0&&side));
            allowed=allowed&&rek_action_cadence::permit(p.policy_action_stride,
                std::uint64_t(a.tick),k,side==0&&!bot,r.terminal);
            v.masks[row*33+k]=uint8_t(allowed);
            if(side==0&&v.learner_masks)v.learner_masks[index*33+k]=uint8_t(allowed);
        }
        for(int j=lane;j<29;j+=32){
            v.qpos[index*72+p.qindices[side][j]]=frame.q[j];
            v.qvel[index*70+p.vindices[side][j]]=(frame.q[j]-old.q[j])/DT;
        }
        if(lane==0){
            float* q=v.qpos+index*72+side*36;float* dq=v.qvel+index*70+side*35;
            q[0]=f.x;q[1]=f.y;q[2]=frame.root_z;root_quaternion(f,frame,q+3);
            dq[0]=f.vx;dq[1]=f.vy;dq[2]=dq[3]=dq[4]=0;dq[5]=f.omega;
            v.rewards[row]=a.reward[side];v.terminals[row]=r.terminal?1.f:0.f;
        }
    }
    if(lane==0){v.out.rewards[index]=a.reward[0];v.out.terminals[index]=r.terminal?1.f:0.f;}
}
__global__ void fast_reset(const __grid_constant__ View v){
    int lane=threadIdx.x&31,index=(blockIdx.x*blockDim.x+threadIdx.x)>>5;
    if(index>=v.arenas)return;
    if(lane==0){round_reset(*v.p,v.state[index],v.rounds[index],true,index);if(v.observable_history)v.observable_history[index]={};if(v.p->rendered_observation){capture_observed_pose(*v.p,v.state[index]);update_observed_pose(v,v.state[index]);}v.actions[index*2]=v.actions[index*2+1]=0;}
    __syncwarp();export_arena(v,index,lane);
}
// Keep the address-taken view in the kernel parameter space. The larger
// primitive path must not require a per-thread local copy of this descriptor.
__device__ void training_autoreset(const View& v,int index,int lane){
    // The learner must choose its next action from the next episode's initial
    // observation. Preserve the completed transition's reward/done separately.
    // Standalone evaluation never calls this path and retains its final pose.
    if(!v.rounds[index].terminal||v.rounds[index].failure_bits)return;
    float reward0=0,reward1=0;
    if(lane==0){
        reward0=v.state[index].reward[0];reward1=v.state[index].reward[1];
        round_reset(*v.p,v.state[index],v.rounds[index],false,index);
        if(v.p->rendered_observation){
            capture_observed_pose(*v.p,v.state[index]);update_observed_pose(v,v.state[index]);
        }
        v.actions[index*2]=v.actions[index*2+1]=0;
    }
    __syncwarp();export_arena(v,index,lane);__syncwarp();
    if(lane==0){
        v.out.rewards[index]=v.rewards[index*2]=reward0;
        v.rewards[index*2+1]=reward1;
        v.out.terminals[index]=v.terminals[index*2]=v.terminals[index*2+1]=1;
    }
}
__global__ void fast_step(const __grid_constant__ View v,bool autoreset=false){
    int lane=threadIdx.x&31,index=(blockIdx.x*blockDim.x+threadIdx.x)>>5;
    if(index>=v.arenas)return;
    if(lane==0)advance_arena(v,index);
    __syncwarp();export_arena(v,index,lane);
    if(autoreset){__syncwarp();training_autoreset(v,index,lane);}
}
__global__ void fast_step_warp(const __grid_constant__ View v,bool autoreset=false){
    __shared__ WarpContactScratch scratch[WARPS_PER_BLOCK];
    int lane=threadIdx.x&31,index=(blockIdx.x*blockDim.x+threadIdx.x)>>5;
    if(index>=v.arenas)return;
    advance_arena_warp(v,index,lane,scratch[threadIdx.x>>5]);
    __syncwarp();export_arena(v,index,lane);
    if(autoreset){__syncwarp();training_autoreset(v,index,lane);}
}
__global__ void encode_rows(const __grid_constant__ View v,float* out){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=v.arenas*446)return;
    int row=i/223,field=i%223;out[i]=policy_value(v,row/2,row%2,field,v.raw[i]);
}
__global__ void copy_masks(const __grid_constant__ View v){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<v.arenas*33)v.learner_masks[i]=v.masks[(i/33)*66+i%33];
}
float environment_float(const char* key,float fallback,float lo,float hi){
    const char* value=getenv(key);if(!value)return fallback;
    char* end=nullptr;float result=strtof(value,&end);
    if(!end||*end||!std::isfinite(result)||result<lo||result>hi)throw std::runtime_error(std::string("Invalid ")+key);
    return result;
}
int opponent_mode_from_environment(){
    const char* value=getenv("REK_FAST_OPPONENT_MODE");if(!value)return 0;
    const char* modes[]={"scripted","neutral","retreat","strafe","mixed"};
    for(int i=0;i<5;i++)if(!strcmp(value,modes[i]))return i;
    throw std::runtime_error("Invalid REK_FAST_OPPONENT_MODE");
}
int random_resets_from_environment(){
    const char* value=getenv("REK_FAST_RANDOM_RESETS");if(!value||!strcmp(value,"0"))return 0;
    if(!strcmp(value,"1"))return 1;
    throw std::runtime_error("REK_FAST_RANDOM_RESETS must be 0 or 1");
}
void valid_runtime(RekNative5Runtime* runtime){if(!runtime)throw std::runtime_error("Null semantic CUDA runtime");}
}

struct RekNative5Runtime { rek5::DeviceStorage storage;View view{};bool cooperative_contacts=false; };

extern "C" RekNative5Runtime* rek_native5_create(const RekNative5Config* config,const RekNative5Buffers* buffers,cudaStream_t stream){
    try{
        error_text.clear();
        const char* backend=getenv("REK_PHYSICS_BACKEND");
        if(!backend||strcmp(backend,"semantic_cuda"))throw std::runtime_error("Fast binary requires REK_PHYSICS_BACKEND=semantic_cuda");
        if(!config||config->abi_version!=REK_NATIVE5_RUNTIME_ABI||config->arenas<=0||!buffers||
           !buffers->observations||!buffers->actions||!buffers->rewards||!buffers->terminals||!buffers->logs||
           buffers->log_stride_bytes<sizeof(RekNative5Log))throw std::runtime_error("Invalid semantic CUDA configuration/buffers");
        const auto feature_mask=rek_policy_features::load(getenv("REK_POLICY_FEATURE_MASK"));
        const auto velocity_mode=rek_contact_velocity::parse(getenv("REK_FAST_CONTACT_VELOCITY"));
        FastAssets assets=load_fast_assets(*config,velocity_mode==rek_contact_velocity::Mode::BodyCvel);Parameters p{};
        p.contact_velocity=velocity_mode;
        const char* schema=getenv("REK_OBSERVATION_SCHEMA");
        const bool observable_balance=schema&&!strcmp(schema,rek_observable_balance::kSchema);
        p.owned_yaw_observation=observable_balance?false:rek_owned_yaw::enabled(schema);
        if(p.owned_yaw_observation)fprintf(stderr,"semantic_cuda_observation_schema=%s;owned_command_column=187;physics_changed=false\n",rek_owned_yaw::kSchema);
        p.policy_action_stride=rek_action_cadence::parse(getenv("REK_POLICY_ACTION_STRIDE"));
        p.yaw_command=rek_keyboard_yaw::parse(getenv("REK_FAST_YAW_COMMAND"));
        fprintf(stderr,"semantic_cuda_yaw_command={\"mode\":\"%s\",\"keyboard_command_ramp_seconds\":%.9g,\"command_parameter_provenance\":\"recovered_schedule_contract_expected_value_not_measured_actuator_lag\",\"command_dt_seconds\":%.9g,\"scope\":\"non_bot_policy_or_scripted_controller\",\"physical_response\":\"%s\",\"recovered_bot1_changed\":false,\"authentic_physical_parity\":false}\n",rek_keyboard_yaw::name(p.yaw_command),rek_keyboard_yaw::kRampSeconds,DT,p.yaw_command==rek_keyboard_yaw::Mode::KeyboardReset?"normalized_command_times_candidate_yaw_speed_no_additional_lag":"legacy_velocity_slew");
        if(p.policy_action_stride!=1)fprintf(stderr,"policy_action_cadence=%s;stride=%d;clock=episode_tick;phase_zero=initial_export;learner_only=true;control_hz=50;reward_aggregation=false;observation_features_unchanged=true\n",rek_action_cadence::kContract,p.policy_action_stride);
        const char* scoring=getenv("REK_FAST_SCORING");
        if(scoring&&strcmp(scoring,"v4_spheres")&&strcmp(scoring,"recovered_hit_rules_v1")&&strcmp(scoring,"recovered_hit_rules_v2"))throw std::runtime_error("Invalid REK_FAST_SCORING");
        p.recovered_scoring=!scoring||!strcmp(scoring,"v4_spheres")?0:!strcmp(scoring,"recovered_hit_rules_v1")?1:2;
        const char* bot_mode=getenv("REK_FAST_OPPONENT");
        if(bot_mode&&strcmp(bot_mode,"v4_scripted")&&strcmp(bot_mode,"recovered_bot1_v1"))throw std::runtime_error("Invalid REK_FAST_OPPONENT");
        p.recovered_bot=bot_mode&&!strcmp(bot_mode,"recovered_bot1_v1");
        const char* observation=getenv("REK_FAST_OBSERVATION");
        if(observation&&strcmp(observation,"v4_logical")&&strcmp(observation,"rendered_pose_v1"))throw std::runtime_error("Invalid REK_FAST_OBSERVATION");
        p.rendered_observation=observation&&!strcmp(observation,"rendered_pose_v1");
        const char* geometry=getenv("REK_FAST_GEOMETRY");
        if(geometry&&strcmp(geometry,"bounding_spheres")&&strcmp(geometry,"primitive_samples_v1"))throw std::runtime_error("Invalid REK_FAST_GEOMETRY");
        p.primitive_contacts=geometry&&!strcmp(geometry,"primitive_samples_v1");
        p.contact_entry=rek_contact_entry::parse(getenv("REK_FAST_CONTACT_ENTRY"));
        if(p.contact_entry==rek_contact_entry::Mode::GeomPair&&(!p.primitive_contacts||p.recovered_scoring!=2))
            throw std::runtime_error("geom_pair_v1 requires primitive_samples_v1 and recovered_hit_rules_v2");
        if(!rek_contact_velocity::compatible(p.contact_velocity,p.contact_entry==rek_contact_entry::Mode::GeomPair,p.primitive_contacts,p.recovered_scoring))
            throw std::runtime_error("body_cvel_v1 requires geom_pair_v1, primitive_samples_v1 and recovered_hit_rules_v2");
        if(p.contact_entry==rek_contact_entry::Mode::GeomPair)fprintf(stderr,"semantic_cuda_contact_entry={\"mode\":\"geom_pair_v1\",\"pair_identity\":\"distinct_striker_geom_target_geom\",\"pairs_per_fighter\":108,\"history_updates_without_intent\":true,\"reset_at_attack_start\":false,\"persistent_state\":\"endpoint_only\",\"sampled_enter_exit\":true,\"initial_overlap_seeded_without_enter\":true,\"velocity_proxy_changed\":%s,\"authentic_parity\":false}\n",p.contact_velocity==rek_contact_velocity::Mode::BodyCvel?"true":"false");
        float substeps=environment_float("REK_FAST_CONTACT_SUBSTEPS",4,1,16);
        if(substeps!=floorf(substeps))throw std::runtime_error("REK_FAST_CONTACT_SUBSTEPS must be an integer");
        p.contact_substeps=int(substeps);
        memcpy(p.strike_limb,assets.strike_limb,sizeof(p.strike_limb));
        if(p.recovered_bot&&!assets.recovered_catalog_compatible)throw std::runtime_error("Recovered Bot1 requires exact verified G1 route and impact metadata");
        if(p.recovered_scoring&&!assets.recovered_catalog_compatible)throw std::runtime_error("Recovered scoring requires exact verified build, strike catalog, route and clip metadata");
        std::copy(assets.impact_events.begin(),assets.impact_events.end(),p.impact_events);
        memcpy(p.impact_offsets,assets.impact_offsets,sizeof(p.impact_offsets));memcpy(p.impact_counts,assets.impact_counts,sizeof(p.impact_counts));
        p.recovered_hit_config=assets.recovered_hit_config;
        std::copy(assets.routes.begin(),assets.routes.end(),p.routes);
        std::copy(assets.action_to_route.begin(),assets.action_to_route.end(),p.action_to_route);
        for(int action=16;action<33;action++)p.move_to_action[assets.routes[assets.action_to_route[action]].move]=action;
        for(int route=7;route<24;route++)for(int j=0;j<assets.impact_counts[route];j++){
            int limb=int(assets.impact_events[assets.impact_offsets[route]+j].limb);
            if(limb){p.bot_catalog.primary_limb[assets.routes[route].move]=limb;break;}
        }
        std::copy(assets.move_duration_ticks.begin(),assets.move_duration_ticks.end(),p.durations);
        memcpy(p.qindices,assets.qindices,sizeof(p.qindices));memcpy(p.vindices,assets.vindices,sizeof(p.vindices));
        memcpy(p.initial_qpos,assets.initial_qpos,sizeof(p.initial_qpos));memcpy(p.spawn,assets.spawn_xy,sizeof(p.spawn));
        memcpy(p.heading,assets.initial_heading,sizeof(p.heading));memcpy(p.half_extent,assets.arena_half_extent,sizeof(p.half_extent));
        p.floor=assets.floor_height;p.round_seconds=config->round_seconds>0?config->round_seconds:120;
        p.move_speed=environment_float("REK_FAST_MOVE_SPEED",1.f,.01f,10.f);
        p.yaw_speed=environment_float("REK_FAST_YAW_SPEED",1.8f,.01f,10.f);
        p.brake_rate=std::max(.01f,assets.stop_brake_rate);p.yaw_ramp=std::max(DT,assets.yaw_ramp_seconds);
        p.settle_speed=std::max(.001f,assets.settle_linear_speed);
        p.body_radius=environment_float("REK_FAST_BODY_RADIUS",.22f,.05f,1.f);
        p.hit_speed=environment_float("REK_FAST_HIT_SPEED",.35f,0,20);
        p.seed=config->seed;p.opponent_mode=opponent_mode_from_environment();p.random_resets=random_resets_from_environment();
        p.reset_gap_min=environment_float("REK_FAST_RESET_GAP_MIN",.55f,.001f,100.f);
        p.reset_gap_max=environment_float("REK_FAST_RESET_GAP_MAX",2.5f,.001f,100.f);
        p.reset_heading_spread=environment_float("REK_FAST_RESET_HEADING_SPREAD_RAD",PI,0,PI);
        if(p.reset_gap_min>p.reset_gap_max)throw std::runtime_error("Reset gap minimum exceeds maximum");
        if(p.random_resets&&(p.reset_gap_min<2*p.body_radius||p.reset_gap_max>2*(std::min(p.half_extent[0],p.half_extent[1])-p.body_radius-.01f)))
            throw std::runtime_error("Random reset gaps must be nonoverlapping and fit every sampled axis inside the arena");
        p.shaping_weight=environment_float("REK_FAST_SHAPING_WEIGHT",0,0,100.f);
        const char* reward=getenv("REK_FAST_REWARD");
        if(reward&&strcmp(reward,"point_difference_v1")&&strcmp(reward,"round_outcome_v1")&&strcmp(reward,rek5_normalized_reward::kMode))
            throw std::runtime_error("Invalid REK_FAST_REWARD");
        p.normalized_rewards=reward&&!strcmp(reward,rek5_normalized_reward::kMode);
        if(p.normalized_rewards&&p.shaping_weight>0)
            throw std::runtime_error("Normalized point/fall reward disables additional spatial shaping");
        p.reward_mode=reward&&!strcmp(reward,"round_outcome_v1")?
            rek5_round_reward::RoundOutcome:rek5_round_reward::PointDifference;
        if(p.reward_mode==rek5_round_reward::RoundOutcome&&(!getenv("REK_FAST_REWARD_GAMMA")||p.shaping_weight>0))
            throw std::runtime_error("Round outcome reward requires explicit learner-matched REK_FAST_REWARD_GAMMA and disables spatial shaping");
        p.reward_gamma=environment_float("REK_FAST_REWARD_GAMMA",1.f,.000001f,1.f);
        if(p.shaping_weight>0&&!getenv("REK_FAST_SHAPING_GAMMA"))throw std::runtime_error("Positive shaping weight requires explicit REK_FAST_SHAPING_GAMMA matching learner discount");
        p.shaping_gamma=environment_float("REK_FAST_SHAPING_GAMMA",1.f,.000001f,1.f);
        p.shaping_target=environment_float("REK_FAST_SHAPING_TARGET",.65f,.001f,100.f);
        p.shaping_bearing_weight=environment_float("REK_FAST_SHAPING_BEARING_WEIGHT",0,0,100.f);
        const char* contact_model=getenv("REK_FAST_CONTACT_POTENTIAL");
        if(contact_model){
            if(!*contact_model||p.shaping_weight<=0||!p.rendered_observation)
                throw std::runtime_error("Contact potential requires a model, positive shaping weight and rendered_pose_v1 observations");
            const auto loaded=rek5_contact_potential::load(contact_model);p.contact_potential=loaded.model;
            fprintf(stderr,"contact_potential={\"schema\":\"rek.contact_potential.v1\",\"model_id\":\"%s\",\"source_sha256\":\"%s\",\"file_sha256\":\"%s\",\"samples\":%d,\"fit_round\":1,\"changes_points\":false,\"hit_probability\":false,\"executed_action_labels\":false,\"candidate_units_per_unity_unit_assumed\":1,\"metre_calibration_verified\":false}\n",loaded.model_id.c_str(),loaded.source_sha256.c_str(),loaded.file_sha256.c_str(),loaded.model.count);
        }
        if(p.shaping_weight>0&&p.shaping_target<2*p.body_radius)throw std::runtime_error("Shaping target is inside the nonoverlap distance");
        // Existing private V2 configs may still carry this setting. It has no
        // effect in V4; retain an explicit diagnostic rather than using it.
        if(getenv("REK_FAST_DOWN_DAMAGE"))fprintf(stderr,"REK_FAST_DOWN_DAMAGE ignored: compact v4 has no synthetic hit-damage knockdowns\n");
        if(!std::isfinite(p.round_seconds)||p.round_seconds<DT||assets.frames.empty())throw std::runtime_error("Invalid semantic CUDA duration/assets");
        for(int k=0;k<24;k++)if(p.routes[k].count<=0||p.routes[k].offset<0||size_t(p.routes[k].offset+p.routes[k].count)>assets.frames.size())throw std::runtime_error("Invalid baked route extent");
        for(int k=16;k<33;k++)if(p.action_to_route[k]<7||p.action_to_route[k]>=24||p.routes[p.action_to_route[k]].move<0||p.routes[p.action_to_route[k]].move>=17)throw std::runtime_error("Invalid baked action mapping");
        auto result=std::make_unique<RekNative5Runtime>();auto& v=result->view;int a=config->arenas;
        result->cooperative_contacts=p.contact_entry==rek_contact_entry::Mode::GeomPair;
        v.arenas=a;v.out=*buffers;v.p=result->storage.upload(&p,1);v.frames=result->storage.upload(assets.frames);
        if(p.contact_velocity==rek_contact_velocity::Mode::BodyCvel){
            if(assets.body_velocity_frames.size()!=assets.frames.size())throw std::runtime_error("Missing optional body cvel frames");
            v.body_velocity_frames=result->storage.upload(assets.body_velocity_frames);
        }
        if(feature_mask.enabled)v.policy_feature_mask=result->storage.upload(feature_mask.values.data(),feature_mask.values.size());
        v.state=result->storage.alloc<Arena>(a);v.rounds=result->storage.alloc<RekNative5RoundResult>(a);
        if(observable_balance){
            v.observable_history=result->storage.alloc<rek_fast_observable::History>(a);
            v.observable_observations=result->storage.alloc<float>(size_t(a)*446);
        }
        if(p.normalized_rewards)v.reward_saturations=result->storage.alloc<unsigned>(a);
        v.raw=result->storage.alloc<float>(size_t(a)*446);v.qpos=result->storage.alloc<float>(size_t(a)*72);
        v.qvel=result->storage.alloc<float>(size_t(a)*70);v.masks=result->storage.alloc<uint8_t>(size_t(a)*66);
        v.actions=result->storage.alloc<float>(size_t(a)*2);v.rewards=result->storage.alloc<float>(size_t(a)*2);v.terminals=result->storage.alloc<float>(size_t(a)*2);
        fast_reset<<<(a+WARPS_PER_BLOCK-1)/WARPS_PER_BLOCK,THREADS,0,stream>>>(v);
        rek5::cuda_check(cudaGetLastError());rek5::cuda_check(cudaStreamSynchronize(stream));
        fprintf(stderr,"semantic_cuda_v4: %d arenas; 50 Hz; one fused GPU step; canned poses=%zu; move_speed=%.6g m/s yaw_speed=%.6g rad/s; points-only slider dynamics; knockdowns unmodeled; parity=false\n",a,assets.frames.size(),p.move_speed,p.yaw_speed);
        fprintf(stderr,"semantic_cuda_assets=%s\n",assets.provenance_json.c_str());
        if(observable_balance)fprintf(stderr,"observable_balance={\"schema\":\"rek.native5.observable_balance.v1\",\"features\":223,\"root\":\"candidate_slider_origin_composed_clip_wxyz\",\"history\":\"preceding_50Hz_observation\",\"history_reset\":\"explicit_reset_or_episode_boundary_only\",\"joint_pose_available\":false,\"referee_available\":false,\"fall_event_source\":\"unavailable_in_compact\",\"raw_inspection_unchanged\":true,\"old_weights_compatible\":false,\"authentic_parity\":false}\n");
        if(p.contact_velocity==rek_contact_velocity::Mode::BodyCvel){
            fprintf(stderr,"semantic_cuda_body_velocity_assets=%s\n",assets.body_velocity_provenance_json.c_str());
            fprintf(stderr,"semantic_cuda_contact_velocity={\"mode\":\"body_cvel_v1\",\"classification\":\"kinematic_proxy\",\"sampling\":\"end_of_tick_rates_for_each_sampled_entry\",\"aggregation\":\"entered_geom_pair_own_body_velocity_norm\",\"persistent_target_lends_speed\":false,\"reference\":\"root_subtree_com\",\"controller_response\":false,\"contact_solver_response\":false,\"balance_dynamics\":false,\"authentic_physical_parity\":false}\n");
        }
        if(feature_mask.enabled)fprintf(stderr,"semantic_cuda_policy_feature_mask={\"enabled\":true,\"bytes\":223,\"sha256\":\"%s\",\"kept_features\":%d,\"raw_diagnostics_changed\":false}\n",
            feature_mask.sha256.c_str(),int(std::count(feature_mask.values.begin(),feature_mask.values.end(),uint8_t(1))));
        fprintf(stderr,"semantic_cuda_reward={\"mode\":\"%s\",\"gamma\":%.9g,\"point_input\":\"awarded_scoreboard_points\",\"terminal_signal\":\"completed_round_only\",\"countout_is_terminal\":false,\"terminal_win\":%d,\"terminal_loss\":%d,\"terminal_draw\":0,\"potential_scale_points\":5,\"terminal_potential\":0,\"adds_balance_dynamics\":false}\n",
            p.normalized_rewards?rek5_normalized_reward::kMode:p.reward_mode==rek5_round_reward::RoundOutcome?"round_outcome_v1":"point_difference_v1",
            p.reward_gamma,p.reward_mode==rek5_round_reward::RoundOutcome?1:0,p.reward_mode==rek5_round_reward::RoundOutcome?-1:0);
        if(p.normalized_rewards)fprintf(stderr,"normalized_reward={\"scale\":0.01,\"bounds\":[-1,1],\"own_confirmed_fall\":-0.01,\"fall_event_source\":\"unavailable_in_compact\",\"terminal_bonus\":0,\"normalization\":\"fixed_scale\"}\n");
        fprintf(stderr,"semantic_cuda_scoring={\"mode\":\"%s\",\"geometry\":\"%s\",\"contact_substeps\":%d,\"continuous_collision_detection\":false,\"speed\":\"%s\",\"speed_threshold_m_s\":%.9g,\"cooldown_seconds\":%.9g,\"apex_gate\":%s,\"per_invocation_apex_dedup\":%s,\"hand_points\":1,\"foot_shin_points\":%d,\"hit_count_is_unweighted\":true,\"upright_model\":\"constant_upright_no_balance_dynamics\",\"contact_enter_model\":\"%s\",\"authentic_parity\":false}\n",
            p.recovered_scoring==2?"recovered_hit_rules_v2":p.recovered_scoring?"recovered_hit_rules_v1":"v4_spheres",p.primitive_contacts?"primitive_samples_v1":"bounding_spheres",p.primitive_contacts?p.contact_substeps:0,p.contact_velocity==rek_contact_velocity::Mode::BodyCvel?"entered_pair_kinematic_body_cvel":p.recovered_scoring?"maximum_relative_sphere_center_finite_difference_proxy":"absolute_striker_sphere_center_finite_difference",p.recovered_scoring?p.recovered_hit_config.speed_threshold_mps:p.hit_speed,p.recovered_scoring?p.recovered_hit_config.per_body_cooldown_seconds:10*DT,p.recovered_scoring?"true":"false",p.recovered_scoring?"true":"false",p.recovered_scoring?2:1,p.contact_entry==rek_contact_entry::Mode::GeomPair?"geom_pair_v1":"compact_per_limb_union_latch_reset_at_move_start");
        const char* modes[]={"scripted","neutral","retreat","strafe","mixed"};
        fprintf(stderr,"semantic_cuda_opponent={\"implementation\":\"%s\",\"replaces\":\"scripted_rows_only\",\"difficulty\":0,\"decision_hz\":50,\"native_update_fixedupdate_equivalence\":false,\"rng\":\"%s\",\"continuous_commands\":%s,\"pose_route\":\"dominant_translation_canned_proxy\",\"actuator_model\":\"compact_slider\",\"own_recovery\":\"unsupported_fail_closed\",\"server_parity\":false}\n",
            p.recovered_bot?"recovered_bot1_v1":"v4_scripted",p.recovered_bot?"candidate_private_xorshift32":"legacy_stateless_reset_hash",p.recovered_bot?"true":"false");
        fprintf(stderr,"semantic_cuda_parameters={\"version\":4,\"scoring_mode\":\"%s\",\"policy_round_feature\":\"episode_local_constant_1\",\"diagnostic_round_counter\":\"cumulative_session\",\"dt_seconds\":%.9g,\"round_seconds\":%.9g,\"move_speed_m_s\":%.9g,\"yaw_speed_rad_s\":%.9g,\"brake_rate_m_s2\":%.9g,\"yaw_ramp_seconds\":%.9g,\"settle_speed_m_s\":%.9g,\"body_radius_m\":%.9g,\"hit_speed_m_s\":%.9g,\"knockdowns_modeled\":false,\"hit_damage_resets\":false,\"hit_cooldown_ticks\":%d,\"floor_z_m\":%.9g,\"arena_half_extent_m\":[%.9g,%.9g],\"physics_parity\":false,"
            "\"target_contract\":\"%s\",\"target_count\":%d,\"legacy_target_count\":3,"
            "\"seed\":%u,\"opponent_mode\":\"%s\",\"opponent_controller\":\"%s\",\"observation_mode\":\"%s\",\"mixed_weights\":[0.25,0.25,0.25,0.25],\"random_resets\":%s,\"reset_gap_min_m\":%.9g,\"reset_gap_max_m\":%.9g,\"reset_heading_spread_rad\":%.9g,"
            "\"shaping_weight\":%.9g,\"shaping_gamma\":%.9g,\"shaping_target_m\":%.9g,\"shaping_bearing_weight\":%.9g,\"shaping_terminal_potential\":0,\"shaping_changes_points\":false}\n",
            p.recovered_scoring==2?"recovered_hit_rules_v2":p.recovered_scoring?"recovered_hit_rules_v1":"v4_spheres",DT,p.round_seconds,p.move_speed,p.yaw_speed,p.brake_rate,p.yaw_ramp,p.settle_speed,p.body_radius,p.recovered_scoring?p.recovered_hit_config.speed_threshold_mps:p.hit_speed,p.recovered_scoring?15:10,p.floor,p.half_extent[0],p.half_extent[1],
            p.primitive_contacts?rek5_native_contact::TargetContract:"legacy_three_target_spheres_v1",p.primitive_contacts?rek5_native_contact::TargetCount:3,
            p.seed,modes[p.opponent_mode],p.recovered_bot?"recovered_bot1_v1":"v4_scripted",p.rendered_observation?"rendered_pose_v1":"v4_logical",p.random_resets?"true":"false",p.reset_gap_min,p.reset_gap_max,p.reset_heading_spread,
            p.shaping_weight,p.shaping_gamma,p.shaping_target,p.shaping_bearing_weight);
        return result.release();
    }catch(const std::exception& e){error_text=e.what();return nullptr;}
}
extern "C" int rek_native5_reset(RekNative5Runtime* r,cudaStream_t s){try{valid_runtime(r);fast_reset<<<(r->view.arenas+3)/4,THREADS,0,s>>>(r->view);rek5::cuda_check(cudaGetLastError());return 0;}catch(const std::exception& e){error_text=e.what();return 1;}}
extern "C" int rek_native5_step(RekNative5Runtime* r,cudaStream_t s){try{valid_runtime(r);if(r->cooperative_contacts)fast_step_warp<<<(r->view.arenas+3)/4,THREADS,0,s>>>(r->view);else fast_step<<<(r->view.arenas+3)/4,THREADS,0,s>>>(r->view);rek5::cuda_check(cudaGetLastError());return 0;}catch(const std::exception& e){error_text=e.what();return 1;}}
extern "C" int rek_native5_step_autoreset(RekNative5Runtime* r,cudaStream_t s){try{valid_runtime(r);if(r->cooperative_contacts)fast_step_warp<<<(r->view.arenas+3)/4,THREADS,0,s>>>(r->view,true);else fast_step<<<(r->view.arenas+3)/4,THREADS,0,s>>>(r->view,true);rek5::cuda_check(cudaGetLastError());return 0;}catch(const std::exception& e){error_text=e.what();return 1;}}
extern "C" int rek_native5_bind_action_mask(RekNative5Runtime* r,uint8_t* mask,cudaStream_t s){try{valid_runtime(r);r->view.learner_masks=mask;if(mask)copy_masks<<<(r->view.arenas*33+127)/128,128,0,s>>>(r->view);rek5::cuda_check(cudaGetLastError());return 0;}catch(const std::exception& e){error_text=e.what();return 1;}}
extern "C" int rek_native5_bind_external_actions(RekNative5Runtime* r,const float* actions,const uint8_t* overrides,cudaStream_t){try{valid_runtime(r);if(bool(actions)!=bool(overrides))throw std::runtime_error("External actions and override mask must be bound together");r->view.external=actions;r->view.override_rows=overrides;return 0;}catch(const std::exception& e){error_text=e.what();return 1;}}
extern "C" int rek_native5_get_device_view(RekNative5Runtime* r,RekNative5DeviceView* out){try{valid_runtime(r);if(!out)throw std::runtime_error("Null device view");const auto& v=r->view;*out={v.arenas,72,70,v.raw,v.masks,v.qpos,v.qvel,v.actions,v.rewards,v.terminals,v.rounds};return 0;}catch(const std::exception& e){error_text=e.what();return 1;}}
extern "C" int rek_native5_encode_fighter_observations(RekNative5Runtime* r,float* out,cudaStream_t s){try{valid_runtime(r);if(!out)throw std::runtime_error("Null encoded observations");encode_rows<<<(r->view.arenas*446+127)/128,128,0,s>>>(r->view,out);rek5::cuda_check(cudaGetLastError());return 0;}catch(const std::exception& e){error_text=e.what();return 1;}}
extern "C" int rek_native5_read_snapshot(RekNative5Runtime* r,int arena,RekNative5Snapshot* out,cudaStream_t s){try{
    valid_runtime(r);if(!out||arena<0||arena>=r->view.arenas)throw std::runtime_error("Invalid snapshot arena");auto& v=r->view;out->arena=arena;
    auto copy=[&](void* target,const void* source,size_t size){rek5::cuda_check(cudaMemcpyAsync(target,source,size,cudaMemcpyDeviceToHost,s));};
    copy(out->raw_observations,v.raw+arena*446,sizeof(out->raw_observations));copy(out->action_masks,v.masks+arena*66,sizeof(out->action_masks));
    copy(out->qpos,v.qpos+arena*72,sizeof(out->qpos));copy(out->qvel,v.qvel+arena*70,sizeof(out->qvel));
    copy(out->actions,v.actions+arena*2,sizeof(out->actions));copy(out->rewards,v.rewards+arena*2,sizeof(out->rewards));
    copy(out->terminals,v.terminals+arena*2,sizeof(out->terminals));copy(&out->round,v.rounds+arena,sizeof(out->round));
    rek5::cuda_check(cudaStreamSynchronize(s));return 0;
}catch(const std::exception& e){error_text=e.what();return 1;}}
extern "C" int rek_native5_check_status(RekNative5Runtime* r,cudaStream_t s){try{
    valid_runtime(r);std::vector<RekNative5RoundResult> results(r->view.arenas);
    rek5::cuda_check(cudaMemcpyAsync(results.data(),r->view.rounds,results.size()*sizeof(results[0]),cudaMemcpyDeviceToHost,s));
    rek5::cuda_check(cudaStreamSynchronize(s));for(size_t a=0;a<results.size();a++)if(results[a].failure_bits)throw std::runtime_error("semantic_cuda arena "+std::to_string(a)+" failure bits "+std::to_string(results[a].failure_bits));return 0;
}catch(const std::exception& e){error_text=e.what();return 1;}}
extern "C" int rek_native5_close(RekNative5Runtime* r){
    int status=0;
    try{if(r&&r->view.reward_saturations){
        std::vector<unsigned> counts(r->view.arenas);
        rek5::cuda_check(cudaMemcpy(counts.data(),r->view.reward_saturations,counts.size()*sizeof(unsigned),cudaMemcpyDeviceToHost));
        unsigned long long total=0;for(auto count:counts)total+=count;
        fprintf(stderr,"normalized_reward_saturations=%llu\n",total);
    }}catch(const std::exception& e){error_text=e.what();status=1;}
    delete r;return status;
}
extern "C" const char* rek_native5_error(void){return error_text.c_str();}
