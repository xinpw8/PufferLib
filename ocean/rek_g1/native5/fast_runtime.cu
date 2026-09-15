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

// Explicit reduced-order candidate. The source clips provide pose and strike
// trajectories; slider motion and sphere contacts are modeling
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
};
struct Arena {
    Fighter fighter[2];
    float elapsed,episode_return,episode_hits,invalid;
    int tick,reset_wait,dummy_move[2];
    int delta[2],hit_count,down_event[2];
    unsigned failures;
};
struct Parameters {
    FastRoute routes[24];
    int action_to_route[33],qindices[2][29],vindices[2][29];
    unsigned durations[17];
    float initial_qpos[72],spawn[2][2],heading[2],half_extent[2];
    float floor,round_seconds,move_speed,yaw_speed,brake_rate,yaw_ramp,settle_speed;
    float body_radius,hit_speed;
};
struct View {
    int arenas;
    const Parameters* p;
    const FastFrame* frames;
    Arena* state;
    RekNative5RoundResult* rounds;
    RekNative5Buffers out;
    float *raw,*qpos,*qvel,*actions,*rewards,*terminals;
    uint8_t *masks,*learner_masks;
    const float* external;
    const uint8_t* override_rows;
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
__device__ void pose_reset(const Parameters& p,Arena& a){
    for(int side=0;side<2;side++){
        Fighter& f=a.fighter[side];int points=f.points,falls=f.falls;
        f={};f.points=points;f.falls=falls;
        f.x=f.old_x=p.spawn[side][0];f.y=f.old_y=p.spawn[side][1];
        f.yaw=f.old_yaw=p.heading[side];f.held=1;
    }
}
__device__ void round_reset(const Parameters& p,Arena& a,RekNative5RoundResult& result,bool full){
    if(full){result={};result.round_number=1;}
    else result.round_number++;
    a={};pose_reset(p,a);
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
__device__ void advance_fighter(const Parameters& p,Fighter& f,int action,Arena& a){
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
        f.strike_active=1;
        f.phase=fminf(float(p.routes[f.route].count-1),
            float(f.move_tick+1)*float(p.routes[f.route].count-1)/float(f.attack_duration));
        f.move_tick++;
        if(f.move_tick>=f.attack_duration)f.attack_duration=0;
        return;
    }
    float forward,strafe,yaw;held_command(f.held,forward,strafe,yaw);
    float sn,cs;sincosf(f.yaw,&sn,&cs);
    float tx=p.move_speed*(cs*forward-sn*strafe),ty=p.move_speed*(sn*forward+cs*strafe);
    float accel=p.brake_rate*DT;
    f.vx=approach(f.vx,tx,accel);f.vy=approach(f.vy,ty,accel);
    f.omega=approach(f.omega,yaw*p.yaw_speed,p.yaw_speed*DT/p.yaw_ramp);
    f.x+=f.vx*DT;f.y+=f.vy*DT;f.yaw=angle(f.yaw+f.omega*DT);
    int route=p.action_to_route[f.held];
    if(route!=f.route){f.route=route;f.phase=0;f.old_route=route;f.old_phase=0;}
    else f.phase+=1;
    const FastRoute& r=p.routes[f.route];
    if(r.loop&&f.phase>=r.count)f.phase-=r.count;
    else f.phase=fminf(f.phase,float(r.count-1));
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
__device__ void strike_contacts(View v,Arena& a,int side,int* hits){
    const Parameters& p=*v.p;Fighter& f=a.fighter[side];Fighter& enemy=a.fighter[side^1];
    if(!f.strike_active||f.route==23)return;
    const FastFrame& now=v.frames[frame_index(p,f,false)];
    const FastFrame& before=v.frames[frame_index(p,f,true)];
    const FastFrame& target=v.frames[frame_index(p,enemy,false)];
    const FastFrame& old_target=v.frames[frame_index(p,enemy,true)];
    for(int limb=0;limb<6;limb++){
        if(!limb_enabled(f.route,limb))continue;
        float tip[3],old_tip[3];point(f,now.strike_xyz[limb],false,tip);point(f,before.strike_xyz[limb],true,old_tip);
        float dx=tip[0]-old_tip[0],dy=tip[1]-old_tip[1],dz=tip[2]-old_tip[2];
        float speed=sqrtf(dx*dx+dy*dy+dz*dz)/DT;
        bool touch=false;
        for(int zone=0;zone<3;zone++){
            float dst[3],old_dst[3],from[3],to[3];
            point(enemy,target.target_xyz[zone],false,dst);point(enemy,old_target.target_xyz[zone],true,old_dst);
            for(int k=0;k<3;k++){from[k]=old_tip[k]-old_dst[k];to[k]=tip[k]-dst[k];}
            float radius=now.strike_radius[limb]+target.target_radius[zone];
            touch=touch||sweep_distance2(from,to)<=radius*radius;
        }
        unsigned bit=1u<<limb;
        if(touch&&!(f.contact_latched&bit)&&f.cooldown[limb]==0&&speed>=p.hit_speed){
            hits[side]++;
            enemy.last_hit_valid=1;enemy.last_hit_age=0;enemy.last_hit_speed=speed;f.cooldown[limb]=10;
        }
        if(touch)f.contact_latched|=bit;else f.contact_latched&=~bit;
    }
}
__device__ void finish_round(View v,int index,Arena& a,RekNative5RoundResult& r){
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
__device__ void advance_arena(View v,int index){
    Arena& a=v.state[index];auto& r=v.rounds[index];const Parameters& p=*v.p;
    if(r.terminal)round_reset(p,a,r,false);
    a.delta[0]=a.delta[1]=a.hit_count=a.down_event[0]=a.down_event[1]=0;
    int actions[2];
    for(int side=0;side<2;side++){
        int row=index*2+side;
        // semantic_cuda extends the inspection override with 2=the same GPU
        // scripted baseline on either side, for side-balanced evaluation.
        int override_value=v.override_rows?v.override_rows[row]:0;
        if(override_value>2){a.failures|=16;override_value=0;}
        float value=override_value==2?float(scripted_action(p,a,side)):
            (override_value==1?v.external[row]:(side?float(scripted_action(p,a,side)):v.out.actions[index]));
        actions[side]=action_value(a,value);v.actions[row]=float(actions[side]);
    }
    if(a.failures){r.failure_bits=a.failures;return;}
    a.tick++;a.elapsed=float(a.tick)*DT;
    if(a.reset_wait){
        if(--a.reset_wait==0)pose_reset(p,a);
    }else{
        for(int s=0;s<2;s++)advance_fighter(p,a.fighter[s],actions[s],a);
        float dx=a.fighter[1].x-a.fighter[0].x,dy=a.fighter[1].y-a.fighter[0].y;
        float distance=hypotf(dx,dy),minimum=2*p.body_radius;
        if(distance<minimum){
            float correction=.5f*(minimum-distance);
            if(distance<1e-6f){dx=1;dy=0;distance=1;}
            for(int s=0;s<2;s++){float sign=s?1.f:-1.f;a.fighter[s].x+=sign*correction*dx/distance;a.fighter[s].y+=sign*correction*dy/distance;}
        }
        for(int s=0;s<2;s++)confine(p,a.fighter[s]);
        int hits[2]={};
        strike_contacts(v,a,0,hits);strike_contacts(v,a,1,hits);
        for(int s=0;s<2;s++)a.delta[s]=hits[s];
        // V3: contact scores are not measured falls. The compact candidate
        // does not integrate balance/fall dynamics, so accumulating two kick
        // hits must not fabricate a knockdown and teleport both fighters.
        // A future knockdown model needs an explicit state/geometry contract.
        for(int s=0;s<2;s++)a.hit_count+=hits[s];
        for(int s=0;s<2;s++)a.fighter[s].points+=a.delta[s];
    }
    for(int s=0;s<2;s++){
        const Fighter& f=a.fighter[s];
        if(!isfinite(f.x)||!isfinite(f.y)||!isfinite(f.yaw)||!isfinite(f.phase)||f.route<0||f.route>=24)a.failures|=1;
        r.points[s]=f.points;r.falls[s]=f.falls;
    }
    a.episode_return+=float(a.delta[0]-a.delta[1]);a.episode_hits+=a.hit_count;
    r.time_remaining_seconds=fmaxf(0,p.round_seconds-a.elapsed);r.failure_bits=a.failures;
    bool terminal=a.elapsed>=p.round_seconds;
    if(terminal&&!a.failures)finish_round(v,index,a,r);
}
__device__ float entity_value(View v,const Arena& a,int side,int field){
    const Fighter& f=a.fighter[side];const Parameters& p=*v.p;
    const FastFrame& frame=v.frames[frame_index(p,f,false)];
    if(field==0)return f.x;if(field==1)return f.y;if(field==2)return frame.root_z;
    if(field>=3&&field<7){float q[4];root_quaternion(f,frame,q);return q[field-3];}
    if(field==7)return cosf(f.yaw)*f.vx+sinf(f.yaw)*f.vy;
    if(field==8)return -sinf(f.yaw)*f.vx+cosf(f.yaw)*f.vy;
    if(field==12)return f.omega;
    if(field>=13&&field<42)return frame.q[field-13];
    if(field>=42&&field<71){const FastFrame& old=v.frames[frame_index(p,f,true)];return (frame.q[field-42]-old.q[field-42])/DT;}
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
__device__ float raw_value(View v,int index,int side,int field){
    const Arena& a=v.state[index];const auto& r=v.rounds[index];const Fighter& f=a.fighter[side];
    if(field<172)return entity_value(v,a,field<86?side:side^1,field%86);
    if(field<184){
        int k=field-172;float fw,st,yaw;held_command(f.held,fw,st,yaw);
        if(k==0)return cosf(.5f*f.yaw);if(k==3)return sinf(.5f*f.yaw);
        if(k==4)return attacking(f)?0:fw;if(k==5)return attacking(f)?0:st;if(k==6)return attacking(f)?0:yaw;
        if(k==7)return float(f.route);if(k==8)return f.route>0&&f.route<7;
        if(k==9)return !translating(f)&&!settled(*v.p,f);
        if(k==10||k==11)return attacking(f)?1.f:0.f;
        return 0;
    }
    int k=field-184,opponent=side^1;
    switch(k){
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
        case 37:case 38:return a.hit_count;
    }
    return 0;
}
__device__ float scaled_value(View v,int index,int side,int field,float raw){
    const Fighter& f=v.state[index].fighter[side];const Fighter& enemy=v.state[index].fighter[side^1];
    if(field==86)return hypotf(enemy.x-f.x,enemy.y-f.y);
    if(field==87)return angle(atan2f(enemy.y-f.y,enemy.x-f.x)-f.yaw)/PI;
    if(field==72||field==158)return raw/180;
    if(field==188||field==189)return raw/120;
    return raw;
}
__device__ void export_arena(View v,int index,int lane){
    const Arena& a=v.state[index];const Parameters& p=*v.p;const auto& r=v.rounds[index];
    for(int side=0;side<2;side++){
        int row=2*index+side;const Fighter& f=a.fighter[side];
        const FastFrame& frame=v.frames[frame_index(p,f,false)];
        const FastFrame& old=v.frames[frame_index(p,f,true)];
        for(int k=lane;k<223;k+=32){
            float value=raw_value(v,index,side,k);v.raw[row*223+k]=value;
            if(side==0)v.out.observations[index*223+k]=scaled_value(v,index,0,k,value);
        }
        for(int k=lane;k<33;k+=32){
            bool yaw_update=k==1||k==6||k==7;
            bool allowed=k==0||(!a.reset_wait&&(yaw_update||(!attacking(f)&&(k<16||settled(p,f)))));
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
            v.rewards[row]=float(a.delta[side]-a.delta[side^1]);v.terminals[row]=r.terminal?1.f:0.f;
        }
    }
    if(lane==0){v.out.rewards[index]=float(a.delta[0]-a.delta[1]);v.out.terminals[index]=r.terminal?1.f:0.f;}
}
__global__ void fast_reset(View v){
    int lane=threadIdx.x&31,index=(blockIdx.x*blockDim.x+threadIdx.x)>>5;
    if(index>=v.arenas)return;
    if(lane==0){round_reset(*v.p,v.state[index],v.rounds[index],true);v.actions[index*2]=v.actions[index*2+1]=0;}
    __syncwarp();export_arena(v,index,lane);
}
__global__ void fast_step(View v){
    int lane=threadIdx.x&31,index=(blockIdx.x*blockDim.x+threadIdx.x)>>5;
    if(index>=v.arenas)return;
    if(lane==0)advance_arena(v,index);
    __syncwarp();export_arena(v,index,lane);
}
__global__ void encode_rows(View v,float* out){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=v.arenas*446)return;
    int row=i/223,field=i%223;out[i]=scaled_value(v,row/2,row%2,field,v.raw[i]);
}
__global__ void copy_masks(View v){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<v.arenas*33)v.learner_masks[i]=v.masks[(i/33)*66+i%33];
}
float environment_float(const char* key,float fallback,float lo,float hi){
    const char* value=getenv(key);if(!value)return fallback;
    char* end=nullptr;float result=strtof(value,&end);
    if(!end||*end||!std::isfinite(result)||result<lo||result>hi)throw std::runtime_error(std::string("Invalid ")+key);
    return result;
}
void valid_runtime(RekNative5Runtime* runtime){if(!runtime)throw std::runtime_error("Null semantic CUDA runtime");}
}

struct RekNative5Runtime { rek5::DeviceStorage storage;View view{}; };

extern "C" RekNative5Runtime* rek_native5_create(const RekNative5Config* config,const RekNative5Buffers* buffers,cudaStream_t stream){
    try{
        error_text.clear();
        const char* backend=getenv("REK_PHYSICS_BACKEND");
        if(!backend||strcmp(backend,"semantic_cuda"))throw std::runtime_error("Fast binary requires REK_PHYSICS_BACKEND=semantic_cuda");
        if(!config||config->abi_version!=REK_NATIVE5_RUNTIME_ABI||config->arenas<=0||!buffers||
           !buffers->observations||!buffers->actions||!buffers->rewards||!buffers->terminals||!buffers->logs||
           buffers->log_stride_bytes<sizeof(RekNative5Log))throw std::runtime_error("Invalid semantic CUDA configuration/buffers");
        FastAssets assets=load_fast_assets(*config);Parameters p{};
        std::copy(assets.routes.begin(),assets.routes.end(),p.routes);
        std::copy(assets.action_to_route.begin(),assets.action_to_route.end(),p.action_to_route);
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
        // Existing private V2 configs may still carry this setting. It has no
        // effect in V3; retain an explicit diagnostic rather than using it.
        if(getenv("REK_FAST_DOWN_DAMAGE"))fprintf(stderr,"REK_FAST_DOWN_DAMAGE ignored: compact v3 has no synthetic hit-damage knockdowns\n");
        if(!std::isfinite(p.round_seconds)||p.round_seconds<DT||assets.frames.empty())throw std::runtime_error("Invalid semantic CUDA duration/assets");
        for(int k=0;k<24;k++)if(p.routes[k].count<=0||p.routes[k].offset<0||size_t(p.routes[k].offset+p.routes[k].count)>assets.frames.size())throw std::runtime_error("Invalid baked route extent");
        for(int k=16;k<33;k++)if(p.action_to_route[k]<7||p.action_to_route[k]>=24||p.routes[p.action_to_route[k]].move<0||p.routes[p.action_to_route[k]].move>=17)throw std::runtime_error("Invalid baked action mapping");
        auto result=std::make_unique<RekNative5Runtime>();auto& v=result->view;int a=config->arenas;
        v.arenas=a;v.out=*buffers;v.p=result->storage.upload(&p,1);v.frames=result->storage.upload(assets.frames);
        v.state=result->storage.alloc<Arena>(a);v.rounds=result->storage.alloc<RekNative5RoundResult>(a);
        v.raw=result->storage.alloc<float>(size_t(a)*446);v.qpos=result->storage.alloc<float>(size_t(a)*72);
        v.qvel=result->storage.alloc<float>(size_t(a)*70);v.masks=result->storage.alloc<uint8_t>(size_t(a)*66);
        v.actions=result->storage.alloc<float>(size_t(a)*2);v.rewards=result->storage.alloc<float>(size_t(a)*2);v.terminals=result->storage.alloc<float>(size_t(a)*2);
        fast_reset<<<(a+WARPS_PER_BLOCK-1)/WARPS_PER_BLOCK,THREADS,0,stream>>>(v);
        rek5::cuda_check(cudaGetLastError());rek5::cuda_check(cudaStreamSynchronize(stream));
        fprintf(stderr,"semantic_cuda_v3: %d arenas; 50 Hz; one fused GPU step; canned poses=%zu; move_speed=%.6g m/s yaw_speed=%.6g rad/s; points-only slider/sphere dynamics; knockdowns unmodeled; parity=false\n",a,assets.frames.size(),p.move_speed,p.yaw_speed);
        fprintf(stderr,"semantic_cuda_assets=%s\n",assets.provenance_json.c_str());
        fprintf(stderr,"semantic_cuda_parameters={\"version\":3,\"policy_round_feature\":\"episode_local_constant_1\",\"diagnostic_round_counter\":\"cumulative_session\",\"dt_seconds\":%.9g,\"round_seconds\":%.9g,\"move_speed_m_s\":%.9g,\"yaw_speed_rad_s\":%.9g,\"brake_rate_m_s2\":%.9g,\"yaw_ramp_seconds\":%.9g,\"settle_speed_m_s\":%.9g,\"body_radius_m\":%.9g,\"hit_speed_m_s\":%.9g,\"knockdowns_modeled\":false,\"hit_damage_resets\":false,\"hit_cooldown_ticks\":10,\"floor_z_m\":%.9g,\"arena_half_extent_m\":[%.9g,%.9g],\"physics_parity\":false}\n",DT,p.round_seconds,p.move_speed,p.yaw_speed,p.brake_rate,p.yaw_ramp,p.settle_speed,p.body_radius,p.hit_speed,p.floor,p.half_extent[0],p.half_extent[1]);
        return result.release();
    }catch(const std::exception& e){error_text=e.what();return nullptr;}
}
extern "C" int rek_native5_reset(RekNative5Runtime* r,cudaStream_t s){try{valid_runtime(r);fast_reset<<<(r->view.arenas+3)/4,THREADS,0,s>>>(r->view);rek5::cuda_check(cudaGetLastError());return 0;}catch(const std::exception& e){error_text=e.what();return 1;}}
extern "C" int rek_native5_step(RekNative5Runtime* r,cudaStream_t s){try{valid_runtime(r);fast_step<<<(r->view.arenas+3)/4,THREADS,0,s>>>(r->view);rek5::cuda_check(cudaGetLastError());return 0;}catch(const std::exception& e){error_text=e.what();return 1;}}
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
extern "C" int rek_native5_close(RekNative5Runtime* r){delete r;return 0;}
extern "C" const char* rek_native5_error(void){return error_text.c_str();}
