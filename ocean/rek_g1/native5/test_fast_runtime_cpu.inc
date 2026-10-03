// Included by test_fast_runtime_cpu_{original,modified}.cpp. Requires
// FAST_DEVICE_SOURCE (device half of fast_runtime.cu, see
// test_fast_runtime_cpu.sh), HARNESS_RUN and HARNESS_MODIFIED.
#include "test_fast_runtime_cpu.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>
// Single-lane emulation: lane-parallel loops are run lane by lane.
#define __shfl_sync(mask,value,lane) (value)
#define __syncwarp(...) ((void)0)
using std::max;using std::min;  // CUDA's integer min/max overloads
// Kernels compile as plain functions; the harness never calls them.
namespace { struct HarnessDim3 { unsigned x,y,z; }; const HarnessDim3 threadIdx{0,0,0},blockIdx{0,0,0},blockDim{128,1,1}; }
#include FAST_DEVICE_SOURCE
#if HARNESS_MODIFIED
#include "lite_fall_dataset_observe.h"
#include <cstdio>
#endif

namespace {
void hash_bytes(uint64_t& h,const void* data,size_t bytes){
    const auto* p=static_cast<const unsigned char*>(data);
    for(size_t i=0;i<bytes;i++){h^=p[i];h*=1099511628211ull;}
}
const uint32_t kDurations[17]={35,27,31,45,32,45,157,145,158,139,134,138,73,75,68,71,103};
const int kRouteMove[24]={-1,-1,-1,-1,-1,-1,-1,6,7,8,9,0,1,2,3,4,5,10,11,12,13,14,15};
constexpr int kFrames=50;

// Synthetic clips: kicks extend one foot and knees one knee; punches extend
// hands. The geometry only needs to create contacts at fighting range.
void synthetic_frame(FastFrame& f,int route,int frame){
    std::memset(&f,0,sizeof(f));
    f.root_wxyz[0]=1;f.root_z=.75f;
    for(int j=0;j<29;j++)f.q[j]=.01f*float((route*7+j*3+frame)%11);
    const float feet[2][3]={{0,.1f,.05f},{0,-.1f,.05f}},hands[2][3]={{.25f,.2f,1.f},{.25f,-.2f,1.f}};
    const float knees[2][3]={{.05f,.1f,.45f},{.05f,-.1f,.45f}};
    for(int s=0;s<2;s++)for(int k=0;k<3;k++){f.strike_xyz[s][k]=feet[s][k];f.strike_xyz[2+s][k]=hands[s][k];f.strike_xyz[4+s][k]=knees[s][k];}
    for(int i=0;i<6;i++)f.strike_radius[i]=.06f;
    const float targets[3][3]={{0,0,.8f},{0,0,1.1f},{0,0,1.35f}};
    for(int t=0;t<3;t++){for(int k=0;k<3;k++)f.target_xyz[t][k]=targets[t][k];f.target_radius[t]=.15f;}
    const int move=kRouteMove[route];if(move<0)return;
    const float t=float(frame)/float(kFrames-1),reach=std::sin(3.14159265f*t);
    if(move>=6&&move<=8){const int s=move==8?1:0;f.strike_xyz[s][0]=.1f+.75f*reach;f.strike_xyz[s][2]=.05f+.95f*reach;}
    else if(move==9){f.strike_xyz[5][0]=.05f+.55f*reach;f.strike_xyz[5][2]=.45f+.5f*reach;}
    else{const int s=(move==3||move==4)?1:0;f.strike_xyz[2+s][0]=.25f+.5f*reach;if(move>=10){f.strike_xyz[3-s][0]=.25f+.5f*reach;}}
}
void synthetic_parameters(Parameters& p,std::vector<FastFrame>& frames,const rek_fast_cpu::Config& c){
    frames.resize(24*kFrames);
    for(int r=0;r<24;r++){
        FastRoute& route=p.routes[r];route={};route.offset=r*kFrames;route.count=kFrames;route.fps=50;
        route.loop=r<7;route.move=kRouteMove[r];route.start_frame=0;route.end_frame=kFrames-1;route.playback_speed=1;
        for(int k=0;k<kFrames;k++)synthetic_frame(frames[r*kFrames+k],r,k);
    }
    const int held_route[16]={-1,0,1,2,3,4,5,6,1,1,2,2,3,3,4,4};
    for(int a=0;a<16;a++)p.action_to_route[a]=held_route[a];
    for(int a=16;a<33;a++)p.action_to_route[a]=a-9;
    for(int a=16;a<33;a++)p.move_to_action[p.routes[a-9].move]=a;
    for(int m=0;m<17;m++)p.durations[m]=kDurations[m];
    for(int s=0;s<2;s++)for(int j=0;j<29;j++){p.qindices[s][j]=s*36+7+j;p.vindices[s][j]=s*35+6+j;}
    p.initial_qpos[2]=p.initial_qpos[38]=.8f;p.initial_qpos[3]=p.initial_qpos[39]=1;
    p.spawn[0][0]=-.9f;p.spawn[1][0]=.9f;p.heading[0]=0;p.heading[1]=PI;
    p.half_extent[0]=p.half_extent[1]=2.f;p.floor=0;p.round_seconds=c.round_seconds;
    p.move_speed=1;p.yaw_speed=1.8f;p.brake_rate=2;p.yaw_ramp=.5f;p.settle_speed=.03f;
    p.body_radius=.22f;p.hit_speed=.35f;p.seed=c.seed;p.opponent_mode=0;p.random_resets=0;
    p.reward_mode=rek5_round_reward::PointDifference;p.reward_gamma=1;
    const int limbs[12]={0,0,0,0,1,1,1,1,2,3,4,5};std::memcpy(p.strike_limb,limbs,sizeof(limbs));
    p.contact_substeps=4;
}
float raw(const std::vector<float>& v,int arena,int side,int field){return v[(arena*2+side)*223+field];}
int learner_action(const rek_fast_cpu::Config& c,const std::vector<float>& raw_obs,const std::vector<uint8_t>& masks,int arena){
    const uint8_t* m=masks.data()+arena*66;
    const auto allowed=[&](int k){return m[k]!=0;};
    if(c.learner==rek_fast_cpu::Idle)return allowed(1)?1:0;
    if(c.learner==rek_fast_cpu::RepeatHH){if(allowed(17))return 17;return allowed(1)?1:0;}
    if(c.learner==rek_fast_cpu::AlternateKicks){if(allowed(17)||allowed(16))return (arena&1)?16:17;return allowed(1)?1:0;}
    // Approach the opponent, face it, then kick HH.
    const float x=raw(raw_obs,arena,0,0),y=raw(raw_obs,arena,0,1),ox=raw(raw_obs,arena,0,86),oy=raw(raw_obs,arena,0,87);
    const float qw=raw(raw_obs,arena,0,3),qz=raw(raw_obs,arena,0,6),qx=raw(raw_obs,arena,0,4),qy=raw(raw_obs,arena,0,5);
    const float yaw=std::atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz));
    const float bearing=std::atan2(std::sin(std::atan2(oy-y,ox-x)-yaw),std::cos(std::atan2(oy-y,ox-x)-yaw));
    if(std::fabs(bearing)>.16f&&(allowed(6)||allowed(7)))return bearing>0?6:7;
    if(std::hypot(ox-x,oy-y)>.8f&&allowed(2))return 2;
    if(allowed(17))return 17;
    return allowed(1)?1:0;
}
}  // namespace

namespace rek_fast_cpu {
Totals HARNESS_RUN(const Config& c){
    Totals t;Parameters p{};std::vector<FastFrame> frames;synthetic_parameters(p,frames,c);
#if HARNESS_MODIFIED
    if(!c.lite_model.empty())p.lite=rek_lite_falls::load(c.lite_model.c_str()).model;
    p.move_reward=c.move_reward;p.reward_move=c.reward_move;p.reward_move_value=c.reward_value;
#else
    if(!c.lite_model.empty()||c.move_reward)throw std::runtime_error("original runtime has no lite falls or move reward");
#endif
    const int n=c.arenas;
    std::vector<Arena> state(n);std::vector<RekNative5RoundResult> rounds(n);
    std::vector<float> obs(size_t(n)*223),actions(n),rewards(n),terminals(n),raw_obs(size_t(n)*446),qpos(size_t(n)*72),qvel(size_t(n)*70);
    std::vector<float> row_actions(size_t(n)*2),row_rewards(size_t(n)*2),row_terminals(size_t(n)*2);
    std::vector<uint8_t> masks(size_t(n)*66);std::vector<RekNative5Log> logs(n);
    View v{};v.arenas=n;v.p=&p;v.frames=frames.data();v.state=state.data();v.rounds=rounds.data();
    v.out.observations=obs.data();v.out.actions=actions.data();v.out.rewards=rewards.data();v.out.terminals=terminals.data();
    v.out.logs=logs.data();v.out.log_stride_bytes=sizeof(RekNative5Log);
    v.raw=raw_obs.data();v.qpos=qpos.data();v.qvel=qvel.data();v.actions=row_actions.data();v.rewards=row_rewards.data();
    v.terminals=row_terminals.data();v.masks=masks.data();
    auto export_all=[&](int i){for(int lane=0;lane<32;lane++)export_arena(v,i,lane);};
    auto hash_tick=[&](){
        hash_bytes(t.trajectory_hash,raw_obs.data(),raw_obs.size()*4);hash_bytes(t.trajectory_hash,masks.data(),masks.size());
        hash_bytes(t.trajectory_hash,qpos.data(),qpos.size()*4);hash_bytes(t.trajectory_hash,qvel.data(),qvel.size()*4);
        hash_bytes(t.trajectory_hash,row_rewards.data(),row_rewards.size()*4);hash_bytes(t.trajectory_hash,row_terminals.data(),row_terminals.size()*4);
        hash_bytes(t.trajectory_hash,row_actions.data(),row_actions.size()*4);hash_bytes(t.trajectory_hash,obs.data(),obs.size()*4);
    };
#if HARNESS_MODIFIED
    rek_lite_dataset::Settings dataset_settings{{.55f,.85f,1.2f,1.8f},{-.5f,.5f},{},1500};
    for(int m=0;m<17;m++)dataset_settings.durations[m]=p.durations[m];
    rek_lite_dataset::HostSink sink(dataset_settings.max_delay);
    std::vector<rek_lite_dataset::RowState> dataset_rows(size_t(n)*2,rek_lite_dataset::RowState{-1,0,0,0,0});
    std::vector<float> previous_distance(n,-1.f);
#endif
    for(int i=0;i<n;i++){round_reset(p,state[i],rounds[i],true,i);row_actions[2*i]=row_actions[2*i+1]=0;export_all(i);}
    hash_tick();
    for(int tick=0;tick<c.ticks;tick++){
        for(int i=0;i<n;i++){
            Arena& a=state[i];
            actions[i]=float(learner_action(c,raw_obs,masks,i));
#if HARNESS_MODIFIED
            const int phase_before[2]={a.lite.fighter[0].phase,a.lite.fighter[1].phase};
#endif
            advance_arena(v,i);export_all(i);
            const auto& r=rounds[i];
            if(r.failure_bits){t.invariant_failures++;continue;}
#if HARNESS_MODIFIED
            using namespace rek_lite_falls;
            for(int s=0;s<2;s++){
                if(a.started_move[s]>=0)(s?t.opponent_starts:t.learner_starts)[a.started_move[s]]++;
                const float expected=p.move_reward&&a.started_move[s]==p.reward_move?p.reward_move_value:0.f;
                if(p.move_reward&&a.reward[s]!=expected){t.invariant_failures++;}
                if(p.move_reward&&a.reward[s]!=0&&a.started_move[s]!=p.reward_move)t.reward_without_target_start++;
                if(p.move_reward&&a.started_move[s]==p.reward_move&&a.reward[s]==0)t.target_start_without_reward++;
                const uint32_t ev=a.lite.fighter[s].events;
                t.onsets[s]+=(ev&FallingStarted)!=0;t.recoveries[s]+=(ev&FallingCleared)!=0;t.falls[s]+=(ev&BecameFallen)!=0;
                if(!upright(a.lite.fighter[s]))for(int k=1;k<33;k++)if(masks[i*66+s*33+k])t.nonzero_mask_while_down++;
            }
            const uint32_t calls=a.lite.referee.calls;
            t.counts_started+=(calls&(CallSlip|CallKnockdown|CallDoubleKnockdown))!=0;
            if(calls&CallKnockout)t.knockouts++;
            if(calls&CallDoubleKnockout)t.double_knockouts++;
            if(calls&(CallKnockout|CallDoubleKnockout)){
                for(int s=0;s<2;s++)t.knockout_points[s]+=a.delta[s]>=kKnockoutPoints?kKnockoutPoints:0;
                if(p.round_seconds-a.elapsed>0){
                    t.spawn_resets++;
                    // A fighter still falling (down, uncounted) when the other's count
                    // expired is cleared by the reset without conceding: the edge case.
                    for(int s=0;s<2;s++)if(phase_before[s]==Falling)t.stuck_cleared_by_reset++;
                }
            }
            if(!a.both_upright&&a.hit_count)t.hits_while_not_upright++;
            if(r.terminal&&counting(a.lite.referee))t.terminal_during_count++;
            if(a.elapsed>=p.round_seconds&&!r.terminal)t.deferred_round_end_ticks++;
            if(a.fighter[0].falls<0||a.fighter[1].falls<0)t.invariant_failures++;
            if(p.move_reward){t.learner_reward+=a.reward[0];t.opponent_reward+=a.reward[1];t.nonzero_reward_ticks+=a.reward[0]!=0;}
            if(!c.dataset_out.empty())rek_lite_dataset::observe_arena(raw_obs.data()+size_t(i)*446,dataset_rows.data()+i*2,
                previous_distance[i],dataset_settings,r.terminal!=0,tick+1,sink.view(),rek_lite_dataset::HostAdd{});
#endif
            // Training autoreset: next episode's observation, completed transition's reward.
            if(r.terminal){
                t.rounds++;
                const float r0=state[i].reward[0],r1=state[i].reward[1];
                round_reset(p,state[i],rounds[i],false,i);row_actions[2*i]=row_actions[2*i+1]=0;export_all(i);
                rewards[i]=row_rewards[2*i]=r0;row_rewards[2*i+1]=r1;
                terminals[i]=row_terminals[2*i]=row_terminals[2*i+1]=1;
            }
        }
        hash_tick();t.ticks++;
    }
#if HARNESS_MODIFIED
    if(!c.dataset_out.empty()){
        FILE* f=std::fopen(c.dataset_out.c_str(),"wx");if(!f)throw std::runtime_error("dataset output must be new");
        rek_lite_dataset::write_dataset(f,"cpu-harness-seed-"+std::to_string(c.seed),dataset_settings,sink,
            "{\"source\":\"test_fast_runtime_cpu\",\"synthetic_assets\":true,\"seed\":"+std::to_string(c.seed)+"}");
        std::fclose(f);
    }
#endif
    return t;
}
}  // namespace rek_fast_cpu
