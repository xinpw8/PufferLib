// Bounded action-only exposure from native spawn. All dynamics run on CUDA.
// This schedule is a diagnostic controller, not a trained fighting policy.
#include "runtime_api.h"
#include "g1_fight_state.h"
#include <cuda_runtime.h>
#include <mujoco/mujoco.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace {
constexpr int arenas=4, ticks=2500;
constexpr uint32_t durations[17]={35,27,31,45,32,45,157,145,158,139,134,138,73,75,68,71,103};
constexpr int registry[17]={6,7,8,9,0,1,2,3,4,5,10,11,12,13,14,15,16};
void ck(cudaError_t x){if(x!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(x));}
void rk(int x){if(x)throw std::runtime_error(rek_native5_error());}
template<class T> void alloc(T*& p,size_t n){ck(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));ck(cudaMemset(p,0,n*sizeof(T)));}
const char* lane_name(int a){const char* names[]={"idle_both","stationary_attack_cycle_vs_idle","approach_move9_right_knee_vs_idle","approach_attack_cycles_both"};return names[a];}
struct Choice {int desired=1,selected=1;double distance=0,bearing=0;};
Choice choose(const RekNative5Snapshot& s,int lane,int side,unsigned& next_move) {
    const float* o=s.raw_observations+side*223;const uint8_t* mask=s.action_masks+side*33;
    Choice c;const bool attacks=lane==3||(lane>0&&side==0);
    const double w=o[3],x=o[4],y=o[5],z=o[6],n=w*w+x*x+y*y+z*z;
    if(!(n>0))throw std::runtime_error("Invalid root quaternion");
    const double fx=(w*w+x*x-y*y-z*z)/n,fy=2*(w*z+x*y)/n;
    const double dx=double(o[86])-o[0],dy=double(o[87])-o[1];
    c.distance=std::hypot(dx,dy);c.bearing=std::atan2(-fy*dx+fx*dy,fx*dx+fy*dy);
    if(attacks&&o[79]==0&&!s.round.terminal) {
        c.desired=lane==2?19:16+int(next_move%17);
        if(lane>=2){
            if(c.distance>1.05)c.desired=2;
            if(c.distance<.65)c.desired=3;
            if(std::abs(c.bearing)>.16)c.desired=c.bearing>0?6:7;
        }
    }
    // Preserve the historical close-contact controller's mask handling.
    // Both the requested and selected actions are recorded every tick.
    c.selected=c.desired;
    if(!mask[c.selected])c.selected=mask[1]?1:0;
    if(!mask[c.selected])for(int i=0;i<33;i++)if(mask[i]){c.selected=i;break;}
    if(!mask[c.selected])throw std::runtime_error("No legal action");
    if(c.selected>=16)++next_move;
    return c;
}
struct Stats {
    double min_gap=1e30;float min_height[2]={1e30f,1e30f},max_tilt[2]={};
    unsigned attack_requests[2]={},falling_events[2]={},fallen_events[2]={},falling_samples[2]={},fallen_samples[2]={};
    unsigned nonfoot_samples[2]={},attributed_contacts=0,scored_contacts=0,ko_ticks=0,reset_ticks=0,terminal_ticks=0;
};
void array(const char* name,const float* data,int count) {
    std::printf(",\"%s\":[",name);for(int i=0;i<count;i++)std::printf("%s%.9g",i?",":"",data[i]);std::putchar(']');
}
void record(int tick,int lane,const RekNative5Snapshot& s,const Choice* choice,Stats& stats) {
    for(float v:s.qpos)if(!std::isfinite(v))throw std::runtime_error("Nonfinite qpos");
    for(float v:s.qvel)if(!std::isfinite(v))throw std::runtime_error("Nonfinite qvel");
    for(float v:s.raw_observations)if(!std::isfinite(v))throw std::runtime_error("Nonfinite observation");
    const float* o=s.raw_observations;const unsigned signals=unsigned(o[215]),referee=unsigned(o[216]);
    const double gap=std::hypot(double(s.qpos[0])-s.qpos[36],double(s.qpos[1])-s.qpos[37]);
    const bool ko=referee&(REK_G1_REFEREE_KNOCKOUT|REK_G1_REFEREE_DOUBLE_KNOCKOUT);
    const bool reset=signals&REK_G1_FIGHT_SIGNAL_RESET_BOTH_TO_SPAWN;
    stats.min_gap=std::min(stats.min_gap,gap);stats.ko_ticks+=ko;stats.reset_ticks+=reset;stats.terminal_ticks+=s.round.terminal!=0;
    stats.attributed_contacts+=unsigned(o[221]);stats.scored_contacts+=unsigned(o[222]);
    std::printf("{\"event\":\"physical_exposure_sample\",\"tick\":%d,\"time_s\":%.9g,\"lane\":%d,\"gap_m\":%.9g,\"points\":[%d,%d],\"falls\":[%u,%u],\"score_delta\":[%.9g,%.9g],\"rewards\":[%.9g,%.9g],\"fight_signals\":%u,\"referee_calls\":%u,\"count_elapsed_s\":%.9g,\"count_duration_s\":%.9g,\"attributed_contacts\":%.9g,\"scored_contacts\":%.9g,\"ko_event\":%s,\"reset_signal\":%s,\"terminal\":%u,\"failure_bits\":%u,\"fighters\":[",tick,tick*.02,lane,gap,s.round.points[0],s.round.points[1],s.round.falls[0],s.round.falls[1],o[217],o[218],s.rewards[0],s.rewards[1],signals,referee,o[208],o[209],o[221],o[222],ko?"true":"false",reset?"true":"false",unsigned(s.round.terminal),s.round.failure_bits);
    for(int side=0;side<2;side++) {
        const float* f=o+side*223;const unsigned events=unsigned(f[85]);
        const int raw=int(f[217])-int(f[218])-((events&4)?1:0);
        const float expected=tick?float(std::clamp(double(raw)/100.0,-1.0,1.0)):0;
        if(s.rewards[side]!=expected)throw std::runtime_error("Recorded reward differs from score delta / own confirmed fall");
        stats.min_height[side]=std::min(stats.min_height[side],f[2]);stats.max_tilt[side]=std::max(stats.max_tilt[side],f[72]);
        stats.falling_events[side]+=(events&1)!=0;stats.fallen_events[side]+=(events&4)!=0;
        stats.falling_samples[side]+=f[79]==1;stats.fallen_samples[side]+=f[79]==2;stats.nonfoot_samples[side]+=f[77]>0;
        unsigned long long mask=0;for(int k=0;k<33;k++)mask|=static_cast<unsigned long long>(s.action_masks[side*33+k])<<k;
        const Choice c=choice?choice[side]:Choice{};
        std::printf("%s{\"requested_action\":%d,\"selected_action\":%d,\"recorded_action\":%.9g,\"selected_native_move\":%d,\"next_mask_bits\":%llu,\"decision_gap_m\":%.9g,\"decision_bearing_rad\":%.9g,\"route\":%.9g,\"busy\":%.9g,\"tilt_degrees\":%.9g,\"pelvis_height_ratio\":%.9g,\"foot_contacts\":[%.9g,%.9g],\"nonfoot_floor_bodies\":%.9g,\"fall_phase\":%.9g,\"fall_events\":%u,\"last_struck_valid\":%.9g,\"last_struck_age_s\":%.9g,\"last_struck_speed_m_s\":%.9g,\"fall_classification\":%.9g,\"count_active\":%.9g,\"count_is_slip\":%.9g}",side?",":"",c.desired,c.selected,s.actions[side],c.selected>=16?registry[c.selected-16]:-1,mask,c.distance,c.bearing,f[179],f[183],f[72],f[73],f[75],f[76],f[77],f[79],events,f[196],f[198],f[200],f[202],f[204],f[206]);
    }
    std::putchar(']');array("qpos",s.qpos,72);array("qvel",s.qvel,70);std::puts("}");
}
}
extern "C" void __wrap_mj_step(const mjModel*,mjData*){std::abort();}
extern "C" void __wrap_mj_step1(const mjModel*,mjData*){std::abort();}
extern "C" void __wrap_mj_step2(const mjModel*,mjData*){std::abort();}
extern "C" void __wrap_mj_forward(const mjModel*,mjData*){std::abort();}
extern "C" void __wrap_mj_kinematics(const mjModel*,mjData*){std::abort();}
int main(int argc,char** argv) {
    int completed=0;
    try {
        if(argc!=7)throw std::runtime_error("Usage: physical-fall-exposure-probe XML EXPORT ASSETS FEATURES ENCODER DECODER");
        if(!std::getenv("REK_PHYSICS_BACKEND")||std::string(std::getenv("REK_PHYSICS_BACKEND"))!="mujoco_cuda")throw std::runtime_error("Explicit mujoco_cuda required");
        if(!std::getenv("REK_NATIVE5_REWARD")||std::string(std::getenv("REK_NATIVE5_REWARD"))!="normalized_points_falls_v1")throw std::runtime_error("Explicit normalized reward mode required");
        cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        RekNative5Config cfg{};cfg.abi_version=REK_NATIVE5_RUNTIME_ABI;cfg.arenas=arenas;cfg.seed=73;cfg.locomotion_segment_ticks=1;cfg.round_seconds=60;
        cfg.model_path=argv[1];cfg.physics_export_path=argv[2];cfg.assets_path=argv[3];cfg.motion_features_path=argv[4];cfg.controller_encoder_path=argv[5];cfg.controller_decoder_path=argv[6];
        std::copy(durations,durations+17,cfg.move_duration_ticks);
        RekNative5Buffers buffers{};buffers.log_stride_bytes=sizeof(RekNative5Log);
        alloc(buffers.observations,arenas*223);alloc(buffers.actions,arenas);alloc(buffers.rewards,arenas);alloc(buffers.terminals,arenas);alloc(buffers.logs,arenas);
        float* external=nullptr;uint8_t* overrides=nullptr;alloc(external,arenas*2);alloc(overrides,arenas*2);ck(cudaMemsetAsync(overrides,1,arenas*2,stream));
        auto* runtime=rek_native5_create(&cfg,&buffers,stream);if(!runtime)throw std::runtime_error(rek_native5_error());
        rk(rek_native5_bind_external_actions(runtime,external,overrides,stream));rk(rek_native5_check_status(runtime,stream));
        cudaGraph_t graph;cudaGraphExec_t executable;ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));rk(rek_native5_step(runtime,stream));ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&executable,graph,0));
        std::array<RekNative5Snapshot,arenas> state{};std::array<Stats,arenas> stats{};unsigned moves[arenas][2]{};float actions[arenas*2];Choice choices[arenas][2];
        std::puts("{\"event\":\"exposure_identity\",\"backend\":\"mujoco_cuda\",\"controller\":\"SONIC_CUDA\",\"initial_state\":\"native_spawn\",\"intervention\":\"categorical_actions_only\",\"arenas\":4,\"ticks\":2500,\"decision_hz\":50,\"physics_substeps_per_tick\":10,\"round_seconds\":60,\"approach_range_m\":[0.65,1.05],\"bearing_tolerance_rad\":0.16,\"native_moves_in_cycle\":[6,7,8,9,0,1,2,3,4,5,10,11,12,13,14,15,16],\"forced_pose_or_impulse\":false,\"authentic_parity\":false,\"ppo_updates\":0}");
        for(int a=0;a<arenas;a++){rk(rek_native5_read_snapshot(runtime,a,&state[a],stream));record(0,a,state[a],nullptr,stats[a]);}
        for(int tick=1;tick<=ticks;tick++) {
            for(int a=0;a<arenas;a++)for(int side=0;side<2;side++) {
                choices[a][side]=choose(state[a],a,side,moves[a][side]);const int action=choices[a][side].selected;
                actions[a*2+side]=float(action);stats[a].attack_requests[side]+=action>=16;
            }
            ck(cudaMemcpyAsync(external,actions,sizeof(actions),cudaMemcpyHostToDevice,stream));ck(cudaGraphLaunch(executable,stream));
            for(int a=0;a<arenas;a++){rk(rek_native5_read_snapshot(runtime,a,&state[a],stream));record(tick,a,state[a],choices[a],stats[a]);}
            float learner_rewards[arenas];ck(cudaMemcpyAsync(learner_rewards,buffers.rewards,sizeof(learner_rewards),cudaMemcpyDeviceToHost,stream));ck(cudaStreamSynchronize(stream));
            for(int a=0;a<arenas;a++)if(learner_rewards[a]!=state[a].rewards[0])throw std::runtime_error("Learner reward buffer differs from fighter snapshot");
            rk(rek_native5_check_status(runtime,stream));completed=tick;if(tick%50==0)std::fflush(stdout);
        }
        for(int a=0;a<arenas;a++){
            const auto& m=stats[a];const auto& s=state[a];
            std::printf("{\"event\":\"physical_exposure_summary\",\"lane\":%d,\"schedule\":\"%s\",\"ticks\":%d,\"points\":[%d,%d],\"falls\":[%u,%u],\"attack_requests\":[%u,%u],\"falling_events\":[%u,%u],\"fallen_events\":[%u,%u],\"falling_samples\":[%u,%u],\"fallen_samples\":[%u,%u],\"nonfoot_samples\":[%u,%u],\"attributed_contacts\":%u,\"scored_contacts\":%u,\"ko_ticks\":%u,\"reset_ticks\":%u,\"terminal_ticks\":%u,\"min_gap_m\":%.9g,\"min_height_m\":[%.9g,%.9g],\"max_tilt_degrees\":[%.9g,%.9g],\"failure_bits\":%u}\n",a,lane_name(a),completed,s.round.points[0],s.round.points[1],s.round.falls[0],s.round.falls[1],m.attack_requests[0],m.attack_requests[1],m.falling_events[0],m.falling_events[1],m.fallen_events[0],m.fallen_events[1],m.falling_samples[0],m.falling_samples[1],m.fallen_samples[0],m.fallen_samples[1],m.nonfoot_samples[0],m.nonfoot_samples[1],m.attributed_contacts,m.scored_contacts,m.ko_ticks,m.reset_ticks,m.terminal_ticks,m.min_gap,m.min_height[0],m.min_height[1],m.max_tilt[0],m.max_tilt[1],s.round.failure_bits);
        }
        rk(rek_native5_close(runtime));ck(cudaGraphExecDestroy(executable));ck(cudaGraphDestroy(graph));
        for(void* p:{static_cast<void*>(external),static_cast<void*>(overrides),static_cast<void*>(buffers.observations),static_cast<void*>(buffers.actions),static_cast<void*>(buffers.rewards),static_cast<void*>(buffers.terminals),static_cast<void*>(buffers.logs)})ck(cudaFree(p));ck(cudaStreamDestroy(stream));
        std::puts("{\"event\":\"physical_exposure_result\",\"status\":\"passed\",\"reward_mode\":\"normalized_points_falls_v1\",\"snapshot_reward_checks\":20000,\"learner_buffer_checks\":10000,\"cpu_physics_calls\":0,\"ppo_updates\":0,\"authentic_parity\":false}");return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"physical-fall-exposure-probe failed after %d ticks: %s\n",completed,e.what());return 1;}
}
