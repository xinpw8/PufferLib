// Separate declared move7 LEFT_FRONT versus move3 RIGHT_HOOK candidate schedule.
// Candidate behavior check from native reset. No authentic state restoration.
// Compile REK_SCHEDULE_CPU_ONLY for schedule inspection without CUDA linkage.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace schedule {
constexpr int arenas=4, warmup=50, yaw_lead=25, first_attack=76;
constexpr int ticks=270, substeps=10;
constexpr int durations[17]={35,27,31,45,32,45,157,145,158,139,134,138,73,75,68,71,103};
constexpr int registry[17]={6,7,8,9,0,1,2,3,4,5,10,11,12,13,14,15,16};
int move(int lane) { return lane<2?7:3; }
bool attack(int lane) { return lane%2==0; }
int held(int yaw) { return yaw>0?6:yaw<0?7:1; }
int category(int runtime_move) {
    for(int i=0;i<17;i++) if(registry[i]==runtime_move)return 16+i;
    throw std::runtime_error("Unsupported runtime move");
}
int action(int tick,int lane,int side,int yaw) {
    if(tick<1||tick>ticks||lane<0||lane>=arenas||side<0||side>1||yaw<-1||yaw>1)
        throw std::runtime_error("Invalid schedule coordinate");
    if(side||tick<=warmup)return 1;
    if(tick<first_attack)return held(yaw);
    if(tick>=first_attack+durations[move(lane)])return 1;
    if(!attack(lane))return held(yaw);
    return tick==first_attack?category(move(lane)):0;
}
void identity() {
    std::printf("{\"event\":\"schedule_identity\",\"classification\":\"candidate_from_reset_behavior_check\",\"authentic_replay\":false,\"authentic_parity\":false,\"arenas\":4,\"yaw_conditions\":[-1,0,1],\"decision_hz\":50,\"physical_step_s\":0.002,\"substeps\":10,\"warmup_ticks\":50,\"yaw_lead_ticks\":25,\"first_attack_tick\":76,\"ticks_per_reset\":270,\"moves\":[7,3],\"categories\":[17,23],\"move_duration_ticks\":[145,45],\"opponent_action\":1,\"actor_lanes\":[\"move7_left_front_attack\",\"move7_left_front_no_attack_control\",\"move3_right_hook_attack\",\"move3_right_hook_no_attack_control\"],\"post_attack_category\":0,\"post_duration_category\":1,\"unknowns\":[\"authentic_full_qvel\",\"authentic_SONIC_history\",\"authentic_clip_phase\",\"server_action_acceptance_and_timing\",\"opponent_action_history\"]}\n");
}
void self_test() {
    int checks=0;
    auto check=[&](bool ok){if(!ok)throw std::runtime_error("Schedule self-test failed");++checks;};
    check(category(7)==17);check(category(3)==23);check(registry[19-16]==9);
    for(int yaw=-1;yaw<=1;yaw++)for(int lane=0;lane<arenas;lane++) {
        int requests=0,continues=0;
        for(int t=1;t<=ticks;t++) {
            const int a=action(t,lane,0,yaw);requests+=a>=16;continues+=a==0;
            check(action(t,lane,1,yaw)==1);
            if(t<=warmup)check(a==1);
        }
        check(requests==(attack(lane)?1:0));
        check(continues==(attack(lane)?durations[move(lane)]-1:0));
        check(action(75,lane,0,yaw)==held(yaw));
        check(action(first_attack+durations[move(lane)],lane,0,yaw)==1);
    }
    bool rejected=false;try{action(0,0,0,0);}catch(...){rejected=true;}check(rejected);
    std::printf("{\"event\":\"schedule_cpu_tests\",\"checks\":%d,\"passed\":true,\"cuda_initialized\":false}\n",checks);
}
}

#ifndef REK_SCHEDULE_CPU_ONLY
#include "runtime_api.h"
#include "g1_fight_state.h"
#include <cuda_runtime.h>
#include <mujoco/mujoco.h>
namespace {
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void rk(int e){if(e)throw std::runtime_error(rek_native5_error());}
template<class T> void allocate(T*& p,size_t n){ck(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));ck(cudaMemset(p,0,n*sizeof(T)));}
double heading(const float* q){return std::atan2(2*(double(q[0])*q[3]+double(q[1])*q[2]),1-2*(double(q[2])*q[2]+double(q[3])*q[3]));}
struct Metrics {
    std::array<float,72> origin{};
    double path=0,net=0,yaw=0,min_gap=1e30,min_height=1e30,max_tilt=0,max_joint_speed=0;
    int busy_samples=0,fall_samples=0,nonfoot_samples=0,terminal_samples=0,reset_samples=0;
};
void record(int yaw,int tick,int lane,const RekNative5Snapshot& s,const RekNative5Snapshot& previous,Metrics& m) {
    for(float v:s.qpos)if(!std::isfinite(v))throw std::runtime_error("Nonfinite qpos");
    for(float v:s.qvel)if(!std::isfinite(v))throw std::runtime_error("Nonfinite qvel");
    for(float v:s.raw_observations)if(!std::isfinite(v))throw std::runtime_error("Nonfinite observation");
    const float* o=s.raw_observations;
    m.terminal_samples+=s.round.terminal!=0;
    const unsigned fight_signals=unsigned(o[215]),referee_calls=unsigned(o[216]);
    m.reset_samples+=(fight_signals&REK_G1_FIGHT_SIGNAL_RESET_BOTH_TO_SPAWN)!=0;
    double gap=std::hypot(double(s.qpos[0])-s.qpos[36],double(s.qpos[1])-s.qpos[37]);
    if(tick==schedule::first_attack-1)std::copy(s.qpos,s.qpos+72,m.origin.begin());
    if(tick>=schedule::first_attack&&tick<schedule::first_attack+schedule::durations[schedule::move(lane)]) {
        m.path+=std::hypot(double(s.qpos[0])-previous.qpos[0],double(s.qpos[1])-previous.qpos[1]);
        m.net=std::hypot(double(s.qpos[0])-m.origin[0],double(s.qpos[1])-m.origin[1]);
        m.yaw+=std::remainder(heading(s.qpos+3)-heading(previous.qpos+3),2*std::acos(-1.));
        m.min_gap=std::min(m.min_gap,gap);m.min_height=std::min(m.min_height,double(o[2]));m.max_tilt=std::max(m.max_tilt,double(o[72]));
        for(int j=42;j<71;j++)m.max_joint_speed=std::max(m.max_joint_speed,std::abs(double(o[j])));
        m.busy_samples+=o[183]!=0;m.fall_samples+=o[79]!=0;m.nonfoot_samples+=o[77]>0;
    }
    std::printf("{\"event\":\"physical_schedule_sample\",\"yaw_condition\":%d,\"tick\":%d,\"time_s\":%.9g,\"lane\":%d,\"planned_move\":%d,\"attack_lane\":%s,\"actor_action\":%.9g,\"gap_m\":%.9g,\"points\":[%d,%d],\"falls\":[%u,%u],\"failure_bits\":%u,\"terminal\":%u,\"fighters\":[",yaw,tick,tick*.02,lane,schedule::move(lane),schedule::attack(lane)?"true":"false",s.actions[0],gap,s.round.points[0],s.round.points[1],s.round.falls[0],s.round.falls[1],s.round.failure_bits,unsigned(s.round.terminal));
    for(int side=0;side<2;side++) {
        const float* f=o+side*223;const float* q=s.qpos+side*36;const float* v=s.qvel+side*35;
        std::printf("%s{\"root_xyz_m\":[%.9g,%.9g,%.9g],\"root_wxyz\":[%.9g,%.9g,%.9g,%.9g],\"root_qvel_model_order\":[%.9g,%.9g,%.9g,%.9g,%.9g,%.9g],\"physical_heading_rad\":%.9g,\"logical_heading_rad\":%.9g,\"effective_yaw\":%.9g,\"route\":%.9g,\"busy\":%.9g,\"tilt_degrees\":%.9g,\"fall_phase\":%.9g,\"foot_contacts\":[%.9g,%.9g],\"nonfoot_floor_bodies\":%.9g}",side?",":"",q[0],q[1],q[2],q[3],q[4],q[5],q[6],v[0],v[1],v[2],v[3],v[4],v[5],heading(q+3),2*std::atan2(double(f[175]),double(f[172])),f[178],f[179],f[183],f[72],f[79],f[75],f[76],f[77]);
    }
    std::printf("],\"fight_signals\":%u,\"referee_calls\":%u,\"reset_both_to_spawn\":%s}\n",fight_signals,referee_calls,(fight_signals&REK_G1_FIGHT_SIGNAL_RESET_BOTH_TO_SPAWN)?"true":"false");
}
int run(char** argv) {
    const char* backend=std::getenv("REK_PHYSICS_BACKEND");
    if(!backend||std::string(backend)!="mujoco_cuda")throw std::runtime_error("Explicit mujoco_cuda required");
    cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    RekNative5Config c{};c.abi_version=REK_NATIVE5_RUNTIME_ABI;c.arenas=4;c.seed=73;c.locomotion_segment_ticks=1;c.round_seconds=20;
    c.model_path=argv[2];c.physics_export_path=argv[3];c.assets_path=argv[4];c.motion_features_path=argv[5];c.controller_encoder_path=argv[6];c.controller_decoder_path=argv[7];
    for(int i=0;i<17;i++)c.move_duration_ticks[i]=schedule::durations[i];
    RekNative5Buffers b{};b.log_stride_bytes=sizeof(RekNative5Log);
    allocate(b.observations,4*223);allocate(b.actions,4);allocate(b.rewards,4);allocate(b.terminals,4);allocate(b.logs,4);
    float* external=nullptr;uint8_t* overrides=nullptr;allocate(external,8);allocate(overrides,8);ck(cudaMemsetAsync(overrides,1,8,stream));
    auto* runtime=rek_native5_create(&c,&b,stream);if(!runtime)throw std::runtime_error(rek_native5_error());
    rk(rek_native5_bind_external_actions(runtime,external,overrides,stream));rk(rek_native5_check_status(runtime,stream));
    cudaGraph_t graph;cudaGraphExec_t executable;ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));rk(rek_native5_step(runtime,stream));ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&executable,graph,0));
    schedule::identity();
    for(int yaw=-1;yaw<=1;yaw++) {
        rk(rek_native5_reset(runtime,stream));rk(rek_native5_check_status(runtime,stream));
        std::array<RekNative5Snapshot,4> s{},previous{};std::array<Metrics,4> metrics{};
        for(int lane=0;lane<4;lane++){rk(rek_native5_read_snapshot(runtime,lane,&s[lane],stream));record(yaw,0,lane,s[lane],s[lane],metrics[lane]);}
        for(int tick=1;tick<=schedule::ticks;tick++) {
            float a[8];previous=s;
            for(int lane=0;lane<4;lane++)for(int side=0;side<2;side++) {
                int chosen=schedule::action(tick,lane,side,yaw);
                if(!s[lane].action_masks[side*33+chosen]) {
                    std::fprintf(stderr,"Masked fixed schedule yaw=%d tick=%d lane=%d side=%d action=%d\n",yaw,tick,lane,side,chosen);
                    throw std::runtime_error("Fixed action unsupported; no fallback or retry");
                }
                a[lane*2+side]=float(chosen);
            }
            ck(cudaMemcpyAsync(external,a,sizeof(a),cudaMemcpyHostToDevice,stream));ck(cudaGraphLaunch(executable,stream));
            for(int lane=0;lane<4;lane++){rk(rek_native5_read_snapshot(runtime,lane,&s[lane],stream));record(yaw,tick,lane,s[lane],previous[lane],metrics[lane]);}
            rk(rek_native5_check_status(runtime,stream));
            if(tick==75)for(int pair=0;pair<2;pair++) {
                double dq=0,dv=0;for(int k=0;k<72;k++)dq=std::max(dq,std::abs(double(s[pair*2].qpos[k])-s[pair*2+1].qpos[k]));for(int k=0;k<70;k++)dv=std::max(dv,std::abs(double(s[pair*2].qvel[k])-s[pair*2+1].qvel[k]));
                std::printf("{\"event\":\"pre_action_pair_difference\",\"yaw_condition\":%d,\"move\":%d,\"max_qpos_difference\":%.9g,\"max_qvel_difference\":%.9g}\n",yaw,pair==0?7:3,dq,dv);
            }
            if(tick%50==0)std::fflush(stdout);
        }
        for(int lane=0;lane<4;lane++) {const auto& m=metrics[lane];
            std::printf("{\"event\":\"physical_schedule_summary\",\"yaw_condition\":%d,\"lane\":%d,\"move\":%d,\"attack_lane\":%s,\"window_start_pre_action_tick\":75,\"window_end_tick\":%d,\"nominal_duration_s\":%.9g,\"net_xy_m\":%.9g,\"path_xy_m\":%.9g,\"net_physical_yaw_rad\":%.9g,\"min_gap_m\":%.9g,\"min_root_height_m\":%.9g,\"max_tilt_degrees\":%.9g,\"max_joint_speed_rad_s\":%.9g,\"busy_samples\":%d,\"fall_samples\":%d,\"nonfoot_floor_samples\":%d,\"authentic_parity\":false}\n",yaw,lane,schedule::move(lane),schedule::attack(lane)?"true":"false",75+schedule::durations[schedule::move(lane)],schedule::durations[schedule::move(lane)]*.02,m.net,m.path,m.yaw,m.min_gap,m.min_height,m.max_tilt,m.max_joint_speed,m.busy_samples,m.fall_samples,m.nonfoot_samples);
            std::printf("{\"event\":\"whole_schedule_outcome_check\",\"yaw_condition\":%d,\"lane\":%d,\"terminal_samples\":%d,\"reset_samples\":%d,\"failure_bits\":%u,\"window_confounded_by_terminal_or_reset\":%s}\n",yaw,lane,m.terminal_samples,m.reset_samples,s[lane].round.failure_bits,(m.terminal_samples||m.reset_samples)?"true":"false");
        }
    }
    rk(rek_native5_close(runtime));ck(cudaGraphExecDestroy(executable));ck(cudaGraphDestroy(graph));
    for(void* p:{static_cast<void*>(external),static_cast<void*>(overrides),static_cast<void*>(b.observations),static_cast<void*>(b.actions),static_cast<void*>(b.rewards),static_cast<void*>(b.terminals),static_cast<void*>(b.logs)})ck(cudaFree(p));ck(cudaStreamDestroy(stream));
    std::printf("{\"event\":\"physical_schedule_result\",\"status\":\"passed\",\"cpu_physics_calls\":0,\"ppo_updates\":0,\"authentic_parity\":false}\n");return 0;
}
}
extern "C" void __wrap_mj_step(const mjModel*,mjData*){std::abort();}
extern "C" void __wrap_mj_forward(const mjModel*,mjData*){std::abort();}
extern "C" void __wrap_mj_kinematics(const mjModel*,mjData*){std::abort();}
#endif

int main(int argc,char** argv) {
    try {
        if(argc==2&&std::string(argv[1])=="--self-test"){schedule::self_test();return 0;}
        if(argc==2&&std::string(argv[1])=="--schedule-only") {
            schedule::identity();for(int yaw=-1;yaw<=1;yaw++)for(int t=1;t<=schedule::ticks;t++) {
                std::printf("{\"yaw\":%d,\"tick\":%d,\"actions\":[",yaw,t);
                for(int lane=0;lane<4;lane++)std::printf("%s%d,1",lane?",":"",schedule::action(t,lane,0,yaw));std::printf("]}\n");
            }return 0;
        }
#ifndef REK_SCHEDULE_CPU_ONLY
        if(argc==8&&std::string(argv[1])=="--run")return run(argv);
#endif
        throw std::runtime_error("Usage: physical-schedule-probe --self-test | --schedule-only | --run XML EXPORT ASSETS FEATURES ENCODER DECODER");
    }catch(const std::exception& e){std::fprintf(stderr,"physical-schedule-probe: %s\n",e.what());return 1;}
}
