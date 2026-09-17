// Bounded no-PPO diagnostic linked to the preserved native MuJoCo runtime.
// Host action selection and readback are diagnostic work. Physics, the SONIC
// tracking network, measurements and referee transitions execute on CUDA.
#include "runtime_api.h"
#include "g1_fight_state.h"
#include <cuda_runtime.h>
#include <mujoco/mujoco.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace {
constexpr int arenas = 4, ticks = 1000;
int forbidden_cpu_calls = 0;
void ck(cudaError_t x) { if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
void rk(int x) { if (x) throw std::runtime_error(rek_native5_error()); }
template<class T> void alloc(T*& p, size_t n) {
    ck(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)));
    ck(cudaMemset(p, 0, n * sizeof(T)));
}
struct Stats {
    float min_height[2] = {1e30f, 1e30f}, max_tilt[2] = {}, min_ratio[2] = {1e30f, 1e30f};
    float max_joint_speed[2] = {}, max_root_step[2] = {};
    unsigned attack_requests[2] = {}, ko_ticks = 0, reset_ticks = 0, fallen_samples[2] = {};
    unsigned terminal_ticks = 0;
};
const char* schedule(int a) {
    const char* names[] = {"idle_vs_idle", "stationary_attack_cycle_vs_idle",
        "approach_right_hook_move3_vs_idle", "approach_attack_cycles_both"};
    return names[a];
}
int legal_action(const RekNative5Snapshot& s, int arena, int side, unsigned& next_move) {
    const float* o = s.raw_observations + side * 223;
    const uint8_t* mask = s.action_masks + side * 33;
    int desired = 1;
    const bool attacks = arena == 3 || (arena > 0 && side == 0);
    if (attacks && o[79] == 0) {
        desired = arena==2 ? 19 : 16 + int(next_move % 17);
        if (arena >= 2) {
            const double w=o[3], x=o[4], y=o[5], z=o[6];
            const double n=w*w+x*x+y*y+z*z;
            const double fx=(w*w+x*x-y*y-z*z)/n, fy=2*(w*z+x*y)/n;
            const double dx=double(o[86])-o[0], dy=double(o[87])-o[1];
            const double distance=std::hypot(dx,dy), bearing=std::atan2(-fy*dx+fx*dy,fx*dx+fy*dy);
            if (distance > 1.05) desired=2;
            if (distance < .65) desired=3;
            if (std::abs(bearing) > .16) desired=bearing>0?6:7;
        }
    }
    // Clear held locomotion when an attack is currently masked. Continue during
    // the attack when neutral itself is masked. No action-mask bypass is used.
    if (!mask[desired]) desired=mask[1]?1:0;
    if (!mask[desired]) {
        for (int i=0;i<33;i++) if(mask[i]) { desired=i; break; }
    }
    if (!mask[desired]) throw std::runtime_error("Empty legal-action mask");
    if (desired>=16) ++next_move;
    return desired;
}
void record(int tick, int a, const RekNative5Snapshot& s,
            const RekNative5Snapshot& previous, Stats& stats) {
    const float* o=s.raw_observations;
    for(float v:s.raw_observations) if(!std::isfinite(v)) throw std::runtime_error("Nonfinite observation");
    for(float v:s.qpos) if(!std::isfinite(v)) throw std::runtime_error("Nonfinite qpos");
    for(float v:s.qvel) if(!std::isfinite(v)) throw std::runtime_error("Nonfinite qvel");
    const unsigned signals=unsigned(o[215]), referee=unsigned(o[216]);
    const bool ko=referee&(REK_G1_REFEREE_KNOCKOUT|REK_G1_REFEREE_DOUBLE_KNOCKOUT);
    const bool reset=signals&REK_G1_FIGHT_SIGNAL_RESET_BOTH_TO_SPAWN;
    stats.ko_ticks += ko; stats.reset_ticks += reset; stats.terminal_ticks += s.round.terminal!=0;
    std::printf("{\"event\":\"physical_sample\",\"tick\":%d,\"arena\":%d,\"schedule\":\"%s\",\"remaining_s\":%.9g,\"points\":[%d,%d],\"falls\":[%u,%u],\"score_delta\":[%.9g,%.9g],\"fight_signals\":%u,\"referee_calls\":%u,\"ko_event\":%s,\"reset_signal\":%s,\"failure_bits\":%u,\"terminal\":%u,\"count_elapsed_s\":%.9g,\"count_duration_s\":%.9g,\"fighters\":[",
        tick,a,schedule(a),s.round.time_remaining_seconds,s.round.points[0],s.round.points[1],
        s.round.falls[0],s.round.falls[1],o[217],o[218],signals,referee,ko?"true":"false",reset?"true":"false",
        s.round.failure_bits,unsigned(s.round.terminal),o[208],o[209]);
    for(int side=0;side<2;side++) {
        const float* f=s.raw_observations+side*223;
        float speed=0; for(int j=42;j<71;j++)speed=std::max(speed,std::abs(f[j]));
        double jump=0;for(int k=0;k<3;k++) {double d=double(s.qpos[side*36+k])-previous.qpos[side*36+k];jump+=d*d;}
        jump=std::sqrt(jump);
        stats.min_height[side]=std::min(stats.min_height[side],f[2]);
        stats.max_tilt[side]=std::max(stats.max_tilt[side],f[72]);
        stats.min_ratio[side]=std::min(stats.min_ratio[side],f[73]);
        stats.max_joint_speed[side]=std::max(stats.max_joint_speed[side],speed);
        stats.max_root_step[side]=std::max(stats.max_root_step[side],float(jump));
        stats.fallen_samples[side]+=f[79]!=0;
        std::printf("%s{\"action\":%.9g,\"root_xyz_m\":[%.9g,%.9g,%.9g],\"root_step_m\":%.9g,\"tilt_degrees\":%.9g,\"pelvis_height_ratio\":%.9g,\"both_feet_off_floor\":%.9g,\"left_foot_contact\":%.9g,\"right_foot_contact\":%.9g,\"nonfoot_floor_body_count\":%.9g,\"can_get_up\":%.9g,\"fall_phase\":%.9g,\"fallen_hold_s\":%.9g,\"fallen_elapsed_s\":%.9g,\"fallen_timer_s\":%.9g,\"reset_grace_s\":%.9g,\"fall_events\":%.9g,\"max_joint_speed_rad_s\":%.9g}",
            side?",":"",s.actions[side],f[0],f[1],f[2],jump,f[72],f[73],f[74],f[75],f[76],f[77],f[78],f[79],f[80],f[81],f[82],f[83],f[85],speed);
    }
    std::printf("]}\n");
}
}
extern "C" void __wrap_mj_step(const mjModel*,mjData*) { ++forbidden_cpu_calls;std::abort(); }
extern "C" void __wrap_mj_forward(const mjModel*,mjData*) { ++forbidden_cpu_calls;std::abort(); }
extern "C" void __wrap_mj_kinematics(const mjModel*,mjData*) { ++forbidden_cpu_calls;std::abort(); }
int main(int argc,char** argv) {
    int completed=0;
    try {
        if(argc!=7)throw std::runtime_error("Usage: physical-quality-probe XML EXPORT ASSETS FEATURES ENCODER DECODER");
        if(!std::getenv("REK_PHYSICS_BACKEND")||std::string(std::getenv("REK_PHYSICS_BACKEND"))!="mujoco_cuda")
            throw std::runtime_error("Explicit mujoco_cuda backend required");
        cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        RekNative5Config cfg{};cfg.abi_version=REK_NATIVE5_RUNTIME_ABI;cfg.arenas=arenas;cfg.seed=73;
        cfg.model_path=argv[1];cfg.physics_export_path=argv[2];cfg.assets_path=argv[3];cfg.motion_features_path=argv[4];
        cfg.controller_encoder_path=argv[5];cfg.controller_decoder_path=argv[6];cfg.locomotion_segment_ticks=1;cfg.round_seconds=20;
        const uint32_t durations[]={35,27,31,45,32,45,157,145,158,139,134,138,73,75,68,71,103};
        std::copy(durations,durations+17,cfg.move_duration_ticks);
        RekNative5Buffers buffers{};buffers.log_stride_bytes=sizeof(RekNative5Log);
        alloc(buffers.observations,arenas*223);alloc(buffers.actions,arenas);alloc(buffers.rewards,arenas);
        alloc(buffers.terminals,arenas);alloc(buffers.logs,arenas);
        float* external=nullptr;uint8_t* overrides=nullptr;alloc(external,arenas*2);alloc(overrides,arenas*2);
        ck(cudaMemsetAsync(overrides,1,arenas*2,stream));
        auto* runtime=rek_native5_create(&cfg,&buffers,stream);if(!runtime)throw std::runtime_error(rek_native5_error());
        rk(rek_native5_bind_external_actions(runtime,external,overrides,stream));rk(rek_native5_check_status(runtime,stream));
        cudaGraph_t graph;cudaGraphExec_t executable;
        ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));rk(rek_native5_step(runtime,stream));
        ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&executable,graph,0));
        std::array<RekNative5Snapshot,arenas> state{},before{};
        std::array<Stats,arenas> stats{};unsigned moves[arenas][2]{};float actions[arenas*2];
        for(int a=0;a<arenas;a++){rk(rek_native5_read_snapshot(runtime,a,&state[a],stream));record(0,a,state[a],state[a],stats[a]);}
        std::printf("{\"event\":\"probe_identity\",\"backend\":\"mujoco_cuda\",\"native_controller\":\"SONIC_encoder_decoder_CUDA\",\"arenas\":4,\"decision_ticks\":1000,\"physics_substeps_per_tick\":10,\"simulated_seconds\":20,\"round_seconds\":20,\"ppo_updates\":0,\"policy_checkpoint_loaded\":false,\"cpu_physics_calls\":0,\"authentic_parity\":false,\"diagnostic_host_action_selection_and_readback\":true}\n");
        auto start=std::chrono::steady_clock::now();
        for(int tick=1;tick<=ticks;tick++) {
            before=state;
            for(int a=0;a<arenas;a++)for(int side=0;side<2;side++) {
                int action=legal_action(state[a],a,side,moves[a][side]);actions[a*2+side]=float(action);
                stats[a].attack_requests[side]+=action>=16;
            }
            ck(cudaMemcpyAsync(external,actions,sizeof(actions),cudaMemcpyHostToDevice,stream));
            ck(cudaGraphLaunch(executable,stream));
            for(int a=0;a<arenas;a++){rk(rek_native5_read_snapshot(runtime,a,&state[a],stream));record(tick,a,state[a],before[a],stats[a]);}
            rk(rek_native5_check_status(runtime,stream));completed=tick;
            if(tick%50==0)std::fflush(stdout);
        }
        double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        for(int a=0;a<arenas;a++){
            auto& t=stats[a];auto& s=state[a];
            std::printf("{\"event\":\"physical_arena_summary\",\"arena\":%d,\"schedule\":\"%s\",\"ticks\":%d,\"wall_seconds_all_arenas\":%.9g,\"points\":[%d,%d],\"falls\":[%u,%u],\"attack_requests\":[%u,%u],\"ko_event_ticks\":%u,\"reset_signal_ticks\":%u,\"terminal_ticks\":%u,\"min_root_height_m\":[%.9g,%.9g],\"max_tilt_degrees\":[%.9g,%.9g],\"min_pelvis_height_ratio\":[%.9g,%.9g],\"max_joint_speed_rad_s\":[%.9g,%.9g],\"max_root_step_m\":[%.9g,%.9g],\"nonupright_phase_samples\":[%u,%u],\"failure_bits\":%u,\"ppo_updates\":0}\n",
                a,schedule(a),completed,wall,s.round.points[0],s.round.points[1],s.round.falls[0],s.round.falls[1],t.attack_requests[0],t.attack_requests[1],t.ko_ticks,t.reset_ticks,t.terminal_ticks,t.min_height[0],t.min_height[1],t.max_tilt[0],t.max_tilt[1],t.min_ratio[0],t.min_ratio[1],t.max_joint_speed[0],t.max_joint_speed[1],t.max_root_step[0],t.max_root_step[1],t.fallen_samples[0],t.fallen_samples[1],s.round.failure_bits);
        }
        rk(rek_native5_close(runtime));ck(cudaGraphExecDestroy(executable));ck(cudaGraphDestroy(graph));
        for(void* p:{static_cast<void*>(external),static_cast<void*>(overrides),static_cast<void*>(buffers.observations),static_cast<void*>(buffers.actions),static_cast<void*>(buffers.rewards),static_cast<void*>(buffers.terminals),static_cast<void*>(buffers.logs)})ck(cudaFree(p));
        ck(cudaStreamDestroy(stream));
        std::printf("{\"event\":\"physical_probe_result\",\"status\":\"passed\",\"ticks\":%d,\"cpu_step_forward_kinematics_calls\":%d,\"ppo_updates\":0,\"authentic_parity\":false}\n",completed,forbidden_cpu_calls);
        return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"physical-quality-probe failed after %d ticks: %s\n",completed,e.what());
        return 1;
    }
}
