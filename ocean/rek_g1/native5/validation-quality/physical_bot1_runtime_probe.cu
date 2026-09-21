// Source-grounded dispatch fixtures and real GPU physics are reported separately.
#include "runtime.cu"
#include <array>

namespace {
int checks=0;
void require_probe(bool value,const char* message){++checks;if(!value)throw std::runtime_error(message);}
void check_runtime(int status){if(status)throw std::runtime_error(rek_native5_error());}
template<class T> std::vector<T> read_device(const T* source,size_t count,cudaStream_t stream){
    std::vector<T> values(count);
    cuda_check(cudaMemcpyAsync(values.data(),source,count*sizeof(T),cudaMemcpyDeviceToHost,stream));
    cuda_check(cudaStreamSynchronize(stream));return values;
}
__global__ void idle_learner(const uint8_t* masks,float* actions,int arenas){
    int a=blockIdx.x*blockDim.x+threadIdx.x;
    if(a<arenas)actions[a]=masks[a*66+1]?1:0;
}
// Deliberately synthetic state fixtures. They exercise production kernels and
// assets, but are not evidence that any attack or physical contact caused a fall.
__global__ void binding_fixture(RuntimeView v,int mode){
    int a=blockIdx.x*blockDim.x+threadIdx.x;if(a>=v.p.arenas)return;
    auto& b=v.bot_states[a];b={};rek5_bot1::activate(b.tactical,123u+a);
    b.tactical.phase=rek5_bot1::Engaging;b.tactical.timer=1;b.tactical.min_timer=.1f;
    b.tactical.attempts=4;b.tactical.accepted=3;b.tactical.rejected=1;
    auto& c=v.c.states[a];c.combat.fight.phase=REK_G1_FIGHT_ROUND_ACTIVE;
    c.combat.fight.round_duration_seconds=120;c.combat.fight.time_remaining_seconds=115;
    c.fall[1].phase=mode==0?REK_G1_FALL_FALLEN:REK_G1_FALL_UPRIGHT;
    c.fall[1].recovery_armed=0;c.fall[0].phase=REK_G1_FALL_UPRIGHT;
    v.c.episode_reset[a]=0;v.c.begin[a]=0;v.c.complete[a]=0;v.p.time[a]=10;
    for(int side=0;side<2;side++){
        const int row=a*2+side;v.robot_dampened[row]=0;v.robot_resetting[row]=0;
        float* pose=v.p.qpos+a*72+side*36;
        pose[0]=side?0:.55f;pose[1]=0;pose[3]=1;pose[4]=pose[5]=pose[6]=0;
        v.actions[row]=side?0:(v.masks[row*33+1]?1:0);
    }
    if(mode==1){b.tactical.phase=rek5_bot1::Settling;b.tactical.timer=-.01f;v.robot_resetting[a*2+1]=1;}
    if(mode==2)v.c.episode_reset[a]=1;
}
__global__ void mark_counted_reset(RuntimeView v){
    int a=blockIdx.x*blockDim.x+threadIdx.x;if(a<v.p.arenas){v.c.begin[a]=0;v.c.complete[a]=1;}
}
void binding_tests(RekNative5Runtime* runtime,cudaStream_t stream){
    auto& v=runtime->view;const int arenas=v.p.arenas;
    auto catalog=read_device(v.bot_catalog,1,stream);
    auto routes=read_device(v.bot_move_routes,17,stream);
    auto offsets=read_device(v.c.route_offsets,24,stream),counts=read_device(v.c.route_counts,24,stream);
    auto impacts=read_device(v.c.impacts,rek_g1_cuda_native_combat_impact_event_count(),stream);
    for(int move=0;move<17;move++){
        int limb=0;const int route=routes[move];require_probe(route>=7&&route<24,"assigned route out of range");
        for(int k=0;k<counts[route];k++)if(impacts[offsets[route]+k].limb){limb=impacts[offsets[route]+k].limb;break;}
        require_probe(limb>=1&&limb<=4&&catalog[0].primary_limb[move]==limb,"assigned limb catalog mismatch");
    }
    binding_fixture<<<1,128,0,stream>>>(v,0);prepare_bot<<<1,128,0,stream>>>(v);
    auto fallen=read_device(v.bot_states,arenas,stream);
    auto dampen=read_device(v.bot_dampened,arenas*2,stream);
    auto commands=read_device(v.bot_commands,arenas*2,stream);
    for(int a=0;a<arenas;a++){
        const auto& b=fallen[a];require_probe(b.tactical.phase==rek5_bot1::Engaging&&b.tactical.rng==123u+a,"fall changed tactical phase/RNG");
        require_probe(std::abs(b.tactical.timer-.98f)<1e-6f&&b.tactical.attempts==4,"fall timer/counters mismatch");
        require_probe(!b.recovery.straighten_issued&&dampen[a*2+1]==1&&commands[a*2+1].move_index==-1,"G1 fall dispatch mismatch");
    }
    mark_counted_reset<<<1,128,0,stream>>>(v);counted_reset<<<1,128,0,stream>>>(v);
    auto reset=read_device(v.bot_states,arenas,stream);
    require_probe(std::memcmp(reset.data(),fallen.data(),arenas*sizeof(PhysicalBotState))==0,"counted reset changed bot state");
    check_runtime(rek_native5_reset(runtime,stream));
    binding_fixture<<<1,128,0,stream>>>(v,1);flags<<<1,128,0,stream>>>(v,false);prepare_bot<<<1,128,0,stream>>>(v);
    runtime->motion->pre_direct(v.actions,v.bot_commands,v.bot_enabled,v.bot_results,v.local_velocity,v.suspended,stream);
    finish_bot<<<1,128,0,stream>>>(v);
    auto results=read_device(v.bot_results,arenas*2,stream);auto accepted=read_device(v.bot_states,arenas,stream);
    auto composers=read_device(v.bot_composers,arenas*2,stream);
    for(int a=0;a<arenas;a++){
        const auto& r=results[a*2+1];require_probe(r.status==0&&r.suspended&&r.move_attempted&&r.move_accepted&&!r.move_rejected,"suspended logical move incorrectly rejected");
        require_probe(composers[a*2+1].action_playing&&accepted[a].tactical.phase==rek5_bot1::Attacking&&accepted[a].tactical.accepted==4,"accepted move feedback missing");
        require_probe(results[a*2].status==0&&!results[a*2].move_attempted,"direct command reached learner");
    }
    check_runtime(rek_native5_check_status(runtime,stream));
    check_runtime(rek_native5_reset(runtime,stream));
    binding_fixture<<<1,128,0,stream>>>(v,2);prepare_bot<<<1,128,0,stream>>>(v);
    auto rounds=read_device(v.bot_states,arenas,stream);
    for(int a=0;a<arenas;a++){
        require_probe(rounds[a].tactical.phase==rek5_bot1::Settling&&rounds[a].tactical.rng==123u+a,"round activation phase/RNG mismatch");
        require_probe(std::abs(rounds[a].tactical.timer-(rek5_bot1::initial_delay-.02f))<1e-6f,"round activation delay mismatch");
    }
    check_runtime(rek_native5_reset(runtime,stream));
    std::printf("{\"event\":\"physical_bot1_binding_fixtures\",\"passed\":true,\"checks\":%d,\"synthetic_dispatch_fixtures\":true,\"physical_contact_claim\":false}\n",checks);std::fflush(stdout);
}
void integration(int argc,char** argv){
    require_probe(argc==7,"usage: physical-bot1-runtime-probe XML EXPORT ASSETS FEATURES ENCODER DECODER");
    constexpr int arenas=4,ticks=1200;
    cudaStream_t stream;cuda_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    rek5::DeviceStorage storage;RekNative5Config config{};
    config.abi_version=REK_NATIVE5_RUNTIME_ABI;config.arenas=arenas;config.seed=73;config.locomotion_segment_ticks=1;config.round_seconds=10;
    config.model_path=argv[1];config.physics_export_path=argv[2];config.assets_path=argv[3];config.motion_features_path=argv[4];config.controller_encoder_path=argv[5];config.controller_decoder_path=argv[6];
    const uint32_t durations[17]={35,27,31,45,32,45,157,145,158,139,134,138,73,75,68,71,103};std::copy(durations,durations+17,config.move_duration_ticks);
    RekNative5Buffers buffers{};buffers.observations=storage.alloc<float>(arenas*223);buffers.actions=storage.alloc<float>(arenas);buffers.rewards=storage.alloc<float>(arenas);buffers.terminals=storage.alloc<float>(arenas);buffers.logs=storage.alloc<RekNative5Log>(arenas);buffers.log_stride_bytes=sizeof(RekNative5Log);
    auto* runtime=rek_native5_create(&config,&buffers,stream);require_probe(runtime,rek_native5_error());
    require_probe(runtime->view.recovered_bot&&runtime->view.observable_balance&&runtime->view.normalized_rewards,"probe modes missing");
    binding_tests(runtime,stream);
    cudaGraph_t graph;cudaGraphExec_t executable;
    cuda_check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    idle_learner<<<1,128,0,stream>>>(runtime->view.masks,buffers.actions,arenas);
    check_runtime(rek_native5_step(runtime,stream));
    cuda_check(cudaStreamEndCapture(stream,&graph));cuda_check(cudaGraphInstantiate(&executable,graph,0));
    unsigned attempts=0,accepted=0,rejected=0,moving=0,terminals=0,body_resets=0;
    for(int tick=0;tick<ticks;tick++){
        cuda_check(cudaGraphLaunch(executable,stream));
        auto bots=read_device(runtime->view.bot_states,arenas,stream);
        auto result=read_device(runtime->view.bot_results,arenas*2,stream);
        auto completed=read_device(runtime->view.completed,arenas*2,stream);
        auto terminal=read_device(buffers.terminals,arenas,stream);
        auto rewards=read_device(buffers.rewards,arenas,stream);
        for(int a=0;a<arenas;a++){
            const auto& r=result[a*2+1];require_probe(r.status==0,"direct result failure");
            require_probe(!r.move_attempted||(r.move_accepted!=r.move_rejected),"move result ambiguous");
            attempts+=r.move_attempted;accepted+=r.move_accepted;rejected+=r.move_rejected;
            moving+=bots[a].velocity.forward!=0||bots[a].velocity.strafe!=0||bots[a].velocity.yaw!=0;
            terminals+=terminal[a]!=0;body_resets+=completed[a*2]!=0||completed[a*2+1]!=0;
            require_probe(std::isfinite(rewards[a])&&rewards[a]>=-1&&rewards[a]<=1,"normalized reward bounds");
        }
        if(tick%200==199){check_runtime(rek_native5_check_status(runtime,stream));std::printf("{\"event\":\"physical_bot1_progress\",\"ticks\":%d,\"attempts\":%u,\"accepted\":%u,\"rejected\":%u}\n",tick+1,attempts,accepted,rejected);std::fflush(stdout);}
    }
    require_probe(attempts>0&&accepted>0&&moving>0,"no actual bot movement or accepted attack");
    require_probe(terminals>=arenas,"no actual round boundary");
    auto falls=read_device(runtime->view.confirmed_falls,arenas*2,stream);
    auto points=read_device(runtime->view.awarded_points,arenas*2,stream);
    uint64_t total_falls[2]={};int64_t total_points[2]={};
    for(int a=0;a<arenas;a++)for(int side=0;side<2;side++){total_falls[side]+=falls[a*2+side];total_points[side]+=points[a*2+side];}
    check_runtime(rek_native5_check_status(runtime,stream));
    cuda_check(cudaGraphExecDestroy(executable));cuda_check(cudaGraphDestroy(graph));
    check_runtime(rek_native5_close(runtime));cuda_check(cudaStreamDestroy(stream));
    std::printf("{\"event\":\"physical_bot1_integration\",\"passed\":true,\"arenas\":4,\"ticks\":1200,\"diagnostic_round_seconds\":10,\"graph_replays\":1200,\"attempts\":%u,\"accepted\":%u,\"rejected\":%u,\"moving_bot_ticks\":%u,\"terminal_transitions\":%u,\"counted_reset_ticks\":%u,\"confirmed_falls\":[%llu,%llu],\"awarded_points\":[%lld,%lld],\"cpu_physics_calls\":0,\"ppo_updates\":0,\"authentic_parity\":false,\"performance_claim\":false}\n",attempts,accepted,rejected,moving,terminals,body_resets,(unsigned long long)total_falls[0],(unsigned long long)total_falls[1],(long long)total_points[0],(long long)total_points[1]);
}
}
extern "C" void __wrap_mj_step(const mjModel*,mjData*){std::abort();}
extern "C" void __wrap_mj_step1(const mjModel*,mjData*){std::abort();}
extern "C" void __wrap_mj_step2(const mjModel*,mjData*){std::abort();}
extern "C" void __wrap_mj_forward(const mjModel*,mjData*){std::abort();}
extern "C" void __wrap_mj_kinematics(const mjModel*,mjData*){std::abort();}
int main(int argc,char** argv){try{integration(argc,argv);return 0;}catch(const std::exception& e){std::fprintf(stderr,"physical Bot1 probe: %s\n",e.what());return 1;}}
