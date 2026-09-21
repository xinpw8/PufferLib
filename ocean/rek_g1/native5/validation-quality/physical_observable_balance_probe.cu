// Tests the actual adapter implementation, then the public runtime boundary.
// Synthetic adapter fixtures are not physical fall or combat evidence.
#include "runtime.cu"
#include <array>

namespace {
int checks=0;
void require_probe(bool value,const char* message){++checks;if(!value)throw std::runtime_error(message);}
void near_probe(float actual,float expected){require_probe(std::isfinite(actual)&&std::abs(double(actual)-expected)<2e-5,"projection mismatch");}
RekG1FightState fixture_fight(){
    RekG1FightState fight{};fight.phase=REK_G1_FIGHT_ROUND_ACTIVE;
    fight.round_duration_seconds=fight.time_remaining_seconds=120;return fight;
}
std::array<float,72> fixture_pose(){
    std::array<float,72> pose{};pose[2]=pose[38]=.7f;pose[3]=pose[39]=1;pose[36]=1;
    for(int side=0;side<2;side++)for(int k=7;k<36;k++)pose[side*36+k]=float(k);
    return pose;
}
void cpu_adapter_test(){
    auto pose=fixture_pose();auto fight=fixture_fight();ObservableHistory history{};float out[446];
    require_probe(project_observable(history,pose.data(),fight,false,false,false,out)==rek_observable_balance::kOk,"initial adapter failed");
    require_probe(out[203]==0&&out[426]==0&&history.ticks==0,"initial history present");
    for(int actor=0;actor<2;actor++)for(int side=0;side<2;side++){
        const int base=actor*223+side*86;
        for(int k=13;k<=70;k++)require_probe(out[base+k]==0,"unproven joints exposed");
        require_probe(out[base+74]==0&&out[base+75]==0,"joint availability incorrect");
    }
    pose[0]+=.02f;pose[2]+=.01f;fight.clean_hits[0]=2;fight.clean_hits[1]=5;fight.count_active[1]=1;
    require_probe(project_observable(history,pose.data(),fight,false,true,false,out)==rek_observable_balance::kOk,"second adapter failed");
    near_probe(out[7],1);near_probe(out[9],.5f);
    require_probe(out[203]==1&&out[426]==1&&out[204]==0&&out[205]==1&&out[427]==1&&out[428]==0,"history/count perspectives failed");
    require_probe(out[217]==2&&out[218]==5&&out[440]==5&&out[441]==2,"point perspectives failed");
    // Same-round body-reset-like displacement must retain observed history.
    pose[0]+=.4f;
    require_probe(project_observable(history,pose.data(),fight,false,true,false,out)==rek_observable_balance::kOk,"body displacement failed");
    near_probe(out[7],20);require_probe(out[203]==1,"body displacement cleared history");
    require_probe(project_observable(history,pose.data(),fight,true,true,false,out)==rek_observable_balance::kOk&&out[185]==4,"terminal phase failed");
    const auto key=history.round_key;fight=fixture_fight();pose=fixture_pose();
    require_probe(project_observable(history,pose.data(),fight,false,true,true,out)==rek_observable_balance::kOk,"new round failed");
    require_probe(history.round_key==key+1&&out[203]==0&&out[217]==0&&out[7]==0,"new round leaked history");
    require_probe(project_observable(history,pose.data(),fight,false,true,false,out)==rek_observable_balance::kOk&&out[203]==1,"new round history failed");
    require_probe(project_observable(history,pose.data(),fight,false,false,false,out)==rek_observable_balance::kOk&&out[203]==0&&history.ticks==0,"explicit reset failed");
    fight.clean_hits[0]=2;require_probe(project_observable(history,pose.data(),fight,false,true,false,out)==rek_observable_balance::kOk,"score fixture failed");
    fight.clean_hits[0]=1;require_probe(project_observable(history,pose.data(),fight,false,true,false,out)==rek_observable_balance::kCounterRegressed,"score regression accepted");
    require_probe(!history.available,"invalid observation retained history");
    for(float value:out)require_probe(value==0,"failed projection not discarded");
    std::printf("{\"event\":\"cpu_adapter_test\",\"passed\":true,\"checks\":%d,\"cuda_calls\":0,\"synthetic_adapter_fixtures\":true}\n",checks);
}
void gpu_adapter_test(cudaStream_t stream){
    constexpr int count=8;rek5::DeviceStorage storage;RuntimeView view{};view.p.arenas=count;
    std::vector<float> poses(count*72),expected(count*446),actual(count*446);
    std::vector<RekG1CudaNativeCombatState> combat(count);std::vector<ObservableHistory> history(count);
    std::vector<uint8_t> reset(count);std::vector<float> terminal(count*2);
    for(int a=0;a<count;a++){
        auto pose=fixture_pose();auto fight=fixture_fight();float initial[446];
        require_probe(project_observable(history[a],pose.data(),fight,false,false,false,initial)==rek_observable_balance::kOk,"fixture initialization failed");
        pose[0]+=.02f*(a+1);pose[2]+=.001f*a;fight.clean_hits[0]=a;fight.clean_hits[1]=a*2;
        fight.count_active[0]=a%2;fight.count_active[1]=(a/2)%2;
        std::copy(pose.begin(),pose.end(),poses.begin()+a*72);combat[a].combat.fight=fight;
        reset[a]=a==0;terminal[a*2]=a==1;
    }
    view.p.qpos=storage.upload(poses);view.c.states=storage.upload(combat);view.c.episode_reset=storage.upload(reset);
    view.c.terminals=storage.upload(terminal);view.observable_history=storage.upload(history);
    view.observable_observations=storage.alloc<float>(count*446);view.failures=storage.alloc<int>(count);
    cudaGraph_t graph;cudaGraphExec_t executable;
    cuda_check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    observable_pack<<<1,128,0,stream>>>(view,true);
    cuda_check(cudaStreamEndCapture(stream,&graph));cuda_check(cudaGraphInstantiate(&executable,graph,0));
    double maximum_error=0;
    for(int repeat=0;repeat<3;repeat++){
        for(int a=0;a<count;a++)require_probe(project_observable(history[a],poses.data()+a*72,combat[a].combat.fight,terminal[a*2]!=0,true,reset[a]!=0,expected.data()+a*446)==rek_observable_balance::kOk,"CPU reference failed");
        cuda_check(cudaGraphLaunch(executable,stream));
        cuda_check(cudaMemcpyAsync(actual.data(),view.observable_observations,actual.size()*sizeof(float),cudaMemcpyDeviceToHost,stream));cuda_check(cudaStreamSynchronize(stream));
        for(size_t i=0;i<actual.size();i++){near_probe(actual[i],expected[i]);maximum_error=std::max(maximum_error,std::abs(double(actual[i])-expected[i]));}
    }
    int failures[count];cuda_check(cudaMemcpy(failures,view.failures,sizeof(failures),cudaMemcpyDeviceToHost));for(int failure:failures)require_probe(failure==0,"CUDA adapter failure");
    cuda_check(cudaGraphExecDestroy(executable));cuda_check(cudaGraphDestroy(graph));
    std::printf("{\"event\":\"cuda_adapter_test\",\"passed\":true,\"features_compared\":10704,\"graph_replays\":3,\"maximum_error\":%.9g,\"synthetic_adapter_fixtures\":true}\n",maximum_error);
}
void check_runtime(int status){if(status)throw std::runtime_error(rek_native5_error());}
__global__ void idle_actions(const uint8_t* mask,float* actions,int rows){int row=blockIdx.x*blockDim.x+threadIdx.x;if(row<rows)actions[row]=mask[row*33+1]?1:0;}
void integration(int argc,char** argv){
    require_probe(argc==7,"usage: physical-observable-balance-probe XML EXPORT ASSETS FEATURES ENCODER DECODER");
    constexpr int arenas=4,ticks=120;
    const bool observable=getenv("REK_OBSERVATION_SCHEMA")&&std::string(getenv("REK_OBSERVATION_SCHEMA"))==rek_observable_balance::kSchema;
    cudaStream_t stream;cuda_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    gpu_adapter_test(stream);
    rek5::DeviceStorage storage;RekNative5Config config{};config.abi_version=REK_NATIVE5_RUNTIME_ABI;config.arenas=arenas;config.seed=73;config.locomotion_segment_ticks=1;config.round_seconds=1;
    config.model_path=argv[1];config.physics_export_path=argv[2];config.assets_path=argv[3];config.motion_features_path=argv[4];config.controller_encoder_path=argv[5];config.controller_decoder_path=argv[6];
    const uint32_t durations[17]={35,27,31,45,32,45,157,145,158,139,134,138,73,75,68,71,103};std::copy(durations,durations+17,config.move_duration_ticks);
    RekNative5Buffers buffers{};buffers.observations=storage.alloc<float>(arenas*223);buffers.actions=storage.alloc<float>(arenas);buffers.rewards=storage.alloc<float>(arenas);buffers.terminals=storage.alloc<float>(arenas);buffers.logs=storage.alloc<RekNative5Log>(arenas);buffers.log_stride_bytes=sizeof(RekNative5Log);
    auto* actions=storage.alloc<float>(arenas*2);auto* overrides=storage.alloc<uint8_t>(arenas*2);cuda_check(cudaMemsetAsync(overrides,1,arenas*2,stream));
    auto* encoded=storage.alloc<float>(arenas*446);auto* repeated=storage.alloc<float>(arenas*446);
    auto* runtime=rek_native5_create(&config,&buffers,stream);require_probe(runtime,rek_native5_error());
    check_runtime(rek_native5_bind_external_actions(runtime,actions,overrides,stream));RekNative5DeviceView view{};check_runtime(rek_native5_get_device_view(runtime,&view));
    cudaGraph_t graph;cudaGraphExec_t executable;cuda_check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    idle_actions<<<1,128,0,stream>>>(view.action_masks,actions,arenas*2);check_runtime(rek_native5_step(runtime,stream));
    cuda_check(cudaStreamEndCapture(stream,&graph));cuda_check(cudaGraphInstantiate(&executable,graph,0));
    std::array<ObservableHistory,arenas> history{};std::array<bool,arenas> previous_terminal{};
    unsigned new_rounds=0,terminals=0;double maximum_error=0;
    for(int tick=0;tick<=ticks;tick++){
        if(tick)cuda_check(cudaGraphLaunch(executable,stream));
        check_runtime(rek_native5_encode_fighter_observations(runtime,encoded,stream));check_runtime(rek_native5_encode_fighter_observations(runtime,repeated,stream));
        std::array<float,arenas*446> actual,again;std::array<float,arenas*223> learner;
        cuda_check(cudaMemcpyAsync(actual.data(),encoded,sizeof(actual),cudaMemcpyDeviceToHost,stream));cuda_check(cudaMemcpyAsync(again.data(),repeated,sizeof(again),cudaMemcpyDeviceToHost,stream));cuda_check(cudaMemcpyAsync(learner.data(),buffers.observations,sizeof(learner),cudaMemcpyDeviceToHost,stream));cuda_check(cudaStreamSynchronize(stream));
        require_probe(std::memcmp(actual.data(),again.data(),sizeof(actual))==0,"repeated encode changed cached observation");
        for(int a=0;a<arenas;a++){
            RekNative5Snapshot sample{};check_runtime(rek_native5_read_snapshot(runtime,a,&sample,stream));
            require_probe(std::memcmp(actual.data()+a*446,learner.data()+a*223,223*sizeof(float))==0,"learner and both-perspective exports differ");
            require_probe(sample.round.failure_bits==0,"physical runtime failure");
            require_probe(sample.raw_observations[0]==sample.qpos[0]&&sample.raw_observations[86]==sample.qpos[36],"raw diagnostic roots changed");
            if(observable){
                RekG1FightState fight{};fight.phase=RekG1FightPhase(sample.round.phase);fight.round_duration_seconds=sample.raw_observations[188];fight.time_remaining_seconds=sample.round.time_remaining_seconds;
                for(int side=0;side<2;side++){fight.clean_hits[side]=sample.round.points[side];fight.count_active[side]=uint8_t(sample.raw_observations[204+side]);}
                float expected[446];require_probe(project_observable(history[a],sample.qpos,fight,sample.round.terminal!=0,tick!=0,previous_terminal[a],expected)==rek_observable_balance::kOk,"physical CPU reference failed");
                for(int k=0;k<446;k++){near_probe(actual[a*446+k],expected[k]);maximum_error=std::max(maximum_error,std::abs(double(actual[a*446+k])-expected[k]));}
                for(int side=0;side<2;side++)for(int k=0;k<223;k++)if(!rek_observable_balance::structurally_available(k))require_probe(actual[a*446+side*223+k]==0,"excluded privileged feature leaked");
                require_probe(actual[a*446+203]==(tick&&!previous_terminal[a]?1.f:0.f),"physical observation history boundary failed");
            }else{
                for(int side=0;side<2;side++){
                    const float* raw=sample.raw_observations+side*223;const float* result=actual.data()+a*446+side*223;
                    for(int k=0;k<223;k++)if(k!=72&&k!=86&&k!=87&&k!=158&&k!=188&&k!=189)require_probe(result[k]==raw[k],"legacy projection changed");
                    near_probe(result[72],float(double(raw[72])/180));near_probe(result[158],float(double(raw[158])/180));near_probe(result[188],float(double(raw[188])/120));near_probe(result[189],float(double(raw[189])/120));
                    near_probe(result[86],float(std::hypot(double(raw[86])-raw[0],double(raw[87])-raw[1])));
                }
            }
            new_rounds+=previous_terminal[a];terminals+=sample.round.terminal!=0;previous_terminal[a]=sample.round.terminal!=0;
        }
        check_runtime(rek_native5_check_status(runtime,stream));
    }
    require_probe(new_rounds>=arenas&&terminals>=arenas,"no actual round boundary exercised");
    check_runtime(rek_native5_reset(runtime,stream));check_runtime(rek_native5_encode_fighter_observations(runtime,encoded,stream));
    if(observable){float first[446];cuda_check(cudaMemcpyAsync(first,encoded,sizeof(first),cudaMemcpyDeviceToHost,stream));cuda_check(cudaStreamSynchronize(stream));require_probe(first[203]==0&&first[426]==0,"explicit API reset retained history");}
    check_runtime(rek_native5_close(runtime));cuda_check(cudaGraphExecDestroy(executable));cuda_check(cudaGraphDestroy(graph));cuda_check(cudaStreamDestroy(stream));
    std::printf("{\"event\":\"physical_adapter_integration\",\"schema\":\"%s\",\"passed\":true,\"arenas\":4,\"ticks\":120,\"diagnostic_round_seconds\":1,\"observations_checked\":968,\"new_rounds\":%u,\"terminal_arena_transitions\":%u,\"maximum_cpu_projection_error\":%.9g,\"joint_pose_available\":%s,\"raw_inspection_unchanged\":true,\"cpu_physics_calls\":0,\"ppo_updates\":0,\"authentic_parity\":false,\"concurrent_authentic_inference\":true,\"performance_claim\":false}\n",observable?rek_observable_balance::kSchema:"rek.native5.scaled_polar_xy.v1",new_rounds,terminals,maximum_error,observable?"false":"null");
}
}
extern "C" void __wrap_mj_step(const mjModel*,mjData*){std::abort();}
extern "C" void __wrap_mj_step1(const mjModel*,mjData*){std::abort();}
extern "C" void __wrap_mj_step2(const mjModel*,mjData*){std::abort();}
extern "C" void __wrap_mj_forward(const mjModel*,mjData*){std::abort();}
extern "C" void __wrap_mj_kinematics(const mjModel*,mjData*){std::abort();}
int main(int argc,char** argv){
    try{cpu_adapter_test();if(argc==2&&std::string(argv[1])=="--cpu-self-test")return 0;integration(argc,argv);return 0;}
    catch(const std::exception& error){std::fprintf(stderr,"physical observable adapter: %s\n",error.what());return 1;}
}
