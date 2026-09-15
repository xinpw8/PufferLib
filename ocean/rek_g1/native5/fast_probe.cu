#include "runtime_api.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static void ok(int result){if(result)throw std::runtime_error(rek_native5_error());}
static void require(bool value,const char* reason){if(!value)throw std::runtime_error(reason);}
template<class T>static T* gpu(size_t n){T* p=nullptr;ck(cudaMalloc(&p,n*sizeof(T)));ck(cudaMemset(p,0,n*sizeof(T)));return p;}
__global__ void legal_actions(const uint8_t* mask,float* actions,int arenas,int preferred0,int preferred1){
    int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=arenas*2)return;
    int want=(r%2)?preferred1:preferred0;
    const uint8_t* m=mask+r*33;
    if(m[want])actions[r]=float(want);
    else if(m[0])actions[r]=0;
    else if(m[1])actions[r]=1;
    else{for(int k=0;k<33;k++)if(m[k]){actions[r]=float(k);break;}}
}
static RekNative5Snapshot snapshot(RekNative5Runtime* r,int a,cudaStream_t s){RekNative5Snapshot out{};ok(rek_native5_read_snapshot(r,a,&out,s));return out;}
int main(int argc,char** argv){
    try{
        require(argc==5,"Usage: fast-probe MODEL EXPORT ASSETS FEATURES");
        setenv("REK_PHYSICS_BACKEND","semantic_cuda",1);
        cudaStream_t stream;ck(cudaStreamCreate(&stream));
        constexpr int arenas=4;
        RekNative5Config cfg{};cfg.abi_version=REK_NATIVE5_RUNTIME_ABI;cfg.model_path=argv[1];cfg.physics_export_path=argv[2];cfg.assets_path=argv[3];cfg.motion_features_path=argv[4];cfg.arenas=arenas;cfg.seed=73;cfg.locomotion_segment_ticks=1;cfg.round_seconds=20;
        const unsigned durations[17]={35,27,31,45,32,45,157,145,158,139,134,138,73,75,68,71,103};
        for(int i=0;i<17;i++)cfg.move_duration_ticks[i]=durations[i];
        RekNative5Buffers out{};out.observations=gpu<float>(arenas*223);out.actions=gpu<float>(arenas);out.rewards=gpu<float>(arenas);out.terminals=gpu<float>(arenas);out.logs=gpu<RekNative5Log>(arenas);out.log_stride_bytes=sizeof(RekNative5Log);
        RekNative5Runtime* runtime=rek_native5_create(&cfg,&out,stream);require(runtime!=nullptr,rek_native5_error());
        float* actions=gpu<float>(arenas*2);uint8_t* overrides=gpu<uint8_t>(arenas*2);ck(cudaMemset(overrides,1,arenas*2));
        ok(rek_native5_bind_external_actions(runtime,actions,overrides,stream));
        ok(rek_native5_reset(runtime,stream));
        auto initial=snapshot(runtime,0,stream);RekNative5DeviceView view{};ok(rek_native5_get_device_view(runtime,&view));
        auto tick=[&](int p0,int p1){legal_actions<<<1,32,0,stream>>>(view.action_masks,actions,arenas,p0,p1);ok(rek_native5_step(runtime,stream));};
        for(int t=0;t<50;t++)tick(1,1);
        auto neutral=snapshot(runtime,0,stream);
        require(std::fabs((initial.round.time_remaining_seconds-neutral.round.time_remaining_seconds)-1.0f)<.002f,"50 ticks did not advance match clock by one second");
        for(float x:neutral.raw_observations)require(std::isfinite(x),"nonfinite neutral observations");
        for(int t=0;t<20;t++)tick(2,1);
        auto walked=snapshot(runtime,0,stream);
        float displacement=std::hypot(walked.qpos[0]-neutral.qpos[0],walked.qpos[1]-neutral.qpos[1]);
        require(displacement>.005f,"held walk produced no appreciable translation");
        ok(rek_native5_reset(runtime,stream));
        for(int t=0;t<5;t++)tick(1,1);
        auto before=snapshot(runtime,0,stream);require(before.action_masks[16],"first attack unavailable after neutral reset");
        tick(16,1);auto started=snapshot(runtime,0,stream);
        require(!started.action_masks[16],"attack did not lock additional attacks");
        bool locked=true;
        for(unsigned t=1;t<durations[6];t++){tick(0,1);auto current=snapshot(runtime,0,stream);if(t+1<durations[6])locked=locked&&!current.action_masks[16];}
        auto finished=snapshot(runtime,0,stream);
        require(locked&&finished.action_masks[16],"canned attack duration differs from configured157 ticks");
        for(int t=0;t<10;t++)tick(0,1);
        auto after=snapshot(runtime,0,stream);require(after.action_masks[16],"attack repeated without a new action");
        ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        for(int t=0;t<16;t++)tick(1,1);
        cudaGraph_t graph;ck(cudaStreamEndCapture(stream,&graph));cudaGraphExec_t executable;ck(cudaGraphInstantiate(&executable,graph,0));
        auto clock_before=snapshot(runtime,0,stream);ck(cudaGraphLaunch(executable,stream));auto clock_after=snapshot(runtime,0,stream);
        require(std::fabs((clock_before.round.time_remaining_seconds-clock_after.round.time_remaining_seconds)-.32f)<.002f,"graph replay altered simulated timing");
        for(int t=0;t<1200;t++)tick(1,1);
        ok(rek_native5_check_status(runtime,stream));auto round=snapshot(runtime,0,stream);
        require(round.round.completed_rounds>0,"no completed round after20simseconds");
        std::printf("{\"test\":\"reduced_gpu_semantic_runtime\",\"status\":\"passed\",\"arenas\":4,\"control_hz\":50,\"one_second_ticks\":50,\"held_walk_displacement_m\":%.9g,\"attack_duration_ticks\":157,\"no_attack_autorepeat\":true,\"graph_replay_ticks\":16,\"completed_rounds_arena0\":%llu,\"failure_bits\":%u,\"python_runtime\":false,\"cpu_physics\":false,\"training_sps\":null}\n",displacement,(unsigned long long)round.round.completed_rounds,round.round.failure_bits);
        ck(cudaGraphExecDestroy(executable));ck(cudaGraphDestroy(graph));ok(rek_native5_close(runtime));
        for(void* p:{(void*)out.observations,(void*)out.actions,(void*)out.rewards,(void*)out.terminals,(void*)out.logs,(void*)actions,(void*)overrides})ck(cudaFree(p));ck(cudaStreamDestroy(stream));return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"fast probe: %s\n",e.what());return 2;}
}
