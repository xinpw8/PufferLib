// Long native-controller rollout. No Python, CPU physics, or optimizer updates.
#include "runtime_api.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static void rk(int e){if(e)throw std::runtime_error(rek_native5_error());}
__global__ void probe_actions(float* actions,const uint8_t* masks,int arenas,const int* tick){
    int a=blockIdx.x*blockDim.x+threadIdx.x;if(a>=arenas)return;
    int first=(a+*tick/8)%33;
    for(int k=0;k<33;k++){int action=(first+k)%33;if(masks[a*33+action]){actions[a]=float(action);return;}}
    actions[a]=-1; // Existing runtime detects invalid masks/actions.
}
__global__ void advance_probe(int* tick){++*tick;}
int main(int argc,char** argv){
    try{
        if(argc!=9)throw std::runtime_error("usage: physics_stability_probe ARENAS TICKS XML EXPORT ASSETS FEATURES ENCODER DECODER");
        int arenas=std::stoi(argv[1]),ticks=std::stoi(argv[2]);
        if(arenas<1||ticks<1)throw std::runtime_error("positive dimensions required");
        cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        RekNative5Config cfg{};cfg.abi_version=REK_NATIVE5_RUNTIME_ABI;cfg.arenas=arenas;cfg.seed=73;
        cfg.model_path=argv[3];cfg.physics_export_path=argv[4];cfg.assets_path=argv[5];
        cfg.motion_features_path=argv[6];cfg.controller_encoder_path=argv[7];cfg.controller_decoder_path=argv[8];
        cfg.locomotion_segment_ticks=1;
        const uint32_t duration[]={35,27,31,45,32,45,157,145,158,139,134,138,73,75,68,71,103};
        std::copy(duration,duration+17,cfg.move_duration_ticks);
        RekNative5Buffers out{};out.log_stride_bytes=sizeof(RekNative5Log);
        ck(cudaMalloc(&out.observations,arenas*223*sizeof(float)));ck(cudaMalloc(&out.actions,arenas*sizeof(float)));
        ck(cudaMalloc(&out.rewards,arenas*sizeof(float)));ck(cudaMalloc(&out.terminals,arenas*sizeof(float)));
        ck(cudaMalloc(&out.logs,arenas*sizeof(RekNative5Log)));ck(cudaMemsetAsync(out.logs,0,arenas*sizeof(RekNative5Log),stream));
        uint8_t* mask;int* tick;ck(cudaMalloc(&mask,arenas*33));ck(cudaMalloc(&tick,sizeof(int)));ck(cudaMemsetAsync(tick,0,sizeof(int),stream));
        auto* runtime=rek_native5_create(&cfg,&out,stream);if(!runtime)throw std::runtime_error(rek_native5_error());
        rk(rek_native5_bind_action_mask(runtime,mask,stream));ck(cudaStreamSynchronize(stream));
        const char* backend=std::getenv("REK_PHYSICS_BACKEND");
        bool cpu_eval=backend && (!std::strcmp(backend,"mujoco_cpu_eval") || !std::strcmp(backend,"puffysics_cpu_eval"));
        auto step=[&](){probe_actions<<<(arenas+127)/128,128,0,stream>>>(out.actions,mask,arenas,tick);
            rk(rek_native5_step(runtime,stream));advance_probe<<<1,1,0,stream>>>(tick);};
        cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
        if(!cpu_eval){ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));step();
            ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&executable,graph,0));}
        auto start=std::chrono::steady_clock::now();std::vector<RekNative5Log> logs(arenas);
        for(int completed=0;completed<ticks;){
            int count=std::min(100,ticks-completed);
            for(int k=0;k<count;k++){if(cpu_eval)step();else ck(cudaGraphLaunch(executable,stream));}
            rk(rek_native5_check_status(runtime,stream));completed+=count;
            ck(cudaMemcpy(logs.data(),out.logs,arenas*sizeof(RekNative5Log),cudaMemcpyDeviceToHost));
            double episodes=0,wins=0,losses=0,draws=0,score=0,falls=0;
            for(const auto& l:logs){episodes+=l.n;wins+=l.wins;losses+=l.losses;draws+=l.draws;score+=l.score;falls+=l.falls;}
            double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            std::printf("{\"ticks\":%d,\"arenas\":%d,\"simulated_seconds_per_arena\":%.3f,\"wall_seconds\":%.6f,\"completed_episodes\":%.0f,\"wins\":%.0f,\"losses\":%.0f,\"draws\":%.0f,\"score_sum\":%.0f,\"falls_sum\":%.0f,\"ppo_updates\":0}\n",completed,arenas,completed*.02,elapsed,episodes,wins,losses,draws,score,falls);std::fflush(stdout);
        }
        rk(rek_native5_close(runtime));if(executable)ck(cudaGraphExecDestroy(executable));if(graph)ck(cudaGraphDestroy(graph));
        cudaFree(mask);cudaFree(tick);cudaFree(out.observations);cudaFree(out.actions);cudaFree(out.rewards);cudaFree(out.terminals);cudaFree(out.logs);cudaStreamDestroy(stream);
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
