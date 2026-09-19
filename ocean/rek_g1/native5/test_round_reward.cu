#include "round_reward.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>

static void cuda_ok(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__global__ void rewards(float* output){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=4096)return;
    const int a=i%51,b=(i/51)%51,da=(i/2601)%6,db=(i/137)%6;
    const bool terminal=(i%3)==0;const int winner=(i/3)%3-1,side=i%2;
    output[i]=rek5_round_reward::value(rek5_round_reward::RoundOutcome,.9998844821426083f,a,b,a+da,b+db,terminal,winner,side);
}
int main(){try{
    float* result=nullptr;cuda_ok(cudaMallocManaged(&result,4096*sizeof(float)));
    cudaStream_t stream;cuda_ok(cudaStreamCreate(&stream));
    cuda_ok(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    rewards<<<16,256,0,stream>>>(result);
    cudaGraph_t graph;cudaGraphExec_t execution;
    cuda_ok(cudaStreamEndCapture(stream,&graph));cuda_ok(cudaGraphInstantiate(&execution,graph,0));
    double worst=0;unsigned checks=0;
    for(int replay=0;replay<4;replay++){
        cuda_ok(cudaGraphLaunch(execution,stream));cuda_ok(cudaStreamSynchronize(stream));
        for(int i=0;i<4096;i++){
            const int a=i%51,b=(i/51)%51,da=(i/2601)%6,db=(i/137)%6;
            const float expected=rek5_round_reward::value(rek5_round_reward::RoundOutcome,.9998844821426083f,a,b,a+da,b+db,(i%3)==0,(i/3)%3-1,i%2);
            worst=std::fmax(worst,std::abs(double(result[i])-expected));checks++;
            if(!std::isfinite(result[i])||std::abs(result[i]-expected)>3e-7f)throw std::runtime_error("CPU/CUDA reward mismatch");
        }
    }
    std::printf("{\"test\":\"round_reward_cuda\",\"checks\":%u,\"graph_replays\":4,\"max_error\":%.12g,\"status\":\"passed\"}\n",checks,worst);
    cuda_ok(cudaGraphExecDestroy(execution));cuda_ok(cudaGraphDestroy(graph));cuda_ok(cudaStreamDestroy(stream));cuda_ok(cudaFree(result));return 0;
}catch(const std::exception& e){std::fprintf(stderr,"reward CUDA test: %s\n",e.what());return 1;}}
