#include "conditional.h"
#include <cstdio>
#include <stdexcept>
#include <string>

namespace {
void check(CUresult result){if(result!=CUDA_SUCCESS){const char* text=nullptr;cuGetErrorString(result,&text);throw std::runtime_error(text?text:"CUDA error");}}
}
int main(int argc,char** argv){
    if(argc!=3){std::fprintf(stderr,"Usage: conditional-probe TEST_PTX SHA256\n");return 2;}
    try{
        check(cuInit(0));CUdevice device;check(cuDeviceGet(&device,0));CUcontext context;
        check(cuDevicePrimaryCtxRetain(&context,device));check(cuCtxSetCurrent(context));
        CUstream stream;check(cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING));
        CUdeviceptr remaining,executed;check(cuMemAlloc(&remaining,4));check(cuMemAlloc(&executed,4));
        auto* module=rek_mjgpu_module_load(argv[1],argv[2]);if(!module)throw std::runtime_error(rek_mjgpu_error());
        CUfunction decrement;
        if(rek_mjgpu_module_function(module,"rek_mjgpu_test_decrement",&decrement))throw std::runtime_error(rek_mjgpu_error());
        {
            rek_mjgpu::DeviceWhile loop(argv[1],argv[2]);
            check(cuStreamBeginCapture(stream,CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
            loop.capture(stream,remaining,[&](CUgraph body){
                CUDA_KERNEL_NODE_PARAMS p={};p.func=decrement;
                p.gridDimX=p.gridDimY=p.gridDimZ=p.blockDimX=p.blockDimY=p.blockDimZ=1;
                void* arguments[]={&remaining,&executed};p.kernelParams=arguments;
                CUgraphNode node;check(cuGraphAddKernelNode(&node,body,nullptr,0,&p));return node;
            });
            CUgraph graph;check(cuStreamEndCapture(stream,&graph));CUgraphExec executable;check(cuGraphInstantiate(&executable,graph,0));
            int checks=0;
            for(int repetitions=0;repetitions<3;repetitions++)for(int initial:{0,1,7,100}){
                check(cuMemcpyHtoD(remaining,&initial,4));check(cuMemsetD32(executed,0,1));
                check(cuGraphLaunch(executable,stream));check(cuStreamSynchronize(stream));
                int left=-1,runs=-1;check(cuMemcpyDtoH(&left,remaining,4));check(cuMemcpyDtoH(&runs,executed,4));
                if(left!=0||runs!=initial)throw std::runtime_error("GPU while iteration count mismatch");checks++;
            }
            std::printf("{\"test\":\"native_gpu_conditional_while\",\"cases\":%d,\"initial_counts\":[0,1,7,100],\"graph_reuse\":true,\"host_condition_reads_during_execution\":0,\"python_interpreter\":false,\"training_sps\":null}\n",checks);
            check(cuGraphExecDestroy(executable));check(cuGraphDestroy(graph));
        }
        if(rek_mjgpu_module_unload(module))throw std::runtime_error(rek_mjgpu_error());
        check(cuMemFree(remaining));check(cuMemFree(executed));check(cuStreamDestroy(stream));
        check(cuCtxSetCurrent(nullptr));check(cuDevicePrimaryCtxRelease(device));return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"native conditional probe: %s\n",e.what());return 1;}
}
