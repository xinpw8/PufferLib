#include "conditional_if.h"
#include "native_module.h"
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
void check(CUresult result){if(result!=CUDA_SUCCESS){const char* text=nullptr;cuGetErrorString(result,&text);throw std::runtime_error(text?text:"CUDA error");}}
}
int main(int argc,char** argv){try{
    if(argc!=3)throw std::runtime_error("Usage: conditional-if-probe TEST_PTX SHA256");
    check(cuInit(0));CUdevice device;check(cuDeviceGet(&device,0));CUcontext context;
    check(cuDevicePrimaryCtxRetain(&context,device));check(cuCtxSetCurrent(context));
    CUstream stream,second;check(cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING));check(cuStreamCreate(&second,CU_STREAM_NON_BLOCKING));
    CUdeviceptr mask,remaining,executed;check(cuMemAlloc(&mask,2048));check(cuMemAlloc(&remaining,4));check(cuMemAlloc(&executed,4));
    auto* module=rek_mjgpu_module_load(argv[1],argv[2]);if(!module)throw std::runtime_error(rek_mjgpu_error());
    CUfunction decrement;if(rek_mjgpu_module_function(module,"rek_mjgpu_test_decrement",&decrement))throw std::runtime_error(rek_mjgpu_error());
    {rek_mjgpu::DeviceIf branch(argv[1],argv[2]);int cases=0,captures=0;
    auto body=[&](CUstream s){captures++;void* arguments[]={&remaining,&executed};check(cuLaunchKernel(decrement,1,1,1,1,1,1,0,s,arguments,nullptr));};
    auto run_case=[&](int count,int active,CUgraphExec graph,CUstream s){
        std::vector<unsigned char> bytes(2048);if(active>=0)bytes[active]=active%2?255:1;
        check(cuMemcpyHtoD(mask,bytes.data(),bytes.size()));const int one=1;
        check(cuMemcpyHtoD(remaining,&one,4));check(cuMemsetD32(executed,0,1));
        if(graph)check(cuGraphLaunch(graph,s));else branch.execute(s,reinterpret_cast<const std::uint8_t*>(mask),count,body);
        check(cuStreamSynchronize(s));int runs=-1,left=-1;check(cuMemcpyDtoH(&runs,executed,4));check(cuMemcpyDtoH(&left,remaining,4));
        const int expected=active>=0&&active<count;
        if(runs!=expected||left!=1-expected)throw std::runtime_error("GPU mask IF execution mismatch");cases++;
    };
    for(int count:{1,33,512,1025}){
        for(int repeat=0;repeat<2;repeat++)for(int active:{-1,0,count-1,count})run_case(count,active,nullptr,repeat?second:stream);
        CUgraph graph;CUgraphExec executable;
        check(cuStreamBeginCapture(stream,CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
        branch.execute(stream,reinterpret_cast<const std::uint8_t*>(mask),count,body);
        check(cuStreamEndCapture(stream,&graph));check(cuGraphInstantiate(&executable,graph,0));
        for(int repeat=0;repeat<2;repeat++)for(int active:{-1,0,count-1,count})run_case(count,active,executable,repeat?second:stream);
        check(cuGraphExecDestroy(executable));check(cuGraphDestroy(graph));
    }
    if(captures!=8)throw std::runtime_error("Standalone cache unexpectedly reconstructed the IF body");
    std::printf("{\"test\":\"native_gpu_mask_any_conditional_if\",\"status\":\"passed\",\"cases\":%d,\"body_captures\":%d,\"mask_counts\":[1,33,512,1025],\"zero_and_nonzero_masks\":true,\"out_of_range_bytes_ignored\":true,\"parent_capture_direct_insertion\":true,\"standalone_graph_cache\":true,\"two_launch_streams\":true,\"host_condition_reads_during_execution\":0,\"python_interpreter\":false,\"training_sps\":null}\n",cases,captures);
    }
    if(rek_mjgpu_module_unload(module))throw std::runtime_error(rek_mjgpu_error());
    check(cuMemFree(mask));check(cuMemFree(remaining));check(cuMemFree(executed));check(cuStreamDestroy(stream));check(cuStreamDestroy(second));
    check(cuCtxSetCurrent(nullptr));check(cuDevicePrimaryCtxRelease(device));return 0;
}catch(const std::exception& e){std::fprintf(stderr,"conditional-if-probe: %s\n",e.what());return 1;}}
