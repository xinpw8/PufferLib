#include "conditional.h"
#include <stdexcept>

namespace rek_mjgpu {
namespace {
void checked(CUresult result,const char* operation){
    if(result==CUDA_SUCCESS)return;
    const char* error=nullptr;cuGetErrorString(result,&error);
    throw std::runtime_error(std::string(operation)+": "+(error?error:"CUDA error"));
}
}
DeviceWhile::DeviceWhile(const std::string& ptx,const std::string& hash){
    module_=rek_mjgpu_module_load(ptx.c_str(),hash.c_str());
    if(!module_)throw std::runtime_error(rek_mjgpu_error());
    if(rek_mjgpu_module_function(module_,"rek_mjgpu_set_condition",&setter_)){
        const std::string error=rek_mjgpu_error();rek_mjgpu_module_unload(module_);module_=nullptr;
        throw std::runtime_error(error);
    }
}
DeviceWhile::~DeviceWhile(){rek_mjgpu_module_unload(module_);}
void DeviceWhile::capture(CUstream stream,CUdeviceptr condition,
        const std::function<CUgraphNode(CUgraph)>& build_body)const{
    if(!condition||!build_body)throw std::runtime_error("GPU while requires condition and body");
    CUstreamCaptureStatus status;CUgraph graph=nullptr;
    const CUgraphNode* dependencies=nullptr;const CUgraphEdgeData* edges=nullptr;std::size_t count=0;
    checked(cuStreamGetCaptureInfo(stream,&status,nullptr,&graph,&dependencies,&edges,&count),"get parent capture");
    if(status!=CU_STREAM_CAPTURE_STATUS_ACTIVE||!graph)throw std::runtime_error("GPU while requires active graph capture");
    CUcontext context=nullptr;checked(cuCtxGetCurrent(&context),"get GPU while context");
    CUgraphConditionalHandle handle;
    checked(cuGraphConditionalHandleCreate(&handle,graph,context,0,CU_GRAPH_COND_ASSIGN_DEFAULT),"create GPU while handle");
    void* arguments[]={&handle,&condition};
    checked(cuLaunchKernel(setter_,1,1,1,1,1,1,0,stream,arguments,nullptr),"set initial GPU while condition");
    checked(cuStreamGetCaptureInfo(stream,&status,nullptr,&graph,&dependencies,&edges,&count),"get while dependencies");
    CUgraphNodeParams parameters={};parameters.type=CU_GRAPH_NODE_TYPE_CONDITIONAL;
    parameters.conditional.handle=handle;parameters.conditional.type=CU_GRAPH_COND_TYPE_WHILE;
    parameters.conditional.size=1;parameters.conditional.ctx=context;
    CUgraphNode loop=nullptr;
    checked(cuGraphAddNode(&loop,graph,dependencies,edges,count,&parameters),"insert GPU while node");
    const auto body=parameters.conditional.phGraph_out[0];
    CUgraphNode tail=build_body(body);
    if(!tail)throw std::runtime_error("GPU while body is empty");
    CUDA_KERNEL_NODE_PARAMS setter={};setter.func=setter_;setter.gridDimX=setter.gridDimY=setter.gridDimZ=1;
    setter.blockDimX=setter.blockDimY=setter.blockDimZ=1;setter.kernelParams=arguments;
    CUgraphNode update=nullptr;
    checked(cuGraphAddKernelNode(&update,body,&tail,1,&setter),"append GPU condition update");
    checked(cuStreamUpdateCaptureDependencies(stream,&loop,nullptr,1,CU_STREAM_SET_CAPTURE_DEPENDENCIES),"join GPU while to capture");
}
}
