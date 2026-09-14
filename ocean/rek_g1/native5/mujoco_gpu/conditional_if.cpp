#include "conditional_if.h"
#include "native_module.h"
#include <map>
#include <stdexcept>
#include <utility>

namespace rek_mjgpu {
namespace {
void checked(CUresult result,const char* operation){
    if(result==CUDA_SUCCESS)return;
    const char* error=nullptr;cuGetErrorString(result,&error);
    throw std::runtime_error(std::string(operation)+": "+(error?error:"CUDA error"));
}
}
struct DeviceIf::Impl {
    RekMjGpuModule* module=nullptr;
    CUfunction setter=nullptr;
    CUcontext context=nullptr;
    CUdeviceptr any=0;
    CUstream record_stream=nullptr,body_stream=nullptr;
    struct Cached {CUgraph graph=nullptr;CUgraphExec executable=nullptr;};
    std::map<std::pair<std::uintptr_t,int>,Cached> cached;

    Impl(const std::string& ptx,const std::string& hash){
        try{
            checked(cuCtxGetCurrent(&context),"get GPU IF context");
            if(!context)throw std::runtime_error("GPU IF requires a current CUDA context");
            module=rek_mjgpu_module_load(ptx.c_str(),hash.c_str());
            if(!module)throw std::runtime_error(rek_mjgpu_error());
            if(rek_mjgpu_module_function(module,"rek_mjgpu_mask_any_set_condition",&setter))
                throw std::runtime_error(rek_mjgpu_error());
            checked(cuMemAlloc(&any,sizeof(int)),"allocate GPU IF condition");
            checked(cuStreamCreate(&record_stream,CU_STREAM_NON_BLOCKING),"create GPU IF recording stream");
            checked(cuStreamCreate(&body_stream,CU_STREAM_NON_BLOCKING),"create GPU IF body stream");
        }catch(...){release();throw;}
    }
    ~Impl(){release();}
    void release(){
        for(auto& item:cached){if(item.second.executable)cuGraphExecDestroy(item.second.executable);if(item.second.graph)cuGraphDestroy(item.second.graph);}
        if(body_stream)cuStreamDestroy(body_stream);
        if(record_stream)cuStreamDestroy(record_stream);
        if(any)cuMemFree(any);
        if(module)rek_mjgpu_module_unload(module);
    }
    void capture(CUstream stream,const std::uint8_t* mask,int count,const std::function<void(CUstream)>& body){
        CUstreamCaptureStatus status;CUgraph graph=nullptr;
        const CUgraphNode* dependencies=nullptr;const CUgraphEdgeData* edges=nullptr;std::size_t dependency_count=0;
        checked(cuStreamGetCaptureInfo(stream,&status,nullptr,&graph,&dependencies,&edges,&dependency_count),"get parent IF capture");
        if(status!=CU_STREAM_CAPTURE_STATUS_ACTIVE||!graph)throw std::runtime_error("GPU IF requires active capture");
        CUgraphConditionalHandle handle;
        checked(cuGraphConditionalHandleCreate(&handle,graph,context,0,CU_GRAPH_COND_ASSIGN_DEFAULT),"create GPU IF handle");
        void* arguments[]={&handle,&mask,&count,&any};
        checked(cuLaunchKernel(setter,1,1,1,256,1,1,0,stream,arguments,nullptr),"reduce GPU reset mask");
        checked(cuStreamGetCaptureInfo(stream,&status,nullptr,&graph,&dependencies,&edges,&dependency_count),"get IF dependencies");
        CUgraphNodeParams parameters={};parameters.type=CU_GRAPH_NODE_TYPE_CONDITIONAL;
        parameters.conditional.handle=handle;parameters.conditional.type=CU_GRAPH_COND_TYPE_IF;
        parameters.conditional.size=1;parameters.conditional.ctx=context;
        CUgraphNode branch=nullptr;
        checked(cuGraphAddNode(&branch,graph,dependencies,edges,dependency_count,&parameters),"insert GPU IF node");
        CUgraph body_graph=parameters.conditional.phGraph_out[0];
        checked(cuStreamBeginCaptureToGraph(body_stream,body_graph,nullptr,nullptr,0,CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),"capture GPU IF body");
        try{
            body(body_stream);
            CUgraph captured=nullptr;
            checked(cuStreamEndCapture(body_stream,&captured),"finish GPU IF body capture");
            if(captured!=body_graph)throw std::runtime_error("GPU IF body capture returned another graph");
        }catch(...){
            CUstreamCaptureStatus state;
            if(cuStreamIsCapturing(body_stream,&state)==CUDA_SUCCESS&&state!=CU_STREAM_CAPTURE_STATUS_NONE){
                CUgraph abandoned=nullptr;cuStreamEndCapture(body_stream,&abandoned);
            }
            throw;
        }
        checked(cuStreamUpdateCaptureDependencies(stream,&branch,nullptr,1,CU_STREAM_SET_CAPTURE_DEPENDENCIES),"join GPU IF to capture");
    }
    void execute(CUstream stream,const std::uint8_t* mask,int count,const std::function<void(CUstream)>& body){
        if(!mask||count<=0||!body)throw std::runtime_error("GPU IF requires a positive mask and body");
        CUcontext current=nullptr;checked(cuCtxGetCurrent(&current),"check GPU IF context");
        if(current!=context)throw std::runtime_error("GPU IF called from a different CUDA context");
        CUstreamCaptureStatus status;checked(cuStreamIsCapturing(stream,&status),"inspect GPU IF stream capture");
        if(status==CU_STREAM_CAPTURE_STATUS_ACTIVE){capture(stream,mask,count,body);return;}
        if(status!=CU_STREAM_CAPTURE_STATUS_NONE)throw std::runtime_error("GPU IF called on invalidated capture");
        const auto key=std::make_pair(reinterpret_cast<std::uintptr_t>(mask),count);
        auto found=cached.find(key);
        if(found==cached.end()){
            Cached entry;
            try{
                checked(cuStreamBeginCapture(record_stream,CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),"begin standalone GPU IF capture");
                capture(record_stream,mask,count,body);
                checked(cuStreamEndCapture(record_stream,&entry.graph),"finish standalone GPU IF capture");
                checked(cuGraphInstantiate(&entry.executable,entry.graph,0),"instantiate standalone GPU IF");
            }catch(...){
                CUstreamCaptureStatus state;
                if(cuStreamIsCapturing(record_stream,&state)==CUDA_SUCCESS&&state!=CU_STREAM_CAPTURE_STATUS_NONE){
                    CUgraph abandoned=nullptr;cuStreamEndCapture(record_stream,&abandoned);
                    if(abandoned&&abandoned!=entry.graph)cuGraphDestroy(abandoned);
                }
                if(entry.executable)cuGraphExecDestroy(entry.executable);
                if(entry.graph)cuGraphDestroy(entry.graph);
                throw;
            }
            found=cached.emplace(key,entry).first;
        }
        checked(cuGraphLaunch(found->second.executable,stream),"launch standalone GPU IF");
    }
};
DeviceIf::DeviceIf(const std::string& ptx,const std::string& hash):impl_(std::make_unique<Impl>(ptx,hash)){}
DeviceIf::~DeviceIf()=default;
void DeviceIf::execute(CUstream stream,const std::uint8_t* mask,int count,const std::function<void(CUstream)>& body){impl_->execute(stream,mask,count,body);}
}
