#include "native_module.h"
#include "builtin.h" // Installed Warp umbrella header includes array.h in its required order.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static_assert(sizeof(RekMjGpuArray)==sizeof(wp::array_t<float>));
static_assert(alignof(RekMjGpuArray)==alignof(wp::array_t<float>));
static_assert(offsetof(RekMjGpuArray,data)==offsetof(wp::array_t<float>,data));
static_assert(offsetof(RekMjGpuArray,grad)==offsetof(wp::array_t<float>,grad));
static_assert(offsetof(RekMjGpuArray,shape)==offsetof(wp::array_t<float>,shape));
static_assert(offsetof(RekMjGpuArray,strides)==offsetof(wp::array_t<float>,strides));
static_assert(offsetof(RekMjGpuArray,ndim)==offsetof(wp::array_t<float>,ndim));
static_assert(sizeof(RekMjGpuLaunchBounds)==sizeof(wp::launch_bounds_t));
static_assert(alignof(RekMjGpuLaunchBounds)==alignof(wp::launch_bounds_t));
static_assert(offsetof(RekMjGpuLaunchBounds,shape)==offsetof(wp::launch_bounds_t,shape));
static_assert(offsetof(RekMjGpuLaunchBounds,ndim)==offsetof(wp::launch_bounds_t,ndim));
static_assert(offsetof(RekMjGpuLaunchBounds,size)==offsetof(wp::launch_bounds_t,size));

namespace {
void check(CUresult result,const char* operation) {
    if(result==CUDA_SUCCESS)return;
    const char* name=nullptr;const char* message=nullptr;cuGetErrorName(result,&name);cuGetErrorString(result,&message);
    throw std::runtime_error(std::string(operation)+": "+(name?name:"")+": "+(message?message:""));
}
void require(int result) { if(result)throw std::runtime_error(rek_mjgpu_error()); }
struct Context {
    CUdevice device=0;CUcontext context=nullptr;CUstream stream=nullptr;
    Context(){check(cuInit(0),"initialize CUDA driver");check(cuDeviceGet(&device,0),"select GPU");check(cuDevicePrimaryCtxRetain(&context,device),"retain primary context");check(cuCtxSetCurrent(context),"set CUDA context");check(cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING),"create CUDA stream");}
    ~Context(){if(stream)cuStreamDestroy(stream);if(context){cuCtxSetCurrent(nullptr);cuDevicePrimaryCtxRelease(device);}}
};
struct Buffer {
    CUdeviceptr pointer=0;size_t bytes=0;
    explicit Buffer(size_t bytes):bytes(bytes){check(cuMemAlloc(&pointer,bytes),"allocate GPU buffer");CUmemorytype kind;check(cuPointerGetAttribute(&kind,CU_POINTER_ATTRIBUTE_MEMORY_TYPE,pointer),"verify GPU allocation");if(kind!=CU_MEMORYTYPE_DEVICE)throw std::runtime_error("Buffer is not CUDA device memory");}
    ~Buffer(){if(pointer)cuMemFree(pointer);}
    void upload(const std::vector<float>& values){if(values.size()*sizeof(float)!=bytes)throw std::runtime_error("Upload size mismatch");check(cuMemcpyHtoD(pointer,values.data(),bytes),"upload fixture");}
    std::vector<float> read(){std::vector<float> result(bytes/sizeof(float));check(cuMemcpyDtoH(result.data(),pointer,bytes),"download result");return result;}
};
struct Graph {
    CUgraph graph=nullptr;CUgraphExec executable=nullptr;
    ~Graph(){if(executable)cuGraphExecDestroy(executable);if(graph)cuGraphDestroy(graph);}
};
struct Event {
    CUevent event=nullptr;Event(){check(cuEventCreate(&event,CU_EVENT_DEFAULT),"create event");}~Event(){if(event)cuEventDestroy(event);}
};
double verify(const std::vector<float>& result,const std::vector<float>& velocity,const std::vector<float>& acceleration,
    const std::vector<float>& timestep,int worlds,int dofs,int row_stride,float scale,float guard){
    double maximum=0;
    for(int world=0;world<worlds;world++){
        for(int dof=0;dof<dofs;dof++){
            size_t i=size_t(world)*dofs+dof;
            const double expected=double(velocity[i])+double(scale)*double(acceleration[i])*double(timestep[world%timestep.size()]);
            const float actual=result[size_t(world)*row_stride+dof];
            const double error=std::abs(double(actual)-expected);
            if(!std::isfinite(actual)||error>2e-7)throw std::runtime_error("Cached velocity integration disagrees with analytical fixture");
            maximum=std::max(maximum,error);
        }
        for(int padding=dofs;padding<row_stride;padding++)if(result[size_t(world)*row_stride+padding]!=guard)throw std::runtime_error("Output row padding was overwritten");
    }
    return maximum;
}
}

int main(int argc,char** argv) {
    if(argc!=4){std::fprintf(stderr,"Usage: %s CACHED_PTX EXPECTED_SHA256 NEXT_VELOCITY_SYMBOL\n",argv[0]);return 2;}
    try {
        Context context;
        constexpr int worlds=512,dofs=70,replays=1000;
        constexpr float scale=0.5f,guard=-12345.0f;
        const char* wrong="0000000000000000000000000000000000000000000000000000000000000000";
        auto* rejected=rek_mjgpu_module_load(argv[1],wrong);
        if(rejected){rek_mjgpu_module_unload(rejected);throw std::runtime_error("Module hash mismatch was accepted");}
        auto* module=rek_mjgpu_module_load(argv[1],argv[2]);
        if(!module)throw std::runtime_error(rek_mjgpu_error());
        CUfunction function=nullptr;
        if(rek_mjgpu_module_function(module,"__rek_missing_kernel__",&function)==0)throw std::runtime_error("Missing kernel lookup was accepted");
        require(rek_mjgpu_module_function(module,argv[3],&function));
        std::vector<float> velocity(size_t(worlds)*dofs),acceleration(velocity.size());
        for(size_t i=0;i<velocity.size();i++){velocity[i]=float(int(i%31)-15)*0.03125f;acceleration[i]=float(int(i%17)-8)*0.125f;}
        Buffer qvel(velocity.size()*4),qacc(acceleration.size()*4);qvel.upload(velocity);qacc.upload(acceleration);
        auto bounds=rek_mjgpu::launch_bounds({worlds,dofs});
        auto velocity_array=rek_mjgpu::contiguous_array(reinterpret_cast<void*>(qvel.pointer),4,{worlds,dofs});
        auto acceleration_array=rek_mjgpu::contiguous_array(reinterpret_cast<void*>(qacc.pointer),4,{worlds,dofs});
        // This cached module declares WP_TILE_BLOCK_DIM=32. Its generated
        // shared-memory allocator also requires the actual launch block size.
        unsigned block[3]={32,1,1},grid[3]={unsigned((bounds.size+31)/32),1,1};
        double maximum=0;float graph_milliseconds=0;double graph_wall_seconds=0;
        for(int case_index=0;case_index<2;case_index++){
            const int timestep_rows=case_index?worlds:1,row_stride=case_index?dofs+2:dofs;
            std::vector<float> timestep(timestep_rows,0.002f);
            if(case_index)for(int world=0;world<worlds;world++)timestep[world]=0.001f*float(1+world%3);
            Buffer dt(timestep.size()*4),output(size_t(worlds)*row_stride*4);dt.upload(timestep);output.upload(std::vector<float>(size_t(worlds)*row_stride,guard));
            auto dt_array=rek_mjgpu::contiguous_array(reinterpret_cast<void*>(dt.pointer),4,{timestep_rows});
            auto output_array=rek_mjgpu::contiguous_array(reinterpret_cast<void*>(output.pointer),4,{worlds,dofs});output_array.strides[0]=row_stride*4;
            float acceleration_scale=scale;
            void* arguments[]={&bounds,&dt_array,&velocity_array,&acceleration_array,&acceleration_scale,&output_array};
            require(rek_mjgpu_launch(function,grid,block,0,arguments,context.stream));
            check(cuStreamSynchronize(context.stream),"synchronize velocity integration");
            maximum=std::max(maximum,verify(output.read(),velocity,acceleration,timestep,worlds,dofs,row_stride,scale,guard));
            if(case_index){
                Graph graph;
                check(cuStreamBeginCapture(context.stream,CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),"begin graph capture");
                require(rek_mjgpu_launch(function,grid,block,0,arguments,context.stream));
                check(cuStreamEndCapture(context.stream,&graph.graph),"end graph capture");
                check(cuGraphInstantiate(&graph.executable,graph.graph,0),"instantiate captured kernel graph");
                Event begin,end;const auto started=std::chrono::steady_clock::now();
                check(cuEventRecord(begin.event,context.stream),"record start event");
                for(int replay=0;replay<replays;replay++)check(cuGraphLaunch(graph.executable,context.stream),"replay cached kernel graph");
                check(cuEventRecord(end.event,context.stream),"record end event");check(cuEventSynchronize(end.event),"wait for graph replays");
                graph_wall_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
                check(cuEventElapsedTime(&graph_milliseconds,begin.event,end.event),"measure graph replays");
                maximum=std::max(maximum,verify(output.read(),velocity,acceleration,timestep,worlds,dofs,row_stride,scale,guard));
            }
        }
        if(qvel.read()!=velocity||qacc.read()!=acceleration)throw std::runtime_error("Input fixture was mutated");
        char device_name[256]={};check(cuDeviceGetName(device_name,sizeof(device_name),context.device),"get GPU name");
        std::printf("{\"test\":\"cached_mujoco_warp_next_velocity_native_driver\",\"gpu\":\"%s\",\"kernel\":\"%s\",\"ptx_sha256\":\"%s\",\"worlds\":%d,\"dofs\":%d,\"verified_values\":%d,\"max_abs_error_vs_double_fixture\":%.17g,\"array_abi_bytes\":%zu,\"launch_abi_bytes\":%zu,\"broadcast_timestep\":true,\"per_world_timestep\":true,\"strided_output_padding\":true,\"inputs_unchanged\":true,\"hash_mismatch_rejected\":true,\"missing_kernel_rejected\":true,\"graph_capture\":true,\"graph_replays\":%d,\"graph_cuda_milliseconds\":%.9g,\"graph_wall_seconds\":%.9g,\"python_interpreter\":false,\"cpu_physics_steps\":0,\"full_mujoco_step\":false,\"training_sps\":null}\n",device_name,argv[3],rek_mjgpu_module_sha256(module),worlds,dofs,worlds*dofs*3,maximum,sizeof(RekMjGpuArray),sizeof(RekMjGpuLaunchBounds),replays,graph_milliseconds,graph_wall_seconds);
        require(rek_mjgpu_module_unload(module));return 0;
    } catch(const std::exception& error){std::fprintf(stderr,"native cached-kernel smoke failed: %s\n",error.what());return 1;}
}
