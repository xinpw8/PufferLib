#include "kernel_program.h"
#include <cuda.h>
#include <mujoco/mujoco.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(CUresult result,const char* operation){if(result==CUDA_SUCCESS)return;const char* message=nullptr;cuGetErrorString(result,&message);throw std::runtime_error(std::string(operation)+": "+(message?message:"CUDA error"));}
struct Context {
    CUdevice device=0;CUcontext context=nullptr;CUstream stream=nullptr;
    Context(){check(cuInit(0),"initialize driver");check(cuDeviceGet(&device,0),"select GPU");check(cuDevicePrimaryCtxRetain(&context,device),"retain context");check(cuCtxSetCurrent(context),"set context");check(cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING),"create stream");}
    ~Context(){if(stream)cuStreamDestroy(stream);if(context){cuCtxSetCurrent(nullptr);cuDevicePrimaryCtxRelease(device);}}
};
struct Graph {CUgraph graph=nullptr;CUgraphExec executable=nullptr;~Graph(){if(executable)cuGraphExecDestroy(executable);if(graph)cuGraphDestroy(graph);}};
struct Event {CUevent event=nullptr;Event(){check(cuEventCreate(&event,CU_EVENT_DEFAULT),"create timing event");}~Event(){if(event)cuEventDestroy(event);}};
std::vector<float> read(const rek_mjgpu::ModelData& data,const std::string& name){
    const auto& storage=data.arrays().at(name);
    if(storage.element!=rek_mjgpu::Element::F32||storage.bytes%4)throw std::runtime_error("Unexpected float output storage: "+name);
    std::vector<float> values(storage.bytes/4);
    if(storage.bytes)check(cuMemcpyDtoH(values.data(),reinterpret_cast<CUdeviceptr>(storage.view.data),storage.bytes),"read GPU output");
    for(float value:values)if(!std::isfinite(value))throw std::runtime_error("Nonfinite GPU output: "+name);
    return values;
}
float component(const rek_mjgpu::ModelData& data,const std::string& name,const std::vector<float>& values,int row,int item,int c){
    const auto view=data.array(name);
    if(view.ndim!=2||row>=view.shape[0]||item>=view.shape[1])throw std::runtime_error("Unexpected output shape: "+name);
    return values.at((size_t(row)*view.strides[0]+size_t(item)*view.strides[1])/4+c);
}
}

int main(int argc,char** argv){
    if(argc!=3){std::fprintf(stderr,"Usage: %s REK_MODEL_XML KERNEL_CATALOG_JSON\n",argv[0]);return 2;}
    try {
        Context context;
        char error[2048]={};
        std::unique_ptr<mjModel,decltype(&mj_deleteModel)> model(mj_loadXML(argv[1],nullptr,error,sizeof(error)),mj_deleteModel);
        if(!model)throw std::runtime_error(std::string("Model parse failed: ")+error);
        rek_mjgpu::ModelDataConfig config;config.nworld=4;
        rek_mjgpu::ModelData data(model.get(),config);
        const auto input_positions=read(data,"d.qpos");
        const auto input_velocities=read(data,"d.qvel");
        const auto spec=rek_mjgpu::build_kinematics_com(data);
        rek_mjgpu::KernelProgram program(data,argv[2],spec);
        program.launch(context.stream);check(cuStreamSynchronize(context.stream),"complete GPU kinematics and COM");
        const std::vector<std::string> names={"d.xpos","d.xquat","d.xmat","d.xipos","d.ximat","d.geom_xpos","d.geom_xmat","d.subtree_com","d.cinert","d.cdof"};
        std::vector<std::vector<float>> before;
        size_t finite_values=0;
        for(const auto& name:names){before.push_back(read(data,name));finite_values+=before.back().size();}
        const auto xpos=read(data,"d.xpos"),xquat=read(data,"d.xquat");
        double root_error=0,quaternion_norm_error=0;int free_roots=0;
        for(int joint=0;joint<model->njnt;joint++)if(model->jnt_type[joint]==mjJNT_FREE){
            const int body=model->jnt_bodyid[joint],qpos_address=model->jnt_qposadr[joint];
            if(model->body_parentid[body]!=0)throw std::runtime_error("Expected top-level free-joint body");
            free_roots++;
            for(int world=0;world<config.nworld;world++){
                for(int c=0;c<3;c++)root_error=std::max(root_error,std::abs(double(component(data,"d.xpos",xpos,world,body,c))-input_positions.at(size_t(world)*model->nq+qpos_address+c)));
                double norm=0;for(int c=0;c<4;c++){double value=component(data,"d.xquat",xquat,world,body,c);norm+=value*value;}
                quaternion_norm_error=std::max(quaternion_norm_error,std::abs(norm-1));
            }
        }
        if(free_roots!=2)throw std::runtime_error("REK smoke expects exactly two free-joint fighter roots");
        if(root_error>1e-6||quaternion_norm_error>1e-6)throw std::runtime_error("GPU root transform does not match packed initial positions/unit orientations");
        Graph graph;check(cuStreamBeginCapture(context.stream,CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),"begin program capture");
        program.launch(context.stream);check(cuStreamEndCapture(context.stream,&graph.graph),"finish program capture");
        check(cuGraphInstantiate(&graph.executable,graph.graph,0),"instantiate program graph");
        constexpr int repeats=32;Event start,finish;
        const auto began=std::chrono::steady_clock::now();check(cuEventRecord(start.event,context.stream),"record graph start");
        for(int i=0;i<repeats;i++)check(cuGraphLaunch(graph.executable,context.stream),"launch program graph");
        check(cuEventRecord(finish.event,context.stream),"record graph finish");check(cuEventSynchronize(finish.event),"wait for program graph");
        const double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();float milliseconds=0;
        check(cuEventElapsedTime(&milliseconds,start.event,finish.event),"measure program graph");
        double replay_difference=0;
        for(size_t field=0;field<names.size();field++){
            const auto after=read(data,names[field]);
            for(size_t i=0;i<after.size();i++)replay_difference=std::max(replay_difference,std::abs(double(after[i])-before[field][i]));
        }
        if(read(data,"d.qpos")!=input_positions||read(data,"d.qvel")!=input_velocities)throw std::runtime_error("Kinematics/COM unexpectedly mutated integration state");
        char gpu[256]={};check(cuDeviceGetName(gpu,sizeof(gpu),context.device),"read device name");
        std::printf("{\"test\":\"actual_rek_gpu_kinematics_com_native_driver\",\"gpu\":\"%s\",\"worlds\":%d,\"nq\":%d,\"nv\":%d,\"bodies\":%d,\"geometries\":%d,\"schedule_nodes\":%zu,\"checked_finite_output_values\":%zu,\"free_fighter_roots\":%d,\"max_root_position_error\":%.17g,\"max_root_quaternion_norm_error\":%.17g,\"integration_state_unchanged\":true,\"graph_capture\":true,\"graph_replays\":%d,\"graph_cuda_milliseconds\":%.9g,\"graph_wall_seconds\":%.9g,\"max_graph_output_difference\":%.17g,\"python_interpreter\":false,\"cpu_physics_steps\":0,\"cpu_physics_reference\":false,\"full_mujoco_step\":false,\"training_sps\":null}\n",gpu,config.nworld,model->nq,model->nv,model->nbody,model->ngeom,program.nodes(),finite_values,free_roots,root_error,quaternion_norm_error,repeats,milliseconds,wall,replay_difference);
        return 0;
    }catch(const std::exception& error){std::fprintf(stderr,"Native MuJoCo GPU program smoke failed: %s\n",error.what());return 1;}
}
