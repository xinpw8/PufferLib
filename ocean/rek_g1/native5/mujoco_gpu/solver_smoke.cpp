#include "solver_schedule.h"
#include "constraint_schedule.h"
#include "conditional.h"
#include "kernel_program.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
void check(CUresult result,const char* where){if(result!=CUDA_SUCCESS){const char* why=nullptr;cuGetErrorString(result,&why);throw std::runtime_error(std::string(where)+": "+(why?why:"CUDA error"));}}
template<class T>std::vector<T> read(const rek_mjgpu::ModelData& data,const char* name){
    std::vector<T> values(data.byte_size(name)/sizeof(T));
    if(!values.empty())check(cuMemcpyDtoH(values.data(),reinterpret_cast<CUdeviceptr>(data.array(name).data),data.byte_size(name)),name);
    return values;
}
}
int main(int argc,char** argv){try{
    if(argc!=5)throw std::runtime_error("Usage: solver-smoke MODEL.xml CATALOG.json CONDITIONAL.ptx SHA256");
    check(cuInit(0),"cuInit");CUdevice dev;CUcontext context;CUstream stream;
    check(cuDeviceGet(&dev,0),"cuDeviceGet");check(cuDevicePrimaryCtxRetain(&context,dev),"context retain");
    check(cuCtxSetCurrent(context),"set context");check(cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING),"stream create");
    char error[1024]={};std::unique_ptr<mjModel,decltype(&mj_deleteModel)> model(mj_loadXML(argv[1],nullptr,error,sizeof(error)),mj_deleteModel);
    if(!model)throw std::runtime_error(error);model->opt.timestep=.002;
    {rek_mjgpu::ModelDataConfig config;config.nworld=4;rek_mjgpu::ModelData data(model.get(),config);
    auto pre=rek_mjgpu::build_kinematics_com(data);rek_mjgpu::append_crb(pre,data);rek_mjgpu::append_constraints(pre,data);
    const auto solver=rek_mjgpu::build_newton_solver(data);
    rek_mjgpu::KernelProgram prepare(data,argv[2],pre),init(data,argv[2],solver.initialize),iteration(data,argv[2],solver.iteration);
    rek_mjgpu::DeviceWhile loop(argv[3],argv[4]);
    prepare.launch(stream);check(cuStreamSynchronize(stream),"prepare real model mass matrix and constraints");
    const auto rows=read<int>(data,"d.nefc");
    for(int row:rows)if(row!=58)throw std::runtime_error("Fixture expects 58 joint-friction rows and no contact/limit rows");
    // Deliberately nonzero warm start, zero external generalized forces and
    // zero velocities. The analytically stationary solution is qacc=0.
    std::vector<float> warm(std::size_t(config.nworld)*model->nv);
    for(int w=0;w<config.nworld;w++)for(int j=0;j<model->nv;j++)warm[w*model->nv+j]=float((j%7)-3)*(.01f*(w+1));
    data.upload("d.qacc_warmstart",warm.data(),warm.size()*sizeof(float));
    // First execute each initialization kernel separately to localize ABI or
    // launch errors before constructing the convergence-controlled graph.
    for(std::size_t i=0;i<init.nodes();i++){
        std::fprintf(stderr,"init node %zu %s\n",i,solver.initialize[i].entry_prefix.c_str());
        init.launch_one(i,stream);check(cuStreamSynchronize(stream),"solver initialization node");
    }
    CUgraph graph=nullptr;CUgraphExec executable=nullptr;
    check(cuStreamBeginCapture(stream,CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),"begin solver graph");
    init.launch(stream);
    loop.capture(stream,reinterpret_cast<CUdeviceptr>(data.array(solver.condition_field).data),
                 [&](CUgraph body){return iteration.append_to_graph(body);});
    check(cuStreamEndCapture(stream,&graph),"finish solver graph");
    check(cuGraphInstantiate(&executable,graph,0),"instantiate solver graph");
    double max_acceleration=0;int min_iterations=1000000,max_iterations=0;
    constexpr int repeats=8;
    for(int repeat=0;repeat<repeats;repeat++){
        check(cuGraphLaunch(executable,stream),"launch solver graph");check(cuStreamSynchronize(stream),"complete solver graph");
        const auto pending=read<int>(data,"solver.nsolving");
        if(pending[0]!=0)throw std::runtime_error("GPU convergence loop left unfinished worlds");
        const auto counts=read<int>(data,"d.solver_niter");
        for(int count:counts){if(count<1||count>solver.iteration_limit)throw std::runtime_error("Invalid iteration count");min_iterations=std::min(min_iterations,count);max_iterations=std::max(max_iterations,count);}
        for(const char* field:{"d.qacc","d.qfrc_constraint","solver.cost","solver.grad","solver.Mgrad"})
            for(float value:read<float>(data,field))if(!std::isfinite(value))throw std::runtime_error(std::string("Nonfinite solver output ")+field);
        for(float value:read<float>(data,"d.qacc"))max_acceleration=std::max(max_acceleration,std::abs(double(value)));
    }
    if(max_acceleration>1e-4)throw std::runtime_error("Stationary constraint solve did not converge near zero acceleration");
    std::printf("{\"status\":\"passed\",\"test\":\"native_gpu_newton_stationary_warmstart_fixture\",\"worlds\":4,\"nv\":70,\"constraint_rows_per_world\":58,\"initialization_nodes\":%zu,\"iteration_nodes\":%zu,\"graph_replays\":%d,\"min_newton_iterations\":%d,\"max_newton_iterations\":%d,\"max_final_acceleration\":%.17g,\"all_worlds_finished\":true,\"gpu_conditional_convergence\":true,\"host_condition_reads_during_solve\":0,\"python_invocations\":0,\"cpu_physics_steps\":0,\"full_physics_step\":false,\"training_sps\":null}\n",init.nodes(),iteration.nodes(),repeats,min_iterations,max_iterations,max_acceleration);
    check(cuGraphExecDestroy(executable),"destroy solver executable");check(cuGraphDestroy(graph),"destroy solver graph");
    }check(cuStreamDestroy(stream),"destroy stream");check(cuDevicePrimaryCtxRelease(dev),"release context");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"solver-smoke: %s\n",e.what());return 1;}}
