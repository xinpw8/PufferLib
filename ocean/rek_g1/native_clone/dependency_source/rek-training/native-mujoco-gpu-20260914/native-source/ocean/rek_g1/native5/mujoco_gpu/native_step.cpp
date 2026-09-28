#include "native_step.h"
#include "collision_schedule.h"
#include "constraint_schedule.h"
#include "solver_schedule.h"
#include "kernel_program.h"
#include "conditional.h"
#include <stdexcept>

namespace rek_mjgpu {
namespace {
void check(CUresult result,const char* label){
    if(result==CUDA_SUCCESS)return;
    const char* error=nullptr;cuGetErrorString(result,&error);
    throw std::runtime_error(std::string(label)+": "+(error?error:"CUDA error"));
}
}
struct NativeStep::Impl {
    ModelData& data;
    DeviceWhile loop;
    std::unique_ptr<KernelProgram> forward,initialize,iteration,integrate,refresh,collision_refresh;
    CUdeviceptr condition=0;
    int iteration_limit=0;
    CUgraph graph=nullptr;
    CUgraphExec executable=nullptr;

    Impl(ModelData& d,const std::string& catalog,const std::string& ptx,const std::string& hash)
        :data(d),loop(ptx,hash){
        ScheduleSpec pre;
        append_kinematics_com(pre,data);
        append_camlight(pre,data);
        append_crb(pre,data);
        ScheduleSpec collisions;append_collision(collisions,data);
        pre.insert(pre.end(),collisions.begin(),collisions.end());
        append_constraints(pre,data);
        append_transmission(pre,data);
        append_velocity_forces(pre,data);
        append_actuation(pre,data);
        append_smooth_acceleration(pre,data);
        auto solver=build_newton_solver(data);
        iteration_limit=solver.iteration_limit;
        condition=reinterpret_cast<CUdeviceptr>(data.array(solver.condition_field).data);
        ScheduleSpec after;append_implicitfast_integration(after,data);
        ScheduleSpec export_pose;
        append_kinematics_com(export_pose,data);append_spatial_velocity(export_pose,data);
        forward=std::make_unique<KernelProgram>(data,catalog,pre);
        initialize=std::make_unique<KernelProgram>(data,catalog,solver.initialize);
        iteration=std::make_unique<KernelProgram>(data,catalog,solver.iteration);
        integrate=std::make_unique<KernelProgram>(data,catalog,after);
        refresh=std::make_unique<KernelProgram>(data,catalog,export_pose);
        collision_refresh=std::make_unique<KernelProgram>(data,catalog,collisions);
        CUstream setup=nullptr;
        check(cuStreamCreate(&setup,CU_STREAM_NON_BLOCKING),"create native step graph setup stream");
        try{
            check(cuStreamBeginCapture(setup,CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),"begin native step graph");
            capture(setup);
            check(cuStreamEndCapture(setup,&graph),"end native step graph");
            check(cuGraphInstantiate(&executable,graph,0),"instantiate native MuJoCo step");
            check(cuStreamDestroy(setup),"destroy native graph setup stream");
        }catch(...){
            CUstreamCaptureStatus status;
            if(cuStreamIsCapturing(setup,&status)==CUDA_SUCCESS&&status!=CU_STREAM_CAPTURE_STATUS_NONE){
                CUgraph partial=nullptr;cuStreamEndCapture(setup,&partial);if(partial&&partial!=graph)cuGraphDestroy(partial);
            }
            cuStreamDestroy(setup);if(executable)cuGraphExecDestroy(executable);if(graph)cuGraphDestroy(graph);
            throw;
        }
    }
    ~Impl(){if(executable)cuGraphExecDestroy(executable);if(graph)cuGraphDestroy(graph);}
    void capture(CUstream stream)const{
        forward->launch(stream);initialize->launch(stream);
        if(iteration_limit>0)loop.capture(stream,condition,[&](CUgraph body){return iteration->append_to_graph(body);});
        integrate->launch(stream);
    }
};
NativeStep::NativeStep(ModelData& data,const std::string& catalog,const std::string& ptx,const std::string& hash)
    :impl_(std::make_unique<Impl>(data,catalog,ptx,hash)){}
NativeStep::~NativeStep()=default;
void NativeStep::step(CUstream stream)const{
    CUstreamCaptureStatus capture;
    check(cuStreamIsCapturing(stream,&capture),"inspect native MuJoCo stream");
    if(capture==CU_STREAM_CAPTURE_STATUS_ACTIVE){
        // CUDA disallows conditional nodes hidden inside a child graph. Insert
        // this schedule directly into the caller's graph at construction time.
        impl_->capture(stream);
    }else if(capture==CU_STREAM_CAPTURE_STATUS_NONE){
        check(cuGraphLaunch(impl_->executable,stream),"launch native MuJoCo step graph");
    }else throw std::runtime_error("Native MuJoCo called on invalidated CUDA capture");
}
void NativeStep::refresh(CUstream stream)const{impl_->refresh->launch(stream);}
void NativeStep::refresh_contacts(CUstream stream)const{refresh(stream);impl_->collision_refresh->launch(stream);}
std::size_t NativeStep::phase_nodes()const{
    return impl_->forward->nodes()+impl_->initialize->nodes()+impl_->iteration->nodes()+impl_->integrate->nodes();
}
}
