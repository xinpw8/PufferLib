#include "collision_schedule.h"
#include "constraint_schedule.h"
#include "kernel_program.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void checked(CUresult result,const char* where){if(result!=CUDA_SUCCESS){const char* error=nullptr;cuGetErrorString(result,&error);throw std::runtime_error(std::string(where)+": "+(error?error:"CUDA error"));}}
std::vector<float> download(const rek_mjgpu::ModelData& data,const char* name){
    const auto& storage=data.arrays().at(name);if(storage.element!=rek_mjgpu::Element::F32)throw std::runtime_error("Expected float field");
    std::vector<float> result(storage.bytes/4);if(storage.bytes)checked(cuMemcpyDtoH(result.data(),reinterpret_cast<CUdeviceptr>(storage.view.data),storage.bytes),"download field");
    for(float value:result)if(!std::isfinite(value))throw std::runtime_error(std::string("Nonfinite partial forward output: ")+name);
    return result;
}
}
int main(int argc,char** argv){try{
    if(argc!=3)throw std::runtime_error("Usage: forward-probe MODEL.xml CATALOG.json");
    checked(cuInit(0),"cuInit");CUdevice dev;CUcontext context;CUstream stream;
    checked(cuDeviceGet(&dev,0),"device");checked(cuDevicePrimaryCtxRetain(&context,dev),"context");checked(cuCtxSetCurrent(context),"set context");checked(cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING),"stream");
    char error[2048]={};std::unique_ptr<mjModel,decltype(&mj_deleteModel)> model(mj_loadXML(argv[1],nullptr,error,sizeof(error)),mj_deleteModel);if(!model)throw std::runtime_error(error);model->opt.timestep=.002;
    {rek_mjgpu::ModelDataConfig config;config.nworld=4;rek_mjgpu::ModelData data(model.get(),config);
    rek_mjgpu::ScheduleSpec spec;
    rek_mjgpu::append_kinematics_com(spec,data);rek_mjgpu::append_crb(spec,data);rek_mjgpu::append_camlight(spec,data);
    rek_mjgpu::append_collision(spec,data);rek_mjgpu::append_constraints(spec,data);rek_mjgpu::append_transmission(spec,data);
    rek_mjgpu::append_velocity_forces(spec,data);rek_mjgpu::append_actuation(spec,data);rek_mjgpu::append_smooth_acceleration(spec,data);
    std::vector<std::string> labels;for(const auto& node:spec){
        if(node.kind==rek_mjgpu::ScheduleNode::Kind::Kernel){if(node.bounds.size)labels.push_back(node.module+":"+node.entry_prefix);}
        else if(data.byte_size(node.destination))labels.push_back((node.kind==rek_mjgpu::ScheduleNode::Kind::Copy?"copy:":"zero:")+node.destination);
    }
    rek_mjgpu::KernelProgram program(data,argv[2],spec);if(labels.size()!=program.nodes())throw std::runtime_error("Probe label/node mismatch");
    for(std::size_t i=0;i<program.nodes();i++){
        std::fprintf(stderr,"node %zu/%zu %s\n",i,program.nodes(),labels[i].c_str());std::fflush(stderr);
        program.launch_one(i,stream);checked(cuStreamSynchronize(stream),"partial forward node synchronize");
    }
    std::size_t values=0;double max_acc=0,min_inverse=1e300;
    for(const char* name:{"d.qpos","d.qvel","d.geom_xpos","d.geom_xmat","d.cinert","d.cdof","d.qM","d.qLD","d.qLDiagInv",
        "d.actuator_length","d.actuator_moment","d.actuator_velocity","d.qfrc_bias","d.qfrc_passive","d.actuator_force","d.qfrc_actuator","d.qfrc_smooth","d.qacc_smooth"}){
        const auto result=download(data,name);values+=result.size();
        if(std::string(name)=="d.qacc_smooth")for(float value:result)max_acc=std::max(max_acc,std::abs(double(value)));
        if(std::string(name)=="d.qLDiagInv")for(float value:result)min_inverse=std::min(min_inverse,double(value));
    }
    if(min_inverse<=0)throw std::runtime_error("Nonpositive inverse mass-factor diagonal");
    const auto capacity=rek_mjgpu::inspect_constraint_capacity(data,stream);
    std::printf("{\"test\":\"actual_rek_native_gpu_partial_forward\",\"status\":\"%s\",\"worlds\":4,\"schedule_nodes\":%zu,\"checked_finite_values\":%zu,\"collisions\":%d,\"contacts\":%d,\"max_constraint_rows\":%d,\"max_constraint_nonzeros\":%d,\"capacity_overflow\":%s,\"minimum_inverse_mass_diagonal\":%.9g,\"max_abs_unconstrained_acceleration\":%.9g,\"collision_detection_tested\":true,\"cpu_physics_steps\":0,\"python_invocations\":0,\"newton_solver_tested\":false,\"integration_tested\":false,\"full_physics_step\":false,\"training_sps\":null}\n",
        capacity.overflow?"failed":"passed",program.nodes(),values,capacity.collisions,capacity.contacts,capacity.max_rows,capacity.max_nonzeros,capacity.overflow?"true":"false",min_inverse,max_acc);
    if(capacity.overflow)throw std::runtime_error("Partial forward collision/constraint capacity overflow");
    }checked(cuStreamDestroy(stream),"destroy stream");checked(cuDevicePrimaryCtxRelease(dev),"release context");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"forward-probe: %s\n",e.what());return 1;}}
