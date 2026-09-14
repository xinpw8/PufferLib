#include "physics.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

static int forbidden_cpu_calls=0;
extern "C" void __wrap_mj_step(const mjModel*,mjData*){forbidden_cpu_calls++;std::fprintf(stderr,"FORBIDDEN mj_step\n");std::abort();}
extern "C" void __wrap_mj_forward(const mjModel*,mjData*){forbidden_cpu_calls++;std::fprintf(stderr,"FORBIDDEN mj_forward\n");std::abort();}
extern "C" void __wrap_mj_kinematics(const mjModel*,mjData*){forbidden_cpu_calls++;std::fprintf(stderr,"FORBIDDEN mj_kinematics\n");std::abort();}
namespace {
void check(cudaError_t result,const char* operation){if(result!=cudaSuccess)throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(result));}
std::vector<float> read(const float* pointer,int count){std::vector<float> values(count);check(cudaMemcpy(values.data(),pointer,size_t(count)*4,cudaMemcpyDeviceToHost),"read state");for(float v:values)if(!std::isfinite(v))throw std::runtime_error("Nonfinite physics state");return values;}
__global__ void pose_controls(const float* qpos,float* ctrl,const int* indices,int arenas){const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<arenas*58)ctrl[i]=qpos[(i/58)*72+indices[i%58]];}
__global__ void reset_first(rek5::PhysicsDescriptor data,const float* initial){const int k=threadIdx.x;if(k<72)data.qpos[k]=initial[k];if(k<70)data.qvel[k]=0;if(!k)data.time[0]=0;}
}
int main(int argc,char** argv){try{
    if(argc!=3)throw std::runtime_error("Usage: physics-mujoco-gpu-probe MODEL.xml EXPORT.json (native GPU catalog/PTX env required)");
    setenv("REK_PHYSICS_BACKEND","mujoco_cuda",1);cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"create stream");
    {std::unique_ptr<rek5::Physics,decltype(&rek5::physics_close)> p(rek5::physics_create(argv[1],argv[2],4,stream),rek5::physics_close);
    if(std::string(rek5::physics_backend_name(p.get()))!="mujoco_cuda"||p->cpu_evaluation)throw std::runtime_error("Wrong physics backend");
    int* indices=nullptr;float* initial=nullptr;uint8_t* mask=nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&indices),58*sizeof(int)),"indices");check(cudaMalloc(reinterpret_cast<void**>(&initial),72*sizeof(float)),"initial pose");check(cudaMalloc(reinterpret_cast<void**>(&mask),4),"mask");
    check(cudaMemcpyAsync(indices,p->joint_qpos.data(),58*sizeof(int),cudaMemcpyHostToDevice,stream),"indices upload");check(cudaMemcpyAsync(initial,p->initial_qpos.data(),72*sizeof(float),cudaMemcpyHostToDevice,stream),"pose upload");
    pose_controls<<<1,256,0,stream>>>(p->data.qpos,p->ctrl,indices,4);
    for(int i=0;i<10;i++)rek5::physics_step(p.get(),p->ctrl);
    check(cudaStreamSynchronize(stream),"complete ten GPU steps");rek5::physics_check_status(p.get());
    const auto qpos_before=read(p->data.qpos,4*72),qvel_before=read(p->data.qvel,4*70),time_before=read(p->data.time,4);
    for(float t:time_before)if(std::abs(t-.020f)>1e-6f)throw std::runtime_error("Wrong native GPU timestep");
    const uint8_t selected[4]={1,0,0,0};check(cudaMemcpyAsync(mask,selected,4,cudaMemcpyHostToDevice,stream),"selected mask");
    reset_first<<<1,128,0,stream>>>(p->data,initial);rek5::physics_forward_selected(p.get(),mask);check(cudaStreamSynchronize(stream),"selected GPU reset");
    const auto qp=read(p->data.qpos,4*72),qv=read(p->data.qvel,4*70),time=read(p->data.time,4);
    for(int a=1;a<4;a++){for(int k=0;k<72;k++)if(qp[a*72+k]!=qpos_before[a*72+k])throw std::runtime_error("Reset changed unselected qpos");for(int k=0;k<70;k++)if(qv[a*70+k]!=qvel_before[a*70+k])throw std::runtime_error("Reset changed unselected qvel");if(time[a]!=time_before[a])throw std::runtime_error("Reset changed unselected clock");}
    for(int k=0;k<72;k++)if(qp[k]!=p->initial_qpos[k])throw std::runtime_error("Reset initial pose mismatch");for(int k=0;k<70;k++)if(qv[k]!=0)throw std::runtime_error("Reset velocity mismatch");if(time[0]!=0)throw std::runtime_error("Reset clock mismatch");
    check(cudaMemsetAsync(mask,0,4,stream),"zero reset mask");rek5::physics_forward_selected(p.get(),mask);check(cudaStreamSynchronize(stream),"unselected reset");
    if(read(p->data.qpos,4*72)!=qp||read(p->data.qvel,4*70)!=qv||read(p->data.time,4)!=time)throw std::runtime_error("Empty reset mask changed integration state");
    rek5::physics_check_status(p.get());auto stats=rek5::physics_stats(p.get());int peak=0;for(int a=0;a<4;a++)peak=std::max(peak,stats[a*4]);
    check(cudaFree(mask),"free mask");check(cudaFree(initial),"free initial");check(cudaFree(indices),"free indices");
    std::printf("{\"test\":\"native_mujoco_physics_adapter_wrapped_cpu_calls\",\"status\":\"passed\",\"backend\":\"mujoco_cuda\",\"worlds\":4,\"gpu_substeps\":10,\"simulated_seconds\":0.020,\"cpu_step_forward_kinematics_calls\":%d,\"python_runtime\":false,\"selected_reset_isolated\":true,\"empty_reset_preserved_integration_state\":true,\"peak_contacts_per_arena\":%d,\"training_sps\":null}\n",forbidden_cpu_calls,peak);
    }check(cudaStreamDestroy(stream),"destroy stream");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"physics-mujoco-gpu-probe: %s\n",e.what());return 1;}}
