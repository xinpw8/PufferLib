#include "native_step.h"
#include "constraint_schedule.h"
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
void check(CUresult status,const char* operation){
    if(status==CUDA_SUCCESS)return;
    const char* error=nullptr;cuGetErrorString(status,&error);
    throw std::runtime_error(std::string(operation)+": "+(error?error:"CUDA error"));
}
struct Context {
    CUdevice device=0;CUcontext context=nullptr;CUstream stream=nullptr;
    Context(){check(cuInit(0),"initialize driver");check(cuDeviceGet(&device,0),"select GPU");
        check(cuDevicePrimaryCtxRetain(&context,device),"retain GPU context");check(cuCtxSetCurrent(context),"set GPU context");
        check(cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING),"create GPU stream");}
    ~Context(){if(stream)cuStreamDestroy(stream);if(context){cuCtxSetCurrent(nullptr);cuDevicePrimaryCtxRelease(device);}}
};
struct Event {
    CUevent value=nullptr;
    Event(){check(cuEventCreate(&value,CU_EVENT_DEFAULT),"create CUDA event");}
    ~Event(){if(value)cuEventDestroy(value);}
};
template<class T> std::vector<T> read(const rek_mjgpu::ModelData& data,const char* name,rek_mjgpu::Element element){
    const auto& storage=data.arrays().at(name);
    if(storage.element!=element||storage.bytes%sizeof(T))throw std::runtime_error(std::string("Unexpected probe storage: ")+name);
    std::vector<T> values(storage.bytes/sizeof(T));
    if(storage.bytes)check(cuMemcpyDtoH(values.data(),reinterpret_cast<CUdeviceptr>(storage.view.data),storage.bytes),"read diagnostic GPU state");
    return values;
}
std::vector<float> floats(const rek_mjgpu::ModelData& data,const char* name){
    auto result=read<float>(data,name,rek_mjgpu::Element::F32);
    for(float value:result)if(!std::isfinite(value))throw std::runtime_error(std::string("Nonfinite physics state: ")+name);
    return result;
}
double maximum_abs(const std::vector<float>& values){double result=0;for(float value:values)result=std::max(result,std::abs(double(value)));return result;}
void print_array(const char* name,const std::vector<float>& values){
    std::printf(",\"%s\":[",name);
    for(std::size_t i=0;i<values.size();i++)std::printf("%s%.9g",i?",":"",double(values[i]));
    std::printf("]");
}
}

int main(int argc,char** argv){
    if(argc!=5&&argc!=6){std::fprintf(stderr,"Usage: %s REK_MODEL_XML KERNEL_CATALOG_JSON CONDITIONAL_PTX CONDITIONAL_SHA256 [MAX_STEPS=1000]\n",argv[0]);return 2;}
    int completed=0;
    try {
        int maximum_steps=1000;
        if(argc==6){std::size_t consumed=0;maximum_steps=std::stoi(argv[5],&consumed);
            if(consumed!=std::string(argv[5]).size()||(maximum_steps!=1&&maximum_steps!=10&&maximum_steps!=1000))
                throw std::runtime_error("MAX_STEPS must be1,10,or1000");}
        Context context;char error[2048]={};
        std::unique_ptr<mjModel,decltype(&mj_deleteModel)> model(mj_loadXML(argv[1],nullptr,error,sizeof(error)),mj_deleteModel);
        if(!model)throw std::runtime_error(std::string("Model parse failed: ")+error);
        // Pin the native5 physics timestep. Parsing model constants is not CPU
        // physics evolution; mj_step/mj_forward are absent.
        model->opt.timestep=.002;
        rek_mjgpu::ModelDataConfig config;config.nworld=4;config.nconmax=512;config.njmax=512;
        rek_mjgpu::ModelData data(model.get(),config);
        data.zero("d.ctrl");
        const auto initial_positions=floats(data,"d.qpos"),initial_times=floats(data,"d.time");
        const auto setup_started=std::chrono::steady_clock::now();
        rek_mjgpu::NativeStep runtime(data,argv[2],argv[3],argv[4]);
        const double setup_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-setup_started).count();
        Event begin,end;double gpu_milliseconds=0,max_time_error=0,max_quaternion_error=0;
        int peak_collisions=0,peak_contacts=0,peak_rows=0,peak_nonzeros=0,peak_ccd=0;
        const auto started=std::chrono::steady_clock::now();
        for(int target:{1,10,1000}){
            if(target>maximum_steps)break;
            while(completed<target){
                check(cuEventRecord(begin.value,context.stream),"record step start");runtime.step(context.stream);
                check(cuEventRecord(end.value,context.stream),"record step end");check(cuEventSynchronize(end.value),"complete native GPU physics step");
                float milliseconds=0;check(cuEventElapsedTime(&milliseconds,begin.value,end.value),"time native GPU step");gpu_milliseconds+=milliseconds;
                completed++;
                const auto positions=floats(data,"d.qpos"),velocities=floats(data,"d.qvel"),times=floats(data,"d.time");
                const auto capacity=rek_mjgpu::inspect_constraint_capacity(data,context.stream);
                if(capacity.overflow)throw std::runtime_error("Contact/constraint capacity overflow at step "+std::to_string(completed));
                peak_collisions=std::max(peak_collisions,capacity.collisions);peak_contacts=std::max(peak_contacts,capacity.contacts);
                peak_rows=std::max(peak_rows,capacity.max_rows);peak_nonzeros=std::max(peak_nonzeros,capacity.max_nonzeros);
                if(data.contains("collision.nccd"))for(int count:read<int>(data,"collision.nccd",rek_mjgpu::Element::I32)){
                    if(count<0||count>data.integer("d.naccdmax"))throw std::runtime_error("CCD workspace overflow");
                    peak_ccd=std::max(peak_ccd,count);
                }
                for(int world=0;world<config.nworld;world++){
                    const double expected=initial_times.at(world)+completed*double(data.scalar("m.opt.timestep"));
                    const double difference=std::abs(times.at(world)-expected);max_time_error=std::max(max_time_error,difference);
                    if(difference>1e-4*(1+std::abs(expected)))throw std::runtime_error("Native simulation time did not advance at configured timestep");
                    for(int joint=0;joint<model->njnt;joint++)if(model->jnt_type[joint]==mjJNT_FREE||model->jnt_type[joint]==mjJNT_BALL){
                        int address=world*model->nq+model->jnt_qposadr[joint]+(model->jnt_type[joint]==mjJNT_FREE?3:0);
                        double norm=0;for(int c=0;c<4;c++)norm+=double(positions.at(address+c))*positions.at(address+c);
                        max_quaternion_error=std::max(max_quaternion_error,std::abs(norm-1));
                        if(std::abs(norm-1)>1e-4)throw std::runtime_error("Integration produced non-unit orientation quaternion");
                    }
                }
            }
            const auto positions=floats(data,"d.qpos"),velocities=floats(data,"d.qvel"),times=floats(data,"d.time");
            std::printf("{\"event\":\"native_full_step_milestone\",\"completed_steps\":%d,\"worlds\":%d,\"max_abs_qpos\":%.9g,\"max_abs_qvel\":%.9g,\"time_first_world\":%.9g,\"cumulative_gpu_event_milliseconds\":%.9g,\"peak_contacts\":%d,\"peak_constraint_rows\":%d,\"overflow\":false}\n",
                completed,config.nworld,maximum_abs(positions),maximum_abs(velocities),times.at(0),gpu_milliseconds,peak_contacts,peak_rows);
            std::fflush(stdout);
        }
        const auto before_positions=floats(data,"d.qpos"),before_velocities=floats(data,"d.qvel"),before_times=floats(data,"d.time");
        runtime.refresh(context.stream);check(cuStreamSynchronize(context.stream),"refresh exported spatial state");
        if(floats(data,"d.qpos")!=before_positions||floats(data,"d.qvel")!=before_velocities||floats(data,"d.time")!=before_times)
            throw std::runtime_error("Readout refresh advanced or mutated integration state");
        floats(data,"d.xpos");floats(data,"d.cvel");
        if(before_positions==initial_positions)throw std::runtime_error("Native steps left all positions untouched");
        const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        char gpu[256]={};check(cuDeviceGetName(gpu,sizeof(gpu),context.device),"read GPU identity");
        std::printf("{\"event\":\"native_full_step_result\",\"status\":\"passed\",\"gpu\":\"%s\",\"worlds\":%d,\"steps_per_world\":%d,\"phase_nodes\":%zu,\"setup_seconds\":%.9g,\"diagnostic_wall_seconds\":%.9g,\"cumulative_gpu_event_milliseconds\":%.9g,\"max_time_error_seconds\":%.9g,\"max_quaternion_norm_error\":%.9g,\"peak_collisions\":%d,\"peak_contacts\":%d,\"peak_constraint_rows\":%d,\"peak_constraint_nonzeros\":%d,\"peak_ccd_per_type\":%d,\"collision_capacity\":%d,\"ccd_capacity\":%d,\"constraint_capacity_per_world\":%d,\"zero_controls\":true,\"finite_state\":true,\"refresh_preserves_integration_state\":true,\"diagnostic_readback_every_step\":true,\"python_interpreter\":false,\"cpu_physics_steps\":0,\"full_step_exercised\":true,\"training_sps\":null",
            gpu,config.nworld,completed,runtime.phase_nodes(),setup_seconds,elapsed,gpu_milliseconds,max_time_error,max_quaternion_error,
            peak_collisions,peak_contacts,peak_rows,peak_nonzeros,peak_ccd,data.integer("d.naconmax"),data.integer("d.naccdmax"),data.integer("d.njmax"));
        print_array("qpos",before_positions);print_array("qvel",before_velocities);print_array("time",before_times);std::printf("}\n");
        return 0;
    }catch(const std::exception& error){
        std::fprintf(stderr,"Native MuJoCo full-step probe failed after %d completed steps: %s\n",completed,error.what());return 1;
    }
}
