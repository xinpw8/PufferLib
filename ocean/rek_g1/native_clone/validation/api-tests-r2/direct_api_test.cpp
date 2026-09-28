// One fresh process per scenario. Executes one real full-runtime control tick.
#include "runtime_api.h"
#include "g1_semantic_scheduler_cuda.h"
#include "vendor/cJSON.h"
#include <cuda_runtime.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool b,const char* s){if(!b)throw std::runtime_error(s);}
static void cu(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static void rt(int e){if(e)throw std::runtime_error(rek_native5_error());}
struct DeleteJson{void operator()(cJSON* p)const{cJSON_Delete(p);}};
using Json=std::unique_ptr<cJSON,DeleteJson>;
static const cJSON* field(const cJSON* p,const char* name){return cJSON_GetObjectItemCaseSensitive(p,name);}
static std::string text(const cJSON* p,const char* name){auto* x=field(p,name);require(cJSON_IsString(x),name);return x->valuestring;}
static int integer(const cJSON* p,const char* name){auto* x=field(p,name);require(cJSON_IsNumber(x)&&x->valuedouble==x->valueint,name);return x->valueint;}
struct Storage{
    cudaStream_t stream=nullptr;RekNative5Runtime* runtime=nullptr;std::vector<void*> buffers;
    Storage(){cu(cudaStreamCreate(&stream));}
    template<class T>T* alloc(size_t n){T* p=nullptr;cu(cudaMalloc((void**)&p,n*sizeof(T)));buffers.push_back(p);cu(cudaMemsetAsync(p,0,n*sizeof(T),stream));return p;}
    ~Storage(){if(runtime)rek_native5_close(runtime);for(void* p:buffers)cudaFree(p);if(stream)cudaStreamDestroy(stream);}
};
static void emit(cJSON* p){char* s=cJSON_PrintUnformatted(p);require(s,"serialize");std::cout<<s<<'\n';cJSON_free(s);}

int main(int argc,char** argv){
    try{
        require(argc==5&&std::string(argv[1])=="--config"&&std::string(argv[3])=="--scenario","--config FILE --scenario NAME required");
        const std::string name=argv[4];int direct_side=-1,bad_side=-1;bool nan=false,expected_failure=false;
        if(name=="ignore_side0_nan"){direct_side=bad_side=0;nan=true;}
        else if(name=="ignore_side1_nan"){direct_side=bad_side=1;nan=true;}
        else if(name=="ignore_side0_oob"){direct_side=bad_side=0;}
        else if(name=="ignore_side1_oob"){direct_side=bad_side=1;}
        else if(name=="reject_side0_nan_with_side1_direct"){direct_side=1;bad_side=0;nan=true;expected_failure=true;}
        else if(name=="reject_side1_oob_with_side0_direct"){direct_side=0;bad_side=1;expected_failure=true;}
        else throw std::runtime_error("Unknown scenario");
        std::ifstream input(argv[2]);require(bool(input),"read config");std::stringstream ss;ss<<input.rdbuf();Json config(cJSON_Parse(ss.str().c_str()));require(bool(config),"parse config");
        require(text(config.get(),"backend")=="mujoco","full physics config required");
        const int arenas=integer(config.get(),"arenas");require(arenas==4,"same eight-fighter motor batch required");
        const std::string model=text(config.get(),"model_path"),physics=text(config.get(),"physics_export_path"),assets=text(config.get(),"assets_path"),features=text(config.get(),"motion_features_path"),encoder=text(config.get(),"controller_encoder_path"),decoder=text(config.get(),"controller_decoder_path");
        RekNative5Config cfg{};cfg.abi_version=REK_NATIVE5_RUNTIME_ABI;cfg.arenas=arenas;cfg.seed=integer(config.get(),"seed");
        cfg.model_path=model.c_str();cfg.physics_export_path=physics.c_str();cfg.assets_path=assets.c_str();cfg.motion_features_path=features.c_str();cfg.controller_encoder_path=encoder.c_str();cfg.controller_decoder_path=decoder.c_str();
        cfg.round_seconds=float(integer(config.get(),"round_seconds"));cfg.locomotion_segment_ticks=integer(config.get(),"locomotion_segment_ticks");
        auto* durations=field(config.get(),"move_duration_ticks");require(cJSON_IsArray(durations)&&cJSON_GetArraySize(durations)==17,"17 durations required");
        for(int i=0;i<17;i++){auto* v=cJSON_GetArrayItem(durations,i);require(cJSON_IsNumber(v)&&v->valuedouble==v->valueint&&v->valueint>0,"positive integer duration");cfg.move_duration_ticks[i]=v->valueint;}
        Storage owner;const int rows=2*arenas;
        RekNative5Buffers out{};out.observations=owner.alloc<float>(arenas*223);out.actions=owner.alloc<float>(arenas);out.rewards=owner.alloc<float>(arenas);out.terminals=owner.alloc<float>(arenas);out.logs=owner.alloc<RekNative5Log>(arenas);out.log_stride_bytes=sizeof(RekNative5Log);
        auto* cats=owner.alloc<float>(rows);auto* overrides=owner.alloc<uint8_t>(rows);auto* commands=owner.alloc<RekG1CudaDirectCommand>(rows);auto* enabled=owner.alloc<uint8_t>(rows);
        owner.runtime=rek_native5_create(&cfg,&out,owner.stream);require(owner.runtime!=nullptr,rek_native5_error());rt(rek_native5_check_status(owner.runtime,owner.stream));
        std::vector<float> host_cats(rows,1.f);host_cats[bad_side]=nan?std::numeric_limits<float>::quiet_NaN():99.f;
        std::vector<uint8_t> host_overrides(rows,1),host_enabled(rows,0);host_enabled[direct_side]=1;
        std::vector<RekG1CudaDirectCommand> host_commands(rows);for(auto& c:host_commands)c.move_index=-1;
        auto& cmd=host_commands[direct_side];cmd.velocity={.25f,-.5f,.125f};cmd.rejection_velocity=cmd.velocity;
        cu(cudaMemcpyAsync(cats,host_cats.data(),rows*sizeof(float),cudaMemcpyHostToDevice,owner.stream));
        cu(cudaMemcpyAsync(overrides,host_overrides.data(),rows,cudaMemcpyHostToDevice,owner.stream));
        cu(cudaMemcpyAsync(commands,host_commands.data(),rows*sizeof(RekG1CudaDirectCommand),cudaMemcpyHostToDevice,owner.stream));
        cu(cudaMemcpyAsync(enabled,host_enabled.data(),rows,cudaMemcpyHostToDevice,owner.stream));
        rt(rek_native5_bind_external_actions(owner.runtime,cats,overrides,owner.stream));rt(rek_native5_bind_direct_commands(owner.runtime,commands,enabled,owner.stream));
        rt(rek_native5_step(owner.runtime,owner.stream));
        const int status=rek_native5_check_status(owner.runtime,owner.stream);const std::string status_error=status?rek_native5_error():"";
        RekNative5Snapshot snapshot{};rt(rek_native5_read_snapshot(owner.runtime,0,&snapshot,owner.stream));
        RekG1CudaDirectResult results[2]{};rt(rek_native5_read_command_results(owner.runtime,0,results,owner.stream));
        require(snapshot.actions[direct_side]==-1.f,"direct action must have uncategorized sentinel");
        require(results[direct_side].status==0&&!results[direct_side].move_attempted,"valid direct velocity must not be rejected or reported as move");
        if(expected_failure){require(status!=0,"invalid active categorical row was accepted");require((snapshot.round.failure_bits&16)!=0,"categorical action failure16 not recorded");}
        else{
            require(status==0,"ignored categorical value caused sticky failure");require(snapshot.round.failure_bits==0,"unexpected failure bits");
            require(snapshot.actions[1-direct_side]==1.f,"unaffected categorical row changed");
            for(float x:snapshot.qpos)require(std::isfinite(x),"qpos nonfinite");for(float x:snapshot.qvel)require(std::isfinite(x),"qvel nonfinite");
        }
        const int close_status=rek_native5_close(owner.runtime);owner.runtime=nullptr;require((close_status!=0)==expected_failure,"close sticky status mismatch");
        Json result(cJSON_CreateObject());cJSON_AddBoolToObject(result.get(),"pass",true);cJSON_AddStringToObject(result.get(),"scenario",name.c_str());cJSON_AddNumberToObject(result.get(),"direct_side",direct_side);cJSON_AddNumberToObject(result.get(),"bad_category_side",bad_side);cJSON_AddStringToObject(result.get(),"bad_category",nan?"NaN":"99");cJSON_AddBoolToObject(result.get(),"expected_categorical_failure",expected_failure);cJSON_AddNumberToObject(result.get(),"failure_bits",snapshot.round.failure_bits);cJSON_AddNumberToObject(result.get(),"runtime_check_status",status);cJSON_AddStringToObject(result.get(),"runtime_error",status_error.c_str());cJSON_AddNumberToObject(result.get(),"full_runtime_control_ticks",1);cJSON_AddStringToObject(result.get(),"scope","direct API dispatch regression; no gameplay equivalence claim");emit(result.get());return 0;
    }catch(const std::exception& e){Json result(cJSON_CreateObject());cJSON_AddBoolToObject(result.get(),"pass",false);cJSON_AddStringToObject(result.get(),"error",e.what());emit(result.get());return 1;}
}
