#include "runtime_api.h"
#include "native_policy.h"
#include "fast_mode_config.h"
#include "eval_renderer.h"
#include "../../../vendor/cJSON.h"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#ifndef REK_EVAL_BACKEND
#define REK_EVAL_BACKEND ""
#endif

namespace {
struct JsonDelete{void operator()(cJSON* p)const{cJSON_Delete(p);}};
using Json=std::unique_ptr<cJSON,JsonDelete>;
Json object(){return Json(cJSON_CreateObject());}
cJSON* field(const cJSON* p,const char* key){return cJSON_GetObjectItemCaseSensitive(p,key);}
std::string str(const cJSON* p,const char* key,const char* fallback=""){
    auto* v=field(p,key);return cJSON_IsString(v)?v->valuestring:fallback;
}
int integer(const cJSON* p,const char* key,int fallback){
    auto* v=field(p,key);if(!v)return fallback;
    if(!cJSON_IsNumber(v)||v->valuedouble!=v->valueint)throw std::runtime_error(std::string("Invalid integer ")+key);
    return v->valueint;
}
void num(cJSON* o,const char* key,double value){cJSON_AddNumberToObject(o,key,value);}
void text(cJSON* o,const char* key,const std::string& value){cJSON_AddStringToObject(o,key,value.c_str());}
template<class T>void array(cJSON* o,const char* key,const T* values,int n){
    auto* a=cJSON_AddArrayToObject(o,key);for(int i=0;i<n;i++)cJSON_AddItemToArray(a,cJSON_CreateNumber(values[i]));
}
void output(cJSON* value){char* data=cJSON_PrintUnformatted(value);
    if(!data)throw std::runtime_error("JSON serialization failed");
    std::cout<<data<<'\n'<<std::flush;cJSON_free(data);
}
void cuda_ok(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void runtime_ok(int e){if(e)throw std::runtime_error(rek_native5_error());}
void policy_ok(int e){if(e)throw std::runtime_error(rek_native_policy_error());}
void policy_side_ok(int e,int side){if(e)throw std::runtime_error("policy: "+std::to_string(side)+" "+rek_native_policy_error());}

class Worker {
    int arenas=4;
    uint64_t tick=0;
    cudaStream_t stream=nullptr;
    RekNative5Runtime* runtime=nullptr;
    RekNativePolicy* policies[2]={nullptr,nullptr};
    bool deterministic[2]={true,true};
    bool raw_policy[2]={false,false};
    int recurrent_reset_ticks[2]={0,0};
    bool scripted_player=false;
    const bool gpu_scripted=std::string(REK_EVAL_BACKEND)=="semantic_cuda";
    int scripted_move=0;
    RekNative5DeviceView view{};
    RekNative5Buffers buffers{};
    float *external=nullptr,*encoded=nullptr;
    uint8_t* override_rows=nullptr;
    std::vector<void*> allocations;
    std::vector<float> host_actions;
    std::vector<uint8_t> host_override;
    std::unique_ptr<rek_eval::Renderer> renderer;
    std::string model_path;
    std::string scoring_mode;
    rek5_modes::Identity mode_identity;
    bool failed=false;
    template<class T>T* allocate(size_t count){
        T* p=nullptr;cuda_ok(cudaMalloc((void**)&p,count*sizeof(T)));
        allocations.push_back(p);cuda_ok(cudaMemsetAsync(p,0,count*sizeof(T),stream));return p;
    }
    RekNative5Snapshot snapshot(){
        RekNative5Snapshot s{};runtime_ok(rek_native5_read_snapshot(runtime,0,&s,stream));return s;
    }
    Json state(const RekNative5Snapshot& s){
        auto o=object();cJSON_AddBoolToObject(o.get(),"ok",!failed&&!s.round.failure_bits);
        if(REK_EVAL_BACKEND[0])text(o.get(),"runtimeBackend",REK_EVAL_BACKEND);
        if(gpu_scripted)text(o.get(),"scoringMode",scoring_mode);
        if(gpu_scripted){text(o.get(),"opponentController",mode_identity.opponent);text(o.get(),"observationMode",mode_identity.observation);}
        num(o.get(),"tick",double(tick));num(o.get(),"timeRemaining",s.round.time_remaining_seconds);
        num(o.get(),"completedRounds",double(s.round.completed_rounds));
        num(o.get(),"terminal",s.round.terminal);num(o.get(),"winner",s.round.round_winner);
        num(o.get(),"roundResult",s.round.round_result);num(o.get(),"roundNumber",s.round.round_number);
        num(o.get(),"failureBits",s.round.failure_bits);
        array(o.get(),"score",s.round.points,2);array(o.get(),"falls",s.round.falls,2);
        array(o.get(),"wins",s.round.wins,2);array(o.get(),"completedPoints",s.round.completed_points,2);
        array(o.get(),"actions",s.actions,2);array(o.get(),"rewards",s.rewards,2);
        array(o.get(),"mask",s.action_masks,66);array(o.get(),"raw",s.raw_observations,446);
        if(failed||s.round.failure_bits)text(o.get(),"failure","Native physics/runtime failure");
        else cJSON_AddNullToObject(o.get(),"failure");
        return o;
    }
public:
    explicit Worker(const cJSON* config){
        if(REK_EVAL_BACKEND[0]&&str(config,"backend")!=REK_EVAL_BACKEND)
            throw std::runtime_error("Evaluator binary/backend configuration mismatch");
        if(gpu_scripted){
            const auto* fast=field(config,"fast");const auto* scoring=fast?field(fast,"scoring_mode"):nullptr;
            mode_identity=rek5_modes::configure(fast);
            if(scoring&&!cJSON_IsString(scoring))throw std::runtime_error("Invalid scoring mode type");
            scoring_mode=scoring?scoring->valuestring:"v4_spheres";
            if(scoring_mode!="v4_spheres"&&scoring_mode!="recovered_hit_rules_v1"&&scoring_mode!="recovered_hit_rules_v2")throw std::runtime_error("Invalid scoring mode");
            setenv("REK_FAST_SCORING",scoring_mode.c_str(),1);
        }
        arenas=integer(config,"arenas",4);if(arenas<=0)throw std::runtime_error("Invalid arena count");
        cuda_ok(cudaStreamCreate(&stream));
        buffers.observations=allocate<float>(arenas*223);buffers.actions=allocate<float>(arenas);
        buffers.rewards=allocate<float>(arenas);buffers.terminals=allocate<float>(arenas);
        buffers.logs=allocate<RekNative5Log>(arenas);buffers.log_stride_bytes=sizeof(RekNative5Log);
        external=allocate<float>(arenas*2);encoded=allocate<float>(arenas*446);
        override_rows=allocate<uint8_t>(arenas*2);host_actions.resize(arenas*2,1);host_override.resize(arenas*2);
        model_path=str(config,"model_path");
        const auto physics=str(config,"physics_export_path"),assets=str(config,"assets_path"),
            features=str(config,"motion_features_path"),encoder=str(config,"controller_encoder_path"),
            decoder=str(config,"controller_decoder_path");
        RekNative5Config cfg{};cfg.abi_version=REK_NATIVE5_RUNTIME_ABI;
        cfg.model_path=model_path.c_str();cfg.physics_export_path=physics.c_str();cfg.assets_path=assets.c_str();
        cfg.motion_features_path=features.c_str();cfg.controller_encoder_path=encoder.c_str();cfg.controller_decoder_path=decoder.c_str();
        cfg.arenas=arenas;cfg.seed=integer(config,"seed",73);
        const int segment_ticks=integer(config,"locomotion_segment_ticks",1);
        if(segment_ticks<=0)throw std::runtime_error("Invalid locomotion segment ticks");
        cfg.locomotion_segment_ticks=uint32_t(segment_ticks);
        cfg.round_seconds=float(integer(config,"round_seconds",0));
        const uint32_t durations[]={35,27,31,45,32,45,157,145,158,139,134,138,73,75,68,71,103};
        std::copy(durations,durations+17,cfg.move_duration_ticks);
        if(auto* configured=field(config,"move_duration_ticks")){
            if(!cJSON_IsArray(configured)||cJSON_GetArraySize(configured)!=17)
                throw std::runtime_error("Expected 17 move durations");
            for(int i=0;i<17;i++){
                auto* value=cJSON_GetArrayItem(configured,i);
                if(!cJSON_IsNumber(value)||value->valuedouble!=value->valueint||value->valueint<=0)
                    throw std::runtime_error("Invalid move duration");
                cfg.move_duration_ticks[i]=uint32_t(value->valueint);
            }
        }
        runtime=rek_native5_create(&cfg,&buffers,stream);if(!runtime)throw std::runtime_error(rek_native5_error());
        runtime_ok(rek_native5_bind_external_actions(runtime,external,override_rows,stream));
        runtime_ok(rek_native5_get_device_view(runtime,&view));
        reset();
    }
    void reset(){
        // Engine failures require process recreation, never a hidden repair/reset.
        if(failed)throw std::runtime_error("Recreate evaluator after engine failure");
        runtime_ok(rek_native5_reset(runtime,stream));
        for(auto* p:policies)if(p)policy_ok(rek_native_policy_reset(p,stream));
        runtime_ok(rek_native5_check_status(runtime,stream));tick=0;scripted_move=0;
    }
    Json request(const cJSON* command){
        auto reply=object();num(reply.get(),"id",integer(command,"id",0));
        cJSON_AddBoolToObject(reply.get(),"ok",true);
        const auto op=str(command,"op");
        if(op=="policy"){
            const int side=integer(command,"side",1);if(side<0||side>1)throw std::runtime_error("Invalid fighter side");
            const auto path=str(command,"checkpoint"),hash=str(command,"sha256");
            RekNativePolicy* replacement=nullptr;
            if(!path.empty()){
                RekNativePolicyConfig cfg{};cfg.abi_version=REK_NATIVE_POLICY_ABI;
                cfg.checkpoint_path=path.c_str();cfg.expected_sha256=hash.empty()?nullptr:hash.c_str();
                cfg.hidden_size=integer(command,"hiddenSize",256);cfg.num_layers=integer(command,"layers",2);
                cfg.batch=arenas;cfg.precision=str(command,"precision","bf16")=="fp32"?REK_NATIVE_POLICY_FP32:REK_NATIVE_POLICY_BF16;
                cfg.seed=integer(command,"seed",73);
                cfg.legacy_fast_hidden=integer(command,"legacyFastHidden",0);
                replacement=rek_native_policy_create(&cfg,stream);
                if(!replacement)throw std::runtime_error(rek_native_policy_error());
            }
            cuda_ok(cudaStreamSynchronize(stream));rek_native_policy_destroy(policies[side]);policies[side]=replacement;
            if(side==0)scripted_player=cJSON_IsTrue(field(command,"scripted"));
            deterministic[side]=!cJSON_IsFalse(field(command,"deterministic"));
            const auto encoding=str(command,"observationEncoding","scaled_polar_xy");
            if(encoding!="scaled_polar_xy"&&encoding!="raw223")throw std::runtime_error("Unsupported observation encoding");
            raw_policy[side]=encoding=="raw223";
            recurrent_reset_ticks[side]=integer(command,"recurrentResetTicks",0);
            if(recurrent_reset_ticks[side]<0)throw std::runtime_error("Invalid recurrent reset interval");
            if(replacement)text(reply.get(),"sha256",rek_native_policy_sha256(replacement));
            reset();
        }else if(op=="step"){
            if(failed)throw std::runtime_error("Engine is stopped after failure");
            int steps=integer(command,"steps",1),action=integer(command,"action",1);
            int human_side=integer(command,"humanSide",0);
            if(human_side<0||human_side>1)throw std::runtime_error("Invalid human side");
            if(steps<1||steps>512||action<0||action>=33)throw std::runtime_error("Invalid step/actions");
            auto* events=cJSON_AddArrayToObject(reply.get(),"rounds");
            auto before=snapshot();uint64_t rounds=before.round.completed_rounds;
            const bool stop_at_round=cJSON_IsTrue(field(command,"stopAtRound"));
            for(int i=0;i<steps;i++){
                int actual_action=action;
                if(scripted_player&&!policies[0]&&!gpu_scripted){
                    const auto current=snapshot();const float* o=current.raw_observations;
                    const double n=std::sqrt(double(o[3])*o[3]+double(o[4])*o[4]+double(o[5])*o[5]+double(o[6])*o[6]);
                    const double w=o[3]/n,x=o[4]/n,y=o[5]/n,z=o[6]/n;
                    const double fx=1-2*(y*y+z*z),fy=2*(w*z+x*y),yaw=std::atan2(fy,fx);
                    const double dx=double(o[86])-o[0],dy=double(o[87])-o[1],distance=std::sqrt(dx*dx+dy*dy);
                    const double bearing=std::atan2(-std::sin(yaw)*dx+std::cos(yaw)*dy,std::cos(yaw)*dx+std::sin(yaw)*dy);
                    int preferred=16+scripted_move;
                    if(distance<.72)preferred=3;if(distance>1.25)preferred=2;
                    if(std::abs(bearing)>.16)preferred=bearing>0?6:7;if(o[79]!=0)preferred=1;
                    actual_action=current.action_masks[preferred]?preferred:(current.action_masks[1]?1:0);
                    if(actual_action>=16&&actual_action<32)scripted_move=(actual_action-15)%16;
                }
                for(int a=0;a<arenas;a++){
                    host_actions[a*2]=a==0&&human_side==0?float(actual_action):1.f;
                    if(scripted_player&&!policies[0])host_actions[a*2]=a==0?float(actual_action):1.f;
                    host_actions[a*2+1]=a==0&&human_side==1?float(action):1.f;
                    // Compact runtime override2 invokes the identical GPU
                    // scripted opponent on fighter0 for side-reversed tests.
                    host_override[a*2]=gpu_scripted&&scripted_player&&!policies[0]?2:1;
                    host_override[a*2+1]=policies[1]||human_side==1?1:0;
                }
                cuda_ok(cudaMemcpyAsync(external,host_actions.data(),host_actions.size()*sizeof(float),cudaMemcpyHostToDevice,stream));
                cuda_ok(cudaMemcpyAsync(override_rows,host_override.data(),host_override.size(),cudaMemcpyHostToDevice,stream));
                if(policies[0]||policies[1]){
                    runtime_ok(rek_native5_encode_fighter_observations(runtime,encoded,stream));
                    for(int side=0;side<2;side++)if(policies[side]){
                        if(recurrent_reset_ticks[side]&&tick%recurrent_reset_ticks[side]==0)
                            policy_side_ok(rek_native_policy_reset_recurrent(policies[side],stream),side);
                        policy_side_ok(rek_native_policy_step_rows(policies[side],raw_policy[side]?view.raw_observations:encoded,
                            view.action_masks,view.terminals,external,side,2,deterministic[side],stream),side);
                    }
                }
                runtime_ok(rek_native5_step(runtime,stream));tick++;
                try{
                    runtime_ok(rek_native5_check_status(runtime,stream));
                    for(int side=0;side<2;side++)if(policies[side])policy_side_ok(rek_native_policy_check_status(policies[side],stream),side);
                }catch(...){failed=true;throw;}
                auto s=snapshot();
                if(s.round.completed_rounds!=rounds){
                    cJSON_AddItemToArray(events,state(s).release());rounds=s.round.completed_rounds;
                    if(stop_at_round)break;
                }
            }
        }else if(op=="reset")reset();
        else if(op=="frame"){
            if(!renderer)renderer=std::make_unique<rek_eval::Renderer>(model_path.c_str());
            auto s=snapshot();text(reply.get(),"png",renderer->frame(s.qpos));return reply;
        }else if(op!="snapshot")throw std::runtime_error("Unknown worker operation");
        cJSON_AddItemToObject(reply.get(),"state",state(snapshot()).release());return reply;
    }
    ~Worker(){
        renderer.reset();for(auto* p:policies)rek_native_policy_destroy(p);
        if(runtime)rek_native5_close(runtime);for(auto* p:allocations)cudaFree(p);
        if(stream)cudaStreamDestroy(stream);
    }
};
}

int main(int argc,char** argv){
    try{
        if(argc!=3||std::string(argv[1])!="--config")throw std::runtime_error("Usage: rek-eval-worker --config FILE.json");
        std::ifstream file(argv[2]);if(!file)throw std::runtime_error("Cannot open evaluator config");
        std::stringstream content;content<<file.rdbuf();Json config(cJSON_Parse(content.str().c_str()));
        if(!config)throw std::runtime_error("Invalid evaluator config");
        Worker worker(config.get());auto ready=object();text(ready.get(),"event","ready");
        if(REK_EVAL_BACKEND[0])text(ready.get(),"runtimeBackend",REK_EVAL_BACKEND);
        output(ready.get());
        std::string line;
        while(std::getline(std::cin,line)){
            if(line.size()>16384)throw std::runtime_error("Worker command too large");
            Json command(cJSON_Parse(line.c_str()));auto error=object();
            try{if(!command)throw std::runtime_error("Invalid JSON command");auto reply=worker.request(command.get());output(reply.get());}
            catch(const std::exception& e){num(error.get(),"id",command?integer(command.get(),"id",0):0);
                cJSON_AddBoolToObject(error.get(),"ok",false);text(error.get(),"error",e.what());output(error.get());}
        }
        return 0;
    }catch(const std::exception& e){auto error=object();text(error.get(),"event","fatal");text(error.get(),"error",e.what());
        output(error.get());return 1;}
}
