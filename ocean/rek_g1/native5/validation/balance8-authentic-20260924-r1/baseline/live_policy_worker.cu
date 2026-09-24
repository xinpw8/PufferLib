#include "../../../vendor/cJSON.h"
#include "policy_feature_mask.h"
#include "owned_yaw_observation.h"
#include "observable_balance.h"
#include "observable_prev_action.h"
#ifndef REK_LIVE_PROTOCOL_TEST
#include "native_policy.h"
#include <cuda_runtime.h>
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>

namespace {
constexpr char PROTOCOL[]="rek.live_policy.v1";
constexpr size_t MAX_LINE=65536;
using Json=std::unique_ptr<cJSON,decltype(&cJSON_Delete)>;
struct Invalid:std::runtime_error { explicit Invalid(const char* code):std::runtime_error(code){} };
void require(bool ok,const char* code){if(!ok)throw Invalid(code);}
Json object(){return Json(cJSON_CreateObject(),cJSON_Delete);}
void str(cJSON* j,const char* key,const std::string& value){cJSON_AddStringToObject(j,key,value.c_str());}
void number(cJSON* j,const char* key,double value){cJSON_AddNumberToObject(j,key,value);}
void boolean(cJSON* j,const char* key,bool value){cJSON_AddBoolToObject(j,key,value);}
void emit(cJSON* j){char* text=cJSON_PrintUnformatted(j);if(!text)throw std::runtime_error("json_output_allocation");std::cout<<text<<'\n'<<std::flush;cJSON_free(text);if(!std::cout)throw std::runtime_error("stdout_closed");}
const cJSON* field(const cJSON* j,const char* name){const auto* p=cJSON_GetObjectItemCaseSensitive(j,name);require(p,"missing_field");return p;}
std::string string_field(const cJSON* j,const char* name){auto* p=field(j,name);require(cJSON_IsString(p)&&p->valuestring,"string_required");return p->valuestring;}
bool digest(const std::string& value){return value.size()==64&&std::all_of(value.begin(),value.end(),[](unsigned char c){return(c>='0'&&c<='9')||(c>='a'&&c<='f');});}
struct Request {
    enum Kind { Step, Reset, Close } kind;
    uint64_t seq=0;
    std::string round;
    bool terminal=false;
    std::array<float,223> observation{};
    std::array<uint8_t,33> mask{};
    int legal=0;
};
Request parse(const std::string& text,const char* schema){
    require(text.find('\0')==std::string::npos,"embedded_nul");
    const char* end=nullptr;
    Json j(cJSON_ParseWithOpts(text.c_str(),&end,1),cJSON_Delete);
    require(j&&cJSON_IsObject(j.get()),"invalid_json_object");
    const std::string type=string_field(j.get(),"type");Request r;
    if(type=="step")r.kind=Request::Step;else if(type=="reset")r.kind=Request::Reset;else if(type=="close")r.kind=Request::Close;else throw Invalid("unknown_type");
    const std::set<std::string> allowed=r.kind==Request::Step?std::set<std::string>{"type","seq","round_id","observation_schema","observation","mask","terminal"}:
        r.kind==Request::Reset?std::set<std::string>{"type","seq","round_id"}:std::set<std::string>{"type","seq"};
    std::set<std::string> seen;
    for(const cJSON* p=j->child;p;p=p->next){require(p->string,"invalid_field");require(seen.insert(p->string).second,"duplicate_field");require(allowed.count(p->string),"unsupported_field");}
    const auto* seq=field(j.get(),"seq");
    require(cJSON_IsNumber(seq)&&std::isfinite(seq->valuedouble)&&seq->valuedouble>=0&&seq->valuedouble<=9007199254740991.0&&seq->valuedouble==std::floor(seq->valuedouble),"invalid_seq");
    r.seq=uint64_t(seq->valuedouble);
    if(r.kind==Request::Close)return r;
    r.round=string_field(j.get(),"round_id");require(digest(r.round),"invalid_round_id");
    if(r.kind==Request::Reset)return r;
    require(string_field(j.get(),"observation_schema")==schema,"observation_schema_mismatch");
    const auto* terminal=field(j.get(),"terminal");require(cJSON_IsBool(terminal),"terminal_boolean_required");r.terminal=cJSON_IsTrue(terminal);
    const auto* obs=field(j.get(),"observation");require(cJSON_IsArray(obs)&&cJSON_GetArraySize(obs)==223,"observation_shape");
    for(int i=0;i<223;i++){const auto* x=cJSON_GetArrayItem(obs,i);require(cJSON_IsNumber(x)&&std::isfinite(x->valuedouble)&&std::fabs(x->valuedouble)<=std::numeric_limits<float>::max(),"observation_value");r.observation[i]=float(x->valuedouble);}
    if(std::strcmp(schema,rek_observable_balance::kSchema)==0||std::strcmp(schema,rek_observable_prev_action::kSchema)==0){
        // Incoming rows contain measured base features only. The worker owns
        // sampled-action history; callers cannot supply or spoof these cells.
        for(int i=0;i<223;i++)if(!rek_observable_balance::structurally_available(i))
            require(r.observation[i]==0,"observable_balance_padding");
        for(int b:{0,86}){
            for(int i:{71,74,75,76})require(r.observation[b+i]==0||r.observation[b+i]==1,"observable_balance_availability");
            require(r.observation[b+72]>=0&&r.observation[b+72]<=1,"observable_balance_tilt");
            if(!r.observation[b+74])for(int i=13;i<=41;i++)require(r.observation[b+i]==0,"observable_balance_missing_joints");
            if(!r.observation[b+75])for(int i=42;i<=70;i++)require(r.observation[b+i]==0,"observable_balance_missing_rates");
        }
        for(int i:{202,203,204,205})require(r.observation[i]==0||r.observation[i]==1,"observable_balance_availability");
        if(!r.observation[202])require(r.observation[204]==0&&r.observation[205]==0,"observable_balance_missing_referee");
    }else if(rek_owned_yaw::enabled(schema)){
        const float value=r.observation[rek_owned_yaw::kColumn];
        require((value==-1||value==0||value==1)&&(!r.terminal||value==0),"owned_yaw_intent_value");
    }
    const auto* mask=field(j.get(),"mask");require(cJSON_IsArray(mask)&&cJSON_GetArraySize(mask)==33,"mask_shape");
    for(int i=0;i<33;i++){const auto* x=cJSON_GetArrayItem(mask,i);if(cJSON_IsBool(x))r.mask[i]=cJSON_IsTrue(x);else{require(cJSON_IsNumber(x)&&(x->valuedouble==0||x->valuedouble==1),"mask_value");r.mask[i]=uint8_t(x->valuedouble);}r.legal+=r.mask[i];}
    require(r.terminal||r.legal>0,"empty_action_mask");return r;
}
bool read_line(std::string& line,bool& overlong){line.clear();overlong=false;char c;bool any=false;while(std::cin.get(c)){any=true;if(c=='\n')break;if(line.size()<MAX_LINE)line.push_back(c);else overlong=true;}if(!line.empty()&&line.back()=='\r')line.pop_back();return any;}
struct ProtocolState {
    bool have_seq=false;
    uint64_t seq=0;
    std::string round;
    bool reset_pending=false;
    void check(const Request& r)const{require(!have_seq||r.seq>seq,"stale_seq");}
    bool new_round(const Request& r)const{return round.empty()||round!=r.round;}
    void accept(const Request& r){seq=r.seq;have_seq=true;if(r.kind!=Request::Close)round=r.round;}
};
Json response(const char* type,const Request* r=nullptr){auto j=object();str(j.get(),"type",type);str(j.get(),"protocol",PROTOCOL);if(r){number(j.get(),"seq",double(r->seq));if(r->kind!=Request::Close)str(j.get(),"round_id",r->round);}return j;}

#ifndef REK_LIVE_PROTOCOL_TEST
void cuda_ok(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
void policy_ok(int status){if(status)throw std::runtime_error(rek_native_policy_error());}
struct Engine {
    cudaStream_t stream=nullptr;
    cudaGraph_t graph=nullptr;
    cudaGraphExec_t executable=nullptr;
    cudaEvent_t begin=nullptr,end=nullptr;
    float *obs=nullptr,*actions=nullptr,*terminals=nullptr;
    uint8_t* masks=nullptr;
    std::unique_ptr<RekNativePolicy,decltype(&rek_native_policy_destroy)> policy{nullptr,rek_native_policy_destroy};
    std::string sha,device;
    uint64_t seed;
    bool deterministic;
    bool previous_action_enabled;
    rek_observable_prev_action::History previous_action{},last_input_history{};
    rek_policy_features::Mask features;
    const char* selection()const{return deterministic?"argmax":"sampled";}
    Engine(const char* checkpoint,const std::string& expected,uint64_t rng_seed,bool greedy,
           const char* feature_path,bool action_feedback):seed(rng_seed),deterministic(greedy),previous_action_enabled(action_feedback),features(rek_policy_features::load(feature_path)){
        require(digest(expected),"invalid_checkpoint_sha256");cuda_ok(cudaSetDevice(0));
        cudaDeviceProp prop{};cuda_ok(cudaGetDeviceProperties(&prop,0));device=prop.name;
        cuda_ok(cudaStreamCreate(&stream));cuda_ok(cudaEventCreate(&begin));cuda_ok(cudaEventCreate(&end));
        cuda_ok(cudaMalloc(&obs,223*sizeof(float)));cuda_ok(cudaMalloc(&masks,33));cuda_ok(cudaMalloc(&actions,sizeof(float)));cuda_ok(cudaMalloc(&terminals,sizeof(float)));
        cuda_ok(cudaMemsetAsync(terminals,0,sizeof(float),stream));
        RekNativePolicyConfig cfg{};cfg.abi_version=REK_NATIVE_POLICY_ABI;cfg.checkpoint_path=checkpoint;cfg.expected_sha256=expected.c_str();cfg.hidden_size=256;cfg.num_layers=2;cfg.batch=1;cfg.precision=REK_NATIVE_POLICY_BF16;cfg.seed=seed;
        policy.reset(rek_native_policy_create(&cfg,stream));if(!policy)throw std::runtime_error(rek_native_policy_error());sha=rek_native_policy_sha256(policy.get());require(sha==expected,"checkpoint_sha256_mismatch");
        cuda_ok(cudaStreamSynchronize(stream));cuda_ok(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        policy_ok(rek_native_policy_step_rows(policy.get(),obs,masks,terminals,actions,0,1,deterministic,stream));
        cuda_ok(cudaStreamEndCapture(stream,&graph));cuda_ok(cudaGraphInstantiate(&executable,graph,0));
    }
    ~Engine(){if(stream)cudaStreamSynchronize(stream);if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);policy.reset();if(obs)cudaFree(obs);if(masks)cudaFree(masks);if(actions)cudaFree(actions);if(terminals)cudaFree(terminals);if(begin)cudaEventDestroy(begin);if(end)cudaEventDestroy(end);if(stream)cudaStreamDestroy(stream);}
    void reset(){policy_ok(rek_native_policy_reset_recurrent(policy.get(),stream));cuda_ok(cudaStreamSynchronize(stream));rek_observable_prev_action::clear(previous_action);rek_observable_prev_action::clear(last_input_history);}
    int infer(const Request& r,float& gpu_ms){
        float action=-1;
        auto input=r.observation;
        if(previous_action_enabled){last_input_history=previous_action;require(rek_observable_prev_action::write(input.data(),previous_action),"invalid_previous_sample_history");}
        rek_policy_features::apply(input.data(),features);
        cuda_ok(cudaEventRecord(begin,stream));
        cuda_ok(cudaMemcpyAsync(obs,input.data(),223*sizeof(float),cudaMemcpyHostToDevice,stream));
        cuda_ok(cudaMemcpyAsync(masks,r.mask.data(),33,cudaMemcpyHostToDevice,stream));
        cuda_ok(cudaGraphLaunch(executable,stream));
        cuda_ok(cudaMemcpyAsync(&action,actions,sizeof(float),cudaMemcpyDeviceToHost,stream));
        cuda_ok(cudaEventRecord(end,stream));cuda_ok(cudaStreamSynchronize(stream));
        policy_ok(rek_native_policy_check_status(policy.get(),stream));cuda_ok(cudaEventElapsedTime(&gpu_ms,begin,end));
        require(std::isfinite(action)&&action==std::floor(action)&&action>=0&&action<33,"invalid_policy_action");
        require(r.mask[int(action)]!=0,"masked_policy_action");
        if(previous_action_enabled)require(rek_observable_prev_action::record(previous_action,action),"invalid_previous_sample_history");
        return int(action);
    }
};
#endif
}

int main(int argc,char** argv){
    try{
        const char* selected_schema=std::getenv("REK_OBSERVATION_SCHEMA");
        const bool action_feedback=selected_schema&&std::strcmp(selected_schema,rek_observable_prev_action::kSchema)==0;
        const char* observation_schema=action_feedback?rek_observable_prev_action::kSchema:
            selected_schema&&std::strcmp(selected_schema,rek_observable_balance::kSchema)==0
            ?rek_observable_balance::kSchema:rek_owned_yaw::schema(rek_owned_yaw::enabled(selected_schema));
#ifndef REK_LIVE_PROTOCOL_TEST
        require(argc>=3&&argc<=6,"usage_checkpoint_sha256_optional_seed_selection_feature_mask");
        uint64_t seed=73;
        if(argc>=4){char* end=nullptr;require(argv[3][0]>='0'&&argv[3][0]<='9',"invalid_seed");seed=std::strtoull(argv[3],&end,10);require(end&&!*end&&seed<=9007199254740991ULL,"invalid_seed");}
        const std::string selection=argc>=5?argv[4]:"sampled";
        require(selection=="sampled"||selection=="argmax","invalid_selection");
        Engine engine(argv[1],argv[2],seed,selection=="argmax",argc>=6?argv[5]:nullptr,action_feedback);
        auto ready=response("ready");str(ready.get(),"checkpoint_sha256",engine.sha);str(ready.get(),"observation_schema",observation_schema);str(ready.get(),"selection",engine.selection());str(ready.get(),"feature_mask_sha256",engine.features.sha256);str(ready.get(),"precision","bf16");str(ready.get(),"device",engine.device);number(ready.get(),"seed",double(seed));number(ready.get(),"observations",223);number(ready.get(),"actions",33);number(ready.get(),"hidden_size",256);number(ready.get(),"num_layers",2);boolean(ready.get(),"native_cuda",true);boolean(ready.get(),"environment_stepping",false);emit(ready.get());
#else
        require(argc==1,"protocol_test_takes_no_checkpoint");auto ready=response("protocol_ready");boolean(ready.get(),"inference_available",false);emit(ready.get());
#endif
        ProtocolState state;std::string line;bool overlong=false;uint64_t decisions=0;
        while(read_line(line,overlong)){
            Request r;bool parsed=false;
            try{
                require(!overlong,"line_too_long");r=parse(line,observation_schema);parsed=true;state.check(r);
            }catch(const Invalid& e){auto error=response("error",parsed?&r:nullptr);str(error.get(),"code",e.what());boolean(error.get(),"action_available",false);emit(error.get());continue;}
            if(r.kind==Request::Close){state.accept(r);auto j=response("closed",&r);emit(j.get());return 0;}
            const bool changed=state.new_round(r);
            const bool clear=changed||r.kind==Request::Reset||r.terminal;
            const auto started=std::chrono::steady_clock::now();
#ifndef REK_LIVE_PROTOCOL_TEST
            if(clear)engine.reset();
#endif
            const char* response_type=r.kind==Request::Reset?"reset":r.terminal?"terminal":"action";
#ifdef REK_LIVE_PROTOCOL_TEST
            if(r.kind==Request::Step&&!r.terminal)response_type="validated";
#endif
            auto j=response(response_type,&r);boolean(j.get(),"recurrent_reset",clear||state.reset_pending);boolean(j.get(),"round_changed",changed);number(j.get(),"legal_actions",r.legal);
            if(r.kind==Request::Step&&!r.terminal){
#ifndef REK_LIVE_PROTOCOL_TEST
                float gpu_ms=0;int action=engine.infer(r,gpu_ms);number(j.get(),"action",action);number(j.get(),"gpu_ms",gpu_ms);number(j.get(),"decision_index",double(++decisions));str(j.get(),"checkpoint_sha256",engine.sha);str(j.get(),"observation_schema",observation_schema);str(j.get(),"selection",engine.selection());str(j.get(),"feature_mask_sha256",engine.features.sha256);str(j.get(),"precision","bf16");
                if(action_feedback){
                    auto* memory=cJSON_AddObjectToObject(j.get(),"policy_memory_input");
                    str(memory,"source","previous_successful_sampler_output_before_feature_mask");
                    boolean(memory,"available",engine.last_input_history.available!=0);
                    if(engine.last_input_history.available)number(memory,"action",engine.last_input_history.action);else cJSON_AddNullToObject(memory,"action");
                    boolean(memory,"execution_or_acceptance_claim",false);
                }
#else
                boolean(j.get(),"inference_available",false);
#endif
                state.reset_pending=false;
            }else state.reset_pending=true;
            number(j.get(),"latency_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count());state.accept(r);emit(j.get());
        }
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"live policy worker fatal: %s\n",e.what());try{auto j=response("fatal");str(j.get(),"code","worker_failure");boolean(j.get(),"action_available",false);emit(j.get());}catch(...){}return 2;}
}
