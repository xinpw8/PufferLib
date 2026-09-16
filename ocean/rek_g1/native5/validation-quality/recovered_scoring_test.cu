#include "../runtime_api.h"
#include "../native_policy.h"
#include "../fast_assets.h"
#include "../recovered_contact_rules.cuh"
#include "../../../../vendor/cJSON.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
RekNative5Runtime* legacy_create(const RekNative5Config*,const RekNative5Buffers*,cudaStream_t);
int legacy_step(RekNative5Runtime*,cudaStream_t);
int legacy_close(RekNative5Runtime*);
int legacy_get_device_view(RekNative5Runtime*,RekNative5DeviceView*);
int legacy_bind_action_mask(RekNative5Runtime*,uint8_t*,cudaStream_t);
int legacy_check_status(RekNative5Runtime*,cudaStream_t);
const char* legacy_error();
}
namespace {
void need(bool b,const std::string& s){if(!b)throw std::runtime_error(s);}
void ck(cudaError_t e){need(e==cudaSuccess,cudaGetErrorString(e));}
void rt(int e){need(!e,rek_native5_error());}
void old(int e){need(!e,legacy_error());}
void po(int e){need(!e,rek_native_policy_error());}
const cJSON* field(const cJSON* j,const char* k){auto* p=cJSON_GetObjectItemCaseSensitive(j,k);need(p,k);return p;}
std::string text(const cJSON* p){need(cJSON_IsString(p),"not string");return p->valuestring;}
template<class T>T* allocate(std::vector<void*>& all,size_t n){T* p=nullptr;ck(cudaMalloc(&p,n*sizeof(T)));ck(cudaMemset(p,0,n*sizeof(T)));all.push_back(p);return p;}
__host__ __device__ int acceptance_fixtures(){
    const auto config=rek5_recovered::rek_g1_current_build_hit_detector_config();
    const RekG1ImpactEvent event{.5f,.25f,.25f,1,REK_G1_AIM_LIMB_LEFT_UPPER_BODY};
    RekG1StrikeIntent intent{};intent.impact_events=&event;intent.impact_event_count=1;intent.clip_fps=50;intent.clip_cursor_frames=25;intent.move_id=1;intent.action_playing=intent.layer_active=1;
    RekG1HitDetectorState state{};int failures=0;
    auto a=rek5_recovered::score(state,config,intent,0,2,1.749f,1);failures+=a.reject!=rek5_recovered::Speed;
    a=rek5_recovered::score(state,config,intent,0,3,3,1);failures+=a.reject!=rek5_recovered::Apex;
    intent.clip_cursor_frames=0;a=rek5_recovered::score(state,config,intent,0,2,3,1);failures+=a.reject!=rek5_recovered::Apex;intent.clip_cursor_frames=25;
    intent.layer_active=0;a=rek5_recovered::score(state,config,intent,0,2,3,1);failures+=a.reject!=rek5_recovered::Apex;intent.layer_active=1;
    a=rek5_recovered::score(state,config,intent,0,2,1.75f,1);failures+=a.points!=1;
    a=rek5_recovered::score(state,config,intent,0,2,3,1.125f);failures+=a.reject!=rek5_recovered::Cooldown;
    a=rek5_recovered::score(state,config,intent,0,2,3,1.5f);failures+=a.reject!=rek5_recovered::Duplicate;
    intent.move_id=2;a=rek5_recovered::score(state,config,intent,0,2,3,1.125f);failures+=a.reject!=rek5_recovered::Cooldown;
    a=rek5_recovered::score(state,config,intent,0,2,3,1.5f);failures+=a.points!=1;
    a=rek5_recovered::score(state,config,intent,1,2,3,1.5f);failures+=a.points!=1;
    const RekG1ImpactEvent foot{.5f,.25f,.25f,1,REK_G1_AIM_LIMB_RIGHT_LOWER_BODY};intent.impact_events=&foot;intent.move_id=3;
    a=rek5_recovered::score(state,config,intent,0,1,3,2);failures+=a.points!=2;
    intent.move_id=4;a=rek5_recovered::score(state,config,intent,0,5,3,2);failures+=a.points!=2;
    return failures;
}
__global__ void gpu_fixtures(int* errors){*errors=acceptance_fixtures();}
__global__ void compare(const float* a,const float* b,size_t n,int* errors){size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&__float_as_uint(a[i])!=__float_as_uint(b[i]))atomicAdd(errors,1);}
__global__ void compare_mask(const uint8_t* a,const uint8_t* b,size_t n,int* errors){size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&a[i]!=b[i])atomicAdd(errors,1);}
}
int main(int argc,char** argv){try{
    need(argc==4||argc==2,"Usage: scoring-test RUNTIME_JSON CHECKPOINT SHA256, or --cpu-only");need(acceptance_fixtures()==0,"CPU acceptance fixtures failed");printf("{\"event\":\"cpu_acceptance\",\"cases\":12,\"failures\":0}\n");
    if(argc==2){need(std::string(argv[1])=="--cpu-only","unknown option");return 0;}
    std::ifstream file(argv[1]);need(bool(file),"missing config");std::string raw((std::istreambuf_iterator<char>(file)),{});std::unique_ptr<cJSON,decltype(&cJSON_Delete)> json(cJSON_Parse(raw.c_str()),cJSON_Delete);need(bool(json),"bad config");
    std::string model=text(field(json.get(),"model_path")),assets=text(field(json.get(),"assets_path")),features=text(field(json.get(),"motion_features_path"));
    constexpr int N=64,T=1000;RekNative5Config cfg{};cfg.abi_version=REK_NATIVE5_RUNTIME_ABI;cfg.model_path=model.c_str();cfg.assets_path=assets.c_str();cfg.motion_features_path=features.c_str();cfg.arenas=N;cfg.seed=10001;cfg.round_seconds=20;cfg.locomotion_segment_ticks=1;
    for(int i=0;i<17;i++)cfg.move_duration_ticks[i]=uint32_t(cJSON_GetArrayItem(field(json.get(),"move_duration_ticks"),i)->valuedouble);
    setenv("REK_PHYSICS_BACKEND","semantic_cuda",1);setenv("REK_FAST_OPPONENT_MODE","scripted",1);setenv("REK_FAST_RANDOM_RESETS","1",1);setenv("REK_FAST_SHAPING_WEIGHT","0",1);
    for(const char* key:{"REK_FAST_MOVE_SPEED","REK_FAST_YAW_SPEED","REK_FAST_BODY_RADIUS","REK_FAST_HIT_SPEED","REK_FAST_RESET_GAP_MIN","REK_FAST_RESET_GAP_MAX","REK_FAST_RESET_HEADING_SPREAD_RAD"})unsetenv(key);
    for(const char* mode:{"v4_spheres","recovered_hit_rules_v1"}){
        setenv("REK_FAST_SCORING",mode,1);std::vector<void*> allocated;cudaStream_t stream;ck(cudaStreamCreate(&stream));
        auto make_buffers=[&](){RekNative5Buffers b{};b.observations=allocate<float>(allocated,N*223);b.actions=allocate<float>(allocated,N);b.rewards=allocate<float>(allocated,N);b.terminals=allocate<float>(allocated,N);b.logs=allocate<RekNative5Log>(allocated,N);b.log_stride_bytes=sizeof(RekNative5Log);return b;};
        auto original=make_buffers(),candidate=make_buffers();candidate.actions=original.actions;
        auto* reference=legacy_create(&cfg,&original,stream);need(reference,legacy_error());auto* runtime=rek_native5_create(&cfg,&candidate,stream);need(runtime,rek_native5_error());
        RekNative5DeviceView a{},b{};old(legacy_get_device_view(reference,&a));rt(rek_native5_get_device_view(runtime,&b));
        auto* mask=allocate<uint8_t>(allocated,N*33);old(legacy_bind_action_mask(reference,mask,stream));auto* errors=allocate<int>(allocated,1);
        gpu_fixtures<<<1,1,0,stream>>>(errors);int cpu_errors=0;ck(cudaMemcpyAsync(&cpu_errors,errors,sizeof(int),cudaMemcpyDeviceToHost,stream));ck(cudaStreamSynchronize(stream));need(!cpu_errors,"GPU acceptance fixtures failed");
        RekNativePolicyConfig pc{};pc.abi_version=REK_NATIVE_POLICY_ABI;pc.checkpoint_path=argv[2];pc.expected_sha256=argv[3];pc.hidden_size=256;pc.num_layers=2;pc.batch=N;pc.precision=REK_NATIVE_POLICY_BF16;pc.seed=10001;auto* policy=rek_native_policy_create(&pc,stream);need(policy,rek_native_policy_error());
        bool legacy_mode=std::string(mode)=="v4_spheres";ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        for(int t=0;t<20;t++){
            po(rek_native_policy_step_rows(policy,original.observations,mask,original.terminals,original.actions,0,1,0,stream));old(legacy_step(reference,stream));rt(rek_native5_step(runtime,stream));
            compare<<<(N*72+127)/128,128,0,stream>>>(a.qpos,b.qpos,N*72,errors);compare_mask<<<(N*66+127)/128,128,0,stream>>>(a.action_masks,b.action_masks,N*66,errors);
            if(legacy_mode){compare<<<(N*446+127)/128,128,0,stream>>>(a.raw_observations,b.raw_observations,N*446,errors);compare<<<1,128,0,stream>>>(a.rewards,b.rewards,N*2,errors);}
        }
        cudaGraph_t graph;ck(cudaStreamEndCapture(stream,&graph));cudaGraphExec_t executable;ck(cudaGraphInstantiate(&executable,graph,0));for(int t=0;t<T;t+=20)ck(cudaGraphLaunch(executable,stream));ck(cudaStreamSynchronize(stream));
        ck(cudaMemcpy(&cpu_errors,errors,sizeof(int),cudaMemcpyDeviceToHost));need(!cpu_errors,"trajectory/default parity differences="+std::to_string(cpu_errors));old(legacy_check_status(reference,stream));rt(rek_native5_check_status(runtime,stream));po(rek_native_policy_check_status(policy,stream));
        std::vector<RekNative5RoundResult> ar(N),br(N);ck(cudaMemcpy(ar.data(),a.rounds,N*sizeof(ar[0]),cudaMemcpyDeviceToHost));ck(cudaMemcpy(br.data(),b.rounds,N*sizeof(br[0]),cudaMemcpyDeviceToHost));long long before[2]={},after[2]={};int wins[2]={};for(int i=0;i<N;i++){need(ar[i].terminal&&br[i].terminal,"terminal missing");for(int side=0;side<2;side++){before[side]+=ar[i].points[side];after[side]+=br[i].points[side];}wins[0]+=ar[i].round_winner==0;wins[1]+=br[i].round_winner==0;}
        printf("{\"event\":\"matched_action_scoring\",\"mode\":\"%s\",\"arenas\":64,\"ticks\":1000,\"seed\":10001,\"opponent\":\"scripted\",\"policy_observes\":\"preserved_V4_reference\",\"checkpoint_sha256\":\"%s\",\"gpu_acceptance_cases\":12,\"gpu_acceptance_failures\":0,\"bitwise_qpos_mask_differences\":0,\"default_all_raw_reward_differences\":%s,\"old_points\":[%lld,%lld],\"new_points\":[%lld,%lld],\"old_policy_wins\":%d,\"shadow_policy_wins\":%d}\n",mode,argv[3],legacy_mode?"0":"null",before[0],before[1],after[0],after[1],wins[0],wins[1]);
        rek_native_policy_destroy(policy);old(legacy_close(reference));rt(rek_native5_close(runtime));ck(cudaGraphExecDestroy(executable));ck(cudaGraphDestroy(graph));for(void* p:allocated)ck(cudaFree(p));ck(cudaStreamDestroy(stream));
    }
    return 0;
}catch(const std::exception& e){fprintf(stderr,"scoring test failed: %s\n",e.what());return 2;}}
