#include "../runtime_api.h"
#include "../fast_assets.h"
#include "../native_bot1.cuh"
#include "../rendered_pose_observation.h"
#include "../../../../vendor/cJSON.h"
#include <cuda_runtime.h>
#include <cmath>
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
int legacy_check_status(RekNative5Runtime*,cudaStream_t);
const char* legacy_error();
}
namespace {
constexpr int N=32,T=1200;
void need(bool ok,const std::string& what){if(!ok)throw std::runtime_error(what);}
void ck(cudaError_t e){need(e==cudaSuccess,cudaGetErrorString(e));}
void rt(int e){need(!e,rek_native5_error());}
template<class T>T* alloc(std::vector<void*>& storage,size_t n){T* p=nullptr;ck(cudaMalloc(&p,n*sizeof(T)));ck(cudaMemset(p,0,n*sizeof(T)));storage.push_back(p);return p;}
const cJSON* field(const cJSON* j,const char* key){auto* v=cJSON_GetObjectItemCaseSensitive(j,key);need(v,key);return v;}
struct Counters { unsigned checks,mismatch[6],attacks[2],moves[17];float max_observation_error,max_abs[3];unsigned max_ulp[3];int first_tick,first_index;float first_old,first_new; };
__global__ void fixture(int* errors){
    using namespace rek5_bot1;State s{};activate(s,73);Catalog c{{1,1,1,2,2,1,3,3,4,4,1,2,1,1,1,1,1}};Random rng{s.rng};
    Input i{.5f,0,.02f,1,1,false,false,false,true};s.timer=0;auto d=update(s,i,c,rng);
    int e=d.move!=3&&d.move!=4&&d.move!=11&&d.move!=8&&d.move!=9;
    attack_result(s,i,true);i.punching=true;s.timer=0;d=update(s,i,c,rng);e+=!d.clear_punching||s.phase!=Recovering;
    e+=facing_yaw(-45)!=1.5f||facing_yaw(45)!=-1.5f;*errors=e;
}
__global__ void actions(float* out,const uint8_t* masks,int tick){
    int a=blockIdx.x*blockDim.x+threadIdx.x;if(a>=N)return;
    int phase=tick%100,action=phase<18?2+(tick/100)%6:phase<32?1:phase==32?16+(tick/100)%17:0;
    out[a]=masks[a*66+action]?float(action):1.f;
}
__global__ void save(const float* qpos,const RekNative5RoundResult* rounds,float* previous,int* reset){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<N*72)previous[i]=qpos[i];if(i<N)reset[i]=rounds[i].terminal;
}
__device__ bool different(float a,float b){return __float_as_uint(a)!=__float_as_uint(b);}
__device__ void numeric(float a,float b,int kind,int index,int tick,Counters* c){
    atomicMax(reinterpret_cast<int*>(c->max_abs+kind),__float_as_int(fabsf(a-b)));
    const unsigned ua=__float_as_uint(a),ub=__float_as_uint(b);const unsigned oa=ua&0x80000000u?~ua:ua|0x80000000u,ob=ub&0x80000000u?~ub:ub|0x80000000u;
    atomicMax(c->max_ulp+kind,oa>ob?oa-ob:ob-oa);
    if(kind==0&&atomicCAS(&c->first_tick,0,tick+1)==0){c->first_index=index;c->first_old=a;c->first_new=b;}
}
__global__ void compare(RekNative5DeviceView a,RekNative5DeviceView b,bool raw,int tick,Counters* c){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N*72&&different(a.qpos[i],b.qpos[i])){atomicAdd(c->mismatch+0,1u);numeric(a.qpos[i],b.qpos[i],0,i,tick,c);}
    if(i<N*70&&different(a.qvel[i],b.qvel[i])){atomicAdd(c->mismatch+1,1u);numeric(a.qvel[i],b.qvel[i],1,i,tick,c);}
    if(i<N*66&&a.action_masks[i]!=b.action_masks[i])atomicAdd(c->mismatch+2,1u);
    if(i<N*2&&(different(a.rewards[i],b.rewards[i])||different(a.terminals[i],b.terminals[i])||different(a.actions[i],b.actions[i])))atomicAdd(c->mismatch+3,1u);
    if(raw&&i<N*446&&different(a.raw_observations[i],b.raw_observations[i])){atomicAdd(c->mismatch+4,1u);numeric(a.raw_observations[i],b.raw_observations[i],2,i,tick,c);}
}
__device__ void near(float a,float b,Counters* c){
    atomicAdd(&c->checks,1u);float e=fabsf(a-b);if(!isfinite(a)||e>3e-4f)atomicAdd(c->mismatch+5,1u);
    atomicMax(reinterpret_cast<int*>(&c->max_observation_error),__float_as_int(e));
}
__global__ void rendered(RekNative5DeviceView v,const float* previous,const int* reset,const int* joints,Counters* c){
    int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=N*2||reset[row/2])return;
    int arena=row/2,side=row%2;const float* q=v.qpos+arena*72+side*36;const float* old=previous+arena*72+side*36;
    const float h=rek_rendered_pose::heading(q+3),oh=rek_rendered_pose::heading(old+3);float vel[2];
    rek_rendered_pose::local_velocity(q[0],q[1],old[0],old[1],h,.02f,vel);
    const float* raw=v.raw_observations+row*223;near(raw[7],vel[0],c);near(raw[8],vel[1],c);
    near(raw[12],rek_rendered_pose::angular_rate(h,oh,.02f),c);near(raw[172],cosf(.5f*h),c);near(raw[175],sinf(.5f*h),c);
    near(raw[221],raw[217]+raw[218],c);near(raw[222],raw[217]+raw[218],c);
    for(int j=0;j<29;j++){int k=joints[side*29+j]-side*36;near(raw[42+j],rek_rendered_pose::joint_rate(q[k],old[k],.02f),c);}
}
__global__ void setup_bot(uint8_t* overrides,float* external){int row=blockIdx.x*blockDim.x+threadIdx.x;if(row<N*2){overrides[row]=row%2==(row/2)%2?2:1;external[row]=1;}}
__global__ void bot_count(RekNative5DeviceView v,const int* move_for_action,Counters* c){
    int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=N*2)return;int action=int(v.actions[row]);
    if(action>=16){atomicAdd(c->attacks+row%2,1u);atomicAdd(c->moves+move_for_action[action-16],1u);}
}
}
int main(int argc,char** argv){try{
    need(argc==2,"runtime-test RUNTIME_JSON");std::ifstream file(argv[1]);std::string text((std::istreambuf_iterator<char>(file)),{});
    std::unique_ptr<cJSON,decltype(&cJSON_Delete)> json(cJSON_Parse(text.c_str()),cJSON_Delete);need(bool(json),"runtime json");
    std::string model=field(json.get(),"model_path")->valuestring,assets=field(json.get(),"assets_path")->valuestring,features=field(json.get(),"motion_features_path")->valuestring;
    RekNative5Config cfg{};cfg.abi_version=REK_NATIVE5_RUNTIME_ABI;cfg.model_path=model.c_str();cfg.assets_path=assets.c_str();cfg.motion_features_path=features.c_str();cfg.arenas=N;cfg.seed=7829;cfg.round_seconds=20;cfg.locomotion_segment_ticks=1;
    for(int i=0;i<17;i++)cfg.move_duration_ticks[i]=uint32_t(cJSON_GetArrayItem(field(json.get(),"move_duration_ticks"),i)->valuedouble);
    setenv("REK_PHYSICS_BACKEND","semantic_cuda",1);setenv("REK_FAST_OPPONENT_MODE","scripted",1);setenv("REK_FAST_RANDOM_RESETS","1",1);setenv("REK_FAST_SHAPING_WEIGHT","0",1);
    for(const char* key:{"REK_FAST_MOVE_SPEED","REK_FAST_YAW_SPEED","REK_FAST_BODY_RADIUS","REK_FAST_HIT_SPEED","REK_FAST_RESET_GAP_MIN","REK_FAST_RESET_GAP_MAX","REK_FAST_RESET_HEADING_SPREAD_RAD"})unsetenv(key);
    auto baked=load_fast_assets(cfg);std::vector<void*> memory;auto* joints=alloc<int>(memory,58);ck(cudaMemcpy(joints,baked.qindices,58*sizeof(int),cudaMemcpyHostToDevice));
    int move_host[17];for(int a=16;a<33;a++)move_host[a-16]=baked.routes[baked.action_to_route[a]].move;
    auto* move_map=alloc<int>(memory,17);ck(cudaMemcpy(move_map,move_host,sizeof(move_host),cudaMemcpyHostToDevice));
    cudaStream_t stream;ck(cudaStreamCreate(&stream));auto* checks=alloc<Counters>(memory,1);auto* previous=alloc<float>(memory,N*72);auto* reset=alloc<int>(memory,N);auto* errors=alloc<int>(memory,1);
    fixture<<<1,1,0,stream>>>(errors);int errors_host=0;ck(cudaMemcpyAsync(&errors_host,errors,sizeof(int),cudaMemcpyDeviceToHost,stream));ck(cudaStreamSynchronize(stream));need(!errors_host,"device FSM fixture");
    auto buffers=[&](){RekNative5Buffers b{};b.observations=alloc<float>(memory,N*223);b.actions=alloc<float>(memory,N);b.rewards=alloc<float>(memory,N);b.terminals=alloc<float>(memory,N);b.logs=alloc<RekNative5Log>(memory,N);b.log_stride_bytes=sizeof(RekNative5Log);return b;};
    for(int test=0;test<4;test++){
        const char* scoring=test==0?"v4_spheres":"recovered_hit_rules_v1";
        setenv("REK_FAST_SCORING",scoring,1);setenv("REK_FAST_OPPONENT","v4_scripted",1);setenv("REK_FAST_OBSERVATION","v4_logical",1);
        auto a=buffers(),b=buffers();b.actions=a.actions;auto* old=test==2?rek_native5_create(&cfg,&a,stream):legacy_create(&cfg,&a,stream);need(old,test==2?rek_native5_error():legacy_error());RekNative5DeviceView va{};need(!(test==2?rek_native5_get_device_view(old,&va):legacy_get_device_view(old,&va)),"reference view");
        if(test>=2)setenv("REK_FAST_OBSERVATION","rendered_pose_v1",1);
        if(test==3){setenv("REK_FAST_OPPONENT","recovered_bot1_v1",1);setenv("REK_FAST_SCORING","recovered_hit_rules_v2",1);}
        auto* runtime=rek_native5_create(&cfg,&b,stream);need(runtime,rek_native5_error());RekNative5DeviceView vb{};rt(rek_native5_get_device_view(runtime,&vb));
        auto* overrides=alloc<uint8_t>(memory,N*2);auto* external=alloc<float>(memory,N*2);
        if(test==3){setup_bot<<<1,128,0,stream>>>(overrides,external);rt(rek_native5_bind_external_actions(runtime,external,overrides,stream));}
        ck(cudaMemsetAsync(checks,0,sizeof(Counters),stream));
        for(int chunk=0;chunk<T;chunk+=100){
            ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            for(int t=chunk;t<chunk+100;t++){
                actions<<<1,128,0,stream>>>(a.actions,va.action_masks,t);save<<<(N*72+127)/128,128,0,stream>>>(vb.qpos,vb.rounds,previous,reset);
                if(test<3)need(!(test==2?rek_native5_step(old,stream):legacy_step(old,stream)),"reference step");rt(rek_native5_step(runtime,stream));
                if(test<3)compare<<<(N*446+127)/128,128,0,stream>>>(va,vb,test<2,t,checks);
                if(test>=2)rendered<<<1,128,0,stream>>>(vb,previous,reset,joints,checks);
                if(test==3)bot_count<<<1,128,0,stream>>>(vb,move_map,checks);
            }
            cudaGraph_t graph;cudaGraphExec_t executable;ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&executable,graph,0));ck(cudaGraphLaunch(executable,stream));ck(cudaStreamSynchronize(stream));ck(cudaGraphExecDestroy(executable));ck(cudaGraphDestroy(graph));
        }
        Counters result{};ck(cudaMemcpy(&result,checks,sizeof(result),cudaMemcpyDeviceToHost));rt(rek_native5_check_status(runtime,stream));
        if(test==3)need(result.attacks[0]>0&&result.attacks[1]>0,"bot must attack on both sides");
        printf("{\"event\":\"native_bot1_runtime_fixture\",\"test\":%d,\"arenas\":32,\"ticks\":1200,\"comparison\":\"%s\",\"mismatches\":[%u,%u,%u,%u,%u,%u],\"rendered_checks\":%u,\"max_error\":%.9g,\"bot_attacks_by_side\":[%u,%u],\"bot_move_counts\":[",test,test==0?"legacy_v4_bitwise":test==1?"legacy_recovered_v1_bitwise":test==2?"rendered_observation_physics_reward_mask_invariance":"recovered_bot_both_sides",result.mismatch[0],result.mismatch[1],result.mismatch[2],result.mismatch[3],result.mismatch[4],result.mismatch[5],result.checks,result.max_observation_error,result.attacks[0],result.attacks[1]);
        for(int m=0;m<17;m++)printf("%s%u",m?",":"",result.moves[m]);printf("],\"server_parity\":false}\n");fflush(stdout);
        printf("{\"event\":\"numerical_audit\",\"test\":%d,\"first_tick\":%d,\"first_qpos_index\":%d,\"first_old\":%.9g,\"first_new\":%.9g,\"max_abs_qpos_qvel_raw\":[%.9g,%.9g,%.9g],\"max_ulp_qpos_qvel_raw\":[%u,%u,%u]}\n",test,result.first_tick,result.first_index,result.first_old,result.first_new,result.max_abs[0],result.max_abs[1],result.max_abs[2],result.max_ulp[0],result.max_ulp[1],result.max_ulp[2]);fflush(stdout);
        need(result.mismatch[2]==0&&result.mismatch[3]==0&&result.mismatch[5]==0,"mask/reward/action/observation mismatch");
        if(test==2)need(result.mismatch[0]==0&&result.mismatch[1]==0,"rendered mode changes physics");
        for(float error:result.max_abs)need(error<1e-4f,"legacy numerical error exceeds 1e-4");
        rt(rek_native5_close(runtime));need(!(test==2?rek_native5_close(old):legacy_close(old)),"reference close");
    }
    for(void* p:memory)ck(cudaFree(p));ck(cudaStreamDestroy(stream));return 0;
}catch(const std::exception& e){fprintf(stderr,"Bot1 runtime probe failed: %s\n",e.what());return 2;}}
