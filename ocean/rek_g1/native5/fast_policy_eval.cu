#include "runtime_api.h"
#include "native_policy.h"
#include "fast_mode_config.h"
#include "../../../vendor/cJSON.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool value,const std::string& message){if(!value)throw std::runtime_error(message);}
void cuda_ok(cudaError_t value){if(value!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(value));}
void runtime_ok(int value){if(value)throw std::runtime_error(rek_native5_error());}
void policy_ok(int value){if(value)throw std::runtime_error(rek_native_policy_error());}
const cJSON* field(const cJSON* object,const char* key){auto* v=cJSON_GetObjectItemCaseSensitive(object,key);require(v,std::string("Missing config field ")+key);return v;}
std::string string(const cJSON* value){require(cJSON_IsString(value)&&value->valuestring,"Expected config string");return value->valuestring;}
int integer(const cJSON* value){require(cJSON_IsNumber(value)&&std::isfinite(value->valuedouble)&&value->valuedouble==std::floor(value->valuedouble)&&value->valuedouble>=0&&value->valuedouble<=INT32_MAX,"Expected nonnegative int32");return int(value->valuedouble);}
int number_arg(const char* text,int maximum){char* end=nullptr;long n=std::strtol(text,&end,10);require(end&&!*end&&n>0&&n<=maximum,"Invalid positive integer argument");return int(n);}
template<class T>T* allocate(std::vector<void*>& owned,size_t n){T* p=nullptr;cuda_ok(cudaMalloc(&p,n*sizeof(T)));owned.push_back(p);cuda_ok(cudaMemset(p,0,n*sizeof(T)));return p;}
struct Record {
    unsigned long long completed;
    int arena,episode,policy_side,tick,duration_ticks;
    int points[2],falls[2],winner,round_result;
};
__global__ void set_overrides(uint8_t* values,int arenas,int side){int row=blockIdx.x*blockDim.x+threadIdx.x;if(row<arenas*2)values[row]=(row%2)==side?1:2;}
__global__ void override_round_feature(float* observations,int rows,float value){int row=blockIdx.x*blockDim.x+threadIdx.x;if(row<rows)observations[row*223+186]=value;}
// Captured tick counter keeps timestamps correct across repeated graph launches.
__global__ void increment_tick(int* tick){if(!blockIdx.x&&!threadIdx.x)++*tick;}
__global__ void collect_captured(const RekNative5RoundResult* rounds,Record* records,int* counts,
    unsigned long long* seen,int* start_ticks,int* failures,int arenas,int limit,int side,const int* tick){
    int a=blockIdx.x*blockDim.x+threadIdx.x;if(a>=arenas)return;
    const auto& r=rounds[a];if(r.failure_bits)atomicOr(failures,1);
    if(r.completed_rounds==seen[a])return;
    if(r.completed_rounds!=seen[a]+1)atomicOr(failures,2);
    seen[a]=r.completed_rounds;
    if(counts[a]>=limit)return;
    if(!r.terminal||(r.round_result!=1&&r.round_result!=2&&r.round_result!=3)
        ||(r.round_result==3?r.round_winner!=-1:(r.round_winner!=0&&r.round_winner!=1)))atomicOr(failures,4);
    int episode=counts[a]++;Record& out=records[a*limit+episode];
    out.completed=r.completed_rounds;out.arena=a;out.episode=episode;out.policy_side=side;out.tick=*tick;out.duration_ticks=*tick-start_ticks[a];start_ticks[a]=*tick;
    for(int s=0;s<2;s++){out.points[s]=r.points[s];out.falls[s]=int(r.falls[s]);if(r.points[s]<0)atomicOr(failures,8);}
    out.winner=r.round_winner;out.round_result=r.round_result;
}
}

int main(int argc,char** argv){
    try{
        require(argc==10||argc==12,"Usage: fast-policy-eval RUNTIME_JSON CHECKPOINT SHA256 ARENAS ROUNDS_PER_SIDE SEED sampled|greedy bf16|fp32 NEW_RECORDS_JSONL [HIDDEN LAYERS]");
        int arenas=number_arg(argv[4],65536),rounds=number_arg(argv[5],10000),seed=number_arg(argv[6],INT32_MAX);
        require(size_t(arenas)*rounds<=1048576,"At most 1048576 recorded rounds per side");
        std::string selection=argv[7],precision=argv[8];require(selection=="sampled"||selection=="greedy","Explicit sampled/greedy selection required");require(precision=="bf16"||precision=="fp32","Explicit precision required");
        int hidden=argc==12?number_arg(argv[10],4096):256,layers=argc==12?number_arg(argv[11],16):2;
        const char* diagnostic_env=std::getenv("REK_EVAL_ROUND_FEATURE");
        const bool diagnostic=diagnostic_env&&*diagnostic_env;
        float diagnostic_round=diagnostic?float(number_arg(diagnostic_env,1000000)):0;
        const std::string diagnostic_json=diagnostic?std::to_string(int(diagnostic_round)):"null";
        std::ifstream file(argv[1]);require(bool(file),"Runtime JSON missing");std::string text((std::istreambuf_iterator<char>(file)),{});
        std::unique_ptr<cJSON,decltype(&cJSON_Delete)> json(cJSON_Parse(text.c_str()),cJSON_Delete);require(json&&cJSON_IsObject(json.get()),"Invalid runtime JSON");
        require(string(field(json.get(),"backend"))=="semantic_cuda","Explicit semantic_cuda required");
        std::string model=string(field(json.get(),"model_path")),assets=string(field(json.get(),"assets_path")),features=string(field(json.get(),"motion_features_path"));
        RekNative5Config cfg{};cfg.abi_version=REK_NATIVE5_RUNTIME_ABI;cfg.model_path=model.c_str();cfg.assets_path=assets.c_str();cfg.motion_features_path=features.c_str();cfg.arenas=arenas;cfg.seed=seed;
        cfg.round_seconds=float(integer(field(json.get(),"round_seconds")));require(cfg.round_seconds>0&&cfg.round_seconds<=3600,"Invalid round duration");
        auto* segment=cJSON_GetObjectItemCaseSensitive(json.get(),"locomotion_segment_ticks");cfg.locomotion_segment_ticks=segment?integer(segment):1;
        auto* durations=cJSON_GetObjectItemCaseSensitive(json.get(),"move_duration_ticks");
        const uint32_t defaults[17]={35,27,31,45,32,45,157,145,158,139,134,138,73,75,68,71,103};
        if(durations)require(cJSON_IsArray(durations)&&cJSON_GetArraySize(durations)==17,"Expected17 durations");
        for(int i=0;i<17;i++)cfg.move_duration_ticks[i]=durations?integer(cJSON_GetArrayItem(durations,i)):defaults[i];
        setenv("REK_PHYSICS_BACKEND","semantic_cuda",1);
        auto* fast=cJSON_GetObjectItemCaseSensitive(json.get(),"fast");
        const auto mode_identity=rek5_modes::configure(fast);
        auto* scoring=fast?cJSON_GetObjectItemCaseSensitive(fast,"scoring_mode"):nullptr;
        const std::string scoring_mode=scoring?string(scoring):"v4_spheres";
        require(scoring_mode=="v4_spheres"||scoring_mode=="recovered_hit_rules_v1"||scoring_mode=="recovered_hit_rules_v2","Invalid explicit scoring mode");
        setenv("REK_FAST_SCORING",scoring_mode.c_str(),1);
        const std::string identity_fields=rek5_modes::json_fields(mode_identity,scoring_mode);
        std::printf("{\"event\":\"scoring_identity\",\"scoring_mode\":\"%s\",\"source\":\"runtime_config\"}\n",scoring_mode.c_str());
        const char* keys[]={"move_speed","yaw_speed","body_radius","hit_speed","down_damage"};
        const char* env[]={"REK_FAST_MOVE_SPEED","REK_FAST_YAW_SPEED","REK_FAST_BODY_RADIUS","REK_FAST_HIT_SPEED","REK_FAST_DOWN_DAMAGE"};
        for(const char* key:env)unsetenv(key); // Config controls behavior, not ambient shell overrides.
        if(fast)for(int i=0;i<5;i++){auto* value=cJSON_GetObjectItemCaseSensitive(fast,keys[i]);if(value){require(cJSON_IsNumber(value)&&std::isfinite(value->valuedouble),"Invalid compact setting");char b[64];std::snprintf(b,sizeof(b),"%.17g",value->valuedouble);setenv(env[i],b,1);}}
        std::unique_ptr<FILE,decltype(&std::fclose)> output(std::fopen(argv[9],"wx"),std::fclose);require(bool(output),"Records file must be new and writable");
        cudaStream_t stream;cuda_ok(cudaStreamCreate(&stream));std::vector<void*> owned;
        RekNative5Buffers buffers{};buffers.observations=allocate<float>(owned,size_t(arenas)*223);buffers.actions=allocate<float>(owned,arenas);buffers.rewards=allocate<float>(owned,arenas);buffers.terminals=allocate<float>(owned,arenas);buffers.logs=allocate<RekNative5Log>(owned,arenas);buffers.log_stride_bytes=sizeof(RekNative5Log);
        RekNative5Runtime* runtime=rek_native5_create(&cfg,&buffers,stream);require(runtime,rek_native5_error());RekNative5DeviceView view{};runtime_ok(rek_native5_get_device_view(runtime,&view));
        float* encoded=allocate<float>(owned,size_t(arenas)*446);float* actions=allocate<float>(owned,size_t(arenas)*2);auto* overrides=allocate<uint8_t>(owned,size_t(arenas)*2);
        auto* records=allocate<Record>(owned,size_t(arenas)*rounds);auto* counts=allocate<int>(owned,arenas);auto* seen=allocate<unsigned long long>(owned,arenas);auto* starts=allocate<int>(owned,arenas);auto* failures=allocate<int>(owned,1);auto* device_tick=allocate<int>(owned,1);
        runtime_ok(rek_native5_bind_external_actions(runtime,actions,overrides,stream));
        RekNativePolicyConfig pc{};pc.abi_version=REK_NATIVE_POLICY_ABI;pc.checkpoint_path=argv[2];pc.expected_sha256=argv[3];pc.hidden_size=hidden;pc.num_layers=layers;pc.batch=arenas;pc.precision=precision=="bf16"?REK_NATIVE_POLICY_BF16:REK_NATIVE_POLICY_FP32;pc.seed=seed;
        std::unique_ptr<RekNativePolicy,decltype(&rek_native_policy_destroy)> policy(rek_native_policy_create(&pc,stream),rek_native_policy_destroy);require(bool(policy),rek_native_policy_error());
        const std::string digest=rek_native_policy_sha256(policy.get());
        long long total_wins=0,total_losses=0,total_ties=0;double total_wall=0;
        for(int side=0;side<2;side++){
            runtime_ok(rek_native5_reset(runtime,stream));policy_ok(rek_native_policy_reset(policy.get(),stream));
            cuda_ok(cudaMemsetAsync(counts,0,arenas*sizeof(int),stream));cuda_ok(cudaMemsetAsync(seen,0,arenas*sizeof(unsigned long long),stream));cuda_ok(cudaMemsetAsync(starts,0,arenas*sizeof(int),stream));cuda_ok(cudaMemsetAsync(device_tick,0,sizeof(int),stream));cuda_ok(cudaMemsetAsync(failures,0,sizeof(int),stream));
            set_overrides<<<(arenas*2+127)/128,128,0,stream>>>(overrides,arenas,side);cuda_ok(cudaStreamSynchronize(stream));
            constexpr int chunk=64;
            auto step=[&](){runtime_ok(rek_native5_encode_fighter_observations(runtime,encoded,stream));if(diagnostic)override_round_feature<<<(arenas*2+127)/128,128,0,stream>>>(encoded,arenas*2,diagnostic_round);policy_ok(rek_native_policy_step_rows(policy.get(),encoded,view.action_masks,view.terminals,actions,side,2,selection=="greedy",stream));runtime_ok(rek_native5_step(runtime,stream));increment_tick<<<1,1,0,stream>>>(device_tick);collect_captured<<<(arenas+127)/128,128,0,stream>>>(view.rounds,records,counts,seen,starts,failures,arenas,rounds,side,device_tick);};
            cuda_ok(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));for(int t=0;t<chunk;t++)step();cudaGraph_t graph;cuda_ok(cudaStreamEndCapture(stream,&graph));cudaGraphExec_t executable;cuda_ok(cudaGraphInstantiate(&executable,graph,0));
            auto start=std::chrono::steady_clock::now();int ticks=0;std::vector<int> host_counts(arenas);
            const int max_ticks=int(std::ceil(cfg.round_seconds*50))*rounds+rounds+chunk;
            do {
                cuda_ok(cudaGraphLaunch(executable,stream));ticks+=chunk;
                cuda_ok(cudaMemcpyAsync(host_counts.data(),counts,arenas*sizeof(int),cudaMemcpyDeviceToHost,stream));cuda_ok(cudaStreamSynchronize(stream));
                if(*std::min_element(host_counts.begin(),host_counts.end())>=rounds)break;
                require(ticks<=max_ticks,"Evaluation exceeded bounded simulation budget");
            }while(true);
            double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();total_wall+=wall;
            runtime_ok(rek_native5_check_status(runtime,stream));policy_ok(rek_native_policy_check_status(policy.get(),stream));int failure=0;cuda_ok(cudaMemcpy(&failure,failures,sizeof(int),cudaMemcpyDeviceToHost));require(!failure,"Invalid terminal record or missed round boundary");
            std::vector<Record> host(size_t(arenas)*rounds);cuda_ok(cudaMemcpy(host.data(),records,host.size()*sizeof(Record),cudaMemcpyDeviceToHost));
            long long wins=0,losses=0,ties=0,points=0,opponent_points=0,falls=0,opponent_falls=0;
            for(const auto& r:host){require(r.completed==static_cast<unsigned long long>(r.episode+1)&&r.duration_ticks>0,"Incomplete terminal record");if(r.winner<0)ties++;else if(r.winner==side)wins++;else losses++;
                points+=r.points[side];opponent_points+=r.points[side^1];falls+=r.falls[side];opponent_falls+=r.falls[side^1];
                std::fprintf(output.get(),"{\"backend\":\"semantic_cuda\",\"policy_sha256\":\"%s\",\"selection\":\"%s\",\"seed\":%d,\"arena\":%d,\"episode\":%d,\"policy_side\":%d,\"score\":[%d,%d],\"falls\":[%d,%d],\"winner\":%d,\"round_result\":%d,\"duration_ticks\":%d,\"duration_seconds\":%.9g,\"diagnostic\":%s,\"round_feature_override\":%s%s}\n",digest.c_str(),selection.c_str(),seed,r.arena,r.episode,side,r.points[0],r.points[1],r.falls[0],r.falls[1],r.winner,r.round_result,r.duration_ticks,r.duration_ticks*.02,diagnostic?"true":"false",diagnostic_json.c_str(),identity_fields.c_str());
            }
            require(std::fflush(output.get())==0,"Could not flush match records");total_wins+=wins;total_losses+=losses;total_ties+=ties;
            std::printf("{\"event\":\"side_result\",\"policy_side\":%d,\"arenas\":%d,\"rounds_per_arena\":%d,\"wins\":%lld,\"losses\":%lld,\"draws\":%lld,\"points\":%lld,\"opponent_points\":%lld,\"falls\":%lld,\"opponent_falls\":%lld,\"evaluated_ticks\":%d,\"execution_wall_seconds\":%.9g}\n",side,arenas,rounds,wins,losses,ties,points,opponent_points,falls,opponent_falls,ticks,wall);std::fflush(stdout);
            cuda_ok(cudaGraphExecDestroy(executable));cuda_ok(cudaGraphDestroy(graph));
        }
        std::printf("{\"event\":\"frozen_policy_evaluation\",\"backend\":\"semantic_cuda\",\"checkpoint_sha256\":\"%s\",\"precision\":\"%s\",\"observation_encoding\":\"scaled_polar_xy\",\"selection\":\"%s\",\"policy_rng_seed\":%d,\"wins\":%lld,\"losses\":%lld,\"draws\":%lld,\"win_rate\":%.9g,\"execution_wall_seconds\":%.9g,\"environment_randomizes_seed\":false,\"greedy_repeats_duplicate_fixed_fixtures\":%s,\"opponent\":\"same_runtime_GPU_scripted\",\"both_sides\":true,\"terminal_recurrent_reset\":true,\"failure_bits\":0,\"python_runtime\":false,\"cpu_physics\":false,\"training_sps\":null,\"diagnostic\":%s,\"round_feature_override\":%s%s}\n",digest.c_str(),precision.c_str(),selection.c_str(),seed,total_wins,total_losses,total_ties,double(total_wins)/double(total_wins+total_losses+total_ties),total_wall,selection=="greedy"&&mode_identity.opponent=="v4_scripted"?"true":"false",diagnostic?"true":"false",diagnostic_json.c_str(),identity_fields.c_str());
        policy.reset();runtime_ok(rek_native5_close(runtime));for(void* p:owned)cuda_ok(cudaFree(p));cuda_ok(cudaStreamDestroy(stream));return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"fast policy evaluation failed: %s\n",e.what());return 2;}
}
