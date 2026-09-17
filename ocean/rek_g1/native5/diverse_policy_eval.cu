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
    int first_hit_tick[2],point_delta_sum[2],busy_ticks,facing_ticks,near_ticks,attack_commands;
    float path_length,min_gap,max_gap,mean_gap,initial_xy[4],final_xy[4];
};
struct Measure {
    int ticks,first_hit_tick[2],point_delta_sum[2],busy_ticks,facing_ticks,near_ticks,attack_commands,previous_valid;
    float path_length,min_gap,max_gap,gap_sum,previous_x,previous_y,initial_xy[4];

};
__global__ void set_overrides(uint8_t* values,int arenas,int side,int learned_opponent){int row=blockIdx.x*blockDim.x+threadIdx.x;if(row<arenas*2)values[row]=(row%2)==side||learned_opponent?1:2;}
// Captured tick counter keeps timestamps correct across repeated graph launches.
__global__ void increment_tick(int* tick){if(!blockIdx.x&&!threadIdx.x)++*tick;}
__global__ void collect_captured(const RekNative5RoundResult* rounds,const float* raw,const float* actions,
    Record* records,Measure* measures,int* counts,unsigned long long* seen,int* start_ticks,
    int* failures,int arenas,int limit,int side,const int* tick){
    int a=blockIdx.x*blockDim.x+threadIdx.x;if(a>=arenas)return;
    const auto& r=rounds[a];if(r.failure_bits)atomicOr(failures,1);
    if(counts[a]>=limit)return;
    Measure& m=measures[a];const float* obs=raw+(a*2+side)*223;
    float x=obs[0],y=obs[1],ox=obs[86],oy=obs[87],gap=hypotf(ox-x,oy-y);
    float heading=2*atan2f(obs[175],obs[172]);
    float bearing=atan2f(sinf(atan2f(oy-y,ox-x)-heading),cosf(atan2f(oy-y,ox-x)-heading));
    if(!isfinite(gap)||!isfinite(heading)||!isfinite(bearing))atomicOr(failures,32);
    if(!m.previous_valid){
        m.initial_xy[0]=x;m.initial_xy[1]=y;m.initial_xy[2]=ox;m.initial_xy[3]=oy;
        m.min_gap=m.max_gap=gap;m.previous_valid=1;
    }else m.path_length+=hypotf(x-m.previous_x,y-m.previous_y);
    m.previous_x=x;m.previous_y=y;m.min_gap=fminf(m.min_gap,gap);m.max_gap=fmaxf(m.max_gap,gap);m.gap_sum+=gap;
    ++m.ticks;m.busy_ticks+=obs[182]>.5f;m.facing_ticks+=fabsf(bearing)<.16f;m.near_ticks+=gap<1.05f;
    m.attack_commands+=actions[a*2+side]>=16;
    for(int fighter=0;fighter<2;fighter++){
        float delta=raw[(a*2+fighter)*223+217];
        if(delta<0||!isfinite(delta)||delta!=floorf(delta))atomicOr(failures,64);
        m.point_delta_sum[fighter]+=int(delta);
        if(delta>0&&!m.first_hit_tick[fighter])m.first_hit_tick[fighter]=m.ticks;
    }
    if(r.completed_rounds==seen[a])return;
    if(r.completed_rounds!=seen[a]+1)atomicOr(failures,2);
    seen[a]=r.completed_rounds;
    if(!r.terminal||(r.round_result!=1&&r.round_result!=2&&r.round_result!=3)
        ||(r.round_result==3?r.round_winner!=-1:(r.round_winner!=0&&r.round_winner!=1)))atomicOr(failures,4);
    int episode=counts[a]++;Record& out=records[a*limit+episode];
    out.completed=r.completed_rounds;out.arena=a;out.episode=episode;out.policy_side=side;out.tick=*tick;out.duration_ticks=*tick-start_ticks[a];start_ticks[a]=*tick;
    if(out.duration_ticks!=m.ticks)atomicOr(failures,128);
    for(int fighter=0;fighter<2;fighter++){
        out.points[fighter]=r.points[fighter];out.falls[fighter]=int(r.falls[fighter]);
        out.first_hit_tick[fighter]=m.first_hit_tick[fighter];out.point_delta_sum[fighter]=m.point_delta_sum[fighter];
        if(r.points[fighter]<0||r.points[fighter]!=m.point_delta_sum[fighter])atomicOr(failures,8);
    }
    out.winner=r.round_winner;out.round_result=r.round_result;
    out.busy_ticks=m.busy_ticks;out.facing_ticks=m.facing_ticks;out.near_ticks=m.near_ticks;out.attack_commands=m.attack_commands;
    out.path_length=m.path_length;out.min_gap=m.min_gap;out.max_gap=m.max_gap;out.mean_gap=m.gap_sum/m.ticks;
    for(int j=0;j<4;j++)out.initial_xy[j]=m.initial_xy[j];
    out.final_xy[0]=x;out.final_xy[1]=y;out.final_xy[2]=ox;out.final_xy[3]=oy;
    m=Measure{};

}
}

int main(int argc,char** argv){
    try{
        require(argc==13||argc==15,"Usage: diverse-policy-eval RUNTIME_JSON CHECKPOINT SHA256 ARENAS ROUNDS SEED sampled|greedy bf16|fp32 NEW_RECORDS_JSONL neutral|scripted|retreat|strafe|checkpoint fixed|heldout ROUND_SECONDS [OPP_CHECKPOINT OPP_SHA256]");
        int arenas=number_arg(argv[4],65536),rounds=number_arg(argv[5],10000),seed=number_arg(argv[6],INT32_MAX);
        require(size_t(arenas)*rounds<=1048576,"At most 1048576 recorded rounds per side");
        std::string selection=argv[7],precision=argv[8];require(selection=="sampled"||selection=="greedy","Explicit sampled/greedy selection required");require(precision=="bf16"||precision=="fp32","Explicit precision required");
        int hidden=256,layers=2;
        const std::string opponent_name=argv[10],fixture=argv[11];
        require(opponent_name=="neutral"||opponent_name=="scripted"||opponent_name=="retreat"||opponent_name=="strafe"||opponent_name=="checkpoint","Invalid explicit opponent");
        require(fixture=="fixed"||fixture=="heldout","Invalid fixture kind");
        bool learned_opponent=opponent_name=="checkpoint";
        require((learned_opponent&&argc==15)||(!learned_opponent&&argc==13),"Opponent checkpoint and SHA required only for checkpoint mode");
        int round_seconds=number_arg(argv[12],3600);
        std::ifstream file(argv[1]);require(bool(file),"Runtime JSON missing");std::string text((std::istreambuf_iterator<char>(file)),{});
        std::unique_ptr<cJSON,decltype(&cJSON_Delete)> json(cJSON_Parse(text.c_str()),cJSON_Delete);require(json&&cJSON_IsObject(json.get()),"Invalid runtime JSON");
        require(string(field(json.get(),"backend"))=="semantic_cuda","Explicit semantic_cuda required");
        std::string model=string(field(json.get(),"model_path")),assets=string(field(json.get(),"assets_path")),features=string(field(json.get(),"motion_features_path"));
        RekNative5Config cfg{};cfg.abi_version=REK_NATIVE5_RUNTIME_ABI;cfg.model_path=model.c_str();cfg.assets_path=assets.c_str();cfg.motion_features_path=features.c_str();cfg.arenas=arenas;cfg.seed=seed;
        cfg.round_seconds=float(round_seconds);
        auto* segment=cJSON_GetObjectItemCaseSensitive(json.get(),"locomotion_segment_ticks");cfg.locomotion_segment_ticks=segment?integer(segment):1;
        auto* durations=cJSON_GetObjectItemCaseSensitive(json.get(),"move_duration_ticks");
        const uint32_t defaults[17]={35,27,31,45,32,45,157,145,158,139,134,138,73,75,68,71,103};
        if(durations)require(cJSON_IsArray(durations)&&cJSON_GetArraySize(durations)==17,"Expected17 durations");
        for(int i=0;i<17;i++)cfg.move_duration_ticks[i]=durations?integer(cJSON_GetArrayItem(durations,i)):defaults[i];
        setenv("REK_PHYSICS_BACKEND","semantic_cuda",1);
        setenv("REK_FAST_OPPONENT_MODE",learned_opponent?"neutral":opponent_name.c_str(),1);
        setenv("REK_FAST_RANDOM_RESETS",fixture=="heldout"?"1":"0",1);
        setenv("REK_FAST_RESET_GAP_MIN","0.55",1);
        setenv("REK_FAST_RESET_GAP_MAX","2.5",1);
        setenv("REK_FAST_RESET_HEADING_SPREAD_RAD","3.141592653589793",1);
        setenv("REK_FAST_SHAPING_WEIGHT","0",1);
        unsetenv("REK_FAST_SHAPING_GAMMA");
        unsetenv("REK_FAST_SHAPING_TARGET");
        unsetenv("REK_FAST_SHAPING_BEARING_WEIGHT");
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
        auto* records=allocate<Record>(owned,size_t(arenas)*rounds);auto* measures=allocate<Measure>(owned,arenas);auto* counts=allocate<int>(owned,arenas);auto* seen=allocate<unsigned long long>(owned,arenas);auto* starts=allocate<int>(owned,arenas);auto* failures=allocate<int>(owned,1);auto* device_tick=allocate<int>(owned,1);
        runtime_ok(rek_native5_bind_external_actions(runtime,actions,overrides,stream));
        RekNativePolicyConfig pc{};pc.abi_version=REK_NATIVE_POLICY_ABI;pc.checkpoint_path=argv[2];pc.expected_sha256=argv[3];pc.hidden_size=hidden;pc.num_layers=layers;pc.batch=arenas;pc.precision=precision=="bf16"?REK_NATIVE_POLICY_BF16:REK_NATIVE_POLICY_FP32;pc.seed=seed;
        std::unique_ptr<RekNativePolicy,decltype(&rek_native_policy_destroy)> policy(rek_native_policy_create(&pc,stream),rek_native_policy_destroy);require(bool(policy),rek_native_policy_error());
        const std::string digest=rek_native_policy_sha256(policy.get());
        std::unique_ptr<RekNativePolicy,decltype(&rek_native_policy_destroy)> opponent_policy(nullptr,rek_native_policy_destroy);
        std::string opponent_digest;
        if(learned_opponent){
            RekNativePolicyConfig opc=pc;opc.checkpoint_path=argv[13];opc.expected_sha256=argv[14];opc.seed=uint64_t(seed)^0x9e3779b9ULL;
            opponent_policy.reset(rek_native_policy_create(&opc,stream));require(bool(opponent_policy),rek_native_policy_error());
            opponent_digest=rek_native_policy_sha256(opponent_policy.get());
        }
        long long total_wins=0,total_losses=0,total_ties=0;double total_wall=0;
        for(int side=0;side<2;side++){
            runtime_ok(rek_native5_reset(runtime,stream));policy_ok(rek_native_policy_reset(policy.get(),stream));if(opponent_policy)policy_ok(rek_native_policy_reset(opponent_policy.get(),stream));cuda_ok(cudaMemsetAsync(measures,0,arenas*sizeof(Measure),stream));
            cuda_ok(cudaMemsetAsync(counts,0,arenas*sizeof(int),stream));cuda_ok(cudaMemsetAsync(seen,0,arenas*sizeof(unsigned long long),stream));cuda_ok(cudaMemsetAsync(starts,0,arenas*sizeof(int),stream));cuda_ok(cudaMemsetAsync(device_tick,0,sizeof(int),stream));cuda_ok(cudaMemsetAsync(failures,0,sizeof(int),stream));
            set_overrides<<<(arenas*2+127)/128,128,0,stream>>>(overrides,arenas,side,learned_opponent);cuda_ok(cudaStreamSynchronize(stream));
            constexpr int chunk=64;
            auto step=[&](){runtime_ok(rek_native5_encode_fighter_observations(runtime,encoded,stream));policy_ok(rek_native_policy_step_rows(policy.get(),encoded,view.action_masks,view.terminals,actions,side,2,selection=="greedy",stream));if(opponent_policy)policy_ok(rek_native_policy_step_rows(opponent_policy.get(),encoded,view.action_masks,view.terminals,actions,side^1,2,selection=="greedy",stream));runtime_ok(rek_native5_step(runtime,stream));increment_tick<<<1,1,0,stream>>>(device_tick);collect_captured<<<(arenas+127)/128,128,0,stream>>>(view.rounds,view.raw_observations,view.actions,records,measures,counts,seen,starts,failures,arenas,rounds,side,device_tick);};
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
            runtime_ok(rek_native5_check_status(runtime,stream));policy_ok(rek_native_policy_check_status(policy.get(),stream));if(opponent_policy)policy_ok(rek_native_policy_check_status(opponent_policy.get(),stream));int failure=0;cuda_ok(cudaMemcpy(&failure,failures,sizeof(int),cudaMemcpyDeviceToHost));require(!failure,"Invalid terminal/measurement record, bits="+std::to_string(failure));
            std::vector<Record> host(size_t(arenas)*rounds);cuda_ok(cudaMemcpy(host.data(),records,host.size()*sizeof(Record),cudaMemcpyDeviceToHost));
            long long wins=0,losses=0,ties=0,points=0,opponent_points=0,falls=0,opponent_falls=0,zero_hits=0,first_hit_sum=0,hit_games=0,busy_sum=0,facing_sum=0,tick_sum=0;double path_sum=0,min_gap_sum=0;
            for(const auto& r:host){require(r.completed==static_cast<unsigned long long>(r.episode+1)&&r.duration_ticks>0,"Incomplete terminal record");if(r.winner<0)ties++;else if(r.winner==side)wins++;else losses++;
                points+=r.points[side];opponent_points+=r.points[side^1];
                zero_hits+=r.point_delta_sum[side]==0;
                if(r.first_hit_tick[side]){first_hit_sum+=r.first_hit_tick[side];hit_games++;}
                busy_sum+=r.busy_ticks;facing_sum+=r.facing_ticks;tick_sum+=r.duration_ticks;path_sum+=r.path_length;min_gap_sum+=r.min_gap;falls+=r.falls[side];opponent_falls+=r.falls[side^1];
                std::string first_hit=r.first_hit_tick[side]?std::to_string(r.first_hit_tick[side]*.02):"null";
                std::fprintf(output.get(),"{\"backend\":\"semantic_cuda\",\"policy_sha256\":\"%s\",\"opponent\":\"%s\",\"opponent_sha256\":\"%s\",\"fixture\":\"%s\",\"seed\":%d,\"arena\":%d,\"episode\":%d,\"policy_side\":%d,\"score\":[%d,%d],\"point_delta_sum\":[%d,%d],\"winner\":%d,\"round_result\":%d,\"duration_ticks\":%d,\"duration_seconds\":%.9g,\"first_hit_seconds\":%s,\"zero_hits\":%s,\"busy_ticks\":%d,\"facing_ticks\":%d,\"gap_below_1_05m_ticks\":%d,\"attack_commands\":%d,\"path_length_m\":%.9g,\"minimum_gap_m\":%.9g,\"maximum_gap_m\":%.9g,\"mean_gap_m\":%.9g,\"initial_xy\":[%.9g,%.9g,%.9g,%.9g],\"final_xy\":[%.9g,%.9g,%.9g,%.9g],\"shaping_weight\":0%s}\n",
                    digest.c_str(),opponent_name.c_str(),opponent_digest.c_str(),fixture.c_str(),seed,r.arena,r.episode,side,r.points[0],r.points[1],r.point_delta_sum[0],r.point_delta_sum[1],r.winner,r.round_result,r.duration_ticks,r.duration_ticks*.02,first_hit.c_str(),r.point_delta_sum[side]==0?"true":"false",r.busy_ticks,r.facing_ticks,r.near_ticks,r.attack_commands,r.path_length,r.min_gap,r.max_gap,r.mean_gap,r.initial_xy[0],r.initial_xy[1],r.initial_xy[2],r.initial_xy[3],r.final_xy[0],r.final_xy[1],r.final_xy[2],r.final_xy[3],identity_fields.c_str());
            }
            require(std::fflush(output.get())==0,"Could not flush match records");total_wins+=wins;total_losses+=losses;total_ties+=ties;
            std::printf("{\"event\":\"side_result\",\"policy_side\":%d,\"arenas\":%d,\"rounds_per_arena\":%d,\"wins\":%lld,\"losses\":%lld,\"draws\":%lld,\"points\":%lld,\"opponent_points\":%lld,\"falls\":%lld,\"opponent_falls\":%lld,\"evaluated_ticks\":%d,\"execution_wall_seconds\":%.9g}\n",side,arenas,rounds,wins,losses,ties,points,opponent_points,falls,opponent_falls,ticks,wall);std::fflush(stdout);
            std::string mean_first_hit=hit_games?std::to_string(first_hit_sum*.02/hit_games):"null";
            std::printf("{\"event\":\"behavior_result\",\"policy_side\":%d,\"opponent\":\"%s\",\"fixture\":\"%s\",\"configured_round_seconds\":%d,\"zero_hit_games\":%lld,\"hit_games\":%lld,\"mean_first_hit_seconds_when_hit\":%s,\"busy_fraction\":%.9g,\"facing_fraction\":%.9g,\"mean_path_length_m\":%.9g,\"mean_minimum_gap_m\":%.9g,\"shaping_weight\":0}\n",side,opponent_name.c_str(),fixture.c_str(),round_seconds,zero_hits,hit_games,mean_first_hit.c_str(),double(busy_sum)/tick_sum,double(facing_sum)/tick_sum,path_sum/(arenas*rounds),min_gap_sum/(arenas*rounds));
            cuda_ok(cudaGraphExecDestroy(executable));cuda_ok(cudaGraphDestroy(graph));
        }
        std::printf("{\"event\":\"frozen_policy_evaluation\",\"backend\":\"semantic_cuda\",\"checkpoint_sha256\":\"%s\",\"opponent_sha256\":\"%s\",\"precision\":\"%s\",\"observation_encoding\":\"scaled_polar_xy\",\"selection\":\"%s\",\"policy_rng_seed\":%d,\"opponent_rng_seed\":%llu,\"wins\":%lld,\"losses\":%lld,\"draws\":%lld,\"win_rate\":%.9g,\"execution_wall_seconds\":%.9g,\"fixture\":\"%s\",\"environment_randomizes_seed\":%s,\"opponent\":\"%s\",\"round_seconds\":%d,\"both_sides\":true,\"terminal_recurrent_reset\":true,\"failure_bits\":0,\"python_runtime\":false,\"cpu_physics\":false,\"training_sps\":null,\"shaping_weight\":0,\"physics_parity\":false%s}\n",
            digest.c_str(),opponent_digest.c_str(),precision.c_str(),selection.c_str(),seed,static_cast<unsigned long long>(uint64_t(seed)^0x9e3779b9ULL),total_wins,total_losses,total_ties,double(total_wins)/double(total_wins+total_losses+total_ties),total_wall,fixture.c_str(),fixture=="heldout"?"true":"false",opponent_name.c_str(),round_seconds,identity_fields.c_str());
        opponent_policy.reset();policy.reset();runtime_ok(rek_native5_close(runtime));for(void* p:owned)cuda_ok(cudaFree(p));cuda_ok(cudaStreamDestroy(stream));return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"diverse policy evaluation failed: %s\n",e.what());return 2;}
}
