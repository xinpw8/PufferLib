// Phase 0 dataset logger for lite falls.
//
// Runs any runtime implementing runtime_api.h (normally the physical MuJoCo +
// SONIC backend) with a mixed GPU behavior policy on both fighters, and
// aggregates on the GPU the sufficient statistics of the lite fall model:
// upright exposure ticks and fall onsets per (own move bin, opponent move bin,
// distance bin, closing bin, struck) cell, plus per-class outcome delays
// (recover, fallen, censored by a spawn reset or round end). Aggregation is
// lite_fall_dataset_observe.h, the code the CPU tests exercise. Binning is
// lite_falls.h, the code the lite runtime uses. Output is a
// rek.lite_fall_dataset.v1 JSON for fit_lite_falls.py.
//
// Usage: lite-fall-dataset CONFIG.json NEW_DATASET.json
#include "runtime_api.h"
#include "lite_fall_dataset_observe.h"
#include "../../../vendor/cJSON.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <openssl/sha.h>

namespace {
using namespace rek_lite_dataset;
constexpr int kStyles=4;  // aggressive, random, kicker, passive

void require(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
void cuda_ok(cudaError_t value){if(value!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(value));}
void runtime_ok(int value){if(value)throw std::runtime_error(rek_native5_error());}

struct Behavior {
    float style_cdf[kStyles];
    uint32_t seed;
    int arenas,external_opponent;
};
struct HoldState { int32_t action,ticks; };
struct DeviceAdd {
    __device__ void operator()(unsigned long long* p,unsigned long long v) const {atomicAdd(p,v);}
    __device__ void operator()(double* p,double v) const {atomicAdd(p,v);}
};

__host__ __device__ uint32_t mix(uint32_t v){v^=v>>16;v*=0x7feb352du;v^=v>>15;v*=0x846ca68bu;return v^(v>>16);}
__device__ float uniform(uint32_t seed,int row,int tick,uint32_t stream){
    return float(mix(seed^mix(uint32_t(row)*0x9e3779b9u+stream)^mix(uint32_t(tick)+0x85ebca6bu))>>8)*(1.f/16777216.f);
}
// Behavior policy. Styles are fixed per fighter row; all draws are counter-based.
__global__ void behave(const float* raw,const uint8_t* masks,HoldState* holds,float* actions,const Behavior b,const int* tick){
    const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=b.arenas*2)return;
    const float* o=raw+size_t(row)*223;const uint8_t* m=masks+size_t(row)*33;HoldState& h=holds[row];const int t=*tick;
    const float style_u=float(mix(b.seed^mix(uint32_t(row)+0x632be5abu))>>8)*(1.f/16777216.f);
    int style=0;while(style<kStyles-1&&style_u>=b.style_cdf[style])style++;
    if(h.ticks>0&&m[h.action]){h.ticks--;actions[row]=float(h.action);return;}
    const float dx=o[kOpponent+kX]-o[kX],dy=o[kOpponent+kY]-o[kY],distance=hypotf(dx,dy);
    const float yaw=atan2f(2.f*(o[kQw]*o[kQz]+o[kQx]*o[kQy]),1.f-2.f*(o[kQy]*o[kQy]+o[kQz]*o[kQz]));
    float bearing=atan2f(dy,dx)-yaw;bearing=atan2f(sinf(bearing),cosf(bearing));
    const float u=uniform(b.seed,row,t,1),v=uniform(b.seed,row,t,2);
    int action=1;
    if(style==1){
        // Random legal action held for 5 to 30 ticks.
        int choice=int(u*33.f)%33;for(int k=0;k<33&&!m[choice];k++)choice=(choice+1)%33;
        h.action=m[choice]?choice:0;h.ticks=5+int(v*25.f);action=h.action;
    }else if(style==3){
        action=u<.1f?(bearing>0?6:7):1;
    }else{
        const float reach=.6f+.5f*uniform(b.seed,row,t/100,3);
        if(fabsf(bearing)>.16f)action=bearing>0?6:7;
        else if(distance>reach+.1f)action=u<.85f?2:(v<.5f?4:5);
        else if(distance<.45f)action=3;
        else action=u<.6f?(style==2?16+int(v*4.f)%4:16+int(v*17.f)%17):1;
    }
    actions[row]=float(m[action]?action:(m[1]?1:0));
}
__global__ void observe(const float* raw,RowState* rows,float* previous_distance,const Settings s,
        const Sink sink,const RekNative5RoundResult* rounds,int arenas,const int* tick){
    const int arena=blockIdx.x*blockDim.x+threadIdx.x;if(arena>=arenas)return;
    observe_arena(raw+size_t(arena)*2*223,rows+arena*2,previous_distance[arena],s,rounds[arena].terminal!=0,*tick,sink,DeviceAdd{});
}
__global__ void increment(int* tick){if(!blockIdx.x&&!threadIdx.x)++*tick;}
__global__ void set_overrides(uint8_t* values,int arenas,int external_opponent){
    const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row<arenas*2)values[row]=(row%2==0||external_opponent)?1:0;
}

const cJSON* field(const cJSON* o,const char* k){auto* v=cJSON_GetObjectItemCaseSensitive(o,k);require(v,std::string("Missing config field ")+k);return v;}
std::string text(const cJSON* v){require(cJSON_IsString(v)&&v->valuestring,"Expected string");return v->valuestring;}
double number(const cJSON* v){require(cJSON_IsNumber(v)&&std::isfinite(v->valuedouble),"Expected finite number");return v->valuedouble;}
int whole(const cJSON* v,int lo,int hi){const double x=number(v);require(x==std::floor(x)&&x>=lo&&x<=hi,"Integer out of range");return int(x);}
void floats(const cJSON* o,const char* k,float* out,int n){
    const auto* a=field(o,k);require(cJSON_IsArray(a)&&cJSON_GetArraySize(a)==n,std::string(k)+" length");
    for(int i=0;i<n;i++)out[i]=float(number(cJSON_GetArrayItem(a,i)));
    for(int i=1;i<n;i++)require(out[i]>out[i-1],std::string(k)+" must increase");
}
std::string sha256_hex(const std::string& bytes){
    unsigned char d[SHA256_DIGEST_LENGTH];SHA256(reinterpret_cast<const unsigned char*>(bytes.data()),bytes.size(),d);
    char h[65];for(int i=0;i<32;i++)std::snprintf(h+2*i,3,"%02x",d[i]);return h;
}
template<class T>T* device(std::vector<void*>& owned,size_t n){T* p=nullptr;cuda_ok(cudaMalloc(&p,n*sizeof(T)));owned.push_back(p);cuda_ok(cudaMemset(p,0,n*sizeof(T)));return p;}
template<class T>void download(std::vector<T>& out,const T* in){cuda_ok(cudaMemcpy(out.data(),in,out.size()*sizeof(T),cudaMemcpyDeviceToHost));}
}  // namespace

int main(int argc,char** argv){
    try{
        require(argc==3,"Usage: lite-fall-dataset CONFIG.json NEW_DATASET.json");
        std::ifstream file(argv[1]);require(bool(file),"Config missing");
        const std::string config_text((std::istreambuf_iterator<char>(file)),{});
        std::unique_ptr<cJSON,decltype(&cJSON_Delete)> json(cJSON_Parse(config_text.c_str()),cJSON_Delete);
        require(json&&cJSON_IsObject(json.get()),"Invalid config JSON");
        require(text(field(json.get(),"schema"))=="rek.lite_fall_dataset_config.v1","Config schema");
        Behavior b{};Settings s{};
        b.arenas=whole(field(json.get(),"arenas"),1,65536);
        const int ticks=whole(field(json.get(),"ticks"),1,INT32_MAX-1);
        b.seed=uint32_t(whole(field(json.get(),"seed"),0,INT32_MAX));
        s.max_delay=whole(field(json.get(),"max_delay_ticks"),50,30000);
        const std::string opponent=text(field(json.get(),"opponent"));
        require(opponent=="external"||opponent=="internal","opponent must be external or internal");
        b.external_opponent=opponent=="external";
        floats(json.get(),"distance_edges_m",s.distance_edges,kDistanceBins-1);
        floats(json.get(),"closing_edges_m_s",s.closing_edges,kClosingBins-1);
        {   const auto* w=field(json.get(),"style_weights");require(cJSON_IsArray(w)&&cJSON_GetArraySize(w)==kStyles,"style_weights length");
            double total=0,acc=0;for(int i=0;i<kStyles;i++){const double x=number(cJSON_GetArrayItem(w,i));require(x>=0,"style weight");total+=x;}
            require(total>0,"style weights sum");for(int i=0;i<kStyles;i++){acc+=number(cJSON_GetArrayItem(w,i));b.style_cdf[i]=float(acc/total);}}
        // Explicit environment for the runtime under test; never inherited implicitly.
        const auto* env=field(json.get(),"env");require(cJSON_IsObject(env),"env must be an object");
        for(const cJSON* item=env->child;item;item=item->next){require(cJSON_IsString(item),"env values must be strings");setenv(item->string,item->valuestring,1);}
        const auto* rt=field(json.get(),"runtime");
        const std::string model=text(field(rt,"model_path")),assets=text(field(rt,"assets_path")),features=text(field(rt,"motion_features_path"));
        const auto optional=[&](const char* k){const auto* v=cJSON_GetObjectItemCaseSensitive(rt,k);return v?text(v):std::string();};
        const std::string physics=optional("physics_export_path"),encoder=optional("controller_encoder_path"),decoder=optional("controller_decoder_path");
        RekNative5Config cfg{};cfg.abi_version=REK_NATIVE5_RUNTIME_ABI;cfg.model_path=model.c_str();cfg.assets_path=assets.c_str();
        cfg.motion_features_path=features.c_str();cfg.physics_export_path=physics.empty()?nullptr:physics.c_str();
        cfg.controller_encoder_path=encoder.empty()?nullptr:encoder.c_str();cfg.controller_decoder_path=decoder.empty()?nullptr:decoder.c_str();
        cfg.arenas=b.arenas;cfg.seed=b.seed;cfg.locomotion_segment_ticks=1;cfg.round_seconds=float(number(field(rt,"round_seconds")));
        const uint32_t defaults[17]={35,27,31,45,32,45,157,145,158,139,134,138,73,75,68,71,103};
        for(int i=0;i<17;i++)s.durations[i]=cfg.move_duration_ticks[i]=defaults[i];
        std::unique_ptr<FILE,decltype(&std::fclose)> out(std::fopen(argv[2],"wx"),std::fclose);require(bool(out),"Dataset file must be new and writable");

        cudaStream_t stream;cuda_ok(cudaStreamCreate(&stream));std::vector<void*> owned;
        RekNative5Buffers buffers{};buffers.observations=device<float>(owned,size_t(b.arenas)*223);buffers.actions=device<float>(owned,b.arenas);
        buffers.rewards=device<float>(owned,b.arenas);buffers.terminals=device<float>(owned,b.arenas);
        buffers.logs=device<RekNative5Log>(owned,b.arenas);buffers.log_stride_bytes=sizeof(RekNative5Log);
        RekNative5Runtime* runtime=rek_native5_create(&cfg,&buffers,stream);require(runtime,rek_native5_error());
        RekNative5DeviceView view{};runtime_ok(rek_native5_get_device_view(runtime,&view));
        float* actions=device<float>(owned,size_t(b.arenas)*2);uint8_t* overrides=device<uint8_t>(owned,size_t(b.arenas)*2);
        set_overrides<<<(b.arenas*2+127)/128,128,0,stream>>>(overrides,b.arenas,b.external_opponent);
        runtime_ok(rek_native5_bind_external_actions(runtime,actions,overrides,stream));
        HoldState* holds=device<HoldState>(owned,size_t(b.arenas)*2);RowState* rows=device<RowState>(owned,size_t(b.arenas)*2);
        {   std::vector<RowState> initial(size_t(b.arenas)*2,RowState{-1,0,0,0,0});
            cuda_ok(cudaMemcpy(rows,initial.data(),initial.size()*sizeof(RowState),cudaMemcpyHostToDevice));}
        float* previous=device<float>(owned,b.arenas);
        {   std::vector<float> minus(b.arenas,-1.f);cuda_ok(cudaMemcpy(previous,minus.data(),minus.size()*sizeof(float),cudaMemcpyHostToDevice));}
        const size_t delays=size_t(kOutcomeClasses)*(s.max_delay+1);
        Sink sink{device<unsigned long long>(owned,kCells),device<unsigned long long>(owned,kCells),
            device<unsigned long long>(owned,delays),device<unsigned long long>(owned,delays),
            device<unsigned long long>(owned,delays),device<double>(owned,6)};
        int* tick=device<int>(owned,1);
        runtime_ok(rek_native5_reset(runtime,stream));
        constexpr int chunk=64;
        auto step=[&](){
            behave<<<(b.arenas*2+127)/128,128,0,stream>>>(view.raw_observations,view.action_masks,holds,actions,b,tick);
            runtime_ok(rek_native5_step(runtime,stream));increment<<<1,1,0,stream>>>(tick);
            observe<<<(b.arenas+127)/128,128,0,stream>>>(view.raw_observations,rows,previous,s,sink,view.rounds,b.arenas,tick);
        };
        cuda_ok(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));for(int t=0;t<chunk;t++)step();
        cudaGraph_t graph;cuda_ok(cudaStreamEndCapture(stream,&graph));cudaGraphExec_t exec;cuda_ok(cudaGraphInstantiate(&exec,graph,0));
        const auto start=std::chrono::steady_clock::now();int done=0;
        while(done<ticks){
            cuda_ok(cudaGraphLaunch(exec,stream));done+=chunk;
            if(done%(chunk*100)==0||done>=ticks){
                runtime_ok(rek_native5_check_status(runtime,stream));
                const double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                std::fprintf(stderr,"lite_fall_dataset progress ticks=%d arena_ticks_per_second=%.1f\n",done,done*double(b.arenas)/wall);
            }
        }
        cuda_ok(cudaStreamSynchronize(stream));
        const double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        HostSink host(s.max_delay);
        download(host.exposures,sink.exposures);download(host.onsets,sink.onsets);download(host.recover,sink.recover);
        download(host.fallen,sink.fallen);download(host.censored,sink.censored);
        cuda_ok(cudaMemcpy(host.observation,sink.observation,sizeof(host.observation),cudaMemcpyDeviceToHost));
        unsigned long long exposure=0,onsets=0;for(int i=0;i<kCells;i++){exposure+=host.exposures[i];onsets+=host.onsets[i];}
        const std::string id=sha256_hex(config_text);const char* backend=std::getenv("REK_PHYSICS_BACKEND");
        char provenance[1024];
        std::snprintf(provenance,sizeof(provenance),"{\"config_sha256\":\"%s\",\"backend\":\"%s\",\"arenas\":%d,\"ticks\":%d,\"seed\":%u,\"opponent\":\"%s\",\"exposure_ticks\":%llu,\"onsets\":%llu,\"execution_wall_seconds\":%.6f,\"arena_ticks_per_second\":%.3f}",
            id.c_str(),backend?backend:"unset",b.arenas,done,b.seed,opponent.c_str(),exposure,onsets,wall,double(done)*b.arenas/wall);
        write_dataset(out.get(),id,s,host,provenance);require(std::fflush(out.get())==0,"Could not write dataset");
        std::printf("{\"event\":\"lite_fall_dataset\",\"dataset_id\":\"%s\",\"exposure_ticks\":%llu,\"onsets\":%llu,\"execution_wall_seconds\":%.3f}\n",id.c_str(),exposure,onsets,wall);
        runtime_ok(rek_native5_check_status(runtime,stream));
        cuda_ok(cudaGraphExecDestroy(exec));cuda_ok(cudaGraphDestroy(graph));runtime_ok(rek_native5_close(runtime));
        for(void* p:owned)cuda_ok(cudaFree(p));cuda_ok(cudaStreamDestroy(stream));return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"lite fall dataset failed: %s\n",e.what());return 2;}
}
