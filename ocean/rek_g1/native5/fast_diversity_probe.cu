#include "runtime_api.h"
#include "fast_assets.h"
#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
void ck(cudaError_t x){if(x!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(x));}
void ok(int x){if(x)throw std::runtime_error(rek_native5_error());}
void require(bool x,const char* message){if(!x)throw std::runtime_error(message);}
void options(const char* mode,bool random,float weight=0){
    setenv("REK_PHYSICS_BACKEND","semantic_cuda",1);setenv("REK_FAST_OPPONENT_MODE",mode,1);
    setenv("REK_FAST_MOVE_SPEED","1",1);setenv("REK_FAST_YAW_SPEED","1.8",1);
    setenv("REK_FAST_BODY_RADIUS",".22",1);setenv("REK_FAST_HIT_SPEED",".35",1);
    setenv("REK_FAST_RANDOM_RESETS",random?"1":"0",1);
    setenv("REK_FAST_RESET_GAP_MIN",".55",1);setenv("REK_FAST_RESET_GAP_MAX","2.5",1);
    setenv("REK_FAST_RESET_HEADING_SPREAD_RAD",".2",1);
    setenv("REK_FAST_SHAPING_WEIGHT",weight?".7":"0",1);
    setenv("REK_FAST_SHAPING_GAMMA",".999",1);setenv("REK_FAST_SHAPING_TARGET",".65",1);
    setenv("REK_FAST_SHAPING_BEARING_WEIGHT",".25",1);
}
__global__ void actions(float* values,uint8_t* overrides,int arenas,int action,bool opponent){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=arenas*2)return;
    values[i]=i%2?1:action;overrides[i]=i%2&&opponent?2:1;
}
struct Fixture {
    static constexpr int A=64;
    cudaStream_t stream{};RekNative5Runtime* runtime{};RekNative5Config config{};
    RekNative5Buffers buffers{};float* external{};uint8_t* overrides{};
    std::vector<void*> owned;
    template<class T>T* alloc(size_t n){T* p{};ck(cudaMalloc(&p,n*sizeof(T)));owned.push_back(p);ck(cudaMemset(p,0,n*sizeof(T)));return p;}
    Fixture(char** paths,uint32_t seed,float seconds=.12f){
        ck(cudaStreamCreate(&stream));config.abi_version=REK_NATIVE5_RUNTIME_ABI;
        config.model_path=paths[0];config.physics_export_path=paths[1];config.assets_path=paths[2];config.motion_features_path=paths[3];
        config.arenas=A;config.seed=seed;config.round_seconds=seconds;config.locomotion_segment_ticks=1;
        const unsigned d[17]={35,27,31,45,32,45,157,145,158,139,134,138,73,75,68,71,103};
        for(int i=0;i<17;i++)config.move_duration_ticks[i]=d[i];
        buffers.observations=alloc<float>(A*223);buffers.actions=alloc<float>(A);
        buffers.rewards=alloc<float>(A);buffers.terminals=alloc<float>(A);
        buffers.logs=alloc<RekNative5Log>(A);buffers.log_stride_bytes=sizeof(RekNative5Log);
        runtime=rek_native5_create(&config,&buffers,stream);require(runtime,rek_native5_error());
        external=alloc<float>(A*2);overrides=alloc<uint8_t>(A*2);
        ok(rek_native5_bind_external_actions(runtime,external,overrides,stream));
    }
    ~Fixture(){if(runtime)rek_native5_close(runtime);for(void* p:owned)cudaFree(p);if(stream)cudaStreamDestroy(stream);}
    RekNative5Snapshot snapshot(int arena=0){RekNative5Snapshot s{};ok(rek_native5_read_snapshot(runtime,arena,&s,stream));require(!s.round.failure_bits,"runtime failure");for(float x:s.raw_observations)require(std::isfinite(x),"nonfinite observation");return s;}
    void tick(int action=1,bool opponent=false){actions<<<1,128,0,stream>>>(external,overrides,A,action,opponent);ok(rek_native5_step(runtime,stream));}
    void reset(){ok(rek_native5_reset(runtime,stream));}
};
using Roots=std::array<float,8>;
Roots roots(const RekNative5Snapshot& s){const float* o=s.raw_observations;return {o[0],o[1],o[172],o[175],o[223],o[224],o[395],o[398]};}
float wrap(float x){return std::atan2(std::sin(x),std::cos(x));}
uint32_t hash(uint32_t x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
int mode_for(uint32_t seed,int arena,int round){return int(hash(hash(seed^hash(uint32_t(arena)+0x9e3779b9u)^hash(uint32_t(round)+0x85ebca6bu))^0xa511e9b3u)&3u);}
int expected_action(const RekNative5Snapshot& s,int mode){
    const float* o=s.raw_observations+223;const float dx=o[86]-o[0],dy=o[87]-o[1];
    const float bearing=wrap(std::atan2(dy,dx)-2*std::atan2(o[175],o[172]));
    if(mode==1)return 1;if(std::fabs(bearing)>.16f)return bearing>0?6:7;
    const float gap=std::hypot(dx,dy);
    if(mode==0)return gap>1.05f?2:(gap<.55f?3:16);
    if(mode==2)return gap<1.25f?3:1;return 5;
}
std::vector<Roots> sample(Fixture& f,bool validate){
    std::vector<Roots> result;auto assets=load_fast_assets(f.config);
    for(int i=0;i<Fixture::A;i++){
        auto s=f.snapshot(i);Roots r=roots(s);result.push_back(r);
        if(validate){
            float dx=r[4]-r[0],dy=r[5]-r[1],gap=std::hypot(dx,dy),axis=std::atan2(dy,dx);
            require(gap>=.55f-1e-5f&&gap<=2.5f+1e-5f,"random gap out of range");
            for(int side=0;side<2;side++){
                require(std::fabs(r[side*4])<=assets.arena_half_extent[0]-.22f-.0099f,"random x outside interior");
                require(std::fabs(r[side*4+1])<=assets.arena_half_extent[1]-.22f-.0099f,"random y outside interior");
                float yaw=2*std::atan2(r[side*4+3],r[side*4+2]);
                require(std::fabs(wrap(yaw-axis-(side?3.14159265358979323846f:0)))<=.20001f,"heading spread exceeded");
            }
        }
    }
    return result;
}
double potential(const RekNative5Snapshot& s,int side){
    const float* o=s.raw_observations+side*223;
    double dx=double(o[86])-o[0],dy=double(o[87])-o[1],error=std::fabs(std::hypot(dx,dy)-.65);
    double yaw=2*std::atan2(double(o[175]),o[172]);
    double bearing=std::fabs(std::atan2(std::sin(std::atan2(dy,dx)-yaw),std::cos(std::atan2(dy,dx)-yaw)))/3.14159265358979323846;
    return -(error/(1+error)+.25*bearing)/1.25;
}
void resets_and_modes(char** paths){
    options("scripted",false);Fixture fixed(paths,73);auto fixed_start=sample(fixed,false);
    for(int i=0;i<7;i++)fixed.tick();require(sample(fixed,false)==fixed_start,"fixed roots changed across rounds");
    options("mixed",true);Fixture randomized(paths,73);auto original=sample(randomized,true);
    int mode_counts[4]={};std::vector<int> expected;
    for(int i=0;i<Fixture::A;i++){int mode=mode_for(73,i,1);mode_counts[mode]++;expected.push_back(expected_action(randomized.snapshot(i),mode));}
    randomized.tick(1,true);for(int i=0;i<Fixture::A;i++)require(randomized.snapshot(i).actions[1]==expected[i],"mixed seeded mode action mismatch");
    randomized.reset();require(sample(randomized,true)==original,"explicit reset did not replay seed exactly");
    for(int i=0;i<7;i++)randomized.tick();auto next=sample(randomized,true);require(next!=original,"next round reused all randomized starts");
    options("mixed",true);Fixture other(paths,74);require(sample(other,true)!=original,"changed seed did not change starts");
    for(int n:mode_counts)require(n>0,"mixed fixture omitted an opponent mode");
    const char* modes[]={"scripted","neutral","retreat","strafe"};
    for(int mode=0;mode<4;mode++){
        options(modes[mode],true);setenv("REK_FAST_RESET_GAP_MIN",".65",1);setenv("REK_FAST_RESET_GAP_MAX",".65",1);setenv("REK_FAST_RESET_HEADING_SPREAD_RAD","0",1);
        Fixture f(paths,73);auto before=f.snapshot();f.tick(1,true);auto after=f.snapshot();
        require(after.actions[1]==expected_action(before,mode),"explicit opponent mode mismatch");
        require(after.round.falls[0]==0&&after.round.falls[1]==0,"opponent mode created falls");
    }
    std::printf("{\"test\":\"v4_seeded_resets_and_modes\",\"status\":\"passed\",\"arenas\":64,\"seed\":73,\"changed_seed\":74,\"mixed_counts\":[%d,%d,%d,%d],\"explicit_reset_bitwise_replay\":true,\"fixed_defaults_preserved\":true,\"bounds_checked\":true}\n",mode_counts[0],mode_counts[1],mode_counts[2],mode_counts[3]);
}
void shaping(char** paths){
    options("neutral",true);std::vector<RekNative5Snapshot> baseline;
    {Fixture f(paths,73);baseline.push_back(f.snapshot());for(int i=0;i<20;i++){f.tick(i==0?2:1);baseline.push_back(f.snapshot());if(baseline.back().round.terminal)break;}}
    options("neutral",true,.7f);Fixture f(paths,73);auto before=f.snapshot();double discounted[2]={},discount=1,peak_error=0;
    for(size_t i=1;i<baseline.size();i++){
        f.tick(i==1?2:1);auto after=f.snapshot();const auto& ref=baseline[i];
        require(!std::memcmp(after.qpos,ref.qpos,sizeof(ref.qpos))&&!std::memcmp(after.raw_observations,ref.raw_observations,sizeof(ref.raw_observations)),"shaping changed actual state or observations");
        require(!std::memcmp(after.action_masks,ref.action_masks,sizeof(ref.action_masks)),"shaping changed legal actions");
        for(int side=0;side<2;side++){
            double expected=.7*(.999*(after.round.terminal?0:potential(after,side))-potential(before,side));
            double error=std::fabs(after.rewards[side]-expected);peak_error=std::max(peak_error,error);
            require(error<2e-6,"potential shaping reward mismatch");discounted[side]+=discount*after.rewards[side];
            require(after.round.points[side]==0&&after.round.falls[side]==0,"shaping fabricated points or falls");
        }
        discount*=.999;before=after;
    }
    require(before.round.terminal,"shaping fixture did not reach terminal");
    for(int side=0;side<2;side++)require(std::fabs(discounted[side]+.7*potential(baseline.front(),side))<2e-6,"discounted potential did not telescope to initial potential");
    std::printf("{\"test\":\"v4_potential_shaping\",\"status\":\"passed\",\"maximum_reward_error\":%.9g,\"terminal_potential_zero\":true,\"discounted_telescoping\":true,\"state_masks_scores_unchanged\":true,\"weight\":0.7,\"gamma\":0.999}\n",peak_error);
}
void rejected_options(char** paths){
    options("neutral",true);Fixture f(paths,73);int count=0;
    auto reject=[&](const char* key,const char* value){
        options("neutral",true);setenv(key,value,1);
        auto* bad=rek_native5_create(&f.config,&f.buffers,f.stream);
        if(bad)rek_native5_close(bad);require(!bad,"invalid diversity parameter was accepted");count++;
    };
    reject("REK_FAST_OPPONENT_MODE","unknown");reject("REK_FAST_RANDOM_RESETS","2");
    reject("REK_FAST_RESET_GAP_MIN",".1");reject("REK_FAST_RESET_GAP_MIN","3");
    reject("REK_FAST_RESET_GAP_MAX","100");reject("REK_FAST_RESET_HEADING_SPREAD_RAD","3.2");
    reject("REK_FAST_SHAPING_GAMMA","1.1");reject("REK_FAST_SHAPING_WEIGHT","-1");
    options("neutral",true,.7f);unsetenv("REK_FAST_SHAPING_GAMMA");
    auto* bad=rek_native5_create(&f.config,&f.buffers,f.stream);
    if(bad)rek_native5_close(bad);require(!bad,"positive shaping accepted implicit gamma");count++;
    options("neutral",true,.7f);setenv("REK_FAST_SHAPING_TARGET",".1",1);
    bad=rek_native5_create(&f.config,&f.buffers,f.stream);
    if(bad)rek_native5_close(bad);require(!bad,"shaping accepted unreachable overlap target");count++;
    std::printf("{\"test\":\"v4_parameter_rejection\",\"status\":\"passed\",\"invalid_cases\":%d}\n",count);
}
}
int main(int argc,char** argv){try{require(argc==5,"Usage: fast-diversity-probe MODEL EXPORT ASSETS FEATURES");resets_and_modes(argv+1);shaping(argv+1);rejected_options(argv+1);return 0;}catch(const std::exception& e){std::fprintf(stderr,"V4 diversity probe: %s\n",e.what());return 2;}}
