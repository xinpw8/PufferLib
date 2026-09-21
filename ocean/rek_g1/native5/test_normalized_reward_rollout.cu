#include "runtime_api.h"
#include "normalized_reward.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

static void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
static void gpu(cudaError_t status){check(status==cudaSuccess,cudaGetErrorString(status));}
template<class T> static T* alloc(size_t count,std::vector<void*>& memory){
    T* p=nullptr;gpu(cudaMallocManaged(&p,count*sizeof(T)));gpu(cudaMemset(p,0,count*sizeof(T)));memory.push_back(p);return p;
}
int main(int argc,char** argv){try{
    check(argc==5,"MODEL EXPORT ASSETS FEATURES required");
    constexpr int arenas=4,ticks=1600;
    std::vector<void*> memory;RekNative5Buffers b{};
    b.observations=alloc<float>(arenas*223,memory);b.actions=alloc<float>(arenas,memory);
    b.rewards=alloc<float>(arenas,memory);b.terminals=alloc<float>(arenas,memory);
    b.logs=alloc<RekNative5Log>(arenas,memory);b.log_stride_bytes=sizeof(RekNative5Log);
    auto* actions=alloc<float>(arenas*2,memory);auto* overrides=alloc<uint8_t>(arenas*2,memory);
    for(int i=0;i<arenas*2;i++)overrides[i]=2; // Recovered compact opponent on both sides.
    RekNative5Config c{};c.abi_version=REK_NATIVE5_RUNTIME_ABI;c.arenas=arenas;c.seed=419;c.round_seconds=10;
    c.model_path=argv[1];c.physics_export_path=argv[2];c.assets_path=argv[3];c.motion_features_path=argv[4];
    const unsigned durations[]={35,27,31,45,32,45,157,145,158,139,134,138,73,75,68,71,103};
    for(int i=0;i<17;i++)c.move_duration_ticks[i]=durations[i];c.locomotion_segment_ticks=25;
    cudaStream_t stream;gpu(cudaStreamCreate(&stream));
    auto* runtime=rek_native5_create(&c,&b,stream);check(runtime,rek_native5_error());
    check(rek_native5_bind_external_actions(runtime,actions,overrides,stream)==0,rek_native5_error());
    unsigned checks=0,point_steps=0,terminal_steps=0,reset_steps=0;
    unsigned last_round[arenas]={};double total_reward=0;long long score_difference=0;
    for(int tick=0;tick<ticks;tick++){
        check(rek_native5_step(runtime,stream)==0,rek_native5_error());
        for(int arena=0;arena<arenas;arena++){
            RekNative5Snapshot s{};check(rek_native5_read_snapshot(runtime,arena,&s,stream)==0,rek_native5_error());
            check(s.round.failure_bits==0,"runtime failure");
            if(last_round[arena]&&last_round[arena]!=s.round.round_number)reset_steps++;
            last_round[arena]=s.round.round_number;terminal_steps+=s.round.terminal!=0;
            for(int side=0;side<2;side++){
                const float* raw=s.raw_observations+side*223;
                const int own=int(raw[217]),other=int(raw[218]);
                check(raw[217]==own&&raw[218]==other,"noninteger score delta");
                check(raw[219]==0,"compact unexpectedly reports physical fall");
                const auto expected=rek5_normalized_reward::value(own,other,0);
                check(!expected.saturated,"unexpected reward saturation");
                check(s.rewards[side]==expected.reward,"runtime reward differs from awarded deltas");
                check(std::isfinite(s.rewards[side])&&std::abs(s.rewards[side])<=1,"reward outside normalized bounds");
                checks+=5;point_steps+=own||other;
                if(side==0){check(b.rewards[arena]==expected.reward,"learner buffer differs from snapshot");checks++;total_reward+=s.rewards[0];score_difference+=own-other;}
            }
        }
    }
    check(point_steps>0&&terminal_steps>0&&reset_steps>0,"missing scoring/terminal/reset coverage");
    check(std::abs(total_reward-double(score_difference)/100)<1e-5,"score return accumulated incorrectly");
    check(rek_native5_check_status(runtime,stream)==0,rek_native5_error());
    check(rek_native5_close(runtime)==0,rek_native5_error());
    gpu(cudaStreamDestroy(stream));for(void* p:memory)gpu(cudaFree(p));
    std::printf("{\"test\":\"normalized_compact_rollout\",\"ticks\":%d,\"arenas\":%d,\"checks\":%u,\"point_rows\":%u,\"terminals\":%u,\"round_resets\":%u,\"reward_sum\":%.9g,\"score_difference\":%lld,\"passed\":true,\"fall_dynamics_tested\":false}\n",ticks,arenas,checks,point_steps,terminal_steps,reset_steps,total_reward,score_difference);
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"normalized rollout: %s\n",e.what());return 2;}}
