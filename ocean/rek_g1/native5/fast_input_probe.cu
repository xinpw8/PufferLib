#include "runtime_api.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

// Independent behavioral probe. Only the runtime CUDA kernel advances state.
// Host snapshots/assertions are diagnostics, never part of training.
namespace {
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void ok(int e){if(e)throw std::runtime_error(rek_native5_error());}
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
__global__ void exact_actions(float* actions,int arenas,int first,int second){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<2*arenas)actions[i]=float(i%2?second:first);
}
__global__ void check_episode_feature(RekNative5DeviceView view,const float* encoded,
    const float* learner,int* failures){
    int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=view.arenas*2)return;
    if(view.raw_observations[row*223+186]!=1.f)atomicOr(failures,1);
    if(encoded[row*223+186]!=1.f)atomicOr(failures,2);
    if(!(row%2)&&learner[(row/2)*223+186]!=1.f)atomicOr(failures,4);
}
__global__ void contact_script(RekNative5DeviceView view,float* actions,int tick){
    int arena=blockIdx.x*blockDim.x+threadIdx.x;if(arena>=view.arenas)return;
    int row=2*arena;const float* o=view.raw_observations+row*223;
    const uint8_t* mask=view.action_masks+row*33;
    float dx=o[86]-o[0],dy=o[87]-o[1],distance=hypotf(dx,dy);
    float yaw=2*atan2f(o[175],o[172]);
    float bearing=atan2f(sinf(atan2f(dy,dx)-yaw),cosf(atan2f(dy,dx)-yaw));
    int action=0;
    if(!o[183]){
        if(fabsf(bearing)>.06f)action=bearing>0?6:7;
        else if(distance>.75f)action=2;
        else if(distance<.52f)action=3;
        else{
            // Try every attack family over deterministic long intervals. A
            // continuing move gets no extra input or queued next attack.
            int attack=16+(tick/400)%16;
            action=mask[attack]?attack:1;
        }
    }
    actions[row]=float(action);actions[row+1]=1;
}
struct Probe {
    static constexpr int arenas=4;
    cudaStream_t stream=nullptr;
    RekNative5Runtime* runtime=nullptr;
    RekNative5DeviceView view{};
    RekNative5Buffers out{};
    float* actions=nullptr;
    std::vector<void*> allocations;
    template<class T>T* alloc(size_t n){T* p=nullptr;ck(cudaMalloc(&p,n*sizeof(T)));allocations.push_back(p);ck(cudaMemset(p,0,n*sizeof(T)));return p;}
    Probe(char** paths){
        setenv("REK_PHYSICS_BACKEND","semantic_cuda",1);ck(cudaStreamCreate(&stream));
        RekNative5Config c{};c.abi_version=REK_NATIVE5_RUNTIME_ABI;c.model_path=paths[0];
        c.physics_export_path=paths[1];c.assets_path=paths[2];c.motion_features_path=paths[3];
        c.arenas=arenas;c.seed=73;c.locomotion_segment_ticks=1;c.round_seconds=20;
        const unsigned durations[17]={35,27,31,45,32,45,157,145,158,139,134,138,73,75,68,71,103};
        for(int i=0;i<17;i++)c.move_duration_ticks[i]=durations[i];
        out.observations=alloc<float>(arenas*223);out.actions=alloc<float>(arenas);
        out.rewards=alloc<float>(arenas);out.terminals=alloc<float>(arenas);
        out.logs=alloc<RekNative5Log>(arenas);out.log_stride_bytes=sizeof(RekNative5Log);
        runtime=rek_native5_create(&c,&out,stream);require(runtime!=nullptr,rek_native5_error());
        actions=alloc<float>(arenas*2);auto* override_rows=alloc<uint8_t>(arenas*2);
        ck(cudaMemsetAsync(override_rows,1,arenas*2,stream));
        ok(rek_native5_bind_external_actions(runtime,actions,override_rows,stream));
        ok(rek_native5_get_device_view(runtime,&view));
    }
    ~Probe(){if(runtime)rek_native5_close(runtime);for(void* p:allocations)cudaFree(p);if(stream)cudaStreamDestroy(stream);}
    void reset(){ok(rek_native5_reset(runtime,stream));}
    void tick(int first,int second=1){exact_actions<<<1,32,0,stream>>>(actions,arenas,first,second);ok(rek_native5_step(runtime,stream));}
    void ticks(int n,int first,int second=1){for(int i=0;i<n;i++)tick(first,second);}
    RekNative5Snapshot snapshot(){RekNative5Snapshot s{};ok(rek_native5_read_snapshot(runtime,0,&s,stream));return s;}
};
float heading(const RekNative5Snapshot& s){return 2*std::atan2(s.raw_observations[175],s.raw_observations[172]);}
float delta_angle(float a,float b){return std::atan2(std::sin(a-b),std::cos(a-b));}
void finite(const RekNative5Snapshot& s){
    require(!s.round.failure_bits,"runtime reported a sticky failure");
    for(float f:s.raw_observations)require(std::isfinite(f),"nonfinite raw observation");
    for(float f:s.qpos)require(std::isfinite(f),"nonfinite rendered qpos");
}
void episode_observation(Probe& p){
    float* encoded=p.alloc<float>(Probe::arenas*2*223);int* failures=p.alloc<int>(1);
    auto check=[&](){
        ok(rek_native5_encode_fighter_observations(p.runtime,encoded,p.stream));
        check_episode_feature<<<1,32,0,p.stream>>>(p.view,encoded,p.out.observations,failures);
    };
    p.reset();check();
    for(unsigned episode=1;episode<=3;episode++){
        // Each independent neutral round is exactly 1,000 ticks / 20 seconds.
        // The first tick after an automatic reset was advanced below already.
        p.ticks(episode==1?1000:999,1);check();auto terminal=p.snapshot();
        require(terminal.round.terminal==1&&terminal.round.time_remaining_seconds==0,
            "neutral episode did not terminate after 20 seconds");
        require(terminal.round.completed_rounds==episode&&terminal.round.round_number==episode,
            "diagnostic completed-round counter did not remain cumulative");
        p.tick(1);check();auto next=p.snapshot();
        require(next.round.terminal==0&&next.round.round_number==episode+1,
            "automatic reset did not start the next independent episode");
        std::printf("{\"event\":\"episode_observation_boundary\",\"completed_rounds\":%llu,\"diagnostic_round_number\":%u,\"raw_round_feature\":%.9g}\n",
            (unsigned long long)next.round.completed_rounds,next.round.round_number,next.raw_observations[186]);
    }
    p.reset();check();auto fresh=p.snapshot();
    require(fresh.round.completed_rounds==0&&fresh.round.round_number==1,
        "explicit reset did not clear diagnostic round counters");
    int result=0;ck(cudaMemcpyAsync(&result,failures,sizeof(result),cudaMemcpyDeviceToHost,p.stream));
    ck(cudaStreamSynchronize(p.stream));
    std::printf("{\"event\":\"episode_observation_gpu_check\",\"failure_bits\":%d,\"bit1\":\"raw\",\"bit2\":\"encoded\",\"bit4\":\"learner\"}\n",result);
    require(result==0,
        "raw/encoded/learner observation 186 leaked cumulative episode count");
    std::printf("{\"test\":\"fast_episode_observation\",\"status\":\"passed\",\"contract_version\":2,\"arenas\":4,\"independent_rounds\":3,\"ticks_per_round\":1000,\"round_seconds\":20,\"policy_round_feature\":1,\"raw_encoded_learner_checked_on_gpu\":true,\"diagnostic_round_counter_cumulative\":true}\n");
}
void yaw_and_buffering(Probe& p){
    p.reset();p.ticks(25,6);auto turning=p.snapshot();require(turning.qvel[5]>.5f,"held Q did not establish yaw velocity");
    float before=heading(turning);p.tick(16);auto started=p.snapshot();
    require(started.raw_observations[183]==1,"attack over held Q did not start");
    require(std::fabs(started.qvel[5])<1e-7f,"Q rotation was not interrupted by attack");
    require(std::fabs(delta_angle(heading(started),before))<1e-6f,"attack moved logical held-yaw heading");
    require(started.action_masks[1]&&started.action_masks[6]&&started.action_masks[7],"busy mask cannot update or release desired yaw");
    require(!started.action_masks[17]&&!started.action_masks[2],"busy mask allows stacked attacks/translation");
    p.tick(17);p.tick(18);p.tick(2); // Deliberately unfiltered invalid/masked commands.
    for(int i=4;i<157;i++)p.tick(0);
    auto complete=p.snapshot();require(complete.raw_observations[183]==0&&complete.action_masks[16],"attack did not finish at157ticks");
    require(std::fabs(delta_angle(heading(complete),before))<1e-6f,"held rotation leaked through canned attack");
    p.ticks(20,0);auto resumed=p.snapshot();finite(resumed);
    require(resumed.qvel[5]>.5f&&delta_angle(heading(resumed),before)>.05f,"Q did not resume after attack");
    require(resumed.raw_observations[183]==0&&resumed.raw_observations[179]<7,"a discarded attack or translation queued later");

    p.reset();p.ticks(25,6);p.tick(16);p.tick(7);p.ticks(155,0);p.ticks(20,0);
    auto switched=p.snapshot();require(switched.qvel[5]<-.5f,"desired E entered during attack did not replace Q");
    p.reset();p.ticks(25,7);p.tick(16);p.tick(1);p.ticks(155,0);p.ticks(20,0);
    auto released=p.snapshot();require(std::fabs(released.qvel[5])<1e-7f,"neutral entered during attack did not release yaw");

    p.reset();p.ticks(10,2);p.tick(16);auto blocked=p.snapshot();
    require(blocked.raw_observations[183]==0,"attack started before translation settled");
    p.ticks(50,1);auto stopped=p.snapshot();
    require(stopped.raw_observations[183]==0&&stopped.action_masks[16],"translation attack was queued or failed to settle");
    p.tick(16);require(p.snapshot().raw_observations[183]==1,"attack did not start after translation settled");
    std::printf("{\"test\":\"fast_input_semantics\",\"status\":\"passed\",\"held_q_pause_resume\":true,\"desired_yaw_replace_release\":true,\"masked_attack_translation_discarded\":true,\"duration_ticks\":157,\"translation_settle_gate\":true}\n");
}
struct ContactResult {RekNative5Snapshot final{};unsigned max_falls=0;int max_points=0;};
ContactResult contacts(Probe& p){
    p.reset();ContactResult result;
    for(int t=0;t<6400;t++){
        contact_script<<<1,32,0,p.stream>>>(p.view,p.actions,t);ok(rek_native5_step(p.runtime,p.stream));
        if(t%50==49){
            auto s=p.snapshot();finite(s);
            result.max_falls=std::max(result.max_falls,s.round.falls[1]);
            result.max_points=std::max(result.max_points,s.round.points[0]);
        }
    }
    result.final=p.snapshot();ok(rek_native5_check_status(p.runtime,p.stream));return result;
}
}
int main(int argc,char** argv){try{
    require(argc==5,"Usage: fast-input-probe MODEL EXPORT ASSETS FEATURES");
    Probe probe(argv+1);episode_observation(probe);yaw_and_buffering(probe);
    ContactResult a=contacts(probe),b=contacts(probe);
    require(a.max_points>0,"no scripted attacks produced any contact score");
    require(a.max_falls==0,"contact scores fabricated a knockdown in the points-only compact model");
    require(a.max_falls==b.max_falls&&a.max_points==b.max_points&&
        a.final.round.completed_rounds==b.final.round.completed_rounds&&
        a.final.round.completed_points[0]==b.final.round.completed_points[0]&&
        std::memcmp(a.final.qpos,b.final.qpos,sizeof(a.final.qpos))==0,
        "reset replay changed deterministic contact outcomes");
    std::printf("{\"test\":\"fast_contact_replay\",\"status\":\"passed\",\"ticks_per_replay\":6400,\"maximum_points_in_round\":%d,\"maximum_opponent_falls\":%u,\"completed_rounds\":%llu,\"completed_player_points\":%lld,\"bitwise_qpos_replay\":true,\"cpu_physics\":false,\"python_runtime\":false}\n",a.max_points,a.max_falls,(unsigned long long)a.final.round.completed_rounds,(long long)a.final.round.completed_points[0]);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"fast input/contact probe: %s\n",e.what());return 2;}}
