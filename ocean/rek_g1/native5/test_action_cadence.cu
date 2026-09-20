// Small differential fixture for archived/current runtime objects. No learner.
// Real asset loading and runtime stepping; no claim of authentic game parity.
#include "runtime_api.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void cuda_ok(cudaError_t e){check(e==cudaSuccess,cudaGetErrorString(e));}
template<class T>T* allocate(size_t n,std::vector<void*>& mem){T* p=nullptr;cuda_ok(cudaMallocManaged(&p,n*sizeof(T)));cuda_ok(cudaMemset(p,0,n*sizeof(T)));mem.push_back(p);return p;}
template<class T>void write(FILE* f,const T* p,size_t n){check(fwrite(p,sizeof(T),n,f)==n,"fixture_write_failed");}
int main(int argc,char** argv){try{
 check(argc==6,"test-action-cadence MODEL EXPORT ASSETS FEATURES NEW_BINARY");
 FILE* output=fopen(argv[5],"wbx");check(output,"new_output_required");
 std::vector<void*> mem;RekNative5Buffers b{};
 b.observations=allocate<float>(3*223,mem);b.actions=allocate<float>(3,mem);
 b.rewards=allocate<float>(3,mem);b.terminals=allocate<float>(3,mem);
 b.logs=allocate<RekNative5Log>(3,mem);b.log_stride_bytes=sizeof(RekNative5Log);
 auto* masks=allocate<uint8_t>(3*33,mem);auto* external=allocate<float>(6,mem);
 auto* overrides=allocate<uint8_t>(6,mem);overrides[2]=2; // Bot1 controls arena1 learner side.
 RekNative5Config c{};c.abi_version=REK_NATIVE5_RUNTIME_ABI;c.arenas=3;c.seed=419;
 c.model_path=argv[1];c.physics_export_path=argv[2];c.assets_path=argv[3];c.motion_features_path=argv[4];c.round_seconds=2;
 const unsigned durations[]={35,27,31,45,32,45,157,145,158,139,134,138,73,75,68,71,103};
 for(int i=0;i<17;i++)c.move_duration_ticks[i]=durations[i];c.locomotion_segment_ticks=25;
 cudaStream_t stream;cuda_ok(cudaStreamCreate(&stream));
 auto* runtime=rek_native5_create(&c,&b,stream);check(runtime,rek_native5_error());
 check(rek_native5_bind_action_mask(runtime,masks,stream)==0,rek_native5_error());
 check(rek_native5_bind_external_actions(runtime,external,overrides,stream)==0,rek_native5_error());
 unsigned records=0,terminal_records=0,initial_records=0,busy_records=0,checks=0;
 for(int autoreset=0;autoreset<2;autoreset++){
  check(rek_native5_reset(runtime,stream)==0,rek_native5_error());
  for(int step=0;step<=110;step++){
   cuda_ok(cudaStreamSynchronize(stream));
   for(int arena=0;arena<3;arena++){
    RekNative5Snapshot s{};check(rek_native5_read_snapshot(runtime,arena,&s,stream)==0,rek_native5_error());
    check(!s.round.failure_bits,"runtime_failure_bits");checks++;
    int tick=int(std::lround((2.-s.round.time_remaining_seconds)*50.));
    const uint32_t tag[]={uint32_t(autoreset),uint32_t(step),uint32_t(arena),uint32_t(tick),uint32_t(s.round.terminal)};
    write(output,tag,5);write(output,s.action_masks,66);write(output,masks+arena*33,33);
    write(output,b.observations+arena*223,223);write(output,s.raw_observations,446);
    write(output,s.qpos,72);write(output,s.qvel,70);write(output,s.actions,2);
    write(output,s.rewards,2);write(output,s.terminals,2);
    write(output,b.rewards+arena,1);write(output,b.terminals+arena,1);
    // Avoid struct padding in the byte-level comparison.
    const double round[]={double(s.round.round_number),double(s.round.phase),double(s.round.round_result),double(s.round.round_winner),double(s.round.completed_rounds),double(s.round.wins[0]),double(s.round.wins[1]),double(s.round.ties),double(s.round.points[0]),double(s.round.points[1]),double(s.round.completed_points[0]),double(s.round.completed_points[1]),double(s.round.time_remaining_seconds),double(s.round.failure_bits)};
    write(output,round,14);records++;terminal_records+=s.round.terminal!=0;
    initial_records+=tick==0;busy_records+=s.raw_observations[182]!=0;
    // Exactly the same external decisions under all three binaries. Hold0 on
    // intervening rows retains owned state and never retriggers an attack.
    int action=0;
    if(tick==0)action=6;
    if(tick==5)action=23; // native move3,45 ticks
    if(tick==10)action=7;
    if(tick==15)action=1;
    if(tick==20)action=6;
    if(tick==55)action=17; // native move7,145 ticks; terminal cuts it short
    if(s.round.terminal)action=0;
    if(arena!=1){check(s.action_masks[action]!=0&&masks[arena*33+action]!=0,"fixed_action_not_legal");checks++;}
    b.actions[arena]=float(action);
   }
   if(step<110)check((autoreset?rek_native5_step_autoreset(runtime,stream):rek_native5_step(runtime,stream))==0,rek_native5_error());
  }
 }
 check(terminal_records>0&&initial_records>6&&busy_records>0,"fixture_missing_terminal_reset_or_busy");
 check(fclose(output)==0,"fixture_close_failed");check(rek_native5_close(runtime)==0,rek_native5_error());
 cuda_ok(cudaStreamDestroy(stream));for(void* p:mem)cuda_ok(cudaFree(p));
 printf("{\"test\":\"action_cadence_runtime_fixture\",\"records\":%u,\"checks\":%u,\"terminal_records\":%u,\"initial_records\":%u,\"busy_records\":%u,\"passed\":true,\"training\":false,\"authentic_parity\":false}\n",records,checks,terminal_records,initial_records,busy_records);
 return 0;
}catch(const std::exception& e){fprintf(stderr,"cadence fixture: %s\n",e.what());return 2;}}
