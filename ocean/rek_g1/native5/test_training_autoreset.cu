// Exercise the exact compact runtime functions, including graph replay.
// Synthetic state is a software contract fixture, never REK parity evidence.
#include "fast_runtime.cu"

__global__ void fixture_reset(View v){
    if(threadIdx.x||blockIdx.x)return;
    auto* p=const_cast<Parameters*>(v.p);*p={};p->round_seconds=120;
    p->spawn[0][0]=-2;p->spawn[1][0]=2;
    p->half_extent[0]=p->half_extent[1]=10;p->settle_speed=.01f;
    for(int i=0;i<24;i++){p->routes[i].count=1;p->routes[i].loop=1;}
    for(int s=0;s<2;s++)for(int j=0;j<29;j++){
        p->qindices[s][j]=s*36+7+j;p->vindices[s][j]=s*35+6+j;
    }
    auto* f=const_cast<FastFrame*>(v.frames);*f={};f->root_z=1;f->root_wxyz[0]=1;
    for(int i=0;i<3;i++){
        v.rounds[i]={};auto& r=v.rounds[i];r.round_number=7;
        r.completed_rounds=2;r.wins[0]=1;r.completed_points[0]=13;
        r.terminal=i!=1;r.failure_bits=i==2?4:0;r.points[0]=7;r.points[1]=3;
        auto& a=v.state[i];a={};pose_reset(*p,a);
        a.tick=6000;a.fighter[0].x=3;a.fighter[1].x=5;
        a.fighter[0].points=7;a.fighter[1].points=3;
        a.reward[0]=4;a.reward[1]=-4;
        v.out.actions[i]=1;
    }
}
__global__ void exercise_reset(const __grid_constant__ View v){
    int i=threadIdx.x/32,lane=threadIdx.x%32;if(i>=3)return;
    export_arena(v,i,lane);__syncwarp();training_autoreset(v,i,lane);
}
void require_test(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
template<class T>T* managed(size_t n,std::vector<void*>& allocations){
    T* p=nullptr;rek5::cuda_check(cudaMallocManaged(&p,n*sizeof(T)));
    rek5::cuda_check(cudaMemset(p,0,n*sizeof(T)));allocations.push_back(p);return p;
}
int main(){try{
    std::vector<void*> mem;View v{};v.arenas=3;
    v.p=managed<Parameters>(1,mem);v.frames=managed<FastFrame>(1,mem);
    v.state=managed<Arena>(3,mem);v.rounds=managed<RekNative5RoundResult>(3,mem);
    v.raw=managed<float>(3*446,mem);v.qpos=managed<float>(3*72,mem);
    v.qvel=managed<float>(3*70,mem);v.actions=managed<float>(6,mem);
    v.rewards=managed<float>(6,mem);v.terminals=managed<float>(6,mem);
    v.masks=managed<uint8_t>(3*66,mem);v.learner_masks=managed<uint8_t>(3*33,mem);
    v.out.observations=managed<float>(3*223,mem);v.out.actions=managed<float>(3,mem);
    v.out.rewards=managed<float>(3,mem);v.out.terminals=managed<float>(3,mem);
    v.out.logs=managed<RekNative5Log>(3,mem);v.out.log_stride_bytes=sizeof(RekNative5Log);
    cudaStream_t stream;rek5::cuda_check(cudaStreamCreate(&stream));
    rek5::cuda_check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    fixture_reset<<<1,1,0,stream>>>(v);exercise_reset<<<1,96,0,stream>>>(v);
    cudaGraph_t graph;cudaGraphExec_t executable;
    rek5::cuda_check(cudaStreamEndCapture(stream,&graph));
    rek5::cuda_check(cudaGraphInstantiate(&executable,graph,0));
    unsigned checks=0;
    for(int replay=0;replay<4;replay++){
        rek5::cuda_check(cudaGraphLaunch(executable,stream));rek5::cuda_check(cudaStreamSynchronize(stream));
        auto check=[&](bool ok,const char* what){++checks;require_test(ok,what);};
        check(v.out.rewards[0]==4&&v.rewards[0]==4&&v.rewards[1]==-4,"lost terminal reward");
        check(v.out.terminals[0]==1&&v.terminals[0]==1&&v.terminals[1]==1,"lost recurrent reset mask");
        check(v.rounds[0].round_number==8&&v.rounds[0].terminal==0,"wrong next episode state");
        check(v.rounds[0].completed_rounds==2&&v.rounds[0].wins[0]==1&&v.rounds[0].completed_points[0]==13,"lost cumulative metrics");
        check(v.state[0].tick==0&&v.state[0].fighter[0].points==0,"state not reset");
        check(v.out.observations[0]==-2&&v.out.observations[190]==0&&v.out.observations[189]==1,"next observation not initial");
        check(v.raw[189]==120&&v.raw[223]==2,"frozen opponent observes final pose");
        check(v.masks[16]&&v.learner_masks[16],"initial action masks missing");
        check(v.rounds[1].round_number==7&&v.state[1].tick==6000&&v.out.observations[223]==3,"nonterminal state mutated");
        check(v.rounds[2].round_number==7&&v.rounds[2].failure_bits==4&&v.out.observations[446]==3,"failure reset away");
        View single=v;single.arenas=1;
        fast_step<<<1,32,0,stream>>>(single,true);rek5::cuda_check(cudaStreamSynchronize(stream));
        check(v.rounds[0].round_number==8&&v.state[0].tick==1,"double reset or dropped first action");
        check(v.out.terminals[0]==0&&v.out.rewards[0]==0,"terminal transition repeated");
    }
    printf("{\"test\":\"compact_training_autoreset\",\"checks\":%u,\"graph_replays\":4,\"status\":\"passed\",\"server_parity\":false}\n",checks);
    rek5::cuda_check(cudaGraphExecDestroy(executable));rek5::cuda_check(cudaGraphDestroy(graph));
    rek5::cuda_check(cudaStreamDestroy(stream));for(void* p:mem)rek5::cuda_check(cudaFree(p));
    return 0;
}catch(const std::exception& e){fprintf(stderr,"autoreset test: %s\n",e.what());return 1;}}
