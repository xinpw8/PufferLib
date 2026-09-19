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
__global__ void fixture_reward_terminal(View v,bool round_outcome){
    if(threadIdx.x||blockIdx.x)return;
    auto* p=const_cast<Parameters*>(v.p);
    p->opponent_mode=1;
    // Leave the zero-initialized default reward mode untouched in the control.
    if(round_outcome){p->reward_mode=rek5_round_reward::RoundOutcome;p->reward_gamma=.99988448f;}
    for(int i=0;i<3;i++){
        const int own=i==0?7:i==1?3:5,other=i==0?3:i==1?7:5;
        auto& r=v.rounds[i];r.terminal=0;r.failure_bits=0;
        r.wins[1]=1;r.completed_points[1]=17;r.points[0]=own;r.points[1]=other;
        auto& a=v.state[i];a.tick=5999;a.elapsed=float(a.tick)*DT;a.opponent_mode=1;
        a.fighter[0].points=own;a.fighter[1].points=other;
        a.episode_return=-.25f;a.episode_hits=2;
        auto* log=reinterpret_cast<RekNative5Log*>(reinterpret_cast<char*>(v.out.logs)+i*v.out.log_stride_bytes);
        *log={};log->score=13;log->episode_return=2.25f;log->episode_length=8000;
        log->hits=4;log->falls=1;log->wins=1;log->losses=1;log->actions_invalid=3;log->n=2;
    }
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
    const unsigned legacy_checks=checks;
    require_test(legacy_checks==48,"legacy checks changed");
    rek5::cuda_check(cudaGraphExecDestroy(executable));rek5::cuda_check(cudaGraphDestroy(graph));
    for(int mode=0;mode<2;mode++){
        rek5::cuda_check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        fixture_reset<<<1,1,0,stream>>>(v);fixture_reward_terminal<<<1,1,0,stream>>>(v,mode!=0);
        fast_step<<<1,96,0,stream>>>(v,true);
        rek5::cuda_check(cudaStreamEndCapture(stream,&graph));
        rek5::cuda_check(cudaGraphInstantiate(&executable,graph,0));
        for(int replay=0;replay<4;replay++){
            rek5::cuda_check(cudaGraphLaunch(executable,stream));rek5::cuda_check(cudaStreamSynchronize(stream));
            auto check=[&](bool ok,const char* what){++checks;require_test(ok,what);};
            auto near=[](float a,float b){return fabsf(a-b)<1e-6f;};
            check(v.p->reward_mode==(mode?rek5_round_reward::RoundOutcome:rek5_round_reward::PointDifference),"wrong reward mode fixture");
            check(!mode||v.p->reward_gamma==.99988448f,"wrong reward discount fixture");
            for(int i=0;i<3;i++){
                const int row=2*i,obs=223*i,raw=446*i;
                // Independent expected values: Phi(7,3)=4/(5+4), terminal Phi=0.
                const float expected=mode?(i==0?1.f-4.f/9.f:i==1?-1.f+4.f/9.f:0.f):0.f;
                const auto& r=v.rounds[i];const auto& a=v.state[i];
                check(near(v.out.rewards[i],expected)&&near(v.rewards[row],expected)&&near(v.rewards[row+1],-expected),"actual fast_step terminal reward wrong");
                check(v.out.terminals[i]==1&&v.terminals[row]==1&&v.terminals[row+1]==1,"actual terminal reset mask missing");
                check(r.round_number==8&&!r.terminal&&!r.failure_bits&&a.tick==0,"actual terminal did not autoreset once");
                check(a.fighter[0].points==0&&a.fighter[1].points==0&&r.points[0]==0&&r.points[1]==0,"terminal scoreboard leaked into new episode");
                check(v.out.observations[obs]==-2&&v.out.observations[obs+189]==1&&v.out.observations[obs+190]==0&&v.out.observations[obs+191]==0,"actual terminal observation not initial");
                check(v.raw[raw+189]==120&&v.raw[raw+223]==2&&v.raw[raw+223+190]==0,"opponent terminal observation not initial");
                check(v.masks[i*66+16]&&v.masks[i*66+33+16]&&v.learner_masks[i*33+16],"actual terminal initial masks missing");
                check(a.reward[0]==0&&a.reward[1]==0&&a.episode_return==0&&a.episode_hits==0,"new episode retained terminal accumulators");
            }
            auto check_cumulative=[&](){
                for(int i=0;i<3;i++){
                    const int own=i==0?7:i==1?3:5,other=i==0?3:i==1?7:5;
                    const float expected=mode?(i==0?1.f-4.f/9.f:i==1?-1.f+4.f/9.f:0.f):0.f;
                    const auto& r=v.rounds[i];
                    const auto& log=reinterpret_cast<const RekNative5Log*>(reinterpret_cast<const char*>(v.out.logs)+i*v.out.log_stride_bytes)[0];
                    check(r.completed_rounds==3&&r.wins[0]==uint64_t(1+(i==0))&&r.wins[1]==uint64_t(1+(i==1))&&r.ties==uint64_t(i==2),"completed round outcomes wrong or repeated");
                    check(r.completed_points[0]==13+own&&r.completed_points[1]==17+other,"completed score totals wrong or repeated");
                    check(log.score==13+own&&near(log.episode_return,2.f+expected)&&log.episode_length==14000&&log.n==3,"terminal return or length logs wrong or repeated");
                    check(log.hits==6&&log.falls==1&&log.actions_invalid==3&&log.finite_failures==0,"terminal event logs wrong or repeated");
                    check(log.wins==1+(i==0)&&log.losses==1+(i==1)&&log.draws==(i==2),"terminal outcome logs wrong or repeated");
                }
            };
            check_cumulative();
            fast_step<<<1,96,0,stream>>>(v,true);rek5::cuda_check(cudaStreamSynchronize(stream));
            for(int i=0;i<3;i++){
                check(v.rounds[i].round_number==8&&v.state[i].tick==1&&!v.rounds[i].terminal&&!v.rounds[i].failure_bits,"next fast_step reset twice or failed");
                check(v.out.rewards[i]==0&&v.rewards[i*2]==0&&v.rewards[i*2+1]==0,"next fast_step repeated terminal reward");
                check(v.out.terminals[i]==0&&v.terminals[i*2]==0&&v.terminals[i*2+1]==0,"next fast_step repeated terminal mask");
                check(v.rounds[i].points[0]==0&&v.rounds[i].points[1]==0&&v.out.observations[i*223+190]==0&&v.out.observations[i*223+191]==0,"next fast_step retained terminal score");
            }
            check_cumulative();
        }
        rek5::cuda_check(cudaGraphExecDestroy(executable));rek5::cuda_check(cudaGraphDestroy(graph));
    }
    printf("{\"test\":\"compact_training_autoreset\",\"checks\":%u,\"legacy_checks\":%u,\"graph_replays\":12,\"reward_modes\":2,\"terminal_cases\":3,\"actual_runtime_terminal_steps\":24,\"status\":\"passed\",\"server_parity\":false}\n",checks,legacy_checks);
    rek5::cuda_check(cudaStreamDestroy(stream));for(void* p:mem)rek5::cuda_check(cudaFree(p));
    return 0;
}catch(const std::exception& e){fprintf(stderr,"autoreset test: %s\n",e.what());return 1;}}
