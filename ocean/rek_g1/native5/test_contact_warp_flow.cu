// Private prototype fixture: no game assets or learner needed.
#define main contact_equivalence_main
#include "test_contact_warp_runtime.cu"
#undef main

struct FlowState {
    Arena arena[2];
    RekNative5RoundResult rounds[2];
    RekNative5Log logs[2];
    float actions[2][2],policy_actions[2],external[2];
    uint8_t overrides[2];
};
template<class T> __device__ void equal_bytes(IdleResults* out,const T& a,const T& b,unsigned label){
    const unsigned char* x=reinterpret_cast<const unsigned char*>(&a);
    const unsigned char* y=reinterpret_cast<const unsigned char*>(&b);
    for(unsigned i=0;i<sizeof(T);i++)verify(out,x[i]==y[i],label+i);
}
__global__ void exercise_flow(Parameters* params,FastFrame* frames,FlowState* flow,IdleResults* out){
    if(blockIdx.x)return;const int lane=threadIdx.x&31;
    __shared__ WarpContactScratch scratch;
    auto& p=*params;
    for(unsigned scenario=0;scenario<10;scenario++){
        if(lane==0){
            p={};*flow={};
            p.contact_entry=rek_contact_entry::Mode::GeomPair;
            p.primitive_contacts=1;p.recovered_scoring=2;p.contact_substeps=8;
            p.recovered_hit_config=rek5_recovered::rek_g1_current_build_hit_detector_config();
            p.round_seconds=scenario==3?DT:120;p.body_radius=.22f;p.half_extent[0]=p.half_extent[1]=3;
            p.spawn[0][0]=-.5f;p.spawn[1][0]=.5f;p.rendered_observation=1;
            p.yaw_command=rek_keyboard_yaw::Mode::KeyboardReset;p.yaw_speed=1.8f;p.yaw_ramp=.5f;p.brake_rate=2;
            p.settle_speed=.03f;p.policy_action_stride=1;
            for(int route=0;route<24;route++){p.routes[route].count=2;p.routes[route].end_frame=100;p.routes[route].playback_speed=1;}
            for(int frame=0;frame<2;frame++){
                frames[frame]={};frames[frame].root_wxyz[0]=1;
                for(int limb=0;limb<6;limb++)frames[frame].strike_radius[limb]=.5f;
                for(int i=0;i<12;i++)make_shape(frames[frame].strike_shapes[i],i%3,0,0,0);
                for(int zone=0;zone<9;zone++){
                    make_shape(frames[frame].target_shapes[zone],zone%3,0,0,0);
                    frames[frame].target_shape_radius[zone]=.5f;
                }
            }
            const int limbs[12]={0,0,0,0,1,1,1,1,2,3,4,5};
            for(int i=0;i<12;i++)p.strike_limb[i]=limbs[i];
            flow->overrides[0]=flow->overrides[1]=1;
            if(scenario==4)flow->external[0]=__int_as_float(0x7fc00000);
            if(scenario==5){p.recovered_bot=1;flow->overrides[1]=0;}
            if(scenario==8)flow->external[0]=6;
            if(scenario==9){flow->external[0]=16;p.action_to_route[16]=7;p.routes[7].move=0;p.durations[0]=2;}
            for(int arm=0;arm<2;arm++){
                auto& a=flow->arena[arm];auto& r=flow->rounds[arm];
                round_reset(p,a,r,true,0);
                if(scenario==1)a.reset_wait=1;
                if(scenario==2)r.terminal=1;
                if(scenario==5)a.fighter[1].down=1;
                if(scenario==6)a.failures=32;
                if(scenario==7)a.reset_wait=2;
            }
        }
        __syncwarp();
        View reference{},cooperative{};
        reference.arenas=cooperative.arenas=1;reference.p=cooperative.p=params;reference.frames=cooperative.frames=frames;
        reference.state=&flow->arena[0];cooperative.state=&flow->arena[1];
        reference.rounds=&flow->rounds[0];cooperative.rounds=&flow->rounds[1];
        reference.actions=flow->actions[0];cooperative.actions=flow->actions[1];
        reference.external=cooperative.external=flow->external;reference.override_rows=cooperative.override_rows=flow->overrides;
        reference.out.actions=&flow->policy_actions[0];cooperative.out.actions=&flow->policy_actions[1];
        reference.out.logs=&flow->logs[0];cooperative.out.logs=&flow->logs[1];
        reference.out.log_stride_bytes=cooperative.out.log_stride_bytes=sizeof(RekNative5Log);
        if(lane==0)advance_arena(reference,0);
        __syncwarp();
        advance_arena_warp(cooperative,0,lane,scratch);
        __syncwarp();
        if(lane==0){
            equal_bytes(out,flow->arena[0],flow->arena[1],100000+scenario*10000);
            equal_bytes(out,flow->rounds[0],flow->rounds[1],102000+scenario*10000);
            equal_bytes(out,flow->logs[0],flow->logs[1],104000+scenario*10000);
            equal_bytes(out,flow->actions[0],flow->actions[1],106000+scenario*10000);
            if(scenario==4)verify(out,flow->rounds[0].failure_bits==16,200000);
            if(scenario==5)verify(out,flow->rounds[0].failure_bits==1024,200001);
            if(scenario==6)verify(out,flow->rounds[0].failure_bits==32,200002);
            if(scenario==3)verify(out,flow->rounds[0].terminal==1,200003);
        }
        __syncwarp();
    }
}
int main(){
    if(contact_equivalence_main())return 2;
    try{
        Parameters* params=nullptr;FastFrame* frames=nullptr;FlowState* flow=nullptr;IdleResults* out=nullptr;
        rek5::cuda_check(cudaMallocManaged(&params,sizeof(Parameters)));rek5::cuda_check(cudaMemset(params,0,sizeof(Parameters)));
        rek5::cuda_check(cudaMallocManaged(&frames,2*sizeof(FastFrame)));rek5::cuda_check(cudaMemset(frames,0,2*sizeof(FastFrame)));
        rek5::cuda_check(cudaMallocManaged(&flow,sizeof(FlowState)));rek5::cuda_check(cudaMemset(flow,0,sizeof(FlowState)));
        rek5::cuda_check(cudaMallocManaged(&out,sizeof(IdleResults)));rek5::cuda_check(cudaMemset(out,0,sizeof(IdleResults)));
        exercise_flow<<<1,32>>>(params,frames,flow,out);rek5::cuda_check(cudaDeviceSynchronize());
        std::printf("{\"test\":\"contact_warp_uniform_flow\",\"scenarios\":10,\"checks\":%u,\"failures\":%u,\"first_failure\":%u,\"passed\":%s}\n",out->checks,out->failures,out->first_failure,out->failures?"false":"true");
        const bool failed=out->failures;
        rek5::cuda_check(cudaFree(params));rek5::cuda_check(cudaFree(frames));rek5::cuda_check(cudaFree(flow));rek5::cuda_check(cudaFree(out));
        return failed?2:0;
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 2;}
}
