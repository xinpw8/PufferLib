// Exercise the existing sixteen body-velocity cases through a full warp.
// Each cooperative call is compared byte-for-byte with the serial scorer.
#define main serial_cvel_test_main
#include "test_contact_velocity_runtime.cu"
#undef main
struct WarpCvelChecks {unsigned checks,failures,first_failure;};
__device__ void cvel_check(WarpCvelChecks* out,bool ok,unsigned label){
    out->checks++;if(!ok){if(!out->failures)out->first_failure=label;out->failures++;}
}
__device__ int warp_cvel_contacts(View& v,Arena& a,Arena& reference,int lane,
        WarpContactScratch& scratch,WarpCvelChecks* out,int move=1,float time=10,bool active=true){
    int expected=0;
    if(lane==0){
        reference=a;expected=cvel_contacts(v,reference,move,time,active);
        a.fighter[0].strike_active=active;a.fighter[0].move_instance=move;a.elapsed=time;
    }
    __syncwarp();
    int hits[2]={},points[2]={};
    geom_pair_contacts_warp(v,a,0,hits,points,lane,scratch);
    if(lane==0){
        cvel_check(out,points[0]==expected,1);
        const auto* x=reinterpret_cast<const unsigned char*>(&a);
        const auto* y=reinterpret_cast<const unsigned char*>(&reference);
        for(unsigned i=0;i<sizeof(Arena);i++)cvel_check(out,x[i]==y[i],100+i);
    }
    __syncwarp();
    return __shfl_sync(0xffffffffu,points[0],0);
}
__global__ void cvel_exercise_warp(CvelResults* out,Parameters* params,FastFrame* frame_data,
        FastBodyVelocityFrame* velocity_data,Arena* arena,Arena* reference,WarpCvelChecks* checks){
    if(blockIdx.x)return;const int lane=threadIdx.x&31;
    __shared__ WarpContactScratch scratch;
    auto& p=*params;auto& frames=*reinterpret_cast<FastFrame(*)[4]>(frame_data);
    auto& velocities=*reinterpret_cast<FastBodyVelocityFrame(*)[4]>(velocity_data);auto& a=*arena;
    View v{};v.p=&p;v.frames=frames;v.body_velocity_frames=velocities;
    if(lane==0)cvel_fixture(p,frames,velocities,a);
    __syncwarp();
    const float high=p.recovered_hit_config.speed_threshold_mps+2;
    // New slow pelvis contact cannot borrow speed from an already-latched fast torso.
    if(lane==0){for(auto& frame:frames)frame.target_shapes[1].center[1]=0;}
    if(lane==0){velocities[3].linear[1][rek_contact_velocity::target_slot(1)][0]=high;}
    if(lane==0){rek_contact_entry::update(a.fighter[0].contact_pairs,8*9+1,true);}
    {const int value=warp_cvel_contacts(v,a,*reference,lane,scratch,checks);if(lane==0)out->values[0]=value;}
    if(lane==0){out->values[1]=!a.recovered_hits.cooldown_seen[0][2];}
    if(lane==0){out->values[2]=!a.recovered_hits.scored_move_seen[0];}
    // Distinct same-body torso geometry enters later; rejected speed must not
    // have consumed body cooldown or this invocation's apex.
    if(lane==0){for(auto& frame:frames)frame.target_shapes[2].center[1]=0;}
    {const int value=warp_cvel_contacts(v,a,*reference,lane,scratch,checks);if(lane==0)out->values[3]=value;}
    if(lane==0){out->values[4]=a.fighter[1].last_hit_speed==high;}
    {const int value=warp_cvel_contacts(v,a,*reference,lane,scratch,checks,2,11);if(lane==0)out->values[5]=value;}
    // A qualifying pair later in the same call also survives the slow rejection.
    if(lane==0){cvel_fixture(p,frames,velocities,a);}
    if(lane==0){for(auto& frame:frames)frame.target_shapes[1].center[1]=0;}
    if(lane==0){velocities[3].linear[1][rek_contact_velocity::target_slot(1)][0]=high;}
    {const int value=warp_cvel_contacts(v,a,*reference,lane,scratch,checks);if(lane==0)out->values[6]=value;}
    if(lane==0){out->values[7]=a.recovered_hits.cooldown_seen[0][2]&&a.recovered_hits.scored_move_seen[0];}
    // A held frame suppresses stored incoming clip rate, retaining base motion.
    if(lane==0){cvel_fixture(p,frames,velocities,a);a.fighter[0].old_phase=1;}
    if(lane==0){velocities[1].linear[0][2][0]=high;}
    {const int value=warp_cvel_contacts(v,a,*reference,lane,scratch,checks);if(lane==0)out->values[8]=value;}
    if(lane==0){cvel_fixture(p,frames,velocities,a);a.fighter[0].old_phase=1;a.fighter[0].vx=high;}
    {const int value=warp_cvel_contacts(v,a,*reference,lane,scratch,checks);if(lane==0)out->values[9]=value;}
    if(lane==0){cvel_fixture(p,frames,velocities,a);velocities[1].linear[0][2][0]=high;}
    {const int value=warp_cvel_contacts(v,a,*reference,lane,scratch,checks);if(lane==0)out->values[10]=value;}
    if(lane==0){cvel_fixture(p,frames,velocities,a);velocities[1].root_com[0][0]=1;a.fighter[0].omega=high;}
    {const int value=warp_cvel_contacts(v,a,*reference,lane,scratch,checks);if(lane==0)out->values[11]=value;}
    // Invalid apex remains unscored but must retain endpoint contact history.
    if(lane==0){cvel_fixture(p,frames,velocities,a);a.fighter[0].phase=50;a.fighter[0].vx=high;}
    {const int value=warp_cvel_contacts(v,a,*reference,lane,scratch,checks);if(lane==0)out->values[12]=value;}
    if(lane==0){out->values[13]=!rek_contact_entry::update(a.fighter[0].contact_pairs,8*9,true);}
    if(lane==0){cvel_fixture(p,frames,velocities,a);seed_contact_pairs(v,a);a.fighter[0].vx=high;}
    {const int value=warp_cvel_contacts(v,a,*reference,lane,scratch,checks);if(lane==0)out->values[14]=value;}
    if(lane==0){cvel_fixture(p,frames,velocities,a);p.contact_velocity=rek_contact_velocity::Mode::LegacySphereProxy;v.body_velocity_frames=nullptr;}
    {const int value=warp_cvel_contacts(v,a,*reference,lane,scratch,checks);if(lane==0)out->values[15]=value;} // Same legacy 5 m/s sphere-center proxy.

    if(lane==0){
        cvel_fixture(p,frames,velocities,a);v.body_velocity_frames=velocities;
        for(auto& frame:frames)frame.target_shapes[1].center[1]=0;
        velocities[3].linear[1][rek_contact_velocity::target_slot(0)][0]=high;
        velocities[3].linear[1][rek_contact_velocity::target_slot(1)][0]=high+10;
    }
    // Previous legacy case cleared its thread-local View pointer on lane0.
    v.body_velocity_frames=velocities;
    {const int value=warp_cvel_contacts(v,a,*reference,lane,scratch,checks);
        if(lane==0){cvel_check(checks,value==1,2000);cvel_check(checks,a.fighter[1].last_hit_speed==high,2001);}}
    if(lane==0){
        cvel_fixture(p,frames,velocities,a);
        p.impact_events[0].limb=REK_G1_AIM_LIMB_LEFT_LOWER_BODY;
        for(auto& frame:frames){
            frame.strike_xyz[0][0]=frame.strike_xyz[4][0]=0;
            frame.strike_shapes[0].center[0]=frame.strike_shapes[10].center[0]=0;
        }
        velocities[1].linear[0][0][0]=high;
        velocities[1].linear[0][4][0]=high+10;
    }
    {const int value=warp_cvel_contacts(v,a,*reference,lane,scratch,checks);
        if(lane==0){cvel_check(checks,value==2,2002);cvel_check(checks,a.fighter[1].last_hit_speed==high&&!a.recovered_hits.cooldown_seen[0][4],2003);}}
}
int main(int argc,char** argv){
    if(serial_cvel_test_main(argc,argv))return 2;
    try{
        rek5::DeviceStorage storage;
        auto* out=storage.alloc<CvelResults>(1);auto* params=storage.alloc<Parameters>(1);
        auto* frames=storage.alloc<FastFrame>(4);auto* velocity=storage.alloc<FastBodyVelocityFrame>(4);
        auto* arena=storage.alloc<Arena>(1);auto* reference=storage.alloc<Arena>(1);auto* checks=storage.alloc<WarpCvelChecks>(1);
        cvel_exercise_warp<<<1,32>>>(out,params,frames,velocity,arena,reference,checks);
        rek5::cuda_check(cudaGetLastError());rek5::cuda_check(cudaDeviceSynchronize());
        CvelResults actual{};WarpCvelChecks state{};
        rek5::cuda_check(cudaMemcpy(&actual,out,sizeof(actual),cudaMemcpyDeviceToHost));
        rek5::cuda_check(cudaMemcpy(&state,checks,sizeof(state),cudaMemcpyDeviceToHost));
        const int expected[16]={0,1,1,1,1,0,1,1,0,1,1,1,0,1,0,1};
        for(int i=0;i<16;i++)if(actual.values[i]!=expected[i])throw std::runtime_error("cooperative scoring fixture "+std::to_string(i)+" expected "+std::to_string(expected[i])+" got "+std::to_string(actual.values[i]));
        std::printf("{\"test\":\"contact_velocity_full_warp\",\"original_checks\":16,\"additional_order_checks\":4,\"reference_checks\":%u,\"failures\":%u,\"first_failure\":%u,\"passed\":%s}\n",state.checks,state.failures,state.first_failure,state.failures?"false":"true");
        return state.failures?2:0;
    }catch(const std::exception& error){std::fprintf(stderr,"%s\n",error.what());return 2;}
}
