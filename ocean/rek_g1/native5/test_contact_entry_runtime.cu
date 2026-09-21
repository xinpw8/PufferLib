// Synthetic geometry exercises actual production contacts and unchanged score
// gates. The separate ABI snapshot fixture verifies default real-asset bytes.
#include "fast_runtime.cu"
struct Results {int values[16];};
__device__ int call_contacts(View& v,Arena& a,bool intent,int move,float time){
    a.fighter[0].strike_active=intent;a.fighter[0].move_instance=move;a.elapsed=time;
    int hits[2]={},points[2]={};strike_contacts(v,a,0,hits,points);return points[0];
}
__global__ void exercise(Results* out){
    if(threadIdx.x||blockIdx.x)return;
    Parameters p{};p.contact_entry=rek_contact_entry::Mode::GeomPair;p.primitive_contacts=1;p.recovered_scoring=2;p.contact_substeps=8;
    p.recovered_hit_config=rek5_recovered::rek_g1_current_build_hit_detector_config();
    p.impact_events[0]={.02f,.1f,.1f,1.f,REK_G1_AIM_LIMB_LEFT_UPPER_BODY};p.impact_counts[23]=1;
    p.routes[23].count=2;p.routes[23].end_frame=100;p.routes[23].playback_speed=1;
    p.routes[0].offset=2;p.routes[0].count=2;
    const int limbs[12]={0,0,0,0,1,1,1,1,2,3,4,5};for(int i=0;i<12;i++)p.strike_limb[i]=limbs[i];
    FastFrame frames[4]{};
    for(auto& frame:frames){
        for(int i=0;i<6;i++){frame.strike_xyz[i][0]=i==2?0:100;frame.strike_radius[i]=.3f;}
        for(int i=0;i<12;i++){auto& s=frame.strike_shapes[i];s.kind=rek5_primitive::Sphere;s.size[0]=.2f;s.center[0]=i==8?0:100;s.axes[0]=s.axes[4]=s.axes[8]=1;}
        for(int j=0;j<9;j++){auto& s=frame.target_shapes[j];s.kind=rek5_primitive::Sphere;s.size[0]=.2f;s.center[1]=j?100:0;s.axes[0]=s.axes[4]=s.axes[8]=1;frame.target_shape_radius[j]=.3f;}
    }
    Arena a{};auto& f=a.fighter[0];f.route=f.old_route=23;f.phase=f.old_phase=1;
    View v{};v.p=&p;v.frames=frames;f.old_x=-.1f;
    out->values[0]=call_contacts(v,a,true,1,10); // entry with5m/s proxy,1point
    // Continuous overlap after a later move, beyond body cooldown. A move
    // start may clear the legacy limb latch, but not geom-pair history.
    f.contact_latched=0;out->values[1]=call_contacts(v,a,true,2,11);
    f.x=f.old_x=2;out->values[2]=call_contacts(v,a,false,2,12); // idle exit
    f.old_x=-.1f;f.x=0;out->values[3]=call_contacts(v,a,true,3,13); // reentry
    a={};f.route=f.old_route=23;f.phase=f.old_phase=1;f.old_x=-.1f;
    out->values[4]=call_contacts(v,a,false,1,10); // idle contact updates history
    out->values[5]=call_contacts(v,a,true,2,11); // attack cannot fabricate entry
    // A distinct target geom, including the same native torso body, may enter.
    for(auto& frame:frames)frame.target_shapes[1].center[1]=0;
    out->values[6]=call_contacts(v,a,true,3,12);
    out->values[7]=call_contacts(v,a,true,4,12.1f); // persistent+cooldown
    // Round reset clears pair history and seeds existing overlaps without enter.
    RekNative5RoundResult round{};round_reset(p,a,round,true,0);seed_contact_pairs(v,a);
    f.route=f.old_route=23;f.phase=f.old_phase=1;f.old_x=-.1f;
    out->values[8]=call_contacts(v,a,true,1,10);
    out->values[9]=a.contact_pairs_initialized;
    // A sampled enter then exit has no endpoint latch; another crossing enters.
    a={};f.route=f.old_route=23;f.phase=f.old_phase=1;f.old_x=-1;f.x=1;
    out->values[10]=call_contacts(v,a,true,1,10);
    out->values[11]=f.contact_pairs.words[0]!=0||f.contact_pairs.words[1]!=0;
    f.old_x=1;f.x=-1;out->values[12]=call_contacts(v,a,true,2,11);
    // Actual new-attack integration retains all108 pair bits.
    p.routes[7].count=2;p.routes[7].move=0;p.action_to_route[16]=7;p.durations[0]=2;p.settle_speed=.01f;
    a={};f.held=1;f.contact_pairs.words[0]=~std::uint64_t(0);f.contact_pairs.words[1]=(std::uint64_t(1)<<44)-1;
    advance_fighter(p,f,16,a);out->values[13]=f.contact_pairs.words[0]==~std::uint64_t(0)&&f.contact_pairs.words[1]==(std::uint64_t(1)<<44)-1;
    pose_reset(p,a);out->values[14]=f.contact_pairs.words[0]==0&&f.contact_pairs.words[1]==0&&!a.contact_pairs_initialized;
    Parameters legacy=p;legacy.contact_entry=rek_contact_entry::Mode::LegacyLimbUnion;v.p=&legacy;
    a={};f.route=f.old_route=23;f.phase=f.old_phase=1;f.old_x=-.1f;
    call_contacts(v,a,true,1,10);f.contact_latched=0;out->values[15]=call_contacts(v,a,true,2,11);
}
int main(){try{
    Results* result=nullptr;rek5::cuda_check(cudaMallocManaged(&result,sizeof(Results)));rek5::cuda_check(cudaMemset(result,0,sizeof(Results)));
    exercise<<<1,1>>>(result);rek5::cuda_check(cudaDeviceSynchronize());
    const int expected[16]={1,0,0,1,0,0,1,0,0,1,1,0,1,1,1,1};
    for(int i=0;i<16;i++)if(result->values[i]!=expected[i])throw std::runtime_error("fixture "+std::to_string(i)+" expected "+std::to_string(expected[i])+" got "+std::to_string(result->values[i]));
    rek5::cuda_check(cudaFree(result));std::printf("{\"test\":\"contact_entry_runtime\",\"checks\":16,\"passed\":true,\"authentic_parity\":false}\n");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 2;}}
