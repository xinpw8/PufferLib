// Synthetic contacts exercise the actual production scoring path. They are
// not a physical trajectory or an authentic hit claim.
#include "fast_runtime.cu"
struct CvelResults {int values[16];};
__device__ void cvel_fixture(Parameters& p,FastFrame* frames,FastBodyVelocityFrame* velocity,Arena& a){
    p={};a={};p.contact_entry=rek_contact_entry::Mode::GeomPair;p.contact_velocity=rek_contact_velocity::Mode::BodyCvel;
    p.primitive_contacts=1;p.recovered_scoring=2;p.contact_substeps=8;
    p.recovered_hit_config=rek5_recovered::rek_g1_current_build_hit_detector_config();
    p.impact_events[0]={.02f,.1f,.1f,1.f,REK_G1_AIM_LIMB_LEFT_UPPER_BODY};p.impact_counts[23]=1;
    p.routes[23].count=2;p.routes[23].end_frame=100;p.routes[23].playback_speed=1;p.routes[0].offset=2;p.routes[0].count=2;
    const int limbs[12]={0,0,0,0,1,1,1,1,2,3,4,5};for(int i=0;i<12;i++)p.strike_limb[i]=limbs[i];
    for(int index=0;index<4;index++){
        auto& frame=frames[index];frame={};velocity[index]={};
        for(int i=0;i<6;i++){frame.strike_xyz[i][0]=i==2?0:100;frame.strike_radius[i]=.3f;}
        for(int i=0;i<12;i++){auto& s=frame.strike_shapes[i];s.kind=rek5_primitive::Sphere;s.size[0]=.2f;s.center[0]=i==8?0:100;s.axes[0]=s.axes[4]=s.axes[8]=1;}
        for(int j=0;j<9;j++){auto& s=frame.target_shapes[j];s.kind=rek5_primitive::Sphere;s.size[0]=.2f;s.center[1]=j?100:0;s.axes[0]=s.axes[4]=s.axes[8]=1;frame.target_shape_radius[j]=.3f;}
    }
    auto& f=a.fighter[0];f.route=f.old_route=23;f.phase=1;f.old_phase=0;f.old_x=-.1f;
    a.fighter[1].route=a.fighter[1].old_route=0;a.fighter[1].phase=1;a.fighter[1].old_phase=0;
}
__device__ int cvel_contacts(View& v,Arena& a,int move=1,float time=10,bool active=true){
    a.fighter[0].strike_active=active;a.fighter[0].move_instance=move;a.elapsed=time;
    int hits[2]={},points[2]={};strike_contacts(v,a,0,hits,points);return points[0];
}
__global__ void cvel_exercise(CvelResults* out){
    if(threadIdx.x||blockIdx.x)return;
    Parameters p{};FastFrame frames[4]{};FastBodyVelocityFrame velocities[4]{};Arena a{};
    View v{};v.p=&p;v.frames=frames;v.body_velocity_frames=velocities;
    cvel_fixture(p,frames,velocities,a);const float high=p.recovered_hit_config.speed_threshold_mps+2;
    // New slow pelvis contact cannot borrow speed from an already-latched fast torso.
    for(auto& frame:frames)frame.target_shapes[1].center[1]=0;
    velocities[3].linear[1][rek_contact_velocity::target_slot(1)][0]=high;
    rek_contact_entry::update(a.fighter[0].contact_pairs,8*9+1,true);
    out->values[0]=cvel_contacts(v,a);
    out->values[1]=!a.recovered_hits.cooldown_seen[0][2];
    out->values[2]=!a.recovered_hits.scored_move_seen[0];
    // Distinct same-body torso geometry enters later; rejected speed must not
    // have consumed body cooldown or this invocation's apex.
    for(auto& frame:frames)frame.target_shapes[2].center[1]=0;
    out->values[3]=cvel_contacts(v,a);
    out->values[4]=a.fighter[1].last_hit_speed==high;
    out->values[5]=cvel_contacts(v,a,2,11);
    // A qualifying pair later in the same call also survives the slow rejection.
    cvel_fixture(p,frames,velocities,a);
    for(auto& frame:frames)frame.target_shapes[1].center[1]=0;
    velocities[3].linear[1][rek_contact_velocity::target_slot(1)][0]=high;
    out->values[6]=cvel_contacts(v,a);
    out->values[7]=a.recovered_hits.cooldown_seen[0][2]&&a.recovered_hits.scored_move_seen[0];
    // A held frame suppresses stored incoming clip rate, retaining base motion.
    cvel_fixture(p,frames,velocities,a);a.fighter[0].old_phase=1;
    velocities[1].linear[0][2][0]=high;
    out->values[8]=cvel_contacts(v,a);
    cvel_fixture(p,frames,velocities,a);a.fighter[0].old_phase=1;a.fighter[0].vx=high;
    out->values[9]=cvel_contacts(v,a);
    cvel_fixture(p,frames,velocities,a);velocities[1].linear[0][2][0]=high;
    out->values[10]=cvel_contacts(v,a);
    cvel_fixture(p,frames,velocities,a);velocities[1].root_com[0][0]=1;a.fighter[0].omega=high;
    out->values[11]=cvel_contacts(v,a);
    // Invalid apex remains unscored but must retain endpoint contact history.
    cvel_fixture(p,frames,velocities,a);a.fighter[0].phase=50;a.fighter[0].vx=high;
    out->values[12]=cvel_contacts(v,a);
    out->values[13]=!rek_contact_entry::update(a.fighter[0].contact_pairs,8*9,true);
    cvel_fixture(p,frames,velocities,a);seed_contact_pairs(v,a);a.fighter[0].vx=high;
    out->values[14]=cvel_contacts(v,a);
    cvel_fixture(p,frames,velocities,a);p.contact_velocity=rek_contact_velocity::Mode::LegacySphereProxy;v.body_velocity_frames=nullptr;
    out->values[15]=cvel_contacts(v,a); // Same legacy 5 m/s sphere-center proxy.
}
int main(int argc,char** argv){try{
    if(argc!=4)throw std::runtime_error("Usage: test-contact-velocity-runtime MODEL_XML ASSETS_DIRECTORY FEATURES_DIRECTORY");
    rek5::DeviceStorage storage;auto* out=storage.alloc<CvelResults>(1);CvelResults actual{};
    cvel_exercise<<<1,1>>>(out);rek5::cuda_check(cudaGetLastError());rek5::cuda_check(cudaMemcpy(&actual,out,sizeof(actual),cudaMemcpyDeviceToHost));
    const int expected[16]={0,1,1,1,1,0,1,1,0,1,1,1,0,1,0,1};
    for(int i=0;i<16;i++)if(actual.values[i]!=expected[i])throw std::runtime_error("scoring fixture "+std::to_string(i)+" expected "+std::to_string(expected[i])+" got "+std::to_string(actual.values[i]));
    RekNative5Config cfg{};cfg.abi_version=REK_NATIVE5_RUNTIME_ABI;cfg.model_path=argv[1];cfg.assets_path=argv[2];cfg.motion_features_path=argv[3];cfg.arenas=1;cfg.locomotion_segment_ticks=1;
    const std::uint32_t durations[]={35,27,31,45,32,45,157,145,158,139,134,138,73,75,68,71,103};std::copy(durations,durations+17,cfg.move_duration_ticks);
    RekNative5Buffers buffers{};buffers.observations=storage.alloc<float>(223);buffers.actions=storage.alloc<float>(1);buffers.rewards=storage.alloc<float>(1);buffers.terminals=storage.alloc<float>(1);buffers.logs=storage.alloc<RekNative5Log>(1);buffers.log_stride_bytes=sizeof(RekNative5Log);
    setenv("REK_PHYSICS_BACKEND","semantic_cuda",1);setenv("REK_FAST_SCORING","recovered_hit_rules_v2",1);setenv("REK_FAST_GEOMETRY","primitive_samples_v1",1);setenv("REK_FAST_CONTACT_ENTRY","geom_pair_v1",1);
    unsetenv("REK_FAST_CONTACT_VELOCITY");
    auto* legacy=rek_native5_create(&cfg,&buffers,nullptr);if(!legacy)throw std::runtime_error(rek_native5_error());
    if(legacy->view.body_velocity_frames)throw std::runtime_error("default uploaded optional velocity data");
    const auto legacy_allocations=legacy->storage.owned.size();rek_native5_close(legacy);
    setenv("REK_FAST_CONTACT_VELOCITY","body_cvel_v1",1);
    auto* body=rek_native5_create(&cfg,&buffers,nullptr);if(!body)throw std::runtime_error(rek_native5_error());
    if(!body->view.body_velocity_frames||body->storage.owned.size()!=legacy_allocations+1)throw std::runtime_error("opt-in allocation mismatch");
    rek_native5_close(body);
    setenv("REK_FAST_CONTACT_ENTRY","legacy_limb_union_v1",1);
    auto* invalid=rek_native5_create(&cfg,&buffers,nullptr);
    if(invalid){rek_native5_close(invalid);throw std::runtime_error("incompatible body cvel mode accepted");}
    if(std::string(rek_native5_error()).find("body_cvel_v1 requires")==std::string::npos)throw std::runtime_error("unexpected mode rejection");
    std::puts("{\"test\":\"contact_velocity_production\",\"scoring_checks\":16,\"default_optional_allocation\":false,\"opt_in_extra_allocations\":1,\"incompatible_mode_rejected\":true,\"passed\":true,\"authentic_physical_parity\":false}");return 0;
}catch(const std::exception& error){std::fprintf(stderr,"%s\n",error.what());return 1;}}
