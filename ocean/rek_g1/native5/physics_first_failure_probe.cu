// Standalone diagnostic executable. Includes the original physics adapter
// under a renamed entry point and intercepts calls only in this executable.
// No PPO, resets-on-failure, force changes, or production source modifications.
#define physics_step physics_step_baseline
#include "physics.cu"
#undef physics_step
#include "runtime_api.h"
#include <chrono>

namespace diagnostic {
using namespace rek5_native;
struct Stage { int bad_body=-1,bad_field=-1; float linear=0,angular=0,impulse=0; };
static B3World* previous=nullptr;
static Stage* stage_device=nullptr;
static int call=0;
static void ck(cudaError_t result){if(result!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(result));}
__device__ void scalar(Stage& s,float value,int body,int field) {
    if(!isfinite(value)&&s.bad_body<0){s.bad_body=body;s.bad_field=field;}
}
__device__ void vec(Stage& s,B3Vec3 v,int body,int field) {
    scalar(s,v.x,body,field);scalar(s,v.y,body,field);scalar(s,v.z,body,field);
}
__device__ Stage inspect(const B3World* w) {
    Stage s;
    for(int b=0;b<w->body_count;b++) {
        const B3Body& v=w->bodies[b];
        vec(s,v.position,b,0);vec(s,v.center,b,1);vec(s,v.lin_vel,b,2);vec(s,v.ang_vel,b,3);
        vec(s,v.rotation.v,b,4);scalar(s,v.rotation.s,b,4);vec(s,v.delta_pos,b,5);
        vec(s,v.delta_rot.v,b,6);scalar(s,v.delta_rot.s,b,6);vec(s,v.force,b,7);vec(s,v.torque,b,8);
        s.linear=fmaxf(s.linear,fmaxf(fabsf(v.lin_vel.x),fmaxf(fabsf(v.lin_vel.y),fabsf(v.lin_vel.z))));
        s.angular=fmaxf(s.angular,fmaxf(fabsf(v.ang_vel.x),fmaxf(fabsf(v.ang_vel.y),fabsf(v.ang_vel.z))));
    }
    for(int j=0;j<w->joint_count;j++) {
        const B3Joint& v=w->joints[j];
        vec(s,v.linear_impulse,j,9);scalar(s,v.perp_impulse.x,j,10);scalar(s,v.perp_impulse.y,j,10);
        scalar(s,v.lower_impulse,j,11);scalar(s,v.upper_impulse,j,11);scalar(s,v.axial_mass,j,12);
        s.impulse=fmaxf(s.impulse,fmaxf(fabsf(v.lower_impulse),fabsf(v.upper_impulse)));
    }
    for(int c=0;c<w->contact_count;c++) {
        const B3Contact& v=w->contacts[c];
        vec(s,v.normal,c,13);vec(s,v.tangent1,c,13);vec(s,v.tangent2,c,13);
        scalar(s,v.tangent_mass.cx.x,c,14);scalar(s,v.tangent_mass.cx.y,c,14);
        scalar(s,v.tangent_mass.cy.x,c,14);scalar(s,v.tangent_mass.cy.y,c,14);
        scalar(s,v.friction_impulse.x,c,15);scalar(s,v.friction_impulse.y,c,15);
        for(int k=0;k<v.point_count;k++) {
            scalar(s,v.points[k].normal_mass,c,16);scalar(s,v.points[k].normal_impulse,c,17);
            vec(s,v.points[k].r_a,c,18);vec(s,v.points[k].r_b,c,18);
        }
    }
    return s;
}
__global__ void replay_stages(B3World* w,const float* ctrl,Stage* result) {
    result[0]=inspect(w);
    rp_forces(w,ctrl);result[1]=inspect(w);
    float h,ih,id;B3Soft cs,ss;b3_soft_step_params(w,.002f,1,&h,&ih,&id,&cs,&ss);
    b3_find_contacts(w);result[2]=inspect(w);
    b3_prepare_contacts(w,cs,ss);result[3]=inspect(w);
    b3_prepare_joints(w,h);result[4]=inspect(w);
    b3_warm_start(w);result[5]=inspect(w);
    b3_warm_start_joints(w);result[6]=inspect(w);
    b3_integrate_velocities(w,h);result[7]=inspect(w);
    b3_solve_contacts(w,ih,w->contact_speed,1);result[8]=inspect(w);
    b3_solve_joints(w,h,ih,1);result[9]=inspect(w);
    b3_integrate_positions(w,h,id,w->max_linear_speed);result[10]=inspect(w);
    b3_solve_contacts(w,ih,w->contact_speed,0);result[11]=inspect(w);
    b3_solve_joints(w,h,ih,0);result[12]=inspect(w);
    b3_apply_restitution(w,w->restitution_threshold);result[13]=inspect(w);
    b3_finalize_transforms(w);result[14]=inspect(w);
}
static void report_failure(rek5::Physics* p,const float* ctrl,int arena,const std::vector<int>& stats) {
    auto* handle=static_cast<RpHandle*>(p->native_handle);
    auto pre=std::make_unique<B3World>(),post=std::make_unique<B3World>();
    ck(cudaMemcpy(pre.get(),previous+arena,sizeof(B3World),cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(post.get(),handle->worlds+arena,sizeof(B3World),cudaMemcpyDeviceToHost));
    std::array<float,58> controls{};ck(cudaMemcpy(controls.data(),ctrl+arena*58,58*sizeof(float),cudaMemcpyDeviceToHost));
    bool finite_controls=true;float max_control=0;
    for(float v:controls){finite_controls=finite_controls&&std::isfinite(v);max_control=std::max(max_control,std::abs(v));}
    std::printf("{\"event\":\"first_physics_failure\",\"physics_call\":%d,\"semantic_tick\":%d,\"substep\":%d,\"arena\":%d,\"stats\":[%d,%d,%d,%d],\"finite_controls\":%s,\"max_abs_control\":%.9g,\"contacts_before\":%d,\"contacts_after\":%d,\"wrapped_hinges_before\":[",
            call,(call-1)/10,(call-1)%10,arena,stats[arena*4],stats[arena*4+1],stats[arena*4+2],stats[arena*4+3],
            finite_controls?"true":"false",max_control,pre->contact_count,post->contact_count);
    int mismatches=0;
    for(int j=0;j<58;j++) {
        const B3Joint& joint=pre->joints[j];
        float raw=b3_joint_angle(pre.get(),j),midpoint=.5f*(joint.lower_angle+joint.upper_angle);
        float mapped=raw+2*B3_PI*nearbyintf((midpoint-raw)/(2*B3_PI));
        if(fabsf(mapped-raw)>1) {
            std::printf("%s{\"joint\":%d,\"raw\":%.9g,\"mapped\":%.9g}",mismatches?",":"",j,raw,mapped);mismatches++;
        }
    }
    std::printf("]}\n");
    std::vector<int> ranking(58);
    for(int j=0;j<58;j++)ranking[j]=j;
    std::sort(ranking.begin(),ranking.end(),[&](int a,int b){
        const B3Joint& x=pre->joints[a];const B3Joint& y=pre->joints[b];
        return std::max(fabsf(x.lower_impulse),fabsf(x.upper_impulse))>std::max(fabsf(y.lower_impulse),fabsf(y.upper_impulse));
    });
    for(int rank=0;rank<6;rank++) {
        int j=ranking[rank];const B3Joint& v=pre->joints[j];
        std::printf("{\"event\":\"pre_failure_joint\",\"joint\":%d,\"body_a\":%d,\"body_b\":%d,\"lower_impulse\":%.9g,\"upper_impulse\":%.9g,\"perp_impulse\":[%.9g,%.9g],\"linear_impulse\":[%.9g,%.9g,%.9g],\"limits\":[%.9g,%.9g],\"angle\":%.9g,\"speed\":%.9g}\n",
                j,v.body_a,v.body_b,v.lower_impulse,v.upper_impulse,v.perp_impulse.x,v.perp_impulse.y,
                v.linear_impulse.x,v.linear_impulse.y,v.linear_impulse.z,v.lower_angle,v.upper_angle,
                b3_joint_angle(pre.get(),j),b3_joint_speed(pre.get(),j));
    }
    for(const auto& entry:{std::make_pair("before.world.bin",pre.get()),std::make_pair("after.world.bin",post.get())}) {
        std::ofstream file(entry.first,std::ios::binary);file.write(reinterpret_cast<const char*>(entry.second),sizeof(B3World));
    }
    {std::ofstream file("before.controls.bin",std::ios::binary);file.write(reinterpret_cast<const char*>(controls.data()),sizeof(controls));}
    replay_stages<<<1,1,0,p->stream>>>(previous+arena,ctrl+arena*58,stage_device);
    ck(cudaStreamSynchronize(p->stream));std::array<Stage,15> stages{};
    ck(cudaMemcpy(stages.data(),stage_device,sizeof(stages),cudaMemcpyDeviceToHost));
    const char* names[]={"before","forces","find_contacts","prepare_contacts","prepare_joints","warm_contacts","warm_joints","integrate_velocity","solve_contacts","solve_joints","integrate_position","relax_contacts","relax_joints","restitution","finalize"};
    for(int i=0;i<15;i++)std::printf("{\"stage\":\"%s\",\"bad_body_or_joint\":%d,\"bad_field\":%d,\"max_linear_component\":%.9g,\"max_angular_component\":%.9g,\"max_limit_impulse\":%.9g}\n",
            names[i],stages[i].bad_body,stages[i].bad_field,stages[i].linear,stages[i].angular,stages[i].impulse);
    std::fflush(stdout);
}
}

namespace rek5 {
void physics_step(Physics* p,const float* ctrl) {
    using namespace diagnostic;
    auto* handle=static_cast<rek5_native::RpHandle*>(p->native_handle);
    if(!previous){ck(cudaMalloc(&previous,size_t(p->data.arenas)*sizeof(rek5_native::B3World)));ck(cudaMalloc(&stage_device,15*sizeof(Stage)));}
    ck(cudaMemcpyAsync(previous,handle->worlds,size_t(p->data.arenas)*sizeof(rek5_native::B3World),cudaMemcpyDeviceToDevice,p->stream));
    physics_step_baseline(p,ctrl);call++;
    const auto stats=physics_stats(p);
    for(int a=0;a<p->data.arenas;a++)if(stats[a*4+1]||stats[a*4+2]||stats[a*4]>=128) {
        report_failure(p,ctrl,a,stats);throw std::runtime_error("First physics failure captured by isolated diagnostic");
    }
}
}

int main(int argc,char** argv) {
    using namespace diagnostic;
    try {
        if(argc!=7)throw std::runtime_error("usage: physics_first_failure_probe XML EXPORT ASSETS FEATURES ENCODER DECODER");
        constexpr int arenas=512,ticks=100;
        RekNative5Config cfg{};cfg.abi_version=REK_NATIVE5_RUNTIME_ABI;cfg.model_path=argv[1];cfg.physics_export_path=argv[2];
        cfg.assets_path=argv[3];cfg.motion_features_path=argv[4];cfg.controller_encoder_path=argv[5];cfg.controller_decoder_path=argv[6];
        cfg.arenas=arenas;cfg.seed=73;cfg.locomotion_segment_ticks=1;
        const uint32_t durations[]={35,27,31,45,32,45,157,145,158,139,134,138,73,75,68,71,103};
        std::copy(durations,durations+17,cfg.move_duration_ticks);
        RekNative5Buffers out{};
        ck(cudaMalloc(&out.observations,arenas*223*sizeof(float)));ck(cudaMalloc(&out.actions,arenas*sizeof(float)));
        ck(cudaMalloc(&out.rewards,arenas*sizeof(float)));ck(cudaMalloc(&out.terminals,arenas*sizeof(float)));
        ck(cudaMalloc(&out.logs,arenas*sizeof(RekNative5Log)));out.log_stride_bytes=sizeof(RekNative5Log);
        uint8_t* mask=nullptr;ck(cudaMalloc(&mask,arenas*33));
        RekNative5Runtime* runtime=rek_native5_create(&cfg,&out,nullptr);
        if(!runtime)throw std::runtime_error(rek_native5_error());
        if(rek_native5_bind_action_mask(runtime,mask,nullptr))throw std::runtime_error(rek_native5_error());
        const auto start=std::chrono::steady_clock::now();
        std::vector<uint8_t> host_mask(arenas*33);std::vector<float> actions(arenas);
        int completed=0;
        for(int tick=0;tick<ticks;tick++) {
            ck(cudaDeviceSynchronize());ck(cudaMemcpy(host_mask.data(),mask,host_mask.size(),cudaMemcpyDeviceToHost));
            for(int a=0;a<arenas;a++) {
                int selected=0;
                for(int offset=0;offset<33;offset++){int candidate=(a+tick/8+offset)%33;if(host_mask[a*33+candidate]){selected=candidate;break;}}
                actions[a]=float(selected);
            }
            ck(cudaMemcpy(out.actions,actions.data(),actions.size()*sizeof(float),cudaMemcpyHostToDevice));
            if(rek_native5_step(runtime,nullptr)){std::fprintf(stderr,"%s\n",rek_native5_error());break;}
            if(rek_native5_check_status(runtime,nullptr)){std::fprintf(stderr,"%s\n",rek_native5_error());break;}
            completed++;
        }
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::printf("{\"event\":\"trajectory_complete\",\"arenas\":%d,\"ticks\":%d,\"physics_calls\":%d,\"wall_seconds\":%.6f,\"ppo_updates\":0,\"production_semantics_changed\":false}\n",arenas,completed,call,seconds);
        rek_native5_close(runtime);cudaFree(previous);cudaFree(stage_device);cudaFree(mask);
        cudaFree(out.observations);cudaFree(out.actions);cudaFree(out.rewards);cudaFree(out.terminals);cudaFree(out.logs);
        return completed==ticks?0:2;
    }catch(const std::exception& error){std::fprintf(stderr,"%s\n",error.what());return 1;}
}
