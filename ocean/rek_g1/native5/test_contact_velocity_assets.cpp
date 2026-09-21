#include "fast_assets.h"
#include <mujoco/mujoco.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
static unsigned forbidden_calls=0;
extern "C" void __wrap_mj_step(const mjModel*,mjData*) {++forbidden_calls;throw std::runtime_error("physics step forbidden");}
extern "C" void __wrap_mj_forward(const mjModel*,mjData*) {++forbidden_calls;throw std::runtime_error("forward dynamics forbidden");}
extern "C" void __wrap_mj_step1(const mjModel*,mjData*) {++forbidden_calls;throw std::runtime_error("split step forbidden");}
extern "C" void __wrap_mj_step2(const mjModel*,mjData*) {++forbidden_calls;throw std::runtime_error("split step forbidden");}
static_assert(sizeof(FastFrame)==1664,"legacy GPU frame stride changed");

int main(int argc,char** argv){try{
    unsigned long long checks=0;auto check=[&](bool ok,const char* why){++checks;if(!ok)throw std::runtime_error(why);};
    check(argc==4,"Usage: test-contact-velocity-assets MODEL_XML ASSETS_DIRECTORY FEATURES_DIRECTORY");
    RekNative5Config cfg{};cfg.abi_version=REK_NATIVE5_RUNTIME_ABI;cfg.model_path=argv[1];cfg.assets_path=argv[2];cfg.motion_features_path=argv[3];cfg.arenas=1;cfg.locomotion_segment_ticks=1;
    const std::uint32_t durations[]={35,27,31,45,32,45,157,145,158,139,134,138,73,75,68,71,103};std::copy(durations,durations+17,cfg.move_duration_ticks);
    const auto legacy=load_fast_assets(cfg),explicit_legacy=load_fast_assets(cfg,false),baked=load_fast_assets(cfg,true);
    check(legacy.body_velocity_frames.empty()&&legacy.body_velocity_provenance_json.empty(),"default allocated velocity storage");
    check(explicit_legacy.body_velocity_frames.empty()&&explicit_legacy.body_velocity_provenance_json.empty(),"explicit false allocated velocity storage");
    for(const auto* other:{&explicit_legacy,&baked}) {
        check(other->frames.size()==legacy.frames.size(),"frame count changed");
        check(std::memcmp(other->frames.data(),legacy.frames.data(),legacy.frames.size()*sizeof(FastFrame))==0,"legacy frame bytes changed");
        check(std::memcmp(other->routes.data(),legacy.routes.data(),sizeof(legacy.routes))==0,"legacy route bytes changed");
        check(other->action_to_route==legacy.action_to_route&&other->move_duration_ticks==legacy.move_duration_ticks,"legacy route mapping changed");
        check(std::memcmp(other->qindices,legacy.qindices,sizeof(legacy.qindices))==0&&std::memcmp(other->vindices,legacy.vindices,sizeof(legacy.vindices))==0,"joint addresses changed");
        check(std::memcmp(other->initial_qpos,legacy.initial_qpos,sizeof(legacy.initial_qpos))==0,"initial state changed");
        check(other->provenance_json==legacy.provenance_json,"legacy provenance changed");
    }
    check(baked.body_velocity_frames.size()==baked.frames.size(),"optional frame count mismatch");
    check(!baked.body_velocity_provenance_json.empty(),"missing optional provenance");
    char error[2048]{};std::unique_ptr<mjModel,decltype(&mj_deleteModel)> model(mj_loadXML(argv[1],nullptr,error,sizeof(error)),mj_deleteModel);
    check(bool(model),error);const auto* m=model.get();
    using Data=std::unique_ptr<mjData,decltype(&mj_deleteData)>;Data canonical(mj_makeData(m),mj_deleteData),world(mj_makeData(m),mj_deleteData);
    check(canonical&&world,"data allocation failed");
    int ids[2][14],roots[2],qa[2]={-1,-1},va[2]={-1,-1};const char* prefixes[]={"player__","opponent__"};
    for(int side=0;side<2;side++) {
        for(int b=0;b<14;b++){ids[side][b]=mj_name2id(m,mjOBJ_BODY,(std::string(prefixes[side])+rek_contact_velocity::BodyNames[b]).c_str());check(ids[side][b]>=0,"missing reference body");}
        roots[side]=m->body_rootid[ids[side][0]];
        for(int j=0;j<m->njnt;j++)if(m->jnt_type[j]==mjJNT_FREE&&m->jnt_bodyid[j]==roots[side]){qa[side]=m->jnt_qposadr[j];va[side]=m->jnt_dofadr[j];}
        check(qa[side]>=0&&va[side]>=0,"reference free joint missing");
    }
    auto pose=[&](const FastFrame& f,std::array<mjtNum,72>& q) {
        std::copy(m->qpos0,m->qpos0+72,q.begin());
        for(int side=0;side<2;side++) {
            const int p=qa[side];q[p]=q[p+1]=0;q[p+2]=f.root_z;
            for(int k=0;k<4;k++)q[p+3+k]=f.root_wxyz[k];
            mju_normalize4(q.data()+p+3);
            for(int j=0;j<29;j++)q[baked.qindices[side][j]]=f.q[j];
        }
    };
    double max_world_error=0,max_pair_error=0;unsigned long long canonical_triplets=0,world_triplets=0,loop_frames=0,zero_edges=0;
    for(const auto& route:baked.routes)for(int i=0;i<route.count;i++) {
        const int previous=i?i-1:route.loop?route.count-1:0;loop_frames+=i==0&&route.loop;
        std::array<mjtNum,72> q0{},q1{};pose(baked.frames[route.offset+previous],q0);pose(baked.frames[route.offset+i],q1);
        if(q0==q1){std::fill(canonical->qvel,canonical->qvel+70,0);++zero_edges;}
        else mj_differentiatePos(m,canonical->qvel,.02,q0.data(),q1.data());
        std::copy(q1.begin(),q1.end(),canonical->qpos);mj_kinematics(m,canonical.get());mj_comPos(m,canonical.get());mj_comVel(m,canonical.get());
        const auto& f=baked.body_velocity_frames[route.offset+i];
        for(int side=0;side<2;side++) {
            for(int k=0;k<3;k++)check(f.root_com[side][k]==float(canonical->subtree_com[3*roots[side]+k]),"baked COM not exact float reference");
            for(int b=0;b<14;b++) {
                for(int k=0;k<3;k++)check(f.linear[side][b][k]==float(canonical->cvel[6*ids[side][b]+3+k]),"baked cvel not exact float reference");
                ++canonical_triplets;
            }
        }
        for(int held=0;held<2;held++)for(int scenario=0;scenario<3;scenario++) {
            const float yaw[2]={scenario==0?0.f:scenario==1?.73f:-2.1f,scenario==0?0.f:scenario==1?-1.21f:2.8f};
            const float omega[2]={scenario==0?0.f:scenario==1?1.3f:-.95f,scenario==0?0.f:scenario==1?-.8f:1.7f};
            const float vx[2]={scenario==0?0.f:.6f,scenario==0?0.f:-.2f},vy[2]={scenario==0?0.f:-.4f,scenario==0?0.f:.3f};
            std::copy(q1.begin(),q1.end(),world->qpos);
            if(held)std::fill(world->qvel,world->qvel+70,0);else std::copy(canonical->qvel,canonical->qvel+70,world->qvel);
            for(int side=0;side<2;side++) {
                const int p=qa[side];world->qpos[p]=side?-1.5:1.2;world->qpos[p+1]=side?.9:-.7;
                const mjtNum yawq[4]={std::cos(double(yaw[side])/2),0,0,std::sin(double(yaw[side])/2)};
                mju_mulQuat(world->qpos+p+3,yawq,q1.data()+p+3);
            }
            mj_kinematics(m,world.get());
            for(int side=0;side<2;side++) {
                world->qvel[va[side]]=vx[side];world->qvel[va[side]+1]=vy[side];
                for(int k=0;k<3;k++)world->qvel[va[side]+3+k]+=omega[side]*world->xmat[9*roots[side]+6+k];
            }
            mj_comPos(m,world.get());mj_comVel(m,world.get());
            rek_contact_velocity::Linear composed[2][14];
            for(int side=0;side<2;side++)for(int b=0;b<14;b++) {
                const auto v=rek_contact_velocity::compose(f,side,b,yaw[side],vx[side],vy[side],omega[side],!held);composed[side][b]=v;
                const float values[]={v.x,v.y,v.z};for(int k=0;k<3;k++)max_world_error=std::max(max_world_error,std::abs(values[k]-world->cvel[6*ids[side][b]+3+k]));++world_triplets;
            }
            for(int side=0;side<2;side++)for(int limb=0;limb<6;limb++)for(int target=0;target<9;target++) {
                const int a=rek_contact_velocity::limb_slot(limb),b=rek_contact_velocity::target_slot(target);double norm2=0;
                for(int k=0;k<3;k++){const double diff=world->cvel[6*ids[side][a]+3+k]-world->cvel[6*ids[1-side][b]+3+k];norm2+=diff*diff;}
                max_pair_error=std::max(max_pair_error,std::abs(rek_contact_velocity::relative_speed(composed[side][a],composed[1-side][b])-std::sqrt(norm2)));
            }
        }
    }
    // The independent prior probe measured ~1.6e-6 m/s host-FP32 component
    // error. This bound also covers the now-FP32 relative norm arithmetic.
    check(std::isfinite(max_world_error)&&max_world_error<1e-5,"world cvel composition error exceeds measured-reference bound");
    check(std::isfinite(max_pair_error)&&max_pair_error<1e-5,"pair cvel norm error exceeds measured-reference bound");
    check(loop_frames==7&&zero_edges>=17,"missing wrap/start coverage");check(forbidden_calls==0,"CPU dynamics call");
    std::puts(baked.body_velocity_provenance_json.c_str());
    std::printf("{\"test\":\"contact_velocity_assets\",\"passed\":true,\"checks\":%llu,\"frames\":%zu,\"legacy_frame_bytes\":%zu,\"optional_frame_bytes\":%zu,\"default_frame_bytes_unchanged\":true,\"canonical_triplets\":%llu,\"world_triplets\":%llu,\"loop_frames\":%llu,\"zero_edges\":%llu,\"max_world_component_error_m_s\":%.17g,\"max_float_pair_norm_error_m_s\":%.17g,\"bound_m_s\":1e-5,\"cpu_physics_calls\":%u}\n",checks,baked.frames.size(),sizeof(FastFrame),sizeof(FastBodyVelocityFrame),canonical_triplets,world_triplets,loop_frames,zero_edges,max_world_error,max_pair_error,forbidden_calls);
    return 0;
}catch(const std::exception& error){std::fprintf(stderr,"%s\n",error.what());return 1;}}
