// CPU-only kinematic reference experiment. No runtime integration or physics.
#include "../fast_assets.h"
#include <mujoco/mujoco.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" {
#include "../../native_motion_routes.h"
}

static unsigned forbidden_calls=0;
extern "C" void __wrap_mj_step(const mjModel*,mjData*) {++forbidden_calls;throw std::runtime_error("physics step forbidden");}
extern "C" void __wrap_mj_forward(const mjModel*,mjData*) {++forbidden_calls;throw std::runtime_error("forward dynamics forbidden");}
extern "C" void __wrap_mj_step1(const mjModel*,mjData*) {++forbidden_calls;throw std::runtime_error("split step forbidden");}
extern "C" void __wrap_mj_step2(const mjModel*,mjData*) {++forbidden_calls;throw std::runtime_error("split step forbidden");}

namespace {
constexpr double dt=.02;
using Q=std::array<mjtNum,72>;
using V=std::array<mjtNum,70>;
using Vec=std::array<double,3>;
void require(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
struct Peak {
    double value=0;int route=-1,frame=-1,scenario=-1,side=-1,body=-1;
    void add(double x,int r,int f,int s,int p,int b) {
        require(std::isfinite(x),"nonfinite comparison");
        if(x>value){value=x;route=r;frame=f;scenario=s;side=p;body=b;}
    }
    void print(const char* name) const {
        std::printf("\"%s\":{\"max_abs\":%.17g,\"route\":%d,\"frame\":%d,\"scenario\":%d,\"side\":%d,\"body\":%d}",name,value,route,frame,scenario,side,body);
    }
};
Vec rotate(Vec v,double yaw) {
    const double c=std::cos(yaw),s=std::sin(yaw);
    return {c*v[0]-s*v[1],s*v[0]+c*v[1],v[2]};
}
Vec read3(const mjtNum* v) {return {v[0],v[1],v[2]};}
double norm(Vec v) {return std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);}
Vec subtract(Vec a,Vec b) {return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
struct Tree {int root=-1,qa=-1,va=-1;std::array<int,12> strikers{};std::array<int,9> targets{};std::vector<int> bodies;};
int id(const mjModel* m,int type,const std::string& name) {
    const int result=mj_name2id(m,type,name.c_str());require(result>=0,"missing model identity");return result;
}
Tree tree(const mjModel* m,const char* prefix) {
    Tree t;const std::string p=prefix;
    const char* limbs[]={"left_ankle_roll_link_3045","right_ankle_roll_link_3090","left_wrist_yaw_link_3467","right_wrist_yaw_link_3293","left_knee_link_3106","right_knee_link_3429"};
    const int counts[]={4,4,1,1,1,1};int n=0;
    for(int k=0;k<6;k++) {
        const int body=id(m,mjOBJ_BODY,p+limbs[k]);int count=0;
        for(int g=0;g<m->ngeom;g++)if(m->geom_bodyid[g]==body){require(n<12,"too many strikers");t.strikers[n++]=g;count++;}
        require(count==counts[k],"striker geometry count mismatch");
    }
    for(int k=0;k<9;k++) {
        t.targets[k]=id(m,mjOBJ_GEOM,p+rek5_native_contact::TargetNames[k]);
        require(m->geom_bodyid[t.targets[k]]==id(m,mjOBJ_BODY,p+rek5_native_contact::TargetBodies[k]),"target body mismatch");
    }
    t.root=m->body_rootid[m->geom_bodyid[t.targets[0]]];
    for(int j=0;j<m->njnt;j++)if(m->jnt_bodyid[j]==t.root&&m->jnt_type[j]==mjJNT_FREE) {
        require(t.qa<0,"multiple root free joints");t.qa=m->jnt_qposadr[j];t.va=m->jnt_dofadr[j];
    }
    require(t.qa>=0&&t.va>=0,"missing free root joint");
    std::set<int> unique;
    for(int g:t.strikers)unique.insert(m->geom_bodyid[g]);
    for(int g:t.targets)unique.insert(m->geom_bodyid[g]);
    t.bodies.assign(unique.begin(),unique.end());require(t.bodies.size()==14,"expected six striker plus eight target bodies");
    for(int b:t.bodies)require(m->body_rootid[b]==t.root,"mixed root tree");
    return t;
}
void pose(const mjModel* m,const FastAssets& a,const std::array<Tree,2>& trees,const FastFrame& frame,Q& q,bool reflected) {
    // Synthetic sensitivity sweep of real route frames, using the loader's
    // existing reflection map. It is not an assertion of a mirrored route.
    const int mirror_index[29]={6,7,8,9,10,11,0,1,2,3,4,5,12,13,14,22,23,24,25,26,27,28,15,16,17,18,19,20,21};
    const int mirror_negate[29]={0,1,1,0,0,1,0,1,1,0,0,1,1,1,0,0,1,1,0,1,0,1,0,1,1,0,1,0,1};
    std::copy(m->qpos0,m->qpos0+72,q.begin());
    for(int side=0;side<2;side++) {
        const int p=trees[side].qa;q[p]=q[p+1]=0;q[p+2]=frame.root_z;
        for(int k=0;k<4;k++)q[p+3+k]=frame.root_wxyz[k];
        if(reflected){q[p+4]=-q[p+4];q[p+6]=-q[p+6];}
        // FastFrame quaternions are float; explicitly put them on the manifold.
        mju_normalize4(q.data()+p+3);
        for(int j=0;j<29;j++)q[a.qindices[side][j]]=frame.q[reflected?mirror_index[j]:j]*(reflected&&mirror_negate[j]?-1:1);
    }
}
struct Motion {double yaw,omega;Vec translation,velocity;};
using Motions=std::array<Motion,2>;
Q world_pose(const Q& canonical,const std::array<Tree,2>& trees,const Motions& motion,double time=0) {
    Q result=canonical;
    for(int side=0;side<2;side++) {
        const int p=trees[side].qa;const auto& x=motion[side];const double yaw=x.yaw+time*x.omega;
        const Vec pos=rotate(read3(canonical.data()+p),yaw);
        for(int k=0;k<3;k++)result[p+k]=pos[k]+x.translation[k]+time*x.velocity[k];
        const mjtNum yawq[4]={std::cos(yaw/2),0,0,std::sin(yaw/2)};
        mju_mulQuat(result.data()+p+3,yawq,canonical.data()+p+3);
    }
    return result;
}
V world_velocity(const Q& canonical,const Q& world,const V& velocity,const std::array<Tree,2>& trees,const Motions& motion) {
    V result=velocity;
    for(int side=0;side<2;side++) {
        const auto& t=trees[side];const auto& x=motion[side];
        const Vec pos=rotate(read3(canonical.data()+t.qa),x.yaw);
        const Vec v=rotate(read3(velocity.data()+t.va),x.yaw);
        result[t.va]=v[0]+x.velocity[0]-x.omega*pos[1];
        result[t.va+1]=v[1]+x.velocity[1]+x.omega*pos[0];
        result[t.va+2]=v[2]+x.velocity[2];
        // Free-joint angular qvel is expressed in the rotating root frame.
        mjtNum inverse[4],local[3];const mjtNum omega[3]={0,0,x.omega};
        mju_negQuat(inverse,world.data()+t.qa+3);mju_rotVecQuat(local,omega,inverse);
        for(int k=0;k<3;k++)result[t.va+3+k]+=local[k];
    }
    return result;
}
void calculate(const mjModel* m,mjData* d,const Q& q,const V& v) {
    std::copy(q.begin(),q.end(),d->qpos);std::copy(v.begin(),v.end(),d->qvel);
    mj_kinematics(m,d);mj_comPos(m,d);mj_comVel(m,d);
}
Vec compose(const mjData* canonical,int root,int body,const Motion& x) {
    Vec linear=rotate(read3(canonical->cvel+6*body+3),x.yaw);
    const Vec com=rotate(read3(canonical->subtree_com+3*root),x.yaw);
    linear[0]+=x.velocity[0]-x.omega*com[1];linear[1]+=x.velocity[1]+x.omega*com[0];linear[2]+=x.velocity[2];
    return linear;
}
Vec compose_float(const mjData* canonical,int root,int body,const Motion& x) {
    const float c=std::cos(float(x.yaw)),s=std::sin(float(x.yaw)),w=float(x.omega);
    const auto* l=canonical->cvel+6*body+3;const auto* p=canonical->subtree_com+3*root;
    const float cx=c*float(p[0])-s*float(p[1]),cy=s*float(p[0])+c*float(p[1]);
    return {float(c*float(l[0])-s*float(l[1])+float(x.velocity[0])-w*cy),
            float(s*float(l[0])+c*float(l[1])+float(x.velocity[1])+w*cx),float(float(l[2])+float(x.velocity[2]))};
}
struct Stats {
    Peak linear,float_linear,speed,float_speed,basis,com,bad_world_angular,bad_geom_reference;
    unsigned long long cases=0,body_triplets=0,pair_speeds=0,shared_checks=0,basis_cases=0,zero_edges=0,switches=0,clamps=0,loops=0;
    double max_root_tilt=0,max_zero_rate=0,max_same_pose_differentiate_residual=0,max_discarded_switch_rate=0;
};
void compare(const mjModel* m,const std::array<Tree,2>& trees,mjData* canonical,mjData* world,mjData* wrong,
             const Q& q,const V& v,const Motions& motion,int r,int frame,int scenario,bool basis_check,Stats& stats) {
    const Q wq=world_pose(q,trees,motion);const V wv=world_velocity(q,wq,v,trees,motion);calculate(m,world,wq,wv);
    ++stats.cases;
    for(int side=0;side<2;side++) {
        const auto& t=trees[side];const auto& x=motion[side];
        const Vec com=rotate(read3(canonical->subtree_com+3*t.root),x.yaw);
        for(int k=0;k<3;k++)stats.com.add(std::abs(world->subtree_com[3*t.root+k]-com[k]-x.translation[k]),r,frame,scenario,side,t.root);
        for(int b:t.bodies) {
            const Vec l=compose(canonical,t.root,b,x),f=compose_float(canonical,t.root,b,x),direct=read3(world->cvel+6*b+3);
            for(int k=0;k<3;k++){stats.linear.add(std::abs(l[k]-direct[k]),r,frame,scenario,side,b);stats.float_linear.add(std::abs(f[k]-direct[k]),r,frame,scenario,side,b);}
            ++stats.body_triplets;
        }
        // Geometry IDs stay distinct; multiple geometries read exactly one body entry.
        std::vector<int> geoms(t.strikers.begin(),t.strikers.end());geoms.insert(geoms.end(),t.targets.begin(),t.targets.end());
        for(size_t a=0;a<geoms.size();a++)for(size_t b=a+1;b<geoms.size();b++)if(m->geom_bodyid[geoms[a]]==m->geom_bodyid[geoms[b]]) {
            require(geoms[a]!=geoms[b],"duplicated geometry identity");
            const Vec va=read3(world->cvel+6*m->geom_bodyid[geoms[a]]+3),vb=read3(world->cvel+6*m->geom_bodyid[geoms[b]]+3);
            require(va==vb,"same-body geometry velocity mismatch");++stats.shared_checks;
        }
        const auto& other=trees[1-side];
        for(int g:t.strikers)for(int h:other.targets) {
            const int b=m->geom_bodyid[g],ob=m->geom_bodyid[h];
            const double direct=norm(subtract(read3(world->cvel+6*b+3),read3(world->cvel+6*ob+3)));
            const double composed=norm(subtract(compose(canonical,t.root,b,x),compose(canonical,other.root,ob,motion[1-side])));
            const double f=norm(subtract(compose_float(canonical,t.root,b,x),compose_float(canonical,other.root,ob,motion[1-side])));
            stats.speed.add(std::abs(composed-direct),r,frame,scenario,side,b);stats.float_speed.add(std::abs(f-direct),r,frame,scenario,side,b);++stats.pair_speeds;
        }
        // Deliberately incorrect point reference, retained as a sensitivity diagnostic.
        for(int g:t.strikers) {
            const int b=m->geom_bodyid[g];const Vec point=rotate(read3(canonical->geom_xpos+3*g),x.yaw);
            const Vec offset={-x.omega*(point[1]-com[1]),x.omega*(point[0]-com[0]),0};
            stats.bad_geom_reference.add(norm(offset),r,frame,scenario,side,b);
        }
    }
    if(basis_check) {
        // Independent centered derivative of a composed pose path. integratePos
        // advances coordinates only; it executes no forces, controller, or solver.
        constexpr double epsilon=1e-6;Q qm=q,qp=q;
        mj_integratePos(m,qm.data(),v.data(),-epsilon);mj_integratePos(m,qp.data(),v.data(),epsilon);
        const Q wm=world_pose(qm,trees,motion,-epsilon),wp=world_pose(qp,trees,motion,epsilon);V numeric{};
        mj_differentiatePos(m,numeric.data(),2*epsilon,wm.data(),wp.data());
        for(int k=0;k<70;k++)stats.basis.add(std::abs(numeric[k]-wv[k]),r,frame,scenario,-1,k);
        ++stats.basis_cases;
        V wrong_v=wv;
        for(int side=0;side<2;side++)for(int k=0;k<3;k++)wrong_v[trees[side].va+3+k]=v[trees[side].va+3+k]+(k==2?motion[side].omega:0);
        calculate(m,wrong,wq,wrong_v);
        for(int side=0;side<2;side++)for(int b:trees[side].bodies)for(int k=0;k<3;k++)
            stats.bad_world_angular.add(std::abs(wrong->cvel[6*b+3+k]-world->cvel[6*b+3+k]),r,frame,scenario,side,b);
    }
}
}

int main(int argc,char** argv) {
    try {
        require(argc==4,"Usage: contact-velocity-probe MODEL_XML ASSETS_DIRECTORY FEATURES_DIRECTORY");
        require(mj_version()==mjVERSION_HEADER&&std::string(mj_versionString())=="3.7.0","probe requires matching pinned MuJoCo 3.7.0 headers and library");
        RekNative5Config cfg{};cfg.abi_version=REK_NATIVE5_RUNTIME_ABI;cfg.model_path=argv[1];cfg.assets_path=argv[2];cfg.motion_features_path=argv[3];cfg.arenas=1;cfg.locomotion_segment_ticks=1;
        const std::uint32_t durations[]={35,27,31,45,32,45,157,145,158,139,134,138,73,75,68,71,103};
        std::copy(durations,durations+17,cfg.move_duration_ticks);const auto assets=load_fast_assets(cfg);
        require(assets.recovered_catalog_compatible,"route catalog differs; mirrored/reverse coverage would be unbound");
        char error[2048]{};std::unique_ptr<mjModel,decltype(&mj_deleteModel)> model(mj_loadXML(argv[1],nullptr,error,sizeof(error)),mj_deleteModel);
        require(bool(model),error);const auto* m=model.get();require(m->nq==72&&m->nv==70,"unexpected model dimensions");
        using Data=std::unique_ptr<mjData,decltype(&mj_deleteData)>;
        Data canonical(mj_makeData(m),mj_deleteData),world(mj_makeData(m),mj_deleteData),wrong(mj_makeData(m),mj_deleteData);
        require(canonical&&world&&wrong,"data allocation failed");
        const std::array<Tree,2> trees={tree(m,"player__"),tree(m,"opponent__")};
        const std::array<Motions,5> scenarios={{
            {{{0,0,{0,0,0},{0,0,0}},{0,0,{0,0,0},{0,0,0}}}},
            {{{.73,1.3,{1.2,-.7,0},{.6,-.4,0}},{-1.21,-.8,{-1.5,.9,0},{-.2,.3,0}}}},
            {{{-2.1,-.95,{-2,.5,0},{-.7,.15,0}},{2.8,1.7,{2,-.5,0},{.4,-.35,0}}}},
            {{{0,0,{.6,.2,0},{.37,-.91,0}},{0,0,{-.6,-.2,0},{-.23,.41,0}}}},
            {{{1.4,-1.6,{.8,1.1,0},{0,0,0}},{-2.4,1.2,{-.8,-1.1,0},{0,0,0}}}}
        }};
        Stats stats;int mirrored_routes=0,reverse_routes=0,loop_routes=0;const auto* catalog=rek_g1_native_static_motion_routes();
        for(int reflected=0;reflected<2;reflected++)for(int r=0;r<24;r++) {
            const auto& route=assets.routes[r];
            if(!reflected){mirrored_routes+=catalog->routes[r].mirror!=0;reverse_routes+=route.playback_speed<0;loop_routes+=route.loop!=0;}
            auto edge=[&](int before,int after,const char* kind) {
                Q q0{},q1{};V v{};pose(m,assets,trees,assets.frames[route.offset+before],q0,reflected);pose(m,assets,trees,assets.frames[route.offset+after],q1,reflected);
                mj_differentiatePos(m,v.data(),dt,q0.data(),q1.data());
                if(before==after) {
                    require(q0==q1,"identical-frame edge changed pose");++stats.zero_edges;
                    for(double x:v)stats.max_same_pose_differentiate_residual=std::max(stats.max_same_pose_differentiate_residual,std::abs(x));
                    // Quaternion self-products can leave numerical residue.
                    // A known identical-coordinate edge has exactly zero rate.
                    v.fill(0);for(double x:v)stats.max_zero_rate=std::max(stats.max_zero_rate,std::abs(x));
                }
                if(std::string(kind)=="route_switch") {
                    ++stats.switches;Q unrelated{};const auto& prior=assets.routes[(r+23)%24];
                    pose(m,assets,trees,assets.frames[prior.offset+prior.count-1],unrelated,reflected);V jump{};mj_differentiatePos(m,jump.data(),dt,unrelated.data(),q1.data());
                    for(double x:jump)stats.max_discarded_switch_rate=std::max(stats.max_discarded_switch_rate,std::abs(x));
                }
                if(std::string(kind)=="clamped")++stats.clamps;
                if(std::string(kind)=="loop")++stats.loops;
                calculate(m,canonical.get(),q1,v);
                for(const auto& t:trees)stats.max_root_tilt=std::max(stats.max_root_tilt,std::acos(std::clamp(canonical->xmat[9*t.root+8],-1.0,1.0)));
                for(int s=0;s<int(scenarios.size());s++)compare(m,trees,canonical.get(),world.get(),wrong.get(),q1,v,scenarios[s],r,after,s+5*reflected,after%19==0||before==after,stats);
            };
            edge(0,0,"route_switch"); // Existing runtime reconciles old pose to the new route's t=0.
            for(int frame=1;frame<route.count;frame++)edge(frame-1,frame,"incoming");
            if(route.loop)edge(route.count-1,0,"loop");else edge(route.count-1,route.count-1,"clamped");
        }
        require(stats.max_zero_rate==0,"identical pose edge acquired velocity");require(forbidden_calls==0,"forbidden CPU dynamics call");
        require(reverse_routes>0&&loop_routes>0&&stats.max_root_tilt>.01,"missing actual route coverage");
        std::printf("{\"schema\":\"rek.contact_velocity_probe.v1\",\"structural_checks_passed\":true,\"numerical_tolerance_selected\":false,\"physical_parity_claim\":false,\"cpu_physics_steps\":%u,\"mujoco_version\":\"%s\",\"model_sha256\":\"%s\",\"asset_manifest_sha256\":\"%s\",\"features_manifest_sha256\":\"%s\",\"routes\":24,\"frames\":%zu,\"mirrored_routes\":%d,\"reverse_routes\":%d,\"loop_routes\":%d,\"scenarios\":5,\"cases\":%llu,\"body_triplets\":%llu,\"geometry_pair_speeds\":%llu,\"shared_body_geometry_checks\":%llu,\"basis_cases\":%llu,\"zero_edges\":%llu,\"route_switch_edges\":%llu,\"clamped_edges\":%llu,\"loop_edges\":%llu,\"max_root_tilt_rad\":%.17g,\"max_zero_edge_rate\":%.17g,\"max_discarded_route_jump_rate\":%.17g,",
            forbidden_calls,mj_versionString(),assets.model_sha256.c_str(),assets.manifest_sha256.c_str(),assets.features_sha256.c_str(),assets.frames.size(),mirrored_routes,reverse_routes,loop_routes,stats.cases,stats.body_triplets,stats.pair_speeds,stats.shared_checks,stats.basis_cases,stats.zero_edges,stats.switches,stats.clamps,stats.loops,stats.max_root_tilt,stats.max_zero_rate,stats.max_discarded_switch_rate);
        std::printf("\"synthetic_reflection_sweeps\":1,\"scenario_indices_0_to_4\":\"actual_frames\",\"scenario_indices_5_to_9\":\"synthetically_reflected_frames\",\"max_same_pose_differentiate_residual\":%.17g,",stats.max_same_pose_differentiate_residual);
        stats.linear.print("double_linear_m_per_s");std::putchar(',');stats.float_linear.print("float_linear_m_per_s");std::putchar(',');
        stats.speed.print("double_relative_speed_m_per_s");std::putchar(',');stats.float_speed.print("float_relative_speed_m_per_s");std::putchar(',');
        stats.com.print("com_position_m");std::putchar(',');stats.basis.print("world_qvel_centered_derivative");std::putchar(',');
        stats.bad_world_angular.print("incorrect_world_angular_basis_linear_m_per_s");std::putchar(',');stats.bad_geom_reference.print("incorrect_geom_reference_linear_m_per_s");std::puts("}");
        return 0;
    }catch(const std::exception& error){std::fprintf(stderr,"contact velocity probe: %s\n",error.what());return 1;}
}
