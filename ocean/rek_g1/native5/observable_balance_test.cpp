#include "observable_balance.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace ob=rek_observable_balance;
namespace {
int assertions=0;double maximum_projection_difference=0;
void check(bool pass,const char* message){++assertions;if(!pass){std::fprintf(stderr,"FAIL %s\n",message);std::exit(1);}}
void near(double actual,double expected,double tolerance,const char* message){
    ++assertions;const double error=std::abs(actual-expected);
    if(!std::isfinite(actual)||error>tolerance){std::fprintf(stderr,"FAIL %s actual=%.12g expected=%.12g error=%.9g\n",message,actual,expected,error);std::exit(1);}
}
using Q=std::array<float,4>;
Q euler(double yaw,double pitch=0,double roll=0){
    const double cy=std::cos(yaw/2),sy=std::sin(yaw/2),cp=std::cos(pitch/2),sp=std::sin(pitch/2),cr=std::cos(roll/2),sr=std::sin(roll/2);
    return {float(cy*cp*cr+sy*sp*sr),float(cy*cp*sr-sy*sp*cr),float(cy*sp*cr+sy*cp*sr),float(sy*cp*cr-cy*sp*sr)};
}
Q multiply(const Q&a,const Q&b){return {
    a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
    a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
    a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
    a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]};}
void rotation(ob::Pose&p,const Q&q){std::copy(q.begin(),q.end(),p.root_wxyz);}
ob::Snapshot fixture(){
    ob::Snapshot s{};s.round_key=1;s.sample_seconds=1;s.actor_slot=0;s.round_active=1;
    s.round_duration_seconds=120;s.round_remaining_seconds=119;s.referee_available=1;
    for(int i=0;i<2;i++){
        s.fighter[i].root_xyz[0]=float(i);s.fighter[i].root_xyz[2]=.7f;
        s.fighter[i].root_wxyz[0]=1;s.fighter[i].joint_pose_available=1;
    }
    return s;
}
std::array<float,223> project(const ob::Snapshot&s,const ob::Snapshot*old=nullptr){
    std::array<float,223> o;check(ob::project(s,old,o.data())==ob::kOk,"projection successful");return o;
}
ob::Snapshot live_equivalent(const ob::Snapshot& native){
    ob::Snapshot live=native;
    for(int i=0;i<2;i++){
        const auto&p=native.fighter[i];const float xyz[3]={p.root_xyz[0],p.root_xyz[2],p.root_xyz[1]};
        // Inverse of Unity XYZ -> common XZY and XYZW -> common (-W,X,Z,Y).
        const float xyzw[4]={-p.root_wxyz[1],-p.root_wxyz[3],-p.root_wxyz[2],p.root_wxyz[0]};
        check(ob::from_unity_root(xyz,xyzw,live.fighter[i]),"Unity root conversion");
    }
    return live;
}
void compare(const ob::Snapshot&now,const ob::Snapshot&old){
    const auto ln=live_equivalent(now),lo=live_equivalent(old);
    const auto a=project(now,&old),b=project(ln,&lo);
    for(int i=0;i<223;i++){
        maximum_projection_difference=std::max(maximum_projection_difference,std::abs(double(a[i])-b[i]));
        near(a[i],b[i],3e-5,"native/common pose and Unity/common pose agree");
        if(!ob::structurally_available(i))near(a[i],0,0,"unavailable fields remain structural padding");
    }
}
uint32_t random_state=0xabe12571u;
float random(float low,float high){random_state^=random_state<<13;random_state^=random_state>>17;random_state^=random_state<<5;return low+(high-low)*float(random_state>>8)/16777216.f;}
}
int main(){
    auto old=fixture(),now=old;auto first=project(now);
    near(first[203],0,0,"first observation history unavailable");near(first[75],0,0,"first joint derivative unavailable");
    near(first[74],1,0,"first projected joint pose available");near(first[9],0,0,"first vertical rate masked");
    near(first[202],1,0,"known referee state available");near(first[204],0,0,"observed no count");

    // Independent fixed Unity fixtures establish the handedness and local +X
    // convention; they do not rely on the inverse conversion in compare().
    auto unity_old=fixture();const float unity_position[3]={0,.7f,0};
    const float unity_positive_yaw[4]={0,float(std::sin(ob::kPi/4)),0,float(std::cos(ob::kPi/4))};
    check(ob::from_unity_root(unity_position,unity_positive_yaw,unity_old.fighter[0]),"fixed Unity positive yaw");
    unity_old.fighter[1].root_xyz[0]=0;unity_old.fighter[1].root_xyz[1]=-1;
    auto unity_now=unity_old;unity_now.sample_seconds+=.02;unity_now.fighter[0].root_xyz[1]-=.02f;
    const auto unity_observation=project(unity_now,&unity_old);
    near(unity_observation[7],1,2e-6,"Unity positive Y yaw faces negative common Y");
    near(unity_observation[8],0,2e-6,"Unity forward movement has zero lateral speed");
    near(unity_observation[87],0,2e-7,"Unity negative Z target is on local positive X bearing");
    near(unity_observation[175],-std::sqrt(.5),2e-7,"common heading quaternion has negative yaw");
    const float unity_roll[4]={0,0,.5f,float(std::sqrt(.75))};
    check(ob::from_unity_root(unity_position,unity_roll,unity_now.fighter[0]),"fixed Unity tilted root");
    const auto tilted=project(unity_now,&unity_old);
    near(tilted[72],1./3,2e-7,"Unity Z tilt has matching gravity-up angle");

    now.sample_seconds+=.02;now.round_remaining_seconds-=.02f;now.fighter[0].root_xyz[0]+=.02f;
    now.fighter[0].root_xyz[1]-=.01f;now.fighter[0].root_xyz[2]+=.03f;now.points[1]=5;now.count_mask=1;
    auto o=project(now,&old);near(o[7],1,2e-6,"heading-local horizontal rate");near(o[8],-.5,2e-6,"heading-local lateral rate");
    near(o[9],1.5,2e-6,"vertical rate from observed root origin");near(o[203],1,0,"valid shared observation history");
    near(o[217],0,0,"local weighted point delta");near(o[218],5,0,"opponent referee award remains five points");
    near(o[204],1,0,"local count mask observed");near(o[205],0,0,"opponent no count observed");compare(now,old);

    // Unavailability is distinct from a known zero, with the values masked.
    now.referee_available=0;now.count_mask=0xffffffffu;o=project(now,&old);
    near(o[202],0,0,"unavailable referee marker");near(o[204],0,0,"unknown count masked");
    now.fighter[0].joint_pose_available=0;now.fighter[0].projected_joints[0]=std::numeric_limits<float>::quiet_NaN();
    o=project(now,&old);near(o[74],0,0,"unavailable joint pose marker");near(o[75],0,0,"unavailable joint derivative marker");near(o[13],0,0,"unavailable joint values masked");

    // History cannot bridge new rounds, changed perspective or clock gaps.
    for(int mode=0;mode<4;mode++){
        now=old;now.sample_seconds+=.02;
        if(mode==0)now.round_key++;
        if(mode==1)now.actor_slot^=1;
        if(mode==2)now.sample_seconds+=1;
        if(mode==3)now.sample_seconds=old.sample_seconds;
        o=project(now,&old);near(o[203],0,0,"history boundary masked");near(o[76],0,0,"yaw derivative boundary masked");
    }
    // A same-round physical reset remains an observed displacement, matching
    // the live producer which has no privileged teleport callback.
    now=old;now.sample_seconds+=.02;now.fighter[0].root_xyz[2]=.2f;
    o=project(now,&old);near(o[9],-25,2e-6,"body teleport does not secretly reset observation history");

    // Quaternion sign aliases and angular branch crossings preserve rates.
    old=fixture();now=old;now.sample_seconds+=.02;
    rotation(old.fighter[0],euler(ob::kPi-.01));rotation(now.fighter[0],euler(-ob::kPi+.01));
    old.fighter[0].projected_joints[0]=float(ob::kPi-.01);now.fighter[0].projected_joints[0]=float(-ob::kPi+.01);
    o=project(now,&old);near(o[12],1,2e-5,"yaw crossing uses shortest angular difference");near(o[42],1,2e-5,"joint crossing uses shortest angular difference");
    const auto positive=o;for(float&v:now.fighter[0].root_wxyz)v=-v;o=project(now,&old);
    for(int i=0;i<223;i++)near(o[i],positive[i],0,"quaternion sign is not a policy feature");
    rotation(now.fighter[0],euler(0,ob::kPi/2));o=project(now,&old);
    near(o[71],0,0,"vertical forward axis makes horizontal heading unavailable");near(o[76],0,0,"undefined heading rate masked");near(o[72],.5,2e-7,"root tilt derives from quaternion");

    // Both producers project measured local bone orientation through the same
    // rest/axis contract. This fixture does not assert live joints equal qpos.
    const Q rest=euler(.3,-.2,.7);const float axis[3]={0,1,0};
    for(double angle:{-3.141, -2.,-.1,0.,.1,2.,3.141}){
        const Q twist={float(std::cos(angle/2)),0,float(std::sin(angle/2)),0};const Q measured=multiply(rest,twist);
        float recovered=0,residual=0;check(ob::project_joint(rest.data(),measured.data(),axis,recovered,residual),"hinge twist projection");
        near(std::remainder(recovered-angle,2*ob::kPi),0,5e-7,"hinge topology round trip");near(residual,0,2e-7,"pure hinge residual");
        Q negative=measured;for(float&v:negative)v=-v;float second=0;
        check(ob::project_joint(rest.data(),negative.data(),axis,second,residual),"negative bone quaternion projection");near(second,recovered,0,"joint quaternion sign invariance");
    }
    float angle=0,residual=0;const float bad_axis[3]={0,0,0};
    check(!ob::project_joint(rest.data(),rest.data(),bad_axis,angle,residual),"missing hinge axis rejected");

    old=fixture();now=old;now.sample_seconds+=.02;old.points[0]=1;
    check(ob::project(now,&old,o.data())==ob::kCounterRegressed,"same-round point decrease rejected");
    now=fixture();for(float&v:now.fighter[0].root_wxyz)v=0;
    check(ob::project(now,nullptr,o.data())==ob::kInvalidCurrent,"invalid root pose rejected");

    for(int fixture_id=0;fixture_id<4096;fixture_id++){
        old=fixture();old.actor_slot=fixture_id%2;now=old;now.sample_seconds+=fixture_id%2?.02:1./60;
        now.referee_available=fixture_id%3!=0;now.count_mask=unsigned(fixture_id%4);
        for(int side=0;side<2;side++){
            for(int k=0;k<3;k++){old.fighter[side].root_xyz[k]=random(-5,5);now.fighter[side].root_xyz[k]=old.fighter[side].root_xyz[k]+random(-.04f,.04f);}
            rotation(old.fighter[side],euler(random(-3,3),random(-1,1),random(-1,1)));
            rotation(now.fighter[side],euler(random(-3,3),random(-1,1),random(-1,1)));
            for(int j=0;j<29;j++){old.fighter[side].projected_joints[j]=random(-3,3);now.fighter[side].projected_joints[j]=old.fighter[side].projected_joints[j]+random(-.04f,.04f);}
        }
        compare(now,old);
    }
    unsigned char mask[223];ob::feature_mask(mask);int retained=0;for(int i=0;i<223;i++)retained+=mask[i];
    check(retained==166,"expected explicit structural feature count");
    std::printf("PASS observable_balance_v1 assertions=%d retained_columns=%d max_native_unity_projection_difference=%.9g no_GPU_no_runtime_changes\n",assertions,retained,maximum_projection_difference);
    return 0;
}
