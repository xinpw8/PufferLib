#include "rendered_pose_observation.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

// CPU-only fixture regression. The independent double oracle below follows
// encode_live.cpp's normalized-root, finite-difference and score-delta formulas.
// It does not invoke the live bridge, a policy, Python, physics, or CUDA.
namespace {
using Q=std::array<float,4>;
constexpr double PI=3.14159265358979323846;
int assertions=0;
double maximum_error=0;
void check(bool ok,const char* label){
    ++assertions;if(!ok){std::fprintf(stderr,"FAIL %s\n",label);std::exit(1);}
}
void near(double actual,double expected,double tolerance,const char* label){
    const double error=std::abs(actual-expected);maximum_error=std::fmax(maximum_error,error);
    ++assertions;if(!std::isfinite(actual)||error>tolerance){
        std::fprintf(stderr,"FAIL %s actual=%.17g expected=%.17g error=%.9g tolerance=%.9g\n",label,actual,expected,error,tolerance);std::exit(1);
    }
}
double live_heading(const Q& q){
    double norm=0;for(float v:q)norm+=double(v)*v;
    const double w=q[0]/std::sqrt(norm),x=q[1]/std::sqrt(norm),y=q[2]/std::sqrt(norm),z=q[3]/std::sqrt(norm);
    return std::atan2(2*(w*z+x*y),1-2*(y*y+z*z));
}
double live_rate(double current,double previous,double dt){
    return dt>0?std::remainder(current-previous,2*PI)/dt:0;
}
Q euler(double yaw,double pitch=0,double roll=0){
    const double cy=std::cos(yaw/2),sy=std::sin(yaw/2),cp=std::cos(pitch/2),sp=std::sin(pitch/2),cr=std::cos(roll/2),sr=std::sin(roll/2);
    return {float(cy*cp*cr+sy*sp*sr),float(cy*cp*sr-sy*sp*cr),float(cy*sp*cr+sy*cp*sr),float(sy*cp*cr-cy*sp*sr)};
}
Q world(float logical,const Q& clip){Q out;rek_rendered_pose::compose_root(logical,clip.data(),out.data());return out;}
struct Pose {float x,y;Q root;std::array<float,29> joints;};
std::array<double,223> oracle(const Pose& now,const Pose& old,const Pose& enemy,float dt,int delta0,int delta1){
    std::array<double,223> result{};
    const double theta=live_heading(now.root),oldtheta=live_heading(old.root);
    const double vx=dt>0?(double(now.x)-old.x)/dt:0,vy=dt>0?(double(now.y)-old.y)/dt:0;
    result[7]=std::cos(theta)*vx+std::sin(theta)*vy;result[8]=-std::sin(theta)*vx+std::cos(theta)*vy;
    result[12]=live_rate(theta,oldtheta,dt);
    for(int k=0;k<29;k++)result[42+k]=live_rate(now.joints[k],old.joints[k],dt);
    result[87]=std::remainder(std::atan2(double(enemy.y)-now.y,double(enemy.x)-now.x)-theta,2*PI)/PI;
    result[172]=std::cos(theta/2);result[175]=std::sin(theta/2);
    result[217]=delta0;result[218]=delta1;result[221]=result[222]=delta0+delta1;
    return result;
}
void compare(const Pose& now,const Pose& old,const Pose& enemy,float dt,int delta0=0,int delta1=0){
    const auto expected=oracle(now,old,enemy,dt,delta0,delta1);
    const float theta=rek_rendered_pose::heading(now.root.data()),oldtheta=rek_rendered_pose::heading(old.root.data());
    float v[2],q[4];rek_rendered_pose::local_velocity(now.x,now.y,old.x,old.y,theta,dt,v);
    rek_rendered_pose::heading_quaternion(theta,q);
    near(v[0],expected[7],5e-5,"actor-local vx from exported positions");near(v[1],expected[8],5e-5,"actor-local vy from exported positions");
    near(rek_rendered_pose::angular_rate(theta,oldtheta,dt),expected[12],5e-5,"rendered root yaw rate");
    for(int k=0;k<29;k++)near(rek_rendered_pose::joint_rate(now.joints[k],old.joints[k],dt),expected[42+k],2e-5,"wrapped joint finite difference");
    near(rek_rendered_pose::bearing(now.x,now.y,enemy.x,enemy.y,theta),expected[87],2e-7,"rendered relative bearing");
    near(q[0],expected[172],2e-7,"semantic heading w");near(q[3],expected[175],2e-7,"semantic heading z");
    near(q[1],0,0,"semantic heading x");near(q[2],0,0,"semantic heading y");
    near(rek_rendered_pose::weighted_delta_total(delta0,delta1),expected[221],0,"weighted actor window total");
    near(rek_rendered_pose::weighted_delta_total(delta1,delta0),expected[222],0,"weighted opponent window total");
}
uint32_t random_state=0x6ab21f09u;
float sample(float lo,float hi){random_state^=random_state<<13;random_state^=random_state>>17;random_state^=random_state<<5;return lo+(hi-lo)*float(random_state>>8)*(1.f/16777216.f);}
}
int main(){
    Pose old{1,2,world(.4f,euler(.1,.3,-.2)),{}},now=old,enemy{2,4,euler(-.1),{}};
    now.root=world(.4f,euler(.5,.3,-.2));
    compare(now,old,enemy,.02f);
    check(std::abs(rek_rendered_pose::angular_rate(rek_rendered_pose::heading(now.root.data()),rek_rendered_pose::heading(old.root.data()),.02f))>19,"clip yaw produces yaw rate with unchanged logical heading");

    // Crossing the yaw and joint +/-pi branch uses the shortest delta.
    old.root=euler(PI-.01);now.root=euler(-PI+.01);
    old.joints[0]=float(PI-.02);now.joints[0]=float(-PI+.02);
    compare(now,old,enemy,.02f);
    near(rek_rendered_pose::angular_rate(rek_rendered_pose::heading(now.root.data()),rek_rendered_pose::heading(old.root.data()),.02f),1,2e-5,"yaw wrap crossing");
    Q negative=now.root;for(float& value:negative)value=-value;
    near(rek_rendered_pose::heading(negative.data()),rek_rendered_pose::heading(now.root.data()),0,"quaternion sign invariance");
    Q scaled=now.root;for(float& value:scaled)value*=1.02f;
    near(rek_rendered_pose::heading(scaled.data()),live_heading(scaled),2e-7,"quaternion normalization agreement");

    // A route-entry collision snapshot is replaced with the new frame. The
    // separate observation snapshot must retain the actual prior exported pose.
    Pose observed_previous=old;
    now.root=world(.2f,euler(.9,.1,.2));now.joints[0]=.7f;
    Pose collision_previous=now;
    compare(now,observed_previous,enemy,.02f);
    check(std::abs(rek_rendered_pose::joint_rate(now.joints[0],observed_previous.joints[0],.02f))>10,"route transition uses true previous joint frame");
    near(rek_rendered_pose::joint_rate(now.joints[0],collision_previous.joints[0],.02f),0,0,"collision replacement would conceal transition");
    check(std::abs(rek_rendered_pose::angular_rate(rek_rendered_pose::heading(now.root.data()),rek_rendered_pose::heading(observed_previous.root.data()),.02f))>10,"route transition uses true previous rendered root");

    // A separation or wall correction contributes measured velocity even when
    // the internal locomotion velocity is zero. Runtime integration owns it.
    old=now;now.x+=.012f;now.y-=.006f;compare(now,old,enemy,.02f);
    float velocity[2];rek_rendered_pose::local_velocity(now.x,now.y,old.x,old.y,rek_rendered_pose::heading(now.root.data()),.02f,velocity);
    check(std::hypot(velocity[0],velocity[1])>.6,"position correction survives into observed velocity");
    compare(now,now,enemy,.02f);compare(now,old,enemy,0);compare(now,old,enemy,-.02f);

    // Counter observations cannot distinguish a two-point foot/shin hit from
    // two one-point hand hits. Referee increments have the same ambiguity.
    for(const auto delta:std::array<std::array<int,2>,7>{{{{0,0}},{{1,0}},{{2,0}},{{0,2}},{{1,2}},{{2,2}},{{3,0}}}})compare(now,old,enemy,.02f,delta[0],delta[1]);
    near(rek_rendered_pose::weighted_delta_total(2,0),2,0,"one kick emits two weighted window points");

    for(int fixture=0;fixture<4096;fixture++){
        const float logical=sample(-float(PI),float(PI));
        old={sample(-5,5),sample(-5,5),world(logical,euler(sample(-3,3),sample(-1,1),sample(-1,1))),{}};
        now=old;now.x+=sample(-.03f,.03f);now.y+=sample(-.03f,.03f);
        now.root=world(logical+sample(-.04f,.04f),euler(sample(-3,3),sample(-1,1),sample(-1,1)));
        enemy={sample(-5,5),sample(-5,5),euler(0),{}};
        for(int k=0;k<29;k++){old.joints[k]=sample(-float(PI),float(PI));now.joints[k]=sample(-float(PI),float(PI));}
        compare(now,old,enemy,fixture%3==0?.02f:fixture%3==1?1.f/60:1.f/30,fixture%3,(fixture/3)%3);
        compare(old,now,enemy,.02f);
    }
    std::printf("PASS rendered_pose_v1 CPU fixtures assertions=%d max_abs_error=%.9g oracle=encode_live_formulas_double no_policy_no_physics_no_GPU\n",assertions,maximum_error);
}
