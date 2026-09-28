#ifndef REK_NATIVE5_OBSERVABLE_BALANCE_H
#define REK_NATIVE5_OBSERVABLE_BALANCE_H

#include <math.h>
#include <stdint.h>

#ifdef __CUDACC__
#define REK_OBSERVABLE_BALANCE_FN __host__ __device__ inline
#else
#define REK_OBSERVABLE_BALANCE_FN inline
#endif

// An opt-in observation contract, independent of dynamics and rewards.
// The caller must use this same projection for training and inference. These
// features are incompatible with existing scaled_polar_xy/owned_yaw weights.
namespace rek_observable_balance {
constexpr const char* kSchema="rek.native5.observable_balance.v1";
constexpr int kFeatures=223;
constexpr double kPi=3.14159265358979323846;
constexpr double kMaximumHistorySeconds=.25;
constexpr int kRefereeAvailable=202;
constexpr int kHistoryAvailable=203;
constexpr int kCountActive=204;

struct Pose {
    float root_xyz[3];             // Common z-up coordinates, root origin.
    float root_wxyz[4];            // Common frame, quaternion sign immaterial.
    float projected_joints[29];    // Pose-derived hinge twists, radians.
    int joint_pose_available;      // Zero means angles are unavailable.
};

struct Snapshot {
    Pose fighter[2];               // Absolute fighter slots, not perspective.
    uint64_t round_key;            // Same round identity; never a body-reset ID.
    double sample_seconds;        // Current observation time in one clock domain.
    float round_duration_seconds;
    float round_remaining_seconds;
    int points[2];                // Awarded points, including referee awards.
    int actor_slot;
    int round_active;
    int terminal;
    int referee_available;        // Live: validated fresh lifecycle-bound receipt.
    unsigned count_mask;          // Live received bits or native count_active[].
};

enum Status {kOk=0,kInvalidCurrent=1,kCounterRegressed=2,kNonfiniteOutput=3};

REK_OBSERVABLE_BALANCE_FN bool flag(int value){return value==0||value==1;}
REK_OBSERVABLE_BALANCE_FN float wrap_angle(float angle){
    float result=float(remainder(double(angle),2*kPi));
    // Canonical endpoint avoids a +pi/-pi difference caused only by q versus -q.
    return result>=float(kPi)?-float(kPi):result;
}
REK_OBSERVABLE_BALANCE_FN bool normalize_quaternion(const float* source,float* out){
    double norm=0;
    for(int i=0;i<4;i++){if(!isfinite(source[i]))return false;norm+=double(source[i])*source[i];}
    if(!(norm>1e-20)||!isfinite(norm))return false;
    const double inverse=1/sqrt(norm);
    for(int i=0;i<4;i++)out[i]=float(double(source[i])*inverse);
    // A deterministic hemisphere, including the exact w==0 half-turn case.
    int first=0;while(first<4&&out[first]==0.f)first++;
    if(first<4&&out[first]<0.f)for(int i=0;i<4;i++)out[i]=-out[i];
    return true;
}
REK_OBSERVABLE_BALANCE_FN bool from_unity_root(const float* xyz,const float* xyzw,Pose& out){
    out.root_xyz[0]=xyz[0];out.root_xyz[1]=xyz[2];out.root_xyz[2]=xyz[1];
    const float common[4]={-xyzw[3],xyzw[0],xyzw[2],xyzw[1]};
    return isfinite(xyz[0])&&isfinite(xyz[1])&&isfinite(xyz[2])&&normalize_quaternion(common,out.root_wxyz);
}
REK_OBSERVABLE_BALANCE_FN bool valid_pose(const Pose& pose){
    float normalized[4];
    if(!flag(pose.joint_pose_available)||!normalize_quaternion(pose.root_wxyz,normalized))return false;
    for(int i=0;i<3;i++)if(!isfinite(pose.root_xyz[i]))return false;
    if(pose.joint_pose_available)for(int i=0;i<29;i++)if(!isfinite(pose.projected_joints[i]))return false;
    return true;
}
REK_OBSERVABLE_BALANCE_FN bool heading(const float* normalized,float& angle){
    const double w=normalized[0],x=normalized[1],y=normalized[2],z=normalized[3];
    const double forward_x=w*w+x*x-y*y-z*z,forward_y=2*(w*z+x*y);
    // Numerical observability of projected local +X, not a fall threshold.
    const bool available=forward_x*forward_x+forward_y*forward_y>1e-12;
    angle=available?float(atan2(forward_y,forward_x)):0.f;
    return available;
}
REK_OBSERVABLE_BALANCE_FN float tilt_fraction(const float* q){
    double cosine=1-2*(double(q[1])*q[1]+double(q[2])*q[2]);
    cosine=cosine<-1?-1:cosine>1?1:cosine;
    return float(acos(cosine)/kPi);
}

// The live producer uses client-local bone orientation, model rest orientation
// and model hinge axis. A physical producer may construct the equivalent local
// bone orientation or read it from its body frames and call this same function.
// Do not declare raw physical hinge coordinates equivalent without that mapping.
// Off-axis residual is diagnostic. It is never converted into a contact/fall.
REK_OBSERVABLE_BALANCE_FN bool project_joint(const float* rest_wxyz,const float* measured_wxyz,
        const float* hinge_axis,float& angle,float& off_axis_radians){
    float a[4],b[4];
    if(!normalize_quaternion(rest_wxyz,a)||!normalize_quaternion(measured_wxyz,b))return false;
    double axis_norm=0;for(int i=0;i<3;i++){if(!isfinite(hinge_axis[i]))return false;axis_norm+=double(hinge_axis[i])*hinge_axis[i];}
    if(fabs(axis_norm-1)>1e-4)return false;
    const double rw=double(a[0])*b[0]+double(a[1])*b[1]+double(a[2])*b[2]+double(a[3])*b[3];
    const double r[3]={double(a[0])*b[1]-double(a[1])*b[0]-double(a[2])*b[3]+double(a[3])*b[2],
        double(a[0])*b[2]+double(a[1])*b[3]-double(a[2])*b[0]-double(a[3])*b[1],
        double(a[0])*b[3]-double(a[1])*b[2]+double(a[2])*b[1]-double(a[3])*b[0]};
    double along=0;for(int i=0;i<3;i++)along+=r[i]*hinge_axis[i];
    if(rw*rw+along*along<1e-16)return false;
    angle=wrap_angle(float(2*atan2(along,rw)));
    double residual=0;for(int i=0;i<3;i++){const double off=r[i]-along*hinge_axis[i];residual+=off*off;}
    residual=sqrt(residual);off_axis_radians=float(2*asin(residual<1?residual:1));
    return isfinite(angle)&&isfinite(off_axis_radians);
}

// An excluded column is structurally unavailable in this schema. Its numerical
// zero is padding, not an observed zero. This mask can be used by the existing
// policy_feature_mask pipeline. Per-sample availability has separate features.
REK_OBSERVABLE_BALANCE_FN bool structurally_available(int column){
    if(column<0||column>=kFeatures)return false;
    if(column<172){const int local=column%86;return local<=9||(local>=12&&local<=76);}
    return (column>=172&&column<=175)||column==184||column==185||
        (column>=188&&column<=191)||(column>=202&&column<=205)||column==217||column==218;
}
REK_OBSERVABLE_BALANCE_FN void feature_mask(unsigned char* out){
    for(int i=0;i<kFeatures;i++)out[i]=structurally_available(i)?1:0;
}

// Pass the actual preceding observation snapshot. Clear history only for a new
// round, explicit stream reset or unavailable observation. Never substitute a
// collision-sweep frame or clear on a privileged physical fall/body teleport.
// Gaps >250 ms and nonpositive deltas produce unavailable derivatives. The
// caller still retains the current valid snapshot as the next history entry.
// A successful first observation is legal: history/rates are explicitly masked.
// The caller must discard output when the returned status is not kOk.
REK_OBSERVABLE_BALANCE_FN Status project(const Snapshot& now,const Snapshot* previous,float* out){
    if(!out||!valid_pose(now.fighter[0])||!valid_pose(now.fighter[1])||
            !isfinite(now.sample_seconds)||now.actor_slot<0||now.actor_slot>1||
            !flag(now.round_active)||!flag(now.terminal)||!flag(now.referee_available)||
            !isfinite(now.round_duration_seconds)||now.round_duration_seconds<=0||
            !isfinite(now.round_remaining_seconds)||now.round_remaining_seconds<0||
            now.round_remaining_seconds>now.round_duration_seconds+.1f||
            now.points[0]<0||now.points[1]<0||(now.referee_available&&now.count_mask>3))return kInvalidCurrent;
    const double dt=previous?now.sample_seconds-previous->sample_seconds:0;
    const bool history=previous&&previous->round_key==now.round_key&&previous->actor_slot==now.actor_slot&&
        isfinite(dt)&&dt>0&&dt<=kMaximumHistorySeconds&&
        previous->points[0]>=0&&previous->points[1]>=0&&
        valid_pose(previous->fighter[0])&&valid_pose(previous->fighter[1]);
    if(history&&(now.points[0]<previous->points[0]||now.points[1]<previous->points[1]))return kCounterRegressed;
    for(int i=0;i<kFeatures;i++)out[i]=0.f;
    float actor_heading=0;bool actor_heading_available=false;
    for(int relative=0;relative<2;relative++){
        const int slot=now.actor_slot^relative,base=relative*86;
        const Pose& pose=now.fighter[slot];float q[4],theta=0;
        normalize_quaternion(pose.root_wxyz,q);const bool heading_available=heading(q,theta);
        if(relative==0){actor_heading=theta;actor_heading_available=heading_available;}
        for(int i=0;i<3;i++)out[base+i]=pose.root_xyz[i];
        for(int i=0;i<4;i++)out[base+3+i]=q[i];
        out[base+71]=heading_available?1.f:0.f;
        out[base+72]=tilt_fraction(q);out[base+73]=pose.root_xyz[2];
        out[base+74]=float(pose.joint_pose_available);
        if(pose.joint_pose_available)for(int i=0;i<29;i++)out[base+13+i]=wrap_angle(pose.projected_joints[i]);
        if(history){
            const Pose& old=previous->fighter[slot];
            const double vx=(double(pose.root_xyz[0])-old.root_xyz[0])/dt;
            const double vy=(double(pose.root_xyz[1])-old.root_xyz[1])/dt;
            if(heading_available){
                out[base+7]=float(cos(double(theta))*vx+sin(double(theta))*vy);
                out[base+8]=float(-sin(double(theta))*vx+cos(double(theta))*vy);
            }
            out[base+9]=float((double(pose.root_xyz[2])-old.root_xyz[2])/dt);
            float old_q[4],old_theta=0;normalize_quaternion(old.root_wxyz,old_q);
            const bool yaw_rate_available=heading_available&&heading(old_q,old_theta);
            out[base+76]=yaw_rate_available?1.f:0.f;
            if(yaw_rate_available)out[base+12]=float(remainder(double(theta)-old_theta,2*kPi)/dt);
            const bool joint_rate_available=pose.joint_pose_available&&old.joint_pose_available;
            out[base+75]=joint_rate_available?1.f:0.f;
            if(joint_rate_available)for(int i=0;i<29;i++)
                out[base+42+i]=float(remainder(double(pose.projected_joints[i])-old.projected_joints[i],2*kPi)/dt);
        }
    }
    const Pose& actor=now.fighter[now.actor_slot];const Pose& opponent=now.fighter[now.actor_slot^1];
    const double dx=double(opponent.root_xyz[0])-actor.root_xyz[0],dy=double(opponent.root_xyz[1])-actor.root_xyz[1];
    const double distance=sqrt(dx*dx+dy*dy);out[86]=float(distance);
    out[87]=actor_heading_available&&distance>0?float(remainder(atan2(dy,dx)-actor_heading,2*kPi)/kPi):0.f;
    if(actor_heading_available){out[172]=float(cos(double(actor_heading)/2));out[175]=float(sin(double(actor_heading)/2));}
    out[184]=float(now.actor_slot);out[185]=now.terminal?4.f:now.round_active?2.f:0.f;
    out[188]=now.round_duration_seconds/120.f;out[189]=now.round_remaining_seconds/120.f;
    out[202]=float(now.referee_available);out[203]=history?1.f:0.f;
    for(int relative=0;relative<2;relative++){
        const int slot=now.actor_slot^relative;out[190+relative]=float(now.points[slot]);
        if(now.referee_available)out[204+relative]=float((now.count_mask>>slot)&1u);
        if(history)out[217+relative]=float(now.points[slot]-previous->points[slot]);
    }
    for(int i=0;i<kFeatures;i++)if(!isfinite(out[i]))return kNonfiniteOutput;
    return kOk;
}
}
#undef REK_OBSERVABLE_BALANCE_FN
#endif
