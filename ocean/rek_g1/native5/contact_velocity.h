#pragma once
#include <cmath>
#include <cstring>
#include <stdexcept>

// Optional kinematic data, separate from FastFrame so the legacy GPU frame
// layout and allocation remain unchanged. Units: metres and metres/second.
struct FastBodyVelocityFrame {
    float linear[2][14][3];
    float root_com[2][3];
};
static_assert(sizeof(FastBodyVelocityFrame)==360,"body velocity frame layout changed");

namespace rek_contact_velocity {
constexpr int Bodies=14;
enum class Mode { LegacySphereProxy=0, BodyCvel=1 };
inline Mode parse(const char* value) {
    if(!value||!std::strcmp(value,"legacy_sphere_proxy_v1"))return Mode::LegacySphereProxy;
    if(!std::strcmp(value,"body_cvel_v1"))return Mode::BodyCvel;
    throw std::invalid_argument("REK_FAST_CONTACT_VELOCITY_must_be_legacy_sphere_proxy_v1_or_body_cvel_v1");
}
inline const char* name(Mode mode) {return mode==Mode::BodyCvel?"body_cvel_v1":"legacy_sphere_proxy_v1";}
inline bool compatible(Mode mode,bool geom_pairs,bool primitive_geometry,int recovered_scoring) {
    return mode==Mode::LegacySphereProxy||(geom_pairs&&primitive_geometry&&recovered_scoring==2);
}
inline constexpr const char* BodyNames[Bodies]={
    "left_ankle_roll_link_3045","right_ankle_roll_link_3090",
    "left_wrist_yaw_link_3467","right_wrist_yaw_link_3293",
    "left_knee_link_3106","right_knee_link_3429",
    "pelvis_3266","torso_link_3347",
    "left_hip_pitch_link_3457","left_hip_roll_link_3425","left_hip_yaw_link_2943",
    "right_hip_pitch_link_3469","right_hip_roll_link_3345","right_hip_yaw_link_3191"
};
#ifdef __CUDACC__
#define REK_CVEL_HD __host__ __device__ inline
#else
#define REK_CVEL_HD inline
#endif
REK_CVEL_HD int limb_slot(int limb) {return limb;}
REK_CVEL_HD int target_slot(int target) {return target==0?6:target<3?7:target+5;}
struct Linear {float x,y,z;};
// End-of-tick velocity approximation. C is the canonical root-subtree COM,
// not a body/geom centre. The baked quaternion tilt is already incorporated.
// An identical old/current frame has zero clip rate, including held endpoints
// and reconciled route starts, but modeled external root motion still applies.
REK_CVEL_HD Linear compose(const FastBodyVelocityFrame& frame,int side,int body,
        float yaw,float vx,float vy,float omega,bool moving_clip) {
    const float c=cosf(yaw),s=sinf(yaw);const float* com=frame.root_com[side];
    const float cx=c*com[0]-s*com[1],cy=s*com[0]+c*com[1];
    const float* l=frame.linear[side][body];
    const float lx=moving_clip?l[0]:0,ly=moving_clip?l[1]:0,lz=moving_clip?l[2]:0;
    return {c*lx-s*ly+vx-omega*cy,s*lx+c*ly+vy+omega*cx,lz};
}
REK_CVEL_HD float relative_speed(Linear a,Linear b) {
    const float dx=a.x-b.x,dy=a.y-b.y,dz=a.z-b.z;
    return sqrtf(dx*dx+dy*dy+dz*dz);
}
#undef REK_CVEL_HD
}
