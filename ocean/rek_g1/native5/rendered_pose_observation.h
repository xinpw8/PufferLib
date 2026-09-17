#pragma once
#include <math.h>

#if defined(__CUDACC__)
#define REK_RENDERED_POSE_FN __host__ __device__ inline
#else
#define REK_RENDERED_POSE_FN inline
#endif

// Optional rendered_pose_v1 observation math. These functions do not modify
// dynamics, logical command heading, collision history, masks, or rewards.
// Inputs are the exported MuJoCo root pose and projected joint angles. The
// caller must retain the actual previous observation pose, independently of
// old_route/old_phase used by collision sweeps. Reset history to the current
// pose. A nonpositive dt produces zero derivatives, as encode_live.cpp does.
namespace rek_rendered_pose {
constexpr double pi=3.14159265358979323846;
constexpr const char* mode_name="rendered_pose_v1";

REK_RENDERED_POSE_FN void compose_root(float logical_yaw,const float* clip_wxyz,float* world_wxyz){
    const float s=sinf(.5f*logical_yaw),c=cosf(.5f*logical_yaw);
    world_wxyz[0]=c*clip_wxyz[0]-s*clip_wxyz[3];
    world_wxyz[1]=c*clip_wxyz[1]-s*clip_wxyz[2];
    world_wxyz[2]=c*clip_wxyz[2]+s*clip_wxyz[1];
    world_wxyz[3]=c*clip_wxyz[3]+s*clip_wxyz[0];
}

REK_RENDERED_POSE_FN float heading(const float* world_wxyz){
    const float w=world_wxyz[0],x=world_wxyz[1],y=world_wxyz[2],z=world_wxyz[3];
    // Homogeneous form of the normalized-quaternion yaw used by encode_live.
    // It also removes small float32 root-quaternion norm drift without a sqrt.
    return atan2f(2.f*(w*z+x*y),w*w+x*x-y*y-z*z);
}

REK_RENDERED_POSE_FN void heading_quaternion(float rendered_heading,float* wxyz){
    wxyz[0]=cosf(.5f*rendered_heading);wxyz[1]=0.f;wxyz[2]=0.f;
    wxyz[3]=sinf(.5f*rendered_heading);
}

REK_RENDERED_POSE_FN float angular_rate(float current,float previous,float dt){
    // Double subtraction and the same 2*pi divisor as the live encoder avoid
    // float32 wrap-period drift at +/-pi. Returned policy features are float32.
    return dt>0.f?float(remainder(double(current)-double(previous),2.0*pi)/double(dt)):0.f;
}

REK_RENDERED_POSE_FN float joint_rate(float current,float previous,float dt){
    return angular_rate(current,previous,dt);
}

REK_RENDERED_POSE_FN void local_velocity(float x,float y,float old_x,float old_y,
        float rendered_heading,float dt,float* local_xy){
    const float vx=dt>0.f?float((double(x)-double(old_x))/double(dt)):0.f;
    const float vy=dt>0.f?float((double(y)-double(old_y))/double(dt)):0.f;
    const float c=cosf(rendered_heading),s=sinf(rendered_heading);
    local_xy[0]=c*vx+s*vy;local_xy[1]=-s*vx+c*vy;
}

REK_RENDERED_POSE_FN float bearing(float actor_x,float actor_y,float opponent_x,
        float opponent_y,float rendered_heading){
    const double world_bearing=atan2(double(opponent_y)-double(actor_y),double(opponent_x)-double(actor_x));
    return float(remainder(world_bearing-double(rendered_heading),2.0*pi)/pi);
}

REK_RENDERED_POSE_FN float weighted_delta_total(int actor_delta,int opponent_delta){
    // Native RoundState.CleanHits contains weighted points. This observation
    // total cannot distinguish one two-point kick from two one-point punches.
    return float(actor_delta+opponent_delta);
}
}

#undef REK_RENDERED_POSE_FN
