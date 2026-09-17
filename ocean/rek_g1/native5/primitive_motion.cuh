#pragma once
#include "primitive_contacts.cuh"
#include <math.h>

// Rigid-pose interpolation between baked frames. This is a temporal sampling
// approximation, not continuous collision detection or an articulated solve.
#if defined(__CUDACC__)
#define REK_PRIMITIVE_MOTION_HD __host__ __device__ inline
#else
#define REK_PRIMITIVE_MOTION_HD inline
#endif
namespace rek5_primitive {
REK_PRIMITIVE_MOTION_HD void matrix_quaternion(const float* m,float* q) {
    float trace=m[0]+m[4]+m[8];
    if(trace>0){
        float s=2*sqrtf(trace+1);q[0]=.25f*s;
        q[1]=(m[7]-m[5])/s;q[2]=(m[2]-m[6])/s;q[3]=(m[3]-m[1])/s;
    }else{
        int i=m[0]>m[4]?(m[0]>m[8]?0:2):(m[4]>m[8]?1:2);
        int j=(i+1)%3,k=(i+2)%3;
        float s=2*sqrtf(fmaxf(0,1+m[3*i+i]-m[3*j+j]-m[3*k+k]));
        q[i+1]=.25f*s;q[0]=(m[3*k+j]-m[3*j+k])/s;
        q[j+1]=(m[3*j+i]+m[3*i+j])/s;q[k+1]=(m[3*k+i]+m[3*i+k])/s;
    }
}
REK_PRIMITIVE_MOTION_HD void quaternion_matrix(const float* q,float* m) {
    float w=q[0],x=q[1],y=q[2],z=q[3];
    m[0]=1-2*(y*y+z*z);m[1]=2*(x*y-w*z);m[2]=2*(x*z+w*y);
    m[3]=2*(x*y+w*z);m[4]=1-2*(x*x+z*z);m[5]=2*(y*z-w*x);
    m[6]=2*(x*z-w*y);m[7]=2*(y*z+w*x);m[8]=1-2*(x*x+y*y);
}
REK_PRIMITIVE_MOTION_HD Shape world_shape(const Shape& local,float x,float y,float yaw) {
    Shape out=local;float s=sinf(yaw),c=cosf(yaw);
    out.center[0]=x+c*local.center[0]-s*local.center[1];
    out.center[1]=y+s*local.center[0]+c*local.center[1];
    for(int k=0;k<3;k++){
        out.axes[k]=c*local.axes[k]-s*local.axes[3+k];
        out.axes[3+k]=s*local.axes[k]+c*local.axes[3+k];
    }
    return out;
}
REK_PRIMITIVE_MOTION_HD Shape interpolate_shape(const Shape& a,const Shape& b,float t) {
    if(t<=0)return a;
    if(t>=1)return b;
    Shape out=b;
    for(int k=0;k<3;k++)out.center[k]=a.center[k]+t*(b.center[k]-a.center[k]);
    if(out.kind==Sphere)return out;
    float qa[4],qb[4],q[4],dot=0,norm=0;
    matrix_quaternion(a.axes,qa);matrix_quaternion(b.axes,qb);
    for(int k=0;k<4;k++)dot+=qa[k]*qb[k];
    float sign=dot<0?-1.f:1.f;
    for(int k=0;k<4;k++){q[k]=(1-t)*qa[k]+t*sign*qb[k];norm+=q[k]*q[k];}
    float inv=1/sqrtf(norm);for(int k=0;k<4;k++)q[k]*=inv;
    quaternion_matrix(q,out.axes);return out;
}
REK_PRIMITIVE_MOTION_HD bool sampled_overlap(const Shape& old_a,const Shape& a,
        const Shape& old_b,const Shape& b,int samples) {
    // Exclude t=0, already tested on the preceding tick. Include t=1 exactly.
    for(int i=1;i<=samples;i++){
        float t=float(i)/samples;
        if(overlap(interpolate_shape(old_a,a,t),interpolate_shape(old_b,b,t)))return true;
    }
    return false;
}
}
#undef REK_PRIMITIVE_MOTION_HD
