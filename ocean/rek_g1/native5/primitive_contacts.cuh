#pragma once

#include <math.h>

#if defined(__CUDACC__)
#define REK5_PRIMITIVE_HD __host__ __device__ inline
#else
#define REK5_PRIMITIVE_HD inline
#endif

namespace rek5_primitive {

enum Kind { Sphere = 0, Capsule = 1, Box = 2 };

// Axes is an orthonormal, row-major rotation matrix with local axes in columns,
// matching MuJoCo geom_xmat. Capsule endpoints are center +/- column Z * size[1].
// Sphere/capsule radius is size[0]; box size contains its three half-extents.
// Inputs must be finite, with nonnegative sizes. Zero radii/lengths are allowed.
struct Shape {
    int kind;
    float center[3];
    float axes[9];
    float size[3];
};

namespace detail {
struct Vec { float x, y, z; };
REK5_PRIMITIVE_HD Vec make(const float* a) { return {a[0], a[1], a[2]}; }
REK5_PRIMITIVE_HD Vec add(Vec a, Vec b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
REK5_PRIMITIVE_HD Vec sub(Vec a, Vec b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
REK5_PRIMITIVE_HD Vec mul(Vec a, float b) { return {a.x*b, a.y*b, a.z*b}; }
REK5_PRIMITIVE_HD float dot(Vec a, Vec b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
REK5_PRIMITIVE_HD Vec cross(Vec a, Vec b) {
    return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}
REK5_PRIMITIVE_HD float clamp(float v, float lo, float hi) {
    return fminf(hi, fmaxf(lo, v));
}
REK5_PRIMITIVE_HD Vec axis(const Shape& s, int i) {
    return {s.axes[i], s.axes[3+i], s.axes[6+i]};
}
REK5_PRIMITIVE_HD Vec local(const Shape& s, Vec p) {
    const Vec d=sub(p, make(s.center));
    return {dot(d,axis(s,0)), dot(d,axis(s,1)), dot(d,axis(s,2))};
}
REK5_PRIMITIVE_HD float point_segment2(Vec p, Vec a, Vec b) {
    const Vec d=sub(b,a);
    const float dd=dot(d,d);
    const float t=dd>0 ? clamp(dot(sub(p,a),d)/dd,0,1) : 0;
    const Vec delta=sub(p,add(a,mul(d,t)));
    return dot(delta,delta);
}

REK5_PRIMITIVE_HD float segment_segment2(Vec a, Vec b, Vec c, Vec d) {
    // A constrained quadratic minimum is either on a boundary or at the
    // unconstrained interior closest points. Cross products avoid subtracting
    // almost equal squared dot products for nearly parallel segments.
    float best=fminf(fminf(point_segment2(a,c,d),point_segment2(b,c,d)),
                     fminf(point_segment2(c,a,b),point_segment2(d,a,b)));
    const Vec u=sub(b,a), v=sub(d,c), w=sub(a,c), n=cross(u,v);
    const float nn=dot(n,n);
    if(nn>0) {
        const float s=dot(cross(v,w),n)/nn;
        const float t=dot(cross(u,w),n)/nn;
        if(s>=0 && s<=1 && t>=0 && t<=1) {
            const Vec delta=sub(add(w,mul(u,s)),mul(v,t));
            best=fminf(best,dot(delta,delta));
        }
    }
    return best;
}

REK5_PRIMITIVE_HD float point_aabb2(Vec p, const float* half) {
    const float x=fmaxf(fabsf(p.x)-half[0],0);
    const float y=fmaxf(fabsf(p.y)-half[1],0);
    const float z=fmaxf(fabsf(p.z)-half[2],0);
    return x*x+y*y+z*z;
}

REK5_PRIMITIVE_HD float segment_aabb2(Vec a, Vec b, const float* half) {
    // Squared distance to an AABB is a convex piecewise quadratic in segment
    // parameter t. Its only breakpoints are the six slab-plane crossings.
    // Enumerate every interval and minimize its quadratic, including endpoints.
    const Vec direction=sub(b,a);
    const float p[3]={a.x,a.y,a.z};
    const float v[3]={direction.x,direction.y,direction.z};
    float cuts[8]={0,1};
    int count=2;
    for(int k=0;k<3;k++) if(v[k]!=0) {
        const float t0=(-half[k]-p[k])/v[k];
        const float t1=( half[k]-p[k])/v[k];
        if(t0>0 && t0<1) cuts[count++]=t0;
        if(t1>0 && t1<1) cuts[count++]=t1;
    }
    for(int i=1;i<count;i++) {
        const float value=cuts[i];
        int j=i;
        while(j>0 && cuts[j-1]>value) { cuts[j]=cuts[j-1]; --j; }
        cuts[j]=value;
    }
    float best=fminf(point_aabb2(a,half),point_aabb2(b,half));
    for(int i=0;i+1<count;i++) {
        const float lo=cuts[i], hi=cuts[i+1];
        if(hi<=lo) continue;
        const float mid=lo+(hi-lo)*.5f;
        float quadratic=0, linear=0;
        for(int k=0;k<3;k++) {
            const float x=p[k]+mid*v[k];
            if(x < -half[k] || x > half[k]) {
                const float offset=p[k]-(x<0 ? -half[k] : half[k]);
                quadratic+=v[k]*v[k];
                linear+=v[k]*offset;
            }
        }
        const float t=quadratic>0 ? clamp(-linear/quadratic,lo,hi) : mid;
        best=fminf(best,point_aabb2(add(a,mul(direction,t)),half));
    }
    return best;
}

REK5_PRIMITIVE_HD void endpoints(const Shape& s, Vec& a, Vec& b) {
    const Vec center=make(s.center), extent=mul(axis(s,2),s.size[1]);
    a=sub(center,extent); b=add(center,extent);
}

REK5_PRIMITIVE_HD bool separated(const Shape& a, const Shape& b, Vec n) {
    const float distance=fabsf(dot(sub(make(b.center),make(a.center)),n));
    float radius=0;
    for(int i=0;i<3;i++)
        radius+=a.size[i]*fabsf(dot(axis(a,i),n))
               +b.size[i]*fabsf(dot(axis(b,i),n));
    return distance>radius;
}
REK5_PRIMITIVE_HD bool box_box(const Shape& a, const Shape& b) {
    // Full 15-axis SAT. Unnormalized cross axes require no division and an
    // exactly zero axis is harmless. No near-parallel axis is discarded.
    for(int i=0;i<3;i++) {
        if(separated(a,b,axis(a,i)) || separated(a,b,axis(b,i))) return false;
        for(int j=0;j<3;j++) if(separated(a,b,cross(axis(a,i),axis(b,j)))) return false;
    }
    return true;
}
} // namespace detail

// Static geometric overlap, including touching. There is no speculative margin
// or positive epsilon: all final tests use <= radius^2 or strict SAT separation.
// Degeneracy checks use exact zero. Consequently a boundary within float rounding
// error may classify differently after a transform. Test invariance away from
// that boundary. This does not model sweeps, contact response, or gameplay parity.
REK5_PRIMITIVE_HD bool overlap(const Shape& a, const Shape& b) {
    using namespace detail;
    if(a.kind<0 || a.kind>Box || b.kind<0 || b.kind>Box) return false;
    // Canonical type ordering also makes mixed-type results symmetric.
    const Shape& first=a.kind<=b.kind ? a : b;
    const Shape& second=a.kind<=b.kind ? b : a;
    if(first.kind==Sphere) {
        if(second.kind==Sphere) {
            const Vec d=sub(make(first.center),make(second.center));
            const float radius=first.size[0]+second.size[0];
            return dot(d,d)<=radius*radius;
        }
        if(second.kind==Capsule) {
            Vec c,d; endpoints(second,c,d);
            const float radius=first.size[0]+second.size[0];
            return point_segment2(make(first.center),c,d)<=radius*radius;
        }
        return point_aabb2(local(second,make(first.center)),second.size)
            <=first.size[0]*first.size[0];
    }
    if(first.kind==Capsule) {
        Vec a0,a1; endpoints(first,a0,a1);
        if(second.kind==Capsule) {
            Vec b0,b1; endpoints(second,b0,b1);
            const float radius=first.size[0]+second.size[0];
            return segment_segment2(a0,a1,b0,b1)<=radius*radius;
        }
        return segment_aabb2(local(second,a0),local(second,a1),second.size)
            <=first.size[0]*first.size[0];
    }
    return box_box(first,second);
}

} // namespace rek5_primitive

#undef REK5_PRIMITIVE_HD
