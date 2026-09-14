// Experimental semantic ABI over the pinned Puffysics prototype solver.
// Including the original translation unit keeps force laws and stepping intact.
#include "puffysics_native.cu"

struct RpsDescriptor {
    int arenas, bodies, geoms, capacity;
    float *qpos, *qvel, *base, *angular, *time;
    float *xpos, *xquat, *xmat, *xipos, *ximat, *com, *cvel;
    float *geom_xpos, *geom_xmat;
    int *contact_geom, *contact_world, *nacon;
    float *contact_dist, *contact_pos, *contact_frame;
    int *counts, *offsets;
    float *body_map, *geom_map, *pre_centers;
};

__host__ __device__ static B3Vec3 rps_v(const float* p) { return b3_v(p[0], p[1], p[2]); }
__host__ __device__ static B3Quat rps_q(const float* p) { return b3_q(p[0], p[1], p[2], p[3]); }
__host__ __device__ static void rps_store(float* p, B3Vec3 v) { p[0]=v.x; p[1]=v.y; p[2]=v.z; }
__host__ __device__ static void rps_matrix(float* p, B3Quat q) {
    B3Vec3 x=b3_rotate(q,b3_v(1,0,0)), y=b3_rotate(q,b3_v(0,1,0)), z=b3_rotate(q,b3_v(0,0,1));
    p[0]=x.x; p[1]=y.x; p[2]=z.x;
    p[3]=x.y; p[4]=y.y; p[5]=z.y;
    p[6]=x.z; p[7]=y.z; p[8]=z.z;
}

// Export native body state in the source model's frames. Spatial velocity uses
// MuJoCo's root-subtree COM origin so unchanged observation/hit readers apply.
__host__ __device__ __forceinline__ static void rps_body_fields(B3World* w, int a, RpsDescriptor d) {
    float* xpos=d.xpos+a*d.bodies*3;
    float* xipos=d.xipos+a*d.bodies*3;
    float* com=d.com+a*d.bodies*3;
    for (int b=0;b<d.bodies;++b) {
        const float* m=d.body_map+b*18;
        const B3Body& rigid=w->bodies[int(m[0])];
        B3Vec3 p=b3_add(rigid.position,b3_rotate(rigid.rotation,rps_v(m+4)));
        B3Quat q=b3_qnorm(b3_qmul(rigid.rotation,rps_q(m+7)));
        B3Vec3 ip=b3_add(rigid.position,b3_rotate(rigid.rotation,rps_v(m+11)));
        B3Quat iq=b3_qnorm(b3_qmul(rigid.rotation,rps_q(m+14)));
        rps_store(xpos+b*3,p); rps_store(xipos+b*3,ip);
        float* quat=d.xquat+(a*d.bodies+b)*4;
        quat[0]=q.s; quat[1]=q.v.x; quat[2]=q.v.y; quat[3]=q.v.z;
        rps_matrix(d.xmat+(a*d.bodies+b)*9,q);
        rps_matrix(d.ximat+(a*d.bodies+b)*9,iq);
        rps_store(com+b*3,b3_mul(ip,m[3]));
    }
    // Every source parent precedes its descendants in the compiled model.
    float mass[64];
    for (int b=0;b<d.bodies;++b) mass[b]=d.body_map[b*18+3];
    for (int b=d.bodies-1;b>0;--b) {
        int p=int(d.body_map[b*18+1]);
        rps_store(com+p*3,b3_add(rps_v(com+p*3),rps_v(com+b*3)));
        mass[p]+=mass[b];
    }
    for (int b=0;b<d.bodies;++b) {
        rps_store(com+b*3,mass[b]>0 ? b3_mul(rps_v(com+b*3),1.0f/mass[b]) : rps_v(xipos+b*3));
    }
    for (int b=0;b<d.bodies;++b) {
        const float* m=d.body_map+b*18;
        const B3Body& rigid=w->bodies[int(m[0])];
        B3Vec3 offset=b3_sub(rps_v(xipos+b*3),rigid.center);
        B3Vec3 linear=b3_add(rigid.lin_vel,b3_cross(rigid.ang_vel,offset));
        B3Vec3 lever=b3_sub(rps_v(xipos+b*3),rps_v(com+int(m[2])*3));
        float* cv=d.cvel+(a*d.bodies+b)*6;
        rps_store(cv,rigid.ang_vel);
        rps_store(cv+3,b3_sub(linear,b3_cross(rigid.ang_vel,lever)));
    }
    for (int g=0;g<d.geoms;++g) {
        const float* m=d.geom_map+g*8;
        const B3Body& rigid=w->bodies[int(m[0])];
        rps_store(d.geom_xpos+(a*d.geoms+g)*3,
            b3_add(rigid.position,b3_rotate(rigid.rotation,rps_v(m+1))));
        rps_matrix(d.geom_xmat+(a*d.geoms+g)*9,b3_qnorm(b3_qmul(rigid.rotation,rps_q(m+4))));
    }
    int count=0;
    for (int c=0;c<w->contact_count;++c) count+=w->contacts[c].point_count;
    d.counts[a]=count;
}

__global__ static void rps_body_kernel(RpHandle h, RpsDescriptor d) {
    int a=blockIdx.x*blockDim.x+threadIdx.x;
    if (a<h.arenas) rps_body_fields(h.worlds+a,a,d);
}
__global__ static void rps_offsets_kernel(RpsDescriptor d) {
    int total=0;
    for (int a=0;a<d.arenas;++a) { d.offsets[a]=total; total+=d.counts[a]; }
    *d.nacon=total;
}
__host__ __device__ __forceinline__ static void rps_contacts_one(RpHandle h,RpsDescriptor d,int a) {
    B3World* w=h.worlds+a;
    int slot=d.offsets[a];
    const float* centers=d.pre_centers+a*61*3;
    for (int ci=0;ci<w->contact_count;++ci) {
        const B3Contact& c=w->contacts[ci];
        for (int p=0;p<c.point_count;++p,++slot) {
            if (slot>=d.capacity) continue; // nacon preserves overflow for validation.
            const B3Point& point=c.points[p];
            B3Vec3 pa=b3_add(rps_v(centers+c.body_a*3),point.r_a);
            B3Vec3 pb=b3_add(rps_v(centers+c.body_b*3),point.r_b);
            d.contact_geom[slot*2]=c.shape_a; d.contact_geom[slot*2+1]=c.shape_b;
            d.contact_world[slot]=a;
            d.contact_dist[slot]=point.base_sep+b3_dot(b3_sub(point.r_b,point.r_a),c.normal);
            rps_store(d.contact_pos+slot*3,b3_mul(b3_add(pa,pb),0.5f));
            rps_store(d.contact_frame+slot*9,c.normal);
            rps_store(d.contact_frame+slot*9+3,c.tangent1);
            // MuJoCo frame is right handed: t2 = normal cross t1.
            rps_store(d.contact_frame+slot*9+6,b3_cross(c.normal,c.tangent1));
        }
    }
}
__global__ static void rps_contacts_kernel(RpHandle h, RpsDescriptor d) {
    int a=blockIdx.x*blockDim.x+threadIdx.x;
    if(a<h.arenas)rps_contacts_one(h,d,a);
}

__global__ static void rps_before_kernel(RpHandle h,RpsDescriptor d) {
    int a=blockIdx.x*blockDim.x+threadIdx.x;
    if (a>=h.arenas) return;
    for (int b=0;b<61;++b) rps_store(d.pre_centers+(a*61+b)*3,h.worlds[a].bodies[b].center);
}

// Reconstruct selected worlds from generalized state. No integration occurs.
// Copying the initial world clears solver/contact warm starts only in reset
// worlds. Cumulative failure counters in h.stats are deliberately preserved.
__host__ __device__ __forceinline__ static void rps_forward_one(RpHandle h,RpsDescriptor d,int a) {
    B3World* w=h.worlds+a;
    *w=*h.initial;
    const float* qp=d.qpos+a*72;
    const float* qv=d.qvel+a*70;
    for (int r=0;r<2;++r) {
        const float* p=rp_parameters().root[r];
        B3Body& b=w->bodies[int(p[0])+1];
        int qi=int(p[1]),vi=int(p[2]);
        B3Quat root=b3_q(qp[qi+4],qp[qi+5],qp[qi+6],qp[qi+3]);
        b.rotation=b3_qnorm(b3_qmul(root,b3_qconj(rps_q(p+6))));
        B3Vec3 offset=b3_rotate(b.rotation,rps_v(p+3));
        b.position=b3_sub(rps_v(qp+qi),offset);
        b.ang_vel=b3_rotate(root,rps_v(qv+vi+3));
        b.lin_vel=b3_sub(rps_v(qv+vi),b3_cross(b.ang_vel,offset));
    }
    for (int j=0;j<58;++j) {
        const float* p=rp_parameters().joint[j];
        const B3Joint& joint=w->joints[j];
        const B3Body& pa=w->bodies[joint.body_a];
        B3Body& child=w->bodies[joint.body_b];
        // At relative hinge angle zero these two constraint frames coincide.
        B3Quat turn=b3_q_axis_angle(b3_v(0,0,1),qp[int(p[2])]-p[13]);
        child.rotation=b3_qnorm(b3_qmul(b3_qmul(b3_qmul(pa.rotation,joint.local_rot_a),turn),b3_qconj(joint.local_rot_b)));
        B3Vec3 ra=b3_rotate(pa.rotation,joint.local_anchor_a);
        B3Vec3 rb=b3_rotate(child.rotation,joint.local_anchor_b);
        child.position=b3_sub(b3_add(pa.position,ra),rb);
        B3Vec3 axis=b3_rotate(b3_qmul(pa.rotation,joint.local_rot_a),b3_v(0,0,1));
        child.ang_vel=b3_madd(pa.ang_vel,qv[int(p[3])],axis);
        child.lin_vel=b3_sub(b3_add(pa.lin_vel,b3_cross(pa.ang_vel,ra)),b3_cross(child.ang_vel,rb));
    }
    for (int b=0;b<61;++b) {
        B3Body& body=w->bodies[b];
        body.center=b3_add(body.position,b3_rotate(body.rotation,body.local_center));
        body.inv_i_world=b3_world_inv_i(body.rotation,body.inv_inertia);
        rps_store(d.pre_centers+(a*61+b)*3,body.center);
    }
    b3_find_contacts(w);
    h.stats[a*4+2]|=w->collision_status;
    rp_gather(w,a,d.qpos,d.qvel,d.base,d.angular,h.stats);
}
__global__ static void rps_forward_kernel(RpHandle h,RpsDescriptor d,const unsigned char* mask) {
    int a=blockIdx.x*blockDim.x+threadIdx.x;
    if(a<h.arenas && mask[a])rps_forward_one(h,d,a);
}

static int rps_export(RpHandle h,RpsDescriptor d,cudaStream_t stream) {
    rps_body_kernel<<<(h.arenas+31)/32,32,0,stream>>>(h,d);
    rps_offsets_kernel<<<1,1,0,stream>>>(d);
    rps_contacts_kernel<<<(h.arenas+31)/32,32,0,stream>>>(h,d);
    return rp_check(cudaGetLastError(),"semantic export") ? 0 : 1;
}
extern "C" size_t rps_descriptor_size() { return sizeof(RpsDescriptor); }
extern "C" int rps_refresh(void* handle,const RpsDescriptor* descriptor,void* stream) {
    RpHandle h=*static_cast<RpHandle*>(handle);
    RpsDescriptor d=*descriptor;
    auto s=static_cast<cudaStream_t>(stream);
    rps_before_kernel<<<(h.arenas+31)/32,32,0,s>>>(h,d);
    return rps_export(h,d,s);
}
extern "C" int rps_step_controls(void* handle,const RpsDescriptor* descriptor,const float* ctrl,void* stream) {
    RpHandle h=*static_cast<RpHandle*>(handle);
    RpsDescriptor d=*descriptor;
    auto s=static_cast<cudaStream_t>(stream);
    rps_before_kernel<<<(h.arenas+31)/32,32,0,s>>>(h,d);
    rp_step_kernel<<<(h.arenas+31)/32,32,0,s>>>(h,ctrl,d.qpos,d.qvel,d.base,d.angular,d.time);
    return rps_export(h,d,s);
}
extern "C" int rps_forward_selected(void* handle,const RpsDescriptor* descriptor,const unsigned char* mask,void* stream) {
    RpHandle h=*static_cast<RpHandle*>(handle);
    RpsDescriptor d=*descriptor;
    auto s=static_cast<cudaStream_t>(stream);
    rps_forward_kernel<<<(h.arenas+31)/32,32,0,s>>>(h,d,mask);
    return rps_export(h,d,s);
}
