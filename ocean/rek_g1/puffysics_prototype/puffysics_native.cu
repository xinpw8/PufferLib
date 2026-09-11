#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>

struct RpParameters { float joint[64][22]; float root[2][12]; int nj; int nr; };
static RpParameters rp_host_parameters;
__constant__ RpParameters rp_device_parameters;
__host__ __device__ static float rp_armature(int joint) {
    if (joint < 0) return 0;
#ifdef __CUDA_ARCH__
    return rp_device_parameters.joint[joint][19];
#else
    return rp_host_parameters.joint[joint][19];
#endif
}
#define B3_ART_JOINT_ARMATURE(joint_id) rp_armature(joint_id)
#define B3_ART_TORSION 0
#define B3_MAX_JOINTS 64
#define B3_MUJOCO_COLLISION_FILTER 1
#include "engine/puffysics.cuh"

struct RpHandle { B3World* worlds; B3World* initial; int* stats; int arenas; int mode; };
static char rp_error[512];
static bool rp_live = false;
static bool rp_check(cudaError_t err, const char* where) {
    if (err == cudaSuccess) return true;
    std::snprintf(rp_error, sizeof(rp_error), "%s: %s", where, cudaGetErrorString(err));
    return false;
}
extern "C" const char* rp_last_error() { return rp_error; }
extern "C" int rp_cache_enabled() {
#ifdef RP_USE_ART_CACHE
    return 1;
#else
    return 0;
#endif
}
static B3Vec3 rp_vec(const float* p) { return b3_v(p[0], p[1], p[2]); }
static B3Quat rp_quat(const float* p) { return b3_q(p[0], p[1], p[2], p[3]); }

__device__ static float rp_joint_angle(B3World* w, int joint) {
    const float* p = rp_device_parameters.joint[joint];
    const float raw = b3_joint_angle(w, joint);
    const float midpoint = 0.5f * (p[14] + p[15]);
    // Every exported limited interval is narrower than 2*pi. Select its branch.
    return raw + 2.0f*B3_PI*nearbyintf((midpoint-raw)/(2.0f*B3_PI));
}

__device__ static void rp_gather(B3World* w, int arena, float* qpos,
        float* qvel, float* base, float* angular, int* stats) {
    float* qp = qpos + arena * 72;
    float* qv = qvel + arena * 70;
    for (int j = 0; j < rp_device_parameters.nj; ++j) {
        const float* p = rp_device_parameters.joint[j];
        qp[int(p[2])] = p[13] + rp_joint_angle(w, j);
        qv[int(p[3])] = b3_joint_speed(w, j);
    }
    for (int r = 0; r < 2; ++r) {
        const float* p = rp_device_parameters.root[r];
        const B3Body* b = &w->bodies[int(p[0]) + 1];
        B3Vec3 offset = b3_rotate(b->rotation, b3_v(p[3], p[4], p[5]));
        B3Vec3 origin = b3_add(b->position, offset);
        B3Quat rotation = b3_qnorm(b3_qmul(b->rotation, b3_q(p[6], p[7], p[8], p[9])));
        B3Vec3 velocity = b3_add(b->lin_vel, b3_cross(b->ang_vel, offset));
        B3Vec3 local_omega = b3_rotate(b3_qconj(rotation), b->ang_vel);
        B3Vec3 inertial_omega = b3_rotate(b3_qconj(b->rotation), b->ang_vel);
        int qi = int(p[1]), vi = int(p[2]), row = arena * 2 + r;
        qp[qi] = origin.x; qp[qi+1] = origin.y; qp[qi+2] = origin.z;
        qp[qi+3] = rotation.s; qp[qi+4] = rotation.v.x;
        qp[qi+5] = rotation.v.y; qp[qi+6] = rotation.v.z;
        qv[vi] = velocity.x; qv[vi+1] = velocity.y; qv[vi+2] = velocity.z;
        qv[vi+3] = local_omega.x; qv[vi+4] = local_omega.y; qv[vi+5] = local_omega.z;
        for (int k = 0; k < 4; ++k) base[row*4+k] = qp[qi+3+k];
        angular[row*3] = inertial_omega.x;
        angular[row*3+1] = inertial_omega.y;
        angular[row*3+2] = inertial_omega.z;
    }
    for (int k = 0; k < 72; ++k) if (!isfinite(qp[k])) stats[arena*4+1] = 1;
    for (int k = 0; k < 70; ++k) if (!isfinite(qv[k])) stats[arena*4+1] = 1;
    stats[arena*4] = max(stats[arena*4], w->contact_count);
}

__device__ static void rp_forces(B3World* w, const float* ctrl) {
    for (int b = 0; b < w->body_count; ++b) {
        w->bodies[b].force = b3_v(0,0,0);
        w->bodies[b].torque = b3_v(0,0,0);
    }
    for (int j = 0; j < rp_device_parameters.nj; ++j) {
        const float* p = rp_device_parameters.joint[j];
        B3Joint* joint = &w->joints[j];
        float angle = p[13] + rp_joint_angle(w, j), velocity = b3_joint_speed(w, j);
        float force = b3_clamp(p[16]*(ctrl[j]-angle)-p[17]*velocity, -p[18], p[18]);
        force -= p[20] * velocity;
        // Moving Coulomb friction only. Static friction remains an explicit gap.
        if (velocity != 0) force -= copysignf(p[21], velocity);
        B3Body *a = &w->bodies[joint->body_a], *b = &w->bodies[joint->body_b];
        B3Vec3 axis = b3_rotate(b3_qmul(a->rotation, joint->local_rot_a), b3_v(0,0,1));
        B3Vec3 torque = b3_mul(axis, force);
        a->torque = b3_sub(a->torque, torque);
        b->torque = b3_add(b->torque, torque);
    }
}

// Reduced-coordinate diagnostic preserves rotor inertia in the ABA operator.
// The contact solver and predictive joint stops are not MuJoCo-equivalent.
__device__ static bool rp_art_step(B3World* w, int* stats) {
    B3Art art;
    if (!b3_art_bind(&art, w)) { stats[2] = 1 << 24; return false; }
    constexpr float h = 0.002f, inv_h = 500.0f;
    float hh, ih, id;
    B3Soft cs, ss;
    b3_soft_step_params(w, h, 1, &hh, &ih, &id, &cs, &ss);
    b3_find_contacts(w);
    b3_prepare_contacts(w, cs, ss);
    b3_warm_start(w);
    b3_art_integrate_vel(&art, w, h);
    b3_art_solve_contacts(&art, w, inv_h, w->contact_speed, 1, B3_ART_CONTACT_ITERS);
    for (int j = 0; j < rp_device_parameters.nj; ++j) {
        const float* p = rp_device_parameters.joint[j];
        B3Joint* joint = &w->joints[j];
        float angle = rp_joint_angle(w, j), velocity = b3_joint_speed(w, j);
        float target = b3_clamp(velocity, (p[14]-angle)*inv_h, (p[15]-angle)*inv_h);
        if (target == velocity) continue;
        B3Vec3 axis = b3_rotate(b3_qmul(w->bodies[joint->body_a].rotation,
            joint->local_rot_a), b3_v(0,0,1));
        B3ArtRow row;
        b3_art_make_row(&art, joint->body_a, joint->body_b, b3_v(0,0,0), b3_v(0,0,0), axis, &row);
        row.torque = 1;
        float response = b3_art_response_w(&art, w, &row);
        if (!(response > 0)) { stats[2] = 1 << 25; return false; }
        b3_art_apply_impulse(&art, w, &row, (target-velocity)/response);
        stats[3] += 1;
    }
    b3_art_integrate_pos(&art, w, h, inv_h);
    b3_art_solve_contacts(&art, w, inv_h, w->contact_speed, 0, B3_ART_CONTACT_ITERS);
    b3_finalize_transforms(w);
    return true;
}

__global__ void rp_reset_kernel(RpHandle h, float* qp, float* qv,
        float* base, float* angular, float* time) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= h.arenas) return;
    h.worlds[i] = *h.initial;
    for (int k = 0; k < 4; ++k) h.stats[i*4+k] = 0;
    time[i] = 0;
    rp_gather(h.worlds+i, i, qp, qv, base, angular, h.stats);
}
__global__ void rp_step_kernel(RpHandle h, const float* ctrl, float* qp,
        float* qv, float* base, float* angular, float* time) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= h.arenas || h.stats[i*4+1] || h.stats[i*4+2]) return;
    B3World* w = h.worlds+i;
    rp_forces(w, ctrl+i*58);
    if (h.mode == 1) {
        if (!rp_art_step(w, h.stats+i*4)) return;
    } else b3_step(w, 0.002f, 1);
    h.stats[i*4+2] |= w->collision_status;
    time[i] += 0.002f;
    rp_gather(w, i, qp, qv, base, angular, h.stats);
}

extern "C" void* rp_create(int arenas, const float* bodies, int nb,
        const float* shapes, int ns, const float* joints, int nj,
        const float* roots, int nr, int mode) {
    if (rp_live || arenas < 1 || nb != 60 || ns != 91 || nj != 58 || nr != 2 || (mode != 0 && mode != 1)) {
        std::snprintf(rp_error, sizeof(rp_error), "invalid model dimensions/mode or concurrent handle"); return nullptr;
    }
    auto* initial = new B3World;
    b3_world_init(initial);
    initial->gravity = b3_v(0,0,-9.8100004196167f);
    B3BodyDef static_body = b3_default_body();
    static_body.type = B3_STATIC;
    b3_create_body(initial, &static_body);
    for (int i = 0; i < nb; ++i) {
        const float* p = bodies+i*11;
        B3BodyDef body = b3_default_body(); body.type = B3_DYNAMIC;
        body.position = rp_vec(p); body.rotation = rp_quat(p+3);
        int id = b3_create_body(initial, &body);
        b3_set_inertial(initial, id, p[7], b3_v(0,0,0), rp_vec(p+8));
    }
    for (int i = 0; i < ns; ++i) {
        const float* p = shapes+i*17;
        B3ShapeDef shape = b3_default_shape();
        shape.density = 0; shape.friction = p[12];
        shape.category = uint64_t(p[13]); shape.mask = uint64_t(p[14]);
        shape.restitution = p[15]; shape.rolling = p[16];
        int type = int(p[1]);
        B3Vec3 half = type == B3_BOX ? rp_vec(p+9) : b3_v(0,p[10],0);
        b3_add_shape(initial, int(p[0])+1, type, rp_vec(p+2), rp_quat(p+5), p[9], half, &shape);
    }
    rp_host_parameters.nj = nj; rp_host_parameters.nr = nr;
    std::memcpy(rp_host_parameters.joint, joints, nj*22*sizeof(float));
    std::memcpy(rp_host_parameters.root, roots, nr*12*sizeof(float));
    for (int i = 0; i < nj; ++i) {
        const float* p = joints+i*22;
        int j = b3_create_revolute(initial, int(p[0])+1, int(p[1])+1,
            rp_vec(p+4), rp_vec(p+7), rp_vec(p+10));
        b3_joint_enable_limit(initial, j, 1);
        // Preserve the exported relative interval without the public +/-pi clamp.
        initial->joints[j].lower_angle = p[14];
        initial->joints[j].upper_angle = p[15];
    }
    RpHandle* h = new RpHandle{}; h->arenas = arenas; h->mode = mode;
    bool ok = rp_check(cudaMalloc(&h->worlds, arenas*sizeof(B3World)), "allocate worlds") &&
        rp_check(cudaMalloc(&h->initial, sizeof(B3World)), "allocate initial") &&
        rp_check(cudaMalloc(&h->stats, arenas*4*sizeof(int)), "allocate stats") &&
        rp_check(cudaMemcpy(h->initial, initial, sizeof(B3World), cudaMemcpyHostToDevice), "upload initial") &&
        rp_check(cudaMemcpyToSymbol(rp_device_parameters, &rp_host_parameters, sizeof(rp_host_parameters)), "upload parameters");
    delete initial;
    if (!ok) { cudaFree(h->worlds); cudaFree(h->initial); cudaFree(h->stats); delete h; return nullptr; }
    rp_live = true;
    return h;
}
extern "C" int rp_reset(void* handle, float* qp, float* qv, float* base,
        float* angular, float* time, void* stream) {
    RpHandle h = *static_cast<RpHandle*>(handle);
    rp_reset_kernel<<<(h.arenas+31)/32,32,0,static_cast<cudaStream_t>(stream)>>>(h,qp,qv,base,angular,time);
    return rp_check(cudaGetLastError(), "reset kernel") ? 0 : 1;
}
extern "C" int rp_step(void* handle, const float* ctrl, float* qp, float* qv,
        float* base, float* angular, float* time, void* stream) {
    RpHandle h = *static_cast<RpHandle*>(handle);
    rp_step_kernel<<<(h.arenas+31)/32,32,0,static_cast<cudaStream_t>(stream)>>>(h,ctrl,qp,qv,base,angular,time);
    return rp_check(cudaGetLastError(), "step kernel") ? 0 : 1;
}
extern "C" int rp_get_stats(void* handle, int* output) {
    RpHandle* h = static_cast<RpHandle*>(handle);
    return rp_check(cudaMemcpy(output,h->stats,h->arenas*4*sizeof(int),cudaMemcpyDeviceToHost),"download stats") ? 0 : 1;
}
extern "C" int rp_get_failure(void* handle, int arena, int* meta, float* data) {
    RpHandle* h = static_cast<RpHandle*>(handle);
    if (arena < 0 || arena >= h->arenas) return 1;
    bool ok = rp_check(cudaMemcpy(meta, h->worlds[arena].collision_failure_meta,
        7*sizeof(int), cudaMemcpyDeviceToHost), "download failure metadata") &&
        rp_check(cudaMemcpy(data, h->worlds[arena].collision_failure_data,
        36*sizeof(float), cudaMemcpyDeviceToHost), "download failure operands");
    return ok ? 0 : 1;
}
extern "C" void rp_destroy(void* handle) {
    if (!handle) return;
    auto* h = static_cast<RpHandle*>(handle);
    cudaFree(h->worlds); cudaFree(h->initial); cudaFree(h->stats); delete h; rp_live = false;
}
