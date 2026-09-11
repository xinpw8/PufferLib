// Puffysics — PufferLib physics engine (puffer + physics).
// SPDX-License-Identifier: MIT
// Grew out of a Box3D Soft Step port; the solver is no longer a port.
// Scope: kinematics + ABA/Delassus articulations + primitive contacts
// + weld/revolute joints (motor, spring, limits).
// Out of scope: CCD, sleep, meshes, sensors, events, other joints.
//
// One B3World per environment. Host: build bodies/shapes, copy worlds to
// device, launch b3_step_kernel. Device: b3_step(&worlds[env], dt, 4).
// Override B3_MAX_BODIES / B3_MAX_SHAPES / B3_MAX_CONTACTS before include.
// ABI is B3_* / b3_*. Optional modules keep their file-stem prefix:
// nbody.cuh (nbody_), nbody_rigid.cuh (b3_nbody_), ambient.h (b3_fluid_).

#pragma once

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef __CUDACC__
#define B3_HD __host__ __device__
#define B3_INL __forceinline__
#else
#define B3_HD
#define B3_INL inline
#endif

/* Defaults fit an articulated agent plus extra rigid cubes / stair boxes.
 * Override before include if a scene needs more (or less, for occupancy). */
#ifndef B3_MAX_BODIES
#define B3_MAX_BODIES 80
#endif
#ifndef B3_MAX_SHAPES
#define B3_MAX_SHAPES 128
#endif
#ifndef B3_MAX_CONTACTS
#define B3_MAX_CONTACTS 128
#endif
#ifndef B3_MAX_JOINTS
#define B3_MAX_JOINTS 32
#endif

// Experimental: faster on some dense-contact batches, but increases register
// pressure and can slow sparse worlds. Benchmark before enabling.
#ifndef B3_MERGE_WARM_CACHE
#define B3_MERGE_WARM_CACHE 0
#endif
/* 1: compact metadata. 2: also iterate eligible shape bitsets and reject
 * separated spheres from cached centers. Both preserve contact order and use
 * the original narrow phase, including its final separation calculation. */
#ifndef B3_COMPACT_PAIRS
#define B3_COMPACT_PAIRS 0
#endif
#if B3_COMPACT_PAIRS >= 2 && B3_MAX_SHAPES > 32
#error "Bitset broad phase requires at most 32 shapes"
#endif
#ifndef B3_UNCLAMPED_ROTATION
#define B3_UNCLAMPED_ROTATION 0
#endif
#ifndef B3_STATIC_RESTITUTION
#define B3_STATIC_RESTITUTION 0
#endif
#ifndef B3_JOINT_ITERS
#define B3_JOINT_ITERS 2
#endif
#ifndef B3_RELAX_ITERS
#define B3_RELAX_ITERS B3_JOINT_ITERS
#endif
#if B3_JOINT_ITERS < 1 || B3_RELAX_ITERS < 1
#error "Solver iteration counts must be positive"
#endif
#ifndef B3_PERSISTENT_GS
#define B3_PERSISTENT_GS 0
#endif
#ifndef B3_REUSE_GS_CACHE
#define B3_REUSE_GS_CACHE 0
#endif
#ifndef B3_PACKED_INTEGRATE
#define B3_PACKED_INTEGRATE 0
#endif
#if B3_PACKED_INTEGRATE && !B3_PERSISTENT_GS
#error "Packed integration requires persistent packed state"
#endif
#ifndef B3_COUPLED_HINGE
#define B3_COUPLED_HINGE 0
#endif
#ifndef B3_ALTERNATE_JOINT_ORDER
#define B3_ALTERNATE_JOINT_ORDER 0
#endif
#if B3_ALTERNATE_JOINT_ORDER && !defined(B3_PACKED_GS)
#error "Alternate joint ordering experiment requires packed solver"
#endif
#if B3_COUPLED_HINGE && (!defined(B3_PACKED_GS) || !defined(B3_REVOLUTE_ONLY))
#error "Coupled hinge experiment requires packed revolute-only solver"
#endif
#if B3_COUPLED_HINGE && (defined(B3_ABLATE_NO_PERP) || defined(B3_ABLATE_NO_POINT))
#error "Coupled hinge experiment cannot ablate one of its coupled blocks"
#endif
#if B3_PERSISTENT_GS && (!defined(B3_PACKED_GS) || !defined(B3_REVOLUTE_ONLY) \
    || !defined(B3_INTERLEAVE_CONTACTS) || !defined(B3_SKIP_RESTITUTION))
#error "Persistent GS requires packed revolute-only, interleaved contacts, and no restitution"
#endif
#ifndef B3_CONNECT_WORDS
#define B3_CONNECT_WORDS ((B3_MAX_BODIES + 63) / 64)
#endif

#define B3_PI 3.14159265359f
#define B3_LINEAR_SLOP 0.005f
#define B3_SPECULATIVE 0.020f
#define B3_MAX_MANIFOLD 4
#define B3_MAX_ROTATION (0.25f * B3_PI)
#define B3_GYRO_ITERS 1
#define B3_MIN_FRICTION_W 1.0e-10f

// Experimental exponential-map integration; benchmark before enabling.
// Changes integration semantics from normalized Euler, not solver iterations.
#ifndef B3_POLY_ROTATION
#define B3_POLY_ROTATION 0
#endif

// Experimental packed tree projection. Contacts and axial actuator/limit rows
// retain projected GS; bilateral hinge rows use leaf-to-root condensation.
#ifndef B3_TREE_DUAL
#define B3_TREE_DUAL 0
#endif
#if B3_TREE_DUAL && !defined(B3_PACKED_GS)
#error "Tree dual experiment requires packed solver"
#endif
#if B3_TREE_DUAL && (defined(B3_ABLATE_NO_PERP) || defined(B3_ABLATE_NO_POINT))
#error "Tree dual experiment requires all five hinge rows"
#endif
// Include the soft axial spring in the tree system, with torque-bound checks.
#ifndef B3_TREE_SPRINGS
#define B3_TREE_SPRINGS 0
#endif
#if B3_TREE_SPRINGS != 0 && B3_TREE_SPRINGS != 1
#error "B3_TREE_SPRINGS must be 0 or 1"
#endif
#if B3_TREE_SPRINGS && !B3_TREE_DUAL
#error "Condensed springs require tree dual solver"
#endif
#if B3_TREE_SPRINGS && defined(B3_ABLATE_NO_SPRING)
#error "Condensed springs cannot ablate the spring rows"
#endif
#ifndef B3_TREE_OUTER_ITERS
#define B3_TREE_OUTER_ITERS 2
#endif
#if B3_TREE_OUTER_ITERS < 1
#error "B3_TREE_OUTER_ITERS must be positive"
#endif

// Experimental: exact block-tridiagonal dual KKT joint solve for plain
// revolute chains (anchor + 2 alignment rows, M-ABD style) instead of
// iterated Gauss-Seidel sweeps. Falls back to the legacy path for any
// other topology, actuated joints, or singular systems.
#ifndef B3_JOINT_DUAL
#define B3_JOINT_DUAL 0
#endif

// Experimental: fuse sqrt+divide pairs into the reciprocal-square-root
// intrinsic on device (max 2 ulp vs ~1 ulp for the IEEE pair). Touches
// normalization and cone/velocity clamps only; solver reciprocals of
// effective masses stay exact. Benchmark before enabling.
#ifndef B3_RSQRT_MATH
#define B3_RSQRT_MATH 0
#endif

/* 1: jointed worlds use Delassus contact mass/apply in b3_solve_contacts
 * and in packed-GS interleaved contacts (b3_solve_contacts_gs_w).
 * Grain worlds (joint_count==0) stay on independent 1/m. Joint GS for
 * motors/springs/limits/welds is unchanged. Packed GS builds B3Art once
 * per solve. Rolling resistance applies in the velocity pass. Tree-dual
 * reuses the same packed-GS Delassus apply. Set 0 to compile without art. */
#ifndef B3_ART_CONTACTS
#define B3_ART_CONTACTS 1
#endif

/* Optional per-substep force law. Define before include. The hook may add
 * to externally applied forces: its contributions are consumed once, then
 * the original force/torque are restored for the next substep. Read current
 * poses as center + delta_pos and delta_rot * rotation. No default overhead. */
#ifndef B3_USER_FORCES
#define B3_USER_FORCES(w, h) ((void)0)
#define B3_HAS_USER_FORCES 0
#else
#define B3_HAS_USER_FORCES 1
#endif

#define B3_STATIC 0
#define B3_KINEMATIC 1
#define B3_DYNAMIC 2

#define B3_SPHERE 0
#define B3_CAPSULE 1
#define B3_BOX 2
#define B3_CYLINDER 3

/* Prototype opt-in: MuJoCo accepts a pair when either directional mask
 * matches. Upstream Puffysics requires both directions. */
#ifndef B3_MUJOCO_COLLISION_FILTER
#define B3_MUJOCO_COLLISION_FILTER 0
#endif

#define B3_COLLISION_GJK_LIMIT 1u
#define B3_COLLISION_EPA_SEED 2u
#define B3_COLLISION_EPA_LIMIT 4u
#define B3_COLLISION_EPA_CAPACITY 8u
#define B3_COLLISION_DEGENERATE 16u
#define B3_COLLISION_CONTACT_CAPACITY 32u
#define B3_COLLISION_INVALID_SHAPE 64u

#define B3_JOINT_WELD 0
#define B3_JOINT_REVOLUTE 1

#define B3_FLAG_DYNAMIC 0x00001000u
#define B3_LOCK_LIN_X 0x00000001u
#define B3_LOCK_LIN_Y 0x00000002u
#define B3_LOCK_LIN_Z 0x00000004u
#define B3_LOCK_ANG_X 0x00000008u
#define B3_LOCK_ANG_Y 0x00000010u
#define B3_LOCK_ANG_Z 0x00000020u

typedef struct B3Vec3 {
    float x, y, z;
} B3Vec3;

typedef struct B3Vec2 {
    float x, y;
} B3Vec2;

typedef struct B3Quat {
    B3Vec3 v;
    float s;
} B3Quat;

typedef struct B3Mat3 {
    B3Vec3 cx, cy, cz;
} B3Mat3;

typedef struct B3Mat2 {
    B3Vec2 cx, cy;
} B3Mat2;

typedef struct B3AABB {
    B3Vec3 lo, hi;
} B3AABB;

typedef struct B3Soft {
    float bias_rate;
    float mass_scale;
    float impulse_scale;
} B3Soft;

typedef struct B3Body {
    B3Vec3 position;
    B3Quat rotation;
    B3Vec3 center;
    B3Vec3 local_center;
    B3Vec3 lin_vel;
    B3Vec3 ang_vel;
    B3Vec3 force;
    B3Vec3 torque;
    B3Vec3 delta_pos;
    B3Quat delta_rot;
    float inv_mass;
    B3Vec3 inv_inertia;
    B3Mat3 inv_i_world;
    float linear_damping;
    float angular_damping;
    float gravity_scale;
    int type;
    uint32_t flags;
} B3Body;

typedef struct B3Shape {
    int body;
    int type;
    B3Vec3 local_pos;
    B3Quat local_rot;
    float radius;
    B3Vec3 half;
    float friction;
    float restitution;
    float rolling;
    float density;
    uint64_t category;
    uint64_t mask;
} B3Shape;

typedef struct B3Point {
    B3Vec3 r_a;
    B3Vec3 r_b;
    float base_sep;
    float rel_vel;
    float normal_impulse;
    float total_normal;
    float normal_mass;
    float lever;
    uint32_t feature;
} B3Point;

typedef struct B3Contact {
    int shape_a;
    int shape_b;
    int body_a;
    int body_b;
    int point_count;
    int static_contact;
    B3Vec3 normal;
    B3Vec3 tangent1;
    B3Vec3 tangent2;
    B3Point points[B3_MAX_MANIFOLD];
    B3Vec3 center_a;
    B3Vec3 center_b;
    float friction;
    float restitution;
    float rolling;
    float twist_mass;
    float twist_impulse;
    B3Vec2 friction_impulse;
    B3Vec3 rolling_impulse;

    B3Mat2 tangent_mass;
    B3Mat3 rolling_mass;
    float inv_mass_a;
    float inv_mass_b;
    B3Mat3 inv_i_a;
    B3Mat3 inv_i_b;
    B3Soft softness;
} B3Contact;

typedef struct B3Warm {
    int shape_a;
    int shape_b;
    int point_count;
    uint32_t feature[B3_MAX_MANIFOLD];
    float normal_impulse[B3_MAX_MANIFOLD];
    B3Vec2 friction_impulse;
    float twist_impulse;
    B3Vec3 rolling_impulse;
} B3Warm;

typedef struct B3Joint {
#ifndef B3_REVOLUTE_ONLY
    int type;
#endif
    int body_a;
    int body_b;
    int collide_connected;
    int fixed_rotation;
    B3Vec3 local_anchor_a;
    B3Vec3 local_anchor_b;
    B3Quat local_rot_a;
    B3Quat local_rot_b;
    float constraint_hertz;
    float constraint_damping;
    B3Soft softness;
    float inv_mass_a;
    float inv_mass_b;
    B3Mat3 inv_i_a;
    B3Mat3 inv_i_b;
    B3Vec3 frame_p_a;
    B3Vec3 frame_p_b;
    B3Quat frame_q_a;
    B3Quat frame_q_b;
    B3Vec3 delta_center;
    B3Vec3 linear_impulse;
#ifndef B3_REVOLUTE_ONLY
    B3Vec3 angular_impulse;
    B3Mat3 angular_mass;
    float linear_hertz;
    float linear_damping;
    float angular_hertz;
    float angular_damping;
    B3Soft linear_spring;
    B3Soft angular_spring;
#endif
    B3Vec2 perp_impulse;
    float spring_impulse;
#ifndef B3_REVOLUTE_ONLY
    float motor_impulse;
#endif
    float lower_impulse;
    float upper_impulse;
    float hertz;
    float damping_ratio;
    float max_motor_torque;
#ifndef B3_REVOLUTE_ONLY
    float motor_speed;
#endif
    float target_angle;
    float lower_angle;
    float upper_angle;
    int enable_spring;
#ifndef B3_REVOLUTE_ONLY
    int enable_motor;
#endif
    int enable_limit;
    B3Vec3 rotation_axis;
    B3Vec3 perp_x;
    B3Vec3 perp_y;
    float axial_mass;
    B3Soft spring_softness;
#ifdef B3_CACHE_JOINTS
    /* Geometry + inv(K) for one GS pass. K is constant while delta_rot
     * is frozen, so invert once and matvec in the inner loop. */
    B3Vec3 cache_ra;
    B3Vec3 cache_rb;
    B3Mat3 cache_point_invk;
    B3Mat2 cache_ang_invk;
    B3Vec3 cache_ia_ax;
    B3Vec3 cache_ib_ax;
    float cache_twist;
    float cache_rel_x;
    float cache_rel_y;
#endif
} B3Joint;

#ifdef B3_PACKED_GS
/* Inner-loop snapshot. Bodies keep v/w/Δp/Δq + dyn inv(M). Joints keep
 * impulses, cached inv(K), and the hinge rows. Contacts drop unused
 * prepare fields. inv_i lives on the body so 14 hinges do not each
 * carry two 3x3 copies. */
#define B3_GS_FIXED 1
#define B3_GS_SPRING 2
#define B3_GS_LIMIT 4

typedef struct B3GsBody {
    B3Vec3 lin_vel;
    B3Vec3 ang_vel;
    B3Vec3 delta_pos;
    B3Quat delta_rot;
    float inv_mass;
    B3Mat3 inv_i;
    uint32_t flags;
} B3GsBody;

typedef struct B3GsJoint {
    int body_a;
    int body_b;
    int bits;
    float target_angle;
    float lower_angle;
    float upper_angle;
    float axial_mass;
    float max_motor_torque;
    float spring_impulse;
    float lower_impulse;
    float upper_impulse;
    float cache_twist;
    float cache_rel_x;
    float cache_rel_y;
    B3Soft softness;
    B3Soft spring_softness;
    B3Vec3 rotation_axis;
    B3Vec3 perp_x;
    B3Vec3 perp_y;
    B3Vec3 cache_ra;
    B3Vec3 cache_rb;
    B3Vec3 cache_ia_ax;
    B3Vec3 cache_ib_ax;
    B3Vec3 delta_center;
    B3Vec3 linear_impulse;
    B3Vec2 perp_impulse;
    B3Mat3 cache_point_invk;
    B3Mat2 cache_ang_invk;
#if B3_COUPLED_HINGE
    B3Vec3 cache_point_perp_x;
    B3Vec3 cache_point_perp_y;
#endif
} B3GsJoint;

typedef struct B3GsPoint {
    B3Vec3 r_a;
    B3Vec3 r_b;
    float base_sep;
    float normal_impulse;
    float total_normal;
    float normal_mass;
    float lever;
} B3GsPoint;

typedef struct B3GsContact {
    int body_a;
    int body_b;
    int point_count;
    B3Vec3 normal;
    B3Vec3 tangent1;
    B3Vec3 tangent2;
    B3Vec3 center_a;
    B3Vec3 center_b;
    float friction;
    float rolling;
    B3Vec3 rolling_impulse;
    float twist_mass;
    float twist_impulse;
    B3Vec2 friction_impulse;
    B3Mat2 tangent_mass;
    B3Soft softness;
    B3GsPoint points[B3_MAX_MANIFOLD];
} B3GsContact;
#endif


typedef struct B3World {
    B3Vec3 gravity;
    float contact_hertz;
    float contact_damping;
    float contact_speed;
    float restitution_threshold;
    float max_linear_speed;
    int body_count;
    int shape_count;
    int contact_count;
    int joint_count;
    /* Sticky diagnostic state. A nonzero status invalidates a prototype run. */
    uint32_t collision_status;
    uint32_t collision_failed_queries;
    uint32_t collision_cylinder_queries;
    uint32_t collision_contact_overflows;
    int collision_max_gjk_iterations;
    int collision_max_epa_iterations;
    /* First failed narrowphase's raw operands, retained until world reset.
     * meta: status, shape IDs A/B, types A/B, GJK/EPA iterations.
     * data: two 18-float records: body pos3/quat xyzw4, shape local
     * pos3/quat xyzw4, radius, half xyz3. */
    int collision_failure_meta[7];
    float collision_failure_data[36];
    B3Body bodies[B3_MAX_BODIES];
    B3Shape shapes[B3_MAX_SHAPES];
    B3Contact contacts[B3_MAX_CONTACTS];
    B3Joint joints[B3_MAX_JOINTS];
} B3World;

typedef struct B3BodyDef {
    int type;
    B3Vec3 position;
    B3Quat rotation;
    B3Vec3 lin_vel;
    B3Vec3 ang_vel;
    float linear_damping;
    float angular_damping;
    float gravity_scale;
    uint32_t flags;
} B3BodyDef;

typedef struct B3ShapeDef {
    float density;
    float friction;
    float restitution;
    float rolling;
    uint64_t category;
    uint64_t mask;
} B3ShapeDef;

B3_HD B3_INL B3Vec3 b3_v(float x, float y, float z) {
    B3Vec3 r;
    r.x = x;
    r.y = y;
    r.z = z;
    return r;
}

B3_HD B3_INL B3Vec3 b3_add(B3Vec3 a, B3Vec3 b) {
    return b3_v(a.x + b.x, a.y + b.y, a.z + b.z);
}

B3_HD B3_INL B3Vec3 b3_sub(B3Vec3 a, B3Vec3 b) {
    return b3_v(a.x - b.x, a.y - b.y, a.z - b.z);
}

B3_HD B3_INL B3Vec3 b3_neg(B3Vec3 a) {
    return b3_v(-a.x, -a.y, -a.z);
}

B3_HD B3_INL B3Vec3 b3_mul(B3Vec3 a, float s) {
    return b3_v(a.x * s, a.y * s, a.z * s);
}

B3_HD B3_INL B3Vec3 b3_madd(B3Vec3 a, float s, B3Vec3 b) {
    return b3_v(a.x + s * b.x, a.y + s * b.y, a.z + s * b.z);
}

B3_HD B3_INL B3Vec3 b3_msub(B3Vec3 a, float s, B3Vec3 b) {
    return b3_v(a.x - s * b.x, a.y - s * b.y, a.z - s * b.z);
}

B3_HD B3_INL float b3_dot(B3Vec3 a, B3Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

B3_HD B3_INL B3Vec3 b3_cross(B3Vec3 a, B3Vec3 b) {
    return b3_v(a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

B3_HD B3_INL float b3_len2(B3Vec3 a) {
    return b3_dot(a, a);
}

B3_HD B3_INL float b3_len(B3Vec3 a) {
    return sqrtf(b3_len2(a));
}

/* a / sqrt(x): single-rounded when off, fused intrinsic when on. The clamp
 * sites need the off path to stay bit-identical to the historical code. */
B3_HD B3_INL float b3_rsqrt_scale(float a, float x) {
#if defined(__CUDA_ARCH__) && B3_RSQRT_MATH
    return a * rsqrtf(x);
#else
    return a / sqrtf(x);
#endif
}

B3_HD B3_INL float b3_rsqrt(float x) {
#if defined(__CUDA_ARCH__) && B3_RSQRT_MATH
    return rsqrtf(x);
#else
    return 1.0f / sqrtf(x);
#endif
}

B3_HD B3_INL B3Vec3 b3_norm(B3Vec3 a) {
    float l2 = b3_len2(a);
    return l2 > 0.0f ? b3_mul(a, b3_rsqrt(l2)) : b3_v(0.0f, 1.0f, 0.0f);
}

B3_HD B3_INL float b3_clamp(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

B3_HD B3_INL float b3_maxf(float a, float b) {
    return a > b ? a : b;
}

B3_HD B3_INL float b3_minf(float a, float b) {
    return a < b ? a : b;
}

B3_HD B3_INL B3Quat b3_q(float x, float y, float z, float s) {
    B3Quat q;
    q.v = b3_v(x, y, z);
    q.s = s;
    return q;
}

B3_HD B3_INL B3Quat b3_q_id(void) {
    return b3_q(0.0f, 0.0f, 0.0f, 1.0f);
}

B3_HD B3_INL float b3_qdot(B3Quat a, B3Quat b) {
    return b3_dot(a.v, b.v) + a.s * b.s;
}

B3_HD B3_INL B3Quat b3_qnorm(B3Quat q) {
    float d2 = b3_qdot(q, q);
    if (d2 <= 0.0f) {
        return b3_q_id();
    }
    float inv = b3_rsqrt(d2);
    return b3_q(q.v.x * inv, q.v.y * inv, q.v.z * inv, q.s * inv);
}

B3_HD B3_INL B3Quat b3_qmul(B3Quat a, B3Quat b) {
    B3Quat r;
    r.v = b3_add(b3_add(b3_mul(b.v, a.s), b3_mul(a.v, b.s)),
        b3_cross(a.v, b.v));
    r.s = a.s * b.s - b3_dot(a.v, b.v);
    return r;
}

B3_HD B3_INL B3Vec3 b3_rotate(B3Quat q, B3Vec3 v) {
    B3Vec3 t = b3_mul(b3_cross(q.v, v), 2.0f);
    return b3_add(v, b3_add(b3_mul(t, q.s), b3_cross(q.v, t)));
}

B3_HD B3_INL B3Vec3 b3_inv_rotate(B3Quat q, B3Vec3 v) {
    B3Quat c = b3_q(-q.v.x, -q.v.y, -q.v.z, q.s);
    return b3_rotate(c, v);
}

B3_HD B3_INL B3Quat b3_q_integrate(B3Quat q, B3Vec3 dw) {
#if B3_POLY_ROTATION
    float a2 = b3_len2(dw);
    float s, c;
    if (a2 <= B3_MAX_ROTATION * B3_MAX_ROTATION) {
        // Taylor polynomials in |dw|^2 for sin(|dw|/2)/|dw| and
        // cos(|dw|/2). At pi/4, omitted terms are < 7.9e-10 and
        // 1.6e-8 respectively, before float rounding. No sqrt/trig.
        s = 0.5f + a2 * (-1.0f / 48.0f + a2 * (1.0f / 3840.0f
            + a2 * (-1.0f / 645120.0f)));
        c = 1.0f + a2 * (-1.0f / 8.0f + a2 * (1.0f / 384.0f
            + a2 * (-1.0f / 46080.0f)));
    } else {
        // Public helper callers need not respect the stepping speed cap.
        float a = sqrtf(a2);
        s = sinf(0.5f * a) / a;
        c = cosf(0.5f * a);
    }
    // Retain normalization to prevent accumulated float norm drift.
    return b3_qnorm(b3_qmul(b3_q(s * dw.x, s * dw.y, s * dw.z, c), q));
#else
    B3Quat qd = b3_q(0.5f * dw.x, 0.5f * dw.y, 0.5f * dw.z, 0.0f);
    qd = b3_qmul(qd, q);
    return b3_qnorm(b3_q(q.v.x + qd.v.x, q.v.y + qd.v.y,
        q.v.z + qd.v.z, q.s + qd.s));
#endif
}

B3_HD B3_INL void b3_axes(B3Quat q, B3Vec3* x, B3Vec3* y, B3Vec3* z) {
    *x = b3_rotate(q, b3_v(1.0f, 0.0f, 0.0f));
    *y = b3_rotate(q, b3_v(0.0f, 1.0f, 0.0f));
    *z = b3_rotate(q, b3_v(0.0f, 0.0f, 1.0f));
}

B3_HD B3_INL B3Vec3 b3_mv(B3Mat3 m, B3Vec3 v) {
    return b3_add(b3_add(b3_mul(m.cx, v.x), b3_mul(m.cy, v.y)),
        b3_mul(m.cz, v.z));
}

B3_HD B3_INL B3Mat3 b3_maddm(B3Mat3 a, B3Mat3 b) {
    B3Mat3 r;
    r.cx = b3_add(a.cx, b.cx);
    r.cy = b3_add(a.cy, b.cy);
    r.cz = b3_add(a.cz, b.cz);
    return r;
}

B3_HD B3_INL B3Mat3 b3_mat0(void) {
    B3Mat3 m;
    m.cx = b3_v(0.0f, 0.0f, 0.0f);
    m.cy = b3_v(0.0f, 0.0f, 0.0f);
    m.cz = b3_v(0.0f, 0.0f, 0.0f);
    return m;
}

B3_HD B3_INL B3Mat3 b3_world_inv_i(B3Quat q, B3Vec3 inv) {
    B3Vec3 ax, ay, az;
    b3_axes(q, &ax, &ay, &az);
    B3Mat3 r;
    r.cx = b3_add(b3_add(b3_mul(ax, inv.x * ax.x),
        b3_mul(ay, inv.y * ay.x)), b3_mul(az, inv.z * az.x));
    r.cy = b3_add(b3_add(b3_mul(ax, inv.x * ax.y),
        b3_mul(ay, inv.y * ay.y)), b3_mul(az, inv.z * az.y));
    r.cz = b3_add(b3_add(b3_mul(ax, inv.x * ax.z),
        b3_mul(ay, inv.y * ay.z)), b3_mul(az, inv.z * az.z));
    return r;
}

B3_HD B3_INL B3Mat3 b3_invert3(B3Mat3 m) {
    B3Vec3 c0 = b3_cross(m.cy, m.cz);
    float det = b3_dot(m.cx, c0);
    B3Mat3 r = b3_mat0();
    if (fabsf(det) < 1.0e-12f) {
        return r;
    }
    float inv = 1.0f / det;
    B3Vec3 r0 = b3_mul(c0, inv);
    B3Vec3 r1 = b3_mul(b3_cross(m.cz, m.cx), inv);
    B3Vec3 r2 = b3_mul(b3_cross(m.cx, m.cy), inv);
    r.cx = b3_v(r0.x, r1.x, r2.x);
    r.cy = b3_v(r0.y, r1.y, r2.y);
    r.cz = b3_v(r0.z, r1.z, r2.z);
    return r;
}

B3_HD B3_INL B3Mat2 b3_invert2(B3Mat2 k) {
    float det = k.cx.x * k.cy.y - k.cx.y * k.cy.x;
    B3Mat2 r;
    r.cx = (B3Vec2){0.0f, 0.0f};
    r.cy = (B3Vec2){0.0f, 0.0f};
    if (fabsf(det) < 1.0e-12f) {
        return r;
    }
    float inv = 1.0f / det;
    r.cx.x = k.cy.y * inv;
    r.cx.y = -k.cx.y * inv;
    r.cy.x = -k.cy.x * inv;
    r.cy.y = k.cx.x * inv;
    return r;
}

B3_HD B3_INL B3Vec2 b3_mv2(B3Mat2 m, B3Vec2 v) {
    B3Vec2 r;
    r.x = m.cx.x * v.x + m.cy.x * v.y;
    r.y = m.cx.y * v.x + m.cy.y * v.y;
    return r;
}

B3_HD B3_INL B3Vec3 b3_perp(B3Vec3 n) {
    if (fabsf(n.x) >= 0.57735027f) {
        return b3_norm(b3_v(-n.y, n.x, 0.0f));
    }
    return b3_norm(b3_v(0.0f, -n.z, n.y));
}

B3_HD B3_INL B3Soft b3_make_soft(float hertz, float zeta, float h) {
    B3Soft s;
    s.bias_rate = 0.0f;
    s.mass_scale = 0.0f;
    s.impulse_scale = 0.0f;
    if (hertz == 0.0f) {
        return s;
    }
    float omega = 2.0f * B3_PI * hertz;
    float a1 = 2.0f * zeta + h * omega;
    float a2 = h * omega * a1;
    float a3 = 1.0f / (1.0f + a2);
    s.bias_rate = omega / a1;
    s.mass_scale = a2 * a3;
    s.impulse_scale = a3;
    return s;
}

B3_HD B3_INL B3Quat b3_qneg(B3Quat q) {
    return b3_q(-q.v.x, -q.v.y, -q.v.z, -q.s);
}

B3_HD B3_INL B3Quat b3_qconj(B3Quat q) {
    return b3_q(-q.v.x, -q.v.y, -q.v.z, q.s);
}

B3_HD B3_INL B3Quat b3_qinv_mul(B3Quat a, B3Quat b) {
    B3Vec3 t1 = b3_cross(b.v, a.v);
    B3Vec3 t2 = b3_madd(t1, a.s, b.v);
    B3Vec3 t3 = b3_msub(t2, b.s, a.v);
    return b3_q(t3.x, t3.y, t3.z, a.s * b.s + b3_dot(a.v, b.v));
}

// 0: libm; 1: degree-19 polynomial; 2: interpolated read-only LUT;
// 3: same LUT in CUDA constant memory; 4: cubic/quadratic rational.
// All experiments default off.
#ifndef B3_TWIST_APPROX
#define B3_TWIST_APPROX 0
#endif
#if B3_TWIST_APPROX < 0 || B3_TWIST_APPROX > 4
#error "Unknown twist approximation"
#endif
#if B3_TWIST_APPROX
#include "atan_approx.cuh"
#endif

B3_HD B3_INL float b3_twist(B3Quat q) {
#if B3_TWIST_APPROX
    float y = q.s < 0.0f ? -q.v.z : q.v.z;
    float x = q.s < 0.0f ? -q.s : q.s;
    float ay = fabsf(y);
    float hi = b3_maxf(x, ay), lo = b3_minf(x, ay);
    // Preserve atan2's signed-zero, nonfinite and degenerate behavior.
    if (!(hi > 0.0f) || !isfinite(x) || !isfinite(y) || x == 0.0f)
        return 2.0f * atan2f(y, x);
    float a = b3_atan_unit(lo / hi);
    if (ay > x) a = 0.5f * B3_PI - a;
    return 2.0f * copysignf(a, y);
#else
    float t = q.s < 0.0f ? atan2f(-q.v.z, -q.s) : atan2f(q.v.z, q.s);
    return 2.0f * t;
#endif
}

B3_HD B3_INL void b3_hinge_perps(B3Quat qa, B3Quat rel,
        B3Vec3* px, B3Vec3* py) {
    B3Vec3 ex = b3_v(1.0f, 0.0f, 0.0f);
    B3Vec3 ey = b3_v(0.0f, 1.0f, 0.0f);
    *px = b3_mul(b3_rotate(qa,
        b3_add(b3_mul(ex, rel.s), b3_cross(rel.v, ex))), 0.5f);
    *py = b3_mul(b3_rotate(qa,
        b3_add(b3_mul(ey, rel.s), b3_cross(rel.v, ey))), 0.5f);
}

B3_HD B3_INL B3Quat b3_q_axis_angle(B3Vec3 axis, float radians) {
    axis = b3_norm(axis);
    float h = 0.5f * radians;
    float s = sinf(h);
    return b3_q(s * axis.x, s * axis.y, s * axis.z, cosf(h));
}

B3_HD B3_INL B3Quat b3_q_from_z(B3Vec3 z) {
    z = b3_norm(z);
    B3Vec3 from = b3_v(0.0f, 0.0f, 1.0f);
    float d = b3_dot(from, z);
    if (d > 0.999999f) {
        return b3_q_id();
    }
    if (d < -0.999999f) {
        B3Vec3 axis = b3_perp(from);
        return b3_q(axis.x, axis.y, axis.z, 0.0f);
    }
    B3Vec3 axis = b3_cross(from, z);
    return b3_qnorm(b3_q(axis.x, axis.y, axis.z, 1.0f + d));
}

B3_HD B3_INL B3Vec3 b3_solve3(B3Mat3 a, B3Vec3 b) {
    return b3_mv(b3_invert3(a), b);
}

B3_HD B3_INL B3Vec2 b3_solve2(B3Mat2 m, B3Vec2 b) {
    float det = m.cx.x * m.cy.y - m.cx.y * m.cy.x;
    B3Vec2 r;
    r.x = 0.0f;
    r.y = 0.0f;
    if (det <= 1.0e-12f) {
        return r;
    }
    float inv = 1.0f / det;
    r.x = inv * (m.cy.y * b.x - m.cy.x * b.y);
    r.y = inv * (-m.cx.y * b.x + m.cx.x * b.y);
    return r;
}

B3_HD B3_INL B3Vec3 b3_xf_point(B3Vec3 p, B3Quat q, B3Vec3 local) {
    return b3_add(p, b3_rotate(q, local));
}

B3_HD B3_INL int b3_aabb_overlap(B3AABB a, B3AABB b) {
    return a.lo.x <= b.hi.x && a.hi.x >= b.lo.x
        && a.lo.y <= b.hi.y && a.hi.y >= b.lo.y
        && a.lo.z <= b.hi.z && a.hi.z >= b.lo.z;
}

B3_HD B3_INL B3BodyDef b3_default_body(void) {
    B3BodyDef d;
    memset(&d, 0, sizeof(d));
    d.type = B3_STATIC;
    d.rotation = b3_q_id();
    d.gravity_scale = 1.0f;
    return d;
}

B3_HD B3_INL B3ShapeDef b3_default_shape(void) {
    B3ShapeDef d;
    d.density = 1000.0f;
    d.friction = 0.6f;
    d.restitution = 0.0f;
    d.rolling = 0.0f;
    d.category = ~0ull;
    d.mask = ~0ull;
    return d;
}

B3_HD B3_INL void b3_world_init(B3World* w) {
    memset(w, 0, sizeof(*w));
    w->gravity = b3_v(0.0f, -10.0f, 0.0f);
    w->contact_hertz = 30.0f;
    w->contact_damping = 10.0f;
    w->contact_speed = 3.0f;
    w->restitution_threshold = 1.0f;
    w->max_linear_speed = 400.0f;
}

B3_HD B3_INL void b3_body_from_def(B3Body* b, const B3BodyDef* def) {
    memset(b, 0, sizeof(*b));
    b->position = def->position;
    b->rotation = b3_qnorm(def->rotation);
    b->center = def->position;
    b->lin_vel = def->lin_vel;
    b->ang_vel = def->ang_vel;
    b->delta_rot = b3_q_id();
    b->linear_damping = def->linear_damping;
    b->angular_damping = def->angular_damping;
    b->gravity_scale = def->gravity_scale;
    b->type = def->type;
    b->flags = def->flags;
    if (def->type == B3_DYNAMIC) {
        b->flags |= B3_FLAG_DYNAMIC;
    }
}

B3_HD B3_INL void b3_shape_fill(B3Shape* s, int body, int type,
        B3Vec3 local_pos, B3Quat local_rot, float radius, B3Vec3 half,
        const B3ShapeDef* def) {
    s->body = body;
    s->type = type;
    s->local_pos = local_pos;
    s->local_rot = b3_qnorm(local_rot);
    s->radius = radius;
    s->half = half;
    s->friction = def->friction;
    s->restitution = def->restitution;
    s->rolling = def->rolling;
    s->density = def->density;
    s->category = def->category;
    s->mask = def->mask;
}

B3_HD B3_INL int b3_create_body(B3World* w, const B3BodyDef* def) {
    assert(w->body_count < B3_MAX_BODIES);
    int id = w->body_count++;
    b3_body_from_def(&w->bodies[id], def);
    return id;
}

B3_HD B3_INL int b3_add_shape(B3World* w, int body, int type,
        B3Vec3 local_pos, B3Quat local_rot, float radius, B3Vec3 half,
        const B3ShapeDef* def) {
    assert(w->shape_count < B3_MAX_SHAPES);
    int id = w->shape_count++;
    b3_shape_fill(&w->shapes[id], body, type, local_pos, local_rot,
        radius, half, def);
    return id;
}

B3_HD B3_INL int b3_create_sphere(B3World* w, int body, B3Vec3 c,
        float r, const B3ShapeDef* def) {
    return b3_add_shape(w, body, B3_SPHERE, c, b3_q_id(), r,
        b3_v(0.0f, 0.0f, 0.0f), def);
}

B3_HD B3_INL int b3_create_capsule(B3World* w, int body, float half_len,
        float r, const B3ShapeDef* def) {
    return b3_add_shape(w, body, B3_CAPSULE, b3_v(0.0f, 0.0f, 0.0f),
        b3_q_id(), r, b3_v(0.0f, half_len, 0.0f), def);
}

B3_HD B3_INL int b3_create_box(B3World* w, int body, B3Vec3 half,
        const B3ShapeDef* def) {
    return b3_add_shape(w, body, B3_BOX, b3_v(0.0f, 0.0f, 0.0f),
        b3_q_id(), 0.0f, half, def);
}

B3_HD B3_INL int b3_create_box_local(B3World* w, int body, B3Vec3 half,
        B3Vec3 local_pos, B3Quat local_rot, const B3ShapeDef* def) {
    return b3_add_shape(w, body, B3_BOX, local_pos, local_rot, 0.0f,
        half, def);
}

/* Analytic finite cylinder, local +Y axis, radius r and half length h. */
B3_HD B3_INL int b3_create_cylinder_local(B3World* w, int body, float h,
        float r, B3Vec3 local_pos, B3Quat local_rot, const B3ShapeDef* def) {
    assert(r > 0.0f && h > 0.0f);
    return b3_add_shape(w, body, B3_CYLINDER, local_pos, local_rot, r,
        b3_v(0.0f, h, 0.0f), def);
}

B3_HD B3_INL void b3_set_inertial(B3World* w, int body, float mass,
        B3Vec3 local_com, B3Vec3 inertia) {
    B3Body* b = &w->bodies[body];
    b->local_center = local_com;
    b->center = b3_xf_point(b->position, b->rotation, local_com);
    if (b->type != B3_DYNAMIC || mass <= 0.0f) {
        b->inv_mass = 0.0f;
        b->inv_inertia = b3_v(0.0f, 0.0f, 0.0f);
        b->inv_i_world = b3_mat0();
        return;
    }
    b->inv_mass = 1.0f / mass;
    b->inv_inertia = b3_v(
        inertia.x > 0.0f ? 1.0f / inertia.x : 0.0f,
        inertia.y > 0.0f ? 1.0f / inertia.y : 0.0f,
        inertia.z > 0.0f ? 1.0f / inertia.z : 0.0f);
    b->inv_i_world = b3_world_inv_i(b->rotation, b->inv_inertia);
}

B3_HD B3_INL void b3_shape_mass(const B3Shape* s, float* mass,
        B3Vec3* com, B3Vec3* inertia) {
    if (s->type == B3_CYLINDER) {
        float r = s->radius;
        float h = s->half.y;
        float m = 2.0f * B3_PI * r * r * h * s->density;
        *mass = m;
        *com = s->local_pos;
        float transverse = m * (3.0f * r * r + 4.0f * h * h) / 12.0f;
        *inertia = b3_v(transverse, 0.5f * m * r * r, transverse);
        return;
    }
    if (s->type == B3_SPHERE) {
        float r = s->radius;
        float m = (4.0f / 3.0f) * B3_PI * r * r * r * s->density;
        *mass = m;
        *com = s->local_pos;
        float i = 0.4f * m * r * r;
        *inertia = b3_v(i, i, i);
        return;
    }
    if (s->type == B3_BOX) {
        float hx = s->half.x;
        float hy = s->half.y;
        float hz = s->half.z;
        float m = 8.0f * hx * hy * hz * s->density;
        *mass = m;
        *com = s->local_pos;
        *inertia = b3_v(
            (m / 3.0f) * (hy * hy + hz * hz),
            (m / 3.0f) * (hx * hx + hz * hz),
            (m / 3.0f) * (hx * hx + hy * hy));
        return;
    }
    float r = s->radius;
    float h = 2.0f * s->half.y;
    float cyl_v = B3_PI * r * r * h;
    float sph_v = (4.0f / 3.0f) * B3_PI * r * r * r;
    float cyl_m = cyl_v * s->density;
    float sph_m = sph_v * s->density;
    float m = cyl_m + sph_m;
    *mass = m;
    *com = s->local_pos;
    float ix = 0.5f * cyl_m * r * r + 0.4f * sph_m * r * r;
    float iy = (1.0f / 12.0f) * cyl_m * (3.0f * r * r + h * h)
        + 0.4f * sph_m * r * r
        + 0.125f * sph_m * (3.0f * r + 2.0f * h) * h;
    *inertia = b3_v(iy, ix, iy);
}

B3_HD B3_INL void b3_finalize_mass_of(B3Body* b, B3Shape* shapes, int n,
        int body) {
    if (b->type != B3_DYNAMIC) {
        b->inv_mass = 0.0f;
        b->inv_inertia = b3_v(0.0f, 0.0f, 0.0f);
        b->inv_i_world = b3_mat0();
        b->center = b->position;
        return;
    }
    float mass = 0.0f;
    B3Vec3 com = b3_v(0.0f, 0.0f, 0.0f);
    B3Vec3 inertia = b3_v(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < n; i++) {
        B3Shape* s = &shapes[i];
        if (s->body != body || s->density <= 0.0f) {
            continue;
        }
        float sm;
        B3Vec3 sc, si;
        b3_shape_mass(s, &sm, &sc, &si);
        mass += sm;
        com = b3_madd(com, sm, sc);
        inertia = b3_add(inertia, si);
    }
    if (mass <= 0.0f) {
        b->inv_mass = 0.0f;
        b->inv_inertia = b3_v(0.0f, 0.0f, 0.0f);
        return;
    }
    com = b3_mul(com, 1.0f / mass);
    b->local_center = com;
    b->center = b3_xf_point(b->position, b->rotation, com);
    b->inv_mass = 1.0f / mass;
    b->inv_inertia = b3_v(
        inertia.x > 0.0f ? 1.0f / inertia.x : 0.0f,
        inertia.y > 0.0f ? 1.0f / inertia.y : 0.0f,
        inertia.z > 0.0f ? 1.0f / inertia.z : 0.0f);
    b->inv_i_world = b3_world_inv_i(b->rotation, b->inv_inertia);
}

B3_HD B3_INL void b3_finalize_mass(B3World* w, int body) {
    b3_finalize_mass_of(&w->bodies[body], w->shapes, w->shape_count, body);
}

B3_HD B3_INL B3Quat b3_shape_rot(const B3Body* b, const B3Shape* s) {
    return b3_qmul(b->rotation, s->local_rot);
}

B3_HD B3_INL B3Vec3 b3_shape_pos(const B3Body* b, const B3Shape* s) {
    return b3_xf_point(b->position, b->rotation, s->local_pos);
}

B3_HD B3_INL B3AABB b3_shape_aabb(const B3Body* b, const B3Shape* s) {
    B3Vec3 p = b3_shape_pos(b, s);
    B3Quat q = b3_shape_rot(b, s);
    B3AABB a;
    if (s->type == B3_CYLINDER) {
        B3Vec3 axis = b3_rotate(q, b3_v(0.0f, 1.0f, 0.0f));
        B3Vec3 e = b3_v(
            s->half.y * fabsf(axis.x) + s->radius * sqrtf(b3_maxf(0.0f, 1.0f - axis.x * axis.x)),
            s->half.y * fabsf(axis.y) + s->radius * sqrtf(b3_maxf(0.0f, 1.0f - axis.y * axis.y)),
            s->half.y * fabsf(axis.z) + s->radius * sqrtf(b3_maxf(0.0f, 1.0f - axis.z * axis.z)));
        a.lo = b3_sub(p, e);
        a.hi = b3_add(p, e);
        return a;
    }
    if (s->type == B3_SPHERE) {
        B3Vec3 e = b3_v(s->radius, s->radius, s->radius);
        a.lo = b3_sub(p, e);
        a.hi = b3_add(p, e);
        return a;
    }
    if (s->type == B3_CAPSULE) {
        B3Vec3 y = b3_rotate(q, b3_v(0.0f, s->half.y, 0.0f));
        B3Vec3 c1 = b3_sub(p, y);
        B3Vec3 c2 = b3_add(p, y);
        B3Vec3 e = b3_v(s->radius, s->radius, s->radius);
        a.lo = b3_v(b3_minf(c1.x, c2.x), b3_minf(c1.y, c2.y),
            b3_minf(c1.z, c2.z));
        a.hi = b3_v(b3_maxf(c1.x, c2.x), b3_maxf(c1.y, c2.y),
            b3_maxf(c1.z, c2.z));
        a.lo = b3_sub(a.lo, e);
        a.hi = b3_add(a.hi, e);
        return a;
    }
    B3Vec3 ax, ay, az;
    b3_axes(q, &ax, &ay, &az);
    B3Vec3 e = b3_v(
        fabsf(ax.x) * s->half.x + fabsf(ay.x) * s->half.y
            + fabsf(az.x) * s->half.z,
        fabsf(ax.y) * s->half.x + fabsf(ay.y) * s->half.y
            + fabsf(az.y) * s->half.z,
        fabsf(ax.z) * s->half.x + fabsf(ay.z) * s->half.y
            + fabsf(az.z) * s->half.z);
    a.lo = b3_sub(p, e);
    a.hi = b3_add(p, e);
    return a;
}

B3_HD B3_INL B3Vec3 b3_closest_seg(B3Vec3 a, B3Vec3 b, B3Vec3 p) {
    B3Vec3 ab = b3_sub(b, a);
    float d = b3_len2(ab);
    if (d <= 1.0e-12f) {
        return a;
    }
    float t = b3_clamp(b3_dot(b3_sub(p, a), ab) / d, 0.0f, 1.0f);
    return b3_madd(a, t, ab);
}

B3_HD B3_INL void b3_closest_segs(B3Vec3 a1, B3Vec3 a2, B3Vec3 b1,
        B3Vec3 b2, B3Vec3* pa, B3Vec3* pb) {
    B3Vec3 d1 = b3_sub(a2, a1);
    B3Vec3 d2 = b3_sub(b2, b1);
    B3Vec3 r = b3_sub(a1, b1);
    float a = b3_len2(d1);
    float e = b3_len2(d2);
    float f = b3_dot(d2, r);
    float s;
    float t;
    if (a <= 1.0e-12f && e <= 1.0e-12f) {
        *pa = a1;
        *pb = b1;
        return;
    }
    if (a <= 1.0e-12f) {
        s = 0.0f;
        t = b3_clamp(f / e, 0.0f, 1.0f);
    } else {
        float c = b3_dot(d1, r);
        if (e <= 1.0e-12f) {
            t = 0.0f;
            s = b3_clamp(-c / a, 0.0f, 1.0f);
        } else {
            float b = b3_dot(d1, d2);
            float den = a * e - b * b;
            s = den != 0.0f ? b3_clamp((b * f - c * e) / den, 0.0f, 1.0f)
                : 0.0f;
            t = (b * s + f) / e;
            if (t < 0.0f) {
                t = 0.0f;
                s = b3_clamp(-c / a, 0.0f, 1.0f);
            } else if (t > 1.0f) {
                t = 1.0f;
                s = b3_clamp((b - c) / a, 0.0f, 1.0f);
            }
        }
    }
    *pa = b3_madd(a1, s, d1);
    *pb = b3_madd(b1, t, d2);
}

typedef struct B3Mani {
    int count;
    uint32_t status;
    int gjk_iterations;
    int epa_iterations;
    B3Vec3 normal;
    B3Vec3 p_a[B3_MAX_MANIFOLD];
    B3Vec3 p_b[B3_MAX_MANIFOLD];
    float sep[B3_MAX_MANIFOLD];
    uint32_t feature[B3_MAX_MANIFOLD];
} B3Mani;

B3_HD B3_INL void b3_mani_clear(B3Mani* m) {
    m->count = 0;
    m->status = 0;
    m->gjk_iterations = 0;
    m->epa_iterations = 0;
    m->normal = b3_v(0.0f, 1.0f, 0.0f);
}

B3_HD B3_INL void b3_mani_push(B3Mani* m, B3Vec3 pa, B3Vec3 pb,
        float sep, uint32_t feat) {
    if (m->count >= B3_MAX_MANIFOLD) {
        return;
    }
    int i = m->count++;
    m->p_a[i] = pa;
    m->p_b[i] = pb;
    m->sep[i] = sep;
    m->feature[i] = feat;
}

B3_HD B3_INL void b3_collide_balls(B3Mani* m, B3Vec3 ca, float ra,
        B3Vec3 cb, float rb, uint32_t feat) {
    b3_mani_clear(m);
    B3Vec3 d = b3_sub(cb, ca);
    float dist2 = b3_len2(d);
    float rad = ra + rb;
    if (dist2 > rad * rad) {
        return;
    }
    float dist = sqrtf(dist2);
    B3Vec3 n = dist > 1.0e-8f ? b3_mul(d, b3_rsqrt(dist2))
        : b3_v(0.0f, 1.0f, 0.0f);
    m->normal = n;
    b3_mani_push(m, b3_madd(ca, ra, n), b3_msub(cb, rb, n),
        dist - rad, feat);
}

B3_HD B3_INL B3Vec3 b3_closest_obb(B3Vec3 c, B3Quat q, B3Vec3 half,
        B3Vec3 p, int* inside, int* face) {
    B3Vec3 local = b3_inv_rotate(q, b3_sub(p, c));
    B3Vec3 cl = b3_v(
        b3_clamp(local.x, -half.x, half.x),
        b3_clamp(local.y, -half.y, half.y),
        b3_clamp(local.z, -half.z, half.z));
    int in = fabsf(local.x) <= half.x
        && fabsf(local.y) <= half.y
        && fabsf(local.z) <= half.z;
    *inside = in;
    if (in) {
        float dx = half.x - fabsf(local.x);
        float dy = half.y - fabsf(local.y);
        float dz = half.z - fabsf(local.z);
        if (dx <= dy && dx <= dz) {
            cl.x = local.x >= 0.0f ? half.x : -half.x;
            *face = 0;
        } else if (dy <= dz) {
            cl.y = local.y >= 0.0f ? half.y : -half.y;
            *face = 1;
        } else {
            cl.z = local.z >= 0.0f ? half.z : -half.z;
            *face = 2;
        }
    } else {
        *face = -1;
    }
    return b3_xf_point(c, q, cl);
}

B3_HD B3_INL void b3_collide_ball_point(B3Mani* m, B3Vec3 p, float r,
        B3Vec3 q, int inside, B3Vec3 n_fb, uint32_t feat) {
    B3Vec3 d = b3_sub(p, q);
    float dist2 = b3_len2(d);
    if (!inside && dist2 > r * r) {
        return;
    }
    B3Vec3 n;
    float sep;
    if (inside) {
        n = dist2 > 1.0e-12f ? b3_norm(d) : n_fb;
        sep = -b3_len(d) - r;
    } else {
        float dist = sqrtf(dist2);
        n = b3_mul(d, b3_rsqrt(dist2));
        sep = dist - r;
    }
    m->normal = n;
    b3_mani_push(m, b3_msub(p, r, n), q, sep, feat);
}

B3_HD B3_INL void b3_collide_sphere_box(B3Mani* m, B3Vec3 sc, float r,
        B3Vec3 bc, B3Quat bq, B3Vec3 half) {
    b3_mani_clear(m);
    int inside;
    int face;
    B3Vec3 q = b3_closest_obb(bc, bq, half, sc, &inside, &face);
    B3Vec3 ax, ay, az;
    b3_axes(bq, &ax, &ay, &az);
    B3Vec3 n_fb = face == 0 ? ax : (face == 1 ? ay : az);
    b3_collide_ball_point(m, sc, r, q, inside, n_fb, 3u);
}

B3_HD B3_INL void b3_collide_capsule_box(B3Mani* m,
        B3Vec3 c1, B3Vec3 c2, float r, B3Vec3 bc, B3Quat bq, B3Vec3 half) {
    b3_mani_clear(m);
    B3Vec3 mid = b3_mul(b3_add(c1, c2), 0.5f);
    int inside;
    int face;
    B3Vec3 q = b3_closest_obb(bc, bq, half, mid, &inside, &face);
    B3Vec3 p = b3_closest_seg(c1, c2, q);
    q = b3_closest_obb(bc, bq, half, p, &inside, &face);
    p = b3_closest_seg(c1, c2, q);
    b3_collide_ball_point(m, p, r, q, inside, b3_v(0.0f, 1.0f, 0.0f), 5u);
}

typedef struct B3Obb {
    B3Vec3 c;
    B3Vec3 ax[3];
    B3Vec3 h;
} B3Obb;

B3_HD B3_INL B3Obb b3_obb(B3Vec3 c, B3Quat q, B3Vec3 h) {
    B3Obb o;
    o.c = c;
    o.h = h;
    b3_axes(q, &o.ax[0], &o.ax[1], &o.ax[2]);
    return o;
}

B3_HD B3_INL float b3_half_i(B3Vec3 h, int i) {
    return i == 0 ? h.x : (i == 1 ? h.y : h.z);
}

B3_HD B3_INL B3Vec3 b3_support(const B3Obb* o, B3Vec3 dir) {
    float sx = b3_dot(o->ax[0], dir) < 0.0f ? -o->h.x : o->h.x;
    float sy = b3_dot(o->ax[1], dir) < 0.0f ? -o->h.y : o->h.y;
    float sz = b3_dot(o->ax[2], dir) < 0.0f ? -o->h.z : o->h.z;
    return b3_add(o->c, b3_add(b3_mul(o->ax[0], sx),
        b3_add(b3_mul(o->ax[1], sy), b3_mul(o->ax[2], sz))));
}

B3_HD B3_INL int b3_clip_plane(const B3Vec3* in, int n, B3Vec3* out,
        B3Vec3 plane_n, float plane_o) {
    if (n <= 0) {
        return 0;
    }
    int c = 0;
    B3Vec3 prev = in[n - 1];
    float pd = b3_dot(prev, plane_n) - plane_o;
    int pin = pd <= 0.0f;
    for (int i = 0; i < n; i++) {
        B3Vec3 cur = in[i];
        float cd = b3_dot(cur, plane_n) - plane_o;
        int cin = cd <= 0.0f;
        if (cin != pin && c < 8) {
            float t = pd / (pd - cd);
            out[c++] = b3_madd(prev, t, b3_sub(cur, prev));
        }
        if (cin && c < 8) {
            out[c++] = cur;
        }
        prev = cur;
        pd = cd;
        pin = cin;
    }
    return c;
}

B3_HD B3_INL void b3_face_pts(const B3Obb* o, int axis, float sign,
        B3Vec3 out[4]) {
    int ta = (axis + 1) % 3;
    int tb = (axis + 2) % 3;
    B3Vec3 c = b3_madd(o->c, sign * b3_half_i(o->h, axis), o->ax[axis]);
    B3Vec3 a = b3_mul(o->ax[ta], b3_half_i(o->h, ta));
    B3Vec3 b = b3_mul(o->ax[tb], b3_half_i(o->h, tb));
    out[0] = b3_sub(b3_sub(c, a), b);
    out[1] = b3_add(b3_sub(c, a), b);
    out[2] = b3_add(b3_add(c, a), b);
    out[3] = b3_sub(b3_add(c, a), b);
}

B3_HD B3_INL int b3_best_axis(const B3Obb* o, B3Vec3 n) {
    float b0 = fabsf(b3_dot(o->ax[0], n));
    float b1 = fabsf(b3_dot(o->ax[1], n));
    float b2 = fabsf(b3_dot(o->ax[2], n));
    if (b0 >= b1 && b0 >= b2) {
        return 0;
    }
    return b1 >= b2 ? 1 : 2;
}

B3_HD B3_INL void b3_mani_support(B3Mani* m, const B3Obb* a, const B3Obb* b,
        B3Vec3 n, int feat) {
    B3Vec3 pa = b3_support(a, n);
    B3Vec3 pb = b3_support(b, b3_neg(n));
    b3_mani_push(m, pa, pb, b3_dot(b3_sub(pb, pa), n), (uint32_t)feat);
}

B3_HD B3_INL int b3_try_face(float s, B3Vec3 axis, float side,
        float* best, B3Vec3* n, int feat, int* out_feat) {
    if (s > B3_SPECULATIVE) {
        return -1;
    }
    if (s > *best) {
        *best = s;
        *n = side < 0.0f ? b3_neg(axis) : axis;
        *out_feat = feat;
    }
    return 0;
}

B3_HD B3_INL int b3_try_edge(float e, float ra, float rb, B3Vec3 cr,
        float* best, B3Vec3* n, int feat, int* out_feat) {
    float cl2 = b3_len2(cr);
    if (cl2 <= 1.0e-3f) {
        return 0;
    }
    float inv = b3_rsqrt(cl2);
    float s = (fabsf(e) - ra - rb) * inv;
    if (s > B3_SPECULATIVE && cl2 > 0.05f) {
        return -1;
    }
    if (s > B3_SPECULATIVE) {
        return 0;
    }
    if (s > *best + 0.002f) {
        *best = s;
        *n = e < 0.0f ? b3_mul(cr, -inv) : b3_mul(cr, inv);
        *out_feat = feat;
    }
    return 0;
}

B3_HD B3_INL void b3_collide_boxes(B3Mani* m, const B3Obb* a,
        const B3Obb* b) {
    b3_mani_clear(m);
    B3Vec3 d = b3_sub(b->c, a->c);
    float t0 = b3_dot(d, a->ax[0]);
    float t1 = b3_dot(d, a->ax[1]);
    float t2 = b3_dot(d, a->ax[2]);
    float r00 = b3_dot(a->ax[0], b->ax[0]);
    float r01 = b3_dot(a->ax[0], b->ax[1]);
    float r02 = b3_dot(a->ax[0], b->ax[2]);
    float r10 = b3_dot(a->ax[1], b->ax[0]);
    float r11 = b3_dot(a->ax[1], b->ax[1]);
    float r12 = b3_dot(a->ax[1], b->ax[2]);
    float r20 = b3_dot(a->ax[2], b->ax[0]);
    float r21 = b3_dot(a->ax[2], b->ax[1]);
    float r22 = b3_dot(a->ax[2], b->ax[2]);
    float ar00 = fabsf(r00);
    float ar01 = fabsf(r01);
    float ar02 = fabsf(r02);
    float ar10 = fabsf(r10);
    float ar11 = fabsf(r11);
    float ar12 = fabsf(r12);
    float ar20 = fabsf(r20);
    float ar21 = fabsf(r21);
    float ar22 = fabsf(r22);
    float ax = a->h.x;
    float ay = a->h.y;
    float az = a->h.z;
    float bx = b->h.x;
    float by = b->h.y;
    float bz = b->h.z;
    float best = -FLT_MAX;
    B3Vec3 n = a->ax[0];
    int feat = 0;

    if (b3_try_face(fabsf(t0) - ax - (bx * ar00 + by * ar01 + bz * ar02),
            a->ax[0], t0, &best, &n, 0, &feat) < 0) {
        return;
    }
    if (b3_try_face(fabsf(t1) - ay - (bx * ar10 + by * ar11 + bz * ar12),
            a->ax[1], t1, &best, &n, 1, &feat) < 0) {
        return;
    }
    if (b3_try_face(fabsf(t2) - az - (bx * ar20 + by * ar21 + bz * ar22),
            a->ax[2], t2, &best, &n, 2, &feat) < 0) {
        return;
    }
    float u0 = t0 * r00 + t1 * r10 + t2 * r20;
    float u1 = t0 * r01 + t1 * r11 + t2 * r21;
    float u2 = t0 * r02 + t1 * r12 + t2 * r22;
    if (b3_try_face(fabsf(u0) - bx - (ax * ar00 + ay * ar10 + az * ar20),
            b->ax[0], u0, &best, &n, 3, &feat) < 0) {
        return;
    }
    if (b3_try_face(fabsf(u1) - by - (ax * ar01 + ay * ar11 + az * ar21),
            b->ax[1], u1, &best, &n, 4, &feat) < 0) {
        return;
    }
    if (b3_try_face(fabsf(u2) - bz - (ax * ar02 + ay * ar12 + az * ar22),
            b->ax[2], u2, &best, &n, 5, &feat) < 0) {
        return;
    }

    if (b3_try_edge(t2 * r10 - t1 * r20,
            ay * ar20 + az * ar10, by * ar02 + bz * ar01,
            b3_cross(a->ax[0], b->ax[0]), &best, &n, 6, &feat) < 0) {
        return;
    }
    if (b3_try_edge(t2 * r11 - t1 * r21,
            ay * ar21 + az * ar11, bz * ar00 + bx * ar02,
            b3_cross(a->ax[0], b->ax[1]), &best, &n, 7, &feat) < 0) {
        return;
    }
    if (b3_try_edge(t2 * r12 - t1 * r22,
            ay * ar22 + az * ar12, bx * ar01 + by * ar00,
            b3_cross(a->ax[0], b->ax[2]), &best, &n, 8, &feat) < 0) {
        return;
    }
    if (b3_try_edge(t0 * r20 - t2 * r00,
            az * ar00 + ax * ar20, by * ar12 + bz * ar11,
            b3_cross(a->ax[1], b->ax[0]), &best, &n, 9, &feat) < 0) {
        return;
    }
    if (b3_try_edge(t0 * r21 - t2 * r01,
            az * ar01 + ax * ar21, bz * ar10 + bx * ar12,
            b3_cross(a->ax[1], b->ax[1]), &best, &n, 10, &feat) < 0) {
        return;
    }
    if (b3_try_edge(t0 * r22 - t2 * r02,
            az * ar02 + ax * ar22, bx * ar11 + by * ar10,
            b3_cross(a->ax[1], b->ax[2]), &best, &n, 11, &feat) < 0) {
        return;
    }
    if (b3_try_edge(t1 * r00 - t0 * r10,
            ax * ar10 + ay * ar00, by * ar22 + bz * ar21,
            b3_cross(a->ax[2], b->ax[0]), &best, &n, 12, &feat) < 0) {
        return;
    }
    if (b3_try_edge(t1 * r01 - t0 * r11,
            ax * ar11 + ay * ar01, bz * ar20 + bx * ar22,
            b3_cross(a->ax[2], b->ax[1]), &best, &n, 13, &feat) < 0) {
        return;
    }
    if (b3_try_edge(t1 * r02 - t0 * r12,
            ax * ar12 + ay * ar02, bx * ar21 + by * ar20,
            b3_cross(a->ax[2], b->ax[2]), &best, &n, 14, &feat) < 0) {
        return;
    }

    if (b3_dot(n, d) < 0.0f) {
        n = b3_neg(n);
    }
    m->normal = n;

    if (feat >= 6) {
        b3_mani_support(m, a, b, n, feat);
        return;
    }

    const B3Obb* ref = feat < 3 ? a : b;
    const B3Obb* inc = feat < 3 ? b : a;
    int ref_axis = feat < 3 ? feat : feat - 3;
    float ref_sign;
    if (feat < 3) {
        ref_sign = b3_dot(ref->ax[ref_axis], n) < 0.0f ? -1.0f : 1.0f;
    } else {
        ref_sign = b3_dot(ref->ax[ref_axis], n) > 0.0f ? -1.0f : 1.0f;
    }
    // The incident face opposes the reference face's outward normal.
    // When B is the reference, that normal is opposite the A-to-B normal n.
    B3Vec3 rn = b3_mul(ref->ax[ref_axis], ref_sign);
    int inc_axis = b3_best_axis(inc, rn);
    float inc_sign = b3_dot(inc->ax[inc_axis], rn) > 0.0f ? -1.0f : 1.0f;
    B3Vec3 poly[8];
    B3Vec3 tmp[8];
    b3_face_pts(inc, inc_axis, inc_sign, poly);
    int count = 4;
    int ta = (ref_axis + 1) % 3;
    int tb = (ref_axis + 2) % 3;
    B3Vec3 face_c = b3_madd(ref->c,
        ref_sign * b3_half_i(ref->h, ref_axis), ref->ax[ref_axis]);
    B3Vec3 clip_n[4];
    float clip_o[4];
    clip_n[0] = b3_neg(ref->ax[ta]);
    clip_n[1] = ref->ax[ta];
    clip_n[2] = b3_neg(ref->ax[tb]);
    clip_n[3] = ref->ax[tb];
    clip_o[0] = b3_dot(clip_n[0], ref->c)
        + b3_half_i(ref->h, ta);
    clip_o[1] = b3_dot(clip_n[1], ref->c)
        + b3_half_i(ref->h, ta);
    clip_o[2] = b3_dot(clip_n[2], ref->c)
        + b3_half_i(ref->h, tb);
    clip_o[3] = b3_dot(clip_n[3], ref->c)
        + b3_half_i(ref->h, tb);
    for (int i = 0; i < 4; i++) {
        count = b3_clip_plane(poly, count, tmp, clip_n[i], clip_o[i]);
        for (int k = 0; k < count; k++) {
            poly[k] = tmp[k];
        }
        if (count == 0) {
            b3_mani_support(m, a, b, n, feat);
            return;
        }
    }
    float ro = b3_dot(rn, face_c);
    for (int i = 0; i < count && m->count < B3_MAX_MANIFOLD; i++) {
        float plane_s = b3_dot(poly[i], rn) - ro;
        if (plane_s <= B3_SPECULATIVE) {
            B3Vec3 on_ref = b3_msub(poly[i], plane_s, rn);
            B3Vec3 pa = feat < 3 ? on_ref : poly[i];
            B3Vec3 pb = feat < 3 ? poly[i] : on_ref;
            float sep = b3_dot(b3_sub(pb, pa), n);
            b3_mani_push(m, pa, pb, sep,
                (feat < 3 ? 16u : 20u) + (uint32_t)i);
        }
    }
    if (m->count == 0) {
        b3_mani_support(m, a, b, n, feat);
    }
}

B3_HD B3_INL void b3_mani_flip(B3Mani* m) {
    m->normal = b3_neg(m->normal);
    for (int i = 0; i < m->count; i++) {
        B3Vec3 t = m->p_a[i];
        m->p_a[i] = m->p_b[i];
        m->p_b[i] = t;
    }
}

#include "b3_convex.cuh"

B3_HD B3_INL void b3_collide_pair(B3Mani* m, const B3Body* ba, const B3Shape* sa,
        const B3Body* bb, const B3Shape* sb) {
    if (sa->type == B3_CYLINDER || sb->type == B3_CYLINDER) {
        b3_collide_convex(m, ba, sa, bb, sb);
        return;
    }
    B3Vec3 pa = b3_shape_pos(ba, sa);
    B3Vec3 pb = b3_shape_pos(bb, sb);
    B3Vec3 center_a = pa;
    B3Vec3 center_b = pb;
    B3Quat qa = b3_shape_rot(ba, sa);
    B3Quat qb = b3_shape_rot(bb, sb);
    int ta = sa->type;
    int tb = sb->type;
    int flip = 0;
    if (ta > tb) {
        const B3Shape* t = sa;
        sa = sb;
        sb = t;
        B3Vec3 tp = pa;
        pa = pb;
        pb = tp;
        B3Quat tq = qa;
        qa = qb;
        qb = tq;
        int tt = ta;
        ta = tb;
        tb = tt;
        flip = 1;
    }
    if (ta == B3_SPHERE && tb == B3_SPHERE) {
        b3_collide_balls(m, pa, sa->radius, pb, sb->radius, 1u);
    } else if (ta == B3_SPHERE && tb == B3_CAPSULE) {
        B3Vec3 y = b3_rotate(qb, b3_v(0.0f, sb->half.y, 0.0f));
        b3_collide_balls(m, b3_closest_seg(b3_sub(pb, y), b3_add(pb, y),
            pa), sb->radius, pa, sa->radius, 4u);
        b3_mani_flip(m);
    } else if (ta == B3_SPHERE && tb == B3_BOX) {
        b3_collide_sphere_box(m, pa, sa->radius, pb, qb, sb->half);
    } else if (ta == B3_CAPSULE && tb == B3_CAPSULE) {
        B3Vec3 ya = b3_rotate(qa, b3_v(0.0f, sa->half.y, 0.0f));
        B3Vec3 yb = b3_rotate(qb, b3_v(0.0f, sb->half.y, 0.0f));
        B3Vec3 ca, cb;
        b3_closest_segs(b3_sub(pa, ya), b3_add(pa, ya),
            b3_sub(pb, yb), b3_add(pb, yb), &ca, &cb);
        b3_collide_balls(m, ca, sa->radius, cb, sb->radius, 2u);
    } else if (ta == B3_CAPSULE && tb == B3_BOX) {
        B3Vec3 y = b3_rotate(qa, b3_v(0.0f, sa->half.y, 0.0f));
        b3_collide_capsule_box(m, b3_sub(pa, y), b3_add(pa, y),
            sa->radius, pb, qb, sb->half);
    } else {
        B3Obb oa = b3_obb(pa, qa, sa->half);
        B3Obb ob = b3_obb(pb, qb, sb->half);
        b3_collide_boxes(m, &oa, &ob);
    }
    if (flip && m->count > 0) {
        b3_mani_flip(m);
    }
    if (m->count > 0) {
        if (b3_dot(m->normal, b3_sub(center_b, center_a)) < 0.0f) {
            m->normal = b3_neg(m->normal);
        }
        for (int i = 0; i < m->count; i++) {
            m->sep[i] = b3_dot(b3_sub(m->p_b[i], m->p_a[i]), m->normal);
        }
    }
}

B3_HD B3_INL void b3_collide_shapes(B3Mani* m, const B3World* w,
        const B3Shape* sa, const B3Shape* sb) {
    b3_collide_pair(m, &w->bodies[sa->body], sa, &w->bodies[sb->body], sb);
}

B3_HD B3_INL B3Joint* b3_add_joint(B3World* w, int type, int body_a,
        int body_b, B3Vec3 anchor_a, B3Vec3 anchor_b, B3Quat rot_a,
        B3Quat rot_b) {
    assert(w->joint_count < B3_MAX_JOINTS);
    int id = w->joint_count++;
    B3Joint* j = &w->joints[id];
    memset(j, 0, sizeof(*j));
#ifndef B3_REVOLUTE_ONLY
    j->type = type;
#endif
    j->body_a = body_a;
    j->body_b = body_b;
    j->local_anchor_a = anchor_a;
    j->local_anchor_b = anchor_b;
    j->local_rot_a = b3_qnorm(rot_a);
    j->local_rot_b = b3_qnorm(rot_b);
    j->constraint_hertz = 90.0f;
    j->constraint_damping = 2.0f;
    return j;
}

#ifdef B3_REVOLUTE_ONLY
B3_HD B3_INL int b3_create_weld(B3World* w, int body_a, int body_b,
        B3Vec3 local_anchor_a, B3Vec3 local_anchor_b) {
    return -1;
}
#else
B3_HD B3_INL int b3_create_weld(B3World* w, int body_a, int body_b,
        B3Vec3 local_anchor_a, B3Vec3 local_anchor_b) {
    const B3Body* ba = &w->bodies[body_a];
    const B3Body* bb = &w->bodies[body_b];
    B3Quat rot_a = b3_q_id();
    B3Quat rot_b = b3_qinv_mul(bb->rotation, ba->rotation);
    B3Joint* j = b3_add_joint(w, B3_JOINT_WELD, body_a, body_b,
        local_anchor_a, local_anchor_b, rot_a, rot_b);
    return (int)(j - w->joints);
}

#endif

B3_HD B3_INL int b3_create_revolute(B3World* w, int body_a, int body_b,
        B3Vec3 local_anchor_a, B3Vec3 local_anchor_b, B3Vec3 local_axis_a) {
    const B3Body* ba = &w->bodies[body_a];
    const B3Body* bb = &w->bodies[body_b];
    B3Quat rot_a = b3_q_from_z(local_axis_a);
    B3Quat world = b3_qmul(ba->rotation, rot_a);
    B3Quat rot_b = b3_qinv_mul(bb->rotation, world);
    B3Joint* j = b3_add_joint(w, B3_JOINT_REVOLUTE, body_a, body_b,
        local_anchor_a, local_anchor_b, rot_a, rot_b);
    return (int)(j - w->joints);
}

B3_HD B3_INL void b3_joint_enable_motor(B3World* w, int joint, int enable) {
#ifndef B3_REVOLUTE_ONLY
    B3Joint* j = &w->joints[joint];
    if (j->enable_motor != enable) {
        j->motor_impulse = 0.0f;
    }
    j->enable_motor = enable;
#endif
}

B3_HD B3_INL void b3_joint_set_motor(B3World* w, int joint, float speed,
        float max_torque) {
    B3Joint* j = &w->joints[joint];
#ifndef B3_REVOLUTE_ONLY
    j->motor_speed = speed;
#endif
    j->max_motor_torque = b3_maxf(max_torque, 0.0f);
}

B3_HD B3_INL void b3_joint_enable_limit(B3World* w, int joint, int enable) {
    B3Joint* j = &w->joints[joint];
    if (j->enable_limit != enable) {
        j->lower_impulse = 0.0f;
        j->upper_impulse = 0.0f;
    }
    j->enable_limit = enable;
}

B3_HD B3_INL void b3_joint_set_limits(B3World* w, int joint, float lower,
        float upper) {
    B3Joint* j = &w->joints[joint];
    float lo = b3_minf(lower, upper);
    float hi = b3_maxf(lower, upper);
    j->lower_angle = b3_clamp(lo, -0.99f * B3_PI, 0.99f * B3_PI);
    j->upper_angle = b3_clamp(hi, -0.99f * B3_PI, 0.99f * B3_PI);
}

B3_HD B3_INL void b3_joint_enable_spring(B3World* w, int joint, int enable) {
    B3Joint* j = &w->joints[joint];
    if (j->enable_spring != enable) {
        j->spring_impulse = 0.0f;
    }
    j->enable_spring = enable;
}

B3_HD B3_INL void b3_joint_set_spring(B3World* w, int joint, float target,
        float hertz, float damping) {
    B3Joint* j = &w->joints[joint];
    j->target_angle = b3_clamp(target, -B3_PI, B3_PI);
    j->hertz = b3_maxf(hertz, 0.0f);
    j->damping_ratio = b3_maxf(damping, 0.0f);
}

B3_HD B3_INL float b3_joint_angle(const B3World* w, int joint) {
    const B3Joint* j = &w->joints[joint];
    const B3Body* ba = &w->bodies[j->body_a];
    const B3Body* bb = &w->bodies[j->body_b];
    B3Quat qa = b3_qmul(ba->rotation, j->local_rot_a);
    B3Quat qb = b3_qmul(bb->rotation, j->local_rot_b);
    if (b3_qdot(qa, qb) < 0.0f) {
        qb = b3_qneg(qb);
    }
    return b3_twist(b3_qinv_mul(qa, qb));
}

B3_HD B3_INL float b3_joint_speed(const B3World* w, int joint) {
    const B3Joint* j = &w->joints[joint];
    const B3Body* ba = &w->bodies[j->body_a];
    const B3Body* bb = &w->bodies[j->body_b];
    B3Vec3 axis = b3_rotate(b3_qmul(ba->rotation, j->local_rot_a),
        b3_v(0.0f, 0.0f, 1.0f));
    return b3_dot(b3_sub(bb->ang_vel, ba->ang_vel), axis);
}

B3_HD B3_INL void b3_prepare_joints(B3World* w, float h) {
    for (int i = 0; i < w->joint_count; i++) {
        B3Joint* j = &w->joints[i];
        const B3Body* ba = &w->bodies[j->body_a];
        const B3Body* bb = &w->bodies[j->body_b];
        j->inv_mass_a = ba->type == B3_DYNAMIC ? ba->inv_mass : 0.0f;
        j->inv_mass_b = bb->type == B3_DYNAMIC ? bb->inv_mass : 0.0f;
        j->inv_i_a = ba->type == B3_DYNAMIC ? ba->inv_i_world : b3_mat0();
        j->inv_i_b = bb->type == B3_DYNAMIC ? bb->inv_i_world : b3_mat0();
        B3Mat3 isum = b3_maddm(j->inv_i_a, j->inv_i_b);
        j->fixed_rotation = b3_dot(isum.cx, b3_cross(isum.cy, isum.cz))
            < 1.0e-20f;
        j->softness = b3_make_soft(j->constraint_hertz,
            j->constraint_damping, h);
        j->frame_q_a = b3_qmul(ba->rotation, j->local_rot_a);
        j->frame_q_b = b3_qmul(bb->rotation, j->local_rot_b);
        j->frame_p_a = b3_rotate(ba->rotation,
            b3_sub(j->local_anchor_a, ba->local_center));
        j->frame_p_b = b3_rotate(bb->rotation,
            b3_sub(j->local_anchor_b, bb->local_center));
        j->delta_center = b3_sub(bb->center, ba->center);
#ifdef B3_REVOLUTE_ONLY
        {
#else
        if (j->type == B3_JOINT_WELD) {
            j->angular_mass = b3_invert3(isum);
            j->linear_spring = j->linear_hertz == 0.0f ? j->softness
                : b3_make_soft(j->linear_hertz, j->linear_damping, h);
            j->angular_spring = j->angular_hertz == 0.0f ? j->softness
                : b3_make_soft(j->angular_hertz, j->angular_damping, h);
        } else {
#endif
            B3Vec3 axis = b3_rotate(j->frame_q_a, b3_v(0.0f, 0.0f, 1.0f));
            float k = b3_dot(axis, b3_mv(isum, axis));
            j->axial_mass = k > 0.0f ? 1.0f / k : 0.0f;
            j->rotation_axis = axis;
            B3Quat rel = b3_qinv_mul(j->frame_q_a, j->frame_q_b);
            b3_hinge_perps(j->frame_q_a, rel, &j->perp_x, &j->perp_y);
            j->spring_softness = b3_make_soft(j->hertz, j->damping_ratio, h);
        }
    }
}

B3_HD B3_INL void b3_warm_start_joints(B3World* w) {
    for (int i = 0; i < w->joint_count; i++) {
        B3Joint* j = &w->joints[i];
        B3Body* ba = &w->bodies[j->body_a];
        B3Body* bb = &w->bodies[j->body_b];
        B3Vec3 ra = b3_rotate(ba->delta_rot, j->frame_p_a);
        B3Vec3 rb = b3_rotate(bb->delta_rot, j->frame_p_b);
#ifdef B3_REVOLUTE_ONLY
        B3Vec3 ang = b3_v(0.0f, 0.0f, 0.0f);
        {
            float axial = j->spring_impulse
                + j->lower_impulse - j->upper_impulse;
#else
        B3Vec3 ang = j->angular_impulse;
        if (j->type == B3_JOINT_REVOLUTE) {
            float axial = j->spring_impulse + j->motor_impulse
                + j->lower_impulse - j->upper_impulse;
#endif
            ang = b3_add(b3_mul(j->perp_x, j->perp_impulse.x),
                b3_mul(j->perp_y, j->perp_impulse.y));
            ang = b3_madd(ang, axial, j->rotation_axis);
        }
        if (ba->flags & B3_FLAG_DYNAMIC) {
            ba->lin_vel = b3_msub(ba->lin_vel, j->inv_mass_a,
                j->linear_impulse);
            ba->ang_vel = b3_sub(ba->ang_vel, b3_mv(j->inv_i_a,
                b3_add(b3_cross(ra, j->linear_impulse), ang)));
        }
        if (bb->flags & B3_FLAG_DYNAMIC) {
            bb->lin_vel = b3_madd(bb->lin_vel, j->inv_mass_b,
                j->linear_impulse);
            bb->ang_vel = b3_add(bb->ang_vel, b3_mv(j->inv_i_b,
                b3_add(b3_cross(rb, j->linear_impulse), ang)));
        }
    }
}

B3_HD B3_INL B3Mat3 b3_point_k_gs(float ma, float mb, B3Mat3 ia, B3Mat3 ib,
        B3Vec3 ra, B3Vec3 rb) {
    float msum = ma + mb;
    B3Mat3 k;
    k.cx = b3_v(msum, 0.0f, 0.0f);
    k.cy = b3_v(0.0f, msum, 0.0f);
    k.cz = b3_v(0.0f, 0.0f, msum);
    k.cx = b3_sub(k.cx, b3_cross(ra, b3_mv(ia, b3_v(0.0f, ra.z, -ra.y))));
    k.cy = b3_sub(k.cy, b3_cross(ra, b3_mv(ia, b3_v(-ra.z, 0.0f, ra.x))));
    k.cz = b3_sub(k.cz, b3_cross(ra, b3_mv(ia, b3_v(ra.y, -ra.x, 0.0f))));
    k.cx = b3_sub(k.cx, b3_cross(rb, b3_mv(ib, b3_v(0.0f, rb.z, -rb.y))));
    k.cy = b3_sub(k.cy, b3_cross(rb, b3_mv(ib, b3_v(-rb.z, 0.0f, rb.x))));
    k.cz = b3_sub(k.cz, b3_cross(rb, b3_mv(ib, b3_v(rb.y, -rb.x, 0.0f))));
    return k;
}

B3_HD B3_INL void b3_solve_point(B3Joint* j, B3Body* ba, B3Body* bb,
        B3Vec3* va, B3Vec3* wa, B3Vec3* vb, B3Vec3* wb, B3Soft soft,
        int use_bias) {
    B3Vec3 ra = b3_rotate(ba->delta_rot, j->frame_p_a);
    B3Vec3 rb = b3_rotate(bb->delta_rot, j->frame_p_b);
    B3Vec3 cdot = b3_sub(b3_add(*vb, b3_cross(*wb, rb)),
        b3_add(*va, b3_cross(*wa, ra)));
    B3Vec3 bias = b3_v(0.0f, 0.0f, 0.0f);
    float mscale = 1.0f;
    float iscale = 0.0f;
    if (use_bias) {
        B3Vec3 sep = b3_add(b3_add(b3_sub(bb->delta_pos, ba->delta_pos),
            b3_sub(rb, ra)), j->delta_center);
        bias = b3_mul(sep, soft.bias_rate);
        mscale = soft.mass_scale;
        iscale = soft.impulse_scale;
    }
    B3Vec3 b = b3_solve3(b3_point_k_gs(j->inv_mass_a, j->inv_mass_b,
        j->inv_i_a, j->inv_i_b, ra, rb), b3_add(cdot, bias));
    B3Vec3 impulse = b3_msub(b3_mul(b, -mscale), iscale, j->linear_impulse);
    j->linear_impulse = b3_add(j->linear_impulse, impulse);
    *va = b3_msub(*va, j->inv_mass_a, impulse);
    *wa = b3_sub(*wa, b3_mv(j->inv_i_a, b3_cross(ra, impulse)));
    *vb = b3_madd(*vb, j->inv_mass_b, impulse);
    *wb = b3_add(*wb, b3_mv(j->inv_i_b, b3_cross(rb, impulse)));
}

#ifndef B3_REVOLUTE_ONLY
B3_HD B3_INL void b3_solve_weld(B3Joint* j, B3Body* ba, B3Body* bb,
        int use_bias) {
    B3Vec3 va = ba->lin_vel;
    B3Vec3 wa = ba->ang_vel;
    B3Vec3 vb = bb->lin_vel;
    B3Vec3 wb = bb->ang_vel;
    B3Quat qa = b3_qmul(ba->delta_rot, j->frame_q_a);
    B3Quat qb = b3_qmul(bb->delta_rot, j->frame_q_b);
    if (b3_qdot(qa, qb) < 0.0f) {
        qb = b3_qneg(qb);
    }
    B3Quat rel = b3_qinv_mul(qa, qb);
    if (!j->fixed_rotation) {
        B3Vec3 bias = b3_v(0.0f, 0.0f, 0.0f);
        float mscale = 1.0f;
        float iscale = 0.0f;
        if (use_bias || j->angular_hertz > 0.0f) {
            B3Quat s = rel;
            if (rel.s < 0.0f) {
                s = b3_qneg(rel);
            }
            B3Quat diff = b3_q(-s.v.x, -s.v.y, -s.v.z, 1.0f - s.s);
            B3Vec3 c = b3_neg(b3_rotate(qa,
                b3_mul(b3_qmul(diff, b3_qconj(s)).v, 2.0f)));
            bias = b3_mul(c, j->angular_spring.bias_rate);
            mscale = j->angular_spring.mass_scale;
            iscale = j->angular_spring.impulse_scale;
        }
        B3Vec3 cdot = b3_sub(wb, wa);
        B3Vec3 impulse = b3_msub(
            b3_mul(b3_mv(j->angular_mass, b3_add(cdot, bias)), -mscale),
            iscale, j->angular_impulse);
        j->angular_impulse = b3_add(j->angular_impulse, impulse);
        wa = b3_sub(wa, b3_mv(j->inv_i_a, impulse));
        wb = b3_add(wb, b3_mv(j->inv_i_b, impulse));
    }
    int lin_bias = use_bias || j->linear_hertz > 0.0f;
    b3_solve_point(j, ba, bb, &va, &wa, &vb, &wb,
        j->linear_spring, lin_bias);
    if (ba->flags & B3_FLAG_DYNAMIC) {
        ba->lin_vel = va;
        ba->ang_vel = wa;
    }
    if (bb->flags & B3_FLAG_DYNAMIC) {
        bb->lin_vel = vb;
        bb->ang_vel = wb;
    }
}

#endif

#ifdef B3_CACHE_JOINTS
B3_HD B3_INL void b3_cache_revolute(B3Joint* j, const B3Body* ba,
        const B3Body* bb) {
    B3Vec3 ra = b3_rotate(ba->delta_rot, j->frame_p_a);
    B3Vec3 rb = b3_rotate(bb->delta_rot, j->frame_p_b);
    j->cache_ra = ra;
    j->cache_rb = rb;
    j->cache_point_invk = b3_invert3(b3_point_k_gs(j->inv_mass_a,
        j->inv_mass_b, j->inv_i_a, j->inv_i_b, ra, rb));
    B3Quat qa = b3_qmul(ba->delta_rot, j->frame_q_a);
    B3Quat qb = b3_qmul(bb->delta_rot, j->frame_q_b);
    if (b3_qdot(qa, qb) < 0.0f) {
        qb = b3_qneg(qb);
    }
    B3Quat rel = b3_qinv_mul(qa, qb);
    j->cache_rel_x = rel.v.x;
    j->cache_rel_y = rel.v.y;
    j->cache_twist = b3_twist(rel);
    if (j->fixed_rotation) {
        j->cache_ang_invk.cx.x = 0.0f;
        j->cache_ang_invk.cx.y = 0.0f;
        j->cache_ang_invk.cy.x = 0.0f;
        j->cache_ang_invk.cy.y = 0.0f;
        j->cache_ia_ax = b3_v(0.0f, 0.0f, 0.0f);
        j->cache_ib_ax = b3_v(0.0f, 0.0f, 0.0f);
        return;
    }
    B3Vec3 px, py;
    b3_hinge_perps(qa, rel, &px, &py);
    j->perp_x = px;
    j->perp_y = py;
    B3Mat3 isum = b3_maddm(j->inv_i_a, j->inv_i_b);
    B3Mat2 k;
    k.cx.x = b3_dot(px, b3_mv(isum, px));
    k.cy.y = b3_dot(py, b3_mv(isum, py));
    k.cx.y = k.cy.x = b3_dot(px, b3_mv(isum, py));
    j->cache_ang_invk = b3_invert2(k);
    j->cache_ia_ax = b3_mv(j->inv_i_a, j->rotation_axis);
    j->cache_ib_ax = b3_mv(j->inv_i_b, j->rotation_axis);
}
#endif

B3_HD B3_INL void b3_solve_revolute(B3Joint* j, B3Body* ba, B3Body* bb,
        float h, float inv_h, int use_bias) {
    B3Vec3 va = ba->lin_vel;
    B3Vec3 wa = ba->ang_vel;
    B3Vec3 vb = bb->lin_vel;
    B3Vec3 wb = bb->ang_vel;
#ifdef B3_CACHE_JOINTS
    B3Vec3 axis = j->rotation_axis;
#ifdef B3_ABLATE_NO_IA_CACHE
    B3Vec3 ia_ax = b3_v(0.0f, 0.0f, 0.0f);
    B3Vec3 ib_ax = b3_v(0.0f, 0.0f, 0.0f);
    if (!j->fixed_rotation) {
        ia_ax = b3_mv(j->inv_i_a, axis);
        ib_ax = b3_mv(j->inv_i_b, axis);
    }
#else
    B3Vec3 ia_ax = j->cache_ia_ax;
    B3Vec3 ib_ax = j->cache_ib_ax;
#endif
    float twist = j->cache_twist;
#else
    B3Quat qa = b3_qmul(ba->delta_rot, j->frame_q_a);
    B3Quat qb = b3_qmul(bb->delta_rot, j->frame_q_b);
    if (b3_qdot(qa, qb) < 0.0f) {
        qb = b3_qneg(qb);
    }
    B3Quat rel = b3_qinv_mul(qa, qb);
    B3Vec3 axis = j->rotation_axis;
    B3Vec3 ia_ax = b3_v(0.0f, 0.0f, 0.0f);
    B3Vec3 ib_ax = b3_v(0.0f, 0.0f, 0.0f);
    float twist = 0.0f;
    int need_twist = (j->enable_spring || j->enable_limit
#ifndef B3_REVOLUTE_ONLY
            || j->enable_motor
#endif
            ) && !j->fixed_rotation;
    if (!j->fixed_rotation) {
        ia_ax = b3_mv(j->inv_i_a, axis);
        ib_ax = b3_mv(j->inv_i_b, axis);
    }
    if (need_twist) {
        twist = b3_twist(rel);
    }
#endif
#ifndef B3_ABLATE_NO_SPRING
    if (j->enable_spring && !j->fixed_rotation) {
        float c = twist - j->target_angle;
        float bias = j->spring_softness.bias_rate * c;
        float mscale = j->spring_softness.mass_scale;
        float iscale = j->spring_softness.impulse_scale;
        float cdot = b3_dot(b3_sub(wb, wa), axis);
        float dimp = -mscale * j->axial_mass * (cdot + bias)
            - iscale * j->spring_impulse;
        float old = j->spring_impulse;
        float nimp = old + dimp;
        if (j->max_motor_torque > 0.0f) {
            float maxp = j->max_motor_torque * h;
            nimp = b3_clamp(nimp, -maxp, maxp);
        }
        dimp = nimp - old;
        j->spring_impulse = nimp;
        wa = b3_msub(wa, dimp, ia_ax);
        wb = b3_madd(wb, dimp, ib_ax);
    }
#endif
#ifndef B3_REVOLUTE_ONLY
    if (j->enable_motor && !j->fixed_rotation) {
        int blocked = 0;
        if (j->enable_limit) {
            float ang = twist;
            if (ang >= j->upper_angle - 0.01f && j->motor_speed > 0.0f) {
                blocked = 1;
            }
            if (ang <= j->lower_angle + 0.01f && j->motor_speed < 0.0f) {
                blocked = 1;
            }
        }
        if (blocked) {
            j->motor_impulse = 0.0f;
        } else {
        float cdot = b3_dot(b3_sub(wb, wa), axis) - j->motor_speed;
        float dimp = -j->axial_mass * cdot;
        float nimp = j->motor_impulse + dimp;
        float maxp = j->max_motor_torque * h;
        nimp = b3_clamp(nimp, -maxp, maxp);
        dimp = nimp - j->motor_impulse;
        j->motor_impulse = nimp;
        wa = b3_msub(wa, dimp, ia_ax);
        wb = b3_madd(wb, dimp, ib_ax);
        }
    }
#endif
#ifndef B3_ABLATE_NO_LIMIT
    if (j->enable_limit && !j->fixed_rotation) {
        float angle = twist;
        {
            float c = angle - j->lower_angle;
            float bias = 0.0f;
            float mscale = 1.0f;
            float iscale = 0.0f;
            if (c > 0.0f) {
                bias = c * inv_h;
            } else if (use_bias) {
                bias = j->softness.bias_rate * c;
                mscale = j->softness.mass_scale;
                iscale = j->softness.impulse_scale;
            }
            float cdot = b3_dot(b3_sub(wb, wa), axis);
            float old = j->lower_impulse;
            float dimp = -mscale * j->axial_mass * (cdot + bias)
                - iscale * old;
            j->lower_impulse = b3_maxf(old + dimp, 0.0f);
            dimp = j->lower_impulse - old;
            wa = b3_msub(wa, dimp, ia_ax);
            wb = b3_madd(wb, dimp, ib_ax);
        }
        {
            float c = j->upper_angle - angle;
            float bias = 0.0f;
            float mscale = 1.0f;
            float iscale = 0.0f;
            if (c > 0.0f) {
                bias = c * inv_h;
            } else if (use_bias) {
                bias = j->softness.bias_rate * c;
                mscale = j->softness.mass_scale;
                iscale = j->softness.impulse_scale;
            }
            float cdot = b3_dot(b3_sub(wa, wb), axis);
            float old = j->upper_impulse;
            float dimp = -mscale * j->axial_mass * (cdot + bias)
                - iscale * old;
            j->upper_impulse = b3_maxf(old + dimp, 0.0f);
            dimp = j->upper_impulse - old;
            wa = b3_madd(wa, dimp, ia_ax);
            wb = b3_msub(wb, dimp, ib_ax);
        }
    }
#endif
#ifndef B3_ABLATE_NO_PERP
    if (!j->fixed_rotation) {
        B3Vec2 bias;
        bias.x = 0.0f;
        bias.y = 0.0f;
        float mscale = 1.0f;
        float iscale = 0.0f;
        if (use_bias) {
#ifdef B3_CACHE_JOINTS
            bias.x = j->softness.bias_rate * j->cache_rel_x;
            bias.y = j->softness.bias_rate * j->cache_rel_y;
#else
            bias.x = j->softness.bias_rate * rel.v.x;
            bias.y = j->softness.bias_rate * rel.v.y;
#endif
            mscale = j->softness.mass_scale;
            iscale = j->softness.impulse_scale;
        }
#ifdef B3_CACHE_JOINTS
        B3Vec3 px = j->perp_x;
        B3Vec3 py = j->perp_y;
        B3Mat2 kang = j->cache_ang_invk;
#else
        B3Vec3 px, py;
        b3_hinge_perps(qa, rel, &px, &py);
        j->perp_x = px;
        j->perp_y = py;
        B3Mat3 isum = b3_maddm(j->inv_i_a, j->inv_i_b);
        B3Mat2 kang;
        kang.cx.x = b3_dot(px, b3_mv(isum, px));
        kang.cy.y = b3_dot(py, b3_mv(isum, py));
        kang.cx.y = kang.cy.x = b3_dot(px, b3_mv(isum, py));
#endif
        B3Vec3 wrel = b3_sub(wb, wa);
        B3Vec2 rhs;
        rhs.x = b3_dot(wrel, px) + bias.x;
        rhs.y = b3_dot(wrel, py) + bias.y;
#ifdef B3_CACHE_JOINTS
        B3Vec2 sol = b3_mv2(kang, rhs);
#else
        B3Vec2 sol = b3_solve2(kang, rhs);
#endif
        B3Vec2 old = j->perp_impulse;
        B3Vec2 dimp;
        dimp.x = -mscale * sol.x - iscale * old.x;
        dimp.y = -mscale * sol.y - iscale * old.y;
        j->perp_impulse.x += dimp.x;
        j->perp_impulse.y += dimp.y;
        B3Vec3 ang = b3_add(b3_mul(px, dimp.x), b3_mul(py, dimp.y));
        wa = b3_sub(wa, b3_mv(j->inv_i_a, ang));
        wb = b3_add(wb, b3_mv(j->inv_i_b, ang));
    }
#endif
#ifndef B3_ABLATE_NO_POINT
#ifdef B3_CACHE_JOINTS
    {
        B3Vec3 ra = j->cache_ra;
        B3Vec3 rb = j->cache_rb;
        B3Vec3 cdot = b3_sub(b3_add(vb, b3_cross(wb, rb)),
            b3_add(va, b3_cross(wa, ra)));
        B3Vec3 bias = b3_v(0.0f, 0.0f, 0.0f);
        float mscale = 1.0f;
        float iscale = 0.0f;
        if (use_bias) {
            B3Vec3 sep = b3_add(b3_add(b3_sub(bb->delta_pos, ba->delta_pos),
                b3_sub(rb, ra)), j->delta_center);
            bias = b3_mul(sep, j->softness.bias_rate);
            mscale = j->softness.mass_scale;
            iscale = j->softness.impulse_scale;
        }
        B3Vec3 rhs = b3_mv(j->cache_point_invk, b3_add(cdot, bias));
        B3Vec3 impulse = b3_msub(b3_mul(rhs, -mscale), iscale,
            j->linear_impulse);
        j->linear_impulse = b3_add(j->linear_impulse, impulse);
        va = b3_msub(va, j->inv_mass_a, impulse);
        wa = b3_sub(wa, b3_mv(j->inv_i_a, b3_cross(ra, impulse)));
        vb = b3_madd(vb, j->inv_mass_b, impulse);
        wb = b3_add(wb, b3_mv(j->inv_i_b, b3_cross(rb, impulse)));
    }
#else
    b3_solve_point(j, ba, bb, &va, &wa, &vb, &wb, j->softness, use_bias);
#endif
#endif
    if (ba->flags & B3_FLAG_DYNAMIC) {
        ba->lin_vel = va;
        ba->ang_vel = wa;
    }
    if (bb->flags & B3_FLAG_DYNAMIC) {
        bb->lin_vel = vb;
        bb->ang_vel = wb;
    }
}

B3_HD B3_INL void b3_prepare_one_contact(B3Contact* c, const B3Body* ba,
        const B3Body* bb, B3Soft contact_s, B3Soft static_s);
B3_HD B3_INL void b3_warm_one_contact(B3Contact* c, B3Body* ba, B3Body* bb);
B3_HD B3_INL void b3_solve_one_contact(B3Contact* c, B3Body* ba, B3Body* bb,
        float inv_h, float contact_speed, int use_bias);
B3_HD B3_INL void b3_solve_contacts_n(B3Contact* contacts, int n,
        B3Body* bodies, float inv_h, float contact_speed, int use_bias);

B3_HD B3_INL void b3_solve_contacts(B3World* w, float inv_h,
        float contact_speed, int use_bias);
B3_HD B3_INL void b3_integrate_position_state(const B3Body* b, float h,
        float max_lin, float max_ang, float max_lin2, float max_ang2,
        B3Vec3* lin_vel, B3Vec3* ang_vel, B3Vec3* delta_pos, B3Quat* delta_rot);
#if B3_ART_CONTACTS
#ifdef RP_USE_ART_CACHE
#include "b3_art_cached.cuh"
#else
#include "b3_art.cuh"
#endif
#endif

/* Rolling is a torque-only disk in the contact tangent plane. Solve the
 * projected 2x2 inertia so locked normal-axis rotation is harmless. */
B3_HD B3_INL void b3_solve_rolling(B3Vec3 t1, B3Vec3 t2, float rolling,
        float total_n, B3Mat3 ia, B3Mat3 ib, B3Vec3* impulse,
        B3Vec3* wa, B3Vec3* wb) {
    if (rolling <= 0.0f) return;
    B3Mat3 isum = b3_maddm(ia, ib);
    B3Mat2 k;
    k.cx.x = b3_dot(t1, b3_mv(isum, t1));
    k.cy.y = b3_dot(t2, b3_mv(isum, t2));
    k.cx.y = k.cy.x = b3_dot(t1, b3_mv(isum, t2));
    B3Vec3 wr = b3_sub(*wb, *wa);
    B3Vec2 rhs = {b3_dot(wr, t1), b3_dot(wr, t2)};
    B3Vec2 d = b3_mv2(b3_invert2(k), rhs);
    B3Vec3 next = b3_sub(*impulse, b3_add(b3_mul(t1, d.x), b3_mul(t2, d.y)));
    float limit = rolling * total_n;
    float l2 = b3_len2(next);
    if (l2 > limit * limit && l2 > 0.0f)
        next = b3_mul(next, b3_rsqrt_scale(limit, l2));
    B3Vec3 delta = b3_sub(next, *impulse);
    *impulse = next;
    *wa = b3_sub(*wa, b3_mv(ia, delta));
    *wb = b3_add(*wb, b3_mv(ib, delta));
}


B3_HD B3_INL void b3_solve_joints_global(B3World* w, float h, float inv_h,
        int use_bias) {
#ifdef B3_CACHE_JOINTS
    for (int i = 0; i < w->joint_count; i++) {
        B3Joint* j = &w->joints[i];
#ifndef B3_REVOLUTE_ONLY
        if (j->type == B3_JOINT_WELD) {
            continue;
        }
#endif
        b3_cache_revolute(j, &w->bodies[j->body_a], &w->bodies[j->body_b]);
    }
#endif
    int iters = use_bias ? B3_JOINT_ITERS : B3_RELAX_ITERS;
    for (int iter = 0; iter < iters; iter++) {
#if defined(B3_INTERLEAVE_CONTACTS) && !defined(B3_ABLATE_NO_CONTACT)
        b3_solve_contacts(w, inv_h, w->contact_speed, use_bias);
#endif
        for (int i = 0; i < w->joint_count; i++) {
            B3Joint* j = &w->joints[i];
            B3Body* ba = &w->bodies[j->body_a];
            B3Body* bb = &w->bodies[j->body_b];
#ifdef B3_REVOLUTE_ONLY
            b3_solve_revolute(j, ba, bb, h, inv_h, use_bias);
#else
            if (j->type == B3_JOINT_WELD) {
                b3_solve_weld(j, ba, bb, use_bias);
            } else {
                b3_solve_revolute(j, ba, bb, h, inv_h, use_bias);
            }
#endif
        }
    }
}

/* Snapshot joints/bodies/contacts into per-thread local memory for the
 * 8-iter GS sweep. Measured 1.7x vs global; does not prove DRAM vs ALU. */
B3_HD B3_INL void b3_solve_joints_local(B3World* w, float h, float inv_h,
        int use_bias) {
#if B3_ART_CONTACTS && defined(B3_INTERLEAVE_CONTACTS)
    // Delassus impulses can change every link; use the world-backed sweep.
    if (w->joint_count > 0 && w->contact_count > 0) {
        b3_solve_joints_global(w, h, inv_h, use_bias);
        return;
    }
#endif
    B3Body bl[B3_MAX_BODIES];
    B3Joint jl[B3_MAX_JOINTS];
    B3Contact cl[B3_MAX_CONTACTS];
    int nb = w->body_count;
    int nj = w->joint_count;
    int nc = w->contact_count;
    for (int i = 0; i < nb; i++) {
        bl[i] = w->bodies[i];
    }
    for (int i = 0; i < nj; i++) {
        jl[i] = w->joints[i];
    }
    for (int i = 0; i < nc; i++) {
        cl[i] = w->contacts[i];
    }
#ifdef B3_CACHE_JOINTS
    for (int i = 0; i < nj; i++) {
#ifndef B3_REVOLUTE_ONLY
        if (jl[i].type == B3_JOINT_WELD) {
            continue;
        }
#endif
        b3_cache_revolute(&jl[i], &bl[jl[i].body_a], &bl[jl[i].body_b]);
    }
#endif
    int iters = use_bias ? B3_JOINT_ITERS : B3_RELAX_ITERS;
    for (int iter = 0; iter < iters; iter++) {
#if defined(B3_INTERLEAVE_CONTACTS) && !defined(B3_ABLATE_NO_CONTACT)
        b3_solve_contacts_n(cl, nc, bl, inv_h, w->contact_speed, use_bias);
#endif
        for (int i = 0; i < nj; i++) {
#ifdef B3_REVOLUTE_ONLY
            b3_solve_revolute(&jl[i], &bl[jl[i].body_a], &bl[jl[i].body_b],
                h, inv_h, use_bias);
#else
            if (jl[i].type == B3_JOINT_WELD) {
                b3_solve_weld(&jl[i], &bl[jl[i].body_a],
                    &bl[jl[i].body_b], use_bias);
            } else {
                b3_solve_revolute(&jl[i], &bl[jl[i].body_a],
                    &bl[jl[i].body_b], h, inv_h, use_bias);
            }
#endif
        }
    }
    for (int i = 0; i < nb; i++) {
        w->bodies[i].lin_vel = bl[i].lin_vel;
        w->bodies[i].ang_vel = bl[i].ang_vel;
    }
    for (int i = 0; i < nj; i++) {
        w->joints[i] = jl[i];
    }
    for (int i = 0; i < nc; i++) {
        w->contacts[i] = cl[i];
    }
}

#ifdef B3_PACKED_GS
B3_HD B3_INL void b3_cache_revolute_gs(B3GsJoint* j, const B3Joint* src,
        const B3GsBody* ba, const B3GsBody* bb) {
    B3Vec3 ra = b3_rotate(ba->delta_rot, src->frame_p_a);
    B3Vec3 rb = b3_rotate(bb->delta_rot, src->frame_p_b);
    j->cache_ra = ra;
    j->cache_rb = rb;
    j->cache_point_invk = b3_invert3(b3_point_k_gs(
        ba->inv_mass, bb->inv_mass, ba->inv_i, bb->inv_i, ra, rb));
    B3Quat qa = b3_qmul(ba->delta_rot, src->frame_q_a);
    B3Quat qb = b3_qmul(bb->delta_rot, src->frame_q_b);
    if (b3_qdot(qa, qb) < 0.0f) {
        qb = b3_qneg(qb);
    }
    B3Quat rel = b3_qinv_mul(qa, qb);
    j->cache_rel_x = rel.v.x;
    j->cache_rel_y = rel.v.y;
    j->cache_twist = b3_twist(rel);
    if (j->bits & B3_GS_FIXED) {
        j->cache_ang_invk.cx.x = 0.0f;
        j->cache_ang_invk.cx.y = 0.0f;
        j->cache_ang_invk.cy.x = 0.0f;
        j->cache_ang_invk.cy.y = 0.0f;
        j->cache_ia_ax = b3_v(0.0f, 0.0f, 0.0f);
        j->cache_ib_ax = b3_v(0.0f, 0.0f, 0.0f);
        return;
    }
    B3Vec3 px, py;
    b3_hinge_perps(qa, rel, &px, &py);
    j->perp_x = px;
    j->perp_y = py;
    B3Mat3 isum = b3_maddm(ba->inv_i, bb->inv_i);
    B3Mat2 k;
    k.cx.x = b3_dot(px, b3_mv(isum, px));
    k.cy.y = b3_dot(py, b3_mv(isum, py));
    k.cx.y = k.cy.x = b3_dot(px, b3_mv(isum, py));
    j->cache_ang_invk = b3_invert2(k);
#if B3_COUPLED_HINGE
    // Block effective mass [A B; B^T D] for anchor xyz + hinge alignment xy.
    // A^-1 already exists. Cache A^-1 B and (D - B^T A^-1 B)^-1.
    B3Vec3 bx = b3_neg(b3_add(b3_cross(ra, b3_mv(ba->inv_i, px)),
        b3_cross(rb, b3_mv(bb->inv_i, px))));
    B3Vec3 by = b3_neg(b3_add(b3_cross(ra, b3_mv(ba->inv_i, py)),
        b3_cross(rb, b3_mv(bb->inv_i, py))));
    j->cache_point_perp_x = b3_mv(j->cache_point_invk, bx);
    j->cache_point_perp_y = b3_mv(j->cache_point_invk, by);
    k.cx.x -= b3_dot(bx, j->cache_point_perp_x);
    k.cy.y -= b3_dot(by, j->cache_point_perp_y);
    k.cx.y = k.cy.x = k.cx.y - b3_dot(bx, j->cache_point_perp_y);
    j->cache_ang_invk = b3_invert2(k);
#endif
    j->cache_ia_ax = b3_mv(ba->inv_i, j->rotation_axis);
    j->cache_ib_ax = b3_mv(bb->inv_i, j->rotation_axis);
}

B3_HD B3_INL void b3_solve_axial_gs(B3GsJoint* j, B3GsBody* ba,
        B3GsBody* bb, float h, float inv_h, int use_bias) {
    B3Vec3 wa = ba->ang_vel;
    B3Vec3 wb = bb->ang_vel;
    B3Vec3 axis = j->rotation_axis;
    B3Vec3 ia_ax = j->cache_ia_ax;
    B3Vec3 ib_ax = j->cache_ib_ax;
    float twist = j->cache_twist;
    int fixed = j->bits & B3_GS_FIXED;
#ifndef B3_ABLATE_NO_SPRING
    if ((j->bits & B3_GS_SPRING) && !fixed) {
        float c = twist - j->target_angle;
        float bias = j->spring_softness.bias_rate * c;
        float mscale = j->spring_softness.mass_scale;
        float iscale = j->spring_softness.impulse_scale;
        float cdot = b3_dot(b3_sub(wb, wa), axis);
        float dimp = -mscale * j->axial_mass * (cdot + bias)
            - iscale * j->spring_impulse;
        float old = j->spring_impulse;
        float nimp = old + dimp;
        if (j->max_motor_torque > 0.0f) {
            float maxp = j->max_motor_torque * h;
            nimp = b3_clamp(nimp, -maxp, maxp);
        }
        dimp = nimp - old;
        j->spring_impulse = nimp;
        wa = b3_msub(wa, dimp, ia_ax);
        wb = b3_madd(wb, dimp, ib_ax);
    }
#endif
#ifndef B3_ABLATE_NO_LIMIT
    if ((j->bits & B3_GS_LIMIT) && !fixed) {
        float angle = twist;
        {
            float c = angle - j->lower_angle;
            float bias = 0.0f;
            float mscale = 1.0f;
            float iscale = 0.0f;
            if (c > 0.0f) {
                bias = c * inv_h;
            } else if (use_bias) {
                bias = j->softness.bias_rate * c;
                mscale = j->softness.mass_scale;
                iscale = j->softness.impulse_scale;
            }
            float cdot = b3_dot(b3_sub(wb, wa), axis);
            float old = j->lower_impulse;
            float dimp = -mscale * j->axial_mass * (cdot + bias)
                - iscale * old;
            j->lower_impulse = b3_maxf(old + dimp, 0.0f);
            dimp = j->lower_impulse - old;
            wa = b3_msub(wa, dimp, ia_ax);
            wb = b3_madd(wb, dimp, ib_ax);
        }
        {
            float c = j->upper_angle - angle;
            float bias = 0.0f;
            float mscale = 1.0f;
            float iscale = 0.0f;
            if (c > 0.0f) {
                bias = c * inv_h;
            } else if (use_bias) {
                bias = j->softness.bias_rate * c;
                mscale = j->softness.mass_scale;
                iscale = j->softness.impulse_scale;
            }
            float cdot = b3_dot(b3_sub(wa, wb), axis);
            float old = j->upper_impulse;
            float dimp = -mscale * j->axial_mass * (cdot + bias)
                - iscale * old;
            j->upper_impulse = b3_maxf(old + dimp, 0.0f);
            dimp = j->upper_impulse - old;
            wa = b3_madd(wa, dimp, ia_ax);
            wb = b3_msub(wb, dimp, ib_ax);
        }
    }
#endif
    if (ba->flags & B3_FLAG_DYNAMIC) ba->ang_vel = wa;
    if (bb->flags & B3_FLAG_DYNAMIC) bb->ang_vel = wb;
}

B3_HD B3_INL void b3_solve_revolute_gs(B3GsJoint* j, B3GsBody* ba,
        B3GsBody* bb, float h, float inv_h, int use_bias) {
    b3_solve_axial_gs(j, ba, bb, h, inv_h, use_bias);
    B3Vec3 va = ba->lin_vel, wa = ba->ang_vel;
    B3Vec3 vb = bb->lin_vel, wb = bb->ang_vel;
    int fixed = j->bits & B3_GS_FIXED;
#if B3_COUPLED_HINGE
    {
        B3Vec3 ra = j->cache_ra;
        B3Vec3 rb = j->cache_rb;
        B3Vec3 rhs_p = b3_sub(b3_add(vb, b3_cross(wb, rb)),
            b3_add(va, b3_cross(wa, ra)));
        float mscale = use_bias ? j->softness.mass_scale : 1.0f;
        float iscale = use_bias ? j->softness.impulse_scale : 0.0f;
        if (use_bias) {
            B3Vec3 sep = b3_add(b3_add(b3_sub(bb->delta_pos, ba->delta_pos),
                b3_sub(rb, ra)), j->delta_center);
            rhs_p = b3_madd(rhs_p, j->softness.bias_rate, sep);
        }
        B3Vec3 sol_p = b3_mv(j->cache_point_invk, rhs_p);
        B3Vec3 ang = b3_v(0, 0, 0);
        if (!fixed) {
            B3Vec3 wrel = b3_sub(wb, wa);
            B3Vec2 rhs_a;
            rhs_a.x = b3_dot(wrel, j->perp_x);
            rhs_a.y = b3_dot(wrel, j->perp_y);
            if (use_bias) {
                rhs_a.x += j->softness.bias_rate * j->cache_rel_x;
                rhs_a.y += j->softness.bias_rate * j->cache_rel_y;
            }
            rhs_a.x -= b3_dot(j->cache_point_perp_x, rhs_p);
            rhs_a.y -= b3_dot(j->cache_point_perp_y, rhs_p);
            B3Vec2 sol_a = b3_mv2(j->cache_ang_invk, rhs_a);
            sol_p = b3_sub(sol_p, b3_add(b3_mul(j->cache_point_perp_x, sol_a.x),
                b3_mul(j->cache_point_perp_y, sol_a.y)));
            B3Vec2 da;
            da.x = -mscale * sol_a.x - iscale * j->perp_impulse.x;
            da.y = -mscale * sol_a.y - iscale * j->perp_impulse.y;
            j->perp_impulse.x += da.x;
            j->perp_impulse.y += da.y;
            ang = b3_add(b3_mul(j->perp_x, da.x), b3_mul(j->perp_y, da.y));
        }
        B3Vec3 impulse = b3_msub(b3_mul(sol_p, -mscale), iscale, j->linear_impulse);
        j->linear_impulse = b3_add(j->linear_impulse, impulse);
        va = b3_msub(va, ba->inv_mass, impulse);
        wa = b3_sub(wa, b3_mv(ba->inv_i, b3_add(b3_cross(ra, impulse), ang)));
        vb = b3_madd(vb, bb->inv_mass, impulse);
        wb = b3_add(wb, b3_mv(bb->inv_i, b3_add(b3_cross(rb, impulse), ang)));
    }
#else
#ifndef B3_ABLATE_NO_PERP
    if (!fixed) {
        B3Vec2 bias;
        bias.x = 0.0f;
        bias.y = 0.0f;
        float mscale = 1.0f;
        float iscale = 0.0f;
        if (use_bias) {
            bias.x = j->softness.bias_rate * j->cache_rel_x;
            bias.y = j->softness.bias_rate * j->cache_rel_y;
            mscale = j->softness.mass_scale;
            iscale = j->softness.impulse_scale;
        }
        B3Vec3 px = j->perp_x;
        B3Vec3 py = j->perp_y;
        B3Vec3 wrel = b3_sub(wb, wa);
        B3Vec2 rhs;
        rhs.x = b3_dot(wrel, px) + bias.x;
        rhs.y = b3_dot(wrel, py) + bias.y;
        B3Vec2 sol = b3_mv2(j->cache_ang_invk, rhs);
        B3Vec2 old = j->perp_impulse;
        B3Vec2 dimp;
        dimp.x = -mscale * sol.x - iscale * old.x;
        dimp.y = -mscale * sol.y - iscale * old.y;
        j->perp_impulse.x += dimp.x;
        j->perp_impulse.y += dimp.y;
        B3Vec3 ang = b3_add(b3_mul(px, dimp.x), b3_mul(py, dimp.y));
        wa = b3_sub(wa, b3_mv(ba->inv_i, ang));
        wb = b3_add(wb, b3_mv(bb->inv_i, ang));
    }
#endif
#ifndef B3_ABLATE_NO_POINT
    {
        B3Vec3 ra = j->cache_ra;
        B3Vec3 rb = j->cache_rb;
        B3Vec3 cdot = b3_sub(b3_add(vb, b3_cross(wb, rb)),
            b3_add(va, b3_cross(wa, ra)));
        B3Vec3 bias = b3_v(0.0f, 0.0f, 0.0f);
        float mscale = 1.0f;
        float iscale = 0.0f;
        if (use_bias) {
            B3Vec3 sep = b3_add(b3_add(b3_sub(bb->delta_pos, ba->delta_pos),
                b3_sub(rb, ra)), j->delta_center);
            bias = b3_mul(sep, j->softness.bias_rate);
            mscale = j->softness.mass_scale;
            iscale = j->softness.impulse_scale;
        }
        B3Vec3 rhs = b3_mv(j->cache_point_invk, b3_add(cdot, bias));
        B3Vec3 impulse = b3_msub(b3_mul(rhs, -mscale), iscale,
            j->linear_impulse);
        j->linear_impulse = b3_add(j->linear_impulse, impulse);
        va = b3_msub(va, ba->inv_mass, impulse);
        wa = b3_sub(wa, b3_mv(ba->inv_i, b3_cross(ra, impulse)));
        vb = b3_madd(vb, bb->inv_mass, impulse);
        wb = b3_add(wb, b3_mv(bb->inv_i, b3_cross(rb, impulse)));
    }
#endif
#endif // B3_COUPLED_HINGE
    if (ba->flags & B3_FLAG_DYNAMIC) {
        ba->lin_vel = va;
        ba->ang_vel = wa;
    }
    if (bb->flags & B3_FLAG_DYNAMIC) {
        bb->lin_vel = vb;
        bb->ang_vel = wb;
    }
}

B3_HD B3_INL void b3_solve_contacts_gs(B3GsContact* contacts, int n,
        B3GsBody* bodies, float inv_h, float contact_speed, int use_bias) {
    for (int i = 0; i < n; i++) {
        B3GsContact* c = &contacts[i];
        B3GsBody* ba = &bodies[c->body_a];
        B3GsBody* bb = &bodies[c->body_b];
        B3Vec3 va = ba->lin_vel;
        B3Vec3 wa = ba->ang_vel;
        B3Vec3 vb = bb->lin_vel;
        B3Vec3 wb = bb->ang_vel;
        B3Quat dqa = ba->delta_rot;
        B3Quat dqb = bb->delta_rot;
        B3Vec3 dp = b3_sub(bb->delta_pos, ba->delta_pos);
        B3Vec3 nrm = c->normal;
        float total_n = 0.0f;
        float twist_lim = 0.0f;
        for (int p = 0; p < c->point_count; p++) {
            B3GsPoint* cp = &c->points[p];
            B3Vec3 ra = cp->r_a;
            B3Vec3 rb = cp->r_b;
            B3Vec3 ds = b3_add(dp, b3_sub(b3_rotate(dqb, rb),
                b3_rotate(dqa, ra)));
            float sep = b3_dot(ds, nrm) + cp->base_sep;
            float vbias = 0.0f;
            float mscale = 1.0f;
            float iscale = 0.0f;
            if (sep > 0.0f) {
                vbias = sep * inv_h;
            } else if (use_bias) {
                vbias = b3_maxf(c->softness.mass_scale
                    * c->softness.bias_rate * sep, -contact_speed);
                mscale = c->softness.mass_scale;
                iscale = c->softness.impulse_scale;
            }
            B3Vec3 vra = b3_add(va, b3_cross(wa, ra));
            B3Vec3 vrb = b3_add(vb, b3_cross(wb, rb));
            float vn = b3_dot(b3_sub(vrb, vra), nrm);
            float dimp = -cp->normal_mass * (mscale * vn + vbias)
                - iscale * cp->normal_impulse;
            float nimp = b3_maxf(cp->normal_impulse + dimp, 0.0f);
            dimp = nimp - cp->normal_impulse;
            cp->normal_impulse = nimp;
            cp->total_normal += nimp;
            total_n += nimp;
            twist_lim += cp->lever * cp->normal_impulse;
            B3Vec3 P = b3_mul(nrm, dimp);
            va = b3_msub(va, ba->inv_mass, P);
            wa = b3_sub(wa, b3_mv(ba->inv_i, b3_cross(ra, P)));
            vb = b3_madd(vb, bb->inv_mass, P);
            wb = b3_add(wb, b3_mv(bb->inv_i, b3_cross(rb, P)));
        }
        if (!use_bias) {
            float twist_s = b3_dot(nrm, b3_sub(wb, wa));
            float max_t = c->friction * twist_lim;
            float dtw = -c->twist_mass * twist_s;
            float old_t = c->twist_impulse;
            c->twist_impulse = b3_clamp(old_t + dtw, -max_t, max_t);
            dtw = c->twist_impulse - old_t;
            wa = b3_sub(wa, b3_mv(ba->inv_i, b3_mul(nrm, dtw)));
            wb = b3_add(wb, b3_mv(bb->inv_i, b3_mul(nrm, dtw)));

            B3Vec3 t1 = c->tangent1;
            B3Vec3 t2 = c->tangent2;
            B3Vec3 ra = c->center_a;
            B3Vec3 rb = c->center_b;
            B3Vec3 vra = b3_add(va, b3_cross(wa, ra));
            B3Vec3 vrb = b3_add(vb, b3_cross(wb, rb));
            B3Vec3 vr = b3_sub(vrb, vra);
            B3Vec2 vt;
            vt.x = b3_dot(vr, t1);
            vt.y = b3_dot(vr, t2);
            B3Vec2 tm = b3_mv2(c->tangent_mass, vt);
            B3Vec2 ni;
            ni.x = c->friction_impulse.x - tm.x;
            ni.y = c->friction_impulse.y - tm.y;
            float max_f = c->friction * total_n;
            float fl2 = ni.x * ni.x + ni.y * ni.y;
            if (fl2 > max_f * max_f && fl2 > 0.0f) {
                float sc = b3_rsqrt_scale(max_f, fl2);
                ni.x *= sc;
                ni.y *= sc;
            }
            B3Vec2 df;
            df.x = ni.x - c->friction_impulse.x;
            df.y = ni.y - c->friction_impulse.y;
            c->friction_impulse = ni;
            B3Vec3 P = b3_add(b3_mul(t1, df.x), b3_mul(t2, df.y));
            va = b3_msub(va, ba->inv_mass, P);
            wa = b3_sub(wa, b3_mv(ba->inv_i, b3_cross(ra, P)));
            vb = b3_madd(vb, bb->inv_mass, P);
            wb = b3_add(wb, b3_mv(bb->inv_i, b3_cross(rb, P)));
            b3_solve_rolling(c->tangent1, c->tangent2, c->rolling, total_n,
                ba->inv_i, bb->inv_i, &c->rolling_impulse, &wa, &wb);
        }
        if (ba->flags & B3_FLAG_DYNAMIC) {
            ba->lin_vel = va;
            ba->ang_vel = wa;
        }
        if (bb->flags & B3_FLAG_DYNAMIC) {
            bb->lin_vel = vb;
            bb->ang_vel = wb;
        }
    }
}

#if B3_ART_CONTACTS
B3_HD B3_INL void b3_solve_contacts_gs_w(B3World* w, B3Art* art,
        B3GsContact* contacts, int n, B3GsBody* bodies, float inv_h,
        float contact_speed, int use_bias) {
    if (n == 0) return;
    if (art) {
        b3_art_solve_contacts_gs(art, w, contacts, n, bodies,
            inv_h, contact_speed, use_bias, 1);
        return;
    }
    if (w->joint_count > 0) {
        B3Art local;
        if (b3_art_bind(&local, w)) {
            b3_art_solve_contacts_gs(&local, w, contacts, n, bodies,
                inv_h, contact_speed, use_bias, 1);
            return;
        }
    }
    b3_solve_contacts_gs(contacts, n, bodies, inv_h, contact_speed, use_bias);
}
#endif

B3_HD B3_INL void b3_gs_load(const B3World* w, B3GsBody* bl,
        B3GsJoint* jl, B3GsContact* cl) {
    int nb = w->body_count;
    int nj = w->joint_count;
    int nc = w->contact_count;
    for (int i = 0; i < nb; i++) {
        const B3Body* b = &w->bodies[i];
        bl[i].lin_vel = b->lin_vel;
        bl[i].ang_vel = b->ang_vel;
        bl[i].delta_pos = b->delta_pos;
        bl[i].delta_rot = b->delta_rot;
        bl[i].inv_mass = b->type == B3_DYNAMIC ? b->inv_mass : 0.0f;
        bl[i].inv_i = b->type == B3_DYNAMIC ? b->inv_i_world : b3_mat0();
        bl[i].flags = b->flags;
    }
    for (int i = 0; i < nj; i++) {
        const B3Joint* src = &w->joints[i];
        B3GsJoint* j = &jl[i];
        j->body_a = src->body_a;
        j->body_b = src->body_b;
        j->bits = 0;
        if (src->fixed_rotation) {
            j->bits |= B3_GS_FIXED;
        }
        if (src->enable_spring) {
            j->bits |= B3_GS_SPRING;
        }
        if (src->enable_limit) {
            j->bits |= B3_GS_LIMIT;
        }
        j->target_angle = src->target_angle;
        j->lower_angle = src->lower_angle;
        j->upper_angle = src->upper_angle;
        j->axial_mass = src->axial_mass;
        j->max_motor_torque = src->max_motor_torque;
        j->spring_impulse = src->spring_impulse;
        j->lower_impulse = src->lower_impulse;
        j->upper_impulse = src->upper_impulse;
        j->softness = src->softness;
        j->spring_softness = src->spring_softness;
        j->rotation_axis = src->rotation_axis;
        j->delta_center = src->delta_center;
        j->linear_impulse = src->linear_impulse;
        j->perp_impulse = src->perp_impulse;
    }
    for (int i = 0; i < nc; i++) {
        const B3Contact* src = &w->contacts[i];
        B3GsContact* c = &cl[i];
        c->body_a = src->body_a;
        c->body_b = src->body_b;
        c->point_count = src->point_count;
        c->normal = src->normal;
        c->tangent1 = src->tangent1;
        c->tangent2 = src->tangent2;
        c->center_a = src->center_a;
        c->center_b = src->center_b;
        c->friction = src->friction;
        c->rolling = src->rolling;
        c->rolling_impulse = src->rolling_impulse;
        c->twist_mass = src->twist_mass;
        c->twist_impulse = src->twist_impulse;
        c->friction_impulse = src->friction_impulse;
        c->tangent_mass = src->tangent_mass;
        c->softness = src->softness;
        for (int p = 0; p < src->point_count; p++) {
            c->points[p].r_a = src->points[p].r_a;
            c->points[p].r_b = src->points[p].r_b;
            c->points[p].base_sep = src->points[p].base_sep;
            c->points[p].normal_impulse = src->points[p].normal_impulse;
            c->points[p].total_normal = src->points[p].total_normal;
            c->points[p].normal_mass = src->points[p].normal_mass;
            c->points[p].lever = src->points[p].lever;
        }
    }
}

// Only velocities and delta transforms change during the substeps. Mass,
// inertia, flags, prepared joint data and contact geometry stay fixed.
B3_HD B3_INL void b3_gs_refresh_bodies(const B3World* w, B3GsBody* bl) {
    for (int i = 0; i < w->body_count; i++) {
        bl[i].lin_vel = w->bodies[i].lin_vel;
        bl[i].ang_vel = w->bodies[i].ang_vel;
        bl[i].delta_pos = w->bodies[i].delta_pos;
        bl[i].delta_rot = w->bodies[i].delta_rot;
    }
}

#if B3_TREE_DUAL
#include "b3_tree_dual.cuh"
#endif

B3_HD B3_INL void b3_gs_solve_cached(const B3World* w, B3GsBody* bl,
        B3GsJoint* jl, B3GsContact* cl, float h, float inv_h, int use_bias, int refresh) {
    int nj = w->joint_count;
    int nc = w->contact_count;
    // Only position integration invalidates these pose-dependent caches.
    if (refresh) for (int i = 0; i < nj; i++) {
        b3_cache_revolute_gs(&jl[i], &w->joints[i],
            &bl[jl[i].body_a], &bl[jl[i].body_b]);
    }
#if B3_TREE_DUAL
    if (b3_tree_solve(w, bl, jl, cl, h, inv_h, use_bias)) return;
#endif
#if B3_ART_CONTACTS && defined(B3_INTERLEAVE_CONTACTS) && !defined(B3_ABLATE_NO_CONTACT)
    B3Art art;
    B3Art* artp = nc > 0 && b3_art_bind(&art, w) ? &art : 0;
#endif
    int iters = use_bias ? B3_JOINT_ITERS : B3_RELAX_ITERS;
    for (int iter = 0; iter < iters; iter++) {
#if defined(B3_INTERLEAVE_CONTACTS) && !defined(B3_ABLATE_NO_CONTACT)
#if B3_ART_CONTACTS
        b3_solve_contacts_gs_w((B3World*)w, artp, cl, nc, bl,
            inv_h, w->contact_speed, use_bias);
#else
        b3_solve_contacts_gs(cl, nc, bl, inv_h, w->contact_speed, use_bias);
#endif
#endif
        for (int k = 0; k < nj; k++) {
#if B3_ALTERNATE_JOINT_ORDER
            int i = (iter & 1) ? nj - 1 - k : k;
#else
            int i = k;
#endif
            b3_solve_revolute_gs(&jl[i], &bl[jl[i].body_a],
                &bl[jl[i].body_b], h, inv_h, use_bias);
        }
    }
}

B3_HD B3_INL void b3_gs_solve(const B3World* w, B3GsBody* bl,
        B3GsJoint* jl, B3GsContact* cl, float h, float inv_h, int use_bias) {
    b3_gs_solve_cached(w, bl, jl, cl, h, inv_h, use_bias, 1);
}

B3_HD B3_INL void b3_gs_store_velocities(B3World* w, const B3GsBody* bl) {
    for (int i = 0; i < w->body_count; i++) {
        w->bodies[i].lin_vel = bl[i].lin_vel;
        w->bodies[i].ang_vel = bl[i].ang_vel;
    }
}

B3_HD B3_INL void b3_gs_store_constraints(B3World* w, const B3GsJoint* jl,
        const B3GsContact* cl) {
    for (int i = 0; i < w->joint_count; i++) {
        B3Joint* dst = &w->joints[i];
        const B3GsJoint* j = &jl[i];
        dst->spring_impulse = j->spring_impulse;
        dst->lower_impulse = j->lower_impulse;
        dst->upper_impulse = j->upper_impulse;
        dst->linear_impulse = j->linear_impulse;
        dst->perp_impulse = j->perp_impulse;
    }
    for (int i = 0; i < w->contact_count; i++) {
        B3Contact* dst = &w->contacts[i];
        const B3GsContact* c = &cl[i];
        dst->twist_impulse = c->twist_impulse;
        dst->friction_impulse = c->friction_impulse;
        dst->rolling_impulse = c->rolling_impulse;
        for (int p = 0; p < dst->point_count; p++) {
            dst->points[p].normal_impulse = c->points[p].normal_impulse;
            dst->points[p].total_normal = c->points[p].total_normal;
        }
    }
}

B3_HD B3_INL void b3_solve_joints_packed(B3World* w, float h, float inv_h,
        int use_bias) {
    B3GsBody bl[B3_MAX_BODIES];
    B3GsJoint jl[B3_MAX_JOINTS];
    B3GsContact cl[B3_MAX_CONTACTS];
    b3_gs_load(w, bl, jl, cl);
    b3_gs_solve(w, bl, jl, cl, h, inv_h, use_bias);
    b3_gs_store_velocities(w, bl);
    b3_gs_store_constraints(w, jl, cl);
}
#endif

#if B3_JOINT_DUAL
#ifdef B3_DUAL_TRACE
#include <stdio.h>
#endif
typedef struct B3Mat5 {
    float m[25];
} B3Mat5;

static B3_HD B3_INL int b3_lu5(B3Mat5* a, int* piv) {
    for (int i = 0; i < 5; i++) {
        piv[i] = i;
    }
    for (int k = 0; k < 5; k++) {
        int p = k;
        float best = fabsf(a->m[k * 5 + k]);
        for (int i = k + 1; i < 5; i++) {
            float v = fabsf(a->m[i * 5 + k]);
            if (v > best) {
                best = v;
                p = i;
            }
        }
        if (!(best > 1.0e-14f)) {
            return 0;
        }
        if (p != k) {
            for (int c = 0; c < 5; c++) {
                float t = a->m[k * 5 + c];
                a->m[k * 5 + c] = a->m[p * 5 + c];
                a->m[p * 5 + c] = t;
            }
            int t = piv[k];
            piv[k] = piv[p];
            piv[p] = t;
        }
        float d = a->m[k * 5 + k];
        for (int i = k + 1; i < 5; i++) {
            float f = a->m[i * 5 + k] / d;
            a->m[i * 5 + k] = f;
            for (int c = k + 1; c < 5; c++) {
                a->m[i * 5 + c] -= f * a->m[k * 5 + c];
            }
        }
    }
    return 1;
}

static B3_HD B3_INL void b3_lu5_apply(const B3Mat5* a, const int* piv,
        const float* rhs, float* out) {
    float t[5];
    for (int i = 0; i < 5; i++) {
        t[i] = rhs[piv[i]];
    }
    for (int i = 1; i < 5; i++) {
        for (int c = 0; c < i; c++) {
            t[i] -= a->m[i * 5 + c] * t[c];
        }
    }
    for (int i = 4; i >= 0; i--) {
        for (int c = i + 1; c < 5; c++) {
            t[i] -= a->m[i * 5 + c] * t[c];
        }
        t[i] /= a->m[i * 5 + i];
    }
    for (int i = 0; i < 5; i++) {
        out[i] = t[i];
    }
}

/* Verify the joints form one open revolute chain with no spring, motor,
 * limit or fixed rotation, and record a root-to-leaf joint order. */
static B3_HD B3_INL int b3_joint_chain_order(B3World* w, int* order,
        int* body_at) {
    int nj = w->joint_count;
    if (nj < 1 || nj > B3_MAX_JOINTS) {
        return 0;
    }
    unsigned char deg[B3_MAX_BODIES];
    for (int i = 0; i < w->body_count; i++) {
        deg[i] = 0;
    }
    for (int k = 0; k < nj; k++) {
        const B3Joint* j = &w->joints[k];
        if (j->type != B3_JOINT_REVOLUTE || j->enable_spring
            || j->enable_limit || j->fixed_rotation) {
            return 0;
        }
#ifndef B3_REVOLUTE_ONLY
        if (j->enable_motor) {
            return 0;
        }
#endif
        deg[j->body_a]++;
        deg[j->body_b]++;
    }
    int endpoints[2];
    int ne = 0;
    for (int i = 0; i < w->body_count; i++) {
        if (deg[i] == 1) {
            if (ne < 2) {
                endpoints[ne] = i;
            }
            ne++;
        } else if (deg[i] != 0 && deg[i] != 2) {
            return 0;
        }
    }
    if (ne != 2) {
        return 0;
    }
    unsigned char used[B3_MAX_JOINTS];
    for (int k = 0; k < nj; k++) {
        used[k] = 0;
    }
    int cur = endpoints[0];
    for (int step = 0; step < nj; step++) {
        int found = -1;
        for (int q = 0; q < nj; q++) {
            if (!used[q]) {
                const B3Joint* j = &w->joints[q];
                if (j->body_a == cur || j->body_b == cur) {
                    found = q;
                    break;
                }
            }
        }
        if (found < 0) {
            return 0;
        }
        used[found] = 1;
        order[step] = found;
        const B3Joint* j = &w->joints[found];
        body_at[step] = cur;
        cur = j->body_a == cur ? j->body_b : j->body_a;
    }
    return 1;
}

/* One exact block-tridiagonal dual KKT solve per pass. Constraint rows per
 * joint: 3 anchor + 2 alignment (plain revolute). Softness folds into the
 * diagonal block as the Gauss-Seidel fixed point of the legacy path. */
static B3_HD B3_INL int b3_solve_joints_dual(B3World* w, float inv_h,
        int use_bias) {
    int order[B3_MAX_JOINTS];
    int body_at[B3_MAX_JOINTS];
    if (!b3_joint_chain_order(w, order, body_at)) {
        return 0;
    }
    int nj = w->joint_count;
    B3Mat5 D[B3_MAX_JOINTS];
    B3Mat5 Bc[B3_MAX_JOINTS];
    float rhs[B3_MAX_JOINTS][5];
    float yv[B3_MAX_JOINTS][5];
    float lam[B3_MAX_JOINTS][5];
    int pivs[B3_MAX_JOINTS][5];
    // Per-joint walk-side data for coupling assembly.
    B3Vec3 leva[B3_MAX_JOINTS];
    B3Vec3 levb[B3_MAX_JOINTS];
    B3Vec3 pp[B3_MAX_JOINTS][2];
    B3Vec3 relv[B3_MAX_JOINTS];
    for (int step = 0; step < nj; step++) {
        const B3Joint* j = &w->joints[order[step]];
        leva[step] = b3_rotate(w->bodies[j->body_a].delta_rot,
            j->frame_p_a);
        levb[step] = b3_rotate(w->bodies[j->body_b].delta_rot,
            j->frame_p_b);
        B3Quat qa = b3_qmul(w->bodies[j->body_a].delta_rot, j->frame_q_a);
        B3Quat qb = b3_qmul(w->bodies[j->body_b].delta_rot, j->frame_q_b);
        if (b3_qdot(qa, qb) < 0.0f) {
            qb = b3_qneg(qb);
        }
        B3Quat rel = b3_qinv_mul(qa, qb);
        relv[step] = rel.v;
        b3_hinge_perps(qa, rel, &pp[step][0], &pp[step][1]);
    }
    for (int step = 0; step < nj; step++) {
        const B3Joint* j = &w->joints[order[step]];
        const B3Body* ba = &w->bodies[j->body_a];
        const B3Body* bb = &w->bodies[j->body_b];
        float ma = ba->type == B3_DYNAMIC ? ba->inv_mass : 0.0f;
        float mb = bb->type == B3_DYNAMIC ? bb->inv_mass : 0.0f;
        B3Mat3 ia = ba->type == B3_DYNAMIC ? ba->inv_i_world : b3_mat0();
        B3Mat3 ib = bb->type == B3_DYNAMIC ? bb->inv_i_world : b3_mat0();
        B3Vec3 ra = b3_rotate(ba->delta_rot, j->frame_p_a);
        B3Vec3 rb = b3_rotate(bb->delta_rot, j->frame_p_b);
        B3Vec3 va = ba->lin_vel;
        B3Vec3 wa = ba->ang_vel;
        B3Vec3 vb = bb->lin_vel;
        B3Vec3 wb = bb->ang_vel;
        B3Vec3 wrel = b3_sub(wb, wa);
        B3Vec3 cdotp = b3_sub(b3_add(vb, b3_cross(wb, rb)),
            b3_add(va, b3_cross(wa, ra)));
        B3Vec3 pxx = pp[step][0];
        B3Vec3 pyy = pp[step][1];
        B3Vec3 bias3 = b3_v(0.0f, 0.0f, 0.0f);
        float bias2x = 0.0f;
        float bias2y = 0.0f;
        float s = 0.0f;
        if (use_bias) {
            B3Vec3 sep = b3_add(b3_add(b3_sub(bb->delta_pos, ba->delta_pos),
                b3_sub(rb, ra)), j->delta_center);
            bias3 = b3_mul(sep, j->softness.bias_rate);
            bias2x = j->softness.bias_rate * relv[step].x;
            bias2y = j->softness.bias_rate * relv[step].y;
            s = j->softness.impulse_scale / j->softness.mass_scale;
        }
        // D = G_a^T M_a^-1 G_a + G_b^T M_b^-1 G_b for the 5 rows:
        // point rows [s I, -s skew(r)], alignment rows [0, s P].
        B3Mat3 isum = b3_maddm(ia, ib);
        B3Vec3 cax[3];
        B3Vec3 cbx[3];
        B3Vec3 exs[3];
        exs[0] = b3_v(1.0f, 0.0f, 0.0f);
        exs[1] = b3_v(0.0f, 1.0f, 0.0f);
        exs[2] = b3_v(0.0f, 0.0f, 1.0f);
        for (int c = 0; c < 3; c++) {
            cax[c] = b3_cross(ra, exs[c]);
            cbx[c] = b3_cross(rb, exs[c]);
            cax[c] = b3_mv(ia, cax[c]);
            cbx[c] = b3_mv(ib, cbx[c]);
        }
        for (int i = 0; i < 3; i++) {
            for (int c = 0; c < 3; c++) {
                float v = (ma + mb) * (i == c ? 1.0f : 0.0f)
                    + b3_dot(b3_cross(ra, exs[i]), cax[c])
                    + b3_dot(b3_cross(rb, exs[i]), cbx[c]);
                D[step].m[i * 5 + c] = v;
            }
        }
        float kxx = b3_dot(pxx, b3_mv(isum, pxx));
        float kxy = b3_dot(pxx, b3_mv(isum, pyy));
        float kyy = b3_dot(pyy, b3_mv(isum, pyy));
        D[step].m[3 * 5 + 3] = kxx;
        D[step].m[3 * 5 + 4] = kxy;
        D[step].m[4 * 5 + 3] = kxy;
        D[step].m[4 * 5 + 4] = kyy;
        for (int i = 0; i < 3; i++) {
            B3Vec3 fax = b3_cross(ra, exs[i]);
            B3Vec3 fbx = b3_cross(rb, exs[i]);
            float vx = b3_dot(fax, b3_mv(ia, pxx))
                + b3_dot(fbx, b3_mv(ib, pxx));
            float vy = b3_dot(fax, b3_mv(ia, pyy))
                + b3_dot(fbx, b3_mv(ib, pyy));
            D[step].m[i * 5 + 3] = vx;
            D[step].m[3 * 5 + i] = vx;
            D[step].m[i * 5 + 4] = vy;
            D[step].m[4 * 5 + i] = vy;
        }
        // rhs = -(Cdot + bias) - s * D * lambda_old
        float lam_old[5];
        lam_old[0] = j->linear_impulse.x;
        lam_old[1] = j->linear_impulse.y;
        lam_old[2] = j->linear_impulse.z;
        lam_old[3] = j->perp_impulse.x;
        lam_old[4] = j->perp_impulse.y;
        float dl[5];
        for (int i = 0; i < 5; i++) {
            float v = 0.0f;
            for (int c = 0; c < 5; c++) {
                v += D[step].m[i * 5 + c] * lam_old[c];
            }
            dl[i] = v;
        }
        for (int i = 0; i < 25; i++) {
            D[step].m[i] *= 1.0f + s;
        }
        for (int i = 0; i < 3; i++) {
            float e[3] = {cdotp.x, cdotp.y, cdotp.z};
            rhs[step][i] = -(e[i] + (&bias3.x)[i]) - s * dl[i];
        }
        rhs[step][3] = -(b3_dot(wrel, pxx) + bias2x) - s * dl[3];
        rhs[step][4] = -(b3_dot(wrel, pyy) + bias2y) - s * dl[4];
        // Coupling to the next joint through their shared body.
        if (step + 1 < nj) {
            int shared = body_at[step + 1];
            const B3Body* bx = &w->bodies[shared];
            float mx = bx->type == B3_DYNAMIC ? bx->inv_mass : 0.0f;
            B3Mat3 ix = bx->type == B3_DYNAMIC ? bx->inv_i_world : b3_mat0();
            // Both joints' gradients evaluated AT the shared body.
            const B3Joint* jcur = &w->joints[order[step]];
            const B3Joint* jnext = &w->joints[order[step + 1]];
            B3Vec3 r1 = shared == jcur->body_a ? leva[step] : levb[step];
            B3Vec3 r2 = shared == jnext->body_a ? leva[step + 1]
                : levb[step + 1];
            float s1 = shared == jcur->body_b ? 1.0f : -1.0f;
            float s2 = shared == jnext->body_a ? -1.0f : 1.0f;
            B3Vec3 p1x = pp[step][0];
            B3Vec3 p1y = pp[step][1];
            B3Vec3 p2x = pp[step + 1][0];
            B3Vec3 p2y = pp[step + 1][1];
            float sc = s1 * s2;
            for (int i = 0; i < 3; i++) {
                for (int c = 0; c < 3; c++) {
                    float v = sc * (mx * (i == c ? 1.0f : 0.0f)
                        + b3_dot(b3_cross(r1, exs[i]),
                            b3_mv(ix, b3_cross(r2, exs[c]))));
                    Bc[step].m[i * 5 + c] = v;
                }
                B3Vec3 t1x = b3_mv(ix, p2x);
                B3Vec3 t1y = b3_mv(ix, p2y);
                float vpx = sc * b3_dot(b3_cross(r1, exs[i]), t1x);
                float vpy = sc * b3_dot(b3_cross(r1, exs[i]), t1y);
                Bc[step].m[i * 5 + 3] = vpx;
                Bc[step].m[i * 5 + 4] = vpy;
                float upx = sc * b3_dot(p1x, b3_mv(ix, b3_cross(r2, exs[i])));
                float upy = sc * b3_dot(p1y, b3_mv(ix, b3_cross(r2, exs[i])));
                Bc[step].m[3 * 5 + i] = upx;
                Bc[step].m[4 * 5 + i] = upy;
            }
            Bc[step].m[3 * 5 + 3] = sc * b3_dot(p1x, b3_mv(ix, p2x));
            Bc[step].m[3 * 5 + 4] = sc * b3_dot(p1x, b3_mv(ix, p2y));
            Bc[step].m[4 * 5 + 3] = sc * b3_dot(p1y, b3_mv(ix, p2x));
            Bc[step].m[4 * 5 + 4] = sc * b3_dot(p1y, b3_mv(ix, p2y));
        }
    }
    // Block-Thomas forward sweep.
    for (int step = 0; step < nj; step++) {
        if (!b3_lu5(&D[step], pivs[step])) {
            return 0;
        }
        b3_lu5_apply(&D[step], pivs[step], rhs[step], yv[step]);
        if (step + 1 < nj) {
            B3Mat5 sol;
            for (int c = 0; c < 5; c++) {
                float col[5];
                float out[5];
                for (int i = 0; i < 5; i++) {
                    col[i] = Bc[step].m[i * 5 + c];
                }
                b3_lu5_apply(&D[step], pivs[step], col, out);
                for (int i = 0; i < 5; i++) {
                    sol.m[i * 5 + c] = out[i];
                }
            }
            for (int i = 0; i < 5; i++) {
                for (int c = 0; c < 5; c++) {
                    float v = 0.0f;
                    for (int q = 0; q < 5; q++) {
                        v += Bc[step].m[q * 5 + i] * sol.m[q * 5 + c];
                    }
                    D[step + 1].m[i * 5 + c] -= v;
                }
                float v = 0.0f;
                for (int q = 0; q < 5; q++) {
                    v += Bc[step].m[q * 5 + i] * yv[step][q];
                }
                rhs[step + 1][i] -= v;
            }
        }
    }
#ifdef B3_DUAL_TRACE
    for (int step = 0; step < ((nj < 2) ? nj : 2); step++) {
        printf("T FULL D[%d]:\n", step);
        for (int i = 0; i < 5; i++) {
            printf("T  %.5f %.5f %.5f %.5f %.5f\n", D[step].m[i*5+0],
                D[step].m[i*5+1], D[step].m[i*5+2], D[step].m[i*5+3],
                D[step].m[i*5+4]);
        }
        printf("T rhs[%d]: %.5f %.5f %.5f %.5f %.5f\n", step, rhs[step][0],
            rhs[step][1], rhs[step][2], rhs[step][3], rhs[step][4]);
        if (step + 1 < nj) {
            printf("T FULL Bc[%d]:\n", step);
            for (int i = 0; i < 5; i++) {
                printf("T  %.5f %.5f %.5f %.5f %.5f\n", Bc[step].m[i*5+0],
                    Bc[step].m[i*5+1], Bc[step].m[i*5+2], Bc[step].m[i*5+3],
                    Bc[step].m[i*5+4]);
            }
        }
    }
#endif
    // Back substitution.
    for (int step = nj - 1; step >= 0; step--) {
        for (int i = 0; i < 5; i++) {
            lam[step][i] = yv[step][i];
        }
        if (step + 1 < nj) {
            for (int c = 0; c < 5; c++) {
                float col[5];
                float out[5];
                for (int i = 0; i < 5; i++) {
                    col[i] = Bc[step].m[i * 5 + c];
                }
                b3_lu5_apply(&D[step], pivs[step], col, out);
                for (int i = 0; i < 5; i++) {
                    lam[step][i] -= out[i] * lam[step + 1][c];
                }
            }
        }
    }
#ifdef B3_DUAL_TRACE
    for (int step = 0; step < nj; step++) {
        printf("T lam[%d]=(%.3e,%.3e,%.3e,%.3e,%.3e)\n", step, lam[step][0],
            lam[step][1], lam[step][2], lam[step][3], lam[step][4]);
        for (int i = 0; i < 5; i++) b3_trace_lam[step][i] = lam[step][i];
    }
#endif
    // Apply impulses.
    for (int step = 0; step < nj; step++) {
        B3Joint* j = &w->joints[order[step]];
        B3Body* ba = &w->bodies[j->body_a];
        B3Body* bb = &w->bodies[j->body_b];
        B3Vec3 dp = b3_v(lam[step][0], lam[step][1], lam[step][2]);
        B3Vec3 da = b3_add(b3_mul(pp[step][0], lam[step][3]),
            b3_mul(pp[step][1], lam[step][4]));
        j->linear_impulse = b3_add(j->linear_impulse, dp);
        j->perp_impulse.x += lam[step][3];
        j->perp_impulse.y += lam[step][4];
        if (ba->flags & B3_FLAG_DYNAMIC) {
            ba->lin_vel = b3_msub(ba->lin_vel, ba->inv_mass, dp);
            ba->ang_vel = b3_sub(ba->ang_vel,
                b3_mv(ba->inv_i_world, b3_add(b3_cross(
                    b3_rotate(ba->delta_rot, j->frame_p_a), dp), da)));
        }
        if (bb->flags & B3_FLAG_DYNAMIC) {
            bb->lin_vel = b3_madd(bb->lin_vel, bb->inv_mass, dp);
            bb->ang_vel = b3_add(bb->ang_vel,
                b3_mv(bb->inv_i_world, b3_add(b3_cross(
                    b3_rotate(bb->delta_rot, j->frame_p_b), dp), da)));
        }
    }
    return 1;
}
#endif

B3_HD B3_INL void b3_solve_joints(B3World* w, float h, float inv_h,
        int use_bias) {
#if B3_JOINT_DUAL
    if (b3_solve_joints_dual(w, inv_h, use_bias)) {
        return;
    }
#endif
#ifdef B3_PACKED_GS
    b3_solve_joints_packed(w, h, inv_h, use_bias);
#elif defined(B3_COMPACT_GS)
    b3_solve_joints_local(w, h, inv_h, use_bias);
#else
    b3_solve_joints_global(w, h, inv_h, use_bias);
#endif
}


B3_HD B3_INL void b3_contact_from_mani(B3Contact* c, int si, int sj,
        const B3Shape* sa, const B3Shape* sb, const B3Body* ba,
        const B3Body* bb, const B3Mani* mani) {
    c->shape_a = si;
    c->shape_b = sj;
    c->body_a = sa->body;
    c->body_b = sb->body;
    c->point_count = mani->count;
    c->normal = mani->normal;
    c->tangent1 = b3_perp(mani->normal);
    c->tangent2 = b3_cross(c->tangent1, mani->normal);
    c->friction = sqrtf(sa->friction * sb->friction);
    c->restitution = sa->restitution > sb->restitution ? sa->restitution : sb->restitution;
    c->rolling = b3_maxf(sa->rolling, sb->rolling);
    c->static_contact = ba->type != B3_DYNAMIC || bb->type != B3_DYNAMIC;
    B3Vec3 ca = ba->center;
    B3Vec3 cb = bb->center;
    c->friction_impulse.x = 0.0f;
    c->friction_impulse.y = 0.0f;
    c->twist_impulse = 0.0f;
    c->rolling_impulse = b3_v(0.0f, 0.0f, 0.0f);
    for (int p = 0; p < mani->count; p++) {
        c->points[p].r_a = b3_sub(mani->p_a[p], ca);
        c->points[p].r_b = b3_sub(mani->p_b[p], cb);
        c->points[p].base_sep = mani->sep[p]
            - b3_dot(b3_sub(c->points[p].r_b, c->points[p].r_a),
                mani->normal);
        c->points[p].feature = mani->feature[p];
        c->points[p].normal_impulse = 0.0f;
        c->points[p].total_normal = 0.0f;
        c->points[p].rel_vel = 0.0f;
        c->points[p].normal_mass = 0.0f;
        c->points[p].lever = 0.0f;
    }
}

B3_HD B3_INL void b3_collision_failure_operand(float* out,
        const B3Body* body, const B3Shape* shape) {
    out[0] = body->position.x; out[1] = body->position.y; out[2] = body->position.z;
    out[3] = body->rotation.v.x; out[4] = body->rotation.v.y;
    out[5] = body->rotation.v.z; out[6] = body->rotation.s;
    out[7] = shape->local_pos.x; out[8] = shape->local_pos.y; out[9] = shape->local_pos.z;
    out[10] = shape->local_rot.v.x; out[11] = shape->local_rot.v.y;
    out[12] = shape->local_rot.v.z; out[13] = shape->local_rot.s;
    out[14] = shape->radius;
    out[15] = shape->half.x; out[16] = shape->half.y; out[17] = shape->half.z;
}

B3_HD B3_INL void b3_find_contacts(B3World* w) {
    B3Warm old[B3_MAX_CONTACTS];
    B3AABB aabb[B3_MAX_SHAPES];
    int old_n = w->contact_count;
#if B3_MERGE_WARM_CACHE
    int old_sorted = 1;
#endif
    for (int i = 0; i < old_n; i++) {
        const B3Contact* src = &w->contacts[i];
#if B3_MERGE_WARM_CACHE
        if (i > 0 && (old[i - 1].shape_a > src->shape_a
                || (old[i - 1].shape_a == src->shape_a
                    && old[i - 1].shape_b >= src->shape_b))) {
            old_sorted = 0;
        }
#endif
        old[i].shape_a = src->shape_a;
        old[i].shape_b = src->shape_b;
        old[i].point_count = src->point_count;
        old[i].friction_impulse = src->friction_impulse;
        old[i].twist_impulse = src->twist_impulse;
        old[i].rolling_impulse = src->rolling_impulse;
        for (int p = 0; p < src->point_count; p++) {
            old[i].feature[p] = src->points[p].feature;
            old[i].normal_impulse[p] = src->points[p].normal_impulse;
        }
    }
    w->contact_count = 0;
#if B3_MERGE_WARM_CACHE
    int old_cursor = 0;
#endif
    uint64_t connected[B3_MAX_BODIES][B3_CONNECT_WORDS];
    for (int i = 0; i < w->body_count; i++) {
        for (int wdi = 0; wdi < B3_CONNECT_WORDS; wdi++) {
            connected[i][wdi] = 0;
        }
    }
    for (int i = 0; i < w->joint_count; i++) {
        const B3Joint* jn = &w->joints[i];
        if (jn->collide_connected) {
            continue;
        }
        connected[jn->body_a][jn->body_b >> 6] |= (1ull << (jn->body_b & 63));
        connected[jn->body_b][jn->body_a >> 6] |= (1ull << (jn->body_a & 63));
    }
    B3Vec3 pad = b3_v(B3_SPECULATIVE, B3_SPECULATIVE, B3_SPECULATIVE);
    for (int i = 0; i < w->shape_count; i++) {
        aabb[i] = b3_shape_aabb(&w->bodies[w->shapes[i].body], &w->shapes[i]);
        aabb[i].lo = b3_sub(aabb[i].lo, pad);
        aabb[i].hi = b3_add(aabb[i].hi, pad);
    }
    for (int i = 0; i < w->shape_count; i++) {
        B3Shape* sa = &w->shapes[i];
        B3Body* ba = &w->bodies[sa->body];
        B3AABB aa = aabb[i];
        for (int j = i + 1; j < w->shape_count; j++) {
            B3Shape* sb = &w->shapes[j];
            if (sa->body == sb->body
                    || (connected[sa->body][sb->body >> 6]
                        & (1ull << (sb->body & 63))) != 0) {
                continue;
            }
            B3Body* bb = &w->bodies[sb->body];
            int mask_a = (sa->category & sb->mask) != 0;
            int mask_b = (sb->category & sa->mask) != 0;
            if ((ba->type == B3_STATIC && bb->type == B3_STATIC)
#if B3_MUJOCO_COLLISION_FILTER
                    || !(mask_a || mask_b)
#else
                    || !(mask_a && mask_b)
#endif
                    || !b3_aabb_overlap(aa, aabb[j])) {
                continue;
            }
            B3Mani mani;
            b3_collide_shapes(&mani, w, sa, sb);
            if (sa->type == B3_CYLINDER || sb->type == B3_CYLINDER) {
                w->collision_cylinder_queries++;
            }
            if (mani.status != 0 && w->collision_failure_meta[0] == 0) {
                w->collision_failure_meta[0] = int(mani.status);
                w->collision_failure_meta[1] = i;
                w->collision_failure_meta[2] = j;
                w->collision_failure_meta[3] = sa->type;
                w->collision_failure_meta[4] = sb->type;
                w->collision_failure_meta[5] = mani.gjk_iterations;
                w->collision_failure_meta[6] = mani.epa_iterations;
                b3_collision_failure_operand(w->collision_failure_data, ba, sa);
                b3_collision_failure_operand(w->collision_failure_data + 18, bb, sb);
            }
            w->collision_status |= mani.status;
            w->collision_failed_queries += mani.status != 0;
            if (mani.gjk_iterations > w->collision_max_gjk_iterations)
                w->collision_max_gjk_iterations = mani.gjk_iterations;
            if (mani.epa_iterations > w->collision_max_epa_iterations)
                w->collision_max_epa_iterations = mani.epa_iterations;
            if (mani.count == 0) {
                continue;
            }
            if (w->contact_count >= B3_MAX_CONTACTS) {
                w->collision_status |= B3_COLLISION_CONTACT_CAPACITY;
                w->collision_contact_overflows++;
                continue;
            }
            B3Contact* c = &w->contacts[w->contact_count++];
            b3_contact_from_mani(c, i, j, sa, sb, ba, bb, &mani);
#if B3_MERGE_WARM_CACHE
            // Both generated streams are ordered by (shape_a, shape_b).
            // Merge them in O(old contacts + new contacts), rather than
            // scanning the entire old cache for every new contact. Retain
            // the original scan if a caller reordered/duplicated contacts.
            int first = 0;
            int end = old_n;
            if (old_sorted) {
                while (old_cursor < old_n && (old[old_cursor].shape_a < i
                        || (old[old_cursor].shape_a == i
                            && old[old_cursor].shape_b < j))) {
                    old_cursor++;
                }
                first = old_cursor;
                end = old_cursor < old_n ? old_cursor + 1 : old_n;
            }
            for (int k = first; k < end; k++) {
#else
            for (int k = 0; k < old_n; k++) {
#endif
                if (old[k].shape_a != i || old[k].shape_b != j) {
                    continue;
                }
                for (int p = 0; p < c->point_count; p++) {
                    for (int q = 0; q < old[k].point_count; q++) {
                        if (c->points[p].feature == old[k].feature[q]) {
                            c->points[p].normal_impulse =
                                old[k].normal_impulse[q];
                        }
                    }
                }
                c->friction_impulse = old[k].friction_impulse;
                c->twist_impulse = old[k].twist_impulse;
                c->rolling_impulse = old[k].rolling_impulse;
            }
        }
    }
}

B3_HD B3_INL void b3_prepare_contacts(B3World* w, B3Soft contact_s,
        B3Soft static_s) {
    for (int i = 0; i < w->contact_count; i++) {
        B3Contact* c = &w->contacts[i];
        b3_prepare_one_contact(c, &w->bodies[c->body_a],
            &w->bodies[c->body_b], contact_s, static_s);
    }
}

B3_HD B3_INL void b3_apply_impulse(B3Body* b, float m, B3Mat3 i,
        B3Vec3 r, B3Vec3 p, int sign) {
    if ((b->flags & B3_FLAG_DYNAMIC) == 0) {
        return;
    }
    if (sign > 0) {
        b->lin_vel = b3_madd(b->lin_vel, m, p);
        b->ang_vel = b3_add(b->ang_vel, b3_mv(i, b3_cross(r, p)));
    } else {
        b->lin_vel = b3_msub(b->lin_vel, m, p);
        b->ang_vel = b3_sub(b->ang_vel, b3_mv(i, b3_cross(r, p)));
    }
}

B3_HD B3_INL void b3_warm_start(B3World* w) {
#if B3_ART_CONTACTS
    if (w->joint_count > 0 && w->contact_count > 0) {
        B3Art art;
        if (b3_art_bind(&art, w)) {
            for (int i = 0; i < w->contact_count; i++) {
                B3Contact* c = &w->contacts[i];
                B3ArtRow row;
                for (int p = 0; p < c->point_count; p++) {
                    B3Point* cp = &c->points[p];
                    if (b3_art_make_row(&art, c->body_a, c->body_b,
                            cp->r_a, cp->r_b, c->normal, &row))
                        b3_art_apply_impulse(&art, w, &row, cp->normal_impulse);
                }
                B3Vec3 t[2] = {c->tangent1, c->tangent2};
                float f[2] = {c->friction_impulse.x, c->friction_impulse.y};
                for (int k = 0; k < 2; k++) {
                    if (b3_art_make_row(&art, c->body_a, c->body_b,
                            c->center_a, c->center_b, t[k], &row))
                        b3_art_apply_impulse(&art, w, &row, f[k]);
                    row.torque = 1;
                    b3_art_apply_impulse(&art, w, &row,
                        b3_dot(c->rolling_impulse, t[k]));
                }
                if (b3_art_make_row(&art, c->body_a, c->body_b,
                        b3_v(0, 0, 0), b3_v(0, 0, 0), c->normal, &row)) {
                    row.torque = 1;
                    b3_art_apply_impulse(&art, w, &row, c->twist_impulse);
                }
            }
            return;
        }
    }
#endif
    for (int i = 0; i < w->contact_count; i++) {
        B3Contact* c = &w->contacts[i];
        b3_warm_one_contact(c, &w->bodies[c->body_a], &w->bodies[c->body_b]);
    }
}

B3_HD B3_INL void b3_prepare_one_contact(B3Contact* c, const B3Body* ba,
        const B3Body* bb, B3Soft contact_s, B3Soft static_s) {
    float ma = ba->type == B3_DYNAMIC ? ba->inv_mass : 0.0f;
    float mb = bb->type == B3_DYNAMIC ? bb->inv_mass : 0.0f;
    B3Mat3 ia = ba->type == B3_DYNAMIC ? ba->inv_i_world : b3_mat0();
    B3Mat3 ib = bb->type == B3_DYNAMIC ? bb->inv_i_world : b3_mat0();
    c->inv_mass_a = ma;
    c->inv_mass_b = mb;
    c->inv_i_a = ia;
    c->inv_i_b = ib;
    c->rolling_mass = b3_invert3(b3_maddm(ia, ib));
    c->softness = c->static_contact ? static_s : contact_s;
    B3Vec3 n = c->normal;
    B3Vec3 center_a = b3_v(0.0f, 0.0f, 0.0f);
    B3Vec3 center_b = b3_v(0.0f, 0.0f, 0.0f);
    float wsum = 0.0f;
    float inv_tau = 1.0f / B3_SPECULATIVE;
    for (int p = 0; p < c->point_count; p++) {
        B3Point* cp = &c->points[p];
        B3Vec3 rn_a = b3_cross(cp->r_a, n);
        B3Vec3 rn_b = b3_cross(cp->r_b, n);
        float kn = ma + mb + b3_dot(rn_a, b3_mv(ia, rn_a))
            + b3_dot(rn_b, b3_mv(ib, rn_b));
        cp->normal_mass = kn > 0.0f ? 1.0f / kn : 0.0f;
        B3Vec3 vra = b3_add(ba->lin_vel, b3_cross(ba->ang_vel, cp->r_a));
        B3Vec3 vrb = b3_add(bb->lin_vel, b3_cross(bb->ang_vel, cp->r_b));
        cp->rel_vel = b3_dot(n, b3_sub(vrb, vra));
        cp->total_normal = 0.0f;
        float sep = cp->base_sep
            + b3_dot(b3_sub(cp->r_b, cp->r_a), n);
        float weight = b3_clamp(2.0f - sep * inv_tau,
            B3_MIN_FRICTION_W, 1.0f);
        center_a = b3_madd(center_a, weight, cp->r_a);
        center_b = b3_madd(center_b, weight, cp->r_b);
        wsum += weight;
    }
    float invw = wsum > 0.0f ? 1.0f / wsum : 0.0f;
    c->center_a = b3_mul(center_a, invw);
    c->center_b = b3_mul(center_b, invw);
    for (int p = 0; p < c->point_count; p++) {
        c->points[p].lever = b3_len(
            b3_sub(c->points[p].r_a, c->center_a));
    }
    B3Vec3 rt_a1 = b3_cross(c->center_a, c->tangent1);
    B3Vec3 rt_a2 = b3_cross(c->center_a, c->tangent2);
    B3Vec3 rt_b1 = b3_cross(c->center_b, c->tangent1);
    B3Vec3 rt_b2 = b3_cross(c->center_b, c->tangent2);
    B3Mat2 k;
    k.cx.x = ma + mb + b3_dot(rt_a1, b3_mv(ia, rt_a1))
        + b3_dot(rt_b1, b3_mv(ib, rt_b1));
    k.cy.y = ma + mb + b3_dot(rt_a2, b3_mv(ia, rt_a2))
        + b3_dot(rt_b2, b3_mv(ib, rt_b2));
    k.cx.y = k.cy.x = b3_dot(rt_a1, b3_mv(ia, rt_a2))
        + b3_dot(rt_b1, b3_mv(ib, rt_b2));
    c->tangent_mass = b3_invert2(k);
    float kt = b3_dot(n, b3_mv(b3_maddm(ia, ib), n));
    c->twist_mass = kt > 0.0f ? 1.0f / kt : 0.0f;
}

B3_HD B3_INL void b3_warm_one_contact(B3Contact* c, B3Body* ba, B3Body* bb) {
    B3Vec3 n = c->normal;
    for (int p = 0; p < c->point_count; p++) {
        B3Point* cp = &c->points[p];
        B3Vec3 P = b3_mul(n, cp->normal_impulse);
        b3_apply_impulse(ba, c->inv_mass_a, c->inv_i_a, cp->r_a, P, -1);
        b3_apply_impulse(bb, c->inv_mass_b, c->inv_i_b, cp->r_b, P, 1);
    }
    B3Vec3 f = b3_add(b3_mul(c->tangent1, c->friction_impulse.x),
        b3_mul(c->tangent2, c->friction_impulse.y));
    b3_apply_impulse(ba, c->inv_mass_a, c->inv_i_a, c->center_a, f, -1);
    b3_apply_impulse(bb, c->inv_mass_b, c->inv_i_b, c->center_b, f, 1);
    if ((ba->flags & B3_FLAG_DYNAMIC) != 0) {
        ba->ang_vel = b3_sub(ba->ang_vel,
            b3_mv(c->inv_i_a, b3_add(b3_mul(n, c->twist_impulse), c->rolling_impulse)));
    }
    if ((bb->flags & B3_FLAG_DYNAMIC) != 0) {
        bb->ang_vel = b3_add(bb->ang_vel,
            b3_mv(c->inv_i_b, b3_add(b3_mul(n, c->twist_impulse), c->rolling_impulse)));
    }
}

B3_HD B3_INL void b3_solve_one_contact(B3Contact* c, B3Body* ba, B3Body* bb,
        float inv_h, float contact_speed, int use_bias) {
    B3Vec3 va = ba->lin_vel;
    B3Vec3 wa = ba->ang_vel;
    B3Vec3 vb = bb->lin_vel;
    B3Vec3 wb = bb->ang_vel;
    B3Quat dqa = ba->delta_rot;
    B3Quat dqb = bb->delta_rot;
    B3Vec3 dp = b3_sub(bb->delta_pos, ba->delta_pos);
    B3Vec3 n = c->normal;
    float total_n = 0.0f;
    float twist_lim = 0.0f;
    for (int p = 0; p < c->point_count; p++) {
        B3Point* cp = &c->points[p];
        B3Vec3 ra = cp->r_a;
        B3Vec3 rb = cp->r_b;
        B3Vec3 ds = b3_add(dp, b3_sub(b3_rotate(dqb, rb),
            b3_rotate(dqa, ra)));
        float sep = b3_dot(ds, n) + cp->base_sep;
        float vbias = 0.0f;
        float mscale = 1.0f;
        float iscale = 0.0f;
        if (sep > 0.0f) {
            vbias = sep * inv_h;
        } else if (use_bias) {
            vbias = b3_maxf(c->softness.mass_scale
                * c->softness.bias_rate * sep, -contact_speed);
            mscale = c->softness.mass_scale;
            iscale = c->softness.impulse_scale;
        }
        B3Vec3 vra = b3_add(va, b3_cross(wa, ra));
        B3Vec3 vrb = b3_add(vb, b3_cross(wb, rb));
        float vn = b3_dot(b3_sub(vrb, vra), n);
        float dimp = -cp->normal_mass * (mscale * vn + vbias)
            - iscale * cp->normal_impulse;
        float nimp = b3_maxf(cp->normal_impulse + dimp, 0.0f);
        dimp = nimp - cp->normal_impulse;
        cp->normal_impulse = nimp;
        cp->total_normal += nimp;
        total_n += nimp;
        twist_lim += cp->lever * cp->normal_impulse;
        B3Vec3 P = b3_mul(n, dimp);
        va = b3_msub(va, c->inv_mass_a, P);
        wa = b3_sub(wa, b3_mv(c->inv_i_a, b3_cross(ra, P)));
        vb = b3_madd(vb, c->inv_mass_b, P);
        wb = b3_add(wb, b3_mv(c->inv_i_b, b3_cross(rb, P)));
    }
    if (!use_bias) {
        float twist_s = b3_dot(n, b3_sub(wb, wa));
        float max_t = c->friction * twist_lim;
        float dtw = -c->twist_mass * twist_s;
        float old_t = c->twist_impulse;
        c->twist_impulse = b3_clamp(old_t + dtw, -max_t, max_t);
        dtw = c->twist_impulse - old_t;
        wa = b3_sub(wa, b3_mv(c->inv_i_a, b3_mul(n, dtw)));
        wb = b3_add(wb, b3_mv(c->inv_i_b, b3_mul(n, dtw)));

        B3Vec3 t1 = c->tangent1;
        B3Vec3 t2 = c->tangent2;
        B3Vec3 ra = c->center_a;
        B3Vec3 rb = c->center_b;
        B3Vec3 vra = b3_add(va, b3_cross(wa, ra));
        B3Vec3 vrb = b3_add(vb, b3_cross(wb, rb));
        B3Vec3 vr = b3_sub(vrb, vra);
        B3Vec2 vt;
        vt.x = b3_dot(vr, t1);
        vt.y = b3_dot(vr, t2);
        B3Vec2 tm = b3_mv2(c->tangent_mass, vt);
        B3Vec2 ni;
        ni.x = c->friction_impulse.x - tm.x;
        ni.y = c->friction_impulse.y - tm.y;
        float max_f = c->friction * total_n;
        float fl2 = ni.x * ni.x + ni.y * ni.y;
        if (fl2 > max_f * max_f && fl2 > 0.0f) {
            float sc = b3_rsqrt_scale(max_f, fl2);
            ni.x *= sc;
            ni.y *= sc;
        }
        B3Vec2 df;
        df.x = ni.x - c->friction_impulse.x;
        df.y = ni.y - c->friction_impulse.y;
        c->friction_impulse = ni;
        B3Vec3 P = b3_add(b3_mul(t1, df.x), b3_mul(t2, df.y));
        va = b3_msub(va, c->inv_mass_a, P);
        wa = b3_sub(wa, b3_mv(c->inv_i_a, b3_cross(ra, P)));
        vb = b3_madd(vb, c->inv_mass_b, P);
        wb = b3_add(wb, b3_mv(c->inv_i_b, b3_cross(rb, P)));
        b3_solve_rolling(c->tangent1, c->tangent2, c->rolling, total_n,
            c->inv_i_a, c->inv_i_b, &c->rolling_impulse, &wa, &wb);
    }
    if (ba->flags & B3_FLAG_DYNAMIC) {
        ba->lin_vel = va;
        ba->ang_vel = wa;
    }
    if (bb->flags & B3_FLAG_DYNAMIC) {
        bb->lin_vel = vb;
        bb->ang_vel = wb;
    }
}

B3_HD B3_INL void b3_solve_contacts_n(B3Contact* contacts, int n,
        B3Body* bodies, float inv_h, float contact_speed, int use_bias) {
    for (int i = 0; i < n; i++) {
        B3Contact* c = &contacts[i];
        b3_solve_one_contact(c, &bodies[c->body_a], &bodies[c->body_b],
            inv_h, contact_speed, use_bias);
    }
}

B3_HD B3_INL void b3_solve_contacts(B3World* w, float inv_h,
        float contact_speed, int use_bias) {
#if B3_ART_CONTACTS
    if (w->joint_count > 0 && w->contact_count > 0) {
        B3Art art;
        if (b3_art_bind(&art, w)) {
            b3_art_solve_contacts(&art, w, inv_h, contact_speed, use_bias, 1);
            return;
        }
    }
#endif
    b3_solve_contacts_n(w->contacts, w->contact_count, w->bodies,
        inv_h, contact_speed, use_bias);
}

B3_HD B3_INL void b3_apply_restitution(B3World* w, float threshold) {
    for (int i = 0; i < w->contact_count; i++) {
        B3Contact* c = &w->contacts[i];
        if (c->restitution == 0.0f) {
            continue;
        }
        B3Body* ba = &w->bodies[c->body_a];
        B3Body* bb = &w->bodies[c->body_b];
        B3Vec3 va = ba->lin_vel;
        B3Vec3 wa = ba->ang_vel;
        B3Vec3 vb = bb->lin_vel;
        B3Vec3 wb = bb->ang_vel;
        B3Vec3 n = c->normal;
        for (int p = 0; p < c->point_count; p++) {
            B3Point* cp = &c->points[p];
            if (cp->rel_vel > -threshold || cp->total_normal == 0.0f) {
                continue;
            }
            B3Vec3 vra = b3_add(va, b3_cross(wa, cp->r_a));
            B3Vec3 vrb = b3_add(vb, b3_cross(wb, cp->r_b));
            float vn = b3_dot(b3_sub(vrb, vra), n);
            float imp = -cp->normal_mass
                * (vn + c->restitution * cp->rel_vel);
            float nimp = b3_maxf(cp->normal_impulse + imp, 0.0f);
            imp = nimp - cp->normal_impulse;
            cp->normal_impulse = nimp;
            B3Vec3 P = b3_mul(n, imp);
            va = b3_msub(va, c->inv_mass_a, P);
            wa = b3_sub(wa, b3_mv(c->inv_i_a, b3_cross(cp->r_a, P)));
            vb = b3_madd(vb, c->inv_mass_b, P);
            wb = b3_add(wb, b3_mv(c->inv_i_b, b3_cross(cp->r_b, P)));
        }
        if (ba->flags & B3_FLAG_DYNAMIC) {
            ba->lin_vel = va;
            ba->ang_vel = wa;
        }
        if (bb->flags & B3_FLAG_DYNAMIC) {
            bb->lin_vel = vb;
            bb->ang_vel = wb;
        }
    }
}

B3_HD B3_INL void b3_integrate_velocity_state(const B3Body* b, B3Vec3 gravity,
        float h, B3Vec3* lin_vel, B3Vec3* ang_vel) {
    if (b->type != B3_DYNAMIC) {
        return;
    }
    float ld = 1.0f / (1.0f + h * b->linear_damping);
    float ad = 1.0f / (1.0f + h * b->angular_damping);
    float gs = b->inv_mass > 0.0f ? b->gravity_scale : 0.0f;
    B3Vec3 dv = b3_add(b3_mul(b->force, h * b->inv_mass),
        b3_mul(gravity, h * gs));
    *lin_vel = b3_madd(dv, ld, *lin_vel);
    B3Vec3 dw = b3_mul(b3_mv(b->inv_i_world, b->torque), h);
    *ang_vel = b3_madd(dw, ad, *ang_vel);
}

#if B3_HAS_USER_FORCES
typedef struct B3UserForceState {
    B3Vec3 force[B3_MAX_BODIES], torque[B3_MAX_BODIES];
} B3UserForceState;

B3_HD B3_INL void b3_user_forces_begin(B3World* w, float h, B3UserForceState* saved) {
    for (int i = 0; i < w->body_count; i++) {
        saved->force[i] = w->bodies[i].force;
        saved->torque[i] = w->bodies[i].torque;
    }
    B3_USER_FORCES(w, h);
}

B3_HD B3_INL void b3_user_forces_end(B3World* w, const B3UserForceState* saved) {
    for (int i = 0; i < w->body_count; i++) {
        w->bodies[i].force = saved->force[i];
        w->bodies[i].torque = saved->torque[i];
    }
}
#endif

B3_HD B3_INL void b3_integrate_velocities(B3World* w, float h) {
#if B3_HAS_USER_FORCES
    B3UserForceState saved;
    b3_user_forces_begin(w, h, &saved);
#endif
    for (int i = 0; i < w->body_count; i++) {
        B3Body* b = &w->bodies[i];
        b3_integrate_velocity_state(b, w->gravity, h, &b->lin_vel, &b->ang_vel);
    }
#if B3_HAS_USER_FORCES
    b3_user_forces_end(w, &saved);
#endif
}

B3_HD B3_INL void b3_integrate_position_state(const B3Body* b, float h,
        float max_lin, float max_ang, float max_lin2, float max_ang2,
        B3Vec3* lin_vel, B3Vec3* ang_vel, B3Vec3* delta_pos, B3Quat* delta_rot) {
    if (b->type == B3_STATIC) {
        return;
    }
    B3Vec3 v = *lin_vel;
    B3Vec3 av = *ang_vel;
    if (b->flags & B3_LOCK_LIN_X) {
        v.x = 0.0f;
    }
    if (b->flags & B3_LOCK_LIN_Y) {
        v.y = 0.0f;
    }
    if (b->flags & B3_LOCK_LIN_Z) {
        v.z = 0.0f;
    }
    if (b->flags & B3_LOCK_ANG_X) {
        av.x = 0.0f;
    }
    if (b->flags & B3_LOCK_ANG_Y) {
        av.y = 0.0f;
    }
    if (b->flags & B3_LOCK_ANG_Z) {
        av.z = 0.0f;
    }
    float v2 = b3_len2(v);
    if (v2 > max_lin2 && v2 > 0.0f) {
        v = b3_mul(v, b3_rsqrt_scale(max_lin, v2));
    }
#if !B3_UNCLAMPED_ROTATION
    float w2 = b3_len2(av);
    if (w2 > max_ang2 && w2 > 0.0f) {
        av = b3_mul(av, b3_rsqrt_scale(max_ang, w2));
    }
#endif
    *lin_vel = v;
    *ang_vel = av;
    *delta_pos = b3_madd(*delta_pos, h, v);
    *delta_rot = b3_q_integrate(*delta_rot, b3_mul(av, h));
}

B3_HD B3_INL void b3_integrate_positions(B3World* w, float h,
        float inv_dt, float max_lin) {
    float max_ang = B3_MAX_ROTATION * inv_dt;
    float max_lin2 = max_lin * max_lin;
    float max_ang2 = max_ang * max_ang;
    for (int i = 0; i < w->body_count; i++) {
        B3Body* b = &w->bodies[i];
        b3_integrate_position_state(b, h, max_lin, max_ang, max_lin2, max_ang2,
            &b->lin_vel, &b->ang_vel, &b->delta_pos, &b->delta_rot);
    }
}

B3_HD B3_INL void b3_body_fin(B3Body* b) {
    if (b->type == B3_STATIC) {
        return;
    }
    b->center = b3_add(b->center, b->delta_pos);
    b->rotation = b3_qnorm(b3_qmul(b->delta_rot, b->rotation));
    b->position = b3_sub(b->center,
        b3_rotate(b->rotation, b->local_center));
    b->delta_pos = b3_v(0.0f, 0.0f, 0.0f);
    b->delta_rot = b3_q_id();
    b->force = b3_v(0.0f, 0.0f, 0.0f);
    b->torque = b3_v(0.0f, 0.0f, 0.0f);
    if (b->type == B3_DYNAMIC) {
        b->inv_i_world = b3_world_inv_i(b->rotation, b->inv_inertia);
    }
}

B3_HD B3_INL void b3_finalize_transforms(B3World* w) {
    for (int i = 0; i < w->body_count; i++) {
        b3_body_fin(&w->bodies[i]);
    }
}

B3_HD B3_INL void b3_apply_force(B3World* w, int body, B3Vec3 force) {
    w->bodies[body].force = b3_add(w->bodies[body].force, force);
}

B3_HD B3_INL void b3_apply_torque(B3World* w, int body, B3Vec3 torque) {
    w->bodies[body].torque = b3_add(w->bodies[body].torque, torque);
}

B3_HD B3_INL void b3_apply_linear_impulse(B3World* w, int body,
        B3Vec3 impulse) {
    B3Body* b = &w->bodies[body];
    if (b->type == B3_DYNAMIC) {
        b->lin_vel = b3_madd(b->lin_vel, b->inv_mass, impulse);
    }
}

B3_HD B3_INL void b3_soft_step_params(const B3World* w, float dt, int substeps,
        float* h, float* inv_h, float* inv_dt, B3Soft* cs, B3Soft* ss) {
    int subs = substeps < 1 ? 1 : substeps;
    *h = dt / (float)subs;
    *inv_dt = 1.0f / dt;
    *inv_h = (float)subs * (*inv_dt);
    float hertz = b3_minf(w->contact_hertz, 0.25f * (*inv_h));
    *cs = b3_make_soft(hertz, w->contact_damping, *h);
    *ss = b3_make_soft(2.0f * hertz, 0.5f * w->contact_damping, *h);
}

B3_HD B3_INL void b3_step_begin(B3World* w, float h, B3Soft cs, B3Soft ss) {
    b3_find_contacts(w);
    b3_prepare_contacts(w, cs, ss);
    b3_prepare_joints(w, h);
    b3_warm_start(w);
    b3_warm_start_joints(w);
}

B3_HD B3_INL void b3_step_sub(B3World* w, float h, float inv_h, float inv_dt) {
    b3_integrate_velocities(w, h);
#ifndef B3_INTERLEAVE_CONTACTS
    b3_solve_contacts(w, inv_h, w->contact_speed, 1);
#endif
    b3_solve_joints(w, h, inv_h, 1);
    b3_integrate_positions(w, h, inv_dt, w->max_linear_speed);
#ifndef B3_INTERLEAVE_CONTACTS
    b3_solve_contacts(w, inv_h, w->contact_speed, 0);
#endif
    b3_solve_joints(w, h, inv_h, 0);
#ifndef B3_SKIP_RESTITUTION
    b3_apply_restitution(w, w->restitution_threshold);
#endif
}

B3_HD B3_INL void b3_step_indep(B3World* w, float dt, int substeps) {
    if (dt <= 0.0f) {
        return;
    }
    int subs = substeps < 1 ? 1 : substeps;
    float h, inv_h, inv_dt;
    B3Soft cs, ss;
    b3_soft_step_params(w, dt, substeps, &h, &inv_h, &inv_dt, &cs, &ss);
    b3_step_begin(w, h, cs, ss);
#if B3_PERSISTENT_GS
    B3GsBody bl[B3_MAX_BODIES];
    B3GsJoint jl[B3_MAX_JOINTS];
    B3GsContact cl[B3_MAX_CONTACTS];
    b3_gs_load(w, bl, jl, cl);
#if B3_PACKED_INTEGRATE
    float max_ang = B3_MAX_ROTATION * inv_dt;
    float max_lin = w->max_linear_speed;
    float max_lin2 = max_lin * max_lin;
    float max_ang2 = max_ang * max_ang;
    for (int s = 0; s < subs; s++) {
#if B3_HAS_USER_FORCES
        // Hooks see current packed velocities and delta poses, not last step.
        for (int i = 0; i < w->body_count; i++) {
            w->bodies[i].lin_vel = bl[i].lin_vel;
            w->bodies[i].ang_vel = bl[i].ang_vel;
            w->bodies[i].delta_pos = bl[i].delta_pos;
            w->bodies[i].delta_rot = bl[i].delta_rot;
        }
        b3_integrate_velocities(w, h);
        for (int i = 0; i < w->body_count; i++) {
            bl[i].lin_vel = w->bodies[i].lin_vel;
            bl[i].ang_vel = w->bodies[i].ang_vel;
        }
#else
        for (int i = 0; i < w->body_count; i++) {
            b3_integrate_velocity_state(&w->bodies[i], w->gravity, h,
                &bl[i].lin_vel, &bl[i].ang_vel);
        }
#endif
        // Velocity integration and relaxation leave delta poses unchanged.
        b3_gs_solve_cached(w, bl, jl, cl, h, inv_h, 1,
            !B3_REUSE_GS_CACHE || s == 0);
        for (int i = 0; i < w->body_count; i++) {
            b3_integrate_position_state(&w->bodies[i], h, max_lin, max_ang,
                max_lin2, max_ang2, &bl[i].lin_vel, &bl[i].ang_vel,
                &bl[i].delta_pos, &bl[i].delta_rot);
        }
        b3_gs_solve(w, bl, jl, cl, h, inv_h, 0);
    }
    b3_gs_store_velocities(w, bl);
    for (int i = 0; i < w->body_count; i++) {
        w->bodies[i].delta_pos = bl[i].delta_pos;
        w->bodies[i].delta_rot = bl[i].delta_rot;
    }
#else
    for (int s = 0; s < subs; s++) {
        b3_integrate_velocities(w, h);
        b3_gs_refresh_bodies(w, bl);
        // Velocity integration and relaxation leave delta poses unchanged.
        b3_gs_solve_cached(w, bl, jl, cl, h, inv_h, 1,
            !B3_REUSE_GS_CACHE || s == 0);
        b3_gs_store_velocities(w, bl);
        b3_integrate_positions(w, h, inv_dt, w->max_linear_speed);
        b3_gs_refresh_bodies(w, bl);
        b3_gs_solve(w, bl, jl, cl, h, inv_h, 0);
        b3_gs_store_velocities(w, bl);
    }
#endif
    b3_gs_store_constraints(w, jl, cl);
#else
    for (int s = 0; s < subs; s++) {
        b3_step_sub(w, h, inv_h, inv_dt);
    }
#endif
    b3_finalize_transforms(w);
}

/* The ordinary integrator retains Soft Step joint motors/springs/limits;
 * articulated contact response is selected inside the contact sweeps. */
B3_HD B3_INL void b3_step(B3World* w, float dt, int substeps) {
    b3_step_indep(w, dt, substeps);
}

#if B3_ART_CONTACTS
/* Reduced-coordinate ABA/FK for passive revolute trees and loop cuts.
 * Actuated, welded, locked or kinematic worlds use the ordinary joint solver,
 * rather than silently ignoring a contract the reduced integrator lacks. */
B3_HD B3_INL void b3_art_step(B3World* w, float dt, int substeps) {
    if (dt <= 0.0f) return;
    for (int j = 0; j < w->joint_count; j++) {
        const B3Joint* joint = &w->joints[j];
        int unsupported = joint->fixed_rotation || joint->enable_spring || joint->enable_limit;
#ifndef B3_REVOLUTE_ONLY
        unsupported |= joint->type != B3_JOINT_REVOLUTE || joint->enable_motor;
#endif
        if (unsupported) {
            b3_step_indep(w, dt, substeps);
            return;
        }
    }
    for (int i = 0; i < w->body_count; i++) {
        if (w->bodies[i].type == B3_KINEMATIC ||
                (w->bodies[i].flags & (B3_LOCK_LIN_X | B3_LOCK_LIN_Y | B3_LOCK_LIN_Z |
                    B3_LOCK_ANG_X | B3_LOCK_ANG_Y | B3_LOCK_ANG_Z))) {
            b3_step_indep(w, dt, substeps);
            return;
        }
    }
    B3Art art;
    if (!b3_art_bind(&art, w)) {
        b3_step_indep(w, dt, substeps);
        return;
    }
    int subs = substeps < 1 ? 1 : substeps;
    float h, inv_h, inv_dt;
    B3Soft cs, ss;
    b3_soft_step_params(w, dt, subs, &h, &inv_h, &inv_dt, &cs, &ss);
    b3_find_contacts(w);
    b3_prepare_contacts(w, cs, ss);
    b3_warm_start(w);
    for (int s = 0; s < subs; s++) {
#if B3_HAS_USER_FORCES
        B3UserForceState saved;
        b3_user_forces_begin(w, h, &saved);
#endif
        b3_art_integrate_vel(&art, w, h);
#if B3_HAS_USER_FORCES
        b3_user_forces_end(w, &saved);
#endif
        b3_art_solve_contacts(&art, w, inv_h, w->contact_speed, 1, B3_ART_CONTACT_ITERS);
        b3_art_solve_cuts(&art, w, inv_h, 1, B3_ART_CUT_ITERS);
        b3_art_integrate_pos(&art, w, h, inv_dt);
        b3_art_solve_contacts(&art, w, inv_h, w->contact_speed, 0, B3_ART_CONTACT_ITERS);
        b3_art_solve_cuts(&art, w, inv_h, 0, B3_ART_CUT_ITERS);
#ifndef B3_SKIP_RESTITUTION
        b3_apply_restitution(w, w->restitution_threshold);
#endif
    }
    b3_finalize_transforms(w);
}
#endif

#ifdef __CUDACC__
__global__ void b3_step_kernel(B3World* worlds, int n, float dt,
        int substeps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        b3_step(&worlds[i], dt, substeps);
    }
}
#endif
