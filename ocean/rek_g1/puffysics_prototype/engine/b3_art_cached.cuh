// Reduced-coordinate articulation: Featherstone ABA + matrix-free Delassus.
// Device/host, one world per caller (do not split a tree across a warp).
//
// Operators (Sathya, Montaut, de Mont-Marin, Carpentier, RAL 2026):
//   J^T λ   Alg. 1  RNEA backward
//   M^{-1} τ  Alg. 2  ABA
//   J qdd   Alg. 3  FK forward
//   Δx = J M^{-1} J^T x   merged two-sweep (zero bias)
// Damped inverse: dense (μ^{-1}+Δ) from matrix-free Delassus. Default
// b3_step is unchanged.
// Position: reduced-q + FK (tree hinges exact). Loops: spanning tree + cuts.
//
// Spatial convention: angular-first at each link COM, world-aligned axes.
//   motion ν = (ω, v_com), force f = (n, f)
// Revolute S at the child COM: (u, u × (com - anchor)).
// Included from puffysics.cuh after the rigid types.
// b3_art_step lives in puffysics.cuh (needs b3_step_indep).
#pragma once

// Experimental copy of b3_art.cuh SHA256
// 299dfcfcbe3de97252520b454c4db0eddc30ecb2c370163e91e5877fb13c8969.
// A cache exists only within one explicit fixed-pose contact-solver scope.
#ifndef B3_ART_FIXED_POSE_CACHE
#define B3_ART_FIXED_POSE_CACHE 1
#endif
#ifndef B3_ART_CACHE_DIAGNOSTICS
#define B3_ART_CACHE_DIAGNOSTICS 0
#endif

// Optional generalized rotor inertia supplied by the private model adapter.
#ifndef B3_ART_JOINT_ARMATURE
#define B3_ART_JOINT_ARMATURE(joint_id) 0.0f
#endif
#ifndef B3_ART_TORSION
#define B3_ART_TORSION 1
#endif

#ifndef B3_ART_MAX_LINKS
#define B3_ART_MAX_LINKS B3_MAX_BODIES
#endif
#ifndef B3_ART_MAX_Q
#define B3_ART_MAX_Q B3_MAX_JOINTS
#endif
#ifndef B3_ART_MAX_ROWS
#define B3_ART_MAX_ROWS 32
#endif
#ifndef B3_ART_MAX_DEG
#define B3_ART_MAX_DEG 8
#endif
#ifndef B3_ART_MAX_CUTS
#define B3_ART_MAX_CUTS 8
#endif
#ifndef B3_ART_CUT_ITERS
#define B3_ART_CUT_ITERS 4
#endif
#ifndef B3_ART_DMIN
#define B3_ART_DMIN 1.0e-8f
#endif

typedef struct B3Motion {
    B3Vec3 w;
    B3Vec3 v;
} B3Motion;

typedef struct B3Force {
    B3Vec3 n;
    B3Vec3 f;
} B3Force;

typedef struct B3Inertia {
    B3Mat3 ww;
    B3Mat3 wv;
    B3Mat3 vw;
    B3Mat3 vv;
} B3Inertia;

typedef struct B3ArtRow {
    int link_a;
    int link_b;
    int torque;
    B3Vec3 ra;
    B3Vec3 rb;
    B3Vec3 n;
} B3ArtRow;

typedef struct B3Art {
    int n_links;
    int n_q;
    int ok;
    int linear_factor_scope;
#if B3_ART_CACHE_DIAGNOSTICS
    unsigned linear_factor_builds;
    unsigned linear_rhs_calls;
#endif
    int parent[B3_ART_MAX_LINKS];
    int body[B3_ART_MAX_LINKS];
    int joint[B3_ART_MAX_LINKS];
    int fixed[B3_ART_MAX_LINKS];
    int floating[B3_ART_MAX_LINKS];
    B3Vec3 local_anchor_a[B3_ART_MAX_LINKS];
    B3Vec3 local_anchor_b[B3_ART_MAX_LINKS];
    B3Quat local_rot_a[B3_ART_MAX_LINKS];
    B3Quat local_rot_b[B3_ART_MAX_LINKS];
    B3Vec3 local_center[B3_ART_MAX_LINKS];
    B3Vec3 I_local[B3_ART_MAX_LINKS];
    float mass[B3_ART_MAX_LINKS];
    float gravity_scale[B3_ART_MAX_LINKS];
    B3Vec3 com[B3_ART_MAX_LINKS];
    B3Vec3 pos[B3_ART_MAX_LINKS];
    B3Quat rot[B3_ART_MAX_LINKS];
    B3Vec3 r[B3_ART_MAX_LINKS];
    B3Motion S[B3_ART_MAX_LINKS];
    B3Motion v[B3_ART_MAX_LINKS];
    B3Motion a[B3_ART_MAX_LINKS];
    B3Motion c[B3_ART_MAX_LINKS];
    B3Inertia I[B3_ART_MAX_LINKS];
    B3Inertia IA[B3_ART_MAX_LINKS];
    B3Force U[B3_ART_MAX_LINKS];
    B3Force pA[B3_ART_MAX_LINKS];
    float q[B3_ART_MAX_LINKS];
    float qd[B3_ART_MAX_LINKS];
    float qdd[B3_ART_MAX_LINKS];
    float tau[B3_ART_MAX_LINKS];
    float Dinv[B3_ART_MAX_LINKS];
    int n_cuts;
    int cut_joint[B3_ART_MAX_CUTS];
    int cut_body_a[B3_ART_MAX_CUTS];
    int cut_body_b[B3_ART_MAX_CUTS];
} B3Art;

static B3_HD B3_INL B3Motion b3_motion0(void) {
    B3Motion m;
    m.w = b3_v(0.0f, 0.0f, 0.0f);
    m.v = b3_v(0.0f, 0.0f, 0.0f);
    return m;
}

static B3_HD B3_INL B3Force b3_force0(void) {
    B3Force f;
    f.n = b3_v(0.0f, 0.0f, 0.0f);
    f.f = b3_v(0.0f, 0.0f, 0.0f);
    return f;
}

static B3_HD B3_INL B3Inertia b3_I0(void) {
    B3Inertia I;
    I.ww = b3_mat0();
    I.wv = b3_mat0();
    I.vw = b3_mat0();
    I.vv = b3_mat0();
    return I;
}

static B3_HD B3_INL B3Mat3 b3_mat3_sub(B3Mat3 a, B3Mat3 b) {
    B3Mat3 r;
    r.cx = b3_sub(a.cx, b.cx);
    r.cy = b3_sub(a.cy, b.cy);
    r.cz = b3_sub(a.cz, b.cz);
    return r;
}

static B3_HD B3_INL B3Mat3 b3_mat3_scale(B3Mat3 a, float s) {
    B3Mat3 r;
    r.cx = b3_mul(a.cx, s);
    r.cy = b3_mul(a.cy, s);
    r.cz = b3_mul(a.cz, s);
    return r;
}

static B3_HD B3_INL B3Mat3 b3_outer(B3Vec3 a, B3Vec3 b) {
    B3Mat3 r;
    r.cx = b3_mul(a, b.x);
    r.cy = b3_mul(a, b.y);
    r.cz = b3_mul(a, b.z);
    return r;
}

static B3_HD B3_INL void b3_I_set_col(B3Inertia* I, int col, B3Force f) {
    B3Mat3* ang = col < 3 ? &I->ww : &I->wv;
    B3Mat3* lin = col < 3 ? &I->vw : &I->vv;
    int c = col < 3 ? col : col - 3;
    if (c == 0) {
        ang->cx = f.n;
        lin->cx = f.f;
    } else if (c == 1) {
        ang->cy = f.n;
        lin->cy = f.f;
    } else {
        ang->cz = f.n;
        lin->cz = f.f;
    }
}

static B3_HD B3_INL B3Force b3_I_mul(B3Inertia I, B3Motion m) {
    B3Force f;
    f.n = b3_add(b3_mv(I.ww, m.w), b3_mv(I.wv, m.v));
    f.f = b3_add(b3_mv(I.vw, m.w), b3_mv(I.vv, m.v));
    return f;
}

static B3_HD B3_INL B3Inertia b3_I_add(B3Inertia a, B3Inertia b) {
    B3Inertia r;
    r.ww = b3_maddm(a.ww, b.ww);
    r.wv = b3_maddm(a.wv, b.wv);
    r.vw = b3_maddm(a.vw, b.vw);
    r.vv = b3_maddm(a.vv, b.vv);
    return r;
}

static B3_HD B3_INL B3Inertia b3_I_shift(B3Inertia I, B3Vec3 r) {
    B3Inertia O = b3_I0();
    for (int k = 0; k < 6; k++) {
        B3Motion m = b3_motion0();
        if (k < 3) {
            m.w = b3_v(k == 0 ? 1.0f : 0.0f, k == 1 ? 1.0f : 0.0f,
                k == 2 ? 1.0f : 0.0f);
        } else {
            int t = k - 3;
            m.v = b3_v(t == 0 ? 1.0f : 0.0f, t == 1 ? 1.0f : 0.0f,
                t == 2 ? 1.0f : 0.0f);
        }
        m.v = b3_add(m.v, b3_cross(m.w, r));
        B3Force f = b3_I_mul(I, m);
        B3Force fp;
        fp.n = b3_add(f.n, b3_cross(r, f.f));
        fp.f = f.f;
        b3_I_set_col(&O, k, fp);
    }
    return O;
}

static B3_HD B3_INL B3Inertia b3_I_rank1(B3Inertia I, B3Force U, float dinv) {
    I.ww = b3_mat3_sub(I.ww, b3_mat3_scale(b3_outer(U.n, U.n), dinv));
    I.wv = b3_mat3_sub(I.wv, b3_mat3_scale(b3_outer(U.n, U.f), dinv));
    I.vw = b3_mat3_sub(I.vw, b3_mat3_scale(b3_outer(U.f, U.n), dinv));
    I.vv = b3_mat3_sub(I.vv, b3_mat3_scale(b3_outer(U.f, U.f), dinv));
    return I;
}

static B3_HD B3_INL float b3_S_dot(B3Motion S, B3Force f) {
    return b3_dot(S.w, f.n) + b3_dot(S.v, f.f);
}

static B3_HD B3_INL B3Motion b3_S_mul(B3Motion S, float s) {
    B3Motion m;
    m.w = b3_mul(S.w, s);
    m.v = b3_mul(S.v, s);
    return m;
}

static B3_HD B3_INL int b3_solve6(const float A[36], const float b[6],
        float x[6]) {
    float M[6][7];
    for (int r = 0; r < 6; r++) {
        for (int c = 0; c < 6; c++) {
            M[r][c] = A[r * 6 + c];
        }
        M[r][6] = b[r];
    }
    for (int k = 0; k < 6; k++) {
        int piv = k;
        float best = fabsf(M[k][k]);
        for (int r = k + 1; r < 6; r++) {
            float v = fabsf(M[r][k]);
            if (v > best) {
                best = v;
                piv = r;
            }
        }
        if (best < 1.0e-12f) {
            return 0;
        }
        if (piv != k) {
            for (int c = k; c < 7; c++) {
                float t = M[k][c];
                M[k][c] = M[piv][c];
                M[piv][c] = t;
            }
        }
        float inv = 1.0f / M[k][k];
        for (int c = k; c < 7; c++) {
            M[k][c] *= inv;
        }
        for (int r = 0; r < 6; r++) {
            if (r == k) {
                continue;
            }
            float s = M[r][k];
            for (int c = k; c < 7; c++) {
                M[r][c] -= s * M[k][c];
            }
        }
    }
    for (int r = 0; r < 6; r++) {
        x[r] = M[r][6];
    }
    return 1;
}

static B3_HD B3_INL void b3_I_pack(B3Inertia I, float A[36]) {
    for (int c = 0; c < 3; c++) {
        B3Vec3 n = c == 0 ? I.ww.cx : (c == 1 ? I.ww.cy : I.ww.cz);
        B3Vec3 f = c == 0 ? I.vw.cx : (c == 1 ? I.vw.cy : I.vw.cz);
        A[0 * 6 + c] = n.x;
        A[1 * 6 + c] = n.y;
        A[2 * 6 + c] = n.z;
        A[3 * 6 + c] = f.x;
        A[4 * 6 + c] = f.y;
        A[5 * 6 + c] = f.z;
    }
    for (int c = 0; c < 3; c++) {
        B3Vec3 n = c == 0 ? I.wv.cx : (c == 1 ? I.wv.cy : I.wv.cz);
        B3Vec3 f = c == 0 ? I.vv.cx : (c == 1 ? I.vv.cy : I.vv.cz);
        A[0 * 6 + 3 + c] = n.x;
        A[1 * 6 + 3 + c] = n.y;
        A[2 * 6 + 3 + c] = n.z;
        A[3 * 6 + 3 + c] = f.x;
        A[4 * 6 + 3 + c] = f.y;
        A[5 * 6 + 3 + c] = f.z;
    }
}

static B3_HD B3_INL int b3_I_solve(B3Inertia I, B3Force rhs, B3Motion* a) {
    float A[36], b[6], x[6];
    b3_I_pack(I, A);
    b[0] = rhs.n.x;
    b[1] = rhs.n.y;
    b[2] = rhs.n.z;
    b[3] = rhs.f.x;
    b[4] = rhs.f.y;
    b[5] = rhs.f.z;
    if (!b3_solve6(A, b, x)) {
        *a = b3_motion0();
        return 0;
    }
    a->w = b3_v(x[0], x[1], x[2]);
    a->v = b3_v(x[3], x[4], x[5]);
    return 1;
}

static B3_HD B3_INL int b3_art_clear(B3Art* art) {
    memset(art, 0, sizeof(*art));
    for (int i = 0; i < B3_ART_MAX_LINKS; i++) {
        art->parent[i] = -1;
        art->joint[i] = -1;
    }
    return 1;
}

static B3_HD B3_INL void b3_art_link_from_body(B3Art* art, int li,
        const B3Body* bd) {
    art->mass[li] = bd->inv_mass > 0.0f ? 1.0f / bd->inv_mass : 0.0f;
    art->I_local[li] = b3_v(
        bd->inv_inertia.x > 0.0f ? 1.0f / bd->inv_inertia.x : 0.0f,
        bd->inv_inertia.y > 0.0f ? 1.0f / bd->inv_inertia.y : 0.0f,
        bd->inv_inertia.z > 0.0f ? 1.0f / bd->inv_inertia.z : 0.0f);
    art->local_center[li] = bd->local_center;
    art->gravity_scale[li] = bd->gravity_scale;
}

static B3_HD B3_INL int b3_art_from_world(B3Art* art, const B3World* w) {
    b3_art_clear(art);
    int deg[B3_MAX_BODIES];
    int adj_b[B3_MAX_BODIES][B3_ART_MAX_DEG];
    int adj_j[B3_MAX_BODIES][B3_ART_MAX_DEG];
    int body_link[B3_MAX_BODIES];
    unsigned char seen[B3_MAX_BODIES];
    for (int i = 0; i < B3_MAX_BODIES; i++) {
        deg[i] = 0;
        body_link[i] = -1;
        seen[i] = 0;
    }
    for (int k = 0; k < w->joint_count; k++) {
        const B3Joint* j = &w->joints[k];
#ifndef B3_REVOLUTE_ONLY
        if (j->type != B3_JOINT_REVOLUTE) {
            continue;
        }
#endif
        if (j->body_a == j->body_b) {
            art->ok = 0;
            return 0;
        }
        int a = j->body_a;
        int b = j->body_b;
        if (deg[a] >= B3_ART_MAX_DEG || deg[b] >= B3_ART_MAX_DEG) {
            art->ok = 0;
            return 0;
        }
        adj_b[a][deg[a]] = b;
        adj_j[a][deg[a]] = k;
        deg[a]++;
        adj_b[b][deg[b]] = a;
        adj_j[b][deg[b]] = k;
        deg[b]++;
    }

    int q[B3_MAX_BODIES];
    int parent_body[B3_MAX_BODIES];
    int parent_joint[B3_MAX_BODIES];
    for (int i = 0; i < B3_MAX_BODIES; i++) {
        parent_body[i] = -1;
        parent_joint[i] = -1;
    }

    for (int pass = 0; pass < 2; pass++) {
        for (int s = 0; s < w->body_count; s++) {
            int is_static = (w->bodies[s].type != B3_DYNAMIC);
            if (deg[s] == 0 || seen[s] || is_static != (pass == 0)) {
                continue;
            }
            int qh = 0, qt = 0;
            q[qt++] = s;
            seen[s] = 1;
            parent_body[s] = -1;
            while (qh < qt) {
                int u = q[qh++];
                for (int e = 0; e < deg[u]; e++) {
                    int v = adj_b[u][e];
                    if (seen[v]) {
                        if (parent_body[u] != v && parent_body[v] != u) {
                            int jk = adj_j[u][e];
                            int dup = 0;
                            for (int c = 0; c < art->n_cuts; c++) {
                                if (art->cut_joint[c] == jk) {
                                    dup = 1;
                                    break;
                                }
                            }
                            if (!dup) {
                                if (art->n_cuts >= B3_ART_MAX_CUTS) {
                                    art->ok = 0;
                                    return 0;
                                }
                                int ci = art->n_cuts++;
                                art->cut_joint[ci] = jk;
                                art->cut_body_a[ci] = w->joints[jk].body_a;
                                art->cut_body_b[ci] = w->joints[jk].body_b;
                            }
                        }
                        continue;
                    }
                    seen[v] = 1;
                    parent_body[v] = u;
                    parent_joint[v] = adj_j[u][e];
                    q[qt++] = v;
                }
            }
            for (int t = 0; t < qt; t++) {
                int b = q[t];
                if (art->n_links >= B3_ART_MAX_LINKS) {
                    art->ok = 0;
                    return 0;
                }
                int li = art->n_links++;
                body_link[b] = li;
                art->body[li] = b;
                const B3Body* bd = &w->bodies[b];
                art->fixed[li] = (bd->type != B3_DYNAMIC);
                art->floating[li] = (parent_body[b] < 0 && !art->fixed[li]);
                b3_art_link_from_body(art, li, bd);
                if (parent_body[b] < 0) {
                    art->parent[li] = -1;
                    art->joint[li] = -1;
                } else {
                    art->parent[li] = body_link[parent_body[b]];
                    int jk = parent_joint[b];
                    art->joint[li] = jk;
                    const B3Joint* j = &w->joints[jk];
                    if (j->body_b == b) {
                        art->local_anchor_a[li] = j->local_anchor_a;
                        art->local_anchor_b[li] = j->local_anchor_b;
                        art->local_rot_a[li] = j->local_rot_a;
                        art->local_rot_b[li] = j->local_rot_b;
                    } else {
                        art->local_anchor_a[li] = j->local_anchor_b;
                        art->local_anchor_b[li] = j->local_anchor_a;
                        art->local_rot_a[li] = j->local_rot_b;
                        art->local_rot_b[li] = j->local_rot_a;
                    }
                    art->n_q++;
                }
            }
        }
    }
    for (int s = 0; s < w->body_count; s++) {
        if (seen[s] || w->bodies[s].type != B3_DYNAMIC) {
            continue;
        }
        if (art->n_links >= B3_ART_MAX_LINKS) {
            art->ok = 0;
            return 0;
        }
        int li = art->n_links++;
        body_link[s] = li;
        art->body[li] = s;
        const B3Body* bd = &w->bodies[s];
        art->fixed[li] = 0;
        art->floating[li] = 1;
        art->parent[li] = -1;
        art->joint[li] = -1;
        b3_art_link_from_body(art, li, bd);
        seen[s] = 1;
    }
    art->ok = art->n_links > 0;
    return art->ok;
}

static B3_HD B3_INL int b3_art_bind(B3Art* art, const B3World* w) {
    return w->joint_count > 0 && b3_art_from_world(art, w);
}

static B3_HD B3_INL void b3_art_lived_pose(const B3Body* b, B3Vec3 lc,
        B3Vec3* pos, B3Quat* rot, B3Vec3* com) {
    *rot = b3_qnorm(b3_qmul(b->delta_rot, b->rotation));
    *com = b3_add(b->center, b->delta_pos);
    *pos = b3_sub(*com, b3_rotate(*rot, lc));
}

static B3_HD B3_INL float b3_art_joint_q(B3Quat rp, B3Quat rc,
        B3Quat la, B3Quat lb) {
    B3Quat qa = b3_qmul(rp, la);
    B3Quat qb = b3_qmul(rc, lb);
    if (b3_qdot(qa, qb) < 0.0f) {
        qb = b3_qneg(qb);
    }
    return b3_twist(b3_qinv_mul(qa, qb));
}

static B3_HD B3_INL void b3_art_fk_link(B3Art* art, int i) {
    art->linear_factor_scope = 0;
    int p = art->parent[i];
    if (p < 0) {
        return;
    }
    B3Quat rz = b3_q_axis_angle(b3_v(0.0f, 0.0f, 1.0f), art->q[i]);
    B3Quat rot = b3_qnorm(b3_qmul(art->rot[p],
        b3_qmul(art->local_rot_a[i],
        b3_qmul(rz, b3_qconj(art->local_rot_b[i])))));
    B3Vec3 anchor = b3_xf_point(art->pos[p], art->rot[p],
        art->local_anchor_a[i]);
    B3Vec3 com = b3_add(anchor, b3_rotate(rot,
        b3_sub(art->local_center[i], art->local_anchor_b[i])));
    art->rot[i] = rot;
    art->com[i] = com;
    art->pos[i] = b3_sub(com, b3_rotate(rot, art->local_center[i]));
}

static B3_HD B3_INL void b3_art_refresh(B3Art* art, const B3World* w) {
    art->linear_factor_scope = 0;
    for (int i = 0; i < art->n_links; i++) {
        const B3Body* b = &w->bodies[art->body[i]];
        b3_art_lived_pose(b, art->local_center[i],
            &art->pos[i], &art->rot[i], &art->com[i]);
        art->v[i].w = b->ang_vel;
        art->v[i].v = b->lin_vel;
        if (art->fixed[i]) {
            art->I[i] = b3_I0();
            art->mass[i] = 0.0f;
        } else {
            B3Mat3 Iw = b3_world_inv_i(art->rot[i], art->I_local[i]);
            art->I[i] = b3_I0();
            art->I[i].ww = Iw;
            float m = art->mass[i];
            art->I[i].vv.cx = b3_v(m, 0.0f, 0.0f);
            art->I[i].vv.cy = b3_v(0.0f, m, 0.0f);
            art->I[i].vv.cz = b3_v(0.0f, 0.0f, m);
        }
        art->tau[i] = 0.0f;
        art->qdd[i] = 0.0f;
        art->S[i] = b3_motion0();
        art->r[i] = b3_v(0.0f, 0.0f, 0.0f);
        art->qd[i] = 0.0f;
        art->q[i] = 0.0f;
        art->c[i] = b3_motion0();
        int p = art->parent[i];
        if (p < 0) {
            continue;
        }
        B3Vec3 anchor = b3_xf_point(art->pos[p], art->rot[p],
            art->local_anchor_a[i]);
        B3Vec3 axis_local = b3_rotate(art->local_rot_a[i],
            b3_v(0.0f, 0.0f, 1.0f));
        B3Vec3 u = b3_norm(b3_rotate(art->rot[p], axis_local));
        art->r[i] = b3_sub(art->com[i], art->com[p]);
        art->S[i].w = u;
        art->S[i].v = b3_cross(u, b3_sub(art->com[i], anchor));
        B3Vec3 wrel = b3_sub(art->v[i].w, art->v[p].w);
        art->qd[i] = b3_dot(wrel, u);
        art->q[i] = b3_art_joint_q(art->rot[p], art->rot[i],
            art->local_rot_a[i], art->local_rot_b[i]);
    }
}

static B3_HD B3_INL void b3_art_seed_bias(B3Art* art, const B3World* w,
        int linear) {
    B3Vec3 g = w->gravity;
    for (int i = 0; i < art->n_links; i++) {
        art->IA[i] = art->I[i];
        art->pA[i] = b3_force0();
        if (art->fixed[i]) {
            art->c[i] = b3_motion0();
            continue;
        }
        const B3Body* b = &w->bodies[art->body[i]];
        B3Force ext = b3_force0();
        if (!linear) {
            ext.f = b3_add(b->force, b3_mul(g, art->mass[i] * art->gravity_scale[i]));
            ext.n = b->torque;
            B3Force Iw = b3_I_mul(art->I[i], art->v[i]);
            art->pA[i].n = b3_cross(art->v[i].w, Iw.n);
            // Linear velocity and acceleration are world-aligned COM values.
            // A force-free translating body has no omega cross (mass*v) bias.
        }
        art->pA[i].n = b3_sub(art->pA[i].n, ext.n);
        art->pA[i].f = b3_sub(art->pA[i].f, ext.f);

        int p = art->parent[i];
        if (p < 0 || linear) {
            art->c[i] = b3_motion0();
            continue;
        }
        B3Motion Sqd = b3_S_mul(art->S[i], art->qd[i]);
        B3Vec3 wp = art->v[p].w;
        art->c[i].w = b3_cross(art->v[i].w, Sqd.w);
        art->c[i].v = b3_add(b3_cross(b3_add(art->v[i].w, wp), Sqd.v),
            b3_cross(wp, b3_cross(wp, art->r[i])));
    }
}

static B3_HD B3_INL void b3_art_backward(B3Art* art) {
    for (int i = art->n_links - 1; i >= 0; i--) {
        int p = art->parent[i];
        if (p < 0) {
            continue;
        }
        B3Motion S = art->S[i];
        art->U[i] = b3_I_mul(art->IA[i], S);
        float D = b3_S_dot(S, art->U[i]) + B3_ART_JOINT_ARMATURE(art->joint[i]);
        if (D < B3_ART_DMIN) {
            D = B3_ART_DMIN;
        }
        art->Dinv[i] = 1.0f / D;
        float u = art->tau[i] - b3_S_dot(S, art->pA[i]);
        B3Inertia IAp = b3_I_rank1(art->IA[i], art->U[i], art->Dinv[i]);
        B3Force pA = art->pA[i];
        B3Force Ic = b3_I_mul(IAp, art->c[i]);
        pA.n = b3_add(b3_add(pA.n, Ic.n), b3_mul(art->U[i].n, u * art->Dinv[i]));
        pA.f = b3_add(b3_add(pA.f, Ic.f), b3_mul(art->U[i].f, u * art->Dinv[i]));
        if (!art->fixed[p]) {
            art->IA[p] = b3_I_add(art->IA[p], b3_I_shift(IAp, art->r[i]));
            art->pA[p].n = b3_add(art->pA[p].n,
                b3_add(pA.n, b3_cross(art->r[i], pA.f)));
            art->pA[p].f = b3_add(art->pA[p].f, pA.f);
        }
    }
}

static B3_HD B3_INL void b3_art_forward(B3Art* art) {
    for (int i = 0; i < art->n_links; i++) {
        int p = art->parent[i];
        if (p < 0) {
            if (art->fixed[i]) {
                art->a[i] = b3_motion0();
            } else if (art->floating[i]) {
                B3Force rhs;
                rhs.n = b3_neg(art->pA[i].n);
                rhs.f = b3_neg(art->pA[i].f);
                if (!b3_I_solve(art->IA[i], rhs, &art->a[i])) {
                    art->a[i] = b3_motion0();
                }
            } else {
                art->a[i] = b3_motion0();
            }
            continue;
        }
        B3Motion ap;
        ap.w = art->a[p].w;
        ap.v = b3_add(art->a[p].v, b3_cross(art->a[p].w, art->r[i]));
        B3Motion rhs = ap;
        rhs.w = b3_add(rhs.w, art->c[i].w);
        rhs.v = b3_add(rhs.v, art->c[i].v);
        float u = art->tau[i] - b3_S_dot(art->S[i], art->pA[i]);
        u -= b3_S_dot(art->S[i], b3_I_mul(art->IA[i], rhs));
        art->qdd[i] = art->Dinv[i] * u;
        B3Motion Sq = b3_S_mul(art->S[i], art->qdd[i]);
        art->a[i].w = b3_add(rhs.w, Sq.w);
        art->a[i].v = b3_add(rhs.v, Sq.v);
    }
}

/* IA/U/Dinv depend on pose, mass and armature, but not on the contact RHS.
 * Build them with the original zero-bias backward pass and retain its exact
 * per-link arithmetic. Velocities may change during this scope; poses, mass,
 * topology and armature must remain unchanged. */
static B3_HD B3_INL void b3_art_linear_cache_begin(B3Art* art, const B3World* w) {
    assert(!art->linear_factor_scope);
    b3_art_refresh(art, w);
    for (int i = 0; i < art->n_links; i++) {
        art->v[i] = b3_motion0();
        art->qd[i] = 0.0f;
    }
    b3_art_seed_bias(art, w, 1);
    b3_art_backward(art);
    art->linear_factor_scope = 1;
#if B3_ART_CACHE_DIAGNOSTICS
    art->linear_factor_builds++;
#endif
}

static B3_HD B3_INL void b3_art_linear_cache_end(B3Art* art) {
    art->linear_factor_scope = 0;
}

/* Same force accumulation order as b3_art_backward. The scope guarantees
 * c=tau=v=0, and the previously factored IA/U/Dinv remain read-only here. */
static B3_HD B3_INL void b3_art_backward_linear_rhs(B3Art* art) {
    for (int i = art->n_links - 1; i >= 0; i--) {
        int p = art->parent[i];
        if (p < 0) continue;
        float u = art->tau[i] - b3_S_dot(art->S[i], art->pA[i]);
        B3Force pA = art->pA[i];
        B3Force Ic = b3_force0();
        pA.n = b3_add(b3_add(pA.n, Ic.n), b3_mul(art->U[i].n, u * art->Dinv[i]));
        pA.f = b3_add(b3_add(pA.f, Ic.f), b3_mul(art->U[i].f, u * art->Dinv[i]));
        if (!art->fixed[p]) {
            art->pA[p].n = b3_add(art->pA[p].n,
                b3_add(pA.n, b3_cross(art->r[i], pA.f)));
            art->pA[p].f = b3_add(art->pA[p].f, pA.f);
        }
    }
}

static B3_HD B3_INL void b3_art_aba(B3Art* art, const B3World* w) {
    b3_art_refresh(art, w);
    b3_art_seed_bias(art, w, 0);
    b3_art_backward(art);
    b3_art_forward(art);
}

static B3_HD B3_INL B3Force b3_art_row_wrench_ex(B3Vec3 r, B3Vec3 n, float x,
        int torque) {
    B3Force f;
    if (torque == 1) {
        f.n = b3_mul(n, x);
        f.f = b3_v(0.0f, 0.0f, 0.0f);
    } else {
        f.f = b3_mul(n, x);
        f.n = b3_cross(r, f.f);
    }
    return f;
}

static B3_HD B3_INL float b3_art_row_eval(const B3Art* art, const B3ArtRow* row,
        int accel) {
    float y = 0.0f;
    const B3Motion* m = accel ? art->a : art->v;
    if (row->torque == 1) {
        if (row->link_a >= 0) {
            y += b3_dot(m[row->link_a].w, row->n);
        }
        if (row->link_b >= 0) {
            y -= b3_dot(m[row->link_b].w, row->n);
        }
        return y;
    }
    if (row->link_a >= 0) {
        B3Vec3 ac = b3_add(m[row->link_a].v,
            b3_cross(m[row->link_a].w, row->ra));
        if (accel) {
            const B3Motion* v = &art->v[row->link_a];
            ac = b3_add(ac, b3_cross(v->w, b3_cross(v->w, row->ra)));
        }
        y += b3_dot(ac, row->n);
    }
    if (row->link_b >= 0) {
        B3Vec3 ac = b3_add(m[row->link_b].v,
            b3_cross(m[row->link_b].w, row->rb));
        if (accel) {
            const B3Motion* v = &art->v[row->link_b];
            ac = b3_add(ac, b3_cross(v->w, b3_cross(v->w, row->rb)));
        }
        y -= b3_dot(ac, row->n);
    }
    return y;
}

/* Δx = J M^{-1} J^T x. Bias / gravity / velocity products off. */
static B3_HD B3_INL void b3_art_delassus_apply(B3Art* art, const B3World* w,
        const B3ArtRow* rows, const float* x, float* y, int n_rows) {
    int cached = art->linear_factor_scope;
    if (cached) {
        for (int i = 0; i < art->n_links; i++) art->pA[i] = b3_force0();
    } else {
    b3_art_refresh(art, w);
    for (int i = 0; i < art->n_links; i++) {
        art->v[i] = b3_motion0();
        art->qd[i] = 0.0f;
    }
    b3_art_seed_bias(art, w, 1);
#if B3_ART_CACHE_DIAGNOSTICS
    art->linear_factor_builds++;
#endif
    }
    for (int e = 0; e < n_rows; e++) {
        const B3ArtRow* row = &rows[e];
        if (row->link_a >= 0 && !art->fixed[row->link_a]) {
            B3Force f = b3_art_row_wrench_ex(row->ra, row->n, x[e], row->torque);
            art->pA[row->link_a].n = b3_sub(art->pA[row->link_a].n, f.n);
            art->pA[row->link_a].f = b3_sub(art->pA[row->link_a].f, f.f);
        }
        if (row->link_b >= 0 && !art->fixed[row->link_b]) {
            B3Force f = b3_art_row_wrench_ex(row->rb, row->n, x[e], row->torque);
            art->pA[row->link_b].n = b3_add(art->pA[row->link_b].n, f.n);
            art->pA[row->link_b].f = b3_add(art->pA[row->link_b].f, f.f);
        }
    }
    if (cached) b3_art_backward_linear_rhs(art);
    else b3_art_backward(art);
    b3_art_forward(art);
#if B3_ART_CACHE_DIAGNOSTICS
    art->linear_rhs_calls++;
#endif
    for (int e = 0; e < n_rows; e++) {
        y[e] = b3_art_row_eval(art, &rows[e], 1);
    }
}

static B3_HD B3_INL int b3_art_link_of(const B3Art* art, int body) {
    for (int i = 0; i < art->n_links; i++) {
        if (art->body[i] == body) {
            return i;
        }
    }
    return -1;
}

/* Independent-body contact mass (Soft Step). Wrong for a jointed tree. */
static B3_HD B3_INL float b3_art_indep_w(const B3Body* b, B3Vec3 r, B3Vec3 n) {
    if (b->type != B3_DYNAMIC || b->inv_mass <= 0.0f) {
        return 0.0f;
    }
    B3Vec3 rn = b3_cross(r, n);
    return b->inv_mass + b3_dot(rn, b3_mv(b->inv_i_world, rn));
}

static B3_HD B3_INL int b3_art_make_row(const B3Art* art, int body_a, int body_b,
        B3Vec3 ra, B3Vec3 rb, B3Vec3 n, B3ArtRow* row) {
    row->link_a = body_a >= 0 ? b3_art_link_of(art, body_a) : -1;
    row->link_b = body_b >= 0 ? b3_art_link_of(art, body_b) : -1;
    row->torque = 0;
    row->ra = ra;
    row->rb = rb;
    row->n = n;
    return row->link_a >= 0 || row->link_b >= 0;
}

/* Δ = n · J M^{-1} J^T n. Isolated free bodies match b3_art_indep_w. */
static B3_HD B3_INL float b3_art_response_w(B3Art* art, const B3World* w,
        const B3ArtRow* row) {
    float x = 1.0f;
    float y = 0.0f;
    b3_art_delassus_apply(art, w, row, &x, &y, 1);
    return y;
}

static B3_HD B3_INL void b3_art_add_delta_vel(const B3Art* art, B3World* w) {
    for (int i = 0; i < art->n_links; i++) {
        if (art->fixed[i]) {
            continue;
        }
        B3Body* b = &w->bodies[art->body[i]];
        b->lin_vel = b3_add(b->lin_vel, art->a[i].v);
        b->ang_vel = b3_add(b->ang_vel, art->a[i].w);
    }
}

/* Soft Step contact convention: λ > 0 pushes B along n and A against n. */
static B3_HD B3_INL void b3_art_apply_impulse(B3Art* art, B3World* w,
        const B3ArtRow* row, float lambda) {
    if (lambda == 0.0f) return;
    float x = -lambda;
    float y = 0.0f;
    b3_art_delassus_apply(art, w, row, &x, &y, 1);
    b3_art_add_delta_vel(art, w);
}

/* Torque-only rolling in the tangent plane. Twist stays on the normal row. */
static B3_HD B3_INL void b3_art_solve_rolling_fields(B3Art* art, B3World* w,
        int body_a, int body_b, B3Vec3 t1, B3Vec3 t2, float rolling,
        float total_n, B3Vec3* rolling_impulse) {
    float max_r = rolling * total_n;
    if (!(max_r > 0.0f)) {
        return;
    }
    B3ArtRow rows[2];
    if (!b3_art_make_row(art, body_a, body_b,
            b3_v(0.0f, 0.0f, 0.0f), b3_v(0.0f, 0.0f, 0.0f), t1, &rows[0])) {
        return;
    }
    rows[0].torque = 1;
    b3_art_make_row(art, body_a, body_b,
        b3_v(0.0f, 0.0f, 0.0f), b3_v(0.0f, 0.0f, 0.0f), t2, &rows[1]);
    rows[1].torque = 1;
    float e0[2] = {1.0f, 0.0f};
    float e1[2] = {0.0f, 1.0f};
    float col0[2], col1[2];
    b3_art_delassus_apply(art, w, rows, e0, col0, 2);
    b3_art_delassus_apply(art, w, rows, e1, col1, 2);
    B3Mat2 K;
    K.cx.x = col0[0];
    K.cx.y = col0[1];
    K.cy.x = col1[0];
    K.cy.y = col1[1];
    B3Mat2 Minv = b3_invert2(K);
    B3Body* ba = &w->bodies[body_a];
    B3Body* bb = &w->bodies[body_b];
    B3Vec3 dw = b3_sub(bb->ang_vel, ba->ang_vel);
    B3Vec2 vt;
    vt.x = b3_dot(dw, t1);
    vt.y = b3_dot(dw, t2);
    B3Vec2 tm = b3_mv2(Minv, vt);
    B3Vec3 old = *rolling_impulse;
    B3Vec2 oi;
    oi.x = b3_dot(old, t1);
    oi.y = b3_dot(old, t2);
    B3Vec2 ni;
    ni.x = oi.x - tm.x;
    ni.y = oi.y - tm.y;
    float fl2 = ni.x * ni.x + ni.y * ni.y;
    if (fl2 > max_r * max_r && fl2 > 0.0f) {
        float sc = b3_rsqrt_scale(max_r, fl2);
        ni.x *= sc;
        ni.y *= sc;
    }
    B3Vec2 df;
    df.x = ni.x - oi.x;
    df.y = ni.y - oi.y;
    *rolling_impulse = b3_add(b3_mul(t1, ni.x), b3_mul(t2, ni.y));
    if (df.x != 0.0f) {
        b3_art_apply_impulse(art, w, &rows[0], df.x);
    }
    if (df.y != 0.0f) {
        b3_art_apply_impulse(art, w, &rows[1], df.y);
    }
}

static B3_HD B3_INL int b3_solve_n(int n, const float* A, const float* b,
        float* x) {
    if (n < 1) {
        return 1;
    }
    if (n > B3_ART_MAX_ROWS) {
        n = B3_ART_MAX_ROWS;
    }
    float M[B3_ART_MAX_ROWS][B3_ART_MAX_ROWS + 1];
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) {
            M[r][c] = A[r * n + c];
        }
        M[r][n] = b[r];
    }
    for (int k = 0; k < n; k++) {
        int piv = k;
        float best = fabsf(M[k][k]);
        for (int r = k + 1; r < n; r++) {
            float v = fabsf(M[r][k]);
            if (v > best) {
                best = v;
                piv = r;
            }
        }
        if (best < 1.0e-12f) {
            return 0;
        }
        if (piv != k) {
            for (int c = k; c <= n; c++) {
                float tmp = M[k][c];
                M[k][c] = M[piv][c];
                M[piv][c] = tmp;
            }
        }
        float inv = 1.0f / M[k][k];
        for (int c = k; c <= n; c++) {
            M[k][c] *= inv;
        }
        for (int r = 0; r < n; r++) {
            if (r == k) {
                continue;
            }
            float s = M[r][k];
            for (int c = k; c <= n; c++) {
                M[r][c] -= s * M[k][c];
            }
        }
    }
    for (int r = 0; r < n; r++) {
        x[r] = M[r][n];
    }
    return 1;
}

/* Δ = J M^{-1} J^T, including two-body and row-row fill-in. */
static B3_HD B3_INL void b3_art_delassus_matrix(B3Art* art, const B3World* w,
        const B3ArtRow* rows, float* D, int n) {
    float x[B3_ART_MAX_ROWS];
    float y[B3_ART_MAX_ROWS];
    for (int i = 0; i < n; i++) {
        x[i] = 0.0f;
    }
    for (int c = 0; c < n; c++) {
        x[c] = 1.0f;
        b3_art_delassus_apply(art, w, rows, x, y, n);
        for (int r = 0; r < n; r++) {
            D[r * n + c] = y[r];
        }
        x[c] = 0.0f;
    }
}

static B3_HD B3_INL void b3_art_apply_impulses(B3Art* art, B3World* w,
        const B3ArtRow* rows, const float* lambda, int n) {
    float x[B3_ART_MAX_ROWS];
    float y[B3_ART_MAX_ROWS];
    int any = 0;
    for (int e = 0; e < n; e++) {
        x[e] = -lambda[e];
        if (lambda[e] != 0.0f) {
            any = 1;
        }
    }
    if (!any) {
        return;
    }
    b3_art_delassus_apply(art, w, rows, x, y, n);
    b3_art_add_delta_vel(art, w);
}

/* λ = (μ^{-1} + Δ)^{-1} y. Δ is J M^{-1} J^T from the two-sweep. */
static B3_HD B3_INL void b3_art_damped_solve(B3Art* art, const B3World* w,
        const B3ArtRow* rows, const float* mu, const float* y, float* lambda,
        int n_rows) {
    int n = n_rows < B3_ART_MAX_ROWS ? n_rows : B3_ART_MAX_ROWS;
    if (n < 1) {
        return;
    }
    float D[B3_ART_MAX_ROWS * B3_ART_MAX_ROWS];
    b3_art_delassus_matrix(art, w, rows, D, n);
    for (int e = 0; e < n; e++) {
        float r = mu[e] > 1.0e-12f ? (1.0f / mu[e]) : 1.0e12f;
        D[e * n + e] += r;
    }
    if (!b3_solve_n(n, D, y, lambda)) {
        for (int e = 0; e < n; e++) {
            lambda[e] = 0.0f;
        }
    }
}

static B3_HD B3_INL void b3_art_write_delta(B3Art* art, B3World* w, int i) {
    art->linear_factor_scope = 0;
    if (art->fixed[i]) {
        return;
    }
    B3Body* b = &w->bodies[art->body[i]];
    b->delta_pos = b3_sub(art->com[i], b->center);
    b->delta_rot = b3_qnorm(b3_qmul(art->rot[i], b3_qconj(b->rotation)));
}

static B3_HD B3_INL void b3_art_integrate_vel(B3Art* art, B3World* w, float h) {
    b3_art_aba(art, w);
    for (int i = 0; i < art->n_links; i++) {
        if (art->fixed[i]) {
            continue;
        }
        B3Body* b = &w->bodies[art->body[i]];
        b->lin_vel = b3_madd(b->lin_vel, h, art->a[i].v);
        b->ang_vel = b3_madd(b->ang_vel, h, art->a[i].w);
    }
}

static B3_HD B3_INL void b3_art_integrate_pos(B3Art* art, B3World* w,
        float h, float inv_dt) {
    float max_lin = w->max_linear_speed;
    float max_ang = B3_MAX_ROTATION * inv_dt;
    float max_lin2 = max_lin * max_lin;
    float max_ang2 = max_ang * max_ang;
    b3_art_refresh(art, w);
    for (int i = 0; i < art->n_links; i++) {
        int p = art->parent[i];
        if (p < 0) {
            if (art->fixed[i]) {
                continue;
            }
            B3Body* b = &w->bodies[art->body[i]];
            b3_integrate_position_state(b, h, max_lin, max_ang, max_lin2,
                max_ang2, &b->lin_vel, &b->ang_vel, &b->delta_pos,
                &b->delta_rot);
            b3_art_lived_pose(b, art->local_center[i],
                &art->pos[i], &art->rot[i], &art->com[i]);
            continue;
        }
        float qd = art->qd[i];
        if (qd * qd > max_ang2 && max_ang2 > 0.0f) {
            qd = copysignf(max_ang, qd);
            art->qd[i] = qd;
        }
        art->q[i] += h * qd;
        b3_art_fk_link(art, i);
        b3_art_write_delta(art, w, i);
    }
}

static B3_HD B3_INL void b3_art_solve_friction(B3Art* art, B3World* w,
        B3Contact* c) {
    float total_n = 0.0f;
    float twist_lim = 0.0f;
    for (int p = 0; p < c->point_count; p++) {
        total_n += c->points[p].normal_impulse;
        twist_lim += c->points[p].lever * c->points[p].normal_impulse;
    }
    float max_t = B3_ART_TORSION ? c->friction * twist_lim : 0.0f;
    if (max_t > 0.0f) {
        B3ArtRow tw;
        if (b3_art_make_row(art, c->body_a, c->body_b,
                b3_v(0.0f, 0.0f, 0.0f), b3_v(0.0f, 0.0f, 0.0f),
                c->normal, &tw)) {
            tw.torque = 1;
            float ww = b3_art_response_w(art, w, &tw);
            c->twist_mass = ww > 0.0f ? 1.0f / ww : 0.0f;
            B3Body* ba = &w->bodies[c->body_a];
            B3Body* bb = &w->bodies[c->body_b];
            float twist_s = b3_dot(c->normal,
                b3_sub(bb->ang_vel, ba->ang_vel));
            float dtw = -c->twist_mass * twist_s;
            float old_t = c->twist_impulse;
            float nt = old_t + dtw;
            if (nt > max_t) {
                nt = max_t;
            } else if (nt < -max_t) {
                nt = -max_t;
            }
            dtw = nt - old_t;
            c->twist_impulse = nt;
            if (dtw != 0.0f) {
                b3_art_apply_impulse(art, w, &tw, dtw);
            }
        }
    }
    float max_f = c->friction * total_n;
    if (max_f > 0.0f) {
    B3ArtRow rows[2];
    if (b3_art_make_row(art, c->body_a, c->body_b, c->center_a, c->center_b,
            c->tangent1, &rows[0])) {
    b3_art_make_row(art, c->body_a, c->body_b, c->center_a, c->center_b,
        c->tangent2, &rows[1]);
    float e0[2] = {1.0f, 0.0f};
    float e1[2] = {0.0f, 1.0f};
    float col0[2], col1[2];
    b3_art_delassus_apply(art, w, rows, e0, col0, 2);
    b3_art_delassus_apply(art, w, rows, e1, col1, 2);
    B3Mat2 K;
    K.cx.x = col0[0];
    K.cx.y = col0[1];
    K.cy.x = col1[0];
    K.cy.y = col1[1];
    B3Mat2 Minv = b3_invert2(K);
    c->tangent_mass = Minv;
    B3Body* ba = &w->bodies[c->body_a];
    B3Body* bb = &w->bodies[c->body_b];
    B3Vec3 vra = b3_add(ba->lin_vel, b3_cross(ba->ang_vel, c->center_a));
    B3Vec3 vrb = b3_add(bb->lin_vel, b3_cross(bb->ang_vel, c->center_b));
    B3Vec3 vr = b3_sub(vrb, vra);
    B3Vec2 vt;
    vt.x = b3_dot(vr, c->tangent1);
    vt.y = b3_dot(vr, c->tangent2);
    B3Vec2 tm = b3_mv2(Minv, vt);
    B3Vec2 ni;
    ni.x = c->friction_impulse.x - tm.x;
    ni.y = c->friction_impulse.y - tm.y;
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
    if (df.x != 0.0f) {
        b3_art_apply_impulse(art, w, &rows[0], df.x);
    }
    if (df.y != 0.0f) {
        b3_art_apply_impulse(art, w, &rows[1], df.y);
    }
        }
    }
    b3_art_solve_rolling_fields(art, w, c->body_a, c->body_b,
        c->tangent1, c->tangent2, c->rolling, total_n, &c->rolling_impulse);
}

static B3_HD B3_INL int b3_art_cut_rows(B3Art* art, const B3World* w, int ci,
        B3ArtRow rows[5], float err[5]) {
    const B3Joint* j = &w->joints[art->cut_joint[ci]];
    int la = b3_art_link_of(art, j->body_a);
    int lb = b3_art_link_of(art, j->body_b);
    if (la < 0 || lb < 0) {
        return 0;
    }
    B3Vec3 pa = b3_xf_point(art->pos[la], art->rot[la], j->local_anchor_a);
    B3Vec3 pb = b3_xf_point(art->pos[lb], art->rot[lb], j->local_anchor_b);
    B3Vec3 ra = b3_sub(pa, art->com[la]);
    B3Vec3 rb = b3_sub(pb, art->com[lb]);
    B3Vec3 d = b3_sub(pb, pa);
    B3Vec3 ax[3] = {
        b3_v(1.0f, 0.0f, 0.0f),
        b3_v(0.0f, 1.0f, 0.0f),
        b3_v(0.0f, 0.0f, 1.0f)
    };
    for (int k = 0; k < 3; k++) {
        rows[k].link_a = la;
        rows[k].link_b = lb;
        rows[k].torque = 0;
        rows[k].ra = ra;
        rows[k].rb = rb;
        rows[k].n = ax[k];
        err[k] = b3_dot(d, ax[k]);
    }
    B3Quat qa = b3_qmul(art->rot[la], j->local_rot_a);
    B3Quat qb = b3_qmul(art->rot[lb], j->local_rot_b);
    if (b3_qdot(qa, qb) < 0.0f) {
        qb = b3_qneg(qb);
    }
    B3Quat rel = b3_qinv_mul(qa, qb);
    B3Vec3 px = b3_rotate(qa, b3_v(1.0f, 0.0f, 0.0f));
    B3Vec3 py = b3_rotate(qa, b3_v(0.0f, 1.0f, 0.0f));
    rows[3].link_a = la;
    rows[3].link_b = lb;
    rows[3].torque = 1;
    rows[3].ra = b3_v(0.0f, 0.0f, 0.0f);
    rows[3].rb = b3_v(0.0f, 0.0f, 0.0f);
    rows[3].n = px;
    err[3] = 2.0f * rel.v.x;
    rows[4].link_a = la;
    rows[4].link_b = lb;
    rows[4].torque = 1;
    rows[4].ra = b3_v(0.0f, 0.0f, 0.0f);
    rows[4].rb = b3_v(0.0f, 0.0f, 0.0f);
    rows[4].n = py;
    err[4] = 2.0f * rel.v.y;
    return 5;
}

/* Cut revolute = 3 linear + 2 angular rows. One 5×5 Delassus block per cut.
 * PGS is only the fallback if Δ is singular. */
static B3_HD B3_INL void b3_art_solve_cuts(B3Art* art, B3World* w,
        float inv_h, int use_bias, int iters) {
    if (art->n_cuts < 1) {
        return;
    }
    if (iters < 1) {
        iters = 1;
    }
    b3_art_refresh(art, w);
    for (int it = 0; it < iters; it++) {
        for (int c = 0; c < art->n_cuts; c++) {
            B3ArtRow rows[5];
            float err[5];
            if (b3_art_cut_rows(art, w, c, rows, err) != 5) {
                continue;
            }
            b3_art_refresh(art, w);
            float rhs[5];
            for (int k = 0; k < 5; k++) {
                float vn = b3_art_row_eval(art, &rows[k], 0);
                float vbias = 0.0f;
                if (use_bias) {
                    vbias = 0.2f * inv_h * err[k];
                }
                rhs[k] = vn - vbias;
            }
            float D[25];
            b3_art_delassus_matrix(art, w, rows, D, 5);
            for (int k = 0; k < 5; k++) {
                D[k * 5 + k] += 1.0e-8f;
            }
            float lam[5];
            if (!b3_solve_n(5, D, rhs, lam)) {
                for (int k = 0; k < 5; k++) {
                    float ww = D[k * 5 + k];
                    lam[k] = ww > 1.0e-10f ? rhs[k] / ww : 0.0f;
                }
            }
            b3_art_apply_impulses(art, w, rows, lam, 5);
        }
    }
}

static B3_HD B3_INL void b3_art_solve_contacts(B3Art* art, B3World* w,
        float inv_h, float contact_speed, int use_bias, int iters) {
#if B3_ART_FIXED_POSE_CACHE
    assert(!art->linear_factor_scope);
    if (w->contact_count == 0) return;
    b3_art_linear_cache_begin(art, w);
#endif
    if (iters < 1) {
        iters = 1;
    }
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < w->contact_count; i++) {
            B3Contact* c = &w->contacts[i];
            B3Body* ba = &w->bodies[c->body_a];
            B3Body* bb = &w->bodies[c->body_b];
            B3Vec3 n = c->normal;
            B3Vec3 dp = b3_sub(bb->delta_pos, ba->delta_pos);
            B3Quat dqa = ba->delta_rot;
            B3Quat dqb = bb->delta_rot;
            float total_n = 0.0f;
            for (int p = 0; p < c->point_count; p++) {
                B3Point* cp = &c->points[p];
                B3ArtRow row;
                if (!b3_art_make_row(art, c->body_a, c->body_b,
                        cp->r_a, cp->r_b, n, &row)) {
                    continue;
                }
                if (it == 0) {
                    float ww = b3_art_response_w(art, w, &row);
                    cp->normal_mass = ww > 0.0f ? 1.0f / ww : 0.0f;
                }
                B3Vec3 ds = b3_add(dp, b3_sub(b3_rotate(dqb, cp->r_b),
                    b3_rotate(dqa, cp->r_a)));
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
                B3Vec3 vra = b3_add(ba->lin_vel, b3_cross(ba->ang_vel, cp->r_a));
                B3Vec3 vrb = b3_add(bb->lin_vel, b3_cross(bb->ang_vel, cp->r_b));
                float vn = b3_dot(b3_sub(vrb, vra), n);
                float dimp = -cp->normal_mass * (mscale * vn + vbias)
                    - iscale * cp->normal_impulse;
                float nimp = b3_maxf(cp->normal_impulse + dimp, 0.0f);
                dimp = nimp - cp->normal_impulse;
                cp->normal_impulse = nimp;
                cp->total_normal += dimp;
                total_n += nimp;
                if (dimp != 0.0f) {
                    b3_art_apply_impulse(art, w, &row, dimp);
                }
            }
            if (!use_bias) {
                b3_art_solve_friction(art, w, c);
            }
        }
    }
#if B3_ART_FIXED_POSE_CACHE
    b3_art_linear_cache_end(art);
#endif
}

#ifndef B3_ART_CONTACT_ITERS
#define B3_ART_CONTACT_ITERS 4
#endif

#ifdef B3_PACKED_GS
static B3_HD B3_INL void b3_art_pull_gs(B3World* w, const B3GsBody* bl) {
    for (int i = 0; i < w->body_count; i++) {
        w->bodies[i].lin_vel = bl[i].lin_vel;
        w->bodies[i].ang_vel = bl[i].ang_vel;
        w->bodies[i].delta_pos = bl[i].delta_pos;
        w->bodies[i].delta_rot = bl[i].delta_rot;
    }
}

static B3_HD B3_INL void b3_art_push_gs(const B3World* w, B3GsBody* bl) {
    for (int i = 0; i < w->body_count; i++) {
        if ((bl[i].flags & B3_FLAG_DYNAMIC) == 0) {
            continue;
        }
        bl[i].lin_vel = w->bodies[i].lin_vel;
        bl[i].ang_vel = w->bodies[i].ang_vel;
    }
}

static B3_HD B3_INL void b3_art_solve_friction_gs(B3Art* art, B3World* w,
        B3GsContact* c) {
    float total_n = 0.0f;
    float twist_lim = 0.0f;
    for (int p = 0; p < c->point_count; p++) {
        total_n += c->points[p].normal_impulse;
        twist_lim += c->points[p].lever * c->points[p].normal_impulse;
    }
    float max_t = B3_ART_TORSION ? c->friction * twist_lim : 0.0f;
    if (max_t > 0.0f) {
        B3ArtRow tw;
        if (b3_art_make_row(art, c->body_a, c->body_b,
                b3_v(0.0f, 0.0f, 0.0f), b3_v(0.0f, 0.0f, 0.0f),
                c->normal, &tw)) {
            tw.torque = 1;
            float ww = b3_art_response_w(art, w, &tw);
            c->twist_mass = ww > 0.0f ? 1.0f / ww : 0.0f;
            B3Body* ba = &w->bodies[c->body_a];
            B3Body* bb = &w->bodies[c->body_b];
            float twist_s = b3_dot(c->normal,
                b3_sub(bb->ang_vel, ba->ang_vel));
            float dtw = -c->twist_mass * twist_s;
            float old_t = c->twist_impulse;
            float nt = old_t + dtw;
            if (nt > max_t) {
                nt = max_t;
            } else if (nt < -max_t) {
                nt = -max_t;
            }
            dtw = nt - old_t;
            c->twist_impulse = nt;
            if (dtw != 0.0f) {
                b3_art_apply_impulse(art, w, &tw, dtw);
            }
        }
    }
    float max_f = c->friction * total_n;
    if (max_f > 0.0f) {
    B3ArtRow rows[2];
    if (b3_art_make_row(art, c->body_a, c->body_b, c->center_a, c->center_b,
            c->tangent1, &rows[0])) {
    b3_art_make_row(art, c->body_a, c->body_b, c->center_a, c->center_b,
        c->tangent2, &rows[1]);
    float e0[2] = {1.0f, 0.0f};
    float e1[2] = {0.0f, 1.0f};
    float col0[2], col1[2];
    b3_art_delassus_apply(art, w, rows, e0, col0, 2);
    b3_art_delassus_apply(art, w, rows, e1, col1, 2);
    B3Mat2 K;
    K.cx.x = col0[0];
    K.cx.y = col0[1];
    K.cy.x = col1[0];
    K.cy.y = col1[1];
    B3Mat2 Minv = b3_invert2(K);
    c->tangent_mass = Minv;
    B3Body* ba = &w->bodies[c->body_a];
    B3Body* bb = &w->bodies[c->body_b];
    B3Vec3 vra = b3_add(ba->lin_vel, b3_cross(ba->ang_vel, c->center_a));
    B3Vec3 vrb = b3_add(bb->lin_vel, b3_cross(bb->ang_vel, c->center_b));
    B3Vec3 vr = b3_sub(vrb, vra);
    B3Vec2 vt;
    vt.x = b3_dot(vr, c->tangent1);
    vt.y = b3_dot(vr, c->tangent2);
    B3Vec2 tm = b3_mv2(Minv, vt);
    B3Vec2 ni;
    ni.x = c->friction_impulse.x - tm.x;
    ni.y = c->friction_impulse.y - tm.y;
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
    if (df.x != 0.0f) {
        b3_art_apply_impulse(art, w, &rows[0], df.x);
    }
    if (df.y != 0.0f) {
        b3_art_apply_impulse(art, w, &rows[1], df.y);
    }
    }
    }
    b3_art_solve_rolling_fields(art, w, c->body_a, c->body_b,
        c->tangent1, c->tangent2, c->rolling, total_n, &c->rolling_impulse);
}

static B3_HD B3_INL void b3_art_solve_contacts_gs(B3Art* art, B3World* w,
        B3GsContact* contacts, int n, B3GsBody* bodies, float inv_h,
        float contact_speed, int use_bias, int iters) {
    if (iters < 1) {
        iters = 1;
    }
    b3_art_pull_gs(w, bodies);
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < n; i++) {
            B3GsContact* c = &contacts[i];
            B3Body* ba = &w->bodies[c->body_a];
            B3Body* bb = &w->bodies[c->body_b];
            B3Vec3 nrm = c->normal;
            B3Vec3 dp = b3_sub(bb->delta_pos, ba->delta_pos);
            B3Quat dqa = ba->delta_rot;
            B3Quat dqb = bb->delta_rot;
            for (int p = 0; p < c->point_count; p++) {
                B3GsPoint* cp = &c->points[p];
                B3ArtRow row;
                if (!b3_art_make_row(art, c->body_a, c->body_b,
                        cp->r_a, cp->r_b, nrm, &row)) {
                    continue;
                }
                if (it == 0) {
                    float ww = b3_art_response_w(art, w, &row);
                    cp->normal_mass = ww > 0.0f ? 1.0f / ww : 0.0f;
                }
                B3Vec3 ds = b3_add(dp, b3_sub(b3_rotate(dqb, cp->r_b),
                    b3_rotate(dqa, cp->r_a)));
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
                B3Vec3 vra = b3_add(ba->lin_vel, b3_cross(ba->ang_vel, cp->r_a));
                B3Vec3 vrb = b3_add(bb->lin_vel, b3_cross(bb->ang_vel, cp->r_b));
                float vn = b3_dot(b3_sub(vrb, vra), nrm);
                float dimp = -cp->normal_mass * (mscale * vn + vbias)
                    - iscale * cp->normal_impulse;
                float nimp = b3_maxf(cp->normal_impulse + dimp, 0.0f);
                dimp = nimp - cp->normal_impulse;
                cp->normal_impulse = nimp;
                cp->total_normal += dimp;
                if (dimp != 0.0f) {
                    b3_art_apply_impulse(art, w, &row, dimp);
                }
            }
            if (!use_bias) {
                b3_art_solve_friction_gs(art, w, c);
            }
        }
    }
    b3_art_push_gs(w, bodies);
}
#endif
