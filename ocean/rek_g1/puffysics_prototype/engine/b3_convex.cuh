// SPDX-License-Identifier: MIT
// Analytic support-mapped finite cylinders. Experimental single-point contacts.
// Included after B3Mani helpers. No allocations, CPU callbacks, or proxy shapes.
#pragma once

#ifndef B3_CONVEX_GJK_ITERATIONS
#define B3_CONVEX_GJK_ITERATIONS 64
#endif
#ifndef B3_CONVEX_EPA_ITERATIONS
#define B3_CONVEX_EPA_ITERATIONS 96
#endif
#ifndef B3_CONVEX_VERTICES
#define B3_CONVEX_VERTICES 100
#endif
#ifndef B3_CONVEX_FACES
#define B3_CONVEX_FACES 200
#endif

typedef struct B3Convex {
    const B3Shape* shape;
    B3Vec3 position;
    B3Quat rotation;
} B3Convex;

typedef struct B3SupportVertex {
    B3Vec3 v, a, b;
} B3SupportVertex;

typedef struct B3ConvexFace {
    int a, b, c;
    B3Vec3 n;
    float d;
} B3ConvexFace;

B3_HD B3_INL double b3_convex_dot64(B3Vec3 a, B3Vec3 b) {
    return double(a.x) * b.x + double(a.y) * b.y + double(a.z) * b.z;
}

/* A cylinder's disk support is radial r*d/|d|; its caps are +/- half.y.
 * Choosing the disk center for an exactly axial direction is also exact. */
B3_HD B3_INL B3Vec3 b3_convex_support(const B3Convex* c, B3Vec3 direction) {
    const B3Shape* s = c->shape;
    B3Vec3 d = b3_inv_rotate(c->rotation, direction);
    B3Vec3 p;
    if (s->type == B3_BOX) {
        p = b3_v(d.x >= 0.0f ? s->half.x : -s->half.x,
            d.y >= 0.0f ? s->half.y : -s->half.y,
            d.z >= 0.0f ? s->half.z : -s->half.z);
    } else if (s->type == B3_CYLINDER) {
        float radial = sqrtf(d.x * d.x + d.z * d.z);
        float k = radial > 0.0f ? s->radius / radial : 0.0f;
        p = b3_v(k * d.x, d.y >= 0.0f ? s->half.y : -s->half.y, k * d.z);
    } else {
        float length = b3_len(d);
        p = length > 0.0f ? b3_mul(d, s->radius / length) : b3_v(s->radius, 0, 0);
        if (s->type == B3_CAPSULE)
            p.y += d.y >= 0.0f ? s->half.y : -s->half.y;
    }
    return b3_xf_point(c->position, c->rotation, p);
}

B3_HD B3_INL B3SupportVertex b3_convex_difference(const B3Convex* a,
        const B3Convex* b, B3Vec3 direction) {
    B3SupportVertex p;
    p.a = b3_convex_support(a, direction);
    p.b = b3_convex_support(b, b3_neg(direction));
    p.v = b3_sub(p.a, p.b);
    return p;
}

/* Barycentric origin coordinates in a nondegenerate tetrahedron. */
B3_HD B3_INL int b3_convex_tetra(const B3SupportVertex* p, float* w) {
    B3Vec3 a = b3_sub(p[0].v, p[3].v);
    B3Vec3 b = b3_sub(p[1].v, p[3].v);
    B3Vec3 c = b3_sub(p[2].v, p[3].v);
    B3Vec3 rhs = b3_neg(p[3].v);
    float det = b3_dot(a, b3_cross(b, c));
    float scale = b3_len(a) * b3_len(b) * b3_len(c);
    if (fabsf(det) <= 1.0e-7f * scale || scale == 0.0f) return 0;
    w[0] = b3_dot(rhs, b3_cross(b, c)) / det;
    w[1] = b3_dot(a, b3_cross(rhs, c)) / det;
    w[2] = b3_dot(a, b3_cross(b, rhs)) / det;
    w[3] = 1.0f - w[0] - w[1] - w[2];
    return w[0] >= 0.0f && w[1] >= 0.0f && w[2] >= 0.0f && w[3] >= 0.0f;
}

B3_HD B3_INL void b3_convex_candidate(const B3SupportVertex* p, int n,
        const double* candidate, double* best, double* distance) {
    double x = 0, y = 0, z = 0;
    for (int i = 0; i < n; i++) {
        x += candidate[i] * p[i].v.x;
        y += candidate[i] * p[i].v.y;
        z += candidate[i] * p[i].v.z;
    }
    double d = x*x + y*y + z*z;
    if (d < *distance) {
        *distance = d;
        for (int i = 0; i < n; i++) best[i] = candidate[i];
    }
}

/* Enumerate the simplex's Voronoi candidates, then remove zero-weight points.
 * At most four vertices, six edges, and four triangles are evaluated. */
B3_HD B3_INL B3Vec3 b3_convex_closest(B3SupportVertex* p, int* count,
        float* weights, B3Vec3* witness_a, B3Vec3* witness_b) {
    int n = *count;
    double best[4] = {0, 0, 0, 0};
    float tetra_weights[4];
    double d = DBL_MAX;
    if (n == 4 && b3_convex_tetra(p, tetra_weights)) {
        for (int i = 0; i < 4; i++) best[i] = tetra_weights[i];
        d = 0.0f;
    } else {
        for (int i = 0; i < n; i++) {
            double w[4] = {0, 0, 0, 0}; w[i] = 1.0;
            b3_convex_candidate(p, n, w, best, &d);
        }
        for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) {
            // Subtract after promotion: float edge subtraction leaves a
            // tangential residual that is amplified by near-contact normals.
            double ex = double(p[j].v.x) - p[i].v.x;
            double ey = double(p[j].v.y) - p[i].v.y;
            double ez = double(p[j].v.z) - p[i].v.z;
            double den = ex*ex + ey*ey + ez*ez;
            if (den == 0.0f) continue;
            double t = -(p[i].v.x*ex + p[i].v.y*ey + p[i].v.z*ez) / den;
            t = t < 0 ? 0 : (t > 1 ? 1 : t);
            double w[4] = {0, 0, 0, 0}; w[i] = 1.0 - t; w[j] = t;
            b3_convex_candidate(p, n, w, best, &d);
        }
        for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++)
        for (int k = j + 1; k < n; k++) {
            double ex = double(p[j].v.x) - p[i].v.x, fx = double(p[k].v.x) - p[i].v.x;
            double ey = double(p[j].v.y) - p[i].v.y, fy = double(p[k].v.y) - p[i].v.y;
            double ez = double(p[j].v.z) - p[i].v.z, fz = double(p[k].v.z) - p[i].v.z;
            double ee = ex*ex + ey*ey + ez*ez, ef = ex*fx + ey*fy + ez*fz, ff = fx*fx + fy*fy + fz*fz;
            double den = ee * ff - ef * ef;
            if (den <= 1.0e-14 * ee * ff) continue;
            double r = -(p[i].v.x*ex + p[i].v.y*ey + p[i].v.z*ez);
            double s = -(p[i].v.x*fx + p[i].v.y*fy + p[i].v.z*fz);
            double u = (r * ff - s * ef) / den, v = (s * ee - r * ef) / den;
            if (u < 0 || v < 0 || u + v > 1.0f) continue;
            double w[4] = {0, 0, 0, 0}; w[i] = 1.0 - u - v; w[j] = u; w[k] = v;
            b3_convex_candidate(p, n, w, best, &d);
        }
    }
    *witness_a = b3_v(0, 0, 0); *witness_b = b3_v(0, 0, 0);
    double cx = 0, cy = 0, cz = 0;
    int kept = 0;
    for (int i = 0; i < n; i++) if (best[i] > 0.0f) {
        *witness_a = b3_madd(*witness_a, best[i], p[i].a);
        *witness_b = b3_madd(*witness_b, best[i], p[i].b);
        cx += best[i] * p[i].v.x; cy += best[i] * p[i].v.y; cz += best[i] * p[i].v.z;
        p[kept] = p[i]; weights[kept++] = best[i];
    }
    *count = kept;
    return b3_v(float(cx),float(cy),float(cz));
}

B3_HD B3_INL int b3_convex_face(B3ConvexFace* f, const B3SupportVertex* p,
        int a, int b, int c) {
    B3Vec3 n = b3_cross(b3_sub(p[b].v, p[a].v), b3_sub(p[c].v, p[a].v));
    float length = b3_len(n);
    if (length <= 1.0e-14f) return 0;
    n = b3_mul(n, 1.0f / length);
    float d = b3_dot(n, p[a].v);
    if (d < 0.0f) { int t = b; b = c; c = t; n = b3_neg(n); d = -d; }
    f->a = a; f->b = b; f->c = c; f->n = n; f->d = d;
    return 1;
}

B3_HD B3_INL int b3_convex_epa_seed(const B3Convex* a, const B3Convex* b,
        B3SupportVertex* seed, int n, float eps) {
    B3SupportVertex points[18];
    int count = n;
    for (int i = 0; i < n; i++) points[i] = seed[i];
    const B3Vec3 dirs[10] = {
        {1,1,1}, {1,-1,-1}, {-1,1,-1}, {-1,-1,1},
        {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}};
    B3Vec3 extra[4]; int extras = 0;
    if (n >= 2) {
        B3Vec3 edge = b3_sub(seed[1].v, seed[0].v);
        if (b3_len2(edge) > eps * eps) {
            extra[extras++] = b3_perp(b3_norm(edge));
            extra[extras++] = b3_neg(extra[0]);
            extra[extras++] = b3_cross(b3_norm(edge), extra[0]);
            extra[extras++] = b3_neg(extra[2]);
        }
    }
    for (int i = 0; i < 10 + extras; i++) {
        B3SupportVertex v = b3_convex_difference(a, b, i < 10 ? dirs[i] : extra[i - 10]);
        int unique = 1;
        for (int j = 0; j < count; j++)
            if (b3_len2(b3_sub(v.v, points[j].v)) <= eps * eps) unique = 0;
        if (unique) points[count++] = v;
    }
    for (int i = 0; i < count; i++) for (int j = i + 1; j < count; j++)
    for (int k = j + 1; k < count; k++) for (int l = k + 1; l < count; l++) {
        B3SupportVertex tetra[4] = {points[i], points[j], points[k], points[l]};
        float weights[4];
        if (b3_convex_tetra(tetra, weights) && weights[0] > 1.0e-6f &&
                weights[1] > 1.0e-6f && weights[2] > 1.0e-6f && weights[3] > 1.0e-6f) {
            for (int p = 0; p < 4; p++) seed[p] = tetra[p];
            return 1;
        }
    }
    return 0;
}

B3_HD B3_INL void b3_convex_face_contact(B3Mani* m, const B3SupportVertex* p,
        const B3ConvexFace* face) {
    B3Vec3 a = p[face->a].v, e = b3_sub(p[face->b].v, a), f = b3_sub(p[face->c].v, a);
    B3Vec3 rhs = b3_sub(b3_mul(face->n, face->d), a);
    float ee = b3_len2(e), ef = b3_dot(e, f), ff = b3_len2(f);
    float den = ee * ff - ef * ef;
    if (den <= 0.0f) { m->status |= B3_COLLISION_DEGENERATE; return; }
    float u = (b3_dot(rhs, e) * ff - b3_dot(rhs, f) * ef) / den;
    float v = (b3_dot(rhs, f) * ee - b3_dot(rhs, e) * ef) / den;
    /* Small negative barycentric values can occur on a shared edge. */
    float weights[3] = {b3_maxf(0.0f, 1.0f - u - v), b3_maxf(0.0f, u), b3_maxf(0.0f, v)};
    float sum = weights[0] + weights[1] + weights[2];
    int indices[3] = {face->a, face->b, face->c};
    B3Vec3 pa = b3_v(0,0,0), pb = b3_v(0,0,0);
    for (int i = 0; i < 3; i++) {
        pa = b3_madd(pa, weights[i] / sum, p[indices[i]].a);
        pb = b3_madd(pb, weights[i] / sum, p[indices[i]].b);
    }
    m->normal = face->n;
    b3_mani_push(m, pa, pb, b3_dot(b3_sub(pb, pa), m->normal), 0xC0000000u);
}

B3_HD B3_INL void b3_convex_epa(B3Mani* m, const B3Convex* a,
        const B3Convex* b, B3SupportVertex* seed, int n, float eps) {
    B3SupportVertex points[B3_CONVEX_VERTICES];
    B3ConvexFace faces[B3_CONVEX_FACES];
    int edge_a[B3_CONVEX_FACES * 3], edge_b[B3_CONVEX_FACES * 3];
    if (!b3_convex_epa_seed(a, b, seed, n, eps)) { m->status |= B3_COLLISION_EPA_SEED; return; }
    for (int i = 0; i < 4; i++) points[i] = seed[i];
    const int initial[4][3] = {{0,1,2}, {0,3,1}, {0,2,3}, {1,3,2}};
    int np = 4, nf = 4;
    for (int i = 0; i < 4; i++) if (!b3_convex_face(&faces[i], points,
            initial[i][0], initial[i][1], initial[i][2])) {
        m->status |= B3_COLLISION_DEGENERATE; return;
    }
    for (int iteration = 0; iteration < B3_CONVEX_EPA_ITERATIONS; iteration++) {
        m->epa_iterations = iteration + 1;
        int closest = 0;
        for (int i = 1; i < nf; i++) if (faces[i].d < faces[closest].d) closest = i;
        B3ConvexFace best = faces[closest];
        B3SupportVertex next = b3_convex_difference(a, b, best.n);
        float gap = b3_dot(next.v, best.n) - best.d;
        if (gap <= eps) { b3_convex_face_contact(m, points, &best); return; }
        if (np >= B3_CONVEX_VERTICES) { m->status |= B3_COLLISION_EPA_CAPACITY; return; }
        for (int i = 0; i < np; i++) if (b3_len2(b3_sub(next.v, points[i].v)) <= eps * eps * 0.01f) {
            m->status |= B3_COLLISION_DEGENERATE; return;
        }
        int edges = 0, kept = 0;
        for (int i = 0; i < nf; i++) {
            B3ConvexFace face = faces[i];
            if (b3_dot(face.n, b3_sub(next.v, points[face.a].v)) > 0.0f) {
                int from[3] = {face.a, face.b, face.c}, to[3] = {face.b, face.c, face.a};
                for (int k = 0; k < 3; k++) {
                    int reverse = -1;
                    for (int j = 0; j < edges; j++) if (edge_a[j] == to[k] && edge_b[j] == from[k]) { reverse = j; break; }
                    if (reverse >= 0) {
                        edge_a[reverse] = edge_a[--edges]; edge_b[reverse] = edge_b[edges];
                    } else {
                        if (edges == B3_CONVEX_FACES * 3) { m->status |= B3_COLLISION_EPA_CAPACITY; return; }
                        edge_a[edges] = from[k]; edge_b[edges++] = to[k];
                    }
                }
            } else faces[kept++] = face;
        }
        nf = kept;
        if (edges == 0) { m->status |= B3_COLLISION_DEGENERATE; return; }
        points[np] = next;
        for (int i = 0; i < edges; i++) {
            if (nf == B3_CONVEX_FACES) { m->status |= B3_COLLISION_EPA_CAPACITY; return; }
            if (!b3_convex_face(&faces[nf], points, edge_a[i], edge_b[i], np)) {
                m->status |= B3_COLLISION_DEGENERATE; return;
            }
            nf++;
        }
        np++;
    }
    m->status |= B3_COLLISION_EPA_LIMIT;
}

/* Closed form sphere versus finite cylinder, including cap/rim and interior. */
B3_HD B3_INL void b3_convex_cylinder_sphere(B3Mani* m, const B3Convex* c,
        const B3Convex* s) {
    B3Vec3 local = b3_inv_rotate(c->rotation, b3_sub(s->position, c->position));
    float radial = sqrtf(local.x * local.x + local.z * local.z);
    float radius = c->shape->radius, half = c->shape->half.y;
    B3Vec3 unit = radial > 0.0f ? b3_v(local.x / radial, 0, local.z / radial) : b3_v(1,0,0);
    B3Vec3 nearest = b3_mul(unit, b3_minf(radial, radius));
    nearest.y = b3_clamp(local.y, -half, half);
    int inside = radial <= radius && fabsf(local.y) <= half;
    B3Vec3 normal;
    if (inside) {
        if (radius - radial <= half - fabsf(local.y)) { nearest = b3_mul(unit, radius); nearest.y = local.y; normal = unit; }
        else { nearest = local; nearest.y = local.y >= 0.0f ? half : -half; normal = b3_v(0, local.y >= 0.0f ? 1.0f : -1.0f, 0); }
    } else normal = b3_norm(b3_sub(local, nearest));
    float distance = b3_len(b3_sub(local, nearest));
    float sep = (inside ? -distance : distance) - s->shape->radius;
    if (sep > B3_SPECULATIVE) return;
    m->normal = b3_rotate(c->rotation, normal);
    b3_mani_push(m, b3_xf_point(c->position, c->rotation, nearest),
        b3_msub(s->position, s->shape->radius, m->normal), sep, 0xC1000000u);
}

/* Parallel cylinders reduce exactly to a disk interval product. */
B3_HD B3_INL void b3_convex_parallel_cylinders(B3Mani* m,
        const B3Convex* a, const B3Convex* b) {
    B3Vec3 axis = b3_rotate(a->rotation, b3_v(0,1,0));
    B3Vec3 delta = b3_sub(b->position, a->position);
    float y = b3_dot(delta, axis), sy = y >= 0.0f ? 1.0f : -1.0f;
    B3Vec3 radial = b3_msub(delta, y, axis);
    float length = b3_len(radial);
    B3Vec3 unit = length > 0.0f ? b3_mul(radial, 1.0f / length) : b3_perp(axis);
    float ra = a->shape->radius, rb = b->shape->radius;
    float ha = a->shape->half.y, hb = b->shape->half.y;
    float dr = length - ra - rb, dy = fabsf(y) - ha - hb;
    B3Vec3 pa, pb, normal;
    if (dr > 0.0f && dy > 0.0f) {
        pa = b3_madd(b3_madd(a->position, ra, unit), sy * ha, axis);
        pb = b3_msub(b3_msub(b->position, rb, unit), sy * hb, axis);
        normal = b3_norm(b3_sub(pb, pa));
    } else if (dr >= dy) {
        float shared = 0.5f * (b3_maxf(-ha, y - hb) + b3_minf(ha, y + hb));
        pa = b3_madd(b3_madd(a->position, ra, unit), shared, axis);
        pb = b3_madd(b3_msub(b->position, rb, unit), shared - y, axis);
        normal = unit;
    } else {
        float shared = b3_clamp(0.5f * length, b3_maxf(0.0f, length - rb), b3_minf(ra, length));
        B3Vec3 base = b3_madd(a->position, shared, unit);
        pa = b3_madd(base, sy * ha, axis);
        pb = b3_madd(base, y - sy * hb, axis);
        normal = b3_mul(axis, sy);
    }
    float sep = b3_dot(b3_sub(pb, pa), normal);
    if (sep > B3_SPECULATIVE) return;
    m->normal = normal;
    b3_mani_push(m, pa, pb, sep, 0xC2000000u);
}

B3_HD B3_INL void b3_collide_convex(B3Mani* m, const B3Body* ba,
        const B3Shape* sa, const B3Body* bb, const B3Shape* sb) {
    b3_mani_clear(m);
    if (sa->type < B3_SPHERE || sa->type > B3_CYLINDER ||
            sb->type < B3_SPHERE || sb->type > B3_CYLINDER) {
        m->status = B3_COLLISION_INVALID_SHAPE; return;
    }
    B3Convex a = {sa, b3_shape_pos(ba, sa), b3_shape_rot(ba, sa)};
    B3Convex b = {sb, b3_shape_pos(bb, sb), b3_shape_rot(bb, sb)};
    if (sa->type == B3_CYLINDER && sb->type == B3_SPHERE) { b3_convex_cylinder_sphere(m, &a, &b); return; }
    if (sb->type == B3_CYLINDER && sa->type == B3_SPHERE) { b3_convex_cylinder_sphere(m, &b, &a); b3_mani_flip(m); return; }
    if (sa->type == B3_CYLINDER && sb->type == B3_CYLINDER) {
        B3Vec3 axis_a = b3_rotate(a.rotation, b3_v(0,1,0)), axis_b = b3_rotate(b.rotation, b3_v(0,1,0));
        if (b3_len2(b3_cross(axis_a, axis_b)) == 0.0f) { b3_convex_parallel_cylinders(m, &a, &b); return; }
    }
    /* Translation-relative support improves numerical conditioning away from
     * the world origin. Witnesses are translated back only on publication. */
    B3Vec3 origin = a.position;
    b.position = b3_sub(b.position, origin); a.position = b3_v(0,0,0);
    float scale = b3_maxf(1.0f, b3_maxf(b3_len(sa->half) + sa->radius, b3_len(sb->half) + sb->radius));
    float eps = 2.0e-6f * scale;
    B3SupportVertex simplex[4];
    B3Vec3 direction = b.position;
    if (b3_len2(direction) == 0.0f) direction = b3_v(1,0,0);
    simplex[0] = b3_convex_difference(&a, &b, direction);
    int count = 1;
    B3Vec3 pa, pb, closest;
    float weights[4];
    B3Vec3 last_normal = b3_norm(direction);
    // Any separating support plane remains valid throughout the simplex
    // search. In particular, exact-touch directions can be lost when a
    // near-zero closest point is normalized in a later iteration.
    B3Vec3 bound_normal = last_normal;
    double lower_bound = -b3_convex_dot64(simplex[0].v, bound_normal);
    for (int iteration = 0; iteration < B3_CONVEX_GJK_ITERATIONS; iteration++) {
        m->gjk_iterations = iteration + 1;
        closest = b3_convex_closest(simplex, &count, weights, &pa, &pb);
        float distance = b3_len(closest);
        if (distance <= eps) {
            // A small simplex distance does not imply penetration. Re-evaluate
            // its current normal before invoking EPA: the simplex distance is
            // an upper bound and the support plane gives the lower bound.
            // A stale preceding normal can send a separated near-touching
            // pair to EPA, where no origin-containing seed can exist.
            B3Vec3 near_normal = distance > 0.0f && count < 4
                ? b3_mul(closest, -1.0f / distance) : last_normal;
            B3SupportVertex support = b3_convex_difference(&a, &b, near_normal);
            double lower = -b3_convex_dot64(support.v, near_normal);
            if (lower > lower_bound) { lower_bound = lower; bound_normal = near_normal; }
#ifdef B3_CONVEX_TRACE
            printf("near_origin n%d distance %.9g current_depth %.9g best_lower %.9g bound_width %.9g\n", count, distance, b3_dot(support.v,near_normal), lower_bound, double(distance)-lower_bound);
#endif
            if (lower_bound >= 0.0 && double(distance) - lower_bound <= eps) {
                m->normal = bound_normal;
                b3_mani_push(m, pa, pb, b3_dot(b3_sub(pb, pa), bound_normal), 0xC3000000u);
            } else b3_convex_epa(m, &a, &b, simplex, count, eps * 4.0f);
            break;
        }
        direction = b3_mul(closest, -1.0f / distance);
        last_normal = direction;
        B3SupportVertex next = b3_convex_difference(&a, &b, direction);
        double lower = -b3_convex_dot64(next.v, direction);
        if (lower > lower_bound) { lower_bound = lower; bound_normal = direction; }
        float gap = distance + b3_dot(next.v, direction);
#ifdef B3_CONVEX_TRACE
        printf("gjk %d n%d distance %.9g gap %.9g closest %.9g %.9g %.9g next %.9g %.9g %.9g\n", iteration, count, distance, gap, closest.x, closest.y, closest.z, next.v.x, next.v.y, next.v.z);
#endif
        if (gap <= eps) {
            if (distance <= B3_SPECULATIVE) {
                m->normal = direction;
                b3_mani_push(m, pa, pb, distance, 0xC4000000u);
            }
            break;
        }
        if (count == 4) { m->status |= B3_COLLISION_DEGENERATE; break; }
        simplex[count++] = next;
        if (iteration + 1 == B3_CONVEX_GJK_ITERATIONS) m->status |= B3_COLLISION_GJK_LIMIT;
    }
    for (int i = 0; i < m->count; i++) {
        m->p_a[i] = b3_add(m->p_a[i], origin);
        m->p_b[i] = b3_add(m->p_b[i], origin);
    }
}
