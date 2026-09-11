// SPDX-License-Identifier: MIT
// Host-only geometry diagnostics. Does not step physics or load game assets.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "engine/puffysics.cuh"

static int checks = 0;
static void require(bool ok, const char* message) {
    checks++;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
static B3Body body(B3Vec3 position = {0,0,0}, B3Quat rotation = {{0,0,0},1}) {
    B3Body b; B3BodyDef def = b3_default_body();
    def.position = position; def.rotation = rotation; def.type = B3_DYNAMIC;
    b3_body_from_def(&b, &def); return b;
}
static B3Shape shape(int type, float radius, B3Vec3 half) {
    B3Shape s; B3ShapeDef def = b3_default_shape();
    b3_shape_fill(&s, 0, type, b3_v(0,0,0), b3_q_id(), radius, half, &def);
    return s;
}
static void contact(const char* name, B3Shape a, B3Shape b,
        B3Vec3 position, B3Quat rotation, float expected, float tolerance = 3.0e-5f) {
    B3Body ba = body(), bb = body(position, rotation);
    B3Mani m, reverse;
    b3_collide_pair(&m, &ba, &a, &bb, &b);
    std::printf("%s status=%u count=%d gjk=%d epa=%d sep=%.9g expected=%.9g\n",
        name, m.status, m.count, m.gjk_iterations, m.epa_iterations,
        m.count ? m.sep[0] : 0.0f, expected);
    require(m.status == 0, name);
    require(m.count == (expected <= B3_SPECULATIVE ? 1 : 0), name);
    if (!m.count) return;
    require(std::fabs(m.sep[0] - expected) <= tolerance, name);
    require(std::fabs(b3_len(m.normal) - 1.0f) <= 2.0e-5f, name);
    require(std::fabs(b3_dot(b3_sub(m.p_b[0], m.p_a[0]), m.normal) - m.sep[0]) <= tolerance, name);
    b3_collide_pair(&reverse, &bb, &b, &ba, &a);
    require(reverse.status == 0 && reverse.count == 1, "pair reversal status");
    require(std::fabs(reverse.sep[0] - m.sep[0]) <= tolerance, "pair reversal distance");
    if (b3_len2(position) > 0.01f)
        require(b3_len(b3_add(reverse.normal, m.normal)) <= 0.03f, "pair reversal normal");
}

extern "C" void b3_test_query(int ta, int tb, const float* parameters, float* out) {
    // radii, half lengths/dimensions; positions; quaternion xyzw.
    B3Shape a = shape(ta, parameters[0], b3_v(parameters[1],parameters[2],parameters[3]));
    B3Shape b = shape(tb, parameters[4], b3_v(parameters[5],parameters[6],parameters[7]));
    B3Body ba = body(b3_v(parameters[8],parameters[9],parameters[10]),
        b3_q(parameters[11],parameters[12],parameters[13],parameters[14]));
    B3Body bb = body(b3_v(parameters[15],parameters[16],parameters[17]),
        b3_q(parameters[18],parameters[19],parameters[20],parameters[21]));
    B3Mani m; b3_collide_pair(&m, &ba, &a, &bb, &b);
    for (int i = 0; i < 15; i++) out[i] = 0;
    out[0] = float(m.status); out[1] = float(m.count);
    out[2] = float(m.gjk_iterations); out[3] = float(m.epa_iterations);
    if (m.count) {
        out[4] = m.sep[0]; out[5] = m.normal.x; out[6] = m.normal.y; out[7] = m.normal.z;
        out[8] = m.p_a[0].x; out[9] = m.p_a[0].y; out[10] = m.p_a[0].z;
        out[11] = m.p_b[0].x; out[12] = m.p_b[0].y; out[13] = m.p_b[0].z;
    }
}

int main() {
    B3Shape c = shape(B3_CYLINDER, .5f, b3_v(0,1,0));
    B3Shape s = shape(B3_SPHERE, .2f, b3_v(0,0,0));
    B3Shape d = shape(B3_CYLINDER, .25f, b3_v(0,.4f,0));
    B3Shape box = shape(B3_BOX, 0, b3_v(.3f,.4f,.2f));
    B3Shape capsule = shape(B3_CAPSULE, .2f, b3_v(0,.3f,0));
    contact("sphere side overlap", c, s, b3_v(.6f,0,0), b3_q_id(), -.1f);
    contact("sphere cap overlap", c, s, b3_v(0,1.1f,0), b3_q_id(), -.1f);
    contact("sphere rim overlap", c, s, b3_v(.6f,1.1f,0), b3_q_id(), std::sqrt(.02f)-.2f);
    contact("sphere outside", c, s, b3_v(2,0,0), b3_q_id(), 1.3f);
    contact("sphere contained", c, s, b3_v(0,0,0), b3_q_id(), -.7f);
    contact("sphere exact cap touch", c, s, b3_v(0,1.2f,0), b3_q_id(), 0);
    contact("parallel cylinders side", c, d, b3_v(.65f,0,0), b3_q_id(), -.1f);
    contact("parallel cylinders cap", c, d, b3_v(0,1.3f,0), b3_q_id(), -.1f);
    contact("parallel cylinders concentric", c, d, b3_v(0,0,0), b3_q_id(), -.75f);
    contact("parallel cylinders rim separation", c, d, b3_v(.76f,1.41f,0), b3_q_id(), std::sqrt(.0002f));
    contact("box cylinder side", c, box, b3_v(.7f,0,0), b3_q_id(), -.1f);
    contact("box cylinder cap", c, box, b3_v(0,1.3f,0), b3_q_id(), -.1f);
    contact("box cylinder side separation", c, box, b3_v(.81f,0,0), b3_q_id(), .01f);
    contact("box cylinder cap separation", c, box, b3_v(0,1.41f,0), b3_q_id(), .01f);
    contact("box cylinder outside", c, box, b3_v(1,0,0), b3_q_id(), .2f);
    contact("capsule cylinder side", c, capsule, b3_v(.6f,0,0), b3_q_id(), -.1f);
    contact("capsule cylinder cap", c, capsule, b3_v(0,1.4f,0), b3_q_id(), -.1f);
    contact("capsule cylinder outside", c, capsule, b3_v(2,0,0), b3_q_id(), 1.3f);
    // Perpendicular equal cylinders are separated along X by r_a+r_b.
    B3Quat transverse = b3_q_axis_angle(b3_v(1,0,0), B3_PI * .5f);
    contact("skew cylinders side", c, d, b3_v(.65f,0,0), transverse, -.1f, 6.0e-5f);
    contact("skew cylinders separated", c, d, b3_v(.76f,0,0), transverse, .01f, 6.0e-5f);

    // Exact support value and tight rotated AABB against a closed-form support function.
    for (int i = 1; i <= 300; i++) {
        B3Body b = body(b3_v(2,-3,4), b3_q_axis_angle(b3_v(.2f,.6f,.3f), i * .031f));
        B3Convex convex = {&c, b.position, b.rotation};
        B3Vec3 n = b3_norm(b3_v(std::sin(i*.17f), std::cos(i*.11f), std::sin(i*.31f)));
        B3Vec3 axis = b3_rotate(b.rotation, b3_v(0,1,0));
        float axial = b3_dot(n, axis);
        float expected = b3_dot(n,b.position) + c.half.y*std::fabs(axial)
            + c.radius*std::sqrt(b3_maxf(0,1-axial*axial));
        require(std::fabs(b3_dot(n,b3_convex_support(&convex,n))-expected) <= 2e-6f, "analytic support value");
        B3AABB bounds = b3_shape_aabb(&b,&c);
        B3Vec3 p = b3_convex_support(&convex,n);
        require(p.x >= bounds.lo.x-1e-6f && p.x <= bounds.hi.x+1e-6f &&
            p.y >= bounds.lo.y-1e-6f && p.y <= bounds.hi.y+1e-6f &&
            p.z >= bounds.lo.z-1e-6f && p.z <= bounds.hi.z+1e-6f, "rotated cylinder AABB");
    }
    B3World* world = new B3World; b3_world_init(world);
    B3BodyDef bd = b3_default_body(); bd.type = B3_DYNAMIC;
    int a = b3_create_body(world,&bd); bd.type = B3_STATIC;
    int b = b3_create_body(world,&bd);
    B3ShapeDef sd = b3_default_shape(); sd.category = 1; sd.mask = 0;
    b3_create_cylinder_local(world,a,1,.5f,b3_v(0,0,0),b3_q_id(),&sd);
    sd.category = 0; sd.mask = 1;
    b3_create_sphere(world,b,b3_v(.6f,0,0),.2f,&sd);
    b3_find_contacts(world);
    require(world->contact_count == (B3_MUJOCO_COLLISION_FILTER ? 1 : 0), "directional collision filter");
    require(world->collision_status == 0, "sticky collision status");
    delete world;
    std::printf("PASS checks=%d filter_or=%d\n",checks,B3_MUJOCO_COLLISION_FILTER);
    return 0;
}
