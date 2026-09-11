#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>

static float test_armature = .02f;
#define B3_ART_JOINT_ARMATURE(joint_id) ((joint_id) >= 0 ? test_armature : 0.0f)
#define B3_MAX_JOINTS 64
#define B3_ART_TORSION 0
#include "../engine/puffysics.cuh"

static int dynamic_body(B3World* w, B3Vec3 position) {
    B3BodyDef d = b3_default_body();
    d.type = B3_DYNAMIC;
    d.position = position;
    int id = b3_create_body(w, &d);
    b3_set_inertial(w, id, 1, b3_v(0,0,0), b3_v(.1f,.1f,.1f));
    return id;
}

static void solve4(double M[4][4], double b[4], double out[4]) {
    for (int col = 0; col < 4; ++col) {
        int pivot = col;
        for (int row = col + 1; row < 4; ++row)
            if (std::fabs(M[row][col]) > std::fabs(M[pivot][col])) pivot = row;
        for (int j = 0; j < 4; ++j) std::swap(M[col][j], M[pivot][j]);
        std::swap(b[col], b[pivot]);
        double scale = M[col][col];
        if (std::fabs(scale) < 1e-12) std::exit(3);
        for (int j = col; j < 4; ++j) M[col][j] /= scale;
        b[col] /= scale;
        for (int row = 0; row < 4; ++row) {
            if (row == col) continue;
            scale = M[row][col];
            for (int j = col; j < 4; ++j) M[row][j] -= scale * M[col][j];
            b[row] -= scale * b[col];
        }
    }
    for (int j = 0; j < 4; ++j) out[j] = b[j];
}

static void free_com() {
    auto w = std::make_unique<B3World>();
    auto art = std::make_unique<B3Art>();
    b3_world_init(w.get());
    w->gravity = b3_v(0,0,0);
    int body = dynamic_body(w.get(), b3_v(0,0,0));
    w->bodies[body].lin_vel = b3_v(1,0,0);
    w->bodies[body].ang_vel = b3_v(0,0,1);
    if (!b3_art_from_world(art.get(), w.get())) std::exit(2);
    int link = b3_art_link_of(art.get(), body);
    b3_art_aba(art.get(), w.get());
    B3Vec3 a = art->a[link].v;
    std::printf("{\"case\":\"corrected_free_COM\",\"acceleration\":[%.9g,%.9g,%.9g],\"expected\":[0,0,0]}\n",a.x,a.y,a.z);
    if (b3_dot(a,a) > 1e-10 || b3_dot(art->a[link].w,art->a[link].w)>1e-10) std::exit(5);
    b3_art_integrate_vel(art.get(), w.get(), .002f);
    if (b3_len(b3_sub(w->bodies[body].lin_vel,b3_v(1,0,0)))>1e-7 ||
        b3_len(b3_sub(w->bodies[body].ang_vel,b3_v(0,0,1)))>1e-7) std::exit(8);
}

static void floating_hinge() {
    auto w = std::make_unique<B3World>();
    auto art = std::make_unique<B3Art>();
    b3_world_init(w.get());
    w->gravity = b3_v(0,0,0);
    int parent = dynamic_body(w.get(), b3_v(0,0,0));
    int child = dynamic_body(w.get(), b3_v(1,1,0));
    b3_create_revolute(w.get(), parent, child, b3_v(1,0,0), b3_v(0,-1,0), b3_v(0,0,1));
    w->bodies[parent].lin_vel = b3_v(1,0,0);
    w->bodies[parent].ang_vel = b3_v(0,0,1);
    w->bodies[child].lin_vel = b3_v(-1,1,0);
    w->bodies[child].ang_vel = b3_v(0,0,2);
    if (!b3_art_bind(art.get(), w.get())) std::exit(2);
    int p = b3_art_link_of(art.get(), parent);
    int c = b3_art_link_of(art.get(), child);
    // Independent generalized mass matrix, including hinge rotor inertia.
    double M[4][4] = {{2,0,-1,-1},{0,2,1,0},{-1,1,2.2,1.1},{-1,0,1.1,1.12}};
    double rhs[4] = {1,4,3,-1};
    double expected[4];
    solve4(M, rhs, expected);
    b3_art_aba(art.get(), w.get());
    double actual[4] = {art->a[p].v.x,art->a[p].v.y,art->a[p].w.z,art->qdd[c]};
    double error=0;
    for (int j=0;j<4;++j) error=std::max(error,std::fabs(actual[j]-expected[j]));
    B3Vec3 pdot=b3_add(art->a[p].v,art->a[c].v);
    std::printf("{\"case\":\"corrected_floating_hinge\",\"qdd\":[%.9g,%.9g,%.9g,%.9g],\"expected\":[%.9g,%.9g,%.9g,%.9g],\"max_error\":%.9g,\"linear_momentum_derivative\":[%.9g,%.9g,%.9g]}\n",actual[0],actual[1],actual[2],actual[3],expected[0],expected[1],expected[2],expected[3],error,pdot.x,pdot.y,pdot.z);
    if (error > 2e-5 || b3_dot(pdot,pdot)>1e-9) std::exit(7);
}

int main() {
    free_com();
    floating_hinge();
    std::puts("PASS: actual corrected b3_art_aba matches independent free-COM and floating-hinge mechanics");
}
