#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>

static float test_armature = 0.02f;
#define B3_ART_JOINT_ARMATURE(joint_id) ((joint_id) >= 0 ? test_armature : 0.0f)
#define B3_MAX_JOINTS 64
#define B3_ART_TORSION 0
#include "../engine/puffysics.cuh"

static void close_to(const char* label, float actual, float expected) {
    float error = std::fabs(actual - expected);
    if (!std::isfinite(actual) || error > 2e-6f * std::fmax(1.0f, std::fabs(expected))) {
        std::fprintf(stderr, "%s actual=%.9g expected=%.9g error=%.9g\n", label, actual, expected, error);
        std::exit(1);
    }
}

static void run(float rotor_inertia) {
    test_armature = rotor_inertia;
    auto world = std::make_unique<B3World>();
    auto art = std::make_unique<B3Art>();
    b3_world_init(world.get());
    world->gravity = b3_v(0, 0, 0);
    B3BodyDef definition = b3_default_body();
    definition.type = B3_STATIC;
    int parent = b3_create_body(world.get(), &definition);
    definition.type = B3_DYNAMIC;
    int child = b3_create_body(world.get(), &definition);
    b3_set_inertial(world.get(), child, 1.0f, b3_v(0, 0, 0), b3_v(.1f, .1f, .1f));
    int joint = b3_create_revolute(world.get(), parent, child,
        b3_v(0, 0, 0), b3_v(0, 0, 0), b3_v(0, 0, 1));
    if (joint != 0 || !b3_art_bind(art.get(), world.get())) std::exit(2);
    world->bodies[child].torque = b3_v(0, 0, 1);
    constexpr float h = .002f;
    float inverse_inertia = 1.0f / (.1f + rotor_inertia);
    b3_art_integrate_vel(art.get(), world.get(), h);
    float qd = b3_joint_speed(world.get(), joint);
    close_to("velocity", qd, h * inverse_inertia);
    close_to("stationary COM x", world->bodies[child].lin_vel.x, 0);
    close_to("stationary COM y", world->bodies[child].lin_vel.y, 0);
    close_to("stationary COM z", world->bodies[child].lin_vel.z, 0);

    B3ArtRow torque_row;
    b3_art_make_row(art.get(), parent, child, b3_v(0, 0, 0), b3_v(0, 0, 0),
        b3_v(0, 0, 1), &torque_row);
    torque_row.torque = 1;
    float torque_response = b3_art_response_w(art.get(), world.get(), &torque_row);
    close_to("torque response", torque_response, inverse_inertia);
    world->bodies[child].ang_vel = b3_v(0, 0, 0);
    b3_art_apply_impulse(art.get(), world.get(), &torque_row, 1.0f);
    close_to("torque impulse velocity", b3_joint_speed(world.get(), joint), inverse_inertia);

    B3ArtRow contact_row;
    b3_art_make_row(art.get(), parent, child, b3_v(1, 0, 0), b3_v(1, 0, 0),
        b3_v(0, 1, 0), &contact_row);
    float contact_response = b3_art_response_w(art.get(), world.get(), &contact_row);
    close_to("unit lever contact response", contact_response, inverse_inertia);
    world->bodies[child].ang_vel = b3_v(0, 0, 0);
    world->bodies[child].lin_vel = b3_v(0, 0, 0);
    b3_art_apply_impulse(art.get(), world.get(), &contact_row, 1.0f);
    close_to("contact impulse velocity", b3_joint_speed(world.get(), joint), inverse_inertia);
    std::printf("{\"armature\":%.9g,\"Iz\":0.1,\"torque\":1,\"h\":0.002,\"qd\":%.9g,\"expected_qd\":%.9g,\"torque_response\":%.9g,\"contact_response\":%.9g,\"expected_response\":%.9g}\n",
        rotor_inertia, qd, h * inverse_inertia, torque_response, contact_response, inverse_inertia);
}

int main() {
    run(0.0f);
    run(0.02f);
    std::puts("PASS: ABA acceleration and Delassus torque/contact response include joint armature");
}
