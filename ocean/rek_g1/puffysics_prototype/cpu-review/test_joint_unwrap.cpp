#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#define B3_MAX_JOINTS 64
#include "../engine/puffysics.cuh"

int main(int argc, char** argv) {
    if (argc < 3 || argc % 2 != 1) return 2;
    auto w = std::make_unique<B3World>();
    b3_world_init(w.get());
    B3BodyDef d = b3_default_body();
    int parent = b3_create_body(w.get(), &d);
    d.type = B3_DYNAMIC;
    int child = b3_create_body(w.get(), &d);
    int joint = b3_create_revolute(w.get(),parent,child,b3_v(0,0,0),b3_v(0,0,0),b3_v(0,0,1));
    float max_error = 0;
    int samples = 0, wrap_intervals = 0, boundary_samples = 0;
    for (int index = 1; index < argc; index += 2) {
        float lo = std::strtof(argv[index], nullptr);
        float hi = std::strtof(argv[index + 1], nullptr);
        float midpoint = .5f * (lo + hi);
        if (!(hi > lo && hi - lo < 2 * B3_PI)) return 3;
        bool crosses_wrap = (lo < -B3_PI || hi > B3_PI);
        wrap_intervals += crosses_wrap;
        auto check = [&](float angle, bool boundary) {
            if (angle < lo || angle > hi) return;
            w->bodies[child].rotation = b3_q_axis_angle(b3_v(0,0,1),angle);
            float raw = b3_joint_angle(w.get(),joint);
            // Identical float expression to native rp_joint_angle.
            float unwrapped = raw + 2.0f * B3_PI * std::nearbyint((midpoint - raw) / (2.0f * B3_PI));
            float error = std::fabs(unwrapped - angle);
            max_error = std::fmax(max_error,error);
            ++samples;
            boundary_samples += boundary;
            if (!std::isfinite(unwrapped) || error > 1e-6f) {
                std::fprintf(stderr,"interval=%d angle=%.9g raw=%.9g unwrapped=%.9g error=%.9g\n",index/2,angle,raw,unwrapped,error);
                std::exit(4);
            }
        };
        for (int i = 0; i <= 10000; ++i) check(lo+(hi-lo)*float(i)/10000.0f,false);
        for (int sign : {-1,1}) for (float delta : {-1e-4f,-1e-6f,0.0f,1e-6f,1e-4f})
            check(float(sign)*B3_PI+delta,true);
    }
    std::printf("{\"case\":\"actual_exported_hinge_intervals\",\"intervals\":%d,\"intervals_crossing_principal_wrap\":%d,\"samples\":%d,\"boundary_samples\":%d,\"max_abs_error_rad\":%.9g}\n",(argc-1)/2,wrap_intervals,samples,boundary_samples,max_error);
    std::puts("PASS: native midpoint unwrap recovers interval-contained angles through +/-pi");
}
