#include "fast_assets.h"
#include <mujoco/mujoco.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <stdexcept>

static unsigned forbidden_calls=0;
extern "C" void __wrap_mj_step(const mjModel*,mjData*) {++forbidden_calls;throw std::runtime_error("CPU physics step forbidden");}
extern "C" void __wrap_mj_forward(const mjModel*,mjData*) {++forbidden_calls;throw std::runtime_error("CPU forward dynamics forbidden");}
extern "C" void __wrap_mj_step1(const mjModel*,mjData*) {++forbidden_calls;throw std::runtime_error("CPU split step forbidden");}
extern "C" void __wrap_mj_step2(const mjModel*,mjData*) {++forbidden_calls;throw std::runtime_error("CPU split step forbidden");}

int main(int argc,char** argv) {
    try {
        if(argc!=4)throw std::runtime_error("Usage: fast-assets-probe MODEL_XML ASSETS_DIRECTORY FEATURES_DIRECTORY");
        RekNative5Config cfg{};cfg.abi_version=REK_NATIVE5_RUNTIME_ABI;cfg.model_path=argv[1];cfg.assets_path=argv[2];cfg.motion_features_path=argv[3];cfg.arenas=1;cfg.locomotion_segment_ticks=1;
        const std::uint32_t durations[]={35,27,31,45,32,45,157,145,158,139,134,138,73,75,68,71,103};std::copy(durations,durations+17,cfg.move_duration_ticks);
        const auto assets=load_fast_assets(cfg);size_t values=0;double max_q_error=0,min_radius=1e30,max_radius=0;
        for(const auto& frame:assets.frames){
            double norm=0;for(float q:frame.root_wxyz)norm+=double(q)*q;max_q_error=std::max(max_q_error,std::abs(std::sqrt(norm)-1));
            auto finite=[&](float x){if(!std::isfinite(x))throw std::runtime_error("Nonfinite frame value");++values;};
            for(float q:frame.q)finite(q);
            for(float q:frame.root_wxyz)finite(q);
            finite(frame.root_z);finite(frame.clip_yaw);
            for(int j=0;j<6;j++){for(float x:frame.strike_xyz[j])finite(x);finite(frame.strike_radius[j]);min_radius=std::min(min_radius,double(frame.strike_radius[j]));max_radius=std::max(max_radius,double(frame.strike_radius[j]));}
            for(int j=0;j<3;j++){for(float x:frame.target_xyz[j])finite(x);finite(frame.target_radius[j]);}
        }
        if(max_q_error>1e-5||forbidden_calls)throw std::runtime_error("Quaternion or CPU execution invariant failed");
        for(int category=16;category<33;category++){const auto& route=assets.routes.at(assets.action_to_route[category]);if(route.count!=int(cfg.move_duration_ticks[route.move])+1)throw std::runtime_error("Duration mapping mismatch");}
        std::puts(assets.provenance_json.c_str());
        std::printf("{\"test\":\"fast_asset_offline_fk\",\"passed\":true,\"checked_finite_values\":%zu,\"max_root_quaternion_norm_error\":%.9g,\"min_striker_proxy_radius_m\":%.9g,\"max_striker_proxy_radius_m\":%.9g,\"forbidden_cpu_physics_calls\":%u,\"python_runtime\":false}\n",values,max_q_error,min_radius,max_radius,forbidden_calls);
        return 0;
    }catch(const std::exception& error){std::fprintf(stderr,"%s\n",error.what());return 1;}
}
