#pragma once

#include "model_data.h"
#include "warp_abi.h"
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

namespace rek_mjgpu {

// These descriptors are created once on the host and bound to cached CUDA
// functions/device addresses before graph capture. No string lookup, allocation,
// or model traversal belongs in the execution of the bound schedule.
struct ScheduleParameter {
    enum class Kind { Array, I32, F32, Boolean };
    Kind kind;
    std::string field;
    int32_t integer = 0;
    float scalar = 0;
    bool boolean = false;
    ScheduleParameter(const char* value):kind(Kind::Array),field(value){}
    ScheduleParameter(const std::string& value):kind(Kind::Array),field(value){}
    ScheduleParameter(int32_t value):kind(Kind::I32),integer(value){}
    ScheduleParameter(float value):kind(Kind::F32),scalar(value){}
    ScheduleParameter(bool value):kind(Kind::Boolean),boolean(value){}
};

struct ScheduleNode {
    enum class Kind { Kernel, Copy, Zero };
    Kind kind = Kind::Kernel;
    std::string module;
    std::string entry_prefix;
    LaunchBounds bounds;
    // Zero selects the exact WP_TILE_BLOCK_DIM recorded in the cached module.
    int block_dim = 0;
    int shared_bytes = 0;
    std::vector<ScheduleParameter> parameters;
    std::string destination;
    std::string source;
};
using ScheduleSpec = std::vector<ScheduleNode>;

// Exact installed MuJoCo-Warp 1.12.0 subpasses, expressed using their cached
// compiled kernels. These named partial passes do not claim a complete step.
void append_kinematics_com(ScheduleSpec&, const ModelData&);
void append_crb(ScheduleSpec&, const ModelData&);
void append_camlight(ScheduleSpec&, const ModelData&);
void append_transmission(ScheduleSpec&, ModelData&);
void append_spatial_velocity(ScheduleSpec&, const ModelData&);
void append_velocity_forces(ScheduleSpec&, const ModelData&);
void append_actuation(ScheduleSpec&, const ModelData&);
void append_smooth_acceleration(ScheduleSpec&, const ModelData&);
void append_implicitfast_integration(ScheduleSpec&, ModelData&);
void append_sparse_factor_solve(ScheduleSpec&, const ModelData&, const std::string& matrix,
    const std::string& factor, const std::string& diagonal_inverse,
    const std::string& output, const std::string& rhs);
ScheduleSpec build_kinematics_com(const ModelData&);

} // namespace rek_mjgpu
