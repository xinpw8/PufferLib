#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

// Host descriptor; every array pointer refers to contiguous device storage.
// Parent kernels may consume these buffers directly. Operations enqueue work
// on stream without allocating or synchronizing, except create/destroy.
struct RobotState {
    std::size_t rows = 0;
    cudaStream_t stream = nullptr;
    float* encoder_observations = nullptr; // [rows,1762]
    float* decoder_observations = nullptr; // [rows,994]
    float* history = nullptr;              // [rows,930], decoder feature groups
    float* last_actions = nullptr;         // [rows,29], clipped policy order
    float* targets = nullptr;              // [rows,29], MuJoCo order
    float* filtered = nullptr;             // [rows,29], before joint limits
    float* retained_targets = nullptr;     // [rows,29]
    float* controls = nullptr;             // [rows,29]
    std::uint8_t* active = nullptr;         // controller_enabled && !dampened && !resetting
    std::uint8_t* controller_enabled = nullptr;
    std::uint8_t* initialized = nullptr;    // drive filter initialized
    std::uint8_t* dampened = nullptr;
    std::uint8_t* resetting = nullptr;
    std::uint8_t* joint_limited = nullptr;  // [2,29]
    float* joint_ranges = nullptr;         // [2,29,2]
    void* float_storage = nullptr;
    void* flag_storage = nullptr;
};

// Joint metadata is host storage for the two fighter sides. Rows must be even.
RobotState* robot_state_create(std::size_t rows, cudaStream_t stream,
    const std::uint8_t* joint_limited, const float* joint_ranges,
    char* error, std::size_t error_capacity);
void robot_state_destroy(RobotState* state);

// enabled is a device uint8 mask supplied by the parent's scheduler. It gates
// controller history/actions. Drive filtering retains its separate suspended
// branches, matching gpu_actuator_drive.py. Null enables every controller row.
cudaError_t robot_state_set_active(RobotState* state, const std::uint8_t* enabled);

// All inputs are device float32. Base/heading [rows,4] are WXYZ; omega [rows,3]
// is local. Joint arrays are [rows,29] in MuJoCo order. References have ten
// future samples: [rows,10,29], [rows,10,29], [rows,10,4] (XYZW).
cudaError_t robot_state_prepare(RobotState* state, const float* base_wxyz,
    const float* local_omega, const float* joints, const float* velocities,
    const float* heading_wxyz, const float* reference_joints,
    const float* reference_next_joints, const float* reference_root_xyzw);
cudaError_t robot_state_decoder_input(RobotState* state, const float* tokens);
cudaError_t robot_state_apply_actions(RobotState* state, const float* raw_actions);

// Row masks are device uint8 arrays; null means every row for reset operations.
// History reset preserves drive flags. Complete reset clears both history and
// drive state, and re-enables the selected controller rows.
cudaError_t robot_state_reset_history(RobotState* state, const std::uint8_t* rows);
cudaError_t robot_state_complete_reset(RobotState* state, const std::uint8_t* rows);
cudaError_t robot_state_set_dampened(RobotState* state,
    const std::uint8_t* desired, const float* live_controls);
cudaError_t robot_state_begin_reset(RobotState* state,
    const std::uint8_t* rows, const float* live_controls);

// Substep 0..9 within the existing 500 Hz physics / 50 Hz controller tick.
// Filter updates at even substeps; writes state->controls without modifying
// state->targets or pre-limit filter storage.
cudaError_t robot_state_drive_prepare(RobotState* state,
    const float* joints, const float* velocities, int substep);
