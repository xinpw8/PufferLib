#define REK_G1_CUDA_DEVICE 1
#include "g1_semantic_scheduler_cuda.h"
#include <math.h>

namespace {
constexpr unsigned THREADS = 128;
unsigned blocks_for(size_t n) {
    const size_t blocks = n / THREADS + (n % THREADS != 0);
    return unsigned(blocks > 65535 ? 65535 : blocks);
}

__device__ int busy(const SonicMotionComposerNative& composer) {
    return composer.current_layer.active && !composer.current_layer.config.loop;
}

__device__ RekG1NativeBaseVelocitySample velocity_sample(const float* values) {
    return {{values[0], values[1], values[2]},
        {values[3], values[4], values[5]}, 1};
}

__device__ int finite_command(RekG1NativeVelocityCommand command) {
    return isfinite(command.forward) && isfinite(command.strafe) && isfinite(command.yaw);
}

/* Same branch conditions as semantic_duel_runtime.c:transition_gate_open.
 * The route-kind array is the validated original static route table. */
__device__ int transition_gate(const RekG1CudaSemanticBuffers& b,
        const RekG1NativeLocomotionState& state,
        const RekG1NativeBaseVelocitySample& velocity, uint8_t* open) {
    const int current = int(state.current_route_id);
    const int transition = int(state.transition_from_route_id);
    const int momentum = int(state.momentum_route_id);
    if (current < 0 || current >= 24 || transition < 0 || transition >= 24
            || momentum < 0 || momentum >= 24) return 0;
    const int ck = b.route_kinds[current], tk = b.route_kinds[transition];
    const int mk = b.route_kinds[momentum];
    if (ck > REK_G1_NATIVE_ROUTE_TURN || tk > REK_G1_NATIVE_ROUTE_TURN
            || mk > REK_G1_NATIVE_ROUTE_TURN
            || state.locomotion_active > 1 || state.transition_settling > 1
            || state.stop_braking > 1 || state.has_momentum > 1
            || !finite_command(state.last_driven_command)
            || !finite_command(state.stop_brake_command)
            || (state.locomotion_active && state.transition_settling)
            || (state.locomotion_active && state.has_momentum)
            || (state.transition_settling && state.has_momentum)
            || (state.transition_settling && state.stop_braking)
            || (state.locomotion_active && ck != 1 && ck != 2)
            || (state.transition_settling && tk != 1 && tk != 2)
            || (state.has_momentum && mk != 1 && mk != 2)
            || (state.stop_braking && ck != 1 && ck != 2)) return 0;
    const int last_translation = fabsf(state.last_driven_command.forward)
            >= REK_G1_NATIVE_COMMAND_EPSILON
        || fabsf(state.last_driven_command.strafe) >= REK_G1_NATIVE_COMMAND_EPSILON;
    const int brake_translation = fabsf(state.stop_brake_command.forward)
            >= REK_G1_NATIVE_COMMAND_EPSILON
        || fabsf(state.stop_brake_command.strafe) >= REK_G1_NATIVE_COMMAND_EPSILON;
    if ((state.locomotion_active && (ck == 1 || last_translation))
            || (state.transition_settling && (tk == 1 || last_translation))
            || (state.stop_braking && (ck == 1 || brake_translation))) {
        *open = 0;
        return 1;
    }
    if (!state.has_momentum || mk == 2) {
        *open = 1;
        return 1;
    }
    return rek_g1_native_transition_settled(state.momentum_route_id,
        &b.config->locomotion, &velocity, open) == REK_G1_NATIVE_COMMAND_OK;
}

__device__ void write_outputs(const RekG1CudaSemanticBuffers& b, size_t i) {
    auto& row = b.rows[i];
    float* out = b.observation12 + i * 12;
    for (int axis = 0; axis < 4; ++axis) out[axis] = b.heading_wxyz[i * 4 + axis];
    out[4] = row.effective_velocity.forward;
    out[5] = row.effective_velocity.strafe;
    out[6] = row.effective_velocity.yaw;
    out[7] = float(row.active_route_id);
    out[8] = float(row.locomotion.locomotion_active);
    out[9] = float(row.locomotion.transition_settling);
    out[10] = b.composers[i].action_playing ? 1.0f : 0.0f;
    out[11] = busy(b.composers[i]) ? 1.0f : 0.0f;
    if (!row.status) {
        const auto status = rek_g1_puffer_write_mask(&row.adapter,
            row.translation_settled, row.action_busy || row.recovery_active,
            b.masks + i * 33, 33);
        if (status) row.status = 100 + int(status);
    }
    if (row.status) for (int c = 0; c < 33; ++c) b.masks[i * 33 + c] = 0;
}

__global__ void table_kernel(RekG1SemanticActionTableStorage* table,
        uint32_t ticks, const uint32_t* durations, int* status) {
    *status = int(rek_g1_semantic_action_table_init(table, ticks, durations));
}

__global__ void reset_kernel(const RekG1CudaSemanticBuffers* buffers,
        const uint8_t* reset, const float* heading, size_t count) {
    const auto& b = *buffers;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            i < count; i += size_t(blockDim.x) * gridDim.x) {
        if (!reset[i]) continue;
        auto& row = b.rows[i];
        row = {};
        auto status = rek_g1_puffer_init(&row.adapter, &b.table->table);
        if (status) row.status = 100 + int(status);
        auto backend = b.composers[i].backends;
        auto composer_status = sonic_motion_composer_native_init(&b.composers[i],
            b.config->command.controller_rate_hz, &backend);
        if (!composer_status) composer_status = sonic_motion_composer_native_play_action_immediate(
            &b.composers[i], &b.routes[0].clip, &b.routes[0].config);
        if (composer_status) row.status = 300 + int(composer_status);
        row.translation_settled = 1;
        for (int axis = 0; axis < 4; ++axis) {
            b.heading_wxyz[i * 4 + axis] = heading[i * 4 + axis];
            if (!isfinite(heading[i * 4 + axis])) row.status = 400;
        }
        write_outputs(b, i);
    }
}

__device__ int play_route(const RekG1CudaSemanticBuffers& b, size_t i, int route) {
    if (route < 0 || route >= 24) return 400;
    const auto status = sonic_motion_composer_native_play_action(&b.composers[i],
        &b.routes[route].clip, &b.routes[route].config);
    return status ? 300 + int(status) : 0;
}

__device__ int compose(const RekG1CudaSemanticBuffers& b, size_t i,
        const RekG1NativeBaseVelocitySample& velocity) {
    auto& row = b.rows[i];
    auto& composer = b.composers[i];
    const auto& semantic = row.semantic;
    const RekG1NativeVelocityCommand command = {
        float(semantic.input.forward), float(semantic.input.strafe), semantic.input.yaw};
    if (semantic.kind == REK_G1_SEMANTIC_DISCRETE_MOVE
            && !semantic.move_start_edge && !composer.action_playing) return 400;
    if (semantic.kind == REK_G1_SEMANTIC_DISCRETE_MOVE && semantic.move_start_edge) {
        if (semantic.move_registry_index >= b.table->table.move_registry_count
                || composer.action_playing) return 400;
        const int move = b.table->move_indices[semantic.move_registry_index];
        const int route = b.move_routes[move];
        const int status = play_route(b, i, route);
        if (status) return status;
        row.locomotion.locomotion_active = 0;
        row.active_route_id = static_cast<RekG1NativeRouteId>(route);
        row.effective_velocity = command;
    } else {
        RekG1NativeRouteSelection selection = {};
        auto status = rek_g1_native_select_locomotion_route(command, &selection);
        if (status) return 200 + int(status);
        if (selection.route_id < 0 || selection.route_id >= 24) return 400;
        const RekG1NativeLocomotionStepInput input = {
            command, velocity, b.config->timing.elapsed_seconds,
            1, uint8_t(composer.action_playing != 0), uint8_t(busy(composer)), 1};
        RekG1NativeLocomotionStepResult result = {};
        status = rek_g1_native_locomotion_step(&row.locomotion,
            &b.config->locomotion, &input, &result);
        if (status) return 200 + int(status);
        if (result.event != REK_G1_NATIVE_LOCOMOTION_EVENT_NONE) {
            const int play_status = play_route(b, i, result.event_route_id);
            if (play_status) return play_status;
            row.active_route_id = result.event_route_id;
        }
        row.locomotion = result.next_state;
        row.effective_velocity = result.effective_velocity;
    }
    RekG1NativePlaybackUpdate playback = {};
    auto status = rek_g1_native_playback_update(row.effective_velocity,
        &b.config->command, &playback);
    if (status) return 200 + int(status);
    if (playback.apply) {
        const auto cs = sonic_motion_composer_native_set_locomotion_speed(&composer, playback.scale);
        if (cs) return 300 + int(cs);
    }
    const auto cs = sonic_motion_composer_native_build_reference_rows(&composer,
        b.reference_timing, b.mirror, &b.references[i]);
    return cs ? 300 + int(cs) : 0;
}

__global__ void pre_kernel(const RekG1CudaSemanticBuffers* buffers,
        const float* actions, const float* velocity, const uint8_t* suspended,
        size_t count) {
    const auto& b = *buffers;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            i < count; i += size_t(blockDim.x) * gridDim.x) {
        auto& row = b.rows[i];
        if (row.status) continue;
        // Native gather_velocity_sample validates every measured component,
        // including rows whose controller is suspended or executing a move.
        const auto sample = velocity_sample(velocity + i * 6);
        if (!finite_command(sample.angular_velocity_local)
                || !finite_command(sample.linear_velocity_local)) {
            row.status = 202;
            write_outputs(b, i);
            continue;
        }
        const auto step = rek_g1_puffer_step(&row.adapter, actions[i], b.config->timing,
            row.translation_settled, row.action_busy || row.recovery_active);
        row.semantic = step.semantic;
        if (step.status) row.status = 100 + int(step.status);
        else if (!suspended[i]) row.status = compose(b, i, sample);
        if (row.status) write_outputs(b, i);
    }
}

__device__ int advance_heading(const RekG1CudaSemanticBuffers& b, size_t i) {
    auto& composer = b.composers[i];
    SonicMotionComposerNativeAdvanceResult advance = {};
    auto cs = sonic_motion_composer_native_advance(&composer, &advance);
    float delta = 0.0f, ownership = 0.0f;
    if (!cs) cs = sonic_motion_composer_native_consume_heading_delta(&composer, &delta);
    if (!cs) cs = sonic_motion_composer_native_heading_clip_ownership(&composer, &ownership);
    if (cs) return 300 + int(cs);
    RekG1NativeHeadingUpdate update = {};
    /* binding.c:current_build_zero_forgiveness returns exact zero for this build. */
    const auto status = rek_g1_native_heading_update(b.rows[i].effective_velocity,
        &b.config->command, ownership, delta, 0.0f, &update);
    if (status) return 200 + int(status);
    volatile float half = update.total_delta_radians * 0.5f;
    float sine = 0.0f, cosine = 0.0f;
    if (!composer.backends.sin_cos_f(composer.backends.context, half, &sine, &cosine)
            || !isfinite(sine) || !isfinite(cosine)) return 400;
    const float* old = b.heading_wxyz + i * 4;
    const float next[4] = {cosine * old[0] - sine * old[3],
        cosine * old[1] - sine * old[2], cosine * old[2] + sine * old[1],
        cosine * old[3] + sine * old[0]};
    for (int axis = 0; axis < 4; ++axis) {
        if (!isfinite(next[axis])) return 400;
        b.heading_wxyz[i * 4 + axis] = next[axis];
    }
    return 0;
}

__global__ void post_kernel(const RekG1CudaSemanticBuffers* buffers,
        const float* velocity, const int32_t* phase, const uint8_t* suspended,
        const uint8_t* input_reset, const uint8_t* reset_event,
        const uint8_t* terminal, size_t count) {
    const auto& b = *buffers;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            i < count; i += size_t(blockDim.x) * gridDim.x) {
        auto& row = b.rows[i];
        if (row.status) continue;
        if (phase[i] < 0 || phase[i] > 2) row.status = 400;
        if (!row.status && !reset_event[i] && !suspended[i]) {
            row.status = advance_heading(b, i);
            if (row.semantic.kind == REK_G1_SEMANTIC_DISCRETE_MOVE
                    && ((row.semantic.segment_complete && b.composers[i].action_playing)
                        || (!row.semantic.segment_complete && !b.composers[i].action_playing)))
                row.status = 400;
        }
        if (terminal[i]) {
            row.translation_settled = 1;
            row.action_busy = row.recovery_active = 0;
        } else {
            uint8_t settled = 0;
            if (!transition_gate(b, row.locomotion, velocity_sample(velocity + i * 6), &settled))
                row.status = 400;
            row.recovery_active = phase[i] == 2 || suspended[i];
            row.translation_settled = row.recovery_active ? 0 : settled;
            row.action_busy = busy(b.composers[i]);
        }
        if (terminal[i] || input_reset[i]) rek_g1_puffer_reset(&row.adapter);
        write_outputs(b, i);
    }
}
} // namespace

extern "C" cudaError_t rek_g1_cuda_semantic_table_init(
        RekG1SemanticActionTableStorage* table, uint32_t ticks,
        const uint32_t* durations, int* status, cudaStream_t stream) {
    if (!table || !durations || !status) return cudaErrorInvalidValue;
    table_kernel<<<1, 1, 0, stream>>>(table, ticks, durations, status);
    return cudaGetLastError();
}

extern "C" cudaError_t rek_g1_cuda_semantic_reset(
        const RekG1CudaSemanticBuffers* b, const uint8_t* reset,
        const float* heading, size_t count, cudaStream_t stream) {
    if (!count) return cudaSuccess;
    if (!b || !reset || !heading) return cudaErrorInvalidValue;
    reset_kernel<<<blocks_for(count), THREADS, 0, stream>>>(b, reset, heading, count);
    return cudaGetLastError();
}

extern "C" cudaError_t rek_g1_cuda_semantic_pre(
        const RekG1CudaSemanticBuffers* b, const float* actions,
        const float* velocity, const uint8_t* suspended, size_t count, cudaStream_t stream) {
    if (!count) return cudaSuccess;
    if (!b || !actions || !velocity || !suspended) return cudaErrorInvalidValue;
    pre_kernel<<<blocks_for(count), THREADS, 0, stream>>>(b, actions, velocity, suspended, count);
    return cudaGetLastError();
}

extern "C" cudaError_t rek_g1_cuda_semantic_post(
        const RekG1CudaSemanticBuffers* b, const float* velocity,
        const int32_t* phase, const uint8_t* suspended, const uint8_t* input_reset,
        const uint8_t* reset_event, const uint8_t* terminal, size_t count, cudaStream_t stream) {
    if (!count) return cudaSuccess;
    if (!b || !velocity || !phase || !suspended || !input_reset || !reset_event || !terminal)
        return cudaErrorInvalidValue;
    post_kernel<<<blocks_for(count), THREADS, 0, stream>>>(b, velocity, phase,
        suspended, input_reset, reset_event, terminal, count);
    return cudaGetLastError();
}
