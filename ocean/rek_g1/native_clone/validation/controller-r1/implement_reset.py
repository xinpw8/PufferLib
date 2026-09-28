from pathlib import Path

root=Path(__file__).parent/'source-r1/ocean/rek_g1'
def edit(name, old, new):
    p=root/name
    s=p.read_text()
    assert s.count(old)==1, (name,old[:80],s.count(old))
    p.write_text(s.replace(old,new),newline='\n')

edit('sonic_motion_composer_native.h', 'REK_G1_FN SonicMotionComposerNativeStatus sonic_motion_composer_native_build_reference_rows(', '''/* One original GetReferenceFrame call, with the same native joint order as
 * build_reference_rows. Outputs are changed only after a successful sample. */
REK_G1_FN SonicMotionComposerNativeStatus sonic_motion_composer_native_reference_frame(
    const SonicMotionComposerNative* composer, int32_t frames_ahead,
    const SonicMotionComposerNativeMirrorTable* mirror_table,
    float joint_positions[29], float root_quaternion_wxyz[4]);

REK_G1_FN SonicMotionComposerNativeStatus sonic_motion_composer_native_build_reference_rows(''')
edit('sonic_motion_composer_native.c','REK_G1_FN SonicMotionComposerNativeStatus sonic_motion_composer_native_build_reference_rows(','''REK_G1_FN SonicMotionComposerNativeStatus sonic_motion_composer_native_reference_frame(
        const SonicMotionComposerNative* composer, int32_t frames_ahead,
        const SonicMotionComposerNativeMirrorTable* mirror_table,
        float joint_positions[29], float root_quaternion_wxyz[4]) {
    SonicMotionComposerNativePose pose;
    SonicMotionComposerNativeStatus status = validate_composer(composer);
    if (status) return status;
    if (joint_positions == NULL || root_quaternion_wxyz == NULL)
        return SONIC_MOTION_COMPOSER_NATIVE_INVALID_ARGUMENT;
    if (composer->current_layer.active && (composer->current_layer.config.mirror
            || (composer->from_layer.active && composer->from_layer.config.mirror))) {
        status = validate_mirror_table(mirror_table);
        if (status) return status;
    }
    status = get_reference_frame(composer, frames_ahead, mirror_table, &pose);
    if (status) return status;
    memcpy(joint_positions, pose.joint_positions, sizeof(pose.joint_positions));
    memcpy(root_quaternion_wxyz, pose.root_quaternion_wxyz, sizeof(pose.root_quaternion_wxyz));
    return SONIC_MOTION_COMPOSER_NATIVE_OK;
}

REK_G1_FN SonicMotionComposerNativeStatus sonic_motion_composer_native_build_reference_rows(''')

(root/'g1_heading_native.h').write_text('''#pragma once
#include "sonic_motion_composer_native.h"
#include <math.h>

/* Recovered SonicPolicyRunner.CalcHeadingMj/YawQuatMj operation order.
 * Transcendental callbacks retain their declared backend provenance. */
REK_G1_FN static inline int rek_g1_heading_angle(
        const SonicMotionComposerNativeBackends* b, const float q[4], float* out) {
    if (!b || !b->atan2_f || !q || !out) return 0;
    for (int i=0; i<4; ++i) if (!isfinite(q[i])) return 0;
    volatile float yx=q[2]*q[1], zw=q[3]*q[0];
    volatile float sum=yx+zw, numerator=2.0f*sum;
    volatile float zz=q[3]*q[3], yy=q[2]*q[2];
    volatile float sq=zz+yy, twice=2.0f*sq, denominator=1.0f-twice;
    return b->atan2_f(b->context,numerator,denominator,out) && isfinite(*out);
}

REK_G1_FN static inline int rek_g1_initial_heading(
        const SonicMotionComposerNativeBackends* b, const float base[4],
        const float reference[4], float out[4]) {
    float base_angle, reference_angle, sine, cosine;
    if (!b || !b->sin_cos_f || !out
            || !rek_g1_heading_angle(b,base,&base_angle)
            || !rek_g1_heading_angle(b,reference,&reference_angle)) return 0;
    volatile float difference=base_angle-reference_angle, half=difference*0.5f;
    if (!b->sin_cos_f(b->context,half,&sine,&cosine)
            || !isfinite(sine) || !isfinite(cosine)) return 0;
    out[0]=cosine; out[1]=0; out[2]=0; out[3]=sine;
    return 1;
}
''',newline='\n')
edit('g1_semantic_scheduler_cuda.h','/* local_velocity order:', '''/* Original callback bodies are ordered explicitly: the source does not prove
 * global Unity event subscription order. Cold construction uses semantic_reset.
 * base_wxyz is the actual current body orientation, not a cached idle heading.
 * preserve_dispatch_mode retains the direct/categorical latch across body reset;
 * a full external reset passes zero and separately clears its runtime mode. */
typedef enum RekG1CudaResetOrder {
    REK_G1_RESET_RUNNER_THEN_INPUT = 1,
    REK_G1_RESET_INPUT_THEN_RUNNER = 2
} RekG1CudaResetOrder;
extern "C" cudaError_t rek_g1_cuda_semantic_reset_ordered(
    const RekG1CudaSemanticBuffers* buffers, const uint8_t* reset,
    const float* base_wxyz, RekG1CudaResetOrder order,
    uint8_t preserve_dispatch_mode, size_t count, cudaStream_t stream);

/* local_velocity order:''')
edit('g1_semantic_scheduler_cuda.cu','#include <math.h>','#include <math.h>\n#include "g1_heading_native.h"')
edit('g1_semantic_scheduler_cuda.cu','__device__ int play_route(','''__global__ void reset_ordered_kernel(const RekG1CudaSemanticBuffers* buffers,
        const uint8_t* reset, const float* base, RekG1CudaResetOrder order,
        uint8_t preserve_dispatch_mode, size_t count) {
    const auto& b = *buffers;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            i < count; i += size_t(blockDim.x) * gridDim.x) {
        if (!reset[i]) continue;
        auto& row = b.rows[i];
        if ((order != REK_G1_RESET_RUNNER_THEN_INPUT && order != REK_G1_RESET_INPUT_THEN_RUNNER)
                || preserve_dispatch_mode > 1) {
            row.status = 400; write_outputs(b,i); continue;
        }
        auto candidate = b.composers[i];
        SonicMotionComposerNativeStatus status = SONIC_MOTION_COMPOSER_NATIVE_OK;
        if (order == REK_G1_RESET_INPUT_THEN_RUNNER)
            status = sonic_motion_composer_native_play_action(&candidate,&b.routes[0].clip,&b.routes[0].config);
        if (!status) status = sonic_motion_composer_native_reset(&candidate);
        float positions[29], reference[4], heading[4];
        if (!status) status = sonic_motion_composer_native_reference_frame(&candidate,0,b.mirror,positions,reference);
        if (!status && !rek_g1_initial_heading(&candidate.backends,base+i*4,reference,heading))
            status = SONIC_MOTION_COMPOSER_NATIVE_BACKEND_FAILURE;
        if (!status && order == REK_G1_RESET_RUNNER_THEN_INPUT)
            status = sonic_motion_composer_native_play_action(&candidate,&b.routes[0].clip,&b.routes[0].config);
        if (status) { row.status=300+int(status); write_outputs(b,i); continue; }
        const auto previous_mode = row.direct_native_active;
        row = {};
        const auto adapter_status = rek_g1_puffer_init(&row.adapter,&b.table->table);
        if (adapter_status) row.status=100+int(adapter_status);
        row.direct_native_active = preserve_dispatch_mode ? previous_mode : 0;
        row.translation_settled=1;
        b.composers[i]=candidate;
        for (int axis=0;axis<4;++axis) b.heading_wxyz[i*4+axis]=heading[axis];
        write_outputs(b,i);
    }
}

__device__ int play_route(''')
edit('g1_semantic_scheduler_cuda.cu','extern "C" cudaError_t rek_g1_cuda_semantic_pre(','''extern "C" cudaError_t rek_g1_cuda_semantic_reset_ordered(
        const RekG1CudaSemanticBuffers* b, const uint8_t* reset,
        const float* base, RekG1CudaResetOrder order,
        uint8_t preserve_dispatch_mode, size_t count, cudaStream_t stream) {
    if (order != REK_G1_RESET_RUNNER_THEN_INPUT && order != REK_G1_RESET_INPUT_THEN_RUNNER)
        return cudaErrorInvalidValue;
    if (preserve_dispatch_mode > 1) return cudaErrorInvalidValue;
    if (!count) return cudaSuccess;
    if (!b || !reset || !base) return cudaErrorInvalidValue;
    reset_ordered_kernel<<<blocks_for(count), THREADS, 0, stream>>>(b,reset,base,order,preserve_dispatch_mode,count);
    return cudaGetLastError();
}

extern "C" cudaError_t rek_g1_cuda_semantic_pre(''')
edit('native5/motion_assets.cuh','    void pre(const float* actions,', '''    void reset_ordered(const uint8_t* reset_flags, const float* base_wxyz,
        RekG1CudaResetOrder order, bool preserve_dispatch_mode, cudaStream_t stream);
    void pre(const float* actions,''')
edit('native5/motion_assets.cu','void RekNative5Motion::pre(const float* actions,', '''void RekNative5Motion::reset_ordered(const uint8_t* flags, const float* base,
        RekG1CudaResetOrder order, bool preserve_dispatch_mode, cudaStream_t stream) {
    cuda_check(rek_g1_cuda_semantic_reset_ordered(scheduler,flags,base,order,
        uint8_t(preserve_dispatch_mode),count,stream), "ordered semantic reset");
}

void RekNative5Motion::pre(const float* actions,''')
print('staged ordered reset implementation')
