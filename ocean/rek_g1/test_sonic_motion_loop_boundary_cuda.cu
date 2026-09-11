#define REK_G1_CUDA_DEVICE 1
#include "sonic_motion_composer_native.h"
#include "sonic_motion_composer_libm_candidate.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

struct Result { int resolve_status, f0, f1, reference_status, advance_status; float cursor, first_dof; };

__global__ void probe(Result* output) {
    const int start = 2 * threadIdx.x;
    float dofs[11 * 29] = {};
    float roots[11 * 4] = {};
    for (int frame = 0; frame < 10; ++frame) {
        roots[frame * 4] = 1.0f;
        for (int joint = 0; joint < 29; ++joint) dofs[frame * 29 + joint] = float(frame);
    }
    SonicMotionComposerNativeLayer layer = {};
    layer.clip = {dofs, roots, 290, 40, 10, 50.0f};
    layer.config = {0, 1, -0.5f, start, 9, 0.1f, 0.1f, 0.0f};
    layer.has_clip = layer.has_config = layer.active = 1;
    layer.speed = layer.per_tick = -0.5f;
    layer.start_frame = start;
    layer.end_frame = 9;
    layer.cursor = nextafterf(float(start) + 0.5f, float(start));
    SonicMotionComposerNativeResolvedFrames frames = {};
    Result result = {};
    result.resolve_status = sonic_motion_composer_native_resolve_frames(&layer, 1, &frames);
    result.f0 = frames.f0;
    result.f1 = frames.f1;
    SonicMotionComposerNative composer;
    SonicMotionComposerNativeBackends backends = {
        sonic_motion_composer_libm_candidate_quaternion_slerp,
        sonic_motion_composer_libm_candidate_atan2_f,
        sonic_motion_composer_libm_candidate_sin_cos_f, nullptr, nullptr};
    sonic_motion_composer_native_init(&composer, 50, &backends);
    composer.current_layer = layer;
    SonicMotionComposerNativeReferenceTiming timing;
    uint32_t indices[29];
    uint8_t negate[29] = {};
    for (int i = 0; i < 29; ++i) indices[i] = i;
    for (int i = 0; i < 10; ++i) timing.current_offsets[i] = timing.next_offsets[i] = 1;
    SonicMotionComposerNativeMirrorTable mirror = {indices, negate, 29, 29};
    float current[290] = {}, next[290] = {}, rotations[40] = {};
    SonicMotionComposerNativeReferenceOutput reference = {current, next, rotations, 290, 290, 40};
    result.reference_status = sonic_motion_composer_native_build_reference_rows(
        &composer, &timing, &mirror, &reference);
    result.first_dof = current[0];
    SonicMotionComposerNativeAdvanceResult advanced;
    result.advance_status = sonic_motion_composer_native_advance(&composer, &advanced);
    result.cursor = composer.current_layer.cursor;
    output[threadIdx.x] = result;
}

int main() {
    Result* device = nullptr;
    if (cudaMalloc(&device, 2 * sizeof(Result)) != cudaSuccess) return 2;
    probe<<<1, 2>>>(device);
    Result rows[2];
    if (cudaMemcpy(rows, device, sizeof(rows), cudaMemcpyDeviceToHost) != cudaSuccess) return 3;
    cudaFree(device);
    int failed = 0;
    for (int i = 0; i < 2; ++i) {
        const auto& r = rows[i];
        printf("start=%d resolve=%d frames=%d,%d reference=%d advance=%d cursor=%.9g first_dof=%.9g\n",
            2 * i, r.resolve_status, r.f0, r.f1, r.reference_status, r.advance_status, r.cursor, r.first_dof);
        if (r.resolve_status || r.reference_status || r.advance_status || r.f0 != 2 * i
                || r.f1 != 2 * i + 1 || r.cursor != float(2 * i) || r.first_dof != float(2 * i)) failed = 1;
    }
    return failed;
}
