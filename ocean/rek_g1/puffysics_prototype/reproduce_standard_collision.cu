// SPDX-License-Identifier: MIT
// Standalone raw-operand replay adapted from replay_collision_query.cpp/.cu.
// Requires only these engine headers; no model, controller, learner or assets.
// The failing finite-cylinder path is a LOCAL extension of pinned Puffysics.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
#define B3_ART_CONTACTS 0
#include "engine/puffysics.cuh"

// Exact float32 operands captured by the native first-failure recorder.
// Each row: body XYZ, body quaternion XYZW, local XYZ, local quaternion XYZW,
// radius, half XYZ. Cylinder long axis is local Y. Do not renormalize inputs.
static const float query[36] = {
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    -1.6546316146850586f, -1.6546279191970825f, 1.0f,
    0.0f, 0.0f, 0.9238793253898621f, 0.3826839327812195f,
    0.8399999737739563f, 0.8399999737739563f, 0.20000000298023224f, 1.0f,
    -1.2501157522201538f, -1.7214422225952148f, 0.6942744255065918f,
    -0.4436926543712616f, 0.2849937081336975f, -0.4024416208267212f, 0.7483022809028625f,
    0.004355692304670811f, -0.00041464349487796426f, -0.000618116173427552f,
    -0.4985445439815521f, -0.14256155490875244f, 0.8378573060035706f, 0.170659601688385f,
    0.029999999329447746f, 0.0f, 0.02500000037252903f, 0.0f,
};

static B3_HD void unpack(const float* p, int type, B3Body* body, B3Shape* shape) {
    *body = {}; *shape = {};
    body->position = b3_v(p[0], p[1], p[2]);
    body->rotation = b3_q(p[3], p[4], p[5], p[6]);
    shape->type = type;
    shape->local_pos = b3_v(p[7], p[8], p[9]);
    shape->local_rot = b3_q(p[10], p[11], p[12], p[13]);
    shape->radius = p[14];
    shape->half = b3_v(p[15], p[16], p[17]);
}

static B3_HD void collide(const float* data, B3Mani* result) {
    B3Body a, b; B3Shape sa, sb;
    unpack(data, B3_BOX, &a, &sa);
    unpack(data + 18, B3_CYLINDER, &b, &sb);
    b3_collide_pair(result, &a, &sa, &b, &sb);
}

#ifdef __CUDACC__
__global__ void replay_kernel(const float* data, B3Mani* result) { collide(data, result); }
static void check(cudaError_t error) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        std::exit(2);
    }
}
#endif

int main(int argc, char** argv) {
    int expected = -1;
    if (argc == 3 && std::strcmp(argv[1], "--expect-status") == 0) {
        char* end = nullptr;
        long value = std::strtol(argv[2], &end, 10);
        if (!end || *end || value < 0 || value > 255) return 2;
        expected = int(value);
    } else if (argc != 1) {
        std::fprintf(stderr, "usage: %s [--expect-status INTEGER]\n", argv[0]);
        return 2;
    }
    B3Mani result{};
#ifdef __CUDACC__
    float* device_data = nullptr; B3Mani* device_result = nullptr;
    check(cudaMalloc(&device_data, sizeof(query)));
    check(cudaMalloc(&device_result, sizeof(result)));
    check(cudaMemcpy(device_data, query, sizeof(query), cudaMemcpyHostToDevice));
    replay_kernel<<<1, 1>>>(device_data, device_result);
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
    check(cudaMemcpy(&result, device_result, sizeof(result), cudaMemcpyDeviceToHost));
    check(cudaFree(device_data)); check(cudaFree(device_result));
    const char* backend = "cuda";
#else
    collide(query, &result);
    const char* backend = "cpu";
#endif
    std::printf("{\"backend\":\"%s\",\"status\":%u,\"count\":%d,"
                "\"gjk_iterations\":%d,\"epa_iterations\":%d,\"separation\":",
        backend, result.status, result.count, result.gjk_iterations, result.epa_iterations);
    if (result.count) std::printf("%.9g", result.sep[0]); else std::printf("null");
    std::printf(",\"normal\":[%.9g,%.9g,%.9g],\"expected_status\":",
        result.normal.x, result.normal.y, result.normal.z);
    if (expected >= 0) std::printf("%d", expected); else std::printf("null");
    std::printf("}\n");
    if (expected >= 0) return result.status == unsigned(expected) ? 0 : 1;
    return result.status != 0;
}
