// Time puff_advantage variants:
//   simple  — scalar reverse scan
//   orig    — original float4/uint4 overload vec path
//   compact — current algo.cu if-constexpr adv_ld/adv_st
//
//   nvcc -O3 -std=c++17 -arch=sm_120 --use_fast_math \
//     tests/bench_puff_advantage.cu -o /tmp/bench_puff_advantage && /tmp/bench_puff_advantage

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while (0)

#ifndef PRECISION_FLOAT
using prec_t = __nv_bfloat16;
__device__ __forceinline__ float to_f(prec_t x) { return __bfloat162float(x); }
__device__ __forceinline__ prec_t from_f(float x) { return __float2bfloat16(x); }
__host__ static prec_t host_from(float x) { return __float2bfloat16(x); }
static const char* PREC = "bf16";
constexpr int W = 8;
#else
using prec_t = float;
__device__ __forceinline__ float to_f(prec_t x) { return x; }
__device__ __forceinline__ prec_t from_f(float x) { return x; }
__host__ static prec_t host_from(float x) { return x; }
static const char* PREC = "float";
constexpr int W = 4;
#endif

// ---- simple scalar ----
__global__ void adv_simple(
        const prec_t* values, const prec_t* rewards, const prec_t* dones,
        const prec_t* importance, prec_t* advantages,
        float gamma, float lambda, float rho_clip, float c_clip,
        int num_steps, int horizon) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_steps) return;
    int base = row * horizon;
    float lastlam = 0.f;
    float next_v = to_f(values[base + horizon - 1]);
    float next_d = to_f(dones[base + horizon - 1]);
    float next_r = to_f(rewards[base + horizon - 1]);
    advantages[base + horizon - 1] = from_f(0.f);
    for (int t = horizon - 2; t >= 0; t--) {
        float v = to_f(values[base + t]);
        float r = to_f(rewards[base + t]);
        float d = to_f(dones[base + t]);
        float imp = to_f(importance[base + t]);
        float nnt = 1.f - next_d;
        float rho_t = fminf(imp, rho_clip), c_t = fminf(imp, c_clip);
        float delta = rho_t * (next_r + gamma * next_v * nnt - v);
        lastlam = delta + gamma * lambda * c_t * lastlam * nnt;
        advantages[base + t] = from_f(lastlam);
        next_v = v; next_d = d; next_r = r;
    }
}

// ---- original: separate float / bf16 overloads ----
__device__ __forceinline__ void orig_ld(const float* ptr, float* out) {
    float4 v = *(const float4*)ptr;
    out[0] = v.x; out[1] = v.y; out[2] = v.z; out[3] = v.w;
}
__device__ __forceinline__ void orig_ld(const __nv_bfloat16* ptr, float* out) {
    uint4 raw = *(const uint4*)ptr;
    const __nv_bfloat16* bf = (const __nv_bfloat16*)&raw;
    #pragma unroll
    for (int i = 0; i < 8; i++) out[i] = __bfloat162float(bf[i]);
}
__device__ __forceinline__ void orig_st(float* ptr, const float* vals) {
    *(float4*)ptr = make_float4(vals[0], vals[1], vals[2], vals[3]);
}
__device__ __forceinline__ void orig_st(__nv_bfloat16* ptr, const float* vals) {
    __nv_bfloat16 tmp[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) tmp[i] = __float2bfloat16(vals[i]);
    *(uint4*)ptr = *(const uint4*)tmp;
}

__global__ void adv_orig(
        const prec_t* values, const prec_t* rewards, const prec_t* dones,
        const prec_t* importance, prec_t* advantages,
        float gamma, float lambda, float rho_clip, float c_clip,
        int num_steps, int horizon) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_steps) return;
    int offset = row * horizon;
    float lastlam = 0.f;
    float next_value = to_f(values[offset + horizon - 1]);
    float next_done = to_f(dones[offset + horizon - 1]);
    float next_reward = to_f(rewards[offset + horizon - 1]);
    int num_chunks = horizon / W;
    for (int chunk = num_chunks - 1; chunk >= 0; chunk--) {
        int base = offset + chunk * W;
        float v[W], r[W], d[W], imp[W], adv[W] = {0};
        orig_ld(values + base, v);
        orig_ld(rewards + base, r);
        orig_ld(dones + base, d);
        orig_ld(importance + base, imp);
        int start = (chunk == num_chunks - 1) ? (W - 2) : (W - 1);
        #pragma unroll
        for (int i = start; i >= 0; i--) {
            float nnt = 1.f - next_done;
            float rho_t = fminf(imp[i], rho_clip), c_t = fminf(imp[i], c_clip);
            float delta = rho_t * (next_reward + gamma * next_value * nnt - v[i]);
            lastlam = delta + gamma * lambda * c_t * lastlam * nnt;
            adv[i] = lastlam;
            next_value = v[i]; next_done = d[i]; next_reward = r[i];
        }
        orig_st(advantages + base, adv);
    }
}

// ---- compact: if-constexpr adv_ld/st (matches algo.cu) ----
__device__ __forceinline__ void compact_ld(const prec_t* p, float* o) {
    if constexpr (sizeof(prec_t) == sizeof(float)) {
        float4 v = *(const float4*)p;
        o[0] = v.x; o[1] = v.y; o[2] = v.z; o[3] = v.w;
    } else {
        uint4 u = *(const uint4*)p;
        auto* b = (const __nv_bfloat16*)&u;
        #pragma unroll
        for (int i = 0; i < 8; i++) o[i] = __bfloat162float(b[i]);
    }
}
__device__ __forceinline__ void compact_st(prec_t* p, const float* o) {
    if constexpr (sizeof(prec_t) == sizeof(float)) {
        *(float4*)p = make_float4(o[0], o[1], o[2], o[3]);
    } else {
        __nv_bfloat16 b[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) b[i] = __float2bfloat16(o[i]);
        *(uint4*)p = *(const uint4*)b;
    }
}

__global__ void adv_compact(
        const prec_t* values, const prec_t* rewards, const prec_t* dones,
        const prec_t* importance, prec_t* advantages,
        float gamma, float lambda, float rho_clip, float c_clip,
        int num_steps, int horizon) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_steps) return;
    int off = row * horizon;
    float lastlam = 0.f;
    float next_v = to_f(values[off + horizon - 1]);
    float next_d = to_f(dones[off + horizon - 1]);
    float next_r = to_f(rewards[off + horizon - 1]);
    for (int seg = horizon / W - 1; seg >= 0; seg--) {
        int base = off + seg * W;
        float v[W], r[W], d[W], imp[W], adv[W] = {};
        compact_ld(values + base, v);
        compact_ld(rewards + base, r);
        compact_ld(dones + base, d);
        compact_ld(importance + base, imp);
        int i0 = (seg + 1 == horizon / W) ? W - 2 : W - 1;
        #pragma unroll
        for (int i = i0; i >= 0; i--) {
            float nnt = 1.f - next_d;
            float rho_t = fminf(imp[i], rho_clip), c_t = fminf(imp[i], c_clip);
            float delta = rho_t * (next_r + gamma * next_v * nnt - v[i]);
            lastlam = delta + gamma * lambda * c_t * lastlam * nnt;
            adv[i] = lastlam;
            next_v = v[i]; next_d = d[i]; next_r = r[i];
        }
        compact_st(advantages + base, adv);
    }
}

enum { V_SIMPLE, V_ORIG, V_COMPACT, V_N };
static const char* VNAME[V_N] = { "simple", "orig", "compact" };

struct Bufs {
    prec_t *values, *rewards, *dones, *importance, *adv;
    int rows, horizon;
};

static void fill(prec_t* d, int n, unsigned seed) {
    std::vector<prec_t> h(n);
    for (int i = 0; i < n; i++) {
        unsigned x = seed + (unsigned)i * 2654435761u;
        h[i] = host_from(((x >> 8) & 0xffff) / 65535.f * 2.f - 1.f);
    }
    CUDA_CHECK(cudaMemcpy(d, h.data(), (size_t)n * sizeof(prec_t), cudaMemcpyHostToDevice));
}

static void launch(Bufs* b, int which) {
    int grid = (b->rows + 255) / 256;
    const float g = 0.99f, lam = 0.95f, rho = 1.f, c = 1.f;
    auto args = [&](auto* k) {
        k<<<grid, 256>>>(b->values, b->rewards, b->dones, b->importance,
            b->adv, g, lam, rho, c, b->rows, b->horizon);
    };
    if (which == V_SIMPLE) args(adv_simple);
    else if (which == V_ORIG) args(adv_orig);
    else args(adv_compact);
}

static float time_us(Bufs* b, int which, int warm, int iters) {
    for (int i = 0; i < warm; i++) launch(b, which);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t a, z;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&z));
    CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; i++) launch(b, which);
    CUDA_CHECK(cudaEventRecord(z));
    CUDA_CHECK(cudaEventSynchronize(z));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, z));
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(z));
    return ms / iters * 1000.f;
}

static float median_us(Bufs* b, int which, int warm, int iters, int reps) {
    std::vector<float> s(reps);
    for (int r = 0; r < reps; r++) s[r] = time_us(b, which, warm, iters);
    std::sort(s.begin(), s.end());
    return s[reps / 2];
}

static float max_diff(const prec_t* a, const prec_t* b, int n) {
    std::vector<prec_t> ha(n), hb(n);
    CUDA_CHECK(cudaMemcpy(ha.data(), a, (size_t)n * sizeof(prec_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.data(), b, (size_t)n * sizeof(prec_t), cudaMemcpyDeviceToHost));
    float m = 0.f;
    for (int i = 0; i < n; i++) {
#ifndef PRECISION_FLOAT
        float d = fabsf(__bfloat162float(ha[i]) - __bfloat162float(hb[i]));
#else
        float d = fabsf(ha[i] - hb[i]);
#endif
        if (d > m) m = d;
    }
    return m;
}

int main() {
    printf("precision=%s  W=%d\n\n", PREC, W);
    printf("%8s %8s %10s %10s %10s %10s %10s %10s\n",
        "rows", "H", "simple", "orig", "compact", "cmp/orig", "orig/sim", "max|d|");
    printf("------------------------------------------------------------------------------\n");

    struct Shape { int rows, horizon; };
    Shape shapes[] = {
        {4096, 32}, {8192, 32}, {4096, 64}, {4096, 128},
        {16384, 32}, {4096, 256}, {65536, 32},
    };

    for (Shape sh : shapes) {
        if (sh.horizon % W != 0) continue;
        long n = (long)sh.rows * sh.horizon;
        Bufs b{};
        b.rows = sh.rows;
        b.horizon = sh.horizon;
        size_t bytes = (size_t)n * sizeof(prec_t);
        CUDA_CHECK(cudaMalloc(&b.values, bytes));
        CUDA_CHECK(cudaMalloc(&b.rewards, bytes));
        CUDA_CHECK(cudaMalloc(&b.dones, bytes));
        CUDA_CHECK(cudaMalloc(&b.importance, bytes));
        CUDA_CHECK(cudaMalloc(&b.adv, bytes));
        prec_t *ref = NULL, *cmp = NULL;
        CUDA_CHECK(cudaMalloc(&ref, bytes));
        CUDA_CHECK(cudaMalloc(&cmp, bytes));

        fill(b.values, (int)n, 1);
        fill(b.rewards, (int)n, 2);
        {
            std::vector<prec_t> h(n);
            for (int i = 0; i < n; i++)
                h[i] = host_from((i % sh.horizon == sh.horizon - 1) ? 1.f : 0.f);
            CUDA_CHECK(cudaMemcpy(b.dones, h.data(), bytes, cudaMemcpyHostToDevice));
        }
        {
            std::vector<prec_t> h(n, host_from(1.f));
            CUDA_CHECK(cudaMemcpy(b.importance, h.data(), bytes, cudaMemcpyHostToDevice));
        }

        launch(&b, V_ORIG);
        CUDA_CHECK(cudaMemcpy(ref, b.adv, bytes, cudaMemcpyDeviceToDevice));
        launch(&b, V_COMPACT);
        CUDA_CHECK(cudaMemcpy(cmp, b.adv, bytes, cudaMemcpyDeviceToDevice));
        float diff = max_diff(ref, cmp, (int)n);

        int iters = std::max(50, 40'000'000 / std::max((int)n, 1));
        iters = std::min(iters, 1500);
        int warm = std::min(80, iters);
        float us[V_N];
        for (int v = 0; v < V_N; v++)
            us[v] = median_us(&b, v, warm, iters, 9);

        printf("%8d %8d %10.2f %10.2f %10.2f %10.3f %10.3f %10.3g%s\n",
            sh.rows, sh.horizon, us[V_SIMPLE], us[V_ORIG], us[V_COMPACT],
            us[V_COMPACT] / us[V_ORIG], us[V_ORIG] / us[V_SIMPLE], diff,
            diff > 1e-3f ? " ERR" : "");

        cudaFree(b.values); cudaFree(b.rewards); cudaFree(b.dones);
        cudaFree(b.importance); cudaFree(b.adv); cudaFree(ref); cudaFree(cmp);
    }

    printf("\nµs median of 9. cmp/orig ≈ 1 means compact matches original perf.\n");
    printf("Breakout: rows=4096 H=32.\n");
    return 0;
}
