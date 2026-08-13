// Correctness + timing: fused prio kernels vs reference unfused.
// Shapes: B=8k..32k, T=32..512, N=B/T (mb rows).
// nvcc -O3 -std=c++17 -arch=sm_120 -DPRECISION_FLOAT tests/test_prio_fuse.cu -o /tmp/test_prio_fuse -lcurand

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>

#define PRIO_BLOCK_SIZE 512
#define CUDA_CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while (0)

// ---- reference (unfused) ----
__global__ void ref_reduce(const float* adv, float* pw, float alpha, int stride) {
    int row = blockIdx.x, tx = threadIdx.x, off = row * stride;
    float s = 0.f;
    for (int t = tx; t < stride; t += blockDim.x) s += fabsf(adv[off + t]);
    for (int k = 16; k >= 1; k /= 2) s += __shfl_down_sync(0xffffffff, s, k);
    if (tx == 0) {
        float pwv = powf(s, alpha);
        if (isnan(pwv) || isinf(pwv)) pwv = 0.f;
        pw[row] = pwv;
    }
}

__global__ void ref_normalize(float* pw, int B) {
    __shared__ float shmem[PRIO_BLOCK_SIZE / 32];
    __shared__ float block_sum;
    int tid = threadIdx.x, lane = tid % 32, wid = tid / 32;
    float s = 0.f;
    for (int t = tid; t < B; t += blockDim.x) s += pw[t];
    for (int k = 16; k >= 1; k /= 2) s += __shfl_down_sync(0xffffffff, s, k);
    if (lane == 0) shmem[wid] = s;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < PRIO_BLOCK_SIZE / 32) ? shmem[lane] : 0.f;
        for (int k = PRIO_BLOCK_SIZE / 64; k >= 1; k /= 2)
            v += __shfl_down_sync(0xffffffff, v, k);
        if (tid == 0) block_sum = v + 1e-6f;
    }
    __syncthreads();
    for (int t = tid; t < B; t += blockDim.x)
        pw[t] = (pw[t] + 1e-6f) / block_sum;
}

__global__ void ref_cdf(float* cdf, const float* probs, int B) {
    __shared__ float warp_tot[PRIO_BLOCK_SIZE / 32];
    __shared__ float warp_exc[PRIO_BLOCK_SIZE / 32];
    int tid = threadIdx.x, lane = tid % 32, wid = tid / 32;
    int chunk = (B + blockDim.x - 1) / blockDim.x;
    int start = tid * chunk, end = min(start + chunk, B);
    float run = 0.f;
    for (int i = start; i < end; i++) { run += probs[i]; cdf[i] = run; }
    float my = (start < B) ? run : 0.f;
    float incl = my;
    for (int off = 1; off < 32; off <<= 1) {
        float n = __shfl_up_sync(0xffffffff, incl, off);
        if (lane >= off) incl += n;
    }
    float wexc = incl - my;
    if (lane == 31) warp_tot[wid] = incl;
    __syncthreads();
    if (wid == 0) {
        float w = (lane < PRIO_BLOCK_SIZE / 32) ? warp_tot[lane] : 0.f;
        float wins = w;
        for (int off = 1; off < 32; off <<= 1) {
            float n = __shfl_up_sync(0xffffffff, wins, off);
            if (lane >= off) wins += n;
        }
        if (lane < PRIO_BLOCK_SIZE / 32) warp_exc[lane] = wins - w;
    }
    __syncthreads();
    float ex = wexc + warp_exc[wid];
    if (ex != 0.f) for (int i = start; i < end; i++) cdf[i] += ex;
}

__global__ void ref_multi(int* idx, const float* cdf, int B, int N,
        uint64_t seed, const int64_t* off) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    uint64_t base = (uint64_t)(*off);
    curandStatePhilox4_32_10_t st;
    curand_init(seed, base + tid, 0, &st);
    float u = curand_uniform(&st);
    int lo = 0, hi = B - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cdf[mid] < u) lo = mid + 1; else hi = mid;
    }
    idx[tid] = lo;
}

__global__ void ref_imp(const int* idx, const float* pw, float* mb,
        int B, float beta, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        float v = pw[idx[tid]] * (float)B;
        mb[tid] = powf(v, -beta);
    }
}

__global__ void ref_adv(int64_t* off, int64_t d) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *off += d;
}

// ---- fused (mirror algo.cu) ----
__global__ void fuse_normalize_cdf(float* pw, float* cdf, int B) {
    __shared__ float shmem[PRIO_BLOCK_SIZE / 32];
    __shared__ float block_sum;
    __shared__ float warp_tot[PRIO_BLOCK_SIZE / 32];
    __shared__ float warp_exc[PRIO_BLOCK_SIZE / 32];
    int tid = threadIdx.x, lane = tid % 32, wid = tid / 32;
    float s = 0.f;
    for (int t = tid; t < B; t += blockDim.x) s += pw[t];
    for (int k = 16; k >= 1; k /= 2) s += __shfl_down_sync(0xffffffff, s, k);
    if (lane == 0) shmem[wid] = s;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < PRIO_BLOCK_SIZE / 32) ? shmem[lane] : 0.f;
        for (int k = PRIO_BLOCK_SIZE / 64; k >= 1; k /= 2)
            v += __shfl_down_sync(0xffffffff, v, k);
        if (tid == 0) block_sum = v + 1e-6f;
    }
    __syncthreads();
    for (int t = tid; t < B; t += blockDim.x)
        pw[t] = (pw[t] + 1e-6f) / block_sum;
    __syncthreads();

    int chunk = (B + blockDim.x - 1) / blockDim.x;
    int start = tid * chunk, end = min(start + chunk, B);
    float run = 0.f;
    for (int i = start; i < end; i++) { run += pw[i]; cdf[i] = run; }
    float my = (start < B) ? run : 0.f;
    float incl = my;
    for (int off = 1; off < 32; off <<= 1) {
        float n = __shfl_up_sync(0xffffffff, incl, off);
        if (lane >= off) incl += n;
    }
    float wexc = incl - my;
    if (lane == 31) warp_tot[wid] = incl;
    __syncthreads();
    if (wid == 0) {
        float w = (lane < PRIO_BLOCK_SIZE / 32) ? warp_tot[lane] : 0.f;
        float wins = w;
        for (int off = 1; off < 32; off <<= 1) {
            float n = __shfl_up_sync(0xffffffff, wins, off);
            if (lane >= off) wins += n;
        }
        if (lane < PRIO_BLOCK_SIZE / 32) warp_exc[lane] = wins - w;
    }
    __syncthreads();
    float ex = wexc + warp_exc[wid];
    if (ex != 0.f) for (int i = start; i < end; i++) cdf[i] += ex;
}

__global__ void fuse_sample(int* idx, float* mb, const float* cdf, const float* pw,
        int B, int N, float beta, uint64_t seed, const int64_t* off) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    uint64_t base = (uint64_t)(*off);
    curandStatePhilox4_32_10_t st;
    curand_init(seed, base + tid, 0, &st);
    float u = curand_uniform(&st);
    int lo = 0, hi = B - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cdf[mid] < u) lo = mid + 1; else hi = mid;
    }
    idx[tid] = lo;
    mb[tid] = powf(pw[lo] * (float)B, -beta);
}

static bool close_f(const float* a, const float* b, int n, float rtol = 1e-5f, float atol = 1e-6f) {
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > atol + rtol * fmaxf(fabsf(a[i]), fabsf(b[i]))) {
            printf("  mismatch i=%d a=%g b=%g diff=%g\n", i, a[i], b[i], d);
            return false;
        }
    }
    return true;
}

static bool eq_i(const int* a, const int* b, int n) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            printf("  idx mismatch i=%d a=%d b=%d\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

static float time_us(void (*fn)(void*), void* ctx, int warm, int iters) {
    for (int i = 0; i < warm; i++) fn(ctx);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < iters; i++) fn(ctx);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / iters * 1000.f;
}

struct Pipe {
    float *adv, *pw_r, *pw_f, *cdf_r, *cdf_f, *mb_r, *mb_f;
    int *idx_r, *idx_f;
    int64_t *off_r, *off_f;
    int B, T, N;
    float alpha, beta;
    uint64_t seed;
};

static void run_ref_build(void* p) {
    Pipe* c = (Pipe*)p;
    ref_reduce<<<c->B, 32>>>(c->adv, c->pw_r, c->alpha, c->T);
    ref_normalize<<<1, PRIO_BLOCK_SIZE>>>(c->pw_r, c->B);
    ref_cdf<<<1, PRIO_BLOCK_SIZE>>>(c->cdf_r, c->pw_r, c->B);
}
static void run_fuse_build(void* p) {
    Pipe* c = (Pipe*)p;
    ref_reduce<<<c->B, 32>>>(c->adv, c->pw_f, c->alpha, c->T);
    fuse_normalize_cdf<<<1, PRIO_BLOCK_SIZE>>>(c->pw_f, c->cdf_f, c->B);
}
static void run_ref_sample(void* p) {
    Pipe* c = (Pipe*)p;
    int blocks = (c->N + PRIO_BLOCK_SIZE - 1) / PRIO_BLOCK_SIZE;
    ref_multi<<<blocks, PRIO_BLOCK_SIZE>>>(c->idx_r, c->cdf_r, c->B, c->N, c->seed, c->off_r);
    ref_adv<<<1, 1>>>(c->off_r, c->N);
    ref_imp<<<blocks, PRIO_BLOCK_SIZE>>>(c->idx_r, c->pw_r, c->mb_r, c->B, c->beta, c->N);
}
static void run_fuse_sample(void* p) {
    Pipe* c = (Pipe*)p;
    int blocks = (c->N + PRIO_BLOCK_SIZE - 1) / PRIO_BLOCK_SIZE;
    fuse_sample<<<blocks, PRIO_BLOCK_SIZE>>>(c->idx_f, c->mb_f, c->cdf_f, c->pw_f,
        c->B, c->N, c->beta, c->seed, c->off_f);
    ref_adv<<<1, 1>>>(c->off_f, c->N);
}

int main() {
    const float alpha = 0.5f, beta = 0.6f;
    const uint64_t seed = 0xC0FFEEULL;
    int fail = 0, pass = 0;

    printf("=== Correctness: fused vs unfused ===\n");
    printf("%6s %5s %6s  %s\n", "B", "T", "N=B/T", "result");

    int Bs[] = {8192, 16384, 32768};
    int Ts[] = {32, 64, 128, 256, 512};
    // also "hidden" doesn't enter prio — user said hidden 32-1024 but prio only sees B,T,N.
    // Exercise extra N = B/T as specified; skip T that don't divide B cleanly use max(1,B/T).

    for (int B : Bs) {
        for (int T : Ts) {
            int N = B / T; // mb rows
            if (N < 1) continue;

            size_t adv_n = (size_t)B * T;
            Pipe c{};
            c.B = B; c.T = T; c.N = N; c.alpha = alpha; c.beta = beta; c.seed = seed;
            CUDA_CHECK(cudaMalloc(&c.adv, adv_n * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&c.pw_r, B * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&c.pw_f, B * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&c.cdf_r, B * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&c.cdf_f, B * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&c.mb_r, N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&c.mb_f, N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&c.idx_r, N * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&c.idx_f, N * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&c.off_r, sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&c.off_f, sizeof(int64_t)));

            std::vector<float> hadv(adv_n);
            for (size_t i = 0; i < adv_n; i++)
                hadv[i] = 0.01f * ((int)((i * 17 + T * 3) % 101) - 50);
            CUDA_CHECK(cudaMemcpy(c.adv, hadv.data(), adv_n * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(c.off_r, 0, sizeof(int64_t)));
            CUDA_CHECK(cudaMemset(c.off_f, 0, sizeof(int64_t)));

            // fresh unnormalized weights path: reduce into both
            ref_reduce<<<B, 32>>>(c.adv, c.pw_r, alpha, T);
            ref_reduce<<<B, 32>>>(c.adv, c.pw_f, alpha, T);
            CUDA_CHECK(cudaDeviceSynchronize());

            ref_normalize<<<1, PRIO_BLOCK_SIZE>>>(c.pw_r, B);
            ref_cdf<<<1, PRIO_BLOCK_SIZE>>>(c.cdf_r, c.pw_r, B);
            fuse_normalize_cdf<<<1, PRIO_BLOCK_SIZE>>>(c.pw_f, c.cdf_f, B);
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<float> pw_r(B), pw_f(B), cdf_r(B), cdf_f(B);
            CUDA_CHECK(cudaMemcpy(pw_r.data(), c.pw_r, B * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(pw_f.data(), c.pw_f, B * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(cdf_r.data(), c.cdf_r, B * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(cdf_f.data(), c.cdf_f, B * sizeof(float), cudaMemcpyDeviceToHost));

            bool ok_build = close_f(pw_r.data(), pw_f.data(), B)
                && close_f(cdf_r.data(), cdf_f.data(), B, 1e-4f, 1e-5f);

            // sample from matched cdf/pw — use ref buffers for both paths' inputs
            // to isolate sample fusion (same cdf)
            CUDA_CHECK(cudaMemcpy(c.cdf_f, c.cdf_r, B * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(c.pw_f, c.pw_r, B * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemset(c.off_r, 0, sizeof(int64_t)));
            CUDA_CHECK(cudaMemset(c.off_f, 0, sizeof(int64_t)));

            int blocks = (N + PRIO_BLOCK_SIZE - 1) / PRIO_BLOCK_SIZE;
            ref_multi<<<blocks, PRIO_BLOCK_SIZE>>>(c.idx_r, c.cdf_r, B, N, seed, c.off_r);
            ref_imp<<<blocks, PRIO_BLOCK_SIZE>>>(c.idx_r, c.pw_r, c.mb_r, B, beta, N);
            ref_adv<<<1, 1>>>(c.off_r, N);

            fuse_sample<<<blocks, PRIO_BLOCK_SIZE>>>(c.idx_f, c.mb_f, c.cdf_f, c.pw_f,
                B, N, beta, seed, c.off_f);
            ref_adv<<<1, 1>>>(c.off_f, N);
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<int> ir(N), ifr(N);
            std::vector<float> mr(N), mf(N);
            int64_t orr, ofr;
            CUDA_CHECK(cudaMemcpy(ir.data(), c.idx_r, N * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(ifr.data(), c.idx_f, N * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(mr.data(), c.mb_r, N * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(mf.data(), c.mb_f, N * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&orr, c.off_r, sizeof(int64_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&ofr, c.off_f, sizeof(int64_t), cudaMemcpyDeviceToHost));

            bool ok_samp = eq_i(ir.data(), ifr.data(), N)
                && close_f(mr.data(), mf.data(), N)
                && orr == ofr && orr == (int64_t)N;

            bool ok = ok_build && ok_samp;
            printf("%6d %5d %6d  %s%s%s\n", B, T, N,
                ok ? "ok" : "FAIL",
                ok_build ? "" : " [build]",
                ok_samp ? "" : " [sample]");
            if (ok) pass++; else fail++;

            CUDA_CHECK(cudaFree(c.adv)); CUDA_CHECK(cudaFree(c.pw_r)); CUDA_CHECK(cudaFree(c.pw_f));
            CUDA_CHECK(cudaFree(c.cdf_r)); CUDA_CHECK(cudaFree(c.cdf_f));
            CUDA_CHECK(cudaFree(c.mb_r)); CUDA_CHECK(cudaFree(c.mb_f));
            CUDA_CHECK(cudaFree(c.idx_r)); CUDA_CHECK(cudaFree(c.idx_f));
            CUDA_CHECK(cudaFree(c.off_r)); CUDA_CHECK(cudaFree(c.off_f));
        }
    }

    printf("\n=== Timing full build+sample (us)  N=B/T ===\n");
    printf("%6s %5s %6s %10s %10s %8s\n", "B", "T", "N", "ref", "fused", "ratio");
    for (int B : Bs) {
        for (int T : {32, 128, 512}) {
            int N = B / T;
            Pipe c{};
            c.B = B; c.T = T; c.N = N; c.alpha = alpha; c.beta = beta; c.seed = seed;
            size_t adv_n = (size_t)B * T;
            CUDA_CHECK(cudaMalloc(&c.adv, adv_n * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&c.pw_r, B * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&c.pw_f, B * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&c.cdf_r, B * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&c.cdf_f, B * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&c.mb_r, N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&c.mb_f, N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&c.idx_r, N * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&c.idx_f, N * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&c.off_r, sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&c.off_f, sizeof(int64_t)));
            std::vector<float> hadv(adv_n, 0.1f);
            CUDA_CHECK(cudaMemcpy(c.adv, hadv.data(), adv_n * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(c.off_r, 0, 8));
            CUDA_CHECK(cudaMemset(c.off_f, 0, 8));

            auto full_ref = [](void* p) {
                run_ref_build(p);
                // need normalize output for sample — run_ref_build does it
                Pipe* c = (Pipe*)p;
                // reset offset each time for fair sample timing
                CUDA_CHECK(cudaMemset(c->off_r, 0, 8));
                run_ref_sample(p);
            };
            auto full_fuse = [](void* p) {
                run_fuse_build(p);
                Pipe* c = (Pipe*)p;
                CUDA_CHECK(cudaMemset(c->off_f, 0, 8));
                run_fuse_sample(p);
            };

            // seed pw from reduce once for timing stability of sample half
            float uref = time_us(full_ref, &c, 20, 80);
            float ufus = time_us(full_fuse, &c, 20, 80);
            printf("%6d %5d %6d %10.2f %10.2f %8.3f\n",
                B, T, N, uref, ufus, ufus / uref);

            CUDA_CHECK(cudaFree(c.adv)); CUDA_CHECK(cudaFree(c.pw_r)); CUDA_CHECK(cudaFree(c.pw_f));
            CUDA_CHECK(cudaFree(c.cdf_r)); CUDA_CHECK(cudaFree(c.cdf_f));
            CUDA_CHECK(cudaFree(c.mb_r)); CUDA_CHECK(cudaFree(c.mb_f));
            CUDA_CHECK(cudaFree(c.idx_r)); CUDA_CHECK(cudaFree(c.idx_f));
            CUDA_CHECK(cudaFree(c.off_r)); CUDA_CHECK(cudaFree(c.off_f));
        }
    }

    printf("\npass=%d fail=%d\n", pass, fail);
    return fail ? 1 : 0;
}
