// Microbench prio kernels + launch-config sweep.
// nvcc -O3 -std=c++17 -arch=sm_120 -DPRECISION_FLOAT tests/bench_prio.cu -o /tmp/bench_prio -lcurand

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <tuple>

#define CUDA_CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while (0)

// --- kernels parameterized by block size via template or runtime blockDim ---

__global__ void prio_adv_reduction(const float* __restrict__ advantages,
        float* prio_weights, float prio_alpha, int stride) {
    int row = blockIdx.x;
    int tx = threadIdx.x;
    int offset = row * stride;
    float local_sum = 0.0f;
    for (int t = tx; t < stride; t += blockDim.x) {
        local_sum += fabsf(advantages[offset + t]);
    }
    // warp reduce then tree across warps if block > 32
    unsigned mask = 0xffffffffu;
    for (int s = 16; s >= 1; s /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, s);
    }
    __shared__ float sm[32]; // max 32 warps for block 1024
    int lane = tx % 32;
    int wid = tx / 32;
    if (lane == 0) sm[wid] = local_sum;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < (blockDim.x + 31) / 32) ? sm[lane] : 0.0f;
        for (int s = 16; s >= 1; s /= 2) {
            v += __shfl_down_sync(mask, v, s);
        }
        if (tx == 0) {
            float pw = powf(v, prio_alpha);
            if (isnan(pw) || isinf(pw)) pw = 0.0f;
            prio_weights[row] = pw;
        }
    }
}

// Current production: only warp-level, assumes blockDim.x == 32
__global__ void prio_adv_reduction_warp_only(const float* __restrict__ advantages,
        float* prio_weights, float prio_alpha, int stride) {
    int row = blockIdx.x;
    int tx = threadIdx.x;
    int offset = row * stride;
    float local_sum = 0.0f;
    for (int t = tx; t < stride; t += blockDim.x) {
        local_sum += fabsf(advantages[offset + t]);
    }
    for (int s = 16; s >= 1; s /= 2) {
        local_sum += __shfl_down_sync(0xffffffffu, local_sum, s);
    }
    if (tx == 0) {
        float pw = powf(local_sum, prio_alpha);
        if (isnan(pw) || isinf(pw)) pw = 0.0f;
        prio_weights[row] = pw;
    }
}

__global__ void prio_normalize(float* prio_weights, int length) {
    extern __shared__ float sh[];
    float* shmem = sh;
    __shared__ float block_sum;
    int tx = threadIdx.x;
    int lane = tx % 32;
    int warp_id = tx / 32;
    int nwarps = (blockDim.x + 31) / 32;
    const float eps = 1e-6f;

    float local_sum = 0.0f;
    for (int t = tx; t < length; t += blockDim.x) {
        local_sum += prio_weights[t];
    }
    for (int s = 16; s >= 1; s /= 2) {
        local_sum += __shfl_down_sync(0xffffffffu, local_sum, s);
    }
    if (lane == 0) shmem[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane < nwarps) ? shmem[lane] : 0.0f;
        for (int s = 16; s >= 1; s /= 2) {
            val += __shfl_down_sync(0xffffffffu, val, s);
        }
        if (tx == 0) block_sum = val + eps;
    }
    __syncthreads();
    for (int t = tx; t < length; t += blockDim.x) {
        prio_weights[t] = (prio_weights[t] + eps) / block_sum;
    }
}

__global__ void build_cdf(float* __restrict__ cdf, const float* __restrict__ probs, int B) {
    extern __shared__ float smem[];
    float* warp_tot = smem;
    float* warp_exc = smem + 32;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp_id = tid / 32;
    const int nwarps = (blockDim.x + 31) / 32;
    const int chunk = (B + blockDim.x - 1) / blockDim.x;
    const int start = tid * chunk;
    const int end = min(start + chunk, B);

    float run = 0.0f;
    for (int i = start; i < end; i++) {
        run += probs[i];
        cdf[i] = run;
    }
    float my_total = (start < B) ? run : 0.0f;
    float incl = my_total;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        float n = __shfl_up_sync(0xffffffffu, incl, off);
        if (lane >= off) incl += n;
    }
    float warp_exclusive = incl - my_total;
    if (lane == 31) warp_tot[warp_id] = incl;
    __syncthreads();
    if (warp_id == 0) {
        float w = (lane < nwarps) ? warp_tot[lane] : 0.0f;
        float wins = w;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            float n = __shfl_up_sync(0xffffffffu, wins, off);
            if (lane >= off) wins += n;
        }
        if (lane < nwarps) warp_exc[lane] = wins - w;
    }
    __syncthreads();
    float exclusive = warp_exclusive + warp_exc[warp_id];
    if (exclusive != 0.0f) {
        for (int i = start; i < end; i++) cdf[i] += exclusive;
    }
}

// Serial baseline (old <<<1,1>>>)
__global__ void build_cdf_serial(float* cdf, const float* probs, int B) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float run = 0.0f;
    for (int i = 0; i < B; i++) {
        run += probs[i];
        cdf[i] = run;
    }
}

__global__ void multinomial_sample(int* __restrict__ out_idx, const float* __restrict__ cdf,
        int B, int num_samples, uint64_t seed, const int64_t* __restrict__ offset_ptr) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_samples) return;
    uint64_t base_off = (uint64_t)(*offset_ptr);
    curandStatePhilox4_32_10_t rng_state;
    curand_init(seed, base_off + tid, 0, &rng_state);
    float u = curand_uniform(&rng_state);
    int lo = 0, hi = B - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cdf[mid] < u) lo = mid + 1;
        else hi = mid;
    }
    out_idx[tid] = lo;
}

__global__ void prio_imp_weights(const int* __restrict__ indices,
        const float* __restrict__ prio_probs, float* mb_prio,
        int total_agents, float anneal_beta, int minibatch_segments) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tx < minibatch_segments) {
        float value = prio_probs[indices[tx]] * (float)total_agents;
        mb_prio[tx] = powf(value, -anneal_beta);
    }
}

static float time_launch(void (*fn)(void*), void* ctx, int warmup, int iters) {
    for (int i = 0; i < warmup; i++) fn(ctx);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; i++) fn(ctx);
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    return (ms / iters) * 1000.0f; // us
}

struct AdvCtx { float *adv, *pw; int B, T; float alpha; int threads; bool warp_only; };
static void run_adv(void* p) {
    AdvCtx* c = (AdvCtx*)p;
    if (c->warp_only) {
        prio_adv_reduction_warp_only<<<c->B, c->threads>>>(c->adv, c->pw, c->alpha, c->T);
    } else {
        prio_adv_reduction<<<c->B, c->threads>>>(c->adv, c->pw, c->alpha, c->T);
    }
}

struct NormCtx { float* pw; int B, threads; };
static void run_norm(void* p) {
    NormCtx* c = (NormCtx*)p;
    prio_normalize<<<1, c->threads, 32 * sizeof(float)>>>(c->pw, c->B);
}

struct CdfCtx { float *cdf, *pw; int B, threads; bool serial; };
static void run_cdf(void* p) {
    CdfCtx* c = (CdfCtx*)p;
    if (c->serial) build_cdf_serial<<<1, 1>>>(c->cdf, c->pw, c->B);
    else build_cdf<<<1, c->threads, 64 * sizeof(float)>>>(c->cdf, c->pw, c->B);
}

struct SampCtx { int *idx; float *cdf; int64_t *off; int B, N, threads; };
static void run_samp(void* p) {
    SampCtx* c = (SampCtx*)p;
    int blocks = (c->N + c->threads - 1) / c->threads;
    multinomial_sample<<<blocks, c->threads>>>(c->idx, c->cdf, c->B, c->N, 12345ull, c->off);
}

struct ImpCtx { int *idx; float *pw, *mb; int B, N, threads; };
static void run_imp(void* p) {
    ImpCtx* c = (ImpCtx*)p;
    int blocks = (c->N + c->threads - 1) / c->threads;
    prio_imp_weights<<<blocks, c->threads>>>(c->idx, c->pw, c->mb, c->B, 0.5f, c->N);
}

struct PipeCtx {
    float *adv, *pw, *cdf, *mb; int *idx; int64_t *off;
    int B, T, N, block, adv_threads; bool warp_only;
};
static void run_pipe(void* p) {
    PipeCtx* c = (PipeCtx*)p;
    if (c->warp_only) {
        prio_adv_reduction_warp_only<<<c->B, c->adv_threads>>>(c->adv, c->pw, 0.5f, c->T);
    } else {
        prio_adv_reduction<<<c->B, c->adv_threads>>>(c->adv, c->pw, 0.5f, c->T);
    }
    prio_normalize<<<1, c->block, 32 * sizeof(float)>>>(c->pw, c->B);
    build_cdf<<<1, c->block, 64 * sizeof(float)>>>(c->cdf, c->pw, c->B);
    int blocks = (c->N + c->block - 1) / c->block;
    multinomial_sample<<<blocks, c->block>>>(c->idx, c->cdf, c->B, c->N, 12345ull, c->off);
    prio_imp_weights<<<blocks, c->block>>>(c->idx, c->pw, c->mb, c->B, 0.5f, c->N);
}

int main() {
    const int warmup = 30, iters = 200;
    printf("GPU microbench: prio kernels (us/call). Current prod: adv<<<B,32>>>, rest block=256\n\n");

    // ---- 1) adv reduction: threads vs T ----
    printf("=== compute_prio_adv_reduction  B=8192  (us) ===\n");
    printf("%6s", "T\\thr");
    int thr_list[] = {32, 64, 128, 256};
    for (int th : thr_list) printf(" %8d", th);
    printf("   warp32*\n");
    for (int T : {32, 64, 128, 256, 512}) {
        int B = 8192;
        float *d_adv, *d_pw;
        CUDA_CHECK(cudaMalloc(&d_adv, (size_t)B * T * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pw, B * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_adv, 0, (size_t)B * T * sizeof(float)));
        // non-zero pattern
        std::vector<float> h(B * T);
        for (size_t i = 0; i < h.size(); i++) h[i] = 0.01f * ((int)i % 17 - 8);
        CUDA_CHECK(cudaMemcpy(d_adv, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));

        printf("%6d", T);
        for (int th : thr_list) {
            AdvCtx c{d_adv, d_pw, B, T, 0.5f, th, false};
            float us = time_launch(run_adv, &c, warmup, iters);
            printf(" %8.2f", us);
        }
        AdvCtx cw{d_adv, d_pw, B, T, 0.5f, 32, true};
        printf(" %8.2f\n", time_launch(run_adv, &cw, warmup, iters));
        CUDA_CHECK(cudaFree(d_adv));
        CUDA_CHECK(cudaFree(d_pw));
    }
    printf("  *warp32 = production-style kernel (no multi-warp reduce)\n\n");

    // ---- 2) normalize + build_cdf block size vs B ----
    printf("=== normalize + build_cdf  (us)  block in {128,256,512} ===\n");
    printf("%8s %10s %10s %10s %10s %10s %10s %10s\n",
        "B", "norm128", "norm256", "norm512", "cdf128", "cdf256", "cdf512", "cdf_serial");
    for (int B : {1024, 4096, 8192, 16384, 32768, 65536}) {
        float *d_pw, *d_cdf;
        CUDA_CHECK(cudaMalloc(&d_pw, B * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cdf, B * sizeof(float)));
        std::vector<float> h(B);
        double s = 0;
        for (int i = 0; i < B; i++) { h[i] = 0.1f + 0.01f * (i % 10); s += h[i]; }
        for (int i = 0; i < B; i++) h[i] = (float)(h[i] / s);
        CUDA_CHECK(cudaMemcpy(d_pw, h.data(), B * sizeof(float), cudaMemcpyHostToDevice));

        printf("%8d", B);
        for (int th : {128, 256, 512}) {
            // fresh copy for normalize (mutates)
            float *tmp; CUDA_CHECK(cudaMalloc(&tmp, B * sizeof(float)));
            auto time_norm = [&](int thr) {
                CUDA_CHECK(cudaMemcpy(tmp, h.data(), B * sizeof(float), cudaMemcpyHostToDevice));
                NormCtx c{tmp, B, thr};
                // one timed path that includes the memcpy cost is wrong — memcpy once then time kernel only on already-normalized-like data
                return 0.f;
            };
            (void)time_norm;
            CUDA_CHECK(cudaFree(tmp));
            NormCtx c{d_pw, B, th};
            // restore probs each time? normalize is idempotent-ish after first call messes magnitudes
            CUDA_CHECK(cudaMemcpy(d_pw, h.data(), B * sizeof(float), cudaMemcpyHostToDevice));
            // warm one call
            prio_normalize<<<1, th, 32 * sizeof(float)>>>(d_pw, B);
            CUDA_CHECK(cudaDeviceSynchronize());
            // re-seed then time: for fair kernel time use fixed data each launch
            float usn = 0;
            {
                for (int w = 0; w < warmup; w++) {
                    CUDA_CHECK(cudaMemcpy(d_pw, h.data(), B * sizeof(float), cudaMemcpyHostToDevice));
                    prio_normalize<<<1, th, 32 * sizeof(float)>>>(d_pw, B);
                }
                CUDA_CHECK(cudaDeviceSynchronize());
                cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
                cudaEventRecord(a);
                for (int i = 0; i < iters; i++) {
                    // kernel-only: run on already-normalized values after first (distribution changes)
                    prio_normalize<<<1, th, 32 * sizeof(float)>>>(d_pw, B);
                }
                cudaEventRecord(b); cudaEventSynchronize(b);
                float ms; cudaEventElapsedTime(&ms, a, b);
                usn = (ms / iters) * 1000.f;
                cudaEventDestroy(a); cudaEventDestroy(b);
            }
            printf(" %10.2f", usn);
        }
        CUDA_CHECK(cudaMemcpy(d_pw, h.data(), B * sizeof(float), cudaMemcpyHostToDevice));
        for (int th : {128, 256, 512}) {
            CdfCtx c{d_cdf, d_pw, B, th, false};
            printf(" %10.2f", time_launch(run_cdf, &c, warmup, iters));
        }
        CdfCtx cs{d_cdf, d_pw, B, 1, true};
        printf(" %10.2f\n", time_launch(run_cdf, &cs, warmup, iters / 4 + 1));
        CUDA_CHECK(cudaFree(d_pw));
        CUDA_CHECK(cudaFree(d_cdf));
    }

    // ---- 3) sample + imp weights vs N and block ----
    printf("\n=== multinomial_sample + imp_weights  B=8192  (us) ===\n");
    printf("%8s %10s %10s %10s %10s %10s %10s\n",
        "N", "samp128", "samp256", "samp512", "imp128", "imp256", "imp512");
    {
        int B = 8192;
        float *d_cdf, *d_pw, *d_mb;
        int *d_idx;
        int64_t *d_off;
        CUDA_CHECK(cudaMalloc(&d_cdf, B * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pw, B * sizeof(float)));
        std::vector<float> h(B);
        for (int i = 0; i < B; i++) h[i] = (i + 1.0f) / B; // already a cdf-ish
        CUDA_CHECK(cudaMemcpy(d_cdf, h.data(), B * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pw, h.data(), B * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_off, sizeof(int64_t)));
        CUDA_CHECK(cudaMemset(d_off, 0, sizeof(int64_t)));

        for (int N : {256, 1024, 2048, 4096, 8192, 16384}) {
            CUDA_CHECK(cudaMalloc(&d_idx, N * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_mb, N * sizeof(float)));
            printf("%8d", N);
            for (int th : {128, 256, 512}) {
                SampCtx c{d_idx, d_cdf, d_off, B, N, th};
                printf(" %10.2f", time_launch(run_samp, &c, warmup, iters));
            }
            // need valid idx for imp
            multinomial_sample<<<(N + 255) / 256, 256>>>(d_idx, d_cdf, B, N, 1, d_off);
            CUDA_CHECK(cudaDeviceSynchronize());
            for (int th : {128, 256, 512}) {
                ImpCtx c{d_idx, d_pw, d_mb, B, N, th};
                printf(" %10.2f", time_launch(run_imp, &c, warmup, iters));
            }
            printf("\n");
            CUDA_CHECK(cudaFree(d_idx));
            CUDA_CHECK(cudaFree(d_mb));
        }
        CUDA_CHECK(cudaFree(d_cdf));
        CUDA_CHECK(cudaFree(d_pw));
        CUDA_CHECK(cudaFree(d_off));
    }

    // ---- 4) full pipeline: prod vs variants ----
    printf("\n=== full prio pipeline once (build+sample) us ===\n");
    printf("%6s %6s %6s | %10s %10s %10s %10s\n",
        "B", "T", "N", "prod", "adv256", "blk128", "blk512");
    // prod: adv thr=32 warp_only, block=256
    struct Cfg { int adv_th; int block; bool warp_only; const char* name; };
    for (auto dims : {std::tuple<int,int,int>{4096, 32, 2048},
                      {8192, 64, 2048},
                      {16384, 128, 4096},
                      {32768, 32, 2048},
                      {8192, 512, 1024}}) {
        int B, T, N;
        std::tie(B, T, N) = dims;
        float *adv, *pw, *cdf, *mb; int *idx; int64_t *off;
        CUDA_CHECK(cudaMalloc(&adv, (size_t)B * T * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&pw, B * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&cdf, B * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&mb, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&idx, N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&off, sizeof(int64_t)));
        CUDA_CHECK(cudaMemset(off, 0, sizeof(int64_t)));
        std::vector<float> h((size_t)B * T);
        for (size_t i = 0; i < h.size(); i++) h[i] = 0.01f * ((int)i % 13 - 6);
        CUDA_CHECK(cudaMemcpy(adv, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));

        printf("%6d %6d %6d |", B, T, N);
        Cfg cfgs[] = {
            {32, 256, true, "prod"},
            {256, 256, false, "adv256"},
            {32, 128, true, "blk128"},
            {32, 512, true, "blk512"},
        };
        for (auto& cfg : cfgs) {
            PipeCtx c{adv, pw, cdf, mb, idx, off, B, T, N, cfg.block, cfg.adv_th, cfg.warp_only};
            printf(" %10.2f", time_launch(run_pipe, &c, warmup, iters));
        }
        printf("\n");
        CUDA_CHECK(cudaFree(adv)); CUDA_CHECK(cudaFree(pw)); CUDA_CHECK(cudaFree(cdf));
        CUDA_CHECK(cudaFree(mb)); CUDA_CHECK(cudaFree(idx)); CUDA_CHECK(cudaFree(off));
    }

    // ---- 5) share of train from existing nsys if present ----
    printf("\nDone.\n");
    return 0;
}
