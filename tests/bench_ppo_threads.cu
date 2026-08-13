// Sweep PPO_THREADS for ppo_loss_compute (+ light VM moments sweep).
// "Size" axis A_total ∈ {32,128,1024} (single discrete head) — work/thread scales
// with action dim the way larger heads/hidden-ish fan-outs do. Also T and N.
//
// nvcc -O3 -std=c++17 -arch=sm_120 -DPRECISION_FLOAT tests/bench_ppo_threads.cu -o /tmp/bench_ppo_threads

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>

#define CUDA_CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while (0)

enum { LOSS_N = 7 };

// Minimal discrete PPO loss body matching algo.cu structure, THREADS templated.
template<int THREADS>
__global__ void ppo_loss_bench(
        float* __restrict__ ppo_partials,
        float* __restrict__ grad_logits,
        float* __restrict__ grad_values,
        const float* __restrict__ logits, // (N*T, A+1) last col value
        const float* __restrict__ actions,
        const float* __restrict__ old_logprobs,
        const float* __restrict__ advantages,
        const float* __restrict__ prio,
        const float* __restrict__ values,
        const float* __restrict__ returns,
        const float* __restrict__ mask,
        const float* __restrict__ adv_mean,
        const float* __restrict__ adv_var,
        int A, int N, int T,
        float clip_coef, float vf_clip_coef, float vf_coef, float ent_coef) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int total = N * T;
    float inv_NT = 1.0f / float(total);

    __shared__ float block_losses[LOSS_N][THREADS];
    for (int c = 0; c < LOSS_N; c++) block_losses[c][tid] = 0.0f;

    if (idx < total) {
        int n = idx / T, t = idx % T, nt = n * T + t;
        int logits_base = nt * (A + 1);
        int grad_base = nt * A;
        int mask_base = nt * A;

        float old_logp = old_logprobs[nt];
        float adv = advantages[nt];
        float w = prio[n];
        float val = values[nt];
        float ret = returns[nt];
        float val_pred = logits[logits_base + A];
        float adv_std = sqrtf(adv_var[0]);
        float adv_normalized = (adv - adv_mean[0]) / (adv_std + 1e-8f);
        float dL = inv_NT;
        float d_entropy_term = dL * (-ent_coef);

        float v_error = val_pred - val;
        float v_clipped = val + fmaxf(-vf_clip_coef, fminf(vf_clip_coef, v_error));
        float v_loss_unclipped = (val_pred - ret) * (val_pred - ret);
        float v_loss_clipped = (v_clipped - ret) * (v_clipped - ret);
        float v_loss = 0.5f * fmaxf(v_loss_unclipped, v_loss_clipped);
        float d_val = (v_loss_clipped > v_loss_unclipped) ? 0.f : (val_pred - ret);
        grad_values[nt] = dL * vf_coef * d_val;

        int act = (int)actions[nt];
        // softmax head
        float max_logit = -INFINITY, sum = 0.f, act_logit = 0.f;
        for (int a = 0; a < A; ++a) {
            float l = logits[logits_base + a];
            float m = mask[mask_base + a];
            if (m == 0.f) l = -1e4f;
            if (a == act) act_logit = l;
            if (l > max_logit) { sum *= expf(max_logit - l); max_logit = l; }
            sum += expf(l - max_logit);
        }
        float logsumexp = max_logit + logf(sum);
        float ent = 0.f;
        for (int a = 0; a < A; ++a) {
            float l = logits[logits_base + a];
            if (mask[mask_base + a] == 0.f) l = -1e4f;
            float logp = l - logsumexp;
            float p = expf(logp);
            ent -= p * logp;
        }
        float total_log_prob = act_logit - logsumexp;
        float logratio = total_log_prob - old_logp;
        float ratio = expf(logratio);
        float ratio_clipped = fmaxf(1.f - clip_coef, fminf(1.f + clip_coef, ratio));
        float wa = -w * adv_normalized;
        float pg1 = wa * ratio, pg2 = wa * ratio_clipped;
        float pg_loss = fmaxf(pg1, pg2);
        float d_ratio = wa * dL;
        if (pg2 > pg1 && (ratio <= 1.f - clip_coef || ratio >= 1.f + clip_coef))
            d_ratio = 0.f;
        float d_new_logp = d_ratio * ratio;

        for (int j = 0; j < A; ++j) {
            float l = logits[logits_base + j];
            if (mask[mask_base + j] == 0.f) l = -1e4f;
            float logp = l - logsumexp;
            float p = expf(logp);
            float d_logit = (j == act) ? d_new_logp : 0.f;
            d_logit -= p * d_new_logp;
            d_logit += d_entropy_term * p * (-ent - logp);
            grad_logits[grad_base + j] = d_logit;
        }

        float thread_loss = (pg_loss + vf_coef * v_loss - ent_coef * ent) * inv_NT;
        block_losses[0][tid] = pg_loss * inv_NT;
        block_losses[1][tid] = v_loss * inv_NT;
        block_losses[2][tid] = ent * inv_NT;
        block_losses[3][tid] = thread_loss;
        block_losses[4][tid] = (-logratio) * inv_NT;
        block_losses[5][tid] = ((ratio - 1.f) - logratio) * inv_NT;
        block_losses[6][tid] = (fabsf(ratio - 1.f) > clip_coef ? 1.f : 0.f) * inv_NT;
    }
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (int c = 0; c < LOSS_N; c++)
                block_losses[c][tid] += block_losses[c][tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        int base = blockIdx.x * LOSS_N;
        for (int c = 0; c < LOSS_N; c++)
            ppo_partials[base + c] = block_losses[c][0];
    }
}

// VM moments with templated threads
template<int TH>
__device__ __forceinline__ float bsum(float v) {
    __shared__ float w[TH / 32];
    int t = threadIdx.x, lane = t & 31, wid = t >> 5;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    if (lane == 0) w[wid] = v;
    __syncthreads();
    v = (t < TH / 32) ? w[t] : 0.f;
    if (wid == 0) {
        for (int o = TH / 64; o > 0; o >>= 1)
            v += __shfl_down_sync(0xffffffff, v, o);
    }
    return v;
}

template<int TH>
__global__ void adv_moments(const float* src, float* partial, int n) {
    float s = 0.f, q = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float x = src[i]; s += x; q += x * x;
    }
    float bs = bsum<TH>(s);
    __syncthreads();
    float bq = bsum<TH>(q);
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = bs;
        partial[blockIdx.x + gridDim.x] = bq;
    }
}

template<int TH>
__global__ void adv_finalize(const float* partial, float* var_out, float* mean_out, int n_blocks, int n) {
    float s = 0.f, q = 0.f;
    for (int i = threadIdx.x; i < n_blocks; i += blockDim.x) {
        s += partial[i]; q += partial[i + n_blocks];
    }
    float bs = bsum<TH>(s);
    __syncthreads();
    float bq = bsum<TH>(q);
    if (threadIdx.x == 0) {
        float mean = bs / (float)n;
        *mean_out = mean;
        *var_out = fmaxf(0.f, (bq - bs * mean) / (float)(n - 1));
    }
}

static float elapsed_us(void (*launch)(void*), void* ctx, int warm, int iters) {
    for (int i = 0; i < warm; i++) launch(ctx);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < iters; i++) launch(ctx);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / iters * 1000.f;
}

struct LossCtx {
    float *partials, *glog, *gval, *logits, *act, *oldlp, *adv, *prio, *val, *ret, *mask, *amean, *avar;
    int A, N, T, threads;
};

template<int TH>
static void launch_loss(void* p) {
    LossCtx* c = (LossCtx*)p;
    int total = c->N * c->T;
    int grid = (total + TH - 1) / TH;
    ppo_loss_bench<TH><<<grid, TH>>>(
        c->partials, c->glog, c->gval, c->logits, c->act, c->oldlp, c->adv, c->prio,
        c->val, c->ret, c->mask, c->amean, c->avar,
        c->A, c->N, c->T, 0.2f, 0.2f, 0.5f, 0.01f);
}

struct VmCtx { float *src, *partial, *var, *mean; int n, blocks, th; };

// Prod path: multi-block moments + finalize
template<int TH>
static void launch_vm_multi(void* p) {
    VmCtx* c = (VmCtx*)p;
    int blocks = min((c->n + TH - 1) / TH, 1024);
    c->blocks = blocks;
    adv_moments<TH><<<blocks, TH>>>(c->src, c->partial, c->n);
    adv_finalize<TH><<<1, TH>>>(c->partial, c->var, c->mean, blocks, c->n);
}

// One-pass sum+sumsq, single block (same formula as prod finalize, no partials).
template<int TH>
__global__ void adv_mean_var_1pass(const float* src, float* var_out, float* mean_out, int n) {
    float s = 0.f, q = 0.f;
    for (int i = threadIdx.x; i < n; i += TH) {
        float x = src[i];
        s += x;
        q += x * x;
    }
    float bs = bsum<TH>(s);
    __syncthreads();
    float bq = bsum<TH>(q);
    if (threadIdx.x == 0) {
        float mean = bs / (float)n;
        *mean_out = mean;
        *var_out = fmaxf(0.f, (bq - bs * mean) / (float)(n - 1));
    }
}

// Pre-76f04970 ppo_var_mean: two full passes over src (mean then centered sumsq).
template<int TH>
__global__ void adv_var_mean_2pass(const float* src, float* var_out, float* mean_out, int n) {
    __shared__ float sdata[TH];
    int tid = threadIdx.x;
    float sum = 0.f;
    for (int i = tid; i < n; i += TH) sum += src[i];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = TH / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / (float)n;
    if (tid == 0) *mean_out = mean;
    __syncthreads();
    float ss = 0.f;
    for (int i = tid; i < n; i += TH) {
        float d = src[i] - mean;
        ss += d * d;
    }
    sdata[tid] = ss;
    __syncthreads();
    for (int s = TH / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) *var_out = sdata[0] / (float)(n - 1);
}

template<int TH>
static void launch_vm_1pass(void* p) {
    VmCtx* c = (VmCtx*)p;
    adv_mean_var_1pass<TH><<<1, TH>>>(c->src, c->var, c->mean, c->n);
}

template<int TH>
static void launch_vm_2pass(void* p) {
    VmCtx* c = (VmCtx*)p;
    adv_var_mean_2pass<TH><<<1, TH>>>(c->src, c->var, c->mean, c->n);
}

static float run_loss_th(LossCtx* c, int th) {
    switch (th) {
        case 64:  return elapsed_us(launch_loss<64>, c, 15, 60);
        case 128: return elapsed_us(launch_loss<128>, c, 15, 60);
        case 256: return elapsed_us(launch_loss<256>, c, 15, 60);
        case 512: return elapsed_us(launch_loss<512>, c, 15, 60);
        case 1024:return elapsed_us(launch_loss<1024>, c, 15, 60);
        default: return -1.f;
    }
}

static float run_vm_multi(VmCtx* c, int th) {
    switch (th) {
        case 64:  return elapsed_us(launch_vm_multi<64>, c, 30, 200);
        case 128: return elapsed_us(launch_vm_multi<128>, c, 30, 200);
        case 256: return elapsed_us(launch_vm_multi<256>, c, 30, 200);
        case 512: return elapsed_us(launch_vm_multi<512>, c, 30, 200);
        case 1024:return elapsed_us(launch_vm_multi<1024>, c, 30, 200);
        default: return -1.f;
    }
}

static float run_vm_1pass(VmCtx* c, int th) {
    switch (th) {
        case 64:  return elapsed_us(launch_vm_1pass<64>, c, 30, 200);
        case 128: return elapsed_us(launch_vm_1pass<128>, c, 30, 200);
        case 256: return elapsed_us(launch_vm_1pass<256>, c, 30, 200);
        case 512: return elapsed_us(launch_vm_1pass<512>, c, 30, 200);
        case 1024:return elapsed_us(launch_vm_1pass<1024>, c, 30, 200);
        default: return -1.f;
    }
}

static float run_vm_2pass(VmCtx* c, int th) {
    switch (th) {
        case 64:  return elapsed_us(launch_vm_2pass<64>, c, 30, 200);
        case 128: return elapsed_us(launch_vm_2pass<128>, c, 30, 200);
        case 256: return elapsed_us(launch_vm_2pass<256>, c, 30, 200);
        case 512: return elapsed_us(launch_vm_2pass<512>, c, 30, 200);
        case 1024:return elapsed_us(launch_vm_2pass<1024>, c, 30, 200);
        default: return -1.f;
    }
}

static void fill_rand(float* d, size_t n, float scale) {
    std::vector<float> h(n);
    for (size_t i = 0; i < n; i++) h[i] = scale * ((int)(i * 17 % 100) - 50) * 0.02f;
    CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
}

int main() {
    int thr_list[] = {64, 128, 256, 512, 1024};

    printf("=== ppo_loss_compute thread sweep (us) ===\n");
    printf("A_total is discrete head size (work/element). N*T = minibatch steps.\n");
    printf("Prod: PPO_THREADS=256\n\n");

    // Shapes: A ∈ {32,128,1024} as requested "hidden" scale of work;
    // T ∈ {32,128,512}; N = 65536/T style mb segments (capped) and N=B-like
    struct Case { int A, N, T; };
    std::vector<Case> cases;
    for (int A : {32, 128, 1024}) {
        for (int T : {32, 128, 512}) {
            // realistic mb: ~64k steps total, and a smaller one
            int N_big = std::max(1, 65536 / T);
            int N_mid = std::max(1, 16384 / T);
            cases.push_back({A, N_mid, T});
            cases.push_back({A, N_big, T});
        }
    }

    printf("%5s %6s %5s %8s", "A", "N", "T", "NT");
    for (int th : thr_list) printf(" %8d", th);
    printf("   best\n");

    for (auto cs : cases) {
        int A = cs.A, N = cs.N, T = cs.T;
        long NT = (long)N * T;
        long logits_n = NT * (A + 1);
        long mask_n = NT * A;

        LossCtx c{};
        c.A = A; c.N = N; c.T = T;
        CUDA_CHECK(cudaMalloc(&c.partials, ((NT + 1023) / 64) * LOSS_N * sizeof(float))); // ample
        CUDA_CHECK(cudaMalloc(&c.glog, mask_n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.gval, NT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.logits, logits_n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.act, NT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.oldlp, NT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.adv, NT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.prio, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.val, NT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.ret, NT * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.mask, mask_n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.amean, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.avar, sizeof(float)));

        fill_rand(c.logits, logits_n, 1.f);
        fill_rand(c.oldlp, NT, 0.1f);
        fill_rand(c.adv, NT, 1.f);
        fill_rand(c.val, NT, 1.f);
        fill_rand(c.ret, NT, 1.f);
        // actions in [0,A)
        {
            std::vector<float> h(NT);
            for (long i = 0; i < NT; i++) h[i] = (float)(i % A);
            CUDA_CHECK(cudaMemcpy(c.act, h.data(), NT * sizeof(float), cudaMemcpyHostToDevice));
        }
        {
            std::vector<float> h(N, 1.f);
            CUDA_CHECK(cudaMemcpy(c.prio, h.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        }
        {
            std::vector<float> h(mask_n, 1.f);
            CUDA_CHECK(cudaMemcpy(c.mask, h.data(), mask_n * sizeof(float), cudaMemcpyHostToDevice));
        }
        float one = 1.f, zero = 0.f;
        CUDA_CHECK(cudaMemcpy(c.amean, &zero, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(c.avar, &one, 4, cudaMemcpyHostToDevice));

        printf("%5d %6d %5d %8ld", A, N, T, NT);
        float best = 1e30f; int best_th = 0;
        for (int th : thr_list) {
            // skip absurd grids for tiny NT
            if (NT < th / 4 && th >= 512) { printf(" %8s", "-"); continue; }
            float us = run_loss_th(&c, th);
            printf(" %8.1f", us);
            if (us > 0 && us < best) { best = us; best_th = th; }
        }
        printf("  %4d%s\n", best_th, best_th == 256 ? " *" : "");

        cudaFree(c.partials); cudaFree(c.glog); cudaFree(c.gval); cudaFree(c.logits);
        cudaFree(c.act); cudaFree(c.oldlp); cudaFree(c.adv); cudaFree(c.prio);
        cudaFree(c.val); cudaFree(c.ret); cudaFree(c.mask); cudaFree(c.amean); cudaFree(c.avar);
    }

    // Head-to-head: multi-block prod vs single-block (one-pass) vs old two-pass.
    // Warm/iters high — these are a few us; noise matters.
    printf("\n=== adv mean/var: multi vs single-block (us @ 256 threads) ===\n");
    printf("%10s %10s %10s %10s %10s %10s\n",
        "n=NT", "multi", "1pass", "2pass_old", "multi/1p", "2p/1p");
    for (int n : {2048, 8192, 16384, 32768, 65536, 131072, 262144, 1048576}) {
        VmCtx c{};
        c.n = n;
        CUDA_CHECK(cudaMalloc(&c.src, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.partial, 2 * 1024 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.var, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.mean, sizeof(float)));
        fill_rand(c.src, n, 1.f);

        float um = run_vm_multi(&c, 256);
        float u1 = run_vm_1pass(&c, 256);
        float u2 = run_vm_2pass(&c, 256);
        printf("%10d %10.2f %10.2f %10.2f %10.2f %10.2f\n",
            n, um, u1, u2, um / u1, u2 / u1);

        // Spot-check mean/var agreement (1pass vs multi)
        float hm[2], hv[2];
        launch_vm_multi<256>(&c);
        CUDA_CHECK(cudaMemcpy(&hm[0], c.mean, 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&hv[0], c.var, 4, cudaMemcpyDeviceToHost));
        launch_vm_1pass<256>(&c);
        CUDA_CHECK(cudaMemcpy(&hm[1], c.mean, 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&hv[1], c.var, 4, cudaMemcpyDeviceToHost));
        float dm = fabsf(hm[0] - hm[1]), dv = fabsf(hv[0] - hv[1]);
        if (dm > 1e-4f || dv > 1e-3f) {
            printf("  WARN multi vs 1pass mean/var: mean %g vs %g  var %g vs %g\n",
                hm[0], hm[1], hv[0], hv[1]);
        }

        cudaFree(c.src); cudaFree(c.partial); cudaFree(c.var); cudaFree(c.mean);
    }

    printf("\n=== multi-block only: thread sweep (us) ===\n");
    printf("%10s", "n=NT");
    for (int th : thr_list) printf(" %8d", th);
    printf("   best\n");
    for (int n : {8192, 65536, 262144}) {
        VmCtx c{};
        c.n = n;
        CUDA_CHECK(cudaMalloc(&c.src, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.partial, 2 * 1024 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.var, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.mean, sizeof(float)));
        fill_rand(c.src, n, 1.f);
        printf("%10d", n);
        float best = 1e30f; int best_th = 0;
        for (int th : thr_list) {
            float us = run_vm_multi(&c, th);
            printf(" %8.2f", us);
            if (us < best) { best = us; best_th = th; }
        }
        printf("  %4d%s\n", best_th, best_th == 256 ? " *" : "");
        cudaFree(c.src); cudaFree(c.partial); cudaFree(c.var); cudaFree(c.mean);
    }

    printf("\nmulti = prod ppo_adv_moments+finalize; 1pass = <<<1,256>>> sum+sumsq;\n"
           "2pass_old = pre-change ppo_var_mean (two reads of src).\n"
           "* = prod default threads.\n");
    return 0;
}
