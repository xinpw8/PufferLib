// Bench advantage mean/var variants the way train actually uses them:
//   - bf16 source (prod default) and float
//   - multi-block vs single-block
//   - bare kernels vs CUDA-graph of (mean/var + follow-on work)
//   - optional concurrent stream load (async env-like traffic)
//
// Isolated kernel µs alone under-predicted the 1-pass breakout hit (~2ms expected,
// ~60ms train). Prefer the "graph+follow" and "graph+contended" columns.
//
// Build (prod-like bf16):
//   nvcc -O3 -std=c++17 -arch=sm_120 --use_fast_math \
//     tests/bench_ppo_adv_mean.cu -o /tmp/bench_ppo_adv_mean
// Float:
//   nvcc -O3 -std=c++17 -arch=sm_120 -DPRECISION_FLOAT \
//     tests/bench_ppo_adv_mean.cu -o /tmp/bench_ppo_adv_mean_f

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
__device__ __forceinline__ float ld(const prec_t* p, int i) {
    return __bfloat162float(p[i]);
}
__host__ static prec_t host_from_float(float x) { return __float2bfloat16(x); }
static const char* PREC_NAME = "bf16";
#else
using prec_t = float;
__device__ __forceinline__ float ld(const prec_t* p, int i) { return p[i]; }
__host__ static prec_t host_from_float(float x) { return x; }
static const char* PREC_NAME = "float";
#endif

constexpr int TH = 256;
constexpr int MAX_BLOCKS = 1024;

// ---- baseline: warp+shared ppo_bsum, reduce s and q separately (prod today) ----
__device__ __forceinline__ float bsum_warp(float v) {
    __shared__ float w[TH / 32];
    int t = threadIdx.x, lane = t & 31, wid = t >> 5;
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffff, v, o);
    if (lane == 0) w[wid] = v;
    __syncthreads();
    v = (t < TH / 32) ? w[t] : 0.f;
    if (wid == 0) {
        for (int o = 4; o > 0; o >>= 1)
            v += __shfl_down_sync(0xffffffff, v, o);
    }
    return v;
}

__global__ void moments_bsum(const prec_t* src, float* partial, int n) {
    float s = 0.f, q = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
            i += blockDim.x * gridDim.x) {
        float x = ld(src, i);
        s += x;
        q += x * x;
    }
    float bs = bsum_warp(s);
    __syncthreads();
    float bq = bsum_warp(q);
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = bs;
        partial[blockIdx.x + gridDim.x] = bq;
    }
}

__global__ void finalize_bsum(const float* partial, float* var_out, float* mean_out,
        int n_blocks, int n) {
    float s = 0.f, q = 0.f;
    for (int i = threadIdx.x; i < n_blocks; i += blockDim.x) {
        s += partial[i];
        q += partial[i + n_blocks];
    }
    float bs = bsum_warp(s);
    __syncthreads();
    float bq = bsum_warp(q);
    if (threadIdx.x == 0) {
        float mean = bs / (float)n;
        *mean_out = mean;
        *var_out = fmaxf(0.f, (bq - bs * mean) / (float)(n - 1));
    }
}

// ---- multi: shared tree (s,q) together (simpler, often slower) ----
__device__ __forceinline__ void block_sum2_shared(float& s, float& q) {
    __shared__ float ss[TH], qq[TH];
    int tid = threadIdx.x;
    ss[tid] = s;
    qq[tid] = q;
    __syncthreads();
    for (int stride = TH / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ss[tid] += ss[tid + stride];
            qq[tid] += qq[tid + stride];
        }
        __syncthreads();
    }
    s = ss[0];
    q = qq[0];
}

__global__ void moments_sum2(const prec_t* src, float* partial, int n) {
    float s = 0.f, q = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
            i += blockDim.x * gridDim.x) {
        float x = ld(src, i);
        s += x;
        q += x * x;
    }
    block_sum2_shared(s, q);
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = s;
        partial[blockIdx.x + gridDim.x] = q;
    }
}

__global__ void finalize_sum2(const float* partial, float* var_out, float* mean_out,
        int n_blocks, int n) {
    float s = 0.f, q = 0.f;
    for (int i = threadIdx.x; i < n_blocks; i += blockDim.x) {
        s += partial[i];
        q += partial[i + n_blocks];
    }
    block_sum2_shared(s, q);
    if (threadIdx.x == 0) {
        float mean = s / (float)n;
        *mean_out = mean;
        *var_out = fmaxf(0.f, (q - s * mean) / (float)(n - 1));
    }
}

// ---- multi: warp bsum2 (s,q) float2 — simpler than 2× bsum, keep multi launch ----
__device__ __forceinline__ float2 bsum2_warp(float2 v) {
    __shared__ float2 w[TH / 32];
    int t = threadIdx.x, lane = t & 31, wid = t >> 5;
    for (int o = 16; o > 0; o >>= 1) {
        v.x += __shfl_down_sync(0xffffffff, v.x, o);
        v.y += __shfl_down_sync(0xffffffff, v.y, o);
    }
    if (lane == 0) w[wid] = v;
    __syncthreads();
    v = (t < TH / 32) ? w[t] : make_float2(0.f, 0.f);
    if (wid == 0) {
        for (int o = 4; o > 0; o >>= 1) {
            v.x += __shfl_down_sync(0xffffffff, v.x, o);
            v.y += __shfl_down_sync(0xffffffff, v.y, o);
        }
    }
    return v;
}

__global__ void moments_bsum2(const prec_t* src, float* partial, int n) {
    float2 acc = make_float2(0.f, 0.f);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
            i += blockDim.x * gridDim.x) {
        float x = ld(src, i);
        acc.x += x;
        acc.y += x * x;
    }
    float2 b = bsum2_warp(acc);
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = b.x;
        partial[blockIdx.x + gridDim.x] = b.y;
    }
}

__global__ void finalize_bsum2(const float* partial, float* var_out, float* mean_out,
        int n_blocks, int n) {
    float2 acc = make_float2(0.f, 0.f);
    for (int i = threadIdx.x; i < n_blocks; i += blockDim.x) {
        acc.x += partial[i];
        acc.y += partial[i + n_blocks];
    }
    float2 b = bsum2_warp(acc);
    if (threadIdx.x == 0) {
        float mean = b.x / (float)n;
        *mean_out = mean;
        *var_out = fmaxf(0.f, (b.y - b.x * mean) / (float)(n - 1));
    }
}

// ---- single-block one-pass (lost ~5% on breakout train) ----
__global__ void mean_var_1block(const prec_t* src, float* var_out, float* mean_out, int n) {
    float s = 0.f, q = 0.f;
    for (int i = threadIdx.x; i < n; i += TH) {
        float x = ld(src, i);
        s += x;
        q += x * x;
    }
    block_sum2_shared(s, q);
    if (threadIdx.x == 0) {
        float mean = s / (float)n;
        *mean_out = mean;
        *var_out = fmaxf(0.f, (q - s * mean) / (float)(n - 1));
    }
}

// Follow-on work: cheap stand-in for ppo_loss bandwidth (touch adv + write scratch).
// Sized so mean/var is a small but non-zero fraction of the graph — like train.
__global__ void follow_touch(const prec_t* src, float* out, int n, float mean, float inv_std) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = (ld(src, i) - mean) * inv_std;
        out[i] = x * x;
    }
}

// Contending stream: keep SMs busy like async GPU env.
__global__ void spam_fma(float* buf, int n, int reps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = buf[i];
    for (int r = 0; r < reps; r++) x = fmaf(x, 1.0001f, 0.001f);
    buf[i] = x;
}

enum Variant { VAR_BSUM = 0, VAR_BSUM2 = 1, VAR_SUM2 = 2, VAR_1BLOCK = 3, VAR_N = 4 };
static const char* VAR_NAME[VAR_N] = { "multi_bsum", "multi_bsum2", "multi_sum2", "1block" };

struct Ctx {
    prec_t* src;
    float* partial;
    float* var;
    float* mean;
    float* follow_out;
    float* spam;
    int n;
    int spam_n;
};

static int nblocks(int n) {
    return std::min((n + TH - 1) / TH, MAX_BLOCKS);
}

static void launch_meanvar(Variant v, Ctx* c, cudaStream_t s) {
    int B = nblocks(c->n);
    switch (v) {
    case VAR_BSUM:
        moments_bsum<<<B, TH, 0, s>>>(c->src, c->partial, c->n);
        finalize_bsum<<<1, TH, 0, s>>>(c->partial, c->var, c->mean, B, c->n);
        break;
    case VAR_BSUM2:
        moments_bsum2<<<B, TH, 0, s>>>(c->src, c->partial, c->n);
        finalize_bsum2<<<1, TH, 0, s>>>(c->partial, c->var, c->mean, B, c->n);
        break;
    case VAR_SUM2:
        moments_sum2<<<B, TH, 0, s>>>(c->src, c->partial, c->n);
        finalize_sum2<<<1, TH, 0, s>>>(c->partial, c->var, c->mean, B, c->n);
        break;
    case VAR_1BLOCK:
        mean_var_1block<<<1, TH, 0, s>>>(c->src, c->var, c->mean, c->n);
        break;
    default: break;
    }
}

static void launch_follow(Ctx* c, cudaStream_t s) {
    int g = (c->n + TH - 1) / TH;
    follow_touch<<<g, TH, 0, s>>>(c->src, c->follow_out, c->n, 0.f, 1.f);
}

static float time_ms(void (*fn)(void*), void* arg, int warm, int iters) {
    for (int i = 0; i < warm; i++) fn(arg);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; i++) fn(arg);
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    return ms / iters;
}

struct LaunchArg {
    Ctx* c;
    Variant v;
    cudaStream_t stream;
    cudaStream_t cont;
    bool with_follow;
    bool with_contend;
    cudaGraphExec_t graph_exec;  // if non-null, launch this instead
};

static void do_eager(void* p) {
    LaunchArg* a = (LaunchArg*)p;
    if (a->with_contend) {
        int g = (a->c->spam_n + TH - 1) / TH;
        spam_fma<<<g, TH, 0, a->cont>>>(a->c->spam, a->c->spam_n, 64);
    }
    launch_meanvar(a->v, a->c, a->stream);
    if (a->with_follow) launch_follow(a->c, a->stream);
}

static void do_graph(void* p) {
    LaunchArg* a = (LaunchArg*)p;
    if (a->with_contend) {
        int g = (a->c->spam_n + TH - 1) / TH;
        spam_fma<<<g, TH, 0, a->cont>>>(a->c->spam, a->c->spam_n, 64);
    }
    CUDA_CHECK(cudaGraphLaunch(a->graph_exec, a->stream));
}

static cudaGraphExec_t capture_graph(Variant v, Ctx* c, cudaStream_t stream, bool with_follow) {
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    launch_meanvar(v, c, stream);
    if (with_follow) launch_follow(c, stream);
    cudaGraph_t g;
    CUDA_CHECK(cudaStreamEndCapture(stream, &g));
    cudaGraphExec_t exec;
    CUDA_CHECK(cudaGraphInstantiate(&exec, g, 0));
    CUDA_CHECK(cudaGraphDestroy(g));
    return exec;
}

static void check_close(Variant va, Variant vb, Ctx* c) {
    float ma, va_, mb, vb_;
    launch_meanvar(va, c, 0);
    CUDA_CHECK(cudaMemcpy(&ma, c->mean, 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&va_, c->var, 4, cudaMemcpyDeviceToHost));
    launch_meanvar(vb, c, 0);
    CUDA_CHECK(cudaMemcpy(&mb, c->mean, 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&vb_, c->var, 4, cudaMemcpyDeviceToHost));
    float dm = fabsf(ma - mb), dv = fabsf(va_ - vb_);
    if (dm > 1e-3f || dv > 1e-2f) {
        printf("  WARN %s vs %s mean %g/%g var %g/%g\n",
            VAR_NAME[va], VAR_NAME[vb], ma, mb, va_, vb_);
    }
}

int main() {
    printf("precision=%s  TH=%d\n", PREC_NAME, TH);
    printf("Modes: bare = mean/var only; graph = captured mean/var+follow; "
           "contend = spam on side stream during graph launch\n\n");

    cudaStream_t stream, cont;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaStreamCreate(&cont));

    // Breakout minibatch NT=65536; also stress larger.
    int ns[] = {8192, 16384, 32768, 65536, 131072, 262144};
    int n_ns = (int)(sizeof(ns) / sizeof(ns[0]));

    printf("%8s %12s %10s %10s %10s %10s\n",
        "n", "variant", "bare_us", "graph_us", "cont_us", "vs_bsum%");
    printf("------------------------------------------------------------------------\n");

    for (int ni = 0; ni < n_ns; ni++) {
        int n = ns[ni];
        Ctx c{};
        c.n = n;
        c.spam_n = 1 << 20;
        CUDA_CHECK(cudaMalloc(&c.src, (size_t)n * sizeof(prec_t)));
        CUDA_CHECK(cudaMalloc(&c.partial, 2 * MAX_BLOCKS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.var, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.mean, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.follow_out, (size_t)n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.spam, (size_t)c.spam_n * sizeof(float)));

        std::vector<prec_t> h(n);
        for (int i = 0; i < n; i++)
            h[i] = host_from_float(0.01f * ((i * 17 % 200) - 100));
        CUDA_CHECK(cudaMemcpy(c.src, h.data(), (size_t)n * sizeof(prec_t),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(c.spam, 0, (size_t)c.spam_n * sizeof(float)));

        check_close(VAR_BSUM, VAR_BSUM2, &c);
        check_close(VAR_BSUM, VAR_SUM2, &c);
        check_close(VAR_BSUM, VAR_1BLOCK, &c);

        float bare_bsum = 0.f;
        for (int v = 0; v < VAR_N; v++) {
            LaunchArg arg{};
            arg.c = &c;
            arg.v = (Variant)v;
            arg.stream = stream;
            arg.cont = cont;

            // bare
            arg.with_follow = false;
            arg.with_contend = false;
            arg.graph_exec = NULL;
            float bare_ms = time_ms(do_eager, &arg, 50, 400);

            // graph mean/var + follow
            arg.graph_exec = capture_graph((Variant)v, &c, stream, true);
            arg.with_follow = true;
            arg.with_contend = false;
            float graph_ms = time_ms(do_graph, &arg, 50, 400);

            // contended graph
            arg.with_contend = true;
            float cont_ms = time_ms(do_graph, &arg, 50, 400);

            CUDA_CHECK(cudaGraphExecDestroy(arg.graph_exec));

            if (v == VAR_BSUM) bare_bsum = bare_ms;
            float vs = 100.f * (bare_ms - bare_bsum) / bare_bsum;

            printf("%8d %12s %10.2f %10.2f %10.2f %9.1f%%\n",
                n, VAR_NAME[v], bare_ms * 1000.f, graph_ms * 1000.f,
                cont_ms * 1000.f, vs);
        }
        printf("\n");

        cudaFree(c.src);
        cudaFree(c.partial);
        cudaFree(c.var);
        cudaFree(c.mean);
        cudaFree(c.follow_out);
        cudaFree(c.spam);
    }

    // Extrapolate breakout: 999 minibatches, n=65536
    printf("Breakout-scale (n=65536, ~999 mbs): prefer graph_us/cont_us, not bare_us alone.\n");
    printf("  multi_bsum2 ≈ multi_bsum (prod candidate). multi_sum2/1block lose at 64k+.\n");
    printf("  Full gate: ./build.sh breakout --gpu && ./puffer train breakout (uptime/SPS).\n");

    cudaStreamDestroy(stream);
    cudaStreamDestroy(cont);
    return 0;
}
