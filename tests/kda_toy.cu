// Minimal Kimi Delta Attention (KDA) recurrent toy vs MinGRU seq scan.
//
// Paper: arXiv:2510.26692  (KDA recurrence only — no ShortConv / MLA hybrid / MoE)
//
//   S_t = Diag(α_t) (I − β_t k_t k_t^T) S_{t−1} + β_t k_t v_t^T
//   o_t = S_t^T q_t
//
// Single head, d_k = d_v = D. State is D×D per sequence (vs MinGRU's D vector).
//
// Build:
//   nvcc -O3 --use_fast_math -arch=native -std=c++17 tests/kda_toy.cu -o kda_toy
// Run:
//   ./kda_toy              # H=64, fixed B*T=65536, T sweep
//   ./kda_toy 128          # H=128
//   ./kda_toy 64 32 2048   # H T B explicit

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA %s: %s\n", what, cudaGetErrorString(e));
        exit(1);
    }
}

// ---- device helpers ----

__device__ __forceinline__ float d_sigmoid(float x) {
    float z = expf(-fabsf(x));
    return x >= 0.0f ? 1.0f / (1.0f + z) : z / (1.0f + z);
}

__device__ __forceinline__ float d_lerp(float a, float b, float w) {
    float diff = b - a;
    return (fabsf(w) < 0.5f) ? a + w * diff : b - diff * (1.0f - w);
}

// ---- MinGRU seq scan (linear-h, highway, no terminals) ----
// One thread per (b, h). combined layout (B, T, 3H): [h_tilde, gate, proj] along last dim.

__global__ void mingru_scan_fwd(
        float* __restrict__ out,
        float* __restrict__ next_state,
        const float* __restrict__ combined,
        const float* __restrict__ state0,
        const float* __restrict__ x_in,
        int B, int T, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) {
        return;
    }
    int b = idx / H;
    int h = idx % H;
    int H3 = 3 * H;
    float h_t = state0[b * H + h];
    const float* cbase = combined + (long)b * T * H3 + h;
    const float* xbase = x_in + (long)b * T * H + h;
    float* obase = out + (long)b * T * H + h;

    for (int t = 0; t < T; t++) {
        float hidden = cbase[t * H3];
        float gate = cbase[t * H3 + H];
        float proj = cbase[t * H3 + 2 * H];
        float x = xbase[t * H];
        float z = d_sigmoid(gate);
        float h_tilde = (hidden >= 0.0f) ? hidden + 0.5f : d_sigmoid(hidden);
        h_t = d_lerp(h_t, h_tilde, z);
        float s = d_sigmoid(proj);
        obase[t * H] = s * h_t + (1.0f - s) * x;
    }
    next_state[b * H + h] = h_t;
}

// ---- KDA recurrent scan ----
// One block per sequence. Threads = min(D, 256) cooperate on S ∈ R^{D×D} in shared mem.
// Layout: q,k,v,alpha (B,T,D); beta (B,T); out (B,T,D). S starts at 0.

__global__ void kda_scan_fwd(
        float* __restrict__ out,
        float* __restrict__ S_final,  // optional (B, D, D); may be null
        const float* __restrict__ q,
        const float* __restrict__ k,
        const float* __restrict__ v,
        const float* __restrict__ alpha,
        const float* __restrict__ beta,
        int B, int T, int D) {
    int b = blockIdx.x;
    if (b >= B) {
        return;
    }

    extern __shared__ float sm[];
    float* S = sm;                 // D*D
    float* u = S + D * D;          // D  = k^T S
    float* k_s = u + D;
    float* v_s = k_s + D;
    float* a_s = v_s + D;
    float* q_s = a_s + D;

    for (int i = threadIdx.x; i < D * D; i += blockDim.x) {
        S[i] = 0.0f;
    }
    __syncthreads();

    for (int t = 0; t < T; t++) {
        long base = ((long)b * T + t) * D;
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            k_s[i] = k[base + i];
            v_s[i] = v[base + i];
            // Paper: α ∈ (0,1). Host stores pre-sigmoid logits → sigmoid here.
            a_s[i] = d_sigmoid(alpha[base + i]);
            q_s[i] = q[base + i];
        }
        float bt = d_sigmoid(beta[b * T + t]);
        __syncthreads();

        // u_j = sum_i k_i S_{i j}
        for (int j = threadIdx.x; j < D; j += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < D; i++) {
                sum += k_s[i] * S[i * D + j];
            }
            u[j] = sum;
        }
        __syncthreads();

        // S_{ij} ← α_i (S_{ij} − β k_i u_j) + β k_i v_j
        for (int idx = threadIdx.x; idx < D * D; idx += blockDim.x) {
            int i = idx / D;
            int j = idx % D;
            float s = S[idx] - bt * k_s[i] * u[j];
            S[idx] = a_s[i] * s + bt * k_s[i] * v_s[j];
        }
        __syncthreads();

        // o_j = sum_i S_{i j} q_i
        for (int j = threadIdx.x; j < D; j += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < D; i++) {
                sum += S[i * D + j] * q_s[i];
            }
            out[base + j] = sum;
        }
        __syncthreads();
    }

    if (S_final) {
        float* dst = S_final + (long)b * D * D;
        for (int i = threadIdx.x; i < D * D; i += blockDim.x) {
            dst[i] = S[i];
        }
    }
}

// ---- timing / host ----

struct Bench {
    int B, T, H;
    // MinGRU
    float *mg_combined, *mg_state, *mg_x, *mg_out, *mg_ns;
    // KDA
    float *kda_q, *kda_k, *kda_v, *kda_alpha, *kda_beta, *kda_out, *kda_S;
};

static float* dalloc(size_t n) {
    float* p = NULL;
    ck(cudaMalloc(&p, n * sizeof(float)), "malloc");
    return p;
}

static void fill_randn(float* d, size_t n, float scale) {
    float* h = (float*)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        // crude but fine for microbench
        h[i] = scale * ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f);
    }
    ck(cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice), "H2D");
    free(h);
}

static Bench make_bench(int B, int T, int H) {
    Bench b = {};
    b.B = B;
    b.T = T;
    b.H = H;
    size_t BT = (size_t)B * T;
    size_t BTH = BT * H;

    b.mg_combined = dalloc(BTH * 3);
    b.mg_state = dalloc((size_t)B * H);
    b.mg_x = dalloc(BTH);
    b.mg_out = dalloc(BTH);
    b.mg_ns = dalloc((size_t)B * H);

    b.kda_q = dalloc(BTH);
    b.kda_k = dalloc(BTH);
    b.kda_v = dalloc(BTH);
    b.kda_alpha = dalloc(BTH);
    b.kda_beta = dalloc(BT);
    b.kda_out = dalloc(BTH);
    b.kda_S = dalloc((size_t)B * H * H);

    fill_randn(b.mg_combined, BTH * 3, 2.0f);
    fill_randn(b.mg_state, (size_t)B * H, 0.5f);
    // state non-negative-ish like mingru
    {
        float* h = (float*)malloc((size_t)B * H * sizeof(float));
        ck(cudaMemcpy(h, b.mg_state, (size_t)B * H * sizeof(float), cudaMemcpyDeviceToHost), "D2H");
        for (size_t i = 0; i < (size_t)B * H; i++) {
            h[i] = fabsf(h[i]) + 0.05f;
        }
        ck(cudaMemcpy(b.mg_state, h, (size_t)B * H * sizeof(float), cudaMemcpyHostToDevice), "H2D");
        free(h);
    }
    fill_randn(b.mg_x, BTH, 1.0f);

    fill_randn(b.kda_q, BTH, 0.5f);
    fill_randn(b.kda_k, BTH, 0.5f);
    fill_randn(b.kda_v, BTH, 0.5f);
    fill_randn(b.kda_alpha, BTH, 1.0f);  // logits → sigmoid in kernel
    fill_randn(b.kda_beta, BT, 1.0f);
    return b;
}

static void free_bench(Bench& b) {
    cudaFree(b.mg_combined);
    cudaFree(b.mg_state);
    cudaFree(b.mg_x);
    cudaFree(b.mg_out);
    cudaFree(b.mg_ns);
    cudaFree(b.kda_q);
    cudaFree(b.kda_k);
    cudaFree(b.kda_v);
    cudaFree(b.kda_alpha);
    cudaFree(b.kda_beta);
    cudaFree(b.kda_out);
    cudaFree(b.kda_S);
}

static void launch_mingru(Bench& b) {
    int n = b.B * b.H;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    mingru_scan_fwd<<<blocks, threads>>>(
        b.mg_out, b.mg_ns, b.mg_combined, b.mg_state, b.mg_x, b.B, b.T, b.H);
}

static void launch_kda(Bench& b) {
    int D = b.H;
    int thr = (D < 256) ? D : 256;
    // S + u + k,v,a,q
    size_t smem = (size_t)(D * D + 5 * D) * sizeof(float);
    kda_scan_fwd<<<b.B, thr, smem>>>(
        b.kda_out, b.kda_S,
        b.kda_q, b.kda_k, b.kda_v, b.kda_alpha, b.kda_beta,
        b.B, b.T, D);
}

static float time_ms(void (*fn)(Bench&), Bench& b, int warmup, int iters) {
    for (int i = 0; i < warmup; i++) {
        fn(b);
    }
    ck(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        fn(b);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / (float)iters;
}

// Tiny CPU reference for KDA (one sequence, small D,T) — smoke correctness.
static void kda_cpu_ref(
        float* out, const float* q, const float* k, const float* v,
        const float* alpha, const float* beta, int T, int D) {
    float* S = (float*)calloc((size_t)D * D, sizeof(float));
    auto sig = [](float x) {
        return 1.0f / (1.0f + expf(-x));
    };
    for (int t = 0; t < T; t++) {
        const float* qt = q + t * D;
        const float* kt = k + t * D;
        const float* vt = v + t * D;
        const float* at = alpha + t * D;
        float bt = sig(beta[t]);
        float* u = (float*)malloc(D * sizeof(float));
        for (int j = 0; j < D; j++) {
            float sum = 0;
            for (int i = 0; i < D; i++) {
                sum += kt[i] * S[i * D + j];
            }
            u[j] = sum;
        }
        for (int i = 0; i < D; i++) {
            float ai = sig(at[i]);
            for (int j = 0; j < D; j++) {
                float s = S[i * D + j] - bt * kt[i] * u[j];
                S[i * D + j] = ai * s + bt * kt[i] * vt[j];
            }
        }
        free(u);
        for (int j = 0; j < D; j++) {
            float sum = 0;
            for (int i = 0; i < D; i++) {
                sum += S[i * D + j] * qt[i];
            }
            out[t * D + j] = sum;
        }
    }
    free(S);
}

static void check_kda_correctness() {
    const int B = 1, T = 5, D = 8;
    Bench b = make_bench(B, T, D);
    // copy inputs to host
    size_t n = (size_t)T * D;
    float *hq = (float*)malloc(n * sizeof(float));
    float *hk = (float*)malloc(n * sizeof(float));
    float *hv = (float*)malloc(n * sizeof(float));
    float *ha = (float*)malloc(n * sizeof(float));
    float *hb = (float*)malloc(T * sizeof(float));
    float *href = (float*)malloc(n * sizeof(float));
    float *hgpu = (float*)malloc(n * sizeof(float));
    cudaMemcpy(hq, b.kda_q, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hk, b.kda_k, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hv, b.kda_v, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(ha, b.kda_alpha, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hb, b.kda_beta, T * sizeof(float), cudaMemcpyDeviceToHost);

    kda_cpu_ref(href, hq, hk, hv, ha, hb, T, D);
    launch_kda(b);
    ck(cudaDeviceSynchronize(), "kda check");
    cudaMemcpy(hgpu, b.kda_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    float max_abs = 0, max_rel = 0;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(hgpu[i] - href[i]);
        max_abs = std::max(max_abs, d);
        float denom = std::max(1e-3f, fabsf(href[i]));
        max_rel = std::max(max_rel, d / denom);
    }
    printf("correctness (T=%d D=%d): max_abs=%.3e max_rel=%.3e %s\n",
        T, D, max_abs, max_rel, (max_abs < 1e-4f || max_rel < 1e-4f) ? "OK" : "CHECK");
    free(hq);
    free(hk);
    free(hv);
    free(ha);
    free(hb);
    free(href);
    free(hgpu);
    free_bench(b);
}

static void print_bytes(const char* name, double bytes, float ms) {
    double gbps = bytes / (ms * 1e-3) / 1e9;
    printf("    %-18s traffic≈%.1f MB  %.1f GB/s (naive bound)\n",
        name, bytes / 1e6, gbps);
}

static void run_case(int B, int T, int H, int warmup, int iters) {
    Bench b = make_bench(B, T, H);
    // touch once for alloc
    launch_mingru(b);
    launch_kda(b);
    ck(cudaDeviceSynchronize(), "touch");
    ck(cudaGetLastError(), "launch");

    float mg_ms = time_ms(launch_mingru, b, warmup, iters);
    float kda_ms = time_ms(launch_kda, b, warmup, iters);

    size_t BT = (size_t)B * T;
    size_t BTH = BT * H;
    // rough read/write estimates (f32)
    double mg_bytes =
        (double)(BTH * 3 + B * H + BTH) * 4 +  // combined + state + x
        (double)(BTH + B * H) * 4;             // out + next
    double kda_bytes =
        (double)(BTH * 4 + BT) * 4 +           // q k v alpha beta
        (double)BTH * 4 +                      // out
        (double)B * H * H * 4;                 // S_final write (and smem not in DRAM)

    printf("T=%4d  B=%5d  H=%3d  tokens=%7zu\n", T, B, H, BT * (size_t)H);
    printf("  MinGRU seq          %7.1f us   (%.2fx vs KDA)\n",
        mg_ms * 1000.0, kda_ms / std::max(mg_ms, 1e-9f));
    printf("  KDA recurrent       %7.1f us   (%.2fx vs MinGRU)\n",
        kda_ms * 1000.0, mg_ms / std::max(kda_ms, 1e-9f));
    print_bytes("MinGRU I/O est", mg_bytes, mg_ms);
    print_bytes("KDA I/O est", kda_bytes, kda_ms);
    printf("  state sizes: MinGRU %zu B  |  KDA S %zu B  (×%.1f)\n",
        (size_t)B * H * 4,
        (size_t)B * H * H * 4,
        (double)H);
    printf("\n");

    // finiteness check on a few outs
    float sample[8];
    cudaMemcpy(sample, b.kda_out, sizeof(sample), cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int i = 0; i < 8; i++) {
        if (!isfinite(sample[i])) {
            bad++;
        }
    }
    if (bad) {
        printf("  WARN: non-finite KDA outputs in sample\n");
    }

    free_bench(b);
}

int main(int argc, char** argv) {
    srand(42);
    int H = 64;
    int fixed_BT = 65536;
    int warmup = 50;
    int iters = 200;

    if (argc >= 2) {
        H = atoi(argv[1]);
    }

    printf("KDA toy vs MinGRU seq  (f32, single-head KDA, D=H)\n");
    printf("device: ");
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("%s\n", prop.name);
    }
    printf("shared mem need @H: %.1f KB (S + vecs)\n\n",
        (H * H + 5.0 * H) * 4.0 / 1024.0);

    check_kda_correctness();
    printf("\n");

    if (argc >= 4) {
        int T = atoi(argv[2]);
        int B = atoi(argv[3]);
        run_case(B, T, H, warmup, iters);
        return 0;
    }

    // Fixed token mass B*T = 65536 (breakout-train-like)
    const int Ts[] = {16, 32, 64, 128, 256, 512};
    printf("=== fixed B*T=%d  H=%d ===\n\n", fixed_BT, H);
    for (int T : Ts) {
        if (fixed_BT % T != 0) {
            continue;
        }
        int B = fixed_BT / T;
        // shared mem limit: D*D*4 + 5*D*4 must fit
        size_t smem = (size_t)(H * H + 5 * H) * 4;
        if (smem > 48 * 1024) {
            // still try; 5090 has more dynamic shared
        }
        run_case(B, T, H, warmup, iters);
    }

    // Also fixed B (many short parallel envs) at H
    printf("=== fixed B=8192 (rollout-like agents)  H=%d ===\n\n", H);
    for (int T : {8, 16, 32, 64}) {
        run_case(8192, T, H, warmup, iters);
    }

    printf("Notes:\n");
    printf("  - KDA state is D×D per sequence; MinGRU is D. Ratio = D.\n");
    printf("  - This KDA kernel is a faithful recurrent toy (not FLA chunk kernel).\n");
    printf("  - No ShortConv / output gate / MLA hybrid from the full paper.\n");
    printf("  - Times are kernel-only; a real policy still pays QKV-like projs for KDA\n");
    printf("    (MinGRU already includes its 3H combined as input traffic here).\n");
    return 0;
}
