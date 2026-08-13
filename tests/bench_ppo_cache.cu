// Cache vs re-read discrete PPO head: is caching always faster for A <= 512?
// Single head, float. Matches loss structure: lse → entropy → (fake d_logp) → grads.
//
// nvcc -O3 -std=c++17 -arch=sm_120 tests/bench_ppo_cache.cu -o /tmp/bench_ppo_cache

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "%s\n", cudaGetErrorString(e)); exit(1); \
    } \
} while (0)

// ---- no cache: 3 global passes ----
__global__ void head_nocache(
        const float* __restrict__ logits, // (NT, A)
        const int* __restrict__ acts,
        float* __restrict__ glog,
        float d_nlp, float d_ent, int A, int NT) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NT) return;
    const float* row = logits + (size_t)i * A;
    int act = acts[i];
    float* gout = glog + (size_t)i * A;

    float maxl = -INFINITY, sum = 0.f, al = 0.f;
    for (int a = 0; a < A; a++) {
        float l = row[a];
        if (a == act) al = l;
        if (l > maxl) { sum *= expf(maxl - l); maxl = l; }
        sum += expf(l - maxl);
    }
    float lse = maxl + logf(sum);

    float ent = 0.f;
    for (int a = 0; a < A; a++) {
        float lp = row[a] - lse, p = expf(lp);
        ent -= p * lp;
    }
    (void)al;

    for (int j = 0; j < A; j++) {
        float lp = row[j] - lse, p = expf(lp);
        float d = (j == act) ? d_nlp : 0.f;
        d -= p * d_nlp;
        d += d_ent * p * (-ent - lp);
        gout[j] = d;
    }
}

// ---- cache: 1 global pass fill, then entropy+grad from registers/stack ----
// Dynamic shared: each thread gets A floats in shared if A*blockDim fits, else stack VLA not allowed —
// use fixed max 512 on stack (user: no env > 512).
__global__ void head_cache(
        const float* __restrict__ logits,
        const int* __restrict__ acts,
        float* __restrict__ glog,
        float d_nlp, float d_ent, int A, int NT) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NT) return;
    const float* row = logits + (size_t)i * A;
    int act = acts[i];
    float* gout = glog + (size_t)i * A;

    float cache[512]; // A <= 512
    float maxl = -INFINITY, sum = 0.f, al = 0.f;
    for (int a = 0; a < A; a++) {
        float l = row[a];
        cache[a] = l;
        if (a == act) al = l;
        if (l > maxl) { sum *= expf(maxl - l); maxl = l; }
        sum += expf(l - maxl);
    }
    float lse = maxl + logf(sum);
    (void)al;

    float ent = 0.f;
    for (int a = 0; a < A; a++) {
        float lp = cache[a] - lse, p = expf(lp);
        ent -= p * lp;
    }

    for (int j = 0; j < A; j++) {
        float lp = cache[j] - lse, p = expf(lp);
        float d = (j == act) ? d_nlp : 0.f;
        d -= p * d_nlp;
        d += d_ent * p * (-ent - lp);
        gout[j] = d;
    }
}

// Cache only for entropy+grad after lse still reading global once for lse only
// (same as "always cache" above)

static float time_us(void (*fn)(const float*, const int*, float*, float, float, int, int),
        const float* logits, const int* acts, float* glog,
        float d_nlp, float d_ent, int A, int NT, int threads, int warm, int iters) {
    int grid = (NT + threads - 1) / threads;
    for (int i = 0; i < warm; i++)
        fn<<<grid, threads>>>(logits, acts, glog, d_nlp, d_ent, A, NT);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < iters; i++)
        fn<<<grid, threads>>>(logits, acts, glog, d_nlp, d_ent, A, NT);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / iters * 1000.f;
}

// wrapper types for kernel launch
static void launch_nc(const float* l, const int* a, float* g, float dn, float de, int A, int NT) {
    // unused - we use direct templates
    (void)l;(void)a;(void)g;(void)dn;(void)de;(void)A;(void)NT;
}

int main() {
    const int threads = 256;
    const float d_nlp = 0.01f, d_ent = -0.001f;
    const int warm = 20, iters = 100;

    int As[] = {3, 8, 16, 32, 64, 128, 256, 512};
    int NTs[] = {1024, 4096, 8192, 16384, 65536, 262144};

    printf("Discrete head: lse + entropy + grad. cache = 1 global pass + stack; nocache = 3 global passes.\n");
    printf("threads=%d  (us/call). ratio = cache/nocache (<1 means cache faster)\n\n", threads);

    printf("%6s", "A\\NT");
    for (int nt : NTs) printf(" %10d", nt);
    printf("\n");

    for (int A : As) {
        printf("%6d", A);
        for (int NT : NTs) {
            float *logits, *glog;
            int *acts;
            CUDA_CHECK(cudaMalloc(&logits, (size_t)NT * A * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&glog, (size_t)NT * A * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&acts, NT * sizeof(int)));
            std::vector<float> hl((size_t)NT * A);
            std::vector<int> ha(NT);
            for (size_t i = 0; i < hl.size(); i++) hl[i] = 0.01f * ((int)(i % 97) - 48);
            for (int i = 0; i < NT; i++) ha[i] = i % A;
            CUDA_CHECK(cudaMemcpy(logits, hl.data(), hl.size() * 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(acts, ha.data(), NT * 4, cudaMemcpyHostToDevice));

            auto t_nc = [&]() {
                int grid = (NT + threads - 1) / threads;
                for (int i = 0; i < warm; i++)
                    head_nocache<<<grid, threads>>>(logits, acts, glog, d_nlp, d_ent, A, NT);
                CUDA_CHECK(cudaDeviceSynchronize());
                cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
                cudaEventRecord(a);
                for (int i = 0; i < iters; i++)
                    head_nocache<<<grid, threads>>>(logits, acts, glog, d_nlp, d_ent, A, NT);
                cudaEventRecord(b); cudaEventSynchronize(b);
                float ms; cudaEventElapsedTime(&ms, a, b);
                cudaEventDestroy(a); cudaEventDestroy(b);
                return ms / iters * 1000.f;
            };
            auto t_c = [&]() {
                int grid = (NT + threads - 1) / threads;
                for (int i = 0; i < warm; i++)
                    head_cache<<<grid, threads>>>(logits, acts, glog, d_nlp, d_ent, A, NT);
                CUDA_CHECK(cudaDeviceSynchronize());
                cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
                cudaEventRecord(a);
                for (int i = 0; i < iters; i++)
                    head_cache<<<grid, threads>>>(logits, acts, glog, d_nlp, d_ent, A, NT);
                cudaEventRecord(b); cudaEventSynchronize(b);
                float ms; cudaEventElapsedTime(&ms, a, b);
                cudaEventDestroy(a); cudaEventDestroy(b);
                return ms / iters * 1000.f;
            };

            float unc = t_nc();
            float uc = t_c();
            // print ratio only to keep table readable; also absolute cache us
            printf(" %6.1f/%4.2f", uc, uc / unc);

            CUDA_CHECK(cudaFree(logits));
            CUDA_CHECK(cudaFree(glog));
            CUDA_CHECK(cudaFree(acts));
        }
        printf("\n");
    }
    printf("\nEach cell: cache_us / (cache/nocache). ratio<1 => cache faster.\n");

    // Correctness: max abs err cache vs nocache
    printf("\n=== Correctness (max |cache - nocache|) ===\n");
    {
        int A = 128, NT = 4096;
        float *logits, *g0, *g1;
        int *acts;
        CUDA_CHECK(cudaMalloc(&logits, (size_t)NT * A * 4));
        CUDA_CHECK(cudaMalloc(&g0, (size_t)NT * A * 4));
        CUDA_CHECK(cudaMalloc(&g1, (size_t)NT * A * 4));
        CUDA_CHECK(cudaMalloc(&acts, NT * 4));
        std::vector<float> hl((size_t)NT * A);
        std::vector<int> ha(NT);
        for (size_t i = 0; i < hl.size(); i++) hl[i] = 0.01f * ((int)(i % 97) - 48);
        for (int i = 0; i < NT; i++) ha[i] = i % A;
        CUDA_CHECK(cudaMemcpy(logits, hl.data(), hl.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(acts, ha.data(), NT * 4, cudaMemcpyHostToDevice));
        int grid = (NT + threads - 1) / threads;
        head_nocache<<<grid, threads>>>(logits, acts, g0, d_nlp, d_ent, A, NT);
        head_cache<<<grid, threads>>>(logits, acts, g1, d_nlp, d_ent, A, NT);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> h0((size_t)NT * A), h1((size_t)NT * A);
        CUDA_CHECK(cudaMemcpy(h0.data(), g0, h0.size() * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h1.data(), g1, h1.size() * 4, cudaMemcpyDeviceToHost));
        double maxe = 0;
        for (size_t i = 0; i < h0.size(); i++) maxe = fmax(maxe, fabs(h0[i] - h1[i]));
        printf("A=%d NT=%d max_abs_err=%g %s\n", A, NT, maxe, maxe < 1e-5 ? "ok" : "FAIL");
        cudaFree(logits); cudaFree(g0); cudaFree(g1); cudaFree(acts);
    }
    return 0;
}
