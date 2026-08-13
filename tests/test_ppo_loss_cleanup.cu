// PPO loss cleanup validation:
//  - run-to-run determinism (bitwise)
//  - numerical error vs double reference (and vs legacy float path)
//  - timing grid: A∈{3,8,16}, N*T mb steps, T∈{32,128,512}
//
// Legacy = triple-scan discrete / double continuous (pre-cleanup).
// nvcc -O3 -std=c++17 -arch=sm_120 -DPRECISION_FLOAT -DNUM_ATNS=1 \
//   tests/test_ppo_loss_cleanup.cu -o /tmp/test_ppo_loss

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>

#ifndef NUM_ATNS
#define NUM_ATNS 1
#endif
#define PPO_THREADS 256
#define PPO_LOGIT_CACHE 16
#define LOSS_N 7

#define CUDA_CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while (0)

// ---- shared device helpers matching cleaned algo.cu ----
__device__ __forceinline__ float load_m(
        const float* logits, int lb, int off, int a, const float* mask, int mb) {
    float l = logits[lb + off + a];
    if (mask[mb + off + a] == 0.f) l = -1e4f;
    return l;
}

__device__ float lse_f(const float* logits, int lb, int off, int A,
        const float* mask, int mb, float* cache, int act, float* act_l) {
    float maxl = -INFINITY, sum = 0.f, al = 0.f;
    for (int a = 0; a < A; a++) {
        float l = load_m(logits, lb, off, a, mask, mb);
        if (cache) cache[a] = l;
        if (a == act) al = l;
        if (l > maxl) { sum *= expf(maxl - l); maxl = l; }
        sum += expf(l - maxl);
    }
    if (act_l) *act_l = al;
    return maxl + logf(sum);
}

__device__ float ent_f(const float* logits, int lb, int off, int A,
        const float* mask, int mb, const float* cache, float lse) {
    float ent = 0.f;
    for (int a = 0; a < A; a++) {
        float l = cache ? cache[a] : load_m(logits, lb, off, a, mask, mb);
        float lp = l - lse, p = expf(lp);
        ent -= p * lp;
    }
    return ent;
}

__device__ void grad_f(const float* logits, int lb, int off, int A, int act,
        const float* mask, int mb, const float* cache, float lse, float ent,
        float d_nlp, float d_ent, float* grad) {
    for (int j = 0; j < A; j++) {
        float l = cache ? cache[j] : load_m(logits, lb, off, j, mask, mb);
        float lp = l - lse, p = expf(lp);
        float d = (j == act) ? d_nlp : 0.f;
        d -= p * d_nlp;
        d += d_ent * p * (-ent - lp);
        grad[j] = d;
    }
}

// ---- NEW path (cache + continuous stash) ----
__global__ void ppo_new(
        float* partials, float* glog, float* gval,
        const float* logits, const float* actions, const float* oldlp,
        const float* adv, const float* prio, const float* values, const float* returns,
        const float* mask, const float* amean, const float* avar,
        int A, int N, int T, float clip, float vfclip, float vfcoef, float entcoef) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int total = N * T;
    float inv = 1.f / float(total);
    __shared__ float bl[LOSS_N][PPO_THREADS];
    for (int c = 0; c < LOSS_N; c++) bl[c][tid] = 0.f;
    if (idx < total) {
        int n = idx / T, t = idx % T, nt = n * T + t;
        int lb = nt * (A + 1), gb = nt * A, mb = nt * A;
        float old_logp = oldlp[nt], ad = adv[nt], w = prio[n];
        float val = values[nt], ret = returns[nt], vp = logits[lb + A];
        float adv_n = (ad - amean[0]) / (sqrtf(avar[0]) + 1e-8f);
        float dL = inv, d_ent = dL * (-entcoef);
        float ve = vp - val;
        float vc = val + fmaxf(-vfclip, fminf(vfclip, ve));
        float vu = (vp - ret) * (vp - ret), vcc = (vc - ret) * (vc - ret);
        float vloss = 0.5f * fmaxf(vu, vcc);
        float dvp = 0.f;
        if (vcc > vu) { if (ve >= -vfclip && ve <= vfclip) dvp = vc - ret; }
        else dvp = vp - ret;
        gval[nt] = dL * vfcoef * dvp;

        int act = (int)actions[nt];
        float cache[PPO_LOGIT_CACHE];
        float* cp = (A <= PPO_LOGIT_CACHE) ? cache : NULL;
        float al = 0.f;
        float lse = lse_f(logits, lb, 0, A, mask, mb, cp, act, &al);
        float ent = ent_f(logits, lb, 0, A, mask, mb, cp, lse);
        float tlp = al - lse;
        float logratio = tlp - old_logp, ratio = expf(logratio);
        float rc = fmaxf(1.f - clip, fminf(1.f + clip, ratio));
        float wa = -w * adv_n;
        float pg1 = wa * ratio, pg2 = wa * rc, pg = fmaxf(pg1, pg2);
        float dr = wa * dL;
        if (pg2 > pg1 && (ratio <= 1.f - clip || ratio >= 1.f + clip)) dr = 0.f;
        float dnlp = dr * ratio;
        grad_f(logits, lb, 0, A, act, mask, mb, cp, lse, ent, dnlp, d_ent, glog + gb);

        bl[0][tid] = pg * inv; bl[1][tid] = vloss * inv; bl[2][tid] = ent * inv;
        bl[3][tid] = (pg + vfcoef * vloss - entcoef * ent) * inv;
        bl[4][tid] = (-logratio) * inv;
        bl[5][tid] = ((ratio - 1.f) - logratio) * inv;
        bl[6][tid] = (fabsf(ratio - 1.f) > clip ? 1.f : 0.f) * inv;
    }
    __syncthreads();
    for (int s = PPO_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) for (int c = 0; c < LOSS_N; c++) bl[c][tid] += bl[c][tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        int b = blockIdx.x * LOSS_N;
        for (int c = 0; c < LOSS_N; c++) partials[b + c] = bl[c][0];
    }
}

// ---- LEGACY path: triple logit scan, no cache ----
__global__ void ppo_legacy(
        float* partials, float* glog, float* gval,
        const float* logits, const float* actions, const float* oldlp,
        const float* adv, const float* prio, const float* values, const float* returns,
        const float* mask, const float* amean, const float* avar,
        int A, int N, int T, float clip, float vfclip, float vfcoef, float entcoef) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int total = N * T;
    float inv = 1.f / float(total);
    __shared__ float bl[LOSS_N][PPO_THREADS];
    for (int c = 0; c < LOSS_N; c++) bl[c][tid] = 0.f;
    if (idx < total) {
        int n = idx / T, t = idx % T, nt = n * T + t;
        int lb = nt * (A + 1), gb = nt * A, mb = nt * A;
        float old_logp = oldlp[nt], ad = adv[nt], w = prio[n];
        float val = values[nt], ret = returns[nt], vp = logits[lb + A];
        float adv_n = (ad - amean[0]) / (sqrtf(avar[0]) + 1e-8f);
        float dL = inv, d_ent = dL * (-entcoef);
        float ve = vp - val;
        float vc = val + fmaxf(-vfclip, fminf(vfclip, ve));
        float vu = (vp - ret) * (vp - ret), vcc = (vc - ret) * (vc - ret);
        float vloss = 0.5f * fmaxf(vu, vcc);
        float dvp = 0.f;
        if (vcc > vu) { if (ve >= -vfclip && ve <= vfclip) dvp = vc - ret; }
        else dvp = vp - ret;
        gval[nt] = dL * vfcoef * dvp;

        int act = (int)actions[nt];
        // pass1 lse
        float maxl = -INFINITY, sum = 0.f, al = 0.f;
        for (int a = 0; a < A; a++) {
            float l = load_m(logits, lb, 0, a, mask, mb);
            if (a == act) al = l;
            if (l > maxl) { sum *= expf(maxl - l); maxl = l; }
            sum += expf(l - maxl);
        }
        float lse = maxl + logf(sum);
        // pass2 ent
        float ent = 0.f;
        for (int a = 0; a < A; a++) {
            float l = load_m(logits, lb, 0, a, mask, mb);
            float lp = l - lse, p = expf(lp); ent -= p * lp;
        }
        float tlp = al - lse;
        float logratio = tlp - old_logp, ratio = expf(logratio);
        float rc = fmaxf(1.f - clip, fminf(1.f + clip, ratio));
        float wa = -w * adv_n;
        float pg1 = wa * ratio, pg2 = wa * rc, pg = fmaxf(pg1, pg2);
        float dr = wa * dL;
        if (pg2 > pg1 && (ratio <= 1.f - clip || ratio >= 1.f + clip)) dr = 0.f;
        float dnlp = dr * ratio;
        // pass3 grads
        for (int j = 0; j < A; j++) {
            float l = load_m(logits, lb, 0, j, mask, mb);
            float lp = l - lse, p = expf(lp);
            float d = (j == act) ? dnlp : 0.f;
            d -= p * dnlp; d += d_ent * p * (-ent - lp);
            glog[gb + j] = d;
        }
        bl[0][tid] = pg * inv; bl[1][tid] = vloss * inv; bl[2][tid] = ent * inv;
        bl[3][tid] = (pg + vfcoef * vloss - entcoef * ent) * inv;
        bl[4][tid] = (-logratio) * inv;
        bl[5][tid] = ((ratio - 1.f) - logratio) * inv;
        bl[6][tid] = (fabsf(ratio - 1.f) > clip ? 1.f : 0.f) * inv;
    }
    __syncthreads();
    for (int s = PPO_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) for (int c = 0; c < LOSS_N; c++) bl[c][tid] += bl[c][tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        int b = blockIdx.x * LOSS_N;
        for (int c = 0; c < LOSS_N; c++) partials[b + c] = bl[c][0];
    }
}

// ---- Double-precision CPU reference (single discrete head) ----
static void cpu_double_ref(
        const float* logits, const float* actions, const float* oldlp,
        const float* adv, const float* prio, const float* values, const float* returns,
        const float* mask, float amean, float avar,
        int A, int N, int T,
        double* glog, double* gval, double loss_acc[LOSS_N],
        float clip, float vfclip, float vfcoef, float entcoef) {
    long NT = (long)N * T;
    double inv = 1.0 / (double)NT;
    for (int c = 0; c < LOSS_N; c++) loss_acc[c] = 0.0;
    for (long nt = 0; nt < NT; nt++) {
        int n = (int)(nt / T);
        int lb = (int)(nt * (A + 1)), gb = (int)(nt * A), mb = (int)(nt * A);
        double old_logp = oldlp[nt], ad = adv[nt], w = prio[n];
        double val = values[nt], ret = returns[nt], vp = logits[lb + A];
        double adv_n = (ad - amean) / (sqrt((double)avar) + 1e-8);
        double dL = inv, d_ent = dL * (-(double)entcoef);
        double ve = vp - val;
        double vc = val + fmax(-vfclip, fmin(vfclip, ve));
        double vu = (vp - ret) * (vp - ret), vcc = (vc - ret) * (vc - ret);
        double vloss = 0.5 * fmax(vu, vcc);
        double dvp = 0.0;
        if (vcc > vu) { if (ve >= -vfclip && ve <= vfclip) dvp = vc - ret; }
        else dvp = vp - ret;
        gval[nt] = dL * vfcoef * dvp;

        int act = (int)actions[nt];
        double maxl = -1e300, sum = 0.0, al = 0.0;
        std::vector<double> L(A);
        for (int a = 0; a < A; a++) {
            double l = logits[lb + a];
            if (mask[mb + a] == 0.f) l = -1e4;
            L[a] = l;
            if (a == act) al = l;
            if (l > maxl) { sum *= exp(maxl - l); maxl = l; }
            sum += exp(l - maxl);
        }
        double lse = maxl + log(sum);
        double ent = 0.0;
        for (int a = 0; a < A; a++) {
            double lp = L[a] - lse, p = exp(lp);
            ent -= p * lp;
        }
        double tlp = al - lse;
        double logratio = tlp - old_logp, ratio = exp(logratio);
        double rc = fmax(1.0 - clip, fmin(1.0 + clip, ratio));
        double wa = -w * adv_n;
        double pg1 = wa * ratio, pg2 = wa * rc, pg = fmax(pg1, pg2);
        double dr = wa * dL;
        if (pg2 > pg1 && (ratio <= 1.0 - clip || ratio >= 1.0 + clip)) dr = 0.0;
        double dnlp = dr * ratio;
        for (int j = 0; j < A; j++) {
            double lp = L[j] - lse, p = exp(lp);
            double d = (j == act) ? dnlp : 0.0;
            d -= p * dnlp; d += d_ent * p * (-ent - lp);
            glog[gb + j] = d;
        }
        loss_acc[0] += pg * inv; loss_acc[1] += vloss * inv; loss_acc[2] += ent * inv;
        loss_acc[3] += (pg + vfcoef * vloss - entcoef * ent) * inv;
        loss_acc[4] += (-logratio) * inv;
        loss_acc[5] += ((ratio - 1.0) - logratio) * inv;
        loss_acc[6] += (fabs(ratio - 1.0) > clip ? 1.0 : 0.0) * inv;
    }
}

static void fill(float* d, size_t n, unsigned seed) {
    std::vector<float> h(n);
    for (size_t i = 0; i < n; i++) {
        seed = seed * 1664525u + 1013904223u;
        h[i] = ((int)(seed % 2001) - 1000) * 0.001f;
    }
    CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
}

static bool bytes_eq(const void* a, const void* b, size_t n) {
    return memcmp(a, b, n) == 0;
}

static double max_abs_err(const float* a, const double* b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) m = fmax(m, fabs((double)a[i] - b[i]));
    return m;
}

static double max_abs_err_f(const float* a, const float* b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) m = fmax(m, fabs((double)a[i] - (double)b[i]));
    return m;
}

struct Buf {
    float *part, *glog, *gval, *logits, *act, *oldlp, *adv, *prio, *val, *ret, *mask, *am, *av;
    int A, N, T;
};

static void alloc_buf(Buf* b, int A, int N, int T) {
    b->A = A; b->N = N; b->T = T;
    long NT = (long)N * T, ln = NT * (A + 1), mn = NT * A;
    int grid = (int)((NT + PPO_THREADS - 1) / PPO_THREADS);
    CUDA_CHECK(cudaMalloc(&b->part, grid * LOSS_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b->glog, mn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b->gval, NT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b->logits, ln * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b->act, NT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b->oldlp, NT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b->adv, NT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b->prio, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b->val, NT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b->ret, NT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b->mask, mn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b->am, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b->av, sizeof(float)));
}

static void free_buf(Buf* b) {
    cudaFree(b->part); cudaFree(b->glog); cudaFree(b->gval); cudaFree(b->logits);
    cudaFree(b->act); cudaFree(b->oldlp); cudaFree(b->adv); cudaFree(b->prio);
    cudaFree(b->val); cudaFree(b->ret); cudaFree(b->mask); cudaFree(b->am); cudaFree(b->av);
}

static void init_buf(Buf* b, unsigned seed) {
    long NT = (long)b->N * b->T, ln = NT * (b->A + 1), mn = NT * b->A;
    fill(b->logits, ln, seed);
    fill(b->oldlp, NT, seed + 1);
    fill(b->adv, NT, seed + 2);
    fill(b->val, NT, seed + 3);
    fill(b->ret, NT, seed + 4);
    std::vector<float> act(NT), prio(b->N, 1.f), mask(mn, 1.f);
    for (long i = 0; i < NT; i++) act[i] = (float)(i % b->A);
    CUDA_CHECK(cudaMemcpy(b->act, act.data(), NT * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b->prio, prio.data(), b->N * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b->mask, mask.data(), mn * 4, cudaMemcpyHostToDevice));
    float z = 0.f, o = 1.f;
    CUDA_CHECK(cudaMemcpy(b->am, &z, 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b->av, &o, 4, cudaMemcpyHostToDevice));
}

typedef void (*kern_fn)(float*, float*, float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*, const float*,
    int, int, int, float, float, float, float);

static void run_k(kern_fn k, Buf* b) {
    long NT = (long)b->N * b->T;
    int grid = (int)((NT + PPO_THREADS - 1) / PPO_THREADS);
    k<<<grid, PPO_THREADS>>>(b->part, b->glog, b->gval, b->logits, b->act, b->oldlp,
        b->adv, b->prio, b->val, b->ret, b->mask, b->am, b->av,
        b->A, b->N, b->T, 0.2f, 0.2f, 0.5f, 0.01f);
    CUDA_CHECK(cudaDeviceSynchronize());
}

static float time_us(kern_fn k, Buf* b, int warm, int iters) {
    for (int i = 0; i < warm; i++) run_k(k, b);
    cudaEvent_t a, c; cudaEventCreate(&a); cudaEventCreate(&c);
    cudaEventRecord(a);
    for (int i = 0; i < iters; i++) {
        long NT = (long)b->N * b->T;
        int grid = (int)((NT + PPO_THREADS - 1) / PPO_THREADS);
        k<<<grid, PPO_THREADS>>>(b->part, b->glog, b->gval, b->logits, b->act, b->oldlp,
            b->adv, b->prio, b->val, b->ret, b->mask, b->am, b->av,
            b->A, b->N, b->T, 0.2f, 0.2f, 0.5f, 0.01f);
    }
    cudaEventRecord(c); cudaEventSynchronize(c);
    float ms; cudaEventElapsedTime(&ms, a, c);
    cudaEventDestroy(a); cudaEventDestroy(c);
    return ms / iters * 1000.f;
}

int main() {
    int fail = 0;
    printf("=== Determinism (new path, 3 runs bitwise) ===\n");
    {
        Buf b; alloc_buf(&b, 8, 256, 32); init_buf(&b, 42);
        long NT = (long)b.N * b.T, mn = NT * b.A;
        int grid = (int)((NT + PPO_THREADS - 1) / PPO_THREADS);
        std::vector<float> g0(mn), g1(mn), v0(NT), v1(NT), p0(grid * LOSS_N), p1(grid * LOSS_N);
        run_k(ppo_new, &b);
        CUDA_CHECK(cudaMemcpy(g0.data(), b.glog, mn * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(v0.data(), b.gval, NT * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(p0.data(), b.part, grid * LOSS_N * 4, cudaMemcpyDeviceToHost));
        bool ok = true;
        for (int r = 0; r < 3; r++) {
            run_k(ppo_new, &b);
            CUDA_CHECK(cudaMemcpy(g1.data(), b.glog, mn * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(v1.data(), b.gval, NT * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(p1.data(), b.part, grid * LOSS_N * 4, cudaMemcpyDeviceToHost));
            if (!bytes_eq(g0.data(), g1.data(), mn * 4) ||
                !bytes_eq(v0.data(), v1.data(), NT * 4) ||
                !bytes_eq(p0.data(), p1.data(), grid * LOSS_N * 4)) {
                ok = false;
            }
        }
        printf("  %s\n", ok ? "ok" : "FAIL");
        if (!ok) fail++;
        free_buf(&b);
    }

    printf("\n=== Numerical: max|float - double|  (new vs legacy) ===\n");
    printf("%5s %6s %5s %12s %12s %12s\n", "A", "N", "T", "new_vs_f64", "leg_vs_f64", "new_vs_leg");
    for (int A : {3, 8, 16}) {
        for (int T : {32, 128}) {
            for (int Ntot : {8192, 16384}) {
                int N = std::max(1, Ntot / T);
                Buf bn, bl; alloc_buf(&bn, A, N, T); alloc_buf(&bl, A, N, T);
                init_buf(&bn, 99); // copy same init to legacy
                long NT = (long)N * T, ln = NT * (A + 1), mn = NT * A;
                // sync host inputs from bn
                std::vector<float> hlog(ln), hact(NT), hold(NT), hadv(NT), hpri(N),
                    hval(NT), hret(NT), hmask(mn);
                CUDA_CHECK(cudaMemcpy(hlog.data(), bn.logits, ln * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(hact.data(), bn.act, NT * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(hold.data(), bn.oldlp, NT * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(hadv.data(), bn.adv, NT * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(hpri.data(), bn.prio, N * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(hval.data(), bn.val, NT * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(hret.data(), bn.ret, NT * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(hmask.data(), bn.mask, mn * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(bl.logits, bn.logits, ln * 4, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(bl.act, bn.act, NT * 4, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(bl.oldlp, bn.oldlp, NT * 4, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(bl.adv, bn.adv, NT * 4, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(bl.prio, bn.prio, N * 4, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(bl.val, bn.val, NT * 4, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(bl.ret, bn.ret, NT * 4, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(bl.mask, bn.mask, mn * 4, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(bl.am, bn.am, 4, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(bl.av, bn.av, 4, cudaMemcpyDeviceToDevice));

                run_k(ppo_new, &bn);
                run_k(ppo_legacy, &bl);

                std::vector<float> glog_n(mn), glog_l(mn), gval_n(NT), gval_l(NT);
                CUDA_CHECK(cudaMemcpy(glog_n.data(), bn.glog, mn * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(glog_l.data(), bl.glog, mn * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(gval_n.data(), bn.gval, NT * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(gval_l.data(), bl.gval, NT * 4, cudaMemcpyDeviceToHost));

                std::vector<double> glog_d(mn), gval_d(NT);
                double loss_d[LOSS_N];
                cpu_double_ref(hlog.data(), hact.data(), hold.data(), hadv.data(), hpri.data(),
                    hval.data(), hret.data(), hmask.data(), 0.f, 1.f, A, N, T,
                    glog_d.data(), gval_d.data(), loss_d, 0.2f, 0.2f, 0.5f, 0.01f);

                double e_new = fmax(max_abs_err(glog_n.data(), glog_d.data(), mn),
                                    max_abs_err(gval_n.data(), gval_d.data(), NT));
                double e_leg = fmax(max_abs_err(glog_l.data(), glog_d.data(), mn),
                                    max_abs_err(gval_l.data(), gval_d.data(), NT));
                double e_nl = fmax(max_abs_err_f(glog_n.data(), glog_l.data(), mn),
                                   max_abs_err_f(gval_n.data(), gval_l.data(), NT));
                printf("%5d %6d %5d %12.3e %12.3e %12.3e\n", A, N, T, e_new, e_leg, e_nl);
                // new should not be much worse than legacy vs double
                if (e_new > e_leg * 10.0 + 1e-5) {
                    printf("  FAIL: new much worse than legacy vs f64\n");
                    fail++;
                }
                free_buf(&bn); free_buf(&bl);
            }
        }
    }

    printf("\n=== Timing new vs legacy (us) ===\n");
    printf("%5s %6s %5s %10s %10s %8s\n", "A", "N", "T", "legacy", "new", "ratio");
    for (int A : {3, 8, 16}) {
        for (int T : {32, 128, 512}) {
            for (int mb : {8192, 16384, 65536}) {
                int N = std::max(1, mb / T);
                Buf b; alloc_buf(&b, A, N, T); init_buf(&b, 7);
                float ul = time_us(ppo_legacy, &b, 10, 40);
                float un = time_us(ppo_new, &b, 10, 40);
                printf("%5d %6d %5d %10.2f %10.2f %8.3f\n", A, N, T, ul, un, un / ul);
                free_buf(&b);
            }
        }
    }

    printf("\nfail=%d\n", fail);
    return fail ? 1 : 0;
}
