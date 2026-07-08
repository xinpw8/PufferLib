// protein.cu -- Protein hyperparameter optimizer (CUDA).

#ifndef PROTEIN_CU
#define PROTEIN_CU

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <curand.h>
#include <curand_kernel.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define PROTEIN_EPSILON 1e-6f
#define PROTEIN_NUM_COST_RATIOS 6
#define PROTEIN_ACQ_MAX_CAP 65536
#define PROTEIN_COST_QUANTILE 0.97f
#define PROTEIN_MIN_OBS_NO_FAIL 100
#define PROTEIN_COST_GROWTH 1.01f
#define PROTEIN_CLF_ITERS 100
#define PROTEIN_THRESHOLD_COST_CAP 1.2f
#define PROTEIN_THRESHOLD_FALLBACK 0.9f
#define PROTEIN_RUNNING_BUF_CAP 30

typedef enum {
    SPACE_LINEAR,
    SPACE_LOG,
    SPACE_POW2,
    SPACE_LOGIT,
} SpaceType;

typedef struct {
    SpaceType type;
    float min, max, scale;
    float norm_min, norm_max;
    int is_integer;
} Space;

typedef struct {
    Space* spaces;
    int num;
    int cost_idx;
    int optimize_direction;
} SweepSpace;

static inline float space_normalize(const Space* s, float value) {
    float zero_one = 0;
    switch (s->type) {
    case SPACE_LINEAR:
        zero_one = (value - s->min) / (s->max - s->min);
        break;
    case SPACE_LOG:
        zero_one = (log10f(value) - log10f(s->min)) /
            (log10f(s->max) - log10f(s->min));
        break;
    case SPACE_POW2:
        zero_one = (log2f(value) - log2f(s->min)) /
            (log2f(s->max) - log2f(s->min));
        break;
    case SPACE_LOGIT: {
        float clamped = fmaxf(s->min, fminf(value, s->max));
        zero_one = (log10f(1.0f - clamped) - log10f(1.0f - s->min)) /
            (log10f(1.0f - s->max) - log10f(1.0f - s->min));
        break;
    }
    }
    return 2.0f * zero_one - 1.0f;
}

static inline float space_unnormalize(const Space* s, float norm) {
    float zero_one = (norm + 1.0f) * 0.5f;
    float val = 0;
    switch (s->type) {
    case SPACE_LINEAR:
        val = zero_one * (s->max - s->min) + s->min;
        if (s->is_integer) {
            val = roundf(val);
        }
        break;
    case SPACE_LOG: {
        float log_val = zero_one * (log10f(s->max) - log10f(s->min)) +
            log10f(s->min);
        val = powf(10.0f, log_val);
        if (s->is_integer) {
            val = roundf(val);
        }
        break;
    }
    case SPACE_POW2: {
        float log_val = zero_one * (log2f(s->max) - log2f(s->min)) +
            log2f(s->min);
        val = powf(2.0f, roundf(log_val));
        break;
    }
    case SPACE_LOGIT: {
        float log_val = zero_one * (log10f(1.0f - s->max) -
            log10f(1.0f - s->min)) + log10f(1.0f - s->min);
        val = 1.0f - powf(10.0f, log_val);
        break;
    }
    }
    return val;
}

static inline void space_init(Space* s, SpaceType type, float min, float max,
        float scale, int is_integer) {
    s->type = type;
    s->min = min;
    s->max = max;
    s->scale = scale;
    s->is_integer = is_integer;
    s->norm_min = space_normalize(s, min);
    s->norm_max = space_normalize(s, max);
}

static inline SweepSpace* sweep_space_create(int capacity, int cost_idx,
        int optimize_direction) {
    SweepSpace* space = (SweepSpace*)calloc(1, sizeof(SweepSpace));
    space->spaces = (Space*)calloc((size_t)capacity, sizeof(Space));
    space->num = capacity;
    space->cost_idx = cost_idx;
    space->optimize_direction = optimize_direction;
    return space;
}

static inline void sweep_space_destroy(SweepSpace* space) {
    if (!space) {
        return;
    }
    free(space->spaces);
    free(space);
}

static int protein_float_cmp(const void *a, const void *b) {
    float fa = *(const float *)a, fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

static float protein_quantile(const float *data, int n, float q) {
    float *sorted = (float *)malloc((size_t)n * sizeof(float));
    memcpy(sorted, data, (size_t)n * sizeof(float));
    qsort(sorted, n, sizeof(float), protein_float_cmp);
    float idx = q * (float)(n - 1);
    int lo = (int)idx, hi = lo + 1;
    if (hi >= n)
        hi = n - 1;
    float frac = idx - (float)lo;
    float val = sorted[lo] * (1.0f - frac) + sorted[hi] * frac;
    free(sorted);
    return val;
}

typedef struct {
    const float *x;
    const float *y;
    int n;
    float q;
} PinballData;

static float pinball_loss(float a, float b, const void *data) {
    const PinballData *pd = (const PinballData *)data;
    float loss = 0.0f;
    for (int i = 0; i < pd->n; i++) {
        float r = pd->y[i] - (a + b * pd->x[i]);
        loss += (r > 0.0f) ? pd->q * r : (pd->q - 1.0f) * r;
    }
    return loss;
}

// clang-format off
#define SWAP_F(a, b) do { float _t = (a); (a) = (b); (b) = _t; } while (0)
// clang-format on

static void nelder_mead_2d(float (*f)(float, float, const void *), const void *data, float *out_a, float *out_b,
                           float a0, float b0, int max_iter, float tol) {
    float sx[3] = {a0, a0 + 0.05f, a0 - 0.05f};
    float sy[3] = {b0, b0 - 0.05f, b0 + 0.05f};
    float sv[3];
    for (int i = 0; i < 3; i++)
        sv[i] = f(sx[i], sy[i], data);

    for (int iter = 0; iter < max_iter; iter++) {
        for (int i = 0; i < 2; i++)
            for (int j = i + 1; j < 3; j++)
                if (sv[j] < sv[i]) {
                    SWAP_F(sx[i], sx[j]);
                    SWAP_F(sy[i], sy[j]);
                    SWAP_F(sv[i], sv[j]);
                }

        if (fabsf(sv[2] - sv[0]) < tol)
            break;

        float cx = (sx[0] + sx[1]) * 0.5f;
        float cy = (sy[0] + sy[1]) * 0.5f;

        float rx = cx + (cx - sx[2]), ry = cy + (cy - sy[2]);
        float rv = f(rx, ry, data);
        if (rv < sv[1] && rv >= sv[0]) {
            sx[2] = rx;
            sy[2] = ry;
            sv[2] = rv;
            continue;
        }
        if (rv < sv[0]) {
            float ex = cx + 2.0f * (rx - cx), ey = cy + 2.0f * (ry - cy);
            float ev = f(ex, ey, data);
            if (ev < rv) {
                sx[2] = ex;
                sy[2] = ey;
                sv[2] = ev;
            } else {
                sx[2] = rx;
                sy[2] = ry;
                sv[2] = rv;
            }
            continue;
        }
        float ccx, ccy;
        if (rv < sv[2]) {
            ccx = cx + 0.5f * (rx - cx);
            ccy = cy + 0.5f * (ry - cy);
        } else {
            ccx = cx + 0.5f * (sx[2] - cx);
            ccy = cy + 0.5f * (sy[2] - cy);
        }
        float ccv = f(ccx, ccy, data);
        if (ccv < sv[2]) {
            sx[2] = ccx;
            sy[2] = ccy;
            sv[2] = ccv;
            continue;
        }

        for (int i = 1; i < 3; i++) {
            sx[i] = sx[0] + 0.5f * (sx[i] - sx[0]);
            sy[i] = sy[0] + 0.5f * (sy[i] - sy[0]);
            sv[i] = f(sx[i], sy[i], data);
        }
    }
    *out_a = sx[0];
    *out_b = sy[0];
}

#undef SWAP_F

static void polyfit_1d(const float *x, const float *y, int n, float *intercept, float *slope) {
    float sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i = 0; i < n; i++) {
        sx += x[i];
        sy += y[i];
        sxx += x[i] * x[i];
        sxy += x[i] * y[i];
    }
    float det = (float)n * sxx - sx * sx;
    if (fabsf(det) < 1e-10f) {
        *intercept = sy / fmaxf((float)n, 1.0f);
        *slope = 0.0f;
        return;
    }
    *slope = ((float)n * sxy - sx * sy) / det;
    *intercept = (sy - *slope * sx) / (float)n;
}

typedef struct {
    float A, B;
    float max_score;
    float upper_cost_threshold;
    float quantile;
    int min_samples;
    int is_fitted;
} ProteinCostModel;

void protein_cost_model_fit(ProteinCostModel *m, const float *scores, const float *costs, int n,
                            float upper_cost_threshold) {
    m->is_fitted = 0;
    if (n == 0)
        return;

    float s_max = scores[0];
    for (int i = 1; i < n; i++)
        if (scores[i] > s_max)
            s_max = scores[i];
    m->max_score = s_max;
    m->upper_cost_threshold = upper_cost_threshold;

    int n_valid = 0;
    for (int i = 0; i < n; i++)
        if (costs[i] > PROTEIN_EPSILON && isfinite(scores[i]))
            n_valid++;

    if (n_valid < m->min_samples)
        return;

    float *x = (float *)malloc((size_t)n_valid * sizeof(float));
    float *y = (float *)malloc((size_t)n_valid * sizeof(float));
    int j = 0;
    for (int i = 0; i < n; i++) {
        if (costs[i] > PROTEIN_EPSILON && isfinite(scores[i])) {
            x[j] = logf(costs[i]);
            y[j] = scores[i];
            j++;
        }
    }

    float a_init, b_init;
    polyfit_1d(x, y, n_valid, &a_init, &b_init);

    PinballData pd = {x, y, n_valid, m->quantile};
    nelder_mead_2d(pinball_loss, &pd, &m->A, &m->B, a_init, b_init, 500, 1e-7f);

    free(x);
    free(y);
    m->is_fitted = 1;
}

float protein_cost_model_threshold(const ProteinCostModel *m, float cost, float min_cost_frac, float abs_min_cost) {
    if (!m->is_fitted)
        return -FLT_MAX;
    float min_allowed = m->upper_cost_threshold * min_cost_frac + abs_min_cost;
    if (cost < min_allowed)
        return -FLT_MAX;
    if (cost > PROTEIN_THRESHOLD_COST_CAP * m->upper_cost_threshold)
        return PROTEIN_THRESHOLD_FALLBACK * m->max_score;
    return m->A + m->B * logf(cost);
}

static const float *g_protein_pareto_costs;

static int protein_pareto_cmp(const void *a, const void *b) {
    float ca = g_protein_pareto_costs[*(const int *)a];
    float cb = g_protein_pareto_costs[*(const int *)b];
    return (ca > cb) - (ca < cb);
}

int protein_pareto_front(const float *scores, const float *costs, int n, int *out_indices) {
    if (n == 0)
        return 0;

    int *sorted = (int *)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++)
        sorted[i] = i;
    g_protein_pareto_costs = costs;
    qsort(sorted, n, sizeof(int), protein_pareto_cmp);

    int count = 0;
    float max_score = -FLT_MAX;
    for (int i = 0; i < n; i++) {
        int idx = sorted[i];
        if (scores[idx] > max_score + PROTEIN_EPSILON) {
            out_indices[count++] = idx;
            max_score = scores[idx];
        }
    }

    free(sorted);
    return count;
}

int protein_prune_pareto(const float *scores, const float *costs, int *indices, int n, float eff_threshold,
                         float stop_fraction) {
    if (n < 2)
        return n;

    float s_min = scores[indices[0]], s_max = s_min;
    float c_min = costs[indices[0]], c_max = c_min;
    for (int i = 1; i < n; i++) {
        float s = scores[indices[i]], c = costs[indices[i]];
        if (s < s_min)
            s_min = s;
        if (s > s_max)
            s_max = s;
        if (c < c_min)
            c_min = c;
        if (c > c_max)
            c_max = c;
    }
    float s_range = fmaxf(s_max - s_min, PROTEIN_EPSILON);
    float c_range = fmaxf(c_max - c_min, PROTEIN_EPSILON);
    float max_pareto_s = scores[indices[n - 1]];

    int pruned = n;
    for (int i = pruned - 1; i > 1; i--) {
        if (scores[indices[i - 1]] < stop_fraction * max_pareto_s)
            break;
        float ng = (scores[indices[i]] - scores[indices[i - 1]]) / s_range;
        float nc = (costs[indices[i]] - costs[indices[i - 1]]) / c_range;
        float eff = ng / (nc + PROTEIN_EPSILON);
        if (eff < eff_threshold) {
            for (int j = i; j < pruned - 1; j++)
                indices[j] = indices[j + 1];
            pruned--;
        } else {
            break;
        }
    }
    return pruned;
}

int protein_build_search_centers(const float *obs_params, int dim, const int *pareto_indices, int n_pareto,
                                 const int *top_indices, int n_top, int cost_dim, float *out_centers) {
    int count = 0;
    for (int i = 0; i < n_pareto; i++) {
        memcpy(&out_centers[(size_t)count * dim], &obs_params[(size_t)pareto_indices[i] * dim],
               (size_t)dim * sizeof(float));
        count++;
    }
    if (cost_dim >= 0) {
        float divisors[] = {2.0f, 3.0f};
        for (int r = 0; r < 2; r++) {
            for (int i = 0; i < n_top; i++) {
                const float *src = &obs_params[(size_t)top_indices[i] * dim];
                float orig = src[cost_dim];
                memcpy(&out_centers[(size_t)count * dim], src, (size_t)dim * sizeof(float));
                out_centers[(size_t)count * dim + cost_dim] = orig - (orig + 1.0f) / divisors[r];
                count++;
            }
        }
    } else {
        for (int i = 0; i < n_top; i++) {
            memcpy(&out_centers[(size_t)count * dim], &obs_params[(size_t)top_indices[i] * dim],
                   (size_t)dim * sizeof(float));
            count++;
        }
    }
    return count;
}

float protein_sample_target_cost(float *ratio_pool, int *pool_remaining, int pool_total, float expansion_rate) {
    if (*pool_remaining <= 0) {
        for (int i = pool_total - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            float tmp = ratio_pool[i];
            ratio_pool[i] = ratio_pool[j];
            ratio_pool[j] = tmp;
        }
        *pool_remaining = pool_total;
    }
    float ratio = ratio_pool[--(*pool_remaining)];

    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float noise = 0.1f * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);

    ratio = fmaxf(0.0f, fminf(1.0f, ratio + noise));
    return (1.0f + expansion_rate) * ratio;
}

static float protein_logit_transform(float value) {
    float epsilon = 1e-9f;
    value = fmaxf(epsilon, fminf(1.0f - epsilon, value));
    float logit = logf(value / (1.0f - value));
    return fmaxf(-5.0f, fminf(100.0f, logit));
}


// Exact GP regression, CUDA (cuBLAS/cuSOLVER).

// clang-format off
static inline void _cuda_check(cudaError_t e, const char* f, int l) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", f, l, cudaGetErrorString(e)); abort(); }
}
static inline void _cublas_check(cublasStatus_t s, const char* f, int l) {
    if (s != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cuBLAS error %s:%d: %d\n", f, l, (int)s); abort(); }
}
#define CUDA_CHECK(call) _cuda_check((call), __FILE__, __LINE__)
#define CUBLAS_CHECK(call) _cublas_check((call), __FILE__, __LINE__)
// clang-format on

#define SP_LB 1e-4
__host__ __device__ __forceinline__ float softplus(float x) { return (x > 20.0 ? x : log1p(exp(x))) + SP_LB; }
__host__ __device__ __forceinline__ float inv_softplus(float x) {
    float v = x - SP_LB;
    return v > 20.0 ? v : log(expm1(v));
}
__host__ __device__ __forceinline__ float softplus_grad(float x) { return 1.0 / (1.0 + exp(-x)); }

typedef struct GPKernel GPKernel;
struct GPKernel {
    int n_params;
    float *raw_params;
    char tag[4];
};

#define GP_KERNEL_TAG_MATERN32_LINEAR "M32L"
#define SF_IDX(np) ((np)-2)
#define OFF_IDX(np) ((np)-1)
#define GP_BLOCK 16
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

inline int gp_grid(int n) { return (n + GP_BLOCK - 1) / GP_BLOCK; }

typedef struct {
    float sigma_f, offset;
} KParams;

static inline KParams kernel_params(const GPKernel *k) {
    int np = k->n_params;
    return (KParams){softplus(k->raw_params[SF_IDX(np)]), softplus(k->raw_params[OFF_IDX(np)])};
}

__global__ void matern32lin_k_kernel(const float *__restrict__ X1, const float *__restrict__ X2, float *__restrict__ K,
                                     const float *__restrict__ inv_ells, int n, int m, int d, float sigma_f,
                                     float diag_noise, float offset) {
    int col = blockIdx.x * GP_BLOCK + threadIdx.x;
    int row = blockIdx.y * GP_BLOCK + threadIdx.y;
    if (row >= n || col >= m)
        return;
    float dot = 0.0, r2 = 0.0;
    for (int k = 0; k < d; k++) {
        float xr = X1[row * d + k], xc = X2[col * d + k];
        dot += xr * xc;
        float diff = (xr - xc) * inv_ells[k];
        r2 += diff * diff;
    }
    if (r2 < 0.0)
        r2 = 0.0;
    float u = sqrt(3.0) * sqrt(r2);
    float val = sigma_f * (dot + offset + (1.0 + u) * exp(-u));
    if (diag_noise != 0.0 && row == col)
        val += diag_noise;
    K[col * n + row] = val;
}

__global__ void matern32lin_k_build_D_ard(const float *__restrict__ X, float *__restrict__ D,
                                          const float *__restrict__ inv_ells, int n, int d) {
    int col = blockIdx.x * GP_BLOCK + threadIdx.x;
    int row = blockIdx.y * GP_BLOCK + threadIdx.y;
    if (row >= n || col >= n)
        return;
    float sq = 0.0;
    for (int k = 0; k < d; k++) {
        float diff = (X[row * d + k] - X[col * d + k]) * inv_ells[k];
        sq += diff * diff;
    }
    D[col * n + row] = sq < 0.0 ? 0.0 : sq;
}

__global__ void matern32lin_k_compute_W(const float *__restrict__ alpha, const float *__restrict__ Kinv,
                                        float *__restrict__ W, int n) {
    int col = blockIdx.x * GP_BLOCK + threadIdx.x;
    int row = blockIdx.y * GP_BLOCK + threadIdx.y;
    if (row >= n || col >= n)
        return;
    W[col * n + row] = alpha[row] * alpha[col] - Kinv[col * n + row];
}

__global__ void matern32lin_k_dk_dell_d(const float *__restrict__ X, const float *__restrict__ D_ard,
                                        float *__restrict__ out, int n, int d, int dd, float sigma_f,
                                        float inv_ell_d3) {
    int col = blockIdx.x * GP_BLOCK + threadIdx.x;
    int row = blockIdx.y * GP_BLOCK + threadIdx.y;
    if (row >= n || col >= n)
        return;
    float r2 = D_ard[col * n + row];
    float diff = X[row * d + dd] - X[col * d + dd];
    out[col * n + row] = sigma_f * 3.0 * diff * diff * inv_ell_d3 * exp(-sqrt(3.0) * sqrt(r2));
}

__global__ void matern32lin_k_fill(float *v, int n, float val) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n)
        v[i] = val;
}

__global__ void matern32lin_k_kself_batch(const float *__restrict__ X, float *__restrict__ out, int m, int d,
                                          float sigma_f, float offset) {
    int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (row >= m)
        return;
    float norm2 = 0.0;
    for (int k = 0; k < d; k++) {
        float v = X[row * d + k];
        norm2 += v * v;
    }
    out[row] = sigma_f * (norm2 + offset + 1.0);
}

static float *h2d(const float *h, int n, cudaStream_t stream) {
    float *d;
    CUDA_CHECK(cudaMallocAsync(&d, (size_t)n * sizeof(float), stream));
    CUDA_CHECK(cudaMemcpyAsync(d, h, (size_t)n * sizeof(float), cudaMemcpyHostToDevice, stream));
    return d;
}

static float *device_inv_ells(const GPKernel *k, int d, cudaStream_t stream) {
    float *h = (float *)malloc(d * sizeof(float));
    for (int i = 0; i < d; i++)
        h[i] = 1.0 / softplus(k->raw_params[i]);
    float *dev = h2d(h, d, stream);
    free(h);
    return dev;
}

static void matern32lin_build_K(const GPKernel *k, const float *d_X, int n, int d, float sigma_n, float *d_K,
                                cudaStream_t stream) {
    KParams p = kernel_params(k);
    float *d_inv = device_inv_ells(k, d, stream);
    dim3 block(GP_BLOCK, GP_BLOCK), grid(gp_grid(n), gp_grid(n));
    matern32lin_k_kernel<<<grid, block, 0, stream>>>(d_X, d_X, d_K, d_inv, n, n, d, p.sigma_f, sigma_n, p.offset);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFreeAsync(d_inv, stream));
}

static void matern32lin_build_Ks(const GPKernel *k, const float *d_Xtr, const float *d_Xte, int n, int m, int d,
                                 float *d_Ks, cudaStream_t stream) {
    KParams p = kernel_params(k);
    float *d_inv = device_inv_ells(k, d, stream);
    dim3 block(GP_BLOCK, GP_BLOCK), grid(gp_grid(m), gp_grid(n));
    matern32lin_k_kernel<<<grid, block, 0, stream>>>(d_Xtr, d_Xte, d_Ks, d_inv, n, m, d, p.sigma_f, 0.0, p.offset);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFreeAsync(d_inv, stream));
}

static float matern32lin_k_self(const GPKernel *k, const float *x, int d) {
    KParams p = kernel_params(k);
    float norm2 = 0.0;
    for (int i = 0; i < d; i++)
        norm2 += x[i] * x[i];
    return p.sigma_f * (norm2 + p.offset + 1.0);
}

static void matern32lin_build_kself_batch(const GPKernel *k, const float *d_X, int m, int d, float *d_out,
                                          cudaStream_t stream) {
    KParams p = kernel_params(k);
    matern32lin_k_kself_batch<<<(m + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(d_X, d_out, m, d, p.sigma_f,
                                                                                            p.offset);
    CUDA_CHECK(cudaGetLastError());
}

static void matern32lin_mll_grad(const GPKernel *k, const float *d_X, int n, int d, cublasHandle_t cublas,
                                 cudaStream_t stream, const float *d_alpha, const float *d_Kinv, float *kernel_grads) {
    int np = k->n_params;
    KParams p = kernel_params(k);
    const float one = 1.0, zero = 0.0;
    CUBLAS_CHECK(cublasSetStream(cublas, stream));

    float *d_inv = device_inv_ells(k, d, stream);
    float *d_D, *d_W, *d_temp;
    CUDA_CHECK(cudaMallocAsync(&d_D, (size_t)n * n * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_W, (size_t)n * n * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_temp, (size_t)n * n * sizeof(float), stream));
    dim3 block(GP_BLOCK, GP_BLOCK), grid(gp_grid(n), gp_grid(n));

    matern32lin_k_build_D_ard<<<grid, block, 0, stream>>>(d_X, d_D, d_inv, n, d);
    CUDA_CHECK(cudaGetLastError());
    matern32lin_k_compute_W<<<grid, block, 0, stream>>>(d_alpha, d_Kinv, d_W, n);
    CUDA_CHECK(cudaGetLastError());

    for (int dd = 0; dd < d; dd++) {
        float ell_d = softplus(k->raw_params[dd]);
        float inv_ell3 = 1.0 / (ell_d * ell_d * ell_d);
        matern32lin_k_dk_dell_d<<<grid, block, 0, stream>>>(d_X, d_D, d_temp, n, d, dd, p.sigma_f, inv_ell3);
        CUDA_CHECK(cudaGetLastError());
        float dot_val;
        CUBLAS_CHECK(cublasSdot(cublas, n * n, d_W, 1, d_temp, 1, &dot_val));
        kernel_grads[dd] = 0.5 * dot_val * softplus_grad(k->raw_params[dd]);
    }

    matern32lin_k_kernel<<<grid, block, 0, stream>>>(d_X, d_X, d_temp, d_inv, n, n, d, p.sigma_f, 0.0, p.offset);
    CUDA_CHECK(cudaGetLastError());
    float dot_val;
    CUBLAS_CHECK(cublasSdot(cublas, n * n, d_W, 1, d_temp, 1, &dot_val));
    kernel_grads[SF_IDX(np)] = 0.5 * dot_val / p.sigma_f * softplus_grad(k->raw_params[SF_IDX(np)]);

    float *d_ones, *d_wrow;
    CUDA_CHECK(cudaMallocAsync(&d_ones, (size_t)n * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_wrow, (size_t)n * sizeof(float), stream));
    matern32lin_k_fill<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(d_ones, n, 1.0);
    CUDA_CHECK(cudaGetLastError());
    float w_sum;
    CUBLAS_CHECK(cublasSgemv(cublas, CUBLAS_OP_N, n, n, &one, d_W, n, d_ones, 1, &zero, d_wrow, 1));
    CUBLAS_CHECK(cublasSdot(cublas, n, d_wrow, 1, d_ones, 1, &w_sum));
    kernel_grads[OFF_IDX(np)] = 0.5 * p.sigma_f * w_sum * softplus_grad(k->raw_params[OFF_IDX(np)]);
    CUDA_CHECK(cudaFreeAsync(d_ones, stream));
    CUDA_CHECK(cudaFreeAsync(d_wrow, stream));

    CUDA_CHECK(cudaFreeAsync(d_inv, stream));
    CUDA_CHECK(cudaFreeAsync(d_D, stream));
    CUDA_CHECK(cudaFreeAsync(d_W, stream));
    CUDA_CHECK(cudaFreeAsync(d_temp, stream));
}

GPKernel *gp_kernel_matern32_linear(int dim, float lengthscale, float outputscale, float offset) {
    int n_params = dim + 2;
    GPKernel *k = (GPKernel *)malloc(sizeof(GPKernel) + (size_t)n_params * sizeof(float));
    k->n_params = n_params;
    k->raw_params = (float *)(k + 1);
    memcpy(k->tag, GP_KERNEL_TAG_MATERN32_LINEAR, 4);
    for (int i = 0; i < dim; i++)
        k->raw_params[i] = inv_softplus(lengthscale);
    k->raw_params[SF_IDX(n_params)] = inv_softplus(outputscale);
    k->raw_params[OFF_IDX(n_params)] = inv_softplus(offset);
    return k;
}

float gp_kernel_get_lengthscale(const GPKernel *k, int d) { return softplus(k->raw_params[d]); }
float gp_kernel_get_outputscale(const GPKernel *k) { return softplus(k->raw_params[SF_IDX(k->n_params)]); }
float gp_kernel_get_offset(const GPKernel *k) { return softplus(k->raw_params[OFF_IDX(k->n_params)]); }
void gp_kernel_set_lengthscale(GPKernel *k, int d, float v) { k->raw_params[d] = inv_softplus(v); }
void gp_kernel_set_outputscale(GPKernel *k, float v) { k->raw_params[SF_IDX(k->n_params)] = inv_softplus(v); }
void gp_kernel_set_offset(GPKernel *k, float v) { k->raw_params[OFF_IDX(k->n_params)] = inv_softplus(v); }


// clang-format off
static inline void _cusolver_check(cusolverStatus_t s, const char* f, int l) {
    if (s != CUSOLVER_STATUS_SUCCESS) { fprintf(stderr, "cuSOLVER error %s:%d: %d\n", f, l, (int)s); abort(); }
}
#define CUSOLVER_CHECK(call) _cusolver_check((call), __FILE__, __LINE__)
// clang-format on

typedef struct {
    int dim, n, cap;
    float *d_X;
    float *d_y;
    float *d_L;
    float *d_alpha;
    float *d_work;
    int lwork;
    int *d_info;
    float raw_noise;
    float dedup_threshold;
    cublasHandle_t cublas;
    cusolverDnHandle_t cusolver;
    GPKernel *kernel;
} GaussianProcess;

__global__ void gp_k_add_diag(float *A, int n, float val) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n)
        A[(size_t)i * (n + 1)] += val;
}

__global__ void gp_k_extract_diag(const float *A, float *d, int n) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n)
        d[i] = A[(size_t)i * (n + 1)];
}

__global__ void gp_k_var_from_chol(float *vars, const float *V, int n, int m) {
    int j = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (j >= m)
        return;
    float sq = 0.0;
    const float *col = V + (size_t)j * n;
    for (int i = 0; i < n; i++)
        sq += col[i] * col[i];
    float v = vars[j] - sq;
    vars[j] = v > 0.0 ? v : 0.0;
}

#define KD_LEAF 16

typedef struct {
    int lo, hi, left, right, split_dim;
    float split_val;
} KDNode;
typedef struct {
    const float *X;
    int n, dim, n_nodes;
    int *idx;
    KDNode *nodes;
} KDTree;

static int g_kd_split_dim, g_kd_d;
static const float *g_kd_X;

static int kd_cmp_fn(const void *a, const void *b) {
    float va = g_kd_X[*(const int *)a * g_kd_d + g_kd_split_dim];
    float vb = g_kd_X[*(const int *)b * g_kd_d + g_kd_split_dim];
    return (va > vb) - (va < vb);
}

static void kd_build(KDTree *t, int node, int lo, int hi) {
    KDNode *nd = &t->nodes[node];
    nd->lo = lo;
    nd->hi = hi;
    nd->left = nd->right = -1;
    nd->split_dim = -1;
    if (hi - lo <= KD_LEAF)
        return;

    int best = 0;
    float spread = -1.0;
    for (int k = 0; k < t->dim; k++) {
        float mn = t->X[t->idx[lo] * t->dim + k], mx = mn;
        for (int i = lo + 1; i < hi; i++) {
            float v = t->X[t->idx[i] * t->dim + k];
            if (v < mn)
                mn = v;
            if (v > mx)
                mx = v;
        }
        if (mx - mn > spread) {
            spread = mx - mn;
            best = k;
        }
    }
    int mid = (lo + hi) / 2;
    g_kd_split_dim = best;
    g_kd_d = t->dim;
    g_kd_X = t->X;
    qsort(t->idx + lo, hi - lo, sizeof(int), kd_cmp_fn);
    nd->split_dim = best;
    nd->split_val = t->X[t->idx[mid] * t->dim + best];
    int ln = t->n_nodes++, rn = t->n_nodes++;
    nd->left = ln;
    nd->right = rn;
    kd_build(t, ln, lo, mid);
    kd_build(t, rn, mid, hi);
}

static KDTree kd_create(const float *X, int n, int dim) {
    KDTree t = {.X = X, .n = n, .dim = dim, .n_nodes = 1};
    t.idx = (int *)malloc(n * sizeof(int));
    t.nodes = (KDNode *)malloc(2 * n * sizeof(KDNode));
    for (int i = 0; i < n; i++)
        t.idx[i] = i;
    kd_build(&t, 0, 0, n);
    return t;
}

static void kd_query_ball(const KDTree *t, int node, const float *q, float r, float r2, int *out, int *cnt) {
    const KDNode *nd = &t->nodes[node];
    if (nd->split_dim < 0) {
        for (int i = nd->lo; i < nd->hi; i++) {
            int p = t->idx[i];
            float sq = 0.0;
            for (int k = 0; k < t->dim; k++) {
                float df = q[k] - t->X[p * t->dim + k];
                sq += df * df;
            }
            if (sq <= r2)
                out[(*cnt)++] = p;
        }
        return;
    }
    float dv = q[nd->split_dim] - nd->split_val;
    if (nd->left >= 0 && dv <= r)
        kd_query_ball(t, nd->left, q, r, r2, out, cnt);
    if (nd->right >= 0 && dv >= -r)
        kd_query_ball(t, nd->right, q, r, r2, out, cnt);
}

static int gp_filter_near_duplicates(const float *X, int n, int dim, float threshold, int *kept_indices) {
    if (n <= 0)
        return 0;
    if (n == 1) {
        kept_indices[0] = 0;
        return 1;
    }

    KDTree tree = kd_create(X, n, dim);
    int *keep = (int *)malloc(n * sizeof(int));
    int *nearby = (int *)malloc(n * sizeof(int));
    float r2 = threshold * threshold;
    for (int i = 0; i < n; i++)
        keep[i] = 1;

    for (int i = n - 1; i >= 0; i--) {
        if (!keep[i])
            continue;
        int cnt = 0;
        kd_query_ball(&tree, 0, &X[(size_t)i * dim], threshold, r2, nearby, &cnt);
        for (int j = 0; j < cnt; j++)
            if (nearby[j] != i)
                keep[nearby[j]] = 0;
    }

    int count = 0;
    for (int i = 0; i < n; i++)
        if (keep[i])
            kept_indices[count++] = i;

    free(nearby);
    free(keep);
    free(tree.idx);
    free(tree.nodes);
    return count;
}

float gp_get_noise(const GaussianProcess *gp) { return softplus(gp->raw_noise); }
void gp_set_noise(GaussianProcess *gp, float v) { gp->raw_noise = inv_softplus(v); }

GaussianProcess *gp_create(int dim, int cap, GPKernel *kernel, float noise) {
    GaussianProcess *gp = (GaussianProcess *)calloc(1, sizeof(GaussianProcess));
    gp->dim = dim;
    gp->cap = cap;
    gp->kernel = kernel;
    gp_set_noise(gp, noise);
    CUDA_CHECK(cudaMalloc(&gp->d_X, (size_t)cap * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gp->d_y, (size_t)cap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gp->d_L, (size_t)cap * cap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gp->d_alpha, (size_t)cap * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gp->d_info, sizeof(int)));
    CUBLAS_CHECK(cublasCreate(&gp->cublas));
    CUSOLVER_CHECK(cusolverDnCreate(&gp->cusolver));
    return gp;
}

void gp_destroy(GaussianProcess *gp) {
    if (!gp)
        return;
    cudaFree(gp->d_X);
    cudaFree(gp->d_y);
    cudaFree(gp->d_L);
    cudaFree(gp->d_alpha);
    cudaFree(gp->d_work);
    cudaFree(gp->d_info);
    cublasDestroy(gp->cublas);
    cusolverDnDestroy(gp->cusolver);
    free(gp->kernel);
    free(gp);
}

static int run_potrf(GaussianProcess *gp, float *d_K, int n, cudaStream_t stream) {
    CUSOLVER_CHECK(cusolverDnSetStream(gp->cusolver, stream));
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(gp->cusolver, CUBLAS_FILL_MODE_LOWER, n, d_K, n, &lwork));
    if (lwork > gp->lwork) {
        cudaFree(gp->d_work);
        CUDA_CHECK(cudaMalloc(&gp->d_work, (size_t)lwork * sizeof(float)));
        gp->lwork = lwork;
    }
    CUSOLVER_CHECK(
        cusolverDnSpotrf(gp->cusolver, CUBLAS_FILL_MODE_LOWER, n, d_K, n, gp->d_work, gp->lwork, gp->d_info));
    int h_info;
    CUDA_CHECK(cudaMemcpyAsync(&h_info, gp->d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return h_info;
}

int gp_recompute(GaussianProcess *gp, cudaStream_t stream) {
    int n = gp->n;
    if (n == 0)
        return 0;

    matern32lin_build_K(gp->kernel, gp->d_X, n, gp->dim, gp_get_noise(gp), gp->d_L, stream);

    int info = run_potrf(gp, gp->d_L, n, stream);
    if (info != 0) {
        matern32lin_build_K(gp->kernel, gp->d_X, n, gp->dim, gp_get_noise(gp), gp->d_L, stream);
        gp_k_add_diag<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(gp->d_L, n, 1e-8);
        CUDA_CHECK(cudaGetLastError());
        info = run_potrf(gp, gp->d_L, n, stream);
        if (info != 0)
            return -2;
    }

    CUDA_CHECK(cudaMemcpyAsync(gp->d_alpha, gp->d_y, (size_t)n * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    CUBLAS_CHECK(cublasSetStream(gp->cublas, stream));
    CUBLAS_CHECK(cublasStrsv(gp->cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, gp->d_L, n,
                             gp->d_alpha, 1));
    CUBLAS_CHECK(cublasStrsv(gp->cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, n, gp->d_L, n,
                             gp->d_alpha, 1));
    return 0;
}

int gp_fit(GaussianProcess *gp, const float *X, const float *y, int n, cudaStream_t stream) {
    if (n > gp->cap)
        return -1;

    if (gp->dedup_threshold > 0.0 && n > 1) {
        int *idx = (int *)malloc(n * sizeof(int));
        int n_kept = gp_filter_near_duplicates(X, n, gp->dim, gp->dedup_threshold, idx);
        float *X_c = (float *)malloc((size_t)n_kept * gp->dim * sizeof(float));
        float *y_c = (float *)malloc((size_t)n_kept * sizeof(float));
        for (int i = 0; i < n_kept; i++) {
            memcpy(&X_c[i * gp->dim], &X[idx[i] * gp->dim], gp->dim * sizeof(float));
            y_c[i] = y[idx[i]];
        }
        free(idx);
        CUDA_CHECK(
            cudaMemcpyAsync(gp->d_X, X_c, (size_t)n_kept * gp->dim * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(gp->d_y, y_c, (size_t)n_kept * sizeof(float), cudaMemcpyHostToDevice, stream));
        free(X_c);
        free(y_c);
        gp->n = n_kept;
    } else {
        CUDA_CHECK(cudaMemcpyAsync(gp->d_X, X, (size_t)n * gp->dim * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(gp->d_y, y, (size_t)n * sizeof(float), cudaMemcpyHostToDevice, stream));
        gp->n = n;
    }
    return gp_recompute(gp, stream);
}

static void predict_dev(const GaussianProcess *gp, const float *d_Xte, float *d_means, float *d_vars, int m,
                        cudaStream_t stream) {
    int n = gp->n, d = gp->dim;
    const float one = 1.0, zero = 0.0;
    CUBLAS_CHECK(cublasSetStream(gp->cublas, stream));

    float *d_Ks;
    CUDA_CHECK(cudaMallocAsync(&d_Ks, (size_t)n * m * sizeof(float), stream));
    matern32lin_build_Ks(gp->kernel, gp->d_X, d_Xte, n, m, d, d_Ks, stream);
    CUBLAS_CHECK(cublasSgemv(gp->cublas, CUBLAS_OP_T, n, m, &one, d_Ks, n, gp->d_alpha, 1, &zero, d_means, 1));

    if (d_vars) {
        matern32lin_build_kself_batch(gp->kernel, d_Xte, m, d, d_vars, stream);
        CUBLAS_CHECK(cublasStrsm(gp->cublas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                 CUBLAS_DIAG_NON_UNIT, n, m, &one, gp->d_L, n, d_Ks, n));
        gp_k_var_from_chol<<<(m + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(d_vars, d_Ks, n, m);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaFreeAsync(d_Ks, stream));
}

void gp_predict(const GaussianProcess *gp, const float *Xs, float *means, float *vars, int m, cudaStream_t stream) {
    int n = gp->n;
    if (n == 0) {
        for (int j = 0; j < m; j++) {
            means[j] = 0.0;
            if (vars)
                vars[j] = matern32lin_k_self(gp->kernel, &Xs[j * gp->dim], gp->dim);
        }
        return;
    }

    float *d_Xte;
    CUDA_CHECK(cudaMallocAsync(&d_Xte, (size_t)m * gp->dim * sizeof(float), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_Xte, Xs, (size_t)m * gp->dim * sizeof(float), cudaMemcpyHostToDevice, stream));

    float *d_means, *d_vars = NULL;
    CUDA_CHECK(cudaMallocAsync(&d_means, (size_t)m * sizeof(float), stream));
    if (vars)
        CUDA_CHECK(cudaMallocAsync(&d_vars, (size_t)m * sizeof(float), stream));

    predict_dev(gp, d_Xte, d_means, d_vars, m, stream);
    CUDA_CHECK(cudaFreeAsync(d_Xte, stream));
    CUDA_CHECK(cudaMemcpyAsync(means, d_means, (size_t)m * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaFreeAsync(d_means, stream));
    if (vars) {
        CUDA_CHECK(cudaMemcpyAsync(vars, d_vars, (size_t)m * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaFreeAsync(d_vars, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void gp_predict_d(const GaussianProcess *gp, const float *d_Xs, float *d_means, float *d_vars, int m,
                  cudaStream_t stream) {
    if (gp->n == 0) {
        CUDA_CHECK(cudaMemsetAsync(d_means, 0, (size_t)m * sizeof(float), stream));
        if (d_vars)
            matern32lin_build_kself_batch(gp->kernel, d_Xs, m, gp->dim, d_vars, stream);
        return;
    }
    predict_dev(gp, d_Xs, d_means, d_vars, m, stream);
}

float gp_marginal_log_likelihood(const GaussianProcess *gp) {
    int n = gp->n;
    if (n == 0)
        return 0.0;

    CUBLAS_CHECK(cublasSetStream(gp->cublas, 0));
    float data_fit;
    CUBLAS_CHECK(cublasSdot(gp->cublas, n, gp->d_y, 1, gp->d_alpha, 1, &data_fit));

    float *d_diag;
    CUDA_CHECK(cudaMalloc(&d_diag, (size_t)n * sizeof(float)));
    gp_k_extract_diag<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(gp->d_L, d_diag, n);
    CUDA_CHECK(cudaGetLastError());

    float *h_diag = (float *)malloc(n * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_diag, d_diag, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_diag));

    float log_det = 0.0;
    for (int i = 0; i < n; i++)
        log_det += log(h_diag[i]);
    free(h_diag);

    return (-0.5 * data_fit - log_det - 0.5 * n * log(2.0 * M_PI)) / n;
}

void gp_mll_grad(const GaussianProcess *gp, float *d_raw_noise, float *kernel_grads, cudaStream_t stream) {
    int n = gp->n;
    if (n == 0) {
        if (d_raw_noise)
            *d_raw_noise = 0.0;
        if (kernel_grads)
            for (int i = 0; i < gp->kernel->n_params; i++)
                kernel_grads[i] = 0.0;
        return;
    }

    CUSOLVER_CHECK(cusolverDnSetStream(gp->cusolver, stream));
    CUBLAS_CHECK(cublasSetStream(gp->cublas, stream));

    float *d_Kinv;
    CUDA_CHECK(cudaMallocAsync(&d_Kinv, (size_t)n * n * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(d_Kinv, 0, (size_t)n * n * sizeof(float), stream));
    gp_k_add_diag<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(d_Kinv, n, 1.0);
    CUDA_CHECK(cudaGetLastError());
    CUSOLVER_CHECK(cusolverDnSpotrs(gp->cusolver, CUBLAS_FILL_MODE_LOWER, n, n, gp->d_L, n, d_Kinv, n, gp->d_info));

    if (d_raw_noise) {
        float term1, term2;
        CUBLAS_CHECK(cublasSdot(gp->cublas, n, gp->d_alpha, 1, gp->d_alpha, 1, &term1));
        CUBLAS_CHECK(cublasSasum(gp->cublas, n, d_Kinv, n + 1, &term2));
        *d_raw_noise = 0.5 * (term1 - term2) * softplus_grad(gp->raw_noise) / n;
    }

    if (kernel_grads) {
        for (int i = 0; i < gp->kernel->n_params; i++)
            kernel_grads[i] = 0.0;
        matern32lin_mll_grad(gp->kernel, gp->d_X, n, gp->dim, gp->cublas, stream, gp->d_alpha, d_Kinv,
                             kernel_grads);
        for (int i = 0; i < gp->kernel->n_params; i++)
            kernel_grads[i] /= n;
    }
    CUDA_CHECK(cudaFreeAsync(d_Kinv, stream));
}

static GPKernel *kernel_from_tag(const char *tag, float *rp, int np) {
    if (memcmp(tag, GP_KERNEL_TAG_MATERN32_LINEAR, 4) != 0) {
        return NULL;
    }
    GPKernel *k = gp_kernel_matern32_linear(np - 2, 1.0, 1.0, 1.0);
    memcpy(k->raw_params, rp, (size_t)np * sizeof(float));
    return k;
}

int gp_save(const GaussianProcess *gp, const char *path) {
    if (gp->n == 0)
        return -1;
    CUDA_CHECK(cudaDeviceSynchronize());
    FILE *f = fopen(path, "wb");
    if (!f)
        return -1;

    int n = gp->n, dim = gp->dim, np = gp->kernel->n_params;
    fwrite("GC04", 1, 4, f);
    fwrite(&dim, sizeof(int), 1, f);
    fwrite(&n, sizeof(int), 1, f);
    fwrite(&gp->raw_noise, sizeof(float), 1, f);
    fwrite(gp->kernel->tag, 1, 4, f);
    fwrite(&np, sizeof(int), 1, f);
    fwrite(gp->kernel->raw_params, sizeof(float), np, f);

    const float *d_ptrs[] = {gp->d_X, gp->d_y, gp->d_L, gp->d_alpha};
    const size_t sizes[] = {(size_t)n * dim, (size_t)n, (size_t)n * n, (size_t)n};
    for (int i = 0; i < 4; i++) {
        float *h = (float *)malloc(sizes[i] * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h, d_ptrs[i], sizes[i] * sizeof(float), cudaMemcpyDeviceToHost));
        fwrite(h, sizeof(float), sizes[i], f);
        free(h);
    }

    fclose(f);
    return 0;
}

GaussianProcess *gp_load(const char *path, int extra_cap) {
    GaussianProcess *gp = NULL;
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;

#define RD(ptr, sz, cnt)                                                                                               \
    if (fread(ptr, sz, cnt, f) != (size_t)(cnt))                                                                       \
    break

    do {
        char magic[4], ktag[4];
        int dim, n, np;
        float rn;

        RD(magic, 1, 4);
        if (memcmp(magic, "GC04", 4) != 0)
            break;
        RD(&dim, sizeof(int), 1);
        RD(&n, sizeof(int), 1);
        RD(&rn, sizeof(float), 1);
        RD(ktag, 1, 4);
        RD(&np, sizeof(int), 1);

        float *rp = (float *)malloc((size_t)np * sizeof(float));
        if (!rp)
            break;
        if (fread(rp, sizeof(float), np, f) != (size_t)np) {
            free(rp);
            break;
        }
        GPKernel *k = kernel_from_tag(ktag, rp, np);
        free(rp);
        if (!k)
            break;

        int cap = n + (extra_cap > 0 ? extra_cap : 0);
        gp = gp_create(dim, cap, k, 1.0);
        gp->raw_noise = rn;
        gp->n = n;

        float *d_ptrs[] = {gp->d_X, gp->d_y, gp->d_L, gp->d_alpha};
        const size_t sizes[] = {(size_t)n * dim, (size_t)n, (size_t)n * n, (size_t)n};
        int ok = 1;
        for (int i = 0; i < 4 && ok; i++) {
            float *h = (float *)malloc(sizes[i] * sizeof(float));
            if (fread(h, sizeof(float), sizes[i], f) != sizes[i]) {
                free(h);
                ok = 0;
            } else {
                CUDA_CHECK(cudaMemcpy(d_ptrs[i], h, sizes[i] * sizeof(float), cudaMemcpyHostToDevice));
                free(h);
            }
        }
        if (!ok)
            break;

        fclose(f);
        return gp;
    } while (0);

#undef RD
    if (gp)
        gp_destroy(gp);
    fclose(f);
    return NULL;
}

// clang-format off
static inline void _curand_check(curandStatus_t s, const char* f, int l) {
    if (s != CURAND_STATUS_SUCCESS) { fprintf(stderr, "cuRAND error %s:%d: %d\n", f, l, (int)s); abort(); }
}
#define CURAND_CHECK(call) _curand_check((call), __FILE__, __LINE__)
// clang-format on

typedef struct {
    int dim;
    int capacity;
    float *d_bounds_min;
    float *d_bounds_max;
    float *d_scales;
    float *d_candidates;
    float *d_pred_y;
    float *d_pred_c;
    float *d_scores;
    float *d_success_prob;
    int *d_best_idx;
    curandStatePhilox4_32_10_t *d_rng;
} ProteinAcq;

typedef struct {
    int best_idx;
    float predicted_score;
    float predicted_cost;
    float rating;
} ProteinAcqResult;

typedef struct {
    float *weights;
    float *d_weights;
    float bias;
    int dim;
    int is_fitted;
} ProteinClassifier;

typedef struct {
    float *params;
    float *scores;
    float *costs;
    int n, cap;
} ProteinObsList;

typedef struct {
    ProteinObsList success;
    ProteinObsList failure;
    int *top_idx;
    int n_top, top_k;
    int dim, cost_dim;
    float min_score, max_score;
    float log_c_min, log_c_max;
} ProteinObs;

typedef struct {
    curandGenerator_t gen;
    int dim;
} Sobol;

__global__ void protein_k_init_rng(curandStatePhilox4_32_10_t *states, int n, unsigned long long seed) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n)
        curand_init(seed, i, 0, &states[i]);
}

__global__ void protein_k_sample(float *__restrict__ candidates, const float *__restrict__ centers,
                                 const float *__restrict__ bounds_min, const float *__restrict__ bounds_max,
                                 const float *__restrict__ scales, curandStatePhilox4_32_10_t *__restrict__ rng,
                                 int n_candidates, int n_centers, int dim, float global_scale, int cost_dim,
                                 float fixed_cost) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= n_candidates)
        return;

    curandStatePhilox4_32_10_t state = rng[i];
    int center = ((int)(curand_uniform(&state) * n_centers)) % n_centers;

    for (int d = 0; d < dim; d++) {
        float s = scales[d] * global_scale;
        float val = s * (2.0f * curand_uniform(&state) - 1.0f) + centers[center * dim + d];
        val = fmaxf(bounds_min[d], fminf(bounds_max[d], val));
        candidates[i * dim + d] = val;
    }

    if (cost_dim >= 0 && !isnan(fixed_cost))
        candidates[i * dim + cost_dim] = fixed_cost;

    rng[i] = state;
}

__global__ void protein_k_score(const float *__restrict__ pred_y_norm, const float *__restrict__ pred_c_norm,
                                const float *__restrict__ success_prob, float *__restrict__ scores, int m,
                                float min_score, float max_score, float log_c_min, float log_c_max, float max_cost,
                                float target_cost, int optimize_dir, int fixed_cost) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= m)
        return;

    float y_norm = pred_y_norm[i];
    float c_norm = pred_c_norm[i];
    float s = (float)optimize_dir * y_norm;

    if (!fixed_cost) {
        float log_c = c_norm * (log_c_max - log_c_min) + log_c_min;
        float c = expf(log_c);
        float mask = (c < max_cost) ? 1.0f : 0.0f;
        float w = 1.0f - fabsf(target_cost - c_norm);
        s *= mask * w;
    }

    if (success_prob)
        s *= success_prob[i];

    scores[i] = s;
}

__global__ void protein_k_argmax(const float *__restrict__ scores, int n, int *out_idx) {
    __shared__ float s_val[BLOCK_SIZE];
    __shared__ int s_idx[BLOCK_SIZE];

    int tid = threadIdx.x;
    float best = -FLT_MAX;
    int best_i = 0;

    for (int i = tid; i < n; i += BLOCK_SIZE) {
        if (scores[i] > best) {
            best = scores[i];
            best_i = i;
        }
    }
    s_val[tid] = best;
    s_idx[tid] = best_i;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s && s_val[tid + s] > s_val[tid]) {
            s_val[tid] = s_val[tid + s];
            s_idx[tid] = s_idx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        *out_idx = s_idx[0];
}

__global__ void protein_k_classifier_predict(const float *__restrict__ X, const float *__restrict__ weights, float bias,
                                             float *__restrict__ probs, int m, int dim) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= m)
        return;
    float z = bias;
    for (int d = 0; d < dim; d++)
        z += X[i * dim + d] * weights[d];
    probs[i] = 1.0f / (1.0f + expf(-z));
}

ProteinClassifier *protein_classifier_create(int dim) {
    ProteinClassifier *clf = (ProteinClassifier *)calloc(1, sizeof(ProteinClassifier));
    clf->dim = dim;
    clf->weights = (float *)calloc((size_t)dim, sizeof(float));
    CUDA_CHECK(cudaMalloc(&clf->d_weights, (size_t)dim * sizeof(float)));
    return clf;
}

void protein_classifier_destroy(ProteinClassifier *clf) {
    if (!clf)
        return;
    free(clf->weights);
    cudaFree(clf->d_weights);
    free(clf);
}

void protein_classifier_fit(ProteinClassifier *clf, const float *X, const int *y, int n, float C_reg, int max_iter) {
    int dim = clf->dim;
    clf->is_fitted = 0;

    int n_pos = 0;
    for (int i = 0; i < n; i++)
        n_pos += y[i];
    int n_neg = n - n_pos;
    if (n_pos == 0 || n_neg == 0)
        return;

    float w_pos = (float)n / (2.0f * (float)n_pos);
    float w_neg = (float)n / (2.0f * (float)n_neg);
    float lambda = 1.0f / C_reg;

    memset(clf->weights, 0, (size_t)dim * sizeof(float));
    clf->bias = 0.0f;
    float *grad = (float *)calloc((size_t)dim, sizeof(float));

    for (int iter = 0; iter < max_iter; iter++) {
        float lr = 0.1f / (1.0f + 0.01f * (float)iter);
        memset(grad, 0, (size_t)dim * sizeof(float));
        float grad_b = 0.0f;

        for (int i = 0; i < n; i++) {
            float z = clf->bias;
            for (int d = 0; d < dim; d++)
                z += X[(size_t)i * dim + d] * clf->weights[d];
            float sig = 1.0f / (1.0f + expf(-z));
            float sw = y[i] ? w_pos : w_neg;
            float err = sw * (sig - (float)y[i]) / (float)n;
            for (int d = 0; d < dim; d++)
                grad[d] += err * X[(size_t)i * dim + d];
            grad_b += err;
        }

        for (int d = 0; d < dim; d++)
            grad[d] += lambda * clf->weights[d];

        for (int d = 0; d < dim; d++)
            clf->weights[d] -= lr * grad[d];
        clf->bias -= lr * grad_b;
    }

    free(grad);
    clf->is_fitted = 1;

    CUDA_CHECK(cudaMemcpy(clf->d_weights, clf->weights, (size_t)dim * sizeof(float), cudaMemcpyHostToDevice));
}

void protein_classifier_predict_d(const ProteinClassifier *clf, const float *d_X, float *d_probs, int m,
                                  cudaStream_t stream) {
    protein_k_classifier_predict<<<(m + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        d_X, clf->d_weights, clf->bias, d_probs, m, clf->dim);
    CUDA_CHECK(cudaGetLastError());
}

ProteinAcq *protein_acq_create(int dim, int capacity, const Space *spaces, unsigned long long rng_seed) {
    ProteinAcq *acq = (ProteinAcq *)calloc(1, sizeof(ProteinAcq));
    acq->dim = dim;
    acq->capacity = capacity;

    float *bounds = (float *)malloc((size_t)3 * dim * sizeof(float));
    float *bounds_min = bounds;
    float *bounds_max = bounds + dim;
    float *scales = bounds + 2 * dim;
    for (int i = 0; i < dim; i++) {
        bounds_min[i] = spaces[i].norm_min;
        bounds_max[i] = spaces[i].norm_max;
        scales[i] = spaces[i].scale;
    }

    CUDA_CHECK(cudaMalloc(&acq->d_bounds_min, (size_t)dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&acq->d_bounds_max, (size_t)dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&acq->d_scales, (size_t)dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(acq->d_bounds_min, bounds_min, (size_t)dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(acq->d_bounds_max, bounds_max, (size_t)dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(acq->d_scales, scales, (size_t)dim * sizeof(float), cudaMemcpyHostToDevice));
    free(bounds);

    CUDA_CHECK(cudaMalloc(&acq->d_candidates, (size_t)capacity * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&acq->d_pred_y, (size_t)capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&acq->d_pred_c, (size_t)capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&acq->d_scores, (size_t)capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&acq->d_success_prob, (size_t)capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&acq->d_best_idx, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&acq->d_rng, (size_t)capacity * sizeof(curandStatePhilox4_32_10_t)));

    protein_k_init_rng<<<(capacity + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(acq->d_rng, capacity, rng_seed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return acq;
}

void protein_acq_destroy(ProteinAcq *acq) {
    if (!acq)
        return;
    void *ptrs[] = {acq->d_bounds_min, acq->d_bounds_max, acq->d_scales,       acq->d_candidates, acq->d_pred_y,
                    acq->d_pred_c,     acq->d_scores,     acq->d_success_prob, acq->d_best_idx,   acq->d_rng};
    for (int i = 0; i < 10; i++)
        cudaFree(ptrs[i]);
    free(acq);
}

int protein_acq_sample(ProteinAcq *acq, const float *centers, int n_centers, int n_total, float global_scale,
                       int cost_dim, float fixed_cost_norm, float dedup_threshold, cudaStream_t stream) {
    int dim = acq->dim;
    if (n_total > acq->capacity)
        n_total = acq->capacity;

    float *d_centers;
    CUDA_CHECK(cudaMallocAsync(&d_centers, (size_t)n_centers * dim * sizeof(float), stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_centers, centers, (size_t)n_centers * dim * sizeof(float), cudaMemcpyHostToDevice, stream));

    protein_k_sample<<<(n_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        acq->d_candidates, d_centers, acq->d_bounds_min, acq->d_bounds_max, acq->d_scales, acq->d_rng, n_total,
        n_centers, dim, global_scale, cost_dim, fixed_cost_norm);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFreeAsync(d_centers, stream));

    if (dedup_threshold <= 0.0f) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return n_total;
    }

    float *h_cands = (float *)malloc((size_t)n_total * dim * sizeof(float));
    CUDA_CHECK(cudaMemcpyAsync(h_cands, acq->d_candidates, (size_t)n_total * dim * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int *kept = (int *)malloc((size_t)n_total * sizeof(int));
    int n_kept = gp_filter_near_duplicates(h_cands, n_total, dim, dedup_threshold, kept);

    if (n_kept > 0 && n_kept < n_total) {
        float *h_compact = (float *)malloc((size_t)n_kept * dim * sizeof(float));
        for (int i = 0; i < n_kept; i++)
            memcpy(&h_compact[(size_t)i * dim], &h_cands[(size_t)kept[i] * dim], (size_t)dim * sizeof(float));
        CUDA_CHECK(cudaMemcpyAsync(acq->d_candidates, h_compact, (size_t)n_kept * dim * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        free(h_compact);
    }

    free(h_cands);
    free(kept);
    return n_kept;
}

ProteinAcqResult protein_acq_suggest(ProteinAcq *acq, int m, const GaussianProcess *gp_score,
                                     const GaussianProcess *gp_cost, float min_score, float max_score, float log_c_min,
                                     float log_c_max, float max_suggestion_cost, float target_cost_ratio,
                                     int optimize_direction, int fixed_cost, const float *d_success_prob,
                                     cudaStream_t stream) {
    gp_predict_d(gp_score, acq->d_candidates, acq->d_pred_y, NULL, m, stream);
    gp_predict_d(gp_cost, acq->d_candidates, acq->d_pred_c, NULL, m, stream);

    protein_k_score<<<(m + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        acq->d_pred_y, acq->d_pred_c, d_success_prob, acq->d_scores, m, min_score, max_score, log_c_min, log_c_max,
        max_suggestion_cost, target_cost_ratio, optimize_direction, fixed_cost);
    CUDA_CHECK(cudaGetLastError());

    protein_k_argmax<<<1, BLOCK_SIZE, 0, stream>>>(acq->d_scores, m, acq->d_best_idx);
    CUDA_CHECK(cudaGetLastError());

    int best;
    CUDA_CHECK(cudaMemcpyAsync(&best, acq->d_best_idx, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float y_norm, c_norm, rating;
    CUDA_CHECK(cudaMemcpy(&y_norm, &acq->d_pred_y[best], sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&c_norm, &acq->d_pred_c[best], sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&rating, &acq->d_scores[best], sizeof(float), cudaMemcpyDeviceToHost));

    return (ProteinAcqResult){
        .best_idx = best,
        .predicted_score = y_norm * (max_score - min_score) + min_score,
        .predicted_cost = expf(c_norm * (log_c_max - log_c_min) + log_c_min),
        .rating = rating,
    };
}

#define NOISE_PRIOR_MU (-4.60517f)
#define NOISE_PRIOR_SIGMA 0.5f

typedef struct {
    float *m, *v, *v_max;
    float lr, beta1, beta2, eps;
    int t, n;
} Adam;

Adam *adam_create(int n_kernel_params, float lr) {
    int n = n_kernel_params + 1;
    Adam *opt = (Adam *)calloc(1, sizeof(Adam));
    opt->n = n;
    opt->lr = lr;
    opt->beta1 = 0.9f;
    opt->beta2 = 0.999f;
    opt->eps = 1e-8f;
    opt->m = (float *)calloc((size_t)n, sizeof(float));
    opt->v = (float *)calloc((size_t)n, sizeof(float));
    opt->v_max = (float *)calloc((size_t)n, sizeof(float));
    return opt;
}

void adam_destroy(Adam *opt) {
    if (!opt)
        return;
    free(opt->m);
    free(opt->v);
    free(opt->v_max);
    free(opt);
}

void adam_reset(Adam *opt) {
    memset(opt->m, 0, (size_t)opt->n * sizeof(float));
    memset(opt->v, 0, (size_t)opt->n * sizeof(float));
    memset(opt->v_max, 0, (size_t)opt->n * sizeof(float));
    opt->t = 0;
}

static float noise_log_prior_grad(float raw_noise) {
    float sig = softplus_grad(raw_noise);
    float noise = softplus(raw_noise);
    float log_noise = logf(noise);
    float d_log_p = (-(log_noise - NOISE_PRIOR_MU) / (NOISE_PRIOR_SIGMA * NOISE_PRIOR_SIGMA) - 1.0f) * sig / noise;
    float d_log_jac = 1.0f - sig;
    return d_log_p + d_log_jac;
}

float protein_train_gp(GaussianProcess *gp, Adam *opt, const float *X, const float *y, int n_data, int training_iter,
                       cudaStream_t stream) {
    int rc = gp_fit(gp, X, y, n_data, stream);
    if (rc != 0)
        return 0.0f;

    int n_kp = gp->kernel->n_params;
    float *kg = (float *)malloc((size_t)n_kp * sizeof(float));
    float loss = 0.0f;

    for (int iter = 0; iter < training_iter; iter++) {
        rc = gp_recompute(gp, stream);
        if (rc != 0)
            break;

        float mll = gp_marginal_log_likelihood(gp);
        float d_noise;
        gp_mll_grad(gp, &d_noise, kg, stream);
        cudaStreamSynchronize(stream);

        float np_grad = noise_log_prior_grad(gp->raw_noise);

        float noise_val = softplus(gp->raw_noise);
        float ln_noise = logf(noise_val);
        float lp =
            -0.5f * powf((ln_noise - NOISE_PRIOR_MU) / NOISE_PRIOR_SIGMA, 2.0f) - ln_noise - logf(NOISE_PRIOR_SIGMA);
        float lj = -log1pf(expf(-gp->raw_noise));
        loss = -mll - (lp + lj);

        opt->t++;
        float bc1 = 1.0f - powf(opt->beta1, (float)opt->t);
        float bc2 = 1.0f - powf(opt->beta2, (float)opt->t);

        for (int i = 0; i < opt->n; i++) {
            float g = (i < n_kp) ? -kg[i] : -d_noise - np_grad;
            opt->m[i] = opt->beta1 * opt->m[i] + (1.0f - opt->beta1) * g;
            opt->v[i] = opt->beta2 * opt->v[i] + (1.0f - opt->beta2) * g * g;
            float m_hat = opt->m[i] / bc1;
            float v_hat = opt->v[i] / bc2;
            if (v_hat > opt->v_max[i])
                opt->v_max[i] = v_hat;
            float step = opt->lr * m_hat / (sqrtf(opt->v_max[i]) + opt->eps);
            if (i < n_kp)
                gp->kernel->raw_params[i] -= step;
            else
                gp->raw_noise -= step;
        }
    }

    gp_recompute(gp, stream);
    free(kg);
    return loss;
}

Sobol *sobol_create(int dim, unsigned long long seed) {
    Sobol *sob = (Sobol *)calloc(1, sizeof(Sobol));
    sob->dim = dim;
    CURAND_CHECK(curandCreateGeneratorHost(&sob->gen, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32));
    CURAND_CHECK(curandSetQuasiRandomGeneratorDimensions(sob->gen, dim));
    CURAND_CHECK(curandSetGeneratorOffset(sob->gen, seed));
    return sob;
}

void sobol_destroy(Sobol *sob) {
    if (!sob)
        return;
    curandDestroyGenerator(sob->gen);
    free(sob);
}

ProteinObs *protein_obs_create(int dim, int success_cap, int failure_cap, int top_k, int cost_dim) {
    ProteinObs *obs = (ProteinObs *)calloc(1, sizeof(ProteinObs));
    obs->dim = dim;
    obs->cost_dim = cost_dim;
    obs->top_k = top_k;
    obs->top_idx = (int *)malloc((size_t)top_k * sizeof(int));
    obs->min_score = FLT_MAX;
    obs->max_score = -FLT_MAX;
    obs->log_c_min = FLT_MAX;
    obs->log_c_max = -FLT_MAX;
    obs->success.params = (float *)calloc((size_t)success_cap * dim, sizeof(float));
    obs->success.scores = (float *)calloc((size_t)success_cap, sizeof(float));
    obs->success.costs = (float *)calloc((size_t)success_cap, sizeof(float));
    obs->success.cap = success_cap;
    obs->failure.params = (float *)calloc((size_t)failure_cap * dim, sizeof(float));
    obs->failure.scores = (float *)calloc((size_t)failure_cap, sizeof(float));
    obs->failure.costs = (float *)calloc((size_t)failure_cap, sizeof(float));
    obs->failure.cap = failure_cap;
    return obs;
}

void protein_obs_destroy(ProteinObs *obs) {
    if (!obs)
        return;
    free(obs->success.params);
    free(obs->success.scores);
    free(obs->success.costs);
    free(obs->failure.params);
    free(obs->failure.scores);
    free(obs->failure.costs);
    free(obs->top_idx);
    free(obs);
}

void protein_obs_add(ProteinObs *obs, const float *params, float score, float cost, int is_failure) {
    int dim = obs->dim;

    if (is_failure || !isfinite(score) || isnan(score)) {
        ProteinObsList *f = &obs->failure;
        if (f->n < f->cap) {
            int i = f->n++;
            memcpy(&f->params[(size_t)i * dim], params, (size_t)dim * sizeof(float));
            f->scores[i] = score;
            f->costs[i] = cost;
        }
        return;
    }

    ProteinObsList *s = &obs->success;

    for (int i = 0; i < s->n; i++) {
        float dist2 = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff = params[d] - s->params[(size_t)i * dim + d];
            dist2 += diff * diff;
        }
        if (sqrtf(dist2) < PROTEIN_EPSILON) {
            memcpy(&s->params[(size_t)i * dim], params, (size_t)dim * sizeof(float));
            s->scores[i] = score;
            s->costs[i] = cost;
            return;
        }
    }

    if (obs->cost_dim >= 0 && params[obs->cost_dim] <= -1.0f)
        return;

    if (s->n >= s->cap)
        return;

    int idx = s->n++;
    memcpy(&s->params[(size_t)idx * dim], params, (size_t)dim * sizeof(float));
    s->scores[idx] = score;
    s->costs[idx] = cost;

    if (obs->n_top < obs->top_k)
        obs->top_idx[obs->n_top++] = idx;
    else if (score > s->scores[obs->top_idx[obs->n_top - 1]])
        obs->top_idx[obs->n_top - 1] = idx;
    else
        return;

    for (int j = obs->n_top - 1; j > 0; j--) {
        if (s->scores[obs->top_idx[j]] > s->scores[obs->top_idx[j - 1]]) {
            int tmp = obs->top_idx[j];
            obs->top_idx[j] = obs->top_idx[j - 1];
            obs->top_idx[j - 1] = tmp;
        } else
            break;
    }
}

static void protein_obs_extract(const float *ext, int ext_dim, int dim, int idx, float *out_params, float *out_score,
                                float *out_cost) {
    memcpy(out_params, &ext[(size_t)idx * ext_dim], (size_t)dim * sizeof(float));
    *out_score = ext[(size_t)idx * ext_dim + dim];
    *out_cost = ext[(size_t)idx * ext_dim + dim + 1];
}

int protein_obs_sample_for_gp(ProteinObs *obs, int max_size, float recent_ratio, float *out_params, float *out_scores,
                              float *out_costs) {
    ProteinObsList *s = &obs->success;
    int dim = obs->dim;
    if (s->n == 0)
        return 0;

    obs->min_score = s->scores[0];
    obs->max_score = s->scores[0];
    for (int i = 1; i < s->n; i++) {
        if (s->scores[i] < obs->min_score)
            obs->min_score = s->scores[i];
        if (s->scores[i] > obs->max_score)
            obs->max_score = s->scores[i];
    }

    float *log_c_buf = (float *)malloc((size_t)s->n * sizeof(float));
    for (int i = 0; i < s->n; i++)
        log_c_buf[i] = logf(fmaxf(s->costs[i], PROTEIN_EPSILON));
    obs->log_c_min = log_c_buf[0];
    for (int i = 1; i < s->n; i++)
        if (log_c_buf[i] < obs->log_c_min)
            obs->log_c_min = log_c_buf[i];
    obs->log_c_max = protein_quantile(log_c_buf, s->n, PROTEIN_COST_QUANTILE);
    free(log_c_buf);

    ProteinObsList *f = &obs->failure;
    int use_failures = (s->n < PROTEIN_MIN_OBS_NO_FAIL && f->n > 0);
    int combined_n = use_failures ? f->n + s->n : s->n;
    int ext_dim = dim + 2;
    float *ext = (float *)malloc((size_t)combined_n * ext_dim * sizeof(float));

    int ci = 0;
    if (use_failures) {
        for (int i = 0; i < f->n; i++) {
            memcpy(&ext[(size_t)ci * ext_dim], &f->params[(size_t)i * dim], (size_t)dim * sizeof(float));
            ext[(size_t)ci * ext_dim + dim] = obs->min_score;
            ext[(size_t)ci * ext_dim + dim + 1] = f->costs[i];
            ci++;
        }
    }
    for (int i = 0; i < s->n; i++) {
        memcpy(&ext[(size_t)ci * ext_dim], &s->params[(size_t)i * dim], (size_t)dim * sizeof(float));
        ext[(size_t)ci * ext_dim + dim] = s->scores[i];
        ext[(size_t)ci * ext_dim + dim + 1] = s->costs[i];
        ci++;
    }

    int *kept = (int *)malloc((size_t)combined_n * sizeof(int));
    int n_kept = gp_filter_near_duplicates(ext, combined_n, ext_dim, PROTEIN_EPSILON, kept);

    if (n_kept <= max_size) {
        for (int i = 0; i < n_kept; i++)
            protein_obs_extract(ext, ext_dim, dim, kept[i], &out_params[(size_t)i * dim], &out_scores[i],
                                &out_costs[i]);
        free(kept);
        free(ext);
        return n_kept;
    }

    int recent_size = (int)(recent_ratio * (float)max_size);
    int older_size = n_kept - recent_size;
    int num_sample = max_size - recent_size;

    int *sample_idx = (int *)malloc((size_t)older_size * sizeof(int));
    for (int i = 0; i < older_size; i++)
        sample_idx[i] = i;
    for (int i = 0; i < num_sample && i < older_size; i++) {
        int j = i + rand() % (older_size - i);
        int tmp = sample_idx[i];
        sample_idx[i] = sample_idx[j];
        sample_idx[j] = tmp;
    }

    int out_n = 0;
    for (int i = 0; i < num_sample && i < older_size; i++) {
        protein_obs_extract(ext, ext_dim, dim, kept[sample_idx[i]], &out_params[(size_t)out_n * dim],
                            &out_scores[out_n], &out_costs[out_n]);
        out_n++;
    }
    for (int i = n_kept - recent_size; i < n_kept; i++) {
        protein_obs_extract(ext, ext_dim, dim, kept[i], &out_params[(size_t)out_n * dim], &out_scores[out_n],
                            &out_costs[out_n]);
        out_n++;
    }

    free(sample_idx);
    free(kept);
    free(ext);
    return out_n;
}

typedef struct {
    float score_loss;
    float cost_loss;
    float predicted_score;
    float predicted_cost;
    float rating;
    int n_pareto;
    int n_gp_obs;
    int n_candidates;
    int is_random;
} ProteinSweepInfo;

typedef struct {
    SweepSpace *space;
    ProteinObs *obs;
    ProteinAcq *acq;
    ProteinCostModel cost_model;
    ProteinClassifier *clf;
    Sobol *sobol;
    GaussianProcess *gp_score;
    GaussianProcess *gp_cost;
    Adam *opt_score;
    Adam *opt_cost;
    cudaStream_t stream;
    int suggestion_idx;
    int num_random_samples;
    int suggestions_per_pareto;
    int gp_training_iter;
    int optimizer_reset_frequency;
    int gp_max_obs;
    int use_success_prob;
    int prune_pareto;
    int use_logit;
    float global_search_scale;
    float max_suggestion_cost;
    float expansion_rate;
    float cost_random_suggestion;
    float upper_cost_threshold;
    float ratio_pool[PROTEIN_NUM_COST_RATIOS];
    int pool_remaining;
    float running_buf[PROTEIN_RUNNING_BUF_CAP];
    int running_pos;
    int running_len;
    float *gp_train_params;
    float *gp_train_y;
    float *gp_train_c;
    int *pareto_buf;
    int *pruned_buf;
    float *centers_buf;
    int obs_cap, centers_cap;
} ProteinSweep;

ProteinSweep *protein_sweep_create(SweepSpace *space, int num_random_samples, int suggestions_per_pareto,
                                   int gp_training_iter, float gp_learning_rate, int optimizer_reset_frequency,
                                   int gp_max_obs, int infer_batch_size, int use_success_prob, int prune_pareto,
                                   int use_logit, float global_search_scale, float max_suggestion_cost,
                                   float expansion_rate, float cost_random_suggestion, float early_stop_quantile,
                                   int success_cap, int failure_cap, int top_k, unsigned long long rng_seed) {
    int dim = space->num;
    ProteinSweep *sw = (ProteinSweep *)calloc(1, sizeof(ProteinSweep));
    sw->space = space;
    sw->num_random_samples = num_random_samples;
    sw->suggestions_per_pareto = suggestions_per_pareto;
    sw->gp_training_iter = gp_training_iter;
    sw->optimizer_reset_frequency = optimizer_reset_frequency;
    sw->gp_max_obs = gp_max_obs;
    sw->use_success_prob = use_success_prob;
    sw->prune_pareto = prune_pareto;
    sw->use_logit = use_logit;
    sw->global_search_scale = global_search_scale;
    sw->max_suggestion_cost = max_suggestion_cost;
    sw->expansion_rate = expansion_rate;
    sw->cost_random_suggestion = cost_random_suggestion;
    sw->upper_cost_threshold = -FLT_MAX;

    static const float default_ratios[PROTEIN_NUM_COST_RATIOS] = {0.16f, 0.32f, 0.48f, 0.64f, 0.8f, 1.0f};
    memcpy(sw->ratio_pool, default_ratios, sizeof(default_ratios));

    sw->obs = protein_obs_create(dim, success_cap, failure_cap, top_k, space->cost_idx);

    int max_centers = success_cap + 3 * top_k;
    int acq_cap = max_centers * suggestions_per_pareto;
    if (acq_cap < infer_batch_size)
        acq_cap = infer_batch_size;
    if (acq_cap > PROTEIN_ACQ_MAX_CAP)
        acq_cap = PROTEIN_ACQ_MAX_CAP;
    sw->acq = protein_acq_create(dim, acq_cap, space->spaces, rng_seed);
    sw->cost_model.quantile = early_stop_quantile;
    sw->cost_model.min_samples = 30;
    sw->clf = protein_classifier_create(dim);
    sw->sobol = sobol_create(dim, rng_seed);

    sw->gp_score = gp_create(dim, gp_max_obs, gp_kernel_matern32_linear(dim, 1.0f, 1.0f, 1.0f), 0.01f);
    sw->gp_cost = gp_create(dim, gp_max_obs, gp_kernel_matern32_linear(dim, 1.0f, 1.0f, 1.0f), 0.01f);
    sw->opt_score = adam_create(sw->gp_score->kernel->n_params, gp_learning_rate);
    sw->opt_cost = adam_create(sw->gp_cost->kernel->n_params, gp_learning_rate);

    sw->obs_cap = success_cap;
    sw->centers_cap = max_centers;
    sw->gp_train_params = (float *)malloc((size_t)gp_max_obs * dim * sizeof(float));
    sw->gp_train_y = (float *)malloc((size_t)gp_max_obs * sizeof(float));
    sw->gp_train_c = (float *)malloc((size_t)gp_max_obs * sizeof(float));
    sw->pareto_buf = (int *)malloc((size_t)success_cap * sizeof(int));
    sw->pruned_buf = (int *)malloc((size_t)success_cap * sizeof(int));
    sw->centers_buf = (float *)malloc((size_t)max_centers * dim * sizeof(float));

    CUDA_CHECK(cudaStreamCreate(&sw->stream));
    return sw;
}

void protein_sweep_destroy(ProteinSweep *sw) {
    if (!sw)
        return;
    protein_obs_destroy(sw->obs);
    protein_acq_destroy(sw->acq);
    protein_classifier_destroy(sw->clf);
    sobol_destroy(sw->sobol);
    gp_destroy(sw->gp_score);
    gp_destroy(sw->gp_cost);
    adam_destroy(sw->opt_score);
    adam_destroy(sw->opt_cost);
    if (sw->stream)
        cudaStreamDestroy(sw->stream);
    free(sw->gp_train_params);
    free(sw->gp_train_y);
    free(sw->gp_train_c);
    free(sw->pareto_buf);
    free(sw->pruned_buf);
    free(sw->centers_buf);
    free(sw);
}

void protein_sweep_observe(ProteinSweep *sw, const float *norm_params, float score, float cost, int is_failure) {
    if (sw->use_logit)
        score = protein_logit_transform(score);
    protein_obs_add(sw->obs, norm_params, score, cost, is_failure);
}

void protein_sweep_add_running(ProteinSweep *sw, float val) {
    sw->running_buf[sw->running_pos] = val;
    sw->running_pos = (sw->running_pos + 1) % PROTEIN_RUNNING_BUF_CAP;
    if (sw->running_len < PROTEIN_RUNNING_BUF_CAP)
        sw->running_len++;
}

float protein_sweep_running_mean(const ProteinSweep *sw) {
    if (sw->running_len == 0)
        return 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < sw->running_len; i++)
        sum += sw->running_buf[i];
    return sum / (float)sw->running_len;
}

float protein_sweep_get_threshold(const ProteinSweep *sw, float cost) {
    return protein_cost_model_threshold(&sw->cost_model, cost, 0.3f, 10.0f);
}

int protein_sweep_should_stop(const ProteinSweep *sw, float score, float cost) {
    float threshold = protein_sweep_get_threshold(sw, cost);
    if (sw->use_logit)
        score = protein_logit_transform(score);
    return score < threshold;
}

static void protein_sobol_fallback(ProteinSweep *sw, float *out, int is_fixed_cost, float fixed_cost_norm) {
    int dim = sw->space->num;
    int cost_dim = sw->space->cost_idx;
    CURAND_CHECK(curandGenerateUniform(sw->sobol->gen, out, sw->sobol->dim));
    for (int d = 0; d < dim; d++)
        out[d] = 2.0f * out[d] - 1.0f;
    if (is_fixed_cost && cost_dim >= 0) {
        out[cost_dim] = fixed_cost_norm;
    } else if (cost_dim >= 0) {
        float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
        float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
        float noise = 0.1f * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
        out[cost_dim] = fmaxf(-1.0f, fminf(1.0f, sw->cost_random_suggestion + noise));
    }
}

ProteinSweepInfo protein_sweep_suggest(ProteinSweep *sw, float *out, float fixed_cost_norm) {
    ProteinSweepInfo info = {0};
    sw->suggestion_idx++;

    int dim = sw->space->num;
    int cost_dim = sw->space->cost_idx;
    int is_fixed_cost = !isnan(fixed_cost_norm);

    ProteinObsList *s = &sw->obs->success;
    if (sw->suggestion_idx <= sw->num_random_samples || s->n == 0) {
        protein_sobol_fallback(sw, out, is_fixed_cost, fixed_cost_norm);
        info.is_random = 1;
        return info;
    }

    int n_gp =
        protein_obs_sample_for_gp(sw->obs, sw->gp_max_obs, 0.5f, sw->gp_train_params, sw->gp_train_y, sw->gp_train_c);

    float min_score = sw->obs->min_score;
    float max_score = sw->obs->max_score;
    float log_c_min = sw->obs->log_c_min;
    float log_c_max = sw->obs->log_c_max;
    float score_range = fabsf(max_score - min_score) + PROTEIN_EPSILON;
    float cost_range = log_c_max - log_c_min + PROTEIN_EPSILON;

    float *y_norm = sw->gp_train_y;
    float *c_norm = sw->gp_train_c;
    for (int i = 0; i < n_gp; i++) {
        y_norm[i] = (y_norm[i] - min_score) / score_range;
        float lc = logf(fmaxf(c_norm[i], PROTEIN_EPSILON));
        c_norm[i] = (lc - log_c_min) / cost_range;
    }

    float score_loss = protein_train_gp(sw->gp_score, sw->opt_score, sw->gp_train_params, y_norm, n_gp,
                                        sw->gp_training_iter, sw->stream);
    float cost_loss = protein_train_gp(sw->gp_cost, sw->opt_cost, sw->gp_train_params, c_norm, n_gp,
                                       sw->gp_training_iter, sw->stream);

    if (sw->optimizer_reset_frequency > 0 && sw->suggestion_idx % sw->optimizer_reset_frequency == 0) {
        adam_reset(sw->opt_score);
        adam_reset(sw->opt_cost);
    }

    int n_pareto = protein_pareto_front(s->scores, s->costs, s->n, sw->pareto_buf);

    memcpy(sw->pruned_buf, sw->pareto_buf, (size_t)n_pareto * sizeof(int));
    int n_pruned = protein_prune_pareto(s->scores, s->costs, sw->pruned_buf, n_pareto, 0.5f, 0.98f);

    if (n_pruned > 0) {
        float pruned_max_cost = s->costs[sw->pruned_buf[n_pruned - 1]];
        if (sw->upper_cost_threshold < 0)
            sw->upper_cost_threshold = pruned_max_cost;
        else if (sw->upper_cost_threshold < pruned_max_cost)
            sw->upper_cost_threshold *= PROTEIN_COST_GROWTH;
    }
    protein_cost_model_fit(&sw->cost_model, s->scores, s->costs, s->n, sw->upper_cost_threshold);

    int *use_pareto = sw->prune_pareto ? sw->pruned_buf : sw->pareto_buf;
    int n_use = sw->prune_pareto ? n_pruned : n_pareto;

    int n_top = sw->obs->n_top;
    int n_centers = protein_build_search_centers(s->params, dim, use_pareto, n_use, sw->obs->top_idx, n_top, cost_dim,
                                                 sw->centers_buf);

    float target_cost = 0.0f;
    if (!is_fixed_cost) {
        target_cost = protein_sample_target_cost(sw->ratio_pool, &sw->pool_remaining, PROTEIN_NUM_COST_RATIOS,
                                                 sw->expansion_rate);
    }

    int n_total = n_centers * sw->suggestions_per_pareto;
    int n_cands = protein_acq_sample(sw->acq, sw->centers_buf, n_centers, n_total, sw->global_search_scale, cost_dim,
                                     is_fixed_cost ? fixed_cost_norm : NAN, PROTEIN_EPSILON, sw->stream);

    if (n_cands == 0) {
        protein_sobol_fallback(sw, out, is_fixed_cost, fixed_cost_norm);
        info.is_random = 1;
        return info;
    }

    float *d_success_prob = NULL;
    if (sw->use_success_prob && s->n > 9 && sw->obs->failure.n > 9) {
        int n_s = s->n, n_f = sw->obs->failure.n;
        int n_clf = n_s + n_f;
        float *X = (float *)malloc((size_t)n_clf * dim * sizeof(float));
        int *y = (int *)malloc((size_t)n_clf * sizeof(int));
        memcpy(X, s->params, (size_t)n_s * dim * sizeof(float));
        memcpy(X + (size_t)n_s * dim, sw->obs->failure.params, (size_t)n_f * dim * sizeof(float));
        for (int i = 0; i < n_s; i++)
            y[i] = 1;
        for (int i = 0; i < n_f; i++)
            y[n_s + i] = 0;
        protein_classifier_fit(sw->clf, X, y, n_clf, 1.0f, PROTEIN_CLF_ITERS);
        free(X);
        free(y);

        if (sw->clf->is_fitted) {
            protein_classifier_predict_d(sw->clf, sw->acq->d_candidates, sw->acq->d_success_prob, n_cands, sw->stream);
            d_success_prob = sw->acq->d_success_prob;
        }
    }

    ProteinAcqResult acq_result =
        protein_acq_suggest(sw->acq, n_cands, sw->gp_score, sw->gp_cost, min_score, max_score, log_c_min, log_c_max,
                            sw->max_suggestion_cost, target_cost, sw->space->optimize_direction, is_fixed_cost,
                            d_success_prob, sw->stream);

    CUDA_CHECK(cudaMemcpyAsync(out, &sw->acq->d_candidates[(size_t)acq_result.best_idx * sw->acq->dim],
                               (size_t)sw->acq->dim * sizeof(float), cudaMemcpyDeviceToHost, sw->stream));
    CUDA_CHECK(cudaStreamSynchronize(sw->stream));

    info.score_loss = score_loss;
    info.cost_loss = cost_loss;
    info.predicted_score = acq_result.predicted_score;
    info.predicted_cost = acq_result.predicted_cost;
    info.rating = acq_result.rating;
    info.n_pareto = n_use;
    info.n_gp_obs = n_gp;
    info.n_candidates = n_cands;
    return info;
}

#endif // PROTEIN_CU
