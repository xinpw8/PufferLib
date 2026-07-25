#ifndef PUFFERLIB_ALGO_CU
#define PUFFERLIB_ALGO_CU

#include <cub/block/block_scan.cuh>

// --- GEMM (needs batch_size/ndim, CUBLAS_PRECISION* from pufferl) ---
#ifndef CUBLAS_WS_BYTES
#define CUBLAS_WS_BYTES (32 * 1024 * 1024)
#endif
#ifndef CUBLAS_MATH_MODE
#define CUBLAS_MATH_MODE CUBLAS_DEFAULT_MATH
#endif
#ifndef CUBLAS_GEMM_ALGO
#define CUBLAS_GEMM_ALGO CUBLAS_GEMM_DEFAULT
#endif

thread_local cublasHandle_t g_cublas_handle = nullptr;
thread_local void* g_cublas_workspace = nullptr;
// Side-stream dW (mm_tn) overlaps dX (mm_nn) on main during linear bwd.
thread_local cublasHandle_t g_cublas_dw_handle = nullptr;
thread_local cudaEvent_t g_main_ready = nullptr;
thread_local void* g_cublas_dw_workspace = nullptr;
thread_local cudaStream_t g_dw_stream = nullptr;
thread_local cudaEvent_t g_dw_done = nullptr;

static void cublas_init_one(cublasHandle_t* handle, void** workspace) {
    cublasCreate(handle);
    cudaMalloc(workspace, (size_t)CUBLAS_WS_BYTES);
    cublasSetWorkspace(*handle, *workspace, (size_t)CUBLAS_WS_BYTES);
    cublasSetMathMode(*handle, (cublasMath_t)CUBLAS_MATH_MODE);
}

void cublas_init_handle() {
    cublas_init_one(&g_cublas_handle, &g_cublas_workspace);
    cublas_init_one(&g_cublas_dw_handle, &g_cublas_dw_workspace);
    cudaStreamCreateWithFlags(&g_dw_stream, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&g_dw_done, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&g_main_ready, cudaEventDisableTiming);
}

// Dense row-major GEMM on an explicit handle/stream: C = alpha * op_a(A) @ op_b(B) + beta * C
static void cublasGemmExDenseOn(
        cublasHandle_t handle,
        cublasOperation_t op_a, cublasOperation_t op_b,
        int M, int N, int K, void* A, void* B, void* C,
        cudaStream_t stream, float alpha, float beta) {
    int lda = (op_a == CUBLAS_OP_N) ? K : M;
    int ldb = (op_b == CUBLAS_OP_N) ? N : K;
    cublasSetStream(handle, stream);
    cublasGemmEx(handle, op_b, op_a, N, M, K, &alpha,
        B, CUBLAS_PRECISION, ldb, A, CUBLAS_PRECISION, lda, &beta,
        C, CUBLAS_PRECISION, N, CUBLAS_COMPUTE_PRECISION, (cublasGemmAlgo_t)CUBLAS_GEMM_ALGO);
}

void cublasGemmExDense(
        cublasOperation_t op_a, cublasOperation_t op_b,
        int M, int N, int K, void* A, void* B, void* C,
        cudaStream_t stream, float alpha = 1.0f, float beta = 0.0f) {
    cublasGemmExDenseOn(g_cublas_handle, op_a, op_b, M, N, K, A, B, C, stream, alpha, beta);
}

// out(...,N) = a(...,K) @ b(N,K)^T  — leading dims folded into M
void puf_mm(PrecisionTensor* a, PrecisionTensor* b, PrecisionTensor* out, cudaStream_t stream) {
    int M = batch_size(a->shape) * a->shape[ndim(a->shape)-2];
    int K = a->shape[ndim(a->shape)-1];
    int N = b->shape[ndim(b->shape)-2];
    cublasGemmExDense(CUBLAS_OP_N, CUBLAS_OP_T, M, N, K,
        a->data, b->data, out->data, stream);
}

// out(M,N) = a(...,M)^T @ b(...,N)  — leading dims folded into K
void puf_mm_tn(PrecisionTensor* a, PrecisionTensor* b, PrecisionTensor* out, cudaStream_t stream) {
    int M = a->shape[ndim(a->shape)-1];
    int K = batch_size(a->shape) * a->shape[ndim(a->shape)-2];
    int N = b->shape[ndim(b->shape)-1];
    cublasGemmExDense(CUBLAS_OP_T, CUBLAS_OP_N, M, N, K,
        a->data, b->data, out->data, stream);
}

// out(...,N) = a(...,K) @ b(K,N)  — leading dims folded into M
void puf_mm_nn(PrecisionTensor* a, PrecisionTensor* b, PrecisionTensor* out, cudaStream_t stream) {
    int M = batch_size(a->shape) * a->shape[ndim(a->shape)-2];
    int K = a->shape[ndim(a->shape)-1];
    int N = b->shape[ndim(b->shape)-1];
    cublasGemmExDense(CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
        a->data, b->data, out->data, stream);
}

// dW on side stream after main has produced grad. Per-layer wgrad buffers are
// disjoint so multiple dWs can queue. Join before muon reads weight grads.
void puf_mm_tn_async_after(PrecisionTensor* a, PrecisionTensor* b, PrecisionTensor* out,
        cudaStream_t main_stream) {
    cudaEventRecord(g_main_ready, main_stream);
    cudaStreamWaitEvent(g_dw_stream, g_main_ready, 0);
    int M = a->shape[ndim(a->shape)-1];
    int K = batch_size(a->shape) * a->shape[ndim(a->shape)-2];
    int N = b->shape[ndim(b->shape)-1];
    cublasGemmExDenseOn(g_cublas_dw_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K,
        a->data, b->data, out->data, g_dw_stream, 1.0f, 0.0f);
    cudaEventRecord(g_dw_done, g_dw_stream);
}

void puf_dw_join(cudaStream_t consumer) {
    cudaStreamWaitEvent(consumer, g_dw_done, 0);
}

// Weight init (needs cast, grid_size, numel/ndim from pufferl substrate).
__global__ void uniform_scale_kernel(float* data, float bound, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f * bound - bound;
    }
}

// Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))
void puf_kaiming_init(PrecisionTensor* dst, float gain, ulong seed, cudaStream_t stream) {
    assert(ndim(dst->shape) == 2);
    long rows = dst->shape[0], cols = dst->shape[1];
    assert(rows > 0 && cols > 0);
    long n = rows * cols;
    float bound = gain / sqrtf((float)cols);
    float* buf;
    cudaMalloc(&buf, n * sizeof(float));
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniform(gen, buf, n);
    curandDestroyGenerator(gen);
    uniform_scale_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(buf, bound, n);
    cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(dst->data, buf, n);
    cudaFree(buf);
}

__device__ __forceinline__ float sigmoid(float x) {
    float z = expf(-fabsf(x));
    return x >= 0.0f ? 1.0f / (1.0f + z) : z / (1.0f + z);
}

__device__ __forceinline__ float lerp(float a, float b, float w) {
    float diff = b - a;
    return (fabsf(w) < 0.5f) ? a + w * diff : b - diff * (1.0f - w);
}

// PufferNet model API + architecture
// Writing custom nets in 4.0+ requires a fair bit of code because you are
// responsible for defining your own activation and gradient buffers.
// You usually only ever need a custom Encoder.
typedef void (*init_weights_fn)(void* weights, ulong* seed, cudaStream_t stream);
typedef void (*reg_params_fn)(void* weights, Allocator* alloc);
typedef void (*reg_train_fn)(void* weights, void* buf, Allocator* acts, Allocator* grads, int B_TT);
typedef void (*reg_rollout_fn)(void* weights, void* buf, Allocator* alloc, int B);
typedef void* (*create_weights_fn)(void* self);
typedef PrecisionTensor (*forward_fn)(void* weights, void* activations, PrecisionTensor input, cudaStream_t stream);
typedef void (*encoder_backward_fn)(void* weights, void* activations,
    PrecisionTensor grad, cudaStream_t stream);
typedef PrecisionTensor (*decoder_backward_fn)(void* weights, void* activations,
    FloatTensor grad_logits, FloatTensor grad_logstd, FloatTensor grad_value, cudaStream_t stream);
typedef PrecisionTensor (*network_forward_fn)(void* weights, PrecisionTensor x,
    PrecisionTensor state, void* activations, cudaStream_t stream);
typedef PrecisionTensor (*network_forward_train_fn)(void* weights, PrecisionTensor x,
    PrecisionTensor state, PrecisionTensor terminals, void* activations, cudaStream_t stream);
typedef PrecisionTensor (*network_backward_fn)(void* weights,
    PrecisionTensor grad, void* activations, cudaStream_t stream);

struct Encoder {
    forward_fn forward;
    encoder_backward_fn backward;
    init_weights_fn init_weights;
    reg_params_fn reg_params;
    reg_train_fn reg_train;
    reg_rollout_fn reg_rollout;
    create_weights_fn create_weights;
    int in_dim, out_dim;
    size_t activation_size;  // sizeof(EncoderActivations) or custom override
};

struct Decoder {
    forward_fn forward;
    decoder_backward_fn backward;
    init_weights_fn init_weights;
    reg_params_fn reg_params;
    reg_train_fn reg_train;
    reg_rollout_fn reg_rollout;
    create_weights_fn create_weights;
    int hidden_dim, output_dim;
    bool continuous;
    size_t activation_size;
};

struct Network {
    network_forward_fn forward;
    network_forward_train_fn forward_train;
    network_backward_fn backward;
    init_weights_fn init_weights;
    reg_params_fn reg_params;
    reg_train_fn reg_train;
    reg_rollout_fn reg_rollout;
    create_weights_fn create_weights;
    int hidden, num_layers, horizon;
};

struct EncoderWeights {
    PrecisionTensor weight;
    int in_dim, out_dim;
};

struct EncoderActivations {
    PrecisionTensor out, saved_input, wgrad_scratch;
};

// Rollout MinGRU step: h = (1-z)*h + z*h_tilde, highway out = s*h + (1-s)*x.
__global__ void mingru_gate(precision_t* out, precision_t* next_state,
        const precision_t* combined, const precision_t* state_in,
        const precision_t* x_in, int H, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * H;
    if (idx >= N) {
        return;
    }

    int b = idx / H;
    int h = idx % H;

    int combined_base = b * 3 * H;
    float hidden = to_float(combined[combined_base + h]);
    float gate = to_float(combined[combined_base + H + h]);
    float proj = to_float(combined[combined_base + 2*H + h]);
    float state = to_float(state_in[idx]);
    float x = to_float(x_in[idx]);

    float z = sigmoid(gate);
    float h_tilde = (hidden >= 0.0f) ? hidden + 0.5f : sigmoid(hidden);
    float h_out = lerp(state, h_tilde, z);
    next_state[idx] = from_float(h_out);

    float s = sigmoid(proj);
    out[idx] = from_float(s * h_out + (1.0f - s) * x);
}

// Train scan buffers. Cache post-reset h inputs; recompute outputs in bwd.
struct PrefixScan {
    precision_t* combined_ptr = nullptr;
    precision_t* state_ptr = nullptr;
    precision_t* input_ptr = nullptr;  // (B, T, H) pre-projection input (highway)
    precision_t* terminals_ptr = nullptr;  // (B, T), reset before step t if terminal[t]
    int B = 0, T = 0, H = 0;
    PrecisionTensor scan_h;  // (B, T, H), recurrent input to each step
    PrecisionTensor out, next_state;
    PrecisionTensor grad_combined, grad_state;
    PrecisionTensor grad_input;
};

// Train scan: match rollout's precision_t recurrent state between steps.
__global__ void mingru_scan_forward_seq(PrefixScan scan) {
    int T_seq = scan.T, H = scan.H, B = scan.B;
    precision_t* __restrict__ out = scan.out.data;
    precision_t* __restrict__ next_state = scan.next_state.data;
    precision_t* __restrict__ scan_h = scan.scan_h.data;
    const precision_t* __restrict__ combined = scan.combined_ptr;
    const precision_t* __restrict__ state = scan.state_ptr;
    const precision_t* __restrict__ input = scan.input_ptr;
    const precision_t* __restrict__ terminals = scan.terminals_ptr;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) {
        return;
    }

    int b = idx / H;
    int h = idx % H;

    int bH = b * H;
    int H3 = 3 * H;
    int H2 = 2 * H;
    int bHT = bH * T_seq;
    const int out_base = bHT + h;
    int cbase = 3 * bHT;

    // h_0
    float h_t = to_float(state[bH + h]);
    int h_base = b * T_seq * H + h;

    const precision_t* __restrict__ combined_h_base = &combined[cbase + h];
    const precision_t* __restrict__ combined_g_base = &combined[cbase + H + h];
    const precision_t* __restrict__ combined_p_base = &combined[cbase + H2 + h];

    int out_curr = out_base;
    int t_offset = 0;

    for (int t = 0; t < T_seq; t++) {
        // terminal[t] and observation[t] are emitted together after the prior step.
        if (terminals != nullptr && to_float(terminals[b * T_seq + t]) != 0.0f) {
            h_t = 0.0f;
        }
        scan_h[h_base + t * H] = from_float(h_t);

        float hidden_val = to_float(__ldg(&combined_h_base[t_offset]));
        float gate_val = to_float(__ldg(&combined_g_base[t_offset]));
        float proj_val = to_float(__ldg(&combined_p_base[t_offset]));
        float x_val = to_float(__ldg(&input[out_base + t * H]));

        // h = (1-z)*h + z*h_tilde  (exact sigmoid; matches prior train log-space)
        float z = sigmoid(gate_val);
        float h_tilde = (hidden_val >= 0.0f) ? hidden_val + 0.5f : sigmoid(hidden_val);
        h_t = lerp(h_t, h_tilde, z);

        float proj_sigmoid = sigmoid(proj_val);
        out[out_curr] = from_float(proj_sigmoid * h_t + (1.0f - proj_sigmoid) * x_val);
        h_t = to_float(from_float(h_t));

        out_curr += H;
        t_offset += H3;
    }

    next_state[bH + h] = from_float(h_t);
}

// Linear reverse scan. Cache is bf16 h only; a,z,h_tilde recomputed from gates in f32.
__global__ void mingru_scan_backward(PrefixScan scan,
        const precision_t* __restrict__ grad_out,
        const precision_t* __restrict__ grad_next_state) {
    int T_seq = scan.T, H = scan.H, B = scan.B;
    precision_t* __restrict__ grad_combined = scan.grad_combined.data;
    precision_t* __restrict__ grad_state = scan.grad_state.data;
    precision_t* __restrict__ grad_input = scan.grad_input.data;
    const precision_t* __restrict__ combined = scan.combined_ptr;
    const precision_t* __restrict__ input = scan.input_ptr;
    const precision_t* __restrict__ terminals = scan.terminals_ptr;
    const precision_t* __restrict__ scan_h = scan.scan_h.data;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) {
        return;
    }

    int b = idx / H;
    int h = idx % H;

    int bHT = b * H * T_seq;
    int cbase = 3 * bHT;
    int H3 = 3 * H;
    int H2 = 2 * H;
    const int state_idx = b * H + h;
    const int out_base = bHT + h;
    const int h_base = b * T_seq * H + h;

    const precision_t* __restrict__ combined_h_base = &combined[cbase + h];
    const precision_t* __restrict__ combined_g_base = &combined[cbase + H + h];
    const precision_t* __restrict__ combined_p_base = &combined[cbase + H2 + h];

    precision_t* __restrict__ grad_combined_h_base = &grad_combined[cbase + h];
    precision_t* __restrict__ grad_combined_g_base = &grad_combined[cbase + H + h];
    precision_t* __restrict__ grad_combined_p_base = &grad_combined[cbase + H2 + h];

    // dh flowing into h_t from the future (and grad_next at t=T).
    float dh = to_float(grad_next_state[state_idx]);

    for (int t = T_seq; t >= 1; --t) {
        int t0 = t - 1;  // 0-based step
        int t_offset = t0 * H3;
        int input_idx = out_base + t0 * H;

        float h_prev = to_float(__ldg(&scan_h[h_base + (t - 1) * H]));

        float hidden_val = to_float(__ldg(&combined_h_base[t_offset]));
        float gate_val = to_float(__ldg(&combined_g_base[t_offset]));
        float proj_val = to_float(__ldg(&combined_p_base[t_offset]));
        float x_val = to_float(__ldg(&input[input_idx]));
        float grad_out_val = to_float(__ldg(&grad_out[input_idx]));

        float z = sigmoid(gate_val);
        float h_tilde = (hidden_val >= 0.0f) ? hidden_val + 0.5f : sigmoid(hidden_val);
        float h_t = lerp(h_prev, h_tilde, z);
        float proj_sigmoid = sigmoid(proj_val);

        // highway: out = s*h + (1-s)*x
        float dh_from_out = grad_out_val * proj_sigmoid;
        float grad_proj = grad_out_val * (h_t - x_val) * proj_sigmoid * (1.0f - proj_sigmoid);
        grad_input[input_idx] = from_float(grad_out_val * (1.0f - proj_sigmoid));
        grad_combined_p_base[t_offset] = from_float(grad_proj);

        float dh_total = dh + dh_from_out;

        // h = (1-z)*h_prev + z*h_tilde
        float d_h_prev = dh_total * (1.0f - z);
        float d_h_tilde = dh_total * z;
        float d_z = dh_total * (h_tilde - h_prev);

        // z = sigmoid(gate)
        float d_gate = d_z * z * (1.0f - z);
        // h_tilde = hidden+0.5  or  sigmoid(hidden)
        float d_hidden = (hidden_val >= 0.0f)
            ? d_h_tilde
            : d_h_tilde * h_tilde * (1.0f - h_tilde);

        grad_combined_h_base[t_offset] = from_float(d_hidden);
        grad_combined_g_base[t_offset] = from_float(d_gate);

        // terminal before this step zeroed h_prev contribution into this step;
        // after bwd through the step, cut gradient flow into pre-reset state.
        dh = d_h_prev;
        if (terminals != nullptr &&
                to_float(terminals[b * T_seq + t0]) != 0.0f) {
            dh = 0.0f;
        }
    }

    grad_state[state_idx] = from_float(dh);
}

__global__ void sum_rows_to_precision_kernel(precision_t* __restrict__ dst,
        const float* __restrict__ src, int R, int C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= C) {
        return;
    }
    float sum = 0.0f;
    for (int r = 0; r < R; r++) {
        sum += src[r * C + col];
    }
    dst[col] = from_float(sum);
}

__global__ void assemble_decoder_grad(
        precision_t* __restrict__ dst, const float* __restrict__ grad_logits,
        const float* __restrict__ grad_value, int B_TT, int od, int od_plus_1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B_TT * od_plus_1) {
        return;
    }
    int row = idx / od_plus_1, col = idx % od_plus_1;
    dst[idx] = from_float((col < od) ? grad_logits[row * od + col] : grad_value[row]);
}

PrecisionTensor encoder_forward(void* w, void* activations, PrecisionTensor input, cudaStream_t stream) {
    EncoderWeights* ew = (EncoderWeights*)w;
    EncoderActivations* a = (EncoderActivations*)activations;
    // Train acts register saved_input; rollout acts leave it null.
    if (a->saved_input.data) {
        puf_copy(&a->saved_input, &input, stream);
    }
    puf_mm(&input, &ew->weight, &a->out, stream);
    return a->out;
}

void encoder_backward(void* w, void* activations, PrecisionTensor grad, cudaStream_t stream) {
    EncoderActivations* a = (EncoderActivations*)activations;
    // Encoder has no dX GEMM; still async so it can overlap with later host work / muon prep.
    puf_mm_tn_async_after(&grad, &a->saved_input, &a->wgrad_scratch, stream);
}

void encoder_init_weights(void* w, ulong* seed, cudaStream_t stream) {
    EncoderWeights* ew = (EncoderWeights*)w;
    PrecisionTensor wt = {
        .data = ew->weight.data,
        .shape = {ew->out_dim, ew->in_dim},
    };
    puf_kaiming_init(&wt, sqrtf(2.0f), (*seed)++, stream);
}

void encoder_reg_params(void* w, Allocator* alloc) {
    EncoderWeights* ew = (EncoderWeights*)w;
    ew->weight = {.shape = {ew->out_dim, ew->in_dim}};
    alloc_register(alloc,&ew->weight);
}

void encoder_reg_train(void* w, void* activations, Allocator* acts, Allocator* grads, int B_TT) {
    EncoderWeights* ew = (EncoderWeights*)w;
    EncoderActivations* a = (EncoderActivations*)activations;
    *a = (EncoderActivations){
        .out =              {.shape = {B_TT, ew->out_dim}},
        .saved_input =      {.shape = {B_TT, ew->in_dim}},
        .wgrad_scratch =    {.shape = {ew->out_dim, ew->in_dim}},
    };
    alloc_register(acts,&a->out);
    alloc_register(acts,&a->saved_input);
    alloc_register(grads,&a->wgrad_scratch);
}

void encoder_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    EncoderWeights* ew = (EncoderWeights*)w;
    EncoderActivations* a = (EncoderActivations*)activations;
    a->out = {.shape = {B, ew->out_dim}};
    alloc_register(alloc,&a->out);
}

void* encoder_create_weights(void* self) {
    Encoder* e = (Encoder*)self;
    EncoderWeights* ew = (EncoderWeights*)calloc(1, sizeof(EncoderWeights));
    ew->in_dim = e->in_dim; ew->out_dim = e->out_dim;
    return ew;
}

struct DecoderWeights {
    PrecisionTensor weight, logstd;
    int hidden_dim, output_dim;
    bool continuous;
};

struct DecoderActivations {
    PrecisionTensor out, grad_out, saved_input, grad_input, wgrad_scratch, logstd_scratch;
};

PrecisionTensor decoder_forward(void* w, void* activations, PrecisionTensor input, cudaStream_t stream) {
    DecoderWeights* dw = (DecoderWeights*)w;
    DecoderActivations* a = (DecoderActivations*)activations;
    if (a->saved_input.data) {
        puf_copy(&a->saved_input, &input, stream);
    }
    puf_mm(&input, &dw->weight, &a->out, stream);
    return a->out;
}

void decoder_init_weights(void* w, ulong* seed, cudaStream_t stream) {
    DecoderWeights* dw = (DecoderWeights*)w;
    PrecisionTensor wt = {
        .data = dw->weight.data,
        .shape = {dw->output_dim + 1, dw->hidden_dim},
    };
    puf_kaiming_init(&wt, 1.0f, (*seed)++, stream);
}

void decoder_reg_params(void* w, Allocator* alloc) {
    DecoderWeights* dw = (DecoderWeights*)w;
    dw->weight = {.shape = {dw->output_dim + 1, dw->hidden_dim}};
    alloc_register(alloc,&dw->weight);
    if (dw->continuous) {
        dw->logstd = {.shape = {1, dw->output_dim}};
        alloc_register(alloc,&dw->logstd);
    }
}

void decoder_reg_train(void* w, void* activations, Allocator* acts, Allocator* grads, int B_TT) {
    DecoderWeights* dw = (DecoderWeights*)w;
    DecoderActivations* a = (DecoderActivations*)activations;
    int od1 = dw->output_dim + 1;
    *a = (DecoderActivations){
        .out =              {.shape = {B_TT, od1}},
        .grad_out =         {.shape = {B_TT, od1}},
        .saved_input =      {.shape = {B_TT, dw->hidden_dim}},
        .grad_input =       {.shape = {B_TT, dw->hidden_dim}},
        .wgrad_scratch =    {.shape = {od1, dw->hidden_dim}},
        .logstd_scratch =   {.shape = {1, dw->output_dim}},
    };
    alloc_register(acts,&a->out);
    alloc_register(acts,&a->saved_input);
    alloc_register(acts,&a->grad_out);
    alloc_register(acts,&a->grad_input);
    alloc_register(grads,&a->wgrad_scratch);
    if (dw->continuous) {
        alloc_register(grads, &a->logstd_scratch);
    }
}

void decoder_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    DecoderWeights* dw = (DecoderWeights*)w;
    DecoderActivations* a = (DecoderActivations*)activations;
    a->out = {.shape = {B, dw->output_dim + 1}};
    alloc_register(alloc,&a->out);
}

void* decoder_create_weights(void* self) {
    Decoder* d = (Decoder*)self;
    DecoderWeights* dw = (DecoderWeights*)calloc(1, sizeof(DecoderWeights));
    dw->hidden_dim = d->hidden_dim; dw->output_dim = d->output_dim; dw->continuous = d->continuous;
    return dw;
}

PrecisionTensor decoder_backward(void* w, void* activations,
    FloatTensor grad_logits, FloatTensor grad_logstd, FloatTensor grad_value, cudaStream_t stream) {
    DecoderWeights* dw = (DecoderWeights*)w;
    DecoderActivations* a = (DecoderActivations*)activations;
    int B_TT = a->saved_input.shape[0];
    int od = dw->output_dim, od1 = od + 1;
    assemble_decoder_grad<<<grid_size(B_TT * od1), BLOCK_SIZE, 0, stream>>>(
        a->grad_out.data, grad_logits.data, grad_value.data, B_TT, od, od1);
    // dW // dX: weight grad on side stream, dX on main (needed for residual chain).
    puf_mm_tn_async_after(&a->grad_out, &a->saved_input, &a->wgrad_scratch, stream);
    if (dw->continuous && grad_logstd.data != nullptr) {
        sum_rows_to_precision_kernel<<<grid_size(dw->output_dim), BLOCK_SIZE, 0, stream>>>(
            a->logstd_scratch.data, grad_logstd.data, B_TT, dw->output_dim);
    }
    puf_mm_nn(&a->grad_out, &dw->weight, &a->grad_input, stream);
    return a->grad_input;
}

struct MinGRUActivations {
    // Rollout
    PrecisionTensor* combined;       // (B rollout, 3*T)[num_layers]
    PrecisionTensor out;             // (B rollout, T)
    PrecisionTensor next_state;      // (B rollout, T)
    // Training
    PrecisionTensor* saved_inputs;   // (B, TT, T)[num_layers]
    PrefixScan* scan_bufs;           // [num_layers]
    PrecisionTensor* combined_bufs;  // (B*TT, 3*T)[num_layers]
    PrecisionTensor* wgrad_scratch;  // (3*T, T)[num_layers]
    PrecisionTensor grad_input_buf;  // (B*TT, T)
    PrecisionTensor grad_next_state; // (B, 1, T)
};

struct MinGRUWeights {
    int hidden, num_layers, horizon;
    PrecisionTensor* weights;  // [num_layers]
};

PrecisionTensor mingru_state_layer(MinGRUWeights* m, PrecisionTensor& state, int i) {
    long B = state.shape[1], H = state.shape[2];
    return {.data = state.data + i * B * H, .shape = {B, H}};
}

void mingru_init_weights(void* w, ulong* seed, cudaStream_t stream) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    for (int i = 0; i < m->num_layers; i++) {
        PrecisionTensor w2d = {
            .data = m->weights[i].data,
            .shape = {3 * m->hidden, m->hidden},
        };
        puf_kaiming_init(&w2d, 1.0f, (*seed)++, stream);
    }
}

void mingru_reg_params(void* w, Allocator* alloc) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    for (int i = 0; i < m->num_layers; i++) {
        m->weights[i] = {.shape = {3 * m->hidden, m->hidden}};
        alloc_register(alloc,&m->weights[i]);
    }
}

void mingru_reg_train(void* w, void* activations, Allocator* acts, Allocator* grads, int B_TT) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    int H = m->hidden, TT = m->horizon, B = B_TT / TT;
    a->saved_inputs = (PrecisionTensor*)calloc(m->num_layers, sizeof(PrecisionTensor));
    a->scan_bufs = (PrefixScan*)calloc(m->num_layers, sizeof(PrefixScan));
    a->combined_bufs = (PrecisionTensor*)calloc(m->num_layers, sizeof(PrecisionTensor));
    a->wgrad_scratch = (PrecisionTensor*)calloc(m->num_layers, sizeof(PrecisionTensor));
    a->grad_input_buf = {.shape = {B_TT, H}};
    a->grad_next_state = {.shape = {B, 1, H}};
    alloc_register(acts,&a->grad_input_buf);
    alloc_register(acts,&a->grad_next_state);
    for (int i = 0; i < m->num_layers; i++) {
        a->scan_bufs[i] = {
            .B = B, .T = TT, .H = H,
            .scan_h =           {.shape = {B, TT, H}},
            .out =              {.shape = {B, TT, H}},
            .next_state =       {.shape = {B, 1, H}},
            .grad_combined =    {.shape = {B, TT, 3 * H}},
            .grad_state =       {.shape = {B, 1, H}},
            .grad_input =       {.shape = {B, TT, H}},
        };
        a->saved_inputs[i]  = {.shape = {B, TT, H}};
        a->combined_bufs[i] = {.shape = {B_TT, 3 * H}};
        a->wgrad_scratch[i] = {.shape = {3 * H, H}};
        alloc_register(acts,&a->saved_inputs[i]);
        alloc_register(acts,&a->combined_bufs[i]);
        alloc_register(acts,&a->scan_bufs[i].out);
        alloc_register(acts,&a->scan_bufs[i].next_state);
        alloc_register(acts,&a->scan_bufs[i].scan_h);
        alloc_register(acts,&a->scan_bufs[i].grad_combined);
        alloc_register(acts,&a->scan_bufs[i].grad_state);
        alloc_register(acts,&a->scan_bufs[i].grad_input);
        alloc_register(grads,&a->wgrad_scratch[i]);
    }
}

void mingru_reg_rollout(void* weights, void* activations, Allocator* alloc, int B_inf) {
    MinGRUWeights* w = (MinGRUWeights*)weights;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    int H = w->hidden;
    a->combined = (PrecisionTensor*)calloc(w->num_layers, sizeof(PrecisionTensor));
    for (int i = 0; i < w->num_layers; i++) {
        a->combined[i] = {.shape = {B_inf, 3 * H}};
        alloc_register(alloc,&a->combined[i]);
    }
    a->out = {.shape = {B_inf, H}};
    a->next_state = {.shape = {B_inf, H}};
    alloc_register(alloc,&a->out);
    alloc_register(alloc,&a->next_state);
}

void* mingru_create_weights(void* self) {
    Network* n = (Network*)self;
    MinGRUWeights* mw = (MinGRUWeights*)calloc(1, sizeof(MinGRUWeights));
    mw->hidden = n->hidden; mw->num_layers = n->num_layers; mw->horizon = n->horizon;
    mw->weights = (PrecisionTensor*)calloc(n->num_layers, sizeof(PrecisionTensor));
    return mw;
}

PrecisionTensor mingru_forward(void* w, PrecisionTensor x, PrecisionTensor state,
        void* activations, cudaStream_t stream) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    int B = state.shape[1];
    int H = state.shape[2];
    for (int i = 0; i < m->num_layers; i++) {
        PrecisionTensor state_i = mingru_state_layer(m, state, i);
        puf_mm(&x, &m->weights[i], &a->combined[i], stream);
        mingru_gate<<<grid_size(B*H), BLOCK_SIZE, 0, stream>>>(
            a->out.data, a->next_state.data,
            a->combined[i].data, state_i.data, x.data, H, B);
        puf_copy(&state_i, &a->next_state, stream);
        x = a->out;
    }
    return x;
}

PrecisionTensor mingru_forward_train(void* w, PrecisionTensor x, PrecisionTensor state,
        PrecisionTensor terminals, void* activations, cudaStream_t stream) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    for (int i = 0; i < m->num_layers; i++) {
        puf_copy(&a->saved_inputs[i], &x, stream);
        PrecisionTensor state_i = mingru_state_layer(m, state, i);
        puf_mm(&x, &m->weights[i], &a->combined_bufs[i], stream);
        PrefixScan& scan = a->scan_bufs[i];
        scan.combined_ptr = a->combined_bufs[i].data;
        scan.state_ptr = state_i.data;
        scan.input_ptr = a->saved_inputs[i].data;
        scan.terminals_ptr = terminals.data;
        mingru_scan_forward_seq<<<grid_size(scan.B * scan.H), BLOCK_SIZE, 0, stream>>>(scan);
        x = scan.out;
    }
    return x;
}

__global__ void add_kernel(float* __restrict__ dst,
        const precision_t* __restrict__ src, int n) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
            idx += blockDim.x * gridDim.x) {
        dst[idx] += to_float(src[idx]);
    }
}

#ifndef PRECISION_FLOAT
__global__ void add_kernel(precision_t* __restrict__ dst,
        const precision_t* __restrict__ src, int n) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
            idx += blockDim.x * gridDim.x) {
        dst[idx] = from_float(to_float(dst[idx]) + to_float(src[idx]));
    }
}
#endif

PrecisionTensor mingru_backward(void* w, PrecisionTensor grad, void* activations, cudaStream_t stream) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    for (int i = m->num_layers - 1; i >= 0; i--) {
        PrefixScan& scan = a->scan_bufs[i];
        mingru_scan_backward<<<grid_size(scan.B*scan.H), BLOCK_SIZE, 0, stream>>>(
            scan, grad.data, a->grad_next_state.data);
        // dW on side stream (per-layer scratch); dX on main continues the bwd chain.
        puf_mm_tn_async_after(&scan.grad_combined, &a->saved_inputs[i],
            &a->wgrad_scratch[i], stream);
        puf_mm_nn(&scan.grad_combined, &m->weights[i], &a->grad_input_buf, stream);
        int n = numel(scan.grad_input.shape);
        add_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
            a->grad_input_buf.data, scan.grad_input.data, n);
        grad = a->grad_input_buf;
    }
    return grad;
}

struct Policy {
    Encoder encoder;
    Decoder decoder;
    Network network;
};

struct PolicyActivations {
    void* encoder;
    void* decoder;
    void* network;
};

struct PolicyWeights {
    void* encoder;
    void* decoder;
    void* network;
};

PrecisionTensor policy_forward(Policy* p, PolicyWeights& w, PolicyActivations& activations,
        PrecisionTensor obs, PrecisionTensor state, cudaStream_t stream) {
    PrecisionTensor enc_out = p->encoder.forward(w.encoder, activations.encoder, obs, stream);
    PrecisionTensor h = p->network.forward(w.network, enc_out, state, activations.network, stream);
    return p->decoder.forward(w.decoder, activations.decoder, h, stream);
}

PrecisionTensor policy_forward_train(Policy* p, PolicyWeights& w, PolicyActivations& activations,
        PrecisionTensor x, PrecisionTensor state, PrecisionTensor terminals, cudaStream_t stream) {
    int B = x.shape[0], TT = x.shape[1];
    PrecisionTensor h = p->encoder.forward(w.encoder, activations.encoder, *puf_squeeze(&x, 0), stream);
    h = p->network.forward_train(w.network, *puf_unsqueeze(&h, 0, B, TT), state, terminals, activations.network, stream);
    PrecisionTensor dec_out = p->decoder.forward(w.decoder, activations.decoder, *puf_squeeze(&h, 0), stream);
    return *puf_unsqueeze(&dec_out, 0, B, TT);
}

void policy_backward(Policy* p, PolicyWeights& w, PolicyActivations& activations,
        FloatTensor grad_logits, FloatTensor grad_logstd, FloatTensor grad_value, cudaStream_t stream) {
    int B = grad_logits.shape[0], TT = grad_logits.shape[1];
    PrecisionTensor grad_h = p->decoder.backward(w.decoder, activations.decoder,
        *puf_squeeze(&grad_logits, 0), grad_logstd, *puf_squeeze(&grad_value, 0), stream);
    grad_h = p->network.backward(w.network, *puf_unsqueeze(&grad_h, 0, B, TT), activations.network, stream);
    p->encoder.backward(w.encoder, activations.encoder, grad_h, stream);
    // All async dW GEMMs must complete before muon reads weight grads.
    puf_dw_join(stream);
}

PolicyActivations policy_reg_train(Policy* p, PolicyWeights& w,
        Allocator* acts, Allocator* grads, int B_TT) {
    PolicyActivations a;
    a.encoder = calloc(1, p->encoder.activation_size);
    a.decoder = calloc(1, p->decoder.activation_size);
    a.network = calloc(1, sizeof(MinGRUActivations));
    p->encoder.reg_train(w.encoder, a.encoder, acts, grads, B_TT);
    p->decoder.reg_train(w.decoder, a.decoder, acts, grads, B_TT);
    p->network.reg_train(w.network, a.network, acts, grads, B_TT);
    return a;
}

PolicyActivations policy_reg_rollout(Policy* p, PolicyWeights& w, Allocator* acts, int B_inf) {
    PolicyActivations a;
    a.encoder = calloc(1, p->encoder.activation_size);
    a.decoder = calloc(1, p->decoder.activation_size);
    a.network = calloc(1, sizeof(MinGRUActivations));
    p->encoder.reg_rollout(w.encoder, a.encoder, acts, B_inf);
    p->decoder.reg_rollout(w.decoder, a.decoder, acts, B_inf);
    p->network.reg_rollout(w.network, a.network, acts, B_inf);
    return a;
}

void policy_init_weights(Policy* p, PolicyWeights& w, uint64_t* seed, cudaStream_t stream) {
    p->encoder.init_weights(w.encoder, seed, stream);
    p->decoder.init_weights(w.decoder, seed, stream);
    p->network.init_weights(w.network, seed, stream);
}

PolicyWeights policy_weights_create(Policy* p, Allocator* params) {
    PolicyWeights w;
    w.encoder = p->encoder.create_weights(&p->encoder);
    w.decoder = p->decoder.create_weights(&p->decoder);
    w.network = p->network.create_weights(&p->network);
    p->encoder.reg_params(w.encoder, params);
    p->decoder.reg_params(w.decoder, params);
    p->network.reg_params(w.network, params);
    return w;
}

// Custom architectures for specific envs. Not yet
// happy with the API, but we can't narrow it yet
// because fast, general encoder arch is an
// unsolved research problem.
#include "ocean.cu"

// Build a Policy value for a given env + arch. Encoder/decoder algorithms are
// fixed by the env; hidden_size/num_layers/horizon parameterize shape. Policy
// has no heap state so this returns by value; callers store it wherever.
Policy build_policy(const char* env_name, int input_size, int hidden_size,
        int num_layers, int decoder_output_size, bool is_continuous,
        int horizon) {
    Encoder encoder = {
        .forward = encoder_forward,
        .backward = encoder_backward,
        .init_weights = encoder_init_weights,
        .reg_params = encoder_reg_params,
        .reg_train = encoder_reg_train,
        .reg_rollout = encoder_reg_rollout,
        .create_weights = encoder_create_weights,
        .in_dim = input_size, .out_dim = hidden_size,
        .activation_size = sizeof(EncoderActivations),
    };
    create_custom_encoder(env_name, &encoder);
    Decoder decoder = {
        .forward = decoder_forward,
        .backward = decoder_backward,
        .init_weights = decoder_init_weights,
        .reg_params = decoder_reg_params,
        .reg_train = decoder_reg_train,
        .reg_rollout = decoder_reg_rollout,
        .create_weights = decoder_create_weights,
        .hidden_dim = hidden_size, .output_dim = decoder_output_size, .continuous = is_continuous,
        .activation_size = sizeof(DecoderActivations),
    };
    create_custom_decoder(env_name, &decoder);
    Network network = {
        .forward = mingru_forward,
        .forward_train = mingru_forward_train,
        .backward = mingru_backward,
        .init_weights = mingru_init_weights,
        .reg_params = mingru_reg_params,
        .reg_train = mingru_reg_train,
        .reg_rollout = mingru_reg_rollout,
        .create_weights = mingru_create_weights,
        .hidden = hidden_size, .num_layers = num_layers, .horizon = horizon,
    };
    return Policy{
        .encoder = encoder, .decoder = decoder, .network = network,
    };
}

// Muon optimizer. Our benchmarks show this is a major
// upgrade over Adam (weight decay not needed in RL).
__global__ void muon_norm_reduce(float* __restrict__ out, const float* __restrict__ partials, int num_blocks) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    sdata[tid] = (tid < num_blocks) ? partials[tid] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        *out = sdata[0];
    }
}

__global__ void muon_norm_partials(float* __restrict__ partials, const precision_t* __restrict__ src, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        float v = to_float(src[i]);
        sum += v * v;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partials[blockIdx.x] = sdata[0];
    }
}

__global__ void muon_norm_apply(precision_t* __restrict__ dst, const float* __restrict__ norm_ptr, float eps, int n) {
    float inv_norm = 1.0f / fmaxf(sqrtf(*norm_ptr), eps);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float(to_float(dst[idx]) * inv_norm);
    }
}

// Nesterov with f32 momentum accumulator and precision_t gradients
__global__ void muon_nesterov(float* __restrict__ mb, precision_t* __restrict__ gc, float mu, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float m = mu * mb[idx] + to_float(gc[idx]);
        mb[idx] = m;
        gc[idx] = from_float(to_float(gc[idx]) + mu * m);
    }
}

// Fused weight update: wb = wb * (1 - lr*wd) - lr * scale * update
__global__ void muon_weight_update(float* __restrict__ wb, const precision_t* __restrict__ update,
        const float* __restrict__ lr_ptr, float wd, float scale, int n) {
    float lr = *lr_ptr;
    float wd_scale = 1.0f - lr * wd;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        wb[idx] = wb[idx] * wd_scale - lr * scale * to_float(update[idx]);
    }
}

__global__ void muon_clip_norm(precision_t* __restrict__ dst,
        const float* __restrict__ sum_sq_ptr, float max_norm, float eps, int n) {
    float clip_coef = fminf(max_norm / (sqrtf(*sum_sq_ptr) + eps), 1.0f);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float(to_float(dst[idx]) * clip_coef);
    }
}

constexpr double ns_coeffs[5][3] = {
    {4.0848, -6.8946, 2.9270},
    {3.9505, -6.3029, 2.6377},
    {3.7418, -5.5913, 2.3037},
    {2.8769, -3.1427, 1.2046},
    {2.8366, -3.0525, 1.2012},
};

struct Muon {
    double momentum;
    FloatTensor lr_puf, ns_norm_puf, grad_norm_puf, mb_puf, norm_partials;
    PrecisionTensor gram, gram_buf, x_buf;
    Allocator* param_alloc;  // params allocator — shapes used by muon_step
};

void muon_init(Muon* m, Allocator* param_alloc, double momentum, Allocator* alloc) {
    m->momentum = momentum;
    m->param_alloc = param_alloc;
    m->lr_puf =         {.shape = {1}};
    m->mb_puf =         {.shape = {param_alloc->total_elems}};
    m->norm_partials =  {.shape = {256}};
    m->grad_norm_puf =  {.shape = {1}};
    alloc_register(alloc, &m->lr_puf);
    alloc_register(alloc, &m->mb_puf);
    alloc_register(alloc, &m->norm_partials);
    alloc_register(alloc, &m->grad_norm_puf);
    long max_M = 0, max_N = 0;
    for (int _i = 0; _i < param_alloc->num_regs; _i++) {
        AllocEntry& e = param_alloc->regs[_i];
        if (ndim(e.shape) >= 2) {
            long R = e.shape[0], C = numel(e.shape) / R;
            max_M = max(max_M, min(R, C));
            max_N = max(max_N, max(R, C));
        }
    }
    m->gram =        {.shape = {max_M, max_M}};
    m->gram_buf =    {.shape = {max_M, max_M}};
    m->x_buf =       {.shape = {max_M, max_N}};
    m->ns_norm_puf = {.shape = {1}};
    alloc_register(alloc, &m->gram);
    alloc_register(alloc, &m->gram_buf);
    alloc_register(alloc, &m->x_buf);
    alloc_register(alloc, &m->ns_norm_puf);
}

void muon_step(Muon* m, FloatTensor weights, PrecisionTensor grads, float max_grad_norm, cudaStream_t stream = 0) {
    // Clip gradients by norm
    int clip_blocks = min((int)grid_size(numel(grads.shape)), 256);
    muon_norm_partials<<<clip_blocks, 256, 0, stream>>>(
        m->norm_partials.data, grads.data, numel(grads.shape));
    muon_norm_reduce<<<1, 256, 0, stream>>>(m->grad_norm_puf.data, m->norm_partials.data, clip_blocks);
    muon_clip_norm<<<grid_size(numel(grads.shape)), BLOCK_SIZE, 0, stream>>>(
        grads.data, m->grad_norm_puf.data, max_grad_norm, 1e-6f, numel(grads.shape));

    // Nesterov momentum
    muon_nesterov<<<grid_size(numel(m->mb_puf.shape)), BLOCK_SIZE, 0, stream>>>(
        m->mb_puf.data, grads.data, (float)m->momentum, numel(m->mb_puf.shape));

    long offset = 0;
    for (int _i = 0; _i < m->param_alloc->num_regs; _i++) {
        AllocEntry& e = m->param_alloc->regs[_i];
        precision_t* gc_ptr = grads.data + offset;
        float* wb_ptr = weights.data + offset;
        long ne = numel(e.shape);
        const precision_t* update_ptr = gc_ptr;
        float scale = 1.0f;

        // Orthogonalize the update
        if (ndim(e.shape) >= 2) {
            long R = e.shape[0], C = ne / R;
            long M = min(R, C), N = max(R, C);
            bool tall = R > C;
            PrecisionTensor x = {.data = gc_ptr, .shape = {R, C}};
            PrecisionTensor x_buf = {.data = m->x_buf.data, .shape = {R, C}};
            PrecisionTensor gram = {.data = m->gram.data, .shape = {M, M}};
            PrecisionTensor gram_buf = {.data = m->gram_buf.data, .shape = {M, M}};

            int nblk = min((int)grid_size(numel(x.shape)), 256);
            muon_norm_partials<<<nblk, 256, 0, stream>>>(
                m->norm_partials.data, x.data, numel(x.shape));
            muon_norm_reduce<<<1, 256, 0, stream>>>(m->ns_norm_puf.data, m->norm_partials.data, nblk);
            muon_norm_apply<<<grid_size(numel(x.shape)), BLOCK_SIZE, 0, stream>>>(
                x.data, m->ns_norm_puf.data, 1e-7f, numel(x.shape));

            cublasOperation_t gram_op_a = tall ? CUBLAS_OP_T : CUBLAS_OP_N;
            cublasOperation_t gram_op_b = tall ? CUBLAS_OP_N : CUBLAS_OP_T;
            for (int i = 0; i < 5; ++i) {
                PrecisionTensor& src = (i % 2 == 0) ? x : x_buf;
                PrecisionTensor& dst = (i % 2 == 0) ? x_buf : x;
                cublasGemmExDense(gram_op_a, gram_op_b, (int)M, (int)M, (int)N,
                    src.data, src.data, gram.data, stream);
                puf_copy(&gram_buf, &gram, stream);
                cublasGemmExDense(CUBLAS_OP_N, CUBLAS_OP_N, (int)M, (int)M, (int)M,
                    gram.data, gram.data, gram_buf.data, stream,
                    ns_coeffs[i][2], ns_coeffs[i][1]);
                puf_copy(&dst, &src, stream);
                cublasGemmExDense(CUBLAS_OP_N, CUBLAS_OP_N, (int)R, (int)C, (int)M,
                    tall ? src.data : gram_buf.data, tall ? gram_buf.data : src.data, dst.data,
                    stream, 1.0f, ns_coeffs[i][0]);
            }

            update_ptr = x_buf.data;
            scale = sqrtf(fmaxf(1.0f, (float)R / (float)C));
        }

        muon_weight_update<<<grid_size(ne), BLOCK_SIZE, 0, stream>>>(
            wb_ptr, update_ptr, m->lr_puf.data, 0.0f, scale, (int)ne);
        offset += ne;
    }
}

// Train data layout is transposed to (B, T) from rollouts layout (T, B)
// This allows env workers to collect data with contiguous writes and
// training to perform several (though not all) ops in contiguous memory
struct TrainGraph {
    PrecisionTensor mb_state;       // (layers, B, hidden)
    PrecisionTensor mb_obs;         // (B, T, input_size)
    PrecisionTensor mb_actions;     // (B, T, num_atns)
    PrecisionTensor mb_logprobs;    // (B, T)
    PrecisionTensor mb_terminals;   // (B, T), resets recurrent state before timestep t
    PrecisionTensor mb_advantages;  // ...
    PrecisionTensor mb_values;
    PrecisionTensor mb_returns;
    PrecisionTensor mb_ratio;
    PrecisionTensor mb_newvalue;
    PrecisionTensor mb_prio;        // (B,)
    PrecisionTensor mb_action_mask; // (B, T, mask_size); always allocated
};

void register_train_buffers(TrainGraph& bufs, Allocator* alloc, int B, int T, int input_size,
        int hidden_size, int num_atns, int num_layers, int mask_size) {
    bufs = (TrainGraph){
        .mb_state =         {.shape = {num_layers, B, hidden_size}},
        .mb_obs =           {.shape = {B, T, input_size}},
        .mb_actions =       {.shape = {B, T, num_atns}},
        .mb_logprobs =      {.shape = {B, T}},
        .mb_terminals =     {.shape = {B, T}},
        .mb_advantages =    {.shape = {B, T}},
        .mb_values =        {.shape = {B, T}},
        .mb_returns =       {.shape = {B, T}},
        .mb_ratio =         {.shape = {B, T}},
        .mb_newvalue =      {.shape = {B, T}},
        .mb_prio =          {.shape = {B}},
        .mb_action_mask =   {.shape = {B, T, mask_size}},
    };
    alloc_register(alloc, &bufs.mb_obs);
    alloc_register(alloc, &bufs.mb_state);
    alloc_register(alloc, &bufs.mb_actions);
    alloc_register(alloc, &bufs.mb_logprobs);
    alloc_register(alloc, &bufs.mb_terminals);
    alloc_register(alloc, &bufs.mb_advantages);
    alloc_register(alloc, &bufs.mb_prio);
    alloc_register(alloc, &bufs.mb_values);
    alloc_register(alloc, &bufs.mb_returns);
    alloc_register(alloc, &bufs.mb_ratio);
    alloc_register(alloc, &bufs.mb_newvalue);
    alloc_register(alloc, &bufs.mb_action_mask);
}

// Prioritized replay over single-epoch data. These kernels are
// the least cleaned because we will likely have a better method in 5.0.
struct PrioBuffers {
    FloatTensor prio_probs, cdf, mb_prio;
    IntTensor idx;
};

void register_prio_buffers(PrioBuffers& bufs, Allocator* alloc, int B, int minibatch_segments) {
    bufs = (PrioBuffers){
        .prio_probs = {.shape = {B}},
        .cdf = {.shape = {B}},
        .mb_prio = {.shape = {minibatch_segments}},
        .idx = {.shape = {minibatch_segments}},
    };
    alloc_register(alloc, &bufs.prio_probs);
    alloc_register(alloc, &bufs.cdf);
    alloc_register(alloc, &bufs.idx);
    alloc_register(alloc, &bufs.mb_prio);
}

#define PRIO_WARP_SIZE 32
#define PRIO_FULL_MASK 0xffffffff
#define PRIO_BLOCK_SIZE 256
#define PRIO_NUM_WARPS (PRIO_BLOCK_SIZE / PRIO_WARP_SIZE)

__global__ void compute_prio_adv_reduction(
        const precision_t* __restrict__ advantages,
        float* prio_weights, float prio_alpha, int stride) {
    int row = blockIdx.x;
    int tx = threadIdx.x;
    int offset = row * stride;

    float local_sum = 0.0f;
    for (int t = tx; t < stride; t += blockDim.x) {
        local_sum += fabsf(to_float(advantages[offset + t]));
    }

    for (int s = PRIO_WARP_SIZE / 2; s >= 1; s /= 2) {
        local_sum += __shfl_down_sync(PRIO_FULL_MASK, local_sum, s);
    }
    if (tx == 0) {
        float pw = __powf(local_sum, prio_alpha);
        if (isnan(pw) || isinf(pw)) {
            pw = 0.0f;
        }
        prio_weights[row] = pw;
    }
}

__global__ void compute_prio_normalize(float* prio_weights, int length) {
    __shared__ float shmem[PRIO_NUM_WARPS];
    __shared__ float block_sum;

    int tx = threadIdx.x;
    int lane = tx % PRIO_WARP_SIZE;
    int warp_id = tx / PRIO_WARP_SIZE;
    const float eps = 1e-6f;

    float local_sum = 0.0f;
    for (int t = tx; t < length; t += blockDim.x) {
        local_sum += prio_weights[t];
    }
    for (int s = PRIO_WARP_SIZE / 2; s >= 1; s /= 2) {
        local_sum += __shfl_down_sync(PRIO_FULL_MASK, local_sum, s);
    }
    if (lane == 0) {
        shmem[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < PRIO_NUM_WARPS) ? shmem[lane] : 0.0f;
        for (int s = PRIO_NUM_WARPS / 2; s >= 1; s /= 2) {
            val += __shfl_down_sync(PRIO_FULL_MASK, val, s);
        }
        if (tx == 0) {
            block_sum = val + eps;
        }
    }
    __syncthreads();

    for (int t = tx; t < length; t += blockDim.x) {
        prio_weights[t] = (prio_weights[t] + eps) / block_sum;
    }
}

// mb_prio[i] = pow(total_agents * prio_probs[idx[i]], -anneal_beta)
__global__ void compute_prio_imp_weights(
        const int* __restrict__ indices,
        const float* __restrict__ prio_probs,
        float* mb_prio, int total_agents,
        float anneal_beta, int minibatch_segments) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tx < minibatch_segments) {
        float value = prio_probs[indices[tx]] * (float)total_agents;
        mb_prio[tx] = __powf(value, -anneal_beta);
    }
}

// Inclusive prefix sum of probs → cdf. Was <<<1,1>>> serial (B=8192 ≈ 100µs,
// ~8% of train GPU). Single-block chunked scan is ~6µs on the same size.
__global__ void build_cdf(
        float* __restrict__ cdf, const float* __restrict__ probs, int B) {
    typedef cub::BlockScan<float, PRIO_BLOCK_SIZE> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    const int tid = threadIdx.x;
    // Contiguous chunk per thread so the second pass is sequential + coalesced.
    const int chunk = (B + PRIO_BLOCK_SIZE - 1) / PRIO_BLOCK_SIZE;
    const int start = tid * chunk;
    const int end = min(start + chunk, B);

    float run = 0.0f;
    for (int i = start; i < end; i++) {
        run += probs[i];
        cdf[i] = run;
    }
    float my_total = (start < B) ? run : 0.0f;

    float exclusive = 0.0f;
    BlockScan(temp_storage).ExclusiveSum(my_total, exclusive);

    if (exclusive != 0.0f) {
        for (int i = start; i < end; i++) {
            cdf[i] += exclusive;
        }
    }
}

__global__ void advance_rng_offset(int64_t* __restrict__ offset_ptr, int64_t delta) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *offset_ptr += delta;
    }
}

// Multinomial with replacement (uses cuRAND)
__global__ void multinomial_sample(int* __restrict__ out_idx, const float* __restrict__ cdf,
        int B, int num_samples, uint64_t seed, const int64_t* __restrict__ offset_ptr) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_samples) {
        return;
    }

    uint64_t base_off = (uint64_t)(*offset_ptr);
    curandStatePhilox4_32_10_t rng_state;
    curand_init(seed, base_off + tid, 0, &rng_state);
    float u = curand_uniform(&rng_state);

    int lo = 0;
    int hi = B - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cdf[mid] < u) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    out_idx[tid] = lo;
}

// Build per-trajectory prio probs + CDF from advantages. Independent of the
// minibatch sample — call once per train_impl, not once per minibatch.
void prio_build_cdf_cuda(PrecisionTensor& advantages, float prio_alpha,
        PrioBuffers& bufs, cudaStream_t stream) {
    int B = advantages.shape[0];
    int T = advantages.shape[1];
    compute_prio_adv_reduction<<<B, PRIO_WARP_SIZE, 0, stream>>>(
        advantages.data, bufs.prio_probs.data, prio_alpha, T);
    compute_prio_normalize<<<1, PRIO_BLOCK_SIZE, 0, stream>>>(
        bufs.prio_probs.data, B);
    build_cdf<<<1, PRIO_BLOCK_SIZE, 0, stream>>>(bufs.cdf.data, bufs.prio_probs.data, B);
}

// Draw a minibatch index set from a prebuilt CDF and importance weights.
void prio_sample_cuda(float anneal_beta, PrioBuffers& bufs, ulong seed,
        long* offset_ptr, cudaStream_t stream) {
    int B = (int)bufs.cdf.shape[0];
    int N = (int)bufs.idx.shape[0];
    int blocks = (N + PRIO_BLOCK_SIZE - 1) / PRIO_BLOCK_SIZE;
    multinomial_sample<<<blocks, PRIO_BLOCK_SIZE, 0, stream>>>(
        bufs.idx.data, bufs.cdf.data, B, N, seed, offset_ptr);
    advance_rng_offset<<<1, 1, 0, stream>>>(offset_ptr, N);
    compute_prio_imp_weights<<<blocks, PRIO_BLOCK_SIZE, 0, stream>>>(
        bufs.idx.data, bufs.prio_probs.data,
        bufs.mb_prio.data, B, anneal_beta, N);
}

// TODO: test whether these finite/clamp guards improve continuous-control stability
// or just hide bad logits/actions.
__device__ __forceinline__ float finite_or_clamp(float x, float lo, float hi) {
    if (isnan(x)) {
        return 0.0f;
    }
    if (isinf(x)) {
        return x > 0.0f ? hi : lo;
    }
    return fminf(hi, fmaxf(lo, x));
}

__device__ __forceinline__ float safe_continuous_mean(const precision_t* logits, int idx) {
    return finite_or_clamp(to_float(logits[idx]), -1.0e6f, 1.0e6f);
}

__device__ __forceinline__ float safe_continuous_logstd(const precision_t* logstd, int idx) {
    return finite_or_clamp(to_float(logstd[idx]), -20.0f, 2.0f);
}

// Fused loss function. PPO clipped loss + value + entropy
constexpr int PPO_THREADS = 256;
constexpr int MAX_ATN_HEADS = 16; // TODO: use env atn dim directly

enum LossIdx {
    LOSS_PG = 0, LOSS_VF = 1, LOSS_ENT = 2, LOSS_TOTAL = 3,
    LOSS_OLD_APPROX_KL = 4, LOSS_APPROX_KL = 5, LOSS_CLIPFRAC = 6,
    LOSS_N = 7, NUM_LOSSES = 8,
};

// PPO buffers + args are quite complex. We do the entire
// forward + backwards pass for the full loss function in one kernel
struct PPOGraphArgs {
    precision_t* out_ratio;
    precision_t* out_newvalue;
    const precision_t* actions;
    const precision_t* old_logprobs;
    const precision_t* advantages;
    const precision_t* prio;
    const precision_t* values;
    const precision_t* returns;
};

struct PPOKernelArgs {
    float* grad_logits;
    float* grad_logstd; // For continuous actions
    float* grad_values_pred;
    const precision_t* logits;
    const precision_t* logstd; // Continuous only
    const precision_t* values_pred;
    const float* adv_mean;
    const float* adv_var;
    const int* act_sizes;
    const precision_t* action_mask; // (N, T, A_total); always present
    int num_atns;
    float clip_coef, vf_clip_coef, vf_coef;
    const float* ent_coef;  // device ptr — host by-value bakes into CUDA graphs
    int T_seq, A_total, N;
    bool is_continuous;
};

struct PPOBuffersPuf {
    FloatTensor grad_logits, grad_values, grad_logstd, adv_scratch;
    FloatTensor ent_coef;
    FloatTensor ppo_partials;
};

void register_ppo_buffers(PPOBuffersPuf& bufs, Allocator* alloc, int N, int T, int A_total, bool is_continuous) {
    long total = (long)N * T;
    int ppo_grid = ((int)total + PPO_THREADS - 1) / PPO_THREADS;
    bufs = (PPOBuffersPuf){
        .grad_logits = {.shape = {N, T, A_total}},
        .grad_values = {.shape = {N, T, 1}},
        .grad_logstd = {.shape = {N, T, A_total}},
        // [0]=var, [1]=mean, [2..)= partials (sums then sumsq, up to 1024 each)
        .adv_scratch = {.shape = {2 + 2 * 1024}},
        .ent_coef = {.shape = {1}},
        .ppo_partials = {.shape = {ppo_grid * LOSS_N}},
    };
    alloc_register(alloc, &bufs.grad_logits);
    alloc_register(alloc, &bufs.grad_values);
    if (is_continuous) {
        alloc_register(alloc, &bufs.grad_logstd);
    }
    alloc_register(alloc, &bufs.adv_scratch);
    alloc_register(alloc, &bufs.ent_coef);
    alloc_register(alloc, &bufs.ppo_partials);
}

// Discrete only. mask is always present (env mask or synthetic all-ones).
__device__ __forceinline__ float load_logit_masked(
        const precision_t* __restrict__ logits, int logits_base,
        int logits_offset, int a,
        const precision_t* __restrict__ mask, int mask_base) {
    float l = to_float(logits[logits_base + logits_offset + a]);
    float m = to_float(mask[mask_base + logits_offset + a]);
    if (m == 0.0f) {
        l = -1e4f;
    }
    return l;
}

__device__ __forceinline__ void ppo_discrete_head(
        const precision_t* __restrict__ logits, int logits_base,
        int logits_offset, int A, int act,
        const precision_t* __restrict__ mask, int mask_base,
        float* out_logsumexp, float* out_entropy, float* out_logp) {
    float max_logit = -INFINITY;
    float sum = 0.0f;
    float act_logit = 0.0f;

    for (int a = 0; a < A; ++a) {
        float l = load_logit_masked(logits, logits_base, logits_offset, a, mask, mask_base);
        if (a == act) {
            act_logit = l;
        }
        if (l > max_logit) {
            sum *= __expf(max_logit - l);
            max_logit = l;
        }
        sum += __expf(l - max_logit);
    }
    float logsumexp = max_logit + __logf(sum);

    float ent = 0.0f;
    for (int a = 0; a < A; ++a) {
        float l = load_logit_masked(logits, logits_base, logits_offset, a, mask, mask_base);
        float logp = l - logsumexp;
        float p = __expf(logp);
        ent -= p * logp;
    }

    *out_logsumexp = logsumexp;
    *out_entropy = ent;
    *out_logp = act_logit - logsumexp;
}

__device__ __forceinline__ void ppo_continuous_head(
        float mean, float log_std, float action,
        float* out_logp, float* out_entropy) {
    constexpr float HALF_LOG_2PI = 0.9189385332046727f;
    constexpr float HALF_1_PLUS_LOG_2PI = 1.4189385332046727f;
    float std = __expf(log_std);
    float normalized = (action - mean) / std;
    *out_logp = -0.5f * normalized * normalized - HALF_LOG_2PI - log_std;
    *out_entropy = HALF_1_PLUS_LOG_2PI + log_std;
}

__global__ void ppo_loss_compute(
        float* __restrict__ ppo_partials,
        PPOKernelArgs a, PPOGraphArgs g) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int total_elements = a.N * a.T_seq;
    float inv_NT = 1.0f / float(total_elements);

    __shared__ float block_losses[LOSS_N][PPO_THREADS];
    for (int c = 0; c < LOSS_N; c++) {
        block_losses[c][tid] = 0.0f;
    }

    if (idx >= total_elements) {
        goto reduce;
    }

    {
    int n = idx / a.T_seq;
    int t = idx % a.T_seq;
    int nt = n * a.T_seq + t;

    int logits_base = nt * (a.A_total + 1);
    int grad_logits_base = nt * a.A_total;

    // Shared computation (used by both forward and backward)

    float old_logp = to_float(g.old_logprobs[nt]);
    float adv = to_float(g.advantages[nt]);
    float w = to_float(g.prio[n]);
    float val = to_float(g.values[nt]);
    float ret = to_float(g.returns[nt]);
    float val_pred = to_float(a.values_pred[logits_base]);
    g.out_newvalue[nt] = from_float(val_pred);

    float adv_std = sqrtf(float(a.adv_var[0]));
    float adv_normalized = (adv - float(a.adv_mean[0])) / (adv_std + 1e-8f);

    float dL = inv_NT;
    float ent_coef = *a.ent_coef;
    float d_entropy_term = dL * (-ent_coef);

    // Value loss (forward) + value gradient (backward)

    float v_error = val_pred - val;
    float v_clipped = val + fmaxf(-a.vf_clip_coef, fminf(a.vf_clip_coef, v_error));
    float v_loss_unclipped = (val_pred - ret) * (val_pred - ret);
    float v_loss_clipped = (v_clipped - ret) * (v_clipped - ret);
    float v_loss = 0.5f * fmaxf(v_loss_unclipped, v_loss_clipped);

    // Value gradient
    bool use_clipped_vf = (v_loss_clipped > v_loss_unclipped);
    float d_val_pred = 0.0f;
    if (use_clipped_vf) {
        if (v_error >= -a.vf_clip_coef && v_error <= a.vf_clip_coef) {
            d_val_pred = v_clipped - ret;
        }
    } else {
        d_val_pred = val_pred - ret;
    }
    a.grad_values_pred[nt] = dL * a.vf_coef * d_val_pred;

    // Policy loss + gradients

    float pg_loss, total_entropy, logratio, ratio;
    float total_log_prob = 0.0f;
    total_entropy = 0.0f;

    // Discrete-only: per-head arrays needed across forward + backward
    float head_logsumexp[MAX_ATN_HEADS];
    float head_entropy[MAX_ATN_HEADS];
    int head_act[MAX_ATN_HEADS];

    // Discrete always registers mb_action_mask; continuous never uses this arm.
    int mask_base = nt * a.A_total;

    if (!a.is_continuous) {
        int logits_offset = 0;
        for (int h = 0; h < a.num_atns; ++h) {
            int A = a.act_sizes[h];
            int act = static_cast<int>(g.actions[nt * a.num_atns + h]);
            head_act[h] = act;
            float lse, ent, lp;
            ppo_discrete_head(a.logits, logits_base, logits_offset, A, act,
                              a.action_mask, mask_base, &lse, &ent, &lp);
            head_logsumexp[h] = lse;
            head_entropy[h] = ent;
            total_log_prob += lp;
            total_entropy += ent;
            logits_offset += A;
        }
    } else {
        for (int h = 0; h < a.num_atns; ++h) {
            float mean = safe_continuous_mean(a.logits, logits_base + h);
            float log_std = safe_continuous_logstd(a.logstd, h);
            float action = finite_or_clamp(float(g.actions[nt * a.num_atns + h]), -1.0e6f, 1.0e6f);
            float lp, ent;
            ppo_continuous_head(mean, log_std, action, &lp, &ent);
            total_log_prob += lp;
            total_entropy += ent;
        }
    }

    // Shared pg loss computation
    logratio = total_log_prob - old_logp;
    ratio = __expf(logratio);
    g.out_ratio[nt] = from_float(ratio);
    float ratio_clipped = fmaxf(1.0f - a.clip_coef, fminf(1.0f + a.clip_coef, ratio));
    float wa = -w * adv_normalized;
    float pg_loss1 = wa * ratio;
    float pg_loss2 = wa * ratio_clipped;
    pg_loss = fmaxf(pg_loss1, pg_loss2);

    float d_ratio = wa * dL;
    if (pg_loss2 > pg_loss1) {
        if (ratio <= (1.0f - a.clip_coef) || ratio >= (1.0f + a.clip_coef)) {
            d_ratio = 0.0f;
        }
    }
    float d_new_logp = d_ratio * ratio;

    if (!a.is_continuous) {
        int logits_offset = 0;
        for (int h = 0; h < a.num_atns; ++h) {
            int A = a.act_sizes[h];
            int act = head_act[h];
            float logsumexp = head_logsumexp[h];
            float ent = head_entropy[h];

            for (int j = 0; j < A; ++j) {
                float l = load_logit_masked(a.logits, logits_base,
                                            logits_offset, j, a.action_mask, mask_base);
                float logp = l - logsumexp;
                float p = __expf(logp);
                float d_logit = (j == act) ? d_new_logp : 0.0f;
                d_logit -= p * d_new_logp;
                d_logit += d_entropy_term * p * (-ent - logp);
                a.grad_logits[grad_logits_base + logits_offset + j] = d_logit;
            }
            logits_offset += A;
        }
    } else {
        for (int h = 0; h < a.num_atns; ++h) {
            float mean = safe_continuous_mean(a.logits, logits_base + h);
            float log_std = safe_continuous_logstd(a.logstd, h);
            float std = __expf(log_std);
            float var = std * std;
            float action = finite_or_clamp(float(g.actions[nt * a.num_atns + h]), -1.0e6f, 1.0e6f);
            float diff = action - mean;

            a.grad_logits[grad_logits_base + h] = d_new_logp * diff / var;
            a.grad_logstd[nt * a.num_atns + h] = d_new_logp * (diff * diff / var - 1.0f) + d_entropy_term;
        }
    }

    // Forward: loss partials
    float thread_loss = (pg_loss + a.vf_coef * v_loss - ent_coef * total_entropy) * inv_NT;
    block_losses[LOSS_PG][tid] = pg_loss * inv_NT;
    block_losses[LOSS_VF][tid] = v_loss * inv_NT;
    block_losses[LOSS_ENT][tid] = total_entropy * inv_NT;
    block_losses[LOSS_TOTAL][tid] = thread_loss;
    block_losses[LOSS_OLD_APPROX_KL][tid] = (-logratio) * inv_NT;
    block_losses[LOSS_APPROX_KL][tid] = ((ratio - 1.0f) - logratio) * inv_NT;
    block_losses[LOSS_CLIPFRAC][tid] = (fabsf(ratio - 1.0f) > a.clip_coef ? 1.0f : 0.0f) * inv_NT;
    } // end if (idx < total_elements)

// Deterministic aggregation
reduce:
    __syncthreads();

    for (int stride = PPO_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (int c = 0; c < LOSS_N; c++) {
                block_losses[c][tid] += block_losses[c][tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        int base = blockIdx.x * LOSS_N;
        for (int c = 0; c < LOSS_N; c++) {
            ppo_partials[base + c] = block_losses[c][0];
        }
    }
}

// Deterministic reduction of per-block PPO loss partials + count increment
__global__ void ppo_loss_reduce(
        float* __restrict__ losses_acc,
        const float* __restrict__ partials,
        int num_blocks) {
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int b = 0; b < num_blocks; b++) {
        sum += partials[b * LOSS_N + tid];
    }
    losses_acc[tid] += sum;
    if (tid == 0) {
        losses_acc[LOSS_N] += 1.0f;
    }
}

// Multi-block advantage mean/var (sample var). One pass sum/sumsq + finalize.
// Replaces <<<1,256>>> over full NT. Partials layout: [0,B)=sums, [B,2B)=sumsq.
constexpr int PPO_VM_THREADS = 256;
constexpr int PPO_VM_MAX_BLOCKS = 1024;

__device__ __forceinline__ float ppo_bsum(float v) {
    __shared__ float w[PPO_VM_THREADS / 32];
    int t = threadIdx.x, lane = t & 31, wid = t >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, o);
    }
    if (lane == 0) {
        w[wid] = v;
    }
    __syncthreads();
    v = (t < PPO_VM_THREADS / 32) ? w[t] : 0.0f;
    if (wid == 0) {
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) {
            v += __shfl_down_sync(0xffffffff, v, o);
        }
    }
    return v;  // valid on thread 0
}

__global__ void ppo_adv_moments(const precision_t* __restrict__ src,
        float* __restrict__ partial, int n) {
    float s = 0.0f, q = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
            i += blockDim.x * gridDim.x) {
        float x = to_float(src[i]);
        s += x;
        q += x * x;
    }
    float bs = ppo_bsum(s);
    __syncthreads();
    float bq = ppo_bsum(q);
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = bs;
        partial[blockIdx.x + gridDim.x] = bq;
    }
}

__global__ void ppo_adv_finalize(const float* __restrict__ partial,
        float* __restrict__ var_out, float* __restrict__ mean_out, int n_blocks, int n) {
    float s = 0.0f, q = 0.0f;
    for (int i = threadIdx.x; i < n_blocks; i += blockDim.x) {
        s += partial[i];
        q += partial[i + n_blocks];
    }
    float bs = ppo_bsum(s);
    __syncthreads();
    float bq = ppo_bsum(q);
    if (threadIdx.x == 0) {
        float mean = bs / (float)n;
        *mean_out = mean;
        // sample var; clamp tiny negatives from fp noise
        *var_out = fmaxf(0.0f, (bq - bs * mean) / (float)(n - 1));
    }
}

// This is a huge kernel for a relatively cheap operation. But without this,
// it's death by a thousand cuts with repeated kernel launches. Even graphed, you
// blow up the memory bandwidth.
void ppo_loss_fwd_bwd(
        PrecisionTensor& dec_out,    // (N, T, fused_cols) - fused logits+value from decoder
        PrecisionTensor& logstd,     // continuous logstd or empty
        TrainGraph& graph,
        IntTensor& act_sizes, FloatTensor& losses_acc,
        float clip_coef, float vf_clip_coef, float vf_coef, const float* ent_coef,
        PPOBuffersPuf& bufs, bool is_continuous,
        cudaStream_t stream) {
    int N = dec_out.shape[0], T = dec_out.shape[1], fused_cols = dec_out.shape[2];
    int A_total = fused_cols - 1;  // last column is value
    int total = N * T;

    float* adv_var = bufs.adv_scratch.data;
    float* adv_mean = adv_var + 1;
    float* adv_partials = adv_var + 2;
    int adv_n = (int)numel(graph.mb_advantages.shape);
    int adv_blocks = (adv_n + PPO_VM_THREADS - 1) / PPO_VM_THREADS;
    adv_blocks = min(adv_blocks, PPO_VM_MAX_BLOCKS);
    ppo_adv_moments<<<adv_blocks, PPO_VM_THREADS, 0, stream>>>(
        graph.mb_advantages.data, adv_partials, adv_n);
    ppo_adv_finalize<<<1, PPO_VM_THREADS, 0, stream>>>(
        adv_partials, adv_var, adv_mean, adv_blocks, adv_n);

    int ppo_grid = (total + PPO_THREADS - 1) / PPO_THREADS;

    PPOGraphArgs graph_args = {
        .out_ratio = graph.mb_ratio.data,
        .out_newvalue = graph.mb_newvalue.data,
        .actions = graph.mb_actions.data,
        .old_logprobs = graph.mb_logprobs.data,
        .advantages = graph.mb_advantages.data,
        .prio = graph.mb_prio.data,
        .values = graph.mb_values.data,
        .returns = graph.mb_returns.data,
    };

    PPOKernelArgs args = {
        .grad_logits = bufs.grad_logits.data,
        .grad_logstd = is_continuous ? bufs.grad_logstd.data : nullptr,
        .grad_values_pred = bufs.grad_values.data,
        .logits = dec_out.data,
        .logstd = is_continuous ? logstd.data : nullptr,
        .values_pred = dec_out.data + A_total,
        .adv_mean = adv_mean,
        .adv_var = adv_var,
        .act_sizes = act_sizes.data,
        .action_mask = graph.mb_action_mask.data,
        .num_atns = (int)numel(act_sizes.shape),
        .clip_coef = clip_coef, .vf_clip_coef = vf_clip_coef,
        .vf_coef = vf_coef, .ent_coef = ent_coef,
        .T_seq = T, .A_total = A_total, .N = N,
        .is_continuous = is_continuous,
    };

    ppo_loss_compute<<<ppo_grid, PPO_THREADS, 0, stream>>>(bufs.ppo_partials.data, args, graph_args);

    ppo_loss_reduce<<<1, LOSS_N, 0, stream>>>(
        losses_acc.data, bufs.ppo_partials.data, ppo_grid);
}

// Puffer advantage (GAE/V-Trace generalization). Horizon must be a multiple of
// ADV_VEC_WIDTH (asserted at create) so rows vectorize with 128-bit loads.
constexpr int ADV_VEC_WIDTH = 16 / (int)sizeof(precision_t);

__device__ __forceinline__ void adv_vec_load(const float* ptr, float* out) {
    float4 v = *reinterpret_cast<const float4*>(ptr);
    out[0] = v.x;
    out[1] = v.y;
    out[2] = v.z;
    out[3] = v.w;
}

__device__ __forceinline__ void adv_vec_load(const __nv_bfloat16* ptr, float* out) {
    uint4 raw = *reinterpret_cast<const uint4*>(ptr);
    const __nv_bfloat16* bf = reinterpret_cast<const __nv_bfloat16*>(&raw);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[i] = __bfloat162float(bf[i]);
    }
}

__device__ __forceinline__ void adv_vec_store(float* ptr, const float* vals) {
    *reinterpret_cast<float4*>(ptr) = make_float4(vals[0], vals[1], vals[2], vals[3]);
}

__device__ __forceinline__ void adv_vec_store(__nv_bfloat16* ptr, const float* vals) {
    __nv_bfloat16 tmp[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        tmp[i] = __float2bfloat16(vals[i]);
    }
    *reinterpret_cast<uint4*>(ptr) = *reinterpret_cast<const uint4*>(tmp);
}

__device__ __forceinline__ void puff_advantage_row(
        const precision_t* values, const precision_t* rewards, const precision_t* dones,
        const precision_t* importance, precision_t* advantages, float gamma, float lambda,
        float rho_clip, float c_clip, int horizon) {
    float lastpufferlam = 0.0f;
    int num_chunks = horizon / ADV_VEC_WIDTH;

    float next_value = to_float(values[horizon - 1]);
    float next_done = to_float(dones[horizon - 1]);
    float next_reward = to_float(rewards[horizon - 1]);

    for (int chunk = num_chunks - 1; chunk >= 0; chunk--) {
        int base = chunk * ADV_VEC_WIDTH;

        float v[ADV_VEC_WIDTH];
        float r[ADV_VEC_WIDTH];
        float d[ADV_VEC_WIDTH];
        float imp[ADV_VEC_WIDTH];
        adv_vec_load(values + base, v);
        adv_vec_load(rewards + base, r);
        adv_vec_load(dones + base, d);
        adv_vec_load(importance + base, imp);

        float adv[ADV_VEC_WIDTH] = {0};
        int start_idx = (chunk == num_chunks - 1) ? (ADV_VEC_WIDTH - 2) : (ADV_VEC_WIDTH - 1);

        #pragma unroll
        for (int i = start_idx; i >= 0; i--) {
            float nextnonterminal = 1.0f - next_done;
            float rho_t = fminf(imp[i], rho_clip);
            float c_t = fminf(imp[i], c_clip);
            float delta = rho_t * (next_reward + gamma * next_value * nextnonterminal - v[i]);
            lastpufferlam = delta + gamma * lambda * c_t * lastpufferlam * nextnonterminal;
            adv[i] = lastpufferlam;
            next_value = v[i];
            next_done = d[i];
            next_reward = r[i];
        }

        adv_vec_store(advantages + base, adv);
    }
}

__global__ void puff_advantage(const precision_t* values, const precision_t* rewards,
        const precision_t* dones, const precision_t* importance, precision_t* advantages, float gamma,
        float lambda, float rho_clip, float c_clip, int num_steps, int horizon) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_steps) {
        return;
    }
    int offset = row * horizon;
    puff_advantage_row(values + offset, rewards + offset, dones + offset,
        importance + offset, advantages + offset, gamma, lambda, rho_clip, c_clip, horizon);
}

void puff_advantage_cuda(PrecisionTensor& values, PrecisionTensor& rewards,
        PrecisionTensor& dones, PrecisionTensor& importance, PrecisionTensor& advantages,
        float gamma, float lambda, float rho_clip, float c_clip, cudaStream_t stream) {
    int num_steps = values.shape[0];
    int horizon = values.shape[1];
    puff_advantage<<<grid_size(num_steps), 256, 0, stream>>>(
        values.data, rewards.data, dones.data, importance.data,
        advantages.data, gamma, lambda, rho_clip, c_clip, num_steps, horizon);
}

#endif // PUFFERLIB_ALGO_CU
