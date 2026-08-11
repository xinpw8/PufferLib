// PufferNet model API + architecture
// Writing custom nets in 4.0+ requires a fair bit of code because you are
// responsible for defining your own activation and gradient buffers.
// You usually only ever need a custom Encoder.
typedef void (*init_weights_fn)(void* weights, ulong* seed, cudaStream_t stream);
typedef void (*reg_params_fn)(void* weights, Allocator* alloc);
typedef void (*reg_train_fn)(void* weights, void* buf, Allocator* acts, Allocator* grads, int B_TT);
typedef void (*reg_rollout_fn)(void* weights, void* buf, Allocator* alloc, int B);
typedef void* (*create_weights_fn)(void* self);
typedef Prec (*forward_fn)(void* weights, void* activations, Prec input, cudaStream_t stream);
typedef void (*encoder_backward_fn)(void* weights, void* activations,
    Prec grad, cudaStream_t stream);
typedef Prec (*decoder_backward_fn)(void* weights, void* activations,
    Float grad_logits, Float grad_logstd, Float grad_value, cudaStream_t stream);
typedef Prec (*network_forward_fn)(void* weights, Prec x,
    Prec state, void* activations, cudaStream_t stream);
typedef Prec (*network_forward_train_fn)(void* weights, Prec x,
    Prec state, Prec terminals, void* activations, cudaStream_t stream);
typedef Prec (*network_backward_fn)(void* weights,
    Prec grad, void* activations, cudaStream_t stream);

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

struct EncoderWeights {
    Prec weight;
    int in_dim, out_dim;
};

struct EncoderActivations {
    Prec out, saved_input, wgrad_scratch;
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

struct DecoderWeights {
    Prec weight, logstd;
    int hidden_dim, output_dim;
    bool continuous;
};

struct DecoderActivations {
    Prec out, grad_out, saved_input, grad_input, wgrad_scratch, logstd_scratch;
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

thread_local cublasHandle_t g_cublas_handle = NULL;
thread_local void* g_cublas_workspace = NULL;
// Side-stream dW (mm_tn) overlaps dX (mm_nn) on main during linear bwd.
thread_local cublasHandle_t g_cublas_dw_handle = NULL;
thread_local cudaEvent_t g_main_ready = NULL;
thread_local void* g_cublas_dw_workspace = NULL;
thread_local cudaStream_t g_dw_stream = NULL;
thread_local cudaEvent_t g_dw_done = NULL;

static void cublas_init_one(cublasHandle_t* handle, void** workspace) {
    const size_t ws_bytes = 32 * 1024 * 1024;
    cublasCreate(handle);
    cudaMalloc(workspace, ws_bytes);
    cublasSetWorkspace(*handle, *workspace, ws_bytes);
    cublasSetMathMode(*handle, CUBLAS_DEFAULT_MATH);
}

void cublas_init_handle() {
    cublas_init_one(&g_cublas_handle, &g_cublas_workspace);
    cublas_init_one(&g_cublas_dw_handle, &g_cublas_dw_workspace);
    cudaStreamCreateWithFlags(&g_dw_stream, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&g_dw_done, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&g_main_ready, cudaEventDisableTiming);
}

// Dense row-major GEMM: C = alpha * op_a(A) @ op_b(B) + beta * C
static void cublasGemmExDense(cublasHandle_t handle,
        cublasOperation_t op_a, cublasOperation_t op_b,
        int M, int N, int K, void* A, void* B, void* C,
        cudaStream_t stream, float alpha = 1.0f, float beta = 0.0f) {
    int lda = (op_a == CUBLAS_OP_N) ? K : M;
    int ldb = (op_b == CUBLAS_OP_N) ? N : K;
    cublasSetStream(handle, stream);
    cublasGemmEx(handle, op_b, op_a, N, M, K, &alpha,
        B, CUBLAS_PRECISION, ldb, A, CUBLAS_PRECISION, lda, &beta,
        C, CUBLAS_PRECISION, N, CUBLAS_COMPUTE, CUBLAS_GEMM_DEFAULT);
}

// out(...,N) = alpha * a(...,K) @ b(N,K)^T + beta * out  — leading dims folded into M
void puf_mm(Prec* a, Prec* b, Prec* out, cudaStream_t stream,
        float alpha = 1.0f, float beta = 0.0f) {
    int M = batch_size(a->shape) * a->shape[ndim(a->shape)-2];
    int K = a->shape[ndim(a->shape)-1];
    int N = b->shape[ndim(b->shape)-2];
    cublasGemmExDense(g_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K,
        a->data, b->data, out->data, stream, alpha, beta);
}

// out(M,N) = alpha * a(...,M)^T @ b(...,N) + beta * out  — leading dims folded into K
void puf_mm_tn(Prec* a, Prec* b, Prec* out, cudaStream_t stream,
        float alpha = 1.0f, float beta = 0.0f,
        cublasHandle_t handle = g_cublas_handle) {
    int M = a->shape[ndim(a->shape)-1];
    int K = batch_size(a->shape) * a->shape[ndim(a->shape)-2];
    int N = b->shape[ndim(b->shape)-1];
    cublasGemmExDense(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K,
        a->data, b->data, out->data, stream, alpha, beta);
}

// out(...,N) = alpha * a(...,K) @ b(K,N) + beta * out  — leading dims folded into M
void puf_mm_nn(Prec* a, Prec* b, Prec* out, cudaStream_t stream,
        float alpha = 1.0f, float beta = 0.0f) {
    int M = batch_size(a->shape) * a->shape[ndim(a->shape)-2];
    int K = a->shape[ndim(a->shape)-1];
    int N = b->shape[ndim(b->shape)-1];
    cublasGemmExDense(g_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
        a->data, b->data, out->data, stream, alpha, beta);
}

// Queue dW (mm_tn) on side stream once inputs are ready on main. Per-layer
// wgrad buffers are disjoint so multiple dWs can queue; puf_dw_join before muon.
void puf_mm_tn_async_after(Prec* a, Prec* b, Prec* out, cudaStream_t main_stream) {
    cudaEventRecord(g_main_ready, main_stream);
    cudaStreamWaitEvent(g_dw_stream, g_main_ready, 0);
    puf_mm_tn(a, b, out, g_dw_stream, 1.0f, 0.0f, g_cublas_dw_handle);
}

void puf_dw_join(cudaStream_t consumer) {
    cudaEventRecord(g_dw_done, g_dw_stream);
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
void puf_kaiming_init(Prec* dst,
        float gain, ulong seed, cudaStream_t stream) {
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
    precision_t* combined_ptr = NULL;
    precision_t* state_ptr = NULL;
    precision_t* input_ptr = NULL;  // (B, T, H) pre-projection input (highway)
    precision_t* terminals_ptr = NULL;  // (B, T), reset before step t if terminal[t]
    int B = 0, T = 0, H = 0;
    Prec scan_h;  // (B, T, H), recurrent input to each step
    Prec out, next_state;
    Prec grad_combined, grad_state;
    Prec grad_input;
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
        if (terminals != NULL && to_float(terminals[b * T_seq + t]) != 0.0f) {
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
        if (terminals != NULL &&
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

Prec encoder_forward(void* w, void* activations, Prec input, cudaStream_t stream) {
    EncoderWeights* ew = (EncoderWeights*)w;
    EncoderActivations* a = (EncoderActivations*)activations;
    // Train acts register saved_input; rollout acts leave it null.
    if (a->saved_input.data) {
        puf_copy(&a->saved_input, &input, stream);
    }
    puf_mm(&input, &ew->weight, &a->out, stream);
    return a->out;
}

void encoder_backward(void* w, void* activations, Prec grad, cudaStream_t stream) {
    EncoderActivations* a = (EncoderActivations*)activations;
    // Last layer before join: nothing left on main to overlap, so sync dW.
    puf_mm_tn(&grad, &a->saved_input, &a->wgrad_scratch, stream);
}

void encoder_init_weights(void* w, ulong* seed, cudaStream_t stream) {
    EncoderWeights* ew = (EncoderWeights*)w;
    Prec wt = {
        .data = ew->weight.data,
        .shape = {ew->out_dim, ew->in_dim},
    };
    puf_kaiming_init(&wt, sqrtf(2.0f), (*seed)++, stream);
}

void encoder_reg_params(void* w, Allocator* alloc) {
    EncoderWeights* ew = (EncoderWeights*)w;
    ew->weight = {.shape = {ew->out_dim, ew->in_dim}};
    alloc_register(alloc, &ew->weight);
}

void encoder_reg_train(void* w, void* activations,
        Allocator* acts, Allocator* grads, int B_TT) {
    EncoderWeights* ew = (EncoderWeights*)w;
    EncoderActivations* a = (EncoderActivations*)activations;
    *a = (EncoderActivations){
        .out =              {.shape = {B_TT, ew->out_dim}},
        .saved_input =      {.shape = {B_TT, ew->in_dim}},
        .wgrad_scratch =    {.shape = {ew->out_dim, ew->in_dim}},
    };
    alloc_register(acts, &a->out);
    alloc_register(acts, &a->saved_input);
    alloc_register(grads, &a->wgrad_scratch);
}

void encoder_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    EncoderWeights* ew = (EncoderWeights*)w;
    EncoderActivations* a = (EncoderActivations*)activations;
    a->out = {.shape = {B, ew->out_dim}};
    alloc_register(alloc, &a->out);
}

void* encoder_create_weights(void* self) {
    Encoder* e = (Encoder*)self;
    EncoderWeights* ew = (EncoderWeights*)calloc(1, sizeof(EncoderWeights));
    ew->in_dim = e->in_dim; ew->out_dim = e->out_dim;
    return ew;
}

Prec decoder_forward(void* w, void* activations, Prec input, cudaStream_t stream) {
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
    Prec wt = {
        .data = dw->weight.data,
        .shape = {dw->output_dim + 1, dw->hidden_dim},
    };
    puf_kaiming_init(&wt, 1.0f, (*seed)++, stream);
}

void decoder_reg_params(void* w, Allocator* alloc) {
    DecoderWeights* dw = (DecoderWeights*)w;
    dw->weight = {.shape = {dw->output_dim + 1, dw->hidden_dim}};
    alloc_register(alloc, &dw->weight);
    if (dw->continuous) {
        dw->logstd = {.shape = {1, dw->output_dim}};
        alloc_register(alloc,&dw->logstd);
    }
}

void decoder_reg_train(void* w, void* activations,
        Allocator* acts, Allocator* grads, int B_TT) {
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
    alloc_register(acts, &a->out);
    alloc_register(acts, &a->saved_input);
    alloc_register(acts, &a->grad_out);
    alloc_register(acts, &a->grad_input);
    alloc_register(grads, &a->wgrad_scratch);
    if (dw->continuous) {
        alloc_register(grads, &a->logstd_scratch);
    }
}

void decoder_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    DecoderWeights* dw = (DecoderWeights*)w;
    DecoderActivations* a = (DecoderActivations*)activations;
    a->out = {.shape = {B, dw->output_dim + 1}};
    alloc_register(alloc, &a->out);
}

void* decoder_create_weights(void* self) {
    Decoder* d = (Decoder*)self;
    DecoderWeights* dw = (DecoderWeights*)calloc(1, sizeof(DecoderWeights));
    dw->hidden_dim = d->hidden_dim;
    dw->output_dim = d->output_dim;
    dw->continuous = d->continuous;
    return dw;
}

Prec decoder_backward(void* w, void* activations, Float grad_logits,
        Float grad_logstd, Float grad_value, cudaStream_t stream) {
    DecoderWeights* dw = (DecoderWeights*)w;
    DecoderActivations* a = (DecoderActivations*)activations;
    int B_TT = a->saved_input.shape[0];
    int od = dw->output_dim, od1 = od + 1;
    assemble_decoder_grad<<<grid_size(B_TT * od1), BLOCK_SIZE, 0, stream>>>(
        a->grad_out.data, grad_logits.data, grad_value.data, B_TT, od, od1);
    // dW // dX: weight grad on side stream, dX on main (needed for residual chain).
    puf_mm_tn_async_after(&a->grad_out, &a->saved_input, &a->wgrad_scratch, stream);
    if (dw->continuous && grad_logstd.data != NULL) {
        sum_rows_to_precision_kernel<<<grid_size(dw->output_dim), BLOCK_SIZE, 0, stream>>>(
            a->logstd_scratch.data, grad_logstd.data, B_TT, dw->output_dim);
    }
    puf_mm_nn(&a->grad_out, &dw->weight, &a->grad_input, stream);
    return a->grad_input;
}

struct MinGRUActivations {
    // Rollout
    Prec* combined;       // (B rollout, 3*T)[num_layers]
    Prec out;             // (B rollout, T)
    Prec next_state;      // (B rollout, T)
    // Training
    Prec* saved_inputs;   // (B, TT, T)[num_layers]
    PrefixScan* scan_bufs;           // [num_layers]
    Prec* combined_bufs;  // (B*TT, 3*T)[num_layers]
    Prec* wgrad_scratch;  // (3*T, T)[num_layers]
    Prec grad_input_buf;  // (B*TT, T)
    Prec grad_next_state; // (B, 1, T)
};

struct MinGRUWeights {
    int hidden, num_layers, horizon;
    Prec* weights;  // [num_layers]
};

Prec mingru_state_layer(MinGRUWeights* m, Prec& state, int i) {
    long B = state.shape[1], H = state.shape[2];
    return {.data = state.data + i * B * H, .shape = {B, H}};
}

void mingru_init_weights(void* w, ulong* seed, cudaStream_t stream) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    for (int i = 0; i < m->num_layers; i++) {
        Prec w2d = {
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
        alloc_register(alloc, &m->weights[i]);
    }
}

void mingru_reg_train(void* w, void* activations,
        Allocator* acts, Allocator* grads, int B_TT) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    int H = m->hidden, TT = m->horizon, B = B_TT / TT;
    a->saved_inputs = (Prec*)calloc(m->num_layers, sizeof(Prec));
    a->scan_bufs = (PrefixScan*)calloc(m->num_layers, sizeof(PrefixScan));
    a->combined_bufs = (Prec*)calloc(m->num_layers, sizeof(Prec));
    a->wgrad_scratch = (Prec*)calloc(m->num_layers, sizeof(Prec));
    a->grad_input_buf = {.shape = {B_TT, H}};
    a->grad_next_state = {.shape = {B, 1, H}};
    alloc_register(acts, &a->grad_input_buf);
    alloc_register(acts, &a->grad_next_state);
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
        alloc_register(acts, &a->saved_inputs[i]);
        alloc_register(acts, &a->combined_bufs[i]);
        alloc_register(acts, &a->scan_bufs[i].out);
        alloc_register(acts, &a->scan_bufs[i].next_state);
        alloc_register(acts, &a->scan_bufs[i].scan_h);
        alloc_register(acts, &a->scan_bufs[i].grad_combined);
        alloc_register(acts, &a->scan_bufs[i].grad_state);
        alloc_register(acts, &a->scan_bufs[i].grad_input);
        alloc_register(grads, &a->wgrad_scratch[i]);
    }
}

void mingru_reg_rollout(void* weights, void* activations,
        Allocator* alloc, int B_inf) {
    MinGRUWeights* w = (MinGRUWeights*)weights;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    int H = w->hidden;
    a->combined = (Prec*)calloc(w->num_layers, sizeof(Prec));
    for (int i = 0; i < w->num_layers; i++) {
        a->combined[i] = {.shape = {B_inf, 3 * H}};
        alloc_register(alloc, &a->combined[i]);
    }
    a->out = {.shape = {B_inf, H}};
    a->next_state = {.shape = {B_inf, H}};
    alloc_register(alloc, &a->out);
    alloc_register(alloc, &a->next_state);
}

void* mingru_create_weights(void* self) {
    Network* n = (Network*)self;
    MinGRUWeights* mw = (MinGRUWeights*)calloc(1, sizeof(MinGRUWeights));
    mw->hidden = n->hidden;
    mw->num_layers = n->num_layers;
    mw->horizon = n->horizon;
    mw->weights = (Prec*)calloc(n->num_layers, sizeof(Prec));
    return mw;
}

Prec mingru_forward(void* w, Prec x, Prec state,
        void* activations, cudaStream_t stream) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    int B = state.shape[1];
    int H = state.shape[2];
    for (int i = 0; i < m->num_layers; i++) {
        Prec state_i = mingru_state_layer(m, state, i);
        puf_mm(&x, &m->weights[i], &a->combined[i], stream);
        mingru_gate<<<grid_size(B*H), BLOCK_SIZE, 0, stream>>>(
            a->out.data, a->next_state.data,
            a->combined[i].data, state_i.data, x.data, H, B);
        puf_copy(&state_i, &a->next_state, stream);
        x = a->out;
    }
    return x;
}

Prec mingru_forward_train(void* w, Prec x, Prec state, Prec terminals,
        void* activations, cudaStream_t stream) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    for (int i = 0; i < m->num_layers; i++) {
        puf_copy(&a->saved_inputs[i], &x, stream);
        Prec state_i = mingru_state_layer(m, state, i);
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

__global__ void add_kernel(precision_t* __restrict__ dst,
        const precision_t* __restrict__ src, int n) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
            idx += blockDim.x * gridDim.x) {
        dst[idx] = from_float(to_float(dst[idx]) + to_float(src[idx]));
    }
}

Prec mingru_backward(void* w, Prec grad, void* activations, cudaStream_t stream) {
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

struct Arch {
    Encoder encoder;
    Decoder decoder;
    Network network;
};

struct Activations {
    void* encoder;
    void* decoder;
    void* network;
};

struct Weights {
    void* encoder;
    void* decoder;
    void* network;
};

Prec arch_forward(Arch* p, Weights& w, Activations& activations,
        Prec obs, Prec state, cudaStream_t stream) {
    Prec enc_out = p->encoder.forward(
        w.encoder, activations.encoder, obs, stream);
    Prec h = p->network.forward(
        w.network, enc_out, state, activations.network, stream);
    return p->decoder.forward(w.decoder, activations.decoder, h, stream);
}

Prec arch_forward_train(Arch* p, Weights& w,
        Activations& activations, Prec x,
        Prec state, Prec terminals, cudaStream_t stream) {
    int B = x.shape[0], TT = x.shape[1];
    Prec h = p->encoder.forward(w.encoder,
        activations.encoder, *puf_squeeze(&x, 0), stream);
    h = p->network.forward_train(w.network, *puf_unsqueeze(&h, 0, B, TT),
        state, terminals, activations.network, stream);
    Prec dec_out = p->decoder.forward(
        w.decoder, activations.decoder, *puf_squeeze(&h, 0), stream);
    return *puf_unsqueeze(&dec_out, 0, B, TT);
}

void arch_backward(Arch* p, Weights& w,
        Activations& activations, Float grad_logits,
        Float grad_logstd, Float grad_value, cudaStream_t stream) {
    int B = grad_logits.shape[0], TT = grad_logits.shape[1];
    Prec grad_h = p->decoder.backward(w.decoder,
        activations.decoder, *puf_squeeze(&grad_logits, 0),
        grad_logstd, *puf_squeeze(&grad_value, 0), stream);
    grad_h = p->network.backward(w.network,
        *puf_unsqueeze(&grad_h, 0, B, TT), activations.network, stream);
    p->encoder.backward(w.encoder, activations.encoder, grad_h, stream);
    puf_dw_join(stream); // sycn dW GEMMs before muon reads weight grads.
}

Activations arch_reg_train(Arch* p, Weights& w,
        Allocator* acts, Allocator* grads, int B_TT) {
    Activations a;
    a.encoder = calloc(1, p->encoder.activation_size);
    a.decoder = calloc(1, p->decoder.activation_size);
    a.network = calloc(1, sizeof(MinGRUActivations));
    p->encoder.reg_train(w.encoder, a.encoder, acts, grads, B_TT);
    p->decoder.reg_train(w.decoder, a.decoder, acts, grads, B_TT);
    p->network.reg_train(w.network, a.network, acts, grads, B_TT);
    return a;
}

Activations arch_reg_rollout(Arch* p, Weights& w,
        Allocator* acts, int B_inf) {
    Activations a;
    a.encoder = calloc(1, p->encoder.activation_size);
    a.decoder = calloc(1, p->decoder.activation_size);
    a.network = calloc(1, sizeof(MinGRUActivations));
    p->encoder.reg_rollout(w.encoder, a.encoder, acts, B_inf);
    p->decoder.reg_rollout(w.decoder, a.decoder, acts, B_inf);
    p->network.reg_rollout(w.network, a.network, acts, B_inf);
    return a;
}

void weights_init(Arch* p, Weights& w,
        uint64_t* seed, cudaStream_t stream) {
    p->encoder.init_weights(w.encoder, seed, stream);
    p->decoder.init_weights(w.decoder, seed, stream);
    p->network.init_weights(w.network, seed, stream);
}

Weights weights_create(Arch* p, Allocator* params) {
    Weights w;
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

// Build an Arch (ops + dims) for a given env. Encoder/decoder algorithms are
// fixed by the env; hidden_size/num_layers/horizon parameterize shape. Arch
// has no heap state so this returns by value; callers store it wherever.
Arch build_arch(const char* env_name, int input_size, int hidden_size,
        int num_layers, int decoder_output_size, bool is_continuous, int horizon) {
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
        .hidden_dim = hidden_size,
        .output_dim = decoder_output_size,
        .continuous = is_continuous,
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
        .hidden = hidden_size,
        .num_layers = num_layers,
        .horizon = horizon,
    };
    return Arch{
        .encoder = encoder, .decoder = decoder, .network = network,
    };
}

__global__ void muon_sum_sq_partials(float* __restrict__ partials,
        const precision_t* __restrict__ src, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        float v = to_float(src[i]);
        sum += v * v;
    }
    sdata[tid] = sum;
    block_reduce_sum(sdata, &partials[blockIdx.x], tid, blockDim.x, 1);
}

__global__ void muon_sum_sq_reduce(float* __restrict__ out,
        const float* __restrict__ partials, int num_blocks) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    sdata[tid] = (tid < num_blocks) ? partials[tid] : 0.0f;
    block_reduce_sum(sdata, out, tid, blockDim.x, 1);
}

// Global grad clip by L2, then Nesterov into f32 momentum buffer.
__global__ void muon_clip_nesterov(float* __restrict__ mb,
        precision_t* __restrict__ gc, const float* __restrict__ sum_sq_ptr,
        float max_norm, float eps, float mu, int n) {
    float clip_coef = fminf(max_norm / (sqrtf(*sum_sq_ptr) + eps), 1.0f);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = to_float(gc[idx]) * clip_coef;
        float m = mu * mb[idx] + g;
        mb[idx] = m;
        gc[idx] = from_float(g + mu * m);
    }
}

// x *= 1 / max(sqrt(sum_sq), eps)  — NS input normalize
__global__ void muon_l2_normalize(precision_t* __restrict__ dst,
        const float* __restrict__ sum_sq_ptr, float eps, int n) {
    float inv_norm = 1.0f / fmaxf(sqrtf(*sum_sq_ptr), eps);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float(to_float(dst[idx]) * inv_norm);
    }
}

// dst = scale * src  (write NS result + aspect scale into flat grad buffer)
__global__ void muon_store_update(precision_t* __restrict__ dst,
        const precision_t* __restrict__ src, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float(scale * to_float(src[idx]));
    }
}

// wb = wb * (1 - lr*wd) - lr * update  (update already scaled; one call for all params)
__global__ void muon_weight_update(float* __restrict__ wb,
        const precision_t* __restrict__ update,
        const float* __restrict__ lr_ptr, float wd, int n) {
    float lr = *lr_ptr;
    float wd_scale = 1.0f - lr * wd;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        wb[idx] = wb[idx] * wd_scale - lr * to_float(update[idx]);
    }
}

constexpr double ns_coeffs[5][3] = {
    {4.0848, -6.8946, 2.9270},
    {3.9505, -6.3029, 2.6377},
    {3.7418, -5.5913, 2.3037},
    {2.8769, -3.1427, 1.2046},
    {2.8366, -3.0525, 1.2012},
};

// Muon optimizer. Our benchmarks show this is a major
// upgrade over Adam (weight decay not needed in RL).
struct Muon {
    double momentum;
    // Scalars / scratch: raw device ptrs. Tensors: allocator.
    float* lr;
    float* grad_norm;
    float* ns_norm;
    float* norm_partials;  // 256
    Float mb;              // flat momentum buffer (param-sized)
    Prec gram, gram_buf, x_buf;
    Allocator* param_alloc;
};

void muon_init(Muon* m, Allocator* param_alloc, double momentum, Allocator* alloc) {
    m->momentum = momentum;
    m->param_alloc = param_alloc;
    cudaMalloc((void**)&m->lr, sizeof(float));
    cudaMalloc((void**)&m->grad_norm, sizeof(float));
    cudaMalloc((void**)&m->ns_norm, sizeof(float));
    cudaMalloc((void**)&m->norm_partials, 256 * sizeof(float));
    m->mb = {.shape = {param_alloc->total_elems}};
    alloc_register(alloc, &m->mb);
    long max_M = 0, max_N = 0;
    for (int _i = 0; _i < param_alloc->num_regs; _i++) {
        AllocEntry& e = param_alloc->regs[_i];
        if (ndim(e.shape) >= 2) {
            long R = e.shape[0], C = numel(e.shape) / R;
            max_M = max(max_M, min(R, C));
            max_N = max(max_N, max(R, C));
        }
    }
    m->gram =     {.shape = {max_M, max_M}};
    m->gram_buf = {.shape = {max_M, max_M}};
    m->x_buf =    {.shape = {max_M, max_N}};
    alloc_register(alloc, &m->gram);
    alloc_register(alloc, &m->gram_buf);
    alloc_register(alloc, &m->x_buf);
}

void muon_step(Muon* m, Float weights, Prec grads,
        float max_grad_norm, cudaStream_t stream = 0) {
    int n_grad = (int)numel(grads.shape);
    int sum_blocks = min((int)grid_size(n_grad), 256);
    muon_sum_sq_partials<<<sum_blocks, 256, 0, stream>>>(
        m->norm_partials, grads.data, n_grad);
    muon_sum_sq_reduce<<<1, 256, 0, stream>>>(
        m->grad_norm, m->norm_partials, sum_blocks);
    muon_clip_nesterov<<<grid_size(n_grad), BLOCK_SIZE, 0, stream>>>(
        m->mb.data, grads.data, m->grad_norm,
        max_grad_norm, 1e-6f, (float)m->momentum, n_grad);

    // Per-param NS into workspace; write scaled update back into flat grads.
    // 1D params already hold their update in-place (scale 1).
    long offset = 0;
    for (int _i = 0; _i < m->param_alloc->num_regs; _i++) {
        AllocEntry& e = m->param_alloc->regs[_i];
        precision_t* gc_ptr = grads.data + offset;
        long ne = numel(e.shape);
        offset += ne;
        if (ndim(e.shape) < 2) {
            continue;
        }

        long R = e.shape[0], C = ne / R;
        long M = min(R, C);
        bool tall = R > C;
        Prec x = {.data = gc_ptr, .shape = {R, C}};
        Prec x_buf = {.data = m->x_buf.data, .shape = {R, C}};
        Prec gram = {.data = m->gram.data, .shape = {M, M}};
        Prec gram_buf = {.data = m->gram_buf.data, .shape = {M, M}};

        int nblk = min((int)grid_size(ne), 256);
        muon_sum_sq_partials<<<nblk, 256, 0, stream>>>(
            m->norm_partials, x.data, (int)ne);
        muon_sum_sq_reduce<<<1, 256, 0, stream>>>(
            m->ns_norm, m->norm_partials, nblk);
        muon_l2_normalize<<<grid_size(ne), BLOCK_SIZE, 0, stream>>>(
            x.data, m->ns_norm, 1e-7f, (int)ne);

        // 5 steps land in x_buf. 4 = you break it.
        for (int i = 0; i < 5; ++i) {
            Prec& src = (i % 2 == 0) ? x : x_buf;
            Prec& dst = (i % 2 == 0) ? x_buf : x;
            if (tall) {
                puf_mm_tn(&src, &src, &gram, stream);
            } else {
                puf_mm(&src, &src, &gram, stream);
            }
            puf_copy(&gram_buf, &gram, stream);
            puf_mm_nn(&gram, &gram, &gram_buf, stream,
                (float)ns_coeffs[i][2], (float)ns_coeffs[i][1]);
            puf_copy(&dst, &src, stream);
            if (tall) {
                puf_mm_nn(&src, &gram_buf, &dst,
                    stream, 1.0f, (float)ns_coeffs[i][0]);
            } else {
                puf_mm_nn(&gram_buf, &src, &dst,
                    stream, 1.0f, (float)ns_coeffs[i][0]);
            }
        }
        float scale = sqrtf(fmaxf(1.0f, (float)R / (float)C));
        muon_store_update<<<grid_size(ne), BLOCK_SIZE, 0, stream>>>(
            gc_ptr, x_buf.data, scale, (int)ne);
    }
    muon_weight_update<<<grid_size(n_grad), BLOCK_SIZE, 0, stream>>>(
        weights.data, grads.data, m->lr, 0.0f, n_grad);
}

// Train data layout is transposed to (B, T) from rollouts layout (T, B)
// This allows env workers to collect data with contiguous writes and
// training to perform several (though not all) ops in contiguous memory
struct TrainGraph {
    Prec mb_state;       // (layers, B, hidden)
    Prec mb_obs;         // (B, T, input_size)
    Float mb_actions;    // (B, T, num_atns) float32: large discrete IDs
    Prec mb_logprobs;    // (B, T)
    Prec mb_terminals;   // (B, T), resets recurrent state before timestep t
    Prec mb_rewards;     // (B, T) for per-mb GAE
    Prec mb_advantages;  // ...
    Prec mb_values;
    Prec mb_returns;
    Prec mb_ratio;
    Prec mb_newvalue;
    Prec mb_prio;        // (B,)
    Prec mb_action_mask; // (B, T, mask_size); always allocated
    // Per-mb GAE scratch: ρ and backup V (not read by PPO loss).
    Prec mb_imp;
    Prec mb_gae_v;
};

void register_train_buffers(TrainGraph& bufs, Allocator* alloc,
        int B, int T, int input_size, int hidden_size,
        int num_atns, int num_layers, int mask_size) {
    bufs = (TrainGraph){
        .mb_state =         {.shape = {num_layers, B, hidden_size}},
        .mb_obs =           {.shape = {B, T, input_size}},
        .mb_actions =       {.shape = {B, T, num_atns}},
        .mb_logprobs =      {.shape = {B, T}},
        .mb_terminals =     {.shape = {B, T}},
        .mb_rewards =       {.shape = {B, T}},
        .mb_advantages =    {.shape = {B, T}},
        .mb_values =        {.shape = {B, T}},
        .mb_returns =       {.shape = {B, T}},
        .mb_ratio =         {.shape = {B, T}},
        .mb_newvalue =      {.shape = {B, T}},
        .mb_prio =          {.shape = {B}},
        .mb_action_mask =   {.shape = {B, T, mask_size}},
        .mb_imp =           {.shape = {B, T}},
        .mb_gae_v =         {.shape = {B, T}},
    };
    alloc_register(alloc, &bufs.mb_obs);
    alloc_register(alloc, &bufs.mb_state);
    alloc_register(alloc, &bufs.mb_actions);
    alloc_register(alloc, &bufs.mb_logprobs);
    alloc_register(alloc, &bufs.mb_terminals);
    alloc_register(alloc, &bufs.mb_rewards);
    alloc_register(alloc, &bufs.mb_advantages);
    alloc_register(alloc, &bufs.mb_prio);
    alloc_register(alloc, &bufs.mb_values);
    alloc_register(alloc, &bufs.mb_returns);
    alloc_register(alloc, &bufs.mb_ratio);
    alloc_register(alloc, &bufs.mb_newvalue);
    alloc_register(alloc, &bufs.mb_action_mask);
    alloc_register(alloc, &bufs.mb_imp);
    alloc_register(alloc, &bufs.mb_gae_v);
}

// Prioritized replay over single-epoch data. These kernels are
// the least cleaned because we will likely have a better method in 5.0.
struct PrioBuffers {
    Float prio_probs, cdf, mb_prio;
    Int idx;
};

void register_prio_buffers(PrioBuffers& bufs, Allocator* alloc,
        int B, int minibatch_segments) {
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

// Prio block for normalize/cdf/sample; 512 beat 256 on B>=8k in microbench.
#define PRIO_BLOCK_SIZE 512

// Per-trajectory |adv| reduction → unnormalized priority weight.
__global__ void prio_reduce(const precision_t* __restrict__ advantages,
        float* prio_weights, float prio_alpha, int stride) {
    int row = blockIdx.x;
    int tx = threadIdx.x;
    int offset = row * stride;

    float local_sum = 0.0f;
    for (int t = tx; t < stride; t += blockDim.x) {
        local_sum += fabsf(to_float(advantages[offset + t]));
    }

    for (int s = 16; s >= 1; s /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, s);
    }
    if (tx == 0) {
        float pw = __powf(local_sum, prio_alpha);
        if (isnan(pw) || isinf(pw)) {
            pw = 0.0f;
        }
        prio_weights[row] = pw;
    }
}

// Normalize prio_weights in place, then inclusive prefix sum → cdf.
// Single-block: sum → scale → chunked scan (same geometry as old normalize+cdf).
__global__ void prio_normalize_cdf(float* __restrict__ prio_weights,
        float* __restrict__ cdf, int B) {
    __shared__ float shmem[PRIO_BLOCK_SIZE / 32];
    __shared__ float block_sum;
    __shared__ float warp_tot[PRIO_BLOCK_SIZE / 32];
    __shared__ float warp_exc[PRIO_BLOCK_SIZE / 32];

    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    const float eps = 1e-6f;

    float local_sum = 0.0f;
    for (int t = tid; t < B; t += blockDim.x) {
        local_sum += prio_weights[t];
    }
    for (int s = 16; s >= 1; s /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, s);
    }
    if (lane == 0) {
        shmem[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < PRIO_BLOCK_SIZE / 32) ? shmem[lane] : 0.0f;
        for (int s = PRIO_BLOCK_SIZE / 64; s >= 1; s /= 2) {
            val += __shfl_down_sync(0xffffffff, val, s);
        }
        if (tid == 0) {
            block_sum = val + eps;
        }
    }
    __syncthreads();

    for (int t = tid; t < B; t += blockDim.x) {
        prio_weights[t] = (prio_weights[t] + eps) / block_sum;
    }
    __syncthreads();

    // Chunked inclusive scan into cdf
    const int chunk = (B + blockDim.x - 1) / blockDim.x;
    const int start = tid * chunk;
    const int end = min(start + chunk, B);

    float run = 0.0f;
    for (int i = start; i < end; i++) {
        run += prio_weights[i];
        cdf[i] = run;
    }
    float my_total = (start < B) ? run : 0.0f;

    float incl = my_total;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        float n = __shfl_up_sync(0xffffffff, incl, off);
        if (lane >= off) {
            incl += n;
        }
    }
    float warp_exclusive = incl - my_total;
    if (lane == 31) {
        warp_tot[warp_id] = incl;
    }
    __syncthreads();

    if (warp_id == 0) {
        float w = (lane < PRIO_BLOCK_SIZE / 32) ? warp_tot[lane] : 0.0f;
        float wins = w;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            float n = __shfl_up_sync(0xffffffff, wins, off);
            if (lane >= off) {
                wins += n;
            }
        }
        if (lane < PRIO_BLOCK_SIZE / 32) {
            warp_exc[lane] = wins - w;
        }
    }
    __syncthreads();

    float exclusive = warp_exclusive + warp_exc[warp_id];
    if (exclusive != 0.0f) {
        for (int i = start; i < end; i++) {
            cdf[i] += exclusive;
        }
    }
}

// Multinomial sample + importance weights. RNG offset bumped in a follow-up
// <<<1,1>>> so multi-block loads of *offset_ptr stay race-free.
__global__ void prio_sample(int* __restrict__ out_idx,
        float* __restrict__ mb_prio, const float* __restrict__ cdf,
        const float* __restrict__ prio_probs, int B, int num_samples,
        int total_agents, float anneal_beta, uint64_t seed,
        const int64_t* __restrict__ offset_ptr) {
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
    float value = prio_probs[lo] * (float)total_agents;
    mb_prio[tid] = __powf(value, -anneal_beta);
}

__global__ void prio_advance_rng(int64_t* __restrict__ offset_ptr, int64_t delta) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *offset_ptr += delta;
    }
}

// Build per-trajectory prio probs + CDF from advantages. Independent of the
// minibatch sample — call once per train_impl, not once per minibatch.
void prio_build_cdf_cuda(Prec& advantages, float prio_alpha,
        PrioBuffers& bufs, cudaStream_t stream) {
    int B = advantages.shape[0];
    int T = advantages.shape[1];
    prio_reduce<<<B, 32, 0, stream>>>(
        advantages.data, bufs.prio_probs.data, prio_alpha, T);
    prio_normalize_cdf<<<1, PRIO_BLOCK_SIZE, 0, stream>>>(
        bufs.prio_probs.data, bufs.cdf.data, B);
}

// Draw a minibatch index set from a prebuilt CDF and importance weights.
void prio_sample_cuda(float anneal_beta, PrioBuffers& bufs, ulong seed,
        long* offset_ptr, cudaStream_t stream) {
    int B = (int)bufs.cdf.shape[0];
    int N = (int)bufs.idx.shape[0];
    int blocks = (N + PRIO_BLOCK_SIZE - 1) / PRIO_BLOCK_SIZE;
    prio_sample<<<blocks, PRIO_BLOCK_SIZE, 0, stream>>>(
        bufs.idx.data, bufs.mb_prio.data, bufs.cdf.data, bufs.prio_probs.data,
        B, N, B, anneal_beta, seed, offset_ptr);
    prio_advance_rng<<<1, 1, 0, stream>>>(offset_ptr, N);
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

enum LossIdx {
    LOSS_PG = 0, LOSS_VF = 1, LOSS_ENT = 2, LOSS_TOTAL = 3,
    LOSS_OLD_APPROX_KL = 4, LOSS_APPROX_KL = 5, LOSS_CLIPFRAC = 6,
    LOSS_N = 7, NUM_LOSSES = 8,
};

// consumed-head gating is a experimental feature currently used by Nethack only
// Must be initialized *before* CUDA graph capture: cudaMalloc/Memcpy are
// illegal during stream capture, and the device pointer is baked into the
// captured sample_logits / ppo_loss kernels as a by-value arg.
#ifndef PUFFER_PROVIDES_HEAD_CONSUME_MAP
extern "C" __attribute__((weak)) const signed char* env_head_consume_map(int*, int*);
#endif
static const signed char* g_hc_dev = NULL;
static int g_hc_stride = 0;
static bool g_hc_init = false;

// Eager host->device upload. Call from create_pufferl (never during capture).
static void init_head_consume_map(void) {
    if (g_hc_init) return;
    g_hc_init = true;
    const char* hg = getenv("PUFFER_HEAD_GATING");
    if (!(hg && hg[0] && hg[0] != '0' && env_head_consume_map)) return;
    int nv = 0, na = 0;
    const signed char* host = env_head_consume_map(&nv, &na);
    if (!(host && nv > 0 && na > 0)) return;
    signed char* dev = NULL;
    assert(cudaMalloc(&dev, (size_t)nv * na) == cudaSuccess
            && "head_consume cudaMalloc failed");
    assert(cudaMemcpy(dev, host, (size_t)nv * na, cudaMemcpyHostToDevice) == cudaSuccess
            && "head_consume cudaMemcpy failed");
    g_hc_dev = dev;
    g_hc_stride = na;
}

static const signed char* get_head_consume_dev(int* stride) {
    // Lazy path for non-cudagraph / tests; create_pufferl pre-inits for graphs.
    if (!g_hc_init) init_head_consume_map();
    *stride = g_hc_stride;
    return g_hc_dev;
}

constexpr int PPO_THREADS = 256;

// Per-env from ENV_HEADER (ocean/<env>/<env>.h).
#ifndef NUM_ATNS
#error "ENV_HEADER must #define NUM_ATNS (number of action heads)"
#endif
#ifndef ACT_SIZES
#error "ENV_HEADER must #define ACT_SIZES { ... } (classes per head)"
#endif
// Exact max head width for this env build — discrete logit cache size.
constexpr int ppo_max_head_classes() {
    constexpr int s[] = ACT_SIZES;
    int m = 0;
    for (int i = 0; i < NUM_ATNS; i++) {
        if (s[i] > m) {
            m = s[i];
        }
    }
    return m > 0 ? m : 1;
}
constexpr int PPO_MAX_HEAD_A = ppo_max_head_classes();

// Fused loss function. PPO clipped loss + value + entropy
// buffers + args are quite complex. We do the entire
// forward + backwards pass for the full loss function in one kernel
struct PPOGraphArgs {
    precision_t* out_ratio;
    precision_t* out_newvalue;
    const float* actions;
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
    // Running reward mean/var [2]; scale PG A by 1/√var when reward_std_scale.
    const float* reward_stats;
    const int* act_sizes;
    const precision_t* action_mask; // (N, T, A_total); always present
    const signed char* head_consume; // (nverbs, num_atns) or NULL
    int hc_stride;
    int num_atns;
    float clip_coef, vf_clip_coef, vf_coef;
    float reward_scale_max;  // cap on 1/√var; 0 = uncapped
    const float* ent_coef;  // device ptr — host by-value bakes into CUDA graphs
    int T_seq, A_total, N;
    bool is_continuous;
    // Batch mean/std whitening of A in PG. Off = raw A (PG scale tracks |A|).
    bool normalize_advantages;
    // Scale PG A by 1/√(EMA reward var). Not batch whitening.
    bool reward_std_scale;
};

// adv_scratch layout: [var, mean, 2*PPO_VM_MAX_BLOCKS partials, flag-as-float-slot]
#define PPO_ADV_SCRATCH_N (2 + 2 * 1024 + 1)

struct PPOBufs {
    Float grad_logits, grad_values, grad_logstd;
    Float ppo_partials;
    float* ent_coef;     // device scalar (graphs cannot bake host by-value)
    float* adv_scratch;  // raw workspace, not a real tensor
};

void register_ppo_buffers(PPOBufs& bufs, Allocator* alloc, int N, int T, int A_total, bool is_continuous) {
    long total = (long)N * T;
    int ppo_grid = ((int)total + PPO_THREADS - 1) / PPO_THREADS;
    bufs = (PPOBufs){
        .grad_logits = {.shape = {N, T, A_total}},
        .grad_values = {.shape = {N, T, 1}},
        .grad_logstd = {.shape = {N, T, A_total}},
        .ppo_partials = {.shape = {ppo_grid * LOSS_N}},
        .ent_coef = NULL,
        .adv_scratch = NULL,
    };
    alloc_register(alloc, &bufs.grad_logits);
    alloc_register(alloc, &bufs.grad_values);
    if (is_continuous) {
        alloc_register(alloc, &bufs.grad_logstd);
    }
    alloc_register(alloc, &bufs.ppo_partials);
    cudaMalloc((void**)&bufs.ent_coef, sizeof(float));
    cudaMalloc((void**)&bufs.adv_scratch, PPO_ADV_SCRATCH_N * sizeof(float));
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

// Shared by sample_logits + ppo_loss. Fills cache[0..A) (sized PPO_MAX_HEAD_A).
__device__ __forceinline__ float ppo_discrete_logsumexp(
        const precision_t* __restrict__ logits, int logits_base,
        int logits_offset, int A,
        const precision_t* __restrict__ mask, int mask_base,
        float* __restrict__ cache) {
    float max_logit = -INFINITY;
    float sum = 0.0f;
    for (int a = 0; a < A; ++a) {
        float l = load_logit_masked(
            logits, logits_base, logits_offset, a, mask, mask_base);
        cache[a] = l;
        if (l > max_logit) {
            sum *= __expf(max_logit - l);
            max_logit = l;
        }
        sum += __expf(l - max_logit);
    }
    return max_logit + __logf(sum);
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

    if (idx < total_elements) {
        int n = idx / a.T_seq;
        int t = idx % a.T_seq;
        int nt = n * a.T_seq + t;
        int logits_base = nt * (a.A_total + 1);
        int at_base = nt * a.A_total;  // logits-grad + mask base (A_total cols)

        float old_logp = to_float(g.old_logprobs[nt]);
        float adv = to_float(g.advantages[nt]);
        float w = to_float(g.prio[n]);
        float val = to_float(g.values[nt]);
        float ret = to_float(g.returns[nt]);
        float val_pred = to_float(a.values_pred[logits_base]);
        g.out_newvalue[nt] = from_float(val_pred);

        // Optional batch whitening of A. Raw A: PG scale tracks |A|.
        float adv_for_pg = adv;
        if (a.normalize_advantages) {
            float adv_std = sqrtf(a.adv_var[0]);
            adv_for_pg = (adv - a.adv_mean[0]) / (adv_std + 1e-8f);
        }
        // Running reward-std scale (5.0-simplecl): A *= 1/√var_run. Not batch.
        if (a.reward_std_scale && a.reward_stats != NULL) {
            float reward_scale = rsqrtf(fmaxf(a.reward_stats[1], 0.0f) + 1e-8f);
            if (a.reward_scale_max > 0.0f) {
                reward_scale = fminf(reward_scale, a.reward_scale_max);
            }
            adv_for_pg *= reward_scale;
        }
        float ent_coef = *a.ent_coef;
        float d_entropy_term = inv_NT * (-ent_coef);

        // Value loss + gradient: 0.5 * max((v-r)^2, (v_clip-r)^2).
        // When clipped term wins, v_clip is constant in val_pred → grad 0.
        float v_error = val_pred - val;
        float v_clipped = val + fmaxf(-a.vf_clip_coef, fminf(a.vf_clip_coef, v_error));
        float v_loss_unclipped = (val_pred - ret) * (val_pred - ret);
        float v_loss_clipped = (v_clipped - ret) * (v_clipped - ret);
        float v_loss = 0.5f * fmaxf(v_loss_unclipped, v_loss_clipped);
        float d_val_pred = (v_loss_clipped > v_loss_unclipped) ? 0.0f : (val_pred - ret);
        a.grad_values_pred[nt] = inv_NT * a.vf_coef * d_val_pred;

        float total_log_prob = 0.0f;
        float total_entropy = 0.0f;
        // Stash across policy fwd → d_new_logp → bwd (need total logp before head grads).
        float head_logsumexp[NUM_ATNS];
        float head_entropy[NUM_ATNS];
        int head_act[NUM_ATNS];
        int head_used[NUM_ATNS];
        float logit_cache[NUM_ATNS][PPO_MAX_HEAD_A];
        float c_mean[NUM_ATNS], c_logstd[NUM_ATNS], c_action[NUM_ATNS];

        if (a.is_continuous) {
            for (int h = 0; h < a.num_atns; ++h) {
                c_mean[h] = safe_continuous_mean(a.logits, logits_base + h);
                c_logstd[h] = safe_continuous_logstd(a.logstd, h);
                c_action[h] = finite_or_clamp(
                    g.actions[nt * a.num_atns + h], -1.0e6f, 1.0e6f);
                float lp, ent;
                ppo_continuous_head(c_mean[h], c_logstd[h], c_action[h], &lp, &ent);
                total_log_prob += lp;
                total_entropy += ent;
            }
        } else {
            int verb = (int)g.actions[nt * a.num_atns];
            int logits_offset = 0;
            for (int h = 0; h < a.num_atns; ++h) {
                int A = a.act_sizes[h];
                int act = (int)g.actions[nt * a.num_atns + h];
                head_act[h] = act;
                head_used[h] = (a.head_consume == NULL || h == 0)
                    ? 1 : (int)a.head_consume[verb * a.hc_stride + h];
                float* cache = logit_cache[h];
                float lse = ppo_discrete_logsumexp(
                    a.logits, logits_base, logits_offset,
                    A, a.action_mask, at_base, cache);
                float ent = 0.0f;
                for (int aa = 0; aa < A; ++aa) {
                    float logp = cache[aa] - lse;
                    ent -= __expf(logp) * logp;
                }
                head_logsumexp[h] = lse;
                head_entropy[h] = ent;
                if (head_used[h]) {
                    total_log_prob += cache[act] - lse;
                    total_entropy += ent;
                }
                logits_offset += A;
            }
        }

        float logratio = total_log_prob - old_logp;
        float ratio = __expf(logratio);
        g.out_ratio[nt] = from_float(ratio);
        float clip_lo = 1.0f - a.clip_coef;
        float clip_hi = 1.0f + a.clip_coef;
        float ratio_clipped = fmaxf(clip_lo, fminf(clip_hi, ratio));
        float wa = -w * adv_for_pg;
        float pg_loss1 = wa * ratio;
        float pg_loss2 = wa * ratio_clipped;
        // Classic PPO: min(r·A, clip(r)·A)  ⇒  grad ∝ A * r (when unclipped)
        float pg_loss = fmaxf(pg_loss1, pg_loss2);
        float d_ratio = wa * inv_NT;
        if (pg_loss2 > pg_loss1 && (ratio <= clip_lo || ratio >= clip_hi)) {
            d_ratio = 0.0f;
        }
        float d_new_logp = d_ratio * ratio;  // ∝ A * r

        if (a.is_continuous) {
            for (int h = 0; h < a.num_atns; ++h) {
                float std = __expf(c_logstd[h]);
                float var = std * std;
                float diff = c_action[h] - c_mean[h];
                a.grad_logits[at_base + h] = d_new_logp * diff / var;
                a.grad_logstd[nt * a.num_atns + h] =
                    d_new_logp * (diff * diff / var - 1.0f) + d_entropy_term;
            }
        } else {
            int logits_offset = 0;
            for (int h = 0; h < a.num_atns; ++h) {
                int A = a.act_sizes[h];
                if (!head_used[h]) {
                    for (int j = 0; j < A; ++j) {
                        a.grad_logits[at_base + logits_offset + j] = 0.0f;
                    }
                    logits_offset += A;
                    continue;
                }
                float lse = head_logsumexp[h];
                float ent = head_entropy[h];
                int act = head_act[h];
                float* cache = logit_cache[h];
                for (int j = 0; j < A; ++j) {
                    float logp = cache[j] - lse;
                    float p = __expf(logp);
                    a.grad_logits[at_base + logits_offset + j] =
                        ((j == act ? 1.0f : 0.0f) - p) * d_new_logp
                        + d_entropy_term * p * (-ent - logp);
                }
                logits_offset += A;
            }
        }

        float thread_loss = (pg_loss + a.vf_coef * v_loss
            - ent_coef * total_entropy) * inv_NT;

        block_losses[LOSS_PG][tid] = pg_loss * inv_NT;
        block_losses[LOSS_VF][tid] = v_loss * inv_NT;
        block_losses[LOSS_ENT][tid] = total_entropy * inv_NT;
        block_losses[LOSS_TOTAL][tid] = thread_loss;
        block_losses[LOSS_OLD_APPROX_KL][tid] = (-logratio) * inv_NT;
        block_losses[LOSS_APPROX_KL][tid] = ((ratio - 1.0f) - logratio) * inv_NT;
        block_losses[LOSS_CLIPFRAC][tid] =
            (fabsf(ratio - 1.0f) > a.clip_coef ? 1.0f : 0.0f) * inv_NT;
    }

    block_reduce_sum(&block_losses[0][0], &ppo_partials[blockIdx.x * LOSS_N],
        tid, PPO_THREADS, LOSS_N);
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

// Advantage mean + sample variance (one multi-block kernel).
// Each block: sum (x, x²) → partial. Last block (atomic ticket) folds partials.
// Multi-block data pass matters for large NT; see tests/bench_ppo_adv_mean.cu.
constexpr int PPO_VM_THREADS = 256;
constexpr int PPO_VM_MAX_BLOCKS = 1024;

// Just a mean and adv. Naive impl is annoyingly high overhead on small nets
__global__ void ppo_adv_mean_var(const precision_t* __restrict__ src,
        float* __restrict__ partial, float* __restrict__ var_out,
        float* __restrict__ mean_out, int* __restrict__ flag, int n) {
    float s = 0.0f, q = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
            i += blockDim.x * gridDim.x) {
        float x = to_float(src[i]);
        s += x;
        q += x * x;
    }
    // Block reduce (s, q) — warp shuffles then 8-way shared.
    __shared__ float wa[PPO_VM_THREADS / 32], wb[PPO_VM_THREADS / 32];
    int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        s += __shfl_down_sync(0xffffffff, s, o);
        q += __shfl_down_sync(0xffffffff, q, o);
    }
    if (lane == 0) {
        wa[wid] = s;
        wb[wid] = q;
    }
    __syncthreads();
    s = (tid < PPO_VM_THREADS / 32) ? wa[tid] : 0.0f;
    q = (tid < PPO_VM_THREADS / 32) ? wb[tid] : 0.0f;
    if (wid == 0) {
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) {
            s += __shfl_down_sync(0xffffffff, s, o);
            q += __shfl_down_sync(0xffffffff, q, o);
        }
    }
    if (tid == 0) {
        partial[blockIdx.x] = s;
        partial[blockIdx.x + gridDim.x] = q;
        __threadfence();  // publish partials before claiming "done"
        if (atomicAdd(flag, 1) == gridDim.x - 1) {
            float S = 0.0f, Q = 0.0f;
            for (int b = 0; b < gridDim.x; b++) {
                S += partial[b];
                Q += partial[b + gridDim.x];
            }
            float mean = S / (float)n;
            *mean_out = mean;
            *var_out = fmaxf(0.0f, (Q - S * mean) / (float)(n - 1));
            *flag = 0;  // ready for next launch (graphs re-run this kernel)
        }
    }
}

// This is a huge kernel for a relatively cheap operation. But without this,
// it's death by a thousand cuts with repeated kernel launches. Even graphed, you
// blow up the memory bandwidth.
void ppo_loss_fwd_bwd(
        Prec& dec_out,    // (N, T, fused_cols) - fused logits+value from decoder
        Prec& logstd,     // continuous logstd or empty
        TrainGraph& graph,
        int* act_sizes, float* losses_acc,
        float clip_coef, float vf_clip_coef, float vf_coef, const float* ent_coef,
        PPOBufs& bufs, bool is_continuous,
        bool normalize_advantages,
        bool reward_std_scale,
        float* reward_stats,
        float reward_scale_max,
        cudaStream_t stream) {
    int N = dec_out.shape[0], T = dec_out.shape[1], fused_cols = dec_out.shape[2];
    int A_total = fused_cols - 1;  // last column is value
    int total = N * T;

    float* adv_var = bufs.adv_scratch;
    float* adv_mean = adv_var + 1;
    if (normalize_advantages) {
        float* adv_partials = adv_var + 2;
        int* adv_flag = (int*)(adv_var + 2 + 2 * PPO_VM_MAX_BLOCKS);
        int adv_n = (int)numel(graph.mb_advantages.shape);
        int adv_blocks = min((adv_n + PPO_VM_THREADS - 1) / PPO_VM_THREADS, PPO_VM_MAX_BLOCKS);
        ppo_adv_mean_var<<<adv_blocks, PPO_VM_THREADS, 0, stream>>>(
            graph.mb_advantages.data, adv_partials, adv_var, adv_mean, adv_flag, adv_n);
    }

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

    int hc_stride_l = 0;
    const signed char* hc_dev_l = get_head_consume_dev(&hc_stride_l);
    PPOKernelArgs args = {
        .grad_logits = bufs.grad_logits.data,
        .grad_logstd = is_continuous ? bufs.grad_logstd.data : NULL,
        .grad_values_pred = bufs.grad_values.data,
        .logits = dec_out.data,
        .logstd = is_continuous ? logstd.data : NULL,
        .values_pred = dec_out.data + A_total,
        .adv_mean = adv_mean,
        .adv_var = adv_var,
        .reward_stats = reward_stats,
        .act_sizes = act_sizes,
        .action_mask = graph.mb_action_mask.data,
        .head_consume = hc_dev_l,
        .hc_stride = hc_stride_l,
        .num_atns = NUM_ATNS,
        .clip_coef = clip_coef, .vf_clip_coef = vf_clip_coef,
        .vf_coef = vf_coef,
        .reward_scale_max = reward_scale_max,
        .ent_coef = ent_coef,
        .T_seq = T, .A_total = A_total, .N = N,
        .is_continuous = is_continuous,
        .normalize_advantages = normalize_advantages,
        .reward_std_scale = reward_std_scale,
    };
    ppo_loss_compute<<<ppo_grid, PPO_THREADS, 0, stream>>>(
            bufs.ppo_partials.data, args, graph_args);
    ppo_loss_reduce<<<1, LOSS_N, 0, stream>>>(
        losses_acc, bufs.ppo_partials.data, ppo_grid);
}

// Puffer advantage (GAE + V-trace ρ/c clips). One thread per row.
// 16B segment I/O (H % ADV_VEC_WIDTH == 0). Bench: tests/bench_puff_advantage.cu
constexpr int ADV_VEC_WIDTH = 16 / (int)sizeof(precision_t);

// Wide load/store of one ADV_VEC_WIDTH segment into float regs.
__device__ __forceinline__ void adv_ld(const precision_t* p, float* o) {
    if constexpr (sizeof(precision_t) == sizeof(float)) {
        float4 v = *(const float4*)p;
        o[0] = v.x; o[1] = v.y; o[2] = v.z; o[3] = v.w;
    } else {
        uint4 u = *(const uint4*)p;
        auto* b = (const __nv_bfloat16*)&u;
        #pragma unroll
        for (int i = 0; i < 8; i++) o[i] = __bfloat162float(b[i]);
    }
}
__device__ __forceinline__ void adv_st(precision_t* p, const float* o) {
    if constexpr (sizeof(precision_t) == sizeof(float)) {
        *(float4*)p = make_float4(o[0], o[1], o[2], o[3]);
    } else {
        __nv_bfloat16 b[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) b[i] = __float2bfloat16(o[i]);
        *(uint4*)p = *(const uint4*)b;
    }
}

// GAE / full truncated IS (V-trace ρ/c): ρ̄ on δ, c̄ on λ product.
// Same R_{t+1},D_{t+1} indexing as classic puffer_advantage.
__global__ void puff_advantage(const precision_t* values,
        const precision_t* rewards, const precision_t* dones,
        const precision_t* importance, precision_t* advantages,
        float gamma, float lambda, float rho_clip, float c_clip,
        int num_steps, int horizon) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_steps) return;
    int off = row * horizon;
    float lastlam = 0.f;
    float next_v = to_float(values[off + horizon - 1]);
    float next_d = to_float(dones[off + horizon - 1]);
    float next_r = to_float(rewards[off + horizon - 1]);

    for (int seg = horizon / ADV_VEC_WIDTH - 1; seg >= 0; seg--) {
        int base = off + seg * ADV_VEC_WIDTH;
        float v[ADV_VEC_WIDTH], r[ADV_VEC_WIDTH], d[ADV_VEC_WIDTH], imp[ADV_VEC_WIDTH];
        float adv[ADV_VEC_WIDTH] = {};
        adv_ld(values + base, v);
        adv_ld(rewards + base, r);
        adv_ld(dones + base, d);
        adv_ld(importance + base, imp);
        // Last index H-1 left 0. First seg starts at width-2.
        int i0 = (seg + 1 == horizon / ADV_VEC_WIDTH) ? ADV_VEC_WIDTH - 2 : ADV_VEC_WIDTH - 1;
        #pragma unroll
        for (int i = i0; i >= 0; i--) {
            float nnt = 1.f - next_d;
            float rho_t = fminf(imp[i], rho_clip);
            float c_t = fminf(imp[i], c_clip);
            float delta = rho_t * (next_r + gamma * next_v * nnt - v[i]);
            lastlam = delta + gamma * lambda * c_t * lastlam * nnt;
            adv[i] = lastlam;
            next_v = v[i]; next_d = d[i]; next_r = r[i];
        }
        adv_st(advantages + base, adv);
    }
}
