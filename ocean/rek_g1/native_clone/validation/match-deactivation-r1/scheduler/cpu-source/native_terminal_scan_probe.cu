// Standalone diagnostic using the actual learner kernels, without an environment.
// Build with -I/path/to/src. Define REK_LEGACY_SCAN for the pre-fix source tree.
#include "models.cu"

extern "C" int rek_scan_precision_bytes() { return sizeof(precision_t); }
extern "C" int rek_scan_supports_terminals() {
#ifdef REK_LEGACY_SCAN
    return 0;
#else
    return 1;
#endif
}

extern "C" int rek_scan_run(int B, int T, int H,
        void* combined, void* state, void* input, const void* terminals,
        void* out, void* next_state, float* a_star, float* s_vals,
        float* log_values, void* grad_combined, void* grad_state,
        void* grad_input, const void* grad_out, const void* grad_next_state,
        void* stream_ptr) {
    // The production checkpoint scan uses horizons divisible by four.
    if (B <= 0 || T <= 0 || T % CHECKPOINT_INTERVAL != 0 || H <= 0) return -1;
    PrefixScan scan{};
    scan.B = B;
    scan.T = T;
    scan.H = H;
    scan.combined_ptr = static_cast<precision_t*>(combined);
    scan.state_ptr = static_cast<precision_t*>(state);
    scan.input_ptr = static_cast<precision_t*>(input);
#ifndef REK_LEGACY_SCAN
    scan.terminal_ptr = static_cast<const precision_t*>(terminals);
#endif
    scan.out.data = static_cast<precision_t*>(out);
    scan.next_state.data = static_cast<precision_t*>(next_state);
    scan.a_star.data = a_star;
    scan.s_vals.data = s_vals;
    scan.log_values_buf.data = log_values;
    scan.grad_combined.data = static_cast<precision_t*>(grad_combined);
    scan.grad_state.data = static_cast<precision_t*>(grad_state);
    scan.grad_input.data = static_cast<precision_t*>(grad_input);
    auto stream = static_cast<cudaStream_t>(stream_ptr);
    mingru_scan_forward<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(scan);
    mingru_scan_backward<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(
        scan, static_cast<const precision_t*>(grad_out),
        static_cast<const precision_t*>(grad_next_state));
    return static_cast<int>(cudaPeekAtLastError());
}
