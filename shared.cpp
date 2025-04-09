#include <torch/extension.h>

// TODO: Find a better way to do conditional compilation
#ifndef __CUDA_ARCH__
#define __host__
#define __device__
#endif

// [horizon]
__host__ __device__ void gae_row(float* values, float* rewards, float* dones, float* advantages,
        float gamma, float gae_lambda, int horizon) {
    float lastgaelam = 0;
    for (int t = horizon-2; t >= 0; t--) {
        int t_next = t + 1;
        float nextnonterminal = 1.0 - dones[t_next];
        float delta = rewards[t_next] + gamma*values[t_next]*nextnonterminal - values[t];
        lastgaelam = delta + gamma*gae_lambda*nextnonterminal * lastgaelam;
        advantages[t] = lastgaelam;
    }
}

torch::Tensor gae_check(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, int num_steps, int horizon) {

    // Validate input tensors
    torch::Device device = values.device();
    for (const torch::Tensor& t : {values, rewards, dones}) {
        TORCH_CHECK(t.dim() == 2, "Tensor must be 2D");
        TORCH_CHECK(t.device() == device, "All tensors must be on same device");
        TORCH_CHECK(t.size(0) == num_steps, "First dimension must match num_steps");
        TORCH_CHECK(t.size(1) == horizon, "Second dimension must match horizon");
        TORCH_CHECK(t.dtype() == torch::kFloat32, "All tensors must be float32");
        if (!t.is_contiguous()) {
            t.contiguous();
        }
    }

    torch::Tensor advantages = torch::zeros(
        {num_steps, horizon}, 
        torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(device)
    );
    return advantages;
}


