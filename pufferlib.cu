#include <torch/extension.h>
#include "c_advantage.cu"

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_p3o", [](torch::Tensor reward_block,
                                torch::Tensor reward_mask,
                                torch::Tensor values_mean,
                                torch::Tensor values_std,
                                torch::Tensor buf,
                                torch::Tensor dones,
                                torch::Tensor rewards,
                                torch::Tensor advantages,
                                torch::Tensor bounds,
                                int num_steps,
                                float vstd_max,
                                float puf,
                                int horizon) {
        // Launch the kernel
        int threads_per_block = 256;
        int blocks = (num_steps + threads_per_block - 1) / threads_per_block;

        p3o_kernel<<<blocks, threads_per_block>>>(
            reward_block.data_ptr<float>(),
            reward_mask.data_ptr<float>(),
            values_mean.data_ptr<float>(),
            values_std.data_ptr<float>(),
            buf.data_ptr<float>(),
            dones.data_ptr<float>(),
            rewards.data_ptr<float>(),
            advantages.data_ptr<float>(),
            bounds.data_ptr<int>(),
            num_steps,
            vstd_max, 
            puf,
            horizon
        );

        // Check for CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
    }, "Compute p3o advantages with CUDA");

    m.def("compute_gae", [](
            torch::Tensor values,
            torch::Tensor rewards,
            torch::Tensor dones,
            float gamma,
            float gae_lambda) {
        torch::Device device = values.device();
        int num_steps = values.size(0);
        int horizon = values.size(1);
        
        // Validate input tensors
        for (const torch::Tensor& t : {values, rewards, dones}) {
            TORCH_CHECK(t.dim() == 2, "Tensor must be 2D");
            TORCH_CHECK(t.device() == device, "All tensors must be on same device");
            TORCH_CHECK(t.size(0) == num_steps, "First dimension must match num_steps");
            TORCH_CHECK(t.size(1) == horizon, "Second dimension must match horizon");
            TORCH_CHECK(t.is_cuda(), "All tensors must be on GPU");
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

        int threads_per_block = 256;
        int blocks = (num_steps + threads_per_block - 1) / threads_per_block;

        gae_kernel<<<blocks, threads_per_block>>>(
            values.data_ptr<float>(),
            rewards.data_ptr<float>(),
            dones.data_ptr<float>(),
            advantages.data_ptr<float>(),
            gamma,
            gae_lambda,
            num_steps,
            horizon
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }

        return advantages;
    }, "Compute GAE with CUDA");
}
