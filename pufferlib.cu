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

    m.def("compute_gae", [](torch::Tensor values,
                                torch::Tensor rewards,
                                torch::Tensor dones,
                                torch::Tensor advantages,
                                float gamma,
                                float gae_lambda,
                                int num_steps,
                                int horizon) {
        // Launch the kernel
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
        // Check for CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
    }, "Compute GAE with CUDA");
}
