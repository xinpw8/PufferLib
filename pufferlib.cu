#include <torch/extension.h>
#include "c_advantage.cu"

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("advantage_kernel", [](torch::Tensor reward_block,
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
                                int horizon) {
        // Launch the kernel
        int threads_per_block = 256;
        int blocks = (num_steps + threads_per_block - 1) / threads_per_block;

        advantage_kernel<<<blocks, threads_per_block>>>(
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
            horizon
        );

        // Check for CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
    }, "Compute advantages with CUDA");
}
