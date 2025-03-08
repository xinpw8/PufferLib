#include <torch/extension.h>

__global__ void advantage_kernel(
    float* reward_block,    // [num_steps, horizon]
    float* reward_mask,     // [num_steps, horizon]
    float* values_mean,     // [num_steps, horizon]
    float* values_std,      // [num_steps, horizon]
    float* buf,            // [num_steps, horizon]
    float* dones,          // [num_steps]
    float* rewards,        // [num_steps]
    float* advantages,     // [num_steps]
    int* bounds,          // [num_steps]
    int num_steps,
    float r_std,
    int horizon
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_steps) return;

    int k = 0;

    for (int j = 0; j < horizon; j++) {
        int t = i + j;
        if (t >= num_steps - 1 || dones[t]) {
            break;
        }
        k = j + 1;
    }

    float R = 0.0f;
    for (int j = k-2; j > 0; j--) {
        int t = i + j;
        int idx = i * horizon + j;

        float r = rewards[t+1];
        float vstd = values_std[idx];
        if (vstd == 0.0f) {
            buf[idx] = 0.0f;
            continue;
        }

        R += r/(vstd*vstd);
        if (r_std != 0.0f) {
            R -= r/(r_std*r_std);
        }
        reward_block[idx] = R;
        reward_mask[idx] = 1.0f;
        buf[idx] = R;
    }

    advantages[i] = R - values_mean[i*horizon];
    bounds[i] = k;
}

/*
__global__ void advantage_kernel(
    float* reward_block,    // [num_steps, horizon]
    float* reward_mask,     // [num_steps, horizon]
    float* values_mean,     // [num_steps, horizon]
    float* values_std,      // [num_steps, horizon]
    float* buf,            // [num_steps, horizon]
    float* dones,          // [num_steps]
    float* rewards,        // [num_steps]
    float* advantages,     // [num_steps]
    int* bounds,          // [num_steps]
    int num_steps,
    float r_std,
    int horizon
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_steps) return;

    int k = 0;

    for (int j = 0; j < horizon; j++) {
        int t = i + j;
        if (t >= num_steps - 1 || dones[t]) {
            break;
        }
        k = j + 1;
    }

    float R = 0.0f;
    for (int j = k-2; j > 0; j--) {
        int t = i + j;
        int idx = i * horizon + j;

        float r = rewards[t+1];
        float vstd = values_std[idx];
        if (vstd == 0.0f) {
            buf[idx] = 0.0f;
            continue;
        }

        R += r/(vstd*vstd);
        if (r_std != 0.0f) {
            R -= r/(r_std*r_std);
        }
        reward_block[idx] = R;
        reward_mask[idx] = 1.0f;
        buf[idx] = R;
    }

    advantages[i] = R - values_mean[i*horizon];
    bounds[i] = k;
}
*/

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
