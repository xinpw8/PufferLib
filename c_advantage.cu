extern "C" {
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
        int horizon,
        float vstd_min,
        float vstd_max
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= num_steps) return;

        // Initialize variables
        int k = 0;
        float adv_sum = 0.0f;
        float delta = vstd_min - vstd_max;

        // First pass: compute reward_block, reward_mask, buf, and adv_sum
        for (int j = 0; j < horizon; j++) {
            int t = i + j;

            // Early termination conditions
            if (t >= num_steps - 1 || dones[t]) {
                break;
            }

            k = j + 1;  // Update valid column count

            // Compute indices (2D to 1D)
            int idx = i * horizon + j;

            // Fill arrays
            reward_block[idx] = rewards[t + 1];
            reward_mask[idx] = 1.0f;

            float vstd = values_std[idx];
            buf[idx] = vstd;

            // Compute adv_scale
            float adv_scale = (delta == 0) ? 1.0f : (vstd_max - vstd) / delta;
            adv_scale = max(min(adv_scale, 1.0f), 0.05f);
            buf[idx] = adv_scale;
            adv_sum += adv_scale;
        }

        bounds[i] = k;

        // Special case: delta == 0
        if (delta == 0) {
            float adv = 0.0f;
            for (int j = 0; j < k; j++) {
                int idx = i * horizon + j;
                adv += (reward_block[idx] - values_mean[idx]);
            }
            advantages[i] = adv;
            return;
        }

        // Second pass: compute advantages
        float adv = 0.0f;
        for (int j = 0; j < k; j++) {
            int idx = i * horizon + j;
            adv += (buf[idx] / adv_sum) * (reward_block[idx] - values_mean[idx]);
        }
        advantages[i] = adv;
    }
}
