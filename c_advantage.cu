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
    for (int j = 0; j < horizon-1; j++) {
        int t = i + j;
        if (t >= num_steps - 1) {
            break;
        }
        if (dones[t+1]) {
            k++;
            break;
        }
        k++;
    }

    float gamma_max = 0.0f;
    float n = 0.0f;
    for (int j = k; j >= 0; j--) {
        int idx = i * horizon + j;
        n++;

        float vstd = values_std[idx];
        if (vstd == 0.0f) {
            buf[idx] = 0.0f;
            continue;
        }

        float gamma = 1.0f / (vstd*vstd) / n;
        /*
        if (r_std != 0.0f) {
            gamma -= 1.0f/(r_std*r_std);
        }
        */

        if (gamma < 0.0f) {
            gamma = 0.0f;
        }

        if (gamma > gamma_max) {
            gamma_max = gamma;
        }
        buf[idx] = gamma;
        reward_mask[idx] = 1.0f;
    }

    float bootstrap = 0.0f;
    //if (k == horizon-1) {
    //    bootstrap = buf[i*horizon + horizon - 1]*values_mean[i*horizon + horizon - 1];
    //}

    float R = 0.0f;
    for (int j = k-1; j >= 0; j--) {
        int t = i + j;
        int idx = i * horizon + j;
        float r = rewards[t+1];

        float gamma = buf[idx];
        if (gamma_max > 0) {
            gamma /= gamma_max;
        }

        R += r;
        reward_block[idx] = gamma;
        buf[idx] = gamma * R;
    }

    advantages[i] = R - values_mean[i*horizon];
    bounds[i] = k;
}
