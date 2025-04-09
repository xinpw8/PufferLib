#include "shared.cpp"

// [num_steps, horizon]
void gae(float* values, float* rewards, float* dones, float* advantages,
        float gamma, float gae_lambda, int num_steps, int horizon){
    for (int offset = 0; offset < num_steps*horizon; offset+=horizon) {
        gae_row(values + offset, rewards + offset, dones + offset,
            advantages + offset, gamma, gae_lambda, horizon);
    }
}

torch::Tensor compute_gae(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, float gamma, float gae_lambda) {
    int num_steps = values.size(0);
    int horizon = values.size(1);
    torch::Tensor advantages = gae_check(values, rewards, dones, num_steps, horizon);
    gae(values.data_ptr<float>(), rewards.data_ptr<float>(),
        dones.data_ptr<float>(), advantages.data_ptr<float>(),
        gamma, gae_lambda, num_steps, horizon
    );
    return advantages;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_gae", &compute_gae, "Compute GAE with C");
}
