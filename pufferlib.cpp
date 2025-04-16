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

// [num_steps, horizon]
void vtrace(float* values, float* rewards, float* dones, float* importance,
        float* trace, float gamma, float rho_clip, float c_clip, int num_steps, const int horizon){
    for (int offset = 0; offset < num_steps*horizon; offset+=horizon) {
        vtrace_row(values + offset, rewards + offset, dones + offset,
            importance + offset, trace + offset, gamma, rho_clip, c_clip, horizon);
    }
}

// [num_steps, horizon]
void puff_advantage(float* values, float* rewards, float* dones, float* importance,
        float* trace, float gamma, float rho_clip, float c_clip, int num_steps, const int horizon){
    for (int offset = 0; offset < num_steps*horizon; offset+=horizon) {
        puff_row(values + offset, rewards + offset, dones + offset,
            importance + offset, trace + offset, gamma, rho_clip, c_clip, horizon);
    }
}


torch::Tensor compute_vtrace(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, float gamma,
        float rho_clip, float c_clip) {
    int num_steps = values.size(0);
    int horizon = values.size(1);
    torch::Tensor trace = vtrace_check(values, rewards, dones, importance, num_steps, horizon);
    vtrace(values.data_ptr<float>(), rewards.data_ptr<float>(),
        dones.data_ptr<float>(), importance.data_ptr<float>(),
        trace.data_ptr<float>(), gamma, rho_clip, c_clip, num_steps, horizon
    );
    return trace;
}

torch::Tensor compute_puff_advantage(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, float gamma,
        float rho_clip, float c_clip) {
    int num_steps = values.size(0);
    int horizon = values.size(1);
    torch::Tensor trace = vtrace_check(values, rewards, dones, importance, num_steps, horizon);
    vtrace(values.data_ptr<float>(), rewards.data_ptr<float>(),
        dones.data_ptr<float>(), importance.data_ptr<float>(),
        trace.data_ptr<float>(), gamma, rho_clip, c_clip, num_steps, horizon
    );
    return trace;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_gae", &compute_gae, "Compute GAE with C");
    m.def("compute_vtrace", &compute_vtrace, "Compute VTrace with C");
    m.def("compute_puff_advantage", &compute_puff_advantage, "Compute PuffAdvantage with C");
}
