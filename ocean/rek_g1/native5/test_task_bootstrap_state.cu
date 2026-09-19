// Include the patched native learner to exercise its actual network and
// boundary helper. No rollout, optimizer update or checkpoint write occurs.
#include PUFFER_TEST_TRAINER_SOURCE
#include <vector>

static void test_cuda(cudaError_t result) {
    if (result != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(result)); exit(2); }
}
template<class T> static std::vector<unsigned char> snapshot(T* ptr, size_t bytes) {
    std::vector<unsigned char> out(bytes);
    test_cuda(cudaMemcpy(out.data(), ptr, bytes, cudaMemcpyDeviceToHost));
    return out;
}
int main(int argc, char** argv) {
    Ini ini = {0};
    puf_ini_load_env(&ini, PUFFER_ENV_NAME, argc-1, argv+1);
    puf_ini_put(&ini, "vec.total_agents", "8");
    puf_ini_put(&ini, "train.horizon", "8");
    puf_ini_put(&ini, "train.minibatch_size", "64");
    puf_ini_put(&ini, "base.reset_every_horizon", "0");
    puf_ini_put(&ini, "base.async", "0");
    TrainContext context = {.world_size=1, .artifact_owner=1};
    PuffeRL* p = create_pufferl(&ini, &context);
    cudaStream_t stream = p->streams[0];
    const int n = p->hypers.total_agents;
    Prec& actor = p->policies[0].buffer_states[0];
    const size_t state_bytes = numel(actor.shape) * sizeof(precision_t);
    std::vector<precision_t> initial(numel(actor.shape));
    for(size_t i=0;i<initial.size();i++) initial[i]=from_float(float(int(i%13)-6)/32);
    test_cuda(cudaMemcpy(actor.data, initial.data(), state_bytes, cudaMemcpyHostToDevice));
    std::vector<float> rewards(n), terminals(n);
    for(int i=0;i<n;i++) { rewards[i]=float(i+1);terminals[i]=float(i%2); }
    test_cuda(cudaMemcpy(p->env.rewards.data,rewards.data(),n*sizeof(float),cudaMemcpyHostToDevice));
    test_cuda(cudaMemcpy(p->env.terminals.data,terminals.data(),n*sizeof(float),cudaMemcpyHostToDevice));
    const auto state_before=snapshot(actor.data,state_bytes);
    const auto rng_before=snapshot(p->rng_states[0],n*sizeof(curandStatePhilox4_32_10_t));
    const auto actions_before=snapshot(p->env.actions.data,numel(p->env.actions.shape)*sizeof(float));
    const auto obs_before=snapshot(p->env.obs.data,numel(p->env.obs.shape)*sizeof(obs_t));
    const auto mask_before=snapshot(p->env.action_mask.data,numel(p->env.action_mask.shape));
    const auto weights_before=snapshot(p->policies[0].param.data,numel(p->policies[0].param.shape)*sizeof(precision_t));
    pufferl_boundary_bootstrap(p,stream);
    test_cuda(cudaGetLastError());test_cuda(cudaStreamSynchronize(stream));
    const auto first=snapshot(p->boundary_values.data,n*sizeof(precision_t));
    bool state_ok=state_before==snapshot(actor.data,state_bytes);
    bool rng_ok=rng_before==snapshot(p->rng_states[0],n*sizeof(curandStatePhilox4_32_10_t));
    bool actions_ok=actions_before==snapshot(p->env.actions.data,numel(p->env.actions.shape)*sizeof(float));
    bool env_ok=obs_before==snapshot(p->env.obs.data,numel(p->env.obs.shape)*sizeof(obs_t)) &&
        mask_before==snapshot(p->env.action_mask.data,numel(p->env.action_mask.shape));
    bool weights_ok=weights_before==snapshot(p->policies[0].param.data,numel(p->policies[0].param.shape)*sizeof(precision_t));
    bool scratch_changed=state_before!=snapshot(p->boundary_state.data,state_bytes);
    std::vector<precision_t> actual_rewards(n),actual_terminals(n),actual_values(n);
    test_cuda(cudaMemcpy(actual_rewards.data(),p->boundary_rewards.data,n*sizeof(precision_t),cudaMemcpyDeviceToHost));
    test_cuda(cudaMemcpy(actual_terminals.data(),p->boundary_terminals.data,n*sizeof(precision_t),cudaMemcpyDeviceToHost));
    test_cuda(cudaMemcpy(actual_values.data(),p->boundary_values.data,n*sizeof(precision_t),cudaMemcpyDeviceToHost));
    bool tuple_ok=true;
    for(int i=0;i<n;i++) tuple_ok &= to_float(actual_rewards[i])==rewards[i] &&
        to_float(actual_terminals[i])==terminals[i] && std::isfinite(to_float(actual_values[i]));
    pufferl_boundary_bootstrap(p,stream);
    test_cuda(cudaStreamSynchronize(stream));
    bool repeat_ok=first==snapshot(p->boundary_values.data,n*sizeof(precision_t));
    cudaGraph_t graph;cudaGraphExec_t executable;
    test_cuda(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    pufferl_boundary_bootstrap(p,stream);
    test_cuda(cudaStreamEndCapture(stream,&graph));
    test_cuda(cudaGraphInstantiate(&executable,graph,0));
    bool graph_ok=true;
    for(int i=0;i<4;i++) {
        test_cuda(cudaGraphLaunch(executable,stream));test_cuda(cudaStreamSynchronize(stream));
        graph_ok &= first==snapshot(p->boundary_values.data,n*sizeof(precision_t)) &&
            state_before==snapshot(actor.data,state_bytes) &&
            rng_before==snapshot(p->rng_states[0],n*sizeof(curandStatePhilox4_32_10_t)) &&
            actions_before==snapshot(p->env.actions.data,numel(p->env.actions.shape)*sizeof(float));
    }
    test_cuda(cudaGraphExecDestroy(executable));test_cuda(cudaGraphDestroy(graph));
    bool passed=state_ok&&rng_ok&&actions_ok&&env_ok&&weights_ok&&scratch_changed&&tuple_ok&&repeat_ok&&graph_ok;
    printf("task_bootstrap_state_test={\"actual_native_network\":true,\"actor_state_unchanged\":%s,\"rng_unchanged\":%s,\"actions_unchanged\":%s,\"observations_masks_unchanged\":%s,\"parameters_unchanged\":%s,\"scratch_state_advanced\":%s,\"boundary_tuple_correct\":%s,\"repeat_value_identical\":%s,\"graph_replays\":4,\"graph_checks_pass\":%s,\"training_updates\":0,\"passed\":%s}\n",
        state_ok?"true":"false",rng_ok?"true":"false",actions_ok?"true":"false",env_ok?"true":"false",
        weights_ok?"true":"false",scratch_changed?"true":"false",tuple_ok?"true":"false",repeat_ok?"true":"false",graph_ok?"true":"false",passed?"true":"false");
    close_pufferl(p);puf_ini_free(&ini);
    return passed?0:1;
}
