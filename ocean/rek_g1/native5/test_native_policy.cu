/* Compile against the exact committed native5 src directory. This invokes its
 * actual arch_forward implementation as the oracle, not a rewritten equation.
 * argv[1] is a private native checkpoint; no coefficients are printed/saved. */
#include "pufferl.cu"
#include "native_policy.h"
#include <algorithm>
#include <vector>
#include <stdexcept>

static void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
static void cuda_ok(cudaError_t e){require(e==cudaSuccess,cudaGetErrorString(e));}
int main(int argc,char** argv){
    try{
        require(argc==2,"usage: test_native_policy PRIVATE_CHECKPOINT");
        constexpr int B=4,H=256,L=2,O=223,A=33,D=34;
        cudaStream_t stream;cuda_ok(cudaStreamCreate(&stream));cublas_init_handle();
        RekNativePolicyConfig cfg={REK_NATIVE_POLICY_ABI,argv[1],nullptr,H,L,B,
            USE_BF16?REK_NATIVE_POLICY_BF16:REK_NATIVE_POLICY_FP32,73};
        auto* policy=rek_native_policy_create(&cfg,stream);require(policy,rek_native_policy_error());
        Arch arch=build_arch("rek_native5",O,H,L,A,false,8);Allocator parameters{},activations{};
        Weights weights=weights_create(&arch,&parameters);auto acts=arch_reg_rollout(&arch,weights,&activations,B);
        alloc_create(&parameters);alloc_create(&activations);
        Prec params={.data=(precision_t*)parameters.mem,.shape={parameters.total_elems}};
        Float master={.shape={parameters.total_elems}};cuda_ok(cudaMalloc(&master.data,parameters.total_elems*sizeof(float)));
        puf_load_weights_into(master,params,stream,argv[1]);
        if(!USE_BF16)cuda_ok(cudaMemcpyAsync(params.data,master.data,parameters.total_elems*sizeof(float),cudaMemcpyDeviceToDevice,stream));
        Prec state={.shape={L,B,H}},input={.shape={B,O}};
        cuda_ok(cudaMalloc(&state.data,L*B*H*sizeof(precision_t)));cuda_ok(cudaMemsetAsync(state.data,0,L*B*H*sizeof(precision_t),stream));
        cuda_ok(cudaMalloc(&input.data,B*O*sizeof(precision_t)));
        float *obs,*terminal,*actions,*reference,*packed;
        uint8_t* masks;
        cuda_ok(cudaMalloc(&obs,2*B*O*sizeof(float)));cuda_ok(cudaMalloc(&packed,B*O*sizeof(float)));
        cuda_ok(cudaMalloc(&terminal,2*B*sizeof(float)));cuda_ok(cudaMalloc(&actions,2*B*sizeof(float)));
        cuda_ok(cudaMalloc(&reference,B*D*sizeof(float)));cuda_ok(cudaMalloc(&masks,2*B*A));
        precision_t *reference_masks,*reference_logp,*reference_values;float *reference_actions,*mask_float;int* action_sizes;
        curandStatePhilox4_32_10_t* reference_rng;
        cuda_ok(cudaMalloc(&reference_masks,B*A*sizeof(precision_t)));cuda_ok(cudaMalloc(&mask_float,B*A*sizeof(float)));
        cuda_ok(cudaMalloc(&reference_logp,B*sizeof(precision_t)));cuda_ok(cudaMalloc(&reference_values,B*sizeof(precision_t)));
        cuda_ok(cudaMalloc(&reference_actions,B*sizeof(float)));cuda_ok(cudaMalloc(&action_sizes,sizeof(int)));
        cuda_ok(cudaMemcpy(action_sizes,&A,sizeof(int),cudaMemcpyHostToDevice));
        cuda_ok(cudaMalloc(&reference_rng,B*sizeof(curandStatePhilox4_32_10_t)));rng_init<<<1,128,0,stream>>>(reference_rng,73,B);
        std::vector<float> host_obs(2*B*O),host_packed(B*O),host_terminal(2*B),host_actions(2*B,-99),oracle(B*D),actual(B*D);
        std::vector<uint8_t> host_masks(2*B*A,1);
        std::vector<float> host_reference_masks(B*A),expected_actions(B);
        double max_error=0;size_t compared=0;
        for(int tick=0;tick<12;tick++){
            std::fill(host_terminal.begin(),host_terminal.end(),0);
            if(tick==4)host_terminal[3]=1;
            if(tick==8){require(!rek_native_policy_reset(policy,stream),rek_native_policy_error());cuda_ok(cudaMemsetAsync(state.data,0,L*B*H*sizeof(precision_t),stream));rng_init<<<1,128,0,stream>>>(reference_rng,73,B);}
            for(int b=0;b<B;b++){
                int row=2*b+1;
                for(int k=0;k<O;k++)host_obs[row*O+k]=host_packed[b*O+k]=.2f*sinf(float(k+3*b+7*tick)*.03125f);
                if(host_terminal[row])for(int l=0;l<L;l++)cuda_ok(cudaMemsetAsync(state.data+(l*B+b)*H,0,H*sizeof(precision_t),stream));
                for(int k=0;k<A;k++)host_reference_masks[b*A+k]=host_masks[row*A+k]=(k+b+tick)%4!=0;
            }
            cuda_ok(cudaMemcpyAsync(obs,host_obs.data(),host_obs.size()*sizeof(float),cudaMemcpyHostToDevice,stream));
            cuda_ok(cudaMemcpyAsync(packed,host_packed.data(),host_packed.size()*sizeof(float),cudaMemcpyHostToDevice,stream));
            cuda_ok(cudaMemcpyAsync(terminal,host_terminal.data(),host_terminal.size()*sizeof(float),cudaMemcpyHostToDevice,stream));
            cuda_ok(cudaMemcpyAsync(actions,host_actions.data(),host_actions.size()*sizeof(float),cudaMemcpyHostToDevice,stream));
            cuda_ok(cudaMemcpyAsync(masks,host_masks.data(),host_masks.size(),cudaMemcpyHostToDevice,stream));
            cuda_ok(cudaMemcpyAsync(mask_float,host_reference_masks.data(),B*A*sizeof(float),cudaMemcpyHostToDevice,stream));
            cast<<<grid_size(B*A),BLOCK_SIZE,0,stream>>>(reference_masks,mask_float,B*A);
            cast<<<grid_size(B*O),BLOCK_SIZE,0,stream>>>(input.data,packed,B*O);
            Prec logits=arch_forward(&arch,weights,acts,input,state,stream);
            bool deterministic=tick<6;
            sample_logits<<<1,128,0,stream>>>(logits,Prec{},action_sizes,reference_actions,reference_actions,reference_logp,reference_values,reference_rng,reference_masks,A,deterministic);
            if(USE_BF16){
#ifndef PRECISION_FLOAT
                cast<<<grid_size(B*D),BLOCK_SIZE,0,stream>>>(reference,logits.data,B*D);
#endif
            }else cuda_ok(cudaMemcpyAsync(reference,logits.data,B*D*sizeof(float),cudaMemcpyDeviceToDevice,stream));
            if(tick==10){
                cudaGraph_t graph;cudaGraphExec_t exec;
                cuda_ok(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
                require(!rek_native_policy_step_rows(policy,obs,masks,terminal,actions,1,2,deterministic,stream),rek_native_policy_error());
                cuda_ok(cudaStreamEndCapture(stream,&graph));cuda_ok(cudaGraphInstantiate(&exec,graph,0));
                cuda_ok(cudaGraphLaunch(exec,stream));cuda_ok(cudaStreamSynchronize(stream));cuda_ok(cudaGraphExecDestroy(exec));cuda_ok(cudaGraphDestroy(graph));
            }else require(!rek_native_policy_step_rows(policy,obs,masks,terminal,actions,1,2,deterministic,stream),rek_native_policy_error());
            require(!rek_native_policy_check_status(policy,stream),rek_native_policy_error());
            cuda_ok(cudaMemcpy(oracle.data(),reference,B*D*sizeof(float),cudaMemcpyDeviceToHost));
            cuda_ok(cudaMemcpy(actual.data(),rek_native_policy_logits(policy),B*D*sizeof(float),cudaMemcpyDeviceToHost));
            std::vector<float> selected(2*B);cuda_ok(cudaMemcpy(selected.data(),actions,2*B*sizeof(float),cudaMemcpyDeviceToHost));
            cuda_ok(cudaMemcpy(expected_actions.data(),reference_actions,B*sizeof(float),cudaMemcpyDeviceToHost));
            for(int b=0;b<B;b++){
                require(selected[2*b]==-99,"frozen policy wrote an unselected fighter row");
                int best=-1;for(int k=0;k<A;k++)if(host_masks[(2*b+1)*A+k]&&(best<0||oracle[b*D+k]>oracle[b*D+best]))best=k;
                if(deterministic)require(selected[2*b+1]==best,"native masked argmax differs from pinned trainer logits");
                require(selected[2*b+1]==expected_actions[b],"native sampled action differs from pinned trainer sampler");
                for(int k=0;k<D;k++){max_error=std::max(max_error,double(fabs(oracle[b*D+k]-actual[b*D+k])));compared++;}
            }
        }
        require(max_error==0,"frozen policy logits differ from pinned native trainer");
        cuda_ok(cudaMemsetAsync(masks,0,2*B*A,stream));
        require(!rek_native_policy_step_rows(policy,obs,masks,terminal,actions,1,2,1,stream),rek_native_policy_error());
        require(rek_native_policy_check_status(policy,stream)!=0,"empty legal-action mask was not rejected");
        require(!rek_native_policy_reset(policy,stream),rek_native_policy_error());
        require(rek_native_policy_check_status(policy,stream)!=0,"reset cleared a sticky policy failure");
        printf("native_policy_test precision=%s batch=%d ticks=12 logits_compared=%zu max_abs_error=%.9g masked_argmax=exact stochastic_philox=exact terminal_reset=passed explicit_reset=passed graph=passed row_isolation=passed invalid_mask=passed\n",USE_BF16?"bf16":"fp32",B,compared,max_error);
        rek_native_policy_destroy(policy);cuda_ok(cudaStreamSynchronize(stream));return 0;
    }catch(const std::exception& e){fprintf(stderr,"native_policy_test: %s\n",e.what());return 1;}
}
