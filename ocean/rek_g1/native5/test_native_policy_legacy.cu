/* Native FP32 oracle from the original checkpoint-producing models.cu.
 * Include path must identify that immutable legacy source snapshot. */
#include "models.cu"
#include "native_policy.h"
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <memory>
static void require(bool x,const char* m){if(!x)throw std::runtime_error(m);}
static void cuda_ok(cudaError_t x){require(x==cudaSuccess,cudaGetErrorString(x));}
int main(int argc,char** argv){try{
    static_assert(!USE_BF16,"legacy recorded checkpoints used PRECISION_FLOAT");
    require(argc==2,"usage: test_native_policy_legacy PRIVATE_CHECKPOINT");
    constexpr int B=4,H=256,L=2,O=223,A=33,D=34;
    cudaStream_t stream;cuda_ok(cudaStreamCreate(&stream));
    RekNativePolicyConfig cfg={REK_NATIVE_POLICY_ABI,argv[1],nullptr,H,L,B,REK_NATIVE_POLICY_FP32,73,1};
    auto* p=rek_native_policy_create(&cfg,stream);require(p,rek_native_policy_error());
    Policy original{};
    original.encoder.forward=encoder_forward;original.encoder.reg_params=encoder_reg_params;
    original.encoder.reg_rollout=encoder_reg_rollout;original.encoder.create_weights=encoder_create_weights;
    original.encoder.in_dim=O;original.encoder.out_dim=H;original.encoder.activation_size=sizeof(EncoderActivations);
    original.decoder.forward=decoder_forward;original.decoder.reg_params=decoder_reg_params;
    original.decoder.reg_rollout=decoder_reg_rollout;original.decoder.create_weights=decoder_create_weights;
    original.decoder.hidden_dim=H;original.decoder.output_dim=A;original.decoder.continuous=false;
    original.network.forward=mingru_forward;original.network.reg_params=mingru_reg_params;
    original.network.reg_rollout=mingru_reg_rollout;original.network.create_weights=mingru_create_weights;
    original.network.hidden=H;original.network.num_layers=L;original.network.horizon=64;
    Allocator parameters{},activations{};PolicyWeights w=policy_weights_create(&original,&parameters);
    auto act=policy_reg_rollout(&original,w,&activations,B);cuda_ok(alloc_create(&parameters));cuda_ok(alloc_create(&activations));
    std::vector<float> weights(parameters.total_elems);
    std::unique_ptr<FILE,decltype(&fclose)> fp(fopen(argv[1],"rb"),fclose);require(bool(fp),"cannot read legacy checkpoint");
    require(fread(weights.data(),sizeof(float),weights.size(),fp.get())==weights.size()&&fgetc(fp.get())==EOF,"legacy checkpoint size mismatch");
    cuda_ok(cudaMemcpy(parameters.mem,weights.data(),weights.size()*sizeof(float),cudaMemcpyHostToDevice));
    PrecisionTensor state={.shape={L,B,H}},input={.shape={B,O}};
    cuda_ok(cudaMalloc(&state.data,L*B*H*sizeof(float)));cuda_ok(cudaMemsetAsync(state.data,0,L*B*H*sizeof(float),stream));
    cuda_ok(cudaMalloc(&input.data,B*O*sizeof(float)));float *terminal,*actions;uint8_t* masks;
    cuda_ok(cudaMalloc(&terminal,B*sizeof(float)));cuda_ok(cudaMemsetAsync(terminal,0,B*sizeof(float),stream));
    cuda_ok(cudaMalloc(&actions,B*sizeof(float)));cuda_ok(cudaMalloc(&masks,B*A));cuda_ok(cudaMemsetAsync(masks,1,B*A,stream));
    std::vector<float> host_input(B*O),expected(B*D),actual(B*D),host_actions(B);
    double max_error=0;size_t count=0;
    for(int tick=0;tick<70;tick++){
        if(tick==64){require(!rek_native_policy_reset_recurrent(p,stream),rek_native_policy_error());cuda_ok(cudaMemsetAsync(state.data,0,L*B*H*sizeof(float),stream));}
        for(int k=0;k<B*O;k++)host_input[k]=.25f*sinf(float(k+5*tick)*.02f);
        cuda_ok(cudaMemcpyAsync(input.data,host_input.data(),B*O*sizeof(float),cudaMemcpyHostToDevice,stream));
        auto logits=policy_forward(&original,w,act,input,state,stream);
        require(!rek_native_policy_step_rows(p,input.data,masks,terminal,actions,0,1,1,stream),rek_native_policy_error());
        require(!rek_native_policy_check_status(p,stream),rek_native_policy_error());
        cuda_ok(cudaMemcpy(expected.data(),logits.data,B*D*sizeof(float),cudaMemcpyDeviceToHost));
        cuda_ok(cudaMemcpy(actual.data(),rek_native_policy_logits(p),B*D*sizeof(float),cudaMemcpyDeviceToHost));
        cuda_ok(cudaMemcpy(host_actions.data(),actions,B*sizeof(float),cudaMemcpyDeviceToHost));
        for(int b=0;b<B;b++){
            int best=0;for(int k=1;k<A;k++)if(expected[b*D+k]>expected[b*D+best])best=k;
            require(host_actions[b]==best,"legacy greedy action mismatch");
            for(int k=0;k<D;k++){max_error=std::max(max_error,double(fabs(expected[b*D+k]-actual[b*D+k])));count++;}
        }
    }
    require(max_error==0,"legacy FP32 logits differ from original checkpoint-producing model code");
    printf("native_policy_legacy_test fp32 batch=4 ticks=70 logits_compared=%zu max_abs_error=%.9g masked_argmax=exact reset64=passed legacy_fast_hidden=passed\n",count,max_error);
    rek_native_policy_destroy(p);return 0;
}catch(const std::exception& e){fprintf(stderr,"native_policy_legacy_test: %s\n",e.what());return 1;}}
