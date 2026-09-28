#include "native_policy.h"
#include "device_storage.cuh"
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <openssl/evp.h>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

/* Inference equations, GEMM layout, BF16 boundaries, and categorical sampler
 * follow PufferLib 773f923d80e73bdc255a2ba730c918b28e416aa1:
 * src/algo.cu:105-123, 189-225, 813-829; src/pufferl.cu:587-590, 656-703.
 * The checkpoint is frozen. There is no optimizer or CPU inference path. */
namespace {
thread_local std::string policy_error;
constexpr int OBS=223, ACT=33, DEC=34;
constexpr size_t WORKSPACE=32*1024*1024;
void blas_check(cublasStatus_t x){if(x!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("Native policy cuBLAS status="+std::to_string(int(x)));}
template<class T> __device__ float value(T x){return float(x);}
template<class T> __device__ T reduced(float x){return T(x);}
template<> __device__ float value(__nv_bfloat16 x){return __bfloat162float(x);}
template<> __device__ __nv_bfloat16 reduced(float x){return __float2bfloat16(x);}
__device__ float sigmoid(float x){float z=expf(-fabsf(x));return x>=0?1/(1+z):z/(1+z);}
__device__ float native_lerp(float a,float b,float w){float diff=b-a;return fabsf(w)<.5f?a+w*diff:b-diff*(1-w);}
/* Exact legacy kernels.cu fast_tanh/fast_sigmoid polynomial. Selected only by
 * an explicit legacy registry entry; native5 checkpoints use sigmoid above. */
__device__ float legacy_hidden_sigmoid(float x){
    float v1=fminf(fmaxf(x*.5f,-9.0f),9.0f),v2=v1*v1;
    float p=v2*-2.76076847742355e-16f+2.00018790482477e-13f;
    p=v2*p+-8.60467152213735e-11f;p=v2*p+5.12229709037114e-08f;
    p=v2*p+1.48572235717979e-05f;p=v2*p+6.37261928875436e-04f;p=v2*p+4.89352455891786e-03f;p=v1*p;
    float q=v2*1.19825839466702e-06f+1.18534705686654e-04f;q=v2*q+2.26843463243900e-03f;q=v2*q+4.89352518554385e-03f;
    return fminf(1.0f,fmaxf(0.0f,(p/q+1.0f)*.5f));
}
template<class T> __global__ void cast_weights(T* dst,const float* src,size_t n){
    size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)dst[i]=reduced<T>(src[i]);
}
__global__ void init_rng(curandStatePhilox4_32_10_t* rng,int n,uint64_t seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)curand_init(seed,i,0,rng+i);
}
template<class T> __global__ void prepare(const float* obs,const float* terminals,T* input,T* states,
        int batch,int hidden,int layers,int offset,int stride,int* failures){
    int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=batch)return;int row=offset+b*stride;
    float terminal=terminals[row];
    if(terminal!=0&&terminal!=1)atomicOr(failures+b,1);
    if(terminal!=0)for(int l=0;l<layers;l++)for(int h=0;h<hidden;h++)states[(l*batch+b)*hidden+h]=reduced<T>(0);
    for(int k=0;k<OBS;k++){float x=obs[row*OBS+k];if(!isfinite(x))atomicOr(failures+b,1);input[b*OBS+k]=reduced<T>(x);}
}
template<class T> __global__ void gate(T* out,T* state,const T* combined,const T* input,int h,int batch,bool legacy){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=h*batch)return;
    int b=i/h,j=i%h,base=b*3*h;
    float hidden=value(combined[base+j]),z=sigmoid(value(combined[base+h+j]));
    float projection=value(combined[base+2*h+j]),x=value(input[i]);
    float h_tilde=hidden>=0?hidden+.5f:(legacy?legacy_hidden_sigmoid(hidden):sigmoid(hidden));
    float h_out=native_lerp(value(state[i]),h_tilde,z);state[i]=reduced<T>(h_out);
    float s=sigmoid(projection);out[i]=reduced<T>(s*h_out+(1-s)*x);
}
template<class T> __global__ void sample(const T* logits,float* diagnostic,const uint8_t* masks,
        float* actions,curandStatePhilox4_32_10_t* rng,int* failures,int batch,int offset,int stride,bool deterministic){
    int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=batch)return;int row=offset+b*stride;
    float cache[ACT],max_logit=-INFINITY,sum=0;int legal=0;
    for(int k=0;k<DEC;k++){float x=value(logits[b*DEC+k]);diagnostic[b*DEC+k]=x;if(!isfinite(x))atomicOr(failures+b,2);}
    for(int k=0;k<ACT;k++){
        uint8_t mask=masks[row*ACT+k];if(mask>1)atomicOr(failures+b,4);legal+=mask!=0;
        float x=mask?value(logits[b*DEC+k]):-1e4f;cache[k]=x;
        if(x>max_logit){sum*=__expf(max_logit-x);max_logit=x;}sum+=__expf(x-max_logit);
    }
    if(!legal){atomicOr(failures+b,4);actions[row]=NAN;return;}
    int selected=-1;auto state=rng[b];
    if(deterministic){float best=-INFINITY;for(int k=0;k<ACT;k++)if(masks[row*ACT+k]&&(selected<0||cache[k]>best)){best=cache[k];selected=k;}}
    else{
        float lse=max_logit+__logf(sum),u=curand_uniform(&state),cdf=0;selected=ACT-1;
        for(int k=0;k<ACT;k++){cdf+=expf(cache[k]-lse);if(u<cdf){selected=k;break;}}
        if(selected==ACT-1)for(int k=ACT-1;k>=0;k--)if(masks[row*ACT+k]){selected=k;break;}
    }
    if(selected<0||!masks[row*ACT+selected])atomicOr(failures+b,4);
    actions[row]=failures[b]?NAN:float(selected);rng[b]=state;
}
}

struct RekNativePolicy {
    rek5::DeviceStorage storage;
    RekNativePolicyConfig config{};
    cublasHandle_t blas=nullptr;
    void *weights=nullptr,*input=nullptr,*hidden[2]{},*combined=nullptr,*state=nullptr,*decoder=nullptr,*workspace=nullptr;
    float* diagnostic=nullptr;
    int* failures=nullptr;
    curandStatePhilox4_32_10_t* rng=nullptr;
    std::string digest;
    ~RekNativePolicy(){if(blas)cublasDestroy(blas);}
    size_t params()const{return size_t(OBS)*config.hidden_size+size_t(DEC)*config.hidden_size+size_t(config.num_layers)*3*config.hidden_size*config.hidden_size;}
    template<class T> void allocate(const std::vector<float>& w,cudaStream_t stream){
        int b=config.batch,h=config.hidden_size,l=config.num_layers;
        weights=storage.alloc<T>(w.size());input=storage.alloc<T>(size_t(b)*OBS);
        for(auto& p:hidden)p=storage.alloc<T>(size_t(b)*h);
        combined=storage.alloc<T>(size_t(b)*3*h);state=storage.alloc<T>(size_t(l)*b*h);decoder=storage.alloc<T>(size_t(b)*DEC);
        float* fp=storage.upload(w);cast_weights<<<(w.size()+255)/256,256,0,stream>>>(static_cast<T*>(weights),fp,w.size());
    }
    template<class T> void mm(const T* x,const T* w,T* out,int n,int k,cudaStream_t stream){
        constexpr bool bf=sizeof(T)==2;cudaDataType_t type=bf?CUDA_R_16BF:CUDA_R_32F;
        float alpha=1,beta=0;
        blas_check(cublasGemmEx(blas,CUBLAS_OP_T,CUBLAS_OP_N,n,config.batch,k,&alpha,w,type,k,x,type,k,&beta,out,type,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
    }
    template<class T> void step(const float* obs,const uint8_t* masks,const float* terminal,float* actions,
            int offset,int stride,bool deterministic,cudaStream_t stream){
        int b=config.batch,h=config.hidden_size;T* w=static_cast<T*>(weights);
        prepare<<<(b+127)/128,128,0,stream>>>(obs,terminal,static_cast<T*>(input),static_cast<T*>(state),b,h,config.num_layers,offset,stride,failures);
        mm(static_cast<T*>(input),w,static_cast<T*>(hidden[0]),h,OBS,stream);
        size_t recurrent_offset=size_t(OBS+DEC)*h;
        int current=0;
        for(int l=0;l<config.num_layers;l++){
            mm(static_cast<T*>(hidden[current]),w+recurrent_offset+size_t(l)*3*h*h,static_cast<T*>(combined),3*h,h,stream);
            gate<<<(b*h+255)/256,256,0,stream>>>(static_cast<T*>(hidden[current^1]),static_cast<T*>(state)+size_t(l)*b*h,
                static_cast<T*>(combined),static_cast<T*>(hidden[current]),h,b,config.legacy_fast_hidden!=0);current^=1;
        }
        mm(static_cast<T*>(hidden[current]),w+size_t(OBS)*h,static_cast<T*>(decoder),DEC,h,stream);
        sample<<<(b+127)/128,128,0,stream>>>(static_cast<T*>(decoder),diagnostic,masks,actions,rng,failures,b,offset,stride,deterministic);
        rek5::cuda_check(cudaGetLastError());
    }
};

extern "C" RekNativePolicy* rek_native_policy_create(const RekNativePolicyConfig* cfg,cudaStream_t stream){
    try{
        policy_error.clear();
        if(!cfg||cfg->abi_version!=REK_NATIVE_POLICY_ABI||!cfg->checkpoint_path||cfg->batch<=0||cfg->batch>1048576
            ||cfg->hidden_size<=0||cfg->hidden_size>4096||cfg->num_layers<=0||cfg->num_layers>16
            ||(cfg->legacy_fast_hidden!=0&&cfg->legacy_fast_hidden!=1)
            ||(cfg->precision!=REK_NATIVE_POLICY_BF16&&cfg->precision!=REK_NATIVE_POLICY_FP32))throw std::runtime_error("Invalid native frozen policy configuration");
        auto p=std::make_unique<RekNativePolicy>();p->config=*cfg;
        size_t count=p->params();if(count>size_t(1)<<29)throw std::runtime_error("Policy checkpoint exceeds supported allocation");
        std::unique_ptr<FILE,decltype(&fclose)> fp(fopen(cfg->checkpoint_path,"rb"),fclose);
        if(!fp)throw std::runtime_error("Cannot open native policy checkpoint");
        std::vector<float> weights(count);
        if(fread(weights.data(),sizeof(float),count,fp.get())!=count||fgetc(fp.get())!=EOF||ferror(fp.get()))throw std::runtime_error("Checkpoint byte count differs from explicit architecture");
        for(float v:weights)if(!std::isfinite(v))throw std::runtime_error("Checkpoint contains nonfinite weights");
        unsigned char digest[32];unsigned length=0;
        if(EVP_Digest(weights.data(),count*sizeof(float),digest,&length,EVP_sha256(),nullptr)!=1||length!=32)throw std::runtime_error("Checkpoint SHA256 failed");
        const char* hex="0123456789abcdef";for(auto byte:digest){p->digest+=hex[byte>>4];p->digest+=hex[byte&15];}
        if(cfg->expected_sha256&&p->digest!=cfg->expected_sha256)throw std::runtime_error("Checkpoint SHA256 does not match registry");
        blas_check(cublasCreate(&p->blas));blas_check(cublasSetMathMode(p->blas,CUBLAS_DEFAULT_MATH));
        p->workspace=p->storage.alloc<uint8_t>(WORKSPACE);blas_check(cublasSetStream(p->blas,stream));blas_check(cublasSetWorkspace(p->blas,p->workspace,WORKSPACE));
        if(cfg->precision==REK_NATIVE_POLICY_BF16)p->allocate<__nv_bfloat16>(weights,stream);else p->allocate<float>(weights,stream);
        p->diagnostic=p->storage.alloc<float>(size_t(cfg->batch)*DEC);p->failures=p->storage.alloc<int>(cfg->batch);p->rng=p->storage.alloc<curandStatePhilox4_32_10_t>(cfg->batch);
        init_rng<<<(cfg->batch+127)/128,128,0,stream>>>(p->rng,cfg->batch,cfg->seed);rek5::cuda_check(cudaGetLastError());rek5::cuda_check(cudaStreamSynchronize(stream));
        return p.release();
    }catch(const std::exception& e){policy_error=e.what();return nullptr;}
}
extern "C" int rek_native_policy_step_rows(RekNativePolicy* p,const float* obs,const uint8_t* masks,const float* terminals,
        float* actions,int offset,int stride,int deterministic,cudaStream_t stream){
    try{
        if(!p||!obs||!masks||!terminals||!actions||offset<0||stride<1||int64_t(offset)+int64_t(p->config.batch)*stride>INT32_MAX)
            throw std::runtime_error("Invalid native policy row dispatch");
        blas_check(cublasSetStream(p->blas,stream));blas_check(cublasSetWorkspace(p->blas,p->workspace,WORKSPACE));
        if(p->config.precision==REK_NATIVE_POLICY_BF16)p->step<__nv_bfloat16>(obs,masks,terminals,actions,offset,stride,deterministic!=0,stream);
        else p->step<float>(obs,masks,terminals,actions,offset,stride,deterministic!=0,stream);
        return 0;
    }catch(const std::exception& e){policy_error=e.what();return 1;}
}
extern "C" int rek_native_policy_reset(RekNativePolicy* p,cudaStream_t stream){
    try{
        if(!p)throw std::runtime_error("Null frozen policy");size_t width=p->config.precision==REK_NATIVE_POLICY_BF16?2:4;
        rek5::cuda_check(cudaMemsetAsync(p->state,0,size_t(p->config.batch)*p->config.hidden_size*p->config.num_layers*width,stream));
        // A reset restarts recurrent state and RNG, never erases a detected failure.
        init_rng<<<(p->config.batch+127)/128,128,0,stream>>>(p->rng,p->config.batch,p->config.seed);rek5::cuda_check(cudaGetLastError());return 0;
    }catch(const std::exception& e){policy_error=e.what();return 1;}
}
extern "C" int rek_native_policy_reset_recurrent(RekNativePolicy* p,cudaStream_t stream){
    try{
        if(!p)throw std::runtime_error("Null frozen policy");size_t width=p->config.precision==REK_NATIVE_POLICY_BF16?2:4;
        rek5::cuda_check(cudaMemsetAsync(p->state,0,size_t(p->config.batch)*p->config.hidden_size*p->config.num_layers*width,stream));return 0;
    }catch(const std::exception& e){policy_error=e.what();return 1;}
}
extern "C" int rek_native_policy_check_status(RekNativePolicy* p,cudaStream_t stream){
    try{
        if(!p)throw std::runtime_error("Null frozen policy");std::vector<int> f(p->config.batch);
        rek5::cuda_check(cudaMemcpyAsync(f.data(),p->failures,f.size()*sizeof(int),cudaMemcpyDeviceToHost,stream));rek5::cuda_check(cudaStreamSynchronize(stream));
        for(size_t i=0;i<f.size();i++)if(f[i])throw std::runtime_error("Frozen policy row "+std::to_string(i)+" invalid bits="+std::to_string(f[i])+" (1=input,2=logits,4=mask)");
        return 0;
    }catch(const std::exception& e){policy_error=e.what();return 1;}
}
extern "C" void rek_native_policy_destroy(RekNativePolicy* p){delete p;}
extern "C" const char* rek_native_policy_sha256(const RekNativePolicy* p){return p?p->digest.c_str():nullptr;}
extern "C" const char* rek_native_policy_error(void){return policy_error.c_str();}
extern "C" const float* rek_native_policy_logits(const RekNativePolicy* p){return p?p->diagnostic:nullptr;}
