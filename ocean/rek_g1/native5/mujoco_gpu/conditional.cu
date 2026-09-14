#include <cuda_runtime.h>

extern "C" __global__ void rek_mjgpu_set_condition(cudaGraphConditionalHandle handle,const int* count){
    if(threadIdx.x==0&&blockIdx.x==0)cudaGraphSetConditional(handle,*count!=0);
}

extern "C" __global__ void rek_mjgpu_mask_any_set_condition(cudaGraphConditionalHandle handle,
        const unsigned char* mask,int count,int* condition){
    int active=0;
    for(int i=threadIdx.x;i<count;i+=blockDim.x)active|=mask[i]!=0;
    const int any=__syncthreads_or(active);
    if(threadIdx.x==0){*condition=any;cudaGraphSetConditional(handle,any);}
}

#ifdef REK_CONDITIONAL_TEST
extern "C" __global__ void rek_mjgpu_test_decrement(int* remaining,int* executed){
    if(threadIdx.x==0&&blockIdx.x==0){--*remaining;++*executed;}
}
#endif
