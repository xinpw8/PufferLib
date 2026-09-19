#include "contact_potential_loader.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void evaluate(rek5_contact_potential::Model model,float* result,int n){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const float distance=.25f+float(i%100)*.02f;
    const float bearing=-3.14159265f+float(i/100)*.062831853f;
    result[i]=rek5_contact_potential::potential(model,distance,bearing);
}
int main(int argc,char** argv){
    try{
        if(argc!=2)throw std::runtime_error("model path required");
        const auto model=rek5_contact_potential::load(argv[1]).model;
        auto checked=[](cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));};
        constexpr int n=10100;float* device=nullptr;checked(cudaMalloc(&device,n*sizeof(float)));
        evaluate<<<(n+127)/128,128>>>(model,device,n);checked(cudaGetLastError());
        std::vector<float> gpu(n);checked(cudaMemcpy(gpu.data(),device,n*sizeof(float),cudaMemcpyDeviceToHost));
        float maximum=0;
        for(int i=0;i<n;i++){
            const float reference=rek5_contact_potential::potential(model,.25f+float(i%100)*.02f,-3.14159265f+float(i/100)*.062831853f);
            const float difference=fabsf(gpu[i]-reference);maximum=fmaxf(maximum,difference);
            if(!std::isfinite(gpu[i])||gpu[i]>0||gpu[i]<-1||difference>2e-6f)
                throw std::runtime_error("host/device mismatch at "+std::to_string(i));
        }
        checked(cudaFree(device));
        std::cout<<"{\"comparisons\":"<<n<<",\"max_abs_error\":"<<maximum<<",\"passed\":true}\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
