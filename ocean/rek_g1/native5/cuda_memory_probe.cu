#include <cuda_runtime.h>
#include <cstdio>

// Read-only capacity diagnostic apart from a temporary 1 MiB allocation.
// Do not free another process's memory or modify driver/system settings.
int main(){
    auto report=[](const char* stage,cudaError_t e){
        std::printf("{\"stage\":\"%s\",\"cuda_status\":%d,\"message\":\"%s\"}\n",stage,int(e),cudaGetErrorString(e));
        return e==cudaSuccess;
    };
    int devices=0;if(!report("get_device_count",cudaGetDeviceCount(&devices)))return 1;
    std::printf("{\"devices\":%d}\n",devices);
    if(!report("set_device",cudaSetDevice(0)))return 1;
    size_t free_bytes=0,total_bytes=0;if(!report("memory_info",cudaMemGetInfo(&free_bytes,&total_bytes)))return 1;
    std::printf("{\"free_bytes\":%zu,\"total_bytes\":%zu}\n",free_bytes,total_bytes);
    void* p=nullptr;if(!report("allocate_1MiB",cudaMalloc(&p,1024*1024)))return 1;
    bool ok=report("free_owned_1MiB",cudaFree(p));return ok?0:1;
}
