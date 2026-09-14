#include "native_module.h"

#include <openssl/evp.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct RekMjGpuModule {
    CUmodule module = nullptr;
    CUcontext context = nullptr;
    char sha256[65] = {};
};

namespace {
thread_local char error_text[4096] = {};
int fail(const char* message) { std::snprintf(error_text,sizeof(error_text),"%s",message);return 1; }
int cuda_result(CUresult result,const char* operation) {
    if(result==CUDA_SUCCESS)return 0;
    const char* name=nullptr;const char* message=nullptr;
    cuGetErrorName(result,&name);cuGetErrorString(result,&message);
    std::snprintf(error_text,sizeof(error_text),"%s: %s (%d): %s",operation,name?name:"CUDA error",int(result),message?message:"");
    return 1;
}
bool context_matches(const RekMjGpuModule* module) {
    CUcontext context=nullptr;
    if(cuda_result(cuCtxGetCurrent(&context),"get current CUDA context"))return false;
    if(context!=module->context){fail("Module used outside its loading CUDA context");return false;}
    return true;
}
}

extern "C" RekMjGpuModule* rek_mjgpu_module_load(const char* path,const char* expected) {
    error_text[0]=0;
    try {
        if(!path||!path[0])throw std::runtime_error("Missing CUDA module path");
        auto module=std::make_unique<RekMjGpuModule>();
        if(cuda_result(cuCtxGetCurrent(&module->context),"get current CUDA context"))return nullptr;
        if(!module->context)throw std::runtime_error("Caller must establish a current CUDA context before module load");
        std::ifstream input(path,std::ios::binary|std::ios::ate);
        if(!input)throw std::runtime_error("Cannot open CUDA module file");
        const auto size=input.tellg();
        if(size<=0)throw std::runtime_error("Empty CUDA module file");
        std::vector<char> bytes(static_cast<size_t>(size)+1,0);
        input.seekg(0);input.read(bytes.data(),size);
        if(!input)throw std::runtime_error("CUDA module file read failed");
        unsigned char digest[EVP_MAX_MD_SIZE];unsigned digest_size=0;
        if(!EVP_Digest(bytes.data(),static_cast<size_t>(size),digest,&digest_size,EVP_sha256(),nullptr)||digest_size!=32)
            throw std::runtime_error("CUDA module SHA-256 failed");
        for(unsigned i=0;i<digest_size;i++)std::snprintf(module->sha256+i*2,3,"%02x",digest[i]);
        if(expected&&expected[0]&&std::strcmp(expected,module->sha256))throw std::runtime_error("CUDA module SHA-256 mismatch");
        char jit_error[2048]={};
        CUjit_option options[]={CU_JIT_ERROR_LOG_BUFFER,CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
        void* values[]={jit_error,reinterpret_cast<void*>(sizeof(jit_error))};
        if(cuda_result(cuModuleLoadDataEx(&module->module,bytes.data(),2,options,values),"load cached CUDA module")){
            const size_t used=std::strlen(error_text);
            if(jit_error[0]&&used<sizeof(error_text)-2)std::snprintf(error_text+used,sizeof(error_text)-used,"; %s",jit_error);
            return nullptr;
        }
        return module.release();
    } catch(const std::exception& error){fail(error.what());return nullptr;}
}

extern "C" int rek_mjgpu_module_function(RekMjGpuModule* module,const char* symbol,CUfunction* function) {
    error_text[0]=0;
    if(!module||!symbol||!symbol[0]||!function)return fail("Invalid CUDA function lookup arguments");
    if(!context_matches(module))return 1;
    return cuda_result(cuModuleGetFunction(function,module->module,symbol),"resolve cached CUDA kernel");
}
extern "C" const char* rek_mjgpu_module_sha256(const RekMjGpuModule* module) { return module?module->sha256:nullptr; }
extern "C" int rek_mjgpu_module_unload(RekMjGpuModule* module) {
    error_text[0]=0;
    if(!module)return 0;
    if(!context_matches(module))return 1;
    if(cuda_result(cuModuleUnload(module->module),"unload cached CUDA module"))return 1;
    delete module;return 0;
}
extern "C" int rek_mjgpu_launch(CUfunction function,const unsigned grid[3],const unsigned block[3],
    unsigned shared_bytes,void** parameters,CUstream stream) {
    error_text[0]=0;
    if(!function||!grid||!block||!parameters)return fail("Invalid CUDA kernel launch arguments");
    for(int i=0;i<3;i++)if(!grid[i]||!block[i])return fail("CUDA grid and block dimensions must be positive");
    return cuda_result(cuLaunchKernel(function,grid[0],grid[1],grid[2],block[0],block[1],block[2],shared_bytes,stream,parameters,nullptr),"launch cached CUDA kernel");
}
extern "C" const char* rek_mjgpu_error(void) { return error_text; }
