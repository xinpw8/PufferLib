#ifndef REK_MJGPU_NATIVE_MODULE_H
#define REK_MJGPU_NATIVE_MODULE_H

#include <cuda.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#include "warp_abi.h"
using RekMjGpuArray = rek_mjgpu::WarpArray;
using RekMjGpuLaunchBounds = rek_mjgpu::LaunchBounds;
#endif

typedef struct RekMjGpuModule RekMjGpuModule;

#ifdef __cplusplus
extern "C" {
#endif

/* Requires a current CUDA context owned by the caller. Loads the exact PTX or
 * cubin bytes without importing Warp, compiling Python, or creating a context.
 * SHA-256 is always computed; an optional expected hash is checked before load.
 * Module lifetime must contain all function use and asynchronous execution. */
RekMjGpuModule* rek_mjgpu_module_load(const char* path, const char* expected_sha256);
int rek_mjgpu_module_function(RekMjGpuModule* module, const char* symbol, CUfunction* function);
const char* rek_mjgpu_module_sha256(const RekMjGpuModule* module);
int rek_mjgpu_module_unload(RekMjGpuModule* module);

/* kernel_parameters points at host-side by-value argument storage, matching
 * the selected generated entry point exactly. Launch is asynchronous and
 * performs no allocations, copies, synchronization, or module lookup. */
int rek_mjgpu_launch(CUfunction function, const unsigned grid[3], const unsigned block[3],
    unsigned shared_bytes, void** kernel_parameters, CUstream stream);
const char* rek_mjgpu_error(void);

#ifdef __cplusplus
}
static_assert(sizeof(CUdeviceptr)==8, "Warp cache ABI requires 64-bit device addresses");
static_assert(sizeof(RekMjGpuArray)==56 && alignof(RekMjGpuArray)==8, "Warp regular array ABI");
static_assert(offsetof(RekMjGpuArray,data)==0 && offsetof(RekMjGpuArray,grad)==8
    && offsetof(RekMjGpuArray,shape)==16 && offsetof(RekMjGpuArray,strides)==32
    && offsetof(RekMjGpuArray,ndim)==48, "Warp regular array field offsets");
static_assert(sizeof(RekMjGpuLaunchBounds)==32 && alignof(RekMjGpuLaunchBounds)==8
    && offsetof(RekMjGpuLaunchBounds,ndim)==16 && offsetof(RekMjGpuLaunchBounds,size)==24,
    "Warp launch bounds ABI");
#endif

#endif
