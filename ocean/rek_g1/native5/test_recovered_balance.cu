#define REK_G1_CUDA_DEVICE 1
#include "test_recovered_balance_fixture.h"
#include <cuda_runtime.h>

__global__ void balance_test_kernel(rek5_balance_test::Outcome* outcomes, unsigned count) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) outcomes[index] = rek5_balance_test::run_case(index);
}

extern "C" int rek5_balance_test_cuda(rek5_balance_test::Outcome* output, unsigned count) {
    rek5_balance_test::Outcome* device = nullptr;
    const size_t bytes = count * sizeof(*device);
    cudaError_t status = cudaMalloc(&device, bytes);
    if (status != cudaSuccess) return int(status);
    balance_test_kernel<<<(count + 127) / 128, 128>>>(device, count);
    status = cudaGetLastError();
    if (status == cudaSuccess) status = cudaMemcpy(output, device, bytes, cudaMemcpyDeviceToHost);
    const cudaError_t freed = cudaFree(device);
    return int(status == cudaSuccess ? freed : status);
}
