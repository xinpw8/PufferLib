#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <string>

namespace rek5 {
inline void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct DeviceStorage {
    std::vector<void*> owned;
    DeviceStorage() = default;
    DeviceStorage(const DeviceStorage&) = delete;
    DeviceStorage& operator=(const DeviceStorage&) = delete;
    template<class T> T* alloc(size_t n) {
        T* p = nullptr;
        cuda_check(cudaMalloc(&p, n*sizeof(T)));
        owned.push_back(p);
        cuda_check(cudaMemset(p, 0, n*sizeof(T)));
        return p;
    }
    template<class T> T* upload(const T* values, size_t n) {
        T* p = alloc<T>(n);
        cuda_check(cudaMemcpy(p, values, n*sizeof(T), cudaMemcpyHostToDevice));
        return p;
    }
    template<class T> T* upload(const std::vector<T>& v) { return upload(v.data(), v.size()); }
    ~DeviceStorage() { for (void* p : owned) cudaFree(p); }
};
}
