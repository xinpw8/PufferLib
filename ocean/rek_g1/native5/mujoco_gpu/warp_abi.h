#pragma once

// By-value launch ABI of the installed Warp 1.12.0 generated CUDA kernels.
// No Warp/Python runtime is required to describe or launch these parameters.
// Native smoke tests compare these layouts against the installed C++ headers.
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace rek_mjgpu {
struct WarpArray {
    void* data = nullptr;
    void* grad = nullptr;
    int shape[4] = {};
    int strides[4] = {};
    int ndim = 0;
};
struct LaunchBounds {
    int shape[4] = {};
    int ndim = 0;
    std::size_t size = 0;
};
static_assert(sizeof(void*) == 8, "Native MuJoCo GPU ABI requires 64-bit pointers");
static_assert(sizeof(WarpArray) == 56 && offsetof(WarpArray,ndim) == 48);
static_assert(sizeof(LaunchBounds) == 32 && offsetof(LaunchBounds,size) == 24);
static_assert(std::is_trivially_copyable<WarpArray>::value);

inline LaunchBounds launch_bounds(std::initializer_list<int> dimensions) {
    if(dimensions.size()==0 || dimensions.size()>4)throw std::runtime_error("Invalid GPU launch dimensions");
    LaunchBounds value;value.ndim=int(dimensions.size());value.size=1;
    int k=0;for(int dimension:dimensions){
        if(dimension<0)throw std::runtime_error("Negative GPU launch dimension");
        if(dimension && value.size>std::numeric_limits<std::size_t>::max()/std::size_t(dimension))
            throw std::runtime_error("GPU launch size overflow");
        value.shape[k++]=dimension;value.size*=std::size_t(dimension);
    }
    return value;
}
inline WarpArray contiguous_array(void* device_pointer,std::size_t element_bytes,
                                 std::initializer_list<int> dimensions) {
    const auto bounds=launch_bounds(dimensions);
    if(element_bytes==0)throw std::runtime_error("Zero GPU element width");
    WarpArray value;value.data=device_pointer;value.ndim=bounds.ndim;
    std::size_t stride=element_bytes;
    for(int i=bounds.ndim-1;i>=0;--i){
        if(stride>std::size_t(std::numeric_limits<int>::max()))throw std::runtime_error("Warp byte stride overflow");
        value.shape[i]=bounds.shape[i];value.strides[i]=int(stride);
        stride*=std::size_t(bounds.shape[i]);
    }
    return value;
}
} // namespace rek_mjgpu
