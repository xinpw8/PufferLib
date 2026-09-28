
#define WP_TILE_BLOCK_DIM 256
#define WP_NO_CRT
#include "builtin.h"

// Map wp.breakpoint() to a device brkpt at the call site so cuda-gdb attributes the stop to the generated .cu line
#if defined(__CUDACC__) && !defined(_MSC_VER)
#define __debugbreak() __brkpt()
#endif

// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)

#define builtin_tid1d() wp::tid(_idx, dim)
#define builtin_tid2d(x, y) wp::tid(x, y, _idx, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, _idx, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, _idx, dim)

#define builtin_block_dim() wp::block_dim()



extern "C" __global__ void observe_construction_7e6f1846_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_nacon,
    wp::int32 var_capacity,
    wp::array_t<wp::int64> var_statistics)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        const wp::int32 var_0 = 0;
        wp::int32* var_1;
        const wp::int32 var_2 = 0;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int64 var_6;
        const wp::int32 var_7 = 1;
        wp::int64 var_8;
        const wp::int32 var_9 = 0;
        wp::int64 var_10;
        const wp::int32 var_11 = 1;
        wp::int64 var_12;
        const wp::int32 var_13 = 2;
        wp::int64* var_14;
        wp::int64 var_15;
        wp::int64 var_16;
        const wp::int32 var_17 = 2;
        //---------
        // forward
        // def observe_construction(nacon: wp.array[int], capacity: int, statistics: wp.array[wp.int64]):       <L 19>
        // live = wp.int64(wp.min(wp.max(nacon[0], 0), capacity))                                 <L 22>
        var_1 = wp::address(var_nacon, var_0);
        var_4 = wp::load(var_1);
        var_3 = wp::max(var_4, var_2);
        var_5 = wp::min(var_3, var_capacity);
        var_6 = wp::int64(var_5);
        // statistics[0] += wp.int64(1)                                                           <L 23>
        var_8 = wp::int64(var_7);
        var_10 = wp::atomic_add(var_statistics, var_9, var_8);
        // statistics[1] += live                                                                  <L 24>
        var_12 = wp::atomic_add(var_statistics, var_11, var_6);
        // statistics[2] = wp.max(statistics[2], live)                                            <L 25>
        var_14 = wp::address(var_statistics, var_13);
        var_16 = wp::load(var_14);
        var_15 = wp::max(var_16, var_6);
        wp::array_store(var_statistics, var_17, var_15);
    }
}

