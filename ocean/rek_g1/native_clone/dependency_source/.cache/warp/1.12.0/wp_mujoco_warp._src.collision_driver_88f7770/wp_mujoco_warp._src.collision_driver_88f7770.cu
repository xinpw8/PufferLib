
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_driver.py:0
static CUDA_CALLABLE wp::int32 _binary_search_0(
    wp::array_t<wp::float32> var_values,
    wp::float32 var_value,
    wp::int32 var_lower,
    wp::int32 var_upper)
{
    //---------
    // primal vars
    bool var_0;
    wp::int32 var_1;
    const wp::int32 var_2 = 1;
    wp::int32 var_3;
    wp::float32* var_4;
    bool var_5;
    wp::float32 var_6;
    wp::int32 var_7;
    wp::int32 var_8;
    const wp::int32 var_9 = 1;
    wp::int32 var_10;
    wp::int32 var_11;
    //---------
    // forward
    // def _binary_search(values: wp.array[Any], value: Any, lower: int, upper: int) -> int:       <L 1>
    // while lower < upper:                                                                   <L 2>
    start_while_0:;
    var_0 = (var_lower < var_upper);
    if ((var_0) == false) goto end_while_0;
        // mid = (lower + upper) >> 1                                                         <L 3>
        var_1 = wp::add(var_lower, var_upper);
        var_3 = wp::rshift(var_1, var_2);
        // if values[mid] > value:                                                            <L 4>
        var_4 = wp::address(var_values, var_3);
        var_6 = wp::load(var_4);
        var_5 = (var_6 > var_value);
        if (var_5) {
            // upper = mid                                                                    <L 5>
            var_7 = wp::copy(var_3);
        }
        var_8 = wp::where(var_5, var_7, var_upper);
        if (!var_5) {
            // lower = mid + 1                                                                <L 7>
            var_10 = wp::add(var_3, var_9);
        }
        var_11 = wp::where(var_5, var_lower, var_10);
        wp::assign(var_lower, var_11);
        wp::assign(var_upper, var_8);
    goto start_while_0;
    end_while_0:;
    // return upper                                                                           <L 9>
    return var_upper;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_driver.py:0
static CUDA_CALLABLE void adj__binary_search_0(
    wp::array_t<wp::float32> var_values,
    wp::float32 var_value,
    wp::int32 var_lower,
    wp::int32 var_upper,
    wp::array_t<wp::float32> & adj_values,
    wp::float32 & adj_value,
    wp::int32 & adj_lower,
    wp::int32 & adj_upper,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void _zero_nacon_ncollision_262d46ad_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_nacon_out,
    wp::array_t<wp::int32> var_ncollision_out)
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
        const wp::int32 var_1 = 0;
        const wp::int32 var_2 = 0;
        const wp::int32 var_3 = 0;
        //---------
        // forward
        // def _zero_nacon_ncollision(                                                            <L 81>
        // ncollision_out[0] = 0                                                                  <L 86>
        wp::array_store(var_ncollision_out, var_1, var_0);
        // nacon_out[0] = 0                                                                       <L 87>
        wp::array_store(var_nacon_out, var_3, var_2);
    }
}



extern "C" __global__ void _sap_range_a967ef14_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_ngeom,
    wp::array_t<wp::float32> var_projection_lower_in,
    wp::array_t<wp::float32> var_projection_upper_in,
    wp::array_t<wp::int32> var_sort_index_in,
    wp::array_t<wp::int32> var_range_out)
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
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::float32* var_5;
        wp::float32 var_6;
        wp::float32 var_7;
        wp::slice_t var_8;
        const wp::int32 var_9 = 0;
        wp::array_t<wp::float32> var_10;
        const wp::int32 var_11 = 1;
        wp::int32 var_12;
        wp::int32 var_13;
        const wp::int32 var_14 = 1;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        //---------
        // forward
        // def _sap_range(                                                                        <L 421>
        // worldid, geomid = wp.tid()                                                             <L 431>
        builtin_tid2d(var_0, var_1);
        // idx = sort_index_in[worldid, geomid]                                                   <L 434>
        var_2 = wp::address(var_sort_index_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // upper = projection_upper_in[worldid, idx]                                              <L 436>
        var_5 = wp::address(var_projection_upper_in, var_0, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // limit = _binary_search(projection_lower_in[worldid], upper, geomid + 1, ngeom)         <L 438>
        var_8 = wp::slice_t(var_0, var_0, var_9);
        var_10 = wp::view(var_projection_lower_in, var_8);
        var_12 = wp::add(var_1, var_11);
        var_13 = _binary_search_0(var_10, var_6, var_12, var_ngeom);
        // limit = wp.min(ngeom - 1, limit)                                                       <L 439>
        var_15 = wp::sub(var_ngeom, var_14);
        var_16 = wp::min(var_15, var_13);
        // range_out[worldid, geomid] = limit - geomid                                            <L 442>
        var_17 = wp::sub(var_16, var_1);
        wp::array_store(var_range_out, var_0, var_1, var_17);
    }
}

