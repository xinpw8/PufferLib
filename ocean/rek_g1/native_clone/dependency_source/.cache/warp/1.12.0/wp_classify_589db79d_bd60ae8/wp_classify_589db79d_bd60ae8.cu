
#define WP_TILE_BLOCK_DIM 32
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


// /home/spark-advantage/rek-training/physics-cholesky-split-v2/cholesky_zero_candidate.py:43
static CUDA_CALLABLE wp::int32 nonzero_bits_0(
    wp::float32 value)
{

    unsigned int bits;
#if defined(__CUDA_ARCH__)
    bits = __float_as_uint(value);
#else
    memcpy(&bits, &value, sizeof(bits));
#endif
    return (bits & 0x7fffffffu) != 0u ? 1 : 0;
}


// /home/spark-advantage/rek-training/physics-cholesky-split-v2/cholesky_zero_candidate.py:56
static CUDA_CALLABLE bool certified_zero_rectangle_0(
    wp::array_t<wp::float32> var_A)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 32;
    const wp::int32 var_1 = 32;
    wp::tuple_t<wp::int32, wp::int32> var_2;
    const wp::int32 var_3 = 48;
    const wp::int32 var_4 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_5;
    const wp::str var_6 = "shared";
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<32,32>, wp::tile_stride_t<32,1>>, true> var_7 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<32,32>,wp::tile_stride_t<32,1>,false>();
    const wp::int32 var_8 = 48;
    const wp::int32 var_9 = 0;
    wp::tile_register_t<wp::int32,wp::tile_layout_register_t<wp::tile_shape_t<32,32>>> var_10 = wp::tile_register_t<wp::int32,wp::tile_layout_register_t<wp::tile_shape_t<32,32>>>{};
    wp::tile_shared_t<wp::int32,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_11 = wp::tile_alloc_empty<wp::int32,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
    const wp::int32 var_12 = 0;
    wp::int32 var_13;
    const wp::int32 var_14 = 0;
    bool var_15;
    //---------
    // forward
    // def certified_zero_rectangle(A: wp.array2d[float]) -> bool:                            <L 57>
    // rectangle = wp.tile_load(A, shape=(32, 32), offset=(48, 0), storage="shared")          <L 58>
    var_2 = wp::tuple(var_0, var_1);
    var_5 = wp::tuple(var_3, var_4);
    var_7 = wp::tile_load<wp::float32, true, 32, 32>(var_A, var_8, var_9);
    // flags = wp.tile_map(nonzero_bits, rectangle)                                           <L 59>
    var_10 = wp::tile_unary_map(nonzero_bits_0, var_7);
    // return wp.tile_sum(flags)[0] == 0                                                      <L 60>
    var_11 = wp::tile_sum(var_10);
    var_13 = wp::tile_extract(var_11, var_12);
    var_15 = (var_13 == var_14);
    return var_15;
}


// /home/spark-advantage/rek-training/physics-cholesky-split-v2/cholesky_zero_candidate.py:43
static CUDA_CALLABLE void adj_nonzero_bits_0(
    wp::float32 value,
    wp::float32 & adj_value,
    wp::int32 & adj_ret)
{
}


// /home/spark-advantage/rek-training/physics-cholesky-split-v2/cholesky_zero_candidate.py:56
static CUDA_CALLABLE void adj_certified_zero_rectangle_0(
    wp::array_t<wp::float32> var_A,
    wp::array_t<wp::float32> & adj_A,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void classify_c8df25c0_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_h,
    wp::array_t<wp::int32> var_flags)
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
        wp::slice_t var_1;
        const wp::int32 var_2 = 0;
        wp::array_t<wp::float32> var_3;
        bool var_4;
        wp::int32 var_5;
        //---------
        // forward
        // def classify(h: wp.array3d[float], flags: wp.array[int]):                              <L 17>
        // world = wp.tid()                                                                       <L 18>
        var_0 = builtin_tid1d();
        // flags[world] = int(certified_zero_rectangle(h[world]))                                 <L 19>
        var_1 = wp::slice_t(var_0, var_0, var_2);
        var_3 = wp::view(var_h, var_1);
        var_4 = certified_zero_rectangle_0(var_3);
        var_5 = wp::int(var_4);
        wp::array_store(var_flags, var_0, var_5);
    }
}

