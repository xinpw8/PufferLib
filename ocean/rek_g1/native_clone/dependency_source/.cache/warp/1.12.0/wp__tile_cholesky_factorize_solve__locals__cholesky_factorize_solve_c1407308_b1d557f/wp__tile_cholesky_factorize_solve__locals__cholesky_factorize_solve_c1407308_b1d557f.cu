
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

extern "C" {
void potrf_35_35_1_121_32_1_1_5_x_x_1(wp::float32*, int*);
void potrs_35_35_1_121_32_1_1_5_x_x_1(wp::float32*, wp::float32*);
}


extern "C" __global__ void _tile_cholesky_factorize_solve__locals__cholesky_factorize_solve_f740699c_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_M,
    wp::array_t<wp::float32> var_y,
    wp::array_t<wp::int32> var_adr,
    wp::array_t<wp::float32> var_x,
    wp::array_t<wp::float32> var_L)
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
        const wp::int32 var_2 = 35;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::slice_t var_6;
        const wp::int32 var_7 = 0;
        wp::array_t<wp::float32> var_8;
        wp::tuple_t<wp::int32, wp::int32> var_9;
        wp::tuple_t<wp::int32, wp::int32> var_10;
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<35,35>, wp::tile_stride_t<35,1>>, true> var_11 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<35,35>,wp::tile_stride_t<35,1>,false>();
        wp::slice_t var_12;
        const wp::int32 var_13 = 0;
        wp::array_t<wp::float32> var_14;
        wp::tuple_t<wp::int32> var_15;
        wp::tuple_t<wp::int32> var_16;
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<35>, wp::tile_stride_t<1>>, true> var_17 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<35>,wp::tile_stride_t<1>,false>();
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<35,35>, wp::tile_stride_t<35,1>>, true> var_18 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<35,35>,wp::tile_stride_t<35,1>,false>();
        wp::slice_t var_19;
        const wp::int32 var_20 = 0;
        wp::array_t<wp::float32> var_21;
        wp::tuple_t<wp::int32, wp::int32> var_22;
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<35>, wp::tile_stride_t<1>>, true> var_23 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<35>,wp::tile_stride_t<1>,false>();
        wp::slice_t var_24;
        const wp::int32 var_25 = 0;
        wp::array_t<wp::float32> var_26;
        wp::tuple_t<wp::int32> var_27;
        //---------
        // forward
        // def cholesky_factorize_solve(                                                          <L 2865>
        // worldid, nodeid = wp.tid()                                                             <L 2874>
        builtin_tid2d(var_0, var_1);
        // TILE_SIZE = wp.static(tile.size)                                                       <L 2875>
        // dofid = adr[nodeid]                                                                    <L 2877>
        var_3 = wp::address(var_adr, var_1);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // M_tile = wp.tile_load(M[worldid], shape=(TILE_SIZE, TILE_SIZE), offset=(dofid, dofid))       <L 2878>
        var_6 = wp::slice_t(var_0, var_0, var_7);
        var_8 = wp::view(var_M, var_6);
        var_9 = wp::tuple(var_2, var_2);
        var_10 = wp::tuple(var_4, var_4);
        var_11 = wp::tile_load<wp::float32, true, 35, 35>(var_8, var_4, var_4);
        // y_slice = wp.tile_load(y[worldid], shape=(TILE_SIZE,), offset=(dofid,))                <L 2879>
        var_12 = wp::slice_t(var_0, var_0, var_13);
        var_14 = wp::view(var_y, var_12);
        var_15 = wp::tuple(var_2);
        var_16 = wp::tuple(var_4);
        var_17 = wp::tile_load<wp::float32, true, 35>(var_14, var_4);
        // L_tile = wp.tile_cholesky(M_tile)                                                      <L 2881>
        var_18 = tile_cholesky(potrf_35_35_1_121_32_1_1_5_x_x_1, var_11, var_18);
        // wp.tile_store(L[worldid], L_tile, offset=(dofid, dofid))                               <L 2882>
        var_19 = wp::slice_t(var_0, var_0, var_20);
        var_21 = wp::view(var_L, var_19);
        var_22 = wp::tuple(var_4, var_4);
        wp::tile_store<wp::float32, true>(var_21, var_4, var_4, var_18);
        // x_slice = wp.tile_cholesky_solve(L_tile, y_slice)                                      <L 2883>
        var_23 = tile_cholesky_solve(potrs_35_35_1_121_32_1_1_5_x_x_1, var_18, var_17, var_23);
        // wp.tile_store(x[worldid], x_slice, offset=(dofid,))                                    <L 2884>
        var_24 = wp::slice_t(var_0, var_0, var_25);
        var_26 = wp::view(var_x, var_24);
        var_27 = wp::tuple(var_4);
        wp::tile_store<wp::float32, true>(var_26, var_4, var_23);
    }
}

