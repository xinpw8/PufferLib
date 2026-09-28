
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



extern "C" __global__ void linesearch_jv_fused__locals__kernel_629c1851_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::int32> var_efc_J_rownnz_in,
    wp::array_t<wp::int32> var_efc_J_rowadr_in,
    wp::array_t<wp::int32> var_efc_J_colind_in,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::float32> var_ctx_search_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_ctx_jv_out)
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
        wp::int32 var_2;
        wp::int32* var_3;
        bool var_4;
        wp::int32 var_5;
        bool* var_6;
        bool var_7;
        bool var_8;
        const wp::float32 var_9 = 0.0;
        wp::float32 var_10;
        const bool var_11 = false;
        const bool var_12 = false;
        const wp::int32 var_13 = 20;
        wp::range_t var_14;
        wp::int32 var_15;
        const wp::int32 var_16 = 20;
        wp::int32 var_17;
        wp::int32 var_18;
        const wp::int32 var_19 = 70;
        bool var_20;
        wp::float32* var_21;
        wp::float32* var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        //---------
        // forward
        // def kernel(                                                                            <L 1416>
        // worldid, efcid, dofstart = wp.tid()                                                    <L 1429>
        builtin_tid3d(var_0, var_1, var_2);
        // if efcid >= nefc_in[worldid]:                                                          <L 1431>
        var_3 = wp::address(var_nefc_in, var_0);
        var_5 = wp::load(var_3);
        var_4 = (var_1 >= var_5);
        if (var_4) {
            // return                                                                             <L 1432>
            continue;
        }
        // if ctx_done_in[worldid]:                                                               <L 1434>
        var_6 = wp::address(var_ctx_done_in, var_0);
        var_7 = wp::load(var_6);
        if (var_7) {
            // return                                                                             <L 1435>
            continue;
        }
        var_8 = wp::load(var_6);
        // jv_out = float(0.0)                                                                    <L 1437>
        var_10 = wp::float(var_9);
        // if wp.static(dofs_per_thread >= nv):                                                   <L 1439>
        // if wp.static(is_sparse):                                                               <L 1454>
        // for i in range(wp.static(dofs_per_thread)):                                            <L 1465>
        var_14 = wp::range(var_13);
        start_for_2:;
            if (iter_cmp(var_14) == 0) goto end_for_2;
            var_15 = wp::iter_next(var_14);
            // ii = dofstart * wp.static(dofs_per_thread) + i                                     <L 1466>
            var_17 = wp::mul(var_2, var_16);
            var_18 = wp::add(var_17, var_15);
            // if ii < nv:                                                                        <L 1467>
            var_20 = (var_18 < var_19);
            if (var_20) {
                // jv_out += efc_J_in[worldid, efcid, ii] * ctx_search_in[worldid, ii]            <L 1468>
                var_21 = wp::address(var_efc_J_in, var_0, var_1, var_18);
                var_22 = wp::address(var_ctx_search_in, var_0, var_18);
                var_24 = wp::load(var_21);
                var_25 = wp::load(var_22);
                var_23 = wp::mul(var_24, var_25);
                var_26 = wp::add(var_10, var_23);
            }
            var_27 = wp::where(var_20, var_26, var_10);
            wp::assign(var_10, var_27);
            goto start_for_2;
        end_for_2:;
        // wp.atomic_add(ctx_jv_out, worldid, efcid, jv_out)                                      <L 1469>
        var_28 = wp::atomic_add(var_ctx_jv_out, var_0, var_1, var_10);
    }
}

