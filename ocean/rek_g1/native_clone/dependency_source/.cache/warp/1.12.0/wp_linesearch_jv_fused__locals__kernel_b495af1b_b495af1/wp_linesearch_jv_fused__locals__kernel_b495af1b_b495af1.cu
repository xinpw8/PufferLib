
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



extern "C" __global__ void linesearch_jv_fused__locals__kernel_2d0c3c0f_cuda_kernel_forward(
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
        const bool var_12 = true;
        const wp::int32 var_13 = 0;
        bool var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::range_t var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        const wp::int32 var_24 = 0;
        wp::int32* var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        const wp::int32 var_28 = 0;
        wp::float32* var_29;
        wp::float32* var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        wp::float32 var_34;
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
        // if dofstart == 0:                                                                      <L 1456>
        var_14 = (var_2 == var_13);
        if (var_14) {
            // rownnz = efc_J_rownnz_in[worldid, efcid]                                           <L 1457>
            var_15 = wp::address(var_efc_J_rownnz_in, var_0, var_1);
            var_17 = wp::load(var_15);
            var_16 = wp::copy(var_17);
            // rowadr = efc_J_rowadr_in[worldid, efcid]                                           <L 1458>
            var_18 = wp::address(var_efc_J_rowadr_in, var_0, var_1);
            var_20 = wp::load(var_18);
            var_19 = wp::copy(var_20);
            // for k in range(rownnz):                                                            <L 1459>
            var_21 = wp::range(var_16);
            start_for_2:;
                if (iter_cmp(var_21) == 0) goto end_for_2;
                var_22 = wp::iter_next(var_21);
                // sparseid = rowadr + k                                                          <L 1460>
                var_23 = wp::add(var_19, var_22);
                // colind = efc_J_colind_in[worldid, 0, sparseid]                                 <L 1461>
                var_25 = wp::address(var_efc_J_colind_in, var_0, var_24, var_23);
                var_27 = wp::load(var_25);
                var_26 = wp::copy(var_27);
                // jv_out += efc_J_in[worldid, 0, sparseid] * ctx_search_in[worldid, colind]       <L 1462>
                var_29 = wp::address(var_efc_J_in, var_0, var_28, var_23);
                var_30 = wp::address(var_ctx_search_in, var_0, var_26);
                var_32 = wp::load(var_29);
                var_33 = wp::load(var_30);
                var_31 = wp::mul(var_32, var_33);
                var_34 = wp::add(var_10, var_31);
                wp::assign(var_10, var_34);
                goto start_for_2;
            end_for_2:;
            // ctx_jv_out[worldid, efcid] = jv_out                                                <L 1463>
            wp::array_store(var_ctx_jv_out, var_0, var_1, var_10);
        }
    }
}

