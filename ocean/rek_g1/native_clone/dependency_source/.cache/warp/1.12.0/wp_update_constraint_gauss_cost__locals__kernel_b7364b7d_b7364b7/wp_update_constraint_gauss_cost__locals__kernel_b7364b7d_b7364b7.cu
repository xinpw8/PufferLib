
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



extern "C" __global__ void update_constraint_gauss_cost__locals__kernel_706504b4_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_qacc_in,
    wp::array_t<wp::float32> var_qfrc_smooth_in,
    wp::array_t<wp::float32> var_qacc_smooth_in,
    wp::array_t<wp::float32> var_efc_Ma_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_ctx_gauss_out,
    wp::array_t<wp::float32> var_ctx_cost_out)
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
        bool* var_2;
        bool var_3;
        bool var_4;
        const wp::float32 var_5 = 0.0;
        wp::float32 var_6;
        const bool var_7 = false;
        const wp::int32 var_8 = 20;
        wp::range_t var_9;
        wp::int32 var_10;
        const wp::int32 var_11 = 20;
        wp::int32 var_12;
        wp::int32 var_13;
        const wp::int32 var_14 = 70;
        bool var_15;
        wp::float32* var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::float32* var_21;
        wp::float32* var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        const wp::float32 var_29 = 0.5;
        wp::float32 var_30;
        wp::float32 var_31;
        const wp::float32 var_32 = 0.5;
        wp::float32 var_33;
        wp::float32 var_34;
        //---------
        // forward
        // def kernel(                                                                            <L 2016>
        // worldid, dofstart = wp.tid()                                                           <L 2028>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 2030>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 2031>
            continue;
        }
        var_4 = wp::load(var_2);
        // gauss_cost = float(0.0)                                                                <L 2033>
        var_6 = wp::float(var_5);
        // if wp.static(dofs_per_thread >= nv):                                                   <L 2035>
        // for i in range(wp.static(dofs_per_thread)):                                            <L 2042>
        var_9 = wp::range(var_8);
        start_for_1:;
            if (iter_cmp(var_9) == 0) goto end_for_1;
            var_10 = wp::iter_next(var_9);
            // ii = dofstart * wp.static(dofs_per_thread) + i                                     <L 2043>
            var_12 = wp::mul(var_1, var_11);
            var_13 = wp::add(var_12, var_10);
            // if ii < nv:                                                                        <L 2044>
            var_15 = (var_13 < var_14);
            if (var_15) {
                // gauss_cost += (efc_Ma_in[worldid, ii] - qfrc_smooth_in[worldid, ii]) * (       <L 2045>
                var_16 = wp::address(var_efc_Ma_in, var_0, var_13);
                var_17 = wp::address(var_qfrc_smooth_in, var_0, var_13);
                var_19 = wp::load(var_16);
                var_20 = wp::load(var_17);
                var_18 = wp::sub(var_19, var_20);
                // qacc_in[worldid, ii] - qacc_smooth_in[worldid, ii]                             <L 2046>
                var_21 = wp::address(var_qacc_in, var_0, var_13);
                var_22 = wp::address(var_qacc_smooth_in, var_0, var_13);
                var_24 = wp::load(var_21);
                var_25 = wp::load(var_22);
                var_23 = wp::sub(var_24, var_25);
                var_26 = wp::mul(var_18, var_23);
                // gauss_cost += (efc_Ma_in[worldid, ii] - qfrc_smooth_in[worldid, ii]) * (       <L 2045>
                var_27 = wp::add(var_6, var_26);
            }
            var_28 = wp::where(var_15, var_27, var_6);
            wp::assign(var_6, var_28);
            goto start_for_1;
        end_for_1:;
        // wp.atomic_add(ctx_gauss_out, worldid, 0.5 * gauss_cost)                                <L 2048>
        var_30 = wp::mul(var_29, var_6);
        var_31 = wp::atomic_add(var_ctx_gauss_out, var_0, var_30);
        // wp.atomic_add(ctx_cost_out, worldid, 0.5 * gauss_cost)                                 <L 2049>
        var_33 = wp::mul(var_32, var_6);
        var_34 = wp::atomic_add(var_ctx_cost_out, var_0, var_33);
    }
}

