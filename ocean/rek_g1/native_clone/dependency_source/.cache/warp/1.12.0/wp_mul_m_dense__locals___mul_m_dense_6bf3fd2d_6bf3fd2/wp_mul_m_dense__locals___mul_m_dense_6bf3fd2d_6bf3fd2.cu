
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



extern "C" __global__ void mul_m_dense__locals___mul_m_dense_fe9df961_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_qM_in,
    wp::array_t<wp::float32> var_vec,
    wp::array_t<bool> var_skip,
    wp::array_t<wp::float32> var_res)
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
        const bool var_2 = true;
        bool* var_3;
        bool var_4;
        bool var_5;
        const wp::float32 var_6 = 0.0;
        wp::float32 var_7;
        const wp::int32 var_8 = 70;
        wp::range_t var_9;
        wp::int32 var_10;
        wp::float32* var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        //---------
        // forward
        // def _mul_m_dense(                                                                      <L 109>
        // worldid, i = wp.tid()                                                                  <L 118>
        builtin_tid2d(var_0, var_1);
        // if wp.static(check_skip):                                                              <L 120>
        // if skip[worldid]:                                                                      <L 121>
        var_3 = wp::address(var_skip, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // return                                                                             <L 122>
            continue;
        }
        var_5 = wp::load(var_3);
        // acc = float(0.0)                                                                       <L 124>
        var_7 = wp::float(var_6);
        // for j in range(wp.static(nv)):                                                         <L 125>
        var_9 = wp::range(var_8);
        start_for_1:;
            if (iter_cmp(var_9) == 0) goto end_for_1;
            var_10 = wp::iter_next(var_9);
            // acc += qM_in[worldid, i, j] * vec[worldid, j]                                      <L 126>
            var_11 = wp::address(var_qM_in, var_0, var_1, var_10);
            var_12 = wp::address(var_vec, var_0, var_10);
            var_14 = wp::load(var_11);
            var_15 = wp::load(var_12);
            var_13 = wp::mul(var_14, var_15);
            var_16 = wp::add(var_7, var_13);
            wp::assign(var_7, var_16);
            goto start_for_1;
        end_for_1:;
        // res[worldid, i] = acc                                                                  <L 127>
        wp::array_store(var_res, var_0, var_1, var_7);
    }
}



extern "C" __global__ void mul_m_dense__locals___mul_m_dense_fe9df961_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_qM_in,
    wp::array_t<wp::float32> var_vec,
    wp::array_t<bool> var_skip,
    wp::array_t<wp::float32> var_res,
    wp::array_t<wp::float32> adj_qM_in,
    wp::array_t<wp::float32> adj_vec,
    wp::array_t<bool> adj_skip,
    wp::array_t<wp::float32> adj_res)
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
        const bool var_2 = true;
        bool* var_3;
        bool var_4;
        bool var_5;
        const wp::float32 var_6 = 0.0;
        wp::float32 var_7;
        const wp::int32 var_8 = 70;
        wp::range_t var_9;
        wp::int32 var_10;
        wp::float32* var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        bool adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        wp::float32 adj_6 = {};
        wp::float32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::range_t adj_9 = {};
        wp::int32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::float32 adj_16 = {};
        //---------
        // forward
        // def _mul_m_dense(                                                                      <L 109>
        // worldid, i = wp.tid()                                                                  <L 118>
        builtin_tid2d(var_0, var_1);
        // if wp.static(check_skip):                                                              <L 120>
        // if skip[worldid]:                                                                      <L 121>
        var_3 = wp::address(var_skip, var_0);
        var_4 = wp::load(var_3);
        if (var_4) {
            // return                                                                             <L 122>
            goto label0;
        }
        var_5 = wp::load(var_3);
        // acc = float(0.0)                                                                       <L 124>
        var_7 = wp::float(var_6);
        // for j in range(wp.static(nv)):                                                         <L 125>
        var_9 = wp::range(var_8);
        // res[worldid, i] = acc                                                                  <L 127>
        // wp::array_store(var_res, var_0, var_1, var_7);
        //---------
        // reverse
        wp::adj_array_store(var_res, var_0, var_1, var_7, adj_res, adj_0, adj_1, adj_7);
        // adj: res[worldid, i] = acc                                                             <L 127>
        var_9 = wp::iter_reverse(var_9);
        start_for_1:;
            if (iter_cmp(var_9) == 0) goto end_for_1;
            var_10 = wp::iter_next(var_9);
        	adj_11 = {};
        	adj_12 = {};
        	adj_13 = {};
        	adj_14 = {};
        	adj_15 = {};
        	adj_16 = {};
            // acc += qM_in[worldid, i, j] * vec[worldid, j]                                      <L 126>
            var_11 = wp::address(var_qM_in, var_0, var_1, var_10);
            var_12 = wp::address(var_vec, var_0, var_10);
            var_14 = wp::load(var_11);
            var_15 = wp::load(var_12);
            var_13 = wp::mul(var_14, var_15);
            var_16 = wp::add(var_7, var_13);
            wp::assign(var_7, var_16);
            wp::adj_assign(var_7, var_16, adj_7, adj_16);
            wp::adj_add(var_7, var_13, adj_7, adj_13, adj_16);
            wp::adj_mul(var_14, var_15, adj_11, adj_12, adj_13);
            wp::adj_address(var_vec, var_0, var_10, adj_vec, adj_0, adj_10, adj_12);
            wp::adj_address(var_qM_in, var_0, var_1, var_10, adj_qM_in, adj_0, adj_1, adj_10, adj_11);
            // adj: acc += qM_in[worldid, i, j] * vec[worldid, j]                                 <L 126>
        	goto start_for_1;
        end_for_1:;
        wp::adj_range(var_8, adj_8, adj_9);
        // adj: for j in range(wp.static(nv)):                                                    <L 125>
        wp::adj_float(var_6, adj_6, adj_7);
        // adj: acc = float(0.0)                                                                  <L 124>
        if (var_5) {
            label0:;
            // adj: return                                                                        <L 122>
        }
        wp::adj_address(var_skip, var_0, adj_skip, adj_0, adj_3);
        // adj: if skip[worldid]:                                                                 <L 121>
        // adj: if wp.static(check_skip):                                                         <L 120>
        // adj: worldid, i = wp.tid()                                                             <L 118>
        // adj: def _mul_m_dense(                                                                 <L 109>
        continue;
    }
}

