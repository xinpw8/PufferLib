
#define WP_TILE_BLOCK_DIM 96
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:2281
static CUDA_CALLABLE wp::float32 state_check_0(
    wp::float32 var_D,
    wp::int32 var_state)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    bool var_1;
    const wp::float32 var_2 = 0.0;
    //---------
    // forward
    // def state_check(D: float, state: int) -> float:                                        <L 2282>
    // if state == types.ConstraintState.QUADRATIC.value:                                     <L 2283>
    var_1 = (var_state == var_0);
    if (var_1) {
        // return D                                                                           <L 2284>
        return var_D;
    }
    if (!var_1) {
        // return 0.0                                                                         <L 2286>
        return var_2;
    }
    return {};
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:2289
static CUDA_CALLABLE wp::float32 active_check_0(
    wp::int32 var_tid,
    wp::int32 var_threshold)
{
    //---------
    // primal vars
    bool var_0;
    const wp::float32 var_1 = 0.0;
    const wp::float32 var_2 = 1.0;
    //---------
    // forward
    // def active_check(tid: int, threshold: int) -> float:                                   <L 2290>
    // if tid >= threshold:                                                                   <L 2291>
    var_0 = (var_tid >= var_threshold);
    if (var_0) {
        // return 0.0                                                                         <L 2292>
        return var_1;
    }
    if (!var_0) {
        // return 1.0                                                                         <L 2294>
        return var_2;
    }
    return {};
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:2281
static CUDA_CALLABLE void adj_state_check_0(
    wp::float32 var_D,
    wp::int32 var_state,
    wp::float32 & adj_D,
    wp::int32 & adj_state,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:2289
static CUDA_CALLABLE void adj_active_check_0(
    wp::int32 var_tid,
    wp::int32 var_threshold,
    wp::int32 & adj_tid,
    wp::int32 & adj_threshold,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void update_gradient_JTDAJ_dense_tiled__locals__kernel_da1232b0_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::float32> var_qM_in,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::int32> var_efc_state_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_ctx_h_out)
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
        bool* var_1;
        bool var_2;
        bool var_3;
        wp::int32* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::slice_t var_7;
        const wp::int32 var_8 = 0;
        wp::array_t<wp::float32> var_9;
        const wp::int32 var_10 = 80;
        wp::tuple_t<wp::int32, wp::int32> var_11;
        const bool var_12 = true;
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<80,80>>> var_13 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<80,80>>>{};
        const wp::int32 var_14 = 0;
        const wp::int32 var_15 = 0;
        const wp::int32 var_16 = 0;
        const wp::int32 var_17 = 1024;
        const wp::int32 var_18 = 16;
        wp::range_t var_19;
        wp::int32 var_20;
        bool var_21;
        wp::slice_t var_22;
        const wp::int32 var_23 = 0;
        wp::array_t<wp::float32> var_24;
        const wp::int32 var_25 = 16;
        wp::tuple_t<wp::int32, wp::int32> var_26;
        const wp::int32 var_27 = 0;
        wp::tuple_t<wp::int32, wp::int32> var_28;
        const bool var_29 = false;
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,80>, wp::tile_stride_t<80,1>>, true> var_30 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,80>,wp::tile_stride_t<80,1>,false>();
        const wp::int32 var_31 = 0;
        wp::slice_t var_32;
        const wp::int32 var_33 = 0;
        wp::array_t<wp::float32> var_34;
        const bool var_35 = false;
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<16>>> var_36 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<16>>>{};
        wp::slice_t var_37;
        const wp::int32 var_38 = 0;
        wp::array_t<wp::int32> var_39;
        const bool var_40 = false;
        wp::tile_register_t<wp::int32,wp::tile_layout_register_t<wp::tile_shape_t<16>>> var_41 = wp::tile_register_t<wp::int32,wp::tile_layout_register_t<wp::tile_shape_t<16>>>{};
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<16>>> var_42 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<16>>>{};
        wp::tile_register_t<wp::int32,wp::tile_layout_register_t<wp::tile_shape_t<16>>> var_43 = wp::tile_register_t<wp::int32,wp::tile_layout_register_t<wp::tile_shape_t<16>>>{};
        const wp::int32 var_44 = 0;
        const wp::int32 var_45 = 1;
        wp::tile_register_t<wp::int32,wp::tile_layout_register_t<wp::tile_shape_t<16>>> var_46 = wp::tile_register_t<wp::int32,wp::tile_layout_register_t<wp::tile_shape_t<16>>>{};
        wp::int32 var_47;
        wp::tile_register_t<wp::int32,wp::tile_layout_register_t<wp::tile_shape_t<16>>> var_48 = wp::tile_register_t<wp::int32,wp::tile_layout_register_t<wp::tile_shape_t<16>>>{};
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<16>>> var_49 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<16>>>{};
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16>, wp::tile_stride_t<1>>, true> var_50 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16>,wp::tile_stride_t<1>,false>();
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<80,16>, wp::tile_stride_t<1,80>>, false> var_51 = nullptr;
        wp::tuple_t<wp::int32, wp::int32> var_52;
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<80,16>, wp::tile_stride_t<0,1>>, false> var_53 = nullptr;
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<80,16>, wp::tile_stride_t<16,1>>, true> var_54 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<80,16>,wp::tile_stride_t<16,1>,false>();
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<80,80>, wp::tile_stride_t<80,1>>, true> var_55 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<80,80>,wp::tile_stride_t<80,1>,false>();
        const wp::int32 var_56 = 0;
        const wp::int32 var_57 = 0;
        const wp::int32 var_58 = 0;
        const wp::float32 var_59 = 1.0;
        const wp::float32 var_60 = 0.0;
        wp::slice_t var_61;
        const wp::int32 var_62 = 0;
        wp::array_t<wp::float32> var_63;
        const bool var_64 = false;
        const wp::int32 var_65 = 0;
        const wp::int32 var_66 = 0;
        //---------
        // forward
        // def kernel(                                                                            <L 2375>
        // worldid = wp.tid()                                                                     <L 2387>
        var_0 = builtin_tid1d();
        // if ctx_done_in[worldid]:                                                               <L 2389>
        var_1 = wp::address(var_ctx_done_in, var_0);
        var_2 = wp::load(var_1);
        if (var_2) {
            // return                                                                             <L 2390>
            continue;
        }
        var_3 = wp::load(var_1);
        // nefc = nefc_in[worldid]                                                                <L 2392>
        var_4 = wp::address(var_nefc_in, var_0);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // sum_val = wp.tile_load(qM_in[worldid], shape=(nv_pad, nv_pad), bounds_check=True)       <L 2394>
        var_7 = wp::slice_t(var_0, var_0, var_8);
        var_9 = wp::view(var_qM_in, var_7);
        var_11 = wp::tuple(var_10, var_10);
        var_13 = wp::tile_load<wp::float32, true, 80, 80>(var_9, var_14, var_15);
        // for k in range(0, njmax, TILE_SIZE_K):                                                 <L 2397>
        var_19 = wp::range(var_16, var_17, var_18);
        start_for_1:;
            if (iter_cmp(var_19) == 0) goto end_for_1;
            var_20 = wp::iter_next(var_19);
            // if k >= nefc:                                                                      <L 2398>
            var_21 = (var_20 >= var_5);
            if (var_21) {
                // break                                                                          <L 2399>
                goto end_for_1;
            }
            // J_kj = wp.tile_load(efc_J_in[worldid], shape=(TILE_SIZE_K, nv_pad), offset=(k, 0), bounds_check=False)       <L 2404>
            var_22 = wp::slice_t(var_0, var_0, var_23);
            var_24 = wp::view(var_efc_J_in, var_22);
            var_26 = wp::tuple(var_25, var_10);
            var_28 = wp::tuple(var_20, var_27);
            var_30 = wp::tile_load<wp::float32, false, 16, 80>(var_24, var_20, var_31);
            // D_k = wp.tile_load(efc_D_in[worldid], shape=TILE_SIZE_K, offset=k, bounds_check=False)       <L 2407>
            var_32 = wp::slice_t(var_0, var_0, var_33);
            var_34 = wp::view(var_efc_D_in, var_32);
            var_36 = wp::tile_load<wp::float32, false, 16>(var_34, var_20);
            // state = wp.tile_load(efc_state_in[worldid], shape=TILE_SIZE_K, offset=k, bounds_check=False)       <L 2408>
            var_37 = wp::slice_t(var_0, var_0, var_38);
            var_39 = wp::view(var_efc_state_in, var_37);
            var_41 = wp::tile_load<wp::int32, false, 16>(var_39, var_20);
            // D_k = wp.tile_map(state_check, D_k, state)                                         <L 2410>
            var_42 = wp::tile_binary_map(state_check_0, var_36, var_41);
            // tid_tile = wp.tile_arange(TILE_SIZE_K, dtype=int)                                  <L 2413>
            var_43 = wp::tile_arange<wp::int32, 16>(var_44, var_25, var_45);
            // threshold_tile = wp.tile_ones(shape=TILE_SIZE_K, dtype=int) * (nefc - k)           <L 2414>
            var_46 = wp::tile_ones<int, 16>();
            var_47 = wp::sub(var_5, var_20);
            var_48 = wp::tile_mul(var_46, var_47);
            // active_tile = wp.tile_map(active_check, tid_tile, threshold_tile)                  <L 2416>
            var_49 = wp::tile_binary_map(active_check_0, var_43, var_48);
            // D_k = wp.tile_map(wp.mul, active_tile, D_k)                                        <L 2417>
            var_50 = wp::tile_binary_map(wp::mul, var_49, var_42);
            // J_ki = wp.tile_map(wp.mul, wp.tile_transpose(J_kj), wp.tile_broadcast(D_k, shape=(nv_pad, TILE_SIZE_K)))       <L 2419>
            var_51 = wp::tile_transpose(var_30);
            var_52 = wp::tuple(var_10, var_25);
            var_53 = wp::tile_broadcast<80, 16, 0, 1>(var_50);
            var_54 = wp::tile_binary_map(wp::mul, var_51, var_53);
            // sum_val += wp.tile_matmul(J_ki, J_kj)                                              <L 2421>
            var_55 = wp::tile_matmul(var_56, var_57, var_58, var_54, var_30, var_55, var_59, var_60);
            wp::tile_add_inplace(var_13, var_55);
            goto start_for_1;
        end_for_1:;
        // wp.tile_store(ctx_h_out[worldid], sum_val, bounds_check=False)                         <L 2423>
        var_61 = wp::slice_t(var_0, var_0, var_62);
        var_63 = wp::view(var_ctx_h_out, var_61);
        wp::tile_store<wp::float32, false>(var_63, var_65, var_66, var_13);
    }
}

