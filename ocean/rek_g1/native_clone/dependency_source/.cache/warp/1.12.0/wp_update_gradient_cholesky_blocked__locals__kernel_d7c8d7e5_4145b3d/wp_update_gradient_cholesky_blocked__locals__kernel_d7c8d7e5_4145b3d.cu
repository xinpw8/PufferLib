
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
void dot_16_16_16_121_32_1_0_1_5_5_5_0(wp::float32*, wp::float32*, wp::float32*, wp::float32*, wp::float32*);
void dot_16_16_16_121_32_1_1_1_5_5_5_0(wp::float32*, wp::float32*, wp::float32*, wp::float32*, wp::float32*);
void dot_16_16_16_121_32_0_1_0_5_5_5_0(wp::float32*, wp::float32*, wp::float32*, wp::float32*, wp::float32*);
void potrf_16_16_1_121_32_1_1_5_x_x_1(wp::float32*, int*);
void trsm_16_16_1_121_32_1_0_5_0_1_1(wp::float32*, wp::float32*);
void dot_16_1_16_121_32_1_1_1_5_5_5_0(wp::float32*, wp::float32*, wp::float32*, wp::float32*, wp::float32*);
void dot_16_16_1_121_32_1_0_1_5_5_5_0(wp::float32*, wp::float32*, wp::float32*, wp::float32*, wp::float32*);
void dot_16_1_16_121_32_0_1_1_5_5_5_0(wp::float32*, wp::float32*, wp::float32*, wp::float32*, wp::float32*);
void trsm_16_1_1_121_32_1_1_5_0_1_1(wp::float32*, wp::float32*);
void dot_16_16_1_121_32_1_0_0_5_5_5_0(wp::float32*, wp::float32*, wp::float32*, wp::float32*, wp::float32*);
void trsm_16_1_1_121_32_0_1_5_0_1_0(wp::float32*, wp::float32*);
}

// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/block_cholesky.py:23
static CUDA_CALLABLE void create_blocked_cholesky_func__locals__blocked_cholesky_func_0(
    wp::array_t<wp::float32> var_A,
    wp::int32 var_matrix_size,
    wp::array_t<wp::float32> var_L)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    const wp::int32 var_1 = 16;
    wp::range_t var_2;
    wp::int32 var_3;
    const wp::int32 var_4 = 16;
    wp::int32 var_5;
    wp::tuple_t<wp::int32, wp::int32> var_6;
    wp::tuple_t<wp::int32, wp::int32> var_7;
    const wp::str var_8 = "shared";
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_9 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    const wp::int32 var_10 = 0;
    const wp::int32 var_11 = 16;
    wp::range_t var_12;
    wp::int32 var_13;
    wp::tuple_t<wp::int32, wp::int32> var_14;
    wp::tuple_t<wp::int32, wp::int32> var_15;
    const wp::str var_16 = "shared";
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_17 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<1,16>>, false> var_18 = nullptr;
    const wp::float32 var_19 = 1.0;
    const wp::float32 var_20 = -1.0;
    const wp::float32 var_21 = 1.0;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_22 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    wp::tuple_t<wp::int32, wp::int32> var_23;
    const wp::int32 var_24 = 16;
    wp::range_t var_25;
    wp::int32 var_26;
    wp::tuple_t<wp::int32, wp::int32> var_27;
    wp::tuple_t<wp::int32, wp::int32> var_28;
    const wp::str var_29 = "shared";
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_30 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    const wp::int32 var_31 = 0;
    const wp::int32 var_32 = 16;
    wp::range_t var_33;
    wp::int32 var_34;
    wp::tuple_t<wp::int32, wp::int32> var_35;
    wp::tuple_t<wp::int32, wp::int32> var_36;
    const wp::str var_37 = "shared";
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_38 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    wp::tuple_t<wp::int32, wp::int32> var_39;
    wp::tuple_t<wp::int32, wp::int32> var_40;
    const wp::str var_41 = "shared";
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_42 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<1,16>>, false> var_43 = nullptr;
    const wp::float32 var_44 = 1.0;
    const wp::float32 var_45 = -1.0;
    const wp::float32 var_46 = 1.0;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<1,16>>, false> var_47 = nullptr;
    wp::tuple_t<wp::int32, wp::int32> var_48;
    //---------
    // forward
    // def blocked_cholesky_func(                                                             <L 24>
    // for k in range(0, matrix_size, block_size):                                            <L 36>
    var_2 = wp::range(var_0, var_matrix_size, var_1);
    start_for_0:;
        if (iter_cmp(var_2) == 0) goto end_for_0;
        var_3 = wp::iter_next(var_2);
        // end = k + block_size                                                               <L 37>
        var_5 = wp::add(var_3, var_4);
        // A_kk_tile = wp.tile_load(A, shape=(block_size, block_size), offset=(k, k), storage="shared")       <L 41>
        var_6 = wp::tuple(var_4, var_4);
        var_7 = wp::tuple(var_3, var_3);
        var_9 = wp::tile_load<wp::float32, true, 16, 16>(var_A, var_3, var_3);
        // for j in range(0, k, block_size):                                                  <L 43>
        var_12 = wp::range(var_10, var_3, var_11);
        start_for_2:;
            if (iter_cmp(var_12) == 0) goto end_for_2;
            var_13 = wp::iter_next(var_12);
            // L_block = wp.tile_load(L, shape=(block_size, block_size), offset=(k, j), storage="shared")       <L 44>
            var_14 = wp::tuple(var_4, var_4);
            var_15 = wp::tuple(var_3, var_13);
            var_17 = wp::tile_load<wp::float32, true, 16, 16>(var_L, var_3, var_13);
            // wp.tile_matmul(L_block, wp.tile_transpose(L_block), A_kk_tile, alpha=-1.0)       <L 45>
            var_18 = wp::tile_transpose(var_17);
            wp::tile_matmul_acc(dot_16_16_16_121_32_1_0_1_5_5_5_0, dot_16_16_16_121_32_1_1_1_5_5_5_0, dot_16_16_16_121_32_0_1_0_5_5_5_0, var_17, var_18, var_9, var_20, var_21);
            goto start_for_2;
        end_for_2:;
        // L_kk_tile = wp.tile_cholesky(A_kk_tile)                                            <L 48>
        var_22 = tile_cholesky(potrf_16_16_1_121_32_1_1_5_x_x_1, var_9, var_22);
        // wp.tile_store(L, L_kk_tile, offset=(k, k))                                         <L 49>
        var_23 = wp::tuple(var_3, var_3);
        wp::tile_store<wp::float32, true>(var_L, var_3, var_3, var_22);
        // for i in range(end, matrix_size, block_size):                                      <L 52>
        var_25 = wp::range(var_5, var_matrix_size, var_24);
        start_for_4:;
            if (iter_cmp(var_25) == 0) goto end_for_4;
            var_26 = wp::iter_next(var_25);
            // A_ik_tile = wp.tile_load(A, shape=(block_size, block_size), offset=(i, k), storage="shared")       <L 53>
            var_27 = wp::tuple(var_4, var_4);
            var_28 = wp::tuple(var_26, var_3);
            var_30 = wp::tile_load<wp::float32, true, 16, 16>(var_A, var_26, var_3);
            // for j in range(0, k, block_size):                                              <L 55>
            var_33 = wp::range(var_31, var_3, var_32);
            start_for_6:;
                if (iter_cmp(var_33) == 0) goto end_for_6;
                var_34 = wp::iter_next(var_33);
                // L_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(i, j), storage="shared")       <L 56>
                var_35 = wp::tuple(var_4, var_4);
                var_36 = wp::tuple(var_26, var_34);
                var_38 = wp::tile_load<wp::float32, true, 16, 16>(var_L, var_26, var_34);
                // L_2_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(k, j), storage="shared")       <L 57>
                var_39 = wp::tuple(var_4, var_4);
                var_40 = wp::tuple(var_3, var_34);
                var_42 = wp::tile_load<wp::float32, true, 16, 16>(var_L, var_3, var_34);
                // wp.tile_matmul(L_tile, wp.tile_transpose(L_2_tile), A_ik_tile, alpha=-1.0)       <L 58>
                var_43 = wp::tile_transpose(var_42);
                wp::tile_matmul_acc(dot_16_16_16_121_32_1_0_1_5_5_5_0, dot_16_16_16_121_32_1_1_1_5_5_5_0, dot_16_16_16_121_32_0_1_0_5_5_5_0, var_38, var_43, var_30, var_45, var_46);
                goto start_for_6;
            end_for_6:;
            // wp.tile_lower_solve_inplace(L_kk_tile, wp.tile_transpose(A_ik_tile))           <L 60>
            var_47 = wp::tile_transpose(var_30);
            tile_lower_solve_inplace(trsm_16_16_1_121_32_1_0_5_0_1_1, var_22, var_47);
            // wp.tile_store(L, A_ik_tile, offset=(i, k))                                     <L 61>
            var_48 = wp::tuple(var_26, var_3);
            wp::tile_store<wp::float32, true>(var_L, var_26, var_3, var_30);
            wp::assign(var_13, var_34);
            goto start_for_4;
        end_for_4:;
        goto start_for_0;
    end_for_0:;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/block_cholesky.py:68
static CUDA_CALLABLE void create_blocked_cholesky_solve_func__locals__blocked_cholesky_solve_func_0(
    wp::array_t<wp::float32> var_L,
    wp::array_t<wp::float32> var_b,
    wp::int32 var_matrix_size,
    wp::array_t<wp::float32> var_x)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 80;
    const wp::int32 var_1 = 1;
    wp::tuple_t<wp::int32, wp::int32> var_2;
    const wp::int32 var_3 = 0;
    const wp::int32 var_4 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_5;
    const wp::str var_6 = "shared";
    const bool var_7 = false;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<80,1>, wp::tile_stride_t<1,1>>, true> var_8 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<80,1>,wp::tile_stride_t<1,1>,false>();
    const wp::int32 var_9 = 0;
    const wp::int32 var_10 = 0;
    const wp::int32 var_11 = 0;
    const wp::int32 var_12 = 16;
    wp::range_t var_13;
    wp::int32 var_14;
    const wp::int32 var_15 = 16;
    const wp::int32 var_16 = 1;
    wp::tuple_t<wp::int32, wp::int32> var_17;
    const wp::int32 var_18 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_19;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<1,1>>, false> var_20 = nullptr;
    const wp::int32 var_21 = 0;
    const wp::int32 var_22 = 0;
    const wp::int32 var_23 = 16;
    wp::range_t var_24;
    wp::int32 var_25;
    wp::tuple_t<wp::int32, wp::int32> var_26;
    wp::tuple_t<wp::int32, wp::int32> var_27;
    const wp::str var_28 = "shared";
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_29 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    const wp::int32 var_30 = 1;
    wp::tuple_t<wp::int32, wp::int32> var_31;
    const wp::int32 var_32 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_33;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<1,1>>, false> var_34 = nullptr;
    const wp::int32 var_35 = 0;
    const wp::float32 var_36 = 1.0;
    const wp::float32 var_37 = -1.0;
    const wp::float32 var_38 = 1.0;
    wp::tuple_t<wp::int32, wp::int32> var_39;
    wp::tuple_t<wp::int32, wp::int32> var_40;
    const wp::str var_41 = "shared";
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_42 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    wp::int32 var_43;
    const wp::int32 var_44 = -16;
    const wp::int32 var_45 = -1;
    const wp::int32 var_46 = -16;
    wp::range_t var_47;
    wp::int32 var_48;
    wp::int32 var_49;
    const wp::int32 var_50 = 1;
    wp::tuple_t<wp::int32, wp::int32> var_51;
    const wp::int32 var_52 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_53;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<1,1>>, false> var_54 = nullptr;
    const wp::int32 var_55 = 0;
    const wp::int32 var_56 = 16;
    wp::range_t var_57;
    wp::int32 var_58;
    wp::tuple_t<wp::int32, wp::int32> var_59;
    wp::tuple_t<wp::int32, wp::int32> var_60;
    const wp::str var_61 = "shared";
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_62 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    const wp::int32 var_63 = 1;
    wp::tuple_t<wp::int32, wp::int32> var_64;
    const wp::int32 var_65 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_66;
    const wp::str var_67 = "shared";
    const bool var_68 = false;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<1,1>>, true> var_69 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,1>,wp::tile_stride_t<1,1>,false>();
    const wp::int32 var_70 = 0;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<1,16>>, false> var_71 = nullptr;
    const wp::float32 var_72 = 1.0;
    const wp::float32 var_73 = -1.0;
    const wp::float32 var_74 = 1.0;
    wp::tuple_t<wp::int32, wp::int32> var_75;
    wp::tuple_t<wp::int32, wp::int32> var_76;
    const wp::str var_77 = "shared";
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_78 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<1,16>>, false> var_79 = nullptr;
    const wp::int32 var_80 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_81;
    const bool var_82 = false;
    const wp::int32 var_83 = 0;
    //---------
    // forward
    // def blocked_cholesky_solve_func(                                                       <L 69>
    // rhs_tile = wp.tile_load(b, shape=(matrix_size_static, 1), offset=(0, 0), storage="shared", bounds_check=False)       <L 82>
    var_2 = wp::tuple(var_0, var_1);
    var_5 = wp::tuple(var_3, var_4);
    var_8 = wp::tile_load<wp::float32, false, 80, 1>(var_b, var_9, var_10);
    // for i in range(0, matrix_size, block_size):                                            <L 85>
    var_13 = wp::range(var_11, var_matrix_size, var_12);
    start_for_0:;
        if (iter_cmp(var_13) == 0) goto end_for_0;
        var_14 = wp::iter_next(var_13);
        // rhs_view = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(i, 0))            <L 86>
        var_17 = wp::tuple(var_15, var_16);
        var_19 = wp::tuple(var_14, var_18);
        var_20 = wp::tile_view<wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<1,1>>, false>>(var_8, var_14, var_21);
        // for j in range(0, i, block_size):                                                  <L 87>
        var_24 = wp::range(var_22, var_14, var_23);
        start_for_2:;
            if (iter_cmp(var_24) == 0) goto end_for_2;
            var_25 = wp::iter_next(var_24);
            // L_block = wp.tile_load(L, shape=(block_size, block_size), offset=(i, j), storage="shared")       <L 88>
            var_26 = wp::tuple(var_15, var_15);
            var_27 = wp::tuple(var_14, var_25);
            var_29 = wp::tile_load<wp::float32, true, 16, 16>(var_L, var_14, var_25);
            // y_block = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(j, 0))         <L 89>
            var_31 = wp::tuple(var_15, var_30);
            var_33 = wp::tuple(var_25, var_32);
            var_34 = wp::tile_view<wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<1,1>>, false>>(var_8, var_25, var_35);
            // wp.tile_matmul(L_block, y_block, rhs_view, alpha=-1.0)                         <L 90>
            wp::tile_matmul_acc(dot_16_1_16_121_32_1_1_1_5_5_5_0, dot_16_16_1_121_32_1_0_1_5_5_5_0, dot_16_1_16_121_32_0_1_1_5_5_5_0, var_29, var_34, var_20, var_37, var_38);
            goto start_for_2;
        end_for_2:;
        // L_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(i, i), storage="shared")       <L 92>
        var_39 = wp::tuple(var_15, var_15);
        var_40 = wp::tuple(var_14, var_14);
        var_42 = wp::tile_load<wp::float32, true, 16, 16>(var_L, var_14, var_14);
        // wp.tile_lower_solve_inplace(L_tile, rhs_view)                                      <L 93>
        tile_lower_solve_inplace(trsm_16_1_1_121_32_1_1_5_0_1_1, var_42, var_20);
        goto start_for_0;
    end_for_0:;
    // for i in range(matrix_size - block_size, -1, -block_size):                             <L 96>
    var_43 = wp::sub(var_matrix_size, var_15);
    var_47 = wp::range(var_43, var_45, var_46);
    start_for_4:;
        if (iter_cmp(var_47) == 0) goto end_for_4;
        var_48 = wp::iter_next(var_47);
        // i_end = i + block_size                                                             <L 97>
        var_49 = wp::add(var_48, var_15);
        // tmp_tile = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(i, 0))            <L 98>
        var_51 = wp::tuple(var_15, var_50);
        var_53 = wp::tuple(var_48, var_52);
        var_54 = wp::tile_view<wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<1,1>>, false>>(var_8, var_48, var_55);
        // for j in range(i_end, matrix_size, block_size):                                    <L 99>
        var_57 = wp::range(var_49, var_matrix_size, var_56);
        start_for_6:;
            if (iter_cmp(var_57) == 0) goto end_for_6;
            var_58 = wp::iter_next(var_57);
            // L_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(j, i), storage="shared")       <L 100>
            var_59 = wp::tuple(var_15, var_15);
            var_60 = wp::tuple(var_58, var_48);
            var_62 = wp::tile_load<wp::float32, true, 16, 16>(var_L, var_58, var_48);
            // x_tile = wp.tile_load(x, shape=(block_size, 1), offset=(j, 0), storage="shared", bounds_check=False)       <L 101>
            var_64 = wp::tuple(var_15, var_63);
            var_66 = wp::tuple(var_58, var_65);
            var_69 = wp::tile_load<wp::float32, false, 16, 1>(var_x, var_58, var_70);
            // wp.tile_matmul(wp.tile_transpose(L_tile), x_tile, tmp_tile, alpha=-1.0)        <L 102>
            var_71 = wp::tile_transpose(var_62);
            wp::tile_matmul_acc(dot_16_1_16_121_32_0_1_1_5_5_5_0, dot_16_16_1_121_32_1_0_0_5_5_5_0, dot_16_1_16_121_32_1_1_1_5_5_5_0, var_71, var_69, var_54, var_73, var_74);
            wp::assign(var_42, var_62);
            goto start_for_6;
        end_for_6:;
        // L_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(i, i), storage="shared")       <L 103>
        var_75 = wp::tuple(var_15, var_15);
        var_76 = wp::tuple(var_48, var_48);
        var_78 = wp::tile_load<wp::float32, true, 16, 16>(var_L, var_48, var_48);
        // wp.tile_upper_solve_inplace(wp.tile_transpose(L_tile), tmp_tile)                   <L 105>
        var_79 = wp::tile_transpose(var_78);
        tile_upper_solve_inplace(trsm_16_1_1_121_32_0_1_5_0_1_0, var_79, var_54);
        // wp.tile_store(x, tmp_tile, offset=(i, 0), bounds_check=False)                      <L 106>
        var_81 = wp::tuple(var_48, var_80);
        wp::tile_store<wp::float32, false>(var_x, var_48, var_83, var_54);
        wp::assign(var_25, var_58);
        wp::assign(var_42, var_78);
        goto start_for_4;
    end_for_4:;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/block_cholesky.py:23
static CUDA_CALLABLE void adj_create_blocked_cholesky_func__locals__blocked_cholesky_func_0(
    wp::array_t<wp::float32> var_A,
    wp::int32 var_matrix_size,
    wp::array_t<wp::float32> var_L,
    wp::array_t<wp::float32> & adj_A,
    wp::int32 & adj_matrix_size,
    wp::array_t<wp::float32> & adj_L)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/block_cholesky.py:68
static CUDA_CALLABLE void adj_create_blocked_cholesky_solve_func__locals__blocked_cholesky_solve_func_0(
    wp::array_t<wp::float32> var_L,
    wp::array_t<wp::float32> var_b,
    wp::int32 var_matrix_size,
    wp::array_t<wp::float32> var_x,
    wp::array_t<wp::float32> & adj_L,
    wp::array_t<wp::float32> & adj_b,
    wp::int32 & adj_matrix_size,
    wp::array_t<wp::float32> & adj_x)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void update_gradient_cholesky_blocked__locals__kernel_3b712445_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_ctx_grad_in,
    wp::array_t<wp::float32> var_ctx_h_in,
    wp::array_t<wp::float32> var_ctx_hfactor,
    wp::array_t<wp::float32> var_ctx_Mgrad_out)
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
        const wp::int32 var_1 = 16;
        bool* var_2;
        bool var_3;
        bool var_4;
        wp::slice_t var_5;
        const wp::int32 var_6 = 0;
        wp::array_t<wp::float32> var_7;
        const wp::int32 var_8 = 80;
        wp::slice_t var_9;
        const wp::int32 var_10 = 0;
        wp::array_t<wp::float32> var_11;
        wp::slice_t var_12;
        const wp::int32 var_13 = 0;
        wp::array_t<wp::float32> var_14;
        wp::slice_t var_15;
        const wp::int32 var_16 = 0;
        wp::array_t<wp::float32> var_17;
        wp::slice_t var_18;
        const wp::int32 var_19 = 0;
        wp::array_t<wp::float32> var_20;
        //---------
        // forward
        // def kernel(                                                                            <L 2760>
        // worldid = wp.tid()                                                                     <L 2769>
        var_0 = builtin_tid1d();
        // TILE_SIZE = wp.static(tile_size)                                                       <L 2770>
        // if ctx_done_in[worldid]:                                                               <L 2772>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 2773>
            continue;
        }
        var_4 = wp::load(var_2);
        // wp.static(create_blocked_cholesky_func(TILE_SIZE))(ctx_h_in[worldid], matrix_size, ctx_hfactor[worldid])       <L 2780>
        var_5 = wp::slice_t(var_0, var_0, var_6);
        var_7 = wp::view(var_ctx_h_in, var_5);
        var_9 = wp::slice_t(var_0, var_0, var_10);
        var_11 = wp::view(var_ctx_hfactor, var_9);
        create_blocked_cholesky_func__locals__blocked_cholesky_func_0(var_7, var_8, var_11);
        // wp.static(create_blocked_cholesky_solve_func(TILE_SIZE, matrix_size))(                 <L 2781>
        // ctx_hfactor[worldid], ctx_grad_in[worldid], matrix_size, ctx_Mgrad_out[worldid]        <L 2782>
        var_12 = wp::slice_t(var_0, var_0, var_13);
        var_14 = wp::view(var_ctx_hfactor, var_12);
        var_15 = wp::slice_t(var_0, var_0, var_16);
        var_17 = wp::view(var_ctx_grad_in, var_15);
        var_18 = wp::slice_t(var_0, var_0, var_19);
        var_20 = wp::view(var_ctx_Mgrad_out, var_18);
        create_blocked_cholesky_solve_func__locals__blocked_cholesky_solve_func_0(var_14, var_17, var_8, var_20);
    }
}

