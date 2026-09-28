
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

// /home/spark-advantage/rek-training/physics-cholesky-split-v2/cholesky_zero_split.py:26
static CUDA_CALLABLE void zero_factor__locals__factor_0(
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
    const wp::int32 var_14 = 48;
    bool var_15;
    const wp::int32 var_16 = 32;
    bool var_17;
    bool var_18;
    bool var_19;
    wp::tuple_t<wp::int32, wp::int32> var_20;
    wp::tuple_t<wp::int32, wp::int32> var_21;
    const wp::str var_22 = "shared";
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_23 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<1,16>>, false> var_24 = nullptr;
    const wp::float32 var_25 = 1.0;
    const wp::float32 var_26 = -1.0;
    const wp::float32 var_27 = 1.0;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_28 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    wp::tuple_t<wp::int32, wp::int32> var_29;
    const wp::int32 var_30 = 16;
    wp::range_t var_31;
    wp::int32 var_32;
    const wp::int32 var_33 = 48;
    bool var_34;
    const wp::int32 var_35 = 32;
    bool var_36;
    bool var_37;
    wp::tuple_t<wp::int32, wp::int32> var_38;
    const wp::str var_39 = "register";
    wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<16,16>>> var_40 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<16,16>>>{};
    wp::tuple_t<wp::int32, wp::int32> var_41;
    wp::tuple_t<wp::int32, wp::int32> var_42;
    wp::tuple_t<wp::int32, wp::int32> var_43;
    const wp::str var_44 = "shared";
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_45 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    const wp::int32 var_46 = 0;
    const wp::int32 var_47 = 16;
    wp::range_t var_48;
    wp::int32 var_49;
    const wp::int32 var_50 = 48;
    bool var_51;
    const wp::int32 var_52 = 32;
    bool var_53;
    bool var_54;
    bool var_55;
    const wp::int32 var_56 = 48;
    bool var_57;
    const wp::int32 var_58 = 32;
    bool var_59;
    bool var_60;
    bool var_61;
    bool var_62;
    wp::tuple_t<wp::int32, wp::int32> var_63;
    wp::tuple_t<wp::int32, wp::int32> var_64;
    const wp::str var_65 = "shared";
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_66 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    wp::tuple_t<wp::int32, wp::int32> var_67;
    wp::tuple_t<wp::int32, wp::int32> var_68;
    const wp::str var_69 = "shared";
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_70 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<1,16>>, false> var_71 = nullptr;
    const wp::float32 var_72 = 1.0;
    const wp::float32 var_73 = -1.0;
    const wp::float32 var_74 = 1.0;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<1,16>>, false> var_75 = nullptr;
    wp::tuple_t<wp::int32, wp::int32> var_76;
    wp::int32 var_77;
    //---------
    // forward
    // def factor(A: wp.array2d[float], matrix_size: int, L: wp.array2d[float]):              <L 27>
    // for k in range(0, matrix_size, block_size):                                            <L 28>
    var_2 = wp::range(var_0, var_matrix_size, var_1);
    start_for_0:;
        if (iter_cmp(var_2) == 0) goto end_for_0;
        var_3 = wp::iter_next(var_2);
        // end = k + block_size                                                               <L 29>
        var_5 = wp::add(var_3, var_4);
        // A_kk_tile = wp.tile_load(A, shape=(block_size, block_size), offset=(k, k), storage="shared")       <L 30>
        var_6 = wp::tuple(var_4, var_4);
        var_7 = wp::tuple(var_3, var_3);
        var_9 = wp::tile_load<wp::float32, true, 16, 16>(var_A, var_3, var_3);
        // for j in range(0, k, block_size):                                                  <L 31>
        var_12 = wp::range(var_10, var_3, var_11);
        start_for_2:;
            if (iter_cmp(var_12) == 0) goto end_for_2;
            var_13 = wp::iter_next(var_12);
            // if not (k >= 48 and j < 32):                                                   <L 32>
            var_15 = (var_3 >= var_14);
            var_17 = (var_13 < var_16);
            var_18 = var_15 && var_17;
            var_19 = wp::unot(var_18);
            if (var_19) {
                // L_block = wp.tile_load(L, shape=(block_size, block_size), offset=(k, j), storage="shared")       <L 33>
                var_20 = wp::tuple(var_4, var_4);
                var_21 = wp::tuple(var_3, var_13);
                var_23 = wp::tile_load<wp::float32, true, 16, 16>(var_L, var_3, var_13);
                // wp.tile_matmul(L_block, wp.tile_transpose(L_block), A_kk_tile, alpha=-1.0)       <L 34>
                var_24 = wp::tile_transpose(var_23);
                wp::tile_matmul_acc(dot_16_16_16_121_32_1_0_1_5_5_5_0, dot_16_16_16_121_32_1_1_1_5_5_5_0, dot_16_16_16_121_32_0_1_0_5_5_5_0, var_23, var_24, var_9, var_26, var_27);
            }
            goto start_for_2;
        end_for_2:;
        // L_kk_tile = wp.tile_cholesky(A_kk_tile)                                            <L 35>
        var_28 = tile_cholesky(potrf_16_16_1_121_32_1_1_5_x_x_1, var_9, var_28);
        // wp.tile_store(L, L_kk_tile, offset=(k, k))                                         <L 36>
        var_29 = wp::tuple(var_3, var_3);
        wp::tile_store<wp::float32, true>(var_L, var_3, var_3, var_28);
        // for i in range(end, matrix_size, block_size):                                      <L 37>
        var_31 = wp::range(var_5, var_matrix_size, var_30);
        start_for_4:;
            if (iter_cmp(var_31) == 0) goto end_for_4;
            var_32 = wp::iter_next(var_31);
            // if (i >= 48 and k < 32):                                                       <L 38>
            var_34 = (var_32 >= var_33);
            var_36 = (var_3 < var_35);
            var_37 = var_34 && var_36;
            if (var_37) {
                // zero_tile = wp.tile_zeros(shape=(block_size, block_size), dtype=float, storage="register")       <L 39>
                var_38 = wp::tuple(var_4, var_4);
                var_40 = wp::tile_zeros<float, 16, 16>();
                // wp.tile_store(L, zero_tile, offset=(i, k))                                 <L 40>
                var_41 = wp::tuple(var_32, var_3);
                wp::tile_store<wp::float32, true>(var_L, var_32, var_3, var_40);
            }
            if (!var_37) {
                // A_ik_tile = wp.tile_load(A, shape=(block_size, block_size), offset=(i, k), storage="shared")       <L 42>
                var_42 = wp::tuple(var_4, var_4);
                var_43 = wp::tuple(var_32, var_3);
                var_45 = wp::tile_load<wp::float32, true, 16, 16>(var_A, var_32, var_3);
                // for j in range(0, k, block_size):                                          <L 43>
                var_48 = wp::range(var_46, var_3, var_47);
                start_for_6:;
                    if (iter_cmp(var_48) == 0) goto end_for_6;
                    var_49 = wp::iter_next(var_48);
                    // if not (i >= 48 and j < 32) and not (k >= 48 and j < 32):              <L 44>
                    var_51 = (var_32 >= var_50);
                    var_53 = (var_49 < var_52);
                    var_54 = var_51 && var_53;
                    var_55 = wp::unot(var_54);
                    var_57 = (var_3 >= var_56);
                    var_59 = (var_49 < var_58);
                    var_60 = var_57 && var_59;
                    var_61 = wp::unot(var_60);
                    var_62 = var_55 && var_61;
                    if (var_62) {
                        // L_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(i, j), storage="shared")       <L 45>
                        var_63 = wp::tuple(var_4, var_4);
                        var_64 = wp::tuple(var_32, var_49);
                        var_66 = wp::tile_load<wp::float32, true, 16, 16>(var_L, var_32, var_49);
                        // L_2_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(k, j), storage="shared")       <L 46>
                        var_67 = wp::tuple(var_4, var_4);
                        var_68 = wp::tuple(var_3, var_49);
                        var_70 = wp::tile_load<wp::float32, true, 16, 16>(var_L, var_3, var_49);
                        // wp.tile_matmul(L_tile, wp.tile_transpose(L_2_tile), A_ik_tile, alpha=-1.0)       <L 47>
                        var_71 = wp::tile_transpose(var_70);
                        wp::tile_matmul_acc(dot_16_16_16_121_32_1_0_1_5_5_5_0, dot_16_16_16_121_32_1_1_1_5_5_5_0, dot_16_16_16_121_32_0_1_0_5_5_5_0, var_66, var_71, var_45, var_73, var_74);
                    }
                    goto start_for_6;
                end_for_6:;
                // wp.tile_lower_solve_inplace(L_kk_tile, wp.tile_transpose(A_ik_tile))       <L 48>
                var_75 = wp::tile_transpose(var_45);
                tile_lower_solve_inplace(trsm_16_16_1_121_32_1_0_5_0_1_1, var_28, var_75);
                // wp.tile_store(L, A_ik_tile, offset=(i, k))                                 <L 49>
                var_76 = wp::tuple(var_32, var_3);
                wp::tile_store<wp::float32, true>(var_L, var_32, var_3, var_45);
            }
            var_77 = wp::where(var_37, var_13, var_49);
            wp::assign(var_13, var_77);
            goto start_for_4;
        end_for_4:;
        goto start_for_0;
    end_for_0:;
}


// /home/spark-advantage/rek-training/physics-cholesky-split-v2/cholesky_zero_split.py:58
static CUDA_CALLABLE void zero_solve__locals__solve_0(
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
    const wp::int32 var_26 = 48;
    bool var_27;
    const wp::int32 var_28 = 32;
    bool var_29;
    bool var_30;
    bool var_31;
    wp::tuple_t<wp::int32, wp::int32> var_32;
    wp::tuple_t<wp::int32, wp::int32> var_33;
    const wp::str var_34 = "shared";
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_35 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    const wp::int32 var_36 = 1;
    wp::tuple_t<wp::int32, wp::int32> var_37;
    const wp::int32 var_38 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_39;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<1,1>>, false> var_40 = nullptr;
    const wp::int32 var_41 = 0;
    const wp::float32 var_42 = 1.0;
    const wp::float32 var_43 = -1.0;
    const wp::float32 var_44 = 1.0;
    wp::tuple_t<wp::int32, wp::int32> var_45;
    wp::tuple_t<wp::int32, wp::int32> var_46;
    const wp::str var_47 = "shared";
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_48 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    wp::int32 var_49;
    const wp::int32 var_50 = -16;
    const wp::int32 var_51 = -1;
    const wp::int32 var_52 = -16;
    wp::range_t var_53;
    wp::int32 var_54;
    wp::int32 var_55;
    const wp::int32 var_56 = 1;
    wp::tuple_t<wp::int32, wp::int32> var_57;
    const wp::int32 var_58 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_59;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<1,1>>, false> var_60 = nullptr;
    const wp::int32 var_61 = 0;
    const wp::int32 var_62 = 16;
    wp::range_t var_63;
    wp::int32 var_64;
    const wp::int32 var_65 = 48;
    bool var_66;
    const wp::int32 var_67 = 32;
    bool var_68;
    bool var_69;
    bool var_70;
    wp::tuple_t<wp::int32, wp::int32> var_71;
    wp::tuple_t<wp::int32, wp::int32> var_72;
    const wp::str var_73 = "shared";
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_74 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    const wp::int32 var_75 = 1;
    wp::tuple_t<wp::int32, wp::int32> var_76;
    const wp::int32 var_77 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_78;
    const wp::str var_79 = "shared";
    const bool var_80 = false;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<1,1>>, true> var_81 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,1>,wp::tile_stride_t<1,1>,false>();
    const wp::int32 var_82 = 0;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<1,16>>, false> var_83 = nullptr;
    const wp::float32 var_84 = 1.0;
    const wp::float32 var_85 = -1.0;
    const wp::float32 var_86 = 1.0;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_87 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    wp::tuple_t<wp::int32, wp::int32> var_88;
    wp::tuple_t<wp::int32, wp::int32> var_89;
    const wp::str var_90 = "shared";
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_91 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<1,16>>, false> var_92 = nullptr;
    const wp::int32 var_93 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_94;
    const bool var_95 = false;
    const wp::int32 var_96 = 0;
    //---------
    // forward
    // def solve(L: wp.array2d[float], b: wp.array2d[float], matrix_size: int, x: wp.array2d[float]):       <L 59>
    // rhs_tile = wp.tile_load(b, shape=(matrix_size_static, 1), offset=(0, 0), storage="shared", bounds_check=False)       <L 60>
    var_2 = wp::tuple(var_0, var_1);
    var_5 = wp::tuple(var_3, var_4);
    var_8 = wp::tile_load<wp::float32, false, 80, 1>(var_b, var_9, var_10);
    // for i in range(0, matrix_size, block_size):                                            <L 61>
    var_13 = wp::range(var_11, var_matrix_size, var_12);
    start_for_0:;
        if (iter_cmp(var_13) == 0) goto end_for_0;
        var_14 = wp::iter_next(var_13);
        // rhs_view = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(i, 0))            <L 62>
        var_17 = wp::tuple(var_15, var_16);
        var_19 = wp::tuple(var_14, var_18);
        var_20 = wp::tile_view<wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<1,1>>, false>>(var_8, var_14, var_21);
        // for j in range(0, i, block_size):                                                  <L 63>
        var_24 = wp::range(var_22, var_14, var_23);
        start_for_2:;
            if (iter_cmp(var_24) == 0) goto end_for_2;
            var_25 = wp::iter_next(var_24);
            // if not (i >= 48 and j < 32):                                                   <L 64>
            var_27 = (var_14 >= var_26);
            var_29 = (var_25 < var_28);
            var_30 = var_27 && var_29;
            var_31 = wp::unot(var_30);
            if (var_31) {
                // L_block = wp.tile_load(L, shape=(block_size, block_size), offset=(i, j), storage="shared")       <L 65>
                var_32 = wp::tuple(var_15, var_15);
                var_33 = wp::tuple(var_14, var_25);
                var_35 = wp::tile_load<wp::float32, true, 16, 16>(var_L, var_14, var_25);
                // y_block = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(j, 0))       <L 66>
                var_37 = wp::tuple(var_15, var_36);
                var_39 = wp::tuple(var_25, var_38);
                var_40 = wp::tile_view<wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<1,1>>, false>>(var_8, var_25, var_41);
                // wp.tile_matmul(L_block, y_block, rhs_view, alpha=-1.0)                     <L 67>
                wp::tile_matmul_acc(dot_16_1_16_121_32_1_1_1_5_5_5_0, dot_16_16_1_121_32_1_0_1_5_5_5_0, dot_16_1_16_121_32_0_1_1_5_5_5_0, var_35, var_40, var_20, var_43, var_44);
            }
            goto start_for_2;
        end_for_2:;
        // L_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(i, i), storage="shared")       <L 68>
        var_45 = wp::tuple(var_15, var_15);
        var_46 = wp::tuple(var_14, var_14);
        var_48 = wp::tile_load<wp::float32, true, 16, 16>(var_L, var_14, var_14);
        // wp.tile_lower_solve_inplace(L_tile, rhs_view)                                      <L 69>
        tile_lower_solve_inplace(trsm_16_1_1_121_32_1_1_5_0_1_1, var_48, var_20);
        goto start_for_0;
    end_for_0:;
    // for i in range(matrix_size - block_size, -1, -block_size):                             <L 70>
    var_49 = wp::sub(var_matrix_size, var_15);
    var_53 = wp::range(var_49, var_51, var_52);
    start_for_4:;
        if (iter_cmp(var_53) == 0) goto end_for_4;
        var_54 = wp::iter_next(var_53);
        // i_end = i + block_size                                                             <L 71>
        var_55 = wp::add(var_54, var_15);
        // tmp_tile = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(i, 0))            <L 72>
        var_57 = wp::tuple(var_15, var_56);
        var_59 = wp::tuple(var_54, var_58);
        var_60 = wp::tile_view<wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<1,1>>, false>>(var_8, var_54, var_61);
        // for j in range(i_end, matrix_size, block_size):                                    <L 73>
        var_63 = wp::range(var_55, var_matrix_size, var_62);
        start_for_6:;
            if (iter_cmp(var_63) == 0) goto end_for_6;
            var_64 = wp::iter_next(var_63);
            // if not (j >= 48 and i < 32):                                                   <L 74>
            var_66 = (var_64 >= var_65);
            var_68 = (var_54 < var_67);
            var_69 = var_66 && var_68;
            var_70 = wp::unot(var_69);
            if (var_70) {
                // L_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(j, i), storage="shared")       <L 75>
                var_71 = wp::tuple(var_15, var_15);
                var_72 = wp::tuple(var_64, var_54);
                var_74 = wp::tile_load<wp::float32, true, 16, 16>(var_L, var_64, var_54);
                // x_tile = wp.tile_load(x, shape=(block_size, 1), offset=(j, 0), storage="shared", bounds_check=False)       <L 76>
                var_76 = wp::tuple(var_15, var_75);
                var_78 = wp::tuple(var_64, var_77);
                var_81 = wp::tile_load<wp::float32, false, 16, 1>(var_x, var_64, var_82);
                // wp.tile_matmul(wp.tile_transpose(L_tile), x_tile, tmp_tile, alpha=-1.0)       <L 77>
                var_83 = wp::tile_transpose(var_74);
                wp::tile_matmul_acc(dot_16_1_16_121_32_0_1_1_5_5_5_0, dot_16_16_1_121_32_1_0_0_5_5_5_0, dot_16_1_16_121_32_1_1_1_5_5_5_0, var_83, var_81, var_60, var_85, var_86);
            }
            var_87 = wp::where(var_70, var_74, var_48);
            wp::assign(var_48, var_87);
            goto start_for_6;
        end_for_6:;
        // L_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(i, i), storage="shared")       <L 78>
        var_88 = wp::tuple(var_15, var_15);
        var_89 = wp::tuple(var_54, var_54);
        var_91 = wp::tile_load<wp::float32, true, 16, 16>(var_L, var_54, var_54);
        // wp.tile_upper_solve_inplace(wp.tile_transpose(L_tile), tmp_tile)                   <L 79>
        var_92 = wp::tile_transpose(var_91);
        tile_upper_solve_inplace(trsm_16_1_1_121_32_0_1_5_0_1_0, var_92, var_60);
        // wp.tile_store(x, tmp_tile, offset=(i, 0), bounds_check=False)                      <L 80>
        var_94 = wp::tuple(var_54, var_93);
        wp::tile_store<wp::float32, false>(var_x, var_54, var_96, var_60);
        wp::assign(var_25, var_64);
        wp::assign(var_48, var_91);
        goto start_for_4;
    end_for_4:;
}


// /home/spark-advantage/rek-training/physics-cholesky-split-v2/cholesky_zero_split.py:26
static CUDA_CALLABLE void adj_zero_factor__locals__factor_0(
    wp::array_t<wp::float32> var_A,
    wp::int32 var_matrix_size,
    wp::array_t<wp::float32> var_L,
    wp::array_t<wp::float32> & adj_A,
    wp::int32 & adj_matrix_size,
    wp::array_t<wp::float32> & adj_L)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/physics-cholesky-split-v2/cholesky_zero_split.py:58
static CUDA_CALLABLE void adj_zero_solve__locals__solve_0(
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



extern "C" __global__ void zero_kernel__locals__kernel_1f186a73_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<bool> var_done,
    wp::array_t<wp::float32> var_grad,
    wp::array_t<wp::float32> var_h,
    wp::array_t<wp::int32> var_flags,
    wp::array_t<wp::float32> var_factor,
    wp::array_t<wp::float32> var_mgrad)
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
        wp::int32* var_2;
        const wp::int32 var_3 = 0;
        bool var_4;
        wp::int32 var_5;
        bool var_6;
        bool var_7;
        wp::slice_t var_8;
        const wp::int32 var_9 = 0;
        wp::array_t<wp::float32> var_10;
        const wp::int32 var_11 = 80;
        wp::slice_t var_12;
        const wp::int32 var_13 = 0;
        wp::array_t<wp::float32> var_14;
        wp::slice_t var_15;
        const wp::int32 var_16 = 0;
        wp::array_t<wp::float32> var_17;
        wp::slice_t var_18;
        const wp::int32 var_19 = 0;
        wp::array_t<wp::float32> var_20;
        wp::slice_t var_21;
        const wp::int32 var_22 = 0;
        wp::array_t<wp::float32> var_23;
        //---------
        // forward
        // def kernel(done: wp.array[bool], grad: wp.array3d[float], h: wp.array3d[float], flags: wp.array[int],       <L 90>
        // world = wp.tid()                                                                       <L 92>
        var_0 = builtin_tid1d();
        // if done[world] or flags[world] == 0:                                                   <L 93>
        var_1 = wp::address(var_done, var_0);
        var_2 = wp::address(var_flags, var_0);
        var_5 = wp::load(var_2);
        var_4 = (var_5 == var_3);
        var_6 = wp::load(var_1);
        var_7 = var_6 || var_4;
        if (var_7) {
            // return                                                                             <L 94>
            continue;
        }
        // wp.static(zero_factor())(h[world], matrix_size, factor[world])                         <L 95>
        var_8 = wp::slice_t(var_0, var_0, var_9);
        var_10 = wp::view(var_h, var_8);
        var_12 = wp::slice_t(var_0, var_0, var_13);
        var_14 = wp::view(var_factor, var_12);
        zero_factor__locals__factor_0(var_10, var_11, var_14);
        // wp.static(zero_solve())(factor[world], grad[world], matrix_size, mgrad[world])         <L 96>
        var_15 = wp::slice_t(var_0, var_0, var_16);
        var_17 = wp::view(var_factor, var_15);
        var_18 = wp::slice_t(var_0, var_0, var_19);
        var_20 = wp::view(var_grad, var_18);
        var_21 = wp::slice_t(var_0, var_0, var_22);
        var_23 = wp::view(var_mgrad, var_21);
        zero_solve__locals__solve_0(var_17, var_20, var_11, var_23);
    }
}

