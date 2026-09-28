
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

// /home/spark-advantage/rek-training/physics-cholesky-zero-v1/cholesky_zero_candidate.py:43
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


// /home/spark-advantage/rek-training/physics-cholesky-zero-v1/cholesky_zero_candidate.py:56
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


// /home/spark-advantage/rek-training/physics-cholesky-zero-v1/cholesky_zero_candidate.py:63
static CUDA_CALLABLE bool known_zero_0(
    wp::int32 var_row,
    wp::int32 var_col,
    bool var_certified)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 48;
    bool var_1;
    const wp::int32 var_2 = 32;
    bool var_3;
    bool var_4;
    //---------
    // forward
    // def known_zero(row: int, col: int, certified: bool) -> bool:                           <L 64>
    // return certified and row >= 48 and col < 32                                            <L 65>
    var_1 = (var_row >= var_0);
    var_3 = (var_col < var_2);
    var_4 = var_certified && var_1 && var_3;
    return var_4;
}


// /home/spark-advantage/rek-training/physics-cholesky-zero-v1/cholesky_zero_candidate.py:72
static CUDA_CALLABLE void create_factor__locals__factor_0(
    wp::array_t<wp::float32> var_A,
    wp::int32 var_matrix_size,
    bool var_certified,
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
    bool var_14;
    bool var_15;
    wp::tuple_t<wp::int32, wp::int32> var_16;
    wp::tuple_t<wp::int32, wp::int32> var_17;
    const wp::str var_18 = "shared";
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_19 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<1,16>>, false> var_20 = nullptr;
    const wp::float32 var_21 = 1.0;
    const wp::float32 var_22 = -1.0;
    const wp::float32 var_23 = 1.0;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_24 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    wp::tuple_t<wp::int32, wp::int32> var_25;
    const wp::int32 var_26 = 16;
    wp::range_t var_27;
    wp::int32 var_28;
    bool var_29;
    wp::tuple_t<wp::int32, wp::int32> var_30;
    const wp::str var_31 = "shared";
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_32 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    wp::tuple_t<wp::int32, wp::int32> var_33;
    wp::tuple_t<wp::int32, wp::int32> var_34;
    wp::tuple_t<wp::int32, wp::int32> var_35;
    const wp::str var_36 = "shared";
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_37 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    const wp::int32 var_38 = 0;
    const wp::int32 var_39 = 16;
    wp::range_t var_40;
    wp::int32 var_41;
    bool var_42;
    bool var_43;
    bool var_44;
    bool var_45;
    bool var_46;
    wp::tuple_t<wp::int32, wp::int32> var_47;
    wp::tuple_t<wp::int32, wp::int32> var_48;
    const wp::str var_49 = "shared";
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_50 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    wp::tuple_t<wp::int32, wp::int32> var_51;
    wp::tuple_t<wp::int32, wp::int32> var_52;
    const wp::str var_53 = "shared";
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_54 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<1,16>>, false> var_55 = nullptr;
    const wp::float32 var_56 = 1.0;
    const wp::float32 var_57 = -1.0;
    const wp::float32 var_58 = 1.0;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<1,16>>, false> var_59 = nullptr;
    wp::tuple_t<wp::int32, wp::int32> var_60;
    wp::int32 var_61;
    //---------
    // forward
    // def factor(A: wp.array2d[float], matrix_size: int, certified: bool, L: wp.array2d[float]):       <L 73>
    // for k in range(0, matrix_size, block_size):                                            <L 74>
    var_2 = wp::range(var_0, var_matrix_size, var_1);
    start_for_0:;
        if (iter_cmp(var_2) == 0) goto end_for_0;
        var_3 = wp::iter_next(var_2);
        // end = k + block_size                                                               <L 75>
        var_5 = wp::add(var_3, var_4);
        // A_kk_tile = wp.tile_load(A, shape=(block_size, block_size), offset=(k, k), storage="shared")       <L 76>
        var_6 = wp::tuple(var_4, var_4);
        var_7 = wp::tuple(var_3, var_3);
        var_9 = wp::tile_load<wp::float32, true, 16, 16>(var_A, var_3, var_3);
        // for j in range(0, k, block_size):                                                  <L 77>
        var_12 = wp::range(var_10, var_3, var_11);
        start_for_2:;
            if (iter_cmp(var_12) == 0) goto end_for_2;
            var_13 = wp::iter_next(var_12);
            // if not known_zero(k, j, certified):                                            <L 78>
            var_14 = known_zero_0(var_3, var_13, var_certified);
            var_15 = wp::unot(var_14);
            if (var_15) {
                // L_block = wp.tile_load(L, shape=(block_size, block_size), offset=(k, j), storage="shared")       <L 79>
                var_16 = wp::tuple(var_4, var_4);
                var_17 = wp::tuple(var_3, var_13);
                var_19 = wp::tile_load<wp::float32, true, 16, 16>(var_L, var_3, var_13);
                // wp.tile_matmul(L_block, wp.tile_transpose(L_block), A_kk_tile, alpha=-1.0)       <L 80>
                var_20 = wp::tile_transpose(var_19);
                wp::tile_matmul_acc(dot_16_16_16_121_32_1_0_1_5_5_5_0, dot_16_16_16_121_32_1_1_1_5_5_5_0, dot_16_16_16_121_32_0_1_0_5_5_5_0, var_19, var_20, var_9, var_22, var_23);
            }
            goto start_for_2;
        end_for_2:;
        // L_kk_tile = wp.tile_cholesky(A_kk_tile)                                            <L 81>
        var_24 = tile_cholesky(potrf_16_16_1_121_32_1_1_5_x_x_1, var_9, var_24);
        // wp.tile_store(L, L_kk_tile, offset=(k, k))                                         <L 82>
        var_25 = wp::tuple(var_3, var_3);
        wp::tile_store<wp::float32, true>(var_L, var_3, var_3, var_24);
        // for i in range(end, matrix_size, block_size):                                      <L 83>
        var_27 = wp::range(var_5, var_matrix_size, var_26);
        start_for_4:;
            if (iter_cmp(var_27) == 0) goto end_for_4;
            var_28 = wp::iter_next(var_27);
            // if known_zero(i, k, certified):                                                <L 84>
            var_29 = known_zero_0(var_28, var_3, var_certified);
            if (var_29) {
                // zero_tile = wp.tile_zeros(shape=(block_size, block_size), dtype=float, storage="shared")       <L 85>
                var_30 = wp::tuple(var_4, var_4);
                var_32 = wp::tile_zeros<float, 16, 16>();
                // wp.tile_store(L, zero_tile, offset=(i, k))                                 <L 86>
                var_33 = wp::tuple(var_28, var_3);
                wp::tile_store<wp::float32, true>(var_L, var_28, var_3, var_32);
            }
            if (!var_29) {
                // A_ik_tile = wp.tile_load(A, shape=(block_size, block_size), offset=(i, k), storage="shared")       <L 88>
                var_34 = wp::tuple(var_4, var_4);
                var_35 = wp::tuple(var_28, var_3);
                var_37 = wp::tile_load<wp::float32, true, 16, 16>(var_A, var_28, var_3);
                // for j in range(0, k, block_size):                                          <L 89>
                var_40 = wp::range(var_38, var_3, var_39);
                start_for_6:;
                    if (iter_cmp(var_40) == 0) goto end_for_6;
                    var_41 = wp::iter_next(var_40);
                    // if not known_zero(i, j, certified) and not known_zero(k, j, certified):       <L 90>
                    var_42 = known_zero_0(var_28, var_41, var_certified);
                    var_43 = wp::unot(var_42);
                    var_44 = known_zero_0(var_3, var_41, var_certified);
                    var_45 = wp::unot(var_44);
                    var_46 = var_43 && var_45;
                    if (var_46) {
                        // L_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(i, j), storage="shared")       <L 91>
                        var_47 = wp::tuple(var_4, var_4);
                        var_48 = wp::tuple(var_28, var_41);
                        var_50 = wp::tile_load<wp::float32, true, 16, 16>(var_L, var_28, var_41);
                        // L_2_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(k, j), storage="shared")       <L 92>
                        var_51 = wp::tuple(var_4, var_4);
                        var_52 = wp::tuple(var_3, var_41);
                        var_54 = wp::tile_load<wp::float32, true, 16, 16>(var_L, var_3, var_41);
                        // wp.tile_matmul(L_tile, wp.tile_transpose(L_2_tile), A_ik_tile, alpha=-1.0)       <L 93>
                        var_55 = wp::tile_transpose(var_54);
                        wp::tile_matmul_acc(dot_16_16_16_121_32_1_0_1_5_5_5_0, dot_16_16_16_121_32_1_1_1_5_5_5_0, dot_16_16_16_121_32_0_1_0_5_5_5_0, var_50, var_55, var_37, var_57, var_58);
                    }
                    goto start_for_6;
                end_for_6:;
                // wp.tile_lower_solve_inplace(L_kk_tile, wp.tile_transpose(A_ik_tile))       <L 94>
                var_59 = wp::tile_transpose(var_37);
                tile_lower_solve_inplace(trsm_16_16_1_121_32_1_0_5_0_1_1, var_24, var_59);
                // wp.tile_store(L, A_ik_tile, offset=(i, k))                                 <L 95>
                var_60 = wp::tuple(var_28, var_3);
                wp::tile_store<wp::float32, true>(var_L, var_28, var_3, var_37);
            }
            var_61 = wp::where(var_29, var_13, var_41);
            wp::assign(var_13, var_61);
            goto start_for_4;
        end_for_4:;
        goto start_for_0;
    end_for_0:;
}


// /home/spark-advantage/rek-training/physics-cholesky-zero-v1/cholesky_zero_candidate.py:104
static CUDA_CALLABLE void create_solve__locals__solve_0(
    wp::array_t<wp::float32> var_L,
    wp::array_t<wp::float32> var_b,
    wp::int32 var_matrix_size,
    bool var_certified,
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
    bool var_26;
    bool var_27;
    wp::tuple_t<wp::int32, wp::int32> var_28;
    wp::tuple_t<wp::int32, wp::int32> var_29;
    const wp::str var_30 = "shared";
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_31 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    const wp::int32 var_32 = 1;
    wp::tuple_t<wp::int32, wp::int32> var_33;
    const wp::int32 var_34 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_35;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<1,1>>, false> var_36 = nullptr;
    const wp::int32 var_37 = 0;
    const wp::float32 var_38 = 1.0;
    const wp::float32 var_39 = -1.0;
    const wp::float32 var_40 = 1.0;
    wp::tuple_t<wp::int32, wp::int32> var_41;
    wp::tuple_t<wp::int32, wp::int32> var_42;
    const wp::str var_43 = "shared";
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_44 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    wp::int32 var_45;
    const wp::int32 var_46 = -16;
    const wp::int32 var_47 = -1;
    const wp::int32 var_48 = -16;
    wp::range_t var_49;
    wp::int32 var_50;
    wp::int32 var_51;
    const wp::int32 var_52 = 1;
    wp::tuple_t<wp::int32, wp::int32> var_53;
    const wp::int32 var_54 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_55;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<1,1>>, false> var_56 = nullptr;
    const wp::int32 var_57 = 0;
    const wp::int32 var_58 = 16;
    wp::range_t var_59;
    wp::int32 var_60;
    bool var_61;
    bool var_62;
    wp::tuple_t<wp::int32, wp::int32> var_63;
    wp::tuple_t<wp::int32, wp::int32> var_64;
    const wp::str var_65 = "shared";
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_66 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    const wp::int32 var_67 = 1;
    wp::tuple_t<wp::int32, wp::int32> var_68;
    const wp::int32 var_69 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_70;
    const wp::str var_71 = "shared";
    const bool var_72 = false;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<1,1>>, true> var_73 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,1>,wp::tile_stride_t<1,1>,false>();
    const wp::int32 var_74 = 0;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<1,16>>, false> var_75 = nullptr;
    const wp::float32 var_76 = 1.0;
    const wp::float32 var_77 = -1.0;
    const wp::float32 var_78 = 1.0;
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_79 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    wp::tuple_t<wp::int32, wp::int32> var_80;
    wp::tuple_t<wp::int32, wp::int32> var_81;
    const wp::str var_82 = "shared";
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<16,1>>, true> var_83 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,16>,wp::tile_stride_t<16,1>,false>();
    wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,16>, wp::tile_stride_t<1,16>>, false> var_84 = nullptr;
    const wp::int32 var_85 = 0;
    wp::tuple_t<wp::int32, wp::int32> var_86;
    const bool var_87 = false;
    const wp::int32 var_88 = 0;
    //---------
    // forward
    // def solve(L: wp.array2d[float], b: wp.array2d[float], matrix_size: int, certified: bool, x: wp.array2d[float]):       <L 105>
    // rhs_tile = wp.tile_load(b, shape=(matrix_size_static, 1), offset=(0, 0), storage="shared", bounds_check=False)       <L 106>
    var_2 = wp::tuple(var_0, var_1);
    var_5 = wp::tuple(var_3, var_4);
    var_8 = wp::tile_load<wp::float32, false, 80, 1>(var_b, var_9, var_10);
    // for i in range(0, matrix_size, block_size):                                            <L 107>
    var_13 = wp::range(var_11, var_matrix_size, var_12);
    start_for_0:;
        if (iter_cmp(var_13) == 0) goto end_for_0;
        var_14 = wp::iter_next(var_13);
        // rhs_view = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(i, 0))            <L 108>
        var_17 = wp::tuple(var_15, var_16);
        var_19 = wp::tuple(var_14, var_18);
        var_20 = wp::tile_view<wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<1,1>>, false>>(var_8, var_14, var_21);
        // for j in range(0, i, block_size):                                                  <L 109>
        var_24 = wp::range(var_22, var_14, var_23);
        start_for_2:;
            if (iter_cmp(var_24) == 0) goto end_for_2;
            var_25 = wp::iter_next(var_24);
            // if not known_zero(i, j, certified):                                            <L 110>
            var_26 = known_zero_0(var_14, var_25, var_certified);
            var_27 = wp::unot(var_26);
            if (var_27) {
                // L_block = wp.tile_load(L, shape=(block_size, block_size), offset=(i, j), storage="shared")       <L 111>
                var_28 = wp::tuple(var_15, var_15);
                var_29 = wp::tuple(var_14, var_25);
                var_31 = wp::tile_load<wp::float32, true, 16, 16>(var_L, var_14, var_25);
                // y_block = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(j, 0))       <L 112>
                var_33 = wp::tuple(var_15, var_32);
                var_35 = wp::tuple(var_25, var_34);
                var_36 = wp::tile_view<wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<1,1>>, false>>(var_8, var_25, var_37);
                // wp.tile_matmul(L_block, y_block, rhs_view, alpha=-1.0)                     <L 113>
                wp::tile_matmul_acc(dot_16_1_16_121_32_1_1_1_5_5_5_0, dot_16_16_1_121_32_1_0_1_5_5_5_0, dot_16_1_16_121_32_0_1_1_5_5_5_0, var_31, var_36, var_20, var_39, var_40);
            }
            goto start_for_2;
        end_for_2:;
        // L_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(i, i), storage="shared")       <L 114>
        var_41 = wp::tuple(var_15, var_15);
        var_42 = wp::tuple(var_14, var_14);
        var_44 = wp::tile_load<wp::float32, true, 16, 16>(var_L, var_14, var_14);
        // wp.tile_lower_solve_inplace(L_tile, rhs_view)                                      <L 115>
        tile_lower_solve_inplace(trsm_16_1_1_121_32_1_1_5_0_1_1, var_44, var_20);
        goto start_for_0;
    end_for_0:;
    // for i in range(matrix_size - block_size, -1, -block_size):                             <L 116>
    var_45 = wp::sub(var_matrix_size, var_15);
    var_49 = wp::range(var_45, var_47, var_48);
    start_for_4:;
        if (iter_cmp(var_49) == 0) goto end_for_4;
        var_50 = wp::iter_next(var_49);
        // i_end = i + block_size                                                             <L 117>
        var_51 = wp::add(var_50, var_15);
        // tmp_tile = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(i, 0))            <L 118>
        var_53 = wp::tuple(var_15, var_52);
        var_55 = wp::tuple(var_50, var_54);
        var_56 = wp::tile_view<wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,1>, wp::tile_stride_t<1,1>>, false>>(var_8, var_50, var_57);
        // for j in range(i_end, matrix_size, block_size):                                    <L 119>
        var_59 = wp::range(var_51, var_matrix_size, var_58);
        start_for_6:;
            if (iter_cmp(var_59) == 0) goto end_for_6;
            var_60 = wp::iter_next(var_59);
            // if not known_zero(j, i, certified):                                            <L 120>
            var_61 = known_zero_0(var_60, var_50, var_certified);
            var_62 = wp::unot(var_61);
            if (var_62) {
                // L_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(j, i), storage="shared")       <L 121>
                var_63 = wp::tuple(var_15, var_15);
                var_64 = wp::tuple(var_60, var_50);
                var_66 = wp::tile_load<wp::float32, true, 16, 16>(var_L, var_60, var_50);
                // x_tile = wp.tile_load(x, shape=(block_size, 1), offset=(j, 0), storage="shared", bounds_check=False)       <L 122>
                var_68 = wp::tuple(var_15, var_67);
                var_70 = wp::tuple(var_60, var_69);
                var_73 = wp::tile_load<wp::float32, false, 16, 1>(var_x, var_60, var_74);
                // wp.tile_matmul(wp.tile_transpose(L_tile), x_tile, tmp_tile, alpha=-1.0)       <L 123>
                var_75 = wp::tile_transpose(var_66);
                wp::tile_matmul_acc(dot_16_1_16_121_32_0_1_1_5_5_5_0, dot_16_16_1_121_32_1_0_0_5_5_5_0, dot_16_1_16_121_32_1_1_1_5_5_5_0, var_75, var_73, var_56, var_77, var_78);
            }
            var_79 = wp::where(var_62, var_66, var_44);
            wp::assign(var_44, var_79);
            goto start_for_6;
        end_for_6:;
        // L_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(i, i), storage="shared")       <L 124>
        var_80 = wp::tuple(var_15, var_15);
        var_81 = wp::tuple(var_50, var_50);
        var_83 = wp::tile_load<wp::float32, true, 16, 16>(var_L, var_50, var_50);
        // wp.tile_upper_solve_inplace(wp.tile_transpose(L_tile), tmp_tile)                   <L 125>
        var_84 = wp::tile_transpose(var_83);
        tile_upper_solve_inplace(trsm_16_1_1_121_32_0_1_5_0_1_0, var_84, var_56);
        // wp.tile_store(x, tmp_tile, offset=(i, 0), bounds_check=False)                      <L 126>
        var_86 = wp::tuple(var_50, var_85);
        wp::tile_store<wp::float32, false>(var_x, var_50, var_88, var_56);
        wp::assign(var_25, var_60);
        wp::assign(var_44, var_83);
        goto start_for_4;
    end_for_4:;
}


// /home/spark-advantage/rek-training/physics-cholesky-zero-v1/cholesky_zero_candidate.py:43
static CUDA_CALLABLE void adj_nonzero_bits_0(
    wp::float32 value,
    wp::float32 & adj_value,
    wp::int32 & adj_ret)
{
}


// /home/spark-advantage/rek-training/physics-cholesky-zero-v1/cholesky_zero_candidate.py:56
static CUDA_CALLABLE void adj_certified_zero_rectangle_0(
    wp::array_t<wp::float32> var_A,
    wp::array_t<wp::float32> & adj_A,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/physics-cholesky-zero-v1/cholesky_zero_candidate.py:63
static CUDA_CALLABLE void adj_known_zero_0(
    wp::int32 var_row,
    wp::int32 var_col,
    bool var_certified,
    wp::int32 & adj_row,
    wp::int32 & adj_col,
    bool & adj_certified,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/physics-cholesky-zero-v1/cholesky_zero_candidate.py:72
static CUDA_CALLABLE void adj_create_factor__locals__factor_0(
    wp::array_t<wp::float32> var_A,
    wp::int32 var_matrix_size,
    bool var_certified,
    wp::array_t<wp::float32> var_L,
    wp::array_t<wp::float32> & adj_A,
    wp::int32 & adj_matrix_size,
    bool & adj_certified,
    wp::array_t<wp::float32> & adj_L)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/physics-cholesky-zero-v1/cholesky_zero_candidate.py:104
static CUDA_CALLABLE void adj_create_solve__locals__solve_0(
    wp::array_t<wp::float32> var_L,
    wp::array_t<wp::float32> var_b,
    wp::int32 var_matrix_size,
    bool var_certified,
    wp::array_t<wp::float32> var_x,
    wp::array_t<wp::float32> & adj_L,
    wp::array_t<wp::float32> & adj_b,
    wp::int32 & adj_matrix_size,
    bool & adj_certified,
    wp::array_t<wp::float32> & adj_x)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void candidate_kernel__locals__kernel_07bfd255_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<bool> var_done,
    wp::array_t<wp::float32> var_grad,
    wp::array_t<wp::float32> var_h,
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
        bool var_2;
        bool var_3;
        wp::slice_t var_4;
        const wp::int32 var_5 = 0;
        wp::array_t<wp::float32> var_6;
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
        // def kernel(                                                                            <L 137>
        // world = wp.tid()                                                                       <L 144>
        var_0 = builtin_tid1d();
        // if done[world]:                                                                        <L 145>
        var_1 = wp::address(var_done, var_0);
        var_2 = wp::load(var_1);
        if (var_2) {
            // return                                                                             <L 146>
            continue;
        }
        var_3 = wp::load(var_1);
        // certified = certified_zero_rectangle(h[world])                                         <L 147>
        var_4 = wp::slice_t(var_0, var_0, var_5);
        var_6 = wp::view(var_h, var_4);
        var_7 = certified_zero_rectangle_0(var_6);
        // wp.static(create_factor())(h[world], matrix_size, certified, factor[world])            <L 148>
        var_8 = wp::slice_t(var_0, var_0, var_9);
        var_10 = wp::view(var_h, var_8);
        var_12 = wp::slice_t(var_0, var_0, var_13);
        var_14 = wp::view(var_factor, var_12);
        create_factor__locals__factor_0(var_10, var_11, var_7, var_14);
        // wp.static(create_solve())(factor[world], grad[world], matrix_size, certified, mgrad[world])       <L 149>
        var_15 = wp::slice_t(var_0, var_0, var_16);
        var_17 = wp::view(var_factor, var_15);
        var_18 = wp::slice_t(var_0, var_0, var_19);
        var_20 = wp::view(var_grad, var_18);
        var_21 = wp::slice_t(var_0, var_0, var_22);
        var_23 = wp::view(var_mgrad, var_21);
        create_solve__locals__solve_0(var_17, var_20, var_11, var_7, var_23);
    }
}

