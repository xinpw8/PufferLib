
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:0
static CUDA_CALLABLE wp::float32 safe_div_0(
    wp::float32 var_x,
    wp::float32 var_y)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    bool var_1;
    const wp::float32 var_2 = 1e-15;
    const wp::float32 var_3 = 1e-15;
    wp::float32 var_4;
    wp::float32 var_5;
    //---------
    // forward
    // def safe_div(x: Any, y: Any) -> Any:                                                   <L 1>
    // return x / wp.where(y != 0.0, y, types.MJ_MINVAL)                                      <L 2>
    var_1 = (var_y != var_0);
    var_4 = wp::where(var_1, var_y, var_3);
    var_5 = wp::div(var_x, var_4);
    return var_5;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:149
static CUDA_CALLABLE wp::float32 _rescale_0(
    wp::int32 var_nv,
    wp::float32 var_meaninertia,
    wp::float32 var_value)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    //---------
    // forward
    // def _rescale(nv: int, meaninertia: float, value: float) -> float:                      <L 150>
    // return value / (meaninertia * float(nv))                                               <L 151>
    var_0 = wp::float(var_nv);
    var_1 = wp::mul(var_meaninertia, var_0);
    var_2 = wp::div(var_value, var_1);
    return var_2;
}


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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:324
static CUDA_CALLABLE wp::float32 _log_scale_0(
    wp::float32 var_min_value,
    wp::float32 var_max_value,
    wp::int32 var_num_values,
    wp::int32 var_i)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    const wp::float32 var_3 = 1.0;
    const wp::int32 var_4 = 1;
    wp::int32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    //---------
    // forward
    // def _log_scale(min_value: float, max_value: float, num_values: int, i: int) -> float:       <L 325>
    // step = (wp.log(max_value) - wp.log(min_value)) / wp.max(1.0, float(num_values - 1))       <L 326>
    var_0 = wp::log(var_max_value);
    var_1 = wp::log(var_min_value);
    var_2 = wp::sub(var_0, var_1);
    var_5 = wp::sub(var_num_values, var_4);
    var_6 = wp::float(var_5);
    var_7 = wp::max(var_3, var_6);
    var_8 = wp::div(var_2, var_7);
    // return wp.exp(wp.log(min_value) + float(i) * step)                                     <L 327>
    var_9 = wp::log(var_min_value);
    var_10 = wp::float(var_i);
    var_11 = wp::mul(var_10, var_8);
    var_12 = wp::add(var_9, var_11);
    var_13 = wp::exp(var_12);
    return var_13;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:192
static CUDA_CALLABLE wp::float32 _eval_cost_0(
    wp::vec_t<3, wp::float32> var_quad,
    wp::float32 var_alpha)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::int32 var_1 = 2;
    wp::float32 var_2;
    wp::float32 var_3;
    const wp::int32 var_4 = 1;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::int32 var_8 = 0;
    wp::float32 var_9;
    wp::float32 var_10;
    //---------
    // forward
    // def _eval_cost(quad: wp.vec3, alpha: float) -> float:                                  <L 193>
    // return alpha * alpha * quad[2] + alpha * quad[1] + quad[0]                             <L 194>
    var_0 = wp::mul(var_alpha, var_alpha);
    var_2 = wp::extract(var_quad, var_1);
    var_3 = wp::mul(var_0, var_2);
    var_5 = wp::extract(var_quad, var_4);
    var_6 = wp::mul(var_alpha, var_5);
    var_7 = wp::add(var_3, var_6);
    var_9 = wp::extract(var_quad, var_8);
    var_10 = wp::add(var_7, var_9);
    return var_10;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:0
static CUDA_CALLABLE void adj_safe_div_0(
    wp::float32 var_x,
    wp::float32 var_y,
    wp::float32 & adj_x,
    wp::float32 & adj_y,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:149
static CUDA_CALLABLE void adj__rescale_0(
    wp::int32 var_nv,
    wp::float32 var_meaninertia,
    wp::float32 var_value,
    wp::int32 & adj_nv,
    wp::float32 & adj_meaninertia,
    wp::float32 & adj_value,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:324
static CUDA_CALLABLE void adj__log_scale_0(
    wp::float32 var_min_value,
    wp::float32 var_max_value,
    wp::int32 var_num_values,
    wp::int32 var_i,
    wp::float32 & adj_min_value,
    wp::float32 & adj_max_value,
    wp::int32 & adj_num_values,
    wp::int32 & adj_i,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:192
static CUDA_CALLABLE void adj__eval_cost_0(
    wp::vec_t<3, wp::float32> var_quad,
    wp::float32 var_alpha,
    wp::vec_t<3, wp::float32> & adj_quad,
    wp::float32 & adj_alpha,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void update_gradient_JTCJ_sparse_8a11766b_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_opt_impratio_invsqrt,
    wp::array_t<wp::int32> var_dof_tri_row,
    wp::array_t<wp::int32> var_dof_tri_col,
    wp::array_t<wp::float32> var_contact_dist_in,
    wp::array_t<wp::float32> var_contact_includemargin_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::int32> var_contact_worldid_in,
    wp::array_t<wp::int32> var_efc_J_rownnz_in,
    wp::array_t<wp::int32> var_efc_J_rowadr_in,
    wp::array_t<wp::int32> var_efc_J_colind_in,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::int32> var_efc_state_in,
    wp::int32 var_naconmax_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::float32> var_ctx_Jaref_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::int32 var_nblocks_perblock,
    wp::int32 var_dim_block,
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
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::range_t var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        const wp::int32 var_12 = 0;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        bool var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        bool* var_20;
        bool var_21;
        bool var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        const wp::int32 var_26 = 1;
        bool var_27;
        wp::float32* var_28;
        wp::float32* var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        const wp::float32 var_33 = 0.0;
        bool var_34;
        const wp::int32 var_35 = 0;
        wp::int32* var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        wp::int32* var_39;
        const wp::int32 var_40 = 4;
        bool var_41;
        wp::int32 var_42;
        wp::int32* var_43;
        wp::int32 var_44;
        wp::int32 var_45;
        wp::int32* var_46;
        wp::int32 var_47;
        wp::int32 var_48;
        const wp::int32 var_49 = 1;
        const wp::int32 var_50 = -1;
        wp::int32 var_51;
        const wp::int32 var_52 = 1;
        const wp::int32 var_53 = -1;
        wp::int32 var_54;
        wp::range_t var_55;
        wp::int32 var_56;
        const wp::int32 var_57 = 0;
        wp::int32 var_58;
        wp::int32* var_59;
        wp::int32 var_60;
        wp::int32 var_61;
        bool var_62;
        wp::int32 var_63;
        wp::int32 var_64;
        bool var_65;
        wp::int32 var_66;
        wp::int32 var_67;
        const wp::int32 var_68 = 0;
        bool var_69;
        const wp::int32 var_70 = 0;
        bool var_71;
        bool var_72;
        wp::int32 var_73;
        wp::int32 var_74;
        const wp::int32 var_75 = 0;
        bool var_76;
        const wp::int32 var_77 = 0;
        bool var_78;
        bool var_79;
        wp::vec_t<5, wp::float32>* var_80;
        wp::vec_t<5, wp::float32> var_81;
        wp::vec_t<5, wp::float32> var_82;
        const wp::int32 var_83 = 0;
        wp::float32 var_84;
        wp::shape_t* var_85;
        const wp::int32 var_86 = 0;
        wp::int32 var_87;
        wp::shape_t var_88;
        wp::int32 var_89;
        wp::float32* var_90;
        wp::float32 var_91;
        wp::float32 var_92;
        wp::float32 var_93;
        wp::float32* var_94;
        const wp::float32 var_95 = 1.0;
        wp::float32 var_96;
        wp::float32 var_97;
        wp::float32 var_98;
        wp::float32 var_99;
        const wp::float32 var_100 = 0.0;
        bool var_101;
        wp::float32* var_102;
        wp::float32 var_103;
        wp::float32 var_104;
        const wp::float32 var_105 = 0.0;
        const wp::float32 var_106 = 0.0;
        const wp::float32 var_107 = 0.0;
        const wp::float32 var_108 = 0.0;
        const wp::float32 var_109 = 0.0;
        wp::vec_t<6, wp::float32> var_110;
        const wp::float32 var_111 = 0.0;
        wp::float32 var_112;
        const wp::int32 var_113 = 1;
        wp::range_t var_114;
        wp::int32 var_115;
        wp::int32* var_116;
        wp::int32 var_117;
        wp::int32 var_118;
        wp::float32* var_119;
        const wp::int32 var_120 = 1;
        wp::int32 var_121;
        wp::float32 var_122;
        wp::float32 var_123;
        wp::float32 var_124;
        wp::float32 var_125;
        wp::float32 var_126;
        const wp::float32 var_127 = 0.0;
        bool var_128;
        const wp::float32 var_129 = 0.0;
        wp::float32 var_130;
        wp::float32 var_131;
        const wp::float32 var_132 = 1e-15;
        const wp::float32 var_133 = 1e-15;
        wp::float32 var_134;
        wp::float32 var_135;
        wp::float32 var_136;
        const wp::float32 var_137 = 1e-15;
        const wp::float32 var_138 = 1e-15;
        wp::float32 var_139;
        wp::float32 var_140;
        wp::float32 var_141;
        wp::float32 var_142;
        wp::float32 var_143;
        wp::float32 var_144;
        wp::float32 var_145;
        const wp::float32 var_146 = 0.0;
        wp::float32 var_147;
        wp::range_t var_148;
        wp::int32 var_149;
        const wp::int32 var_150 = 0;
        bool var_151;
        wp::int32 var_152;
        wp::float32 var_153;
        wp::int32* var_154;
        wp::int32 var_155;
        wp::int32 var_156;
        wp::int32* var_157;
        wp::int32 var_158;
        wp::int32 var_159;
        const wp::int32 var_160 = 1;
        wp::int32 var_161;
        wp::float32 var_162;
        wp::float32 var_163;
        wp::int32 var_164;
        wp::float32 var_165;
        const wp::int32 var_166 = 0;
        wp::int32 var_167;
        wp::float32* var_168;
        wp::float32 var_169;
        wp::float32 var_170;
        const wp::int32 var_171 = 0;
        wp::int32 var_172;
        wp::float32* var_173;
        wp::float32 var_174;
        wp::float32 var_175;
        wp::float32 var_176;
        const wp::int32 var_177 = 1;
        wp::int32 var_178;
        const wp::int32 var_179 = 0;
        wp::range_t var_180;
        wp::int32 var_181;
        const wp::int32 var_182 = 0;
        bool var_183;
        wp::int32 var_184;
        wp::float32 var_185;
        wp::int32* var_186;
        wp::int32 var_187;
        wp::int32 var_188;
        wp::int32* var_189;
        wp::int32 var_190;
        wp::int32 var_191;
        const wp::int32 var_192 = 1;
        wp::int32 var_193;
        wp::float32 var_194;
        wp::float32 var_195;
        wp::int32 var_196;
        wp::float32 var_197;
        const wp::int32 var_198 = 0;
        wp::int32 var_199;
        wp::float32* var_200;
        wp::float32 var_201;
        wp::float32 var_202;
        const wp::int32 var_203 = 0;
        wp::int32 var_204;
        wp::float32* var_205;
        wp::float32 var_206;
        wp::float32 var_207;
        wp::float32 var_208;
        const wp::int32 var_209 = 0;
        bool var_210;
        const wp::int32 var_211 = 0;
        bool var_212;
        bool var_213;
        const wp::float32 var_214 = 1.0;
        const wp::int32 var_215 = 0;
        bool var_216;
        wp::float32 var_217;
        wp::float32 var_218;
        wp::float32 var_219;
        const wp::int32 var_220 = 0;
        bool var_221;
        wp::float32 var_222;
        wp::float32 var_223;
        wp::float32 var_224;
        wp::float32 var_225;
        wp::float32 var_226;
        bool var_227;
        wp::float32 var_228;
        wp::float32 var_229;
        wp::float32 var_230;
        wp::float32 var_231;
        wp::float32 var_232;
        wp::float32 var_233;
        const wp::float32 var_234 = 0.0;
        bool var_235;
        wp::float32 var_236;
        wp::float32 var_237;
        wp::float32 var_238;
        bool var_239;
        wp::float32 var_240;
        wp::float32 var_241;
        wp::float32 var_242;
        wp::float32 var_243;
        wp::float32 var_244;
        wp::float32 var_245;
        //---------
        // forward
        // def update_gradient_JTCJ_sparse(                                                       <L 2430>
        // conid_start, elementid = wp.tid()                                                      <L 2458>
        builtin_tid2d(var_0, var_1);
        // dof1id = dof_tri_row[elementid]                                                        <L 2460>
        var_2 = wp::address(var_dof_tri_row, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // dof2id = dof_tri_col[elementid]                                                        <L 2461>
        var_5 = wp::address(var_dof_tri_col, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // for i in range(nblocks_perblock):                                                      <L 2463>
        var_8 = wp::range(var_nblocks_perblock);
        start_for_0:;
            if (iter_cmp(var_8) == 0) goto end_for_0;
            var_9 = wp::iter_next(var_8);
            // conid = conid_start + i * dim_block                                                <L 2464>
            var_10 = wp::mul(var_9, var_dim_block);
            var_11 = wp::add(var_0, var_10);
            // if conid >= min(nacon_in[0], naconmax_in):                                         <L 2466>
            var_13 = wp::address(var_nacon_in, var_12);
            var_15 = wp::load(var_13);
            var_14 = wp::min(var_15, var_naconmax_in);
            var_16 = (var_11 >= var_14);
            if (var_16) {
                // return                                                                         <L 2467>
                continue;
            }
            // worldid = contact_worldid_in[conid]                                                <L 2469>
            var_17 = wp::address(var_contact_worldid_in, var_11);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
            // if ctx_done_in[worldid]:                                                           <L 2470>
            var_20 = wp::address(var_ctx_done_in, var_18);
            var_21 = wp::load(var_20);
            if (var_21) {
                // continue                                                                       <L 2471>
                goto start_for_0;
            }
            var_22 = wp::load(var_20);
            // condim = contact_dim_in[conid]                                                     <L 2473>
            var_23 = wp::address(var_contact_dim_in, var_11);
            var_25 = wp::load(var_23);
            var_24 = wp::copy(var_25);
            // if condim == 1:                                                                    <L 2475>
            var_27 = (var_24 == var_26);
            if (var_27) {
                // continue                                                                       <L 2476>
                goto start_for_0;
            }
            // if contact_dist_in[conid] - contact_includemargin_in[conid] >= 0.0:                <L 2479>
            var_28 = wp::address(var_contact_dist_in, var_11);
            var_29 = wp::address(var_contact_includemargin_in, var_11);
            var_31 = wp::load(var_28);
            var_32 = wp::load(var_29);
            var_30 = wp::sub(var_31, var_32);
            var_34 = (var_30 >= var_33);
            if (var_34) {
                // continue                                                                       <L 2480>
                goto start_for_0;
            }
            // efcid0 = contact_efc_address_in[conid, 0]                                          <L 2482>
            var_36 = wp::address(var_contact_efc_address_in, var_11, var_35);
            var_38 = wp::load(var_36);
            var_37 = wp::copy(var_38);
            // if efc_state_in[worldid, efcid0] != types.ConstraintState.CONE:                    <L 2483>
            var_39 = wp::address(var_efc_state_in, var_18, var_37);
            var_42 = wp::load(var_39);
            var_41 = (var_42 != var_40);
            if (var_41) {
                // continue                                                                       <L 2484>
                goto start_for_0;
            }
            // rownnz = efc_J_rownnz_in[worldid, efcid0]                                          <L 2488>
            var_43 = wp::address(var_efc_J_rownnz_in, var_18, var_37);
            var_45 = wp::load(var_43);
            var_44 = wp::copy(var_45);
            // rowadr0 = efc_J_rowadr_in[worldid, efcid0]                                         <L 2489>
            var_46 = wp::address(var_efc_J_rowadr_in, var_18, var_37);
            var_48 = wp::load(var_46);
            var_47 = wp::copy(var_48);
            // pos1 = int(-1)                                                                     <L 2490>
            var_51 = wp::int(var_50);
            // pos2 = int(-1)                                                                     <L 2491>
            var_54 = wp::int(var_53);
            // for k in range(rownnz):                                                            <L 2492>
            var_55 = wp::range(var_44);
            start_for_3:;
                if (iter_cmp(var_55) == 0) goto end_for_3;
                var_56 = wp::iter_next(var_55);
                // col = efc_J_colind_in[worldid, 0, rowadr0 + k]                                 <L 2493>
                var_58 = wp::add(var_47, var_56);
                var_59 = wp::address(var_efc_J_colind_in, var_18, var_57, var_58);
                var_61 = wp::load(var_59);
                var_60 = wp::copy(var_61);
                // if col == dof1id:                                                              <L 2494>
                var_62 = (var_60 == var_3);
                if (var_62) {
                    // pos1 = k                                                                   <L 2495>
                    var_63 = wp::copy(var_56);
                }
                var_64 = wp::where(var_62, var_63, var_51);
                // if col == dof2id:                                                              <L 2496>
                var_65 = (var_60 == var_6);
                if (var_65) {
                    // pos2 = k                                                                   <L 2497>
                    var_66 = wp::copy(var_56);
                }
                var_67 = wp::where(var_65, var_66, var_54);
                // if pos1 >= 0 and pos2 >= 0:                                                    <L 2498>
                var_69 = (var_64 >= var_68);
                var_71 = (var_67 >= var_70);
                var_72 = var_69 && var_71;
                if (var_72) {
                    // break                                                                      <L 2499>
                    wp::assign(var_51, var_64);
                    wp::assign(var_54, var_67);
                    goto end_for_3;
                }
                var_73 = wp::where(var_72, var_51, var_64);
                var_74 = wp::where(var_72, var_54, var_67);
                wp::assign(var_51, var_73);
                wp::assign(var_54, var_74);
                goto start_for_3;
            end_for_3:;
            // if pos1 < 0 or pos2 < 0:                                                           <L 2500>
            var_76 = (var_51 < var_75);
            var_78 = (var_54 < var_77);
            var_79 = var_76 || var_78;
            if (var_79) {
                // continue                                                                       <L 2501>
                goto start_for_0;
            }
            // fri = contact_friction_in[conid]                                                   <L 2503>
            var_80 = wp::address(var_contact_friction_in, var_11);
            var_82 = wp::load(var_80);
            var_81 = wp::copy(var_82);
            // mu = fri[0] * opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]        <L 2504>
            var_84 = wp::extract(var_81, var_83);
            var_85 = &(var_opt_impratio_invsqrt.shape);
            var_88 = wp::load(var_85);
            var_87 = wp::extract(var_88, var_86);
            var_89 = wp::mod(var_18, var_87);
            var_90 = wp::address(var_opt_impratio_invsqrt, var_89);
            var_92 = wp::load(var_90);
            var_91 = wp::mul(var_84, var_92);
            // mu2 = mu * mu                                                                      <L 2506>
            var_93 = wp::mul(var_91, var_91);
            // dm = math.safe_div(efc_D_in[worldid, efcid0], mu2 * (1.0 + mu2))                   <L 2507>
            var_94 = wp::address(var_efc_D_in, var_18, var_37);
            var_96 = wp::add(var_95, var_93);
            var_97 = wp::mul(var_93, var_96);
            var_99 = wp::load(var_94);
            var_98 = safe_div_0(var_99, var_97);
            // if dm == 0.0:                                                                      <L 2509>
            var_101 = (var_98 == var_100);
            if (var_101) {
                // continue                                                                       <L 2510>
                goto start_for_0;
            }
            // n = ctx_Jaref_in[worldid, efcid0] * mu                                             <L 2512>
            var_102 = wp::address(var_ctx_Jaref_in, var_18, var_37);
            var_104 = wp::load(var_102);
            var_103 = wp::mul(var_104, var_91);
            // u = types.vec6(n, 0.0, 0.0, 0.0, 0.0, 0.0)                                         <L 2513>
            var_110 = wp::vec_t<6, wp::float32>({var_103, var_105, var_106, var_107, var_108, var_109});
            // tt = float(0.0)                                                                    <L 2515>
            var_112 = wp::float(var_111);
            // for j in range(1, condim):                                                         <L 2516>
            var_114 = wp::range(var_113, var_24);
            start_for_5:;
                if (iter_cmp(var_114) == 0) goto end_for_5;
                var_115 = wp::iter_next(var_114);
                // efcidj = contact_efc_address_in[conid, j]                                      <L 2517>
                var_116 = wp::address(var_contact_efc_address_in, var_11, var_115);
                var_118 = wp::load(var_116);
                var_117 = wp::copy(var_118);
                // uj = ctx_Jaref_in[worldid, efcidj] * fri[j - 1]                                <L 2518>
                var_119 = wp::address(var_ctx_Jaref_in, var_18, var_117);
                var_121 = wp::sub(var_115, var_120);
                var_122 = wp::extract(var_81, var_121);
                var_124 = wp::load(var_119);
                var_123 = wp::mul(var_124, var_122);
                // tt += uj * uj                                                                  <L 2519>
                var_125 = wp::mul(var_123, var_123);
                var_126 = wp::add(var_112, var_125);
                // u[j] = uj                                                                      <L 2520>
                wp::assign_inplace(var_110, var_115, var_123);
                wp::assign(var_112, var_126);
                goto start_for_5;
            end_for_5:;
            // if tt <= 0.0:                                                                      <L 2522>
            var_128 = (var_112 <= var_127);
            if (var_128) {
                // t = 0.0                                                                        <L 2523>
            }
            if (!var_128) {
                // t = wp.sqrt(tt)                                                                <L 2525>
                var_130 = wp::sqrt(var_112);
            }
            var_131 = wp::where(var_128, var_129, var_130);
            // t = wp.max(t, types.MJ_MINVAL)                                                     <L 2526>
            var_134 = wp::max(var_131, var_133);
            // ttt = wp.max(t * t * t, types.MJ_MINVAL)                                           <L 2527>
            var_135 = wp::mul(var_134, var_134);
            var_136 = wp::mul(var_135, var_134);
            var_139 = wp::max(var_136, var_138);
            // mu_over_t = math.safe_div(mu, t)                                                   <L 2530>
            var_140 = safe_div_0(var_91, var_134);
            // mu_n_over_ttt = mu * math.safe_div(n, ttt)                                         <L 2531>
            var_141 = safe_div_0(var_103, var_139);
            var_142 = wp::mul(var_91, var_141);
            // mu2_minus_mu_n_over_t = mu2 - mu * math.safe_div(n, t)                             <L 2532>
            var_143 = safe_div_0(var_103, var_134);
            var_144 = wp::mul(var_91, var_143);
            var_145 = wp::sub(var_93, var_144);
            // h = float(0.0)                                                                     <L 2534>
            var_147 = wp::float(var_146);
            // for dim1id in range(condim):                                                       <L 2536>
            var_148 = wp::range(var_24);
            start_for_7:;
                if (iter_cmp(var_148) == 0) goto end_for_7;
                var_149 = wp::iter_next(var_148);
                // if dim1id == 0:                                                                <L 2537>
                var_151 = (var_149 == var_150);
                if (var_151) {
                    // rowadr1 = rowadr0                                                          <L 2538>
                    var_152 = wp::copy(var_47);
                    // dm_fri1 = dm * mu                                                          <L 2539>
                    var_153 = wp::mul(var_98, var_91);
                }
                if (!var_151) {
                    // efcid1 = contact_efc_address_in[conid, dim1id]                             <L 2541>
                    var_154 = wp::address(var_contact_efc_address_in, var_11, var_149);
                    var_156 = wp::load(var_154);
                    var_155 = wp::copy(var_156);
                    // rowadr1 = efc_J_rowadr_in[worldid, efcid1]                                 <L 2542>
                    var_157 = wp::address(var_efc_J_rowadr_in, var_18, var_155);
                    var_159 = wp::load(var_157);
                    var_158 = wp::copy(var_159);
                    // dm_fri1 = dm * fri[dim1id - 1]                                             <L 2543>
                    var_161 = wp::sub(var_149, var_160);
                    var_162 = wp::extract(var_81, var_161);
                    var_163 = wp::mul(var_98, var_162);
                }
                var_164 = wp::where(var_151, var_152, var_158);
                var_165 = wp::where(var_151, var_153, var_163);
                // efc_J11 = efc_J_in[worldid, 0, rowadr1 + pos1]                                 <L 2546>
                var_167 = wp::add(var_164, var_51);
                var_168 = wp::address(var_efc_J_in, var_18, var_166, var_167);
                var_170 = wp::load(var_168);
                var_169 = wp::copy(var_170);
                // efc_J12 = efc_J_in[worldid, 0, rowadr1 + pos2]                                 <L 2547>
                var_172 = wp::add(var_164, var_54);
                var_173 = wp::address(var_efc_J_in, var_18, var_171, var_172);
                var_175 = wp::load(var_173);
                var_174 = wp::copy(var_175);
                // ui = u[dim1id]                                                                 <L 2549>
                var_176 = wp::extract(var_110, var_149);
                // for dim2id in range(0, dim1id + 1):                                            <L 2551>
                var_178 = wp::add(var_149, var_177);
                var_180 = wp::range(var_179, var_178);
                start_for_9:;
                    if (iter_cmp(var_180) == 0) goto end_for_9;
                    var_181 = wp::iter_next(var_180);
                    // if dim2id == 0:                                                            <L 2552>
                    var_183 = (var_181 == var_182);
                    if (var_183) {
                        // rowadr2 = rowadr0                                                      <L 2553>
                        var_184 = wp::copy(var_47);
                        // dm_fri12 = dm_fri1 * mu                                                <L 2554>
                        var_185 = wp::mul(var_165, var_91);
                    }
                    if (!var_183) {
                        // efcid2 = contact_efc_address_in[conid, dim2id]                         <L 2556>
                        var_186 = wp::address(var_contact_efc_address_in, var_11, var_181);
                        var_188 = wp::load(var_186);
                        var_187 = wp::copy(var_188);
                        // rowadr2 = efc_J_rowadr_in[worldid, efcid2]                             <L 2557>
                        var_189 = wp::address(var_efc_J_rowadr_in, var_18, var_187);
                        var_191 = wp::load(var_189);
                        var_190 = wp::copy(var_191);
                        // dm_fri12 = dm_fri1 * fri[dim2id - 1]                                   <L 2558>
                        var_193 = wp::sub(var_181, var_192);
                        var_194 = wp::extract(var_81, var_193);
                        var_195 = wp::mul(var_165, var_194);
                    }
                    var_196 = wp::where(var_183, var_184, var_190);
                    var_197 = wp::where(var_183, var_185, var_195);
                    // efc_J21 = efc_J_in[worldid, 0, rowadr2 + pos1]                             <L 2561>
                    var_199 = wp::add(var_196, var_51);
                    var_200 = wp::address(var_efc_J_in, var_18, var_198, var_199);
                    var_202 = wp::load(var_200);
                    var_201 = wp::copy(var_202);
                    // efc_J22 = efc_J_in[worldid, 0, rowadr2 + pos2]                             <L 2562>
                    var_204 = wp::add(var_196, var_54);
                    var_205 = wp::address(var_efc_J_in, var_18, var_203, var_204);
                    var_207 = wp::load(var_205);
                    var_206 = wp::copy(var_207);
                    // uj = u[dim2id]                                                             <L 2564>
                    var_208 = wp::extract(var_110, var_181);
                    // if dim1id == 0 and dim2id == 0:                                            <L 2567>
                    var_210 = (var_149 == var_209);
                    var_212 = (var_181 == var_211);
                    var_213 = var_210 && var_212;
                    if (var_213) {
                        // hcone = 1.0                                                            <L 2568>
                    }
                    if (!var_213) {
                        // elif dim1id == 0:                                                      <L 2569>
                        var_216 = (var_149 == var_215);
                        if (var_216) {
                            // hcone = -mu_over_t * uj                                            <L 2570>
                            var_217 = wp::neg(var_140);
                            var_218 = wp::mul(var_217, var_208);
                        }
                        var_219 = wp::where(var_216, var_218, var_214);
                        if (!var_216) {
                            // elif dim2id == 0:                                                  <L 2571>
                            var_221 = (var_181 == var_220);
                            if (var_221) {
                                // hcone = -mu_over_t * ui                                        <L 2572>
                                var_222 = wp::neg(var_140);
                                var_223 = wp::mul(var_222, var_176);
                            }
                            var_224 = wp::where(var_221, var_223, var_219);
                            if (!var_221) {
                                // hcone = mu_n_over_ttt * ui * uj                                <L 2574>
                                var_225 = wp::mul(var_142, var_176);
                                var_226 = wp::mul(var_225, var_208);
                                // if dim1id == dim2id:                                           <L 2577>
                                var_227 = (var_149 == var_181);
                                if (var_227) {
                                    // hcone += mu2_minus_mu_n_over_t                             <L 2578>
                                    var_228 = wp::add(var_226, var_145);
                                }
                                var_229 = wp::where(var_227, var_228, var_226);
                            }
                            var_230 = wp::where(var_221, var_224, var_229);
                        }
                        var_231 = wp::where(var_216, var_219, var_230);
                    }
                    var_232 = wp::where(var_213, var_214, var_231);
                    // hcone *= dm_fri12                                                          <L 2580>
                    var_233 = wp::mul(var_232, var_197);
                    // if hcone != 0.0:                                                           <L 2582>
                    var_235 = (var_233 != var_234);
                    if (var_235) {
                        // h += hcone * efc_J11 * efc_J22                                         <L 2583>
                        var_236 = wp::mul(var_233, var_169);
                        var_237 = wp::mul(var_236, var_206);
                        var_238 = wp::add(var_147, var_237);
                        // if dim1id != dim2id:                                                   <L 2585>
                        var_239 = (var_149 != var_181);
                        if (var_239) {
                            // h += hcone * efc_J12 * efc_J21                                     <L 2586>
                            var_240 = wp::mul(var_233, var_174);
                            var_241 = wp::mul(var_240, var_201);
                            var_242 = wp::add(var_238, var_241);
                        }
                        var_243 = wp::where(var_239, var_242, var_238);
                    }
                    var_244 = wp::where(var_235, var_243, var_147);
                    wp::assign(var_123, var_208);
                    wp::assign(var_147, var_244);
                    goto start_for_9;
                end_for_9:;
                goto start_for_7;
            end_for_7:;
            // ctx_h_out[worldid, dof1id, dof2id] += h                                            <L 2588>
            var_245 = wp::atomic_add(var_ctx_h_out, var_18, var_3, var_6, var_147);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void update_gradient_JTCJ_dense_7716f0c8_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_opt_impratio_invsqrt,
    wp::array_t<wp::int32> var_dof_tri_row,
    wp::array_t<wp::int32> var_dof_tri_col,
    wp::array_t<wp::float32> var_contact_dist_in,
    wp::array_t<wp::float32> var_contact_includemargin_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::int32> var_contact_worldid_in,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::int32> var_efc_state_in,
    wp::int32 var_naconmax_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::float32> var_ctx_Jaref_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::int32 var_nblocks_perblock,
    wp::int32 var_dim_block,
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
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::range_t var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        const wp::int32 var_12 = 0;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        bool var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        bool* var_20;
        bool var_21;
        bool var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        const wp::int32 var_26 = 1;
        bool var_27;
        wp::float32* var_28;
        wp::float32* var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        const wp::float32 var_33 = 0.0;
        bool var_34;
        const wp::int32 var_35 = 0;
        wp::int32* var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        wp::int32* var_39;
        const wp::int32 var_40 = 4;
        bool var_41;
        wp::int32 var_42;
        wp::vec_t<5, wp::float32>* var_43;
        wp::vec_t<5, wp::float32> var_44;
        wp::vec_t<5, wp::float32> var_45;
        const wp::int32 var_46 = 0;
        wp::float32 var_47;
        wp::shape_t* var_48;
        const wp::int32 var_49 = 0;
        wp::int32 var_50;
        wp::shape_t var_51;
        wp::int32 var_52;
        wp::float32* var_53;
        wp::float32 var_54;
        wp::float32 var_55;
        wp::float32 var_56;
        wp::float32* var_57;
        const wp::float32 var_58 = 1.0;
        wp::float32 var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::float32 var_62;
        const wp::float32 var_63 = 0.0;
        bool var_64;
        wp::float32* var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        const wp::float32 var_68 = 0.0;
        const wp::float32 var_69 = 0.0;
        const wp::float32 var_70 = 0.0;
        const wp::float32 var_71 = 0.0;
        const wp::float32 var_72 = 0.0;
        wp::vec_t<6, wp::float32> var_73;
        const wp::float32 var_74 = 0.0;
        wp::float32 var_75;
        const wp::int32 var_76 = 1;
        wp::range_t var_77;
        wp::int32 var_78;
        wp::int32* var_79;
        wp::int32 var_80;
        wp::int32 var_81;
        wp::float32* var_82;
        const wp::int32 var_83 = 1;
        wp::int32 var_84;
        wp::float32 var_85;
        wp::float32 var_86;
        wp::float32 var_87;
        wp::float32 var_88;
        wp::float32 var_89;
        const wp::float32 var_90 = 0.0;
        bool var_91;
        const wp::float32 var_92 = 0.0;
        wp::float32 var_93;
        wp::float32 var_94;
        const wp::float32 var_95 = 1e-15;
        const wp::float32 var_96 = 1e-15;
        wp::float32 var_97;
        wp::float32 var_98;
        wp::float32 var_99;
        const wp::float32 var_100 = 1e-15;
        const wp::float32 var_101 = 1e-15;
        wp::float32 var_102;
        const wp::float32 var_103 = 0.0;
        wp::float32 var_104;
        wp::range_t var_105;
        wp::int32 var_106;
        const wp::int32 var_107 = 0;
        bool var_108;
        wp::int32 var_109;
        wp::int32* var_110;
        wp::int32 var_111;
        wp::int32 var_112;
        wp::int32 var_113;
        wp::float32* var_114;
        wp::float32 var_115;
        wp::float32 var_116;
        wp::float32* var_117;
        wp::float32 var_118;
        wp::float32 var_119;
        wp::float32 var_120;
        const wp::int32 var_121 = 1;
        wp::int32 var_122;
        const wp::int32 var_123 = 0;
        wp::range_t var_124;
        wp::int32 var_125;
        const wp::int32 var_126 = 0;
        bool var_127;
        wp::int32 var_128;
        wp::int32* var_129;
        wp::int32 var_130;
        wp::int32 var_131;
        wp::int32 var_132;
        wp::float32* var_133;
        wp::float32 var_134;
        wp::float32 var_135;
        wp::float32* var_136;
        wp::float32 var_137;
        wp::float32 var_138;
        wp::float32 var_139;
        const wp::int32 var_140 = 0;
        bool var_141;
        const wp::int32 var_142 = 0;
        bool var_143;
        bool var_144;
        const wp::float32 var_145 = 1.0;
        const wp::int32 var_146 = 0;
        bool var_147;
        wp::float32 var_148;
        wp::float32 var_149;
        wp::float32 var_150;
        wp::float32 var_151;
        const wp::int32 var_152 = 0;
        bool var_153;
        wp::float32 var_154;
        wp::float32 var_155;
        wp::float32 var_156;
        wp::float32 var_157;
        wp::float32 var_158;
        wp::float32 var_159;
        wp::float32 var_160;
        wp::float32 var_161;
        bool var_162;
        wp::float32 var_163;
        wp::float32 var_164;
        wp::float32 var_165;
        wp::float32 var_166;
        wp::float32 var_167;
        wp::float32 var_168;
        wp::float32 var_169;
        wp::float32 var_170;
        const wp::int32 var_171 = 0;
        bool var_172;
        wp::float32 var_173;
        const wp::int32 var_174 = 1;
        wp::int32 var_175;
        wp::float32 var_176;
        wp::float32 var_177;
        const wp::int32 var_178 = 0;
        bool var_179;
        wp::float32 var_180;
        const wp::int32 var_181 = 1;
        wp::int32 var_182;
        wp::float32 var_183;
        wp::float32 var_184;
        wp::float32 var_185;
        wp::float32 var_186;
        wp::float32 var_187;
        const wp::float32 var_188 = 0.0;
        bool var_189;
        wp::float32 var_190;
        wp::float32 var_191;
        wp::float32 var_192;
        bool var_193;
        wp::float32 var_194;
        wp::float32 var_195;
        wp::float32 var_196;
        wp::float32 var_197;
        wp::float32 var_198;
        wp::float32 var_199;
        //---------
        // forward
        // def update_gradient_JTCJ_dense(                                                        <L 2592>
        // conid_start, elementid = wp.tid()                                                      <L 2617>
        builtin_tid2d(var_0, var_1);
        // dof1id = dof_tri_row[elementid]                                                        <L 2619>
        var_2 = wp::address(var_dof_tri_row, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // dof2id = dof_tri_col[elementid]                                                        <L 2620>
        var_5 = wp::address(var_dof_tri_col, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // for i in range(nblocks_perblock):                                                      <L 2622>
        var_8 = wp::range(var_nblocks_perblock);
        start_for_0:;
            if (iter_cmp(var_8) == 0) goto end_for_0;
            var_9 = wp::iter_next(var_8);
            // conid = conid_start + i * dim_block                                                <L 2623>
            var_10 = wp::mul(var_9, var_dim_block);
            var_11 = wp::add(var_0, var_10);
            // if conid >= min(nacon_in[0], naconmax_in):                                         <L 2625>
            var_13 = wp::address(var_nacon_in, var_12);
            var_15 = wp::load(var_13);
            var_14 = wp::min(var_15, var_naconmax_in);
            var_16 = (var_11 >= var_14);
            if (var_16) {
                // return                                                                         <L 2626>
                continue;
            }
            // worldid = contact_worldid_in[conid]                                                <L 2628>
            var_17 = wp::address(var_contact_worldid_in, var_11);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
            // if ctx_done_in[worldid]:                                                           <L 2629>
            var_20 = wp::address(var_ctx_done_in, var_18);
            var_21 = wp::load(var_20);
            if (var_21) {
                // continue                                                                       <L 2630>
                goto start_for_0;
            }
            var_22 = wp::load(var_20);
            // condim = contact_dim_in[conid]                                                     <L 2632>
            var_23 = wp::address(var_contact_dim_in, var_11);
            var_25 = wp::load(var_23);
            var_24 = wp::copy(var_25);
            // if condim == 1:                                                                    <L 2634>
            var_27 = (var_24 == var_26);
            if (var_27) {
                // continue                                                                       <L 2635>
                goto start_for_0;
            }
            // if contact_dist_in[conid] - contact_includemargin_in[conid] >= 0.0:                <L 2638>
            var_28 = wp::address(var_contact_dist_in, var_11);
            var_29 = wp::address(var_contact_includemargin_in, var_11);
            var_31 = wp::load(var_28);
            var_32 = wp::load(var_29);
            var_30 = wp::sub(var_31, var_32);
            var_34 = (var_30 >= var_33);
            if (var_34) {
                // continue                                                                       <L 2639>
                goto start_for_0;
            }
            // efcid0 = contact_efc_address_in[conid, 0]                                          <L 2641>
            var_36 = wp::address(var_contact_efc_address_in, var_11, var_35);
            var_38 = wp::load(var_36);
            var_37 = wp::copy(var_38);
            // if efc_state_in[worldid, efcid0] != types.ConstraintState.CONE:                    <L 2642>
            var_39 = wp::address(var_efc_state_in, var_18, var_37);
            var_42 = wp::load(var_39);
            var_41 = (var_42 != var_40);
            if (var_41) {
                // continue                                                                       <L 2643>
                goto start_for_0;
            }
            // fri = contact_friction_in[conid]                                                   <L 2645>
            var_43 = wp::address(var_contact_friction_in, var_11);
            var_45 = wp::load(var_43);
            var_44 = wp::copy(var_45);
            // mu = fri[0] * opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]        <L 2646>
            var_47 = wp::extract(var_44, var_46);
            var_48 = &(var_opt_impratio_invsqrt.shape);
            var_51 = wp::load(var_48);
            var_50 = wp::extract(var_51, var_49);
            var_52 = wp::mod(var_18, var_50);
            var_53 = wp::address(var_opt_impratio_invsqrt, var_52);
            var_55 = wp::load(var_53);
            var_54 = wp::mul(var_47, var_55);
            // mu2 = mu * mu                                                                      <L 2648>
            var_56 = wp::mul(var_54, var_54);
            // dm = math.safe_div(efc_D_in[worldid, efcid0], mu2 * (1.0 + mu2))                   <L 2649>
            var_57 = wp::address(var_efc_D_in, var_18, var_37);
            var_59 = wp::add(var_58, var_56);
            var_60 = wp::mul(var_56, var_59);
            var_62 = wp::load(var_57);
            var_61 = safe_div_0(var_62, var_60);
            // if dm == 0.0:                                                                      <L 2651>
            var_64 = (var_61 == var_63);
            if (var_64) {
                // continue                                                                       <L 2652>
                goto start_for_0;
            }
            // n = ctx_Jaref_in[worldid, efcid0] * mu                                             <L 2654>
            var_65 = wp::address(var_ctx_Jaref_in, var_18, var_37);
            var_67 = wp::load(var_65);
            var_66 = wp::mul(var_67, var_54);
            // u = types.vec6(n, 0.0, 0.0, 0.0, 0.0, 0.0)                                         <L 2655>
            var_73 = wp::vec_t<6, wp::float32>({var_66, var_68, var_69, var_70, var_71, var_72});
            // tt = float(0.0)                                                                    <L 2657>
            var_75 = wp::float(var_74);
            // for j in range(1, condim):                                                         <L 2658>
            var_77 = wp::range(var_76, var_24);
            start_for_3:;
                if (iter_cmp(var_77) == 0) goto end_for_3;
                var_78 = wp::iter_next(var_77);
                // efcidj = contact_efc_address_in[conid, j]                                      <L 2659>
                var_79 = wp::address(var_contact_efc_address_in, var_11, var_78);
                var_81 = wp::load(var_79);
                var_80 = wp::copy(var_81);
                // uj = ctx_Jaref_in[worldid, efcidj] * fri[j - 1]                                <L 2660>
                var_82 = wp::address(var_ctx_Jaref_in, var_18, var_80);
                var_84 = wp::sub(var_78, var_83);
                var_85 = wp::extract(var_44, var_84);
                var_87 = wp::load(var_82);
                var_86 = wp::mul(var_87, var_85);
                // tt += uj * uj                                                                  <L 2661>
                var_88 = wp::mul(var_86, var_86);
                var_89 = wp::add(var_75, var_88);
                // u[j] = uj                                                                      <L 2662>
                wp::assign_inplace(var_73, var_78, var_86);
                wp::assign(var_75, var_89);
                goto start_for_3;
            end_for_3:;
            // if tt <= 0.0:                                                                      <L 2664>
            var_91 = (var_75 <= var_90);
            if (var_91) {
                // t = 0.0                                                                        <L 2665>
            }
            if (!var_91) {
                // t = wp.sqrt(tt)                                                                <L 2667>
                var_93 = wp::sqrt(var_75);
            }
            var_94 = wp::where(var_91, var_92, var_93);
            // t = wp.max(t, types.MJ_MINVAL)                                                     <L 2668>
            var_97 = wp::max(var_94, var_96);
            // ttt = wp.max(t * t * t, types.MJ_MINVAL)                                           <L 2669>
            var_98 = wp::mul(var_97, var_97);
            var_99 = wp::mul(var_98, var_97);
            var_102 = wp::max(var_99, var_101);
            // h = float(0.0)                                                                     <L 2671>
            var_104 = wp::float(var_103);
            // for dim1id in range(condim):                                                       <L 2673>
            var_105 = wp::range(var_24);
            start_for_5:;
                if (iter_cmp(var_105) == 0) goto end_for_5;
                var_106 = wp::iter_next(var_105);
                // if dim1id == 0:                                                                <L 2674>
                var_108 = (var_106 == var_107);
                if (var_108) {
                    // efcid1 = efcid0                                                            <L 2675>
                    var_109 = wp::copy(var_37);
                }
                if (!var_108) {
                    // efcid1 = contact_efc_address_in[conid, dim1id]                             <L 2677>
                    var_110 = wp::address(var_contact_efc_address_in, var_11, var_106);
                    var_112 = wp::load(var_110);
                    var_111 = wp::copy(var_112);
                }
                var_113 = wp::where(var_108, var_109, var_111);
                // efc_J11 = efc_J_in[worldid, efcid1, dof1id]                                    <L 2679>
                var_114 = wp::address(var_efc_J_in, var_18, var_113, var_3);
                var_116 = wp::load(var_114);
                var_115 = wp::copy(var_116);
                // efc_J12 = efc_J_in[worldid, efcid1, dof2id]                                    <L 2680>
                var_117 = wp::address(var_efc_J_in, var_18, var_113, var_6);
                var_119 = wp::load(var_117);
                var_118 = wp::copy(var_119);
                // ui = u[dim1id]                                                                 <L 2682>
                var_120 = wp::extract(var_73, var_106);
                // for dim2id in range(0, dim1id + 1):                                            <L 2684>
                var_122 = wp::add(var_106, var_121);
                var_124 = wp::range(var_123, var_122);
                start_for_7:;
                    if (iter_cmp(var_124) == 0) goto end_for_7;
                    var_125 = wp::iter_next(var_124);
                    // if dim2id == 0:                                                            <L 2685>
                    var_127 = (var_125 == var_126);
                    if (var_127) {
                        // efcid2 = efcid0                                                        <L 2686>
                        var_128 = wp::copy(var_37);
                    }
                    if (!var_127) {
                        // efcid2 = contact_efc_address_in[conid, dim2id]                         <L 2688>
                        var_129 = wp::address(var_contact_efc_address_in, var_11, var_125);
                        var_131 = wp::load(var_129);
                        var_130 = wp::copy(var_131);
                    }
                    var_132 = wp::where(var_127, var_128, var_130);
                    // efc_J21 = efc_J_in[worldid, efcid2, dof1id]                                <L 2690>
                    var_133 = wp::address(var_efc_J_in, var_18, var_132, var_3);
                    var_135 = wp::load(var_133);
                    var_134 = wp::copy(var_135);
                    // efc_J22 = efc_J_in[worldid, efcid2, dof2id]                                <L 2691>
                    var_136 = wp::address(var_efc_J_in, var_18, var_132, var_6);
                    var_138 = wp::load(var_136);
                    var_137 = wp::copy(var_138);
                    // uj = u[dim2id]                                                             <L 2693>
                    var_139 = wp::extract(var_73, var_125);
                    // if dim1id == 0 and dim2id == 0:                                            <L 2696>
                    var_141 = (var_106 == var_140);
                    var_143 = (var_125 == var_142);
                    var_144 = var_141 && var_143;
                    if (var_144) {
                        // hcone = 1.0                                                            <L 2697>
                    }
                    if (!var_144) {
                        // elif dim1id == 0:                                                      <L 2698>
                        var_147 = (var_106 == var_146);
                        if (var_147) {
                            // hcone = -math.safe_div(mu, t) * uj                                 <L 2699>
                            var_148 = safe_div_0(var_54, var_97);
                            var_149 = wp::neg(var_148);
                            var_150 = wp::mul(var_149, var_139);
                        }
                        var_151 = wp::where(var_147, var_150, var_145);
                        if (!var_147) {
                            // elif dim2id == 0:                                                  <L 2700>
                            var_153 = (var_125 == var_152);
                            if (var_153) {
                                // hcone = -math.safe_div(mu, t) * ui                             <L 2701>
                                var_154 = safe_div_0(var_54, var_97);
                                var_155 = wp::neg(var_154);
                                var_156 = wp::mul(var_155, var_120);
                            }
                            var_157 = wp::where(var_153, var_156, var_151);
                            if (!var_153) {
                                // hcone = mu * math.safe_div(n, ttt) * ui * uj                   <L 2703>
                                var_158 = safe_div_0(var_66, var_102);
                                var_159 = wp::mul(var_54, var_158);
                                var_160 = wp::mul(var_159, var_120);
                                var_161 = wp::mul(var_160, var_139);
                                // if dim1id == dim2id:                                           <L 2706>
                                var_162 = (var_106 == var_125);
                                if (var_162) {
                                    // hcone += mu2 - mu * math.safe_div(n, t)                    <L 2707>
                                    var_163 = safe_div_0(var_66, var_97);
                                    var_164 = wp::mul(var_54, var_163);
                                    var_165 = wp::sub(var_56, var_164);
                                    var_166 = wp::add(var_161, var_165);
                                }
                                var_167 = wp::where(var_162, var_166, var_161);
                            }
                            var_168 = wp::where(var_153, var_157, var_167);
                        }
                        var_169 = wp::where(var_147, var_151, var_168);
                    }
                    var_170 = wp::where(var_144, var_145, var_169);
                    // if dim1id == 0:                                                            <L 2710>
                    var_172 = (var_106 == var_171);
                    if (var_172) {
                        // fri1 = mu                                                              <L 2711>
                        var_173 = wp::copy(var_54);
                    }
                    if (!var_172) {
                        // fri1 = fri[dim1id - 1]                                                 <L 2713>
                        var_175 = wp::sub(var_106, var_174);
                        var_176 = wp::extract(var_44, var_175);
                    }
                    var_177 = wp::where(var_172, var_173, var_176);
                    // if dim2id == 0:                                                            <L 2715>
                    var_179 = (var_125 == var_178);
                    if (var_179) {
                        // fri2 = mu                                                              <L 2716>
                        var_180 = wp::copy(var_54);
                    }
                    if (!var_179) {
                        // fri2 = fri[dim2id - 1]                                                 <L 2718>
                        var_182 = wp::sub(var_125, var_181);
                        var_183 = wp::extract(var_44, var_182);
                    }
                    var_184 = wp::where(var_179, var_180, var_183);
                    // hcone *= dm * fri1 * fri2                                                  <L 2720>
                    var_185 = wp::mul(var_61, var_177);
                    var_186 = wp::mul(var_185, var_184);
                    var_187 = wp::mul(var_170, var_186);
                    // if hcone != 0.0:                                                           <L 2722>
                    var_189 = (var_187 != var_188);
                    if (var_189) {
                        // h += hcone * efc_J11 * efc_J22                                         <L 2723>
                        var_190 = wp::mul(var_187, var_115);
                        var_191 = wp::mul(var_190, var_137);
                        var_192 = wp::add(var_104, var_191);
                        // if dim1id != dim2id:                                                   <L 2725>
                        var_193 = (var_106 != var_125);
                        if (var_193) {
                            // h += hcone * efc_J12 * efc_J21                                     <L 2726>
                            var_194 = wp::mul(var_187, var_118);
                            var_195 = wp::mul(var_194, var_134);
                            var_196 = wp::add(var_192, var_195);
                        }
                        var_197 = wp::where(var_193, var_196, var_192);
                    }
                    var_198 = wp::where(var_189, var_197, var_104);
                    wp::assign(var_86, var_139);
                    wp::assign(var_104, var_198);
                    goto start_for_7;
                end_for_7:;
                goto start_for_5;
            end_for_5:;
            // ctx_h_out[worldid, dof1id, dof2id] += h                                            <L 2728>
            var_199 = wp::atomic_add(var_ctx_h_out, var_18, var_3, var_6, var_104);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void update_constraint_init_qfrc_constraint_sparse_77da3ea2_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::int32> var_efc_J_rownnz_in,
    wp::array_t<wp::int32> var_efc_J_rowadr_in,
    wp::array_t<wp::int32> var_efc_J_colind_in,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::float32> var_efc_force_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_qfrc_constraint_out)
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
        wp::int32* var_5;
        bool var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::range_t var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        const wp::int32 var_20 = 0;
        wp::int32* var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        const wp::int32 var_24 = 0;
        wp::float32* var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        wp::slice_t var_28;
        const wp::int32 var_29 = 0;
        wp::array_t<wp::float32> var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        //---------
        // forward
        // def update_constraint_init_qfrc_constraint_sparse(                                     <L 1955>
        // worldid, efcid = wp.tid()                                                              <L 1968>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 1970>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 1971>
            continue;
        }
        var_4 = wp::load(var_2);
        // if efcid >= nefc_in[worldid]:                                                          <L 1973>
        var_5 = wp::address(var_nefc_in, var_0);
        var_7 = wp::load(var_5);
        var_6 = (var_1 >= var_7);
        if (var_6) {
            // return                                                                             <L 1974>
            continue;
        }
        // force = efc_force_in[worldid, efcid]                                                   <L 1976>
        var_8 = wp::address(var_efc_force_in, var_0, var_1);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // rownnz = efc_J_rownnz_in[worldid, efcid]                                               <L 1978>
        var_11 = wp::address(var_efc_J_rownnz_in, var_0, var_1);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // rowadr = efc_J_rowadr_in[worldid, efcid]                                               <L 1979>
        var_14 = wp::address(var_efc_J_rowadr_in, var_0, var_1);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // for i in range(rownnz):                                                                <L 1980>
        var_17 = wp::range(var_12);
        start_for_2:;
            if (iter_cmp(var_17) == 0) goto end_for_2;
            var_18 = wp::iter_next(var_17);
            // sparseid = rowadr + i                                                              <L 1981>
            var_19 = wp::add(var_15, var_18);
            // colind = efc_J_colind_in[worldid, 0, sparseid]                                     <L 1982>
            var_21 = wp::address(var_efc_J_colind_in, var_0, var_20, var_19);
            var_23 = wp::load(var_21);
            var_22 = wp::copy(var_23);
            // efc_J = efc_J_in[worldid, 0, sparseid]                                             <L 1983>
            var_25 = wp::address(var_efc_J_in, var_0, var_24, var_19);
            var_27 = wp::load(var_25);
            var_26 = wp::copy(var_27);
            // wp.atomic_add(qfrc_constraint_out[worldid], colind, efc_J * force)                 <L 1984>
            var_28 = wp::slice_t(var_0, var_0, var_29);
            var_30 = wp::view(var_qfrc_constraint_out, var_28);
            var_31 = wp::mul(var_26, var_9);
            var_32 = wp::atomic_add(var_30, var_22, var_31);
            goto start_for_2;
        end_for_2:;
    }
}



extern "C" __global__ void update_constraint_init_qfrc_constraint_dense_94e63668_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::float32> var_efc_force_in,
    wp::int32 var_njmax_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_qfrc_constraint_out)
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
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::range_t var_10;
        wp::int32 var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::float32* var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        //---------
        // forward
        // def update_constraint_init_qfrc_constraint_dense(                                      <L 1988>
        // worldid, dofid = wp.tid()                                                              <L 1999>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 2001>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 2002>
            continue;
        }
        var_4 = wp::load(var_2);
        // sum_qfrc = float(0.0)                                                                  <L 2004>
        var_6 = wp::float(var_5);
        // for efcid in range(min(njmax_in, nefc_in[worldid])):                                   <L 2005>
        var_7 = wp::address(var_nefc_in, var_0);
        var_9 = wp::load(var_7);
        var_8 = wp::min(var_njmax_in, var_9);
        var_10 = wp::range(var_8);
        start_for_1:;
            if (iter_cmp(var_10) == 0) goto end_for_1;
            var_11 = wp::iter_next(var_10);
            // efc_J = efc_J_in[worldid, efcid, dofid]                                            <L 2006>
            var_12 = wp::address(var_efc_J_in, var_0, var_11, var_1);
            var_14 = wp::load(var_12);
            var_13 = wp::copy(var_14);
            // force = efc_force_in[worldid, efcid]                                               <L 2007>
            var_15 = wp::address(var_efc_force_in, var_0, var_11);
            var_17 = wp::load(var_15);
            var_16 = wp::copy(var_17);
            // sum_qfrc += efc_J * force                                                          <L 2008>
            var_18 = wp::mul(var_13, var_16);
            var_19 = wp::add(var_6, var_18);
            wp::assign(var_6, var_19);
            goto start_for_1;
        end_for_1:;
        // qfrc_constraint_out[worldid, dofid] = sum_qfrc                                         <L 2010>
        wp::array_store(var_qfrc_constraint_out, var_0, var_1, var_6);
    }
}



extern "C" __global__ void solve_done_65c723f0_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_opt_tolerance,
    wp::int32 var_opt_iterations,
    wp::array_t<wp::float32> var_stat_meaninertia,
    wp::array_t<wp::float32> var_ctx_grad_dot_in,
    wp::array_t<wp::float32> var_ctx_cost_in,
    wp::array_t<wp::float32> var_ctx_prev_cost_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::int32> var_solver_niter_out,
    wp::array_t<wp::int32> var_nsolving_out,
    wp::array_t<bool> var_ctx_done_out)
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
        const wp::int32 var_4 = 1;
        wp::int32 var_5;
        wp::shape_t* var_6;
        const wp::int32 var_7 = 0;
        wp::int32 var_8;
        wp::shape_t var_9;
        wp::int32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::shape_t* var_14;
        const wp::int32 var_15 = 0;
        wp::int32 var_16;
        wp::shape_t var_17;
        wp::int32 var_18;
        wp::float32* var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32* var_22;
        wp::float32* var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        wp::float32* var_28;
        wp::float32 var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        bool var_32;
        bool var_33;
        bool var_34;
        wp::int32* var_35;
        bool var_36;
        wp::int32 var_37;
        bool var_38;
        const bool var_39 = true;
        const wp::int32 var_40 = 0;
        const wp::int32 var_41 = 1;
        const wp::int32 var_42 = -1;
        wp::int32 var_43;
        //---------
        // forward
        // def solve_done(                                                                        <L 3151>
        // worldid = wp.tid()                                                                     <L 3168>
        var_0 = builtin_tid1d();
        // if ctx_done_in[worldid]:                                                               <L 3170>
        var_1 = wp::address(var_ctx_done_in, var_0);
        var_2 = wp::load(var_1);
        if (var_2) {
            // return                                                                             <L 3171>
            continue;
        }
        var_3 = wp::load(var_1);
        // solver_niter_out[worldid] += 1                                                         <L 3173>
        var_5 = wp::atomic_add(var_solver_niter_out, var_0, var_4);
        // tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]                            <L 3174>
        var_6 = &(var_opt_tolerance.shape);
        var_9 = wp::load(var_6);
        var_8 = wp::extract(var_9, var_7);
        var_10 = wp::mod(var_0, var_8);
        var_11 = wp::address(var_opt_tolerance, var_10);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // meaninertia = stat_meaninertia[worldid % stat_meaninertia.shape[0]]                    <L 3175>
        var_14 = &(var_stat_meaninertia.shape);
        var_17 = wp::load(var_14);
        var_16 = wp::extract(var_17, var_15);
        var_18 = wp::mod(var_0, var_16);
        var_19 = wp::address(var_stat_meaninertia, var_18);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // improvement = _rescale(nv, meaninertia, ctx_prev_cost_in[worldid] - ctx_cost_in[worldid])       <L 3177>
        var_22 = wp::address(var_ctx_prev_cost_in, var_0);
        var_23 = wp::address(var_ctx_cost_in, var_0);
        var_25 = wp::load(var_22);
        var_26 = wp::load(var_23);
        var_24 = wp::sub(var_25, var_26);
        var_27 = _rescale_0(var_nv, var_20, var_24);
        // gradient = _rescale(nv, meaninertia, wp.sqrt(ctx_grad_dot_in[worldid]))                <L 3178>
        var_28 = wp::address(var_ctx_grad_dot_in, var_0);
        var_30 = wp::load(var_28);
        var_29 = wp::sqrt(var_30);
        var_31 = _rescale_0(var_nv, var_20, var_29);
        // done = (improvement < tolerance) or (gradient < tolerance)                             <L 3179>
        var_32 = (var_27 < var_12);
        var_33 = (var_31 < var_12);
        var_34 = var_32 || var_33;
        // if done or solver_niter_out[worldid] == opt_iterations:                                <L 3180>
        var_35 = wp::address(var_solver_niter_out, var_0);
        var_37 = wp::load(var_35);
        var_36 = (var_37 == var_opt_iterations);
        var_38 = var_34 || var_36;
        if (var_38) {
            // ctx_done_out[worldid] = True                                                       <L 3183>
            wp::array_store(var_ctx_done_out, var_0, var_39);
            // wp.atomic_add(nsolving_out, 0, -1)                                                 <L 3184>
            var_43 = wp::atomic_add(var_nsolving_out, var_40, var_42);
        }
    }
}



extern "C" __global__ void padding_h_767c7623_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_nv,
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
        wp::int32 var_1;
        bool* var_2;
        bool var_3;
        bool var_4;
        wp::int32 var_5;
        const wp::float32 var_6 = 1.0;
        //---------
        // forward
        // def padding_h(nv: int, ctx_done_in: wp.array[bool], ctx_h_out: wp.array3d[float]):       <L 2789>
        // worldid, elementid = wp.tid()                                                          <L 2790>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 2792>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 2793>
            continue;
        }
        var_4 = wp::load(var_2);
        // dofid = nv + elementid                                                                 <L 2795>
        var_5 = wp::add(var_nv, var_1);
        // ctx_h_out[worldid, dofid, dofid] = 1.0                                                 <L 2796>
        wp::array_store(var_ctx_h_out, var_0, var_5, var_5, var_6);
    }
}



extern "C" __global__ void _JTDAJ_sparse_74a213ab_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::int32> var_efc_J_rownnz_in,
    wp::array_t<wp::int32> var_efc_J_rowadr_in,
    wp::array_t<wp::int32> var_efc_J_colind_in,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::int32> var_efc_state_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_h_out)
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
        wp::int32* var_5;
        bool var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::float32 var_14;
        const wp::float32 var_15 = 0.0;
        bool var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::int32* var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        wp::range_t var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        const wp::int32 var_26 = 0;
        wp::float32* var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        const wp::int32 var_30 = 0;
        wp::int32* var_31;
        wp::int32 var_32;
        wp::int32 var_33;
        wp::range_t var_34;
        wp::int32 var_35;
        bool var_36;
        wp::int32 var_37;
        wp::float32 var_38;
        wp::int32 var_39;
        wp::int32 var_40;
        const wp::int32 var_41 = 0;
        wp::float32* var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        const wp::int32 var_45 = 0;
        wp::int32* var_46;
        wp::int32 var_47;
        wp::int32 var_48;
        wp::int32 var_49;
        wp::float32 var_50;
        wp::int32 var_51;
        wp::float32 var_52;
        wp::float32 var_53;
        wp::int32 var_54;
        wp::int32 var_55;
        wp::slice_t var_56;
        const wp::int32 var_57 = 0;
        wp::slice_t var_58;
        const wp::int32 var_59 = 0;
        wp::array_t<wp::float32> var_60;
        wp::float32 var_61;
        //---------
        // forward
        // def _JTDAJ_sparse(                                                                     <L 2827>
        // worldid, efcid = wp.tid()                                                              <L 2841>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 2843>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 2844>
            continue;
        }
        var_4 = wp::load(var_2);
        // if efcid >= nefc_in[worldid]:                                                          <L 2846>
        var_5 = wp::address(var_nefc_in, var_0);
        var_7 = wp::load(var_5);
        var_6 = (var_1 >= var_7);
        if (var_6) {
            // return                                                                             <L 2847>
            continue;
        }
        // efc_D = efc_D_in[worldid, efcid]                                                       <L 2849>
        var_8 = wp::address(var_efc_D_in, var_0, var_1);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // efc_state = efc_state_in[worldid, efcid]                                               <L 2850>
        var_11 = wp::address(var_efc_state_in, var_0, var_1);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // if state_check(efc_D, efc_state) == 0.0:                                               <L 2852>
        var_14 = state_check_0(var_9, var_12);
        var_16 = (var_14 == var_15);
        if (var_16) {
            // return                                                                             <L 2853>
            continue;
        }
        // rownnz = efc_J_rownnz_in[worldid, efcid]                                               <L 2855>
        var_17 = wp::address(var_efc_J_rownnz_in, var_0, var_1);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // rowadr = efc_J_rowadr_in[worldid, efcid]                                               <L 2856>
        var_20 = wp::address(var_efc_J_rowadr_in, var_0, var_1);
        var_22 = wp::load(var_20);
        var_21 = wp::copy(var_22);
        // for i in range(rownnz):                                                                <L 2858>
        var_23 = wp::range(var_18);
        start_for_3:;
            if (iter_cmp(var_23) == 0) goto end_for_3;
            var_24 = wp::iter_next(var_23);
            // sparseidi = rowadr + i                                                             <L 2859>
            var_25 = wp::add(var_21, var_24);
            // Ji = efc_J_in[worldid, 0, sparseidi]                                               <L 2860>
            var_27 = wp::address(var_efc_J_in, var_0, var_26, var_25);
            var_29 = wp::load(var_27);
            var_28 = wp::copy(var_29);
            // colindi = efc_J_colind_in[worldid, 0, sparseidi]                                   <L 2861>
            var_31 = wp::address(var_efc_J_colind_in, var_0, var_30, var_25);
            var_33 = wp::load(var_31);
            var_32 = wp::copy(var_33);
            // for j in range(i, rownnz):                                                         <L 2862>
            var_34 = wp::range(var_24, var_18);
            start_for_5:;
                if (iter_cmp(var_34) == 0) goto end_for_5;
                var_35 = wp::iter_next(var_34);
                // if j == i:                                                                     <L 2863>
                var_36 = (var_35 == var_24);
                if (var_36) {
                    // sparseidj = sparseidi                                                      <L 2864>
                    var_37 = wp::copy(var_25);
                    // Jj = Ji                                                                    <L 2865>
                    var_38 = wp::copy(var_28);
                    // colindj = colindi                                                          <L 2866>
                    var_39 = wp::copy(var_32);
                }
                if (!var_36) {
                    // sparseidj = rowadr + j                                                     <L 2868>
                    var_40 = wp::add(var_21, var_35);
                    // Jj = efc_J_in[worldid, 0, sparseidj]                                       <L 2869>
                    var_42 = wp::address(var_efc_J_in, var_0, var_41, var_40);
                    var_44 = wp::load(var_42);
                    var_43 = wp::copy(var_44);
                    // colindj = efc_J_colind_in[worldid, 0, sparseidj]                           <L 2870>
                    var_46 = wp::address(var_efc_J_colind_in, var_0, var_45, var_40);
                    var_48 = wp::load(var_46);
                    var_47 = wp::copy(var_48);
                }
                var_49 = wp::where(var_36, var_37, var_40);
                var_50 = wp::where(var_36, var_38, var_43);
                var_51 = wp::where(var_36, var_39, var_47);
                // h = Ji * Jj * efc_D                                                            <L 2872>
                var_52 = wp::mul(var_28, var_50);
                var_53 = wp::mul(var_52, var_9);
                // row = wp.max(colindi, colindj)                                                 <L 2874>
                var_54 = wp::max(var_32, var_51);
                // col = wp.min(colindi, colindj)                                                 <L 2875>
                var_55 = wp::min(var_32, var_51);
                // wp.atomic_add(h_out[worldid, row], col, h)                                     <L 2876>
                var_56 = wp::slice_t(var_0, var_0, var_57);
                var_58 = wp::slice_t(var_54, var_54, var_59);
                var_60 = wp::view(var_h_out, var_56, var_58);
                var_61 = wp::atomic_add(var_60, var_55, var_53);
                goto start_for_5;
            end_for_5:;
            goto start_for_3;
        end_for_3:;
    }
}



extern "C" __global__ void update_gradient_zero_grad_dot_a7f57027_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_ctx_grad_dot_out)
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
        const wp::float32 var_4 = 0.0;
        //---------
        // forward
        // def update_gradient_zero_grad_dot(                                                     <L 2223>
        // worldid = wp.tid()                                                                     <L 2229>
        var_0 = builtin_tid1d();
        // if ctx_done_in[worldid]:                                                               <L 2231>
        var_1 = wp::address(var_ctx_done_in, var_0);
        var_2 = wp::load(var_1);
        if (var_2) {
            // return                                                                             <L 2232>
            continue;
        }
        var_3 = wp::load(var_1);
        // ctx_grad_dot_out[worldid] = 0.0                                                        <L 2234>
        wp::array_store(var_ctx_grad_dot_out, var_0, var_4);
    }
}



extern "C" __global__ void linesearch_zero_jv_3d6e7eb5_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_nefc_in,
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
        wp::int32* var_2;
        bool var_3;
        wp::int32 var_4;
        bool* var_5;
        bool var_6;
        bool var_7;
        const wp::float32 var_8 = 0.0;
        //---------
        // forward
        // def linesearch_zero_jv(                                                                <L 1394>
        // worldid, efcid = wp.tid()                                                              <L 1402>
        builtin_tid2d(var_0, var_1);
        // if efcid >= nefc_in[worldid]:                                                          <L 1404>
        var_2 = wp::address(var_nefc_in, var_0);
        var_4 = wp::load(var_2);
        var_3 = (var_1 >= var_4);
        if (var_3) {
            // return                                                                             <L 1405>
            continue;
        }
        // if ctx_done_in[worldid]:                                                               <L 1407>
        var_5 = wp::address(var_ctx_done_in, var_0);
        var_6 = wp::load(var_5);
        if (var_6) {
            // return                                                                             <L 1408>
            continue;
        }
        var_7 = wp::load(var_5);
        // ctx_jv_out[worldid, efcid] = 0.0                                                       <L 1410>
        wp::array_store(var_ctx_jv_out, var_0, var_1, var_8);
    }
}



extern "C" __global__ void solve_prev_grad_Mgrad_ac349060_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_ctx_grad_in,
    wp::array_t<wp::float32> var_ctx_Mgrad_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_ctx_prev_grad_out,
    wp::array_t<wp::float32> var_ctx_prev_Mgrad_out)
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
        wp::float32* var_5;
        wp::float32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        //---------
        // forward
        // def solve_prev_grad_Mgrad(                                                             <L 3062>
        // worldid, dofid = wp.tid()                                                              <L 3071>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 3073>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 3074>
            continue;
        }
        var_4 = wp::load(var_2);
        // ctx_prev_grad_out[worldid, dofid] = ctx_grad_in[worldid, dofid]                        <L 3076>
        var_5 = wp::address(var_ctx_grad_in, var_0, var_1);
        var_6 = wp::load(var_5);
        wp::array_store(var_ctx_prev_grad_out, var_0, var_1, var_6);
        // ctx_prev_Mgrad_out[worldid, dofid] = ctx_Mgrad_in[worldid, dofid]                      <L 3077>
        var_7 = wp::address(var_ctx_Mgrad_in, var_0, var_1);
        var_8 = wp::load(var_7);
        wp::array_store(var_ctx_prev_Mgrad_out, var_0, var_1, var_8);
    }
}



extern "C" __global__ void solve_beta_d9b576c4_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_ctx_grad_in,
    wp::array_t<wp::float32> var_ctx_Mgrad_in,
    wp::array_t<wp::float32> var_ctx_prev_grad_in,
    wp::array_t<wp::float32> var_ctx_prev_Mgrad_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_ctx_beta_out)
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
        const wp::float32 var_4 = 0.0;
        wp::float32 var_5;
        const wp::float32 var_6 = 0.0;
        wp::float32 var_7;
        wp::range_t var_8;
        wp::int32 var_9;
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        wp::float32* var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32* var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        const wp::float32 var_24 = 0.0;
        const wp::float32 var_25 = 1e-15;
        const wp::float32 var_26 = 1e-15;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        //---------
        // forward
        // def solve_beta(                                                                        <L 3081>
        // worldid = wp.tid()                                                                     <L 3093>
        var_0 = builtin_tid1d();
        // if ctx_done_in[worldid]:                                                               <L 3095>
        var_1 = wp::address(var_ctx_done_in, var_0);
        var_2 = wp::load(var_1);
        if (var_2) {
            // return                                                                             <L 3096>
            continue;
        }
        var_3 = wp::load(var_1);
        // beta_num = float(0.0)                                                                  <L 3098>
        var_5 = wp::float(var_4);
        // beta_den = float(0.0)                                                                  <L 3099>
        var_7 = wp::float(var_6);
        // for dofid in range(nv):                                                                <L 3100>
        var_8 = wp::range(var_nv);
        start_for_1:;
            if (iter_cmp(var_8) == 0) goto end_for_1;
            var_9 = wp::iter_next(var_8);
            // prev_Mgrad = ctx_prev_Mgrad_in[worldid][dofid]                                     <L 3101>
            var_10 = wp::address(var_ctx_prev_Mgrad_in, var_0, var_9);
            var_12 = wp::load(var_10);
            var_11 = wp::copy(var_12);
            // beta_num += ctx_grad_in[worldid, dofid] * (ctx_Mgrad_in[worldid, dofid] - prev_Mgrad)       <L 3102>
            var_13 = wp::address(var_ctx_grad_in, var_0, var_9);
            var_14 = wp::address(var_ctx_Mgrad_in, var_0, var_9);
            var_16 = wp::load(var_14);
            var_15 = wp::sub(var_16, var_11);
            var_18 = wp::load(var_13);
            var_17 = wp::mul(var_18, var_15);
            var_19 = wp::add(var_5, var_17);
            // beta_den += ctx_prev_grad_in[worldid, dofid] * prev_Mgrad                          <L 3103>
            var_20 = wp::address(var_ctx_prev_grad_in, var_0, var_9);
            var_22 = wp::load(var_20);
            var_21 = wp::mul(var_22, var_11);
            var_23 = wp::add(var_7, var_21);
            wp::assign(var_5, var_19);
            wp::assign(var_7, var_23);
            goto start_for_1;
        end_for_1:;
        // ctx_beta_out[worldid] = wp.max(0.0, beta_num / wp.max(types.MJ_MINVAL, beta_den))       <L 3105>
        var_27 = wp::max(var_26, var_7);
        var_28 = wp::div(var_5, var_27);
        var_29 = wp::max(var_24, var_28);
        wp::array_store(var_ctx_beta_out, var_0, var_29);
    }
}



extern "C" __global__ void linesearch_prepare_quad_d4d10bb8_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_opt_impratio_invsqrt,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::int32> var_efc_type_in,
    wp::array_t<wp::int32> var_efc_id_in,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::float32> var_ctx_Jaref_in,
    wp::array_t<wp::float32> var_ctx_jv_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ctx_quad_out)
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
        wp::int32* var_2;
        bool var_3;
        wp::int32 var_4;
        bool* var_5;
        bool var_6;
        bool var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        const wp::float32 var_17 = 0.5;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        const wp::float32 var_23 = 0.5;
        wp::float32 var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::int32* var_28;
        const wp::int32 var_29 = 7;
        bool var_30;
        wp::int32 var_31;
        wp::int32* var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        const wp::int32 var_35 = 0;
        wp::int32* var_36;
        bool var_37;
        wp::int32 var_38;
        const wp::int32 var_39 = 0;
        wp::int32* var_40;
        wp::int32 var_41;
        wp::int32 var_42;
        bool var_43;
        wp::int32* var_44;
        wp::int32 var_45;
        wp::int32 var_46;
        wp::vec_t<5, wp::float32>* var_47;
        wp::vec_t<5, wp::float32> var_48;
        wp::vec_t<5, wp::float32> var_49;
        const wp::int32 var_50 = 0;
        wp::float32 var_51;
        wp::shape_t* var_52;
        const wp::int32 var_53 = 0;
        wp::int32 var_54;
        wp::shape_t var_55;
        wp::int32 var_56;
        wp::float32* var_57;
        wp::float32 var_58;
        wp::float32 var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        const wp::float32 var_62 = 0.0;
        wp::float32 var_63;
        const wp::float32 var_64 = 0.0;
        wp::float32 var_65;
        const wp::float32 var_66 = 0.0;
        wp::float32 var_67;
        const wp::int32 var_68 = 1;
        wp::range_t var_69;
        wp::int32 var_70;
        wp::int32* var_71;
        wp::int32 var_72;
        wp::int32 var_73;
        const wp::int32 var_74 = 0;
        bool var_75;
        wp::float32* var_76;
        wp::float32 var_77;
        wp::float32 var_78;
        wp::float32* var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        wp::float32* var_82;
        wp::float32 var_83;
        wp::float32 var_84;
        wp::float32 var_85;
        const wp::float32 var_86 = 0.5;
        wp::float32 var_87;
        wp::float32 var_88;
        wp::float32 var_89;
        const wp::float32 var_90 = 0.5;
        wp::float32 var_91;
        wp::float32 var_92;
        wp::float32 var_93;
        wp::vec_t<3, wp::float32> var_94;
        wp::vec_t<3, wp::float32> var_95;
        const wp::int32 var_96 = 1;
        wp::int32 var_97;
        wp::float32 var_98;
        wp::float32 var_99;
        wp::float32 var_100;
        wp::float32 var_101;
        wp::float32 var_102;
        wp::float32 var_103;
        wp::float32 var_104;
        wp::float32 var_105;
        wp::float32 var_106;
        wp::vec_t<3, wp::float32> var_107;
        const wp::int32 var_108 = 1;
        wp::int32* var_109;
        wp::int32 var_110;
        wp::int32 var_111;
        wp::float32 var_112;
        const wp::float32 var_113 = 1.0;
        wp::float32 var_114;
        wp::float32 var_115;
        wp::float32 var_116;
        wp::vec_t<3, wp::float32> var_117;
        const wp::int32 var_118 = 2;
        wp::int32* var_119;
        wp::int32 var_120;
        wp::int32 var_121;
        //---------
        // forward
        // def linesearch_prepare_quad(                                                           <L 1524>
        // worldid, efcid = wp.tid()                                                              <L 1543>
        builtin_tid2d(var_0, var_1);
        // if efcid >= nefc_in[worldid]:                                                          <L 1545>
        var_2 = wp::address(var_nefc_in, var_0);
        var_4 = wp::load(var_2);
        var_3 = (var_1 >= var_4);
        if (var_3) {
            // return                                                                             <L 1546>
            continue;
        }
        // if ctx_done_in[worldid]:                                                               <L 1548>
        var_5 = wp::address(var_ctx_done_in, var_0);
        var_6 = wp::load(var_5);
        if (var_6) {
            // return                                                                             <L 1549>
            continue;
        }
        var_7 = wp::load(var_5);
        // Jaref = ctx_Jaref_in[worldid, efcid]                                                   <L 1551>
        var_8 = wp::address(var_ctx_Jaref_in, var_0, var_1);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // jv = ctx_jv_in[worldid, efcid]                                                         <L 1552>
        var_11 = wp::address(var_ctx_jv_in, var_0, var_1);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // efc_D = efc_D_in[worldid, efcid]                                                       <L 1553>
        var_14 = wp::address(var_efc_D_in, var_0, var_1);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // quad = wp.vec3(0.5 * Jaref * Jaref * efc_D, jv * Jaref * efc_D, 0.5 * jv * jv * efc_D)       <L 1556>
        var_18 = wp::mul(var_17, var_9);
        var_19 = wp::mul(var_18, var_9);
        var_20 = wp::mul(var_19, var_15);
        var_21 = wp::mul(var_12, var_9);
        var_22 = wp::mul(var_21, var_15);
        var_24 = wp::mul(var_23, var_12);
        var_25 = wp::mul(var_24, var_12);
        var_26 = wp::mul(var_25, var_15);
        var_27 = wp::vec_t<3, wp::float32>(var_20, var_22, var_26);
        // if efc_type_in[worldid, efcid] == types.ConstraintType.CONTACT_ELLIPTIC:               <L 1559>
        var_28 = wp::address(var_efc_type_in, var_0, var_1);
        var_31 = wp::load(var_28);
        var_30 = (var_31 == var_29);
        if (var_30) {
            // conid = efc_id_in[worldid, efcid]                                                  <L 1561>
            var_32 = wp::address(var_efc_id_in, var_0, var_1);
            var_34 = wp::load(var_32);
            var_33 = wp::copy(var_34);
            // if conid >= nacon_in[0]:                                                           <L 1563>
            var_36 = wp::address(var_nacon_in, var_35);
            var_38 = wp::load(var_36);
            var_37 = (var_33 >= var_38);
            if (var_37) {
                // return                                                                         <L 1564>
                continue;
            }
            // efcid0 = contact_efc_address_in[conid, 0]                                          <L 1566>
            var_40 = wp::address(var_contact_efc_address_in, var_33, var_39);
            var_42 = wp::load(var_40);
            var_41 = wp::copy(var_42);
            // if efcid != efcid0:                                                                <L 1568>
            var_43 = (var_1 != var_41);
            if (var_43) {
                // return                                                                         <L 1569>
                continue;
            }
            // dim = contact_dim_in[conid]                                                        <L 1571>
            var_44 = wp::address(var_contact_dim_in, var_33);
            var_46 = wp::load(var_44);
            var_45 = wp::copy(var_46);
            // friction = contact_friction_in[conid]                                              <L 1572>
            var_47 = wp::address(var_contact_friction_in, var_33);
            var_49 = wp::load(var_47);
            var_48 = wp::copy(var_49);
            // mu = friction[0] * opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]       <L 1573>
            var_51 = wp::extract(var_48, var_50);
            var_52 = &(var_opt_impratio_invsqrt.shape);
            var_55 = wp::load(var_52);
            var_54 = wp::extract(var_55, var_53);
            var_56 = wp::mod(var_0, var_54);
            var_57 = wp::address(var_opt_impratio_invsqrt, var_56);
            var_59 = wp::load(var_57);
            var_58 = wp::mul(var_51, var_59);
            // u0 = Jaref * mu                                                                    <L 1575>
            var_60 = wp::mul(var_9, var_58);
            // v0 = jv * mu                                                                       <L 1576>
            var_61 = wp::mul(var_12, var_58);
            // uu = float(0.0)                                                                    <L 1578>
            var_63 = wp::float(var_62);
            // uv = float(0.0)                                                                    <L 1579>
            var_65 = wp::float(var_64);
            // vv = float(0.0)                                                                    <L 1580>
            var_67 = wp::float(var_66);
            // for j in range(1, dim):                                                            <L 1581>
            var_69 = wp::range(var_68, var_45);
            start_for_4:;
                if (iter_cmp(var_69) == 0) goto end_for_4;
                var_70 = wp::iter_next(var_69);
                // efcidj = contact_efc_address_in[conid, j]                                      <L 1583>
                var_71 = wp::address(var_contact_efc_address_in, var_33, var_70);
                var_73 = wp::load(var_71);
                var_72 = wp::copy(var_73);
                // if efcidj < 0:                                                                 <L 1584>
                var_75 = (var_72 < var_74);
                if (var_75) {
                    // return                                                                     <L 1585>
                    continue;
                }
                // jvj = ctx_jv_in[worldid, efcidj]                                               <L 1586>
                var_76 = wp::address(var_ctx_jv_in, var_0, var_72);
                var_78 = wp::load(var_76);
                var_77 = wp::copy(var_78);
                // jarefj = ctx_Jaref_in[worldid, efcidj]                                         <L 1587>
                var_79 = wp::address(var_ctx_Jaref_in, var_0, var_72);
                var_81 = wp::load(var_79);
                var_80 = wp::copy(var_81);
                // dj = efc_D_in[worldid, efcidj]                                                 <L 1588>
                var_82 = wp::address(var_efc_D_in, var_0, var_72);
                var_84 = wp::load(var_82);
                var_83 = wp::copy(var_84);
                // DJj = dj * jarefj                                                              <L 1589>
                var_85 = wp::mul(var_83, var_80);
                // quad += wp.vec3(                                                               <L 1591>
                // 0.5 * jarefj * DJj,                                                            <L 1592>
                var_87 = wp::mul(var_86, var_80);
                var_88 = wp::mul(var_87, var_85);
                // jvj * DJj,                                                                     <L 1593>
                var_89 = wp::mul(var_77, var_85);
                // 0.5 * jvj * dj * jvj,                                                          <L 1594>
                var_91 = wp::mul(var_90, var_77);
                var_92 = wp::mul(var_91, var_83);
                var_93 = wp::mul(var_92, var_77);
                var_94 = wp::vec_t<3, wp::float32>(var_88, var_89, var_93);
                // quad += wp.vec3(                                                               <L 1591>
                var_95 = wp::add(var_27, var_94);
                // frictionj = friction[j - 1]                                                    <L 1598>
                var_97 = wp::sub(var_70, var_96);
                var_98 = wp::extract(var_48, var_97);
                // uj = jarefj * frictionj                                                        <L 1599>
                var_99 = wp::mul(var_80, var_98);
                // vj = jvj * frictionj                                                           <L 1600>
                var_100 = wp::mul(var_77, var_98);
                // uu += uj * uj                                                                  <L 1603>
                var_101 = wp::mul(var_99, var_99);
                var_102 = wp::add(var_63, var_101);
                // uv += uj * vj                                                                  <L 1604>
                var_103 = wp::mul(var_99, var_100);
                var_104 = wp::add(var_65, var_103);
                // vv += vj * vj                                                                  <L 1605>
                var_105 = wp::mul(var_100, var_100);
                var_106 = wp::add(var_67, var_105);
                wp::assign(var_27, var_95);
                wp::assign(var_63, var_102);
                wp::assign(var_65, var_104);
                wp::assign(var_67, var_106);
                goto start_for_4;
            end_for_4:;
            // quad1 = wp.vec3(u0, v0, uu)                                                        <L 1607>
            var_107 = wp::vec_t<3, wp::float32>(var_60, var_61, var_63);
            // efcid1 = contact_efc_address_in[conid, 1]                                          <L 1608>
            var_109 = wp::address(var_contact_efc_address_in, var_33, var_108);
            var_111 = wp::load(var_109);
            var_110 = wp::copy(var_111);
            // ctx_quad_out[worldid, efcid1] = quad1                                              <L 1609>
            wp::array_store(var_ctx_quad_out, var_0, var_110, var_107);
            // mu2 = mu * mu                                                                      <L 1611>
            var_112 = wp::mul(var_58, var_58);
            // quad2 = wp.vec3(uv, vv, efc_D / (mu2 * (1.0 + mu2)))                               <L 1612>
            var_114 = wp::add(var_113, var_112);
            var_115 = wp::mul(var_112, var_114);
            var_116 = wp::div(var_15, var_115);
            var_117 = wp::vec_t<3, wp::float32>(var_65, var_67, var_116);
            // efcid2 = contact_efc_address_in[conid, 2]                                          <L 1613>
            var_119 = wp::address(var_contact_efc_address_in, var_33, var_118);
            var_121 = wp::load(var_119);
            var_120 = wp::copy(var_121);
            // ctx_quad_out[worldid, efcid2] = quad2                                              <L 1614>
            wp::array_store(var_ctx_quad_out, var_0, var_120, var_117);
        }
        // ctx_quad_out[worldid, efcid] = quad                                                    <L 1616>
        wp::array_store(var_ctx_quad_out, var_0, var_1, var_27);
    }
}



extern "C" __global__ void solve_search_update_ae582eee_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_opt_solver,
    wp::array_t<wp::float32> var_ctx_Mgrad_in,
    wp::array_t<wp::float32> var_ctx_search_in,
    wp::array_t<wp::float32> var_ctx_beta_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_ctx_search_out,
    wp::array_t<wp::float32> var_ctx_search_dot_out)
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
        const wp::float32 var_5 = 1.0;
        const wp::float32 var_6 = -1.0;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        const wp::int32 var_10 = 1;
        bool var_11;
        wp::float32* var_12;
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        //---------
        // forward
        // def solve_search_update(                                                               <L 3124>
        // worldid, dofid = wp.tid()                                                              <L 3136>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 3138>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 3139>
            continue;
        }
        var_4 = wp::load(var_2);
        // search = -1.0 * ctx_Mgrad_in[worldid, dofid]                                           <L 3141>
        var_7 = wp::address(var_ctx_Mgrad_in, var_0, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::mul(var_6, var_9);
        // if opt_solver == types.SolverType.CG:                                                  <L 3143>
        var_11 = (var_opt_solver == var_10);
        if (var_11) {
            // search += ctx_beta_in[worldid] * ctx_search_in[worldid, dofid]                     <L 3144>
            var_12 = wp::address(var_ctx_beta_in, var_0);
            var_13 = wp::address(var_ctx_search_in, var_0, var_1);
            var_15 = wp::load(var_12);
            var_16 = wp::load(var_13);
            var_14 = wp::mul(var_15, var_16);
            var_17 = wp::add(var_8, var_14);
        }
        var_18 = wp::where(var_11, var_17, var_8);
        // ctx_search_out[worldid, dofid] = search                                                <L 3146>
        wp::array_store(var_ctx_search_out, var_0, var_1, var_18);
        // wp.atomic_add(ctx_search_dot_out, worldid, search * search)                            <L 3147>
        var_19 = wp::mul(var_18, var_18);
        var_20 = wp::atomic_add(var_ctx_search_dot_out, var_0, var_19);
    }
}



extern "C" __global__ void linesearch_parallel_fused_00b4f775_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_opt_ls_iterations,
    wp::array_t<wp::float32> var_opt_impratio_invsqrt,
    wp::float32 var_opt_ls_parallel_min_step,
    wp::array_t<wp::int32> var_ne_in,
    wp::array_t<wp::int32> var_nf_in,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::int32> var_efc_type_in,
    wp::array_t<wp::int32> var_efc_id_in,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::float32> var_efc_frictionloss_in,
    wp::int32 var_njmax_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::float32> var_ctx_Jaref_in,
    wp::array_t<wp::float32> var_ctx_jv_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ctx_quad_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ctx_quad_gauss_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_cost_out)
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
        const wp::float32 var_5 = 1.0;
        wp::float32 var_6;
        wp::vec_t<3, wp::float32>* var_7;
        wp::float32 var_8;
        wp::vec_t<3, wp::float32> var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::range_t var_19;
        wp::int32 var_20;
        bool var_21;
        wp::vec_t<3, wp::float32>* var_22;
        wp::float32 var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        wp::int32 var_27;
        bool var_28;
        wp::float32* var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        wp::float32* var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::float32* var_37;
        wp::float32 var_38;
        wp::float32 var_39;
        wp::float32* var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        bool var_44;
        bool var_45;
        bool var_46;
        wp::vec_t<3, wp::float32>* var_47;
        wp::vec_t<3, wp::float32> var_48;
        wp::vec_t<3, wp::float32> var_49;
        wp::float32 var_50;
        bool var_51;
        const wp::float32 var_52 = 0.5;
        const wp::float32 var_53 = -0.5;
        wp::float32 var_54;
        wp::float32 var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        wp::float32 var_58;
        const wp::float32 var_59 = 0.0;
        wp::vec_t<3, wp::float32> var_60;
        wp::vec_t<3, wp::float32> var_61;
        const wp::float32 var_62 = 0.5;
        const wp::float32 var_63 = -0.5;
        wp::float32 var_64;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        const wp::float32 var_68 = 0.0;
        wp::vec_t<3, wp::float32> var_69;
        wp::vec_t<3, wp::float32> var_70;
        wp::vec_t<3, wp::float32> var_71;
        wp::float32 var_72;
        wp::float32 var_73;
        wp::float32 var_74;
        wp::int32* var_75;
        const wp::int32 var_76 = 7;
        bool var_77;
        wp::int32 var_78;
        wp::int32* var_79;
        wp::int32 var_80;
        wp::int32 var_81;
        const wp::int32 var_82 = 0;
        wp::int32* var_83;
        bool var_84;
        wp::int32 var_85;
        wp::float32 var_86;
        const wp::int32 var_87 = 0;
        wp::int32* var_88;
        wp::int32 var_89;
        wp::int32 var_90;
        bool var_91;
        wp::float32 var_92;
        wp::vec_t<5, wp::float32>* var_93;
        wp::vec_t<5, wp::float32> var_94;
        wp::vec_t<5, wp::float32> var_95;
        const wp::int32 var_96 = 0;
        wp::float32 var_97;
        wp::shape_t* var_98;
        const wp::int32 var_99 = 0;
        wp::int32 var_100;
        wp::shape_t var_101;
        wp::int32 var_102;
        wp::float32* var_103;
        wp::float32 var_104;
        wp::float32 var_105;
        const wp::int32 var_106 = 1;
        wp::int32* var_107;
        wp::int32 var_108;
        wp::int32 var_109;
        const wp::int32 var_110 = 2;
        wp::int32* var_111;
        wp::int32 var_112;
        wp::int32 var_113;
        wp::vec_t<3, wp::float32>* var_114;
        const wp::int32 var_115 = 0;
        wp::float32 var_116;
        wp::vec_t<3, wp::float32> var_117;
        wp::vec_t<3, wp::float32>* var_118;
        const wp::int32 var_119 = 1;
        wp::float32 var_120;
        wp::vec_t<3, wp::float32> var_121;
        wp::vec_t<3, wp::float32>* var_122;
        const wp::int32 var_123 = 2;
        wp::float32 var_124;
        wp::vec_t<3, wp::float32> var_125;
        wp::vec_t<3, wp::float32>* var_126;
        const wp::int32 var_127 = 0;
        wp::float32 var_128;
        wp::vec_t<3, wp::float32> var_129;
        wp::vec_t<3, wp::float32>* var_130;
        const wp::int32 var_131 = 1;
        wp::float32 var_132;
        wp::vec_t<3, wp::float32> var_133;
        wp::vec_t<3, wp::float32>* var_134;
        const wp::int32 var_135 = 2;
        wp::float32 var_136;
        wp::vec_t<3, wp::float32> var_137;
        wp::float32 var_138;
        wp::float32 var_139;
        const wp::float32 var_140 = 2.0;
        wp::float32 var_141;
        wp::float32 var_142;
        wp::float32 var_143;
        wp::float32 var_144;
        wp::float32 var_145;
        const wp::float32 var_146 = 0.0;
        bool var_147;
        const wp::float32 var_148 = 0.0;
        bool var_149;
        wp::vec_t<3, wp::float32>* var_150;
        wp::float32 var_151;
        wp::vec_t<3, wp::float32> var_152;
        wp::float32 var_153;
        wp::float32 var_154;
        wp::float32 var_155;
        wp::float32 var_156;
        wp::float32 var_157;
        bool var_158;
        wp::float32 var_159;
        wp::float32 var_160;
        const wp::float32 var_161 = 0.0;
        bool var_162;
        wp::vec_t<3, wp::float32>* var_163;
        wp::float32 var_164;
        wp::vec_t<3, wp::float32> var_165;
        wp::float32 var_166;
        wp::float32 var_167;
        const wp::float32 var_168 = 0.5;
        wp::float32 var_169;
        wp::float32 var_170;
        wp::float32 var_171;
        wp::float32 var_172;
        wp::float32 var_173;
        wp::float32 var_174;
        wp::float32 var_175;
        wp::float32 var_176;
        wp::float32 var_177;
        wp::float32 var_178;
        wp::float32 var_179;
        wp::float32 var_180;
        wp::float32* var_181;
        wp::float32* var_182;
        wp::float32 var_183;
        wp::float32 var_184;
        wp::float32 var_185;
        wp::float32 var_186;
        const wp::float32 var_187 = 0.0;
        bool var_188;
        wp::vec_t<3, wp::float32>* var_189;
        wp::float32 var_190;
        wp::vec_t<3, wp::float32> var_191;
        wp::float32 var_192;
        wp::float32 var_193;
        wp::float32 var_194;
        wp::float32 var_195;
        wp::float32 var_196;
        wp::float32 var_197;
        wp::float32 var_198;
        //---------
        // forward
        // def linesearch_parallel_fused(                                                         <L 331>
        // worldid, alphaid = wp.tid()                                                            <L 357>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 359>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 360>
            continue;
        }
        var_4 = wp::load(var_2);
        // alpha = _log_scale(opt_ls_parallel_min_step, 1.0, opt_ls_iterations, alphaid)          <L 362>
        var_6 = _log_scale_0(var_opt_ls_parallel_min_step, var_5, var_opt_ls_iterations, var_1);
        // out = _eval_cost(ctx_quad_gauss_in[worldid], alpha)                                    <L 364>
        var_7 = wp::address(var_ctx_quad_gauss_in, var_0);
        var_9 = wp::load(var_7);
        var_8 = _eval_cost_0(var_9, var_6);
        // ne = ne_in[worldid]                                                                    <L 366>
        var_10 = wp::address(var_ne_in, var_0);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // nf = nf_in[worldid]                                                                    <L 367>
        var_13 = wp::address(var_nf_in, var_0);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // for efcid in range(min(njmax_in, nefc_in[worldid])):                                   <L 370>
        var_16 = wp::address(var_nefc_in, var_0);
        var_18 = wp::load(var_16);
        var_17 = wp::min(var_njmax_in, var_18);
        var_19 = wp::range(var_17);
        start_for_1:;
            if (iter_cmp(var_19) == 0) goto end_for_1;
            var_20 = wp::iter_next(var_19);
            // if efcid < ne:                                                                     <L 372>
            var_21 = (var_20 < var_11);
            if (var_21) {
                // out += _eval_cost(ctx_quad_in[worldid, efcid], alpha)                          <L 373>
                var_22 = wp::address(var_ctx_quad_in, var_0, var_20);
                var_24 = wp::load(var_22);
                var_23 = _eval_cost_0(var_24, var_6);
                var_25 = wp::add(var_8, var_23);
            }
            var_26 = wp::where(var_21, var_25, var_8);
            if (!var_21) {
                // elif efcid < ne + nf:                                                          <L 375>
                var_27 = wp::add(var_11, var_14);
                var_28 = (var_20 < var_27);
                if (var_28) {
                    // start = ctx_Jaref_in[worldid, efcid]                                       <L 377>
                    var_29 = wp::address(var_ctx_Jaref_in, var_0, var_20);
                    var_31 = wp::load(var_29);
                    var_30 = wp::copy(var_31);
                    // dir = ctx_jv_in[worldid, efcid]                                            <L 378>
                    var_32 = wp::address(var_ctx_jv_in, var_0, var_20);
                    var_34 = wp::load(var_32);
                    var_33 = wp::copy(var_34);
                    // x = start + alpha * dir                                                    <L 379>
                    var_35 = wp::mul(var_6, var_33);
                    var_36 = wp::add(var_30, var_35);
                    // f = efc_frictionloss_in[worldid, efcid]                                    <L 380>
                    var_37 = wp::address(var_efc_frictionloss_in, var_0, var_20);
                    var_39 = wp::load(var_37);
                    var_38 = wp::copy(var_39);
                    // rf = math.safe_div(f, efc_D_in[worldid, efcid])                            <L 381>
                    var_40 = wp::address(var_efc_D_in, var_0, var_20);
                    var_42 = wp::load(var_40);
                    var_41 = safe_div_0(var_38, var_42);
                    // if (-rf < x) and (x < rf):                                                 <L 384>
                    var_43 = wp::neg(var_41);
                    var_44 = (var_43 < var_36);
                    var_45 = (var_36 < var_41);
                    var_46 = var_44 && var_45;
                    if (var_46) {
                        // quad = ctx_quad_in[worldid, efcid]                                     <L 385>
                        var_47 = wp::address(var_ctx_quad_in, var_0, var_20);
                        var_49 = wp::load(var_47);
                        var_48 = wp::copy(var_49);
                    }
                    if (!var_46) {
                        // elif x <= -rf:                                                         <L 387>
                        var_50 = wp::neg(var_41);
                        var_51 = (var_36 <= var_50);
                        if (var_51) {
                            // quad = wp.vec3(f * (-0.5 * rf - start), -f * dir, 0.0)             <L 388>
                            var_54 = wp::mul(var_53, var_41);
                            var_55 = wp::sub(var_54, var_30);
                            var_56 = wp::mul(var_38, var_55);
                            var_57 = wp::neg(var_38);
                            var_58 = wp::mul(var_57, var_33);
                            var_60 = wp::vec_t<3, wp::float32>(var_56, var_58, var_59);
                        }
                        var_61 = wp::where(var_51, var_60, var_48);
                        if (!var_51) {
                            // quad = wp.vec3(f * (-0.5 * rf + start), f * dir, 0.0)              <L 391>
                            var_64 = wp::mul(var_63, var_41);
                            var_65 = wp::add(var_64, var_30);
                            var_66 = wp::mul(var_38, var_65);
                            var_67 = wp::mul(var_38, var_33);
                            var_69 = wp::vec_t<3, wp::float32>(var_66, var_67, var_68);
                        }
                        var_70 = wp::where(var_51, var_61, var_69);
                    }
                    var_71 = wp::where(var_46, var_48, var_70);
                    // out += _eval_cost(quad, alpha)                                             <L 393>
                    var_72 = _eval_cost_0(var_71, var_6);
                    var_73 = wp::add(var_26, var_72);
                }
                var_74 = wp::where(var_28, var_73, var_26);
                if (!var_28) {
                    // elif efc_type_in[worldid, efcid] == types.ConstraintType.CONTACT_ELLIPTIC:       <L 395>
                    var_75 = wp::address(var_efc_type_in, var_0, var_20);
                    var_78 = wp::load(var_75);
                    var_77 = (var_78 == var_76);
                    if (var_77) {
                        // conid = efc_id_in[worldid, efcid]                                      <L 397>
                        var_79 = wp::address(var_efc_id_in, var_0, var_20);
                        var_81 = wp::load(var_79);
                        var_80 = wp::copy(var_81);
                        // if conid >= nacon_in[0]:                                               <L 399>
                        var_83 = wp::address(var_nacon_in, var_82);
                        var_85 = wp::load(var_83);
                        var_84 = (var_80 >= var_85);
                        if (var_84) {
                            // continue                                                           <L 400>
                            wp::assign(var_8, var_74);
                            goto start_for_1;
                        }
                        var_86 = wp::where(var_84, var_8, var_74);
                        // efcid0 = contact_efc_address_in[conid, 0]                              <L 402>
                        var_88 = wp::address(var_contact_efc_address_in, var_80, var_87);
                        var_90 = wp::load(var_88);
                        var_89 = wp::copy(var_90);
                        // if efcid != efcid0:                                                    <L 403>
                        var_91 = (var_20 != var_89);
                        if (var_91) {
                            // continue                                                           <L 404>
                            wp::assign(var_8, var_86);
                            goto start_for_1;
                        }
                        var_92 = wp::where(var_91, var_8, var_86);
                        // friction = contact_friction_in[conid]                                  <L 406>
                        var_93 = wp::address(var_contact_friction_in, var_80);
                        var_95 = wp::load(var_93);
                        var_94 = wp::copy(var_95);
                        // mu = friction[0] * opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]       <L 407>
                        var_97 = wp::extract(var_94, var_96);
                        var_98 = &(var_opt_impratio_invsqrt.shape);
                        var_101 = wp::load(var_98);
                        var_100 = wp::extract(var_101, var_99);
                        var_102 = wp::mod(var_0, var_100);
                        var_103 = wp::address(var_opt_impratio_invsqrt, var_102);
                        var_105 = wp::load(var_103);
                        var_104 = wp::mul(var_97, var_105);
                        // efcid1 = contact_efc_address_in[conid, 1]                              <L 410>
                        var_107 = wp::address(var_contact_efc_address_in, var_80, var_106);
                        var_109 = wp::load(var_107);
                        var_108 = wp::copy(var_109);
                        // efcid2 = contact_efc_address_in[conid, 2]                              <L 411>
                        var_111 = wp::address(var_contact_efc_address_in, var_80, var_110);
                        var_113 = wp::load(var_111);
                        var_112 = wp::copy(var_113);
                        // u0 = ctx_quad_in[worldid, efcid1][0]                                   <L 412>
                        var_114 = wp::address(var_ctx_quad_in, var_0, var_108);
                        var_117 = wp::load(var_114);
                        var_116 = wp::extract(var_117, var_115);
                        // v0 = ctx_quad_in[worldid, efcid1][1]                                   <L 413>
                        var_118 = wp::address(var_ctx_quad_in, var_0, var_108);
                        var_121 = wp::load(var_118);
                        var_120 = wp::extract(var_121, var_119);
                        // uu = ctx_quad_in[worldid, efcid1][2]                                   <L 414>
                        var_122 = wp::address(var_ctx_quad_in, var_0, var_108);
                        var_125 = wp::load(var_122);
                        var_124 = wp::extract(var_125, var_123);
                        // uv = ctx_quad_in[worldid, efcid2][0]                                   <L 415>
                        var_126 = wp::address(var_ctx_quad_in, var_0, var_112);
                        var_129 = wp::load(var_126);
                        var_128 = wp::extract(var_129, var_127);
                        // vv = ctx_quad_in[worldid, efcid2][1]                                   <L 416>
                        var_130 = wp::address(var_ctx_quad_in, var_0, var_112);
                        var_133 = wp::load(var_130);
                        var_132 = wp::extract(var_133, var_131);
                        // dm = ctx_quad_in[worldid, efcid2][2]                                   <L 417>
                        var_134 = wp::address(var_ctx_quad_in, var_0, var_112);
                        var_137 = wp::load(var_134);
                        var_136 = wp::extract(var_137, var_135);
                        // N = u0 + alpha * v0                                                    <L 420>
                        var_138 = wp::mul(var_6, var_120);
                        var_139 = wp::add(var_116, var_138);
                        // Tsqr = uu + alpha * (2.0 * uv + alpha * vv)                            <L 421>
                        var_141 = wp::mul(var_140, var_128);
                        var_142 = wp::mul(var_6, var_132);
                        var_143 = wp::add(var_141, var_142);
                        var_144 = wp::mul(var_6, var_143);
                        var_145 = wp::add(var_124, var_144);
                        // if Tsqr <= 0.0:                                                        <L 424>
                        var_147 = (var_145 <= var_146);
                        if (var_147) {
                            // if N < 0.0:                                                        <L 426>
                            var_149 = (var_139 < var_148);
                            if (var_149) {
                                // out += _eval_cost(ctx_quad_in[worldid, efcid], alpha)          <L 427>
                                var_150 = wp::address(var_ctx_quad_in, var_0, var_20);
                                var_152 = wp::load(var_150);
                                var_151 = _eval_cost_0(var_152, var_6);
                                var_153 = wp::add(var_92, var_151);
                            }
                            var_154 = wp::where(var_149, var_153, var_92);
                        }
                        var_155 = wp::where(var_147, var_154, var_92);
                        if (!var_147) {
                            // T = wp.sqrt(Tsqr)                                                  <L 431>
                            var_156 = wp::sqrt(var_145);
                            // if N >= mu * T:                                                    <L 434>
                            var_157 = wp::mul(var_104, var_156);
                            var_158 = (var_139 >= var_157);
                            if (var_158) {
                                // pass                                                           <L 436>
                            }
                            if (!var_158) {
                                // elif mu * N + T <= 0.0:                                        <L 438>
                                var_159 = wp::mul(var_104, var_139);
                                var_160 = wp::add(var_159, var_156);
                                var_162 = (var_160 <= var_161);
                                if (var_162) {
                                    // out += _eval_cost(ctx_quad_in[worldid, efcid], alpha)       <L 439>
                                    var_163 = wp::address(var_ctx_quad_in, var_0, var_20);
                                    var_165 = wp::load(var_163);
                                    var_164 = _eval_cost_0(var_165, var_6);
                                    var_166 = wp::add(var_155, var_164);
                                }
                                var_167 = wp::where(var_162, var_166, var_155);
                                if (!var_162) {
                                    // out += 0.5 * dm * (N - mu * T) * (N - mu * T)              <L 442>
                                    var_169 = wp::mul(var_168, var_136);
                                    var_170 = wp::mul(var_104, var_156);
                                    var_171 = wp::sub(var_139, var_170);
                                    var_172 = wp::mul(var_169, var_171);
                                    var_173 = wp::mul(var_104, var_156);
                                    var_174 = wp::sub(var_139, var_173);
                                    var_175 = wp::mul(var_172, var_174);
                                    var_176 = wp::add(var_167, var_175);
                                }
                                var_177 = wp::where(var_162, var_167, var_176);
                            }
                            var_178 = wp::where(var_158, var_155, var_177);
                        }
                        var_179 = wp::where(var_147, var_155, var_178);
                    }
                    var_180 = wp::where(var_77, var_179, var_74);
                    if (!var_77) {
                        // x = ctx_Jaref_in[worldid, efcid] + alpha * ctx_jv_in[worldid, efcid]       <L 445>
                        var_181 = wp::address(var_ctx_Jaref_in, var_0, var_20);
                        var_182 = wp::address(var_ctx_jv_in, var_0, var_20);
                        var_184 = wp::load(var_182);
                        var_183 = wp::mul(var_6, var_184);
                        var_186 = wp::load(var_181);
                        var_185 = wp::add(var_186, var_183);
                        // if x < 0.0:                                                            <L 448>
                        var_188 = (var_185 < var_187);
                        if (var_188) {
                            // out += _eval_cost(ctx_quad_in[worldid, efcid], alpha)              <L 449>
                            var_189 = wp::address(var_ctx_quad_in, var_0, var_20);
                            var_191 = wp::load(var_189);
                            var_190 = _eval_cost_0(var_191, var_6);
                            var_192 = wp::add(var_180, var_190);
                        }
                        var_193 = wp::where(var_188, var_192, var_180);
                    }
                    var_194 = wp::where(var_77, var_180, var_193);
                    var_195 = wp::where(var_77, var_36, var_185);
                }
                var_196 = wp::where(var_28, var_74, var_194);
                var_197 = wp::where(var_28, var_36, var_195);
            }
            var_198 = wp::where(var_21, var_26, var_196);
            wp::assign(var_8, var_198);
            goto start_for_1;
        end_for_1:;
        // cost_out[worldid, alphaid] = out                                                       <L 451>
        wp::array_store(var_cost_out, var_0, var_1, var_8);
    }
}



extern "C" __global__ void solve_init_efc_122eb780_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_solver_niter_out,
    wp::array_t<wp::float32> var_ctx_search_dot_out,
    wp::array_t<wp::float32> var_ctx_cost_out,
    wp::array_t<bool> var_ctx_done_out)
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
        const wp::float32 var_1 = 10000000000.0;
        const wp::int32 var_2 = 0;
        const bool var_3 = false;
        const wp::float32 var_4 = 0.0;
        //---------
        // forward
        // def solve_init_efc(                                                                    <L 1707>
        // worldid = wp.tid()                                                                     <L 1715>
        var_0 = builtin_tid1d();
        // ctx_cost_out[worldid] = types.MJ_MAXVAL                                                <L 1716>
        wp::array_store(var_ctx_cost_out, var_0, var_1);
        // solver_niter_out[worldid] = 0                                                          <L 1717>
        wp::array_store(var_solver_niter_out, var_0, var_2);
        // ctx_done_out[worldid] = False                                                          <L 1718>
        wp::array_store(var_ctx_done_out, var_0, var_3);
        // ctx_search_dot_out[worldid] = 0.0                                                      <L 1719>
        wp::array_store(var_ctx_search_dot_out, var_0, var_4);
    }
}



extern "C" __global__ void linesearch_qacc_ma_fe8edc83_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_ctx_search_in,
    wp::array_t<wp::float32> var_ctx_mv_in,
    wp::array_t<wp::float32> var_ctx_alpha_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_qacc_out,
    wp::array_t<wp::float32> var_efc_Ma_out)
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
        wp::float32* var_5;
        wp::float32 var_6;
        wp::float32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        //---------
        // forward
        // def linesearch_qacc_ma(                                                                <L 1620>
        // worldid, dofid = wp.tid()                                                              <L 1630>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 1632>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 1633>
            continue;
        }
        var_4 = wp::load(var_2);
        // alpha = ctx_alpha_in[worldid]                                                          <L 1635>
        var_5 = wp::address(var_ctx_alpha_in, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // qacc_out[worldid, dofid] += alpha * ctx_search_in[worldid, dofid]                      <L 1636>
        var_8 = wp::address(var_ctx_search_in, var_0, var_1);
        var_10 = wp::load(var_8);
        var_9 = wp::mul(var_6, var_10);
        var_11 = wp::atomic_add(var_qacc_out, var_0, var_1, var_9);
        // efc_Ma_out[worldid, dofid] += alpha * ctx_mv_in[worldid, dofid]                        <L 1637>
        var_12 = wp::address(var_ctx_mv_in, var_0, var_1);
        var_14 = wp::load(var_12);
        var_13 = wp::mul(var_6, var_14);
        var_15 = wp::atomic_add(var_efc_Ma_out, var_0, var_1, var_13);
    }
}



extern "C" __global__ void linesearch_parallel_best_alpha_04b01f61_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_opt_ls_iterations,
    wp::float32 var_opt_ls_parallel_min_step,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_cost_in,
    wp::array_t<wp::float32> var_ctx_alpha_out)
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
        const wp::int32 var_4 = 0;
        wp::int32 var_5;
        const wp::float32 var_6 = 10000000000.0;
        const wp::float32 var_7 = 10000000000.0;
        wp::float32 var_8;
        wp::range_t var_9;
        wp::int32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        bool var_14;
        wp::float32 var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::float32 var_18;
        const wp::float32 var_19 = 1.0;
        wp::float32 var_20;
        //---------
        // forward
        // def linesearch_parallel_best_alpha(                                                    <L 455>
        // worldid = wp.tid()                                                                     <L 465>
        var_0 = builtin_tid1d();
        // if ctx_done_in[worldid]:                                                               <L 467>
        var_1 = wp::address(var_ctx_done_in, var_0);
        var_2 = wp::load(var_1);
        if (var_2) {
            // return                                                                             <L 468>
            continue;
        }
        var_3 = wp::load(var_1);
        // bestid = int(0)                                                                        <L 470>
        var_5 = wp::int(var_4);
        // best_cost = float(types.MJ_MAXVAL)                                                     <L 471>
        var_8 = wp::float(var_7);
        // for i in range(opt_ls_iterations):                                                     <L 472>
        var_9 = wp::range(var_opt_ls_iterations);
        start_for_1:;
            if (iter_cmp(var_9) == 0) goto end_for_1;
            var_10 = wp::iter_next(var_9);
            // cost = cost_in[worldid, i]                                                         <L 473>
            var_11 = wp::address(var_cost_in, var_0, var_10);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
            // if cost < best_cost:                                                               <L 474>
            var_14 = (var_12 < var_8);
            if (var_14) {
                // best_cost = cost                                                               <L 475>
                var_15 = wp::copy(var_12);
                // bestid = i                                                                     <L 476>
                var_16 = wp::copy(var_10);
            }
            var_17 = wp::where(var_14, var_16, var_5);
            var_18 = wp::where(var_14, var_15, var_8);
            wp::assign(var_5, var_17);
            wp::assign(var_8, var_18);
            goto start_for_1;
        end_for_1:;
        // ctx_alpha_out[worldid] = _log_scale(opt_ls_parallel_min_step, 1.0, opt_ls_iterations, bestid)       <L 478>
        var_20 = _log_scale_0(var_opt_ls_parallel_min_step, var_19, var_opt_ls_iterations, var_5);
        wp::array_store(var_ctx_alpha_out, var_0, var_20);
    }
}



extern "C" __global__ void update_gradient_grad_19709deb_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_qfrc_smooth_in,
    wp::array_t<wp::float32> var_qfrc_constraint_in,
    wp::array_t<wp::float32> var_efc_Ma_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_ctx_grad_out,
    wp::array_t<wp::float32> var_ctx_grad_dot_out)
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
        wp::float32* var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        //---------
        // forward
        // def update_gradient_grad(                                                              <L 2238>
        // worldid, dofid = wp.tid()                                                              <L 2249>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 2251>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 2252>
            continue;
        }
        var_4 = wp::load(var_2);
        // grad = efc_Ma_in[worldid, dofid] - qfrc_smooth_in[worldid, dofid] - qfrc_constraint_in[worldid, dofid]       <L 2254>
        var_5 = wp::address(var_efc_Ma_in, var_0, var_1);
        var_6 = wp::address(var_qfrc_smooth_in, var_0, var_1);
        var_8 = wp::load(var_5);
        var_9 = wp::load(var_6);
        var_7 = wp::sub(var_8, var_9);
        var_10 = wp::address(var_qfrc_constraint_in, var_0, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::sub(var_7, var_12);
        // ctx_grad_out[worldid, dofid] = grad                                                    <L 2255>
        wp::array_store(var_ctx_grad_out, var_0, var_1, var_11);
        // wp.atomic_add(ctx_grad_dot_out, worldid, grad * grad)                                  <L 2256>
        var_13 = wp::mul(var_11, var_11);
        var_14 = wp::atomic_add(var_ctx_grad_dot_out, var_0, var_13);
    }
}



extern "C" __global__ void solve_init_search_afe6eba1_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_ctx_Mgrad_in,
    wp::array_t<wp::float32> var_ctx_search_out,
    wp::array_t<wp::float32> var_ctx_search_dot_out)
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
        const wp::float32 var_2 = 1.0;
        const wp::float32 var_3 = -1.0;
        wp::float32* var_4;
        wp::float32 var_5;
        wp::float32 var_6;
        wp::float32 var_7;
        wp::float32 var_8;
        //---------
        // forward
        // def solve_init_search(                                                                 <L 1772>
        // worldid, dofid = wp.tid()                                                              <L 1779>
        builtin_tid2d(var_0, var_1);
        // search = -1.0 * ctx_Mgrad_in[worldid, dofid]                                           <L 1780>
        var_4 = wp::address(var_ctx_Mgrad_in, var_0, var_1);
        var_6 = wp::load(var_4);
        var_5 = wp::mul(var_3, var_6);
        // ctx_search_out[worldid, dofid] = search                                                <L 1781>
        wp::array_store(var_ctx_search_out, var_0, var_1, var_5);
        // wp.atomic_add(ctx_search_dot_out, worldid, search * search)                            <L 1782>
        var_7 = wp::mul(var_5, var_5);
        var_8 = wp::atomic_add(var_ctx_search_dot_out, var_0, var_7);
    }
}



extern "C" __global__ void linesearch_jaref_d290ce2b_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::float32> var_ctx_jv_in,
    wp::array_t<wp::float32> var_ctx_alpha_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_ctx_Jaref_out)
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
        wp::int32* var_2;
        bool var_3;
        wp::int32 var_4;
        bool* var_5;
        bool var_6;
        bool var_7;
        wp::float32* var_8;
        wp::float32* var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        //---------
        // forward
        // def linesearch_jaref(                                                                  <L 1641>
        // worldid, efcid = wp.tid()                                                              <L 1651>
        builtin_tid2d(var_0, var_1);
        // if efcid >= nefc_in[worldid]:                                                          <L 1653>
        var_2 = wp::address(var_nefc_in, var_0);
        var_4 = wp::load(var_2);
        var_3 = (var_1 >= var_4);
        if (var_3) {
            // return                                                                             <L 1654>
            continue;
        }
        // if ctx_done_in[worldid]:                                                               <L 1656>
        var_5 = wp::address(var_ctx_done_in, var_0);
        var_6 = wp::load(var_5);
        if (var_6) {
            // return                                                                             <L 1657>
            continue;
        }
        var_7 = wp::load(var_5);
        // ctx_Jaref_out[worldid, efcid] += ctx_alpha_in[worldid] * ctx_jv_in[worldid, efcid]       <L 1659>
        var_8 = wp::address(var_ctx_alpha_in, var_0);
        var_9 = wp::address(var_ctx_jv_in, var_0, var_1);
        var_11 = wp::load(var_8);
        var_12 = wp::load(var_9);
        var_10 = wp::mul(var_11, var_12);
        var_13 = wp::atomic_add(var_ctx_Jaref_out, var_0, var_1, var_10);
    }
}



extern "C" __global__ void update_gradient_set_h_qM_lower_sparse_9caabb03_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_qM_fullm_i,
    wp::array_t<wp::int32> var_qM_fullm_j,
    wp::array_t<wp::float32> var_qM_in,
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
        wp::int32 var_1;
        bool* var_2;
        bool var_3;
        bool var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        const wp::int32 var_11 = 0;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        //---------
        // forward
        // def update_gradient_set_h_qM_lower_sparse(                                             <L 2260>
        // worldid, elementid = wp.tid()                                                          <L 2271>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 2273>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 2274>
            continue;
        }
        var_4 = wp::load(var_2);
        // i = qM_fullm_i[elementid]                                                              <L 2276>
        var_5 = wp::address(var_qM_fullm_i, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // j = qM_fullm_j[elementid]                                                              <L 2277>
        var_8 = wp::address(var_qM_fullm_j, var_1);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // ctx_h_out[worldid, i, j] += qM_in[worldid, 0, elementid]                               <L 2278>
        var_12 = wp::address(var_qM_in, var_0, var_11, var_1);
        var_14 = wp::load(var_12);
        var_13 = wp::atomic_add(var_ctx_h_out, var_0, var_6, var_9, var_14);
    }
}



extern "C" __global__ void update_gradient_h_incremental_219b7155_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::int32> var_efc_state_in,
    wp::array_t<wp::int32> var_changed_ids_in,
    wp::array_t<wp::int32> var_changed_count_in,
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
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        const wp::int32 var_7 = 1;
        const wp::int32 var_8 = 8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        wp::int32 var_13;
        const wp::int32 var_14 = 1;
        wp::int32 var_15;
        const wp::int32 var_16 = 2;
        wp::int32 var_17;
        const wp::int32 var_18 = 1;
        wp::int32 var_19;
        wp::int32 var_20;
        const wp::int32 var_21 = 2;
        wp::int32 var_22;
        wp::int32 var_23;
        const wp::float32 var_24 = 0.0;
        wp::float32 var_25;
        wp::range_t var_26;
        wp::int32 var_27;
        wp::int32* var_28;
        wp::int32 var_29;
        wp::int32 var_30;
        wp::float32* var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        const wp::float32 var_34 = 0.0;
        bool var_35;
        wp::float32* var_36;
        wp::float32 var_37;
        wp::float32 var_38;
        const wp::float32 var_39 = 0.0;
        bool var_40;
        wp::float32* var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::int32* var_44;
        const wp::int32 var_45 = 1;
        bool var_46;
        wp::int32 var_47;
        wp::float32 var_48;
        wp::float32 var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        wp::float32 var_52;
        wp::float32 var_53;
        wp::float32 var_54;
        wp::float32 var_55;
        const wp::float32 var_56 = 0.0;
        bool var_57;
        wp::float32 var_58;
        //---------
        // forward
        // def update_gradient_h_incremental(                                                     <L 2055>
        // worldid, elementid = wp.tid()                                                          <L 2071>
        builtin_tid2d(var_0, var_1);
        // n_changes = changed_count_in[worldid]                                                  <L 2073>
        var_2 = wp::address(var_changed_count_in, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if n_changes == 0:                                                                     <L 2074>
        var_6 = (var_3 == var_5);
        if (var_6) {
            // return                                                                             <L 2075>
            continue;
        }
        // i = (int(wp.sqrt(float(1 + 8 * elementid))) - 1) // 2                                  <L 2078>
        var_9 = wp::mul(var_8, var_1);
        var_10 = wp::add(var_7, var_9);
        var_11 = wp::float(var_10);
        var_12 = wp::sqrt(var_11);
        var_13 = wp::int(var_12);
        var_15 = wp::sub(var_13, var_14);
        var_17 = wp::floordiv(var_15, var_16);
        // j = elementid - (i * (i + 1)) // 2                                                     <L 2079>
        var_19 = wp::add(var_17, var_18);
        var_20 = wp::mul(var_17, var_19);
        var_22 = wp::floordiv(var_20, var_21);
        var_23 = wp::sub(var_1, var_22);
        // delta = float(0.0)                                                                     <L 2081>
        var_25 = wp::float(var_24);
        // for change_idx in range(n_changes):                                                    <L 2082>
        var_26 = wp::range(var_3);
        start_for_1:;
            if (iter_cmp(var_26) == 0) goto end_for_1;
            var_27 = wp::iter_next(var_26);
            // efcid = changed_ids_in[worldid, change_idx]                                        <L 2083>
            var_28 = wp::address(var_changed_ids_in, var_0, var_27);
            var_30 = wp::load(var_28);
            var_29 = wp::copy(var_30);
            // Ji = efc_J_in[worldid, efcid, i]                                                   <L 2084>
            var_31 = wp::address(var_efc_J_in, var_0, var_29, var_17);
            var_33 = wp::load(var_31);
            var_32 = wp::copy(var_33);
            // if Ji == 0.0:                                                                      <L 2085>
            var_35 = (var_32 == var_34);
            if (var_35) {
                // continue                                                                       <L 2086>
                goto start_for_1;
            }
            // Jj = efc_J_in[worldid, efcid, j]                                                   <L 2087>
            var_36 = wp::address(var_efc_J_in, var_0, var_29, var_23);
            var_38 = wp::load(var_36);
            var_37 = wp::copy(var_38);
            // if Jj == 0.0:                                                                      <L 2088>
            var_40 = (var_37 == var_39);
            if (var_40) {
                // continue                                                                       <L 2089>
                goto start_for_1;
            }
            // D = efc_D_in[worldid, efcid]                                                       <L 2091>
            var_41 = wp::address(var_efc_D_in, var_0, var_29);
            var_43 = wp::load(var_41);
            var_42 = wp::copy(var_43);
            // if efc_state_in[worldid, efcid] == types.ConstraintState.QUADRATIC.value:          <L 2092>
            var_44 = wp::address(var_efc_state_in, var_0, var_29);
            var_47 = wp::load(var_44);
            var_46 = (var_47 == var_45);
            if (var_46) {
                // delta += D * Ji * Jj                                                           <L 2093>
                var_48 = wp::mul(var_42, var_32);
                var_49 = wp::mul(var_48, var_37);
                var_50 = wp::add(var_25, var_49);
            }
            var_51 = wp::where(var_46, var_50, var_25);
            if (!var_46) {
                // delta -= D * Ji * Jj                                                           <L 2095>
                var_52 = wp::mul(var_42, var_32);
                var_53 = wp::mul(var_52, var_37);
                var_54 = wp::sub(var_51, var_53);
            }
            var_55 = wp::where(var_46, var_51, var_54);
            wp::assign(var_25, var_55);
            goto start_for_1;
        end_for_1:;
        // if delta != 0.0:                                                                       <L 2097>
        var_57 = (var_25 != var_56);
        if (var_57) {
            // ctx_h_out[worldid, i, j] += delta                                                  <L 2098>
            var_58 = wp::atomic_add(var_ctx_h_out, var_0, var_17, var_23, var_25);
        }
    }
}



extern "C" __global__ void solve_zero_search_dot_07234fd8_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_ctx_search_dot_out)
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
        const wp::float32 var_4 = 0.0;
        //---------
        // forward
        // def solve_zero_search_dot(                                                             <L 3109>
        // worldid = wp.tid()                                                                     <L 3115>
        var_0 = builtin_tid1d();
        // if ctx_done_in[worldid]:                                                               <L 3117>
        var_1 = wp::address(var_ctx_done_in, var_0);
        var_2 = wp::load(var_1);
        if (var_2) {
            // return                                                                             <L 3118>
            continue;
        }
        var_3 = wp::load(var_1);
        // ctx_search_dot_out[worldid] = 0.0                                                      <L 3120>
        wp::array_store(var_ctx_search_dot_out, var_0, var_4);
    }
}



extern "C" __global__ void update_gradient_h_incremental_sparse_aa98a1ee_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_efc_J_rownnz_in,
    wp::array_t<wp::int32> var_efc_J_rowadr_in,
    wp::array_t<wp::int32> var_efc_J_colind_in,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::int32> var_efc_state_in,
    wp::array_t<wp::int32> var_changed_ids_in,
    wp::array_t<wp::int32> var_changed_count_in,
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
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        bool var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::float32* var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        const wp::float32 var_12 = 0.0;
        wp::float32 var_13;
        wp::int32* var_14;
        const wp::int32 var_15 = 1;
        bool var_16;
        wp::int32 var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        wp::int32* var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        wp::range_t var_28;
        wp::int32 var_29;
        wp::int32 var_30;
        const wp::int32 var_31 = 0;
        wp::float32* var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        const wp::float32 var_35 = 0.0;
        bool var_36;
        const wp::int32 var_37 = 0;
        wp::int32* var_38;
        wp::int32 var_39;
        wp::int32 var_40;
        const wp::int32 var_41 = 1;
        wp::int32 var_42;
        wp::range_t var_43;
        wp::int32 var_44;
        wp::int32 var_45;
        const wp::int32 var_46 = 0;
        wp::float32* var_47;
        wp::float32 var_48;
        wp::float32 var_49;
        const wp::float32 var_50 = 0.0;
        bool var_51;
        const wp::int32 var_52 = 0;
        wp::int32* var_53;
        wp::int32 var_54;
        wp::int32 var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        bool var_58;
        wp::slice_t var_59;
        const wp::int32 var_60 = 0;
        wp::slice_t var_61;
        const wp::int32 var_62 = 0;
        wp::array_t<wp::float32> var_63;
        wp::float32 var_64;
        wp::slice_t var_65;
        const wp::int32 var_66 = 0;
        wp::slice_t var_67;
        const wp::int32 var_68 = 0;
        wp::array_t<wp::float32> var_69;
        wp::float32 var_70;
        //---------
        // forward
        // def update_gradient_h_incremental_sparse(                                              <L 2102>
        // worldid, change_idx = wp.tid()                                                         <L 2117>
        builtin_tid2d(var_0, var_1);
        // n_changes = changed_count_in[worldid]                                                  <L 2119>
        var_2 = wp::address(var_changed_count_in, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if change_idx >= n_changes:                                                            <L 2120>
        var_5 = (var_1 >= var_3);
        if (var_5) {
            // return                                                                             <L 2121>
            continue;
        }
        // efcid = changed_ids_in[worldid, change_idx]                                            <L 2123>
        var_6 = wp::address(var_changed_ids_in, var_0, var_1);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // D = efc_D_in[worldid, efcid]                                                           <L 2124>
        var_9 = wp::address(var_efc_D_in, var_0, var_7);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // sign = float(0.0)                                                                      <L 2125>
        var_13 = wp::float(var_12);
        // if efc_state_in[worldid, efcid] == types.ConstraintState.QUADRATIC.value:              <L 2126>
        var_14 = wp::address(var_efc_state_in, var_0, var_7);
        var_17 = wp::load(var_14);
        var_16 = (var_17 == var_15);
        if (var_16) {
            // sign = D                                                                           <L 2127>
            var_18 = wp::copy(var_10);
        }
        var_19 = wp::where(var_16, var_18, var_13);
        if (!var_16) {
            // sign = -D                                                                          <L 2129>
            var_20 = wp::neg(var_10);
        }
        var_21 = wp::where(var_16, var_19, var_20);
        // rownnz = efc_J_rownnz_in[worldid, efcid]                                               <L 2131>
        var_22 = wp::address(var_efc_J_rownnz_in, var_0, var_7);
        var_24 = wp::load(var_22);
        var_23 = wp::copy(var_24);
        // rowadr = efc_J_rowadr_in[worldid, efcid]                                               <L 2132>
        var_25 = wp::address(var_efc_J_rowadr_in, var_0, var_7);
        var_27 = wp::load(var_25);
        var_26 = wp::copy(var_27);
        // for ii in range(rownnz):                                                               <L 2134>
        var_28 = wp::range(var_23);
        start_for_1:;
            if (iter_cmp(var_28) == 0) goto end_for_1;
            var_29 = wp::iter_next(var_28);
            // sparseidi = rowadr + ii                                                            <L 2135>
            var_30 = wp::add(var_26, var_29);
            // Ji = efc_J_in[worldid, 0, sparseidi]                                               <L 2136>
            var_32 = wp::address(var_efc_J_in, var_0, var_31, var_30);
            var_34 = wp::load(var_32);
            var_33 = wp::copy(var_34);
            // if Ji == 0.0:                                                                      <L 2137>
            var_36 = (var_33 == var_35);
            if (var_36) {
                // continue                                                                       <L 2138>
                goto start_for_1;
            }
            // colindi = efc_J_colind_in[worldid, 0, sparseidi]                                   <L 2139>
            var_38 = wp::address(var_efc_J_colind_in, var_0, var_37, var_30);
            var_40 = wp::load(var_38);
            var_39 = wp::copy(var_40);
            // for jj in range(ii + 1):                                                           <L 2140>
            var_42 = wp::add(var_29, var_41);
            var_43 = wp::range(var_42);
            start_for_3:;
                if (iter_cmp(var_43) == 0) goto end_for_3;
                var_44 = wp::iter_next(var_43);
                // sparseidj = rowadr + jj                                                        <L 2141>
                var_45 = wp::add(var_26, var_44);
                // Jj = efc_J_in[worldid, 0, sparseidj]                                           <L 2142>
                var_47 = wp::address(var_efc_J_in, var_0, var_46, var_45);
                var_49 = wp::load(var_47);
                var_48 = wp::copy(var_49);
                // if Jj == 0.0:                                                                  <L 2143>
                var_51 = (var_48 == var_50);
                if (var_51) {
                    // continue                                                                   <L 2144>
                    goto start_for_3;
                }
                // colindj = efc_J_colind_in[worldid, 0, sparseidj]                               <L 2145>
                var_53 = wp::address(var_efc_J_colind_in, var_0, var_52, var_45);
                var_55 = wp::load(var_53);
                var_54 = wp::copy(var_55);
                // h = sign * Ji * Jj                                                             <L 2146>
                var_56 = wp::mul(var_21, var_33);
                var_57 = wp::mul(var_56, var_48);
                // if colindi >= colindj:                                                         <L 2148>
                var_58 = (var_39 >= var_54);
                if (var_58) {
                    // wp.atomic_add(ctx_h_out[worldid, colindi], colindj, h)                     <L 2149>
                    var_59 = wp::slice_t(var_0, var_0, var_60);
                    var_61 = wp::slice_t(var_39, var_39, var_62);
                    var_63 = wp::view(var_ctx_h_out, var_59, var_61);
                    var_64 = wp::atomic_add(var_63, var_54, var_57);
                }
                if (!var_58) {
                    // wp.atomic_add(ctx_h_out[worldid, colindj], colindi, h)                     <L 2151>
                    var_65 = wp::slice_t(var_0, var_0, var_66);
                    var_67 = wp::slice_t(var_54, var_54, var_68);
                    var_69 = wp::view(var_ctx_h_out, var_65, var_67);
                    var_70 = wp::atomic_add(var_69, var_39, var_57);
                }
                goto start_for_3;
            end_for_3:;
            goto start_for_1;
        end_for_1:;
    }
}



extern "C" __global__ void update_constraint_init_cost_f27a2186_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_ctx_cost_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_ctx_gauss_out,
    wp::array_t<wp::float32> var_ctx_cost_out,
    wp::array_t<wp::float32> var_ctx_prev_cost_out)
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
        const wp::float32 var_4 = 0.0;
        wp::float32* var_5;
        wp::float32 var_6;
        const wp::float32 var_7 = 0.0;
        //---------
        // forward
        // def update_constraint_init_cost(                                                       <L 1786>
        // worldid = wp.tid()                                                                     <L 1795>
        var_0 = builtin_tid1d();
        // if ctx_done_in[worldid]:                                                               <L 1797>
        var_1 = wp::address(var_ctx_done_in, var_0);
        var_2 = wp::load(var_1);
        if (var_2) {
            // return                                                                             <L 1798>
            continue;
        }
        var_3 = wp::load(var_1);
        // ctx_gauss_out[worldid] = 0.0                                                           <L 1800>
        wp::array_store(var_ctx_gauss_out, var_0, var_4);
        // ctx_prev_cost_out[worldid] = ctx_cost_in[worldid]                                      <L 1801>
        var_5 = wp::address(var_ctx_cost_in, var_0);
        var_6 = wp::load(var_5);
        wp::array_store(var_ctx_prev_cost_out, var_0, var_6);
        // ctx_cost_out[worldid] = 0.0                                                            <L 1802>
        wp::array_store(var_ctx_cost_out, var_0, var_7);
    }
}

