
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



extern "C" __global__ void unchanged_clone_98f34061_cuda_kernel_forward(
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
        // def unchanged_clone(                                                                   <L 8>
        // conid_start, elementid = wp.tid()                                                      <L 36>
        builtin_tid2d(var_0, var_1);
        // dof1id = dof_tri_row[elementid]                                                        <L 38>
        var_2 = wp::address(var_dof_tri_row, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // dof2id = dof_tri_col[elementid]                                                        <L 39>
        var_5 = wp::address(var_dof_tri_col, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // for i in range(nblocks_perblock):                                                      <L 41>
        var_8 = wp::range(var_nblocks_perblock);
        start_for_0:;
            if (iter_cmp(var_8) == 0) goto end_for_0;
            var_9 = wp::iter_next(var_8);
            // conid = conid_start + i * dim_block                                                <L 42>
            var_10 = wp::mul(var_9, var_dim_block);
            var_11 = wp::add(var_0, var_10);
            // if conid >= min(nacon_in[0], naconmax_in):                                         <L 44>
            var_13 = wp::address(var_nacon_in, var_12);
            var_15 = wp::load(var_13);
            var_14 = wp::min(var_15, var_naconmax_in);
            var_16 = (var_11 >= var_14);
            if (var_16) {
                // return                                                                         <L 45>
                continue;
            }
            // worldid = contact_worldid_in[conid]                                                <L 47>
            var_17 = wp::address(var_contact_worldid_in, var_11);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
            // if ctx_done_in[worldid]:                                                           <L 48>
            var_20 = wp::address(var_ctx_done_in, var_18);
            var_21 = wp::load(var_20);
            if (var_21) {
                // continue                                                                       <L 49>
                goto start_for_0;
            }
            var_22 = wp::load(var_20);
            // condim = contact_dim_in[conid]                                                     <L 51>
            var_23 = wp::address(var_contact_dim_in, var_11);
            var_25 = wp::load(var_23);
            var_24 = wp::copy(var_25);
            // if condim == 1:                                                                    <L 53>
            var_27 = (var_24 == var_26);
            if (var_27) {
                // continue                                                                       <L 54>
                goto start_for_0;
            }
            // if contact_dist_in[conid] - contact_includemargin_in[conid] >= 0.0:                <L 57>
            var_28 = wp::address(var_contact_dist_in, var_11);
            var_29 = wp::address(var_contact_includemargin_in, var_11);
            var_31 = wp::load(var_28);
            var_32 = wp::load(var_29);
            var_30 = wp::sub(var_31, var_32);
            var_34 = (var_30 >= var_33);
            if (var_34) {
                // continue                                                                       <L 58>
                goto start_for_0;
            }
            // efcid0 = contact_efc_address_in[conid, 0]                                          <L 60>
            var_36 = wp::address(var_contact_efc_address_in, var_11, var_35);
            var_38 = wp::load(var_36);
            var_37 = wp::copy(var_38);
            // if efc_state_in[worldid, efcid0] != types.ConstraintState.CONE:                    <L 61>
            var_39 = wp::address(var_efc_state_in, var_18, var_37);
            var_42 = wp::load(var_39);
            var_41 = (var_42 != var_40);
            if (var_41) {
                // continue                                                                       <L 62>
                goto start_for_0;
            }
            // rownnz = efc_J_rownnz_in[worldid, efcid0]                                          <L 66>
            var_43 = wp::address(var_efc_J_rownnz_in, var_18, var_37);
            var_45 = wp::load(var_43);
            var_44 = wp::copy(var_45);
            // rowadr0 = efc_J_rowadr_in[worldid, efcid0]                                         <L 67>
            var_46 = wp::address(var_efc_J_rowadr_in, var_18, var_37);
            var_48 = wp::load(var_46);
            var_47 = wp::copy(var_48);
            // pos1 = int(-1)                                                                     <L 68>
            var_51 = wp::int(var_50);
            // pos2 = int(-1)                                                                     <L 69>
            var_54 = wp::int(var_53);
            // for k in range(rownnz):                                                            <L 70>
            var_55 = wp::range(var_44);
            start_for_3:;
                if (iter_cmp(var_55) == 0) goto end_for_3;
                var_56 = wp::iter_next(var_55);
                // col = efc_J_colind_in[worldid, 0, rowadr0 + k]                                 <L 71>
                var_58 = wp::add(var_47, var_56);
                var_59 = wp::address(var_efc_J_colind_in, var_18, var_57, var_58);
                var_61 = wp::load(var_59);
                var_60 = wp::copy(var_61);
                // if col == dof1id:                                                              <L 72>
                var_62 = (var_60 == var_3);
                if (var_62) {
                    // pos1 = k                                                                   <L 73>
                    var_63 = wp::copy(var_56);
                }
                var_64 = wp::where(var_62, var_63, var_51);
                // if col == dof2id:                                                              <L 74>
                var_65 = (var_60 == var_6);
                if (var_65) {
                    // pos2 = k                                                                   <L 75>
                    var_66 = wp::copy(var_56);
                }
                var_67 = wp::where(var_65, var_66, var_54);
                // if pos1 >= 0 and pos2 >= 0:                                                    <L 76>
                var_69 = (var_64 >= var_68);
                var_71 = (var_67 >= var_70);
                var_72 = var_69 && var_71;
                if (var_72) {
                    // break                                                                      <L 77>
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
            // if pos1 < 0 or pos2 < 0:                                                           <L 78>
            var_76 = (var_51 < var_75);
            var_78 = (var_54 < var_77);
            var_79 = var_76 || var_78;
            if (var_79) {
                // continue                                                                       <L 79>
                goto start_for_0;
            }
            // fri = contact_friction_in[conid]                                                   <L 81>
            var_80 = wp::address(var_contact_friction_in, var_11);
            var_82 = wp::load(var_80);
            var_81 = wp::copy(var_82);
            // mu = fri[0] * opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]        <L 82>
            var_84 = wp::extract(var_81, var_83);
            var_85 = &(var_opt_impratio_invsqrt.shape);
            var_88 = wp::load(var_85);
            var_87 = wp::extract(var_88, var_86);
            var_89 = wp::mod(var_18, var_87);
            var_90 = wp::address(var_opt_impratio_invsqrt, var_89);
            var_92 = wp::load(var_90);
            var_91 = wp::mul(var_84, var_92);
            // mu2 = mu * mu                                                                      <L 84>
            var_93 = wp::mul(var_91, var_91);
            // dm = math.safe_div(efc_D_in[worldid, efcid0], mu2 * (1.0 + mu2))                   <L 85>
            var_94 = wp::address(var_efc_D_in, var_18, var_37);
            var_96 = wp::add(var_95, var_93);
            var_97 = wp::mul(var_93, var_96);
            var_99 = wp::load(var_94);
            var_98 = safe_div_0(var_99, var_97);
            // if dm == 0.0:                                                                      <L 87>
            var_101 = (var_98 == var_100);
            if (var_101) {
                // continue                                                                       <L 88>
                goto start_for_0;
            }
            // n = ctx_Jaref_in[worldid, efcid0] * mu                                             <L 90>
            var_102 = wp::address(var_ctx_Jaref_in, var_18, var_37);
            var_104 = wp::load(var_102);
            var_103 = wp::mul(var_104, var_91);
            // u = types.vec6(n, 0.0, 0.0, 0.0, 0.0, 0.0)                                         <L 91>
            var_110 = wp::vec_t<6, wp::float32>({var_103, var_105, var_106, var_107, var_108, var_109});
            // tt = float(0.0)                                                                    <L 93>
            var_112 = wp::float(var_111);
            // for j in range(1, condim):                                                         <L 94>
            var_114 = wp::range(var_113, var_24);
            start_for_5:;
                if (iter_cmp(var_114) == 0) goto end_for_5;
                var_115 = wp::iter_next(var_114);
                // efcidj = contact_efc_address_in[conid, j]                                      <L 95>
                var_116 = wp::address(var_contact_efc_address_in, var_11, var_115);
                var_118 = wp::load(var_116);
                var_117 = wp::copy(var_118);
                // uj = ctx_Jaref_in[worldid, efcidj] * fri[j - 1]                                <L 96>
                var_119 = wp::address(var_ctx_Jaref_in, var_18, var_117);
                var_121 = wp::sub(var_115, var_120);
                var_122 = wp::extract(var_81, var_121);
                var_124 = wp::load(var_119);
                var_123 = wp::mul(var_124, var_122);
                // tt += uj * uj                                                                  <L 97>
                var_125 = wp::mul(var_123, var_123);
                var_126 = wp::add(var_112, var_125);
                // u[j] = uj                                                                      <L 98>
                wp::assign_inplace(var_110, var_115, var_123);
                wp::assign(var_112, var_126);
                goto start_for_5;
            end_for_5:;
            // if tt <= 0.0:                                                                      <L 100>
            var_128 = (var_112 <= var_127);
            if (var_128) {
                // t = 0.0                                                                        <L 101>
            }
            if (!var_128) {
                // t = wp.sqrt(tt)                                                                <L 103>
                var_130 = wp::sqrt(var_112);
            }
            var_131 = wp::where(var_128, var_129, var_130);
            // t = wp.max(t, types.MJ_MINVAL)                                                     <L 104>
            var_134 = wp::max(var_131, var_133);
            // ttt = wp.max(t * t * t, types.MJ_MINVAL)                                           <L 105>
            var_135 = wp::mul(var_134, var_134);
            var_136 = wp::mul(var_135, var_134);
            var_139 = wp::max(var_136, var_138);
            // mu_over_t = math.safe_div(mu, t)                                                   <L 108>
            var_140 = safe_div_0(var_91, var_134);
            // mu_n_over_ttt = mu * math.safe_div(n, ttt)                                         <L 109>
            var_141 = safe_div_0(var_103, var_139);
            var_142 = wp::mul(var_91, var_141);
            // mu2_minus_mu_n_over_t = mu2 - mu * math.safe_div(n, t)                             <L 110>
            var_143 = safe_div_0(var_103, var_134);
            var_144 = wp::mul(var_91, var_143);
            var_145 = wp::sub(var_93, var_144);
            // h = float(0.0)                                                                     <L 112>
            var_147 = wp::float(var_146);
            // for dim1id in range(condim):                                                       <L 114>
            var_148 = wp::range(var_24);
            start_for_7:;
                if (iter_cmp(var_148) == 0) goto end_for_7;
                var_149 = wp::iter_next(var_148);
                // if dim1id == 0:                                                                <L 115>
                var_151 = (var_149 == var_150);
                if (var_151) {
                    // rowadr1 = rowadr0                                                          <L 116>
                    var_152 = wp::copy(var_47);
                    // dm_fri1 = dm * mu                                                          <L 117>
                    var_153 = wp::mul(var_98, var_91);
                }
                if (!var_151) {
                    // efcid1 = contact_efc_address_in[conid, dim1id]                             <L 119>
                    var_154 = wp::address(var_contact_efc_address_in, var_11, var_149);
                    var_156 = wp::load(var_154);
                    var_155 = wp::copy(var_156);
                    // rowadr1 = efc_J_rowadr_in[worldid, efcid1]                                 <L 120>
                    var_157 = wp::address(var_efc_J_rowadr_in, var_18, var_155);
                    var_159 = wp::load(var_157);
                    var_158 = wp::copy(var_159);
                    // dm_fri1 = dm * fri[dim1id - 1]                                             <L 121>
                    var_161 = wp::sub(var_149, var_160);
                    var_162 = wp::extract(var_81, var_161);
                    var_163 = wp::mul(var_98, var_162);
                }
                var_164 = wp::where(var_151, var_152, var_158);
                var_165 = wp::where(var_151, var_153, var_163);
                // efc_J11 = efc_J_in[worldid, 0, rowadr1 + pos1]                                 <L 124>
                var_167 = wp::add(var_164, var_51);
                var_168 = wp::address(var_efc_J_in, var_18, var_166, var_167);
                var_170 = wp::load(var_168);
                var_169 = wp::copy(var_170);
                // efc_J12 = efc_J_in[worldid, 0, rowadr1 + pos2]                                 <L 125>
                var_172 = wp::add(var_164, var_54);
                var_173 = wp::address(var_efc_J_in, var_18, var_171, var_172);
                var_175 = wp::load(var_173);
                var_174 = wp::copy(var_175);
                // ui = u[dim1id]                                                                 <L 127>
                var_176 = wp::extract(var_110, var_149);
                // for dim2id in range(0, dim1id + 1):                                            <L 129>
                var_178 = wp::add(var_149, var_177);
                var_180 = wp::range(var_179, var_178);
                start_for_9:;
                    if (iter_cmp(var_180) == 0) goto end_for_9;
                    var_181 = wp::iter_next(var_180);
                    // if dim2id == 0:                                                            <L 130>
                    var_183 = (var_181 == var_182);
                    if (var_183) {
                        // rowadr2 = rowadr0                                                      <L 131>
                        var_184 = wp::copy(var_47);
                        // dm_fri12 = dm_fri1 * mu                                                <L 132>
                        var_185 = wp::mul(var_165, var_91);
                    }
                    if (!var_183) {
                        // efcid2 = contact_efc_address_in[conid, dim2id]                         <L 134>
                        var_186 = wp::address(var_contact_efc_address_in, var_11, var_181);
                        var_188 = wp::load(var_186);
                        var_187 = wp::copy(var_188);
                        // rowadr2 = efc_J_rowadr_in[worldid, efcid2]                             <L 135>
                        var_189 = wp::address(var_efc_J_rowadr_in, var_18, var_187);
                        var_191 = wp::load(var_189);
                        var_190 = wp::copy(var_191);
                        // dm_fri12 = dm_fri1 * fri[dim2id - 1]                                   <L 136>
                        var_193 = wp::sub(var_181, var_192);
                        var_194 = wp::extract(var_81, var_193);
                        var_195 = wp::mul(var_165, var_194);
                    }
                    var_196 = wp::where(var_183, var_184, var_190);
                    var_197 = wp::where(var_183, var_185, var_195);
                    // efc_J21 = efc_J_in[worldid, 0, rowadr2 + pos1]                             <L 139>
                    var_199 = wp::add(var_196, var_51);
                    var_200 = wp::address(var_efc_J_in, var_18, var_198, var_199);
                    var_202 = wp::load(var_200);
                    var_201 = wp::copy(var_202);
                    // efc_J22 = efc_J_in[worldid, 0, rowadr2 + pos2]                             <L 140>
                    var_204 = wp::add(var_196, var_54);
                    var_205 = wp::address(var_efc_J_in, var_18, var_203, var_204);
                    var_207 = wp::load(var_205);
                    var_206 = wp::copy(var_207);
                    // uj = u[dim2id]                                                             <L 142>
                    var_208 = wp::extract(var_110, var_181);
                    // if dim1id == 0 and dim2id == 0:                                            <L 145>
                    var_210 = (var_149 == var_209);
                    var_212 = (var_181 == var_211);
                    var_213 = var_210 && var_212;
                    if (var_213) {
                        // hcone = 1.0                                                            <L 146>
                    }
                    if (!var_213) {
                        // elif dim1id == 0:                                                      <L 147>
                        var_216 = (var_149 == var_215);
                        if (var_216) {
                            // hcone = -mu_over_t * uj                                            <L 148>
                            var_217 = wp::neg(var_140);
                            var_218 = wp::mul(var_217, var_208);
                        }
                        var_219 = wp::where(var_216, var_218, var_214);
                        if (!var_216) {
                            // elif dim2id == 0:                                                  <L 149>
                            var_221 = (var_181 == var_220);
                            if (var_221) {
                                // hcone = -mu_over_t * ui                                        <L 150>
                                var_222 = wp::neg(var_140);
                                var_223 = wp::mul(var_222, var_176);
                            }
                            var_224 = wp::where(var_221, var_223, var_219);
                            if (!var_221) {
                                // hcone = mu_n_over_ttt * ui * uj                                <L 152>
                                var_225 = wp::mul(var_142, var_176);
                                var_226 = wp::mul(var_225, var_208);
                                // if dim1id == dim2id:                                           <L 155>
                                var_227 = (var_149 == var_181);
                                if (var_227) {
                                    // hcone += mu2_minus_mu_n_over_t                             <L 156>
                                    var_228 = wp::add(var_226, var_145);
                                }
                                var_229 = wp::where(var_227, var_228, var_226);
                            }
                            var_230 = wp::where(var_221, var_224, var_229);
                        }
                        var_231 = wp::where(var_216, var_219, var_230);
                    }
                    var_232 = wp::where(var_213, var_214, var_231);
                    // hcone *= dm_fri12                                                          <L 158>
                    var_233 = wp::mul(var_232, var_197);
                    // if hcone != 0.0:                                                           <L 160>
                    var_235 = (var_233 != var_234);
                    if (var_235) {
                        // h += hcone * efc_J11 * efc_J22                                         <L 161>
                        var_236 = wp::mul(var_233, var_169);
                        var_237 = wp::mul(var_236, var_206);
                        var_238 = wp::add(var_147, var_237);
                        // if dim1id != dim2id:                                                   <L 163>
                        var_239 = (var_149 != var_181);
                        if (var_239) {
                            // h += hcone * efc_J12 * efc_J21                                     <L 164>
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
            // ctx_h_out[worldid, dof1id, dof2id] += h                                            <L 166>
            var_245 = wp::atomic_add(var_ctx_h_out, var_18, var_3, var_6, var_147);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void inverse_candidate_61a0358d_cuda_kernel_forward(
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
    wp::array_t<wp::int32> var_inverse_position_in,
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
        wp::int32* var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        const wp::int32 var_52 = 0;
        bool var_53;
        const wp::int32 var_54 = 0;
        bool var_55;
        bool var_56;
        wp::vec_t<5, wp::float32>* var_57;
        wp::vec_t<5, wp::float32> var_58;
        wp::vec_t<5, wp::float32> var_59;
        const wp::int32 var_60 = 0;
        wp::float32 var_61;
        wp::shape_t* var_62;
        const wp::int32 var_63 = 0;
        wp::int32 var_64;
        wp::shape_t var_65;
        wp::int32 var_66;
        wp::float32* var_67;
        wp::float32 var_68;
        wp::float32 var_69;
        wp::float32 var_70;
        wp::float32* var_71;
        const wp::float32 var_72 = 1.0;
        wp::float32 var_73;
        wp::float32 var_74;
        wp::float32 var_75;
        wp::float32 var_76;
        const wp::float32 var_77 = 0.0;
        bool var_78;
        wp::float32* var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        const wp::float32 var_82 = 0.0;
        const wp::float32 var_83 = 0.0;
        const wp::float32 var_84 = 0.0;
        const wp::float32 var_85 = 0.0;
        const wp::float32 var_86 = 0.0;
        wp::vec_t<6, wp::float32> var_87;
        const wp::float32 var_88 = 0.0;
        wp::float32 var_89;
        const wp::int32 var_90 = 1;
        wp::range_t var_91;
        wp::int32 var_92;
        wp::int32* var_93;
        wp::int32 var_94;
        wp::int32 var_95;
        wp::float32* var_96;
        const wp::int32 var_97 = 1;
        wp::int32 var_98;
        wp::float32 var_99;
        wp::float32 var_100;
        wp::float32 var_101;
        wp::float32 var_102;
        wp::float32 var_103;
        const wp::float32 var_104 = 0.0;
        bool var_105;
        const wp::float32 var_106 = 0.0;
        wp::float32 var_107;
        wp::float32 var_108;
        const wp::float32 var_109 = 1e-15;
        const wp::float32 var_110 = 1e-15;
        wp::float32 var_111;
        wp::float32 var_112;
        wp::float32 var_113;
        const wp::float32 var_114 = 1e-15;
        const wp::float32 var_115 = 1e-15;
        wp::float32 var_116;
        wp::float32 var_117;
        wp::float32 var_118;
        wp::float32 var_119;
        wp::float32 var_120;
        wp::float32 var_121;
        wp::float32 var_122;
        const wp::float32 var_123 = 0.0;
        wp::float32 var_124;
        wp::range_t var_125;
        wp::int32 var_126;
        const wp::int32 var_127 = 0;
        bool var_128;
        wp::int32 var_129;
        wp::float32 var_130;
        wp::int32* var_131;
        wp::int32 var_132;
        wp::int32 var_133;
        wp::int32* var_134;
        wp::int32 var_135;
        wp::int32 var_136;
        const wp::int32 var_137 = 1;
        wp::int32 var_138;
        wp::float32 var_139;
        wp::float32 var_140;
        wp::int32 var_141;
        wp::float32 var_142;
        const wp::int32 var_143 = 0;
        wp::int32 var_144;
        wp::float32* var_145;
        wp::float32 var_146;
        wp::float32 var_147;
        const wp::int32 var_148 = 0;
        wp::int32 var_149;
        wp::float32* var_150;
        wp::float32 var_151;
        wp::float32 var_152;
        wp::float32 var_153;
        const wp::int32 var_154 = 1;
        wp::int32 var_155;
        const wp::int32 var_156 = 0;
        wp::range_t var_157;
        wp::int32 var_158;
        const wp::int32 var_159 = 0;
        bool var_160;
        wp::int32 var_161;
        wp::float32 var_162;
        wp::int32* var_163;
        wp::int32 var_164;
        wp::int32 var_165;
        wp::int32* var_166;
        wp::int32 var_167;
        wp::int32 var_168;
        const wp::int32 var_169 = 1;
        wp::int32 var_170;
        wp::float32 var_171;
        wp::float32 var_172;
        wp::int32 var_173;
        wp::float32 var_174;
        const wp::int32 var_175 = 0;
        wp::int32 var_176;
        wp::float32* var_177;
        wp::float32 var_178;
        wp::float32 var_179;
        const wp::int32 var_180 = 0;
        wp::int32 var_181;
        wp::float32* var_182;
        wp::float32 var_183;
        wp::float32 var_184;
        wp::float32 var_185;
        const wp::int32 var_186 = 0;
        bool var_187;
        const wp::int32 var_188 = 0;
        bool var_189;
        bool var_190;
        const wp::float32 var_191 = 1.0;
        const wp::int32 var_192 = 0;
        bool var_193;
        wp::float32 var_194;
        wp::float32 var_195;
        wp::float32 var_196;
        const wp::int32 var_197 = 0;
        bool var_198;
        wp::float32 var_199;
        wp::float32 var_200;
        wp::float32 var_201;
        wp::float32 var_202;
        wp::float32 var_203;
        bool var_204;
        wp::float32 var_205;
        wp::float32 var_206;
        wp::float32 var_207;
        wp::float32 var_208;
        wp::float32 var_209;
        wp::float32 var_210;
        const wp::float32 var_211 = 0.0;
        bool var_212;
        wp::float32 var_213;
        wp::float32 var_214;
        wp::float32 var_215;
        bool var_216;
        wp::float32 var_217;
        wp::float32 var_218;
        wp::float32 var_219;
        wp::float32 var_220;
        wp::float32 var_221;
        wp::float32 var_222;
        //---------
        // forward
        // def inverse_candidate(                                                                 <L 170>
        // conid_start, elementid = wp.tid()                                                      <L 199>
        builtin_tid2d(var_0, var_1);
        // dof1id = dof_tri_row[elementid]                                                        <L 201>
        var_2 = wp::address(var_dof_tri_row, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // dof2id = dof_tri_col[elementid]                                                        <L 202>
        var_5 = wp::address(var_dof_tri_col, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // for i in range(nblocks_perblock):                                                      <L 204>
        var_8 = wp::range(var_nblocks_perblock);
        start_for_0:;
            if (iter_cmp(var_8) == 0) goto end_for_0;
            var_9 = wp::iter_next(var_8);
            // conid = conid_start + i * dim_block                                                <L 205>
            var_10 = wp::mul(var_9, var_dim_block);
            var_11 = wp::add(var_0, var_10);
            // if conid >= min(nacon_in[0], naconmax_in):                                         <L 207>
            var_13 = wp::address(var_nacon_in, var_12);
            var_15 = wp::load(var_13);
            var_14 = wp::min(var_15, var_naconmax_in);
            var_16 = (var_11 >= var_14);
            if (var_16) {
                // return                                                                         <L 208>
                continue;
            }
            // worldid = contact_worldid_in[conid]                                                <L 210>
            var_17 = wp::address(var_contact_worldid_in, var_11);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
            // if ctx_done_in[worldid]:                                                           <L 211>
            var_20 = wp::address(var_ctx_done_in, var_18);
            var_21 = wp::load(var_20);
            if (var_21) {
                // continue                                                                       <L 212>
                goto start_for_0;
            }
            var_22 = wp::load(var_20);
            // condim = contact_dim_in[conid]                                                     <L 214>
            var_23 = wp::address(var_contact_dim_in, var_11);
            var_25 = wp::load(var_23);
            var_24 = wp::copy(var_25);
            // if condim == 1:                                                                    <L 216>
            var_27 = (var_24 == var_26);
            if (var_27) {
                // continue                                                                       <L 217>
                goto start_for_0;
            }
            // if contact_dist_in[conid] - contact_includemargin_in[conid] >= 0.0:                <L 220>
            var_28 = wp::address(var_contact_dist_in, var_11);
            var_29 = wp::address(var_contact_includemargin_in, var_11);
            var_31 = wp::load(var_28);
            var_32 = wp::load(var_29);
            var_30 = wp::sub(var_31, var_32);
            var_34 = (var_30 >= var_33);
            if (var_34) {
                // continue                                                                       <L 221>
                goto start_for_0;
            }
            // efcid0 = contact_efc_address_in[conid, 0]                                          <L 223>
            var_36 = wp::address(var_contact_efc_address_in, var_11, var_35);
            var_38 = wp::load(var_36);
            var_37 = wp::copy(var_38);
            // if efc_state_in[worldid, efcid0] != types.ConstraintState.CONE:                    <L 224>
            var_39 = wp::address(var_efc_state_in, var_18, var_37);
            var_42 = wp::load(var_39);
            var_41 = (var_42 != var_40);
            if (var_41) {
                // continue                                                                       <L 225>
                goto start_for_0;
            }
            // rowadr0 = efc_J_rowadr_in[worldid, efcid0]                                         <L 230>
            var_43 = wp::address(var_efc_J_rowadr_in, var_18, var_37);
            var_45 = wp::load(var_43);
            var_44 = wp::copy(var_45);
            // pos1 = inverse_position_in[conid, dof1id]                                          <L 231>
            var_46 = wp::address(var_inverse_position_in, var_11, var_3);
            var_48 = wp::load(var_46);
            var_47 = wp::copy(var_48);
            // pos2 = inverse_position_in[conid, dof2id]                                          <L 232>
            var_49 = wp::address(var_inverse_position_in, var_11, var_6);
            var_51 = wp::load(var_49);
            var_50 = wp::copy(var_51);
            // if pos1 < 0 or pos2 < 0:                                                           <L 233>
            var_53 = (var_47 < var_52);
            var_55 = (var_50 < var_54);
            var_56 = var_53 || var_55;
            if (var_56) {
                // continue                                                                       <L 234>
                goto start_for_0;
            }
            // fri = contact_friction_in[conid]                                                   <L 236>
            var_57 = wp::address(var_contact_friction_in, var_11);
            var_59 = wp::load(var_57);
            var_58 = wp::copy(var_59);
            // mu = fri[0] * opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]        <L 237>
            var_61 = wp::extract(var_58, var_60);
            var_62 = &(var_opt_impratio_invsqrt.shape);
            var_65 = wp::load(var_62);
            var_64 = wp::extract(var_65, var_63);
            var_66 = wp::mod(var_18, var_64);
            var_67 = wp::address(var_opt_impratio_invsqrt, var_66);
            var_69 = wp::load(var_67);
            var_68 = wp::mul(var_61, var_69);
            // mu2 = mu * mu                                                                      <L 239>
            var_70 = wp::mul(var_68, var_68);
            // dm = math.safe_div(efc_D_in[worldid, efcid0], mu2 * (1.0 + mu2))                   <L 240>
            var_71 = wp::address(var_efc_D_in, var_18, var_37);
            var_73 = wp::add(var_72, var_70);
            var_74 = wp::mul(var_70, var_73);
            var_76 = wp::load(var_71);
            var_75 = safe_div_0(var_76, var_74);
            // if dm == 0.0:                                                                      <L 242>
            var_78 = (var_75 == var_77);
            if (var_78) {
                // continue                                                                       <L 243>
                goto start_for_0;
            }
            // n = ctx_Jaref_in[worldid, efcid0] * mu                                             <L 245>
            var_79 = wp::address(var_ctx_Jaref_in, var_18, var_37);
            var_81 = wp::load(var_79);
            var_80 = wp::mul(var_81, var_68);
            // u = types.vec6(n, 0.0, 0.0, 0.0, 0.0, 0.0)                                         <L 246>
            var_87 = wp::vec_t<6, wp::float32>({var_80, var_82, var_83, var_84, var_85, var_86});
            // tt = float(0.0)                                                                    <L 248>
            var_89 = wp::float(var_88);
            // for j in range(1, condim):                                                         <L 249>
            var_91 = wp::range(var_90, var_24);
            start_for_3:;
                if (iter_cmp(var_91) == 0) goto end_for_3;
                var_92 = wp::iter_next(var_91);
                // efcidj = contact_efc_address_in[conid, j]                                      <L 250>
                var_93 = wp::address(var_contact_efc_address_in, var_11, var_92);
                var_95 = wp::load(var_93);
                var_94 = wp::copy(var_95);
                // uj = ctx_Jaref_in[worldid, efcidj] * fri[j - 1]                                <L 251>
                var_96 = wp::address(var_ctx_Jaref_in, var_18, var_94);
                var_98 = wp::sub(var_92, var_97);
                var_99 = wp::extract(var_58, var_98);
                var_101 = wp::load(var_96);
                var_100 = wp::mul(var_101, var_99);
                // tt += uj * uj                                                                  <L 252>
                var_102 = wp::mul(var_100, var_100);
                var_103 = wp::add(var_89, var_102);
                // u[j] = uj                                                                      <L 253>
                wp::assign_inplace(var_87, var_92, var_100);
                wp::assign(var_89, var_103);
                goto start_for_3;
            end_for_3:;
            // if tt <= 0.0:                                                                      <L 255>
            var_105 = (var_89 <= var_104);
            if (var_105) {
                // t = 0.0                                                                        <L 256>
            }
            if (!var_105) {
                // t = wp.sqrt(tt)                                                                <L 258>
                var_107 = wp::sqrt(var_89);
            }
            var_108 = wp::where(var_105, var_106, var_107);
            // t = wp.max(t, types.MJ_MINVAL)                                                     <L 259>
            var_111 = wp::max(var_108, var_110);
            // ttt = wp.max(t * t * t, types.MJ_MINVAL)                                           <L 260>
            var_112 = wp::mul(var_111, var_111);
            var_113 = wp::mul(var_112, var_111);
            var_116 = wp::max(var_113, var_115);
            // mu_over_t = math.safe_div(mu, t)                                                   <L 263>
            var_117 = safe_div_0(var_68, var_111);
            // mu_n_over_ttt = mu * math.safe_div(n, ttt)                                         <L 264>
            var_118 = safe_div_0(var_80, var_116);
            var_119 = wp::mul(var_68, var_118);
            // mu2_minus_mu_n_over_t = mu2 - mu * math.safe_div(n, t)                             <L 265>
            var_120 = safe_div_0(var_80, var_111);
            var_121 = wp::mul(var_68, var_120);
            var_122 = wp::sub(var_70, var_121);
            // h = float(0.0)                                                                     <L 267>
            var_124 = wp::float(var_123);
            // for dim1id in range(condim):                                                       <L 269>
            var_125 = wp::range(var_24);
            start_for_5:;
                if (iter_cmp(var_125) == 0) goto end_for_5;
                var_126 = wp::iter_next(var_125);
                // if dim1id == 0:                                                                <L 270>
                var_128 = (var_126 == var_127);
                if (var_128) {
                    // rowadr1 = rowadr0                                                          <L 271>
                    var_129 = wp::copy(var_44);
                    // dm_fri1 = dm * mu                                                          <L 272>
                    var_130 = wp::mul(var_75, var_68);
                }
                if (!var_128) {
                    // efcid1 = contact_efc_address_in[conid, dim1id]                             <L 274>
                    var_131 = wp::address(var_contact_efc_address_in, var_11, var_126);
                    var_133 = wp::load(var_131);
                    var_132 = wp::copy(var_133);
                    // rowadr1 = efc_J_rowadr_in[worldid, efcid1]                                 <L 275>
                    var_134 = wp::address(var_efc_J_rowadr_in, var_18, var_132);
                    var_136 = wp::load(var_134);
                    var_135 = wp::copy(var_136);
                    // dm_fri1 = dm * fri[dim1id - 1]                                             <L 276>
                    var_138 = wp::sub(var_126, var_137);
                    var_139 = wp::extract(var_58, var_138);
                    var_140 = wp::mul(var_75, var_139);
                }
                var_141 = wp::where(var_128, var_129, var_135);
                var_142 = wp::where(var_128, var_130, var_140);
                // efc_J11 = efc_J_in[worldid, 0, rowadr1 + pos1]                                 <L 279>
                var_144 = wp::add(var_141, var_47);
                var_145 = wp::address(var_efc_J_in, var_18, var_143, var_144);
                var_147 = wp::load(var_145);
                var_146 = wp::copy(var_147);
                // efc_J12 = efc_J_in[worldid, 0, rowadr1 + pos2]                                 <L 280>
                var_149 = wp::add(var_141, var_50);
                var_150 = wp::address(var_efc_J_in, var_18, var_148, var_149);
                var_152 = wp::load(var_150);
                var_151 = wp::copy(var_152);
                // ui = u[dim1id]                                                                 <L 282>
                var_153 = wp::extract(var_87, var_126);
                // for dim2id in range(0, dim1id + 1):                                            <L 284>
                var_155 = wp::add(var_126, var_154);
                var_157 = wp::range(var_156, var_155);
                start_for_7:;
                    if (iter_cmp(var_157) == 0) goto end_for_7;
                    var_158 = wp::iter_next(var_157);
                    // if dim2id == 0:                                                            <L 285>
                    var_160 = (var_158 == var_159);
                    if (var_160) {
                        // rowadr2 = rowadr0                                                      <L 286>
                        var_161 = wp::copy(var_44);
                        // dm_fri12 = dm_fri1 * mu                                                <L 287>
                        var_162 = wp::mul(var_142, var_68);
                    }
                    if (!var_160) {
                        // efcid2 = contact_efc_address_in[conid, dim2id]                         <L 289>
                        var_163 = wp::address(var_contact_efc_address_in, var_11, var_158);
                        var_165 = wp::load(var_163);
                        var_164 = wp::copy(var_165);
                        // rowadr2 = efc_J_rowadr_in[worldid, efcid2]                             <L 290>
                        var_166 = wp::address(var_efc_J_rowadr_in, var_18, var_164);
                        var_168 = wp::load(var_166);
                        var_167 = wp::copy(var_168);
                        // dm_fri12 = dm_fri1 * fri[dim2id - 1]                                   <L 291>
                        var_170 = wp::sub(var_158, var_169);
                        var_171 = wp::extract(var_58, var_170);
                        var_172 = wp::mul(var_142, var_171);
                    }
                    var_173 = wp::where(var_160, var_161, var_167);
                    var_174 = wp::where(var_160, var_162, var_172);
                    // efc_J21 = efc_J_in[worldid, 0, rowadr2 + pos1]                             <L 294>
                    var_176 = wp::add(var_173, var_47);
                    var_177 = wp::address(var_efc_J_in, var_18, var_175, var_176);
                    var_179 = wp::load(var_177);
                    var_178 = wp::copy(var_179);
                    // efc_J22 = efc_J_in[worldid, 0, rowadr2 + pos2]                             <L 295>
                    var_181 = wp::add(var_173, var_50);
                    var_182 = wp::address(var_efc_J_in, var_18, var_180, var_181);
                    var_184 = wp::load(var_182);
                    var_183 = wp::copy(var_184);
                    // uj = u[dim2id]                                                             <L 297>
                    var_185 = wp::extract(var_87, var_158);
                    // if dim1id == 0 and dim2id == 0:                                            <L 300>
                    var_187 = (var_126 == var_186);
                    var_189 = (var_158 == var_188);
                    var_190 = var_187 && var_189;
                    if (var_190) {
                        // hcone = 1.0                                                            <L 301>
                    }
                    if (!var_190) {
                        // elif dim1id == 0:                                                      <L 302>
                        var_193 = (var_126 == var_192);
                        if (var_193) {
                            // hcone = -mu_over_t * uj                                            <L 303>
                            var_194 = wp::neg(var_117);
                            var_195 = wp::mul(var_194, var_185);
                        }
                        var_196 = wp::where(var_193, var_195, var_191);
                        if (!var_193) {
                            // elif dim2id == 0:                                                  <L 304>
                            var_198 = (var_158 == var_197);
                            if (var_198) {
                                // hcone = -mu_over_t * ui                                        <L 305>
                                var_199 = wp::neg(var_117);
                                var_200 = wp::mul(var_199, var_153);
                            }
                            var_201 = wp::where(var_198, var_200, var_196);
                            if (!var_198) {
                                // hcone = mu_n_over_ttt * ui * uj                                <L 307>
                                var_202 = wp::mul(var_119, var_153);
                                var_203 = wp::mul(var_202, var_185);
                                // if dim1id == dim2id:                                           <L 310>
                                var_204 = (var_126 == var_158);
                                if (var_204) {
                                    // hcone += mu2_minus_mu_n_over_t                             <L 311>
                                    var_205 = wp::add(var_203, var_122);
                                }
                                var_206 = wp::where(var_204, var_205, var_203);
                            }
                            var_207 = wp::where(var_198, var_201, var_206);
                        }
                        var_208 = wp::where(var_193, var_196, var_207);
                    }
                    var_209 = wp::where(var_190, var_191, var_208);
                    // hcone *= dm_fri12                                                          <L 313>
                    var_210 = wp::mul(var_209, var_174);
                    // if hcone != 0.0:                                                           <L 315>
                    var_212 = (var_210 != var_211);
                    if (var_212) {
                        // h += hcone * efc_J11 * efc_J22                                         <L 316>
                        var_213 = wp::mul(var_210, var_146);
                        var_214 = wp::mul(var_213, var_183);
                        var_215 = wp::add(var_124, var_214);
                        // if dim1id != dim2id:                                                   <L 318>
                        var_216 = (var_126 != var_158);
                        if (var_216) {
                            // h += hcone * efc_J12 * efc_J21                                     <L 319>
                            var_217 = wp::mul(var_210, var_151);
                            var_218 = wp::mul(var_217, var_178);
                            var_219 = wp::add(var_215, var_218);
                        }
                        var_220 = wp::where(var_216, var_219, var_215);
                    }
                    var_221 = wp::where(var_212, var_220, var_124);
                    wp::assign(var_100, var_185);
                    wp::assign(var_124, var_221);
                    goto start_for_7;
                end_for_7:;
                goto start_for_5;
            end_for_5:;
            // ctx_h_out[worldid, dof1id, dof2id] += h                                            <L 321>
            var_222 = wp::atomic_add(var_ctx_h_out, var_18, var_3, var_6, var_124);
            goto start_for_0;
        end_for_0:;
    }
}

