
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



extern "C" __global__ void refresh_cone_coefficients_4495f439_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_opt_impratio_invsqrt,
    wp::array_t<wp::float32> var_contact_dist_in,
    wp::array_t<wp::float32> var_contact_includemargin_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::int32> var_contact_worldid_in,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::int32> var_efc_state_in,
    wp::int32 var_naconmax_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::float32> var_ctx_Jaref_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_coefficients_out,
    wp::array_t<wp::int32> var_valid_out)
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
        const wp::int32 var_1 = 0;
        const wp::int32 var_2 = 0;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        bool var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        bool* var_10;
        bool var_11;
        bool var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        const wp::int32 var_16 = 1;
        bool var_17;
        wp::float32* var_18;
        wp::float32* var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        const wp::float32 var_23 = 0.0;
        bool var_24;
        const wp::int32 var_25 = 0;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::int32* var_29;
        const wp::int32 var_30 = 4;
        bool var_31;
        wp::int32 var_32;
        wp::vec_t<5, wp::float32>* var_33;
        wp::vec_t<5, wp::float32> var_34;
        wp::vec_t<5, wp::float32> var_35;
        const wp::int32 var_36 = 0;
        wp::float32 var_37;
        wp::shape_t* var_38;
        const wp::int32 var_39 = 0;
        wp::int32 var_40;
        wp::shape_t var_41;
        wp::int32 var_42;
        wp::float32* var_43;
        wp::float32 var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::float32* var_47;
        const wp::float32 var_48 = 1.0;
        wp::float32 var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        wp::float32 var_52;
        const wp::float32 var_53 = 0.0;
        bool var_54;
        wp::float32* var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        const wp::float32 var_58 = 0.0;
        const wp::float32 var_59 = 0.0;
        const wp::float32 var_60 = 0.0;
        const wp::float32 var_61 = 0.0;
        const wp::float32 var_62 = 0.0;
        wp::vec_t<6, wp::float32> var_63;
        const wp::float32 var_64 = 0.0;
        wp::float32 var_65;
        const wp::int32 var_66 = 1;
        wp::range_t var_67;
        wp::int32 var_68;
        wp::int32* var_69;
        wp::int32 var_70;
        wp::int32 var_71;
        wp::float32* var_72;
        const wp::int32 var_73 = 1;
        wp::int32 var_74;
        wp::float32 var_75;
        wp::float32 var_76;
        wp::float32 var_77;
        wp::float32 var_78;
        wp::float32 var_79;
        const wp::float32 var_80 = 0.0;
        bool var_81;
        const wp::float32 var_82 = 0.0;
        wp::float32 var_83;
        wp::float32 var_84;
        const wp::float32 var_85 = 1e-15;
        const wp::float32 var_86 = 1e-15;
        wp::float32 var_87;
        wp::float32 var_88;
        wp::float32 var_89;
        const wp::float32 var_90 = 1e-15;
        const wp::float32 var_91 = 1e-15;
        wp::float32 var_92;
        wp::float32 var_93;
        wp::float32 var_94;
        wp::float32 var_95;
        wp::float32 var_96;
        wp::float32 var_97;
        wp::float32 var_98;
        wp::range_t var_99;
        wp::int32 var_100;
        const wp::int32 var_101 = 0;
        bool var_102;
        wp::float32 var_103;
        const wp::int32 var_104 = 1;
        wp::int32 var_105;
        wp::float32 var_106;
        wp::float32 var_107;
        wp::float32 var_108;
        wp::float32 var_109;
        const wp::int32 var_110 = 1;
        wp::int32 var_111;
        const wp::int32 var_112 = 0;
        wp::range_t var_113;
        wp::int32 var_114;
        const wp::int32 var_115 = 0;
        bool var_116;
        wp::float32 var_117;
        const wp::int32 var_118 = 1;
        wp::int32 var_119;
        wp::float32 var_120;
        wp::float32 var_121;
        wp::float32 var_122;
        wp::float32 var_123;
        const wp::int32 var_124 = 0;
        bool var_125;
        const wp::int32 var_126 = 0;
        bool var_127;
        bool var_128;
        const wp::float32 var_129 = 1.0;
        const wp::int32 var_130 = 0;
        bool var_131;
        wp::float32 var_132;
        wp::float32 var_133;
        wp::float32 var_134;
        const wp::int32 var_135 = 0;
        bool var_136;
        wp::float32 var_137;
        wp::float32 var_138;
        wp::float32 var_139;
        wp::float32 var_140;
        wp::float32 var_141;
        bool var_142;
        wp::float32 var_143;
        wp::float32 var_144;
        wp::float32 var_145;
        wp::float32 var_146;
        wp::float32 var_147;
        wp::float32 var_148;
        const wp::int32 var_149 = 1;
        wp::int32 var_150;
        wp::int32 var_151;
        const wp::int32 var_152 = 2;
        wp::int32 var_153;
        wp::int32 var_154;
        const wp::int32 var_155 = 1;
        //---------
        // forward
        // def refresh_cone_coefficients(                                                         <L 13>
        // conid = wp.tid()                                                                       <L 30>
        var_0 = builtin_tid1d();
        // valid_out[conid] = 0                                                                   <L 31>
        wp::array_store(var_valid_out, var_0, var_1);
        // if conid >= min(nacon_in[0], naconmax_in):                                             <L 32>
        var_3 = wp::address(var_nacon_in, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::min(var_5, var_naconmax_in);
        var_6 = (var_0 >= var_4);
        if (var_6) {
            // return                                                                             <L 33>
            continue;
        }
        // worldid = contact_worldid_in[conid]                                                    <L 34>
        var_7 = wp::address(var_contact_worldid_in, var_0);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // if ctx_done_in[worldid]:                                                               <L 35>
        var_10 = wp::address(var_ctx_done_in, var_8);
        var_11 = wp::load(var_10);
        if (var_11) {
            // return                                                                             <L 36>
            continue;
        }
        var_12 = wp::load(var_10);
        // condim = contact_dim_in[conid]                                                         <L 37>
        var_13 = wp::address(var_contact_dim_in, var_0);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // if condim == 1:                                                                        <L 38>
        var_17 = (var_14 == var_16);
        if (var_17) {
            // return                                                                             <L 39>
            continue;
        }
        // if contact_dist_in[conid] - contact_includemargin_in[conid] >= 0.0:                    <L 40>
        var_18 = wp::address(var_contact_dist_in, var_0);
        var_19 = wp::address(var_contact_includemargin_in, var_0);
        var_21 = wp::load(var_18);
        var_22 = wp::load(var_19);
        var_20 = wp::sub(var_21, var_22);
        var_24 = (var_20 >= var_23);
        if (var_24) {
            // return                                                                             <L 41>
            continue;
        }
        // efcid0 = contact_efc_address_in[conid, 0]                                              <L 42>
        var_26 = wp::address(var_contact_efc_address_in, var_0, var_25);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // if efc_state_in[worldid, efcid0] != types.ConstraintState.CONE:                        <L 43>
        var_29 = wp::address(var_efc_state_in, var_8, var_27);
        var_32 = wp::load(var_29);
        var_31 = (var_32 != var_30);
        if (var_31) {
            // return                                                                             <L 44>
            continue;
        }
        // fri = contact_friction_in[conid]                                                       <L 46>
        var_33 = wp::address(var_contact_friction_in, var_0);
        var_35 = wp::load(var_33);
        var_34 = wp::copy(var_35);
        // mu = fri[0] * opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]            <L 47>
        var_37 = wp::extract(var_34, var_36);
        var_38 = &(var_opt_impratio_invsqrt.shape);
        var_41 = wp::load(var_38);
        var_40 = wp::extract(var_41, var_39);
        var_42 = wp::mod(var_8, var_40);
        var_43 = wp::address(var_opt_impratio_invsqrt, var_42);
        var_45 = wp::load(var_43);
        var_44 = wp::mul(var_37, var_45);
        // mu2 = mu * mu                                                                          <L 48>
        var_46 = wp::mul(var_44, var_44);
        // dm = math.safe_div(efc_D_in[worldid, efcid0], mu2 * (1.0 + mu2))                       <L 49>
        var_47 = wp::address(var_efc_D_in, var_8, var_27);
        var_49 = wp::add(var_48, var_46);
        var_50 = wp::mul(var_46, var_49);
        var_52 = wp::load(var_47);
        var_51 = safe_div_0(var_52, var_50);
        // if dm == 0.0:                                                                          <L 50>
        var_54 = (var_51 == var_53);
        if (var_54) {
            // return                                                                             <L 51>
            continue;
        }
        // n = ctx_Jaref_in[worldid, efcid0] * mu                                                 <L 52>
        var_55 = wp::address(var_ctx_Jaref_in, var_8, var_27);
        var_57 = wp::load(var_55);
        var_56 = wp::mul(var_57, var_44);
        // u = types.vec6(n, 0.0, 0.0, 0.0, 0.0, 0.0)                                             <L 53>
        var_63 = wp::vec_t<6, wp::float32>({var_56, var_58, var_59, var_60, var_61, var_62});
        // tt = float(0.0)                                                                        <L 54>
        var_65 = wp::float(var_64);
        // for j in range(1, condim):                                                             <L 55>
        var_67 = wp::range(var_66, var_14);
        start_for_6:;
            if (iter_cmp(var_67) == 0) goto end_for_6;
            var_68 = wp::iter_next(var_67);
            // efcidj = contact_efc_address_in[conid, j]                                          <L 56>
            var_69 = wp::address(var_contact_efc_address_in, var_0, var_68);
            var_71 = wp::load(var_69);
            var_70 = wp::copy(var_71);
            // uj = ctx_Jaref_in[worldid, efcidj] * fri[j - 1]                                    <L 57>
            var_72 = wp::address(var_ctx_Jaref_in, var_8, var_70);
            var_74 = wp::sub(var_68, var_73);
            var_75 = wp::extract(var_34, var_74);
            var_77 = wp::load(var_72);
            var_76 = wp::mul(var_77, var_75);
            // tt += uj * uj                                                                      <L 58>
            var_78 = wp::mul(var_76, var_76);
            var_79 = wp::add(var_65, var_78);
            // u[j] = uj                                                                          <L 59>
            wp::assign_inplace(var_63, var_68, var_76);
            wp::assign(var_65, var_79);
            goto start_for_6;
        end_for_6:;
        // if tt <= 0.0:                                                                          <L 60>
        var_81 = (var_65 <= var_80);
        if (var_81) {
            // t = 0.0                                                                            <L 61>
        }
        if (!var_81) {
            // t = wp.sqrt(tt)                                                                    <L 63>
            var_83 = wp::sqrt(var_65);
        }
        var_84 = wp::where(var_81, var_82, var_83);
        // t = wp.max(t, types.MJ_MINVAL)                                                         <L 64>
        var_87 = wp::max(var_84, var_86);
        // ttt = wp.max(t * t * t, types.MJ_MINVAL)                                               <L 65>
        var_88 = wp::mul(var_87, var_87);
        var_89 = wp::mul(var_88, var_87);
        var_92 = wp::max(var_89, var_91);
        // mu_over_t = math.safe_div(mu, t)                                                       <L 66>
        var_93 = safe_div_0(var_44, var_87);
        // mu_n_over_ttt = mu * math.safe_div(n, ttt)                                             <L 67>
        var_94 = safe_div_0(var_56, var_92);
        var_95 = wp::mul(var_44, var_94);
        // mu2_minus_mu_n_over_t = mu2 - mu * math.safe_div(n, t)                                 <L 68>
        var_96 = safe_div_0(var_56, var_87);
        var_97 = wp::mul(var_44, var_96);
        var_98 = wp::sub(var_46, var_97);
        // for dim1id in range(condim):                                                           <L 70>
        var_99 = wp::range(var_14);
        start_for_8:;
            if (iter_cmp(var_99) == 0) goto end_for_8;
            var_100 = wp::iter_next(var_99);
            // if dim1id == 0:                                                                    <L 71>
            var_102 = (var_100 == var_101);
            if (var_102) {
                // dm_fri1 = dm * mu                                                              <L 72>
                var_103 = wp::mul(var_51, var_44);
            }
            if (!var_102) {
                // dm_fri1 = dm * fri[dim1id - 1]                                                 <L 74>
                var_105 = wp::sub(var_100, var_104);
                var_106 = wp::extract(var_34, var_105);
                var_107 = wp::mul(var_51, var_106);
            }
            var_108 = wp::where(var_102, var_103, var_107);
            // ui = u[dim1id]                                                                     <L 75>
            var_109 = wp::extract(var_63, var_100);
            // for dim2id in range(0, dim1id + 1):                                                <L 76>
            var_111 = wp::add(var_100, var_110);
            var_113 = wp::range(var_112, var_111);
            start_for_10:;
                if (iter_cmp(var_113) == 0) goto end_for_10;
                var_114 = wp::iter_next(var_113);
                // if dim2id == 0:                                                                <L 77>
                var_116 = (var_114 == var_115);
                if (var_116) {
                    // dm_fri12 = dm_fri1 * mu                                                    <L 78>
                    var_117 = wp::mul(var_108, var_44);
                }
                if (!var_116) {
                    // dm_fri12 = dm_fri1 * fri[dim2id - 1]                                       <L 80>
                    var_119 = wp::sub(var_114, var_118);
                    var_120 = wp::extract(var_34, var_119);
                    var_121 = wp::mul(var_108, var_120);
                }
                var_122 = wp::where(var_116, var_117, var_121);
                // uj = u[dim2id]                                                                 <L 81>
                var_123 = wp::extract(var_63, var_114);
                // if dim1id == 0 and dim2id == 0:                                                <L 82>
                var_125 = (var_100 == var_124);
                var_127 = (var_114 == var_126);
                var_128 = var_125 && var_127;
                if (var_128) {
                    // hcone = 1.0                                                                <L 83>
                }
                if (!var_128) {
                    // elif dim1id == 0:                                                          <L 84>
                    var_131 = (var_100 == var_130);
                    if (var_131) {
                        // hcone = -mu_over_t * uj                                                <L 85>
                        var_132 = wp::neg(var_93);
                        var_133 = wp::mul(var_132, var_123);
                    }
                    var_134 = wp::where(var_131, var_133, var_129);
                    if (!var_131) {
                        // elif dim2id == 0:                                                      <L 86>
                        var_136 = (var_114 == var_135);
                        if (var_136) {
                            // hcone = -mu_over_t * ui                                            <L 87>
                            var_137 = wp::neg(var_93);
                            var_138 = wp::mul(var_137, var_109);
                        }
                        var_139 = wp::where(var_136, var_138, var_134);
                        if (!var_136) {
                            // hcone = mu_n_over_ttt * ui * uj                                    <L 89>
                            var_140 = wp::mul(var_95, var_109);
                            var_141 = wp::mul(var_140, var_123);
                            // if dim1id == dim2id:                                               <L 90>
                            var_142 = (var_100 == var_114);
                            if (var_142) {
                                // hcone += mu2_minus_mu_n_over_t                                 <L 91>
                                var_143 = wp::add(var_141, var_98);
                            }
                            var_144 = wp::where(var_142, var_143, var_141);
                        }
                        var_145 = wp::where(var_136, var_139, var_144);
                    }
                    var_146 = wp::where(var_131, var_134, var_145);
                }
                var_147 = wp::where(var_128, var_129, var_146);
                // hcone *= dm_fri12                                                              <L 92>
                var_148 = wp::mul(var_147, var_122);
                // coefficients_out[conid, dim1id * (dim1id + 1) // 2 + dim2id] = hcone           <L 93>
                var_150 = wp::add(var_100, var_149);
                var_151 = wp::mul(var_100, var_150);
                var_153 = wp::floordiv(var_151, var_152);
                var_154 = wp::add(var_153, var_114);
                wp::array_store(var_coefficients_out, var_0, var_154, var_148);
                wp::assign(var_76, var_123);
                goto start_for_10;
            end_for_10:;
            goto start_for_8;
        end_for_8:;
        // valid_out[conid] = 1                                                                   <L 94>
        wp::array_store(var_valid_out, var_0, var_155);
    }
}

