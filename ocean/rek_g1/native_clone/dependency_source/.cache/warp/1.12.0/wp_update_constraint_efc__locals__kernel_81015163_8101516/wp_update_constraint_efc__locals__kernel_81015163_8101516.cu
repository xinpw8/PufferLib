
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
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    bool var_1;
    const wp::float32 var_2 = 1e-15;
    const wp::float32 var_3 = 1e-15;
    wp::float32 var_4;
    wp::float32 var_5;
    //---------
    // dual vars
    wp::float32 adj_0 = {};
    bool adj_1 = {};
    wp::float32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::float32 adj_4 = {};
    wp::float32 adj_5 = {};
    //---------
    // forward
    // def safe_div(x: Any, y: Any) -> Any:                                                   <L 1>
    // return x / wp.where(y != 0.0, y, types.MJ_MINVAL)                                      <L 2>
    var_1 = (var_y != var_0);
    var_4 = wp::where(var_1, var_y, var_3);
    var_5 = wp::div(var_x, var_4);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_5 += adj_ret;
    wp::adj_div(var_x, var_4, var_5, adj_x, adj_4, adj_5);
    wp::adj_where(var_1, var_y, var_3, adj_1, adj_y, adj_3, adj_4);
    // adj: return x / wp.where(y != 0.0, y, types.MJ_MINVAL)                                 <L 2>
    // adj: def safe_div(x: Any, y: Any) -> Any:                                              <L 1>
    return;
}



extern "C" __global__ void update_constraint_efc__locals__kernel_9a9ed9bb_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_opt_impratio_invsqrt,
    wp::array_t<wp::int32> var_ne_in,
    wp::array_t<wp::int32> var_nf_in,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::int32> var_efc_type_in,
    wp::array_t<wp::int32> var_efc_id_in,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::float32> var_efc_frictionloss_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::float32> var_ctx_Jaref_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_efc_force_out,
    wp::array_t<wp::int32> var_efc_state_out,
    wp::array_t<wp::float32> var_ctx_cost_out,
    wp::array_t<wp::int32> var_changed_ids_out,
    wp::array_t<wp::int32> var_changed_count_out)
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
        const bool var_8 = false;
        wp::float32* var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        const wp::int32 var_21 = 0;
        bool var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        const wp::int32 var_25 = 1;
        const wp::float32 var_26 = 0.5;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        wp::float32 var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        bool var_33;
        wp::float32* var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        wp::float32 var_38;
        bool var_39;
        const wp::int32 var_40 = 2;
        wp::float32 var_41;
        const wp::float32 var_42 = 0.5;
        wp::float32 var_43;
        wp::float32 var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::int32 var_47;
        bool var_48;
        wp::float32 var_49;
        const wp::int32 var_50 = 3;
        wp::float32 var_51;
        const wp::float32 var_52 = 0.5;
        wp::float32 var_53;
        wp::float32 var_54;
        wp::float32 var_55;
        wp::float32 var_56;
        wp::int32 var_57;
        wp::float32 var_58;
        wp::float32 var_59;
        const wp::int32 var_60 = 1;
        const wp::float32 var_61 = 0.5;
        wp::float32 var_62;
        wp::float32 var_63;
        wp::float32 var_64;
        wp::float32 var_65;
        wp::int32 var_66;
        wp::int32 var_67;
        wp::int32 var_68;
        wp::int32* var_69;
        const wp::int32 var_70 = 7;
        bool var_71;
        wp::int32 var_72;
        const wp::float32 var_73 = 0.0;
        bool var_74;
        const wp::float32 var_75 = 0.0;
        const wp::int32 var_76 = 0;
        wp::int32 var_77;
        wp::float32 var_78;
        wp::float32 var_79;
        const wp::int32 var_80 = 1;
        const wp::float32 var_81 = 0.5;
        wp::float32 var_82;
        wp::float32 var_83;
        wp::float32 var_84;
        wp::float32 var_85;
        wp::int32 var_86;
        wp::int32 var_87;
        wp::int32* var_88;
        wp::int32 var_89;
        wp::int32 var_90;
        const wp::int32 var_91 = 0;
        wp::int32* var_92;
        bool var_93;
        wp::int32 var_94;
        wp::int32* var_95;
        wp::int32 var_96;
        wp::int32 var_97;
        wp::vec_t<5, wp::float32>* var_98;
        wp::vec_t<5, wp::float32> var_99;
        wp::vec_t<5, wp::float32> var_100;
        const wp::int32 var_101 = 0;
        wp::float32 var_102;
        wp::shape_t* var_103;
        const wp::int32 var_104 = 0;
        wp::int32 var_105;
        wp::shape_t var_106;
        wp::int32 var_107;
        wp::float32* var_108;
        wp::float32 var_109;
        wp::float32 var_110;
        const wp::int32 var_111 = 0;
        wp::int32* var_112;
        wp::int32 var_113;
        wp::int32 var_114;
        const wp::int32 var_115 = 0;
        bool var_116;
        wp::float32* var_117;
        wp::float32 var_118;
        wp::float32 var_119;
        const wp::float32 var_120 = 0.0;
        wp::float32 var_121;
        const wp::float32 var_122 = 0.0;
        wp::float32 var_123;
        const wp::int32 var_124 = 1;
        wp::range_t var_125;
        wp::int32 var_126;
        wp::int32* var_127;
        wp::int32 var_128;
        wp::int32 var_129;
        const wp::int32 var_130 = 0;
        bool var_131;
        const wp::int32 var_132 = 1;
        wp::int32 var_133;
        wp::float32 var_134;
        wp::float32* var_135;
        wp::float32 var_136;
        wp::float32 var_137;
        wp::float32 var_138;
        wp::float32 var_139;
        bool var_140;
        wp::float32 var_141;
        wp::float32 var_142;
        const wp::float32 var_143 = 0.0;
        bool var_144;
        const wp::float32 var_145 = 0.0;
        wp::float32 var_146;
        wp::float32 var_147;
        wp::float32 var_148;
        bool var_149;
        const wp::float32 var_150 = 0.0;
        bool var_151;
        const wp::float32 var_152 = 0.0;
        bool var_153;
        bool var_154;
        bool var_155;
        const wp::float32 var_156 = 0.0;
        const wp::int32 var_157 = 0;
        wp::int32 var_158;
        wp::float32 var_159;
        wp::float32 var_160;
        const wp::float32 var_161 = 0.0;
        bool var_162;
        const wp::float32 var_163 = 0.0;
        bool var_164;
        const wp::float32 var_165 = 0.0;
        bool var_166;
        bool var_167;
        bool var_168;
        wp::float32 var_169;
        wp::float32 var_170;
        const wp::int32 var_171 = 1;
        const wp::float32 var_172 = 0.5;
        wp::float32 var_173;
        wp::float32 var_174;
        wp::float32 var_175;
        wp::float32 var_176;
        wp::int32 var_177;
        wp::float32* var_178;
        wp::float32 var_179;
        const wp::float32 var_180 = 1.0;
        wp::float32 var_181;
        wp::float32 var_182;
        wp::float32 var_183;
        wp::float32 var_184;
        wp::float32 var_185;
        wp::float32 var_186;
        wp::float32 var_187;
        wp::float32 var_188;
        wp::float32 var_189;
        wp::float32 var_190;
        bool var_191;
        const wp::float32 var_192 = 0.5;
        wp::float32 var_193;
        wp::float32 var_194;
        wp::float32 var_195;
        wp::float32 var_196;
        wp::float32 var_197;
        wp::float32 var_198;
        wp::float32 var_199;
        const wp::int32 var_200 = 4;
        wp::int32 var_201;
        wp::int32 var_202;
        wp::int32 var_203;
        wp::int32 var_204;
        wp::int32 var_205;
        const bool var_206 = false;
        //---------
        // forward
        // def kernel(                                                                            <L 1810>
        // worldid, efcid = wp.tid()                                                              <L 1836>
        builtin_tid2d(var_0, var_1);
        // if efcid >= nefc_in[worldid]:                                                          <L 1838>
        var_2 = wp::address(var_nefc_in, var_0);
        var_4 = wp::load(var_2);
        var_3 = (var_1 >= var_4);
        if (var_3) {
            // return                                                                             <L 1839>
            continue;
        }
        // if ctx_done_in[worldid]:                                                               <L 1841>
        var_5 = wp::address(var_ctx_done_in, var_0);
        var_6 = wp::load(var_5);
        if (var_6) {
            // return                                                                             <L 1842>
            continue;
        }
        var_7 = wp::load(var_5);
        // if wp.static(TRACK_CHANGES):                                                           <L 1845>
        // efc_D = efc_D_in[worldid, efcid]                                                       <L 1848>
        var_9 = wp::address(var_efc_D_in, var_0, var_1);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // Jaref = ctx_Jaref_in[worldid, efcid]                                                   <L 1849>
        var_12 = wp::address(var_ctx_Jaref_in, var_0, var_1);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // ne = ne_in[worldid]                                                                    <L 1851>
        var_15 = wp::address(var_ne_in, var_0);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // nf = nf_in[worldid]                                                                    <L 1852>
        var_18 = wp::address(var_nf_in, var_0);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // new_state = types.ConstraintState.SATISFIED.value                                      <L 1854>
        // if efcid < ne:                                                                         <L 1856>
        var_22 = (var_1 < var_16);
        if (var_22) {
            // efc_force_out[worldid, efcid] = -efc_D * Jaref                                     <L 1858>
            var_23 = wp::neg(var_10);
            var_24 = wp::mul(var_23, var_13);
            wp::array_store(var_efc_force_out, var_0, var_1, var_24);
            // new_state = types.ConstraintState.QUADRATIC.value                                  <L 1859>
            // wp.atomic_add(ctx_cost_out, worldid, 0.5 * efc_D * Jaref * Jaref)                  <L 1860>
            var_27 = wp::mul(var_26, var_10);
            var_28 = wp::mul(var_27, var_13);
            var_29 = wp::mul(var_28, var_13);
            var_30 = wp::atomic_add(var_ctx_cost_out, var_0, var_29);
        }
        var_31 = wp::where(var_22, var_25, var_21);
        if (!var_22) {
            // elif efcid < ne + nf:                                                              <L 1861>
            var_32 = wp::add(var_16, var_19);
            var_33 = (var_1 < var_32);
            if (var_33) {
                // f = efc_frictionloss_in[worldid, efcid]                                        <L 1863>
                var_34 = wp::address(var_efc_frictionloss_in, var_0, var_1);
                var_36 = wp::load(var_34);
                var_35 = wp::copy(var_36);
                // rf = math.safe_div(f, efc_D)                                                   <L 1864>
                var_37 = safe_div_0(var_35, var_10);
                // if Jaref <= -rf:                                                               <L 1865>
                var_38 = wp::neg(var_37);
                var_39 = (var_13 <= var_38);
                if (var_39) {
                    // efc_force_out[worldid, efcid] = f                                          <L 1866>
                    wp::array_store(var_efc_force_out, var_0, var_1, var_35);
                    // new_state = types.ConstraintState.LINEARNEG.value                          <L 1867>
                    // wp.atomic_add(ctx_cost_out, worldid, -f * (0.5 * rf + Jaref))              <L 1868>
                    var_41 = wp::neg(var_35);
                    var_43 = wp::mul(var_42, var_37);
                    var_44 = wp::add(var_43, var_13);
                    var_45 = wp::mul(var_41, var_44);
                    var_46 = wp::atomic_add(var_ctx_cost_out, var_0, var_45);
                }
                var_47 = wp::where(var_39, var_40, var_31);
                if (!var_39) {
                    // elif Jaref >= rf:                                                          <L 1869>
                    var_48 = (var_13 >= var_37);
                    if (var_48) {
                        // efc_force_out[worldid, efcid] = -f                                     <L 1870>
                        var_49 = wp::neg(var_35);
                        wp::array_store(var_efc_force_out, var_0, var_1, var_49);
                        // new_state = types.ConstraintState.LINEARPOS.value                      <L 1871>
                        // wp.atomic_add(ctx_cost_out, worldid, -f * (0.5 * rf - Jaref))          <L 1872>
                        var_51 = wp::neg(var_35);
                        var_53 = wp::mul(var_52, var_37);
                        var_54 = wp::sub(var_53, var_13);
                        var_55 = wp::mul(var_51, var_54);
                        var_56 = wp::atomic_add(var_ctx_cost_out, var_0, var_55);
                    }
                    var_57 = wp::where(var_48, var_50, var_47);
                    if (!var_48) {
                        // efc_force_out[worldid, efcid] = -efc_D * Jaref                         <L 1874>
                        var_58 = wp::neg(var_10);
                        var_59 = wp::mul(var_58, var_13);
                        wp::array_store(var_efc_force_out, var_0, var_1, var_59);
                        // new_state = types.ConstraintState.QUADRATIC.value                      <L 1875>
                        // wp.atomic_add(ctx_cost_out, worldid, 0.5 * efc_D * Jaref * Jaref)       <L 1876>
                        var_62 = wp::mul(var_61, var_10);
                        var_63 = wp::mul(var_62, var_13);
                        var_64 = wp::mul(var_63, var_13);
                        var_65 = wp::atomic_add(var_ctx_cost_out, var_0, var_64);
                    }
                    var_66 = wp::where(var_48, var_57, var_60);
                }
                var_67 = wp::where(var_39, var_47, var_66);
            }
            var_68 = wp::where(var_33, var_67, var_31);
            if (!var_33) {
                // elif efc_type_in[worldid, efcid] != types.ConstraintType.CONTACT_ELLIPTIC:       <L 1877>
                var_69 = wp::address(var_efc_type_in, var_0, var_1);
                var_72 = wp::load(var_69);
                var_71 = (var_72 != var_70);
                if (var_71) {
                    // if Jaref >= 0.0:                                                           <L 1879>
                    var_74 = (var_13 >= var_73);
                    if (var_74) {
                        // efc_force_out[worldid, efcid] = 0.0                                    <L 1880>
                        wp::array_store(var_efc_force_out, var_0, var_1, var_75);
                        // new_state = types.ConstraintState.SATISFIED.value                      <L 1881>
                    }
                    var_77 = wp::where(var_74, var_76, var_68);
                    if (!var_74) {
                        // efc_force_out[worldid, efcid] = -efc_D * Jaref                         <L 1883>
                        var_78 = wp::neg(var_10);
                        var_79 = wp::mul(var_78, var_13);
                        wp::array_store(var_efc_force_out, var_0, var_1, var_79);
                        // new_state = types.ConstraintState.QUADRATIC.value                      <L 1884>
                        // wp.atomic_add(ctx_cost_out, worldid, 0.5 * efc_D * Jaref * Jaref)       <L 1885>
                        var_82 = wp::mul(var_81, var_10);
                        var_83 = wp::mul(var_82, var_13);
                        var_84 = wp::mul(var_83, var_13);
                        var_85 = wp::atomic_add(var_ctx_cost_out, var_0, var_84);
                    }
                    var_86 = wp::where(var_74, var_77, var_80);
                }
                var_87 = wp::where(var_71, var_86, var_68);
                if (!var_71) {
                    // conid = efc_id_in[worldid, efcid]                                          <L 1887>
                    var_88 = wp::address(var_efc_id_in, var_0, var_1);
                    var_90 = wp::load(var_88);
                    var_89 = wp::copy(var_90);
                    // if conid >= nacon_in[0]:                                                   <L 1889>
                    var_92 = wp::address(var_nacon_in, var_91);
                    var_94 = wp::load(var_92);
                    var_93 = (var_89 >= var_94);
                    if (var_93) {
                        // return                                                                 <L 1890>
                        continue;
                    }
                    // dim = contact_dim_in[conid]                                                <L 1892>
                    var_95 = wp::address(var_contact_dim_in, var_89);
                    var_97 = wp::load(var_95);
                    var_96 = wp::copy(var_97);
                    // friction = contact_friction_in[conid]                                      <L 1893>
                    var_98 = wp::address(var_contact_friction_in, var_89);
                    var_100 = wp::load(var_98);
                    var_99 = wp::copy(var_100);
                    // mu = friction[0] * opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]       <L 1894>
                    var_102 = wp::extract(var_99, var_101);
                    var_103 = &(var_opt_impratio_invsqrt.shape);
                    var_106 = wp::load(var_103);
                    var_105 = wp::extract(var_106, var_104);
                    var_107 = wp::mod(var_0, var_105);
                    var_108 = wp::address(var_opt_impratio_invsqrt, var_107);
                    var_110 = wp::load(var_108);
                    var_109 = wp::mul(var_102, var_110);
                    // efcid0 = contact_efc_address_in[conid, 0]                                  <L 1896>
                    var_112 = wp::address(var_contact_efc_address_in, var_89, var_111);
                    var_114 = wp::load(var_112);
                    var_113 = wp::copy(var_114);
                    // if efcid0 < 0:                                                             <L 1897>
                    var_116 = (var_113 < var_115);
                    if (var_116) {
                        // return                                                                 <L 1898>
                        continue;
                    }
                    // N = ctx_Jaref_in[worldid, efcid0] * mu                                     <L 1900>
                    var_117 = wp::address(var_ctx_Jaref_in, var_0, var_113);
                    var_119 = wp::load(var_117);
                    var_118 = wp::mul(var_119, var_109);
                    // ufrictionj = float(0.0)                                                    <L 1902>
                    var_121 = wp::float(var_120);
                    // TT = float(0.0)                                                            <L 1903>
                    var_123 = wp::float(var_122);
                    // for j in range(1, dim):                                                    <L 1904>
                    var_125 = wp::range(var_124, var_96);
                    start_for_4:;
                        if (iter_cmp(var_125) == 0) goto end_for_4;
                        var_126 = wp::iter_next(var_125);
                        // efcidj = contact_efc_address_in[conid, j]                              <L 1905>
                        var_127 = wp::address(var_contact_efc_address_in, var_89, var_126);
                        var_129 = wp::load(var_127);
                        var_128 = wp::copy(var_129);
                        // if efcidj < 0:                                                         <L 1906>
                        var_131 = (var_128 < var_130);
                        if (var_131) {
                            // return                                                             <L 1907>
                            continue;
                        }
                        // frictionj = friction[j - 1]                                            <L 1908>
                        var_133 = wp::sub(var_126, var_132);
                        var_134 = wp::extract(var_99, var_133);
                        // uj = ctx_Jaref_in[worldid, efcidj] * frictionj                         <L 1909>
                        var_135 = wp::address(var_ctx_Jaref_in, var_0, var_128);
                        var_137 = wp::load(var_135);
                        var_136 = wp::mul(var_137, var_134);
                        // TT += uj * uj                                                          <L 1910>
                        var_138 = wp::mul(var_136, var_136);
                        var_139 = wp::add(var_123, var_138);
                        // if efcid == efcidj:                                                    <L 1911>
                        var_140 = (var_1 == var_128);
                        if (var_140) {
                            // ufrictionj = uj * frictionj                                        <L 1912>
                            var_141 = wp::mul(var_136, var_134);
                        }
                        var_142 = wp::where(var_140, var_141, var_121);
                        wp::assign(var_121, var_142);
                        wp::assign(var_123, var_139);
                        goto start_for_4;
                    end_for_4:;
                    // if TT <= 0.0:                                                              <L 1914>
                    var_144 = (var_123 <= var_143);
                    if (var_144) {
                        // T = 0.0                                                                <L 1915>
                    }
                    if (!var_144) {
                        // T = wp.sqrt(TT)                                                        <L 1917>
                        var_146 = wp::sqrt(var_123);
                    }
                    var_147 = wp::where(var_144, var_145, var_146);
                    // if (N >= mu * T) or ((T <= 0.0) and (N >= 0.0)):                           <L 1920>
                    var_148 = wp::mul(var_109, var_147);
                    var_149 = (var_118 >= var_148);
                    var_151 = (var_147 <= var_150);
                    var_153 = (var_118 >= var_152);
                    var_154 = var_151 && var_153;
                    var_155 = var_149 || var_154;
                    if (var_155) {
                        // efc_force_out[worldid, efcid] = 0.0                                    <L 1921>
                        wp::array_store(var_efc_force_out, var_0, var_1, var_156);
                        // new_state = types.ConstraintState.SATISFIED.value                      <L 1922>
                    }
                    var_158 = wp::where(var_155, var_157, var_87);
                    if (!var_155) {
                        // elif (mu * N + T <= 0.0) or ((T <= 0.0) and (N < 0.0)):                <L 1924>
                        var_159 = wp::mul(var_109, var_118);
                        var_160 = wp::add(var_159, var_147);
                        var_162 = (var_160 <= var_161);
                        var_164 = (var_147 <= var_163);
                        var_166 = (var_118 < var_165);
                        var_167 = var_164 && var_166;
                        var_168 = var_162 || var_167;
                        if (var_168) {
                            // efc_force_out[worldid, efcid] = -efc_D * Jaref                     <L 1925>
                            var_169 = wp::neg(var_10);
                            var_170 = wp::mul(var_169, var_13);
                            wp::array_store(var_efc_force_out, var_0, var_1, var_170);
                            // new_state = types.ConstraintState.QUADRATIC.value                  <L 1926>
                            // wp.atomic_add(ctx_cost_out, worldid, 0.5 * efc_D * Jaref * Jaref)       <L 1927>
                            var_173 = wp::mul(var_172, var_10);
                            var_174 = wp::mul(var_173, var_13);
                            var_175 = wp::mul(var_174, var_13);
                            var_176 = wp::atomic_add(var_ctx_cost_out, var_0, var_175);
                        }
                        var_177 = wp::where(var_168, var_171, var_158);
                        if (!var_168) {
                            // dm = math.safe_div(efc_D_in[worldid, efcid0], mu * mu * (1.0 + mu * mu))       <L 1930>
                            var_178 = wp::address(var_efc_D_in, var_0, var_113);
                            var_179 = wp::mul(var_109, var_109);
                            var_181 = wp::mul(var_109, var_109);
                            var_182 = wp::add(var_180, var_181);
                            var_183 = wp::mul(var_179, var_182);
                            var_185 = wp::load(var_178);
                            var_184 = safe_div_0(var_185, var_183);
                            // nmt = N - mu * T                                                   <L 1931>
                            var_186 = wp::mul(var_109, var_147);
                            var_187 = wp::sub(var_118, var_186);
                            // force = -dm * nmt * mu                                             <L 1933>
                            var_188 = wp::neg(var_184);
                            var_189 = wp::mul(var_188, var_187);
                            var_190 = wp::mul(var_189, var_109);
                            // if efcid == efcid0:                                                <L 1935>
                            var_191 = (var_1 == var_113);
                            if (var_191) {
                                // efc_force_out[worldid, efcid] = force                          <L 1936>
                                wp::array_store(var_efc_force_out, var_0, var_1, var_190);
                                // wp.atomic_add(ctx_cost_out, worldid, 0.5 * dm * nmt * nmt)       <L 1937>
                                var_193 = wp::mul(var_192, var_184);
                                var_194 = wp::mul(var_193, var_187);
                                var_195 = wp::mul(var_194, var_187);
                                var_196 = wp::atomic_add(var_ctx_cost_out, var_0, var_195);
                            }
                            if (!var_191) {
                                // efc_force_out[worldid, efcid] = -math.safe_div(force, T) * ufrictionj       <L 1939>
                                var_197 = safe_div_0(var_190, var_147);
                                var_198 = wp::neg(var_197);
                                var_199 = wp::mul(var_198, var_121);
                                wp::array_store(var_efc_force_out, var_0, var_1, var_199);
                            }
                            // new_state = types.ConstraintState.CONE.value                       <L 1941>
                        }
                        var_201 = wp::where(var_168, var_177, var_200);
                    }
                    var_202 = wp::where(var_155, var_158, var_201);
                }
                var_203 = wp::where(var_71, var_87, var_202);
            }
            var_204 = wp::where(var_33, var_68, var_203);
        }
        var_205 = wp::where(var_22, var_31, var_204);
        // efc_state_out[worldid, efcid] = new_state                                              <L 1943>
        wp::array_store(var_efc_state_out, var_0, var_1, var_205);
        // if wp.static(TRACK_CHANGES):                                                           <L 1945>
    }
}

