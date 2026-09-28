
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



extern "C" __global__ void build_inverse_positions_74938071_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_nacon,
    wp::int32 var_naconmax,
    wp::array_t<wp::float32> var_dist,
    wp::array_t<wp::float32> var_margin,
    wp::array_t<wp::int32> var_condim,
    wp::array_t<wp::int32> var_world,
    wp::array_t<wp::int32> var_address,
    wp::array_t<wp::int32> var_rownnz,
    wp::array_t<wp::int32> var_rowadr,
    wp::array_t<wp::int32> var_colind,
    wp::array_t<wp::int32> var_inverse,
    wp::array_t<wp::int32> var_status)
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
        const wp::int32 var_2 = 0;
        bool var_3;
        const wp::int32 var_4 = 0;
        bool var_5;
        const wp::int32 var_6 = 0;
        wp::int32* var_7;
        const wp::int32 var_8 = 0;
        bool var_9;
        wp::int32 var_10;
        const wp::int32 var_11 = 0;
        wp::int32* var_12;
        bool var_13;
        wp::int32 var_14;
        bool var_15;
        bool var_16;
        const wp::int32 var_17 = 0;
        const wp::int32 var_18 = 512;
        wp::int32 var_19;
        const wp::int32 var_20 = 0;
        wp::int32* var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        bool var_24;
        const wp::int32 var_25 = 1;
        const wp::int32 var_26 = -1;
        wp::int32* var_27;
        const wp::int32 var_28 = 1;
        bool var_29;
        wp::int32 var_30;
        wp::float32* var_31;
        wp::float32* var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        const wp::float32 var_36 = 0.0;
        bool var_37;
        bool var_38;
        wp::int32* var_39;
        wp::int32 var_40;
        wp::int32 var_41;
        const wp::int32 var_42 = 0;
        bool var_43;
        wp::shape_t* var_44;
        const wp::int32 var_45 = 0;
        wp::int32 var_46;
        wp::shape_t var_47;
        bool var_48;
        bool var_49;
        const wp::int32 var_50 = 0;
        const wp::int32 var_51 = 1;
        wp::int32 var_52;
        wp::int32* var_53;
        wp::int32 var_54;
        wp::int32 var_55;
        const wp::int32 var_56 = 3;
        bool var_57;
        const wp::int32 var_58 = 4;
        bool var_59;
        const wp::int32 var_60 = 6;
        bool var_61;
        bool var_62;
        const wp::int32 var_63 = 0;
        const wp::int32 var_64 = 2;
        wp::int32 var_65;
        const wp::int32 var_66 = 0;
        wp::int32* var_67;
        wp::int32 var_68;
        wp::int32 var_69;
        const wp::int32 var_70 = 0;
        bool var_71;
        wp::shape_t* var_72;
        const wp::int32 var_73 = 1;
        wp::int32 var_74;
        wp::shape_t var_75;
        bool var_76;
        bool var_77;
        const wp::int32 var_78 = 0;
        const wp::int32 var_79 = 4;
        wp::int32 var_80;
        wp::int32* var_81;
        wp::int32 var_82;
        wp::int32 var_83;
        wp::int32* var_84;
        wp::int32 var_85;
        wp::int32 var_86;
        const wp::int32 var_87 = 0;
        bool var_88;
        wp::shape_t* var_89;
        const wp::int32 var_90 = 1;
        wp::int32 var_91;
        wp::shape_t var_92;
        bool var_93;
        const wp::int32 var_94 = 0;
        bool var_95;
        wp::int32 var_96;
        wp::shape_t* var_97;
        const wp::int32 var_98 = 2;
        wp::int32 var_99;
        wp::shape_t var_100;
        bool var_101;
        bool var_102;
        const wp::int32 var_103 = 0;
        const wp::int32 var_104 = 8;
        wp::int32 var_105;
        const wp::int32 var_106 = 1;
        const wp::int32 var_107 = -1;
        wp::int32 var_108;
        const wp::int32 var_109 = 0;
        wp::int32 var_110;
        wp::range_t var_111;
        wp::int32 var_112;
        const wp::int32 var_113 = 0;
        wp::int32 var_114;
        wp::int32* var_115;
        wp::int32 var_116;
        wp::int32 var_117;
        const wp::int32 var_118 = 0;
        bool var_119;
        wp::shape_t* var_120;
        const wp::int32 var_121 = 1;
        wp::int32 var_122;
        wp::shape_t var_123;
        bool var_124;
        bool var_125;
        const wp::int32 var_126 = 0;
        const wp::int32 var_127 = 16;
        wp::int32 var_128;
        bool var_129;
        wp::int32 var_130;
        const wp::int32 var_131 = 1;
        wp::int32 var_132;
        wp::int32 var_133;
        wp::int32 var_134;
        const wp::int32 var_135 = 1;
        bool var_136;
        const wp::int32 var_137 = 0;
        const wp::int32 var_138 = 32;
        wp::int32 var_139;
        const wp::int32 var_140 = 0;
        bool var_141;
        const wp::int32 var_142 = 1;
        wp::range_t var_143;
        wp::int32 var_144;
        wp::int32* var_145;
        wp::int32 var_146;
        wp::int32 var_147;
        const wp::int32 var_148 = 0;
        bool var_149;
        wp::shape_t* var_150;
        const wp::int32 var_151 = 1;
        wp::int32 var_152;
        wp::shape_t var_153;
        bool var_154;
        bool var_155;
        const wp::int32 var_156 = 0;
        const wp::int32 var_157 = 64;
        wp::int32 var_158;
        wp::int32* var_159;
        wp::int32 var_160;
        wp::int32 var_161;
        wp::int32* var_162;
        wp::int32 var_163;
        wp::int32 var_164;
        bool var_165;
        const wp::int32 var_166 = 0;
        bool var_167;
        wp::int32 var_168;
        wp::shape_t* var_169;
        const wp::int32 var_170 = 2;
        wp::int32 var_171;
        wp::shape_t var_172;
        bool var_173;
        bool var_174;
        const wp::int32 var_175 = 0;
        const wp::int32 var_176 = 128;
        wp::int32 var_177;
        wp::range_t var_178;
        wp::int32 var_179;
        const wp::int32 var_180 = 0;
        wp::int32 var_181;
        wp::int32* var_182;
        const wp::int32 var_183 = 0;
        wp::int32 var_184;
        wp::int32* var_185;
        bool var_186;
        wp::int32 var_187;
        wp::int32 var_188;
        const wp::int32 var_189 = 0;
        const wp::int32 var_190 = 256;
        wp::int32 var_191;
        wp::int32 var_192;
        wp::int32 var_193;
        //---------
        // forward
        // def build_inverse_positions(                                                           <L 6>
        // conid, dof = wp.tid()                                                                  <L 20>
        builtin_tid2d(var_0, var_1);
        // if conid == 0 and dof == 0 and (nacon[0] < 0 or nacon[0] > naconmax):                  <L 21>
        var_3 = (var_0 == var_2);
        var_5 = (var_1 == var_4);
        var_7 = wp::address(var_nacon, var_6);
        var_10 = wp::load(var_7);
        var_9 = (var_10 < var_8);
        var_12 = wp::address(var_nacon, var_11);
        var_14 = wp::load(var_12);
        var_13 = (var_14 > var_naconmax);
        var_15 = var_9 || var_13;
        var_16 = var_3 && var_5 && var_15;
        if (var_16) {
            // wp.atomic_or(status, 0, 512)                                                       <L 22>
            var_19 = wp::atomic_or(var_status, var_17, var_18);
        }
        // if conid >= wp.min(nacon[0], naconmax):                                                <L 23>
        var_21 = wp::address(var_nacon, var_20);
        var_23 = wp::load(var_21);
        var_22 = wp::min(var_23, var_naconmax);
        var_24 = (var_0 >= var_22);
        if (var_24) {
            // return                                                                             <L 24>
            continue;
        }
        // inverse[conid, dof] = -1                                                               <L 25>
        wp::array_store(var_inverse, var_0, var_1, var_26);
        // if condim[conid] == 1 or dist[conid] - margin[conid] >= 0.0:                           <L 27>
        var_27 = wp::address(var_condim, var_0);
        var_30 = wp::load(var_27);
        var_29 = (var_30 == var_28);
        var_31 = wp::address(var_dist, var_0);
        var_32 = wp::address(var_margin, var_0);
        var_34 = wp::load(var_31);
        var_35 = wp::load(var_32);
        var_33 = wp::sub(var_34, var_35);
        var_37 = (var_33 >= var_36);
        var_38 = var_29 || var_37;
        if (var_38) {
            // return                                                                             <L 28>
            continue;
        }
        // wid = world[conid]                                                                     <L 29>
        var_39 = wp::address(var_world, var_0);
        var_41 = wp::load(var_39);
        var_40 = wp::copy(var_41);
        // if wid < 0 or wid >= rownnz.shape[0]:                                                  <L 30>
        var_43 = (var_40 < var_42);
        var_44 = &(var_rownnz.shape);
        var_47 = wp::load(var_44);
        var_46 = wp::extract(var_47, var_45);
        var_48 = (var_40 >= var_46);
        var_49 = var_43 || var_48;
        if (var_49) {
            // wp.atomic_or(status, 0, 1)                                                         <L 31>
            var_52 = wp::atomic_or(var_status, var_50, var_51);
            // return                                                                             <L 32>
            continue;
        }
        // dimension = condim[conid]                                                              <L 33>
        var_53 = wp::address(var_condim, var_0);
        var_55 = wp::load(var_53);
        var_54 = wp::copy(var_55);
        // if dimension != 3 and dimension != 4 and dimension != 6:                               <L 34>
        var_57 = (var_54 != var_56);
        var_59 = (var_54 != var_58);
        var_61 = (var_54 != var_60);
        var_62 = var_57 && var_59 && var_61;
        if (var_62) {
            // wp.atomic_or(status, 0, 2)                                                         <L 35>
            var_65 = wp::atomic_or(var_status, var_63, var_64);
            // return                                                                             <L 36>
            continue;
        }
        // row = address[conid, 0]                                                                <L 37>
        var_67 = wp::address(var_address, var_0, var_66);
        var_69 = wp::load(var_67);
        var_68 = wp::copy(var_69);
        // if row < 0 or row >= rownnz.shape[1]:                                                  <L 38>
        var_71 = (var_68 < var_70);
        var_72 = &(var_rownnz.shape);
        var_75 = wp::load(var_72);
        var_74 = wp::extract(var_75, var_73);
        var_76 = (var_68 >= var_74);
        var_77 = var_71 || var_76;
        if (var_77) {
            // wp.atomic_or(status, 0, 4)                                                         <L 39>
            var_80 = wp::atomic_or(var_status, var_78, var_79);
            // return                                                                             <L 40>
            continue;
        }
        // count = rownnz[wid, row]                                                               <L 41>
        var_81 = wp::address(var_rownnz, var_40, var_68);
        var_83 = wp::load(var_81);
        var_82 = wp::copy(var_83);
        // offset = rowadr[wid, row]                                                              <L 42>
        var_84 = wp::address(var_rowadr, var_40, var_68);
        var_86 = wp::load(var_84);
        var_85 = wp::copy(var_86);
        // if count < 0 or count > inverse.shape[1] or offset < 0 or offset + count > colind.shape[2]:       <L 43>
        var_88 = (var_82 < var_87);
        var_89 = &(var_inverse.shape);
        var_92 = wp::load(var_89);
        var_91 = wp::extract(var_92, var_90);
        var_93 = (var_82 > var_91);
        var_95 = (var_85 < var_94);
        var_96 = wp::add(var_85, var_82);
        var_97 = &(var_colind.shape);
        var_100 = wp::load(var_97);
        var_99 = wp::extract(var_100, var_98);
        var_101 = (var_96 > var_99);
        var_102 = var_88 || var_93 || var_95 || var_101;
        if (var_102) {
            // wp.atomic_or(status, 0, 8)                                                         <L 44>
            var_105 = wp::atomic_or(var_status, var_103, var_104);
            // return                                                                             <L 45>
            continue;
        }
        // position = int(-1)                                                                     <L 46>
        var_108 = wp::int(var_107);
        // matches = int(0)                                                                       <L 47>
        var_110 = wp::int(var_109);
        // for k in range(count):                                                                 <L 48>
        var_111 = wp::range(var_82);
        start_for_6:;
            if (iter_cmp(var_111) == 0) goto end_for_6;
            var_112 = wp::iter_next(var_111);
            // col = colind[wid, 0, offset + k]                                                   <L 49>
            var_114 = wp::add(var_85, var_112);
            var_115 = wp::address(var_colind, var_40, var_113, var_114);
            var_117 = wp::load(var_115);
            var_116 = wp::copy(var_117);
            // if col < 0 or col >= inverse.shape[1]:                                             <L 50>
            var_119 = (var_116 < var_118);
            var_120 = &(var_inverse.shape);
            var_123 = wp::load(var_120);
            var_122 = wp::extract(var_123, var_121);
            var_124 = (var_116 >= var_122);
            var_125 = var_119 || var_124;
            if (var_125) {
                // wp.atomic_or(status, 0, 16)                                                    <L 51>
                var_128 = wp::atomic_or(var_status, var_126, var_127);
            }
            // if col == dof:                                                                     <L 52>
            var_129 = (var_116 == var_1);
            if (var_129) {
                // position = k                                                                   <L 53>
                var_130 = wp::copy(var_112);
                // matches += 1                                                                   <L 54>
                var_132 = wp::add(var_110, var_131);
            }
            var_133 = wp::where(var_129, var_130, var_108);
            var_134 = wp::where(var_129, var_132, var_110);
            wp::assign(var_108, var_133);
            wp::assign(var_110, var_134);
            goto start_for_6;
        end_for_6:;
        // if matches > 1:                                                                        <L 55>
        var_136 = (var_110 > var_135);
        if (var_136) {
            // wp.atomic_or(status, 0, 32)                                                        <L 58>
            var_139 = wp::atomic_or(var_status, var_137, var_138);
        }
        // inverse[conid, dof] = position                                                         <L 59>
        wp::array_store(var_inverse, var_0, var_1, var_108);
        // if dof == 0:                                                                           <L 60>
        var_141 = (var_1 == var_140);
        if (var_141) {
            // for axis in range(1, dimension):                                                   <L 62>
            var_143 = wp::range(var_142, var_54);
            start_for_8:;
                if (iter_cmp(var_143) == 0) goto end_for_8;
                var_144 = wp::iter_next(var_143);
                // other = address[conid, axis]                                                   <L 63>
                var_145 = wp::address(var_address, var_0, var_144);
                var_147 = wp::load(var_145);
                var_146 = wp::copy(var_147);
                // if other < 0 or other >= rownnz.shape[1]:                                      <L 64>
                var_149 = (var_146 < var_148);
                var_150 = &(var_rownnz.shape);
                var_153 = wp::load(var_150);
                var_152 = wp::extract(var_153, var_151);
                var_154 = (var_146 >= var_152);
                var_155 = var_149 || var_154;
                if (var_155) {
                    // wp.atomic_or(status, 0, 64)                                                <L 65>
                    var_158 = wp::atomic_or(var_status, var_156, var_157);
                }
                if (!var_155) {
                    // other_count = rownnz[wid, other]                                           <L 67>
                    var_159 = wp::address(var_rownnz, var_40, var_146);
                    var_161 = wp::load(var_159);
                    var_160 = wp::copy(var_161);
                    // other_offset = rowadr[wid, other]                                          <L 68>
                    var_162 = wp::address(var_rowadr, var_40, var_146);
                    var_164 = wp::load(var_162);
                    var_163 = wp::copy(var_164);
                    // if other_count != count or other_offset < 0 or other_offset + count > colind.shape[2]:       <L 69>
                    var_165 = (var_160 != var_82);
                    var_167 = (var_163 < var_166);
                    var_168 = wp::add(var_163, var_82);
                    var_169 = &(var_colind.shape);
                    var_172 = wp::load(var_169);
                    var_171 = wp::extract(var_172, var_170);
                    var_173 = (var_168 > var_171);
                    var_174 = var_165 || var_167 || var_173;
                    if (var_174) {
                        // wp.atomic_or(status, 0, 128)                                           <L 70>
                        var_177 = wp::atomic_or(var_status, var_175, var_176);
                    }
                    if (!var_174) {
                        // for k in range(count):                                                 <L 72>
                        var_178 = wp::range(var_82);
                        start_for_10:;
                            if (iter_cmp(var_178) == 0) goto end_for_10;
                            var_179 = wp::iter_next(var_178);
                            // if colind[wid, 0, offset + k] != colind[wid, 0, other_offset + k]:       <L 73>
                            var_181 = wp::add(var_85, var_179);
                            var_182 = wp::address(var_colind, var_40, var_180, var_181);
                            var_184 = wp::add(var_163, var_179);
                            var_185 = wp::address(var_colind, var_40, var_183, var_184);
                            var_187 = wp::load(var_182);
                            var_188 = wp::load(var_185);
                            var_186 = (var_187 != var_188);
                            if (var_186) {
                                // wp.atomic_or(status, 0, 256)                                   <L 74>
                                var_191 = wp::atomic_or(var_status, var_189, var_190);
                            }
                            goto start_for_10;
                        end_for_10:;
                    }
                    var_192 = wp::where(var_174, var_112, var_179);
                }
                var_193 = wp::where(var_155, var_112, var_192);
                wp::assign(var_112, var_193);
                goto start_for_8;
            end_for_8:;
        }
    }
}

