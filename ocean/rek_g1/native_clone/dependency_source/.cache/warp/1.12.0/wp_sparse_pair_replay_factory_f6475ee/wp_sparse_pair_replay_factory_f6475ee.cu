
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



extern "C" __global__ void check_contact_sparse_rows_f4ed1685_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_nv,
    wp::int32 var_worlds,
    wp::int32 var_njmax,
    wp::int32 var_nnz_capacity,
    wp::array_t<wp::int32> var_nacon,
    wp::array_t<wp::int32> var_worldid,
    wp::array_t<wp::int32> var_dim,
    wp::array_t<wp::float32> var_dist,
    wp::array_t<wp::float32> var_margin,
    wp::array_t<wp::int32> var_addresses,
    wp::array_t<wp::int32> var_rownnz,
    wp::array_t<wp::int32> var_rowadr,
    wp::array_t<wp::int32> var_colind,
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
        const wp::int32 var_1 = 0;
        bool var_2;
        const wp::int32 var_3 = 0;
        wp::int32* var_4;
        wp::shape_t* var_5;
        const wp::int32 var_6 = 0;
        wp::int32 var_7;
        wp::shape_t var_8;
        bool var_9;
        wp::int32 var_10;
        bool var_11;
        const wp::int32 var_12 = 0;
        const wp::int32 var_13 = 1;
        wp::int32 var_14;
        const wp::int32 var_15 = 0;
        wp::int32* var_16;
        wp::shape_t* var_17;
        const wp::int32 var_18 = 0;
        wp::int32 var_19;
        wp::shape_t var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        bool var_23;
        wp::int32* var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        const wp::int32 var_27 = 0;
        bool var_28;
        bool var_29;
        bool var_30;
        const wp::int32 var_31 = 0;
        const wp::int32 var_32 = 1;
        wp::int32 var_33;
        wp::int32* var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        const wp::int32 var_37 = 1;
        bool var_38;
        wp::float32* var_39;
        wp::float32* var_40;
        bool var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        bool var_44;
        const wp::int32 var_45 = 1;
        bool var_46;
        const wp::int32 var_47 = 6;
        bool var_48;
        bool var_49;
        const wp::int32 var_50 = 1;
        const wp::int32 var_51 = 1;
        wp::int32 var_52;
        const wp::int32 var_53 = 0;
        wp::int32* var_54;
        wp::int32 var_55;
        wp::int32 var_56;
        const wp::int32 var_57 = 0;
        bool var_58;
        bool var_59;
        bool var_60;
        const wp::int32 var_61 = 2;
        const wp::int32 var_62 = 1;
        wp::int32 var_63;
        wp::int32* var_64;
        wp::int32 var_65;
        wp::int32 var_66;
        wp::int32* var_67;
        wp::int32 var_68;
        wp::int32 var_69;
        const wp::int32 var_70 = 0;
        bool var_71;
        bool var_72;
        const wp::int32 var_73 = 0;
        bool var_74;
        wp::int32 var_75;
        bool var_76;
        bool var_77;
        const wp::int32 var_78 = 3;
        const wp::int32 var_79 = 1;
        wp::int32 var_80;
        const wp::int32 var_81 = 6;
        wp::int32 var_82;
        const wp::int32 var_83 = 7;
        const wp::int32 var_84 = 1;
        wp::int32 var_85;
        wp::range_t var_86;
        wp::int32 var_87;
        const wp::int32 var_88 = 0;
        wp::int32 var_89;
        wp::int32* var_90;
        wp::int32 var_91;
        wp::int32 var_92;
        const wp::int32 var_93 = 0;
        bool var_94;
        bool var_95;
        bool var_96;
        const wp::int32 var_97 = 4;
        const wp::int32 var_98 = 1;
        wp::int32 var_99;
        wp::range_t var_100;
        wp::int32 var_101;
        const wp::int32 var_102 = 0;
        wp::int32 var_103;
        wp::int32* var_104;
        bool var_105;
        wp::int32 var_106;
        const wp::int32 var_107 = 4;
        const wp::int32 var_108 = 1;
        wp::int32 var_109;
        const wp::int32 var_110 = 1;
        wp::range_t var_111;
        wp::int32 var_112;
        wp::int32* var_113;
        wp::int32 var_114;
        wp::int32 var_115;
        const wp::int32 var_116 = 0;
        bool var_117;
        bool var_118;
        bool var_119;
        const wp::int32 var_120 = 2;
        const wp::int32 var_121 = 1;
        wp::int32 var_122;
        wp::int32* var_123;
        wp::int32 var_124;
        wp::int32 var_125;
        wp::int32* var_126;
        wp::int32 var_127;
        wp::int32 var_128;
        bool var_129;
        const wp::int32 var_130 = 5;
        const wp::int32 var_131 = 1;
        wp::int32 var_132;
        const wp::int32 var_133 = 0;
        bool var_134;
        wp::int32 var_135;
        bool var_136;
        bool var_137;
        const wp::int32 var_138 = 3;
        const wp::int32 var_139 = 1;
        wp::int32 var_140;
        wp::range_t var_141;
        wp::int32 var_142;
        const wp::int32 var_143 = 0;
        wp::int32 var_144;
        wp::int32* var_145;
        const wp::int32 var_146 = 0;
        wp::int32 var_147;
        wp::int32* var_148;
        bool var_149;
        wp::int32 var_150;
        wp::int32 var_151;
        const wp::int32 var_152 = 5;
        const wp::int32 var_153 = 1;
        wp::int32 var_154;
        //---------
        // forward
        // def check_contact_sparse_rows(                                                         <L 23>
        // contact = wp.tid()                                                                     <L 30>
        var_0 = builtin_tid1d();
        // if contact == 0 and nacon[0] > worldid.shape[0]:                                       <L 31>
        var_2 = (var_0 == var_1);
        var_4 = wp::address(var_nacon, var_3);
        var_5 = &(var_worldid.shape);
        var_8 = wp::load(var_5);
        var_7 = wp::extract(var_8, var_6);
        var_10 = wp::load(var_4);
        var_9 = (var_10 > var_7);
        var_11 = var_2 && var_9;
        if (var_11) {
            // wp.atomic_max(status, 0, 1)                                                        <L 32>
            var_14 = wp::atomic_max(var_status, var_12, var_13);
        }
        // if contact >= wp.min(nacon[0], worldid.shape[0]):                                      <L 33>
        var_16 = wp::address(var_nacon, var_15);
        var_17 = &(var_worldid.shape);
        var_20 = wp::load(var_17);
        var_19 = wp::extract(var_20, var_18);
        var_22 = wp::load(var_16);
        var_21 = wp::min(var_22, var_19);
        var_23 = (var_0 >= var_21);
        if (var_23) {
            // return                                                                             <L 34>
            continue;
        }
        // world = worldid[contact]                                                               <L 35>
        var_24 = wp::address(var_worldid, var_0);
        var_26 = wp::load(var_24);
        var_25 = wp::copy(var_26);
        // if world < 0 or world >= worlds:                                                       <L 36>
        var_28 = (var_25 < var_27);
        var_29 = (var_25 >= var_worlds);
        var_30 = var_28 || var_29;
        if (var_30) {
            // wp.atomic_max(status, 0, 1)                                                        <L 37>
            var_33 = wp::atomic_max(var_status, var_31, var_32);
            // return                                                                             <L 38>
            continue;
        }
        // count = dim[contact]                                                                   <L 39>
        var_34 = wp::address(var_dim, var_0);
        var_36 = wp::load(var_34);
        var_35 = wp::copy(var_36);
        // if count == 1 or dist[contact] >= margin[contact]:                                     <L 40>
        var_38 = (var_35 == var_37);
        var_39 = wp::address(var_dist, var_0);
        var_40 = wp::address(var_margin, var_0);
        var_42 = wp::load(var_39);
        var_43 = wp::load(var_40);
        var_41 = (var_42 >= var_43);
        var_44 = var_38 || var_41;
        if (var_44) {
            // return                                                                             <L 41>
            continue;
        }
        // if count < 1 or count > 6:                                                             <L 42>
        var_46 = (var_35 < var_45);
        var_48 = (var_35 > var_47);
        var_49 = var_46 || var_48;
        if (var_49) {
            // wp.atomic_max(status, 1, 1)                                                        <L 43>
            var_52 = wp::atomic_max(var_status, var_50, var_51);
            // return                                                                             <L 44>
            continue;
        }
        // first = addresses[contact, 0]                                                          <L 45>
        var_54 = wp::address(var_addresses, var_0, var_53);
        var_56 = wp::load(var_54);
        var_55 = wp::copy(var_56);
        // if first < 0 or first >= njmax:                                                        <L 46>
        var_58 = (var_55 < var_57);
        var_59 = (var_55 >= var_njmax);
        var_60 = var_58 || var_59;
        if (var_60) {
            // wp.atomic_max(status, 2, 1)                                                        <L 47>
            var_63 = wp::atomic_max(var_status, var_61, var_62);
            // return                                                                             <L 48>
            continue;
        }
        // nnz = rownnz[world, first]                                                             <L 49>
        var_64 = wp::address(var_rownnz, var_25, var_55);
        var_66 = wp::load(var_64);
        var_65 = wp::copy(var_66);
        // offset = rowadr[world, first]                                                          <L 50>
        var_67 = wp::address(var_rowadr, var_25, var_55);
        var_69 = wp::load(var_67);
        var_68 = wp::copy(var_69);
        // if nnz < 0 or nnz > nv or offset < 0 or offset + nnz > nnz_capacity:                   <L 51>
        var_71 = (var_65 < var_70);
        var_72 = (var_65 > var_nv);
        var_74 = (var_68 < var_73);
        var_75 = wp::add(var_68, var_65);
        var_76 = (var_75 > var_nnz_capacity);
        var_77 = var_71 || var_72 || var_74 || var_76;
        if (var_77) {
            // wp.atomic_max(status, 3, 1)                                                        <L 52>
            var_80 = wp::atomic_max(var_status, var_78, var_79);
            // return                                                                             <L 53>
            continue;
        }
        // wp.atomic_max(status, 6, nnz)                                                          <L 54>
        var_82 = wp::atomic_max(var_status, var_81, var_65);
        // wp.atomic_add(status, 7, 1)                                                            <L 55>
        var_85 = wp::atomic_add(var_status, var_83, var_84);
        // for i in range(nnz):                                                                   <L 56>
        var_86 = wp::range(var_65);
        start_for_6:;
            if (iter_cmp(var_86) == 0) goto end_for_6;
            var_87 = wp::iter_next(var_86);
            // col = colind[world, 0, offset + i]                                                 <L 57>
            var_89 = wp::add(var_68, var_87);
            var_90 = wp::address(var_colind, var_25, var_88, var_89);
            var_92 = wp::load(var_90);
            var_91 = wp::copy(var_92);
            // if col < 0 or col >= nv:                                                           <L 58>
            var_94 = (var_91 < var_93);
            var_95 = (var_91 >= var_nv);
            var_96 = var_94 || var_95;
            if (var_96) {
                // wp.atomic_max(status, 4, 1)                                                    <L 59>
                var_99 = wp::atomic_max(var_status, var_97, var_98);
            }
            // for j in range(i):                                                                 <L 60>
            var_100 = wp::range(var_87);
            start_for_8:;
                if (iter_cmp(var_100) == 0) goto end_for_8;
                var_101 = wp::iter_next(var_100);
                // if colind[world, 0, offset + j] == col:                                        <L 61>
                var_103 = wp::add(var_68, var_101);
                var_104 = wp::address(var_colind, var_25, var_102, var_103);
                var_106 = wp::load(var_104);
                var_105 = (var_106 == var_91);
                if (var_105) {
                    // wp.atomic_max(status, 4, 1)                                                <L 62>
                    var_109 = wp::atomic_max(var_status, var_107, var_108);
                }
                goto start_for_8;
            end_for_8:;
            goto start_for_6;
        end_for_6:;
        // for axis in range(1, count):                                                           <L 63>
        var_111 = wp::range(var_110, var_35);
        start_for_10:;
            if (iter_cmp(var_111) == 0) goto end_for_10;
            var_112 = wp::iter_next(var_111);
            // row = addresses[contact, axis]                                                     <L 64>
            var_113 = wp::address(var_addresses, var_0, var_112);
            var_115 = wp::load(var_113);
            var_114 = wp::copy(var_115);
            // if row < 0 or row >= njmax:                                                        <L 65>
            var_117 = (var_114 < var_116);
            var_118 = (var_114 >= var_njmax);
            var_119 = var_117 || var_118;
            if (var_119) {
                // wp.atomic_max(status, 2, 1)                                                    <L 66>
                var_122 = wp::atomic_max(var_status, var_120, var_121);
                // continue                                                                       <L 67>
                goto start_for_10;
            }
            // other_nnz = rownnz[world, row]                                                     <L 68>
            var_123 = wp::address(var_rownnz, var_25, var_114);
            var_125 = wp::load(var_123);
            var_124 = wp::copy(var_125);
            // other_offset = rowadr[world, row]                                                  <L 69>
            var_126 = wp::address(var_rowadr, var_25, var_114);
            var_128 = wp::load(var_126);
            var_127 = wp::copy(var_128);
            // if other_nnz != nnz:                                                               <L 70>
            var_129 = (var_124 != var_65);
            if (var_129) {
                // wp.atomic_max(status, 5, 1)                                                    <L 71>
                var_132 = wp::atomic_max(var_status, var_130, var_131);
                // continue                                                                       <L 72>
                goto start_for_10;
            }
            // if other_offset < 0 or other_offset + nnz > nnz_capacity:                          <L 73>
            var_134 = (var_127 < var_133);
            var_135 = wp::add(var_127, var_65);
            var_136 = (var_135 > var_nnz_capacity);
            var_137 = var_134 || var_136;
            if (var_137) {
                // wp.atomic_max(status, 3, 1)                                                    <L 74>
                var_140 = wp::atomic_max(var_status, var_138, var_139);
                // continue                                                                       <L 75>
                goto start_for_10;
            }
            // for i in range(nnz):                                                               <L 76>
            var_141 = wp::range(var_65);
            start_for_12:;
                if (iter_cmp(var_141) == 0) goto end_for_12;
                var_142 = wp::iter_next(var_141);
                // if colind[world, 0, other_offset + i] != colind[world, 0, offset + i]:         <L 77>
                var_144 = wp::add(var_127, var_142);
                var_145 = wp::address(var_colind, var_25, var_143, var_144);
                var_147 = wp::add(var_68, var_142);
                var_148 = wp::address(var_colind, var_25, var_146, var_147);
                var_150 = wp::load(var_145);
                var_151 = wp::load(var_148);
                var_149 = (var_150 != var_151);
                if (var_149) {
                    // wp.atomic_max(status, 5, 1)                                                <L 78>
                    var_154 = wp::atomic_max(var_status, var_152, var_153);
                }
                goto start_for_12;
            end_for_12:;
            wp::assign(var_87, var_142);
            goto start_for_10;
        end_for_10:;
    }
}

