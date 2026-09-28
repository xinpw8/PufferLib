
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



extern "C" __global__ void _capacity_high_water_1b6d14b0_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_nefc,
    wp::array_t<wp::int32> var_nacon,
    wp::array_t<wp::int32> var_ncollision,
    wp::array_t<wp::int32> var_efc_nnz,
    bool var_sparse,
    wp::int32 var_njmax,
    wp::int32 var_naconmax,
    wp::int32 var_njmax_nnz,
    wp::array_t<wp::int32> var_counters)
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
        wp::int32* var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        const wp::int32 var_4 = 0;
        wp::int32 var_5;
        bool var_6;
        const wp::int32 var_7 = 5;
        const wp::int32 var_8 = 1;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 3;
        wp::int32 var_14;
        bool var_15;
        const wp::int32 var_16 = 5;
        const wp::int32 var_17 = 2;
        wp::int32 var_18;
        const wp::int32 var_19 = 0;
        bool var_20;
        const wp::int32 var_21 = 1;
        const wp::int32 var_22 = 0;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        const wp::int32 var_26 = 2;
        const wp::int32 var_27 = 0;
        wp::int32* var_28;
        wp::int32 var_29;
        wp::int32 var_30;
        const wp::int32 var_31 = 4;
        const wp::int32 var_32 = 1;
        wp::int32 var_33;
        const wp::int32 var_34 = 0;
        wp::int32* var_35;
        bool var_36;
        wp::int32 var_37;
        const wp::int32 var_38 = 5;
        const wp::int32 var_39 = 4;
        wp::int32 var_40;
        const wp::int32 var_41 = 0;
        wp::int32* var_42;
        bool var_43;
        wp::int32 var_44;
        const wp::int32 var_45 = 5;
        const wp::int32 var_46 = 8;
        wp::int32 var_47;
        //---------
        // forward
        // def _capacity_high_water(                                                              <L 24>
        // world = wp.tid()                                                                       <L 35>
        var_0 = builtin_tid1d();
        // constraints = nefc[world]                                                              <L 36>
        var_1 = wp::address(var_nefc, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // wp.atomic_max(counters, 0, constraints)                                                <L 37>
        var_5 = wp::atomic_max(var_counters, var_4, var_2);
        // if constraints > njmax:                                                                <L 38>
        var_6 = (var_2 > var_njmax);
        if (var_6) {
            // wp.atomic_or(counters, 5, 1)                                                       <L 39>
            var_9 = wp::atomic_or(var_counters, var_7, var_8);
        }
        // if sparse:                                                                             <L 40>
        if (var_sparse) {
            // nonzeros = efc_nnz[world]                                                          <L 41>
            var_10 = wp::address(var_efc_nnz, var_0);
            var_12 = wp::load(var_10);
            var_11 = wp::copy(var_12);
            // wp.atomic_max(counters, 3, nonzeros)                                               <L 42>
            var_14 = wp::atomic_max(var_counters, var_13, var_11);
            // if nonzeros > njmax_nnz:                                                           <L 43>
            var_15 = (var_11 > var_njmax_nnz);
            if (var_15) {
                // wp.atomic_or(counters, 5, 2)                                                   <L 44>
                var_18 = wp::atomic_or(var_counters, var_16, var_17);
            }
        }
        // if world == 0:                                                                         <L 45>
        var_20 = (var_0 == var_19);
        if (var_20) {
            // wp.atomic_max(counters, 1, nacon[0])                                               <L 46>
            var_23 = wp::address(var_nacon, var_22);
            var_25 = wp::load(var_23);
            var_24 = wp::atomic_max(var_counters, var_21, var_25);
            // wp.atomic_max(counters, 2, ncollision[0])                                          <L 47>
            var_28 = wp::address(var_ncollision, var_27);
            var_30 = wp::load(var_28);
            var_29 = wp::atomic_max(var_counters, var_26, var_30);
            // wp.atomic_add(counters, 4, 1)                                                      <L 48>
            var_33 = wp::atomic_add(var_counters, var_31, var_32);
            // if nacon[0] > naconmax:                                                            <L 49>
            var_35 = wp::address(var_nacon, var_34);
            var_37 = wp::load(var_35);
            var_36 = (var_37 > var_naconmax);
            if (var_36) {
                // wp.atomic_or(counters, 5, 4)                                                   <L 50>
                var_40 = wp::atomic_or(var_counters, var_38, var_39);
            }
            // if ncollision[0] > naconmax:                                                       <L 51>
            var_42 = wp::address(var_ncollision, var_41);
            var_44 = wp::load(var_42);
            var_43 = (var_44 > var_naconmax);
            if (var_43) {
                // wp.atomic_or(counters, 5, 8)                                                   <L 52>
                var_47 = wp::atomic_or(var_counters, var_45, var_46);
            }
        }
    }
}



extern "C" __global__ void _capacity_high_water_1b6d14b0_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_nefc,
    wp::array_t<wp::int32> var_nacon,
    wp::array_t<wp::int32> var_ncollision,
    wp::array_t<wp::int32> var_efc_nnz,
    bool var_sparse,
    wp::int32 var_njmax,
    wp::int32 var_naconmax,
    wp::int32 var_njmax_nnz,
    wp::array_t<wp::int32> var_counters,
    wp::array_t<wp::int32> adj_nefc,
    wp::array_t<wp::int32> adj_nacon,
    wp::array_t<wp::int32> adj_ncollision,
    wp::array_t<wp::int32> adj_efc_nnz,
    bool adj_sparse,
    wp::int32 adj_njmax,
    wp::int32 adj_naconmax,
    wp::int32 adj_njmax_nnz,
    wp::array_t<wp::int32> adj_counters)
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
        wp::int32* var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        const wp::int32 var_4 = 0;
        wp::int32 var_5;
        bool var_6;
        const wp::int32 var_7 = 5;
        const wp::int32 var_8 = 1;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 3;
        wp::int32 var_14;
        bool var_15;
        const wp::int32 var_16 = 5;
        const wp::int32 var_17 = 2;
        wp::int32 var_18;
        const wp::int32 var_19 = 0;
        bool var_20;
        const wp::int32 var_21 = 1;
        const wp::int32 var_22 = 0;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        const wp::int32 var_26 = 2;
        const wp::int32 var_27 = 0;
        wp::int32* var_28;
        wp::int32 var_29;
        wp::int32 var_30;
        const wp::int32 var_31 = 4;
        const wp::int32 var_32 = 1;
        wp::int32 var_33;
        const wp::int32 var_34 = 0;
        wp::int32* var_35;
        bool var_36;
        wp::int32 var_37;
        const wp::int32 var_38 = 5;
        const wp::int32 var_39 = 4;
        wp::int32 var_40;
        const wp::int32 var_41 = 0;
        wp::int32* var_42;
        bool var_43;
        wp::int32 var_44;
        const wp::int32 var_45 = 5;
        const wp::int32 var_46 = 8;
        wp::int32 var_47;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        bool adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        bool adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::int32 adj_19 = {};
        bool adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::int32 adj_31 = {};
        wp::int32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::int32 adj_35 = {};
        bool adj_36 = {};
        wp::int32 adj_37 = {};
        wp::int32 adj_38 = {};
        wp::int32 adj_39 = {};
        wp::int32 adj_40 = {};
        wp::int32 adj_41 = {};
        wp::int32 adj_42 = {};
        bool adj_43 = {};
        wp::int32 adj_44 = {};
        wp::int32 adj_45 = {};
        wp::int32 adj_46 = {};
        wp::int32 adj_47 = {};
        //---------
        // forward
        // def _capacity_high_water(                                                              <L 24>
        // world = wp.tid()                                                                       <L 35>
        var_0 = builtin_tid1d();
        // constraints = nefc[world]                                                              <L 36>
        var_1 = wp::address(var_nefc, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // wp.atomic_max(counters, 0, constraints)                                                <L 37>
        // var_5 = wp::atomic_max(var_counters, var_4, var_2);
        // if constraints > njmax:                                                                <L 38>
        var_6 = (var_2 > var_njmax);
        if (var_6) {
            // wp.atomic_or(counters, 5, 1)                                                       <L 39>
            // var_9 = wp::atomic_or(var_counters, var_7, var_8);
        }
        // if sparse:                                                                             <L 40>
        if (var_sparse) {
            // nonzeros = efc_nnz[world]                                                          <L 41>
            var_10 = wp::address(var_efc_nnz, var_0);
            var_12 = wp::load(var_10);
            var_11 = wp::copy(var_12);
            // wp.atomic_max(counters, 3, nonzeros)                                               <L 42>
            // var_14 = wp::atomic_max(var_counters, var_13, var_11);
            // if nonzeros > njmax_nnz:                                                           <L 43>
            var_15 = (var_11 > var_njmax_nnz);
            if (var_15) {
                // wp.atomic_or(counters, 5, 2)                                                   <L 44>
                // var_18 = wp::atomic_or(var_counters, var_16, var_17);
            }
        }
        // if world == 0:                                                                         <L 45>
        var_20 = (var_0 == var_19);
        if (var_20) {
            // wp.atomic_max(counters, 1, nacon[0])                                               <L 46>
            var_23 = wp::address(var_nacon, var_22);
            var_25 = wp::load(var_23);
            // var_24 = wp::atomic_max(var_counters, var_21, var_25);
            // wp.atomic_max(counters, 2, ncollision[0])                                          <L 47>
            var_28 = wp::address(var_ncollision, var_27);
            var_30 = wp::load(var_28);
            // var_29 = wp::atomic_max(var_counters, var_26, var_30);
            // wp.atomic_add(counters, 4, 1)                                                      <L 48>
            // var_33 = wp::atomic_add(var_counters, var_31, var_32);
            // if nacon[0] > naconmax:                                                            <L 49>
            var_35 = wp::address(var_nacon, var_34);
            var_37 = wp::load(var_35);
            var_36 = (var_37 > var_naconmax);
            if (var_36) {
                // wp.atomic_or(counters, 5, 4)                                                   <L 50>
                // var_40 = wp::atomic_or(var_counters, var_38, var_39);
            }
            // if ncollision[0] > naconmax:                                                       <L 51>
            var_42 = wp::address(var_ncollision, var_41);
            var_44 = wp::load(var_42);
            var_43 = (var_44 > var_naconmax);
            if (var_43) {
                // wp.atomic_or(counters, 5, 8)                                                   <L 52>
                // var_47 = wp::atomic_or(var_counters, var_45, var_46);
            }
        }
        //---------
        // reverse
        if (var_20) {
            if (var_43) {
                // adj: wp.atomic_or(counters, 5, 8)                                              <L 52>
            }
            wp::adj_address(var_ncollision, var_41, adj_ncollision, adj_41, adj_42);
            // adj: if ncollision[0] > naconmax:                                                  <L 51>
            if (var_36) {
                // adj: wp.atomic_or(counters, 5, 4)                                              <L 50>
            }
            wp::adj_address(var_nacon, var_34, adj_nacon, adj_34, adj_35);
            // adj: if nacon[0] > naconmax:                                                       <L 49>
            wp::adj_atomic_add(var_counters, var_31, var_32, adj_counters, adj_31, adj_32, adj_33);
            // adj: wp.atomic_add(counters, 4, 1)                                                 <L 48>
            wp::adj_atomic_max(var_counters, var_26, var_30, adj_counters, adj_26, adj_28, adj_29);
            wp::adj_address(var_ncollision, var_27, adj_ncollision, adj_27, adj_28);
            // adj: wp.atomic_max(counters, 2, ncollision[0])                                     <L 47>
            wp::adj_atomic_max(var_counters, var_21, var_25, adj_counters, adj_21, adj_23, adj_24);
            wp::adj_address(var_nacon, var_22, adj_nacon, adj_22, adj_23);
            // adj: wp.atomic_max(counters, 1, nacon[0])                                          <L 46>
        }
        // adj: if world == 0:                                                                    <L 45>
        if (var_sparse) {
            if (var_15) {
                // adj: wp.atomic_or(counters, 5, 2)                                              <L 44>
            }
            // adj: if nonzeros > njmax_nnz:                                                      <L 43>
            wp::adj_atomic_max(var_counters, var_13, var_11, adj_counters, adj_13, adj_11, adj_14);
            // adj: wp.atomic_max(counters, 3, nonzeros)                                          <L 42>
            wp::adj_copy(var_12, adj_10, adj_11);
            wp::adj_address(var_efc_nnz, var_0, adj_efc_nnz, adj_0, adj_10);
            // adj: nonzeros = efc_nnz[world]                                                     <L 41>
        }
        // adj: if sparse:                                                                        <L 40>
        if (var_6) {
            // adj: wp.atomic_or(counters, 5, 1)                                                  <L 39>
        }
        // adj: if constraints > njmax:                                                           <L 38>
        wp::adj_atomic_max(var_counters, var_4, var_2, adj_counters, adj_4, adj_2, adj_5);
        // adj: wp.atomic_max(counters, 0, constraints)                                           <L 37>
        wp::adj_copy(var_3, adj_1, adj_2);
        wp::adj_address(var_nefc, var_0, adj_nefc, adj_0, adj_1);
        // adj: constraints = nefc[world]                                                         <L 36>
        // adj: world = wp.tid()                                                                  <L 35>
        // adj: def _capacity_high_water(                                                         <L 24>
        continue;
    }
}

