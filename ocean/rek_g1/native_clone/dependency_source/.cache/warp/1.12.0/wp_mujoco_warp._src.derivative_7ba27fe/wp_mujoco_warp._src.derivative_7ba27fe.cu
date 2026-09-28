
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/support.py:37
static CUDA_CALLABLE wp::float32 next_act_0(
    wp::float32 var_opt_timestep,
    wp::int32 var_actuator_dyntype,
    wp::vec_t<10, wp::float32> var_actuator_dynprm,
    wp::vec_t<2, wp::float32> var_actuator_actrange,
    wp::float32 var_act_in,
    wp::float32 var_act_dot_in,
    wp::float32 var_act_dot_scale,
    bool var_clamp)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 3;
    bool var_1;
    const wp::float32 var_2 = 1e-15;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::float32 var_8 = 1.0;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    const wp::int32 var_15 = 6;
    bool var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    const wp::int32 var_22 = 0;
    wp::float32 var_23;
    const wp::int32 var_24 = 1;
    wp::float32 var_25;
    wp::float32 var_26;
    wp::float32 var_27;
    //---------
    // forward
    // def next_act(                                                                          <L 38>
    // if actuator_dyntype == DynType.FILTEREXACT:                                            <L 52>
    var_1 = (var_actuator_dyntype == var_0);
    if (var_1) {
        // tau = wp.max(MJ_MINVAL, actuator_dynprm[0])                                        <L 53>
        var_4 = wp::extract(var_actuator_dynprm, var_3);
        var_5 = wp::max(var_2, var_4);
        // act = act_in + act_dot_scale * act_dot_in * tau * (1.0 - wp.exp(-opt_timestep / tau))       <L 54>
        var_6 = wp::mul(var_act_dot_scale, var_act_dot_in);
        var_7 = wp::mul(var_6, var_5);
        var_9 = wp::neg(var_opt_timestep);
        var_10 = wp::div(var_9, var_5);
        var_11 = wp::exp(var_10);
        var_12 = wp::sub(var_8, var_11);
        var_13 = wp::mul(var_7, var_12);
        var_14 = wp::add(var_act_in, var_13);
    }
    if (!var_1) {
        // elif actuator_dyntype == DynType.USER:                                             <L 55>
        var_16 = (var_actuator_dyntype == var_15);
        if (var_16) {
            // return act_in                                                                  <L 56>
            return var_act_in;
        }
        if (!var_16) {
            // act = act_in + act_dot_scale * act_dot_in * opt_timestep                       <L 58>
            var_17 = wp::mul(var_act_dot_scale, var_act_dot_in);
            var_18 = wp::mul(var_17, var_opt_timestep);
            var_19 = wp::add(var_act_in, var_18);
        }
        var_20 = wp::where(var_16, var_14, var_19);
    }
    var_21 = wp::where(var_1, var_14, var_20);
    // if clamp:                                                                              <L 61>
    if (var_clamp) {
        // act = wp.clamp(act, actuator_actrange[0], actuator_actrange[1])                    <L 62>
        var_23 = wp::extract(var_actuator_actrange, var_22);
        var_25 = wp::extract(var_actuator_actrange, var_24);
        var_26 = wp::clamp(var_21, var_23, var_25);
    }
    var_27 = wp::where(var_clamp, var_26, var_21);
    // return act                                                                             <L 64>
    return var_27;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/support.py:37
static CUDA_CALLABLE void adj_next_act_0(
    wp::float32 var_opt_timestep,
    wp::int32 var_actuator_dyntype,
    wp::vec_t<10, wp::float32> var_actuator_dynprm,
    wp::vec_t<2, wp::float32> var_actuator_actrange,
    wp::float32 var_act_in,
    wp::float32 var_act_dot_in,
    wp::float32 var_act_dot_scale,
    bool var_clamp,
    wp::float32 & adj_opt_timestep,
    wp::int32 & adj_actuator_dyntype,
    wp::vec_t<10, wp::float32> & adj_actuator_dynprm,
    wp::vec_t<2, wp::float32> & adj_actuator_actrange,
    wp::float32 & adj_act_in,
    wp::float32 & adj_act_dot_in,
    wp::float32 & adj_act_dot_scale,
    bool & adj_clamp,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void _qderiv_actuator_passive_actuation_sparse_030f6563_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_M_rownnz,
    wp::array_t<wp::int32> var_M_rowadr,
    wp::array_t<wp::int32> var_moment_rownnz_in,
    wp::array_t<wp::int32> var_moment_rowadr_in,
    wp::array_t<wp::int32> var_moment_colind_in,
    wp::array_t<wp::float32> var_actuator_moment_in,
    wp::array_t<wp::float32> var_vel_in,
    wp::array_t<wp::int32> var_qMj,
    wp::array_t<wp::float32> var_qDeriv_out)
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
        wp::float32* var_2;
        wp::float32 var_3;
        wp::float32 var_4;
        const wp::float32 var_5 = 0.0;
        bool var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::range_t var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::float32* var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        const wp::float32 var_19 = 0.0;
        bool var_20;
        wp::int32* var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        const wp::int32 var_24 = 1;
        wp::int32 var_25;
        wp::range_t var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::float32* var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        const wp::float32 var_32 = 0.0;
        bool var_33;
        wp::int32* var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        wp::float32 var_37;
        wp::float32 var_38;
        wp::int32 var_39;
        wp::int32 var_40;
        wp::int32* var_41;
        const wp::int32 var_42 = 1;
        wp::int32 var_43;
        wp::int32 var_44;
        wp::int32* var_45;
        wp::int32 var_46;
        wp::int32 var_47;
        wp::range_t var_48;
        wp::int32 var_49;
        const wp::int32 var_50 = 1;
        wp::int32 var_51;
        wp::int32* var_52;
        bool var_53;
        wp::int32 var_54;
        const wp::int32 var_55 = 0;
        wp::slice_t var_56;
        const wp::int32 var_57 = 0;
        wp::slice_t var_58;
        const wp::int32 var_59 = 0;
        wp::array_t<wp::float32> var_60;
        wp::float32 var_61;
        wp::int32 var_62;
        //---------
        // forward
        // def _qderiv_actuator_passive_actuation_sparse(                                         <L 172>
        // worldid, actid = wp.tid()                                                              <L 187>
        builtin_tid2d(var_0, var_1);
        // vel = vel_in[worldid, actid]                                                           <L 189>
        var_2 = wp::address(var_vel_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if vel == 0.0:                                                                         <L 190>
        var_6 = (var_3 == var_5);
        if (var_6) {
            // return                                                                             <L 191>
            continue;
        }
        // rownnz = moment_rownnz_in[worldid, actid]                                              <L 193>
        var_7 = wp::address(var_moment_rownnz_in, var_0, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // rowadr = moment_rowadr_in[worldid, actid]                                              <L 194>
        var_10 = wp::address(var_moment_rowadr_in, var_0, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // for i in range(rownnz):                                                                <L 196>
        var_13 = wp::range(var_8);
        start_for_1:;
            if (iter_cmp(var_13) == 0) goto end_for_1;
            var_14 = wp::iter_next(var_13);
            // rowadri = rowadr + i                                                               <L 197>
            var_15 = wp::add(var_11, var_14);
            // moment_i = actuator_moment_in[worldid, rowadri]                                    <L 198>
            var_16 = wp::address(var_actuator_moment_in, var_0, var_15);
            var_18 = wp::load(var_16);
            var_17 = wp::copy(var_18);
            // if moment_i == 0.0:                                                                <L 199>
            var_20 = (var_17 == var_19);
            if (var_20) {
                // continue                                                                       <L 200>
                goto start_for_1;
            }
            // dofi = moment_colind_in[worldid, rowadri]                                          <L 201>
            var_21 = wp::address(var_moment_colind_in, var_0, var_15);
            var_23 = wp::load(var_21);
            var_22 = wp::copy(var_23);
            // for j in range(i + 1):                                                             <L 203>
            var_25 = wp::add(var_14, var_24);
            var_26 = wp::range(var_25);
            start_for_3:;
                if (iter_cmp(var_26) == 0) goto end_for_3;
                var_27 = wp::iter_next(var_26);
                // rowadrj = rowadr + j                                                           <L 204>
                var_28 = wp::add(var_11, var_27);
                // moment_j = actuator_moment_in[worldid, rowadrj]                                <L 205>
                var_29 = wp::address(var_actuator_moment_in, var_0, var_28);
                var_31 = wp::load(var_29);
                var_30 = wp::copy(var_31);
                // if moment_j == 0.0:                                                            <L 206>
                var_33 = (var_30 == var_32);
                if (var_33) {
                    // continue                                                                   <L 207>
                    goto start_for_3;
                }
                // dofj = moment_colind_in[worldid, rowadrj]                                      <L 208>
                var_34 = wp::address(var_moment_colind_in, var_0, var_28);
                var_36 = wp::load(var_34);
                var_35 = wp::copy(var_36);
                // contrib = moment_i * moment_j * vel                                            <L 210>
                var_37 = wp::mul(var_17, var_30);
                var_38 = wp::mul(var_37, var_3);
                // row = dofi                                                                     <L 214>
                var_39 = wp::copy(var_22);
                // col = dofj                                                                     <L 215>
                var_40 = wp::copy(var_35);
                // row_startk = M_rowadr[row] - 1                                                 <L 216>
                var_41 = wp::address(var_M_rowadr, var_39);
                var_44 = wp::load(var_41);
                var_43 = wp::sub(var_44, var_42);
                // row_nnz = M_rownnz[row]                                                        <L 217>
                var_45 = wp::address(var_M_rownnz, var_39);
                var_47 = wp::load(var_45);
                var_46 = wp::copy(var_47);
                // for k in range(row_nnz):                                                       <L 218>
                var_48 = wp::range(var_46);
                start_for_5:;
                    if (iter_cmp(var_48) == 0) goto end_for_5;
                    var_49 = wp::iter_next(var_48);
                    // row_startk += 1                                                            <L 219>
                    var_51 = wp::add(var_43, var_50);
                    // if qMj[row_startk] == col:                                                 <L 220>
                    var_52 = wp::address(var_qMj, var_51);
                    var_54 = wp::load(var_52);
                    var_53 = (var_54 == var_40);
                    if (var_53) {
                        // wp.atomic_add(qDeriv_out[worldid, 0], row_startk, contrib)             <L 221>
                        var_56 = wp::slice_t(var_0, var_0, var_57);
                        var_58 = wp::slice_t(var_55, var_55, var_59);
                        var_60 = wp::view(var_qDeriv_out, var_56, var_58);
                        var_61 = wp::atomic_add(var_60, var_51, var_38);
                        // break                                                                  <L 222>
                        wp::assign(var_43, var_51);
                        goto end_for_5;
                    }
                    var_62 = wp::where(var_53, var_43, var_51);
                    wp::assign(var_43, var_62);
                    goto start_for_5;
                end_for_5:;
                goto start_for_3;
            end_for_3:;
            goto start_for_1;
        end_for_1:;
    }
}



extern "C" __global__ void _qderiv_actuator_passive_7ab7826a_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::int32 var_opt_disableflags,
    wp::array_t<wp::float32> var_dof_damping,
    bool var_is_sparse,
    wp::array_t<wp::float32> var_qM_in,
    wp::array_t<wp::int32> var_qMi,
    wp::array_t<wp::int32> var_qMj,
    wp::array_t<wp::float32> var_qDeriv_in,
    wp::array_t<wp::float32> var_qDeriv_out)
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
        const wp::int32 var_8 = 0;
        wp::float32* var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        const wp::int32 var_16 = 64;
        wp::int32 var_17;
        bool var_18;
        bool var_19;
        bool var_20;
        wp::shape_t* var_21;
        const wp::int32 var_22 = 0;
        wp::int32 var_23;
        wp::shape_t var_24;
        wp::int32 var_25;
        wp::float32* var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        wp::shape_t* var_30;
        const wp::int32 var_31 = 0;
        wp::int32 var_32;
        wp::shape_t var_33;
        wp::int32 var_34;
        wp::float32* var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        const wp::int32 var_38 = 0;
        wp::float32* var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        const wp::int32 var_42 = 0;
        wp::float32* var_43;
        wp::float32 var_44;
        wp::float32 var_45;
        bool var_46;
        //---------
        // forward
        // def _qderiv_actuator_passive(                                                          <L 226>
        // worldid, elemid = wp.tid()                                                             <L 241>
        builtin_tid2d(var_0, var_1);
        // dofiid = qMi[elemid]                                                                   <L 243>
        var_2 = wp::address(var_qMi, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // dofjid = qMj[elemid]                                                                   <L 244>
        var_5 = wp::address(var_qMj, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if is_sparse:                                                                          <L 246>
        if (var_is_sparse) {
            // qderiv = qDeriv_in[worldid, 0, elemid]                                             <L 247>
            var_9 = wp::address(var_qDeriv_in, var_0, var_8, var_1);
            var_11 = wp::load(var_9);
            var_10 = wp::copy(var_11);
        }
        if (!var_is_sparse) {
            // qderiv = qDeriv_in[worldid, dofiid, dofjid]                                        <L 249>
            var_12 = wp::address(var_qDeriv_in, var_0, var_3, var_6);
            var_14 = wp::load(var_12);
            var_13 = wp::copy(var_14);
        }
        var_15 = wp::where(var_is_sparse, var_10, var_13);
        // if not (opt_disableflags & DisableBit.DAMPER) and dofiid == dofjid:                    <L 251>
        var_17 = wp::bit_and(var_opt_disableflags, var_16);
        var_18 = wp::unot(var_17);
        var_19 = (var_3 == var_6);
        var_20 = var_18 && var_19;
        if (var_20) {
            // qderiv -= dof_damping[worldid % dof_damping.shape[0], dofiid]                      <L 252>
            var_21 = &(var_dof_damping.shape);
            var_24 = wp::load(var_21);
            var_23 = wp::extract(var_24, var_22);
            var_25 = wp::mod(var_0, var_23);
            var_26 = wp::address(var_dof_damping, var_25, var_3);
            var_28 = wp::load(var_26);
            var_27 = wp::sub(var_15, var_28);
        }
        var_29 = wp::where(var_20, var_27, var_15);
        // qderiv *= opt_timestep[worldid % opt_timestep.shape[0]]                                <L 254>
        var_30 = &(var_opt_timestep.shape);
        var_33 = wp::load(var_30);
        var_32 = wp::extract(var_33, var_31);
        var_34 = wp::mod(var_0, var_32);
        var_35 = wp::address(var_opt_timestep, var_34);
        var_37 = wp::load(var_35);
        var_36 = wp::mul(var_29, var_37);
        // if is_sparse:                                                                          <L 256>
        if (var_is_sparse) {
            // qDeriv_out[worldid, 0, elemid] = qM_in[worldid, 0, elemid] - qderiv                <L 257>
            var_39 = wp::address(var_qM_in, var_0, var_38, var_1);
            var_41 = wp::load(var_39);
            var_40 = wp::sub(var_41, var_36);
            wp::array_store(var_qDeriv_out, var_0, var_42, var_1, var_40);
        }
        if (!var_is_sparse) {
            // qM = qM_in[worldid, dofiid, dofjid] - qderiv                                       <L 259>
            var_43 = wp::address(var_qM_in, var_0, var_3, var_6);
            var_45 = wp::load(var_43);
            var_44 = wp::sub(var_45, var_36);
            // qDeriv_out[worldid, dofiid, dofjid] = qM                                           <L 260>
            wp::array_store(var_qDeriv_out, var_0, var_3, var_6, var_44);
            // if dofiid != dofjid:                                                               <L 261>
            var_46 = (var_3 != var_6);
            if (var_46) {
                // qDeriv_out[worldid, dofjid, dofiid] = qM                                       <L 262>
                wp::array_store(var_qDeriv_out, var_0, var_6, var_3, var_44);
            }
        }
    }
}



extern "C" __global__ void _qderiv_tendon_damping_b185cc4e_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_ntendon,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::float32> var_tendon_damping,
    bool var_is_sparse,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::array_t<wp::int32> var_qMi,
    wp::array_t<wp::int32> var_qMj,
    wp::array_t<wp::float32> var_qDeriv_out)
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
        const wp::float32 var_8 = 0.0;
        wp::float32 var_9;
        wp::shape_t* var_10;
        const wp::int32 var_11 = 0;
        wp::int32 var_12;
        wp::shape_t var_13;
        wp::int32 var_14;
        wp::range_t var_15;
        wp::int32 var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        const wp::float32 var_20 = 0.0;
        bool var_21;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        wp::int32* var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        const wp::float32 var_28 = 0.0;
        wp::float32 var_29;
        const wp::float32 var_30 = 0.0;
        wp::float32 var_31;
        wp::range_t var_32;
        wp::int32 var_33;
        const wp::float32 var_34 = 0.0;
        bool var_35;
        const wp::float32 var_36 = 0.0;
        bool var_37;
        bool var_38;
        wp::int32 var_39;
        wp::int32* var_40;
        wp::int32 var_41;
        wp::int32 var_42;
        bool var_43;
        wp::float32* var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::float32 var_47;
        bool var_48;
        wp::float32* var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        wp::float32 var_52;
        wp::float32 var_53;
        wp::float32 var_54;
        wp::float32 var_55;
        wp::shape_t* var_56;
        const wp::int32 var_57 = 0;
        wp::int32 var_58;
        wp::shape_t var_59;
        wp::int32 var_60;
        wp::float32* var_61;
        wp::float32 var_62;
        wp::float32 var_63;
        const wp::int32 var_64 = 0;
        wp::float32 var_65;
        wp::float32 var_66;
        bool var_67;
        wp::float32 var_68;
        //---------
        // forward
        // def _qderiv_tendon_damping(                                                            <L 267>
        // worldid, elemid = wp.tid()                                                             <L 284>
        builtin_tid2d(var_0, var_1);
        // dofiid = qMi[elemid]                                                                   <L 285>
        var_2 = wp::address(var_qMi, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // dofjid = qMj[elemid]                                                                   <L 286>
        var_5 = wp::address(var_qMj, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // qderiv = float(0.0)                                                                    <L 288>
        var_9 = wp::float(var_8);
        // tendon_damping_id = worldid % tendon_damping.shape[0]                                  <L 289>
        var_10 = &(var_tendon_damping.shape);
        var_13 = wp::load(var_10);
        var_12 = wp::extract(var_13, var_11);
        var_14 = wp::mod(var_0, var_12);
        // for tenid in range(ntendon):                                                           <L 290>
        var_15 = wp::range(var_ntendon);
        start_for_0:;
            if (iter_cmp(var_15) == 0) goto end_for_0;
            var_16 = wp::iter_next(var_15);
            // damping = tendon_damping[tendon_damping_id, tenid]                                 <L 291>
            var_17 = wp::address(var_tendon_damping, var_14, var_16);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
            // if damping == 0.0:                                                                 <L 292>
            var_21 = (var_18 == var_20);
            if (var_21) {
                // continue                                                                       <L 293>
                goto start_for_0;
            }
            // rownnz = ten_J_rownnz[tenid]                                                       <L 295>
            var_22 = wp::address(var_ten_J_rownnz, var_16);
            var_24 = wp::load(var_22);
            var_23 = wp::copy(var_24);
            // rowadr = ten_J_rowadr[tenid]                                                       <L 296>
            var_25 = wp::address(var_ten_J_rowadr, var_16);
            var_27 = wp::load(var_25);
            var_26 = wp::copy(var_27);
            // Ji = float(0.0)                                                                    <L 297>
            var_29 = wp::float(var_28);
            // Jj = float(0.0)                                                                    <L 298>
            var_31 = wp::float(var_30);
            // for k in range(rownnz):                                                            <L 299>
            var_32 = wp::range(var_23);
            start_for_2:;
                if (iter_cmp(var_32) == 0) goto end_for_2;
                var_33 = wp::iter_next(var_32);
                // if Ji != 0.0 and Jj != 0.0:                                                    <L 300>
                var_35 = (var_29 != var_34);
                var_37 = (var_31 != var_36);
                var_38 = var_35 && var_37;
                if (var_38) {
                    // break                                                                      <L 301>
                    goto end_for_2;
                }
                // sparseid = rowadr + k                                                          <L 302>
                var_39 = wp::add(var_26, var_33);
                // colind = ten_J_colind[sparseid]                                                <L 303>
                var_40 = wp::address(var_ten_J_colind, var_39);
                var_42 = wp::load(var_40);
                var_41 = wp::copy(var_42);
                // if colind == dofiid:                                                           <L 304>
                var_43 = (var_41 == var_3);
                if (var_43) {
                    // Ji = ten_J_in[worldid, sparseid]                                           <L 305>
                    var_44 = wp::address(var_ten_J_in, var_0, var_39);
                    var_46 = wp::load(var_44);
                    var_45 = wp::copy(var_46);
                }
                var_47 = wp::where(var_43, var_45, var_29);
                // if colind == dofjid:                                                           <L 306>
                var_48 = (var_41 == var_6);
                if (var_48) {
                    // Jj = ten_J_in[worldid, sparseid]                                           <L 307>
                    var_49 = wp::address(var_ten_J_in, var_0, var_39);
                    var_51 = wp::load(var_49);
                    var_50 = wp::copy(var_51);
                }
                var_52 = wp::where(var_48, var_50, var_31);
                wp::assign(var_29, var_47);
                wp::assign(var_31, var_52);
                goto start_for_2;
            end_for_2:;
            // qderiv -= Ji * Jj * damping                                                        <L 308>
            var_53 = wp::mul(var_29, var_31);
            var_54 = wp::mul(var_53, var_18);
            var_55 = wp::sub(var_9, var_54);
            wp::assign(var_9, var_55);
            goto start_for_0;
        end_for_0:;
        // qderiv *= opt_timestep[worldid % opt_timestep.shape[0]]                                <L 310>
        var_56 = &(var_opt_timestep.shape);
        var_59 = wp::load(var_56);
        var_58 = wp::extract(var_59, var_57);
        var_60 = wp::mod(var_0, var_58);
        var_61 = wp::address(var_opt_timestep, var_60);
        var_63 = wp::load(var_61);
        var_62 = wp::mul(var_9, var_63);
        // if is_sparse:                                                                          <L 312>
        if (var_is_sparse) {
            // qDeriv_out[worldid, 0, elemid] -= qderiv                                           <L 313>
            var_65 = wp::atomic_sub(var_qDeriv_out, var_0, var_64, var_1, var_62);
        }
        if (!var_is_sparse) {
            // qDeriv_out[worldid, dofiid, dofjid] -= qderiv                                      <L 315>
            var_66 = wp::atomic_sub(var_qDeriv_out, var_0, var_3, var_6, var_62);
            // if dofiid != dofjid:                                                               <L 316>
            var_67 = (var_3 != var_6);
            if (var_67) {
                // qDeriv_out[worldid, dofjid, dofiid] -= qderiv                                  <L 317>
                var_68 = wp::atomic_sub(var_qDeriv_out, var_0, var_6, var_3, var_62);
            }
        }
    }
}



extern "C" __global__ void _qderiv_actuator_passive_vel_161f6e76_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::int32> var_actuator_dyntype,
    wp::array_t<wp::int32> var_actuator_gaintype,
    wp::array_t<wp::int32> var_actuator_biastype,
    wp::array_t<wp::int32> var_actuator_actadr,
    wp::array_t<wp::int32> var_actuator_actnum,
    wp::array_t<bool> var_actuator_forcelimited,
    wp::array_t<bool> var_actuator_actlimited,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_dynprm,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_gainprm,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_biasprm,
    wp::array_t<bool> var_actuator_actearly,
    wp::array_t<wp::vec_t<2, wp::float32>> var_actuator_forcerange,
    wp::array_t<wp::vec_t<2, wp::float32>> var_actuator_actrange,
    wp::array_t<wp::float32> var_act_in,
    wp::array_t<wp::float32> var_ctrl_in,
    wp::array_t<wp::float32> var_act_dot_in,
    wp::array_t<wp::float32> var_actuator_force_in,
    wp::array_t<wp::float32> var_vel_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::shape_t* var_7;
        const wp::int32 var_8 = 0;
        wp::int32 var_9;
        wp::shape_t var_10;
        wp::int32 var_11;
        wp::int32* var_12;
        const wp::int32 var_13 = 1;
        bool var_14;
        wp::int32 var_15;
        wp::vec_t<10, wp::float32>* var_16;
        const wp::int32 var_17 = 2;
        wp::float32 var_18;
        wp::vec_t<10, wp::float32> var_19;
        const wp::float32 var_20 = 0.0;
        wp::float32 var_21;
        wp::int32* var_22;
        const wp::int32 var_23 = 1;
        bool var_24;
        wp::int32 var_25;
        wp::vec_t<10, wp::float32>* var_26;
        const wp::int32 var_27 = 2;
        wp::float32 var_28;
        wp::vec_t<10, wp::float32> var_29;
        const wp::float32 var_30 = 0.0;
        wp::float32 var_31;
        const wp::float32 var_32 = 0.0;
        bool var_33;
        const wp::float32 var_34 = 0.0;
        bool var_35;
        bool var_36;
        const wp::float32 var_37 = 0.0;
        bool* var_38;
        bool var_39;
        wp::float32* var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::shape_t* var_43;
        const wp::int32 var_44 = 0;
        wp::int32 var_45;
        wp::shape_t var_46;
        wp::int32 var_47;
        wp::vec_t<2, wp::float32>* var_48;
        wp::vec_t<2, wp::float32> var_49;
        wp::vec_t<2, wp::float32> var_50;
        const wp::int32 var_51 = 0;
        wp::float32 var_52;
        bool var_53;
        const wp::int32 var_54 = 1;
        wp::float32 var_55;
        bool var_56;
        bool var_57;
        const wp::float32 var_58 = 0.0;
        bool var_59;
        wp::float32 var_60;
        wp::int32* var_61;
        const wp::int32 var_62 = 0;
        bool var_63;
        wp::int32 var_64;
        const wp::float32 var_65 = 0.0;
        bool var_66;
        wp::int32* var_67;
        wp::int32* var_68;
        wp::int32 var_69;
        wp::int32 var_70;
        wp::int32 var_71;
        const wp::int32 var_72 = 1;
        wp::int32 var_73;
        bool* var_74;
        bool var_75;
        wp::shape_t* var_76;
        const wp::int32 var_77 = 0;
        wp::int32 var_78;
        wp::shape_t var_79;
        wp::int32 var_80;
        wp::float32* var_81;
        wp::int32* var_82;
        wp::shape_t* var_83;
        const wp::int32 var_84 = 0;
        wp::int32 var_85;
        wp::shape_t var_86;
        wp::int32 var_87;
        wp::vec_t<10, wp::float32>* var_88;
        wp::shape_t* var_89;
        const wp::int32 var_90 = 0;
        wp::int32 var_91;
        wp::shape_t var_92;
        wp::int32 var_93;
        wp::vec_t<2, wp::float32>* var_94;
        wp::float32* var_95;
        wp::float32* var_96;
        const wp::float32 var_97 = 1.0;
        bool* var_98;
        wp::float32 var_99;
        wp::float32 var_100;
        wp::int32 var_101;
        wp::vec_t<10, wp::float32> var_102;
        wp::vec_t<2, wp::float32> var_103;
        wp::float32 var_104;
        wp::float32 var_105;
        bool var_106;
        bool var_107;
        bool var_108;
        wp::float32* var_109;
        wp::float32 var_110;
        wp::float32 var_111;
        bool var_112;
        wp::float32 var_113;
        bool var_114;
        wp::float32 var_115;
        wp::float32 var_116;
        wp::float32 var_117;
        wp::float32 var_118;
        const wp::float32 var_119 = 0.0;
        bool var_120;
        wp::float32* var_121;
        wp::float32 var_122;
        wp::float32 var_123;
        wp::float32 var_124;
        wp::float32 var_125;
        wp::float32 var_126;
        //---------
        // forward
        // def _qderiv_actuator_passive_vel(                                                      <L 32>
        // worldid, actid = wp.tid()                                                              <L 56>
        builtin_tid2d(var_0, var_1);
        // actuator_gainprm_id = worldid % actuator_gainprm.shape[0]                              <L 58>
        var_2 = &(var_actuator_gainprm.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // actuator_biasprm_id = worldid % actuator_biasprm.shape[0]                              <L 59>
        var_7 = &(var_actuator_biasprm.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::mod(var_0, var_9);
        // if actuator_gaintype[actid] == GainType.AFFINE:                                        <L 61>
        var_12 = wp::address(var_actuator_gaintype, var_1);
        var_15 = wp::load(var_12);
        var_14 = (var_15 == var_13);
        if (var_14) {
            // gain = actuator_gainprm[actuator_gainprm_id, actid][2]                             <L 62>
            var_16 = wp::address(var_actuator_gainprm, var_6, var_1);
            var_19 = wp::load(var_16);
            var_18 = wp::extract(var_19, var_17);
        }
        if (!var_14) {
            // gain = 0.0                                                                         <L 64>
        }
        var_21 = wp::where(var_14, var_18, var_20);
        // if actuator_biastype[actid] == BiasType.AFFINE:                                        <L 66>
        var_22 = wp::address(var_actuator_biastype, var_1);
        var_25 = wp::load(var_22);
        var_24 = (var_25 == var_23);
        if (var_24) {
            // bias = actuator_biasprm[actuator_biasprm_id, actid][2]                             <L 67>
            var_26 = wp::address(var_actuator_biasprm, var_11, var_1);
            var_29 = wp::load(var_26);
            var_28 = wp::extract(var_29, var_27);
        }
        if (!var_24) {
            // bias = 0.0                                                                         <L 69>
        }
        var_31 = wp::where(var_24, var_28, var_30);
        // if bias == 0.0 and gain == 0.0:                                                        <L 71>
        var_33 = (var_31 == var_32);
        var_35 = (var_21 == var_34);
        var_36 = var_33 && var_35;
        if (var_36) {
            // vel_out[worldid, actid] = 0.0                                                      <L 72>
            wp::array_store(var_vel_out, var_0, var_1, var_37);
            // return                                                                             <L 73>
            continue;
        }
        // if actuator_forcelimited[actid]:                                                       <L 76>
        var_38 = wp::address(var_actuator_forcelimited, var_1);
        var_39 = wp::load(var_38);
        if (var_39) {
            // force = actuator_force_in[worldid, actid]                                          <L 77>
            var_40 = wp::address(var_actuator_force_in, var_0, var_1);
            var_42 = wp::load(var_40);
            var_41 = wp::copy(var_42);
            // forcerange = actuator_forcerange[worldid % actuator_forcerange.shape[0], actid]       <L 78>
            var_43 = &(var_actuator_forcerange.shape);
            var_46 = wp::load(var_43);
            var_45 = wp::extract(var_46, var_44);
            var_47 = wp::mod(var_0, var_45);
            var_48 = wp::address(var_actuator_forcerange, var_47, var_1);
            var_50 = wp::load(var_48);
            var_49 = wp::copy(var_50);
            // if force <= forcerange[0] or force >= forcerange[1]:                               <L 79>
            var_52 = wp::extract(var_49, var_51);
            var_53 = (var_41 <= var_52);
            var_55 = wp::extract(var_49, var_54);
            var_56 = (var_41 >= var_55);
            var_57 = var_53 || var_56;
            if (var_57) {
                // vel_out[worldid, actid] = 0.0                                                  <L 80>
                wp::array_store(var_vel_out, var_0, var_1, var_58);
                // return                                                                         <L 81>
                continue;
            }
        }
        var_59 = wp::load(var_38);
        // vel = float(bias)                                                                      <L 83>
        var_60 = wp::float(var_31);
        // if actuator_dyntype[actid] != DynType.NONE:                                            <L 84>
        var_61 = wp::address(var_actuator_dyntype, var_1);
        var_64 = wp::load(var_61);
        var_63 = (var_64 != var_62);
        if (var_63) {
            // if gain != 0.0:                                                                    <L 85>
            var_66 = (var_21 != var_65);
            if (var_66) {
                // act_adr = actuator_actadr[actid] + actuator_actnum[actid] - 1                  <L 86>
                var_67 = wp::address(var_actuator_actadr, var_1);
                var_68 = wp::address(var_actuator_actnum, var_1);
                var_70 = wp::load(var_67);
                var_71 = wp::load(var_68);
                var_69 = wp::add(var_70, var_71);
                var_73 = wp::sub(var_69, var_72);
                // if actuator_actearly[actid]:                                                   <L 89>
                var_74 = wp::address(var_actuator_actearly, var_1);
                var_75 = wp::load(var_74);
                if (var_75) {
                    // act = next_act(                                                            <L 90>
                    // opt_timestep[worldid % opt_timestep.shape[0]],                             <L 91>
                    var_76 = &(var_opt_timestep.shape);
                    var_79 = wp::load(var_76);
                    var_78 = wp::extract(var_79, var_77);
                    var_80 = wp::mod(var_0, var_78);
                    var_81 = wp::address(var_opt_timestep, var_80);
                    // actuator_dyntype[actid],                                                   <L 92>
                    var_82 = wp::address(var_actuator_dyntype, var_1);
                    // actuator_dynprm[worldid % actuator_dynprm.shape[0], actid],                <L 93>
                    var_83 = &(var_actuator_dynprm.shape);
                    var_86 = wp::load(var_83);
                    var_85 = wp::extract(var_86, var_84);
                    var_87 = wp::mod(var_0, var_85);
                    var_88 = wp::address(var_actuator_dynprm, var_87, var_1);
                    // actuator_actrange[worldid % actuator_actrange.shape[0], actid],            <L 94>
                    var_89 = &(var_actuator_actrange.shape);
                    var_92 = wp::load(var_89);
                    var_91 = wp::extract(var_92, var_90);
                    var_93 = wp::mod(var_0, var_91);
                    var_94 = wp::address(var_actuator_actrange, var_93, var_1);
                    // act_in[worldid, act_adr],                                                  <L 95>
                    var_95 = wp::address(var_act_in, var_0, var_73);
                    // act_dot_in[worldid, act_adr],                                              <L 96>
                    var_96 = wp::address(var_act_dot_in, var_0, var_73);
                    // 1.0,                                                                       <L 97>
                    // actuator_actlimited[actid],                                                <L 98>
                    var_98 = wp::address(var_actuator_actlimited, var_1);
                    var_100 = wp::load(var_81);
                    var_101 = wp::load(var_82);
                    var_102 = wp::load(var_88);
                    var_103 = wp::load(var_94);
                    var_104 = wp::load(var_95);
                    var_105 = wp::load(var_96);
                    var_106 = wp::load(var_98);
                    var_99 = next_act_0(var_100, var_101, var_102, var_103, var_104, var_105, var_97, var_106);
                }
                var_107 = wp::load(var_74);
                var_108 = wp::load(var_74);
                if (!var_108) {
                    // act = act_in[worldid, act_adr]                                             <L 101>
                    var_109 = wp::address(var_act_in, var_0, var_73);
                    var_111 = wp::load(var_109);
                    var_110 = wp::copy(var_111);
                }
                var_112 = wp::load(var_74);
                var_114 = wp::load(var_74);
                var_113 = wp::where(var_114, var_99, var_110);
                // vel += gain * act                                                              <L 103>
                var_115 = wp::mul(var_21, var_113);
                var_116 = wp::add(var_60, var_115);
            }
            var_117 = wp::where(var_66, var_116, var_60);
        }
        var_118 = wp::where(var_63, var_117, var_60);
        if (!var_63) {
            // if gain != 0.0:                                                                    <L 105>
            var_120 = (var_21 != var_119);
            if (var_120) {
                // vel += gain * ctrl_in[worldid, actid]                                          <L 106>
                var_121 = wp::address(var_ctrl_in, var_0, var_1);
                var_123 = wp::load(var_121);
                var_122 = wp::mul(var_21, var_123);
                var_124 = wp::add(var_118, var_122);
            }
            var_125 = wp::where(var_120, var_124, var_118);
        }
        var_126 = wp::where(var_63, var_118, var_125);
        // vel_out[worldid, actid] = vel                                                          <L 108>
        wp::array_store(var_vel_out, var_0, var_1, var_126);
    }
}



extern "C" __global__ void _qderiv_actuator_passive_actuation_dense_d8dc877b_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_nu,
    wp::array_t<wp::int32> var_moment_rownnz_in,
    wp::array_t<wp::int32> var_moment_rowadr_in,
    wp::array_t<wp::int32> var_moment_colind_in,
    wp::array_t<wp::float32> var_actuator_moment_in,
    wp::array_t<wp::float32> var_vel_in,
    wp::array_t<wp::int32> var_qMi,
    wp::array_t<wp::int32> var_qMj,
    wp::array_t<wp::float32> var_qDeriv_out)
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
        const wp::float32 var_8 = 0.0;
        wp::float32 var_9;
        wp::range_t var_10;
        wp::int32 var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        const wp::float32 var_15 = 0.0;
        bool var_16;
        const wp::float32 var_17 = 0.0;
        wp::float32 var_18;
        const wp::float32 var_19 = 0.0;
        wp::float32 var_20;
        wp::int32* var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        wp::int32* var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        wp::range_t var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        bool var_33;
        wp::float32* var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        bool var_38;
        wp::float32* var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        const wp::float32 var_43 = 0.0;
        bool var_44;
        const wp::float32 var_45 = 0.0;
        bool var_46;
        bool var_47;
        wp::float32 var_48;
        wp::float32 var_49;
        const wp::int32 var_50 = 0;
        bool var_51;
        const wp::int32 var_52 = 0;
        bool var_53;
        bool var_54;
        wp::float32 var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        bool var_58;
        //---------
        // forward
        // def _qderiv_actuator_passive_actuation_dense(                                          <L 120>
        // worldid, elemid = wp.tid()                                                             <L 135>
        builtin_tid2d(var_0, var_1);
        // dofiid = qMi[elemid]                                                                   <L 137>
        var_2 = wp::address(var_qMi, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // dofjid = qMj[elemid]                                                                   <L 138>
        var_5 = wp::address(var_qMj, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // qderiv_contrib = float(0.0)                                                            <L 139>
        var_9 = wp::float(var_8);
        // for actid in range(nu):                                                                <L 140>
        var_10 = wp::range(var_nu);
        start_for_0:;
            if (iter_cmp(var_10) == 0) goto end_for_0;
            var_11 = wp::iter_next(var_10);
            // vel = vel_in[worldid, actid]                                                       <L 141>
            var_12 = wp::address(var_vel_in, var_0, var_11);
            var_14 = wp::load(var_12);
            var_13 = wp::copy(var_14);
            // if vel == 0.0:                                                                     <L 142>
            var_16 = (var_13 == var_15);
            if (var_16) {
                // continue                                                                       <L 143>
                goto start_for_0;
            }
            // moment_i = float(0.0)                                                              <L 146>
            var_18 = wp::float(var_17);
            // moment_j = float(0.0)                                                              <L 147>
            var_20 = wp::float(var_19);
            // rownnz = moment_rownnz_in[worldid, actid]                                          <L 149>
            var_21 = wp::address(var_moment_rownnz_in, var_0, var_11);
            var_23 = wp::load(var_21);
            var_22 = wp::copy(var_23);
            // rowadr = moment_rowadr_in[worldid, actid]                                          <L 150>
            var_24 = wp::address(var_moment_rowadr_in, var_0, var_11);
            var_26 = wp::load(var_24);
            var_25 = wp::copy(var_26);
            // for i in range(rownnz):                                                            <L 151>
            var_27 = wp::range(var_22);
            start_for_2:;
                if (iter_cmp(var_27) == 0) goto end_for_2;
                var_28 = wp::iter_next(var_27);
                // sparseid = rowadr + i                                                          <L 152>
                var_29 = wp::add(var_25, var_28);
                // colind = moment_colind_in[worldid, sparseid]                                   <L 153>
                var_30 = wp::address(var_moment_colind_in, var_0, var_29);
                var_32 = wp::load(var_30);
                var_31 = wp::copy(var_32);
                // if colind == dofiid:                                                           <L 154>
                var_33 = (var_31 == var_3);
                if (var_33) {
                    // moment_i = actuator_moment_in[worldid, sparseid]                           <L 155>
                    var_34 = wp::address(var_actuator_moment_in, var_0, var_29);
                    var_36 = wp::load(var_34);
                    var_35 = wp::copy(var_36);
                }
                var_37 = wp::where(var_33, var_35, var_18);
                // if colind == dofjid:                                                           <L 156>
                var_38 = (var_31 == var_6);
                if (var_38) {
                    // moment_j = actuator_moment_in[worldid, sparseid]                           <L 157>
                    var_39 = wp::address(var_actuator_moment_in, var_0, var_29);
                    var_41 = wp::load(var_39);
                    var_40 = wp::copy(var_41);
                }
                var_42 = wp::where(var_38, var_40, var_20);
                // if moment_i != 0.0 and moment_j != 0.0:                                        <L 158>
                var_44 = (var_37 != var_43);
                var_46 = (var_42 != var_45);
                var_47 = var_44 && var_46;
                if (var_47) {
                    // break                                                                      <L 159>
                    wp::assign(var_18, var_37);
                    wp::assign(var_20, var_42);
                    goto end_for_2;
                }
                var_48 = wp::where(var_47, var_18, var_37);
                var_49 = wp::where(var_47, var_20, var_42);
                wp::assign(var_18, var_48);
                wp::assign(var_20, var_49);
                goto start_for_2;
            end_for_2:;
            // if moment_i == 0 and moment_j == 0:                                                <L 161>
            var_51 = (var_18 == var_50);
            var_53 = (var_20 == var_52);
            var_54 = var_51 && var_53;
            if (var_54) {
                // continue                                                                       <L 162>
                goto start_for_0;
            }
            // qderiv_contrib += moment_i * moment_j * vel                                        <L 164>
            var_55 = wp::mul(var_18, var_20);
            var_56 = wp::mul(var_55, var_13);
            var_57 = wp::add(var_9, var_56);
            wp::assign(var_9, var_57);
            goto start_for_0;
        end_for_0:;
        // qDeriv_out[worldid, dofiid, dofjid] = qderiv_contrib                                   <L 166>
        wp::array_store(var_qDeriv_out, var_0, var_3, var_6, var_9);
        // if dofiid != dofjid:                                                                   <L 167>
        var_58 = (var_3 != var_6);
        if (var_58) {
            // qDeriv_out[worldid, dofjid, dofiid] = qderiv_contrib                               <L 168>
            wp::array_store(var_qDeriv_out, var_0, var_6, var_3, var_9);
        }
    }
}

