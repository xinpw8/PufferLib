
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/support.py:240
static CUDA_CALLABLE wp::vec_t<6, wp::float32> _decode_pyramid_0(
    wp::int32 var_njmax_in,
    wp::array_t<wp::float32> var_pyramid,
    wp::int32 var_efc_address,
    wp::vec_t<5, wp::float32> var_mu,
    wp::int32 var_condim)
{
    //---------
    // primal vars
    wp::vec_t<6, wp::float32> var_0;
    const wp::int32 var_1 = 1;
    bool var_2;
    wp::float32* var_3;
    const wp::int32 var_4 = 0;
    wp::float32 var_5;
    const wp::float32 var_6 = 0.0;
    wp::float32 var_7;
    const wp::int32 var_8 = 0;
    const wp::int32 var_9 = 1;
    wp::int32 var_10;
    wp::range_t var_11;
    wp::int32 var_12;
    const wp::int32 var_13 = 2;
    wp::int32 var_14;
    wp::int32 var_15;
    bool var_16;
    wp::float32* var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    const wp::float32 var_20 = 0.0;
    wp::float32 var_21;
    const wp::int32 var_22 = 1;
    wp::int32 var_23;
    bool var_24;
    const wp::int32 var_25 = 1;
    wp::int32 var_26;
    wp::float32* var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    const wp::float32 var_30 = 0.0;
    wp::float32 var_31;
    wp::float32 var_32;
    const wp::int32 var_33 = 0;
    wp::float32 var_34;
    wp::float32 var_35;
    wp::float32 var_36;
    const wp::int32 var_37 = 1;
    wp::int32 var_38;
    //---------
    // forward
    // def _decode_pyramid(njmax_in: int, pyramid: wp.array[float], efc_address: int, mu: vec5, condim: int) -> wp.spatial_vector:       <L 241>
    // force = wp.spatial_vector()                                                            <L 243>
    var_0 = wp::vec_t<6, wp::float32>();
    // if condim == 1:                                                                        <L 245>
    var_2 = (var_condim == var_1);
    if (var_2) {
        // force[0] = pyramid[efc_address]                                                    <L 246>
        var_3 = wp::address(var_pyramid, var_efc_address);
        var_5 = wp::load(var_3);
        wp::assign_inplace(var_0, var_4, var_5);
        // return force                                                                       <L 247>
        return var_0;
    }
    // force[0] = float(0.0)                                                                  <L 249>
    var_7 = wp::float(var_6);
    wp::assign_inplace(var_0, var_8, var_7);
    // for i in range(condim - 1):                                                            <L 250>
    var_10 = wp::sub(var_condim, var_9);
    var_11 = wp::range(var_10);
    start_for_1:;
        if (iter_cmp(var_11) == 0) goto end_for_1;
        var_12 = wp::iter_next(var_11);
        // adr = 2 * i + efc_address                                                          <L 251>
        var_14 = wp::mul(var_13, var_12);
        var_15 = wp::add(var_14, var_efc_address);
        // if adr < njmax_in:                                                                 <L 252>
        var_16 = (var_15 < var_njmax_in);
        if (var_16) {
            // dir1 = pyramid[adr]                                                            <L 253>
            var_17 = wp::address(var_pyramid, var_15);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
        }
        if (!var_16) {
            // dir1 = 0.0                                                                     <L 255>
        }
        var_21 = wp::where(var_16, var_18, var_20);
        // if adr + 1 < njmax_in:                                                             <L 256>
        var_23 = wp::add(var_15, var_22);
        var_24 = (var_23 < var_njmax_in);
        if (var_24) {
            // dir2 = pyramid[adr + 1]                                                        <L 257>
            var_26 = wp::add(var_15, var_25);
            var_27 = wp::address(var_pyramid, var_26);
            var_29 = wp::load(var_27);
            var_28 = wp::copy(var_29);
        }
        if (!var_24) {
            // dir2 = 0.0                                                                     <L 259>
        }
        var_31 = wp::where(var_24, var_28, var_30);
        // force[0] += dir1 + dir2                                                            <L 260>
        var_32 = wp::add(var_21, var_31);
        wp::add_inplace(var_0, var_33, var_32);
        // force[i + 1] = (dir1 - dir2) * mu[i]                                               <L 261>
        var_34 = wp::sub(var_21, var_31);
        var_35 = wp::extract(var_mu, var_12);
        var_36 = wp::mul(var_34, var_35);
        var_38 = wp::add(var_12, var_37);
        wp::assign_inplace(var_0, var_38, var_36);
        goto start_for_1;
    end_for_1:;
    // return force                                                                           <L 263>
    return var_0;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/support.py:266
static CUDA_CALLABLE wp::vec_t<6, wp::float32> contact_force_fn_0(
    wp::int32 var_opt_cone,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::float32> var_efc_force_in,
    wp::int32 var_njmax_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::int32 var_worldid,
    wp::int32 var_contact_id,
    bool var_to_world_frame)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    const wp::float32 var_1 = 0.0;
    const wp::float32 var_2 = 0.0;
    const wp::float32 var_3 = 0.0;
    const wp::float32 var_4 = 0.0;
    const wp::float32 var_5 = 0.0;
    wp::vec_t<6, wp::float32> var_6;
    wp::int32* var_7;
    wp::int32 var_8;
    wp::int32 var_9;
    const wp::int32 var_10 = 0;
    wp::int32* var_11;
    wp::int32 var_12;
    wp::int32 var_13;
    const wp::int32 var_14 = 0;
    bool var_15;
    const wp::int32 var_16 = 0;
    wp::int32* var_17;
    bool var_18;
    wp::int32 var_19;
    const wp::int32 var_20 = 0;
    bool var_21;
    bool var_22;
    const wp::int32 var_23 = 0;
    bool var_24;
    wp::slice_t var_25;
    const wp::int32 var_26 = 0;
    wp::array_t<wp::float32> var_27;
    wp::vec_t<5, wp::float32>* var_28;
    wp::vec_t<6, wp::float32> var_29;
    wp::vec_t<5, wp::float32> var_30;
    wp::vec_t<6, wp::float32> var_31;
    wp::range_t var_32;
    wp::int32 var_33;
    wp::int32* var_34;
    bool var_35;
    wp::int32 var_36;
    wp::int32* var_37;
    wp::float32* var_38;
    wp::int32 var_39;
    wp::float32 var_40;
    wp::vec_t<6, wp::float32> var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::mat_t<3, 3, wp::float32>* var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::mat_t<3, 3, wp::float32> var_45;
    wp::vec_t<3, wp::float32> var_46;
    wp::mat_t<3, 3, wp::float32>* var_47;
    wp::vec_t<3, wp::float32> var_48;
    wp::mat_t<3, 3, wp::float32> var_49;
    wp::vec_t<6, wp::float32> var_50;
    wp::vec_t<6, wp::float32> var_51;
    //---------
    // forward
    // def contact_force_fn(                                                                  <L 267>
    // force = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)                                <L 284>
    var_6 = wp::vec_t<6, wp::float32>({var_0, var_1, var_2, var_3, var_4, var_5});
    // condim = contact_dim_in[contact_id]                                                    <L 285>
    var_7 = wp::address(var_contact_dim_in, var_contact_id);
    var_9 = wp::load(var_7);
    var_8 = wp::copy(var_9);
    // efc_address = contact_efc_address_in[contact_id, 0]                                    <L 286>
    var_11 = wp::address(var_contact_efc_address_in, var_contact_id, var_10);
    var_13 = wp::load(var_11);
    var_12 = wp::copy(var_13);
    // if contact_id >= 0 and contact_id <= nacon_in[0] and efc_address >= 0:                 <L 288>
    var_15 = (var_contact_id >= var_14);
    var_17 = wp::address(var_nacon_in, var_16);
    var_19 = wp::load(var_17);
    var_18 = (var_contact_id <= var_19);
    var_21 = (var_12 >= var_20);
    var_22 = var_15 && var_18 && var_21;
    if (var_22) {
        // if opt_cone == ConeType.PYRAMIDAL:                                                 <L 289>
        var_24 = (var_opt_cone == var_23);
        if (var_24) {
            // force = _decode_pyramid(                                                       <L 290>
            // njmax_in,                                                                      <L 291>
            // efc_force_in[worldid],                                                         <L 292>
            var_25 = wp::slice_t(var_worldid, var_worldid, var_26);
            var_27 = wp::view(var_efc_force_in, var_25);
            // efc_address,                                                                   <L 293>
            // contact_friction_in[contact_id],                                               <L 294>
            var_28 = wp::address(var_contact_friction_in, var_contact_id);
            // condim,                                                                        <L 295>
            var_30 = wp::load(var_28);
            var_29 = _decode_pyramid_0(var_njmax_in, var_27, var_12, var_30, var_8);
        }
        var_31 = wp::where(var_24, var_29, var_6);
        if (!var_24) {
            // for i in range(condim):                                                        <L 298>
            var_32 = wp::range(var_8);
            start_for_0:;
                if (iter_cmp(var_32) == 0) goto end_for_0;
                var_33 = wp::iter_next(var_32);
                // if contact_efc_address_in[contact_id, i] < njmax_in:                       <L 299>
                var_34 = wp::address(var_contact_efc_address_in, var_contact_id, var_33);
                var_36 = wp::load(var_34);
                var_35 = (var_36 < var_njmax_in);
                if (var_35) {
                    // force[i] = efc_force_in[worldid, contact_efc_address_in[contact_id, i]]       <L 300>
                    var_37 = wp::address(var_contact_efc_address_in, var_contact_id, var_33);
                    var_39 = wp::load(var_37);
                    var_38 = wp::address(var_efc_force_in, var_worldid, var_39);
                    var_40 = wp::load(var_38);
                    wp::assign_inplace(var_31, var_33, var_40);
                }
                goto start_for_0;
            end_for_0:;
        }
    }
    var_41 = wp::where(var_22, var_31, var_6);
    // if to_world_frame:                                                                     <L 302>
    if (var_to_world_frame) {
        // t = wp.spatial_top(force) @ contact_frame_in[contact_id]                           <L 304>
        var_42 = wp::spatial_top(var_41);
        var_43 = wp::address(var_contact_frame_in, var_contact_id);
        var_45 = wp::load(var_43);
        var_44 = wp::mul(var_42, var_45);
        // b = wp.spatial_bottom(force) @ contact_frame_in[contact_id]                        <L 305>
        var_46 = wp::spatial_bottom(var_41);
        var_47 = wp::address(var_contact_frame_in, var_contact_id);
        var_49 = wp::load(var_47);
        var_48 = wp::mul(var_46, var_49);
        // force = wp.spatial_vector(t, b)                                                    <L 306>
        var_50 = wp::vec_t<6, wp::float32>(var_44, var_48);
    }
    var_51 = wp::where(var_to_world_frame, var_50, var_41);
    // return force                                                                           <L 308>
    return var_51;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/support.py:240
static CUDA_CALLABLE void adj__decode_pyramid_0(
    wp::int32 var_njmax_in,
    wp::array_t<wp::float32> var_pyramid,
    wp::int32 var_efc_address,
    wp::vec_t<5, wp::float32> var_mu,
    wp::int32 var_condim,
    wp::int32 & adj_njmax_in,
    wp::array_t<wp::float32> & adj_pyramid,
    wp::int32 & adj_efc_address,
    wp::vec_t<5, wp::float32> & adj_mu,
    wp::int32 & adj_condim,
    wp::vec_t<6, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/support.py:266
static CUDA_CALLABLE void adj_contact_force_fn_0(
    wp::int32 var_opt_cone,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::float32> var_efc_force_in,
    wp::int32 var_njmax_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::int32 var_worldid,
    wp::int32 var_contact_id,
    bool var_to_world_frame,
    wp::int32 & adj_opt_cone,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_contact_frame_in,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_friction_in,
    wp::array_t<wp::int32> & adj_contact_dim_in,
    wp::array_t<wp::int32> & adj_contact_efc_address_in,
    wp::array_t<wp::float32> & adj_efc_force_in,
    wp::int32 & adj_njmax_in,
    wp::array_t<wp::int32> & adj_nacon_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_contact_id,
    bool & adj_to_world_frame,
    wp::vec_t<6, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void _apply_ft_1667a928_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_nbody,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_ft_in,
    bool var_flg_add,
    wp::array_t<wp::float32> var_qfrc_out)
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
        wp::vec_t<6, wp::float32>* var_2;
        wp::vec_t<6, wp::float32> var_3;
        wp::vec_t<6, wp::float32> var_4;
        const wp::int32 var_5 = 0;
        wp::float32 var_6;
        const wp::int32 var_7 = 1;
        wp::float32 var_8;
        const wp::int32 var_9 = 2;
        wp::float32 var_10;
        wp::vec_t<3, wp::float32> var_11;
        const wp::int32 var_12 = 3;
        wp::float32 var_13;
        const wp::int32 var_14 = 4;
        wp::float32 var_15;
        const wp::int32 var_16 = 5;
        wp::float32 var_17;
        const wp::int32 var_18 = 0;
        wp::float32 var_19;
        const wp::int32 var_20 = 1;
        wp::float32 var_21;
        const wp::int32 var_22 = 2;
        wp::float32 var_23;
        wp::vec_t<6, wp::float32> var_24;
        wp::int32* var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        const wp::float32 var_28 = 0.0;
        wp::float32 var_29;
        wp::range_t var_30;
        wp::int32 var_31;
        wp::vec_t<6, wp::float32>* var_32;
        wp::vec_t<6, wp::float32> var_33;
        wp::vec_t<6, wp::float32> var_34;
        wp::vec_t<6, wp::float32> var_35;
        bool var_36;
        wp::int32 var_37;
        const wp::int32 var_38 = 0;
        bool var_39;
        bool var_40;
        bool var_41;
        wp::int32* var_42;
        wp::int32 var_43;
        wp::int32 var_44;
        const wp::int32 var_45 = 0;
        bool var_46;
        wp::vec_t<3, wp::float32>* var_47;
        wp::int32* var_48;
        wp::vec_t<3, wp::float32>* var_49;
        wp::int32 var_50;
        wp::vec_t<3, wp::float32> var_51;
        wp::vec_t<3, wp::float32> var_52;
        wp::vec_t<3, wp::float32> var_53;
        wp::vec_t<3, wp::float32> var_54;
        wp::float32 var_55;
        wp::vec_t<3, wp::float32> var_56;
        wp::float32 var_57;
        wp::float32 var_58;
        wp::float32 var_59;
        wp::float32 var_60;
        //---------
        // forward
        // def _apply_ft(                                                                         <L 175>
        // worldid, dofid = wp.tid()                                                              <L 191>
        builtin_tid2d(var_0, var_1);
        // cdof = cdof_in[worldid, dofid]                                                         <L 192>
        var_2 = wp::address(var_cdof_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // rotational_cdof = wp.vec3(cdof[0], cdof[1], cdof[2])                                   <L 193>
        var_6 = wp::extract(var_3, var_5);
        var_8 = wp::extract(var_3, var_7);
        var_10 = wp::extract(var_3, var_9);
        var_11 = wp::vec_t<3, wp::float32>(var_6, var_8, var_10);
        // jac = wp.spatial_vector(cdof[3], cdof[4], cdof[5], cdof[0], cdof[1], cdof[2])          <L 194>
        var_13 = wp::extract(var_3, var_12);
        var_15 = wp::extract(var_3, var_14);
        var_17 = wp::extract(var_3, var_16);
        var_19 = wp::extract(var_3, var_18);
        var_21 = wp::extract(var_3, var_20);
        var_23 = wp::extract(var_3, var_22);
        var_24 = wp::vec_t<6, wp::float32>({var_13, var_15, var_17, var_19, var_21, var_23});
        // dofbodyid = dof_bodyid[dofid]                                                          <L 196>
        var_25 = wp::address(var_dof_bodyid, var_1);
        var_27 = wp::load(var_25);
        var_26 = wp::copy(var_27);
        // accumul = float(0.0)                                                                   <L 197>
        var_29 = wp::float(var_28);
        // for bodyid in range(dofbodyid, nbody):                                                 <L 199>
        var_30 = wp::range(var_26, var_nbody);
        start_for_0:;
            if (iter_cmp(var_30) == 0) goto end_for_0;
            var_31 = wp::iter_next(var_30);
            // ft_body = ft_in[worldid, bodyid]                                                   <L 200>
            var_32 = wp::address(var_ft_in, var_0, var_31);
            var_34 = wp::load(var_32);
            var_33 = wp::copy(var_34);
            // if ft_body == wp.spatial_vector():                                                 <L 201>
            var_35 = wp::vec_t<6, wp::float32>();
            var_36 = (var_33 == var_35);
            if (var_36) {
                // continue                                                                       <L 202>
                goto start_for_0;
            }
            // parentid = bodyid                                                                  <L 204>
            var_37 = wp::copy(var_31);
            // while parentid != 0 and parentid != dofbodyid:                                     <L 205>
        start_while_2:;
            var_39 = (var_37 != var_38);
            var_40 = (var_37 != var_26);
            var_41 = var_39 && var_40;
        if ((var_41) == false) goto end_while_2;
                // parentid = body_parentid[parentid]                                             <L 206>
                var_42 = wp::address(var_body_parentid, var_37);
                var_44 = wp::load(var_42);
                var_43 = wp::copy(var_44);
                wp::assign(var_37, var_43);
        goto start_while_2;
        end_while_2:;
            // if parentid == 0:                                                                  <L 207>
            var_46 = (var_37 == var_45);
            if (var_46) {
                // continue  # body is not part of the subtree                                    <L 208>
                goto start_for_0;
            }
            // offset = xipos_in[worldid, bodyid] - subtree_com_in[worldid, body_rootid[bodyid]]       <L 209>
            var_47 = wp::address(var_xipos_in, var_0, var_31);
            var_48 = wp::address(var_body_rootid, var_31);
            var_50 = wp::load(var_48);
            var_49 = wp::address(var_subtree_com_in, var_0, var_50);
            var_52 = wp::load(var_47);
            var_53 = wp::load(var_49);
            var_51 = wp::sub(var_52, var_53);
            // cross_term = wp.cross(rotational_cdof, offset)                                     <L 210>
            var_54 = wp::cross(var_11, var_51);
            // accumul += wp.dot(jac, ft_body) + wp.dot(cross_term, wp.spatial_top(ft_body))       <L 211>
            var_55 = wp::dot(var_24, var_33);
            var_56 = wp::spatial_top(var_33);
            var_57 = wp::dot(var_54, var_56);
            var_58 = wp::add(var_55, var_57);
            var_59 = wp::add(var_29, var_58);
            wp::assign(var_29, var_59);
            goto start_for_0;
        end_for_0:;
        // if flg_add:                                                                            <L 213>
        if (var_flg_add) {
            // qfrc_out[worldid, dofid] += accumul                                                <L 214>
            var_60 = wp::atomic_add(var_qfrc_out, var_0, var_1, var_29);
        }
        if (!var_flg_add) {
            // qfrc_out[worldid, dofid] = accumul                                                 <L 216>
            wp::array_store(var_qfrc_out, var_0, var_1, var_29);
        }
    }
}



extern "C" __global__ void contact_force_kernel_17475cf5_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_opt_cone,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::int32> var_contact_worldid_in,
    wp::array_t<wp::float32> var_efc_force_in,
    wp::int32 var_njmax_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::int32> var_contact_ids,
    bool var_to_world_frame,
    wp::array_t<wp::vec_t<6, wp::float32>> var_out)
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
        wp::int32* var_5;
        bool var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::vec_t<6, wp::float32> var_11;
        //---------
        // forward
        // def contact_force_kernel(                                                              <L 312>
        // tid = wp.tid()                                                                         <L 330>
        var_0 = builtin_tid1d();
        // contactid = contact_ids[tid]                                                           <L 332>
        var_1 = wp::address(var_contact_ids, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // if contactid >= nacon_in[0]:                                                           <L 334>
        var_5 = wp::address(var_nacon_in, var_4);
        var_7 = wp::load(var_5);
        var_6 = (var_2 >= var_7);
        if (var_6) {
            // return                                                                             <L 335>
            continue;
        }
        // worldid = contact_worldid_in[contactid]                                                <L 337>
        var_8 = wp::address(var_contact_worldid_in, var_2);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // out[tid] = contact_force_fn(                                                           <L 339>
        // opt_cone,                                                                              <L 340>
        // contact_frame_in,                                                                      <L 341>
        // contact_friction_in,                                                                   <L 342>
        // contact_dim_in,                                                                        <L 343>
        // contact_efc_address_in,                                                                <L 344>
        // efc_force_in,                                                                          <L 345>
        // njmax_in,                                                                              <L 346>
        // nacon_in,                                                                              <L 347>
        // worldid,                                                                               <L 348>
        // contactid,                                                                             <L 349>
        // to_world_frame,                                                                        <L 350>
        var_11 = contact_force_fn_0(var_opt_cone, var_contact_frame_in, var_contact_friction_in, var_contact_dim_in, var_contact_efc_address_in, var_efc_force_in, var_njmax_in, var_nacon_in, var_9, var_2, var_to_world_frame);
        // out[tid] = contact_force_fn(                                                           <L 339>
        wp::array_store(var_out, var_0, var_11);
    }
}

