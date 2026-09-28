
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/support.py:396
static CUDA_CALLABLE void jac_dof_0(
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::vec_t<3, wp::float32> var_point,
    wp::int32 var_bodyid,
    wp::int32 var_dofid,
    wp::int32 var_worldid,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    const wp::int32 var_3 = 0;
    bool var_4;
    wp::int32 var_5;
    wp::int32 var_6;
    const wp::int32 var_7 = 0;
    bool var_8;
    bool var_9;
    const wp::int32 var_10 = 1;
    wp::int32* var_11;
    wp::int32 var_12;
    wp::int32 var_13;
    bool var_14;
    const wp::float32 var_15 = 0.0;
    wp::vec_t<3, wp::float32> var_16;
    const wp::float32 var_17 = 0.0;
    wp::vec_t<3, wp::float32> var_18;
    wp::int32* var_19;
    wp::vec_t<3, wp::float32>* var_20;
    wp::int32 var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<6, wp::float32>* var_25;
    wp::vec_t<6, wp::float32> var_26;
    wp::vec_t<6, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    //---------
    // forward
    // def jac_dof(                                                                           <L 397>
    // dof_bodyid_ = dof_bodyid[dofid]                                                        <L 411>
    var_0 = wp::address(var_dof_bodyid, var_dofid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // in_tree = int(dof_bodyid_ == 0)                                                        <L 412>
    var_4 = (var_1 == var_3);
    var_5 = wp::int(var_4);
    // parentid = bodyid                                                                      <L 413>
    var_6 = wp::copy(var_bodyid);
    // while parentid != 0:                                                                   <L 414>
    start_while_0:;
    var_8 = (var_6 != var_7);
    if ((var_8) == false) goto end_while_0;
        // if parentid == dof_bodyid_:                                                        <L 415>
        var_9 = (var_6 == var_1);
        if (var_9) {
            // in_tree = 1                                                                    <L 416>
            // break                                                                          <L 417>
            wp::assign(var_5, var_10);
            goto end_while_0;
        }
        // parentid = body_parentid[parentid]                                                 <L 418>
        var_11 = wp::address(var_body_parentid, var_6);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        wp::assign(var_6, var_12);
    goto start_while_0;
    end_while_0:;
    // if not in_tree:                                                                        <L 420>
    var_14 = wp::unot(var_5);
    if (var_14) {
        // return wp.vec3(0.0), wp.vec3(0.0)                                                  <L 421>
        var_16 = wp::vec_t<3, wp::float32>(var_15);
        var_18 = wp::vec_t<3, wp::float32>(var_17);
        ret_0 = var_16;
        ret_1 = var_18;
        return;
    }
    // offset = point - wp.vec3(subtree_com_in[worldid, body_rootid[bodyid]])                 <L 423>
    var_19 = wp::address(var_body_rootid, var_bodyid);
    var_21 = wp::load(var_19);
    var_20 = wp::address(var_subtree_com_in, var_worldid, var_21);
    var_23 = wp::load(var_20);
    var_22 = wp::vec_t<3, wp::float32>(var_23);
    var_24 = wp::sub(var_point, var_22);
    // cdof = cdof_in[worldid, dofid]                                                         <L 425>
    var_25 = wp::address(var_cdof_in, var_worldid, var_dofid);
    var_27 = wp::load(var_25);
    var_26 = wp::copy(var_27);
    // cdof_ang = wp.spatial_top(cdof)                                                        <L 426>
    var_28 = wp::spatial_top(var_26);
    // cdof_lin = wp.spatial_bottom(cdof)                                                     <L 427>
    var_29 = wp::spatial_bottom(var_26);
    // jacp = cdof_lin + wp.cross(cdof_ang, offset)                                           <L 429>
    var_30 = wp::cross(var_28, var_24);
    var_31 = wp::add(var_29, var_30);
    // jacr = cdof_ang                                                                        <L 430>
    var_32 = wp::copy(var_28);
    // return jacp, jacr                                                                      <L 432>
    ret_0 = var_31;
    ret_1 = var_32;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/constraint.py:52
static CUDA_CALLABLE void _efc_row_0(
    wp::int32 var_opt_disableflags,
    wp::int32 var_worldid,
    wp::float32 var_timestep,
    wp::int32 var_efcid,
    wp::float32 var_pos_aref,
    wp::float32 var_pos_imp,
    wp::float32 var_invweight,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::float32 var_margin,
    wp::float32 var_vel,
    wp::float32 var_frictionloss,
    wp::int32 var_type,
    wp::int32 var_id,
    wp::array_t<wp::int32> var_type_out,
    wp::array_t<wp::int32> var_id_out,
    wp::array_t<wp::float32> var_pos_out,
    wp::array_t<wp::float32> var_margin_out,
    wp::array_t<wp::float32> var_D_out,
    wp::array_t<wp::float32> var_vel_out,
    wp::array_t<wp::float32> var_aref_out,
    wp::array_t<wp::float32> var_frictionloss_out)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 0;
    wp::float32 var_5;
    const wp::int32 var_6 = 1;
    wp::float32 var_7;
    const wp::int32 var_8 = 2;
    wp::float32 var_9;
    const wp::int32 var_10 = 3;
    wp::float32 var_11;
    const wp::int32 var_12 = 4;
    wp::float32 var_13;
    const wp::int32 var_14 = 4096;
    wp::int32 var_15;
    bool var_16;
    const wp::float32 var_17 = 2.0;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    const wp::float32 var_21 = 0.0001;
    const wp::float32 var_22 = 0.0001;
    const wp::float32 var_23 = 0.9999;
    const wp::float32 var_24 = 0.9999;
    wp::float32 var_25;
    const wp::float32 var_26 = 0.0001;
    const wp::float32 var_27 = 0.0001;
    const wp::float32 var_28 = 0.9999;
    const wp::float32 var_29 = 0.9999;
    wp::float32 var_30;
    const wp::float32 var_31 = 1e-15;
    const wp::float32 var_32 = 1e-15;
    wp::float32 var_33;
    const wp::float32 var_34 = 0.0001;
    const wp::float32 var_35 = 0.0001;
    const wp::float32 var_36 = 0.9999;
    const wp::float32 var_37 = 0.9999;
    wp::float32 var_38;
    const wp::float32 var_39 = 1.0;
    wp::float32 var_40;
    wp::float32 var_41;
    const wp::float32 var_42 = 1.0;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    wp::float32 var_47;
    const wp::float32 var_48 = 2.0;
    wp::float32 var_49;
    wp::float32 var_50;
    const wp::int32 var_51 = 0;
    wp::float32 var_52;
    const wp::int32 var_53 = 0;
    bool var_54;
    const wp::int32 var_55 = 0;
    wp::float32 var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    const wp::int32 var_60 = 1;
    wp::float32 var_61;
    const wp::int32 var_62 = 0;
    bool var_63;
    const wp::int32 var_64 = 1;
    wp::float32 var_65;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    wp::float32 var_70;
    const wp::float32 var_71 = 1.0;
    const wp::float32 var_72 = 1.0;
    wp::float32 var_73;
    wp::float32 var_74;
    wp::float32 var_75;
    wp::float32 var_76;
    wp::float32 var_77;
    const wp::float32 var_78 = 1.0;
    const wp::float32 var_79 = 1.0;
    const wp::float32 var_80 = 1.0;
    wp::float32 var_81;
    const wp::float32 var_82 = 1.0;
    wp::float32 var_83;
    wp::float32 var_84;
    wp::float32 var_85;
    const wp::float32 var_86 = 1.0;
    wp::float32 var_87;
    wp::float32 var_88;
    wp::float32 var_89;
    wp::float32 var_90;
    bool var_91;
    wp::float32 var_92;
    wp::float32 var_93;
    wp::float32 var_94;
    wp::float32 var_95;
    wp::float32 var_96;
    const wp::float32 var_97 = 1.0;
    bool var_98;
    wp::float32 var_99;
    const wp::float32 var_100 = 1.0;
    const wp::float32 var_101 = 1.0;
    wp::float32 var_102;
    wp::float32 var_103;
    wp::float32 var_104;
    const wp::float32 var_105 = 1e-15;
    const wp::float32 var_106 = 1e-15;
    wp::float32 var_107;
    wp::float32 var_108;
    wp::float32 var_109;
    wp::float32 var_110;
    wp::float32 var_111;
    wp::float32 var_112;
    wp::float32 var_113;
    wp::float32 var_114;
    //---------
    // forward
    // def _efc_row(                                                                          <L 53>
    // timeconst = solref[0]                                                                  <L 81>
    var_1 = wp::extract(var_solref, var_0);
    // dampratio = solref[1]                                                                  <L 82>
    var_3 = wp::extract(var_solref, var_2);
    // dmin = solimp[0]                                                                       <L 83>
    var_5 = wp::extract(var_solimp, var_4);
    // dmax = solimp[1]                                                                       <L 84>
    var_7 = wp::extract(var_solimp, var_6);
    // width = solimp[2]                                                                      <L 85>
    var_9 = wp::extract(var_solimp, var_8);
    // mid = solimp[3]                                                                        <L 86>
    var_11 = wp::extract(var_solimp, var_10);
    // power = solimp[4]                                                                      <L 87>
    var_13 = wp::extract(var_solimp, var_12);
    // if not (opt_disableflags & DisableBit.REFSAFE):                                        <L 89>
    var_15 = wp::bit_and(var_opt_disableflags, var_14);
    var_16 = wp::unot(var_15);
    if (var_16) {
        // timeconst = wp.max(timeconst, 2.0 * timestep)                                      <L 90>
        var_18 = wp::mul(var_17, var_timestep);
        var_19 = wp::max(var_1, var_18);
    }
    var_20 = wp::where(var_16, var_19, var_1);
    // dmin = wp.clamp(dmin, types.MJ_MINIMP, types.MJ_MAXIMP)                                <L 92>
    var_25 = wp::clamp(var_5, var_22, var_24);
    // dmax = wp.clamp(dmax, types.MJ_MINIMP, types.MJ_MAXIMP)                                <L 93>
    var_30 = wp::clamp(var_7, var_27, var_29);
    // width = wp.max(types.MJ_MINVAL, width)                                                 <L 94>
    var_33 = wp::max(var_32, var_9);
    // mid = wp.clamp(mid, types.MJ_MINIMP, types.MJ_MAXIMP)                                  <L 95>
    var_38 = wp::clamp(var_11, var_35, var_37);
    // power = wp.max(1.0, power)                                                             <L 96>
    var_40 = wp::max(var_39, var_13);
    // dmax_sq = dmax * dmax                                                                  <L 99>
    var_41 = wp::mul(var_30, var_30);
    // k = 1.0 / (dmax_sq * timeconst * timeconst * dampratio * dampratio)                    <L 100>
    var_43 = wp::mul(var_41, var_20);
    var_44 = wp::mul(var_43, var_20);
    var_45 = wp::mul(var_44, var_3);
    var_46 = wp::mul(var_45, var_3);
    var_47 = wp::div(var_42, var_46);
    // b = 2.0 / (dmax * timeconst)                                                           <L 101>
    var_49 = wp::mul(var_30, var_20);
    var_50 = wp::div(var_48, var_49);
    // k = wp.where(solref[0] <= 0, -solref[0] / dmax_sq, k)                                  <L 102>
    var_52 = wp::extract(var_solref, var_51);
    var_54 = (var_52 <= var_53);
    var_56 = wp::extract(var_solref, var_55);
    var_57 = wp::neg(var_56);
    var_58 = wp::div(var_57, var_41);
    var_59 = wp::where(var_54, var_58, var_47);
    // b = wp.where(solref[1] <= 0, -solref[1] / dmax, b)                                     <L 103>
    var_61 = wp::extract(var_solref, var_60);
    var_63 = (var_61 <= var_62);
    var_65 = wp::extract(var_solref, var_64);
    var_66 = wp::neg(var_65);
    var_67 = wp::div(var_66, var_30);
    var_68 = wp::where(var_63, var_67, var_50);
    // imp_x = wp.abs(pos_imp) / width                                                        <L 105>
    var_69 = wp::abs(var_pos_imp);
    var_70 = wp::div(var_69, var_33);
    // imp_a = (1.0 / wp.pow(mid, power - 1.0)) * wp.pow(imp_x, power)                        <L 106>
    var_73 = wp::sub(var_40, var_72);
    var_74 = wp::pow(var_38, var_73);
    var_75 = wp::div(var_71, var_74);
    var_76 = wp::pow(var_70, var_40);
    var_77 = wp::mul(var_75, var_76);
    // imp_b = 1.0 - (1.0 / wp.pow(1.0 - mid, power - 1.0)) * wp.pow(1.0 - imp_x, power)       <L 107>
    var_81 = wp::sub(var_80, var_38);
    var_83 = wp::sub(var_40, var_82);
    var_84 = wp::pow(var_81, var_83);
    var_85 = wp::div(var_79, var_84);
    var_87 = wp::sub(var_86, var_70);
    var_88 = wp::pow(var_87, var_40);
    var_89 = wp::mul(var_85, var_88);
    var_90 = wp::sub(var_78, var_89);
    // imp_y = wp.where(imp_x < mid, imp_a, imp_b)                                            <L 108>
    var_91 = (var_70 < var_38);
    var_92 = wp::where(var_91, var_77, var_90);
    // imp = dmin + imp_y * (dmax - dmin)                                                     <L 109>
    var_93 = wp::sub(var_30, var_25);
    var_94 = wp::mul(var_92, var_93);
    var_95 = wp::add(var_25, var_94);
    // imp = wp.clamp(imp, dmin, dmax)                                                        <L 110>
    var_96 = wp::clamp(var_95, var_25, var_30);
    // imp = wp.where(imp_x > 1.0, dmax, imp)                                                 <L 111>
    var_98 = (var_70 > var_97);
    var_99 = wp::where(var_98, var_30, var_96);
    // D_out[worldid, efcid] = 1.0 / wp.max(invweight * (1.0 - imp) / imp, types.MJ_MINVAL)       <L 114>
    var_102 = wp::sub(var_101, var_99);
    var_103 = wp::mul(var_invweight, var_102);
    var_104 = wp::div(var_103, var_99);
    var_107 = wp::max(var_104, var_106);
    var_108 = wp::div(var_100, var_107);
    wp::array_store(var_D_out, var_worldid, var_efcid, var_108);
    // vel_out[worldid, efcid] = vel                                                          <L 115>
    wp::array_store(var_vel_out, var_worldid, var_efcid, var_vel);
    // aref_out[worldid, efcid] = -k * imp * pos_aref - b * vel                               <L 116>
    var_109 = wp::neg(var_59);
    var_110 = wp::mul(var_109, var_99);
    var_111 = wp::mul(var_110, var_pos_aref);
    var_112 = wp::mul(var_68, var_vel);
    var_113 = wp::sub(var_111, var_112);
    wp::array_store(var_aref_out, var_worldid, var_efcid, var_113);
    // pos_out[worldid, efcid] = pos_aref + margin                                            <L 117>
    var_114 = wp::add(var_pos_aref, var_margin);
    wp::array_store(var_pos_out, var_worldid, var_efcid, var_114);
    // margin_out[worldid, efcid] = margin                                                    <L 118>
    wp::array_store(var_margin_out, var_worldid, var_efcid, var_margin);
    // frictionloss_out[worldid, efcid] = frictionloss                                        <L 119>
    wp::array_store(var_frictionloss_out, var_worldid, var_efcid, var_frictionloss);
    // type_out[worldid, efcid] = type                                                        <L 120>
    wp::array_store(var_type_out, var_worldid, var_efcid, var_type);
    // id_out[worldid, efcid] = id                                                            <L 121>
    wp::array_store(var_id_out, var_worldid, var_efcid, var_id);
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/warp/_src/math.py:0
static CUDA_CALLABLE wp::float32 norm_l2_0(
    wp::vec_t<3, wp::float32> var_v)
{
    //---------
    // primal vars
    wp::float32 var_0;
    //---------
    // forward
    // def norm_l2(v: Any) -> float:                                                          <L 1>
    // return wp.length(v)                                                                    <L 12>
    var_0 = wp::length(var_v);
    return var_0;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:161
static CUDA_CALLABLE wp::vec_t<3, wp::float32> quat_to_vel_0(
    wp::quat_t<wp::float32> var_quat)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::float32 var_1;
    const wp::int32 var_2 = 2;
    wp::float32 var_3;
    const wp::int32 var_4 = 3;
    wp::float32 var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::float32 var_7;
    const wp::float32 var_8 = 0.0;
    bool var_9;
    const wp::float32 var_10 = 0.0;
    wp::vec_t<3, wp::float32> var_11;
    const wp::float32 var_12 = 2.0;
    const wp::int32 var_13 = 0;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::float32 var_17 = 3.141592653589793;
    bool var_18;
    const wp::float32 var_19 = 2.0;
    const wp::float32 var_20 = 3.141592653589793;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32> var_25;
    //---------
    // forward
    // def quat_to_vel(quat: wp.quat) -> wp.vec3:                                             <L 162>
    // axis = wp.vec3(quat[1], quat[2], quat[3])                                              <L 163>
    var_1 = wp::extract(var_quat, var_0);
    var_3 = wp::extract(var_quat, var_2);
    var_5 = wp::extract(var_quat, var_4);
    var_6 = wp::vec_t<3, wp::float32>(var_1, var_3, var_5);
    // sin_a_2 = wp.norm_l2(axis)                                                             <L 164>
    var_7 = norm_l2_0(var_6);
    // if sin_a_2 == 0.0:                                                                     <L 166>
    var_9 = (var_7 == var_8);
    if (var_9) {
        // return wp.vec3(0.0)                                                                <L 167>
        var_11 = wp::vec_t<3, wp::float32>(var_10);
        return var_11;
    }
    // speed = 2.0 * wp.atan2(sin_a_2, quat[0])                                               <L 169>
    var_14 = wp::extract(var_quat, var_13);
    var_15 = wp::atan2(var_7, var_14);
    var_16 = wp::mul(var_12, var_15);
    // if speed > wp.pi:                                                                      <L 171>
    var_18 = (var_16 > var_17);
    if (var_18) {
        // speed -= 2.0 * wp.pi                                                               <L 172>
        var_21 = wp::mul(var_19, var_20);
        var_22 = wp::sub(var_16, var_21);
    }
    var_23 = wp::where(var_18, var_22, var_16);
    // return axis * speed / sin_a_2                                                          <L 174>
    var_24 = wp::mul(var_6, var_23);
    var_25 = wp::div(var_24, var_7);
    return var_25;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:0
static CUDA_CALLABLE void normalize_with_norm_0(
    wp::vec_t<3, wp::float32> var_x,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::float32 & ret_1)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 0.0;
    bool var_2;
    const wp::float32 var_3 = 0.0;
    wp::vec_t<3, wp::float32> var_4;
    //---------
    // forward
    // def normalize_with_norm(x: Any):                                                       <L 1>
    // norm = wp.length(x)                                                                    <L 2>
    var_0 = wp::length(var_x);
    // if norm == 0.0:                                                                        <L 3>
    var_2 = (var_0 == var_1);
    if (var_2) {
        // return x, 0.0                                                                      <L 4>
        ret_0 = var_x;
        ret_1 = var_3;
        return;
    }
    // return x / norm, norm                                                                  <L 5>
    var_4 = wp::div(var_x, var_0);
    ret_0 = var_4;
    ret_1 = var_0;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:23
static CUDA_CALLABLE wp::quat_t<wp::float32> mul_quat_0(
    wp::quat_t<wp::float32> var_u,
    wp::quat_t<wp::float32> var_v)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::int32 var_5 = 1;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::int32 var_11 = 2;
    wp::float32 var_12;
    const wp::int32 var_13 = 2;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::int32 var_17 = 3;
    wp::float32 var_18;
    const wp::int32 var_19 = 3;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::int32 var_23 = 0;
    wp::float32 var_24;
    const wp::int32 var_25 = 1;
    wp::float32 var_26;
    wp::float32 var_27;
    const wp::int32 var_28 = 1;
    wp::float32 var_29;
    const wp::int32 var_30 = 0;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    const wp::int32 var_34 = 2;
    wp::float32 var_35;
    const wp::int32 var_36 = 3;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::int32 var_40 = 3;
    wp::float32 var_41;
    const wp::int32 var_42 = 2;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    const wp::int32 var_46 = 0;
    wp::float32 var_47;
    const wp::int32 var_48 = 2;
    wp::float32 var_49;
    wp::float32 var_50;
    const wp::int32 var_51 = 1;
    wp::float32 var_52;
    const wp::int32 var_53 = 3;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::float32 var_56;
    const wp::int32 var_57 = 2;
    wp::float32 var_58;
    const wp::int32 var_59 = 0;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    const wp::int32 var_63 = 3;
    wp::float32 var_64;
    const wp::int32 var_65 = 1;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    const wp::int32 var_69 = 0;
    wp::float32 var_70;
    const wp::int32 var_71 = 3;
    wp::float32 var_72;
    wp::float32 var_73;
    const wp::int32 var_74 = 1;
    wp::float32 var_75;
    const wp::int32 var_76 = 2;
    wp::float32 var_77;
    wp::float32 var_78;
    wp::float32 var_79;
    const wp::int32 var_80 = 2;
    wp::float32 var_81;
    const wp::int32 var_82 = 1;
    wp::float32 var_83;
    wp::float32 var_84;
    wp::float32 var_85;
    const wp::int32 var_86 = 3;
    wp::float32 var_87;
    const wp::int32 var_88 = 0;
    wp::float32 var_89;
    wp::float32 var_90;
    wp::float32 var_91;
    wp::quat_t<wp::float32> var_92;
    //---------
    // forward
    // def mul_quat(u: wp.quat, v: wp.quat) -> wp.quat:                                       <L 24>
    // return wp.quat(                                                                        <L 25>
    // u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],                                 <L 26>
    var_1 = wp::extract(var_u, var_0);
    var_3 = wp::extract(var_v, var_2);
    var_4 = wp::mul(var_1, var_3);
    var_6 = wp::extract(var_u, var_5);
    var_8 = wp::extract(var_v, var_7);
    var_9 = wp::mul(var_6, var_8);
    var_10 = wp::sub(var_4, var_9);
    var_12 = wp::extract(var_u, var_11);
    var_14 = wp::extract(var_v, var_13);
    var_15 = wp::mul(var_12, var_14);
    var_16 = wp::sub(var_10, var_15);
    var_18 = wp::extract(var_u, var_17);
    var_20 = wp::extract(var_v, var_19);
    var_21 = wp::mul(var_18, var_20);
    var_22 = wp::sub(var_16, var_21);
    // u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],                                 <L 27>
    var_24 = wp::extract(var_u, var_23);
    var_26 = wp::extract(var_v, var_25);
    var_27 = wp::mul(var_24, var_26);
    var_29 = wp::extract(var_u, var_28);
    var_31 = wp::extract(var_v, var_30);
    var_32 = wp::mul(var_29, var_31);
    var_33 = wp::add(var_27, var_32);
    var_35 = wp::extract(var_u, var_34);
    var_37 = wp::extract(var_v, var_36);
    var_38 = wp::mul(var_35, var_37);
    var_39 = wp::add(var_33, var_38);
    var_41 = wp::extract(var_u, var_40);
    var_43 = wp::extract(var_v, var_42);
    var_44 = wp::mul(var_41, var_43);
    var_45 = wp::sub(var_39, var_44);
    // u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],                                 <L 28>
    var_47 = wp::extract(var_u, var_46);
    var_49 = wp::extract(var_v, var_48);
    var_50 = wp::mul(var_47, var_49);
    var_52 = wp::extract(var_u, var_51);
    var_54 = wp::extract(var_v, var_53);
    var_55 = wp::mul(var_52, var_54);
    var_56 = wp::sub(var_50, var_55);
    var_58 = wp::extract(var_u, var_57);
    var_60 = wp::extract(var_v, var_59);
    var_61 = wp::mul(var_58, var_60);
    var_62 = wp::add(var_56, var_61);
    var_64 = wp::extract(var_u, var_63);
    var_66 = wp::extract(var_v, var_65);
    var_67 = wp::mul(var_64, var_66);
    var_68 = wp::add(var_62, var_67);
    // u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],                                 <L 29>
    var_70 = wp::extract(var_u, var_69);
    var_72 = wp::extract(var_v, var_71);
    var_73 = wp::mul(var_70, var_72);
    var_75 = wp::extract(var_u, var_74);
    var_77 = wp::extract(var_v, var_76);
    var_78 = wp::mul(var_75, var_77);
    var_79 = wp::add(var_73, var_78);
    var_81 = wp::extract(var_u, var_80);
    var_83 = wp::extract(var_v, var_82);
    var_84 = wp::mul(var_81, var_83);
    var_85 = wp::sub(var_79, var_84);
    var_87 = wp::extract(var_u, var_86);
    var_89 = wp::extract(var_v, var_88);
    var_90 = wp::mul(var_87, var_89);
    var_91 = wp::add(var_85, var_90);
    var_92 = wp::quat_t<wp::float32>(var_22, var_45, var_68, var_91);
    return var_92;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:115
static CUDA_CALLABLE wp::quat_t<wp::float32> quat_inv_0(
    wp::quat_t<wp::float32> var_quat)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::int32 var_5 = 2;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::int32 var_8 = 3;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::quat_t<wp::float32> var_11;
    //---------
    // forward
    // def quat_inv(quat: wp.quat) -> wp.quat:                                                <L 116>
    // return wp.quat(quat[0], -quat[1], -quat[2], -quat[3])                                  <L 117>
    var_1 = wp::extract(var_quat, var_0);
    var_3 = wp::extract(var_quat, var_2);
    var_4 = wp::neg(var_3);
    var_6 = wp::extract(var_quat, var_5);
    var_7 = wp::neg(var_6);
    var_9 = wp::extract(var_quat, var_8);
    var_10 = wp::neg(var_9);
    var_11 = wp::quat_t<wp::float32>(var_1, var_4, var_7, var_10);
    return var_11;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:33
static CUDA_CALLABLE wp::quat_t<wp::float32> quat_mul_axis_0(
    wp::quat_t<wp::float32> var_q,
    wp::vec_t<3, wp::float32> var_axis)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::float32 var_1;
    wp::float32 var_2;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    wp::float32 var_5;
    const wp::int32 var_6 = 2;
    wp::float32 var_7;
    const wp::int32 var_8 = 1;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    const wp::int32 var_12 = 3;
    wp::float32 var_13;
    const wp::int32 var_14 = 2;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    const wp::int32 var_18 = 0;
    wp::float32 var_19;
    const wp::int32 var_20 = 0;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::int32 var_23 = 2;
    wp::float32 var_24;
    const wp::int32 var_25 = 2;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    const wp::int32 var_29 = 3;
    wp::float32 var_30;
    const wp::int32 var_31 = 1;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    const wp::int32 var_35 = 0;
    wp::float32 var_36;
    const wp::int32 var_37 = 1;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::int32 var_40 = 3;
    wp::float32 var_41;
    const wp::int32 var_42 = 0;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    const wp::int32 var_46 = 1;
    wp::float32 var_47;
    const wp::int32 var_48 = 2;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    const wp::int32 var_52 = 0;
    wp::float32 var_53;
    const wp::int32 var_54 = 2;
    wp::float32 var_55;
    wp::float32 var_56;
    const wp::int32 var_57 = 1;
    wp::float32 var_58;
    const wp::int32 var_59 = 1;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    const wp::int32 var_63 = 2;
    wp::float32 var_64;
    const wp::int32 var_65 = 0;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::quat_t<wp::float32> var_69;
    //---------
    // forward
    // def quat_mul_axis(q: wp.quat, axis: wp.vec3f) -> wp.quat:                              <L 34>
    // return wp.quat(                                                                        <L 36>
    // -q[1] * axis[0] - q[2] * axis[1] - q[3] * axis[2],                                     <L 37>
    var_1 = wp::extract(var_q, var_0);
    var_2 = wp::neg(var_1);
    var_4 = wp::extract(var_axis, var_3);
    var_5 = wp::mul(var_2, var_4);
    var_7 = wp::extract(var_q, var_6);
    var_9 = wp::extract(var_axis, var_8);
    var_10 = wp::mul(var_7, var_9);
    var_11 = wp::sub(var_5, var_10);
    var_13 = wp::extract(var_q, var_12);
    var_15 = wp::extract(var_axis, var_14);
    var_16 = wp::mul(var_13, var_15);
    var_17 = wp::sub(var_11, var_16);
    // q[0] * axis[0] + q[2] * axis[2] - q[3] * axis[1],                                      <L 38>
    var_19 = wp::extract(var_q, var_18);
    var_21 = wp::extract(var_axis, var_20);
    var_22 = wp::mul(var_19, var_21);
    var_24 = wp::extract(var_q, var_23);
    var_26 = wp::extract(var_axis, var_25);
    var_27 = wp::mul(var_24, var_26);
    var_28 = wp::add(var_22, var_27);
    var_30 = wp::extract(var_q, var_29);
    var_32 = wp::extract(var_axis, var_31);
    var_33 = wp::mul(var_30, var_32);
    var_34 = wp::sub(var_28, var_33);
    // q[0] * axis[1] + q[3] * axis[0] - q[1] * axis[2],                                      <L 39>
    var_36 = wp::extract(var_q, var_35);
    var_38 = wp::extract(var_axis, var_37);
    var_39 = wp::mul(var_36, var_38);
    var_41 = wp::extract(var_q, var_40);
    var_43 = wp::extract(var_axis, var_42);
    var_44 = wp::mul(var_41, var_43);
    var_45 = wp::add(var_39, var_44);
    var_47 = wp::extract(var_q, var_46);
    var_49 = wp::extract(var_axis, var_48);
    var_50 = wp::mul(var_47, var_49);
    var_51 = wp::sub(var_45, var_50);
    // q[0] * axis[2] + q[1] * axis[1] - q[2] * axis[0],                                      <L 40>
    var_53 = wp::extract(var_q, var_52);
    var_55 = wp::extract(var_axis, var_54);
    var_56 = wp::mul(var_53, var_55);
    var_58 = wp::extract(var_q, var_57);
    var_60 = wp::extract(var_axis, var_59);
    var_61 = wp::mul(var_58, var_60);
    var_62 = wp::add(var_56, var_61);
    var_64 = wp::extract(var_q, var_63);
    var_66 = wp::extract(var_axis, var_65);
    var_67 = wp::mul(var_64, var_66);
    var_68 = wp::sub(var_62, var_67);
    var_69 = wp::quat_t<wp::float32>(var_17, var_34, var_51, var_68);
    return var_69;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/support.py:396
static CUDA_CALLABLE void adj_jac_dof_0(
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::vec_t<3, wp::float32> var_point,
    wp::int32 var_bodyid,
    wp::int32 var_dofid,
    wp::int32 var_worldid,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::array_t<wp::int32> & adj_body_parentid,
    wp::array_t<wp::int32> & adj_body_rootid,
    wp::array_t<wp::int32> & adj_dof_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cdof_in,
    wp::vec_t<3, wp::float32> & adj_point,
    wp::int32 & adj_bodyid,
    wp::int32 & adj_dofid,
    wp::int32 & adj_worldid,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/constraint.py:52
static CUDA_CALLABLE void adj__efc_row_0(
    wp::int32 var_opt_disableflags,
    wp::int32 var_worldid,
    wp::float32 var_timestep,
    wp::int32 var_efcid,
    wp::float32 var_pos_aref,
    wp::float32 var_pos_imp,
    wp::float32 var_invweight,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::float32 var_margin,
    wp::float32 var_vel,
    wp::float32 var_frictionloss,
    wp::int32 var_type,
    wp::int32 var_id,
    wp::array_t<wp::int32> var_type_out,
    wp::array_t<wp::int32> var_id_out,
    wp::array_t<wp::float32> var_pos_out,
    wp::array_t<wp::float32> var_margin_out,
    wp::array_t<wp::float32> var_D_out,
    wp::array_t<wp::float32> var_vel_out,
    wp::array_t<wp::float32> var_aref_out,
    wp::array_t<wp::float32> var_frictionloss_out,
    wp::int32 & adj_opt_disableflags,
    wp::int32 & adj_worldid,
    wp::float32 & adj_timestep,
    wp::int32 & adj_efcid,
    wp::float32 & adj_pos_aref,
    wp::float32 & adj_pos_imp,
    wp::float32 & adj_invweight,
    wp::vec_t<2, wp::float32> & adj_solref,
    wp::vec_t<5, wp::float32> & adj_solimp,
    wp::float32 & adj_margin,
    wp::float32 & adj_vel,
    wp::float32 & adj_frictionloss,
    wp::int32 & adj_type,
    wp::int32 & adj_id,
    wp::array_t<wp::int32> & adj_type_out,
    wp::array_t<wp::int32> & adj_id_out,
    wp::array_t<wp::float32> & adj_pos_out,
    wp::array_t<wp::float32> & adj_margin_out,
    wp::array_t<wp::float32> & adj_D_out,
    wp::array_t<wp::float32> & adj_vel_out,
    wp::array_t<wp::float32> & adj_aref_out,
    wp::array_t<wp::float32> & adj_frictionloss_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/warp/_src/math.py:0
static CUDA_CALLABLE void adj_norm_l2_0(
    wp::vec_t<3, wp::float32> var_v,
    wp::vec_t<3, wp::float32> & adj_v,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:161
static CUDA_CALLABLE void adj_quat_to_vel_0(
    wp::quat_t<wp::float32> var_quat,
    wp::quat_t<wp::float32> & adj_quat,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:0
static CUDA_CALLABLE void adj_normalize_with_norm_0(
    wp::vec_t<3, wp::float32> var_x,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::float32 & ret_1,
    wp::vec_t<3, wp::float32> & adj_x,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::float32 & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:23
static CUDA_CALLABLE void adj_mul_quat_0(
    wp::quat_t<wp::float32> var_u,
    wp::quat_t<wp::float32> var_v,
    wp::quat_t<wp::float32> & adj_u,
    wp::quat_t<wp::float32> & adj_v,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:115
static CUDA_CALLABLE void adj_quat_inv_0(
    wp::quat_t<wp::float32> var_quat,
    wp::quat_t<wp::float32> & adj_quat,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:33
static CUDA_CALLABLE void adj_quat_mul_axis_0(
    wp::quat_t<wp::float32> var_q,
    wp::vec_t<3, wp::float32> var_axis,
    wp::quat_t<wp::float32> & adj_q,
    wp::vec_t<3, wp::float32> & adj_axis,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void _equality_connect_08b61158_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_nv,
    wp::int32 var_nsite,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::int32 var_opt_disableflags,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_weldid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::vec_t<2, wp::float32>> var_body_invweight0,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::int32> var_dof_parentid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_eq_obj1id,
    wp::array_t<wp::int32> var_eq_obj2id,
    wp::array_t<wp::int32> var_eq_objtype,
    wp::array_t<wp::vec_t<2, wp::float32>> var_eq_solref,
    wp::array_t<wp::vec_t<5, wp::float32>> var_eq_solimp,
    wp::array_t<wp::vec_t<11, wp::float32>> var_eq_data,
    bool var_is_sparse,
    wp::array_t<wp::int32> var_eq_connect_adr,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<bool> var_eq_active_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::int32 var_njmax_in,
    wp::int32 var_njmax_nnz_in,
    wp::array_t<wp::int32> var_ne_out,
    wp::array_t<wp::int32> var_nefc_out,
    wp::array_t<wp::int32> var_efc_type_out,
    wp::array_t<wp::int32> var_efc_id_out,
    wp::array_t<wp::int32> var_efc_J_rownnz_out,
    wp::array_t<wp::int32> var_efc_J_rowadr_out,
    wp::array_t<wp::int32> var_efc_J_colind_out,
    wp::array_t<wp::float32> var_efc_J_out,
    wp::array_t<wp::float32> var_efc_pos_out,
    wp::array_t<wp::float32> var_efc_margin_out,
    wp::array_t<wp::float32> var_efc_D_out,
    wp::array_t<wp::float32> var_efc_vel_out,
    wp::array_t<wp::float32> var_efc_aref_out,
    wp::array_t<wp::float32> var_efc_frictionloss_out,
    wp::array_t<wp::int32> var_efc_nnz_out)
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
        bool* var_5;
        bool var_6;
        bool var_7;
        const wp::int32 var_8 = 3;
        wp::int32 var_9;
        const wp::int32 var_10 = 3;
        wp::int32 var_11;
        const wp::int32 var_12 = 3;
        wp::int32 var_13;
        bool var_14;
        const wp::int32 var_15 = 0;
        wp::int32 var_16;
        const wp::int32 var_17 = 1;
        wp::int32 var_18;
        const wp::int32 var_19 = 2;
        wp::int32 var_20;
        wp::shape_t* var_21;
        const wp::int32 var_22 = 0;
        wp::int32 var_23;
        wp::shape_t var_24;
        wp::int32 var_25;
        wp::vec_t<11, wp::float32>* var_26;
        wp::vec_t<11, wp::float32> var_27;
        wp::vec_t<11, wp::float32> var_28;
        const wp::int32 var_29 = 0;
        wp::float32 var_30;
        const wp::int32 var_31 = 1;
        wp::float32 var_32;
        const wp::int32 var_33 = 2;
        wp::float32 var_34;
        wp::vec_t<3, wp::float32> var_35;
        const wp::int32 var_36 = 3;
        wp::float32 var_37;
        const wp::int32 var_38 = 4;
        wp::float32 var_39;
        const wp::int32 var_40 = 5;
        wp::float32 var_41;
        wp::vec_t<3, wp::float32> var_42;
        wp::int32* var_43;
        wp::int32 var_44;
        wp::int32 var_45;
        wp::int32* var_46;
        wp::int32 var_47;
        wp::int32 var_48;
        const wp::int32 var_49 = 0;
        bool var_50;
        wp::int32* var_51;
        const wp::int32 var_52 = 6;
        bool var_53;
        wp::int32 var_54;
        bool var_55;
        wp::int32* var_56;
        wp::int32 var_57;
        wp::int32 var_58;
        wp::int32* var_59;
        wp::int32 var_60;
        wp::int32 var_61;
        wp::vec_t<3, wp::float32>* var_62;
        wp::vec_t<3, wp::float32> var_63;
        wp::vec_t<3, wp::float32> var_64;
        wp::vec_t<3, wp::float32>* var_65;
        wp::vec_t<3, wp::float32> var_66;
        wp::vec_t<3, wp::float32> var_67;
        wp::int32 var_68;
        wp::int32 var_69;
        wp::vec_t<3, wp::float32>* var_70;
        wp::mat_t<3, 3, wp::float32>* var_71;
        wp::vec_t<3, wp::float32> var_72;
        wp::mat_t<3, 3, wp::float32> var_73;
        wp::vec_t<3, wp::float32> var_74;
        wp::vec_t<3, wp::float32> var_75;
        wp::vec_t<3, wp::float32>* var_76;
        wp::mat_t<3, 3, wp::float32>* var_77;
        wp::vec_t<3, wp::float32> var_78;
        wp::mat_t<3, 3, wp::float32> var_79;
        wp::vec_t<3, wp::float32> var_80;
        wp::vec_t<3, wp::float32> var_81;
        wp::int32 var_82;
        wp::int32 var_83;
        wp::vec_t<3, wp::float32> var_84;
        wp::vec_t<3, wp::float32> var_85;
        wp::vec_t<3, wp::float32> var_86;
        const wp::float32 var_87 = 0.0;
        const wp::float32 var_88 = 0.0;
        const wp::float32 var_89 = 0.0;
        wp::vec_t<3, wp::float32> var_90;
        wp::int32* var_91;
        wp::int32 var_92;
        wp::int32 var_93;
        wp::int32* var_94;
        wp::int32 var_95;
        wp::int32 var_96;
        wp::int32* var_97;
        wp::int32* var_98;
        wp::int32 var_99;
        wp::int32 var_100;
        wp::int32 var_101;
        const wp::int32 var_102 = 1;
        wp::int32 var_103;
        wp::int32 var_104;
        wp::int32* var_105;
        wp::int32* var_106;
        wp::int32 var_107;
        wp::int32 var_108;
        wp::int32 var_109;
        const wp::int32 var_110 = 1;
        wp::int32 var_111;
        wp::int32 var_112;
        wp::int32 var_113;
        wp::int32 var_114;
        const wp::int32 var_115 = 0;
        wp::int32 var_116;
        const wp::int32 var_117 = 0;
        bool var_118;
        const wp::int32 var_119 = 0;
        bool var_120;
        bool var_121;
        wp::int32 var_122;
        bool var_123;
        wp::int32* var_124;
        wp::int32 var_125;
        wp::int32 var_126;
        wp::int32 var_127;
        bool var_128;
        wp::int32* var_129;
        wp::int32 var_130;
        wp::int32 var_131;
        wp::int32 var_132;
        const wp::int32 var_133 = 1;
        wp::int32 var_134;
        const wp::int32 var_135 = 3;
        wp::int32 var_136;
        wp::int32 var_137;
        const wp::int32 var_138 = 3;
        wp::int32 var_139;
        wp::int32 var_140;
        bool var_141;
        wp::int32 var_142;
        const wp::int32 var_143 = 2;
        wp::int32 var_144;
        wp::int32 var_145;
        const wp::int32 var_146 = 0;
        wp::int32 var_147;
        const wp::int32 var_148 = 0;
        bool var_149;
        const wp::int32 var_150 = 0;
        bool var_151;
        bool var_152;
        wp::int32 var_153;
        bool var_154;
        wp::int32* var_155;
        wp::int32 var_156;
        wp::int32 var_157;
        wp::int32 var_158;
        bool var_159;
        wp::int32* var_160;
        wp::int32 var_161;
        wp::int32 var_162;
        wp::int32 var_163;
        wp::vec_t<3, wp::float32> var_164;
        wp::vec_t<3, wp::float32> var_165;
        wp::vec_t<3, wp::float32> var_166;
        wp::vec_t<3, wp::float32> var_167;
        wp::vec_t<3, wp::float32> var_168;
        wp::int32 var_169;
        wp::int32 var_170;
        wp::int32 var_171;
        const wp::int32 var_172 = 2;
        wp::int32 var_173;
        wp::int32 var_174;
        wp::int32 var_175;
        const wp::int32 var_176 = 0;
        const wp::int32 var_177 = 0;
        const wp::int32 var_178 = 0;
        const wp::int32 var_179 = 0;
        wp::float32 var_180;
        const wp::int32 var_181 = 0;
        const wp::int32 var_182 = 1;
        wp::float32 var_183;
        const wp::int32 var_184 = 0;
        const wp::int32 var_185 = 2;
        wp::float32 var_186;
        const wp::int32 var_187 = 0;
        wp::float32* var_188;
        wp::vec_t<3, wp::float32> var_189;
        wp::float32 var_190;
        wp::vec_t<3, wp::float32> var_191;
        const wp::int32 var_192 = 1;
        wp::int32 var_193;
        wp::int32 var_194;
        wp::int32 var_195;
        wp::range_t var_196;
        wp::int32 var_197;
        wp::vec_t<3, wp::float32> var_198;
        wp::vec_t<3, wp::float32> var_199;
        wp::vec_t<3, wp::float32> var_200;
        wp::vec_t<3, wp::float32> var_201;
        wp::vec_t<3, wp::float32> var_202;
        const wp::int32 var_203 = 0;
        wp::float32 var_204;
        const wp::int32 var_205 = 1;
        wp::float32 var_206;
        const wp::int32 var_207 = 2;
        wp::float32 var_208;
        wp::float32* var_209;
        wp::vec_t<3, wp::float32> var_210;
        wp::float32 var_211;
        wp::vec_t<3, wp::float32> var_212;
        wp::shape_t* var_213;
        const wp::int32 var_214 = 0;
        wp::int32 var_215;
        wp::shape_t var_216;
        wp::int32 var_217;
        wp::vec_t<2, wp::float32>* var_218;
        const wp::int32 var_219 = 0;
        wp::float32 var_220;
        wp::vec_t<2, wp::float32> var_221;
        wp::vec_t<2, wp::float32>* var_222;
        const wp::int32 var_223 = 0;
        wp::float32 var_224;
        wp::vec_t<2, wp::float32> var_225;
        wp::float32 var_226;
        wp::float32 var_227;
        wp::shape_t* var_228;
        const wp::int32 var_229 = 0;
        wp::int32 var_230;
        wp::shape_t var_231;
        wp::int32 var_232;
        wp::vec_t<2, wp::float32>* var_233;
        wp::vec_t<2, wp::float32> var_234;
        wp::vec_t<2, wp::float32> var_235;
        wp::shape_t* var_236;
        const wp::int32 var_237 = 0;
        wp::int32 var_238;
        wp::shape_t var_239;
        wp::int32 var_240;
        wp::vec_t<5, wp::float32>* var_241;
        wp::vec_t<5, wp::float32> var_242;
        wp::vec_t<5, wp::float32> var_243;
        wp::shape_t* var_244;
        const wp::int32 var_245 = 0;
        wp::int32 var_246;
        wp::shape_t var_247;
        wp::int32 var_248;
        wp::float32* var_249;
        wp::float32 var_250;
        wp::float32 var_251;
        const wp::int32 var_252 = 0;
        wp::int32 var_253;
        wp::float32 var_254;
        const wp::float32 var_255 = 0.0;
        wp::float32 var_256;
        const wp::float32 var_257 = 0.0;
        const wp::int32 var_258 = 0;
        const wp::int32 var_259 = 0;
        const wp::int32 var_260 = 1;
        wp::int32 var_261;
        wp::float32 var_262;
        const wp::float32 var_263 = 0.0;
        wp::float32 var_264;
        const wp::float32 var_265 = 0.0;
        const wp::int32 var_266 = 0;
        const wp::int32 var_267 = 0;
        const wp::int32 var_268 = 2;
        wp::int32 var_269;
        wp::float32 var_270;
        const wp::float32 var_271 = 0.0;
        wp::float32 var_272;
        const wp::float32 var_273 = 0.0;
        const wp::int32 var_274 = 0;
        const wp::int32 var_275 = 0;
        //---------
        // forward
        // def _equality_connect(                                                                 <L 125>
        // worldid, eqconnectid = wp.tid()                                                        <L 177>
        builtin_tid2d(var_0, var_1);
        // eqid = eq_connect_adr[eqconnectid]                                                     <L 178>
        var_2 = wp::address(var_eq_connect_adr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if not eq_active_in[worldid, eqid]:                                                    <L 180>
        var_5 = wp::address(var_eq_active_in, var_0, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::unot(var_7);
        if (var_6) {
            // return                                                                             <L 181>
            continue;
        }
        // wp.atomic_add(ne_out, worldid, 3)                                                      <L 183>
        var_9 = wp::atomic_add(var_ne_out, var_0, var_8);
        // efcid = wp.atomic_add(nefc_out, worldid, 3)                                            <L 184>
        var_11 = wp::atomic_add(var_nefc_out, var_0, var_10);
        // if efcid >= njmax_in - 3:                                                              <L 186>
        var_13 = wp::sub(var_njmax_in, var_12);
        var_14 = (var_11 >= var_13);
        if (var_14) {
            // return                                                                             <L 187>
            continue;
        }
        // efcid0 = efcid + 0                                                                     <L 189>
        var_16 = wp::add(var_11, var_15);
        // efcid1 = efcid + 1                                                                     <L 190>
        var_18 = wp::add(var_11, var_17);
        // efcid2 = efcid + 2                                                                     <L 191>
        var_20 = wp::add(var_11, var_19);
        // data = eq_data[worldid % eq_data.shape[0], eqid]                                       <L 193>
        var_21 = &(var_eq_data.shape);
        var_24 = wp::load(var_21);
        var_23 = wp::extract(var_24, var_22);
        var_25 = wp::mod(var_0, var_23);
        var_26 = wp::address(var_eq_data, var_25, var_3);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // anchor1 = wp.vec3f(data[0], data[1], data[2])                                          <L 194>
        var_30 = wp::extract(var_27, var_29);
        var_32 = wp::extract(var_27, var_31);
        var_34 = wp::extract(var_27, var_33);
        var_35 = wp::vec_t<3, wp::float32>(var_30, var_32, var_34);
        // anchor2 = wp.vec3f(data[3], data[4], data[5])                                          <L 195>
        var_37 = wp::extract(var_27, var_36);
        var_39 = wp::extract(var_27, var_38);
        var_41 = wp::extract(var_27, var_40);
        var_42 = wp::vec_t<3, wp::float32>(var_37, var_39, var_41);
        // obj1id = eq_obj1id[eqid]                                                               <L 197>
        var_43 = wp::address(var_eq_obj1id, var_3);
        var_45 = wp::load(var_43);
        var_44 = wp::copy(var_45);
        // obj2id = eq_obj2id[eqid]                                                               <L 198>
        var_46 = wp::address(var_eq_obj2id, var_3);
        var_48 = wp::load(var_46);
        var_47 = wp::copy(var_48);
        // if nsite > 0 and eq_objtype[eqid] == types.ObjType.SITE:                               <L 200>
        var_50 = (var_nsite > var_49);
        var_51 = wp::address(var_eq_objtype, var_3);
        var_54 = wp::load(var_51);
        var_53 = (var_54 == var_52);
        var_55 = var_50 && var_53;
        if (var_55) {
            // body1 = site_bodyid[obj1id]                                                        <L 201>
            var_56 = wp::address(var_site_bodyid, var_44);
            var_58 = wp::load(var_56);
            var_57 = wp::copy(var_58);
            // body2 = site_bodyid[obj2id]                                                        <L 202>
            var_59 = wp::address(var_site_bodyid, var_47);
            var_61 = wp::load(var_59);
            var_60 = wp::copy(var_61);
            // pos1 = site_xpos_in[worldid, obj1id]                                               <L 203>
            var_62 = wp::address(var_site_xpos_in, var_0, var_44);
            var_64 = wp::load(var_62);
            var_63 = wp::copy(var_64);
            // pos2 = site_xpos_in[worldid, obj2id]                                               <L 204>
            var_65 = wp::address(var_site_xpos_in, var_0, var_47);
            var_67 = wp::load(var_65);
            var_66 = wp::copy(var_67);
        }
        if (!var_55) {
            // body1 = obj1id                                                                     <L 206>
            var_68 = wp::copy(var_44);
            // body2 = obj2id                                                                     <L 207>
            var_69 = wp::copy(var_47);
            // pos1 = xpos_in[worldid, body1] + xmat_in[worldid, body1] @ anchor1                 <L 208>
            var_70 = wp::address(var_xpos_in, var_0, var_68);
            var_71 = wp::address(var_xmat_in, var_0, var_68);
            var_73 = wp::load(var_71);
            var_72 = wp::mul(var_73, var_35);
            var_75 = wp::load(var_70);
            var_74 = wp::add(var_75, var_72);
            // pos2 = xpos_in[worldid, body2] + xmat_in[worldid, body2] @ anchor2                 <L 209>
            var_76 = wp::address(var_xpos_in, var_0, var_69);
            var_77 = wp::address(var_xmat_in, var_0, var_69);
            var_79 = wp::load(var_77);
            var_78 = wp::mul(var_79, var_42);
            var_81 = wp::load(var_76);
            var_80 = wp::add(var_81, var_78);
        }
        var_82 = wp::where(var_55, var_57, var_68);
        var_83 = wp::where(var_55, var_60, var_69);
        var_84 = wp::where(var_55, var_63, var_74);
        var_85 = wp::where(var_55, var_66, var_80);
        // pos = pos1 - pos2                                                                      <L 212>
        var_86 = wp::sub(var_84, var_85);
        // Jqvel = wp.vec3f(0.0, 0.0, 0.0)                                                        <L 215>
        var_90 = wp::vec_t<3, wp::float32>(var_87, var_88, var_89);
        // if is_sparse:                                                                          <L 217>
        if (var_is_sparse) {
            // body1 = body_weldid[body1]                                                         <L 219>
            var_91 = wp::address(var_body_weldid, var_82);
            var_93 = wp::load(var_91);
            var_92 = wp::copy(var_93);
            // body2 = body_weldid[body2]                                                         <L 220>
            var_94 = wp::address(var_body_weldid, var_83);
            var_96 = wp::load(var_94);
            var_95 = wp::copy(var_96);
            // da1 = int(body_dofadr[body1] + body_dofnum[body1] - 1)                             <L 222>
            var_97 = wp::address(var_body_dofadr, var_92);
            var_98 = wp::address(var_body_dofnum, var_92);
            var_100 = wp::load(var_97);
            var_101 = wp::load(var_98);
            var_99 = wp::add(var_100, var_101);
            var_103 = wp::sub(var_99, var_102);
            var_104 = wp::int(var_103);
            // da2 = int(body_dofadr[body2] + body_dofnum[body2] - 1)                             <L 223>
            var_105 = wp::address(var_body_dofadr, var_95);
            var_106 = wp::address(var_body_dofnum, var_95);
            var_108 = wp::load(var_105);
            var_109 = wp::load(var_106);
            var_107 = wp::add(var_108, var_109);
            var_111 = wp::sub(var_107, var_110);
            var_112 = wp::int(var_111);
            // pda1 = da1                                                                         <L 226>
            var_113 = wp::copy(var_104);
            // pda2 = da2                                                                         <L 227>
            var_114 = wp::copy(var_112);
            // rownnz = int(0)                                                                    <L 228>
            var_116 = wp::int(var_115);
            // while pda1 >= 0 or pda2 >= 0:                                                      <L 229>
        start_while_2:;
            var_118 = (var_113 >= var_117);
            var_120 = (var_114 >= var_119);
            var_121 = var_118 || var_120;
        if ((var_121) == false) goto end_while_2;
                // da = wp.max(pda1, pda2)                                                        <L 230>
                var_122 = wp::max(var_113, var_114);
                // if pda1 == da:                                                                 <L 231>
                var_123 = (var_113 == var_122);
                if (var_123) {
                    // pda1 = dof_parentid[pda1]                                                  <L 232>
                    var_124 = wp::address(var_dof_parentid, var_113);
                    var_126 = wp::load(var_124);
                    var_125 = wp::copy(var_126);
                }
                var_127 = wp::where(var_123, var_125, var_113);
                // if pda2 == da:                                                                 <L 233>
                var_128 = (var_114 == var_122);
                if (var_128) {
                    // pda2 = dof_parentid[pda2]                                                  <L 234>
                    var_129 = wp::address(var_dof_parentid, var_114);
                    var_131 = wp::load(var_129);
                    var_130 = wp::copy(var_131);
                }
                var_132 = wp::where(var_128, var_130, var_114);
                // rownnz += 1                                                                    <L 235>
                var_134 = wp::add(var_116, var_133);
                wp::assign(var_113, var_127);
                wp::assign(var_114, var_132);
                wp::assign(var_116, var_134);
        goto start_while_2;
        end_while_2:;
            // rowadr = wp.atomic_add(efc_nnz_out, worldid, 3 * rownnz)                           <L 238>
            var_136 = wp::mul(var_135, var_116);
            var_137 = wp::atomic_add(var_efc_nnz_out, var_0, var_136);
            // if rowadr + 3 * rownnz > njmax_nnz_in:                                             <L 239>
            var_139 = wp::mul(var_138, var_116);
            var_140 = wp::add(var_137, var_139);
            var_141 = (var_140 > var_njmax_nnz_in);
            if (var_141) {
                // return                                                                         <L 240>
                continue;
            }
            // efc_J_rowadr_out[worldid, efcid0] = rowadr                                         <L 241>
            wp::array_store(var_efc_J_rowadr_out, var_0, var_16, var_137);
            // efc_J_rowadr_out[worldid, efcid1] = rowadr + rownnz                                <L 242>
            var_142 = wp::add(var_137, var_116);
            wp::array_store(var_efc_J_rowadr_out, var_0, var_18, var_142);
            // efc_J_rowadr_out[worldid, efcid2] = rowadr + 2 * rownnz                            <L 243>
            var_144 = wp::mul(var_143, var_116);
            var_145 = wp::add(var_137, var_144);
            wp::array_store(var_efc_J_rowadr_out, var_0, var_20, var_145);
            // efc_J_rownnz_out[worldid, efcid0] = rownnz                                         <L 245>
            wp::array_store(var_efc_J_rownnz_out, var_0, var_16, var_116);
            // efc_J_rownnz_out[worldid, efcid1] = rownnz                                         <L 246>
            wp::array_store(var_efc_J_rownnz_out, var_0, var_18, var_116);
            // efc_J_rownnz_out[worldid, efcid2] = rownnz                                         <L 247>
            wp::array_store(var_efc_J_rownnz_out, var_0, var_20, var_116);
            // nnz = int(0)                                                                       <L 250>
            var_147 = wp::int(var_146);
            // while da1 >= 0 or da2 >= 0:                                                        <L 251>
        start_while_5:;
            var_149 = (var_104 >= var_148);
            var_151 = (var_112 >= var_150);
            var_152 = var_149 || var_151;
        if ((var_152) == false) goto end_while_5;
                // da = wp.max(da1, da2)                                                          <L 252>
                var_153 = wp::max(var_104, var_112);
                // if da1 == da:                                                                  <L 253>
                var_154 = (var_104 == var_153);
                if (var_154) {
                    // da1 = dof_parentid[da1]                                                    <L 254>
                    var_155 = wp::address(var_dof_parentid, var_104);
                    var_157 = wp::load(var_155);
                    var_156 = wp::copy(var_157);
                }
                var_158 = wp::where(var_154, var_156, var_104);
                // if da2 == da:                                                                  <L 255>
                var_159 = (var_112 == var_153);
                if (var_159) {
                    // da2 = dof_parentid[da2]                                                    <L 256>
                    var_160 = wp::address(var_dof_parentid, var_112);
                    var_162 = wp::load(var_160);
                    var_161 = wp::copy(var_162);
                }
                var_163 = wp::where(var_159, var_161, var_112);
                // jacp1, _ = support.jac_dof(                                                    <L 258>
                // body_parentid,                                                                 <L 259>
                // body_rootid,                                                                   <L 260>
                // dof_bodyid,                                                                    <L 261>
                // subtree_com_in,                                                                <L 262>
                // cdof_in,                                                                       <L 263>
                // pos1,                                                                          <L 264>
                // body1,                                                                         <L 265>
                // da,                                                                            <L 266>
                // worldid,                                                                       <L 267>
                jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_subtree_com_in, var_cdof_in, var_84, var_92, var_153, var_0, var_164, var_165);
                // jacp2, _ = support.jac_dof(                                                    <L 269>
                // body_parentid,                                                                 <L 270>
                // body_rootid,                                                                   <L 271>
                // dof_bodyid,                                                                    <L 272>
                // subtree_com_in,                                                                <L 273>
                // cdof_in,                                                                       <L 274>
                // pos2,                                                                          <L 275>
                // body2,                                                                         <L 276>
                // da,                                                                            <L 277>
                // worldid,                                                                       <L 278>
                jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_subtree_com_in, var_cdof_in, var_85, var_95, var_153, var_0, var_166, var_167);
                // j1mj2 = jacp1 - jacp2                                                          <L 280>
                var_168 = wp::sub(var_164, var_166);
                // sparseid0 = rowadr + nnz                                                       <L 282>
                var_169 = wp::add(var_137, var_147);
                // sparseid1 = rowadr + rownnz + nnz                                              <L 283>
                var_170 = wp::add(var_137, var_116);
                var_171 = wp::add(var_170, var_147);
                // sparseid2 = rowadr + 2 * rownnz + nnz                                          <L 284>
                var_173 = wp::mul(var_172, var_116);
                var_174 = wp::add(var_137, var_173);
                var_175 = wp::add(var_174, var_147);
                // efc_J_colind_out[worldid, 0, sparseid0] = da                                   <L 286>
                wp::array_store(var_efc_J_colind_out, var_0, var_176, var_169, var_153);
                // efc_J_colind_out[worldid, 0, sparseid1] = da                                   <L 287>
                wp::array_store(var_efc_J_colind_out, var_0, var_177, var_171, var_153);
                // efc_J_colind_out[worldid, 0, sparseid2] = da                                   <L 288>
                wp::array_store(var_efc_J_colind_out, var_0, var_178, var_175, var_153);
                // efc_J_out[worldid, 0, sparseid0] = j1mj2[0]                                    <L 290>
                var_180 = wp::extract(var_168, var_179);
                wp::array_store(var_efc_J_out, var_0, var_181, var_169, var_180);
                // efc_J_out[worldid, 0, sparseid1] = j1mj2[1]                                    <L 291>
                var_183 = wp::extract(var_168, var_182);
                wp::array_store(var_efc_J_out, var_0, var_184, var_171, var_183);
                // efc_J_out[worldid, 0, sparseid2] = j1mj2[2]                                    <L 292>
                var_186 = wp::extract(var_168, var_185);
                wp::array_store(var_efc_J_out, var_0, var_187, var_175, var_186);
                // Jqvel += j1mj2 * qvel_in[worldid, da]                                          <L 294>
                var_188 = wp::address(var_qvel_in, var_0, var_153);
                var_190 = wp::load(var_188);
                var_189 = wp::mul(var_168, var_190);
                var_191 = wp::add(var_90, var_189);
                // nnz += 1                                                                       <L 296>
                var_193 = wp::add(var_147, var_192);
                wp::assign(var_90, var_191);
                wp::assign(var_104, var_158);
                wp::assign(var_112, var_163);
                wp::assign(var_122, var_153);
                wp::assign(var_147, var_193);
        goto start_while_5;
        end_while_5:;
        }
        var_194 = wp::where(var_is_sparse, var_92, var_82);
        var_195 = wp::where(var_is_sparse, var_95, var_83);
        if (!var_is_sparse) {
            // for dofid in range(nv):                                                            <L 299>
            var_196 = wp::range(var_nv);
            start_for_7:;
                if (iter_cmp(var_196) == 0) goto end_for_7;
                var_197 = wp::iter_next(var_196);
                // jacp1, _ = support.jac_dof(                                                    <L 300>
                // body_parentid,                                                                 <L 301>
                // body_rootid,                                                                   <L 302>
                // dof_bodyid,                                                                    <L 303>
                // subtree_com_in,                                                                <L 304>
                // cdof_in,                                                                       <L 305>
                // pos1,                                                                          <L 306>
                // body1,                                                                         <L 307>
                // dofid,                                                                         <L 308>
                // worldid,                                                                       <L 309>
                jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_subtree_com_in, var_cdof_in, var_84, var_194, var_197, var_0, var_198, var_199);
                // jacp2, _ = support.jac_dof(                                                    <L 311>
                // body_parentid,                                                                 <L 312>
                // body_rootid,                                                                   <L 313>
                // dof_bodyid,                                                                    <L 314>
                // subtree_com_in,                                                                <L 315>
                // cdof_in,                                                                       <L 316>
                // pos2,                                                                          <L 317>
                // body2,                                                                         <L 318>
                // dofid,                                                                         <L 319>
                // worldid,                                                                       <L 320>
                jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_subtree_com_in, var_cdof_in, var_85, var_195, var_197, var_0, var_200, var_201);
                // j1mj2 = jacp1 - jacp2                                                          <L 322>
                var_202 = wp::sub(var_198, var_200);
                // efc_J_out[worldid, efcid0, dofid] = j1mj2[0]                                   <L 324>
                var_204 = wp::extract(var_202, var_203);
                wp::array_store(var_efc_J_out, var_0, var_16, var_197, var_204);
                // efc_J_out[worldid, efcid1, dofid] = j1mj2[1]                                   <L 325>
                var_206 = wp::extract(var_202, var_205);
                wp::array_store(var_efc_J_out, var_0, var_18, var_197, var_206);
                // efc_J_out[worldid, efcid2, dofid] = j1mj2[2]                                   <L 326>
                var_208 = wp::extract(var_202, var_207);
                wp::array_store(var_efc_J_out, var_0, var_20, var_197, var_208);
                // Jqvel += j1mj2 * qvel_in[worldid, dofid]                                       <L 328>
                var_209 = wp::address(var_qvel_in, var_0, var_197);
                var_211 = wp::load(var_209);
                var_210 = wp::mul(var_202, var_211);
                var_212 = wp::add(var_90, var_210);
                wp::assign(var_90, var_212);
                wp::assign(var_164, var_198);
                wp::assign(var_167, var_201);
                wp::assign(var_166, var_200);
                wp::assign(var_168, var_202);
                goto start_for_7;
            end_for_7:;
        }
        // body_invweight0_id = worldid % body_invweight0.shape[0]                                <L 330>
        var_213 = &(var_body_invweight0.shape);
        var_216 = wp::load(var_213);
        var_215 = wp::extract(var_216, var_214);
        var_217 = wp::mod(var_0, var_215);
        // invweight = body_invweight0[body_invweight0_id, body1][0] + body_invweight0[body_invweight0_id, body2][0]       <L 331>
        var_218 = wp::address(var_body_invweight0, var_217, var_194);
        var_221 = wp::load(var_218);
        var_220 = wp::extract(var_221, var_219);
        var_222 = wp::address(var_body_invweight0, var_217, var_195);
        var_225 = wp::load(var_222);
        var_224 = wp::extract(var_225, var_223);
        var_226 = wp::add(var_220, var_224);
        // pos_imp = wp.length(pos)                                                               <L 332>
        var_227 = wp::length(var_86);
        // solref = eq_solref[worldid % eq_solref.shape[0], eqid]                                 <L 334>
        var_228 = &(var_eq_solref.shape);
        var_231 = wp::load(var_228);
        var_230 = wp::extract(var_231, var_229);
        var_232 = wp::mod(var_0, var_230);
        var_233 = wp::address(var_eq_solref, var_232, var_3);
        var_235 = wp::load(var_233);
        var_234 = wp::copy(var_235);
        // solimp = eq_solimp[worldid % eq_solimp.shape[0], eqid]                                 <L 335>
        var_236 = &(var_eq_solimp.shape);
        var_239 = wp::load(var_236);
        var_238 = wp::extract(var_239, var_237);
        var_240 = wp::mod(var_0, var_238);
        var_241 = wp::address(var_eq_solimp, var_240, var_3);
        var_243 = wp::load(var_241);
        var_242 = wp::copy(var_243);
        // timestep = opt_timestep[worldid % opt_timestep.shape[0]]                               <L 336>
        var_244 = &(var_opt_timestep.shape);
        var_247 = wp::load(var_244);
        var_246 = wp::extract(var_247, var_245);
        var_248 = wp::mod(var_0, var_246);
        var_249 = wp::address(var_opt_timestep, var_248);
        var_251 = wp::load(var_249);
        var_250 = wp::copy(var_251);
        // for i in range(3):                                                                     <L 338>
        // efcidi = efcid + i                                                                     <L 339>
        var_253 = wp::add(var_11, var_252);
        // _efc_row(                                                                              <L 341>
        // opt_disableflags,                                                                      <L 342>
        // worldid,                                                                               <L 343>
        // timestep,                                                                              <L 344>
        // efcidi,                                                                                <L 345>
        // pos[i],                                                                                <L 346>
        var_254 = wp::extract(var_86, var_252);
        // pos_imp,                                                                               <L 347>
        // invweight,                                                                             <L 348>
        // solref,                                                                                <L 349>
        // solimp,                                                                                <L 350>
        // 0.0,                                                                                   <L 351>
        // Jqvel[i],                                                                              <L 352>
        var_256 = wp::extract(var_90, var_252);
        // 0.0,                                                                                   <L 353>
        // ConstraintType.EQUALITY,                                                               <L 354>
        // eqid,                                                                                  <L 355>
        // efc_type_out,                                                                          <L 356>
        // efc_id_out,                                                                            <L 357>
        // efc_pos_out,                                                                           <L 358>
        // efc_margin_out,                                                                        <L 359>
        // efc_D_out,                                                                             <L 360>
        // efc_vel_out,                                                                           <L 361>
        // efc_aref_out,                                                                          <L 362>
        // efc_frictionloss_out,                                                                  <L 363>
        _efc_row_0(var_opt_disableflags, var_0, var_250, var_253, var_254, var_227, var_226, var_234, var_242, var_255, var_256, var_257, var_259, var_3, var_efc_type_out, var_efc_id_out, var_efc_pos_out, var_efc_margin_out, var_efc_D_out, var_efc_vel_out, var_efc_aref_out, var_efc_frictionloss_out);
        // efcidi = efcid + i                                                                     <L 339>
        var_261 = wp::add(var_11, var_260);
        // _efc_row(                                                                              <L 341>
        // opt_disableflags,                                                                      <L 342>
        // worldid,                                                                               <L 343>
        // timestep,                                                                              <L 344>
        // efcidi,                                                                                <L 345>
        // pos[i],                                                                                <L 346>
        var_262 = wp::extract(var_86, var_260);
        // pos_imp,                                                                               <L 347>
        // invweight,                                                                             <L 348>
        // solref,                                                                                <L 349>
        // solimp,                                                                                <L 350>
        // 0.0,                                                                                   <L 351>
        // Jqvel[i],                                                                              <L 352>
        var_264 = wp::extract(var_90, var_260);
        // 0.0,                                                                                   <L 353>
        // ConstraintType.EQUALITY,                                                               <L 354>
        // eqid,                                                                                  <L 355>
        // efc_type_out,                                                                          <L 356>
        // efc_id_out,                                                                            <L 357>
        // efc_pos_out,                                                                           <L 358>
        // efc_margin_out,                                                                        <L 359>
        // efc_D_out,                                                                             <L 360>
        // efc_vel_out,                                                                           <L 361>
        // efc_aref_out,                                                                          <L 362>
        // efc_frictionloss_out,                                                                  <L 363>
        _efc_row_0(var_opt_disableflags, var_0, var_250, var_261, var_262, var_227, var_226, var_234, var_242, var_263, var_264, var_265, var_267, var_3, var_efc_type_out, var_efc_id_out, var_efc_pos_out, var_efc_margin_out, var_efc_D_out, var_efc_vel_out, var_efc_aref_out, var_efc_frictionloss_out);
        // efcidi = efcid + i                                                                     <L 339>
        var_269 = wp::add(var_11, var_268);
        // _efc_row(                                                                              <L 341>
        // opt_disableflags,                                                                      <L 342>
        // worldid,                                                                               <L 343>
        // timestep,                                                                              <L 344>
        // efcidi,                                                                                <L 345>
        // pos[i],                                                                                <L 346>
        var_270 = wp::extract(var_86, var_268);
        // pos_imp,                                                                               <L 347>
        // invweight,                                                                             <L 348>
        // solref,                                                                                <L 349>
        // solimp,                                                                                <L 350>
        // 0.0,                                                                                   <L 351>
        // Jqvel[i],                                                                              <L 352>
        var_272 = wp::extract(var_90, var_268);
        // 0.0,                                                                                   <L 353>
        // ConstraintType.EQUALITY,                                                               <L 354>
        // eqid,                                                                                  <L 355>
        // efc_type_out,                                                                          <L 356>
        // efc_id_out,                                                                            <L 357>
        // efc_pos_out,                                                                           <L 358>
        // efc_margin_out,                                                                        <L 359>
        // efc_D_out,                                                                             <L 360>
        // efc_vel_out,                                                                           <L 361>
        // efc_aref_out,                                                                          <L 362>
        // efc_frictionloss_out,                                                                  <L 363>
        _efc_row_0(var_opt_disableflags, var_0, var_250, var_269, var_270, var_227, var_226, var_234, var_242, var_271, var_272, var_273, var_275, var_3, var_efc_type_out, var_efc_id_out, var_efc_pos_out, var_efc_margin_out, var_efc_D_out, var_efc_vel_out, var_efc_aref_out, var_efc_frictionloss_out);
    }
}



extern "C" __global__ void _limit_ball_4d819ddf_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::int32 var_opt_disableflags,
    wp::array_t<wp::int32> var_jnt_qposadr,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::vec_t<2, wp::float32>> var_jnt_solref,
    wp::array_t<wp::vec_t<5, wp::float32>> var_jnt_solimp,
    wp::array_t<wp::vec_t<2, wp::float32>> var_jnt_range,
    wp::array_t<wp::float32> var_jnt_margin,
    wp::array_t<wp::float32> var_dof_invweight0,
    bool var_is_sparse,
    wp::array_t<wp::int32> var_jnt_limited_ball_adr,
    wp::array_t<wp::float32> var_qpos_in,
    wp::array_t<wp::float32> var_qvel_in,
    wp::int32 var_njmax_in,
    wp::int32 var_njmax_nnz_in,
    wp::array_t<wp::int32> var_nl_out,
    wp::array_t<wp::int32> var_nefc_out,
    wp::array_t<wp::int32> var_efc_type_out,
    wp::array_t<wp::int32> var_efc_id_out,
    wp::array_t<wp::int32> var_efc_J_rownnz_out,
    wp::array_t<wp::int32> var_efc_J_rowadr_out,
    wp::array_t<wp::int32> var_efc_J_colind_out,
    wp::array_t<wp::float32> var_efc_J_out,
    wp::array_t<wp::float32> var_efc_pos_out,
    wp::array_t<wp::float32> var_efc_margin_out,
    wp::array_t<wp::float32> var_efc_D_out,
    wp::array_t<wp::float32> var_efc_vel_out,
    wp::array_t<wp::float32> var_efc_aref_out,
    wp::array_t<wp::float32> var_efc_frictionloss_out,
    wp::array_t<wp::int32> var_efc_nnz_out)
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
        wp::slice_t var_8;
        const wp::int32 var_9 = 0;
        wp::array_t<wp::float32> var_10;
        const wp::int32 var_11 = 0;
        wp::int32 var_12;
        wp::float32* var_13;
        const wp::int32 var_14 = 1;
        wp::int32 var_15;
        wp::float32* var_16;
        const wp::int32 var_17 = 2;
        wp::int32 var_18;
        wp::float32* var_19;
        const wp::int32 var_20 = 3;
        wp::int32 var_21;
        wp::float32* var_22;
        wp::quat_t<wp::float32> var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        wp::quat_t<wp::float32> var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::shape_t* var_30;
        const wp::int32 var_31 = 0;
        wp::int32 var_32;
        wp::shape_t var_33;
        wp::int32 var_34;
        wp::vec_t<2, wp::float32>* var_35;
        wp::vec_t<2, wp::float32> var_36;
        wp::vec_t<2, wp::float32> var_37;
        wp::vec_t<3, wp::float32> var_38;
        wp::float32 var_39;
        wp::shape_t* var_40;
        const wp::int32 var_41 = 0;
        wp::int32 var_42;
        wp::shape_t var_43;
        wp::int32 var_44;
        wp::float32* var_45;
        wp::float32 var_46;
        wp::float32 var_47;
        const wp::int32 var_48 = 0;
        wp::float32 var_49;
        const wp::int32 var_50 = 1;
        wp::float32 var_51;
        wp::float32 var_52;
        wp::float32 var_53;
        wp::float32 var_54;
        const wp::int32 var_55 = 0;
        bool var_56;
        const wp::int32 var_57 = 1;
        wp::int32 var_58;
        const wp::int32 var_59 = 1;
        wp::int32 var_60;
        bool var_61;
        wp::int32* var_62;
        wp::int32 var_63;
        wp::int32 var_64;
        const wp::int32 var_65 = 0;
        wp::int32 var_66;
        const wp::int32 var_67 = 1;
        wp::int32 var_68;
        const wp::int32 var_69 = 2;
        wp::int32 var_70;
        const wp::int32 var_71 = 3;
        const wp::int32 var_72 = 3;
        wp::int32 var_73;
        const wp::int32 var_74 = 3;
        wp::int32 var_75;
        bool var_76;
        const wp::int32 var_77 = 0;
        wp::int32 var_78;
        const wp::int32 var_79 = 1;
        wp::int32 var_80;
        const wp::int32 var_81 = 2;
        wp::int32 var_82;
        const wp::int32 var_83 = 0;
        const wp::int32 var_84 = 0;
        const wp::int32 var_85 = 0;
        const wp::int32 var_86 = 0;
        wp::float32 var_87;
        wp::float32 var_88;
        const wp::int32 var_89 = 0;
        const wp::int32 var_90 = 1;
        wp::float32 var_91;
        wp::float32 var_92;
        const wp::int32 var_93 = 0;
        const wp::int32 var_94 = 2;
        wp::float32 var_95;
        wp::float32 var_96;
        const wp::int32 var_97 = 0;
        wp::range_t var_98;
        wp::int32 var_99;
        const wp::float32 var_100 = 0.0;
        const wp::int32 var_101 = 0;
        wp::float32 var_102;
        wp::float32 var_103;
        const wp::int32 var_104 = 1;
        wp::float32 var_105;
        wp::float32 var_106;
        const wp::int32 var_107 = 2;
        wp::float32 var_108;
        wp::float32 var_109;
        const wp::int32 var_110 = 0;
        wp::float32 var_111;
        wp::float32 var_112;
        wp::float32* var_113;
        wp::float32 var_114;
        wp::float32 var_115;
        const wp::int32 var_116 = 1;
        wp::float32 var_117;
        wp::float32* var_118;
        wp::float32 var_119;
        wp::float32 var_120;
        wp::float32 var_121;
        const wp::int32 var_122 = 2;
        wp::float32 var_123;
        wp::float32* var_124;
        wp::float32 var_125;
        wp::float32 var_126;
        wp::float32 var_127;
        wp::shape_t* var_128;
        const wp::int32 var_129 = 0;
        wp::int32 var_130;
        wp::shape_t var_131;
        wp::int32 var_132;
        wp::shape_t* var_133;
        const wp::int32 var_134 = 0;
        wp::int32 var_135;
        wp::shape_t var_136;
        wp::int32 var_137;
        wp::shape_t* var_138;
        const wp::int32 var_139 = 0;
        wp::int32 var_140;
        wp::shape_t var_141;
        wp::int32 var_142;
        wp::shape_t* var_143;
        const wp::int32 var_144 = 0;
        wp::int32 var_145;
        wp::shape_t var_146;
        wp::int32 var_147;
        wp::float32* var_148;
        wp::float32* var_149;
        wp::vec_t<2, wp::float32>* var_150;
        wp::vec_t<5, wp::float32>* var_151;
        const wp::float32 var_152 = 0.0;
        const wp::int32 var_153 = 3;
        const wp::int32 var_154 = 3;
        wp::float32 var_155;
        wp::float32 var_156;
        wp::vec_t<2, wp::float32> var_157;
        wp::vec_t<5, wp::float32> var_158;
        //---------
        // forward
        // def _limit_ball(                                                                       <L 1422>
        // worldid, jntlimitedid = wp.tid()                                                       <L 1459>
        builtin_tid2d(var_0, var_1);
        // jntid = jnt_limited_ball_adr[jntlimitedid]                                             <L 1460>
        var_2 = wp::address(var_jnt_limited_ball_adr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // qposadr = jnt_qposadr[jntid]                                                           <L 1461>
        var_5 = wp::address(var_jnt_qposadr, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // qpos = qpos_in[worldid]                                                                <L 1463>
        var_8 = wp::slice_t(var_0, var_0, var_9);
        var_10 = wp::view(var_qpos_in, var_8);
        // jnt_quat = wp.quat(qpos[qposadr + 0], qpos[qposadr + 1], qpos[qposadr + 2], qpos[qposadr + 3])       <L 1464>
        var_12 = wp::add(var_6, var_11);
        var_13 = wp::address(var_10, var_12);
        var_15 = wp::add(var_6, var_14);
        var_16 = wp::address(var_10, var_15);
        var_18 = wp::add(var_6, var_17);
        var_19 = wp::address(var_10, var_18);
        var_21 = wp::add(var_6, var_20);
        var_22 = wp::address(var_10, var_21);
        var_24 = wp::load(var_13);
        var_25 = wp::load(var_16);
        var_26 = wp::load(var_19);
        var_27 = wp::load(var_22);
        var_23 = wp::quat_t<wp::float32>(var_24, var_25, var_26, var_27);
        // jnt_quat = wp.normalize(jnt_quat)                                                      <L 1465>
        var_28 = wp::normalize(var_23);
        // axis_angle = math.quat_to_vel(jnt_quat)                                                <L 1466>
        var_29 = quat_to_vel_0(var_28);
        // jnt_range_id = worldid % jnt_range.shape[0]                                            <L 1467>
        var_30 = &(var_jnt_range.shape);
        var_33 = wp::load(var_30);
        var_32 = wp::extract(var_33, var_31);
        var_34 = wp::mod(var_0, var_32);
        // jntrange = jnt_range[jnt_range_id, jntid]                                              <L 1468>
        var_35 = wp::address(var_jnt_range, var_34, var_3);
        var_37 = wp::load(var_35);
        var_36 = wp::copy(var_37);
        // axis, angle = math.normalize_with_norm(axis_angle)                                     <L 1469>
        normalize_with_norm_0(var_29, var_38, var_39);
        // jnt_margin_id = worldid % jnt_margin.shape[0]                                          <L 1470>
        var_40 = &(var_jnt_margin.shape);
        var_43 = wp::load(var_40);
        var_42 = wp::extract(var_43, var_41);
        var_44 = wp::mod(var_0, var_42);
        // jntmargin = jnt_margin[jnt_margin_id, jntid]                                           <L 1471>
        var_45 = wp::address(var_jnt_margin, var_44, var_3);
        var_47 = wp::load(var_45);
        var_46 = wp::copy(var_47);
        // pos = wp.max(jntrange[0], jntrange[1]) - angle - jntmargin                             <L 1473>
        var_49 = wp::extract(var_36, var_48);
        var_51 = wp::extract(var_36, var_50);
        var_52 = wp::max(var_49, var_51);
        var_53 = wp::sub(var_52, var_39);
        var_54 = wp::sub(var_53, var_46);
        // active = pos < 0                                                                       <L 1474>
        var_56 = (var_54 < var_55);
        // if active:                                                                             <L 1476>
        if (var_56) {
            // wp.atomic_add(nl_out, worldid, 1)                                                  <L 1477>
            var_58 = wp::atomic_add(var_nl_out, var_0, var_57);
            // efcid = wp.atomic_add(nefc_out, worldid, 1)                                        <L 1478>
            var_60 = wp::atomic_add(var_nefc_out, var_0, var_59);
            // if efcid >= njmax_in:                                                              <L 1480>
            var_61 = (var_60 >= var_njmax_in);
            if (var_61) {
                // return                                                                         <L 1481>
                continue;
            }
            // dofadr = jnt_dofadr[jntid]                                                         <L 1483>
            var_62 = wp::address(var_jnt_dofadr, var_3);
            var_64 = wp::load(var_62);
            var_63 = wp::copy(var_64);
            // dof0 = dofadr + 0                                                                  <L 1484>
            var_66 = wp::add(var_63, var_65);
            // dof1 = dofadr + 1                                                                  <L 1485>
            var_68 = wp::add(var_63, var_67);
            // dof2 = dofadr + 2                                                                  <L 1486>
            var_70 = wp::add(var_63, var_69);
            // if is_sparse:                                                                      <L 1488>
            if (var_is_sparse) {
                // efc_J_rownnz_out[worldid, efcid] = 3                                           <L 1489>
                wp::array_store(var_efc_J_rownnz_out, var_0, var_60, var_71);
                // rowadr = wp.atomic_add(efc_nnz_out, worldid, 3)                                <L 1490>
                var_73 = wp::atomic_add(var_efc_nnz_out, var_0, var_72);
                // if rowadr + 3 > njmax_nnz_in:                                                  <L 1491>
                var_75 = wp::add(var_73, var_74);
                var_76 = (var_75 > var_njmax_nnz_in);
                if (var_76) {
                    // return                                                                     <L 1492>
                    continue;
                }
                // efc_J_rowadr_out[worldid, efcid] = rowadr                                      <L 1493>
                wp::array_store(var_efc_J_rowadr_out, var_0, var_60, var_73);
                // sparseid0 = rowadr + 0                                                         <L 1495>
                var_78 = wp::add(var_73, var_77);
                // sparseid1 = rowadr + 1                                                         <L 1496>
                var_80 = wp::add(var_73, var_79);
                // sparseid2 = rowadr + 2                                                         <L 1497>
                var_82 = wp::add(var_73, var_81);
                // efc_J_colind_out[worldid, 0, sparseid0] = dof0                                 <L 1499>
                wp::array_store(var_efc_J_colind_out, var_0, var_83, var_78, var_66);
                // efc_J_colind_out[worldid, 0, sparseid1] = dof1                                 <L 1500>
                wp::array_store(var_efc_J_colind_out, var_0, var_84, var_80, var_68);
                // efc_J_colind_out[worldid, 0, sparseid2] = dof2                                 <L 1501>
                wp::array_store(var_efc_J_colind_out, var_0, var_85, var_82, var_70);
                // efc_J_out[worldid, 0, sparseid0] = -axis[0]                                    <L 1503>
                var_87 = wp::extract(var_38, var_86);
                var_88 = wp::neg(var_87);
                wp::array_store(var_efc_J_out, var_0, var_89, var_78, var_88);
                // efc_J_out[worldid, 0, sparseid1] = -axis[1]                                    <L 1504>
                var_91 = wp::extract(var_38, var_90);
                var_92 = wp::neg(var_91);
                wp::array_store(var_efc_J_out, var_0, var_93, var_80, var_92);
                // efc_J_out[worldid, 0, sparseid2] = -axis[2]                                    <L 1505>
                var_95 = wp::extract(var_38, var_94);
                var_96 = wp::neg(var_95);
                wp::array_store(var_efc_J_out, var_0, var_97, var_82, var_96);
            }
            if (!var_is_sparse) {
                // for i in range(nv):                                                            <L 1507>
                var_98 = wp::range(var_nv);
                start_for_2:;
                    if (iter_cmp(var_98) == 0) goto end_for_2;
                    var_99 = wp::iter_next(var_98);
                    // efc_J_out[worldid, efcid, i] = 0.0                                         <L 1508>
                    wp::array_store(var_efc_J_out, var_0, var_60, var_99, var_100);
                    goto start_for_2;
                end_for_2:;
                // efc_J_out[worldid, efcid, dof0] = -axis[0]                                     <L 1509>
                var_102 = wp::extract(var_38, var_101);
                var_103 = wp::neg(var_102);
                wp::array_store(var_efc_J_out, var_0, var_60, var_66, var_103);
                // efc_J_out[worldid, efcid, dof1] = -axis[1]                                     <L 1510>
                var_105 = wp::extract(var_38, var_104);
                var_106 = wp::neg(var_105);
                wp::array_store(var_efc_J_out, var_0, var_60, var_68, var_106);
                // efc_J_out[worldid, efcid, dof2] = -axis[2]                                     <L 1511>
                var_108 = wp::extract(var_38, var_107);
                var_109 = wp::neg(var_108);
                wp::array_store(var_efc_J_out, var_0, var_60, var_70, var_109);
            }
            // Jqvel = -axis[0] * qvel_in[worldid, dof0]                                          <L 1513>
            var_111 = wp::extract(var_38, var_110);
            var_112 = wp::neg(var_111);
            var_113 = wp::address(var_qvel_in, var_0, var_66);
            var_115 = wp::load(var_113);
            var_114 = wp::mul(var_112, var_115);
            // Jqvel -= axis[1] * qvel_in[worldid, dof1]                                          <L 1514>
            var_117 = wp::extract(var_38, var_116);
            var_118 = wp::address(var_qvel_in, var_0, var_68);
            var_120 = wp::load(var_118);
            var_119 = wp::mul(var_117, var_120);
            var_121 = wp::sub(var_114, var_119);
            // Jqvel -= axis[2] * qvel_in[worldid, dof2]                                          <L 1515>
            var_123 = wp::extract(var_38, var_122);
            var_124 = wp::address(var_qvel_in, var_0, var_70);
            var_126 = wp::load(var_124);
            var_125 = wp::mul(var_123, var_126);
            var_127 = wp::sub(var_121, var_125);
            // dof_invweight0_id = worldid % dof_invweight0.shape[0]                              <L 1517>
            var_128 = &(var_dof_invweight0.shape);
            var_131 = wp::load(var_128);
            var_130 = wp::extract(var_131, var_129);
            var_132 = wp::mod(var_0, var_130);
            // jnt_solref_id = worldid % jnt_solref.shape[0]                                      <L 1518>
            var_133 = &(var_jnt_solref.shape);
            var_136 = wp::load(var_133);
            var_135 = wp::extract(var_136, var_134);
            var_137 = wp::mod(var_0, var_135);
            // jnt_solimp_id = worldid % jnt_solimp.shape[0]                                      <L 1519>
            var_138 = &(var_jnt_solimp.shape);
            var_141 = wp::load(var_138);
            var_140 = wp::extract(var_141, var_139);
            var_142 = wp::mod(var_0, var_140);
            // _efc_row(                                                                          <L 1520>
            // opt_disableflags,                                                                  <L 1521>
            // worldid,                                                                           <L 1522>
            // opt_timestep[worldid % opt_timestep.shape[0]],                                     <L 1523>
            var_143 = &(var_opt_timestep.shape);
            var_146 = wp::load(var_143);
            var_145 = wp::extract(var_146, var_144);
            var_147 = wp::mod(var_0, var_145);
            var_148 = wp::address(var_opt_timestep, var_147);
            // efcid,                                                                             <L 1524>
            // pos,                                                                               <L 1525>
            // pos,                                                                               <L 1526>
            // dof_invweight0[dof_invweight0_id, dofadr],                                         <L 1527>
            var_149 = wp::address(var_dof_invweight0, var_132, var_63);
            // jnt_solref[jnt_solref_id, jntid],                                                  <L 1528>
            var_150 = wp::address(var_jnt_solref, var_137, var_3);
            // jnt_solimp[jnt_solimp_id, jntid],                                                  <L 1529>
            var_151 = wp::address(var_jnt_solimp, var_142, var_3);
            // jntmargin,                                                                         <L 1530>
            // Jqvel,                                                                             <L 1531>
            // 0.0,                                                                               <L 1532>
            // ConstraintType.LIMIT_JOINT,                                                        <L 1533>
            // jntid,                                                                             <L 1534>
            // efc_type_out,                                                                      <L 1535>
            // efc_id_out,                                                                        <L 1536>
            // efc_pos_out,                                                                       <L 1537>
            // efc_margin_out,                                                                    <L 1538>
            // efc_D_out,                                                                         <L 1539>
            // efc_vel_out,                                                                       <L 1540>
            // efc_aref_out,                                                                      <L 1541>
            // efc_frictionloss_out,                                                              <L 1542>
            var_155 = wp::load(var_148);
            var_156 = wp::load(var_149);
            var_157 = wp::load(var_150);
            var_158 = wp::load(var_151);
            _efc_row_0(var_opt_disableflags, var_0, var_155, var_60, var_54, var_54, var_156, var_157, var_158, var_46, var_127, var_152, var_154, var_3, var_efc_type_out, var_efc_id_out, var_efc_pos_out, var_efc_margin_out, var_efc_D_out, var_efc_vel_out, var_efc_aref_out, var_efc_frictionloss_out);
        }
    }
}



extern "C" __global__ void _zero_constraint_counts_a05c4a78_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_ne_out,
    wp::array_t<wp::int32> var_nf_out,
    wp::array_t<wp::int32> var_nl_out,
    wp::array_t<wp::int32> var_nefc_out,
    wp::array_t<wp::int32> var_efc_nnz_out)
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
        const wp::int32 var_3 = 0;
        const wp::int32 var_4 = 0;
        const wp::int32 var_5 = 0;
        //---------
        // forward
        // def _zero_constraint_counts(                                                           <L 33>
        // worldid = wp.tid()                                                                     <L 42>
        var_0 = builtin_tid1d();
        // ne_out[worldid] = 0                                                                    <L 45>
        wp::array_store(var_ne_out, var_0, var_1);
        // nf_out[worldid] = 0                                                                    <L 46>
        wp::array_store(var_nf_out, var_0, var_2);
        // nl_out[worldid] = 0                                                                    <L 47>
        wp::array_store(var_nl_out, var_0, var_3);
        // nefc_out[worldid] = 0                                                                  <L 48>
        wp::array_store(var_nefc_out, var_0, var_4);
        // efc_nnz_out[worldid] = 0                                                               <L 49>
        wp::array_store(var_efc_nnz_out, var_0, var_5);
    }
}



extern "C" __global__ void _equality_tendon_b5d72584_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::int32 var_opt_disableflags,
    wp::array_t<wp::int32> var_eq_obj1id,
    wp::array_t<wp::int32> var_eq_obj2id,
    wp::array_t<wp::vec_t<2, wp::float32>> var_eq_solref,
    wp::array_t<wp::vec_t<5, wp::float32>> var_eq_solimp,
    wp::array_t<wp::vec_t<11, wp::float32>> var_eq_data,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::float32> var_tendon_length0,
    wp::array_t<wp::float32> var_tendon_invweight0,
    bool var_is_sparse,
    wp::array_t<wp::int32> var_eq_ten_adr,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<bool> var_eq_active_in,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::array_t<wp::float32> var_ten_length_in,
    wp::int32 var_njmax_in,
    wp::int32 var_njmax_nnz_in,
    wp::array_t<wp::int32> var_ne_out,
    wp::array_t<wp::int32> var_nefc_out,
    wp::array_t<wp::int32> var_efc_type_out,
    wp::array_t<wp::int32> var_efc_id_out,
    wp::array_t<wp::int32> var_efc_J_rownnz_out,
    wp::array_t<wp::int32> var_efc_J_rowadr_out,
    wp::array_t<wp::int32> var_efc_J_colind_out,
    wp::array_t<wp::float32> var_efc_J_out,
    wp::array_t<wp::float32> var_efc_pos_out,
    wp::array_t<wp::float32> var_efc_margin_out,
    wp::array_t<wp::float32> var_efc_D_out,
    wp::array_t<wp::float32> var_efc_vel_out,
    wp::array_t<wp::float32> var_efc_aref_out,
    wp::array_t<wp::float32> var_efc_frictionloss_out,
    wp::array_t<wp::int32> var_efc_nnz_out)
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
        bool* var_5;
        bool var_6;
        bool var_7;
        const wp::int32 var_8 = 1;
        wp::int32 var_9;
        const wp::int32 var_10 = 1;
        wp::int32 var_11;
        bool var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::shape_t* var_19;
        const wp::int32 var_20 = 0;
        wp::int32 var_21;
        wp::shape_t var_22;
        wp::int32 var_23;
        wp::vec_t<11, wp::float32>* var_24;
        wp::vec_t<11, wp::float32> var_25;
        wp::vec_t<11, wp::float32> var_26;
        wp::shape_t* var_27;
        const wp::int32 var_28 = 0;
        wp::int32 var_29;
        wp::shape_t var_30;
        wp::int32 var_31;
        wp::vec_t<2, wp::float32>* var_32;
        wp::vec_t<2, wp::float32> var_33;
        wp::vec_t<2, wp::float32> var_34;
        wp::shape_t* var_35;
        const wp::int32 var_36 = 0;
        wp::int32 var_37;
        wp::shape_t var_38;
        wp::int32 var_39;
        wp::vec_t<5, wp::float32>* var_40;
        wp::vec_t<5, wp::float32> var_41;
        wp::vec_t<5, wp::float32> var_42;
        wp::shape_t* var_43;
        const wp::int32 var_44 = 0;
        wp::int32 var_45;
        wp::shape_t var_46;
        wp::int32 var_47;
        wp::shape_t* var_48;
        const wp::int32 var_49 = 0;
        wp::int32 var_50;
        wp::shape_t var_51;
        wp::int32 var_52;
        wp::float32* var_53;
        wp::float32* var_54;
        wp::float32 var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        const wp::int32 var_58 = 1;
        const wp::int32 var_59 = -1;
        bool var_60;
        wp::float32* var_61;
        wp::float32* var_62;
        wp::float32 var_63;
        wp::float32 var_64;
        wp::float32 var_65;
        wp::float32* var_66;
        wp::float32* var_67;
        wp::float32 var_68;
        wp::float32 var_69;
        wp::float32 var_70;
        wp::float32 var_71;
        wp::float32 var_72;
        wp::float32 var_73;
        wp::float32 var_74;
        const wp::int32 var_75 = 0;
        wp::float32 var_76;
        const wp::int32 var_77 = 1;
        wp::float32 var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        const wp::int32 var_81 = 2;
        wp::float32 var_82;
        wp::float32 var_83;
        wp::float32 var_84;
        const wp::int32 var_85 = 3;
        wp::float32 var_86;
        wp::float32 var_87;
        wp::float32 var_88;
        const wp::int32 var_89 = 4;
        wp::float32 var_90;
        wp::float32 var_91;
        wp::float32 var_92;
        wp::float32 var_93;
        const wp::int32 var_94 = 1;
        wp::float32 var_95;
        const wp::float32 var_96 = 2.0;
        const wp::int32 var_97 = 2;
        wp::float32 var_98;
        wp::float32 var_99;
        wp::float32 var_100;
        wp::float32 var_101;
        const wp::float32 var_102 = 3.0;
        const wp::int32 var_103 = 3;
        wp::float32 var_104;
        wp::float32 var_105;
        wp::float32 var_106;
        wp::float32 var_107;
        const wp::float32 var_108 = 4.0;
        const wp::int32 var_109 = 4;
        wp::float32 var_110;
        wp::float32 var_111;
        wp::float32 var_112;
        wp::float32 var_113;
        wp::float32* var_114;
        wp::float32 var_115;
        wp::float32 var_116;
        const wp::int32 var_117 = 0;
        wp::float32 var_118;
        wp::float32 var_119;
        const wp::float32 var_120 = 0.0;
        wp::float32 var_121;
        wp::float32 var_122;
        wp::float32 var_123;
        wp::int32* var_124;
        wp::int32 var_125;
        wp::int32 var_126;
        wp::int32* var_127;
        wp::int32 var_128;
        wp::int32 var_129;
        const wp::int32 var_130 = 0;
        const wp::int32 var_131 = 0;
        const wp::float32 var_132 = 0.0;
        bool var_133;
        wp::int32* var_134;
        wp::int32 var_135;
        wp::int32 var_136;
        wp::int32* var_137;
        wp::int32 var_138;
        wp::int32 var_139;
        wp::int32 var_140;
        wp::int32 var_141;
        const wp::int32 var_142 = 0;
        wp::int32 var_143;
        const wp::int32 var_144 = 0;
        wp::int32 var_145;
        const wp::int32 var_146 = 0;
        wp::int32 var_147;
        bool var_148;
        bool var_149;
        bool var_150;
        wp::int32 var_151;
        wp::int32 var_152;
        bool var_153;
        wp::int32 var_154;
        wp::int32* var_155;
        wp::int32 var_156;
        wp::int32 var_157;
        wp::int32 var_158;
        bool var_159;
        wp::int32 var_160;
        wp::int32* var_161;
        wp::int32 var_162;
        wp::int32 var_163;
        wp::int32 var_164;
        bool var_165;
        const wp::int32 var_166 = 1;
        wp::int32 var_167;
        wp::int32 var_168;
        bool var_169;
        const wp::int32 var_170 = 1;
        wp::int32 var_171;
        wp::int32 var_172;
        const wp::int32 var_173 = 1;
        wp::int32 var_174;
        wp::int32 var_175;
        wp::int32 var_176;
        bool var_177;
        const wp::int32 var_178 = 0;
        wp::int32 var_179;
        const wp::int32 var_180 = 0;
        wp::int32 var_181;
        const wp::float32 var_182 = 0.0;
        wp::float32 var_183;
        const wp::int32 var_184 = 0;
        wp::int32 var_185;
        wp::range_t var_186;
        wp::int32 var_187;
        const wp::float32 var_188 = 0.0;
        wp::float32 var_189;
        bool var_190;
        wp::int32 var_191;
        wp::int32* var_192;
        bool var_193;
        wp::int32 var_194;
        wp::float32* var_195;
        wp::float32 var_196;
        wp::float32 var_197;
        const wp::int32 var_198 = 1;
        wp::int32 var_199;
        wp::int32 var_200;
        wp::float32 var_201;
        wp::int32 var_202;
        wp::float32 var_203;
        wp::float32 var_204;
        const wp::float32 var_205 = 0.0;
        bool var_206;
        const wp::float32 var_207 = 0.0;
        wp::float32 var_208;
        bool var_209;
        wp::int32 var_210;
        wp::int32* var_211;
        bool var_212;
        wp::int32 var_213;
        wp::float32* var_214;
        wp::float32 var_215;
        wp::float32 var_216;
        const wp::int32 var_217 = 1;
        wp::int32 var_218;
        wp::int32 var_219;
        wp::float32 var_220;
        wp::int32 var_221;
        wp::float32 var_222;
        wp::float32 var_223;
        wp::float32 var_224;
        wp::float32 var_225;
        wp::int32 var_226;
        wp::float32 var_227;
        const wp::float32 var_228 = 0.0;
        bool var_229;
        wp::int32 var_230;
        const wp::int32 var_231 = 0;
        const wp::int32 var_232 = 0;
        const wp::int32 var_233 = 1;
        wp::int32 var_234;
        wp::int32 var_235;
        wp::int32 var_236;
        wp::float32* var_237;
        wp::float32 var_238;
        wp::float32 var_239;
        wp::float32 var_240;
        wp::shape_t* var_241;
        const wp::int32 var_242 = 0;
        wp::int32 var_243;
        wp::shape_t var_244;
        wp::int32 var_245;
        wp::float32* var_246;
        const wp::float32 var_247 = 0.0;
        const wp::float32 var_248 = 0.0;
        const wp::int32 var_249 = 0;
        const wp::int32 var_250 = 0;
        wp::float32 var_251;
        //---------
        // forward
        // def _equality_tendon(                                                                  <L 499>
        // worldid, eqtenid = wp.tid()                                                            <L 541>
        builtin_tid2d(var_0, var_1);
        // eqid = eq_ten_adr[eqtenid]                                                             <L 542>
        var_2 = wp::address(var_eq_ten_adr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if not eq_active_in[worldid, eqid]:                                                    <L 544>
        var_5 = wp::address(var_eq_active_in, var_0, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::unot(var_7);
        if (var_6) {
            // return                                                                             <L 545>
            continue;
        }
        // wp.atomic_add(ne_out, worldid, 1)                                                      <L 547>
        var_9 = wp::atomic_add(var_ne_out, var_0, var_8);
        // efcid = wp.atomic_add(nefc_out, worldid, 1)                                            <L 548>
        var_11 = wp::atomic_add(var_nefc_out, var_0, var_10);
        // if efcid >= njmax_in:                                                                  <L 550>
        var_12 = (var_11 >= var_njmax_in);
        if (var_12) {
            // return                                                                             <L 551>
            continue;
        }
        // obj1id = eq_obj1id[eqid]                                                               <L 553>
        var_13 = wp::address(var_eq_obj1id, var_3);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // obj2id = eq_obj2id[eqid]                                                               <L 554>
        var_16 = wp::address(var_eq_obj2id, var_3);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // data = eq_data[worldid % eq_data.shape[0], eqid]                                       <L 556>
        var_19 = &(var_eq_data.shape);
        var_22 = wp::load(var_19);
        var_21 = wp::extract(var_22, var_20);
        var_23 = wp::mod(var_0, var_21);
        var_24 = wp::address(var_eq_data, var_23, var_3);
        var_26 = wp::load(var_24);
        var_25 = wp::copy(var_26);
        // solref = eq_solref[worldid % eq_solref.shape[0], eqid]                                 <L 557>
        var_27 = &(var_eq_solref.shape);
        var_30 = wp::load(var_27);
        var_29 = wp::extract(var_30, var_28);
        var_31 = wp::mod(var_0, var_29);
        var_32 = wp::address(var_eq_solref, var_31, var_3);
        var_34 = wp::load(var_32);
        var_33 = wp::copy(var_34);
        // solimp = eq_solimp[worldid % eq_solimp.shape[0], eqid]                                 <L 558>
        var_35 = &(var_eq_solimp.shape);
        var_38 = wp::load(var_35);
        var_37 = wp::extract(var_38, var_36);
        var_39 = wp::mod(var_0, var_37);
        var_40 = wp::address(var_eq_solimp, var_39, var_3);
        var_42 = wp::load(var_40);
        var_41 = wp::copy(var_42);
        // tendon_length0_id = worldid % tendon_length0.shape[0]                                  <L 559>
        var_43 = &(var_tendon_length0.shape);
        var_46 = wp::load(var_43);
        var_45 = wp::extract(var_46, var_44);
        var_47 = wp::mod(var_0, var_45);
        // tendon_invweight0_id = worldid % tendon_invweight0.shape[0]                            <L 560>
        var_48 = &(var_tendon_invweight0.shape);
        var_51 = wp::load(var_48);
        var_50 = wp::extract(var_51, var_49);
        var_52 = wp::mod(var_0, var_50);
        // pos1 = ten_length_in[worldid, obj1id] - tendon_length0[tendon_length0_id, obj1id]       <L 561>
        var_53 = wp::address(var_ten_length_in, var_0, var_14);
        var_54 = wp::address(var_tendon_length0, var_47, var_14);
        var_56 = wp::load(var_53);
        var_57 = wp::load(var_54);
        var_55 = wp::sub(var_56, var_57);
        // if obj2id > -1:                                                                        <L 563>
        var_60 = (var_17 > var_59);
        if (var_60) {
            // invweight = tendon_invweight0[tendon_invweight0_id, obj1id] + tendon_invweight0[tendon_invweight0_id, obj2id]       <L 564>
            var_61 = wp::address(var_tendon_invweight0, var_52, var_14);
            var_62 = wp::address(var_tendon_invweight0, var_52, var_17);
            var_64 = wp::load(var_61);
            var_65 = wp::load(var_62);
            var_63 = wp::add(var_64, var_65);
            // pos2 = ten_length_in[worldid, obj2id] - tendon_length0[tendon_length0_id, obj2id]       <L 566>
            var_66 = wp::address(var_ten_length_in, var_0, var_17);
            var_67 = wp::address(var_tendon_length0, var_47, var_17);
            var_69 = wp::load(var_66);
            var_70 = wp::load(var_67);
            var_68 = wp::sub(var_69, var_70);
            // dif = pos2                                                                         <L 568>
            var_71 = wp::copy(var_68);
            // dif2 = dif * dif                                                                   <L 569>
            var_72 = wp::mul(var_71, var_71);
            // dif3 = dif2 * dif                                                                  <L 570>
            var_73 = wp::mul(var_72, var_71);
            // dif4 = dif3 * dif                                                                  <L 571>
            var_74 = wp::mul(var_73, var_71);
            // pos = pos1 - (data[0] + data[1] * dif + data[2] * dif2 + data[3] * dif3 + data[4] * dif4)       <L 573>
            var_76 = wp::extract(var_25, var_75);
            var_78 = wp::extract(var_25, var_77);
            var_79 = wp::mul(var_78, var_71);
            var_80 = wp::add(var_76, var_79);
            var_82 = wp::extract(var_25, var_81);
            var_83 = wp::mul(var_82, var_72);
            var_84 = wp::add(var_80, var_83);
            var_86 = wp::extract(var_25, var_85);
            var_87 = wp::mul(var_86, var_73);
            var_88 = wp::add(var_84, var_87);
            var_90 = wp::extract(var_25, var_89);
            var_91 = wp::mul(var_90, var_74);
            var_92 = wp::add(var_88, var_91);
            var_93 = wp::sub(var_55, var_92);
            // deriv = data[1] + 2.0 * data[2] * dif + 3.0 * data[3] * dif2 + 4.0 * data[4] * dif3       <L 574>
            var_95 = wp::extract(var_25, var_94);
            var_98 = wp::extract(var_25, var_97);
            var_99 = wp::mul(var_96, var_98);
            var_100 = wp::mul(var_99, var_71);
            var_101 = wp::add(var_95, var_100);
            var_104 = wp::extract(var_25, var_103);
            var_105 = wp::mul(var_102, var_104);
            var_106 = wp::mul(var_105, var_72);
            var_107 = wp::add(var_101, var_106);
            var_110 = wp::extract(var_25, var_109);
            var_111 = wp::mul(var_108, var_110);
            var_112 = wp::mul(var_111, var_73);
            var_113 = wp::add(var_107, var_112);
        }
        if (!var_60) {
            // invweight = tendon_invweight0[tendon_invweight0_id, obj1id]                        <L 576>
            var_114 = wp::address(var_tendon_invweight0, var_52, var_14);
            var_116 = wp::load(var_114);
            var_115 = wp::copy(var_116);
            // pos = pos1 - data[0]                                                               <L 577>
            var_118 = wp::extract(var_25, var_117);
            var_119 = wp::sub(var_55, var_118);
            // deriv = 0.0                                                                        <L 578>
        }
        var_121 = wp::where(var_60, var_63, var_115);
        var_122 = wp::where(var_60, var_93, var_119);
        var_123 = wp::where(var_60, var_113, var_120);
        // rownnz1 = ten_J_rownnz[obj1id]                                                         <L 580>
        var_124 = wp::address(var_ten_J_rownnz, var_14);
        var_126 = wp::load(var_124);
        var_125 = wp::copy(var_126);
        // rowadr1 = ten_J_rowadr[obj1id]                                                         <L 581>
        var_127 = wp::address(var_ten_J_rowadr, var_14);
        var_129 = wp::load(var_127);
        var_128 = wp::copy(var_129);
        // rownnz2 = 0                                                                            <L 582>
        // rowadr2 = 0                                                                            <L 583>
        // if deriv != 0.0:                                                                       <L 585>
        var_133 = (var_123 != var_132);
        if (var_133) {
            // rownnz2 = ten_J_rownnz[obj2id]                                                     <L 586>
            var_134 = wp::address(var_ten_J_rownnz, var_17);
            var_136 = wp::load(var_134);
            var_135 = wp::copy(var_136);
            // rowadr2 = ten_J_rowadr[obj2id]                                                     <L 587>
            var_137 = wp::address(var_ten_J_rowadr, var_17);
            var_139 = wp::load(var_137);
            var_138 = wp::copy(var_139);
        }
        var_140 = wp::where(var_133, var_135, var_130);
        var_141 = wp::where(var_133, var_138, var_131);
        // if is_sparse:                                                                          <L 589>
        if (var_is_sparse) {
            // p1, p2 = int(0), int(0)                                                            <L 592>
            var_143 = wp::int(var_142);
            var_145 = wp::int(var_144);
            // rownnz = int(0)                                                                    <L 593>
            var_147 = wp::int(var_146);
            // while p1 < rownnz1 or p2 < rownnz2:                                                <L 594>
        start_while_2:;
            var_148 = (var_143 < var_125);
            var_149 = (var_145 < var_140);
            var_150 = var_148 || var_149;
        if ((var_150) == false) goto end_while_2;
                // col1 = nv                                                                      <L 595>
                var_151 = wp::copy(var_nv);
                // col2 = nv                                                                      <L 596>
                var_152 = wp::copy(var_nv);
                // if p1 < rownnz1:                                                               <L 597>
                var_153 = (var_143 < var_125);
                if (var_153) {
                    // col1 = ten_J_colind[rowadr1 + p1]                                          <L 598>
                    var_154 = wp::add(var_128, var_143);
                    var_155 = wp::address(var_ten_J_colind, var_154);
                    var_157 = wp::load(var_155);
                    var_156 = wp::copy(var_157);
                }
                var_158 = wp::where(var_153, var_156, var_151);
                // if p2 < rownnz2:                                                               <L 599>
                var_159 = (var_145 < var_140);
                if (var_159) {
                    // col2 = ten_J_colind[rowadr2 + p2]                                          <L 600>
                    var_160 = wp::add(var_141, var_145);
                    var_161 = wp::address(var_ten_J_colind, var_160);
                    var_163 = wp::load(var_161);
                    var_162 = wp::copy(var_163);
                }
                var_164 = wp::where(var_159, var_162, var_152);
                // if col1 <= col2:                                                               <L 601>
                var_165 = (var_158 <= var_164);
                if (var_165) {
                    // p1 += 1                                                                    <L 602>
                    var_167 = wp::add(var_143, var_166);
                }
                var_168 = wp::where(var_165, var_167, var_143);
                // if col2 <= col1:                                                               <L 603>
                var_169 = (var_164 <= var_158);
                if (var_169) {
                    // p2 += 1                                                                    <L 604>
                    var_171 = wp::add(var_145, var_170);
                }
                var_172 = wp::where(var_169, var_171, var_145);
                // rownnz += 1                                                                    <L 605>
                var_174 = wp::add(var_147, var_173);
                wp::assign(var_143, var_168);
                wp::assign(var_145, var_172);
                wp::assign(var_147, var_174);
        goto start_while_2;
        end_while_2:;
            // rowadr = wp.atomic_add(efc_nnz_out, worldid, rownnz)                               <L 607>
            var_175 = wp::atomic_add(var_efc_nnz_out, var_0, var_147);
            // if rowadr + rownnz > njmax_nnz_in:                                                 <L 608>
            var_176 = wp::add(var_175, var_147);
            var_177 = (var_176 > var_njmax_nnz_in);
            if (var_177) {
                // return                                                                         <L 609>
                continue;
            }
            // efc_J_rowadr_out[worldid, efcid] = rowadr                                          <L 610>
            wp::array_store(var_efc_J_rowadr_out, var_0, var_11, var_175);
        }
        // ptr1 = int(0)                                                                          <L 612>
        var_179 = wp::int(var_178);
        // ptr2 = int(0)                                                                          <L 613>
        var_181 = wp::int(var_180);
        // Jqvel = float(0.0)                                                                     <L 615>
        var_183 = wp::float(var_182);
        // nnz = int(0)                                                                           <L 617>
        var_185 = wp::int(var_184);
        // for i in range(nv):                                                                    <L 618>
        var_186 = wp::range(var_nv);
        start_for_5:;
            if (iter_cmp(var_186) == 0) goto end_for_5;
            var_187 = wp::iter_next(var_186);
            // J1 = float(0.0)                                                                    <L 619>
            var_189 = wp::float(var_188);
            // if ptr1 < rownnz1:                                                                 <L 620>
            var_190 = (var_179 < var_125);
            if (var_190) {
                // sparseid1 = rowadr1 + ptr1                                                     <L 621>
                var_191 = wp::add(var_128, var_179);
                // if ten_J_colind[sparseid1] == i:                                               <L 622>
                var_192 = wp::address(var_ten_J_colind, var_191);
                var_194 = wp::load(var_192);
                var_193 = (var_194 == var_187);
                if (var_193) {
                    // J1 = ten_J_in[worldid, sparseid1]                                          <L 623>
                    var_195 = wp::address(var_ten_J_in, var_0, var_191);
                    var_197 = wp::load(var_195);
                    var_196 = wp::copy(var_197);
                    // ptr1 += 1                                                                  <L 624>
                    var_199 = wp::add(var_179, var_198);
                }
                var_200 = wp::where(var_193, var_199, var_179);
                var_201 = wp::where(var_193, var_196, var_189);
            }
            var_202 = wp::where(var_190, var_200, var_179);
            var_203 = wp::where(var_190, var_201, var_189);
            // J = J1                                                                             <L 626>
            var_204 = wp::copy(var_203);
            // if deriv != 0.0:                                                                   <L 627>
            var_206 = (var_123 != var_205);
            if (var_206) {
                // J2 = float(0.0)                                                                <L 628>
                var_208 = wp::float(var_207);
                // if ptr2 < rownnz2:                                                             <L 629>
                var_209 = (var_181 < var_140);
                if (var_209) {
                    // sparseid2 = rowadr2 + ptr2                                                 <L 630>
                    var_210 = wp::add(var_141, var_181);
                    // if ten_J_colind[sparseid2] == i:                                           <L 631>
                    var_211 = wp::address(var_ten_J_colind, var_210);
                    var_213 = wp::load(var_211);
                    var_212 = (var_213 == var_187);
                    if (var_212) {
                        // J2 = ten_J_in[worldid, sparseid2]                                      <L 632>
                        var_214 = wp::address(var_ten_J_in, var_0, var_210);
                        var_216 = wp::load(var_214);
                        var_215 = wp::copy(var_216);
                        // ptr2 += 1                                                              <L 633>
                        var_218 = wp::add(var_181, var_217);
                    }
                    var_219 = wp::where(var_212, var_218, var_181);
                    var_220 = wp::where(var_212, var_215, var_208);
                }
                var_221 = wp::where(var_209, var_219, var_181);
                var_222 = wp::where(var_209, var_220, var_208);
                // J += J2 * -deriv                                                               <L 634>
                var_223 = wp::neg(var_123);
                var_224 = wp::mul(var_222, var_223);
                var_225 = wp::add(var_204, var_224);
            }
            var_226 = wp::where(var_206, var_221, var_181);
            var_227 = wp::where(var_206, var_225, var_204);
            // if is_sparse:                                                                      <L 636>
            if (var_is_sparse) {
                // if J != 0.0:                                                                   <L 637>
                var_229 = (var_227 != var_228);
                if (var_229) {
                    // sparseid = rowadr + nnz                                                    <L 638>
                    var_230 = wp::add(var_175, var_185);
                    // efc_J_colind_out[worldid, 0, sparseid] = i                                 <L 639>
                    wp::array_store(var_efc_J_colind_out, var_0, var_231, var_230, var_187);
                    // efc_J_out[worldid, 0, sparseid] = J                                        <L 640>
                    wp::array_store(var_efc_J_out, var_0, var_232, var_230, var_227);
                    // nnz += 1                                                                   <L 641>
                    var_234 = wp::add(var_185, var_233);
                }
                var_235 = wp::where(var_229, var_234, var_185);
            }
            var_236 = wp::where(var_is_sparse, var_235, var_185);
            if (!var_is_sparse) {
                // efc_J_out[worldid, efcid, i] = J                                               <L 643>
                wp::array_store(var_efc_J_out, var_0, var_11, var_187, var_227);
            }
            // Jqvel += J * qvel_in[worldid, i]                                                   <L 645>
            var_237 = wp::address(var_qvel_in, var_0, var_187);
            var_239 = wp::load(var_237);
            var_238 = wp::mul(var_227, var_239);
            var_240 = wp::add(var_183, var_238);
            wp::assign(var_179, var_202);
            wp::assign(var_181, var_226);
            wp::assign(var_183, var_240);
            wp::assign(var_185, var_236);
            goto start_for_5;
        end_for_5:;
        // if is_sparse:                                                                          <L 647>
        if (var_is_sparse) {
            // efc_J_rownnz_out[worldid, efcid] = nnz                                             <L 648>
            wp::array_store(var_efc_J_rownnz_out, var_0, var_11, var_185);
        }
        // _efc_row(                                                                              <L 650>
        // opt_disableflags,                                                                      <L 651>
        // worldid,                                                                               <L 652>
        // opt_timestep[worldid % opt_timestep.shape[0]],                                         <L 653>
        var_241 = &(var_opt_timestep.shape);
        var_244 = wp::load(var_241);
        var_243 = wp::extract(var_244, var_242);
        var_245 = wp::mod(var_0, var_243);
        var_246 = wp::address(var_opt_timestep, var_245);
        // efcid,                                                                                 <L 654>
        // pos,                                                                                   <L 655>
        // pos,                                                                                   <L 656>
        // invweight,                                                                             <L 657>
        // solref,                                                                                <L 658>
        // solimp,                                                                                <L 659>
        // 0.0,                                                                                   <L 660>
        // Jqvel,                                                                                 <L 661>
        // 0.0,                                                                                   <L 662>
        // ConstraintType.EQUALITY,                                                               <L 663>
        // eqid,                                                                                  <L 664>
        // efc_type_out,                                                                          <L 665>
        // efc_id_out,                                                                            <L 666>
        // efc_pos_out,                                                                           <L 667>
        // efc_margin_out,                                                                        <L 668>
        // efc_D_out,                                                                             <L 669>
        // efc_vel_out,                                                                           <L 670>
        // efc_aref_out,                                                                          <L 671>
        // efc_frictionloss_out,                                                                  <L 672>
        var_251 = wp::load(var_246);
        _efc_row_0(var_opt_disableflags, var_0, var_251, var_11, var_122, var_122, var_121, var_33, var_41, var_247, var_183, var_248, var_250, var_3, var_efc_type_out, var_efc_id_out, var_efc_pos_out, var_efc_margin_out, var_efc_D_out, var_efc_vel_out, var_efc_aref_out, var_efc_frictionloss_out);
    }
}



extern "C" __global__ void _contact_elliptic_dcb0f059_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::int32 var_opt_disableflags,
    wp::array_t<wp::float32> var_opt_impratio_invsqrt,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_weldid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::vec_t<2, wp::float32>> var_body_invweight0,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::int32> var_dof_parentid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_flex_vertadr,
    wp::array_t<wp::int32> var_flex_vertbodyid,
    bool var_is_sparse,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::int32 var_njmax_in,
    wp::int32 var_njmax_nnz_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::float32> var_dist_in,
    wp::array_t<wp::int32> var_condim_in,
    wp::array_t<wp::float32> var_includemargin_in,
    wp::array_t<wp::int32> var_worldid_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_geom_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_flex_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_vert_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_pos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_frame_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_friction_in,
    wp::array_t<wp::vec_t<2, wp::float32>> var_solref_in,
    wp::array_t<wp::vec_t<2, wp::float32>> var_solreffriction_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_solimp_in,
    wp::array_t<wp::int32> var_type_in,
    wp::array_t<wp::int32> var_nefc_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_efc_type_out,
    wp::array_t<wp::int32> var_efc_id_out,
    wp::array_t<wp::int32> var_efc_J_rownnz_out,
    wp::array_t<wp::int32> var_efc_J_rowadr_out,
    wp::array_t<wp::int32> var_efc_J_colind_out,
    wp::array_t<wp::float32> var_efc_J_out,
    wp::array_t<wp::float32> var_efc_pos_out,
    wp::array_t<wp::float32> var_efc_margin_out,
    wp::array_t<wp::float32> var_efc_D_out,
    wp::array_t<wp::float32> var_efc_vel_out,
    wp::array_t<wp::float32> var_efc_aref_out,
    wp::array_t<wp::float32> var_efc_frictionloss_out,
    wp::array_t<wp::int32> var_efc_nnz_out)
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
        wp::int32* var_3;
        bool var_4;
        wp::int32 var_5;
        wp::int32* var_6;
        const wp::int32 var_7 = 1;
        wp::int32 var_8;
        wp::int32 var_9;
        bool var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        const wp::int32 var_14 = 1;
        wp::int32 var_15;
        bool var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32* var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        const wp::float32 var_23 = 0.0;
        bool var_24;
        wp::int32* var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        const wp::int32 var_28 = 1;
        wp::int32 var_29;
        bool var_30;
        const wp::int32 var_31 = 1;
        const wp::int32 var_32 = -1;
        wp::shape_t* var_33;
        const wp::int32 var_34 = 0;
        wp::int32 var_35;
        wp::shape_t var_36;
        wp::int32 var_37;
        wp::float32* var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        wp::shape_t* var_41;
        const wp::int32 var_42 = 0;
        wp::int32 var_43;
        wp::shape_t var_44;
        wp::int32 var_45;
        wp::float32* var_46;
        wp::float32 var_47;
        wp::float32 var_48;
        wp::vec_t<2, wp::int32>* var_49;
        wp::vec_t<2, wp::int32> var_50;
        wp::vec_t<2, wp::int32> var_51;
        const wp::int32 var_52 = 0;
        wp::int32 var_53;
        const wp::int32 var_54 = 0;
        bool var_55;
        const wp::int32 var_56 = 0;
        wp::int32 var_57;
        wp::int32* var_58;
        wp::int32 var_59;
        wp::int32 var_60;
        wp::vec_t<2, wp::int32>* var_61;
        wp::vec_t<2, wp::int32> var_62;
        wp::vec_t<2, wp::int32> var_63;
        wp::vec_t<2, wp::int32>* var_64;
        wp::vec_t<2, wp::int32> var_65;
        wp::vec_t<2, wp::int32> var_66;
        const wp::int32 var_67 = 0;
        wp::int32 var_68;
        wp::int32* var_69;
        const wp::int32 var_70 = 0;
        wp::int32 var_71;
        wp::int32 var_72;
        wp::int32 var_73;
        wp::int32* var_74;
        wp::int32 var_75;
        wp::int32 var_76;
        wp::int32 var_77;
        const wp::int32 var_78 = 1;
        wp::int32 var_79;
        const wp::int32 var_80 = 0;
        bool var_81;
        const wp::int32 var_82 = 1;
        wp::int32 var_83;
        wp::int32* var_84;
        wp::int32 var_85;
        wp::int32 var_86;
        wp::vec_t<2, wp::int32>* var_87;
        wp::vec_t<2, wp::int32> var_88;
        wp::vec_t<2, wp::int32> var_89;
        wp::vec_t<2, wp::int32>* var_90;
        wp::vec_t<2, wp::int32> var_91;
        wp::vec_t<2, wp::int32> var_92;
        const wp::int32 var_93 = 1;
        wp::int32 var_94;
        wp::int32* var_95;
        const wp::int32 var_96 = 1;
        wp::int32 var_97;
        wp::int32 var_98;
        wp::int32 var_99;
        wp::int32* var_100;
        wp::int32 var_101;
        wp::int32 var_102;
        wp::vec_t<2, wp::int32> var_103;
        wp::vec_t<2, wp::int32> var_104;
        wp::int32 var_105;
        wp::vec_t<3, wp::float32>* var_106;
        wp::vec_t<3, wp::float32> var_107;
        wp::vec_t<3, wp::float32> var_108;
        wp::mat_t<3, 3, wp::float32>* var_109;
        wp::mat_t<3, 3, wp::float32> var_110;
        wp::mat_t<3, 3, wp::float32> var_111;
        const wp::float32 var_112 = 0.0;
        wp::float32 var_113;
        wp::int32* var_114;
        wp::int32 var_115;
        wp::int32 var_116;
        wp::int32* var_117;
        wp::int32 var_118;
        wp::int32 var_119;
        wp::int32* var_120;
        wp::int32* var_121;
        wp::int32 var_122;
        wp::int32 var_123;
        wp::int32 var_124;
        const wp::int32 var_125 = 1;
        wp::int32 var_126;
        wp::int32 var_127;
        wp::int32* var_128;
        wp::int32* var_129;
        wp::int32 var_130;
        wp::int32 var_131;
        wp::int32 var_132;
        const wp::int32 var_133 = 1;
        wp::int32 var_134;
        wp::int32 var_135;
        wp::int32 var_136;
        wp::int32 var_137;
        const wp::int32 var_138 = 0;
        wp::int32 var_139;
        const wp::int32 var_140 = 0;
        bool var_141;
        const wp::int32 var_142 = 0;
        bool var_143;
        bool var_144;
        wp::int32 var_145;
        bool var_146;
        bool var_147;
        bool var_148;
        bool var_149;
        wp::int32* var_150;
        wp::int32 var_151;
        wp::int32 var_152;
        wp::int32 var_153;
        bool var_154;
        wp::int32* var_155;
        wp::int32 var_156;
        wp::int32 var_157;
        wp::int32 var_158;
        const wp::int32 var_159 = 1;
        wp::int32 var_160;
        wp::int32 var_161;
        wp::int32 var_162;
        bool var_163;
        wp::int32 var_164;
        const wp::int32 var_165 = 0;
        wp::int32 var_166;
        wp::int32 var_167;
        const wp::int32 var_168 = 1;
        wp::int32 var_169;
        wp::int32 var_170;
        wp::int32 var_171;
        const bool var_172 = true;
        bool var_173;
        const wp::int32 var_174 = 0;
        bool var_175;
        bool var_176;
        wp::vec_t<3, wp::float32> var_177;
        wp::vec_t<3, wp::float32> var_178;
        wp::vec_t<3, wp::float32> var_179;
        wp::vec_t<3, wp::float32> var_180;
        const wp::float32 var_181 = 0.0;
        wp::float32 var_182;
        const wp::int32 var_183 = 0;
        const wp::int32 var_184 = 3;
        bool var_185;
        wp::float32 var_186;
        wp::float32 var_187;
        wp::float32 var_188;
        wp::float32 var_189;
        wp::float32 var_190;
        wp::float32 var_191;
        wp::float32 var_192;
        wp::float32 var_193;
        wp::float32 var_194;
        wp::float32 var_195;
        const wp::int32 var_196 = 3;
        wp::int32 var_197;
        wp::float32 var_198;
        wp::float32 var_199;
        wp::float32 var_200;
        wp::float32 var_201;
        wp::float32 var_202;
        const wp::int32 var_203 = 1;
        const wp::int32 var_204 = 3;
        bool var_205;
        wp::float32 var_206;
        wp::float32 var_207;
        wp::float32 var_208;
        wp::float32 var_209;
        wp::float32 var_210;
        wp::float32 var_211;
        wp::float32 var_212;
        wp::float32 var_213;
        wp::float32 var_214;
        wp::float32 var_215;
        wp::float32 var_216;
        const wp::int32 var_217 = 3;
        wp::int32 var_218;
        wp::float32 var_219;
        wp::float32 var_220;
        wp::float32 var_221;
        wp::float32 var_222;
        wp::float32 var_223;
        const wp::int32 var_224 = 2;
        const wp::int32 var_225 = 3;
        bool var_226;
        wp::float32 var_227;
        wp::float32 var_228;
        wp::float32 var_229;
        wp::float32 var_230;
        wp::float32 var_231;
        wp::float32 var_232;
        wp::float32 var_233;
        wp::float32 var_234;
        wp::float32 var_235;
        wp::float32 var_236;
        wp::float32 var_237;
        const wp::int32 var_238 = 3;
        wp::int32 var_239;
        wp::float32 var_240;
        wp::float32 var_241;
        wp::float32 var_242;
        wp::float32 var_243;
        wp::float32 var_244;
        wp::int32 var_245;
        const wp::int32 var_246 = 0;
        const wp::int32 var_247 = 0;
        const wp::int32 var_248 = 1;
        wp::int32 var_249;
        wp::int32 var_250;
        wp::float32* var_251;
        wp::float32 var_252;
        wp::float32 var_253;
        wp::float32 var_254;
        bool var_255;
        bool var_256;
        wp::float32 var_257;
        wp::int32 var_258;
        bool var_259;
        wp::int32* var_260;
        wp::int32 var_261;
        wp::int32 var_262;
        wp::int32 var_263;
        bool var_264;
        wp::int32* var_265;
        wp::int32 var_266;
        wp::int32 var_267;
        wp::int32 var_268;
        wp::int32 var_269;
        wp::int32 var_270;
        wp::int32 var_271;
        const wp::int32 var_272 = 1;
        wp::int32 var_273;
        wp::int32 var_274;
        wp::float32 var_275;
        wp::int32 var_276;
        wp::int32 var_277;
        wp::int32 var_278;
        wp::int32 var_279;
        wp::int32 var_280;
        bool var_281;
        const wp::float32 var_282 = 0.0;
        const wp::int32 var_283 = 1;
        wp::int32 var_284;
        wp::int32 var_285;
        wp::int32 var_286;
        wp::shape_t* var_287;
        const wp::int32 var_288 = 0;
        wp::int32 var_289;
        wp::shape_t var_290;
        wp::int32 var_291;
        wp::vec_t<2, wp::float32>* var_292;
        const wp::int32 var_293 = 0;
        wp::float32 var_294;
        wp::vec_t<2, wp::float32> var_295;
        wp::vec_t<2, wp::float32>* var_296;
        const wp::int32 var_297 = 0;
        wp::float32 var_298;
        wp::vec_t<2, wp::float32> var_299;
        wp::float32 var_300;
        wp::vec_t<2, wp::float32>* var_301;
        wp::vec_t<2, wp::float32> var_302;
        wp::vec_t<2, wp::float32> var_303;
        wp::float32 var_304;
        const wp::int32 var_305 = 0;
        bool var_306;
        wp::vec_t<2, wp::float32>* var_307;
        wp::vec_t<2, wp::float32> var_308;
        wp::vec_t<2, wp::float32> var_309;
        const wp::int32 var_310 = 0;
        wp::float32 var_311;
        const wp::int32 var_312 = 1;
        wp::float32 var_313;
        bool var_314;
        wp::vec_t<2, wp::float32> var_315;
        wp::vec_t<2, wp::float32> var_316;
        wp::float32 var_317;
        wp::float32 var_318;
        wp::vec_t<5, wp::float32>* var_319;
        wp::vec_t<5, wp::float32> var_320;
        wp::vec_t<5, wp::float32> var_321;
        const wp::int32 var_322 = 1;
        bool var_323;
        const wp::int32 var_324 = 0;
        wp::float32 var_325;
        const wp::int32 var_326 = 1;
        wp::int32 var_327;
        wp::float32 var_328;
        wp::float32 var_329;
        wp::float32 var_330;
        wp::float32 var_331;
        wp::float32 var_332;
        wp::float32 var_333;
        const wp::float32 var_334 = 0.0;
        wp::float32 var_335;
        wp::vec_t<2, wp::float32> var_336;
        wp::float32 var_337;
        const wp::int32 var_338 = 1;
        bool var_339;
        const wp::int32 var_340 = 5;
        const wp::int32 var_341 = 7;
        wp::int32 var_342;
        wp::vec_t<5, wp::float32>* var_343;
        const wp::float32 var_344 = 0.0;
        wp::vec_t<5, wp::float32> var_345;
        //---------
        // forward
        // def _contact_elliptic(                                                                 <L 1940>
        // conid, dimid = wp.tid()                                                                <L 1998>
        builtin_tid2d(var_0, var_1);
        // if conid >= nacon_in[0]:                                                               <L 2000>
        var_3 = wp::address(var_nacon_in, var_2);
        var_5 = wp::load(var_3);
        var_4 = (var_0 >= var_5);
        if (var_4) {
            // return                                                                             <L 2001>
            continue;
        }
        // if not type_in[conid] & ContactType.CONSTRAINT:                                        <L 2003>
        var_6 = wp::address(var_type_in, var_0);
        var_9 = wp::load(var_6);
        var_8 = wp::bit_and(var_9, var_7);
        var_10 = wp::unot(var_8);
        if (var_10) {
            // return                                                                             <L 2004>
            continue;
        }
        // condim = condim_in[conid]                                                              <L 2006>
        var_11 = wp::address(var_condim_in, var_0);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // if dimid > condim - 1:                                                                 <L 2008>
        var_15 = wp::sub(var_12, var_14);
        var_16 = (var_1 > var_15);
        if (var_16) {
            // return                                                                             <L 2009>
            continue;
        }
        // includemargin = includemargin_in[conid]                                                <L 2011>
        var_17 = wp::address(var_includemargin_in, var_0);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // pos = dist_in[conid] - includemargin                                                   <L 2012>
        var_20 = wp::address(var_dist_in, var_0);
        var_22 = wp::load(var_20);
        var_21 = wp::sub(var_22, var_18);
        // active = pos < 0.0                                                                     <L 2013>
        var_24 = (var_21 < var_23);
        // if active:                                                                             <L 2015>
        if (var_24) {
            // worldid = worldid_in[conid]                                                        <L 2016>
            var_25 = wp::address(var_worldid_in, var_0);
            var_27 = wp::load(var_25);
            var_26 = wp::copy(var_27);
            // efcid = wp.atomic_add(nefc_out, worldid, 1)                                        <L 2018>
            var_29 = wp::atomic_add(var_nefc_out, var_26, var_28);
            // if efcid >= njmax_in:                                                              <L 2019>
            var_30 = (var_29 >= var_njmax_in);
            if (var_30) {
                // contact_efc_address_out[conid, dimid] = -1                                     <L 2020>
                wp::array_store(var_contact_efc_address_out, var_0, var_1, var_32);
                // return                                                                         <L 2021>
                continue;
            }
            // timestep = opt_timestep[worldid % opt_timestep.shape[0]]                           <L 2023>
            var_33 = &(var_opt_timestep.shape);
            var_36 = wp::load(var_33);
            var_35 = wp::extract(var_36, var_34);
            var_37 = wp::mod(var_26, var_35);
            var_38 = wp::address(var_opt_timestep, var_37);
            var_40 = wp::load(var_38);
            var_39 = wp::copy(var_40);
            // impratio_invsqrt = opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]       <L 2024>
            var_41 = &(var_opt_impratio_invsqrt.shape);
            var_44 = wp::load(var_41);
            var_43 = wp::extract(var_44, var_42);
            var_45 = wp::mod(var_26, var_43);
            var_46 = wp::address(var_opt_impratio_invsqrt, var_45);
            var_48 = wp::load(var_46);
            var_47 = wp::copy(var_48);
            // contact_efc_address_out[conid, dimid] = efcid                                      <L 2025>
            wp::array_store(var_contact_efc_address_out, var_0, var_1, var_29);
            // geom = geom_in[conid]                                                              <L 2027>
            var_49 = wp::address(var_geom_in, var_0);
            var_51 = wp::load(var_49);
            var_50 = wp::copy(var_51);
            // if geom[0] >= 0:                                                                   <L 2029>
            var_53 = wp::extract(var_50, var_52);
            var_55 = (var_53 >= var_54);
            if (var_55) {
                // body1 = geom_bodyid[geom[0]]                                                   <L 2030>
                var_57 = wp::extract(var_50, var_56);
                var_58 = wp::address(var_geom_bodyid, var_57);
                var_60 = wp::load(var_58);
                var_59 = wp::copy(var_60);
            }
            if (!var_55) {
                // flex = flex_in[conid]                                                          <L 2032>
                var_61 = wp::address(var_flex_in, var_0);
                var_63 = wp::load(var_61);
                var_62 = wp::copy(var_63);
                // vert = vert_in[conid]                                                          <L 2033>
                var_64 = wp::address(var_vert_in, var_0);
                var_66 = wp::load(var_64);
                var_65 = wp::copy(var_66);
                // body1 = flex_vertbodyid[flex_vertadr[flex[0]] + vert[0]]                       <L 2034>
                var_68 = wp::extract(var_62, var_67);
                var_69 = wp::address(var_flex_vertadr, var_68);
                var_71 = wp::extract(var_65, var_70);
                var_73 = wp::load(var_69);
                var_72 = wp::add(var_73, var_71);
                var_74 = wp::address(var_flex_vertbodyid, var_72);
                var_76 = wp::load(var_74);
                var_75 = wp::copy(var_76);
            }
            var_77 = wp::where(var_55, var_59, var_75);
            // if geom[1] >= 0:                                                                   <L 2036>
            var_79 = wp::extract(var_50, var_78);
            var_81 = (var_79 >= var_80);
            if (var_81) {
                // body2 = geom_bodyid[geom[1]]                                                   <L 2037>
                var_83 = wp::extract(var_50, var_82);
                var_84 = wp::address(var_geom_bodyid, var_83);
                var_86 = wp::load(var_84);
                var_85 = wp::copy(var_86);
            }
            if (!var_81) {
                // flex = flex_in[conid]                                                          <L 2039>
                var_87 = wp::address(var_flex_in, var_0);
                var_89 = wp::load(var_87);
                var_88 = wp::copy(var_89);
                // vert = vert_in[conid]                                                          <L 2040>
                var_90 = wp::address(var_vert_in, var_0);
                var_92 = wp::load(var_90);
                var_91 = wp::copy(var_92);
                // body2 = flex_vertbodyid[flex_vertadr[flex[1]] + vert[1]]                       <L 2041>
                var_94 = wp::extract(var_88, var_93);
                var_95 = wp::address(var_flex_vertadr, var_94);
                var_97 = wp::extract(var_91, var_96);
                var_99 = wp::load(var_95);
                var_98 = wp::add(var_99, var_97);
                var_100 = wp::address(var_flex_vertbodyid, var_98);
                var_102 = wp::load(var_100);
                var_101 = wp::copy(var_102);
            }
            var_103 = wp::where(var_81, var_62, var_88);
            var_104 = wp::where(var_81, var_65, var_91);
            var_105 = wp::where(var_81, var_85, var_101);
            // con_pos = pos_in[conid]                                                            <L 2043>
            var_106 = wp::address(var_pos_in, var_0);
            var_108 = wp::load(var_106);
            var_107 = wp::copy(var_108);
            // frame = frame_in[conid]                                                            <L 2044>
            var_109 = wp::address(var_frame_in, var_0);
            var_111 = wp::load(var_109);
            var_110 = wp::copy(var_111);
            // Jqvel = float(0.0)                                                                 <L 2046>
            var_113 = wp::float(var_112);
            // body1 = body_weldid[body1]                                                         <L 2049>
            var_114 = wp::address(var_body_weldid, var_77);
            var_116 = wp::load(var_114);
            var_115 = wp::copy(var_116);
            // body2 = body_weldid[body2]                                                         <L 2050>
            var_117 = wp::address(var_body_weldid, var_105);
            var_119 = wp::load(var_117);
            var_118 = wp::copy(var_119);
            // da1 = int(body_dofadr[body1] + body_dofnum[body1] - 1)                             <L 2052>
            var_120 = wp::address(var_body_dofadr, var_115);
            var_121 = wp::address(var_body_dofnum, var_115);
            var_123 = wp::load(var_120);
            var_124 = wp::load(var_121);
            var_122 = wp::add(var_123, var_124);
            var_126 = wp::sub(var_122, var_125);
            var_127 = wp::int(var_126);
            // da2 = int(body_dofadr[body2] + body_dofnum[body2] - 1)                             <L 2053>
            var_128 = wp::address(var_body_dofadr, var_118);
            var_129 = wp::address(var_body_dofnum, var_118);
            var_131 = wp::load(var_128);
            var_132 = wp::load(var_129);
            var_130 = wp::add(var_131, var_132);
            var_134 = wp::sub(var_130, var_133);
            var_135 = wp::int(var_134);
            // if is_sparse:                                                                      <L 2055>
            if (var_is_sparse) {
                // pda1 = da1                                                                     <L 2057>
                var_136 = wp::copy(var_127);
                // pda2 = da2                                                                     <L 2058>
                var_137 = wp::copy(var_135);
                // rownnz = int(0)                                                                <L 2059>
                var_139 = wp::int(var_138);
                // while pda1 >= 0 or pda2 >= 0:                                                  <L 2060>
        start_while_4:;
                var_141 = (var_136 >= var_140);
                var_143 = (var_137 >= var_142);
                var_144 = var_141 || var_143;
        if ((var_144) == false) goto end_while_4;
                    // da = wp.max(pda1, pda2)                                                    <L 2061>
                    var_145 = wp::max(var_136, var_137);
                    // if pda1 == da and pda2 == da:                                              <L 2063>
                    var_146 = (var_136 == var_145);
                    var_147 = (var_137 == var_145);
                    var_148 = var_146 && var_147;
                    if (var_148) {
                        // break                                                                  <L 2064>
                        goto end_while_4;
                    }
                    // if pda1 == da:                                                             <L 2065>
                    var_149 = (var_136 == var_145);
                    if (var_149) {
                        // pda1 = dof_parentid[pda1]                                              <L 2066>
                        var_150 = wp::address(var_dof_parentid, var_136);
                        var_152 = wp::load(var_150);
                        var_151 = wp::copy(var_152);
                    }
                    var_153 = wp::where(var_149, var_151, var_136);
                    // if pda2 == da:                                                             <L 2067>
                    var_154 = (var_137 == var_145);
                    if (var_154) {
                        // pda2 = dof_parentid[pda2]                                              <L 2068>
                        var_155 = wp::address(var_dof_parentid, var_137);
                        var_157 = wp::load(var_155);
                        var_156 = wp::copy(var_157);
                    }
                    var_158 = wp::where(var_154, var_156, var_137);
                    // rownnz += 1                                                                <L 2069>
                    var_160 = wp::add(var_139, var_159);
                    wp::assign(var_136, var_153);
                    wp::assign(var_137, var_158);
                    wp::assign(var_139, var_160);
        goto start_while_4;
        end_while_4:;
                // rowadr = wp.atomic_add(efc_nnz_out, worldid, rownnz)                           <L 2072>
                var_161 = wp::atomic_add(var_efc_nnz_out, var_26, var_139);
                // if rowadr + rownnz > njmax_nnz_in:                                             <L 2073>
                var_162 = wp::add(var_161, var_139);
                var_163 = (var_162 > var_njmax_nnz_in);
                if (var_163) {
                    // return                                                                     <L 2074>
                    continue;
                }
                // efc_J_rowadr_out[worldid, efcid] = rowadr                                      <L 2075>
                wp::array_store(var_efc_J_rowadr_out, var_26, var_29, var_161);
                // efc_J_rownnz_out[worldid, efcid] = rownnz                                      <L 2076>
                wp::array_store(var_efc_J_rownnz_out, var_26, var_29, var_139);
            }
            // da = wp.max(da1, da2)                                                              <L 2078>
            var_164 = wp::max(var_127, var_135);
            // if is_sparse:                                                                      <L 2080>
            if (var_is_sparse) {
                // nnz = int(0)                                                                   <L 2081>
                var_166 = wp::int(var_165);
                // dofid = int(da)                                                                <L 2082>
                var_167 = wp::int(var_164);
            }
            if (!var_is_sparse) {
                // dofid = int(nv - 1)                                                            <L 2084>
                var_169 = wp::sub(var_nv, var_168);
                var_170 = wp::int(var_169);
            }
            var_171 = wp::where(var_is_sparse, var_167, var_170);
            // while True:                                                                        <L 2086>
        start_while_7:;
        if ((var_172) == false) goto end_while_7;
                // if is_sparse:                                                                  <L 2087>
                if (var_is_sparse) {
                    // if nnz >= rownnz:                                                          <L 2088>
                    var_173 = (var_166 >= var_139);
                    if (var_173) {
                        // break                                                                  <L 2089>
                        goto end_while_7;
                    }
                }
                if (!var_is_sparse) {
                    // if dofid < 0:                                                              <L 2091>
                    var_175 = (var_171 < var_174);
                    if (var_175) {
                        // break                                                                  <L 2092>
                        goto end_while_7;
                    }
                }
                // if dofid == da:                                                                <L 2094>
                var_176 = (var_171 == var_164);
                if (var_176) {
                    // jac1p, jac1r = support.jac_dof(                                            <L 2096>
                    // body_parentid,                                                             <L 2097>
                    // body_rootid,                                                               <L 2098>
                    // dof_bodyid,                                                                <L 2099>
                    // subtree_com_in,                                                            <L 2100>
                    // cdof_in,                                                                   <L 2101>
                    // con_pos,                                                                   <L 2102>
                    // body1,                                                                     <L 2103>
                    // dofid,                                                                     <L 2104>
                    // worldid,                                                                   <L 2105>
                    jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_subtree_com_in, var_cdof_in, var_107, var_115, var_171, var_26, var_177, var_178);
                    // jac2p, jac2r = support.jac_dof(                                            <L 2107>
                    // body_parentid,                                                             <L 2108>
                    // body_rootid,                                                               <L 2109>
                    // dof_bodyid,                                                                <L 2110>
                    // subtree_com_in,                                                            <L 2111>
                    // cdof_in,                                                                   <L 2112>
                    // con_pos,                                                                   <L 2113>
                    // body2,                                                                     <L 2114>
                    // dofid,                                                                     <L 2115>
                    // worldid,                                                                   <L 2116>
                    jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_subtree_com_in, var_cdof_in, var_107, var_118, var_171, var_26, var_179, var_180);
                    // J = float(0.0)                                                             <L 2119>
                    var_182 = wp::float(var_181);
                    // for xyz in range(3):                                                       <L 2120>
                    // if dimid < 3:                                                              <L 2121>
                    var_185 = (var_1 < var_184);
                    if (var_185) {
                        // jac_dif = jac2p[xyz] - jac1p[xyz]                                      <L 2122>
                        var_186 = wp::extract(var_179, var_183);
                        var_187 = wp::extract(var_177, var_183);
                        var_188 = wp::sub(var_186, var_187);
                        // J += frame[dimid, xyz] * jac_dif                                       <L 2123>
                        var_189 = wp::extract(var_110, var_1, var_183);
                        var_190 = wp::mul(var_189, var_188);
                        var_191 = wp::add(var_182, var_190);
                    }
                    var_192 = wp::where(var_185, var_191, var_182);
                    if (!var_185) {
                        // jac_dif = jac2r[xyz] - jac1r[xyz]                                      <L 2125>
                        var_193 = wp::extract(var_180, var_183);
                        var_194 = wp::extract(var_178, var_183);
                        var_195 = wp::sub(var_193, var_194);
                        // J += frame[dimid - 3, xyz] * jac_dif                                   <L 2126>
                        var_197 = wp::sub(var_1, var_196);
                        var_198 = wp::extract(var_110, var_197, var_183);
                        var_199 = wp::mul(var_198, var_195);
                        var_200 = wp::add(var_192, var_199);
                    }
                    var_201 = wp::where(var_185, var_192, var_200);
                    var_202 = wp::where(var_185, var_188, var_195);
                    // if dimid < 3:                                                              <L 2121>
                    var_205 = (var_1 < var_204);
                    if (var_205) {
                        // jac_dif = jac2p[xyz] - jac1p[xyz]                                      <L 2122>
                        var_206 = wp::extract(var_179, var_203);
                        var_207 = wp::extract(var_177, var_203);
                        var_208 = wp::sub(var_206, var_207);
                        // J += frame[dimid, xyz] * jac_dif                                       <L 2123>
                        var_209 = wp::extract(var_110, var_1, var_203);
                        var_210 = wp::mul(var_209, var_208);
                        var_211 = wp::add(var_201, var_210);
                    }
                    var_212 = wp::where(var_205, var_211, var_201);
                    var_213 = wp::where(var_205, var_208, var_202);
                    if (!var_205) {
                        // jac_dif = jac2r[xyz] - jac1r[xyz]                                      <L 2125>
                        var_214 = wp::extract(var_180, var_203);
                        var_215 = wp::extract(var_178, var_203);
                        var_216 = wp::sub(var_214, var_215);
                        // J += frame[dimid - 3, xyz] * jac_dif                                   <L 2126>
                        var_218 = wp::sub(var_1, var_217);
                        var_219 = wp::extract(var_110, var_218, var_203);
                        var_220 = wp::mul(var_219, var_216);
                        var_221 = wp::add(var_212, var_220);
                    }
                    var_222 = wp::where(var_205, var_212, var_221);
                    var_223 = wp::where(var_205, var_213, var_216);
                    // if dimid < 3:                                                              <L 2121>
                    var_226 = (var_1 < var_225);
                    if (var_226) {
                        // jac_dif = jac2p[xyz] - jac1p[xyz]                                      <L 2122>
                        var_227 = wp::extract(var_179, var_224);
                        var_228 = wp::extract(var_177, var_224);
                        var_229 = wp::sub(var_227, var_228);
                        // J += frame[dimid, xyz] * jac_dif                                       <L 2123>
                        var_230 = wp::extract(var_110, var_1, var_224);
                        var_231 = wp::mul(var_230, var_229);
                        var_232 = wp::add(var_222, var_231);
                    }
                    var_233 = wp::where(var_226, var_232, var_222);
                    var_234 = wp::where(var_226, var_229, var_223);
                    if (!var_226) {
                        // jac_dif = jac2r[xyz] - jac1r[xyz]                                      <L 2125>
                        var_235 = wp::extract(var_180, var_224);
                        var_236 = wp::extract(var_178, var_224);
                        var_237 = wp::sub(var_235, var_236);
                        // J += frame[dimid - 3, xyz] * jac_dif                                   <L 2126>
                        var_239 = wp::sub(var_1, var_238);
                        var_240 = wp::extract(var_110, var_239, var_224);
                        var_241 = wp::mul(var_240, var_237);
                        var_242 = wp::add(var_233, var_241);
                    }
                    var_243 = wp::where(var_226, var_233, var_242);
                    var_244 = wp::where(var_226, var_234, var_237);
                    // if is_sparse:                                                              <L 2128>
                    if (var_is_sparse) {
                        // sparseid = rowadr + nnz                                                <L 2129>
                        var_245 = wp::add(var_161, var_166);
                        // efc_J_colind_out[worldid, 0, sparseid] = dofid                         <L 2130>
                        wp::array_store(var_efc_J_colind_out, var_26, var_246, var_245, var_171);
                        // efc_J_out[worldid, 0, sparseid] = J                                    <L 2131>
                        wp::array_store(var_efc_J_out, var_26, var_247, var_245, var_243);
                        // nnz += 1                                                               <L 2132>
                        var_249 = wp::add(var_166, var_248);
                    }
                    var_250 = wp::where(var_is_sparse, var_249, var_166);
                    if (!var_is_sparse) {
                        // efc_J_out[worldid, efcid, dofid] = J                                   <L 2134>
                        wp::array_store(var_efc_J_out, var_26, var_29, var_171, var_243);
                    }
                    // Jqvel += J * qvel_in[worldid, dofid]                                       <L 2135>
                    var_251 = wp::address(var_qvel_in, var_26, var_171);
                    var_253 = wp::load(var_251);
                    var_252 = wp::mul(var_243, var_253);
                    var_254 = wp::add(var_113, var_252);
                    // if is_sparse and nnz >= rownnz:                                            <L 2136>
                    var_255 = (var_250 >= var_139);
                    var_256 = var_is_sparse && var_255;
                    if (var_256) {
                        // break                                                                  <L 2137>
                        wp::assign(var_113, var_254);
                        wp::assign(var_166, var_250);
                        goto end_while_7;
                    }
                    var_257 = wp::where(var_256, var_113, var_254);
                    var_258 = wp::where(var_256, var_166, var_250);
                    // if da1 == da:                                                              <L 2140>
                    var_259 = (var_127 == var_164);
                    if (var_259) {
                        // da1 = dof_parentid[da1]                                                <L 2141>
                        var_260 = wp::address(var_dof_parentid, var_127);
                        var_262 = wp::load(var_260);
                        var_261 = wp::copy(var_262);
                    }
                    var_263 = wp::where(var_259, var_261, var_127);
                    // if da2 == da:                                                              <L 2142>
                    var_264 = (var_135 == var_164);
                    if (var_264) {
                        // da2 = dof_parentid[da2]                                                <L 2143>
                        var_265 = wp::address(var_dof_parentid, var_135);
                        var_267 = wp::load(var_265);
                        var_266 = wp::copy(var_267);
                    }
                    var_268 = wp::where(var_264, var_266, var_135);
                    // da = wp.max(da1, da2)                                                      <L 2144>
                    var_269 = wp::max(var_263, var_268);
                    // if is_sparse:                                                              <L 2145>
                    if (var_is_sparse) {
                        // dofid = da                                                             <L 2146>
                        var_270 = wp::copy(var_269);
                    }
                    var_271 = wp::where(var_is_sparse, var_270, var_171);
                    if (!var_is_sparse) {
                        // dofid -= 1                                                             <L 2148>
                        var_273 = wp::sub(var_271, var_272);
                    }
                    var_274 = wp::where(var_is_sparse, var_271, var_273);
                }
                var_275 = wp::where(var_176, var_257, var_113);
                var_276 = wp::where(var_176, var_263, var_127);
                var_277 = wp::where(var_176, var_268, var_135);
                var_278 = wp::where(var_176, var_269, var_164);
                var_279 = wp::where(var_176, var_258, var_166);
                var_280 = wp::where(var_176, var_274, var_171);
                if (!var_176) {
                    // if not is_sparse:                                                          <L 2150>
                    var_281 = wp::unot(var_is_sparse);
                    if (var_281) {
                        // efc_J_out[worldid, efcid, dofid] = 0.0                                 <L 2151>
                        wp::array_store(var_efc_J_out, var_26, var_29, var_280, var_282);
                        // dofid -= 1                                                             <L 2152>
                        var_284 = wp::sub(var_280, var_283);
                    }
                    var_285 = wp::where(var_281, var_284, var_280);
                }
                var_286 = wp::where(var_176, var_280, var_285);
                wp::assign(var_113, var_275);
                wp::assign(var_127, var_276);
                wp::assign(var_135, var_277);
                wp::assign(var_164, var_278);
                wp::assign(var_166, var_279);
                wp::assign(var_171, var_286);
        goto start_while_7;
        end_while_7:;
            // body_invweight0_id = worldid % body_invweight0.shape[0]                            <L 2154>
            var_287 = &(var_body_invweight0.shape);
            var_290 = wp::load(var_287);
            var_289 = wp::extract(var_290, var_288);
            var_291 = wp::mod(var_26, var_289);
            // invweight = body_invweight0[body_invweight0_id, body1][0] + body_invweight0[body_invweight0_id, body2][0]       <L 2155>
            var_292 = wp::address(var_body_invweight0, var_291, var_115);
            var_295 = wp::load(var_292);
            var_294 = wp::extract(var_295, var_293);
            var_296 = wp::address(var_body_invweight0, var_291, var_118);
            var_299 = wp::load(var_296);
            var_298 = wp::extract(var_299, var_297);
            var_300 = wp::add(var_294, var_298);
            // ref = solref_in[conid]                                                             <L 2157>
            var_301 = wp::address(var_solref_in, var_0);
            var_303 = wp::load(var_301);
            var_302 = wp::copy(var_303);
            // pos_aref = pos                                                                     <L 2158>
            var_304 = wp::copy(var_21);
            // if dimid > 0:                                                                      <L 2160>
            var_306 = (var_1 > var_305);
            if (var_306) {
                // solreffriction = solreffriction_in[conid]                                      <L 2161>
                var_307 = wp::address(var_solreffriction_in, var_0);
                var_309 = wp::load(var_307);
                var_308 = wp::copy(var_309);
                // if solreffriction[0] or solreffriction[1]:                                     <L 2164>
                var_311 = wp::extract(var_308, var_310);
                var_313 = wp::extract(var_308, var_312);
                var_314 = var_311 || var_313;
                if (var_314) {
                    // ref = solreffriction                                                       <L 2165>
                    var_315 = wp::copy(var_308);
                }
                var_316 = wp::where(var_314, var_315, var_302);
                // invweight = invweight * impratio_invsqrt * impratio_invsqrt                    <L 2167>
                var_317 = wp::mul(var_300, var_47);
                var_318 = wp::mul(var_317, var_47);
                // friction = friction_in[conid]                                                  <L 2168>
                var_319 = wp::address(var_friction_in, var_0);
                var_321 = wp::load(var_319);
                var_320 = wp::copy(var_321);
                // if dimid > 1:                                                                  <L 2170>
                var_323 = (var_1 > var_322);
                if (var_323) {
                    // fri0 = friction[0]                                                         <L 2171>
                    var_325 = wp::extract(var_320, var_324);
                    // frii = friction[dimid - 1]                                                 <L 2172>
                    var_327 = wp::sub(var_1, var_326);
                    var_328 = wp::extract(var_320, var_327);
                    // fri = fri0 * fri0 / (frii * frii)                                          <L 2173>
                    var_329 = wp::mul(var_325, var_325);
                    var_330 = wp::mul(var_328, var_328);
                    var_331 = wp::div(var_329, var_330);
                    // invweight *= fri                                                           <L 2174>
                    var_332 = wp::mul(var_318, var_331);
                }
                var_333 = wp::where(var_323, var_332, var_318);
                // pos_aref = 0.0                                                                 <L 2176>
            }
            var_335 = wp::where(var_306, var_333, var_300);
            var_336 = wp::where(var_306, var_316, var_302);
            var_337 = wp::where(var_306, var_334, var_304);
            // if condim == 1:                                                                    <L 2178>
            var_339 = (var_12 == var_338);
            if (var_339) {
                // efc_type = ConstraintType.CONTACT_FRICTIONLESS                                 <L 2179>
            }
            if (!var_339) {
                // efc_type = ConstraintType.CONTACT_ELLIPTIC                                     <L 2181>
            }
            var_342 = wp::where(var_339, var_340, var_341);
            // _efc_row(                                                                          <L 2183>
            // opt_disableflags,                                                                  <L 2184>
            // worldid,                                                                           <L 2185>
            // timestep,                                                                          <L 2186>
            // efcid,                                                                             <L 2187>
            // pos_aref,                                                                          <L 2188>
            // pos,                                                                               <L 2189>
            // invweight,                                                                         <L 2190>
            // ref,                                                                               <L 2191>
            // solimp_in[conid],                                                                  <L 2192>
            var_343 = wp::address(var_solimp_in, var_0);
            // includemargin,                                                                     <L 2193>
            // Jqvel,                                                                             <L 2194>
            // 0.0,                                                                               <L 2195>
            // efc_type,                                                                          <L 2196>
            // conid,                                                                             <L 2197>
            // efc_type_out,                                                                      <L 2198>
            // efc_id_out,                                                                        <L 2199>
            // efc_pos_out,                                                                       <L 2200>
            // efc_margin_out,                                                                    <L 2201>
            // efc_D_out,                                                                         <L 2202>
            // efc_vel_out,                                                                       <L 2203>
            // efc_aref_out,                                                                      <L 2204>
            // efc_frictionloss_out,                                                              <L 2205>
            var_345 = wp::load(var_343);
            _efc_row_0(var_opt_disableflags, var_26, var_39, var_29, var_337, var_21, var_335, var_336, var_345, var_18, var_113, var_344, var_342, var_0, var_efc_type_out, var_efc_id_out, var_efc_pos_out, var_efc_margin_out, var_efc_D_out, var_efc_vel_out, var_efc_aref_out, var_efc_frictionloss_out);
        }
    }
}



extern "C" __global__ void _limit_slide_hinge_7dba3730_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::int32 var_opt_disableflags,
    wp::array_t<wp::int32> var_jnt_qposadr,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::vec_t<2, wp::float32>> var_jnt_solref,
    wp::array_t<wp::vec_t<5, wp::float32>> var_jnt_solimp,
    wp::array_t<wp::vec_t<2, wp::float32>> var_jnt_range,
    wp::array_t<wp::float32> var_jnt_margin,
    wp::array_t<wp::float32> var_dof_invweight0,
    bool var_is_sparse,
    wp::array_t<wp::int32> var_jnt_limited_slide_hinge_adr,
    wp::array_t<wp::float32> var_qpos_in,
    wp::array_t<wp::float32> var_qvel_in,
    wp::int32 var_njmax_in,
    wp::int32 var_njmax_nnz_in,
    wp::array_t<wp::int32> var_nl_out,
    wp::array_t<wp::int32> var_nefc_out,
    wp::array_t<wp::int32> var_efc_type_out,
    wp::array_t<wp::int32> var_efc_id_out,
    wp::array_t<wp::int32> var_efc_J_rownnz_out,
    wp::array_t<wp::int32> var_efc_J_rowadr_out,
    wp::array_t<wp::int32> var_efc_J_colind_out,
    wp::array_t<wp::float32> var_efc_J_out,
    wp::array_t<wp::float32> var_efc_pos_out,
    wp::array_t<wp::float32> var_efc_margin_out,
    wp::array_t<wp::float32> var_efc_D_out,
    wp::array_t<wp::float32> var_efc_vel_out,
    wp::array_t<wp::float32> var_efc_aref_out,
    wp::array_t<wp::float32> var_efc_frictionloss_out,
    wp::array_t<wp::int32> var_efc_nnz_out)
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
        wp::shape_t* var_5;
        const wp::int32 var_6 = 0;
        wp::int32 var_7;
        wp::shape_t var_8;
        wp::int32 var_9;
        wp::vec_t<2, wp::float32>* var_10;
        wp::vec_t<2, wp::float32> var_11;
        wp::vec_t<2, wp::float32> var_12;
        wp::int32* var_13;
        wp::float32* var_14;
        wp::int32 var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        wp::shape_t* var_18;
        const wp::int32 var_19 = 0;
        wp::int32 var_20;
        wp::shape_t var_21;
        wp::int32 var_22;
        wp::float32* var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        const wp::int32 var_26 = 0;
        wp::float32 var_27;
        wp::float32 var_28;
        const wp::int32 var_29 = 1;
        wp::float32 var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        const wp::int32 var_34 = 0;
        bool var_35;
        const wp::int32 var_36 = 1;
        wp::int32 var_37;
        const wp::int32 var_38 = 1;
        wp::int32 var_39;
        bool var_40;
        wp::int32* var_41;
        wp::int32 var_42;
        wp::int32 var_43;
        bool var_44;
        wp::float32 var_45;
        const wp::float32 var_46 = 2.0;
        wp::float32 var_47;
        const wp::float32 var_48 = 1.0;
        wp::float32 var_49;
        const wp::int32 var_50 = 1;
        const wp::int32 var_51 = 1;
        wp::int32 var_52;
        const wp::int32 var_53 = 1;
        wp::int32 var_54;
        bool var_55;
        const wp::int32 var_56 = 0;
        const wp::int32 var_57 = 0;
        wp::range_t var_58;
        wp::int32 var_59;
        const wp::float32 var_60 = 0.0;
        wp::float32* var_61;
        wp::float32 var_62;
        wp::float32 var_63;
        wp::shape_t* var_64;
        const wp::int32 var_65 = 0;
        wp::int32 var_66;
        wp::shape_t var_67;
        wp::int32 var_68;
        wp::shape_t* var_69;
        const wp::int32 var_70 = 0;
        wp::int32 var_71;
        wp::shape_t var_72;
        wp::int32 var_73;
        wp::shape_t* var_74;
        const wp::int32 var_75 = 0;
        wp::int32 var_76;
        wp::shape_t var_77;
        wp::int32 var_78;
        wp::shape_t* var_79;
        const wp::int32 var_80 = 0;
        wp::int32 var_81;
        wp::shape_t var_82;
        wp::int32 var_83;
        wp::float32* var_84;
        wp::float32* var_85;
        wp::vec_t<2, wp::float32>* var_86;
        wp::vec_t<5, wp::float32>* var_87;
        const wp::float32 var_88 = 0.0;
        const wp::int32 var_89 = 3;
        const wp::int32 var_90 = 3;
        wp::float32 var_91;
        wp::float32 var_92;
        wp::vec_t<2, wp::float32> var_93;
        wp::vec_t<5, wp::float32> var_94;
        //---------
        // forward
        // def _limit_slide_hinge(                                                                <L 1317>
        // worldid, jntlimitedid = wp.tid()                                                       <L 1354>
        builtin_tid2d(var_0, var_1);
        // jntid = jnt_limited_slide_hinge_adr[jntlimitedid]                                      <L 1355>
        var_2 = wp::address(var_jnt_limited_slide_hinge_adr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // jnt_range_id = worldid % jnt_range.shape[0]                                            <L 1356>
        var_5 = &(var_jnt_range.shape);
        var_8 = wp::load(var_5);
        var_7 = wp::extract(var_8, var_6);
        var_9 = wp::mod(var_0, var_7);
        // jntrange = jnt_range[jnt_range_id, jntid]                                              <L 1357>
        var_10 = wp::address(var_jnt_range, var_9, var_3);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // qpos = qpos_in[worldid, jnt_qposadr[jntid]]                                            <L 1359>
        var_13 = wp::address(var_jnt_qposadr, var_3);
        var_15 = wp::load(var_13);
        var_14 = wp::address(var_qpos_in, var_0, var_15);
        var_17 = wp::load(var_14);
        var_16 = wp::copy(var_17);
        // jnt_margin_id = worldid % jnt_margin.shape[0]                                          <L 1360>
        var_18 = &(var_jnt_margin.shape);
        var_21 = wp::load(var_18);
        var_20 = wp::extract(var_21, var_19);
        var_22 = wp::mod(var_0, var_20);
        // jntmargin = jnt_margin[jnt_margin_id, jntid]                                           <L 1361>
        var_23 = wp::address(var_jnt_margin, var_22, var_3);
        var_25 = wp::load(var_23);
        var_24 = wp::copy(var_25);
        // dist_min, dist_max = qpos - jntrange[0], jntrange[1] - qpos                            <L 1362>
        var_27 = wp::extract(var_11, var_26);
        var_28 = wp::sub(var_16, var_27);
        var_30 = wp::extract(var_11, var_29);
        var_31 = wp::sub(var_30, var_16);
        // pos = wp.min(dist_min, dist_max) - jntmargin                                           <L 1363>
        var_32 = wp::min(var_28, var_31);
        var_33 = wp::sub(var_32, var_24);
        // active = pos < 0                                                                       <L 1364>
        var_35 = (var_33 < var_34);
        // if active:                                                                             <L 1366>
        if (var_35) {
            // wp.atomic_add(nl_out, worldid, 1)                                                  <L 1367>
            var_37 = wp::atomic_add(var_nl_out, var_0, var_36);
            // efcid = wp.atomic_add(nefc_out, worldid, 1)                                        <L 1368>
            var_39 = wp::atomic_add(var_nefc_out, var_0, var_38);
            // if efcid >= njmax_in:                                                              <L 1370>
            var_40 = (var_39 >= var_njmax_in);
            if (var_40) {
                // return                                                                         <L 1371>
                continue;
            }
            // dofadr = jnt_dofadr[jntid]                                                         <L 1373>
            var_41 = wp::address(var_jnt_dofadr, var_3);
            var_43 = wp::load(var_41);
            var_42 = wp::copy(var_43);
            // J = float(dist_min < dist_max) * 2.0 - 1.0                                         <L 1375>
            var_44 = (var_28 < var_31);
            var_45 = wp::float(var_44);
            var_47 = wp::mul(var_45, var_46);
            var_49 = wp::sub(var_47, var_48);
            // if is_sparse:                                                                      <L 1377>
            if (var_is_sparse) {
                // efc_J_rownnz_out[worldid, efcid] = 1                                           <L 1378>
                wp::array_store(var_efc_J_rownnz_out, var_0, var_39, var_50);
                // rowadr = wp.atomic_add(efc_nnz_out, worldid, 1)                                <L 1379>
                var_52 = wp::atomic_add(var_efc_nnz_out, var_0, var_51);
                // if rowadr + 1 > njmax_nnz_in:                                                  <L 1380>
                var_54 = wp::add(var_52, var_53);
                var_55 = (var_54 > var_njmax_nnz_in);
                if (var_55) {
                    // return                                                                     <L 1381>
                    continue;
                }
                // efc_J_rowadr_out[worldid, efcid] = rowadr                                      <L 1382>
                wp::array_store(var_efc_J_rowadr_out, var_0, var_39, var_52);
                // efc_J_colind_out[worldid, 0, rowadr] = dofadr                                  <L 1383>
                wp::array_store(var_efc_J_colind_out, var_0, var_56, var_52, var_42);
                // efc_J_out[worldid, 0, rowadr] = J                                              <L 1384>
                wp::array_store(var_efc_J_out, var_0, var_57, var_52, var_49);
            }
            if (!var_is_sparse) {
                // for i in range(nv):                                                            <L 1386>
                var_58 = wp::range(var_nv);
                start_for_2:;
                    if (iter_cmp(var_58) == 0) goto end_for_2;
                    var_59 = wp::iter_next(var_58);
                    // efc_J_out[worldid, efcid, i] = 0.0                                         <L 1387>
                    wp::array_store(var_efc_J_out, var_0, var_39, var_59, var_60);
                    goto start_for_2;
                end_for_2:;
                // efc_J_out[worldid, efcid, dofadr] = J                                          <L 1388>
                wp::array_store(var_efc_J_out, var_0, var_39, var_42, var_49);
            }
            // Jqvel = J * qvel_in[worldid, dofadr]                                               <L 1390>
            var_61 = wp::address(var_qvel_in, var_0, var_42);
            var_63 = wp::load(var_61);
            var_62 = wp::mul(var_49, var_63);
            // dof_invweight0_id = worldid % dof_invweight0.shape[0]                              <L 1392>
            var_64 = &(var_dof_invweight0.shape);
            var_67 = wp::load(var_64);
            var_66 = wp::extract(var_67, var_65);
            var_68 = wp::mod(var_0, var_66);
            // jnt_solref_id = worldid % jnt_solref.shape[0]                                      <L 1393>
            var_69 = &(var_jnt_solref.shape);
            var_72 = wp::load(var_69);
            var_71 = wp::extract(var_72, var_70);
            var_73 = wp::mod(var_0, var_71);
            // jnt_solimp_id = worldid % jnt_solimp.shape[0]                                      <L 1394>
            var_74 = &(var_jnt_solimp.shape);
            var_77 = wp::load(var_74);
            var_76 = wp::extract(var_77, var_75);
            var_78 = wp::mod(var_0, var_76);
            // _efc_row(                                                                          <L 1395>
            // opt_disableflags,                                                                  <L 1396>
            // worldid,                                                                           <L 1397>
            // opt_timestep[worldid % opt_timestep.shape[0]],                                     <L 1398>
            var_79 = &(var_opt_timestep.shape);
            var_82 = wp::load(var_79);
            var_81 = wp::extract(var_82, var_80);
            var_83 = wp::mod(var_0, var_81);
            var_84 = wp::address(var_opt_timestep, var_83);
            // efcid,                                                                             <L 1399>
            // pos,                                                                               <L 1400>
            // pos,                                                                               <L 1401>
            // dof_invweight0[dof_invweight0_id, dofadr],                                         <L 1402>
            var_85 = wp::address(var_dof_invweight0, var_68, var_42);
            // jnt_solref[jnt_solref_id, jntid],                                                  <L 1403>
            var_86 = wp::address(var_jnt_solref, var_73, var_3);
            // jnt_solimp[jnt_solimp_id, jntid],                                                  <L 1404>
            var_87 = wp::address(var_jnt_solimp, var_78, var_3);
            // jntmargin,                                                                         <L 1405>
            // Jqvel,                                                                             <L 1406>
            // 0.0,                                                                               <L 1407>
            // ConstraintType.LIMIT_JOINT,                                                        <L 1408>
            // jntid,                                                                             <L 1409>
            // efc_type_out,                                                                      <L 1410>
            // efc_id_out,                                                                        <L 1411>
            // efc_pos_out,                                                                       <L 1412>
            // efc_margin_out,                                                                    <L 1413>
            // efc_D_out,                                                                         <L 1414>
            // efc_vel_out,                                                                       <L 1415>
            // efc_aref_out,                                                                      <L 1416>
            // efc_frictionloss_out,                                                              <L 1417>
            var_91 = wp::load(var_84);
            var_92 = wp::load(var_85);
            var_93 = wp::load(var_86);
            var_94 = wp::load(var_87);
            _efc_row_0(var_opt_disableflags, var_0, var_91, var_39, var_33, var_33, var_92, var_93, var_94, var_24, var_62, var_88, var_90, var_3, var_efc_type_out, var_efc_id_out, var_efc_pos_out, var_efc_margin_out, var_efc_D_out, var_efc_vel_out, var_efc_aref_out, var_efc_frictionloss_out);
        }
    }
}



extern "C" __global__ void _limit_tendon_b09b81d0_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::int32 var_opt_disableflags,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::vec_t<2, wp::float32>> var_tendon_solref_lim,
    wp::array_t<wp::vec_t<5, wp::float32>> var_tendon_solimp_lim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_tendon_range,
    wp::array_t<wp::float32> var_tendon_margin,
    wp::array_t<wp::float32> var_tendon_invweight0,
    bool var_is_sparse,
    wp::array_t<wp::int32> var_tendon_limited_adr,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::array_t<wp::float32> var_ten_length_in,
    wp::int32 var_njmax_in,
    wp::int32 var_njmax_nnz_in,
    wp::array_t<wp::int32> var_nl_out,
    wp::array_t<wp::int32> var_nefc_out,
    wp::array_t<wp::int32> var_efc_type_out,
    wp::array_t<wp::int32> var_efc_id_out,
    wp::array_t<wp::int32> var_efc_J_rownnz_out,
    wp::array_t<wp::int32> var_efc_J_rowadr_out,
    wp::array_t<wp::int32> var_efc_J_colind_out,
    wp::array_t<wp::float32> var_efc_J_out,
    wp::array_t<wp::float32> var_efc_pos_out,
    wp::array_t<wp::float32> var_efc_margin_out,
    wp::array_t<wp::float32> var_efc_D_out,
    wp::array_t<wp::float32> var_efc_vel_out,
    wp::array_t<wp::float32> var_efc_aref_out,
    wp::array_t<wp::float32> var_efc_frictionloss_out,
    wp::array_t<wp::int32> var_efc_nnz_out)
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
        wp::shape_t* var_5;
        const wp::int32 var_6 = 0;
        wp::int32 var_7;
        wp::shape_t var_8;
        wp::int32 var_9;
        wp::vec_t<2, wp::float32>* var_10;
        wp::vec_t<2, wp::float32> var_11;
        wp::vec_t<2, wp::float32> var_12;
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        const wp::int32 var_16 = 0;
        wp::float32 var_17;
        wp::float32 var_18;
        const wp::int32 var_19 = 1;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::shape_t* var_22;
        const wp::int32 var_23 = 0;
        wp::int32 var_24;
        wp::shape_t var_25;
        wp::int32 var_26;
        wp::float32* var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        const wp::int32 var_32 = 0;
        bool var_33;
        const wp::int32 var_34 = 1;
        wp::int32 var_35;
        const wp::int32 var_36 = 1;
        wp::int32 var_37;
        bool var_38;
        const wp::float32 var_39 = 0.0;
        wp::float32 var_40;
        bool var_41;
        wp::float32 var_42;
        const wp::float32 var_43 = 2.0;
        wp::float32 var_44;
        const wp::float32 var_45 = 1.0;
        wp::float32 var_46;
        wp::int32* var_47;
        wp::int32 var_48;
        wp::int32 var_49;
        wp::int32* var_50;
        wp::int32 var_51;
        wp::int32 var_52;
        wp::int32 var_53;
        wp::int32 var_54;
        bool var_55;
        wp::range_t var_56;
        wp::int32 var_57;
        wp::int32 var_58;
        wp::int32 var_59;
        wp::int32* var_60;
        wp::int32 var_61;
        wp::int32 var_62;
        wp::float32* var_63;
        wp::float32 var_64;
        wp::float32 var_65;
        const wp::int32 var_66 = 0;
        const wp::int32 var_67 = 0;
        wp::float32* var_68;
        wp::float32 var_69;
        wp::float32 var_70;
        wp::float32 var_71;
        const wp::int32 var_72 = 0;
        wp::int32 var_73;
        wp::int32* var_74;
        wp::int32 var_75;
        wp::int32 var_76;
        wp::range_t var_77;
        wp::int32 var_78;
        bool var_79;
        bool var_80;
        bool var_81;
        wp::int32 var_82;
        wp::float32* var_83;
        wp::float32 var_84;
        wp::float32 var_85;
        wp::float32* var_86;
        wp::float32 var_87;
        wp::float32 var_88;
        wp::float32 var_89;
        const wp::int32 var_90 = 1;
        wp::int32 var_91;
        bool var_92;
        wp::int32 var_93;
        wp::int32* var_94;
        wp::int32 var_95;
        wp::int32 var_96;
        wp::int32 var_97;
        wp::float32 var_98;
        wp::int32 var_99;
        wp::float32 var_100;
        wp::int32 var_101;
        const wp::float32 var_102 = 0.0;
        wp::int32 var_103;
        wp::int32 var_104;
        wp::shape_t* var_105;
        const wp::int32 var_106 = 0;
        wp::int32 var_107;
        wp::shape_t var_108;
        wp::int32 var_109;
        wp::shape_t* var_110;
        const wp::int32 var_111 = 0;
        wp::int32 var_112;
        wp::shape_t var_113;
        wp::int32 var_114;
        wp::shape_t* var_115;
        const wp::int32 var_116 = 0;
        wp::int32 var_117;
        wp::shape_t var_118;
        wp::int32 var_119;
        wp::shape_t* var_120;
        const wp::int32 var_121 = 0;
        wp::int32 var_122;
        wp::shape_t var_123;
        wp::int32 var_124;
        wp::float32* var_125;
        wp::float32* var_126;
        wp::vec_t<2, wp::float32>* var_127;
        wp::vec_t<5, wp::float32>* var_128;
        const wp::float32 var_129 = 0.0;
        const wp::int32 var_130 = 4;
        const wp::int32 var_131 = 4;
        wp::float32 var_132;
        wp::float32 var_133;
        wp::vec_t<2, wp::float32> var_134;
        wp::vec_t<5, wp::float32> var_135;
        //---------
        // forward
        // def _limit_tendon(                                                                     <L 1547>
        // worldid, tenlimitedid = wp.tid()                                                       <L 1586>
        builtin_tid2d(var_0, var_1);
        // tenid = tendon_limited_adr[tenlimitedid]                                               <L 1587>
        var_2 = wp::address(var_tendon_limited_adr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // tendon_range_id = worldid % tendon_range.shape[0]                                      <L 1589>
        var_5 = &(var_tendon_range.shape);
        var_8 = wp::load(var_5);
        var_7 = wp::extract(var_8, var_6);
        var_9 = wp::mod(var_0, var_7);
        // tenrange = tendon_range[tendon_range_id, tenid]                                        <L 1590>
        var_10 = wp::address(var_tendon_range, var_9, var_3);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // length = ten_length_in[worldid, tenid]                                                 <L 1591>
        var_13 = wp::address(var_ten_length_in, var_0, var_3);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // dist_min, dist_max = length - tenrange[0], tenrange[1] - length                        <L 1592>
        var_17 = wp::extract(var_11, var_16);
        var_18 = wp::sub(var_14, var_17);
        var_20 = wp::extract(var_11, var_19);
        var_21 = wp::sub(var_20, var_14);
        // tendon_margin_id = worldid % tendon_margin.shape[0]                                    <L 1593>
        var_22 = &(var_tendon_margin.shape);
        var_25 = wp::load(var_22);
        var_24 = wp::extract(var_25, var_23);
        var_26 = wp::mod(var_0, var_24);
        // tenmargin = tendon_margin[tendon_margin_id, tenid]                                     <L 1594>
        var_27 = wp::address(var_tendon_margin, var_26, var_3);
        var_29 = wp::load(var_27);
        var_28 = wp::copy(var_29);
        // pos = wp.min(dist_min, dist_max) - tenmargin                                           <L 1595>
        var_30 = wp::min(var_18, var_21);
        var_31 = wp::sub(var_30, var_28);
        // active = pos < 0                                                                       <L 1596>
        var_33 = (var_31 < var_32);
        // if active:                                                                             <L 1598>
        if (var_33) {
            // wp.atomic_add(nl_out, worldid, 1)                                                  <L 1599>
            var_35 = wp::atomic_add(var_nl_out, var_0, var_34);
            // efcid = wp.atomic_add(nefc_out, worldid, 1)                                        <L 1600>
            var_37 = wp::atomic_add(var_nefc_out, var_0, var_36);
            // if efcid >= njmax_in:                                                              <L 1602>
            var_38 = (var_37 >= var_njmax_in);
            if (var_38) {
                // return                                                                         <L 1603>
                continue;
            }
            // Jqvel = float(0.0)                                                                 <L 1605>
            var_40 = wp::float(var_39);
            // scl = float(dist_min < dist_max) * 2.0 - 1.0                                       <L 1606>
            var_41 = (var_18 < var_21);
            var_42 = wp::float(var_41);
            var_44 = wp::mul(var_42, var_43);
            var_46 = wp::sub(var_44, var_45);
            // rownnz_tenJ = ten_J_rownnz[tenid]                                                  <L 1608>
            var_47 = wp::address(var_ten_J_rownnz, var_3);
            var_49 = wp::load(var_47);
            var_48 = wp::copy(var_49);
            // rowadr_tenJ = ten_J_rowadr[tenid]                                                  <L 1609>
            var_50 = wp::address(var_ten_J_rowadr, var_3);
            var_52 = wp::load(var_50);
            var_51 = wp::copy(var_52);
            // if is_sparse:                                                                      <L 1610>
            if (var_is_sparse) {
                // efc_J_rownnz_out[worldid, efcid] = rownnz_tenJ                                 <L 1611>
                wp::array_store(var_efc_J_rownnz_out, var_0, var_37, var_48);
                // rowadr_efc = wp.atomic_add(efc_nnz_out, worldid, rownnz_tenJ)                  <L 1612>
                var_53 = wp::atomic_add(var_efc_nnz_out, var_0, var_48);
                // if rowadr_efc + rownnz_tenJ > njmax_nnz_in:                                    <L 1613>
                var_54 = wp::add(var_53, var_48);
                var_55 = (var_54 > var_njmax_nnz_in);
                if (var_55) {
                    // return                                                                     <L 1614>
                    continue;
                }
                // efc_J_rowadr_out[worldid, efcid] = rowadr_efc                                  <L 1615>
                wp::array_store(var_efc_J_rowadr_out, var_0, var_37, var_53);
                // for i in range(rownnz_tenJ):                                                   <L 1617>
                var_56 = wp::range(var_48);
                start_for_2:;
                    if (iter_cmp(var_56) == 0) goto end_for_2;
                    var_57 = wp::iter_next(var_56);
                    // sparseid_ten = rowadr_tenJ + i                                             <L 1618>
                    var_58 = wp::add(var_51, var_57);
                    // sparseid_efc = rowadr_efc + i                                              <L 1619>
                    var_59 = wp::add(var_53, var_57);
                    // colind = ten_J_colind[sparseid_ten]                                        <L 1620>
                    var_60 = wp::address(var_ten_J_colind, var_58);
                    var_62 = wp::load(var_60);
                    var_61 = wp::copy(var_62);
                    // J = scl * ten_J_in[worldid, sparseid_ten]                                  <L 1621>
                    var_63 = wp::address(var_ten_J_in, var_0, var_58);
                    var_65 = wp::load(var_63);
                    var_64 = wp::mul(var_46, var_65);
                    // efc_J_colind_out[worldid, 0, sparseid_efc] = colind                        <L 1622>
                    wp::array_store(var_efc_J_colind_out, var_0, var_66, var_59, var_61);
                    // efc_J_out[worldid, 0, sparseid_efc] = J                                    <L 1623>
                    wp::array_store(var_efc_J_out, var_0, var_67, var_59, var_64);
                    // Jqvel += J * qvel_in[worldid, colind]                                      <L 1624>
                    var_68 = wp::address(var_qvel_in, var_0, var_61);
                    var_70 = wp::load(var_68);
                    var_69 = wp::mul(var_64, var_70);
                    var_71 = wp::add(var_40, var_69);
                    wp::assign(var_40, var_71);
                    goto start_for_2;
                end_for_2:;
            }
            if (!var_is_sparse) {
                // nnz = int(0)                                                                   <L 1626>
                var_73 = wp::int(var_72);
                // colind = ten_J_colind[rowadr_tenJ]                                             <L 1627>
                var_74 = wp::address(var_ten_J_colind, var_51);
                var_76 = wp::load(var_74);
                var_75 = wp::copy(var_76);
                // for i in range(nv):                                                            <L 1628>
                var_77 = wp::range(var_nv);
                start_for_4:;
                    if (iter_cmp(var_77) == 0) goto end_for_4;
                    var_78 = wp::iter_next(var_77);
                    // if nnz < rownnz_tenJ and i == colind:                                      <L 1629>
                    var_79 = (var_73 < var_48);
                    var_80 = (var_78 == var_75);
                    var_81 = var_79 && var_80;
                    if (var_81) {
                        // J = scl * ten_J_in[worldid, rowadr_tenJ + nnz]                         <L 1630>
                        var_82 = wp::add(var_51, var_73);
                        var_83 = wp::address(var_ten_J_in, var_0, var_82);
                        var_85 = wp::load(var_83);
                        var_84 = wp::mul(var_46, var_85);
                        // efc_J_out[worldid, efcid, i] = J                                       <L 1631>
                        wp::array_store(var_efc_J_out, var_0, var_37, var_78, var_84);
                        // Jqvel += J * qvel_in[worldid, i]                                       <L 1632>
                        var_86 = wp::address(var_qvel_in, var_0, var_78);
                        var_88 = wp::load(var_86);
                        var_87 = wp::mul(var_84, var_88);
                        var_89 = wp::add(var_40, var_87);
                        // nnz += 1                                                               <L 1633>
                        var_91 = wp::add(var_73, var_90);
                        // if nnz < rownnz_tenJ:                                                  <L 1634>
                        var_92 = (var_91 < var_48);
                        if (var_92) {
                            // colind = ten_J_colind[rowadr_tenJ + nnz]                           <L 1635>
                            var_93 = wp::add(var_51, var_91);
                            var_94 = wp::address(var_ten_J_colind, var_93);
                            var_96 = wp::load(var_94);
                            var_95 = wp::copy(var_96);
                        }
                        var_97 = wp::where(var_92, var_95, var_75);
                    }
                    var_98 = wp::where(var_81, var_89, var_40);
                    var_99 = wp::where(var_81, var_97, var_75);
                    var_100 = wp::where(var_81, var_84, var_64);
                    var_101 = wp::where(var_81, var_91, var_73);
                    if (!var_81) {
                        // efc_J_out[worldid, efcid, i] = 0.0                                     <L 1637>
                        wp::array_store(var_efc_J_out, var_0, var_37, var_78, var_102);
                    }
                    wp::assign(var_40, var_98);
                    wp::assign(var_75, var_99);
                    wp::assign(var_64, var_100);
                    wp::assign(var_73, var_101);
                    goto start_for_4;
                end_for_4:;
            }
            var_103 = wp::where(var_is_sparse, var_57, var_78);
            var_104 = wp::where(var_is_sparse, var_61, var_75);
            // tendon_invweight0_id = worldid % tendon_invweight0.shape[0]                        <L 1639>
            var_105 = &(var_tendon_invweight0.shape);
            var_108 = wp::load(var_105);
            var_107 = wp::extract(var_108, var_106);
            var_109 = wp::mod(var_0, var_107);
            // tendon_solref_lim_id = worldid % tendon_solref_lim.shape[0]                        <L 1640>
            var_110 = &(var_tendon_solref_lim.shape);
            var_113 = wp::load(var_110);
            var_112 = wp::extract(var_113, var_111);
            var_114 = wp::mod(var_0, var_112);
            // tendon_solimp_lim_id = worldid % tendon_solimp_lim.shape[0]                        <L 1641>
            var_115 = &(var_tendon_solimp_lim.shape);
            var_118 = wp::load(var_115);
            var_117 = wp::extract(var_118, var_116);
            var_119 = wp::mod(var_0, var_117);
            // _efc_row(                                                                          <L 1642>
            // opt_disableflags,                                                                  <L 1643>
            // worldid,                                                                           <L 1644>
            // opt_timestep[worldid % opt_timestep.shape[0]],                                     <L 1645>
            var_120 = &(var_opt_timestep.shape);
            var_123 = wp::load(var_120);
            var_122 = wp::extract(var_123, var_121);
            var_124 = wp::mod(var_0, var_122);
            var_125 = wp::address(var_opt_timestep, var_124);
            // efcid,                                                                             <L 1646>
            // pos,                                                                               <L 1647>
            // pos,                                                                               <L 1648>
            // tendon_invweight0[tendon_invweight0_id, tenid],                                    <L 1649>
            var_126 = wp::address(var_tendon_invweight0, var_109, var_3);
            // tendon_solref_lim[tendon_solref_lim_id, tenid],                                    <L 1650>
            var_127 = wp::address(var_tendon_solref_lim, var_114, var_3);
            // tendon_solimp_lim[tendon_solimp_lim_id, tenid],                                    <L 1651>
            var_128 = wp::address(var_tendon_solimp_lim, var_119, var_3);
            // tenmargin,                                                                         <L 1652>
            // Jqvel,                                                                             <L 1653>
            // 0.0,                                                                               <L 1654>
            // ConstraintType.LIMIT_TENDON,                                                       <L 1655>
            // tenid,                                                                             <L 1656>
            // efc_type_out,                                                                      <L 1657>
            // efc_id_out,                                                                        <L 1658>
            // efc_pos_out,                                                                       <L 1659>
            // efc_margin_out,                                                                    <L 1660>
            // efc_D_out,                                                                         <L 1661>
            // efc_vel_out,                                                                       <L 1662>
            // efc_aref_out,                                                                      <L 1663>
            // efc_frictionloss_out,                                                              <L 1664>
            var_132 = wp::load(var_125);
            var_133 = wp::load(var_126);
            var_134 = wp::load(var_127);
            var_135 = wp::load(var_128);
            _efc_row_0(var_opt_disableflags, var_0, var_132, var_37, var_31, var_31, var_133, var_134, var_135, var_28, var_40, var_129, var_131, var_3, var_efc_type_out, var_efc_id_out, var_efc_pos_out, var_efc_margin_out, var_efc_D_out, var_efc_vel_out, var_efc_aref_out, var_efc_frictionloss_out);
        }
    }
}



extern "C" __global__ void _equality_joint_5d33c76e_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::int32 var_opt_disableflags,
    wp::array_t<wp::float32> var_qpos0,
    wp::array_t<wp::int32> var_jnt_qposadr,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::float32> var_dof_invweight0,
    wp::array_t<wp::int32> var_eq_obj1id,
    wp::array_t<wp::int32> var_eq_obj2id,
    wp::array_t<wp::vec_t<2, wp::float32>> var_eq_solref,
    wp::array_t<wp::vec_t<5, wp::float32>> var_eq_solimp,
    wp::array_t<wp::vec_t<11, wp::float32>> var_eq_data,
    bool var_is_sparse,
    wp::array_t<wp::int32> var_eq_jnt_adr,
    wp::array_t<wp::float32> var_qpos_in,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<bool> var_eq_active_in,
    wp::int32 var_njmax_in,
    wp::int32 var_njmax_nnz_in,
    wp::array_t<wp::int32> var_ne_out,
    wp::array_t<wp::int32> var_nefc_out,
    wp::array_t<wp::int32> var_efc_type_out,
    wp::array_t<wp::int32> var_efc_id_out,
    wp::array_t<wp::int32> var_efc_J_rownnz_out,
    wp::array_t<wp::int32> var_efc_J_rowadr_out,
    wp::array_t<wp::int32> var_efc_J_colind_out,
    wp::array_t<wp::float32> var_efc_J_out,
    wp::array_t<wp::float32> var_efc_pos_out,
    wp::array_t<wp::float32> var_efc_margin_out,
    wp::array_t<wp::float32> var_efc_D_out,
    wp::array_t<wp::float32> var_efc_vel_out,
    wp::array_t<wp::float32> var_efc_aref_out,
    wp::array_t<wp::float32> var_efc_frictionloss_out,
    wp::array_t<wp::int32> var_efc_nnz_out)
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
        bool* var_5;
        bool var_6;
        bool var_7;
        const wp::int32 var_8 = 1;
        wp::int32 var_9;
        const wp::int32 var_10 = 1;
        wp::int32 var_11;
        bool var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::shape_t* var_19;
        const wp::int32 var_20 = 0;
        wp::int32 var_21;
        wp::shape_t var_22;
        wp::int32 var_23;
        wp::vec_t<11, wp::float32>* var_24;
        wp::vec_t<11, wp::float32> var_25;
        wp::vec_t<11, wp::float32> var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        wp::shape_t* var_33;
        const wp::int32 var_34 = 0;
        wp::int32 var_35;
        wp::shape_t var_36;
        wp::int32 var_37;
        wp::shape_t* var_38;
        const wp::int32 var_39 = 0;
        wp::int32 var_40;
        wp::shape_t var_41;
        wp::int32 var_42;
        const wp::int32 var_43 = 1;
        const wp::int32 var_44 = -1;
        bool var_45;
        const wp::int32 var_46 = 2;
        const wp::int32 var_47 = 1;
        wp::int32 var_48;
        wp::int32 var_49;
        wp::int32 var_50;
        bool var_51;
        const wp::int32 var_52 = 0;
        const wp::float32 var_53 = 1.0;
        const wp::int32 var_54 = 0;
        wp::range_t var_55;
        wp::int32 var_56;
        const wp::float32 var_57 = 0.0;
        const wp::float32 var_58 = 1.0;
        const wp::int32 var_59 = 1;
        const wp::int32 var_60 = -1;
        bool var_61;
        wp::int32* var_62;
        wp::int32 var_63;
        wp::int32 var_64;
        wp::int32* var_65;
        wp::int32 var_66;
        wp::int32 var_67;
        wp::float32* var_68;
        wp::float32* var_69;
        wp::float32 var_70;
        wp::float32 var_71;
        wp::float32 var_72;
        const wp::int32 var_73 = 0;
        wp::float32 var_74;
        const wp::int32 var_75 = 1;
        wp::float32 var_76;
        const wp::int32 var_77 = 2;
        wp::float32 var_78;
        const wp::int32 var_79 = 3;
        wp::float32 var_80;
        const wp::int32 var_81 = 4;
        wp::float32 var_82;
        wp::float32 var_83;
        wp::float32 var_84;
        wp::float32 var_85;
        wp::float32 var_86;
        wp::float32 var_87;
        wp::float32 var_88;
        wp::float32 var_89;
        wp::float32 var_90;
        const wp::int32 var_91 = 1;
        wp::float32 var_92;
        const wp::float32 var_93 = 2.0;
        const wp::int32 var_94 = 2;
        wp::float32 var_95;
        wp::float32 var_96;
        const wp::float32 var_97 = 3.0;
        const wp::int32 var_98 = 3;
        wp::float32 var_99;
        wp::float32 var_100;
        const wp::float32 var_101 = 4.0;
        wp::float32 var_102;
        const wp::int32 var_103 = 4;
        wp::float32 var_104;
        wp::float32 var_105;
        wp::float32 var_106;
        wp::float32 var_107;
        wp::float32 var_108;
        wp::float32 var_109;
        wp::float32 var_110;
        wp::float32* var_111;
        wp::float32* var_112;
        wp::float32 var_113;
        wp::float32 var_114;
        wp::float32 var_115;
        wp::float32 var_116;
        wp::float32* var_117;
        wp::float32* var_118;
        wp::float32 var_119;
        wp::float32 var_120;
        wp::float32 var_121;
        wp::float32 var_122;
        wp::float32* var_123;
        wp::float32* var_124;
        wp::float32 var_125;
        wp::float32 var_126;
        wp::float32 var_127;
        const wp::int32 var_128 = 1;
        wp::int32 var_129;
        const wp::int32 var_130 = 0;
        wp::float32 var_131;
        const wp::int32 var_132 = 0;
        wp::float32 var_133;
        wp::float32* var_134;
        wp::float32* var_135;
        wp::float32 var_136;
        wp::float32 var_137;
        wp::float32 var_138;
        const wp::int32 var_139 = 0;
        wp::float32 var_140;
        wp::float32 var_141;
        wp::float32* var_142;
        wp::float32 var_143;
        wp::float32 var_144;
        wp::float32* var_145;
        wp::float32 var_146;
        wp::float32 var_147;
        wp::float32 var_148;
        wp::float32 var_149;
        wp::float32 var_150;
        wp::shape_t* var_151;
        const wp::int32 var_152 = 0;
        wp::int32 var_153;
        wp::shape_t var_154;
        wp::int32 var_155;
        wp::float32* var_156;
        wp::shape_t* var_157;
        const wp::int32 var_158 = 0;
        wp::int32 var_159;
        wp::shape_t var_160;
        wp::int32 var_161;
        wp::vec_t<2, wp::float32>* var_162;
        wp::shape_t* var_163;
        const wp::int32 var_164 = 0;
        wp::int32 var_165;
        wp::shape_t var_166;
        wp::int32 var_167;
        wp::vec_t<5, wp::float32>* var_168;
        const wp::float32 var_169 = 0.0;
        const wp::float32 var_170 = 0.0;
        const wp::int32 var_171 = 0;
        const wp::int32 var_172 = 0;
        wp::float32 var_173;
        wp::vec_t<2, wp::float32> var_174;
        wp::vec_t<5, wp::float32> var_175;
        //---------
        // forward
        // def _equality_joint(                                                                   <L 368>
        // worldid, eqjntid = wp.tid()                                                            <L 408>
        builtin_tid2d(var_0, var_1);
        // eqid = eq_jnt_adr[eqjntid]                                                             <L 409>
        var_2 = wp::address(var_eq_jnt_adr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if not eq_active_in[worldid, eqid]:                                                    <L 411>
        var_5 = wp::address(var_eq_active_in, var_0, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::unot(var_7);
        if (var_6) {
            // return                                                                             <L 412>
            continue;
        }
        // wp.atomic_add(ne_out, worldid, 1)                                                      <L 414>
        var_9 = wp::atomic_add(var_ne_out, var_0, var_8);
        // efcid = wp.atomic_add(nefc_out, worldid, 1)                                            <L 415>
        var_11 = wp::atomic_add(var_nefc_out, var_0, var_10);
        // if efcid >= njmax_in:                                                                  <L 417>
        var_12 = (var_11 >= var_njmax_in);
        if (var_12) {
            // return                                                                             <L 418>
            continue;
        }
        // jntid_1 = eq_obj1id[eqid]                                                              <L 420>
        var_13 = wp::address(var_eq_obj1id, var_3);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // jntid_2 = eq_obj2id[eqid]                                                              <L 421>
        var_16 = wp::address(var_eq_obj2id, var_3);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // data = eq_data[worldid % eq_data.shape[0], eqid]                                       <L 422>
        var_19 = &(var_eq_data.shape);
        var_22 = wp::load(var_19);
        var_21 = wp::extract(var_22, var_20);
        var_23 = wp::mod(var_0, var_21);
        var_24 = wp::address(var_eq_data, var_23, var_3);
        var_26 = wp::load(var_24);
        var_25 = wp::copy(var_26);
        // dofadr1 = jnt_dofadr[jntid_1]                                                          <L 423>
        var_27 = wp::address(var_jnt_dofadr, var_14);
        var_29 = wp::load(var_27);
        var_28 = wp::copy(var_29);
        // qposadr1 = jnt_qposadr[jntid_1]                                                        <L 424>
        var_30 = wp::address(var_jnt_qposadr, var_14);
        var_32 = wp::load(var_30);
        var_31 = wp::copy(var_32);
        // qpos0_id = worldid % qpos0.shape[0]                                                    <L 425>
        var_33 = &(var_qpos0.shape);
        var_36 = wp::load(var_33);
        var_35 = wp::extract(var_36, var_34);
        var_37 = wp::mod(var_0, var_35);
        // dof_invweight0_id = worldid % dof_invweight0.shape[0]                                  <L 426>
        var_38 = &(var_dof_invweight0.shape);
        var_41 = wp::load(var_38);
        var_40 = wp::extract(var_41, var_39);
        var_42 = wp::mod(var_0, var_40);
        // if is_sparse:                                                                          <L 428>
        if (var_is_sparse) {
            // if jntid_2 > -1:                                                                   <L 429>
            var_45 = (var_17 > var_44);
            if (var_45) {
                // rownnz = 2                                                                     <L 430>
            }
            if (!var_45) {
                // rownnz = 1                                                                     <L 432>
            }
            var_48 = wp::where(var_45, var_46, var_47);
            // efc_J_rownnz_out[worldid, efcid] = rownnz                                          <L 433>
            wp::array_store(var_efc_J_rownnz_out, var_0, var_11, var_48);
            // rowadr = wp.atomic_add(efc_nnz_out, worldid, rownnz)                               <L 434>
            var_49 = wp::atomic_add(var_efc_nnz_out, var_0, var_48);
            // if rowadr + rownnz > njmax_nnz_in:                                                 <L 435>
            var_50 = wp::add(var_49, var_48);
            var_51 = (var_50 > var_njmax_nnz_in);
            if (var_51) {
                // return                                                                         <L 436>
                continue;
            }
            // efc_J_rowadr_out[worldid, efcid] = rowadr                                          <L 437>
            wp::array_store(var_efc_J_rowadr_out, var_0, var_11, var_49);
            // efc_J_colind_out[worldid, 0, rowadr] = dofadr1                                     <L 438>
            wp::array_store(var_efc_J_colind_out, var_0, var_52, var_49, var_28);
            // efc_J_out[worldid, 0, rowadr] = 1.0                                                <L 439>
            wp::array_store(var_efc_J_out, var_0, var_54, var_49, var_53);
        }
        if (!var_is_sparse) {
            // for i in range(nv):                                                                <L 441>
            var_55 = wp::range(var_nv);
            start_for_3:;
                if (iter_cmp(var_55) == 0) goto end_for_3;
                var_56 = wp::iter_next(var_55);
                // efc_J_out[worldid, efcid, i] = 0.0                                             <L 442>
                wp::array_store(var_efc_J_out, var_0, var_11, var_56, var_57);
                goto start_for_3;
            end_for_3:;
            // efc_J_out[worldid, efcid, dofadr1] = 1.0                                           <L 443>
            wp::array_store(var_efc_J_out, var_0, var_11, var_28, var_58);
        }
        // if jntid_2 > -1:                                                                       <L 445>
        var_61 = (var_17 > var_60);
        if (var_61) {
            // qposadr2 = jnt_qposadr[jntid_2]                                                    <L 447>
            var_62 = wp::address(var_jnt_qposadr, var_17);
            var_64 = wp::load(var_62);
            var_63 = wp::copy(var_64);
            // dofadr2 = jnt_dofadr[jntid_2]                                                      <L 448>
            var_65 = wp::address(var_jnt_dofadr, var_17);
            var_67 = wp::load(var_65);
            var_66 = wp::copy(var_67);
            // dif = qpos_in[worldid, qposadr2] - qpos0[qpos0_id, qposadr2]                       <L 449>
            var_68 = wp::address(var_qpos_in, var_0, var_63);
            var_69 = wp::address(var_qpos0, var_37, var_63);
            var_71 = wp::load(var_68);
            var_72 = wp::load(var_69);
            var_70 = wp::sub(var_71, var_72);
            // rhs = data[0] + dif * (data[1] + dif * (data[2] + dif * (data[3] + dif * data[4])))       <L 452>
            var_74 = wp::extract(var_25, var_73);
            var_76 = wp::extract(var_25, var_75);
            var_78 = wp::extract(var_25, var_77);
            var_80 = wp::extract(var_25, var_79);
            var_82 = wp::extract(var_25, var_81);
            var_83 = wp::mul(var_70, var_82);
            var_84 = wp::add(var_80, var_83);
            var_85 = wp::mul(var_70, var_84);
            var_86 = wp::add(var_78, var_85);
            var_87 = wp::mul(var_70, var_86);
            var_88 = wp::add(var_76, var_87);
            var_89 = wp::mul(var_70, var_88);
            var_90 = wp::add(var_74, var_89);
            // deriv_2 = data[1] + dif * (2.0 * data[2] + dif * (3.0 * data[3] + dif * 4.0 * data[4]))       <L 453>
            var_92 = wp::extract(var_25, var_91);
            var_95 = wp::extract(var_25, var_94);
            var_96 = wp::mul(var_93, var_95);
            var_99 = wp::extract(var_25, var_98);
            var_100 = wp::mul(var_97, var_99);
            var_102 = wp::mul(var_70, var_101);
            var_104 = wp::extract(var_25, var_103);
            var_105 = wp::mul(var_102, var_104);
            var_106 = wp::add(var_100, var_105);
            var_107 = wp::mul(var_70, var_106);
            var_108 = wp::add(var_96, var_107);
            var_109 = wp::mul(var_70, var_108);
            var_110 = wp::add(var_92, var_109);
            // pos = qpos_in[worldid, qposadr1] - qpos0[qpos0_id, qposadr1] - rhs                 <L 455>
            var_111 = wp::address(var_qpos_in, var_0, var_31);
            var_112 = wp::address(var_qpos0, var_37, var_31);
            var_114 = wp::load(var_111);
            var_115 = wp::load(var_112);
            var_113 = wp::sub(var_114, var_115);
            var_116 = wp::sub(var_113, var_90);
            // Jqvel = qvel_in[worldid, dofadr1] - qvel_in[worldid, dofadr2] * deriv_2            <L 456>
            var_117 = wp::address(var_qvel_in, var_0, var_28);
            var_118 = wp::address(var_qvel_in, var_0, var_66);
            var_120 = wp::load(var_118);
            var_119 = wp::mul(var_120, var_110);
            var_122 = wp::load(var_117);
            var_121 = wp::sub(var_122, var_119);
            // invweight = dof_invweight0[dof_invweight0_id, dofadr1] + dof_invweight0[dof_invweight0_id, dofadr2]       <L 457>
            var_123 = wp::address(var_dof_invweight0, var_42, var_28);
            var_124 = wp::address(var_dof_invweight0, var_42, var_66);
            var_126 = wp::load(var_123);
            var_127 = wp::load(var_124);
            var_125 = wp::add(var_126, var_127);
            // if is_sparse:                                                                      <L 459>
            if (var_is_sparse) {
                // sparseid = rowadr + 1                                                          <L 460>
                var_129 = wp::add(var_49, var_128);
                // efc_J_colind_out[worldid, 0, sparseid] = dofadr2                               <L 461>
                wp::array_store(var_efc_J_colind_out, var_0, var_130, var_129, var_66);
                // efc_J_out[worldid, 0, sparseid] = -deriv_2                                     <L 462>
                var_131 = wp::neg(var_110);
                wp::array_store(var_efc_J_out, var_0, var_132, var_129, var_131);
            }
            if (!var_is_sparse) {
                // efc_J_out[worldid, efcid, dofadr2] = -deriv_2                                  <L 464>
                var_133 = wp::neg(var_110);
                wp::array_store(var_efc_J_out, var_0, var_11, var_66, var_133);
            }
        }
        if (!var_61) {
            // pos = qpos_in[worldid, qposadr1] - qpos0[qpos0_id, qposadr1] - data[0]             <L 467>
            var_134 = wp::address(var_qpos_in, var_0, var_31);
            var_135 = wp::address(var_qpos0, var_37, var_31);
            var_137 = wp::load(var_134);
            var_138 = wp::load(var_135);
            var_136 = wp::sub(var_137, var_138);
            var_140 = wp::extract(var_25, var_139);
            var_141 = wp::sub(var_136, var_140);
            // Jqvel = qvel_in[worldid, dofadr1]                                                  <L 468>
            var_142 = wp::address(var_qvel_in, var_0, var_28);
            var_144 = wp::load(var_142);
            var_143 = wp::copy(var_144);
            // invweight = dof_invweight0[dof_invweight0_id, dofadr1]                             <L 469>
            var_145 = wp::address(var_dof_invweight0, var_42, var_28);
            var_147 = wp::load(var_145);
            var_146 = wp::copy(var_147);
        }
        var_148 = wp::where(var_61, var_116, var_141);
        var_149 = wp::where(var_61, var_121, var_143);
        var_150 = wp::where(var_61, var_125, var_146);
        // _efc_row(                                                                              <L 472>
        // opt_disableflags,                                                                      <L 473>
        // worldid,                                                                               <L 474>
        // opt_timestep[worldid % opt_timestep.shape[0]],                                         <L 475>
        var_151 = &(var_opt_timestep.shape);
        var_154 = wp::load(var_151);
        var_153 = wp::extract(var_154, var_152);
        var_155 = wp::mod(var_0, var_153);
        var_156 = wp::address(var_opt_timestep, var_155);
        // efcid,                                                                                 <L 476>
        // pos,                                                                                   <L 477>
        // pos,                                                                                   <L 478>
        // invweight,                                                                             <L 479>
        // eq_solref[worldid % eq_solref.shape[0], eqid],                                         <L 480>
        var_157 = &(var_eq_solref.shape);
        var_160 = wp::load(var_157);
        var_159 = wp::extract(var_160, var_158);
        var_161 = wp::mod(var_0, var_159);
        var_162 = wp::address(var_eq_solref, var_161, var_3);
        // eq_solimp[worldid % eq_solimp.shape[0], eqid],                                         <L 481>
        var_163 = &(var_eq_solimp.shape);
        var_166 = wp::load(var_163);
        var_165 = wp::extract(var_166, var_164);
        var_167 = wp::mod(var_0, var_165);
        var_168 = wp::address(var_eq_solimp, var_167, var_3);
        // 0.0,                                                                                   <L 482>
        // Jqvel,                                                                                 <L 483>
        // 0.0,                                                                                   <L 484>
        // ConstraintType.EQUALITY,                                                               <L 485>
        // eqid,                                                                                  <L 486>
        // efc_type_out,                                                                          <L 487>
        // efc_id_out,                                                                            <L 488>
        // efc_pos_out,                                                                           <L 489>
        // efc_margin_out,                                                                        <L 490>
        // efc_D_out,                                                                             <L 491>
        // efc_vel_out,                                                                           <L 492>
        // efc_aref_out,                                                                          <L 493>
        // efc_frictionloss_out,                                                                  <L 494>
        var_173 = wp::load(var_156);
        var_174 = wp::load(var_162);
        var_175 = wp::load(var_168);
        _efc_row_0(var_opt_disableflags, var_0, var_173, var_11, var_148, var_148, var_150, var_174, var_175, var_169, var_149, var_170, var_172, var_3, var_efc_type_out, var_efc_id_out, var_efc_pos_out, var_efc_margin_out, var_efc_D_out, var_efc_vel_out, var_efc_aref_out, var_efc_frictionloss_out);
    }
}



extern "C" __global__ void _friction_dof_a00ce1f7_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::int32 var_opt_disableflags,
    wp::array_t<wp::vec_t<2, wp::float32>> var_dof_solref,
    wp::array_t<wp::vec_t<5, wp::float32>> var_dof_solimp,
    wp::array_t<wp::float32> var_dof_frictionloss,
    wp::array_t<wp::float32> var_dof_invweight0,
    bool var_is_sparse,
    wp::array_t<wp::float32> var_qvel_in,
    wp::int32 var_njmax_in,
    wp::int32 var_njmax_nnz_in,
    wp::array_t<wp::int32> var_nf_out,
    wp::array_t<wp::int32> var_nefc_out,
    wp::array_t<wp::int32> var_efc_type_out,
    wp::array_t<wp::int32> var_efc_id_out,
    wp::array_t<wp::int32> var_efc_J_rownnz_out,
    wp::array_t<wp::int32> var_efc_J_rowadr_out,
    wp::array_t<wp::int32> var_efc_J_colind_out,
    wp::array_t<wp::float32> var_efc_J_out,
    wp::array_t<wp::float32> var_efc_pos_out,
    wp::array_t<wp::float32> var_efc_margin_out,
    wp::array_t<wp::float32> var_efc_D_out,
    wp::array_t<wp::float32> var_efc_vel_out,
    wp::array_t<wp::float32> var_efc_aref_out,
    wp::array_t<wp::float32> var_efc_frictionloss_out,
    wp::array_t<wp::int32> var_efc_nnz_out)
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
        wp::float32* var_7;
        const wp::float32 var_8 = 0.0;
        bool var_9;
        wp::float32 var_10;
        const wp::int32 var_11 = 1;
        wp::int32 var_12;
        const wp::int32 var_13 = 1;
        wp::int32 var_14;
        bool var_15;
        const wp::int32 var_16 = 1;
        const wp::int32 var_17 = 1;
        wp::int32 var_18;
        const wp::int32 var_19 = 1;
        wp::int32 var_20;
        bool var_21;
        const wp::int32 var_22 = 0;
        const wp::float32 var_23 = 1.0;
        const wp::int32 var_24 = 0;
        wp::range_t var_25;
        wp::int32 var_26;
        const wp::float32 var_27 = 0.0;
        const wp::float32 var_28 = 1.0;
        wp::float32* var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        wp::shape_t* var_32;
        const wp::int32 var_33 = 0;
        wp::int32 var_34;
        wp::shape_t var_35;
        wp::int32 var_36;
        wp::shape_t* var_37;
        const wp::int32 var_38 = 0;
        wp::int32 var_39;
        wp::shape_t var_40;
        wp::int32 var_41;
        wp::shape_t* var_42;
        const wp::int32 var_43 = 0;
        wp::int32 var_44;
        wp::shape_t var_45;
        wp::int32 var_46;
        wp::shape_t* var_47;
        const wp::int32 var_48 = 0;
        wp::int32 var_49;
        wp::shape_t var_50;
        wp::int32 var_51;
        wp::float32* var_52;
        const wp::float32 var_53 = 0.0;
        const wp::float32 var_54 = 0.0;
        wp::float32* var_55;
        wp::vec_t<2, wp::float32>* var_56;
        wp::vec_t<5, wp::float32>* var_57;
        const wp::float32 var_58 = 0.0;
        wp::float32* var_59;
        const wp::int32 var_60 = 1;
        const wp::int32 var_61 = 1;
        wp::float32 var_62;
        wp::float32 var_63;
        wp::vec_t<2, wp::float32> var_64;
        wp::vec_t<5, wp::float32> var_65;
        wp::float32 var_66;
        //---------
        // forward
        // def _friction_dof(                                                                     <L 1114>
        // worldid, dofid = wp.tid()                                                              <L 1146>
        builtin_tid2d(var_0, var_1);
        // dof_frictionloss_id = worldid % dof_frictionloss.shape[0]                              <L 1148>
        var_2 = &(var_dof_frictionloss.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // if dof_frictionloss[dof_frictionloss_id, dofid] <= 0.0:                                <L 1150>
        var_7 = wp::address(var_dof_frictionloss, var_6, var_1);
        var_10 = wp::load(var_7);
        var_9 = (var_10 <= var_8);
        if (var_9) {
            // return                                                                             <L 1151>
            continue;
        }
        // wp.atomic_add(nf_out, worldid, 1)                                                      <L 1153>
        var_12 = wp::atomic_add(var_nf_out, var_0, var_11);
        // efcid = wp.atomic_add(nefc_out, worldid, 1)                                            <L 1154>
        var_14 = wp::atomic_add(var_nefc_out, var_0, var_13);
        // if efcid >= njmax_in:                                                                  <L 1156>
        var_15 = (var_14 >= var_njmax_in);
        if (var_15) {
            // return                                                                             <L 1157>
            continue;
        }
        // if is_sparse:                                                                          <L 1159>
        if (var_is_sparse) {
            // efc_J_rownnz_out[worldid, efcid] = 1                                               <L 1160>
            wp::array_store(var_efc_J_rownnz_out, var_0, var_14, var_16);
            // rowadr = wp.atomic_add(efc_nnz_out, worldid, 1)                                    <L 1161>
            var_18 = wp::atomic_add(var_efc_nnz_out, var_0, var_17);
            // if rowadr + 1 > njmax_nnz_in:                                                      <L 1162>
            var_20 = wp::add(var_18, var_19);
            var_21 = (var_20 > var_njmax_nnz_in);
            if (var_21) {
                // return                                                                         <L 1163>
                continue;
            }
            // efc_J_rowadr_out[worldid, efcid] = rowadr                                          <L 1164>
            wp::array_store(var_efc_J_rowadr_out, var_0, var_14, var_18);
            // efc_J_colind_out[worldid, 0, rowadr] = dofid                                       <L 1165>
            wp::array_store(var_efc_J_colind_out, var_0, var_22, var_18, var_1);
            // efc_J_out[worldid, 0, rowadr] = 1.0                                                <L 1166>
            wp::array_store(var_efc_J_out, var_0, var_24, var_18, var_23);
        }
        if (!var_is_sparse) {
            // for i in range(nv):                                                                <L 1168>
            var_25 = wp::range(var_nv);
            start_for_3:;
                if (iter_cmp(var_25) == 0) goto end_for_3;
                var_26 = wp::iter_next(var_25);
                // efc_J_out[worldid, efcid, i] = 0.0                                             <L 1169>
                wp::array_store(var_efc_J_out, var_0, var_14, var_26, var_27);
                goto start_for_3;
            end_for_3:;
            // efc_J_out[worldid, efcid, dofid] = 1.0                                             <L 1170>
            wp::array_store(var_efc_J_out, var_0, var_14, var_1, var_28);
        }
        // Jqvel = qvel_in[worldid, dofid]                                                        <L 1172>
        var_29 = wp::address(var_qvel_in, var_0, var_1);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // dof_invweight0_id = worldid % dof_invweight0.shape[0]                                  <L 1174>
        var_32 = &(var_dof_invweight0.shape);
        var_35 = wp::load(var_32);
        var_34 = wp::extract(var_35, var_33);
        var_36 = wp::mod(var_0, var_34);
        // dof_solref_id = worldid % dof_solref.shape[0]                                          <L 1175>
        var_37 = &(var_dof_solref.shape);
        var_40 = wp::load(var_37);
        var_39 = wp::extract(var_40, var_38);
        var_41 = wp::mod(var_0, var_39);
        // dof_solimp_id = worldid % dof_solimp.shape[0]                                          <L 1176>
        var_42 = &(var_dof_solimp.shape);
        var_45 = wp::load(var_42);
        var_44 = wp::extract(var_45, var_43);
        var_46 = wp::mod(var_0, var_44);
        // _efc_row(                                                                              <L 1177>
        // opt_disableflags,                                                                      <L 1178>
        // worldid,                                                                               <L 1179>
        // opt_timestep[worldid % opt_timestep.shape[0]],                                         <L 1180>
        var_47 = &(var_opt_timestep.shape);
        var_50 = wp::load(var_47);
        var_49 = wp::extract(var_50, var_48);
        var_51 = wp::mod(var_0, var_49);
        var_52 = wp::address(var_opt_timestep, var_51);
        // efcid,                                                                                 <L 1181>
        // 0.0,                                                                                   <L 1182>
        // 0.0,                                                                                   <L 1183>
        // dof_invweight0[dof_invweight0_id, dofid],                                              <L 1184>
        var_55 = wp::address(var_dof_invweight0, var_36, var_1);
        // dof_solref[dof_solref_id, dofid],                                                      <L 1185>
        var_56 = wp::address(var_dof_solref, var_41, var_1);
        // dof_solimp[dof_solimp_id, dofid],                                                      <L 1186>
        var_57 = wp::address(var_dof_solimp, var_46, var_1);
        // 0.0,                                                                                   <L 1187>
        // Jqvel,                                                                                 <L 1188>
        // dof_frictionloss[dof_frictionloss_id, dofid],                                          <L 1189>
        var_59 = wp::address(var_dof_frictionloss, var_6, var_1);
        // ConstraintType.FRICTION_DOF,                                                           <L 1190>
        // dofid,                                                                                 <L 1191>
        // efc_type_out,                                                                          <L 1192>
        // efc_id_out,                                                                            <L 1193>
        // efc_pos_out,                                                                           <L 1194>
        // efc_margin_out,                                                                        <L 1195>
        // efc_D_out,                                                                             <L 1196>
        // efc_vel_out,                                                                           <L 1197>
        // efc_aref_out,                                                                          <L 1198>
        // efc_frictionloss_out,                                                                  <L 1199>
        var_62 = wp::load(var_52);
        var_63 = wp::load(var_55);
        var_64 = wp::load(var_56);
        var_65 = wp::load(var_57);
        var_66 = wp::load(var_59);
        _efc_row_0(var_opt_disableflags, var_0, var_62, var_14, var_53, var_54, var_63, var_64, var_65, var_58, var_30, var_66, var_61, var_1, var_efc_type_out, var_efc_id_out, var_efc_pos_out, var_efc_margin_out, var_efc_D_out, var_efc_vel_out, var_efc_aref_out, var_efc_frictionloss_out);
    }
}



extern "C" __global__ void _equality_weld_86afa96d_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_nv,
    wp::int32 var_nsite,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::int32 var_opt_disableflags,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_weldid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::vec_t<2, wp::float32>> var_body_invweight0,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::int32> var_dof_parentid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> var_site_quat,
    wp::array_t<wp::int32> var_eq_obj1id,
    wp::array_t<wp::int32> var_eq_obj2id,
    wp::array_t<wp::int32> var_eq_objtype,
    wp::array_t<wp::vec_t<2, wp::float32>> var_eq_solref,
    wp::array_t<wp::vec_t<5, wp::float32>> var_eq_solimp,
    wp::array_t<wp::vec_t<11, wp::float32>> var_eq_data,
    bool var_is_sparse,
    wp::array_t<wp::int32> var_eq_wld_adr,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<bool> var_eq_active_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::int32 var_njmax_in,
    wp::int32 var_njmax_nnz_in,
    wp::array_t<wp::int32> var_ne_out,
    wp::array_t<wp::int32> var_nefc_out,
    wp::array_t<wp::int32> var_efc_type_out,
    wp::array_t<wp::int32> var_efc_id_out,
    wp::array_t<wp::int32> var_efc_J_rownnz_out,
    wp::array_t<wp::int32> var_efc_J_rowadr_out,
    wp::array_t<wp::int32> var_efc_J_colind_out,
    wp::array_t<wp::float32> var_efc_J_out,
    wp::array_t<wp::float32> var_efc_pos_out,
    wp::array_t<wp::float32> var_efc_margin_out,
    wp::array_t<wp::float32> var_efc_D_out,
    wp::array_t<wp::float32> var_efc_vel_out,
    wp::array_t<wp::float32> var_efc_aref_out,
    wp::array_t<wp::float32> var_efc_frictionloss_out,
    wp::array_t<wp::int32> var_efc_nnz_out)
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
        bool* var_5;
        bool var_6;
        bool var_7;
        const wp::int32 var_8 = 6;
        wp::int32 var_9;
        const wp::int32 var_10 = 6;
        wp::int32 var_11;
        const wp::int32 var_12 = 6;
        wp::int32 var_13;
        bool var_14;
        const wp::int32 var_15 = 0;
        wp::int32 var_16;
        const wp::int32 var_17 = 1;
        wp::int32 var_18;
        const wp::int32 var_19 = 2;
        wp::int32 var_20;
        const wp::int32 var_21 = 3;
        wp::int32 var_22;
        const wp::int32 var_23 = 4;
        wp::int32 var_24;
        const wp::int32 var_25 = 5;
        wp::int32 var_26;
        wp::int32* var_27;
        const wp::int32 var_28 = 6;
        bool var_29;
        wp::int32 var_30;
        const wp::int32 var_31 = 0;
        bool var_32;
        bool var_33;
        wp::int32* var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        wp::int32* var_37;
        wp::int32 var_38;
        wp::int32 var_39;
        wp::shape_t* var_40;
        const wp::int32 var_41 = 0;
        wp::int32 var_42;
        wp::shape_t var_43;
        wp::int32 var_44;
        wp::vec_t<11, wp::float32>* var_45;
        wp::vec_t<11, wp::float32> var_46;
        wp::vec_t<11, wp::float32> var_47;
        const wp::int32 var_48 = 0;
        wp::float32 var_49;
        const wp::int32 var_50 = 1;
        wp::float32 var_51;
        const wp::int32 var_52 = 2;
        wp::float32 var_53;
        wp::vec_t<3, wp::float32> var_54;
        const wp::int32 var_55 = 3;
        wp::float32 var_56;
        const wp::int32 var_57 = 4;
        wp::float32 var_58;
        const wp::int32 var_59 = 5;
        wp::float32 var_60;
        wp::vec_t<3, wp::float32> var_61;
        const wp::int32 var_62 = 6;
        wp::float32 var_63;
        const wp::int32 var_64 = 7;
        wp::float32 var_65;
        const wp::int32 var_66 = 8;
        wp::float32 var_67;
        const wp::int32 var_68 = 9;
        wp::float32 var_69;
        wp::quat_t<wp::float32> var_70;
        const wp::int32 var_71 = 10;
        wp::float32 var_72;
        wp::int32* var_73;
        wp::int32 var_74;
        wp::int32 var_75;
        wp::int32* var_76;
        wp::int32 var_77;
        wp::int32 var_78;
        wp::vec_t<3, wp::float32>* var_79;
        wp::vec_t<3, wp::float32> var_80;
        wp::vec_t<3, wp::float32> var_81;
        wp::vec_t<3, wp::float32>* var_82;
        wp::vec_t<3, wp::float32> var_83;
        wp::vec_t<3, wp::float32> var_84;
        wp::shape_t* var_85;
        const wp::int32 var_86 = 0;
        wp::int32 var_87;
        wp::shape_t var_88;
        wp::int32 var_89;
        wp::quat_t<wp::float32>* var_90;
        wp::quat_t<wp::float32>* var_91;
        wp::quat_t<wp::float32> var_92;
        wp::quat_t<wp::float32> var_93;
        wp::quat_t<wp::float32> var_94;
        wp::quat_t<wp::float32>* var_95;
        wp::quat_t<wp::float32>* var_96;
        wp::quat_t<wp::float32> var_97;
        wp::quat_t<wp::float32> var_98;
        wp::quat_t<wp::float32> var_99;
        wp::quat_t<wp::float32> var_100;
        wp::int32 var_101;
        wp::int32 var_102;
        wp::vec_t<3, wp::float32>* var_103;
        wp::mat_t<3, 3, wp::float32>* var_104;
        wp::vec_t<3, wp::float32> var_105;
        wp::mat_t<3, 3, wp::float32> var_106;
        wp::vec_t<3, wp::float32> var_107;
        wp::vec_t<3, wp::float32> var_108;
        wp::vec_t<3, wp::float32>* var_109;
        wp::mat_t<3, 3, wp::float32>* var_110;
        wp::vec_t<3, wp::float32> var_111;
        wp::mat_t<3, 3, wp::float32> var_112;
        wp::vec_t<3, wp::float32> var_113;
        wp::vec_t<3, wp::float32> var_114;
        wp::quat_t<wp::float32>* var_115;
        wp::quat_t<wp::float32> var_116;
        wp::quat_t<wp::float32> var_117;
        wp::quat_t<wp::float32>* var_118;
        wp::quat_t<wp::float32> var_119;
        wp::quat_t<wp::float32> var_120;
        wp::int32 var_121;
        wp::int32 var_122;
        wp::vec_t<3, wp::float32> var_123;
        wp::vec_t<3, wp::float32> var_124;
        wp::quat_t<wp::float32> var_125;
        wp::quat_t<wp::float32> var_126;
        const wp::float32 var_127 = 0.0;
        const wp::float32 var_128 = 0.0;
        const wp::float32 var_129 = 0.0;
        wp::vec_t<3, wp::float32> var_130;
        const wp::float32 var_131 = 0.0;
        const wp::float32 var_132 = 0.0;
        const wp::float32 var_133 = 0.0;
        wp::vec_t<3, wp::float32> var_134;
        wp::int32* var_135;
        wp::int32 var_136;
        wp::int32 var_137;
        wp::int32* var_138;
        wp::int32 var_139;
        wp::int32 var_140;
        wp::int32* var_141;
        wp::int32* var_142;
        wp::int32 var_143;
        wp::int32 var_144;
        wp::int32 var_145;
        const wp::int32 var_146 = 1;
        wp::int32 var_147;
        wp::int32 var_148;
        wp::int32* var_149;
        wp::int32* var_150;
        wp::int32 var_151;
        wp::int32 var_152;
        wp::int32 var_153;
        const wp::int32 var_154 = 1;
        wp::int32 var_155;
        wp::int32 var_156;
        wp::int32 var_157;
        wp::int32 var_158;
        const wp::int32 var_159 = 0;
        wp::int32 var_160;
        const wp::int32 var_161 = 0;
        bool var_162;
        const wp::int32 var_163 = 0;
        bool var_164;
        bool var_165;
        wp::int32 var_166;
        bool var_167;
        wp::int32* var_168;
        wp::int32 var_169;
        wp::int32 var_170;
        wp::int32 var_171;
        bool var_172;
        wp::int32* var_173;
        wp::int32 var_174;
        wp::int32 var_175;
        wp::int32 var_176;
        const wp::int32 var_177 = 1;
        wp::int32 var_178;
        const wp::int32 var_179 = 6;
        wp::int32 var_180;
        wp::int32 var_181;
        const wp::int32 var_182 = 6;
        wp::int32 var_183;
        wp::int32 var_184;
        bool var_185;
        wp::int32 var_186;
        const wp::int32 var_187 = 2;
        wp::int32 var_188;
        wp::int32 var_189;
        const wp::int32 var_190 = 3;
        wp::int32 var_191;
        wp::int32 var_192;
        const wp::int32 var_193 = 4;
        wp::int32 var_194;
        wp::int32 var_195;
        const wp::int32 var_196 = 5;
        wp::int32 var_197;
        wp::int32 var_198;
        const wp::int32 var_199 = 0;
        wp::int32 var_200;
        const wp::int32 var_201 = 0;
        bool var_202;
        const wp::int32 var_203 = 0;
        bool var_204;
        bool var_205;
        wp::int32 var_206;
        bool var_207;
        wp::int32* var_208;
        wp::int32 var_209;
        wp::int32 var_210;
        wp::int32 var_211;
        bool var_212;
        wp::int32* var_213;
        wp::int32 var_214;
        wp::int32 var_215;
        wp::int32 var_216;
        wp::vec_t<3, wp::float32> var_217;
        wp::vec_t<3, wp::float32> var_218;
        wp::vec_t<3, wp::float32> var_219;
        wp::vec_t<3, wp::float32> var_220;
        wp::vec_t<3, wp::float32> var_221;
        wp::vec_t<3, wp::float32> var_222;
        wp::vec_t<3, wp::float32> var_223;
        wp::quat_t<wp::float32> var_224;
        wp::quat_t<wp::float32> var_225;
        const wp::float32 var_226 = 0.5;
        const wp::int32 var_227 = 1;
        wp::float32 var_228;
        const wp::int32 var_229 = 2;
        wp::float32 var_230;
        const wp::int32 var_231 = 3;
        wp::float32 var_232;
        wp::vec_t<3, wp::float32> var_233;
        wp::vec_t<3, wp::float32> var_234;
        wp::int32 var_235;
        wp::int32 var_236;
        wp::int32 var_237;
        const wp::int32 var_238 = 2;
        wp::int32 var_239;
        wp::int32 var_240;
        wp::int32 var_241;
        const wp::int32 var_242 = 3;
        wp::int32 var_243;
        wp::int32 var_244;
        wp::int32 var_245;
        const wp::int32 var_246 = 4;
        wp::int32 var_247;
        wp::int32 var_248;
        wp::int32 var_249;
        const wp::int32 var_250 = 5;
        wp::int32 var_251;
        wp::int32 var_252;
        wp::int32 var_253;
        const wp::int32 var_254 = 0;
        const wp::int32 var_255 = 0;
        const wp::int32 var_256 = 0;
        const wp::int32 var_257 = 0;
        const wp::int32 var_258 = 0;
        const wp::int32 var_259 = 0;
        const wp::int32 var_260 = 0;
        wp::float32 var_261;
        const wp::int32 var_262 = 0;
        const wp::int32 var_263 = 1;
        wp::float32 var_264;
        const wp::int32 var_265 = 0;
        const wp::int32 var_266 = 2;
        wp::float32 var_267;
        const wp::int32 var_268 = 0;
        const wp::int32 var_269 = 0;
        wp::float32 var_270;
        const wp::int32 var_271 = 0;
        const wp::int32 var_272 = 1;
        wp::float32 var_273;
        const wp::int32 var_274 = 0;
        const wp::int32 var_275 = 2;
        wp::float32 var_276;
        const wp::int32 var_277 = 0;
        wp::float32* var_278;
        wp::vec_t<3, wp::float32> var_279;
        wp::float32 var_280;
        wp::vec_t<3, wp::float32> var_281;
        wp::float32* var_282;
        wp::vec_t<3, wp::float32> var_283;
        wp::float32 var_284;
        wp::vec_t<3, wp::float32> var_285;
        const wp::int32 var_286 = 1;
        wp::int32 var_287;
        wp::int32 var_288;
        wp::int32 var_289;
        wp::range_t var_290;
        wp::int32 var_291;
        wp::vec_t<3, wp::float32> var_292;
        wp::vec_t<3, wp::float32> var_293;
        wp::vec_t<3, wp::float32> var_294;
        wp::vec_t<3, wp::float32> var_295;
        wp::vec_t<3, wp::float32> var_296;
        const wp::int32 var_297 = 0;
        wp::float32 var_298;
        const wp::int32 var_299 = 1;
        wp::float32 var_300;
        const wp::int32 var_301 = 2;
        wp::float32 var_302;
        wp::vec_t<3, wp::float32> var_303;
        wp::vec_t<3, wp::float32> var_304;
        wp::quat_t<wp::float32> var_305;
        wp::quat_t<wp::float32> var_306;
        const wp::float32 var_307 = 0.5;
        const wp::int32 var_308 = 1;
        wp::float32 var_309;
        const wp::int32 var_310 = 2;
        wp::float32 var_311;
        const wp::int32 var_312 = 3;
        wp::float32 var_313;
        wp::vec_t<3, wp::float32> var_314;
        wp::vec_t<3, wp::float32> var_315;
        const wp::int32 var_316 = 0;
        wp::float32 var_317;
        const wp::int32 var_318 = 1;
        wp::float32 var_319;
        const wp::int32 var_320 = 2;
        wp::float32 var_321;
        wp::float32* var_322;
        wp::vec_t<3, wp::float32> var_323;
        wp::float32 var_324;
        wp::vec_t<3, wp::float32> var_325;
        wp::float32* var_326;
        wp::vec_t<3, wp::float32> var_327;
        wp::float32 var_328;
        wp::vec_t<3, wp::float32> var_329;
        wp::vec_t<3, wp::float32> var_330;
        wp::quat_t<wp::float32> var_331;
        const wp::int32 var_332 = 1;
        wp::float32 var_333;
        const wp::int32 var_334 = 2;
        wp::float32 var_335;
        const wp::int32 var_336 = 3;
        wp::float32 var_337;
        wp::vec_t<3, wp::float32> var_338;
        wp::vec_t<3, wp::float32> var_339;
        wp::shape_t* var_340;
        const wp::int32 var_341 = 0;
        wp::int32 var_342;
        wp::shape_t var_343;
        wp::int32 var_344;
        wp::vec_t<2, wp::float32>* var_345;
        const wp::int32 var_346 = 0;
        wp::float32 var_347;
        wp::vec_t<2, wp::float32> var_348;
        wp::vec_t<2, wp::float32>* var_349;
        const wp::int32 var_350 = 0;
        wp::float32 var_351;
        wp::vec_t<2, wp::float32> var_352;
        wp::float32 var_353;
        wp::float32 var_354;
        wp::float32 var_355;
        wp::float32 var_356;
        wp::float32 var_357;
        wp::shape_t* var_358;
        const wp::int32 var_359 = 0;
        wp::int32 var_360;
        wp::shape_t var_361;
        wp::int32 var_362;
        wp::vec_t<2, wp::float32>* var_363;
        wp::vec_t<2, wp::float32> var_364;
        wp::vec_t<2, wp::float32> var_365;
        wp::shape_t* var_366;
        const wp::int32 var_367 = 0;
        wp::int32 var_368;
        wp::shape_t var_369;
        wp::int32 var_370;
        wp::vec_t<5, wp::float32>* var_371;
        wp::vec_t<5, wp::float32> var_372;
        wp::vec_t<5, wp::float32> var_373;
        wp::shape_t* var_374;
        const wp::int32 var_375 = 0;
        wp::int32 var_376;
        wp::shape_t var_377;
        wp::int32 var_378;
        wp::float32* var_379;
        wp::float32 var_380;
        wp::float32 var_381;
        const wp::int32 var_382 = 0;
        wp::int32 var_383;
        wp::float32 var_384;
        const wp::float32 var_385 = 0.0;
        wp::float32 var_386;
        const wp::float32 var_387 = 0.0;
        const wp::int32 var_388 = 0;
        const wp::int32 var_389 = 0;
        const wp::int32 var_390 = 1;
        wp::int32 var_391;
        wp::float32 var_392;
        const wp::float32 var_393 = 0.0;
        wp::float32 var_394;
        const wp::float32 var_395 = 0.0;
        const wp::int32 var_396 = 0;
        const wp::int32 var_397 = 0;
        const wp::int32 var_398 = 2;
        wp::int32 var_399;
        wp::float32 var_400;
        const wp::float32 var_401 = 0.0;
        wp::float32 var_402;
        const wp::float32 var_403 = 0.0;
        const wp::int32 var_404 = 0;
        const wp::int32 var_405 = 0;
        wp::vec_t<2, wp::float32>* var_406;
        const wp::int32 var_407 = 1;
        wp::float32 var_408;
        wp::vec_t<2, wp::float32> var_409;
        wp::vec_t<2, wp::float32>* var_410;
        const wp::int32 var_411 = 1;
        wp::float32 var_412;
        wp::vec_t<2, wp::float32> var_413;
        wp::float32 var_414;
        const wp::int32 var_415 = 0;
        const wp::int32 var_416 = 3;
        wp::int32 var_417;
        wp::int32 var_418;
        wp::float32 var_419;
        const wp::float32 var_420 = 0.0;
        wp::float32 var_421;
        const wp::float32 var_422 = 0.0;
        const wp::int32 var_423 = 0;
        const wp::int32 var_424 = 0;
        const wp::int32 var_425 = 1;
        const wp::int32 var_426 = 3;
        wp::int32 var_427;
        wp::int32 var_428;
        wp::float32 var_429;
        const wp::float32 var_430 = 0.0;
        wp::float32 var_431;
        const wp::float32 var_432 = 0.0;
        const wp::int32 var_433 = 0;
        const wp::int32 var_434 = 0;
        const wp::int32 var_435 = 2;
        const wp::int32 var_436 = 3;
        wp::int32 var_437;
        wp::int32 var_438;
        wp::float32 var_439;
        const wp::float32 var_440 = 0.0;
        wp::float32 var_441;
        const wp::float32 var_442 = 0.0;
        const wp::int32 var_443 = 0;
        const wp::int32 var_444 = 0;
        //---------
        // forward
        // def _equality_weld(                                                                    <L 793>
        // worldid, eqweldid = wp.tid()                                                           <L 846>
        builtin_tid2d(var_0, var_1);
        // eqid = eq_wld_adr[eqweldid]                                                            <L 847>
        var_2 = wp::address(var_eq_wld_adr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if not eq_active_in[worldid, eqid]:                                                    <L 849>
        var_5 = wp::address(var_eq_active_in, var_0, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::unot(var_7);
        if (var_6) {
            // return                                                                             <L 850>
            continue;
        }
        // wp.atomic_add(ne_out, worldid, 6)                                                      <L 852>
        var_9 = wp::atomic_add(var_ne_out, var_0, var_8);
        // efcid = wp.atomic_add(nefc_out, worldid, 6)                                            <L 853>
        var_11 = wp::atomic_add(var_nefc_out, var_0, var_10);
        // if efcid >= njmax_in - 6:                                                              <L 855>
        var_13 = wp::sub(var_njmax_in, var_12);
        var_14 = (var_11 >= var_13);
        if (var_14) {
            // return                                                                             <L 856>
            continue;
        }
        // efcid0 = efcid + 0                                                                     <L 858>
        var_16 = wp::add(var_11, var_15);
        // efcid1 = efcid + 1                                                                     <L 859>
        var_18 = wp::add(var_11, var_17);
        // efcid2 = efcid + 2                                                                     <L 860>
        var_20 = wp::add(var_11, var_19);
        // efcid3 = efcid + 3                                                                     <L 861>
        var_22 = wp::add(var_11, var_21);
        // efcid4 = efcid + 4                                                                     <L 862>
        var_24 = wp::add(var_11, var_23);
        // efcid5 = efcid + 5                                                                     <L 863>
        var_26 = wp::add(var_11, var_25);
        // is_site = eq_objtype[eqid] == types.ObjType.SITE and nsite > 0                         <L 865>
        var_27 = wp::address(var_eq_objtype, var_3);
        var_30 = wp::load(var_27);
        var_29 = (var_30 == var_28);
        var_32 = (var_nsite > var_31);
        var_33 = var_29 && var_32;
        // obj1id = eq_obj1id[eqid]                                                               <L 867>
        var_34 = wp::address(var_eq_obj1id, var_3);
        var_36 = wp::load(var_34);
        var_35 = wp::copy(var_36);
        // obj2id = eq_obj2id[eqid]                                                               <L 868>
        var_37 = wp::address(var_eq_obj2id, var_3);
        var_39 = wp::load(var_37);
        var_38 = wp::copy(var_39);
        // data = eq_data[worldid % eq_data.shape[0], eqid]                                       <L 870>
        var_40 = &(var_eq_data.shape);
        var_43 = wp::load(var_40);
        var_42 = wp::extract(var_43, var_41);
        var_44 = wp::mod(var_0, var_42);
        var_45 = wp::address(var_eq_data, var_44, var_3);
        var_47 = wp::load(var_45);
        var_46 = wp::copy(var_47);
        // anchor1 = wp.vec3(data[0], data[1], data[2])                                           <L 871>
        var_49 = wp::extract(var_46, var_48);
        var_51 = wp::extract(var_46, var_50);
        var_53 = wp::extract(var_46, var_52);
        var_54 = wp::vec_t<3, wp::float32>(var_49, var_51, var_53);
        // anchor2 = wp.vec3(data[3], data[4], data[5])                                           <L 872>
        var_56 = wp::extract(var_46, var_55);
        var_58 = wp::extract(var_46, var_57);
        var_60 = wp::extract(var_46, var_59);
        var_61 = wp::vec_t<3, wp::float32>(var_56, var_58, var_60);
        // relpose = wp.quat(data[6], data[7], data[8], data[9])                                  <L 873>
        var_63 = wp::extract(var_46, var_62);
        var_65 = wp::extract(var_46, var_64);
        var_67 = wp::extract(var_46, var_66);
        var_69 = wp::extract(var_46, var_68);
        var_70 = wp::quat_t<wp::float32>(var_63, var_65, var_67, var_69);
        // torquescale = data[10]                                                                 <L 874>
        var_72 = wp::extract(var_46, var_71);
        // if is_site:                                                                            <L 876>
        if (var_33) {
            // body1 = site_bodyid[obj1id]                                                        <L 877>
            var_73 = wp::address(var_site_bodyid, var_35);
            var_75 = wp::load(var_73);
            var_74 = wp::copy(var_75);
            // body2 = site_bodyid[obj2id]                                                        <L 878>
            var_76 = wp::address(var_site_bodyid, var_38);
            var_78 = wp::load(var_76);
            var_77 = wp::copy(var_78);
            // pos1 = site_xpos_in[worldid, obj1id]                                               <L 879>
            var_79 = wp::address(var_site_xpos_in, var_0, var_35);
            var_81 = wp::load(var_79);
            var_80 = wp::copy(var_81);
            // pos2 = site_xpos_in[worldid, obj2id]                                               <L 880>
            var_82 = wp::address(var_site_xpos_in, var_0, var_38);
            var_84 = wp::load(var_82);
            var_83 = wp::copy(var_84);
            // site_quat_id = worldid % site_quat.shape[0]                                        <L 882>
            var_85 = &(var_site_quat.shape);
            var_88 = wp::load(var_85);
            var_87 = wp::extract(var_88, var_86);
            var_89 = wp::mod(var_0, var_87);
            // quat = math.mul_quat(xquat_in[worldid, body1], site_quat[site_quat_id, obj1id])       <L 883>
            var_90 = wp::address(var_xquat_in, var_0, var_74);
            var_91 = wp::address(var_site_quat, var_89, var_35);
            var_93 = wp::load(var_90);
            var_94 = wp::load(var_91);
            var_92 = mul_quat_0(var_93, var_94);
            // quat1 = math.quat_inv(math.mul_quat(xquat_in[worldid, body2], site_quat[site_quat_id, obj2id]))       <L 884>
            var_95 = wp::address(var_xquat_in, var_0, var_77);
            var_96 = wp::address(var_site_quat, var_89, var_38);
            var_98 = wp::load(var_95);
            var_99 = wp::load(var_96);
            var_97 = mul_quat_0(var_98, var_99);
            var_100 = quat_inv_0(var_97);
        }
        if (!var_33) {
            // body1 = obj1id                                                                     <L 887>
            var_101 = wp::copy(var_35);
            // body2 = obj2id                                                                     <L 888>
            var_102 = wp::copy(var_38);
            // pos1 = xpos_in[worldid, body1] + xmat_in[worldid, body1] @ anchor2                 <L 889>
            var_103 = wp::address(var_xpos_in, var_0, var_101);
            var_104 = wp::address(var_xmat_in, var_0, var_101);
            var_106 = wp::load(var_104);
            var_105 = wp::mul(var_106, var_61);
            var_108 = wp::load(var_103);
            var_107 = wp::add(var_108, var_105);
            // pos2 = xpos_in[worldid, body2] + xmat_in[worldid, body2] @ anchor1                 <L 890>
            var_109 = wp::address(var_xpos_in, var_0, var_102);
            var_110 = wp::address(var_xmat_in, var_0, var_102);
            var_112 = wp::load(var_110);
            var_111 = wp::mul(var_112, var_54);
            var_114 = wp::load(var_109);
            var_113 = wp::add(var_114, var_111);
            // quat = math.mul_quat(xquat_in[worldid, body1], relpose)                            <L 892>
            var_115 = wp::address(var_xquat_in, var_0, var_101);
            var_117 = wp::load(var_115);
            var_116 = mul_quat_0(var_117, var_70);
            // quat1 = math.quat_inv(xquat_in[worldid, body2])                                    <L 893>
            var_118 = wp::address(var_xquat_in, var_0, var_102);
            var_120 = wp::load(var_118);
            var_119 = quat_inv_0(var_120);
        }
        var_121 = wp::where(var_33, var_74, var_101);
        var_122 = wp::where(var_33, var_77, var_102);
        var_123 = wp::where(var_33, var_80, var_107);
        var_124 = wp::where(var_33, var_83, var_113);
        var_125 = wp::where(var_33, var_92, var_116);
        var_126 = wp::where(var_33, var_100, var_119);
        // Jqvelp = wp.vec3f(0.0, 0.0, 0.0)                                                       <L 896>
        var_130 = wp::vec_t<3, wp::float32>(var_127, var_128, var_129);
        // Jqvelr = wp.vec3f(0.0, 0.0, 0.0)                                                       <L 897>
        var_134 = wp::vec_t<3, wp::float32>(var_131, var_132, var_133);
        // if is_sparse:                                                                          <L 899>
        if (var_is_sparse) {
            // body1 = body_weldid[body1]                                                         <L 901>
            var_135 = wp::address(var_body_weldid, var_121);
            var_137 = wp::load(var_135);
            var_136 = wp::copy(var_137);
            // body2 = body_weldid[body2]                                                         <L 902>
            var_138 = wp::address(var_body_weldid, var_122);
            var_140 = wp::load(var_138);
            var_139 = wp::copy(var_140);
            // da1 = int(body_dofadr[body1] + body_dofnum[body1] - 1)                             <L 904>
            var_141 = wp::address(var_body_dofadr, var_136);
            var_142 = wp::address(var_body_dofnum, var_136);
            var_144 = wp::load(var_141);
            var_145 = wp::load(var_142);
            var_143 = wp::add(var_144, var_145);
            var_147 = wp::sub(var_143, var_146);
            var_148 = wp::int(var_147);
            // da2 = int(body_dofadr[body2] + body_dofnum[body2] - 1)                             <L 905>
            var_149 = wp::address(var_body_dofadr, var_139);
            var_150 = wp::address(var_body_dofnum, var_139);
            var_152 = wp::load(var_149);
            var_153 = wp::load(var_150);
            var_151 = wp::add(var_152, var_153);
            var_155 = wp::sub(var_151, var_154);
            var_156 = wp::int(var_155);
            // pda1 = da1                                                                         <L 908>
            var_157 = wp::copy(var_148);
            // pda2 = da2                                                                         <L 909>
            var_158 = wp::copy(var_156);
            // rownnz = int(0)                                                                    <L 910>
            var_160 = wp::int(var_159);
            // while pda1 >= 0 or pda2 >= 0:                                                      <L 911>
        start_while_2:;
            var_162 = (var_157 >= var_161);
            var_164 = (var_158 >= var_163);
            var_165 = var_162 || var_164;
        if ((var_165) == false) goto end_while_2;
                // da = wp.max(pda1, pda2)                                                        <L 912>
                var_166 = wp::max(var_157, var_158);
                // if pda1 == da:                                                                 <L 913>
                var_167 = (var_157 == var_166);
                if (var_167) {
                    // pda1 = dof_parentid[da]                                                    <L 914>
                    var_168 = wp::address(var_dof_parentid, var_166);
                    var_170 = wp::load(var_168);
                    var_169 = wp::copy(var_170);
                }
                var_171 = wp::where(var_167, var_169, var_157);
                // if pda2 == da:                                                                 <L 915>
                var_172 = (var_158 == var_166);
                if (var_172) {
                    // pda2 = dof_parentid[da]                                                    <L 916>
                    var_173 = wp::address(var_dof_parentid, var_166);
                    var_175 = wp::load(var_173);
                    var_174 = wp::copy(var_175);
                }
                var_176 = wp::where(var_172, var_174, var_158);
                // rownnz += 1                                                                    <L 917>
                var_178 = wp::add(var_160, var_177);
                wp::assign(var_157, var_171);
                wp::assign(var_158, var_176);
                wp::assign(var_160, var_178);
        goto start_while_2;
        end_while_2:;
            // rowadr = wp.atomic_add(efc_nnz_out, worldid, 6 * rownnz)                           <L 920>
            var_180 = wp::mul(var_179, var_160);
            var_181 = wp::atomic_add(var_efc_nnz_out, var_0, var_180);
            // if rowadr + 6 * rownnz > njmax_nnz_in:                                             <L 921>
            var_183 = wp::mul(var_182, var_160);
            var_184 = wp::add(var_181, var_183);
            var_185 = (var_184 > var_njmax_nnz_in);
            if (var_185) {
                // return                                                                         <L 922>
                continue;
            }
            // efc_J_rowadr_out[worldid, efcid0] = rowadr                                         <L 923>
            wp::array_store(var_efc_J_rowadr_out, var_0, var_16, var_181);
            // efc_J_rowadr_out[worldid, efcid1] = rowadr + rownnz                                <L 924>
            var_186 = wp::add(var_181, var_160);
            wp::array_store(var_efc_J_rowadr_out, var_0, var_18, var_186);
            // efc_J_rowadr_out[worldid, efcid2] = rowadr + 2 * rownnz                            <L 925>
            var_188 = wp::mul(var_187, var_160);
            var_189 = wp::add(var_181, var_188);
            wp::array_store(var_efc_J_rowadr_out, var_0, var_20, var_189);
            // efc_J_rowadr_out[worldid, efcid3] = rowadr + 3 * rownnz                            <L 926>
            var_191 = wp::mul(var_190, var_160);
            var_192 = wp::add(var_181, var_191);
            wp::array_store(var_efc_J_rowadr_out, var_0, var_22, var_192);
            // efc_J_rowadr_out[worldid, efcid4] = rowadr + 4 * rownnz                            <L 927>
            var_194 = wp::mul(var_193, var_160);
            var_195 = wp::add(var_181, var_194);
            wp::array_store(var_efc_J_rowadr_out, var_0, var_24, var_195);
            // efc_J_rowadr_out[worldid, efcid5] = rowadr + 5 * rownnz                            <L 928>
            var_197 = wp::mul(var_196, var_160);
            var_198 = wp::add(var_181, var_197);
            wp::array_store(var_efc_J_rowadr_out, var_0, var_26, var_198);
            // efc_J_rownnz_out[worldid, efcid0] = rownnz                                         <L 930>
            wp::array_store(var_efc_J_rownnz_out, var_0, var_16, var_160);
            // efc_J_rownnz_out[worldid, efcid1] = rownnz                                         <L 931>
            wp::array_store(var_efc_J_rownnz_out, var_0, var_18, var_160);
            // efc_J_rownnz_out[worldid, efcid2] = rownnz                                         <L 932>
            wp::array_store(var_efc_J_rownnz_out, var_0, var_20, var_160);
            // efc_J_rownnz_out[worldid, efcid3] = rownnz                                         <L 933>
            wp::array_store(var_efc_J_rownnz_out, var_0, var_22, var_160);
            // efc_J_rownnz_out[worldid, efcid4] = rownnz                                         <L 934>
            wp::array_store(var_efc_J_rownnz_out, var_0, var_24, var_160);
            // efc_J_rownnz_out[worldid, efcid5] = rownnz                                         <L 935>
            wp::array_store(var_efc_J_rownnz_out, var_0, var_26, var_160);
            // nnz = int(0)                                                                       <L 938>
            var_200 = wp::int(var_199);
            // while da1 >= 0 or da2 >= 0:                                                        <L 939>
        start_while_5:;
            var_202 = (var_148 >= var_201);
            var_204 = (var_156 >= var_203);
            var_205 = var_202 || var_204;
        if ((var_205) == false) goto end_while_5;
                // da = wp.max(da1, da2)                                                          <L 940>
                var_206 = wp::max(var_148, var_156);
                // if da1 == da:                                                                  <L 941>
                var_207 = (var_148 == var_206);
                if (var_207) {
                    // da1 = dof_parentid[da]                                                     <L 942>
                    var_208 = wp::address(var_dof_parentid, var_206);
                    var_210 = wp::load(var_208);
                    var_209 = wp::copy(var_210);
                }
                var_211 = wp::where(var_207, var_209, var_148);
                // if da2 == da:                                                                  <L 943>
                var_212 = (var_156 == var_206);
                if (var_212) {
                    // da2 = dof_parentid[da]                                                     <L 944>
                    var_213 = wp::address(var_dof_parentid, var_206);
                    var_215 = wp::load(var_213);
                    var_214 = wp::copy(var_215);
                }
                var_216 = wp::where(var_212, var_214, var_156);
                // jacp1, jacr1 = support.jac_dof(                                                <L 946>
                // body_parentid,                                                                 <L 947>
                // body_rootid,                                                                   <L 948>
                // dof_bodyid,                                                                    <L 949>
                // subtree_com_in,                                                                <L 950>
                // cdof_in,                                                                       <L 951>
                // pos1,                                                                          <L 952>
                // body1,                                                                         <L 953>
                // da,                                                                            <L 954>
                // worldid,                                                                       <L 955>
                jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_subtree_com_in, var_cdof_in, var_123, var_136, var_206, var_0, var_217, var_218);
                // jacp2, jacr2 = support.jac_dof(                                                <L 957>
                // body_parentid,                                                                 <L 958>
                // body_rootid,                                                                   <L 959>
                // dof_bodyid,                                                                    <L 960>
                // subtree_com_in,                                                                <L 961>
                // cdof_in,                                                                       <L 962>
                // pos2,                                                                          <L 963>
                // body2,                                                                         <L 964>
                // da,                                                                            <L 965>
                // worldid,                                                                       <L 966>
                jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_subtree_com_in, var_cdof_in, var_124, var_139, var_206, var_0, var_219, var_220);
                // jacdifp = jacp1 - jacp2                                                        <L 969>
                var_221 = wp::sub(var_217, var_219);
                // jacdifr = (jacr1 - jacr2) * torquescale                                        <L 971>
                var_222 = wp::sub(var_218, var_220);
                var_223 = wp::mul(var_222, var_72);
                // jacdifrq = math.mul_quat(math.quat_mul_axis(quat1, jacdifr), quat)             <L 972>
                var_224 = quat_mul_axis_0(var_126, var_223);
                var_225 = mul_quat_0(var_224, var_125);
                // jacdifr = 0.5 * wp.vec3(jacdifrq[1], jacdifrq[2], jacdifrq[3])                 <L 973>
                var_228 = wp::extract(var_225, var_227);
                var_230 = wp::extract(var_225, var_229);
                var_232 = wp::extract(var_225, var_231);
                var_233 = wp::vec_t<3, wp::float32>(var_228, var_230, var_232);
                var_234 = wp::mul(var_226, var_233);
                // sparseid0 = rowadr + nnz                                                       <L 975>
                var_235 = wp::add(var_181, var_200);
                // sparseid1 = rowadr + rownnz + nnz                                              <L 976>
                var_236 = wp::add(var_181, var_160);
                var_237 = wp::add(var_236, var_200);
                // sparseid2 = rowadr + 2 * rownnz + nnz                                          <L 977>
                var_239 = wp::mul(var_238, var_160);
                var_240 = wp::add(var_181, var_239);
                var_241 = wp::add(var_240, var_200);
                // sparseid3 = rowadr + 3 * rownnz + nnz                                          <L 978>
                var_243 = wp::mul(var_242, var_160);
                var_244 = wp::add(var_181, var_243);
                var_245 = wp::add(var_244, var_200);
                // sparseid4 = rowadr + 4 * rownnz + nnz                                          <L 979>
                var_247 = wp::mul(var_246, var_160);
                var_248 = wp::add(var_181, var_247);
                var_249 = wp::add(var_248, var_200);
                // sparseid5 = rowadr + 5 * rownnz + nnz                                          <L 980>
                var_251 = wp::mul(var_250, var_160);
                var_252 = wp::add(var_181, var_251);
                var_253 = wp::add(var_252, var_200);
                // efc_J_colind_out[worldid, 0, sparseid0] = da                                   <L 982>
                wp::array_store(var_efc_J_colind_out, var_0, var_254, var_235, var_206);
                // efc_J_colind_out[worldid, 0, sparseid1] = da                                   <L 983>
                wp::array_store(var_efc_J_colind_out, var_0, var_255, var_237, var_206);
                // efc_J_colind_out[worldid, 0, sparseid2] = da                                   <L 984>
                wp::array_store(var_efc_J_colind_out, var_0, var_256, var_241, var_206);
                // efc_J_colind_out[worldid, 0, sparseid3] = da                                   <L 985>
                wp::array_store(var_efc_J_colind_out, var_0, var_257, var_245, var_206);
                // efc_J_colind_out[worldid, 0, sparseid4] = da                                   <L 986>
                wp::array_store(var_efc_J_colind_out, var_0, var_258, var_249, var_206);
                // efc_J_colind_out[worldid, 0, sparseid5] = da                                   <L 987>
                wp::array_store(var_efc_J_colind_out, var_0, var_259, var_253, var_206);
                // efc_J_out[worldid, 0, sparseid0] = jacdifp[0]                                  <L 989>
                var_261 = wp::extract(var_221, var_260);
                wp::array_store(var_efc_J_out, var_0, var_262, var_235, var_261);
                // efc_J_out[worldid, 0, sparseid1] = jacdifp[1]                                  <L 990>
                var_264 = wp::extract(var_221, var_263);
                wp::array_store(var_efc_J_out, var_0, var_265, var_237, var_264);
                // efc_J_out[worldid, 0, sparseid2] = jacdifp[2]                                  <L 991>
                var_267 = wp::extract(var_221, var_266);
                wp::array_store(var_efc_J_out, var_0, var_268, var_241, var_267);
                // efc_J_out[worldid, 0, sparseid3] = jacdifr[0]                                  <L 992>
                var_270 = wp::extract(var_234, var_269);
                wp::array_store(var_efc_J_out, var_0, var_271, var_245, var_270);
                // efc_J_out[worldid, 0, sparseid4] = jacdifr[1]                                  <L 993>
                var_273 = wp::extract(var_234, var_272);
                wp::array_store(var_efc_J_out, var_0, var_274, var_249, var_273);
                // efc_J_out[worldid, 0, sparseid5] = jacdifr[2]                                  <L 994>
                var_276 = wp::extract(var_234, var_275);
                wp::array_store(var_efc_J_out, var_0, var_277, var_253, var_276);
                // Jqvelp += jacdifp * qvel_in[worldid, da]                                       <L 996>
                var_278 = wp::address(var_qvel_in, var_0, var_206);
                var_280 = wp::load(var_278);
                var_279 = wp::mul(var_221, var_280);
                var_281 = wp::add(var_130, var_279);
                // Jqvelr += jacdifr * qvel_in[worldid, da]                                       <L 997>
                var_282 = wp::address(var_qvel_in, var_0, var_206);
                var_284 = wp::load(var_282);
                var_283 = wp::mul(var_234, var_284);
                var_285 = wp::add(var_134, var_283);
                // nnz += 1                                                                       <L 999>
                var_287 = wp::add(var_200, var_286);
                wp::assign(var_130, var_281);
                wp::assign(var_134, var_285);
                wp::assign(var_148, var_211);
                wp::assign(var_156, var_216);
                wp::assign(var_166, var_206);
                wp::assign(var_200, var_287);
        goto start_while_5;
        end_while_5:;
        }
        var_288 = wp::where(var_is_sparse, var_136, var_121);
        var_289 = wp::where(var_is_sparse, var_139, var_122);
        if (!var_is_sparse) {
            // for dofid in range(nv):                                                            <L 1001>
            var_290 = wp::range(var_nv);
            start_for_7:;
                if (iter_cmp(var_290) == 0) goto end_for_7;
                var_291 = wp::iter_next(var_290);
                // jacp1, jacr1 = support.jac_dof(                                                <L 1002>
                // body_parentid,                                                                 <L 1003>
                // body_rootid,                                                                   <L 1004>
                // dof_bodyid,                                                                    <L 1005>
                // subtree_com_in,                                                                <L 1006>
                // cdof_in,                                                                       <L 1007>
                // pos1,                                                                          <L 1008>
                // body1,                                                                         <L 1009>
                // dofid,                                                                         <L 1010>
                // worldid,                                                                       <L 1011>
                jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_subtree_com_in, var_cdof_in, var_123, var_288, var_291, var_0, var_292, var_293);
                // jacp2, jacr2 = support.jac_dof(                                                <L 1013>
                // body_parentid,                                                                 <L 1014>
                // body_rootid,                                                                   <L 1015>
                // dof_bodyid,                                                                    <L 1016>
                // subtree_com_in,                                                                <L 1017>
                // cdof_in,                                                                       <L 1018>
                // pos2,                                                                          <L 1019>
                // body2,                                                                         <L 1020>
                // dofid,                                                                         <L 1021>
                // worldid,                                                                       <L 1022>
                jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_subtree_com_in, var_cdof_in, var_124, var_289, var_291, var_0, var_294, var_295);
                // jacdifp = jacp1 - jacp2                                                        <L 1025>
                var_296 = wp::sub(var_292, var_294);
                // efc_J_out[worldid, efcid0, dofid] = jacdifp[0]                                 <L 1027>
                var_298 = wp::extract(var_296, var_297);
                wp::array_store(var_efc_J_out, var_0, var_16, var_291, var_298);
                // efc_J_out[worldid, efcid1, dofid] = jacdifp[1]                                 <L 1028>
                var_300 = wp::extract(var_296, var_299);
                wp::array_store(var_efc_J_out, var_0, var_18, var_291, var_300);
                // efc_J_out[worldid, efcid2, dofid] = jacdifp[2]                                 <L 1029>
                var_302 = wp::extract(var_296, var_301);
                wp::array_store(var_efc_J_out, var_0, var_20, var_291, var_302);
                // jacdifr = (jacr1 - jacr2) * torquescale                                        <L 1031>
                var_303 = wp::sub(var_293, var_295);
                var_304 = wp::mul(var_303, var_72);
                // jacdifrq = math.mul_quat(math.quat_mul_axis(quat1, jacdifr), quat)             <L 1032>
                var_305 = quat_mul_axis_0(var_126, var_304);
                var_306 = mul_quat_0(var_305, var_125);
                // jacdifr = 0.5 * wp.vec3(jacdifrq[1], jacdifrq[2], jacdifrq[3])                 <L 1033>
                var_309 = wp::extract(var_306, var_308);
                var_311 = wp::extract(var_306, var_310);
                var_313 = wp::extract(var_306, var_312);
                var_314 = wp::vec_t<3, wp::float32>(var_309, var_311, var_313);
                var_315 = wp::mul(var_307, var_314);
                // efc_J_out[worldid, efcid3, dofid] = jacdifr[0]                                 <L 1035>
                var_317 = wp::extract(var_315, var_316);
                wp::array_store(var_efc_J_out, var_0, var_22, var_291, var_317);
                // efc_J_out[worldid, efcid4, dofid] = jacdifr[1]                                 <L 1036>
                var_319 = wp::extract(var_315, var_318);
                wp::array_store(var_efc_J_out, var_0, var_24, var_291, var_319);
                // efc_J_out[worldid, efcid5, dofid] = jacdifr[2]                                 <L 1037>
                var_321 = wp::extract(var_315, var_320);
                wp::array_store(var_efc_J_out, var_0, var_26, var_291, var_321);
                // Jqvelp += jacdifp * qvel_in[worldid, dofid]                                    <L 1039>
                var_322 = wp::address(var_qvel_in, var_0, var_291);
                var_324 = wp::load(var_322);
                var_323 = wp::mul(var_296, var_324);
                var_325 = wp::add(var_130, var_323);
                // Jqvelr += jacdifr * qvel_in[worldid, dofid]                                    <L 1040>
                var_326 = wp::address(var_qvel_in, var_0, var_291);
                var_328 = wp::load(var_326);
                var_327 = wp::mul(var_315, var_328);
                var_329 = wp::add(var_134, var_327);
                wp::assign(var_130, var_325);
                wp::assign(var_134, var_329);
                wp::assign(var_217, var_292);
                wp::assign(var_218, var_293);
                wp::assign(var_219, var_294);
                wp::assign(var_220, var_295);
                wp::assign(var_221, var_296);
                wp::assign(var_234, var_315);
                wp::assign(var_225, var_306);
                goto start_for_7;
            end_for_7:;
        }
        // cpos = pos1 - pos2                                                                     <L 1043>
        var_330 = wp::sub(var_123, var_124);
        // crotq = math.mul_quat(quat1, quat)  # copy axis components                             <L 1045>
        var_331 = mul_quat_0(var_126, var_125);
        // crot = wp.vec3(crotq[1], crotq[2], crotq[3]) * torquescale                             <L 1046>
        var_333 = wp::extract(var_331, var_332);
        var_335 = wp::extract(var_331, var_334);
        var_337 = wp::extract(var_331, var_336);
        var_338 = wp::vec_t<3, wp::float32>(var_333, var_335, var_337);
        var_339 = wp::mul(var_338, var_72);
        // body_invweight0_id = worldid % body_invweight0.shape[0]                                <L 1048>
        var_340 = &(var_body_invweight0.shape);
        var_343 = wp::load(var_340);
        var_342 = wp::extract(var_343, var_341);
        var_344 = wp::mod(var_0, var_342);
        // invweight_t = body_invweight0[body_invweight0_id, body1][0] + body_invweight0[body_invweight0_id, body2][0]       <L 1049>
        var_345 = wp::address(var_body_invweight0, var_344, var_288);
        var_348 = wp::load(var_345);
        var_347 = wp::extract(var_348, var_346);
        var_349 = wp::address(var_body_invweight0, var_344, var_289);
        var_352 = wp::load(var_349);
        var_351 = wp::extract(var_352, var_350);
        var_353 = wp::add(var_347, var_351);
        // pos_imp = wp.sqrt(wp.length_sq(cpos) + wp.length_sq(crot))                             <L 1051>
        var_354 = wp::length_sq(var_330);
        var_355 = wp::length_sq(var_339);
        var_356 = wp::add(var_354, var_355);
        var_357 = wp::sqrt(var_356);
        // solref = eq_solref[worldid % eq_solref.shape[0], eqid]                                 <L 1053>
        var_358 = &(var_eq_solref.shape);
        var_361 = wp::load(var_358);
        var_360 = wp::extract(var_361, var_359);
        var_362 = wp::mod(var_0, var_360);
        var_363 = wp::address(var_eq_solref, var_362, var_3);
        var_365 = wp::load(var_363);
        var_364 = wp::copy(var_365);
        // solimp = eq_solimp[worldid % eq_solimp.shape[0], eqid]                                 <L 1054>
        var_366 = &(var_eq_solimp.shape);
        var_369 = wp::load(var_366);
        var_368 = wp::extract(var_369, var_367);
        var_370 = wp::mod(var_0, var_368);
        var_371 = wp::address(var_eq_solimp, var_370, var_3);
        var_373 = wp::load(var_371);
        var_372 = wp::copy(var_373);
        // timestep = opt_timestep[worldid % opt_timestep.shape[0]]                               <L 1056>
        var_374 = &(var_opt_timestep.shape);
        var_377 = wp::load(var_374);
        var_376 = wp::extract(var_377, var_375);
        var_378 = wp::mod(var_0, var_376);
        var_379 = wp::address(var_opt_timestep, var_378);
        var_381 = wp::load(var_379);
        var_380 = wp::copy(var_381);
        // for i in range(3):                                                                     <L 1058>
        // _efc_row(                                                                              <L 1059>
        // opt_disableflags,                                                                      <L 1060>
        // worldid,                                                                               <L 1061>
        // timestep,                                                                              <L 1062>
        // efcid + i,                                                                             <L 1063>
        var_383 = wp::add(var_11, var_382);
        // cpos[i],                                                                               <L 1064>
        var_384 = wp::extract(var_330, var_382);
        // pos_imp,                                                                               <L 1065>
        // invweight_t,                                                                           <L 1066>
        // solref,                                                                                <L 1067>
        // solimp,                                                                                <L 1068>
        // 0.0,                                                                                   <L 1069>
        // Jqvelp[i],                                                                             <L 1070>
        var_386 = wp::extract(var_130, var_382);
        // 0.0,                                                                                   <L 1071>
        // ConstraintType.EQUALITY,                                                               <L 1072>
        // eqid,                                                                                  <L 1073>
        // efc_type_out,                                                                          <L 1074>
        // efc_id_out,                                                                            <L 1075>
        // efc_pos_out,                                                                           <L 1076>
        // efc_margin_out,                                                                        <L 1077>
        // efc_D_out,                                                                             <L 1078>
        // efc_vel_out,                                                                           <L 1079>
        // efc_aref_out,                                                                          <L 1080>
        // efc_frictionloss_out,                                                                  <L 1081>
        _efc_row_0(var_opt_disableflags, var_0, var_380, var_383, var_384, var_357, var_353, var_364, var_372, var_385, var_386, var_387, var_389, var_3, var_efc_type_out, var_efc_id_out, var_efc_pos_out, var_efc_margin_out, var_efc_D_out, var_efc_vel_out, var_efc_aref_out, var_efc_frictionloss_out);
        // _efc_row(                                                                              <L 1059>
        // opt_disableflags,                                                                      <L 1060>
        // worldid,                                                                               <L 1061>
        // timestep,                                                                              <L 1062>
        // efcid + i,                                                                             <L 1063>
        var_391 = wp::add(var_11, var_390);
        // cpos[i],                                                                               <L 1064>
        var_392 = wp::extract(var_330, var_390);
        // pos_imp,                                                                               <L 1065>
        // invweight_t,                                                                           <L 1066>
        // solref,                                                                                <L 1067>
        // solimp,                                                                                <L 1068>
        // 0.0,                                                                                   <L 1069>
        // Jqvelp[i],                                                                             <L 1070>
        var_394 = wp::extract(var_130, var_390);
        // 0.0,                                                                                   <L 1071>
        // ConstraintType.EQUALITY,                                                               <L 1072>
        // eqid,                                                                                  <L 1073>
        // efc_type_out,                                                                          <L 1074>
        // efc_id_out,                                                                            <L 1075>
        // efc_pos_out,                                                                           <L 1076>
        // efc_margin_out,                                                                        <L 1077>
        // efc_D_out,                                                                             <L 1078>
        // efc_vel_out,                                                                           <L 1079>
        // efc_aref_out,                                                                          <L 1080>
        // efc_frictionloss_out,                                                                  <L 1081>
        _efc_row_0(var_opt_disableflags, var_0, var_380, var_391, var_392, var_357, var_353, var_364, var_372, var_393, var_394, var_395, var_397, var_3, var_efc_type_out, var_efc_id_out, var_efc_pos_out, var_efc_margin_out, var_efc_D_out, var_efc_vel_out, var_efc_aref_out, var_efc_frictionloss_out);
        // _efc_row(                                                                              <L 1059>
        // opt_disableflags,                                                                      <L 1060>
        // worldid,                                                                               <L 1061>
        // timestep,                                                                              <L 1062>
        // efcid + i,                                                                             <L 1063>
        var_399 = wp::add(var_11, var_398);
        // cpos[i],                                                                               <L 1064>
        var_400 = wp::extract(var_330, var_398);
        // pos_imp,                                                                               <L 1065>
        // invweight_t,                                                                           <L 1066>
        // solref,                                                                                <L 1067>
        // solimp,                                                                                <L 1068>
        // 0.0,                                                                                   <L 1069>
        // Jqvelp[i],                                                                             <L 1070>
        var_402 = wp::extract(var_130, var_398);
        // 0.0,                                                                                   <L 1071>
        // ConstraintType.EQUALITY,                                                               <L 1072>
        // eqid,                                                                                  <L 1073>
        // efc_type_out,                                                                          <L 1074>
        // efc_id_out,                                                                            <L 1075>
        // efc_pos_out,                                                                           <L 1076>
        // efc_margin_out,                                                                        <L 1077>
        // efc_D_out,                                                                             <L 1078>
        // efc_vel_out,                                                                           <L 1079>
        // efc_aref_out,                                                                          <L 1080>
        // efc_frictionloss_out,                                                                  <L 1081>
        _efc_row_0(var_opt_disableflags, var_0, var_380, var_399, var_400, var_357, var_353, var_364, var_372, var_401, var_402, var_403, var_405, var_3, var_efc_type_out, var_efc_id_out, var_efc_pos_out, var_efc_margin_out, var_efc_D_out, var_efc_vel_out, var_efc_aref_out, var_efc_frictionloss_out);
        // invweight_r = body_invweight0[body_invweight0_id, body1][1] + body_invweight0[body_invweight0_id, body2][1]       <L 1084>
        var_406 = wp::address(var_body_invweight0, var_344, var_288);
        var_409 = wp::load(var_406);
        var_408 = wp::extract(var_409, var_407);
        var_410 = wp::address(var_body_invweight0, var_344, var_289);
        var_413 = wp::load(var_410);
        var_412 = wp::extract(var_413, var_411);
        var_414 = wp::add(var_408, var_412);
        // for i in range(3):                                                                     <L 1086>
        // _efc_row(                                                                              <L 1087>
        // opt_disableflags,                                                                      <L 1088>
        // worldid,                                                                               <L 1089>
        // timestep,                                                                              <L 1090>
        // efcid + 3 + i,                                                                         <L 1091>
        var_417 = wp::add(var_11, var_416);
        var_418 = wp::add(var_417, var_415);
        // crot[i],                                                                               <L 1092>
        var_419 = wp::extract(var_339, var_415);
        // pos_imp,                                                                               <L 1093>
        // invweight_r,                                                                           <L 1094>
        // solref,                                                                                <L 1095>
        // solimp,                                                                                <L 1096>
        // 0.0,                                                                                   <L 1097>
        // Jqvelr[i],                                                                             <L 1098>
        var_421 = wp::extract(var_134, var_415);
        // 0.0,                                                                                   <L 1099>
        // ConstraintType.EQUALITY,                                                               <L 1100>
        // eqid,                                                                                  <L 1101>
        // efc_type_out,                                                                          <L 1102>
        // efc_id_out,                                                                            <L 1103>
        // efc_pos_out,                                                                           <L 1104>
        // efc_margin_out,                                                                        <L 1105>
        // efc_D_out,                                                                             <L 1106>
        // efc_vel_out,                                                                           <L 1107>
        // efc_aref_out,                                                                          <L 1108>
        // efc_frictionloss_out,                                                                  <L 1109>
        _efc_row_0(var_opt_disableflags, var_0, var_380, var_418, var_419, var_357, var_414, var_364, var_372, var_420, var_421, var_422, var_424, var_3, var_efc_type_out, var_efc_id_out, var_efc_pos_out, var_efc_margin_out, var_efc_D_out, var_efc_vel_out, var_efc_aref_out, var_efc_frictionloss_out);
        // _efc_row(                                                                              <L 1087>
        // opt_disableflags,                                                                      <L 1088>
        // worldid,                                                                               <L 1089>
        // timestep,                                                                              <L 1090>
        // efcid + 3 + i,                                                                         <L 1091>
        var_427 = wp::add(var_11, var_426);
        var_428 = wp::add(var_427, var_425);
        // crot[i],                                                                               <L 1092>
        var_429 = wp::extract(var_339, var_425);
        // pos_imp,                                                                               <L 1093>
        // invweight_r,                                                                           <L 1094>
        // solref,                                                                                <L 1095>
        // solimp,                                                                                <L 1096>
        // 0.0,                                                                                   <L 1097>
        // Jqvelr[i],                                                                             <L 1098>
        var_431 = wp::extract(var_134, var_425);
        // 0.0,                                                                                   <L 1099>
        // ConstraintType.EQUALITY,                                                               <L 1100>
        // eqid,                                                                                  <L 1101>
        // efc_type_out,                                                                          <L 1102>
        // efc_id_out,                                                                            <L 1103>
        // efc_pos_out,                                                                           <L 1104>
        // efc_margin_out,                                                                        <L 1105>
        // efc_D_out,                                                                             <L 1106>
        // efc_vel_out,                                                                           <L 1107>
        // efc_aref_out,                                                                          <L 1108>
        // efc_frictionloss_out,                                                                  <L 1109>
        _efc_row_0(var_opt_disableflags, var_0, var_380, var_428, var_429, var_357, var_414, var_364, var_372, var_430, var_431, var_432, var_434, var_3, var_efc_type_out, var_efc_id_out, var_efc_pos_out, var_efc_margin_out, var_efc_D_out, var_efc_vel_out, var_efc_aref_out, var_efc_frictionloss_out);
        // _efc_row(                                                                              <L 1087>
        // opt_disableflags,                                                                      <L 1088>
        // worldid,                                                                               <L 1089>
        // timestep,                                                                              <L 1090>
        // efcid + 3 + i,                                                                         <L 1091>
        var_437 = wp::add(var_11, var_436);
        var_438 = wp::add(var_437, var_435);
        // crot[i],                                                                               <L 1092>
        var_439 = wp::extract(var_339, var_435);
        // pos_imp,                                                                               <L 1093>
        // invweight_r,                                                                           <L 1094>
        // solref,                                                                                <L 1095>
        // solimp,                                                                                <L 1096>
        // 0.0,                                                                                   <L 1097>
        // Jqvelr[i],                                                                             <L 1098>
        var_441 = wp::extract(var_134, var_435);
        // 0.0,                                                                                   <L 1099>
        // ConstraintType.EQUALITY,                                                               <L 1100>
        // eqid,                                                                                  <L 1101>
        // efc_type_out,                                                                          <L 1102>
        // efc_id_out,                                                                            <L 1103>
        // efc_pos_out,                                                                           <L 1104>
        // efc_margin_out,                                                                        <L 1105>
        // efc_D_out,                                                                             <L 1106>
        // efc_vel_out,                                                                           <L 1107>
        // efc_aref_out,                                                                          <L 1108>
        // efc_frictionloss_out,                                                                  <L 1109>
        _efc_row_0(var_opt_disableflags, var_0, var_380, var_438, var_439, var_357, var_414, var_364, var_372, var_440, var_441, var_442, var_444, var_3, var_efc_type_out, var_efc_id_out, var_efc_pos_out, var_efc_margin_out, var_efc_D_out, var_efc_vel_out, var_efc_aref_out, var_efc_frictionloss_out);
    }
}



extern "C" __global__ void _friction_tendon_1d1c5743_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::int32 var_opt_disableflags,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::vec_t<2, wp::float32>> var_tendon_solref_fri,
    wp::array_t<wp::vec_t<5, wp::float32>> var_tendon_solimp_fri,
    wp::array_t<wp::float32> var_tendon_frictionloss,
    wp::array_t<wp::float32> var_tendon_invweight0,
    bool var_is_sparse,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::int32 var_njmax_in,
    wp::int32 var_njmax_nnz_in,
    wp::array_t<wp::int32> var_nf_out,
    wp::array_t<wp::int32> var_nefc_out,
    wp::array_t<wp::int32> var_efc_type_out,
    wp::array_t<wp::int32> var_efc_id_out,
    wp::array_t<wp::int32> var_efc_J_rownnz_out,
    wp::array_t<wp::int32> var_efc_J_rowadr_out,
    wp::array_t<wp::int32> var_efc_J_colind_out,
    wp::array_t<wp::float32> var_efc_J_out,
    wp::array_t<wp::float32> var_efc_pos_out,
    wp::array_t<wp::float32> var_efc_margin_out,
    wp::array_t<wp::float32> var_efc_D_out,
    wp::array_t<wp::float32> var_efc_vel_out,
    wp::array_t<wp::float32> var_efc_aref_out,
    wp::array_t<wp::float32> var_efc_frictionloss_out,
    wp::array_t<wp::int32> var_efc_nnz_out)
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
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        const wp::float32 var_10 = 0.0;
        bool var_11;
        const wp::int32 var_12 = 1;
        wp::int32 var_13;
        const wp::int32 var_14 = 1;
        wp::int32 var_15;
        bool var_16;
        const wp::float32 var_17 = 0.0;
        wp::float32 var_18;
        wp::int32* var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        bool var_27;
        wp::range_t var_28;
        wp::int32 var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        wp::int32* var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        wp::float32* var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        const wp::int32 var_38 = 0;
        const wp::int32 var_39 = 0;
        wp::float32* var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        const wp::int32 var_44 = 0;
        wp::int32 var_45;
        wp::int32* var_46;
        wp::int32 var_47;
        wp::int32 var_48;
        wp::range_t var_49;
        wp::int32 var_50;
        bool var_51;
        bool var_52;
        bool var_53;
        wp::int32 var_54;
        wp::float32* var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        wp::float32* var_58;
        wp::float32 var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        const wp::int32 var_62 = 1;
        wp::int32 var_63;
        bool var_64;
        wp::int32 var_65;
        wp::int32* var_66;
        wp::int32 var_67;
        wp::int32 var_68;
        wp::int32 var_69;
        wp::float32 var_70;
        wp::int32 var_71;
        wp::float32 var_72;
        wp::int32 var_73;
        const wp::float32 var_74 = 0.0;
        wp::int32 var_75;
        wp::int32 var_76;
        wp::shape_t* var_77;
        const wp::int32 var_78 = 0;
        wp::int32 var_79;
        wp::shape_t var_80;
        wp::int32 var_81;
        wp::shape_t* var_82;
        const wp::int32 var_83 = 0;
        wp::int32 var_84;
        wp::shape_t var_85;
        wp::int32 var_86;
        wp::shape_t* var_87;
        const wp::int32 var_88 = 0;
        wp::int32 var_89;
        wp::shape_t var_90;
        wp::int32 var_91;
        wp::shape_t* var_92;
        const wp::int32 var_93 = 0;
        wp::int32 var_94;
        wp::shape_t var_95;
        wp::int32 var_96;
        wp::float32* var_97;
        const wp::float32 var_98 = 0.0;
        const wp::float32 var_99 = 0.0;
        wp::float32* var_100;
        wp::vec_t<2, wp::float32>* var_101;
        wp::vec_t<5, wp::float32>* var_102;
        const wp::float32 var_103 = 0.0;
        const wp::int32 var_104 = 2;
        const wp::int32 var_105 = 2;
        wp::float32 var_106;
        wp::float32 var_107;
        wp::vec_t<2, wp::float32> var_108;
        wp::vec_t<5, wp::float32> var_109;
        //---------
        // forward
        // def _friction_tendon(                                                                  <L 1204>
        // worldid, tenid = wp.tid()                                                              <L 1240>
        builtin_tid2d(var_0, var_1);
        // tendon_frictionloss_id = worldid % tendon_frictionloss.shape[0]                        <L 1242>
        var_2 = &(var_tendon_frictionloss.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // frictionloss = tendon_frictionloss[tendon_frictionloss_id, tenid]                      <L 1244>
        var_7 = wp::address(var_tendon_frictionloss, var_6, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // if frictionloss <= 0.0:                                                                <L 1245>
        var_11 = (var_8 <= var_10);
        if (var_11) {
            // return                                                                             <L 1246>
            continue;
        }
        // wp.atomic_add(nf_out, worldid, 1)                                                      <L 1248>
        var_13 = wp::atomic_add(var_nf_out, var_0, var_12);
        // efcid = wp.atomic_add(nefc_out, worldid, 1)                                            <L 1249>
        var_15 = wp::atomic_add(var_nefc_out, var_0, var_14);
        // if efcid >= njmax_in:                                                                  <L 1251>
        var_16 = (var_15 >= var_njmax_in);
        if (var_16) {
            // return                                                                             <L 1252>
            continue;
        }
        // Jqvel = float(0.0)                                                                     <L 1254>
        var_18 = wp::float(var_17);
        // rownnz_tenJ = ten_J_rownnz[tenid]                                                      <L 1256>
        var_19 = wp::address(var_ten_J_rownnz, var_1);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // rowadr_tenJ = ten_J_rowadr[tenid]                                                      <L 1257>
        var_22 = wp::address(var_ten_J_rowadr, var_1);
        var_24 = wp::load(var_22);
        var_23 = wp::copy(var_24);
        // if is_sparse:                                                                          <L 1258>
        if (var_is_sparse) {
            // efc_J_rownnz_out[worldid, efcid] = rownnz_tenJ                                     <L 1259>
            wp::array_store(var_efc_J_rownnz_out, var_0, var_15, var_20);
            // rowadr_efc = wp.atomic_add(efc_nnz_out, worldid, rownnz_tenJ)                      <L 1260>
            var_25 = wp::atomic_add(var_efc_nnz_out, var_0, var_20);
            // if rowadr_efc + rownnz_tenJ > njmax_nnz_in:                                        <L 1261>
            var_26 = wp::add(var_25, var_20);
            var_27 = (var_26 > var_njmax_nnz_in);
            if (var_27) {
                // return                                                                         <L 1262>
                continue;
            }
            // efc_J_rowadr_out[worldid, efcid] = rowadr_efc                                      <L 1263>
            wp::array_store(var_efc_J_rowadr_out, var_0, var_15, var_25);
            // for i in range(rownnz_tenJ):                                                       <L 1265>
            var_28 = wp::range(var_20);
            start_for_3:;
                if (iter_cmp(var_28) == 0) goto end_for_3;
                var_29 = wp::iter_next(var_28);
                // sparseid_ten = rowadr_tenJ + i                                                 <L 1266>
                var_30 = wp::add(var_23, var_29);
                // sparseid_efc = rowadr_efc + i                                                  <L 1267>
                var_31 = wp::add(var_25, var_29);
                // colind = ten_J_colind[sparseid_ten]                                            <L 1268>
                var_32 = wp::address(var_ten_J_colind, var_30);
                var_34 = wp::load(var_32);
                var_33 = wp::copy(var_34);
                // J = ten_J_in[worldid, sparseid_ten]                                            <L 1269>
                var_35 = wp::address(var_ten_J_in, var_0, var_30);
                var_37 = wp::load(var_35);
                var_36 = wp::copy(var_37);
                // efc_J_colind_out[worldid, 0, sparseid_efc] = colind                            <L 1270>
                wp::array_store(var_efc_J_colind_out, var_0, var_38, var_31, var_33);
                // efc_J_out[worldid, 0, sparseid_efc] = J                                        <L 1271>
                wp::array_store(var_efc_J_out, var_0, var_39, var_31, var_36);
                // Jqvel += J * qvel_in[worldid, colind]                                          <L 1272>
                var_40 = wp::address(var_qvel_in, var_0, var_33);
                var_42 = wp::load(var_40);
                var_41 = wp::mul(var_36, var_42);
                var_43 = wp::add(var_18, var_41);
                wp::assign(var_18, var_43);
                goto start_for_3;
            end_for_3:;
        }
        if (!var_is_sparse) {
            // nnz = int(0)                                                                       <L 1274>
            var_45 = wp::int(var_44);
            // colind = ten_J_colind[rowadr_tenJ]                                                 <L 1275>
            var_46 = wp::address(var_ten_J_colind, var_23);
            var_48 = wp::load(var_46);
            var_47 = wp::copy(var_48);
            // for i in range(nv):                                                                <L 1276>
            var_49 = wp::range(var_nv);
            start_for_5:;
                if (iter_cmp(var_49) == 0) goto end_for_5;
                var_50 = wp::iter_next(var_49);
                // if nnz < rownnz_tenJ and i == colind:                                          <L 1277>
                var_51 = (var_45 < var_20);
                var_52 = (var_50 == var_47);
                var_53 = var_51 && var_52;
                if (var_53) {
                    // J = ten_J_in[worldid, rowadr_tenJ + nnz]                                   <L 1278>
                    var_54 = wp::add(var_23, var_45);
                    var_55 = wp::address(var_ten_J_in, var_0, var_54);
                    var_57 = wp::load(var_55);
                    var_56 = wp::copy(var_57);
                    // efc_J_out[worldid, efcid, i] = J                                           <L 1279>
                    wp::array_store(var_efc_J_out, var_0, var_15, var_50, var_56);
                    // Jqvel += J * qvel_in[worldid, i]                                           <L 1280>
                    var_58 = wp::address(var_qvel_in, var_0, var_50);
                    var_60 = wp::load(var_58);
                    var_59 = wp::mul(var_56, var_60);
                    var_61 = wp::add(var_18, var_59);
                    // nnz += 1                                                                   <L 1281>
                    var_63 = wp::add(var_45, var_62);
                    // if nnz < rownnz_tenJ:                                                      <L 1282>
                    var_64 = (var_63 < var_20);
                    if (var_64) {
                        // colind = ten_J_colind[rowadr_tenJ + nnz]                               <L 1283>
                        var_65 = wp::add(var_23, var_63);
                        var_66 = wp::address(var_ten_J_colind, var_65);
                        var_68 = wp::load(var_66);
                        var_67 = wp::copy(var_68);
                    }
                    var_69 = wp::where(var_64, var_67, var_47);
                }
                var_70 = wp::where(var_53, var_61, var_18);
                var_71 = wp::where(var_53, var_69, var_47);
                var_72 = wp::where(var_53, var_56, var_36);
                var_73 = wp::where(var_53, var_63, var_45);
                if (!var_53) {
                    // efc_J_out[worldid, efcid, i] = 0.0                                         <L 1285>
                    wp::array_store(var_efc_J_out, var_0, var_15, var_50, var_74);
                }
                wp::assign(var_18, var_70);
                wp::assign(var_47, var_71);
                wp::assign(var_36, var_72);
                wp::assign(var_45, var_73);
                goto start_for_5;
            end_for_5:;
        }
        var_75 = wp::where(var_is_sparse, var_29, var_50);
        var_76 = wp::where(var_is_sparse, var_33, var_47);
        // tendon_invweight0_id = worldid % tendon_invweight0.shape[0]                            <L 1287>
        var_77 = &(var_tendon_invweight0.shape);
        var_80 = wp::load(var_77);
        var_79 = wp::extract(var_80, var_78);
        var_81 = wp::mod(var_0, var_79);
        // tendon_solref_fri_id = worldid % tendon_solref_fri.shape[0]                            <L 1288>
        var_82 = &(var_tendon_solref_fri.shape);
        var_85 = wp::load(var_82);
        var_84 = wp::extract(var_85, var_83);
        var_86 = wp::mod(var_0, var_84);
        // tendon_solimp_fri_id = worldid % tendon_solimp_fri.shape[0]                            <L 1289>
        var_87 = &(var_tendon_solimp_fri.shape);
        var_90 = wp::load(var_87);
        var_89 = wp::extract(var_90, var_88);
        var_91 = wp::mod(var_0, var_89);
        // _efc_row(                                                                              <L 1290>
        // opt_disableflags,                                                                      <L 1291>
        // worldid,                                                                               <L 1292>
        // opt_timestep[worldid % opt_timestep.shape[0]],                                         <L 1293>
        var_92 = &(var_opt_timestep.shape);
        var_95 = wp::load(var_92);
        var_94 = wp::extract(var_95, var_93);
        var_96 = wp::mod(var_0, var_94);
        var_97 = wp::address(var_opt_timestep, var_96);
        // efcid,                                                                                 <L 1294>
        // 0.0,                                                                                   <L 1295>
        // 0.0,                                                                                   <L 1296>
        // tendon_invweight0[tendon_invweight0_id, tenid],                                        <L 1297>
        var_100 = wp::address(var_tendon_invweight0, var_81, var_1);
        // tendon_solref_fri[tendon_solref_fri_id, tenid],                                        <L 1298>
        var_101 = wp::address(var_tendon_solref_fri, var_86, var_1);
        // tendon_solimp_fri[tendon_solimp_fri_id, tenid],                                        <L 1299>
        var_102 = wp::address(var_tendon_solimp_fri, var_91, var_1);
        // 0.0,                                                                                   <L 1300>
        // Jqvel,                                                                                 <L 1301>
        // frictionloss,                                                                          <L 1302>
        // ConstraintType.FRICTION_TENDON,                                                        <L 1303>
        // tenid,                                                                                 <L 1304>
        // efc_type_out,                                                                          <L 1305>
        // efc_id_out,                                                                            <L 1306>
        // efc_pos_out,                                                                           <L 1307>
        // efc_margin_out,                                                                        <L 1308>
        // efc_D_out,                                                                             <L 1309>
        // efc_vel_out,                                                                           <L 1310>
        // efc_aref_out,                                                                          <L 1311>
        // efc_frictionloss_out,                                                                  <L 1312>
        var_106 = wp::load(var_97);
        var_107 = wp::load(var_100);
        var_108 = wp::load(var_101);
        var_109 = wp::load(var_102);
        _efc_row_0(var_opt_disableflags, var_0, var_106, var_15, var_98, var_99, var_107, var_108, var_109, var_103, var_18, var_8, var_105, var_1, var_efc_type_out, var_efc_id_out, var_efc_pos_out, var_efc_margin_out, var_efc_D_out, var_efc_vel_out, var_efc_aref_out, var_efc_frictionloss_out);
    }
}



extern "C" __global__ void _contact_pyramidal_9db164a6_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::int32 var_opt_disableflags,
    wp::array_t<wp::float32> var_opt_impratio_invsqrt,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_weldid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::vec_t<2, wp::float32>> var_body_invweight0,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::int32> var_dof_parentid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_flex_vertadr,
    wp::array_t<wp::int32> var_flex_vertbodyid,
    bool var_is_sparse,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::int32 var_njmax_in,
    wp::int32 var_njmax_nnz_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::float32> var_dist_in,
    wp::array_t<wp::int32> var_condim_in,
    wp::array_t<wp::float32> var_includemargin_in,
    wp::array_t<wp::int32> var_worldid_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_geom_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_flex_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_vert_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_pos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_frame_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_friction_in,
    wp::array_t<wp::vec_t<2, wp::float32>> var_solref_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_solimp_in,
    wp::array_t<wp::int32> var_type_in,
    wp::array_t<wp::int32> var_nefc_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_efc_type_out,
    wp::array_t<wp::int32> var_efc_id_out,
    wp::array_t<wp::int32> var_efc_J_rownnz_out,
    wp::array_t<wp::int32> var_efc_J_rowadr_out,
    wp::array_t<wp::int32> var_efc_J_colind_out,
    wp::array_t<wp::float32> var_efc_J_out,
    wp::array_t<wp::float32> var_efc_pos_out,
    wp::array_t<wp::float32> var_efc_margin_out,
    wp::array_t<wp::float32> var_efc_D_out,
    wp::array_t<wp::float32> var_efc_vel_out,
    wp::array_t<wp::float32> var_efc_aref_out,
    wp::array_t<wp::float32> var_efc_frictionloss_out,
    wp::array_t<wp::int32> var_efc_nnz_out)
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
        wp::int32* var_3;
        bool var_4;
        wp::int32 var_5;
        wp::int32* var_6;
        const wp::int32 var_7 = 1;
        wp::int32 var_8;
        wp::int32 var_9;
        bool var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        const wp::int32 var_14 = 1;
        bool var_15;
        const wp::int32 var_16 = 0;
        bool var_17;
        bool var_18;
        const wp::int32 var_19 = 1;
        bool var_20;
        const wp::int32 var_21 = 2;
        const wp::int32 var_22 = 1;
        wp::int32 var_23;
        wp::int32 var_24;
        bool var_25;
        bool var_26;
        wp::float32* var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        wp::float32* var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        const wp::int32 var_33 = 0;
        bool var_34;
        wp::int32* var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        const wp::int32 var_38 = 1;
        wp::int32 var_39;
        bool var_40;
        const wp::int32 var_41 = 1;
        const wp::int32 var_42 = -1;
        wp::shape_t* var_43;
        const wp::int32 var_44 = 0;
        wp::int32 var_45;
        wp::shape_t var_46;
        wp::int32 var_47;
        wp::float32* var_48;
        wp::float32 var_49;
        wp::float32 var_50;
        wp::shape_t* var_51;
        const wp::int32 var_52 = 0;
        wp::int32 var_53;
        wp::shape_t var_54;
        wp::int32 var_55;
        wp::float32* var_56;
        wp::float32 var_57;
        wp::float32 var_58;
        wp::vec_t<2, wp::int32>* var_59;
        wp::vec_t<2, wp::int32> var_60;
        wp::vec_t<2, wp::int32> var_61;
        const wp::int32 var_62 = 0;
        wp::int32 var_63;
        const wp::int32 var_64 = 0;
        bool var_65;
        const wp::int32 var_66 = 0;
        wp::int32 var_67;
        wp::int32* var_68;
        wp::int32 var_69;
        wp::int32 var_70;
        wp::vec_t<2, wp::int32>* var_71;
        wp::vec_t<2, wp::int32> var_72;
        wp::vec_t<2, wp::int32> var_73;
        wp::vec_t<2, wp::int32>* var_74;
        wp::vec_t<2, wp::int32> var_75;
        wp::vec_t<2, wp::int32> var_76;
        const wp::int32 var_77 = 0;
        wp::int32 var_78;
        wp::int32* var_79;
        const wp::int32 var_80 = 0;
        wp::int32 var_81;
        wp::int32 var_82;
        wp::int32 var_83;
        wp::int32* var_84;
        wp::int32 var_85;
        wp::int32 var_86;
        wp::int32 var_87;
        const wp::int32 var_88 = 1;
        wp::int32 var_89;
        const wp::int32 var_90 = 0;
        bool var_91;
        const wp::int32 var_92 = 1;
        wp::int32 var_93;
        wp::int32* var_94;
        wp::int32 var_95;
        wp::int32 var_96;
        wp::vec_t<2, wp::int32>* var_97;
        wp::vec_t<2, wp::int32> var_98;
        wp::vec_t<2, wp::int32> var_99;
        wp::vec_t<2, wp::int32>* var_100;
        wp::vec_t<2, wp::int32> var_101;
        wp::vec_t<2, wp::int32> var_102;
        const wp::int32 var_103 = 1;
        wp::int32 var_104;
        wp::int32* var_105;
        const wp::int32 var_106 = 1;
        wp::int32 var_107;
        wp::int32 var_108;
        wp::int32 var_109;
        wp::int32* var_110;
        wp::int32 var_111;
        wp::int32 var_112;
        wp::vec_t<2, wp::int32> var_113;
        wp::vec_t<2, wp::int32> var_114;
        wp::int32 var_115;
        wp::vec_t<3, wp::float32>* var_116;
        wp::vec_t<3, wp::float32> var_117;
        wp::vec_t<3, wp::float32> var_118;
        wp::mat_t<3, 3, wp::float32>* var_119;
        wp::mat_t<3, 3, wp::float32> var_120;
        wp::mat_t<3, 3, wp::float32> var_121;
        wp::shape_t* var_122;
        const wp::int32 var_123 = 0;
        wp::int32 var_124;
        wp::shape_t var_125;
        wp::int32 var_126;
        wp::vec_t<2, wp::float32>* var_127;
        const wp::int32 var_128 = 0;
        wp::float32 var_129;
        wp::vec_t<2, wp::float32> var_130;
        wp::vec_t<2, wp::float32>* var_131;
        const wp::int32 var_132 = 0;
        wp::float32 var_133;
        wp::vec_t<2, wp::float32> var_134;
        wp::float32 var_135;
        const wp::int32 var_136 = 1;
        bool var_137;
        const wp::int32 var_138 = 2;
        wp::int32 var_139;
        const wp::int32 var_140 = 1;
        wp::int32 var_141;
        wp::vec_t<5, wp::float32>* var_142;
        wp::vec_t<5, wp::float32> var_143;
        wp::vec_t<5, wp::float32> var_144;
        const wp::int32 var_145 = 0;
        wp::float32 var_146;
        const wp::int32 var_147 = 1;
        wp::int32 var_148;
        wp::float32 var_149;
        wp::float32 var_150;
        wp::float32 var_151;
        wp::float32 var_152;
        const wp::float32 var_153 = 2.0;
        wp::float32 var_154;
        wp::float32 var_155;
        wp::float32 var_156;
        wp::float32 var_157;
        wp::float32 var_158;
        wp::float32 var_159;
        const wp::float32 var_160 = 0.0;
        wp::float32 var_161;
        wp::int32* var_162;
        wp::int32 var_163;
        wp::int32 var_164;
        wp::int32* var_165;
        wp::int32 var_166;
        wp::int32 var_167;
        wp::int32* var_168;
        wp::int32* var_169;
        wp::int32 var_170;
        wp::int32 var_171;
        wp::int32 var_172;
        const wp::int32 var_173 = 1;
        wp::int32 var_174;
        wp::int32 var_175;
        wp::int32* var_176;
        wp::int32* var_177;
        wp::int32 var_178;
        wp::int32 var_179;
        wp::int32 var_180;
        const wp::int32 var_181 = 1;
        wp::int32 var_182;
        wp::int32 var_183;
        wp::int32 var_184;
        wp::int32 var_185;
        const wp::int32 var_186 = 0;
        wp::int32 var_187;
        const wp::int32 var_188 = 0;
        bool var_189;
        const wp::int32 var_190 = 0;
        bool var_191;
        bool var_192;
        wp::int32 var_193;
        bool var_194;
        bool var_195;
        bool var_196;
        bool var_197;
        wp::int32* var_198;
        wp::int32 var_199;
        wp::int32 var_200;
        wp::int32 var_201;
        bool var_202;
        wp::int32* var_203;
        wp::int32 var_204;
        wp::int32 var_205;
        wp::int32 var_206;
        const wp::int32 var_207 = 1;
        wp::int32 var_208;
        wp::int32 var_209;
        wp::int32 var_210;
        bool var_211;
        wp::int32 var_212;
        const wp::int32 var_213 = 0;
        wp::int32 var_214;
        wp::int32 var_215;
        const wp::int32 var_216 = 1;
        wp::int32 var_217;
        wp::int32 var_218;
        wp::int32 var_219;
        const bool var_220 = true;
        bool var_221;
        const wp::int32 var_222 = 0;
        bool var_223;
        bool var_224;
        wp::vec_t<3, wp::float32> var_225;
        wp::vec_t<3, wp::float32> var_226;
        wp::vec_t<3, wp::float32> var_227;
        wp::vec_t<3, wp::float32> var_228;
        const wp::float32 var_229 = 0.0;
        wp::float32 var_230;
        const wp::float32 var_231 = 0.0;
        wp::float32 var_232;
        const wp::int32 var_233 = 1;
        bool var_234;
        const wp::int32 var_235 = 2;
        wp::int32 var_236;
        const wp::int32 var_237 = 1;
        wp::int32 var_238;
        wp::int32 var_239;
        const wp::int32 var_240 = 0;
        wp::float32 var_241;
        wp::float32 var_242;
        wp::float32 var_243;
        const wp::int32 var_244 = 0;
        wp::float32 var_245;
        wp::float32 var_246;
        wp::float32 var_247;
        const wp::int32 var_248 = 1;
        bool var_249;
        const wp::int32 var_250 = 3;
        bool var_251;
        wp::float32 var_252;
        wp::float32 var_253;
        wp::float32 var_254;
        wp::float32 var_255;
        const wp::int32 var_256 = 3;
        wp::int32 var_257;
        wp::float32 var_258;
        wp::float32 var_259;
        wp::float32 var_260;
        wp::float32 var_261;
        wp::float32 var_262;
        wp::float32 var_263;
        wp::float32 var_264;
        wp::float32 var_265;
        const wp::int32 var_266 = 1;
        wp::float32 var_267;
        wp::float32 var_268;
        wp::float32 var_269;
        const wp::int32 var_270 = 0;
        wp::float32 var_271;
        wp::float32 var_272;
        wp::float32 var_273;
        const wp::int32 var_274 = 1;
        bool var_275;
        const wp::int32 var_276 = 3;
        bool var_277;
        wp::float32 var_278;
        wp::float32 var_279;
        wp::float32 var_280;
        wp::float32 var_281;
        const wp::int32 var_282 = 3;
        wp::int32 var_283;
        wp::float32 var_284;
        wp::float32 var_285;
        wp::float32 var_286;
        wp::float32 var_287;
        wp::float32 var_288;
        wp::float32 var_289;
        wp::float32 var_290;
        wp::float32 var_291;
        const wp::int32 var_292 = 2;
        wp::float32 var_293;
        wp::float32 var_294;
        wp::float32 var_295;
        const wp::int32 var_296 = 0;
        wp::float32 var_297;
        wp::float32 var_298;
        wp::float32 var_299;
        const wp::int32 var_300 = 1;
        bool var_301;
        const wp::int32 var_302 = 3;
        bool var_303;
        wp::float32 var_304;
        wp::float32 var_305;
        wp::float32 var_306;
        wp::float32 var_307;
        const wp::int32 var_308 = 3;
        wp::int32 var_309;
        wp::float32 var_310;
        wp::float32 var_311;
        wp::float32 var_312;
        wp::float32 var_313;
        wp::float32 var_314;
        wp::float32 var_315;
        wp::float32 var_316;
        wp::float32 var_317;
        const wp::int32 var_318 = 1;
        bool var_319;
        const wp::int32 var_320 = 2;
        wp::int32 var_321;
        const wp::int32 var_322 = 0;
        bool var_323;
        wp::float32 var_324;
        wp::float32 var_325;
        wp::float32 var_326;
        wp::float32 var_327;
        wp::float32 var_328;
        wp::float32 var_329;
        wp::float32 var_330;
        wp::int32 var_331;
        const wp::int32 var_332 = 0;
        const wp::int32 var_333 = 0;
        const wp::int32 var_334 = 1;
        wp::int32 var_335;
        wp::int32 var_336;
        wp::float32* var_337;
        wp::float32 var_338;
        wp::float32 var_339;
        wp::float32 var_340;
        bool var_341;
        bool var_342;
        wp::int32 var_343;
        wp::float32 var_344;
        wp::int32 var_345;
        bool var_346;
        wp::int32* var_347;
        wp::int32 var_348;
        wp::int32 var_349;
        wp::int32 var_350;
        bool var_351;
        wp::int32* var_352;
        wp::int32 var_353;
        wp::int32 var_354;
        wp::int32 var_355;
        wp::int32 var_356;
        wp::int32 var_357;
        wp::int32 var_358;
        const wp::int32 var_359 = 1;
        wp::int32 var_360;
        wp::int32 var_361;
        wp::int32 var_362;
        wp::float32 var_363;
        wp::int32 var_364;
        wp::int32 var_365;
        wp::int32 var_366;
        wp::int32 var_367;
        wp::int32 var_368;
        bool var_369;
        const wp::float32 var_370 = 0.0;
        const wp::int32 var_371 = 1;
        wp::int32 var_372;
        wp::int32 var_373;
        wp::int32 var_374;
        const wp::int32 var_375 = 1;
        bool var_376;
        const wp::int32 var_377 = 5;
        const wp::int32 var_378 = 6;
        wp::int32 var_379;
        wp::vec_t<2, wp::float32>* var_380;
        wp::vec_t<5, wp::float32>* var_381;
        const wp::float32 var_382 = 0.0;
        wp::vec_t<2, wp::float32> var_383;
        wp::vec_t<5, wp::float32> var_384;
        //---------
        // forward
        // def _contact_pyramidal(                                                                <L 1669>
        // conid, dimid = wp.tid()                                                                <L 1726>
        builtin_tid2d(var_0, var_1);
        // if conid >= nacon_in[0]:                                                               <L 1728>
        var_3 = wp::address(var_nacon_in, var_2);
        var_5 = wp::load(var_3);
        var_4 = (var_0 >= var_5);
        if (var_4) {
            // return                                                                             <L 1729>
            continue;
        }
        // if not type_in[conid] & ContactType.CONSTRAINT:                                        <L 1731>
        var_6 = wp::address(var_type_in, var_0);
        var_9 = wp::load(var_6);
        var_8 = wp::bit_and(var_9, var_7);
        var_10 = wp::unot(var_8);
        if (var_10) {
            // return                                                                             <L 1732>
            continue;
        }
        // condim = condim_in[conid]                                                              <L 1734>
        var_11 = wp::address(var_condim_in, var_0);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // if condim == 1 and dimid > 0:                                                          <L 1736>
        var_15 = (var_12 == var_14);
        var_17 = (var_1 > var_16);
        var_18 = var_15 && var_17;
        if (var_18) {
            // return                                                                             <L 1737>
            continue;
        }
        if (!var_18) {
            // elif condim > 1 and dimid >= 2 * (condim - 1):                                     <L 1738>
            var_20 = (var_12 > var_19);
            var_23 = wp::sub(var_12, var_22);
            var_24 = wp::mul(var_21, var_23);
            var_25 = (var_1 >= var_24);
            var_26 = var_20 && var_25;
            if (var_26) {
                // return                                                                         <L 1739>
                continue;
            }
        }
        // includemargin = includemargin_in[conid]                                                <L 1741>
        var_27 = wp::address(var_includemargin_in, var_0);
        var_29 = wp::load(var_27);
        var_28 = wp::copy(var_29);
        // pos = dist_in[conid] - includemargin                                                   <L 1742>
        var_30 = wp::address(var_dist_in, var_0);
        var_32 = wp::load(var_30);
        var_31 = wp::sub(var_32, var_28);
        // active = pos < 0                                                                       <L 1743>
        var_34 = (var_31 < var_33);
        // if active:                                                                             <L 1745>
        if (var_34) {
            // worldid = worldid_in[conid]                                                        <L 1746>
            var_35 = wp::address(var_worldid_in, var_0);
            var_37 = wp::load(var_35);
            var_36 = wp::copy(var_37);
            // efcid = wp.atomic_add(nefc_out, worldid, 1)                                        <L 1748>
            var_39 = wp::atomic_add(var_nefc_out, var_36, var_38);
            // if efcid >= njmax_in:                                                              <L 1749>
            var_40 = (var_39 >= var_njmax_in);
            if (var_40) {
                // contact_efc_address_out[conid, dimid] = -1                                     <L 1750>
                wp::array_store(var_contact_efc_address_out, var_0, var_1, var_42);
                // return                                                                         <L 1751>
                continue;
            }
            // timestep = opt_timestep[worldid % opt_timestep.shape[0]]                           <L 1753>
            var_43 = &(var_opt_timestep.shape);
            var_46 = wp::load(var_43);
            var_45 = wp::extract(var_46, var_44);
            var_47 = wp::mod(var_36, var_45);
            var_48 = wp::address(var_opt_timestep, var_47);
            var_50 = wp::load(var_48);
            var_49 = wp::copy(var_50);
            // impratio_invsqrt = opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]       <L 1754>
            var_51 = &(var_opt_impratio_invsqrt.shape);
            var_54 = wp::load(var_51);
            var_53 = wp::extract(var_54, var_52);
            var_55 = wp::mod(var_36, var_53);
            var_56 = wp::address(var_opt_impratio_invsqrt, var_55);
            var_58 = wp::load(var_56);
            var_57 = wp::copy(var_58);
            // contact_efc_address_out[conid, dimid] = efcid                                      <L 1755>
            wp::array_store(var_contact_efc_address_out, var_0, var_1, var_39);
            // geom = geom_in[conid]                                                              <L 1757>
            var_59 = wp::address(var_geom_in, var_0);
            var_61 = wp::load(var_59);
            var_60 = wp::copy(var_61);
            // if geom[0] >= 0:                                                                   <L 1759>
            var_63 = wp::extract(var_60, var_62);
            var_65 = (var_63 >= var_64);
            if (var_65) {
                // body1 = geom_bodyid[geom[0]]                                                   <L 1760>
                var_67 = wp::extract(var_60, var_66);
                var_68 = wp::address(var_geom_bodyid, var_67);
                var_70 = wp::load(var_68);
                var_69 = wp::copy(var_70);
            }
            if (!var_65) {
                // flex = flex_in[conid]                                                          <L 1762>
                var_71 = wp::address(var_flex_in, var_0);
                var_73 = wp::load(var_71);
                var_72 = wp::copy(var_73);
                // vert = vert_in[conid]                                                          <L 1763>
                var_74 = wp::address(var_vert_in, var_0);
                var_76 = wp::load(var_74);
                var_75 = wp::copy(var_76);
                // body1 = flex_vertbodyid[flex_vertadr[flex[0]] + vert[0]]                       <L 1764>
                var_78 = wp::extract(var_72, var_77);
                var_79 = wp::address(var_flex_vertadr, var_78);
                var_81 = wp::extract(var_75, var_80);
                var_83 = wp::load(var_79);
                var_82 = wp::add(var_83, var_81);
                var_84 = wp::address(var_flex_vertbodyid, var_82);
                var_86 = wp::load(var_84);
                var_85 = wp::copy(var_86);
            }
            var_87 = wp::where(var_65, var_69, var_85);
            // if geom[1] >= 0:                                                                   <L 1766>
            var_89 = wp::extract(var_60, var_88);
            var_91 = (var_89 >= var_90);
            if (var_91) {
                // body2 = geom_bodyid[geom[1]]                                                   <L 1767>
                var_93 = wp::extract(var_60, var_92);
                var_94 = wp::address(var_geom_bodyid, var_93);
                var_96 = wp::load(var_94);
                var_95 = wp::copy(var_96);
            }
            if (!var_91) {
                // flex = flex_in[conid]                                                          <L 1769>
                var_97 = wp::address(var_flex_in, var_0);
                var_99 = wp::load(var_97);
                var_98 = wp::copy(var_99);
                // vert = vert_in[conid]                                                          <L 1770>
                var_100 = wp::address(var_vert_in, var_0);
                var_102 = wp::load(var_100);
                var_101 = wp::copy(var_102);
                // body2 = flex_vertbodyid[flex_vertadr[flex[1]] + vert[1]]                       <L 1771>
                var_104 = wp::extract(var_98, var_103);
                var_105 = wp::address(var_flex_vertadr, var_104);
                var_107 = wp::extract(var_101, var_106);
                var_109 = wp::load(var_105);
                var_108 = wp::add(var_109, var_107);
                var_110 = wp::address(var_flex_vertbodyid, var_108);
                var_112 = wp::load(var_110);
                var_111 = wp::copy(var_112);
            }
            var_113 = wp::where(var_91, var_72, var_98);
            var_114 = wp::where(var_91, var_75, var_101);
            var_115 = wp::where(var_91, var_95, var_111);
            // con_pos = pos_in[conid]                                                            <L 1773>
            var_116 = wp::address(var_pos_in, var_0);
            var_118 = wp::load(var_116);
            var_117 = wp::copy(var_118);
            // frame = frame_in[conid]                                                            <L 1774>
            var_119 = wp::address(var_frame_in, var_0);
            var_121 = wp::load(var_119);
            var_120 = wp::copy(var_121);
            // body_invweight0_id = worldid % body_invweight0.shape[0]                            <L 1777>
            var_122 = &(var_body_invweight0.shape);
            var_125 = wp::load(var_122);
            var_124 = wp::extract(var_125, var_123);
            var_126 = wp::mod(var_36, var_124);
            // invweight = body_invweight0[body_invweight0_id, body1][0] + body_invweight0[body_invweight0_id, body2][0]       <L 1778>
            var_127 = wp::address(var_body_invweight0, var_126, var_87);
            var_130 = wp::load(var_127);
            var_129 = wp::extract(var_130, var_128);
            var_131 = wp::address(var_body_invweight0, var_126, var_115);
            var_134 = wp::load(var_131);
            var_133 = wp::extract(var_134, var_132);
            var_135 = wp::add(var_129, var_133);
            // if condim > 1:                                                                     <L 1780>
            var_137 = (var_12 > var_136);
            if (var_137) {
                // dimid2 = dimid / 2 + 1                                                         <L 1781>
                var_139 = wp::div(var_1, var_138);
                var_141 = wp::add(var_139, var_140);
                // friction = friction_in[conid]                                                  <L 1783>
                var_142 = wp::address(var_friction_in, var_0);
                var_144 = wp::load(var_142);
                var_143 = wp::copy(var_144);
                // fri0 = friction[0]                                                             <L 1784>
                var_146 = wp::extract(var_143, var_145);
                // frii = friction[dimid2 - 1]                                                    <L 1785>
                var_148 = wp::sub(var_141, var_147);
                var_149 = wp::extract(var_143, var_148);
                // invweight = invweight + fri0 * fri0 * invweight                                <L 1786>
                var_150 = wp::mul(var_146, var_146);
                var_151 = wp::mul(var_150, var_135);
                var_152 = wp::add(var_135, var_151);
                // invweight = invweight * 2.0 * fri0 * fri0 * impratio_invsqrt * impratio_invsqrt       <L 1787>
                var_154 = wp::mul(var_152, var_153);
                var_155 = wp::mul(var_154, var_146);
                var_156 = wp::mul(var_155, var_146);
                var_157 = wp::mul(var_156, var_57);
                var_158 = wp::mul(var_157, var_57);
            }
            var_159 = wp::where(var_137, var_158, var_135);
            // Jqvel = float(0.0)                                                                 <L 1789>
            var_161 = wp::float(var_160);
            // body1 = body_weldid[body1]                                                         <L 1792>
            var_162 = wp::address(var_body_weldid, var_87);
            var_164 = wp::load(var_162);
            var_163 = wp::copy(var_164);
            // body2 = body_weldid[body2]                                                         <L 1793>
            var_165 = wp::address(var_body_weldid, var_115);
            var_167 = wp::load(var_165);
            var_166 = wp::copy(var_167);
            // da1 = int(body_dofadr[body1] + body_dofnum[body1] - 1)                             <L 1795>
            var_168 = wp::address(var_body_dofadr, var_163);
            var_169 = wp::address(var_body_dofnum, var_163);
            var_171 = wp::load(var_168);
            var_172 = wp::load(var_169);
            var_170 = wp::add(var_171, var_172);
            var_174 = wp::sub(var_170, var_173);
            var_175 = wp::int(var_174);
            // da2 = int(body_dofadr[body2] + body_dofnum[body2] - 1)                             <L 1796>
            var_176 = wp::address(var_body_dofadr, var_166);
            var_177 = wp::address(var_body_dofnum, var_166);
            var_179 = wp::load(var_176);
            var_180 = wp::load(var_177);
            var_178 = wp::add(var_179, var_180);
            var_182 = wp::sub(var_178, var_181);
            var_183 = wp::int(var_182);
            // if is_sparse:                                                                      <L 1798>
            if (var_is_sparse) {
                // pda1 = da1                                                                     <L 1799>
                var_184 = wp::copy(var_175);
                // pda2 = da2                                                                     <L 1800>
                var_185 = wp::copy(var_183);
                // rownnz = int(0)                                                                <L 1801>
                var_187 = wp::int(var_186);
                // while pda1 >= 0 or pda2 >= 0:                                                  <L 1802>
        start_while_5:;
                var_189 = (var_184 >= var_188);
                var_191 = (var_185 >= var_190);
                var_192 = var_189 || var_191;
        if ((var_192) == false) goto end_while_5;
                    // da = wp.max(pda1, pda2)                                                    <L 1803>
                    var_193 = wp::max(var_184, var_185);
                    // if pda1 == da and pda2 == da:                                              <L 1805>
                    var_194 = (var_184 == var_193);
                    var_195 = (var_185 == var_193);
                    var_196 = var_194 && var_195;
                    if (var_196) {
                        // break                                                                  <L 1806>
                        goto end_while_5;
                    }
                    // if pda1 == da:                                                             <L 1807>
                    var_197 = (var_184 == var_193);
                    if (var_197) {
                        // pda1 = dof_parentid[pda1]                                              <L 1808>
                        var_198 = wp::address(var_dof_parentid, var_184);
                        var_200 = wp::load(var_198);
                        var_199 = wp::copy(var_200);
                    }
                    var_201 = wp::where(var_197, var_199, var_184);
                    // if pda2 == da:                                                             <L 1809>
                    var_202 = (var_185 == var_193);
                    if (var_202) {
                        // pda2 = dof_parentid[pda2]                                              <L 1810>
                        var_203 = wp::address(var_dof_parentid, var_185);
                        var_205 = wp::load(var_203);
                        var_204 = wp::copy(var_205);
                    }
                    var_206 = wp::where(var_202, var_204, var_185);
                    // rownnz += 1                                                                <L 1811>
                    var_208 = wp::add(var_187, var_207);
                    wp::assign(var_184, var_201);
                    wp::assign(var_185, var_206);
                    wp::assign(var_187, var_208);
        goto start_while_5;
        end_while_5:;
                // rowadr = wp.atomic_add(efc_nnz_out, worldid, rownnz)                           <L 1814>
                var_209 = wp::atomic_add(var_efc_nnz_out, var_36, var_187);
                // if rowadr + rownnz > njmax_nnz_in:                                             <L 1815>
                var_210 = wp::add(var_209, var_187);
                var_211 = (var_210 > var_njmax_nnz_in);
                if (var_211) {
                    // return                                                                     <L 1816>
                    continue;
                }
                // efc_J_rowadr_out[worldid, efcid] = rowadr                                      <L 1817>
                wp::array_store(var_efc_J_rowadr_out, var_36, var_39, var_209);
                // efc_J_rownnz_out[worldid, efcid] = rownnz                                      <L 1818>
                wp::array_store(var_efc_J_rownnz_out, var_36, var_39, var_187);
            }
            // da = wp.max(da1, da2)                                                              <L 1820>
            var_212 = wp::max(var_175, var_183);
            // if is_sparse:                                                                      <L 1822>
            if (var_is_sparse) {
                // nnz = int(0)                                                                   <L 1823>
                var_214 = wp::int(var_213);
                // dofid = int(da)                                                                <L 1824>
                var_215 = wp::int(var_212);
            }
            if (!var_is_sparse) {
                // dofid = int(nv - 1)                                                            <L 1826>
                var_217 = wp::sub(var_nv, var_216);
                var_218 = wp::int(var_217);
            }
            var_219 = wp::where(var_is_sparse, var_215, var_218);
            // while True:                                                                        <L 1828>
        start_while_8:;
        if ((var_220) == false) goto end_while_8;
                // if is_sparse:                                                                  <L 1829>
                if (var_is_sparse) {
                    // if nnz >= rownnz:                                                          <L 1830>
                    var_221 = (var_214 >= var_187);
                    if (var_221) {
                        // break                                                                  <L 1831>
                        goto end_while_8;
                    }
                }
                if (!var_is_sparse) {
                    // if dofid < 0:                                                              <L 1833>
                    var_223 = (var_219 < var_222);
                    if (var_223) {
                        // break                                                                  <L 1834>
                        goto end_while_8;
                    }
                }
                // if dofid == da:                                                                <L 1836>
                var_224 = (var_219 == var_212);
                if (var_224) {
                    // jac1p, jac1r = support.jac_dof(                                            <L 1838>
                    // body_parentid,                                                             <L 1839>
                    // body_rootid,                                                               <L 1840>
                    // dof_bodyid,                                                                <L 1841>
                    // subtree_com_in,                                                            <L 1842>
                    // cdof_in,                                                                   <L 1843>
                    // con_pos,                                                                   <L 1844>
                    // body1,                                                                     <L 1845>
                    // dofid,                                                                     <L 1846>
                    // worldid,                                                                   <L 1847>
                    jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_subtree_com_in, var_cdof_in, var_117, var_163, var_219, var_36, var_225, var_226);
                    // jac2p, jac2r = support.jac_dof(                                            <L 1849>
                    // body_parentid,                                                             <L 1850>
                    // body_rootid,                                                               <L 1851>
                    // dof_bodyid,                                                                <L 1852>
                    // subtree_com_in,                                                            <L 1853>
                    // cdof_in,                                                                   <L 1854>
                    // con_pos,                                                                   <L 1855>
                    // body2,                                                                     <L 1856>
                    // dofid,                                                                     <L 1857>
                    // worldid,                                                                   <L 1858>
                    jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_subtree_com_in, var_cdof_in, var_117, var_166, var_219, var_36, var_227, var_228);
                    // J = float(0.0)                                                             <L 1861>
                    var_230 = wp::float(var_229);
                    // Ji = float(0.0)                                                            <L 1862>
                    var_232 = wp::float(var_231);
                    // if condim > 1:                                                             <L 1863>
                    var_234 = (var_12 > var_233);
                    if (var_234) {
                        // dimid2 = dimid / 2 + 1                                                 <L 1864>
                        var_236 = wp::div(var_1, var_235);
                        var_238 = wp::add(var_236, var_237);
                    }
                    var_239 = wp::where(var_234, var_238, var_141);
                    // for xyz in range(3):                                                       <L 1866>
                    // jacp_dif = jac2p[xyz] - jac1p[xyz]                                         <L 1867>
                    var_241 = wp::extract(var_227, var_240);
                    var_242 = wp::extract(var_225, var_240);
                    var_243 = wp::sub(var_241, var_242);
                    // J += frame[0, xyz] * jacp_dif                                              <L 1868>
                    var_245 = wp::extract(var_120, var_244, var_240);
                    var_246 = wp::mul(var_245, var_243);
                    var_247 = wp::add(var_230, var_246);
                    // if condim > 1:                                                             <L 1870>
                    var_249 = (var_12 > var_248);
                    if (var_249) {
                        // if dimid2 < 3:                                                         <L 1871>
                        var_251 = (var_239 < var_250);
                        if (var_251) {
                            // Ji += frame[dimid2, xyz] * jacp_dif                                <L 1872>
                            var_252 = wp::extract(var_120, var_239, var_240);
                            var_253 = wp::mul(var_252, var_243);
                            var_254 = wp::add(var_232, var_253);
                        }
                        var_255 = wp::where(var_251, var_254, var_232);
                        if (!var_251) {
                            // Ji += frame[dimid2 - 3, xyz] * (jac2r[xyz] - jac1r[xyz])           <L 1874>
                            var_257 = wp::sub(var_239, var_256);
                            var_258 = wp::extract(var_120, var_257, var_240);
                            var_259 = wp::extract(var_228, var_240);
                            var_260 = wp::extract(var_226, var_240);
                            var_261 = wp::sub(var_259, var_260);
                            var_262 = wp::mul(var_258, var_261);
                            var_263 = wp::add(var_255, var_262);
                        }
                        var_264 = wp::where(var_251, var_255, var_263);
                    }
                    var_265 = wp::where(var_249, var_264, var_232);
                    // jacp_dif = jac2p[xyz] - jac1p[xyz]                                         <L 1867>
                    var_267 = wp::extract(var_227, var_266);
                    var_268 = wp::extract(var_225, var_266);
                    var_269 = wp::sub(var_267, var_268);
                    // J += frame[0, xyz] * jacp_dif                                              <L 1868>
                    var_271 = wp::extract(var_120, var_270, var_266);
                    var_272 = wp::mul(var_271, var_269);
                    var_273 = wp::add(var_247, var_272);
                    // if condim > 1:                                                             <L 1870>
                    var_275 = (var_12 > var_274);
                    if (var_275) {
                        // if dimid2 < 3:                                                         <L 1871>
                        var_277 = (var_239 < var_276);
                        if (var_277) {
                            // Ji += frame[dimid2, xyz] * jacp_dif                                <L 1872>
                            var_278 = wp::extract(var_120, var_239, var_266);
                            var_279 = wp::mul(var_278, var_269);
                            var_280 = wp::add(var_265, var_279);
                        }
                        var_281 = wp::where(var_277, var_280, var_265);
                        if (!var_277) {
                            // Ji += frame[dimid2 - 3, xyz] * (jac2r[xyz] - jac1r[xyz])           <L 1874>
                            var_283 = wp::sub(var_239, var_282);
                            var_284 = wp::extract(var_120, var_283, var_266);
                            var_285 = wp::extract(var_228, var_266);
                            var_286 = wp::extract(var_226, var_266);
                            var_287 = wp::sub(var_285, var_286);
                            var_288 = wp::mul(var_284, var_287);
                            var_289 = wp::add(var_281, var_288);
                        }
                        var_290 = wp::where(var_277, var_281, var_289);
                    }
                    var_291 = wp::where(var_275, var_290, var_265);
                    // jacp_dif = jac2p[xyz] - jac1p[xyz]                                         <L 1867>
                    var_293 = wp::extract(var_227, var_292);
                    var_294 = wp::extract(var_225, var_292);
                    var_295 = wp::sub(var_293, var_294);
                    // J += frame[0, xyz] * jacp_dif                                              <L 1868>
                    var_297 = wp::extract(var_120, var_296, var_292);
                    var_298 = wp::mul(var_297, var_295);
                    var_299 = wp::add(var_273, var_298);
                    // if condim > 1:                                                             <L 1870>
                    var_301 = (var_12 > var_300);
                    if (var_301) {
                        // if dimid2 < 3:                                                         <L 1871>
                        var_303 = (var_239 < var_302);
                        if (var_303) {
                            // Ji += frame[dimid2, xyz] * jacp_dif                                <L 1872>
                            var_304 = wp::extract(var_120, var_239, var_292);
                            var_305 = wp::mul(var_304, var_295);
                            var_306 = wp::add(var_291, var_305);
                        }
                        var_307 = wp::where(var_303, var_306, var_291);
                        if (!var_303) {
                            // Ji += frame[dimid2 - 3, xyz] * (jac2r[xyz] - jac1r[xyz])           <L 1874>
                            var_309 = wp::sub(var_239, var_308);
                            var_310 = wp::extract(var_120, var_309, var_292);
                            var_311 = wp::extract(var_228, var_292);
                            var_312 = wp::extract(var_226, var_292);
                            var_313 = wp::sub(var_311, var_312);
                            var_314 = wp::mul(var_310, var_313);
                            var_315 = wp::add(var_307, var_314);
                        }
                        var_316 = wp::where(var_303, var_307, var_315);
                    }
                    var_317 = wp::where(var_301, var_316, var_291);
                    // if condim > 1:                                                             <L 1876>
                    var_319 = (var_12 > var_318);
                    if (var_319) {
                        // if dimid % 2 == 0:                                                     <L 1877>
                        var_321 = wp::mod(var_1, var_320);
                        var_323 = (var_321 == var_322);
                        if (var_323) {
                            // J += Ji * frii                                                     <L 1878>
                            var_324 = wp::mul(var_317, var_149);
                            var_325 = wp::add(var_299, var_324);
                        }
                        var_326 = wp::where(var_323, var_325, var_299);
                        if (!var_323) {
                            // J -= Ji * frii                                                     <L 1880>
                            var_327 = wp::mul(var_317, var_149);
                            var_328 = wp::sub(var_326, var_327);
                        }
                        var_329 = wp::where(var_323, var_326, var_328);
                    }
                    var_330 = wp::where(var_319, var_329, var_299);
                    // if is_sparse:                                                              <L 1882>
                    if (var_is_sparse) {
                        // sparseid = rowadr + nnz                                                <L 1883>
                        var_331 = wp::add(var_209, var_214);
                        // efc_J_colind_out[worldid, 0, sparseid] = dofid                         <L 1884>
                        wp::array_store(var_efc_J_colind_out, var_36, var_332, var_331, var_219);
                        // efc_J_out[worldid, 0, sparseid] = J                                    <L 1885>
                        wp::array_store(var_efc_J_out, var_36, var_333, var_331, var_330);
                        // nnz += 1                                                               <L 1886>
                        var_335 = wp::add(var_214, var_334);
                    }
                    var_336 = wp::where(var_is_sparse, var_335, var_214);
                    if (!var_is_sparse) {
                        // efc_J_out[worldid, efcid, dofid] = J                                   <L 1888>
                        wp::array_store(var_efc_J_out, var_36, var_39, var_219, var_330);
                    }
                    // Jqvel += J * qvel_in[worldid, dofid]                                       <L 1889>
                    var_337 = wp::address(var_qvel_in, var_36, var_219);
                    var_339 = wp::load(var_337);
                    var_338 = wp::mul(var_330, var_339);
                    var_340 = wp::add(var_161, var_338);
                    // if is_sparse and nnz >= rownnz:                                            <L 1890>
                    var_341 = (var_336 >= var_187);
                    var_342 = var_is_sparse && var_341;
                    if (var_342) {
                        // break                                                                  <L 1891>
                        wp::assign(var_141, var_239);
                        wp::assign(var_161, var_340);
                        wp::assign(var_214, var_336);
                        goto end_while_8;
                    }
                    var_343 = wp::where(var_342, var_141, var_239);
                    var_344 = wp::where(var_342, var_161, var_340);
                    var_345 = wp::where(var_342, var_214, var_336);
                    // if da1 == da:                                                              <L 1894>
                    var_346 = (var_175 == var_212);
                    if (var_346) {
                        // da1 = dof_parentid[da1]                                                <L 1895>
                        var_347 = wp::address(var_dof_parentid, var_175);
                        var_349 = wp::load(var_347);
                        var_348 = wp::copy(var_349);
                    }
                    var_350 = wp::where(var_346, var_348, var_175);
                    // if da2 == da:                                                              <L 1896>
                    var_351 = (var_183 == var_212);
                    if (var_351) {
                        // da2 = dof_parentid[da2]                                                <L 1897>
                        var_352 = wp::address(var_dof_parentid, var_183);
                        var_354 = wp::load(var_352);
                        var_353 = wp::copy(var_354);
                    }
                    var_355 = wp::where(var_351, var_353, var_183);
                    // da = wp.max(da1, da2)                                                      <L 1898>
                    var_356 = wp::max(var_350, var_355);
                    // if is_sparse:                                                              <L 1899>
                    if (var_is_sparse) {
                        // dofid = da                                                             <L 1900>
                        var_357 = wp::copy(var_356);
                    }
                    var_358 = wp::where(var_is_sparse, var_357, var_219);
                    if (!var_is_sparse) {
                        // dofid -= 1                                                             <L 1902>
                        var_360 = wp::sub(var_358, var_359);
                    }
                    var_361 = wp::where(var_is_sparse, var_358, var_360);
                }
                var_362 = wp::where(var_224, var_343, var_141);
                var_363 = wp::where(var_224, var_344, var_161);
                var_364 = wp::where(var_224, var_350, var_175);
                var_365 = wp::where(var_224, var_355, var_183);
                var_366 = wp::where(var_224, var_356, var_212);
                var_367 = wp::where(var_224, var_345, var_214);
                var_368 = wp::where(var_224, var_361, var_219);
                if (!var_224) {
                    // if not is_sparse:                                                          <L 1904>
                    var_369 = wp::unot(var_is_sparse);
                    if (var_369) {
                        // efc_J_out[worldid, efcid, dofid] = 0.0                                 <L 1905>
                        wp::array_store(var_efc_J_out, var_36, var_39, var_368, var_370);
                        // dofid -= 1                                                             <L 1906>
                        var_372 = wp::sub(var_368, var_371);
                    }
                    var_373 = wp::where(var_369, var_372, var_368);
                }
                var_374 = wp::where(var_224, var_368, var_373);
                wp::assign(var_141, var_362);
                wp::assign(var_161, var_363);
                wp::assign(var_175, var_364);
                wp::assign(var_183, var_365);
                wp::assign(var_212, var_366);
                wp::assign(var_214, var_367);
                wp::assign(var_219, var_374);
        goto start_while_8;
        end_while_8:;
            // if condim == 1:                                                                    <L 1908>
            var_376 = (var_12 == var_375);
            if (var_376) {
                // efc_type = ConstraintType.CONTACT_FRICTIONLESS                                 <L 1909>
            }
            if (!var_376) {
                // efc_type = ConstraintType.CONTACT_PYRAMIDAL                                    <L 1911>
            }
            var_379 = wp::where(var_376, var_377, var_378);
            // _efc_row(                                                                          <L 1913>
            // opt_disableflags,                                                                  <L 1914>
            // worldid,                                                                           <L 1915>
            // timestep,                                                                          <L 1916>
            // efcid,                                                                             <L 1917>
            // pos,                                                                               <L 1918>
            // pos,                                                                               <L 1919>
            // invweight,                                                                         <L 1920>
            // solref_in[conid],                                                                  <L 1921>
            var_380 = wp::address(var_solref_in, var_0);
            // solimp_in[conid],                                                                  <L 1922>
            var_381 = wp::address(var_solimp_in, var_0);
            // includemargin,                                                                     <L 1923>
            // Jqvel,                                                                             <L 1924>
            // 0.0,                                                                               <L 1925>
            // efc_type,                                                                          <L 1926>
            // conid,                                                                             <L 1927>
            // efc_type_out,                                                                      <L 1928>
            // efc_id_out,                                                                        <L 1929>
            // efc_pos_out,                                                                       <L 1930>
            // efc_margin_out,                                                                    <L 1931>
            // efc_D_out,                                                                         <L 1932>
            // efc_vel_out,                                                                       <L 1933>
            // efc_aref_out,                                                                      <L 1934>
            // efc_frictionloss_out,                                                              <L 1935>
            var_383 = wp::load(var_380);
            var_384 = wp::load(var_381);
            _efc_row_0(var_opt_disableflags, var_36, var_49, var_39, var_31, var_31, var_159, var_383, var_384, var_28, var_161, var_382, var_379, var_0, var_efc_type_out, var_efc_id_out, var_efc_pos_out, var_efc_margin_out, var_efc_D_out, var_efc_vel_out, var_efc_aref_out, var_efc_frictionloss_out);
        }
    }
}

