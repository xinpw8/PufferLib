
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:177
static CUDA_CALLABLE wp::vec_t<3, wp::float32> quat_sub_0(
    wp::quat_t<wp::float32> var_qa,
    wp::quat_t<wp::float32> var_qb)
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
    wp::quat_t<wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    //---------
    // forward
    // def quat_sub(qa: wp.quat, qb: wp.quat) -> wp.vec3:                                     <L 178>
    // qneg = wp.quat(qb[0], -qb[1], -qb[2], -qb[3])                                          <L 181>
    var_1 = wp::extract(var_qb, var_0);
    var_3 = wp::extract(var_qb, var_2);
    var_4 = wp::neg(var_3);
    var_6 = wp::extract(var_qb, var_5);
    var_7 = wp::neg(var_6);
    var_9 = wp::extract(var_qb, var_8);
    var_10 = wp::neg(var_9);
    var_11 = wp::quat_t<wp::float32>(var_1, var_4, var_7, var_10);
    // qdif = mul_quat(qneg, qa)                                                              <L 182>
    var_12 = mul_quat_0(var_11, var_qa);
    // return quat_to_vel(qdif)                                                               <L 185>
    var_13 = quat_to_vel_0(var_12);
    return var_13;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/passive.py:42
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _geom_semiaxes_0(
    wp::vec_t<3, wp::float32> var_size,
    wp::int32 var_geom_type)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 2;
    bool var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    wp::vec_t<3, wp::float32> var_4;
    const wp::int32 var_5 = 3;
    bool var_6;
    const wp::int32 var_7 = 0;
    wp::float32 var_8;
    const wp::int32 var_9 = 1;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::vec_t<3, wp::float32> var_12;
    const wp::int32 var_13 = 5;
    bool var_14;
    const wp::int32 var_15 = 0;
    wp::float32 var_16;
    const wp::int32 var_17 = 1;
    wp::float32 var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    //---------
    // forward
    // def _geom_semiaxes(size: wp.vec3, geom_type: int) -> wp.vec3:  # kernel_analyzer: ignore       <L 43>
    // if geom_type == GeomType.SPHERE:                                                       <L 44>
    var_1 = (var_geom_type == var_0);
    if (var_1) {
        // r = size[0]                                                                        <L 45>
        var_3 = wp::extract(var_size, var_2);
        // return wp.vec3(r, r, r)                                                            <L 46>
        var_4 = wp::vec_t<3, wp::float32>(var_3, var_3, var_3);
        return var_4;
    }
    // if geom_type == GeomType.CAPSULE:                                                      <L 48>
    var_6 = (var_geom_type == var_5);
    if (var_6) {
        // radius = size[0]                                                                   <L 49>
        var_8 = wp::extract(var_size, var_7);
        // half_length = size[1]                                                              <L 50>
        var_10 = wp::extract(var_size, var_9);
        // return wp.vec3(radius, radius, half_length + radius)                               <L 51>
        var_11 = wp::add(var_10, var_8);
        var_12 = wp::vec_t<3, wp::float32>(var_8, var_8, var_11);
        return var_12;
    }
    // if geom_type == GeomType.CYLINDER:                                                     <L 53>
    var_14 = (var_geom_type == var_13);
    if (var_14) {
        // radius = size[0]                                                                   <L 54>
        var_16 = wp::extract(var_size, var_15);
        // half_length = size[1]                                                              <L 55>
        var_18 = wp::extract(var_size, var_17);
        // return wp.vec3(radius, radius, half_length)                                        <L 56>
        var_19 = wp::vec_t<3, wp::float32>(var_16, var_16, var_18);
        return var_19;
    }
    var_20 = wp::where(var_14, var_16, var_8);
    var_21 = wp::where(var_14, var_18, var_10);
    // return size                                                                            <L 59>
    return var_size;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/passive.py:36
static CUDA_CALLABLE wp::float32 _pow4_0(
    wp::float32 var_val)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::float32 var_1;
    //---------
    // forward
    // def _pow4(val: float) -> float:                                                        <L 37>
    // sq = val * val                                                                         <L 38>
    var_0 = wp::mul(var_val, var_val);
    // return sq * sq                                                                         <L 39>
    var_1 = wp::mul(var_0, var_0);
    return var_1;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/passive.py:31
static CUDA_CALLABLE wp::float32 _pow2_0(
    wp::float32 var_val)
{
    //---------
    // primal vars
    wp::float32 var_0;
    //---------
    // forward
    // def _pow2(val: float) -> float:                                                        <L 32>
    // return val * val                                                                       <L 33>
    var_0 = wp::mul(var_val, var_val);
    return var_0;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/passive.py:62
static CUDA_CALLABLE wp::float32 _ellipsoid_max_moment_0(
    wp::vec_t<3, wp::float32> var_size,
    wp::int32 var_dir)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::int32 var_1 = 1;
    wp::int32 var_2;
    const wp::int32 var_3 = 3;
    wp::int32 var_4;
    wp::float32 var_5;
    const wp::int32 var_6 = 2;
    wp::int32 var_7;
    const wp::int32 var_8 = 3;
    wp::int32 var_9;
    wp::float32 var_10;
    const wp::float32 var_11 = 1.6755160819145563;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    //---------
    // forward
    // def _ellipsoid_max_moment(size: wp.vec3, dir: int) -> float:                           <L 63>
    // d0 = size[dir]                                                                         <L 64>
    var_0 = wp::extract(var_size, var_dir);
    // d1 = size[(dir + 1) % 3]                                                               <L 65>
    var_2 = wp::add(var_dir, var_1);
    var_4 = wp::mod(var_2, var_3);
    var_5 = wp::extract(var_size, var_4);
    // d2 = size[(dir + 2) % 3]                                                               <L 66>
    var_7 = wp::add(var_dir, var_6);
    var_9 = wp::mod(var_7, var_8);
    var_10 = wp::extract(var_size, var_9);
    // return wp.static(8.0 / 15.0 * wp.pi) * d0 * _pow4(wp.max(d1, d2))                      <L 67>
    var_12 = wp::mul(var_11, var_0);
    var_13 = wp::max(var_5, var_10);
    var_14 = _pow4_0(var_13);
    var_15 = wp::mul(var_12, var_14);
    return var_15;
}


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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:177
static CUDA_CALLABLE void adj_quat_sub_0(
    wp::quat_t<wp::float32> var_qa,
    wp::quat_t<wp::float32> var_qb,
    wp::quat_t<wp::float32> & adj_qa,
    wp::quat_t<wp::float32> & adj_qb,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/passive.py:42
static CUDA_CALLABLE void adj__geom_semiaxes_0(
    wp::vec_t<3, wp::float32> var_size,
    wp::int32 var_geom_type,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::int32 & adj_geom_type,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/passive.py:36
static CUDA_CALLABLE void adj__pow4_0(
    wp::float32 var_val,
    wp::float32 & adj_val,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/passive.py:31
static CUDA_CALLABLE void adj__pow2_0(
    wp::float32 var_val,
    wp::float32 & adj_val,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/passive.py:62
static CUDA_CALLABLE void adj__ellipsoid_max_moment_0(
    wp::vec_t<3, wp::float32> var_size,
    wp::int32 var_dir,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::int32 & adj_dir,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
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



extern "C" __global__ void _flex_elasticity_e86c04ab_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_nflex,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_flex_dim,
    wp::array_t<wp::int32> var_flex_vertadr,
    wp::array_t<wp::int32> var_flex_edgeadr,
    wp::array_t<wp::int32> var_flex_elemadr,
    wp::array_t<wp::int32> var_flex_elemnum,
    wp::array_t<wp::int32> var_flex_elemdataadr,
    wp::array_t<wp::int32> var_flex_elemedgeadr,
    wp::array_t<wp::int32> var_flex_vertbodyid,
    wp::array_t<wp::int32> var_flex_elem,
    wp::array_t<wp::int32> var_flex_elemedge,
    wp::array_t<wp::float32> var_flexedge_length0,
    wp::array_t<wp::float32> var_flex_stiffness,
    wp::array_t<wp::float32> var_flex_damping,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexvert_xpos_in,
    wp::array_t<wp::float32> var_flexedge_length_in,
    wp::array_t<wp::float32> var_flexedge_velocity_in,
    bool var_dsbl_damper,
    wp::array_t<wp::float32> var_qfrc_spring_out)
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
        wp::range_t var_10;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        const wp::int32 var_15 = 0;
        bool var_16;
        wp::int32* var_17;
        bool var_18;
        wp::int32 var_19;
        bool var_20;
        wp::int32 var_21;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        wp::int32* var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        const wp::int32 var_28 = 1;
        wp::int32 var_29;
        const wp::int32 var_30 = 1;
        wp::int32 var_31;
        wp::int32 var_32;
        const wp::int32 var_33 = 2;
        wp::int32 var_34;
        const wp::int32 var_35 = 1;
        bool var_36;
        const wp::int32 var_37 = 0;
        const wp::int32 var_38 = 1;
        const wp::int32 var_39 = 0;
        const wp::int32 var_40 = 0;
        const wp::int32 var_41 = 0;
        const wp::int32 var_42 = 0;
        const wp::int32 var_43 = 0;
        const wp::int32 var_44 = 0;
        const wp::int32 var_45 = 0;
        const wp::int32 var_46 = 0;
        const wp::int32 var_47 = 0;
        const wp::int32 var_48 = 0;
        const wp::int32 var_49 = 6;
        const wp::int32 var_50 = 2;
        wp::tuple_t<wp::int32, wp::int32> var_51;
        wp::mat_t<6, 2, wp::int32> var_52;
        const wp::int32 var_53 = 3;
        bool var_54;
        const wp::int32 var_55 = 0;
        const wp::int32 var_56 = 1;
        const wp::int32 var_57 = 1;
        const wp::int32 var_58 = 2;
        const wp::int32 var_59 = 2;
        const wp::int32 var_60 = 0;
        const wp::int32 var_61 = 2;
        const wp::int32 var_62 = 3;
        const wp::int32 var_63 = 0;
        const wp::int32 var_64 = 3;
        const wp::int32 var_65 = 1;
        const wp::int32 var_66 = 3;
        const wp::int32 var_67 = 6;
        const wp::int32 var_68 = 2;
        wp::tuple_t<wp::int32, wp::int32> var_69;
        wp::mat_t<6, 2, wp::int32> var_70;
        const wp::int32 var_71 = 1;
        const wp::int32 var_72 = 2;
        const wp::int32 var_73 = 2;
        const wp::int32 var_74 = 0;
        const wp::int32 var_75 = 0;
        const wp::int32 var_76 = 1;
        const wp::int32 var_77 = 0;
        const wp::int32 var_78 = 0;
        const wp::int32 var_79 = 0;
        const wp::int32 var_80 = 0;
        const wp::int32 var_81 = 0;
        const wp::int32 var_82 = 0;
        const wp::int32 var_83 = 6;
        const wp::int32 var_84 = 2;
        wp::tuple_t<wp::int32, wp::int32> var_85;
        wp::mat_t<6, 2, wp::int32> var_86;
        wp::mat_t<6, 2, wp::int32> var_87;
        wp::mat_t<6, 2, wp::int32> var_88;
        const wp::float32 var_89 = 0.0;
        bool var_90;
        bool var_91;
        bool var_92;
        wp::float32* var_93;
        wp::float32 var_94;
        wp::float32 var_95;
        const wp::float32 var_96 = 0.0;
        wp::float32 var_97;
        wp::int32* var_98;
        const wp::int32 var_99 = 1;
        wp::int32 var_100;
        wp::int32 var_101;
        wp::int32 var_102;
        wp::int32 var_103;
        wp::int32* var_104;
        wp::int32 var_105;
        wp::int32 var_106;
        const wp::float32 var_107 = 0.0;
        const wp::int32 var_108 = 6;
        const wp::int32 var_109 = 6;
        wp::tuple_t<wp::int32, wp::int32> var_110;
        wp::mat_t<6, 6, wp::float32> var_111;
        wp::range_t var_112;
        wp::int32 var_113;
        const wp::int32 var_114 = 0;
        wp::int32 var_115;
        wp::int32 var_116;
        wp::int32* var_117;
        wp::int32 var_118;
        wp::int32 var_119;
        const wp::int32 var_120 = 1;
        wp::int32 var_121;
        wp::int32 var_122;
        wp::int32* var_123;
        wp::int32 var_124;
        wp::int32 var_125;
        wp::int32 var_126;
        wp::vec_t<3, wp::float32>* var_127;
        wp::vec_t<3, wp::float32> var_128;
        wp::vec_t<3, wp::float32> var_129;
        wp::int32 var_130;
        wp::vec_t<3, wp::float32>* var_131;
        wp::vec_t<3, wp::float32> var_132;
        wp::vec_t<3, wp::float32> var_133;
        const wp::int32 var_134 = 0;
        wp::float32 var_135;
        wp::float32 var_136;
        wp::float32 var_137;
        const wp::int32 var_138 = 0;
        wp::int32 var_139;
        wp::float32 var_140;
        wp::float32 var_141;
        wp::float32 var_142;
        const wp::int32 var_143 = 3;
        wp::int32 var_144;
        const wp::int32 var_145 = 1;
        wp::float32 var_146;
        wp::float32 var_147;
        wp::float32 var_148;
        const wp::int32 var_149 = 0;
        wp::int32 var_150;
        wp::float32 var_151;
        wp::float32 var_152;
        wp::float32 var_153;
        const wp::int32 var_154 = 3;
        wp::int32 var_155;
        const wp::int32 var_156 = 2;
        wp::float32 var_157;
        wp::float32 var_158;
        wp::float32 var_159;
        const wp::int32 var_160 = 0;
        wp::int32 var_161;
        wp::float32 var_162;
        wp::float32 var_163;
        wp::float32 var_164;
        const wp::int32 var_165 = 3;
        wp::int32 var_166;
        const wp::float32 var_167 = 0.0;
        wp::vec_t<6, wp::float32> var_168;
        wp::range_t var_169;
        wp::int32 var_170;
        wp::int32* var_171;
        wp::int32 var_172;
        wp::int32 var_173;
        wp::int32 var_174;
        wp::int32 var_175;
        wp::int32* var_176;
        wp::int32 var_177;
        wp::int32 var_178;
        wp::int32* var_179;
        wp::int32 var_180;
        wp::int32 var_181;
        wp::float32* var_182;
        wp::float32 var_183;
        wp::float32 var_184;
        wp::int32* var_185;
        wp::int32 var_186;
        wp::int32 var_187;
        wp::float32* var_188;
        wp::float32 var_189;
        wp::float32 var_190;
        wp::int32* var_191;
        wp::int32 var_192;
        wp::int32 var_193;
        wp::float32* var_194;
        wp::float32 var_195;
        wp::float32 var_196;
        wp::float32 var_197;
        wp::float32 var_198;
        wp::float32 var_199;
        wp::float32 var_200;
        wp::float32 var_201;
        wp::float32 var_202;
        wp::float32 var_203;
        wp::float32 var_204;
        wp::float32 var_205;
        wp::float32 var_206;
        const wp::float32 var_207 = 0.0;
        const wp::int32 var_208 = 6;
        const wp::int32 var_209 = 6;
        wp::tuple_t<wp::int32, wp::int32> var_210;
        wp::mat_t<6, 6, wp::float32> var_211;
        const wp::int32 var_212 = 0;
        wp::int32 var_213;
        wp::range_t var_214;
        wp::int32 var_215;
        wp::range_t var_216;
        wp::int32 var_217;
        wp::float32* var_218;
        wp::float32 var_219;
        wp::float32* var_220;
        wp::float32 var_221;
        const wp::int32 var_222 = 1;
        wp::int32 var_223;
        const wp::float32 var_224 = 0.0;
        const wp::int32 var_225 = 6;
        const wp::int32 var_226 = 3;
        wp::tuple_t<wp::int32, wp::int32> var_227;
        wp::mat_t<6, 3, wp::float32> var_228;
        wp::range_t var_229;
        wp::int32 var_230;
        wp::range_t var_231;
        wp::int32 var_232;
        const wp::int32 var_233 = 0;
        const wp::int32 var_234 = 0;
        wp::float32 var_235;
        const wp::int32 var_236 = 3;
        wp::int32 var_237;
        wp::int32 var_238;
        wp::float32 var_239;
        wp::float32 var_240;
        wp::float32 var_241;
        wp::float32 var_242;
        wp::int32 var_243;
        const wp::int32 var_244 = 1;
        wp::float32 var_245;
        const wp::int32 var_246 = 3;
        wp::int32 var_247;
        wp::int32 var_248;
        wp::float32 var_249;
        wp::float32 var_250;
        wp::float32 var_251;
        wp::float32 var_252;
        wp::int32 var_253;
        const wp::int32 var_254 = 2;
        wp::float32 var_255;
        const wp::int32 var_256 = 3;
        wp::int32 var_257;
        wp::int32 var_258;
        wp::float32 var_259;
        wp::float32 var_260;
        wp::float32 var_261;
        wp::float32 var_262;
        wp::int32 var_263;
        const wp::int32 var_264 = 1;
        const wp::int32 var_265 = 0;
        wp::float32 var_266;
        const wp::int32 var_267 = 3;
        wp::int32 var_268;
        wp::int32 var_269;
        wp::float32 var_270;
        wp::float32 var_271;
        wp::float32 var_272;
        wp::float32 var_273;
        wp::int32 var_274;
        const wp::int32 var_275 = 1;
        wp::float32 var_276;
        const wp::int32 var_277 = 3;
        wp::int32 var_278;
        wp::int32 var_279;
        wp::float32 var_280;
        wp::float32 var_281;
        wp::float32 var_282;
        wp::float32 var_283;
        wp::int32 var_284;
        const wp::int32 var_285 = 2;
        wp::float32 var_286;
        const wp::int32 var_287 = 3;
        wp::int32 var_288;
        wp::int32 var_289;
        wp::float32 var_290;
        wp::float32 var_291;
        wp::float32 var_292;
        wp::float32 var_293;
        wp::int32 var_294;
        wp::range_t var_295;
        wp::int32 var_296;
        wp::int32 var_297;
        wp::int32* var_298;
        wp::int32 var_299;
        wp::int32 var_300;
        wp::int32* var_301;
        wp::int32 var_302;
        wp::int32 var_303;
        wp::int32* var_304;
        wp::int32 var_305;
        wp::int32 var_306;
        const wp::int32 var_307 = 0;
        wp::int32* var_308;
        wp::int32 var_309;
        wp::int32 var_310;
        wp::float32 var_311;
        wp::float32 var_312;
        const wp::int32 var_313 = 1;
        wp::int32* var_314;
        wp::int32 var_315;
        wp::int32 var_316;
        wp::float32 var_317;
        wp::float32 var_318;
        const wp::int32 var_319 = 2;
        wp::int32* var_320;
        wp::int32 var_321;
        wp::int32 var_322;
        wp::float32 var_323;
        wp::float32 var_324;
        //---------
        // forward
        // def _flex_elasticity(                                                                  <L 567>
        // worldid, elemid = wp.tid()                                                             <L 594>
        builtin_tid2d(var_0, var_1);
        // timestep = opt_timestep[worldid % opt_timestep.shape[0]]                               <L 595>
        var_2 = &(var_opt_timestep.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_opt_timestep, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // for i in range(nflex):                                                                 <L 597>
        var_10 = wp::range(var_nflex);
        start_for_0:;
            if (iter_cmp(var_10) == 0) goto end_for_0;
            var_11 = wp::iter_next(var_10);
            // locid = elemid - flex_elemadr[i]                                                   <L 598>
            var_12 = wp::address(var_flex_elemadr, var_11);
            var_14 = wp::load(var_12);
            var_13 = wp::sub(var_1, var_14);
            // if locid >= 0 and locid < flex_elemnum[i]:                                         <L 599>
            var_16 = (var_13 >= var_15);
            var_17 = wp::address(var_flex_elemnum, var_11);
            var_19 = wp::load(var_17);
            var_18 = (var_13 < var_19);
            var_20 = var_16 && var_18;
            if (var_20) {
                // f = i                                                                          <L 600>
                var_21 = wp::copy(var_11);
                // break                                                                          <L 601>
                goto end_for_0;
            }
            goto start_for_0;
        end_for_0:;
        // local_elemid = elemid - flex_elemadr[f]                                                <L 603>
        var_22 = wp::address(var_flex_elemadr, var_21);
        var_24 = wp::load(var_22);
        var_23 = wp::sub(var_1, var_24);
        // dim = flex_dim[f]                                                                      <L 604>
        var_25 = wp::address(var_flex_dim, var_21);
        var_27 = wp::load(var_25);
        var_26 = wp::copy(var_27);
        // nvert = dim + 1                                                                        <L 605>
        var_29 = wp::add(var_26, var_28);
        // nedge = nvert * (nvert - 1) / 2                                                        <L 606>
        var_31 = wp::sub(var_29, var_30);
        var_32 = wp::mul(var_29, var_31);
        var_34 = wp::div(var_32, var_33);
        // edges = wp.where(                                                                      <L 607>
        // dim == 1,                                                                              <L 608>
        var_36 = (var_26 == var_35);
        // wp.matrix(0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, shape=(6, 2), dtype=int),                <L 609>
        var_51 = wp::tuple(var_49, var_50);
        var_52 = wp::mat_t<6, 2, wp::int32>({var_37, var_38, var_39, var_40, var_41, var_42, var_43, var_44, var_45, var_46, var_47, var_48});
        // wp.where(                                                                              <L 610>
        // dim == 3,                                                                              <L 611>
        var_54 = (var_26 == var_53);
        // wp.matrix(0, 1, 1, 2, 2, 0, 2, 3, 0, 3, 1, 3, shape=(6, 2), dtype=int),                <L 612>
        var_69 = wp::tuple(var_67, var_68);
        var_70 = wp::mat_t<6, 2, wp::int32>({var_55, var_56, var_57, var_58, var_59, var_60, var_61, var_62, var_63, var_64, var_65, var_66});
        // wp.matrix(1, 2, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, shape=(6, 2), dtype=int),                <L 613>
        var_85 = wp::tuple(var_83, var_84);
        var_86 = wp::mat_t<6, 2, wp::int32>({var_71, var_72, var_73, var_74, var_75, var_76, var_77, var_78, var_79, var_80, var_81, var_82});
        var_87 = wp::where(var_54, var_70, var_86);
        var_88 = wp::where(var_36, var_52, var_87);
        // if timestep > 0.0 and not dsbl_damper:                                                 <L 616>
        var_90 = (var_8 > var_89);
        var_91 = wp::unot(var_dsbl_damper);
        var_92 = var_90 && var_91;
        if (var_92) {
            // kD = flex_damping[f] / timestep                                                    <L 617>
            var_93 = wp::address(var_flex_damping, var_21);
            var_95 = wp::load(var_93);
            var_94 = wp::div(var_95, var_8);
        }
        if (!var_92) {
            // kD = 0.0                                                                           <L 619>
        }
        var_97 = wp::where(var_92, var_94, var_96);
        // elem_data_adr = flex_elemdataadr[f] + local_elemid * (dim + 1)                         <L 621>
        var_98 = wp::address(var_flex_elemdataadr, var_21);
        var_100 = wp::add(var_26, var_99);
        var_101 = wp::mul(var_23, var_100);
        var_103 = wp::load(var_98);
        var_102 = wp::add(var_103, var_101);
        // vbase = flex_vertadr[f]                                                                <L 622>
        var_104 = wp::address(var_flex_vertadr, var_21);
        var_106 = wp::load(var_104);
        var_105 = wp::copy(var_106);
        // gradient = wp.matrix(0.0, shape=(6, 6))                                                <L 623>
        var_110 = wp::tuple(var_108, var_109);
        var_111 = wp::mat_t<6, 6, wp::float32>(var_107);
        // for e in range(nedge):                                                                 <L 624>
        var_112 = wp::range(var_34);
        start_for_2:;
            if (iter_cmp(var_112) == 0) goto end_for_2;
            var_113 = wp::iter_next(var_112);
            // vert0 = flex_elem[elem_data_adr + edges[e, 0]]                                     <L 625>
            var_115 = wp::extract(var_88, var_113, var_114);
            var_116 = wp::add(var_102, var_115);
            var_117 = wp::address(var_flex_elem, var_116);
            var_119 = wp::load(var_117);
            var_118 = wp::copy(var_119);
            // vert1 = flex_elem[elem_data_adr + edges[e, 1]]                                     <L 626>
            var_121 = wp::extract(var_88, var_113, var_120);
            var_122 = wp::add(var_102, var_121);
            var_123 = wp::address(var_flex_elem, var_122);
            var_125 = wp::load(var_123);
            var_124 = wp::copy(var_125);
            // xpos0 = flexvert_xpos_in[worldid, vbase + vert0]                                   <L 627>
            var_126 = wp::add(var_105, var_118);
            var_127 = wp::address(var_flexvert_xpos_in, var_0, var_126);
            var_129 = wp::load(var_127);
            var_128 = wp::copy(var_129);
            // xpos1 = flexvert_xpos_in[worldid, vbase + vert1]                                   <L 628>
            var_130 = wp::add(var_105, var_124);
            var_131 = wp::address(var_flexvert_xpos_in, var_0, var_130);
            var_133 = wp::load(var_131);
            var_132 = wp::copy(var_133);
            // for i in range(3):                                                                 <L 629>
            // gradient[e, 0 + i] = xpos0[i] - xpos1[i]                                           <L 630>
            var_135 = wp::extract(var_128, var_134);
            var_136 = wp::extract(var_132, var_134);
            var_137 = wp::sub(var_135, var_136);
            var_139 = wp::add(var_138, var_134);
            wp::assign_inplace(var_111, var_113, var_139, var_137);
            // gradient[e, 3 + i] = xpos1[i] - xpos0[i]                                           <L 631>
            var_140 = wp::extract(var_132, var_134);
            var_141 = wp::extract(var_128, var_134);
            var_142 = wp::sub(var_140, var_141);
            var_144 = wp::add(var_143, var_134);
            wp::assign_inplace(var_111, var_113, var_144, var_142);
            // gradient[e, 0 + i] = xpos0[i] - xpos1[i]                                           <L 630>
            var_146 = wp::extract(var_128, var_145);
            var_147 = wp::extract(var_132, var_145);
            var_148 = wp::sub(var_146, var_147);
            var_150 = wp::add(var_149, var_145);
            wp::assign_inplace(var_111, var_113, var_150, var_148);
            // gradient[e, 3 + i] = xpos1[i] - xpos0[i]                                           <L 631>
            var_151 = wp::extract(var_132, var_145);
            var_152 = wp::extract(var_128, var_145);
            var_153 = wp::sub(var_151, var_152);
            var_155 = wp::add(var_154, var_145);
            wp::assign_inplace(var_111, var_113, var_155, var_153);
            // gradient[e, 0 + i] = xpos0[i] - xpos1[i]                                           <L 630>
            var_157 = wp::extract(var_128, var_156);
            var_158 = wp::extract(var_132, var_156);
            var_159 = wp::sub(var_157, var_158);
            var_161 = wp::add(var_160, var_156);
            wp::assign_inplace(var_111, var_113, var_161, var_159);
            // gradient[e, 3 + i] = xpos1[i] - xpos0[i]                                           <L 631>
            var_162 = wp::extract(var_132, var_156);
            var_163 = wp::extract(var_128, var_156);
            var_164 = wp::sub(var_162, var_163);
            var_166 = wp::add(var_165, var_156);
            wp::assign_inplace(var_111, var_113, var_166, var_164);
            goto start_for_2;
        end_for_2:;
        // elongation = wp.spatial_vectorf(0.0)                                                   <L 633>
        var_168 = wp::vec_t<6, wp::float32>(var_167);
        // for e in range(nedge):                                                                 <L 634>
        var_169 = wp::range(var_34);
        start_for_4:;
            if (iter_cmp(var_169) == 0) goto end_for_4;
            var_170 = wp::iter_next(var_169);
            // idx = flex_elemedge[flex_elemedgeadr[f] + local_elemid * nedge + e]                <L 635>
            var_171 = wp::address(var_flex_elemedgeadr, var_21);
            var_172 = wp::mul(var_23, var_34);
            var_174 = wp::load(var_171);
            var_173 = wp::add(var_174, var_172);
            var_175 = wp::add(var_173, var_170);
            var_176 = wp::address(var_flex_elemedge, var_175);
            var_178 = wp::load(var_176);
            var_177 = wp::copy(var_178);
            // vel = flexedge_velocity_in[worldid, flex_edgeadr[f] + idx]                         <L 636>
            var_179 = wp::address(var_flex_edgeadr, var_21);
            var_181 = wp::load(var_179);
            var_180 = wp::add(var_181, var_177);
            var_182 = wp::address(var_flexedge_velocity_in, var_0, var_180);
            var_184 = wp::load(var_182);
            var_183 = wp::copy(var_184);
            // deformed = flexedge_length_in[worldid, flex_edgeadr[f] + idx]                      <L 637>
            var_185 = wp::address(var_flex_edgeadr, var_21);
            var_187 = wp::load(var_185);
            var_186 = wp::add(var_187, var_177);
            var_188 = wp::address(var_flexedge_length_in, var_0, var_186);
            var_190 = wp::load(var_188);
            var_189 = wp::copy(var_190);
            // reference = flexedge_length0[flex_edgeadr[f] + idx]                                <L 638>
            var_191 = wp::address(var_flex_edgeadr, var_21);
            var_193 = wp::load(var_191);
            var_192 = wp::add(var_193, var_177);
            var_194 = wp::address(var_flexedge_length0, var_192);
            var_196 = wp::load(var_194);
            var_195 = wp::copy(var_196);
            // previous = deformed - vel * timestep                                               <L 639>
            var_197 = wp::mul(var_183, var_8);
            var_198 = wp::sub(var_189, var_197);
            // elongation[e] = deformed * deformed - reference * reference + (deformed * deformed - previous * previous) * kD       <L 640>
            var_199 = wp::mul(var_189, var_189);
            var_200 = wp::mul(var_195, var_195);
            var_201 = wp::sub(var_199, var_200);
            var_202 = wp::mul(var_189, var_189);
            var_203 = wp::mul(var_198, var_198);
            var_204 = wp::sub(var_202, var_203);
            var_205 = wp::mul(var_204, var_97);
            var_206 = wp::add(var_201, var_205);
            wp::assign_inplace(var_168, var_170, var_206);
            goto start_for_4;
        end_for_4:;
        // metric = wp.matrix(0.0, shape=(6, 6))                                                  <L 642>
        var_210 = wp::tuple(var_208, var_209);
        var_211 = wp::mat_t<6, 6, wp::float32>(var_207);
        // id = int(0)                                                                            <L 643>
        var_213 = wp::int(var_212);
        // for ed1 in range(nedge):                                                               <L 644>
        var_214 = wp::range(var_34);
        start_for_6:;
            if (iter_cmp(var_214) == 0) goto end_for_6;
            var_215 = wp::iter_next(var_214);
            // for ed2 in range(ed1, nedge):                                                      <L 645>
            var_216 = wp::range(var_215, var_34);
            start_for_8:;
                if (iter_cmp(var_216) == 0) goto end_for_8;
                var_217 = wp::iter_next(var_216);
                // metric[ed1, ed2] = flex_stiffness[elemid, id]                                  <L 646>
                var_218 = wp::address(var_flex_stiffness, var_1, var_213);
                var_219 = wp::load(var_218);
                wp::assign_inplace(var_211, var_215, var_217, var_219);
                // metric[ed2, ed1] = flex_stiffness[elemid, id]                                  <L 647>
                var_220 = wp::address(var_flex_stiffness, var_1, var_213);
                var_221 = wp::load(var_220);
                wp::assign_inplace(var_211, var_217, var_215, var_221);
                // id += 1                                                                        <L 648>
                var_223 = wp::add(var_213, var_222);
                wp::assign(var_213, var_223);
                goto start_for_8;
            end_for_8:;
            goto start_for_6;
        end_for_6:;
        // force = wp.matrix(0.0, shape=(6, 3))                                                   <L 650>
        var_227 = wp::tuple(var_225, var_226);
        var_228 = wp::mat_t<6, 3, wp::float32>(var_224);
        // for ed1 in range(nedge):                                                               <L 651>
        var_229 = wp::range(var_34);
        start_for_10:;
            if (iter_cmp(var_229) == 0) goto end_for_10;
            var_230 = wp::iter_next(var_229);
            // for ed2 in range(nedge):                                                           <L 652>
            var_231 = wp::range(var_34);
            start_for_12:;
                if (iter_cmp(var_231) == 0) goto end_for_12;
                var_232 = wp::iter_next(var_231);
                // for i in range(2):                                                             <L 653>
                // for x in range(3):                                                             <L 654>
                // force[edges[ed2, i], x] -= elongation[ed1] * gradient[ed2, 3 * i + x] * metric[ed1, ed2]       <L 655>
                var_235 = wp::extract(var_168, var_230);
                var_237 = wp::mul(var_236, var_233);
                var_238 = wp::add(var_237, var_234);
                var_239 = wp::extract(var_111, var_232, var_238);
                var_240 = wp::mul(var_235, var_239);
                var_241 = wp::extract(var_211, var_230, var_232);
                var_242 = wp::mul(var_240, var_241);
                var_243 = wp::extract(var_88, var_232, var_233);
                wp::sub_inplace(var_228, var_243, var_234, var_242);
                var_245 = wp::extract(var_168, var_230);
                var_247 = wp::mul(var_246, var_233);
                var_248 = wp::add(var_247, var_244);
                var_249 = wp::extract(var_111, var_232, var_248);
                var_250 = wp::mul(var_245, var_249);
                var_251 = wp::extract(var_211, var_230, var_232);
                var_252 = wp::mul(var_250, var_251);
                var_253 = wp::extract(var_88, var_232, var_233);
                wp::sub_inplace(var_228, var_253, var_244, var_252);
                var_255 = wp::extract(var_168, var_230);
                var_257 = wp::mul(var_256, var_233);
                var_258 = wp::add(var_257, var_254);
                var_259 = wp::extract(var_111, var_232, var_258);
                var_260 = wp::mul(var_255, var_259);
                var_261 = wp::extract(var_211, var_230, var_232);
                var_262 = wp::mul(var_260, var_261);
                var_263 = wp::extract(var_88, var_232, var_233);
                wp::sub_inplace(var_228, var_263, var_254, var_262);
                // for x in range(3):                                                             <L 654>
                // force[edges[ed2, i], x] -= elongation[ed1] * gradient[ed2, 3 * i + x] * metric[ed1, ed2]       <L 655>
                var_266 = wp::extract(var_168, var_230);
                var_268 = wp::mul(var_267, var_264);
                var_269 = wp::add(var_268, var_265);
                var_270 = wp::extract(var_111, var_232, var_269);
                var_271 = wp::mul(var_266, var_270);
                var_272 = wp::extract(var_211, var_230, var_232);
                var_273 = wp::mul(var_271, var_272);
                var_274 = wp::extract(var_88, var_232, var_264);
                wp::sub_inplace(var_228, var_274, var_265, var_273);
                var_276 = wp::extract(var_168, var_230);
                var_278 = wp::mul(var_277, var_264);
                var_279 = wp::add(var_278, var_275);
                var_280 = wp::extract(var_111, var_232, var_279);
                var_281 = wp::mul(var_276, var_280);
                var_282 = wp::extract(var_211, var_230, var_232);
                var_283 = wp::mul(var_281, var_282);
                var_284 = wp::extract(var_88, var_232, var_264);
                wp::sub_inplace(var_228, var_284, var_275, var_283);
                var_286 = wp::extract(var_168, var_230);
                var_288 = wp::mul(var_287, var_264);
                var_289 = wp::add(var_288, var_285);
                var_290 = wp::extract(var_111, var_232, var_289);
                var_291 = wp::mul(var_286, var_290);
                var_292 = wp::extract(var_211, var_230, var_232);
                var_293 = wp::mul(var_291, var_292);
                var_294 = wp::extract(var_88, var_232, var_264);
                wp::sub_inplace(var_228, var_294, var_285, var_293);
                goto start_for_12;
            end_for_12:;
            wp::assign(var_217, var_232);
            goto start_for_10;
        end_for_10:;
        // for v in range(nvert):                                                                 <L 657>
        var_295 = wp::range(var_29);
        start_for_14:;
            if (iter_cmp(var_295) == 0) goto end_for_14;
            var_296 = wp::iter_next(var_295);
            // vert = flex_elem[elem_data_adr + v]                                                <L 658>
            var_297 = wp::add(var_102, var_296);
            var_298 = wp::address(var_flex_elem, var_297);
            var_300 = wp::load(var_298);
            var_299 = wp::copy(var_300);
            // bodyid = flex_vertbodyid[flex_vertadr[f] + vert]                                   <L 659>
            var_301 = wp::address(var_flex_vertadr, var_21);
            var_303 = wp::load(var_301);
            var_302 = wp::add(var_303, var_299);
            var_304 = wp::address(var_flex_vertbodyid, var_302);
            var_306 = wp::load(var_304);
            var_305 = wp::copy(var_306);
            // for x in range(3):                                                                 <L 660>
            // wp.atomic_add(qfrc_spring_out, worldid, body_dofadr[bodyid] + x, force[v, x])       <L 661>
            var_308 = wp::address(var_body_dofadr, var_305);
            var_310 = wp::load(var_308);
            var_309 = wp::add(var_310, var_307);
            var_311 = wp::extract(var_228, var_296, var_307);
            var_312 = wp::atomic_add(var_qfrc_spring_out, var_0, var_309, var_311);
            var_314 = wp::address(var_body_dofadr, var_305);
            var_316 = wp::load(var_314);
            var_315 = wp::add(var_316, var_313);
            var_317 = wp::extract(var_228, var_296, var_313);
            var_318 = wp::atomic_add(var_qfrc_spring_out, var_0, var_315, var_317);
            var_320 = wp::address(var_body_dofadr, var_305);
            var_322 = wp::load(var_320);
            var_321 = wp::add(var_322, var_319);
            var_323 = wp::extract(var_228, var_296, var_319);
            var_324 = wp::atomic_add(var_qfrc_spring_out, var_0, var_321, var_323);
            goto start_for_14;
        end_for_14:;
    }
}



extern "C" __global__ void _spring_damper_dof_passive_fd73f275_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_opt_disableflags,
    wp::array_t<wp::float32> var_qpos_spring,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_jnt_qposadr,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::float32> var_jnt_stiffness,
    wp::array_t<wp::float32> var_dof_damping,
    wp::array_t<wp::float32> var_qpos_in,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::float32> var_qfrc_spring_out,
    wp::array_t<wp::float32> var_qfrc_damper_out)
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
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        wp::shape_t* var_13;
        const wp::int32 var_14 = 0;
        wp::int32 var_15;
        wp::shape_t var_16;
        wp::int32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        const wp::float32 var_21 = 0.0;
        bool var_22;
        const wp::int32 var_23 = 32;
        wp::int32 var_24;
        bool var_25;
        bool var_26;
        const wp::float32 var_27 = 0.0;
        bool var_28;
        const wp::int32 var_29 = 64;
        wp::int32 var_30;
        bool var_31;
        bool var_32;
        bool var_33;
        const wp::float32 var_34 = 0.0;
        bool var_35;
        const wp::float32 var_36 = 0.0;
        bool var_37;
        bool var_38;
        wp::int32* var_39;
        wp::int32 var_40;
        wp::int32 var_41;
        wp::int32* var_42;
        wp::int32 var_43;
        wp::int32 var_44;
        wp::shape_t* var_45;
        const wp::int32 var_46 = 0;
        wp::int32 var_47;
        wp::shape_t var_48;
        wp::int32 var_49;
        const wp::int32 var_50 = 0;
        bool var_51;
        const wp::int32 var_52 = 0;
        wp::int32 var_53;
        wp::float32* var_54;
        const wp::int32 var_55 = 0;
        wp::int32 var_56;
        wp::float32* var_57;
        wp::float32 var_58;
        wp::float32 var_59;
        wp::float32 var_60;
        const wp::int32 var_61 = 1;
        wp::int32 var_62;
        wp::float32* var_63;
        const wp::int32 var_64 = 1;
        wp::int32 var_65;
        wp::float32* var_66;
        wp::float32 var_67;
        wp::float32 var_68;
        wp::float32 var_69;
        const wp::int32 var_70 = 2;
        wp::int32 var_71;
        wp::float32* var_72;
        const wp::int32 var_73 = 2;
        wp::int32 var_74;
        wp::float32* var_75;
        wp::float32 var_76;
        wp::float32 var_77;
        wp::float32 var_78;
        wp::vec_t<3, wp::float32> var_79;
        wp::float32 var_80;
        const wp::int32 var_81 = 0;
        wp::float32 var_82;
        wp::float32 var_83;
        const wp::int32 var_84 = 0;
        wp::int32 var_85;
        wp::float32 var_86;
        const wp::int32 var_87 = 1;
        wp::float32 var_88;
        wp::float32 var_89;
        const wp::int32 var_90 = 1;
        wp::int32 var_91;
        wp::float32 var_92;
        const wp::int32 var_93 = 2;
        wp::float32 var_94;
        wp::float32 var_95;
        const wp::int32 var_96 = 2;
        wp::int32 var_97;
        const wp::int32 var_98 = 3;
        wp::int32 var_99;
        wp::float32* var_100;
        const wp::int32 var_101 = 4;
        wp::int32 var_102;
        wp::float32* var_103;
        const wp::int32 var_104 = 5;
        wp::int32 var_105;
        wp::float32* var_106;
        const wp::int32 var_107 = 6;
        wp::int32 var_108;
        wp::float32* var_109;
        wp::quat_t<wp::float32> var_110;
        wp::float32 var_111;
        wp::float32 var_112;
        wp::float32 var_113;
        wp::float32 var_114;
        wp::quat_t<wp::float32> var_115;
        const wp::int32 var_116 = 3;
        wp::int32 var_117;
        wp::float32* var_118;
        const wp::int32 var_119 = 4;
        wp::int32 var_120;
        wp::float32* var_121;
        const wp::int32 var_122 = 5;
        wp::int32 var_123;
        wp::float32* var_124;
        const wp::int32 var_125 = 6;
        wp::int32 var_126;
        wp::float32* var_127;
        wp::quat_t<wp::float32> var_128;
        wp::float32 var_129;
        wp::float32 var_130;
        wp::float32 var_131;
        wp::float32 var_132;
        wp::vec_t<3, wp::float32> var_133;
        wp::float32 var_134;
        const wp::int32 var_135 = 0;
        wp::float32 var_136;
        wp::float32 var_137;
        const wp::int32 var_138 = 3;
        wp::int32 var_139;
        wp::float32 var_140;
        const wp::int32 var_141 = 1;
        wp::float32 var_142;
        wp::float32 var_143;
        const wp::int32 var_144 = 4;
        wp::int32 var_145;
        wp::float32 var_146;
        const wp::int32 var_147 = 2;
        wp::float32 var_148;
        wp::float32 var_149;
        const wp::int32 var_150 = 5;
        wp::int32 var_151;
        wp::float32 var_152;
        const wp::int32 var_153 = 0;
        wp::int32 var_154;
        wp::float32* var_155;
        wp::float32 var_156;
        wp::float32 var_157;
        const wp::int32 var_158 = 0;
        wp::int32 var_159;
        wp::float32 var_160;
        const wp::int32 var_161 = 1;
        wp::int32 var_162;
        wp::float32* var_163;
        wp::float32 var_164;
        wp::float32 var_165;
        const wp::int32 var_166 = 1;
        wp::int32 var_167;
        wp::float32 var_168;
        const wp::int32 var_169 = 2;
        wp::int32 var_170;
        wp::float32* var_171;
        wp::float32 var_172;
        wp::float32 var_173;
        const wp::int32 var_174 = 2;
        wp::int32 var_175;
        wp::float32 var_176;
        const wp::int32 var_177 = 3;
        wp::int32 var_178;
        wp::float32* var_179;
        wp::float32 var_180;
        wp::float32 var_181;
        const wp::int32 var_182 = 3;
        wp::int32 var_183;
        wp::float32 var_184;
        const wp::int32 var_185 = 4;
        wp::int32 var_186;
        wp::float32* var_187;
        wp::float32 var_188;
        wp::float32 var_189;
        const wp::int32 var_190 = 4;
        wp::int32 var_191;
        wp::float32 var_192;
        const wp::int32 var_193 = 5;
        wp::int32 var_194;
        wp::float32* var_195;
        wp::float32 var_196;
        wp::float32 var_197;
        const wp::int32 var_198 = 5;
        wp::int32 var_199;
        const wp::int32 var_200 = 1;
        bool var_201;
        const wp::int32 var_202 = 0;
        wp::int32 var_203;
        wp::float32* var_204;
        const wp::int32 var_205 = 1;
        wp::int32 var_206;
        wp::float32* var_207;
        const wp::int32 var_208 = 2;
        wp::int32 var_209;
        wp::float32* var_210;
        const wp::int32 var_211 = 3;
        wp::int32 var_212;
        wp::float32* var_213;
        wp::quat_t<wp::float32> var_214;
        wp::float32 var_215;
        wp::float32 var_216;
        wp::float32 var_217;
        wp::float32 var_218;
        wp::quat_t<wp::float32> var_219;
        const wp::int32 var_220 = 0;
        wp::int32 var_221;
        wp::float32* var_222;
        const wp::int32 var_223 = 1;
        wp::int32 var_224;
        wp::float32* var_225;
        const wp::int32 var_226 = 2;
        wp::int32 var_227;
        wp::float32* var_228;
        const wp::int32 var_229 = 3;
        wp::int32 var_230;
        wp::float32* var_231;
        wp::quat_t<wp::float32> var_232;
        wp::float32 var_233;
        wp::float32 var_234;
        wp::float32 var_235;
        wp::float32 var_236;
        wp::vec_t<3, wp::float32> var_237;
        wp::float32 var_238;
        const wp::int32 var_239 = 0;
        wp::float32 var_240;
        wp::float32 var_241;
        const wp::int32 var_242 = 0;
        wp::int32 var_243;
        wp::float32 var_244;
        const wp::int32 var_245 = 1;
        wp::float32 var_246;
        wp::float32 var_247;
        const wp::int32 var_248 = 1;
        wp::int32 var_249;
        wp::float32 var_250;
        const wp::int32 var_251 = 2;
        wp::float32 var_252;
        wp::float32 var_253;
        const wp::int32 var_254 = 2;
        wp::int32 var_255;
        wp::vec_t<3, wp::float32> var_256;
        wp::quat_t<wp::float32> var_257;
        wp::quat_t<wp::float32> var_258;
        wp::float32 var_259;
        const wp::int32 var_260 = 0;
        wp::int32 var_261;
        wp::float32* var_262;
        wp::float32 var_263;
        wp::float32 var_264;
        const wp::int32 var_265 = 0;
        wp::int32 var_266;
        wp::float32 var_267;
        const wp::int32 var_268 = 1;
        wp::int32 var_269;
        wp::float32* var_270;
        wp::float32 var_271;
        wp::float32 var_272;
        const wp::int32 var_273 = 1;
        wp::int32 var_274;
        wp::float32 var_275;
        const wp::int32 var_276 = 2;
        wp::int32 var_277;
        wp::float32* var_278;
        wp::float32 var_279;
        wp::float32 var_280;
        const wp::int32 var_281 = 2;
        wp::int32 var_282;
        wp::vec_t<3, wp::float32> var_283;
        wp::quat_t<wp::float32> var_284;
        wp::quat_t<wp::float32> var_285;
        wp::float32* var_286;
        wp::float32* var_287;
        wp::float32 var_288;
        wp::float32 var_289;
        wp::float32 var_290;
        wp::float32 var_291;
        wp::float32 var_292;
        wp::float32 var_293;
        wp::float32* var_294;
        wp::float32 var_295;
        wp::float32 var_296;
        wp::vec_t<3, wp::float32> var_297;
        wp::quat_t<wp::float32> var_298;
        wp::quat_t<wp::float32> var_299;
        //---------
        // forward
        // def _spring_damper_dof_passive(                                                        <L 71>
        // worldid, jntid = wp.tid()                                                              <L 87>
        builtin_tid2d(var_0, var_1);
        // dofid = jnt_dofadr[jntid]                                                              <L 88>
        var_2 = wp::address(var_jnt_dofadr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // stiffness = jnt_stiffness[worldid % jnt_stiffness.shape[0], jntid]                     <L 89>
        var_5 = &(var_jnt_stiffness.shape);
        var_8 = wp::load(var_5);
        var_7 = wp::extract(var_8, var_6);
        var_9 = wp::mod(var_0, var_7);
        var_10 = wp::address(var_jnt_stiffness, var_9, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // damping = dof_damping[worldid % dof_damping.shape[0], dofid]                           <L 90>
        var_13 = &(var_dof_damping.shape);
        var_16 = wp::load(var_13);
        var_15 = wp::extract(var_16, var_14);
        var_17 = wp::mod(var_0, var_15);
        var_18 = wp::address(var_dof_damping, var_17, var_3);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // has_stiffness = stiffness != 0.0 and not (opt_disableflags & DisableBit.SPRING)        <L 92>
        var_22 = (var_11 != var_21);
        var_24 = wp::bit_and(var_opt_disableflags, var_23);
        var_25 = wp::unot(var_24);
        var_26 = var_22 && var_25;
        // has_damping = damping != 0.0 and not (opt_disableflags & DisableBit.DAMPER)            <L 93>
        var_28 = (var_19 != var_27);
        var_30 = wp::bit_and(var_opt_disableflags, var_29);
        var_31 = wp::unot(var_30);
        var_32 = var_28 && var_31;
        // if not has_stiffness:                                                                  <L 95>
        var_33 = wp::unot(var_26);
        if (var_33) {
            // qfrc_spring_out[worldid, dofid] = 0.0                                              <L 96>
            wp::array_store(var_qfrc_spring_out, var_0, var_3, var_34);
        }
        // if not has_damping:                                                                    <L 98>
        var_35 = wp::unot(var_32);
        if (var_35) {
            // qfrc_damper_out[worldid, dofid] = 0.0                                              <L 99>
            wp::array_store(var_qfrc_damper_out, var_0, var_3, var_36);
        }
        // if not (has_stiffness or has_damping):                                                 <L 101>
        var_37 = var_26 || var_32;
        var_38 = wp::unot(var_37);
        if (var_38) {
            // return                                                                             <L 102>
            continue;
        }
        // jnttype = jnt_type[jntid]                                                              <L 104>
        var_39 = wp::address(var_jnt_type, var_1);
        var_41 = wp::load(var_39);
        var_40 = wp::copy(var_41);
        // qposid = jnt_qposadr[jntid]                                                            <L 105>
        var_42 = wp::address(var_jnt_qposadr, var_1);
        var_44 = wp::load(var_42);
        var_43 = wp::copy(var_44);
        // qpos_spring_id = worldid % qpos_spring.shape[0]                                        <L 106>
        var_45 = &(var_qpos_spring.shape);
        var_48 = wp::load(var_45);
        var_47 = wp::extract(var_48, var_46);
        var_49 = wp::mod(var_0, var_47);
        // if jnttype == JointType.FREE:                                                          <L 108>
        var_51 = (var_40 == var_50);
        if (var_51) {
            // if has_stiffness:                                                                  <L 110>
            if (var_26) {
                // dif = wp.vec3(                                                                 <L 111>
                // qpos_in[worldid, qposid + 0] - qpos_spring[qpos_spring_id, qposid + 0],        <L 112>
                var_53 = wp::add(var_43, var_52);
                var_54 = wp::address(var_qpos_in, var_0, var_53);
                var_56 = wp::add(var_43, var_55);
                var_57 = wp::address(var_qpos_spring, var_49, var_56);
                var_59 = wp::load(var_54);
                var_60 = wp::load(var_57);
                var_58 = wp::sub(var_59, var_60);
                // qpos_in[worldid, qposid + 1] - qpos_spring[qpos_spring_id, qposid + 1],        <L 113>
                var_62 = wp::add(var_43, var_61);
                var_63 = wp::address(var_qpos_in, var_0, var_62);
                var_65 = wp::add(var_43, var_64);
                var_66 = wp::address(var_qpos_spring, var_49, var_65);
                var_68 = wp::load(var_63);
                var_69 = wp::load(var_66);
                var_67 = wp::sub(var_68, var_69);
                // qpos_in[worldid, qposid + 2] - qpos_spring[qpos_spring_id, qposid + 2],        <L 114>
                var_71 = wp::add(var_43, var_70);
                var_72 = wp::address(var_qpos_in, var_0, var_71);
                var_74 = wp::add(var_43, var_73);
                var_75 = wp::address(var_qpos_spring, var_49, var_74);
                var_77 = wp::load(var_72);
                var_78 = wp::load(var_75);
                var_76 = wp::sub(var_77, var_78);
                var_79 = wp::vec_t<3, wp::float32>(var_58, var_67, var_76);
                // qfrc_spring_out[worldid, dofid + 0] = -stiffness * dif[0]                      <L 116>
                var_80 = wp::neg(var_11);
                var_82 = wp::extract(var_79, var_81);
                var_83 = wp::mul(var_80, var_82);
                var_85 = wp::add(var_3, var_84);
                wp::array_store(var_qfrc_spring_out, var_0, var_85, var_83);
                // qfrc_spring_out[worldid, dofid + 1] = -stiffness * dif[1]                      <L 117>
                var_86 = wp::neg(var_11);
                var_88 = wp::extract(var_79, var_87);
                var_89 = wp::mul(var_86, var_88);
                var_91 = wp::add(var_3, var_90);
                wp::array_store(var_qfrc_spring_out, var_0, var_91, var_89);
                // qfrc_spring_out[worldid, dofid + 2] = -stiffness * dif[2]                      <L 118>
                var_92 = wp::neg(var_11);
                var_94 = wp::extract(var_79, var_93);
                var_95 = wp::mul(var_92, var_94);
                var_97 = wp::add(var_3, var_96);
                wp::array_store(var_qfrc_spring_out, var_0, var_97, var_95);
                // rot = wp.quat(                                                                 <L 119>
                // qpos_in[worldid, qposid + 3],                                                  <L 120>
                var_99 = wp::add(var_43, var_98);
                var_100 = wp::address(var_qpos_in, var_0, var_99);
                // qpos_in[worldid, qposid + 4],                                                  <L 121>
                var_102 = wp::add(var_43, var_101);
                var_103 = wp::address(var_qpos_in, var_0, var_102);
                // qpos_in[worldid, qposid + 5],                                                  <L 122>
                var_105 = wp::add(var_43, var_104);
                var_106 = wp::address(var_qpos_in, var_0, var_105);
                // qpos_in[worldid, qposid + 6],                                                  <L 123>
                var_108 = wp::add(var_43, var_107);
                var_109 = wp::address(var_qpos_in, var_0, var_108);
                var_111 = wp::load(var_100);
                var_112 = wp::load(var_103);
                var_113 = wp::load(var_106);
                var_114 = wp::load(var_109);
                var_110 = wp::quat_t<wp::float32>(var_111, var_112, var_113, var_114);
                // rot = wp.normalize(rot)                                                        <L 125>
                var_115 = wp::normalize(var_110);
                // ref = wp.quat(                                                                 <L 126>
                // qpos_spring[qpos_spring_id, qposid + 3],                                       <L 127>
                var_117 = wp::add(var_43, var_116);
                var_118 = wp::address(var_qpos_spring, var_49, var_117);
                // qpos_spring[qpos_spring_id, qposid + 4],                                       <L 128>
                var_120 = wp::add(var_43, var_119);
                var_121 = wp::address(var_qpos_spring, var_49, var_120);
                // qpos_spring[qpos_spring_id, qposid + 5],                                       <L 129>
                var_123 = wp::add(var_43, var_122);
                var_124 = wp::address(var_qpos_spring, var_49, var_123);
                // qpos_spring[qpos_spring_id, qposid + 6],                                       <L 130>
                var_126 = wp::add(var_43, var_125);
                var_127 = wp::address(var_qpos_spring, var_49, var_126);
                var_129 = wp::load(var_118);
                var_130 = wp::load(var_121);
                var_131 = wp::load(var_124);
                var_132 = wp::load(var_127);
                var_128 = wp::quat_t<wp::float32>(var_129, var_130, var_131, var_132);
                // dif = math.quat_sub(rot, ref)                                                  <L 132>
                var_133 = quat_sub_0(var_115, var_128);
                // qfrc_spring_out[worldid, dofid + 3] = -stiffness * dif[0]                      <L 133>
                var_134 = wp::neg(var_11);
                var_136 = wp::extract(var_133, var_135);
                var_137 = wp::mul(var_134, var_136);
                var_139 = wp::add(var_3, var_138);
                wp::array_store(var_qfrc_spring_out, var_0, var_139, var_137);
                // qfrc_spring_out[worldid, dofid + 4] = -stiffness * dif[1]                      <L 134>
                var_140 = wp::neg(var_11);
                var_142 = wp::extract(var_133, var_141);
                var_143 = wp::mul(var_140, var_142);
                var_145 = wp::add(var_3, var_144);
                wp::array_store(var_qfrc_spring_out, var_0, var_145, var_143);
                // qfrc_spring_out[worldid, dofid + 5] = -stiffness * dif[2]                      <L 135>
                var_146 = wp::neg(var_11);
                var_148 = wp::extract(var_133, var_147);
                var_149 = wp::mul(var_146, var_148);
                var_151 = wp::add(var_3, var_150);
                wp::array_store(var_qfrc_spring_out, var_0, var_151, var_149);
            }
            // if has_damping:                                                                    <L 138>
            if (var_32) {
                // qfrc_damper_out[worldid, dofid + 0] = -damping * qvel_in[worldid, dofid + 0]       <L 139>
                var_152 = wp::neg(var_19);
                var_154 = wp::add(var_3, var_153);
                var_155 = wp::address(var_qvel_in, var_0, var_154);
                var_157 = wp::load(var_155);
                var_156 = wp::mul(var_152, var_157);
                var_159 = wp::add(var_3, var_158);
                wp::array_store(var_qfrc_damper_out, var_0, var_159, var_156);
                // qfrc_damper_out[worldid, dofid + 1] = -damping * qvel_in[worldid, dofid + 1]       <L 140>
                var_160 = wp::neg(var_19);
                var_162 = wp::add(var_3, var_161);
                var_163 = wp::address(var_qvel_in, var_0, var_162);
                var_165 = wp::load(var_163);
                var_164 = wp::mul(var_160, var_165);
                var_167 = wp::add(var_3, var_166);
                wp::array_store(var_qfrc_damper_out, var_0, var_167, var_164);
                // qfrc_damper_out[worldid, dofid + 2] = -damping * qvel_in[worldid, dofid + 2]       <L 141>
                var_168 = wp::neg(var_19);
                var_170 = wp::add(var_3, var_169);
                var_171 = wp::address(var_qvel_in, var_0, var_170);
                var_173 = wp::load(var_171);
                var_172 = wp::mul(var_168, var_173);
                var_175 = wp::add(var_3, var_174);
                wp::array_store(var_qfrc_damper_out, var_0, var_175, var_172);
                // qfrc_damper_out[worldid, dofid + 3] = -damping * qvel_in[worldid, dofid + 3]       <L 142>
                var_176 = wp::neg(var_19);
                var_178 = wp::add(var_3, var_177);
                var_179 = wp::address(var_qvel_in, var_0, var_178);
                var_181 = wp::load(var_179);
                var_180 = wp::mul(var_176, var_181);
                var_183 = wp::add(var_3, var_182);
                wp::array_store(var_qfrc_damper_out, var_0, var_183, var_180);
                // qfrc_damper_out[worldid, dofid + 4] = -damping * qvel_in[worldid, dofid + 4]       <L 143>
                var_184 = wp::neg(var_19);
                var_186 = wp::add(var_3, var_185);
                var_187 = wp::address(var_qvel_in, var_0, var_186);
                var_189 = wp::load(var_187);
                var_188 = wp::mul(var_184, var_189);
                var_191 = wp::add(var_3, var_190);
                wp::array_store(var_qfrc_damper_out, var_0, var_191, var_188);
                // qfrc_damper_out[worldid, dofid + 5] = -damping * qvel_in[worldid, dofid + 5]       <L 144>
                var_192 = wp::neg(var_19);
                var_194 = wp::add(var_3, var_193);
                var_195 = wp::address(var_qvel_in, var_0, var_194);
                var_197 = wp::load(var_195);
                var_196 = wp::mul(var_192, var_197);
                var_199 = wp::add(var_3, var_198);
                wp::array_store(var_qfrc_damper_out, var_0, var_199, var_196);
            }
        }
        if (!var_51) {
            // elif jnttype == JointType.BALL:                                                    <L 145>
            var_201 = (var_40 == var_200);
            if (var_201) {
                // if has_stiffness:                                                              <L 147>
                if (var_26) {
                    // rot = wp.quat(                                                             <L 148>
                    // qpos_in[worldid, qposid + 0],                                              <L 149>
                    var_203 = wp::add(var_43, var_202);
                    var_204 = wp::address(var_qpos_in, var_0, var_203);
                    // qpos_in[worldid, qposid + 1],                                              <L 150>
                    var_206 = wp::add(var_43, var_205);
                    var_207 = wp::address(var_qpos_in, var_0, var_206);
                    // qpos_in[worldid, qposid + 2],                                              <L 151>
                    var_209 = wp::add(var_43, var_208);
                    var_210 = wp::address(var_qpos_in, var_0, var_209);
                    // qpos_in[worldid, qposid + 3],                                              <L 152>
                    var_212 = wp::add(var_43, var_211);
                    var_213 = wp::address(var_qpos_in, var_0, var_212);
                    var_215 = wp::load(var_204);
                    var_216 = wp::load(var_207);
                    var_217 = wp::load(var_210);
                    var_218 = wp::load(var_213);
                    var_214 = wp::quat_t<wp::float32>(var_215, var_216, var_217, var_218);
                    // rot = wp.normalize(rot)                                                    <L 154>
                    var_219 = wp::normalize(var_214);
                    // ref = wp.quat(                                                             <L 155>
                    // qpos_spring[qpos_spring_id, qposid + 0],                                   <L 156>
                    var_221 = wp::add(var_43, var_220);
                    var_222 = wp::address(var_qpos_spring, var_49, var_221);
                    // qpos_spring[qpos_spring_id, qposid + 1],                                   <L 157>
                    var_224 = wp::add(var_43, var_223);
                    var_225 = wp::address(var_qpos_spring, var_49, var_224);
                    // qpos_spring[qpos_spring_id, qposid + 2],                                   <L 158>
                    var_227 = wp::add(var_43, var_226);
                    var_228 = wp::address(var_qpos_spring, var_49, var_227);
                    // qpos_spring[qpos_spring_id, qposid + 3],                                   <L 159>
                    var_230 = wp::add(var_43, var_229);
                    var_231 = wp::address(var_qpos_spring, var_49, var_230);
                    var_233 = wp::load(var_222);
                    var_234 = wp::load(var_225);
                    var_235 = wp::load(var_228);
                    var_236 = wp::load(var_231);
                    var_232 = wp::quat_t<wp::float32>(var_233, var_234, var_235, var_236);
                    // dif = math.quat_sub(rot, ref)                                              <L 161>
                    var_237 = quat_sub_0(var_219, var_232);
                    // qfrc_spring_out[worldid, dofid + 0] = -stiffness * dif[0]                  <L 162>
                    var_238 = wp::neg(var_11);
                    var_240 = wp::extract(var_237, var_239);
                    var_241 = wp::mul(var_238, var_240);
                    var_243 = wp::add(var_3, var_242);
                    wp::array_store(var_qfrc_spring_out, var_0, var_243, var_241);
                    // qfrc_spring_out[worldid, dofid + 1] = -stiffness * dif[1]                  <L 163>
                    var_244 = wp::neg(var_11);
                    var_246 = wp::extract(var_237, var_245);
                    var_247 = wp::mul(var_244, var_246);
                    var_249 = wp::add(var_3, var_248);
                    wp::array_store(var_qfrc_spring_out, var_0, var_249, var_247);
                    // qfrc_spring_out[worldid, dofid + 2] = -stiffness * dif[2]                  <L 164>
                    var_250 = wp::neg(var_11);
                    var_252 = wp::extract(var_237, var_251);
                    var_253 = wp::mul(var_250, var_252);
                    var_255 = wp::add(var_3, var_254);
                    wp::array_store(var_qfrc_spring_out, var_0, var_255, var_253);
                }
                var_256 = wp::where(var_26, var_237, var_133);
                var_257 = wp::where(var_26, var_219, var_115);
                var_258 = wp::where(var_26, var_232, var_128);
                // if has_damping:                                                                <L 167>
                if (var_32) {
                    // qfrc_damper_out[worldid, dofid + 0] = -damping * qvel_in[worldid, dofid + 0]       <L 168>
                    var_259 = wp::neg(var_19);
                    var_261 = wp::add(var_3, var_260);
                    var_262 = wp::address(var_qvel_in, var_0, var_261);
                    var_264 = wp::load(var_262);
                    var_263 = wp::mul(var_259, var_264);
                    var_266 = wp::add(var_3, var_265);
                    wp::array_store(var_qfrc_damper_out, var_0, var_266, var_263);
                    // qfrc_damper_out[worldid, dofid + 1] = -damping * qvel_in[worldid, dofid + 1]       <L 169>
                    var_267 = wp::neg(var_19);
                    var_269 = wp::add(var_3, var_268);
                    var_270 = wp::address(var_qvel_in, var_0, var_269);
                    var_272 = wp::load(var_270);
                    var_271 = wp::mul(var_267, var_272);
                    var_274 = wp::add(var_3, var_273);
                    wp::array_store(var_qfrc_damper_out, var_0, var_274, var_271);
                    // qfrc_damper_out[worldid, dofid + 2] = -damping * qvel_in[worldid, dofid + 2]       <L 170>
                    var_275 = wp::neg(var_19);
                    var_277 = wp::add(var_3, var_276);
                    var_278 = wp::address(var_qvel_in, var_0, var_277);
                    var_280 = wp::load(var_278);
                    var_279 = wp::mul(var_275, var_280);
                    var_282 = wp::add(var_3, var_281);
                    wp::array_store(var_qfrc_damper_out, var_0, var_282, var_279);
                }
            }
            var_283 = wp::where(var_201, var_256, var_133);
            var_284 = wp::where(var_201, var_257, var_115);
            var_285 = wp::where(var_201, var_258, var_128);
            if (!var_201) {
                // if has_stiffness:                                                              <L 173>
                if (var_26) {
                    // fdif = qpos_in[worldid, qposid] - qpos_spring[qpos_spring_id, qposid]       <L 174>
                    var_286 = wp::address(var_qpos_in, var_0, var_43);
                    var_287 = wp::address(var_qpos_spring, var_49, var_43);
                    var_289 = wp::load(var_286);
                    var_290 = wp::load(var_287);
                    var_288 = wp::sub(var_289, var_290);
                    // qfrc_spring_out[worldid, dofid] = -stiffness * fdif                        <L 175>
                    var_291 = wp::neg(var_11);
                    var_292 = wp::mul(var_291, var_288);
                    wp::array_store(var_qfrc_spring_out, var_0, var_3, var_292);
                }
                // if has_damping:                                                                <L 178>
                if (var_32) {
                    // qfrc_damper_out[worldid, dofid] = -damping * qvel_in[worldid, dofid]       <L 179>
                    var_293 = wp::neg(var_19);
                    var_294 = wp::address(var_qvel_in, var_0, var_3);
                    var_296 = wp::load(var_294);
                    var_295 = wp::mul(var_293, var_296);
                    wp::array_store(var_qfrc_damper_out, var_0, var_3, var_295);
                }
            }
        }
        var_297 = wp::where(var_51, var_133, var_283);
        var_298 = wp::where(var_51, var_115, var_284);
        var_299 = wp::where(var_51, var_128, var_285);
    }
}



extern "C" __global__ void _spring_damper_tendon_passive_7f94acac_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::float32> var_tendon_stiffness,
    wp::array_t<wp::float32> var_tendon_damping,
    wp::array_t<wp::vec_t<2, wp::float32>> var_tendon_lengthspring,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::array_t<wp::float32> var_ten_length_in,
    wp::array_t<wp::float32> var_ten_velocity_in,
    bool var_dsbl_spring,
    bool var_dsbl_damper,
    wp::array_t<wp::float32> var_qfrc_spring_out,
    wp::array_t<wp::float32> var_qfrc_damper_out)
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
        wp::int32 var_2;
        wp::shape_t* var_3;
        const wp::int32 var_4 = 0;
        wp::int32 var_5;
        wp::shape_t var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::shape_t* var_11;
        const wp::int32 var_12 = 0;
        wp::int32 var_13;
        wp::shape_t var_14;
        wp::int32 var_15;
        wp::float32* var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        const wp::float32 var_19 = 0.0;
        bool var_20;
        bool var_21;
        bool var_22;
        const wp::float32 var_23 = 0.0;
        bool var_24;
        bool var_25;
        bool var_26;
        bool var_27;
        bool var_28;
        bool var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        bool var_33;
        wp::int32* var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        wp::float32* var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        wp::int32* var_41;
        wp::int32 var_42;
        wp::int32 var_43;
        wp::float32* var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::shape_t* var_47;
        const wp::int32 var_48 = 0;
        wp::int32 var_49;
        wp::shape_t var_50;
        wp::int32 var_51;
        wp::vec_t<2, wp::float32>* var_52;
        wp::vec_t<2, wp::float32> var_53;
        wp::vec_t<2, wp::float32> var_54;
        const wp::int32 var_55 = 0;
        wp::float32 var_56;
        const wp::int32 var_57 = 1;
        wp::float32 var_58;
        bool var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        bool var_62;
        wp::float32 var_63;
        wp::float32 var_64;
        wp::float32 var_65;
        const wp::float32 var_66 = 0.0;
        wp::float32 var_67;
        wp::float32 var_68;
        wp::slice_t var_69;
        const wp::int32 var_70 = 0;
        wp::array_t<wp::float32> var_71;
        wp::float32 var_72;
        wp::float32 var_73;
        wp::float32 var_74;
        wp::float32* var_75;
        wp::float32 var_76;
        wp::float32 var_77;
        wp::slice_t var_78;
        const wp::int32 var_79 = 0;
        wp::array_t<wp::float32> var_80;
        wp::float32 var_81;
        wp::float32 var_82;
        //---------
        // forward
        // def _spring_damper_tendon_passive(                                                     <L 183>
        // worldid, tenid, dofid_sparse = wp.tid()                                                <L 202>
        builtin_tid3d(var_0, var_1, var_2);
        // stiffness = tendon_stiffness[worldid % tendon_stiffness.shape[0], tenid]               <L 204>
        var_3 = &(var_tendon_stiffness.shape);
        var_6 = wp::load(var_3);
        var_5 = wp::extract(var_6, var_4);
        var_7 = wp::mod(var_0, var_5);
        var_8 = wp::address(var_tendon_stiffness, var_7, var_1);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // damping = tendon_damping[worldid % tendon_damping.shape[0], tenid]                     <L 205>
        var_11 = &(var_tendon_damping.shape);
        var_14 = wp::load(var_11);
        var_13 = wp::extract(var_14, var_12);
        var_15 = wp::mod(var_0, var_13);
        var_16 = wp::address(var_tendon_damping, var_15, var_1);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // has_stiffness = stiffness != 0.0 and not dsbl_spring                                   <L 207>
        var_20 = (var_9 != var_19);
        var_21 = wp::unot(var_dsbl_spring);
        var_22 = var_20 && var_21;
        // has_damping = damping != 0.0 and not dsbl_damper                                       <L 208>
        var_24 = (var_17 != var_23);
        var_25 = wp::unot(var_dsbl_damper);
        var_26 = var_24 && var_25;
        // if not has_stiffness and not has_damping:                                              <L 210>
        var_27 = wp::unot(var_22);
        var_28 = wp::unot(var_26);
        var_29 = var_27 && var_28;
        if (var_29) {
            // return                                                                             <L 211>
            continue;
        }
        // rownnz = ten_J_rownnz[tenid]                                                           <L 213>
        var_30 = wp::address(var_ten_J_rownnz, var_1);
        var_32 = wp::load(var_30);
        var_31 = wp::copy(var_32);
        // if dofid_sparse >= rownnz:                                                             <L 214>
        var_33 = (var_2 >= var_31);
        if (var_33) {
            // return                                                                             <L 215>
            continue;
        }
        // rowadr = ten_J_rowadr[tenid]                                                           <L 216>
        var_34 = wp::address(var_ten_J_rowadr, var_1);
        var_36 = wp::load(var_34);
        var_35 = wp::copy(var_36);
        // sparseid = rowadr + dofid_sparse                                                       <L 217>
        var_37 = wp::add(var_35, var_2);
        // J = ten_J_in[worldid, sparseid]                                                        <L 218>
        var_38 = wp::address(var_ten_J_in, var_0, var_37);
        var_40 = wp::load(var_38);
        var_39 = wp::copy(var_40);
        // dofid = ten_J_colind[sparseid]                                                         <L 219>
        var_41 = wp::address(var_ten_J_colind, var_37);
        var_43 = wp::load(var_41);
        var_42 = wp::copy(var_43);
        // if has_stiffness:                                                                      <L 221>
        if (var_22) {
            // length = ten_length_in[worldid, tenid]                                             <L 223>
            var_44 = wp::address(var_ten_length_in, var_0, var_1);
            var_46 = wp::load(var_44);
            var_45 = wp::copy(var_46);
            // lengthspring = tendon_lengthspring[worldid % tendon_lengthspring.shape[0], tenid]       <L 224>
            var_47 = &(var_tendon_lengthspring.shape);
            var_50 = wp::load(var_47);
            var_49 = wp::extract(var_50, var_48);
            var_51 = wp::mod(var_0, var_49);
            var_52 = wp::address(var_tendon_lengthspring, var_51, var_1);
            var_54 = wp::load(var_52);
            var_53 = wp::copy(var_54);
            // lower = lengthspring[0]                                                            <L 225>
            var_56 = wp::extract(var_53, var_55);
            // upper = lengthspring[1]                                                            <L 226>
            var_58 = wp::extract(var_53, var_57);
            // if length > upper:                                                                 <L 228>
            var_59 = (var_45 > var_58);
            if (var_59) {
                // frc_spring = stiffness * (upper - length)                                      <L 229>
                var_60 = wp::sub(var_58, var_45);
                var_61 = wp::mul(var_9, var_60);
            }
            if (!var_59) {
                // elif length < lower:                                                           <L 230>
                var_62 = (var_45 < var_56);
                if (var_62) {
                    // frc_spring = stiffness * (lower - length)                                  <L 231>
                    var_63 = wp::sub(var_56, var_45);
                    var_64 = wp::mul(var_9, var_63);
                }
                var_65 = wp::where(var_62, var_64, var_61);
                if (!var_62) {
                    // frc_spring = 0.0                                                           <L 233>
                }
                var_67 = wp::where(var_62, var_65, var_66);
            }
            var_68 = wp::where(var_59, var_61, var_67);
            // wp.atomic_add(qfrc_spring_out[worldid], dofid, J * frc_spring)                     <L 236>
            var_69 = wp::slice_t(var_0, var_0, var_70);
            var_71 = wp::view(var_qfrc_spring_out, var_69);
            var_72 = wp::mul(var_39, var_68);
            var_73 = wp::atomic_add(var_71, var_42, var_72);
        }
        // if has_damping:                                                                        <L 238>
        if (var_26) {
            // frc_damper = -damping * ten_velocity_in[worldid, tenid]                            <L 240>
            var_74 = wp::neg(var_17);
            var_75 = wp::address(var_ten_velocity_in, var_0, var_1);
            var_77 = wp::load(var_75);
            var_76 = wp::mul(var_74, var_77);
            // wp.atomic_add(qfrc_damper_out[worldid], dofid, J * frc_damper)                     <L 243>
            var_78 = wp::slice_t(var_0, var_0, var_79);
            var_80 = wp::view(var_qfrc_damper_out, var_78);
            var_81 = wp::mul(var_39, var_76);
            var_82 = wp::atomic_add(var_80, var_42, var_81);
        }
    }
}



extern "C" __global__ void _fluid_force_d9b21c6f_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_opt_wind,
    wp::array_t<wp::float32> var_opt_density,
    wp::array_t<wp::float32> var_opt_viscosity,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_geomnum,
    wp::array_t<wp::int32> var_body_geomadr,
    wp::array_t<wp::float32> var_body_mass,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_inertia,
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::float32> var_geom_fluid,
    wp::array_t<bool> var_body_fluid_ellipsoid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_fluid_applied_out)
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
        const wp::float32 var_2 = 0.0;
        wp::vec_t<3, wp::float32> var_3;
        const wp::float32 var_4 = 0.0;
        wp::vec_t<3, wp::float32> var_5;
        wp::vec_t<6, wp::float32> var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        wp::shape_t* var_9;
        const wp::int32 var_10 = 0;
        wp::int32 var_11;
        wp::shape_t var_12;
        wp::int32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        const wp::float32 var_17 = 1e-15;
        bool var_18;
        wp::shape_t* var_19;
        const wp::int32 var_20 = 0;
        wp::int32 var_21;
        wp::shape_t var_22;
        wp::int32 var_23;
        wp::vec_t<3, wp::float32>* var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::shape_t* var_27;
        const wp::int32 var_28 = 0;
        wp::int32 var_29;
        wp::shape_t var_30;
        wp::int32 var_31;
        wp::float32* var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::shape_t* var_35;
        const wp::int32 var_36 = 0;
        wp::int32 var_37;
        wp::shape_t var_38;
        wp::int32 var_39;
        wp::float32* var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::vec_t<3, wp::float32>* var_43;
        wp::vec_t<3, wp::float32> var_44;
        wp::vec_t<3, wp::float32> var_45;
        wp::mat_t<3, 3, wp::float32>* var_46;
        wp::mat_t<3, 3, wp::float32> var_47;
        wp::mat_t<3, 3, wp::float32> var_48;
        wp::mat_t<3, 3, wp::float32> var_49;
        wp::vec_t<6, wp::float32>* var_50;
        wp::vec_t<6, wp::float32> var_51;
        wp::vec_t<6, wp::float32> var_52;
        wp::vec_t<3, wp::float32> var_53;
        wp::vec_t<3, wp::float32> var_54;
        wp::int32* var_55;
        wp::vec_t<3, wp::float32>* var_56;
        wp::int32 var_57;
        wp::vec_t<3, wp::float32> var_58;
        wp::vec_t<3, wp::float32> var_59;
        wp::vec_t<3, wp::float32> var_60;
        wp::vec_t<3, wp::float32> var_61;
        wp::vec_t<3, wp::float32> var_62;
        bool* var_63;
        bool var_64;
        const wp::float32 var_65 = 0.0;
        wp::vec_t<3, wp::float32> var_66;
        const wp::float32 var_67 = 0.0;
        wp::vec_t<3, wp::float32> var_68;
        wp::int32* var_69;
        wp::int32 var_70;
        wp::int32 var_71;
        wp::int32* var_72;
        wp::int32 var_73;
        wp::int32 var_74;
        wp::range_t var_75;
        wp::int32 var_76;
        wp::int32 var_77;
        const wp::int32 var_78 = 0;
        wp::float32* var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        const wp::float32 var_82 = 0.0;
        bool var_83;
        wp::shape_t* var_84;
        const wp::int32 var_85 = 0;
        wp::int32 var_86;
        wp::shape_t var_87;
        wp::int32 var_88;
        wp::vec_t<3, wp::float32>* var_89;
        wp::vec_t<3, wp::float32> var_90;
        wp::vec_t<3, wp::float32> var_91;
        wp::int32* var_92;
        wp::vec_t<3, wp::float32> var_93;
        wp::int32 var_94;
        wp::mat_t<3, 3, wp::float32>* var_95;
        wp::mat_t<3, 3, wp::float32> var_96;
        wp::mat_t<3, 3, wp::float32> var_97;
        wp::mat_t<3, 3, wp::float32> var_98;
        wp::vec_t<3, wp::float32>* var_99;
        wp::vec_t<3, wp::float32> var_100;
        wp::vec_t<3, wp::float32> var_101;
        wp::vec_t<3, wp::float32> var_102;
        wp::vec_t<3, wp::float32> var_103;
        wp::vec_t<3, wp::float32> var_104;
        wp::vec_t<3, wp::float32> var_105;
        wp::vec_t<3, wp::float32> var_106;
        const wp::int32 var_107 = 0;
        wp::float32 var_108;
        const wp::int32 var_109 = 1;
        wp::float32 var_110;
        const wp::int32 var_111 = 2;
        wp::float32 var_112;
        bool var_113;
        wp::vec_t<3, wp::float32> var_114;
        wp::vec_t<3, wp::float32> var_115;
        wp::vec_t<3, wp::float32> var_116;
        const wp::float32 var_117 = 0.0;
        wp::vec_t<3, wp::float32> var_118;
        const wp::float32 var_119 = 0.0;
        wp::vec_t<3, wp::float32> var_120;
        const wp::float32 var_121 = 0.0;
        bool var_122;
        const wp::int32 var_123 = 6;
        wp::float32* var_124;
        const wp::int32 var_125 = 7;
        wp::float32* var_126;
        const wp::int32 var_127 = 8;
        wp::float32* var_128;
        wp::vec_t<3, wp::float32> var_129;
        wp::float32 var_130;
        wp::float32 var_131;
        wp::float32 var_132;
        const wp::int32 var_133 = 9;
        wp::float32* var_134;
        const wp::int32 var_135 = 10;
        wp::float32* var_136;
        const wp::int32 var_137 = 11;
        wp::float32* var_138;
        wp::vec_t<3, wp::float32> var_139;
        wp::float32 var_140;
        wp::float32 var_141;
        wp::float32 var_142;
        const wp::int32 var_143 = 0;
        wp::float32 var_144;
        wp::float32 var_145;
        const wp::int32 var_146 = 0;
        wp::float32 var_147;
        wp::float32 var_148;
        const wp::int32 var_149 = 1;
        wp::float32 var_150;
        wp::float32 var_151;
        const wp::int32 var_152 = 1;
        wp::float32 var_153;
        wp::float32 var_154;
        const wp::int32 var_155 = 2;
        wp::float32 var_156;
        wp::float32 var_157;
        const wp::int32 var_158 = 2;
        wp::float32 var_159;
        wp::float32 var_160;
        wp::vec_t<3, wp::float32> var_161;
        const wp::int32 var_162 = 0;
        wp::float32 var_163;
        wp::float32 var_164;
        const wp::int32 var_165 = 0;
        wp::float32 var_166;
        wp::float32 var_167;
        const wp::int32 var_168 = 1;
        wp::float32 var_169;
        wp::float32 var_170;
        const wp::int32 var_171 = 1;
        wp::float32 var_172;
        wp::float32 var_173;
        const wp::int32 var_174 = 2;
        wp::float32 var_175;
        wp::float32 var_176;
        const wp::int32 var_177 = 2;
        wp::float32 var_178;
        wp::float32 var_179;
        wp::vec_t<3, wp::float32> var_180;
        wp::vec_t<3, wp::float32> var_181;
        wp::vec_t<3, wp::float32> var_182;
        wp::vec_t<3, wp::float32> var_183;
        wp::vec_t<3, wp::float32> var_184;
        wp::vec_t<3, wp::float32> var_185;
        wp::vec_t<3, wp::float32> var_186;
        wp::vec_t<3, wp::float32> var_187;
        wp::vec_t<3, wp::float32> var_188;
        const wp::int32 var_189 = 5;
        wp::float32* var_190;
        wp::float32 var_191;
        wp::float32 var_192;
        const wp::int32 var_193 = 4;
        wp::float32* var_194;
        wp::float32 var_195;
        wp::float32 var_196;
        const wp::int32 var_197 = 1;
        wp::float32* var_198;
        wp::float32 var_199;
        wp::float32 var_200;
        const wp::int32 var_201 = 2;
        wp::float32* var_202;
        wp::float32 var_203;
        wp::float32 var_204;
        const wp::int32 var_205 = 3;
        wp::float32* var_206;
        wp::float32 var_207;
        wp::float32 var_208;
        const wp::float32 var_209 = 4.1887902047863905;
        const wp::int32 var_210 = 0;
        wp::float32 var_211;
        wp::float32 var_212;
        const wp::int32 var_213 = 1;
        wp::float32 var_214;
        wp::float32 var_215;
        const wp::int32 var_216 = 2;
        wp::float32 var_217;
        wp::float32 var_218;
        const wp::int32 var_219 = 0;
        wp::float32 var_220;
        const wp::int32 var_221 = 1;
        wp::float32 var_222;
        wp::float32 var_223;
        const wp::int32 var_224 = 2;
        wp::float32 var_225;
        wp::float32 var_226;
        const wp::int32 var_227 = 0;
        wp::float32 var_228;
        const wp::int32 var_229 = 1;
        wp::float32 var_230;
        wp::float32 var_231;
        const wp::int32 var_232 = 2;
        wp::float32 var_233;
        wp::float32 var_234;
        const wp::int32 var_235 = 0;
        wp::float32 var_236;
        const wp::int32 var_237 = 1;
        wp::float32 var_238;
        wp::float32 var_239;
        const wp::int32 var_240 = 2;
        wp::float32 var_241;
        wp::float32 var_242;
        wp::float32 var_243;
        wp::float32 var_244;
        const wp::float32 var_245 = 3.141592653589793;
        wp::float32 var_246;
        wp::float32 var_247;
        wp::float32 var_248;
        wp::vec_t<3, wp::float32> var_249;
        wp::float32 var_250;
        wp::float32 var_251;
        wp::vec_t<3, wp::float32> var_252;
        const wp::int32 var_253 = 1;
        wp::float32 var_254;
        const wp::int32 var_255 = 2;
        wp::float32 var_256;
        wp::float32 var_257;
        const wp::int32 var_258 = 2;
        wp::float32 var_259;
        const wp::int32 var_260 = 0;
        wp::float32 var_261;
        wp::float32 var_262;
        const wp::int32 var_263 = 0;
        wp::float32 var_264;
        const wp::int32 var_265 = 1;
        wp::float32 var_266;
        wp::float32 var_267;
        wp::float32 var_268;
        const wp::int32 var_269 = 0;
        wp::float32 var_270;
        wp::float32 var_271;
        wp::float32 var_272;
        wp::float32 var_273;
        const wp::int32 var_274 = 1;
        wp::float32 var_275;
        wp::float32 var_276;
        wp::float32 var_277;
        wp::float32 var_278;
        wp::float32 var_279;
        const wp::int32 var_280 = 2;
        wp::float32 var_281;
        wp::float32 var_282;
        wp::float32 var_283;
        wp::float32 var_284;
        const wp::int32 var_285 = 0;
        wp::float32 var_286;
        wp::float32 var_287;
        wp::float32 var_288;
        const wp::int32 var_289 = 1;
        wp::float32 var_290;
        wp::float32 var_291;
        wp::float32 var_292;
        wp::float32 var_293;
        const wp::int32 var_294 = 2;
        wp::float32 var_295;
        wp::float32 var_296;
        wp::float32 var_297;
        wp::float32 var_298;
        const wp::float32 var_299 = 0.0;
        const wp::float32 var_300 = 0.0;
        bool var_301;
        bool var_302;
        bool var_303;
        const wp::float32 var_304 = 3.141592653589793;
        wp::float32 var_305;
        wp::float32 var_306;
        wp::float32 var_307;
        wp::float32 var_308;
        bool var_309;
        wp::float32 var_310;
        wp::float32 var_311;
        wp::float32 var_312;
        wp::float32 var_313;
        wp::float32 var_314;
        wp::float32 var_315;
        wp::float32 var_316;
        const wp::int32 var_317 = 0;
        wp::float32 var_318;
        wp::float32 var_319;
        wp::float32 var_320;
        const wp::int32 var_321 = 1;
        wp::float32 var_322;
        wp::float32 var_323;
        wp::float32 var_324;
        const wp::int32 var_325 = 2;
        wp::float32 var_326;
        wp::float32 var_327;
        wp::vec_t<3, wp::float32> var_328;
        const wp::float32 var_329 = 0.0;
        wp::vec_t<3, wp::float32> var_330;
        const wp::float32 var_331 = 0.0;
        bool var_332;
        const wp::float32 var_333 = 0.0;
        bool var_334;
        bool var_335;
        bool var_336;
        wp::vec_t<3, wp::float32> var_337;
        wp::float32 var_338;
        wp::float32 var_339;
        wp::float32 var_340;
        wp::vec_t<3, wp::float32> var_341;
        wp::vec_t<3, wp::float32> var_342;
        wp::vec_t<3, wp::float32> var_343;
        const wp::float32 var_344 = 0.6666666666666666;
        const wp::int32 var_345 = 0;
        wp::float32 var_346;
        const wp::int32 var_347 = 1;
        wp::float32 var_348;
        wp::float32 var_349;
        const wp::int32 var_350 = 2;
        wp::float32 var_351;
        wp::float32 var_352;
        wp::float32 var_353;
        const wp::float32 var_354 = 9.42477796076938;
        wp::float32 var_355;
        const wp::float32 var_356 = 3.141592653589793;
        wp::float32 var_357;
        wp::float32 var_358;
        wp::float32 var_359;
        const wp::float32 var_360 = 1.6755160819145563;
        wp::float32 var_361;
        wp::float32 var_362;
        wp::float32 var_363;
        const wp::int32 var_364 = 0;
        wp::float32 var_365;
        const wp::int32 var_366 = 1;
        wp::float32 var_367;
        const wp::int32 var_368 = 2;
        wp::float32 var_369;
        const wp::int32 var_370 = 0;
        wp::float32 var_371;
        wp::float32 var_372;
        wp::float32 var_373;
        wp::float32 var_374;
        wp::float32 var_375;
        wp::float32 var_376;
        const wp::int32 var_377 = 1;
        wp::float32 var_378;
        wp::float32 var_379;
        wp::float32 var_380;
        wp::float32 var_381;
        wp::float32 var_382;
        wp::float32 var_383;
        const wp::int32 var_384 = 2;
        wp::float32 var_385;
        wp::float32 var_386;
        wp::float32 var_387;
        wp::float32 var_388;
        wp::float32 var_389;
        wp::float32 var_390;
        wp::vec_t<3, wp::float32> var_391;
        wp::float32 var_392;
        wp::float32 var_393;
        wp::float32 var_394;
        wp::float32 var_395;
        wp::float32 var_396;
        wp::float32 var_397;
        wp::float32 var_398;
        wp::float32 var_399;
        wp::float32 var_400;
        wp::float32 var_401;
        wp::float32 var_402;
        wp::float32 var_403;
        wp::vec_t<3, wp::float32> var_404;
        wp::vec_t<3, wp::float32> var_405;
        wp::vec_t<3, wp::float32> var_406;
        wp::vec_t<3, wp::float32> var_407;
        wp::vec_t<3, wp::float32> var_408;
        wp::vec_t<3, wp::float32> var_409;
        wp::vec_t<3, wp::float32> var_410;
        wp::vec_t<3, wp::float32> var_411;
        wp::vec_t<3, wp::float32> var_412;
        wp::vec_t<3, wp::float32> var_413;
        wp::vec_t<3, wp::float32> var_414;
        wp::vec_t<3, wp::float32> var_415;
        wp::vec_t<6, wp::float32> var_416;
        bool var_417;
        wp::vec_t<3, wp::float32> var_418;
        wp::vec_t<3, wp::float32> var_419;
        const wp::int32 var_420 = 0;
        wp::float32 var_421;
        const wp::int32 var_422 = 1;
        wp::float32 var_423;
        const wp::int32 var_424 = 2;
        wp::float32 var_425;
        bool var_426;
        wp::vec_t<3, wp::float32> var_427;
        wp::vec_t<3, wp::float32> var_428;
        wp::vec_t<3, wp::float32> var_429;
        const wp::float32 var_430 = 0.0;
        wp::vec_t<3, wp::float32> var_431;
        const wp::float32 var_432 = 0.0;
        wp::vec_t<3, wp::float32> var_433;
        const wp::float32 var_434 = 0.0;
        bool var_435;
        const wp::float32 var_436 = 0.0;
        bool var_437;
        bool var_438;
        wp::shape_t* var_439;
        const wp::int32 var_440 = 0;
        wp::int32 var_441;
        wp::shape_t var_442;
        wp::int32 var_443;
        wp::vec_t<3, wp::float32>* var_444;
        wp::vec_t<3, wp::float32> var_445;
        wp::vec_t<3, wp::float32> var_446;
        wp::shape_t* var_447;
        const wp::int32 var_448 = 0;
        wp::int32 var_449;
        wp::shape_t var_450;
        wp::int32 var_451;
        wp::float32* var_452;
        wp::float32 var_453;
        wp::float32 var_454;
        const wp::float32 var_455 = 6.0;
        wp::float32 var_456;
        const wp::int32 var_457 = 1;
        wp::float32 var_458;
        const wp::int32 var_459 = 2;
        wp::float32 var_460;
        wp::float32 var_461;
        const wp::int32 var_462 = 0;
        wp::float32 var_463;
        wp::float32 var_464;
        wp::float32 var_465;
        wp::float32 var_466;
        wp::float32 var_467;
        const wp::int32 var_468 = 0;
        wp::float32 var_469;
        const wp::int32 var_470 = 2;
        wp::float32 var_471;
        wp::float32 var_472;
        const wp::int32 var_473 = 1;
        wp::float32 var_474;
        wp::float32 var_475;
        wp::float32 var_476;
        wp::float32 var_477;
        wp::float32 var_478;
        const wp::int32 var_479 = 0;
        wp::float32 var_480;
        const wp::int32 var_481 = 1;
        wp::float32 var_482;
        wp::float32 var_483;
        const wp::int32 var_484 = 2;
        wp::float32 var_485;
        wp::float32 var_486;
        wp::float32 var_487;
        wp::float32 var_488;
        wp::float32 var_489;
        wp::float32 var_490;
        wp::float32 var_491;
        wp::float32 var_492;
        const wp::float32 var_493 = 3.0;
        wp::float32 var_494;
        wp::vec_t<3, wp::float32> var_495;
        const wp::float32 var_496 = 3.0;
        wp::float32 var_497;
        wp::vec_t<3, wp::float32> var_498;
        const wp::float32 var_499 = 3.141592653589793;
        wp::vec_t<3, wp::float32> var_500;
        wp::vec_t<3, wp::float32> var_501;
        const wp::float32 var_502 = 3.0;
        const wp::float32 var_503 = -3.0;
        wp::vec_t<3, wp::float32> var_504;
        wp::vec_t<3, wp::float32> var_505;
        const wp::float32 var_506 = 3.141592653589793;
        wp::vec_t<3, wp::float32> var_507;
        wp::vec_t<3, wp::float32> var_508;
        wp::vec_t<3, wp::float32> var_509;
        wp::vec_t<3, wp::float32> var_510;
        const wp::float32 var_511 = 0.5;
        wp::float32 var_512;
        wp::float32 var_513;
        wp::float32 var_514;
        const wp::int32 var_515 = 0;
        wp::float32 var_516;
        wp::float32 var_517;
        wp::float32 var_518;
        const wp::int32 var_519 = 0;
        wp::float32 var_520;
        wp::float32 var_521;
        const wp::float32 var_522 = 0.5;
        wp::float32 var_523;
        wp::float32 var_524;
        wp::float32 var_525;
        const wp::int32 var_526 = 1;
        wp::float32 var_527;
        wp::float32 var_528;
        wp::float32 var_529;
        const wp::int32 var_530 = 1;
        wp::float32 var_531;
        wp::float32 var_532;
        const wp::float32 var_533 = 0.5;
        wp::float32 var_534;
        wp::float32 var_535;
        wp::float32 var_536;
        const wp::int32 var_537 = 2;
        wp::float32 var_538;
        wp::float32 var_539;
        wp::float32 var_540;
        const wp::int32 var_541 = 2;
        wp::float32 var_542;
        wp::float32 var_543;
        wp::vec_t<3, wp::float32> var_544;
        wp::vec_t<3, wp::float32> var_545;
        const wp::float32 var_546 = 64.0;
        wp::float32 var_547;
        const wp::float32 var_548 = 4.0;
        wp::float32 var_549;
        const wp::float32 var_550 = 4.0;
        wp::float32 var_551;
        const wp::float32 var_552 = 4.0;
        wp::float32 var_553;
        wp::float32 var_554;
        wp::float32 var_555;
        const wp::int32 var_556 = 0;
        wp::float32 var_557;
        wp::float32 var_558;
        wp::float32 var_559;
        const wp::int32 var_560 = 0;
        wp::float32 var_561;
        wp::float32 var_562;
        wp::float32 var_563;
        wp::float32 var_564;
        wp::float32 var_565;
        const wp::int32 var_566 = 1;
        wp::float32 var_567;
        wp::float32 var_568;
        wp::float32 var_569;
        const wp::int32 var_570 = 1;
        wp::float32 var_571;
        wp::float32 var_572;
        wp::float32 var_573;
        wp::float32 var_574;
        wp::float32 var_575;
        const wp::int32 var_576 = 2;
        wp::float32 var_577;
        wp::float32 var_578;
        wp::float32 var_579;
        const wp::int32 var_580 = 2;
        wp::float32 var_581;
        wp::float32 var_582;
        wp::float32 var_583;
        wp::vec_t<3, wp::float32> var_584;
        wp::vec_t<3, wp::float32> var_585;
        wp::vec_t<3, wp::float32> var_586;
        wp::vec_t<3, wp::float32> var_587;
        wp::float32 var_588;
        wp::vec_t<3, wp::float32> var_589;
        wp::vec_t<3, wp::float32> var_590;
        wp::vec_t<6, wp::float32> var_591;
        //---------
        // forward
        // def _fluid_force(                                                                      <L 276>
        // worldid, bodyid = wp.tid()                                                             <L 301>
        builtin_tid2d(var_0, var_1);
        // zero_force = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))                             <L 302>
        var_3 = wp::vec_t<3, wp::float32>(var_2);
        var_5 = wp::vec_t<3, wp::float32>(var_4);
        var_6 = wp::vec_t<6, wp::float32>(var_3, var_5);
        // if bodyid == 0:                                                                        <L 304>
        var_8 = (var_1 == var_7);
        if (var_8) {
            // fluid_applied_out[worldid, bodyid] = zero_force                                    <L 305>
            wp::array_store(var_fluid_applied_out, var_0, var_1, var_6);
            // return                                                                             <L 306>
            continue;
        }
        // mass = body_mass[worldid % body_mass.shape[0], bodyid]                                 <L 309>
        var_9 = &(var_body_mass.shape);
        var_12 = wp::load(var_9);
        var_11 = wp::extract(var_12, var_10);
        var_13 = wp::mod(var_0, var_11);
        var_14 = wp::address(var_body_mass, var_13, var_1);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // if mass < MJ_MINVAL:                                                                   <L 310>
        var_18 = (var_15 < var_17);
        if (var_18) {
            // fluid_applied_out[worldid, bodyid] = zero_force                                    <L 311>
            wp::array_store(var_fluid_applied_out, var_0, var_1, var_6);
            // return                                                                             <L 312>
            continue;
        }
        // wind = opt_wind[worldid % opt_wind.shape[0]]                                           <L 314>
        var_19 = &(var_opt_wind.shape);
        var_22 = wp::load(var_19);
        var_21 = wp::extract(var_22, var_20);
        var_23 = wp::mod(var_0, var_21);
        var_24 = wp::address(var_opt_wind, var_23);
        var_26 = wp::load(var_24);
        var_25 = wp::copy(var_26);
        // density = opt_density[worldid % opt_density.shape[0]]                                  <L 315>
        var_27 = &(var_opt_density.shape);
        var_30 = wp::load(var_27);
        var_29 = wp::extract(var_30, var_28);
        var_31 = wp::mod(var_0, var_29);
        var_32 = wp::address(var_opt_density, var_31);
        var_34 = wp::load(var_32);
        var_33 = wp::copy(var_34);
        // viscosity = opt_viscosity[worldid % opt_viscosity.shape[0]]                            <L 316>
        var_35 = &(var_opt_viscosity.shape);
        var_38 = wp::load(var_35);
        var_37 = wp::extract(var_38, var_36);
        var_39 = wp::mod(var_0, var_37);
        var_40 = wp::address(var_opt_viscosity, var_39);
        var_42 = wp::load(var_40);
        var_41 = wp::copy(var_42);
        // xipos = xipos_in[worldid, bodyid]                                                      <L 319>
        var_43 = wp::address(var_xipos_in, var_0, var_1);
        var_45 = wp::load(var_43);
        var_44 = wp::copy(var_45);
        // rot = ximat_in[worldid, bodyid]                                                        <L 320>
        var_46 = wp::address(var_ximat_in, var_0, var_1);
        var_48 = wp::load(var_46);
        var_47 = wp::copy(var_48);
        // rotT = wp.transpose(rot)                                                               <L 321>
        var_49 = wp::transpose(var_47);
        // cvel = cvel_in[worldid, bodyid]                                                        <L 322>
        var_50 = wp::address(var_cvel_in, var_0, var_1);
        var_52 = wp::load(var_50);
        var_51 = wp::copy(var_52);
        // ang_global = wp.spatial_top(cvel)                                                      <L 323>
        var_53 = wp::spatial_top(var_51);
        // lin_global = wp.spatial_bottom(cvel)                                                   <L 324>
        var_54 = wp::spatial_bottom(var_51);
        // subtree_root = subtree_com_in[worldid, body_rootid[bodyid]]                            <L 325>
        var_55 = wp::address(var_body_rootid, var_1);
        var_57 = wp::load(var_55);
        var_56 = wp::address(var_subtree_com_in, var_0, var_57);
        var_59 = wp::load(var_56);
        var_58 = wp::copy(var_59);
        // lin_com = lin_global - wp.cross(xipos - subtree_root, ang_global)                      <L 326>
        var_60 = wp::sub(var_44, var_58);
        var_61 = wp::cross(var_60, var_53);
        var_62 = wp::sub(var_54, var_61);
        // if body_fluid_ellipsoid[bodyid]:                                                       <L 328>
        var_63 = wp::address(var_body_fluid_ellipsoid, var_1);
        var_64 = wp::load(var_63);
        if (var_64) {
            // force_global = wp.vec3(0.0)                                                        <L 329>
            var_66 = wp::vec_t<3, wp::float32>(var_65);
            // torque_global = wp.vec3(0.0)                                                       <L 330>
            var_68 = wp::vec_t<3, wp::float32>(var_67);
            // start = body_geomadr[bodyid]                                                       <L 332>
            var_69 = wp::address(var_body_geomadr, var_1);
            var_71 = wp::load(var_69);
            var_70 = wp::copy(var_71);
            // count = body_geomnum[bodyid]                                                       <L 333>
            var_72 = wp::address(var_body_geomnum, var_1);
            var_74 = wp::load(var_72);
            var_73 = wp::copy(var_74);
            // for i in range(count):                                                             <L 335>
            var_75 = wp::range(var_73);
            start_for_2:;
                if (iter_cmp(var_75) == 0) goto end_for_2;
                var_76 = wp::iter_next(var_75);
                // geomid = start + i                                                             <L 336>
                var_77 = wp::add(var_70, var_76);
                // coef = geom_fluid[geomid, 0]                                                   <L 337>
                var_79 = wp::address(var_geom_fluid, var_77, var_78);
                var_81 = wp::load(var_79);
                var_80 = wp::copy(var_81);
                // if coef <= 0.0:                                                                <L 338>
                var_83 = (var_80 <= var_82);
                if (var_83) {
                    // continue                                                                   <L 339>
                    goto start_for_2;
                }
                // size = geom_size[worldid % geom_size.shape[0], geomid]                         <L 341>
                var_84 = &(var_geom_size.shape);
                var_87 = wp::load(var_84);
                var_86 = wp::extract(var_87, var_85);
                var_88 = wp::mod(var_0, var_86);
                var_89 = wp::address(var_geom_size, var_88, var_77);
                var_91 = wp::load(var_89);
                var_90 = wp::copy(var_91);
                // semiaxes = _geom_semiaxes(size, geom_type[geomid])                             <L 342>
                var_92 = wp::address(var_geom_type, var_77);
                var_94 = wp::load(var_92);
                var_93 = _geom_semiaxes_0(var_90, var_94);
                // geom_rot = geom_xmat_in[worldid, geomid]                                       <L 343>
                var_95 = wp::address(var_geom_xmat_in, var_0, var_77);
                var_97 = wp::load(var_95);
                var_96 = wp::copy(var_97);
                // geom_rotT = wp.transpose(geom_rot)                                             <L 344>
                var_98 = wp::transpose(var_96);
                // geom_pos = geom_xpos_in[worldid, geomid]                                       <L 345>
                var_99 = wp::address(var_geom_xpos_in, var_0, var_77);
                var_101 = wp::load(var_99);
                var_100 = wp::copy(var_101);
                // lin_point = lin_com + wp.cross(ang_global, geom_pos - xipos)                   <L 347>
                var_102 = wp::sub(var_100, var_44);
                var_103 = wp::cross(var_53, var_102);
                var_104 = wp::add(var_62, var_103);
                // l_ang = geom_rotT @ ang_global                                                 <L 349>
                var_105 = wp::mul(var_98, var_53);
                // l_lin = geom_rotT @ lin_point                                                  <L 350>
                var_106 = wp::mul(var_98, var_104);
                // if wind[0] or wind[1] or wind[2]:                                              <L 352>
                var_108 = wp::extract(var_25, var_107);
                var_110 = wp::extract(var_25, var_109);
                var_112 = wp::extract(var_25, var_111);
                var_113 = var_108 || var_110 || var_112;
                if (var_113) {
                    // l_lin -= geom_rotT @ wind                                                  <L 353>
                    var_114 = wp::mul(var_98, var_25);
                    var_115 = wp::sub(var_106, var_114);
                }
                var_116 = wp::where(var_113, var_115, var_106);
                // lfrc_torque = wp.vec3(0.0)                                                     <L 355>
                var_118 = wp::vec_t<3, wp::float32>(var_117);
                // lfrc_force = wp.vec3(0.0)                                                      <L 356>
                var_120 = wp::vec_t<3, wp::float32>(var_119);
                // if density > 0.0:                                                              <L 358>
                var_122 = (var_33 > var_121);
                if (var_122) {
                    // virtual_mass = wp.vec3(geom_fluid[geomid, 6], geom_fluid[geomid, 7], geom_fluid[geomid, 8])       <L 360>
                    var_124 = wp::address(var_geom_fluid, var_77, var_123);
                    var_126 = wp::address(var_geom_fluid, var_77, var_125);
                    var_128 = wp::address(var_geom_fluid, var_77, var_127);
                    var_130 = wp::load(var_124);
                    var_131 = wp::load(var_126);
                    var_132 = wp::load(var_128);
                    var_129 = wp::vec_t<3, wp::float32>(var_130, var_131, var_132);
                    // virtual_inertia = wp.vec3(geom_fluid[geomid, 9], geom_fluid[geomid, 10], geom_fluid[geomid, 11])       <L 361>
                    var_134 = wp::address(var_geom_fluid, var_77, var_133);
                    var_136 = wp::address(var_geom_fluid, var_77, var_135);
                    var_138 = wp::address(var_geom_fluid, var_77, var_137);
                    var_140 = wp::load(var_134);
                    var_141 = wp::load(var_136);
                    var_142 = wp::load(var_138);
                    var_139 = wp::vec_t<3, wp::float32>(var_140, var_141, var_142);
                    // virtual_lin_mom = wp.vec3(                                                 <L 363>
                    // density * virtual_mass[0] * l_lin[0],                                      <L 364>
                    var_144 = wp::extract(var_129, var_143);
                    var_145 = wp::mul(var_33, var_144);
                    var_147 = wp::extract(var_116, var_146);
                    var_148 = wp::mul(var_145, var_147);
                    // density * virtual_mass[1] * l_lin[1],                                      <L 365>
                    var_150 = wp::extract(var_129, var_149);
                    var_151 = wp::mul(var_33, var_150);
                    var_153 = wp::extract(var_116, var_152);
                    var_154 = wp::mul(var_151, var_153);
                    // density * virtual_mass[2] * l_lin[2],                                      <L 366>
                    var_156 = wp::extract(var_129, var_155);
                    var_157 = wp::mul(var_33, var_156);
                    var_159 = wp::extract(var_116, var_158);
                    var_160 = wp::mul(var_157, var_159);
                    var_161 = wp::vec_t<3, wp::float32>(var_148, var_154, var_160);
                    // virtual_ang_mom = wp.vec3(                                                 <L 368>
                    // density * virtual_inertia[0] * l_ang[0],                                   <L 369>
                    var_163 = wp::extract(var_139, var_162);
                    var_164 = wp::mul(var_33, var_163);
                    var_166 = wp::extract(var_105, var_165);
                    var_167 = wp::mul(var_164, var_166);
                    // density * virtual_inertia[1] * l_ang[1],                                   <L 370>
                    var_169 = wp::extract(var_139, var_168);
                    var_170 = wp::mul(var_33, var_169);
                    var_172 = wp::extract(var_105, var_171);
                    var_173 = wp::mul(var_170, var_172);
                    // density * virtual_inertia[2] * l_ang[2],                                   <L 371>
                    var_175 = wp::extract(var_139, var_174);
                    var_176 = wp::mul(var_33, var_175);
                    var_178 = wp::extract(var_105, var_177);
                    var_179 = wp::mul(var_176, var_178);
                    var_180 = wp::vec_t<3, wp::float32>(var_167, var_173, var_179);
                    // added_mass_force = wp.cross(virtual_lin_mom, l_ang)                        <L 374>
                    var_181 = wp::cross(var_161, var_105);
                    // added_mass_torque = wp.cross(virtual_lin_mom, l_lin) + wp.cross(virtual_ang_mom, l_ang)       <L 375>
                    var_182 = wp::cross(var_161, var_116);
                    var_183 = wp::cross(var_180, var_105);
                    var_184 = wp::add(var_182, var_183);
                    // lfrc_force += added_mass_force                                             <L 377>
                    var_185 = wp::add(var_120, var_181);
                    // lfrc_torque += added_mass_torque                                           <L 378>
                    var_186 = wp::add(var_118, var_184);
                }
                var_187 = wp::where(var_122, var_186, var_118);
                var_188 = wp::where(var_122, var_185, var_120);
                // magnus_coef = geom_fluid[geomid, 5]                                            <L 381>
                var_190 = wp::address(var_geom_fluid, var_77, var_189);
                var_192 = wp::load(var_190);
                var_191 = wp::copy(var_192);
                // kutta_coef = geom_fluid[geomid, 4]                                             <L 382>
                var_194 = wp::address(var_geom_fluid, var_77, var_193);
                var_196 = wp::load(var_194);
                var_195 = wp::copy(var_196);
                // blunt_drag_coef = geom_fluid[geomid, 1]                                        <L 383>
                var_198 = wp::address(var_geom_fluid, var_77, var_197);
                var_200 = wp::load(var_198);
                var_199 = wp::copy(var_200);
                // slender_drag_coef = geom_fluid[geomid, 2]                                      <L 384>
                var_202 = wp::address(var_geom_fluid, var_77, var_201);
                var_204 = wp::load(var_202);
                var_203 = wp::copy(var_204);
                // ang_drag_coef = geom_fluid[geomid, 3]                                          <L 385>
                var_206 = wp::address(var_geom_fluid, var_77, var_205);
                var_208 = wp::load(var_206);
                var_207 = wp::copy(var_208);
                // volume = wp.static(4.0 / 3.0 * wp.pi) * semiaxes[0] * semiaxes[1] * semiaxes[2]       <L 387>
                var_211 = wp::extract(var_93, var_210);
                var_212 = wp::mul(var_209, var_211);
                var_214 = wp::extract(var_93, var_213);
                var_215 = wp::mul(var_212, var_214);
                var_217 = wp::extract(var_93, var_216);
                var_218 = wp::mul(var_215, var_217);
                // d_max = wp.max(wp.max(semiaxes[0], semiaxes[1]), semiaxes[2])                  <L 388>
                var_220 = wp::extract(var_93, var_219);
                var_222 = wp::extract(var_93, var_221);
                var_223 = wp::max(var_220, var_222);
                var_225 = wp::extract(var_93, var_224);
                var_226 = wp::max(var_223, var_225);
                // d_min = wp.min(wp.min(semiaxes[0], semiaxes[1]), semiaxes[2])                  <L 389>
                var_228 = wp::extract(var_93, var_227);
                var_230 = wp::extract(var_93, var_229);
                var_231 = wp::min(var_228, var_230);
                var_233 = wp::extract(var_93, var_232);
                var_234 = wp::min(var_231, var_233);
                // d_mid = semiaxes[0] + semiaxes[1] + semiaxes[2] - d_max - d_min                <L 390>
                var_236 = wp::extract(var_93, var_235);
                var_238 = wp::extract(var_93, var_237);
                var_239 = wp::add(var_236, var_238);
                var_241 = wp::extract(var_93, var_240);
                var_242 = wp::add(var_239, var_241);
                var_243 = wp::sub(var_242, var_226);
                var_244 = wp::sub(var_243, var_234);
                // A_max = wp.pi * d_max * d_mid                                                  <L 391>
                var_246 = wp::mul(var_245, var_226);
                var_247 = wp::mul(var_246, var_244);
                // lin_speed = wp.length(l_lin)                                                   <L 393>
                var_248 = wp::length(var_116);
                // magnus_force = wp.cross(l_ang, l_lin) * (magnus_coef * density * volume)       <L 395>
                var_249 = wp::cross(var_105, var_116);
                var_250 = wp::mul(var_191, var_33);
                var_251 = wp::mul(var_250, var_218);
                var_252 = wp::mul(var_249, var_251);
                // s12 = semiaxes[1] * semiaxes[2]                                                <L 397>
                var_254 = wp::extract(var_93, var_253);
                var_256 = wp::extract(var_93, var_255);
                var_257 = wp::mul(var_254, var_256);
                // s20 = semiaxes[2] * semiaxes[0]                                                <L 398>
                var_259 = wp::extract(var_93, var_258);
                var_261 = wp::extract(var_93, var_260);
                var_262 = wp::mul(var_259, var_261);
                // s01 = semiaxes[0] * semiaxes[1]                                                <L 399>
                var_264 = wp::extract(var_93, var_263);
                var_266 = wp::extract(var_93, var_265);
                var_267 = wp::mul(var_264, var_266);
                // proj_denom = _pow4(s12) * _pow2(l_lin[0]) + _pow4(s20) * _pow2(l_lin[1]) + _pow4(s01) * _pow2(l_lin[2])       <L 401>
                var_268 = _pow4_0(var_257);
                var_270 = wp::extract(var_116, var_269);
                var_271 = _pow2_0(var_270);
                var_272 = wp::mul(var_268, var_271);
                var_273 = _pow4_0(var_262);
                var_275 = wp::extract(var_116, var_274);
                var_276 = _pow2_0(var_275);
                var_277 = wp::mul(var_273, var_276);
                var_278 = wp::add(var_272, var_277);
                var_279 = _pow4_0(var_267);
                var_281 = wp::extract(var_116, var_280);
                var_282 = _pow2_0(var_281);
                var_283 = wp::mul(var_279, var_282);
                var_284 = wp::add(var_278, var_283);
                // proj_num = _pow2(s12 * l_lin[0]) + _pow2(s20 * l_lin[1]) + _pow2(s01 * l_lin[2])       <L 402>
                var_286 = wp::extract(var_116, var_285);
                var_287 = wp::mul(var_257, var_286);
                var_288 = _pow2_0(var_287);
                var_290 = wp::extract(var_116, var_289);
                var_291 = wp::mul(var_262, var_290);
                var_292 = _pow2_0(var_291);
                var_293 = wp::add(var_288, var_292);
                var_295 = wp::extract(var_116, var_294);
                var_296 = wp::mul(var_267, var_295);
                var_297 = _pow2_0(var_296);
                var_298 = wp::add(var_293, var_297);
                // A_proj = 0.0                                                                   <L 404>
                // cos_alpha = 0.0                                                                <L 405>
                // if proj_num > MJ_MINVAL and proj_denom > MJ_MINVAL:                            <L 406>
                var_301 = (var_298 > var_17);
                var_302 = (var_284 > var_17);
                var_303 = var_301 && var_302;
                if (var_303) {
                    // A_proj = wp.pi * wp.sqrt(proj_denom / wp.max(MJ_MINVAL, proj_num))         <L 407>
                    var_305 = wp::max(var_17, var_298);
                    var_306 = wp::div(var_284, var_305);
                    var_307 = wp::sqrt(var_306);
                    var_308 = wp::mul(var_304, var_307);
                    // if lin_speed > MJ_MINVAL:                                                  <L 408>
                    var_309 = (var_248 > var_17);
                    if (var_309) {
                        // cos_alpha = proj_num / wp.max(MJ_MINVAL, lin_speed * proj_denom)       <L 409>
                        var_310 = wp::mul(var_248, var_284);
                        var_311 = wp::max(var_17, var_310);
                        var_312 = wp::div(var_298, var_311);
                    }
                    var_313 = wp::where(var_309, var_312, var_300);
                }
                var_314 = wp::where(var_303, var_308, var_299);
                var_315 = wp::where(var_303, var_313, var_300);
                // norm = wp.vec3(                                                                <L 411>
                // _pow2(s12) * l_lin[0],                                                         <L 412>
                var_316 = _pow2_0(var_257);
                var_318 = wp::extract(var_116, var_317);
                var_319 = wp::mul(var_316, var_318);
                // _pow2(s20) * l_lin[1],                                                         <L 413>
                var_320 = _pow2_0(var_262);
                var_322 = wp::extract(var_116, var_321);
                var_323 = wp::mul(var_320, var_322);
                // _pow2(s01) * l_lin[2],                                                         <L 414>
                var_324 = _pow2_0(var_267);
                var_326 = wp::extract(var_116, var_325);
                var_327 = wp::mul(var_324, var_326);
                var_328 = wp::vec_t<3, wp::float32>(var_319, var_323, var_327);
                // kutta_force = wp.vec3(0.0)                                                     <L 417>
                var_330 = wp::vec_t<3, wp::float32>(var_329);
                // if density > 0.0 and kutta_coef != 0.0 and lin_speed > MJ_MINVAL:              <L 418>
                var_332 = (var_33 > var_331);
                var_334 = (var_195 != var_333);
                var_335 = (var_248 > var_17);
                var_336 = var_332 && var_334 && var_335;
                if (var_336) {
                    // kutta_circ = wp.cross(norm, l_lin) * (kutta_coef * density * cos_alpha * A_proj)       <L 419>
                    var_337 = wp::cross(var_328, var_116);
                    var_338 = wp::mul(var_195, var_33);
                    var_339 = wp::mul(var_338, var_315);
                    var_340 = wp::mul(var_339, var_314);
                    var_341 = wp::mul(var_337, var_340);
                    // kutta_force = wp.cross(kutta_circ, l_lin)                                  <L 420>
                    var_342 = wp::cross(var_341, var_116);
                }
                var_343 = wp::where(var_336, var_342, var_330);
                // eq_sphere_D = wp.static(2.0 / 3.0) * (semiaxes[0] + semiaxes[1] + semiaxes[2])       <L 422>
                var_346 = wp::extract(var_93, var_345);
                var_348 = wp::extract(var_93, var_347);
                var_349 = wp::add(var_346, var_348);
                var_351 = wp::extract(var_93, var_350);
                var_352 = wp::add(var_349, var_351);
                var_353 = wp::mul(var_344, var_352);
                // lin_visc_force_coef = wp.static(3.0 * wp.pi) * eq_sphere_D                     <L 423>
                var_355 = wp::mul(var_354, var_353);
                // lin_visc_torq_coef = wp.pi * eq_sphere_D * eq_sphere_D * eq_sphere_D           <L 424>
                var_357 = wp::mul(var_356, var_353);
                var_358 = wp::mul(var_357, var_353);
                var_359 = wp::mul(var_358, var_353);
                // I_max = wp.static(8.0 / 15.0 * wp.pi) * d_mid * _pow4(d_max)                   <L 426>
                var_361 = wp::mul(var_360, var_244);
                var_362 = _pow4_0(var_226);
                var_363 = wp::mul(var_361, var_362);
                // II0 = _ellipsoid_max_moment(semiaxes, 0)                                       <L 427>
                var_365 = _ellipsoid_max_moment_0(var_93, var_364);
                // II1 = _ellipsoid_max_moment(semiaxes, 1)                                       <L 428>
                var_367 = _ellipsoid_max_moment_0(var_93, var_366);
                // II2 = _ellipsoid_max_moment(semiaxes, 2)                                       <L 429>
                var_369 = _ellipsoid_max_moment_0(var_93, var_368);
                // mom_visc = wp.vec3(                                                            <L 431>
                // l_ang[0] * (ang_drag_coef * II0 + slender_drag_coef * (I_max - II0)),          <L 432>
                var_371 = wp::extract(var_105, var_370);
                var_372 = wp::mul(var_207, var_365);
                var_373 = wp::sub(var_363, var_365);
                var_374 = wp::mul(var_203, var_373);
                var_375 = wp::add(var_372, var_374);
                var_376 = wp::mul(var_371, var_375);
                // l_ang[1] * (ang_drag_coef * II1 + slender_drag_coef * (I_max - II1)),          <L 433>
                var_378 = wp::extract(var_105, var_377);
                var_379 = wp::mul(var_207, var_367);
                var_380 = wp::sub(var_363, var_367);
                var_381 = wp::mul(var_203, var_380);
                var_382 = wp::add(var_379, var_381);
                var_383 = wp::mul(var_378, var_382);
                // l_ang[2] * (ang_drag_coef * II2 + slender_drag_coef * (I_max - II2)),          <L 434>
                var_385 = wp::extract(var_105, var_384);
                var_386 = wp::mul(var_207, var_369);
                var_387 = wp::sub(var_363, var_369);
                var_388 = wp::mul(var_203, var_387);
                var_389 = wp::add(var_386, var_388);
                var_390 = wp::mul(var_385, var_389);
                var_391 = wp::vec_t<3, wp::float32>(var_376, var_383, var_390);
                // drag_lin_coef = viscosity * lin_visc_force_coef + density * lin_speed * (       <L 437>
                var_392 = wp::mul(var_41, var_355);
                var_393 = wp::mul(var_33, var_248);
                // A_proj * blunt_drag_coef + slender_drag_coef * (A_max - A_proj)                <L 438>
                var_394 = wp::mul(var_314, var_199);
                var_395 = wp::sub(var_247, var_314);
                var_396 = wp::mul(var_203, var_395);
                var_397 = wp::add(var_394, var_396);
                var_398 = wp::mul(var_393, var_397);
                var_399 = wp::add(var_392, var_398);
                // drag_ang_coef = viscosity * lin_visc_torq_coef + density * wp.length(mom_visc)       <L 440>
                var_400 = wp::mul(var_41, var_359);
                var_401 = wp::length(var_391);
                var_402 = wp::mul(var_33, var_401);
                var_403 = wp::add(var_400, var_402);
                // lfrc_torque -= drag_ang_coef * l_ang                                           <L 442>
                var_404 = wp::mul(var_403, var_105);
                var_405 = wp::sub(var_187, var_404);
                // lfrc_force += magnus_force + kutta_force - drag_lin_coef * l_lin               <L 443>
                var_406 = wp::add(var_252, var_343);
                var_407 = wp::mul(var_399, var_116);
                var_408 = wp::sub(var_406, var_407);
                var_409 = wp::add(var_188, var_408);
                // lfrc_torque *= coef                                                            <L 445>
                var_410 = wp::mul(var_405, var_80);
                // lfrc_force *= coef                                                             <L 446>
                var_411 = wp::mul(var_409, var_80);
                // torque_global += geom_rot @ lfrc_torque                                        <L 449>
                var_412 = wp::mul(var_96, var_410);
                var_413 = wp::add(var_68, var_412);
                // force_global += geom_rot @ lfrc_force                                          <L 450>
                var_414 = wp::mul(var_96, var_411);
                var_415 = wp::add(var_66, var_414);
                wp::assign(var_66, var_415);
                wp::assign(var_68, var_413);
                goto start_for_2;
            end_for_2:;
            // fluid_applied_out[worldid, bodyid] = wp.spatial_vector(force_global, torque_global)       <L 452>
            var_416 = wp::vec_t<6, wp::float32>(var_66, var_68);
            wp::array_store(var_fluid_applied_out, var_0, var_1, var_416);
            // return                                                                             <L 453>
            continue;
        }
        var_417 = wp::load(var_63);
        // l_ang = rotT @ ang_global                                                              <L 455>
        var_418 = wp::mul(var_49, var_53);
        // l_lin = rotT @ lin_com                                                                 <L 456>
        var_419 = wp::mul(var_49, var_62);
        // if wind[0] or wind[1] or wind[2]:                                                      <L 458>
        var_421 = wp::extract(var_25, var_420);
        var_423 = wp::extract(var_25, var_422);
        var_425 = wp::extract(var_25, var_424);
        var_426 = var_421 || var_423 || var_425;
        if (var_426) {
            // l_lin -= rotT @ wind                                                               <L 459>
            var_427 = wp::mul(var_49, var_25);
            var_428 = wp::sub(var_419, var_427);
        }
        var_429 = wp::where(var_426, var_428, var_419);
        // lfrc_torque = wp.vec3(0.0)                                                             <L 461>
        var_431 = wp::vec_t<3, wp::float32>(var_430);
        // lfrc_force = wp.vec3(0.0)                                                              <L 462>
        var_433 = wp::vec_t<3, wp::float32>(var_432);
        // has_viscosity = viscosity > 0.0                                                        <L 464>
        var_435 = (var_41 > var_434);
        // has_density = density > 0.0                                                            <L 465>
        var_437 = (var_33 > var_436);
        // if has_viscosity or has_density:                                                       <L 467>
        var_438 = var_435 || var_437;
        if (var_438) {
            // inertia = body_inertia[worldid % body_inertia.shape[0], bodyid]                    <L 468>
            var_439 = &(var_body_inertia.shape);
            var_442 = wp::load(var_439);
            var_441 = wp::extract(var_442, var_440);
            var_443 = wp::mod(var_0, var_441);
            var_444 = wp::address(var_body_inertia, var_443, var_1);
            var_446 = wp::load(var_444);
            var_445 = wp::copy(var_446);
            // mass = body_mass[worldid % body_mass.shape[0], bodyid]                             <L 469>
            var_447 = &(var_body_mass.shape);
            var_450 = wp::load(var_447);
            var_449 = wp::extract(var_450, var_448);
            var_451 = wp::mod(var_0, var_449);
            var_452 = wp::address(var_body_mass, var_451, var_1);
            var_454 = wp::load(var_452);
            var_453 = wp::copy(var_454);
            // scl = 6.0 / mass                                                                   <L 470>
            var_456 = wp::div(var_455, var_453);
            // box0 = wp.sqrt(wp.max(MJ_MINVAL, inertia[1] + inertia[2] - inertia[0]) * scl)       <L 471>
            var_458 = wp::extract(var_445, var_457);
            var_460 = wp::extract(var_445, var_459);
            var_461 = wp::add(var_458, var_460);
            var_463 = wp::extract(var_445, var_462);
            var_464 = wp::sub(var_461, var_463);
            var_465 = wp::max(var_17, var_464);
            var_466 = wp::mul(var_465, var_456);
            var_467 = wp::sqrt(var_466);
            // box1 = wp.sqrt(wp.max(MJ_MINVAL, inertia[0] + inertia[2] - inertia[1]) * scl)       <L 472>
            var_469 = wp::extract(var_445, var_468);
            var_471 = wp::extract(var_445, var_470);
            var_472 = wp::add(var_469, var_471);
            var_474 = wp::extract(var_445, var_473);
            var_475 = wp::sub(var_472, var_474);
            var_476 = wp::max(var_17, var_475);
            var_477 = wp::mul(var_476, var_456);
            var_478 = wp::sqrt(var_477);
            // box2 = wp.sqrt(wp.max(MJ_MINVAL, inertia[0] + inertia[1] - inertia[2]) * scl)       <L 473>
            var_480 = wp::extract(var_445, var_479);
            var_482 = wp::extract(var_445, var_481);
            var_483 = wp::add(var_480, var_482);
            var_485 = wp::extract(var_445, var_484);
            var_486 = wp::sub(var_483, var_485);
            var_487 = wp::max(var_17, var_486);
            var_488 = wp::mul(var_487, var_456);
            var_489 = wp::sqrt(var_488);
        }
        var_490 = wp::where(var_438, var_453, var_15);
        // if has_viscosity:                                                                      <L 475>
        if (var_435) {
            // diam = (box0 + box1 + box2) / 3.0                                                  <L 476>
            var_491 = wp::add(var_467, var_478);
            var_492 = wp::add(var_491, var_489);
            var_494 = wp::div(var_492, var_493);
            // lfrc_torque = -l_ang * wp.pow(diam, 3.0) * wp.pi * viscosity                       <L 477>
            var_495 = wp::neg(var_418);
            var_497 = wp::pow(var_494, var_496);
            var_498 = wp::mul(var_495, var_497);
            var_500 = wp::mul(var_498, var_499);
            var_501 = wp::mul(var_500, var_41);
            // lfrc_force = -3.0 * l_lin * diam * wp.pi * viscosity                               <L 478>
            var_504 = wp::mul(var_503, var_429);
            var_505 = wp::mul(var_504, var_494);
            var_507 = wp::mul(var_505, var_506);
            var_508 = wp::mul(var_507, var_41);
        }
        var_509 = wp::where(var_435, var_501, var_431);
        var_510 = wp::where(var_435, var_508, var_433);
        // if has_density:                                                                        <L 480>
        if (var_437) {
            // lfrc_force -= wp.vec3(                                                             <L 481>
            // 0.5 * density * box1 * box2 * wp.abs(l_lin[0]) * l_lin[0],                         <L 482>
            var_512 = wp::mul(var_511, var_33);
            var_513 = wp::mul(var_512, var_478);
            var_514 = wp::mul(var_513, var_489);
            var_516 = wp::extract(var_429, var_515);
            var_517 = wp::abs(var_516);
            var_518 = wp::mul(var_514, var_517);
            var_520 = wp::extract(var_429, var_519);
            var_521 = wp::mul(var_518, var_520);
            // 0.5 * density * box0 * box2 * wp.abs(l_lin[1]) * l_lin[1],                         <L 483>
            var_523 = wp::mul(var_522, var_33);
            var_524 = wp::mul(var_523, var_467);
            var_525 = wp::mul(var_524, var_489);
            var_527 = wp::extract(var_429, var_526);
            var_528 = wp::abs(var_527);
            var_529 = wp::mul(var_525, var_528);
            var_531 = wp::extract(var_429, var_530);
            var_532 = wp::mul(var_529, var_531);
            // 0.5 * density * box0 * box1 * wp.abs(l_lin[2]) * l_lin[2],                         <L 484>
            var_534 = wp::mul(var_533, var_33);
            var_535 = wp::mul(var_534, var_467);
            var_536 = wp::mul(var_535, var_478);
            var_538 = wp::extract(var_429, var_537);
            var_539 = wp::abs(var_538);
            var_540 = wp::mul(var_536, var_539);
            var_542 = wp::extract(var_429, var_541);
            var_543 = wp::mul(var_540, var_542);
            var_544 = wp::vec_t<3, wp::float32>(var_521, var_532, var_543);
            // lfrc_force -= wp.vec3(                                                             <L 481>
            var_545 = wp::sub(var_510, var_544);
            // scl = density / 64.0                                                               <L 487>
            var_547 = wp::div(var_33, var_546);
            // box0_pow4 = wp.pow(box0, 4.0)                                                      <L 488>
            var_549 = wp::pow(var_467, var_548);
            // box1_pow4 = wp.pow(box1, 4.0)                                                      <L 489>
            var_551 = wp::pow(var_478, var_550);
            // box2_pow4 = wp.pow(box2, 4.0)                                                      <L 490>
            var_553 = wp::pow(var_489, var_552);
            // lfrc_torque -= wp.vec3(                                                            <L 491>
            // box0 * (box1_pow4 + box2_pow4) * wp.abs(l_ang[0]) * l_ang[0] * scl,                <L 492>
            var_554 = wp::add(var_551, var_553);
            var_555 = wp::mul(var_467, var_554);
            var_557 = wp::extract(var_418, var_556);
            var_558 = wp::abs(var_557);
            var_559 = wp::mul(var_555, var_558);
            var_561 = wp::extract(var_418, var_560);
            var_562 = wp::mul(var_559, var_561);
            var_563 = wp::mul(var_562, var_547);
            // box1 * (box0_pow4 + box2_pow4) * wp.abs(l_ang[1]) * l_ang[1] * scl,                <L 493>
            var_564 = wp::add(var_549, var_553);
            var_565 = wp::mul(var_478, var_564);
            var_567 = wp::extract(var_418, var_566);
            var_568 = wp::abs(var_567);
            var_569 = wp::mul(var_565, var_568);
            var_571 = wp::extract(var_418, var_570);
            var_572 = wp::mul(var_569, var_571);
            var_573 = wp::mul(var_572, var_547);
            // box2 * (box0_pow4 + box1_pow4) * wp.abs(l_ang[2]) * l_ang[2] * scl,                <L 494>
            var_574 = wp::add(var_549, var_551);
            var_575 = wp::mul(var_489, var_574);
            var_577 = wp::extract(var_418, var_576);
            var_578 = wp::abs(var_577);
            var_579 = wp::mul(var_575, var_578);
            var_581 = wp::extract(var_418, var_580);
            var_582 = wp::mul(var_579, var_581);
            var_583 = wp::mul(var_582, var_547);
            var_584 = wp::vec_t<3, wp::float32>(var_563, var_573, var_583);
            // lfrc_torque -= wp.vec3(                                                            <L 491>
            var_585 = wp::sub(var_509, var_584);
        }
        var_586 = wp::where(var_437, var_585, var_509);
        var_587 = wp::where(var_437, var_545, var_510);
        var_588 = wp::where(var_437, var_547, var_456);
        // torque_global = rot @ lfrc_torque                                                      <L 497>
        var_589 = wp::mul(var_47, var_586);
        // force_global = rot @ lfrc_force                                                        <L 498>
        var_590 = wp::mul(var_47, var_587);
        // fluid_applied_out[worldid, bodyid] = wp.spatial_vector(force_global, torque_global)       <L 500>
        var_591 = wp::vec_t<6, wp::float32>(var_590, var_589);
        wp::array_store(var_fluid_applied_out, var_0, var_1, var_591);
    }
}



extern "C" __global__ void _gravity_force_13ae428a_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_opt_gravity,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::float32> var_body_mass,
    wp::array_t<wp::float32> var_body_gravcomp,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::float32> var_qfrc_gravcomp_out)
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
        wp::int32 var_2;
        const wp::int32 var_3 = 1;
        wp::int32 var_4;
        wp::shape_t* var_5;
        const wp::int32 var_6 = 0;
        wp::int32 var_7;
        wp::shape_t var_8;
        wp::int32 var_9;
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        wp::shape_t* var_13;
        const wp::int32 var_14 = 0;
        wp::int32 var_15;
        wp::shape_t var_16;
        wp::int32 var_17;
        wp::vec_t<3, wp::float32>* var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32> var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::shape_t* var_22;
        const wp::int32 var_23 = 0;
        wp::int32 var_24;
        wp::shape_t var_25;
        wp::int32 var_26;
        wp::float32* var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::float32 var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::vec_t<3, wp::float32>* var_31;
        wp::vec_t<3, wp::float32> var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::vec_t<3, wp::float32> var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::slice_t var_36;
        const wp::int32 var_37 = 0;
        wp::array_t<wp::float32> var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        //---------
        // forward
        // def _gravity_force(                                                                    <L 247>
        // worldid, bodyid, dofid = wp.tid()                                                      <L 262>
        builtin_tid3d(var_0, var_1, var_2);
        // bodyid += 1  # skip world body                                                         <L 263>
        var_4 = wp::add(var_1, var_3);
        // gravcomp = body_gravcomp[worldid % body_gravcomp.shape[0], bodyid]                     <L 264>
        var_5 = &(var_body_gravcomp.shape);
        var_8 = wp::load(var_5);
        var_7 = wp::extract(var_8, var_6);
        var_9 = wp::mod(var_0, var_7);
        var_10 = wp::address(var_body_gravcomp, var_9, var_4);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // gravity = opt_gravity[worldid % opt_gravity.shape[0]]                                  <L 265>
        var_13 = &(var_opt_gravity.shape);
        var_16 = wp::load(var_13);
        var_15 = wp::extract(var_16, var_14);
        var_17 = wp::mod(var_0, var_15);
        var_18 = wp::address(var_opt_gravity, var_17);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // if gravcomp:                                                                           <L 267>
        if (var_11) {
            // force = -gravity * body_mass[worldid % body_mass.shape[0], bodyid] * gravcomp       <L 268>
            var_21 = wp::neg(var_19);
            var_22 = &(var_body_mass.shape);
            var_25 = wp::load(var_22);
            var_24 = wp::extract(var_25, var_23);
            var_26 = wp::mod(var_0, var_24);
            var_27 = wp::address(var_body_mass, var_26, var_4);
            var_29 = wp::load(var_27);
            var_28 = wp::mul(var_21, var_29);
            var_30 = wp::mul(var_28, var_11);
            // pos = xipos_in[worldid, bodyid]                                                    <L 269>
            var_31 = wp::address(var_xipos_in, var_0, var_4);
            var_33 = wp::load(var_31);
            var_32 = wp::copy(var_33);
            // jac, _ = support.jac_dof(body_parentid, body_rootid, dof_bodyid, subtree_com_in, cdof_in, pos, bodyid, dofid, worldid)       <L 270>
            jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_subtree_com_in, var_cdof_in, var_32, var_4, var_2, var_0, var_34, var_35);
            // wp.atomic_add(qfrc_gravcomp_out[worldid], dofid, wp.dot(jac, force))               <L 272>
            var_36 = wp::slice_t(var_0, var_0, var_37);
            var_38 = wp::view(var_qfrc_gravcomp_out, var_36);
            var_39 = wp::dot(var_34, var_30);
            var_40 = wp::atomic_add(var_38, var_2, var_39);
        }
    }
}



extern "C" __global__ void _qfrc_passive_229e062a_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_jnt_actgravcomp,
    wp::array_t<wp::int32> var_dof_jntid,
    bool var_has_fluid,
    wp::array_t<wp::float32> var_qfrc_spring_in,
    wp::array_t<wp::float32> var_qfrc_damper_in,
    wp::array_t<wp::float32> var_qfrc_gravcomp_in,
    wp::array_t<wp::float32> var_qfrc_fluid_in,
    bool var_gravcomp,
    wp::array_t<wp::float32> var_qfrc_passive_out)
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
        wp::float32* var_5;
        wp::float32 var_6;
        wp::float32 var_7;
        wp::int32* var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        bool var_11;
        wp::int32 var_12;
        bool var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        //---------
        // forward
        // def _qfrc_passive(                                                                     <L 536>
        // worldid, dofid = wp.tid()                                                              <L 551>
        builtin_tid2d(var_0, var_1);
        // qfrc_passive = qfrc_spring_in[worldid, dofid]                                          <L 552>
        var_2 = wp::address(var_qfrc_spring_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // qfrc_passive += qfrc_damper_in[worldid, dofid]                                         <L 553>
        var_5 = wp::address(var_qfrc_damper_in, var_0, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::add(var_3, var_7);
        // if gravcomp and not jnt_actgravcomp[dof_jntid[dofid]]:                                 <L 556>
        var_8 = wp::address(var_dof_jntid, var_1);
        var_10 = wp::load(var_8);
        var_9 = wp::address(var_jnt_actgravcomp, var_10);
        var_12 = wp::load(var_9);
        var_11 = wp::unot(var_12);
        var_13 = var_gravcomp && var_11;
        if (var_13) {
            // qfrc_passive += qfrc_gravcomp_in[worldid, dofid]                                   <L 557>
            var_14 = wp::address(var_qfrc_gravcomp_in, var_0, var_1);
            var_16 = wp::load(var_14);
            var_15 = wp::add(var_6, var_16);
        }
        var_17 = wp::where(var_13, var_15, var_6);
        // if has_fluid:                                                                          <L 560>
        if (var_has_fluid) {
            // qfrc_passive += qfrc_fluid_in[worldid, dofid]                                      <L 561>
            var_18 = wp::address(var_qfrc_fluid_in, var_0, var_1);
            var_20 = wp::load(var_18);
            var_19 = wp::add(var_17, var_20);
        }
        var_21 = wp::where(var_has_fluid, var_19, var_17);
        // qfrc_passive_out[worldid, dofid] = qfrc_passive                                        <L 563>
        wp::array_store(var_qfrc_passive_out, var_0, var_1, var_21);
    }
}



extern "C" __global__ void _flex_bending_5321e7d5_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_nflex,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_flex_dim,
    wp::array_t<wp::int32> var_flex_vertadr,
    wp::array_t<wp::int32> var_flex_edgeadr,
    wp::array_t<wp::int32> var_flex_edgenum,
    wp::array_t<wp::int32> var_flex_vertbodyid,
    wp::array_t<wp::vec_t<2, wp::int32>> var_flex_edge,
    wp::array_t<wp::vec_t<2, wp::int32>> var_flex_edgeflap,
    wp::array_t<wp::float32> var_flex_bending,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexvert_xpos_in,
    wp::array_t<wp::float32> var_qfrc_spring_out)
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
        const wp::int32 var_2 = 4;
        wp::range_t var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        const wp::int32 var_8 = 0;
        bool var_9;
        wp::int32* var_10;
        bool var_11;
        wp::int32 var_12;
        bool var_13;
        wp::int32 var_14;
        wp::int32* var_15;
        const wp::int32 var_16 = 2;
        bool var_17;
        wp::int32 var_18;
        wp::vec_t<2, wp::int32>* var_19;
        const wp::int32 var_20 = 1;
        wp::int32 var_21;
        wp::vec_t<2, wp::int32> var_22;
        const wp::int32 var_23 = 1;
        const wp::int32 var_24 = -1;
        bool var_25;
        wp::int32* var_26;
        wp::vec_t<2, wp::int32>* var_27;
        const wp::int32 var_28 = 0;
        wp::int32 var_29;
        wp::vec_t<2, wp::int32> var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        wp::int32* var_33;
        wp::vec_t<2, wp::int32>* var_34;
        const wp::int32 var_35 = 1;
        wp::int32 var_36;
        wp::vec_t<2, wp::int32> var_37;
        wp::int32 var_38;
        wp::int32 var_39;
        wp::int32* var_40;
        wp::vec_t<2, wp::int32>* var_41;
        const wp::int32 var_42 = 0;
        wp::int32 var_43;
        wp::vec_t<2, wp::int32> var_44;
        wp::int32 var_45;
        wp::int32 var_46;
        wp::int32* var_47;
        wp::vec_t<2, wp::int32>* var_48;
        const wp::int32 var_49 = 1;
        wp::int32 var_50;
        wp::vec_t<2, wp::int32> var_51;
        wp::int32 var_52;
        wp::int32 var_53;
        wp::vec_t<4, wp::int32> var_54;
        const wp::float32 var_55 = 0.0;
        const wp::int32 var_56 = 4;
        const wp::int32 var_57 = 3;
        wp::tuple_t<wp::int32, wp::int32> var_58;
        wp::mat_t<4, 3, wp::float32> var_59;
        const wp::int32 var_60 = 16;
        wp::float32* var_61;
        wp::float32 var_62;
        const wp::int32 var_63 = 0;
        wp::int32 var_64;
        wp::vec_t<3, wp::float32>* var_65;
        wp::vec_t<3, wp::float32> var_66;
        wp::vec_t<3, wp::float32> var_67;
        const wp::int32 var_68 = 1;
        wp::int32 var_69;
        wp::vec_t<3, wp::float32>* var_70;
        wp::vec_t<3, wp::float32> var_71;
        wp::vec_t<3, wp::float32> var_72;
        const wp::int32 var_73 = 2;
        wp::int32 var_74;
        wp::vec_t<3, wp::float32>* var_75;
        wp::vec_t<3, wp::float32> var_76;
        wp::vec_t<3, wp::float32> var_77;
        const wp::int32 var_78 = 3;
        wp::int32 var_79;
        wp::vec_t<3, wp::float32>* var_80;
        wp::vec_t<3, wp::float32> var_81;
        wp::vec_t<3, wp::float32> var_82;
        wp::vec_t<3, wp::float32> var_83;
        wp::vec_t<3, wp::float32> var_84;
        wp::vec_t<3, wp::float32> var_85;
        const wp::int32 var_86 = 1;
        wp::vec_t<3, wp::float32> var_87;
        wp::vec_t<3, wp::float32> var_88;
        wp::vec_t<3, wp::float32> var_89;
        const wp::int32 var_90 = 2;
        wp::vec_t<3, wp::float32> var_91;
        wp::vec_t<3, wp::float32> var_92;
        wp::vec_t<3, wp::float32> var_93;
        const wp::int32 var_94 = 3;
        const wp::int32 var_95 = 1;
        wp::vec_t<3, wp::float32> var_96;
        const wp::int32 var_97 = 2;
        wp::vec_t<3, wp::float32> var_98;
        wp::vec_t<3, wp::float32> var_99;
        const wp::int32 var_100 = 3;
        wp::vec_t<3, wp::float32> var_101;
        wp::vec_t<3, wp::float32> var_102;
        wp::vec_t<3, wp::float32> var_103;
        const wp::int32 var_104 = 0;
        wp::float32 var_105;
        const wp::float32 var_106 = 0.0;
        const wp::int32 var_107 = 3;
        wp::tuple_t<wp::int32, wp::int32> var_108;
        wp::mat_t<4, 3, wp::float32> var_109;
        const wp::int32 var_110 = 0;
        const wp::int32 var_111 = 0;
        const wp::int32 var_112 = 0;
        const wp::int32 var_113 = 4;
        wp::int32 var_114;
        wp::int32 var_115;
        wp::float32* var_116;
        wp::int32 var_117;
        wp::vec_t<3, wp::float32>* var_118;
        wp::float32 var_119;
        wp::vec_t<3, wp::float32> var_120;
        wp::float32 var_121;
        wp::float32 var_122;
        const wp::int32 var_123 = 1;
        const wp::int32 var_124 = 4;
        wp::int32 var_125;
        wp::int32 var_126;
        wp::float32* var_127;
        wp::int32 var_128;
        wp::vec_t<3, wp::float32>* var_129;
        wp::float32 var_130;
        wp::vec_t<3, wp::float32> var_131;
        wp::float32 var_132;
        wp::float32 var_133;
        const wp::int32 var_134 = 2;
        const wp::int32 var_135 = 4;
        wp::int32 var_136;
        wp::int32 var_137;
        wp::float32* var_138;
        wp::int32 var_139;
        wp::vec_t<3, wp::float32>* var_140;
        wp::float32 var_141;
        wp::vec_t<3, wp::float32> var_142;
        wp::float32 var_143;
        wp::float32 var_144;
        const wp::int32 var_145 = 3;
        const wp::int32 var_146 = 4;
        wp::int32 var_147;
        wp::int32 var_148;
        wp::float32* var_149;
        wp::int32 var_150;
        wp::vec_t<3, wp::float32>* var_151;
        wp::float32 var_152;
        wp::vec_t<3, wp::float32> var_153;
        wp::float32 var_154;
        wp::float32 var_155;
        const wp::int32 var_156 = 1;
        const wp::int32 var_157 = 0;
        const wp::int32 var_158 = 4;
        wp::int32 var_159;
        wp::int32 var_160;
        wp::float32* var_161;
        wp::int32 var_162;
        wp::vec_t<3, wp::float32>* var_163;
        wp::float32 var_164;
        wp::vec_t<3, wp::float32> var_165;
        wp::float32 var_166;
        wp::float32 var_167;
        const wp::int32 var_168 = 1;
        const wp::int32 var_169 = 4;
        wp::int32 var_170;
        wp::int32 var_171;
        wp::float32* var_172;
        wp::int32 var_173;
        wp::vec_t<3, wp::float32>* var_174;
        wp::float32 var_175;
        wp::vec_t<3, wp::float32> var_176;
        wp::float32 var_177;
        wp::float32 var_178;
        const wp::int32 var_179 = 2;
        const wp::int32 var_180 = 4;
        wp::int32 var_181;
        wp::int32 var_182;
        wp::float32* var_183;
        wp::int32 var_184;
        wp::vec_t<3, wp::float32>* var_185;
        wp::float32 var_186;
        wp::vec_t<3, wp::float32> var_187;
        wp::float32 var_188;
        wp::float32 var_189;
        const wp::int32 var_190 = 3;
        const wp::int32 var_191 = 4;
        wp::int32 var_192;
        wp::int32 var_193;
        wp::float32* var_194;
        wp::int32 var_195;
        wp::vec_t<3, wp::float32>* var_196;
        wp::float32 var_197;
        wp::vec_t<3, wp::float32> var_198;
        wp::float32 var_199;
        wp::float32 var_200;
        const wp::int32 var_201 = 2;
        const wp::int32 var_202 = 0;
        const wp::int32 var_203 = 4;
        wp::int32 var_204;
        wp::int32 var_205;
        wp::float32* var_206;
        wp::int32 var_207;
        wp::vec_t<3, wp::float32>* var_208;
        wp::float32 var_209;
        wp::vec_t<3, wp::float32> var_210;
        wp::float32 var_211;
        wp::float32 var_212;
        const wp::int32 var_213 = 1;
        const wp::int32 var_214 = 4;
        wp::int32 var_215;
        wp::int32 var_216;
        wp::float32* var_217;
        wp::int32 var_218;
        wp::vec_t<3, wp::float32>* var_219;
        wp::float32 var_220;
        wp::vec_t<3, wp::float32> var_221;
        wp::float32 var_222;
        wp::float32 var_223;
        const wp::int32 var_224 = 2;
        const wp::int32 var_225 = 4;
        wp::int32 var_226;
        wp::int32 var_227;
        wp::float32* var_228;
        wp::int32 var_229;
        wp::vec_t<3, wp::float32>* var_230;
        wp::float32 var_231;
        wp::vec_t<3, wp::float32> var_232;
        wp::float32 var_233;
        wp::float32 var_234;
        const wp::int32 var_235 = 3;
        const wp::int32 var_236 = 4;
        wp::int32 var_237;
        wp::int32 var_238;
        wp::float32* var_239;
        wp::int32 var_240;
        wp::vec_t<3, wp::float32>* var_241;
        wp::float32 var_242;
        wp::vec_t<3, wp::float32> var_243;
        wp::float32 var_244;
        wp::float32 var_245;
        const wp::int32 var_246 = 16;
        wp::float32* var_247;
        wp::float32 var_248;
        wp::float32 var_249;
        wp::float32 var_250;
        const wp::int32 var_251 = 1;
        const wp::int32 var_252 = 0;
        const wp::int32 var_253 = 0;
        const wp::int32 var_254 = 4;
        wp::int32 var_255;
        wp::int32 var_256;
        wp::float32* var_257;
        wp::int32 var_258;
        wp::vec_t<3, wp::float32>* var_259;
        wp::float32 var_260;
        wp::vec_t<3, wp::float32> var_261;
        wp::float32 var_262;
        wp::float32 var_263;
        const wp::int32 var_264 = 1;
        const wp::int32 var_265 = 4;
        wp::int32 var_266;
        wp::int32 var_267;
        wp::float32* var_268;
        wp::int32 var_269;
        wp::vec_t<3, wp::float32>* var_270;
        wp::float32 var_271;
        wp::vec_t<3, wp::float32> var_272;
        wp::float32 var_273;
        wp::float32 var_274;
        const wp::int32 var_275 = 2;
        const wp::int32 var_276 = 4;
        wp::int32 var_277;
        wp::int32 var_278;
        wp::float32* var_279;
        wp::int32 var_280;
        wp::vec_t<3, wp::float32>* var_281;
        wp::float32 var_282;
        wp::vec_t<3, wp::float32> var_283;
        wp::float32 var_284;
        wp::float32 var_285;
        const wp::int32 var_286 = 3;
        const wp::int32 var_287 = 4;
        wp::int32 var_288;
        wp::int32 var_289;
        wp::float32* var_290;
        wp::int32 var_291;
        wp::vec_t<3, wp::float32>* var_292;
        wp::float32 var_293;
        wp::vec_t<3, wp::float32> var_294;
        wp::float32 var_295;
        wp::float32 var_296;
        const wp::int32 var_297 = 1;
        const wp::int32 var_298 = 0;
        const wp::int32 var_299 = 4;
        wp::int32 var_300;
        wp::int32 var_301;
        wp::float32* var_302;
        wp::int32 var_303;
        wp::vec_t<3, wp::float32>* var_304;
        wp::float32 var_305;
        wp::vec_t<3, wp::float32> var_306;
        wp::float32 var_307;
        wp::float32 var_308;
        const wp::int32 var_309 = 1;
        const wp::int32 var_310 = 4;
        wp::int32 var_311;
        wp::int32 var_312;
        wp::float32* var_313;
        wp::int32 var_314;
        wp::vec_t<3, wp::float32>* var_315;
        wp::float32 var_316;
        wp::vec_t<3, wp::float32> var_317;
        wp::float32 var_318;
        wp::float32 var_319;
        const wp::int32 var_320 = 2;
        const wp::int32 var_321 = 4;
        wp::int32 var_322;
        wp::int32 var_323;
        wp::float32* var_324;
        wp::int32 var_325;
        wp::vec_t<3, wp::float32>* var_326;
        wp::float32 var_327;
        wp::vec_t<3, wp::float32> var_328;
        wp::float32 var_329;
        wp::float32 var_330;
        const wp::int32 var_331 = 3;
        const wp::int32 var_332 = 4;
        wp::int32 var_333;
        wp::int32 var_334;
        wp::float32* var_335;
        wp::int32 var_336;
        wp::vec_t<3, wp::float32>* var_337;
        wp::float32 var_338;
        wp::vec_t<3, wp::float32> var_339;
        wp::float32 var_340;
        wp::float32 var_341;
        const wp::int32 var_342 = 2;
        const wp::int32 var_343 = 0;
        const wp::int32 var_344 = 4;
        wp::int32 var_345;
        wp::int32 var_346;
        wp::float32* var_347;
        wp::int32 var_348;
        wp::vec_t<3, wp::float32>* var_349;
        wp::float32 var_350;
        wp::vec_t<3, wp::float32> var_351;
        wp::float32 var_352;
        wp::float32 var_353;
        const wp::int32 var_354 = 1;
        const wp::int32 var_355 = 4;
        wp::int32 var_356;
        wp::int32 var_357;
        wp::float32* var_358;
        wp::int32 var_359;
        wp::vec_t<3, wp::float32>* var_360;
        wp::float32 var_361;
        wp::vec_t<3, wp::float32> var_362;
        wp::float32 var_363;
        wp::float32 var_364;
        const wp::int32 var_365 = 2;
        const wp::int32 var_366 = 4;
        wp::int32 var_367;
        wp::int32 var_368;
        wp::float32* var_369;
        wp::int32 var_370;
        wp::vec_t<3, wp::float32>* var_371;
        wp::float32 var_372;
        wp::vec_t<3, wp::float32> var_373;
        wp::float32 var_374;
        wp::float32 var_375;
        const wp::int32 var_376 = 3;
        const wp::int32 var_377 = 4;
        wp::int32 var_378;
        wp::int32 var_379;
        wp::float32* var_380;
        wp::int32 var_381;
        wp::vec_t<3, wp::float32>* var_382;
        wp::float32 var_383;
        wp::vec_t<3, wp::float32> var_384;
        wp::float32 var_385;
        wp::float32 var_386;
        const wp::int32 var_387 = 16;
        wp::float32* var_388;
        wp::float32 var_389;
        wp::float32 var_390;
        wp::float32 var_391;
        const wp::int32 var_392 = 2;
        const wp::int32 var_393 = 0;
        const wp::int32 var_394 = 0;
        const wp::int32 var_395 = 4;
        wp::int32 var_396;
        wp::int32 var_397;
        wp::float32* var_398;
        wp::int32 var_399;
        wp::vec_t<3, wp::float32>* var_400;
        wp::float32 var_401;
        wp::vec_t<3, wp::float32> var_402;
        wp::float32 var_403;
        wp::float32 var_404;
        const wp::int32 var_405 = 1;
        const wp::int32 var_406 = 4;
        wp::int32 var_407;
        wp::int32 var_408;
        wp::float32* var_409;
        wp::int32 var_410;
        wp::vec_t<3, wp::float32>* var_411;
        wp::float32 var_412;
        wp::vec_t<3, wp::float32> var_413;
        wp::float32 var_414;
        wp::float32 var_415;
        const wp::int32 var_416 = 2;
        const wp::int32 var_417 = 4;
        wp::int32 var_418;
        wp::int32 var_419;
        wp::float32* var_420;
        wp::int32 var_421;
        wp::vec_t<3, wp::float32>* var_422;
        wp::float32 var_423;
        wp::vec_t<3, wp::float32> var_424;
        wp::float32 var_425;
        wp::float32 var_426;
        const wp::int32 var_427 = 3;
        const wp::int32 var_428 = 4;
        wp::int32 var_429;
        wp::int32 var_430;
        wp::float32* var_431;
        wp::int32 var_432;
        wp::vec_t<3, wp::float32>* var_433;
        wp::float32 var_434;
        wp::vec_t<3, wp::float32> var_435;
        wp::float32 var_436;
        wp::float32 var_437;
        const wp::int32 var_438 = 1;
        const wp::int32 var_439 = 0;
        const wp::int32 var_440 = 4;
        wp::int32 var_441;
        wp::int32 var_442;
        wp::float32* var_443;
        wp::int32 var_444;
        wp::vec_t<3, wp::float32>* var_445;
        wp::float32 var_446;
        wp::vec_t<3, wp::float32> var_447;
        wp::float32 var_448;
        wp::float32 var_449;
        const wp::int32 var_450 = 1;
        const wp::int32 var_451 = 4;
        wp::int32 var_452;
        wp::int32 var_453;
        wp::float32* var_454;
        wp::int32 var_455;
        wp::vec_t<3, wp::float32>* var_456;
        wp::float32 var_457;
        wp::vec_t<3, wp::float32> var_458;
        wp::float32 var_459;
        wp::float32 var_460;
        const wp::int32 var_461 = 2;
        const wp::int32 var_462 = 4;
        wp::int32 var_463;
        wp::int32 var_464;
        wp::float32* var_465;
        wp::int32 var_466;
        wp::vec_t<3, wp::float32>* var_467;
        wp::float32 var_468;
        wp::vec_t<3, wp::float32> var_469;
        wp::float32 var_470;
        wp::float32 var_471;
        const wp::int32 var_472 = 3;
        const wp::int32 var_473 = 4;
        wp::int32 var_474;
        wp::int32 var_475;
        wp::float32* var_476;
        wp::int32 var_477;
        wp::vec_t<3, wp::float32>* var_478;
        wp::float32 var_479;
        wp::vec_t<3, wp::float32> var_480;
        wp::float32 var_481;
        wp::float32 var_482;
        const wp::int32 var_483 = 2;
        const wp::int32 var_484 = 0;
        const wp::int32 var_485 = 4;
        wp::int32 var_486;
        wp::int32 var_487;
        wp::float32* var_488;
        wp::int32 var_489;
        wp::vec_t<3, wp::float32>* var_490;
        wp::float32 var_491;
        wp::vec_t<3, wp::float32> var_492;
        wp::float32 var_493;
        wp::float32 var_494;
        const wp::int32 var_495 = 1;
        const wp::int32 var_496 = 4;
        wp::int32 var_497;
        wp::int32 var_498;
        wp::float32* var_499;
        wp::int32 var_500;
        wp::vec_t<3, wp::float32>* var_501;
        wp::float32 var_502;
        wp::vec_t<3, wp::float32> var_503;
        wp::float32 var_504;
        wp::float32 var_505;
        const wp::int32 var_506 = 2;
        const wp::int32 var_507 = 4;
        wp::int32 var_508;
        wp::int32 var_509;
        wp::float32* var_510;
        wp::int32 var_511;
        wp::vec_t<3, wp::float32>* var_512;
        wp::float32 var_513;
        wp::vec_t<3, wp::float32> var_514;
        wp::float32 var_515;
        wp::float32 var_516;
        const wp::int32 var_517 = 3;
        const wp::int32 var_518 = 4;
        wp::int32 var_519;
        wp::int32 var_520;
        wp::float32* var_521;
        wp::int32 var_522;
        wp::vec_t<3, wp::float32>* var_523;
        wp::float32 var_524;
        wp::vec_t<3, wp::float32> var_525;
        wp::float32 var_526;
        wp::float32 var_527;
        const wp::int32 var_528 = 16;
        wp::float32* var_529;
        wp::float32 var_530;
        wp::float32 var_531;
        wp::float32 var_532;
        const wp::int32 var_533 = 3;
        const wp::int32 var_534 = 0;
        const wp::int32 var_535 = 0;
        const wp::int32 var_536 = 4;
        wp::int32 var_537;
        wp::int32 var_538;
        wp::float32* var_539;
        wp::int32 var_540;
        wp::vec_t<3, wp::float32>* var_541;
        wp::float32 var_542;
        wp::vec_t<3, wp::float32> var_543;
        wp::float32 var_544;
        wp::float32 var_545;
        const wp::int32 var_546 = 1;
        const wp::int32 var_547 = 4;
        wp::int32 var_548;
        wp::int32 var_549;
        wp::float32* var_550;
        wp::int32 var_551;
        wp::vec_t<3, wp::float32>* var_552;
        wp::float32 var_553;
        wp::vec_t<3, wp::float32> var_554;
        wp::float32 var_555;
        wp::float32 var_556;
        const wp::int32 var_557 = 2;
        const wp::int32 var_558 = 4;
        wp::int32 var_559;
        wp::int32 var_560;
        wp::float32* var_561;
        wp::int32 var_562;
        wp::vec_t<3, wp::float32>* var_563;
        wp::float32 var_564;
        wp::vec_t<3, wp::float32> var_565;
        wp::float32 var_566;
        wp::float32 var_567;
        const wp::int32 var_568 = 3;
        const wp::int32 var_569 = 4;
        wp::int32 var_570;
        wp::int32 var_571;
        wp::float32* var_572;
        wp::int32 var_573;
        wp::vec_t<3, wp::float32>* var_574;
        wp::float32 var_575;
        wp::vec_t<3, wp::float32> var_576;
        wp::float32 var_577;
        wp::float32 var_578;
        const wp::int32 var_579 = 1;
        const wp::int32 var_580 = 0;
        const wp::int32 var_581 = 4;
        wp::int32 var_582;
        wp::int32 var_583;
        wp::float32* var_584;
        wp::int32 var_585;
        wp::vec_t<3, wp::float32>* var_586;
        wp::float32 var_587;
        wp::vec_t<3, wp::float32> var_588;
        wp::float32 var_589;
        wp::float32 var_590;
        const wp::int32 var_591 = 1;
        const wp::int32 var_592 = 4;
        wp::int32 var_593;
        wp::int32 var_594;
        wp::float32* var_595;
        wp::int32 var_596;
        wp::vec_t<3, wp::float32>* var_597;
        wp::float32 var_598;
        wp::vec_t<3, wp::float32> var_599;
        wp::float32 var_600;
        wp::float32 var_601;
        const wp::int32 var_602 = 2;
        const wp::int32 var_603 = 4;
        wp::int32 var_604;
        wp::int32 var_605;
        wp::float32* var_606;
        wp::int32 var_607;
        wp::vec_t<3, wp::float32>* var_608;
        wp::float32 var_609;
        wp::vec_t<3, wp::float32> var_610;
        wp::float32 var_611;
        wp::float32 var_612;
        const wp::int32 var_613 = 3;
        const wp::int32 var_614 = 4;
        wp::int32 var_615;
        wp::int32 var_616;
        wp::float32* var_617;
        wp::int32 var_618;
        wp::vec_t<3, wp::float32>* var_619;
        wp::float32 var_620;
        wp::vec_t<3, wp::float32> var_621;
        wp::float32 var_622;
        wp::float32 var_623;
        const wp::int32 var_624 = 2;
        const wp::int32 var_625 = 0;
        const wp::int32 var_626 = 4;
        wp::int32 var_627;
        wp::int32 var_628;
        wp::float32* var_629;
        wp::int32 var_630;
        wp::vec_t<3, wp::float32>* var_631;
        wp::float32 var_632;
        wp::vec_t<3, wp::float32> var_633;
        wp::float32 var_634;
        wp::float32 var_635;
        const wp::int32 var_636 = 1;
        const wp::int32 var_637 = 4;
        wp::int32 var_638;
        wp::int32 var_639;
        wp::float32* var_640;
        wp::int32 var_641;
        wp::vec_t<3, wp::float32>* var_642;
        wp::float32 var_643;
        wp::vec_t<3, wp::float32> var_644;
        wp::float32 var_645;
        wp::float32 var_646;
        const wp::int32 var_647 = 2;
        const wp::int32 var_648 = 4;
        wp::int32 var_649;
        wp::int32 var_650;
        wp::float32* var_651;
        wp::int32 var_652;
        wp::vec_t<3, wp::float32>* var_653;
        wp::float32 var_654;
        wp::vec_t<3, wp::float32> var_655;
        wp::float32 var_656;
        wp::float32 var_657;
        const wp::int32 var_658 = 3;
        const wp::int32 var_659 = 4;
        wp::int32 var_660;
        wp::int32 var_661;
        wp::float32* var_662;
        wp::int32 var_663;
        wp::vec_t<3, wp::float32>* var_664;
        wp::float32 var_665;
        wp::vec_t<3, wp::float32> var_666;
        wp::float32 var_667;
        wp::float32 var_668;
        const wp::int32 var_669 = 16;
        wp::float32* var_670;
        wp::float32 var_671;
        wp::float32 var_672;
        wp::float32 var_673;
        const wp::int32 var_674 = 0;
        wp::int32 var_675;
        wp::int32* var_676;
        wp::int32 var_677;
        wp::int32 var_678;
        const wp::int32 var_679 = 0;
        wp::int32* var_680;
        wp::int32 var_681;
        wp::int32 var_682;
        wp::float32 var_683;
        wp::float32 var_684;
        const wp::int32 var_685 = 1;
        wp::int32* var_686;
        wp::int32 var_687;
        wp::int32 var_688;
        wp::float32 var_689;
        wp::float32 var_690;
        const wp::int32 var_691 = 2;
        wp::int32* var_692;
        wp::int32 var_693;
        wp::int32 var_694;
        wp::float32 var_695;
        wp::float32 var_696;
        const wp::int32 var_697 = 1;
        wp::int32 var_698;
        wp::int32* var_699;
        wp::int32 var_700;
        wp::int32 var_701;
        const wp::int32 var_702 = 0;
        wp::int32* var_703;
        wp::int32 var_704;
        wp::int32 var_705;
        wp::float32 var_706;
        wp::float32 var_707;
        const wp::int32 var_708 = 1;
        wp::int32* var_709;
        wp::int32 var_710;
        wp::int32 var_711;
        wp::float32 var_712;
        wp::float32 var_713;
        const wp::int32 var_714 = 2;
        wp::int32* var_715;
        wp::int32 var_716;
        wp::int32 var_717;
        wp::float32 var_718;
        wp::float32 var_719;
        const wp::int32 var_720 = 2;
        wp::int32 var_721;
        wp::int32* var_722;
        wp::int32 var_723;
        wp::int32 var_724;
        const wp::int32 var_725 = 0;
        wp::int32* var_726;
        wp::int32 var_727;
        wp::int32 var_728;
        wp::float32 var_729;
        wp::float32 var_730;
        const wp::int32 var_731 = 1;
        wp::int32* var_732;
        wp::int32 var_733;
        wp::int32 var_734;
        wp::float32 var_735;
        wp::float32 var_736;
        const wp::int32 var_737 = 2;
        wp::int32* var_738;
        wp::int32 var_739;
        wp::int32 var_740;
        wp::float32 var_741;
        wp::float32 var_742;
        const wp::int32 var_743 = 3;
        wp::int32 var_744;
        wp::int32* var_745;
        wp::int32 var_746;
        wp::int32 var_747;
        const wp::int32 var_748 = 0;
        wp::int32* var_749;
        wp::int32 var_750;
        wp::int32 var_751;
        wp::float32 var_752;
        wp::float32 var_753;
        const wp::int32 var_754 = 1;
        wp::int32* var_755;
        wp::int32 var_756;
        wp::int32 var_757;
        wp::float32 var_758;
        wp::float32 var_759;
        const wp::int32 var_760 = 2;
        wp::int32* var_761;
        wp::int32 var_762;
        wp::int32 var_763;
        wp::float32 var_764;
        wp::float32 var_765;
        //---------
        // forward
        // def _flex_bending(                                                                     <L 665>
        // worldid, edgeid = wp.tid()                                                             <L 682>
        builtin_tid2d(var_0, var_1);
        // nvert = 4                                                                              <L 683>
        // for i in range(nflex):                                                                 <L 685>
        var_3 = wp::range(var_nflex);
        start_for_0:;
            if (iter_cmp(var_3) == 0) goto end_for_0;
            var_4 = wp::iter_next(var_3);
            // locid = edgeid - flex_edgeadr[i]                                                   <L 686>
            var_5 = wp::address(var_flex_edgeadr, var_4);
            var_7 = wp::load(var_5);
            var_6 = wp::sub(var_1, var_7);
            // if locid >= 0 and locid < flex_edgenum[i]:                                         <L 687>
            var_9 = (var_6 >= var_8);
            var_10 = wp::address(var_flex_edgenum, var_4);
            var_12 = wp::load(var_10);
            var_11 = (var_6 < var_12);
            var_13 = var_9 && var_11;
            if (var_13) {
                // f = i                                                                          <L 688>
                var_14 = wp::copy(var_4);
                // break                                                                          <L 689>
                goto end_for_0;
            }
            goto start_for_0;
        end_for_0:;
        // if flex_dim[f] != 2:                                                                   <L 691>
        var_15 = wp::address(var_flex_dim, var_14);
        var_18 = wp::load(var_15);
        var_17 = (var_18 != var_16);
        if (var_17) {
            // return                                                                             <L 692>
            continue;
        }
        // if flex_edgeflap[edgeid][1] == -1:                                                     <L 694>
        var_19 = wp::address(var_flex_edgeflap, var_1);
        var_22 = wp::load(var_19);
        var_21 = wp::extract(var_22, var_20);
        var_25 = (var_21 == var_24);
        if (var_25) {
            // return                                                                             <L 695>
            continue;
        }
        // v = wp.vec4i(                                                                          <L 697>
        // flex_vertadr[f] + flex_edge[edgeid][0],                                                <L 698>
        var_26 = wp::address(var_flex_vertadr, var_14);
        var_27 = wp::address(var_flex_edge, var_1);
        var_30 = wp::load(var_27);
        var_29 = wp::extract(var_30, var_28);
        var_32 = wp::load(var_26);
        var_31 = wp::add(var_32, var_29);
        // flex_vertadr[f] + flex_edge[edgeid][1],                                                <L 699>
        var_33 = wp::address(var_flex_vertadr, var_14);
        var_34 = wp::address(var_flex_edge, var_1);
        var_37 = wp::load(var_34);
        var_36 = wp::extract(var_37, var_35);
        var_39 = wp::load(var_33);
        var_38 = wp::add(var_39, var_36);
        // flex_vertadr[f] + flex_edgeflap[edgeid][0],                                            <L 700>
        var_40 = wp::address(var_flex_vertadr, var_14);
        var_41 = wp::address(var_flex_edgeflap, var_1);
        var_44 = wp::load(var_41);
        var_43 = wp::extract(var_44, var_42);
        var_46 = wp::load(var_40);
        var_45 = wp::add(var_46, var_43);
        // flex_vertadr[f] + flex_edgeflap[edgeid][1],                                            <L 701>
        var_47 = wp::address(var_flex_vertadr, var_14);
        var_48 = wp::address(var_flex_edgeflap, var_1);
        var_51 = wp::load(var_48);
        var_50 = wp::extract(var_51, var_49);
        var_53 = wp::load(var_47);
        var_52 = wp::add(var_53, var_50);
        var_54 = wp::vec_t<4, wp::int32>(var_31, var_38, var_45, var_52);
        // frc = wp.matrix(0.0, shape=(4, 3))                                                     <L 704>
        var_58 = wp::tuple(var_56, var_57);
        var_59 = wp::mat_t<4, 3, wp::float32>(var_55);
        // if flex_bending[edgeid, 16]:                                                           <L 705>
        var_61 = wp::address(var_flex_bending, var_1, var_60);
        var_62 = wp::load(var_61);
        if (var_62) {
            // v0 = flexvert_xpos_in[worldid, v[0]]                                               <L 706>
            var_64 = wp::extract(var_54, var_63);
            var_65 = wp::address(var_flexvert_xpos_in, var_0, var_64);
            var_67 = wp::load(var_65);
            var_66 = wp::copy(var_67);
            // v1 = flexvert_xpos_in[worldid, v[1]]                                               <L 707>
            var_69 = wp::extract(var_54, var_68);
            var_70 = wp::address(var_flexvert_xpos_in, var_0, var_69);
            var_72 = wp::load(var_70);
            var_71 = wp::copy(var_72);
            // v2 = flexvert_xpos_in[worldid, v[2]]                                               <L 708>
            var_74 = wp::extract(var_54, var_73);
            var_75 = wp::address(var_flexvert_xpos_in, var_0, var_74);
            var_77 = wp::load(var_75);
            var_76 = wp::copy(var_77);
            // v3 = flexvert_xpos_in[worldid, v[3]]                                               <L 709>
            var_79 = wp::extract(var_54, var_78);
            var_80 = wp::address(var_flexvert_xpos_in, var_0, var_79);
            var_82 = wp::load(var_80);
            var_81 = wp::copy(var_82);
            // frc[1] = wp.cross(v2 - v0, v3 - v0)                                                <L 710>
            var_83 = wp::sub(var_76, var_66);
            var_84 = wp::sub(var_81, var_66);
            var_85 = wp::cross(var_83, var_84);
            wp::assign_inplace(var_59, var_86, var_85);
            // frc[2] = wp.cross(v3 - v0, v1 - v0)                                                <L 711>
            var_87 = wp::sub(var_81, var_66);
            var_88 = wp::sub(var_71, var_66);
            var_89 = wp::cross(var_87, var_88);
            wp::assign_inplace(var_59, var_90, var_89);
            // frc[3] = wp.cross(v1 - v0, v2 - v0)                                                <L 712>
            var_91 = wp::sub(var_71, var_66);
            var_92 = wp::sub(var_76, var_66);
            var_93 = wp::cross(var_91, var_92);
            wp::assign_inplace(var_59, var_94, var_93);
            // frc[0] = -(frc[1] + frc[2] + frc[3])                                               <L 713>
            var_96 = wp::extract(var_59, var_95);
            var_98 = wp::extract(var_59, var_97);
            var_99 = wp::add(var_96, var_98);
            var_101 = wp::extract(var_59, var_100);
            var_102 = wp::add(var_99, var_101);
            var_103 = wp::neg(var_102);
            wp::assign_inplace(var_59, var_104, var_103);
        }
        var_105 = wp::load(var_61);
        // force = wp.matrix(0.0, shape=(nvert, 3))                                               <L 715>
        var_108 = wp::tuple(var_2, var_107);
        var_109 = wp::mat_t<4, 3, wp::float32>(var_106);
        // for i in range(nvert):                                                                 <L 716>
        // for x in range(3):                                                                     <L 717>
        // for j in range(nvert):                                                                 <L 718>
        // force[i, x] -= flex_bending[edgeid, 4 * i + j] * flexvert_xpos_in[worldid, v[j]][x]       <L 719>
        var_114 = wp::mul(var_113, var_110);
        var_115 = wp::add(var_114, var_112);
        var_116 = wp::address(var_flex_bending, var_1, var_115);
        var_117 = wp::extract(var_54, var_112);
        var_118 = wp::address(var_flexvert_xpos_in, var_0, var_117);
        var_120 = wp::load(var_118);
        var_119 = wp::extract(var_120, var_111);
        var_122 = wp::load(var_116);
        var_121 = wp::mul(var_122, var_119);
        wp::sub_inplace(var_109, var_110, var_111, var_121);
        var_125 = wp::mul(var_124, var_110);
        var_126 = wp::add(var_125, var_123);
        var_127 = wp::address(var_flex_bending, var_1, var_126);
        var_128 = wp::extract(var_54, var_123);
        var_129 = wp::address(var_flexvert_xpos_in, var_0, var_128);
        var_131 = wp::load(var_129);
        var_130 = wp::extract(var_131, var_111);
        var_133 = wp::load(var_127);
        var_132 = wp::mul(var_133, var_130);
        wp::sub_inplace(var_109, var_110, var_111, var_132);
        var_136 = wp::mul(var_135, var_110);
        var_137 = wp::add(var_136, var_134);
        var_138 = wp::address(var_flex_bending, var_1, var_137);
        var_139 = wp::extract(var_54, var_134);
        var_140 = wp::address(var_flexvert_xpos_in, var_0, var_139);
        var_142 = wp::load(var_140);
        var_141 = wp::extract(var_142, var_111);
        var_144 = wp::load(var_138);
        var_143 = wp::mul(var_144, var_141);
        wp::sub_inplace(var_109, var_110, var_111, var_143);
        var_147 = wp::mul(var_146, var_110);
        var_148 = wp::add(var_147, var_145);
        var_149 = wp::address(var_flex_bending, var_1, var_148);
        var_150 = wp::extract(var_54, var_145);
        var_151 = wp::address(var_flexvert_xpos_in, var_0, var_150);
        var_153 = wp::load(var_151);
        var_152 = wp::extract(var_153, var_111);
        var_155 = wp::load(var_149);
        var_154 = wp::mul(var_155, var_152);
        wp::sub_inplace(var_109, var_110, var_111, var_154);
        // for j in range(nvert):                                                                 <L 718>
        // force[i, x] -= flex_bending[edgeid, 4 * i + j] * flexvert_xpos_in[worldid, v[j]][x]       <L 719>
        var_159 = wp::mul(var_158, var_110);
        var_160 = wp::add(var_159, var_157);
        var_161 = wp::address(var_flex_bending, var_1, var_160);
        var_162 = wp::extract(var_54, var_157);
        var_163 = wp::address(var_flexvert_xpos_in, var_0, var_162);
        var_165 = wp::load(var_163);
        var_164 = wp::extract(var_165, var_156);
        var_167 = wp::load(var_161);
        var_166 = wp::mul(var_167, var_164);
        wp::sub_inplace(var_109, var_110, var_156, var_166);
        var_170 = wp::mul(var_169, var_110);
        var_171 = wp::add(var_170, var_168);
        var_172 = wp::address(var_flex_bending, var_1, var_171);
        var_173 = wp::extract(var_54, var_168);
        var_174 = wp::address(var_flexvert_xpos_in, var_0, var_173);
        var_176 = wp::load(var_174);
        var_175 = wp::extract(var_176, var_156);
        var_178 = wp::load(var_172);
        var_177 = wp::mul(var_178, var_175);
        wp::sub_inplace(var_109, var_110, var_156, var_177);
        var_181 = wp::mul(var_180, var_110);
        var_182 = wp::add(var_181, var_179);
        var_183 = wp::address(var_flex_bending, var_1, var_182);
        var_184 = wp::extract(var_54, var_179);
        var_185 = wp::address(var_flexvert_xpos_in, var_0, var_184);
        var_187 = wp::load(var_185);
        var_186 = wp::extract(var_187, var_156);
        var_189 = wp::load(var_183);
        var_188 = wp::mul(var_189, var_186);
        wp::sub_inplace(var_109, var_110, var_156, var_188);
        var_192 = wp::mul(var_191, var_110);
        var_193 = wp::add(var_192, var_190);
        var_194 = wp::address(var_flex_bending, var_1, var_193);
        var_195 = wp::extract(var_54, var_190);
        var_196 = wp::address(var_flexvert_xpos_in, var_0, var_195);
        var_198 = wp::load(var_196);
        var_197 = wp::extract(var_198, var_156);
        var_200 = wp::load(var_194);
        var_199 = wp::mul(var_200, var_197);
        wp::sub_inplace(var_109, var_110, var_156, var_199);
        // for j in range(nvert):                                                                 <L 718>
        // force[i, x] -= flex_bending[edgeid, 4 * i + j] * flexvert_xpos_in[worldid, v[j]][x]       <L 719>
        var_204 = wp::mul(var_203, var_110);
        var_205 = wp::add(var_204, var_202);
        var_206 = wp::address(var_flex_bending, var_1, var_205);
        var_207 = wp::extract(var_54, var_202);
        var_208 = wp::address(var_flexvert_xpos_in, var_0, var_207);
        var_210 = wp::load(var_208);
        var_209 = wp::extract(var_210, var_201);
        var_212 = wp::load(var_206);
        var_211 = wp::mul(var_212, var_209);
        wp::sub_inplace(var_109, var_110, var_201, var_211);
        var_215 = wp::mul(var_214, var_110);
        var_216 = wp::add(var_215, var_213);
        var_217 = wp::address(var_flex_bending, var_1, var_216);
        var_218 = wp::extract(var_54, var_213);
        var_219 = wp::address(var_flexvert_xpos_in, var_0, var_218);
        var_221 = wp::load(var_219);
        var_220 = wp::extract(var_221, var_201);
        var_223 = wp::load(var_217);
        var_222 = wp::mul(var_223, var_220);
        wp::sub_inplace(var_109, var_110, var_201, var_222);
        var_226 = wp::mul(var_225, var_110);
        var_227 = wp::add(var_226, var_224);
        var_228 = wp::address(var_flex_bending, var_1, var_227);
        var_229 = wp::extract(var_54, var_224);
        var_230 = wp::address(var_flexvert_xpos_in, var_0, var_229);
        var_232 = wp::load(var_230);
        var_231 = wp::extract(var_232, var_201);
        var_234 = wp::load(var_228);
        var_233 = wp::mul(var_234, var_231);
        wp::sub_inplace(var_109, var_110, var_201, var_233);
        var_237 = wp::mul(var_236, var_110);
        var_238 = wp::add(var_237, var_235);
        var_239 = wp::address(var_flex_bending, var_1, var_238);
        var_240 = wp::extract(var_54, var_235);
        var_241 = wp::address(var_flexvert_xpos_in, var_0, var_240);
        var_243 = wp::load(var_241);
        var_242 = wp::extract(var_243, var_201);
        var_245 = wp::load(var_239);
        var_244 = wp::mul(var_245, var_242);
        wp::sub_inplace(var_109, var_110, var_201, var_244);
        // force[i, x] -= flex_bending[edgeid, 16] * frc[i, x]                                    <L 720>
        var_247 = wp::address(var_flex_bending, var_1, var_246);
        var_248 = wp::extract(var_59, var_110, var_201);
        var_250 = wp::load(var_247);
        var_249 = wp::mul(var_250, var_248);
        wp::sub_inplace(var_109, var_110, var_201, var_249);
        // for x in range(3):                                                                     <L 717>
        // for j in range(nvert):                                                                 <L 718>
        // force[i, x] -= flex_bending[edgeid, 4 * i + j] * flexvert_xpos_in[worldid, v[j]][x]       <L 719>
        var_255 = wp::mul(var_254, var_251);
        var_256 = wp::add(var_255, var_253);
        var_257 = wp::address(var_flex_bending, var_1, var_256);
        var_258 = wp::extract(var_54, var_253);
        var_259 = wp::address(var_flexvert_xpos_in, var_0, var_258);
        var_261 = wp::load(var_259);
        var_260 = wp::extract(var_261, var_252);
        var_263 = wp::load(var_257);
        var_262 = wp::mul(var_263, var_260);
        wp::sub_inplace(var_109, var_251, var_252, var_262);
        var_266 = wp::mul(var_265, var_251);
        var_267 = wp::add(var_266, var_264);
        var_268 = wp::address(var_flex_bending, var_1, var_267);
        var_269 = wp::extract(var_54, var_264);
        var_270 = wp::address(var_flexvert_xpos_in, var_0, var_269);
        var_272 = wp::load(var_270);
        var_271 = wp::extract(var_272, var_252);
        var_274 = wp::load(var_268);
        var_273 = wp::mul(var_274, var_271);
        wp::sub_inplace(var_109, var_251, var_252, var_273);
        var_277 = wp::mul(var_276, var_251);
        var_278 = wp::add(var_277, var_275);
        var_279 = wp::address(var_flex_bending, var_1, var_278);
        var_280 = wp::extract(var_54, var_275);
        var_281 = wp::address(var_flexvert_xpos_in, var_0, var_280);
        var_283 = wp::load(var_281);
        var_282 = wp::extract(var_283, var_252);
        var_285 = wp::load(var_279);
        var_284 = wp::mul(var_285, var_282);
        wp::sub_inplace(var_109, var_251, var_252, var_284);
        var_288 = wp::mul(var_287, var_251);
        var_289 = wp::add(var_288, var_286);
        var_290 = wp::address(var_flex_bending, var_1, var_289);
        var_291 = wp::extract(var_54, var_286);
        var_292 = wp::address(var_flexvert_xpos_in, var_0, var_291);
        var_294 = wp::load(var_292);
        var_293 = wp::extract(var_294, var_252);
        var_296 = wp::load(var_290);
        var_295 = wp::mul(var_296, var_293);
        wp::sub_inplace(var_109, var_251, var_252, var_295);
        // for j in range(nvert):                                                                 <L 718>
        // force[i, x] -= flex_bending[edgeid, 4 * i + j] * flexvert_xpos_in[worldid, v[j]][x]       <L 719>
        var_300 = wp::mul(var_299, var_251);
        var_301 = wp::add(var_300, var_298);
        var_302 = wp::address(var_flex_bending, var_1, var_301);
        var_303 = wp::extract(var_54, var_298);
        var_304 = wp::address(var_flexvert_xpos_in, var_0, var_303);
        var_306 = wp::load(var_304);
        var_305 = wp::extract(var_306, var_297);
        var_308 = wp::load(var_302);
        var_307 = wp::mul(var_308, var_305);
        wp::sub_inplace(var_109, var_251, var_297, var_307);
        var_311 = wp::mul(var_310, var_251);
        var_312 = wp::add(var_311, var_309);
        var_313 = wp::address(var_flex_bending, var_1, var_312);
        var_314 = wp::extract(var_54, var_309);
        var_315 = wp::address(var_flexvert_xpos_in, var_0, var_314);
        var_317 = wp::load(var_315);
        var_316 = wp::extract(var_317, var_297);
        var_319 = wp::load(var_313);
        var_318 = wp::mul(var_319, var_316);
        wp::sub_inplace(var_109, var_251, var_297, var_318);
        var_322 = wp::mul(var_321, var_251);
        var_323 = wp::add(var_322, var_320);
        var_324 = wp::address(var_flex_bending, var_1, var_323);
        var_325 = wp::extract(var_54, var_320);
        var_326 = wp::address(var_flexvert_xpos_in, var_0, var_325);
        var_328 = wp::load(var_326);
        var_327 = wp::extract(var_328, var_297);
        var_330 = wp::load(var_324);
        var_329 = wp::mul(var_330, var_327);
        wp::sub_inplace(var_109, var_251, var_297, var_329);
        var_333 = wp::mul(var_332, var_251);
        var_334 = wp::add(var_333, var_331);
        var_335 = wp::address(var_flex_bending, var_1, var_334);
        var_336 = wp::extract(var_54, var_331);
        var_337 = wp::address(var_flexvert_xpos_in, var_0, var_336);
        var_339 = wp::load(var_337);
        var_338 = wp::extract(var_339, var_297);
        var_341 = wp::load(var_335);
        var_340 = wp::mul(var_341, var_338);
        wp::sub_inplace(var_109, var_251, var_297, var_340);
        // for j in range(nvert):                                                                 <L 718>
        // force[i, x] -= flex_bending[edgeid, 4 * i + j] * flexvert_xpos_in[worldid, v[j]][x]       <L 719>
        var_345 = wp::mul(var_344, var_251);
        var_346 = wp::add(var_345, var_343);
        var_347 = wp::address(var_flex_bending, var_1, var_346);
        var_348 = wp::extract(var_54, var_343);
        var_349 = wp::address(var_flexvert_xpos_in, var_0, var_348);
        var_351 = wp::load(var_349);
        var_350 = wp::extract(var_351, var_342);
        var_353 = wp::load(var_347);
        var_352 = wp::mul(var_353, var_350);
        wp::sub_inplace(var_109, var_251, var_342, var_352);
        var_356 = wp::mul(var_355, var_251);
        var_357 = wp::add(var_356, var_354);
        var_358 = wp::address(var_flex_bending, var_1, var_357);
        var_359 = wp::extract(var_54, var_354);
        var_360 = wp::address(var_flexvert_xpos_in, var_0, var_359);
        var_362 = wp::load(var_360);
        var_361 = wp::extract(var_362, var_342);
        var_364 = wp::load(var_358);
        var_363 = wp::mul(var_364, var_361);
        wp::sub_inplace(var_109, var_251, var_342, var_363);
        var_367 = wp::mul(var_366, var_251);
        var_368 = wp::add(var_367, var_365);
        var_369 = wp::address(var_flex_bending, var_1, var_368);
        var_370 = wp::extract(var_54, var_365);
        var_371 = wp::address(var_flexvert_xpos_in, var_0, var_370);
        var_373 = wp::load(var_371);
        var_372 = wp::extract(var_373, var_342);
        var_375 = wp::load(var_369);
        var_374 = wp::mul(var_375, var_372);
        wp::sub_inplace(var_109, var_251, var_342, var_374);
        var_378 = wp::mul(var_377, var_251);
        var_379 = wp::add(var_378, var_376);
        var_380 = wp::address(var_flex_bending, var_1, var_379);
        var_381 = wp::extract(var_54, var_376);
        var_382 = wp::address(var_flexvert_xpos_in, var_0, var_381);
        var_384 = wp::load(var_382);
        var_383 = wp::extract(var_384, var_342);
        var_386 = wp::load(var_380);
        var_385 = wp::mul(var_386, var_383);
        wp::sub_inplace(var_109, var_251, var_342, var_385);
        // force[i, x] -= flex_bending[edgeid, 16] * frc[i, x]                                    <L 720>
        var_388 = wp::address(var_flex_bending, var_1, var_387);
        var_389 = wp::extract(var_59, var_251, var_342);
        var_391 = wp::load(var_388);
        var_390 = wp::mul(var_391, var_389);
        wp::sub_inplace(var_109, var_251, var_342, var_390);
        // for x in range(3):                                                                     <L 717>
        // for j in range(nvert):                                                                 <L 718>
        // force[i, x] -= flex_bending[edgeid, 4 * i + j] * flexvert_xpos_in[worldid, v[j]][x]       <L 719>
        var_396 = wp::mul(var_395, var_392);
        var_397 = wp::add(var_396, var_394);
        var_398 = wp::address(var_flex_bending, var_1, var_397);
        var_399 = wp::extract(var_54, var_394);
        var_400 = wp::address(var_flexvert_xpos_in, var_0, var_399);
        var_402 = wp::load(var_400);
        var_401 = wp::extract(var_402, var_393);
        var_404 = wp::load(var_398);
        var_403 = wp::mul(var_404, var_401);
        wp::sub_inplace(var_109, var_392, var_393, var_403);
        var_407 = wp::mul(var_406, var_392);
        var_408 = wp::add(var_407, var_405);
        var_409 = wp::address(var_flex_bending, var_1, var_408);
        var_410 = wp::extract(var_54, var_405);
        var_411 = wp::address(var_flexvert_xpos_in, var_0, var_410);
        var_413 = wp::load(var_411);
        var_412 = wp::extract(var_413, var_393);
        var_415 = wp::load(var_409);
        var_414 = wp::mul(var_415, var_412);
        wp::sub_inplace(var_109, var_392, var_393, var_414);
        var_418 = wp::mul(var_417, var_392);
        var_419 = wp::add(var_418, var_416);
        var_420 = wp::address(var_flex_bending, var_1, var_419);
        var_421 = wp::extract(var_54, var_416);
        var_422 = wp::address(var_flexvert_xpos_in, var_0, var_421);
        var_424 = wp::load(var_422);
        var_423 = wp::extract(var_424, var_393);
        var_426 = wp::load(var_420);
        var_425 = wp::mul(var_426, var_423);
        wp::sub_inplace(var_109, var_392, var_393, var_425);
        var_429 = wp::mul(var_428, var_392);
        var_430 = wp::add(var_429, var_427);
        var_431 = wp::address(var_flex_bending, var_1, var_430);
        var_432 = wp::extract(var_54, var_427);
        var_433 = wp::address(var_flexvert_xpos_in, var_0, var_432);
        var_435 = wp::load(var_433);
        var_434 = wp::extract(var_435, var_393);
        var_437 = wp::load(var_431);
        var_436 = wp::mul(var_437, var_434);
        wp::sub_inplace(var_109, var_392, var_393, var_436);
        // for j in range(nvert):                                                                 <L 718>
        // force[i, x] -= flex_bending[edgeid, 4 * i + j] * flexvert_xpos_in[worldid, v[j]][x]       <L 719>
        var_441 = wp::mul(var_440, var_392);
        var_442 = wp::add(var_441, var_439);
        var_443 = wp::address(var_flex_bending, var_1, var_442);
        var_444 = wp::extract(var_54, var_439);
        var_445 = wp::address(var_flexvert_xpos_in, var_0, var_444);
        var_447 = wp::load(var_445);
        var_446 = wp::extract(var_447, var_438);
        var_449 = wp::load(var_443);
        var_448 = wp::mul(var_449, var_446);
        wp::sub_inplace(var_109, var_392, var_438, var_448);
        var_452 = wp::mul(var_451, var_392);
        var_453 = wp::add(var_452, var_450);
        var_454 = wp::address(var_flex_bending, var_1, var_453);
        var_455 = wp::extract(var_54, var_450);
        var_456 = wp::address(var_flexvert_xpos_in, var_0, var_455);
        var_458 = wp::load(var_456);
        var_457 = wp::extract(var_458, var_438);
        var_460 = wp::load(var_454);
        var_459 = wp::mul(var_460, var_457);
        wp::sub_inplace(var_109, var_392, var_438, var_459);
        var_463 = wp::mul(var_462, var_392);
        var_464 = wp::add(var_463, var_461);
        var_465 = wp::address(var_flex_bending, var_1, var_464);
        var_466 = wp::extract(var_54, var_461);
        var_467 = wp::address(var_flexvert_xpos_in, var_0, var_466);
        var_469 = wp::load(var_467);
        var_468 = wp::extract(var_469, var_438);
        var_471 = wp::load(var_465);
        var_470 = wp::mul(var_471, var_468);
        wp::sub_inplace(var_109, var_392, var_438, var_470);
        var_474 = wp::mul(var_473, var_392);
        var_475 = wp::add(var_474, var_472);
        var_476 = wp::address(var_flex_bending, var_1, var_475);
        var_477 = wp::extract(var_54, var_472);
        var_478 = wp::address(var_flexvert_xpos_in, var_0, var_477);
        var_480 = wp::load(var_478);
        var_479 = wp::extract(var_480, var_438);
        var_482 = wp::load(var_476);
        var_481 = wp::mul(var_482, var_479);
        wp::sub_inplace(var_109, var_392, var_438, var_481);
        // for j in range(nvert):                                                                 <L 718>
        // force[i, x] -= flex_bending[edgeid, 4 * i + j] * flexvert_xpos_in[worldid, v[j]][x]       <L 719>
        var_486 = wp::mul(var_485, var_392);
        var_487 = wp::add(var_486, var_484);
        var_488 = wp::address(var_flex_bending, var_1, var_487);
        var_489 = wp::extract(var_54, var_484);
        var_490 = wp::address(var_flexvert_xpos_in, var_0, var_489);
        var_492 = wp::load(var_490);
        var_491 = wp::extract(var_492, var_483);
        var_494 = wp::load(var_488);
        var_493 = wp::mul(var_494, var_491);
        wp::sub_inplace(var_109, var_392, var_483, var_493);
        var_497 = wp::mul(var_496, var_392);
        var_498 = wp::add(var_497, var_495);
        var_499 = wp::address(var_flex_bending, var_1, var_498);
        var_500 = wp::extract(var_54, var_495);
        var_501 = wp::address(var_flexvert_xpos_in, var_0, var_500);
        var_503 = wp::load(var_501);
        var_502 = wp::extract(var_503, var_483);
        var_505 = wp::load(var_499);
        var_504 = wp::mul(var_505, var_502);
        wp::sub_inplace(var_109, var_392, var_483, var_504);
        var_508 = wp::mul(var_507, var_392);
        var_509 = wp::add(var_508, var_506);
        var_510 = wp::address(var_flex_bending, var_1, var_509);
        var_511 = wp::extract(var_54, var_506);
        var_512 = wp::address(var_flexvert_xpos_in, var_0, var_511);
        var_514 = wp::load(var_512);
        var_513 = wp::extract(var_514, var_483);
        var_516 = wp::load(var_510);
        var_515 = wp::mul(var_516, var_513);
        wp::sub_inplace(var_109, var_392, var_483, var_515);
        var_519 = wp::mul(var_518, var_392);
        var_520 = wp::add(var_519, var_517);
        var_521 = wp::address(var_flex_bending, var_1, var_520);
        var_522 = wp::extract(var_54, var_517);
        var_523 = wp::address(var_flexvert_xpos_in, var_0, var_522);
        var_525 = wp::load(var_523);
        var_524 = wp::extract(var_525, var_483);
        var_527 = wp::load(var_521);
        var_526 = wp::mul(var_527, var_524);
        wp::sub_inplace(var_109, var_392, var_483, var_526);
        // force[i, x] -= flex_bending[edgeid, 16] * frc[i, x]                                    <L 720>
        var_529 = wp::address(var_flex_bending, var_1, var_528);
        var_530 = wp::extract(var_59, var_392, var_483);
        var_532 = wp::load(var_529);
        var_531 = wp::mul(var_532, var_530);
        wp::sub_inplace(var_109, var_392, var_483, var_531);
        // for x in range(3):                                                                     <L 717>
        // for j in range(nvert):                                                                 <L 718>
        // force[i, x] -= flex_bending[edgeid, 4 * i + j] * flexvert_xpos_in[worldid, v[j]][x]       <L 719>
        var_537 = wp::mul(var_536, var_533);
        var_538 = wp::add(var_537, var_535);
        var_539 = wp::address(var_flex_bending, var_1, var_538);
        var_540 = wp::extract(var_54, var_535);
        var_541 = wp::address(var_flexvert_xpos_in, var_0, var_540);
        var_543 = wp::load(var_541);
        var_542 = wp::extract(var_543, var_534);
        var_545 = wp::load(var_539);
        var_544 = wp::mul(var_545, var_542);
        wp::sub_inplace(var_109, var_533, var_534, var_544);
        var_548 = wp::mul(var_547, var_533);
        var_549 = wp::add(var_548, var_546);
        var_550 = wp::address(var_flex_bending, var_1, var_549);
        var_551 = wp::extract(var_54, var_546);
        var_552 = wp::address(var_flexvert_xpos_in, var_0, var_551);
        var_554 = wp::load(var_552);
        var_553 = wp::extract(var_554, var_534);
        var_556 = wp::load(var_550);
        var_555 = wp::mul(var_556, var_553);
        wp::sub_inplace(var_109, var_533, var_534, var_555);
        var_559 = wp::mul(var_558, var_533);
        var_560 = wp::add(var_559, var_557);
        var_561 = wp::address(var_flex_bending, var_1, var_560);
        var_562 = wp::extract(var_54, var_557);
        var_563 = wp::address(var_flexvert_xpos_in, var_0, var_562);
        var_565 = wp::load(var_563);
        var_564 = wp::extract(var_565, var_534);
        var_567 = wp::load(var_561);
        var_566 = wp::mul(var_567, var_564);
        wp::sub_inplace(var_109, var_533, var_534, var_566);
        var_570 = wp::mul(var_569, var_533);
        var_571 = wp::add(var_570, var_568);
        var_572 = wp::address(var_flex_bending, var_1, var_571);
        var_573 = wp::extract(var_54, var_568);
        var_574 = wp::address(var_flexvert_xpos_in, var_0, var_573);
        var_576 = wp::load(var_574);
        var_575 = wp::extract(var_576, var_534);
        var_578 = wp::load(var_572);
        var_577 = wp::mul(var_578, var_575);
        wp::sub_inplace(var_109, var_533, var_534, var_577);
        // for j in range(nvert):                                                                 <L 718>
        // force[i, x] -= flex_bending[edgeid, 4 * i + j] * flexvert_xpos_in[worldid, v[j]][x]       <L 719>
        var_582 = wp::mul(var_581, var_533);
        var_583 = wp::add(var_582, var_580);
        var_584 = wp::address(var_flex_bending, var_1, var_583);
        var_585 = wp::extract(var_54, var_580);
        var_586 = wp::address(var_flexvert_xpos_in, var_0, var_585);
        var_588 = wp::load(var_586);
        var_587 = wp::extract(var_588, var_579);
        var_590 = wp::load(var_584);
        var_589 = wp::mul(var_590, var_587);
        wp::sub_inplace(var_109, var_533, var_579, var_589);
        var_593 = wp::mul(var_592, var_533);
        var_594 = wp::add(var_593, var_591);
        var_595 = wp::address(var_flex_bending, var_1, var_594);
        var_596 = wp::extract(var_54, var_591);
        var_597 = wp::address(var_flexvert_xpos_in, var_0, var_596);
        var_599 = wp::load(var_597);
        var_598 = wp::extract(var_599, var_579);
        var_601 = wp::load(var_595);
        var_600 = wp::mul(var_601, var_598);
        wp::sub_inplace(var_109, var_533, var_579, var_600);
        var_604 = wp::mul(var_603, var_533);
        var_605 = wp::add(var_604, var_602);
        var_606 = wp::address(var_flex_bending, var_1, var_605);
        var_607 = wp::extract(var_54, var_602);
        var_608 = wp::address(var_flexvert_xpos_in, var_0, var_607);
        var_610 = wp::load(var_608);
        var_609 = wp::extract(var_610, var_579);
        var_612 = wp::load(var_606);
        var_611 = wp::mul(var_612, var_609);
        wp::sub_inplace(var_109, var_533, var_579, var_611);
        var_615 = wp::mul(var_614, var_533);
        var_616 = wp::add(var_615, var_613);
        var_617 = wp::address(var_flex_bending, var_1, var_616);
        var_618 = wp::extract(var_54, var_613);
        var_619 = wp::address(var_flexvert_xpos_in, var_0, var_618);
        var_621 = wp::load(var_619);
        var_620 = wp::extract(var_621, var_579);
        var_623 = wp::load(var_617);
        var_622 = wp::mul(var_623, var_620);
        wp::sub_inplace(var_109, var_533, var_579, var_622);
        // for j in range(nvert):                                                                 <L 718>
        // force[i, x] -= flex_bending[edgeid, 4 * i + j] * flexvert_xpos_in[worldid, v[j]][x]       <L 719>
        var_627 = wp::mul(var_626, var_533);
        var_628 = wp::add(var_627, var_625);
        var_629 = wp::address(var_flex_bending, var_1, var_628);
        var_630 = wp::extract(var_54, var_625);
        var_631 = wp::address(var_flexvert_xpos_in, var_0, var_630);
        var_633 = wp::load(var_631);
        var_632 = wp::extract(var_633, var_624);
        var_635 = wp::load(var_629);
        var_634 = wp::mul(var_635, var_632);
        wp::sub_inplace(var_109, var_533, var_624, var_634);
        var_638 = wp::mul(var_637, var_533);
        var_639 = wp::add(var_638, var_636);
        var_640 = wp::address(var_flex_bending, var_1, var_639);
        var_641 = wp::extract(var_54, var_636);
        var_642 = wp::address(var_flexvert_xpos_in, var_0, var_641);
        var_644 = wp::load(var_642);
        var_643 = wp::extract(var_644, var_624);
        var_646 = wp::load(var_640);
        var_645 = wp::mul(var_646, var_643);
        wp::sub_inplace(var_109, var_533, var_624, var_645);
        var_649 = wp::mul(var_648, var_533);
        var_650 = wp::add(var_649, var_647);
        var_651 = wp::address(var_flex_bending, var_1, var_650);
        var_652 = wp::extract(var_54, var_647);
        var_653 = wp::address(var_flexvert_xpos_in, var_0, var_652);
        var_655 = wp::load(var_653);
        var_654 = wp::extract(var_655, var_624);
        var_657 = wp::load(var_651);
        var_656 = wp::mul(var_657, var_654);
        wp::sub_inplace(var_109, var_533, var_624, var_656);
        var_660 = wp::mul(var_659, var_533);
        var_661 = wp::add(var_660, var_658);
        var_662 = wp::address(var_flex_bending, var_1, var_661);
        var_663 = wp::extract(var_54, var_658);
        var_664 = wp::address(var_flexvert_xpos_in, var_0, var_663);
        var_666 = wp::load(var_664);
        var_665 = wp::extract(var_666, var_624);
        var_668 = wp::load(var_662);
        var_667 = wp::mul(var_668, var_665);
        wp::sub_inplace(var_109, var_533, var_624, var_667);
        // force[i, x] -= flex_bending[edgeid, 16] * frc[i, x]                                    <L 720>
        var_670 = wp::address(var_flex_bending, var_1, var_669);
        var_671 = wp::extract(var_59, var_533, var_624);
        var_673 = wp::load(var_670);
        var_672 = wp::mul(var_673, var_671);
        wp::sub_inplace(var_109, var_533, var_624, var_672);
        // for i in range(nvert):                                                                 <L 722>
        // bodyid = flex_vertbodyid[v[i]]                                                         <L 723>
        var_675 = wp::extract(var_54, var_674);
        var_676 = wp::address(var_flex_vertbodyid, var_675);
        var_678 = wp::load(var_676);
        var_677 = wp::copy(var_678);
        // for x in range(3):                                                                     <L 724>
        // wp.atomic_add(qfrc_spring_out, worldid, body_dofadr[bodyid] + x, force[i, x])          <L 725>
        var_680 = wp::address(var_body_dofadr, var_677);
        var_682 = wp::load(var_680);
        var_681 = wp::add(var_682, var_679);
        var_683 = wp::extract(var_109, var_674, var_679);
        var_684 = wp::atomic_add(var_qfrc_spring_out, var_0, var_681, var_683);
        var_686 = wp::address(var_body_dofadr, var_677);
        var_688 = wp::load(var_686);
        var_687 = wp::add(var_688, var_685);
        var_689 = wp::extract(var_109, var_674, var_685);
        var_690 = wp::atomic_add(var_qfrc_spring_out, var_0, var_687, var_689);
        var_692 = wp::address(var_body_dofadr, var_677);
        var_694 = wp::load(var_692);
        var_693 = wp::add(var_694, var_691);
        var_695 = wp::extract(var_109, var_674, var_691);
        var_696 = wp::atomic_add(var_qfrc_spring_out, var_0, var_693, var_695);
        // bodyid = flex_vertbodyid[v[i]]                                                         <L 723>
        var_698 = wp::extract(var_54, var_697);
        var_699 = wp::address(var_flex_vertbodyid, var_698);
        var_701 = wp::load(var_699);
        var_700 = wp::copy(var_701);
        // for x in range(3):                                                                     <L 724>
        // wp.atomic_add(qfrc_spring_out, worldid, body_dofadr[bodyid] + x, force[i, x])          <L 725>
        var_703 = wp::address(var_body_dofadr, var_700);
        var_705 = wp::load(var_703);
        var_704 = wp::add(var_705, var_702);
        var_706 = wp::extract(var_109, var_697, var_702);
        var_707 = wp::atomic_add(var_qfrc_spring_out, var_0, var_704, var_706);
        var_709 = wp::address(var_body_dofadr, var_700);
        var_711 = wp::load(var_709);
        var_710 = wp::add(var_711, var_708);
        var_712 = wp::extract(var_109, var_697, var_708);
        var_713 = wp::atomic_add(var_qfrc_spring_out, var_0, var_710, var_712);
        var_715 = wp::address(var_body_dofadr, var_700);
        var_717 = wp::load(var_715);
        var_716 = wp::add(var_717, var_714);
        var_718 = wp::extract(var_109, var_697, var_714);
        var_719 = wp::atomic_add(var_qfrc_spring_out, var_0, var_716, var_718);
        // bodyid = flex_vertbodyid[v[i]]                                                         <L 723>
        var_721 = wp::extract(var_54, var_720);
        var_722 = wp::address(var_flex_vertbodyid, var_721);
        var_724 = wp::load(var_722);
        var_723 = wp::copy(var_724);
        // for x in range(3):                                                                     <L 724>
        // wp.atomic_add(qfrc_spring_out, worldid, body_dofadr[bodyid] + x, force[i, x])          <L 725>
        var_726 = wp::address(var_body_dofadr, var_723);
        var_728 = wp::load(var_726);
        var_727 = wp::add(var_728, var_725);
        var_729 = wp::extract(var_109, var_720, var_725);
        var_730 = wp::atomic_add(var_qfrc_spring_out, var_0, var_727, var_729);
        var_732 = wp::address(var_body_dofadr, var_723);
        var_734 = wp::load(var_732);
        var_733 = wp::add(var_734, var_731);
        var_735 = wp::extract(var_109, var_720, var_731);
        var_736 = wp::atomic_add(var_qfrc_spring_out, var_0, var_733, var_735);
        var_738 = wp::address(var_body_dofadr, var_723);
        var_740 = wp::load(var_738);
        var_739 = wp::add(var_740, var_737);
        var_741 = wp::extract(var_109, var_720, var_737);
        var_742 = wp::atomic_add(var_qfrc_spring_out, var_0, var_739, var_741);
        // bodyid = flex_vertbodyid[v[i]]                                                         <L 723>
        var_744 = wp::extract(var_54, var_743);
        var_745 = wp::address(var_flex_vertbodyid, var_744);
        var_747 = wp::load(var_745);
        var_746 = wp::copy(var_747);
        // for x in range(3):                                                                     <L 724>
        // wp.atomic_add(qfrc_spring_out, worldid, body_dofadr[bodyid] + x, force[i, x])          <L 725>
        var_749 = wp::address(var_body_dofadr, var_746);
        var_751 = wp::load(var_749);
        var_750 = wp::add(var_751, var_748);
        var_752 = wp::extract(var_109, var_743, var_748);
        var_753 = wp::atomic_add(var_qfrc_spring_out, var_0, var_750, var_752);
        var_755 = wp::address(var_body_dofadr, var_746);
        var_757 = wp::load(var_755);
        var_756 = wp::add(var_757, var_754);
        var_758 = wp::extract(var_109, var_743, var_754);
        var_759 = wp::atomic_add(var_qfrc_spring_out, var_0, var_756, var_758);
        var_761 = wp::address(var_body_dofadr, var_746);
        var_763 = wp::load(var_761);
        var_762 = wp::add(var_763, var_760);
        var_764 = wp::extract(var_109, var_743, var_760);
        var_765 = wp::atomic_add(var_qfrc_spring_out, var_0, var_762, var_764);
    }
}

