
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:44
static CUDA_CALLABLE wp::vec_t<3, wp::float32> rot_vec_quat_0(
    wp::vec_t<3, wp::float32> var_vec,
    wp::quat_t<wp::float32> var_quat)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 2;
    wp::float32 var_5;
    const wp::int32 var_6 = 3;
    wp::float32 var_7;
    wp::vec_t<3, wp::float32> var_8;
    const wp::float32 var_9 = 2.0;
    wp::float32 var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::vec_t<3, wp::float32> var_17;
    const wp::float32 var_18 = 2.0;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    //---------
    // forward
    // def rot_vec_quat(vec: wp.vec3, quat: wp.quat) -> wp.vec3:                              <L 45>
    // s, u = quat[0], wp.vec3(quat[1], quat[2], quat[3])                                     <L 46>
    var_1 = wp::extract(var_quat, var_0);
    var_3 = wp::extract(var_quat, var_2);
    var_5 = wp::extract(var_quat, var_4);
    var_7 = wp::extract(var_quat, var_6);
    var_8 = wp::vec_t<3, wp::float32>(var_3, var_5, var_7);
    // r = 2.0 * (wp.dot(u, vec) * u) + (s * s - wp.dot(u, u)) * vec                          <L 47>
    var_10 = wp::dot(var_8, var_vec);
    var_11 = wp::mul(var_10, var_8);
    var_12 = wp::mul(var_9, var_11);
    var_13 = wp::mul(var_1, var_1);
    var_14 = wp::dot(var_8, var_8);
    var_15 = wp::sub(var_13, var_14);
    var_16 = wp::mul(var_15, var_vec);
    var_17 = wp::add(var_12, var_16);
    // r = r + 2.0 * s * wp.cross(u, vec)                                                     <L 48>
    var_19 = wp::mul(var_18, var_1);
    var_20 = wp::cross(var_8, var_vec);
    var_21 = wp::mul(var_19, var_20);
    var_22 = wp::add(var_17, var_21);
    // return r                                                                               <L 49>
    return var_22;
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:59
static CUDA_CALLABLE wp::mat_t<3, 3, wp::float32> quat_to_mat_0(
    wp::quat_t<wp::float32> var_quat)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::int32 var_5 = 0;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    wp::float32 var_9;
    const wp::int32 var_10 = 0;
    wp::float32 var_11;
    const wp::int32 var_12 = 2;
    wp::float32 var_13;
    wp::float32 var_14;
    const wp::int32 var_15 = 0;
    wp::float32 var_16;
    const wp::int32 var_17 = 3;
    wp::float32 var_18;
    wp::float32 var_19;
    const wp::int32 var_20 = 1;
    wp::float32 var_21;
    const wp::int32 var_22 = 1;
    wp::float32 var_23;
    wp::float32 var_24;
    const wp::int32 var_25 = 1;
    wp::float32 var_26;
    const wp::int32 var_27 = 2;
    wp::float32 var_28;
    wp::float32 var_29;
    const wp::int32 var_30 = 1;
    wp::float32 var_31;
    const wp::int32 var_32 = 3;
    wp::float32 var_33;
    wp::float32 var_34;
    const wp::int32 var_35 = 2;
    wp::float32 var_36;
    const wp::int32 var_37 = 2;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::int32 var_40 = 2;
    wp::float32 var_41;
    const wp::int32 var_42 = 3;
    wp::float32 var_43;
    wp::float32 var_44;
    const wp::int32 var_45 = 3;
    wp::float32 var_46;
    const wp::int32 var_47 = 3;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    const wp::float32 var_53 = 2.0;
    wp::float32 var_54;
    wp::float32 var_55;
    const wp::float32 var_56 = 2.0;
    wp::float32 var_57;
    wp::float32 var_58;
    const wp::float32 var_59 = 2.0;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    wp::float32 var_64;
    const wp::float32 var_65 = 2.0;
    wp::float32 var_66;
    wp::float32 var_67;
    const wp::float32 var_68 = 2.0;
    wp::float32 var_69;
    wp::float32 var_70;
    const wp::float32 var_71 = 2.0;
    wp::float32 var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    wp::float32 var_75;
    wp::float32 var_76;
    wp::mat_t<3, 3, wp::float32> var_77;
    //---------
    // forward
    // def quat_to_mat(quat: wp.quat) -> wp.mat33:                                            <L 60>
    // q00 = quat[0] * quat[0]                                                                <L 62>
    var_1 = wp::extract(var_quat, var_0);
    var_3 = wp::extract(var_quat, var_2);
    var_4 = wp::mul(var_1, var_3);
    // q01 = quat[0] * quat[1]                                                                <L 63>
    var_6 = wp::extract(var_quat, var_5);
    var_8 = wp::extract(var_quat, var_7);
    var_9 = wp::mul(var_6, var_8);
    // q02 = quat[0] * quat[2]                                                                <L 64>
    var_11 = wp::extract(var_quat, var_10);
    var_13 = wp::extract(var_quat, var_12);
    var_14 = wp::mul(var_11, var_13);
    // q03 = quat[0] * quat[3]                                                                <L 65>
    var_16 = wp::extract(var_quat, var_15);
    var_18 = wp::extract(var_quat, var_17);
    var_19 = wp::mul(var_16, var_18);
    // q11 = quat[1] * quat[1]                                                                <L 66>
    var_21 = wp::extract(var_quat, var_20);
    var_23 = wp::extract(var_quat, var_22);
    var_24 = wp::mul(var_21, var_23);
    // q12 = quat[1] * quat[2]                                                                <L 67>
    var_26 = wp::extract(var_quat, var_25);
    var_28 = wp::extract(var_quat, var_27);
    var_29 = wp::mul(var_26, var_28);
    // q13 = quat[1] * quat[3]                                                                <L 68>
    var_31 = wp::extract(var_quat, var_30);
    var_33 = wp::extract(var_quat, var_32);
    var_34 = wp::mul(var_31, var_33);
    // q22 = quat[2] * quat[2]                                                                <L 69>
    var_36 = wp::extract(var_quat, var_35);
    var_38 = wp::extract(var_quat, var_37);
    var_39 = wp::mul(var_36, var_38);
    // q23 = quat[2] * quat[3]                                                                <L 70>
    var_41 = wp::extract(var_quat, var_40);
    var_43 = wp::extract(var_quat, var_42);
    var_44 = wp::mul(var_41, var_43);
    // q33 = quat[3] * quat[3]                                                                <L 71>
    var_46 = wp::extract(var_quat, var_45);
    var_48 = wp::extract(var_quat, var_47);
    var_49 = wp::mul(var_46, var_48);
    // return wp.mat33(                                                                       <L 73>
    // q00 + q11 - q22 - q33,                                                                 <L 74>
    var_50 = wp::add(var_4, var_24);
    var_51 = wp::sub(var_50, var_39);
    var_52 = wp::sub(var_51, var_49);
    // 2.0 * (q12 - q03),                                                                     <L 75>
    var_54 = wp::sub(var_29, var_19);
    var_55 = wp::mul(var_53, var_54);
    // 2.0 * (q13 + q02),                                                                     <L 76>
    var_57 = wp::add(var_34, var_14);
    var_58 = wp::mul(var_56, var_57);
    // 2.0 * (q12 + q03),                                                                     <L 77>
    var_60 = wp::add(var_29, var_19);
    var_61 = wp::mul(var_59, var_60);
    // q00 - q11 + q22 - q33,                                                                 <L 78>
    var_62 = wp::sub(var_4, var_24);
    var_63 = wp::add(var_62, var_39);
    var_64 = wp::sub(var_63, var_49);
    // 2.0 * (q23 - q01),                                                                     <L 79>
    var_66 = wp::sub(var_44, var_9);
    var_67 = wp::mul(var_65, var_66);
    // 2.0 * (q13 - q02),                                                                     <L 80>
    var_69 = wp::sub(var_34, var_14);
    var_70 = wp::mul(var_68, var_69);
    // 2.0 * (q23 + q01),                                                                     <L 81>
    var_72 = wp::add(var_44, var_9);
    var_73 = wp::mul(var_71, var_72);
    // q00 - q11 - q22 + q33,                                                                 <L 82>
    var_74 = wp::sub(var_4, var_24);
    var_75 = wp::sub(var_74, var_39);
    var_76 = wp::add(var_75, var_49);
    var_77 = wp::mat_t<3, 3, wp::float32>(var_52, var_55, var_58, var_61, var_64, var_67, var_70, var_73, var_76);
    return var_77;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:52
static CUDA_CALLABLE wp::quat_t<wp::float32> axis_angle_to_quat_0(
    wp::vec_t<3, wp::float32> var_axis,
    wp::float32 var_angle)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.5;
    wp::float32 var_1;
    wp::float32 var_2;
    const wp::float32 var_3 = 0.5;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::vec_t<3, wp::float32> var_6;
    const wp::int32 var_7 = 0;
    wp::float32 var_8;
    const wp::int32 var_9 = 1;
    wp::float32 var_10;
    const wp::int32 var_11 = 2;
    wp::float32 var_12;
    wp::quat_t<wp::float32> var_13;
    //---------
    // forward
    // def axis_angle_to_quat(axis: wp.vec3, angle: float) -> wp.quat:                        <L 53>
    // s, c = wp.sin(angle * 0.5), wp.cos(angle * 0.5)                                        <L 54>
    var_1 = wp::mul(var_angle, var_0);
    var_2 = wp::sin(var_1);
    var_4 = wp::mul(var_angle, var_3);
    var_5 = wp::cos(var_4);
    // axis = axis * s                                                                        <L 55>
    var_6 = wp::mul(var_axis, var_2);
    // return wp.quat(c, axis[0], axis[1], axis[2])                                           <L 56>
    var_8 = wp::extract(var_6, var_7);
    var_10 = wp::extract(var_6, var_9);
    var_12 = wp::extract(var_6, var_11);
    var_13 = wp::quat_t<wp::float32>(var_5, var_8, var_10, var_12);
    return var_13;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:120
static CUDA_CALLABLE wp::vec_t<6, wp::float32> inert_vec_0(
    wp::vec_t<10, wp::float32> var_i,
    wp::vec_t<6, wp::float32> var_v)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::int32 var_5 = 3;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::int32 var_11 = 4;
    wp::float32 var_12;
    const wp::int32 var_13 = 2;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::int32 var_17 = 8;
    wp::float32 var_18;
    const wp::int32 var_19 = 4;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::int32 var_23 = 7;
    wp::float32 var_24;
    const wp::int32 var_25 = 5;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    const wp::int32 var_29 = 3;
    wp::float32 var_30;
    const wp::int32 var_31 = 0;
    wp::float32 var_32;
    wp::float32 var_33;
    const wp::int32 var_34 = 1;
    wp::float32 var_35;
    const wp::int32 var_36 = 1;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::int32 var_40 = 5;
    wp::float32 var_41;
    const wp::int32 var_42 = 2;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    const wp::int32 var_46 = 8;
    wp::float32 var_47;
    const wp::int32 var_48 = 3;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    const wp::int32 var_52 = 6;
    wp::float32 var_53;
    const wp::int32 var_54 = 5;
    wp::float32 var_55;
    wp::float32 var_56;
    wp::float32 var_57;
    const wp::int32 var_58 = 4;
    wp::float32 var_59;
    const wp::int32 var_60 = 0;
    wp::float32 var_61;
    wp::float32 var_62;
    const wp::int32 var_63 = 5;
    wp::float32 var_64;
    const wp::int32 var_65 = 1;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    const wp::int32 var_69 = 2;
    wp::float32 var_70;
    const wp::int32 var_71 = 2;
    wp::float32 var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    const wp::int32 var_75 = 7;
    wp::float32 var_76;
    const wp::int32 var_77 = 3;
    wp::float32 var_78;
    wp::float32 var_79;
    wp::float32 var_80;
    const wp::int32 var_81 = 6;
    wp::float32 var_82;
    const wp::int32 var_83 = 4;
    wp::float32 var_84;
    wp::float32 var_85;
    wp::float32 var_86;
    const wp::int32 var_87 = 8;
    wp::float32 var_88;
    const wp::int32 var_89 = 1;
    wp::float32 var_90;
    wp::float32 var_91;
    const wp::int32 var_92 = 7;
    wp::float32 var_93;
    const wp::int32 var_94 = 2;
    wp::float32 var_95;
    wp::float32 var_96;
    wp::float32 var_97;
    const wp::int32 var_98 = 9;
    wp::float32 var_99;
    const wp::int32 var_100 = 3;
    wp::float32 var_101;
    wp::float32 var_102;
    wp::float32 var_103;
    const wp::int32 var_104 = 6;
    wp::float32 var_105;
    const wp::int32 var_106 = 2;
    wp::float32 var_107;
    wp::float32 var_108;
    const wp::int32 var_109 = 8;
    wp::float32 var_110;
    const wp::int32 var_111 = 0;
    wp::float32 var_112;
    wp::float32 var_113;
    wp::float32 var_114;
    const wp::int32 var_115 = 9;
    wp::float32 var_116;
    const wp::int32 var_117 = 4;
    wp::float32 var_118;
    wp::float32 var_119;
    wp::float32 var_120;
    const wp::int32 var_121 = 7;
    wp::float32 var_122;
    const wp::int32 var_123 = 0;
    wp::float32 var_124;
    wp::float32 var_125;
    const wp::int32 var_126 = 6;
    wp::float32 var_127;
    const wp::int32 var_128 = 1;
    wp::float32 var_129;
    wp::float32 var_130;
    wp::float32 var_131;
    const wp::int32 var_132 = 9;
    wp::float32 var_133;
    const wp::int32 var_134 = 5;
    wp::float32 var_135;
    wp::float32 var_136;
    wp::float32 var_137;
    wp::vec_t<6, wp::float32> var_138;
    //---------
    // forward
    // def inert_vec(i: types.vec10, v: wp.spatial_vector) -> wp.spatial_vector:              <L 121>
    // return wp.spatial_vector(                                                              <L 123>
    // i[0] * v[0] + i[3] * v[1] + i[4] * v[2] - i[8] * v[4] + i[7] * v[5],                   <L 124>
    var_1 = wp::extract(var_i, var_0);
    var_3 = wp::extract(var_v, var_2);
    var_4 = wp::mul(var_1, var_3);
    var_6 = wp::extract(var_i, var_5);
    var_8 = wp::extract(var_v, var_7);
    var_9 = wp::mul(var_6, var_8);
    var_10 = wp::add(var_4, var_9);
    var_12 = wp::extract(var_i, var_11);
    var_14 = wp::extract(var_v, var_13);
    var_15 = wp::mul(var_12, var_14);
    var_16 = wp::add(var_10, var_15);
    var_18 = wp::extract(var_i, var_17);
    var_20 = wp::extract(var_v, var_19);
    var_21 = wp::mul(var_18, var_20);
    var_22 = wp::sub(var_16, var_21);
    var_24 = wp::extract(var_i, var_23);
    var_26 = wp::extract(var_v, var_25);
    var_27 = wp::mul(var_24, var_26);
    var_28 = wp::add(var_22, var_27);
    // i[3] * v[0] + i[1] * v[1] + i[5] * v[2] + i[8] * v[3] - i[6] * v[5],                   <L 125>
    var_30 = wp::extract(var_i, var_29);
    var_32 = wp::extract(var_v, var_31);
    var_33 = wp::mul(var_30, var_32);
    var_35 = wp::extract(var_i, var_34);
    var_37 = wp::extract(var_v, var_36);
    var_38 = wp::mul(var_35, var_37);
    var_39 = wp::add(var_33, var_38);
    var_41 = wp::extract(var_i, var_40);
    var_43 = wp::extract(var_v, var_42);
    var_44 = wp::mul(var_41, var_43);
    var_45 = wp::add(var_39, var_44);
    var_47 = wp::extract(var_i, var_46);
    var_49 = wp::extract(var_v, var_48);
    var_50 = wp::mul(var_47, var_49);
    var_51 = wp::add(var_45, var_50);
    var_53 = wp::extract(var_i, var_52);
    var_55 = wp::extract(var_v, var_54);
    var_56 = wp::mul(var_53, var_55);
    var_57 = wp::sub(var_51, var_56);
    // i[4] * v[0] + i[5] * v[1] + i[2] * v[2] - i[7] * v[3] + i[6] * v[4],                   <L 126>
    var_59 = wp::extract(var_i, var_58);
    var_61 = wp::extract(var_v, var_60);
    var_62 = wp::mul(var_59, var_61);
    var_64 = wp::extract(var_i, var_63);
    var_66 = wp::extract(var_v, var_65);
    var_67 = wp::mul(var_64, var_66);
    var_68 = wp::add(var_62, var_67);
    var_70 = wp::extract(var_i, var_69);
    var_72 = wp::extract(var_v, var_71);
    var_73 = wp::mul(var_70, var_72);
    var_74 = wp::add(var_68, var_73);
    var_76 = wp::extract(var_i, var_75);
    var_78 = wp::extract(var_v, var_77);
    var_79 = wp::mul(var_76, var_78);
    var_80 = wp::sub(var_74, var_79);
    var_82 = wp::extract(var_i, var_81);
    var_84 = wp::extract(var_v, var_83);
    var_85 = wp::mul(var_82, var_84);
    var_86 = wp::add(var_80, var_85);
    // i[8] * v[1] - i[7] * v[2] + i[9] * v[3],                                               <L 127>
    var_88 = wp::extract(var_i, var_87);
    var_90 = wp::extract(var_v, var_89);
    var_91 = wp::mul(var_88, var_90);
    var_93 = wp::extract(var_i, var_92);
    var_95 = wp::extract(var_v, var_94);
    var_96 = wp::mul(var_93, var_95);
    var_97 = wp::sub(var_91, var_96);
    var_99 = wp::extract(var_i, var_98);
    var_101 = wp::extract(var_v, var_100);
    var_102 = wp::mul(var_99, var_101);
    var_103 = wp::add(var_97, var_102);
    // i[6] * v[2] - i[8] * v[0] + i[9] * v[4],                                               <L 128>
    var_105 = wp::extract(var_i, var_104);
    var_107 = wp::extract(var_v, var_106);
    var_108 = wp::mul(var_105, var_107);
    var_110 = wp::extract(var_i, var_109);
    var_112 = wp::extract(var_v, var_111);
    var_113 = wp::mul(var_110, var_112);
    var_114 = wp::sub(var_108, var_113);
    var_116 = wp::extract(var_i, var_115);
    var_118 = wp::extract(var_v, var_117);
    var_119 = wp::mul(var_116, var_118);
    var_120 = wp::add(var_114, var_119);
    // i[7] * v[0] - i[6] * v[1] + i[9] * v[5],                                               <L 129>
    var_122 = wp::extract(var_i, var_121);
    var_124 = wp::extract(var_v, var_123);
    var_125 = wp::mul(var_122, var_124);
    var_127 = wp::extract(var_i, var_126);
    var_129 = wp::extract(var_v, var_128);
    var_130 = wp::mul(var_127, var_129);
    var_131 = wp::sub(var_125, var_130);
    var_133 = wp::extract(var_i, var_132);
    var_135 = wp::extract(var_v, var_134);
    var_136 = wp::mul(var_133, var_135);
    var_137 = wp::add(var_131, var_136);
    var_138 = wp::vec_t<6, wp::float32>({var_28, var_57, var_86, var_103, var_120, var_137});
    return var_138;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:147
static CUDA_CALLABLE wp::vec_t<6, wp::float32> motion_cross_force_0(
    wp::vec_t<6, wp::float32> var_v,
    wp::vec_t<6, wp::float32> var_f)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 2;
    wp::float32 var_5;
    wp::vec_t<3, wp::float32> var_6;
    const wp::int32 var_7 = 3;
    wp::float32 var_8;
    const wp::int32 var_9 = 4;
    wp::float32 var_10;
    const wp::int32 var_11 = 5;
    wp::float32 var_12;
    wp::vec_t<3, wp::float32> var_13;
    const wp::int32 var_14 = 0;
    wp::float32 var_15;
    const wp::int32 var_16 = 1;
    wp::float32 var_17;
    const wp::int32 var_18 = 2;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    const wp::int32 var_21 = 3;
    wp::float32 var_22;
    const wp::int32 var_23 = 4;
    wp::float32 var_24;
    const wp::int32 var_25 = 5;
    wp::float32 var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<6, wp::float32> var_32;
    //---------
    // forward
    // def motion_cross_force(v: wp.spatial_vector, f: wp.spatial_vector) -> wp.spatial_vector:       <L 148>
    // v0 = wp.vec3(v[0], v[1], v[2])                                                         <L 150>
    var_1 = wp::extract(var_v, var_0);
    var_3 = wp::extract(var_v, var_2);
    var_5 = wp::extract(var_v, var_4);
    var_6 = wp::vec_t<3, wp::float32>(var_1, var_3, var_5);
    // v1 = wp.vec3(v[3], v[4], v[5])                                                         <L 151>
    var_8 = wp::extract(var_v, var_7);
    var_10 = wp::extract(var_v, var_9);
    var_12 = wp::extract(var_v, var_11);
    var_13 = wp::vec_t<3, wp::float32>(var_8, var_10, var_12);
    // f0 = wp.vec3(f[0], f[1], f[2])                                                         <L 152>
    var_15 = wp::extract(var_f, var_14);
    var_17 = wp::extract(var_f, var_16);
    var_19 = wp::extract(var_f, var_18);
    var_20 = wp::vec_t<3, wp::float32>(var_15, var_17, var_19);
    // f1 = wp.vec3(f[3], f[4], f[5])                                                         <L 153>
    var_22 = wp::extract(var_f, var_21);
    var_24 = wp::extract(var_f, var_23);
    var_26 = wp::extract(var_f, var_25);
    var_27 = wp::vec_t<3, wp::float32>(var_22, var_24, var_26);
    // ang = wp.cross(v0, f0) + wp.cross(v1, f1)                                              <L 155>
    var_28 = wp::cross(var_6, var_20);
    var_29 = wp::cross(var_13, var_27);
    var_30 = wp::add(var_28, var_29);
    // vel = wp.cross(v0, f1)                                                                 <L 156>
    var_31 = wp::cross(var_6, var_27);
    // return wp.spatial_vector(ang, vel)                                                     <L 158>
    var_32 = wp::vec_t<6, wp::float32>(var_30, var_31);
    return var_32;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:133
static CUDA_CALLABLE wp::vec_t<6, wp::float32> motion_cross_0(
    wp::vec_t<6, wp::float32> var_u,
    wp::vec_t<6, wp::float32> var_v)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 2;
    wp::float32 var_5;
    wp::vec_t<3, wp::float32> var_6;
    const wp::int32 var_7 = 3;
    wp::float32 var_8;
    const wp::int32 var_9 = 4;
    wp::float32 var_10;
    const wp::int32 var_11 = 5;
    wp::float32 var_12;
    wp::vec_t<3, wp::float32> var_13;
    const wp::int32 var_14 = 0;
    wp::float32 var_15;
    const wp::int32 var_16 = 1;
    wp::float32 var_17;
    const wp::int32 var_18 = 2;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    const wp::int32 var_21 = 3;
    wp::float32 var_22;
    const wp::int32 var_23 = 4;
    wp::float32 var_24;
    const wp::int32 var_25 = 5;
    wp::float32 var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<6, wp::float32> var_32;
    //---------
    // forward
    // def motion_cross(u: wp.spatial_vector, v: wp.spatial_vector) -> wp.spatial_vector:       <L 134>
    // u0 = wp.vec3(u[0], u[1], u[2])                                                         <L 136>
    var_1 = wp::extract(var_u, var_0);
    var_3 = wp::extract(var_u, var_2);
    var_5 = wp::extract(var_u, var_4);
    var_6 = wp::vec_t<3, wp::float32>(var_1, var_3, var_5);
    // u1 = wp.vec3(u[3], u[4], u[5])                                                         <L 137>
    var_8 = wp::extract(var_u, var_7);
    var_10 = wp::extract(var_u, var_9);
    var_12 = wp::extract(var_u, var_11);
    var_13 = wp::vec_t<3, wp::float32>(var_8, var_10, var_12);
    // v0 = wp.vec3(v[0], v[1], v[2])                                                         <L 138>
    var_15 = wp::extract(var_v, var_14);
    var_17 = wp::extract(var_v, var_16);
    var_19 = wp::extract(var_v, var_18);
    var_20 = wp::vec_t<3, wp::float32>(var_15, var_17, var_19);
    // v1 = wp.vec3(v[3], v[4], v[5])                                                         <L 139>
    var_22 = wp::extract(var_v, var_21);
    var_24 = wp::extract(var_v, var_23);
    var_26 = wp::extract(var_v, var_25);
    var_27 = wp::vec_t<3, wp::float32>(var_22, var_24, var_26);
    // ang = wp.cross(u0, v0)                                                                 <L 141>
    var_28 = wp::cross(var_6, var_20);
    // vel = wp.cross(u1, v0) + wp.cross(u0, v1)                                              <L 142>
    var_29 = wp::cross(var_13, var_20);
    var_30 = wp::cross(var_6, var_27);
    var_31 = wp::add(var_29, var_30);
    // return wp.spatial_vector(ang, vel)                                                     <L 144>
    var_32 = wp::vec_t<6, wp::float32>(var_28, var_31);
    return var_32;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/support.py:384
static CUDA_CALLABLE wp::vec_t<6, wp::float32> transform_force_0(
    wp::vec_t<3, wp::float32> var_force,
    wp::vec_t<3, wp::float32> var_torque,
    wp::vec_t<3, wp::float32> var_offset)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<6, wp::float32> var_2;
    //---------
    // forward
    // def transform_force(force: wp.vec3, torque: wp.vec3, offset: wp.vec3) -> wp.spatial_vector:       <L 385>
    // return wp.spatial_vector(torque - wp.cross(offset, force), force)                      <L 386>
    var_0 = wp::cross(var_offset, var_force);
    var_1 = wp::sub(var_torque, var_0);
    var_2 = wp::vec_t<6, wp::float32>(var_1, var_force);
    return var_2;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/support.py:389
static CUDA_CALLABLE wp::vec_t<6, wp::float32> transform_force_1(
    wp::vec_t<6, wp::float32> var_frc,
    wp::vec_t<3, wp::float32> var_offset)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<6, wp::float32> var_2;
    //---------
    // forward
    // def transform_force(frc: wp.spatial_vector, offset: wp.vec3) -> wp.spatial_vector:       <L 390>
    // force = wp.spatial_top(frc)                                                            <L 391>
    var_0 = wp::spatial_top(var_frc);
    // torque = wp.spatial_bottom(frc)                                                        <L 392>
    var_1 = wp::spatial_bottom(var_frc);
    // return transform_force(force, torque, offset)                                          <L 393>
    var_2 = transform_force_0(var_0, var_1, var_offset);
    return var_2;
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/smooth.py:3126
static CUDA_CALLABLE void _accumulate_jac_chain_0(
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::vec_t<3, wp::float32> var_offset,
    wp::vec_t<3, wp::float32> var_vec,
    wp::int32 var_bodyid,
    wp::int32 var_rowadr,
    wp::int32 var_rownnz,
    wp::float32 var_scale,
    wp::int32 var_worldid,
    wp::array_t<wp::float32> var_ten_J_out)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::int32 var_1;
    wp::int32 var_2;
    const wp::int32 var_3 = 0;
    bool var_4;
    wp::int32* var_5;
    wp::int32 var_6;
    wp::int32 var_7;
    wp::int32* var_8;
    wp::int32 var_9;
    wp::int32 var_10;
    wp::range_t var_11;
    wp::int32 var_12;
    wp::int32 var_13;
    const wp::int32 var_14 = 1;
    wp::int32 var_15;
    wp::int32 var_16;
    const wp::int32 var_17 = 0;
    bool var_18;
    wp::int32 var_19;
    wp::int32* var_20;
    bool var_21;
    wp::int32 var_22;
    const wp::int32 var_23 = 1;
    wp::int32 var_24;
    const wp::int32 var_25 = 0;
    bool var_26;
    wp::int32* var_27;
    bool var_28;
    wp::int32 var_29;
    bool var_30;
    wp::vec_t<6, wp::float32>* var_31;
    wp::vec_t<6, wp::float32> var_32;
    wp::vec_t<6, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    wp::vec_t<3, wp::float32> var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::float32 var_40 = 0.0;
    bool var_41;
    wp::slice_t var_42;
    const wp::int32 var_43 = 0;
    wp::array_t<wp::float32> var_44;
    wp::float32 var_45;
    wp::int32* var_46;
    wp::int32 var_47;
    wp::int32 var_48;
    //---------
    // forward
    // def _accumulate_jac_chain(                                                             <L 3127>
    // ptr = rownnz - 1                                                                       <L 3147>
    var_1 = wp::sub(var_rownnz, var_0);
    // bid = bodyid                                                                           <L 3148>
    var_2 = wp::copy(var_bodyid);
    // while bid > 0:                                                                         <L 3149>
    start_while_0:;
    var_4 = (var_2 > var_3);
    if ((var_4) == false) goto end_while_0;
        // bdofadr = body_dofadr[bid]                                                         <L 3150>
        var_5 = wp::address(var_body_dofadr, var_2);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // bdofnum = body_dofnum[bid]                                                         <L 3151>
        var_8 = wp::address(var_body_dofnum, var_2);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // for k_rev in range(bdofnum):                                                       <L 3153>
        var_11 = wp::range(var_9);
        start_for_2:;
            if (iter_cmp(var_11) == 0) goto end_for_2;
            var_12 = wp::iter_next(var_11);
            // dof = bdofadr + bdofnum - 1 - k_rev                                            <L 3154>
            var_13 = wp::add(var_6, var_9);
            var_15 = wp::sub(var_13, var_14);
            var_16 = wp::sub(var_15, var_12);
            // while ptr >= 0:                                                                <L 3156>
    start_while_4:;
            var_18 = (var_1 >= var_17);
    if ((var_18) == false) goto end_while_4;
                // sparseid = rowadr + ptr                                                    <L 3157>
                var_19 = wp::add(var_rowadr, var_1);
                // if ten_J_colind[sparseid] <= dof:                                          <L 3158>
                var_20 = wp::address(var_ten_J_colind, var_19);
                var_22 = wp::load(var_20);
                var_21 = (var_22 <= var_16);
                if (var_21) {
                    // break                                                                  <L 3159>
                    goto end_while_4;
                }
                // ptr -= 1                                                                   <L 3160>
                var_24 = wp::sub(var_1, var_23);
                wp::assign(var_1, var_24);
    goto start_while_4;
    end_while_4:;
            // if ptr >= 0 and ten_J_colind[sparseid] == dof:                                 <L 3161>
            var_26 = (var_1 >= var_25);
            var_27 = wp::address(var_ten_J_colind, var_19);
            var_29 = wp::load(var_27);
            var_28 = (var_29 == var_16);
            var_30 = var_26 && var_28;
            if (var_30) {
                // cdof = cdof_in[worldid, dof]                                               <L 3162>
                var_31 = wp::address(var_cdof_in, var_worldid, var_16);
                var_33 = wp::load(var_31);
                var_32 = wp::copy(var_33);
                // cdof_ang = wp.spatial_top(cdof)                                            <L 3163>
                var_34 = wp::spatial_top(var_32);
                // cdof_lin = wp.spatial_bottom(cdof)                                         <L 3164>
                var_35 = wp::spatial_bottom(var_32);
                // jacp = cdof_lin + wp.cross(cdof_ang, offset)                               <L 3165>
                var_36 = wp::cross(var_34, var_offset);
                var_37 = wp::add(var_35, var_36);
                // J = wp.dot(jacp, vec) * scale                                              <L 3166>
                var_38 = wp::dot(var_37, var_vec);
                var_39 = wp::mul(var_38, var_scale);
                // if J != 0.0:                                                               <L 3167>
                var_41 = (var_39 != var_40);
                if (var_41) {
                    // wp.atomic_add(ten_J_out[worldid], sparseid, J)                         <L 3168>
                    var_42 = wp::slice_t(var_worldid, var_worldid, var_43);
                    var_44 = wp::view(var_ten_J_out, var_42);
                    var_45 = wp::atomic_add(var_44, var_19, var_39);
                }
            }
            goto start_for_2;
        end_for_2:;
        // bid = body_parentid[bid]                                                           <L 3169>
        var_46 = wp::address(var_body_parentid, var_2);
        var_48 = wp::load(var_46);
        var_47 = wp::copy(var_48);
        wp::assign(var_2, var_47);
    goto start_while_0;
    end_while_0:;
}


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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:0
static CUDA_CALLABLE void normalize_with_norm_0(
    wp::vec_t<2, wp::float32> var_x,
    wp::vec_t<2, wp::float32> & ret_0,
    wp::float32 & ret_1)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 0.0;
    bool var_2;
    const wp::float32 var_3 = 0.0;
    wp::vec_t<2, wp::float32> var_4;
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/warp/_src/math.py:0
static CUDA_CALLABLE wp::float32 norm_l2_0(
    wp::vec_t<2, wp::float32> var_v)
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/util_misc.py:201
static CUDA_CALLABLE void wrap_inside_0(
    wp::vec_t<4, wp::float32> var_end,
    wp::float32 var_radius,
    wp::int32 var_maxiter,
    wp::float32 var_zinit,
    wp::float32 var_tolerance,
    wp::float32 & ret_0,
    wp::vec_t<2, wp::float32> & ret_1,
    wp::vec_t<2, wp::float32> & ret_2)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    wp::vec_t<2, wp::float32> var_4;
    const wp::int32 var_5 = 2;
    wp::float32 var_6;
    const wp::int32 var_7 = 3;
    wp::float32 var_8;
    wp::vec_t<2, wp::float32> var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::vec_t<2, wp::float32> var_12;
    wp::float32 var_13;
    bool var_14;
    bool var_15;
    const wp::float32 var_16 = 1e-15;
    bool var_17;
    bool var_18;
    bool var_19;
    bool var_20;
    const wp::float32 var_21 = 1.0;
    const wp::float32 var_22 = -1.0;
    const wp::float32 var_23 = 10000000000.0;
    wp::vec_t<2, wp::float32> var_24;
    wp::vec_t<2, wp::float32> var_25;
    bool var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    const wp::float32 var_30 = 0.0;
    bool var_31;
    const wp::float32 var_32 = 1.0;
    bool var_33;
    bool var_34;
    wp::vec_t<2, wp::float32> var_35;
    wp::vec_t<2, wp::float32> var_36;
    wp::float32 var_37;
    bool var_38;
    const wp::float32 var_39 = 1.0;
    const wp::float32 var_40 = -1.0;
    wp::vec_t<2, wp::float32> var_41;
    wp::vec_t<2, wp::float32> var_42;
    const wp::float32 var_43 = 0.5;
    wp::vec_t<2, wp::float32> var_44;
    wp::vec_t<2, wp::float32> var_45;
    wp::vec_t<2, wp::float32> var_46;
    wp::float32 var_47;
    wp::vec_t<2, wp::float32> var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::float32 var_56;
    const wp::float32 var_57 = 2.0;
    wp::float32 var_58;
    wp::float32 var_59;
    wp::float32 var_60;
    const wp::float32 var_61 = 1.0;
    const wp::float32 var_62 = -1.0;
    wp::float32 var_63;
    bool var_64;
    const wp::float32 var_65 = 1.0;
    const wp::float32 var_66 = -1.0;
    const wp::float32 var_67 = 1.0;
    wp::float32 var_68;
    bool var_69;
    const wp::float32 var_70 = 0.0;
    wp::float32 var_71;
    wp::float32 var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    wp::float32 var_75;
    wp::float32 var_76;
    wp::float32 var_77;
    const wp::float32 var_78 = 2.0;
    wp::float32 var_79;
    wp::float32 var_80;
    wp::float32 var_81;
    wp::float32 var_82;
    const wp::float32 var_83 = 0.0;
    bool var_84;
    const wp::float32 var_85 = 0.0;
    const wp::int32 var_86 = 0;
    wp::int32 var_87;
    bool var_88;
    wp::float32 var_89;
    bool var_90;
    bool var_91;
    wp::float32 var_92;
    const wp::float32 var_93 = 1.0;
    wp::float32 var_94;
    wp::float32 var_95;
    wp::float32 var_96;
    wp::float32 var_97;
    wp::float32 var_98;
    const wp::float32 var_99 = 1.0;
    wp::float32 var_100;
    wp::float32 var_101;
    wp::float32 var_102;
    wp::float32 var_103;
    wp::float32 var_104;
    wp::float32 var_105;
    const wp::float32 var_106 = 2.0;
    const wp::float32 var_107 = 1.0;
    wp::float32 var_108;
    wp::float32 var_109;
    wp::float32 var_110;
    wp::float32 var_111;
    wp::float32 var_112;
    const wp::float32 var_113 = -1e-15;
    bool var_114;
    const wp::float32 var_115 = 0.0;
    wp::float32 var_116;
    wp::float32 var_117;
    bool var_118;
    const wp::float32 var_119 = 0.0;
    wp::float32 var_120;
    wp::float32 var_121;
    wp::float32 var_122;
    wp::float32 var_123;
    wp::float32 var_124;
    wp::float32 var_125;
    const wp::float32 var_126 = 2.0;
    wp::float32 var_127;
    wp::float32 var_128;
    wp::float32 var_129;
    wp::float32 var_130;
    bool var_131;
    const wp::float32 var_132 = 0.0;
    const wp::int32 var_133 = 1;
    wp::int32 var_134;
    bool var_135;
    const wp::float32 var_136 = 0.0;
    const wp::int32 var_137 = 0;
    wp::float32 var_138;
    const wp::int32 var_139 = 3;
    wp::float32 var_140;
    wp::float32 var_141;
    const wp::int32 var_142 = 1;
    wp::float32 var_143;
    const wp::int32 var_144 = 2;
    wp::float32 var_145;
    wp::float32 var_146;
    wp::float32 var_147;
    const wp::float32 var_148 = 0.0;
    bool var_149;
    wp::vec_t<2, wp::float32> var_150;
    wp::float32 var_151;
    wp::float32 var_152;
    wp::float32 var_153;
    wp::float32 var_154;
    wp::vec_t<2, wp::float32> var_155;
    wp::float32 var_156;
    wp::float32 var_157;
    wp::float32 var_158;
    wp::float32 var_159;
    wp::vec_t<2, wp::float32> var_160;
    wp::float32 var_161;
    wp::vec_t<2, wp::float32> var_162;
    wp::float32 var_163;
    wp::float32 var_164;
    const wp::int32 var_165 = 0;
    wp::float32 var_166;
    wp::float32 var_167;
    wp::float32 var_168;
    const wp::int32 var_169 = 1;
    wp::float32 var_170;
    wp::float32 var_171;
    wp::float32 var_172;
    wp::float32 var_173;
    wp::float32 var_174;
    const wp::int32 var_175 = 0;
    wp::float32 var_176;
    wp::float32 var_177;
    wp::float32 var_178;
    const wp::int32 var_179 = 1;
    wp::float32 var_180;
    wp::float32 var_181;
    wp::float32 var_182;
    wp::float32 var_183;
    wp::vec_t<2, wp::float32> var_184;
    const wp::float32 var_185 = 0.0;
    //---------
    // forward
    // def wrap_inside(                                                                       <L 202>
    // end0 = wp.vec2(end[0], end[1])                                                         <L 223>
    var_1 = wp::extract(var_end, var_0);
    var_3 = wp::extract(var_end, var_2);
    var_4 = wp::vec_t<2, wp::float32>(var_1, var_3);
    // end1 = wp.vec2(end[2], end[3])                                                         <L 224>
    var_6 = wp::extract(var_end, var_5);
    var_8 = wp::extract(var_end, var_7);
    var_9 = wp::vec_t<2, wp::float32>(var_6, var_8);
    // len0 = wp.norm_l2(end0)                                                                <L 227>
    var_10 = norm_l2_0(var_4);
    // len1 = wp.norm_l2(end1)                                                                <L 228>
    var_11 = norm_l2_0(var_9);
    // dif = end1 - end0                                                                      <L 229>
    var_12 = wp::sub(var_9, var_4);
    // dd = wp.dot(dif, dif)                                                                  <L 230>
    var_13 = wp::dot(var_12, var_12);
    // if (len0 <= radius) or (len1 <= radius) or (radius < MJ_MINVAL) or (len0 < MJ_MINVAL) or (len1 < MJ_MINVAL):       <L 233>
    var_14 = (var_10 <= var_radius);
    var_15 = (var_11 <= var_radius);
    var_17 = (var_radius < var_16);
    var_18 = (var_10 < var_16);
    var_19 = (var_11 < var_16);
    var_20 = var_14 || var_15 || var_17 || var_18 || var_19;
    if (var_20) {
        // return -1.0, wp.vec2(MJ_MAXVAL), wp.vec2(MJ_MAXVAL)                                <L 234>
        var_24 = wp::vec_t<2, wp::float32>(var_23);
        var_25 = wp::vec_t<2, wp::float32>(var_23);
        ret_0 = var_22;
        ret_1 = var_24;
        ret_2 = var_25;
        return;
    }
    // if dd > MJ_MINVAL:                                                                     <L 237>
    var_26 = (var_13 > var_16);
    if (var_26) {
        // a = -wp.dot(dif, end0) / dd                                                        <L 239>
        var_27 = wp::dot(var_12, var_4);
        var_28 = wp::neg(var_27);
        var_29 = wp::div(var_28, var_13);
        // if (a > 0.0) and (a < 1.0):                                                        <L 242>
        var_31 = (var_29 > var_30);
        var_33 = (var_29 < var_32);
        var_34 = var_31 && var_33;
        if (var_34) {
            // tmp = end0 + a * dif                                                           <L 243>
            var_35 = wp::mul(var_29, var_12);
            var_36 = wp::add(var_4, var_35);
            // if wp.norm_l2(tmp) <= radius:                                                  <L 244>
            var_37 = norm_l2_0(var_36);
            var_38 = (var_37 <= var_radius);
            if (var_38) {
                // return -1.0, wp.vec2(MJ_MAXVAL), wp.vec2(MJ_MAXVAL)                        <L 245>
                var_41 = wp::vec_t<2, wp::float32>(var_23);
                var_42 = wp::vec_t<2, wp::float32>(var_23);
                ret_0 = var_40;
                ret_1 = var_41;
                ret_2 = var_42;
                return;
            }
        }
    }
    // pnt = 0.5 * (end0 + end1)                                                              <L 248>
    var_44 = wp::add(var_4, var_9);
    var_45 = wp::mul(var_43, var_44);
    // pnt, _ = math.normalize_with_norm(pnt)                                                 <L 249>
    normalize_with_norm_0(var_45, var_46, var_47);
    // pnt *= radius                                                                          <L 250>
    var_48 = wp::mul(var_46, var_radius);
    // A = math.safe_div(radius, len0)                                                        <L 253>
    var_49 = safe_div_0(var_radius, var_10);
    // B = math.safe_div(radius, len1)                                                        <L 254>
    var_50 = safe_div_0(var_radius, var_11);
    // sq_A = A * A                                                                           <L 255>
    var_51 = wp::mul(var_49, var_49);
    // sq_B = B * B                                                                           <L 256>
    var_52 = wp::mul(var_50, var_50);
    // cosG = math.safe_div(len0 * len0 + len1 * len1 - dd, 2.0 * len0 * len1)                <L 257>
    var_53 = wp::mul(var_10, var_10);
    var_54 = wp::mul(var_11, var_11);
    var_55 = wp::add(var_53, var_54);
    var_56 = wp::sub(var_55, var_13);
    var_58 = wp::mul(var_57, var_10);
    var_59 = wp::mul(var_58, var_11);
    var_60 = safe_div_0(var_56, var_59);
    // if cosG < -1.0 + MJ_MINVAL:                                                            <L 258>
    var_63 = wp::add(var_62, var_16);
    var_64 = (var_60 < var_63);
    if (var_64) {
        // return -1.0, pnt, pnt                                                              <L 259>
        ret_0 = var_66;
        ret_1 = var_48;
        ret_2 = var_48;
        return;
    }
    if (!var_64) {
        // elif cosG > 1.0 - MJ_MINVAL:                                                       <L 260>
        var_68 = wp::sub(var_67, var_16);
        var_69 = (var_60 > var_68);
        if (var_69) {
            // return 0.0, pnt, pnt                                                           <L 261>
            ret_0 = var_70;
            ret_1 = var_48;
            ret_2 = var_48;
            return;
        }
    }
    // G = wp.acos(cosG)                                                                      <L 262>
    var_71 = wp::acos(var_60);
    // z = zinit                                                                              <L 265>
    var_72 = wp::copy(var_zinit);
    // f = wp.asin(A * z) + wp.asin(B * z) - 2.0 * wp.asin(z) + G                             <L 266>
    var_73 = wp::mul(var_49, var_72);
    var_74 = wp::asin(var_73);
    var_75 = wp::mul(var_50, var_72);
    var_76 = wp::asin(var_75);
    var_77 = wp::add(var_74, var_76);
    var_79 = wp::asin(var_72);
    var_80 = wp::mul(var_78, var_79);
    var_81 = wp::sub(var_77, var_80);
    var_82 = wp::add(var_81, var_71);
    // if f > 0.0:                                                                            <L 269>
    var_84 = (var_82 > var_83);
    if (var_84) {
        // return 0.0, pnt, pnt                                                               <L 270>
        ret_0 = var_85;
        ret_1 = var_48;
        ret_2 = var_48;
        return;
    }
    // iter = int(0)                                                                          <L 273>
    var_87 = wp::int(var_86);
    // while (iter < maxiter) and (wp.abs(f) > tolerance):                                    <L 275>
    start_while_5:;
    var_88 = (var_87 < var_maxiter);
    var_89 = wp::abs(var_82);
    var_90 = (var_89 > var_tolerance);
    var_91 = var_88 && var_90;
    if ((var_91) == false) goto end_while_5;
        // sq_z = z * z                                                                       <L 277>
        var_92 = wp::mul(var_72, var_72);
        // df = (                                                                             <L 278>
        // A / wp.max(MJ_MINVAL, wp.sqrt(1.0 - sq_z * sq_A))                                  <L 279>
        var_94 = wp::mul(var_92, var_51);
        var_95 = wp::sub(var_93, var_94);
        var_96 = wp::sqrt(var_95);
        var_97 = wp::max(var_16, var_96);
        var_98 = wp::div(var_49, var_97);
        // + B / wp.max(MJ_MINVAL, wp.sqrt(1.0 - sq_z * sq_B))                                <L 280>
        var_100 = wp::mul(var_92, var_52);
        var_101 = wp::sub(var_99, var_100);
        var_102 = wp::sqrt(var_101);
        var_103 = wp::max(var_16, var_102);
        var_104 = wp::div(var_50, var_103);
        var_105 = wp::add(var_98, var_104);
        // - 2.0 / wp.max(MJ_MINVAL, wp.sqrt(1.0 - sq_z))                                     <L 281>
        var_108 = wp::sub(var_107, var_92);
        var_109 = wp::sqrt(var_108);
        var_110 = wp::max(var_16, var_109);
        var_111 = wp::div(var_106, var_110);
        var_112 = wp::sub(var_105, var_111);
        // if df > -MJ_MINVAL:                                                                <L 285>
        var_114 = (var_112 > var_113);
        if (var_114) {
            // return 0.0, pnt, pnt                                                           <L 286>
            ret_0 = var_115;
            ret_1 = var_48;
            ret_2 = var_48;
            return;
        }
        // z1 = z - math.safe_div(f, df)                                                      <L 289>
        var_116 = safe_div_0(var_82, var_112);
        var_117 = wp::sub(var_72, var_116);
        // if z1 > z:                                                                         <L 292>
        var_118 = (var_117 > var_72);
        if (var_118) {
            // return 0.0, pnt, pnt                                                           <L 293>
            ret_0 = var_119;
            ret_1 = var_48;
            ret_2 = var_48;
            return;
        }
        // z = z1                                                                             <L 296>
        var_120 = wp::copy(var_117);
        // f = wp.asin(A * z) + wp.asin(B * z) - 2.0 * wp.asin(z) + G                         <L 297>
        var_121 = wp::mul(var_49, var_120);
        var_122 = wp::asin(var_121);
        var_123 = wp::mul(var_50, var_120);
        var_124 = wp::asin(var_123);
        var_125 = wp::add(var_122, var_124);
        var_127 = wp::asin(var_120);
        var_128 = wp::mul(var_126, var_127);
        var_129 = wp::sub(var_125, var_128);
        var_130 = wp::add(var_129, var_71);
        // if f > tolerance:                                                                  <L 300>
        var_131 = (var_130 > var_tolerance);
        if (var_131) {
            // return 0.0, pnt, pnt                                                           <L 301>
            ret_0 = var_132;
            ret_1 = var_48;
            ret_2 = var_48;
            return;
        }
        // iter += 1                                                                          <L 303>
        var_134 = wp::add(var_87, var_133);
        wp::assign(var_72, var_120);
        wp::assign(var_82, var_130);
        wp::assign(var_87, var_134);
    goto start_while_5;
    end_while_5:;
    // if iter >= maxiter:                                                                    <L 306>
    var_135 = (var_87 >= var_maxiter);
    if (var_135) {
        // return 0.0, pnt, pnt                                                               <L 307>
        ret_0 = var_136;
        ret_1 = var_48;
        ret_2 = var_48;
        return;
    }
    // if end[0] * end[3] - end[1] * end[2] > 0.0:                                            <L 310>
    var_138 = wp::extract(var_end, var_137);
    var_140 = wp::extract(var_end, var_139);
    var_141 = wp::mul(var_138, var_140);
    var_143 = wp::extract(var_end, var_142);
    var_145 = wp::extract(var_end, var_144);
    var_146 = wp::mul(var_143, var_145);
    var_147 = wp::sub(var_141, var_146);
    var_149 = (var_147 > var_148);
    if (var_149) {
        // vec = end0                                                                         <L 311>
        var_150 = wp::copy(var_4);
        // ang = wp.asin(z) - wp.asin(A * z)                                                  <L 312>
        var_151 = wp::asin(var_72);
        var_152 = wp::mul(var_49, var_72);
        var_153 = wp::asin(var_152);
        var_154 = wp::sub(var_151, var_153);
    }
    if (!var_149) {
        // vec = end1                                                                         <L 314>
        var_155 = wp::copy(var_9);
        // ang = wp.asin(z) - wp.asin(B * z)                                                  <L 315>
        var_156 = wp::asin(var_72);
        var_157 = wp::mul(var_50, var_72);
        var_158 = wp::asin(var_157);
        var_159 = wp::sub(var_156, var_158);
    }
    var_160 = wp::where(var_149, var_150, var_155);
    var_161 = wp::where(var_149, var_154, var_159);
    // vec, _ = math.normalize_with_norm(vec)                                                 <L 317>
    normalize_with_norm_0(var_160, var_162, var_163);
    // pnt = wp.vec2(                                                                         <L 318>
    // radius * (wp.cos(ang) * vec[0] - wp.sin(ang) * vec[1]),                                <L 319>
    var_164 = wp::cos(var_161);
    var_166 = wp::extract(var_162, var_165);
    var_167 = wp::mul(var_164, var_166);
    var_168 = wp::sin(var_161);
    var_170 = wp::extract(var_162, var_169);
    var_171 = wp::mul(var_168, var_170);
    var_172 = wp::sub(var_167, var_171);
    var_173 = wp::mul(var_radius, var_172);
    // radius * (wp.sin(ang) * vec[0] + wp.cos(ang) * vec[1]),                                <L 320>
    var_174 = wp::sin(var_161);
    var_176 = wp::extract(var_162, var_175);
    var_177 = wp::mul(var_174, var_176);
    var_178 = wp::cos(var_161);
    var_180 = wp::extract(var_162, var_179);
    var_181 = wp::mul(var_178, var_180);
    var_182 = wp::add(var_177, var_181);
    var_183 = wp::mul(var_radius, var_182);
    var_184 = wp::vec_t<2, wp::float32>(var_173, var_183);
    // return 0.0, pnt, pnt                                                                   <L 323>
    ret_0 = var_185;
    ret_1 = var_184;
    ret_2 = var_184;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/util_misc.py:30
static CUDA_CALLABLE bool is_intersect_0(
    wp::vec_t<2, wp::float32> var_p1,
    wp::vec_t<2, wp::float32> var_p2,
    wp::vec_t<2, wp::float32> var_p3,
    wp::vec_t<2, wp::float32> var_p4)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::int32 var_5 = 0;
    wp::float32 var_6;
    const wp::int32 var_7 = 0;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::int32 var_11 = 0;
    wp::float32 var_12;
    const wp::int32 var_13 = 0;
    wp::float32 var_14;
    wp::float32 var_15;
    const wp::int32 var_16 = 1;
    wp::float32 var_17;
    const wp::int32 var_18 = 1;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    const wp::float32 var_24 = 1e-15;
    bool var_25;
    const bool var_26 = false;
    const wp::int32 var_27 = 0;
    wp::float32 var_28;
    const wp::int32 var_29 = 0;
    wp::float32 var_30;
    wp::float32 var_31;
    const wp::int32 var_32 = 1;
    wp::float32 var_33;
    const wp::int32 var_34 = 1;
    wp::float32 var_35;
    wp::float32 var_36;
    wp::float32 var_37;
    const wp::int32 var_38 = 1;
    wp::float32 var_39;
    const wp::int32 var_40 = 1;
    wp::float32 var_41;
    wp::float32 var_42;
    const wp::int32 var_43 = 0;
    wp::float32 var_44;
    const wp::int32 var_45 = 0;
    wp::float32 var_46;
    wp::float32 var_47;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    const wp::int32 var_51 = 0;
    wp::float32 var_52;
    const wp::int32 var_53 = 0;
    wp::float32 var_54;
    wp::float32 var_55;
    const wp::int32 var_56 = 1;
    wp::float32 var_57;
    const wp::int32 var_58 = 1;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::float32 var_61;
    const wp::int32 var_62 = 1;
    wp::float32 var_63;
    const wp::int32 var_64 = 1;
    wp::float32 var_65;
    wp::float32 var_66;
    const wp::int32 var_67 = 0;
    wp::float32 var_68;
    const wp::int32 var_69 = 0;
    wp::float32 var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    const wp::int32 var_75 = 0;
    bool var_76;
    const wp::float32 var_77 = 1.0;
    bool var_78;
    const wp::float32 var_79 = 0.0;
    bool var_80;
    const wp::float32 var_81 = 1.0;
    bool var_82;
    bool var_83;
    const bool var_84 = true;
    const bool var_85 = false;
    //---------
    // forward
    // def is_intersect(p1: wp.vec2, p2: wp.vec2, p3: wp.vec2, p4: wp.vec2) -> bool:          <L 31>
    // det = (p4[1] - p3[1]) * (p2[0] - p1[0]) - (p4[0] - p3[0]) * (p2[1] - p1[1])            <L 44>
    var_1 = wp::extract(var_p4, var_0);
    var_3 = wp::extract(var_p3, var_2);
    var_4 = wp::sub(var_1, var_3);
    var_6 = wp::extract(var_p2, var_5);
    var_8 = wp::extract(var_p1, var_7);
    var_9 = wp::sub(var_6, var_8);
    var_10 = wp::mul(var_4, var_9);
    var_12 = wp::extract(var_p4, var_11);
    var_14 = wp::extract(var_p3, var_13);
    var_15 = wp::sub(var_12, var_14);
    var_17 = wp::extract(var_p2, var_16);
    var_19 = wp::extract(var_p1, var_18);
    var_20 = wp::sub(var_17, var_19);
    var_21 = wp::mul(var_15, var_20);
    var_22 = wp::sub(var_10, var_21);
    // if wp.abs(det) < MJ_MINVAL:                                                            <L 46>
    var_23 = wp::abs(var_22);
    var_25 = (var_23 < var_24);
    if (var_25) {
        // return False                                                                       <L 47>
        return var_26;
    }
    // a = ((p4[0] - p3[0]) * (p1[1] - p3[1]) - (p4[1] - p3[1]) * (p1[0] - p3[0])) / det       <L 50>
    var_28 = wp::extract(var_p4, var_27);
    var_30 = wp::extract(var_p3, var_29);
    var_31 = wp::sub(var_28, var_30);
    var_33 = wp::extract(var_p1, var_32);
    var_35 = wp::extract(var_p3, var_34);
    var_36 = wp::sub(var_33, var_35);
    var_37 = wp::mul(var_31, var_36);
    var_39 = wp::extract(var_p4, var_38);
    var_41 = wp::extract(var_p3, var_40);
    var_42 = wp::sub(var_39, var_41);
    var_44 = wp::extract(var_p1, var_43);
    var_46 = wp::extract(var_p3, var_45);
    var_47 = wp::sub(var_44, var_46);
    var_48 = wp::mul(var_42, var_47);
    var_49 = wp::sub(var_37, var_48);
    var_50 = wp::div(var_49, var_22);
    // b = ((p2[0] - p1[0]) * (p1[1] - p3[1]) - (p2[1] - p1[1]) * (p1[0] - p3[0])) / det       <L 51>
    var_52 = wp::extract(var_p2, var_51);
    var_54 = wp::extract(var_p1, var_53);
    var_55 = wp::sub(var_52, var_54);
    var_57 = wp::extract(var_p1, var_56);
    var_59 = wp::extract(var_p3, var_58);
    var_60 = wp::sub(var_57, var_59);
    var_61 = wp::mul(var_55, var_60);
    var_63 = wp::extract(var_p2, var_62);
    var_65 = wp::extract(var_p1, var_64);
    var_66 = wp::sub(var_63, var_65);
    var_68 = wp::extract(var_p1, var_67);
    var_70 = wp::extract(var_p3, var_69);
    var_71 = wp::sub(var_68, var_70);
    var_72 = wp::mul(var_66, var_71);
    var_73 = wp::sub(var_61, var_72);
    var_74 = wp::div(var_73, var_22);
    // if a >= 0 and a <= 1.0 and b >= 0.0 and b <= 1.0:                                      <L 53>
    var_76 = (var_50 >= var_75);
    var_78 = (var_50 <= var_77);
    var_80 = (var_74 >= var_79);
    var_82 = (var_74 <= var_81);
    var_83 = var_76 && var_78 && var_80 && var_82;
    if (var_83) {
        // return True                                                                        <L 54>
        return var_84;
    }
    if (!var_83) {
        // return False                                                                       <L 56>
        return var_85;
    }
    return {};
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/util_misc.py:76
static CUDA_CALLABLE wp::float32 length_circle_0(
    wp::vec_t<2, wp::float32> var_p0,
    wp::vec_t<2, wp::float32> var_p1,
    wp::int32 var_ind,
    wp::float32 var_radius)
{
    //---------
    // primal vars
    wp::vec_t<2, wp::float32> var_0;
    wp::float32 var_1;
    wp::vec_t<2, wp::float32> var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    const wp::int32 var_6 = 1;
    wp::float32 var_7;
    const wp::int32 var_8 = 0;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::int32 var_11 = 0;
    wp::float32 var_12;
    const wp::int32 var_13 = 1;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::float32 var_17 = 0.0;
    bool var_18;
    const wp::int32 var_19 = 0;
    bool var_20;
    bool var_21;
    const wp::float32 var_22 = 0.0;
    bool var_23;
    const wp::int32 var_24 = 0;
    bool var_25;
    bool var_26;
    bool var_27;
    const wp::float32 var_28 = 2.0;
    const wp::float32 var_29 = 3.141592653589793;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    //---------
    // forward
    // def length_circle(p0: wp.vec2, p1: wp.vec2, ind: int, radius: float) -> float:         <L 77>
    // p0n, _ = math.normalize_with_norm(p0)                                                  <L 90>
    normalize_with_norm_0(var_p0, var_0, var_1);
    // p1n, _ = math.normalize_with_norm(p1)                                                  <L 91>
    normalize_with_norm_0(var_p1, var_2, var_3);
    // angle = wp.acos(wp.dot(p0n, p1n))                                                      <L 93>
    var_4 = wp::dot(var_0, var_2);
    var_5 = wp::acos(var_4);
    // cross = p0[1] * p1[0] - p0[0] * p1[1]                                                  <L 96>
    var_7 = wp::extract(var_p0, var_6);
    var_9 = wp::extract(var_p1, var_8);
    var_10 = wp::mul(var_7, var_9);
    var_12 = wp::extract(var_p0, var_11);
    var_14 = wp::extract(var_p1, var_13);
    var_15 = wp::mul(var_12, var_14);
    var_16 = wp::sub(var_10, var_15);
    // if (cross > 0.0 and ind != 0) or (cross < 0.0 and ind == 0):                           <L 97>
    var_18 = (var_16 > var_17);
    var_20 = (var_ind != var_19);
    var_21 = var_18 && var_20;
    var_23 = (var_16 < var_22);
    var_25 = (var_ind == var_24);
    var_26 = var_23 && var_25;
    var_27 = var_21 || var_26;
    if (var_27) {
        // angle = 2.0 * wp.pi - angle                                                        <L 98>
        var_30 = wp::mul(var_28, var_29);
        var_31 = wp::sub(var_30, var_5);
    }
    var_32 = wp::where(var_27, var_31, var_5);
    // return radius * angle                                                                  <L 100>
    var_33 = wp::mul(var_radius, var_32);
    return var_33;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/util_misc.py:103
static CUDA_CALLABLE void wrap_circle_0(
    wp::vec_t<4, wp::float32> var_end,
    wp::vec_t<2, wp::float32> var_side,
    wp::float32 var_radius,
    wp::float32 & ret_0,
    wp::vec_t<2, wp::float32> & ret_1,
    wp::vec_t<2, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 10000000000.0;
    bool var_2;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    const wp::int32 var_5 = 1;
    wp::float32 var_6;
    wp::vec_t<2, wp::float32> var_7;
    const wp::int32 var_8 = 2;
    wp::float32 var_9;
    const wp::int32 var_10 = 3;
    wp::float32 var_11;
    wp::vec_t<2, wp::float32> var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    bool var_16;
    bool var_17;
    const wp::float32 var_18 = 1e-15;
    bool var_19;
    bool var_20;
    const wp::float32 var_21 = 1.0;
    const wp::float32 var_22 = -1.0;
    wp::vec_t<2, wp::float32> var_23;
    wp::vec_t<2, wp::float32> var_24;
    wp::vec_t<2, wp::float32> var_25;
    wp::float32 var_26;
    bool var_27;
    const wp::float32 var_28 = 1.0;
    const wp::float32 var_29 = -1.0;
    wp::vec_t<2, wp::float32> var_30;
    wp::vec_t<2, wp::float32> var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    const wp::float32 var_35 = 0.0;
    const wp::float32 var_36 = 1.0;
    wp::float32 var_37;
    wp::vec_t<2, wp::float32> var_38;
    wp::vec_t<2, wp::float32> var_39;
    wp::float32 var_40;
    bool var_41;
    bool var_42;
    wp::float32 var_43;
    const wp::float32 var_44 = 0.0;
    bool var_45;
    bool var_46;
    bool var_47;
    const wp::float32 var_48 = 1.0;
    const wp::float32 var_49 = -1.0;
    wp::vec_t<2, wp::float32> var_50;
    wp::vec_t<2, wp::float32> var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    wp::float32 var_55;
    const wp::int32 var_56 = 0;
    wp::float32 var_57;
    wp::float32 var_58;
    const wp::int32 var_59 = 1;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    wp::float32 var_64;
    const wp::int32 var_65 = 1;
    wp::float32 var_66;
    wp::float32 var_67;
    const wp::int32 var_68 = 0;
    wp::float32 var_69;
    wp::float32 var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    wp::float32 var_73;
    wp::vec_t<2, wp::float32> var_74;
    const wp::int32 var_75 = 2;
    wp::float32 var_76;
    wp::float32 var_77;
    const wp::int32 var_78 = 3;
    wp::float32 var_79;
    wp::float32 var_80;
    wp::float32 var_81;
    wp::float32 var_82;
    wp::float32 var_83;
    const wp::int32 var_84 = 3;
    wp::float32 var_85;
    wp::float32 var_86;
    const wp::int32 var_87 = 2;
    wp::float32 var_88;
    wp::float32 var_89;
    wp::float32 var_90;
    wp::float32 var_91;
    wp::float32 var_92;
    wp::vec_t<2, wp::float32> var_93;
    const wp::int32 var_94 = 0;
    wp::float32 var_95;
    wp::float32 var_96;
    const wp::int32 var_97 = 1;
    wp::float32 var_98;
    wp::float32 var_99;
    wp::float32 var_100;
    wp::float32 var_101;
    wp::float32 var_102;
    const wp::int32 var_103 = 1;
    wp::float32 var_104;
    wp::float32 var_105;
    const wp::int32 var_106 = 0;
    wp::float32 var_107;
    wp::float32 var_108;
    wp::float32 var_109;
    wp::float32 var_110;
    wp::float32 var_111;
    wp::vec_t<2, wp::float32> var_112;
    const wp::int32 var_113 = 2;
    wp::float32 var_114;
    wp::float32 var_115;
    const wp::int32 var_116 = 3;
    wp::float32 var_117;
    wp::float32 var_118;
    wp::float32 var_119;
    wp::float32 var_120;
    wp::float32 var_121;
    const wp::int32 var_122 = 3;
    wp::float32 var_123;
    wp::float32 var_124;
    const wp::int32 var_125 = 2;
    wp::float32 var_126;
    wp::float32 var_127;
    wp::float32 var_128;
    wp::float32 var_129;
    wp::float32 var_130;
    wp::vec_t<2, wp::float32> var_131;
    wp::vec_t<2, wp::float32> var_132;
    wp::vec_t<2, wp::float32> var_133;
    wp::float32 var_134;
    wp::float32 var_135;
    wp::vec_t<2, wp::float32> var_136;
    wp::vec_t<2, wp::float32> var_137;
    wp::float32 var_138;
    wp::float32 var_139;
    wp::vec_t<2, wp::float32> var_140;
    wp::float32 var_141;
    wp::float32 var_142;
    wp::vec_t<2, wp::float32> var_143;
    wp::float32 var_144;
    wp::float32 var_145;
    wp::vec_t<2, wp::float32> var_146;
    wp::float32 var_147;
    wp::vec_t<2, wp::float32> var_148;
    wp::float32 var_149;
    bool var_150;
    const wp::float32 var_151 = 10000.0;
    const wp::float32 var_152 = -10000.0;
    wp::float32 var_153;
    bool var_154;
    const wp::float32 var_155 = 10000.0;
    const wp::float32 var_156 = -10000.0;
    wp::float32 var_157;
    bool var_158;
    wp::vec_t<2, wp::float32> var_159;
    wp::vec_t<2, wp::float32> var_160;
    const wp::int32 var_161 = 0;
    wp::vec_t<2, wp::float32> var_162;
    wp::vec_t<2, wp::float32> var_163;
    const wp::int32 var_164 = 1;
    wp::vec_t<2, wp::float32> var_165;
    wp::vec_t<2, wp::float32> var_166;
    wp::int32 var_167;
    bool var_168;
    const wp::float32 var_169 = 1.0;
    const wp::float32 var_170 = -1.0;
    wp::vec_t<2, wp::float32> var_171;
    wp::vec_t<2, wp::float32> var_172;
    wp::float32 var_173;
    //---------
    // forward
    // def wrap_circle(end: wp.vec4, side: wp.vec2, radius: float) -> Tuple[float, wp.vec2, wp.vec2]:       <L 104>
    // valid_side = wp.norm_l2(side) < MJ_MAXVAL                                              <L 115>
    var_0 = norm_l2_0(var_side);
    var_2 = (var_0 < var_1);
    // end0 = wp.vec2(end[0], end[1])                                                         <L 117>
    var_4 = wp::extract(var_end, var_3);
    var_6 = wp::extract(var_end, var_5);
    var_7 = wp::vec_t<2, wp::float32>(var_4, var_6);
    // end1 = wp.vec2(end[2], end[3])                                                         <L 118>
    var_9 = wp::extract(var_end, var_8);
    var_11 = wp::extract(var_end, var_10);
    var_12 = wp::vec_t<2, wp::float32>(var_9, var_11);
    // sqlen0 = wp.dot(end0, end0)                                                            <L 120>
    var_13 = wp::dot(var_7, var_7);
    // sqlen1 = wp.dot(end1, end1)                                                            <L 121>
    var_14 = wp::dot(var_12, var_12);
    // sqrad = radius * radius                                                                <L 122>
    var_15 = wp::mul(var_radius, var_radius);
    // if (sqlen0 < sqrad) or (sqlen1 < sqrad) or (radius < MJ_MINVAL):                       <L 125>
    var_16 = (var_13 < var_15);
    var_17 = (var_14 < var_15);
    var_19 = (var_radius < var_18);
    var_20 = var_16 || var_17 || var_19;
    if (var_20) {
        // return -1.0, wp.vec2(MJ_MAXVAL), wp.vec2(MJ_MAXVAL)                                <L 126>
        var_23 = wp::vec_t<2, wp::float32>(var_1);
        var_24 = wp::vec_t<2, wp::float32>(var_1);
        ret_0 = var_22;
        ret_1 = var_23;
        ret_2 = var_24;
        return;
    }
    // dif = end1 - end0                                                                      <L 129>
    var_25 = wp::sub(var_12, var_7);
    // dd = wp.dot(dif, dif)                                                                  <L 130>
    var_26 = wp::dot(var_25, var_25);
    // if dd < MJ_MINVAL:                                                                     <L 131>
    var_27 = (var_26 < var_18);
    if (var_27) {
        // return -1.0, wp.vec2(MJ_MAXVAL), wp.vec2(MJ_MAXVAL)                                <L 132>
        var_30 = wp::vec_t<2, wp::float32>(var_1);
        var_31 = wp::vec_t<2, wp::float32>(var_1);
        ret_0 = var_29;
        ret_1 = var_30;
        ret_2 = var_31;
        return;
    }
    // a = -wp.dot(dif, end0) / dd                                                            <L 135>
    var_32 = wp::dot(var_25, var_7);
    var_33 = wp::neg(var_32);
    var_34 = wp::div(var_33, var_26);
    // a = wp.clamp(a, 0.0, 1.0)                                                              <L 136>
    var_37 = wp::clamp(var_34, var_35, var_36);
    // tmp = a * dif + end0                                                                   <L 139>
    var_38 = wp::mul(var_37, var_25);
    var_39 = wp::add(var_38, var_7);
    // if (wp.dot(tmp, tmp) > sqrad) and (not valid_side or wp.dot(side, tmp) >= 0.0):        <L 140>
    var_40 = wp::dot(var_39, var_39);
    var_41 = (var_40 > var_15);
    var_42 = wp::unot(var_2);
    var_43 = wp::dot(var_side, var_39);
    var_45 = (var_43 >= var_44);
    var_46 = var_42 || var_45;
    var_47 = var_41 && var_46;
    if (var_47) {
        // return -1.0, wp.vec2(MJ_MAXVAL), wp.vec2(MJ_MAXVAL)                                <L 141>
        var_50 = wp::vec_t<2, wp::float32>(var_1);
        var_51 = wp::vec_t<2, wp::float32>(var_1);
        ret_0 = var_49;
        ret_1 = var_50;
        ret_2 = var_51;
        return;
    }
    // sqrt0 = wp.sqrt(sqlen0 - sqrad)                                                        <L 143>
    var_52 = wp::sub(var_13, var_15);
    var_53 = wp::sqrt(var_52);
    // sqrt1 = wp.sqrt(sqlen1 - sqrad)                                                        <L 144>
    var_54 = wp::sub(var_14, var_15);
    var_55 = wp::sqrt(var_54);
    // sol00 = wp.vec2(                                                                       <L 147>
    // math.safe_div(end[0] * sqrad + radius * end[1] * sqrt0, sqlen0),                       <L 148>
    var_57 = wp::extract(var_end, var_56);
    var_58 = wp::mul(var_57, var_15);
    var_60 = wp::extract(var_end, var_59);
    var_61 = wp::mul(var_radius, var_60);
    var_62 = wp::mul(var_61, var_53);
    var_63 = wp::add(var_58, var_62);
    var_64 = safe_div_0(var_63, var_13);
    // math.safe_div(end[1] * sqrad - radius * end[0] * sqrt0, sqlen0),                       <L 149>
    var_66 = wp::extract(var_end, var_65);
    var_67 = wp::mul(var_66, var_15);
    var_69 = wp::extract(var_end, var_68);
    var_70 = wp::mul(var_radius, var_69);
    var_71 = wp::mul(var_70, var_53);
    var_72 = wp::sub(var_67, var_71);
    var_73 = safe_div_0(var_72, var_13);
    var_74 = wp::vec_t<2, wp::float32>(var_64, var_73);
    // sol01 = wp.vec2(                                                                       <L 151>
    // math.safe_div(end[2] * sqrad - radius * end[3] * sqrt1, sqlen1),                       <L 152>
    var_76 = wp::extract(var_end, var_75);
    var_77 = wp::mul(var_76, var_15);
    var_79 = wp::extract(var_end, var_78);
    var_80 = wp::mul(var_radius, var_79);
    var_81 = wp::mul(var_80, var_55);
    var_82 = wp::sub(var_77, var_81);
    var_83 = safe_div_0(var_82, var_14);
    // math.safe_div(end[3] * sqrad + radius * end[2] * sqrt1, sqlen1),                       <L 153>
    var_85 = wp::extract(var_end, var_84);
    var_86 = wp::mul(var_85, var_15);
    var_88 = wp::extract(var_end, var_87);
    var_89 = wp::mul(var_radius, var_88);
    var_90 = wp::mul(var_89, var_55);
    var_91 = wp::add(var_86, var_90);
    var_92 = safe_div_0(var_91, var_14);
    var_93 = wp::vec_t<2, wp::float32>(var_83, var_92);
    // sol10 = wp.vec2(                                                                       <L 156>
    // math.safe_div(end[0] * sqrad - radius * end[1] * sqrt0, sqlen0),                       <L 157>
    var_95 = wp::extract(var_end, var_94);
    var_96 = wp::mul(var_95, var_15);
    var_98 = wp::extract(var_end, var_97);
    var_99 = wp::mul(var_radius, var_98);
    var_100 = wp::mul(var_99, var_53);
    var_101 = wp::sub(var_96, var_100);
    var_102 = safe_div_0(var_101, var_13);
    // math.safe_div(end[1] * sqrad + radius * end[0] * sqrt0, sqlen0),                       <L 158>
    var_104 = wp::extract(var_end, var_103);
    var_105 = wp::mul(var_104, var_15);
    var_107 = wp::extract(var_end, var_106);
    var_108 = wp::mul(var_radius, var_107);
    var_109 = wp::mul(var_108, var_53);
    var_110 = wp::add(var_105, var_109);
    var_111 = safe_div_0(var_110, var_13);
    var_112 = wp::vec_t<2, wp::float32>(var_102, var_111);
    // sol11 = wp.vec2(                                                                       <L 160>
    // math.safe_div(end[2] * sqrad + radius * end[3] * sqrt1, sqlen1),                       <L 161>
    var_114 = wp::extract(var_end, var_113);
    var_115 = wp::mul(var_114, var_15);
    var_117 = wp::extract(var_end, var_116);
    var_118 = wp::mul(var_radius, var_117);
    var_119 = wp::mul(var_118, var_55);
    var_120 = wp::add(var_115, var_119);
    var_121 = safe_div_0(var_120, var_14);
    // math.safe_div(end[3] * sqrad - radius * end[2] * sqrt1, sqlen1),                       <L 162>
    var_123 = wp::extract(var_end, var_122);
    var_124 = wp::mul(var_123, var_15);
    var_126 = wp::extract(var_end, var_125);
    var_127 = wp::mul(var_radius, var_126);
    var_128 = wp::mul(var_127, var_55);
    var_129 = wp::sub(var_124, var_128);
    var_130 = safe_div_0(var_129, var_14);
    var_131 = wp::vec_t<2, wp::float32>(var_121, var_130);
    // if valid_side:                                                                         <L 166>
    if (var_2) {
        // tmp0, _ = math.normalize_with_norm(sol00 + sol01)                                  <L 167>
        var_132 = wp::add(var_74, var_93);
        normalize_with_norm_0(var_132, var_133, var_134);
        // good0 = wp.dot(tmp0, side)                                                         <L 168>
        var_135 = wp::dot(var_133, var_side);
        // tmp1, _ = math.normalize_with_norm(sol10 + sol11)                                  <L 169>
        var_136 = wp::add(var_112, var_131);
        normalize_with_norm_0(var_136, var_137, var_138);
        // good1 = wp.dot(tmp1, side)                                                         <L 170>
        var_139 = wp::dot(var_137, var_side);
    }
    if (!var_2) {
        // tmp0 = sol00 - sol01                                                               <L 172>
        var_140 = wp::sub(var_74, var_93);
        // good0 = -wp.dot(tmp0, tmp0)                                                        <L 173>
        var_141 = wp::dot(var_140, var_140);
        var_142 = wp::neg(var_141);
        // tmp1 = sol10 - sol11                                                               <L 174>
        var_143 = wp::sub(var_112, var_131);
        // good1 = -wp.dot(tmp1, tmp1)                                                        <L 175>
        var_144 = wp::dot(var_143, var_143);
        var_145 = wp::neg(var_144);
    }
    var_146 = wp::where(var_2, var_133, var_140);
    var_147 = wp::where(var_2, var_135, var_142);
    var_148 = wp::where(var_2, var_137, var_143);
    var_149 = wp::where(var_2, var_139, var_145);
    // if is_intersect(end0, sol00, end1, sol01):                                             <L 178>
    var_150 = is_intersect_0(var_7, var_74, var_12, var_93);
    if (var_150) {
        // good0 = -10000.0                                                                   <L 179>
    }
    var_153 = wp::where(var_150, var_152, var_147);
    // if is_intersect(end0, sol10, end1, sol11):                                             <L 180>
    var_154 = is_intersect_0(var_7, var_112, var_12, var_131);
    if (var_154) {
        // good1 = -10000.0                                                                   <L 181>
    }
    var_157 = wp::where(var_154, var_156, var_149);
    // if good0 > good1:                                                                      <L 184>
    var_158 = (var_153 > var_157);
    if (var_158) {
        // pnt0 = sol00                                                                       <L 185>
        var_159 = wp::copy(var_74);
        // pnt1 = sol01                                                                       <L 186>
        var_160 = wp::copy(var_93);
        // ind = 0                                                                            <L 187>
    }
    if (!var_158) {
        // pnt0 = sol10                                                                       <L 189>
        var_162 = wp::copy(var_112);
        // pnt1 = sol11                                                                       <L 190>
        var_163 = wp::copy(var_131);
        // ind = 1                                                                            <L 191>
    }
    var_165 = wp::where(var_158, var_159, var_162);
    var_166 = wp::where(var_158, var_160, var_163);
    var_167 = wp::where(var_158, var_161, var_164);
    // if is_intersect(end0, pnt0, end1, pnt1):                                               <L 194>
    var_168 = is_intersect_0(var_7, var_165, var_12, var_166);
    if (var_168) {
        // return -1.0, wp.vec2(MJ_MAXVAL), wp.vec2(MJ_MAXVAL)                                <L 195>
        var_171 = wp::vec_t<2, wp::float32>(var_1);
        var_172 = wp::vec_t<2, wp::float32>(var_1);
        ret_0 = var_170;
        ret_1 = var_171;
        ret_2 = var_172;
        return;
    }
    // return length_circle(pnt0, pnt1, ind, radius), pnt0, pnt1                              <L 198>
    var_173 = length_circle_0(var_165, var_166, var_167, var_radius);
    ret_0 = var_173;
    ret_1 = var_165;
    ret_2 = var_166;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/util_misc.py:326
static CUDA_CALLABLE void wrap_0(
    wp::vec_t<3, wp::float32> var_x0,
    wp::vec_t<3, wp::float32> var_x1,
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::float32 var_radius,
    wp::int32 var_geomtype,
    wp::vec_t<3, wp::float32> var_side,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 4;
    bool var_1;
    const wp::int32 var_2 = 5;
    bool var_3;
    bool var_4;
    const wp::float32 var_5 = 10000000000.0;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::mat_t<3, 3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::float32 var_13;
    const wp::float32 var_14 = 1e-15;
    bool var_15;
    wp::float32 var_16;
    bool var_17;
    bool var_18;
    const wp::float32 var_19 = 1.0;
    const wp::float32 var_20 = -1.0;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    const wp::int32 var_23 = 4;
    bool var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::float32 var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::float32 var_29;
    bool var_30;
    wp::vec_t<3, wp::float32> var_31;
    const wp::int32 var_32 = 0;
    wp::int32 var_33;
    const wp::int32 var_34 = 1;
    wp::float32 var_35;
    const wp::int32 var_36 = 0;
    wp::float32 var_37;
    bool var_38;
    const wp::int32 var_39 = 1;
    wp::float32 var_40;
    const wp::int32 var_41 = 2;
    wp::float32 var_42;
    bool var_43;
    bool var_44;
    const wp::int32 var_45 = 1;
    wp::int32 var_46;
    const wp::int32 var_47 = 2;
    wp::float32 var_48;
    const wp::int32 var_49 = 0;
    wp::float32 var_50;
    bool var_51;
    const wp::int32 var_52 = 2;
    wp::float32 var_53;
    const wp::int32 var_54 = 1;
    wp::float32 var_55;
    bool var_56;
    bool var_57;
    const wp::int32 var_58 = 2;
    wp::int32 var_59;
    const wp::float32 var_60 = 1.0;
    wp::vec_t<3, wp::float32> var_61;
    const wp::float32 var_62 = 0.0;
    wp::vec_t<3, wp::float32> var_63;
    wp::vec_t<3, wp::float32> var_64;
    wp::float32 var_65;
    wp::float32 var_66;
    wp::vec_t<3, wp::float32> var_67;
    wp::vec_t<3, wp::float32> var_68;
    wp::vec_t<3, wp::float32> var_69;
    wp::float32 var_70;
    const wp::float32 var_71 = 1.0;
    const wp::float32 var_72 = 0.0;
    const wp::float32 var_73 = 0.0;
    wp::vec_t<3, wp::float32> var_74;
    const wp::float32 var_75 = 0.0;
    const wp::float32 var_76 = 1.0;
    const wp::float32 var_77 = 0.0;
    wp::vec_t<3, wp::float32> var_78;
    wp::vec_t<3, wp::float32> var_79;
    wp::vec_t<3, wp::float32> var_80;
    wp::float32 var_81;
    wp::float32 var_82;
    wp::float32 var_83;
    wp::float32 var_84;
    wp::vec_t<4, wp::float32> var_85;
    wp::float32 var_86;
    bool var_87;
    wp::vec_t<3, wp::float32> var_88;
    wp::vec_t<3, wp::float32> var_89;
    wp::float32 var_90;
    wp::float32 var_91;
    wp::vec_t<2, wp::float32> var_92;
    wp::vec_t<2, wp::float32> var_93;
    wp::float32 var_94;
    wp::vec_t<2, wp::float32> var_95;
    wp::float32 var_96;
    wp::vec_t<2, wp::float32> var_97;
    wp::vec_t<2, wp::float32> var_98;
    wp::float32 var_99;
    bool var_100;
    bool var_101;
    wp::float32 var_102;
    wp::vec_t<2, wp::float32> var_103;
    wp::vec_t<2, wp::float32> var_104;
    const wp::int32 var_105 = 20;
    const wp::float32 var_106 = 0.9999999;
    const wp::float32 var_107 = 1e-06;
    wp::float32 var_108;
    wp::vec_t<2, wp::float32> var_109;
    wp::vec_t<2, wp::float32> var_110;
    wp::float32 var_111;
    wp::vec_t<2, wp::float32> var_112;
    wp::vec_t<2, wp::float32> var_113;
    const wp::float32 var_114 = 0.0;
    bool var_115;
    const wp::float32 var_116 = 1.0;
    const wp::float32 var_117 = -1.0;
    wp::vec_t<3, wp::float32> var_118;
    wp::vec_t<3, wp::float32> var_119;
    const wp::int32 var_120 = 0;
    wp::float32 var_121;
    wp::vec_t<3, wp::float32> var_122;
    const wp::int32 var_123 = 1;
    wp::float32 var_124;
    wp::vec_t<3, wp::float32> var_125;
    wp::vec_t<3, wp::float32> var_126;
    const wp::int32 var_127 = 0;
    wp::float32 var_128;
    wp::vec_t<3, wp::float32> var_129;
    const wp::int32 var_130 = 1;
    wp::float32 var_131;
    wp::vec_t<3, wp::float32> var_132;
    wp::vec_t<3, wp::float32> var_133;
    const wp::int32 var_134 = 5;
    bool var_135;
    const wp::int32 var_136 = 0;
    wp::float32 var_137;
    const wp::int32 var_138 = 0;
    wp::float32 var_139;
    wp::float32 var_140;
    const wp::int32 var_141 = 0;
    wp::float32 var_142;
    const wp::int32 var_143 = 0;
    wp::float32 var_144;
    wp::float32 var_145;
    wp::float32 var_146;
    const wp::int32 var_147 = 1;
    wp::float32 var_148;
    const wp::int32 var_149 = 1;
    wp::float32 var_150;
    wp::float32 var_151;
    const wp::int32 var_152 = 1;
    wp::float32 var_153;
    const wp::int32 var_154 = 1;
    wp::float32 var_155;
    wp::float32 var_156;
    wp::float32 var_157;
    wp::float32 var_158;
    wp::float32 var_159;
    const wp::int32 var_160 = 0;
    wp::float32 var_161;
    const wp::int32 var_162 = 0;
    wp::float32 var_163;
    wp::float32 var_164;
    const wp::int32 var_165 = 0;
    wp::float32 var_166;
    const wp::int32 var_167 = 0;
    wp::float32 var_168;
    wp::float32 var_169;
    wp::float32 var_170;
    const wp::int32 var_171 = 1;
    wp::float32 var_172;
    const wp::int32 var_173 = 1;
    wp::float32 var_174;
    wp::float32 var_175;
    const wp::int32 var_176 = 1;
    wp::float32 var_177;
    const wp::int32 var_178 = 1;
    wp::float32 var_179;
    wp::float32 var_180;
    wp::float32 var_181;
    wp::float32 var_182;
    wp::float32 var_183;
    const wp::int32 var_184 = 2;
    wp::float32 var_185;
    const wp::int32 var_186 = 2;
    wp::float32 var_187;
    const wp::int32 var_188 = 2;
    wp::float32 var_189;
    wp::float32 var_190;
    wp::float32 var_191;
    wp::float32 var_192;
    wp::float32 var_193;
    wp::float32 var_194;
    wp::float32 var_195;
    const wp::int32 var_196 = 2;
    const wp::int32 var_197 = 2;
    wp::float32 var_198;
    const wp::int32 var_199 = 2;
    wp::float32 var_200;
    const wp::int32 var_201 = 2;
    wp::float32 var_202;
    wp::float32 var_203;
    wp::float32 var_204;
    wp::float32 var_205;
    wp::float32 var_206;
    wp::float32 var_207;
    wp::float32 var_208;
    wp::float32 var_209;
    const wp::int32 var_210 = 2;
    const wp::int32 var_211 = 2;
    wp::float32 var_212;
    const wp::int32 var_213 = 2;
    wp::float32 var_214;
    wp::float32 var_215;
    wp::float32 var_216;
    wp::float32 var_217;
    wp::float32 var_218;
    wp::float32 var_219;
    wp::float32 var_220;
    wp::float32 var_221;
    wp::vec_t<3, wp::float32> var_222;
    wp::vec_t<3, wp::float32> var_223;
    wp::vec_t<3, wp::float32> var_224;
    wp::vec_t<3, wp::float32> var_225;
    //---------
    // forward
    // def wrap(                                                                              <L 327>
    // if geomtype != WrapType.SPHERE and geomtype != WrapType.CYLINDER:                      <L 345>
    var_1 = (var_geomtype != var_0);
    var_3 = (var_geomtype != var_2);
    var_4 = var_1 && var_3;
    if (var_4) {
        // return MJ_MAXVAL, wp.vec3(MJ_MAXVAL), wp.vec3(MJ_MAXVAL)                           <L 346>
        var_6 = wp::vec_t<3, wp::float32>(var_5);
        var_7 = wp::vec_t<3, wp::float32>(var_5);
        ret_0 = var_5;
        ret_1 = var_6;
        ret_2 = var_7;
        return;
    }
    // matT = wp.transpose(mat)                                                               <L 349>
    var_8 = wp::transpose(var_mat);
    // p0 = matT @ (x0 - pos)                                                                 <L 350>
    var_9 = wp::sub(var_x0, var_pos);
    var_10 = wp::mul(var_8, var_9);
    // p1 = matT @ (x1 - pos)                                                                 <L 351>
    var_11 = wp::sub(var_x1, var_pos);
    var_12 = wp::mul(var_8, var_11);
    // if (wp.norm_l2(p0) < MJ_MINVAL) or (wp.norm_l2(p1) < MJ_MINVAL):                       <L 354>
    var_13 = norm_l2_0(var_10);
    var_15 = (var_13 < var_14);
    var_16 = norm_l2_0(var_12);
    var_17 = (var_16 < var_14);
    var_18 = var_15 || var_17;
    if (var_18) {
        // return -1.0, wp.vec3(MJ_MAXVAL), wp.vec3(MJ_MAXVAL)                                <L 355>
        var_21 = wp::vec_t<3, wp::float32>(var_5);
        var_22 = wp::vec_t<3, wp::float32>(var_5);
        ret_0 = var_20;
        ret_1 = var_21;
        ret_2 = var_22;
        return;
    }
    // if geomtype == WrapType.SPHERE:                                                        <L 358>
    var_24 = (var_geomtype == var_23);
    if (var_24) {
        // axis0, _ = math.normalize_with_norm(p0)                                            <L 360>
        normalize_with_norm_0(var_10, var_25, var_26);
        // normal = wp.cross(p0, p1)                                                          <L 363>
        var_27 = wp::cross(var_10, var_12);
        // normal, nrm = math.normalize_with_norm(normal)                                     <L 364>
        normalize_with_norm_0(var_27, var_28, var_29);
        // if nrm < MJ_MINVAL:                                                                <L 367>
        var_30 = (var_29 < var_14);
        if (var_30) {
            // axis0_abs = wp.abs(axis0)                                                      <L 369>
            var_31 = wp::abs(var_25);
            // i = int(0)                                                                     <L 370>
            var_33 = wp::int(var_32);
            // if (axis0_abs[1] > axis0_abs[0]) and (axis0_abs[1] > axis0_abs[2]):            <L 371>
            var_35 = wp::extract(var_31, var_34);
            var_37 = wp::extract(var_31, var_36);
            var_38 = (var_35 > var_37);
            var_40 = wp::extract(var_31, var_39);
            var_42 = wp::extract(var_31, var_41);
            var_43 = (var_40 > var_42);
            var_44 = var_38 && var_43;
            if (var_44) {
                // i = 1                                                                      <L 372>
            }
            var_46 = wp::where(var_44, var_45, var_33);
            // if (axis0_abs[2] > axis0_abs[0]) and (axis0_abs[2] > axis0_abs[1]):            <L 373>
            var_48 = wp::extract(var_31, var_47);
            var_50 = wp::extract(var_31, var_49);
            var_51 = (var_48 > var_50);
            var_53 = wp::extract(var_31, var_52);
            var_55 = wp::extract(var_31, var_54);
            var_56 = (var_53 > var_55);
            var_57 = var_51 && var_56;
            if (var_57) {
                // i = 2                                                                      <L 374>
            }
            var_59 = wp::where(var_57, var_58, var_46);
            // axis1 = wp.vec3(1.0)                                                           <L 377>
            var_61 = wp::vec_t<3, wp::float32>(var_60);
            // axis1[i] = 0.0                                                                 <L 378>
            wp::assign_inplace(var_61, var_59, var_62);
            // normal = wp.cross(axis0, axis1)                                                <L 381>
            var_63 = wp::cross(var_25, var_61);
            // normal, _ = math.normalize_with_norm(normal)                                   <L 382>
            normalize_with_norm_0(var_63, var_64, var_65);
        }
        var_66 = wp::where(var_30, var_65, var_26);
        var_67 = wp::where(var_30, var_64, var_28);
        // axis1 = wp.cross(normal, axis0)                                                    <L 385>
        var_68 = wp::cross(var_67, var_25);
        // axis1, _ = math.normalize_with_norm(axis1)                                         <L 386>
        normalize_with_norm_0(var_68, var_69, var_70);
    }
    if (!var_24) {
        // axis0 = wp.vec3(1.0, 0.0, 0.0)                                                     <L 389>
        var_74 = wp::vec_t<3, wp::float32>(var_71, var_72, var_73);
        // axis1 = wp.vec3(0.0, 1.0, 0.0)                                                     <L 392>
        var_78 = wp::vec_t<3, wp::float32>(var_75, var_76, var_77);
    }
    var_79 = wp::where(var_24, var_25, var_74);
    var_80 = wp::where(var_24, var_69, var_78);
    // end = wp.vec4(                                                                         <L 395>
    // wp.dot(p0, axis0),                                                                     <L 396>
    var_81 = wp::dot(var_10, var_79);
    // wp.dot(p0, axis1),                                                                     <L 397>
    var_82 = wp::dot(var_10, var_80);
    // wp.dot(p1, axis0),                                                                     <L 398>
    var_83 = wp::dot(var_12, var_79);
    // wp.dot(p1, axis1),                                                                     <L 399>
    var_84 = wp::dot(var_12, var_80);
    var_85 = wp::vec_t<4, wp::float32>(var_81, var_82, var_83, var_84);
    // valid_side = wp.norm_l2(side) < MJ_MAXVAL                                              <L 403>
    var_86 = norm_l2_0(var_side);
    var_87 = (var_86 < var_5);
    // if valid_side:                                                                         <L 405>
    if (var_87) {
        // sidepnt = matT @ (side - pos)                                                      <L 407>
        var_88 = wp::sub(var_side, var_pos);
        var_89 = wp::mul(var_8, var_88);
        // sidepnt_proj = wp.vec2(                                                            <L 410>
        // wp.dot(sidepnt, axis0),                                                            <L 411>
        var_90 = wp::dot(var_89, var_79);
        // wp.dot(sidepnt, axis1),                                                            <L 412>
        var_91 = wp::dot(var_89, var_80);
        var_92 = wp::vec_t<2, wp::float32>(var_90, var_91);
        // sidepnt_proj, _ = math.normalize_with_norm(sidepnt_proj)                           <L 415>
        normalize_with_norm_0(var_92, var_93, var_94);
        // sidepnt_proj *= radius                                                             <L 416>
        var_95 = wp::mul(var_93, var_radius);
    }
    var_96 = wp::where(var_87, var_94, var_70);
    if (!var_87) {
        // sidepnt_proj = wp.vec2(MJ_MAXVAL)                                                  <L 418>
        var_97 = wp::vec_t<2, wp::float32>(var_5);
    }
    var_98 = wp::where(var_87, var_95, var_97);
    // if valid_side and wp.norm_l2(sidepnt) < radius:                                        <L 421>
    var_99 = norm_l2_0(var_89);
    var_100 = (var_99 < var_radius);
    var_101 = var_87 && var_100;
    if (var_101) {
        // wlen, pnt0, pnt1 = wrap_inside(end, radius)                                        <L 422>
        wrap_inside_0(var_85, var_radius, var_105, var_106, var_107, var_102, var_103, var_104);
    }
    if (!var_101) {
        // wlen, pnt0, pnt1 = wrap_circle(end, sidepnt_proj, radius)                          <L 424>
        wrap_circle_0(var_85, var_98, var_radius, var_108, var_109, var_110);
    }
    var_111 = wp::where(var_101, var_102, var_108);
    var_112 = wp::where(var_101, var_103, var_109);
    var_113 = wp::where(var_101, var_104, var_110);
    // if wlen < 0.0:                                                                         <L 427>
    var_115 = (var_111 < var_114);
    if (var_115) {
        // return -1.0, wp.vec3(MJ_MAXVAL), wp.vec3(MJ_MAXVAL)                                <L 428>
        var_118 = wp::vec_t<3, wp::float32>(var_5);
        var_119 = wp::vec_t<3, wp::float32>(var_5);
        ret_0 = var_117;
        ret_1 = var_118;
        ret_2 = var_119;
        return;
    }
    // res0 = axis0 * pnt0[0] + axis1 * pnt0[1]                                               <L 431>
    var_121 = wp::extract(var_112, var_120);
    var_122 = wp::mul(var_79, var_121);
    var_124 = wp::extract(var_112, var_123);
    var_125 = wp::mul(var_80, var_124);
    var_126 = wp::add(var_122, var_125);
    // res1 = axis0 * pnt1[0] + axis1 * pnt1[1]                                               <L 432>
    var_128 = wp::extract(var_113, var_127);
    var_129 = wp::mul(var_79, var_128);
    var_131 = wp::extract(var_113, var_130);
    var_132 = wp::mul(var_80, var_131);
    var_133 = wp::add(var_129, var_132);
    // if geomtype == WrapType.CYLINDER:                                                      <L 435>
    var_135 = (var_geomtype == var_134);
    if (var_135) {
        // L0 = wp.sqrt((p0[0] - res0[0]) * (p0[0] - res0[0]) + (p0[1] - res0[1]) * (p0[1] - res0[1]))       <L 437>
        var_137 = wp::extract(var_10, var_136);
        var_139 = wp::extract(var_126, var_138);
        var_140 = wp::sub(var_137, var_139);
        var_142 = wp::extract(var_10, var_141);
        var_144 = wp::extract(var_126, var_143);
        var_145 = wp::sub(var_142, var_144);
        var_146 = wp::mul(var_140, var_145);
        var_148 = wp::extract(var_10, var_147);
        var_150 = wp::extract(var_126, var_149);
        var_151 = wp::sub(var_148, var_150);
        var_153 = wp::extract(var_10, var_152);
        var_155 = wp::extract(var_126, var_154);
        var_156 = wp::sub(var_153, var_155);
        var_157 = wp::mul(var_151, var_156);
        var_158 = wp::add(var_146, var_157);
        var_159 = wp::sqrt(var_158);
        // L1 = wp.sqrt((p1[0] - res1[0]) * (p1[0] - res1[0]) + (p1[1] - res1[1]) * (p1[1] - res1[1]))       <L 438>
        var_161 = wp::extract(var_12, var_160);
        var_163 = wp::extract(var_133, var_162);
        var_164 = wp::sub(var_161, var_163);
        var_166 = wp::extract(var_12, var_165);
        var_168 = wp::extract(var_133, var_167);
        var_169 = wp::sub(var_166, var_168);
        var_170 = wp::mul(var_164, var_169);
        var_172 = wp::extract(var_12, var_171);
        var_174 = wp::extract(var_133, var_173);
        var_175 = wp::sub(var_172, var_174);
        var_177 = wp::extract(var_12, var_176);
        var_179 = wp::extract(var_133, var_178);
        var_180 = wp::sub(var_177, var_179);
        var_181 = wp::mul(var_175, var_180);
        var_182 = wp::add(var_170, var_181);
        var_183 = wp::sqrt(var_182);
        // res0[2] = p0[2] + (p1[2] - p0[2]) * math.safe_div(L0, L0 + wlen + L1)              <L 439>
        var_185 = wp::extract(var_10, var_184);
        var_187 = wp::extract(var_12, var_186);
        var_189 = wp::extract(var_10, var_188);
        var_190 = wp::sub(var_187, var_189);
        var_191 = wp::add(var_159, var_111);
        var_192 = wp::add(var_191, var_183);
        var_193 = safe_div_0(var_159, var_192);
        var_194 = wp::mul(var_190, var_193);
        var_195 = wp::add(var_185, var_194);
        wp::assign_inplace(var_126, var_196, var_195);
        // res1[2] = p0[2] + (p1[2] - p0[2]) * math.safe_div(L0 + wlen, L0 + wlen + L1)       <L 440>
        var_198 = wp::extract(var_10, var_197);
        var_200 = wp::extract(var_12, var_199);
        var_202 = wp::extract(var_10, var_201);
        var_203 = wp::sub(var_200, var_202);
        var_204 = wp::add(var_159, var_111);
        var_205 = wp::add(var_159, var_111);
        var_206 = wp::add(var_205, var_183);
        var_207 = safe_div_0(var_204, var_206);
        var_208 = wp::mul(var_203, var_207);
        var_209 = wp::add(var_198, var_208);
        wp::assign_inplace(var_133, var_210, var_209);
        // height = wp.abs(res1[2] - res0[2])                                                 <L 443>
        var_212 = wp::extract(var_133, var_211);
        var_214 = wp::extract(var_126, var_213);
        var_215 = wp::sub(var_212, var_214);
        var_216 = wp::abs(var_215);
        // wlen = wp.sqrt(wlen * wlen + height * height)                                      <L 444>
        var_217 = wp::mul(var_111, var_111);
        var_218 = wp::mul(var_216, var_216);
        var_219 = wp::add(var_217, var_218);
        var_220 = wp::sqrt(var_219);
    }
    var_221 = wp::where(var_135, var_220, var_111);
    // wpnt0 = mat @ res0 + pos                                                               <L 447>
    var_222 = wp::mul(var_mat, var_126);
    var_223 = wp::add(var_222, var_pos);
    // wpnt1 = mat @ res1 + pos                                                               <L 448>
    var_224 = wp::mul(var_mat, var_133);
    var_225 = wp::add(var_224, var_pos);
    // return wlen, wpnt0, wpnt1                                                              <L 450>
    ret_0 = var_221;
    ret_1 = var_223;
    ret_2 = var_225;
    return;
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
static CUDA_CALLABLE wp::vec_t<3, wp::float32> safe_div_0(
    wp::vec_t<3, wp::float32> var_x,
    wp::float32 var_y)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    bool var_1;
    const wp::float32 var_2 = 1e-15;
    const wp::float32 var_3 = 1e-15;
    wp::float32 var_4;
    wp::vec_t<3, wp::float32> var_5;
    //---------
    // forward
    // def safe_div(x: Any, y: Any) -> Any:                                                   <L 1>
    // return x / wp.where(y != 0.0, y, types.MJ_MINVAL)                                      <L 2>
    var_1 = (var_y != var_0);
    var_4 = wp::where(var_1, var_y, var_3);
    var_5 = wp::div(var_x, var_4);
    return var_5;
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/smooth.py:1584
static CUDA_CALLABLE void _accumulate_jac_dot_chain_0(
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::int32> var_dof_jntid,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_dot_in,
    wp::vec_t<3, wp::float32> var_offset,
    wp::vec_t<3, wp::float32> var_pvel_lin,
    wp::vec_t<3, wp::float32> var_dpnt,
    wp::vec_t<3, wp::float32> var_dvel,
    wp::int32 var_bodyid,
    wp::int32 var_rowadr,
    wp::int32 var_rownnz,
    wp::float32 var_scale,
    wp::int32 var_worldid,
    wp::array_t<wp::float32> var_ten_Jdot_out)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::int32 var_1;
    wp::int32 var_2;
    const wp::int32 var_3 = 0;
    bool var_4;
    wp::int32* var_5;
    wp::int32 var_6;
    wp::int32 var_7;
    wp::int32* var_8;
    wp::int32 var_9;
    wp::int32 var_10;
    wp::range_t var_11;
    wp::int32 var_12;
    wp::int32 var_13;
    const wp::int32 var_14 = 1;
    wp::int32 var_15;
    wp::int32 var_16;
    const wp::int32 var_17 = 0;
    bool var_18;
    wp::int32 var_19;
    wp::int32* var_20;
    bool var_21;
    wp::int32 var_22;
    const wp::int32 var_23 = 1;
    wp::int32 var_24;
    const wp::int32 var_25 = 0;
    bool var_26;
    wp::int32* var_27;
    bool var_28;
    wp::int32 var_29;
    bool var_30;
    wp::vec_t<6, wp::float32>* var_31;
    wp::vec_t<6, wp::float32> var_32;
    wp::vec_t<6, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    wp::vec_t<3, wp::float32> var_35;
    wp::vec_t<6, wp::float32>* var_36;
    wp::vec_t<6, wp::float32> var_37;
    wp::vec_t<6, wp::float32> var_38;
    wp::int32* var_39;
    wp::int32 var_40;
    wp::int32 var_41;
    wp::int32* var_42;
    wp::int32 var_43;
    wp::int32 var_44;
    wp::int32* var_45;
    wp::int32 var_46;
    wp::int32 var_47;
    const wp::int32 var_48 = 1;
    bool var_49;
    const wp::int32 var_50 = 0;
    bool var_51;
    const wp::int32 var_52 = 3;
    wp::int32 var_53;
    bool var_54;
    bool var_55;
    bool var_56;
    wp::vec_t<6, wp::float32>* var_57;
    wp::vec_t<6, wp::float32> var_58;
    wp::vec_t<6, wp::float32> var_59;
    wp::vec_t<6, wp::float32> var_60;
    wp::vec_t<3, wp::float32> var_61;
    wp::vec_t<3, wp::float32> var_62;
    wp::vec_t<3, wp::float32> var_63;
    wp::vec_t<3, wp::float32> var_64;
    wp::vec_t<3, wp::float32> var_65;
    wp::vec_t<3, wp::float32> var_66;
    wp::vec_t<3, wp::float32> var_67;
    wp::vec_t<3, wp::float32> var_68;
    wp::float32 var_69;
    wp::float32 var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    const wp::float32 var_73 = 0.0;
    bool var_74;
    wp::slice_t var_75;
    const wp::int32 var_76 = 0;
    wp::array_t<wp::float32> var_77;
    wp::float32 var_78;
    wp::int32* var_79;
    wp::int32 var_80;
    wp::int32 var_81;
    //---------
    // forward
    // def _accumulate_jac_dot_chain(                                                         <L 1585>
    // ptr = rownnz - 1                                                                       <L 1612>
    var_1 = wp::sub(var_rownnz, var_0);
    // bid = bodyid                                                                           <L 1613>
    var_2 = wp::copy(var_bodyid);
    // while bid > 0:                                                                         <L 1614>
    start_while_0:;
    var_4 = (var_2 > var_3);
    if ((var_4) == false) goto end_while_0;
        // bdofadr = body_dofadr[bid]                                                         <L 1615>
        var_5 = wp::address(var_body_dofadr, var_2);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // bdofnum = body_dofnum[bid]                                                         <L 1616>
        var_8 = wp::address(var_body_dofnum, var_2);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // for k_rev in range(bdofnum):                                                       <L 1618>
        var_11 = wp::range(var_9);
        start_for_2:;
            if (iter_cmp(var_11) == 0) goto end_for_2;
            var_12 = wp::iter_next(var_11);
            // dof = bdofadr + bdofnum - 1 - k_rev                                            <L 1619>
            var_13 = wp::add(var_6, var_9);
            var_15 = wp::sub(var_13, var_14);
            var_16 = wp::sub(var_15, var_12);
            // while ptr >= 0:                                                                <L 1621>
    start_while_4:;
            var_18 = (var_1 >= var_17);
    if ((var_18) == false) goto end_while_4;
                // sparseid = rowadr + ptr                                                    <L 1622>
                var_19 = wp::add(var_rowadr, var_1);
                // if ten_J_colind[sparseid] <= dof:                                          <L 1623>
                var_20 = wp::address(var_ten_J_colind, var_19);
                var_22 = wp::load(var_20);
                var_21 = (var_22 <= var_16);
                if (var_21) {
                    // break                                                                  <L 1624>
                    goto end_while_4;
                }
                // ptr -= 1                                                                   <L 1625>
                var_24 = wp::sub(var_1, var_23);
                wp::assign(var_1, var_24);
    goto start_while_4;
    end_while_4:;
            // if ptr >= 0 and ten_J_colind[sparseid] == dof:                                 <L 1626>
            var_26 = (var_1 >= var_25);
            var_27 = wp::address(var_ten_J_colind, var_19);
            var_29 = wp::load(var_27);
            var_28 = (var_29 == var_16);
            var_30 = var_26 && var_28;
            if (var_30) {
                // cdof = cdof_in[worldid, dof]                                               <L 1627>
                var_31 = wp::address(var_cdof_in, var_worldid, var_16);
                var_33 = wp::load(var_31);
                var_32 = wp::copy(var_33);
                // cdof_ang = wp.spatial_top(cdof)                                            <L 1628>
                var_34 = wp::spatial_top(var_32);
                // cdof_lin = wp.spatial_bottom(cdof)                                         <L 1629>
                var_35 = wp::spatial_bottom(var_32);
                // cdof_dot = cdof_dot_in[worldid, dof]                                       <L 1630>
                var_36 = wp::address(var_cdof_dot_in, var_worldid, var_16);
                var_38 = wp::load(var_36);
                var_37 = wp::copy(var_38);
                // dofjntid = dof_jntid[dof]                                                  <L 1633>
                var_39 = wp::address(var_dof_jntid, var_16);
                var_41 = wp::load(var_39);
                var_40 = wp::copy(var_41);
                // jnttype = jnt_type[dofjntid]                                               <L 1634>
                var_42 = wp::address(var_jnt_type, var_40);
                var_44 = wp::load(var_42);
                var_43 = wp::copy(var_44);
                // jntdofadr = jnt_dofadr[dofjntid]                                           <L 1635>
                var_45 = wp::address(var_jnt_dofadr, var_40);
                var_47 = wp::load(var_45);
                var_46 = wp::copy(var_47);
                // if (jnttype == JointType.BALL) or ((jnttype == JointType.FREE) and dof >= jntdofadr + 3):       <L 1636>
                var_49 = (var_43 == var_48);
                var_51 = (var_43 == var_50);
                var_53 = wp::add(var_46, var_52);
                var_54 = (var_16 >= var_53);
                var_55 = var_51 && var_54;
                var_56 = var_49 || var_55;
                if (var_56) {
                    // cdof_dot = math.motion_cross(cvel_in[worldid, bid], cdof)              <L 1637>
                    var_57 = wp::address(var_cvel_in, var_worldid, var_2);
                    var_59 = wp::load(var_57);
                    var_58 = motion_cross_0(var_59, var_32);
                }
                var_60 = wp::where(var_56, var_58, var_37);
                // cdof_dot_ang = wp.spatial_top(cdof_dot)                                    <L 1639>
                var_61 = wp::spatial_top(var_60);
                // cdof_dot_lin = wp.spatial_bottom(cdof_dot)                                 <L 1640>
                var_62 = wp::spatial_bottom(var_60);
                // jacp_dot = cdof_dot_lin + wp.cross(cdof_dot_ang, offset) + wp.cross(cdof_ang, pvel_lin)       <L 1643>
                var_63 = wp::cross(var_61, var_offset);
                var_64 = wp::add(var_62, var_63);
                var_65 = wp::cross(var_34, var_pvel_lin);
                var_66 = wp::add(var_64, var_65);
                // jacp = cdof_lin + wp.cross(cdof_ang, offset)                               <L 1646>
                var_67 = wp::cross(var_34, var_offset);
                var_68 = wp::add(var_35, var_67);
                // Jdot = (wp.dot(jacp_dot, dpnt) + wp.dot(jacp, dvel)) * scale               <L 1649>
                var_69 = wp::dot(var_66, var_dpnt);
                var_70 = wp::dot(var_68, var_dvel);
                var_71 = wp::add(var_69, var_70);
                var_72 = wp::mul(var_71, var_scale);
                // if Jdot != 0.0:                                                            <L 1650>
                var_74 = (var_72 != var_73);
                if (var_74) {
                    // wp.atomic_add(ten_Jdot_out[worldid], sparseid, Jdot)                   <L 1651>
                    var_75 = wp::slice_t(var_worldid, var_worldid, var_76);
                    var_77 = wp::view(var_ten_Jdot_out, var_75);
                    var_78 = wp::atomic_add(var_77, var_19, var_72);
                }
            }
            goto start_for_2;
        end_for_2:;
        // bid = body_parentid[bid]                                                           <L 1652>
        var_79 = wp::address(var_body_parentid, var_2);
        var_81 = wp::load(var_79);
        var_80 = wp::copy(var_81);
        wp::assign(var_2, var_80);
    goto start_while_0;
    end_while_0:;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:44
static CUDA_CALLABLE void adj_rot_vec_quat_0(
    wp::vec_t<3, wp::float32> var_vec,
    wp::quat_t<wp::float32> var_quat,
    wp::vec_t<3, wp::float32> & adj_vec,
    wp::quat_t<wp::float32> & adj_quat,
    wp::vec_t<3, wp::float32> & adj_ret)
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:59
static CUDA_CALLABLE void adj_quat_to_mat_0(
    wp::quat_t<wp::float32> var_quat,
    wp::quat_t<wp::float32> & adj_quat,
    wp::mat_t<3, 3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:52
static CUDA_CALLABLE void adj_axis_angle_to_quat_0(
    wp::vec_t<3, wp::float32> var_axis,
    wp::float32 var_angle,
    wp::vec_t<3, wp::float32> & adj_axis,
    wp::float32 & adj_angle,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:120
static CUDA_CALLABLE void adj_inert_vec_0(
    wp::vec_t<10, wp::float32> var_i,
    wp::vec_t<6, wp::float32> var_v,
    wp::vec_t<10, wp::float32> & adj_i,
    wp::vec_t<6, wp::float32> & adj_v,
    wp::vec_t<6, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:147
static CUDA_CALLABLE void adj_motion_cross_force_0(
    wp::vec_t<6, wp::float32> var_v,
    wp::vec_t<6, wp::float32> var_f,
    wp::vec_t<6, wp::float32> & adj_v,
    wp::vec_t<6, wp::float32> & adj_f,
    wp::vec_t<6, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:133
static CUDA_CALLABLE void adj_motion_cross_0(
    wp::vec_t<6, wp::float32> var_u,
    wp::vec_t<6, wp::float32> var_v,
    wp::vec_t<6, wp::float32> & adj_u,
    wp::vec_t<6, wp::float32> & adj_v,
    wp::vec_t<6, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/support.py:384
static CUDA_CALLABLE void adj_transform_force_0(
    wp::vec_t<3, wp::float32> var_force,
    wp::vec_t<3, wp::float32> var_torque,
    wp::vec_t<3, wp::float32> var_offset,
    wp::vec_t<3, wp::float32> & adj_force,
    wp::vec_t<3, wp::float32> & adj_torque,
    wp::vec_t<3, wp::float32> & adj_offset,
    wp::vec_t<6, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/support.py:389
static CUDA_CALLABLE void adj_transform_force_1(
    wp::vec_t<6, wp::float32> var_frc,
    wp::vec_t<3, wp::float32> var_offset,
    wp::vec_t<6, wp::float32> & adj_frc,
    wp::vec_t<3, wp::float32> & adj_offset,
    wp::vec_t<6, wp::float32> & adj_ret)
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/smooth.py:3126
static CUDA_CALLABLE void adj__accumulate_jac_chain_0(
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::vec_t<3, wp::float32> var_offset,
    wp::vec_t<3, wp::float32> var_vec,
    wp::int32 var_bodyid,
    wp::int32 var_rowadr,
    wp::int32 var_rownnz,
    wp::float32 var_scale,
    wp::int32 var_worldid,
    wp::array_t<wp::float32> var_ten_J_out,
    wp::array_t<wp::int32> & adj_body_parentid,
    wp::array_t<wp::int32> & adj_body_dofnum,
    wp::array_t<wp::int32> & adj_body_dofadr,
    wp::array_t<wp::int32> & adj_ten_J_colind,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cdof_in,
    wp::vec_t<3, wp::float32> & adj_offset,
    wp::vec_t<3, wp::float32> & adj_vec,
    wp::int32 & adj_bodyid,
    wp::int32 & adj_rowadr,
    wp::int32 & adj_rownnz,
    wp::float32 & adj_scale,
    wp::int32 & adj_worldid,
    wp::array_t<wp::float32> & adj_ten_J_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/warp/_src/math.py:0
static CUDA_CALLABLE void adj_norm_l2_0(
    wp::vec_t<3, wp::float32> var_v,
    wp::vec_t<3, wp::float32> & adj_v,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:0
static CUDA_CALLABLE void adj_normalize_with_norm_0(
    wp::vec_t<2, wp::float32> var_x,
    wp::vec_t<2, wp::float32> & ret_0,
    wp::float32 & ret_1,
    wp::vec_t<2, wp::float32> & adj_x,
    wp::vec_t<2, wp::float32> & adj_ret_0,
    wp::float32 & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/warp/_src/math.py:0
static CUDA_CALLABLE void adj_norm_l2_0(
    wp::vec_t<2, wp::float32> var_v,
    wp::vec_t<2, wp::float32> & adj_v,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/util_misc.py:201
static CUDA_CALLABLE void adj_wrap_inside_0(
    wp::vec_t<4, wp::float32> var_end,
    wp::float32 var_radius,
    wp::int32 var_maxiter,
    wp::float32 var_zinit,
    wp::float32 var_tolerance,
    wp::float32 & ret_0,
    wp::vec_t<2, wp::float32> & ret_1,
    wp::vec_t<2, wp::float32> & ret_2,
    wp::vec_t<4, wp::float32> & adj_end,
    wp::float32 & adj_radius,
    wp::int32 & adj_maxiter,
    wp::float32 & adj_zinit,
    wp::float32 & adj_tolerance,
    wp::float32 & adj_ret_0,
    wp::vec_t<2, wp::float32> & adj_ret_1,
    wp::vec_t<2, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/util_misc.py:30
static CUDA_CALLABLE void adj_is_intersect_0(
    wp::vec_t<2, wp::float32> var_p1,
    wp::vec_t<2, wp::float32> var_p2,
    wp::vec_t<2, wp::float32> var_p3,
    wp::vec_t<2, wp::float32> var_p4,
    wp::vec_t<2, wp::float32> & adj_p1,
    wp::vec_t<2, wp::float32> & adj_p2,
    wp::vec_t<2, wp::float32> & adj_p3,
    wp::vec_t<2, wp::float32> & adj_p4,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/util_misc.py:76
static CUDA_CALLABLE void adj_length_circle_0(
    wp::vec_t<2, wp::float32> var_p0,
    wp::vec_t<2, wp::float32> var_p1,
    wp::int32 var_ind,
    wp::float32 var_radius,
    wp::vec_t<2, wp::float32> & adj_p0,
    wp::vec_t<2, wp::float32> & adj_p1,
    wp::int32 & adj_ind,
    wp::float32 & adj_radius,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/util_misc.py:103
static CUDA_CALLABLE void adj_wrap_circle_0(
    wp::vec_t<4, wp::float32> var_end,
    wp::vec_t<2, wp::float32> var_side,
    wp::float32 var_radius,
    wp::float32 & ret_0,
    wp::vec_t<2, wp::float32> & ret_1,
    wp::vec_t<2, wp::float32> & ret_2,
    wp::vec_t<4, wp::float32> & adj_end,
    wp::vec_t<2, wp::float32> & adj_side,
    wp::float32 & adj_radius,
    wp::float32 & adj_ret_0,
    wp::vec_t<2, wp::float32> & adj_ret_1,
    wp::vec_t<2, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/util_misc.py:326
static CUDA_CALLABLE void adj_wrap_0(
    wp::vec_t<3, wp::float32> var_x0,
    wp::vec_t<3, wp::float32> var_x1,
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::float32 var_radius,
    wp::int32 var_geomtype,
    wp::vec_t<3, wp::float32> var_side,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_x0,
    wp::vec_t<3, wp::float32> & adj_x1,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::float32 & adj_radius,
    wp::int32 & adj_geomtype,
    wp::vec_t<3, wp::float32> & adj_side,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:161
static CUDA_CALLABLE void adj_quat_to_vel_0(
    wp::quat_t<wp::float32> var_quat,
    wp::quat_t<wp::float32> & adj_quat,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:0
static CUDA_CALLABLE void adj_safe_div_0(
    wp::vec_t<3, wp::float32> var_x,
    wp::float32 var_y,
    wp::vec_t<3, wp::float32> & adj_x,
    wp::float32 & adj_y,
    wp::vec_t<3, wp::float32> & adj_ret)
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/smooth.py:1584
static CUDA_CALLABLE void adj__accumulate_jac_dot_chain_0(
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::int32> var_dof_jntid,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_dot_in,
    wp::vec_t<3, wp::float32> var_offset,
    wp::vec_t<3, wp::float32> var_pvel_lin,
    wp::vec_t<3, wp::float32> var_dpnt,
    wp::vec_t<3, wp::float32> var_dvel,
    wp::int32 var_bodyid,
    wp::int32 var_rowadr,
    wp::int32 var_rownnz,
    wp::float32 var_scale,
    wp::int32 var_worldid,
    wp::array_t<wp::float32> var_ten_Jdot_out,
    wp::array_t<wp::int32> & adj_body_parentid,
    wp::array_t<wp::int32> & adj_body_dofnum,
    wp::array_t<wp::int32> & adj_body_dofadr,
    wp::array_t<wp::int32> & adj_jnt_type,
    wp::array_t<wp::int32> & adj_jnt_dofadr,
    wp::array_t<wp::int32> & adj_dof_jntid,
    wp::array_t<wp::int32> & adj_ten_J_colind,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cdof_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cdof_dot_in,
    wp::vec_t<3, wp::float32> & adj_offset,
    wp::vec_t<3, wp::float32> & adj_pvel_lin,
    wp::vec_t<3, wp::float32> & adj_dpnt,
    wp::vec_t<3, wp::float32> & adj_dvel,
    wp::int32 & adj_bodyid,
    wp::int32 & adj_rowadr,
    wp::int32 & adj_rownnz,
    wp::float32 & adj_scale,
    wp::int32 & adj_worldid,
    wp::array_t<wp::float32> & adj_ten_Jdot_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void _site_local_to_global_084f92a4_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_pos,
    wp::array_t<wp::quat_t<wp::float32>> var_site_quat,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_out)
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
        wp::vec_t<3, wp::float32>* var_5;
        wp::vec_t<3, wp::float32> var_6;
        wp::vec_t<3, wp::float32> var_7;
        wp::quat_t<wp::float32>* var_8;
        wp::quat_t<wp::float32> var_9;
        wp::quat_t<wp::float32> var_10;
        wp::shape_t* var_11;
        const wp::int32 var_12 = 0;
        wp::int32 var_13;
        wp::shape_t var_14;
        wp::int32 var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::shape_t* var_20;
        const wp::int32 var_21 = 0;
        wp::int32 var_22;
        wp::shape_t var_23;
        wp::int32 var_24;
        wp::quat_t<wp::float32>* var_25;
        wp::quat_t<wp::float32> var_26;
        wp::quat_t<wp::float32> var_27;
        wp::mat_t<3, 3, wp::float32> var_28;
        //---------
        // forward
        // def _site_local_to_global(                                                             <L 207>
        // worldid, siteid = wp.tid()                                                             <L 219>
        builtin_tid2d(var_0, var_1);
        // bodyid = site_bodyid[siteid]                                                           <L 220>
        var_2 = wp::address(var_site_bodyid, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // xpos = xpos_in[worldid, bodyid]                                                        <L 221>
        var_5 = wp::address(var_xpos_in, var_0, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // xquat = xquat_in[worldid, bodyid]                                                      <L 222>
        var_8 = wp::address(var_xquat_in, var_0, var_3);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // site_xpos_out[worldid, siteid] = xpos + math.rot_vec_quat(site_pos[worldid % site_pos.shape[0], siteid], xquat)       <L 223>
        var_11 = &(var_site_pos.shape);
        var_14 = wp::load(var_11);
        var_13 = wp::extract(var_14, var_12);
        var_15 = wp::mod(var_0, var_13);
        var_16 = wp::address(var_site_pos, var_15, var_1);
        var_18 = wp::load(var_16);
        var_17 = rot_vec_quat_0(var_18, var_9);
        var_19 = wp::add(var_6, var_17);
        wp::array_store(var_site_xpos_out, var_0, var_1, var_19);
        // site_xmat_out[worldid, siteid] = math.quat_to_mat(math.mul_quat(xquat, site_quat[worldid % site_quat.shape[0], siteid]))       <L 224>
        var_20 = &(var_site_quat.shape);
        var_23 = wp::load(var_20);
        var_22 = wp::extract(var_23, var_21);
        var_24 = wp::mod(var_0, var_22);
        var_25 = wp::address(var_site_quat, var_24, var_1);
        var_27 = wp::load(var_25);
        var_26 = mul_quat_0(var_9, var_27);
        var_28 = quat_to_mat_0(var_26);
        wp::array_store(var_site_xmat_out, var_0, var_1, var_28);
    }
}



extern "C" __global__ void _cacc_world_6ec26fdb_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_gravity,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cacc_out)
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
        const wp::float32 var_1 = 0.0;
        wp::vec_t<3, wp::float32> var_2;
        wp::shape_t* var_3;
        const wp::int32 var_4 = 0;
        wp::int32 var_5;
        wp::shape_t var_6;
        wp::int32 var_7;
        wp::vec_t<3, wp::float32>* var_8;
        wp::vec_t<3, wp::float32> var_9;
        wp::vec_t<3, wp::float32> var_10;
        wp::vec_t<6, wp::float32> var_11;
        const wp::int32 var_12 = 0;
        //---------
        // forward
        // def _cacc_world(                                                                       <L 1113>
        // worldid = wp.tid()                                                                     <L 1119>
        var_0 = builtin_tid1d();
        // cacc_out[worldid, 0] = wp.spatial_vector(wp.vec3(0.0), -gravity[worldid % gravity.shape[0]])       <L 1120>
        var_2 = wp::vec_t<3, wp::float32>(var_1);
        var_3 = &(var_gravity.shape);
        var_6 = wp::load(var_3);
        var_5 = wp::extract(var_6, var_4);
        var_7 = wp::mod(var_0, var_5);
        var_8 = wp::address(var_gravity, var_7);
        var_10 = wp::load(var_8);
        var_9 = wp::neg(var_10);
        var_11 = wp::vec_t<6, wp::float32>(var_2, var_9);
        wp::array_store(var_cacc_out, var_0, var_12, var_11);
    }
}



extern "C" __global__ void _tendon_bias_qfrc_2ae21571_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::float32> var_tendon_armature,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::array_t<wp::float32> var_ten_bias_coef_in,
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
        wp::int32 var_2;
        wp::shape_t* var_3;
        const wp::int32 var_4 = 0;
        wp::int32 var_5;
        wp::shape_t var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        const wp::float32 var_11 = 0.0;
        bool var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        bool var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::float32* var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        const wp::float32 var_24 = 0.0;
        bool var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::slice_t var_29;
        const wp::int32 var_30 = 0;
        wp::array_t<wp::float32> var_31;
        wp::float32 var_32;
        wp::float32* var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        //---------
        // forward
        // def _tendon_bias_qfrc(                                                                 <L 1843>
        // worldid, tenid, dofid = wp.tid()                                                       <L 1856>
        builtin_tid3d(var_0, var_1, var_2);
        // armature = tendon_armature[worldid % tendon_armature.shape[0], tenid]                  <L 1858>
        var_3 = &(var_tendon_armature.shape);
        var_6 = wp::load(var_3);
        var_5 = wp::extract(var_6, var_4);
        var_7 = wp::mod(var_0, var_5);
        var_8 = wp::address(var_tendon_armature, var_7, var_1);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // if armature == 0.0:                                                                    <L 1859>
        var_12 = (var_9 == var_11);
        if (var_12) {
            // return                                                                             <L 1860>
            continue;
        }
        // rownnz = ten_J_rownnz[tenid]                                                           <L 1862>
        var_13 = wp::address(var_ten_J_rownnz, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // if dofid >= rownnz:                                                                    <L 1863>
        var_16 = (var_2 >= var_14);
        if (var_16) {
            // return                                                                             <L 1864>
            continue;
        }
        // rowadr = ten_J_rowadr[tenid]                                                           <L 1865>
        var_17 = wp::address(var_ten_J_rowadr, var_1);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // sparseid = rowadr + dofid                                                              <L 1866>
        var_20 = wp::add(var_18, var_2);
        // ten_J = ten_J_in[worldid, sparseid]                                                    <L 1867>
        var_21 = wp::address(var_ten_J_in, var_0, var_20);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // if ten_J == 0.0:                                                                       <L 1869>
        var_25 = (var_22 == var_24);
        if (var_25) {
            // return                                                                             <L 1870>
            continue;
        }
        // dofid = ten_J_colind[sparseid]                                                         <L 1872>
        var_26 = wp::address(var_ten_J_colind, var_20);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // wp.atomic_add(qfrc_out[worldid], dofid, ten_J * armature * ten_bias_coef_in[worldid, tenid])       <L 1874>
        var_29 = wp::slice_t(var_0, var_0, var_30);
        var_31 = wp::view(var_qfrc_out, var_29);
        var_32 = wp::mul(var_22, var_9);
        var_33 = wp::address(var_ten_bias_coef_in, var_0, var_1);
        var_35 = wp::load(var_33);
        var_34 = wp::mul(var_32, var_35);
        var_36 = wp::atomic_add(var_31, var_27, var_34);
    }
}



extern "C" __global__ void _cacc_branch_ffe1d6c4_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_body_branches,
    wp::array_t<wp::int32> var_body_branch_start,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::float32> var_qacc_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_dot_in,
    bool var_flg_acc,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cacc_out)
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
        const wp::int32 var_5 = 1;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::vec_t<6, wp::float32>* var_16;
        wp::vec_t<6, wp::float32> var_17;
        wp::vec_t<6, wp::float32> var_18;
        wp::range_t var_19;
        wp::int32 var_20;
        wp::int32* var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        wp::int32* var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::range_t var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        wp::vec_t<6, wp::float32>* var_33;
        wp::int32 var_34;
        wp::float32* var_35;
        wp::vec_t<6, wp::float32> var_36;
        wp::vec_t<6, wp::float32> var_37;
        wp::float32 var_38;
        wp::vec_t<6, wp::float32> var_39;
        wp::int32 var_40;
        wp::vec_t<6, wp::float32>* var_41;
        wp::int32 var_42;
        wp::float32* var_43;
        wp::vec_t<6, wp::float32> var_44;
        wp::vec_t<6, wp::float32> var_45;
        wp::float32 var_46;
        wp::vec_t<6, wp::float32> var_47;
        wp::vec_t<6, wp::float32> var_48;
        //---------
        // forward
        // def _cacc_branch(                                                                      <L 1131>
        // worldid, branchid = wp.tid()                                                           <L 1148>
        builtin_tid2d(var_0, var_1);
        // start = body_branch_start[branchid]                                                    <L 1150>
        var_2 = wp::address(var_body_branch_start, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // end = body_branch_start[branchid + 1]                                                  <L 1151>
        var_6 = wp::add(var_1, var_5);
        var_7 = wp::address(var_body_branch_start, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // bodyid = body_branches[start]                                                          <L 1153>
        var_10 = wp::address(var_body_branches, var_3);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // pid = body_parentid[bodyid]                                                            <L 1154>
        var_13 = wp::address(var_body_parentid, var_11);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // local_cacc = cacc_out[worldid, pid]                                                    <L 1155>
        var_16 = wp::address(var_cacc_out, var_0, var_14);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // for i in range(start, end):                                                            <L 1156>
        var_19 = wp::range(var_3, var_8);
        start_for_0:;
            if (iter_cmp(var_19) == 0) goto end_for_0;
            var_20 = wp::iter_next(var_19);
            // bodyid = body_branches[i]                                                          <L 1157>
            var_21 = wp::address(var_body_branches, var_20);
            var_23 = wp::load(var_21);
            var_22 = wp::copy(var_23);
            // dofnum = body_dofnum[bodyid]                                                       <L 1158>
            var_24 = wp::address(var_body_dofnum, var_22);
            var_26 = wp::load(var_24);
            var_25 = wp::copy(var_26);
            // dofadr = body_dofadr[bodyid]                                                       <L 1159>
            var_27 = wp::address(var_body_dofadr, var_22);
            var_29 = wp::load(var_27);
            var_28 = wp::copy(var_29);
            // for j in range(dofnum):                                                            <L 1160>
            var_30 = wp::range(var_25);
            start_for_2:;
                if (iter_cmp(var_30) == 0) goto end_for_2;
                var_31 = wp::iter_next(var_30);
                // local_cacc += cdof_dot_in[worldid, dofadr + j] * qvel_in[worldid, dofadr + j]       <L 1161>
                var_32 = wp::add(var_28, var_31);
                var_33 = wp::address(var_cdof_dot_in, var_0, var_32);
                var_34 = wp::add(var_28, var_31);
                var_35 = wp::address(var_qvel_in, var_0, var_34);
                var_37 = wp::load(var_33);
                var_38 = wp::load(var_35);
                var_36 = wp::mul(var_37, var_38);
                var_39 = wp::add(var_17, var_36);
                // if flg_acc:                                                                    <L 1162>
                if (var_flg_acc) {
                    // local_cacc += cdof_in[worldid, dofadr + j] * qacc_in[worldid, dofadr + j]       <L 1163>
                    var_40 = wp::add(var_28, var_31);
                    var_41 = wp::address(var_cdof_in, var_0, var_40);
                    var_42 = wp::add(var_28, var_31);
                    var_43 = wp::address(var_qacc_in, var_0, var_42);
                    var_45 = wp::load(var_41);
                    var_46 = wp::load(var_43);
                    var_44 = wp::mul(var_45, var_46);
                    var_47 = wp::add(var_39, var_44);
                }
                var_48 = wp::where(var_flg_acc, var_47, var_39);
                wp::assign(var_17, var_48);
                goto start_for_2;
            end_for_2:;
            // cacc_out[worldid, bodyid] = local_cacc                                             <L 1164>
            wp::array_store(var_cacc_out, var_0, var_22, var_17);
            wp::assign(var_11, var_22);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void _kinematics_branch_95f8028e_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_qpos0,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_mocapid,
    wp::array_t<wp::int32> var_body_jntnum,
    wp::array_t<wp::int32> var_body_jntadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_pos,
    wp::array_t<wp::quat_t<wp::float32>> var_body_quat,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_jnt_qposadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_jnt_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> var_jnt_axis,
    wp::array_t<wp::int32> var_body_branches,
    wp::array_t<wp::int32> var_body_branch_start,
    wp::array_t<wp::float32> var_qpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mocap_pos_in,
    wp::array_t<wp::quat_t<wp::float32>> var_mocap_quat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_out,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xanchor_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xaxis_out)
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
        const wp::int32 var_5 = 1;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::slice_t var_10;
        const wp::int32 var_11 = 0;
        wp::array_t<wp::float32> var_12;
        wp::range_t var_13;
        wp::int32 var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32* var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        wp::int32* var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        const wp::int32 var_27 = 1;
        bool var_28;
        wp::int32* var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        const wp::int32 var_32 = 0;
        bool var_33;
        wp::int32* var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        wp::float32* var_37;
        const wp::int32 var_38 = 1;
        wp::int32 var_39;
        wp::float32* var_40;
        const wp::int32 var_41 = 2;
        wp::int32 var_42;
        wp::float32* var_43;
        wp::vec_t<3, wp::float32> var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::float32 var_47;
        const wp::int32 var_48 = 3;
        wp::int32 var_49;
        wp::float32* var_50;
        const wp::int32 var_51 = 4;
        wp::int32 var_52;
        wp::float32* var_53;
        const wp::int32 var_54 = 5;
        wp::int32 var_55;
        wp::float32* var_56;
        const wp::int32 var_57 = 6;
        wp::int32 var_58;
        wp::float32* var_59;
        wp::quat_t<wp::float32> var_60;
        wp::float32 var_61;
        wp::float32 var_62;
        wp::float32 var_63;
        wp::float32 var_64;
        wp::quat_t<wp::float32> var_65;
        wp::shape_t* var_66;
        const wp::int32 var_67 = 0;
        wp::int32 var_68;
        wp::shape_t var_69;
        wp::int32 var_70;
        wp::vec_t<3, wp::float32>* var_71;
        wp::vec_t<3, wp::float32> var_72;
        wp::shape_t* var_73;
        const wp::int32 var_74 = 0;
        wp::int32 var_75;
        wp::shape_t var_76;
        wp::int32 var_77;
        wp::int32* var_78;
        wp::int32 var_79;
        wp::int32 var_80;
        wp::int32* var_81;
        wp::int32 var_82;
        wp::int32 var_83;
        const wp::int32 var_84 = 0;
        bool var_85;
        wp::vec_t<3, wp::float32>* var_86;
        wp::vec_t<3, wp::float32> var_87;
        wp::vec_t<3, wp::float32> var_88;
        wp::quat_t<wp::float32>* var_89;
        wp::quat_t<wp::float32> var_90;
        wp::quat_t<wp::float32> var_91;
        wp::vec_t<3, wp::float32> var_92;
        wp::quat_t<wp::float32> var_93;
        wp::shape_t* var_94;
        const wp::int32 var_95 = 0;
        wp::int32 var_96;
        wp::shape_t var_97;
        wp::int32 var_98;
        wp::vec_t<3, wp::float32>* var_99;
        wp::vec_t<3, wp::float32> var_100;
        wp::vec_t<3, wp::float32> var_101;
        wp::shape_t* var_102;
        const wp::int32 var_103 = 0;
        wp::int32 var_104;
        wp::shape_t var_105;
        wp::int32 var_106;
        wp::quat_t<wp::float32>* var_107;
        wp::quat_t<wp::float32> var_108;
        wp::quat_t<wp::float32> var_109;
        wp::vec_t<3, wp::float32> var_110;
        wp::quat_t<wp::float32> var_111;
        const wp::int32 var_112 = 0;
        bool var_113;
        wp::quat_t<wp::float32>* var_114;
        wp::vec_t<3, wp::float32> var_115;
        wp::quat_t<wp::float32> var_116;
        wp::vec_t<3, wp::float32>* var_117;
        wp::vec_t<3, wp::float32> var_118;
        wp::vec_t<3, wp::float32> var_119;
        wp::quat_t<wp::float32>* var_120;
        wp::quat_t<wp::float32> var_121;
        wp::quat_t<wp::float32> var_122;
        wp::vec_t<3, wp::float32> var_123;
        wp::quat_t<wp::float32> var_124;
        wp::range_t var_125;
        wp::int32 var_126;
        wp::int32* var_127;
        wp::int32 var_128;
        wp::int32 var_129;
        wp::int32* var_130;
        wp::int32 var_131;
        wp::int32 var_132;
        wp::shape_t* var_133;
        const wp::int32 var_134 = 0;
        wp::int32 var_135;
        wp::shape_t var_136;
        wp::int32 var_137;
        wp::vec_t<3, wp::float32>* var_138;
        wp::vec_t<3, wp::float32> var_139;
        wp::vec_t<3, wp::float32> var_140;
        wp::vec_t<3, wp::float32>* var_141;
        wp::vec_t<3, wp::float32> var_142;
        wp::vec_t<3, wp::float32> var_143;
        wp::vec_t<3, wp::float32> var_144;
        wp::vec_t<3, wp::float32> var_145;
        const wp::int32 var_146 = 1;
        bool var_147;
        const wp::int32 var_148 = 0;
        wp::int32 var_149;
        wp::float32* var_150;
        const wp::int32 var_151 = 1;
        wp::int32 var_152;
        wp::float32* var_153;
        const wp::int32 var_154 = 2;
        wp::int32 var_155;
        wp::float32* var_156;
        const wp::int32 var_157 = 3;
        wp::int32 var_158;
        wp::float32* var_159;
        wp::quat_t<wp::float32> var_160;
        wp::float32 var_161;
        wp::float32 var_162;
        wp::float32 var_163;
        wp::float32 var_164;
        wp::quat_t<wp::float32> var_165;
        wp::quat_t<wp::float32> var_166;
        wp::vec_t<3, wp::float32>* var_167;
        wp::vec_t<3, wp::float32> var_168;
        wp::vec_t<3, wp::float32> var_169;
        wp::vec_t<3, wp::float32> var_170;
        wp::vec_t<3, wp::float32> var_171;
        wp::quat_t<wp::float32> var_172;
        const wp::int32 var_173 = 2;
        bool var_174;
        wp::float32* var_175;
        wp::shape_t* var_176;
        const wp::int32 var_177 = 0;
        wp::int32 var_178;
        wp::shape_t var_179;
        wp::int32 var_180;
        wp::float32* var_181;
        wp::float32 var_182;
        wp::float32 var_183;
        wp::float32 var_184;
        wp::vec_t<3, wp::float32> var_185;
        wp::vec_t<3, wp::float32> var_186;
        wp::vec_t<3, wp::float32> var_187;
        const wp::int32 var_188 = 3;
        bool var_189;
        wp::shape_t* var_190;
        const wp::int32 var_191 = 0;
        wp::int32 var_192;
        wp::shape_t var_193;
        wp::int32 var_194;
        wp::float32* var_195;
        wp::float32 var_196;
        wp::float32 var_197;
        wp::float32* var_198;
        wp::float32 var_199;
        wp::float32 var_200;
        wp::quat_t<wp::float32> var_201;
        wp::quat_t<wp::float32> var_202;
        wp::vec_t<3, wp::float32>* var_203;
        wp::vec_t<3, wp::float32> var_204;
        wp::vec_t<3, wp::float32> var_205;
        wp::vec_t<3, wp::float32> var_206;
        wp::vec_t<3, wp::float32> var_207;
        wp::quat_t<wp::float32> var_208;
        wp::vec_t<3, wp::float32> var_209;
        wp::quat_t<wp::float32> var_210;
        wp::vec_t<3, wp::float32> var_211;
        wp::quat_t<wp::float32> var_212;
        const wp::int32 var_213 = 1;
        wp::int32 var_214;
        wp::quat_t<wp::float32> var_215;
        //---------
        // forward
        // def _kinematics_branch(                                                                <L 45>
        // worldid, branchid = wp.tid()                                                           <L 70>
        builtin_tid2d(var_0, var_1);
        // start = body_branch_start[branchid]                                                    <L 72>
        var_2 = wp::address(var_body_branch_start, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // end = body_branch_start[branchid + 1]                                                  <L 73>
        var_6 = wp::add(var_1, var_5);
        var_7 = wp::address(var_body_branch_start, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // qpos = qpos_in[worldid]                                                                <L 75>
        var_10 = wp::slice_t(var_0, var_0, var_11);
        var_12 = wp::view(var_qpos_in, var_10);
        // for i in range(start, end):                                                            <L 77>
        var_13 = wp::range(var_3, var_8);
        start_for_0:;
            if (iter_cmp(var_13) == 0) goto end_for_0;
            var_14 = wp::iter_next(var_13);
            // bodyid = body_branches[i]                                                          <L 78>
            var_15 = wp::address(var_body_branches, var_14);
            var_17 = wp::load(var_15);
            var_16 = wp::copy(var_17);
            // pid = body_parentid[bodyid]                                                        <L 79>
            var_18 = wp::address(var_body_parentid, var_16);
            var_20 = wp::load(var_18);
            var_19 = wp::copy(var_20);
            // jntadr = body_jntadr[bodyid]                                                       <L 80>
            var_21 = wp::address(var_body_jntadr, var_16);
            var_23 = wp::load(var_21);
            var_22 = wp::copy(var_23);
            // jntnum = body_jntnum[bodyid]                                                       <L 81>
            var_24 = wp::address(var_body_jntnum, var_16);
            var_26 = wp::load(var_24);
            var_25 = wp::copy(var_26);
            // if jntnum == 1:                                                                    <L 83>
            var_28 = (var_25 == var_27);
            if (var_28) {
                // jnt_type_ = jnt_type[jntadr]                                                   <L 84>
                var_29 = wp::address(var_jnt_type, var_22);
                var_31 = wp::load(var_29);
                var_30 = wp::copy(var_31);
                // if jnt_type_ == JointType.FREE:                                                <L 85>
                var_33 = (var_30 == var_32);
                if (var_33) {
                    // qadr = jnt_qposadr[jntadr]                                                 <L 86>
                    var_34 = wp::address(var_jnt_qposadr, var_22);
                    var_36 = wp::load(var_34);
                    var_35 = wp::copy(var_36);
                    // xpos = wp.vec3(qpos[qadr], qpos[qadr + 1], qpos[qadr + 2])                 <L 87>
                    var_37 = wp::address(var_12, var_35);
                    var_39 = wp::add(var_35, var_38);
                    var_40 = wp::address(var_12, var_39);
                    var_42 = wp::add(var_35, var_41);
                    var_43 = wp::address(var_12, var_42);
                    var_45 = wp::load(var_37);
                    var_46 = wp::load(var_40);
                    var_47 = wp::load(var_43);
                    var_44 = wp::vec_t<3, wp::float32>(var_45, var_46, var_47);
                    // xquat = wp.quat(qpos[qadr + 3], qpos[qadr + 4], qpos[qadr + 5], qpos[qadr + 6])       <L 88>
                    var_49 = wp::add(var_35, var_48);
                    var_50 = wp::address(var_12, var_49);
                    var_52 = wp::add(var_35, var_51);
                    var_53 = wp::address(var_12, var_52);
                    var_55 = wp::add(var_35, var_54);
                    var_56 = wp::address(var_12, var_55);
                    var_58 = wp::add(var_35, var_57);
                    var_59 = wp::address(var_12, var_58);
                    var_61 = wp::load(var_50);
                    var_62 = wp::load(var_53);
                    var_63 = wp::load(var_56);
                    var_64 = wp::load(var_59);
                    var_60 = wp::quat_t<wp::float32>(var_61, var_62, var_63, var_64);
                    // xquat = wp.normalize(xquat)                                                <L 89>
                    var_65 = wp::normalize(var_60);
                    // xpos_out[worldid, bodyid] = xpos                                           <L 91>
                    wp::array_store(var_xpos_out, var_0, var_16, var_44);
                    // xquat_out[worldid, bodyid] = xquat                                         <L 92>
                    wp::array_store(var_xquat_out, var_0, var_16, var_65);
                    // xanchor_out[worldid, jntadr] = xpos                                        <L 93>
                    wp::array_store(var_xanchor_out, var_0, var_22, var_44);
                    // xaxis_out[worldid, jntadr] = jnt_axis[worldid % jnt_axis.shape[0], jntadr]       <L 94>
                    var_66 = &(var_jnt_axis.shape);
                    var_69 = wp::load(var_66);
                    var_68 = wp::extract(var_69, var_67);
                    var_70 = wp::mod(var_0, var_68);
                    var_71 = wp::address(var_jnt_axis, var_70, var_22);
                    var_72 = wp::load(var_71);
                    wp::array_store(var_xaxis_out, var_0, var_22, var_72);
                    // continue                                                                   <L 95>
                    goto start_for_0;
                }
            }
            // jnt_pos_id = worldid % jnt_pos.shape[0]                                            <L 99>
            var_73 = &(var_jnt_pos.shape);
            var_76 = wp::load(var_73);
            var_75 = wp::extract(var_76, var_74);
            var_77 = wp::mod(var_0, var_75);
            // pid = body_parentid[bodyid]                                                        <L 100>
            var_78 = wp::address(var_body_parentid, var_16);
            var_80 = wp::load(var_78);
            var_79 = wp::copy(var_80);
            // mocapid = body_mocapid[bodyid]                                                     <L 103>
            var_81 = wp::address(var_body_mocapid, var_16);
            var_83 = wp::load(var_81);
            var_82 = wp::copy(var_83);
            // if mocapid >= 0:                                                                   <L 104>
            var_85 = (var_82 >= var_84);
            if (var_85) {
                // xpos = mocap_pos_in[worldid, mocapid]                                          <L 105>
                var_86 = wp::address(var_mocap_pos_in, var_0, var_82);
                var_88 = wp::load(var_86);
                var_87 = wp::copy(var_88);
                // xquat = mocap_quat_in[worldid, mocapid]                                        <L 106>
                var_89 = wp::address(var_mocap_quat_in, var_0, var_82);
                var_91 = wp::load(var_89);
                var_90 = wp::copy(var_91);
            }
            var_92 = wp::where(var_85, var_87, var_44);
            var_93 = wp::where(var_85, var_90, var_65);
            if (!var_85) {
                // xpos = body_pos[worldid % body_pos.shape[0], bodyid]                           <L 108>
                var_94 = &(var_body_pos.shape);
                var_97 = wp::load(var_94);
                var_96 = wp::extract(var_97, var_95);
                var_98 = wp::mod(var_0, var_96);
                var_99 = wp::address(var_body_pos, var_98, var_16);
                var_101 = wp::load(var_99);
                var_100 = wp::copy(var_101);
                // xquat = body_quat[worldid % body_quat.shape[0], bodyid]                        <L 109>
                var_102 = &(var_body_quat.shape);
                var_105 = wp::load(var_102);
                var_104 = wp::extract(var_105, var_103);
                var_106 = wp::mod(var_0, var_104);
                var_107 = wp::address(var_body_quat, var_106, var_16);
                var_109 = wp::load(var_107);
                var_108 = wp::copy(var_109);
            }
            var_110 = wp::where(var_85, var_92, var_100);
            var_111 = wp::where(var_85, var_93, var_108);
            // if pid >= 0:                                                                       <L 111>
            var_113 = (var_79 >= var_112);
            if (var_113) {
                // xpos = math.rot_vec_quat(xpos, xquat_out[worldid, pid]) + xpos_out[worldid, pid]       <L 112>
                var_114 = wp::address(var_xquat_out, var_0, var_79);
                var_116 = wp::load(var_114);
                var_115 = rot_vec_quat_0(var_110, var_116);
                var_117 = wp::address(var_xpos_out, var_0, var_79);
                var_119 = wp::load(var_117);
                var_118 = wp::add(var_115, var_119);
                // xquat = math.mul_quat(xquat_out[worldid, pid], xquat)                          <L 113>
                var_120 = wp::address(var_xquat_out, var_0, var_79);
                var_122 = wp::load(var_120);
                var_121 = mul_quat_0(var_122, var_111);
            }
            var_123 = wp::where(var_113, var_118, var_110);
            var_124 = wp::where(var_113, var_121, var_111);
            // for _ in range(jntnum):                                                            <L 115>
            var_125 = wp::range(var_25);
            start_for_2:;
                if (iter_cmp(var_125) == 0) goto end_for_2;
                var_126 = wp::iter_next(var_125);
                // qadr = jnt_qposadr[jntadr]                                                     <L 116>
                var_127 = wp::address(var_jnt_qposadr, var_22);
                var_129 = wp::load(var_127);
                var_128 = wp::copy(var_129);
                // jnt_type_ = jnt_type[jntadr]                                                   <L 117>
                var_130 = wp::address(var_jnt_type, var_22);
                var_132 = wp::load(var_130);
                var_131 = wp::copy(var_132);
                // jnt_axis_ = jnt_axis[worldid % jnt_axis.shape[0], jntadr]                      <L 118>
                var_133 = &(var_jnt_axis.shape);
                var_136 = wp::load(var_133);
                var_135 = wp::extract(var_136, var_134);
                var_137 = wp::mod(var_0, var_135);
                var_138 = wp::address(var_jnt_axis, var_137, var_22);
                var_140 = wp::load(var_138);
                var_139 = wp::copy(var_140);
                // xanchor = math.rot_vec_quat(jnt_pos[jnt_pos_id, jntadr], xquat) + xpos         <L 119>
                var_141 = wp::address(var_jnt_pos, var_77, var_22);
                var_143 = wp::load(var_141);
                var_142 = rot_vec_quat_0(var_143, var_124);
                var_144 = wp::add(var_142, var_123);
                // xaxis = math.rot_vec_quat(jnt_axis_, xquat)                                    <L 120>
                var_145 = rot_vec_quat_0(var_139, var_124);
                // if jnt_type_ == JointType.BALL:                                                <L 122>
                var_147 = (var_131 == var_146);
                if (var_147) {
                    // qloc = wp.quat(qpos[qadr + 0], qpos[qadr + 1], qpos[qadr + 2], qpos[qadr + 3])       <L 123>
                    var_149 = wp::add(var_128, var_148);
                    var_150 = wp::address(var_12, var_149);
                    var_152 = wp::add(var_128, var_151);
                    var_153 = wp::address(var_12, var_152);
                    var_155 = wp::add(var_128, var_154);
                    var_156 = wp::address(var_12, var_155);
                    var_158 = wp::add(var_128, var_157);
                    var_159 = wp::address(var_12, var_158);
                    var_161 = wp::load(var_150);
                    var_162 = wp::load(var_153);
                    var_163 = wp::load(var_156);
                    var_164 = wp::load(var_159);
                    var_160 = wp::quat_t<wp::float32>(var_161, var_162, var_163, var_164);
                    // qloc = wp.normalize(qloc)                                                  <L 124>
                    var_165 = wp::normalize(var_160);
                    // xquat = math.mul_quat(xquat, qloc)                                         <L 125>
                    var_166 = mul_quat_0(var_124, var_165);
                    // xpos = xanchor - math.rot_vec_quat(jnt_pos[jnt_pos_id, jntadr], xquat)       <L 127>
                    var_167 = wp::address(var_jnt_pos, var_77, var_22);
                    var_169 = wp::load(var_167);
                    var_168 = rot_vec_quat_0(var_169, var_166);
                    var_170 = wp::sub(var_144, var_168);
                }
                var_171 = wp::where(var_147, var_170, var_123);
                var_172 = wp::where(var_147, var_166, var_124);
                if (!var_147) {
                    // elif jnt_type_ == JointType.SLIDE:                                         <L 128>
                    var_174 = (var_131 == var_173);
                    if (var_174) {
                        // xpos += xaxis * (qpos[qadr] - qpos0[worldid % qpos0.shape[0], qadr])       <L 129>
                        var_175 = wp::address(var_12, var_128);
                        var_176 = &(var_qpos0.shape);
                        var_179 = wp::load(var_176);
                        var_178 = wp::extract(var_179, var_177);
                        var_180 = wp::mod(var_0, var_178);
                        var_181 = wp::address(var_qpos0, var_180, var_128);
                        var_183 = wp::load(var_175);
                        var_184 = wp::load(var_181);
                        var_182 = wp::sub(var_183, var_184);
                        var_185 = wp::mul(var_145, var_182);
                        var_186 = wp::add(var_171, var_185);
                    }
                    var_187 = wp::where(var_174, var_186, var_171);
                    if (!var_174) {
                        // elif jnt_type_ == JointType.HINGE:                                     <L 130>
                        var_189 = (var_131 == var_188);
                        if (var_189) {
                            // qpos0_ = qpos0[worldid % qpos0.shape[0], qadr]                     <L 131>
                            var_190 = &(var_qpos0.shape);
                            var_193 = wp::load(var_190);
                            var_192 = wp::extract(var_193, var_191);
                            var_194 = wp::mod(var_0, var_192);
                            var_195 = wp::address(var_qpos0, var_194, var_128);
                            var_197 = wp::load(var_195);
                            var_196 = wp::copy(var_197);
                            // qloc_ = math.axis_angle_to_quat(jnt_axis_, qpos[qadr] - qpos0_)       <L 132>
                            var_198 = wp::address(var_12, var_128);
                            var_200 = wp::load(var_198);
                            var_199 = wp::sub(var_200, var_196);
                            var_201 = axis_angle_to_quat_0(var_139, var_199);
                            // xquat = math.mul_quat(xquat, qloc_)                                <L 133>
                            var_202 = mul_quat_0(var_172, var_201);
                            // xpos = xanchor - math.rot_vec_quat(jnt_pos[jnt_pos_id, jntadr], xquat)       <L 135>
                            var_203 = wp::address(var_jnt_pos, var_77, var_22);
                            var_205 = wp::load(var_203);
                            var_204 = rot_vec_quat_0(var_205, var_202);
                            var_206 = wp::sub(var_144, var_204);
                        }
                        var_207 = wp::where(var_189, var_206, var_187);
                        var_208 = wp::where(var_189, var_202, var_172);
                    }
                    var_209 = wp::where(var_174, var_187, var_207);
                    var_210 = wp::where(var_174, var_172, var_208);
                }
                var_211 = wp::where(var_147, var_171, var_209);
                var_212 = wp::where(var_147, var_172, var_210);
                // xanchor_out[worldid, jntadr] = xanchor                                         <L 137>
                wp::array_store(var_xanchor_out, var_0, var_22, var_144);
                // xaxis_out[worldid, jntadr] = xaxis                                             <L 138>
                wp::array_store(var_xaxis_out, var_0, var_22, var_145);
                // jntadr += 1                                                                    <L 139>
                var_214 = wp::add(var_22, var_213);
                wp::assign(var_22, var_214);
                wp::assign(var_30, var_131);
                wp::assign(var_35, var_128);
                wp::assign(var_123, var_211);
                wp::assign(var_124, var_212);
                goto start_for_2;
            end_for_2:;
            // xquat = wp.normalize(xquat)                                                        <L 141>
            var_215 = wp::normalize(var_124);
            // xpos_out[worldid, bodyid] = xpos                                                   <L 142>
            wp::array_store(var_xpos_out, var_0, var_16, var_123);
            // xquat_out[worldid, bodyid] = xquat                                                 <L 143>
            wp::array_store(var_xquat_out, var_0, var_16, var_215);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void _compute_body_matrices_27760750_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_out)
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
        wp::quat_t<wp::float32>* var_2;
        wp::mat_t<3, 3, wp::float32> var_3;
        wp::quat_t<wp::float32> var_4;
        //---------
        // forward
        // def _compute_body_matrices(                                                            <L 166>
        // worldid, bodyid = wp.tid()                                                             <L 172>
        builtin_tid2d(var_0, var_1);
        // xmat_out[worldid, bodyid] = math.quat_to_mat(xquat_in[worldid, bodyid])                <L 173>
        var_2 = wp::address(var_xquat_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = quat_to_mat_0(var_4);
        wp::array_store(var_xmat_out, var_0, var_1, var_3);
    }
}



extern "C" __global__ void _cfrc_e3a9fd02_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<10, wp::float32>> var_cinert_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cacc_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_ext_in,
    bool var_flg_cfrc_ext,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_int_out)
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
        const wp::int32 var_2 = 1;
        wp::int32 var_3;
        wp::vec_t<6, wp::float32>* var_4;
        wp::vec_t<6, wp::float32> var_5;
        wp::vec_t<6, wp::float32> var_6;
        wp::vec_t<10, wp::float32>* var_7;
        wp::vec_t<10, wp::float32> var_8;
        wp::vec_t<10, wp::float32> var_9;
        wp::vec_t<6, wp::float32>* var_10;
        wp::vec_t<6, wp::float32> var_11;
        wp::vec_t<6, wp::float32> var_12;
        wp::vec_t<6, wp::float32> var_13;
        wp::vec_t<6, wp::float32> var_14;
        wp::vec_t<6, wp::float32> var_15;
        wp::vec_t<6, wp::float32> var_16;
        wp::vec_t<6, wp::float32>* var_17;
        wp::vec_t<6, wp::float32> var_18;
        wp::vec_t<6, wp::float32> var_19;
        wp::vec_t<6, wp::float32> var_20;
        //---------
        // forward
        // def _cfrc(                                                                             <L 1188>
        // worldid, bodyid = wp.tid()                                                             <L 1199>
        builtin_tid2d(var_0, var_1);
        // bodyid += 1  # skip world body                                                         <L 1200>
        var_3 = wp::add(var_1, var_2);
        // cacc = cacc_in[worldid, bodyid]                                                        <L 1201>
        var_4 = wp::address(var_cacc_in, var_0, var_3);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // cinert = cinert_in[worldid, bodyid]                                                    <L 1202>
        var_7 = wp::address(var_cinert_in, var_0, var_3);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // cvel = cvel_in[worldid, bodyid]                                                        <L 1203>
        var_10 = wp::address(var_cvel_in, var_0, var_3);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // frc = math.inert_vec(cinert, cacc)                                                     <L 1204>
        var_13 = inert_vec_0(var_8, var_5);
        // frc += math.motion_cross_force(cvel, math.inert_vec(cinert, cvel))                     <L 1205>
        var_14 = inert_vec_0(var_8, var_11);
        var_15 = motion_cross_force_0(var_11, var_14);
        var_16 = wp::add(var_13, var_15);
        // if flg_cfrc_ext:                                                                       <L 1206>
        if (var_flg_cfrc_ext) {
            // frc -= cfrc_ext_in[worldid, bodyid]                                                <L 1207>
            var_17 = wp::address(var_cfrc_ext_in, var_0, var_3);
            var_19 = wp::load(var_17);
            var_18 = wp::sub(var_16, var_19);
        }
        var_20 = wp::where(var_flg_cfrc_ext, var_18, var_16);
        // cfrc_int_out[worldid, bodyid] = frc                                                    <L 1209>
        wp::array_store(var_cfrc_int_out, var_0, var_3, var_20);
    }
}



extern "C" __global__ void _joint_tendon_35c34b14_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_jnt_qposadr,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::int32> var_wrap_objid,
    wp::array_t<wp::float32> var_wrap_prm,
    wp::array_t<wp::int32> var_tendon_jnt_adr,
    wp::array_t<wp::int32> var_wrap_jnt_adr,
    wp::array_t<wp::float32> var_qpos_in,
    wp::array_t<wp::float32> var_ten_J_out,
    wp::array_t<wp::float32> var_ten_length_out)
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
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::int32* var_14;
        wp::float32* var_15;
        wp::int32 var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::slice_t var_19;
        const wp::int32 var_20 = 0;
        wp::array_t<wp::float32> var_21;
        wp::float32 var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::int32* var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        wp::range_t var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        wp::int32* var_35;
        bool var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        //---------
        // forward
        // def _joint_tendon(                                                                     <L 3088>
        // worldid, wrapid = wp.tid()                                                             <L 3105>
        builtin_tid2d(var_0, var_1);
        // tenid = tendon_jnt_adr[wrapid]                                                         <L 3107>
        var_2 = wp::address(var_tendon_jnt_adr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // wrapjntid = wrap_jnt_adr[wrapid]                                                       <L 3108>
        var_5 = wp::address(var_wrap_jnt_adr, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // wrapobjid = wrap_objid[wrapjntid]                                                      <L 3109>
        var_8 = wp::address(var_wrap_objid, var_6);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // prm = wrap_prm[wrapjntid]                                                              <L 3110>
        var_11 = wp::address(var_wrap_prm, var_6);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // L = prm * qpos_in[worldid, jnt_qposadr[wrapobjid]]                                     <L 3113>
        var_14 = wp::address(var_jnt_qposadr, var_9);
        var_16 = wp::load(var_14);
        var_15 = wp::address(var_qpos_in, var_0, var_16);
        var_18 = wp::load(var_15);
        var_17 = wp::mul(var_12, var_18);
        // wp.atomic_add(ten_length_out[worldid], tenid, L)                                       <L 3114>
        var_19 = wp::slice_t(var_0, var_0, var_20);
        var_21 = wp::view(var_ten_length_out, var_19);
        var_22 = wp::atomic_add(var_21, var_3, var_17);
        // dofadr = jnt_dofadr[wrapobjid]                                                         <L 3117>
        var_23 = wp::address(var_jnt_dofadr, var_9);
        var_25 = wp::load(var_23);
        var_24 = wp::copy(var_25);
        // rowadr = ten_J_rowadr[tenid]                                                           <L 3118>
        var_26 = wp::address(var_ten_J_rowadr, var_3);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // rownnz = ten_J_rownnz[tenid]                                                           <L 3119>
        var_29 = wp::address(var_ten_J_rownnz, var_3);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // for k in range(rownnz):                                                                <L 3120>
        var_32 = wp::range(var_30);
        start_for_0:;
            if (iter_cmp(var_32) == 0) goto end_for_0;
            var_33 = wp::iter_next(var_32);
            // if ten_J_colind[rowadr + k] == dofadr:                                             <L 3121>
            var_34 = wp::add(var_27, var_33);
            var_35 = wp::address(var_ten_J_colind, var_34);
            var_37 = wp::load(var_35);
            var_36 = (var_37 == var_24);
            if (var_36) {
                // ten_J_out[worldid, rowadr + k] = prm                                           <L 3122>
                var_38 = wp::add(var_27, var_33);
                wp::array_store(var_ten_J_out, var_0, var_38, var_12);
                // break                                                                          <L 3123>
                goto end_for_0;
            }
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void _cfrc_backward_72237d38_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_int_in,
    wp::array_t<wp::int32> var_body_tree_,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_int_out)
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
        bool var_9;
        wp::slice_t var_10;
        const wp::int32 var_11 = 0;
        wp::array_t<wp::vec_t<6, wp::float32>> var_12;
        wp::vec_t<6, wp::float32>* var_13;
        wp::vec_t<6, wp::float32> var_14;
        wp::vec_t<6, wp::float32> var_15;
        //---------
        // forward
        // def _cfrc_backward(                                                                    <L 1219>
        // worldid, nodeid = wp.tid()                                                             <L 1229>
        builtin_tid2d(var_0, var_1);
        // bodyid = body_tree_[nodeid]                                                            <L 1230>
        var_2 = wp::address(var_body_tree_, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // pid = body_parentid[bodyid]                                                            <L 1231>
        var_5 = wp::address(var_body_parentid, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if bodyid != 0:                                                                        <L 1232>
        var_9 = (var_3 != var_8);
        if (var_9) {
            // wp.atomic_add(cfrc_int_out[worldid], pid, cfrc_int_in[worldid, bodyid])            <L 1233>
            var_10 = wp::slice_t(var_0, var_0, var_11);
            var_12 = wp::view(var_cfrc_int_out, var_10);
            var_13 = wp::address(var_cfrc_int_in, var_0, var_3);
            var_15 = wp::load(var_13);
            var_14 = wp::atomic_add(var_12, var_6, var_15);
        }
    }
}



extern "C" __global__ void _comvel_root_15820f35_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_out)
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
        const wp::int32 var_3 = 0;
        wp::vec_t<6, wp::float32>* var_4;
        wp::float32* var_5;
        //---------
        // forward
        // def _comvel_root(cvel_out: wp.array2d[wp.spatial_vector]):                             <L 1936>
        // worldid, elementid = wp.tid()                                                          <L 1937>
        builtin_tid2d(var_0, var_1);
        // cvel_out[worldid, 0][elementid] = 0.0                                                  <L 1938>
        var_4 = wp::address(var_cvel_out, var_0, var_3);
        var_5 = wp::indexref(var_4, var_1);
        wp::store(var_5, var_2);
    }
}



extern "C" __global__ void _comvel_branch_77de59e6_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_jntnum,
    wp::array_t<wp::int32> var_body_jntadr,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_body_branches,
    wp::array_t<wp::int32> var_body_branch_start,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_out,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_dot_out)
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
        const wp::int32 var_5 = 1;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::slice_t var_10;
        const wp::int32 var_11 = 0;
        wp::array_t<wp::float32> var_12;
        wp::slice_t var_13;
        const wp::int32 var_14 = 0;
        wp::array_t<wp::vec_t<6, wp::float32>> var_15;
        wp::range_t var_16;
        wp::int32 var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32* var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        wp::vec_t<6, wp::float32>* var_24;
        wp::vec_t<6, wp::float32> var_25;
        wp::vec_t<6, wp::float32> var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        wp::int32* var_33;
        wp::int32 var_34;
        wp::int32 var_35;
        const wp::int32 var_36 = 0;
        bool var_37;
        wp::int32 var_38;
        wp::range_t var_39;
        wp::int32 var_40;
        wp::int32* var_41;
        wp::int32 var_42;
        wp::int32 var_43;
        const wp::int32 var_44 = 0;
        bool var_45;
        const wp::int32 var_46 = 0;
        wp::int32 var_47;
        wp::vec_t<6, wp::float32>* var_48;
        const wp::int32 var_49 = 0;
        wp::int32 var_50;
        wp::float32* var_51;
        wp::vec_t<6, wp::float32> var_52;
        wp::vec_t<6, wp::float32> var_53;
        wp::float32 var_54;
        wp::vec_t<6, wp::float32> var_55;
        const wp::int32 var_56 = 1;
        wp::int32 var_57;
        wp::vec_t<6, wp::float32>* var_58;
        const wp::int32 var_59 = 1;
        wp::int32 var_60;
        wp::float32* var_61;
        wp::vec_t<6, wp::float32> var_62;
        wp::vec_t<6, wp::float32> var_63;
        wp::float32 var_64;
        wp::vec_t<6, wp::float32> var_65;
        const wp::int32 var_66 = 2;
        wp::int32 var_67;
        wp::vec_t<6, wp::float32>* var_68;
        const wp::int32 var_69 = 2;
        wp::int32 var_70;
        wp::float32* var_71;
        wp::vec_t<6, wp::float32> var_72;
        wp::vec_t<6, wp::float32> var_73;
        wp::float32 var_74;
        wp::vec_t<6, wp::float32> var_75;
        const wp::int32 var_76 = 3;
        wp::int32 var_77;
        wp::vec_t<6, wp::float32>* var_78;
        wp::vec_t<6, wp::float32> var_79;
        wp::vec_t<6, wp::float32> var_80;
        const wp::int32 var_81 = 3;
        wp::int32 var_82;
        const wp::int32 var_83 = 4;
        wp::int32 var_84;
        wp::vec_t<6, wp::float32>* var_85;
        wp::vec_t<6, wp::float32> var_86;
        wp::vec_t<6, wp::float32> var_87;
        const wp::int32 var_88 = 4;
        wp::int32 var_89;
        const wp::int32 var_90 = 5;
        wp::int32 var_91;
        wp::vec_t<6, wp::float32>* var_92;
        wp::vec_t<6, wp::float32> var_93;
        wp::vec_t<6, wp::float32> var_94;
        const wp::int32 var_95 = 5;
        wp::int32 var_96;
        const wp::int32 var_97 = 3;
        wp::int32 var_98;
        wp::vec_t<6, wp::float32>* var_99;
        const wp::int32 var_100 = 3;
        wp::int32 var_101;
        wp::float32* var_102;
        wp::vec_t<6, wp::float32> var_103;
        wp::vec_t<6, wp::float32> var_104;
        wp::float32 var_105;
        wp::vec_t<6, wp::float32> var_106;
        const wp::int32 var_107 = 4;
        wp::int32 var_108;
        wp::vec_t<6, wp::float32>* var_109;
        const wp::int32 var_110 = 4;
        wp::int32 var_111;
        wp::float32* var_112;
        wp::vec_t<6, wp::float32> var_113;
        wp::vec_t<6, wp::float32> var_114;
        wp::float32 var_115;
        wp::vec_t<6, wp::float32> var_116;
        const wp::int32 var_117 = 5;
        wp::int32 var_118;
        wp::vec_t<6, wp::float32>* var_119;
        const wp::int32 var_120 = 5;
        wp::int32 var_121;
        wp::float32* var_122;
        wp::vec_t<6, wp::float32> var_123;
        wp::vec_t<6, wp::float32> var_124;
        wp::float32 var_125;
        wp::vec_t<6, wp::float32> var_126;
        const wp::int32 var_127 = 6;
        wp::int32 var_128;
        wp::vec_t<6, wp::float32> var_129;
        wp::int32 var_130;
        const wp::int32 var_131 = 1;
        bool var_132;
        const wp::int32 var_133 = 0;
        wp::int32 var_134;
        wp::vec_t<6, wp::float32>* var_135;
        wp::vec_t<6, wp::float32> var_136;
        wp::vec_t<6, wp::float32> var_137;
        const wp::int32 var_138 = 0;
        wp::int32 var_139;
        const wp::int32 var_140 = 1;
        wp::int32 var_141;
        wp::vec_t<6, wp::float32>* var_142;
        wp::vec_t<6, wp::float32> var_143;
        wp::vec_t<6, wp::float32> var_144;
        const wp::int32 var_145 = 1;
        wp::int32 var_146;
        const wp::int32 var_147 = 2;
        wp::int32 var_148;
        wp::vec_t<6, wp::float32>* var_149;
        wp::vec_t<6, wp::float32> var_150;
        wp::vec_t<6, wp::float32> var_151;
        const wp::int32 var_152 = 2;
        wp::int32 var_153;
        const wp::int32 var_154 = 0;
        wp::int32 var_155;
        wp::vec_t<6, wp::float32>* var_156;
        const wp::int32 var_157 = 0;
        wp::int32 var_158;
        wp::float32* var_159;
        wp::vec_t<6, wp::float32> var_160;
        wp::vec_t<6, wp::float32> var_161;
        wp::float32 var_162;
        wp::vec_t<6, wp::float32> var_163;
        const wp::int32 var_164 = 1;
        wp::int32 var_165;
        wp::vec_t<6, wp::float32>* var_166;
        const wp::int32 var_167 = 1;
        wp::int32 var_168;
        wp::float32* var_169;
        wp::vec_t<6, wp::float32> var_170;
        wp::vec_t<6, wp::float32> var_171;
        wp::float32 var_172;
        wp::vec_t<6, wp::float32> var_173;
        const wp::int32 var_174 = 2;
        wp::int32 var_175;
        wp::vec_t<6, wp::float32>* var_176;
        const wp::int32 var_177 = 2;
        wp::int32 var_178;
        wp::float32* var_179;
        wp::vec_t<6, wp::float32> var_180;
        wp::vec_t<6, wp::float32> var_181;
        wp::float32 var_182;
        wp::vec_t<6, wp::float32> var_183;
        const wp::int32 var_184 = 3;
        wp::int32 var_185;
        wp::vec_t<6, wp::float32> var_186;
        wp::int32 var_187;
        wp::vec_t<6, wp::float32>* var_188;
        wp::vec_t<6, wp::float32> var_189;
        wp::vec_t<6, wp::float32> var_190;
        wp::vec_t<6, wp::float32>* var_191;
        wp::float32* var_192;
        wp::vec_t<6, wp::float32> var_193;
        wp::vec_t<6, wp::float32> var_194;
        wp::float32 var_195;
        wp::vec_t<6, wp::float32> var_196;
        const wp::int32 var_197 = 1;
        wp::int32 var_198;
        wp::vec_t<6, wp::float32> var_199;
        wp::int32 var_200;
        wp::vec_t<6, wp::float32> var_201;
        wp::int32 var_202;
        //---------
        // forward
        // def _comvel_branch(                                                                    <L 1942>
        // worldid, branchid = wp.tid()                                                           <L 1958>
        builtin_tid2d(var_0, var_1);
        // start = body_branch_start[branchid]                                                    <L 1960>
        var_2 = wp::address(var_body_branch_start, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // end = body_branch_start[branchid + 1]                                                  <L 1961>
        var_6 = wp::add(var_1, var_5);
        var_7 = wp::address(var_body_branch_start, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // qvel = qvel_in[worldid]                                                                <L 1963>
        var_10 = wp::slice_t(var_0, var_0, var_11);
        var_12 = wp::view(var_qvel_in, var_10);
        // cdof = cdof_in[worldid]                                                                <L 1964>
        var_13 = wp::slice_t(var_0, var_0, var_14);
        var_15 = wp::view(var_cdof_in, var_13);
        // for i in range(start, end):                                                            <L 1966>
        var_16 = wp::range(var_3, var_8);
        start_for_0:;
            if (iter_cmp(var_16) == 0) goto end_for_0;
            var_17 = wp::iter_next(var_16);
            // bodyid = body_branches[i]                                                          <L 1967>
            var_18 = wp::address(var_body_branches, var_17);
            var_20 = wp::load(var_18);
            var_19 = wp::copy(var_20);
            // pid = body_parentid[bodyid]                                                        <L 1968>
            var_21 = wp::address(var_body_parentid, var_19);
            var_23 = wp::load(var_21);
            var_22 = wp::copy(var_23);
            // cvel = cvel_out[worldid, pid]                                                      <L 1969>
            var_24 = wp::address(var_cvel_out, var_0, var_22);
            var_26 = wp::load(var_24);
            var_25 = wp::copy(var_26);
            // dofid = body_dofadr[bodyid]                                                        <L 1970>
            var_27 = wp::address(var_body_dofadr, var_19);
            var_29 = wp::load(var_27);
            var_28 = wp::copy(var_29);
            // jntid = body_jntadr[bodyid]                                                        <L 1971>
            var_30 = wp::address(var_body_jntadr, var_19);
            var_32 = wp::load(var_30);
            var_31 = wp::copy(var_32);
            // jntnum = body_jntnum[bodyid]                                                       <L 1972>
            var_33 = wp::address(var_body_jntnum, var_19);
            var_35 = wp::load(var_33);
            var_34 = wp::copy(var_35);
            // if jntnum == 0:                                                                    <L 1974>
            var_37 = (var_34 == var_36);
            if (var_37) {
                // cvel_out[worldid, bodyid] = cvel                                               <L 1975>
                wp::array_store(var_cvel_out, var_0, var_19, var_25);
                // continue                                                                       <L 1976>
                goto start_for_0;
            }
            // for j in range(jntid, jntid + jntnum):                                             <L 1978>
            var_38 = wp::add(var_31, var_34);
            var_39 = wp::range(var_31, var_38);
            start_for_2:;
                if (iter_cmp(var_39) == 0) goto end_for_2;
                var_40 = wp::iter_next(var_39);
                // jnttype = jnt_type[j]                                                          <L 1979>
                var_41 = wp::address(var_jnt_type, var_40);
                var_43 = wp::load(var_41);
                var_42 = wp::copy(var_43);
                // if jnttype == JointType.FREE:                                                  <L 1981>
                var_45 = (var_42 == var_44);
                if (var_45) {
                    // cvel += cdof[dofid + 0] * qvel[dofid + 0]                                  <L 1982>
                    var_47 = wp::add(var_28, var_46);
                    var_48 = wp::address(var_15, var_47);
                    var_50 = wp::add(var_28, var_49);
                    var_51 = wp::address(var_12, var_50);
                    var_53 = wp::load(var_48);
                    var_54 = wp::load(var_51);
                    var_52 = wp::mul(var_53, var_54);
                    var_55 = wp::add(var_25, var_52);
                    // cvel += cdof[dofid + 1] * qvel[dofid + 1]                                  <L 1983>
                    var_57 = wp::add(var_28, var_56);
                    var_58 = wp::address(var_15, var_57);
                    var_60 = wp::add(var_28, var_59);
                    var_61 = wp::address(var_12, var_60);
                    var_63 = wp::load(var_58);
                    var_64 = wp::load(var_61);
                    var_62 = wp::mul(var_63, var_64);
                    var_65 = wp::add(var_55, var_62);
                    // cvel += cdof[dofid + 2] * qvel[dofid + 2]                                  <L 1984>
                    var_67 = wp::add(var_28, var_66);
                    var_68 = wp::address(var_15, var_67);
                    var_70 = wp::add(var_28, var_69);
                    var_71 = wp::address(var_12, var_70);
                    var_73 = wp::load(var_68);
                    var_74 = wp::load(var_71);
                    var_72 = wp::mul(var_73, var_74);
                    var_75 = wp::add(var_65, var_72);
                    // cdof_dot_out[worldid, dofid + 3] = math.motion_cross(cvel, cdof[dofid + 3])       <L 1986>
                    var_77 = wp::add(var_28, var_76);
                    var_78 = wp::address(var_15, var_77);
                    var_80 = wp::load(var_78);
                    var_79 = motion_cross_0(var_75, var_80);
                    var_82 = wp::add(var_28, var_81);
                    wp::array_store(var_cdof_dot_out, var_0, var_82, var_79);
                    // cdof_dot_out[worldid, dofid + 4] = math.motion_cross(cvel, cdof[dofid + 4])       <L 1987>
                    var_84 = wp::add(var_28, var_83);
                    var_85 = wp::address(var_15, var_84);
                    var_87 = wp::load(var_85);
                    var_86 = motion_cross_0(var_75, var_87);
                    var_89 = wp::add(var_28, var_88);
                    wp::array_store(var_cdof_dot_out, var_0, var_89, var_86);
                    // cdof_dot_out[worldid, dofid + 5] = math.motion_cross(cvel, cdof[dofid + 5])       <L 1988>
                    var_91 = wp::add(var_28, var_90);
                    var_92 = wp::address(var_15, var_91);
                    var_94 = wp::load(var_92);
                    var_93 = motion_cross_0(var_75, var_94);
                    var_96 = wp::add(var_28, var_95);
                    wp::array_store(var_cdof_dot_out, var_0, var_96, var_93);
                    // cvel += cdof[dofid + 3] * qvel[dofid + 3]                                  <L 1990>
                    var_98 = wp::add(var_28, var_97);
                    var_99 = wp::address(var_15, var_98);
                    var_101 = wp::add(var_28, var_100);
                    var_102 = wp::address(var_12, var_101);
                    var_104 = wp::load(var_99);
                    var_105 = wp::load(var_102);
                    var_103 = wp::mul(var_104, var_105);
                    var_106 = wp::add(var_75, var_103);
                    // cvel += cdof[dofid + 4] * qvel[dofid + 4]                                  <L 1991>
                    var_108 = wp::add(var_28, var_107);
                    var_109 = wp::address(var_15, var_108);
                    var_111 = wp::add(var_28, var_110);
                    var_112 = wp::address(var_12, var_111);
                    var_114 = wp::load(var_109);
                    var_115 = wp::load(var_112);
                    var_113 = wp::mul(var_114, var_115);
                    var_116 = wp::add(var_106, var_113);
                    // cvel += cdof[dofid + 5] * qvel[dofid + 5]                                  <L 1992>
                    var_118 = wp::add(var_28, var_117);
                    var_119 = wp::address(var_15, var_118);
                    var_121 = wp::add(var_28, var_120);
                    var_122 = wp::address(var_12, var_121);
                    var_124 = wp::load(var_119);
                    var_125 = wp::load(var_122);
                    var_123 = wp::mul(var_124, var_125);
                    var_126 = wp::add(var_116, var_123);
                    // dofid += 6                                                                 <L 1994>
                    var_128 = wp::add(var_28, var_127);
                }
                var_129 = wp::where(var_45, var_126, var_25);
                var_130 = wp::where(var_45, var_128, var_28);
                if (!var_45) {
                    // elif jnttype == JointType.BALL:                                            <L 1995>
                    var_132 = (var_42 == var_131);
                    if (var_132) {
                        // cdof_dot_out[worldid, dofid + 0] = math.motion_cross(cvel, cdof[dofid + 0])       <L 1996>
                        var_134 = wp::add(var_130, var_133);
                        var_135 = wp::address(var_15, var_134);
                        var_137 = wp::load(var_135);
                        var_136 = motion_cross_0(var_129, var_137);
                        var_139 = wp::add(var_130, var_138);
                        wp::array_store(var_cdof_dot_out, var_0, var_139, var_136);
                        // cdof_dot_out[worldid, dofid + 1] = math.motion_cross(cvel, cdof[dofid + 1])       <L 1997>
                        var_141 = wp::add(var_130, var_140);
                        var_142 = wp::address(var_15, var_141);
                        var_144 = wp::load(var_142);
                        var_143 = motion_cross_0(var_129, var_144);
                        var_146 = wp::add(var_130, var_145);
                        wp::array_store(var_cdof_dot_out, var_0, var_146, var_143);
                        // cdof_dot_out[worldid, dofid + 2] = math.motion_cross(cvel, cdof[dofid + 2])       <L 1998>
                        var_148 = wp::add(var_130, var_147);
                        var_149 = wp::address(var_15, var_148);
                        var_151 = wp::load(var_149);
                        var_150 = motion_cross_0(var_129, var_151);
                        var_153 = wp::add(var_130, var_152);
                        wp::array_store(var_cdof_dot_out, var_0, var_153, var_150);
                        // cvel += cdof[dofid + 0] * qvel[dofid + 0]                              <L 2000>
                        var_155 = wp::add(var_130, var_154);
                        var_156 = wp::address(var_15, var_155);
                        var_158 = wp::add(var_130, var_157);
                        var_159 = wp::address(var_12, var_158);
                        var_161 = wp::load(var_156);
                        var_162 = wp::load(var_159);
                        var_160 = wp::mul(var_161, var_162);
                        var_163 = wp::add(var_129, var_160);
                        // cvel += cdof[dofid + 1] * qvel[dofid + 1]                              <L 2001>
                        var_165 = wp::add(var_130, var_164);
                        var_166 = wp::address(var_15, var_165);
                        var_168 = wp::add(var_130, var_167);
                        var_169 = wp::address(var_12, var_168);
                        var_171 = wp::load(var_166);
                        var_172 = wp::load(var_169);
                        var_170 = wp::mul(var_171, var_172);
                        var_173 = wp::add(var_163, var_170);
                        // cvel += cdof[dofid + 2] * qvel[dofid + 2]                              <L 2002>
                        var_175 = wp::add(var_130, var_174);
                        var_176 = wp::address(var_15, var_175);
                        var_178 = wp::add(var_130, var_177);
                        var_179 = wp::address(var_12, var_178);
                        var_181 = wp::load(var_176);
                        var_182 = wp::load(var_179);
                        var_180 = wp::mul(var_181, var_182);
                        var_183 = wp::add(var_173, var_180);
                        // dofid += 3                                                             <L 2004>
                        var_185 = wp::add(var_130, var_184);
                    }
                    var_186 = wp::where(var_132, var_183, var_129);
                    var_187 = wp::where(var_132, var_185, var_130);
                    if (!var_132) {
                        // cdof_dot_out[worldid, dofid] = math.motion_cross(cvel, cdof[dofid])       <L 2006>
                        var_188 = wp::address(var_15, var_187);
                        var_190 = wp::load(var_188);
                        var_189 = motion_cross_0(var_186, var_190);
                        wp::array_store(var_cdof_dot_out, var_0, var_187, var_189);
                        // cvel += cdof[dofid] * qvel[dofid]                                      <L 2007>
                        var_191 = wp::address(var_15, var_187);
                        var_192 = wp::address(var_12, var_187);
                        var_194 = wp::load(var_191);
                        var_195 = wp::load(var_192);
                        var_193 = wp::mul(var_194, var_195);
                        var_196 = wp::add(var_186, var_193);
                        // dofid += 1                                                             <L 2009>
                        var_198 = wp::add(var_187, var_197);
                    }
                    var_199 = wp::where(var_132, var_186, var_196);
                    var_200 = wp::where(var_132, var_187, var_198);
                }
                var_201 = wp::where(var_45, var_129, var_199);
                var_202 = wp::where(var_45, var_130, var_200);
                wp::assign(var_25, var_201);
                wp::assign(var_28, var_202);
                goto start_for_2;
            end_for_2:;
            // cvel_out[worldid, bodyid] = cvel                                                   <L 2011>
            wp::array_store(var_cvel_out, var_0, var_19, var_25);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void _cfrc_ext_f67cbe30_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::vec_t<6, wp::float32>> var_xfrc_applied_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_ext_out)
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
        const wp::float32 var_4 = 0.0;
        const wp::float32 var_5 = 0.0;
        const wp::float32 var_6 = 0.0;
        const wp::float32 var_7 = 0.0;
        const wp::float32 var_8 = 0.0;
        const wp::float32 var_9 = 0.0;
        wp::vec_t<6, wp::float32> var_10;
        const wp::int32 var_11 = 0;
        wp::vec_t<6, wp::float32>* var_12;
        wp::vec_t<6, wp::float32> var_13;
        wp::vec_t<6, wp::float32> var_14;
        wp::int32* var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::int32 var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32>* var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<6, wp::float32> var_24;
        //---------
        // forward
        // def _cfrc_ext(                                                                         <L 1278>
        // worldid, bodyid = wp.tid()                                                             <L 1288>
        builtin_tid2d(var_0, var_1);
        // if bodyid == 0:                                                                        <L 1289>
        var_3 = (var_1 == var_2);
        if (var_3) {
            // cfrc_ext_out[worldid, 0] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)         <L 1290>
            var_10 = wp::vec_t<6, wp::float32>({var_4, var_5, var_6, var_7, var_8, var_9});
            wp::array_store(var_cfrc_ext_out, var_0, var_11, var_10);
        }
        if (!var_3) {
            // xfrc_applied = xfrc_applied_in[worldid, bodyid]                                    <L 1292>
            var_12 = wp::address(var_xfrc_applied_in, var_0, var_1);
            var_14 = wp::load(var_12);
            var_13 = wp::copy(var_14);
            // subtree_com = subtree_com_in[worldid, body_rootid[bodyid]]                         <L 1293>
            var_15 = wp::address(var_body_rootid, var_1);
            var_17 = wp::load(var_15);
            var_16 = wp::address(var_subtree_com_in, var_0, var_17);
            var_19 = wp::load(var_16);
            var_18 = wp::copy(var_19);
            // xipos = xipos_in[worldid, bodyid]                                                  <L 1294>
            var_20 = wp::address(var_xipos_in, var_0, var_1);
            var_22 = wp::load(var_20);
            var_21 = wp::copy(var_22);
            // cfrc_ext_out[worldid, bodyid] = support.transform_force(xfrc_applied, subtree_com - xipos)       <L 1295>
            var_23 = wp::sub(var_18, var_21);
            var_24 = transform_force_1(var_13, var_23);
            wp::array_store(var_cfrc_ext_out, var_0, var_1, var_24);
        }
    }
}



extern "C" __global__ void _count_equality_constraints_54fc2a85_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_eq_type,
    wp::array_t<wp::int32> var_ne_in,
    wp::array_t<wp::int32> var_efc_type_in,
    wp::array_t<wp::int32> var_efc_id_in,
    wp::array_t<wp::int32> var_ne_connect_out,
    wp::array_t<wp::int32> var_ne_weld_out)
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
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        const wp::int32 var_11 = 0;
        bool var_12;
        const wp::int32 var_13 = 1;
        wp::int32 var_14;
        const wp::int32 var_15 = 1;
        bool var_16;
        const wp::int32 var_17 = 1;
        wp::int32 var_18;
        //---------
        // forward
        // def _count_equality_constraints(                                                       <L 1299>
        // worldid, efcid = wp.tid()                                                              <L 1311>
        builtin_tid2d(var_0, var_1);
        // if efcid >= ne_in[worldid]:                                                            <L 1314>
        var_2 = wp::address(var_ne_in, var_0);
        var_4 = wp::load(var_2);
        var_3 = (var_1 >= var_4);
        if (var_3) {
            // return                                                                             <L 1315>
            continue;
        }
        // eq_id = efc_id_in[worldid, efcid]                                                      <L 1318>
        var_5 = wp::address(var_efc_id_in, var_0, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // eq_constraint_type = eq_type[eq_id]                                                    <L 1319>
        var_8 = wp::address(var_eq_type, var_6);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // if eq_constraint_type == EqType.CONNECT:                                               <L 1322>
        var_12 = (var_9 == var_11);
        if (var_12) {
            // wp.atomic_add(ne_connect_out, worldid, 1)                                          <L 1323>
            var_14 = wp::atomic_add(var_ne_connect_out, var_0, var_13);
        }
        if (!var_12) {
            // elif eq_constraint_type == EqType.WELD:                                            <L 1324>
            var_16 = (var_9 == var_15);
            if (var_16) {
                // wp.atomic_add(ne_weld_out, worldid, 1)                                         <L 1325>
                var_18 = wp::atomic_add(var_ne_weld_out, var_0, var_17);
            }
        }
    }
}



extern "C" __global__ void _cfrc_ext_equality_3475e5f6_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_pos,
    wp::array_t<wp::int32> var_eq_obj1id,
    wp::array_t<wp::int32> var_eq_obj2id,
    wp::array_t<wp::int32> var_eq_objtype,
    wp::array_t<wp::vec_t<11, wp::float32>> var_eq_data,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::int32> var_efc_id_in,
    wp::array_t<wp::float32> var_efc_force_in,
    wp::array_t<wp::int32> var_ne_connect_in,
    wp::array_t<wp::int32> var_ne_weld_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_ext_out)
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
        const wp::int32 var_8 = 3;
        wp::int32 var_9;
        const wp::int32 var_10 = 6;
        wp::int32 var_11;
        wp::int32 var_12;
        bool var_13;
        bool var_14;
        const wp::int32 var_15 = 3;
        wp::int32 var_16;
        const wp::float32 var_17 = 0.0;
        const wp::float32 var_18 = 0.0;
        const wp::float32 var_19 = 0.0;
        wp::vec_t<3, wp::float32> var_20;
        const wp::int32 var_21 = 6;
        wp::int32 var_22;
        wp::int32 var_23;
        const wp::int32 var_24 = 3;
        wp::int32 var_25;
        wp::float32* var_26;
        const wp::int32 var_27 = 4;
        wp::int32 var_28;
        wp::float32* var_29;
        const wp::int32 var_30 = 5;
        wp::int32 var_31;
        wp::float32* var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::int32 var_37;
        wp::vec_t<3, wp::float32> var_38;
        const wp::int32 var_39 = 0;
        wp::int32 var_40;
        wp::float32* var_41;
        const wp::int32 var_42 = 1;
        wp::int32 var_43;
        wp::float32* var_44;
        const wp::int32 var_45 = 2;
        wp::int32 var_46;
        wp::float32* var_47;
        wp::vec_t<3, wp::float32> var_48;
        wp::float32 var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        wp::int32* var_52;
        wp::int32 var_53;
        wp::int32 var_54;
        wp::shape_t* var_55;
        const wp::int32 var_56 = 0;
        wp::int32 var_57;
        wp::shape_t var_58;
        wp::int32 var_59;
        wp::vec_t<11, wp::float32>* var_60;
        wp::vec_t<11, wp::float32> var_61;
        wp::vec_t<11, wp::float32> var_62;
        wp::int32* var_63;
        const wp::int32 var_64 = 1;
        bool var_65;
        wp::int32 var_66;
        wp::int32* var_67;
        wp::int32 var_68;
        wp::int32 var_69;
        wp::int32* var_70;
        wp::int32 var_71;
        wp::int32 var_72;
        wp::int32 var_73;
        wp::int32 var_74;
        wp::int32* var_75;
        wp::int32 var_76;
        wp::int32 var_77;
        wp::int32* var_78;
        wp::int32 var_79;
        wp::int32 var_80;
        wp::int32 var_81;
        wp::int32 var_82;
        const wp::int32 var_83 = 0;
        wp::float32 var_84;
        const wp::int32 var_85 = 1;
        wp::float32 var_86;
        const wp::int32 var_87 = 2;
        wp::float32 var_88;
        wp::vec_t<3, wp::float32> var_89;
        const wp::int32 var_90 = 3;
        wp::float32 var_91;
        const wp::int32 var_92 = 4;
        wp::float32 var_93;
        const wp::int32 var_94 = 5;
        wp::float32 var_95;
        wp::vec_t<3, wp::float32> var_96;
        wp::vec_t<3, wp::float32> var_97;
        wp::shape_t* var_98;
        const wp::int32 var_99 = 0;
        wp::int32 var_100;
        wp::shape_t var_101;
        wp::int32 var_102;
        wp::vec_t<3, wp::float32>* var_103;
        wp::vec_t<3, wp::float32> var_104;
        wp::vec_t<3, wp::float32> var_105;
        wp::vec_t<3, wp::float32> var_106;
        wp::mat_t<3, 3, wp::float32>* var_107;
        wp::vec_t<3, wp::float32> var_108;
        wp::mat_t<3, 3, wp::float32> var_109;
        wp::vec_t<3, wp::float32>* var_110;
        wp::vec_t<3, wp::float32> var_111;
        wp::vec_t<3, wp::float32> var_112;
        wp::int32* var_113;
        wp::vec_t<3, wp::float32>* var_114;
        wp::int32 var_115;
        wp::vec_t<3, wp::float32> var_116;
        wp::vec_t<3, wp::float32> var_117;
        wp::vec_t<3, wp::float32> var_118;
        wp::vec_t<3, wp::float32> var_119;
        wp::vec_t<3, wp::float32> var_120;
        wp::vec_t<6, wp::float32> var_121;
        wp::slice_t var_122;
        const wp::int32 var_123 = 0;
        wp::array_t<wp::vec_t<6, wp::float32>> var_124;
        wp::vec_t<6, wp::float32> var_125;
        const wp::int32 var_126 = 3;
        wp::float32 var_127;
        const wp::int32 var_128 = 4;
        wp::float32 var_129;
        const wp::int32 var_130 = 5;
        wp::float32 var_131;
        wp::vec_t<3, wp::float32> var_132;
        wp::vec_t<3, wp::float32> var_133;
        const wp::int32 var_134 = 0;
        wp::float32 var_135;
        const wp::int32 var_136 = 1;
        wp::float32 var_137;
        const wp::int32 var_138 = 2;
        wp::float32 var_139;
        wp::vec_t<3, wp::float32> var_140;
        wp::vec_t<3, wp::float32> var_141;
        wp::vec_t<3, wp::float32> var_142;
        wp::shape_t* var_143;
        const wp::int32 var_144 = 0;
        wp::int32 var_145;
        wp::shape_t var_146;
        wp::int32 var_147;
        wp::vec_t<3, wp::float32>* var_148;
        wp::vec_t<3, wp::float32> var_149;
        wp::vec_t<3, wp::float32> var_150;
        wp::vec_t<3, wp::float32> var_151;
        wp::mat_t<3, 3, wp::float32>* var_152;
        wp::vec_t<3, wp::float32> var_153;
        wp::mat_t<3, 3, wp::float32> var_154;
        wp::vec_t<3, wp::float32>* var_155;
        wp::vec_t<3, wp::float32> var_156;
        wp::vec_t<3, wp::float32> var_157;
        wp::int32* var_158;
        wp::vec_t<3, wp::float32>* var_159;
        wp::int32 var_160;
        wp::vec_t<3, wp::float32> var_161;
        wp::vec_t<3, wp::float32> var_162;
        wp::vec_t<3, wp::float32> var_163;
        wp::vec_t<3, wp::float32> var_164;
        wp::vec_t<3, wp::float32> var_165;
        wp::vec_t<6, wp::float32> var_166;
        wp::slice_t var_167;
        const wp::int32 var_168 = 0;
        wp::array_t<wp::vec_t<6, wp::float32>> var_169;
        wp::vec_t<6, wp::float32> var_170;
        wp::vec_t<3, wp::float32> var_171;
        wp::vec_t<3, wp::float32> var_172;
        wp::vec_t<3, wp::float32> var_173;
        wp::vec_t<3, wp::float32> var_174;
        wp::vec_t<6, wp::float32> var_175;
        //---------
        // forward
        // def _cfrc_ext_equality(                                                                <L 1329>
        // worldid, eqid = wp.tid()                                                               <L 1350>
        builtin_tid2d(var_0, var_1);
        // ne_connect = ne_connect_in[worldid]                                                    <L 1352>
        var_2 = wp::address(var_ne_connect_in, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // ne_weld = ne_weld_in[worldid]                                                          <L 1353>
        var_5 = wp::address(var_ne_weld_in, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // num_connect = ne_connect // 3                                                          <L 1354>
        var_9 = wp::floordiv(var_3, var_8);
        // if eqid >= num_connect + ne_weld // 6:                                                 <L 1356>
        var_11 = wp::floordiv(var_6, var_10);
        var_12 = wp::add(var_9, var_11);
        var_13 = (var_1 >= var_12);
        if (var_13) {
            // return                                                                             <L 1357>
            continue;
        }
        // is_connect = eqid < num_connect                                                        <L 1359>
        var_14 = (var_1 < var_9);
        // if is_connect:                                                                         <L 1360>
        if (var_14) {
            // efcid = 3 * eqid                                                                   <L 1361>
            var_16 = wp::mul(var_15, var_1);
            // cfrc_torque = wp.vec3(0.0, 0.0, 0.0)  # no torque from connect                     <L 1362>
            var_20 = wp::vec_t<3, wp::float32>(var_17, var_18, var_19);
        }
        if (!var_14) {
            // efcid = 6 * eqid - ne_connect                                                      <L 1364>
            var_22 = wp::mul(var_21, var_1);
            var_23 = wp::sub(var_22, var_3);
            // cfrc_torque = wp.vec3(efc_force_in[worldid, efcid + 3], efc_force_in[worldid, efcid + 4], efc_force_in[worldid, efcid + 5])       <L 1365>
            var_25 = wp::add(var_23, var_24);
            var_26 = wp::address(var_efc_force_in, var_0, var_25);
            var_28 = wp::add(var_23, var_27);
            var_29 = wp::address(var_efc_force_in, var_0, var_28);
            var_31 = wp::add(var_23, var_30);
            var_32 = wp::address(var_efc_force_in, var_0, var_31);
            var_34 = wp::load(var_26);
            var_35 = wp::load(var_29);
            var_36 = wp::load(var_32);
            var_33 = wp::vec_t<3, wp::float32>(var_34, var_35, var_36);
        }
        var_37 = wp::where(var_14, var_16, var_23);
        var_38 = wp::where(var_14, var_20, var_33);
        // cfrc_force = wp.vec3(                                                                  <L 1367>
        // efc_force_in[worldid, efcid + 0],                                                      <L 1368>
        var_40 = wp::add(var_37, var_39);
        var_41 = wp::address(var_efc_force_in, var_0, var_40);
        // efc_force_in[worldid, efcid + 1],                                                      <L 1369>
        var_43 = wp::add(var_37, var_42);
        var_44 = wp::address(var_efc_force_in, var_0, var_43);
        // efc_force_in[worldid, efcid + 2],                                                      <L 1370>
        var_46 = wp::add(var_37, var_45);
        var_47 = wp::address(var_efc_force_in, var_0, var_46);
        var_49 = wp::load(var_41);
        var_50 = wp::load(var_44);
        var_51 = wp::load(var_47);
        var_48 = wp::vec_t<3, wp::float32>(var_49, var_50, var_51);
        // id = efc_id_in[worldid, efcid]                                                         <L 1373>
        var_52 = wp::address(var_efc_id_in, var_0, var_37);
        var_54 = wp::load(var_52);
        var_53 = wp::copy(var_54);
        // eq_data_ = eq_data[worldid % eq_data.shape[0], id]                                     <L 1374>
        var_55 = &(var_eq_data.shape);
        var_58 = wp::load(var_55);
        var_57 = wp::extract(var_58, var_56);
        var_59 = wp::mod(var_0, var_57);
        var_60 = wp::address(var_eq_data, var_59, var_53);
        var_62 = wp::load(var_60);
        var_61 = wp::copy(var_62);
        // body_semantic = eq_objtype[id] == ObjType.BODY                                         <L 1375>
        var_63 = wp::address(var_eq_objtype, var_53);
        var_66 = wp::load(var_63);
        var_65 = (var_66 == var_64);
        // obj1 = eq_obj1id[id]                                                                   <L 1377>
        var_67 = wp::address(var_eq_obj1id, var_53);
        var_69 = wp::load(var_67);
        var_68 = wp::copy(var_69);
        // obj2 = eq_obj2id[id]                                                                   <L 1378>
        var_70 = wp::address(var_eq_obj2id, var_53);
        var_72 = wp::load(var_70);
        var_71 = wp::copy(var_72);
        // if body_semantic:                                                                      <L 1380>
        if (var_65) {
            // bodyid1 = obj1                                                                     <L 1381>
            var_73 = wp::copy(var_68);
            // bodyid2 = obj2                                                                     <L 1382>
            var_74 = wp::copy(var_71);
        }
        if (!var_65) {
            // bodyid1 = site_bodyid[obj1]                                                        <L 1384>
            var_75 = wp::address(var_site_bodyid, var_68);
            var_77 = wp::load(var_75);
            var_76 = wp::copy(var_77);
            // bodyid2 = site_bodyid[obj2]                                                        <L 1385>
            var_78 = wp::address(var_site_bodyid, var_71);
            var_80 = wp::load(var_78);
            var_79 = wp::copy(var_80);
        }
        var_81 = wp::where(var_65, var_73, var_76);
        var_82 = wp::where(var_65, var_74, var_79);
        // if bodyid1:                                                                            <L 1388>
        if (var_81) {
            // if body_semantic:                                                                  <L 1389>
            if (var_65) {
                // if is_connect:                                                                 <L 1390>
                if (var_14) {
                    // offset = wp.vec3(eq_data_[0], eq_data_[1], eq_data_[2])                    <L 1391>
                    var_84 = wp::extract(var_61, var_83);
                    var_86 = wp::extract(var_61, var_85);
                    var_88 = wp::extract(var_61, var_87);
                    var_89 = wp::vec_t<3, wp::float32>(var_84, var_86, var_88);
                }
                if (!var_14) {
                    // offset = wp.vec3(eq_data_[3], eq_data_[4], eq_data_[5])                    <L 1393>
                    var_91 = wp::extract(var_61, var_90);
                    var_93 = wp::extract(var_61, var_92);
                    var_95 = wp::extract(var_61, var_94);
                    var_96 = wp::vec_t<3, wp::float32>(var_91, var_93, var_95);
                }
                var_97 = wp::where(var_14, var_89, var_96);
            }
            if (!var_65) {
                // offset = site_pos[worldid % site_pos.shape[0], obj1]                           <L 1395>
                var_98 = &(var_site_pos.shape);
                var_101 = wp::load(var_98);
                var_100 = wp::extract(var_101, var_99);
                var_102 = wp::mod(var_0, var_100);
                var_103 = wp::address(var_site_pos, var_102, var_68);
                var_105 = wp::load(var_103);
                var_104 = wp::copy(var_105);
            }
            var_106 = wp::where(var_65, var_97, var_104);
            // pos = xmat_in[worldid, bodyid1] @ offset + xpos_in[worldid, bodyid1]               <L 1398>
            var_107 = wp::address(var_xmat_in, var_0, var_81);
            var_109 = wp::load(var_107);
            var_108 = wp::mul(var_109, var_106);
            var_110 = wp::address(var_xpos_in, var_0, var_81);
            var_112 = wp::load(var_110);
            var_111 = wp::add(var_108, var_112);
            // newpos = subtree_com_in[worldid, body_rootid[bodyid1]]                             <L 1401>
            var_113 = wp::address(var_body_rootid, var_81);
            var_115 = wp::load(var_113);
            var_114 = wp::address(var_subtree_com_in, var_0, var_115);
            var_117 = wp::load(var_114);
            var_116 = wp::copy(var_117);
            // dif = newpos - pos                                                                 <L 1403>
            var_118 = wp::sub(var_116, var_111);
            // cfrc_com = wp.spatial_vector(cfrc_torque - wp.cross(dif, cfrc_force), cfrc_force)       <L 1404>
            var_119 = wp::cross(var_118, var_48);
            var_120 = wp::sub(var_38, var_119);
            var_121 = wp::vec_t<6, wp::float32>(var_120, var_48);
            // wp.atomic_add(cfrc_ext_out[worldid], bodyid1, cfrc_com)                            <L 1407>
            var_122 = wp::slice_t(var_0, var_0, var_123);
            var_124 = wp::view(var_cfrc_ext_out, var_122);
            var_125 = wp::atomic_add(var_124, var_81, var_121);
        }
        // if bodyid2:                                                                            <L 1410>
        if (var_82) {
            // if body_semantic:                                                                  <L 1411>
            if (var_65) {
                // if is_connect:                                                                 <L 1412>
                if (var_14) {
                    // offset = wp.vec3(eq_data_[3], eq_data_[4], eq_data_[5])                    <L 1413>
                    var_127 = wp::extract(var_61, var_126);
                    var_129 = wp::extract(var_61, var_128);
                    var_131 = wp::extract(var_61, var_130);
                    var_132 = wp::vec_t<3, wp::float32>(var_127, var_129, var_131);
                }
                var_133 = wp::where(var_14, var_132, var_106);
                if (!var_14) {
                    // offset = wp.vec3(eq_data_[0], eq_data_[1], eq_data_[2])                    <L 1415>
                    var_135 = wp::extract(var_61, var_134);
                    var_137 = wp::extract(var_61, var_136);
                    var_139 = wp::extract(var_61, var_138);
                    var_140 = wp::vec_t<3, wp::float32>(var_135, var_137, var_139);
                }
                var_141 = wp::where(var_14, var_133, var_140);
            }
            var_142 = wp::where(var_65, var_141, var_106);
            if (!var_65) {
                // offset = site_pos[worldid % site_pos.shape[0], obj2]                           <L 1417>
                var_143 = &(var_site_pos.shape);
                var_146 = wp::load(var_143);
                var_145 = wp::extract(var_146, var_144);
                var_147 = wp::mod(var_0, var_145);
                var_148 = wp::address(var_site_pos, var_147, var_71);
                var_150 = wp::load(var_148);
                var_149 = wp::copy(var_150);
            }
            var_151 = wp::where(var_65, var_142, var_149);
            // pos = xmat_in[worldid, bodyid2] @ offset + xpos_in[worldid, bodyid2]               <L 1420>
            var_152 = wp::address(var_xmat_in, var_0, var_82);
            var_154 = wp::load(var_152);
            var_153 = wp::mul(var_154, var_151);
            var_155 = wp::address(var_xpos_in, var_0, var_82);
            var_157 = wp::load(var_155);
            var_156 = wp::add(var_153, var_157);
            // newpos = subtree_com_in[worldid, body_rootid[bodyid2]]                             <L 1423>
            var_158 = wp::address(var_body_rootid, var_82);
            var_160 = wp::load(var_158);
            var_159 = wp::address(var_subtree_com_in, var_0, var_160);
            var_162 = wp::load(var_159);
            var_161 = wp::copy(var_162);
            // dif = newpos - pos                                                                 <L 1425>
            var_163 = wp::sub(var_161, var_156);
            // cfrc_com = wp.spatial_vector(cfrc_torque - wp.cross(dif, cfrc_force), cfrc_force)       <L 1426>
            var_164 = wp::cross(var_163, var_48);
            var_165 = wp::sub(var_38, var_164);
            var_166 = wp::vec_t<6, wp::float32>(var_165, var_48);
            // wp.atomic_sub(cfrc_ext_out[worldid], bodyid2, cfrc_com)                            <L 1429>
            var_167 = wp::slice_t(var_0, var_0, var_168);
            var_169 = wp::view(var_cfrc_ext_out, var_167);
            var_170 = wp::atomic_sub(var_169, var_82, var_166);
        }
        var_171 = wp::where(var_82, var_151, var_106);
        var_172 = wp::where(var_82, var_156, var_111);
        var_173 = wp::where(var_82, var_161, var_116);
        var_174 = wp::where(var_82, var_163, var_118);
        var_175 = wp::where(var_82, var_166, var_121);
    }
}



extern "C" __global__ void _spatial_site_tendon_d90d9688_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::int32> var_wrap_objid,
    wp::array_t<wp::int32> var_tendon_site_pair_adr,
    wp::array_t<wp::int32> var_wrap_site_pair_adr,
    wp::array_t<wp::float32> var_wrap_pulley_scale,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::float32> var_ten_J_out,
    wp::array_t<wp::float32> var_ten_length_out)
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
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        const wp::int32 var_11 = 0;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        const wp::int32 var_16 = 1;
        wp::int32 var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32>* var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::float32 var_29;
        wp::slice_t var_30;
        const wp::int32 var_31 = 0;
        wp::array_t<wp::float32> var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        const wp::float32 var_35 = 1e-15;
        bool var_36;
        const wp::float32 var_37 = 1.0;
        const wp::float32 var_38 = 0.0;
        const wp::float32 var_39 = 0.0;
        wp::vec_t<3, wp::float32> var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::int32* var_42;
        wp::int32 var_43;
        wp::int32 var_44;
        wp::int32* var_45;
        wp::int32 var_46;
        wp::int32 var_47;
        bool var_48;
        wp::int32* var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        wp::int32* var_52;
        wp::int32 var_53;
        wp::int32 var_54;
        wp::int32* var_55;
        wp::vec_t<3, wp::float32>* var_56;
        wp::int32 var_57;
        wp::vec_t<3, wp::float32> var_58;
        wp::vec_t<3, wp::float32> var_59;
        wp::int32* var_60;
        wp::vec_t<3, wp::float32>* var_61;
        wp::int32 var_62;
        wp::vec_t<3, wp::float32> var_63;
        wp::vec_t<3, wp::float32> var_64;
        wp::float32 var_65;
        //---------
        // forward
        // def _spatial_site_tendon(                                                              <L 3173>
        // worldid, elementid = wp.tid()                                                          <L 3195>
        builtin_tid2d(var_0, var_1);
        // site_pair_adr = wrap_site_pair_adr[elementid]                                          <L 3198>
        var_2 = wp::address(var_wrap_site_pair_adr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // tenid = tendon_site_pair_adr[elementid]                                                <L 3199>
        var_5 = wp::address(var_tendon_site_pair_adr, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // pulley_scale = wrap_pulley_scale[site_pair_adr]                                        <L 3202>
        var_8 = wp::address(var_wrap_pulley_scale, var_3);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // id0 = wrap_objid[site_pair_adr + 0]                                                    <L 3204>
        var_12 = wp::add(var_3, var_11);
        var_13 = wp::address(var_wrap_objid, var_12);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // id1 = wrap_objid[site_pair_adr + 1]                                                    <L 3205>
        var_17 = wp::add(var_3, var_16);
        var_18 = wp::address(var_wrap_objid, var_17);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // pnt0 = site_xpos_in[worldid, id0]                                                      <L 3207>
        var_21 = wp::address(var_site_xpos_in, var_0, var_14);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // pnt1 = site_xpos_in[worldid, id1]                                                      <L 3208>
        var_24 = wp::address(var_site_xpos_in, var_0, var_19);
        var_26 = wp::load(var_24);
        var_25 = wp::copy(var_26);
        // dif = pnt1 - pnt0                                                                      <L 3209>
        var_27 = wp::sub(var_25, var_22);
        // vec, length = math.normalize_with_norm(dif)                                            <L 3210>
        normalize_with_norm_0(var_27, var_28, var_29);
        // wp.atomic_add(ten_length_out[worldid], tenid, length * pulley_scale)                   <L 3211>
        var_30 = wp::slice_t(var_0, var_0, var_31);
        var_32 = wp::view(var_ten_length_out, var_30);
        var_33 = wp::mul(var_29, var_9);
        var_34 = wp::atomic_add(var_32, var_6, var_33);
        // if length < MJ_MINVAL:                                                                 <L 3213>
        var_36 = (var_29 < var_35);
        if (var_36) {
            // vec = wp.vec3(1.0, 0.0, 0.0)                                                       <L 3214>
            var_40 = wp::vec_t<3, wp::float32>(var_37, var_38, var_39);
        }
        var_41 = wp::where(var_36, var_40, var_28);
        // body0 = site_bodyid[id0]                                                               <L 3216>
        var_42 = wp::address(var_site_bodyid, var_14);
        var_44 = wp::load(var_42);
        var_43 = wp::copy(var_44);
        // body1 = site_bodyid[id1]                                                               <L 3217>
        var_45 = wp::address(var_site_bodyid, var_19);
        var_47 = wp::load(var_45);
        var_46 = wp::copy(var_47);
        // if body0 != body1:                                                                     <L 3218>
        var_48 = (var_43 != var_46);
        if (var_48) {
            // rownnz = ten_J_rownnz[tenid]                                                       <L 3219>
            var_49 = wp::address(var_ten_J_rownnz, var_6);
            var_51 = wp::load(var_49);
            var_50 = wp::copy(var_51);
            // rowadr = ten_J_rowadr[tenid]                                                       <L 3220>
            var_52 = wp::address(var_ten_J_rowadr, var_6);
            var_54 = wp::load(var_52);
            var_53 = wp::copy(var_54);
            // offset0 = pnt0 - subtree_com_in[worldid, body_rootid[body0]]                       <L 3221>
            var_55 = wp::address(var_body_rootid, var_43);
            var_57 = wp::load(var_55);
            var_56 = wp::address(var_subtree_com_in, var_0, var_57);
            var_59 = wp::load(var_56);
            var_58 = wp::sub(var_22, var_59);
            // offset1 = pnt1 - subtree_com_in[worldid, body_rootid[body1]]                       <L 3222>
            var_60 = wp::address(var_body_rootid, var_46);
            var_62 = wp::load(var_60);
            var_61 = wp::address(var_subtree_com_in, var_0, var_62);
            var_64 = wp::load(var_61);
            var_63 = wp::sub(var_25, var_64);
            // _accumulate_jac_chain(                                                             <L 3223>
            // body_parentid,                                                                     <L 3224>
            // body_dofnum,                                                                       <L 3225>
            // body_dofadr,                                                                       <L 3226>
            // ten_J_colind,                                                                      <L 3227>
            // cdof_in,                                                                           <L 3228>
            // offset0,                                                                           <L 3229>
            // vec,                                                                               <L 3230>
            // body0,                                                                             <L 3231>
            // rowadr,                                                                            <L 3232>
            // rownnz,                                                                            <L 3233>
            // -pulley_scale,                                                                     <L 3234>
            var_65 = wp::neg(var_9);
            // worldid,                                                                           <L 3235>
            // ten_J_out,                                                                         <L 3236>
            _accumulate_jac_chain_0(var_body_parentid, var_body_dofnum, var_body_dofadr, var_ten_J_colind, var_cdof_in, var_58, var_41, var_43, var_53, var_50, var_65, var_0, var_ten_J_out);
            // _accumulate_jac_chain(                                                             <L 3238>
            // body_parentid,                                                                     <L 3239>
            // body_dofnum,                                                                       <L 3240>
            // body_dofadr,                                                                       <L 3241>
            // ten_J_colind,                                                                      <L 3242>
            // cdof_in,                                                                           <L 3243>
            // offset1,                                                                           <L 3244>
            // vec,                                                                               <L 3245>
            // body1,                                                                             <L 3246>
            // rowadr,                                                                            <L 3247>
            // rownnz,                                                                            <L 3248>
            // pulley_scale,                                                                      <L 3249>
            // worldid,                                                                           <L 3250>
            // ten_J_out,                                                                         <L 3251>
            _accumulate_jac_chain_0(var_body_parentid, var_body_dofnum, var_body_dofadr, var_ten_J_colind, var_cdof_in, var_63, var_41, var_46, var_53, var_50, var_9, var_0, var_ten_J_out);
        }
    }
}



extern "C" __global__ void _subtree_com_init_002b02fd_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_body_mass,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_out)
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
        wp::vec_t<3, wp::float32>* var_2;
        wp::shape_t* var_3;
        const wp::int32 var_4 = 0;
        wp::int32 var_5;
        wp::shape_t var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::vec_t<3, wp::float32> var_9;
        wp::vec_t<3, wp::float32> var_10;
        wp::float32 var_11;
        //---------
        // forward
        // def _subtree_com_init(                                                                 <L 464>
        // worldid, bodyid = wp.tid()                                                             <L 472>
        builtin_tid2d(var_0, var_1);
        // subtree_com_out[worldid, bodyid] = xipos_in[worldid, bodyid] * body_mass[worldid % body_mass.shape[0], bodyid]       <L 473>
        var_2 = wp::address(var_xipos_in, var_0, var_1);
        var_3 = &(var_body_mass.shape);
        var_6 = wp::load(var_3);
        var_5 = wp::extract(var_6, var_4);
        var_7 = wp::mod(var_0, var_5);
        var_8 = wp::address(var_body_mass, var_7, var_1);
        var_10 = wp::load(var_2);
        var_11 = wp::load(var_8);
        var_9 = wp::mul(var_10, var_11);
        wp::array_store(var_subtree_com_out, var_0, var_1, var_9);
    }
}



extern "C" __global__ void _qM_sparse_6b376672_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::int32> var_dof_parentid,
    wp::array_t<wp::int32> var_dof_Madr,
    wp::array_t<wp::float32> var_dof_armature,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<10, wp::float32>> var_crb_in,
    wp::array_t<wp::float32> var_qM_out)
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
        wp::shape_t* var_8;
        const wp::int32 var_9 = 0;
        wp::int32 var_10;
        wp::shape_t var_11;
        wp::int32 var_12;
        wp::float32* var_13;
        const wp::int32 var_14 = 0;
        wp::float32 var_15;
        wp::vec_t<10, wp::float32>* var_16;
        wp::vec_t<6, wp::float32>* var_17;
        wp::vec_t<6, wp::float32> var_18;
        wp::vec_t<10, wp::float32> var_19;
        wp::vec_t<6, wp::float32> var_20;
        const wp::int32 var_21 = 0;
        bool var_22;
        wp::vec_t<6, wp::float32>* var_23;
        wp::float32 var_24;
        wp::vec_t<6, wp::float32> var_25;
        const wp::int32 var_26 = 0;
        wp::float32 var_27;
        const wp::int32 var_28 = 1;
        wp::int32 var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        //---------
        // forward
        // def _qM_sparse(                                                                        <L 826>
        // worldid, dofid = wp.tid()                                                              <L 838>
        builtin_tid2d(var_0, var_1);
        // madr_ij = dof_Madr[dofid]  # dof_Madr is not batched                                   <L 839>
        var_2 = wp::address(var_dof_Madr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // bodyid = dof_bodyid[dofid]                                                             <L 840>
        var_5 = wp::address(var_dof_bodyid, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // qM_out[worldid, 0, madr_ij] = dof_armature[worldid % dof_armature.shape[0], dofid]       <L 843>
        var_8 = &(var_dof_armature.shape);
        var_11 = wp::load(var_8);
        var_10 = wp::extract(var_11, var_9);
        var_12 = wp::mod(var_0, var_10);
        var_13 = wp::address(var_dof_armature, var_12, var_1);
        var_15 = wp::load(var_13);
        wp::array_store(var_qM_out, var_0, var_14, var_3, var_15);
        // buf = math.inert_vec(crb_in[worldid, bodyid], cdof_in[worldid, dofid])                 <L 846>
        var_16 = wp::address(var_crb_in, var_0, var_6);
        var_17 = wp::address(var_cdof_in, var_0, var_1);
        var_19 = wp::load(var_16);
        var_20 = wp::load(var_17);
        var_18 = inert_vec_0(var_19, var_20);
        // while dofid >= 0:                                                                      <L 849>
        start_while_0:;
        var_22 = (var_1 >= var_21);
        if ((var_22) == false) goto end_while_0;
            // qM_out[worldid, 0, madr_ij] += wp.dot(cdof_in[worldid, dofid], buf)                <L 850>
            var_23 = wp::address(var_cdof_in, var_0, var_1);
            var_25 = wp::load(var_23);
            var_24 = wp::dot(var_25, var_18);
            var_27 = wp::atomic_add(var_qM_out, var_0, var_26, var_3, var_24);
            // madr_ij += 1                                                                       <L 851>
            var_29 = wp::add(var_3, var_28);
            // dofid = dof_parentid[dofid]                                                        <L 852>
            var_30 = wp::address(var_dof_parentid, var_1);
            var_32 = wp::load(var_30);
            var_31 = wp::copy(var_32);
            wp::assign(var_1, var_31);
            wp::assign(var_3, var_29);
        goto start_while_0;
        end_while_0:;
    }
}



extern "C" __global__ void _subtree_com_acc_350f0560_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::int32> var_body_tree_,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_out)
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
        bool var_9;
        wp::vec_t<3, wp::float32>* var_10;
        wp::vec_t<3, wp::float32> var_11;
        wp::vec_t<3, wp::float32> var_12;
        //---------
        // forward
        // def _subtree_com_acc(                                                                  <L 477>
        // worldid, nodeid = wp.tid()                                                             <L 487>
        builtin_tid2d(var_0, var_1);
        // bodyid = body_tree_[nodeid]                                                            <L 488>
        var_2 = wp::address(var_body_tree_, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // pid = body_parentid[bodyid]                                                            <L 489>
        var_5 = wp::address(var_body_parentid, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if bodyid != 0:                                                                        <L 490>
        var_9 = (var_3 != var_8);
        if (var_9) {
            // wp.atomic_add(subtree_com_out, worldid, pid, subtree_com_in[worldid, bodyid])       <L 491>
            var_10 = wp::address(var_subtree_com_in, var_0, var_3);
            var_12 = wp::load(var_10);
            var_11 = wp::atomic_add(var_subtree_com_out, var_0, var_6, var_12);
        }
    }
}



extern "C" __global__ void _qLD_acc_25842a75_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_M_rownnz,
    wp::array_t<wp::int32> var_M_rowadr,
    wp::array_t<wp::vec_t<3, wp::int32>> var_qLD_updates_,
    wp::array_t<wp::float32> var_L_in,
    wp::array_t<wp::float32> var_L_out)
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
        wp::vec_t<3, wp::int32>* var_2;
        wp::vec_t<3, wp::int32> var_3;
        wp::vec_t<3, wp::int32> var_4;
        const wp::int32 var_5 = 0;
        wp::int32 var_6;
        const wp::int32 var_7 = 1;
        wp::int32 var_8;
        const wp::int32 var_9 = 2;
        wp::int32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32* var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        const wp::int32 var_19 = 1;
        wp::int32 var_20;
        const wp::int32 var_21 = 0;
        wp::float32* var_22;
        const wp::int32 var_23 = 0;
        wp::float32* var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        wp::int32* var_28;
        wp::range_t var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        const wp::int32 var_32 = 0;
        wp::slice_t var_33;
        const wp::int32 var_34 = 0;
        wp::slice_t var_35;
        const wp::int32 var_36 = 0;
        wp::array_t<wp::float32> var_37;
        wp::int32 var_38;
        const wp::int32 var_39 = 0;
        wp::int32* var_40;
        wp::int32 var_41;
        wp::int32 var_42;
        wp::float32* var_43;
        wp::float32 var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        const wp::int32 var_47 = 0;
        //---------
        // forward
        // def _qLD_acc(                                                                          <L 1017>
        // worldid, nodeid = wp.tid()                                                             <L 1027>
        builtin_tid2d(var_0, var_1);
        // update = qLD_updates_[nodeid]                                                          <L 1028>
        var_2 = wp::address(var_qLD_updates_, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // i, k, Madr_ki = update[0], update[1], update[2]                                        <L 1029>
        var_6 = wp::extract(var_3, var_5);
        var_8 = wp::extract(var_3, var_7);
        var_10 = wp::extract(var_3, var_9);
        // Madr_i = M_rowadr[i]  # Address of row being updated                                   <L 1030>
        var_11 = wp::address(var_M_rowadr, var_6);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // diag_k = M_rowadr[k] + M_rownnz[k] - 1  # Address of diagonal element of k             <L 1031>
        var_14 = wp::address(var_M_rowadr, var_8);
        var_15 = wp::address(var_M_rownnz, var_8);
        var_17 = wp::load(var_14);
        var_18 = wp::load(var_15);
        var_16 = wp::add(var_17, var_18);
        var_20 = wp::sub(var_16, var_19);
        // tmp = L_out[worldid, 0, Madr_ki] / L_out[worldid, 0, diag_k]                           <L 1033>
        var_22 = wp::address(var_L_out, var_0, var_21, var_10);
        var_24 = wp::address(var_L_out, var_0, var_23, var_20);
        var_26 = wp::load(var_22);
        var_27 = wp::load(var_24);
        var_25 = wp::div(var_26, var_27);
        // for j in range(M_rownnz[i]):                                                           <L 1034>
        var_28 = wp::address(var_M_rownnz, var_6);
        var_30 = wp::load(var_28);
        var_29 = wp::range(var_30);
        start_for_0:;
            if (iter_cmp(var_29) == 0) goto end_for_0;
            var_31 = wp::iter_next(var_29);
            // wp.atomic_sub(L_out[worldid, 0], Madr_i + j, L_in[worldid, 0, M_rowadr[k] + j] * tmp)       <L 1036>
            var_33 = wp::slice_t(var_0, var_0, var_34);
            var_35 = wp::slice_t(var_32, var_32, var_36);
            var_37 = wp::view(var_L_out, var_33, var_35);
            var_38 = wp::add(var_12, var_31);
            var_40 = wp::address(var_M_rowadr, var_8);
            var_42 = wp::load(var_40);
            var_41 = wp::add(var_42, var_31);
            var_43 = wp::address(var_L_in, var_0, var_39, var_41);
            var_45 = wp::load(var_43);
            var_44 = wp::mul(var_45, var_25);
            var_46 = wp::atomic_sub(var_37, var_38, var_44);
            goto start_for_0;
        end_for_0:;
        // L_out[worldid, 0, Madr_ki] = tmp                                                       <L 1038>
        wp::array_store(var_L_out, var_0, var_47, var_10, var_25);
    }
}



extern "C" __global__ void _crb_accumulate_073a11b8_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::vec_t<10, wp::float32>> var_crb_in,
    wp::array_t<wp::int32> var_body_tree_,
    wp::array_t<wp::vec_t<10, wp::float32>> var_crb_out)
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
        bool var_9;
        wp::vec_t<10, wp::float32>* var_10;
        wp::vec_t<10, wp::float32> var_11;
        wp::vec_t<10, wp::float32> var_12;
        //---------
        // forward
        // def _crb_accumulate(                                                                   <L 807>
        // worldid, nodeid = wp.tid()                                                             <L 817>
        builtin_tid2d(var_0, var_1);
        // bodyid = body_tree_[nodeid]                                                            <L 818>
        var_2 = wp::address(var_body_tree_, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // pid = body_parentid[bodyid]                                                            <L 819>
        var_5 = wp::address(var_body_parentid, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if pid == 0:                                                                           <L 820>
        var_9 = (var_6 == var_8);
        if (var_9) {
            // return                                                                             <L 821>
            continue;
        }
        // wp.atomic_add(crb_out, worldid, pid, crb_in[worldid, bodyid])                          <L 822>
        var_10 = wp::address(var_crb_in, var_0, var_3);
        var_12 = wp::load(var_10);
        var_11 = wp::atomic_add(var_crb_out, var_0, var_6, var_12);
    }
}



extern "C" __global__ void _copy_CSR_acda4f42_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_mapM2M,
    wp::array_t<wp::float32> var_M_in,
    wp::array_t<wp::float32> var_L_out)
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
        wp::float32* var_4;
        wp::int32 var_5;
        const wp::int32 var_6 = 0;
        wp::float32 var_7;
        //---------
        // forward
        // def _copy_CSR(                                                                         <L 1004>
        // worldid, ind = wp.tid()                                                                <L 1012>
        builtin_tid2d(var_0, var_1);
        // L_out[worldid, 0, ind] = M_in[worldid, 0, mapM2M[ind]]                                 <L 1013>
        var_3 = wp::address(var_mapM2M, var_1);
        var_5 = wp::load(var_3);
        var_4 = wp::address(var_M_in, var_0, var_2, var_5);
        var_7 = wp::load(var_4);
        wp::array_store(var_L_out, var_0, var_6, var_1, var_7);
    }
}



extern "C" __global__ void _light_local_to_global_1966b84b_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_light_mode,
    wp::array_t<wp::int32> var_light_bodyid,
    wp::array_t<wp::int32> var_light_targetbodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_dir,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_poscom0,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_pos0,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_dir0,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_xpos_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_light_xdir_out)
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
        const wp::int32 var_13 = 3;
        bool var_14;
        wp::int32 var_15;
        wp::int32* var_16;
        const wp::int32 var_17 = 4;
        bool var_18;
        wp::int32 var_19;
        bool var_20;
        wp::int32* var_21;
        const wp::int32 var_22 = 0;
        bool var_23;
        wp::int32 var_24;
        bool var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::vec_t<3, wp::float32>* var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::quat_t<wp::float32>* var_32;
        wp::quat_t<wp::float32> var_33;
        wp::quat_t<wp::float32> var_34;
        wp::vec_t<3, wp::float32>* var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32> var_37;
        wp::vec_t<3, wp::float32> var_38;
        wp::vec_t<3, wp::float32>* var_39;
        wp::vec_t<3, wp::float32> var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::int32* var_42;
        const wp::int32 var_43 = 1;
        bool var_44;
        wp::int32 var_45;
        wp::shape_t* var_46;
        const wp::int32 var_47 = 0;
        wp::int32 var_48;
        wp::shape_t var_49;
        wp::int32 var_50;
        wp::vec_t<3, wp::float32>* var_51;
        wp::vec_t<3, wp::float32> var_52;
        wp::int32* var_53;
        wp::vec_t<3, wp::float32>* var_54;
        wp::int32 var_55;
        wp::vec_t<3, wp::float32> var_56;
        wp::vec_t<3, wp::float32> var_57;
        wp::shape_t* var_58;
        const wp::int32 var_59 = 0;
        wp::int32 var_60;
        wp::shape_t var_61;
        wp::int32 var_62;
        wp::vec_t<3, wp::float32>* var_63;
        wp::vec_t<3, wp::float32> var_64;
        wp::vec_t<3, wp::float32> var_65;
        wp::int32* var_66;
        const wp::int32 var_67 = 2;
        bool var_68;
        wp::int32 var_69;
        wp::shape_t* var_70;
        const wp::int32 var_71 = 0;
        wp::int32 var_72;
        wp::shape_t var_73;
        wp::int32 var_74;
        wp::vec_t<3, wp::float32>* var_75;
        wp::vec_t<3, wp::float32> var_76;
        wp::int32* var_77;
        wp::vec_t<3, wp::float32>* var_78;
        wp::int32 var_79;
        wp::shape_t* var_80;
        const wp::int32 var_81 = 0;
        wp::int32 var_82;
        wp::shape_t var_83;
        wp::int32 var_84;
        wp::vec_t<3, wp::float32>* var_85;
        wp::vec_t<3, wp::float32> var_86;
        wp::vec_t<3, wp::float32> var_87;
        wp::vec_t<3, wp::float32> var_88;
        wp::int32* var_89;
        const wp::int32 var_90 = 3;
        bool var_91;
        wp::int32 var_92;
        wp::int32* var_93;
        const wp::int32 var_94 = 4;
        bool var_95;
        wp::int32 var_96;
        bool var_97;
        wp::int32* var_98;
        wp::int32 var_99;
        wp::int32 var_100;
        wp::vec_t<3, wp::float32>* var_101;
        wp::vec_t<3, wp::float32> var_102;
        wp::vec_t<3, wp::float32> var_103;
        wp::quat_t<wp::float32>* var_104;
        wp::quat_t<wp::float32> var_105;
        wp::quat_t<wp::float32> var_106;
        wp::vec_t<3, wp::float32>* var_107;
        wp::vec_t<3, wp::float32> var_108;
        wp::vec_t<3, wp::float32> var_109;
        wp::vec_t<3, wp::float32> var_110;
        wp::int32* var_111;
        wp::vec_t<3, wp::float32>* var_112;
        wp::int32 var_113;
        wp::vec_t<3, wp::float32> var_114;
        wp::vec_t<3, wp::float32> var_115;
        wp::int32* var_116;
        const wp::int32 var_117 = 4;
        bool var_118;
        wp::int32 var_119;
        wp::int32* var_120;
        wp::vec_t<3, wp::float32>* var_121;
        wp::int32 var_122;
        wp::vec_t<3, wp::float32> var_123;
        wp::vec_t<3, wp::float32> var_124;
        wp::vec_t<3, wp::float32> var_125;
        wp::vec_t<3, wp::float32>* var_126;
        wp::vec_t<3, wp::float32> var_127;
        wp::vec_t<3, wp::float32> var_128;
        wp::int32 var_129;
        wp::vec_t<3, wp::float32> var_130;
        wp::quat_t<wp::float32> var_131;
        wp::int32* var_132;
        wp::int32 var_133;
        wp::int32 var_134;
        wp::vec_t<3, wp::float32>* var_135;
        wp::vec_t<3, wp::float32> var_136;
        wp::vec_t<3, wp::float32> var_137;
        wp::quat_t<wp::float32>* var_138;
        wp::quat_t<wp::float32> var_139;
        wp::quat_t<wp::float32> var_140;
        wp::vec_t<3, wp::float32>* var_141;
        wp::vec_t<3, wp::float32> var_142;
        wp::vec_t<3, wp::float32> var_143;
        wp::vec_t<3, wp::float32> var_144;
        wp::vec_t<3, wp::float32>* var_145;
        wp::vec_t<3, wp::float32> var_146;
        wp::vec_t<3, wp::float32> var_147;
        wp::int32 var_148;
        wp::vec_t<3, wp::float32> var_149;
        wp::quat_t<wp::float32> var_150;
        wp::int32 var_151;
        wp::vec_t<3, wp::float32> var_152;
        wp::quat_t<wp::float32> var_153;
        wp::int32 var_154;
        wp::vec_t<3, wp::float32> var_155;
        wp::quat_t<wp::float32> var_156;
        wp::int32 var_157;
        wp::vec_t<3, wp::float32> var_158;
        wp::quat_t<wp::float32> var_159;
        wp::vec_t<3, wp::float32>* var_160;
        wp::vec_t<3, wp::float32> var_161;
        wp::vec_t<3, wp::float32> var_162;
        //---------
        // forward
        // def _light_local_to_global(                                                            <L 703>
        // worldid, lightid = wp.tid()                                                            <L 721>
        builtin_tid2d(var_0, var_1);
        // light_pos_id = worldid % light_pos.shape[0]                                            <L 722>
        var_2 = &(var_light_pos.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // light_dir_id = worldid % light_dir.shape[0]                                            <L 723>
        var_7 = &(var_light_dir.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::mod(var_0, var_9);
        // is_target_light = (light_mode[lightid] == CamLightType.TARGETBODY) or (light_mode[lightid] == CamLightType.TARGETBODYCOM)       <L 724>
        var_12 = wp::address(var_light_mode, var_1);
        var_15 = wp::load(var_12);
        var_14 = (var_15 == var_13);
        var_16 = wp::address(var_light_mode, var_1);
        var_19 = wp::load(var_16);
        var_18 = (var_19 == var_17);
        var_20 = var_14 || var_18;
        // invalid_target = is_target_light and (light_targetbodyid[lightid] < 0)                 <L 725>
        var_21 = wp::address(var_light_targetbodyid, var_1);
        var_24 = wp::load(var_21);
        var_23 = (var_24 < var_22);
        var_25 = var_20 && var_23;
        // if invalid_target:                                                                     <L 726>
        if (var_25) {
            // bodyid = light_bodyid[lightid]                                                     <L 727>
            var_26 = wp::address(var_light_bodyid, var_1);
            var_28 = wp::load(var_26);
            var_27 = wp::copy(var_28);
            // xpos = xpos_in[worldid, bodyid]                                                    <L 728>
            var_29 = wp::address(var_xpos_in, var_0, var_27);
            var_31 = wp::load(var_29);
            var_30 = wp::copy(var_31);
            // xquat = xquat_in[worldid, bodyid]                                                  <L 729>
            var_32 = wp::address(var_xquat_in, var_0, var_27);
            var_34 = wp::load(var_32);
            var_33 = wp::copy(var_34);
            // light_xpos_out[worldid, lightid] = xpos + math.rot_vec_quat(light_pos[light_pos_id, lightid], xquat)       <L 730>
            var_35 = wp::address(var_light_pos, var_6, var_1);
            var_37 = wp::load(var_35);
            var_36 = rot_vec_quat_0(var_37, var_33);
            var_38 = wp::add(var_30, var_36);
            wp::array_store(var_light_xpos_out, var_0, var_1, var_38);
            // light_xdir_out[worldid, lightid] = math.rot_vec_quat(light_dir[light_dir_id, lightid], xquat)       <L 731>
            var_39 = wp::address(var_light_dir, var_11, var_1);
            var_41 = wp::load(var_39);
            var_40 = rot_vec_quat_0(var_41, var_33);
            wp::array_store(var_light_xdir_out, var_0, var_1, var_40);
            // return                                                                             <L 732>
            continue;
        }
        if (!var_25) {
            // elif light_mode[lightid] == CamLightType.TRACK:                                    <L 733>
            var_42 = wp::address(var_light_mode, var_1);
            var_45 = wp::load(var_42);
            var_44 = (var_45 == var_43);
            if (var_44) {
                // light_xdir_out[worldid, lightid] = light_dir0[worldid % light_dir0.shape[0], lightid]       <L 734>
                var_46 = &(var_light_dir0.shape);
                var_49 = wp::load(var_46);
                var_48 = wp::extract(var_49, var_47);
                var_50 = wp::mod(var_0, var_48);
                var_51 = wp::address(var_light_dir0, var_50, var_1);
                var_52 = wp::load(var_51);
                wp::array_store(var_light_xdir_out, var_0, var_1, var_52);
                // body_xpos = xpos_in[worldid, light_bodyid[lightid]]                            <L 735>
                var_53 = wp::address(var_light_bodyid, var_1);
                var_55 = wp::load(var_53);
                var_54 = wp::address(var_xpos_in, var_0, var_55);
                var_57 = wp::load(var_54);
                var_56 = wp::copy(var_57);
                // light_xpos_out[worldid, lightid] = body_xpos + light_pos0[worldid % light_pos0.shape[0], lightid]       <L 736>
                var_58 = &(var_light_pos0.shape);
                var_61 = wp::load(var_58);
                var_60 = wp::extract(var_61, var_59);
                var_62 = wp::mod(var_0, var_60);
                var_63 = wp::address(var_light_pos0, var_62, var_1);
                var_65 = wp::load(var_63);
                var_64 = wp::add(var_56, var_65);
                wp::array_store(var_light_xpos_out, var_0, var_1, var_64);
            }
            if (!var_44) {
                // elif light_mode[lightid] == CamLightType.TRACKCOM:                             <L 737>
                var_66 = wp::address(var_light_mode, var_1);
                var_69 = wp::load(var_66);
                var_68 = (var_69 == var_67);
                if (var_68) {
                    // light_xdir_out[worldid, lightid] = light_dir0[worldid % light_dir0.shape[0], lightid]       <L 738>
                    var_70 = &(var_light_dir0.shape);
                    var_73 = wp::load(var_70);
                    var_72 = wp::extract(var_73, var_71);
                    var_74 = wp::mod(var_0, var_72);
                    var_75 = wp::address(var_light_dir0, var_74, var_1);
                    var_76 = wp::load(var_75);
                    wp::array_store(var_light_xdir_out, var_0, var_1, var_76);
                    // light_xpos_out[worldid, lightid] = (                                       <L 739>
                    // subtree_com_in[worldid, light_bodyid[lightid]] + light_poscom0[worldid % light_poscom0.shape[0], lightid]       <L 740>
                    var_77 = wp::address(var_light_bodyid, var_1);
                    var_79 = wp::load(var_77);
                    var_78 = wp::address(var_subtree_com_in, var_0, var_79);
                    var_80 = &(var_light_poscom0.shape);
                    var_83 = wp::load(var_80);
                    var_82 = wp::extract(var_83, var_81);
                    var_84 = wp::mod(var_0, var_82);
                    var_85 = wp::address(var_light_poscom0, var_84, var_1);
                    var_87 = wp::load(var_78);
                    var_88 = wp::load(var_85);
                    var_86 = wp::add(var_87, var_88);
                    // light_xpos_out[worldid, lightid] = (                                       <L 739>
                    wp::array_store(var_light_xpos_out, var_0, var_1, var_86);
                }
                if (!var_68) {
                    // elif light_mode[lightid] == CamLightType.TARGETBODY or light_mode[lightid] == CamLightType.TARGETBODYCOM:       <L 742>
                    var_89 = wp::address(var_light_mode, var_1);
                    var_92 = wp::load(var_89);
                    var_91 = (var_92 == var_90);
                    var_93 = wp::address(var_light_mode, var_1);
                    var_96 = wp::load(var_93);
                    var_95 = (var_96 == var_94);
                    var_97 = var_91 || var_95;
                    if (var_97) {
                        // bodyid = light_bodyid[lightid]                                         <L 743>
                        var_98 = wp::address(var_light_bodyid, var_1);
                        var_100 = wp::load(var_98);
                        var_99 = wp::copy(var_100);
                        // xpos = xpos_in[worldid, bodyid]                                        <L 744>
                        var_101 = wp::address(var_xpos_in, var_0, var_99);
                        var_103 = wp::load(var_101);
                        var_102 = wp::copy(var_103);
                        // xquat = xquat_in[worldid, bodyid]                                      <L 745>
                        var_104 = wp::address(var_xquat_in, var_0, var_99);
                        var_106 = wp::load(var_104);
                        var_105 = wp::copy(var_106);
                        // light_xpos_out[worldid, lightid] = xpos + math.rot_vec_quat(light_pos[light_pos_id, lightid], xquat)       <L 746>
                        var_107 = wp::address(var_light_pos, var_6, var_1);
                        var_109 = wp::load(var_107);
                        var_108 = rot_vec_quat_0(var_109, var_105);
                        var_110 = wp::add(var_102, var_108);
                        wp::array_store(var_light_xpos_out, var_0, var_1, var_110);
                        // pos = xpos_in[worldid, light_targetbodyid[lightid]]                    <L 747>
                        var_111 = wp::address(var_light_targetbodyid, var_1);
                        var_113 = wp::load(var_111);
                        var_112 = wp::address(var_xpos_in, var_0, var_113);
                        var_115 = wp::load(var_112);
                        var_114 = wp::copy(var_115);
                        // if light_mode[lightid] == CamLightType.TARGETBODYCOM:                  <L 748>
                        var_116 = wp::address(var_light_mode, var_1);
                        var_119 = wp::load(var_116);
                        var_118 = (var_119 == var_117);
                        if (var_118) {
                            // pos = subtree_com_in[worldid, light_targetbodyid[lightid]]         <L 749>
                            var_120 = wp::address(var_light_targetbodyid, var_1);
                            var_122 = wp::load(var_120);
                            var_121 = wp::address(var_subtree_com_in, var_0, var_122);
                            var_124 = wp::load(var_121);
                            var_123 = wp::copy(var_124);
                        }
                        var_125 = wp::where(var_118, var_123, var_114);
                        // light_xdir_out[worldid, lightid] = pos - light_xpos_out[worldid, lightid]       <L 750>
                        var_126 = wp::address(var_light_xpos_out, var_0, var_1);
                        var_128 = wp::load(var_126);
                        var_127 = wp::sub(var_125, var_128);
                        wp::array_store(var_light_xdir_out, var_0, var_1, var_127);
                    }
                    var_129 = wp::where(var_97, var_99, var_27);
                    var_130 = wp::where(var_97, var_102, var_30);
                    var_131 = wp::where(var_97, var_105, var_33);
                    if (!var_97) {
                        // bodyid = light_bodyid[lightid]                                         <L 752>
                        var_132 = wp::address(var_light_bodyid, var_1);
                        var_134 = wp::load(var_132);
                        var_133 = wp::copy(var_134);
                        // xpos = xpos_in[worldid, bodyid]                                        <L 753>
                        var_135 = wp::address(var_xpos_in, var_0, var_133);
                        var_137 = wp::load(var_135);
                        var_136 = wp::copy(var_137);
                        // xquat = xquat_in[worldid, bodyid]                                      <L 754>
                        var_138 = wp::address(var_xquat_in, var_0, var_133);
                        var_140 = wp::load(var_138);
                        var_139 = wp::copy(var_140);
                        // light_xpos_out[worldid, lightid] = xpos + math.rot_vec_quat(light_pos[light_pos_id, lightid], xquat)       <L 755>
                        var_141 = wp::address(var_light_pos, var_6, var_1);
                        var_143 = wp::load(var_141);
                        var_142 = rot_vec_quat_0(var_143, var_139);
                        var_144 = wp::add(var_136, var_142);
                        wp::array_store(var_light_xpos_out, var_0, var_1, var_144);
                        // light_xdir_out[worldid, lightid] = math.rot_vec_quat(light_dir[light_dir_id, lightid], xquat)       <L 756>
                        var_145 = wp::address(var_light_dir, var_11, var_1);
                        var_147 = wp::load(var_145);
                        var_146 = rot_vec_quat_0(var_147, var_139);
                        wp::array_store(var_light_xdir_out, var_0, var_1, var_146);
                    }
                    var_148 = wp::where(var_97, var_129, var_133);
                    var_149 = wp::where(var_97, var_130, var_136);
                    var_150 = wp::where(var_97, var_131, var_139);
                }
                var_151 = wp::where(var_68, var_27, var_148);
                var_152 = wp::where(var_68, var_30, var_149);
                var_153 = wp::where(var_68, var_33, var_150);
            }
            var_154 = wp::where(var_44, var_27, var_151);
            var_155 = wp::where(var_44, var_30, var_152);
            var_156 = wp::where(var_44, var_33, var_153);
        }
        var_157 = wp::where(var_25, var_27, var_154);
        var_158 = wp::where(var_25, var_30, var_155);
        var_159 = wp::where(var_25, var_33, var_156);
        // light_xdir_out[worldid, lightid] = wp.normalize(light_xdir_out[worldid, lightid])       <L 758>
        var_160 = wp::address(var_light_xdir_out, var_0, var_1);
        var_162 = wp::load(var_160);
        var_161 = wp::normalize(var_162);
        wp::array_store(var_light_xdir_out, var_0, var_1, var_161);
    }
}



extern "C" __global__ void _cam_local_to_global_ed67ebfa_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_cam_mode,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::int32> var_cam_targetbodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_pos,
    wp::array_t<wp::quat_t<wp::float32>> var_cam_quat,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_poscom0,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_pos0,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_mat0,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_out)
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
        const wp::int32 var_13 = 3;
        bool var_14;
        wp::int32 var_15;
        wp::int32* var_16;
        const wp::int32 var_17 = 4;
        bool var_18;
        wp::int32 var_19;
        bool var_20;
        wp::int32* var_21;
        const wp::int32 var_22 = 0;
        bool var_23;
        wp::int32 var_24;
        bool var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::vec_t<3, wp::float32>* var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::quat_t<wp::float32>* var_32;
        wp::quat_t<wp::float32> var_33;
        wp::quat_t<wp::float32> var_34;
        wp::vec_t<3, wp::float32>* var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32> var_37;
        wp::vec_t<3, wp::float32> var_38;
        wp::quat_t<wp::float32>* var_39;
        wp::quat_t<wp::float32> var_40;
        wp::quat_t<wp::float32> var_41;
        wp::mat_t<3, 3, wp::float32> var_42;
        wp::int32* var_43;
        const wp::int32 var_44 = 1;
        bool var_45;
        wp::int32 var_46;
        wp::shape_t* var_47;
        const wp::int32 var_48 = 0;
        wp::int32 var_49;
        wp::shape_t var_50;
        wp::int32 var_51;
        wp::mat_t<3, 3, wp::float32>* var_52;
        wp::mat_t<3, 3, wp::float32> var_53;
        wp::int32* var_54;
        wp::vec_t<3, wp::float32>* var_55;
        wp::int32 var_56;
        wp::vec_t<3, wp::float32> var_57;
        wp::vec_t<3, wp::float32> var_58;
        wp::shape_t* var_59;
        const wp::int32 var_60 = 0;
        wp::int32 var_61;
        wp::shape_t var_62;
        wp::int32 var_63;
        wp::vec_t<3, wp::float32>* var_64;
        wp::vec_t<3, wp::float32> var_65;
        wp::vec_t<3, wp::float32> var_66;
        wp::int32* var_67;
        const wp::int32 var_68 = 2;
        bool var_69;
        wp::int32 var_70;
        wp::shape_t* var_71;
        const wp::int32 var_72 = 0;
        wp::int32 var_73;
        wp::shape_t var_74;
        wp::int32 var_75;
        wp::mat_t<3, 3, wp::float32>* var_76;
        wp::mat_t<3, 3, wp::float32> var_77;
        wp::int32* var_78;
        wp::vec_t<3, wp::float32>* var_79;
        wp::int32 var_80;
        wp::shape_t* var_81;
        const wp::int32 var_82 = 0;
        wp::int32 var_83;
        wp::shape_t var_84;
        wp::int32 var_85;
        wp::vec_t<3, wp::float32>* var_86;
        wp::vec_t<3, wp::float32> var_87;
        wp::vec_t<3, wp::float32> var_88;
        wp::vec_t<3, wp::float32> var_89;
        wp::int32* var_90;
        const wp::int32 var_91 = 3;
        bool var_92;
        wp::int32 var_93;
        wp::int32* var_94;
        const wp::int32 var_95 = 4;
        bool var_96;
        wp::int32 var_97;
        bool var_98;
        wp::int32* var_99;
        wp::int32 var_100;
        wp::int32 var_101;
        wp::vec_t<3, wp::float32>* var_102;
        wp::vec_t<3, wp::float32> var_103;
        wp::vec_t<3, wp::float32> var_104;
        wp::quat_t<wp::float32>* var_105;
        wp::quat_t<wp::float32> var_106;
        wp::quat_t<wp::float32> var_107;
        wp::vec_t<3, wp::float32>* var_108;
        wp::vec_t<3, wp::float32> var_109;
        wp::vec_t<3, wp::float32> var_110;
        wp::vec_t<3, wp::float32> var_111;
        wp::int32* var_112;
        wp::vec_t<3, wp::float32>* var_113;
        wp::int32 var_114;
        wp::vec_t<3, wp::float32> var_115;
        wp::vec_t<3, wp::float32> var_116;
        wp::int32* var_117;
        const wp::int32 var_118 = 4;
        bool var_119;
        wp::int32 var_120;
        wp::int32* var_121;
        wp::vec_t<3, wp::float32>* var_122;
        wp::int32 var_123;
        wp::vec_t<3, wp::float32> var_124;
        wp::vec_t<3, wp::float32> var_125;
        wp::vec_t<3, wp::float32> var_126;
        wp::vec_t<3, wp::float32>* var_127;
        wp::vec_t<3, wp::float32> var_128;
        wp::vec_t<3, wp::float32> var_129;
        wp::vec_t<3, wp::float32> var_130;
        const wp::float32 var_131 = 0.0;
        const wp::float32 var_132 = 0.0;
        const wp::float32 var_133 = 1.0;
        wp::vec_t<3, wp::float32> var_134;
        wp::vec_t<3, wp::float32> var_135;
        wp::vec_t<3, wp::float32> var_136;
        wp::vec_t<3, wp::float32> var_137;
        wp::vec_t<3, wp::float32> var_138;
        const wp::int32 var_139 = 0;
        wp::float32 var_140;
        const wp::int32 var_141 = 0;
        wp::float32 var_142;
        const wp::int32 var_143 = 0;
        wp::float32 var_144;
        const wp::int32 var_145 = 1;
        wp::float32 var_146;
        const wp::int32 var_147 = 1;
        wp::float32 var_148;
        const wp::int32 var_149 = 1;
        wp::float32 var_150;
        const wp::int32 var_151 = 2;
        wp::float32 var_152;
        const wp::int32 var_153 = 2;
        wp::float32 var_154;
        const wp::int32 var_155 = 2;
        wp::float32 var_156;
        wp::mat_t<3, 3, wp::float32> var_157;
        wp::int32 var_158;
        wp::vec_t<3, wp::float32> var_159;
        wp::quat_t<wp::float32> var_160;
        wp::int32* var_161;
        wp::int32 var_162;
        wp::int32 var_163;
        wp::vec_t<3, wp::float32>* var_164;
        wp::vec_t<3, wp::float32> var_165;
        wp::vec_t<3, wp::float32> var_166;
        wp::quat_t<wp::float32>* var_167;
        wp::quat_t<wp::float32> var_168;
        wp::quat_t<wp::float32> var_169;
        wp::vec_t<3, wp::float32>* var_170;
        wp::vec_t<3, wp::float32> var_171;
        wp::vec_t<3, wp::float32> var_172;
        wp::vec_t<3, wp::float32> var_173;
        wp::quat_t<wp::float32>* var_174;
        wp::quat_t<wp::float32> var_175;
        wp::quat_t<wp::float32> var_176;
        wp::mat_t<3, 3, wp::float32> var_177;
        wp::int32 var_178;
        wp::vec_t<3, wp::float32> var_179;
        wp::quat_t<wp::float32> var_180;
        wp::int32 var_181;
        wp::vec_t<3, wp::float32> var_182;
        wp::quat_t<wp::float32> var_183;
        wp::int32 var_184;
        wp::vec_t<3, wp::float32> var_185;
        wp::quat_t<wp::float32> var_186;
        wp::int32 var_187;
        wp::vec_t<3, wp::float32> var_188;
        wp::quat_t<wp::float32> var_189;
        //---------
        // forward
        // def _cam_local_to_global(                                                              <L 636>
        // worldid, camid = wp.tid()                                                              <L 654>
        builtin_tid2d(var_0, var_1);
        // cam_pos_id = worldid % cam_pos.shape[0]                                                <L 655>
        var_2 = &(var_cam_pos.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // cam_quat_id = worldid % cam_quat.shape[0]                                              <L 656>
        var_7 = &(var_cam_quat.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::mod(var_0, var_9);
        // is_target_cam = (cam_mode[camid] == CamLightType.TARGETBODY) or (cam_mode[camid] == CamLightType.TARGETBODYCOM)       <L 657>
        var_12 = wp::address(var_cam_mode, var_1);
        var_15 = wp::load(var_12);
        var_14 = (var_15 == var_13);
        var_16 = wp::address(var_cam_mode, var_1);
        var_19 = wp::load(var_16);
        var_18 = (var_19 == var_17);
        var_20 = var_14 || var_18;
        // invalid_target = is_target_cam and (cam_targetbodyid[camid] < 0)                       <L 658>
        var_21 = wp::address(var_cam_targetbodyid, var_1);
        var_24 = wp::load(var_21);
        var_23 = (var_24 < var_22);
        var_25 = var_20 && var_23;
        // if invalid_target:                                                                     <L 659>
        if (var_25) {
            // bodyid = cam_bodyid[camid]                                                         <L 660>
            var_26 = wp::address(var_cam_bodyid, var_1);
            var_28 = wp::load(var_26);
            var_27 = wp::copy(var_28);
            // xpos = xpos_in[worldid, bodyid]                                                    <L 661>
            var_29 = wp::address(var_xpos_in, var_0, var_27);
            var_31 = wp::load(var_29);
            var_30 = wp::copy(var_31);
            // xquat = xquat_in[worldid, bodyid]                                                  <L 662>
            var_32 = wp::address(var_xquat_in, var_0, var_27);
            var_34 = wp::load(var_32);
            var_33 = wp::copy(var_34);
            // cam_xpos_out[worldid, camid] = xpos + math.rot_vec_quat(cam_pos[cam_pos_id, camid], xquat)       <L 663>
            var_35 = wp::address(var_cam_pos, var_6, var_1);
            var_37 = wp::load(var_35);
            var_36 = rot_vec_quat_0(var_37, var_33);
            var_38 = wp::add(var_30, var_36);
            wp::array_store(var_cam_xpos_out, var_0, var_1, var_38);
            // cam_xmat_out[worldid, camid] = math.quat_to_mat(math.mul_quat(xquat, cam_quat[cam_quat_id, camid]))       <L 664>
            var_39 = wp::address(var_cam_quat, var_11, var_1);
            var_41 = wp::load(var_39);
            var_40 = mul_quat_0(var_33, var_41);
            var_42 = quat_to_mat_0(var_40);
            wp::array_store(var_cam_xmat_out, var_0, var_1, var_42);
        }
        if (!var_25) {
            // elif cam_mode[camid] == CamLightType.TRACK:                                        <L 665>
            var_43 = wp::address(var_cam_mode, var_1);
            var_46 = wp::load(var_43);
            var_45 = (var_46 == var_44);
            if (var_45) {
                // cam_xmat_out[worldid, camid] = cam_mat0[worldid % cam_mat0.shape[0], camid]       <L 666>
                var_47 = &(var_cam_mat0.shape);
                var_50 = wp::load(var_47);
                var_49 = wp::extract(var_50, var_48);
                var_51 = wp::mod(var_0, var_49);
                var_52 = wp::address(var_cam_mat0, var_51, var_1);
                var_53 = wp::load(var_52);
                wp::array_store(var_cam_xmat_out, var_0, var_1, var_53);
                // body_xpos = xpos_in[worldid, cam_bodyid[camid]]                                <L 667>
                var_54 = wp::address(var_cam_bodyid, var_1);
                var_56 = wp::load(var_54);
                var_55 = wp::address(var_xpos_in, var_0, var_56);
                var_58 = wp::load(var_55);
                var_57 = wp::copy(var_58);
                // cam_xpos_out[worldid, camid] = body_xpos + cam_pos0[worldid % cam_pos0.shape[0], camid]       <L 668>
                var_59 = &(var_cam_pos0.shape);
                var_62 = wp::load(var_59);
                var_61 = wp::extract(var_62, var_60);
                var_63 = wp::mod(var_0, var_61);
                var_64 = wp::address(var_cam_pos0, var_63, var_1);
                var_66 = wp::load(var_64);
                var_65 = wp::add(var_57, var_66);
                wp::array_store(var_cam_xpos_out, var_0, var_1, var_65);
            }
            if (!var_45) {
                // elif cam_mode[camid] == CamLightType.TRACKCOM:                                 <L 669>
                var_67 = wp::address(var_cam_mode, var_1);
                var_70 = wp::load(var_67);
                var_69 = (var_70 == var_68);
                if (var_69) {
                    // cam_xmat_out[worldid, camid] = cam_mat0[worldid % cam_mat0.shape[0], camid]       <L 670>
                    var_71 = &(var_cam_mat0.shape);
                    var_74 = wp::load(var_71);
                    var_73 = wp::extract(var_74, var_72);
                    var_75 = wp::mod(var_0, var_73);
                    var_76 = wp::address(var_cam_mat0, var_75, var_1);
                    var_77 = wp::load(var_76);
                    wp::array_store(var_cam_xmat_out, var_0, var_1, var_77);
                    // cam_xpos_out[worldid, camid] = (                                           <L 671>
                    // subtree_com_in[worldid, cam_bodyid[camid]] + cam_poscom0[worldid % cam_poscom0.shape[0], camid]       <L 672>
                    var_78 = wp::address(var_cam_bodyid, var_1);
                    var_80 = wp::load(var_78);
                    var_79 = wp::address(var_subtree_com_in, var_0, var_80);
                    var_81 = &(var_cam_poscom0.shape);
                    var_84 = wp::load(var_81);
                    var_83 = wp::extract(var_84, var_82);
                    var_85 = wp::mod(var_0, var_83);
                    var_86 = wp::address(var_cam_poscom0, var_85, var_1);
                    var_88 = wp::load(var_79);
                    var_89 = wp::load(var_86);
                    var_87 = wp::add(var_88, var_89);
                    // cam_xpos_out[worldid, camid] = (                                           <L 671>
                    wp::array_store(var_cam_xpos_out, var_0, var_1, var_87);
                }
                if (!var_69) {
                    // elif cam_mode[camid] == CamLightType.TARGETBODY or cam_mode[camid] == CamLightType.TARGETBODYCOM:       <L 674>
                    var_90 = wp::address(var_cam_mode, var_1);
                    var_93 = wp::load(var_90);
                    var_92 = (var_93 == var_91);
                    var_94 = wp::address(var_cam_mode, var_1);
                    var_97 = wp::load(var_94);
                    var_96 = (var_97 == var_95);
                    var_98 = var_92 || var_96;
                    if (var_98) {
                        // bodyid = cam_bodyid[camid]                                             <L 675>
                        var_99 = wp::address(var_cam_bodyid, var_1);
                        var_101 = wp::load(var_99);
                        var_100 = wp::copy(var_101);
                        // xpos = xpos_in[worldid, bodyid]                                        <L 676>
                        var_102 = wp::address(var_xpos_in, var_0, var_100);
                        var_104 = wp::load(var_102);
                        var_103 = wp::copy(var_104);
                        // xquat = xquat_in[worldid, bodyid]                                      <L 677>
                        var_105 = wp::address(var_xquat_in, var_0, var_100);
                        var_107 = wp::load(var_105);
                        var_106 = wp::copy(var_107);
                        // cam_xpos_out[worldid, camid] = xpos + math.rot_vec_quat(cam_pos[cam_pos_id, camid], xquat)       <L 678>
                        var_108 = wp::address(var_cam_pos, var_6, var_1);
                        var_110 = wp::load(var_108);
                        var_109 = rot_vec_quat_0(var_110, var_106);
                        var_111 = wp::add(var_103, var_109);
                        wp::array_store(var_cam_xpos_out, var_0, var_1, var_111);
                        // pos = xpos_in[worldid, cam_targetbodyid[camid]]                        <L 679>
                        var_112 = wp::address(var_cam_targetbodyid, var_1);
                        var_114 = wp::load(var_112);
                        var_113 = wp::address(var_xpos_in, var_0, var_114);
                        var_116 = wp::load(var_113);
                        var_115 = wp::copy(var_116);
                        // if cam_mode[camid] == CamLightType.TARGETBODYCOM:                      <L 680>
                        var_117 = wp::address(var_cam_mode, var_1);
                        var_120 = wp::load(var_117);
                        var_119 = (var_120 == var_118);
                        if (var_119) {
                            // pos = subtree_com_in[worldid, cam_targetbodyid[camid]]             <L 681>
                            var_121 = wp::address(var_cam_targetbodyid, var_1);
                            var_123 = wp::load(var_121);
                            var_122 = wp::address(var_subtree_com_in, var_0, var_123);
                            var_125 = wp::load(var_122);
                            var_124 = wp::copy(var_125);
                        }
                        var_126 = wp::where(var_119, var_124, var_115);
                        // mat_3 = wp.normalize(cam_xpos_out[worldid, camid] - pos)               <L 683>
                        var_127 = wp::address(var_cam_xpos_out, var_0, var_1);
                        var_129 = wp::load(var_127);
                        var_128 = wp::sub(var_129, var_126);
                        var_130 = wp::normalize(var_128);
                        // mat_1 = wp.normalize(wp.cross(wp.vec3(0.0, 0.0, 1.0), mat_3))          <L 685>
                        var_134 = wp::vec_t<3, wp::float32>(var_131, var_132, var_133);
                        var_135 = wp::cross(var_134, var_130);
                        var_136 = wp::normalize(var_135);
                        // mat_2 = wp.normalize(wp.cross(mat_3, mat_1))                           <L 686>
                        var_137 = wp::cross(var_130, var_136);
                        var_138 = wp::normalize(var_137);
                        // cam_xmat_out[worldid, camid] = wp.mat33(                               <L 688>
                        // mat_1[0], mat_2[0], mat_3[0],                                          <L 689>
                        var_140 = wp::extract(var_136, var_139);
                        var_142 = wp::extract(var_138, var_141);
                        var_144 = wp::extract(var_130, var_143);
                        // mat_1[1], mat_2[1], mat_3[1],                                          <L 690>
                        var_146 = wp::extract(var_136, var_145);
                        var_148 = wp::extract(var_138, var_147);
                        var_150 = wp::extract(var_130, var_149);
                        // mat_1[2], mat_2[2], mat_3[2]                                           <L 691>
                        var_152 = wp::extract(var_136, var_151);
                        var_154 = wp::extract(var_138, var_153);
                        var_156 = wp::extract(var_130, var_155);
                        var_157 = wp::mat_t<3, 3, wp::float32>(var_140, var_142, var_144, var_146, var_148, var_150, var_152, var_154, var_156);
                        // cam_xmat_out[worldid, camid] = wp.mat33(                               <L 688>
                        wp::array_store(var_cam_xmat_out, var_0, var_1, var_157);
                    }
                    var_158 = wp::where(var_98, var_100, var_27);
                    var_159 = wp::where(var_98, var_103, var_30);
                    var_160 = wp::where(var_98, var_106, var_33);
                    if (!var_98) {
                        // bodyid = cam_bodyid[camid]                                             <L 695>
                        var_161 = wp::address(var_cam_bodyid, var_1);
                        var_163 = wp::load(var_161);
                        var_162 = wp::copy(var_163);
                        // xpos = xpos_in[worldid, bodyid]                                        <L 696>
                        var_164 = wp::address(var_xpos_in, var_0, var_162);
                        var_166 = wp::load(var_164);
                        var_165 = wp::copy(var_166);
                        // xquat = xquat_in[worldid, bodyid]                                      <L 697>
                        var_167 = wp::address(var_xquat_in, var_0, var_162);
                        var_169 = wp::load(var_167);
                        var_168 = wp::copy(var_169);
                        // cam_xpos_out[worldid, camid] = xpos + math.rot_vec_quat(cam_pos[cam_pos_id, camid], xquat)       <L 698>
                        var_170 = wp::address(var_cam_pos, var_6, var_1);
                        var_172 = wp::load(var_170);
                        var_171 = rot_vec_quat_0(var_172, var_168);
                        var_173 = wp::add(var_165, var_171);
                        wp::array_store(var_cam_xpos_out, var_0, var_1, var_173);
                        // cam_xmat_out[worldid, camid] = math.quat_to_mat(math.mul_quat(xquat, cam_quat[cam_quat_id, camid]))       <L 699>
                        var_174 = wp::address(var_cam_quat, var_11, var_1);
                        var_176 = wp::load(var_174);
                        var_175 = mul_quat_0(var_168, var_176);
                        var_177 = quat_to_mat_0(var_175);
                        wp::array_store(var_cam_xmat_out, var_0, var_1, var_177);
                    }
                    var_178 = wp::where(var_98, var_158, var_162);
                    var_179 = wp::where(var_98, var_159, var_165);
                    var_180 = wp::where(var_98, var_160, var_168);
                }
                var_181 = wp::where(var_69, var_27, var_178);
                var_182 = wp::where(var_69, var_30, var_179);
                var_183 = wp::where(var_69, var_33, var_180);
            }
            var_184 = wp::where(var_45, var_27, var_181);
            var_185 = wp::where(var_45, var_30, var_182);
            var_186 = wp::where(var_45, var_33, var_183);
        }
        var_187 = wp::where(var_25, var_27, var_184);
        var_188 = wp::where(var_25, var_30, var_185);
        var_189 = wp::where(var_25, var_33, var_186);
    }
}



extern "C" __global__ void _qM_dense_747f9fc5_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::int32> var_dof_parentid,
    wp::array_t<wp::float32> var_dof_armature,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<10, wp::float32>> var_crb_in,
    wp::array_t<wp::float32> var_qM_out)
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
        wp::vec_t<10, wp::float32>* var_13;
        wp::vec_t<6, wp::float32>* var_14;
        wp::vec_t<6, wp::float32> var_15;
        wp::vec_t<10, wp::float32> var_16;
        wp::vec_t<6, wp::float32> var_17;
        wp::vec_t<6, wp::float32>* var_18;
        wp::float32 var_19;
        wp::vec_t<6, wp::float32> var_20;
        wp::float32 var_21;
        wp::int32 var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        const wp::int32 var_26 = 0;
        bool var_27;
        wp::vec_t<6, wp::float32>* var_28;
        wp::float32 var_29;
        wp::vec_t<6, wp::float32> var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        wp::int32* var_33;
        wp::int32 var_34;
        wp::int32 var_35;
        //---------
        // forward
        // def _qM_dense(                                                                         <L 856>
        // worldid, dofid = wp.tid()                                                              <L 867>
        builtin_tid2d(var_0, var_1);
        // bodyid = dof_bodyid[dofid]                                                             <L 868>
        var_2 = wp::address(var_dof_bodyid, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // M = dof_armature[worldid % dof_armature.shape[0], dofid]                               <L 870>
        var_5 = &(var_dof_armature.shape);
        var_8 = wp::load(var_5);
        var_7 = wp::extract(var_8, var_6);
        var_9 = wp::mod(var_0, var_7);
        var_10 = wp::address(var_dof_armature, var_9, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // buf = math.inert_vec(crb_in[worldid, bodyid], cdof_in[worldid, dofid])                 <L 873>
        var_13 = wp::address(var_crb_in, var_0, var_3);
        var_14 = wp::address(var_cdof_in, var_0, var_1);
        var_16 = wp::load(var_13);
        var_17 = wp::load(var_14);
        var_15 = inert_vec_0(var_16, var_17);
        // M += wp.dot(cdof_in[worldid, dofid], buf)                                              <L 874>
        var_18 = wp::address(var_cdof_in, var_0, var_1);
        var_20 = wp::load(var_18);
        var_19 = wp::dot(var_20, var_15);
        var_21 = wp::add(var_11, var_19);
        // qM_out[worldid, dofid, dofid] = M                                                      <L 876>
        wp::array_store(var_qM_out, var_0, var_1, var_1, var_21);
        // dofidi = dofid                                                                         <L 879>
        var_22 = wp::copy(var_1);
        // dofid = dof_parentid[dofid]                                                            <L 880>
        var_23 = wp::address(var_dof_parentid, var_1);
        var_25 = wp::load(var_23);
        var_24 = wp::copy(var_25);
        // while dofid >= 0:                                                                      <L 881>
        start_while_0:;
        var_27 = (var_24 >= var_26);
        if ((var_27) == false) goto end_while_0;
            // qMij = wp.dot(cdof_in[worldid, dofid], buf)                                        <L 882>
            var_28 = wp::address(var_cdof_in, var_0, var_24);
            var_30 = wp::load(var_28);
            var_29 = wp::dot(var_30, var_15);
            // qM_out[worldid, dofidi, dofid] += qMij                                             <L 883>
            var_31 = wp::atomic_add(var_qM_out, var_0, var_22, var_24, var_29);
            // qM_out[worldid, dofid, dofidi] += qMij                                             <L 884>
            var_32 = wp::atomic_add(var_qM_out, var_0, var_24, var_22, var_29);
            // dofid = dof_parentid[dofid]                                                        <L 885>
            var_33 = wp::address(var_dof_parentid, var_24);
            var_35 = wp::load(var_33);
            var_34 = wp::copy(var_35);
            wp::assign(var_24, var_34);
        goto start_while_0;
        end_while_0:;
    }
}



extern "C" __global__ void _qLDiag_div_3267bf6b_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_M_rownnz,
    wp::array_t<wp::int32> var_M_rowadr,
    wp::array_t<wp::float32> var_L_in,
    wp::array_t<wp::float32> var_D_out)
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
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        const wp::int32 var_7 = 1;
        wp::int32 var_8;
        const wp::float32 var_9 = 1.0;
        const wp::int32 var_10 = 0;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        //---------
        // forward
        // def _qLDiag_div(                                                                       <L 1042>
        // worldid, dofid = wp.tid()                                                              <L 1051>
        builtin_tid2d(var_0, var_1);
        // diag_i = M_rowadr[dofid] + M_rownnz[dofid] - 1  # Address of diagonal element of i       <L 1052>
        var_2 = wp::address(var_M_rowadr, var_1);
        var_3 = wp::address(var_M_rownnz, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = wp::add(var_5, var_6);
        var_8 = wp::sub(var_4, var_7);
        // D_out[worldid, dofid] = 1.0 / L_in[worldid, 0, diag_i]                                 <L 1053>
        var_11 = wp::address(var_L_in, var_0, var_10, var_8);
        var_13 = wp::load(var_11);
        var_12 = wp::div(var_9, var_13);
        wp::array_store(var_D_out, var_0, var_1, var_12);
    }
}



extern "C" __global__ void _subtree_div_0ef57c26_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_body_subtreemass,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_out)
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
        wp::vec_t<3, wp::float32>* var_2;
        wp::vec_t<3, wp::float32> var_3;
        wp::vec_t<3, wp::float32> var_4;
        wp::shape_t* var_5;
        const wp::int32 var_6 = 0;
        wp::int32 var_7;
        wp::shape_t var_8;
        wp::int32 var_9;
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        const wp::float32 var_13 = 0.0;
        bool var_14;
        wp::vec_t<3, wp::float32> var_15;
        //---------
        // forward
        // def _subtree_div(                                                                      <L 495>
        // worldid, bodyid = wp.tid()                                                             <L 503>
        builtin_tid2d(var_0, var_1);
        // com = subtree_com_in[worldid, bodyid]                                                  <L 504>
        var_2 = wp::address(var_subtree_com_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // mass = body_subtreemass[worldid % body_subtreemass.shape[0], bodyid]                   <L 505>
        var_5 = &(var_body_subtreemass.shape);
        var_8 = wp::load(var_5);
        var_7 = wp::extract(var_8, var_6);
        var_9 = wp::mod(var_0, var_7);
        var_10 = wp::address(var_body_subtreemass, var_9, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // if mass != 0.0:                                                                        <L 506>
        var_14 = (var_11 != var_13);
        if (var_14) {
            // subtree_com_out[worldid, bodyid] = com / mass                                      <L 507>
            var_15 = wp::div(var_3, var_11);
            wp::array_store(var_subtree_com_out, var_0, var_1, var_15);
        }
    }
}



extern "C" __global__ void _cfrc_ext_contact_a1976729_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_opt_cone,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::int32> var_contact_worldid_in,
    wp::array_t<wp::float32> var_efc_force_in,
    wp::int32 var_njmax_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_ext_out)
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
        wp::int32* var_2;
        bool var_3;
        wp::int32 var_4;
        wp::vec_t<2, wp::int32>* var_5;
        wp::vec_t<2, wp::int32> var_6;
        wp::vec_t<2, wp::int32> var_7;
        const wp::int32 var_8 = 0;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 1;
        wp::int32 var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        const wp::int32 var_18 = 0;
        bool var_19;
        const wp::int32 var_20 = 0;
        bool var_21;
        bool var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        const bool var_26 = true;
        wp::vec_t<6, wp::float32> var_27;
        wp::vec_t<3, wp::float32>* var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::int32* var_31;
        wp::vec_t<3, wp::float32>* var_32;
        wp::int32 var_33;
        wp::vec_t<3, wp::float32> var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::slice_t var_36;
        const wp::int32 var_37 = 0;
        wp::array_t<wp::vec_t<6, wp::float32>> var_38;
        wp::vec_t<3, wp::float32> var_39;
        wp::vec_t<6, wp::float32> var_40;
        wp::vec_t<6, wp::float32> var_41;
        wp::int32* var_42;
        wp::vec_t<3, wp::float32>* var_43;
        wp::int32 var_44;
        wp::vec_t<3, wp::float32> var_45;
        wp::vec_t<3, wp::float32> var_46;
        wp::slice_t var_47;
        const wp::int32 var_48 = 0;
        wp::array_t<wp::vec_t<6, wp::float32>> var_49;
        wp::vec_t<3, wp::float32> var_50;
        wp::vec_t<6, wp::float32> var_51;
        wp::vec_t<6, wp::float32> var_52;
        //---------
        // forward
        // def _cfrc_ext_contact(                                                                 <L 1439>
        // contactid = wp.tid()                                                                   <L 1459>
        var_0 = builtin_tid1d();
        // if contactid >= nacon_in[0]:                                                           <L 1461>
        var_2 = wp::address(var_nacon_in, var_1);
        var_4 = wp::load(var_2);
        var_3 = (var_0 >= var_4);
        if (var_3) {
            // return                                                                             <L 1462>
            continue;
        }
        // geom = contact_geom_in[contactid]                                                      <L 1464>
        var_5 = wp::address(var_contact_geom_in, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // id1 = geom_bodyid[geom[0]]                                                             <L 1465>
        var_9 = wp::extract(var_6, var_8);
        var_10 = wp::address(var_geom_bodyid, var_9);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // id2 = geom_bodyid[geom[1]]                                                             <L 1466>
        var_14 = wp::extract(var_6, var_13);
        var_15 = wp::address(var_geom_bodyid, var_14);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // if id1 == 0 and id2 == 0:                                                              <L 1468>
        var_19 = (var_11 == var_18);
        var_21 = (var_16 == var_20);
        var_22 = var_19 && var_21;
        if (var_22) {
            // return                                                                             <L 1469>
            continue;
        }
        // worldid = contact_worldid_in[contactid]                                                <L 1471>
        var_23 = wp::address(var_contact_worldid_in, var_0);
        var_25 = wp::load(var_23);
        var_24 = wp::copy(var_25);
        // force = support.contact_force_fn(                                                      <L 1474>
        // opt_cone,                                                                              <L 1475>
        // contact_frame_in,                                                                      <L 1476>
        // contact_friction_in,                                                                   <L 1477>
        // contact_dim_in,                                                                        <L 1478>
        // contact_efc_address_in,                                                                <L 1479>
        // efc_force_in,                                                                          <L 1480>
        // njmax_in,                                                                              <L 1481>
        // nacon_in,                                                                              <L 1482>
        // worldid,                                                                               <L 1483>
        // contactid,                                                                             <L 1484>
        // to_world_frame=True,                                                                   <L 1485>
        var_27 = contact_force_fn_0(var_opt_cone, var_contact_frame_in, var_contact_friction_in, var_contact_dim_in, var_contact_efc_address_in, var_efc_force_in, var_njmax_in, var_nacon_in, var_24, var_0, var_26);
        // pos = contact_pos_in[contactid]                                                        <L 1488>
        var_28 = wp::address(var_contact_pos_in, var_0);
        var_30 = wp::load(var_28);
        var_29 = wp::copy(var_30);
        // if id1:                                                                                <L 1491>
        if (var_11) {
            // com1 = subtree_com_in[worldid, body_rootid[id1]]                                   <L 1492>
            var_31 = wp::address(var_body_rootid, var_11);
            var_33 = wp::load(var_31);
            var_32 = wp::address(var_subtree_com_in, var_24, var_33);
            var_35 = wp::load(var_32);
            var_34 = wp::copy(var_35);
            // wp.atomic_sub(cfrc_ext_out[worldid], id1, support.transform_force(force, com1 - pos))       <L 1493>
            var_36 = wp::slice_t(var_24, var_24, var_37);
            var_38 = wp::view(var_cfrc_ext_out, var_36);
            var_39 = wp::sub(var_34, var_29);
            var_40 = transform_force_1(var_27, var_39);
            var_41 = wp::atomic_sub(var_38, var_11, var_40);
        }
        // if id2:                                                                                <L 1495>
        if (var_16) {
            // com2 = subtree_com_in[worldid, body_rootid[id2]]                                   <L 1496>
            var_42 = wp::address(var_body_rootid, var_16);
            var_44 = wp::load(var_42);
            var_43 = wp::address(var_subtree_com_in, var_24, var_44);
            var_46 = wp::load(var_43);
            var_45 = wp::copy(var_46);
            // wp.atomic_add(cfrc_ext_out[worldid], id2, support.transform_force(force, com2 - pos))       <L 1497>
            var_47 = wp::slice_t(var_24, var_24, var_48);
            var_49 = wp::view(var_cfrc_ext_out, var_47);
            var_50 = wp::sub(var_45, var_29);
            var_51 = transform_force_1(var_27, var_50);
            var_52 = wp::atomic_add(var_49, var_16, var_51);
        }
    }
}



extern "C" __global__ void _spatial_geom_tendon_1ca37386_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::int32> var_wrap_type,
    wp::array_t<wp::int32> var_wrap_objid,
    wp::array_t<wp::float32> var_wrap_prm,
    wp::array_t<wp::int32> var_tendon_geom_adr,
    wp::array_t<wp::int32> var_wrap_geom_adr,
    wp::array_t<wp::float32> var_wrap_pulley_scale,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::float32> var_ten_J_out,
    wp::array_t<wp::float32> var_ten_length_out,
    wp::array_t<wp::vec_t<6, wp::float32>> var_wrap_geom_xpos_out)
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
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        const wp::int32 var_11 = 1;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        const wp::int32 var_16 = 0;
        wp::int32 var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        const wp::int32 var_21 = 1;
        wp::int32 var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::vec_t<3, wp::float32>* var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32>* var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::vec_t<3, wp::float32>* var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::vec_t<3, wp::float32> var_34;
        wp::mat_t<3, 3, wp::float32>* var_35;
        wp::mat_t<3, 3, wp::float32> var_36;
        wp::mat_t<3, 3, wp::float32> var_37;
        wp::shape_t* var_38;
        const wp::int32 var_39 = 0;
        wp::int32 var_40;
        wp::shape_t var_41;
        wp::int32 var_42;
        wp::vec_t<3, wp::float32>* var_43;
        const wp::int32 var_44 = 0;
        wp::float32 var_45;
        wp::vec_t<3, wp::float32> var_46;
        wp::int32* var_47;
        wp::int32 var_48;
        wp::int32 var_49;
        wp::int32* var_50;
        wp::int32 var_51;
        wp::int32 var_52;
        wp::int32* var_53;
        wp::int32 var_54;
        wp::int32 var_55;
        wp::int32* var_56;
        wp::int32 var_57;
        wp::int32 var_58;
        wp::float32* var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::int32 var_62;
        const wp::int32 var_63 = 0;
        bool var_64;
        wp::vec_t<3, wp::float32>* var_65;
        wp::vec_t<3, wp::float32> var_66;
        wp::vec_t<3, wp::float32> var_67;
        const wp::float32 var_68 = 10000000000.0;
        wp::vec_t<3, wp::float32> var_69;
        wp::vec_t<3, wp::float32> var_70;
        wp::float32 var_71;
        wp::vec_t<3, wp::float32> var_72;
        wp::vec_t<3, wp::float32> var_73;
        wp::vec_t<6, wp::float32> var_74;
        wp::int32* var_75;
        wp::int32 var_76;
        wp::int32 var_77;
        wp::int32* var_78;
        wp::int32 var_79;
        wp::int32 var_80;
        const wp::float32 var_81 = 0.0;
        bool var_82;
        wp::vec_t<3, wp::float32> var_83;
        wp::vec_t<3, wp::float32> var_84;
        wp::vec_t<3, wp::float32> var_85;
        wp::float32 var_86;
        wp::vec_t<3, wp::float32> var_87;
        wp::float32 var_88;
        wp::float32 var_89;
        wp::float32 var_90;
        wp::slice_t var_91;
        const wp::int32 var_92 = 0;
        wp::array_t<wp::float32> var_93;
        wp::float32 var_94;
        wp::float32 var_95;
        const wp::float32 var_96 = 1e-15;
        bool var_97;
        const wp::float32 var_98 = 1.0;
        const wp::float32 var_99 = 0.0;
        const wp::float32 var_100 = 0.0;
        wp::vec_t<3, wp::float32> var_101;
        wp::vec_t<3, wp::float32> var_102;
        bool var_103;
        const wp::float32 var_104 = 1.0;
        const wp::float32 var_105 = 0.0;
        const wp::float32 var_106 = 0.0;
        wp::vec_t<3, wp::float32> var_107;
        wp::vec_t<3, wp::float32> var_108;
        bool var_109;
        bool var_110;
        wp::int32* var_111;
        wp::vec_t<3, wp::float32>* var_112;
        wp::int32 var_113;
        wp::vec_t<3, wp::float32> var_114;
        wp::vec_t<3, wp::float32> var_115;
        wp::int32* var_116;
        wp::vec_t<3, wp::float32>* var_117;
        wp::int32 var_118;
        wp::vec_t<3, wp::float32> var_119;
        wp::vec_t<3, wp::float32> var_120;
        wp::float32 var_121;
        wp::int32* var_122;
        wp::vec_t<3, wp::float32>* var_123;
        wp::int32 var_124;
        wp::vec_t<3, wp::float32> var_125;
        wp::vec_t<3, wp::float32> var_126;
        wp::int32* var_127;
        wp::vec_t<3, wp::float32>* var_128;
        wp::int32 var_129;
        wp::vec_t<3, wp::float32> var_130;
        wp::vec_t<3, wp::float32> var_131;
        wp::float32 var_132;
        wp::vec_t<3, wp::float32> var_133;
        wp::vec_t<3, wp::float32> var_134;
        wp::float32 var_135;
        wp::slice_t var_136;
        const wp::int32 var_137 = 0;
        wp::array_t<wp::float32> var_138;
        wp::float32 var_139;
        wp::float32 var_140;
        bool var_141;
        const wp::float32 var_142 = 1.0;
        const wp::float32 var_143 = 0.0;
        const wp::float32 var_144 = 0.0;
        wp::vec_t<3, wp::float32> var_145;
        wp::vec_t<3, wp::float32> var_146;
        bool var_147;
        wp::int32* var_148;
        wp::vec_t<3, wp::float32>* var_149;
        wp::int32 var_150;
        wp::vec_t<3, wp::float32> var_151;
        wp::vec_t<3, wp::float32> var_152;
        wp::int32* var_153;
        wp::vec_t<3, wp::float32>* var_154;
        wp::int32 var_155;
        wp::vec_t<3, wp::float32> var_156;
        wp::vec_t<3, wp::float32> var_157;
        wp::float32 var_158;
        wp::vec_t<3, wp::float32> var_159;
        wp::vec_t<3, wp::float32> var_160;
        wp::vec_t<3, wp::float32> var_161;
        wp::vec_t<3, wp::float32> var_162;
        //---------
        // forward
        // def _spatial_geom_tendon(                                                              <L 3256>
        // worldid, elementid = wp.tid()                                                          <L 3286>
        builtin_tid2d(var_0, var_1);
        // wrap_adr = wrap_geom_adr[elementid]                                                    <L 3287>
        var_2 = wp::address(var_wrap_geom_adr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // tenid = tendon_geom_adr[elementid]                                                     <L 3288>
        var_5 = wp::address(var_tendon_geom_adr, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // pulley_scale = wrap_pulley_scale[wrap_adr]                                             <L 3291>
        var_8 = wp::address(var_wrap_pulley_scale, var_3);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // wrap_objid_site0 = wrap_objid[wrap_adr - 1]                                            <L 3294>
        var_12 = wp::sub(var_3, var_11);
        var_13 = wp::address(var_wrap_objid, var_12);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // wrap_objid_geom = wrap_objid[wrap_adr + 0]                                             <L 3295>
        var_17 = wp::add(var_3, var_16);
        var_18 = wp::address(var_wrap_objid, var_17);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // wrap_objid_site1 = wrap_objid[wrap_adr + 1]                                            <L 3296>
        var_22 = wp::add(var_3, var_21);
        var_23 = wp::address(var_wrap_objid, var_22);
        var_25 = wp::load(var_23);
        var_24 = wp::copy(var_25);
        // site_pnt0 = site_xpos_in[worldid, wrap_objid_site0]                                    <L 3299>
        var_26 = wp::address(var_site_xpos_in, var_0, var_14);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // site_pnt1 = site_xpos_in[worldid, wrap_objid_site1]                                    <L 3300>
        var_29 = wp::address(var_site_xpos_in, var_0, var_24);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // geom_xpos = geom_xpos_in[worldid, wrap_objid_geom]                                     <L 3303>
        var_32 = wp::address(var_geom_xpos_in, var_0, var_19);
        var_34 = wp::load(var_32);
        var_33 = wp::copy(var_34);
        // geom_xmat = geom_xmat_in[worldid, wrap_objid_geom]                                     <L 3304>
        var_35 = wp::address(var_geom_xmat_in, var_0, var_19);
        var_37 = wp::load(var_35);
        var_36 = wp::copy(var_37);
        // geomsize = geom_size[worldid % geom_size.shape[0], wrap_objid_geom][0]                 <L 3305>
        var_38 = &(var_geom_size.shape);
        var_41 = wp::load(var_38);
        var_40 = wp::extract(var_41, var_39);
        var_42 = wp::mod(var_0, var_40);
        var_43 = wp::address(var_geom_size, var_42, var_19);
        var_46 = wp::load(var_43);
        var_45 = wp::extract(var_46, var_44);
        // geom_type = wrap_type[wrap_adr]                                                        <L 3306>
        var_47 = wp::address(var_wrap_type, var_3);
        var_49 = wp::load(var_47);
        var_48 = wp::copy(var_49);
        // bodyid_site0 = site_bodyid[wrap_objid_site0]                                           <L 3309>
        var_50 = wp::address(var_site_bodyid, var_14);
        var_52 = wp::load(var_50);
        var_51 = wp::copy(var_52);
        // bodyid_geom = geom_bodyid[wrap_objid_geom]                                             <L 3310>
        var_53 = wp::address(var_geom_bodyid, var_19);
        var_55 = wp::load(var_53);
        var_54 = wp::copy(var_55);
        // bodyid_site1 = site_bodyid[wrap_objid_site1]                                           <L 3311>
        var_56 = wp::address(var_site_bodyid, var_24);
        var_58 = wp::load(var_56);
        var_57 = wp::copy(var_58);
        // sideid = int(wp.round(wrap_prm[wrap_adr]))                                             <L 3314>
        var_59 = wp::address(var_wrap_prm, var_3);
        var_61 = wp::load(var_59);
        var_60 = wp::round(var_61);
        var_62 = wp::int(var_60);
        // if sideid >= 0:                                                                        <L 3315>
        var_64 = (var_62 >= var_63);
        if (var_64) {
            // side = site_xpos_in[worldid, sideid]                                               <L 3316>
            var_65 = wp::address(var_site_xpos_in, var_0, var_62);
            var_67 = wp::load(var_65);
            var_66 = wp::copy(var_67);
        }
        if (!var_64) {
            // side = wp.vec3(MJ_MAXVAL)                                                          <L 3318>
            var_69 = wp::vec_t<3, wp::float32>(var_68);
        }
        var_70 = wp::where(var_64, var_66, var_69);
        // length_geomgeom, geom_pnt0, geom_pnt1 = util_misc.wrap(site_pnt0, site_pnt1, geom_xpos, geom_xmat, geomsize, geom_type, side)       <L 3321>
        wrap_0(var_27, var_30, var_33, var_36, var_45, var_48, var_70, var_71, var_72, var_73);
        // wrap_geom_xpos_out[worldid, elementid] = wp.spatial_vector(geom_pnt0, geom_pnt1)       <L 3324>
        var_74 = wp::vec_t<6, wp::float32>(var_72, var_73);
        wp::array_store(var_wrap_geom_xpos_out, var_0, var_1, var_74);
        // rownnz = ten_J_rownnz[tenid]                                                           <L 3326>
        var_75 = wp::address(var_ten_J_rownnz, var_6);
        var_77 = wp::load(var_75);
        var_76 = wp::copy(var_77);
        // rowadr = ten_J_rowadr[tenid]                                                           <L 3327>
        var_78 = wp::address(var_ten_J_rowadr, var_6);
        var_80 = wp::load(var_78);
        var_79 = wp::copy(var_80);
        // if length_geomgeom >= 0.0:                                                             <L 3329>
        var_82 = (var_71 >= var_81);
        if (var_82) {
            // dif_sitegeom = geom_pnt0 - site_pnt0                                               <L 3330>
            var_83 = wp::sub(var_72, var_27);
            // dif_geomsite = site_pnt1 - geom_pnt1                                               <L 3331>
            var_84 = wp::sub(var_30, var_73);
            // vec_sitegeom, length_sitegeom = math.normalize_with_norm(dif_sitegeom)             <L 3332>
            normalize_with_norm_0(var_83, var_85, var_86);
            // vec_geomsite, length_geomsite = math.normalize_with_norm(dif_geomsite)             <L 3333>
            normalize_with_norm_0(var_84, var_87, var_88);
            // length_sitegeomsite = length_sitegeom + length_geomgeom + length_geomsite          <L 3336>
            var_89 = wp::add(var_86, var_71);
            var_90 = wp::add(var_89, var_88);
            // if length_sitegeomsite:                                                            <L 3338>
            if (var_90) {
                // wp.atomic_add(ten_length_out[worldid], tenid, length_sitegeomsite * pulley_scale)       <L 3339>
                var_91 = wp::slice_t(var_0, var_0, var_92);
                var_93 = wp::view(var_ten_length_out, var_91);
                var_94 = wp::mul(var_90, var_9);
                var_95 = wp::atomic_add(var_93, var_6, var_94);
            }
            // if length_sitegeom < MJ_MINVAL:                                                    <L 3342>
            var_97 = (var_86 < var_96);
            if (var_97) {
                // vec_sitegeom = wp.vec3(1.0, 0.0, 0.0)                                          <L 3343>
                var_101 = wp::vec_t<3, wp::float32>(var_98, var_99, var_100);
            }
            var_102 = wp::where(var_97, var_101, var_85);
            // if length_geomsite < MJ_MINVAL:                                                    <L 3345>
            var_103 = (var_88 < var_96);
            if (var_103) {
                // vec_geomsite = wp.vec3(1.0, 0.0, 0.0)                                          <L 3346>
                var_107 = wp::vec_t<3, wp::float32>(var_104, var_105, var_106);
            }
            var_108 = wp::where(var_103, var_107, var_87);
            // dif_body_sitegeom = bodyid_site0 != bodyid_geom                                    <L 3348>
            var_109 = (var_51 != var_54);
            // dif_body_geomsite = bodyid_geom != bodyid_site1                                    <L 3349>
            var_110 = (var_54 != var_57);
            // if dif_body_sitegeom:                                                              <L 3352>
            if (var_109) {
                // offset_site0 = site_pnt0 - subtree_com_in[worldid, body_rootid[bodyid_site0]]       <L 3353>
                var_111 = wp::address(var_body_rootid, var_51);
                var_113 = wp::load(var_111);
                var_112 = wp::address(var_subtree_com_in, var_0, var_113);
                var_115 = wp::load(var_112);
                var_114 = wp::sub(var_27, var_115);
                // offset_geom0 = geom_pnt0 - subtree_com_in[worldid, body_rootid[bodyid_geom]]       <L 3354>
                var_116 = wp::address(var_body_rootid, var_54);
                var_118 = wp::load(var_116);
                var_117 = wp::address(var_subtree_com_in, var_0, var_118);
                var_120 = wp::load(var_117);
                var_119 = wp::sub(var_72, var_120);
                // _accumulate_jac_chain(                                                         <L 3355>
                // body_parentid,                                                                 <L 3356>
                // body_dofnum,                                                                   <L 3357>
                // body_dofadr,                                                                   <L 3358>
                // ten_J_colind,                                                                  <L 3359>
                // cdof_in,                                                                       <L 3360>
                // offset_site0,                                                                  <L 3361>
                // vec_sitegeom,                                                                  <L 3362>
                // bodyid_site0,                                                                  <L 3363>
                // rowadr,                                                                        <L 3364>
                // rownnz,                                                                        <L 3365>
                // -pulley_scale,                                                                 <L 3366>
                var_121 = wp::neg(var_9);
                // worldid,                                                                       <L 3367>
                // ten_J_out,                                                                     <L 3368>
                _accumulate_jac_chain_0(var_body_parentid, var_body_dofnum, var_body_dofadr, var_ten_J_colind, var_cdof_in, var_114, var_102, var_51, var_79, var_76, var_121, var_0, var_ten_J_out);
                // _accumulate_jac_chain(                                                         <L 3370>
                // body_parentid,                                                                 <L 3371>
                // body_dofnum,                                                                   <L 3372>
                // body_dofadr,                                                                   <L 3373>
                // ten_J_colind,                                                                  <L 3374>
                // cdof_in,                                                                       <L 3375>
                // offset_geom0,                                                                  <L 3376>
                // vec_sitegeom,                                                                  <L 3377>
                // bodyid_geom,                                                                   <L 3378>
                // rowadr,                                                                        <L 3379>
                // rownnz,                                                                        <L 3380>
                // pulley_scale,                                                                  <L 3381>
                // worldid,                                                                       <L 3382>
                // ten_J_out,                                                                     <L 3383>
                _accumulate_jac_chain_0(var_body_parentid, var_body_dofnum, var_body_dofadr, var_ten_J_colind, var_cdof_in, var_119, var_102, var_54, var_79, var_76, var_9, var_0, var_ten_J_out);
            }
            // if dif_body_geomsite:                                                              <L 3387>
            if (var_110) {
                // offset_geom1 = geom_pnt1 - subtree_com_in[worldid, body_rootid[bodyid_geom]]       <L 3388>
                var_122 = wp::address(var_body_rootid, var_54);
                var_124 = wp::load(var_122);
                var_123 = wp::address(var_subtree_com_in, var_0, var_124);
                var_126 = wp::load(var_123);
                var_125 = wp::sub(var_73, var_126);
                // offset_site1 = site_pnt1 - subtree_com_in[worldid, body_rootid[bodyid_site1]]       <L 3389>
                var_127 = wp::address(var_body_rootid, var_57);
                var_129 = wp::load(var_127);
                var_128 = wp::address(var_subtree_com_in, var_0, var_129);
                var_131 = wp::load(var_128);
                var_130 = wp::sub(var_30, var_131);
                // _accumulate_jac_chain(                                                         <L 3390>
                // body_parentid,                                                                 <L 3391>
                // body_dofnum,                                                                   <L 3392>
                // body_dofadr,                                                                   <L 3393>
                // ten_J_colind,                                                                  <L 3394>
                // cdof_in,                                                                       <L 3395>
                // offset_geom1,                                                                  <L 3396>
                // vec_geomsite,                                                                  <L 3397>
                // bodyid_geom,                                                                   <L 3398>
                // rowadr,                                                                        <L 3399>
                // rownnz,                                                                        <L 3400>
                // -pulley_scale,                                                                 <L 3401>
                var_132 = wp::neg(var_9);
                // worldid,                                                                       <L 3402>
                // ten_J_out,                                                                     <L 3403>
                _accumulate_jac_chain_0(var_body_parentid, var_body_dofnum, var_body_dofadr, var_ten_J_colind, var_cdof_in, var_125, var_108, var_54, var_79, var_76, var_132, var_0, var_ten_J_out);
                // _accumulate_jac_chain(                                                         <L 3405>
                // body_parentid,                                                                 <L 3406>
                // body_dofnum,                                                                   <L 3407>
                // body_dofadr,                                                                   <L 3408>
                // ten_J_colind,                                                                  <L 3409>
                // cdof_in,                                                                       <L 3410>
                // offset_site1,                                                                  <L 3411>
                // vec_geomsite,                                                                  <L 3412>
                // bodyid_site1,                                                                  <L 3413>
                // rowadr,                                                                        <L 3414>
                // rownnz,                                                                        <L 3415>
                // pulley_scale,                                                                  <L 3416>
                // worldid,                                                                       <L 3417>
                // ten_J_out,                                                                     <L 3418>
                _accumulate_jac_chain_0(var_body_parentid, var_body_dofnum, var_body_dofadr, var_ten_J_colind, var_cdof_in, var_130, var_108, var_57, var_79, var_76, var_9, var_0, var_ten_J_out);
            }
        }
        if (!var_82) {
            // dif_sitesite = site_pnt1 - site_pnt0                                               <L 3421>
            var_133 = wp::sub(var_30, var_27);
            // vec_sitesite, length_sitesite = math.normalize_with_norm(dif_sitesite)             <L 3422>
            normalize_with_norm_0(var_133, var_134, var_135);
            // if length_sitesite:                                                                <L 3425>
            if (var_135) {
                // wp.atomic_add(ten_length_out[worldid], tenid, length_sitesite * pulley_scale)       <L 3426>
                var_136 = wp::slice_t(var_0, var_0, var_137);
                var_138 = wp::view(var_ten_length_out, var_136);
                var_139 = wp::mul(var_135, var_9);
                var_140 = wp::atomic_add(var_138, var_6, var_139);
            }
            // if length_sitesite < MJ_MINVAL:                                                    <L 3429>
            var_141 = (var_135 < var_96);
            if (var_141) {
                // vec_sitesite = wp.vec3(1.0, 0.0, 0.0)                                          <L 3430>
                var_145 = wp::vec_t<3, wp::float32>(var_142, var_143, var_144);
            }
            var_146 = wp::where(var_141, var_145, var_134);
            // if bodyid_site0 != bodyid_site1:                                                   <L 3432>
            var_147 = (var_51 != var_57);
            if (var_147) {
                // offset_site0 = site_pnt0 - subtree_com_in[worldid, body_rootid[bodyid_site0]]       <L 3433>
                var_148 = wp::address(var_body_rootid, var_51);
                var_150 = wp::load(var_148);
                var_149 = wp::address(var_subtree_com_in, var_0, var_150);
                var_152 = wp::load(var_149);
                var_151 = wp::sub(var_27, var_152);
                // offset_site1 = site_pnt1 - subtree_com_in[worldid, body_rootid[bodyid_site1]]       <L 3434>
                var_153 = wp::address(var_body_rootid, var_57);
                var_155 = wp::load(var_153);
                var_154 = wp::address(var_subtree_com_in, var_0, var_155);
                var_157 = wp::load(var_154);
                var_156 = wp::sub(var_30, var_157);
                // _accumulate_jac_chain(                                                         <L 3435>
                // body_parentid,                                                                 <L 3436>
                // body_dofnum,                                                                   <L 3437>
                // body_dofadr,                                                                   <L 3438>
                // ten_J_colind,                                                                  <L 3439>
                // cdof_in,                                                                       <L 3440>
                // offset_site0,                                                                  <L 3441>
                // vec_sitesite,                                                                  <L 3442>
                // bodyid_site0,                                                                  <L 3443>
                // rowadr,                                                                        <L 3444>
                // rownnz,                                                                        <L 3445>
                // -pulley_scale,                                                                 <L 3446>
                var_158 = wp::neg(var_9);
                // worldid,                                                                       <L 3447>
                // ten_J_out,                                                                     <L 3448>
                _accumulate_jac_chain_0(var_body_parentid, var_body_dofnum, var_body_dofadr, var_ten_J_colind, var_cdof_in, var_151, var_146, var_51, var_79, var_76, var_158, var_0, var_ten_J_out);
                // _accumulate_jac_chain(                                                         <L 3450>
                // body_parentid,                                                                 <L 3451>
                // body_dofnum,                                                                   <L 3452>
                // body_dofadr,                                                                   <L 3453>
                // ten_J_colind,                                                                  <L 3454>
                // cdof_in,                                                                       <L 3455>
                // offset_site1,                                                                  <L 3456>
                // vec_sitesite,                                                                  <L 3457>
                // bodyid_site1,                                                                  <L 3458>
                // rowadr,                                                                        <L 3459>
                // rownnz,                                                                        <L 3460>
                // pulley_scale,                                                                  <L 3461>
                // worldid,                                                                       <L 3462>
                // ten_J_out,                                                                     <L 3463>
                _accumulate_jac_chain_0(var_body_parentid, var_body_dofnum, var_body_dofadr, var_ten_J_colind, var_cdof_in, var_156, var_146, var_57, var_79, var_76, var_9, var_0, var_ten_J_out);
            }
            var_159 = wp::where(var_147, var_151, var_114);
            var_160 = wp::where(var_147, var_156, var_130);
        }
        var_161 = wp::where(var_82, var_114, var_159);
        var_162 = wp::where(var_82, var_130, var_160);
    }
}



extern "C" __global__ void _cinert_1261260c_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::float32> var_body_mass,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_inertia,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<10, wp::float32>> var_cinert_out)
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
        wp::mat_t<3, 3, wp::float32>* var_2;
        wp::mat_t<3, 3, wp::float32> var_3;
        wp::mat_t<3, 3, wp::float32> var_4;
        wp::shape_t* var_5;
        const wp::int32 var_6 = 0;
        wp::int32 var_7;
        wp::shape_t var_8;
        wp::int32 var_9;
        wp::vec_t<3, wp::float32>* var_10;
        wp::vec_t<3, wp::float32> var_11;
        wp::vec_t<3, wp::float32> var_12;
        wp::shape_t* var_13;
        const wp::int32 var_14 = 0;
        wp::int32 var_15;
        wp::shape_t var_16;
        wp::int32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::int32* var_22;
        wp::vec_t<3, wp::float32>* var_23;
        wp::int32 var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<10, wp::float32> var_28;
        wp::mat_t<3, 3, wp::float32> var_29;
        wp::mat_t<3, 3, wp::float32> var_30;
        wp::mat_t<3, 3, wp::float32> var_31;
        wp::mat_t<3, 3, wp::float32> var_32;
        const wp::int32 var_33 = 0;
        const wp::int32 var_34 = 0;
        wp::float32 var_35;
        const wp::int32 var_36 = 0;
        const wp::int32 var_37 = 1;
        const wp::int32 var_38 = 1;
        wp::float32 var_39;
        const wp::int32 var_40 = 1;
        const wp::int32 var_41 = 2;
        const wp::int32 var_42 = 2;
        wp::float32 var_43;
        const wp::int32 var_44 = 2;
        const wp::int32 var_45 = 0;
        const wp::int32 var_46 = 1;
        wp::float32 var_47;
        const wp::int32 var_48 = 3;
        const wp::int32 var_49 = 0;
        const wp::int32 var_50 = 2;
        wp::float32 var_51;
        const wp::int32 var_52 = 4;
        const wp::int32 var_53 = 1;
        const wp::int32 var_54 = 2;
        wp::float32 var_55;
        const wp::int32 var_56 = 5;
        const wp::int32 var_57 = 1;
        wp::float32 var_58;
        const wp::int32 var_59 = 1;
        wp::float32 var_60;
        wp::float32 var_61;
        const wp::int32 var_62 = 2;
        wp::float32 var_63;
        const wp::int32 var_64 = 2;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        wp::float32 var_68;
        const wp::int32 var_69 = 0;
        const wp::int32 var_70 = 0;
        wp::float32 var_71;
        const wp::int32 var_72 = 0;
        wp::float32 var_73;
        wp::float32 var_74;
        const wp::int32 var_75 = 2;
        wp::float32 var_76;
        const wp::int32 var_77 = 2;
        wp::float32 var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        const wp::int32 var_82 = 1;
        const wp::int32 var_83 = 0;
        wp::float32 var_84;
        const wp::int32 var_85 = 0;
        wp::float32 var_86;
        wp::float32 var_87;
        const wp::int32 var_88 = 1;
        wp::float32 var_89;
        const wp::int32 var_90 = 1;
        wp::float32 var_91;
        wp::float32 var_92;
        wp::float32 var_93;
        wp::float32 var_94;
        const wp::int32 var_95 = 2;
        const wp::int32 var_96 = 0;
        wp::float32 var_97;
        wp::float32 var_98;
        const wp::int32 var_99 = 1;
        wp::float32 var_100;
        wp::float32 var_101;
        const wp::int32 var_102 = 3;
        const wp::int32 var_103 = 0;
        wp::float32 var_104;
        wp::float32 var_105;
        const wp::int32 var_106 = 2;
        wp::float32 var_107;
        wp::float32 var_108;
        const wp::int32 var_109 = 4;
        const wp::int32 var_110 = 1;
        wp::float32 var_111;
        wp::float32 var_112;
        const wp::int32 var_113 = 2;
        wp::float32 var_114;
        wp::float32 var_115;
        const wp::int32 var_116 = 5;
        const wp::int32 var_117 = 0;
        wp::float32 var_118;
        wp::float32 var_119;
        const wp::int32 var_120 = 6;
        const wp::int32 var_121 = 1;
        wp::float32 var_122;
        wp::float32 var_123;
        const wp::int32 var_124 = 7;
        const wp::int32 var_125 = 2;
        wp::float32 var_126;
        wp::float32 var_127;
        const wp::int32 var_128 = 8;
        const wp::int32 var_129 = 9;
        //---------
        // forward
        // def _cinert(                                                                           <L 511>
        // worldid, bodyid = wp.tid()                                                             <L 523>
        builtin_tid2d(var_0, var_1);
        // mat = ximat_in[worldid, bodyid]                                                        <L 524>
        var_2 = wp::address(var_ximat_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // inert = body_inertia[worldid % body_inertia.shape[0], bodyid]                          <L 525>
        var_5 = &(var_body_inertia.shape);
        var_8 = wp::load(var_5);
        var_7 = wp::extract(var_8, var_6);
        var_9 = wp::mod(var_0, var_7);
        var_10 = wp::address(var_body_inertia, var_9, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // mass = body_mass[worldid % body_mass.shape[0], bodyid]                                 <L 526>
        var_13 = &(var_body_mass.shape);
        var_16 = wp::load(var_13);
        var_15 = wp::extract(var_16, var_14);
        var_17 = wp::mod(var_0, var_15);
        var_18 = wp::address(var_body_mass, var_17, var_1);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // dif = xipos_in[worldid, bodyid] - subtree_com_in[worldid, body_rootid[bodyid]]         <L 527>
        var_21 = wp::address(var_xipos_in, var_0, var_1);
        var_22 = wp::address(var_body_rootid, var_1);
        var_24 = wp::load(var_22);
        var_23 = wp::address(var_subtree_com_in, var_0, var_24);
        var_26 = wp::load(var_21);
        var_27 = wp::load(var_23);
        var_25 = wp::sub(var_26, var_27);
        // res = vec10()                                                                          <L 530>
        var_28 = wp::vec_t<10, wp::float32>();
        // tmp = mat @ wp.diag(inert) @ wp.transpose(mat)                                         <L 532>
        var_29 = wp::diag(var_11);
        var_30 = wp::mul(var_3, var_29);
        var_31 = wp::transpose(var_3);
        var_32 = wp::mul(var_30, var_31);
        // res[0] = tmp[0, 0]                                                                     <L 533>
        var_35 = wp::extract(var_32, var_33, var_34);
        wp::assign_inplace(var_28, var_36, var_35);
        // res[1] = tmp[1, 1]                                                                     <L 534>
        var_39 = wp::extract(var_32, var_37, var_38);
        wp::assign_inplace(var_28, var_40, var_39);
        // res[2] = tmp[2, 2]                                                                     <L 535>
        var_43 = wp::extract(var_32, var_41, var_42);
        wp::assign_inplace(var_28, var_44, var_43);
        // res[3] = tmp[0, 1]                                                                     <L 536>
        var_47 = wp::extract(var_32, var_45, var_46);
        wp::assign_inplace(var_28, var_48, var_47);
        // res[4] = tmp[0, 2]                                                                     <L 537>
        var_51 = wp::extract(var_32, var_49, var_50);
        wp::assign_inplace(var_28, var_52, var_51);
        // res[5] = tmp[1, 2]                                                                     <L 538>
        var_55 = wp::extract(var_32, var_53, var_54);
        wp::assign_inplace(var_28, var_56, var_55);
        // res[0] += mass * (dif[1] * dif[1] + dif[2] * dif[2])                                   <L 540>
        var_58 = wp::extract(var_25, var_57);
        var_60 = wp::extract(var_25, var_59);
        var_61 = wp::mul(var_58, var_60);
        var_63 = wp::extract(var_25, var_62);
        var_65 = wp::extract(var_25, var_64);
        var_66 = wp::mul(var_63, var_65);
        var_67 = wp::add(var_61, var_66);
        var_68 = wp::mul(var_19, var_67);
        wp::add_inplace(var_28, var_69, var_68);
        // res[1] += mass * (dif[0] * dif[0] + dif[2] * dif[2])                                   <L 541>
        var_71 = wp::extract(var_25, var_70);
        var_73 = wp::extract(var_25, var_72);
        var_74 = wp::mul(var_71, var_73);
        var_76 = wp::extract(var_25, var_75);
        var_78 = wp::extract(var_25, var_77);
        var_79 = wp::mul(var_76, var_78);
        var_80 = wp::add(var_74, var_79);
        var_81 = wp::mul(var_19, var_80);
        wp::add_inplace(var_28, var_82, var_81);
        // res[2] += mass * (dif[0] * dif[0] + dif[1] * dif[1])                                   <L 542>
        var_84 = wp::extract(var_25, var_83);
        var_86 = wp::extract(var_25, var_85);
        var_87 = wp::mul(var_84, var_86);
        var_89 = wp::extract(var_25, var_88);
        var_91 = wp::extract(var_25, var_90);
        var_92 = wp::mul(var_89, var_91);
        var_93 = wp::add(var_87, var_92);
        var_94 = wp::mul(var_19, var_93);
        wp::add_inplace(var_28, var_95, var_94);
        // res[3] -= mass * dif[0] * dif[1]                                                       <L 543>
        var_97 = wp::extract(var_25, var_96);
        var_98 = wp::mul(var_19, var_97);
        var_100 = wp::extract(var_25, var_99);
        var_101 = wp::mul(var_98, var_100);
        wp::sub_inplace(var_28, var_102, var_101);
        // res[4] -= mass * dif[0] * dif[2]                                                       <L 544>
        var_104 = wp::extract(var_25, var_103);
        var_105 = wp::mul(var_19, var_104);
        var_107 = wp::extract(var_25, var_106);
        var_108 = wp::mul(var_105, var_107);
        wp::sub_inplace(var_28, var_109, var_108);
        // res[5] -= mass * dif[1] * dif[2]                                                       <L 545>
        var_111 = wp::extract(var_25, var_110);
        var_112 = wp::mul(var_19, var_111);
        var_114 = wp::extract(var_25, var_113);
        var_115 = wp::mul(var_112, var_114);
        wp::sub_inplace(var_28, var_116, var_115);
        // res[6] = mass * dif[0]                                                                 <L 547>
        var_118 = wp::extract(var_25, var_117);
        var_119 = wp::mul(var_19, var_118);
        wp::assign_inplace(var_28, var_120, var_119);
        // res[7] = mass * dif[1]                                                                 <L 548>
        var_122 = wp::extract(var_25, var_121);
        var_123 = wp::mul(var_19, var_122);
        wp::assign_inplace(var_28, var_124, var_123);
        // res[8] = mass * dif[2]                                                                 <L 549>
        var_126 = wp::extract(var_25, var_125);
        var_127 = wp::mul(var_19, var_126);
        wp::assign_inplace(var_28, var_128, var_127);
        // res[9] = mass                                                                          <L 551>
        wp::assign_inplace(var_28, var_129, var_19);
        // cinert_out[worldid, bodyid] = res                                                      <L 553>
        wp::array_store(var_cinert_out, var_0, var_1, var_28);
    }
}



extern "C" __global__ void _spatial_tendon_wrap_2bb6c667_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_ntendon,
    wp::array_t<wp::int32> var_tendon_adr,
    wp::array_t<wp::int32> var_tendon_num,
    wp::array_t<wp::int32> var_wrap_type,
    wp::array_t<wp::int32> var_wrap_objid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_wrap_geom_xpos_in,
    wp::array_t<wp::int32> var_ten_wrapadr_out,
    wp::array_t<wp::int32> var_ten_wrapnum_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_wrap_obj_out,
    wp::array_t<wp::vec_t<6, wp::float32>> var_wrap_xpos_out)
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
        wp::int32 var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::range_t var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        const wp::int32 var_10 = 0;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::int32* var_15;
        const wp::int32 var_16 = 1;
        bool var_17;
        wp::int32 var_18;
        const wp::int32 var_19 = 0;
        wp::int32 var_20;
        const wp::int32 var_21 = 1;
        wp::int32 var_22;
        bool var_23;
        wp::int32 var_24;
        const wp::int32 var_25 = 0;
        wp::int32 var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::int32 var_30;
        const wp::int32 var_31 = 1;
        wp::int32 var_32;
        wp::int32* var_33;
        wp::int32 var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        const wp::int32 var_37 = 0;
        wp::int32 var_38;
        wp::int32* var_39;
        wp::int32 var_40;
        wp::int32 var_41;
        wp::int32 var_42;
        const wp::int32 var_43 = 1;
        wp::int32 var_44;
        wp::int32* var_45;
        wp::int32 var_46;
        wp::int32 var_47;
        const wp::int32 var_48 = 2;
        bool var_49;
        const wp::int32 var_50 = 2;
        bool var_51;
        bool var_52;
        const wp::int32 var_53 = 2;
        wp::int32 var_54;
        const wp::int32 var_55 = 2;
        wp::int32 var_56;
        const wp::float32 var_57 = 0.0;
        wp::vec_t<6, wp::float32>* var_58;
        const wp::int32 var_59 = 3;
        wp::int32 var_60;
        const wp::int32 var_61 = 0;
        wp::int32 var_62;
        wp::float32* var_63;
        const wp::float32 var_64 = 0.0;
        wp::vec_t<6, wp::float32>* var_65;
        const wp::int32 var_66 = 3;
        wp::int32 var_67;
        const wp::int32 var_68 = 1;
        wp::int32 var_69;
        wp::float32* var_70;
        const wp::float32 var_71 = 0.0;
        wp::vec_t<6, wp::float32>* var_72;
        const wp::int32 var_73 = 3;
        wp::int32 var_74;
        const wp::int32 var_75 = 2;
        wp::int32 var_76;
        wp::float32* var_77;
        const wp::int32 var_78 = 2;
        const wp::int32 var_79 = -2;
        wp::vec_t<2, wp::int32>* var_80;
        wp::int32* var_81;
        const wp::int32 var_82 = 1;
        wp::int32 var_83;
        const wp::int32 var_84 = 1;
        wp::int32 var_85;
        wp::int32 var_86;
        wp::int32 var_87;
        const wp::int32 var_88 = 1;
        wp::int32 var_89;
        wp::vec_t<3, wp::float32>* var_90;
        wp::vec_t<3, wp::float32> var_91;
        wp::vec_t<3, wp::float32> var_92;
        const wp::int32 var_93 = 4;
        bool var_94;
        const wp::int32 var_95 = 5;
        bool var_96;
        bool var_97;
        wp::vec_t<6, wp::float32>* var_98;
        wp::vec_t<6, wp::float32> var_99;
        wp::vec_t<6, wp::float32> var_100;
        wp::vec_t<3, wp::float32> var_101;
        const wp::int32 var_102 = 1;
        wp::int32 var_103;
        wp::int32 var_104;
        wp::int32 var_105;
        const wp::int32 var_106 = 2;
        wp::int32 var_107;
        wp::int32* var_108;
        wp::int32 var_109;
        wp::int32 var_110;
        wp::float32 var_111;
        const wp::float32 var_112 = 10000000000.0;
        bool var_113;
        wp::vec_t<3, wp::float32> var_114;
        wp::vec_t<3, wp::float32>* var_115;
        wp::vec_t<3, wp::float32> var_116;
        wp::vec_t<3, wp::float32> var_117;
        const wp::int32 var_118 = 0;
        wp::int32 var_119;
        const wp::int32 var_120 = 2;
        wp::int32 var_121;
        const wp::int32 var_122 = 0;
        wp::int32 var_123;
        const wp::int32 var_124 = 2;
        wp::int32 var_125;
        const wp::int32 var_126 = 1;
        wp::int32 var_127;
        const wp::int32 var_128 = 2;
        wp::int32 var_129;
        const wp::int32 var_130 = 1;
        wp::int32 var_131;
        const wp::int32 var_132 = 2;
        wp::int32 var_133;
        const wp::int32 var_134 = 2;
        wp::int32 var_135;
        const wp::int32 var_136 = 2;
        wp::int32 var_137;
        const wp::int32 var_138 = 2;
        wp::int32 var_139;
        const wp::int32 var_140 = 2;
        wp::int32 var_141;
        const wp::int32 var_142 = 3;
        wp::int32 var_143;
        const wp::int32 var_144 = 2;
        wp::int32 var_145;
        const wp::int32 var_146 = 3;
        wp::int32 var_147;
        const wp::int32 var_148 = 2;
        wp::int32 var_149;
        const wp::int32 var_150 = 0;
        wp::float32 var_151;
        wp::vec_t<6, wp::float32>* var_152;
        const wp::int32 var_153 = 3;
        wp::int32 var_154;
        const wp::int32 var_155 = 0;
        wp::int32 var_156;
        wp::float32* var_157;
        const wp::int32 var_158 = 1;
        wp::float32 var_159;
        wp::vec_t<6, wp::float32>* var_160;
        const wp::int32 var_161 = 3;
        wp::int32 var_162;
        const wp::int32 var_163 = 1;
        wp::int32 var_164;
        wp::float32* var_165;
        const wp::int32 var_166 = 2;
        wp::float32 var_167;
        wp::vec_t<6, wp::float32>* var_168;
        const wp::int32 var_169 = 3;
        wp::int32 var_170;
        const wp::int32 var_171 = 2;
        wp::int32 var_172;
        wp::float32* var_173;
        const wp::int32 var_174 = 0;
        wp::float32 var_175;
        wp::vec_t<6, wp::float32>* var_176;
        const wp::int32 var_177 = 3;
        wp::int32 var_178;
        const wp::int32 var_179 = 0;
        wp::int32 var_180;
        wp::float32* var_181;
        const wp::int32 var_182 = 1;
        wp::float32 var_183;
        wp::vec_t<6, wp::float32>* var_184;
        const wp::int32 var_185 = 3;
        wp::int32 var_186;
        const wp::int32 var_187 = 1;
        wp::int32 var_188;
        wp::float32* var_189;
        const wp::int32 var_190 = 2;
        wp::float32 var_191;
        wp::vec_t<6, wp::float32>* var_192;
        const wp::int32 var_193 = 3;
        wp::int32 var_194;
        const wp::int32 var_195 = 2;
        wp::int32 var_196;
        wp::float32* var_197;
        const wp::int32 var_198 = 0;
        wp::float32 var_199;
        wp::vec_t<6, wp::float32>* var_200;
        const wp::int32 var_201 = 3;
        wp::int32 var_202;
        const wp::int32 var_203 = 0;
        wp::int32 var_204;
        wp::float32* var_205;
        const wp::int32 var_206 = 1;
        wp::float32 var_207;
        wp::vec_t<6, wp::float32>* var_208;
        const wp::int32 var_209 = 3;
        wp::int32 var_210;
        const wp::int32 var_211 = 1;
        wp::int32 var_212;
        wp::float32* var_213;
        const wp::int32 var_214 = 2;
        wp::float32 var_215;
        wp::vec_t<6, wp::float32>* var_216;
        const wp::int32 var_217 = 3;
        wp::int32 var_218;
        const wp::int32 var_219 = 2;
        wp::int32 var_220;
        wp::float32* var_221;
        const wp::int32 var_222 = 0;
        wp::float32 var_223;
        wp::vec_t<6, wp::float32>* var_224;
        const wp::int32 var_225 = 3;
        wp::int32 var_226;
        const wp::int32 var_227 = 0;
        wp::int32 var_228;
        wp::float32* var_229;
        const wp::int32 var_230 = 1;
        wp::float32 var_231;
        wp::vec_t<6, wp::float32>* var_232;
        const wp::int32 var_233 = 3;
        wp::int32 var_234;
        const wp::int32 var_235 = 1;
        wp::int32 var_236;
        wp::float32* var_237;
        const wp::int32 var_238 = 2;
        wp::float32 var_239;
        wp::vec_t<6, wp::float32>* var_240;
        const wp::int32 var_241 = 3;
        wp::int32 var_242;
        const wp::int32 var_243 = 2;
        wp::int32 var_244;
        wp::float32* var_245;
        const wp::int32 var_246 = 1;
        const wp::int32 var_247 = -1;
        wp::vec_t<2, wp::int32>* var_248;
        wp::int32* var_249;
        wp::vec_t<2, wp::int32>* var_250;
        wp::int32* var_251;
        wp::vec_t<2, wp::int32>* var_252;
        wp::int32* var_253;
        const wp::int32 var_254 = 3;
        wp::int32 var_255;
        const wp::int32 var_256 = 3;
        wp::int32 var_257;
        const wp::int32 var_258 = 2;
        wp::int32 var_259;
        wp::int32 var_260;
        wp::int32 var_261;
        wp::int32 var_262;
        const wp::int32 var_263 = 0;
        wp::int32 var_264;
        const wp::int32 var_265 = 2;
        wp::int32 var_266;
        const wp::int32 var_267 = 0;
        wp::int32 var_268;
        const wp::int32 var_269 = 2;
        wp::int32 var_270;
        const wp::int32 var_271 = 0;
        wp::float32 var_272;
        wp::vec_t<6, wp::float32>* var_273;
        const wp::int32 var_274 = 3;
        wp::int32 var_275;
        const wp::int32 var_276 = 0;
        wp::int32 var_277;
        wp::float32* var_278;
        const wp::int32 var_279 = 1;
        wp::float32 var_280;
        wp::vec_t<6, wp::float32>* var_281;
        const wp::int32 var_282 = 3;
        wp::int32 var_283;
        const wp::int32 var_284 = 1;
        wp::int32 var_285;
        wp::float32* var_286;
        const wp::int32 var_287 = 2;
        wp::float32 var_288;
        wp::vec_t<6, wp::float32>* var_289;
        const wp::int32 var_290 = 3;
        wp::int32 var_291;
        const wp::int32 var_292 = 2;
        wp::int32 var_293;
        wp::float32* var_294;
        const wp::int32 var_295 = 1;
        const wp::int32 var_296 = -1;
        wp::vec_t<2, wp::int32>* var_297;
        wp::int32* var_298;
        const wp::int32 var_299 = 1;
        wp::int32 var_300;
        const wp::int32 var_301 = 1;
        wp::int32 var_302;
        const wp::int32 var_303 = 2;
        wp::int32 var_304;
        wp::int32 var_305;
        wp::int32 var_306;
        wp::int32 var_307;
        wp::int32 var_308;
        wp::int32 var_309;
        wp::int32 var_310;
        wp::int32 var_311;
        wp::int32 var_312;
        wp::int32 var_313;
        wp::int32 var_314;
        const wp::int32 var_315 = 0;
        wp::int32 var_316;
        const wp::int32 var_317 = 2;
        wp::int32 var_318;
        const wp::int32 var_319 = 0;
        wp::int32 var_320;
        const wp::int32 var_321 = 2;
        wp::int32 var_322;
        const wp::int32 var_323 = 0;
        wp::float32 var_324;
        wp::vec_t<6, wp::float32>* var_325;
        const wp::int32 var_326 = 3;
        wp::int32 var_327;
        const wp::int32 var_328 = 0;
        wp::int32 var_329;
        wp::float32* var_330;
        const wp::int32 var_331 = 1;
        wp::float32 var_332;
        wp::vec_t<6, wp::float32>* var_333;
        const wp::int32 var_334 = 3;
        wp::int32 var_335;
        const wp::int32 var_336 = 1;
        wp::int32 var_337;
        wp::float32* var_338;
        const wp::int32 var_339 = 2;
        wp::float32 var_340;
        wp::vec_t<6, wp::float32>* var_341;
        const wp::int32 var_342 = 3;
        wp::int32 var_343;
        const wp::int32 var_344 = 2;
        wp::int32 var_345;
        wp::float32* var_346;
        const wp::int32 var_347 = 1;
        const wp::int32 var_348 = -1;
        wp::vec_t<2, wp::int32>* var_349;
        wp::int32* var_350;
        const wp::int32 var_351 = 1;
        wp::int32 var_352;
        const wp::int32 var_353 = 1;
        wp::int32 var_354;
        const wp::int32 var_355 = 1;
        wp::int32 var_356;
        wp::int32 var_357;
        wp::int32 var_358;
        wp::int32 var_359;
        wp::int32 var_360;
        wp::int32 var_361;
        wp::int32 var_362;
        const wp::int32 var_363 = 1;
        wp::int32 var_364;
        wp::shape_t* var_365;
        const wp::int32 var_366 = 0;
        wp::int32 var_367;
        wp::shape_t var_368;
        bool var_369;
        wp::int32 var_370;
        const wp::int32 var_371 = 1;
        wp::int32 var_372;
        wp::int32* var_373;
        const wp::int32 var_374 = 2;
        bool var_375;
        wp::int32 var_376;
        const bool var_377 = false;
        bool var_378;
        const wp::int32 var_379 = 1;
        wp::int32 var_380;
        bool var_381;
        bool var_382;
        const wp::int32 var_383 = 0;
        wp::int32 var_384;
        const wp::int32 var_385 = 2;
        wp::int32 var_386;
        const wp::int32 var_387 = 0;
        wp::int32 var_388;
        const wp::int32 var_389 = 2;
        wp::int32 var_390;
        wp::vec_t<3, wp::float32>* var_391;
        wp::vec_t<3, wp::float32> var_392;
        wp::vec_t<3, wp::float32> var_393;
        const wp::int32 var_394 = 0;
        wp::float32 var_395;
        wp::vec_t<6, wp::float32>* var_396;
        const wp::int32 var_397 = 3;
        wp::int32 var_398;
        const wp::int32 var_399 = 0;
        wp::int32 var_400;
        wp::float32* var_401;
        const wp::int32 var_402 = 1;
        wp::float32 var_403;
        wp::vec_t<6, wp::float32>* var_404;
        const wp::int32 var_405 = 3;
        wp::int32 var_406;
        const wp::int32 var_407 = 1;
        wp::int32 var_408;
        wp::float32* var_409;
        const wp::int32 var_410 = 2;
        wp::float32 var_411;
        wp::vec_t<6, wp::float32>* var_412;
        const wp::int32 var_413 = 3;
        wp::int32 var_414;
        const wp::int32 var_415 = 2;
        wp::int32 var_416;
        wp::float32* var_417;
        const wp::int32 var_418 = 1;
        const wp::int32 var_419 = -1;
        wp::vec_t<2, wp::int32>* var_420;
        wp::int32* var_421;
        const wp::int32 var_422 = 1;
        wp::int32 var_423;
        const wp::int32 var_424 = 1;
        wp::int32 var_425;
        wp::int32 var_426;
        wp::int32 var_427;
        wp::vec_t<3, wp::float32> var_428;
        wp::int32 var_429;
        wp::int32 var_430;
        //---------
        // forward
        // def _spatial_tendon_wrap(                                                              <L 3468>
        // worldid = wp.tid()                                                                     <L 3485>
        var_0 = builtin_tid1d();
        // wrapcount = int(0)                                                                     <L 3487>
        var_2 = wp::int(var_1);
        // wrapgeomid = int(0)                                                                    <L 3488>
        var_4 = wp::int(var_3);
        // for i in range(ntendon):                                                               <L 3489>
        var_5 = wp::range(var_ntendon);
        start_for_0:;
            if (iter_cmp(var_5) == 0) goto end_for_0;
            var_6 = wp::iter_next(var_5);
            // adr = tendon_adr[i]                                                                <L 3490>
            var_7 = wp::address(var_tendon_adr, var_6);
            var_9 = wp::load(var_7);
            var_8 = wp::copy(var_9);
            // ten_wrapadr_out[worldid, i] = wrapcount                                            <L 3491>
            wp::array_store(var_ten_wrapadr_out, var_0, var_6, var_2);
            // wrapnum = int(0)                                                                   <L 3492>
            var_11 = wp::int(var_10);
            // tendonnum = tendon_num[i]                                                          <L 3493>
            var_12 = wp::address(var_tendon_num, var_6);
            var_14 = wp::load(var_12);
            var_13 = wp::copy(var_14);
            // if wrap_type[adr] == WrapType.JOINT:                                               <L 3496>
            var_15 = wp::address(var_wrap_type, var_8);
            var_18 = wp::load(var_15);
            var_17 = (var_18 == var_16);
            if (var_17) {
                // continue                                                                       <L 3497>
                goto start_for_0;
            }
            // j = int(0)                                                                         <L 3500>
            var_20 = wp::int(var_19);
            // while j < tendonnum - 1:                                                           <L 3501>
        start_while_2:;
            var_22 = wp::sub(var_13, var_21);
            var_23 = (var_20 < var_22);
        if ((var_23) == false) goto end_while_2;
                // type0 = wrap_type[adr + j + 0]                                                 <L 3503>
                var_24 = wp::add(var_8, var_20);
                var_26 = wp::add(var_24, var_25);
                var_27 = wp::address(var_wrap_type, var_26);
                var_29 = wp::load(var_27);
                var_28 = wp::copy(var_29);
                // type1 = wrap_type[adr + j + 1]                                                 <L 3504>
                var_30 = wp::add(var_8, var_20);
                var_32 = wp::add(var_30, var_31);
                var_33 = wp::address(var_wrap_type, var_32);
                var_35 = wp::load(var_33);
                var_34 = wp::copy(var_35);
                // id0 = wrap_objid[adr + j + 0]                                                  <L 3505>
                var_36 = wp::add(var_8, var_20);
                var_38 = wp::add(var_36, var_37);
                var_39 = wp::address(var_wrap_objid, var_38);
                var_41 = wp::load(var_39);
                var_40 = wp::copy(var_41);
                // id1 = wrap_objid[adr + j + 1]                                                  <L 3506>
                var_42 = wp::add(var_8, var_20);
                var_44 = wp::add(var_42, var_43);
                var_45 = wp::address(var_wrap_objid, var_44);
                var_47 = wp::load(var_45);
                var_46 = wp::copy(var_47);
                // pulley0 = type0 == WrapType.PULLEY                                             <L 3509>
                var_49 = (var_28 == var_48);
                // if pulley0 or type1 == WrapType.PULLEY:                                        <L 3510>
                var_51 = (var_34 == var_50);
                var_52 = var_49 || var_51;
                if (var_52) {
                    // if pulley0:                                                                <L 3511>
                    if (var_49) {
                        // row = wrapcount // 2                                                   <L 3512>
                        var_54 = wp::floordiv(var_2, var_53);
                        // col = wrapcount % 2                                                    <L 3513>
                        var_56 = wp::mod(var_2, var_55);
                        // wrap_xpos_out[worldid, row][3 * col + 0] = 0.0                         <L 3514>
                        var_58 = wp::address(var_wrap_xpos_out, var_0, var_54);
                        var_60 = wp::mul(var_59, var_56);
                        var_62 = wp::add(var_60, var_61);
                        var_63 = wp::indexref(var_58, var_62);
                        wp::store(var_63, var_57);
                        // wrap_xpos_out[worldid, row][3 * col + 1] = 0.0                         <L 3515>
                        var_65 = wp::address(var_wrap_xpos_out, var_0, var_54);
                        var_67 = wp::mul(var_66, var_56);
                        var_69 = wp::add(var_67, var_68);
                        var_70 = wp::indexref(var_65, var_69);
                        wp::store(var_70, var_64);
                        // wrap_xpos_out[worldid, row][3 * col + 2] = 0.0                         <L 3516>
                        var_72 = wp::address(var_wrap_xpos_out, var_0, var_54);
                        var_74 = wp::mul(var_73, var_56);
                        var_76 = wp::add(var_74, var_75);
                        var_77 = wp::indexref(var_72, var_76);
                        wp::store(var_77, var_71);
                        // wrap_obj_out[worldid, row][col] = -2                                   <L 3518>
                        var_80 = wp::address(var_wrap_obj_out, var_0, var_54);
                        var_81 = wp::indexref(var_80, var_56);
                        wp::store(var_81, var_79);
                        // wrapnum += 1                                                           <L 3520>
                        var_83 = wp::add(var_11, var_82);
                        // wrapcount += 1                                                         <L 3521>
                        var_85 = wp::add(var_2, var_84);
                    }
                    var_86 = wp::where(var_49, var_85, var_2);
                    var_87 = wp::where(var_49, var_83, var_11);
                    // j += 1                                                                     <L 3524>
                    var_89 = wp::add(var_20, var_88);
                    // continue                                                                   <L 3525>
                    wp::assign(var_2, var_86);
                    wp::assign(var_11, var_87);
                    wp::assign(var_20, var_89);
                    goto start_while_2;
                }
                // wpnt_site0 = site_xpos_in[worldid, id0]                                        <L 3528>
                var_90 = wp::address(var_site_xpos_in, var_0, var_40);
                var_92 = wp::load(var_90);
                var_91 = wp::copy(var_92);
                // if type1 == WrapType.SPHERE or type1 == WrapType.CYLINDER:                     <L 3531>
                var_94 = (var_34 == var_93);
                var_96 = (var_34 == var_95);
                var_97 = var_94 || var_96;
                if (var_97) {
                    // wrap_geom_xpos = wrap_geom_xpos_in[worldid, wrapgeomid]                    <L 3532>
                    var_98 = wp::address(var_wrap_geom_xpos_in, var_0, var_4);
                    var_100 = wp::load(var_98);
                    var_99 = wp::copy(var_100);
                    // wpnt_geom0 = wp.spatial_top(wrap_geom_xpos)                                <L 3533>
                    var_101 = wp::spatial_top(var_99);
                    // wrapgeomid += 1                                                            <L 3534>
                    var_103 = wp::add(var_4, var_102);
                    // wrapid = id1                                                               <L 3536>
                    var_104 = wp::copy(var_46);
                    // id1 = wrap_objid[adr + j + 2]                                              <L 3537>
                    var_105 = wp::add(var_8, var_20);
                    var_107 = wp::add(var_105, var_106);
                    var_108 = wp::address(var_wrap_objid, var_107);
                    var_110 = wp::load(var_108);
                    var_109 = wp::copy(var_110);
                    // if wp.norm_l2(wpnt_geom0) < MJ_MAXVAL:                                     <L 3538>
                    var_111 = norm_l2_0(var_101);
                    var_113 = (var_111 < var_112);
                    if (var_113) {
                        // wpnt_geom1 = wp.spatial_bottom(wrap_geom_xpos)                         <L 3539>
                        var_114 = wp::spatial_bottom(var_99);
                        // wpnt_site1 = site_xpos_in[worldid, id1]                                <L 3540>
                        var_115 = wp::address(var_site_xpos_in, var_0, var_109);
                        var_117 = wp::load(var_115);
                        var_116 = wp::copy(var_117);
                        // row0 = (wrapcount + 0) // 2                                            <L 3543>
                        var_119 = wp::add(var_2, var_118);
                        var_121 = wp::floordiv(var_119, var_120);
                        // col0 = (wrapcount + 0) % 2                                             <L 3544>
                        var_123 = wp::add(var_2, var_122);
                        var_125 = wp::mod(var_123, var_124);
                        // row1 = (wrapcount + 1) // 2                                            <L 3545>
                        var_127 = wp::add(var_2, var_126);
                        var_129 = wp::floordiv(var_127, var_128);
                        // col1 = (wrapcount + 1) % 2                                             <L 3546>
                        var_131 = wp::add(var_2, var_130);
                        var_133 = wp::mod(var_131, var_132);
                        // row2 = (wrapcount + 2) // 2                                            <L 3547>
                        var_135 = wp::add(var_2, var_134);
                        var_137 = wp::floordiv(var_135, var_136);
                        // col2 = (wrapcount + 2) % 2                                             <L 3548>
                        var_139 = wp::add(var_2, var_138);
                        var_141 = wp::mod(var_139, var_140);
                        // row3 = (wrapcount + 3) // 2                                            <L 3549>
                        var_143 = wp::add(var_2, var_142);
                        var_145 = wp::floordiv(var_143, var_144);
                        // col3 = (wrapcount + 3) % 2                                             <L 3550>
                        var_147 = wp::add(var_2, var_146);
                        var_149 = wp::mod(var_147, var_148);
                        // wrap_xpos_out[worldid, row0][3 * col0 + 0] = wpnt_site0[0]             <L 3552>
                        var_151 = wp::extract(var_91, var_150);
                        var_152 = wp::address(var_wrap_xpos_out, var_0, var_121);
                        var_154 = wp::mul(var_153, var_125);
                        var_156 = wp::add(var_154, var_155);
                        var_157 = wp::indexref(var_152, var_156);
                        wp::store(var_157, var_151);
                        // wrap_xpos_out[worldid, row0][3 * col0 + 1] = wpnt_site0[1]             <L 3553>
                        var_159 = wp::extract(var_91, var_158);
                        var_160 = wp::address(var_wrap_xpos_out, var_0, var_121);
                        var_162 = wp::mul(var_161, var_125);
                        var_164 = wp::add(var_162, var_163);
                        var_165 = wp::indexref(var_160, var_164);
                        wp::store(var_165, var_159);
                        // wrap_xpos_out[worldid, row0][3 * col0 + 2] = wpnt_site0[2]             <L 3554>
                        var_167 = wp::extract(var_91, var_166);
                        var_168 = wp::address(var_wrap_xpos_out, var_0, var_121);
                        var_170 = wp::mul(var_169, var_125);
                        var_172 = wp::add(var_170, var_171);
                        var_173 = wp::indexref(var_168, var_172);
                        wp::store(var_173, var_167);
                        // wrap_xpos_out[worldid, row1][3 * col1 + 0] = wpnt_geom0[0]             <L 3556>
                        var_175 = wp::extract(var_101, var_174);
                        var_176 = wp::address(var_wrap_xpos_out, var_0, var_129);
                        var_178 = wp::mul(var_177, var_133);
                        var_180 = wp::add(var_178, var_179);
                        var_181 = wp::indexref(var_176, var_180);
                        wp::store(var_181, var_175);
                        // wrap_xpos_out[worldid, row1][3 * col1 + 1] = wpnt_geom0[1]             <L 3557>
                        var_183 = wp::extract(var_101, var_182);
                        var_184 = wp::address(var_wrap_xpos_out, var_0, var_129);
                        var_186 = wp::mul(var_185, var_133);
                        var_188 = wp::add(var_186, var_187);
                        var_189 = wp::indexref(var_184, var_188);
                        wp::store(var_189, var_183);
                        // wrap_xpos_out[worldid, row1][3 * col1 + 2] = wpnt_geom0[2]             <L 3558>
                        var_191 = wp::extract(var_101, var_190);
                        var_192 = wp::address(var_wrap_xpos_out, var_0, var_129);
                        var_194 = wp::mul(var_193, var_133);
                        var_196 = wp::add(var_194, var_195);
                        var_197 = wp::indexref(var_192, var_196);
                        wp::store(var_197, var_191);
                        // wrap_xpos_out[worldid, row2][3 * col2 + 0] = wpnt_geom1[0]             <L 3560>
                        var_199 = wp::extract(var_114, var_198);
                        var_200 = wp::address(var_wrap_xpos_out, var_0, var_137);
                        var_202 = wp::mul(var_201, var_141);
                        var_204 = wp::add(var_202, var_203);
                        var_205 = wp::indexref(var_200, var_204);
                        wp::store(var_205, var_199);
                        // wrap_xpos_out[worldid, row2][3 * col2 + 1] = wpnt_geom1[1]             <L 3561>
                        var_207 = wp::extract(var_114, var_206);
                        var_208 = wp::address(var_wrap_xpos_out, var_0, var_137);
                        var_210 = wp::mul(var_209, var_141);
                        var_212 = wp::add(var_210, var_211);
                        var_213 = wp::indexref(var_208, var_212);
                        wp::store(var_213, var_207);
                        // wrap_xpos_out[worldid, row2][3 * col2 + 2] = wpnt_geom1[2]             <L 3562>
                        var_215 = wp::extract(var_114, var_214);
                        var_216 = wp::address(var_wrap_xpos_out, var_0, var_137);
                        var_218 = wp::mul(var_217, var_141);
                        var_220 = wp::add(var_218, var_219);
                        var_221 = wp::indexref(var_216, var_220);
                        wp::store(var_221, var_215);
                        // wrap_xpos_out[worldid, row3][3 * col3 + 0] = wpnt_site1[0]             <L 3564>
                        var_223 = wp::extract(var_116, var_222);
                        var_224 = wp::address(var_wrap_xpos_out, var_0, var_145);
                        var_226 = wp::mul(var_225, var_149);
                        var_228 = wp::add(var_226, var_227);
                        var_229 = wp::indexref(var_224, var_228);
                        wp::store(var_229, var_223);
                        // wrap_xpos_out[worldid, row3][3 * col3 + 1] = wpnt_site1[1]             <L 3565>
                        var_231 = wp::extract(var_116, var_230);
                        var_232 = wp::address(var_wrap_xpos_out, var_0, var_145);
                        var_234 = wp::mul(var_233, var_149);
                        var_236 = wp::add(var_234, var_235);
                        var_237 = wp::indexref(var_232, var_236);
                        wp::store(var_237, var_231);
                        // wrap_xpos_out[worldid, row3][3 * col3 + 2] = wpnt_site1[2]             <L 3566>
                        var_239 = wp::extract(var_116, var_238);
                        var_240 = wp::address(var_wrap_xpos_out, var_0, var_145);
                        var_242 = wp::mul(var_241, var_149);
                        var_244 = wp::add(var_242, var_243);
                        var_245 = wp::indexref(var_240, var_244);
                        wp::store(var_245, var_239);
                        // wrap_obj_out[worldid, row0][col0] = -1                                 <L 3568>
                        var_248 = wp::address(var_wrap_obj_out, var_0, var_121);
                        var_249 = wp::indexref(var_248, var_125);
                        wp::store(var_249, var_247);
                        // wrap_obj_out[worldid, row1][col1] = wrapid                             <L 3569>
                        var_250 = wp::address(var_wrap_obj_out, var_0, var_129);
                        var_251 = wp::indexref(var_250, var_133);
                        wp::store(var_251, var_104);
                        // wrap_obj_out[worldid, row2][col2] = wrapid                             <L 3570>
                        var_252 = wp::address(var_wrap_obj_out, var_0, var_137);
                        var_253 = wp::indexref(var_252, var_141);
                        wp::store(var_253, var_104);
                        // wrapnum += 3                                                           <L 3572>
                        var_255 = wp::add(var_11, var_254);
                        // wrapcount += 3                                                         <L 3573>
                        var_257 = wp::add(var_2, var_256);
                        // j += 2                                                                 <L 3574>
                        var_259 = wp::add(var_20, var_258);
                    }
                    var_260 = wp::where(var_113, var_257, var_2);
                    var_261 = wp::where(var_113, var_255, var_11);
                    var_262 = wp::where(var_113, var_259, var_20);
                    if (!var_113) {
                        // row0 = (wrapcount + 0) // 2                                            <L 3577>
                        var_264 = wp::add(var_260, var_263);
                        var_266 = wp::floordiv(var_264, var_265);
                        // col0 = (wrapcount + 0) % 2                                             <L 3578>
                        var_268 = wp::add(var_260, var_267);
                        var_270 = wp::mod(var_268, var_269);
                        // wrap_xpos_out[worldid, row0][3 * col0 + 0] = wpnt_site0[0]             <L 3580>
                        var_272 = wp::extract(var_91, var_271);
                        var_273 = wp::address(var_wrap_xpos_out, var_0, var_266);
                        var_275 = wp::mul(var_274, var_270);
                        var_277 = wp::add(var_275, var_276);
                        var_278 = wp::indexref(var_273, var_277);
                        wp::store(var_278, var_272);
                        // wrap_xpos_out[worldid, row0][3 * col0 + 1] = wpnt_site0[1]             <L 3581>
                        var_280 = wp::extract(var_91, var_279);
                        var_281 = wp::address(var_wrap_xpos_out, var_0, var_266);
                        var_283 = wp::mul(var_282, var_270);
                        var_285 = wp::add(var_283, var_284);
                        var_286 = wp::indexref(var_281, var_285);
                        wp::store(var_286, var_280);
                        // wrap_xpos_out[worldid, row0][3 * col0 + 2] = wpnt_site0[2]             <L 3582>
                        var_288 = wp::extract(var_91, var_287);
                        var_289 = wp::address(var_wrap_xpos_out, var_0, var_266);
                        var_291 = wp::mul(var_290, var_270);
                        var_293 = wp::add(var_291, var_292);
                        var_294 = wp::indexref(var_289, var_293);
                        wp::store(var_294, var_288);
                        // wrap_obj_out[worldid, row0][col0] = -1                                 <L 3584>
                        var_297 = wp::address(var_wrap_obj_out, var_0, var_266);
                        var_298 = wp::indexref(var_297, var_270);
                        wp::store(var_298, var_296);
                        // wrapnum += 1                                                           <L 3586>
                        var_300 = wp::add(var_261, var_299);
                        // wrapcount += 1                                                         <L 3587>
                        var_302 = wp::add(var_260, var_301);
                        // j += 2                                                                 <L 3588>
                        var_304 = wp::add(var_262, var_303);
                    }
                    var_305 = wp::where(var_113, var_260, var_302);
                    var_306 = wp::where(var_113, var_261, var_300);
                    var_307 = wp::where(var_113, var_262, var_304);
                    var_308 = wp::where(var_113, var_121, var_266);
                    var_309 = wp::where(var_113, var_125, var_270);
                }
                var_310 = wp::where(var_97, var_305, var_2);
                var_311 = wp::where(var_97, var_103, var_4);
                var_312 = wp::where(var_97, var_306, var_11);
                var_313 = wp::where(var_97, var_307, var_20);
                var_314 = wp::where(var_97, var_109, var_46);
                if (!var_97) {
                    // row0 = (wrapcount + 0) // 2                                                <L 3591>
                    var_316 = wp::add(var_310, var_315);
                    var_318 = wp::floordiv(var_316, var_317);
                    // col0 = (wrapcount + 0) % 2                                                 <L 3592>
                    var_320 = wp::add(var_310, var_319);
                    var_322 = wp::mod(var_320, var_321);
                    // wrap_xpos_out[worldid, row0][3 * col0 + 0] = wpnt_site0[0]                 <L 3594>
                    var_324 = wp::extract(var_91, var_323);
                    var_325 = wp::address(var_wrap_xpos_out, var_0, var_318);
                    var_327 = wp::mul(var_326, var_322);
                    var_329 = wp::add(var_327, var_328);
                    var_330 = wp::indexref(var_325, var_329);
                    wp::store(var_330, var_324);
                    // wrap_xpos_out[worldid, row0][3 * col0 + 1] = wpnt_site0[1]                 <L 3595>
                    var_332 = wp::extract(var_91, var_331);
                    var_333 = wp::address(var_wrap_xpos_out, var_0, var_318);
                    var_335 = wp::mul(var_334, var_322);
                    var_337 = wp::add(var_335, var_336);
                    var_338 = wp::indexref(var_333, var_337);
                    wp::store(var_338, var_332);
                    // wrap_xpos_out[worldid, row0][3 * col0 + 2] = wpnt_site0[2]                 <L 3596>
                    var_340 = wp::extract(var_91, var_339);
                    var_341 = wp::address(var_wrap_xpos_out, var_0, var_318);
                    var_343 = wp::mul(var_342, var_322);
                    var_345 = wp::add(var_343, var_344);
                    var_346 = wp::indexref(var_341, var_345);
                    wp::store(var_346, var_340);
                    // wrap_obj_out[worldid, row0][col0] = -1                                     <L 3598>
                    var_349 = wp::address(var_wrap_obj_out, var_0, var_318);
                    var_350 = wp::indexref(var_349, var_322);
                    wp::store(var_350, var_348);
                    // wrapnum += 1                                                               <L 3600>
                    var_352 = wp::add(var_312, var_351);
                    // wrapcount += 1                                                             <L 3601>
                    var_354 = wp::add(var_310, var_353);
                    // j += 1                                                                     <L 3602>
                    var_356 = wp::add(var_313, var_355);
                }
                var_357 = wp::where(var_97, var_310, var_354);
                var_358 = wp::where(var_97, var_312, var_352);
                var_359 = wp::where(var_97, var_313, var_356);
                var_360 = wp::where(var_97, var_308, var_318);
                var_361 = wp::where(var_97, var_309, var_322);
                // if adr + j + 1 < wrap_type.shape[0]:                                           <L 3605>
                var_362 = wp::add(var_8, var_359);
                var_364 = wp::add(var_362, var_363);
                var_365 = &(var_wrap_type.shape);
                var_368 = wp::load(var_365);
                var_367 = wp::extract(var_368, var_366);
                var_369 = (var_364 < var_367);
                if (var_369) {
                    // last_before_pulley = wrap_type[adr + j + 1] == WrapType.PULLEY             <L 3606>
                    var_370 = wp::add(var_8, var_359);
                    var_372 = wp::add(var_370, var_371);
                    var_373 = wp::address(var_wrap_type, var_372);
                    var_376 = wp::load(var_373);
                    var_375 = (var_376 == var_374);
                }
                if (!var_369) {
                    // last_before_pulley = False                                                 <L 3608>
                }
                var_378 = wp::where(var_369, var_375, var_377);
                // if j == tendonnum - 1 or last_before_pulley:                                   <L 3610>
                var_380 = wp::sub(var_13, var_379);
                var_381 = (var_359 == var_380);
                var_382 = var_381 || var_378;
                if (var_382) {
                    // row0 = (wrapcount + 0) // 2                                                <L 3611>
                    var_384 = wp::add(var_357, var_383);
                    var_386 = wp::floordiv(var_384, var_385);
                    // col0 = (wrapcount + 0) % 2                                                 <L 3612>
                    var_388 = wp::add(var_357, var_387);
                    var_390 = wp::mod(var_388, var_389);
                    // wpnt_site1 = site_xpos_in[worldid, id1]                                    <L 3614>
                    var_391 = wp::address(var_site_xpos_in, var_0, var_314);
                    var_393 = wp::load(var_391);
                    var_392 = wp::copy(var_393);
                    // wrap_xpos_out[worldid, row0][3 * col0 + 0] = wpnt_site1[0]                 <L 3615>
                    var_395 = wp::extract(var_392, var_394);
                    var_396 = wp::address(var_wrap_xpos_out, var_0, var_386);
                    var_398 = wp::mul(var_397, var_390);
                    var_400 = wp::add(var_398, var_399);
                    var_401 = wp::indexref(var_396, var_400);
                    wp::store(var_401, var_395);
                    // wrap_xpos_out[worldid, row0][3 * col0 + 1] = wpnt_site1[1]                 <L 3616>
                    var_403 = wp::extract(var_392, var_402);
                    var_404 = wp::address(var_wrap_xpos_out, var_0, var_386);
                    var_406 = wp::mul(var_405, var_390);
                    var_408 = wp::add(var_406, var_407);
                    var_409 = wp::indexref(var_404, var_408);
                    wp::store(var_409, var_403);
                    // wrap_xpos_out[worldid, row0][3 * col0 + 2] = wpnt_site1[2]                 <L 3617>
                    var_411 = wp::extract(var_392, var_410);
                    var_412 = wp::address(var_wrap_xpos_out, var_0, var_386);
                    var_414 = wp::mul(var_413, var_390);
                    var_416 = wp::add(var_414, var_415);
                    var_417 = wp::indexref(var_412, var_416);
                    wp::store(var_417, var_411);
                    // wrap_obj_out[worldid, row0][col0] = -1                                     <L 3619>
                    var_420 = wp::address(var_wrap_obj_out, var_0, var_386);
                    var_421 = wp::indexref(var_420, var_390);
                    wp::store(var_421, var_419);
                    // wrapnum += 1                                                               <L 3620>
                    var_423 = wp::add(var_358, var_422);
                    // wrapcount += 1                                                             <L 3621>
                    var_425 = wp::add(var_357, var_424);
                }
                var_426 = wp::where(var_382, var_425, var_357);
                var_427 = wp::where(var_382, var_423, var_358);
                var_428 = wp::where(var_382, var_392, var_116);
                var_429 = wp::where(var_382, var_386, var_360);
                var_430 = wp::where(var_382, var_390, var_361);
                wp::assign(var_2, var_426);
                wp::assign(var_4, var_311);
                wp::assign(var_11, var_427);
                wp::assign(var_20, var_359);
        goto start_while_2;
        end_while_2:;
            // ten_wrapnum_out[worldid, i] = wrapnum                                              <L 3623>
            wp::array_store(var_ten_wrapnum_out, var_0, var_6, var_11);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void _qfrc_bias_4085b6de_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_int_in,
    wp::array_t<wp::float32> var_qfrc_bias_out)
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
        wp::vec_t<6, wp::float32>* var_5;
        wp::vec_t<6, wp::float32>* var_6;
        wp::float32 var_7;
        wp::vec_t<6, wp::float32> var_8;
        wp::vec_t<6, wp::float32> var_9;
        //---------
        // forward
        // def _qfrc_bias(                                                                        <L 1244>
        // worldid, dofid = wp.tid()                                                              <L 1253>
        builtin_tid2d(var_0, var_1);
        // bodyid = dof_bodyid[dofid]                                                             <L 1254>
        var_2 = wp::address(var_dof_bodyid, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // qfrc_bias_out[worldid, dofid] = wp.dot(cdof_in[worldid, dofid], cfrc_int_in[worldid, bodyid])       <L 1255>
        var_5 = wp::address(var_cdof_in, var_0, var_1);
        var_6 = wp::address(var_cfrc_int_in, var_0, var_3);
        var_8 = wp::load(var_5);
        var_9 = wp::load(var_6);
        var_7 = wp::dot(var_8, var_9);
        wp::array_store(var_qfrc_bias_out, var_0, var_1, var_7);
    }
}



extern "C" __global__ void _transmission_28153805_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_nv,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_weldid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_jnt_qposadr,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::int32> var_dof_parentid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> var_site_quat,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::int32> var_actuator_trntype,
    wp::array_t<wp::vec_t<2, wp::int32>> var_actuator_trnid,
    wp::array_t<wp::vec_t<6, wp::float32>> var_actuator_gear,
    wp::array_t<wp::float32> var_actuator_cranklength,
    wp::array_t<wp::float32> var_qpos_in,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::array_t<wp::float32> var_ten_length_in,
    wp::array_t<wp::int32> var_moment_nnz,
    wp::array_t<wp::float32> var_actuator_length_out,
    wp::array_t<wp::int32> var_moment_rownnz_out,
    wp::array_t<wp::int32> var_moment_rowadr_out,
    wp::array_t<wp::int32> var_moment_colind_out,
    wp::array_t<wp::float32> var_actuator_moment_out)
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
        wp::vec_t<6, wp::float32>* var_10;
        wp::vec_t<6, wp::float32> var_11;
        wp::vec_t<6, wp::float32> var_12;
        const wp::int32 var_13 = 0;
        bool var_14;
        const wp::int32 var_15 = 1;
        bool var_16;
        bool var_17;
        wp::slice_t var_18;
        const wp::int32 var_19 = 0;
        wp::array_t<wp::float32> var_20;
        wp::vec_t<2, wp::int32>* var_21;
        const wp::int32 var_22 = 0;
        wp::int32 var_23;
        wp::vec_t<2, wp::int32> var_24;
        wp::int32* var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        wp::int32* var_28;
        wp::int32 var_29;
        wp::int32 var_30;
        wp::int32* var_31;
        wp::int32 var_32;
        wp::int32 var_33;
        const wp::int32 var_34 = 0;
        bool var_35;
        const wp::int32 var_36 = 6;
        const wp::int32 var_37 = 6;
        wp::int32 var_38;
        const wp::int32 var_39 = 0;
        wp::int32 var_40;
        const wp::int32 var_41 = 0;
        wp::int32 var_42;
        const wp::int32 var_43 = 1;
        wp::int32 var_44;
        const wp::int32 var_45 = 1;
        wp::int32 var_46;
        const wp::int32 var_47 = 2;
        wp::int32 var_48;
        const wp::int32 var_49 = 2;
        wp::int32 var_50;
        const wp::int32 var_51 = 3;
        wp::int32 var_52;
        const wp::int32 var_53 = 3;
        wp::int32 var_54;
        const wp::int32 var_55 = 4;
        wp::int32 var_56;
        const wp::int32 var_57 = 4;
        wp::int32 var_58;
        const wp::int32 var_59 = 5;
        wp::int32 var_60;
        const wp::int32 var_61 = 5;
        wp::int32 var_62;
        const wp::float32 var_63 = 0.0;
        const wp::int32 var_64 = 1;
        bool var_65;
        const wp::int32 var_66 = 3;
        wp::int32 var_67;
        wp::float32* var_68;
        const wp::int32 var_69 = 4;
        wp::int32 var_70;
        wp::float32* var_71;
        const wp::int32 var_72 = 5;
        wp::int32 var_73;
        wp::float32* var_74;
        const wp::int32 var_75 = 6;
        wp::int32 var_76;
        wp::float32* var_77;
        wp::quat_t<wp::float32> var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        wp::float32 var_82;
        wp::quat_t<wp::float32> var_83;
        wp::quat_t<wp::float32> var_84;
        wp::vec_t<3, wp::float32> var_85;
        wp::vec_t<3, wp::float32> var_86;
        const wp::int32 var_87 = 0;
        wp::float32 var_88;
        const wp::int32 var_89 = 0;
        wp::int32 var_90;
        const wp::int32 var_91 = 1;
        wp::float32 var_92;
        const wp::int32 var_93 = 1;
        wp::int32 var_94;
        const wp::int32 var_95 = 2;
        wp::float32 var_96;
        const wp::int32 var_97 = 2;
        wp::int32 var_98;
        const wp::int32 var_99 = 0;
        wp::float32 var_100;
        const wp::int32 var_101 = 3;
        wp::int32 var_102;
        const wp::int32 var_103 = 1;
        wp::float32 var_104;
        const wp::int32 var_105 = 4;
        wp::int32 var_106;
        const wp::int32 var_107 = 2;
        wp::float32 var_108;
        const wp::int32 var_109 = 5;
        wp::int32 var_110;
        const wp::int32 var_111 = 0;
        wp::float32 var_112;
        const wp::int32 var_113 = 0;
        wp::int32 var_114;
        const wp::int32 var_115 = 1;
        wp::float32 var_116;
        const wp::int32 var_117 = 1;
        wp::int32 var_118;
        const wp::int32 var_119 = 2;
        wp::float32 var_120;
        const wp::int32 var_121 = 2;
        wp::int32 var_122;
        const wp::int32 var_123 = 3;
        wp::float32 var_124;
        const wp::int32 var_125 = 3;
        wp::int32 var_126;
        const wp::int32 var_127 = 4;
        wp::float32 var_128;
        const wp::int32 var_129 = 4;
        wp::int32 var_130;
        const wp::int32 var_131 = 5;
        wp::float32 var_132;
        const wp::int32 var_133 = 5;
        wp::int32 var_134;
        const wp::int32 var_135 = 1;
        bool var_136;
        const wp::int32 var_137 = 0;
        wp::int32 var_138;
        wp::float32* var_139;
        const wp::int32 var_140 = 1;
        wp::int32 var_141;
        wp::float32* var_142;
        const wp::int32 var_143 = 2;
        wp::int32 var_144;
        wp::float32* var_145;
        const wp::int32 var_146 = 3;
        wp::int32 var_147;
        wp::float32* var_148;
        wp::quat_t<wp::float32> var_149;
        wp::float32 var_150;
        wp::float32 var_151;
        wp::float32 var_152;
        wp::float32 var_153;
        wp::quat_t<wp::float32> var_154;
        wp::vec_t<3, wp::float32> var_155;
        wp::vec_t<3, wp::float32> var_156;
        const wp::int32 var_157 = 1;
        bool var_158;
        wp::quat_t<wp::float32> var_159;
        wp::vec_t<3, wp::float32> var_160;
        wp::quat_t<wp::float32> var_161;
        wp::vec_t<3, wp::float32> var_162;
        wp::float32 var_163;
        const wp::int32 var_164 = 3;
        wp::int32 var_165;
        const wp::int32 var_166 = 0;
        wp::int32 var_167;
        wp::int32 var_168;
        wp::float32 var_169;
        const wp::int32 var_170 = 1;
        wp::int32 var_171;
        wp::int32 var_172;
        wp::float32 var_173;
        const wp::int32 var_174 = 2;
        wp::int32 var_175;
        wp::int32 var_176;
        wp::float32 var_177;
        wp::int32 var_178;
        wp::quat_t<wp::float32> var_179;
        wp::vec_t<3, wp::float32> var_180;
        const wp::int32 var_181 = 2;
        bool var_182;
        const wp::int32 var_183 = 3;
        bool var_184;
        bool var_185;
        wp::float32* var_186;
        const wp::int32 var_187 = 0;
        wp::float32 var_188;
        wp::float32 var_189;
        wp::float32 var_190;
        const wp::int32 var_191 = 1;
        wp::int32 var_192;
        const wp::int32 var_193 = 0;
        wp::float32 var_194;
        wp::int32 var_195;
        wp::int32 var_196;
        const wp::str var_197 = "unrecognized joint type";
        wp::int32 var_198;
        wp::int32 var_199;
        wp::int32 var_200;
        wp::quat_t<wp::float32> var_201;
        wp::vec_t<3, wp::float32> var_202;
        const wp::int32 var_203 = 2;
        bool var_204;
        wp::vec_t<2, wp::int32>* var_205;
        wp::vec_t<2, wp::int32> var_206;
        wp::vec_t<2, wp::int32> var_207;
        const wp::int32 var_208 = 0;
        wp::int32 var_209;
        const wp::int32 var_210 = 1;
        wp::int32 var_211;
        const wp::int32 var_212 = 0;
        wp::float32 var_213;
        wp::shape_t* var_214;
        const wp::int32 var_215 = 0;
        wp::int32 var_216;
        wp::shape_t var_217;
        wp::int32 var_218;
        wp::float32* var_219;
        wp::float32 var_220;
        wp::float32 var_221;
        wp::mat_t<3, 3, wp::float32>* var_222;
        wp::mat_t<3, 3, wp::float32> var_223;
        wp::mat_t<3, 3, wp::float32> var_224;
        const wp::int32 var_225 = 0;
        const wp::int32 var_226 = 2;
        wp::float32 var_227;
        const wp::int32 var_228 = 1;
        const wp::int32 var_229 = 2;
        wp::float32 var_230;
        const wp::int32 var_231 = 2;
        const wp::int32 var_232 = 2;
        wp::float32 var_233;
        wp::vec_t<3, wp::float32> var_234;
        wp::vec_t<3, wp::float32>* var_235;
        wp::vec_t<3, wp::float32> var_236;
        wp::vec_t<3, wp::float32> var_237;
        wp::vec_t<3, wp::float32>* var_238;
        wp::vec_t<3, wp::float32> var_239;
        wp::vec_t<3, wp::float32> var_240;
        wp::vec_t<3, wp::float32> var_241;
        wp::float32 var_242;
        wp::float32 var_243;
        wp::float32 var_244;
        wp::float32 var_245;
        wp::float32 var_246;
        wp::float32 var_247;
        const wp::int32 var_248 = 1;
        const wp::float32 var_249 = 0.0;
        bool var_250;
        const wp::int32 var_251 = 0;
        const wp::float32 var_252 = 0.0;
        wp::float32 var_253;
        wp::int32 var_254;
        wp::float32 var_255;
        wp::float32 var_256;
        wp::float32 var_257;
        wp::float32 var_258;
        wp::float32 var_259;
        const wp::int32 var_260 = 1;
        bool var_261;
        const wp::float32 var_262 = 1.0;
        wp::float32 var_263;
        wp::float32 var_264;
        wp::vec_t<3, wp::float32> var_265;
        wp::vec_t<3, wp::float32> var_266;
        wp::vec_t<3, wp::float32> var_267;
        wp::vec_t<3, wp::float32> var_268;
        wp::vec_t<3, wp::float32> var_269;
        wp::vec_t<3, wp::float32> var_270;
        wp::vec_t<3, wp::float32> var_271;
        wp::vec_t<3, wp::float32> var_272;
        wp::int32* var_273;
        wp::int32* var_274;
        wp::int32 var_275;
        wp::int32 var_276;
        wp::int32 var_277;
        wp::int32* var_278;
        wp::int32* var_279;
        wp::int32 var_280;
        wp::int32 var_281;
        wp::int32 var_282;
        const wp::int32 var_283 = 1;
        const wp::int32 var_284 = -1;
        wp::int32 var_285;
        const wp::int32 var_286 = 1;
        const wp::int32 var_287 = -1;
        wp::int32 var_288;
        const wp::int32 var_289 = 0;
        bool var_290;
        wp::int32* var_291;
        wp::int32* var_292;
        wp::int32 var_293;
        wp::int32 var_294;
        wp::int32 var_295;
        const wp::int32 var_296 = 1;
        wp::int32 var_297;
        wp::int32 var_298;
        const wp::int32 var_299 = 0;
        bool var_300;
        wp::int32* var_301;
        wp::int32* var_302;
        wp::int32 var_303;
        wp::int32 var_304;
        wp::int32 var_305;
        const wp::int32 var_306 = 1;
        wp::int32 var_307;
        wp::int32 var_308;
        wp::int32 var_309;
        wp::int32 var_310;
        const wp::int32 var_311 = 0;
        wp::int32 var_312;
        const wp::int32 var_313 = 0;
        bool var_314;
        const wp::int32 var_315 = 0;
        bool var_316;
        bool var_317;
        wp::int32 var_318;
        const wp::int32 var_319 = 1;
        wp::int32 var_320;
        bool var_321;
        wp::int32* var_322;
        wp::int32 var_323;
        wp::int32 var_324;
        wp::int32 var_325;
        bool var_326;
        wp::int32* var_327;
        wp::int32 var_328;
        wp::int32 var_329;
        wp::int32 var_330;
        wp::int32 var_331;
        wp::int32 var_332;
        wp::int32 var_333;
        const wp::int32 var_334 = 1;
        wp::int32 var_335;
        const wp::int32 var_336 = 0;
        bool var_337;
        const wp::int32 var_338 = 0;
        bool var_339;
        bool var_340;
        wp::int32 var_341;
        wp::int32* var_342;
        wp::vec_t<3, wp::float32> var_343;
        wp::vec_t<3, wp::float32> var_344;
        wp::int32 var_345;
        wp::vec_t<3, wp::float32> var_346;
        wp::vec_t<3, wp::float32> var_347;
        wp::int32* var_348;
        wp::vec_t<3, wp::float32> var_349;
        wp::vec_t<3, wp::float32> var_350;
        wp::int32 var_351;
        wp::vec_t<3, wp::float32> var_352;
        wp::float32 var_353;
        wp::float32 var_354;
        wp::float32 var_355;
        wp::int32 var_356;
        wp::float32 var_357;
        const wp::int32 var_358 = 1;
        wp::int32 var_359;
        bool var_360;
        wp::int32* var_361;
        wp::int32 var_362;
        wp::int32 var_363;
        wp::int32 var_364;
        bool var_365;
        wp::int32* var_366;
        wp::int32 var_367;
        wp::int32 var_368;
        wp::int32 var_369;
        wp::int32 var_370;
        const wp::int32 var_371 = 3;
        bool var_372;
        wp::vec_t<2, wp::int32>* var_373;
        const wp::int32 var_374 = 0;
        wp::int32 var_375;
        wp::vec_t<2, wp::int32> var_376;
        const wp::int32 var_377 = 0;
        wp::float32 var_378;
        wp::float32* var_379;
        wp::float32 var_380;
        wp::float32 var_381;
        wp::int32* var_382;
        wp::int32 var_383;
        wp::int32 var_384;
        wp::int32* var_385;
        wp::int32 var_386;
        wp::int32 var_387;
        wp::int32 var_388;
        wp::range_t var_389;
        wp::int32 var_390;
        wp::int32 var_391;
        wp::int32 var_392;
        wp::int32* var_393;
        wp::int32 var_394;
        wp::float32* var_395;
        wp::float32 var_396;
        wp::float32 var_397;
        wp::float32 var_398;
        const wp::int32 var_399 = 5;
        bool var_400;
        const wp::float32 var_401 = 0.0;
        wp::int32 var_402;
        wp::range_t var_403;
        wp::int32 var_404;
        wp::int32 var_405;
        const wp::float32 var_406 = 0.0;
        wp::int32 var_407;
        wp::int32 var_408;
        const wp::int32 var_409 = 4;
        bool var_410;
        wp::vec_t<2, wp::int32>* var_411;
        wp::vec_t<2, wp::int32> var_412;
        wp::vec_t<2, wp::int32> var_413;
        const wp::int32 var_414 = 0;
        wp::int32 var_415;
        const wp::int32 var_416 = 1;
        wp::int32 var_417;
        wp::vec_t<6, wp::float32>* var_418;
        wp::vec_t<6, wp::float32> var_419;
        wp::vec_t<6, wp::float32> var_420;
        wp::shape_t* var_421;
        const wp::int32 var_422 = 0;
        wp::int32 var_423;
        wp::shape_t var_424;
        wp::int32 var_425;
        wp::vec_t<3, wp::float32> var_426;
        wp::vec_t<3, wp::float32> var_427;
        const wp::int32 var_428 = 1;
        const wp::int32 var_429 = -1;
        bool var_430;
        wp::mat_t<3, 3, wp::float32>* var_431;
        wp::mat_t<3, 3, wp::float32> var_432;
        wp::mat_t<3, 3, wp::float32> var_433;
        wp::vec_t<3, wp::float32> var_434;
        wp::vec_t<3, wp::float32> var_435;
        wp::int32* var_436;
        wp::int32* var_437;
        wp::int32 var_438;
        wp::int32 var_439;
        wp::int32 var_440;
        const wp::int32 var_441 = 1;
        const wp::int32 var_442 = -1;
        wp::int32 var_443;
        const wp::int32 var_444 = 0;
        bool var_445;
        wp::int32* var_446;
        wp::int32* var_447;
        wp::int32 var_448;
        wp::int32 var_449;
        wp::int32 var_450;
        const wp::int32 var_451 = 1;
        wp::int32 var_452;
        wp::int32 var_453;
        wp::int32 var_454;
        const wp::int32 var_455 = 0;
        wp::int32 var_456;
        const wp::int32 var_457 = 0;
        bool var_458;
        const wp::int32 var_459 = 1;
        wp::int32 var_460;
        wp::int32* var_461;
        wp::int32 var_462;
        wp::int32 var_463;
        wp::int32 var_464;
        const wp::float32 var_465 = 0.0;
        wp::int32 var_466;
        const wp::int32 var_467 = 1;
        wp::int32 var_468;
        const wp::int32 var_469 = 0;
        bool var_470;
        wp::vec_t<3, wp::float32>* var_471;
        wp::int32* var_472;
        wp::vec_t<3, wp::float32> var_473;
        wp::vec_t<3, wp::float32> var_474;
        wp::vec_t<3, wp::float32> var_475;
        wp::int32 var_476;
        wp::float32 var_477;
        wp::float32 var_478;
        wp::float32 var_479;
        wp::int32 var_480;
        const wp::int32 var_481 = 1;
        wp::int32 var_482;
        wp::int32* var_483;
        wp::int32 var_484;
        wp::int32 var_485;
        wp::int32 var_486;
        wp::mat_t<3, 3, wp::float32> var_487;
        wp::int32 var_488;
        wp::int32 var_489;
        wp::int32 var_490;
        wp::int32 var_491;
        wp::int32* var_492;
        wp::int32 var_493;
        wp::int32 var_494;
        wp::int32* var_495;
        wp::int32 var_496;
        wp::int32 var_497;
        wp::int32* var_498;
        wp::int32 var_499;
        wp::int32 var_500;
        wp::int32* var_501;
        wp::int32 var_502;
        wp::int32 var_503;
        wp::int32* var_504;
        wp::int32* var_505;
        wp::int32 var_506;
        wp::int32 var_507;
        wp::int32 var_508;
        const wp::int32 var_509 = 1;
        wp::int32 var_510;
        wp::int32* var_511;
        wp::int32* var_512;
        wp::int32 var_513;
        wp::int32 var_514;
        wp::int32 var_515;
        const wp::int32 var_516 = 1;
        wp::int32 var_517;
        const wp::int32 var_518 = 1;
        const wp::int32 var_519 = -1;
        const wp::int32 var_520 = 0;
        bool var_521;
        const wp::int32 var_522 = 0;
        bool var_523;
        bool var_524;
        bool var_525;
        bool var_526;
        wp::int32* var_527;
        wp::int32 var_528;
        wp::int32 var_529;
        wp::int32 var_530;
        wp::int32* var_531;
        wp::int32 var_532;
        wp::int32 var_533;
        wp::int32 var_534;
        const wp::int32 var_535 = 1;
        const wp::int32 var_536 = -1;
        bool var_537;
        const wp::int32 var_538 = 1;
        const wp::int32 var_539 = -1;
        bool var_540;
        bool var_541;
        wp::int32 var_542;
        wp::int32 var_543;
        bool var_544;
        wp::int32 var_545;
        wp::int32 var_546;
        wp::int32 var_547;
        const wp::int32 var_548 = 0;
        wp::float32 var_549;
        const wp::float32 var_550 = 0.0;
        bool var_551;
        const wp::int32 var_552 = 1;
        wp::float32 var_553;
        const wp::float32 var_554 = 0.0;
        bool var_555;
        const wp::int32 var_556 = 2;
        wp::float32 var_557;
        const wp::float32 var_558 = 0.0;
        bool var_559;
        bool var_560;
        bool var_561;
        const wp::int32 var_562 = 3;
        wp::float32 var_563;
        const wp::float32 var_564 = 0.0;
        bool var_565;
        const wp::int32 var_566 = 4;
        wp::float32 var_567;
        const wp::float32 var_568 = 0.0;
        bool var_569;
        const wp::int32 var_570 = 5;
        wp::float32 var_571;
        const wp::float32 var_572 = 0.0;
        bool var_573;
        bool var_574;
        bool var_575;
        wp::vec_t<3, wp::float32>* var_576;
        wp::vec_t<3, wp::float32> var_577;
        wp::vec_t<3, wp::float32> var_578;
        wp::vec_t<3, wp::float32>* var_579;
        wp::vec_t<3, wp::float32> var_580;
        wp::vec_t<3, wp::float32> var_581;
        wp::mat_t<3, 3, wp::float32>* var_582;
        wp::mat_t<3, 3, wp::float32> var_583;
        wp::mat_t<3, 3, wp::float32> var_584;
        const wp::float32 var_585 = 0.0;
        wp::float32 var_586;
        wp::mat_t<3, 3, wp::float32> var_587;
        wp::vec_t<3, wp::float32> var_588;
        wp::vec_t<3, wp::float32> var_589;
        wp::float32 var_590;
        wp::float32 var_591;
        wp::vec_t<3, wp::float32> var_592;
        wp::vec_t<3, wp::float32> var_593;
        wp::float32 var_594;
        wp::vec_t<3, wp::float32> var_595;
        wp::quat_t<wp::float32>* var_596;
        wp::quat_t<wp::float32>* var_597;
        wp::quat_t<wp::float32> var_598;
        wp::quat_t<wp::float32> var_599;
        wp::quat_t<wp::float32> var_600;
        wp::quat_t<wp::float32>* var_601;
        wp::quat_t<wp::float32>* var_602;
        wp::quat_t<wp::float32> var_603;
        wp::quat_t<wp::float32> var_604;
        wp::quat_t<wp::float32> var_605;
        wp::vec_t<3, wp::float32> var_606;
        wp::float32 var_607;
        wp::float32 var_608;
        wp::vec_t<3, wp::float32> var_609;
        wp::quat_t<wp::float32> var_610;
        wp::vec_t<3, wp::float32> var_611;
        wp::float32 var_612;
        wp::vec_t<3, wp::float32> var_613;
        const wp::int32 var_614 = 1;
        const wp::int32 var_615 = -1;
        wp::int32 var_616;
        const wp::int32 var_617 = 1;
        const wp::int32 var_618 = -1;
        wp::int32 var_619;
        const wp::int32 var_620 = 0;
        bool var_621;
        wp::int32* var_622;
        wp::int32* var_623;
        wp::int32 var_624;
        wp::int32 var_625;
        wp::int32 var_626;
        const wp::int32 var_627 = 1;
        wp::int32 var_628;
        wp::int32 var_629;
        const wp::int32 var_630 = 0;
        bool var_631;
        wp::int32* var_632;
        wp::int32* var_633;
        wp::int32 var_634;
        wp::int32 var_635;
        wp::int32 var_636;
        const wp::int32 var_637 = 1;
        wp::int32 var_638;
        wp::int32 var_639;
        wp::int32 var_640;
        wp::int32 var_641;
        const wp::int32 var_642 = 0;
        wp::int32 var_643;
        const wp::int32 var_644 = 0;
        bool var_645;
        const wp::int32 var_646 = 0;
        bool var_647;
        bool var_648;
        wp::int32 var_649;
        bool var_650;
        bool var_651;
        bool var_652;
        wp::int32 var_653;
        const wp::int32 var_654 = 1;
        wp::int32 var_655;
        bool var_656;
        wp::int32* var_657;
        wp::int32 var_658;
        wp::int32 var_659;
        wp::int32 var_660;
        bool var_661;
        wp::int32* var_662;
        wp::int32 var_663;
        wp::int32 var_664;
        wp::int32 var_665;
        wp::int32 var_666;
        wp::int32 var_667;
        wp::int32 var_668;
        const wp::int32 var_669 = 1;
        wp::int32 var_670;
        const wp::int32 var_671 = 0;
        bool var_672;
        const wp::int32 var_673 = 0;
        bool var_674;
        bool var_675;
        wp::int32 var_676;
        bool var_677;
        bool var_678;
        bool var_679;
        wp::int32 var_680;
        wp::int32* var_681;
        wp::vec_t<3, wp::float32> var_682;
        wp::vec_t<3, wp::float32> var_683;
        wp::int32 var_684;
        wp::int32* var_685;
        wp::vec_t<3, wp::float32> var_686;
        wp::vec_t<3, wp::float32> var_687;
        wp::int32 var_688;
        const wp::float32 var_689 = 0.0;
        wp::float32 var_690;
        wp::vec_t<3, wp::float32> var_691;
        wp::float32 var_692;
        wp::float32 var_693;
        wp::float32 var_694;
        wp::vec_t<3, wp::float32> var_695;
        wp::float32 var_696;
        wp::float32 var_697;
        wp::float32 var_698;
        wp::int32 var_699;
        const wp::int32 var_700 = 1;
        wp::int32 var_701;
        bool var_702;
        wp::int32* var_703;
        wp::int32 var_704;
        wp::int32 var_705;
        wp::int32 var_706;
        bool var_707;
        wp::int32* var_708;
        wp::int32 var_709;
        wp::int32 var_710;
        wp::int32 var_711;
        wp::int32 var_712;
        wp::quat_t<wp::float32> var_713;
        wp::vec_t<3, wp::float32> var_714;
        wp::float32 var_715;
        wp::int32 var_716;
        wp::int32 var_717;
        wp::int32 var_718;
        wp::int32 var_719;
        wp::int32 var_720;
        wp::int32 var_721;
        wp::int32 var_722;
        wp::vec_t<3, wp::float32> var_723;
        wp::vec_t<3, wp::float32> var_724;
        wp::vec_t<6, wp::float32> var_725;
        wp::int32 var_726;
        wp::quat_t<wp::float32> var_727;
        wp::vec_t<2, wp::int32> var_728;
        wp::mat_t<3, 3, wp::float32> var_729;
        wp::vec_t<3, wp::float32> var_730;
        wp::float32 var_731;
        wp::int32 var_732;
        wp::int32 var_733;
        wp::int32 var_734;
        wp::int32 var_735;
        wp::int32 var_736;
        wp::int32 var_737;
        wp::int32 var_738;
        wp::int32 var_739;
        const wp::str var_740 = "unhandled transmission type %d\n";
        wp::vec_t<6, wp::float32> var_741;
        wp::int32 var_742;
        wp::quat_t<wp::float32> var_743;
        wp::vec_t<2, wp::int32> var_744;
        wp::mat_t<3, 3, wp::float32> var_745;
        wp::vec_t<3, wp::float32> var_746;
        wp::float32 var_747;
        wp::int32 var_748;
        wp::int32 var_749;
        wp::int32 var_750;
        wp::int32 var_751;
        wp::int32 var_752;
        wp::int32 var_753;
        wp::int32 var_754;
        wp::int32 var_755;
        wp::vec_t<6, wp::float32> var_756;
        wp::int32 var_757;
        wp::quat_t<wp::float32> var_758;
        wp::int32 var_759;
        wp::vec_t<2, wp::int32> var_760;
        wp::mat_t<3, 3, wp::float32> var_761;
        wp::vec_t<3, wp::float32> var_762;
        wp::float32 var_763;
        wp::int32 var_764;
        wp::int32 var_765;
        wp::int32 var_766;
        wp::int32 var_767;
        wp::int32 var_768;
        wp::int32 var_769;
        wp::int32 var_770;
        wp::int32 var_771;
        wp::vec_t<6, wp::float32> var_772;
        wp::int32 var_773;
        wp::quat_t<wp::float32> var_774;
        wp::int32 var_775;
        wp::vec_t<2, wp::int32> var_776;
        wp::float32 var_777;
        wp::mat_t<3, 3, wp::float32> var_778;
        wp::vec_t<3, wp::float32> var_779;
        wp::float32 var_780;
        wp::int32 var_781;
        wp::int32 var_782;
        wp::int32 var_783;
        wp::int32 var_784;
        wp::int32 var_785;
        wp::int32 var_786;
        wp::int32 var_787;
        wp::int32 var_788;
        wp::vec_t<6, wp::float32> var_789;
        wp::int32 var_790;
        wp::quat_t<wp::float32> var_791;
        wp::int32 var_792;
        //---------
        // forward
        // def _transmission(                                                                     <L 2042>
        // worldid, actid = wp.tid()                                                              <L 2082>
        builtin_tid2d(var_0, var_1);
        // trntype = actuator_trntype[actid]                                                      <L 2083>
        var_2 = wp::address(var_actuator_trntype, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // actuator_gear_id = worldid % actuator_gear.shape[0]                                    <L 2084>
        var_5 = &(var_actuator_gear.shape);
        var_8 = wp::load(var_5);
        var_7 = wp::extract(var_8, var_6);
        var_9 = wp::mod(var_0, var_7);
        // gear = actuator_gear[actuator_gear_id, actid]                                          <L 2085>
        var_10 = wp::address(var_actuator_gear, var_9, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // if trntype == TrnType.JOINT or trntype == TrnType.JOINTINPARENT:                       <L 2086>
        var_14 = (var_3 == var_13);
        var_16 = (var_3 == var_15);
        var_17 = var_14 || var_16;
        if (var_17) {
            // qpos = qpos_in[worldid]                                                            <L 2087>
            var_18 = wp::slice_t(var_0, var_0, var_19);
            var_20 = wp::view(var_qpos_in, var_18);
            // jntid = actuator_trnid[actid][0]                                                   <L 2088>
            var_21 = wp::address(var_actuator_trnid, var_1);
            var_24 = wp::load(var_21);
            var_23 = wp::extract(var_24, var_22);
            // jnt_typ = jnt_type[jntid]                                                          <L 2089>
            var_25 = wp::address(var_jnt_type, var_23);
            var_27 = wp::load(var_25);
            var_26 = wp::copy(var_27);
            // qadr = jnt_qposadr[jntid]                                                          <L 2090>
            var_28 = wp::address(var_jnt_qposadr, var_23);
            var_30 = wp::load(var_28);
            var_29 = wp::copy(var_30);
            // vadr = jnt_dofadr[jntid]                                                           <L 2091>
            var_31 = wp::address(var_jnt_dofadr, var_23);
            var_33 = wp::load(var_31);
            var_32 = wp::copy(var_33);
            // if jnt_typ == JointType.FREE:                                                      <L 2092>
            var_35 = (var_26 == var_34);
            if (var_35) {
                // moment_rownnz_out[worldid, actid] = 6                                          <L 2093>
                wp::array_store(var_moment_rownnz_out, var_0, var_1, var_36);
                // rowadr = wp.atomic_add(moment_nnz, worldid, 6)                                 <L 2094>
                var_38 = wp::atomic_add(var_moment_nnz, var_0, var_37);
                // moment_rowadr_out[worldid, actid] = rowadr                                     <L 2095>
                wp::array_store(var_moment_rowadr_out, var_0, var_1, var_38);
                // moment_colind_out[worldid, rowadr + 0] = vadr + 0                              <L 2096>
                var_40 = wp::add(var_32, var_39);
                var_42 = wp::add(var_38, var_41);
                wp::array_store(var_moment_colind_out, var_0, var_42, var_40);
                // moment_colind_out[worldid, rowadr + 1] = vadr + 1                              <L 2097>
                var_44 = wp::add(var_32, var_43);
                var_46 = wp::add(var_38, var_45);
                wp::array_store(var_moment_colind_out, var_0, var_46, var_44);
                // moment_colind_out[worldid, rowadr + 2] = vadr + 2                              <L 2098>
                var_48 = wp::add(var_32, var_47);
                var_50 = wp::add(var_38, var_49);
                wp::array_store(var_moment_colind_out, var_0, var_50, var_48);
                // moment_colind_out[worldid, rowadr + 3] = vadr + 3                              <L 2099>
                var_52 = wp::add(var_32, var_51);
                var_54 = wp::add(var_38, var_53);
                wp::array_store(var_moment_colind_out, var_0, var_54, var_52);
                // moment_colind_out[worldid, rowadr + 4] = vadr + 4                              <L 2100>
                var_56 = wp::add(var_32, var_55);
                var_58 = wp::add(var_38, var_57);
                wp::array_store(var_moment_colind_out, var_0, var_58, var_56);
                // moment_colind_out[worldid, rowadr + 5] = vadr + 5                              <L 2101>
                var_60 = wp::add(var_32, var_59);
                var_62 = wp::add(var_38, var_61);
                wp::array_store(var_moment_colind_out, var_0, var_62, var_60);
                // actuator_length_out[worldid, actid] = 0.0                                      <L 2102>
                wp::array_store(var_actuator_length_out, var_0, var_1, var_63);
                // if trntype == TrnType.JOINTINPARENT:                                           <L 2103>
                var_65 = (var_3 == var_64);
                if (var_65) {
                    // quat = wp.normalize(wp.quat(qpos[qadr + 3], qpos[qadr + 4], qpos[qadr + 5], qpos[qadr + 6]))       <L 2104>
                    var_67 = wp::add(var_29, var_66);
                    var_68 = wp::address(var_20, var_67);
                    var_70 = wp::add(var_29, var_69);
                    var_71 = wp::address(var_20, var_70);
                    var_73 = wp::add(var_29, var_72);
                    var_74 = wp::address(var_20, var_73);
                    var_76 = wp::add(var_29, var_75);
                    var_77 = wp::address(var_20, var_76);
                    var_79 = wp::load(var_68);
                    var_80 = wp::load(var_71);
                    var_81 = wp::load(var_74);
                    var_82 = wp::load(var_77);
                    var_78 = wp::quat_t<wp::float32>(var_79, var_80, var_81, var_82);
                    var_83 = wp::normalize(var_78);
                    // quat_neg = math.quat_inv(quat)                                             <L 2105>
                    var_84 = quat_inv_0(var_83);
                    // gearaxis = math.rot_vec_quat(wp.spatial_bottom(gear), quat_neg)            <L 2106>
                    var_85 = wp::spatial_bottom(var_11);
                    var_86 = rot_vec_quat_0(var_85, var_84);
                    // actuator_moment_out[worldid, rowadr + 0] = gear[0]                         <L 2107>
                    var_88 = wp::extract(var_11, var_87);
                    var_90 = wp::add(var_38, var_89);
                    wp::array_store(var_actuator_moment_out, var_0, var_90, var_88);
                    // actuator_moment_out[worldid, rowadr + 1] = gear[1]                         <L 2108>
                    var_92 = wp::extract(var_11, var_91);
                    var_94 = wp::add(var_38, var_93);
                    wp::array_store(var_actuator_moment_out, var_0, var_94, var_92);
                    // actuator_moment_out[worldid, rowadr + 2] = gear[2]                         <L 2109>
                    var_96 = wp::extract(var_11, var_95);
                    var_98 = wp::add(var_38, var_97);
                    wp::array_store(var_actuator_moment_out, var_0, var_98, var_96);
                    // actuator_moment_out[worldid, rowadr + 3] = gearaxis[0]                     <L 2110>
                    var_100 = wp::extract(var_86, var_99);
                    var_102 = wp::add(var_38, var_101);
                    wp::array_store(var_actuator_moment_out, var_0, var_102, var_100);
                    // actuator_moment_out[worldid, rowadr + 4] = gearaxis[1]                     <L 2111>
                    var_104 = wp::extract(var_86, var_103);
                    var_106 = wp::add(var_38, var_105);
                    wp::array_store(var_actuator_moment_out, var_0, var_106, var_104);
                    // actuator_moment_out[worldid, rowadr + 5] = gearaxis[2]                     <L 2112>
                    var_108 = wp::extract(var_86, var_107);
                    var_110 = wp::add(var_38, var_109);
                    wp::array_store(var_actuator_moment_out, var_0, var_110, var_108);
                }
                if (!var_65) {
                    // actuator_moment_out[worldid, rowadr + 0] = gear[0]                         <L 2114>
                    var_112 = wp::extract(var_11, var_111);
                    var_114 = wp::add(var_38, var_113);
                    wp::array_store(var_actuator_moment_out, var_0, var_114, var_112);
                    // actuator_moment_out[worldid, rowadr + 1] = gear[1]                         <L 2115>
                    var_116 = wp::extract(var_11, var_115);
                    var_118 = wp::add(var_38, var_117);
                    wp::array_store(var_actuator_moment_out, var_0, var_118, var_116);
                    // actuator_moment_out[worldid, rowadr + 2] = gear[2]                         <L 2116>
                    var_120 = wp::extract(var_11, var_119);
                    var_122 = wp::add(var_38, var_121);
                    wp::array_store(var_actuator_moment_out, var_0, var_122, var_120);
                    // actuator_moment_out[worldid, rowadr + 3] = gear[3]                         <L 2117>
                    var_124 = wp::extract(var_11, var_123);
                    var_126 = wp::add(var_38, var_125);
                    wp::array_store(var_actuator_moment_out, var_0, var_126, var_124);
                    // actuator_moment_out[worldid, rowadr + 4] = gear[4]                         <L 2118>
                    var_128 = wp::extract(var_11, var_127);
                    var_130 = wp::add(var_38, var_129);
                    wp::array_store(var_actuator_moment_out, var_0, var_130, var_128);
                    // actuator_moment_out[worldid, rowadr + 5] = gear[5]                         <L 2119>
                    var_132 = wp::extract(var_11, var_131);
                    var_134 = wp::add(var_38, var_133);
                    wp::array_store(var_actuator_moment_out, var_0, var_134, var_132);
                }
            }
            if (!var_35) {
                // elif jnt_typ == JointType.BALL:                                                <L 2120>
                var_136 = (var_26 == var_135);
                if (var_136) {
                    // q = wp.quat(qpos[qadr + 0], qpos[qadr + 1], qpos[qadr + 2], qpos[qadr + 3])       <L 2121>
                    var_138 = wp::add(var_29, var_137);
                    var_139 = wp::address(var_20, var_138);
                    var_141 = wp::add(var_29, var_140);
                    var_142 = wp::address(var_20, var_141);
                    var_144 = wp::add(var_29, var_143);
                    var_145 = wp::address(var_20, var_144);
                    var_147 = wp::add(var_29, var_146);
                    var_148 = wp::address(var_20, var_147);
                    var_150 = wp::load(var_139);
                    var_151 = wp::load(var_142);
                    var_152 = wp::load(var_145);
                    var_153 = wp::load(var_148);
                    var_149 = wp::quat_t<wp::float32>(var_150, var_151, var_152, var_153);
                    // q = wp.normalize(q)                                                        <L 2122>
                    var_154 = wp::normalize(var_149);
                    // axis_angle = math.quat_to_vel(q)                                           <L 2123>
                    var_155 = quat_to_vel_0(var_154);
                    // gearaxis = wp.spatial_top(gear)  # [:3]                                    <L 2124>
                    var_156 = wp::spatial_top(var_11);
                    // if trntype == TrnType.JOINTINPARENT:                                       <L 2125>
                    var_158 = (var_3 == var_157);
                    if (var_158) {
                        // quat_neg = math.quat_inv(q)                                            <L 2126>
                        var_159 = quat_inv_0(var_154);
                        // gearaxis = math.rot_vec_quat(gearaxis, quat_neg)                       <L 2127>
                        var_160 = rot_vec_quat_0(var_156, var_159);
                    }
                    var_161 = wp::where(var_158, var_159, var_84);
                    var_162 = wp::where(var_158, var_160, var_156);
                    // actuator_length_out[worldid, actid] = wp.dot(axis_angle, gearaxis)         <L 2128>
                    var_163 = wp::dot(var_155, var_162);
                    wp::array_store(var_actuator_length_out, var_0, var_1, var_163);
                    // nnz = 3                                                                    <L 2130>
                    // moment_rownnz_out[worldid, actid] = nnz                                    <L 2131>
                    wp::array_store(var_moment_rownnz_out, var_0, var_1, var_164);
                    // rowadr = wp.atomic_add(moment_nnz, worldid, nnz)                           <L 2132>
                    var_165 = wp::atomic_add(var_moment_nnz, var_0, var_164);
                    // moment_rowadr_out[worldid, actid] = rowadr                                 <L 2133>
                    wp::array_store(var_moment_rowadr_out, var_0, var_1, var_165);
                    // for i in range(3):                                                         <L 2135>
                    // sparseid = rowadr + i                                                      <L 2136>
                    var_167 = wp::add(var_165, var_166);
                    // moment_colind_out[worldid, sparseid] = vadr + i                            <L 2137>
                    var_168 = wp::add(var_32, var_166);
                    wp::array_store(var_moment_colind_out, var_0, var_167, var_168);
                    // actuator_moment_out[worldid, sparseid] = gearaxis[i]                       <L 2138>
                    var_169 = wp::extract(var_162, var_166);
                    wp::array_store(var_actuator_moment_out, var_0, var_167, var_169);
                    // sparseid = rowadr + i                                                      <L 2136>
                    var_171 = wp::add(var_165, var_170);
                    // moment_colind_out[worldid, sparseid] = vadr + i                            <L 2137>
                    var_172 = wp::add(var_32, var_170);
                    wp::array_store(var_moment_colind_out, var_0, var_171, var_172);
                    // actuator_moment_out[worldid, sparseid] = gearaxis[i]                       <L 2138>
                    var_173 = wp::extract(var_162, var_170);
                    wp::array_store(var_actuator_moment_out, var_0, var_171, var_173);
                    // sparseid = rowadr + i                                                      <L 2136>
                    var_175 = wp::add(var_165, var_174);
                    // moment_colind_out[worldid, sparseid] = vadr + i                            <L 2137>
                    var_176 = wp::add(var_32, var_174);
                    wp::array_store(var_moment_colind_out, var_0, var_175, var_176);
                    // actuator_moment_out[worldid, sparseid] = gearaxis[i]                       <L 2138>
                    var_177 = wp::extract(var_162, var_174);
                    wp::array_store(var_actuator_moment_out, var_0, var_175, var_177);
                }
                var_178 = wp::where(var_136, var_165, var_38);
                var_179 = wp::where(var_136, var_161, var_84);
                var_180 = wp::where(var_136, var_162, var_86);
                if (!var_136) {
                    // elif jnt_typ == JointType.SLIDE or jnt_typ == JointType.HINGE:             <L 2139>
                    var_182 = (var_26 == var_181);
                    var_184 = (var_26 == var_183);
                    var_185 = var_182 || var_184;
                    if (var_185) {
                        // actuator_length_out[worldid, actid] = qpos[qadr] * gear[0]             <L 2140>
                        var_186 = wp::address(var_20, var_29);
                        var_188 = wp::extract(var_11, var_187);
                        var_190 = wp::load(var_186);
                        var_189 = wp::mul(var_190, var_188);
                        wp::array_store(var_actuator_length_out, var_0, var_1, var_189);
                        // nnz = 1                                                                <L 2142>
                        // moment_rownnz_out[worldid, actid] = nnz                                <L 2143>
                        wp::array_store(var_moment_rownnz_out, var_0, var_1, var_191);
                        // rowadr = wp.atomic_add(moment_nnz, worldid, nnz)                       <L 2144>
                        var_192 = wp::atomic_add(var_moment_nnz, var_0, var_191);
                        // moment_rowadr_out[worldid, actid] = rowadr                             <L 2145>
                        wp::array_store(var_moment_rowadr_out, var_0, var_1, var_192);
                        // moment_colind_out[worldid, rowadr] = vadr                              <L 2146>
                        wp::array_store(var_moment_colind_out, var_0, var_192, var_32);
                        // actuator_moment_out[worldid, rowadr] = gear[0]                         <L 2147>
                        var_194 = wp::extract(var_11, var_193);
                        wp::array_store(var_actuator_moment_out, var_0, var_192, var_194);
                    }
                    var_195 = wp::where(var_185, var_192, var_178);
                    var_196 = wp::where(var_185, var_191, var_164);
                    if (!var_185) {
                        // wp.printf("unrecognized joint type")                                   <L 2149>
                        printf(var_197);
                    }
                }
                var_198 = wp::where(var_136, var_178, var_195);
                var_199 = wp::where(var_136, var_164, var_196);
            }
            var_200 = wp::where(var_35, var_38, var_198);
            var_201 = wp::where(var_35, var_84, var_179);
            var_202 = wp::where(var_35, var_86, var_180);
        }
        if (!var_17) {
            // elif trntype == TrnType.SLIDERCRANK:                                               <L 2150>
            var_204 = (var_3 == var_203);
            if (var_204) {
                // trnid = actuator_trnid[actid]                                                  <L 2152>
                var_205 = wp::address(var_actuator_trnid, var_1);
                var_207 = wp::load(var_205);
                var_206 = wp::copy(var_207);
                // id = trnid[0]                                                                  <L 2153>
                var_209 = wp::extract(var_206, var_208);
                // idslider = trnid[1]                                                            <L 2154>
                var_211 = wp::extract(var_206, var_210);
                // gear0 = gear[0]                                                                <L 2155>
                var_213 = wp::extract(var_11, var_212);
                // rod = actuator_cranklength[worldid % actuator_cranklength.shape[0], actid]       <L 2156>
                var_214 = &(var_actuator_cranklength.shape);
                var_217 = wp::load(var_214);
                var_216 = wp::extract(var_217, var_215);
                var_218 = wp::mod(var_0, var_216);
                var_219 = wp::address(var_actuator_cranklength, var_218, var_1);
                var_221 = wp::load(var_219);
                var_220 = wp::copy(var_221);
                // site_xmat = site_xmat_in[worldid, idslider]                                    <L 2157>
                var_222 = wp::address(var_site_xmat_in, var_0, var_211);
                var_224 = wp::load(var_222);
                var_223 = wp::copy(var_224);
                // axis = wp.vec3(site_xmat[0, 2], site_xmat[1, 2], site_xmat[2, 2])              <L 2158>
                var_227 = wp::extract(var_223, var_225, var_226);
                var_230 = wp::extract(var_223, var_228, var_229);
                var_233 = wp::extract(var_223, var_231, var_232);
                var_234 = wp::vec_t<3, wp::float32>(var_227, var_230, var_233);
                // site_xpos_id = site_xpos_in[worldid, id]                                       <L 2159>
                var_235 = wp::address(var_site_xpos_in, var_0, var_209);
                var_237 = wp::load(var_235);
                var_236 = wp::copy(var_237);
                // site_xpos_idslider = site_xpos_in[worldid, idslider]                           <L 2160>
                var_238 = wp::address(var_site_xpos_in, var_0, var_211);
                var_240 = wp::load(var_238);
                var_239 = wp::copy(var_240);
                // vec = site_xpos_id - site_xpos_idslider                                        <L 2161>
                var_241 = wp::sub(var_236, var_239);
                // av = wp.dot(vec, axis)                                                         <L 2165>
                var_242 = wp::dot(var_241, var_234);
                // det = av * av + rod * rod - wp.dot(vec, vec)                                   <L 2166>
                var_243 = wp::mul(var_242, var_242);
                var_244 = wp::mul(var_220, var_220);
                var_245 = wp::add(var_243, var_244);
                var_246 = wp::dot(var_241, var_241);
                var_247 = wp::sub(var_245, var_246);
                // ok = 1                                                                         <L 2167>
                // if det <= 0.0:                                                                 <L 2168>
                var_250 = (var_247 <= var_249);
                if (var_250) {
                    // ok = 0                                                                     <L 2169>
                    // sdet = 0.0                                                                 <L 2170>
                    // length = av                                                                <L 2171>
                    var_253 = wp::copy(var_242);
                }
                var_254 = wp::where(var_250, var_251, var_248);
                if (!var_250) {
                    // sdet = wp.sqrt(det)                                                        <L 2173>
                    var_255 = wp::sqrt(var_247);
                    // length = av - sdet                                                         <L 2174>
                    var_256 = wp::sub(var_242, var_255);
                }
                var_257 = wp::where(var_250, var_252, var_255);
                var_258 = wp::where(var_250, var_253, var_256);
                // actuator_length_out[worldid, actid] = length * gear0                           <L 2176>
                var_259 = wp::mul(var_258, var_213);
                wp::array_store(var_actuator_length_out, var_0, var_1, var_259);
                // if ok == 1:                                                                    <L 2179>
                var_261 = (var_254 == var_260);
                if (var_261) {
                    // scale = 1.0 - math.safe_div(av, sdet)                                      <L 2180>
                    var_263 = safe_div_0(var_242, var_257);
                    var_264 = wp::sub(var_262, var_263);
                    // dldv = axis * scale + math.safe_div(vec, sdet)                             <L 2181>
                    var_265 = wp::mul(var_234, var_264);
                    var_266 = safe_div_0(var_241, var_257);
                    var_267 = wp::add(var_265, var_266);
                    // dlda = vec * scale                                                         <L 2182>
                    var_268 = wp::mul(var_241, var_264);
                }
                if (!var_261) {
                    // dldv = axis                                                                <L 2184>
                    var_269 = wp::copy(var_234);
                    // dlda = vec                                                                 <L 2185>
                    var_270 = wp::copy(var_241);
                }
                var_271 = wp::where(var_261, var_267, var_269);
                var_272 = wp::where(var_261, var_268, var_270);
                // b1 = body_weldid[site_bodyid[id]]                                              <L 2188>
                var_273 = wp::address(var_site_bodyid, var_209);
                var_275 = wp::load(var_273);
                var_274 = wp::address(var_body_weldid, var_275);
                var_277 = wp::load(var_274);
                var_276 = wp::copy(var_277);
                // b2 = body_weldid[site_bodyid[idslider]]                                        <L 2189>
                var_278 = wp::address(var_site_bodyid, var_211);
                var_280 = wp::load(var_278);
                var_279 = wp::address(var_body_weldid, var_280);
                var_282 = wp::load(var_279);
                var_281 = wp::copy(var_282);
                // da1_init = int(-1)                                                             <L 2190>
                var_285 = wp::int(var_284);
                // da2_init = int(-1)                                                             <L 2191>
                var_288 = wp::int(var_287);
                // if b1 > 0:                                                                     <L 2192>
                var_290 = (var_276 > var_289);
                if (var_290) {
                    // da1_init = body_dofadr[b1] + body_dofnum[b1] - 1                           <L 2193>
                    var_291 = wp::address(var_body_dofadr, var_276);
                    var_292 = wp::address(var_body_dofnum, var_276);
                    var_294 = wp::load(var_291);
                    var_295 = wp::load(var_292);
                    var_293 = wp::add(var_294, var_295);
                    var_297 = wp::sub(var_293, var_296);
                }
                var_298 = wp::where(var_290, var_297, var_285);
                // if b2 > 0:                                                                     <L 2194>
                var_300 = (var_281 > var_299);
                if (var_300) {
                    // da2_init = body_dofadr[b2] + body_dofnum[b2] - 1                           <L 2195>
                    var_301 = wp::address(var_body_dofadr, var_281);
                    var_302 = wp::address(var_body_dofnum, var_281);
                    var_304 = wp::load(var_301);
                    var_305 = wp::load(var_302);
                    var_303 = wp::add(var_304, var_305);
                    var_307 = wp::sub(var_303, var_306);
                }
                var_308 = wp::where(var_300, var_307, var_288);
                // da1 = da1_init                                                                 <L 2197>
                var_309 = wp::copy(var_298);
                // da2 = da2_init                                                                 <L 2198>
                var_310 = wp::copy(var_308);
                // ndof = int(0)                                                                  <L 2199>
                var_312 = wp::int(var_311);
                // while da1 >= 0 or da2 >= 0:                                                    <L 2200>
        start_while_0:;
                var_314 = (var_309 >= var_313);
                var_316 = (var_310 >= var_315);
                var_317 = var_314 || var_316;
        if ((var_317) == false) goto end_while_0;
                    // da = wp.max(da1, da2)                                                      <L 2201>
                    var_318 = wp::max(var_309, var_310);
                    // ndof += 1                                                                  <L 2202>
                    var_320 = wp::add(var_312, var_319);
                    // if da1 == da:                                                              <L 2203>
                    var_321 = (var_309 == var_318);
                    if (var_321) {
                        // da1 = dof_parentid[da1]                                                <L 2204>
                        var_322 = wp::address(var_dof_parentid, var_309);
                        var_324 = wp::load(var_322);
                        var_323 = wp::copy(var_324);
                    }
                    var_325 = wp::where(var_321, var_323, var_309);
                    // if da2 == da:                                                              <L 2205>
                    var_326 = (var_310 == var_318);
                    if (var_326) {
                        // da2 = dof_parentid[da2]                                                <L 2206>
                        var_327 = wp::address(var_dof_parentid, var_310);
                        var_329 = wp::load(var_327);
                        var_328 = wp::copy(var_329);
                    }
                    var_330 = wp::where(var_326, var_328, var_310);
                    wp::assign(var_309, var_325);
                    wp::assign(var_310, var_330);
                    wp::assign(var_312, var_320);
        goto start_while_0;
        end_while_0:;
                // moment_rownnz_out[worldid, actid] = ndof                                       <L 2208>
                wp::array_store(var_moment_rownnz_out, var_0, var_1, var_312);
                // rowadr = wp.atomic_add(moment_nnz, worldid, ndof)                              <L 2209>
                var_331 = wp::atomic_add(var_moment_nnz, var_0, var_312);
                // moment_rowadr_out[worldid, actid] = rowadr                                     <L 2210>
                wp::array_store(var_moment_rowadr_out, var_0, var_1, var_331);
                // da1 = da1_init                                                                 <L 2213>
                var_332 = wp::copy(var_298);
                // da2 = da2_init                                                                 <L 2214>
                var_333 = wp::copy(var_308);
                // ptr = ndof - 1                                                                 <L 2216>
                var_335 = wp::sub(var_312, var_334);
                // while da1 >= 0 or da2 >= 0:                                                    <L 2217>
        start_while_2:;
                var_337 = (var_332 >= var_336);
                var_339 = (var_333 >= var_338);
                var_340 = var_337 || var_339;
        if ((var_340) == false) goto end_while_2;
                    // da = wp.max(da1, da2)                                                      <L 2218>
                    var_341 = wp::max(var_332, var_333);
                    // jacp, jacr = support.jac_dof(                                              <L 2221>
                    // body_parentid, body_rootid, dof_bodyid, subtree_com_in, cdof_in, site_xpos_idslider, site_bodyid[idslider], da, worldid       <L 2222>
                    var_342 = wp::address(var_site_bodyid, var_211);
                    var_345 = wp::load(var_342);
                    jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_subtree_com_in, var_cdof_in, var_239, var_345, var_341, var_0, var_343, var_344);
                    // jacS = jacp                                                                <L 2224>
                    var_346 = wp::copy(var_343);
                    // jacA = wp.cross(jacr, axis)                                                <L 2225>
                    var_347 = wp::cross(var_344, var_234);
                    // jac, _ = support.jac_dof(                                                  <L 2226>
                    // body_parentid, body_rootid, dof_bodyid, subtree_com_in, cdof_in, site_xpos_id, site_bodyid[id], da, worldid       <L 2227>
                    var_348 = wp::address(var_site_bodyid, var_209);
                    var_351 = wp::load(var_348);
                    jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_subtree_com_in, var_cdof_in, var_236, var_351, var_341, var_0, var_349, var_350);
                    // jac -= jacS                                                                <L 2229>
                    var_352 = wp::sub(var_349, var_346);
                    // moment = wp.dot(dlda, jacA) + wp.dot(dldv, jac)                            <L 2232>
                    var_353 = wp::dot(var_272, var_347);
                    var_354 = wp::dot(var_271, var_352);
                    var_355 = wp::add(var_353, var_354);
                    // sparseid = rowadr + ptr                                                    <L 2233>
                    var_356 = wp::add(var_331, var_335);
                    // moment_colind_out[worldid, sparseid] = da                                  <L 2234>
                    wp::array_store(var_moment_colind_out, var_0, var_356, var_341);
                    // actuator_moment_out[worldid, sparseid] = moment * gear0                    <L 2235>
                    var_357 = wp::mul(var_355, var_213);
                    wp::array_store(var_actuator_moment_out, var_0, var_356, var_357);
                    // ptr -= 1                                                                   <L 2236>
                    var_359 = wp::sub(var_335, var_358);
                    // if da1 == da:                                                              <L 2238>
                    var_360 = (var_332 == var_341);
                    if (var_360) {
                        // da1 = dof_parentid[da1]                                                <L 2239>
                        var_361 = wp::address(var_dof_parentid, var_332);
                        var_363 = wp::load(var_361);
                        var_362 = wp::copy(var_363);
                    }
                    var_364 = wp::where(var_360, var_362, var_332);
                    // if da2 == da:                                                              <L 2240>
                    var_365 = (var_333 == var_341);
                    if (var_365) {
                        // da2 = dof_parentid[da2]                                                <L 2241>
                        var_366 = wp::address(var_dof_parentid, var_333);
                        var_368 = wp::load(var_366);
                        var_367 = wp::copy(var_368);
                    }
                    var_369 = wp::where(var_365, var_367, var_333);
                    wp::assign(var_175, var_356);
                    wp::assign(var_332, var_364);
                    wp::assign(var_333, var_369);
                    wp::assign(var_318, var_341);
                    wp::assign(var_335, var_359);
        goto start_while_2;
        end_while_2:;
            }
            var_370 = wp::where(var_204, var_331, var_200);
            if (!var_204) {
                // elif trntype == TrnType.TENDON:                                                <L 2242>
                var_372 = (var_3 == var_371);
                if (var_372) {
                    // tenid = actuator_trnid[actid][0]                                           <L 2243>
                    var_373 = wp::address(var_actuator_trnid, var_1);
                    var_376 = wp::load(var_373);
                    var_375 = wp::extract(var_376, var_374);
                    // gear0 = gear[0]                                                            <L 2245>
                    var_378 = wp::extract(var_11, var_377);
                    // actuator_length_out[worldid, actid] = ten_length_in[worldid, tenid] * gear0       <L 2246>
                    var_379 = wp::address(var_ten_length_in, var_0, var_375);
                    var_381 = wp::load(var_379);
                    var_380 = wp::mul(var_381, var_378);
                    wp::array_store(var_actuator_length_out, var_0, var_1, var_380);
                    // rownnz_ten = ten_J_rownnz[tenid]                                           <L 2248>
                    var_382 = wp::address(var_ten_J_rownnz, var_375);
                    var_384 = wp::load(var_382);
                    var_383 = wp::copy(var_384);
                    // rowadr_ten = ten_J_rowadr[tenid]                                           <L 2249>
                    var_385 = wp::address(var_ten_J_rowadr, var_375);
                    var_387 = wp::load(var_385);
                    var_386 = wp::copy(var_387);
                    // rowadr_mom = wp.atomic_add(moment_nnz, worldid, rownnz_ten)                <L 2251>
                    var_388 = wp::atomic_add(var_moment_nnz, var_0, var_383);
                    // moment_rownnz_out[worldid, actid] = rownnz_ten                             <L 2252>
                    wp::array_store(var_moment_rownnz_out, var_0, var_1, var_383);
                    // moment_rowadr_out[worldid, actid] = rowadr_mom                             <L 2253>
                    wp::array_store(var_moment_rowadr_out, var_0, var_1, var_388);
                    // for k in range(rownnz_ten):                                                <L 2255>
                    var_389 = wp::range(var_383);
                    start_for_4:;
                        if (iter_cmp(var_389) == 0) goto end_for_4;
                        var_390 = wp::iter_next(var_389);
                        // sparseid_ten = rowadr_ten + k                                          <L 2256>
                        var_391 = wp::add(var_386, var_390);
                        // sparseid_mom = rowadr_mom + k                                          <L 2257>
                        var_392 = wp::add(var_388, var_390);
                        // moment_colind_out[worldid, sparseid_mom] = ten_J_colind[sparseid_ten]       <L 2258>
                        var_393 = wp::address(var_ten_J_colind, var_391);
                        var_394 = wp::load(var_393);
                        wp::array_store(var_moment_colind_out, var_0, var_392, var_394);
                        // actuator_moment_out[worldid, sparseid_mom] = ten_J_in[worldid, sparseid_ten] * gear0       <L 2259>
                        var_395 = wp::address(var_ten_J_in, var_0, var_391);
                        var_397 = wp::load(var_395);
                        var_396 = wp::mul(var_397, var_378);
                        wp::array_store(var_actuator_moment_out, var_0, var_392, var_396);
                        goto start_for_4;
                    end_for_4:;
                }
                var_398 = wp::where(var_372, var_378, var_213);
                if (!var_372) {
                    // elif trntype == TrnType.BODY:                                              <L 2260>
                    var_400 = (var_3 == var_399);
                    if (var_400) {
                        // actuator_length_out[worldid, actid] = 0.0                              <L 2262>
                        wp::array_store(var_actuator_length_out, var_0, var_1, var_401);
                        // rowadr = wp.atomic_add(moment_nnz, worldid, nv)                        <L 2265>
                        var_402 = wp::atomic_add(var_moment_nnz, var_0, var_nv);
                        // moment_rownnz_out[worldid, actid] = nv                                 <L 2266>
                        wp::array_store(var_moment_rownnz_out, var_0, var_1, var_nv);
                        // moment_rowadr_out[worldid, actid] = rowadr                             <L 2267>
                        wp::array_store(var_moment_rowadr_out, var_0, var_1, var_402);
                        // for i in range(nv):                                                    <L 2268>
                        var_403 = wp::range(var_nv);
                        start_for_6:;
                            if (iter_cmp(var_403) == 0) goto end_for_6;
                            var_404 = wp::iter_next(var_403);
                            // sparseid = rowadr + i                                              <L 2269>
                            var_405 = wp::add(var_402, var_404);
                            // moment_colind_out[worldid, sparseid] = i                           <L 2270>
                            wp::array_store(var_moment_colind_out, var_0, var_405, var_404);
                            // actuator_moment_out[worldid, sparseid] = 0.0                       <L 2271>
                            wp::array_store(var_actuator_moment_out, var_0, var_405, var_406);
                            wp::assign(var_175, var_405);
                            goto start_for_6;
                        end_for_6:;
                    }
                    var_407 = wp::where(var_400, var_402, var_370);
                    var_408 = wp::where(var_400, var_404, var_174);
                    if (!var_400) {
                        // elif trntype == TrnType.SITE:                                          <L 2274>
                        var_410 = (var_3 == var_409);
                        if (var_410) {
                            // trnid = actuator_trnid[actid]                                      <L 2275>
                            var_411 = wp::address(var_actuator_trnid, var_1);
                            var_413 = wp::load(var_411);
                            var_412 = wp::copy(var_413);
                            // siteid = trnid[0]                                                  <L 2276>
                            var_415 = wp::extract(var_412, var_414);
                            // refid = trnid[1]                                                   <L 2277>
                            var_417 = wp::extract(var_412, var_416);
                            // gear = actuator_gear[actuator_gear_id, actid]                      <L 2279>
                            var_418 = wp::address(var_actuator_gear, var_9, var_1);
                            var_420 = wp::load(var_418);
                            var_419 = wp::copy(var_420);
                            // site_quat_id = worldid % site_quat.shape[0]                        <L 2280>
                            var_421 = &(var_site_quat.shape);
                            var_424 = wp::load(var_421);
                            var_423 = wp::extract(var_424, var_422);
                            var_425 = wp::mod(var_0, var_423);
                            // gear_translation = wp.spatial_top(gear)                            <L 2281>
                            var_426 = wp::spatial_top(var_419);
                            // gear_rotational = wp.spatial_bottom(gear)                          <L 2282>
                            var_427 = wp::spatial_bottom(var_419);
                            // if refid == -1:                                                    <L 2285>
                            var_430 = (var_417 == var_429);
                            if (var_430) {
                                // site_xmat = site_xmat_in[worldid, siteid]                      <L 2287>
                                var_431 = wp::address(var_site_xmat_in, var_0, var_415);
                                var_433 = wp::load(var_431);
                                var_432 = wp::copy(var_433);
                                // wrench_translation = site_xmat @ gear_translation              <L 2288>
                                var_434 = wp::mul(var_432, var_426);
                                // wrench_rotation = site_xmat @ gear_rotational                  <L 2289>
                                var_435 = wp::mul(var_432, var_427);
                                // b1 = body_weldid[site_bodyid[siteid]]                          <L 2292>
                                var_436 = wp::address(var_site_bodyid, var_415);
                                var_438 = wp::load(var_436);
                                var_437 = wp::address(var_body_weldid, var_438);
                                var_440 = wp::load(var_437);
                                var_439 = wp::copy(var_440);
                                // da_init = int(-1)                                              <L 2293>
                                var_443 = wp::int(var_442);
                                // if b1 > 0:                                                     <L 2294>
                                var_445 = (var_439 > var_444);
                                if (var_445) {
                                    // da_init = body_dofadr[b1] + body_dofnum[b1] - 1            <L 2295>
                                    var_446 = wp::address(var_body_dofadr, var_439);
                                    var_447 = wp::address(var_body_dofnum, var_439);
                                    var_449 = wp::load(var_446);
                                    var_450 = wp::load(var_447);
                                    var_448 = wp::add(var_449, var_450);
                                    var_452 = wp::sub(var_448, var_451);
                                }
                                var_453 = wp::where(var_445, var_452, var_443);
                                // da = da_init                                                   <L 2297>
                                var_454 = wp::copy(var_453);
                                // ndof = int(0)                                                  <L 2298>
                                var_456 = wp::int(var_455);
                                // while da >= 0:                                                 <L 2299>
        start_while_8:;
                                var_458 = (var_454 >= var_457);
        if ((var_458) == false) goto end_while_8;
                                    // ndof += 1                                                  <L 2300>
                                    var_460 = wp::add(var_456, var_459);
                                    // da = dof_parentid[da]                                      <L 2301>
                                    var_461 = wp::address(var_dof_parentid, var_454);
                                    var_463 = wp::load(var_461);
                                    var_462 = wp::copy(var_463);
                                    wp::assign(var_456, var_460);
                                    wp::assign(var_454, var_462);
        goto start_while_8;
        end_while_8:;
                                // moment_rownnz_out[worldid, actid] = ndof                       <L 2303>
                                wp::array_store(var_moment_rownnz_out, var_0, var_1, var_456);
                                // rowadr = wp.atomic_add(moment_nnz, worldid, ndof)              <L 2304>
                                var_464 = wp::atomic_add(var_moment_nnz, var_0, var_456);
                                // moment_rowadr_out[worldid, actid] = rowadr                     <L 2305>
                                wp::array_store(var_moment_rowadr_out, var_0, var_1, var_464);
                                // actuator_length_out[worldid, actid] = 0.0                      <L 2306>
                                wp::array_store(var_actuator_length_out, var_0, var_1, var_465);
                                // da = da_init                                                   <L 2309>
                                var_466 = wp::copy(var_453);
                                // ptr = ndof - 1                                                 <L 2310>
                                var_468 = wp::sub(var_456, var_467);
                                // while da >= 0:                                                 <L 2311>
        start_while_10:;
                                var_470 = (var_466 >= var_469);
        if ((var_470) == false) goto end_while_10;
                                    // jacp, jacr = support.jac_dof(                              <L 2312>
                                    // body_parentid,                                             <L 2313>
                                    // body_rootid,                                               <L 2314>
                                    // dof_bodyid,                                                <L 2315>
                                    // subtree_com_in,                                            <L 2316>
                                    // cdof_in,                                                   <L 2317>
                                    // site_xpos_in[worldid, siteid],                             <L 2318>
                                    var_471 = wp::address(var_site_xpos_in, var_0, var_415);
                                    // site_bodyid[siteid],                                       <L 2319>
                                    var_472 = wp::address(var_site_bodyid, var_415);
                                    // da,                                                        <L 2320>
                                    // worldid,                                                   <L 2321>
                                    var_475 = wp::load(var_471);
                                    var_476 = wp::load(var_472);
                                    jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_subtree_com_in, var_cdof_in, var_475, var_476, var_466, var_0, var_473, var_474);
                                    // moment = wp.dot(jacp, wrench_translation) + wp.dot(jacr, wrench_rotation)       <L 2323>
                                    var_477 = wp::dot(var_473, var_434);
                                    var_478 = wp::dot(var_474, var_435);
                                    var_479 = wp::add(var_477, var_478);
                                    // sparseid = rowadr + ptr                                    <L 2324>
                                    var_480 = wp::add(var_464, var_468);
                                    // moment_colind_out[worldid, sparseid] = da                  <L 2325>
                                    wp::array_store(var_moment_colind_out, var_0, var_480, var_466);
                                    // actuator_moment_out[worldid, sparseid] = moment            <L 2326>
                                    wp::array_store(var_actuator_moment_out, var_0, var_480, var_479);
                                    // ptr -= 1                                                   <L 2327>
                                    var_482 = wp::sub(var_468, var_481);
                                    // da = dof_parentid[da]                                      <L 2328>
                                    var_483 = wp::address(var_dof_parentid, var_466);
                                    var_485 = wp::load(var_483);
                                    var_484 = wp::copy(var_485);
                                    wp::assign(var_175, var_480);
                                    wp::assign(var_466, var_484);
                                    wp::assign(var_468, var_482);
                                    wp::assign(var_343, var_473);
                                    wp::assign(var_344, var_474);
                                    wp::assign(var_355, var_479);
        goto start_while_10;
        end_while_10:;
                            }
                            var_486 = wp::where(var_430, var_464, var_407);
                            var_487 = wp::where(var_430, var_432, var_223);
                            var_488 = wp::where(var_430, var_439, var_276);
                            var_489 = wp::where(var_430, var_456, var_312);
                            var_490 = wp::where(var_430, var_466, var_318);
                            var_491 = wp::where(var_430, var_468, var_335);
                            if (!var_430) {
                                // bodyid = site_bodyid[siteid]                                   <L 2332>
                                var_492 = wp::address(var_site_bodyid, var_415);
                                var_494 = wp::load(var_492);
                                var_493 = wp::copy(var_494);
                                // bodyrefid = site_bodyid[refid]                                 <L 2333>
                                var_495 = wp::address(var_site_bodyid, var_417);
                                var_497 = wp::load(var_495);
                                var_496 = wp::copy(var_497);
                                // b0 = body_weldid[bodyid]                                       <L 2334>
                                var_498 = wp::address(var_body_weldid, var_493);
                                var_500 = wp::load(var_498);
                                var_499 = wp::copy(var_500);
                                // b1 = body_weldid[bodyrefid]                                    <L 2335>
                                var_501 = wp::address(var_body_weldid, var_496);
                                var_503 = wp::load(var_501);
                                var_502 = wp::copy(var_503);
                                // dofadr0 = body_dofadr[b0] + body_dofnum[b0] - 1                <L 2336>
                                var_504 = wp::address(var_body_dofadr, var_499);
                                var_505 = wp::address(var_body_dofnum, var_499);
                                var_507 = wp::load(var_504);
                                var_508 = wp::load(var_505);
                                var_506 = wp::add(var_507, var_508);
                                var_510 = wp::sub(var_506, var_509);
                                // dofadr1 = body_dofadr[b1] + body_dofnum[b1] - 1                <L 2337>
                                var_511 = wp::address(var_body_dofadr, var_502);
                                var_512 = wp::address(var_body_dofnum, var_502);
                                var_514 = wp::load(var_511);
                                var_515 = wp::load(var_512);
                                var_513 = wp::add(var_514, var_515);
                                var_517 = wp::sub(var_513, var_516);
                                // dofadr_common = -1                                             <L 2340>
                                // if dofadr0 >= 0 and dofadr1 >= 0:                              <L 2341>
                                var_521 = (var_510 >= var_520);
                                var_523 = (var_517 >= var_522);
                                var_524 = var_521 && var_523;
                                if (var_524) {
                                    // while dofadr0 != dofadr1:                                  <L 2343>
        start_while_12:;
                                    var_525 = (var_510 != var_517);
        if ((var_525) == false) goto end_while_12;
                                        // if dofadr0 < dofadr1:                                  <L 2344>
                                        var_526 = (var_510 < var_517);
                                        if (var_526) {
                                            // dofadr1 = dof_parentid[dofadr1]                    <L 2345>
                                            var_527 = wp::address(var_dof_parentid, var_517);
                                            var_529 = wp::load(var_527);
                                            var_528 = wp::copy(var_529);
                                        }
                                        var_530 = wp::where(var_526, var_528, var_517);
                                        if (!var_526) {
                                            // dofadr0 = dof_parentid[dofadr0]                    <L 2347>
                                            var_531 = wp::address(var_dof_parentid, var_510);
                                            var_533 = wp::load(var_531);
                                            var_532 = wp::copy(var_533);
                                        }
                                        var_534 = wp::where(var_526, var_510, var_532);
                                        // if dofadr0 == -1 or dofadr1 == -1:                     <L 2349>
                                        var_537 = (var_534 == var_536);
                                        var_540 = (var_530 == var_539);
                                        var_541 = var_537 || var_540;
                                        if (var_541) {
                                            // break                                              <L 2351>
                                            wp::assign(var_510, var_534);
                                            wp::assign(var_517, var_530);
                                            goto end_while_12;
                                        }
                                        var_542 = wp::where(var_541, var_510, var_534);
                                        var_543 = wp::where(var_541, var_517, var_530);
                                        wp::assign(var_510, var_542);
                                        wp::assign(var_517, var_543);
        goto start_while_12;
        end_while_12:;
                                    // if dofadr0 == dofadr1:                                     <L 2354>
                                    var_544 = (var_510 == var_517);
                                    if (var_544) {
                                        // dofadr_common = dofadr0                                <L 2355>
                                        var_545 = wp::copy(var_510);
                                    }
                                    var_546 = wp::where(var_544, var_545, var_519);
                                }
                                var_547 = wp::where(var_524, var_546, var_519);
                                // translational_transmission = not (gear[0] == 0.0 and gear[1] == 0.0 and gear[2] == 0.0)       <L 2357>
                                var_549 = wp::extract(var_419, var_548);
                                var_551 = (var_549 == var_550);
                                var_553 = wp::extract(var_419, var_552);
                                var_555 = (var_553 == var_554);
                                var_557 = wp::extract(var_419, var_556);
                                var_559 = (var_557 == var_558);
                                var_560 = var_551 && var_555 && var_559;
                                var_561 = wp::unot(var_560);
                                // rotational_transmission = not (gear[3] == 0.0 and gear[4] == 0.0 and gear[5] == 0.0)       <L 2358>
                                var_563 = wp::extract(var_419, var_562);
                                var_565 = (var_563 == var_564);
                                var_567 = wp::extract(var_419, var_566);
                                var_569 = (var_567 == var_568);
                                var_571 = wp::extract(var_419, var_570);
                                var_573 = (var_571 == var_572);
                                var_574 = var_565 && var_569 && var_573;
                                var_575 = wp::unot(var_574);
                                // site_xpos = site_xpos_in[worldid, siteid]                      <L 2360>
                                var_576 = wp::address(var_site_xpos_in, var_0, var_415);
                                var_578 = wp::load(var_576);
                                var_577 = wp::copy(var_578);
                                // ref_xpos = site_xpos_in[worldid, refid]                        <L 2361>
                                var_579 = wp::address(var_site_xpos_in, var_0, var_417);
                                var_581 = wp::load(var_579);
                                var_580 = wp::copy(var_581);
                                // ref_xmat = site_xmat_in[worldid, refid]                        <L 2362>
                                var_582 = wp::address(var_site_xmat_in, var_0, var_417);
                                var_584 = wp::load(var_582);
                                var_583 = wp::copy(var_584);
                                // length = float(0.0)                                            <L 2364>
                                var_586 = wp::float(var_585);
                                // if translational_transmission:                                 <L 2366>
                                if (var_561) {
                                    // vec = wp.transpose(ref_xmat) @ (site_xpos - ref_xpos)       <L 2368>
                                    var_587 = wp::transpose(var_583);
                                    var_588 = wp::sub(var_577, var_580);
                                    var_589 = wp::mul(var_587, var_588);
                                    // length += wp.dot(vec, gear_translation)                    <L 2369>
                                    var_590 = wp::dot(var_589, var_426);
                                    var_591 = wp::add(var_586, var_590);
                                    // wrench_translation = ref_xmat @ gear_translation           <L 2371>
                                    var_592 = wp::mul(var_583, var_426);
                                }
                                var_593 = wp::where(var_561, var_589, var_241);
                                var_594 = wp::where(var_561, var_591, var_586);
                                var_595 = wp::where(var_561, var_592, var_434);
                                // if rotational_transmission:                                    <L 2373>
                                if (var_575) {
                                    // quat = math.mul_quat(site_quat[site_quat_id, siteid], xquat_in[worldid, bodyid])       <L 2375>
                                    var_596 = wp::address(var_site_quat, var_425, var_415);
                                    var_597 = wp::address(var_xquat_in, var_0, var_493);
                                    var_599 = wp::load(var_596);
                                    var_600 = wp::load(var_597);
                                    var_598 = mul_quat_0(var_599, var_600);
                                    // refquat = math.mul_quat(site_quat[site_quat_id, refid], xquat_in[worldid, bodyrefid])       <L 2376>
                                    var_601 = wp::address(var_site_quat, var_425, var_417);
                                    var_602 = wp::address(var_xquat_in, var_0, var_496);
                                    var_604 = wp::load(var_601);
                                    var_605 = wp::load(var_602);
                                    var_603 = mul_quat_0(var_604, var_605);
                                    // vec = math.quat_sub(quat, refquat)                         <L 2379>
                                    var_606 = quat_sub_0(var_598, var_603);
                                    // length += wp.dot(vec, gear_rotational)                     <L 2380>
                                    var_607 = wp::dot(var_606, var_427);
                                    var_608 = wp::add(var_594, var_607);
                                    // wrench_rotation = ref_xmat @ gear_rotational               <L 2382>
                                    var_609 = wp::mul(var_583, var_427);
                                }
                                var_610 = wp::where(var_575, var_598, var_83);
                                var_611 = wp::where(var_575, var_606, var_593);
                                var_612 = wp::where(var_575, var_608, var_594);
                                var_613 = wp::where(var_575, var_609, var_435);
                                // actuator_length_out[worldid, actid] = length                   <L 2384>
                                wp::array_store(var_actuator_length_out, var_0, var_1, var_612);
                                // da1_init = int(-1)                                             <L 2387>
                                var_616 = wp::int(var_615);
                                // da2_init = int(-1)                                             <L 2388>
                                var_619 = wp::int(var_618);
                                // if b0 > 0:                                                     <L 2389>
                                var_621 = (var_499 > var_620);
                                if (var_621) {
                                    // da1_init = body_dofadr[b0] + body_dofnum[b0] - 1           <L 2390>
                                    var_622 = wp::address(var_body_dofadr, var_499);
                                    var_623 = wp::address(var_body_dofnum, var_499);
                                    var_625 = wp::load(var_622);
                                    var_626 = wp::load(var_623);
                                    var_624 = wp::add(var_625, var_626);
                                    var_628 = wp::sub(var_624, var_627);
                                }
                                var_629 = wp::where(var_621, var_628, var_616);
                                // if b1 > 0:                                                     <L 2391>
                                var_631 = (var_502 > var_630);
                                if (var_631) {
                                    // da2_init = body_dofadr[b1] + body_dofnum[b1] - 1           <L 2392>
                                    var_632 = wp::address(var_body_dofadr, var_502);
                                    var_633 = wp::address(var_body_dofnum, var_502);
                                    var_635 = wp::load(var_632);
                                    var_636 = wp::load(var_633);
                                    var_634 = wp::add(var_635, var_636);
                                    var_638 = wp::sub(var_634, var_637);
                                }
                                var_639 = wp::where(var_631, var_638, var_619);
                                // da1 = da1_init                                                 <L 2394>
                                var_640 = wp::copy(var_629);
                                // da2 = da2_init                                                 <L 2395>
                                var_641 = wp::copy(var_639);
                                // ndof = int(0)                                                  <L 2396>
                                var_643 = wp::int(var_642);
                                // while da1 >= 0 or da2 >= 0:                                    <L 2397>
        start_while_14:;
                                var_645 = (var_640 >= var_644);
                                var_647 = (var_641 >= var_646);
                                var_648 = var_645 || var_647;
        if ((var_648) == false) goto end_while_14;
                                    // da = wp.max(da1, da2)                                      <L 2398>
                                    var_649 = wp::max(var_640, var_641);
                                    // if da1 == da and da2 == da:                                <L 2399>
                                    var_650 = (var_640 == var_649);
                                    var_651 = (var_641 == var_649);
                                    var_652 = var_650 && var_651;
                                    if (var_652) {
                                        // break                                                  <L 2400>
                                        wp::assign(var_490, var_649);
                                        goto end_while_14;
                                    }
                                    var_653 = wp::where(var_652, var_490, var_649);
                                    // ndof += 1                                                  <L 2401>
                                    var_655 = wp::add(var_643, var_654);
                                    // if da1 == da:                                              <L 2402>
                                    var_656 = (var_640 == var_653);
                                    if (var_656) {
                                        // da1 = dof_parentid[da1]                                <L 2403>
                                        var_657 = wp::address(var_dof_parentid, var_640);
                                        var_659 = wp::load(var_657);
                                        var_658 = wp::copy(var_659);
                                    }
                                    var_660 = wp::where(var_656, var_658, var_640);
                                    // if da2 == da:                                              <L 2404>
                                    var_661 = (var_641 == var_653);
                                    if (var_661) {
                                        // da2 = dof_parentid[da2]                                <L 2405>
                                        var_662 = wp::address(var_dof_parentid, var_641);
                                        var_664 = wp::load(var_662);
                                        var_663 = wp::copy(var_664);
                                    }
                                    var_665 = wp::where(var_661, var_663, var_641);
                                    wp::assign(var_640, var_660);
                                    wp::assign(var_641, var_665);
                                    wp::assign(var_643, var_655);
                                    wp::assign(var_490, var_653);
        goto start_while_14;
        end_while_14:;
                                // moment_rownnz_out[worldid, actid] = ndof                       <L 2407>
                                wp::array_store(var_moment_rownnz_out, var_0, var_1, var_643);
                                // rowadr = wp.atomic_add(moment_nnz, worldid, ndof)              <L 2408>
                                var_666 = wp::atomic_add(var_moment_nnz, var_0, var_643);
                                // moment_rowadr_out[worldid, actid] = rowadr                     <L 2409>
                                wp::array_store(var_moment_rowadr_out, var_0, var_1, var_666);
                                // da1 = da1_init                                                 <L 2412>
                                var_667 = wp::copy(var_629);
                                // da2 = da2_init                                                 <L 2413>
                                var_668 = wp::copy(var_639);
                                // ptr = ndof - 1                                                 <L 2415>
                                var_670 = wp::sub(var_643, var_669);
                                // while da1 >= 0 or da2 >= 0:                                    <L 2416>
        start_while_16:;
                                var_672 = (var_667 >= var_671);
                                var_674 = (var_668 >= var_673);
                                var_675 = var_672 || var_674;
        if ((var_675) == false) goto end_while_16;
                                    // da = wp.max(da1, da2)                                      <L 2417>
                                    var_676 = wp::max(var_667, var_668);
                                    // if da1 == da and da2 == da:                                <L 2418>
                                    var_677 = (var_667 == var_676);
                                    var_678 = (var_668 == var_676);
                                    var_679 = var_677 && var_678;
                                    if (var_679) {
                                        // break                                                  <L 2419>
                                        wp::assign(var_490, var_676);
                                        goto end_while_16;
                                    }
                                    var_680 = wp::where(var_679, var_490, var_676);
                                    // jacp, jacr = support.jac_dof(                              <L 2421>
                                    // body_parentid, body_rootid, dof_bodyid, subtree_com_in, cdof_in, site_xpos, site_bodyid[siteid], da, worldid       <L 2422>
                                    var_681 = wp::address(var_site_bodyid, var_415);
                                    var_684 = wp::load(var_681);
                                    jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_subtree_com_in, var_cdof_in, var_577, var_684, var_680, var_0, var_682, var_683);
                                    // jacpref, jacrref = support.jac_dof(                        <L 2424>
                                    // body_parentid, body_rootid, dof_bodyid, subtree_com_in, cdof_in, ref_xpos, site_bodyid[refid], da, worldid       <L 2425>
                                    var_685 = wp::address(var_site_bodyid, var_417);
                                    var_688 = wp::load(var_685);
                                    jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_subtree_com_in, var_cdof_in, var_580, var_688, var_680, var_0, var_686, var_687);
                                    // moment = float(0.0)                                        <L 2428>
                                    var_690 = wp::float(var_689);
                                    // if translational_transmission:                             <L 2429>
                                    if (var_561) {
                                        // moment += wp.dot(jacp - jacpref, wrench_translation)       <L 2430>
                                        var_691 = wp::sub(var_682, var_686);
                                        var_692 = wp::dot(var_691, var_595);
                                        var_693 = wp::add(var_690, var_692);
                                    }
                                    var_694 = wp::where(var_561, var_693, var_690);
                                    // if rotational_transmission:                                <L 2431>
                                    if (var_575) {
                                        // moment += wp.dot(jacr - jacrref, wrench_rotation)       <L 2432>
                                        var_695 = wp::sub(var_683, var_687);
                                        var_696 = wp::dot(var_695, var_613);
                                        var_697 = wp::add(var_694, var_696);
                                    }
                                    var_698 = wp::where(var_575, var_697, var_694);
                                    // sparseid = rowadr + ptr                                    <L 2434>
                                    var_699 = wp::add(var_666, var_670);
                                    // moment_colind_out[worldid, sparseid] = da                  <L 2435>
                                    wp::array_store(var_moment_colind_out, var_0, var_699, var_680);
                                    // actuator_moment_out[worldid, sparseid] = moment            <L 2436>
                                    wp::array_store(var_actuator_moment_out, var_0, var_699, var_698);
                                    // ptr -= 1                                                   <L 2437>
                                    var_701 = wp::sub(var_670, var_700);
                                    // if da1 == da:                                              <L 2439>
                                    var_702 = (var_667 == var_680);
                                    if (var_702) {
                                        // da1 = dof_parentid[da1]                                <L 2440>
                                        var_703 = wp::address(var_dof_parentid, var_667);
                                        var_705 = wp::load(var_703);
                                        var_704 = wp::copy(var_705);
                                    }
                                    var_706 = wp::where(var_702, var_704, var_667);
                                    // if da2 == da:                                              <L 2441>
                                    var_707 = (var_668 == var_680);
                                    if (var_707) {
                                        // da2 = dof_parentid[da2]                                <L 2442>
                                        var_708 = wp::address(var_dof_parentid, var_668);
                                        var_710 = wp::load(var_708);
                                        var_709 = wp::copy(var_710);
                                    }
                                    var_711 = wp::where(var_707, var_709, var_668);
                                    wp::assign(var_175, var_699);
                                    wp::assign(var_667, var_706);
                                    wp::assign(var_668, var_711);
                                    wp::assign(var_490, var_680);
                                    wp::assign(var_670, var_701);
                                    wp::assign(var_343, var_682);
                                    wp::assign(var_344, var_683);
                                    wp::assign(var_355, var_698);
        goto start_while_16;
        end_while_16:;
                            }
                            var_712 = wp::where(var_430, var_486, var_666);
                            var_713 = wp::where(var_430, var_83, var_610);
                            var_714 = wp::where(var_430, var_241, var_611);
                            var_715 = wp::where(var_430, var_258, var_612);
                            var_716 = wp::where(var_430, var_488, var_502);
                            var_717 = wp::where(var_430, var_298, var_629);
                            var_718 = wp::where(var_430, var_308, var_639);
                            var_719 = wp::where(var_430, var_332, var_667);
                            var_720 = wp::where(var_430, var_333, var_668);
                            var_721 = wp::where(var_430, var_489, var_643);
                            var_722 = wp::where(var_430, var_491, var_670);
                            var_723 = wp::where(var_430, var_434, var_595);
                            var_724 = wp::where(var_430, var_435, var_613);
                        }
                        var_725 = wp::where(var_410, var_419, var_11);
                        var_726 = wp::where(var_410, var_712, var_407);
                        var_727 = wp::where(var_410, var_713, var_83);
                        var_728 = wp::where(var_410, var_412, var_206);
                        var_729 = wp::where(var_410, var_487, var_223);
                        var_730 = wp::where(var_410, var_714, var_241);
                        var_731 = wp::where(var_410, var_715, var_258);
                        var_732 = wp::where(var_410, var_716, var_276);
                        var_733 = wp::where(var_410, var_717, var_298);
                        var_734 = wp::where(var_410, var_718, var_308);
                        var_735 = wp::where(var_410, var_719, var_332);
                        var_736 = wp::where(var_410, var_720, var_333);
                        var_737 = wp::where(var_410, var_721, var_312);
                        var_738 = wp::where(var_410, var_490, var_318);
                        var_739 = wp::where(var_410, var_722, var_335);
                        if (!var_410) {
                            // wp.printf("unhandled transmission type %d\n", trntype)             <L 2444>
                            printf(var_740, var_3);
                        }
                    }
                    var_741 = wp::where(var_400, var_11, var_725);
                    var_742 = wp::where(var_400, var_407, var_726);
                    var_743 = wp::where(var_400, var_83, var_727);
                    var_744 = wp::where(var_400, var_206, var_728);
                    var_745 = wp::where(var_400, var_223, var_729);
                    var_746 = wp::where(var_400, var_241, var_730);
                    var_747 = wp::where(var_400, var_258, var_731);
                    var_748 = wp::where(var_400, var_276, var_732);
                    var_749 = wp::where(var_400, var_298, var_733);
                    var_750 = wp::where(var_400, var_308, var_734);
                    var_751 = wp::where(var_400, var_332, var_735);
                    var_752 = wp::where(var_400, var_333, var_736);
                    var_753 = wp::where(var_400, var_312, var_737);
                    var_754 = wp::where(var_400, var_318, var_738);
                    var_755 = wp::where(var_400, var_335, var_739);
                }
                var_756 = wp::where(var_372, var_11, var_741);
                var_757 = wp::where(var_372, var_370, var_742);
                var_758 = wp::where(var_372, var_83, var_743);
                var_759 = wp::where(var_372, var_174, var_408);
                var_760 = wp::where(var_372, var_206, var_744);
                var_761 = wp::where(var_372, var_223, var_745);
                var_762 = wp::where(var_372, var_241, var_746);
                var_763 = wp::where(var_372, var_258, var_747);
                var_764 = wp::where(var_372, var_276, var_748);
                var_765 = wp::where(var_372, var_298, var_749);
                var_766 = wp::where(var_372, var_308, var_750);
                var_767 = wp::where(var_372, var_332, var_751);
                var_768 = wp::where(var_372, var_333, var_752);
                var_769 = wp::where(var_372, var_312, var_753);
                var_770 = wp::where(var_372, var_318, var_754);
                var_771 = wp::where(var_372, var_335, var_755);
            }
            var_772 = wp::where(var_204, var_11, var_756);
            var_773 = wp::where(var_204, var_370, var_757);
            var_774 = wp::where(var_204, var_83, var_758);
            var_775 = wp::where(var_204, var_174, var_759);
            var_776 = wp::where(var_204, var_206, var_760);
            var_777 = wp::where(var_204, var_213, var_398);
            var_778 = wp::where(var_204, var_223, var_761);
            var_779 = wp::where(var_204, var_241, var_762);
            var_780 = wp::where(var_204, var_258, var_763);
            var_781 = wp::where(var_204, var_276, var_764);
            var_782 = wp::where(var_204, var_298, var_765);
            var_783 = wp::where(var_204, var_308, var_766);
            var_784 = wp::where(var_204, var_332, var_767);
            var_785 = wp::where(var_204, var_333, var_768);
            var_786 = wp::where(var_204, var_312, var_769);
            var_787 = wp::where(var_204, var_318, var_770);
            var_788 = wp::where(var_204, var_335, var_771);
        }
        var_789 = wp::where(var_17, var_11, var_772);
        var_790 = wp::where(var_17, var_200, var_773);
        var_791 = wp::where(var_17, var_83, var_774);
        var_792 = wp::where(var_17, var_174, var_775);
    }
}



extern "C" __global__ void _geom_local_to_global_e28b714c_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_weldid,
    wp::array_t<wp::int32> var_body_mocapid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_pos,
    wp::array_t<wp::quat_t<wp::float32>> var_geom_quat,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_out)
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
        const wp::int32 var_6 = 0;
        bool var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        const wp::int32 var_12 = 1;
        const wp::int32 var_13 = -1;
        bool var_14;
        wp::int32 var_15;
        bool var_16;
        wp::vec_t<3, wp::float32>* var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::quat_t<wp::float32>* var_20;
        wp::quat_t<wp::float32> var_21;
        wp::quat_t<wp::float32> var_22;
        wp::shape_t* var_23;
        const wp::int32 var_24 = 0;
        wp::int32 var_25;
        wp::shape_t var_26;
        wp::int32 var_27;
        wp::vec_t<3, wp::float32>* var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::shape_t* var_32;
        const wp::int32 var_33 = 0;
        wp::int32 var_34;
        wp::shape_t var_35;
        wp::int32 var_36;
        wp::quat_t<wp::float32>* var_37;
        wp::quat_t<wp::float32> var_38;
        wp::quat_t<wp::float32> var_39;
        wp::mat_t<3, 3, wp::float32> var_40;
        //---------
        // forward
        // def _geom_local_to_global(                                                             <L 177>
        // worldid, geomid = wp.tid()                                                             <L 192>
        builtin_tid2d(var_0, var_1);
        // bodyid = geom_bodyid[geomid]                                                           <L 193>
        var_2 = wp::address(var_geom_bodyid, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if body_weldid[bodyid] == 0 and body_mocapid[body_rootid[bodyid]] == -1:               <L 195>
        var_5 = wp::address(var_body_weldid, var_3);
        var_8 = wp::load(var_5);
        var_7 = (var_8 == var_6);
        var_9 = wp::address(var_body_rootid, var_3);
        var_11 = wp::load(var_9);
        var_10 = wp::address(var_body_mocapid, var_11);
        var_15 = wp::load(var_10);
        var_14 = (var_15 == var_13);
        var_16 = var_7 && var_14;
        if (var_16) {
            // return                                                                             <L 198>
            continue;
        }
        // xpos = xpos_in[worldid, bodyid]                                                        <L 200>
        var_17 = wp::address(var_xpos_in, var_0, var_3);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // xquat = xquat_in[worldid, bodyid]                                                      <L 201>
        var_20 = wp::address(var_xquat_in, var_0, var_3);
        var_22 = wp::load(var_20);
        var_21 = wp::copy(var_22);
        // geom_xpos_out[worldid, geomid] = xpos + math.rot_vec_quat(geom_pos[worldid % geom_pos.shape[0], geomid], xquat)       <L 202>
        var_23 = &(var_geom_pos.shape);
        var_26 = wp::load(var_23);
        var_25 = wp::extract(var_26, var_24);
        var_27 = wp::mod(var_0, var_25);
        var_28 = wp::address(var_geom_pos, var_27, var_1);
        var_30 = wp::load(var_28);
        var_29 = rot_vec_quat_0(var_30, var_21);
        var_31 = wp::add(var_18, var_29);
        wp::array_store(var_geom_xpos_out, var_0, var_1, var_31);
        // geom_xmat_out[worldid, geomid] = math.quat_to_mat(math.mul_quat(xquat, geom_quat[worldid % geom_quat.shape[0], geomid]))       <L 203>
        var_32 = &(var_geom_quat.shape);
        var_35 = wp::load(var_32);
        var_34 = wp::extract(var_35, var_33);
        var_36 = wp::mod(var_0, var_34);
        var_37 = wp::address(var_geom_quat, var_36, var_1);
        var_39 = wp::load(var_37);
        var_38 = mul_quat_0(var_21, var_39);
        var_40 = quat_to_mat_0(var_38);
        wp::array_store(var_geom_xmat_out, var_0, var_1, var_40);
    }
}



extern "C" __global__ void _flex_vertices_a6b6fb24_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_nflex,
    wp::array_t<wp::int32> var_flex_vertadr,
    wp::array_t<wp::int32> var_flex_vertnum,
    wp::array_t<wp::int32> var_flex_vertbodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flex_vert,
    wp::array_t<bool> var_flex_centered,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexvert_xpos_out)
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
        wp::range_t var_2;
        wp::int32 var_3;
        wp::int32* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        wp::int32* var_9;
        bool var_10;
        wp::int32 var_11;
        bool var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        bool* var_19;
        bool var_20;
        bool var_21;
        bool var_22;
        wp::mat_t<3, 3, wp::float32>* var_23;
        wp::mat_t<3, 3, wp::float32> var_24;
        wp::mat_t<3, 3, wp::float32> var_25;
        wp::vec_t<3, wp::float32>* var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::vec_t<3, wp::float32> var_30;
        bool var_31;
        //---------
        // forward
        // def _flex_vertices(                                                                    <L 228>
        // worldid, vertid = wp.tid()                                                             <L 242>
        builtin_tid2d(var_0, var_1);
        // for f in range(nflex):                                                                 <L 244>
        var_2 = wp::range(var_nflex);
        start_for_0:;
            if (iter_cmp(var_2) == 0) goto end_for_0;
            var_3 = wp::iter_next(var_2);
            // locid = vertid - flex_vertadr[f]                                                   <L 245>
            var_4 = wp::address(var_flex_vertadr, var_3);
            var_6 = wp::load(var_4);
            var_5 = wp::sub(var_1, var_6);
            // if locid >= 0 and locid < flex_vertnum[f]:                                         <L 246>
            var_8 = (var_5 >= var_7);
            var_9 = wp::address(var_flex_vertnum, var_3);
            var_11 = wp::load(var_9);
            var_10 = (var_5 < var_11);
            var_12 = var_8 && var_10;
            if (var_12) {
                // break                                                                          <L 247>
                goto end_for_0;
            }
            goto start_for_0;
        end_for_0:;
        // bodyid = flex_vertbodyid[vertid]                                                       <L 249>
        var_13 = wp::address(var_flex_vertbodyid, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // xpos = xpos_in[worldid, bodyid]                                                        <L 250>
        var_16 = wp::address(var_xpos_in, var_0, var_14);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // if flex_centered[f]:                                                                   <L 252>
        var_19 = wp::address(var_flex_centered, var_3);
        var_20 = wp::load(var_19);
        if (var_20) {
            // flexvert_xpos_out[worldid, vertid] = xpos                                          <L 253>
            wp::array_store(var_flexvert_xpos_out, var_0, var_1, var_17);
        }
        var_21 = wp::load(var_19);
        var_22 = wp::load(var_19);
        if (!var_22) {
            // xmat = xmat_in[worldid, bodyid]                                                    <L 255>
            var_23 = wp::address(var_xmat_in, var_0, var_14);
            var_25 = wp::load(var_23);
            var_24 = wp::copy(var_25);
            // local_pos = flex_vert[vertid]                                                      <L 256>
            var_26 = wp::address(var_flex_vert, var_1);
            var_28 = wp::load(var_26);
            var_27 = wp::copy(var_28);
            // flexvert_xpos_out[worldid, vertid] = xmat @ local_pos + xpos                       <L 257>
            var_29 = wp::mul(var_24, var_27);
            var_30 = wp::add(var_29, var_17);
            wp::array_store(var_flexvert_xpos_out, var_0, var_1, var_30);
        }
        var_31 = wp::load(var_19);
    }
}



extern "C" __global__ void _transmission_body_moment_4a4bc8fc_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_opt_cone,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::vec_t<2, wp::int32>> var_actuator_trnid,
    wp::array_t<wp::int32> var_actuator_trntype_body_adr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::int32> var_moment_rowadr_in,
    wp::array_t<wp::float32> var_contact_dist_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_in,
    wp::array_t<wp::float32> var_contact_includemargin_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::int32> var_contact_worldid_in,
    wp::array_t<wp::int32> var_efc_J_rownnz_in,
    wp::array_t<wp::int32> var_efc_J_rowadr_in,
    wp::array_t<wp::int32> var_efc_J_colind_in,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::int32> var_nacon_in,
    bool var_efc_is_sparse,
    wp::array_t<wp::float32> var_actuator_moment_out,
    wp::array_t<wp::int32> var_actuator_trntype_body_ncon_out)
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
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::vec_t<2, wp::int32>* var_6;
        const wp::int32 var_7 = 0;
        wp::int32 var_8;
        wp::vec_t<2, wp::int32> var_9;
        const wp::int32 var_10 = 0;
        wp::int32* var_11;
        bool var_12;
        wp::int32 var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::vec_t<2, wp::int32>* var_17;
        wp::vec_t<2, wp::int32> var_18;
        wp::vec_t<2, wp::int32> var_19;
        const wp::int32 var_20 = 0;
        wp::int32 var_21;
        const wp::int32 var_22 = 1;
        wp::int32 var_23;
        const wp::int32 var_24 = 0;
        bool var_25;
        const wp::int32 var_26 = 0;
        bool var_27;
        bool var_28;
        wp::int32* var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        wp::int32* var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        bool var_35;
        bool var_36;
        bool var_37;
        wp::float32* var_38;
        wp::float32* var_39;
        bool var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::int32 var_43;
        const wp::int32 var_44 = 0;
        bool var_45;
        wp::slice_t var_46;
        const wp::int32 var_47 = 0;
        wp::array_t<wp::int32> var_48;
        const wp::int32 var_49 = 1;
        wp::int32 var_50;
        wp::int32* var_51;
        wp::int32 var_52;
        wp::int32 var_53;
        const wp::int32 var_54 = 0;
        bool var_55;
        wp::int32* var_56;
        wp::int32 var_57;
        wp::int32 var_58;
        wp::slice_t var_59;
        const wp::int32 var_60 = 0;
        wp::array_t<wp::int32> var_61;
        const wp::int32 var_62 = 1;
        bool var_63;
        const wp::int32 var_64 = 1;
        bool var_65;
        bool var_66;
        const wp::int32 var_67 = 0;
        wp::int32* var_68;
        wp::int32 var_69;
        wp::int32 var_70;
        wp::int32* var_71;
        wp::int32 var_72;
        wp::int32 var_73;
        bool var_74;
        wp::int32* var_75;
        wp::int32 var_76;
        wp::int32 var_77;
        wp::int32 var_78;
        const wp::int32 var_79 = 0;
        wp::int32* var_80;
        wp::int32 var_81;
        wp::int32 var_82;
        wp::slice_t var_83;
        const wp::int32 var_84 = 0;
        wp::array_t<wp::float32> var_85;
        wp::int32 var_86;
        const wp::int32 var_87 = 0;
        wp::float32* var_88;
        wp::float32 var_89;
        wp::float32 var_90;
        wp::int32 var_91;
        wp::slice_t var_92;
        const wp::int32 var_93 = 0;
        wp::array_t<wp::float32> var_94;
        wp::int32 var_95;
        wp::float32* var_96;
        wp::float32 var_97;
        wp::float32 var_98;
        wp::int32 var_99;
        const wp::int32 var_100 = 1;
        wp::int32 var_101;
        const wp::float32 var_102 = 0.5;
        wp::float32 var_103;
        wp::float32 var_104;
        const wp::int32 var_105 = 2;
        wp::int32 var_106;
        wp::range_t var_107;
        wp::int32 var_108;
        wp::int32* var_109;
        wp::int32 var_110;
        wp::int32 var_111;
        wp::int32* var_112;
        wp::int32 var_113;
        wp::int32 var_114;
        bool var_115;
        wp::int32* var_116;
        wp::int32 var_117;
        wp::int32 var_118;
        wp::int32 var_119;
        const wp::int32 var_120 = 0;
        wp::int32* var_121;
        wp::int32 var_122;
        wp::int32 var_123;
        wp::slice_t var_124;
        const wp::int32 var_125 = 0;
        wp::array_t<wp::float32> var_126;
        wp::int32 var_127;
        const wp::int32 var_128 = 0;
        wp::float32* var_129;
        wp::float32 var_130;
        wp::float32 var_131;
        wp::float32 var_132;
        wp::int32 var_133;
        wp::int32 var_134;
        wp::int32 var_135;
        wp::int32 var_136;
        wp::int32 var_137;
        wp::int32 var_138;
        wp::int32 var_139;
        wp::int32 var_140;
        wp::slice_t var_141;
        const wp::int32 var_142 = 0;
        wp::array_t<wp::float32> var_143;
        wp::int32 var_144;
        wp::float32* var_145;
        wp::float32 var_146;
        wp::float32 var_147;
        wp::float32 var_148;
        wp::int32 var_149;
        const wp::int32 var_150 = 1;
        bool var_151;
        wp::vec_t<3, wp::float32>* var_152;
        wp::vec_t<3, wp::float32> var_153;
        wp::vec_t<3, wp::float32> var_154;
        wp::mat_t<3, 3, wp::float32>* var_155;
        wp::mat_t<3, 3, wp::float32> var_156;
        wp::mat_t<3, 3, wp::float32> var_157;
        const wp::int32 var_158 = 0;
        const wp::int32 var_159 = 0;
        wp::float32 var_160;
        const wp::int32 var_161 = 0;
        const wp::int32 var_162 = 1;
        wp::float32 var_163;
        const wp::int32 var_164 = 0;
        const wp::int32 var_165 = 2;
        wp::float32 var_166;
        wp::vec_t<3, wp::float32> var_167;
        const wp::int32 var_168 = 0;
        wp::int32* var_169;
        wp::int32 var_170;
        wp::int32 var_171;
        const wp::int32 var_172 = 0;
        bool var_173;
        bool var_174;
        wp::int32* var_175;
        bool var_176;
        wp::int32 var_177;
        wp::int32* var_178;
        wp::int32 var_179;
        wp::int32 var_180;
        const wp::int32 var_181 = 0;
        wp::int32* var_182;
        wp::int32 var_183;
        wp::int32 var_184;
        wp::int32 var_185;
        wp::int32 var_186;
        wp::int32 var_187;
        wp::vec_t<3, wp::float32> var_188;
        wp::vec_t<3, wp::float32> var_189;
        wp::vec_t<3, wp::float32> var_190;
        wp::vec_t<3, wp::float32> var_191;
        wp::vec_t<3, wp::float32> var_192;
        wp::slice_t var_193;
        const wp::int32 var_194 = 0;
        wp::array_t<wp::float32> var_195;
        wp::int32 var_196;
        wp::float32 var_197;
        wp::float32 var_198;
        wp::int32 var_199;
        wp::int32 var_200;
        wp::int32 var_201;
        wp::int32 var_202;
        //---------
        // forward
        // def _transmission_body_moment(                                                         <L 2448>
        // trnbodyid, conid, dofid = wp.tid()                                                     <L 2481>
        builtin_tid3d(var_0, var_1, var_2);
        // actid = actuator_trntype_body_adr[trnbodyid]                                           <L 2482>
        var_3 = wp::address(var_actuator_trntype_body_adr, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // bodyid = actuator_trnid[actid][0]                                                      <L 2483>
        var_6 = wp::address(var_actuator_trnid, var_4);
        var_9 = wp::load(var_6);
        var_8 = wp::extract(var_9, var_7);
        // if conid >= nacon_in[0]:                                                               <L 2485>
        var_11 = wp::address(var_nacon_in, var_10);
        var_13 = wp::load(var_11);
        var_12 = (var_1 >= var_13);
        if (var_12) {
            // return                                                                             <L 2486>
            continue;
        }
        // worldid = contact_worldid_in[conid]                                                    <L 2488>
        var_14 = wp::address(var_contact_worldid_in, var_1);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // geom = contact_geom_in[conid]                                                          <L 2491>
        var_17 = wp::address(var_contact_geom_in, var_1);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // g1 = geom[0]                                                                           <L 2492>
        var_21 = wp::extract(var_18, var_20);
        // g2 = geom[1]                                                                           <L 2493>
        var_23 = wp::extract(var_18, var_22);
        // if g1 < 0 or g2 < 0:                                                                   <L 2496>
        var_25 = (var_21 < var_24);
        var_27 = (var_23 < var_26);
        var_28 = var_25 || var_27;
        if (var_28) {
            // return                                                                             <L 2497>
            continue;
        }
        // b1 = geom_bodyid[g1]                                                                   <L 2500>
        var_29 = wp::address(var_geom_bodyid, var_21);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // b2 = geom_bodyid[g2]                                                                   <L 2501>
        var_32 = wp::address(var_geom_bodyid, var_23);
        var_34 = wp::load(var_32);
        var_33 = wp::copy(var_34);
        // if b1 != bodyid and b2 != bodyid:                                                      <L 2504>
        var_35 = (var_30 != var_8);
        var_36 = (var_33 != var_8);
        var_37 = var_35 && var_36;
        if (var_37) {
            // return                                                                             <L 2505>
            continue;
        }
        // contact_exclude = int(contact_dist_in[conid] >= contact_includemargin_in[conid])       <L 2507>
        var_38 = wp::address(var_contact_dist_in, var_1);
        var_39 = wp::address(var_contact_includemargin_in, var_1);
        var_41 = wp::load(var_38);
        var_42 = wp::load(var_39);
        var_40 = (var_41 >= var_42);
        var_43 = wp::int(var_40);
        // if dofid == 0:                                                                         <L 2509>
        var_45 = (var_2 == var_44);
        if (var_45) {
            // wp.atomic_add(actuator_trntype_body_ncon_out[worldid], trnbodyid, 1)               <L 2510>
            var_46 = wp::slice_t(var_15, var_15, var_47);
            var_48 = wp::view(var_actuator_trntype_body_ncon_out, var_46);
            var_50 = wp::atomic_add(var_48, var_0, var_49);
        }
        // rowadr = moment_rowadr_in[worldid, actid]                                              <L 2512>
        var_51 = wp::address(var_moment_rowadr_in, var_15, var_4);
        var_53 = wp::load(var_51);
        var_52 = wp::copy(var_53);
        // if contact_exclude == 0:                                                               <L 2515>
        var_55 = (var_43 == var_54);
        if (var_55) {
            // contact_dim = contact_dim_in[conid]                                                <L 2516>
            var_56 = wp::address(var_contact_dim_in, var_1);
            var_58 = wp::load(var_56);
            var_57 = wp::copy(var_58);
            // contact_efc_address = contact_efc_address_in[conid]                                <L 2517>
            var_59 = wp::slice_t(var_1, var_1, var_60);
            var_61 = wp::view(var_contact_efc_address_in, var_59);
            // if contact_dim == 1 or opt_cone == ConeType.ELLIPTIC:                              <L 2519>
            var_63 = (var_57 == var_62);
            var_65 = (var_opt_cone == var_64);
            var_66 = var_63 || var_65;
            if (var_66) {
                // efcid0 = contact_efc_address[0]                                                <L 2520>
                var_68 = wp::address(var_61, var_67);
                var_70 = wp::load(var_68);
                var_69 = wp::copy(var_70);
                // if efc_is_sparse:                                                              <L 2521>
                if (var_efc_is_sparse) {
                    // rownnz = efc_J_rownnz_in[worldid, efcid0]                                  <L 2522>
                    var_71 = wp::address(var_efc_J_rownnz_in, var_15, var_69);
                    var_73 = wp::load(var_71);
                    var_72 = wp::copy(var_73);
                    // if dofid < rownnz:                                                         <L 2523>
                    var_74 = (var_2 < var_72);
                    if (var_74) {
                        // efc_rowadr = efc_J_rowadr_in[worldid, efcid0]                          <L 2524>
                        var_75 = wp::address(var_efc_J_rowadr_in, var_15, var_69);
                        var_77 = wp::load(var_75);
                        var_76 = wp::copy(var_77);
                        // efc_sparseid = efc_rowadr + dofid                                      <L 2525>
                        var_78 = wp::add(var_76, var_2);
                        // colind = efc_J_colind_in[worldid, 0, efc_sparseid]                     <L 2526>
                        var_80 = wp::address(var_efc_J_colind_in, var_15, var_79, var_78);
                        var_82 = wp::load(var_80);
                        var_81 = wp::copy(var_82);
                        // wp.atomic_add(actuator_moment_out[worldid], rowadr + colind, efc_J_in[worldid, 0, efc_sparseid])       <L 2527>
                        var_83 = wp::slice_t(var_15, var_15, var_84);
                        var_85 = wp::view(var_actuator_moment_out, var_83);
                        var_86 = wp::add(var_52, var_81);
                        var_88 = wp::address(var_efc_J_in, var_15, var_87, var_78);
                        var_90 = wp::load(var_88);
                        var_89 = wp::atomic_add(var_85, var_86, var_90);
                    }
                    if (!var_74) {
                        // return                                                                 <L 2529>
                        continue;
                    }
                }
                if (!var_efc_is_sparse) {
                    // colind = dofid                                                             <L 2531>
                    var_91 = wp::copy(var_2);
                    // wp.atomic_add(actuator_moment_out[worldid], rowadr + colind, efc_J_in[worldid, efcid0, dofid])       <L 2532>
                    var_92 = wp::slice_t(var_15, var_15, var_93);
                    var_94 = wp::view(var_actuator_moment_out, var_92);
                    var_95 = wp::add(var_52, var_91);
                    var_96 = wp::address(var_efc_J_in, var_15, var_69, var_2);
                    var_98 = wp::load(var_96);
                    var_97 = wp::atomic_add(var_94, var_95, var_98);
                }
                var_99 = wp::where(var_efc_is_sparse, var_81, var_91);
            }
            if (!var_66) {
                // npyramid = contact_dim - 1  # number of frictional directions                  <L 2534>
                var_101 = wp::sub(var_57, var_100);
                // efc_force = 0.5 / float(npyramid)                                              <L 2535>
                var_103 = wp::float(var_101);
                var_104 = wp::div(var_102, var_103);
                // for j in range(2 * npyramid):                                                  <L 2537>
                var_106 = wp::mul(var_105, var_101);
                var_107 = wp::range(var_106);
                start_for_4:;
                    if (iter_cmp(var_107) == 0) goto end_for_4;
                    var_108 = wp::iter_next(var_107);
                    // efcid = contact_efc_address[j]                                             <L 2538>
                    var_109 = wp::address(var_61, var_108);
                    var_111 = wp::load(var_109);
                    var_110 = wp::copy(var_111);
                    // if efc_is_sparse:                                                          <L 2539>
                    if (var_efc_is_sparse) {
                        // rownnz = efc_J_rownnz_in[worldid, efcid]                               <L 2540>
                        var_112 = wp::address(var_efc_J_rownnz_in, var_15, var_110);
                        var_114 = wp::load(var_112);
                        var_113 = wp::copy(var_114);
                        // if dofid < rownnz:                                                     <L 2541>
                        var_115 = (var_2 < var_113);
                        if (var_115) {
                            // efc_rowadr = efc_J_rowadr_in[worldid, efcid]                       <L 2542>
                            var_116 = wp::address(var_efc_J_rowadr_in, var_15, var_110);
                            var_118 = wp::load(var_116);
                            var_117 = wp::copy(var_118);
                            // efc_sparseid = efc_rowadr + dofid                                  <L 2543>
                            var_119 = wp::add(var_117, var_2);
                            // colind = efc_J_colind_in[worldid, 0, efc_sparseid]                 <L 2544>
                            var_121 = wp::address(var_efc_J_colind_in, var_15, var_120, var_119);
                            var_123 = wp::load(var_121);
                            var_122 = wp::copy(var_123);
                            // wp.atomic_add(actuator_moment_out[worldid], rowadr + colind, efc_J_in[worldid, 0, efc_sparseid] * efc_force)       <L 2545>
                            var_124 = wp::slice_t(var_15, var_15, var_125);
                            var_126 = wp::view(var_actuator_moment_out, var_124);
                            var_127 = wp::add(var_52, var_122);
                            var_129 = wp::address(var_efc_J_in, var_15, var_128, var_119);
                            var_131 = wp::load(var_129);
                            var_130 = wp::mul(var_131, var_104);
                            var_132 = wp::atomic_add(var_126, var_127, var_130);
                        }
                        var_133 = wp::where(var_115, var_117, var_76);
                        var_134 = wp::where(var_115, var_119, var_78);
                        var_135 = wp::where(var_115, var_122, var_99);
                        if (!var_115) {
                            // return                                                             <L 2547>
                            continue;
                        }
                    }
                    var_136 = wp::where(var_efc_is_sparse, var_113, var_72);
                    var_137 = wp::where(var_efc_is_sparse, var_133, var_76);
                    var_138 = wp::where(var_efc_is_sparse, var_134, var_78);
                    var_139 = wp::where(var_efc_is_sparse, var_135, var_99);
                    if (!var_efc_is_sparse) {
                        // colind = dofid                                                         <L 2549>
                        var_140 = wp::copy(var_2);
                        // wp.atomic_add(actuator_moment_out[worldid], rowadr + colind, efc_J_in[worldid, efcid, dofid] * efc_force)       <L 2550>
                        var_141 = wp::slice_t(var_15, var_15, var_142);
                        var_143 = wp::view(var_actuator_moment_out, var_141);
                        var_144 = wp::add(var_52, var_140);
                        var_145 = wp::address(var_efc_J_in, var_15, var_110, var_2);
                        var_147 = wp::load(var_145);
                        var_146 = wp::mul(var_147, var_104);
                        var_148 = wp::atomic_add(var_143, var_144, var_146);
                    }
                    var_149 = wp::where(var_efc_is_sparse, var_139, var_140);
                    wp::assign(var_72, var_136);
                    wp::assign(var_76, var_137);
                    wp::assign(var_78, var_138);
                    wp::assign(var_99, var_149);
                    goto start_for_4;
                end_for_4:;
            }
        }
        if (!var_55) {
            // elif contact_exclude == 1:                                                         <L 2553>
            var_151 = (var_43 == var_150);
            if (var_151) {
                // contact_pos = contact_pos_in[conid]                                            <L 2554>
                var_152 = wp::address(var_contact_pos_in, var_1);
                var_154 = wp::load(var_152);
                var_153 = wp::copy(var_154);
                // contact_frame = contact_frame_in[conid]                                        <L 2555>
                var_155 = wp::address(var_contact_frame_in, var_1);
                var_157 = wp::load(var_155);
                var_156 = wp::copy(var_157);
                // normal = wp.vec3(contact_frame[0, 0], contact_frame[0, 1], contact_frame[0, 2])       <L 2556>
                var_160 = wp::extract(var_156, var_158, var_159);
                var_163 = wp::extract(var_156, var_161, var_162);
                var_166 = wp::extract(var_156, var_164, var_165);
                var_167 = wp::vec_t<3, wp::float32>(var_160, var_163, var_166);
                // efcid0 = contact_efc_address_in[conid][0]                                      <L 2559>
                var_169 = wp::address(var_contact_efc_address_in, var_1, var_168);
                var_171 = wp::load(var_169);
                var_170 = wp::copy(var_171);
                // if efc_is_sparse and efcid0 >= 0:                                              <L 2560>
                var_173 = (var_170 >= var_172);
                var_174 = var_efc_is_sparse && var_173;
                if (var_174) {
                    // if dofid >= efc_J_rownnz_in[worldid, efcid0]:                              <L 2562>
                    var_175 = wp::address(var_efc_J_rownnz_in, var_15, var_170);
                    var_177 = wp::load(var_175);
                    var_176 = (var_2 >= var_177);
                    if (var_176) {
                        // return                                                                 <L 2563>
                        continue;
                    }
                    // sparseid = efc_J_rowadr_in[worldid, efcid0] + dofid                        <L 2564>
                    var_178 = wp::address(var_efc_J_rowadr_in, var_15, var_170);
                    var_180 = wp::load(var_178);
                    var_179 = wp::add(var_180, var_2);
                    // colind = efc_J_colind_in[worldid, 0, sparseid]                             <L 2565>
                    var_182 = wp::address(var_efc_J_colind_in, var_15, var_181, var_179);
                    var_184 = wp::load(var_182);
                    var_183 = wp::copy(var_184);
                }
                var_185 = wp::where(var_174, var_183, var_99);
                if (!var_174) {
                    // colind = dofid                                                             <L 2568>
                    var_186 = wp::copy(var_2);
                }
                var_187 = wp::where(var_174, var_185, var_186);
                // jacp1, _ = support.jac_dof(                                                    <L 2570>
                // body_parentid, body_rootid, dof_bodyid, subtree_com_in, cdof_in, contact_pos, b1, colind, worldid       <L 2571>
                jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_subtree_com_in, var_cdof_in, var_153, var_30, var_187, var_15, var_188, var_189);
                // jacp2, _ = support.jac_dof(                                                    <L 2573>
                // body_parentid, body_rootid, dof_bodyid, subtree_com_in, cdof_in, contact_pos, b2, colind, worldid       <L 2574>
                jac_dof_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_subtree_com_in, var_cdof_in, var_153, var_33, var_187, var_15, var_190, var_191);
                // jacdif = jacp2 - jacp1                                                         <L 2577>
                var_192 = wp::sub(var_190, var_188);
                // wp.atomic_add(actuator_moment_out[worldid], rowadr + colind, wp.dot(normal, jacdif))       <L 2580>
                var_193 = wp::slice_t(var_15, var_15, var_194);
                var_195 = wp::view(var_actuator_moment_out, var_193);
                var_196 = wp::add(var_52, var_187);
                var_197 = wp::dot(var_167, var_192);
                var_198 = wp::atomic_add(var_195, var_196, var_197);
            }
            var_199 = wp::where(var_151, var_170, var_69);
            var_200 = wp::where(var_151, var_187, var_99);
        }
        var_201 = wp::where(var_55, var_69, var_199);
        var_202 = wp::where(var_55, var_99, var_200);
    }
}



extern "C" __global__ void _subtree_vel_forward_1d7ed456_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::float32> var_body_mass,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_inertia,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_linvel_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_angmom_out,
    wp::array_t<wp::vec_t<6, wp::float32>> var_subtree_bodyvel_out)
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
        wp::vec_t<6, wp::float32>* var_12;
        wp::vec_t<6, wp::float32> var_13;
        wp::vec_t<6, wp::float32> var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::vec_t<3, wp::float32>* var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::mat_t<3, 3, wp::float32>* var_20;
        wp::mat_t<3, 3, wp::float32> var_21;
        wp::mat_t<3, 3, wp::float32> var_22;
        wp::int32* var_23;
        wp::vec_t<3, wp::float32>* var_24;
        wp::int32 var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::float32* var_31;
        wp::vec_t<3, wp::float32> var_32;
        wp::float32 var_33;
        wp::mat_t<3, 3, wp::float32> var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::vec_t<3, wp::float32>* var_36;
        const wp::int32 var_37 = 0;
        wp::float32 var_38;
        wp::vec_t<3, wp::float32> var_39;
        const wp::int32 var_40 = 0;
        const wp::int32 var_41 = 0;
        wp::float32 var_42;
        wp::vec_t<3, wp::float32>* var_43;
        const wp::int32 var_44 = 0;
        wp::float32 var_45;
        wp::vec_t<3, wp::float32> var_46;
        wp::float32 var_47;
        const wp::int32 var_48 = 0;
        wp::vec_t<3, wp::float32>* var_49;
        const wp::int32 var_50 = 1;
        wp::float32 var_51;
        wp::vec_t<3, wp::float32> var_52;
        const wp::int32 var_53 = 1;
        const wp::int32 var_54 = 1;
        wp::float32 var_55;
        wp::vec_t<3, wp::float32>* var_56;
        const wp::int32 var_57 = 1;
        wp::float32 var_58;
        wp::vec_t<3, wp::float32> var_59;
        wp::float32 var_60;
        const wp::int32 var_61 = 1;
        wp::vec_t<3, wp::float32>* var_62;
        const wp::int32 var_63 = 2;
        wp::float32 var_64;
        wp::vec_t<3, wp::float32> var_65;
        const wp::int32 var_66 = 2;
        const wp::int32 var_67 = 2;
        wp::float32 var_68;
        wp::vec_t<3, wp::float32>* var_69;
        const wp::int32 var_70 = 2;
        wp::float32 var_71;
        wp::vec_t<3, wp::float32> var_72;
        wp::float32 var_73;
        const wp::int32 var_74 = 2;
        wp::vec_t<3, wp::float32> var_75;
        wp::vec_t<6, wp::float32> var_76;
        //---------
        // forward
        // def _subtree_vel_forward(                                                              <L 2932>
        // worldid, bodyid = wp.tid()                                                             <L 2948>
        builtin_tid2d(var_0, var_1);
        // body_mass_id = worldid % body_mass.shape[0]                                            <L 2949>
        var_2 = &(var_body_mass.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // body_inertia_id = worldid % body_inertia.shape[0]                                      <L 2950>
        var_7 = &(var_body_inertia.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::mod(var_0, var_9);
        // cvel = cvel_in[worldid, bodyid]                                                        <L 2952>
        var_12 = wp::address(var_cvel_in, var_0, var_1);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // ang = wp.spatial_top(cvel)                                                             <L 2953>
        var_15 = wp::spatial_top(var_13);
        // lin = wp.spatial_bottom(cvel)                                                          <L 2954>
        var_16 = wp::spatial_bottom(var_13);
        // xipos = xipos_in[worldid, bodyid]                                                      <L 2955>
        var_17 = wp::address(var_xipos_in, var_0, var_1);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // ximat = ximat_in[worldid, bodyid]                                                      <L 2956>
        var_20 = wp::address(var_ximat_in, var_0, var_1);
        var_22 = wp::load(var_20);
        var_21 = wp::copy(var_22);
        // subtree_com_root = subtree_com_in[worldid, body_rootid[bodyid]]                        <L 2957>
        var_23 = wp::address(var_body_rootid, var_1);
        var_25 = wp::load(var_23);
        var_24 = wp::address(var_subtree_com_in, var_0, var_25);
        var_27 = wp::load(var_24);
        var_26 = wp::copy(var_27);
        // lin -= wp.cross(xipos - subtree_com_root, ang)                                         <L 2960>
        var_28 = wp::sub(var_18, var_26);
        var_29 = wp::cross(var_28, var_15);
        var_30 = wp::sub(var_16, var_29);
        // subtree_linvel_out[worldid, bodyid] = body_mass[body_mass_id, bodyid] * lin            <L 2962>
        var_31 = wp::address(var_body_mass, var_6, var_1);
        var_33 = wp::load(var_31);
        var_32 = wp::mul(var_33, var_30);
        wp::array_store(var_subtree_linvel_out, var_0, var_1, var_32);
        // dv = wp.transpose(ximat) @ ang                                                         <L 2963>
        var_34 = wp::transpose(var_21);
        var_35 = wp::mul(var_34, var_15);
        // dv[0] *= body_inertia[body_inertia_id, bodyid][0]                                      <L 2964>
        var_36 = wp::address(var_body_inertia, var_11, var_1);
        var_39 = wp::load(var_36);
        var_38 = wp::extract(var_39, var_37);
        var_42 = wp::extract(var_35, var_41);
        var_43 = wp::address(var_body_inertia, var_11, var_1);
        var_46 = wp::load(var_43);
        var_45 = wp::extract(var_46, var_44);
        var_47 = wp::mul(var_42, var_45);
        wp::assign_inplace(var_35, var_48, var_47);
        // dv[1] *= body_inertia[body_inertia_id, bodyid][1]                                      <L 2965>
        var_49 = wp::address(var_body_inertia, var_11, var_1);
        var_52 = wp::load(var_49);
        var_51 = wp::extract(var_52, var_50);
        var_55 = wp::extract(var_35, var_54);
        var_56 = wp::address(var_body_inertia, var_11, var_1);
        var_59 = wp::load(var_56);
        var_58 = wp::extract(var_59, var_57);
        var_60 = wp::mul(var_55, var_58);
        wp::assign_inplace(var_35, var_61, var_60);
        // dv[2] *= body_inertia[body_inertia_id, bodyid][2]                                      <L 2966>
        var_62 = wp::address(var_body_inertia, var_11, var_1);
        var_65 = wp::load(var_62);
        var_64 = wp::extract(var_65, var_63);
        var_68 = wp::extract(var_35, var_67);
        var_69 = wp::address(var_body_inertia, var_11, var_1);
        var_72 = wp::load(var_69);
        var_71 = wp::extract(var_72, var_70);
        var_73 = wp::mul(var_68, var_71);
        wp::assign_inplace(var_35, var_74, var_73);
        // subtree_angmom_out[worldid, bodyid] = ximat @ dv                                       <L 2967>
        var_75 = wp::mul(var_21, var_35);
        wp::array_store(var_subtree_angmom_out, var_0, var_1, var_75);
        // subtree_bodyvel_out[worldid, bodyid] = wp.spatial_vector(ang, lin)                     <L 2968>
        var_76 = wp::vec_t<6, wp::float32>(var_15, var_30);
        wp::array_store(var_subtree_bodyvel_out, var_0, var_1, var_76);
    }
}



extern "C" __global__ void _transmission_body_moment_scale_e0bec4b0_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_actuator_trntype_body_adr,
    wp::array_t<wp::int32> var_moment_rowadr_in,
    wp::array_t<wp::int32> var_actuator_trntype_body_ncon_in,
    wp::array_t<wp::float32> var_actuator_moment_out)
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
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        const wp::int32 var_6 = 0;
        bool var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::int32 var_23;
        //---------
        // forward
        // def _transmission_body_moment_scale(                                                   <L 2584>
        // worldid, trnbodyid, dofid = wp.tid()                                                   <L 2594>
        builtin_tid3d(var_0, var_1, var_2);
        // ncon = actuator_trntype_body_ncon_in[worldid, trnbodyid]                               <L 2596>
        var_3 = wp::address(var_actuator_trntype_body_ncon_in, var_0, var_1);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // if ncon > 0:                                                                           <L 2598>
        var_7 = (var_4 > var_6);
        if (var_7) {
            // actid = actuator_trntype_body_adr[trnbodyid]                                       <L 2599>
            var_8 = wp::address(var_actuator_trntype_body_adr, var_1);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
            // rowadr = moment_rowadr_in[worldid, actid]                                          <L 2600>
            var_11 = wp::address(var_moment_rowadr_in, var_0, var_9);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
            // actuator_moment_out[worldid, rowadr + dofid] /= -float(ncon)                       <L 2601>
            var_14 = wp::float(var_4);
            var_15 = wp::neg(var_14);
            var_16 = wp::add(var_12, var_2);
            var_17 = wp::add(var_12, var_2);
            var_18 = wp::address(var_actuator_moment_out, var_0, var_17);
            var_19 = wp::float(var_4);
            var_20 = wp::neg(var_19);
            var_22 = wp::load(var_18);
            var_21 = wp::div(var_22, var_20);
            var_23 = wp::add(var_12, var_2);
            wp::array_store(var_actuator_moment_out, var_0, var_23, var_21);
        }
    }
}



extern "C" __global__ void _compute_body_inertial_frames_d3bdd81a_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_ipos,
    wp::array_t<wp::quat_t<wp::float32>> var_body_iquat,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_out)
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
        wp::vec_t<3, wp::float32>* var_2;
        wp::vec_t<3, wp::float32> var_3;
        wp::vec_t<3, wp::float32> var_4;
        wp::quat_t<wp::float32>* var_5;
        wp::quat_t<wp::float32> var_6;
        wp::quat_t<wp::float32> var_7;
        wp::shape_t* var_8;
        const wp::int32 var_9 = 0;
        wp::int32 var_10;
        wp::shape_t var_11;
        wp::int32 var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::shape_t* var_17;
        const wp::int32 var_18 = 0;
        wp::int32 var_19;
        wp::shape_t var_20;
        wp::int32 var_21;
        wp::quat_t<wp::float32>* var_22;
        wp::quat_t<wp::float32> var_23;
        wp::quat_t<wp::float32> var_24;
        wp::mat_t<3, 3, wp::float32> var_25;
        //---------
        // forward
        // def _compute_body_inertial_frames(                                                     <L 147>
        // worldid, bodyid = wp.tid()                                                             <L 158>
        builtin_tid2d(var_0, var_1);
        // xpos = xpos_in[worldid, bodyid]                                                        <L 159>
        var_2 = wp::address(var_xpos_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // xquat = xquat_in[worldid, bodyid]                                                      <L 160>
        var_5 = wp::address(var_xquat_in, var_0, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // xipos_out[worldid, bodyid] = xpos + math.rot_vec_quat(body_ipos[worldid % body_ipos.shape[0], bodyid], xquat)       <L 161>
        var_8 = &(var_body_ipos.shape);
        var_11 = wp::load(var_8);
        var_10 = wp::extract(var_11, var_9);
        var_12 = wp::mod(var_0, var_10);
        var_13 = wp::address(var_body_ipos, var_12, var_1);
        var_15 = wp::load(var_13);
        var_14 = rot_vec_quat_0(var_15, var_6);
        var_16 = wp::add(var_3, var_14);
        wp::array_store(var_xipos_out, var_0, var_1, var_16);
        // ximat_out[worldid, bodyid] = math.quat_to_mat(math.mul_quat(xquat, body_iquat[worldid % body_iquat.shape[0], bodyid]))       <L 162>
        var_17 = &(var_body_iquat.shape);
        var_20 = wp::load(var_17);
        var_19 = wp::extract(var_20, var_18);
        var_21 = wp::mod(var_0, var_19);
        var_22 = wp::address(var_body_iquat, var_21, var_1);
        var_24 = wp::load(var_22);
        var_23 = mul_quat_0(var_6, var_24);
        var_25 = quat_to_mat_0(var_23);
        wp::array_store(var_ximat_out, var_0, var_1, var_25);
    }
}



extern "C" __global__ void _linear_momentum_813ce6af_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::float32> var_body_subtreemass,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_linvel_in,
    wp::array_t<wp::int32> var_body_tree_,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_linvel_out)
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
        wp::array_t<wp::vec_t<3, wp::float32>> var_10;
        wp::vec_t<3, wp::float32>* var_11;
        wp::vec_t<3, wp::float32> var_12;
        wp::vec_t<3, wp::float32> var_13;
        const wp::float32 var_14 = 1e-15;
        wp::shape_t* var_15;
        const wp::int32 var_16 = 0;
        wp::int32 var_17;
        wp::shape_t var_18;
        wp::int32 var_19;
        wp::float32* var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::vec_t<3, wp::float32>* var_23;
        wp::shape_t* var_24;
        const wp::int32 var_25 = 0;
        wp::int32 var_26;
        wp::shape_t var_27;
        wp::int32 var_28;
        wp::float32* var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        wp::vec_t<3, wp::float32> var_32;
        wp::vec_t<3, wp::float32> var_33;
        //---------
        // forward
        // def _linear_momentum(                                                                  <L 2972>
        // worldid, nodeid = wp.tid()                                                             <L 2983>
        builtin_tid2d(var_0, var_1);
        // bodyid = body_tree_[nodeid]                                                            <L 2984>
        var_2 = wp::address(var_body_tree_, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if bodyid:                                                                             <L 2985>
        if (var_3) {
            // pid = body_parentid[bodyid]                                                        <L 2986>
            var_5 = wp::address(var_body_parentid, var_3);
            var_7 = wp::load(var_5);
            var_6 = wp::copy(var_7);
            // wp.atomic_add(subtree_linvel_out[worldid], pid, subtree_linvel_in[worldid, bodyid])       <L 2987>
            var_8 = wp::slice_t(var_0, var_0, var_9);
            var_10 = wp::view(var_subtree_linvel_out, var_8);
            var_11 = wp::address(var_subtree_linvel_in, var_0, var_3);
            var_13 = wp::load(var_11);
            var_12 = wp::atomic_add(var_10, var_6, var_13);
        }
        // subtree_linvel_out[worldid, bodyid] /= wp.max(MJ_MINVAL, body_subtreemass[worldid % body_subtreemass.shape[0], bodyid])       <L 2988>
        var_15 = &(var_body_subtreemass.shape);
        var_18 = wp::load(var_15);
        var_17 = wp::extract(var_18, var_16);
        var_19 = wp::mod(var_0, var_17);
        var_20 = wp::address(var_body_subtreemass, var_19, var_3);
        var_22 = wp::load(var_20);
        var_21 = wp::max(var_14, var_22);
        var_23 = wp::address(var_subtree_linvel_out, var_0, var_3);
        var_24 = &(var_body_subtreemass.shape);
        var_27 = wp::load(var_24);
        var_26 = wp::extract(var_27, var_25);
        var_28 = wp::mod(var_0, var_26);
        var_29 = wp::address(var_body_subtreemass, var_28, var_3);
        var_31 = wp::load(var_29);
        var_30 = wp::max(var_14, var_31);
        var_33 = wp::load(var_23);
        var_32 = wp::div(var_33, var_30);
        wp::array_store(var_subtree_linvel_out, var_0, var_3, var_32);
    }
}



extern "C" __global__ void _cdof_3e73eb78_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::int32> var_jnt_bodyid,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xanchor_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xaxis_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_out)
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
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::vec_t<3, wp::float32>* var_11;
        wp::vec_t<3, wp::float32> var_12;
        wp::vec_t<3, wp::float32> var_13;
        wp::mat_t<3, 3, wp::float32>* var_14;
        wp::mat_t<3, 3, wp::float32> var_15;
        wp::mat_t<3, 3, wp::float32> var_16;
        wp::int32* var_17;
        wp::vec_t<3, wp::float32>* var_18;
        wp::int32 var_19;
        wp::vec_t<3, wp::float32>* var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::slice_t var_24;
        const wp::int32 var_25 = 0;
        wp::array_t<wp::vec_t<6, wp::float32>> var_26;
        const wp::int32 var_27 = 0;
        bool var_28;
        const wp::float32 var_29 = 0.0;
        const wp::float32 var_30 = 0.0;
        const wp::float32 var_31 = 0.0;
        const wp::float32 var_32 = 1.0;
        const wp::float32 var_33 = 0.0;
        const wp::float32 var_34 = 0.0;
        wp::vec_t<6, wp::float32> var_35;
        const wp::int32 var_36 = 0;
        wp::int32 var_37;
        const wp::float32 var_38 = 0.0;
        const wp::float32 var_39 = 0.0;
        const wp::float32 var_40 = 0.0;
        const wp::float32 var_41 = 0.0;
        const wp::float32 var_42 = 1.0;
        const wp::float32 var_43 = 0.0;
        wp::vec_t<6, wp::float32> var_44;
        const wp::int32 var_45 = 1;
        wp::int32 var_46;
        const wp::float32 var_47 = 0.0;
        const wp::float32 var_48 = 0.0;
        const wp::float32 var_49 = 0.0;
        const wp::float32 var_50 = 0.0;
        const wp::float32 var_51 = 0.0;
        const wp::float32 var_52 = 1.0;
        wp::vec_t<6, wp::float32> var_53;
        const wp::int32 var_54 = 2;
        wp::int32 var_55;
        const wp::int32 var_56 = 0;
        wp::vec_t<3, wp::float32> var_57;
        const wp::int32 var_58 = 0;
        wp::vec_t<3, wp::float32> var_59;
        wp::vec_t<3, wp::float32> var_60;
        wp::vec_t<6, wp::float32> var_61;
        const wp::int32 var_62 = 3;
        wp::int32 var_63;
        const wp::int32 var_64 = 1;
        wp::vec_t<3, wp::float32> var_65;
        const wp::int32 var_66 = 1;
        wp::vec_t<3, wp::float32> var_67;
        wp::vec_t<3, wp::float32> var_68;
        wp::vec_t<6, wp::float32> var_69;
        const wp::int32 var_70 = 4;
        wp::int32 var_71;
        const wp::int32 var_72 = 2;
        wp::vec_t<3, wp::float32> var_73;
        const wp::int32 var_74 = 2;
        wp::vec_t<3, wp::float32> var_75;
        wp::vec_t<3, wp::float32> var_76;
        wp::vec_t<6, wp::float32> var_77;
        const wp::int32 var_78 = 5;
        wp::int32 var_79;
        const wp::int32 var_80 = 1;
        bool var_81;
        const wp::int32 var_82 = 0;
        wp::vec_t<3, wp::float32> var_83;
        const wp::int32 var_84 = 0;
        wp::vec_t<3, wp::float32> var_85;
        wp::vec_t<3, wp::float32> var_86;
        wp::vec_t<6, wp::float32> var_87;
        const wp::int32 var_88 = 0;
        wp::int32 var_89;
        const wp::int32 var_90 = 1;
        wp::vec_t<3, wp::float32> var_91;
        const wp::int32 var_92 = 1;
        wp::vec_t<3, wp::float32> var_93;
        wp::vec_t<3, wp::float32> var_94;
        wp::vec_t<6, wp::float32> var_95;
        const wp::int32 var_96 = 1;
        wp::int32 var_97;
        const wp::int32 var_98 = 2;
        wp::vec_t<3, wp::float32> var_99;
        const wp::int32 var_100 = 2;
        wp::vec_t<3, wp::float32> var_101;
        wp::vec_t<3, wp::float32> var_102;
        wp::vec_t<6, wp::float32> var_103;
        const wp::int32 var_104 = 2;
        wp::int32 var_105;
        const wp::int32 var_106 = 2;
        bool var_107;
        const wp::float32 var_108 = 0.0;
        wp::vec_t<3, wp::float32> var_109;
        wp::vec_t<6, wp::float32> var_110;
        const wp::int32 var_111 = 3;
        bool var_112;
        wp::vec_t<3, wp::float32> var_113;
        wp::vec_t<6, wp::float32> var_114;
        //---------
        // forward
        // def _cdof(                                                                             <L 557>
        // worldid, jntid = wp.tid()                                                              <L 571>
        builtin_tid2d(var_0, var_1);
        // bodyid = jnt_bodyid[jntid]                                                             <L 572>
        var_2 = wp::address(var_jnt_bodyid, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // dofid = jnt_dofadr[jntid]                                                              <L 573>
        var_5 = wp::address(var_jnt_dofadr, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // jnt_type_ = jnt_type[jntid]                                                            <L 574>
        var_8 = wp::address(var_jnt_type, var_1);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // xaxis = xaxis_in[worldid, jntid]                                                       <L 575>
        var_11 = wp::address(var_xaxis_in, var_0, var_1);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // xmat = wp.transpose(xmat_in[worldid, bodyid])                                          <L 576>
        var_14 = wp::address(var_xmat_in, var_0, var_3);
        var_16 = wp::load(var_14);
        var_15 = wp::transpose(var_16);
        // offset = subtree_com_in[worldid, body_rootid[bodyid]] - xanchor_in[worldid, jntid]       <L 579>
        var_17 = wp::address(var_body_rootid, var_3);
        var_19 = wp::load(var_17);
        var_18 = wp::address(var_subtree_com_in, var_0, var_19);
        var_20 = wp::address(var_xanchor_in, var_0, var_1);
        var_22 = wp::load(var_18);
        var_23 = wp::load(var_20);
        var_21 = wp::sub(var_22, var_23);
        // res = cdof_out[worldid]                                                                <L 581>
        var_24 = wp::slice_t(var_0, var_0, var_25);
        var_26 = wp::view(var_cdof_out, var_24);
        // if jnt_type_ == JointType.FREE:                                                        <L 582>
        var_28 = (var_9 == var_27);
        if (var_28) {
            // res[dofid + 0] = wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)                   <L 583>
            var_35 = wp::vec_t<6, wp::float32>({var_29, var_30, var_31, var_32, var_33, var_34});
            var_37 = wp::add(var_6, var_36);
            wp::array_store(var_26, var_37, var_35);
            // res[dofid + 1] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)                   <L 584>
            var_44 = wp::vec_t<6, wp::float32>({var_38, var_39, var_40, var_41, var_42, var_43});
            var_46 = wp::add(var_6, var_45);
            wp::array_store(var_26, var_46, var_44);
            // res[dofid + 2] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)                   <L 585>
            var_53 = wp::vec_t<6, wp::float32>({var_47, var_48, var_49, var_50, var_51, var_52});
            var_55 = wp::add(var_6, var_54);
            wp::array_store(var_26, var_55, var_53);
            // res[dofid + 3] = wp.spatial_vector(xmat[0], wp.cross(xmat[0], offset))             <L 587>
            var_57 = wp::extract(var_15, var_56);
            var_59 = wp::extract(var_15, var_58);
            var_60 = wp::cross(var_59, var_21);
            var_61 = wp::vec_t<6, wp::float32>(var_57, var_60);
            var_63 = wp::add(var_6, var_62);
            wp::array_store(var_26, var_63, var_61);
            // res[dofid + 4] = wp.spatial_vector(xmat[1], wp.cross(xmat[1], offset))             <L 588>
            var_65 = wp::extract(var_15, var_64);
            var_67 = wp::extract(var_15, var_66);
            var_68 = wp::cross(var_67, var_21);
            var_69 = wp::vec_t<6, wp::float32>(var_65, var_68);
            var_71 = wp::add(var_6, var_70);
            wp::array_store(var_26, var_71, var_69);
            // res[dofid + 5] = wp.spatial_vector(xmat[2], wp.cross(xmat[2], offset))             <L 589>
            var_73 = wp::extract(var_15, var_72);
            var_75 = wp::extract(var_15, var_74);
            var_76 = wp::cross(var_75, var_21);
            var_77 = wp::vec_t<6, wp::float32>(var_73, var_76);
            var_79 = wp::add(var_6, var_78);
            wp::array_store(var_26, var_79, var_77);
        }
        if (!var_28) {
            // elif jnt_type_ == JointType.BALL:  # ball                                          <L 590>
            var_81 = (var_9 == var_80);
            if (var_81) {
                // res[dofid + 0] = wp.spatial_vector(xmat[0], wp.cross(xmat[0], offset))         <L 592>
                var_83 = wp::extract(var_15, var_82);
                var_85 = wp::extract(var_15, var_84);
                var_86 = wp::cross(var_85, var_21);
                var_87 = wp::vec_t<6, wp::float32>(var_83, var_86);
                var_89 = wp::add(var_6, var_88);
                wp::array_store(var_26, var_89, var_87);
                // res[dofid + 1] = wp.spatial_vector(xmat[1], wp.cross(xmat[1], offset))         <L 593>
                var_91 = wp::extract(var_15, var_90);
                var_93 = wp::extract(var_15, var_92);
                var_94 = wp::cross(var_93, var_21);
                var_95 = wp::vec_t<6, wp::float32>(var_91, var_94);
                var_97 = wp::add(var_6, var_96);
                wp::array_store(var_26, var_97, var_95);
                // res[dofid + 2] = wp.spatial_vector(xmat[2], wp.cross(xmat[2], offset))         <L 594>
                var_99 = wp::extract(var_15, var_98);
                var_101 = wp::extract(var_15, var_100);
                var_102 = wp::cross(var_101, var_21);
                var_103 = wp::vec_t<6, wp::float32>(var_99, var_102);
                var_105 = wp::add(var_6, var_104);
                wp::array_store(var_26, var_105, var_103);
            }
            if (!var_81) {
                // elif jnt_type_ == JointType.SLIDE:                                             <L 595>
                var_107 = (var_9 == var_106);
                if (var_107) {
                    // res[dofid] = wp.spatial_vector(wp.vec3(0.0), xaxis)                        <L 596>
                    var_109 = wp::vec_t<3, wp::float32>(var_108);
                    var_110 = wp::vec_t<6, wp::float32>(var_109, var_12);
                    wp::array_store(var_26, var_6, var_110);
                }
                if (!var_107) {
                    // elif jnt_type_ == JointType.HINGE:  # hinge                                <L 597>
                    var_112 = (var_9 == var_111);
                    if (var_112) {
                        // res[dofid] = wp.spatial_vector(xaxis, wp.cross(xaxis, offset))         <L 598>
                        var_113 = wp::cross(var_12, var_21);
                        var_114 = wp::vec_t<6, wp::float32>(var_12, var_113);
                        wp::array_store(var_26, var_6, var_114);
                    }
                }
            }
        }
    }
}



extern "C" __global__ void _angular_momentum_35b41762_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::float32> var_body_mass,
    wp::array_t<wp::float32> var_body_subtreemass,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_linvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_subtree_bodyvel_in,
    wp::array_t<wp::int32> var_body_tree_,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_angmom_out)
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
        const wp::int32 var_5 = 0;
        bool var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::vec_t<3, wp::float32>* var_10;
        wp::vec_t<3, wp::float32> var_11;
        wp::vec_t<3, wp::float32> var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<6, wp::float32>* var_19;
        wp::vec_t<6, wp::float32> var_20;
        wp::vec_t<6, wp::float32> var_21;
        wp::vec_t<3, wp::float32>* var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::vec_t<3, wp::float32>* var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::shape_t* var_28;
        const wp::int32 var_29 = 0;
        wp::int32 var_30;
        wp::shape_t var_31;
        wp::int32 var_32;
        wp::float32* var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::shape_t* var_36;
        const wp::int32 var_37 = 0;
        wp::int32 var_38;
        wp::shape_t var_39;
        wp::int32 var_40;
        wp::float32* var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::vec_t<3, wp::float32> var_44;
        wp::vec_t<3, wp::float32> var_45;
        wp::vec_t<3, wp::float32> var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32> var_48;
        wp::vec_t<3, wp::float32> var_49;
        wp::slice_t var_50;
        const wp::int32 var_51 = 0;
        wp::array_t<wp::vec_t<3, wp::float32>> var_52;
        wp::vec_t<3, wp::float32>* var_53;
        wp::vec_t<3, wp::float32> var_54;
        wp::vec_t<3, wp::float32> var_55;
        wp::vec_t<3, wp::float32> var_56;
        wp::vec_t<3, wp::float32> var_57;
        wp::vec_t<3, wp::float32> var_58;
        wp::vec_t<3, wp::float32> var_59;
        wp::slice_t var_60;
        const wp::int32 var_61 = 0;
        wp::array_t<wp::vec_t<3, wp::float32>> var_62;
        wp::vec_t<3, wp::float32> var_63;
        //---------
        // forward
        // def _angular_momentum(                                                                 <L 2992>
        // worldid, nodeid = wp.tid()                                                             <L 3007>
        builtin_tid2d(var_0, var_1);
        // bodyid = body_tree_[nodeid]                                                            <L 3008>
        var_2 = wp::address(var_body_tree_, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if bodyid == 0:                                                                        <L 3010>
        var_6 = (var_3 == var_5);
        if (var_6) {
            // return                                                                             <L 3011>
            continue;
        }
        // pid = body_parentid[bodyid]                                                            <L 3013>
        var_7 = wp::address(var_body_parentid, var_3);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // xipos = xipos_in[worldid, bodyid]                                                      <L 3015>
        var_10 = wp::address(var_xipos_in, var_0, var_3);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // com = subtree_com_in[worldid, bodyid]                                                  <L 3016>
        var_13 = wp::address(var_subtree_com_in, var_0, var_3);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // com_parent = subtree_com_in[worldid, pid]                                              <L 3017>
        var_16 = wp::address(var_subtree_com_in, var_0, var_8);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // vel = subtree_bodyvel_in[worldid, bodyid]                                              <L 3018>
        var_19 = wp::address(var_subtree_bodyvel_in, var_0, var_3);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // linvel = subtree_linvel_in[worldid, bodyid]                                            <L 3019>
        var_22 = wp::address(var_subtree_linvel_in, var_0, var_3);
        var_24 = wp::load(var_22);
        var_23 = wp::copy(var_24);
        // linvel_parent = subtree_linvel_in[worldid, pid]  # Data field                          <L 3020>
        var_25 = wp::address(var_subtree_linvel_in, var_0, var_8);
        var_27 = wp::load(var_25);
        var_26 = wp::copy(var_27);
        // mass = body_mass[worldid % body_mass.shape[0], bodyid]                                 <L 3021>
        var_28 = &(var_body_mass.shape);
        var_31 = wp::load(var_28);
        var_30 = wp::extract(var_31, var_29);
        var_32 = wp::mod(var_0, var_30);
        var_33 = wp::address(var_body_mass, var_32, var_3);
        var_35 = wp::load(var_33);
        var_34 = wp::copy(var_35);
        // subtreemass = body_subtreemass[worldid % body_subtreemass.shape[0], bodyid]            <L 3022>
        var_36 = &(var_body_subtreemass.shape);
        var_39 = wp::load(var_36);
        var_38 = wp::extract(var_39, var_37);
        var_40 = wp::mod(var_0, var_38);
        var_41 = wp::address(var_body_subtreemass, var_40, var_3);
        var_43 = wp::load(var_41);
        var_42 = wp::copy(var_43);
        // dx = xipos - com                                                                       <L 3025>
        var_44 = wp::sub(var_11, var_14);
        // dv = wp.spatial_bottom(vel) - linvel                                                   <L 3026>
        var_45 = wp::spatial_bottom(var_20);
        var_46 = wp::sub(var_45, var_23);
        // dp = dv * mass                                                                         <L 3027>
        var_47 = wp::mul(var_46, var_34);
        // dL = wp.cross(dx, dp)                                                                  <L 3028>
        var_48 = wp::cross(var_44, var_47);
        // subtree_angmom_out[worldid, bodyid] += dL                                              <L 3031>
        var_49 = wp::atomic_add(var_subtree_angmom_out, var_0, var_3, var_48);
        // wp.atomic_add(subtree_angmom_out[worldid], pid, subtree_angmom_out[worldid, bodyid])       <L 3034>
        var_50 = wp::slice_t(var_0, var_0, var_51);
        var_52 = wp::view(var_subtree_angmom_out, var_50);
        var_53 = wp::address(var_subtree_angmom_out, var_0, var_3);
        var_55 = wp::load(var_53);
        var_54 = wp::atomic_add(var_52, var_8, var_55);
        // dx = com - com_parent                                                                  <L 3037>
        var_56 = wp::sub(var_14, var_17);
        // dv = linvel - linvel_parent                                                            <L 3038>
        var_57 = wp::sub(var_23, var_26);
        // dv *= subtreemass                                                                      <L 3039>
        var_58 = wp::mul(var_57, var_42);
        // dL = wp.cross(dx, dv)                                                                  <L 3040>
        var_59 = wp::cross(var_56, var_58);
        // wp.atomic_add(subtree_angmom_out[worldid], pid, dL)                                    <L 3041>
        var_60 = wp::slice_t(var_0, var_0, var_61);
        var_62 = wp::view(var_subtree_angmom_out, var_60);
        var_63 = wp::atomic_add(var_62, var_8, var_59);
    }
}



extern "C" __global__ void _tendon_dot_62db264d_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::int32> var_dof_jntid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_tendon_adr,
    wp::array_t<wp::int32> var_tendon_num,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::float32> var_tendon_armature,
    wp::array_t<wp::int32> var_wrap_type,
    wp::array_t<wp::int32> var_wrap_objid,
    wp::array_t<wp::float32> var_wrap_prm,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_dot_in,
    wp::array_t<wp::float32> var_ten_Jdot_out)
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
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::int32* var_15;
        const wp::int32 var_16 = 1;
        bool var_17;
        wp::int32 var_18;
        const wp::float32 var_19 = 1.0;
        wp::float32 var_20;
        wp::int32* var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        const wp::int32 var_24 = 0;
        wp::int32 var_25;
        const wp::int32 var_26 = 1;
        wp::int32 var_27;
        bool var_28;
        wp::int32 var_29;
        const wp::int32 var_30 = 0;
        wp::int32 var_31;
        wp::int32* var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        wp::int32 var_35;
        const wp::int32 var_36 = 1;
        wp::int32 var_37;
        wp::int32* var_38;
        wp::int32 var_39;
        wp::int32 var_40;
        wp::int32 var_41;
        const wp::int32 var_42 = 0;
        wp::int32 var_43;
        wp::int32* var_44;
        wp::int32 var_45;
        wp::int32 var_46;
        wp::int32 var_47;
        const wp::int32 var_48 = 1;
        wp::int32 var_49;
        wp::int32* var_50;
        wp::int32 var_51;
        wp::int32 var_52;
        const wp::int32 var_53 = 2;
        bool var_54;
        bool var_55;
        bool var_56;
        bool var_57;
        wp::int32 var_58;
        wp::float32* var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::float32 var_62;
        const wp::int32 var_63 = 1;
        wp::int32 var_64;
        wp::vec_t<3, wp::float32>* var_65;
        wp::vec_t<3, wp::float32> var_66;
        wp::vec_t<3, wp::float32> var_67;
        wp::int32* var_68;
        wp::int32 var_69;
        wp::int32 var_70;
        wp::vec_t<6, wp::float32>* var_71;
        wp::vec_t<6, wp::float32> var_72;
        wp::vec_t<6, wp::float32> var_73;
        wp::int32* var_74;
        wp::vec_t<3, wp::float32>* var_75;
        wp::int32 var_76;
        wp::vec_t<3, wp::float32> var_77;
        wp::vec_t<3, wp::float32> var_78;
        wp::vec_t<3, wp::float32> var_79;
        wp::vec_t<3, wp::float32> var_80;
        wp::vec_t<3, wp::float32> var_81;
        wp::vec_t<3, wp::float32> var_82;
        wp::vec_t<3, wp::float32> var_83;
        const wp::int32 var_84 = 4;
        bool var_85;
        const wp::int32 var_86 = 5;
        bool var_87;
        bool var_88;
        wp::int32* var_89;
        wp::int32 var_90;
        wp::int32 var_91;
        wp::vec_t<3, wp::float32>* var_92;
        wp::vec_t<3, wp::float32> var_93;
        wp::vec_t<3, wp::float32> var_94;
        wp::vec_t<6, wp::float32>* var_95;
        wp::vec_t<6, wp::float32> var_96;
        wp::vec_t<6, wp::float32> var_97;
        wp::int32* var_98;
        wp::vec_t<3, wp::float32>* var_99;
        wp::int32 var_100;
        wp::vec_t<3, wp::float32> var_101;
        wp::vec_t<3, wp::float32> var_102;
        wp::vec_t<3, wp::float32> var_103;
        wp::vec_t<3, wp::float32> var_104;
        wp::vec_t<3, wp::float32> var_105;
        wp::vec_t<3, wp::float32> var_106;
        wp::vec_t<3, wp::float32> var_107;
        bool var_108;
        wp::vec_t<3, wp::float32> var_109;
        wp::vec_t<3, wp::float32> var_110;
        wp::float32 var_111;
        wp::vec_t<3, wp::float32> var_112;
        wp::vec_t<3, wp::float32> var_113;
        wp::vec_t<3, wp::float32> var_114;
        wp::vec_t<3, wp::float32> var_115;
        wp::vec_t<3, wp::float32> var_116;
        wp::vec_t<3, wp::float32> var_117;
        wp::vec_t<3, wp::float32> var_118;
        wp::vec_t<3, wp::float32> var_119;
        wp::vec_t<3, wp::float32> var_120;
        wp::vec_t<3, wp::float32> var_121;
        wp::vec_t<3, wp::float32> var_122;
        wp::float32 var_123;
        wp::float32 var_124;
        wp::vec_t<3, wp::float32> var_125;
        wp::vec_t<3, wp::float32> var_126;
        const wp::float32 var_127 = 1e-15;
        bool var_128;
        wp::vec_t<3, wp::float32> var_129;
        wp::vec_t<3, wp::float32> var_130;
        const wp::float32 var_131 = 0.0;
        wp::vec_t<3, wp::float32> var_132;
        wp::vec_t<3, wp::float32> var_133;
        wp::int32* var_134;
        wp::int32 var_135;
        wp::int32 var_136;
        wp::int32* var_137;
        wp::int32 var_138;
        wp::int32 var_139;
        const wp::float32 var_140 = 1.0;
        wp::float32 var_141;
        wp::float32 var_142;
        wp::float32 var_143;
        const wp::int32 var_144 = 1;
        wp::int32 var_145;
        //---------
        // forward
        // def _tendon_dot(                                                                       <L 1656>
        // worldid, tenid = wp.tid()                                                              <L 1684>
        builtin_tid2d(var_0, var_1);
        // armature = tendon_armature[worldid % tendon_armature.shape[0], tenid]                  <L 1686>
        var_2 = &(var_tendon_armature.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_tendon_armature, var_6, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // if armature == 0.0:                                                                    <L 1687>
        var_11 = (var_8 == var_10);
        if (var_11) {
            // return                                                                             <L 1688>
            continue;
        }
        // adr = tendon_adr[tenid]                                                                <L 1691>
        var_12 = wp::address(var_tendon_adr, var_1);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // if wrap_type[adr] == WrapType.JOINT:                                                   <L 1692>
        var_15 = wp::address(var_wrap_type, var_13);
        var_18 = wp::load(var_15);
        var_17 = (var_18 == var_16);
        if (var_17) {
            // return                                                                             <L 1693>
            continue;
        }
        // divisor = float(1.0)                                                                   <L 1696>
        var_20 = wp::float(var_19);
        // num = tendon_num[tenid]                                                                <L 1697>
        var_21 = wp::address(var_tendon_num, var_1);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // j = int(0)                                                                             <L 1698>
        var_25 = wp::int(var_24);
        // while j < num - 1:                                                                     <L 1699>
        start_while_2:;
        var_27 = wp::sub(var_22, var_26);
        var_28 = (var_25 < var_27);
        if ((var_28) == false) goto end_while_2;
            // type0 = wrap_type[adr + j + 0]                                                     <L 1701>
            var_29 = wp::add(var_13, var_25);
            var_31 = wp::add(var_29, var_30);
            var_32 = wp::address(var_wrap_type, var_31);
            var_34 = wp::load(var_32);
            var_33 = wp::copy(var_34);
            // type1 = wrap_type[adr + j + 1]                                                     <L 1702>
            var_35 = wp::add(var_13, var_25);
            var_37 = wp::add(var_35, var_36);
            var_38 = wp::address(var_wrap_type, var_37);
            var_40 = wp::load(var_38);
            var_39 = wp::copy(var_40);
            // id0 = wrap_objid[adr + j + 0]                                                      <L 1703>
            var_41 = wp::add(var_13, var_25);
            var_43 = wp::add(var_41, var_42);
            var_44 = wp::address(var_wrap_objid, var_43);
            var_46 = wp::load(var_44);
            var_45 = wp::copy(var_46);
            // id1 = wrap_objid[adr + j + 1]                                                      <L 1704>
            var_47 = wp::add(var_13, var_25);
            var_49 = wp::add(var_47, var_48);
            var_50 = wp::address(var_wrap_objid, var_49);
            var_52 = wp::load(var_50);
            var_51 = wp::copy(var_52);
            // pulley = WrapType.PULLEY                                                           <L 1707>
            // if (type0 == pulley) or (type1 == pulley):                                         <L 1708>
            var_54 = (var_33 == var_53);
            var_55 = (var_39 == var_53);
            var_56 = var_54 || var_55;
            if (var_56) {
                // if type0 == pulley:                                                            <L 1710>
                var_57 = (var_33 == var_53);
                if (var_57) {
                    // divisor = wrap_prm[adr + j]                                                <L 1711>
                    var_58 = wp::add(var_13, var_25);
                    var_59 = wp::address(var_wrap_prm, var_58);
                    var_61 = wp::load(var_59);
                    var_60 = wp::copy(var_61);
                }
                var_62 = wp::where(var_57, var_60, var_20);
                // j += 1                                                                         <L 1713>
                var_64 = wp::add(var_25, var_63);
                // continue                                                                       <L 1714>
                wp::assign(var_20, var_62);
                wp::assign(var_25, var_64);
                goto start_while_2;
            }
            // wpnt0 = site_xpos_in[worldid, id0]                                                 <L 1717>
            var_65 = wp::address(var_site_xpos_in, var_0, var_45);
            var_67 = wp::load(var_65);
            var_66 = wp::copy(var_67);
            // wbody0 = site_bodyid[id0]                                                          <L 1719>
            var_68 = wp::address(var_site_bodyid, var_45);
            var_70 = wp::load(var_68);
            var_69 = wp::copy(var_70);
            // cvel0 = cvel_in[worldid, wbody0]                                                   <L 1720>
            var_71 = wp::address(var_cvel_in, var_0, var_69);
            var_73 = wp::load(var_71);
            var_72 = wp::copy(var_73);
            // subtree_com0 = subtree_com_in[worldid, body_rootid[wbody0]]                        <L 1721>
            var_74 = wp::address(var_body_rootid, var_69);
            var_76 = wp::load(var_74);
            var_75 = wp::address(var_subtree_com_in, var_0, var_76);
            var_78 = wp::load(var_75);
            var_77 = wp::copy(var_78);
            // offset0 = wpnt0 - subtree_com0                                                     <L 1722>
            var_79 = wp::sub(var_66, var_77);
            // pvel_lin0 = wp.spatial_bottom(cvel0) - wp.cross(offset0, wp.spatial_top(cvel0))       <L 1723>
            var_80 = wp::spatial_bottom(var_72);
            var_81 = wp::spatial_top(var_72);
            var_82 = wp::cross(var_79, var_81);
            var_83 = wp::sub(var_80, var_82);
            // if (type1 == WrapType.SPHERE) or (type1 == WrapType.CYLINDER):                     <L 1726>
            var_85 = (var_39 == var_84);
            var_87 = (var_39 == var_86);
            var_88 = var_85 || var_87;
            if (var_88) {
                // return                                                                         <L 1728>
                continue;
            }
            // wbody1 = site_bodyid[id1]                                                          <L 1731>
            var_89 = wp::address(var_site_bodyid, var_51);
            var_91 = wp::load(var_89);
            var_90 = wp::copy(var_91);
            // wpnt1 = site_xpos_in[worldid, id1]                                                 <L 1732>
            var_92 = wp::address(var_site_xpos_in, var_0, var_51);
            var_94 = wp::load(var_92);
            var_93 = wp::copy(var_94);
            // cvel1 = cvel_in[worldid, wbody1]                                                   <L 1734>
            var_95 = wp::address(var_cvel_in, var_0, var_90);
            var_97 = wp::load(var_95);
            var_96 = wp::copy(var_97);
            // subtree_com1 = subtree_com_in[worldid, body_rootid[wbody1]]                        <L 1735>
            var_98 = wp::address(var_body_rootid, var_90);
            var_100 = wp::load(var_98);
            var_99 = wp::address(var_subtree_com_in, var_0, var_100);
            var_102 = wp::load(var_99);
            var_101 = wp::copy(var_102);
            // offset1 = wpnt1 - subtree_com1                                                     <L 1736>
            var_103 = wp::sub(var_93, var_101);
            // pvel_lin1 = wp.spatial_bottom(cvel1) - wp.cross(offset1, wp.spatial_top(cvel1))       <L 1737>
            var_104 = wp::spatial_bottom(var_96);
            var_105 = wp::spatial_top(var_96);
            var_106 = wp::cross(var_103, var_105);
            var_107 = wp::sub(var_104, var_106);
            // if wbody0 != wbody1:                                                               <L 1740>
            var_108 = (var_69 != var_90);
            if (var_108) {
                // dpnt, norm = math.normalize_with_norm(wpnt1 - wpnt0)                           <L 1742>
                var_109 = wp::sub(var_93, var_66);
                normalize_with_norm_0(var_109, var_110, var_111);
                // wvel0 = wp.spatial_bottom(cvel0) - wp.cross(wpnt0 - subtree_com0, wp.spatial_top(cvel0))       <L 1745>
                var_112 = wp::spatial_bottom(var_72);
                var_113 = wp::sub(var_66, var_77);
                var_114 = wp::spatial_top(var_72);
                var_115 = wp::cross(var_113, var_114);
                var_116 = wp::sub(var_112, var_115);
                // wvel1 = wp.spatial_bottom(cvel1) - wp.cross(wpnt1 - subtree_com1, wp.spatial_top(cvel1))       <L 1746>
                var_117 = wp::spatial_bottom(var_96);
                var_118 = wp::sub(var_93, var_101);
                var_119 = wp::spatial_top(var_96);
                var_120 = wp::cross(var_118, var_119);
                var_121 = wp::sub(var_117, var_120);
                // dvel = wvel1 - wvel0                                                           <L 1747>
                var_122 = wp::sub(var_121, var_116);
                // dot = wp.dot(dpnt, dvel)                                                       <L 1748>
                var_123 = wp::dot(var_110, var_122);
                // dvel += dpnt * (-dot)                                                          <L 1749>
                var_124 = wp::neg(var_123);
                var_125 = wp::mul(var_110, var_124);
                var_126 = wp::add(var_122, var_125);
                // if norm > MJ_MINVAL:                                                           <L 1750>
                var_128 = (var_111 > var_127);
                if (var_128) {
                    // dvel /= norm                                                               <L 1751>
                    var_129 = wp::div(var_126, var_111);
                }
                var_130 = wp::where(var_128, var_129, var_126);
                if (!var_128) {
                    // dvel = wp.vec3(0.0)                                                        <L 1753>
                    var_132 = wp::vec_t<3, wp::float32>(var_131);
                }
                var_133 = wp::where(var_128, var_130, var_132);
                // rownnz = ten_J_rownnz[tenid]                                                   <L 1755>
                var_134 = wp::address(var_ten_J_rownnz, var_1);
                var_136 = wp::load(var_134);
                var_135 = wp::copy(var_136);
                // rowadr = ten_J_rowadr[tenid]                                                   <L 1756>
                var_137 = wp::address(var_ten_J_rowadr, var_1);
                var_139 = wp::load(var_137);
                var_138 = wp::copy(var_139);
                // inv_divisor = math.safe_div(float(1.0), divisor)                               <L 1757>
                var_141 = wp::float(var_140);
                var_142 = safe_div_0(var_141, var_20);
                // _accumulate_jac_dot_chain(                                                     <L 1760>
                // body_parentid,                                                                 <L 1761>
                // body_dofnum,                                                                   <L 1762>
                // body_dofadr,                                                                   <L 1763>
                // jnt_type,                                                                      <L 1764>
                // jnt_dofadr,                                                                    <L 1765>
                // dof_jntid,                                                                     <L 1766>
                // ten_J_colind,                                                                  <L 1767>
                // cdof_in,                                                                       <L 1768>
                // cvel_in,                                                                       <L 1769>
                // cdof_dot_in,                                                                   <L 1770>
                // offset0,                                                                       <L 1771>
                // pvel_lin0,                                                                     <L 1772>
                // dpnt,                                                                          <L 1773>
                // dvel,                                                                          <L 1774>
                // wbody0,                                                                        <L 1775>
                // rowadr,                                                                        <L 1776>
                // rownnz,                                                                        <L 1777>
                // -inv_divisor,                                                                  <L 1778>
                var_143 = wp::neg(var_142);
                // worldid,                                                                       <L 1779>
                // ten_Jdot_out,                                                                  <L 1780>
                _accumulate_jac_dot_chain_0(var_body_parentid, var_body_dofnum, var_body_dofadr, var_jnt_type, var_jnt_dofadr, var_dof_jntid, var_ten_J_colind, var_cdof_in, var_cvel_in, var_cdof_dot_in, var_79, var_83, var_110, var_133, var_69, var_138, var_135, var_143, var_0, var_ten_Jdot_out);
                // _accumulate_jac_dot_chain(                                                     <L 1782>
                // body_parentid,                                                                 <L 1783>
                // body_dofnum,                                                                   <L 1784>
                // body_dofadr,                                                                   <L 1785>
                // jnt_type,                                                                      <L 1786>
                // jnt_dofadr,                                                                    <L 1787>
                // dof_jntid,                                                                     <L 1788>
                // ten_J_colind,                                                                  <L 1789>
                // cdof_in,                                                                       <L 1790>
                // cvel_in,                                                                       <L 1791>
                // cdof_dot_in,                                                                   <L 1792>
                // offset1,                                                                       <L 1793>
                // pvel_lin1,                                                                     <L 1794>
                // dpnt,                                                                          <L 1795>
                // dvel,                                                                          <L 1796>
                // wbody1,                                                                        <L 1797>
                // rowadr,                                                                        <L 1798>
                // rownnz,                                                                        <L 1799>
                // inv_divisor,                                                                   <L 1800>
                // worldid,                                                                       <L 1801>
                // ten_Jdot_out,                                                                  <L 1802>
                _accumulate_jac_dot_chain_0(var_body_parentid, var_body_dofnum, var_body_dofadr, var_jnt_type, var_jnt_dofadr, var_dof_jntid, var_ten_J_colind, var_cdof_in, var_cvel_in, var_cdof_dot_in, var_103, var_107, var_110, var_133, var_90, var_138, var_135, var_142, var_0, var_ten_Jdot_out);
            }
            // j += 1                                                                             <L 1806>
            var_145 = wp::add(var_25, var_144);
            wp::assign(var_25, var_145);
        goto start_while_2;
        end_while_2:;
    }
}



extern "C" __global__ void _flex_edges_c2df918d_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_nflex,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_flex_vertadr,
    wp::array_t<wp::int32> var_flex_edgeadr,
    wp::array_t<wp::int32> var_flex_edgenum,
    wp::array_t<wp::int32> var_flex_vertbodyid,
    wp::array_t<wp::vec_t<2, wp::int32>> var_flex_edge,
    wp::array_t<wp::int32> var_flexedge_J_rowadr,
    wp::array_t<wp::int32> var_flexedge_J_colind,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexvert_xpos_in,
    wp::array_t<wp::float32> var_flexedge_J_out,
    wp::array_t<wp::float32> var_flexedge_length_out,
    wp::array_t<wp::float32> var_flexedge_velocity_out)
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
        wp::range_t var_2;
        wp::int32 var_3;
        wp::int32* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        wp::int32* var_9;
        bool var_10;
        wp::int32 var_11;
        bool var_12;
        wp::int32 var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::vec_t<2, wp::int32>* var_17;
        wp::vec_t<2, wp::int32> var_18;
        wp::vec_t<2, wp::int32> var_19;
        const wp::int32 var_20 = 0;
        wp::int32 var_21;
        wp::int32 var_22;
        const wp::int32 var_23 = 1;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::vec_t<3, wp::float32>* var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32>* var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::vec_t<3, wp::float32> var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::float32 var_34;
        wp::int32* var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        wp::int32* var_38;
        wp::int32 var_39;
        wp::int32 var_40;
        wp::int32* var_41;
        wp::int32 var_42;
        wp::int32 var_43;
        wp::int32* var_44;
        wp::int32 var_45;
        wp::int32 var_46;
        const wp::float32 var_47 = 0.0;
        wp::float32 var_48;
        const wp::int32 var_49 = 0;
        bool var_50;
        wp::int32* var_51;
        wp::int32 var_52;
        wp::int32 var_53;
        wp::int32* var_54;
        wp::vec_t<3, wp::float32>* var_55;
        wp::int32 var_56;
        wp::vec_t<3, wp::float32> var_57;
        wp::vec_t<3, wp::float32> var_58;
        wp::vec_t<3, wp::float32> var_59;
        wp::range_t var_60;
        wp::int32 var_61;
        wp::int32 var_62;
        wp::vec_t<6, wp::float32>* var_63;
        wp::vec_t<6, wp::float32> var_64;
        wp::vec_t<6, wp::float32> var_65;
        wp::vec_t<3, wp::float32> var_66;
        wp::vec_t<3, wp::float32> var_67;
        wp::vec_t<3, wp::float32> var_68;
        wp::vec_t<3, wp::float32> var_69;
        wp::float32 var_70;
        wp::int32 var_71;
        wp::float32* var_72;
        wp::float32 var_73;
        wp::float32 var_74;
        wp::float32 var_75;
        const wp::int32 var_76 = 0;
        bool var_77;
        wp::int32* var_78;
        wp::int32 var_79;
        wp::int32 var_80;
        wp::int32* var_81;
        wp::vec_t<3, wp::float32>* var_82;
        wp::int32 var_83;
        wp::vec_t<3, wp::float32> var_84;
        wp::vec_t<3, wp::float32> var_85;
        wp::vec_t<3, wp::float32> var_86;
        wp::range_t var_87;
        wp::int32 var_88;
        wp::int32 var_89;
        wp::vec_t<6, wp::float32>* var_90;
        wp::vec_t<6, wp::float32> var_91;
        wp::vec_t<6, wp::float32> var_92;
        wp::vec_t<3, wp::float32> var_93;
        wp::vec_t<3, wp::float32> var_94;
        wp::vec_t<3, wp::float32> var_95;
        wp::vec_t<3, wp::float32> var_96;
        wp::float32 var_97;
        wp::int32 var_98;
        wp::float32* var_99;
        wp::float32 var_100;
        wp::float32 var_101;
        wp::float32 var_102;
        wp::int32 var_103;
        wp::int32* var_104;
        wp::int32 var_105;
        wp::int32 var_106;
        const wp::int32 var_107 = 0;
        const wp::int32 var_108 = 0;
        bool var_109;
        wp::int32* var_110;
        wp::int32 var_111;
        wp::int32 var_112;
        wp::int32* var_113;
        wp::vec_t<3, wp::float32>* var_114;
        wp::int32 var_115;
        wp::vec_t<3, wp::float32> var_116;
        wp::vec_t<3, wp::float32> var_117;
        wp::vec_t<3, wp::float32> var_118;
        wp::range_t var_119;
        wp::int32 var_120;
        wp::int32 var_121;
        wp::vec_t<6, wp::float32>* var_122;
        wp::vec_t<6, wp::float32> var_123;
        wp::vec_t<6, wp::float32> var_124;
        wp::vec_t<3, wp::float32> var_125;
        wp::vec_t<3, wp::float32> var_126;
        wp::vec_t<3, wp::float32> var_127;
        wp::vec_t<3, wp::float32> var_128;
        wp::vec_t<3, wp::float32> var_129;
        wp::float32 var_130;
        wp::int32 var_131;
        wp::int32 var_132;
        wp::int32 var_133;
        wp::int32 var_134;
        wp::vec_t<3, wp::float32> var_135;
        wp::int32 var_136;
        wp::int32 var_137;
        const wp::int32 var_138 = 0;
        bool var_139;
        wp::int32* var_140;
        wp::int32 var_141;
        wp::int32 var_142;
        wp::int32* var_143;
        wp::vec_t<3, wp::float32>* var_144;
        wp::int32 var_145;
        wp::vec_t<3, wp::float32> var_146;
        wp::vec_t<3, wp::float32> var_147;
        wp::vec_t<3, wp::float32> var_148;
        wp::range_t var_149;
        wp::int32 var_150;
        wp::int32 var_151;
        wp::vec_t<6, wp::float32>* var_152;
        wp::vec_t<6, wp::float32> var_153;
        wp::vec_t<6, wp::float32> var_154;
        wp::vec_t<3, wp::float32> var_155;
        wp::vec_t<3, wp::float32> var_156;
        wp::vec_t<3, wp::float32> var_157;
        wp::vec_t<3, wp::float32> var_158;
        wp::float32 var_159;
        wp::int32 var_160;
        wp::int32 var_161;
        wp::int32 var_162;
        wp::int32 var_163;
        wp::vec_t<3, wp::float32> var_164;
        //---------
        // forward
        // def _flex_edges(                                                                       <L 261>
        // worldid, edgeid = wp.tid()                                                             <L 284>
        builtin_tid2d(var_0, var_1);
        // for i in range(nflex):                                                                 <L 285>
        var_2 = wp::range(var_nflex);
        start_for_0:;
            if (iter_cmp(var_2) == 0) goto end_for_0;
            var_3 = wp::iter_next(var_2);
            // locid = edgeid - flex_edgeadr[i]                                                   <L 286>
            var_4 = wp::address(var_flex_edgeadr, var_3);
            var_6 = wp::load(var_4);
            var_5 = wp::sub(var_1, var_6);
            // if locid >= 0 and locid < flex_edgenum[i]:                                         <L 287>
            var_8 = (var_5 >= var_7);
            var_9 = wp::address(var_flex_edgenum, var_3);
            var_11 = wp::load(var_9);
            var_10 = (var_5 < var_11);
            var_12 = var_8 && var_10;
            if (var_12) {
                // f = i                                                                          <L 288>
                var_13 = wp::copy(var_3);
                // break                                                                          <L 289>
                goto end_for_0;
            }
            goto start_for_0;
        end_for_0:;
        // vbase = flex_vertadr[f]                                                                <L 291>
        var_14 = wp::address(var_flex_vertadr, var_13);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // v = flex_edge[edgeid]                                                                  <L 292>
        var_17 = wp::address(var_flex_edge, var_1);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // vbase0 = vbase + v[0]                                                                  <L 293>
        var_21 = wp::extract(var_18, var_20);
        var_22 = wp::add(var_15, var_21);
        // vbase1 = vbase + v[1]                                                                  <L 294>
        var_24 = wp::extract(var_18, var_23);
        var_25 = wp::add(var_15, var_24);
        // pos1 = flexvert_xpos_in[worldid, vbase0]                                               <L 296>
        var_26 = wp::address(var_flexvert_xpos_in, var_0, var_22);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // pos2 = flexvert_xpos_in[worldid, vbase1]                                               <L 297>
        var_29 = wp::address(var_flexvert_xpos_in, var_0, var_25);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // vec = pos2 - pos1                                                                      <L 298>
        var_32 = wp::sub(var_30, var_27);
        // edge, edge_length = math.normalize_with_norm(vec)                                      <L 299>
        normalize_with_norm_0(var_32, var_33, var_34);
        // flexedge_length_out[worldid, edgeid] = edge_length                                     <L 300>
        wp::array_store(var_flexedge_length_out, var_0, var_1, var_34);
        // b1 = flex_vertbodyid[vbase0]                                                           <L 302>
        var_35 = wp::address(var_flex_vertbodyid, var_22);
        var_37 = wp::load(var_35);
        var_36 = wp::copy(var_37);
        // b2 = flex_vertbodyid[vbase1]                                                           <L 303>
        var_38 = wp::address(var_flex_vertbodyid, var_25);
        var_40 = wp::load(var_38);
        var_39 = wp::copy(var_40);
        // dofnum1 = body_dofnum[b1]                                                              <L 305>
        var_41 = wp::address(var_body_dofnum, var_36);
        var_43 = wp::load(var_41);
        var_42 = wp::copy(var_43);
        // dofnum2 = body_dofnum[b2]                                                              <L 306>
        var_44 = wp::address(var_body_dofnum, var_39);
        var_46 = wp::load(var_44);
        var_45 = wp::copy(var_46);
        // vel = float(0.0)                                                                       <L 309>
        var_48 = wp::float(var_47);
        // if dofnum1 > 0:                                                                        <L 310>
        var_50 = (var_42 > var_49);
        if (var_50) {
            // dofi = body_dofadr[b1]                                                             <L 311>
            var_51 = wp::address(var_body_dofadr, var_36);
            var_53 = wp::load(var_51);
            var_52 = wp::copy(var_53);
            // offset1 = pos1 - wp.vec3(subtree_com_in[worldid, body_rootid[b1]])                 <L 312>
            var_54 = wp::address(var_body_rootid, var_36);
            var_56 = wp::load(var_54);
            var_55 = wp::address(var_subtree_com_in, var_0, var_56);
            var_58 = wp::load(var_55);
            var_57 = wp::vec_t<3, wp::float32>(var_58);
            var_59 = wp::sub(var_27, var_57);
            // for k in range(dofnum1):                                                           <L 313>
            var_60 = wp::range(var_42);
            start_for_2:;
                if (iter_cmp(var_60) == 0) goto end_for_2;
                var_61 = wp::iter_next(var_60);
                // cdof = cdof_in[worldid, dofi + k]                                              <L 314>
                var_62 = wp::add(var_52, var_61);
                var_63 = wp::address(var_cdof_in, var_0, var_62);
                var_65 = wp::load(var_63);
                var_64 = wp::copy(var_65);
                // cdof_ang = wp.spatial_top(cdof)                                                <L 315>
                var_66 = wp::spatial_top(var_64);
                // cdof_lin = wp.spatial_bottom(cdof)                                             <L 316>
                var_67 = wp::spatial_bottom(var_64);
                // jacp1 = cdof_lin + wp.cross(cdof_ang, offset1)                                 <L 317>
                var_68 = wp::cross(var_66, var_59);
                var_69 = wp::add(var_67, var_68);
                // vel -= wp.dot(jacp1, edge) * qvel_in[worldid, dofi + k]                        <L 318>
                var_70 = wp::dot(var_69, var_33);
                var_71 = wp::add(var_52, var_61);
                var_72 = wp::address(var_qvel_in, var_0, var_71);
                var_74 = wp::load(var_72);
                var_73 = wp::mul(var_70, var_74);
                var_75 = wp::sub(var_48, var_73);
                wp::assign(var_48, var_75);
                goto start_for_2;
            end_for_2:;
        }
        // if dofnum2 > 0:                                                                        <L 319>
        var_77 = (var_45 > var_76);
        if (var_77) {
            // dofj = body_dofadr[b2]                                                             <L 320>
            var_78 = wp::address(var_body_dofadr, var_39);
            var_80 = wp::load(var_78);
            var_79 = wp::copy(var_80);
            // offset2 = pos2 - wp.vec3(subtree_com_in[worldid, body_rootid[b2]])                 <L 321>
            var_81 = wp::address(var_body_rootid, var_39);
            var_83 = wp::load(var_81);
            var_82 = wp::address(var_subtree_com_in, var_0, var_83);
            var_85 = wp::load(var_82);
            var_84 = wp::vec_t<3, wp::float32>(var_85);
            var_86 = wp::sub(var_30, var_84);
            // for k in range(dofnum2):                                                           <L 322>
            var_87 = wp::range(var_45);
            start_for_4:;
                if (iter_cmp(var_87) == 0) goto end_for_4;
                var_88 = wp::iter_next(var_87);
                // cdof = cdof_in[worldid, dofj + k]                                              <L 323>
                var_89 = wp::add(var_79, var_88);
                var_90 = wp::address(var_cdof_in, var_0, var_89);
                var_92 = wp::load(var_90);
                var_91 = wp::copy(var_92);
                // cdof_ang = wp.spatial_top(cdof)                                                <L 324>
                var_93 = wp::spatial_top(var_91);
                // cdof_lin = wp.spatial_bottom(cdof)                                             <L 325>
                var_94 = wp::spatial_bottom(var_91);
                // jacp2 = cdof_lin + wp.cross(cdof_ang, offset2)                                 <L 326>
                var_95 = wp::cross(var_93, var_86);
                var_96 = wp::add(var_94, var_95);
                // vel += wp.dot(jacp2, edge) * qvel_in[worldid, dofj + k]                        <L 327>
                var_97 = wp::dot(var_96, var_33);
                var_98 = wp::add(var_79, var_88);
                var_99 = wp::address(var_qvel_in, var_0, var_98);
                var_101 = wp::load(var_99);
                var_100 = wp::mul(var_97, var_101);
                var_102 = wp::add(var_48, var_100);
                wp::assign(var_48, var_102);
                wp::assign(var_64, var_91);
                wp::assign(var_66, var_93);
                wp::assign(var_67, var_94);
                goto start_for_4;
            end_for_4:;
        }
        var_103 = wp::where(var_77, var_88, var_61);
        // flexedge_velocity_out[worldid, edgeid] = vel                                           <L 328>
        wp::array_store(var_flexedge_velocity_out, var_0, var_1, var_48);
        // rowadr = flexedge_J_rowadr[edgeid]                                                     <L 330>
        var_104 = wp::address(var_flexedge_J_rowadr, var_1);
        var_106 = wp::load(var_104);
        var_105 = wp::copy(var_106);
        // nnz_offset = 0                                                                         <L 331>
        // if dofnum1 > 0:                                                                        <L 334>
        var_109 = (var_42 > var_108);
        if (var_109) {
            // dofi = body_dofadr[b1]                                                             <L 335>
            var_110 = wp::address(var_body_dofadr, var_36);
            var_112 = wp::load(var_110);
            var_111 = wp::copy(var_112);
            // offset1 = pos1 - wp.vec3(subtree_com_in[worldid, body_rootid[b1]])                 <L 336>
            var_113 = wp::address(var_body_rootid, var_36);
            var_115 = wp::load(var_113);
            var_114 = wp::address(var_subtree_com_in, var_0, var_115);
            var_117 = wp::load(var_114);
            var_116 = wp::vec_t<3, wp::float32>(var_117);
            var_118 = wp::sub(var_27, var_116);
            // for k in range(dofnum1):                                                           <L 337>
            var_119 = wp::range(var_42);
            start_for_6:;
                if (iter_cmp(var_119) == 0) goto end_for_6;
                var_120 = wp::iter_next(var_119);
                // cdof = cdof_in[worldid, dofi + k]                                              <L 338>
                var_121 = wp::add(var_111, var_120);
                var_122 = wp::address(var_cdof_in, var_0, var_121);
                var_124 = wp::load(var_122);
                var_123 = wp::copy(var_124);
                // cdof_ang = wp.spatial_top(cdof)                                                <L 339>
                var_125 = wp::spatial_top(var_123);
                // cdof_lin = wp.spatial_bottom(cdof)                                             <L 340>
                var_126 = wp::spatial_bottom(var_123);
                // jacp1 = cdof_lin + wp.cross(cdof_ang, offset1)                                 <L 341>
                var_127 = wp::cross(var_125, var_118);
                var_128 = wp::add(var_126, var_127);
                // flexedge_J_out[worldid, rowadr + nnz_offset + k] = wp.dot(-jacp1, edge)        <L 342>
                var_129 = wp::neg(var_128);
                var_130 = wp::dot(var_129, var_33);
                var_131 = wp::add(var_105, var_107);
                var_132 = wp::add(var_131, var_120);
                wp::array_store(var_flexedge_J_out, var_0, var_132, var_130);
                wp::assign(var_64, var_123);
                wp::assign(var_66, var_125);
                wp::assign(var_67, var_126);
                wp::assign(var_69, var_128);
                goto start_for_6;
            end_for_6:;
            // nnz_offset += dofnum1                                                              <L 343>
            var_133 = wp::add(var_107, var_42);
        }
        var_134 = wp::where(var_109, var_111, var_52);
        var_135 = wp::where(var_109, var_118, var_59);
        var_136 = wp::where(var_109, var_120, var_103);
        var_137 = wp::where(var_109, var_133, var_107);
        // if dofnum2 > 0:                                                                        <L 346>
        var_139 = (var_45 > var_138);
        if (var_139) {
            // dofj = body_dofadr[b2]                                                             <L 347>
            var_140 = wp::address(var_body_dofadr, var_39);
            var_142 = wp::load(var_140);
            var_141 = wp::copy(var_142);
            // offset2 = pos2 - wp.vec3(subtree_com_in[worldid, body_rootid[b2]])                 <L 348>
            var_143 = wp::address(var_body_rootid, var_39);
            var_145 = wp::load(var_143);
            var_144 = wp::address(var_subtree_com_in, var_0, var_145);
            var_147 = wp::load(var_144);
            var_146 = wp::vec_t<3, wp::float32>(var_147);
            var_148 = wp::sub(var_30, var_146);
            // for k in range(dofnum2):                                                           <L 349>
            var_149 = wp::range(var_45);
            start_for_8:;
                if (iter_cmp(var_149) == 0) goto end_for_8;
                var_150 = wp::iter_next(var_149);
                // cdof = cdof_in[worldid, dofj + k]                                              <L 350>
                var_151 = wp::add(var_141, var_150);
                var_152 = wp::address(var_cdof_in, var_0, var_151);
                var_154 = wp::load(var_152);
                var_153 = wp::copy(var_154);
                // cdof_ang = wp.spatial_top(cdof)                                                <L 351>
                var_155 = wp::spatial_top(var_153);
                // cdof_lin = wp.spatial_bottom(cdof)                                             <L 352>
                var_156 = wp::spatial_bottom(var_153);
                // jacp2 = cdof_lin + wp.cross(cdof_ang, offset2)                                 <L 353>
                var_157 = wp::cross(var_155, var_148);
                var_158 = wp::add(var_156, var_157);
                // flexedge_J_out[worldid, rowadr + nnz_offset + k] = wp.dot(jacp2, edge)         <L 354>
                var_159 = wp::dot(var_158, var_33);
                var_160 = wp::add(var_105, var_137);
                var_161 = wp::add(var_160, var_150);
                wp::array_store(var_flexedge_J_out, var_0, var_161, var_159);
                wp::assign(var_64, var_153);
                wp::assign(var_66, var_155);
                wp::assign(var_67, var_156);
                wp::assign(var_96, var_158);
                goto start_for_8;
            end_for_8:;
        }
        var_162 = wp::where(var_139, var_150, var_136);
        var_163 = wp::where(var_139, var_141, var_79);
        var_164 = wp::where(var_139, var_148, var_86);
    }
}



extern "C" __global__ void _tendon_bias_coef_2556dfce_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::float32> var_tendon_armature,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::float32> var_ten_Jdot_in,
    wp::array_t<wp::float32> var_ten_bias_coef_out)
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
        const wp::float32 var_11 = 0.0;
        bool var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        bool var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::float32* var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        const wp::float32 var_24 = 0.0;
        bool var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::slice_t var_29;
        const wp::int32 var_30 = 0;
        wp::array_t<wp::float32> var_31;
        wp::float32* var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        //---------
        // forward
        // def _tendon_bias_coef(                                                                 <L 1810>
        // worldid, tenid, dofid_sparse = wp.tid()                                                <L 1823>
        builtin_tid3d(var_0, var_1, var_2);
        // armature = tendon_armature[worldid % tendon_armature.shape[0], tenid]                  <L 1825>
        var_3 = &(var_tendon_armature.shape);
        var_6 = wp::load(var_3);
        var_5 = wp::extract(var_6, var_4);
        var_7 = wp::mod(var_0, var_5);
        var_8 = wp::address(var_tendon_armature, var_7, var_1);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // if armature == 0.0:                                                                    <L 1826>
        var_12 = (var_9 == var_11);
        if (var_12) {
            // return                                                                             <L 1827>
            continue;
        }
        // rownnz = ten_J_rownnz[tenid]                                                           <L 1829>
        var_13 = wp::address(var_ten_J_rownnz, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // if dofid_sparse >= rownnz:                                                             <L 1830>
        var_16 = (var_2 >= var_14);
        if (var_16) {
            // return                                                                             <L 1831>
            continue;
        }
        // rowadr = ten_J_rowadr[tenid]                                                           <L 1832>
        var_17 = wp::address(var_ten_J_rowadr, var_1);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // sparseid = rowadr + dofid_sparse                                                       <L 1833>
        var_20 = wp::add(var_18, var_2);
        // ten_Jdot = ten_Jdot_in[worldid, sparseid]                                              <L 1834>
        var_21 = wp::address(var_ten_Jdot_in, var_0, var_20);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // if ten_Jdot == 0.0:                                                                    <L 1835>
        var_25 = (var_22 == var_24);
        if (var_25) {
            // return                                                                             <L 1836>
            continue;
        }
        // dofid = ten_J_colind[sparseid]                                                         <L 1838>
        var_26 = wp::address(var_ten_J_colind, var_20);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // wp.atomic_add(ten_bias_coef_out[worldid], tenid, ten_Jdot * qvel_in[worldid, dofid])       <L 1839>
        var_29 = wp::slice_t(var_0, var_0, var_30);
        var_31 = wp::view(var_ten_bias_coef_out, var_29);
        var_32 = wp::address(var_qvel_in, var_0, var_27);
        var_34 = wp::load(var_32);
        var_33 = wp::mul(var_22, var_34);
        var_35 = wp::atomic_add(var_31, var_1, var_33);
    }
}



extern "C" __global__ void _tendon_armature_a3b217d6_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_dof_parentid,
    wp::array_t<wp::int32> var_dof_Madr,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::float32> var_tendon_armature,
    bool var_is_sparse,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::array_t<wp::float32> var_qM_out)
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
        const wp::float32 var_11 = 0.0;
        bool var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        bool var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        wp::float32* var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        const wp::float32 var_28 = 0.0;
        bool var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        const wp::int32 var_35 = 0;
        bool var_36;
        bool var_37;
        wp::float32 var_38;
        const wp::int32 var_39 = 0;
        bool var_40;
        wp::int32 var_41;
        wp::int32* var_42;
        bool var_43;
        wp::int32 var_44;
        wp::int32 var_45;
        const wp::int32 var_46 = 1;
        wp::int32 var_47;
        const wp::int32 var_48 = 0;
        bool var_49;
        wp::int32* var_50;
        bool var_51;
        wp::int32 var_52;
        bool var_53;
        wp::float32* var_54;
        wp::float32 var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        const wp::float32 var_58 = 0.0;
        wp::float32 var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::float32 var_62;
        wp::float32 var_63;
        const wp::int32 var_64 = 0;
        wp::slice_t var_65;
        const wp::int32 var_66 = 0;
        wp::slice_t var_67;
        const wp::int32 var_68 = 0;
        wp::array_t<wp::float32> var_69;
        wp::float32 var_70;
        const wp::int32 var_71 = 1;
        wp::int32 var_72;
        wp::int32 var_73;
        wp::slice_t var_74;
        const wp::int32 var_75 = 0;
        wp::slice_t var_76;
        const wp::int32 var_77 = 0;
        wp::array_t<wp::float32> var_78;
        wp::float32 var_79;
        bool var_80;
        wp::slice_t var_81;
        const wp::int32 var_82 = 0;
        wp::slice_t var_83;
        const wp::int32 var_84 = 0;
        wp::array_t<wp::float32> var_85;
        wp::float32 var_86;
        wp::int32* var_87;
        wp::int32 var_88;
        wp::int32 var_89;
        //---------
        // forward
        // def _tendon_armature(                                                                  <L 916>
        // worldid, tenid, dofid = wp.tid()                                                       <L 930>
        builtin_tid3d(var_0, var_1, var_2);
        // armature = tendon_armature[worldid % tendon_armature.shape[0], tenid]                  <L 932>
        var_3 = &(var_tendon_armature.shape);
        var_6 = wp::load(var_3);
        var_5 = wp::extract(var_6, var_4);
        var_7 = wp::mod(var_0, var_5);
        var_8 = wp::address(var_tendon_armature, var_7, var_1);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // if armature == 0.0:                                                                    <L 934>
        var_12 = (var_9 == var_11);
        if (var_12) {
            // return                                                                             <L 935>
            continue;
        }
        // rownnz = ten_J_rownnz[tenid]                                                           <L 937>
        var_13 = wp::address(var_ten_J_rownnz, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // if dofid >= rownnz:                                                                    <L 938>
        var_16 = (var_2 >= var_14);
        if (var_16) {
            // return                                                                             <L 939>
            continue;
        }
        // rowadr = ten_J_rowadr[tenid]                                                           <L 940>
        var_17 = wp::address(var_ten_J_rowadr, var_1);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // dofid_sparse = dofid                                                                   <L 941>
        var_20 = wp::copy(var_2);
        // sparseid = rowadr + dofid_sparse                                                       <L 942>
        var_21 = wp::add(var_18, var_20);
        // dofid = ten_J_colind[sparseid]                                                         <L 943>
        var_22 = wp::address(var_ten_J_colind, var_21);
        var_24 = wp::load(var_22);
        var_23 = wp::copy(var_24);
        // ten_Ji = ten_J_in[worldid, sparseid]                                                   <L 944>
        var_25 = wp::address(var_ten_J_in, var_0, var_21);
        var_27 = wp::load(var_25);
        var_26 = wp::copy(var_27);
        // if ten_Ji == 0.0:                                                                      <L 946>
        var_29 = (var_26 == var_28);
        if (var_29) {
            // return                                                                             <L 947>
            continue;
        }
        // if is_sparse:                                                                          <L 949>
        if (var_is_sparse) {
            // madr_ij = dof_Madr[dofid]                                                          <L 950>
            var_30 = wp::address(var_dof_Madr, var_23);
            var_32 = wp::load(var_30);
            var_31 = wp::copy(var_32);
        }
        // dofidi = dofid                                                                         <L 953>
        var_33 = wp::copy(var_23);
        // ptr = dofid_sparse                                                                     <L 954>
        var_34 = wp::copy(var_20);
        // while dofid >= 0:                                                                      <L 955>
        start_while_3:;
        var_36 = (var_23 >= var_35);
        if ((var_36) == false) goto end_while_3;
            // if dofid == dofidi:                                                                <L 956>
            var_37 = (var_23 == var_33);
            if (var_37) {
                // ten_Jj = ten_Ji                                                                <L 957>
                var_38 = wp::copy(var_26);
            }
            if (!var_37) {
                // while ptr >= 0:                                                                <L 960>
        start_while_5:;
                var_40 = (var_34 >= var_39);
        if ((var_40) == false) goto end_while_5;
                    // sparseid = rowadr + ptr                                                    <L 961>
                    var_41 = wp::add(var_18, var_34);
                    // if ten_J_colind[sparseid] <= dofid:                                        <L 962>
                    var_42 = wp::address(var_ten_J_colind, var_41);
                    var_44 = wp::load(var_42);
                    var_43 = (var_44 <= var_23);
                    if (var_43) {
                        // break                                                                  <L 963>
                        wp::assign(var_21, var_41);
                        goto end_while_5;
                    }
                    var_45 = wp::where(var_43, var_21, var_41);
                    // ptr -= 1                                                                   <L 964>
                    var_47 = wp::sub(var_34, var_46);
                    wp::assign(var_21, var_45);
                    wp::assign(var_34, var_47);
        goto start_while_5;
        end_while_5:;
                // if ptr >= 0 and ten_J_colind[sparseid] == dofid:                               <L 965>
                var_49 = (var_34 >= var_48);
                var_50 = wp::address(var_ten_J_colind, var_21);
                var_52 = wp::load(var_50);
                var_51 = (var_52 == var_23);
                var_53 = var_49 && var_51;
                if (var_53) {
                    // ten_Jj = ten_J_in[worldid, sparseid]                                       <L 966>
                    var_54 = wp::address(var_ten_J_in, var_0, var_21);
                    var_56 = wp::load(var_54);
                    var_55 = wp::copy(var_56);
                }
                var_57 = wp::where(var_53, var_55, var_38);
                if (!var_53) {
                    // ten_Jj = float(0.0)                                                        <L 968>
                    var_59 = wp::float(var_58);
                }
                var_60 = wp::where(var_53, var_57, var_59);
            }
            var_61 = wp::where(var_37, var_38, var_60);
            // qMij = armature * ten_Jj * ten_Ji                                                  <L 970>
            var_62 = wp::mul(var_9, var_61);
            var_63 = wp::mul(var_62, var_26);
            // if is_sparse:                                                                      <L 972>
            if (var_is_sparse) {
                // wp.atomic_add(qM_out[worldid, 0], madr_ij, qMij)                               <L 973>
                var_65 = wp::slice_t(var_0, var_0, var_66);
                var_67 = wp::slice_t(var_64, var_64, var_68);
                var_69 = wp::view(var_qM_out, var_65, var_67);
                var_70 = wp::atomic_add(var_69, var_31, var_63);
                // madr_ij += 1                                                                   <L 974>
                var_72 = wp::add(var_31, var_71);
            }
            var_73 = wp::where(var_is_sparse, var_72, var_31);
            if (!var_is_sparse) {
                // wp.atomic_add(qM_out[worldid, dofidi], dofid, qMij)                            <L 976>
                var_74 = wp::slice_t(var_0, var_0, var_75);
                var_76 = wp::slice_t(var_33, var_33, var_77);
                var_78 = wp::view(var_qM_out, var_74, var_76);
                var_79 = wp::atomic_add(var_78, var_23, var_63);
                // if dofidi != dofid:                                                            <L 977>
                var_80 = (var_33 != var_23);
                if (var_80) {
                    // wp.atomic_add(qM_out[worldid, dofid], dofidi, qMij)                        <L 978>
                    var_81 = wp::slice_t(var_0, var_0, var_82);
                    var_83 = wp::slice_t(var_23, var_23, var_84);
                    var_85 = wp::view(var_qM_out, var_81, var_83);
                    var_86 = wp::atomic_add(var_85, var_33, var_63);
                }
            }
            // dofid = dof_parentid[dofid]                                                        <L 980>
            var_87 = wp::address(var_dof_parentid, var_23);
            var_89 = wp::load(var_87);
            var_88 = wp::copy(var_89);
            wp::assign(var_23, var_88);
            wp::assign(var_31, var_73);
        goto start_while_3;
        end_while_3:;
    }
}

