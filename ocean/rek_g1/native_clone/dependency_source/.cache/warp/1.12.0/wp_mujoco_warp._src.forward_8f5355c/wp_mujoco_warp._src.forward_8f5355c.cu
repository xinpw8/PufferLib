
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:188
static CUDA_CALLABLE wp::quat_t<wp::float32> quat_integrate_0(
    wp::quat_t<wp::float32> var_q,
    wp::vec_t<3, wp::float32> var_v,
    wp::float32 var_dt)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::float32 var_2;
    wp::quat_t<wp::float32> var_3;
    wp::quat_t<wp::float32> var_4;
    wp::quat_t<wp::float32> var_5;
    wp::quat_t<wp::float32> var_6;
    //---------
    // forward
    // def quat_integrate(q: wp.quat, v: wp.vec3, dt: float) -> wp.quat:                      <L 189>
    // norm_ = wp.length(v)                                                                   <L 191>
    var_0 = wp::length(var_v);
    // v = wp.normalize(v)  # does that need proper zero gradient handling?                   <L 192>
    var_1 = wp::normalize(var_v);
    // angle = dt * norm_                                                                     <L 193>
    var_2 = wp::mul(var_dt, var_0);
    // q_res = axis_angle_to_quat(v, angle)                                                   <L 195>
    var_3 = axis_angle_to_quat_0(var_1, var_2);
    // q = wp.normalize(q)                                                                    <L 196>
    var_4 = wp::normalize(var_q);
    // q_res = mul_quat(q, q_res)                                                             <L 197>
    var_5 = mul_quat_0(var_4, var_3);
    // return wp.normalize(q_res)                                                             <L 199>
    var_6 = wp::normalize(var_5);
    return var_6;
}


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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/util_misc.py:553
static CUDA_CALLABLE wp::float32 _sigmoid_0(
    wp::float32 var_x)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    bool var_1;
    const wp::float32 var_2 = 0.0;
    const wp::float32 var_3 = 1.0;
    bool var_4;
    const wp::float32 var_5 = 1.0;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::float32 var_8 = 3.0;
    wp::float32 var_9;
    const wp::float32 var_10 = 2.0;
    wp::float32 var_11;
    const wp::float32 var_12 = 5.0;
    wp::float32 var_13;
    wp::float32 var_14;
    const wp::float32 var_15 = 10.0;
    wp::float32 var_16;
    wp::float32 var_17;
    //---------
    // forward
    // def _sigmoid(x: float) -> float:                                                       <L 554>
    // if x <= 0.0:                                                                           <L 556>
    var_1 = (var_x <= var_0);
    if (var_1) {
        // return 0.0                                                                         <L 557>
        return var_2;
    }
    // if x >= 1.0:                                                                           <L 559>
    var_4 = (var_x >= var_3);
    if (var_4) {
        // return 1.0                                                                         <L 560>
        return var_5;
    }
    // return x * x * x * (3.0 * x * (2.0 * x - 5.0) + 10.0)                                  <L 564>
    var_6 = wp::mul(var_x, var_x);
    var_7 = wp::mul(var_6, var_x);
    var_9 = wp::mul(var_8, var_x);
    var_11 = wp::mul(var_10, var_x);
    var_13 = wp::sub(var_11, var_12);
    var_14 = wp::mul(var_9, var_13);
    var_16 = wp::add(var_14, var_15);
    var_17 = wp::mul(var_7, var_16);
    return var_17;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/util_misc.py:567
static CUDA_CALLABLE wp::float32 muscle_dynamics_timescale_0(
    wp::float32 var_dctrl,
    wp::float32 var_tau_act,
    wp::float32 var_tau_deact,
    wp::float32 var_smooth_width)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 1e-15;
    bool var_1;
    const wp::float32 var_2 = 0.0;
    bool var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    const wp::float32 var_6 = 0.5;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    //---------
    // forward
    // def muscle_dynamics_timescale(dctrl: float, tau_act: float, tau_deact: float, smooth_width: float) -> float:       <L 568>
    // if smooth_width < MJ_MINVAL:                                                           <L 571>
    var_1 = (var_smooth_width < var_0);
    if (var_1) {
        // if dctrl > 0.0:                                                                    <L 572>
        var_3 = (var_dctrl > var_2);
        if (var_3) {
            // return tau_act                                                                 <L 573>
            return var_tau_act;
        }
        if (!var_3) {
            // return tau_deact                                                               <L 575>
            return var_tau_deact;
        }
    }
    if (!var_1) {
        // return tau_deact + (tau_act - tau_deact) * _sigmoid(dctrl / smooth_width + 0.5)       <L 578>
        var_4 = wp::sub(var_tau_act, var_tau_deact);
        var_5 = wp::div(var_dctrl, var_smooth_width);
        var_7 = wp::add(var_5, var_6);
        var_8 = _sigmoid_0(var_7);
        var_9 = wp::mul(var_4, var_8);
        var_10 = wp::add(var_tau_deact, var_9);
        return var_10;
    }
    return {};
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/util_misc.py:581
static CUDA_CALLABLE wp::float32 muscle_dynamics_0(
    wp::float32 var_control,
    wp::float32 var_activation,
    wp::vec_t<10, wp::float32> var_prm)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    const wp::float32 var_1 = 1.0;
    wp::float32 var_2;
    const wp::float32 var_3 = 0.0;
    const wp::float32 var_4 = 1.0;
    wp::float32 var_5;
    const wp::int32 var_6 = 0;
    wp::float32 var_7;
    const wp::float32 var_8 = 0.5;
    const wp::float32 var_9 = 1.5;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    const wp::int32 var_13 = 1;
    wp::float32 var_14;
    const wp::float32 var_15 = 0.5;
    const wp::float32 var_16 = 1.5;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    const wp::int32 var_20 = 2;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    const wp::float32 var_24 = 1e-15;
    wp::float32 var_25;
    wp::float32 var_26;
    //---------
    // forward
    // def muscle_dynamics(control: float, activation: float, prm: vec10) -> float:           <L 582>
    // ctrlclamp = wp.clamp(control, 0.0, 1.0)                                                <L 585>
    var_2 = wp::clamp(var_control, var_0, var_1);
    // actclamp = wp.clamp(activation, 0.0, 1.0)                                              <L 588>
    var_5 = wp::clamp(var_activation, var_3, var_4);
    // tau_act = prm[0] * (0.5 + 1.5 * actclamp)  # activation timescale                      <L 591>
    var_7 = wp::extract(var_prm, var_6);
    var_10 = wp::mul(var_9, var_5);
    var_11 = wp::add(var_8, var_10);
    var_12 = wp::mul(var_7, var_11);
    // tau_deact = prm[1] / (0.5 + 1.5 * actclamp)  # deactivation timescale                  <L 592>
    var_14 = wp::extract(var_prm, var_13);
    var_17 = wp::mul(var_16, var_5);
    var_18 = wp::add(var_15, var_17);
    var_19 = wp::div(var_14, var_18);
    // smooth_width = prm[2]  # width of smoothing sigmoid                                    <L 593>
    var_21 = wp::extract(var_prm, var_20);
    // dctrl = ctrlclamp - activation  # excess excitation                                    <L 594>
    var_22 = wp::sub(var_2, var_activation);
    // tau = muscle_dynamics_timescale(dctrl, tau_act, tau_deact, smooth_width)               <L 596>
    var_23 = muscle_dynamics_timescale_0(var_22, var_12, var_19, var_21);
    // return dctrl / wp.max(MJ_MINVAL, tau)                                                  <L 599>
    var_25 = wp::max(var_24, var_23);
    var_26 = wp::div(var_22, var_25);
    return var_26;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/util_misc.py:453
static CUDA_CALLABLE wp::float32 muscle_gain_length_0(
    wp::float32 var_length,
    wp::float32 var_lmin,
    wp::float32 var_lmax)
{
    //---------
    // primal vars
    bool var_0;
    bool var_1;
    bool var_2;
    const wp::float32 var_3 = 0.0;
    const wp::float32 var_4 = 0.5;
    const wp::float32 var_5 = 1.0;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::float32 var_8 = 0.5;
    const wp::float32 var_9 = 1.0;
    wp::float32 var_10;
    wp::float32 var_11;
    bool var_12;
    wp::float32 var_13;
    const wp::float32 var_14 = 1e-15;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    const wp::float32 var_18 = 0.5;
    wp::float32 var_19;
    wp::float32 var_20;
    const wp::float32 var_21 = 1.0;
    bool var_22;
    const wp::float32 var_23 = 1.0;
    wp::float32 var_24;
    const wp::float32 var_25 = 1.0;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    const wp::float32 var_29 = 1.0;
    const wp::float32 var_30 = 0.5;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    bool var_35;
    const wp::float32 var_36 = 1.0;
    wp::float32 var_37;
    const wp::float32 var_38 = 1.0;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    const wp::float32 var_42 = 1.0;
    const wp::float32 var_43 = 0.5;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    wp::float32 var_47;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    const wp::float32 var_52 = 0.5;
    wp::float32 var_53;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::float32 var_56;
    wp::float32 var_57;
    //---------
    // forward
    // def muscle_gain_length(length: float, lmin: float, lmax: float) -> float:              <L 454>
    // if (lmin > length) or (length > lmax):                                                 <L 456>
    var_0 = (var_lmin > var_length);
    var_1 = (var_length > var_lmax);
    var_2 = var_0 || var_1;
    if (var_2) {
        // return 0.0                                                                         <L 457>
        return var_3;
    }
    // a = 0.5 * (lmin + 1.0)                                                                 <L 460>
    var_6 = wp::add(var_lmin, var_5);
    var_7 = wp::mul(var_4, var_6);
    // b = 0.5 * (1.0 + lmax)                                                                 <L 461>
    var_10 = wp::add(var_9, var_lmax);
    var_11 = wp::mul(var_8, var_10);
    // if length <= a:                                                                        <L 463>
    var_12 = (var_length <= var_7);
    if (var_12) {
        // x = (length - lmin) / wp.max(MJ_MINVAL, a - lmin)                                  <L 464>
        var_13 = wp::sub(var_length, var_lmin);
        var_15 = wp::sub(var_7, var_lmin);
        var_16 = wp::max(var_14, var_15);
        var_17 = wp::div(var_13, var_16);
        // return 0.5 * x * x                                                                 <L 465>
        var_19 = wp::mul(var_18, var_17);
        var_20 = wp::mul(var_19, var_17);
        return var_20;
    }
    if (!var_12) {
        // elif length <= 1.0:                                                                <L 466>
        var_22 = (var_length <= var_21);
        if (var_22) {
            // x = (1.0 - length) / wp.max(MJ_MINVAL, 1.0 - a)                                <L 467>
            var_24 = wp::sub(var_23, var_length);
            var_26 = wp::sub(var_25, var_7);
            var_27 = wp::max(var_14, var_26);
            var_28 = wp::div(var_24, var_27);
            // return 1.0 - 0.5 * x * x                                                       <L 468>
            var_31 = wp::mul(var_30, var_28);
            var_32 = wp::mul(var_31, var_28);
            var_33 = wp::sub(var_29, var_32);
            return var_33;
        }
        var_34 = wp::where(var_22, var_28, var_17);
        if (!var_22) {
            // elif length <= b:                                                              <L 469>
            var_35 = (var_length <= var_11);
            if (var_35) {
                // x = (length - 1.0) / wp.max(MJ_MINVAL, b - 1.0)                            <L 470>
                var_37 = wp::sub(var_length, var_36);
                var_39 = wp::sub(var_11, var_38);
                var_40 = wp::max(var_14, var_39);
                var_41 = wp::div(var_37, var_40);
                // return 1.0 - 0.5 * x * x                                                   <L 471>
                var_44 = wp::mul(var_43, var_41);
                var_45 = wp::mul(var_44, var_41);
                var_46 = wp::sub(var_42, var_45);
                return var_46;
            }
            var_47 = wp::where(var_35, var_41, var_34);
            if (!var_35) {
                // x = (lmax - length) / wp.max(MJ_MINVAL, lmax - b)                          <L 473>
                var_48 = wp::sub(var_lmax, var_length);
                var_49 = wp::sub(var_lmax, var_11);
                var_50 = wp::max(var_14, var_49);
                var_51 = wp::div(var_48, var_50);
                // return 0.5 * x * x                                                         <L 474>
                var_53 = wp::mul(var_52, var_51);
                var_54 = wp::mul(var_53, var_51);
                return var_54;
            }
            var_55 = wp::where(var_35, var_47, var_51);
        }
        var_56 = wp::where(var_22, var_34, var_55);
    }
    var_57 = wp::where(var_12, var_17, var_56);
    return {};
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/util_misc.py:477
static CUDA_CALLABLE wp::float32 muscle_gain_0(
    wp::float32 var_len,
    wp::float32 var_vel,
    wp::vec_t<2, wp::float32> var_lengthrange,
    wp::float32 var_acc0,
    wp::vec_t<10, wp::float32> var_prm)
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
    const wp::int32 var_9 = 4;
    wp::float32 var_10;
    const wp::int32 var_11 = 5;
    wp::float32 var_12;
    const wp::int32 var_13 = 6;
    wp::float32 var_14;
    const wp::int32 var_15 = 8;
    wp::float32 var_16;
    const wp::float32 var_17 = 0.0;
    bool var_18;
    const wp::float32 var_19 = 1e-15;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::int32 var_23 = 1;
    wp::float32 var_24;
    const wp::int32 var_25 = 0;
    wp::float32 var_26;
    wp::float32 var_27;
    const wp::int32 var_28 = 1;
    wp::float32 var_29;
    const wp::int32 var_30 = 0;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    const wp::int32 var_35 = 0;
    wp::float32 var_36;
    const wp::int32 var_37 = 0;
    wp::float32 var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    const wp::float32 var_47 = 1.0;
    wp::float32 var_48;
    const wp::float32 var_49 = 1.0;
    const wp::float32 var_50 = -1.0;
    bool var_51;
    const wp::float32 var_52 = 0.0;
    const wp::float32 var_53 = 0.0;
    bool var_54;
    const wp::float32 var_55 = 1.0;
    wp::float32 var_56;
    const wp::float32 var_57 = 1.0;
    wp::float32 var_58;
    wp::float32 var_59;
    wp::float32 var_60;
    bool var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    wp::float32 var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    wp::float32 var_75;
    //---------
    // forward
    // def muscle_gain(len: float, vel: float, lengthrange: wp.vec2, acc0: float, prm: vec10) -> float:       <L 478>
    // range_ = wp.vec2(prm[0], prm[1])                                                       <L 481>
    var_1 = wp::extract(var_prm, var_0);
    var_3 = wp::extract(var_prm, var_2);
    var_4 = wp::vec_t<2, wp::float32>(var_1, var_3);
    // force = prm[2]                                                                         <L 482>
    var_6 = wp::extract(var_prm, var_5);
    // scale = prm[3]                                                                         <L 483>
    var_8 = wp::extract(var_prm, var_7);
    // lmin = prm[4]                                                                          <L 484>
    var_10 = wp::extract(var_prm, var_9);
    // lmax = prm[5]                                                                          <L 485>
    var_12 = wp::extract(var_prm, var_11);
    // vmax = prm[6]                                                                          <L 486>
    var_14 = wp::extract(var_prm, var_13);
    // fvmax = prm[8]                                                                         <L 487>
    var_16 = wp::extract(var_prm, var_15);
    // if force < 0.0:                                                                        <L 490>
    var_18 = (var_6 < var_17);
    if (var_18) {
        // force = scale / wp.max(MJ_MINVAL, acc0)                                            <L 491>
        var_20 = wp::max(var_19, var_acc0);
        var_21 = wp::div(var_8, var_20);
    }
    var_22 = wp::where(var_18, var_21, var_6);
    // L0 = (lengthrange[1] - lengthrange[0]) / wp.max(MJ_MINVAL, range_[1] - range_[0])       <L 494>
    var_24 = wp::extract(var_lengthrange, var_23);
    var_26 = wp::extract(var_lengthrange, var_25);
    var_27 = wp::sub(var_24, var_26);
    var_29 = wp::extract(var_4, var_28);
    var_31 = wp::extract(var_4, var_30);
    var_32 = wp::sub(var_29, var_31);
    var_33 = wp::max(var_19, var_32);
    var_34 = wp::div(var_27, var_33);
    // L = range_[0] + (len - lengthrange[0]) / wp.max(MJ_MINVAL, L0)                         <L 497>
    var_36 = wp::extract(var_4, var_35);
    var_38 = wp::extract(var_lengthrange, var_37);
    var_39 = wp::sub(var_len, var_38);
    var_40 = wp::max(var_19, var_34);
    var_41 = wp::div(var_39, var_40);
    var_42 = wp::add(var_36, var_41);
    // V = vel / wp.max(MJ_MINVAL, L0 * vmax)                                                 <L 498>
    var_43 = wp::mul(var_34, var_14);
    var_44 = wp::max(var_19, var_43);
    var_45 = wp::div(var_vel, var_44);
    // FL = muscle_gain_length(L, lmin, lmax)                                                 <L 501>
    var_46 = muscle_gain_length_0(var_42, var_10, var_12);
    // y = fvmax - 1.0                                                                        <L 504>
    var_48 = wp::sub(var_16, var_47);
    // if V <= -1.0:                                                                          <L 505>
    var_51 = (var_45 <= var_50);
    if (var_51) {
        // FV = 0.0                                                                           <L 506>
    }
    if (!var_51) {
        // elif V <= 0.0:                                                                     <L 507>
        var_54 = (var_45 <= var_53);
        if (var_54) {
            // FV = (V + 1.0) * (V + 1.0)                                                     <L 508>
            var_56 = wp::add(var_45, var_55);
            var_58 = wp::add(var_45, var_57);
            var_59 = wp::mul(var_56, var_58);
        }
        var_60 = wp::where(var_54, var_59, var_52);
        if (!var_54) {
            // elif V <= y:                                                                   <L 509>
            var_61 = (var_45 <= var_48);
            if (var_61) {
                // FV = fvmax - (y - V) * (y - V) / wp.max(MJ_MINVAL, y)                      <L 510>
                var_62 = wp::sub(var_48, var_45);
                var_63 = wp::sub(var_48, var_45);
                var_64 = wp::mul(var_62, var_63);
                var_65 = wp::max(var_19, var_48);
                var_66 = wp::div(var_64, var_65);
                var_67 = wp::sub(var_16, var_66);
            }
            var_68 = wp::where(var_61, var_67, var_60);
            if (!var_61) {
                // FV = fvmax                                                                 <L 512>
                var_69 = wp::copy(var_16);
            }
            var_70 = wp::where(var_61, var_68, var_69);
        }
        var_71 = wp::where(var_54, var_60, var_70);
    }
    var_72 = wp::where(var_51, var_52, var_71);
    // return -force * FL * FV                                                                <L 515>
    var_73 = wp::neg(var_22);
    var_74 = wp::mul(var_73, var_46);
    var_75 = wp::mul(var_74, var_72);
    return var_75;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/util_misc.py:518
static CUDA_CALLABLE wp::float32 muscle_bias_0(
    wp::float32 var_len,
    wp::vec_t<2, wp::float32> var_lengthrange,
    wp::float32 var_acc0,
    wp::vec_t<10, wp::float32> var_prm)
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
    const wp::int32 var_9 = 5;
    wp::float32 var_10;
    const wp::int32 var_11 = 7;
    wp::float32 var_12;
    const wp::float32 var_13 = 0.0;
    bool var_14;
    const wp::float32 var_15 = 1e-15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    const wp::int32 var_19 = 1;
    wp::float32 var_20;
    const wp::int32 var_21 = 0;
    wp::float32 var_22;
    wp::float32 var_23;
    const wp::int32 var_24 = 1;
    wp::float32 var_25;
    const wp::int32 var_26 = 0;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    const wp::int32 var_31 = 0;
    wp::float32 var_32;
    const wp::int32 var_33 = 0;
    wp::float32 var_34;
    wp::float32 var_35;
    wp::float32 var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    const wp::float32 var_39 = 0.5;
    const wp::float32 var_40 = 1.0;
    wp::float32 var_41;
    wp::float32 var_42;
    const wp::float32 var_43 = 1.0;
    bool var_44;
    const wp::float32 var_45 = 0.0;
    bool var_46;
    const wp::float32 var_47 = 1.0;
    wp::float32 var_48;
    const wp::float32 var_49 = 1.0;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    const wp::float32 var_55 = 0.5;
    wp::float32 var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    const wp::float32 var_60 = 1.0;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    const wp::float32 var_66 = 0.5;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    //---------
    // forward
    // def muscle_bias(len: float, lengthrange: wp.vec2, acc0: float, prm: vec10) -> float:       <L 519>
    // range_ = wp.vec2(prm[0], prm[1])                                                       <L 525>
    var_1 = wp::extract(var_prm, var_0);
    var_3 = wp::extract(var_prm, var_2);
    var_4 = wp::vec_t<2, wp::float32>(var_1, var_3);
    // force = prm[2]                                                                         <L 526>
    var_6 = wp::extract(var_prm, var_5);
    // scale = prm[3]                                                                         <L 527>
    var_8 = wp::extract(var_prm, var_7);
    // lmax = prm[5]                                                                          <L 528>
    var_10 = wp::extract(var_prm, var_9);
    // fpmax = prm[7]                                                                         <L 529>
    var_12 = wp::extract(var_prm, var_11);
    // if force < 0.0:                                                                        <L 532>
    var_14 = (var_6 < var_13);
    if (var_14) {
        // force = scale / wp.max(MJ_MINVAL, acc0)                                            <L 533>
        var_16 = wp::max(var_15, var_acc0);
        var_17 = wp::div(var_8, var_16);
    }
    var_18 = wp::where(var_14, var_17, var_6);
    // L0 = (lengthrange[1] - lengthrange[0]) / wp.max(MJ_MINVAL, range_[1] - range_[0])       <L 536>
    var_20 = wp::extract(var_lengthrange, var_19);
    var_22 = wp::extract(var_lengthrange, var_21);
    var_23 = wp::sub(var_20, var_22);
    var_25 = wp::extract(var_4, var_24);
    var_27 = wp::extract(var_4, var_26);
    var_28 = wp::sub(var_25, var_27);
    var_29 = wp::max(var_15, var_28);
    var_30 = wp::div(var_23, var_29);
    // L = range_[0] + (len - lengthrange[0]) / wp.max(MJ_MINVAL, L0)                         <L 539>
    var_32 = wp::extract(var_4, var_31);
    var_34 = wp::extract(var_lengthrange, var_33);
    var_35 = wp::sub(var_len, var_34);
    var_36 = wp::max(var_15, var_30);
    var_37 = wp::div(var_35, var_36);
    var_38 = wp::add(var_32, var_37);
    // b = 0.5 * (1.0 + lmax)                                                                 <L 542>
    var_41 = wp::add(var_40, var_10);
    var_42 = wp::mul(var_39, var_41);
    // if L <= 1.0:                                                                           <L 543>
    var_44 = (var_38 <= var_43);
    if (var_44) {
        // return 0.0                                                                         <L 544>
        return var_45;
    }
    if (!var_44) {
        // elif L <= b:                                                                       <L 545>
        var_46 = (var_38 <= var_42);
        if (var_46) {
            // x = (L - 1.0) / wp.max(MJ_MINVAL, b - 1.0)                                     <L 546>
            var_48 = wp::sub(var_38, var_47);
            var_50 = wp::sub(var_42, var_49);
            var_51 = wp::max(var_15, var_50);
            var_52 = wp::div(var_48, var_51);
            // return -force * fpmax * 0.5 * x * x                                            <L 547>
            var_53 = wp::neg(var_18);
            var_54 = wp::mul(var_53, var_12);
            var_56 = wp::mul(var_54, var_55);
            var_57 = wp::mul(var_56, var_52);
            var_58 = wp::mul(var_57, var_52);
            return var_58;
        }
        if (!var_46) {
            // x = (L - b) / wp.max(MJ_MINVAL, b - 1.0)                                       <L 549>
            var_59 = wp::sub(var_38, var_42);
            var_61 = wp::sub(var_42, var_60);
            var_62 = wp::max(var_15, var_61);
            var_63 = wp::div(var_59, var_62);
            // return -force * fpmax * (0.5 + x)                                              <L 550>
            var_64 = wp::neg(var_18);
            var_65 = wp::mul(var_64, var_12);
            var_67 = wp::add(var_66, var_63);
            var_68 = wp::mul(var_65, var_67);
            return var_68;
        }
        var_69 = wp::where(var_46, var_52, var_63);
    }
    return {};
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:188
static CUDA_CALLABLE void adj_quat_integrate_0(
    wp::quat_t<wp::float32> var_q,
    wp::vec_t<3, wp::float32> var_v,
    wp::float32 var_dt,
    wp::quat_t<wp::float32> & adj_q,
    wp::vec_t<3, wp::float32> & adj_v,
    wp::float32 & adj_dt,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/util_misc.py:553
static CUDA_CALLABLE void adj__sigmoid_0(
    wp::float32 var_x,
    wp::float32 & adj_x,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/util_misc.py:567
static CUDA_CALLABLE void adj_muscle_dynamics_timescale_0(
    wp::float32 var_dctrl,
    wp::float32 var_tau_act,
    wp::float32 var_tau_deact,
    wp::float32 var_smooth_width,
    wp::float32 & adj_dctrl,
    wp::float32 & adj_tau_act,
    wp::float32 & adj_tau_deact,
    wp::float32 & adj_smooth_width,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/util_misc.py:581
static CUDA_CALLABLE void adj_muscle_dynamics_0(
    wp::float32 var_control,
    wp::float32 var_activation,
    wp::vec_t<10, wp::float32> var_prm,
    wp::float32 & adj_control,
    wp::float32 & adj_activation,
    wp::vec_t<10, wp::float32> & adj_prm,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/util_misc.py:453
static CUDA_CALLABLE void adj_muscle_gain_length_0(
    wp::float32 var_length,
    wp::float32 var_lmin,
    wp::float32 var_lmax,
    wp::float32 & adj_length,
    wp::float32 & adj_lmin,
    wp::float32 & adj_lmax,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/util_misc.py:477
static CUDA_CALLABLE void adj_muscle_gain_0(
    wp::float32 var_len,
    wp::float32 var_vel,
    wp::vec_t<2, wp::float32> var_lengthrange,
    wp::float32 var_acc0,
    wp::vec_t<10, wp::float32> var_prm,
    wp::float32 & adj_len,
    wp::float32 & adj_vel,
    wp::vec_t<2, wp::float32> & adj_lengthrange,
    wp::float32 & adj_acc0,
    wp::vec_t<10, wp::float32> & adj_prm,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/util_misc.py:518
static CUDA_CALLABLE void adj_muscle_bias_0(
    wp::float32 var_len,
    wp::vec_t<2, wp::float32> var_lengthrange,
    wp::float32 var_acc0,
    wp::vec_t<10, wp::float32> var_prm,
    wp::float32 & adj_len,
    wp::vec_t<2, wp::float32> & adj_lengthrange,
    wp::float32 & adj_acc0,
    wp::vec_t<10, wp::float32> & adj_prm,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void _next_position_1c101e8d_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_jnt_qposadr,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::float32> var_qpos_in,
    wp::array_t<wp::float32> var_qvel_in,
    wp::float32 var_qvel_scale_in,
    wp::array_t<wp::float32> var_qpos_out)
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
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::slice_t var_19;
        const wp::int32 var_20 = 0;
        wp::array_t<wp::float32> var_21;
        wp::slice_t var_22;
        const wp::int32 var_23 = 0;
        wp::array_t<wp::float32> var_24;
        wp::slice_t var_25;
        const wp::int32 var_26 = 0;
        wp::array_t<wp::float32> var_27;
        const wp::int32 var_28 = 0;
        bool var_29;
        wp::float32* var_30;
        const wp::int32 var_31 = 1;
        wp::int32 var_32;
        wp::float32* var_33;
        const wp::int32 var_34 = 2;
        wp::int32 var_35;
        wp::float32* var_36;
        wp::vec_t<3, wp::float32> var_37;
        wp::float32 var_38;
        wp::float32 var_39;
        wp::float32 var_40;
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
        wp::vec_t<3, wp::float32> var_52;
        wp::vec_t<3, wp::float32> var_53;
        wp::vec_t<3, wp::float32> var_54;
        const wp::int32 var_55 = 3;
        wp::int32 var_56;
        wp::float32* var_57;
        const wp::int32 var_58 = 4;
        wp::int32 var_59;
        wp::float32* var_60;
        const wp::int32 var_61 = 5;
        wp::int32 var_62;
        wp::float32* var_63;
        const wp::int32 var_64 = 6;
        wp::int32 var_65;
        wp::float32* var_66;
        wp::quat_t<wp::float32> var_67;
        wp::float32 var_68;
        wp::float32 var_69;
        wp::float32 var_70;
        wp::float32 var_71;
        const wp::int32 var_72 = 3;
        wp::int32 var_73;
        wp::float32* var_74;
        const wp::int32 var_75 = 4;
        wp::int32 var_76;
        wp::float32* var_77;
        const wp::int32 var_78 = 5;
        wp::int32 var_79;
        wp::float32* var_80;
        wp::vec_t<3, wp::float32> var_81;
        wp::float32 var_82;
        wp::float32 var_83;
        wp::float32 var_84;
        wp::vec_t<3, wp::float32> var_85;
        wp::quat_t<wp::float32> var_86;
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
        const wp::int32 var_111 = 3;
        wp::float32 var_112;
        const wp::int32 var_113 = 6;
        wp::int32 var_114;
        const wp::int32 var_115 = 1;
        bool var_116;
        const wp::int32 var_117 = 0;
        wp::int32 var_118;
        wp::float32* var_119;
        const wp::int32 var_120 = 1;
        wp::int32 var_121;
        wp::float32* var_122;
        const wp::int32 var_123 = 2;
        wp::int32 var_124;
        wp::float32* var_125;
        const wp::int32 var_126 = 3;
        wp::int32 var_127;
        wp::float32* var_128;
        wp::quat_t<wp::float32> var_129;
        wp::float32 var_130;
        wp::float32 var_131;
        wp::float32 var_132;
        wp::float32 var_133;
        wp::float32* var_134;
        const wp::int32 var_135 = 1;
        wp::int32 var_136;
        wp::float32* var_137;
        const wp::int32 var_138 = 2;
        wp::int32 var_139;
        wp::float32* var_140;
        wp::vec_t<3, wp::float32> var_141;
        wp::float32 var_142;
        wp::float32 var_143;
        wp::float32 var_144;
        wp::vec_t<3, wp::float32> var_145;
        wp::quat_t<wp::float32> var_146;
        const wp::int32 var_147 = 0;
        wp::float32 var_148;
        const wp::int32 var_149 = 0;
        wp::int32 var_150;
        const wp::int32 var_151 = 1;
        wp::float32 var_152;
        const wp::int32 var_153 = 1;
        wp::int32 var_154;
        const wp::int32 var_155 = 2;
        wp::float32 var_156;
        const wp::int32 var_157 = 2;
        wp::int32 var_158;
        const wp::int32 var_159 = 3;
        wp::float32 var_160;
        const wp::int32 var_161 = 3;
        wp::int32 var_162;
        wp::quat_t<wp::float32> var_163;
        wp::vec_t<3, wp::float32> var_164;
        wp::quat_t<wp::float32> var_165;
        wp::float32* var_166;
        wp::float32* var_167;
        wp::float32 var_168;
        wp::float32 var_169;
        wp::float32 var_170;
        wp::float32 var_171;
        wp::float32 var_172;
        wp::quat_t<wp::float32> var_173;
        wp::vec_t<3, wp::float32> var_174;
        wp::quat_t<wp::float32> var_175;
        //---------
        // forward
        // def _next_position(                                                                    <L 52>
        // worldid, jntid = wp.tid()                                                              <L 66>
        builtin_tid2d(var_0, var_1);
        // timestep = opt_timestep[worldid % opt_timestep.shape[0]]                               <L 67>
        var_2 = &(var_opt_timestep.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_opt_timestep, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // jnttype = jnt_type[jntid]                                                              <L 69>
        var_10 = wp::address(var_jnt_type, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // qpos_adr = jnt_qposadr[jntid]                                                          <L 70>
        var_13 = wp::address(var_jnt_qposadr, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // dof_adr = jnt_dofadr[jntid]                                                            <L 71>
        var_16 = wp::address(var_jnt_dofadr, var_1);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // qpos = qpos_in[worldid]                                                                <L 72>
        var_19 = wp::slice_t(var_0, var_0, var_20);
        var_21 = wp::view(var_qpos_in, var_19);
        // qpos_next = qpos_out[worldid]                                                          <L 73>
        var_22 = wp::slice_t(var_0, var_0, var_23);
        var_24 = wp::view(var_qpos_out, var_22);
        // qvel = qvel_in[worldid]                                                                <L 74>
        var_25 = wp::slice_t(var_0, var_0, var_26);
        var_27 = wp::view(var_qvel_in, var_25);
        // if jnttype == JointType.FREE:                                                          <L 76>
        var_29 = (var_11 == var_28);
        if (var_29) {
            // qpos_pos = wp.vec3(qpos[qpos_adr], qpos[qpos_adr + 1], qpos[qpos_adr + 2])         <L 77>
            var_30 = wp::address(var_21, var_14);
            var_32 = wp::add(var_14, var_31);
            var_33 = wp::address(var_21, var_32);
            var_35 = wp::add(var_14, var_34);
            var_36 = wp::address(var_21, var_35);
            var_38 = wp::load(var_30);
            var_39 = wp::load(var_33);
            var_40 = wp::load(var_36);
            var_37 = wp::vec_t<3, wp::float32>(var_38, var_39, var_40);
            // qvel_lin = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2]) * qvel_scale_in       <L 78>
            var_41 = wp::address(var_27, var_17);
            var_43 = wp::add(var_17, var_42);
            var_44 = wp::address(var_27, var_43);
            var_46 = wp::add(var_17, var_45);
            var_47 = wp::address(var_27, var_46);
            var_49 = wp::load(var_41);
            var_50 = wp::load(var_44);
            var_51 = wp::load(var_47);
            var_48 = wp::vec_t<3, wp::float32>(var_49, var_50, var_51);
            var_52 = wp::mul(var_48, var_qvel_scale_in);
            // qpos_new = qpos_pos + timestep * qvel_lin                                          <L 80>
            var_53 = wp::mul(var_8, var_52);
            var_54 = wp::add(var_37, var_53);
            // qpos_quat = wp.quat(                                                               <L 82>
            // qpos[qpos_adr + 3],                                                                <L 83>
            var_56 = wp::add(var_14, var_55);
            var_57 = wp::address(var_21, var_56);
            // qpos[qpos_adr + 4],                                                                <L 84>
            var_59 = wp::add(var_14, var_58);
            var_60 = wp::address(var_21, var_59);
            // qpos[qpos_adr + 5],                                                                <L 85>
            var_62 = wp::add(var_14, var_61);
            var_63 = wp::address(var_21, var_62);
            // qpos[qpos_adr + 6],                                                                <L 86>
            var_65 = wp::add(var_14, var_64);
            var_66 = wp::address(var_21, var_65);
            var_68 = wp::load(var_57);
            var_69 = wp::load(var_60);
            var_70 = wp::load(var_63);
            var_71 = wp::load(var_66);
            var_67 = wp::quat_t<wp::float32>(var_68, var_69, var_70, var_71);
            // qvel_ang = wp.vec3(qvel[dof_adr + 3], qvel[dof_adr + 4], qvel[dof_adr + 5]) * qvel_scale_in       <L 88>
            var_73 = wp::add(var_17, var_72);
            var_74 = wp::address(var_27, var_73);
            var_76 = wp::add(var_17, var_75);
            var_77 = wp::address(var_27, var_76);
            var_79 = wp::add(var_17, var_78);
            var_80 = wp::address(var_27, var_79);
            var_82 = wp::load(var_74);
            var_83 = wp::load(var_77);
            var_84 = wp::load(var_80);
            var_81 = wp::vec_t<3, wp::float32>(var_82, var_83, var_84);
            var_85 = wp::mul(var_81, var_qvel_scale_in);
            // qpos_quat_new = math.quat_integrate(qpos_quat, qvel_ang, timestep)                 <L 90>
            var_86 = quat_integrate_0(var_67, var_85, var_8);
            // qpos_next[qpos_adr + 0] = qpos_new[0]                                              <L 92>
            var_88 = wp::extract(var_54, var_87);
            var_90 = wp::add(var_14, var_89);
            wp::array_store(var_24, var_90, var_88);
            // qpos_next[qpos_adr + 1] = qpos_new[1]                                              <L 93>
            var_92 = wp::extract(var_54, var_91);
            var_94 = wp::add(var_14, var_93);
            wp::array_store(var_24, var_94, var_92);
            // qpos_next[qpos_adr + 2] = qpos_new[2]                                              <L 94>
            var_96 = wp::extract(var_54, var_95);
            var_98 = wp::add(var_14, var_97);
            wp::array_store(var_24, var_98, var_96);
            // qpos_next[qpos_adr + 3] = qpos_quat_new[0]                                         <L 95>
            var_100 = wp::extract(var_86, var_99);
            var_102 = wp::add(var_14, var_101);
            wp::array_store(var_24, var_102, var_100);
            // qpos_next[qpos_adr + 4] = qpos_quat_new[1]                                         <L 96>
            var_104 = wp::extract(var_86, var_103);
            var_106 = wp::add(var_14, var_105);
            wp::array_store(var_24, var_106, var_104);
            // qpos_next[qpos_adr + 5] = qpos_quat_new[2]                                         <L 97>
            var_108 = wp::extract(var_86, var_107);
            var_110 = wp::add(var_14, var_109);
            wp::array_store(var_24, var_110, var_108);
            // qpos_next[qpos_adr + 6] = qpos_quat_new[3]                                         <L 98>
            var_112 = wp::extract(var_86, var_111);
            var_114 = wp::add(var_14, var_113);
            wp::array_store(var_24, var_114, var_112);
        }
        if (!var_29) {
            // elif jnttype == JointType.BALL:                                                    <L 100>
            var_116 = (var_11 == var_115);
            if (var_116) {
                // qpos_quat = wp.quat(qpos[qpos_adr + 0], qpos[qpos_adr + 1], qpos[qpos_adr + 2], qpos[qpos_adr + 3])       <L 101>
                var_118 = wp::add(var_14, var_117);
                var_119 = wp::address(var_21, var_118);
                var_121 = wp::add(var_14, var_120);
                var_122 = wp::address(var_21, var_121);
                var_124 = wp::add(var_14, var_123);
                var_125 = wp::address(var_21, var_124);
                var_127 = wp::add(var_14, var_126);
                var_128 = wp::address(var_21, var_127);
                var_130 = wp::load(var_119);
                var_131 = wp::load(var_122);
                var_132 = wp::load(var_125);
                var_133 = wp::load(var_128);
                var_129 = wp::quat_t<wp::float32>(var_130, var_131, var_132, var_133);
                // qvel_ang = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2]) * qvel_scale_in       <L 102>
                var_134 = wp::address(var_27, var_17);
                var_136 = wp::add(var_17, var_135);
                var_137 = wp::address(var_27, var_136);
                var_139 = wp::add(var_17, var_138);
                var_140 = wp::address(var_27, var_139);
                var_142 = wp::load(var_134);
                var_143 = wp::load(var_137);
                var_144 = wp::load(var_140);
                var_141 = wp::vec_t<3, wp::float32>(var_142, var_143, var_144);
                var_145 = wp::mul(var_141, var_qvel_scale_in);
                // qpos_quat_new = math.quat_integrate(qpos_quat, qvel_ang, timestep)             <L 104>
                var_146 = quat_integrate_0(var_129, var_145, var_8);
                // qpos_next[qpos_adr + 0] = qpos_quat_new[0]                                     <L 106>
                var_148 = wp::extract(var_146, var_147);
                var_150 = wp::add(var_14, var_149);
                wp::array_store(var_24, var_150, var_148);
                // qpos_next[qpos_adr + 1] = qpos_quat_new[1]                                     <L 107>
                var_152 = wp::extract(var_146, var_151);
                var_154 = wp::add(var_14, var_153);
                wp::array_store(var_24, var_154, var_152);
                // qpos_next[qpos_adr + 2] = qpos_quat_new[2]                                     <L 108>
                var_156 = wp::extract(var_146, var_155);
                var_158 = wp::add(var_14, var_157);
                wp::array_store(var_24, var_158, var_156);
                // qpos_next[qpos_adr + 3] = qpos_quat_new[3]                                     <L 109>
                var_160 = wp::extract(var_146, var_159);
                var_162 = wp::add(var_14, var_161);
                wp::array_store(var_24, var_162, var_160);
            }
            var_163 = wp::where(var_116, var_129, var_67);
            var_164 = wp::where(var_116, var_145, var_85);
            var_165 = wp::where(var_116, var_146, var_86);
            if (!var_116) {
                // qpos_next[qpos_adr] = qpos[qpos_adr] + timestep * qvel[dof_adr] * qvel_scale_in       <L 112>
                var_166 = wp::address(var_21, var_14);
                var_167 = wp::address(var_27, var_17);
                var_169 = wp::load(var_167);
                var_168 = wp::mul(var_8, var_169);
                var_170 = wp::mul(var_168, var_qvel_scale_in);
                var_172 = wp::load(var_166);
                var_171 = wp::add(var_172, var_170);
                wp::array_store(var_24, var_14, var_171);
            }
        }
        var_173 = wp::where(var_29, var_67, var_163);
        var_174 = wp::where(var_29, var_85, var_164);
        var_175 = wp::where(var_29, var_86, var_165);
    }
}



extern "C" __global__ void _tendon_actuator_force_clamp_fd85b13d_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<bool> var_tendon_actfrclimited,
    wp::array_t<wp::vec_t<2, wp::float32>> var_tendon_actfrcrange,
    wp::array_t<wp::int32> var_actuator_trntype,
    wp::array_t<wp::vec_t<2, wp::int32>> var_actuator_trnid,
    wp::array_t<wp::float32> var_ten_actfrc_in,
    wp::array_t<wp::float32> var_actuator_force_out)
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
        const wp::int32 var_3 = 3;
        bool var_4;
        wp::int32 var_5;
        wp::vec_t<2, wp::int32>* var_6;
        const wp::int32 var_7 = 0;
        wp::int32 var_8;
        wp::vec_t<2, wp::int32> var_9;
        bool* var_10;
        bool var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::shape_t* var_15;
        const wp::int32 var_16 = 0;
        wp::int32 var_17;
        wp::shape_t var_18;
        wp::int32 var_19;
        wp::vec_t<2, wp::float32>* var_20;
        wp::vec_t<2, wp::float32> var_21;
        wp::vec_t<2, wp::float32> var_22;
        const wp::int32 var_23 = 0;
        wp::float32 var_24;
        bool var_25;
        const wp::int32 var_26 = 0;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::float32* var_29;
        const wp::int32 var_30 = 0;
        wp::float32 var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        const wp::int32 var_35 = 1;
        wp::float32 var_36;
        bool var_37;
        const wp::int32 var_38 = 1;
        wp::float32 var_39;
        wp::float32 var_40;
        wp::float32* var_41;
        const wp::int32 var_42 = 1;
        wp::float32 var_43;
        wp::float32 var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        bool var_47;
        //---------
        // forward
        // def _tendon_actuator_force_clamp(                                                      <L 757>
        // worldid, actid = wp.tid()                                                              <L 768>
        builtin_tid2d(var_0, var_1);
        // if actuator_trntype[actid] == TrnType.TENDON:                                          <L 770>
        var_2 = wp::address(var_actuator_trntype, var_1);
        var_5 = wp::load(var_2);
        var_4 = (var_5 == var_3);
        if (var_4) {
            // tenid = actuator_trnid[actid][0]                                                   <L 771>
            var_6 = wp::address(var_actuator_trnid, var_1);
            var_9 = wp::load(var_6);
            var_8 = wp::extract(var_9, var_7);
            // if tendon_actfrclimited[tenid]:                                                    <L 772>
            var_10 = wp::address(var_tendon_actfrclimited, var_8);
            var_11 = wp::load(var_10);
            if (var_11) {
                // ten_actfrc = ten_actfrc_in[worldid, tenid]                                     <L 773>
                var_12 = wp::address(var_ten_actfrc_in, var_0, var_8);
                var_14 = wp::load(var_12);
                var_13 = wp::copy(var_14);
                // actfrcrange = tendon_actfrcrange[worldid % tendon_actfrcrange.shape[0], tenid]       <L 774>
                var_15 = &(var_tendon_actfrcrange.shape);
                var_18 = wp::load(var_15);
                var_17 = wp::extract(var_18, var_16);
                var_19 = wp::mod(var_0, var_17);
                var_20 = wp::address(var_tendon_actfrcrange, var_19, var_8);
                var_22 = wp::load(var_20);
                var_21 = wp::copy(var_22);
                // if ten_actfrc < actfrcrange[0]:                                                <L 776>
                var_24 = wp::extract(var_21, var_23);
                var_25 = (var_13 < var_24);
                if (var_25) {
                    // actuator_force_out[worldid, actid] *= actfrcrange[0] / ten_actfrc          <L 777>
                    var_27 = wp::extract(var_21, var_26);
                    var_28 = wp::div(var_27, var_13);
                    var_29 = wp::address(var_actuator_force_out, var_0, var_1);
                    var_31 = wp::extract(var_21, var_30);
                    var_32 = wp::div(var_31, var_13);
                    var_34 = wp::load(var_29);
                    var_33 = wp::mul(var_34, var_32);
                    wp::array_store(var_actuator_force_out, var_0, var_1, var_33);
                }
                if (!var_25) {
                    // elif ten_actfrc > actfrcrange[1]:                                          <L 778>
                    var_36 = wp::extract(var_21, var_35);
                    var_37 = (var_13 > var_36);
                    if (var_37) {
                        // actuator_force_out[worldid, actid] *= actfrcrange[1] / ten_actfrc       <L 779>
                        var_39 = wp::extract(var_21, var_38);
                        var_40 = wp::div(var_39, var_13);
                        var_41 = wp::address(var_actuator_force_out, var_0, var_1);
                        var_43 = wp::extract(var_21, var_42);
                        var_44 = wp::div(var_43, var_13);
                        var_46 = wp::load(var_41);
                        var_45 = wp::mul(var_46, var_44);
                        wp::array_store(var_actuator_force_out, var_0, var_1, var_45);
                    }
                }
            }
            var_47 = wp::load(var_10);
        }
    }
}



extern "C" __global__ void _next_velocity_d66c2f53_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::float32> var_qacc_in,
    wp::float32 var_qacc_scale_in,
    wp::array_t<wp::float32> var_qvel_out)
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
        wp::float32* var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        //---------
        // forward
        // def _next_velocity(                                                                    <L 116>
        // worldid, dofid = wp.tid()                                                              <L 127>
        builtin_tid2d(var_0, var_1);
        // timestep = opt_timestep[worldid % opt_timestep.shape[0]]                               <L 128>
        var_2 = &(var_opt_timestep.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_opt_timestep, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // qvel_out[worldid, dofid] = qvel_in[worldid, dofid] + qacc_scale_in * qacc_in[worldid, dofid] * timestep       <L 129>
        var_10 = wp::address(var_qvel_in, var_0, var_1);
        var_11 = wp::address(var_qacc_in, var_0, var_1);
        var_13 = wp::load(var_11);
        var_12 = wp::mul(var_qacc_scale_in, var_13);
        var_14 = wp::mul(var_12, var_8);
        var_16 = wp::load(var_10);
        var_15 = wp::add(var_16, var_14);
        wp::array_store(var_qvel_out, var_0, var_1, var_15);
    }
}



extern "C" __global__ void _rk_accumulate_velocity_acceleration_829bb7cc_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::float32> var_qacc_in,
    wp::float32 var_scale,
    wp::array_t<wp::float32> var_qvel_out,
    wp::array_t<wp::float32> var_qacc_out)
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
        wp::float32 var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        //---------
        // forward
        // def _rk_accumulate_velocity_acceleration(                                              <L 404>
        // worldid, dofid = wp.tid()                                                              <L 414>
        builtin_tid2d(var_0, var_1);
        // qvel_out[worldid, dofid] += scale * qvel_in[worldid, dofid]                            <L 415>
        var_2 = wp::address(var_qvel_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::mul(var_scale, var_4);
        var_5 = wp::atomic_add(var_qvel_out, var_0, var_1, var_3);
        // qacc_out[worldid, dofid] += scale * qacc_in[worldid, dofid]                            <L 416>
        var_6 = wp::address(var_qacc_in, var_0, var_1);
        var_8 = wp::load(var_6);
        var_7 = wp::mul(var_scale, var_8);
        var_9 = wp::atomic_add(var_qacc_out, var_0, var_1, var_7);
    }
}



extern "C" __global__ void _qfrc_actuator_6de5c51c_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_moment_rownnz_in,
    wp::array_t<wp::int32> var_moment_rowadr_in,
    wp::array_t<wp::int32> var_moment_colind_in,
    wp::array_t<wp::float32> var_actuator_moment_in,
    wp::array_t<wp::float32> var_actuator_force_in,
    wp::array_t<wp::float32> var_qfrc_actuator_out)
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
        wp::range_t var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::float32* var_14;
        wp::float32* var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::slice_t var_19;
        const wp::int32 var_20 = 0;
        wp::array_t<wp::float32> var_21;
        wp::float32 var_22;
        //---------
        // forward
        // def _qfrc_actuator(                                                                    <L 783>
        // worldid, actid = wp.tid()                                                              <L 793>
        builtin_tid2d(var_0, var_1);
        // rownnz = moment_rownnz_in[worldid, actid]                                              <L 795>
        var_2 = wp::address(var_moment_rownnz_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // rowadr = moment_rowadr_in[worldid, actid]                                              <L 796>
        var_5 = wp::address(var_moment_rowadr_in, var_0, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // for i in range(rownnz):                                                                <L 798>
        var_8 = wp::range(var_3);
        start_for_0:;
            if (iter_cmp(var_8) == 0) goto end_for_0;
            var_9 = wp::iter_next(var_8);
            // sparseid = rowadr + i                                                              <L 799>
            var_10 = wp::add(var_6, var_9);
            // colind = moment_colind_in[worldid, sparseid]                                       <L 800>
            var_11 = wp::address(var_moment_colind_in, var_0, var_10);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
            // qfrc = actuator_moment_in[worldid, sparseid] * actuator_force_in[worldid, actid]       <L 801>
            var_14 = wp::address(var_actuator_moment_in, var_0, var_10);
            var_15 = wp::address(var_actuator_force_in, var_0, var_1);
            var_17 = wp::load(var_14);
            var_18 = wp::load(var_15);
            var_16 = wp::mul(var_17, var_18);
            // wp.atomic_add(qfrc_actuator_out[worldid], colind, qfrc)                            <L 802>
            var_19 = wp::slice_t(var_0, var_0, var_20);
            var_21 = wp::view(var_qfrc_actuator_out, var_19);
            var_22 = wp::atomic_add(var_21, var_12, var_16);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void _next_activation_92aa6afe_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::int32> var_actuator_dyntype,
    wp::array_t<wp::int32> var_actuator_actadr,
    wp::array_t<wp::int32> var_actuator_actnum,
    wp::array_t<bool> var_actuator_actlimited,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_dynprm,
    wp::array_t<wp::vec_t<2, wp::float32>> var_actuator_actrange,
    wp::array_t<wp::float32> var_act_in,
    wp::array_t<wp::float32> var_act_dot_in,
    wp::float32 var_act_dot_scale,
    bool var_limit,
    wp::array_t<wp::float32> var_act_out)
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
        wp::shape_t* var_12;
        const wp::int32 var_13 = 0;
        wp::int32 var_14;
        wp::shape_t var_15;
        wp::int32 var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::int32* var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        wp::range_t var_24;
        wp::int32 var_25;
        wp::float32* var_26;
        wp::int32* var_27;
        wp::vec_t<10, wp::float32>* var_28;
        wp::vec_t<2, wp::float32>* var_29;
        wp::float32* var_30;
        wp::float32* var_31;
        bool* var_32;
        bool var_33;
        bool var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::int32 var_37;
        wp::vec_t<10, wp::float32> var_38;
        wp::vec_t<2, wp::float32> var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        //---------
        // forward
        // def _next_activation(                                                                  <L 133>
        // worldid, uid = wp.tid()                                                                <L 151>
        builtin_tid2d(var_0, var_1);
        // opt_timestep_id = worldid % opt_timestep.shape[0]                                      <L 152>
        var_2 = &(var_opt_timestep.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // actuator_dynprm_id = worldid % actuator_dynprm.shape[0]                                <L 153>
        var_7 = &(var_actuator_dynprm.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::mod(var_0, var_9);
        // actuator_actrange_id = worldid % actuator_actrange.shape[0]                            <L 154>
        var_12 = &(var_actuator_actrange.shape);
        var_15 = wp::load(var_12);
        var_14 = wp::extract(var_15, var_13);
        var_16 = wp::mod(var_0, var_14);
        // actadr = actuator_actadr[uid]                                                          <L 155>
        var_17 = wp::address(var_actuator_actadr, var_1);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // actnum = actuator_actnum[uid]                                                          <L 156>
        var_20 = wp::address(var_actuator_actnum, var_1);
        var_22 = wp::load(var_20);
        var_21 = wp::copy(var_22);
        // for j in range(actadr, actadr + actnum):                                               <L 157>
        var_23 = wp::add(var_18, var_21);
        var_24 = wp::range(var_18, var_23);
        start_for_0:;
            if (iter_cmp(var_24) == 0) goto end_for_0;
            var_25 = wp::iter_next(var_24);
            // act = next_act(                                                                    <L 158>
            // opt_timestep[opt_timestep_id],                                                     <L 159>
            var_26 = wp::address(var_opt_timestep, var_6);
            // actuator_dyntype[uid],                                                             <L 160>
            var_27 = wp::address(var_actuator_dyntype, var_1);
            // actuator_dynprm[actuator_dynprm_id, uid],                                          <L 161>
            var_28 = wp::address(var_actuator_dynprm, var_11, var_1);
            // actuator_actrange[actuator_actrange_id, uid],                                      <L 162>
            var_29 = wp::address(var_actuator_actrange, var_16, var_1);
            // act_in[worldid, j],                                                                <L 163>
            var_30 = wp::address(var_act_in, var_0, var_25);
            // act_dot_in[worldid, j],                                                            <L 164>
            var_31 = wp::address(var_act_dot_in, var_0, var_25);
            // act_dot_scale,                                                                     <L 165>
            // limit and actuator_actlimited[uid],                                                <L 166>
            var_32 = wp::address(var_actuator_actlimited, var_1);
            var_33 = wp::load(var_32);
            var_34 = var_limit && var_33;
            var_36 = wp::load(var_26);
            var_37 = wp::load(var_27);
            var_38 = wp::load(var_28);
            var_39 = wp::load(var_29);
            var_40 = wp::load(var_30);
            var_41 = wp::load(var_31);
            var_35 = next_act_0(var_36, var_37, var_38, var_39, var_40, var_41, var_act_dot_scale, var_34);
            // act_out[worldid, j] = act                                                          <L 168>
            wp::array_store(var_act_out, var_0, var_25, var_35);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void _rk_accumulate_activation_velocity_5935ab3f_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_act_dot_in,
    wp::float32 var_scale,
    wp::array_t<wp::float32> var_act_dot_out)
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
        wp::float32 var_5;
        //---------
        // forward
        // def _rk_accumulate_activation_velocity(                                                <L 420>
        // worldid, actid = wp.tid()                                                              <L 428>
        builtin_tid2d(var_0, var_1);
        // act_dot_out[worldid, actid] += scale * act_dot_in[worldid, actid]                      <L 429>
        var_2 = wp::address(var_act_dot_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::mul(var_scale, var_4);
        var_5 = wp::atomic_add(var_act_dot_out, var_0, var_1, var_3);
    }
}



extern "C" __global__ void _next_time_dedd0a7f_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_opt_timestep,
    bool var_is_sparse,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::float32> var_time_in,
    wp::array_t<wp::int32> var_efc_J_rownnz_in,
    wp::array_t<wp::int32> var_efc_J_rowadr_in,
    wp::int32 var_nworld_in,
    wp::int32 var_naconmax_in,
    wp::int32 var_njmax_in,
    wp::int32 var_njmax_nnz_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::int32> var_ncollision_in,
    wp::array_t<wp::float32> var_time_out)
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
        wp::float32* var_1;
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        bool var_14;
        const wp::str var_15 = "nefc overflow - please increase njmax to %u\n";
        const wp::int32 var_16 = 0;
        bool var_17;
        bool var_18;
        wp::int32 var_19;
        const wp::int32 var_20 = 1;
        wp::int32 var_21;
        wp::int32* var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        bool var_27;
        const wp::str var_28 = "njmax_nnz overflow - please increase njmax_nnz to %u\n";
        const wp::int32 var_29 = 0;
        bool var_30;
        const wp::int32 var_31 = 0;
        wp::int32* var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        bool var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        wp::float32 var_38;
        wp::float32 var_39;
        wp::int32 var_40;
        const wp::str var_41 = "broadphase overflow - please increase nconmax to %u or naconmax to %u\n";
        const wp::int32 var_42 = 0;
        wp::int32* var_43;
        bool var_44;
        wp::int32 var_45;
        const wp::int32 var_46 = 0;
        wp::int32* var_47;
        wp::float32 var_48;
        wp::int32 var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        wp::float32 var_52;
        wp::int32 var_53;
        const wp::str var_54 = "narrowphase overflow - please increase nconmax to %u or naconmax to %u\n";
        const wp::int32 var_55 = 0;
        wp::int32* var_56;
        wp::int32 var_57;
        wp::int32 var_58;
        //---------
        // forward
        // def _next_time(                                                                        <L 172>
        // worldid = wp.tid()                                                                     <L 190>
        var_0 = builtin_tid1d();
        // time_out[worldid] = time_in[worldid] + opt_timestep[worldid % opt_timestep.shape[0]]       <L 191>
        var_1 = wp::address(var_time_in, var_0);
        var_2 = &(var_opt_timestep.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_opt_timestep, var_6);
        var_9 = wp::load(var_1);
        var_10 = wp::load(var_7);
        var_8 = wp::add(var_9, var_10);
        wp::array_store(var_time_out, var_0, var_8);
        // nefc = nefc_in[worldid]                                                                <L 192>
        var_11 = wp::address(var_nefc_in, var_0);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // if nefc > njmax_in:                                                                    <L 194>
        var_14 = (var_12 > var_njmax_in);
        if (var_14) {
            // wp.printf("nefc overflow - please increase njmax to %u\n", nefc)                   <L 195>
            printf(var_15, var_12);
        }
        if (!var_14) {
            // elif nefc > 0 and is_sparse:                                                       <L 196>
            var_17 = (var_12 > var_16);
            var_18 = var_17 && var_is_sparse;
            if (var_18) {
                // efcid = wp.min(nefc, njmax_in) - 1                                             <L 197>
                var_19 = wp::min(var_12, var_njmax_in);
                var_21 = wp::sub(var_19, var_20);
                // efc_nnz = efc_J_rowadr_in[worldid, efcid] + efc_J_rownnz_in[worldid, efcid]       <L 198>
                var_22 = wp::address(var_efc_J_rowadr_in, var_0, var_21);
                var_23 = wp::address(var_efc_J_rownnz_in, var_0, var_21);
                var_25 = wp::load(var_22);
                var_26 = wp::load(var_23);
                var_24 = wp::add(var_25, var_26);
                // if efc_nnz > njmax_nnz_in:                                                     <L 199>
                var_27 = (var_24 > var_njmax_nnz_in);
                if (var_27) {
                    // wp.printf("njmax_nnz overflow - please increase njmax_nnz to %u\n", efc_nnz)       <L 200>
                    printf(var_28, var_24);
                }
            }
        }
        // if worldid == 0:                                                                       <L 202>
        var_30 = (var_0 == var_29);
        if (var_30) {
            // ncollision = ncollision_in[0]                                                      <L 203>
            var_32 = wp::address(var_ncollision_in, var_31);
            var_34 = wp::load(var_32);
            var_33 = wp::copy(var_34);
            // if ncollision > naconmax_in:                                                       <L 204>
            var_35 = (var_33 > var_naconmax_in);
            if (var_35) {
                // nconmax = int(wp.ceil(float(ncollision) / float(nworld_in)))                   <L 205>
                var_36 = wp::float(var_33);
                var_37 = wp::float(var_nworld_in);
                var_38 = wp::div(var_36, var_37);
                var_39 = wp::ceil(var_38);
                var_40 = wp::int(var_39);
                // wp.printf("broadphase overflow - please increase nconmax to %u or naconmax to %u\n", nconmax, ncollision)       <L 206>
                printf(var_41, var_40, var_33);
            }
            // if nacon_in[0] > naconmax_in:                                                      <L 208>
            var_43 = wp::address(var_nacon_in, var_42);
            var_45 = wp::load(var_43);
            var_44 = (var_45 > var_naconmax_in);
            if (var_44) {
                // nconmax = int(wp.ceil(float(nacon_in[0]) / float(nworld_in)))                  <L 209>
                var_47 = wp::address(var_nacon_in, var_46);
                var_49 = wp::load(var_47);
                var_48 = wp::float(var_49);
                var_50 = wp::float(var_nworld_in);
                var_51 = wp::div(var_48, var_50);
                var_52 = wp::ceil(var_51);
                var_53 = wp::int(var_52);
                // wp.printf("narrowphase overflow - please increase nconmax to %u or naconmax to %u\n", nconmax, nacon_in[0])       <L 210>
                var_56 = wp::address(var_nacon_in, var_55);
                var_57 = wp::load(var_56);
                printf(var_54, var_53, var_57);
            }
            var_58 = wp::where(var_44, var_53, var_40);
        }
    }
}



extern "C" __global__ void _actuator_force_a319cbb4_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_na,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::int32> var_actuator_dyntype,
    wp::array_t<wp::int32> var_actuator_gaintype,
    wp::array_t<wp::int32> var_actuator_biastype,
    wp::array_t<wp::int32> var_actuator_actadr,
    wp::array_t<wp::int32> var_actuator_actnum,
    wp::array_t<bool> var_actuator_ctrllimited,
    wp::array_t<bool> var_actuator_forcelimited,
    wp::array_t<bool> var_actuator_actlimited,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_dynprm,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_gainprm,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_biasprm,
    wp::array_t<bool> var_actuator_actearly,
    wp::array_t<wp::vec_t<2, wp::float32>> var_actuator_ctrlrange,
    wp::array_t<wp::vec_t<2, wp::float32>> var_actuator_forcerange,
    wp::array_t<wp::vec_t<2, wp::float32>> var_actuator_actrange,
    wp::array_t<wp::float32> var_actuator_acc0,
    wp::array_t<wp::vec_t<2, wp::float32>> var_actuator_lengthrange,
    wp::array_t<wp::float32> var_act_in,
    wp::array_t<wp::float32> var_ctrl_in,
    wp::array_t<wp::float32> var_actuator_length_in,
    wp::array_t<wp::float32> var_actuator_velocity_in,
    wp::int32 var_dsbl_clampctrl,
    wp::array_t<wp::float32> var_act_dot_out,
    wp::array_t<wp::float32> var_actuator_force_out)
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
        bool* var_10;
        bool var_11;
        bool var_12;
        bool var_13;
        wp::vec_t<2, wp::float32>* var_14;
        wp::vec_t<2, wp::float32> var_15;
        wp::vec_t<2, wp::float32> var_16;
        const wp::int32 var_17 = 0;
        wp::float32 var_18;
        const wp::int32 var_19 = 1;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::int32* var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        const wp::int32 var_27 = 0;
        bool var_28;
        bool var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        const wp::int32 var_33 = 1;
        wp::int32 var_34;
        wp::int32* var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        wp::shape_t* var_38;
        const wp::int32 var_39 = 0;
        wp::int32 var_40;
        wp::shape_t var_41;
        wp::int32 var_42;
        wp::vec_t<10, wp::float32>* var_43;
        wp::vec_t<10, wp::float32> var_44;
        wp::vec_t<10, wp::float32> var_45;
        const wp::int32 var_46 = 1;
        bool var_47;
        wp::float32 var_48;
        const wp::int32 var_49 = 2;
        bool var_50;
        const wp::int32 var_51 = 3;
        bool var_52;
        bool var_53;
        wp::float32* var_54;
        wp::float32 var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        const wp::int32 var_58 = 0;
        wp::float32 var_59;
        const wp::float32 var_60 = 1e-15;
        wp::float32 var_61;
        wp::float32 var_62;
        wp::float32 var_63;
        const wp::int32 var_64 = 4;
        bool var_65;
        wp::shape_t* var_66;
        const wp::int32 var_67 = 0;
        wp::int32 var_68;
        wp::shape_t var_69;
        wp::int32 var_70;
        wp::vec_t<10, wp::float32>* var_71;
        wp::vec_t<10, wp::float32> var_72;
        wp::vec_t<10, wp::float32> var_73;
        wp::float32* var_74;
        wp::float32 var_75;
        wp::float32 var_76;
        wp::float32 var_77;
        wp::vec_t<10, wp::float32> var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        const wp::int32 var_81 = 6;
        bool var_82;
        const wp::float32 var_83 = 0.0;
        wp::float32 var_84;
        const wp::float32 var_85 = 0.0;
        wp::float32 var_86;
        wp::float32 var_87;
        wp::vec_t<10, wp::float32> var_88;
        wp::float32 var_89;
        wp::float32 var_90;
        wp::vec_t<10, wp::float32> var_91;
        wp::float32 var_92;
        bool* var_93;
        bool var_94;
        const wp::int32 var_95 = 1;
        bool var_96;
        const wp::int32 var_97 = 0;
        bool var_98;
        bool var_99;
        wp::float32* var_100;
        wp::float32 var_101;
        wp::float32 var_102;
        wp::float32 var_103;
        wp::shape_t* var_104;
        const wp::int32 var_105 = 0;
        wp::int32 var_106;
        wp::shape_t var_107;
        wp::int32 var_108;
        wp::float32* var_109;
        wp::shape_t* var_110;
        const wp::int32 var_111 = 0;
        wp::int32 var_112;
        wp::shape_t var_113;
        wp::int32 var_114;
        wp::vec_t<2, wp::float32>* var_115;
        const wp::float32 var_116 = 1.0;
        bool* var_117;
        wp::float32 var_118;
        wp::float32 var_119;
        wp::vec_t<2, wp::float32> var_120;
        bool var_121;
        bool var_122;
        wp::float32 var_123;
        bool var_124;
        wp::float32 var_125;
        bool var_126;
        bool var_127;
        wp::float32* var_128;
        wp::float32 var_129;
        wp::float32 var_130;
        bool var_131;
        wp::float32 var_132;
        bool var_133;
        wp::float32 var_134;
        wp::float32* var_135;
        wp::float32 var_136;
        wp::float32 var_137;
        wp::float32* var_138;
        wp::float32 var_139;
        wp::float32 var_140;
        wp::int32* var_141;
        wp::int32 var_142;
        wp::int32 var_143;
        wp::shape_t* var_144;
        const wp::int32 var_145 = 0;
        wp::int32 var_146;
        wp::shape_t var_147;
        wp::int32 var_148;
        wp::vec_t<10, wp::float32>* var_149;
        wp::vec_t<10, wp::float32> var_150;
        wp::vec_t<10, wp::float32> var_151;
        const wp::float32 var_152 = 0.0;
        const wp::int32 var_153 = 0;
        bool var_154;
        const wp::int32 var_155 = 0;
        wp::float32 var_156;
        wp::float32 var_157;
        const wp::int32 var_158 = 1;
        bool var_159;
        const wp::int32 var_160 = 0;
        wp::float32 var_161;
        const wp::int32 var_162 = 1;
        wp::float32 var_163;
        wp::float32 var_164;
        wp::float32 var_165;
        const wp::int32 var_166 = 2;
        wp::float32 var_167;
        wp::float32 var_168;
        wp::float32 var_169;
        wp::float32 var_170;
        const wp::int32 var_171 = 2;
        bool var_172;
        wp::shape_t* var_173;
        const wp::int32 var_174 = 0;
        wp::int32 var_175;
        wp::shape_t var_176;
        wp::int32 var_177;
        wp::float32* var_178;
        wp::float32 var_179;
        wp::float32 var_180;
        wp::shape_t* var_181;
        const wp::int32 var_182 = 0;
        wp::int32 var_183;
        wp::shape_t var_184;
        wp::int32 var_185;
        wp::vec_t<2, wp::float32>* var_186;
        wp::vec_t<2, wp::float32> var_187;
        wp::vec_t<2, wp::float32> var_188;
        wp::float32 var_189;
        wp::float32 var_190;
        wp::float32 var_191;
        wp::float32 var_192;
        wp::int32* var_193;
        wp::int32 var_194;
        wp::int32 var_195;
        wp::shape_t* var_196;
        const wp::int32 var_197 = 0;
        wp::int32 var_198;
        wp::shape_t var_199;
        wp::int32 var_200;
        wp::vec_t<10, wp::float32>* var_201;
        wp::vec_t<10, wp::float32> var_202;
        wp::vec_t<10, wp::float32> var_203;
        const wp::float32 var_204 = 0.0;
        const wp::int32 var_205 = 1;
        bool var_206;
        const wp::int32 var_207 = 0;
        wp::float32 var_208;
        const wp::int32 var_209 = 1;
        wp::float32 var_210;
        wp::float32 var_211;
        wp::float32 var_212;
        const wp::int32 var_213 = 2;
        wp::float32 var_214;
        wp::float32 var_215;
        wp::float32 var_216;
        wp::float32 var_217;
        const wp::int32 var_218 = 2;
        bool var_219;
        wp::shape_t* var_220;
        const wp::int32 var_221 = 0;
        wp::int32 var_222;
        wp::shape_t var_223;
        wp::int32 var_224;
        wp::float32* var_225;
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
        wp::float32 var_236;
        wp::float32 var_237;
        wp::vec_t<2, wp::float32> var_238;
        wp::float32 var_239;
        wp::float32 var_240;
        wp::vec_t<2, wp::float32> var_241;
        wp::float32 var_242;
        wp::float32 var_243;
        wp::float32 var_244;
        bool* var_245;
        bool var_246;
        wp::shape_t* var_247;
        const wp::int32 var_248 = 0;
        wp::int32 var_249;
        wp::shape_t var_250;
        wp::int32 var_251;
        wp::vec_t<2, wp::float32>* var_252;
        wp::vec_t<2, wp::float32> var_253;
        wp::vec_t<2, wp::float32> var_254;
        const wp::int32 var_255 = 0;
        wp::float32 var_256;
        const wp::int32 var_257 = 1;
        wp::float32 var_258;
        wp::float32 var_259;
        bool var_260;
        wp::float32 var_261;
        bool var_262;
        //---------
        // forward
        // def _actuator_force(                                                                   <L 617>
        // worldid, uid = wp.tid()                                                                <L 649>
        builtin_tid2d(var_0, var_1);
        // actuator_ctrlrange_id = worldid % actuator_ctrlrange.shape[0]                          <L 651>
        var_2 = &(var_actuator_ctrlrange.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // ctrl = ctrl_in[worldid, uid]                                                           <L 653>
        var_7 = wp::address(var_ctrl_in, var_0, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // if actuator_ctrllimited[uid] and not dsbl_clampctrl:                                   <L 655>
        var_10 = wp::address(var_actuator_ctrllimited, var_1);
        var_11 = wp::unot(var_dsbl_clampctrl);
        var_12 = wp::load(var_10);
        var_13 = var_12 && var_11;
        if (var_13) {
            // ctrlrange = actuator_ctrlrange[actuator_ctrlrange_id, uid]                         <L 656>
            var_14 = wp::address(var_actuator_ctrlrange, var_6, var_1);
            var_16 = wp::load(var_14);
            var_15 = wp::copy(var_16);
            // ctrl = wp.clamp(ctrl, ctrlrange[0], ctrlrange[1])                                  <L 657>
            var_18 = wp::extract(var_15, var_17);
            var_20 = wp::extract(var_15, var_19);
            var_21 = wp::clamp(var_8, var_18, var_20);
        }
        var_22 = wp::where(var_13, var_21, var_8);
        // ctrl_act = ctrl                                                                        <L 658>
        var_23 = wp::copy(var_22);
        // act_first = actuator_actadr[uid]                                                       <L 660>
        var_24 = wp::address(var_actuator_actadr, var_1);
        var_26 = wp::load(var_24);
        var_25 = wp::copy(var_26);
        // if na and act_first >= 0:                                                              <L 661>
        var_28 = (var_25 >= var_27);
        var_29 = var_na && var_28;
        if (var_29) {
            // act_last = act_first + actuator_actnum[uid] - 1                                    <L 662>
            var_30 = wp::address(var_actuator_actnum, var_1);
            var_32 = wp::load(var_30);
            var_31 = wp::add(var_25, var_32);
            var_34 = wp::sub(var_31, var_33);
            // dyntype = actuator_dyntype[uid]                                                    <L 663>
            var_35 = wp::address(var_actuator_dyntype, var_1);
            var_37 = wp::load(var_35);
            var_36 = wp::copy(var_37);
            // dynprm = actuator_dynprm[worldid % actuator_dynprm.shape[0], uid]                  <L 664>
            var_38 = &(var_actuator_dynprm.shape);
            var_41 = wp::load(var_38);
            var_40 = wp::extract(var_41, var_39);
            var_42 = wp::mod(var_0, var_40);
            var_43 = wp::address(var_actuator_dynprm, var_42, var_1);
            var_45 = wp::load(var_43);
            var_44 = wp::copy(var_45);
            // if dyntype == DynType.INTEGRATOR:                                                  <L 666>
            var_47 = (var_36 == var_46);
            if (var_47) {
                // act_dot = ctrl                                                                 <L 667>
                var_48 = wp::copy(var_22);
            }
            if (!var_47) {
                // elif dyntype == DynType.FILTER or dyntype == DynType.FILTEREXACT:              <L 668>
                var_50 = (var_36 == var_49);
                var_52 = (var_36 == var_51);
                var_53 = var_50 || var_52;
                if (var_53) {
                    // act = act_in[worldid, act_last]                                            <L 669>
                    var_54 = wp::address(var_act_in, var_0, var_34);
                    var_56 = wp::load(var_54);
                    var_55 = wp::copy(var_56);
                    // act_dot = (ctrl - act) / wp.max(dynprm[0], MJ_MINVAL)                      <L 670>
                    var_57 = wp::sub(var_22, var_55);
                    var_59 = wp::extract(var_44, var_58);
                    var_61 = wp::max(var_59, var_60);
                    var_62 = wp::div(var_57, var_61);
                }
                var_63 = wp::where(var_53, var_62, var_48);
                if (!var_53) {
                    // elif dyntype == DynType.MUSCLE:                                            <L 671>
                    var_65 = (var_36 == var_64);
                    if (var_65) {
                        // dynprm = actuator_dynprm[worldid % actuator_dynprm.shape[0], uid]       <L 672>
                        var_66 = &(var_actuator_dynprm.shape);
                        var_69 = wp::load(var_66);
                        var_68 = wp::extract(var_69, var_67);
                        var_70 = wp::mod(var_0, var_68);
                        var_71 = wp::address(var_actuator_dynprm, var_70, var_1);
                        var_73 = wp::load(var_71);
                        var_72 = wp::copy(var_73);
                        // act = act_in[worldid, act_last]                                        <L 673>
                        var_74 = wp::address(var_act_in, var_0, var_34);
                        var_76 = wp::load(var_74);
                        var_75 = wp::copy(var_76);
                        // act_dot = util_misc.muscle_dynamics(ctrl, act, dynprm)                 <L 674>
                        var_77 = muscle_dynamics_0(var_22, var_75, var_72);
                    }
                    var_78 = wp::where(var_65, var_72, var_44);
                    var_79 = wp::where(var_65, var_77, var_63);
                    var_80 = wp::where(var_65, var_75, var_55);
                    if (!var_65) {
                        // elif dyntype == DynType.USER:                                          <L 675>
                        var_82 = (var_36 == var_81);
                        if (var_82) {
                            // act_dot = 0.0  # set by act_dyn_callback                           <L 676>
                        }
                        var_84 = wp::where(var_82, var_83, var_79);
                        if (!var_82) {
                            // act_dot = 0.0                                                      <L 678>
                        }
                        var_86 = wp::where(var_82, var_84, var_85);
                    }
                    var_87 = wp::where(var_65, var_79, var_86);
                }
                var_88 = wp::where(var_53, var_44, var_78);
                var_89 = wp::where(var_53, var_63, var_87);
                var_90 = wp::where(var_53, var_55, var_80);
            }
            var_91 = wp::where(var_47, var_44, var_88);
            var_92 = wp::where(var_47, var_48, var_89);
            // act_dot_out[worldid, act_last] = act_dot                                           <L 680>
            wp::array_store(var_act_dot_out, var_0, var_34, var_92);
            // if actuator_actearly[uid]:                                                         <L 682>
            var_93 = wp::address(var_actuator_actearly, var_1);
            var_94 = wp::load(var_93);
            if (var_94) {
                // if dyntype == DynType.INTEGRATOR or dyntype == DynType.NONE:                   <L 683>
                var_96 = (var_36 == var_95);
                var_98 = (var_36 == var_97);
                var_99 = var_96 || var_98;
                if (var_99) {
                    // act = act_in[worldid, act_last]                                            <L 684>
                    var_100 = wp::address(var_act_in, var_0, var_34);
                    var_102 = wp::load(var_100);
                    var_101 = wp::copy(var_102);
                }
                var_103 = wp::where(var_99, var_101, var_90);
                // ctrl_act = next_act(                                                           <L 686>
                // opt_timestep[worldid % opt_timestep.shape[0]],                                 <L 687>
                var_104 = &(var_opt_timestep.shape);
                var_107 = wp::load(var_104);
                var_106 = wp::extract(var_107, var_105);
                var_108 = wp::mod(var_0, var_106);
                var_109 = wp::address(var_opt_timestep, var_108);
                // dyntype,                                                                       <L 688>
                // dynprm,                                                                        <L 689>
                // actuator_actrange[worldid % actuator_actrange.shape[0], uid],                  <L 690>
                var_110 = &(var_actuator_actrange.shape);
                var_113 = wp::load(var_110);
                var_112 = wp::extract(var_113, var_111);
                var_114 = wp::mod(var_0, var_112);
                var_115 = wp::address(var_actuator_actrange, var_114, var_1);
                // act,                                                                           <L 691>
                // act_dot,                                                                       <L 692>
                // 1.0,                                                                           <L 693>
                // actuator_actlimited[uid],                                                      <L 694>
                var_117 = wp::address(var_actuator_actlimited, var_1);
                var_119 = wp::load(var_109);
                var_120 = wp::load(var_115);
                var_121 = wp::load(var_117);
                var_118 = next_act_0(var_119, var_36, var_91, var_120, var_103, var_92, var_116, var_121);
            }
            var_122 = wp::load(var_93);
            var_124 = wp::load(var_93);
            var_123 = wp::where(var_124, var_118, var_23);
            var_126 = wp::load(var_93);
            var_125 = wp::where(var_126, var_103, var_90);
            var_127 = wp::load(var_93);
            if (!var_127) {
                // ctrl_act = act_in[worldid, act_last]                                           <L 697>
                var_128 = wp::address(var_act_in, var_0, var_34);
                var_130 = wp::load(var_128);
                var_129 = wp::copy(var_130);
            }
            var_131 = wp::load(var_93);
            var_133 = wp::load(var_93);
            var_132 = wp::where(var_133, var_123, var_129);
        }
        var_134 = wp::where(var_29, var_132, var_23);
        // length = actuator_length_in[worldid, uid]                                              <L 699>
        var_135 = wp::address(var_actuator_length_in, var_0, var_1);
        var_137 = wp::load(var_135);
        var_136 = wp::copy(var_137);
        // velocity = actuator_velocity_in[worldid, uid]                                          <L 700>
        var_138 = wp::address(var_actuator_velocity_in, var_0, var_1);
        var_140 = wp::load(var_138);
        var_139 = wp::copy(var_140);
        // gaintype = actuator_gaintype[uid]                                                      <L 703>
        var_141 = wp::address(var_actuator_gaintype, var_1);
        var_143 = wp::load(var_141);
        var_142 = wp::copy(var_143);
        // gainprm = actuator_gainprm[worldid % actuator_gainprm.shape[0], uid]                   <L 704>
        var_144 = &(var_actuator_gainprm.shape);
        var_147 = wp::load(var_144);
        var_146 = wp::extract(var_147, var_145);
        var_148 = wp::mod(var_0, var_146);
        var_149 = wp::address(var_actuator_gainprm, var_148, var_1);
        var_151 = wp::load(var_149);
        var_150 = wp::copy(var_151);
        // gain = 0.0                                                                             <L 706>
        // if gaintype == GainType.FIXED:                                                         <L 707>
        var_154 = (var_142 == var_153);
        if (var_154) {
            // gain = gainprm[0]                                                                  <L 708>
            var_156 = wp::extract(var_150, var_155);
        }
        var_157 = wp::where(var_154, var_156, var_152);
        if (!var_154) {
            // elif gaintype == GainType.AFFINE:                                                  <L 709>
            var_159 = (var_142 == var_158);
            if (var_159) {
                // gain = gainprm[0] + gainprm[1] * length + gainprm[2] * velocity                <L 710>
                var_161 = wp::extract(var_150, var_160);
                var_163 = wp::extract(var_150, var_162);
                var_164 = wp::mul(var_163, var_136);
                var_165 = wp::add(var_161, var_164);
                var_167 = wp::extract(var_150, var_166);
                var_168 = wp::mul(var_167, var_139);
                var_169 = wp::add(var_165, var_168);
            }
            var_170 = wp::where(var_159, var_169, var_157);
            if (!var_159) {
                // elif gaintype == GainType.MUSCLE:                                              <L 711>
                var_172 = (var_142 == var_171);
                if (var_172) {
                    // acc0 = actuator_acc0[worldid % actuator_acc0.shape[0], uid]                <L 712>
                    var_173 = &(var_actuator_acc0.shape);
                    var_176 = wp::load(var_173);
                    var_175 = wp::extract(var_176, var_174);
                    var_177 = wp::mod(var_0, var_175);
                    var_178 = wp::address(var_actuator_acc0, var_177, var_1);
                    var_180 = wp::load(var_178);
                    var_179 = wp::copy(var_180);
                    // lengthrange = actuator_lengthrange[worldid % actuator_lengthrange.shape[0], uid]       <L 713>
                    var_181 = &(var_actuator_lengthrange.shape);
                    var_184 = wp::load(var_181);
                    var_183 = wp::extract(var_184, var_182);
                    var_185 = wp::mod(var_0, var_183);
                    var_186 = wp::address(var_actuator_lengthrange, var_185, var_1);
                    var_188 = wp::load(var_186);
                    var_187 = wp::copy(var_188);
                    // gain = util_misc.muscle_gain(length, velocity, lengthrange, acc0, gainprm)       <L 714>
                    var_189 = muscle_gain_0(var_136, var_139, var_187, var_179, var_150);
                }
                var_190 = wp::where(var_172, var_189, var_170);
            }
            var_191 = wp::where(var_159, var_170, var_190);
        }
        var_192 = wp::where(var_154, var_157, var_191);
        // biastype = actuator_biastype[uid]                                                      <L 718>
        var_193 = wp::address(var_actuator_biastype, var_1);
        var_195 = wp::load(var_193);
        var_194 = wp::copy(var_195);
        // biasprm = actuator_biasprm[worldid % actuator_biasprm.shape[0], uid]                   <L 719>
        var_196 = &(var_actuator_biasprm.shape);
        var_199 = wp::load(var_196);
        var_198 = wp::extract(var_199, var_197);
        var_200 = wp::mod(var_0, var_198);
        var_201 = wp::address(var_actuator_biasprm, var_200, var_1);
        var_203 = wp::load(var_201);
        var_202 = wp::copy(var_203);
        // bias = 0.0  # BiasType.NONE or BiasType.USER (modified by act_bias_callback)           <L 721>
        // if biastype == BiasType.AFFINE:                                                        <L 722>
        var_206 = (var_194 == var_205);
        if (var_206) {
            // bias = biasprm[0] + biasprm[1] * length + biasprm[2] * velocity                    <L 723>
            var_208 = wp::extract(var_202, var_207);
            var_210 = wp::extract(var_202, var_209);
            var_211 = wp::mul(var_210, var_136);
            var_212 = wp::add(var_208, var_211);
            var_214 = wp::extract(var_202, var_213);
            var_215 = wp::mul(var_214, var_139);
            var_216 = wp::add(var_212, var_215);
        }
        var_217 = wp::where(var_206, var_216, var_204);
        if (!var_206) {
            // elif biastype == BiasType.MUSCLE:                                                  <L 724>
            var_219 = (var_194 == var_218);
            if (var_219) {
                // acc0 = actuator_acc0[worldid % actuator_acc0.shape[0], uid]                    <L 725>
                var_220 = &(var_actuator_acc0.shape);
                var_223 = wp::load(var_220);
                var_222 = wp::extract(var_223, var_221);
                var_224 = wp::mod(var_0, var_222);
                var_225 = wp::address(var_actuator_acc0, var_224, var_1);
                var_227 = wp::load(var_225);
                var_226 = wp::copy(var_227);
                // lengthrange = actuator_lengthrange[worldid % actuator_lengthrange.shape[0], uid]       <L 726>
                var_228 = &(var_actuator_lengthrange.shape);
                var_231 = wp::load(var_228);
                var_230 = wp::extract(var_231, var_229);
                var_232 = wp::mod(var_0, var_230);
                var_233 = wp::address(var_actuator_lengthrange, var_232, var_1);
                var_235 = wp::load(var_233);
                var_234 = wp::copy(var_235);
                // bias = util_misc.muscle_bias(length, lengthrange, acc0, biasprm)               <L 727>
                var_236 = muscle_bias_0(var_136, var_234, var_226, var_202);
            }
            var_237 = wp::where(var_219, var_226, var_179);
            var_238 = wp::where(var_219, var_234, var_187);
            var_239 = wp::where(var_219, var_236, var_217);
        }
        var_240 = wp::where(var_206, var_179, var_237);
        var_241 = wp::where(var_206, var_187, var_238);
        var_242 = wp::where(var_206, var_217, var_239);
        // force = gain * ctrl_act + bias                                                         <L 729>
        var_243 = wp::mul(var_192, var_134);
        var_244 = wp::add(var_243, var_242);
        // if actuator_forcelimited[uid]:                                                         <L 731>
        var_245 = wp::address(var_actuator_forcelimited, var_1);
        var_246 = wp::load(var_245);
        if (var_246) {
            // forcerange = actuator_forcerange[worldid % actuator_forcerange.shape[0], uid]       <L 732>
            var_247 = &(var_actuator_forcerange.shape);
            var_250 = wp::load(var_247);
            var_249 = wp::extract(var_250, var_248);
            var_251 = wp::mod(var_0, var_249);
            var_252 = wp::address(var_actuator_forcerange, var_251, var_1);
            var_254 = wp::load(var_252);
            var_253 = wp::copy(var_254);
            // force = wp.clamp(force, forcerange[0], forcerange[1])                              <L 733>
            var_256 = wp::extract(var_253, var_255);
            var_258 = wp::extract(var_253, var_257);
            var_259 = wp::clamp(var_244, var_256, var_258);
        }
        var_260 = wp::load(var_245);
        var_262 = wp::load(var_245);
        var_261 = wp::where(var_262, var_259, var_244);
        // actuator_force_out[worldid, uid] = force                                               <L 735>
        wp::array_store(var_actuator_force_out, var_0, var_1, var_261);
    }
}



extern "C" __global__ void _qfrc_actuator_gravcomp_limits_4010aec1_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_ngravcomp,
    wp::array_t<bool> var_jnt_actfrclimited,
    wp::array_t<wp::int32> var_jnt_actgravcomp,
    wp::array_t<wp::vec_t<2, wp::float32>> var_jnt_actfrcrange,
    wp::array_t<wp::int32> var_dof_jntid,
    wp::array_t<wp::float32> var_qfrc_gravcomp_in,
    wp::array_t<wp::float32> var_qfrc_actuator_in,
    wp::array_t<wp::float32> var_qfrc_actuator_out)
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
        wp::float32* var_5;
        wp::float32 var_6;
        wp::float32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        bool var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        bool* var_15;
        bool var_16;
        wp::shape_t* var_17;
        const wp::int32 var_18 = 0;
        wp::int32 var_19;
        wp::shape_t var_20;
        wp::int32 var_21;
        wp::vec_t<2, wp::float32>* var_22;
        wp::vec_t<2, wp::float32> var_23;
        wp::vec_t<2, wp::float32> var_24;
        const wp::int32 var_25 = 0;
        wp::float32 var_26;
        const wp::int32 var_27 = 1;
        wp::float32 var_28;
        wp::float32 var_29;
        bool var_30;
        wp::float32 var_31;
        bool var_32;
        //---------
        // forward
        // def _qfrc_actuator_gravcomp_limits(                                                    <L 806>
        // worldid, dofid = wp.tid()                                                              <L 819>
        builtin_tid2d(var_0, var_1);
        // jntid = dof_jntid[dofid]                                                               <L 820>
        var_2 = wp::address(var_dof_jntid, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // qfrc = qfrc_actuator_in[worldid, dofid]                                                <L 822>
        var_5 = wp::address(var_qfrc_actuator_in, var_0, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if ngravcomp and jnt_actgravcomp[jntid]:                                               <L 825>
        var_8 = wp::address(var_jnt_actgravcomp, var_3);
        var_9 = wp::load(var_8);
        var_10 = var_ngravcomp && var_9;
        if (var_10) {
            // qfrc += qfrc_gravcomp_in[worldid, dofid]                                           <L 826>
            var_11 = wp::address(var_qfrc_gravcomp_in, var_0, var_1);
            var_13 = wp::load(var_11);
            var_12 = wp::add(var_6, var_13);
        }
        var_14 = wp::where(var_10, var_12, var_6);
        // if jnt_actfrclimited[jntid]:                                                           <L 829>
        var_15 = wp::address(var_jnt_actfrclimited, var_3);
        var_16 = wp::load(var_15);
        if (var_16) {
            // frcrange = jnt_actfrcrange[worldid % jnt_actfrcrange.shape[0], jntid]              <L 830>
            var_17 = &(var_jnt_actfrcrange.shape);
            var_20 = wp::load(var_17);
            var_19 = wp::extract(var_20, var_18);
            var_21 = wp::mod(var_0, var_19);
            var_22 = wp::address(var_jnt_actfrcrange, var_21, var_3);
            var_24 = wp::load(var_22);
            var_23 = wp::copy(var_24);
            // qfrc = wp.clamp(qfrc, frcrange[0], frcrange[1])                                    <L 831>
            var_26 = wp::extract(var_23, var_25);
            var_28 = wp::extract(var_23, var_27);
            var_29 = wp::clamp(var_14, var_26, var_28);
        }
        var_30 = wp::load(var_15);
        var_32 = wp::load(var_15);
        var_31 = wp::where(var_32, var_29, var_14);
        // qfrc_actuator_out[worldid, dofid] = qfrc                                               <L 833>
        wp::array_store(var_qfrc_actuator_out, var_0, var_1, var_31);
    }
}



extern "C" __global__ void _qfrc_smooth_ad05eebd_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_qfrc_applied_in,
    wp::array_t<wp::float32> var_qfrc_bias_in,
    wp::array_t<wp::float32> var_qfrc_passive_in,
    wp::array_t<wp::float32> var_qfrc_actuator_in,
    wp::array_t<wp::float32> var_qfrc_smooth_out)
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
        wp::float32* var_3;
        wp::float32 var_4;
        wp::float32 var_5;
        wp::float32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        //---------
        // forward
        // def _qfrc_smooth(                                                                      <L 931>
        // worldid, dofid = wp.tid()                                                              <L 940>
        builtin_tid2d(var_0, var_1);
        // qfrc_smooth_out[worldid, dofid] = (                                                    <L 941>
        // qfrc_passive_in[worldid, dofid]                                                        <L 942>
        var_2 = wp::address(var_qfrc_passive_in, var_0, var_1);
        // - qfrc_bias_in[worldid, dofid]                                                         <L 943>
        var_3 = wp::address(var_qfrc_bias_in, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = wp::sub(var_5, var_6);
        // + qfrc_actuator_in[worldid, dofid]                                                     <L 944>
        var_7 = wp::address(var_qfrc_actuator_in, var_0, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::add(var_4, var_9);
        // + qfrc_applied_in[worldid, dofid]                                                      <L 945>
        var_10 = wp::address(var_qfrc_applied_in, var_0, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::add(var_8, var_12);
        // qfrc_smooth_out[worldid, dofid] = (                                                    <L 941>
        wp::array_store(var_qfrc_smooth_out, var_0, var_1, var_11);
    }
}



extern "C" __global__ void _actuator_velocity_00a052f5_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::int32> var_moment_rownnz_in,
    wp::array_t<wp::int32> var_moment_rowadr_in,
    wp::array_t<wp::int32> var_moment_colind_in,
    wp::array_t<wp::float32> var_actuator_moment_in,
    wp::array_t<wp::float32> var_actuator_velocity_out)
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
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::float32* var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        //---------
        // forward
        // def _actuator_velocity(                                                                <L 541>
        // worldid, actid = wp.tid()                                                              <L 551>
        builtin_tid2d(var_0, var_1);
        // rownnz = moment_rownnz_in[worldid, actid]                                              <L 553>
        var_2 = wp::address(var_moment_rownnz_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // rowadr = moment_rowadr_in[worldid, actid]                                              <L 554>
        var_5 = wp::address(var_moment_rowadr_in, var_0, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // vel = float(0.0)                                                                       <L 556>
        var_9 = wp::float(var_8);
        // for i in range(rownnz):                                                                <L 557>
        var_10 = wp::range(var_3);
        start_for_0:;
            if (iter_cmp(var_10) == 0) goto end_for_0;
            var_11 = wp::iter_next(var_10);
            // sparseid = rowadr + i                                                              <L 558>
            var_12 = wp::add(var_6, var_11);
            // colind = moment_colind_in[worldid, sparseid]                                       <L 559>
            var_13 = wp::address(var_moment_colind_in, var_0, var_12);
            var_15 = wp::load(var_13);
            var_14 = wp::copy(var_15);
            // vel += actuator_moment_in[worldid, sparseid] * qvel_in[worldid, colind]            <L 560>
            var_16 = wp::address(var_actuator_moment_in, var_0, var_12);
            var_17 = wp::address(var_qvel_in, var_0, var_14);
            var_19 = wp::load(var_16);
            var_20 = wp::load(var_17);
            var_18 = wp::mul(var_19, var_20);
            var_21 = wp::add(var_9, var_18);
            wp::assign(var_9, var_21);
            goto start_for_0;
        end_for_0:;
        // actuator_velocity_out[worldid, actid] = vel                                            <L 562>
        wp::array_store(var_actuator_velocity_out, var_0, var_1, var_9);
    }
}



extern "C" __global__ void _euler_damp_qfrc_sparse_86f1e571_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::int32> var_dof_Madr,
    wp::array_t<wp::float32> var_dof_damping,
    wp::array_t<wp::float32> var_qM_integration_out)
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
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::shape_t* var_13;
        const wp::int32 var_14 = 0;
        wp::int32 var_15;
        wp::shape_t var_16;
        wp::int32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        const wp::int32 var_21 = 0;
        wp::float32 var_22;
        //---------
        // forward
        // def _euler_damp_qfrc_sparse(                                                           <L 278>
        // worldid, tid = wp.tid()                                                                <L 286>
        builtin_tid2d(var_0, var_1);
        // timestep = opt_timestep[worldid % opt_timestep.shape[0]]                               <L 287>
        var_2 = &(var_opt_timestep.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_opt_timestep, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // adr = dof_Madr[tid]                                                                    <L 289>
        var_10 = wp::address(var_dof_Madr, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // qM_integration_out[worldid, 0, adr] += timestep * dof_damping[worldid % dof_damping.shape[0], tid]       <L 290>
        var_13 = &(var_dof_damping.shape);
        var_16 = wp::load(var_13);
        var_15 = wp::extract(var_16, var_14);
        var_17 = wp::mod(var_0, var_15);
        var_18 = wp::address(var_dof_damping, var_17, var_1);
        var_20 = wp::load(var_18);
        var_19 = wp::mul(var_8, var_20);
        var_22 = wp::atomic_add(var_qM_integration_out, var_0, var_21, var_11, var_19);
    }
}



extern "C" __global__ void _tendon_velocity_ca816a63_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::array_t<wp::float32> var_ten_velocity_out)
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
        wp::float32 var_3;
        wp::int32* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::range_t var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        const wp::float32 var_16 = 0.0;
        bool var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::float32* var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        //---------
        // forward
        // def _tendon_velocity(                                                                  <L 566>
        // worldid, tenid = wp.tid()                                                              <L 577>
        builtin_tid2d(var_0, var_1);
        // velocity = float(0.0)                                                                  <L 579>
        var_3 = wp::float(var_2);
        // rownnz = ten_J_rownnz[tenid]                                                           <L 580>
        var_4 = wp::address(var_ten_J_rownnz, var_1);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // rowadr = ten_J_rowadr[tenid]                                                           <L 581>
        var_7 = wp::address(var_ten_J_rowadr, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // for i in range(rownnz):                                                                <L 582>
        var_10 = wp::range(var_5);
        start_for_0:;
            if (iter_cmp(var_10) == 0) goto end_for_0;
            var_11 = wp::iter_next(var_10);
            // sparseid = rowadr + i                                                              <L 583>
            var_12 = wp::add(var_8, var_11);
            // J = ten_J_in[worldid, sparseid]                                                    <L 584>
            var_13 = wp::address(var_ten_J_in, var_0, var_12);
            var_15 = wp::load(var_13);
            var_14 = wp::copy(var_15);
            // if J != 0.0:                                                                       <L 585>
            var_17 = (var_14 != var_16);
            if (var_17) {
                // colind = ten_J_colind[sparseid]                                                <L 586>
                var_18 = wp::address(var_ten_J_colind, var_12);
                var_20 = wp::load(var_18);
                var_19 = wp::copy(var_20);
                // velocity += J * qvel_in[worldid, colind]                                       <L 587>
                var_21 = wp::address(var_qvel_in, var_0, var_19);
                var_23 = wp::load(var_21);
                var_22 = wp::mul(var_14, var_23);
                var_24 = wp::add(var_3, var_22);
            }
            var_25 = wp::where(var_17, var_24, var_3);
            wp::assign(var_3, var_25);
            goto start_for_0;
        end_for_0:;
        // ten_velocity_out[worldid, tenid] = velocity                                            <L 589>
        wp::array_store(var_ten_velocity_out, var_0, var_1, var_3);
    }
}



extern "C" __global__ void _tendon_actuator_force_2da6552b_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_actuator_trntype,
    wp::array_t<wp::vec_t<2, wp::int32>> var_actuator_trnid,
    wp::array_t<wp::float32> var_actuator_force_in,
    wp::array_t<wp::float32> var_ten_actfrc_out)
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
        const wp::int32 var_3 = 3;
        bool var_4;
        wp::int32 var_5;
        wp::vec_t<2, wp::int32>* var_6;
        const wp::int32 var_7 = 0;
        wp::int32 var_8;
        wp::vec_t<2, wp::int32> var_9;
        wp::slice_t var_10;
        const wp::int32 var_11 = 0;
        wp::array_t<wp::float32> var_12;
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        //---------
        // forward
        // def _tendon_actuator_force(                                                            <L 739>
        // worldid, actid = wp.tid()                                                              <L 748>
        builtin_tid2d(var_0, var_1);
        // if actuator_trntype[actid] == TrnType.TENDON:                                          <L 750>
        var_2 = wp::address(var_actuator_trntype, var_1);
        var_5 = wp::load(var_2);
        var_4 = (var_5 == var_3);
        if (var_4) {
            // tenid = actuator_trnid[actid][0]                                                   <L 751>
            var_6 = wp::address(var_actuator_trnid, var_1);
            var_9 = wp::load(var_6);
            var_8 = wp::extract(var_9, var_7);
            // wp.atomic_add(ten_actfrc_out[worldid], tenid, actuator_force_in[worldid, actid])       <L 753>
            var_10 = wp::slice_t(var_0, var_0, var_11);
            var_12 = wp::view(var_ten_actfrc_out, var_10);
            var_13 = wp::address(var_actuator_force_in, var_0, var_1);
            var_15 = wp::load(var_13);
            var_14 = wp::atomic_add(var_12, var_8, var_15);
        }
    }
}

