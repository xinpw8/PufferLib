
#define WP_TILE_BLOCK_DIM 32
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:902
static CUDA_CALLABLE void linesearch_iterative__locals___syncthreads_0(
    )
{
WP_TILE_SYNC();}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:197
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _eval_pt_0(
    wp::vec_t<3, wp::float32> var_quad,
    wp::float32 var_alpha)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 2;
    wp::float32 var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    const wp::int32 var_4 = 1;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::int32 var_8 = 0;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::float32 var_11 = 2.0;
    wp::float32 var_12;
    const wp::int32 var_13 = 1;
    wp::float32 var_14;
    wp::float32 var_15;
    const wp::float32 var_16 = 2.0;
    const wp::int32 var_17 = 2;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    //---------
    // forward
    // def _eval_pt(quad: wp.vec3, alpha: float) -> wp.vec3:                                  <L 198>
    // aq2 = alpha * quad[2]                                                                  <L 200>
    var_1 = wp::extract(var_quad, var_0);
    var_2 = wp::mul(var_alpha, var_1);
    // return wp.vec3(                                                                        <L 201>
    // alpha * aq2 + alpha * quad[1] + quad[0],                                               <L 202>
    var_3 = wp::mul(var_alpha, var_2);
    var_5 = wp::extract(var_quad, var_4);
    var_6 = wp::mul(var_alpha, var_5);
    var_7 = wp::add(var_3, var_6);
    var_9 = wp::extract(var_quad, var_8);
    var_10 = wp::add(var_7, var_9);
    // 2.0 * aq2 + quad[1],                                                                   <L 203>
    var_12 = wp::mul(var_11, var_2);
    var_14 = wp::extract(var_quad, var_13);
    var_15 = wp::add(var_12, var_14);
    // 2.0 * quad[2],                                                                         <L 204>
    var_18 = wp::extract(var_quad, var_17);
    var_19 = wp::mul(var_16, var_18);
    var_20 = wp::vec_t<3, wp::float32>(var_10, var_15, var_19);
    return var_20;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:262
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _eval_elliptic_0(
    wp::float32 var_impratio_invsqrt,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<3, wp::float32> var_quad,
    wp::vec_t<3, wp::float32> var_quad1,
    wp::vec_t<3, wp::float32> var_quad2,
    wp::float32 var_alpha)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    wp::float32 var_2;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    const wp::int32 var_5 = 1;
    wp::float32 var_6;
    const wp::int32 var_7 = 2;
    wp::float32 var_8;
    const wp::int32 var_9 = 0;
    wp::float32 var_10;
    const wp::int32 var_11 = 1;
    wp::float32 var_12;
    const wp::int32 var_13 = 2;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::float32 var_17 = 2.0;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::float32 var_23 = 0.0;
    bool var_24;
    const wp::float32 var_25 = 0.0;
    bool var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    bool var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    const wp::float32 var_33 = 0.0;
    bool var_34;
    wp::vec_t<3, wp::float32> var_35;
    wp::float32 var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    const wp::float32 var_47 = 0.5;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::float32 var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::float32 var_61;
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
    wp::vec_t<3, wp::float32> var_73;
    const wp::float32 var_74 = 0.0;
    const wp::float32 var_75 = 0.0;
    const wp::float32 var_76 = 0.0;
    wp::vec_t<3, wp::float32> var_77;
    //---------
    // forward
    // def _eval_elliptic(                                                                    <L 263>
    // mu = friction[0] * impratio_invsqrt                                                    <L 272>
    var_1 = wp::extract(var_friction, var_0);
    var_2 = wp::mul(var_1, var_impratio_invsqrt);
    // u0 = quad1[0]                                                                          <L 274>
    var_4 = wp::extract(var_quad1, var_3);
    // v0 = quad1[1]                                                                          <L 275>
    var_6 = wp::extract(var_quad1, var_5);
    // uu = quad1[2]                                                                          <L 276>
    var_8 = wp::extract(var_quad1, var_7);
    // uv = quad2[0]                                                                          <L 277>
    var_10 = wp::extract(var_quad2, var_9);
    // vv = quad2[1]                                                                          <L 278>
    var_12 = wp::extract(var_quad2, var_11);
    // dm = quad2[2]                                                                          <L 279>
    var_14 = wp::extract(var_quad2, var_13);
    // N = u0 + alpha * v0                                                                    <L 282>
    var_15 = wp::mul(var_alpha, var_6);
    var_16 = wp::add(var_4, var_15);
    // Tsqr = uu + alpha * (2.0 * uv + alpha * vv)                                            <L 283>
    var_18 = wp::mul(var_17, var_10);
    var_19 = wp::mul(var_alpha, var_12);
    var_20 = wp::add(var_18, var_19);
    var_21 = wp::mul(var_alpha, var_20);
    var_22 = wp::add(var_8, var_21);
    // if Tsqr <= 0.0:                                                                        <L 286>
    var_24 = (var_22 <= var_23);
    if (var_24) {
        // if N < 0.0:                                                                        <L 288>
        var_26 = (var_16 < var_25);
        if (var_26) {
            // return _eval_pt(quad, alpha)                                                   <L 289>
            var_27 = _eval_pt_0(var_quad, var_alpha);
            return var_27;
        }
    }
    if (!var_24) {
        // T = wp.sqrt(Tsqr)                                                                  <L 295>
        var_28 = wp::sqrt(var_22);
        // if N >= mu * T:                                                                    <L 298>
        var_29 = wp::mul(var_2, var_28);
        var_30 = (var_16 >= var_29);
        if (var_30) {
            // pass                                                                           <L 300>
        }
        if (!var_30) {
            // elif mu * N + T <= 0.0:                                                        <L 302>
            var_31 = wp::mul(var_2, var_16);
            var_32 = wp::add(var_31, var_28);
            var_34 = (var_32 <= var_33);
            if (var_34) {
                // return _eval_pt(quad, alpha)                                               <L 303>
                var_35 = _eval_pt_0(var_quad, var_alpha);
                return var_35;
            }
            if (!var_34) {
                // N1 = v0                                                                    <L 308>
                var_36 = wp::copy(var_6);
                // T1 = (uv + alpha * vv) / T                                                 <L 309>
                var_37 = wp::mul(var_alpha, var_12);
                var_38 = wp::add(var_10, var_37);
                var_39 = wp::div(var_38, var_28);
                // T2 = vv / T - (uv + alpha * vv) * T1 / (T * T)                             <L 310>
                var_40 = wp::div(var_12, var_28);
                var_41 = wp::mul(var_alpha, var_12);
                var_42 = wp::add(var_10, var_41);
                var_43 = wp::mul(var_42, var_39);
                var_44 = wp::mul(var_28, var_28);
                var_45 = wp::div(var_43, var_44);
                var_46 = wp::sub(var_40, var_45);
                // cost = wp.vec3(                                                            <L 313>
                // 0.5 * dm * (N - mu * T) * (N - mu * T),                                    <L 314>
                var_48 = wp::mul(var_47, var_14);
                var_49 = wp::mul(var_2, var_28);
                var_50 = wp::sub(var_16, var_49);
                var_51 = wp::mul(var_48, var_50);
                var_52 = wp::mul(var_2, var_28);
                var_53 = wp::sub(var_16, var_52);
                var_54 = wp::mul(var_51, var_53);
                // dm * (N - mu * T) * (N1 - mu * T1),                                        <L 315>
                var_55 = wp::mul(var_2, var_28);
                var_56 = wp::sub(var_16, var_55);
                var_57 = wp::mul(var_14, var_56);
                var_58 = wp::mul(var_2, var_39);
                var_59 = wp::sub(var_36, var_58);
                var_60 = wp::mul(var_57, var_59);
                // dm * ((N1 - mu * T1) * (N1 - mu * T1) + (N - mu * T) * (-mu * T2)),        <L 316>
                var_61 = wp::mul(var_2, var_39);
                var_62 = wp::sub(var_36, var_61);
                var_63 = wp::mul(var_2, var_39);
                var_64 = wp::sub(var_36, var_63);
                var_65 = wp::mul(var_62, var_64);
                var_66 = wp::mul(var_2, var_28);
                var_67 = wp::sub(var_16, var_66);
                var_68 = wp::neg(var_2);
                var_69 = wp::mul(var_68, var_46);
                var_70 = wp::mul(var_67, var_69);
                var_71 = wp::add(var_65, var_70);
                var_72 = wp::mul(var_14, var_71);
                var_73 = wp::vec_t<3, wp::float32>(var_54, var_60, var_72);
                // return cost                                                                <L 319>
                return var_73;
            }
        }
    }
    // return wp.vec3(0.0, 0.0, 0.0)                                                          <L 321>
    var_77 = wp::vec_t<3, wp::float32>(var_74, var_75, var_76);
    return var_77;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:167
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _eval_pt_direct_alpha_zero_0(
    wp::float32 var_jaref,
    wp::float32 var_jv,
    wp::float32 var_d)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 0.5;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::vec_t<3, wp::float32> var_7;
    //---------
    // forward
    // def _eval_pt_direct_alpha_zero(jaref: float, jv: float, d: float) -> wp.vec3:          <L 168>
    // jvD = jv * d                                                                           <L 170>
    var_0 = wp::mul(var_jv, var_d);
    // return wp.vec3(0.5 * d * jaref * jaref, jvD * jaref, jv * jvD)                         <L 171>
    var_2 = wp::mul(var_1, var_d);
    var_3 = wp::mul(var_2, var_jaref);
    var_4 = wp::mul(var_3, var_jaref);
    var_5 = wp::mul(var_0, var_jaref);
    var_6 = wp::mul(var_jv, var_0);
    var_7 = wp::vec_t<3, wp::float32>(var_4, var_5, var_6);
    return var_7;
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:223
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _eval_frictionloss_pt_0(
    wp::float32 var_x,
    wp::float32 var_f,
    wp::float32 var_rf,
    wp::float32 var_jv,
    wp::float32 var_d)
{
    //---------
    // primal vars
    wp::float32 var_0;
    bool var_1;
    bool var_2;
    bool var_3;
    wp::float32 var_4;
    const wp::float32 var_5 = 0.5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::float32 var_12;
    bool var_13;
    const wp::float32 var_14 = 0.5;
    const wp::float32 var_15 = -0.5;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    const wp::float32 var_21 = 0.0;
    wp::vec_t<3, wp::float32> var_22;
    const wp::float32 var_23 = 0.5;
    const wp::float32 var_24 = -0.5;
    wp::float32 var_25;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    const wp::float32 var_29 = 0.0;
    wp::vec_t<3, wp::float32> var_30;
    //---------
    // forward
    // def _eval_frictionloss_pt(x: float, f: float, rf: float, jv: float, d: float) -> wp.vec3:       <L 224>
    // if (-rf < x) and (x < rf):                                                             <L 226>
    var_0 = wp::neg(var_rf);
    var_1 = (var_0 < var_x);
    var_2 = (var_x < var_rf);
    var_3 = var_1 && var_2;
    if (var_3) {
        // jvD = jv * d                                                                       <L 227>
        var_4 = wp::mul(var_jv, var_d);
        // return wp.vec3(0.5 * d * x * x, jvD * x, jv * jvD)                                 <L 228>
        var_6 = wp::mul(var_5, var_d);
        var_7 = wp::mul(var_6, var_x);
        var_8 = wp::mul(var_7, var_x);
        var_9 = wp::mul(var_4, var_x);
        var_10 = wp::mul(var_jv, var_4);
        var_11 = wp::vec_t<3, wp::float32>(var_8, var_9, var_10);
        return var_11;
    }
    if (!var_3) {
        // elif x <= -rf:                                                                     <L 229>
        var_12 = wp::neg(var_rf);
        var_13 = (var_x <= var_12);
        if (var_13) {
            // return wp.vec3(f * (-0.5 * rf - x), -f * jv, 0.0)                              <L 230>
            var_16 = wp::mul(var_15, var_rf);
            var_17 = wp::sub(var_16, var_x);
            var_18 = wp::mul(var_f, var_17);
            var_19 = wp::neg(var_f);
            var_20 = wp::mul(var_19, var_jv);
            var_22 = wp::vec_t<3, wp::float32>(var_18, var_20, var_21);
            return var_22;
        }
        if (!var_13) {
            // return wp.vec3(f * (-0.5 * rf + x), f * jv, 0.0)                               <L 232>
            var_25 = wp::mul(var_24, var_rf);
            var_26 = wp::add(var_25, var_x);
            var_27 = wp::mul(var_f, var_26);
            var_28 = wp::mul(var_f, var_jv);
            var_30 = wp::vec_t<3, wp::float32>(var_27, var_28, var_29);
            return var_30;
        }
    }
    return {};
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:676
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _compute_efc_eval_pt_alpha_zero_elliptic_0(
    wp::int32 var_efcid,
    wp::int32 var_ne,
    wp::int32 var_nf,
    wp::float32 var_impratio_invsqrt,
    wp::int32 var_efc_type,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::float32> var_efc_frictionloss,
    wp::float32 var_ctx_Jaref,
    wp::float32 var_ctx_jv,
    wp::vec_t<3, wp::float32> var_ctx_quad,
    wp::vec_t<5, wp::float32> var_contact_friction,
    wp::int32 var_efc_address0,
    wp::vec_t<3, wp::float32> var_quad1,
    wp::vec_t<3, wp::float32> var_quad2)
{
    //---------
    // primal vars
    wp::int32 var_0;
    bool var_1;
    const wp::int32 var_2 = 7;
    bool var_3;
    bool var_4;
    const wp::float32 var_5 = 0.0;
    wp::vec_t<3, wp::float32> var_6;
    const wp::float32 var_7 = 0.0;
    wp::vec_t<3, wp::float32> var_8;
    const wp::float32 var_9 = 0.0;
    bool var_10;
    wp::float32* var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::float32 var_13;
    const wp::float32 var_14 = 0.0;
    wp::vec_t<3, wp::float32> var_15;
    bool var_16;
    wp::float32* var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32* var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::float32* var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::float32 var_27;
    //---------
    // forward
    // def _compute_efc_eval_pt_alpha_zero_elliptic(                                          <L 677>
    // if efcid >= ne + nf:                                                                   <L 697>
    var_0 = wp::add(var_ne, var_nf);
    var_1 = (var_efcid >= var_0);
    if (var_1) {
        // if efc_type == types.ConstraintType.CONTACT_ELLIPTIC:                              <L 699>
        var_3 = (var_efc_type == var_2);
        if (var_3) {
            // if efcid != efc_address0:  # Not primary row                                   <L 700>
            var_4 = (var_efcid != var_efc_address0);
            if (var_4) {
                // return wp.vec3(0.0)                                                        <L 701>
                var_6 = wp::vec_t<3, wp::float32>(var_5);
                return var_6;
            }
            // return _eval_elliptic(impratio_invsqrt, contact_friction, ctx_quad, quad1, quad2, 0.0)       <L 702>
            var_8 = _eval_elliptic_0(var_impratio_invsqrt, var_contact_friction, var_ctx_quad, var_quad1, var_quad2, var_7);
            return var_8;
        }
        // if ctx_Jaref < 0.0:                                                                <L 705>
        var_10 = (var_ctx_Jaref < var_9);
        if (var_10) {
            // return _eval_pt_direct_alpha_zero(ctx_Jaref, ctx_jv, efc_D_in[efcid])          <L 706>
            var_11 = wp::address(var_efc_D_in, var_efcid);
            var_13 = wp::load(var_11);
            var_12 = _eval_pt_direct_alpha_zero_0(var_ctx_Jaref, var_ctx_jv, var_13);
            return var_12;
        }
        // return wp.vec3(0.0)                                                                <L 707>
        var_15 = wp::vec_t<3, wp::float32>(var_14);
        return var_15;
    }
    // if efcid >= ne:                                                                        <L 710>
    var_16 = (var_efcid >= var_ne);
    if (var_16) {
        // efc_D = efc_D_in[efcid]                                                            <L 711>
        var_17 = wp::address(var_efc_D_in, var_efcid);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // f = efc_frictionloss[efcid]                                                        <L 712>
        var_20 = wp::address(var_efc_frictionloss, var_efcid);
        var_22 = wp::load(var_20);
        var_21 = wp::copy(var_22);
        // rf = math.safe_div(f, efc_D)                                                       <L 713>
        var_23 = safe_div_0(var_21, var_18);
        // return _eval_frictionloss_pt(ctx_Jaref, f, rf, ctx_jv, efc_D)                      <L 714>
        var_24 = _eval_frictionloss_pt_0(var_ctx_Jaref, var_21, var_23, var_ctx_jv, var_18);
        return var_24;
    }
    // return _eval_pt_direct_alpha_zero(ctx_Jaref, ctx_jv, efc_D_in[efcid])                  <L 717>
    var_25 = wp::address(var_efc_D_in, var_efcid);
    var_27 = wp::load(var_25);
    var_26 = _eval_pt_direct_alpha_zero_0(var_ctx_Jaref, var_ctx_jv, var_27);
    return var_26;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:159
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _eval_pt_direct_0(
    wp::float32 var_jaref,
    wp::float32 var_jv,
    wp::float32 var_d,
    wp::float32 var_alpha)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    const wp::float32 var_3 = 0.5;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::vec_t<3, wp::float32> var_9;
    //---------
    // forward
    // def _eval_pt_direct(jaref: float, jv: float, d: float, alpha: float) -> wp.vec3:       <L 160>
    // x = jaref + alpha * jv                                                                 <L 162>
    var_0 = wp::mul(var_alpha, var_jv);
    var_1 = wp::add(var_jaref, var_0);
    // jvD = jv * d                                                                           <L 163>
    var_2 = wp::mul(var_jv, var_d);
    // return wp.vec3(0.5 * d * x * x, jvD * x, jv * jvD)                                     <L 164>
    var_4 = wp::mul(var_3, var_d);
    var_5 = wp::mul(var_4, var_1);
    var_6 = wp::mul(var_5, var_1);
    var_7 = wp::mul(var_2, var_1);
    var_8 = wp::mul(var_jv, var_2);
    var_9 = wp::vec_t<3, wp::float32>(var_6, var_7, var_8);
    return var_9;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:601
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _compute_efc_eval_pt_elliptic_0(
    wp::int32 var_efcid,
    wp::float32 var_alpha,
    wp::int32 var_ne,
    wp::int32 var_nf,
    wp::float32 var_impratio_invsqrt,
    wp::int32 var_efc_type,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::float32> var_efc_frictionloss,
    wp::float32 var_ctx_Jaref,
    wp::float32 var_ctx_jv,
    wp::vec_t<3, wp::float32> var_ctx_quad,
    wp::vec_t<5, wp::float32> var_contact_friction,
    wp::int32 var_efc_address0,
    wp::vec_t<3, wp::float32> var_quad1,
    wp::vec_t<3, wp::float32> var_quad2)
{
    //---------
    // primal vars
    wp::int32 var_0;
    bool var_1;
    const wp::int32 var_2 = 7;
    bool var_3;
    bool var_4;
    const wp::float32 var_5 = 0.0;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    const wp::float32 var_10 = 0.0;
    bool var_11;
    wp::float32* var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::float32 var_14;
    const wp::float32 var_15 = 0.0;
    wp::vec_t<3, wp::float32> var_16;
    bool var_17;
    wp::float32* var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::float32* var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::float32 var_28;
    wp::float32* var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::float32 var_31;
    //---------
    // forward
    // def _compute_efc_eval_pt_elliptic(                                                     <L 602>
    // if efcid >= ne + nf:                                                                   <L 623>
    var_0 = wp::add(var_ne, var_nf);
    var_1 = (var_efcid >= var_0);
    if (var_1) {
        // if efc_type == types.ConstraintType.CONTACT_ELLIPTIC:                              <L 625>
        var_3 = (var_efc_type == var_2);
        if (var_3) {
            // if efcid != efc_address0:  # Not primary row                                   <L 626>
            var_4 = (var_efcid != var_efc_address0);
            if (var_4) {
                // return wp.vec3(0.0)                                                        <L 627>
                var_6 = wp::vec_t<3, wp::float32>(var_5);
                return var_6;
            }
            // return _eval_elliptic(impratio_invsqrt, contact_friction, ctx_quad, quad1, quad2, alpha)       <L 628>
            var_7 = _eval_elliptic_0(var_impratio_invsqrt, var_contact_friction, var_ctx_quad, var_quad1, var_quad2, var_alpha);
            return var_7;
        }
        // x = ctx_Jaref + alpha * ctx_jv                                                     <L 631>
        var_8 = wp::mul(var_alpha, var_ctx_jv);
        var_9 = wp::add(var_ctx_Jaref, var_8);
        // if x < 0.0:                                                                        <L 632>
        var_11 = (var_9 < var_10);
        if (var_11) {
            // return _eval_pt_direct(ctx_Jaref, ctx_jv, efc_D_in[efcid], alpha)              <L 633>
            var_12 = wp::address(var_efc_D_in, var_efcid);
            var_14 = wp::load(var_12);
            var_13 = _eval_pt_direct_0(var_ctx_Jaref, var_ctx_jv, var_14, var_alpha);
            return var_13;
        }
        // return wp.vec3(0.0)                                                                <L 634>
        var_16 = wp::vec_t<3, wp::float32>(var_15);
        return var_16;
    }
    // if efcid >= ne:                                                                        <L 637>
    var_17 = (var_efcid >= var_ne);
    if (var_17) {
        // efc_D = efc_D_in[efcid]                                                            <L 638>
        var_18 = wp::address(var_efc_D_in, var_efcid);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // f = efc_frictionloss[efcid]                                                        <L 639>
        var_21 = wp::address(var_efc_frictionloss, var_efcid);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // x = ctx_Jaref + alpha * ctx_jv                                                     <L 640>
        var_24 = wp::mul(var_alpha, var_ctx_jv);
        var_25 = wp::add(var_ctx_Jaref, var_24);
        // rf = math.safe_div(f, efc_D)                                                       <L 641>
        var_26 = safe_div_0(var_22, var_19);
        // return _eval_frictionloss_pt(x, f, rf, ctx_jv, efc_D)                              <L 642>
        var_27 = _eval_frictionloss_pt_0(var_25, var_22, var_26, var_ctx_jv, var_19);
        return var_27;
    }
    var_28 = wp::where(var_17, var_25, var_9);
    // return _eval_pt_direct(ctx_Jaref, ctx_jv, efc_D_in[efcid], alpha)                      <L 645>
    var_29 = wp::address(var_efc_D_in, var_efcid);
    var_31 = wp::load(var_29);
    var_30 = _eval_pt_direct_0(var_ctx_Jaref, var_ctx_jv, var_31, var_alpha);
    return var_30;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:174
static CUDA_CALLABLE void _eval_pt_direct_3alphas_0(
    wp::float32 var_jaref,
    wp::float32 var_jv,
    wp::float32 var_d,
    wp::float32 var_lo_alpha,
    wp::float32 var_hi_alpha,
    wp::float32 var_mid_alpha,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::float32 var_8 = 0.5;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::vec_t<3, wp::float32> var_21;
    //---------
    // forward
    // def _eval_pt_direct_3alphas(                                                           <L 175>
    // x_lo = jaref + lo_alpha * jv                                                           <L 179>
    var_0 = wp::mul(var_lo_alpha, var_jv);
    var_1 = wp::add(var_jaref, var_0);
    // x_hi = jaref + hi_alpha * jv                                                           <L 180>
    var_2 = wp::mul(var_hi_alpha, var_jv);
    var_3 = wp::add(var_jaref, var_2);
    // x_mid = jaref + mid_alpha * jv                                                         <L 181>
    var_4 = wp::mul(var_mid_alpha, var_jv);
    var_5 = wp::add(var_jaref, var_4);
    // jvD = jv * d                                                                           <L 182>
    var_6 = wp::mul(var_jv, var_d);
    // hessian = jv * jvD                                                                     <L 183>
    var_7 = wp::mul(var_jv, var_6);
    // half_d = 0.5 * d                                                                       <L 184>
    var_9 = wp::mul(var_8, var_d);
    // return (                                                                               <L 185>
    // wp.vec3(half_d * x_lo * x_lo, jvD * x_lo, hessian),                                    <L 186>
    var_10 = wp::mul(var_9, var_1);
    var_11 = wp::mul(var_10, var_1);
    var_12 = wp::mul(var_6, var_1);
    var_13 = wp::vec_t<3, wp::float32>(var_11, var_12, var_7);
    // wp.vec3(half_d * x_hi * x_hi, jvD * x_hi, hessian),                                    <L 187>
    var_14 = wp::mul(var_9, var_3);
    var_15 = wp::mul(var_14, var_3);
    var_16 = wp::mul(var_6, var_3);
    var_17 = wp::vec_t<3, wp::float32>(var_15, var_16, var_7);
    // wp.vec3(half_d * x_mid * x_mid, jvD * x_mid, hessian),                                 <L 188>
    var_18 = wp::mul(var_9, var_5);
    var_19 = wp::mul(var_18, var_5);
    var_20 = wp::mul(var_6, var_5);
    var_21 = wp::vec_t<3, wp::float32>(var_19, var_20, var_7);
    ret_0 = var_13;
    ret_1 = var_17;
    ret_2 = var_21;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:235
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _eval_frictionloss_pt_one_0(
    wp::float32 var_x,
    wp::float32 var_f,
    wp::float32 var_rf,
    wp::float32 var_half_d,
    wp::float32 var_jvD,
    wp::float32 var_hessian,
    wp::float32 var_f_jv)
{
    //---------
    // primal vars
    wp::float32 var_0;
    bool var_1;
    bool var_2;
    bool var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::float32 var_8;
    bool var_9;
    const wp::float32 var_10 = 0.5;
    const wp::float32 var_11 = -0.5;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    const wp::float32 var_16 = 0.0;
    wp::vec_t<3, wp::float32> var_17;
    const wp::float32 var_18 = 0.5;
    const wp::float32 var_19 = -0.5;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::float32 var_23 = 0.0;
    wp::vec_t<3, wp::float32> var_24;
    //---------
    // forward
    // def _eval_frictionloss_pt_one(x: float, f: float, rf: float, half_d: float, jvD: float, hessian: float, f_jv: float) -> wp.vec3:       <L 236>
    // if (-rf < x) and (x < rf):                                                             <L 238>
    var_0 = wp::neg(var_rf);
    var_1 = (var_0 < var_x);
    var_2 = (var_x < var_rf);
    var_3 = var_1 && var_2;
    if (var_3) {
        // return wp.vec3(half_d * x * x, jvD * x, hessian)                                   <L 239>
        var_4 = wp::mul(var_half_d, var_x);
        var_5 = wp::mul(var_4, var_x);
        var_6 = wp::mul(var_jvD, var_x);
        var_7 = wp::vec_t<3, wp::float32>(var_5, var_6, var_hessian);
        return var_7;
    }
    if (!var_3) {
        // elif x <= -rf:                                                                     <L 240>
        var_8 = wp::neg(var_rf);
        var_9 = (var_x <= var_8);
        if (var_9) {
            // return wp.vec3(f * (-0.5 * rf - x), -f_jv, 0.0)                                <L 241>
            var_12 = wp::mul(var_11, var_rf);
            var_13 = wp::sub(var_12, var_x);
            var_14 = wp::mul(var_f, var_13);
            var_15 = wp::neg(var_f_jv);
            var_17 = wp::vec_t<3, wp::float32>(var_14, var_15, var_16);
            return var_17;
        }
        if (!var_9) {
            // return wp.vec3(f * (-0.5 * rf + x), f_jv, 0.0)                                 <L 243>
            var_20 = wp::mul(var_19, var_rf);
            var_21 = wp::add(var_20, var_x);
            var_22 = wp::mul(var_f, var_21);
            var_24 = wp::vec_t<3, wp::float32>(var_22, var_f_jv, var_23);
            return var_24;
        }
    }
    return {};
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:246
static CUDA_CALLABLE void _eval_frictionloss_pt_3alphas_0(
    wp::float32 var_x_lo,
    wp::float32 var_x_hi,
    wp::float32 var_x_mid,
    wp::float32 var_f,
    wp::float32 var_rf,
    wp::float32 var_jv,
    wp::float32 var_d,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 0.5;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    //---------
    // forward
    // def _eval_frictionloss_pt_3alphas(                                                     <L 247>
    // jvD = jv * d                                                                           <L 251>
    var_0 = wp::mul(var_jv, var_d);
    // half_d = 0.5 * d                                                                       <L 252>
    var_2 = wp::mul(var_1, var_d);
    // hessian = jv * jvD                                                                     <L 253>
    var_3 = wp::mul(var_jv, var_0);
    // f_jv = f * jv                                                                          <L 254>
    var_4 = wp::mul(var_f, var_jv);
    // return (                                                                               <L 255>
    // _eval_frictionloss_pt_one(x_lo, f, rf, half_d, jvD, hessian, f_jv),                    <L 256>
    var_5 = _eval_frictionloss_pt_one_0(var_x_lo, var_f, var_rf, var_2, var_0, var_3, var_4);
    // _eval_frictionloss_pt_one(x_hi, f, rf, half_d, jvD, hessian, f_jv),                    <L 257>
    var_6 = _eval_frictionloss_pt_one_0(var_x_hi, var_f, var_rf, var_2, var_0, var_3, var_4);
    // _eval_frictionloss_pt_one(x_mid, f, rf, half_d, jvD, hessian, f_jv),                   <L 258>
    var_7 = _eval_frictionloss_pt_one_0(var_x_mid, var_f, var_rf, var_2, var_0, var_3, var_4);
    ret_0 = var_5;
    ret_1 = var_6;
    ret_2 = var_7;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:763
static CUDA_CALLABLE void _compute_efc_eval_pt_3alphas_elliptic_0(
    wp::int32 var_efcid,
    wp::float32 var_lo_alpha,
    wp::float32 var_hi_alpha,
    wp::float32 var_mid_alpha,
    wp::int32 var_ne,
    wp::int32 var_nf,
    wp::float32 var_impratio_invsqrt,
    wp::int32 var_efc_type,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::float32> var_efc_frictionloss,
    wp::float32 var_ctx_Jaref,
    wp::float32 var_ctx_jv,
    wp::vec_t<3, wp::float32> var_ctx_quad,
    wp::vec_t<5, wp::float32> var_contact_friction,
    wp::int32 var_efc_address0,
    wp::vec_t<3, wp::float32> var_quad1,
    wp::vec_t<3, wp::float32> var_quad2,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::int32 var_6;
    bool var_7;
    const wp::int32 var_8 = 7;
    bool var_9;
    bool var_10;
    const wp::float32 var_11 = 0.0;
    wp::vec_t<3, wp::float32> var_12;
    const wp::float32 var_13 = 0.0;
    wp::vec_t<3, wp::float32> var_14;
    const wp::float32 var_15 = 0.0;
    wp::vec_t<3, wp::float32> var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::float32* var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32> var_25;
    const wp::float32 var_26 = 0.0;
    bool var_27;
    const wp::float32 var_28 = 0.0;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32> var_30;
    const wp::float32 var_31 = 0.0;
    bool var_32;
    const wp::float32 var_33 = 0.0;
    wp::vec_t<3, wp::float32> var_34;
    wp::vec_t<3, wp::float32> var_35;
    const wp::float32 var_36 = 0.0;
    bool var_37;
    const wp::float32 var_38 = 0.0;
    wp::vec_t<3, wp::float32> var_39;
    wp::vec_t<3, wp::float32> var_40;
    bool var_41;
    wp::float32* var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32* var_45;
    wp::float32 var_46;
    wp::float32 var_47;
    wp::float32 var_48;
    wp::vec_t<3, wp::float32> var_49;
    wp::vec_t<3, wp::float32> var_50;
    wp::vec_t<3, wp::float32> var_51;
    wp::float32 var_52;
    wp::float32* var_53;
    wp::vec_t<3, wp::float32> var_54;
    wp::vec_t<3, wp::float32> var_55;
    wp::vec_t<3, wp::float32> var_56;
    wp::float32 var_57;
    //---------
    // forward
    // def _compute_efc_eval_pt_3alphas_elliptic(                                             <L 764>
    // x_lo = ctx_Jaref + lo_alpha * ctx_jv                                                   <L 791>
    var_0 = wp::mul(var_lo_alpha, var_ctx_jv);
    var_1 = wp::add(var_ctx_Jaref, var_0);
    // x_hi = ctx_Jaref + hi_alpha * ctx_jv                                                   <L 792>
    var_2 = wp::mul(var_hi_alpha, var_ctx_jv);
    var_3 = wp::add(var_ctx_Jaref, var_2);
    // x_mid = ctx_Jaref + mid_alpha * ctx_jv                                                 <L 793>
    var_4 = wp::mul(var_mid_alpha, var_ctx_jv);
    var_5 = wp::add(var_ctx_Jaref, var_4);
    // if efcid >= ne + nf:                                                                   <L 796>
    var_6 = wp::add(var_ne, var_nf);
    var_7 = (var_efcid >= var_6);
    if (var_7) {
        // if efc_type == types.ConstraintType.CONTACT_ELLIPTIC:                              <L 798>
        var_9 = (var_efc_type == var_8);
        if (var_9) {
            // if efcid != efc_address0:  # secondary rows contribute nothing                 <L 799>
            var_10 = (var_efcid != var_efc_address0);
            if (var_10) {
                // return (wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0))                          <L 800>
                var_12 = wp::vec_t<3, wp::float32>(var_11);
                var_14 = wp::vec_t<3, wp::float32>(var_13);
                var_16 = wp::vec_t<3, wp::float32>(var_15);
                ret_0 = var_12;
                ret_1 = var_14;
                ret_2 = var_16;
                return;
            }
            // return (                                                                       <L 801>
            // _eval_elliptic(impratio_invsqrt, contact_friction, ctx_quad, quad1, quad2, lo_alpha),       <L 802>
            var_17 = _eval_elliptic_0(var_impratio_invsqrt, var_contact_friction, var_ctx_quad, var_quad1, var_quad2, var_lo_alpha);
            // _eval_elliptic(impratio_invsqrt, contact_friction, ctx_quad, quad1, quad2, hi_alpha),       <L 803>
            var_18 = _eval_elliptic_0(var_impratio_invsqrt, var_contact_friction, var_ctx_quad, var_quad1, var_quad2, var_hi_alpha);
            // _eval_elliptic(impratio_invsqrt, contact_friction, ctx_quad, quad1, quad2, mid_alpha),       <L 804>
            var_19 = _eval_elliptic_0(var_impratio_invsqrt, var_contact_friction, var_ctx_quad, var_quad1, var_quad2, var_mid_alpha);
            ret_0 = var_17;
            ret_1 = var_18;
            ret_2 = var_19;
            return;
        }
        // efc_D = efc_D_in[efcid]                                                            <L 808>
        var_20 = wp::address(var_efc_D_in, var_efcid);
        var_22 = wp::load(var_20);
        var_21 = wp::copy(var_22);
        // pt_lo, pt_hi, pt_mid = _eval_pt_direct_3alphas(ctx_Jaref, ctx_jv, efc_D, lo_alpha, hi_alpha, mid_alpha)       <L 809>
        _eval_pt_direct_3alphas_0(var_ctx_Jaref, var_ctx_jv, var_21, var_lo_alpha, var_hi_alpha, var_mid_alpha, var_23, var_24, var_25);
        // r_lo = wp.where(x_lo < 0.0, pt_lo, wp.vec3(0.0))                                   <L 810>
        var_27 = (var_1 < var_26);
        var_29 = wp::vec_t<3, wp::float32>(var_28);
        var_30 = wp::where(var_27, var_23, var_29);
        // r_hi = wp.where(x_hi < 0.0, pt_hi, wp.vec3(0.0))                                   <L 811>
        var_32 = (var_3 < var_31);
        var_34 = wp::vec_t<3, wp::float32>(var_33);
        var_35 = wp::where(var_32, var_24, var_34);
        // r_mid = wp.where(x_mid < 0.0, pt_mid, wp.vec3(0.0))                                <L 812>
        var_37 = (var_5 < var_36);
        var_39 = wp::vec_t<3, wp::float32>(var_38);
        var_40 = wp::where(var_37, var_25, var_39);
        // return (r_lo, r_hi, r_mid)                                                         <L 813>
        ret_0 = var_30;
        ret_1 = var_35;
        ret_2 = var_40;
        return;
    }
    // if efcid >= ne:                                                                        <L 816>
    var_41 = (var_efcid >= var_ne);
    if (var_41) {
        // efc_D = efc_D_in[efcid]                                                            <L 817>
        var_42 = wp::address(var_efc_D_in, var_efcid);
        var_44 = wp::load(var_42);
        var_43 = wp::copy(var_44);
        // f = efc_frictionloss[efcid]                                                        <L 818>
        var_45 = wp::address(var_efc_frictionloss, var_efcid);
        var_47 = wp::load(var_45);
        var_46 = wp::copy(var_47);
        // rf = math.safe_div(f, efc_D)                                                       <L 819>
        var_48 = safe_div_0(var_46, var_43);
        // return _eval_frictionloss_pt_3alphas(x_lo, x_hi, x_mid, f, rf, ctx_jv, efc_D)       <L 820>
        _eval_frictionloss_pt_3alphas_0(var_1, var_3, var_5, var_46, var_48, var_ctx_jv, var_43, var_49, var_50, var_51);
        ret_0 = var_49;
        ret_1 = var_50;
        ret_2 = var_51;
        return;
    }
    var_52 = wp::where(var_41, var_43, var_21);
    // return _eval_pt_direct_3alphas(ctx_Jaref, ctx_jv, efc_D_in[efcid], lo_alpha, hi_alpha, mid_alpha)       <L 823>
    var_53 = wp::address(var_efc_D_in, var_efcid);
    var_57 = wp::load(var_53);
    _eval_pt_direct_3alphas_0(var_ctx_Jaref, var_ctx_jv, var_57, var_lo_alpha, var_hi_alpha, var_mid_alpha, var_54, var_55, var_56);
    ret_0 = var_54;
    ret_1 = var_55;
    ret_2 = var_56;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:208
static CUDA_CALLABLE void _eval_pt_3alphas_0(
    wp::vec_t<3, wp::float32> var_quad,
    wp::float32 var_lo_alpha,
    wp::float32 var_hi_alpha,
    wp::float32 var_mid_alpha,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 2;
    wp::float32 var_5;
    const wp::float32 var_6 = 2.0;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    const wp::float32 var_15 = 2.0;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::float32 var_23 = 2.0;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    const wp::float32 var_31 = 2.0;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::vec_t<3, wp::float32> var_34;
    //---------
    // forward
    // def _eval_pt_3alphas(quad: wp.vec3, lo_alpha: float, hi_alpha: float, mid_alpha: float) -> tuple[wp.vec3, wp.vec3, wp.vec3]:       <L 209>
    // q0, q1, q2 = quad[0], quad[1], quad[2]                                                 <L 211>
    var_1 = wp::extract(var_quad, var_0);
    var_3 = wp::extract(var_quad, var_2);
    var_5 = wp::extract(var_quad, var_4);
    // hessian = 2.0 * q2                                                                     <L 212>
    var_7 = wp::mul(var_6, var_5);
    // lo_aq2 = lo_alpha * q2                                                                 <L 213>
    var_8 = wp::mul(var_lo_alpha, var_5);
    // hi_aq2 = hi_alpha * q2                                                                 <L 214>
    var_9 = wp::mul(var_hi_alpha, var_5);
    // mid_aq2 = mid_alpha * q2                                                               <L 215>
    var_10 = wp::mul(var_mid_alpha, var_5);
    // return (                                                                               <L 216>
    // wp.vec3(lo_alpha * lo_aq2 + lo_alpha * q1 + q0, 2.0 * lo_aq2 + q1, hessian),           <L 217>
    var_11 = wp::mul(var_lo_alpha, var_8);
    var_12 = wp::mul(var_lo_alpha, var_3);
    var_13 = wp::add(var_11, var_12);
    var_14 = wp::add(var_13, var_1);
    var_16 = wp::mul(var_15, var_8);
    var_17 = wp::add(var_16, var_3);
    var_18 = wp::vec_t<3, wp::float32>(var_14, var_17, var_7);
    // wp.vec3(hi_alpha * hi_aq2 + hi_alpha * q1 + q0, 2.0 * hi_aq2 + q1, hessian),           <L 218>
    var_19 = wp::mul(var_hi_alpha, var_9);
    var_20 = wp::mul(var_hi_alpha, var_3);
    var_21 = wp::add(var_19, var_20);
    var_22 = wp::add(var_21, var_1);
    var_24 = wp::mul(var_23, var_9);
    var_25 = wp::add(var_24, var_3);
    var_26 = wp::vec_t<3, wp::float32>(var_22, var_25, var_7);
    // wp.vec3(mid_alpha * mid_aq2 + mid_alpha * q1 + q0, 2.0 * mid_aq2 + q1, hessian),       <L 219>
    var_27 = wp::mul(var_mid_alpha, var_10);
    var_28 = wp::mul(var_mid_alpha, var_3);
    var_29 = wp::add(var_27, var_28);
    var_30 = wp::add(var_29, var_1);
    var_32 = wp::mul(var_31, var_10);
    var_33 = wp::add(var_32, var_3);
    var_34 = wp::vec_t<3, wp::float32>(var_30, var_33, var_7);
    ret_0 = var_18;
    ret_1 = var_26;
    ret_2 = var_34;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:154
static CUDA_CALLABLE bool _in_bracket_0(
    wp::vec_t<3, wp::float32> var_x,
    wp::vec_t<3, wp::float32> var_y)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    bool var_4;
    const wp::int32 var_5 = 1;
    wp::float32 var_6;
    const wp::float32 var_7 = 0.0;
    bool var_8;
    bool var_9;
    const wp::int32 var_10 = 1;
    wp::float32 var_11;
    const wp::int32 var_12 = 1;
    wp::float32 var_13;
    bool var_14;
    const wp::int32 var_15 = 1;
    wp::float32 var_16;
    const wp::float32 var_17 = 0.0;
    bool var_18;
    bool var_19;
    bool var_20;
    //---------
    // forward
    // def _in_bracket(x: wp.vec3, y: wp.vec3) -> bool:                                       <L 155>
    // return (x[1] < y[1] and y[1] < 0.0) or (x[1] > y[1] and y[1] > 0.0)                    <L 156>
    var_1 = wp::extract(var_x, var_0);
    var_3 = wp::extract(var_y, var_2);
    var_4 = (var_1 < var_3);
    var_6 = wp::extract(var_y, var_5);
    var_8 = (var_6 < var_7);
    var_9 = var_4 && var_8;
    var_11 = wp::extract(var_x, var_10);
    var_13 = wp::extract(var_y, var_12);
    var_14 = (var_11 > var_13);
    var_16 = wp::extract(var_y, var_15);
    var_18 = (var_16 > var_17);
    var_19 = var_14 && var_18;
    var_20 = var_9 || var_19;
    return var_20;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:902
static CUDA_CALLABLE void adj_linesearch_iterative__locals___syncthreads_0(
    )
{
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:197
static CUDA_CALLABLE void adj__eval_pt_0(
    wp::vec_t<3, wp::float32> var_quad,
    wp::float32 var_alpha,
    wp::vec_t<3, wp::float32> & adj_quad,
    wp::float32 & adj_alpha,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:262
static CUDA_CALLABLE void adj__eval_elliptic_0(
    wp::float32 var_impratio_invsqrt,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<3, wp::float32> var_quad,
    wp::vec_t<3, wp::float32> var_quad1,
    wp::vec_t<3, wp::float32> var_quad2,
    wp::float32 var_alpha,
    wp::float32 & adj_impratio_invsqrt,
    wp::vec_t<5, wp::float32> & adj_friction,
    wp::vec_t<3, wp::float32> & adj_quad,
    wp::vec_t<3, wp::float32> & adj_quad1,
    wp::vec_t<3, wp::float32> & adj_quad2,
    wp::float32 & adj_alpha,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:167
static CUDA_CALLABLE void adj__eval_pt_direct_alpha_zero_0(
    wp::float32 var_jaref,
    wp::float32 var_jv,
    wp::float32 var_d,
    wp::float32 & adj_jaref,
    wp::float32 & adj_jv,
    wp::float32 & adj_d,
    wp::vec_t<3, wp::float32> & adj_ret)
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:223
static CUDA_CALLABLE void adj__eval_frictionloss_pt_0(
    wp::float32 var_x,
    wp::float32 var_f,
    wp::float32 var_rf,
    wp::float32 var_jv,
    wp::float32 var_d,
    wp::float32 & adj_x,
    wp::float32 & adj_f,
    wp::float32 & adj_rf,
    wp::float32 & adj_jv,
    wp::float32 & adj_d,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:676
static CUDA_CALLABLE void adj__compute_efc_eval_pt_alpha_zero_elliptic_0(
    wp::int32 var_efcid,
    wp::int32 var_ne,
    wp::int32 var_nf,
    wp::float32 var_impratio_invsqrt,
    wp::int32 var_efc_type,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::float32> var_efc_frictionloss,
    wp::float32 var_ctx_Jaref,
    wp::float32 var_ctx_jv,
    wp::vec_t<3, wp::float32> var_ctx_quad,
    wp::vec_t<5, wp::float32> var_contact_friction,
    wp::int32 var_efc_address0,
    wp::vec_t<3, wp::float32> var_quad1,
    wp::vec_t<3, wp::float32> var_quad2,
    wp::int32 & adj_efcid,
    wp::int32 & adj_ne,
    wp::int32 & adj_nf,
    wp::float32 & adj_impratio_invsqrt,
    wp::int32 & adj_efc_type,
    wp::array_t<wp::float32> & adj_efc_D_in,
    wp::array_t<wp::float32> & adj_efc_frictionloss,
    wp::float32 & adj_ctx_Jaref,
    wp::float32 & adj_ctx_jv,
    wp::vec_t<3, wp::float32> & adj_ctx_quad,
    wp::vec_t<5, wp::float32> & adj_contact_friction,
    wp::int32 & adj_efc_address0,
    wp::vec_t<3, wp::float32> & adj_quad1,
    wp::vec_t<3, wp::float32> & adj_quad2,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:159
static CUDA_CALLABLE void adj__eval_pt_direct_0(
    wp::float32 var_jaref,
    wp::float32 var_jv,
    wp::float32 var_d,
    wp::float32 var_alpha,
    wp::float32 & adj_jaref,
    wp::float32 & adj_jv,
    wp::float32 & adj_d,
    wp::float32 & adj_alpha,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:601
static CUDA_CALLABLE void adj__compute_efc_eval_pt_elliptic_0(
    wp::int32 var_efcid,
    wp::float32 var_alpha,
    wp::int32 var_ne,
    wp::int32 var_nf,
    wp::float32 var_impratio_invsqrt,
    wp::int32 var_efc_type,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::float32> var_efc_frictionloss,
    wp::float32 var_ctx_Jaref,
    wp::float32 var_ctx_jv,
    wp::vec_t<3, wp::float32> var_ctx_quad,
    wp::vec_t<5, wp::float32> var_contact_friction,
    wp::int32 var_efc_address0,
    wp::vec_t<3, wp::float32> var_quad1,
    wp::vec_t<3, wp::float32> var_quad2,
    wp::int32 & adj_efcid,
    wp::float32 & adj_alpha,
    wp::int32 & adj_ne,
    wp::int32 & adj_nf,
    wp::float32 & adj_impratio_invsqrt,
    wp::int32 & adj_efc_type,
    wp::array_t<wp::float32> & adj_efc_D_in,
    wp::array_t<wp::float32> & adj_efc_frictionloss,
    wp::float32 & adj_ctx_Jaref,
    wp::float32 & adj_ctx_jv,
    wp::vec_t<3, wp::float32> & adj_ctx_quad,
    wp::vec_t<5, wp::float32> & adj_contact_friction,
    wp::int32 & adj_efc_address0,
    wp::vec_t<3, wp::float32> & adj_quad1,
    wp::vec_t<3, wp::float32> & adj_quad2,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:174
static CUDA_CALLABLE void adj__eval_pt_direct_3alphas_0(
    wp::float32 var_jaref,
    wp::float32 var_jv,
    wp::float32 var_d,
    wp::float32 var_lo_alpha,
    wp::float32 var_hi_alpha,
    wp::float32 var_mid_alpha,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::float32 & adj_jaref,
    wp::float32 & adj_jv,
    wp::float32 & adj_d,
    wp::float32 & adj_lo_alpha,
    wp::float32 & adj_hi_alpha,
    wp::float32 & adj_mid_alpha,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:235
static CUDA_CALLABLE void adj__eval_frictionloss_pt_one_0(
    wp::float32 var_x,
    wp::float32 var_f,
    wp::float32 var_rf,
    wp::float32 var_half_d,
    wp::float32 var_jvD,
    wp::float32 var_hessian,
    wp::float32 var_f_jv,
    wp::float32 & adj_x,
    wp::float32 & adj_f,
    wp::float32 & adj_rf,
    wp::float32 & adj_half_d,
    wp::float32 & adj_jvD,
    wp::float32 & adj_hessian,
    wp::float32 & adj_f_jv,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:246
static CUDA_CALLABLE void adj__eval_frictionloss_pt_3alphas_0(
    wp::float32 var_x_lo,
    wp::float32 var_x_hi,
    wp::float32 var_x_mid,
    wp::float32 var_f,
    wp::float32 var_rf,
    wp::float32 var_jv,
    wp::float32 var_d,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::float32 & adj_x_lo,
    wp::float32 & adj_x_hi,
    wp::float32 & adj_x_mid,
    wp::float32 & adj_f,
    wp::float32 & adj_rf,
    wp::float32 & adj_jv,
    wp::float32 & adj_d,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:763
static CUDA_CALLABLE void adj__compute_efc_eval_pt_3alphas_elliptic_0(
    wp::int32 var_efcid,
    wp::float32 var_lo_alpha,
    wp::float32 var_hi_alpha,
    wp::float32 var_mid_alpha,
    wp::int32 var_ne,
    wp::int32 var_nf,
    wp::float32 var_impratio_invsqrt,
    wp::int32 var_efc_type,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::float32> var_efc_frictionloss,
    wp::float32 var_ctx_Jaref,
    wp::float32 var_ctx_jv,
    wp::vec_t<3, wp::float32> var_ctx_quad,
    wp::vec_t<5, wp::float32> var_contact_friction,
    wp::int32 var_efc_address0,
    wp::vec_t<3, wp::float32> var_quad1,
    wp::vec_t<3, wp::float32> var_quad2,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::int32 & adj_efcid,
    wp::float32 & adj_lo_alpha,
    wp::float32 & adj_hi_alpha,
    wp::float32 & adj_mid_alpha,
    wp::int32 & adj_ne,
    wp::int32 & adj_nf,
    wp::float32 & adj_impratio_invsqrt,
    wp::int32 & adj_efc_type,
    wp::array_t<wp::float32> & adj_efc_D_in,
    wp::array_t<wp::float32> & adj_efc_frictionloss,
    wp::float32 & adj_ctx_Jaref,
    wp::float32 & adj_ctx_jv,
    wp::vec_t<3, wp::float32> & adj_ctx_quad,
    wp::vec_t<5, wp::float32> & adj_contact_friction,
    wp::int32 & adj_efc_address0,
    wp::vec_t<3, wp::float32> & adj_quad1,
    wp::vec_t<3, wp::float32> & adj_quad2,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:208
static CUDA_CALLABLE void adj__eval_pt_3alphas_0(
    wp::vec_t<3, wp::float32> var_quad,
    wp::float32 var_lo_alpha,
    wp::float32 var_hi_alpha,
    wp::float32 var_mid_alpha,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_quad,
    wp::float32 & adj_lo_alpha,
    wp::float32 & adj_hi_alpha,
    wp::float32 & adj_mid_alpha,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/solver.py:154
static CUDA_CALLABLE void adj__in_bracket_0(
    wp::vec_t<3, wp::float32> var_x,
    wp::vec_t<3, wp::float32> var_y,
    wp::vec_t<3, wp::float32> & adj_x,
    wp::vec_t<3, wp::float32> & adj_y,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void linesearch_iterative__locals__kernel_f0104db3_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_opt_tolerance,
    wp::array_t<wp::float32> var_opt_ls_tolerance,
    wp::array_t<wp::float32> var_opt_impratio_invsqrt,
    wp::array_t<wp::float32> var_stat_meaninertia,
    wp::array_t<wp::int32> var_ne_in,
    wp::array_t<wp::int32> var_nf_in,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::float32> var_qfrc_smooth_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::int32> var_efc_type_in,
    wp::array_t<wp::int32> var_efc_id_in,
    wp::array_t<wp::int32> var_efc_J_rownnz_in,
    wp::array_t<wp::int32> var_efc_J_rowadr_in,
    wp::array_t<wp::int32> var_efc_J_colind_in,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::float32> var_efc_frictionloss_in,
    wp::int32 var_njmax_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::float32> var_ctx_Jaref_in,
    wp::array_t<wp::float32> var_ctx_search_in,
    wp::array_t<wp::float32> var_ctx_search_dot_in,
    wp::array_t<wp::float32> var_ctx_gauss_in,
    wp::array_t<wp::float32> var_ctx_mv_in,
    wp::array_t<wp::float32> var_ctx_jv_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ctx_quad_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_qacc_out,
    wp::array_t<wp::float32> var_efc_Ma_out,
    wp::array_t<wp::float32> var_ctx_Jaref_out,
    wp::array_t<wp::float32> var_ctx_jv_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ctx_quad_out)
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
        bool* var_2;
        bool var_3;
        bool var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        const bool var_14 = false;
        const bool var_15 = true;
        wp::shape_t* var_16;
        const wp::int32 var_17 = 0;
        wp::int32 var_18;
        wp::shape_t var_19;
        wp::int32 var_20;
        wp::float32* var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        const wp::int32 var_24 = 0;
        wp::int32* var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::range_t var_29;
        wp::int32 var_30;
        wp::int32* var_31;
        const wp::int32 var_32 = 7;
        bool var_33;
        wp::int32 var_34;
        wp::int32* var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        bool var_38;
        const wp::int32 var_39 = 0;
        wp::int32* var_40;
        wp::int32 var_41;
        wp::int32 var_42;
        bool var_43;
        wp::float32* var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::float32* var_47;
        wp::float32 var_48;
        wp::float32 var_49;
        wp::float32* var_50;
        wp::float32 var_51;
        wp::float32 var_52;
        wp::float32 var_53;
        const wp::float32 var_54 = 0.5;
        wp::float32 var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        wp::float32 var_58;
        const wp::float32 var_59 = 0.5;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::vec_t<3, wp::float32> var_62;
        wp::int32* var_63;
        wp::int32 var_64;
        wp::int32 var_65;
        wp::vec_t<5, wp::float32>* var_66;
        wp::vec_t<5, wp::float32> var_67;
        wp::vec_t<5, wp::float32> var_68;
        const wp::int32 var_69 = 0;
        wp::float32 var_70;
        wp::float32 var_71;
        wp::float32 var_72;
        wp::float32 var_73;
        const wp::float32 var_74 = 0.0;
        wp::float32 var_75;
        const wp::float32 var_76 = 0.0;
        wp::float32 var_77;
        const wp::float32 var_78 = 0.0;
        wp::float32 var_79;
        const wp::int32 var_80 = 1;
        wp::range_t var_81;
        wp::int32 var_82;
        wp::int32* var_83;
        wp::int32 var_84;
        wp::int32 var_85;
        const wp::int32 var_86 = 0;
        bool var_87;
        wp::float32* var_88;
        wp::float32 var_89;
        wp::float32 var_90;
        wp::float32* var_91;
        wp::float32 var_92;
        wp::float32 var_93;
        wp::float32* var_94;
        wp::float32 var_95;
        wp::float32 var_96;
        wp::float32 var_97;
        const wp::float32 var_98 = 0.5;
        wp::float32 var_99;
        wp::float32 var_100;
        wp::float32 var_101;
        const wp::float32 var_102 = 0.5;
        wp::float32 var_103;
        wp::float32 var_104;
        wp::float32 var_105;
        wp::vec_t<3, wp::float32> var_106;
        wp::vec_t<3, wp::float32> var_107;
        const wp::int32 var_108 = 1;
        wp::int32 var_109;
        wp::float32 var_110;
        wp::float32 var_111;
        wp::float32 var_112;
        wp::float32 var_113;
        wp::float32 var_114;
        wp::float32 var_115;
        wp::float32 var_116;
        wp::float32 var_117;
        wp::float32 var_118;
        wp::vec_t<3, wp::float32> var_119;
        wp::float32 var_120;
        wp::float32 var_121;
        wp::float32 var_122;
        const wp::int32 var_123 = 1;
        wp::int32* var_124;
        wp::int32 var_125;
        wp::int32 var_126;
        wp::vec_t<3, wp::float32> var_127;
        wp::float32 var_128;
        const wp::int32 var_129 = 2;
        wp::int32* var_130;
        wp::int32 var_131;
        wp::int32 var_132;
        const wp::float32 var_133 = 1.0;
        wp::float32 var_134;
        wp::float32 var_135;
        wp::float32 var_136;
        wp::vec_t<3, wp::float32> var_137;
        wp::shape_t* var_138;
        const wp::int32 var_139 = 0;
        wp::int32 var_140;
        wp::shape_t var_141;
        wp::int32 var_142;
        wp::float32* var_143;
        wp::float32 var_144;
        wp::float32 var_145;
        wp::shape_t* var_146;
        const wp::int32 var_147 = 0;
        wp::int32 var_148;
        wp::shape_t var_149;
        wp::int32 var_150;
        wp::float32* var_151;
        wp::float32 var_152;
        wp::float32 var_153;
        wp::float32* var_154;
        wp::float32 var_155;
        wp::float32 var_156;
        wp::shape_t* var_157;
        const wp::int32 var_158 = 0;
        wp::int32 var_159;
        wp::shape_t var_160;
        wp::int32 var_161;
        wp::float32* var_162;
        wp::float32 var_163;
        wp::float32 var_164;
        wp::float32 var_165;
        wp::float32 var_166;
        wp::float32 var_167;
        wp::float32 var_168;
        wp::float32 var_169;
        const wp::float32 var_170 = 1e-06;
        wp::float32 var_171;
        const wp::float32 var_172 = 0.0;
        wp::vec_t<3, wp::float32> var_173;
        wp::int32 var_174;
        wp::range_t var_175;
        wp::int32 var_176;
        const bool var_177 = true;
        wp::int32* var_178;
        wp::int32 var_179;
        wp::int32 var_180;
        const wp::int32 var_181 = 0;
        const wp::float32 var_182 = 0.0;
        wp::vec_t<5, wp::float32> var_183;
        const wp::int32 var_184 = 0;
        wp::int32 var_185;
        const wp::float32 var_186 = 0.0;
        wp::vec_t<3, wp::float32> var_187;
        const wp::float32 var_188 = 0.0;
        wp::vec_t<3, wp::float32> var_189;
        const wp::float32 var_190 = 0.0;
        wp::vec_t<3, wp::float32> var_191;
        const wp::int32 var_192 = 7;
        bool var_193;
        wp::int32* var_194;
        wp::int32 var_195;
        wp::int32 var_196;
        wp::vec_t<5, wp::float32>* var_197;
        wp::vec_t<5, wp::float32> var_198;
        wp::vec_t<5, wp::float32> var_199;
        const wp::int32 var_200 = 0;
        wp::int32* var_201;
        wp::int32 var_202;
        wp::int32 var_203;
        const wp::int32 var_204 = 1;
        wp::int32* var_205;
        wp::int32 var_206;
        wp::int32 var_207;
        const wp::int32 var_208 = 2;
        wp::int32* var_209;
        wp::int32 var_210;
        wp::int32 var_211;
        wp::vec_t<3, wp::float32>* var_212;
        wp::vec_t<3, wp::float32> var_213;
        wp::vec_t<3, wp::float32> var_214;
        wp::vec_t<3, wp::float32>* var_215;
        wp::vec_t<3, wp::float32> var_216;
        wp::vec_t<3, wp::float32> var_217;
        wp::vec_t<3, wp::float32>* var_218;
        wp::vec_t<3, wp::float32> var_219;
        wp::vec_t<3, wp::float32> var_220;
        wp::int32 var_221;
        wp::vec_t<5, wp::float32> var_222;
        wp::int32 var_223;
        wp::vec_t<3, wp::float32> var_224;
        wp::vec_t<3, wp::float32> var_225;
        wp::vec_t<3, wp::float32> var_226;
        wp::slice_t var_227;
        const wp::int32 var_228 = 0;
        wp::array_t<wp::float32> var_229;
        wp::slice_t var_230;
        const wp::int32 var_231 = 0;
        wp::array_t<wp::float32> var_232;
        wp::float32* var_233;
        wp::float32* var_234;
        wp::vec_t<3, wp::float32> var_235;
        wp::float32 var_236;
        wp::float32 var_237;
        wp::vec_t<3, wp::float32> var_238;
        const bool var_239 = true;
        wp::tile_register_t<wp::vec_t<3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_240 = wp::tile_register_t<wp::vec_t<3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_shared_t<wp::vec_t<3, wp::float32>,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_241 = wp::tile_alloc_empty<wp::vec_t<3, wp::float32>,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
        const wp::float32 var_242 = 0.0;
        wp::vec_t<2, wp::float32> var_243;
        wp::int32 var_244;
        wp::range_t var_245;
        wp::int32 var_246;
        wp::float32* var_247;
        wp::float32 var_248;
        wp::float32 var_249;
        wp::float32* var_250;
        wp::float32* var_251;
        wp::float32 var_252;
        wp::float32 var_253;
        wp::float32 var_254;
        wp::float32 var_255;
        const wp::float32 var_256 = 0.5;
        wp::float32 var_257;
        wp::float32* var_258;
        wp::float32 var_259;
        wp::float32 var_260;
        wp::vec_t<2, wp::float32> var_261;
        wp::vec_t<2, wp::float32> var_262;
        const bool var_263 = true;
        wp::tile_register_t<wp::vec_t<2, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_264 = wp::tile_register_t<wp::vec_t<2, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_shared_t<wp::vec_t<2, wp::float32>,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_265 = wp::tile_alloc_empty<wp::vec_t<2, wp::float32>,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
        const wp::int32 var_266 = 0;
        wp::vec_t<2, wp::float32> var_267;
        wp::float32* var_268;
        const wp::int32 var_269 = 0;
        wp::float32 var_270;
        const wp::int32 var_271 = 1;
        wp::float32 var_272;
        wp::vec_t<3, wp::float32> var_273;
        wp::float32 var_274;
        const wp::int32 var_275 = 0;
        wp::float32 var_276;
        const wp::int32 var_277 = 1;
        wp::float32 var_278;
        const wp::float32 var_279 = 2.0;
        const wp::int32 var_280 = 2;
        wp::float32 var_281;
        wp::float32 var_282;
        wp::vec_t<3, wp::float32> var_283;
        const wp::int32 var_284 = 0;
        wp::vec_t<3, wp::float32> var_285;
        wp::vec_t<3, wp::float32> var_286;
        const wp::int32 var_287 = 1;
        wp::float32 var_288;
        const wp::int32 var_289 = 2;
        wp::float32 var_290;
        wp::float32 var_291;
        wp::float32 var_292;
        const wp::float32 var_293 = 0.0;
        wp::vec_t<3, wp::float32> var_294;
        wp::int32 var_295;
        wp::range_t var_296;
        wp::int32 var_297;
        const bool var_298 = true;
        wp::int32* var_299;
        wp::int32 var_300;
        wp::int32 var_301;
        const wp::int32 var_302 = 0;
        const wp::float32 var_303 = 0.0;
        wp::vec_t<5, wp::float32> var_304;
        const wp::int32 var_305 = 0;
        wp::int32 var_306;
        const wp::float32 var_307 = 0.0;
        wp::vec_t<3, wp::float32> var_308;
        const wp::float32 var_309 = 0.0;
        wp::vec_t<3, wp::float32> var_310;
        const wp::float32 var_311 = 0.0;
        wp::vec_t<3, wp::float32> var_312;
        const wp::int32 var_313 = 7;
        bool var_314;
        wp::int32* var_315;
        wp::int32 var_316;
        wp::int32 var_317;
        wp::vec_t<5, wp::float32>* var_318;
        wp::vec_t<5, wp::float32> var_319;
        wp::vec_t<5, wp::float32> var_320;
        const wp::int32 var_321 = 0;
        wp::int32* var_322;
        wp::int32 var_323;
        wp::int32 var_324;
        const wp::int32 var_325 = 1;
        wp::int32* var_326;
        wp::int32 var_327;
        wp::int32 var_328;
        const wp::int32 var_329 = 2;
        wp::int32* var_330;
        wp::int32 var_331;
        wp::int32 var_332;
        wp::vec_t<3, wp::float32>* var_333;
        wp::vec_t<3, wp::float32> var_334;
        wp::vec_t<3, wp::float32> var_335;
        wp::vec_t<3, wp::float32>* var_336;
        wp::vec_t<3, wp::float32> var_337;
        wp::vec_t<3, wp::float32> var_338;
        wp::vec_t<3, wp::float32>* var_339;
        wp::vec_t<3, wp::float32> var_340;
        wp::vec_t<3, wp::float32> var_341;
        wp::int32 var_342;
        wp::vec_t<5, wp::float32> var_343;
        wp::int32 var_344;
        wp::vec_t<3, wp::float32> var_345;
        wp::vec_t<3, wp::float32> var_346;
        wp::vec_t<3, wp::float32> var_347;
        wp::int32 var_348;
        wp::int32 var_349;
        wp::slice_t var_350;
        const wp::int32 var_351 = 0;
        wp::array_t<wp::float32> var_352;
        wp::slice_t var_353;
        const wp::int32 var_354 = 0;
        wp::array_t<wp::float32> var_355;
        wp::float32* var_356;
        wp::float32* var_357;
        wp::vec_t<3, wp::float32> var_358;
        wp::float32 var_359;
        wp::float32 var_360;
        wp::vec_t<3, wp::float32> var_361;
        const bool var_362 = true;
        wp::tile_register_t<wp::vec_t<3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_363 = wp::tile_register_t<wp::vec_t<3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_shared_t<wp::vec_t<3, wp::float32>,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_364 = wp::tile_alloc_empty<wp::vec_t<3, wp::float32>,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
        wp::vec_t<3, wp::float32> var_365;
        const wp::int32 var_366 = 0;
        wp::vec_t<3, wp::float32> var_367;
        wp::vec_t<3, wp::float32> var_368;
        const wp::int32 var_369 = 1;
        wp::float32 var_370;
        wp::float32 var_371;
        bool var_372;
        const wp::int32 var_373 = 0;
        wp::float32 var_374;
        const wp::int32 var_375 = 0;
        wp::float32 var_376;
        bool var_377;
        bool var_378;
        bool var_379;
        const wp::float32 var_380 = 0.0;
        wp::float32 var_381;
        const wp::int32 var_382 = 1;
        wp::float32 var_383;
        const wp::int32 var_384 = 1;
        wp::float32 var_385;
        bool var_386;
        wp::vec_t<3, wp::float32> var_387;
        const wp::float32 var_388 = 0.0;
        wp::float32 var_389;
        wp::vec_t<3, wp::float32> var_390;
        const wp::float32 var_391 = 0.0;
        wp::float32 var_392;
        const wp::int32 var_393 = 50;
        wp::range_t var_394;
        wp::int32 var_395;
        const wp::int32 var_396 = 1;
        wp::float32 var_397;
        const wp::int32 var_398 = 2;
        wp::float32 var_399;
        wp::float32 var_400;
        wp::float32 var_401;
        const wp::int32 var_402 = 1;
        wp::float32 var_403;
        const wp::int32 var_404 = 2;
        wp::float32 var_405;
        wp::float32 var_406;
        wp::float32 var_407;
        const wp::float32 var_408 = 0.5;
        wp::float32 var_409;
        wp::float32 var_410;
        const wp::float32 var_411 = 0.0;
        wp::vec_t<3, wp::float32> var_412;
        const wp::float32 var_413 = 0.0;
        wp::vec_t<3, wp::float32> var_414;
        const wp::float32 var_415 = 0.0;
        wp::vec_t<3, wp::float32> var_416;
        wp::int32 var_417;
        wp::range_t var_418;
        wp::int32 var_419;
        const bool var_420 = true;
        wp::int32* var_421;
        wp::int32 var_422;
        wp::int32 var_423;
        const wp::int32 var_424 = 0;
        const wp::float32 var_425 = 0.0;
        wp::vec_t<5, wp::float32> var_426;
        const wp::int32 var_427 = 0;
        wp::int32 var_428;
        const wp::float32 var_429 = 0.0;
        wp::vec_t<3, wp::float32> var_430;
        const wp::float32 var_431 = 0.0;
        wp::vec_t<3, wp::float32> var_432;
        const wp::float32 var_433 = 0.0;
        wp::vec_t<3, wp::float32> var_434;
        const wp::int32 var_435 = 7;
        bool var_436;
        wp::int32* var_437;
        wp::int32 var_438;
        wp::int32 var_439;
        wp::vec_t<5, wp::float32>* var_440;
        wp::vec_t<5, wp::float32> var_441;
        wp::vec_t<5, wp::float32> var_442;
        const wp::int32 var_443 = 0;
        wp::int32* var_444;
        wp::int32 var_445;
        wp::int32 var_446;
        const wp::int32 var_447 = 1;
        wp::int32* var_448;
        wp::int32 var_449;
        wp::int32 var_450;
        const wp::int32 var_451 = 2;
        wp::int32* var_452;
        wp::int32 var_453;
        wp::int32 var_454;
        wp::vec_t<3, wp::float32>* var_455;
        wp::vec_t<3, wp::float32> var_456;
        wp::vec_t<3, wp::float32> var_457;
        wp::vec_t<3, wp::float32>* var_458;
        wp::vec_t<3, wp::float32> var_459;
        wp::vec_t<3, wp::float32> var_460;
        wp::vec_t<3, wp::float32>* var_461;
        wp::vec_t<3, wp::float32> var_462;
        wp::vec_t<3, wp::float32> var_463;
        wp::int32 var_464;
        wp::vec_t<5, wp::float32> var_465;
        wp::int32 var_466;
        wp::vec_t<3, wp::float32> var_467;
        wp::vec_t<3, wp::float32> var_468;
        wp::vec_t<3, wp::float32> var_469;
        wp::int32 var_470;
        wp::int32 var_471;
        wp::slice_t var_472;
        const wp::int32 var_473 = 0;
        wp::array_t<wp::float32> var_474;
        wp::slice_t var_475;
        const wp::int32 var_476 = 0;
        wp::array_t<wp::float32> var_477;
        wp::float32* var_478;
        wp::float32* var_479;
        wp::vec_t<3, wp::float32> var_480;
        wp::vec_t<3, wp::float32> var_481;
        wp::vec_t<3, wp::float32> var_482;
        wp::float32 var_483;
        wp::float32 var_484;
        wp::vec_t<3, wp::float32> var_485;
        wp::vec_t<3, wp::float32> var_486;
        wp::vec_t<3, wp::float32> var_487;
        const wp::int32 var_488 = 0;
        wp::float32 var_489;
        const wp::int32 var_490 = 0;
        wp::float32 var_491;
        const wp::int32 var_492 = 0;
        wp::float32 var_493;
        const wp::int32 var_494 = 1;
        wp::float32 var_495;
        const wp::int32 var_496 = 1;
        wp::float32 var_497;
        const wp::int32 var_498 = 1;
        wp::float32 var_499;
        const wp::int32 var_500 = 2;
        wp::float32 var_501;
        const wp::int32 var_502 = 2;
        wp::float32 var_503;
        const wp::int32 var_504 = 2;
        wp::float32 var_505;
        wp::mat_t<3, 3, wp::float32> var_506;
        const bool var_507 = true;
        wp::tile_register_t<wp::mat_t<3, 3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_508 = wp::tile_register_t<wp::mat_t<3, 3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_shared_t<wp::mat_t<3, 3, wp::float32>,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_509 = wp::tile_alloc_empty<wp::mat_t<3, 3, wp::float32>,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
        const wp::int32 var_510 = 0;
        wp::mat_t<3, 3, wp::float32> var_511;
        wp::vec_t<3, wp::float32> var_512;
        wp::vec_t<3, wp::float32> var_513;
        wp::vec_t<3, wp::float32> var_514;
        const wp::int32 var_515 = 0;
        const wp::int32 var_516 = 0;
        wp::float32 var_517;
        const wp::int32 var_518 = 1;
        const wp::int32 var_519 = 0;
        wp::float32 var_520;
        const wp::int32 var_521 = 2;
        const wp::int32 var_522 = 0;
        wp::float32 var_523;
        wp::vec_t<3, wp::float32> var_524;
        wp::vec_t<3, wp::float32> var_525;
        const wp::int32 var_526 = 0;
        const wp::int32 var_527 = 1;
        wp::float32 var_528;
        const wp::int32 var_529 = 1;
        const wp::int32 var_530 = 1;
        wp::float32 var_531;
        const wp::int32 var_532 = 2;
        const wp::int32 var_533 = 1;
        wp::float32 var_534;
        wp::vec_t<3, wp::float32> var_535;
        wp::vec_t<3, wp::float32> var_536;
        const wp::int32 var_537 = 0;
        const wp::int32 var_538 = 2;
        wp::float32 var_539;
        const wp::int32 var_540 = 1;
        const wp::int32 var_541 = 2;
        wp::float32 var_542;
        const wp::int32 var_543 = 2;
        const wp::int32 var_544 = 2;
        wp::float32 var_545;
        wp::vec_t<3, wp::float32> var_546;
        wp::vec_t<3, wp::float32> var_547;
        bool var_548;
        wp::vec_t<3, wp::float32> var_549;
        wp::float32 var_550;
        bool var_551;
        wp::vec_t<3, wp::float32> var_552;
        wp::float32 var_553;
        bool var_554;
        wp::vec_t<3, wp::float32> var_555;
        wp::float32 var_556;
        bool var_557;
        bool var_558;
        wp::vec_t<3, wp::float32> var_559;
        wp::float32 var_560;
        bool var_561;
        wp::vec_t<3, wp::float32> var_562;
        wp::float32 var_563;
        bool var_564;
        wp::vec_t<3, wp::float32> var_565;
        wp::float32 var_566;
        bool var_567;
        bool var_568;
        bool var_569;
        bool var_570;
        const wp::int32 var_571 = 1;
        wp::float32 var_572;
        const wp::float32 var_573 = 0.0;
        bool var_574;
        const wp::int32 var_575 = 1;
        wp::float32 var_576;
        wp::float32 var_577;
        bool var_578;
        bool var_579;
        const wp::int32 var_580 = 1;
        wp::float32 var_581;
        const wp::float32 var_582 = 0.0;
        bool var_583;
        const wp::int32 var_584 = 1;
        wp::float32 var_585;
        bool var_586;
        bool var_587;
        bool var_588;
        const wp::int32 var_589 = 0;
        wp::float32 var_590;
        const wp::int32 var_591 = 0;
        wp::float32 var_592;
        bool var_593;
        const wp::int32 var_594 = 0;
        wp::float32 var_595;
        const wp::int32 var_596 = 0;
        wp::float32 var_597;
        bool var_598;
        bool var_599;
        const wp::int32 var_600 = 0;
        wp::float32 var_601;
        const wp::int32 var_602 = 0;
        wp::float32 var_603;
        bool var_604;
        bool var_605;
        wp::float32 var_606;
        bool var_607;
        bool var_608;
        wp::float32 var_609;
        wp::int32 var_610;
        wp::float32 var_611;
        wp::vec_t<3, wp::float32> var_612;
        wp::float32 var_613;
        wp::vec_t<3, wp::float32> var_614;
        wp::float32 var_615;
        wp::float32 var_616;
        wp::float32 var_617;
        wp::int32 var_618;
        wp::range_t var_619;
        wp::int32 var_620;
        wp::float32* var_621;
        wp::float32 var_622;
        wp::float32 var_623;
        wp::float32 var_624;
        wp::float32* var_625;
        wp::float32 var_626;
        wp::float32 var_627;
        wp::float32 var_628;
        wp::int32 var_629;
        wp::range_t var_630;
        wp::int32 var_631;
        wp::float32* var_632;
        wp::float32 var_633;
        wp::float32 var_634;
        wp::float32 var_635;
        //---------
        // forward
        // def kernel(                                                                            <L 917>
        // worldid, tid = wp.tid()                                                                <L 959>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 961>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 962>
            continue;
        }
        var_4 = wp::load(var_2);
        // ne = ne_in[worldid]                                                                    <L 964>
        var_5 = wp::address(var_ne_in, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // nf = nf_in[worldid]                                                                    <L 965>
        var_8 = wp::address(var_nf_in, var_0);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // nefc = wp.min(njmax_in, nefc_in[worldid])                                              <L 966>
        var_11 = wp::address(var_nefc_in, var_0);
        var_13 = wp::load(var_11);
        var_12 = wp::min(var_njmax_in, var_13);
        // if wp.static(FUSE_JV):                                                                 <L 969>
        // if wp.static(IS_ELLIPTIC):                                                             <L 989>
        // impratio_invsqrt = opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]       <L 991>
        var_16 = &(var_opt_impratio_invsqrt.shape);
        var_19 = wp::load(var_16);
        var_18 = wp::extract(var_19, var_17);
        var_20 = wp::mod(var_0, var_18);
        var_21 = wp::address(var_opt_impratio_invsqrt, var_20);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // nacon = nacon_in[0]                                                                    <L 992>
        var_25 = wp::address(var_nacon_in, var_24);
        var_27 = wp::load(var_25);
        var_26 = wp::copy(var_27);
        // for efcid in range(tid, nefc, wp.block_dim()):                                         <L 994>
        var_28 = builtin_block_dim();
        var_29 = wp::range(var_1, var_12, var_28);
        start_for_1:;
            if (iter_cmp(var_29) == 0) goto end_for_1;
            var_30 = wp::iter_next(var_29);
            // if efc_type_in[worldid, efcid] == types.ConstraintType.CONTACT_ELLIPTIC:           <L 996>
            var_31 = wp::address(var_efc_type_in, var_0, var_30);
            var_34 = wp::load(var_31);
            var_33 = (var_34 == var_32);
            if (var_33) {
                // conid = efc_id_in[worldid, efcid]                                              <L 997>
                var_35 = wp::address(var_efc_id_in, var_0, var_30);
                var_37 = wp::load(var_35);
                var_36 = wp::copy(var_37);
                // if conid < nacon:                                                              <L 998>
                var_38 = (var_36 < var_26);
                if (var_38) {
                    // efcid0 = contact_efc_address_in[conid, 0]                                  <L 999>
                    var_40 = wp::address(var_contact_efc_address_in, var_36, var_39);
                    var_42 = wp::load(var_40);
                    var_41 = wp::copy(var_42);
                    // if efcid == efcid0:                                                        <L 1000>
                    var_43 = (var_30 == var_41);
                    if (var_43) {
                        // Jaref = ctx_Jaref_in[worldid, efcid]                                   <L 1001>
                        var_44 = wp::address(var_ctx_Jaref_in, var_0, var_30);
                        var_46 = wp::load(var_44);
                        var_45 = wp::copy(var_46);
                        // jv = ctx_jv_in[worldid, efcid]                                         <L 1002>
                        var_47 = wp::address(var_ctx_jv_in, var_0, var_30);
                        var_49 = wp::load(var_47);
                        var_48 = wp::copy(var_49);
                        // efc_D = efc_D_in[worldid, efcid]                                       <L 1003>
                        var_50 = wp::address(var_efc_D_in, var_0, var_30);
                        var_52 = wp::load(var_50);
                        var_51 = wp::copy(var_52);
                        // jvD = jv * efc_D                                                       <L 1005>
                        var_53 = wp::mul(var_48, var_51);
                        // quad = wp.vec3(0.5 * Jaref * Jaref * efc_D, jvD * Jaref, 0.5 * jv * jvD)       <L 1006>
                        var_55 = wp::mul(var_54, var_45);
                        var_56 = wp::mul(var_55, var_45);
                        var_57 = wp::mul(var_56, var_51);
                        var_58 = wp::mul(var_53, var_45);
                        var_60 = wp::mul(var_59, var_48);
                        var_61 = wp::mul(var_60, var_53);
                        var_62 = wp::vec_t<3, wp::float32>(var_57, var_58, var_61);
                        // dim = contact_dim_in[conid]                                            <L 1009>
                        var_63 = wp::address(var_contact_dim_in, var_36);
                        var_65 = wp::load(var_63);
                        var_64 = wp::copy(var_65);
                        // friction = contact_friction_in[conid]                                  <L 1010>
                        var_66 = wp::address(var_contact_friction_in, var_36);
                        var_68 = wp::load(var_66);
                        var_67 = wp::copy(var_68);
                        // mu = friction[0] * impratio_invsqrt                                    <L 1011>
                        var_70 = wp::extract(var_67, var_69);
                        var_71 = wp::mul(var_70, var_22);
                        // u0 = Jaref * mu                                                        <L 1013>
                        var_72 = wp::mul(var_45, var_71);
                        // v0 = jv * mu                                                           <L 1014>
                        var_73 = wp::mul(var_48, var_71);
                        // uu = float(0.0)                                                        <L 1016>
                        var_75 = wp::float(var_74);
                        // uv = float(0.0)                                                        <L 1017>
                        var_77 = wp::float(var_76);
                        // vv = float(0.0)                                                        <L 1018>
                        var_79 = wp::float(var_78);
                        // for j in range(1, dim):                                                <L 1019>
                        var_81 = wp::range(var_80, var_64);
                        start_for_3:;
                            if (iter_cmp(var_81) == 0) goto end_for_3;
                            var_82 = wp::iter_next(var_81);
                            // efcidj = contact_efc_address_in[conid, j]                          <L 1020>
                            var_83 = wp::address(var_contact_efc_address_in, var_36, var_82);
                            var_85 = wp::load(var_83);
                            var_84 = wp::copy(var_85);
                            // if efcidj >= 0:                                                    <L 1021>
                            var_87 = (var_84 >= var_86);
                            if (var_87) {
                                // jvj = ctx_jv_in[worldid, efcidj]                               <L 1022>
                                var_88 = wp::address(var_ctx_jv_in, var_0, var_84);
                                var_90 = wp::load(var_88);
                                var_89 = wp::copy(var_90);
                                // jarefj = ctx_Jaref_in[worldid, efcidj]                         <L 1023>
                                var_91 = wp::address(var_ctx_Jaref_in, var_0, var_84);
                                var_93 = wp::load(var_91);
                                var_92 = wp::copy(var_93);
                                // dj = efc_D_in[worldid, efcidj]                                 <L 1024>
                                var_94 = wp::address(var_efc_D_in, var_0, var_84);
                                var_96 = wp::load(var_94);
                                var_95 = wp::copy(var_96);
                                // DJj = dj * jarefj                                              <L 1025>
                                var_97 = wp::mul(var_95, var_92);
                                // quad += wp.vec3(0.5 * jarefj * DJj, jvj * DJj, 0.5 * jvj * dj * jvj)       <L 1027>
                                var_99 = wp::mul(var_98, var_92);
                                var_100 = wp::mul(var_99, var_97);
                                var_101 = wp::mul(var_89, var_97);
                                var_103 = wp::mul(var_102, var_89);
                                var_104 = wp::mul(var_103, var_95);
                                var_105 = wp::mul(var_104, var_89);
                                var_106 = wp::vec_t<3, wp::float32>(var_100, var_101, var_105);
                                var_107 = wp::add(var_62, var_106);
                                // frictionj = friction[j - 1]                                    <L 1030>
                                var_109 = wp::sub(var_82, var_108);
                                var_110 = wp::extract(var_67, var_109);
                                // uj = jarefj * frictionj                                        <L 1031>
                                var_111 = wp::mul(var_92, var_110);
                                // vj = jvj * frictionj                                           <L 1032>
                                var_112 = wp::mul(var_89, var_110);
                                // uu += uj * uj                                                  <L 1034>
                                var_113 = wp::mul(var_111, var_111);
                                var_114 = wp::add(var_75, var_113);
                                // uv += uj * vj                                                  <L 1035>
                                var_115 = wp::mul(var_111, var_112);
                                var_116 = wp::add(var_77, var_115);
                                // vv += vj * vj                                                  <L 1036>
                                var_117 = wp::mul(var_112, var_112);
                                var_118 = wp::add(var_79, var_117);
                            }
                            var_119 = wp::where(var_87, var_107, var_62);
                            var_120 = wp::where(var_87, var_114, var_75);
                            var_121 = wp::where(var_87, var_116, var_77);
                            var_122 = wp::where(var_87, var_118, var_79);
                            wp::assign(var_62, var_119);
                            wp::assign(var_75, var_120);
                            wp::assign(var_77, var_121);
                            wp::assign(var_79, var_122);
                            goto start_for_3;
                        end_for_3:;
                        // ctx_quad_out[worldid, efcid] = quad                                    <L 1038>
                        wp::array_store(var_ctx_quad_out, var_0, var_30, var_62);
                        // efcid1 = contact_efc_address_in[conid, 1]                              <L 1040>
                        var_124 = wp::address(var_contact_efc_address_in, var_36, var_123);
                        var_126 = wp::load(var_124);
                        var_125 = wp::copy(var_126);
                        // ctx_quad_out[worldid, efcid1] = wp.vec3(u0, v0, uu)                    <L 1041>
                        var_127 = wp::vec_t<3, wp::float32>(var_72, var_73, var_75);
                        wp::array_store(var_ctx_quad_out, var_0, var_125, var_127);
                        // mu2 = mu * mu                                                          <L 1043>
                        var_128 = wp::mul(var_71, var_71);
                        // efcid2 = contact_efc_address_in[conid, 2]                              <L 1044>
                        var_130 = wp::address(var_contact_efc_address_in, var_36, var_129);
                        var_132 = wp::load(var_130);
                        var_131 = wp::copy(var_132);
                        // ctx_quad_out[worldid, efcid2] = wp.vec3(uv, vv, efc_D / (mu2 * (1.0 + mu2)))       <L 1045>
                        var_134 = wp::add(var_133, var_128);
                        var_135 = wp::mul(var_128, var_134);
                        var_136 = wp::div(var_51, var_135);
                        var_137 = wp::vec_t<3, wp::float32>(var_77, var_79, var_136);
                        wp::array_store(var_ctx_quad_out, var_0, var_131, var_137);
                    }
                }
            }
            goto start_for_1;
        end_for_1:;
        // _syncthreads()  # ensure all quads are written before reading                          <L 1047>
        linesearch_iterative__locals___syncthreads_0();
        // tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]                            <L 1050>
        var_138 = &(var_opt_tolerance.shape);
        var_141 = wp::load(var_138);
        var_140 = wp::extract(var_141, var_139);
        var_142 = wp::mod(var_0, var_140);
        var_143 = wp::address(var_opt_tolerance, var_142);
        var_145 = wp::load(var_143);
        var_144 = wp::copy(var_145);
        // ls_tolerance = opt_ls_tolerance[worldid % opt_ls_tolerance.shape[0]]                   <L 1051>
        var_146 = &(var_opt_ls_tolerance.shape);
        var_149 = wp::load(var_146);
        var_148 = wp::extract(var_149, var_147);
        var_150 = wp::mod(var_0, var_148);
        var_151 = wp::address(var_opt_ls_tolerance, var_150);
        var_153 = wp::load(var_151);
        var_152 = wp::copy(var_153);
        // snorm = wp.sqrt(ctx_search_dot_in[worldid])                                            <L 1052>
        var_154 = wp::address(var_ctx_search_dot_in, var_0);
        var_156 = wp::load(var_154);
        var_155 = wp::sqrt(var_156);
        // meaninertia = stat_meaninertia[worldid % stat_meaninertia.shape[0]]                    <L 1053>
        var_157 = &(var_stat_meaninertia.shape);
        var_160 = wp::load(var_157);
        var_159 = wp::extract(var_160, var_158);
        var_161 = wp::mod(var_0, var_159);
        var_162 = wp::address(var_stat_meaninertia, var_161);
        var_164 = wp::load(var_162);
        var_163 = wp::copy(var_164);
        // scale = meaninertia * wp.float(nv)                                                     <L 1054>
        var_165 = wp::float(var_nv);
        var_166 = wp::mul(var_163, var_165);
        // gtol = wp.max(tolerance * ls_tolerance * snorm * scale, 1e-6)                          <L 1055>
        var_167 = wp::mul(var_144, var_152);
        var_168 = wp::mul(var_167, var_155);
        var_169 = wp::mul(var_168, var_166);
        var_171 = wp::max(var_169, var_170);
        // local_p0 = wp.vec3(0.0)                                                                <L 1058>
        var_173 = wp::vec_t<3, wp::float32>(var_172);
        // for efcid in range(tid, nefc, wp.block_dim()):                                         <L 1059>
        var_174 = builtin_block_dim();
        var_175 = wp::range(var_1, var_12, var_174);
        start_for_5:;
            if (iter_cmp(var_175) == 0) goto end_for_5;
            var_176 = wp::iter_next(var_175);
            // if wp.static(IS_ELLIPTIC):                                                         <L 1060>
            // efc_type = efc_type_in[worldid, efcid]                                             <L 1061>
            var_178 = wp::address(var_efc_type_in, var_0, var_176);
            var_180 = wp::load(var_178);
            var_179 = wp::copy(var_180);
            // efc_id = 0                                                                         <L 1062>
            // contact_friction = types.vec5(0.0)                                                 <L 1063>
            var_183 = wp::vec_t<5, wp::float32>(var_182);
            // efc_addr0 = int(0)                                                                 <L 1064>
            var_185 = wp::int(var_184);
            // ctx_quad = wp.vec3(0.0)                                                            <L 1065>
            var_187 = wp::vec_t<3, wp::float32>(var_186);
            // quad1 = wp.vec3(0.0)                                                               <L 1066>
            var_189 = wp::vec_t<3, wp::float32>(var_188);
            // quad2 = wp.vec3(0.0)                                                               <L 1067>
            var_191 = wp::vec_t<3, wp::float32>(var_190);
            // if efc_type == types.ConstraintType.CONTACT_ELLIPTIC:                              <L 1069>
            var_193 = (var_179 == var_192);
            if (var_193) {
                // efc_id = efc_id_in[worldid, efcid]                                             <L 1070>
                var_194 = wp::address(var_efc_id_in, var_0, var_176);
                var_196 = wp::load(var_194);
                var_195 = wp::copy(var_196);
                // contact_friction = contact_friction_in[efc_id]                                 <L 1071>
                var_197 = wp::address(var_contact_friction_in, var_195);
                var_199 = wp::load(var_197);
                var_198 = wp::copy(var_199);
                // efc_addr0 = contact_efc_address_in[efc_id, 0]                                  <L 1072>
                var_201 = wp::address(var_contact_efc_address_in, var_195, var_200);
                var_203 = wp::load(var_201);
                var_202 = wp::copy(var_203);
                // efc_addr1 = contact_efc_address_in[efc_id, 1]                                  <L 1073>
                var_205 = wp::address(var_contact_efc_address_in, var_195, var_204);
                var_207 = wp::load(var_205);
                var_206 = wp::copy(var_207);
                // efc_addr2 = contact_efc_address_in[efc_id, 2]                                  <L 1074>
                var_209 = wp::address(var_contact_efc_address_in, var_195, var_208);
                var_211 = wp::load(var_209);
                var_210 = wp::copy(var_211);
                // ctx_quad = ctx_quad_in[worldid, efcid]                                         <L 1075>
                var_212 = wp::address(var_ctx_quad_in, var_0, var_176);
                var_214 = wp::load(var_212);
                var_213 = wp::copy(var_214);
                // quad1 = ctx_quad_in[worldid, efc_addr1]                                        <L 1076>
                var_215 = wp::address(var_ctx_quad_in, var_0, var_206);
                var_217 = wp::load(var_215);
                var_216 = wp::copy(var_217);
                // quad2 = ctx_quad_in[worldid, efc_addr2]                                        <L 1077>
                var_218 = wp::address(var_ctx_quad_in, var_0, var_210);
                var_220 = wp::load(var_218);
                var_219 = wp::copy(var_220);
            }
            var_221 = wp::where(var_193, var_195, var_181);
            var_222 = wp::where(var_193, var_198, var_183);
            var_223 = wp::where(var_193, var_202, var_185);
            var_224 = wp::where(var_193, var_213, var_187);
            var_225 = wp::where(var_193, var_216, var_189);
            var_226 = wp::where(var_193, var_219, var_191);
            // local_p0 += _compute_efc_eval_pt_alpha_zero(                                       <L 1079>
            // efcid,                                                                             <L 1080>
            // ne,                                                                                <L 1081>
            // nf,                                                                                <L 1082>
            // impratio_invsqrt,                                                                  <L 1083>
            // efc_type,                                                                          <L 1084>
            // efc_D_in[worldid],                                                                 <L 1085>
            var_227 = wp::slice_t(var_0, var_0, var_228);
            var_229 = wp::view(var_efc_D_in, var_227);
            // efc_frictionloss_in[worldid],                                                      <L 1086>
            var_230 = wp::slice_t(var_0, var_0, var_231);
            var_232 = wp::view(var_efc_frictionloss_in, var_230);
            // ctx_Jaref_in[worldid, efcid],                                                      <L 1087>
            var_233 = wp::address(var_ctx_Jaref_in, var_0, var_176);
            // ctx_jv_in[worldid, efcid],                                                         <L 1088>
            var_234 = wp::address(var_ctx_jv_in, var_0, var_176);
            // ctx_quad,                                                                          <L 1089>
            // contact_friction,                                                                  <L 1090>
            // efc_addr0,                                                                         <L 1091>
            // quad1,                                                                             <L 1092>
            // quad2,                                                                             <L 1093>
            var_236 = wp::load(var_233);
            var_237 = wp::load(var_234);
            var_235 = _compute_efc_eval_pt_alpha_zero_elliptic_0(var_176, var_6, var_9, var_22, var_179, var_229, var_232, var_236, var_237, var_224, var_222, var_223, var_225, var_226);
            // local_p0 += _compute_efc_eval_pt_alpha_zero(                                       <L 1079>
            var_238 = wp::add(var_173, var_235);
            wp::assign(var_173, var_238);
            goto start_for_5;
        end_for_5:;
        // p0_tile = wp.tile(local_p0, preserve_type=True)                                        <L 1110>
        var_240 = wp::tile<wp::vec_t<3, wp::float32>>(var_173);
        // p0_sum = wp.tile_reduce(wp.add, p0_tile)                                               <L 1111>
        var_241 = wp::tile_reduce(wp::add, var_240);
        // local_gauss = wp.vec2(0.0)  # vec2 since component 0 is constant (ctx_gauss_in)        <L 1114>
        var_243 = wp::vec_t<2, wp::float32>(var_242);
        // for dofid in range(tid, nv, wp.block_dim()):                                           <L 1115>
        var_244 = builtin_block_dim();
        var_245 = wp::range(var_1, var_nv, var_244);
        start_for_7:;
            if (iter_cmp(var_245) == 0) goto end_for_7;
            var_246 = wp::iter_next(var_245);
            // search = ctx_search_in[worldid, dofid]                                             <L 1116>
            var_247 = wp::address(var_ctx_search_in, var_0, var_246);
            var_249 = wp::load(var_247);
            var_248 = wp::copy(var_249);
            // local_gauss += wp.vec2(                                                            <L 1117>
            // search * (efc_Ma_out[worldid, dofid] - qfrc_smooth_in[worldid, dofid]),            <L 1118>
            var_250 = wp::address(var_efc_Ma_out, var_0, var_246);
            var_251 = wp::address(var_qfrc_smooth_in, var_0, var_246);
            var_253 = wp::load(var_250);
            var_254 = wp::load(var_251);
            var_252 = wp::sub(var_253, var_254);
            var_255 = wp::mul(var_248, var_252);
            // 0.5 * search * ctx_mv_in[worldid, dofid],                                          <L 1119>
            var_257 = wp::mul(var_256, var_248);
            var_258 = wp::address(var_ctx_mv_in, var_0, var_246);
            var_260 = wp::load(var_258);
            var_259 = wp::mul(var_257, var_260);
            var_261 = wp::vec_t<2, wp::float32>(var_255, var_259);
            // local_gauss += wp.vec2(                                                            <L 1117>
            var_262 = wp::add(var_243, var_261);
            wp::assign(var_243, var_262);
            goto start_for_7;
        end_for_7:;
        // gauss_tile = wp.tile(local_gauss, preserve_type=True)                                  <L 1122>
        var_264 = wp::tile<wp::vec_t<2, wp::float32>>(var_243);
        // gauss_sum = wp.tile_reduce(wp.add, gauss_tile)                                         <L 1123>
        var_265 = wp::tile_reduce(wp::add, var_264);
        // gauss_reduced = gauss_sum[0]                                                           <L 1124>
        var_267 = wp::tile_extract(var_265, var_266);
        // ctx_quad_gauss = wp.vec3(ctx_gauss_in[worldid], gauss_reduced[0], gauss_reduced[1])       <L 1125>
        var_268 = wp::address(var_ctx_gauss_in, var_0);
        var_270 = wp::extract(var_267, var_269);
        var_272 = wp::extract(var_267, var_271);
        var_274 = wp::load(var_268);
        var_273 = wp::vec_t<3, wp::float32>(var_274, var_270, var_272);
        // p0 = wp.vec3(ctx_quad_gauss[0], ctx_quad_gauss[1], 2.0 * ctx_quad_gauss[2]) + p0_sum[0]       <L 1128>
        var_276 = wp::extract(var_273, var_275);
        var_278 = wp::extract(var_273, var_277);
        var_281 = wp::extract(var_273, var_280);
        var_282 = wp::mul(var_279, var_281);
        var_283 = wp::vec_t<3, wp::float32>(var_276, var_278, var_282);
        var_285 = wp::tile_extract(var_241, var_284);
        var_286 = wp::add(var_283, var_285);
        // lo_alpha_in = -math.safe_div(p0[1], p0[2])                                             <L 1131>
        var_288 = wp::extract(var_286, var_287);
        var_290 = wp::extract(var_286, var_289);
        var_291 = safe_div_0(var_288, var_290);
        var_292 = wp::neg(var_291);
        // local_lo_in = wp.vec3(0.0)                                                             <L 1133>
        var_294 = wp::vec_t<3, wp::float32>(var_293);
        // for efcid in range(tid, nefc, wp.block_dim()):                                         <L 1134>
        var_295 = builtin_block_dim();
        var_296 = wp::range(var_1, var_12, var_295);
        start_for_9:;
            if (iter_cmp(var_296) == 0) goto end_for_9;
            var_297 = wp::iter_next(var_296);
            // if wp.static(IS_ELLIPTIC):                                                         <L 1135>
            // efc_type = efc_type_in[worldid, efcid]                                             <L 1136>
            var_299 = wp::address(var_efc_type_in, var_0, var_297);
            var_301 = wp::load(var_299);
            var_300 = wp::copy(var_301);
            // efc_id = 0                                                                         <L 1137>
            // contact_friction = types.vec5(0.0)                                                 <L 1138>
            var_304 = wp::vec_t<5, wp::float32>(var_303);
            // efc_addr0 = int(0)                                                                 <L 1139>
            var_306 = wp::int(var_305);
            // ctx_quad = wp.vec3(0.0)                                                            <L 1140>
            var_308 = wp::vec_t<3, wp::float32>(var_307);
            // quad1 = wp.vec3(0.0)                                                               <L 1141>
            var_310 = wp::vec_t<3, wp::float32>(var_309);
            // quad2 = wp.vec3(0.0)                                                               <L 1142>
            var_312 = wp::vec_t<3, wp::float32>(var_311);
            // if efc_type == types.ConstraintType.CONTACT_ELLIPTIC:                              <L 1144>
            var_314 = (var_300 == var_313);
            if (var_314) {
                // efc_id = efc_id_in[worldid, efcid]                                             <L 1145>
                var_315 = wp::address(var_efc_id_in, var_0, var_297);
                var_317 = wp::load(var_315);
                var_316 = wp::copy(var_317);
                // contact_friction = contact_friction_in[efc_id]                                 <L 1146>
                var_318 = wp::address(var_contact_friction_in, var_316);
                var_320 = wp::load(var_318);
                var_319 = wp::copy(var_320);
                // efc_addr0 = contact_efc_address_in[efc_id, 0]                                  <L 1147>
                var_322 = wp::address(var_contact_efc_address_in, var_316, var_321);
                var_324 = wp::load(var_322);
                var_323 = wp::copy(var_324);
                // efc_addr1 = contact_efc_address_in[efc_id, 1]                                  <L 1148>
                var_326 = wp::address(var_contact_efc_address_in, var_316, var_325);
                var_328 = wp::load(var_326);
                var_327 = wp::copy(var_328);
                // efc_addr2 = contact_efc_address_in[efc_id, 2]                                  <L 1149>
                var_330 = wp::address(var_contact_efc_address_in, var_316, var_329);
                var_332 = wp::load(var_330);
                var_331 = wp::copy(var_332);
                // ctx_quad = ctx_quad_in[worldid, efcid]                                         <L 1150>
                var_333 = wp::address(var_ctx_quad_in, var_0, var_297);
                var_335 = wp::load(var_333);
                var_334 = wp::copy(var_335);
                // quad1 = ctx_quad_in[worldid, efc_addr1]                                        <L 1151>
                var_336 = wp::address(var_ctx_quad_in, var_0, var_327);
                var_338 = wp::load(var_336);
                var_337 = wp::copy(var_338);
                // quad2 = ctx_quad_in[worldid, efc_addr2]                                        <L 1152>
                var_339 = wp::address(var_ctx_quad_in, var_0, var_331);
                var_341 = wp::load(var_339);
                var_340 = wp::copy(var_341);
            }
            var_342 = wp::where(var_314, var_316, var_302);
            var_343 = wp::where(var_314, var_319, var_304);
            var_344 = wp::where(var_314, var_323, var_306);
            var_345 = wp::where(var_314, var_334, var_308);
            var_346 = wp::where(var_314, var_337, var_310);
            var_347 = wp::where(var_314, var_340, var_312);
            var_348 = wp::where(var_314, var_327, var_206);
            var_349 = wp::where(var_314, var_331, var_210);
            // local_lo_in += _compute_efc_eval_pt(                                               <L 1154>
            // efcid,                                                                             <L 1155>
            // lo_alpha_in,                                                                       <L 1156>
            // ne,                                                                                <L 1157>
            // nf,                                                                                <L 1158>
            // impratio_invsqrt,                                                                  <L 1159>
            // efc_type,                                                                          <L 1160>
            // efc_D_in[worldid],                                                                 <L 1161>
            var_350 = wp::slice_t(var_0, var_0, var_351);
            var_352 = wp::view(var_efc_D_in, var_350);
            // efc_frictionloss_in[worldid],                                                      <L 1162>
            var_353 = wp::slice_t(var_0, var_0, var_354);
            var_355 = wp::view(var_efc_frictionloss_in, var_353);
            // ctx_Jaref_in[worldid, efcid],                                                      <L 1163>
            var_356 = wp::address(var_ctx_Jaref_in, var_0, var_297);
            // ctx_jv_in[worldid, efcid],                                                         <L 1164>
            var_357 = wp::address(var_ctx_jv_in, var_0, var_297);
            // ctx_quad,                                                                          <L 1165>
            // contact_friction,                                                                  <L 1166>
            // efc_addr0,                                                                         <L 1167>
            // quad1,                                                                             <L 1168>
            // quad2,                                                                             <L 1169>
            var_359 = wp::load(var_356);
            var_360 = wp::load(var_357);
            var_358 = _compute_efc_eval_pt_elliptic_0(var_297, var_292, var_6, var_9, var_22, var_300, var_352, var_355, var_359, var_360, var_345, var_343, var_344, var_346, var_347);
            // local_lo_in += _compute_efc_eval_pt(                                               <L 1154>
            var_361 = wp::add(var_294, var_358);
            wp::assign(var_179, var_300);
            wp::assign(var_221, var_342);
            wp::assign(var_222, var_343);
            wp::assign(var_223, var_344);
            wp::assign(var_224, var_345);
            wp::assign(var_225, var_346);
            wp::assign(var_226, var_347);
            wp::assign(var_206, var_348);
            wp::assign(var_210, var_349);
            wp::assign(var_294, var_361);
            goto start_for_9;
        end_for_9:;
        // lo_in_tile = wp.tile(local_lo_in, preserve_type=True)                                  <L 1184>
        var_363 = wp::tile<wp::vec_t<3, wp::float32>>(var_294);
        // lo_in_sum = wp.tile_reduce(wp.add, lo_in_tile)                                         <L 1185>
        var_364 = wp::tile_reduce(wp::add, var_363);
        // lo_in = _eval_pt(ctx_quad_gauss, lo_alpha_in) + lo_in_sum[0]                           <L 1186>
        var_365 = _eval_pt_0(var_273, var_292);
        var_367 = wp::tile_extract(var_364, var_366);
        var_368 = wp::add(var_365, var_367);
        // initial_converged = wp.abs(lo_in[1]) < gtol and lo_in[0] < p0[0]                       <L 1189>
        var_370 = wp::extract(var_368, var_369);
        var_371 = wp::abs(var_370);
        var_372 = (var_371 < var_171);
        var_374 = wp::extract(var_368, var_373);
        var_376 = wp::extract(var_286, var_375);
        var_377 = (var_374 < var_376);
        var_378 = var_372 && var_377;
        // if not initial_converged:                                                              <L 1192>
        var_379 = wp::unot(var_378);
        if (var_379) {
            // alpha = float(0.0)                                                                 <L 1193>
            var_381 = wp::float(var_380);
            // lo_less = lo_in[1] < p0[1]                                                         <L 1196>
            var_383 = wp::extract(var_368, var_382);
            var_385 = wp::extract(var_286, var_384);
            var_386 = (var_383 < var_385);
            // lo = wp.where(lo_less, lo_in, p0)                                                  <L 1197>
            var_387 = wp::where(var_386, var_368, var_286);
            // lo_alpha = wp.where(lo_less, lo_alpha_in, 0.0)                                     <L 1198>
            var_389 = wp::where(var_386, var_292, var_388);
            // hi = wp.where(lo_less, p0, lo_in)                                                  <L 1199>
            var_390 = wp::where(var_386, var_286, var_368);
            // hi_alpha = wp.where(lo_less, 0.0, lo_alpha_in)                                     <L 1200>
            var_392 = wp::where(var_386, var_391, var_292);
            // for _ in range(LS_ITERATIONS):                                                     <L 1202>
            var_394 = wp::range(var_393);
            start_for_11:;
                if (iter_cmp(var_394) == 0) goto end_for_11;
                var_395 = wp::iter_next(var_394);
                // lo_next_alpha = lo_alpha - math.safe_div(lo[1], lo[2])                         <L 1203>
                var_397 = wp::extract(var_387, var_396);
                var_399 = wp::extract(var_387, var_398);
                var_400 = safe_div_0(var_397, var_399);
                var_401 = wp::sub(var_389, var_400);
                // hi_next_alpha = hi_alpha - math.safe_div(hi[1], hi[2])                         <L 1204>
                var_403 = wp::extract(var_390, var_402);
                var_405 = wp::extract(var_390, var_404);
                var_406 = safe_div_0(var_403, var_405);
                var_407 = wp::sub(var_392, var_406);
                // mid_alpha = 0.5 * (lo_alpha + hi_alpha)                                        <L 1205>
                var_409 = wp::add(var_389, var_392);
                var_410 = wp::mul(var_408, var_409);
                // local_lo = wp.vec3(0.0)                                                        <L 1207>
                var_412 = wp::vec_t<3, wp::float32>(var_411);
                // local_hi = wp.vec3(0.0)                                                        <L 1208>
                var_414 = wp::vec_t<3, wp::float32>(var_413);
                // local_mid = wp.vec3(0.0)                                                       <L 1209>
                var_416 = wp::vec_t<3, wp::float32>(var_415);
                // for efcid in range(tid, nefc, wp.block_dim()):                                 <L 1211>
                var_417 = builtin_block_dim();
                var_418 = wp::range(var_1, var_12, var_417);
                start_for_13:;
                    if (iter_cmp(var_418) == 0) goto end_for_13;
                    var_419 = wp::iter_next(var_418);
                    // if wp.static(IS_ELLIPTIC):                                                 <L 1212>
                    // efc_type = efc_type_in[worldid, efcid]                                     <L 1213>
                    var_421 = wp::address(var_efc_type_in, var_0, var_419);
                    var_423 = wp::load(var_421);
                    var_422 = wp::copy(var_423);
                    // efc_id = 0                                                                 <L 1214>
                    // contact_friction = types.vec5(0.0)                                         <L 1215>
                    var_426 = wp::vec_t<5, wp::float32>(var_425);
                    // efc_addr0 = int(0)                                                         <L 1216>
                    var_428 = wp::int(var_427);
                    // ctx_quad = wp.vec3(0.0)                                                    <L 1217>
                    var_430 = wp::vec_t<3, wp::float32>(var_429);
                    // quad1 = wp.vec3(0.0)                                                       <L 1218>
                    var_432 = wp::vec_t<3, wp::float32>(var_431);
                    // quad2 = wp.vec3(0.0)                                                       <L 1219>
                    var_434 = wp::vec_t<3, wp::float32>(var_433);
                    // if efc_type == types.ConstraintType.CONTACT_ELLIPTIC:                      <L 1221>
                    var_436 = (var_422 == var_435);
                    if (var_436) {
                        // efc_id = efc_id_in[worldid, efcid]                                     <L 1222>
                        var_437 = wp::address(var_efc_id_in, var_0, var_419);
                        var_439 = wp::load(var_437);
                        var_438 = wp::copy(var_439);
                        // contact_friction = contact_friction_in[efc_id]                         <L 1223>
                        var_440 = wp::address(var_contact_friction_in, var_438);
                        var_442 = wp::load(var_440);
                        var_441 = wp::copy(var_442);
                        // efc_addr0 = contact_efc_address_in[efc_id, 0]                          <L 1224>
                        var_444 = wp::address(var_contact_efc_address_in, var_438, var_443);
                        var_446 = wp::load(var_444);
                        var_445 = wp::copy(var_446);
                        // efc_addr1 = contact_efc_address_in[efc_id, 1]                          <L 1225>
                        var_448 = wp::address(var_contact_efc_address_in, var_438, var_447);
                        var_450 = wp::load(var_448);
                        var_449 = wp::copy(var_450);
                        // efc_addr2 = contact_efc_address_in[efc_id, 2]                          <L 1226>
                        var_452 = wp::address(var_contact_efc_address_in, var_438, var_451);
                        var_454 = wp::load(var_452);
                        var_453 = wp::copy(var_454);
                        // ctx_quad = ctx_quad_in[worldid, efcid]                                 <L 1227>
                        var_455 = wp::address(var_ctx_quad_in, var_0, var_419);
                        var_457 = wp::load(var_455);
                        var_456 = wp::copy(var_457);
                        // quad1 = ctx_quad_in[worldid, efc_addr1]                                <L 1228>
                        var_458 = wp::address(var_ctx_quad_in, var_0, var_449);
                        var_460 = wp::load(var_458);
                        var_459 = wp::copy(var_460);
                        // quad2 = ctx_quad_in[worldid, efc_addr2]                                <L 1229>
                        var_461 = wp::address(var_ctx_quad_in, var_0, var_453);
                        var_463 = wp::load(var_461);
                        var_462 = wp::copy(var_463);
                    }
                    var_464 = wp::where(var_436, var_438, var_424);
                    var_465 = wp::where(var_436, var_441, var_426);
                    var_466 = wp::where(var_436, var_445, var_428);
                    var_467 = wp::where(var_436, var_456, var_430);
                    var_468 = wp::where(var_436, var_459, var_432);
                    var_469 = wp::where(var_436, var_462, var_434);
                    var_470 = wp::where(var_436, var_449, var_206);
                    var_471 = wp::where(var_436, var_453, var_210);
                    // r_lo, r_hi, r_mid = _compute_efc_eval_pt_3alphas(                          <L 1231>
                    // efcid,                                                                     <L 1232>
                    // lo_next_alpha,                                                             <L 1233>
                    // hi_next_alpha,                                                             <L 1234>
                    // mid_alpha,                                                                 <L 1235>
                    // ne,                                                                        <L 1236>
                    // nf,                                                                        <L 1237>
                    // impratio_invsqrt,                                                          <L 1238>
                    // efc_type,                                                                  <L 1239>
                    // efc_D_in[worldid],                                                         <L 1240>
                    var_472 = wp::slice_t(var_0, var_0, var_473);
                    var_474 = wp::view(var_efc_D_in, var_472);
                    // efc_frictionloss_in[worldid],                                              <L 1241>
                    var_475 = wp::slice_t(var_0, var_0, var_476);
                    var_477 = wp::view(var_efc_frictionloss_in, var_475);
                    // ctx_Jaref_in[worldid, efcid],                                              <L 1242>
                    var_478 = wp::address(var_ctx_Jaref_in, var_0, var_419);
                    // ctx_jv_in[worldid, efcid],                                                 <L 1243>
                    var_479 = wp::address(var_ctx_jv_in, var_0, var_419);
                    // ctx_quad,                                                                  <L 1244>
                    // contact_friction,                                                          <L 1245>
                    // efc_addr0,                                                                 <L 1246>
                    // quad1,                                                                     <L 1247>
                    // quad2,                                                                     <L 1248>
                    var_483 = wp::load(var_478);
                    var_484 = wp::load(var_479);
                    _compute_efc_eval_pt_3alphas_elliptic_0(var_419, var_401, var_407, var_410, var_6, var_9, var_22, var_422, var_474, var_477, var_483, var_484, var_467, var_465, var_466, var_468, var_469, var_480, var_481, var_482);
                    // local_lo += r_lo                                                           <L 1264>
                    var_485 = wp::add(var_412, var_480);
                    // local_hi += r_hi                                                           <L 1265>
                    var_486 = wp::add(var_414, var_481);
                    // local_mid += r_mid                                                         <L 1266>
                    var_487 = wp::add(var_416, var_482);
                    wp::assign(var_179, var_422);
                    wp::assign(var_221, var_464);
                    wp::assign(var_222, var_465);
                    wp::assign(var_223, var_466);
                    wp::assign(var_224, var_467);
                    wp::assign(var_225, var_468);
                    wp::assign(var_226, var_469);
                    wp::assign(var_206, var_470);
                    wp::assign(var_210, var_471);
                    wp::assign(var_412, var_485);
                    wp::assign(var_414, var_486);
                    wp::assign(var_416, var_487);
                    goto start_for_13;
                end_for_13:;
                // local_combined = wp.mat33(                                                     <L 1269>
                // local_lo[0],                                                                   <L 1270>
                var_489 = wp::extract(var_412, var_488);
                // local_hi[0],                                                                   <L 1271>
                var_491 = wp::extract(var_414, var_490);
                // local_mid[0],                                                                  <L 1272>
                var_493 = wp::extract(var_416, var_492);
                // local_lo[1],                                                                   <L 1273>
                var_495 = wp::extract(var_412, var_494);
                // local_hi[1],                                                                   <L 1274>
                var_497 = wp::extract(var_414, var_496);
                // local_mid[1],                                                                  <L 1275>
                var_499 = wp::extract(var_416, var_498);
                // local_lo[2],                                                                   <L 1276>
                var_501 = wp::extract(var_412, var_500);
                // local_hi[2],                                                                   <L 1277>
                var_503 = wp::extract(var_414, var_502);
                // local_mid[2],                                                                  <L 1278>
                var_505 = wp::extract(var_416, var_504);
                var_506 = wp::mat_t<3, 3, wp::float32>(var_489, var_491, var_493, var_495, var_497, var_499, var_501, var_503, var_505);
                // combined_tile = wp.tile(local_combined, preserve_type=True)                    <L 1283>
                var_508 = wp::tile<wp::mat_t<3, 3, wp::float32>>(var_506);
                // combined_sum = wp.tile_reduce(wp.add, combined_tile)                           <L 1284>
                var_509 = wp::tile_reduce(wp::add, var_508);
                // result = combined_sum[0]                                                       <L 1285>
                var_511 = wp::tile_extract(var_509, var_510);
                // gauss_lo, gauss_hi, gauss_mid = _eval_pt_3alphas(ctx_quad_gauss, lo_next_alpha, hi_next_alpha, mid_alpha)       <L 1288>
                _eval_pt_3alphas_0(var_273, var_401, var_407, var_410, var_512, var_513, var_514);
                // lo_next = gauss_lo + wp.vec3(result[0, 0], result[1, 0], result[2, 0])         <L 1289>
                var_517 = wp::extract(var_511, var_515, var_516);
                var_520 = wp::extract(var_511, var_518, var_519);
                var_523 = wp::extract(var_511, var_521, var_522);
                var_524 = wp::vec_t<3, wp::float32>(var_517, var_520, var_523);
                var_525 = wp::add(var_512, var_524);
                // hi_next = gauss_hi + wp.vec3(result[0, 1], result[1, 1], result[2, 1])         <L 1290>
                var_528 = wp::extract(var_511, var_526, var_527);
                var_531 = wp::extract(var_511, var_529, var_530);
                var_534 = wp::extract(var_511, var_532, var_533);
                var_535 = wp::vec_t<3, wp::float32>(var_528, var_531, var_534);
                var_536 = wp::add(var_513, var_535);
                // mid = gauss_mid + wp.vec3(result[0, 2], result[1, 2], result[2, 2])            <L 1291>
                var_539 = wp::extract(var_511, var_537, var_538);
                var_542 = wp::extract(var_511, var_540, var_541);
                var_545 = wp::extract(var_511, var_543, var_544);
                var_546 = wp::vec_t<3, wp::float32>(var_539, var_542, var_545);
                var_547 = wp::add(var_514, var_546);
                // swap_lo_lo_next = _in_bracket(lo, lo_next)                                     <L 1295>
                var_548 = _in_bracket_0(var_387, var_525);
                // lo = wp.where(swap_lo_lo_next, lo_next, lo)                                    <L 1296>
                var_549 = wp::where(var_548, var_525, var_387);
                // lo_alpha = wp.where(swap_lo_lo_next, lo_next_alpha, lo_alpha)                  <L 1297>
                var_550 = wp::where(var_548, var_401, var_389);
                // swap_lo_mid = _in_bracket(lo, mid)                                             <L 1298>
                var_551 = _in_bracket_0(var_549, var_547);
                // lo = wp.where(swap_lo_mid, mid, lo)                                            <L 1299>
                var_552 = wp::where(var_551, var_547, var_549);
                // lo_alpha = wp.where(swap_lo_mid, mid_alpha, lo_alpha)                          <L 1300>
                var_553 = wp::where(var_551, var_410, var_550);
                // swap_lo_hi_next = _in_bracket(lo, hi_next)                                     <L 1301>
                var_554 = _in_bracket_0(var_552, var_536);
                // lo = wp.where(swap_lo_hi_next, hi_next, lo)                                    <L 1302>
                var_555 = wp::where(var_554, var_536, var_552);
                // lo_alpha = wp.where(swap_lo_hi_next, hi_next_alpha, lo_alpha)                  <L 1303>
                var_556 = wp::where(var_554, var_407, var_553);
                // swap_lo = swap_lo_lo_next or swap_lo_mid or swap_lo_hi_next                    <L 1304>
                var_557 = var_548 || var_551 || var_554;
                // swap_hi_hi_next = _in_bracket(hi, hi_next)                                     <L 1307>
                var_558 = _in_bracket_0(var_390, var_536);
                // hi = wp.where(swap_hi_hi_next, hi_next, hi)                                    <L 1308>
                var_559 = wp::where(var_558, var_536, var_390);
                // hi_alpha = wp.where(swap_hi_hi_next, hi_next_alpha, hi_alpha)                  <L 1309>
                var_560 = wp::where(var_558, var_407, var_392);
                // swap_hi_mid = _in_bracket(hi, mid)                                             <L 1310>
                var_561 = _in_bracket_0(var_559, var_547);
                // hi = wp.where(swap_hi_mid, mid, hi)                                            <L 1311>
                var_562 = wp::where(var_561, var_547, var_559);
                // hi_alpha = wp.where(swap_hi_mid, mid_alpha, hi_alpha)                          <L 1312>
                var_563 = wp::where(var_561, var_410, var_560);
                // swap_hi_lo_next = _in_bracket(hi, lo_next)                                     <L 1313>
                var_564 = _in_bracket_0(var_562, var_525);
                // hi = wp.where(swap_hi_lo_next, lo_next, hi)                                    <L 1314>
                var_565 = wp::where(var_564, var_525, var_562);
                // hi_alpha = wp.where(swap_hi_lo_next, lo_next_alpha, hi_alpha)                  <L 1315>
                var_566 = wp::where(var_564, var_401, var_563);
                // swap_hi = swap_hi_hi_next or swap_hi_mid or swap_hi_lo_next                    <L 1316>
                var_567 = var_558 || var_561 || var_564;
                // ls_done = (not swap_lo and not swap_hi) or (lo[1] < 0.0 and lo[1] > -gtol) or (hi[1] > 0.0 and hi[1] < gtol)       <L 1319>
                var_568 = wp::unot(var_557);
                var_569 = wp::unot(var_567);
                var_570 = var_568 && var_569;
                var_572 = wp::extract(var_555, var_571);
                var_574 = (var_572 < var_573);
                var_576 = wp::extract(var_555, var_575);
                var_577 = wp::neg(var_171);
                var_578 = (var_576 > var_577);
                var_579 = var_574 && var_578;
                var_581 = wp::extract(var_565, var_580);
                var_583 = (var_581 > var_582);
                var_585 = wp::extract(var_565, var_584);
                var_586 = (var_585 < var_171);
                var_587 = var_583 && var_586;
                var_588 = var_570 || var_579 || var_587;
                // improved = lo[0] < p0[0] or hi[0] < p0[0]                                      <L 1322>
                var_590 = wp::extract(var_555, var_589);
                var_592 = wp::extract(var_286, var_591);
                var_593 = (var_590 < var_592);
                var_595 = wp::extract(var_565, var_594);
                var_597 = wp::extract(var_286, var_596);
                var_598 = (var_595 < var_597);
                var_599 = var_593 || var_598;
                // lo_better = lo[0] < hi[0]                                                      <L 1323>
                var_601 = wp::extract(var_555, var_600);
                var_603 = wp::extract(var_565, var_602);
                var_604 = (var_601 < var_603);
                // alpha = wp.where(improved and lo_better, lo_alpha, alpha)                      <L 1324>
                var_605 = var_599 && var_604;
                var_606 = wp::where(var_605, var_556, var_381);
                // alpha = wp.where(improved and not lo_better, hi_alpha, alpha)                  <L 1325>
                var_607 = wp::unot(var_604);
                var_608 = var_599 && var_607;
                var_609 = wp::where(var_608, var_566, var_606);
                // if ls_done:                                                                    <L 1327>
                if (var_588) {
                    // break                                                                      <L 1328>
                    wp::assign(var_297, var_419);
                    wp::assign(var_381, var_609);
                    wp::assign(var_387, var_555);
                    wp::assign(var_389, var_556);
                    wp::assign(var_390, var_565);
                    wp::assign(var_392, var_566);
                    goto end_for_11;
                }
                var_610 = wp::where(var_588, var_297, var_419);
                var_611 = wp::where(var_588, var_381, var_609);
                var_612 = wp::where(var_588, var_387, var_555);
                var_613 = wp::where(var_588, var_389, var_556);
                var_614 = wp::where(var_588, var_390, var_565);
                var_615 = wp::where(var_588, var_392, var_566);
                wp::assign(var_297, var_610);
                wp::assign(var_381, var_611);
                wp::assign(var_387, var_612);
                wp::assign(var_389, var_613);
                wp::assign(var_390, var_614);
                wp::assign(var_392, var_615);
                goto start_for_11;
            end_for_11:;
        }
        if (!var_379) {
            // alpha = lo_alpha_in                                                                <L 1330>
            var_616 = wp::copy(var_292);
        }
        var_617 = wp::where(var_379, var_381, var_616);
        // for dofid in range(tid, nv, wp.block_dim()):                                           <L 1333>
        var_618 = builtin_block_dim();
        var_619 = wp::range(var_1, var_nv, var_618);
        start_for_15:;
            if (iter_cmp(var_619) == 0) goto end_for_15;
            var_620 = wp::iter_next(var_619);
            // qacc_out[worldid, dofid] += alpha * ctx_search_in[worldid, dofid]                  <L 1334>
            var_621 = wp::address(var_ctx_search_in, var_0, var_620);
            var_623 = wp::load(var_621);
            var_622 = wp::mul(var_617, var_623);
            var_624 = wp::atomic_add(var_qacc_out, var_0, var_620, var_622);
            // efc_Ma_out[worldid, dofid] += alpha * ctx_mv_in[worldid, dofid]                    <L 1335>
            var_625 = wp::address(var_ctx_mv_in, var_0, var_620);
            var_627 = wp::load(var_625);
            var_626 = wp::mul(var_617, var_627);
            var_628 = wp::atomic_add(var_efc_Ma_out, var_0, var_620, var_626);
            goto start_for_15;
        end_for_15:;
        // for efcid in range(tid, nefc, wp.block_dim()):                                         <L 1338>
        var_629 = builtin_block_dim();
        var_630 = wp::range(var_1, var_12, var_629);
        start_for_17:;
            if (iter_cmp(var_630) == 0) goto end_for_17;
            var_631 = wp::iter_next(var_630);
            // ctx_Jaref_out[worldid, efcid] += alpha * ctx_jv_in[worldid, efcid]                 <L 1339>
            var_632 = wp::address(var_ctx_jv_in, var_0, var_631);
            var_634 = wp::load(var_632);
            var_633 = wp::mul(var_617, var_634);
            var_635 = wp::atomic_add(var_ctx_Jaref_out, var_0, var_631, var_633);
            goto start_for_17;
        end_for_17:;
    }
}

