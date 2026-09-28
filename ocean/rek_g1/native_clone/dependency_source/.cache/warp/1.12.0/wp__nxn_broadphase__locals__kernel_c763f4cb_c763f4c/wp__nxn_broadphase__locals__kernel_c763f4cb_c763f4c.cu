
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_driver.py:90
static CUDA_CALLABLE bool _plane_filter_0(
    wp::float32 var_size1,
    wp::float32 var_size2,
    wp::float32 var_margin1,
    wp::float32 var_margin2,
    wp::vec_t<3, wp::float32> var_xpos1,
    wp::vec_t<3, wp::float32> var_xpos2,
    wp::mat_t<3, 3, wp::float32> var_xmat1,
    wp::mat_t<3, 3, wp::float32> var_xmat2)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    bool var_1;
    wp::vec_t<3, wp::float32> var_2;
    const wp::int32 var_3 = 0;
    const wp::int32 var_4 = 2;
    wp::float32 var_5;
    const wp::int32 var_6 = 1;
    const wp::int32 var_7 = 2;
    wp::float32 var_8;
    const wp::int32 var_9 = 2;
    const wp::int32 var_10 = 2;
    wp::float32 var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    bool var_16;
    const wp::float32 var_17 = 0.0;
    bool var_18;
    wp::vec_t<3, wp::float32> var_19;
    const wp::int32 var_20 = 0;
    const wp::int32 var_21 = 2;
    wp::float32 var_22;
    const wp::int32 var_23 = 1;
    const wp::int32 var_24 = 2;
    wp::float32 var_25;
    const wp::int32 var_26 = 2;
    const wp::int32 var_27 = 2;
    wp::float32 var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    bool var_33;
    wp::float32 var_34;
    wp::float32 var_35;
    const bool var_36 = true;
    //---------
    // forward
    // def _plane_filter(                                                                     <L 91>
    // if size1 == 0.0:                                                                       <L 94>
    var_1 = (var_size1 == var_0);
    if (var_1) {
        // dist = wp.dot(xpos2 - xpos1, wp.vec3(xmat1[0, 2], xmat1[1, 2], xmat1[2, 2]))       <L 96>
        var_2 = wp::sub(var_xpos2, var_xpos1);
        var_5 = wp::extract(var_xmat1, var_3, var_4);
        var_8 = wp::extract(var_xmat1, var_6, var_7);
        var_11 = wp::extract(var_xmat1, var_9, var_10);
        var_12 = wp::vec_t<3, wp::float32>(var_5, var_8, var_11);
        var_13 = wp::dot(var_2, var_12);
        // return dist <= size2 + margin1 + margin2                                           <L 97>
        var_14 = wp::add(var_size2, var_margin1);
        var_15 = wp::add(var_14, var_margin2);
        var_16 = (var_13 <= var_15);
        return var_16;
    }
    if (!var_1) {
        // elif size2 == 0.0:                                                                 <L 98>
        var_18 = (var_size2 == var_17);
        if (var_18) {
            // dist = wp.dot(xpos1 - xpos2, wp.vec3(xmat2[0, 2], xmat2[1, 2], xmat2[2, 2]))       <L 100>
            var_19 = wp::sub(var_xpos1, var_xpos2);
            var_22 = wp::extract(var_xmat2, var_20, var_21);
            var_25 = wp::extract(var_xmat2, var_23, var_24);
            var_28 = wp::extract(var_xmat2, var_26, var_27);
            var_29 = wp::vec_t<3, wp::float32>(var_22, var_25, var_28);
            var_30 = wp::dot(var_19, var_29);
            // return dist <= size1 + margin1 + margin2                                       <L 101>
            var_31 = wp::add(var_size1, var_margin1);
            var_32 = wp::add(var_31, var_margin2);
            var_33 = (var_30 <= var_32);
            return var_33;
        }
        var_34 = wp::where(var_18, var_30, var_13);
    }
    var_35 = wp::where(var_1, var_13, var_34);
    // return True                                                                            <L 103>
    return var_36;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_driver.py:106
static CUDA_CALLABLE bool _sphere_filter_0(
    wp::float32 var_size1,
    wp::float32 var_size2,
    wp::float32 var_margin1,
    wp::float32 var_margin2,
    wp::vec_t<3, wp::float32> var_xpos1,
    wp::vec_t<3, wp::float32> var_xpos2)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    bool var_6;
    //---------
    // forward
    // def _sphere_filter(size1: float, size2: float, margin1: float, margin2: float, xpos1: wp.vec3, xpos2: wp.vec3) -> bool:       <L 107>
    // bound = size1 + size2 + margin1 + margin2                                              <L 108>
    var_0 = wp::add(var_size1, var_size2);
    var_1 = wp::add(var_0, var_margin1);
    var_2 = wp::add(var_1, var_margin2);
    // dif = xpos2 - xpos1                                                                    <L 109>
    var_3 = wp::sub(var_xpos2, var_xpos1);
    // dist_sq = wp.dot(dif, dif)                                                             <L 110>
    var_4 = wp::dot(var_3, var_3);
    // return dist_sq <= bound * bound                                                        <L 111>
    var_5 = wp::mul(var_2, var_2);
    var_6 = (var_4 <= var_5);
    return var_6;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_driver.py:217
static CUDA_CALLABLE bool _obb_filter_0(
    wp::vec_t<3, wp::float32> var_center1,
    wp::vec_t<3, wp::float32> var_center2,
    wp::vec_t<3, wp::float32> var_size1,
    wp::vec_t<3, wp::float32> var_size2,
    wp::float32 var_margin1,
    wp::float32 var_margin2,
    wp::vec_t<3, wp::float32> var_xpos1,
    wp::vec_t<3, wp::float32> var_xpos2,
    wp::mat_t<3, 3, wp::float32> var_xmat1,
    wp::mat_t<3, 3, wp::float32> var_xmat2)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::mat_t<2, 3, wp::float32> var_1;
    wp::mat_t<6, 3, wp::float32> var_2;
    wp::vec_t<2, wp::float32> var_3;
    wp::vec_t<2, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    const wp::int32 var_7 = 0;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    const wp::int32 var_10 = 1;
    const wp::int32 var_11 = 0;
    const wp::int32 var_12 = 0;
    wp::float32 var_13;
    const wp::int32 var_14 = 1;
    const wp::int32 var_15 = 0;
    wp::float32 var_16;
    const wp::int32 var_17 = 2;
    const wp::int32 var_18 = 0;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    const wp::int32 var_21 = 0;
    const wp::int32 var_22 = 0;
    const wp::int32 var_23 = 1;
    wp::float32 var_24;
    const wp::int32 var_25 = 1;
    const wp::int32 var_26 = 1;
    wp::float32 var_27;
    const wp::int32 var_28 = 2;
    const wp::int32 var_29 = 1;
    wp::float32 var_30;
    wp::vec_t<3, wp::float32> var_31;
    const wp::int32 var_32 = 1;
    const wp::int32 var_33 = 0;
    const wp::int32 var_34 = 2;
    wp::float32 var_35;
    const wp::int32 var_36 = 1;
    const wp::int32 var_37 = 2;
    wp::float32 var_38;
    const wp::int32 var_39 = 2;
    const wp::int32 var_40 = 2;
    wp::float32 var_41;
    wp::vec_t<3, wp::float32> var_42;
    const wp::int32 var_43 = 2;
    const wp::int32 var_44 = 0;
    const wp::int32 var_45 = 0;
    wp::float32 var_46;
    const wp::int32 var_47 = 1;
    const wp::int32 var_48 = 0;
    wp::float32 var_49;
    const wp::int32 var_50 = 2;
    const wp::int32 var_51 = 0;
    wp::float32 var_52;
    wp::vec_t<3, wp::float32> var_53;
    const wp::int32 var_54 = 3;
    const wp::int32 var_55 = 0;
    const wp::int32 var_56 = 1;
    wp::float32 var_57;
    const wp::int32 var_58 = 1;
    const wp::int32 var_59 = 1;
    wp::float32 var_60;
    const wp::int32 var_61 = 2;
    const wp::int32 var_62 = 1;
    wp::float32 var_63;
    wp::vec_t<3, wp::float32> var_64;
    const wp::int32 var_65 = 4;
    const wp::int32 var_66 = 0;
    const wp::int32 var_67 = 2;
    wp::float32 var_68;
    const wp::int32 var_69 = 1;
    const wp::int32 var_70 = 2;
    wp::float32 var_71;
    const wp::int32 var_72 = 2;
    const wp::int32 var_73 = 2;
    wp::float32 var_74;
    wp::vec_t<3, wp::float32> var_75;
    const wp::int32 var_76 = 5;
    const wp::int32 var_77 = 0;
    const wp::int32 var_78 = 0;
    const wp::int32 var_79 = 0;
    wp::vec_t<3, wp::float32> var_80;
    const wp::int32 var_81 = 3;
    wp::int32 var_82;
    wp::int32 var_83;
    wp::vec_t<3, wp::float32> var_84;
    wp::float32 var_85;
    const wp::int32 var_86 = 0;
    bool var_87;
    wp::vec_t<3, wp::float32> var_88;
    wp::vec_t<3, wp::float32> var_89;
    wp::vec_t<3, wp::float32> var_90;
    const wp::int32 var_91 = 0;
    wp::float32 var_92;
    const wp::int32 var_93 = 3;
    wp::int32 var_94;
    const wp::int32 var_95 = 0;
    wp::int32 var_96;
    wp::vec_t<3, wp::float32> var_97;
    const wp::int32 var_98 = 3;
    wp::int32 var_99;
    wp::int32 var_100;
    wp::vec_t<3, wp::float32> var_101;
    wp::float32 var_102;
    wp::float32 var_103;
    wp::float32 var_104;
    const wp::int32 var_105 = 1;
    wp::float32 var_106;
    const wp::int32 var_107 = 3;
    wp::int32 var_108;
    const wp::int32 var_109 = 1;
    wp::int32 var_110;
    wp::vec_t<3, wp::float32> var_111;
    const wp::int32 var_112 = 3;
    wp::int32 var_113;
    wp::int32 var_114;
    wp::vec_t<3, wp::float32> var_115;
    wp::float32 var_116;
    wp::float32 var_117;
    wp::float32 var_118;
    wp::float32 var_119;
    const wp::int32 var_120 = 2;
    wp::float32 var_121;
    const wp::int32 var_122 = 3;
    wp::int32 var_123;
    const wp::int32 var_124 = 2;
    wp::int32 var_125;
    wp::vec_t<3, wp::float32> var_126;
    const wp::int32 var_127 = 3;
    wp::int32 var_128;
    wp::int32 var_129;
    wp::vec_t<3, wp::float32> var_130;
    wp::float32 var_131;
    wp::float32 var_132;
    wp::float32 var_133;
    wp::float32 var_134;
    const wp::int32 var_135 = 1;
    wp::vec_t<3, wp::float32> var_136;
    const wp::int32 var_137 = 3;
    wp::int32 var_138;
    wp::int32 var_139;
    wp::vec_t<3, wp::float32> var_140;
    wp::float32 var_141;
    const wp::int32 var_142 = 0;
    bool var_143;
    wp::vec_t<3, wp::float32> var_144;
    wp::vec_t<3, wp::float32> var_145;
    wp::vec_t<3, wp::float32> var_146;
    wp::vec_t<3, wp::float32> var_147;
    const wp::int32 var_148 = 0;
    wp::float32 var_149;
    const wp::int32 var_150 = 3;
    wp::int32 var_151;
    const wp::int32 var_152 = 0;
    wp::int32 var_153;
    wp::vec_t<3, wp::float32> var_154;
    const wp::int32 var_155 = 3;
    wp::int32 var_156;
    wp::int32 var_157;
    wp::vec_t<3, wp::float32> var_158;
    wp::float32 var_159;
    wp::float32 var_160;
    wp::float32 var_161;
    const wp::int32 var_162 = 1;
    wp::float32 var_163;
    const wp::int32 var_164 = 3;
    wp::int32 var_165;
    const wp::int32 var_166 = 1;
    wp::int32 var_167;
    wp::vec_t<3, wp::float32> var_168;
    const wp::int32 var_169 = 3;
    wp::int32 var_170;
    wp::int32 var_171;
    wp::vec_t<3, wp::float32> var_172;
    wp::float32 var_173;
    wp::float32 var_174;
    wp::float32 var_175;
    wp::float32 var_176;
    const wp::int32 var_177 = 2;
    wp::float32 var_178;
    const wp::int32 var_179 = 3;
    wp::int32 var_180;
    const wp::int32 var_181 = 2;
    wp::int32 var_182;
    wp::vec_t<3, wp::float32> var_183;
    const wp::int32 var_184 = 3;
    wp::int32 var_185;
    wp::int32 var_186;
    wp::vec_t<3, wp::float32> var_187;
    wp::float32 var_188;
    wp::float32 var_189;
    wp::float32 var_190;
    wp::float32 var_191;
    const wp::int32 var_192 = 0;
    wp::float32 var_193;
    const wp::int32 var_194 = 1;
    wp::float32 var_195;
    wp::float32 var_196;
    wp::float32 var_197;
    const wp::int32 var_198 = 1;
    wp::float32 var_199;
    const wp::int32 var_200 = 0;
    wp::float32 var_201;
    wp::float32 var_202;
    wp::float32 var_203;
    bool var_204;
    const bool var_205 = false;
    const wp::int32 var_206 = 1;
    const wp::int32 var_207 = 0;
    wp::vec_t<3, wp::float32> var_208;
    const wp::int32 var_209 = 3;
    wp::int32 var_210;
    wp::int32 var_211;
    wp::vec_t<3, wp::float32> var_212;
    wp::float32 var_213;
    const wp::int32 var_214 = 0;
    bool var_215;
    wp::vec_t<3, wp::float32> var_216;
    wp::vec_t<3, wp::float32> var_217;
    wp::vec_t<3, wp::float32> var_218;
    wp::vec_t<3, wp::float32> var_219;
    const wp::int32 var_220 = 0;
    wp::float32 var_221;
    const wp::int32 var_222 = 3;
    wp::int32 var_223;
    const wp::int32 var_224 = 0;
    wp::int32 var_225;
    wp::vec_t<3, wp::float32> var_226;
    const wp::int32 var_227 = 3;
    wp::int32 var_228;
    wp::int32 var_229;
    wp::vec_t<3, wp::float32> var_230;
    wp::float32 var_231;
    wp::float32 var_232;
    wp::float32 var_233;
    const wp::int32 var_234 = 1;
    wp::float32 var_235;
    const wp::int32 var_236 = 3;
    wp::int32 var_237;
    const wp::int32 var_238 = 1;
    wp::int32 var_239;
    wp::vec_t<3, wp::float32> var_240;
    const wp::int32 var_241 = 3;
    wp::int32 var_242;
    wp::int32 var_243;
    wp::vec_t<3, wp::float32> var_244;
    wp::float32 var_245;
    wp::float32 var_246;
    wp::float32 var_247;
    wp::float32 var_248;
    const wp::int32 var_249 = 2;
    wp::float32 var_250;
    const wp::int32 var_251 = 3;
    wp::int32 var_252;
    const wp::int32 var_253 = 2;
    wp::int32 var_254;
    wp::vec_t<3, wp::float32> var_255;
    const wp::int32 var_256 = 3;
    wp::int32 var_257;
    wp::int32 var_258;
    wp::vec_t<3, wp::float32> var_259;
    wp::float32 var_260;
    wp::float32 var_261;
    wp::float32 var_262;
    wp::float32 var_263;
    const wp::int32 var_264 = 1;
    wp::vec_t<3, wp::float32> var_265;
    const wp::int32 var_266 = 3;
    wp::int32 var_267;
    wp::int32 var_268;
    wp::vec_t<3, wp::float32> var_269;
    wp::float32 var_270;
    const wp::int32 var_271 = 0;
    bool var_272;
    wp::vec_t<3, wp::float32> var_273;
    wp::vec_t<3, wp::float32> var_274;
    wp::vec_t<3, wp::float32> var_275;
    wp::vec_t<3, wp::float32> var_276;
    const wp::int32 var_277 = 0;
    wp::float32 var_278;
    const wp::int32 var_279 = 3;
    wp::int32 var_280;
    const wp::int32 var_281 = 0;
    wp::int32 var_282;
    wp::vec_t<3, wp::float32> var_283;
    const wp::int32 var_284 = 3;
    wp::int32 var_285;
    wp::int32 var_286;
    wp::vec_t<3, wp::float32> var_287;
    wp::float32 var_288;
    wp::float32 var_289;
    wp::float32 var_290;
    const wp::int32 var_291 = 1;
    wp::float32 var_292;
    const wp::int32 var_293 = 3;
    wp::int32 var_294;
    const wp::int32 var_295 = 1;
    wp::int32 var_296;
    wp::vec_t<3, wp::float32> var_297;
    const wp::int32 var_298 = 3;
    wp::int32 var_299;
    wp::int32 var_300;
    wp::vec_t<3, wp::float32> var_301;
    wp::float32 var_302;
    wp::float32 var_303;
    wp::float32 var_304;
    wp::float32 var_305;
    const wp::int32 var_306 = 2;
    wp::float32 var_307;
    const wp::int32 var_308 = 3;
    wp::int32 var_309;
    const wp::int32 var_310 = 2;
    wp::int32 var_311;
    wp::vec_t<3, wp::float32> var_312;
    const wp::int32 var_313 = 3;
    wp::int32 var_314;
    wp::int32 var_315;
    wp::vec_t<3, wp::float32> var_316;
    wp::float32 var_317;
    wp::float32 var_318;
    wp::float32 var_319;
    wp::float32 var_320;
    const wp::int32 var_321 = 0;
    wp::float32 var_322;
    const wp::int32 var_323 = 1;
    wp::float32 var_324;
    wp::float32 var_325;
    wp::float32 var_326;
    const wp::int32 var_327 = 1;
    wp::float32 var_328;
    const wp::int32 var_329 = 0;
    wp::float32 var_330;
    wp::float32 var_331;
    wp::float32 var_332;
    bool var_333;
    const bool var_334 = false;
    const wp::int32 var_335 = 2;
    const wp::int32 var_336 = 0;
    wp::vec_t<3, wp::float32> var_337;
    const wp::int32 var_338 = 3;
    wp::int32 var_339;
    wp::int32 var_340;
    wp::vec_t<3, wp::float32> var_341;
    wp::float32 var_342;
    const wp::int32 var_343 = 0;
    bool var_344;
    wp::vec_t<3, wp::float32> var_345;
    wp::vec_t<3, wp::float32> var_346;
    wp::vec_t<3, wp::float32> var_347;
    wp::vec_t<3, wp::float32> var_348;
    const wp::int32 var_349 = 0;
    wp::float32 var_350;
    const wp::int32 var_351 = 3;
    wp::int32 var_352;
    const wp::int32 var_353 = 0;
    wp::int32 var_354;
    wp::vec_t<3, wp::float32> var_355;
    const wp::int32 var_356 = 3;
    wp::int32 var_357;
    wp::int32 var_358;
    wp::vec_t<3, wp::float32> var_359;
    wp::float32 var_360;
    wp::float32 var_361;
    wp::float32 var_362;
    const wp::int32 var_363 = 1;
    wp::float32 var_364;
    const wp::int32 var_365 = 3;
    wp::int32 var_366;
    const wp::int32 var_367 = 1;
    wp::int32 var_368;
    wp::vec_t<3, wp::float32> var_369;
    const wp::int32 var_370 = 3;
    wp::int32 var_371;
    wp::int32 var_372;
    wp::vec_t<3, wp::float32> var_373;
    wp::float32 var_374;
    wp::float32 var_375;
    wp::float32 var_376;
    wp::float32 var_377;
    const wp::int32 var_378 = 2;
    wp::float32 var_379;
    const wp::int32 var_380 = 3;
    wp::int32 var_381;
    const wp::int32 var_382 = 2;
    wp::int32 var_383;
    wp::vec_t<3, wp::float32> var_384;
    const wp::int32 var_385 = 3;
    wp::int32 var_386;
    wp::int32 var_387;
    wp::vec_t<3, wp::float32> var_388;
    wp::float32 var_389;
    wp::float32 var_390;
    wp::float32 var_391;
    wp::float32 var_392;
    const wp::int32 var_393 = 1;
    wp::vec_t<3, wp::float32> var_394;
    const wp::int32 var_395 = 3;
    wp::int32 var_396;
    wp::int32 var_397;
    wp::vec_t<3, wp::float32> var_398;
    wp::float32 var_399;
    const wp::int32 var_400 = 0;
    bool var_401;
    wp::vec_t<3, wp::float32> var_402;
    wp::vec_t<3, wp::float32> var_403;
    wp::vec_t<3, wp::float32> var_404;
    wp::vec_t<3, wp::float32> var_405;
    const wp::int32 var_406 = 0;
    wp::float32 var_407;
    const wp::int32 var_408 = 3;
    wp::int32 var_409;
    const wp::int32 var_410 = 0;
    wp::int32 var_411;
    wp::vec_t<3, wp::float32> var_412;
    const wp::int32 var_413 = 3;
    wp::int32 var_414;
    wp::int32 var_415;
    wp::vec_t<3, wp::float32> var_416;
    wp::float32 var_417;
    wp::float32 var_418;
    wp::float32 var_419;
    const wp::int32 var_420 = 1;
    wp::float32 var_421;
    const wp::int32 var_422 = 3;
    wp::int32 var_423;
    const wp::int32 var_424 = 1;
    wp::int32 var_425;
    wp::vec_t<3, wp::float32> var_426;
    const wp::int32 var_427 = 3;
    wp::int32 var_428;
    wp::int32 var_429;
    wp::vec_t<3, wp::float32> var_430;
    wp::float32 var_431;
    wp::float32 var_432;
    wp::float32 var_433;
    wp::float32 var_434;
    const wp::int32 var_435 = 2;
    wp::float32 var_436;
    const wp::int32 var_437 = 3;
    wp::int32 var_438;
    const wp::int32 var_439 = 2;
    wp::int32 var_440;
    wp::vec_t<3, wp::float32> var_441;
    const wp::int32 var_442 = 3;
    wp::int32 var_443;
    wp::int32 var_444;
    wp::vec_t<3, wp::float32> var_445;
    wp::float32 var_446;
    wp::float32 var_447;
    wp::float32 var_448;
    wp::float32 var_449;
    const wp::int32 var_450 = 0;
    wp::float32 var_451;
    const wp::int32 var_452 = 1;
    wp::float32 var_453;
    wp::float32 var_454;
    wp::float32 var_455;
    const wp::int32 var_456 = 1;
    wp::float32 var_457;
    const wp::int32 var_458 = 0;
    wp::float32 var_459;
    wp::float32 var_460;
    wp::float32 var_461;
    bool var_462;
    const bool var_463 = false;
    const wp::int32 var_464 = 1;
    const wp::int32 var_465 = 0;
    const wp::int32 var_466 = 0;
    wp::vec_t<3, wp::float32> var_467;
    const wp::int32 var_468 = 3;
    wp::int32 var_469;
    wp::int32 var_470;
    wp::vec_t<3, wp::float32> var_471;
    wp::float32 var_472;
    const wp::int32 var_473 = 0;
    bool var_474;
    wp::vec_t<3, wp::float32> var_475;
    wp::vec_t<3, wp::float32> var_476;
    wp::vec_t<3, wp::float32> var_477;
    wp::vec_t<3, wp::float32> var_478;
    const wp::int32 var_479 = 0;
    wp::float32 var_480;
    const wp::int32 var_481 = 3;
    wp::int32 var_482;
    const wp::int32 var_483 = 0;
    wp::int32 var_484;
    wp::vec_t<3, wp::float32> var_485;
    const wp::int32 var_486 = 3;
    wp::int32 var_487;
    wp::int32 var_488;
    wp::vec_t<3, wp::float32> var_489;
    wp::float32 var_490;
    wp::float32 var_491;
    wp::float32 var_492;
    const wp::int32 var_493 = 1;
    wp::float32 var_494;
    const wp::int32 var_495 = 3;
    wp::int32 var_496;
    const wp::int32 var_497 = 1;
    wp::int32 var_498;
    wp::vec_t<3, wp::float32> var_499;
    const wp::int32 var_500 = 3;
    wp::int32 var_501;
    wp::int32 var_502;
    wp::vec_t<3, wp::float32> var_503;
    wp::float32 var_504;
    wp::float32 var_505;
    wp::float32 var_506;
    wp::float32 var_507;
    const wp::int32 var_508 = 2;
    wp::float32 var_509;
    const wp::int32 var_510 = 3;
    wp::int32 var_511;
    const wp::int32 var_512 = 2;
    wp::int32 var_513;
    wp::vec_t<3, wp::float32> var_514;
    const wp::int32 var_515 = 3;
    wp::int32 var_516;
    wp::int32 var_517;
    wp::vec_t<3, wp::float32> var_518;
    wp::float32 var_519;
    wp::float32 var_520;
    wp::float32 var_521;
    wp::float32 var_522;
    const wp::int32 var_523 = 1;
    wp::vec_t<3, wp::float32> var_524;
    const wp::int32 var_525 = 3;
    wp::int32 var_526;
    wp::int32 var_527;
    wp::vec_t<3, wp::float32> var_528;
    wp::float32 var_529;
    const wp::int32 var_530 = 0;
    bool var_531;
    wp::vec_t<3, wp::float32> var_532;
    wp::vec_t<3, wp::float32> var_533;
    wp::vec_t<3, wp::float32> var_534;
    wp::vec_t<3, wp::float32> var_535;
    const wp::int32 var_536 = 0;
    wp::float32 var_537;
    const wp::int32 var_538 = 3;
    wp::int32 var_539;
    const wp::int32 var_540 = 0;
    wp::int32 var_541;
    wp::vec_t<3, wp::float32> var_542;
    const wp::int32 var_543 = 3;
    wp::int32 var_544;
    wp::int32 var_545;
    wp::vec_t<3, wp::float32> var_546;
    wp::float32 var_547;
    wp::float32 var_548;
    wp::float32 var_549;
    const wp::int32 var_550 = 1;
    wp::float32 var_551;
    const wp::int32 var_552 = 3;
    wp::int32 var_553;
    const wp::int32 var_554 = 1;
    wp::int32 var_555;
    wp::vec_t<3, wp::float32> var_556;
    const wp::int32 var_557 = 3;
    wp::int32 var_558;
    wp::int32 var_559;
    wp::vec_t<3, wp::float32> var_560;
    wp::float32 var_561;
    wp::float32 var_562;
    wp::float32 var_563;
    wp::float32 var_564;
    const wp::int32 var_565 = 2;
    wp::float32 var_566;
    const wp::int32 var_567 = 3;
    wp::int32 var_568;
    const wp::int32 var_569 = 2;
    wp::int32 var_570;
    wp::vec_t<3, wp::float32> var_571;
    const wp::int32 var_572 = 3;
    wp::int32 var_573;
    wp::int32 var_574;
    wp::vec_t<3, wp::float32> var_575;
    wp::float32 var_576;
    wp::float32 var_577;
    wp::float32 var_578;
    wp::float32 var_579;
    const wp::int32 var_580 = 0;
    wp::float32 var_581;
    const wp::int32 var_582 = 1;
    wp::float32 var_583;
    wp::float32 var_584;
    wp::float32 var_585;
    const wp::int32 var_586 = 1;
    wp::float32 var_587;
    const wp::int32 var_588 = 0;
    wp::float32 var_589;
    wp::float32 var_590;
    wp::float32 var_591;
    bool var_592;
    const bool var_593 = false;
    const wp::int32 var_594 = 1;
    const wp::int32 var_595 = 0;
    wp::vec_t<3, wp::float32> var_596;
    const wp::int32 var_597 = 3;
    wp::int32 var_598;
    wp::int32 var_599;
    wp::vec_t<3, wp::float32> var_600;
    wp::float32 var_601;
    const wp::int32 var_602 = 0;
    bool var_603;
    wp::vec_t<3, wp::float32> var_604;
    wp::vec_t<3, wp::float32> var_605;
    wp::vec_t<3, wp::float32> var_606;
    wp::vec_t<3, wp::float32> var_607;
    const wp::int32 var_608 = 0;
    wp::float32 var_609;
    const wp::int32 var_610 = 3;
    wp::int32 var_611;
    const wp::int32 var_612 = 0;
    wp::int32 var_613;
    wp::vec_t<3, wp::float32> var_614;
    const wp::int32 var_615 = 3;
    wp::int32 var_616;
    wp::int32 var_617;
    wp::vec_t<3, wp::float32> var_618;
    wp::float32 var_619;
    wp::float32 var_620;
    wp::float32 var_621;
    const wp::int32 var_622 = 1;
    wp::float32 var_623;
    const wp::int32 var_624 = 3;
    wp::int32 var_625;
    const wp::int32 var_626 = 1;
    wp::int32 var_627;
    wp::vec_t<3, wp::float32> var_628;
    const wp::int32 var_629 = 3;
    wp::int32 var_630;
    wp::int32 var_631;
    wp::vec_t<3, wp::float32> var_632;
    wp::float32 var_633;
    wp::float32 var_634;
    wp::float32 var_635;
    wp::float32 var_636;
    const wp::int32 var_637 = 2;
    wp::float32 var_638;
    const wp::int32 var_639 = 3;
    wp::int32 var_640;
    const wp::int32 var_641 = 2;
    wp::int32 var_642;
    wp::vec_t<3, wp::float32> var_643;
    const wp::int32 var_644 = 3;
    wp::int32 var_645;
    wp::int32 var_646;
    wp::vec_t<3, wp::float32> var_647;
    wp::float32 var_648;
    wp::float32 var_649;
    wp::float32 var_650;
    wp::float32 var_651;
    const wp::int32 var_652 = 1;
    wp::vec_t<3, wp::float32> var_653;
    const wp::int32 var_654 = 3;
    wp::int32 var_655;
    wp::int32 var_656;
    wp::vec_t<3, wp::float32> var_657;
    wp::float32 var_658;
    const wp::int32 var_659 = 0;
    bool var_660;
    wp::vec_t<3, wp::float32> var_661;
    wp::vec_t<3, wp::float32> var_662;
    wp::vec_t<3, wp::float32> var_663;
    wp::vec_t<3, wp::float32> var_664;
    const wp::int32 var_665 = 0;
    wp::float32 var_666;
    const wp::int32 var_667 = 3;
    wp::int32 var_668;
    const wp::int32 var_669 = 0;
    wp::int32 var_670;
    wp::vec_t<3, wp::float32> var_671;
    const wp::int32 var_672 = 3;
    wp::int32 var_673;
    wp::int32 var_674;
    wp::vec_t<3, wp::float32> var_675;
    wp::float32 var_676;
    wp::float32 var_677;
    wp::float32 var_678;
    const wp::int32 var_679 = 1;
    wp::float32 var_680;
    const wp::int32 var_681 = 3;
    wp::int32 var_682;
    const wp::int32 var_683 = 1;
    wp::int32 var_684;
    wp::vec_t<3, wp::float32> var_685;
    const wp::int32 var_686 = 3;
    wp::int32 var_687;
    wp::int32 var_688;
    wp::vec_t<3, wp::float32> var_689;
    wp::float32 var_690;
    wp::float32 var_691;
    wp::float32 var_692;
    wp::float32 var_693;
    const wp::int32 var_694 = 2;
    wp::float32 var_695;
    const wp::int32 var_696 = 3;
    wp::int32 var_697;
    const wp::int32 var_698 = 2;
    wp::int32 var_699;
    wp::vec_t<3, wp::float32> var_700;
    const wp::int32 var_701 = 3;
    wp::int32 var_702;
    wp::int32 var_703;
    wp::vec_t<3, wp::float32> var_704;
    wp::float32 var_705;
    wp::float32 var_706;
    wp::float32 var_707;
    wp::float32 var_708;
    const wp::int32 var_709 = 0;
    wp::float32 var_710;
    const wp::int32 var_711 = 1;
    wp::float32 var_712;
    wp::float32 var_713;
    wp::float32 var_714;
    const wp::int32 var_715 = 1;
    wp::float32 var_716;
    const wp::int32 var_717 = 0;
    wp::float32 var_718;
    wp::float32 var_719;
    wp::float32 var_720;
    bool var_721;
    const bool var_722 = false;
    const wp::int32 var_723 = 2;
    const wp::int32 var_724 = 0;
    wp::vec_t<3, wp::float32> var_725;
    const wp::int32 var_726 = 3;
    wp::int32 var_727;
    wp::int32 var_728;
    wp::vec_t<3, wp::float32> var_729;
    wp::float32 var_730;
    const wp::int32 var_731 = 0;
    bool var_732;
    wp::vec_t<3, wp::float32> var_733;
    wp::vec_t<3, wp::float32> var_734;
    wp::vec_t<3, wp::float32> var_735;
    wp::vec_t<3, wp::float32> var_736;
    const wp::int32 var_737 = 0;
    wp::float32 var_738;
    const wp::int32 var_739 = 3;
    wp::int32 var_740;
    const wp::int32 var_741 = 0;
    wp::int32 var_742;
    wp::vec_t<3, wp::float32> var_743;
    const wp::int32 var_744 = 3;
    wp::int32 var_745;
    wp::int32 var_746;
    wp::vec_t<3, wp::float32> var_747;
    wp::float32 var_748;
    wp::float32 var_749;
    wp::float32 var_750;
    const wp::int32 var_751 = 1;
    wp::float32 var_752;
    const wp::int32 var_753 = 3;
    wp::int32 var_754;
    const wp::int32 var_755 = 1;
    wp::int32 var_756;
    wp::vec_t<3, wp::float32> var_757;
    const wp::int32 var_758 = 3;
    wp::int32 var_759;
    wp::int32 var_760;
    wp::vec_t<3, wp::float32> var_761;
    wp::float32 var_762;
    wp::float32 var_763;
    wp::float32 var_764;
    wp::float32 var_765;
    const wp::int32 var_766 = 2;
    wp::float32 var_767;
    const wp::int32 var_768 = 3;
    wp::int32 var_769;
    const wp::int32 var_770 = 2;
    wp::int32 var_771;
    wp::vec_t<3, wp::float32> var_772;
    const wp::int32 var_773 = 3;
    wp::int32 var_774;
    wp::int32 var_775;
    wp::vec_t<3, wp::float32> var_776;
    wp::float32 var_777;
    wp::float32 var_778;
    wp::float32 var_779;
    wp::float32 var_780;
    const wp::int32 var_781 = 1;
    wp::vec_t<3, wp::float32> var_782;
    const wp::int32 var_783 = 3;
    wp::int32 var_784;
    wp::int32 var_785;
    wp::vec_t<3, wp::float32> var_786;
    wp::float32 var_787;
    const wp::int32 var_788 = 0;
    bool var_789;
    wp::vec_t<3, wp::float32> var_790;
    wp::vec_t<3, wp::float32> var_791;
    wp::vec_t<3, wp::float32> var_792;
    wp::vec_t<3, wp::float32> var_793;
    const wp::int32 var_794 = 0;
    wp::float32 var_795;
    const wp::int32 var_796 = 3;
    wp::int32 var_797;
    const wp::int32 var_798 = 0;
    wp::int32 var_799;
    wp::vec_t<3, wp::float32> var_800;
    const wp::int32 var_801 = 3;
    wp::int32 var_802;
    wp::int32 var_803;
    wp::vec_t<3, wp::float32> var_804;
    wp::float32 var_805;
    wp::float32 var_806;
    wp::float32 var_807;
    const wp::int32 var_808 = 1;
    wp::float32 var_809;
    const wp::int32 var_810 = 3;
    wp::int32 var_811;
    const wp::int32 var_812 = 1;
    wp::int32 var_813;
    wp::vec_t<3, wp::float32> var_814;
    const wp::int32 var_815 = 3;
    wp::int32 var_816;
    wp::int32 var_817;
    wp::vec_t<3, wp::float32> var_818;
    wp::float32 var_819;
    wp::float32 var_820;
    wp::float32 var_821;
    wp::float32 var_822;
    const wp::int32 var_823 = 2;
    wp::float32 var_824;
    const wp::int32 var_825 = 3;
    wp::int32 var_826;
    const wp::int32 var_827 = 2;
    wp::int32 var_828;
    wp::vec_t<3, wp::float32> var_829;
    const wp::int32 var_830 = 3;
    wp::int32 var_831;
    wp::int32 var_832;
    wp::vec_t<3, wp::float32> var_833;
    wp::float32 var_834;
    wp::float32 var_835;
    wp::float32 var_836;
    wp::float32 var_837;
    const wp::int32 var_838 = 0;
    wp::float32 var_839;
    const wp::int32 var_840 = 1;
    wp::float32 var_841;
    wp::float32 var_842;
    wp::float32 var_843;
    const wp::int32 var_844 = 1;
    wp::float32 var_845;
    const wp::int32 var_846 = 0;
    wp::float32 var_847;
    wp::float32 var_848;
    wp::float32 var_849;
    bool var_850;
    const bool var_851 = false;
    const bool var_852 = true;
    //---------
    // forward
    // def _obb_filter(                                                                       <L 218>
    // margin = margin1 + margin2                                                             <L 232>
    var_0 = wp::add(var_margin1, var_margin2);
    // xcenter = mat23()                                                                      <L 234>
    var_1 = wp::mat_t<2, 3, wp::float32>();
    // normal = mat63()                                                                       <L 235>
    var_2 = wp::mat_t<6, 3, wp::float32>();
    // proj = wp.vec2()                                                                       <L 236>
    var_3 = wp::vec_t<2, wp::float32>();
    // radius = wp.vec2()                                                                     <L 237>
    var_4 = wp::vec_t<2, wp::float32>();
    // xcenter[0] = xmat1 @ center1 + xpos1                                                   <L 240>
    var_5 = wp::mul(var_xmat1, var_center1);
    var_6 = wp::add(var_5, var_xpos1);
    wp::assign_inplace(var_1, var_7, var_6);
    // xcenter[1] = xmat2 @ center2 + xpos2                                                   <L 241>
    var_8 = wp::mul(var_xmat2, var_center2);
    var_9 = wp::add(var_8, var_xpos2);
    wp::assign_inplace(var_1, var_10, var_9);
    // normal[0] = wp.vec3(xmat1[0, 0], xmat1[1, 0], xmat1[2, 0])                             <L 244>
    var_13 = wp::extract(var_xmat1, var_11, var_12);
    var_16 = wp::extract(var_xmat1, var_14, var_15);
    var_19 = wp::extract(var_xmat1, var_17, var_18);
    var_20 = wp::vec_t<3, wp::float32>(var_13, var_16, var_19);
    wp::assign_inplace(var_2, var_21, var_20);
    // normal[1] = wp.vec3(xmat1[0, 1], xmat1[1, 1], xmat1[2, 1])                             <L 245>
    var_24 = wp::extract(var_xmat1, var_22, var_23);
    var_27 = wp::extract(var_xmat1, var_25, var_26);
    var_30 = wp::extract(var_xmat1, var_28, var_29);
    var_31 = wp::vec_t<3, wp::float32>(var_24, var_27, var_30);
    wp::assign_inplace(var_2, var_32, var_31);
    // normal[2] = wp.vec3(xmat1[0, 2], xmat1[1, 2], xmat1[2, 2])                             <L 246>
    var_35 = wp::extract(var_xmat1, var_33, var_34);
    var_38 = wp::extract(var_xmat1, var_36, var_37);
    var_41 = wp::extract(var_xmat1, var_39, var_40);
    var_42 = wp::vec_t<3, wp::float32>(var_35, var_38, var_41);
    wp::assign_inplace(var_2, var_43, var_42);
    // normal[3] = wp.vec3(xmat2[0, 0], xmat2[1, 0], xmat2[2, 0])                             <L 247>
    var_46 = wp::extract(var_xmat2, var_44, var_45);
    var_49 = wp::extract(var_xmat2, var_47, var_48);
    var_52 = wp::extract(var_xmat2, var_50, var_51);
    var_53 = wp::vec_t<3, wp::float32>(var_46, var_49, var_52);
    wp::assign_inplace(var_2, var_54, var_53);
    // normal[4] = wp.vec3(xmat2[0, 1], xmat2[1, 1], xmat2[2, 1])                             <L 248>
    var_57 = wp::extract(var_xmat2, var_55, var_56);
    var_60 = wp::extract(var_xmat2, var_58, var_59);
    var_63 = wp::extract(var_xmat2, var_61, var_62);
    var_64 = wp::vec_t<3, wp::float32>(var_57, var_60, var_63);
    wp::assign_inplace(var_2, var_65, var_64);
    // normal[5] = wp.vec3(xmat2[0, 2], xmat2[1, 2], xmat2[2, 2])                             <L 249>
    var_68 = wp::extract(var_xmat2, var_66, var_67);
    var_71 = wp::extract(var_xmat2, var_69, var_70);
    var_74 = wp::extract(var_xmat2, var_72, var_73);
    var_75 = wp::vec_t<3, wp::float32>(var_68, var_71, var_74);
    wp::assign_inplace(var_2, var_76, var_75);
    // for j in range(2):                                                                     <L 252>
    // for k in range(3):                                                                     <L 253>
    // for i in range(2):                                                                     <L 254>
    // proj[i] = wp.dot(xcenter[i], normal[3 * j + k])                                        <L 255>
    var_80 = wp::extract(var_1, var_79);
    var_82 = wp::mul(var_81, var_77);
    var_83 = wp::add(var_82, var_78);
    var_84 = wp::extract(var_2, var_83);
    var_85 = wp::dot(var_80, var_84);
    wp::assign_inplace(var_3, var_79, var_85);
    // if i == 0:                                                                             <L 256>
    var_87 = (var_79 == var_86);
    if (var_87) {
        // size = size1                                                                       <L 257>
        var_88 = wp::copy(var_size1);
    }
    if (!var_87) {
        // size = size2                                                                       <L 259>
        var_89 = wp::copy(var_size2);
    }
    var_90 = wp::where(var_87, var_88, var_89);
    // radius[i] = (                                                                          <L 262>
    // wp.abs(size[0] * wp.dot(normal[3 * i + 0], normal[3 * j + k]))                         <L 263>
    var_92 = wp::extract(var_90, var_91);
    var_94 = wp::mul(var_93, var_79);
    var_96 = wp::add(var_94, var_95);
    var_97 = wp::extract(var_2, var_96);
    var_99 = wp::mul(var_98, var_77);
    var_100 = wp::add(var_99, var_78);
    var_101 = wp::extract(var_2, var_100);
    var_102 = wp::dot(var_97, var_101);
    var_103 = wp::mul(var_92, var_102);
    var_104 = wp::abs(var_103);
    // + wp.abs(size[1] * wp.dot(normal[3 * i + 1], normal[3 * j + k]))                       <L 264>
    var_106 = wp::extract(var_90, var_105);
    var_108 = wp::mul(var_107, var_79);
    var_110 = wp::add(var_108, var_109);
    var_111 = wp::extract(var_2, var_110);
    var_113 = wp::mul(var_112, var_77);
    var_114 = wp::add(var_113, var_78);
    var_115 = wp::extract(var_2, var_114);
    var_116 = wp::dot(var_111, var_115);
    var_117 = wp::mul(var_106, var_116);
    var_118 = wp::abs(var_117);
    var_119 = wp::add(var_104, var_118);
    // + wp.abs(size[2] * wp.dot(normal[3 * i + 2], normal[3 * j + k]))                       <L 265>
    var_121 = wp::extract(var_90, var_120);
    var_123 = wp::mul(var_122, var_79);
    var_125 = wp::add(var_123, var_124);
    var_126 = wp::extract(var_2, var_125);
    var_128 = wp::mul(var_127, var_77);
    var_129 = wp::add(var_128, var_78);
    var_130 = wp::extract(var_2, var_129);
    var_131 = wp::dot(var_126, var_130);
    var_132 = wp::mul(var_121, var_131);
    var_133 = wp::abs(var_132);
    var_134 = wp::add(var_119, var_133);
    // radius[i] = (                                                                          <L 262>
    wp::assign_inplace(var_4, var_79, var_134);
    // proj[i] = wp.dot(xcenter[i], normal[3 * j + k])                                        <L 255>
    var_136 = wp::extract(var_1, var_135);
    var_138 = wp::mul(var_137, var_77);
    var_139 = wp::add(var_138, var_78);
    var_140 = wp::extract(var_2, var_139);
    var_141 = wp::dot(var_136, var_140);
    wp::assign_inplace(var_3, var_135, var_141);
    // if i == 0:                                                                             <L 256>
    var_143 = (var_135 == var_142);
    if (var_143) {
        // size = size1                                                                       <L 257>
        var_144 = wp::copy(var_size1);
    }
    var_145 = wp::where(var_143, var_144, var_90);
    if (!var_143) {
        // size = size2                                                                       <L 259>
        var_146 = wp::copy(var_size2);
    }
    var_147 = wp::where(var_143, var_145, var_146);
    // radius[i] = (                                                                          <L 262>
    // wp.abs(size[0] * wp.dot(normal[3 * i + 0], normal[3 * j + k]))                         <L 263>
    var_149 = wp::extract(var_147, var_148);
    var_151 = wp::mul(var_150, var_135);
    var_153 = wp::add(var_151, var_152);
    var_154 = wp::extract(var_2, var_153);
    var_156 = wp::mul(var_155, var_77);
    var_157 = wp::add(var_156, var_78);
    var_158 = wp::extract(var_2, var_157);
    var_159 = wp::dot(var_154, var_158);
    var_160 = wp::mul(var_149, var_159);
    var_161 = wp::abs(var_160);
    // + wp.abs(size[1] * wp.dot(normal[3 * i + 1], normal[3 * j + k]))                       <L 264>
    var_163 = wp::extract(var_147, var_162);
    var_165 = wp::mul(var_164, var_135);
    var_167 = wp::add(var_165, var_166);
    var_168 = wp::extract(var_2, var_167);
    var_170 = wp::mul(var_169, var_77);
    var_171 = wp::add(var_170, var_78);
    var_172 = wp::extract(var_2, var_171);
    var_173 = wp::dot(var_168, var_172);
    var_174 = wp::mul(var_163, var_173);
    var_175 = wp::abs(var_174);
    var_176 = wp::add(var_161, var_175);
    // + wp.abs(size[2] * wp.dot(normal[3 * i + 2], normal[3 * j + k]))                       <L 265>
    var_178 = wp::extract(var_147, var_177);
    var_180 = wp::mul(var_179, var_135);
    var_182 = wp::add(var_180, var_181);
    var_183 = wp::extract(var_2, var_182);
    var_185 = wp::mul(var_184, var_77);
    var_186 = wp::add(var_185, var_78);
    var_187 = wp::extract(var_2, var_186);
    var_188 = wp::dot(var_183, var_187);
    var_189 = wp::mul(var_178, var_188);
    var_190 = wp::abs(var_189);
    var_191 = wp::add(var_176, var_190);
    // radius[i] = (                                                                          <L 262>
    wp::assign_inplace(var_4, var_135, var_191);
    // if radius[0] + radius[1] + margin < wp.abs(proj[1] - proj[0]):                         <L 268>
    var_193 = wp::extract(var_4, var_192);
    var_195 = wp::extract(var_4, var_194);
    var_196 = wp::add(var_193, var_195);
    var_197 = wp::add(var_196, var_0);
    var_199 = wp::extract(var_3, var_198);
    var_201 = wp::extract(var_3, var_200);
    var_202 = wp::sub(var_199, var_201);
    var_203 = wp::abs(var_202);
    var_204 = (var_197 < var_203);
    if (var_204) {
        // return False                                                                       <L 269>
        return var_205;
    }
    // for i in range(2):                                                                     <L 254>
    // proj[i] = wp.dot(xcenter[i], normal[3 * j + k])                                        <L 255>
    var_208 = wp::extract(var_1, var_207);
    var_210 = wp::mul(var_209, var_77);
    var_211 = wp::add(var_210, var_206);
    var_212 = wp::extract(var_2, var_211);
    var_213 = wp::dot(var_208, var_212);
    wp::assign_inplace(var_3, var_207, var_213);
    // if i == 0:                                                                             <L 256>
    var_215 = (var_207 == var_214);
    if (var_215) {
        // size = size1                                                                       <L 257>
        var_216 = wp::copy(var_size1);
    }
    var_217 = wp::where(var_215, var_216, var_147);
    if (!var_215) {
        // size = size2                                                                       <L 259>
        var_218 = wp::copy(var_size2);
    }
    var_219 = wp::where(var_215, var_217, var_218);
    // radius[i] = (                                                                          <L 262>
    // wp.abs(size[0] * wp.dot(normal[3 * i + 0], normal[3 * j + k]))                         <L 263>
    var_221 = wp::extract(var_219, var_220);
    var_223 = wp::mul(var_222, var_207);
    var_225 = wp::add(var_223, var_224);
    var_226 = wp::extract(var_2, var_225);
    var_228 = wp::mul(var_227, var_77);
    var_229 = wp::add(var_228, var_206);
    var_230 = wp::extract(var_2, var_229);
    var_231 = wp::dot(var_226, var_230);
    var_232 = wp::mul(var_221, var_231);
    var_233 = wp::abs(var_232);
    // + wp.abs(size[1] * wp.dot(normal[3 * i + 1], normal[3 * j + k]))                       <L 264>
    var_235 = wp::extract(var_219, var_234);
    var_237 = wp::mul(var_236, var_207);
    var_239 = wp::add(var_237, var_238);
    var_240 = wp::extract(var_2, var_239);
    var_242 = wp::mul(var_241, var_77);
    var_243 = wp::add(var_242, var_206);
    var_244 = wp::extract(var_2, var_243);
    var_245 = wp::dot(var_240, var_244);
    var_246 = wp::mul(var_235, var_245);
    var_247 = wp::abs(var_246);
    var_248 = wp::add(var_233, var_247);
    // + wp.abs(size[2] * wp.dot(normal[3 * i + 2], normal[3 * j + k]))                       <L 265>
    var_250 = wp::extract(var_219, var_249);
    var_252 = wp::mul(var_251, var_207);
    var_254 = wp::add(var_252, var_253);
    var_255 = wp::extract(var_2, var_254);
    var_257 = wp::mul(var_256, var_77);
    var_258 = wp::add(var_257, var_206);
    var_259 = wp::extract(var_2, var_258);
    var_260 = wp::dot(var_255, var_259);
    var_261 = wp::mul(var_250, var_260);
    var_262 = wp::abs(var_261);
    var_263 = wp::add(var_248, var_262);
    // radius[i] = (                                                                          <L 262>
    wp::assign_inplace(var_4, var_207, var_263);
    // proj[i] = wp.dot(xcenter[i], normal[3 * j + k])                                        <L 255>
    var_265 = wp::extract(var_1, var_264);
    var_267 = wp::mul(var_266, var_77);
    var_268 = wp::add(var_267, var_206);
    var_269 = wp::extract(var_2, var_268);
    var_270 = wp::dot(var_265, var_269);
    wp::assign_inplace(var_3, var_264, var_270);
    // if i == 0:                                                                             <L 256>
    var_272 = (var_264 == var_271);
    if (var_272) {
        // size = size1                                                                       <L 257>
        var_273 = wp::copy(var_size1);
    }
    var_274 = wp::where(var_272, var_273, var_219);
    if (!var_272) {
        // size = size2                                                                       <L 259>
        var_275 = wp::copy(var_size2);
    }
    var_276 = wp::where(var_272, var_274, var_275);
    // radius[i] = (                                                                          <L 262>
    // wp.abs(size[0] * wp.dot(normal[3 * i + 0], normal[3 * j + k]))                         <L 263>
    var_278 = wp::extract(var_276, var_277);
    var_280 = wp::mul(var_279, var_264);
    var_282 = wp::add(var_280, var_281);
    var_283 = wp::extract(var_2, var_282);
    var_285 = wp::mul(var_284, var_77);
    var_286 = wp::add(var_285, var_206);
    var_287 = wp::extract(var_2, var_286);
    var_288 = wp::dot(var_283, var_287);
    var_289 = wp::mul(var_278, var_288);
    var_290 = wp::abs(var_289);
    // + wp.abs(size[1] * wp.dot(normal[3 * i + 1], normal[3 * j + k]))                       <L 264>
    var_292 = wp::extract(var_276, var_291);
    var_294 = wp::mul(var_293, var_264);
    var_296 = wp::add(var_294, var_295);
    var_297 = wp::extract(var_2, var_296);
    var_299 = wp::mul(var_298, var_77);
    var_300 = wp::add(var_299, var_206);
    var_301 = wp::extract(var_2, var_300);
    var_302 = wp::dot(var_297, var_301);
    var_303 = wp::mul(var_292, var_302);
    var_304 = wp::abs(var_303);
    var_305 = wp::add(var_290, var_304);
    // + wp.abs(size[2] * wp.dot(normal[3 * i + 2], normal[3 * j + k]))                       <L 265>
    var_307 = wp::extract(var_276, var_306);
    var_309 = wp::mul(var_308, var_264);
    var_311 = wp::add(var_309, var_310);
    var_312 = wp::extract(var_2, var_311);
    var_314 = wp::mul(var_313, var_77);
    var_315 = wp::add(var_314, var_206);
    var_316 = wp::extract(var_2, var_315);
    var_317 = wp::dot(var_312, var_316);
    var_318 = wp::mul(var_307, var_317);
    var_319 = wp::abs(var_318);
    var_320 = wp::add(var_305, var_319);
    // radius[i] = (                                                                          <L 262>
    wp::assign_inplace(var_4, var_264, var_320);
    // if radius[0] + radius[1] + margin < wp.abs(proj[1] - proj[0]):                         <L 268>
    var_322 = wp::extract(var_4, var_321);
    var_324 = wp::extract(var_4, var_323);
    var_325 = wp::add(var_322, var_324);
    var_326 = wp::add(var_325, var_0);
    var_328 = wp::extract(var_3, var_327);
    var_330 = wp::extract(var_3, var_329);
    var_331 = wp::sub(var_328, var_330);
    var_332 = wp::abs(var_331);
    var_333 = (var_326 < var_332);
    if (var_333) {
        // return False                                                                       <L 269>
        return var_334;
    }
    // for i in range(2):                                                                     <L 254>
    // proj[i] = wp.dot(xcenter[i], normal[3 * j + k])                                        <L 255>
    var_337 = wp::extract(var_1, var_336);
    var_339 = wp::mul(var_338, var_77);
    var_340 = wp::add(var_339, var_335);
    var_341 = wp::extract(var_2, var_340);
    var_342 = wp::dot(var_337, var_341);
    wp::assign_inplace(var_3, var_336, var_342);
    // if i == 0:                                                                             <L 256>
    var_344 = (var_336 == var_343);
    if (var_344) {
        // size = size1                                                                       <L 257>
        var_345 = wp::copy(var_size1);
    }
    var_346 = wp::where(var_344, var_345, var_276);
    if (!var_344) {
        // size = size2                                                                       <L 259>
        var_347 = wp::copy(var_size2);
    }
    var_348 = wp::where(var_344, var_346, var_347);
    // radius[i] = (                                                                          <L 262>
    // wp.abs(size[0] * wp.dot(normal[3 * i + 0], normal[3 * j + k]))                         <L 263>
    var_350 = wp::extract(var_348, var_349);
    var_352 = wp::mul(var_351, var_336);
    var_354 = wp::add(var_352, var_353);
    var_355 = wp::extract(var_2, var_354);
    var_357 = wp::mul(var_356, var_77);
    var_358 = wp::add(var_357, var_335);
    var_359 = wp::extract(var_2, var_358);
    var_360 = wp::dot(var_355, var_359);
    var_361 = wp::mul(var_350, var_360);
    var_362 = wp::abs(var_361);
    // + wp.abs(size[1] * wp.dot(normal[3 * i + 1], normal[3 * j + k]))                       <L 264>
    var_364 = wp::extract(var_348, var_363);
    var_366 = wp::mul(var_365, var_336);
    var_368 = wp::add(var_366, var_367);
    var_369 = wp::extract(var_2, var_368);
    var_371 = wp::mul(var_370, var_77);
    var_372 = wp::add(var_371, var_335);
    var_373 = wp::extract(var_2, var_372);
    var_374 = wp::dot(var_369, var_373);
    var_375 = wp::mul(var_364, var_374);
    var_376 = wp::abs(var_375);
    var_377 = wp::add(var_362, var_376);
    // + wp.abs(size[2] * wp.dot(normal[3 * i + 2], normal[3 * j + k]))                       <L 265>
    var_379 = wp::extract(var_348, var_378);
    var_381 = wp::mul(var_380, var_336);
    var_383 = wp::add(var_381, var_382);
    var_384 = wp::extract(var_2, var_383);
    var_386 = wp::mul(var_385, var_77);
    var_387 = wp::add(var_386, var_335);
    var_388 = wp::extract(var_2, var_387);
    var_389 = wp::dot(var_384, var_388);
    var_390 = wp::mul(var_379, var_389);
    var_391 = wp::abs(var_390);
    var_392 = wp::add(var_377, var_391);
    // radius[i] = (                                                                          <L 262>
    wp::assign_inplace(var_4, var_336, var_392);
    // proj[i] = wp.dot(xcenter[i], normal[3 * j + k])                                        <L 255>
    var_394 = wp::extract(var_1, var_393);
    var_396 = wp::mul(var_395, var_77);
    var_397 = wp::add(var_396, var_335);
    var_398 = wp::extract(var_2, var_397);
    var_399 = wp::dot(var_394, var_398);
    wp::assign_inplace(var_3, var_393, var_399);
    // if i == 0:                                                                             <L 256>
    var_401 = (var_393 == var_400);
    if (var_401) {
        // size = size1                                                                       <L 257>
        var_402 = wp::copy(var_size1);
    }
    var_403 = wp::where(var_401, var_402, var_348);
    if (!var_401) {
        // size = size2                                                                       <L 259>
        var_404 = wp::copy(var_size2);
    }
    var_405 = wp::where(var_401, var_403, var_404);
    // radius[i] = (                                                                          <L 262>
    // wp.abs(size[0] * wp.dot(normal[3 * i + 0], normal[3 * j + k]))                         <L 263>
    var_407 = wp::extract(var_405, var_406);
    var_409 = wp::mul(var_408, var_393);
    var_411 = wp::add(var_409, var_410);
    var_412 = wp::extract(var_2, var_411);
    var_414 = wp::mul(var_413, var_77);
    var_415 = wp::add(var_414, var_335);
    var_416 = wp::extract(var_2, var_415);
    var_417 = wp::dot(var_412, var_416);
    var_418 = wp::mul(var_407, var_417);
    var_419 = wp::abs(var_418);
    // + wp.abs(size[1] * wp.dot(normal[3 * i + 1], normal[3 * j + k]))                       <L 264>
    var_421 = wp::extract(var_405, var_420);
    var_423 = wp::mul(var_422, var_393);
    var_425 = wp::add(var_423, var_424);
    var_426 = wp::extract(var_2, var_425);
    var_428 = wp::mul(var_427, var_77);
    var_429 = wp::add(var_428, var_335);
    var_430 = wp::extract(var_2, var_429);
    var_431 = wp::dot(var_426, var_430);
    var_432 = wp::mul(var_421, var_431);
    var_433 = wp::abs(var_432);
    var_434 = wp::add(var_419, var_433);
    // + wp.abs(size[2] * wp.dot(normal[3 * i + 2], normal[3 * j + k]))                       <L 265>
    var_436 = wp::extract(var_405, var_435);
    var_438 = wp::mul(var_437, var_393);
    var_440 = wp::add(var_438, var_439);
    var_441 = wp::extract(var_2, var_440);
    var_443 = wp::mul(var_442, var_77);
    var_444 = wp::add(var_443, var_335);
    var_445 = wp::extract(var_2, var_444);
    var_446 = wp::dot(var_441, var_445);
    var_447 = wp::mul(var_436, var_446);
    var_448 = wp::abs(var_447);
    var_449 = wp::add(var_434, var_448);
    // radius[i] = (                                                                          <L 262>
    wp::assign_inplace(var_4, var_393, var_449);
    // if radius[0] + radius[1] + margin < wp.abs(proj[1] - proj[0]):                         <L 268>
    var_451 = wp::extract(var_4, var_450);
    var_453 = wp::extract(var_4, var_452);
    var_454 = wp::add(var_451, var_453);
    var_455 = wp::add(var_454, var_0);
    var_457 = wp::extract(var_3, var_456);
    var_459 = wp::extract(var_3, var_458);
    var_460 = wp::sub(var_457, var_459);
    var_461 = wp::abs(var_460);
    var_462 = (var_455 < var_461);
    if (var_462) {
        // return False                                                                       <L 269>
        return var_463;
    }
    // for k in range(3):                                                                     <L 253>
    // for i in range(2):                                                                     <L 254>
    // proj[i] = wp.dot(xcenter[i], normal[3 * j + k])                                        <L 255>
    var_467 = wp::extract(var_1, var_466);
    var_469 = wp::mul(var_468, var_464);
    var_470 = wp::add(var_469, var_465);
    var_471 = wp::extract(var_2, var_470);
    var_472 = wp::dot(var_467, var_471);
    wp::assign_inplace(var_3, var_466, var_472);
    // if i == 0:                                                                             <L 256>
    var_474 = (var_466 == var_473);
    if (var_474) {
        // size = size1                                                                       <L 257>
        var_475 = wp::copy(var_size1);
    }
    var_476 = wp::where(var_474, var_475, var_405);
    if (!var_474) {
        // size = size2                                                                       <L 259>
        var_477 = wp::copy(var_size2);
    }
    var_478 = wp::where(var_474, var_476, var_477);
    // radius[i] = (                                                                          <L 262>
    // wp.abs(size[0] * wp.dot(normal[3 * i + 0], normal[3 * j + k]))                         <L 263>
    var_480 = wp::extract(var_478, var_479);
    var_482 = wp::mul(var_481, var_466);
    var_484 = wp::add(var_482, var_483);
    var_485 = wp::extract(var_2, var_484);
    var_487 = wp::mul(var_486, var_464);
    var_488 = wp::add(var_487, var_465);
    var_489 = wp::extract(var_2, var_488);
    var_490 = wp::dot(var_485, var_489);
    var_491 = wp::mul(var_480, var_490);
    var_492 = wp::abs(var_491);
    // + wp.abs(size[1] * wp.dot(normal[3 * i + 1], normal[3 * j + k]))                       <L 264>
    var_494 = wp::extract(var_478, var_493);
    var_496 = wp::mul(var_495, var_466);
    var_498 = wp::add(var_496, var_497);
    var_499 = wp::extract(var_2, var_498);
    var_501 = wp::mul(var_500, var_464);
    var_502 = wp::add(var_501, var_465);
    var_503 = wp::extract(var_2, var_502);
    var_504 = wp::dot(var_499, var_503);
    var_505 = wp::mul(var_494, var_504);
    var_506 = wp::abs(var_505);
    var_507 = wp::add(var_492, var_506);
    // + wp.abs(size[2] * wp.dot(normal[3 * i + 2], normal[3 * j + k]))                       <L 265>
    var_509 = wp::extract(var_478, var_508);
    var_511 = wp::mul(var_510, var_466);
    var_513 = wp::add(var_511, var_512);
    var_514 = wp::extract(var_2, var_513);
    var_516 = wp::mul(var_515, var_464);
    var_517 = wp::add(var_516, var_465);
    var_518 = wp::extract(var_2, var_517);
    var_519 = wp::dot(var_514, var_518);
    var_520 = wp::mul(var_509, var_519);
    var_521 = wp::abs(var_520);
    var_522 = wp::add(var_507, var_521);
    // radius[i] = (                                                                          <L 262>
    wp::assign_inplace(var_4, var_466, var_522);
    // proj[i] = wp.dot(xcenter[i], normal[3 * j + k])                                        <L 255>
    var_524 = wp::extract(var_1, var_523);
    var_526 = wp::mul(var_525, var_464);
    var_527 = wp::add(var_526, var_465);
    var_528 = wp::extract(var_2, var_527);
    var_529 = wp::dot(var_524, var_528);
    wp::assign_inplace(var_3, var_523, var_529);
    // if i == 0:                                                                             <L 256>
    var_531 = (var_523 == var_530);
    if (var_531) {
        // size = size1                                                                       <L 257>
        var_532 = wp::copy(var_size1);
    }
    var_533 = wp::where(var_531, var_532, var_478);
    if (!var_531) {
        // size = size2                                                                       <L 259>
        var_534 = wp::copy(var_size2);
    }
    var_535 = wp::where(var_531, var_533, var_534);
    // radius[i] = (                                                                          <L 262>
    // wp.abs(size[0] * wp.dot(normal[3 * i + 0], normal[3 * j + k]))                         <L 263>
    var_537 = wp::extract(var_535, var_536);
    var_539 = wp::mul(var_538, var_523);
    var_541 = wp::add(var_539, var_540);
    var_542 = wp::extract(var_2, var_541);
    var_544 = wp::mul(var_543, var_464);
    var_545 = wp::add(var_544, var_465);
    var_546 = wp::extract(var_2, var_545);
    var_547 = wp::dot(var_542, var_546);
    var_548 = wp::mul(var_537, var_547);
    var_549 = wp::abs(var_548);
    // + wp.abs(size[1] * wp.dot(normal[3 * i + 1], normal[3 * j + k]))                       <L 264>
    var_551 = wp::extract(var_535, var_550);
    var_553 = wp::mul(var_552, var_523);
    var_555 = wp::add(var_553, var_554);
    var_556 = wp::extract(var_2, var_555);
    var_558 = wp::mul(var_557, var_464);
    var_559 = wp::add(var_558, var_465);
    var_560 = wp::extract(var_2, var_559);
    var_561 = wp::dot(var_556, var_560);
    var_562 = wp::mul(var_551, var_561);
    var_563 = wp::abs(var_562);
    var_564 = wp::add(var_549, var_563);
    // + wp.abs(size[2] * wp.dot(normal[3 * i + 2], normal[3 * j + k]))                       <L 265>
    var_566 = wp::extract(var_535, var_565);
    var_568 = wp::mul(var_567, var_523);
    var_570 = wp::add(var_568, var_569);
    var_571 = wp::extract(var_2, var_570);
    var_573 = wp::mul(var_572, var_464);
    var_574 = wp::add(var_573, var_465);
    var_575 = wp::extract(var_2, var_574);
    var_576 = wp::dot(var_571, var_575);
    var_577 = wp::mul(var_566, var_576);
    var_578 = wp::abs(var_577);
    var_579 = wp::add(var_564, var_578);
    // radius[i] = (                                                                          <L 262>
    wp::assign_inplace(var_4, var_523, var_579);
    // if radius[0] + radius[1] + margin < wp.abs(proj[1] - proj[0]):                         <L 268>
    var_581 = wp::extract(var_4, var_580);
    var_583 = wp::extract(var_4, var_582);
    var_584 = wp::add(var_581, var_583);
    var_585 = wp::add(var_584, var_0);
    var_587 = wp::extract(var_3, var_586);
    var_589 = wp::extract(var_3, var_588);
    var_590 = wp::sub(var_587, var_589);
    var_591 = wp::abs(var_590);
    var_592 = (var_585 < var_591);
    if (var_592) {
        // return False                                                                       <L 269>
        return var_593;
    }
    // for i in range(2):                                                                     <L 254>
    // proj[i] = wp.dot(xcenter[i], normal[3 * j + k])                                        <L 255>
    var_596 = wp::extract(var_1, var_595);
    var_598 = wp::mul(var_597, var_464);
    var_599 = wp::add(var_598, var_594);
    var_600 = wp::extract(var_2, var_599);
    var_601 = wp::dot(var_596, var_600);
    wp::assign_inplace(var_3, var_595, var_601);
    // if i == 0:                                                                             <L 256>
    var_603 = (var_595 == var_602);
    if (var_603) {
        // size = size1                                                                       <L 257>
        var_604 = wp::copy(var_size1);
    }
    var_605 = wp::where(var_603, var_604, var_535);
    if (!var_603) {
        // size = size2                                                                       <L 259>
        var_606 = wp::copy(var_size2);
    }
    var_607 = wp::where(var_603, var_605, var_606);
    // radius[i] = (                                                                          <L 262>
    // wp.abs(size[0] * wp.dot(normal[3 * i + 0], normal[3 * j + k]))                         <L 263>
    var_609 = wp::extract(var_607, var_608);
    var_611 = wp::mul(var_610, var_595);
    var_613 = wp::add(var_611, var_612);
    var_614 = wp::extract(var_2, var_613);
    var_616 = wp::mul(var_615, var_464);
    var_617 = wp::add(var_616, var_594);
    var_618 = wp::extract(var_2, var_617);
    var_619 = wp::dot(var_614, var_618);
    var_620 = wp::mul(var_609, var_619);
    var_621 = wp::abs(var_620);
    // + wp.abs(size[1] * wp.dot(normal[3 * i + 1], normal[3 * j + k]))                       <L 264>
    var_623 = wp::extract(var_607, var_622);
    var_625 = wp::mul(var_624, var_595);
    var_627 = wp::add(var_625, var_626);
    var_628 = wp::extract(var_2, var_627);
    var_630 = wp::mul(var_629, var_464);
    var_631 = wp::add(var_630, var_594);
    var_632 = wp::extract(var_2, var_631);
    var_633 = wp::dot(var_628, var_632);
    var_634 = wp::mul(var_623, var_633);
    var_635 = wp::abs(var_634);
    var_636 = wp::add(var_621, var_635);
    // + wp.abs(size[2] * wp.dot(normal[3 * i + 2], normal[3 * j + k]))                       <L 265>
    var_638 = wp::extract(var_607, var_637);
    var_640 = wp::mul(var_639, var_595);
    var_642 = wp::add(var_640, var_641);
    var_643 = wp::extract(var_2, var_642);
    var_645 = wp::mul(var_644, var_464);
    var_646 = wp::add(var_645, var_594);
    var_647 = wp::extract(var_2, var_646);
    var_648 = wp::dot(var_643, var_647);
    var_649 = wp::mul(var_638, var_648);
    var_650 = wp::abs(var_649);
    var_651 = wp::add(var_636, var_650);
    // radius[i] = (                                                                          <L 262>
    wp::assign_inplace(var_4, var_595, var_651);
    // proj[i] = wp.dot(xcenter[i], normal[3 * j + k])                                        <L 255>
    var_653 = wp::extract(var_1, var_652);
    var_655 = wp::mul(var_654, var_464);
    var_656 = wp::add(var_655, var_594);
    var_657 = wp::extract(var_2, var_656);
    var_658 = wp::dot(var_653, var_657);
    wp::assign_inplace(var_3, var_652, var_658);
    // if i == 0:                                                                             <L 256>
    var_660 = (var_652 == var_659);
    if (var_660) {
        // size = size1                                                                       <L 257>
        var_661 = wp::copy(var_size1);
    }
    var_662 = wp::where(var_660, var_661, var_607);
    if (!var_660) {
        // size = size2                                                                       <L 259>
        var_663 = wp::copy(var_size2);
    }
    var_664 = wp::where(var_660, var_662, var_663);
    // radius[i] = (                                                                          <L 262>
    // wp.abs(size[0] * wp.dot(normal[3 * i + 0], normal[3 * j + k]))                         <L 263>
    var_666 = wp::extract(var_664, var_665);
    var_668 = wp::mul(var_667, var_652);
    var_670 = wp::add(var_668, var_669);
    var_671 = wp::extract(var_2, var_670);
    var_673 = wp::mul(var_672, var_464);
    var_674 = wp::add(var_673, var_594);
    var_675 = wp::extract(var_2, var_674);
    var_676 = wp::dot(var_671, var_675);
    var_677 = wp::mul(var_666, var_676);
    var_678 = wp::abs(var_677);
    // + wp.abs(size[1] * wp.dot(normal[3 * i + 1], normal[3 * j + k]))                       <L 264>
    var_680 = wp::extract(var_664, var_679);
    var_682 = wp::mul(var_681, var_652);
    var_684 = wp::add(var_682, var_683);
    var_685 = wp::extract(var_2, var_684);
    var_687 = wp::mul(var_686, var_464);
    var_688 = wp::add(var_687, var_594);
    var_689 = wp::extract(var_2, var_688);
    var_690 = wp::dot(var_685, var_689);
    var_691 = wp::mul(var_680, var_690);
    var_692 = wp::abs(var_691);
    var_693 = wp::add(var_678, var_692);
    // + wp.abs(size[2] * wp.dot(normal[3 * i + 2], normal[3 * j + k]))                       <L 265>
    var_695 = wp::extract(var_664, var_694);
    var_697 = wp::mul(var_696, var_652);
    var_699 = wp::add(var_697, var_698);
    var_700 = wp::extract(var_2, var_699);
    var_702 = wp::mul(var_701, var_464);
    var_703 = wp::add(var_702, var_594);
    var_704 = wp::extract(var_2, var_703);
    var_705 = wp::dot(var_700, var_704);
    var_706 = wp::mul(var_695, var_705);
    var_707 = wp::abs(var_706);
    var_708 = wp::add(var_693, var_707);
    // radius[i] = (                                                                          <L 262>
    wp::assign_inplace(var_4, var_652, var_708);
    // if radius[0] + radius[1] + margin < wp.abs(proj[1] - proj[0]):                         <L 268>
    var_710 = wp::extract(var_4, var_709);
    var_712 = wp::extract(var_4, var_711);
    var_713 = wp::add(var_710, var_712);
    var_714 = wp::add(var_713, var_0);
    var_716 = wp::extract(var_3, var_715);
    var_718 = wp::extract(var_3, var_717);
    var_719 = wp::sub(var_716, var_718);
    var_720 = wp::abs(var_719);
    var_721 = (var_714 < var_720);
    if (var_721) {
        // return False                                                                       <L 269>
        return var_722;
    }
    // for i in range(2):                                                                     <L 254>
    // proj[i] = wp.dot(xcenter[i], normal[3 * j + k])                                        <L 255>
    var_725 = wp::extract(var_1, var_724);
    var_727 = wp::mul(var_726, var_464);
    var_728 = wp::add(var_727, var_723);
    var_729 = wp::extract(var_2, var_728);
    var_730 = wp::dot(var_725, var_729);
    wp::assign_inplace(var_3, var_724, var_730);
    // if i == 0:                                                                             <L 256>
    var_732 = (var_724 == var_731);
    if (var_732) {
        // size = size1                                                                       <L 257>
        var_733 = wp::copy(var_size1);
    }
    var_734 = wp::where(var_732, var_733, var_664);
    if (!var_732) {
        // size = size2                                                                       <L 259>
        var_735 = wp::copy(var_size2);
    }
    var_736 = wp::where(var_732, var_734, var_735);
    // radius[i] = (                                                                          <L 262>
    // wp.abs(size[0] * wp.dot(normal[3 * i + 0], normal[3 * j + k]))                         <L 263>
    var_738 = wp::extract(var_736, var_737);
    var_740 = wp::mul(var_739, var_724);
    var_742 = wp::add(var_740, var_741);
    var_743 = wp::extract(var_2, var_742);
    var_745 = wp::mul(var_744, var_464);
    var_746 = wp::add(var_745, var_723);
    var_747 = wp::extract(var_2, var_746);
    var_748 = wp::dot(var_743, var_747);
    var_749 = wp::mul(var_738, var_748);
    var_750 = wp::abs(var_749);
    // + wp.abs(size[1] * wp.dot(normal[3 * i + 1], normal[3 * j + k]))                       <L 264>
    var_752 = wp::extract(var_736, var_751);
    var_754 = wp::mul(var_753, var_724);
    var_756 = wp::add(var_754, var_755);
    var_757 = wp::extract(var_2, var_756);
    var_759 = wp::mul(var_758, var_464);
    var_760 = wp::add(var_759, var_723);
    var_761 = wp::extract(var_2, var_760);
    var_762 = wp::dot(var_757, var_761);
    var_763 = wp::mul(var_752, var_762);
    var_764 = wp::abs(var_763);
    var_765 = wp::add(var_750, var_764);
    // + wp.abs(size[2] * wp.dot(normal[3 * i + 2], normal[3 * j + k]))                       <L 265>
    var_767 = wp::extract(var_736, var_766);
    var_769 = wp::mul(var_768, var_724);
    var_771 = wp::add(var_769, var_770);
    var_772 = wp::extract(var_2, var_771);
    var_774 = wp::mul(var_773, var_464);
    var_775 = wp::add(var_774, var_723);
    var_776 = wp::extract(var_2, var_775);
    var_777 = wp::dot(var_772, var_776);
    var_778 = wp::mul(var_767, var_777);
    var_779 = wp::abs(var_778);
    var_780 = wp::add(var_765, var_779);
    // radius[i] = (                                                                          <L 262>
    wp::assign_inplace(var_4, var_724, var_780);
    // proj[i] = wp.dot(xcenter[i], normal[3 * j + k])                                        <L 255>
    var_782 = wp::extract(var_1, var_781);
    var_784 = wp::mul(var_783, var_464);
    var_785 = wp::add(var_784, var_723);
    var_786 = wp::extract(var_2, var_785);
    var_787 = wp::dot(var_782, var_786);
    wp::assign_inplace(var_3, var_781, var_787);
    // if i == 0:                                                                             <L 256>
    var_789 = (var_781 == var_788);
    if (var_789) {
        // size = size1                                                                       <L 257>
        var_790 = wp::copy(var_size1);
    }
    var_791 = wp::where(var_789, var_790, var_736);
    if (!var_789) {
        // size = size2                                                                       <L 259>
        var_792 = wp::copy(var_size2);
    }
    var_793 = wp::where(var_789, var_791, var_792);
    // radius[i] = (                                                                          <L 262>
    // wp.abs(size[0] * wp.dot(normal[3 * i + 0], normal[3 * j + k]))                         <L 263>
    var_795 = wp::extract(var_793, var_794);
    var_797 = wp::mul(var_796, var_781);
    var_799 = wp::add(var_797, var_798);
    var_800 = wp::extract(var_2, var_799);
    var_802 = wp::mul(var_801, var_464);
    var_803 = wp::add(var_802, var_723);
    var_804 = wp::extract(var_2, var_803);
    var_805 = wp::dot(var_800, var_804);
    var_806 = wp::mul(var_795, var_805);
    var_807 = wp::abs(var_806);
    // + wp.abs(size[1] * wp.dot(normal[3 * i + 1], normal[3 * j + k]))                       <L 264>
    var_809 = wp::extract(var_793, var_808);
    var_811 = wp::mul(var_810, var_781);
    var_813 = wp::add(var_811, var_812);
    var_814 = wp::extract(var_2, var_813);
    var_816 = wp::mul(var_815, var_464);
    var_817 = wp::add(var_816, var_723);
    var_818 = wp::extract(var_2, var_817);
    var_819 = wp::dot(var_814, var_818);
    var_820 = wp::mul(var_809, var_819);
    var_821 = wp::abs(var_820);
    var_822 = wp::add(var_807, var_821);
    // + wp.abs(size[2] * wp.dot(normal[3 * i + 2], normal[3 * j + k]))                       <L 265>
    var_824 = wp::extract(var_793, var_823);
    var_826 = wp::mul(var_825, var_781);
    var_828 = wp::add(var_826, var_827);
    var_829 = wp::extract(var_2, var_828);
    var_831 = wp::mul(var_830, var_464);
    var_832 = wp::add(var_831, var_723);
    var_833 = wp::extract(var_2, var_832);
    var_834 = wp::dot(var_829, var_833);
    var_835 = wp::mul(var_824, var_834);
    var_836 = wp::abs(var_835);
    var_837 = wp::add(var_822, var_836);
    // radius[i] = (                                                                          <L 262>
    wp::assign_inplace(var_4, var_781, var_837);
    // if radius[0] + radius[1] + margin < wp.abs(proj[1] - proj[0]):                         <L 268>
    var_839 = wp::extract(var_4, var_838);
    var_841 = wp::extract(var_4, var_840);
    var_842 = wp::add(var_839, var_841);
    var_843 = wp::add(var_842, var_0);
    var_845 = wp::extract(var_3, var_844);
    var_847 = wp::extract(var_3, var_846);
    var_848 = wp::sub(var_845, var_847);
    var_849 = wp::abs(var_848);
    var_850 = (var_843 < var_849);
    if (var_850) {
        // return False                                                                       <L 269>
        return var_851;
    }
    // return True                                                                            <L 271>
    return var_852;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_driver.py:275
static CUDA_CALLABLE bool _broadphase_filter__locals__func_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_aabb,
    wp::array_t<wp::float32> var_geom_rbound,
    wp::array_t<wp::float32> var_geom_margin,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::int32 var_geom1,
    wp::int32 var_geom2,
    wp::int32 var_worldid)
{
    //---------
    // primal vars
    const bool var_0 = false;
    const wp::int32 var_1 = 0;
    const wp::int32 var_2 = 0;
    wp::vec_t<3, wp::float32>* var_3;
    const wp::int32 var_4 = 0;
    wp::vec_t<3, wp::float32>* var_5;
    const wp::int32 var_6 = 1;
    wp::vec_t<3, wp::float32>* var_7;
    const wp::int32 var_8 = 1;
    wp::vec_t<3, wp::float32>* var_9;
    const bool var_10 = false;
    const wp::int32 var_11 = 0;
    wp::float32* var_12;
    wp::float32* var_13;
    const bool var_14 = false;
    const wp::int32 var_15 = 0;
    wp::float32* var_16;
    wp::float32* var_17;
    wp::vec_t<3, wp::float32>* var_18;
    wp::vec_t<3, wp::float32>* var_19;
    wp::mat_t<3, 3, wp::float32>* var_20;
    wp::mat_t<3, 3, wp::float32>* var_21;
    const wp::float32 var_22 = 0.0;
    bool var_23;
    wp::float32 var_24;
    const wp::float32 var_25 = 0.0;
    bool var_26;
    wp::float32 var_27;
    bool var_28;
    const wp::int32 var_29 = 1;
    bool var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    wp::vec_t<3, wp::float32> var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::mat_t<3, 3, wp::float32> var_37;
    wp::mat_t<3, 3, wp::float32> var_38;
    const wp::int32 var_39 = 2;
    bool var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::vec_t<3, wp::float32> var_45;
    wp::vec_t<3, wp::float32> var_46;
    bool var_47;
    const bool var_48 = false;
    const wp::int32 var_49 = 0;
    const wp::int32 var_50 = 8;
    bool var_51;
    wp::vec_t<3, wp::float32> var_52;
    wp::vec_t<3, wp::float32> var_53;
    wp::vec_t<3, wp::float32> var_54;
    wp::vec_t<3, wp::float32> var_55;
    wp::float32 var_56;
    wp::float32 var_57;
    wp::vec_t<3, wp::float32> var_58;
    wp::vec_t<3, wp::float32> var_59;
    wp::mat_t<3, 3, wp::float32> var_60;
    wp::mat_t<3, 3, wp::float32> var_61;
    bool var_62;
    const bool var_63 = false;
    const bool var_64 = true;
    //---------
    // forward
    // def func(                                                                              <L 276>
    // aabb_id = worldid % ngeom_aabb if wp.static(ngeom_aabb > 1) else 0                     <L 294>
    // center1, center2 = geom_aabb[aabb_id, geom1, 0], geom_aabb[aabb_id, geom2, 0]  # kernel_analyzer: ignore       <L 295>
    var_3 = wp::address(var_geom_aabb, var_1, var_geom1, var_2);
    var_5 = wp::address(var_geom_aabb, var_1, var_geom2, var_4);
    // size1, size2 = geom_aabb[aabb_id, geom1, 1], geom_aabb[aabb_id, geom2, 1]  # kernel_analyzer: ignore       <L 296>
    var_7 = wp::address(var_geom_aabb, var_1, var_geom1, var_6);
    var_9 = wp::address(var_geom_aabb, var_1, var_geom2, var_8);
    // rbound_id = worldid % ngeom_rbound if wp.static(ngeom_rbound > 1) else 0               <L 298>
    // rbound1, rbound2 = geom_rbound[rbound_id, geom1], geom_rbound[rbound_id, geom2]  # kernel_analyzer: ignore       <L 299>
    var_12 = wp::address(var_geom_rbound, var_11, var_geom1);
    var_13 = wp::address(var_geom_rbound, var_11, var_geom2);
    // margin_id = worldid % ngeom_margin if wp.static(ngeom_margin > 1) else 0               <L 300>
    // margin1, margin2 = geom_margin[margin_id, geom1], geom_margin[margin_id, geom2]  # kernel_analyzer: ignore       <L 301>
    var_16 = wp::address(var_geom_margin, var_15, var_geom1);
    var_17 = wp::address(var_geom_margin, var_15, var_geom2);
    // xpos1, xpos2 = geom_xpos_in[worldid, geom1], geom_xpos_in[worldid, geom2]              <L 302>
    var_18 = wp::address(var_geom_xpos_in, var_worldid, var_geom1);
    var_19 = wp::address(var_geom_xpos_in, var_worldid, var_geom2);
    // xmat1, xmat2 = geom_xmat_in[worldid, geom1], geom_xmat_in[worldid, geom2]              <L 303>
    var_20 = wp::address(var_geom_xmat_in, var_worldid, var_geom1);
    var_21 = wp::address(var_geom_xmat_in, var_worldid, var_geom2);
    // if rbound1 == 0.0 or rbound2 == 0.0:                                                   <L 305>
    var_24 = wp::load(var_12);
    var_23 = (var_24 == var_22);
    var_27 = wp::load(var_13);
    var_26 = (var_27 == var_25);
    var_28 = var_23 || var_26;
    if (var_28) {
        // if wp.static(opt_broadphase_filter & BroadphaseFilter.PLANE):                      <L 306>
        // return _plane_filter(rbound1, rbound2, margin1, margin2, xpos1, xpos2, xmat1, xmat2)       <L 307>
        var_31 = wp::load(var_12);
        var_32 = wp::load(var_13);
        var_33 = wp::load(var_16);
        var_34 = wp::load(var_17);
        var_35 = wp::load(var_18);
        var_36 = wp::load(var_19);
        var_37 = wp::load(var_20);
        var_38 = wp::load(var_21);
        var_30 = _plane_filter_0(var_31, var_32, var_33, var_34, var_35, var_36, var_37, var_38);
        return var_30;
    }
    if (!var_28) {
        // if wp.static(opt_broadphase_filter & BroadphaseFilter.SPHERE):                     <L 309>
        // if not _sphere_filter(rbound1, rbound2, margin1, margin2, xpos1, xpos2):           <L 310>
        var_41 = wp::load(var_12);
        var_42 = wp::load(var_13);
        var_43 = wp::load(var_16);
        var_44 = wp::load(var_17);
        var_45 = wp::load(var_18);
        var_46 = wp::load(var_19);
        var_40 = _sphere_filter_0(var_41, var_42, var_43, var_44, var_45, var_46);
        var_47 = wp::unot(var_40);
        if (var_47) {
            // return False                                                                   <L 311>
            return var_48;
        }
        // if wp.static(opt_broadphase_filter & BroadphaseFilter.AABB):                       <L 312>
        // if wp.static(opt_broadphase_filter & BroadphaseFilter.OBB):                        <L 315>
        // if not _obb_filter(center1, center2, size1, size2, margin1, margin2, xpos1, xpos2, xmat1, xmat2):       <L 316>
        var_52 = wp::load(var_3);
        var_53 = wp::load(var_5);
        var_54 = wp::load(var_7);
        var_55 = wp::load(var_9);
        var_56 = wp::load(var_16);
        var_57 = wp::load(var_17);
        var_58 = wp::load(var_18);
        var_59 = wp::load(var_19);
        var_60 = wp::load(var_20);
        var_61 = wp::load(var_21);
        var_51 = _obb_filter_0(var_52, var_53, var_54, var_55, var_56, var_57, var_58, var_59, var_60, var_61);
        var_62 = wp::unot(var_51);
        if (var_62) {
            // return False                                                                   <L 317>
            return var_63;
        }
    }
    // return True                                                                            <L 319>
    return var_64;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_driver.py:324
static CUDA_CALLABLE void _add_geom_pair_0(
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::vec_t<2, wp::int32>> var_nxn_pairid,
    wp::int32 var_naconmax_in,
    wp::int32 var_geom1,
    wp::int32 var_geom2,
    wp::int32 var_worldid,
    wp::int32 var_nxnid,
    wp::array_t<wp::int32> var_ncollision_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pair_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pairid_out,
    wp::array_t<wp::int32> var_collision_worldid_out)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    const wp::int32 var_1 = 1;
    wp::int32 var_2;
    bool var_3;
    wp::int32* var_4;
    wp::int32 var_5;
    wp::int32 var_6;
    wp::int32* var_7;
    wp::int32 var_8;
    wp::int32 var_9;
    bool var_10;
    wp::vec_t<2, wp::int32> var_11;
    wp::vec_t<2, wp::int32> var_12;
    wp::vec_t<2, wp::int32> var_13;
    wp::vec_t<2, wp::int32>* var_14;
    wp::vec_t<2, wp::int32> var_15;
    //---------
    // forward
    // def _add_geom_pair(                                                                    <L 325>
    // pairid = wp.atomic_add(ncollision_out, 0, 1)                                           <L 343>
    var_2 = wp::atomic_add(var_ncollision_out, var_0, var_1);
    // if pairid >= naconmax_in:                                                              <L 345>
    var_3 = (var_2 >= var_naconmax_in);
    if (var_3) {
        // return                                                                             <L 346>
        return;
    }
    // type1 = geom_type[geom1]                                                               <L 348>
    var_4 = wp::address(var_geom_type, var_geom1);
    var_6 = wp::load(var_4);
    var_5 = wp::copy(var_6);
    // type2 = geom_type[geom2]                                                               <L 349>
    var_7 = wp::address(var_geom_type, var_geom2);
    var_9 = wp::load(var_7);
    var_8 = wp::copy(var_9);
    // if type1 > type2:                                                                      <L 351>
    var_10 = (var_5 > var_8);
    if (var_10) {
        // pair = wp.vec2i(geom2, geom1)                                                      <L 352>
        var_11 = wp::vec_t<2, wp::int32>(var_geom2, var_geom1);
    }
    if (!var_10) {
        // pair = wp.vec2i(geom1, geom2)                                                      <L 354>
        var_12 = wp::vec_t<2, wp::int32>(var_geom1, var_geom2);
    }
    var_13 = wp::where(var_10, var_11, var_12);
    // collision_pair_out[pairid] = pair                                                      <L 356>
    wp::array_store(var_collision_pair_out, var_2, var_13);
    // collision_pairid_out[pairid] = nxn_pairid[nxnid]                                       <L 357>
    var_14 = wp::address(var_nxn_pairid, var_nxnid);
    var_15 = wp::load(var_14);
    wp::array_store(var_collision_pairid_out, var_2, var_15);
    // collision_worldid_out[pairid] = worldid                                                <L 358>
    wp::array_store(var_collision_worldid_out, var_2, var_worldid);
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_driver.py:90
static CUDA_CALLABLE void adj__plane_filter_0(
    wp::float32 var_size1,
    wp::float32 var_size2,
    wp::float32 var_margin1,
    wp::float32 var_margin2,
    wp::vec_t<3, wp::float32> var_xpos1,
    wp::vec_t<3, wp::float32> var_xpos2,
    wp::mat_t<3, 3, wp::float32> var_xmat1,
    wp::mat_t<3, 3, wp::float32> var_xmat2,
    wp::float32 & adj_size1,
    wp::float32 & adj_size2,
    wp::float32 & adj_margin1,
    wp::float32 & adj_margin2,
    wp::vec_t<3, wp::float32> & adj_xpos1,
    wp::vec_t<3, wp::float32> & adj_xpos2,
    wp::mat_t<3, 3, wp::float32> & adj_xmat1,
    wp::mat_t<3, 3, wp::float32> & adj_xmat2,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_driver.py:106
static CUDA_CALLABLE void adj__sphere_filter_0(
    wp::float32 var_size1,
    wp::float32 var_size2,
    wp::float32 var_margin1,
    wp::float32 var_margin2,
    wp::vec_t<3, wp::float32> var_xpos1,
    wp::vec_t<3, wp::float32> var_xpos2,
    wp::float32 & adj_size1,
    wp::float32 & adj_size2,
    wp::float32 & adj_margin1,
    wp::float32 & adj_margin2,
    wp::vec_t<3, wp::float32> & adj_xpos1,
    wp::vec_t<3, wp::float32> & adj_xpos2,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_driver.py:217
static CUDA_CALLABLE void adj__obb_filter_0(
    wp::vec_t<3, wp::float32> var_center1,
    wp::vec_t<3, wp::float32> var_center2,
    wp::vec_t<3, wp::float32> var_size1,
    wp::vec_t<3, wp::float32> var_size2,
    wp::float32 var_margin1,
    wp::float32 var_margin2,
    wp::vec_t<3, wp::float32> var_xpos1,
    wp::vec_t<3, wp::float32> var_xpos2,
    wp::mat_t<3, 3, wp::float32> var_xmat1,
    wp::mat_t<3, 3, wp::float32> var_xmat2,
    wp::vec_t<3, wp::float32> & adj_center1,
    wp::vec_t<3, wp::float32> & adj_center2,
    wp::vec_t<3, wp::float32> & adj_size1,
    wp::vec_t<3, wp::float32> & adj_size2,
    wp::float32 & adj_margin1,
    wp::float32 & adj_margin2,
    wp::vec_t<3, wp::float32> & adj_xpos1,
    wp::vec_t<3, wp::float32> & adj_xpos2,
    wp::mat_t<3, 3, wp::float32> & adj_xmat1,
    wp::mat_t<3, 3, wp::float32> & adj_xmat2,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_driver.py:275
static CUDA_CALLABLE void adj__broadphase_filter__locals__func_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_aabb,
    wp::array_t<wp::float32> var_geom_rbound,
    wp::array_t<wp::float32> var_geom_margin,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::int32 var_geom1,
    wp::int32 var_geom2,
    wp::int32 var_worldid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_aabb,
    wp::array_t<wp::float32> & adj_geom_rbound,
    wp::array_t<wp::float32> & adj_geom_margin,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_geom_xmat_in,
    wp::int32 & adj_geom1,
    wp::int32 & adj_geom2,
    wp::int32 & adj_worldid,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_driver.py:324
static CUDA_CALLABLE void adj__add_geom_pair_0(
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::vec_t<2, wp::int32>> var_nxn_pairid,
    wp::int32 var_naconmax_in,
    wp::int32 var_geom1,
    wp::int32 var_geom2,
    wp::int32 var_worldid,
    wp::int32 var_nxnid,
    wp::array_t<wp::int32> var_ncollision_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pair_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pairid_out,
    wp::array_t<wp::int32> var_collision_worldid_out,
    wp::array_t<wp::int32> & adj_geom_type,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_nxn_pairid,
    wp::int32 & adj_naconmax_in,
    wp::int32 & adj_geom1,
    wp::int32 & adj_geom2,
    wp::int32 & adj_worldid,
    wp::int32 & adj_nxnid,
    wp::array_t<wp::int32> & adj_ncollision_out,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_collision_pair_out,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_collision_pairid_out,
    wp::array_t<wp::int32> & adj_collision_worldid_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void _nxn_broadphase__locals__kernel_af3abdea_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_aabb,
    wp::array_t<wp::float32> var_geom_rbound,
    wp::array_t<wp::float32> var_geom_margin,
    wp::array_t<wp::vec_t<2, wp::int32>> var_nxn_geom_pair,
    wp::array_t<wp::vec_t<2, wp::int32>> var_nxn_pairid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::int32 var_naconmax_in,
    wp::array_t<wp::int32> var_ncollision_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pair_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pairid_out,
    wp::array_t<wp::int32> var_collision_worldid_out)
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
        wp::vec_t<2, wp::int32>* var_2;
        wp::vec_t<2, wp::int32> var_3;
        wp::vec_t<2, wp::int32> var_4;
        const wp::int32 var_5 = 0;
        wp::int32 var_6;
        const wp::int32 var_7 = 1;
        wp::int32 var_8;
        bool var_9;
        wp::vec_t<2, wp::int32>* var_10;
        const wp::int32 var_11 = 1;
        wp::int32 var_12;
        wp::vec_t<2, wp::int32> var_13;
        const wp::int32 var_14 = 0;
        bool var_15;
        bool var_16;
        //---------
        // forward
        // def kernel(                                                                            <L 649>
        // worldid, elementid = wp.tid()                                                          <L 668>
        builtin_tid2d(var_0, var_1);
        // geom = nxn_geom_pair[elementid]                                                        <L 670>
        var_2 = wp::address(var_nxn_geom_pair, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // geom1 = geom[0]                                                                        <L 671>
        var_6 = wp::extract(var_3, var_5);
        // geom2 = geom[1]                                                                        <L 672>
        var_8 = wp::extract(var_3, var_7);
        // if (                                                                                   <L 674>
        // wp.static(_broadphase_filter(opt_broadphase_filter, ngeom_aabb, ngeom_rbound, ngeom_margin))(       <L 675>
        // geom_aabb, geom_rbound, geom_margin, geom_xpos_in, geom_xmat_in, geom1, geom2, worldid       <L 676>
        var_9 = _broadphase_filter__locals__func_0(var_geom_aabb, var_geom_rbound, var_geom_margin, var_geom_xpos_in, var_geom_xmat_in, var_6, var_8, var_0);
        // or nxn_pairid[elementid][1] >= 0                                                       <L 678>
        var_10 = wp::address(var_nxn_pairid, var_1);
        var_13 = wp::load(var_10);
        var_12 = wp::extract(var_13, var_11);
        var_15 = (var_12 >= var_14);
        var_16 = var_9 || var_15;
        if (var_16) {
            // _add_geom_pair(                                                                    <L 680>
            // geom_type,                                                                         <L 681>
            // nxn_pairid,                                                                        <L 682>
            // naconmax_in,                                                                       <L 683>
            // geom1,                                                                             <L 684>
            // geom2,                                                                             <L 685>
            // worldid,                                                                           <L 686>
            // elementid,                                                                         <L 687>
            // ncollision_out,                                                                    <L 688>
            // collision_pair_out,                                                                <L 689>
            // collision_pairid_out,                                                              <L 690>
            // collision_worldid_out,                                                             <L 691>
            _add_geom_pair_0(var_geom_type, var_nxn_pairid, var_naconmax_in, var_6, var_8, var_0, var_1, var_ncollision_out, var_collision_pair_out, var_collision_pairid_out, var_collision_worldid_out);
        }
    }
}

