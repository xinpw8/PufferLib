
#define WP_TILE_BLOCK_DIM 64
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/smooth.py:2700
static CUDA_CALLABLE void _solve_LD_sparse_fused__locals___syncthreads_0(
    )
{
WP_TILE_SYNC();}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/smooth.py:2700
static CUDA_CALLABLE void adj__solve_LD_sparse_fused__locals___syncthreads_0(
    )
{
}



extern "C" __global__ void _solve_LD_sparse_fused__locals__kernel_ed64ef36_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_L,
    wp::array_t<wp::float32> var_D,
    wp::array_t<wp::vec_t<3, wp::int32>> var_all_updates,
    wp::array_t<wp::int32> var_level_offsets,
    wp::array_t<wp::float32> var_y,
    wp::array_t<wp::float32> var_x_out)
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
        const wp::int32 var_2 = 70;
        const wp::int32 var_3 = 15;
        wp::int32 var_4;
        const wp::int32 var_5 = 70;
        wp::range_t var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        const wp::int32 var_10 = 0;
        const wp::int32 var_11 = 1;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        const wp::int32 var_17 = 1;
        wp::int32 var_18;
        wp::int32* var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        wp::range_t var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        wp::vec_t<3, wp::int32>* var_25;
        wp::vec_t<3, wp::int32> var_26;
        wp::vec_t<3, wp::int32> var_27;
        const wp::int32 var_28 = 0;
        wp::int32 var_29;
        const wp::int32 var_30 = 1;
        wp::int32 var_31;
        const wp::int32 var_32 = 2;
        wp::int32 var_33;
        wp::slice_t var_34;
        const wp::int32 var_35 = 0;
        wp::array_t<wp::float32> var_36;
        const wp::int32 var_37 = 0;
        wp::float32* var_38;
        wp::float32* var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        const wp::int32 var_44 = 1;
        const wp::int32 var_45 = 1;
        wp::int32 var_46;
        wp::int32 var_47;
        wp::int32* var_48;
        wp::int32 var_49;
        wp::int32 var_50;
        const wp::int32 var_51 = 1;
        wp::int32 var_52;
        wp::int32* var_53;
        wp::int32 var_54;
        wp::int32 var_55;
        wp::range_t var_56;
        wp::int32 var_57;
        wp::int32 var_58;
        wp::vec_t<3, wp::int32>* var_59;
        wp::vec_t<3, wp::int32> var_60;
        wp::vec_t<3, wp::int32> var_61;
        const wp::int32 var_62 = 0;
        wp::int32 var_63;
        const wp::int32 var_64 = 1;
        wp::int32 var_65;
        const wp::int32 var_66 = 2;
        wp::int32 var_67;
        wp::slice_t var_68;
        const wp::int32 var_69 = 0;
        wp::array_t<wp::float32> var_70;
        const wp::int32 var_71 = 0;
        wp::float32* var_72;
        wp::float32* var_73;
        wp::float32 var_74;
        wp::float32 var_75;
        wp::float32 var_76;
        wp::float32 var_77;
        const wp::int32 var_78 = 2;
        const wp::int32 var_79 = 1;
        wp::int32 var_80;
        wp::int32 var_81;
        wp::int32* var_82;
        wp::int32 var_83;
        wp::int32 var_84;
        const wp::int32 var_85 = 1;
        wp::int32 var_86;
        wp::int32* var_87;
        wp::int32 var_88;
        wp::int32 var_89;
        wp::range_t var_90;
        wp::int32 var_91;
        wp::int32 var_92;
        wp::vec_t<3, wp::int32>* var_93;
        wp::vec_t<3, wp::int32> var_94;
        wp::vec_t<3, wp::int32> var_95;
        const wp::int32 var_96 = 0;
        wp::int32 var_97;
        const wp::int32 var_98 = 1;
        wp::int32 var_99;
        const wp::int32 var_100 = 2;
        wp::int32 var_101;
        wp::slice_t var_102;
        const wp::int32 var_103 = 0;
        wp::array_t<wp::float32> var_104;
        const wp::int32 var_105 = 0;
        wp::float32* var_106;
        wp::float32* var_107;
        wp::float32 var_108;
        wp::float32 var_109;
        wp::float32 var_110;
        wp::float32 var_111;
        const wp::int32 var_112 = 3;
        const wp::int32 var_113 = 1;
        wp::int32 var_114;
        wp::int32 var_115;
        wp::int32* var_116;
        wp::int32 var_117;
        wp::int32 var_118;
        const wp::int32 var_119 = 1;
        wp::int32 var_120;
        wp::int32* var_121;
        wp::int32 var_122;
        wp::int32 var_123;
        wp::range_t var_124;
        wp::int32 var_125;
        wp::int32 var_126;
        wp::vec_t<3, wp::int32>* var_127;
        wp::vec_t<3, wp::int32> var_128;
        wp::vec_t<3, wp::int32> var_129;
        const wp::int32 var_130 = 0;
        wp::int32 var_131;
        const wp::int32 var_132 = 1;
        wp::int32 var_133;
        const wp::int32 var_134 = 2;
        wp::int32 var_135;
        wp::slice_t var_136;
        const wp::int32 var_137 = 0;
        wp::array_t<wp::float32> var_138;
        const wp::int32 var_139 = 0;
        wp::float32* var_140;
        wp::float32* var_141;
        wp::float32 var_142;
        wp::float32 var_143;
        wp::float32 var_144;
        wp::float32 var_145;
        const wp::int32 var_146 = 4;
        const wp::int32 var_147 = 1;
        wp::int32 var_148;
        wp::int32 var_149;
        wp::int32* var_150;
        wp::int32 var_151;
        wp::int32 var_152;
        const wp::int32 var_153 = 1;
        wp::int32 var_154;
        wp::int32* var_155;
        wp::int32 var_156;
        wp::int32 var_157;
        wp::range_t var_158;
        wp::int32 var_159;
        wp::int32 var_160;
        wp::vec_t<3, wp::int32>* var_161;
        wp::vec_t<3, wp::int32> var_162;
        wp::vec_t<3, wp::int32> var_163;
        const wp::int32 var_164 = 0;
        wp::int32 var_165;
        const wp::int32 var_166 = 1;
        wp::int32 var_167;
        const wp::int32 var_168 = 2;
        wp::int32 var_169;
        wp::slice_t var_170;
        const wp::int32 var_171 = 0;
        wp::array_t<wp::float32> var_172;
        const wp::int32 var_173 = 0;
        wp::float32* var_174;
        wp::float32* var_175;
        wp::float32 var_176;
        wp::float32 var_177;
        wp::float32 var_178;
        wp::float32 var_179;
        const wp::int32 var_180 = 5;
        const wp::int32 var_181 = 1;
        wp::int32 var_182;
        wp::int32 var_183;
        wp::int32* var_184;
        wp::int32 var_185;
        wp::int32 var_186;
        const wp::int32 var_187 = 1;
        wp::int32 var_188;
        wp::int32* var_189;
        wp::int32 var_190;
        wp::int32 var_191;
        wp::range_t var_192;
        wp::int32 var_193;
        wp::int32 var_194;
        wp::vec_t<3, wp::int32>* var_195;
        wp::vec_t<3, wp::int32> var_196;
        wp::vec_t<3, wp::int32> var_197;
        const wp::int32 var_198 = 0;
        wp::int32 var_199;
        const wp::int32 var_200 = 1;
        wp::int32 var_201;
        const wp::int32 var_202 = 2;
        wp::int32 var_203;
        wp::slice_t var_204;
        const wp::int32 var_205 = 0;
        wp::array_t<wp::float32> var_206;
        const wp::int32 var_207 = 0;
        wp::float32* var_208;
        wp::float32* var_209;
        wp::float32 var_210;
        wp::float32 var_211;
        wp::float32 var_212;
        wp::float32 var_213;
        const wp::int32 var_214 = 6;
        const wp::int32 var_215 = 1;
        wp::int32 var_216;
        wp::int32 var_217;
        wp::int32* var_218;
        wp::int32 var_219;
        wp::int32 var_220;
        const wp::int32 var_221 = 1;
        wp::int32 var_222;
        wp::int32* var_223;
        wp::int32 var_224;
        wp::int32 var_225;
        wp::range_t var_226;
        wp::int32 var_227;
        wp::int32 var_228;
        wp::vec_t<3, wp::int32>* var_229;
        wp::vec_t<3, wp::int32> var_230;
        wp::vec_t<3, wp::int32> var_231;
        const wp::int32 var_232 = 0;
        wp::int32 var_233;
        const wp::int32 var_234 = 1;
        wp::int32 var_235;
        const wp::int32 var_236 = 2;
        wp::int32 var_237;
        wp::slice_t var_238;
        const wp::int32 var_239 = 0;
        wp::array_t<wp::float32> var_240;
        const wp::int32 var_241 = 0;
        wp::float32* var_242;
        wp::float32* var_243;
        wp::float32 var_244;
        wp::float32 var_245;
        wp::float32 var_246;
        wp::float32 var_247;
        const wp::int32 var_248 = 7;
        const wp::int32 var_249 = 1;
        wp::int32 var_250;
        wp::int32 var_251;
        wp::int32* var_252;
        wp::int32 var_253;
        wp::int32 var_254;
        const wp::int32 var_255 = 1;
        wp::int32 var_256;
        wp::int32* var_257;
        wp::int32 var_258;
        wp::int32 var_259;
        wp::range_t var_260;
        wp::int32 var_261;
        wp::int32 var_262;
        wp::vec_t<3, wp::int32>* var_263;
        wp::vec_t<3, wp::int32> var_264;
        wp::vec_t<3, wp::int32> var_265;
        const wp::int32 var_266 = 0;
        wp::int32 var_267;
        const wp::int32 var_268 = 1;
        wp::int32 var_269;
        const wp::int32 var_270 = 2;
        wp::int32 var_271;
        wp::slice_t var_272;
        const wp::int32 var_273 = 0;
        wp::array_t<wp::float32> var_274;
        const wp::int32 var_275 = 0;
        wp::float32* var_276;
        wp::float32* var_277;
        wp::float32 var_278;
        wp::float32 var_279;
        wp::float32 var_280;
        wp::float32 var_281;
        const wp::int32 var_282 = 8;
        const wp::int32 var_283 = 1;
        wp::int32 var_284;
        wp::int32 var_285;
        wp::int32* var_286;
        wp::int32 var_287;
        wp::int32 var_288;
        const wp::int32 var_289 = 1;
        wp::int32 var_290;
        wp::int32* var_291;
        wp::int32 var_292;
        wp::int32 var_293;
        wp::range_t var_294;
        wp::int32 var_295;
        wp::int32 var_296;
        wp::vec_t<3, wp::int32>* var_297;
        wp::vec_t<3, wp::int32> var_298;
        wp::vec_t<3, wp::int32> var_299;
        const wp::int32 var_300 = 0;
        wp::int32 var_301;
        const wp::int32 var_302 = 1;
        wp::int32 var_303;
        const wp::int32 var_304 = 2;
        wp::int32 var_305;
        wp::slice_t var_306;
        const wp::int32 var_307 = 0;
        wp::array_t<wp::float32> var_308;
        const wp::int32 var_309 = 0;
        wp::float32* var_310;
        wp::float32* var_311;
        wp::float32 var_312;
        wp::float32 var_313;
        wp::float32 var_314;
        wp::float32 var_315;
        const wp::int32 var_316 = 9;
        const wp::int32 var_317 = 1;
        wp::int32 var_318;
        wp::int32 var_319;
        wp::int32* var_320;
        wp::int32 var_321;
        wp::int32 var_322;
        const wp::int32 var_323 = 1;
        wp::int32 var_324;
        wp::int32* var_325;
        wp::int32 var_326;
        wp::int32 var_327;
        wp::range_t var_328;
        wp::int32 var_329;
        wp::int32 var_330;
        wp::vec_t<3, wp::int32>* var_331;
        wp::vec_t<3, wp::int32> var_332;
        wp::vec_t<3, wp::int32> var_333;
        const wp::int32 var_334 = 0;
        wp::int32 var_335;
        const wp::int32 var_336 = 1;
        wp::int32 var_337;
        const wp::int32 var_338 = 2;
        wp::int32 var_339;
        wp::slice_t var_340;
        const wp::int32 var_341 = 0;
        wp::array_t<wp::float32> var_342;
        const wp::int32 var_343 = 0;
        wp::float32* var_344;
        wp::float32* var_345;
        wp::float32 var_346;
        wp::float32 var_347;
        wp::float32 var_348;
        wp::float32 var_349;
        const wp::int32 var_350 = 10;
        const wp::int32 var_351 = 1;
        wp::int32 var_352;
        wp::int32 var_353;
        wp::int32* var_354;
        wp::int32 var_355;
        wp::int32 var_356;
        const wp::int32 var_357 = 1;
        wp::int32 var_358;
        wp::int32* var_359;
        wp::int32 var_360;
        wp::int32 var_361;
        wp::range_t var_362;
        wp::int32 var_363;
        wp::int32 var_364;
        wp::vec_t<3, wp::int32>* var_365;
        wp::vec_t<3, wp::int32> var_366;
        wp::vec_t<3, wp::int32> var_367;
        const wp::int32 var_368 = 0;
        wp::int32 var_369;
        const wp::int32 var_370 = 1;
        wp::int32 var_371;
        const wp::int32 var_372 = 2;
        wp::int32 var_373;
        wp::slice_t var_374;
        const wp::int32 var_375 = 0;
        wp::array_t<wp::float32> var_376;
        const wp::int32 var_377 = 0;
        wp::float32* var_378;
        wp::float32* var_379;
        wp::float32 var_380;
        wp::float32 var_381;
        wp::float32 var_382;
        wp::float32 var_383;
        const wp::int32 var_384 = 11;
        const wp::int32 var_385 = 1;
        wp::int32 var_386;
        wp::int32 var_387;
        wp::int32* var_388;
        wp::int32 var_389;
        wp::int32 var_390;
        const wp::int32 var_391 = 1;
        wp::int32 var_392;
        wp::int32* var_393;
        wp::int32 var_394;
        wp::int32 var_395;
        wp::range_t var_396;
        wp::int32 var_397;
        wp::int32 var_398;
        wp::vec_t<3, wp::int32>* var_399;
        wp::vec_t<3, wp::int32> var_400;
        wp::vec_t<3, wp::int32> var_401;
        const wp::int32 var_402 = 0;
        wp::int32 var_403;
        const wp::int32 var_404 = 1;
        wp::int32 var_405;
        const wp::int32 var_406 = 2;
        wp::int32 var_407;
        wp::slice_t var_408;
        const wp::int32 var_409 = 0;
        wp::array_t<wp::float32> var_410;
        const wp::int32 var_411 = 0;
        wp::float32* var_412;
        wp::float32* var_413;
        wp::float32 var_414;
        wp::float32 var_415;
        wp::float32 var_416;
        wp::float32 var_417;
        const wp::int32 var_418 = 12;
        const wp::int32 var_419 = 1;
        wp::int32 var_420;
        wp::int32 var_421;
        wp::int32* var_422;
        wp::int32 var_423;
        wp::int32 var_424;
        const wp::int32 var_425 = 1;
        wp::int32 var_426;
        wp::int32* var_427;
        wp::int32 var_428;
        wp::int32 var_429;
        wp::range_t var_430;
        wp::int32 var_431;
        wp::int32 var_432;
        wp::vec_t<3, wp::int32>* var_433;
        wp::vec_t<3, wp::int32> var_434;
        wp::vec_t<3, wp::int32> var_435;
        const wp::int32 var_436 = 0;
        wp::int32 var_437;
        const wp::int32 var_438 = 1;
        wp::int32 var_439;
        const wp::int32 var_440 = 2;
        wp::int32 var_441;
        wp::slice_t var_442;
        const wp::int32 var_443 = 0;
        wp::array_t<wp::float32> var_444;
        const wp::int32 var_445 = 0;
        wp::float32* var_446;
        wp::float32* var_447;
        wp::float32 var_448;
        wp::float32 var_449;
        wp::float32 var_450;
        wp::float32 var_451;
        const wp::int32 var_452 = 13;
        const wp::int32 var_453 = 1;
        wp::int32 var_454;
        wp::int32 var_455;
        wp::int32* var_456;
        wp::int32 var_457;
        wp::int32 var_458;
        const wp::int32 var_459 = 1;
        wp::int32 var_460;
        wp::int32* var_461;
        wp::int32 var_462;
        wp::int32 var_463;
        wp::range_t var_464;
        wp::int32 var_465;
        wp::int32 var_466;
        wp::vec_t<3, wp::int32>* var_467;
        wp::vec_t<3, wp::int32> var_468;
        wp::vec_t<3, wp::int32> var_469;
        const wp::int32 var_470 = 0;
        wp::int32 var_471;
        const wp::int32 var_472 = 1;
        wp::int32 var_473;
        const wp::int32 var_474 = 2;
        wp::int32 var_475;
        wp::slice_t var_476;
        const wp::int32 var_477 = 0;
        wp::array_t<wp::float32> var_478;
        const wp::int32 var_479 = 0;
        wp::float32* var_480;
        wp::float32* var_481;
        wp::float32 var_482;
        wp::float32 var_483;
        wp::float32 var_484;
        wp::float32 var_485;
        const wp::int32 var_486 = 14;
        const wp::int32 var_487 = 1;
        wp::int32 var_488;
        wp::int32 var_489;
        wp::int32* var_490;
        wp::int32 var_491;
        wp::int32 var_492;
        const wp::int32 var_493 = 1;
        wp::int32 var_494;
        wp::int32* var_495;
        wp::int32 var_496;
        wp::int32 var_497;
        wp::range_t var_498;
        wp::int32 var_499;
        wp::int32 var_500;
        wp::vec_t<3, wp::int32>* var_501;
        wp::vec_t<3, wp::int32> var_502;
        wp::vec_t<3, wp::int32> var_503;
        const wp::int32 var_504 = 0;
        wp::int32 var_505;
        const wp::int32 var_506 = 1;
        wp::int32 var_507;
        const wp::int32 var_508 = 2;
        wp::int32 var_509;
        wp::slice_t var_510;
        const wp::int32 var_511 = 0;
        wp::array_t<wp::float32> var_512;
        const wp::int32 var_513 = 0;
        wp::float32* var_514;
        wp::float32* var_515;
        wp::float32 var_516;
        wp::float32 var_517;
        wp::float32 var_518;
        wp::float32 var_519;
        const wp::int32 var_520 = 70;
        wp::range_t var_521;
        wp::int32 var_522;
        wp::float32* var_523;
        wp::float32* var_524;
        wp::float32* var_525;
        wp::float32 var_526;
        wp::float32 var_527;
        wp::float32 var_528;
        const wp::int32 var_529 = 0;
        wp::int32 var_530;
        wp::int32* var_531;
        wp::int32 var_532;
        wp::int32 var_533;
        const wp::int32 var_534 = 1;
        wp::int32 var_535;
        wp::int32* var_536;
        wp::int32 var_537;
        wp::int32 var_538;
        wp::range_t var_539;
        wp::int32 var_540;
        wp::int32 var_541;
        wp::vec_t<3, wp::int32>* var_542;
        wp::vec_t<3, wp::int32> var_543;
        wp::vec_t<3, wp::int32> var_544;
        const wp::int32 var_545 = 0;
        wp::int32 var_546;
        const wp::int32 var_547 = 1;
        wp::int32 var_548;
        const wp::int32 var_549 = 2;
        wp::int32 var_550;
        wp::slice_t var_551;
        const wp::int32 var_552 = 0;
        wp::array_t<wp::float32> var_553;
        const wp::int32 var_554 = 0;
        wp::float32* var_555;
        wp::float32* var_556;
        wp::float32 var_557;
        wp::float32 var_558;
        wp::float32 var_559;
        wp::float32 var_560;
        const wp::int32 var_561 = 1;
        wp::int32 var_562;
        wp::int32* var_563;
        wp::int32 var_564;
        wp::int32 var_565;
        const wp::int32 var_566 = 1;
        wp::int32 var_567;
        wp::int32* var_568;
        wp::int32 var_569;
        wp::int32 var_570;
        wp::range_t var_571;
        wp::int32 var_572;
        wp::int32 var_573;
        wp::vec_t<3, wp::int32>* var_574;
        wp::vec_t<3, wp::int32> var_575;
        wp::vec_t<3, wp::int32> var_576;
        const wp::int32 var_577 = 0;
        wp::int32 var_578;
        const wp::int32 var_579 = 1;
        wp::int32 var_580;
        const wp::int32 var_581 = 2;
        wp::int32 var_582;
        wp::slice_t var_583;
        const wp::int32 var_584 = 0;
        wp::array_t<wp::float32> var_585;
        const wp::int32 var_586 = 0;
        wp::float32* var_587;
        wp::float32* var_588;
        wp::float32 var_589;
        wp::float32 var_590;
        wp::float32 var_591;
        wp::float32 var_592;
        const wp::int32 var_593 = 2;
        wp::int32 var_594;
        wp::int32* var_595;
        wp::int32 var_596;
        wp::int32 var_597;
        const wp::int32 var_598 = 1;
        wp::int32 var_599;
        wp::int32* var_600;
        wp::int32 var_601;
        wp::int32 var_602;
        wp::range_t var_603;
        wp::int32 var_604;
        wp::int32 var_605;
        wp::vec_t<3, wp::int32>* var_606;
        wp::vec_t<3, wp::int32> var_607;
        wp::vec_t<3, wp::int32> var_608;
        const wp::int32 var_609 = 0;
        wp::int32 var_610;
        const wp::int32 var_611 = 1;
        wp::int32 var_612;
        const wp::int32 var_613 = 2;
        wp::int32 var_614;
        wp::slice_t var_615;
        const wp::int32 var_616 = 0;
        wp::array_t<wp::float32> var_617;
        const wp::int32 var_618 = 0;
        wp::float32* var_619;
        wp::float32* var_620;
        wp::float32 var_621;
        wp::float32 var_622;
        wp::float32 var_623;
        wp::float32 var_624;
        const wp::int32 var_625 = 3;
        wp::int32 var_626;
        wp::int32* var_627;
        wp::int32 var_628;
        wp::int32 var_629;
        const wp::int32 var_630 = 1;
        wp::int32 var_631;
        wp::int32* var_632;
        wp::int32 var_633;
        wp::int32 var_634;
        wp::range_t var_635;
        wp::int32 var_636;
        wp::int32 var_637;
        wp::vec_t<3, wp::int32>* var_638;
        wp::vec_t<3, wp::int32> var_639;
        wp::vec_t<3, wp::int32> var_640;
        const wp::int32 var_641 = 0;
        wp::int32 var_642;
        const wp::int32 var_643 = 1;
        wp::int32 var_644;
        const wp::int32 var_645 = 2;
        wp::int32 var_646;
        wp::slice_t var_647;
        const wp::int32 var_648 = 0;
        wp::array_t<wp::float32> var_649;
        const wp::int32 var_650 = 0;
        wp::float32* var_651;
        wp::float32* var_652;
        wp::float32 var_653;
        wp::float32 var_654;
        wp::float32 var_655;
        wp::float32 var_656;
        const wp::int32 var_657 = 4;
        wp::int32 var_658;
        wp::int32* var_659;
        wp::int32 var_660;
        wp::int32 var_661;
        const wp::int32 var_662 = 1;
        wp::int32 var_663;
        wp::int32* var_664;
        wp::int32 var_665;
        wp::int32 var_666;
        wp::range_t var_667;
        wp::int32 var_668;
        wp::int32 var_669;
        wp::vec_t<3, wp::int32>* var_670;
        wp::vec_t<3, wp::int32> var_671;
        wp::vec_t<3, wp::int32> var_672;
        const wp::int32 var_673 = 0;
        wp::int32 var_674;
        const wp::int32 var_675 = 1;
        wp::int32 var_676;
        const wp::int32 var_677 = 2;
        wp::int32 var_678;
        wp::slice_t var_679;
        const wp::int32 var_680 = 0;
        wp::array_t<wp::float32> var_681;
        const wp::int32 var_682 = 0;
        wp::float32* var_683;
        wp::float32* var_684;
        wp::float32 var_685;
        wp::float32 var_686;
        wp::float32 var_687;
        wp::float32 var_688;
        const wp::int32 var_689 = 5;
        wp::int32 var_690;
        wp::int32* var_691;
        wp::int32 var_692;
        wp::int32 var_693;
        const wp::int32 var_694 = 1;
        wp::int32 var_695;
        wp::int32* var_696;
        wp::int32 var_697;
        wp::int32 var_698;
        wp::range_t var_699;
        wp::int32 var_700;
        wp::int32 var_701;
        wp::vec_t<3, wp::int32>* var_702;
        wp::vec_t<3, wp::int32> var_703;
        wp::vec_t<3, wp::int32> var_704;
        const wp::int32 var_705 = 0;
        wp::int32 var_706;
        const wp::int32 var_707 = 1;
        wp::int32 var_708;
        const wp::int32 var_709 = 2;
        wp::int32 var_710;
        wp::slice_t var_711;
        const wp::int32 var_712 = 0;
        wp::array_t<wp::float32> var_713;
        const wp::int32 var_714 = 0;
        wp::float32* var_715;
        wp::float32* var_716;
        wp::float32 var_717;
        wp::float32 var_718;
        wp::float32 var_719;
        wp::float32 var_720;
        const wp::int32 var_721 = 6;
        wp::int32 var_722;
        wp::int32* var_723;
        wp::int32 var_724;
        wp::int32 var_725;
        const wp::int32 var_726 = 1;
        wp::int32 var_727;
        wp::int32* var_728;
        wp::int32 var_729;
        wp::int32 var_730;
        wp::range_t var_731;
        wp::int32 var_732;
        wp::int32 var_733;
        wp::vec_t<3, wp::int32>* var_734;
        wp::vec_t<3, wp::int32> var_735;
        wp::vec_t<3, wp::int32> var_736;
        const wp::int32 var_737 = 0;
        wp::int32 var_738;
        const wp::int32 var_739 = 1;
        wp::int32 var_740;
        const wp::int32 var_741 = 2;
        wp::int32 var_742;
        wp::slice_t var_743;
        const wp::int32 var_744 = 0;
        wp::array_t<wp::float32> var_745;
        const wp::int32 var_746 = 0;
        wp::float32* var_747;
        wp::float32* var_748;
        wp::float32 var_749;
        wp::float32 var_750;
        wp::float32 var_751;
        wp::float32 var_752;
        const wp::int32 var_753 = 7;
        wp::int32 var_754;
        wp::int32* var_755;
        wp::int32 var_756;
        wp::int32 var_757;
        const wp::int32 var_758 = 1;
        wp::int32 var_759;
        wp::int32* var_760;
        wp::int32 var_761;
        wp::int32 var_762;
        wp::range_t var_763;
        wp::int32 var_764;
        wp::int32 var_765;
        wp::vec_t<3, wp::int32>* var_766;
        wp::vec_t<3, wp::int32> var_767;
        wp::vec_t<3, wp::int32> var_768;
        const wp::int32 var_769 = 0;
        wp::int32 var_770;
        const wp::int32 var_771 = 1;
        wp::int32 var_772;
        const wp::int32 var_773 = 2;
        wp::int32 var_774;
        wp::slice_t var_775;
        const wp::int32 var_776 = 0;
        wp::array_t<wp::float32> var_777;
        const wp::int32 var_778 = 0;
        wp::float32* var_779;
        wp::float32* var_780;
        wp::float32 var_781;
        wp::float32 var_782;
        wp::float32 var_783;
        wp::float32 var_784;
        const wp::int32 var_785 = 8;
        wp::int32 var_786;
        wp::int32* var_787;
        wp::int32 var_788;
        wp::int32 var_789;
        const wp::int32 var_790 = 1;
        wp::int32 var_791;
        wp::int32* var_792;
        wp::int32 var_793;
        wp::int32 var_794;
        wp::range_t var_795;
        wp::int32 var_796;
        wp::int32 var_797;
        wp::vec_t<3, wp::int32>* var_798;
        wp::vec_t<3, wp::int32> var_799;
        wp::vec_t<3, wp::int32> var_800;
        const wp::int32 var_801 = 0;
        wp::int32 var_802;
        const wp::int32 var_803 = 1;
        wp::int32 var_804;
        const wp::int32 var_805 = 2;
        wp::int32 var_806;
        wp::slice_t var_807;
        const wp::int32 var_808 = 0;
        wp::array_t<wp::float32> var_809;
        const wp::int32 var_810 = 0;
        wp::float32* var_811;
        wp::float32* var_812;
        wp::float32 var_813;
        wp::float32 var_814;
        wp::float32 var_815;
        wp::float32 var_816;
        const wp::int32 var_817 = 9;
        wp::int32 var_818;
        wp::int32* var_819;
        wp::int32 var_820;
        wp::int32 var_821;
        const wp::int32 var_822 = 1;
        wp::int32 var_823;
        wp::int32* var_824;
        wp::int32 var_825;
        wp::int32 var_826;
        wp::range_t var_827;
        wp::int32 var_828;
        wp::int32 var_829;
        wp::vec_t<3, wp::int32>* var_830;
        wp::vec_t<3, wp::int32> var_831;
        wp::vec_t<3, wp::int32> var_832;
        const wp::int32 var_833 = 0;
        wp::int32 var_834;
        const wp::int32 var_835 = 1;
        wp::int32 var_836;
        const wp::int32 var_837 = 2;
        wp::int32 var_838;
        wp::slice_t var_839;
        const wp::int32 var_840 = 0;
        wp::array_t<wp::float32> var_841;
        const wp::int32 var_842 = 0;
        wp::float32* var_843;
        wp::float32* var_844;
        wp::float32 var_845;
        wp::float32 var_846;
        wp::float32 var_847;
        wp::float32 var_848;
        const wp::int32 var_849 = 10;
        wp::int32 var_850;
        wp::int32* var_851;
        wp::int32 var_852;
        wp::int32 var_853;
        const wp::int32 var_854 = 1;
        wp::int32 var_855;
        wp::int32* var_856;
        wp::int32 var_857;
        wp::int32 var_858;
        wp::range_t var_859;
        wp::int32 var_860;
        wp::int32 var_861;
        wp::vec_t<3, wp::int32>* var_862;
        wp::vec_t<3, wp::int32> var_863;
        wp::vec_t<3, wp::int32> var_864;
        const wp::int32 var_865 = 0;
        wp::int32 var_866;
        const wp::int32 var_867 = 1;
        wp::int32 var_868;
        const wp::int32 var_869 = 2;
        wp::int32 var_870;
        wp::slice_t var_871;
        const wp::int32 var_872 = 0;
        wp::array_t<wp::float32> var_873;
        const wp::int32 var_874 = 0;
        wp::float32* var_875;
        wp::float32* var_876;
        wp::float32 var_877;
        wp::float32 var_878;
        wp::float32 var_879;
        wp::float32 var_880;
        const wp::int32 var_881 = 11;
        wp::int32 var_882;
        wp::int32* var_883;
        wp::int32 var_884;
        wp::int32 var_885;
        const wp::int32 var_886 = 1;
        wp::int32 var_887;
        wp::int32* var_888;
        wp::int32 var_889;
        wp::int32 var_890;
        wp::range_t var_891;
        wp::int32 var_892;
        wp::int32 var_893;
        wp::vec_t<3, wp::int32>* var_894;
        wp::vec_t<3, wp::int32> var_895;
        wp::vec_t<3, wp::int32> var_896;
        const wp::int32 var_897 = 0;
        wp::int32 var_898;
        const wp::int32 var_899 = 1;
        wp::int32 var_900;
        const wp::int32 var_901 = 2;
        wp::int32 var_902;
        wp::slice_t var_903;
        const wp::int32 var_904 = 0;
        wp::array_t<wp::float32> var_905;
        const wp::int32 var_906 = 0;
        wp::float32* var_907;
        wp::float32* var_908;
        wp::float32 var_909;
        wp::float32 var_910;
        wp::float32 var_911;
        wp::float32 var_912;
        const wp::int32 var_913 = 12;
        wp::int32 var_914;
        wp::int32* var_915;
        wp::int32 var_916;
        wp::int32 var_917;
        const wp::int32 var_918 = 1;
        wp::int32 var_919;
        wp::int32* var_920;
        wp::int32 var_921;
        wp::int32 var_922;
        wp::range_t var_923;
        wp::int32 var_924;
        wp::int32 var_925;
        wp::vec_t<3, wp::int32>* var_926;
        wp::vec_t<3, wp::int32> var_927;
        wp::vec_t<3, wp::int32> var_928;
        const wp::int32 var_929 = 0;
        wp::int32 var_930;
        const wp::int32 var_931 = 1;
        wp::int32 var_932;
        const wp::int32 var_933 = 2;
        wp::int32 var_934;
        wp::slice_t var_935;
        const wp::int32 var_936 = 0;
        wp::array_t<wp::float32> var_937;
        const wp::int32 var_938 = 0;
        wp::float32* var_939;
        wp::float32* var_940;
        wp::float32 var_941;
        wp::float32 var_942;
        wp::float32 var_943;
        wp::float32 var_944;
        const wp::int32 var_945 = 13;
        wp::int32 var_946;
        wp::int32* var_947;
        wp::int32 var_948;
        wp::int32 var_949;
        const wp::int32 var_950 = 1;
        wp::int32 var_951;
        wp::int32* var_952;
        wp::int32 var_953;
        wp::int32 var_954;
        wp::range_t var_955;
        wp::int32 var_956;
        wp::int32 var_957;
        wp::vec_t<3, wp::int32>* var_958;
        wp::vec_t<3, wp::int32> var_959;
        wp::vec_t<3, wp::int32> var_960;
        const wp::int32 var_961 = 0;
        wp::int32 var_962;
        const wp::int32 var_963 = 1;
        wp::int32 var_964;
        const wp::int32 var_965 = 2;
        wp::int32 var_966;
        wp::slice_t var_967;
        const wp::int32 var_968 = 0;
        wp::array_t<wp::float32> var_969;
        const wp::int32 var_970 = 0;
        wp::float32* var_971;
        wp::float32* var_972;
        wp::float32 var_973;
        wp::float32 var_974;
        wp::float32 var_975;
        wp::float32 var_976;
        const wp::int32 var_977 = 14;
        wp::int32 var_978;
        wp::int32* var_979;
        wp::int32 var_980;
        wp::int32 var_981;
        const wp::int32 var_982 = 1;
        wp::int32 var_983;
        wp::int32* var_984;
        wp::int32 var_985;
        wp::int32 var_986;
        wp::range_t var_987;
        wp::int32 var_988;
        wp::int32 var_989;
        wp::vec_t<3, wp::int32>* var_990;
        wp::vec_t<3, wp::int32> var_991;
        wp::vec_t<3, wp::int32> var_992;
        const wp::int32 var_993 = 0;
        wp::int32 var_994;
        const wp::int32 var_995 = 1;
        wp::int32 var_996;
        const wp::int32 var_997 = 2;
        wp::int32 var_998;
        wp::slice_t var_999;
        const wp::int32 var_1000 = 0;
        wp::array_t<wp::float32> var_1001;
        const wp::int32 var_1002 = 0;
        wp::float32* var_1003;
        wp::float32* var_1004;
        wp::float32 var_1005;
        wp::float32 var_1006;
        wp::float32 var_1007;
        wp::float32 var_1008;
        //---------
        // forward
        // def kernel(                                                                            <L 2705>
        // worldid, tid = wp.tid()                                                                <L 2715>
        builtin_tid2d(var_0, var_1);
        // NV = wp.static(nv)                                                                     <L 2716>
        // NLEVELS = wp.static(nlevels)                                                           <L 2717>
        // BLOCK_DIM = wp.block_dim()                                                             <L 2718>
        var_4 = builtin_block_dim();
        // for dofid in range(tid, NV, BLOCK_DIM):                                                <L 2721>
        var_6 = wp::range(var_1, var_5, var_4);
        start_for_0:;
            if (iter_cmp(var_6) == 0) goto end_for_0;
            var_7 = wp::iter_next(var_6);
            // x_out[worldid, dofid] = y[worldid, dofid]                                          <L 2722>
            var_8 = wp::address(var_y, var_0, var_7);
            var_9 = wp::load(var_8);
            wp::array_store(var_x_out, var_0, var_7, var_9);
            goto start_for_0;
        end_for_0:;
        // _syncthreads()                                                                         <L 2723>
        _solve_LD_sparse_fused__locals___syncthreads_0();
        // for level in range(NLEVELS):                                                           <L 2726>
        // level_idx = NLEVELS - 1 - level                                                        <L 2727>
        var_12 = wp::sub(var_3, var_11);
        var_13 = wp::sub(var_12, var_10);
        // level_offset = level_offsets[level_idx]                                                <L 2728>
        var_14 = wp::address(var_level_offsets, var_13);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // level_size = level_offsets[level_idx + 1] - level_offset                               <L 2729>
        var_18 = wp::add(var_13, var_17);
        var_19 = wp::address(var_level_offsets, var_18);
        var_21 = wp::load(var_19);
        var_20 = wp::sub(var_21, var_15);
        // for u in range(tid, level_size, BLOCK_DIM):                                            <L 2731>
        var_22 = wp::range(var_1, var_20, var_4);
        start_for_2:;
            if (iter_cmp(var_22) == 0) goto end_for_2;
            var_23 = wp::iter_next(var_22);
            // update = all_updates[level_offset + u]                                             <L 2732>
            var_24 = wp::add(var_15, var_23);
            var_25 = wp::address(var_all_updates, var_24);
            var_27 = wp::load(var_25);
            var_26 = wp::copy(var_27);
            // i, k, Madr_ki = update[0], update[1], update[2]                                    <L 2733>
            var_29 = wp::extract(var_26, var_28);
            var_31 = wp::extract(var_26, var_30);
            var_33 = wp::extract(var_26, var_32);
            // wp.atomic_sub(x_out[worldid], i, L[worldid, 0, Madr_ki] * x_out[worldid, k])       <L 2734>
            var_34 = wp::slice_t(var_0, var_0, var_35);
            var_36 = wp::view(var_x_out, var_34);
            var_38 = wp::address(var_L, var_0, var_37, var_33);
            var_39 = wp::address(var_x_out, var_0, var_31);
            var_41 = wp::load(var_38);
            var_42 = wp::load(var_39);
            var_40 = wp::mul(var_41, var_42);
            var_43 = wp::atomic_sub(var_36, var_29, var_40);
            goto start_for_2;
        end_for_2:;
        // _syncthreads()                                                                         <L 2735>
        _solve_LD_sparse_fused__locals___syncthreads_0();
        // level_idx = NLEVELS - 1 - level                                                        <L 2727>
        var_46 = wp::sub(var_3, var_45);
        var_47 = wp::sub(var_46, var_44);
        // level_offset = level_offsets[level_idx]                                                <L 2728>
        var_48 = wp::address(var_level_offsets, var_47);
        var_50 = wp::load(var_48);
        var_49 = wp::copy(var_50);
        // level_size = level_offsets[level_idx + 1] - level_offset                               <L 2729>
        var_52 = wp::add(var_47, var_51);
        var_53 = wp::address(var_level_offsets, var_52);
        var_55 = wp::load(var_53);
        var_54 = wp::sub(var_55, var_49);
        // for u in range(tid, level_size, BLOCK_DIM):                                            <L 2731>
        var_56 = wp::range(var_1, var_54, var_4);
        start_for_4:;
            if (iter_cmp(var_56) == 0) goto end_for_4;
            var_57 = wp::iter_next(var_56);
            // update = all_updates[level_offset + u]                                             <L 2732>
            var_58 = wp::add(var_49, var_57);
            var_59 = wp::address(var_all_updates, var_58);
            var_61 = wp::load(var_59);
            var_60 = wp::copy(var_61);
            // i, k, Madr_ki = update[0], update[1], update[2]                                    <L 2733>
            var_63 = wp::extract(var_60, var_62);
            var_65 = wp::extract(var_60, var_64);
            var_67 = wp::extract(var_60, var_66);
            // wp.atomic_sub(x_out[worldid], i, L[worldid, 0, Madr_ki] * x_out[worldid, k])       <L 2734>
            var_68 = wp::slice_t(var_0, var_0, var_69);
            var_70 = wp::view(var_x_out, var_68);
            var_72 = wp::address(var_L, var_0, var_71, var_67);
            var_73 = wp::address(var_x_out, var_0, var_65);
            var_75 = wp::load(var_72);
            var_76 = wp::load(var_73);
            var_74 = wp::mul(var_75, var_76);
            var_77 = wp::atomic_sub(var_70, var_63, var_74);
            wp::assign(var_26, var_60);
            wp::assign(var_29, var_63);
            wp::assign(var_31, var_65);
            wp::assign(var_33, var_67);
            goto start_for_4;
        end_for_4:;
        // _syncthreads()                                                                         <L 2735>
        _solve_LD_sparse_fused__locals___syncthreads_0();
        // level_idx = NLEVELS - 1 - level                                                        <L 2727>
        var_80 = wp::sub(var_3, var_79);
        var_81 = wp::sub(var_80, var_78);
        // level_offset = level_offsets[level_idx]                                                <L 2728>
        var_82 = wp::address(var_level_offsets, var_81);
        var_84 = wp::load(var_82);
        var_83 = wp::copy(var_84);
        // level_size = level_offsets[level_idx + 1] - level_offset                               <L 2729>
        var_86 = wp::add(var_81, var_85);
        var_87 = wp::address(var_level_offsets, var_86);
        var_89 = wp::load(var_87);
        var_88 = wp::sub(var_89, var_83);
        // for u in range(tid, level_size, BLOCK_DIM):                                            <L 2731>
        var_90 = wp::range(var_1, var_88, var_4);
        start_for_6:;
            if (iter_cmp(var_90) == 0) goto end_for_6;
            var_91 = wp::iter_next(var_90);
            // update = all_updates[level_offset + u]                                             <L 2732>
            var_92 = wp::add(var_83, var_91);
            var_93 = wp::address(var_all_updates, var_92);
            var_95 = wp::load(var_93);
            var_94 = wp::copy(var_95);
            // i, k, Madr_ki = update[0], update[1], update[2]                                    <L 2733>
            var_97 = wp::extract(var_94, var_96);
            var_99 = wp::extract(var_94, var_98);
            var_101 = wp::extract(var_94, var_100);
            // wp.atomic_sub(x_out[worldid], i, L[worldid, 0, Madr_ki] * x_out[worldid, k])       <L 2734>
            var_102 = wp::slice_t(var_0, var_0, var_103);
            var_104 = wp::view(var_x_out, var_102);
            var_106 = wp::address(var_L, var_0, var_105, var_101);
            var_107 = wp::address(var_x_out, var_0, var_99);
            var_109 = wp::load(var_106);
            var_110 = wp::load(var_107);
            var_108 = wp::mul(var_109, var_110);
            var_111 = wp::atomic_sub(var_104, var_97, var_108);
            wp::assign(var_26, var_94);
            wp::assign(var_29, var_97);
            wp::assign(var_31, var_99);
            wp::assign(var_33, var_101);
            goto start_for_6;
        end_for_6:;
        // _syncthreads()                                                                         <L 2735>
        _solve_LD_sparse_fused__locals___syncthreads_0();
        // level_idx = NLEVELS - 1 - level                                                        <L 2727>
        var_114 = wp::sub(var_3, var_113);
        var_115 = wp::sub(var_114, var_112);
        // level_offset = level_offsets[level_idx]                                                <L 2728>
        var_116 = wp::address(var_level_offsets, var_115);
        var_118 = wp::load(var_116);
        var_117 = wp::copy(var_118);
        // level_size = level_offsets[level_idx + 1] - level_offset                               <L 2729>
        var_120 = wp::add(var_115, var_119);
        var_121 = wp::address(var_level_offsets, var_120);
        var_123 = wp::load(var_121);
        var_122 = wp::sub(var_123, var_117);
        // for u in range(tid, level_size, BLOCK_DIM):                                            <L 2731>
        var_124 = wp::range(var_1, var_122, var_4);
        start_for_8:;
            if (iter_cmp(var_124) == 0) goto end_for_8;
            var_125 = wp::iter_next(var_124);
            // update = all_updates[level_offset + u]                                             <L 2732>
            var_126 = wp::add(var_117, var_125);
            var_127 = wp::address(var_all_updates, var_126);
            var_129 = wp::load(var_127);
            var_128 = wp::copy(var_129);
            // i, k, Madr_ki = update[0], update[1], update[2]                                    <L 2733>
            var_131 = wp::extract(var_128, var_130);
            var_133 = wp::extract(var_128, var_132);
            var_135 = wp::extract(var_128, var_134);
            // wp.atomic_sub(x_out[worldid], i, L[worldid, 0, Madr_ki] * x_out[worldid, k])       <L 2734>
            var_136 = wp::slice_t(var_0, var_0, var_137);
            var_138 = wp::view(var_x_out, var_136);
            var_140 = wp::address(var_L, var_0, var_139, var_135);
            var_141 = wp::address(var_x_out, var_0, var_133);
            var_143 = wp::load(var_140);
            var_144 = wp::load(var_141);
            var_142 = wp::mul(var_143, var_144);
            var_145 = wp::atomic_sub(var_138, var_131, var_142);
            wp::assign(var_26, var_128);
            wp::assign(var_29, var_131);
            wp::assign(var_31, var_133);
            wp::assign(var_33, var_135);
            goto start_for_8;
        end_for_8:;
        // _syncthreads()                                                                         <L 2735>
        _solve_LD_sparse_fused__locals___syncthreads_0();
        // level_idx = NLEVELS - 1 - level                                                        <L 2727>
        var_148 = wp::sub(var_3, var_147);
        var_149 = wp::sub(var_148, var_146);
        // level_offset = level_offsets[level_idx]                                                <L 2728>
        var_150 = wp::address(var_level_offsets, var_149);
        var_152 = wp::load(var_150);
        var_151 = wp::copy(var_152);
        // level_size = level_offsets[level_idx + 1] - level_offset                               <L 2729>
        var_154 = wp::add(var_149, var_153);
        var_155 = wp::address(var_level_offsets, var_154);
        var_157 = wp::load(var_155);
        var_156 = wp::sub(var_157, var_151);
        // for u in range(tid, level_size, BLOCK_DIM):                                            <L 2731>
        var_158 = wp::range(var_1, var_156, var_4);
        start_for_10:;
            if (iter_cmp(var_158) == 0) goto end_for_10;
            var_159 = wp::iter_next(var_158);
            // update = all_updates[level_offset + u]                                             <L 2732>
            var_160 = wp::add(var_151, var_159);
            var_161 = wp::address(var_all_updates, var_160);
            var_163 = wp::load(var_161);
            var_162 = wp::copy(var_163);
            // i, k, Madr_ki = update[0], update[1], update[2]                                    <L 2733>
            var_165 = wp::extract(var_162, var_164);
            var_167 = wp::extract(var_162, var_166);
            var_169 = wp::extract(var_162, var_168);
            // wp.atomic_sub(x_out[worldid], i, L[worldid, 0, Madr_ki] * x_out[worldid, k])       <L 2734>
            var_170 = wp::slice_t(var_0, var_0, var_171);
            var_172 = wp::view(var_x_out, var_170);
            var_174 = wp::address(var_L, var_0, var_173, var_169);
            var_175 = wp::address(var_x_out, var_0, var_167);
            var_177 = wp::load(var_174);
            var_178 = wp::load(var_175);
            var_176 = wp::mul(var_177, var_178);
            var_179 = wp::atomic_sub(var_172, var_165, var_176);
            wp::assign(var_26, var_162);
            wp::assign(var_29, var_165);
            wp::assign(var_31, var_167);
            wp::assign(var_33, var_169);
            goto start_for_10;
        end_for_10:;
        // _syncthreads()                                                                         <L 2735>
        _solve_LD_sparse_fused__locals___syncthreads_0();
        // level_idx = NLEVELS - 1 - level                                                        <L 2727>
        var_182 = wp::sub(var_3, var_181);
        var_183 = wp::sub(var_182, var_180);
        // level_offset = level_offsets[level_idx]                                                <L 2728>
        var_184 = wp::address(var_level_offsets, var_183);
        var_186 = wp::load(var_184);
        var_185 = wp::copy(var_186);
        // level_size = level_offsets[level_idx + 1] - level_offset                               <L 2729>
        var_188 = wp::add(var_183, var_187);
        var_189 = wp::address(var_level_offsets, var_188);
        var_191 = wp::load(var_189);
        var_190 = wp::sub(var_191, var_185);
        // for u in range(tid, level_size, BLOCK_DIM):                                            <L 2731>
        var_192 = wp::range(var_1, var_190, var_4);
        start_for_12:;
            if (iter_cmp(var_192) == 0) goto end_for_12;
            var_193 = wp::iter_next(var_192);
            // update = all_updates[level_offset + u]                                             <L 2732>
            var_194 = wp::add(var_185, var_193);
            var_195 = wp::address(var_all_updates, var_194);
            var_197 = wp::load(var_195);
            var_196 = wp::copy(var_197);
            // i, k, Madr_ki = update[0], update[1], update[2]                                    <L 2733>
            var_199 = wp::extract(var_196, var_198);
            var_201 = wp::extract(var_196, var_200);
            var_203 = wp::extract(var_196, var_202);
            // wp.atomic_sub(x_out[worldid], i, L[worldid, 0, Madr_ki] * x_out[worldid, k])       <L 2734>
            var_204 = wp::slice_t(var_0, var_0, var_205);
            var_206 = wp::view(var_x_out, var_204);
            var_208 = wp::address(var_L, var_0, var_207, var_203);
            var_209 = wp::address(var_x_out, var_0, var_201);
            var_211 = wp::load(var_208);
            var_212 = wp::load(var_209);
            var_210 = wp::mul(var_211, var_212);
            var_213 = wp::atomic_sub(var_206, var_199, var_210);
            wp::assign(var_26, var_196);
            wp::assign(var_29, var_199);
            wp::assign(var_31, var_201);
            wp::assign(var_33, var_203);
            goto start_for_12;
        end_for_12:;
        // _syncthreads()                                                                         <L 2735>
        _solve_LD_sparse_fused__locals___syncthreads_0();
        // level_idx = NLEVELS - 1 - level                                                        <L 2727>
        var_216 = wp::sub(var_3, var_215);
        var_217 = wp::sub(var_216, var_214);
        // level_offset = level_offsets[level_idx]                                                <L 2728>
        var_218 = wp::address(var_level_offsets, var_217);
        var_220 = wp::load(var_218);
        var_219 = wp::copy(var_220);
        // level_size = level_offsets[level_idx + 1] - level_offset                               <L 2729>
        var_222 = wp::add(var_217, var_221);
        var_223 = wp::address(var_level_offsets, var_222);
        var_225 = wp::load(var_223);
        var_224 = wp::sub(var_225, var_219);
        // for u in range(tid, level_size, BLOCK_DIM):                                            <L 2731>
        var_226 = wp::range(var_1, var_224, var_4);
        start_for_14:;
            if (iter_cmp(var_226) == 0) goto end_for_14;
            var_227 = wp::iter_next(var_226);
            // update = all_updates[level_offset + u]                                             <L 2732>
            var_228 = wp::add(var_219, var_227);
            var_229 = wp::address(var_all_updates, var_228);
            var_231 = wp::load(var_229);
            var_230 = wp::copy(var_231);
            // i, k, Madr_ki = update[0], update[1], update[2]                                    <L 2733>
            var_233 = wp::extract(var_230, var_232);
            var_235 = wp::extract(var_230, var_234);
            var_237 = wp::extract(var_230, var_236);
            // wp.atomic_sub(x_out[worldid], i, L[worldid, 0, Madr_ki] * x_out[worldid, k])       <L 2734>
            var_238 = wp::slice_t(var_0, var_0, var_239);
            var_240 = wp::view(var_x_out, var_238);
            var_242 = wp::address(var_L, var_0, var_241, var_237);
            var_243 = wp::address(var_x_out, var_0, var_235);
            var_245 = wp::load(var_242);
            var_246 = wp::load(var_243);
            var_244 = wp::mul(var_245, var_246);
            var_247 = wp::atomic_sub(var_240, var_233, var_244);
            wp::assign(var_26, var_230);
            wp::assign(var_29, var_233);
            wp::assign(var_31, var_235);
            wp::assign(var_33, var_237);
            goto start_for_14;
        end_for_14:;
        // _syncthreads()                                                                         <L 2735>
        _solve_LD_sparse_fused__locals___syncthreads_0();
        // level_idx = NLEVELS - 1 - level                                                        <L 2727>
        var_250 = wp::sub(var_3, var_249);
        var_251 = wp::sub(var_250, var_248);
        // level_offset = level_offsets[level_idx]                                                <L 2728>
        var_252 = wp::address(var_level_offsets, var_251);
        var_254 = wp::load(var_252);
        var_253 = wp::copy(var_254);
        // level_size = level_offsets[level_idx + 1] - level_offset                               <L 2729>
        var_256 = wp::add(var_251, var_255);
        var_257 = wp::address(var_level_offsets, var_256);
        var_259 = wp::load(var_257);
        var_258 = wp::sub(var_259, var_253);
        // for u in range(tid, level_size, BLOCK_DIM):                                            <L 2731>
        var_260 = wp::range(var_1, var_258, var_4);
        start_for_16:;
            if (iter_cmp(var_260) == 0) goto end_for_16;
            var_261 = wp::iter_next(var_260);
            // update = all_updates[level_offset + u]                                             <L 2732>
            var_262 = wp::add(var_253, var_261);
            var_263 = wp::address(var_all_updates, var_262);
            var_265 = wp::load(var_263);
            var_264 = wp::copy(var_265);
            // i, k, Madr_ki = update[0], update[1], update[2]                                    <L 2733>
            var_267 = wp::extract(var_264, var_266);
            var_269 = wp::extract(var_264, var_268);
            var_271 = wp::extract(var_264, var_270);
            // wp.atomic_sub(x_out[worldid], i, L[worldid, 0, Madr_ki] * x_out[worldid, k])       <L 2734>
            var_272 = wp::slice_t(var_0, var_0, var_273);
            var_274 = wp::view(var_x_out, var_272);
            var_276 = wp::address(var_L, var_0, var_275, var_271);
            var_277 = wp::address(var_x_out, var_0, var_269);
            var_279 = wp::load(var_276);
            var_280 = wp::load(var_277);
            var_278 = wp::mul(var_279, var_280);
            var_281 = wp::atomic_sub(var_274, var_267, var_278);
            wp::assign(var_26, var_264);
            wp::assign(var_29, var_267);
            wp::assign(var_31, var_269);
            wp::assign(var_33, var_271);
            goto start_for_16;
        end_for_16:;
        // _syncthreads()                                                                         <L 2735>
        _solve_LD_sparse_fused__locals___syncthreads_0();
        // level_idx = NLEVELS - 1 - level                                                        <L 2727>
        var_284 = wp::sub(var_3, var_283);
        var_285 = wp::sub(var_284, var_282);
        // level_offset = level_offsets[level_idx]                                                <L 2728>
        var_286 = wp::address(var_level_offsets, var_285);
        var_288 = wp::load(var_286);
        var_287 = wp::copy(var_288);
        // level_size = level_offsets[level_idx + 1] - level_offset                               <L 2729>
        var_290 = wp::add(var_285, var_289);
        var_291 = wp::address(var_level_offsets, var_290);
        var_293 = wp::load(var_291);
        var_292 = wp::sub(var_293, var_287);
        // for u in range(tid, level_size, BLOCK_DIM):                                            <L 2731>
        var_294 = wp::range(var_1, var_292, var_4);
        start_for_18:;
            if (iter_cmp(var_294) == 0) goto end_for_18;
            var_295 = wp::iter_next(var_294);
            // update = all_updates[level_offset + u]                                             <L 2732>
            var_296 = wp::add(var_287, var_295);
            var_297 = wp::address(var_all_updates, var_296);
            var_299 = wp::load(var_297);
            var_298 = wp::copy(var_299);
            // i, k, Madr_ki = update[0], update[1], update[2]                                    <L 2733>
            var_301 = wp::extract(var_298, var_300);
            var_303 = wp::extract(var_298, var_302);
            var_305 = wp::extract(var_298, var_304);
            // wp.atomic_sub(x_out[worldid], i, L[worldid, 0, Madr_ki] * x_out[worldid, k])       <L 2734>
            var_306 = wp::slice_t(var_0, var_0, var_307);
            var_308 = wp::view(var_x_out, var_306);
            var_310 = wp::address(var_L, var_0, var_309, var_305);
            var_311 = wp::address(var_x_out, var_0, var_303);
            var_313 = wp::load(var_310);
            var_314 = wp::load(var_311);
            var_312 = wp::mul(var_313, var_314);
            var_315 = wp::atomic_sub(var_308, var_301, var_312);
            wp::assign(var_26, var_298);
            wp::assign(var_29, var_301);
            wp::assign(var_31, var_303);
            wp::assign(var_33, var_305);
            goto start_for_18;
        end_for_18:;
        // _syncthreads()                                                                         <L 2735>
        _solve_LD_sparse_fused__locals___syncthreads_0();
        // level_idx = NLEVELS - 1 - level                                                        <L 2727>
        var_318 = wp::sub(var_3, var_317);
        var_319 = wp::sub(var_318, var_316);
        // level_offset = level_offsets[level_idx]                                                <L 2728>
        var_320 = wp::address(var_level_offsets, var_319);
        var_322 = wp::load(var_320);
        var_321 = wp::copy(var_322);
        // level_size = level_offsets[level_idx + 1] - level_offset                               <L 2729>
        var_324 = wp::add(var_319, var_323);
        var_325 = wp::address(var_level_offsets, var_324);
        var_327 = wp::load(var_325);
        var_326 = wp::sub(var_327, var_321);
        // for u in range(tid, level_size, BLOCK_DIM):                                            <L 2731>
        var_328 = wp::range(var_1, var_326, var_4);
        start_for_20:;
            if (iter_cmp(var_328) == 0) goto end_for_20;
            var_329 = wp::iter_next(var_328);
            // update = all_updates[level_offset + u]                                             <L 2732>
            var_330 = wp::add(var_321, var_329);
            var_331 = wp::address(var_all_updates, var_330);
            var_333 = wp::load(var_331);
            var_332 = wp::copy(var_333);
            // i, k, Madr_ki = update[0], update[1], update[2]                                    <L 2733>
            var_335 = wp::extract(var_332, var_334);
            var_337 = wp::extract(var_332, var_336);
            var_339 = wp::extract(var_332, var_338);
            // wp.atomic_sub(x_out[worldid], i, L[worldid, 0, Madr_ki] * x_out[worldid, k])       <L 2734>
            var_340 = wp::slice_t(var_0, var_0, var_341);
            var_342 = wp::view(var_x_out, var_340);
            var_344 = wp::address(var_L, var_0, var_343, var_339);
            var_345 = wp::address(var_x_out, var_0, var_337);
            var_347 = wp::load(var_344);
            var_348 = wp::load(var_345);
            var_346 = wp::mul(var_347, var_348);
            var_349 = wp::atomic_sub(var_342, var_335, var_346);
            wp::assign(var_26, var_332);
            wp::assign(var_29, var_335);
            wp::assign(var_31, var_337);
            wp::assign(var_33, var_339);
            goto start_for_20;
        end_for_20:;
        // _syncthreads()                                                                         <L 2735>
        _solve_LD_sparse_fused__locals___syncthreads_0();
        // level_idx = NLEVELS - 1 - level                                                        <L 2727>
        var_352 = wp::sub(var_3, var_351);
        var_353 = wp::sub(var_352, var_350);
        // level_offset = level_offsets[level_idx]                                                <L 2728>
        var_354 = wp::address(var_level_offsets, var_353);
        var_356 = wp::load(var_354);
        var_355 = wp::copy(var_356);
        // level_size = level_offsets[level_idx + 1] - level_offset                               <L 2729>
        var_358 = wp::add(var_353, var_357);
        var_359 = wp::address(var_level_offsets, var_358);
        var_361 = wp::load(var_359);
        var_360 = wp::sub(var_361, var_355);
        // for u in range(tid, level_size, BLOCK_DIM):                                            <L 2731>
        var_362 = wp::range(var_1, var_360, var_4);
        start_for_22:;
            if (iter_cmp(var_362) == 0) goto end_for_22;
            var_363 = wp::iter_next(var_362);
            // update = all_updates[level_offset + u]                                             <L 2732>
            var_364 = wp::add(var_355, var_363);
            var_365 = wp::address(var_all_updates, var_364);
            var_367 = wp::load(var_365);
            var_366 = wp::copy(var_367);
            // i, k, Madr_ki = update[0], update[1], update[2]                                    <L 2733>
            var_369 = wp::extract(var_366, var_368);
            var_371 = wp::extract(var_366, var_370);
            var_373 = wp::extract(var_366, var_372);
            // wp.atomic_sub(x_out[worldid], i, L[worldid, 0, Madr_ki] * x_out[worldid, k])       <L 2734>
            var_374 = wp::slice_t(var_0, var_0, var_375);
            var_376 = wp::view(var_x_out, var_374);
            var_378 = wp::address(var_L, var_0, var_377, var_373);
            var_379 = wp::address(var_x_out, var_0, var_371);
            var_381 = wp::load(var_378);
            var_382 = wp::load(var_379);
            var_380 = wp::mul(var_381, var_382);
            var_383 = wp::atomic_sub(var_376, var_369, var_380);
            wp::assign(var_26, var_366);
            wp::assign(var_29, var_369);
            wp::assign(var_31, var_371);
            wp::assign(var_33, var_373);
            goto start_for_22;
        end_for_22:;
        // _syncthreads()                                                                         <L 2735>
        _solve_LD_sparse_fused__locals___syncthreads_0();
        // level_idx = NLEVELS - 1 - level                                                        <L 2727>
        var_386 = wp::sub(var_3, var_385);
        var_387 = wp::sub(var_386, var_384);
        // level_offset = level_offsets[level_idx]                                                <L 2728>
        var_388 = wp::address(var_level_offsets, var_387);
        var_390 = wp::load(var_388);
        var_389 = wp::copy(var_390);
        // level_size = level_offsets[level_idx + 1] - level_offset                               <L 2729>
        var_392 = wp::add(var_387, var_391);
        var_393 = wp::address(var_level_offsets, var_392);
        var_395 = wp::load(var_393);
        var_394 = wp::sub(var_395, var_389);
        // for u in range(tid, level_size, BLOCK_DIM):                                            <L 2731>
        var_396 = wp::range(var_1, var_394, var_4);
        start_for_24:;
            if (iter_cmp(var_396) == 0) goto end_for_24;
            var_397 = wp::iter_next(var_396);
            // update = all_updates[level_offset + u]                                             <L 2732>
            var_398 = wp::add(var_389, var_397);
            var_399 = wp::address(var_all_updates, var_398);
            var_401 = wp::load(var_399);
            var_400 = wp::copy(var_401);
            // i, k, Madr_ki = update[0], update[1], update[2]                                    <L 2733>
            var_403 = wp::extract(var_400, var_402);
            var_405 = wp::extract(var_400, var_404);
            var_407 = wp::extract(var_400, var_406);
            // wp.atomic_sub(x_out[worldid], i, L[worldid, 0, Madr_ki] * x_out[worldid, k])       <L 2734>
            var_408 = wp::slice_t(var_0, var_0, var_409);
            var_410 = wp::view(var_x_out, var_408);
            var_412 = wp::address(var_L, var_0, var_411, var_407);
            var_413 = wp::address(var_x_out, var_0, var_405);
            var_415 = wp::load(var_412);
            var_416 = wp::load(var_413);
            var_414 = wp::mul(var_415, var_416);
            var_417 = wp::atomic_sub(var_410, var_403, var_414);
            wp::assign(var_26, var_400);
            wp::assign(var_29, var_403);
            wp::assign(var_31, var_405);
            wp::assign(var_33, var_407);
            goto start_for_24;
        end_for_24:;
        // _syncthreads()                                                                         <L 2735>
        _solve_LD_sparse_fused__locals___syncthreads_0();
        // level_idx = NLEVELS - 1 - level                                                        <L 2727>
        var_420 = wp::sub(var_3, var_419);
        var_421 = wp::sub(var_420, var_418);
        // level_offset = level_offsets[level_idx]                                                <L 2728>
        var_422 = wp::address(var_level_offsets, var_421);
        var_424 = wp::load(var_422);
        var_423 = wp::copy(var_424);
        // level_size = level_offsets[level_idx + 1] - level_offset                               <L 2729>
        var_426 = wp::add(var_421, var_425);
        var_427 = wp::address(var_level_offsets, var_426);
        var_429 = wp::load(var_427);
        var_428 = wp::sub(var_429, var_423);
        // for u in range(tid, level_size, BLOCK_DIM):                                            <L 2731>
        var_430 = wp::range(var_1, var_428, var_4);
        start_for_26:;
            if (iter_cmp(var_430) == 0) goto end_for_26;
            var_431 = wp::iter_next(var_430);
            // update = all_updates[level_offset + u]                                             <L 2732>
            var_432 = wp::add(var_423, var_431);
            var_433 = wp::address(var_all_updates, var_432);
            var_435 = wp::load(var_433);
            var_434 = wp::copy(var_435);
            // i, k, Madr_ki = update[0], update[1], update[2]                                    <L 2733>
            var_437 = wp::extract(var_434, var_436);
            var_439 = wp::extract(var_434, var_438);
            var_441 = wp::extract(var_434, var_440);
            // wp.atomic_sub(x_out[worldid], i, L[worldid, 0, Madr_ki] * x_out[worldid, k])       <L 2734>
            var_442 = wp::slice_t(var_0, var_0, var_443);
            var_444 = wp::view(var_x_out, var_442);
            var_446 = wp::address(var_L, var_0, var_445, var_441);
            var_447 = wp::address(var_x_out, var_0, var_439);
            var_449 = wp::load(var_446);
            var_450 = wp::load(var_447);
            var_448 = wp::mul(var_449, var_450);
            var_451 = wp::atomic_sub(var_444, var_437, var_448);
            wp::assign(var_26, var_434);
            wp::assign(var_29, var_437);
            wp::assign(var_31, var_439);
            wp::assign(var_33, var_441);
            goto start_for_26;
        end_for_26:;
        // _syncthreads()                                                                         <L 2735>
        _solve_LD_sparse_fused__locals___syncthreads_0();
        // level_idx = NLEVELS - 1 - level                                                        <L 2727>
        var_454 = wp::sub(var_3, var_453);
        var_455 = wp::sub(var_454, var_452);
        // level_offset = level_offsets[level_idx]                                                <L 2728>
        var_456 = wp::address(var_level_offsets, var_455);
        var_458 = wp::load(var_456);
        var_457 = wp::copy(var_458);
        // level_size = level_offsets[level_idx + 1] - level_offset                               <L 2729>
        var_460 = wp::add(var_455, var_459);
        var_461 = wp::address(var_level_offsets, var_460);
        var_463 = wp::load(var_461);
        var_462 = wp::sub(var_463, var_457);
        // for u in range(tid, level_size, BLOCK_DIM):                                            <L 2731>
        var_464 = wp::range(var_1, var_462, var_4);
        start_for_28:;
            if (iter_cmp(var_464) == 0) goto end_for_28;
            var_465 = wp::iter_next(var_464);
            // update = all_updates[level_offset + u]                                             <L 2732>
            var_466 = wp::add(var_457, var_465);
            var_467 = wp::address(var_all_updates, var_466);
            var_469 = wp::load(var_467);
            var_468 = wp::copy(var_469);
            // i, k, Madr_ki = update[0], update[1], update[2]                                    <L 2733>
            var_471 = wp::extract(var_468, var_470);
            var_473 = wp::extract(var_468, var_472);
            var_475 = wp::extract(var_468, var_474);
            // wp.atomic_sub(x_out[worldid], i, L[worldid, 0, Madr_ki] * x_out[worldid, k])       <L 2734>
            var_476 = wp::slice_t(var_0, var_0, var_477);
            var_478 = wp::view(var_x_out, var_476);
            var_480 = wp::address(var_L, var_0, var_479, var_475);
            var_481 = wp::address(var_x_out, var_0, var_473);
            var_483 = wp::load(var_480);
            var_484 = wp::load(var_481);
            var_482 = wp::mul(var_483, var_484);
            var_485 = wp::atomic_sub(var_478, var_471, var_482);
            wp::assign(var_26, var_468);
            wp::assign(var_29, var_471);
            wp::assign(var_31, var_473);
            wp::assign(var_33, var_475);
            goto start_for_28;
        end_for_28:;
        // _syncthreads()                                                                         <L 2735>
        _solve_LD_sparse_fused__locals___syncthreads_0();
        // level_idx = NLEVELS - 1 - level                                                        <L 2727>
        var_488 = wp::sub(var_3, var_487);
        var_489 = wp::sub(var_488, var_486);
        // level_offset = level_offsets[level_idx]                                                <L 2728>
        var_490 = wp::address(var_level_offsets, var_489);
        var_492 = wp::load(var_490);
        var_491 = wp::copy(var_492);
        // level_size = level_offsets[level_idx + 1] - level_offset                               <L 2729>
        var_494 = wp::add(var_489, var_493);
        var_495 = wp::address(var_level_offsets, var_494);
        var_497 = wp::load(var_495);
        var_496 = wp::sub(var_497, var_491);
        // for u in range(tid, level_size, BLOCK_DIM):                                            <L 2731>
        var_498 = wp::range(var_1, var_496, var_4);
        start_for_30:;
            if (iter_cmp(var_498) == 0) goto end_for_30;
            var_499 = wp::iter_next(var_498);
            // update = all_updates[level_offset + u]                                             <L 2732>
            var_500 = wp::add(var_491, var_499);
            var_501 = wp::address(var_all_updates, var_500);
            var_503 = wp::load(var_501);
            var_502 = wp::copy(var_503);
            // i, k, Madr_ki = update[0], update[1], update[2]                                    <L 2733>
            var_505 = wp::extract(var_502, var_504);
            var_507 = wp::extract(var_502, var_506);
            var_509 = wp::extract(var_502, var_508);
            // wp.atomic_sub(x_out[worldid], i, L[worldid, 0, Madr_ki] * x_out[worldid, k])       <L 2734>
            var_510 = wp::slice_t(var_0, var_0, var_511);
            var_512 = wp::view(var_x_out, var_510);
            var_514 = wp::address(var_L, var_0, var_513, var_509);
            var_515 = wp::address(var_x_out, var_0, var_507);
            var_517 = wp::load(var_514);
            var_518 = wp::load(var_515);
            var_516 = wp::mul(var_517, var_518);
            var_519 = wp::atomic_sub(var_512, var_505, var_516);
            wp::assign(var_26, var_502);
            wp::assign(var_29, var_505);
            wp::assign(var_31, var_507);
            wp::assign(var_33, var_509);
            goto start_for_30;
        end_for_30:;
        // _syncthreads()                                                                         <L 2735>
        _solve_LD_sparse_fused__locals___syncthreads_0();
        // for dofid in range(tid, NV, BLOCK_DIM):                                                <L 2738>
        var_521 = wp::range(var_1, var_520, var_4);
        start_for_32:;
            if (iter_cmp(var_521) == 0) goto end_for_32;
            var_522 = wp::iter_next(var_521);
            // x_out[worldid, dofid] *= D[worldid, dofid]                                         <L 2739>
            var_523 = wp::address(var_D, var_0, var_522);
            var_524 = wp::address(var_x_out, var_0, var_522);
            var_525 = wp::address(var_D, var_0, var_522);
            var_527 = wp::load(var_524);
            var_528 = wp::load(var_525);
            var_526 = wp::mul(var_527, var_528);
            wp::array_store(var_x_out, var_0, var_522, var_526);
            goto start_for_32;
        end_for_32:;
        // _syncthreads()                                                                         <L 2740>
        _solve_LD_sparse_fused__locals___syncthreads_0();
        // for level in range(NLEVELS):                                                           <L 2743>
        // level_idx = level                                                                      <L 2744>
        var_530 = wp::copy(var_529);
        // level_offset = level_offsets[level_idx]                                                <L 2745>
        var_531 = wp::address(var_level_offsets, var_530);
        var_533 = wp::load(var_531);
        var_532 = wp::copy(var_533);
        // level_size = level_offsets[level_idx + 1] - level_offset                               <L 2746>
        var_535 = wp::add(var_530, var_534);
        var_536 = wp::address(var_level_offsets, var_535);
        var_538 = wp::load(var_536);
        var_537 = wp::sub(var_538, var_532);
        // for u in range(tid, level_size, BLOCK_DIM):                                            <L 2748>
        var_539 = wp::range(var_1, var_537, var_4);
        start_for_34:;
            if (iter_cmp(var_539) == 0) goto end_for_34;
            var_540 = wp::iter_next(var_539);
            // update = all_updates[level_offset + u]                                             <L 2749>
            var_541 = wp::add(var_532, var_540);
            var_542 = wp::address(var_all_updates, var_541);
            var_544 = wp::load(var_542);
            var_543 = wp::copy(var_544);
            // i, k, Madr_ki = update[0], update[1], update[2]                                    <L 2750>
            var_546 = wp::extract(var_543, var_545);
            var_548 = wp::extract(var_543, var_547);
            var_550 = wp::extract(var_543, var_549);
            // wp.atomic_sub(x_out[worldid], k, L[worldid, 0, Madr_ki] * x_out[worldid, i])       <L 2751>
            var_551 = wp::slice_t(var_0, var_0, var_552);
            var_553 = wp::view(var_x_out, var_551);
            var_555 = wp::address(var_L, var_0, var_554, var_550);
            var_556 = wp::address(var_x_out, var_0, var_546);
            var_558 = wp::load(var_555);
            var_559 = wp::load(var_556);
            var_557 = wp::mul(var_558, var_559);
            var_560 = wp::atomic_sub(var_553, var_548, var_557);
            wp::assign(var_26, var_543);
            wp::assign(var_29, var_546);
            wp::assign(var_31, var_548);
            wp::assign(var_33, var_550);
            goto start_for_34;
        end_for_34:;
        // _syncthreads()                                                                         <L 2752>
        _solve_LD_sparse_fused__locals___syncthreads_0();
        // level_idx = level                                                                      <L 2744>
        var_562 = wp::copy(var_561);
        // level_offset = level_offsets[level_idx]                                                <L 2745>
        var_563 = wp::address(var_level_offsets, var_562);
        var_565 = wp::load(var_563);
        var_564 = wp::copy(var_565);
        // level_size = level_offsets[level_idx + 1] - level_offset                               <L 2746>
        var_567 = wp::add(var_562, var_566);
        var_568 = wp::address(var_level_offsets, var_567);
        var_570 = wp::load(var_568);
        var_569 = wp::sub(var_570, var_564);
        // for u in range(tid, level_size, BLOCK_DIM):                                            <L 2748>
        var_571 = wp::range(var_1, var_569, var_4);
        start_for_36:;
            if (iter_cmp(var_571) == 0) goto end_for_36;
            var_572 = wp::iter_next(var_571);
            // update = all_updates[level_offset + u]                                             <L 2749>
            var_573 = wp::add(var_564, var_572);
            var_574 = wp::address(var_all_updates, var_573);
            var_576 = wp::load(var_574);
            var_575 = wp::copy(var_576);
            // i, k, Madr_ki = update[0], update[1], update[2]                                    <L 2750>
            var_578 = wp::extract(var_575, var_577);
            var_580 = wp::extract(var_575, var_579);
            var_582 = wp::extract(var_575, var_581);
            // wp.atomic_sub(x_out[worldid], k, L[worldid, 0, Madr_ki] * x_out[worldid, i])       <L 2751>
            var_583 = wp::slice_t(var_0, var_0, var_584);
            var_585 = wp::view(var_x_out, var_583);
            var_587 = wp::address(var_L, var_0, var_586, var_582);
            var_588 = wp::address(var_x_out, var_0, var_578);
            var_590 = wp::load(var_587);
            var_591 = wp::load(var_588);
            var_589 = wp::mul(var_590, var_591);
            var_592 = wp::atomic_sub(var_585, var_580, var_589);
            wp::assign(var_26, var_575);
            wp::assign(var_29, var_578);
            wp::assign(var_31, var_580);
            wp::assign(var_33, var_582);
            goto start_for_36;
        end_for_36:;
        // _syncthreads()                                                                         <L 2752>
        _solve_LD_sparse_fused__locals___syncthreads_0();
        // level_idx = level                                                                      <L 2744>
        var_594 = wp::copy(var_593);
        // level_offset = level_offsets[level_idx]                                                <L 2745>
        var_595 = wp::address(var_level_offsets, var_594);
        var_597 = wp::load(var_595);
        var_596 = wp::copy(var_597);
        // level_size = level_offsets[level_idx + 1] - level_offset                               <L 2746>
        var_599 = wp::add(var_594, var_598);
        var_600 = wp::address(var_level_offsets, var_599);
        var_602 = wp::load(var_600);
        var_601 = wp::sub(var_602, var_596);
        // for u in range(tid, level_size, BLOCK_DIM):                                            <L 2748>
        var_603 = wp::range(var_1, var_601, var_4);
        start_for_38:;
            if (iter_cmp(var_603) == 0) goto end_for_38;
            var_604 = wp::iter_next(var_603);
            // update = all_updates[level_offset + u]                                             <L 2749>
            var_605 = wp::add(var_596, var_604);
            var_606 = wp::address(var_all_updates, var_605);
            var_608 = wp::load(var_606);
            var_607 = wp::copy(var_608);
            // i, k, Madr_ki = update[0], update[1], update[2]                                    <L 2750>
            var_610 = wp::extract(var_607, var_609);
            var_612 = wp::extract(var_607, var_611);
            var_614 = wp::extract(var_607, var_613);
            // wp.atomic_sub(x_out[worldid], k, L[worldid, 0, Madr_ki] * x_out[worldid, i])       <L 2751>
            var_615 = wp::slice_t(var_0, var_0, var_616);
            var_617 = wp::view(var_x_out, var_615);
            var_619 = wp::address(var_L, var_0, var_618, var_614);
            var_620 = wp::address(var_x_out, var_0, var_610);
            var_622 = wp::load(var_619);
            var_623 = wp::load(var_620);
            var_621 = wp::mul(var_622, var_623);
            var_624 = wp::atomic_sub(var_617, var_612, var_621);
            wp::assign(var_26, var_607);
            wp::assign(var_29, var_610);
            wp::assign(var_31, var_612);
            wp::assign(var_33, var_614);
            goto start_for_38;
        end_for_38:;
        // _syncthreads()                                                                         <L 2752>
        _solve_LD_sparse_fused__locals___syncthreads_0();
        // level_idx = level                                                                      <L 2744>
        var_626 = wp::copy(var_625);
        // level_offset = level_offsets[level_idx]                                                <L 2745>
        var_627 = wp::address(var_level_offsets, var_626);
        var_629 = wp::load(var_627);
        var_628 = wp::copy(var_629);
        // level_size = level_offsets[level_idx + 1] - level_offset                               <L 2746>
        var_631 = wp::add(var_626, var_630);
        var_632 = wp::address(var_level_offsets, var_631);
        var_634 = wp::load(var_632);
        var_633 = wp::sub(var_634, var_628);
        // for u in range(tid, level_size, BLOCK_DIM):                                            <L 2748>
        var_635 = wp::range(var_1, var_633, var_4);
        start_for_40:;
            if (iter_cmp(var_635) == 0) goto end_for_40;
            var_636 = wp::iter_next(var_635);
            // update = all_updates[level_offset + u]                                             <L 2749>
            var_637 = wp::add(var_628, var_636);
            var_638 = wp::address(var_all_updates, var_637);
            var_640 = wp::load(var_638);
            var_639 = wp::copy(var_640);
            // i, k, Madr_ki = update[0], update[1], update[2]                                    <L 2750>
            var_642 = wp::extract(var_639, var_641);
            var_644 = wp::extract(var_639, var_643);
            var_646 = wp::extract(var_639, var_645);
            // wp.atomic_sub(x_out[worldid], k, L[worldid, 0, Madr_ki] * x_out[worldid, i])       <L 2751>
            var_647 = wp::slice_t(var_0, var_0, var_648);
            var_649 = wp::view(var_x_out, var_647);
            var_651 = wp::address(var_L, var_0, var_650, var_646);
            var_652 = wp::address(var_x_out, var_0, var_642);
            var_654 = wp::load(var_651);
            var_655 = wp::load(var_652);
            var_653 = wp::mul(var_654, var_655);
            var_656 = wp::atomic_sub(var_649, var_644, var_653);
            wp::assign(var_26, var_639);
            wp::assign(var_29, var_642);
            wp::assign(var_31, var_644);
            wp::assign(var_33, var_646);
            goto start_for_40;
        end_for_40:;
        // _syncthreads()                                                                         <L 2752>
        _solve_LD_sparse_fused__locals___syncthreads_0();
        // level_idx = level                                                                      <L 2744>
        var_658 = wp::copy(var_657);
        // level_offset = level_offsets[level_idx]                                                <L 2745>
        var_659 = wp::address(var_level_offsets, var_658);
        var_661 = wp::load(var_659);
        var_660 = wp::copy(var_661);
        // level_size = level_offsets[level_idx + 1] - level_offset                               <L 2746>
        var_663 = wp::add(var_658, var_662);
        var_664 = wp::address(var_level_offsets, var_663);
        var_666 = wp::load(var_664);
        var_665 = wp::sub(var_666, var_660);
        // for u in range(tid, level_size, BLOCK_DIM):                                            <L 2748>
        var_667 = wp::range(var_1, var_665, var_4);
        start_for_42:;
            if (iter_cmp(var_667) == 0) goto end_for_42;
            var_668 = wp::iter_next(var_667);
            // update = all_updates[level_offset + u]                                             <L 2749>
            var_669 = wp::add(var_660, var_668);
            var_670 = wp::address(var_all_updates, var_669);
            var_672 = wp::load(var_670);
            var_671 = wp::copy(var_672);
            // i, k, Madr_ki = update[0], update[1], update[2]                                    <L 2750>
            var_674 = wp::extract(var_671, var_673);
            var_676 = wp::extract(var_671, var_675);
            var_678 = wp::extract(var_671, var_677);
            // wp.atomic_sub(x_out[worldid], k, L[worldid, 0, Madr_ki] * x_out[worldid, i])       <L 2751>
            var_679 = wp::slice_t(var_0, var_0, var_680);
            var_681 = wp::view(var_x_out, var_679);
            var_683 = wp::address(var_L, var_0, var_682, var_678);
            var_684 = wp::address(var_x_out, var_0, var_674);
            var_686 = wp::load(var_683);
            var_687 = wp::load(var_684);
            var_685 = wp::mul(var_686, var_687);
            var_688 = wp::atomic_sub(var_681, var_676, var_685);
            wp::assign(var_26, var_671);
            wp::assign(var_29, var_674);
            wp::assign(var_31, var_676);
            wp::assign(var_33, var_678);
            goto start_for_42;
        end_for_42:;
        // _syncthreads()                                                                         <L 2752>
        _solve_LD_sparse_fused__locals___syncthreads_0();
        // level_idx = level                                                                      <L 2744>
        var_690 = wp::copy(var_689);
        // level_offset = level_offsets[level_idx]                                                <L 2745>
        var_691 = wp::address(var_level_offsets, var_690);
        var_693 = wp::load(var_691);
        var_692 = wp::copy(var_693);
        // level_size = level_offsets[level_idx + 1] - level_offset                               <L 2746>
        var_695 = wp::add(var_690, var_694);
        var_696 = wp::address(var_level_offsets, var_695);
        var_698 = wp::load(var_696);
        var_697 = wp::sub(var_698, var_692);
        // for u in range(tid, level_size, BLOCK_DIM):                                            <L 2748>
        var_699 = wp::range(var_1, var_697, var_4);
        start_for_44:;
            if (iter_cmp(var_699) == 0) goto end_for_44;
            var_700 = wp::iter_next(var_699);
            // update = all_updates[level_offset + u]                                             <L 2749>
            var_701 = wp::add(var_692, var_700);
            var_702 = wp::address(var_all_updates, var_701);
            var_704 = wp::load(var_702);
            var_703 = wp::copy(var_704);
            // i, k, Madr_ki = update[0], update[1], update[2]                                    <L 2750>
            var_706 = wp::extract(var_703, var_705);
            var_708 = wp::extract(var_703, var_707);
            var_710 = wp::extract(var_703, var_709);
            // wp.atomic_sub(x_out[worldid], k, L[worldid, 0, Madr_ki] * x_out[worldid, i])       <L 2751>
            var_711 = wp::slice_t(var_0, var_0, var_712);
            var_713 = wp::view(var_x_out, var_711);
            var_715 = wp::address(var_L, var_0, var_714, var_710);
            var_716 = wp::address(var_x_out, var_0, var_706);
            var_718 = wp::load(var_715);
            var_719 = wp::load(var_716);
            var_717 = wp::mul(var_718, var_719);
            var_720 = wp::atomic_sub(var_713, var_708, var_717);
            wp::assign(var_26, var_703);
            wp::assign(var_29, var_706);
            wp::assign(var_31, var_708);
            wp::assign(var_33, var_710);
            goto start_for_44;
        end_for_44:;
        // _syncthreads()                                                                         <L 2752>
        _solve_LD_sparse_fused__locals___syncthreads_0();
        // level_idx = level                                                                      <L 2744>
        var_722 = wp::copy(var_721);
        // level_offset = level_offsets[level_idx]                                                <L 2745>
        var_723 = wp::address(var_level_offsets, var_722);
        var_725 = wp::load(var_723);
        var_724 = wp::copy(var_725);
        // level_size = level_offsets[level_idx + 1] - level_offset                               <L 2746>
        var_727 = wp::add(var_722, var_726);
        var_728 = wp::address(var_level_offsets, var_727);
        var_730 = wp::load(var_728);
        var_729 = wp::sub(var_730, var_724);
        // for u in range(tid, level_size, BLOCK_DIM):                                            <L 2748>
        var_731 = wp::range(var_1, var_729, var_4);
        start_for_46:;
            if (iter_cmp(var_731) == 0) goto end_for_46;
            var_732 = wp::iter_next(var_731);
            // update = all_updates[level_offset + u]                                             <L 2749>
            var_733 = wp::add(var_724, var_732);
            var_734 = wp::address(var_all_updates, var_733);
            var_736 = wp::load(var_734);
            var_735 = wp::copy(var_736);
            // i, k, Madr_ki = update[0], update[1], update[2]                                    <L 2750>
            var_738 = wp::extract(var_735, var_737);
            var_740 = wp::extract(var_735, var_739);
            var_742 = wp::extract(var_735, var_741);
            // wp.atomic_sub(x_out[worldid], k, L[worldid, 0, Madr_ki] * x_out[worldid, i])       <L 2751>
            var_743 = wp::slice_t(var_0, var_0, var_744);
            var_745 = wp::view(var_x_out, var_743);
            var_747 = wp::address(var_L, var_0, var_746, var_742);
            var_748 = wp::address(var_x_out, var_0, var_738);
            var_750 = wp::load(var_747);
            var_751 = wp::load(var_748);
            var_749 = wp::mul(var_750, var_751);
            var_752 = wp::atomic_sub(var_745, var_740, var_749);
            wp::assign(var_26, var_735);
            wp::assign(var_29, var_738);
            wp::assign(var_31, var_740);
            wp::assign(var_33, var_742);
            goto start_for_46;
        end_for_46:;
        // _syncthreads()                                                                         <L 2752>
        _solve_LD_sparse_fused__locals___syncthreads_0();
        // level_idx = level                                                                      <L 2744>
        var_754 = wp::copy(var_753);
        // level_offset = level_offsets[level_idx]                                                <L 2745>
        var_755 = wp::address(var_level_offsets, var_754);
        var_757 = wp::load(var_755);
        var_756 = wp::copy(var_757);
        // level_size = level_offsets[level_idx + 1] - level_offset                               <L 2746>
        var_759 = wp::add(var_754, var_758);
        var_760 = wp::address(var_level_offsets, var_759);
        var_762 = wp::load(var_760);
        var_761 = wp::sub(var_762, var_756);
        // for u in range(tid, level_size, BLOCK_DIM):                                            <L 2748>
        var_763 = wp::range(var_1, var_761, var_4);
        start_for_48:;
            if (iter_cmp(var_763) == 0) goto end_for_48;
            var_764 = wp::iter_next(var_763);
            // update = all_updates[level_offset + u]                                             <L 2749>
            var_765 = wp::add(var_756, var_764);
            var_766 = wp::address(var_all_updates, var_765);
            var_768 = wp::load(var_766);
            var_767 = wp::copy(var_768);
            // i, k, Madr_ki = update[0], update[1], update[2]                                    <L 2750>
            var_770 = wp::extract(var_767, var_769);
            var_772 = wp::extract(var_767, var_771);
            var_774 = wp::extract(var_767, var_773);
            // wp.atomic_sub(x_out[worldid], k, L[worldid, 0, Madr_ki] * x_out[worldid, i])       <L 2751>
            var_775 = wp::slice_t(var_0, var_0, var_776);
            var_777 = wp::view(var_x_out, var_775);
            var_779 = wp::address(var_L, var_0, var_778, var_774);
            var_780 = wp::address(var_x_out, var_0, var_770);
            var_782 = wp::load(var_779);
            var_783 = wp::load(var_780);
            var_781 = wp::mul(var_782, var_783);
            var_784 = wp::atomic_sub(var_777, var_772, var_781);
            wp::assign(var_26, var_767);
            wp::assign(var_29, var_770);
            wp::assign(var_31, var_772);
            wp::assign(var_33, var_774);
            goto start_for_48;
        end_for_48:;
        // _syncthreads()                                                                         <L 2752>
        _solve_LD_sparse_fused__locals___syncthreads_0();
        // level_idx = level                                                                      <L 2744>
        var_786 = wp::copy(var_785);
        // level_offset = level_offsets[level_idx]                                                <L 2745>
        var_787 = wp::address(var_level_offsets, var_786);
        var_789 = wp::load(var_787);
        var_788 = wp::copy(var_789);
        // level_size = level_offsets[level_idx + 1] - level_offset                               <L 2746>
        var_791 = wp::add(var_786, var_790);
        var_792 = wp::address(var_level_offsets, var_791);
        var_794 = wp::load(var_792);
        var_793 = wp::sub(var_794, var_788);
        // for u in range(tid, level_size, BLOCK_DIM):                                            <L 2748>
        var_795 = wp::range(var_1, var_793, var_4);
        start_for_50:;
            if (iter_cmp(var_795) == 0) goto end_for_50;
            var_796 = wp::iter_next(var_795);
            // update = all_updates[level_offset + u]                                             <L 2749>
            var_797 = wp::add(var_788, var_796);
            var_798 = wp::address(var_all_updates, var_797);
            var_800 = wp::load(var_798);
            var_799 = wp::copy(var_800);
            // i, k, Madr_ki = update[0], update[1], update[2]                                    <L 2750>
            var_802 = wp::extract(var_799, var_801);
            var_804 = wp::extract(var_799, var_803);
            var_806 = wp::extract(var_799, var_805);
            // wp.atomic_sub(x_out[worldid], k, L[worldid, 0, Madr_ki] * x_out[worldid, i])       <L 2751>
            var_807 = wp::slice_t(var_0, var_0, var_808);
            var_809 = wp::view(var_x_out, var_807);
            var_811 = wp::address(var_L, var_0, var_810, var_806);
            var_812 = wp::address(var_x_out, var_0, var_802);
            var_814 = wp::load(var_811);
            var_815 = wp::load(var_812);
            var_813 = wp::mul(var_814, var_815);
            var_816 = wp::atomic_sub(var_809, var_804, var_813);
            wp::assign(var_26, var_799);
            wp::assign(var_29, var_802);
            wp::assign(var_31, var_804);
            wp::assign(var_33, var_806);
            goto start_for_50;
        end_for_50:;
        // _syncthreads()                                                                         <L 2752>
        _solve_LD_sparse_fused__locals___syncthreads_0();
        // level_idx = level                                                                      <L 2744>
        var_818 = wp::copy(var_817);
        // level_offset = level_offsets[level_idx]                                                <L 2745>
        var_819 = wp::address(var_level_offsets, var_818);
        var_821 = wp::load(var_819);
        var_820 = wp::copy(var_821);
        // level_size = level_offsets[level_idx + 1] - level_offset                               <L 2746>
        var_823 = wp::add(var_818, var_822);
        var_824 = wp::address(var_level_offsets, var_823);
        var_826 = wp::load(var_824);
        var_825 = wp::sub(var_826, var_820);
        // for u in range(tid, level_size, BLOCK_DIM):                                            <L 2748>
        var_827 = wp::range(var_1, var_825, var_4);
        start_for_52:;
            if (iter_cmp(var_827) == 0) goto end_for_52;
            var_828 = wp::iter_next(var_827);
            // update = all_updates[level_offset + u]                                             <L 2749>
            var_829 = wp::add(var_820, var_828);
            var_830 = wp::address(var_all_updates, var_829);
            var_832 = wp::load(var_830);
            var_831 = wp::copy(var_832);
            // i, k, Madr_ki = update[0], update[1], update[2]                                    <L 2750>
            var_834 = wp::extract(var_831, var_833);
            var_836 = wp::extract(var_831, var_835);
            var_838 = wp::extract(var_831, var_837);
            // wp.atomic_sub(x_out[worldid], k, L[worldid, 0, Madr_ki] * x_out[worldid, i])       <L 2751>
            var_839 = wp::slice_t(var_0, var_0, var_840);
            var_841 = wp::view(var_x_out, var_839);
            var_843 = wp::address(var_L, var_0, var_842, var_838);
            var_844 = wp::address(var_x_out, var_0, var_834);
            var_846 = wp::load(var_843);
            var_847 = wp::load(var_844);
            var_845 = wp::mul(var_846, var_847);
            var_848 = wp::atomic_sub(var_841, var_836, var_845);
            wp::assign(var_26, var_831);
            wp::assign(var_29, var_834);
            wp::assign(var_31, var_836);
            wp::assign(var_33, var_838);
            goto start_for_52;
        end_for_52:;
        // _syncthreads()                                                                         <L 2752>
        _solve_LD_sparse_fused__locals___syncthreads_0();
        // level_idx = level                                                                      <L 2744>
        var_850 = wp::copy(var_849);
        // level_offset = level_offsets[level_idx]                                                <L 2745>
        var_851 = wp::address(var_level_offsets, var_850);
        var_853 = wp::load(var_851);
        var_852 = wp::copy(var_853);
        // level_size = level_offsets[level_idx + 1] - level_offset                               <L 2746>
        var_855 = wp::add(var_850, var_854);
        var_856 = wp::address(var_level_offsets, var_855);
        var_858 = wp::load(var_856);
        var_857 = wp::sub(var_858, var_852);
        // for u in range(tid, level_size, BLOCK_DIM):                                            <L 2748>
        var_859 = wp::range(var_1, var_857, var_4);
        start_for_54:;
            if (iter_cmp(var_859) == 0) goto end_for_54;
            var_860 = wp::iter_next(var_859);
            // update = all_updates[level_offset + u]                                             <L 2749>
            var_861 = wp::add(var_852, var_860);
            var_862 = wp::address(var_all_updates, var_861);
            var_864 = wp::load(var_862);
            var_863 = wp::copy(var_864);
            // i, k, Madr_ki = update[0], update[1], update[2]                                    <L 2750>
            var_866 = wp::extract(var_863, var_865);
            var_868 = wp::extract(var_863, var_867);
            var_870 = wp::extract(var_863, var_869);
            // wp.atomic_sub(x_out[worldid], k, L[worldid, 0, Madr_ki] * x_out[worldid, i])       <L 2751>
            var_871 = wp::slice_t(var_0, var_0, var_872);
            var_873 = wp::view(var_x_out, var_871);
            var_875 = wp::address(var_L, var_0, var_874, var_870);
            var_876 = wp::address(var_x_out, var_0, var_866);
            var_878 = wp::load(var_875);
            var_879 = wp::load(var_876);
            var_877 = wp::mul(var_878, var_879);
            var_880 = wp::atomic_sub(var_873, var_868, var_877);
            wp::assign(var_26, var_863);
            wp::assign(var_29, var_866);
            wp::assign(var_31, var_868);
            wp::assign(var_33, var_870);
            goto start_for_54;
        end_for_54:;
        // _syncthreads()                                                                         <L 2752>
        _solve_LD_sparse_fused__locals___syncthreads_0();
        // level_idx = level                                                                      <L 2744>
        var_882 = wp::copy(var_881);
        // level_offset = level_offsets[level_idx]                                                <L 2745>
        var_883 = wp::address(var_level_offsets, var_882);
        var_885 = wp::load(var_883);
        var_884 = wp::copy(var_885);
        // level_size = level_offsets[level_idx + 1] - level_offset                               <L 2746>
        var_887 = wp::add(var_882, var_886);
        var_888 = wp::address(var_level_offsets, var_887);
        var_890 = wp::load(var_888);
        var_889 = wp::sub(var_890, var_884);
        // for u in range(tid, level_size, BLOCK_DIM):                                            <L 2748>
        var_891 = wp::range(var_1, var_889, var_4);
        start_for_56:;
            if (iter_cmp(var_891) == 0) goto end_for_56;
            var_892 = wp::iter_next(var_891);
            // update = all_updates[level_offset + u]                                             <L 2749>
            var_893 = wp::add(var_884, var_892);
            var_894 = wp::address(var_all_updates, var_893);
            var_896 = wp::load(var_894);
            var_895 = wp::copy(var_896);
            // i, k, Madr_ki = update[0], update[1], update[2]                                    <L 2750>
            var_898 = wp::extract(var_895, var_897);
            var_900 = wp::extract(var_895, var_899);
            var_902 = wp::extract(var_895, var_901);
            // wp.atomic_sub(x_out[worldid], k, L[worldid, 0, Madr_ki] * x_out[worldid, i])       <L 2751>
            var_903 = wp::slice_t(var_0, var_0, var_904);
            var_905 = wp::view(var_x_out, var_903);
            var_907 = wp::address(var_L, var_0, var_906, var_902);
            var_908 = wp::address(var_x_out, var_0, var_898);
            var_910 = wp::load(var_907);
            var_911 = wp::load(var_908);
            var_909 = wp::mul(var_910, var_911);
            var_912 = wp::atomic_sub(var_905, var_900, var_909);
            wp::assign(var_26, var_895);
            wp::assign(var_29, var_898);
            wp::assign(var_31, var_900);
            wp::assign(var_33, var_902);
            goto start_for_56;
        end_for_56:;
        // _syncthreads()                                                                         <L 2752>
        _solve_LD_sparse_fused__locals___syncthreads_0();
        // level_idx = level                                                                      <L 2744>
        var_914 = wp::copy(var_913);
        // level_offset = level_offsets[level_idx]                                                <L 2745>
        var_915 = wp::address(var_level_offsets, var_914);
        var_917 = wp::load(var_915);
        var_916 = wp::copy(var_917);
        // level_size = level_offsets[level_idx + 1] - level_offset                               <L 2746>
        var_919 = wp::add(var_914, var_918);
        var_920 = wp::address(var_level_offsets, var_919);
        var_922 = wp::load(var_920);
        var_921 = wp::sub(var_922, var_916);
        // for u in range(tid, level_size, BLOCK_DIM):                                            <L 2748>
        var_923 = wp::range(var_1, var_921, var_4);
        start_for_58:;
            if (iter_cmp(var_923) == 0) goto end_for_58;
            var_924 = wp::iter_next(var_923);
            // update = all_updates[level_offset + u]                                             <L 2749>
            var_925 = wp::add(var_916, var_924);
            var_926 = wp::address(var_all_updates, var_925);
            var_928 = wp::load(var_926);
            var_927 = wp::copy(var_928);
            // i, k, Madr_ki = update[0], update[1], update[2]                                    <L 2750>
            var_930 = wp::extract(var_927, var_929);
            var_932 = wp::extract(var_927, var_931);
            var_934 = wp::extract(var_927, var_933);
            // wp.atomic_sub(x_out[worldid], k, L[worldid, 0, Madr_ki] * x_out[worldid, i])       <L 2751>
            var_935 = wp::slice_t(var_0, var_0, var_936);
            var_937 = wp::view(var_x_out, var_935);
            var_939 = wp::address(var_L, var_0, var_938, var_934);
            var_940 = wp::address(var_x_out, var_0, var_930);
            var_942 = wp::load(var_939);
            var_943 = wp::load(var_940);
            var_941 = wp::mul(var_942, var_943);
            var_944 = wp::atomic_sub(var_937, var_932, var_941);
            wp::assign(var_26, var_927);
            wp::assign(var_29, var_930);
            wp::assign(var_31, var_932);
            wp::assign(var_33, var_934);
            goto start_for_58;
        end_for_58:;
        // _syncthreads()                                                                         <L 2752>
        _solve_LD_sparse_fused__locals___syncthreads_0();
        // level_idx = level                                                                      <L 2744>
        var_946 = wp::copy(var_945);
        // level_offset = level_offsets[level_idx]                                                <L 2745>
        var_947 = wp::address(var_level_offsets, var_946);
        var_949 = wp::load(var_947);
        var_948 = wp::copy(var_949);
        // level_size = level_offsets[level_idx + 1] - level_offset                               <L 2746>
        var_951 = wp::add(var_946, var_950);
        var_952 = wp::address(var_level_offsets, var_951);
        var_954 = wp::load(var_952);
        var_953 = wp::sub(var_954, var_948);
        // for u in range(tid, level_size, BLOCK_DIM):                                            <L 2748>
        var_955 = wp::range(var_1, var_953, var_4);
        start_for_60:;
            if (iter_cmp(var_955) == 0) goto end_for_60;
            var_956 = wp::iter_next(var_955);
            // update = all_updates[level_offset + u]                                             <L 2749>
            var_957 = wp::add(var_948, var_956);
            var_958 = wp::address(var_all_updates, var_957);
            var_960 = wp::load(var_958);
            var_959 = wp::copy(var_960);
            // i, k, Madr_ki = update[0], update[1], update[2]                                    <L 2750>
            var_962 = wp::extract(var_959, var_961);
            var_964 = wp::extract(var_959, var_963);
            var_966 = wp::extract(var_959, var_965);
            // wp.atomic_sub(x_out[worldid], k, L[worldid, 0, Madr_ki] * x_out[worldid, i])       <L 2751>
            var_967 = wp::slice_t(var_0, var_0, var_968);
            var_969 = wp::view(var_x_out, var_967);
            var_971 = wp::address(var_L, var_0, var_970, var_966);
            var_972 = wp::address(var_x_out, var_0, var_962);
            var_974 = wp::load(var_971);
            var_975 = wp::load(var_972);
            var_973 = wp::mul(var_974, var_975);
            var_976 = wp::atomic_sub(var_969, var_964, var_973);
            wp::assign(var_26, var_959);
            wp::assign(var_29, var_962);
            wp::assign(var_31, var_964);
            wp::assign(var_33, var_966);
            goto start_for_60;
        end_for_60:;
        // _syncthreads()                                                                         <L 2752>
        _solve_LD_sparse_fused__locals___syncthreads_0();
        // level_idx = level                                                                      <L 2744>
        var_978 = wp::copy(var_977);
        // level_offset = level_offsets[level_idx]                                                <L 2745>
        var_979 = wp::address(var_level_offsets, var_978);
        var_981 = wp::load(var_979);
        var_980 = wp::copy(var_981);
        // level_size = level_offsets[level_idx + 1] - level_offset                               <L 2746>
        var_983 = wp::add(var_978, var_982);
        var_984 = wp::address(var_level_offsets, var_983);
        var_986 = wp::load(var_984);
        var_985 = wp::sub(var_986, var_980);
        // for u in range(tid, level_size, BLOCK_DIM):                                            <L 2748>
        var_987 = wp::range(var_1, var_985, var_4);
        start_for_62:;
            if (iter_cmp(var_987) == 0) goto end_for_62;
            var_988 = wp::iter_next(var_987);
            // update = all_updates[level_offset + u]                                             <L 2749>
            var_989 = wp::add(var_980, var_988);
            var_990 = wp::address(var_all_updates, var_989);
            var_992 = wp::load(var_990);
            var_991 = wp::copy(var_992);
            // i, k, Madr_ki = update[0], update[1], update[2]                                    <L 2750>
            var_994 = wp::extract(var_991, var_993);
            var_996 = wp::extract(var_991, var_995);
            var_998 = wp::extract(var_991, var_997);
            // wp.atomic_sub(x_out[worldid], k, L[worldid, 0, Madr_ki] * x_out[worldid, i])       <L 2751>
            var_999 = wp::slice_t(var_0, var_0, var_1000);
            var_1001 = wp::view(var_x_out, var_999);
            var_1003 = wp::address(var_L, var_0, var_1002, var_998);
            var_1004 = wp::address(var_x_out, var_0, var_994);
            var_1006 = wp::load(var_1003);
            var_1007 = wp::load(var_1004);
            var_1005 = wp::mul(var_1006, var_1007);
            var_1008 = wp::atomic_sub(var_1001, var_996, var_1005);
            wp::assign(var_26, var_991);
            wp::assign(var_29, var_994);
            wp::assign(var_31, var_996);
            wp::assign(var_33, var_998);
            goto start_for_62;
        end_for_62:;
        // _syncthreads()                                                                         <L 2752>
        _solve_LD_sparse_fused__locals___syncthreads_0();
    }
}

