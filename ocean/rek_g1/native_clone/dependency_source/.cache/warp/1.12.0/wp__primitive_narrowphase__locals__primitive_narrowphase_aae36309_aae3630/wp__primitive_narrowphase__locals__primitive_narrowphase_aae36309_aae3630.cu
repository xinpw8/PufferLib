
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


struct Geom_3242f8a8
{
    wp::vec_t<3, wp::float32> pos;
    wp::mat_t<3, 3, wp::float32> rot;
    wp::vec_t<3, wp::float32> normal;
    wp::vec_t<3, wp::float32> size;
    wp::float32 margin;
    wp::mat_t<6, 3, wp::float32> hfprism;
    wp::int32 vertadr;
    wp::int32 vertnum;
    wp::array_t<wp::vec_t<3, wp::float32>> vert;
    wp::int32 graphadr;
    wp::array_t<wp::int32> graph;
    wp::int32 mesh_polynum;
    wp::int32 mesh_polyadr;
    wp::array_t<wp::vec_t<3, wp::float32>> mesh_polynormal;
    wp::array_t<wp::int32> mesh_polyvertadr;
    wp::array_t<wp::int32> mesh_polyvertnum;
    wp::array_t<wp::int32> mesh_polyvert;
    wp::array_t<wp::int32> mesh_polymapadr;
    wp::array_t<wp::int32> mesh_polymapnum;
    wp::array_t<wp::int32> mesh_polymap;
    wp::int32 index;


    Geom_3242f8a8() = default;
    CUDA_CALLABLE Geom_3242f8a8(wp::vec_t<3, wp::float32> const& pos,
    wp::mat_t<3, 3, wp::float32> const& rot = {},
    wp::vec_t<3, wp::float32> const& normal = {},
    wp::vec_t<3, wp::float32> const& size = {},
    wp::float32 const& margin = {},
    wp::mat_t<6, 3, wp::float32> const& hfprism = {},
    wp::int32 const& vertadr = {},
    wp::int32 const& vertnum = {},
    wp::array_t<wp::vec_t<3, wp::float32>> const& vert = {},
    wp::int32 const& graphadr = {},
    wp::array_t<wp::int32> const& graph = {},
    wp::int32 const& mesh_polynum = {},
    wp::int32 const& mesh_polyadr = {},
    wp::array_t<wp::vec_t<3, wp::float32>> const& mesh_polynormal = {},
    wp::array_t<wp::int32> const& mesh_polyvertadr = {},
    wp::array_t<wp::int32> const& mesh_polyvertnum = {},
    wp::array_t<wp::int32> const& mesh_polyvert = {},
    wp::array_t<wp::int32> const& mesh_polymapadr = {},
    wp::array_t<wp::int32> const& mesh_polymapnum = {},
    wp::array_t<wp::int32> const& mesh_polymap = {},
    wp::int32 const& index = {})
        : pos{pos}
        , rot{rot}
        , normal{normal}
        , size{size}
        , margin{margin}
        , hfprism{hfprism}
        , vertadr{vertadr}
        , vertnum{vertnum}
        , vert{vert}
        , graphadr{graphadr}
        , graph{graph}
        , mesh_polynum{mesh_polynum}
        , mesh_polyadr{mesh_polyadr}
        , mesh_polynormal{mesh_polynormal}
        , mesh_polyvertadr{mesh_polyvertadr}
        , mesh_polyvertnum{mesh_polyvertnum}
        , mesh_polyvert{mesh_polyvert}
        , mesh_polymapadr{mesh_polymapadr}
        , mesh_polymapnum{mesh_polymapnum}
        , mesh_polymap{mesh_polymap}
        , index{index}

    {
    }

    CUDA_CALLABLE Geom_3242f8a8& operator += (const Geom_3242f8a8& rhs)
    {    pos += rhs.pos;
    rot += rhs.rot;
    normal += rhs.normal;
    size += rhs.size;
    margin += rhs.margin;
    hfprism += rhs.hfprism;
    vertadr += rhs.vertadr;
    vertnum += rhs.vertnum;
    graphadr += rhs.graphadr;
    mesh_polynum += rhs.mesh_polynum;
    mesh_polyadr += rhs.mesh_polyadr;
    index += rhs.index;

        return *this;}

};

static CUDA_CALLABLE void adj_Geom_3242f8a8(wp::vec_t<3, wp::float32> const&,
    wp::mat_t<3, 3, wp::float32> const&,
    wp::vec_t<3, wp::float32> const&,
    wp::vec_t<3, wp::float32> const&,
    wp::float32 const&,
    wp::mat_t<6, 3, wp::float32> const&,
    wp::int32 const&,
    wp::int32 const&,
    wp::array_t<wp::vec_t<3, wp::float32>> const&,
    wp::int32 const&,
    wp::array_t<wp::int32> const&,
    wp::int32 const&,
    wp::int32 const&,
    wp::array_t<wp::vec_t<3, wp::float32>> const&,
    wp::array_t<wp::int32> const&,
    wp::array_t<wp::int32> const&,
    wp::array_t<wp::int32> const&,
    wp::array_t<wp::int32> const&,
    wp::array_t<wp::int32> const&,
    wp::array_t<wp::int32> const&,
    wp::int32 const&,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_rot,
    wp::vec_t<3, wp::float32> & adj_normal,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::float32 & adj_margin,
    wp::mat_t<6, 3, wp::float32> & adj_hfprism,
    wp::int32 & adj_vertadr,
    wp::int32 & adj_vertnum,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_vert,
    wp::int32 & adj_graphadr,
    wp::array_t<wp::int32> & adj_graph,
    wp::int32 & adj_mesh_polynum,
    wp::int32 & adj_mesh_polyadr,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_mesh_polynormal,
    wp::array_t<wp::int32> & adj_mesh_polyvertadr,
    wp::array_t<wp::int32> & adj_mesh_polyvertnum,
    wp::array_t<wp::int32> & adj_mesh_polyvert,
    wp::array_t<wp::int32> & adj_mesh_polymapadr,
    wp::array_t<wp::int32> & adj_mesh_polymapnum,
    wp::array_t<wp::int32> & adj_mesh_polymap,
    wp::int32 & adj_index,
    Geom_3242f8a8 & adj_ret)
{
    adj_pos += adj_ret.pos;
    adj_rot += adj_ret.rot;
    adj_normal += adj_ret.normal;
    adj_size += adj_ret.size;
    adj_margin += adj_ret.margin;
    adj_hfprism += adj_ret.hfprism;
    adj_vertadr += adj_ret.vertadr;
    adj_vertnum += adj_ret.vertnum;
    adj_vert = adj_ret.vert;
    adj_graphadr += adj_ret.graphadr;
    adj_graph = adj_ret.graph;
    adj_mesh_polynum += adj_ret.mesh_polynum;
    adj_mesh_polyadr += adj_ret.mesh_polyadr;
    adj_mesh_polynormal = adj_ret.mesh_polynormal;
    adj_mesh_polyvertadr = adj_ret.mesh_polyvertadr;
    adj_mesh_polyvertnum = adj_ret.mesh_polyvertnum;
    adj_mesh_polyvert = adj_ret.mesh_polyvert;
    adj_mesh_polymapadr = adj_ret.mesh_polymapadr;
    adj_mesh_polymapnum = adj_ret.mesh_polymapnum;
    adj_mesh_polymap = adj_ret.mesh_polymap;
    adj_index += adj_ret.index;
}

// Required when compiling adjoints.
CUDA_CALLABLE Geom_3242f8a8 add(const Geom_3242f8a8& a, const Geom_3242f8a8& b)
{
    return Geom_3242f8a8();
}

CUDA_CALLABLE void adj_atomic_add(Geom_3242f8a8* p, Geom_3242f8a8 t)
{
    wp::adj_atomic_add(&p->pos, t.pos);
    wp::adj_atomic_add(&p->rot, t.rot);
    wp::adj_atomic_add(&p->normal, t.normal);
    wp::adj_atomic_add(&p->size, t.size);
    wp::adj_atomic_add(&p->margin, t.margin);
    wp::adj_atomic_add(&p->hfprism, t.hfprism);
    wp::adj_atomic_add(&p->vertadr, t.vertadr);
    wp::adj_atomic_add(&p->vertnum, t.vertnum);
    wp::adj_atomic_add(&p->vert, t.vert);
    wp::adj_atomic_add(&p->graphadr, t.graphadr);
    wp::adj_atomic_add(&p->graph, t.graph);
    wp::adj_atomic_add(&p->mesh_polynum, t.mesh_polynum);
    wp::adj_atomic_add(&p->mesh_polyadr, t.mesh_polyadr);
    wp::adj_atomic_add(&p->mesh_polynormal, t.mesh_polynormal);
    wp::adj_atomic_add(&p->mesh_polyvertadr, t.mesh_polyvertadr);
    wp::adj_atomic_add(&p->mesh_polyvertnum, t.mesh_polyvertnum);
    wp::adj_atomic_add(&p->mesh_polyvert, t.mesh_polyvert);
    wp::adj_atomic_add(&p->mesh_polymapadr, t.mesh_polymapadr);
    wp::adj_atomic_add(&p->mesh_polymapnum, t.mesh_polymapnum);
    wp::adj_atomic_add(&p->mesh_polymap, t.mesh_polymap);
    wp::adj_atomic_add(&p->index, t.index);
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_core.py:235
static CUDA_CALLABLE void contact_params_0(
    wp::array_t<wp::int32> var_geom_condim,
    wp::array_t<wp::int32> var_geom_priority,
    wp::array_t<wp::float32> var_geom_solmix,
    wp::array_t<wp::vec_t<2, wp::float32>> var_geom_solref,
    wp::array_t<wp::vec_t<5, wp::float32>> var_geom_solimp,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_friction,
    wp::array_t<wp::float32> var_geom_margin,
    wp::array_t<wp::float32> var_geom_gap,
    wp::array_t<wp::int32> var_pair_dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_pair_solref,
    wp::array_t<wp::vec_t<2, wp::float32>> var_pair_solreffriction,
    wp::array_t<wp::vec_t<5, wp::float32>> var_pair_solimp,
    wp::array_t<wp::float32> var_pair_margin,
    wp::array_t<wp::float32> var_pair_gap,
    wp::array_t<wp::vec_t<5, wp::float32>> var_pair_friction,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pair_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pairid_in,
    wp::int32 var_cid,
    wp::int32 var_worldid,
    wp::vec_t<2, wp::int32> & ret_0,
    wp::float32 & ret_1,
    wp::float32 & ret_2,
    wp::int32 & ret_3,
    wp::vec_t<5, wp::float32> & ret_4,
    wp::vec_t<2, wp::float32> & ret_5,
    wp::vec_t<2, wp::float32> & ret_6,
    wp::vec_t<5, wp::float32> & ret_7)
{
    //---------
    // primal vars
    wp::vec_t<2, wp::int32>* var_0;
    wp::vec_t<2, wp::int32> var_1;
    wp::vec_t<2, wp::int32> var_2;
    wp::vec_t<2, wp::int32>* var_3;
    const wp::int32 var_4 = 0;
    wp::int32 var_5;
    wp::vec_t<2, wp::int32> var_6;
    const wp::int32 var_7 = 1;
    const wp::int32 var_8 = -1;
    bool var_9;
    wp::shape_t* var_10;
    const wp::int32 var_11 = 0;
    wp::int32 var_12;
    wp::shape_t var_13;
    wp::int32 var_14;
    wp::float32* var_15;
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
    wp::int32* var_26;
    wp::int32 var_27;
    wp::int32 var_28;
    wp::shape_t* var_29;
    const wp::int32 var_30 = 0;
    wp::int32 var_31;
    wp::shape_t var_32;
    wp::int32 var_33;
    wp::vec_t<5, wp::float32>* var_34;
    wp::vec_t<5, wp::float32> var_35;
    wp::vec_t<5, wp::float32> var_36;
    wp::shape_t* var_37;
    const wp::int32 var_38 = 0;
    wp::int32 var_39;
    wp::shape_t var_40;
    wp::int32 var_41;
    wp::vec_t<2, wp::float32>* var_42;
    wp::vec_t<2, wp::float32> var_43;
    wp::vec_t<2, wp::float32> var_44;
    wp::shape_t* var_45;
    const wp::int32 var_46 = 0;
    wp::int32 var_47;
    wp::shape_t var_48;
    wp::int32 var_49;
    wp::vec_t<2, wp::float32>* var_50;
    wp::vec_t<2, wp::float32> var_51;
    wp::vec_t<2, wp::float32> var_52;
    wp::shape_t* var_53;
    const wp::int32 var_54 = 0;
    wp::int32 var_55;
    wp::shape_t var_56;
    wp::int32 var_57;
    wp::vec_t<5, wp::float32>* var_58;
    wp::vec_t<5, wp::float32> var_59;
    wp::vec_t<5, wp::float32> var_60;
    const wp::int32 var_61 = 0;
    wp::int32 var_62;
    const wp::int32 var_63 = 1;
    wp::int32 var_64;
    wp::shape_t* var_65;
    const wp::int32 var_66 = 0;
    wp::int32 var_67;
    wp::shape_t var_68;
    wp::int32 var_69;
    wp::shape_t* var_70;
    const wp::int32 var_71 = 0;
    wp::int32 var_72;
    wp::shape_t var_73;
    wp::int32 var_74;
    wp::shape_t* var_75;
    const wp::int32 var_76 = 0;
    wp::int32 var_77;
    wp::shape_t var_78;
    wp::int32 var_79;
    wp::shape_t* var_80;
    const wp::int32 var_81 = 0;
    wp::int32 var_82;
    wp::shape_t var_83;
    wp::int32 var_84;
    wp::shape_t* var_85;
    const wp::int32 var_86 = 0;
    wp::int32 var_87;
    wp::shape_t var_88;
    wp::int32 var_89;
    wp::shape_t* var_90;
    const wp::int32 var_91 = 0;
    wp::int32 var_92;
    wp::shape_t var_93;
    wp::int32 var_94;
    wp::float32* var_95;
    wp::float32 var_96;
    wp::float32 var_97;
    wp::float32* var_98;
    wp::float32 var_99;
    wp::float32 var_100;
    wp::int32* var_101;
    wp::int32 var_102;
    wp::int32 var_103;
    wp::int32* var_104;
    wp::int32 var_105;
    wp::int32 var_106;
    wp::int32* var_107;
    wp::int32 var_108;
    wp::int32 var_109;
    wp::int32* var_110;
    wp::int32 var_111;
    wp::int32 var_112;
    bool var_113;
    const wp::float32 var_114 = 1.0;
    wp::int32 var_115;
    wp::vec_t<3, wp::float32>* var_116;
    wp::vec_t<3, wp::float32> var_117;
    wp::vec_t<3, wp::float32> var_118;
    wp::int32 var_119;
    bool var_120;
    const wp::float32 var_121 = 0.0;
    wp::int32 var_122;
    wp::vec_t<3, wp::float32>* var_123;
    wp::vec_t<3, wp::float32> var_124;
    wp::vec_t<3, wp::float32> var_125;
    wp::int32 var_126;
    wp::float32 var_127;
    wp::vec_t<3, wp::float32> var_128;
    wp::float32 var_129;
    wp::float32 var_130;
    const wp::float32 var_131 = 1e-15;
    bool var_132;
    bool var_133;
    bool var_134;
    const wp::float32 var_135 = 0.5;
    wp::float32 var_136;
    bool var_137;
    bool var_138;
    bool var_139;
    const wp::float32 var_140 = 0.0;
    wp::float32 var_141;
    bool var_142;
    bool var_143;
    bool var_144;
    const wp::float32 var_145 = 1.0;
    wp::float32 var_146;
    wp::int32 var_147;
    wp::vec_t<3, wp::float32>* var_148;
    wp::vec_t<3, wp::float32>* var_149;
    wp::vec_t<3, wp::float32> var_150;
    wp::vec_t<3, wp::float32> var_151;
    wp::vec_t<3, wp::float32> var_152;
    wp::int32 var_153;
    wp::float32 var_154;
    wp::vec_t<3, wp::float32> var_155;
    wp::int32 var_156;
    wp::float32 var_157;
    wp::vec_t<3, wp::float32> var_158;
    const wp::int32 var_159 = 0;
    wp::float32 var_160;
    const wp::int32 var_161 = 0;
    wp::float32 var_162;
    const wp::int32 var_163 = 1;
    wp::float32 var_164;
    const wp::int32 var_165 = 2;
    wp::float32 var_166;
    const wp::int32 var_167 = 2;
    wp::float32 var_168;
    wp::vec_t<5, wp::float32> var_169;
    wp::vec_t<2, wp::float32>* var_170;
    const wp::int32 var_171 = 0;
    wp::float32 var_172;
    wp::vec_t<2, wp::float32> var_173;
    const wp::float32 var_174 = 0.0;
    bool var_175;
    wp::vec_t<2, wp::float32>* var_176;
    const wp::int32 var_177 = 0;
    wp::float32 var_178;
    wp::vec_t<2, wp::float32> var_179;
    const wp::float32 var_180 = 0.0;
    bool var_181;
    bool var_182;
    wp::vec_t<2, wp::float32>* var_183;
    wp::vec_t<2, wp::float32> var_184;
    wp::vec_t<2, wp::float32> var_185;
    const wp::float32 var_186 = 1.0;
    wp::float32 var_187;
    wp::vec_t<2, wp::float32>* var_188;
    wp::vec_t<2, wp::float32> var_189;
    wp::vec_t<2, wp::float32> var_190;
    wp::vec_t<2, wp::float32> var_191;
    wp::vec_t<2, wp::float32> var_192;
    wp::vec_t<2, wp::float32>* var_193;
    wp::vec_t<2, wp::float32>* var_194;
    wp::vec_t<2, wp::float32> var_195;
    wp::vec_t<2, wp::float32> var_196;
    wp::vec_t<2, wp::float32> var_197;
    wp::vec_t<2, wp::float32> var_198;
    const wp::float32 var_199 = 0.0;
    const wp::float32 var_200 = 0.0;
    wp::vec_t<2, wp::float32> var_201;
    wp::vec_t<5, wp::float32>* var_202;
    wp::vec_t<5, wp::float32> var_203;
    wp::vec_t<5, wp::float32> var_204;
    const wp::float32 var_205 = 1.0;
    wp::float32 var_206;
    wp::vec_t<5, wp::float32>* var_207;
    wp::vec_t<5, wp::float32> var_208;
    wp::vec_t<5, wp::float32> var_209;
    wp::vec_t<5, wp::float32> var_210;
    wp::float32* var_211;
    wp::float32* var_212;
    wp::float32 var_213;
    wp::float32 var_214;
    wp::float32 var_215;
    wp::float32* var_216;
    wp::float32* var_217;
    wp::float32 var_218;
    wp::float32 var_219;
    wp::float32 var_220;
    wp::float32 var_221;
    wp::float32 var_222;
    wp::int32 var_223;
    wp::vec_t<5, wp::float32> var_224;
    wp::vec_t<2, wp::float32> var_225;
    wp::vec_t<2, wp::float32> var_226;
    wp::vec_t<5, wp::float32> var_227;
    const wp::float32 var_228 = 1e-05;
    const wp::int32 var_229 = 0;
    wp::float32 var_230;
    wp::float32 var_231;
    const wp::int32 var_232 = 1;
    wp::float32 var_233;
    wp::float32 var_234;
    const wp::int32 var_235 = 2;
    wp::float32 var_236;
    wp::float32 var_237;
    const wp::int32 var_238 = 3;
    wp::float32 var_239;
    wp::float32 var_240;
    const wp::int32 var_241 = 4;
    wp::float32 var_242;
    wp::float32 var_243;
    wp::vec_t<5, wp::float32> var_244;
    //---------
    // forward
    // def contact_params(                                                                    <L 236>
    // geoms = collision_pair_in[cid]                                                         <L 264>
    var_0 = wp::address(var_collision_pair_in, var_cid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // pairid = collision_pairid_in[cid][0]                                                   <L 265>
    var_3 = wp::address(var_collision_pairid_in, var_cid);
    var_6 = wp::load(var_3);
    var_5 = wp::extract(var_6, var_4);
    // if pairid > -1:                                                                        <L 270>
    var_9 = (var_5 > var_8);
    if (var_9) {
        // margin = pair_margin[worldid % pair_margin.shape[0], pairid]                       <L 271>
        var_10 = &(var_pair_margin.shape);
        var_13 = wp::load(var_10);
        var_12 = wp::extract(var_13, var_11);
        var_14 = wp::mod(var_worldid, var_12);
        var_15 = wp::address(var_pair_margin, var_14, var_5);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // gap = pair_gap[worldid % pair_gap.shape[0], pairid]                                <L 272>
        var_18 = &(var_pair_gap.shape);
        var_21 = wp::load(var_18);
        var_20 = wp::extract(var_21, var_19);
        var_22 = wp::mod(var_worldid, var_20);
        var_23 = wp::address(var_pair_gap, var_22, var_5);
        var_25 = wp::load(var_23);
        var_24 = wp::copy(var_25);
        // condim = pair_dim[pairid]                                                          <L 273>
        var_26 = wp::address(var_pair_dim, var_5);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // friction = pair_friction[worldid % pair_friction.shape[0], pairid]                 <L 274>
        var_29 = &(var_pair_friction.shape);
        var_32 = wp::load(var_29);
        var_31 = wp::extract(var_32, var_30);
        var_33 = wp::mod(var_worldid, var_31);
        var_34 = wp::address(var_pair_friction, var_33, var_5);
        var_36 = wp::load(var_34);
        var_35 = wp::copy(var_36);
        // solref = pair_solref[worldid % pair_solref.shape[0], pairid]                       <L 275>
        var_37 = &(var_pair_solref.shape);
        var_40 = wp::load(var_37);
        var_39 = wp::extract(var_40, var_38);
        var_41 = wp::mod(var_worldid, var_39);
        var_42 = wp::address(var_pair_solref, var_41, var_5);
        var_44 = wp::load(var_42);
        var_43 = wp::copy(var_44);
        // solreffriction = pair_solreffriction[worldid % pair_solreffriction.shape[0], pairid]       <L 276>
        var_45 = &(var_pair_solreffriction.shape);
        var_48 = wp::load(var_45);
        var_47 = wp::extract(var_48, var_46);
        var_49 = wp::mod(var_worldid, var_47);
        var_50 = wp::address(var_pair_solreffriction, var_49, var_5);
        var_52 = wp::load(var_50);
        var_51 = wp::copy(var_52);
        // solimp = pair_solimp[worldid % pair_solimp.shape[0], pairid]                       <L 277>
        var_53 = &(var_pair_solimp.shape);
        var_56 = wp::load(var_53);
        var_55 = wp::extract(var_56, var_54);
        var_57 = wp::mod(var_worldid, var_55);
        var_58 = wp::address(var_pair_solimp, var_57, var_5);
        var_60 = wp::load(var_58);
        var_59 = wp::copy(var_60);
    }
    if (!var_9) {
        // g1 = geoms[0]                                                                      <L 279>
        var_62 = wp::extract(var_1, var_61);
        // g2 = geoms[1]                                                                      <L 280>
        var_64 = wp::extract(var_1, var_63);
        // solmix_id = worldid % geom_solmix.shape[0]                                         <L 281>
        var_65 = &(var_geom_solmix.shape);
        var_68 = wp::load(var_65);
        var_67 = wp::extract(var_68, var_66);
        var_69 = wp::mod(var_worldid, var_67);
        // friction_id = worldid % geom_friction.shape[0]                                     <L 282>
        var_70 = &(var_geom_friction.shape);
        var_73 = wp::load(var_70);
        var_72 = wp::extract(var_73, var_71);
        var_74 = wp::mod(var_worldid, var_72);
        // solref_id = worldid % geom_solref.shape[0]                                         <L 283>
        var_75 = &(var_geom_solref.shape);
        var_78 = wp::load(var_75);
        var_77 = wp::extract(var_78, var_76);
        var_79 = wp::mod(var_worldid, var_77);
        // solimp_id = worldid % geom_solimp.shape[0]                                         <L 284>
        var_80 = &(var_geom_solimp.shape);
        var_83 = wp::load(var_80);
        var_82 = wp::extract(var_83, var_81);
        var_84 = wp::mod(var_worldid, var_82);
        // margin_id = worldid % geom_margin.shape[0]                                         <L 285>
        var_85 = &(var_geom_margin.shape);
        var_88 = wp::load(var_85);
        var_87 = wp::extract(var_88, var_86);
        var_89 = wp::mod(var_worldid, var_87);
        // gap_id = worldid % geom_gap.shape[0]                                               <L 286>
        var_90 = &(var_geom_gap.shape);
        var_93 = wp::load(var_90);
        var_92 = wp::extract(var_93, var_91);
        var_94 = wp::mod(var_worldid, var_92);
        // solmix1 = geom_solmix[solmix_id, g1]                                               <L 288>
        var_95 = wp::address(var_geom_solmix, var_69, var_62);
        var_97 = wp::load(var_95);
        var_96 = wp::copy(var_97);
        // solmix2 = geom_solmix[solmix_id, g2]                                               <L 289>
        var_98 = wp::address(var_geom_solmix, var_69, var_64);
        var_100 = wp::load(var_98);
        var_99 = wp::copy(var_100);
        // condim1 = geom_condim[g1]                                                          <L 291>
        var_101 = wp::address(var_geom_condim, var_62);
        var_103 = wp::load(var_101);
        var_102 = wp::copy(var_103);
        // condim2 = geom_condim[g2]                                                          <L 292>
        var_104 = wp::address(var_geom_condim, var_64);
        var_106 = wp::load(var_104);
        var_105 = wp::copy(var_106);
        // p1 = geom_priority[g1]                                                             <L 295>
        var_107 = wp::address(var_geom_priority, var_62);
        var_109 = wp::load(var_107);
        var_108 = wp::copy(var_109);
        // p2 = geom_priority[g2]                                                             <L 296>
        var_110 = wp::address(var_geom_priority, var_64);
        var_112 = wp::load(var_110);
        var_111 = wp::copy(var_112);
        // if p1 > p2:                                                                        <L 298>
        var_113 = (var_108 > var_111);
        if (var_113) {
            // mix = 1.0                                                                      <L 299>
            // condim = condim1                                                               <L 300>
            var_115 = wp::copy(var_102);
            // max_geom_friction = geom_friction[friction_id, g1]                             <L 301>
            var_116 = wp::address(var_geom_friction, var_74, var_62);
            var_118 = wp::load(var_116);
            var_117 = wp::copy(var_118);
        }
        var_119 = wp::where(var_113, var_115, var_27);
        if (!var_113) {
            // elif p2 > p1:                                                                  <L 302>
            var_120 = (var_111 > var_108);
            if (var_120) {
                // mix = 0.0                                                                  <L 303>
                // condim = condim2                                                           <L 304>
                var_122 = wp::copy(var_105);
                // max_geom_friction = geom_friction[friction_id, g2]                         <L 305>
                var_123 = wp::address(var_geom_friction, var_74, var_64);
                var_125 = wp::load(var_123);
                var_124 = wp::copy(var_125);
            }
            var_126 = wp::where(var_120, var_122, var_119);
            var_127 = wp::where(var_120, var_121, var_114);
            var_128 = wp::where(var_120, var_124, var_117);
            if (!var_120) {
                // mix = safe_div(solmix1, solmix1 + solmix2)                                 <L 307>
                var_129 = wp::add(var_96, var_99);
                var_130 = safe_div_0(var_96, var_129);
                // mix = wp.where((solmix1 < MJ_MINVAL) and (solmix2 < MJ_MINVAL), 0.5, mix)       <L 308>
                var_132 = (var_96 < var_131);
                var_133 = (var_99 < var_131);
                var_134 = var_132 && var_133;
                var_136 = wp::where(var_134, var_135, var_130);
                // mix = wp.where((solmix1 < MJ_MINVAL) and (solmix2 >= MJ_MINVAL), 0.0, mix)       <L 309>
                var_137 = (var_96 < var_131);
                var_138 = (var_99 >= var_131);
                var_139 = var_137 && var_138;
                var_141 = wp::where(var_139, var_140, var_136);
                // mix = wp.where((solmix1 >= MJ_MINVAL) and (solmix2 < MJ_MINVAL), 1.0, mix)       <L 310>
                var_142 = (var_96 >= var_131);
                var_143 = (var_99 < var_131);
                var_144 = var_142 && var_143;
                var_146 = wp::where(var_144, var_145, var_141);
                // condim = wp.max(condim1, condim2)                                          <L 311>
                var_147 = wp::max(var_102, var_105);
                // max_geom_friction = wp.max(geom_friction[friction_id, g1], geom_friction[friction_id, g2])       <L 312>
                var_148 = wp::address(var_geom_friction, var_74, var_62);
                var_149 = wp::address(var_geom_friction, var_74, var_64);
                var_151 = wp::load(var_148);
                var_152 = wp::load(var_149);
                var_150 = wp::max(var_151, var_152);
            }
            var_153 = wp::where(var_120, var_126, var_147);
            var_154 = wp::where(var_120, var_127, var_146);
            var_155 = wp::where(var_120, var_128, var_150);
        }
        var_156 = wp::where(var_113, var_119, var_153);
        var_157 = wp::where(var_113, var_114, var_154);
        var_158 = wp::where(var_113, var_117, var_155);
        // friction = vec5(                                                                   <L 314>
        // max_geom_friction[0],                                                              <L 315>
        var_160 = wp::extract(var_158, var_159);
        // max_geom_friction[0],                                                              <L 316>
        var_162 = wp::extract(var_158, var_161);
        // max_geom_friction[1],                                                              <L 317>
        var_164 = wp::extract(var_158, var_163);
        // max_geom_friction[2],                                                              <L 318>
        var_166 = wp::extract(var_158, var_165);
        // max_geom_friction[2],                                                              <L 319>
        var_168 = wp::extract(var_158, var_167);
        var_169 = wp::vec_t<5, wp::float32>({var_160, var_162, var_164, var_166, var_168});
        // if geom_solref[solref_id, g1][0] > 0.0 and geom_solref[solref_id, g2][0] > 0.0:       <L 322>
        var_170 = wp::address(var_geom_solref, var_79, var_62);
        var_173 = wp::load(var_170);
        var_172 = wp::extract(var_173, var_171);
        var_175 = (var_172 > var_174);
        var_176 = wp::address(var_geom_solref, var_79, var_64);
        var_179 = wp::load(var_176);
        var_178 = wp::extract(var_179, var_177);
        var_181 = (var_178 > var_180);
        var_182 = var_175 && var_181;
        if (var_182) {
            // solref = mix * geom_solref[solref_id, g1] + (1.0 - mix) * geom_solref[solref_id, g2]       <L 323>
            var_183 = wp::address(var_geom_solref, var_79, var_62);
            var_185 = wp::load(var_183);
            var_184 = wp::mul(var_157, var_185);
            var_187 = wp::sub(var_186, var_157);
            var_188 = wp::address(var_geom_solref, var_79, var_64);
            var_190 = wp::load(var_188);
            var_189 = wp::mul(var_187, var_190);
            var_191 = wp::add(var_184, var_189);
        }
        var_192 = wp::where(var_182, var_191, var_43);
        if (!var_182) {
            // solref = wp.min(geom_solref[solref_id, g1], geom_solref[solref_id, g2])        <L 325>
            var_193 = wp::address(var_geom_solref, var_79, var_62);
            var_194 = wp::address(var_geom_solref, var_79, var_64);
            var_196 = wp::load(var_193);
            var_197 = wp::load(var_194);
            var_195 = wp::min(var_196, var_197);
        }
        var_198 = wp::where(var_182, var_192, var_195);
        // solreffriction = wp.vec2(0.0, 0.0)                                                 <L 327>
        var_201 = wp::vec_t<2, wp::float32>(var_199, var_200);
        // solimp = mix * geom_solimp[solimp_id, g1] + (1.0 - mix) * geom_solimp[solimp_id, g2]       <L 328>
        var_202 = wp::address(var_geom_solimp, var_84, var_62);
        var_204 = wp::load(var_202);
        var_203 = wp::mul(var_157, var_204);
        var_206 = wp::sub(var_205, var_157);
        var_207 = wp::address(var_geom_solimp, var_84, var_64);
        var_209 = wp::load(var_207);
        var_208 = wp::mul(var_206, var_209);
        var_210 = wp::add(var_203, var_208);
        // margin = geom_margin[margin_id, g1] + geom_margin[margin_id, g2]                   <L 330>
        var_211 = wp::address(var_geom_margin, var_89, var_62);
        var_212 = wp::address(var_geom_margin, var_89, var_64);
        var_214 = wp::load(var_211);
        var_215 = wp::load(var_212);
        var_213 = wp::add(var_214, var_215);
        // gap = geom_gap[gap_id, g1] + geom_gap[gap_id, g2]                                  <L 331>
        var_216 = wp::address(var_geom_gap, var_94, var_62);
        var_217 = wp::address(var_geom_gap, var_94, var_64);
        var_219 = wp::load(var_216);
        var_220 = wp::load(var_217);
        var_218 = wp::add(var_219, var_220);
    }
    var_221 = wp::where(var_9, var_16, var_213);
    var_222 = wp::where(var_9, var_24, var_218);
    var_223 = wp::where(var_9, var_27, var_156);
    var_224 = wp::where(var_9, var_35, var_169);
    var_225 = wp::where(var_9, var_43, var_198);
    var_226 = wp::where(var_9, var_51, var_201);
    var_227 = wp::where(var_9, var_59, var_210);
    // friction = vec5(                                                                       <L 333>
    // wp.max(MJ_MINMU, friction[0]),                                                         <L 334>
    var_230 = wp::extract(var_224, var_229);
    var_231 = wp::max(var_228, var_230);
    // wp.max(MJ_MINMU, friction[1]),                                                         <L 335>
    var_233 = wp::extract(var_224, var_232);
    var_234 = wp::max(var_228, var_233);
    // wp.max(MJ_MINMU, friction[2]),                                                         <L 336>
    var_236 = wp::extract(var_224, var_235);
    var_237 = wp::max(var_228, var_236);
    // wp.max(MJ_MINMU, friction[3]),                                                         <L 337>
    var_239 = wp::extract(var_224, var_238);
    var_240 = wp::max(var_228, var_239);
    // wp.max(MJ_MINMU, friction[4]),                                                         <L 338>
    var_242 = wp::extract(var_224, var_241);
    var_243 = wp::max(var_228, var_242);
    var_244 = wp::vec_t<5, wp::float32>({var_231, var_234, var_237, var_240, var_243});
    // return geoms, margin, gap, condim, friction, solref, solreffriction, solimp            <L 341>
    ret_0 = var_1;
    ret_1 = var_221;
    ret_2 = var_222;
    ret_3 = var_223;
    ret_4 = var_244;
    ret_5 = var_225;
    ret_6 = var_226;
    ret_7 = var_227;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_core.py:65
static CUDA_CALLABLE void geom_collision_pair_0(
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::int32> var_geom_dataid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::int32> var_mesh_vertadr,
    wp::array_t<wp::int32> var_mesh_vertnum,
    wp::array_t<wp::int32> var_mesh_graphadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_vert,
    wp::array_t<wp::int32> var_mesh_graph,
    wp::array_t<wp::int32> var_mesh_polynum,
    wp::array_t<wp::int32> var_mesh_polyadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_polynormal,
    wp::array_t<wp::int32> var_mesh_polyvertadr,
    wp::array_t<wp::int32> var_mesh_polyvertnum,
    wp::array_t<wp::int32> var_mesh_polyvert,
    wp::array_t<wp::int32> var_mesh_polymapadr,
    wp::array_t<wp::int32> var_mesh_polymapnum,
    wp::array_t<wp::int32> var_mesh_polymap,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::int32 var_worldid,
    Geom_3242f8a8 & ret_0,
    Geom_3242f8a8 & ret_1)
{
    //---------
    // primal vars
    Geom_3242f8a8 var_0;
    Geom_3242f8a8 var_1;
    const wp::int32 var_2 = 0;
    wp::int32 var_3;
    const wp::int32 var_4 = 1;
    wp::int32 var_5;
    wp::int32* var_6;
    wp::int32 var_7;
    wp::int32 var_8;
    wp::int32* var_9;
    wp::int32 var_10;
    wp::int32 var_11;
    wp::vec_t<3, wp::float32>* var_12;
    wp::vec_t<3, wp::float32>* var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::mat_t<3, 3, wp::float32>* var_15;
    wp::mat_t<3, 3, wp::float32>* var_16;
    wp::mat_t<3, 3, wp::float32> var_17;
    wp::shape_t* var_18;
    const wp::int32 var_19 = 0;
    wp::int32 var_20;
    wp::shape_t var_21;
    wp::int32 var_22;
    wp::vec_t<3, wp::float32>* var_23;
    wp::vec_t<3, wp::float32>* var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::mat_t<3, 3, wp::float32>* var_26;
    const wp::int32 var_27 = 0;
    const wp::int32 var_28 = 2;
    wp::float32 var_29;
    wp::mat_t<3, 3, wp::float32> var_30;
    wp::mat_t<3, 3, wp::float32>* var_31;
    const wp::int32 var_32 = 1;
    const wp::int32 var_33 = 2;
    wp::float32 var_34;
    wp::mat_t<3, 3, wp::float32> var_35;
    wp::mat_t<3, 3, wp::float32>* var_36;
    const wp::int32 var_37 = 2;
    const wp::int32 var_38 = 2;
    wp::float32 var_39;
    wp::mat_t<3, 3, wp::float32> var_40;
    wp::vec_t<3, wp::float32> var_41;
    wp::vec_t<3, wp::float32>* var_42;
    wp::vec_t<3, wp::float32>* var_43;
    wp::vec_t<3, wp::float32>* var_44;
    wp::vec_t<3, wp::float32> var_45;
    wp::mat_t<3, 3, wp::float32>* var_46;
    wp::mat_t<3, 3, wp::float32>* var_47;
    wp::mat_t<3, 3, wp::float32> var_48;
    wp::shape_t* var_49;
    const wp::int32 var_50 = 0;
    wp::int32 var_51;
    wp::shape_t var_52;
    wp::int32 var_53;
    wp::vec_t<3, wp::float32>* var_54;
    wp::vec_t<3, wp::float32>* var_55;
    wp::vec_t<3, wp::float32> var_56;
    wp::mat_t<3, 3, wp::float32>* var_57;
    const wp::int32 var_58 = 0;
    const wp::int32 var_59 = 2;
    wp::float32 var_60;
    wp::mat_t<3, 3, wp::float32> var_61;
    wp::mat_t<3, 3, wp::float32>* var_62;
    const wp::int32 var_63 = 1;
    const wp::int32 var_64 = 2;
    wp::float32 var_65;
    wp::mat_t<3, 3, wp::float32> var_66;
    wp::mat_t<3, 3, wp::float32>* var_67;
    const wp::int32 var_68 = 2;
    const wp::int32 var_69 = 2;
    wp::float32 var_70;
    wp::mat_t<3, 3, wp::float32> var_71;
    wp::vec_t<3, wp::float32> var_72;
    wp::vec_t<3, wp::float32>* var_73;
    wp::shape_t* var_74;
    const wp::int32 var_75 = 0;
    wp::int32 var_76;
    wp::shape_t var_77;
    wp::int32 var_78;
    const wp::int32 var_79 = 7;
    bool var_80;
    wp::int32* var_81;
    wp::int32 var_82;
    wp::int32 var_83;
    const wp::int32 var_84 = 0;
    bool var_85;
    wp::int32* var_86;
    const wp::int32 var_87 = 1;
    const wp::int32 var_88 = -1;
    wp::int32 var_89;
    wp::int32 var_90;
    wp::int32* var_91;
    const wp::int32 var_92 = 0;
    bool var_93;
    wp::int32* var_94;
    const wp::int32 var_95 = 1;
    const wp::int32 var_96 = -1;
    wp::int32 var_97;
    wp::int32 var_98;
    wp::int32* var_99;
    const wp::int32 var_100 = 0;
    bool var_101;
    wp::int32* var_102;
    const wp::int32 var_103 = 1;
    const wp::int32 var_104 = -1;
    wp::int32 var_105;
    wp::int32 var_106;
    wp::int32* var_107;
    const wp::int32 var_108 = 0;
    bool var_109;
    wp::int32* var_110;
    const wp::int32 var_111 = 1;
    const wp::int32 var_112 = -1;
    wp::int32 var_113;
    wp::int32 var_114;
    wp::int32* var_115;
    const wp::int32 var_116 = 0;
    bool var_117;
    wp::int32* var_118;
    const wp::int32 var_119 = 1;
    const wp::int32 var_120 = -1;
    wp::int32 var_121;
    wp::int32 var_122;
    wp::int32* var_123;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_124;
    wp::array_t<wp::int32>* var_125;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_126;
    wp::array_t<wp::int32>* var_127;
    wp::array_t<wp::int32>* var_128;
    wp::array_t<wp::int32>* var_129;
    wp::array_t<wp::int32>* var_130;
    wp::array_t<wp::int32>* var_131;
    wp::array_t<wp::int32>* var_132;
    const wp::int32 var_133 = 7;
    bool var_134;
    wp::int32* var_135;
    wp::int32 var_136;
    wp::int32 var_137;
    const wp::int32 var_138 = 0;
    bool var_139;
    wp::int32* var_140;
    const wp::int32 var_141 = 1;
    const wp::int32 var_142 = -1;
    wp::int32 var_143;
    wp::int32 var_144;
    wp::int32* var_145;
    const wp::int32 var_146 = 0;
    bool var_147;
    wp::int32* var_148;
    const wp::int32 var_149 = 1;
    const wp::int32 var_150 = -1;
    wp::int32 var_151;
    wp::int32 var_152;
    wp::int32* var_153;
    const wp::int32 var_154 = 0;
    bool var_155;
    wp::int32* var_156;
    const wp::int32 var_157 = 1;
    const wp::int32 var_158 = -1;
    wp::int32 var_159;
    wp::int32 var_160;
    wp::int32* var_161;
    const wp::int32 var_162 = 0;
    bool var_163;
    wp::int32* var_164;
    const wp::int32 var_165 = 1;
    const wp::int32 var_166 = -1;
    wp::int32 var_167;
    wp::int32 var_168;
    wp::int32* var_169;
    const wp::int32 var_170 = 0;
    bool var_171;
    wp::int32* var_172;
    const wp::int32 var_173 = 1;
    const wp::int32 var_174 = -1;
    wp::int32 var_175;
    wp::int32 var_176;
    wp::int32* var_177;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_178;
    wp::array_t<wp::int32>* var_179;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_180;
    wp::array_t<wp::int32>* var_181;
    wp::array_t<wp::int32>* var_182;
    wp::array_t<wp::int32>* var_183;
    wp::array_t<wp::int32>* var_184;
    wp::array_t<wp::int32>* var_185;
    wp::array_t<wp::int32>* var_186;
    wp::int32 var_187;
    const wp::int32 var_188 = 1;
    const wp::int32 var_189 = -1;
    wp::int32* var_190;
    const wp::float32 var_191 = 0.0;
    wp::float32* var_192;
    const wp::int32 var_193 = 1;
    const wp::int32 var_194 = -1;
    wp::int32* var_195;
    const wp::float32 var_196 = 0.0;
    wp::float32* var_197;
    //---------
    // forward
    // def geom_collision_pair(                                                               <L 66>
    // geom1 = Geom()                                                                         <L 92>
    var_0 = Geom_3242f8a8();
    // geom2 = Geom()                                                                         <L 93>
    var_1 = Geom_3242f8a8();
    // g1 = geoms[0]                                                                          <L 95>
    var_3 = wp::extract(var_geoms, var_2);
    // g2 = geoms[1]                                                                          <L 96>
    var_5 = wp::extract(var_geoms, var_4);
    // geom_type1 = geom_type[g1]                                                             <L 97>
    var_6 = wp::address(var_geom_type, var_3);
    var_8 = wp::load(var_6);
    var_7 = wp::copy(var_8);
    // geom_type2 = geom_type[g2]                                                             <L 98>
    var_9 = wp::address(var_geom_type, var_5);
    var_11 = wp::load(var_9);
    var_10 = wp::copy(var_11);
    // geom1.pos = geom_xpos_in[worldid, g1]                                                  <L 100>
    var_12 = wp::address(var_geom_xpos_in, var_worldid, var_3);
    var_13 = &(var_0.pos);
    var_14 = wp::load(var_12);
    wp::store(var_13, var_14);
    // geom1.rot = geom_xmat_in[worldid, g1]                                                  <L 101>
    var_15 = wp::address(var_geom_xmat_in, var_worldid, var_3);
    var_16 = &(var_0.rot);
    var_17 = wp::load(var_15);
    wp::store(var_16, var_17);
    // geom1.size = geom_size[worldid % geom_size.shape[0], g1]                               <L 102>
    var_18 = &(var_geom_size.shape);
    var_21 = wp::load(var_18);
    var_20 = wp::extract(var_21, var_19);
    var_22 = wp::mod(var_worldid, var_20);
    var_23 = wp::address(var_geom_size, var_22, var_3);
    var_24 = &(var_0.size);
    var_25 = wp::load(var_23);
    wp::store(var_24, var_25);
    // geom1.normal = wp.vec3(geom1.rot[0, 2], geom1.rot[1, 2], geom1.rot[2, 2])              <L 104>
    var_26 = &(var_0.rot);
    var_30 = wp::load(var_26);
    var_29 = wp::extract(var_30, var_27, var_28);
    var_31 = &(var_0.rot);
    var_35 = wp::load(var_31);
    var_34 = wp::extract(var_35, var_32, var_33);
    var_36 = &(var_0.rot);
    var_40 = wp::load(var_36);
    var_39 = wp::extract(var_40, var_37, var_38);
    var_41 = wp::vec_t<3, wp::float32>(var_29, var_34, var_39);
    var_42 = &(var_0.normal);
    wp::store(var_42, var_41);
    // geom2.pos = geom_xpos_in[worldid, g2]                                                  <L 106>
    var_43 = wp::address(var_geom_xpos_in, var_worldid, var_5);
    var_44 = &(var_1.pos);
    var_45 = wp::load(var_43);
    wp::store(var_44, var_45);
    // geom2.rot = geom_xmat_in[worldid, g2]                                                  <L 107>
    var_46 = wp::address(var_geom_xmat_in, var_worldid, var_5);
    var_47 = &(var_1.rot);
    var_48 = wp::load(var_46);
    wp::store(var_47, var_48);
    // geom2.size = geom_size[worldid % geom_size.shape[0], g2]                               <L 108>
    var_49 = &(var_geom_size.shape);
    var_52 = wp::load(var_49);
    var_51 = wp::extract(var_52, var_50);
    var_53 = wp::mod(var_worldid, var_51);
    var_54 = wp::address(var_geom_size, var_53, var_5);
    var_55 = &(var_1.size);
    var_56 = wp::load(var_54);
    wp::store(var_55, var_56);
    // geom2.normal = wp.vec3(geom2.rot[0, 2], geom2.rot[1, 2], geom2.rot[2, 2])              <L 110>
    var_57 = &(var_1.rot);
    var_61 = wp::load(var_57);
    var_60 = wp::extract(var_61, var_58, var_59);
    var_62 = &(var_1.rot);
    var_66 = wp::load(var_62);
    var_65 = wp::extract(var_66, var_63, var_64);
    var_67 = &(var_1.rot);
    var_71 = wp::load(var_67);
    var_70 = wp::extract(var_71, var_68, var_69);
    var_72 = wp::vec_t<3, wp::float32>(var_60, var_65, var_70);
    var_73 = &(var_1.normal);
    wp::store(var_73, var_72);
    // dataid_setid = worldid % geom_dataid.shape[0]                                          <L 112>
    var_74 = &(var_geom_dataid.shape);
    var_77 = wp::load(var_74);
    var_76 = wp::extract(var_77, var_75);
    var_78 = wp::mod(var_worldid, var_76);
    // if geom_type1 == GeomType.MESH:                                                        <L 114>
    var_80 = (var_7 == var_79);
    if (var_80) {
        // dataid = geom_dataid[dataid_setid, g1]                                             <L 115>
        var_81 = wp::address(var_geom_dataid, var_78, var_3);
        var_83 = wp::load(var_81);
        var_82 = wp::copy(var_83);
        // geom1.vertadr = wp.where(dataid >= 0, mesh_vertadr[dataid], -1)                    <L 116>
        var_85 = (var_82 >= var_84);
        var_86 = wp::address(var_mesh_vertadr, var_82);
        var_90 = wp::load(var_86);
        var_89 = wp::where(var_85, var_90, var_88);
        var_91 = &(var_0.vertadr);
        wp::store(var_91, var_89);
        // geom1.vertnum = wp.where(dataid >= 0, mesh_vertnum[dataid], -1)                    <L 117>
        var_93 = (var_82 >= var_92);
        var_94 = wp::address(var_mesh_vertnum, var_82);
        var_98 = wp::load(var_94);
        var_97 = wp::where(var_93, var_98, var_96);
        var_99 = &(var_0.vertnum);
        wp::store(var_99, var_97);
        // geom1.graphadr = wp.where(dataid >= 0, mesh_graphadr[dataid], -1)                  <L 118>
        var_101 = (var_82 >= var_100);
        var_102 = wp::address(var_mesh_graphadr, var_82);
        var_106 = wp::load(var_102);
        var_105 = wp::where(var_101, var_106, var_104);
        var_107 = &(var_0.graphadr);
        wp::store(var_107, var_105);
        // geom1.mesh_polynum = wp.where(dataid >= 0, mesh_polynum[dataid], -1)               <L 119>
        var_109 = (var_82 >= var_108);
        var_110 = wp::address(var_mesh_polynum, var_82);
        var_114 = wp::load(var_110);
        var_113 = wp::where(var_109, var_114, var_112);
        var_115 = &(var_0.mesh_polynum);
        wp::store(var_115, var_113);
        // geom1.mesh_polyadr = wp.where(dataid >= 0, mesh_polyadr[dataid], -1)               <L 120>
        var_117 = (var_82 >= var_116);
        var_118 = wp::address(var_mesh_polyadr, var_82);
        var_122 = wp::load(var_118);
        var_121 = wp::where(var_117, var_122, var_120);
        var_123 = &(var_0.mesh_polyadr);
        wp::store(var_123, var_121);
        // geom1.vert = mesh_vert                                                             <L 122>
        var_124 = &(var_0.vert);
        wp::store(var_124, var_mesh_vert);
        // geom1.graph = mesh_graph                                                           <L 123>
        var_125 = &(var_0.graph);
        wp::store(var_125, var_mesh_graph);
        // geom1.mesh_polynormal = mesh_polynormal                                            <L 124>
        var_126 = &(var_0.mesh_polynormal);
        wp::store(var_126, var_mesh_polynormal);
        // geom1.mesh_polyvertadr = mesh_polyvertadr                                          <L 125>
        var_127 = &(var_0.mesh_polyvertadr);
        wp::store(var_127, var_mesh_polyvertadr);
        // geom1.mesh_polyvertnum = mesh_polyvertnum                                          <L 126>
        var_128 = &(var_0.mesh_polyvertnum);
        wp::store(var_128, var_mesh_polyvertnum);
        // geom1.mesh_polyvert = mesh_polyvert                                                <L 127>
        var_129 = &(var_0.mesh_polyvert);
        wp::store(var_129, var_mesh_polyvert);
        // geom1.mesh_polymapadr = mesh_polymapadr                                            <L 128>
        var_130 = &(var_0.mesh_polymapadr);
        wp::store(var_130, var_mesh_polymapadr);
        // geom1.mesh_polymapnum = mesh_polymapnum                                            <L 129>
        var_131 = &(var_0.mesh_polymapnum);
        wp::store(var_131, var_mesh_polymapnum);
        // geom1.mesh_polymap = mesh_polymap                                                  <L 130>
        var_132 = &(var_0.mesh_polymap);
        wp::store(var_132, var_mesh_polymap);
    }
    // if geom_type2 == GeomType.MESH:                                                        <L 132>
    var_134 = (var_10 == var_133);
    if (var_134) {
        // dataid = geom_dataid[dataid_setid, g2]                                             <L 133>
        var_135 = wp::address(var_geom_dataid, var_78, var_5);
        var_137 = wp::load(var_135);
        var_136 = wp::copy(var_137);
        // geom2.vertadr = wp.where(dataid >= 0, mesh_vertadr[dataid], -1)                    <L 134>
        var_139 = (var_136 >= var_138);
        var_140 = wp::address(var_mesh_vertadr, var_136);
        var_144 = wp::load(var_140);
        var_143 = wp::where(var_139, var_144, var_142);
        var_145 = &(var_1.vertadr);
        wp::store(var_145, var_143);
        // geom2.vertnum = wp.where(dataid >= 0, mesh_vertnum[dataid], -1)                    <L 135>
        var_147 = (var_136 >= var_146);
        var_148 = wp::address(var_mesh_vertnum, var_136);
        var_152 = wp::load(var_148);
        var_151 = wp::where(var_147, var_152, var_150);
        var_153 = &(var_1.vertnum);
        wp::store(var_153, var_151);
        // geom2.graphadr = wp.where(dataid >= 0, mesh_graphadr[dataid], -1)                  <L 136>
        var_155 = (var_136 >= var_154);
        var_156 = wp::address(var_mesh_graphadr, var_136);
        var_160 = wp::load(var_156);
        var_159 = wp::where(var_155, var_160, var_158);
        var_161 = &(var_1.graphadr);
        wp::store(var_161, var_159);
        // geom2.mesh_polynum = wp.where(dataid >= 0, mesh_polynum[dataid], -1)               <L 137>
        var_163 = (var_136 >= var_162);
        var_164 = wp::address(var_mesh_polynum, var_136);
        var_168 = wp::load(var_164);
        var_167 = wp::where(var_163, var_168, var_166);
        var_169 = &(var_1.mesh_polynum);
        wp::store(var_169, var_167);
        // geom2.mesh_polyadr = wp.where(dataid >= 0, mesh_polyadr[dataid], -1)               <L 138>
        var_171 = (var_136 >= var_170);
        var_172 = wp::address(var_mesh_polyadr, var_136);
        var_176 = wp::load(var_172);
        var_175 = wp::where(var_171, var_176, var_174);
        var_177 = &(var_1.mesh_polyadr);
        wp::store(var_177, var_175);
        // geom2.vert = mesh_vert                                                             <L 140>
        var_178 = &(var_1.vert);
        wp::store(var_178, var_mesh_vert);
        // geom2.graph = mesh_graph                                                           <L 141>
        var_179 = &(var_1.graph);
        wp::store(var_179, var_mesh_graph);
        // geom2.mesh_polynormal = mesh_polynormal                                            <L 142>
        var_180 = &(var_1.mesh_polynormal);
        wp::store(var_180, var_mesh_polynormal);
        // geom2.mesh_polyvertadr = mesh_polyvertadr                                          <L 143>
        var_181 = &(var_1.mesh_polyvertadr);
        wp::store(var_181, var_mesh_polyvertadr);
        // geom2.mesh_polyvertnum = mesh_polyvertnum                                          <L 144>
        var_182 = &(var_1.mesh_polyvertnum);
        wp::store(var_182, var_mesh_polyvertnum);
        // geom2.mesh_polyvert = mesh_polyvert                                                <L 145>
        var_183 = &(var_1.mesh_polyvert);
        wp::store(var_183, var_mesh_polyvert);
        // geom2.mesh_polymapadr = mesh_polymapadr                                            <L 146>
        var_184 = &(var_1.mesh_polymapadr);
        wp::store(var_184, var_mesh_polymapadr);
        // geom2.mesh_polymapnum = mesh_polymapnum                                            <L 147>
        var_185 = &(var_1.mesh_polymapnum);
        wp::store(var_185, var_mesh_polymapnum);
        // geom2.mesh_polymap = mesh_polymap                                                  <L 148>
        var_186 = &(var_1.mesh_polymap);
        wp::store(var_186, var_mesh_polymap);
    }
    var_187 = wp::where(var_134, var_136, var_82);
    // geom1.index = -1                                                                       <L 150>
    var_190 = &(var_0.index);
    wp::store(var_190, var_189);
    // geom1.margin = 0.0                                                                     <L 151>
    var_192 = &(var_0.margin);
    wp::store(var_192, var_191);
    // geom2.index = -1                                                                       <L 153>
    var_195 = &(var_1.index);
    wp::store(var_195, var_194);
    // geom2.margin = 0.0                                                                     <L 154>
    var_197 = &(var_1.margin);
    wp::store(var_197, var_196);
    // return geom1, geom2                                                                    <L 156>
    ret_0 = var_0;
    ret_1 = var_1;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_primitive_core.py:114
static CUDA_CALLABLE void sphere_sphere_0(
    wp::vec_t<3, wp::float32> var_pos1,
    wp::float32 var_radius1,
    wp::vec_t<3, wp::float32> var_pos2,
    wp::float32 var_radius2,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::float32 var_1;
    const wp::float32 var_2 = 0.0;
    bool var_3;
    const wp::float32 var_4 = 1.0;
    const wp::float32 var_5 = 0.0;
    const wp::float32 var_6 = 0.0;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    const wp::float32 var_12 = 0.5;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32> var_16;
    //---------
    // forward
    // def sphere_sphere(                                                                     <L 115>
    // dir = pos2 - pos1                                                                      <L 135>
    var_0 = wp::sub(var_pos2, var_pos1);
    // dist = wp.length(dir)                                                                  <L 136>
    var_1 = wp::length(var_0);
    // if dist == 0.0:                                                                        <L 137>
    var_3 = (var_1 == var_2);
    if (var_3) {
        // n = wp.vec3(1.0, 0.0, 0.0)                                                         <L 138>
        var_7 = wp::vec_t<3, wp::float32>(var_4, var_5, var_6);
    }
    if (!var_3) {
        // n = dir / dist                                                                     <L 140>
        var_8 = wp::div(var_0, var_1);
    }
    var_9 = wp::where(var_3, var_7, var_8);
    // dist = dist - (radius1 + radius2)                                                      <L 141>
    var_10 = wp::add(var_radius1, var_radius2);
    var_11 = wp::sub(var_1, var_10);
    // pos = pos1 + n * (radius1 + 0.5 * dist)                                                <L 142>
    var_13 = wp::mul(var_12, var_11);
    var_14 = wp::add(var_radius1, var_13);
    var_15 = wp::mul(var_9, var_14);
    var_16 = wp::add(var_pos1, var_15);
    // return dist, pos, n                                                                    <L 143>
    ret_0 = var_11;
    ret_1 = var_16;
    ret_2 = var_9;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:202
static CUDA_CALLABLE void orthogonals_0(
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    const wp::float32 var_1 = 1.0;
    const wp::float32 var_2 = 0.0;
    wp::vec_t<3, wp::float32> var_3;
    const wp::float32 var_4 = 0.0;
    const wp::float32 var_5 = 0.0;
    const wp::float32 var_6 = 1.0;
    wp::vec_t<3, wp::float32> var_7;
    const wp::float32 var_8 = 0.5;
    const wp::float32 var_9 = -0.5;
    const wp::int32 var_10 = 1;
    wp::float32 var_11;
    bool var_12;
    const wp::int32 var_13 = 1;
    wp::float32 var_14;
    const wp::float32 var_15 = 0.5;
    bool var_16;
    bool var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::float32 var_23;
    const wp::float32 var_24 = 0.0;
    bool var_25;
    const wp::float32 var_26 = 0.0;
    const wp::float32 var_27 = 0.0;
    const wp::float32 var_28 = 0.0;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    //---------
    // forward
    // def orthogonals(a: wp.vec3):                                                           <L 203>
    // y = wp.vec3(0.0, 1.0, 0.0)                                                             <L 204>
    var_3 = wp::vec_t<3, wp::float32>(var_0, var_1, var_2);
    // z = wp.vec3(0.0, 0.0, 1.0)                                                             <L 205>
    var_7 = wp::vec_t<3, wp::float32>(var_4, var_5, var_6);
    // b = wp.where((-0.5 < a[1]) and (a[1] < 0.5), y, z)                                     <L 206>
    var_11 = wp::extract(var_a, var_10);
    var_12 = (var_9 < var_11);
    var_14 = wp::extract(var_a, var_13);
    var_16 = (var_14 < var_15);
    var_17 = var_12 && var_16;
    var_18 = wp::where(var_17, var_3, var_7);
    // b = b - a * wp.dot(a, b)                                                               <L 207>
    var_19 = wp::dot(var_a, var_18);
    var_20 = wp::mul(var_a, var_19);
    var_21 = wp::sub(var_18, var_20);
    // b = wp.normalize(b)                                                                    <L 208>
    var_22 = wp::normalize(var_21);
    // if wp.length(a) == 0.0:                                                                <L 209>
    var_23 = wp::length(var_a);
    var_25 = (var_23 == var_24);
    if (var_25) {
        // b = wp.vec3(0.0, 0.0, 0.0)                                                         <L 210>
        var_29 = wp::vec_t<3, wp::float32>(var_26, var_27, var_28);
    }
    var_30 = wp::where(var_25, var_29, var_22);
    // c = wp.cross(a, b)                                                                     <L 211>
    var_31 = wp::cross(var_a, var_30);
    // return b, c                                                                            <L 213>
    ret_0 = var_30;
    ret_1 = var_31;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:246
static CUDA_CALLABLE wp::mat_t<3, 3, wp::float32> make_frame_0(
    wp::vec_t<3, wp::float32> var_a)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
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
    const wp::int32 var_15 = 0;
    wp::float32 var_16;
    const wp::int32 var_17 = 1;
    wp::float32 var_18;
    const wp::int32 var_19 = 2;
    wp::float32 var_20;
    wp::mat_t<3, 3, wp::float32> var_21;
    //---------
    // forward
    // def make_frame(a: wp.vec3):                                                            <L 247>
    // a = wp.normalize(a)                                                                    <L 248>
    var_0 = wp::normalize(var_a);
    // b, c = orthogonals(a)                                                                  <L 249>
    orthogonals_0(var_0, var_1, var_2);
    // return wp.mat33(                                                                       <L 252>
    // a.x, a.y, a.z,                                                                         <L 253>
    var_4 = wp::extract(var_0, var_3);
    var_6 = wp::extract(var_0, var_5);
    var_8 = wp::extract(var_0, var_7);
    // b.x, b.y, b.z,                                                                         <L 254>
    var_10 = wp::extract(var_1, var_9);
    var_12 = wp::extract(var_1, var_11);
    var_14 = wp::extract(var_1, var_13);
    // c.x, c.y, c.z                                                                          <L 255>
    var_16 = wp::extract(var_2, var_15);
    var_18 = wp::extract(var_2, var_17);
    var_20 = wp::extract(var_2, var_19);
    var_21 = wp::mat_t<3, 3, wp::float32>(var_4, var_6, var_8, var_10, var_12, var_14, var_16, var_18, var_20);
    return var_21;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_core.py:159
static CUDA_CALLABLE wp::int32 write_contact_0(
    wp::int32 var_naconmax_in,
    wp::int32 var_id_,
    wp::float32 var_dist_in,
    wp::vec_t<3, wp::float32> var_pos_in,
    wp::mat_t<3, 3, wp::float32> var_frame_in,
    wp::float32 var_margin_in,
    wp::float32 var_gap_in,
    wp::int32 var_condim_in,
    wp::vec_t<5, wp::float32> var_friction_in,
    wp::vec_t<2, wp::float32> var_solref_in,
    wp::vec_t<2, wp::float32> var_solreffriction_in,
    wp::vec_t<5, wp::float32> var_solimp_in,
    wp::vec_t<2, wp::int32> var_geoms_in,
    wp::vec_t<2, wp::int32> var_pairid_in,
    wp::int32 var_worldid_in,
    wp::array_t<wp::float32> var_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_out,
    wp::array_t<wp::float32> var_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_solimp_out,
    wp::array_t<wp::int32> var_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::int32> var_nacon_out)
{
    //---------
    // primal vars
    bool var_0;
    const wp::int32 var_1 = 0;
    wp::int32 var_2;
    const wp::int32 var_3 = 2;
    const wp::int32 var_4 = -2;
    bool var_5;
    bool var_6;
    bool var_7;
    const wp::int32 var_8 = 1;
    wp::int32 var_9;
    const wp::int32 var_10 = 1;
    const wp::int32 var_11 = -1;
    bool var_12;
    bool var_13;
    const wp::int32 var_14 = 0;
    const wp::int32 var_15 = 0;
    const wp::int32 var_16 = 0;
    wp::int32 var_17;
    const wp::int32 var_18 = 1;
    const wp::int32 var_19 = -1;
    bool var_20;
    bool var_21;
    const wp::int32 var_22 = 1;
    wp::int32 var_23;
    wp::int32 var_24;
    const wp::int32 var_25 = 1;
    wp::int32 var_26;
    const wp::int32 var_27 = 0;
    bool var_28;
    const wp::int32 var_29 = 2;
    wp::int32 var_30;
    wp::int32 var_31;
    const wp::int32 var_32 = 0;
    const wp::int32 var_33 = 1;
    wp::int32 var_34;
    bool var_35;
    wp::float32 var_36;
    wp::shape_t* var_37;
    const wp::int32 var_38 = 1;
    wp::int32 var_39;
    wp::shape_t var_40;
    wp::range_t var_41;
    wp::int32 var_42;
    const wp::int32 var_43 = 1;
    const wp::int32 var_44 = -1;
    wp::int32 var_45;
    const wp::int32 var_46 = 0;
    //---------
    // forward
    // def write_contact(                                                                     <L 160>
    // active = dist_in < margin_in                                                           <L 199>
    var_0 = (var_dist_in < var_margin_in);
    // if (pairid_in[0] == -2 or not active) and pairid_in[1] == -1:                          <L 202>
    var_2 = wp::extract(var_pairid_in, var_1);
    var_5 = (var_2 == var_4);
    var_6 = wp::unot(var_0);
    var_7 = var_5 || var_6;
    var_9 = wp::extract(var_pairid_in, var_8);
    var_12 = (var_9 == var_11);
    var_13 = var_7 && var_12;
    if (var_13) {
        // return 0                                                                           <L 203>
        return var_14;
    }
    // contact_type = 0                                                                       <L 205>
    // if pairid_in[0] >= -1 and active:                                                      <L 207>
    var_17 = wp::extract(var_pairid_in, var_16);
    var_20 = (var_17 >= var_19);
    var_21 = var_20 && var_0;
    if (var_21) {
        // contact_type |= ContactType.CONSTRAINT                                             <L 208>
        var_23 = wp::bit_or(var_15, var_22);
    }
    var_24 = wp::where(var_21, var_23, var_15);
    // if pairid_in[1] >= 0:                                                                  <L 210>
    var_26 = wp::extract(var_pairid_in, var_25);
    var_28 = (var_26 >= var_27);
    if (var_28) {
        // contact_type |= ContactType.SENSOR                                                 <L 211>
        var_30 = wp::bit_or(var_24, var_29);
    }
    var_31 = wp::where(var_28, var_30, var_24);
    // cid = wp.atomic_add(nacon_out, 0, 1)                                                   <L 213>
    var_34 = wp::atomic_add(var_nacon_out, var_32, var_33);
    // if cid < naconmax_in:                                                                  <L 214>
    var_35 = (var_34 < var_naconmax_in);
    if (var_35) {
        // contact_dist_out[cid] = dist_in                                                    <L 215>
        wp::array_store(var_contact_dist_out, var_34, var_dist_in);
        // contact_pos_out[cid] = pos_in                                                      <L 216>
        wp::array_store(var_contact_pos_out, var_34, var_pos_in);
        // contact_frame_out[cid] = frame_in                                                  <L 217>
        wp::array_store(var_contact_frame_out, var_34, var_frame_in);
        // contact_geom_out[cid] = geoms_in                                                   <L 218>
        wp::array_store(var_contact_geom_out, var_34, var_geoms_in);
        // contact_worldid_out[cid] = worldid_in                                              <L 219>
        wp::array_store(var_contact_worldid_out, var_34, var_worldid_in);
        // includemargin = margin_in - gap_in                                                 <L 220>
        var_36 = wp::sub(var_margin_in, var_gap_in);
        // contact_includemargin_out[cid] = includemargin                                     <L 221>
        wp::array_store(var_contact_includemargin_out, var_34, var_36);
        // contact_dim_out[cid] = condim_in                                                   <L 222>
        wp::array_store(var_contact_dim_out, var_34, var_condim_in);
        // contact_friction_out[cid] = friction_in                                            <L 223>
        wp::array_store(var_contact_friction_out, var_34, var_friction_in);
        // contact_solref_out[cid] = solref_in                                                <L 224>
        wp::array_store(var_contact_solref_out, var_34, var_solref_in);
        // contact_solreffriction_out[cid] = solreffriction_in                                <L 225>
        wp::array_store(var_contact_solreffriction_out, var_34, var_solreffriction_in);
        // contact_solimp_out[cid] = solimp_in                                                <L 226>
        wp::array_store(var_contact_solimp_out, var_34, var_solimp_in);
        // contact_type_out[cid] = contact_type                                               <L 227>
        wp::array_store(var_contact_type_out, var_34, var_31);
        // contact_geomcollisionid_out[cid] = id_                                             <L 228>
        wp::array_store(var_contact_geomcollisionid_out, var_34, var_id_);
        // for i in range(contact_efc_address_out.shape[1]):                                  <L 229>
        var_37 = &(var_contact_efc_address_out.shape);
        var_40 = wp::load(var_37);
        var_39 = wp::extract(var_40, var_38);
        var_41 = wp::range(var_39);
        start_for_1:;
            if (iter_cmp(var_41) == 0) goto end_for_1;
            var_42 = wp::iter_next(var_41);
            // contact_efc_address_out[cid, i] = -1                                           <L 230>
            wp::array_store(var_contact_efc_address_out, var_34, var_42, var_44);
            goto start_for_1;
        end_for_1:;
        // return int(active)                                                                 <L 231>
        var_45 = wp::int(var_0);
        return var_45;
    }
    // return 0                                                                               <L 232>
    return var_46;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_primitive.py:352
static CUDA_CALLABLE void sphere_sphere_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_3242f8a8 var_sphere1,
    Geom_3242f8a8 var_sphere2,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
    wp::array_t<wp::float32> var_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_out,
    wp::array_t<wp::float32> var_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_solimp_out,
    wp::array_t<wp::int32> var_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::int32> var_nacon_out)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32>* var_0;
    wp::vec_t<3, wp::float32>* var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32>* var_5;
    wp::vec_t<3, wp::float32>* var_6;
    const wp::int32 var_7 = 0;
    wp::float32 var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::float32 var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    const wp::int32 var_15 = 0;
    wp::mat_t<3, 3, wp::float32> var_16;
    wp::int32 var_17;
    //---------
    // forward
    // def sphere_sphere_wrapper(                                                             <L 353>
    // dist, pos, normal = sphere_sphere(sphere1.pos, sphere1.size[0], sphere2.pos, sphere2.size[0])       <L 387>
    var_0 = &(var_sphere1.pos);
    var_1 = &(var_sphere1.size);
    var_4 = wp::load(var_1);
    var_3 = wp::extract(var_4, var_2);
    var_5 = &(var_sphere2.pos);
    var_6 = &(var_sphere2.size);
    var_9 = wp::load(var_6);
    var_8 = wp::extract(var_9, var_7);
    var_13 = wp::load(var_0);
    var_14 = wp::load(var_5);
    sphere_sphere_0(var_13, var_3, var_14, var_8, var_10, var_11, var_12);
    // write_contact(                                                                         <L 389>
    // naconmax_in,                                                                           <L 390>
    // 0,                                                                                     <L 391>
    // dist,                                                                                  <L 392>
    // pos,                                                                                   <L 393>
    // make_frame(normal),                                                                    <L 394>
    var_16 = make_frame_0(var_12);
    // margin,                                                                                <L 395>
    // gap,                                                                                   <L 396>
    // condim,                                                                                <L 397>
    // friction,                                                                              <L 398>
    // solref,                                                                                <L 399>
    // solreffriction,                                                                        <L 400>
    // solimp,                                                                                <L 401>
    // geoms,                                                                                 <L 402>
    // pairid,                                                                                <L 403>
    // worldid,                                                                               <L 404>
    // contact_dist_out,                                                                      <L 405>
    // contact_pos_out,                                                                       <L 406>
    // contact_frame_out,                                                                     <L 407>
    // contact_includemargin_out,                                                             <L 408>
    // contact_friction_out,                                                                  <L 409>
    // contact_solref_out,                                                                    <L 410>
    // contact_solreffriction_out,                                                            <L 411>
    // contact_solimp_out,                                                                    <L 412>
    // contact_dim_out,                                                                       <L 413>
    // contact_geom_out,                                                                      <L 414>
    // contact_efc_address_out,                                                               <L 415>
    // contact_worldid_out,                                                                   <L 416>
    // contact_type_out,                                                                      <L 417>
    // contact_geomcollisionid_out,                                                           <L 418>
    // nacon_out,                                                                             <L 419>
    var_17 = write_contact_0(var_naconmax_in, var_15, var_10, var_11, var_16, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_primitive_core.py:40
static CUDA_CALLABLE wp::vec_t<3, wp::float32> closest_segment_point_1(
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> var_b,
    wp::vec_t<3, wp::float32> var_pt)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    const wp::float32 var_4 = 1e-06;
    wp::float32 var_5;
    wp::float32 var_6;
    const wp::float32 var_7 = 0.0;
    const wp::float32 var_8 = 1.0;
    wp::float32 var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    //---------
    // forward
    // def closest_segment_point(a: wp.vec3, b: wp.vec3, pt: wp.vec3) -> wp.vec3:             <L 41>
    // ab = b - a                                                                             <L 43>
    var_0 = wp::sub(var_b, var_a);
    // t = wp.dot(pt - a, ab) / (wp.dot(ab, ab) + 1e-6)                                       <L 44>
    var_1 = wp::sub(var_pt, var_a);
    var_2 = wp::dot(var_1, var_0);
    var_3 = wp::dot(var_0, var_0);
    var_5 = wp::add(var_3, var_4);
    var_6 = wp::div(var_2, var_5);
    // return a + wp.clamp(t, 0.0, 1.0) * ab                                                  <L 45>
    var_9 = wp::clamp(var_6, var_7, var_8);
    var_10 = wp::mul(var_9, var_0);
    var_11 = wp::add(var_a, var_10);
    return var_11;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_primitive_core.py:146
static CUDA_CALLABLE void sphere_capsule_0(
    wp::vec_t<3, wp::float32> var_sphere_pos,
    wp::float32 var_sphere_radius,
    wp::vec_t<3, wp::float32> var_capsule_pos,
    wp::vec_t<3, wp::float32> var_capsule_axis,
    wp::float32 var_capsule_radius,
    wp::float32 var_capsule_half_length,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::float32 var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    //---------
    // forward
    // def sphere_capsule(                                                                    <L 147>
    // segment = capsule_axis * capsule_half_length                                           <L 172>
    var_0 = wp::mul(var_capsule_axis, var_capsule_half_length);
    // pt = closest_segment_point(capsule_pos - segment, capsule_pos + segment, sphere_pos)       <L 175>
    var_1 = wp::sub(var_capsule_pos, var_0);
    var_2 = wp::add(var_capsule_pos, var_0);
    var_3 = closest_segment_point_1(var_1, var_2, var_sphere_pos);
    // return sphere_sphere(sphere_pos, sphere_radius, pt, capsule_radius)                    <L 178>
    sphere_sphere_0(var_sphere_pos, var_sphere_radius, var_3, var_capsule_radius, var_4, var_5, var_6);
    ret_0 = var_4;
    ret_1 = var_5;
    ret_2 = var_6;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_primitive.py:423
static CUDA_CALLABLE void sphere_capsule_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_3242f8a8 var_sphere,
    Geom_3242f8a8 var_cap,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
    wp::array_t<wp::float32> var_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_out,
    wp::array_t<wp::float32> var_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_solimp_out,
    wp::array_t<wp::int32> var_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::int32> var_nacon_out)
{
    //---------
    // primal vars
    wp::mat_t<3, 3, wp::float32>* var_0;
    const wp::int32 var_1 = 0;
    const wp::int32 var_2 = 2;
    wp::float32 var_3;
    wp::mat_t<3, 3, wp::float32> var_4;
    wp::mat_t<3, 3, wp::float32>* var_5;
    const wp::int32 var_6 = 1;
    const wp::int32 var_7 = 2;
    wp::float32 var_8;
    wp::mat_t<3, 3, wp::float32> var_9;
    wp::mat_t<3, 3, wp::float32>* var_10;
    const wp::int32 var_11 = 2;
    const wp::int32 var_12 = 2;
    wp::float32 var_13;
    wp::mat_t<3, 3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32>* var_16;
    wp::vec_t<3, wp::float32>* var_17;
    const wp::int32 var_18 = 0;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32>* var_21;
    wp::vec_t<3, wp::float32>* var_22;
    const wp::int32 var_23 = 0;
    wp::float32 var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32>* var_26;
    const wp::int32 var_27 = 1;
    wp::float32 var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::float32 var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::vec_t<3, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    const wp::int32 var_35 = 0;
    wp::mat_t<3, 3, wp::float32> var_36;
    wp::int32 var_37;
    //---------
    // forward
    // def sphere_capsule_wrapper(                                                            <L 424>
    // axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])                            <L 459>
    var_0 = &(var_cap.rot);
    var_4 = wp::load(var_0);
    var_3 = wp::extract(var_4, var_1, var_2);
    var_5 = &(var_cap.rot);
    var_9 = wp::load(var_5);
    var_8 = wp::extract(var_9, var_6, var_7);
    var_10 = &(var_cap.rot);
    var_14 = wp::load(var_10);
    var_13 = wp::extract(var_14, var_11, var_12);
    var_15 = wp::vec_t<3, wp::float32>(var_3, var_8, var_13);
    // dist, pos, normal = sphere_capsule(sphere.pos, sphere.size[0], cap.pos, axis, cap.size[0], cap.size[1])       <L 461>
    var_16 = &(var_sphere.pos);
    var_17 = &(var_sphere.size);
    var_20 = wp::load(var_17);
    var_19 = wp::extract(var_20, var_18);
    var_21 = &(var_cap.pos);
    var_22 = &(var_cap.size);
    var_25 = wp::load(var_22);
    var_24 = wp::extract(var_25, var_23);
    var_26 = &(var_cap.size);
    var_29 = wp::load(var_26);
    var_28 = wp::extract(var_29, var_27);
    var_33 = wp::load(var_16);
    var_34 = wp::load(var_21);
    sphere_capsule_0(var_33, var_19, var_34, var_15, var_24, var_28, var_30, var_31, var_32);
    // write_contact(                                                                         <L 463>
    // naconmax_in,                                                                           <L 464>
    // 0,                                                                                     <L 465>
    // dist,                                                                                  <L 466>
    // pos,                                                                                   <L 467>
    // make_frame(normal),                                                                    <L 468>
    var_36 = make_frame_0(var_32);
    // margin,                                                                                <L 469>
    // gap,                                                                                   <L 470>
    // condim,                                                                                <L 471>
    // friction,                                                                              <L 472>
    // solref,                                                                                <L 473>
    // solreffriction,                                                                        <L 474>
    // solimp,                                                                                <L 475>
    // geoms,                                                                                 <L 476>
    // pairid,                                                                                <L 477>
    // worldid,                                                                               <L 478>
    // contact_dist_out,                                                                      <L 479>
    // contact_pos_out,                                                                       <L 480>
    // contact_frame_out,                                                                     <L 481>
    // contact_includemargin_out,                                                             <L 482>
    // contact_friction_out,                                                                  <L 483>
    // contact_solref_out,                                                                    <L 484>
    // contact_solreffriction_out,                                                            <L 485>
    // contact_solimp_out,                                                                    <L 486>
    // contact_dim_out,                                                                       <L 487>
    // contact_geom_out,                                                                      <L 488>
    // contact_efc_address_out,                                                               <L 489>
    // contact_worldid_out,                                                                   <L 490>
    // contact_type_out,                                                                      <L 491>
    // contact_geomcollisionid_out,                                                           <L 492>
    // nacon_out,                                                                             <L 493>
    var_37 = write_contact_0(var_naconmax_in, var_35, var_30, var_31, var_36, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_primitive_core.py:106
static CUDA_CALLABLE void plane_sphere_0(
    wp::vec_t<3, wp::float32> var_plane_normal,
    wp::vec_t<3, wp::float32> var_plane_pos,
    wp::vec_t<3, wp::float32> var_sphere_pos,
    wp::float32 var_sphere_radius,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    const wp::float32 var_3 = 0.5;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    //---------
    // forward
    // def plane_sphere(plane_normal: wp.vec3, plane_pos: wp.vec3, sphere_pos: wp.vec3, sphere_radius: float) -> Tuple[float, wp.vec3]:       <L 107>
    // dist = wp.dot(sphere_pos - plane_pos, plane_normal) - sphere_radius                    <L 109>
    var_0 = wp::sub(var_sphere_pos, var_plane_pos);
    var_1 = wp::dot(var_0, var_plane_normal);
    var_2 = wp::sub(var_1, var_sphere_radius);
    // pos = sphere_pos - plane_normal * (sphere_radius + 0.5 * dist)                         <L 110>
    var_4 = wp::mul(var_3, var_2);
    var_5 = wp::add(var_sphere_radius, var_4);
    var_6 = wp::mul(var_plane_normal, var_5);
    var_7 = wp::sub(var_sphere_pos, var_6);
    // return dist, pos                                                                       <L 111>
    ret_0 = var_2;
    ret_1 = var_7;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_primitive_core.py:0
static CUDA_CALLABLE wp::float32 safe_div_1(
    wp::float32 var_x,
    wp::float32 var_y)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    bool var_1;
    const wp::float32 var_2 = 1e-15;
    wp::float32 var_3;
    wp::float32 var_4;
    //---------
    // forward
    // def safe_div(x: Any, y: Any) -> Any:                                                   <L 1>
    // return x / wp.where(y != 0.0, y, MJ_MINVAL)                                            <L 2>
    var_1 = (var_y != var_0);
    var_3 = wp::where(var_1, var_y, var_2);
    var_4 = wp::div(var_x, var_3);
    return var_4;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_primitive_core.py:446
static CUDA_CALLABLE void sphere_cylinder_0(
    wp::vec_t<3, wp::float32> var_sphere_pos,
    wp::float32 var_sphere_radius,
    wp::vec_t<3, wp::float32> var_cylinder_pos,
    wp::vec_t<3, wp::float32> var_cylinder_axis,
    wp::float32 var_cylinder_radius,
    wp::float32 var_cylinder_half_height,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::float32 var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    bool var_6;
    wp::float32 var_7;
    bool var_8;
    bool var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    bool var_14;
    const bool var_15 = false;
    bool var_16;
    const bool var_17 = false;
    bool var_18;
    bool var_19;
    bool var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::float32 var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    const wp::float32 var_25 = 0.0;
    bool var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::vec_t<3, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    wp::float32 var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32> var_37;
    const wp::float32 var_38 = 1.0;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::vec_t<3, wp::float32> var_45;
    wp::vec_t<3, wp::float32> var_46;
    wp::vec_t<3, wp::float32> var_47;
    const wp::float32 var_48 = 0.0;
    wp::float32 var_49;
    wp::vec_t<3, wp::float32> var_50;
    wp::vec_t<3, wp::float32> var_51;
    wp::vec_t<3, wp::float32> var_52;
    wp::vec_t<3, wp::float32> var_53;
    //---------
    // forward
    // def sphere_cylinder(                                                                   <L 447>
    // vec = sphere_pos - cylinder_pos                                                        <L 471>
    var_0 = wp::sub(var_sphere_pos, var_cylinder_pos);
    // x = wp.dot(vec, cylinder_axis)                                                         <L 472>
    var_1 = wp::dot(var_0, var_cylinder_axis);
    // a_proj = cylinder_axis * x                                                             <L 474>
    var_2 = wp::mul(var_cylinder_axis, var_1);
    // p_proj = vec - a_proj                                                                  <L 475>
    var_3 = wp::sub(var_0, var_2);
    // p_proj_sqr = wp.dot(p_proj, p_proj)                                                    <L 476>
    var_4 = wp::dot(var_3, var_3);
    // collide_side = wp.abs(x) < cylinder_half_height                                        <L 478>
    var_5 = wp::abs(var_1);
    var_6 = (var_5 < var_cylinder_half_height);
    // collide_cap = p_proj_sqr < (cylinder_radius * cylinder_radius)                         <L 479>
    var_7 = wp::mul(var_cylinder_radius, var_cylinder_radius);
    var_8 = (var_4 < var_7);
    // if collide_side and collide_cap:                                                       <L 481>
    var_9 = var_6 && var_8;
    if (var_9) {
        // dist_cap = cylinder_half_height - wp.abs(x)                                        <L 482>
        var_10 = wp::abs(var_1);
        var_11 = wp::sub(var_cylinder_half_height, var_10);
        // dist_radius = cylinder_radius - wp.sqrt(p_proj_sqr)                                <L 483>
        var_12 = wp::sqrt(var_4);
        var_13 = wp::sub(var_cylinder_radius, var_12);
        // if dist_cap < dist_radius:                                                         <L 485>
        var_14 = (var_11 < var_13);
        if (var_14) {
            // collide_side = False                                                           <L 486>
        }
        var_16 = wp::where(var_14, var_15, var_6);
        if (!var_14) {
            // collide_cap = False                                                            <L 488>
        }
        var_18 = wp::where(var_14, var_8, var_17);
    }
    var_19 = wp::where(var_9, var_16, var_6);
    var_20 = wp::where(var_9, var_18, var_8);
    // if collide_side:                                                                       <L 491>
    if (var_19) {
        // pos_target = cylinder_pos + a_proj                                                 <L 492>
        var_21 = wp::add(var_cylinder_pos, var_2);
        // return sphere_sphere(sphere_pos, sphere_radius, pos_target, cylinder_radius)       <L 493>
        sphere_sphere_0(var_sphere_pos, var_sphere_radius, var_21, var_cylinder_radius, var_22, var_23, var_24);
        ret_0 = var_22;
        ret_1 = var_23;
        ret_2 = var_24;
        return;
    }
    if (!var_19) {
        // elif collide_cap:                                                                  <L 495>
        if (var_20) {
            // if x > 0.0:                                                                    <L 496>
            var_26 = (var_1 > var_25);
            if (var_26) {
                // pos_cap = cylinder_pos + cylinder_axis * cylinder_half_height              <L 498>
                var_27 = wp::mul(var_cylinder_axis, var_cylinder_half_height);
                var_28 = wp::add(var_cylinder_pos, var_27);
                // plane_normal = cylinder_axis                                               <L 499>
                var_29 = wp::copy(var_cylinder_axis);
            }
            if (!var_26) {
                // pos_cap = cylinder_pos - cylinder_axis * cylinder_half_height              <L 502>
                var_30 = wp::mul(var_cylinder_axis, var_cylinder_half_height);
                var_31 = wp::sub(var_cylinder_pos, var_30);
                // plane_normal = -cylinder_axis                                              <L 503>
                var_32 = wp::neg(var_cylinder_axis);
            }
            var_33 = wp::where(var_26, var_28, var_31);
            var_34 = wp::where(var_26, var_29, var_32);
            // dist, pos = plane_sphere(plane_normal, pos_cap, sphere_pos, sphere_radius)       <L 505>
            plane_sphere_0(var_34, var_33, var_sphere_pos, var_sphere_radius, var_35, var_36);
            // return dist, pos, -plane_normal  # flip normal after position calculation       <L 506>
            var_37 = wp::neg(var_34);
            ret_0 = var_35;
            ret_1 = var_36;
            ret_2 = var_37;
            return;
        }
        if (!var_20) {
            // inv_len = safe_div(1.0, wp.sqrt(p_proj_sqr))                                   <L 509>
            var_39 = wp::sqrt(var_4);
            var_40 = safe_div_1(var_38, var_39);
            // p_proj = p_proj * (cylinder_radius * inv_len)                                  <L 510>
            var_41 = wp::mul(var_cylinder_radius, var_40);
            var_42 = wp::mul(var_3, var_41);
            // cap_offset = cylinder_axis * (wp.sign(x) * cylinder_half_height)               <L 512>
            var_43 = wp::sign(var_1);
            var_44 = wp::mul(var_43, var_cylinder_half_height);
            var_45 = wp::mul(var_cylinder_axis, var_44);
            // pos_corner = cylinder_pos + cap_offset + p_proj                                <L 513>
            var_46 = wp::add(var_cylinder_pos, var_45);
            var_47 = wp::add(var_46, var_42);
            // return sphere_sphere(sphere_pos, sphere_radius, pos_corner, 0.0)               <L 515>
            sphere_sphere_0(var_sphere_pos, var_sphere_radius, var_47, var_48, var_49, var_50, var_51);
            ret_0 = var_49;
            ret_1 = var_50;
            ret_2 = var_51;
            return;
        }
        var_52 = wp::where(var_20, var_3, var_42);
    }
    var_53 = wp::where(var_19, var_3, var_52);
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_primitive.py:882
static CUDA_CALLABLE void sphere_cylinder_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_3242f8a8 var_sphere,
    Geom_3242f8a8 var_cylinder,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
    wp::array_t<wp::float32> var_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_out,
    wp::array_t<wp::float32> var_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_solimp_out,
    wp::array_t<wp::int32> var_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::int32> var_nacon_out)
{
    //---------
    // primal vars
    wp::mat_t<3, 3, wp::float32>* var_0;
    const wp::int32 var_1 = 0;
    const wp::int32 var_2 = 2;
    wp::float32 var_3;
    wp::mat_t<3, 3, wp::float32> var_4;
    wp::mat_t<3, 3, wp::float32>* var_5;
    const wp::int32 var_6 = 1;
    const wp::int32 var_7 = 2;
    wp::float32 var_8;
    wp::mat_t<3, 3, wp::float32> var_9;
    wp::mat_t<3, 3, wp::float32>* var_10;
    const wp::int32 var_11 = 2;
    const wp::int32 var_12 = 2;
    wp::float32 var_13;
    wp::mat_t<3, 3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32>* var_16;
    wp::vec_t<3, wp::float32>* var_17;
    const wp::int32 var_18 = 0;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32>* var_21;
    wp::vec_t<3, wp::float32>* var_22;
    const wp::int32 var_23 = 0;
    wp::float32 var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32>* var_26;
    const wp::int32 var_27 = 1;
    wp::float32 var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::float32 var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::vec_t<3, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    const wp::int32 var_35 = 0;
    wp::mat_t<3, 3, wp::float32> var_36;
    wp::int32 var_37;
    //---------
    // forward
    // def sphere_cylinder_wrapper(                                                           <L 883>
    // cylinder_axis = wp.vec3(cylinder.rot[0, 2], cylinder.rot[1, 2], cylinder.rot[2, 2])       <L 918>
    var_0 = &(var_cylinder.rot);
    var_4 = wp::load(var_0);
    var_3 = wp::extract(var_4, var_1, var_2);
    var_5 = &(var_cylinder.rot);
    var_9 = wp::load(var_5);
    var_8 = wp::extract(var_9, var_6, var_7);
    var_10 = &(var_cylinder.rot);
    var_14 = wp::load(var_10);
    var_13 = wp::extract(var_14, var_11, var_12);
    var_15 = wp::vec_t<3, wp::float32>(var_3, var_8, var_13);
    // dist, pos, normal = sphere_cylinder(                                                   <L 920>
    // sphere.pos,                                                                            <L 921>
    var_16 = &(var_sphere.pos);
    // sphere.size[0],  # sphere radius                                                       <L 922>
    var_17 = &(var_sphere.size);
    var_20 = wp::load(var_17);
    var_19 = wp::extract(var_20, var_18);
    // cylinder.pos,                                                                          <L 923>
    var_21 = &(var_cylinder.pos);
    // cylinder_axis,                                                                         <L 924>
    // cylinder.size[0],  # cylinder radius                                                   <L 925>
    var_22 = &(var_cylinder.size);
    var_25 = wp::load(var_22);
    var_24 = wp::extract(var_25, var_23);
    // cylinder.size[1],  # cylinder half_height                                              <L 926>
    var_26 = &(var_cylinder.size);
    var_29 = wp::load(var_26);
    var_28 = wp::extract(var_29, var_27);
    var_33 = wp::load(var_16);
    var_34 = wp::load(var_21);
    sphere_cylinder_0(var_33, var_19, var_34, var_15, var_24, var_28, var_30, var_31, var_32);
    // write_contact(                                                                         <L 929>
    // naconmax_in,                                                                           <L 930>
    // 0,                                                                                     <L 931>
    // dist,                                                                                  <L 932>
    // pos,                                                                                   <L 933>
    // make_frame(normal),                                                                    <L 934>
    var_36 = make_frame_0(var_32);
    // margin,                                                                                <L 935>
    // gap,                                                                                   <L 936>
    // condim,                                                                                <L 937>
    // friction,                                                                              <L 938>
    // solref,                                                                                <L 939>
    // solreffriction,                                                                        <L 940>
    // solimp,                                                                                <L 941>
    // geoms,                                                                                 <L 942>
    // pairid,                                                                                <L 943>
    // worldid,                                                                               <L 944>
    // contact_dist_out,                                                                      <L 945>
    // contact_pos_out,                                                                       <L 946>
    // contact_frame_out,                                                                     <L 947>
    // contact_includemargin_out,                                                             <L 948>
    // contact_friction_out,                                                                  <L 949>
    // contact_solref_out,                                                                    <L 950>
    // contact_solreffriction_out,                                                            <L 951>
    // contact_solimp_out,                                                                    <L 952>
    // contact_dim_out,                                                                       <L 953>
    // contact_geom_out,                                                                      <L 954>
    // contact_efc_address_out,                                                               <L 955>
    // contact_worldid_out,                                                                   <L 956>
    // contact_type_out,                                                                      <L 957>
    // contact_geomcollisionid_out,                                                           <L 958>
    // nacon_out,                                                                             <L 959>
    var_37 = write_contact_0(var_naconmax_in, var_35, var_30, var_31, var_36, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_primitive_core.py:0
static CUDA_CALLABLE void normalize_with_norm_1(
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_primitive_core.py:1102
static CUDA_CALLABLE void sphere_box_0(
    wp::vec_t<3, wp::float32> var_sphere_pos,
    wp::float32 var_sphere_radius,
    wp::vec_t<3, wp::float32> var_box_pos,
    wp::mat_t<3, 3, wp::float32> var_box_rot,
    wp::vec_t<3, wp::float32> var_box_size,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::mat_t<3, 3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::float32 var_8;
    const wp::float32 var_9 = 1e-15;
    bool var_10;
    const wp::float32 var_11 = 2.0;
    const wp::int32 var_12 = 0;
    wp::float32 var_13;
    const wp::int32 var_14 = 1;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::int32 var_17 = 2;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    const wp::int32 var_21 = 0;
    wp::int32 var_22;
    const wp::int32 var_23 = 0;
    const wp::int32 var_24 = 2;
    wp::int32 var_25;
    const wp::float32 var_26 = 1.0;
    const wp::float32 var_27 = 1.0;
    const wp::float32 var_28 = -1.0;
    wp::float32 var_29;
    const wp::int32 var_30 = 2;
    wp::int32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    const wp::int32 var_34 = 2;
    wp::int32 var_35;
    wp::float32 var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    bool var_39;
    wp::float32 var_40;
    wp::int32 var_41;
    wp::float32 var_42;
    wp::int32 var_43;
    const wp::int32 var_44 = 1;
    const wp::int32 var_45 = 2;
    wp::int32 var_46;
    const wp::float32 var_47 = 1.0;
    const wp::float32 var_48 = 1.0;
    const wp::float32 var_49 = -1.0;
    wp::float32 var_50;
    const wp::int32 var_51 = 2;
    wp::int32 var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    const wp::int32 var_55 = 2;
    wp::int32 var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    bool var_60;
    wp::float32 var_61;
    wp::int32 var_62;
    wp::float32 var_63;
    wp::int32 var_64;
    const wp::int32 var_65 = 2;
    const wp::int32 var_66 = 2;
    wp::int32 var_67;
    const wp::float32 var_68 = 1.0;
    const wp::float32 var_69 = 1.0;
    const wp::float32 var_70 = -1.0;
    wp::float32 var_71;
    const wp::int32 var_72 = 2;
    wp::int32 var_73;
    wp::float32 var_74;
    wp::float32 var_75;
    const wp::int32 var_76 = 2;
    wp::int32 var_77;
    wp::float32 var_78;
    wp::float32 var_79;
    wp::float32 var_80;
    bool var_81;
    wp::float32 var_82;
    wp::int32 var_83;
    wp::float32 var_84;
    wp::int32 var_85;
    const wp::int32 var_86 = 3;
    const wp::int32 var_87 = 2;
    wp::int32 var_88;
    const wp::float32 var_89 = 1.0;
    const wp::float32 var_90 = 1.0;
    const wp::float32 var_91 = -1.0;
    wp::float32 var_92;
    const wp::int32 var_93 = 2;
    wp::int32 var_94;
    wp::float32 var_95;
    wp::float32 var_96;
    const wp::int32 var_97 = 2;
    wp::int32 var_98;
    wp::float32 var_99;
    wp::float32 var_100;
    wp::float32 var_101;
    bool var_102;
    wp::float32 var_103;
    wp::int32 var_104;
    wp::float32 var_105;
    wp::int32 var_106;
    const wp::int32 var_107 = 4;
    const wp::int32 var_108 = 2;
    wp::int32 var_109;
    const wp::float32 var_110 = 1.0;
    const wp::float32 var_111 = 1.0;
    const wp::float32 var_112 = -1.0;
    wp::float32 var_113;
    const wp::int32 var_114 = 2;
    wp::int32 var_115;
    wp::float32 var_116;
    wp::float32 var_117;
    const wp::int32 var_118 = 2;
    wp::int32 var_119;
    wp::float32 var_120;
    wp::float32 var_121;
    wp::float32 var_122;
    bool var_123;
    wp::float32 var_124;
    wp::int32 var_125;
    wp::float32 var_126;
    wp::int32 var_127;
    const wp::int32 var_128 = 5;
    const wp::int32 var_129 = 2;
    wp::int32 var_130;
    const wp::float32 var_131 = 1.0;
    const wp::float32 var_132 = 1.0;
    const wp::float32 var_133 = -1.0;
    wp::float32 var_134;
    const wp::int32 var_135 = 2;
    wp::int32 var_136;
    wp::float32 var_137;
    wp::float32 var_138;
    const wp::int32 var_139 = 2;
    wp::int32 var_140;
    wp::float32 var_141;
    wp::float32 var_142;
    wp::float32 var_143;
    bool var_144;
    wp::float32 var_145;
    wp::int32 var_146;
    wp::float32 var_147;
    wp::int32 var_148;
    const wp::float32 var_149 = 0.0;
    wp::vec_t<3, wp::float32> var_150;
    const wp::int32 var_151 = 2;
    wp::int32 var_152;
    const wp::float32 var_153 = 1.0;
    const wp::float32 var_154 = -1.0;
    const wp::float32 var_155 = 1.0;
    wp::float32 var_156;
    const wp::int32 var_157 = 2;
    wp::int32 var_158;
    wp::float32 var_159;
    wp::vec_t<3, wp::float32> var_160;
    const wp::float32 var_161 = 2.0;
    wp::vec_t<3, wp::float32> var_162;
    wp::vec_t<3, wp::float32> var_163;
    wp::vec_t<3, wp::float32> var_164;
    wp::float32 var_165;
    wp::float32 var_166;
    wp::vec_t<3, wp::float32> var_167;
    wp::vec_t<3, wp::float32> var_168;
    const wp::float32 var_169 = 0.5;
    wp::vec_t<3, wp::float32> var_170;
    wp::vec_t<3, wp::float32> var_171;
    wp::vec_t<3, wp::float32> var_172;
    wp::float32 var_173;
    wp::vec_t<3, wp::float32> var_174;
    wp::vec_t<3, wp::float32> var_175;
    wp::float32 var_176;
    wp::vec_t<3, wp::float32> var_177;
    wp::vec_t<3, wp::float32> var_178;
    //---------
    // forward
    // def sphere_box(                                                                        <L 1103>
    // center = wp.transpose(box_rot) @ (sphere_pos - box_pos)                                <L 1125>
    var_0 = wp::transpose(var_box_rot);
    var_1 = wp::sub(var_sphere_pos, var_box_pos);
    var_2 = wp::mul(var_0, var_1);
    // clamped = wp.max(-box_size, wp.min(box_size, center))                                  <L 1127>
    var_3 = wp::neg(var_box_size);
    var_4 = wp::min(var_box_size, var_2);
    var_5 = wp::max(var_3, var_4);
    // clamped_dir, dist = normalize_with_norm(clamped - center)                              <L 1128>
    var_6 = wp::sub(var_5, var_2);
    normalize_with_norm_1(var_6, var_7, var_8);
    // if dist <= MJ_MINVAL:                                                                  <L 1131>
    var_10 = (var_8 <= var_9);
    if (var_10) {
        // closest = 2.0 * (box_size[0] + box_size[1] + box_size[2])                          <L 1132>
        var_13 = wp::extract(var_box_size, var_12);
        var_15 = wp::extract(var_box_size, var_14);
        var_16 = wp::add(var_13, var_15);
        var_18 = wp::extract(var_box_size, var_17);
        var_19 = wp::add(var_16, var_18);
        var_20 = wp::mul(var_11, var_19);
        // k = wp.int32(0)                                                                    <L 1133>
        var_22 = wp::int32(var_21);
        // for i in range(6):                                                                 <L 1134>
        // face_dist = wp.abs(wp.where(i % 2, 1.0, -1.0) * box_size[i // 2] - center[i // 2])       <L 1135>
        var_25 = wp::mod(var_23, var_24);
        var_29 = wp::where(var_25, var_26, var_28);
        var_31 = wp::floordiv(var_23, var_30);
        var_32 = wp::extract(var_box_size, var_31);
        var_33 = wp::mul(var_29, var_32);
        var_35 = wp::floordiv(var_23, var_34);
        var_36 = wp::extract(var_2, var_35);
        var_37 = wp::sub(var_33, var_36);
        var_38 = wp::abs(var_37);
        // if closest > face_dist:                                                            <L 1136>
        var_39 = (var_20 > var_38);
        if (var_39) {
            // closest = face_dist                                                            <L 1137>
            var_40 = wp::copy(var_38);
            // k = i                                                                          <L 1138>
            var_41 = wp::copy(var_23);
        }
        var_42 = wp::where(var_39, var_40, var_20);
        var_43 = wp::where(var_39, var_41, var_22);
        // face_dist = wp.abs(wp.where(i % 2, 1.0, -1.0) * box_size[i // 2] - center[i // 2])       <L 1135>
        var_46 = wp::mod(var_44, var_45);
        var_50 = wp::where(var_46, var_47, var_49);
        var_52 = wp::floordiv(var_44, var_51);
        var_53 = wp::extract(var_box_size, var_52);
        var_54 = wp::mul(var_50, var_53);
        var_56 = wp::floordiv(var_44, var_55);
        var_57 = wp::extract(var_2, var_56);
        var_58 = wp::sub(var_54, var_57);
        var_59 = wp::abs(var_58);
        // if closest > face_dist:                                                            <L 1136>
        var_60 = (var_42 > var_59);
        if (var_60) {
            // closest = face_dist                                                            <L 1137>
            var_61 = wp::copy(var_59);
            // k = i                                                                          <L 1138>
            var_62 = wp::copy(var_44);
        }
        var_63 = wp::where(var_60, var_61, var_42);
        var_64 = wp::where(var_60, var_62, var_43);
        // face_dist = wp.abs(wp.where(i % 2, 1.0, -1.0) * box_size[i // 2] - center[i // 2])       <L 1135>
        var_67 = wp::mod(var_65, var_66);
        var_71 = wp::where(var_67, var_68, var_70);
        var_73 = wp::floordiv(var_65, var_72);
        var_74 = wp::extract(var_box_size, var_73);
        var_75 = wp::mul(var_71, var_74);
        var_77 = wp::floordiv(var_65, var_76);
        var_78 = wp::extract(var_2, var_77);
        var_79 = wp::sub(var_75, var_78);
        var_80 = wp::abs(var_79);
        // if closest > face_dist:                                                            <L 1136>
        var_81 = (var_63 > var_80);
        if (var_81) {
            // closest = face_dist                                                            <L 1137>
            var_82 = wp::copy(var_80);
            // k = i                                                                          <L 1138>
            var_83 = wp::copy(var_65);
        }
        var_84 = wp::where(var_81, var_82, var_63);
        var_85 = wp::where(var_81, var_83, var_64);
        // face_dist = wp.abs(wp.where(i % 2, 1.0, -1.0) * box_size[i // 2] - center[i // 2])       <L 1135>
        var_88 = wp::mod(var_86, var_87);
        var_92 = wp::where(var_88, var_89, var_91);
        var_94 = wp::floordiv(var_86, var_93);
        var_95 = wp::extract(var_box_size, var_94);
        var_96 = wp::mul(var_92, var_95);
        var_98 = wp::floordiv(var_86, var_97);
        var_99 = wp::extract(var_2, var_98);
        var_100 = wp::sub(var_96, var_99);
        var_101 = wp::abs(var_100);
        // if closest > face_dist:                                                            <L 1136>
        var_102 = (var_84 > var_101);
        if (var_102) {
            // closest = face_dist                                                            <L 1137>
            var_103 = wp::copy(var_101);
            // k = i                                                                          <L 1138>
            var_104 = wp::copy(var_86);
        }
        var_105 = wp::where(var_102, var_103, var_84);
        var_106 = wp::where(var_102, var_104, var_85);
        // face_dist = wp.abs(wp.where(i % 2, 1.0, -1.0) * box_size[i // 2] - center[i // 2])       <L 1135>
        var_109 = wp::mod(var_107, var_108);
        var_113 = wp::where(var_109, var_110, var_112);
        var_115 = wp::floordiv(var_107, var_114);
        var_116 = wp::extract(var_box_size, var_115);
        var_117 = wp::mul(var_113, var_116);
        var_119 = wp::floordiv(var_107, var_118);
        var_120 = wp::extract(var_2, var_119);
        var_121 = wp::sub(var_117, var_120);
        var_122 = wp::abs(var_121);
        // if closest > face_dist:                                                            <L 1136>
        var_123 = (var_105 > var_122);
        if (var_123) {
            // closest = face_dist                                                            <L 1137>
            var_124 = wp::copy(var_122);
            // k = i                                                                          <L 1138>
            var_125 = wp::copy(var_107);
        }
        var_126 = wp::where(var_123, var_124, var_105);
        var_127 = wp::where(var_123, var_125, var_106);
        // face_dist = wp.abs(wp.where(i % 2, 1.0, -1.0) * box_size[i // 2] - center[i // 2])       <L 1135>
        var_130 = wp::mod(var_128, var_129);
        var_134 = wp::where(var_130, var_131, var_133);
        var_136 = wp::floordiv(var_128, var_135);
        var_137 = wp::extract(var_box_size, var_136);
        var_138 = wp::mul(var_134, var_137);
        var_140 = wp::floordiv(var_128, var_139);
        var_141 = wp::extract(var_2, var_140);
        var_142 = wp::sub(var_138, var_141);
        var_143 = wp::abs(var_142);
        // if closest > face_dist:                                                            <L 1136>
        var_144 = (var_126 > var_143);
        if (var_144) {
            // closest = face_dist                                                            <L 1137>
            var_145 = wp::copy(var_143);
            // k = i                                                                          <L 1138>
            var_146 = wp::copy(var_128);
        }
        var_147 = wp::where(var_144, var_145, var_126);
        var_148 = wp::where(var_144, var_146, var_127);
        // nearest = wp.vec3(0.0)                                                             <L 1140>
        var_150 = wp::vec_t<3, wp::float32>(var_149);
        // nearest[k // 2] = wp.where(k % 2, -1.0, 1.0)                                       <L 1141>
        var_152 = wp::mod(var_148, var_151);
        var_156 = wp::where(var_152, var_154, var_155);
        var_158 = wp::floordiv(var_148, var_157);
        wp::assign_inplace(var_150, var_158, var_156);
        // pos = center + nearest * (sphere_radius - closest) / 2.0                           <L 1142>
        var_159 = wp::sub(var_sphere_radius, var_147);
        var_160 = wp::mul(var_150, var_159);
        var_162 = wp::div(var_160, var_161);
        var_163 = wp::add(var_2, var_162);
        // contact_normal = box_rot @ nearest                                                 <L 1143>
        var_164 = wp::mul(var_box_rot, var_150);
        // contact_distance = -closest - sphere_radius                                        <L 1144>
        var_165 = wp::neg(var_147);
        var_166 = wp::sub(var_165, var_sphere_radius);
    }
    if (!var_10) {
        // deepest = center + clamped_dir * sphere_radius                                     <L 1147>
        var_167 = wp::mul(var_7, var_sphere_radius);
        var_168 = wp::add(var_2, var_167);
        // pos = 0.5 * (clamped + deepest)                                                    <L 1148>
        var_170 = wp::add(var_5, var_168);
        var_171 = wp::mul(var_169, var_170);
        // contact_normal = box_rot @ clamped_dir                                             <L 1149>
        var_172 = wp::mul(var_box_rot, var_7);
        // contact_distance = dist - sphere_radius                                            <L 1150>
        var_173 = wp::sub(var_8, var_sphere_radius);
    }
    var_174 = wp::where(var_10, var_163, var_171);
    var_175 = wp::where(var_10, var_164, var_172);
    var_176 = wp::where(var_10, var_166, var_173);
    // contact_position = box_pos + box_rot @ pos                                             <L 1152>
    var_177 = wp::mul(var_box_rot, var_174);
    var_178 = wp::add(var_box_pos, var_177);
    // return contact_distance, contact_position, contact_normal                              <L 1154>
    ret_0 = var_176;
    ret_1 = var_178;
    ret_2 = var_175;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_primitive.py:1046
static CUDA_CALLABLE void sphere_box_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_3242f8a8 var_sphere,
    Geom_3242f8a8 var_box,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
    wp::array_t<wp::float32> var_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_out,
    wp::array_t<wp::float32> var_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_solimp_out,
    wp::array_t<wp::int32> var_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::int32> var_nacon_out)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32>* var_0;
    wp::vec_t<3, wp::float32>* var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32>* var_5;
    wp::mat_t<3, 3, wp::float32>* var_6;
    wp::vec_t<3, wp::float32>* var_7;
    wp::float32 var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::mat_t<3, 3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    const wp::int32 var_15 = 0;
    wp::mat_t<3, 3, wp::float32> var_16;
    wp::int32 var_17;
    //---------
    // forward
    // def sphere_box_wrapper(                                                                <L 1047>
    // dist, pos, normal = sphere_box(sphere.pos, sphere.size[0], box.pos, box.rot, box.size)       <L 1080>
    var_0 = &(var_sphere.pos);
    var_1 = &(var_sphere.size);
    var_4 = wp::load(var_1);
    var_3 = wp::extract(var_4, var_2);
    var_5 = &(var_box.pos);
    var_6 = &(var_box.rot);
    var_7 = &(var_box.size);
    var_11 = wp::load(var_0);
    var_12 = wp::load(var_5);
    var_13 = wp::load(var_6);
    var_14 = wp::load(var_7);
    sphere_box_0(var_11, var_3, var_12, var_13, var_14, var_8, var_9, var_10);
    // write_contact(                                                                         <L 1082>
    // naconmax_in,                                                                           <L 1083>
    // 0,                                                                                     <L 1084>
    // dist,                                                                                  <L 1085>
    // pos,                                                                                   <L 1086>
    // make_frame(normal),                                                                    <L 1087>
    var_16 = make_frame_0(var_10);
    // margin,                                                                                <L 1088>
    // gap,                                                                                   <L 1089>
    // condim,                                                                                <L 1090>
    // friction,                                                                              <L 1091>
    // solref,                                                                                <L 1092>
    // solreffriction,                                                                        <L 1093>
    // solimp,                                                                                <L 1094>
    // geoms,                                                                                 <L 1095>
    // pairid,                                                                                <L 1096>
    // worldid,                                                                               <L 1097>
    // contact_dist_out,                                                                      <L 1098>
    // contact_pos_out,                                                                       <L 1099>
    // contact_frame_out,                                                                     <L 1100>
    // contact_includemargin_out,                                                             <L 1101>
    // contact_friction_out,                                                                  <L 1102>
    // contact_solref_out,                                                                    <L 1103>
    // contact_solreffriction_out,                                                            <L 1104>
    // contact_solimp_out,                                                                    <L 1105>
    // contact_dim_out,                                                                       <L 1106>
    // contact_geom_out,                                                                      <L 1107>
    // contact_efc_address_out,                                                               <L 1108>
    // contact_worldid_out,                                                                   <L 1109>
    // contact_type_out,                                                                      <L 1110>
    // contact_geomcollisionid_out,                                                           <L 1111>
    // nacon_out,                                                                             <L 1112>
    var_17 = write_contact_0(var_naconmax_in, var_15, var_8, var_9, var_16, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_primitive_core.py:181
static CUDA_CALLABLE void capsule_capsule_0(
    wp::vec_t<3, wp::float32> var_cap1_pos,
    wp::vec_t<3, wp::float32> var_cap1_axis,
    wp::float32 var_cap1_radius,
    wp::float32 var_cap1_half_length,
    wp::vec_t<3, wp::float32> var_cap2_pos,
    wp::vec_t<3, wp::float32> var_cap2_axis,
    wp::float32 var_cap2_radius,
    wp::float32 var_cap2_half_length,
    wp::float32 var_margin,
    wp::vec_t<2, wp::float32> & ret_0,
    wp::mat_t<2, 3, wp::float32> & ret_1,
    wp::mat_t<2, 3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    const wp::float32 var_0 = INFINITY;
    const wp::float32 var_1 = INFINITY;
    const wp::float32 var_2 = INFINITY;
    const wp::float32 var_3 = INFINITY;
    wp::vec_t<2, wp::float32> var_4;
    wp::mat_t<2, 3, wp::float32> var_5;
    wp::mat_t<2, 3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    const wp::float32 var_21 = 1e-15;
    bool var_22;
    const wp::float32 var_23 = 1.0;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    const wp::float32 var_33 = 1.0;
    bool var_34;
    const wp::float32 var_35 = 1.0;
    wp::float32 var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::float32 var_40 = 1.0;
    const wp::float32 var_41 = -1.0;
    bool var_42;
    const wp::float32 var_43 = 1.0;
    const wp::float32 var_44 = -1.0;
    wp::float32 var_45;
    wp::float32 var_46;
    wp::float32 var_47;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    const wp::float32 var_51 = 1.0;
    bool var_52;
    const wp::float32 var_53 = 1.0;
    wp::float32 var_54;
    wp::float32 var_55;
    const wp::float32 var_56 = 1.0;
    const wp::float32 var_57 = -1.0;
    const wp::float32 var_58 = 1.0;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::float32 var_61;
    const wp::float32 var_62 = 1.0;
    const wp::float32 var_63 = -1.0;
    bool var_64;
    const wp::float32 var_65 = 1.0;
    const wp::float32 var_66 = -1.0;
    wp::float32 var_67;
    wp::float32 var_68;
    const wp::float32 var_69 = 1.0;
    const wp::float32 var_70 = -1.0;
    const wp::float32 var_71 = 1.0;
    wp::float32 var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    wp::float32 var_75;
    wp::float32 var_76;
    wp::vec_t<3, wp::float32> var_77;
    wp::vec_t<3, wp::float32> var_78;
    wp::vec_t<3, wp::float32> var_79;
    wp::vec_t<3, wp::float32> var_80;
    wp::float32 var_81;
    wp::vec_t<3, wp::float32> var_82;
    wp::vec_t<3, wp::float32> var_83;
    bool var_84;
    const wp::int32 var_85 = 0;
    const wp::int32 var_86 = 0;
    const wp::int32 var_87 = 0;
    const wp::int32 var_88 = 0;
    wp::vec_t<3, wp::float32> var_89;
    wp::float32 var_90;
    wp::float32 var_91;
    const wp::float32 var_92 = 1.0;
    const wp::float32 var_93 = -1.0;
    const wp::float32 var_94 = 1.0;
    wp::float32 var_95;
    wp::vec_t<3, wp::float32> var_96;
    wp::vec_t<3, wp::float32> var_97;
    wp::float32 var_98;
    wp::vec_t<3, wp::float32> var_99;
    wp::vec_t<3, wp::float32> var_100;
    bool var_101;
    const wp::int32 var_102 = 1;
    wp::int32 var_103;
    wp::int32 var_104;
    wp::vec_t<3, wp::float32> var_105;
    wp::float32 var_106;
    wp::float32 var_107;
    const wp::float32 var_108 = 1.0;
    const wp::float32 var_109 = -1.0;
    const wp::float32 var_110 = 1.0;
    wp::float32 var_111;
    wp::vec_t<3, wp::float32> var_112;
    wp::vec_t<3, wp::float32> var_113;
    wp::float32 var_114;
    wp::vec_t<3, wp::float32> var_115;
    wp::vec_t<3, wp::float32> var_116;
    bool var_117;
    const wp::int32 var_118 = 1;
    wp::int32 var_119;
    wp::int32 var_120;
    const wp::int32 var_121 = 2;
    bool var_122;
    wp::vec_t<3, wp::float32> var_123;
    wp::float32 var_124;
    wp::float32 var_125;
    const wp::float32 var_126 = 1.0;
    const wp::float32 var_127 = -1.0;
    const wp::float32 var_128 = 1.0;
    wp::float32 var_129;
    wp::vec_t<3, wp::float32> var_130;
    wp::vec_t<3, wp::float32> var_131;
    wp::float32 var_132;
    wp::vec_t<3, wp::float32> var_133;
    wp::vec_t<3, wp::float32> var_134;
    bool var_135;
    const wp::int32 var_136 = 1;
    wp::int32 var_137;
    wp::int32 var_138;
    wp::float32 var_139;
    wp::vec_t<3, wp::float32> var_140;
    wp::vec_t<3, wp::float32> var_141;
    wp::float32 var_142;
    wp::vec_t<3, wp::float32> var_143;
    wp::vec_t<3, wp::float32> var_144;
    wp::int32 var_145;
    const wp::int32 var_146 = 2;
    bool var_147;
    wp::vec_t<3, wp::float32> var_148;
    wp::float32 var_149;
    wp::float32 var_150;
    const wp::float32 var_151 = 1.0;
    const wp::float32 var_152 = -1.0;
    const wp::float32 var_153 = 1.0;
    wp::float32 var_154;
    wp::vec_t<3, wp::float32> var_155;
    wp::vec_t<3, wp::float32> var_156;
    wp::float32 var_157;
    wp::vec_t<3, wp::float32> var_158;
    wp::vec_t<3, wp::float32> var_159;
    bool var_160;
    wp::float32 var_161;
    wp::vec_t<3, wp::float32> var_162;
    wp::vec_t<3, wp::float32> var_163;
    wp::float32 var_164;
    wp::vec_t<3, wp::float32> var_165;
    wp::vec_t<3, wp::float32> var_166;
    wp::float32 var_167;
    wp::float32 var_168;
    wp::vec_t<3, wp::float32> var_169;
    wp::vec_t<3, wp::float32> var_170;
    wp::float32 var_171;
    wp::vec_t<3, wp::float32> var_172;
    wp::vec_t<3, wp::float32> var_173;
    //---------
    // forward
    // def capsule_capsule(                                                                   <L 182>
    // contact_dist = wp.vec2(wp.inf, wp.inf)                                                 <L 212>
    var_4 = wp::vec_t<2, wp::float32>(var_1, var_3);
    // contact_pos = mat23f()                                                                 <L 213>
    var_5 = wp::mat_t<2, 3, wp::float32>();
    // contact_normal = mat23f()                                                              <L 214>
    var_6 = wp::mat_t<2, 3, wp::float32>();
    // axis1 = cap1_axis * cap1_half_length                                                   <L 217>
    var_7 = wp::mul(var_cap1_axis, var_cap1_half_length);
    // axis2 = cap2_axis * cap2_half_length                                                   <L 218>
    var_8 = wp::mul(var_cap2_axis, var_cap2_half_length);
    // dif = cap1_pos - cap2_pos                                                              <L 219>
    var_9 = wp::sub(var_cap1_pos, var_cap2_pos);
    // ma = wp.dot(axis1, axis1)                                                              <L 222>
    var_10 = wp::dot(var_7, var_7);
    // mb = -wp.dot(axis1, axis2)                                                             <L 223>
    var_11 = wp::dot(var_7, var_8);
    var_12 = wp::neg(var_11);
    // mc = wp.dot(axis2, axis2)                                                              <L 224>
    var_13 = wp::dot(var_8, var_8);
    // u = -wp.dot(axis1, dif)                                                                <L 225>
    var_14 = wp::dot(var_7, var_9);
    var_15 = wp::neg(var_14);
    // v = wp.dot(axis2, dif)                                                                 <L 226>
    var_16 = wp::dot(var_8, var_9);
    // det = ma * mc - mb * mb                                                                <L 227>
    var_17 = wp::mul(var_10, var_13);
    var_18 = wp::mul(var_12, var_12);
    var_19 = wp::sub(var_17, var_18);
    // if wp.abs(det) >= MJ_MINVAL:                                                           <L 230>
    var_20 = wp::abs(var_19);
    var_22 = (var_20 >= var_21);
    if (var_22) {
        // inv_det = 1.0 / det                                                                <L 231>
        var_24 = wp::div(var_23, var_19);
        // x1 = (mc * u - mb * v) * inv_det                                                   <L 232>
        var_25 = wp::mul(var_13, var_15);
        var_26 = wp::mul(var_12, var_16);
        var_27 = wp::sub(var_25, var_26);
        var_28 = wp::mul(var_27, var_24);
        // x2 = (ma * v - mb * u) * inv_det                                                   <L 233>
        var_29 = wp::mul(var_10, var_16);
        var_30 = wp::mul(var_12, var_15);
        var_31 = wp::sub(var_29, var_30);
        var_32 = wp::mul(var_31, var_24);
        // if x1 > 1.0:                                                                       <L 235>
        var_34 = (var_28 > var_33);
        if (var_34) {
            // x1 = 1.0                                                                       <L 236>
            // x2 = (v - mb) / mc                                                             <L 237>
            var_36 = wp::sub(var_16, var_12);
            var_37 = wp::div(var_36, var_13);
        }
        var_38 = wp::where(var_34, var_35, var_28);
        var_39 = wp::where(var_34, var_37, var_32);
        if (!var_34) {
            // elif x1 < -1.0:                                                                <L 238>
            var_42 = (var_38 < var_41);
            if (var_42) {
                // x1 = -1.0                                                                  <L 239>
                // x2 = (v + mb) / mc                                                         <L 240>
                var_45 = wp::add(var_16, var_12);
                var_46 = wp::div(var_45, var_13);
            }
            var_47 = wp::where(var_42, var_44, var_38);
            var_48 = wp::where(var_42, var_46, var_39);
        }
        var_49 = wp::where(var_34, var_38, var_47);
        var_50 = wp::where(var_34, var_39, var_48);
        // if x2 > 1.0:                                                                       <L 242>
        var_52 = (var_50 > var_51);
        if (var_52) {
            // x2 = 1.0                                                                       <L 243>
            // x1 = wp.clamp((u - mb) / ma, -1.0, 1.0)                                        <L 244>
            var_54 = wp::sub(var_15, var_12);
            var_55 = wp::div(var_54, var_10);
            var_59 = wp::clamp(var_55, var_57, var_58);
        }
        var_60 = wp::where(var_52, var_59, var_49);
        var_61 = wp::where(var_52, var_53, var_50);
        if (!var_52) {
            // elif x2 < -1.0:                                                                <L 245>
            var_64 = (var_61 < var_63);
            if (var_64) {
                // x2 = -1.0                                                                  <L 246>
                // x1 = wp.clamp((u + mb) / ma, -1.0, 1.0)                                    <L 247>
                var_67 = wp::add(var_15, var_12);
                var_68 = wp::div(var_67, var_10);
                var_72 = wp::clamp(var_68, var_70, var_71);
            }
            var_73 = wp::where(var_64, var_72, var_60);
            var_74 = wp::where(var_64, var_66, var_61);
        }
        var_75 = wp::where(var_52, var_60, var_73);
        var_76 = wp::where(var_52, var_61, var_74);
        // vec1 = cap1_pos + axis1 * x1                                                       <L 250>
        var_77 = wp::mul(var_7, var_75);
        var_78 = wp::add(var_cap1_pos, var_77);
        // vec2 = cap2_pos + axis2 * x2                                                       <L 251>
        var_79 = wp::mul(var_8, var_76);
        var_80 = wp::add(var_cap2_pos, var_79);
        // dist, pos, normal = sphere_sphere(vec1, cap1_radius, vec2, cap2_radius)            <L 253>
        sphere_sphere_0(var_78, var_cap1_radius, var_80, var_cap2_radius, var_81, var_82, var_83);
        // if dist <= margin:                                                                 <L 254>
        var_84 = (var_81 <= var_margin);
        if (var_84) {
            // contact_dist[0] = dist                                                         <L 255>
            wp::assign_inplace(var_4, var_85, var_81);
            // contact_pos[0] = pos                                                           <L 256>
            wp::assign_inplace(var_5, var_86, var_82);
            // contact_normal[0] = normal                                                     <L 257>
            wp::assign_inplace(var_6, var_87, var_83);
        }
    }
    if (!var_22) {
        // contact_count = 0                                                                  <L 261>
        // vec1 = cap1_pos + axis1                                                            <L 264>
        var_89 = wp::add(var_cap1_pos, var_7);
        // x2 = wp.clamp((v - mb) / mc, -1.0, 1.0)                                            <L 265>
        var_90 = wp::sub(var_16, var_12);
        var_91 = wp::div(var_90, var_13);
        var_95 = wp::clamp(var_91, var_93, var_94);
        // vec2 = cap2_pos + axis2 * x2                                                       <L 266>
        var_96 = wp::mul(var_8, var_95);
        var_97 = wp::add(var_cap2_pos, var_96);
        // dist, pos, normal = sphere_sphere(vec1, cap1_radius, vec2, cap2_radius)            <L 267>
        sphere_sphere_0(var_89, var_cap1_radius, var_97, var_cap2_radius, var_98, var_99, var_100);
        // if dist <= margin:                                                                 <L 268>
        var_101 = (var_98 <= var_margin);
        if (var_101) {
            // contact_dist[contact_count] = dist                                             <L 269>
            wp::assign_inplace(var_4, var_88, var_98);
            // contact_pos[contact_count] = pos                                               <L 270>
            wp::assign_inplace(var_5, var_88, var_99);
            // contact_normal[contact_count] = normal                                         <L 271>
            wp::assign_inplace(var_6, var_88, var_100);
            // contact_count += 1                                                             <L 272>
            var_103 = wp::add(var_88, var_102);
        }
        var_104 = wp::where(var_101, var_103, var_88);
        // vec1 = cap1_pos - axis1                                                            <L 275>
        var_105 = wp::sub(var_cap1_pos, var_7);
        // x2 = wp.clamp((v + mb) / mc, -1.0, 1.0)                                            <L 276>
        var_106 = wp::add(var_16, var_12);
        var_107 = wp::div(var_106, var_13);
        var_111 = wp::clamp(var_107, var_109, var_110);
        // vec2 = cap2_pos + axis2 * x2                                                       <L 277>
        var_112 = wp::mul(var_8, var_111);
        var_113 = wp::add(var_cap2_pos, var_112);
        // dist, pos, normal = sphere_sphere(vec1, cap1_radius, vec2, cap2_radius)            <L 278>
        sphere_sphere_0(var_105, var_cap1_radius, var_113, var_cap2_radius, var_114, var_115, var_116);
        // if dist <= margin:                                                                 <L 279>
        var_117 = (var_114 <= var_margin);
        if (var_117) {
            // contact_dist[contact_count] = dist                                             <L 280>
            wp::assign_inplace(var_4, var_104, var_114);
            // contact_pos[contact_count] = pos                                               <L 281>
            wp::assign_inplace(var_5, var_104, var_115);
            // contact_normal[contact_count] = normal                                         <L 282>
            wp::assign_inplace(var_6, var_104, var_116);
            // contact_count += 1                                                             <L 283>
            var_119 = wp::add(var_104, var_118);
        }
        var_120 = wp::where(var_117, var_119, var_104);
        // if contact_count < 2:                                                              <L 286>
        var_122 = (var_120 < var_121);
        if (var_122) {
            // vec2 = cap2_pos + axis2                                                        <L 287>
            var_123 = wp::add(var_cap2_pos, var_8);
            // x1 = wp.clamp((u - mb) / ma, -1.0, 1.0)                                        <L 288>
            var_124 = wp::sub(var_15, var_12);
            var_125 = wp::div(var_124, var_10);
            var_129 = wp::clamp(var_125, var_127, var_128);
            // vec1 = cap1_pos + axis1 * x1                                                   <L 289>
            var_130 = wp::mul(var_7, var_129);
            var_131 = wp::add(var_cap1_pos, var_130);
            // dist, pos, normal = sphere_sphere(vec1, cap1_radius, vec2, cap2_radius)        <L 290>
            sphere_sphere_0(var_131, var_cap1_radius, var_123, var_cap2_radius, var_132, var_133, var_134);
            // if dist <= margin:                                                             <L 291>
            var_135 = (var_132 <= var_margin);
            if (var_135) {
                // contact_dist[contact_count] = dist                                         <L 292>
                wp::assign_inplace(var_4, var_120, var_132);
                // contact_pos[contact_count] = pos                                           <L 293>
                wp::assign_inplace(var_5, var_120, var_133);
                // contact_normal[contact_count] = normal                                     <L 294>
                wp::assign_inplace(var_6, var_120, var_134);
                // contact_count += 1                                                         <L 295>
                var_137 = wp::add(var_120, var_136);
            }
            var_138 = wp::where(var_135, var_137, var_120);
        }
        var_139 = wp::where(var_122, var_129, var_75);
        var_140 = wp::where(var_122, var_131, var_105);
        var_141 = wp::where(var_122, var_123, var_113);
        var_142 = wp::where(var_122, var_132, var_114);
        var_143 = wp::where(var_122, var_133, var_115);
        var_144 = wp::where(var_122, var_134, var_116);
        var_145 = wp::where(var_122, var_138, var_120);
        // if contact_count < 2:                                                              <L 298>
        var_147 = (var_145 < var_146);
        if (var_147) {
            // vec2 = cap2_pos - axis2                                                        <L 299>
            var_148 = wp::sub(var_cap2_pos, var_8);
            // x1 = wp.clamp((u + mb) / ma, -1.0, 1.0)                                        <L 300>
            var_149 = wp::add(var_15, var_12);
            var_150 = wp::div(var_149, var_10);
            var_154 = wp::clamp(var_150, var_152, var_153);
            // vec1 = cap1_pos + axis1 * x1                                                   <L 301>
            var_155 = wp::mul(var_7, var_154);
            var_156 = wp::add(var_cap1_pos, var_155);
            // dist, pos, normal = sphere_sphere(vec1, cap1_radius, vec2, cap2_radius)        <L 302>
            sphere_sphere_0(var_156, var_cap1_radius, var_148, var_cap2_radius, var_157, var_158, var_159);
            // if dist <= margin:                                                             <L 303>
            var_160 = (var_157 <= var_margin);
            if (var_160) {
                // contact_dist[contact_count] = dist                                         <L 304>
                wp::assign_inplace(var_4, var_145, var_157);
                // contact_pos[contact_count] = pos                                           <L 305>
                wp::assign_inplace(var_5, var_145, var_158);
                // contact_normal[contact_count] = normal                                     <L 306>
                wp::assign_inplace(var_6, var_145, var_159);
            }
        }
        var_161 = wp::where(var_147, var_154, var_139);
        var_162 = wp::where(var_147, var_156, var_140);
        var_163 = wp::where(var_147, var_148, var_141);
        var_164 = wp::where(var_147, var_157, var_142);
        var_165 = wp::where(var_147, var_158, var_143);
        var_166 = wp::where(var_147, var_159, var_144);
    }
    var_167 = wp::where(var_22, var_75, var_161);
    var_168 = wp::where(var_22, var_76, var_111);
    var_169 = wp::where(var_22, var_78, var_162);
    var_170 = wp::where(var_22, var_80, var_163);
    var_171 = wp::where(var_22, var_81, var_164);
    var_172 = wp::where(var_22, var_82, var_165);
    var_173 = wp::where(var_22, var_83, var_166);
    // return contact_dist, contact_pos, contact_normal                                       <L 308>
    ret_0 = var_4;
    ret_1 = var_5;
    ret_2 = var_6;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_primitive.py:497
static CUDA_CALLABLE void capsule_capsule_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_3242f8a8 var_cap1,
    Geom_3242f8a8 var_cap2,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
    wp::array_t<wp::float32> var_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_out,
    wp::array_t<wp::float32> var_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_solimp_out,
    wp::array_t<wp::int32> var_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::int32> var_nacon_out)
{
    //---------
    // primal vars
    wp::mat_t<3, 3, wp::float32>* var_0;
    const wp::int32 var_1 = 0;
    const wp::int32 var_2 = 2;
    wp::float32 var_3;
    wp::mat_t<3, 3, wp::float32> var_4;
    wp::mat_t<3, 3, wp::float32>* var_5;
    const wp::int32 var_6 = 1;
    const wp::int32 var_7 = 2;
    wp::float32 var_8;
    wp::mat_t<3, 3, wp::float32> var_9;
    wp::mat_t<3, 3, wp::float32>* var_10;
    const wp::int32 var_11 = 2;
    const wp::int32 var_12 = 2;
    wp::float32 var_13;
    wp::mat_t<3, 3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::mat_t<3, 3, wp::float32>* var_16;
    const wp::int32 var_17 = 0;
    const wp::int32 var_18 = 2;
    wp::float32 var_19;
    wp::mat_t<3, 3, wp::float32> var_20;
    wp::mat_t<3, 3, wp::float32>* var_21;
    const wp::int32 var_22 = 1;
    const wp::int32 var_23 = 2;
    wp::float32 var_24;
    wp::mat_t<3, 3, wp::float32> var_25;
    wp::mat_t<3, 3, wp::float32>* var_26;
    const wp::int32 var_27 = 2;
    const wp::int32 var_28 = 2;
    wp::float32 var_29;
    wp::mat_t<3, 3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32>* var_32;
    wp::vec_t<3, wp::float32>* var_33;
    const wp::int32 var_34 = 0;
    wp::float32 var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32>* var_37;
    const wp::int32 var_38 = 1;
    wp::float32 var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::vec_t<3, wp::float32>* var_41;
    wp::vec_t<3, wp::float32>* var_42;
    const wp::int32 var_43 = 0;
    wp::float32 var_44;
    wp::vec_t<3, wp::float32> var_45;
    wp::vec_t<3, wp::float32>* var_46;
    const wp::int32 var_47 = 1;
    wp::float32 var_48;
    wp::vec_t<3, wp::float32> var_49;
    wp::vec_t<2, wp::float32> var_50;
    wp::mat_t<2, 3, wp::float32> var_51;
    wp::mat_t<2, 3, wp::float32> var_52;
    wp::vec_t<3, wp::float32> var_53;
    wp::vec_t<3, wp::float32> var_54;
    const wp::int32 var_55 = 0;
    wp::float32 var_56;
    const wp::int32 var_57 = 0;
    wp::float32 var_58;
    const wp::int32 var_59 = 1;
    wp::float32 var_60;
    const wp::int32 var_61 = 2;
    wp::float32 var_62;
    wp::vec_t<3, wp::float32> var_63;
    const wp::int32 var_64 = 0;
    wp::float32 var_65;
    const wp::int32 var_66 = 1;
    wp::float32 var_67;
    const wp::int32 var_68 = 2;
    wp::float32 var_69;
    wp::vec_t<3, wp::float32> var_70;
    wp::mat_t<3, 3, wp::float32> var_71;
    wp::int32 var_72;
    const wp::int32 var_73 = 1;
    wp::float32 var_74;
    const wp::int32 var_75 = 0;
    wp::float32 var_76;
    const wp::int32 var_77 = 1;
    wp::float32 var_78;
    const wp::int32 var_79 = 2;
    wp::float32 var_80;
    wp::vec_t<3, wp::float32> var_81;
    const wp::int32 var_82 = 0;
    wp::float32 var_83;
    const wp::int32 var_84 = 1;
    wp::float32 var_85;
    const wp::int32 var_86 = 2;
    wp::float32 var_87;
    wp::vec_t<3, wp::float32> var_88;
    wp::mat_t<3, 3, wp::float32> var_89;
    wp::int32 var_90;
    //---------
    // forward
    // def capsule_capsule_wrapper(                                                           <L 498>
    // cap1_axis = wp.vec3(cap1.rot[0, 2], cap1.rot[1, 2], cap1.rot[2, 2])                    <L 533>
    var_0 = &(var_cap1.rot);
    var_4 = wp::load(var_0);
    var_3 = wp::extract(var_4, var_1, var_2);
    var_5 = &(var_cap1.rot);
    var_9 = wp::load(var_5);
    var_8 = wp::extract(var_9, var_6, var_7);
    var_10 = &(var_cap1.rot);
    var_14 = wp::load(var_10);
    var_13 = wp::extract(var_14, var_11, var_12);
    var_15 = wp::vec_t<3, wp::float32>(var_3, var_8, var_13);
    // cap2_axis = wp.vec3(cap2.rot[0, 2], cap2.rot[1, 2], cap2.rot[2, 2])                    <L 534>
    var_16 = &(var_cap2.rot);
    var_20 = wp::load(var_16);
    var_19 = wp::extract(var_20, var_17, var_18);
    var_21 = &(var_cap2.rot);
    var_25 = wp::load(var_21);
    var_24 = wp::extract(var_25, var_22, var_23);
    var_26 = &(var_cap2.rot);
    var_30 = wp::load(var_26);
    var_29 = wp::extract(var_30, var_27, var_28);
    var_31 = wp::vec_t<3, wp::float32>(var_19, var_24, var_29);
    // dist, pos, normal = capsule_capsule(                                                   <L 536>
    // cap1.pos,                                                                              <L 537>
    var_32 = &(var_cap1.pos);
    // cap1_axis,                                                                             <L 538>
    // cap1.size[0],  # radius1                                                               <L 539>
    var_33 = &(var_cap1.size);
    var_36 = wp::load(var_33);
    var_35 = wp::extract(var_36, var_34);
    // cap1.size[1],  # half_length1                                                          <L 540>
    var_37 = &(var_cap1.size);
    var_40 = wp::load(var_37);
    var_39 = wp::extract(var_40, var_38);
    // cap2.pos,                                                                              <L 541>
    var_41 = &(var_cap2.pos);
    // cap2_axis,                                                                             <L 542>
    // cap2.size[0],  # radius2                                                               <L 543>
    var_42 = &(var_cap2.size);
    var_45 = wp::load(var_42);
    var_44 = wp::extract(var_45, var_43);
    // cap2.size[1],  # half_length2                                                          <L 544>
    var_46 = &(var_cap2.size);
    var_49 = wp::load(var_46);
    var_48 = wp::extract(var_49, var_47);
    // margin,                                                                                <L 545>
    var_53 = wp::load(var_32);
    var_54 = wp::load(var_41);
    capsule_capsule_0(var_53, var_15, var_35, var_39, var_54, var_31, var_44, var_48, var_margin, var_50, var_51, var_52);
    // for i in range(2):                                                                     <L 548>
    // write_contact(                                                                         <L 549>
    // naconmax_in,                                                                           <L 550>
    // i,                                                                                     <L 551>
    // dist[i],                                                                               <L 552>
    var_56 = wp::extract(var_50, var_55);
    // wp.vec3(pos[i, 0], pos[i, 1], pos[i, 2]),                                              <L 553>
    var_58 = wp::extract(var_51, var_55, var_57);
    var_60 = wp::extract(var_51, var_55, var_59);
    var_62 = wp::extract(var_51, var_55, var_61);
    var_63 = wp::vec_t<3, wp::float32>(var_58, var_60, var_62);
    // make_frame(wp.vec3(normal[i, 0], normal[i, 1], normal[i, 2])),                         <L 554>
    var_65 = wp::extract(var_52, var_55, var_64);
    var_67 = wp::extract(var_52, var_55, var_66);
    var_69 = wp::extract(var_52, var_55, var_68);
    var_70 = wp::vec_t<3, wp::float32>(var_65, var_67, var_69);
    var_71 = make_frame_0(var_70);
    // margin,                                                                                <L 555>
    // gap,                                                                                   <L 556>
    // condim,                                                                                <L 557>
    // friction,                                                                              <L 558>
    // solref,                                                                                <L 559>
    // solreffriction,                                                                        <L 560>
    // solimp,                                                                                <L 561>
    // geoms,                                                                                 <L 562>
    // pairid,                                                                                <L 563>
    // worldid,                                                                               <L 564>
    // contact_dist_out,                                                                      <L 565>
    // contact_pos_out,                                                                       <L 566>
    // contact_frame_out,                                                                     <L 567>
    // contact_includemargin_out,                                                             <L 568>
    // contact_friction_out,                                                                  <L 569>
    // contact_solref_out,                                                                    <L 570>
    // contact_solreffriction_out,                                                            <L 571>
    // contact_solimp_out,                                                                    <L 572>
    // contact_dim_out,                                                                       <L 573>
    // contact_geom_out,                                                                      <L 574>
    // contact_efc_address_out,                                                               <L 575>
    // contact_worldid_out,                                                                   <L 576>
    // contact_type_out,                                                                      <L 577>
    // contact_geomcollisionid_out,                                                           <L 578>
    // nacon_out,                                                                             <L 579>
    var_72 = write_contact_0(var_naconmax_in, var_55, var_56, var_63, var_71, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
    // write_contact(                                                                         <L 549>
    // naconmax_in,                                                                           <L 550>
    // i,                                                                                     <L 551>
    // dist[i],                                                                               <L 552>
    var_74 = wp::extract(var_50, var_73);
    // wp.vec3(pos[i, 0], pos[i, 1], pos[i, 2]),                                              <L 553>
    var_76 = wp::extract(var_51, var_73, var_75);
    var_78 = wp::extract(var_51, var_73, var_77);
    var_80 = wp::extract(var_51, var_73, var_79);
    var_81 = wp::vec_t<3, wp::float32>(var_76, var_78, var_80);
    // make_frame(wp.vec3(normal[i, 0], normal[i, 1], normal[i, 2])),                         <L 554>
    var_83 = wp::extract(var_52, var_73, var_82);
    var_85 = wp::extract(var_52, var_73, var_84);
    var_87 = wp::extract(var_52, var_73, var_86);
    var_88 = wp::vec_t<3, wp::float32>(var_83, var_85, var_87);
    var_89 = make_frame_0(var_88);
    // margin,                                                                                <L 555>
    // gap,                                                                                   <L 556>
    // condim,                                                                                <L 557>
    // friction,                                                                              <L 558>
    // solref,                                                                                <L 559>
    // solreffriction,                                                                        <L 560>
    // solimp,                                                                                <L 561>
    // geoms,                                                                                 <L 562>
    // pairid,                                                                                <L 563>
    // worldid,                                                                               <L 564>
    // contact_dist_out,                                                                      <L 565>
    // contact_pos_out,                                                                       <L 566>
    // contact_frame_out,                                                                     <L 567>
    // contact_includemargin_out,                                                             <L 568>
    // contact_friction_out,                                                                  <L 569>
    // contact_solref_out,                                                                    <L 570>
    // contact_solreffriction_out,                                                            <L 571>
    // contact_solimp_out,                                                                    <L 572>
    // contact_dim_out,                                                                       <L 573>
    // contact_geom_out,                                                                      <L 574>
    // contact_efc_address_out,                                                               <L 575>
    // contact_worldid_out,                                                                   <L 576>
    // contact_type_out,                                                                      <L 577>
    // contact_geomcollisionid_out,                                                           <L 578>
    // nacon_out,                                                                             <L 579>
    var_90 = write_contact_0(var_naconmax_in, var_73, var_74, var_81, var_89, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_primitive_core.py:1157
static CUDA_CALLABLE void capsule_box_0(
    wp::vec_t<3, wp::float32> var_capsule_pos,
    wp::vec_t<3, wp::float32> var_capsule_axis,
    wp::float32 var_capsule_radius,
    wp::float32 var_capsule_half_length,
    wp::vec_t<3, wp::float32> var_box_pos,
    wp::mat_t<3, 3, wp::float32> var_box_rot,
    wp::vec_t<3, wp::float32> var_box_size,
    wp::vec_t<2, wp::float32> & ret_0,
    wp::mat_t<2, 3, wp::float32> & ret_1,
    wp::mat_t<2, 3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::mat_t<3, 3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    const wp::int32 var_5 = 0;
    wp::float32 var_6;
    const wp::float32 var_7 = 0.0;
    bool var_8;
    wp::int32 var_9;
    const wp::int32 var_10 = 2;
    const wp::int32 var_11 = 1;
    wp::float32 var_12;
    const wp::float32 var_13 = 0.0;
    bool var_14;
    wp::int32 var_15;
    wp::int32 var_16;
    wp::int32 var_17;
    const wp::int32 var_18 = 4;
    const wp::int32 var_19 = 2;
    wp::float32 var_20;
    const wp::float32 var_21 = 0.0;
    bool var_22;
    wp::int32 var_23;
    wp::int32 var_24;
    wp::int32 var_25;
    const wp::float32 var_26 = 1e+32;
    wp::float32 var_27;
    const wp::int32 var_28 = 12;
    const wp::int32 var_29 = -12;
    wp::float32 var_30;
    const wp::int32 var_31 = 4;
    const wp::int32 var_32 = -4;
    wp::int32 var_33;
    const wp::int32 var_34 = 12;
    const wp::int32 var_35 = -12;
    wp::int32 var_36;
    const wp::int32 var_37 = -1;
    const wp::int32 var_38 = 2;
    const wp::int32 var_39 = 2;
    wp::range_t var_40;
    wp::int32 var_41;
    wp::float32 var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::vec_t<3, wp::float32> var_45;
    const wp::int32 var_46 = 0;
    wp::int32 var_47;
    const wp::int32 var_48 = 1;
    const wp::int32 var_49 = -1;
    wp::int32 var_50;
    const wp::int32 var_51 = 0;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    bool var_55;
    const wp::int32 var_56 = 1;
    wp::int32 var_57;
    wp::int32 var_58;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::int32 var_61;
    wp::int32 var_62;
    wp::float32 var_63;
    wp::float32 var_64;
    bool var_65;
    const wp::int32 var_66 = 1;
    wp::int32 var_67;
    wp::int32 var_68;
    wp::float32 var_69;
    wp::int32 var_70;
    wp::int32 var_71;
    wp::int32 var_72;
    wp::int32 var_73;
    const wp::int32 var_74 = 1;
    wp::float32 var_75;
    wp::float32 var_76;
    wp::float32 var_77;
    bool var_78;
    const wp::int32 var_79 = 1;
    wp::int32 var_80;
    wp::int32 var_81;
    wp::float32 var_82;
    wp::float32 var_83;
    wp::int32 var_84;
    wp::int32 var_85;
    wp::float32 var_86;
    wp::float32 var_87;
    bool var_88;
    const wp::int32 var_89 = 1;
    wp::int32 var_90;
    wp::int32 var_91;
    wp::float32 var_92;
    wp::int32 var_93;
    wp::int32 var_94;
    wp::int32 var_95;
    wp::int32 var_96;
    const wp::int32 var_97 = 2;
    wp::float32 var_98;
    wp::float32 var_99;
    wp::float32 var_100;
    bool var_101;
    const wp::int32 var_102 = 1;
    wp::int32 var_103;
    wp::int32 var_104;
    wp::float32 var_105;
    wp::float32 var_106;
    wp::int32 var_107;
    wp::int32 var_108;
    wp::float32 var_109;
    wp::float32 var_110;
    bool var_111;
    const wp::int32 var_112 = 1;
    wp::int32 var_113;
    wp::int32 var_114;
    wp::float32 var_115;
    wp::int32 var_116;
    wp::int32 var_117;
    wp::int32 var_118;
    wp::int32 var_119;
    const wp::int32 var_120 = 1;
    bool var_121;
    wp::vec_t<3, wp::float32> var_122;
    wp::float32 var_123;
    bool var_124;
    wp::float32 var_125;
    wp::float32 var_126;
    const wp::int32 var_127 = 2;
    const wp::int32 var_128 = -2;
    wp::int32 var_129;
    wp::int32 var_130;
    wp::float32 var_131;
    wp::float32 var_132;
    wp::int32 var_133;
    wp::int32 var_134;
    const wp::int32 var_135 = 123;
    const wp::int32 var_136 = -123;
    wp::int32 var_137;
    const wp::int32 var_138 = 123;
    const wp::int32 var_139 = -123;
    wp::int32 var_140;
    const wp::float32 var_141 = 0.0;
    wp::float32 var_142;
    const wp::int32 var_143 = 0;
    const wp::int32 var_144 = 3;
    wp::range_t var_145;
    wp::int32 var_146;
    const wp::int32 var_147 = 1;
    wp::int32 var_148;
    wp::int32 var_149;
    const wp::int32 var_150 = 0;
    bool var_151;
    const wp::int32 var_152 = 123;
    const wp::int32 var_153 = -123;
    wp::int32 var_154;
    const wp::int32 var_155 = 1;
    wp::int32 var_156;
    const wp::float32 var_157 = 1.0;
    const wp::float32 var_158 = 1.0;
    const wp::float32 var_159 = -1.0;
    wp::float32 var_160;
    const wp::int32 var_161 = 2;
    wp::int32 var_162;
    const wp::float32 var_163 = 1.0;
    const wp::float32 var_164 = 1.0;
    const wp::float32 var_165 = -1.0;
    wp::float32 var_166;
    const wp::int32 var_167 = 4;
    wp::int32 var_168;
    const wp::float32 var_169 = 1.0;
    const wp::float32 var_170 = 1.0;
    const wp::float32 var_171 = -1.0;
    wp::float32 var_172;
    wp::vec_t<3, wp::float32> var_173;
    wp::vec_t<3, wp::float32> var_174;
    const wp::float32 var_175 = 0.0;
    wp::vec_t<3, wp::float32> var_176;
    wp::float32 var_177;
    wp::float32 var_178;
    wp::float32 var_179;
    wp::float32 var_180;
    wp::float32 var_181;
    wp::float32 var_182;
    wp::float32 var_183;
    wp::float32 var_184;
    wp::float32 var_185;
    wp::float32 var_186;
    wp::float32 var_187;
    wp::float32 var_188;
    wp::float32 var_189;
    wp::float32 var_190;
    wp::float32 var_191;
    wp::float32 var_192;
    wp::float32 var_193;
    const wp::float32 var_194 = 1e-15;
    bool var_195;
    const wp::float32 var_196 = 1.0;
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
    wp::float32 var_207;
    const wp::int32 var_208 = 1;
    wp::int32 var_209;
    const wp::int32 var_210 = 1;
    wp::int32 var_211;
    const wp::int32 var_212 = 1;
    bool var_213;
    const wp::float32 var_214 = 1.0;
    const wp::int32 var_215 = 2;
    wp::float32 var_216;
    wp::float32 var_217;
    wp::float32 var_218;
    wp::float32 var_219;
    wp::int32 var_220;
    const wp::int32 var_221 = 1;
    const wp::int32 var_222 = -1;
    bool var_223;
    const wp::float32 var_224 = 1.0;
    const wp::float32 var_225 = -1.0;
    const wp::int32 var_226 = 0;
    wp::float32 var_227;
    wp::float32 var_228;
    wp::float32 var_229;
    wp::float32 var_230;
    wp::int32 var_231;
    wp::float32 var_232;
    wp::float32 var_233;
    wp::int32 var_234;
    const wp::float32 var_235 = 1.0;
    bool var_236;
    const wp::float32 var_237 = 1.0;
    const wp::float32 var_238 = -1.0;
    bool var_239;
    bool var_240;
    const wp::float32 var_241 = 1.0;
    const wp::int32 var_242 = 2;
    wp::float32 var_243;
    wp::float32 var_244;
    wp::float32 var_245;
    wp::float32 var_246;
    wp::int32 var_247;
    const wp::float32 var_248 = 1.0;
    const wp::float32 var_249 = -1.0;
    const wp::int32 var_250 = 0;
    wp::float32 var_251;
    wp::float32 var_252;
    wp::float32 var_253;
    wp::float32 var_254;
    wp::int32 var_255;
    const wp::int32 var_256 = 1;
    bool var_257;
    const wp::float32 var_258 = 1.0;
    const wp::int32 var_259 = 2;
    wp::float32 var_260;
    wp::int32 var_261;
    const wp::int32 var_262 = 1;
    const wp::int32 var_263 = -1;
    bool var_264;
    const wp::float32 var_265 = 1.0;
    const wp::float32 var_266 = -1.0;
    const wp::int32 var_267 = 0;
    wp::float32 var_268;
    wp::int32 var_269;
    wp::float32 var_270;
    wp::int32 var_271;
    wp::float32 var_272;
    wp::float32 var_273;
    wp::int32 var_274;
    wp::int32 var_275;
    wp::vec_t<3, wp::float32> var_276;
    wp::vec_t<3, wp::float32> var_277;
    wp::float32 var_278;
    wp::float32 var_279;
    const wp::int32 var_280 = 3;
    wp::int32 var_281;
    wp::int32 var_282;
    wp::float32 var_283;
    wp::float32 var_284;
    bool var_285;
    wp::float32 var_286;
    wp::float32 var_287;
    wp::float32 var_288;
    const wp::int32 var_289 = 6;
    wp::int32 var_290;
    const wp::int32 var_291 = 1;
    wp::int32 var_292;
    wp::int32 var_293;
    wp::int32 var_294;
    wp::int32 var_295;
    wp::int32 var_296;
    wp::float32 var_297;
    wp::float32 var_298;
    wp::int32 var_299;
    wp::int32 var_300;
    wp::int32 var_301;
    wp::float32 var_302;
    wp::int32 var_303;
    const wp::int32 var_304 = 1;
    const wp::int32 var_305 = 3;
    wp::range_t var_306;
    wp::int32 var_307;
    const wp::int32 var_308 = 1;
    wp::int32 var_309;
    wp::int32 var_310;
    const wp::int32 var_311 = 0;
    bool var_312;
    const wp::int32 var_313 = 123;
    const wp::int32 var_314 = -123;
    wp::int32 var_315;
    const wp::int32 var_316 = 1;
    wp::int32 var_317;
    const wp::float32 var_318 = 1.0;
    const wp::float32 var_319 = 1.0;
    const wp::float32 var_320 = -1.0;
    wp::float32 var_321;
    const wp::int32 var_322 = 2;
    wp::int32 var_323;
    const wp::float32 var_324 = 1.0;
    const wp::float32 var_325 = 1.0;
    const wp::float32 var_326 = -1.0;
    wp::float32 var_327;
    const wp::int32 var_328 = 4;
    wp::int32 var_329;
    const wp::float32 var_330 = 1.0;
    const wp::float32 var_331 = 1.0;
    const wp::float32 var_332 = -1.0;
    wp::float32 var_333;
    wp::vec_t<3, wp::float32> var_334;
    wp::vec_t<3, wp::float32> var_335;
    const wp::float32 var_336 = 0.0;
    wp::vec_t<3, wp::float32> var_337;
    wp::float32 var_338;
    wp::float32 var_339;
    wp::float32 var_340;
    wp::float32 var_341;
    wp::float32 var_342;
    wp::float32 var_343;
    wp::float32 var_344;
    wp::float32 var_345;
    wp::float32 var_346;
    wp::float32 var_347;
    wp::float32 var_348;
    wp::float32 var_349;
    wp::float32 var_350;
    wp::float32 var_351;
    wp::float32 var_352;
    wp::float32 var_353;
    wp::float32 var_354;
    bool var_355;
    wp::int32 var_356;
    wp::vec_t<3, wp::float32> var_357;
    wp::vec_t<3, wp::float32> var_358;
    wp::float32 var_359;
    wp::float32 var_360;
    wp::float32 var_361;
    wp::float32 var_362;
    wp::float32 var_363;
    wp::float32 var_364;
    const wp::float32 var_365 = 1.0;
    wp::float32 var_366;
    wp::float32 var_367;
    wp::float32 var_368;
    wp::float32 var_369;
    wp::float32 var_370;
    wp::float32 var_371;
    wp::float32 var_372;
    wp::float32 var_373;
    wp::float32 var_374;
    wp::float32 var_375;
    wp::float32 var_376;
    const wp::int32 var_377 = 1;
    wp::int32 var_378;
    const wp::int32 var_379 = 1;
    wp::int32 var_380;
    const wp::int32 var_381 = 1;
    bool var_382;
    const wp::float32 var_383 = 1.0;
    const wp::int32 var_384 = 2;
    wp::float32 var_385;
    wp::float32 var_386;
    wp::float32 var_387;
    wp::float32 var_388;
    wp::int32 var_389;
    const wp::int32 var_390 = 1;
    const wp::int32 var_391 = -1;
    bool var_392;
    const wp::float32 var_393 = 1.0;
    const wp::float32 var_394 = -1.0;
    const wp::int32 var_395 = 0;
    wp::float32 var_396;
    wp::float32 var_397;
    wp::float32 var_398;
    wp::float32 var_399;
    wp::int32 var_400;
    wp::float32 var_401;
    wp::float32 var_402;
    wp::int32 var_403;
    const wp::float32 var_404 = 1.0;
    bool var_405;
    const wp::float32 var_406 = 1.0;
    const wp::float32 var_407 = -1.0;
    bool var_408;
    bool var_409;
    const wp::float32 var_410 = 1.0;
    const wp::int32 var_411 = 2;
    wp::float32 var_412;
    wp::float32 var_413;
    wp::float32 var_414;
    wp::float32 var_415;
    wp::int32 var_416;
    const wp::float32 var_417 = 1.0;
    const wp::float32 var_418 = -1.0;
    const wp::int32 var_419 = 0;
    wp::float32 var_420;
    wp::float32 var_421;
    wp::float32 var_422;
    wp::float32 var_423;
    wp::int32 var_424;
    const wp::int32 var_425 = 1;
    bool var_426;
    const wp::float32 var_427 = 1.0;
    const wp::int32 var_428 = 2;
    wp::float32 var_429;
    wp::int32 var_430;
    const wp::int32 var_431 = 1;
    const wp::int32 var_432 = -1;
    bool var_433;
    const wp::float32 var_434 = 1.0;
    const wp::float32 var_435 = -1.0;
    const wp::int32 var_436 = 0;
    wp::float32 var_437;
    wp::int32 var_438;
    wp::float32 var_439;
    wp::int32 var_440;
    wp::float32 var_441;
    wp::float32 var_442;
    wp::int32 var_443;
    wp::int32 var_444;
    wp::vec_t<3, wp::float32> var_445;
    wp::vec_t<3, wp::float32> var_446;
    wp::float32 var_447;
    wp::float32 var_448;
    const wp::int32 var_449 = 3;
    wp::int32 var_450;
    wp::int32 var_451;
    wp::float32 var_452;
    wp::float32 var_453;
    bool var_454;
    wp::float32 var_455;
    wp::float32 var_456;
    wp::float32 var_457;
    const wp::int32 var_458 = 6;
    wp::int32 var_459;
    const wp::int32 var_460 = 1;
    wp::int32 var_461;
    wp::int32 var_462;
    wp::int32 var_463;
    wp::int32 var_464;
    wp::int32 var_465;
    wp::float32 var_466;
    wp::float32 var_467;
    wp::int32 var_468;
    wp::int32 var_469;
    wp::int32 var_470;
    wp::float32 var_471;
    wp::int32 var_472;
    const wp::int32 var_473 = 2;
    const wp::int32 var_474 = 3;
    wp::range_t var_475;
    wp::int32 var_476;
    const wp::int32 var_477 = 1;
    wp::int32 var_478;
    wp::int32 var_479;
    const wp::int32 var_480 = 0;
    bool var_481;
    const wp::int32 var_482 = 123;
    const wp::int32 var_483 = -123;
    wp::int32 var_484;
    const wp::int32 var_485 = 1;
    wp::int32 var_486;
    const wp::float32 var_487 = 1.0;
    const wp::float32 var_488 = 1.0;
    const wp::float32 var_489 = -1.0;
    wp::float32 var_490;
    const wp::int32 var_491 = 2;
    wp::int32 var_492;
    const wp::float32 var_493 = 1.0;
    const wp::float32 var_494 = 1.0;
    const wp::float32 var_495 = -1.0;
    wp::float32 var_496;
    const wp::int32 var_497 = 4;
    wp::int32 var_498;
    const wp::float32 var_499 = 1.0;
    const wp::float32 var_500 = 1.0;
    const wp::float32 var_501 = -1.0;
    wp::float32 var_502;
    wp::vec_t<3, wp::float32> var_503;
    wp::vec_t<3, wp::float32> var_504;
    const wp::float32 var_505 = 0.0;
    wp::vec_t<3, wp::float32> var_506;
    wp::float32 var_507;
    wp::float32 var_508;
    wp::float32 var_509;
    wp::float32 var_510;
    wp::float32 var_511;
    wp::float32 var_512;
    wp::float32 var_513;
    wp::float32 var_514;
    wp::float32 var_515;
    wp::float32 var_516;
    wp::float32 var_517;
    wp::float32 var_518;
    wp::float32 var_519;
    wp::float32 var_520;
    wp::float32 var_521;
    wp::float32 var_522;
    wp::float32 var_523;
    bool var_524;
    wp::int32 var_525;
    wp::vec_t<3, wp::float32> var_526;
    wp::vec_t<3, wp::float32> var_527;
    wp::float32 var_528;
    wp::float32 var_529;
    wp::float32 var_530;
    wp::float32 var_531;
    wp::float32 var_532;
    wp::float32 var_533;
    const wp::float32 var_534 = 1.0;
    wp::float32 var_535;
    wp::float32 var_536;
    wp::float32 var_537;
    wp::float32 var_538;
    wp::float32 var_539;
    wp::float32 var_540;
    wp::float32 var_541;
    wp::float32 var_542;
    wp::float32 var_543;
    wp::float32 var_544;
    wp::float32 var_545;
    const wp::int32 var_546 = 1;
    wp::int32 var_547;
    const wp::int32 var_548 = 1;
    wp::int32 var_549;
    const wp::int32 var_550 = 1;
    bool var_551;
    const wp::float32 var_552 = 1.0;
    const wp::int32 var_553 = 2;
    wp::float32 var_554;
    wp::float32 var_555;
    wp::float32 var_556;
    wp::float32 var_557;
    wp::int32 var_558;
    const wp::int32 var_559 = 1;
    const wp::int32 var_560 = -1;
    bool var_561;
    const wp::float32 var_562 = 1.0;
    const wp::float32 var_563 = -1.0;
    const wp::int32 var_564 = 0;
    wp::float32 var_565;
    wp::float32 var_566;
    wp::float32 var_567;
    wp::float32 var_568;
    wp::int32 var_569;
    wp::float32 var_570;
    wp::float32 var_571;
    wp::int32 var_572;
    const wp::float32 var_573 = 1.0;
    bool var_574;
    const wp::float32 var_575 = 1.0;
    const wp::float32 var_576 = -1.0;
    bool var_577;
    bool var_578;
    const wp::float32 var_579 = 1.0;
    const wp::int32 var_580 = 2;
    wp::float32 var_581;
    wp::float32 var_582;
    wp::float32 var_583;
    wp::float32 var_584;
    wp::int32 var_585;
    const wp::float32 var_586 = 1.0;
    const wp::float32 var_587 = -1.0;
    const wp::int32 var_588 = 0;
    wp::float32 var_589;
    wp::float32 var_590;
    wp::float32 var_591;
    wp::float32 var_592;
    wp::int32 var_593;
    const wp::int32 var_594 = 1;
    bool var_595;
    const wp::float32 var_596 = 1.0;
    const wp::int32 var_597 = 2;
    wp::float32 var_598;
    wp::int32 var_599;
    const wp::int32 var_600 = 1;
    const wp::int32 var_601 = -1;
    bool var_602;
    const wp::float32 var_603 = 1.0;
    const wp::float32 var_604 = -1.0;
    const wp::int32 var_605 = 0;
    wp::float32 var_606;
    wp::int32 var_607;
    wp::float32 var_608;
    wp::int32 var_609;
    wp::float32 var_610;
    wp::float32 var_611;
    wp::int32 var_612;
    wp::int32 var_613;
    wp::vec_t<3, wp::float32> var_614;
    wp::vec_t<3, wp::float32> var_615;
    wp::float32 var_616;
    wp::float32 var_617;
    const wp::int32 var_618 = 3;
    wp::int32 var_619;
    wp::int32 var_620;
    wp::float32 var_621;
    wp::float32 var_622;
    bool var_623;
    wp::float32 var_624;
    wp::float32 var_625;
    wp::float32 var_626;
    const wp::int32 var_627 = 6;
    wp::int32 var_628;
    const wp::int32 var_629 = 1;
    wp::int32 var_630;
    wp::int32 var_631;
    wp::int32 var_632;
    wp::int32 var_633;
    wp::int32 var_634;
    wp::float32 var_635;
    wp::float32 var_636;
    wp::int32 var_637;
    wp::int32 var_638;
    wp::int32 var_639;
    wp::float32 var_640;
    wp::int32 var_641;
    const wp::int32 var_642 = 3;
    const wp::int32 var_643 = 3;
    wp::range_t var_644;
    wp::int32 var_645;
    const wp::int32 var_646 = 1;
    wp::int32 var_647;
    wp::int32 var_648;
    const wp::int32 var_649 = 0;
    bool var_650;
    const wp::int32 var_651 = 123;
    const wp::int32 var_652 = -123;
    wp::int32 var_653;
    const wp::int32 var_654 = 1;
    wp::int32 var_655;
    const wp::float32 var_656 = 1.0;
    const wp::float32 var_657 = 1.0;
    const wp::float32 var_658 = -1.0;
    wp::float32 var_659;
    const wp::int32 var_660 = 2;
    wp::int32 var_661;
    const wp::float32 var_662 = 1.0;
    const wp::float32 var_663 = 1.0;
    const wp::float32 var_664 = -1.0;
    wp::float32 var_665;
    const wp::int32 var_666 = 4;
    wp::int32 var_667;
    const wp::float32 var_668 = 1.0;
    const wp::float32 var_669 = 1.0;
    const wp::float32 var_670 = -1.0;
    wp::float32 var_671;
    wp::vec_t<3, wp::float32> var_672;
    wp::vec_t<3, wp::float32> var_673;
    const wp::float32 var_674 = 0.0;
    wp::vec_t<3, wp::float32> var_675;
    wp::float32 var_676;
    wp::float32 var_677;
    wp::float32 var_678;
    wp::float32 var_679;
    wp::float32 var_680;
    wp::float32 var_681;
    wp::float32 var_682;
    wp::float32 var_683;
    wp::float32 var_684;
    wp::float32 var_685;
    wp::float32 var_686;
    wp::float32 var_687;
    wp::float32 var_688;
    wp::float32 var_689;
    wp::float32 var_690;
    wp::float32 var_691;
    wp::float32 var_692;
    bool var_693;
    wp::int32 var_694;
    wp::vec_t<3, wp::float32> var_695;
    wp::vec_t<3, wp::float32> var_696;
    wp::float32 var_697;
    wp::float32 var_698;
    wp::float32 var_699;
    wp::float32 var_700;
    wp::float32 var_701;
    wp::float32 var_702;
    const wp::float32 var_703 = 1.0;
    wp::float32 var_704;
    wp::float32 var_705;
    wp::float32 var_706;
    wp::float32 var_707;
    wp::float32 var_708;
    wp::float32 var_709;
    wp::float32 var_710;
    wp::float32 var_711;
    wp::float32 var_712;
    wp::float32 var_713;
    wp::float32 var_714;
    const wp::int32 var_715 = 1;
    wp::int32 var_716;
    const wp::int32 var_717 = 1;
    wp::int32 var_718;
    const wp::int32 var_719 = 1;
    bool var_720;
    const wp::float32 var_721 = 1.0;
    const wp::int32 var_722 = 2;
    wp::float32 var_723;
    wp::float32 var_724;
    wp::float32 var_725;
    wp::float32 var_726;
    wp::int32 var_727;
    const wp::int32 var_728 = 1;
    const wp::int32 var_729 = -1;
    bool var_730;
    const wp::float32 var_731 = 1.0;
    const wp::float32 var_732 = -1.0;
    const wp::int32 var_733 = 0;
    wp::float32 var_734;
    wp::float32 var_735;
    wp::float32 var_736;
    wp::float32 var_737;
    wp::int32 var_738;
    wp::float32 var_739;
    wp::float32 var_740;
    wp::int32 var_741;
    const wp::float32 var_742 = 1.0;
    bool var_743;
    const wp::float32 var_744 = 1.0;
    const wp::float32 var_745 = -1.0;
    bool var_746;
    bool var_747;
    const wp::float32 var_748 = 1.0;
    const wp::int32 var_749 = 2;
    wp::float32 var_750;
    wp::float32 var_751;
    wp::float32 var_752;
    wp::float32 var_753;
    wp::int32 var_754;
    const wp::float32 var_755 = 1.0;
    const wp::float32 var_756 = -1.0;
    const wp::int32 var_757 = 0;
    wp::float32 var_758;
    wp::float32 var_759;
    wp::float32 var_760;
    wp::float32 var_761;
    wp::int32 var_762;
    const wp::int32 var_763 = 1;
    bool var_764;
    const wp::float32 var_765 = 1.0;
    const wp::int32 var_766 = 2;
    wp::float32 var_767;
    wp::int32 var_768;
    const wp::int32 var_769 = 1;
    const wp::int32 var_770 = -1;
    bool var_771;
    const wp::float32 var_772 = 1.0;
    const wp::float32 var_773 = -1.0;
    const wp::int32 var_774 = 0;
    wp::float32 var_775;
    wp::int32 var_776;
    wp::float32 var_777;
    wp::int32 var_778;
    wp::float32 var_779;
    wp::float32 var_780;
    wp::int32 var_781;
    wp::int32 var_782;
    wp::vec_t<3, wp::float32> var_783;
    wp::vec_t<3, wp::float32> var_784;
    wp::float32 var_785;
    wp::float32 var_786;
    const wp::int32 var_787 = 3;
    wp::int32 var_788;
    wp::int32 var_789;
    wp::float32 var_790;
    wp::float32 var_791;
    bool var_792;
    wp::float32 var_793;
    wp::float32 var_794;
    wp::float32 var_795;
    const wp::int32 var_796 = 6;
    wp::int32 var_797;
    const wp::int32 var_798 = 1;
    wp::int32 var_799;
    wp::int32 var_800;
    wp::int32 var_801;
    wp::int32 var_802;
    wp::int32 var_803;
    wp::float32 var_804;
    wp::float32 var_805;
    wp::int32 var_806;
    wp::int32 var_807;
    wp::int32 var_808;
    wp::float32 var_809;
    wp::int32 var_810;
    const wp::int32 var_811 = 4;
    const wp::int32 var_812 = 3;
    wp::range_t var_813;
    wp::int32 var_814;
    const wp::int32 var_815 = 1;
    wp::int32 var_816;
    wp::int32 var_817;
    const wp::int32 var_818 = 0;
    bool var_819;
    const wp::int32 var_820 = 123;
    const wp::int32 var_821 = -123;
    wp::int32 var_822;
    const wp::int32 var_823 = 1;
    wp::int32 var_824;
    const wp::float32 var_825 = 1.0;
    const wp::float32 var_826 = 1.0;
    const wp::float32 var_827 = -1.0;
    wp::float32 var_828;
    const wp::int32 var_829 = 2;
    wp::int32 var_830;
    const wp::float32 var_831 = 1.0;
    const wp::float32 var_832 = 1.0;
    const wp::float32 var_833 = -1.0;
    wp::float32 var_834;
    const wp::int32 var_835 = 4;
    wp::int32 var_836;
    const wp::float32 var_837 = 1.0;
    const wp::float32 var_838 = 1.0;
    const wp::float32 var_839 = -1.0;
    wp::float32 var_840;
    wp::vec_t<3, wp::float32> var_841;
    wp::vec_t<3, wp::float32> var_842;
    const wp::float32 var_843 = 0.0;
    wp::vec_t<3, wp::float32> var_844;
    wp::float32 var_845;
    wp::float32 var_846;
    wp::float32 var_847;
    wp::float32 var_848;
    wp::float32 var_849;
    wp::float32 var_850;
    wp::float32 var_851;
    wp::float32 var_852;
    wp::float32 var_853;
    wp::float32 var_854;
    wp::float32 var_855;
    wp::float32 var_856;
    wp::float32 var_857;
    wp::float32 var_858;
    wp::float32 var_859;
    wp::float32 var_860;
    wp::float32 var_861;
    bool var_862;
    wp::int32 var_863;
    wp::vec_t<3, wp::float32> var_864;
    wp::vec_t<3, wp::float32> var_865;
    wp::float32 var_866;
    wp::float32 var_867;
    wp::float32 var_868;
    wp::float32 var_869;
    wp::float32 var_870;
    wp::float32 var_871;
    const wp::float32 var_872 = 1.0;
    wp::float32 var_873;
    wp::float32 var_874;
    wp::float32 var_875;
    wp::float32 var_876;
    wp::float32 var_877;
    wp::float32 var_878;
    wp::float32 var_879;
    wp::float32 var_880;
    wp::float32 var_881;
    wp::float32 var_882;
    wp::float32 var_883;
    const wp::int32 var_884 = 1;
    wp::int32 var_885;
    const wp::int32 var_886 = 1;
    wp::int32 var_887;
    const wp::int32 var_888 = 1;
    bool var_889;
    const wp::float32 var_890 = 1.0;
    const wp::int32 var_891 = 2;
    wp::float32 var_892;
    wp::float32 var_893;
    wp::float32 var_894;
    wp::float32 var_895;
    wp::int32 var_896;
    const wp::int32 var_897 = 1;
    const wp::int32 var_898 = -1;
    bool var_899;
    const wp::float32 var_900 = 1.0;
    const wp::float32 var_901 = -1.0;
    const wp::int32 var_902 = 0;
    wp::float32 var_903;
    wp::float32 var_904;
    wp::float32 var_905;
    wp::float32 var_906;
    wp::int32 var_907;
    wp::float32 var_908;
    wp::float32 var_909;
    wp::int32 var_910;
    const wp::float32 var_911 = 1.0;
    bool var_912;
    const wp::float32 var_913 = 1.0;
    const wp::float32 var_914 = -1.0;
    bool var_915;
    bool var_916;
    const wp::float32 var_917 = 1.0;
    const wp::int32 var_918 = 2;
    wp::float32 var_919;
    wp::float32 var_920;
    wp::float32 var_921;
    wp::float32 var_922;
    wp::int32 var_923;
    const wp::float32 var_924 = 1.0;
    const wp::float32 var_925 = -1.0;
    const wp::int32 var_926 = 0;
    wp::float32 var_927;
    wp::float32 var_928;
    wp::float32 var_929;
    wp::float32 var_930;
    wp::int32 var_931;
    const wp::int32 var_932 = 1;
    bool var_933;
    const wp::float32 var_934 = 1.0;
    const wp::int32 var_935 = 2;
    wp::float32 var_936;
    wp::int32 var_937;
    const wp::int32 var_938 = 1;
    const wp::int32 var_939 = -1;
    bool var_940;
    const wp::float32 var_941 = 1.0;
    const wp::float32 var_942 = -1.0;
    const wp::int32 var_943 = 0;
    wp::float32 var_944;
    wp::int32 var_945;
    wp::float32 var_946;
    wp::int32 var_947;
    wp::float32 var_948;
    wp::float32 var_949;
    wp::int32 var_950;
    wp::int32 var_951;
    wp::vec_t<3, wp::float32> var_952;
    wp::vec_t<3, wp::float32> var_953;
    wp::float32 var_954;
    wp::float32 var_955;
    const wp::int32 var_956 = 3;
    wp::int32 var_957;
    wp::int32 var_958;
    wp::float32 var_959;
    wp::float32 var_960;
    bool var_961;
    wp::float32 var_962;
    wp::float32 var_963;
    wp::float32 var_964;
    const wp::int32 var_965 = 6;
    wp::int32 var_966;
    const wp::int32 var_967 = 1;
    wp::int32 var_968;
    wp::int32 var_969;
    wp::int32 var_970;
    wp::int32 var_971;
    wp::int32 var_972;
    wp::float32 var_973;
    wp::float32 var_974;
    wp::int32 var_975;
    wp::int32 var_976;
    wp::int32 var_977;
    wp::float32 var_978;
    wp::int32 var_979;
    const wp::int32 var_980 = 5;
    const wp::int32 var_981 = 3;
    wp::range_t var_982;
    wp::int32 var_983;
    const wp::int32 var_984 = 1;
    wp::int32 var_985;
    wp::int32 var_986;
    const wp::int32 var_987 = 0;
    bool var_988;
    const wp::int32 var_989 = 123;
    const wp::int32 var_990 = -123;
    wp::int32 var_991;
    const wp::int32 var_992 = 1;
    wp::int32 var_993;
    const wp::float32 var_994 = 1.0;
    const wp::float32 var_995 = 1.0;
    const wp::float32 var_996 = -1.0;
    wp::float32 var_997;
    const wp::int32 var_998 = 2;
    wp::int32 var_999;
    const wp::float32 var_1000 = 1.0;
    const wp::float32 var_1001 = 1.0;
    const wp::float32 var_1002 = -1.0;
    wp::float32 var_1003;
    const wp::int32 var_1004 = 4;
    wp::int32 var_1005;
    const wp::float32 var_1006 = 1.0;
    const wp::float32 var_1007 = 1.0;
    const wp::float32 var_1008 = -1.0;
    wp::float32 var_1009;
    wp::vec_t<3, wp::float32> var_1010;
    wp::vec_t<3, wp::float32> var_1011;
    const wp::float32 var_1012 = 0.0;
    wp::vec_t<3, wp::float32> var_1013;
    wp::float32 var_1014;
    wp::float32 var_1015;
    wp::float32 var_1016;
    wp::float32 var_1017;
    wp::float32 var_1018;
    wp::float32 var_1019;
    wp::float32 var_1020;
    wp::float32 var_1021;
    wp::float32 var_1022;
    wp::float32 var_1023;
    wp::float32 var_1024;
    wp::float32 var_1025;
    wp::float32 var_1026;
    wp::float32 var_1027;
    wp::float32 var_1028;
    wp::float32 var_1029;
    wp::float32 var_1030;
    bool var_1031;
    wp::int32 var_1032;
    wp::vec_t<3, wp::float32> var_1033;
    wp::vec_t<3, wp::float32> var_1034;
    wp::float32 var_1035;
    wp::float32 var_1036;
    wp::float32 var_1037;
    wp::float32 var_1038;
    wp::float32 var_1039;
    wp::float32 var_1040;
    const wp::float32 var_1041 = 1.0;
    wp::float32 var_1042;
    wp::float32 var_1043;
    wp::float32 var_1044;
    wp::float32 var_1045;
    wp::float32 var_1046;
    wp::float32 var_1047;
    wp::float32 var_1048;
    wp::float32 var_1049;
    wp::float32 var_1050;
    wp::float32 var_1051;
    wp::float32 var_1052;
    const wp::int32 var_1053 = 1;
    wp::int32 var_1054;
    const wp::int32 var_1055 = 1;
    wp::int32 var_1056;
    const wp::int32 var_1057 = 1;
    bool var_1058;
    const wp::float32 var_1059 = 1.0;
    const wp::int32 var_1060 = 2;
    wp::float32 var_1061;
    wp::float32 var_1062;
    wp::float32 var_1063;
    wp::float32 var_1064;
    wp::int32 var_1065;
    const wp::int32 var_1066 = 1;
    const wp::int32 var_1067 = -1;
    bool var_1068;
    const wp::float32 var_1069 = 1.0;
    const wp::float32 var_1070 = -1.0;
    const wp::int32 var_1071 = 0;
    wp::float32 var_1072;
    wp::float32 var_1073;
    wp::float32 var_1074;
    wp::float32 var_1075;
    wp::int32 var_1076;
    wp::float32 var_1077;
    wp::float32 var_1078;
    wp::int32 var_1079;
    const wp::float32 var_1080 = 1.0;
    bool var_1081;
    const wp::float32 var_1082 = 1.0;
    const wp::float32 var_1083 = -1.0;
    bool var_1084;
    bool var_1085;
    const wp::float32 var_1086 = 1.0;
    const wp::int32 var_1087 = 2;
    wp::float32 var_1088;
    wp::float32 var_1089;
    wp::float32 var_1090;
    wp::float32 var_1091;
    wp::int32 var_1092;
    const wp::float32 var_1093 = 1.0;
    const wp::float32 var_1094 = -1.0;
    const wp::int32 var_1095 = 0;
    wp::float32 var_1096;
    wp::float32 var_1097;
    wp::float32 var_1098;
    wp::float32 var_1099;
    wp::int32 var_1100;
    const wp::int32 var_1101 = 1;
    bool var_1102;
    const wp::float32 var_1103 = 1.0;
    const wp::int32 var_1104 = 2;
    wp::float32 var_1105;
    wp::int32 var_1106;
    const wp::int32 var_1107 = 1;
    const wp::int32 var_1108 = -1;
    bool var_1109;
    const wp::float32 var_1110 = 1.0;
    const wp::float32 var_1111 = -1.0;
    const wp::int32 var_1112 = 0;
    wp::float32 var_1113;
    wp::int32 var_1114;
    wp::float32 var_1115;
    wp::int32 var_1116;
    wp::float32 var_1117;
    wp::float32 var_1118;
    wp::int32 var_1119;
    wp::int32 var_1120;
    wp::vec_t<3, wp::float32> var_1121;
    wp::vec_t<3, wp::float32> var_1122;
    wp::float32 var_1123;
    wp::float32 var_1124;
    const wp::int32 var_1125 = 3;
    wp::int32 var_1126;
    wp::int32 var_1127;
    wp::float32 var_1128;
    wp::float32 var_1129;
    bool var_1130;
    wp::float32 var_1131;
    wp::float32 var_1132;
    wp::float32 var_1133;
    const wp::int32 var_1134 = 6;
    wp::int32 var_1135;
    const wp::int32 var_1136 = 1;
    wp::int32 var_1137;
    wp::int32 var_1138;
    wp::int32 var_1139;
    wp::int32 var_1140;
    wp::int32 var_1141;
    wp::float32 var_1142;
    wp::float32 var_1143;
    wp::int32 var_1144;
    wp::int32 var_1145;
    wp::int32 var_1146;
    wp::float32 var_1147;
    wp::int32 var_1148;
    const wp::int32 var_1149 = 6;
    const wp::int32 var_1150 = 3;
    wp::range_t var_1151;
    wp::int32 var_1152;
    const wp::int32 var_1153 = 1;
    wp::int32 var_1154;
    wp::int32 var_1155;
    const wp::int32 var_1156 = 0;
    bool var_1157;
    const wp::int32 var_1158 = 123;
    const wp::int32 var_1159 = -123;
    wp::int32 var_1160;
    const wp::int32 var_1161 = 1;
    wp::int32 var_1162;
    const wp::float32 var_1163 = 1.0;
    const wp::float32 var_1164 = 1.0;
    const wp::float32 var_1165 = -1.0;
    wp::float32 var_1166;
    const wp::int32 var_1167 = 2;
    wp::int32 var_1168;
    const wp::float32 var_1169 = 1.0;
    const wp::float32 var_1170 = 1.0;
    const wp::float32 var_1171 = -1.0;
    wp::float32 var_1172;
    const wp::int32 var_1173 = 4;
    wp::int32 var_1174;
    const wp::float32 var_1175 = 1.0;
    const wp::float32 var_1176 = 1.0;
    const wp::float32 var_1177 = -1.0;
    wp::float32 var_1178;
    wp::vec_t<3, wp::float32> var_1179;
    wp::vec_t<3, wp::float32> var_1180;
    const wp::float32 var_1181 = 0.0;
    wp::vec_t<3, wp::float32> var_1182;
    wp::float32 var_1183;
    wp::float32 var_1184;
    wp::float32 var_1185;
    wp::float32 var_1186;
    wp::float32 var_1187;
    wp::float32 var_1188;
    wp::float32 var_1189;
    wp::float32 var_1190;
    wp::float32 var_1191;
    wp::float32 var_1192;
    wp::float32 var_1193;
    wp::float32 var_1194;
    wp::float32 var_1195;
    wp::float32 var_1196;
    wp::float32 var_1197;
    wp::float32 var_1198;
    wp::float32 var_1199;
    bool var_1200;
    wp::int32 var_1201;
    wp::vec_t<3, wp::float32> var_1202;
    wp::vec_t<3, wp::float32> var_1203;
    wp::float32 var_1204;
    wp::float32 var_1205;
    wp::float32 var_1206;
    wp::float32 var_1207;
    wp::float32 var_1208;
    wp::float32 var_1209;
    const wp::float32 var_1210 = 1.0;
    wp::float32 var_1211;
    wp::float32 var_1212;
    wp::float32 var_1213;
    wp::float32 var_1214;
    wp::float32 var_1215;
    wp::float32 var_1216;
    wp::float32 var_1217;
    wp::float32 var_1218;
    wp::float32 var_1219;
    wp::float32 var_1220;
    wp::float32 var_1221;
    const wp::int32 var_1222 = 1;
    wp::int32 var_1223;
    const wp::int32 var_1224 = 1;
    wp::int32 var_1225;
    const wp::int32 var_1226 = 1;
    bool var_1227;
    const wp::float32 var_1228 = 1.0;
    const wp::int32 var_1229 = 2;
    wp::float32 var_1230;
    wp::float32 var_1231;
    wp::float32 var_1232;
    wp::float32 var_1233;
    wp::int32 var_1234;
    const wp::int32 var_1235 = 1;
    const wp::int32 var_1236 = -1;
    bool var_1237;
    const wp::float32 var_1238 = 1.0;
    const wp::float32 var_1239 = -1.0;
    const wp::int32 var_1240 = 0;
    wp::float32 var_1241;
    wp::float32 var_1242;
    wp::float32 var_1243;
    wp::float32 var_1244;
    wp::int32 var_1245;
    wp::float32 var_1246;
    wp::float32 var_1247;
    wp::int32 var_1248;
    const wp::float32 var_1249 = 1.0;
    bool var_1250;
    const wp::float32 var_1251 = 1.0;
    const wp::float32 var_1252 = -1.0;
    bool var_1253;
    bool var_1254;
    const wp::float32 var_1255 = 1.0;
    const wp::int32 var_1256 = 2;
    wp::float32 var_1257;
    wp::float32 var_1258;
    wp::float32 var_1259;
    wp::float32 var_1260;
    wp::int32 var_1261;
    const wp::float32 var_1262 = 1.0;
    const wp::float32 var_1263 = -1.0;
    const wp::int32 var_1264 = 0;
    wp::float32 var_1265;
    wp::float32 var_1266;
    wp::float32 var_1267;
    wp::float32 var_1268;
    wp::int32 var_1269;
    const wp::int32 var_1270 = 1;
    bool var_1271;
    const wp::float32 var_1272 = 1.0;
    const wp::int32 var_1273 = 2;
    wp::float32 var_1274;
    wp::int32 var_1275;
    const wp::int32 var_1276 = 1;
    const wp::int32 var_1277 = -1;
    bool var_1278;
    const wp::float32 var_1279 = 1.0;
    const wp::float32 var_1280 = -1.0;
    const wp::int32 var_1281 = 0;
    wp::float32 var_1282;
    wp::int32 var_1283;
    wp::float32 var_1284;
    wp::int32 var_1285;
    wp::float32 var_1286;
    wp::float32 var_1287;
    wp::int32 var_1288;
    wp::int32 var_1289;
    wp::vec_t<3, wp::float32> var_1290;
    wp::vec_t<3, wp::float32> var_1291;
    wp::float32 var_1292;
    wp::float32 var_1293;
    const wp::int32 var_1294 = 3;
    wp::int32 var_1295;
    wp::int32 var_1296;
    wp::float32 var_1297;
    wp::float32 var_1298;
    bool var_1299;
    wp::float32 var_1300;
    wp::float32 var_1301;
    wp::float32 var_1302;
    const wp::int32 var_1303 = 6;
    wp::int32 var_1304;
    const wp::int32 var_1305 = 1;
    wp::int32 var_1306;
    wp::int32 var_1307;
    wp::int32 var_1308;
    wp::int32 var_1309;
    wp::int32 var_1310;
    wp::float32 var_1311;
    wp::float32 var_1312;
    wp::int32 var_1313;
    wp::int32 var_1314;
    wp::int32 var_1315;
    wp::float32 var_1316;
    wp::int32 var_1317;
    const wp::int32 var_1318 = 7;
    const wp::int32 var_1319 = 3;
    wp::range_t var_1320;
    wp::int32 var_1321;
    const wp::int32 var_1322 = 1;
    wp::int32 var_1323;
    wp::int32 var_1324;
    const wp::int32 var_1325 = 0;
    bool var_1326;
    const wp::int32 var_1327 = 123;
    const wp::int32 var_1328 = -123;
    wp::int32 var_1329;
    const wp::int32 var_1330 = 1;
    wp::int32 var_1331;
    const wp::float32 var_1332 = 1.0;
    const wp::float32 var_1333 = 1.0;
    const wp::float32 var_1334 = -1.0;
    wp::float32 var_1335;
    const wp::int32 var_1336 = 2;
    wp::int32 var_1337;
    const wp::float32 var_1338 = 1.0;
    const wp::float32 var_1339 = 1.0;
    const wp::float32 var_1340 = -1.0;
    wp::float32 var_1341;
    const wp::int32 var_1342 = 4;
    wp::int32 var_1343;
    const wp::float32 var_1344 = 1.0;
    const wp::float32 var_1345 = 1.0;
    const wp::float32 var_1346 = -1.0;
    wp::float32 var_1347;
    wp::vec_t<3, wp::float32> var_1348;
    wp::vec_t<3, wp::float32> var_1349;
    const wp::float32 var_1350 = 0.0;
    wp::vec_t<3, wp::float32> var_1351;
    wp::float32 var_1352;
    wp::float32 var_1353;
    wp::float32 var_1354;
    wp::float32 var_1355;
    wp::float32 var_1356;
    wp::float32 var_1357;
    wp::float32 var_1358;
    wp::float32 var_1359;
    wp::float32 var_1360;
    wp::float32 var_1361;
    wp::float32 var_1362;
    wp::float32 var_1363;
    wp::float32 var_1364;
    wp::float32 var_1365;
    wp::float32 var_1366;
    wp::float32 var_1367;
    wp::float32 var_1368;
    bool var_1369;
    wp::int32 var_1370;
    wp::vec_t<3, wp::float32> var_1371;
    wp::vec_t<3, wp::float32> var_1372;
    wp::float32 var_1373;
    wp::float32 var_1374;
    wp::float32 var_1375;
    wp::float32 var_1376;
    wp::float32 var_1377;
    wp::float32 var_1378;
    const wp::float32 var_1379 = 1.0;
    wp::float32 var_1380;
    wp::float32 var_1381;
    wp::float32 var_1382;
    wp::float32 var_1383;
    wp::float32 var_1384;
    wp::float32 var_1385;
    wp::float32 var_1386;
    wp::float32 var_1387;
    wp::float32 var_1388;
    wp::float32 var_1389;
    wp::float32 var_1390;
    const wp::int32 var_1391 = 1;
    wp::int32 var_1392;
    const wp::int32 var_1393 = 1;
    wp::int32 var_1394;
    const wp::int32 var_1395 = 1;
    bool var_1396;
    const wp::float32 var_1397 = 1.0;
    const wp::int32 var_1398 = 2;
    wp::float32 var_1399;
    wp::float32 var_1400;
    wp::float32 var_1401;
    wp::float32 var_1402;
    wp::int32 var_1403;
    const wp::int32 var_1404 = 1;
    const wp::int32 var_1405 = -1;
    bool var_1406;
    const wp::float32 var_1407 = 1.0;
    const wp::float32 var_1408 = -1.0;
    const wp::int32 var_1409 = 0;
    wp::float32 var_1410;
    wp::float32 var_1411;
    wp::float32 var_1412;
    wp::float32 var_1413;
    wp::int32 var_1414;
    wp::float32 var_1415;
    wp::float32 var_1416;
    wp::int32 var_1417;
    const wp::float32 var_1418 = 1.0;
    bool var_1419;
    const wp::float32 var_1420 = 1.0;
    const wp::float32 var_1421 = -1.0;
    bool var_1422;
    bool var_1423;
    const wp::float32 var_1424 = 1.0;
    const wp::int32 var_1425 = 2;
    wp::float32 var_1426;
    wp::float32 var_1427;
    wp::float32 var_1428;
    wp::float32 var_1429;
    wp::int32 var_1430;
    const wp::float32 var_1431 = 1.0;
    const wp::float32 var_1432 = -1.0;
    const wp::int32 var_1433 = 0;
    wp::float32 var_1434;
    wp::float32 var_1435;
    wp::float32 var_1436;
    wp::float32 var_1437;
    wp::int32 var_1438;
    const wp::int32 var_1439 = 1;
    bool var_1440;
    const wp::float32 var_1441 = 1.0;
    const wp::int32 var_1442 = 2;
    wp::float32 var_1443;
    wp::int32 var_1444;
    const wp::int32 var_1445 = 1;
    const wp::int32 var_1446 = -1;
    bool var_1447;
    const wp::float32 var_1448 = 1.0;
    const wp::float32 var_1449 = -1.0;
    const wp::int32 var_1450 = 0;
    wp::float32 var_1451;
    wp::int32 var_1452;
    wp::float32 var_1453;
    wp::int32 var_1454;
    wp::float32 var_1455;
    wp::float32 var_1456;
    wp::int32 var_1457;
    wp::int32 var_1458;
    wp::vec_t<3, wp::float32> var_1459;
    wp::vec_t<3, wp::float32> var_1460;
    wp::float32 var_1461;
    wp::float32 var_1462;
    const wp::int32 var_1463 = 3;
    wp::int32 var_1464;
    wp::int32 var_1465;
    wp::float32 var_1466;
    wp::float32 var_1467;
    bool var_1468;
    wp::float32 var_1469;
    wp::float32 var_1470;
    wp::float32 var_1471;
    const wp::int32 var_1472 = 6;
    wp::int32 var_1473;
    const wp::int32 var_1474 = 1;
    wp::int32 var_1475;
    wp::int32 var_1476;
    wp::int32 var_1477;
    wp::int32 var_1478;
    wp::int32 var_1479;
    wp::float32 var_1480;
    wp::float32 var_1481;
    wp::int32 var_1482;
    wp::int32 var_1483;
    wp::int32 var_1484;
    wp::float32 var_1485;
    wp::int32 var_1486;
    const wp::float32 var_1487 = 0.0;
    wp::float32 var_1488;
    const wp::int32 var_1489 = 0;
    wp::float32 var_1490;
    const wp::int32 var_1491 = 1;
    wp::float32 var_1492;
    wp::vec_t<2, wp::float32> var_1493;
    const wp::int32 var_1494 = 0;
    wp::float32 var_1495;
    const wp::int32 var_1496 = 1;
    wp::float32 var_1497;
    wp::vec_t<2, wp::float32> var_1498;
    const wp::int32 var_1499 = 0;
    wp::float32 var_1500;
    const wp::int32 var_1501 = 1;
    wp::float32 var_1502;
    wp::vec_t<2, wp::float32> var_1503;
    const wp::float32 var_1504 = 4.0;
    const wp::float32 var_1505 = -4.0;
    wp::float32 var_1506;
    const wp::int32 var_1507 = 0;
    wp::float32 var_1508;
    const wp::int32 var_1509 = 1;
    wp::float32 var_1510;
    wp::float32 var_1511;
    const wp::int32 var_1512 = 1;
    wp::float32 var_1513;
    const wp::int32 var_1514 = 0;
    wp::float32 var_1515;
    wp::float32 var_1516;
    const wp::int32 var_1517 = 0;
    wp::float32 var_1518;
    const wp::int32 var_1519 = 1;
    wp::float32 var_1520;
    wp::float32 var_1521;
    const wp::int32 var_1522 = 1;
    wp::float32 var_1523;
    const wp::int32 var_1524 = 0;
    wp::float32 var_1525;
    wp::float32 var_1526;
    wp::float32 var_1527;
    const wp::int32 var_1528 = 0;
    bool var_1529;
    const wp::float32 var_1530 = 1.0;
    const wp::float32 var_1531 = -1.0;
    wp::float32 var_1532;
    wp::float32 var_1533;
    wp::float32 var_1534;
    wp::float32 var_1535;
    bool var_1536;
    wp::float32 var_1537;
    const wp::int32 var_1538 = 0;
    bool var_1539;
    bool var_1540;
    const wp::int32 var_1541 = 0;
    const wp::int32 var_1542 = 3;
    wp::int32 var_1543;
    wp::float32 var_1544;
    wp::float32 var_1545;
    bool var_1546;
    wp::float32 var_1547;
    const wp::int32 var_1548 = 0;
    bool var_1549;
    bool var_1550;
    const wp::int32 var_1551 = 1;
    const wp::int32 var_1552 = 2;
    wp::int32 var_1553;
    wp::float32 var_1554;
    wp::int32 var_1555;
    const wp::int32 var_1556 = 4;
    const wp::int32 var_1557 = -4;
    bool var_1558;
    const wp::float32 var_1559 = 10000000000.0;
    wp::vec_t<2, wp::float32> var_1560;
    wp::mat_t<2, 3, wp::float32> var_1561;
    wp::mat_t<2, 3, wp::float32> var_1562;
    const wp::int32 var_1563 = 0;
    bool var_1564;
    const wp::int32 var_1565 = 3;
    wp::int32 var_1566;
    const wp::int32 var_1567 = 1;
    bool var_1568;
    bool var_1569;
    wp::int32 var_1570;
    const wp::int32 var_1571 = 0;
    bool var_1572;
    const wp::int32 var_1573 = 7;
    bool var_1574;
    bool var_1575;
    const wp::int32 var_1576 = 1;
    bool var_1577;
    const wp::int32 var_1578 = 2;
    bool var_1579;
    const wp::int32 var_1580 = 4;
    bool var_1581;
    bool var_1582;
    const wp::int32 var_1583 = 1;
    const wp::int32 var_1584 = 1;
    const wp::int32 var_1585 = -1;
    const wp::int32 var_1586 = 7;
    wp::int32 var_1587;
    wp::int32 var_1588;
    wp::int32 var_1589;
    const wp::int32 var_1590 = 1;
    bool var_1591;
    const wp::int32 var_1592 = 0;
    const wp::int32 var_1593 = 1;
    const wp::int32 var_1594 = 2;
    const wp::int32 var_1595 = 2;
    bool var_1596;
    const wp::int32 var_1597 = 1;
    const wp::int32 var_1598 = 2;
    const wp::int32 var_1599 = 0;
    wp::int32 var_1600;
    wp::int32 var_1601;
    wp::int32 var_1602;
    const wp::int32 var_1603 = 4;
    bool var_1604;
    const wp::int32 var_1605 = 2;
    const wp::int32 var_1606 = 0;
    const wp::int32 var_1607 = 1;
    wp::int32 var_1608;
    wp::int32 var_1609;
    wp::int32 var_1610;
    wp::int32 var_1611;
    wp::int32 var_1612;
    wp::int32 var_1613;
    wp::int32 var_1614;
    wp::int32 var_1615;
    wp::int32 var_1616;
    wp::float32 var_1617;
    wp::float32 var_1618;
    wp::float32 var_1619;
    const wp::float32 var_1620 = 0.5;
    bool var_1621;
    const wp::float32 var_1622 = 2.0;
    wp::float32 var_1623;
    wp::float32 var_1624;
    wp::float32 var_1625;
    wp::float32 var_1626;
    wp::float32 var_1627;
    const wp::float32 var_1628 = 1.0;
    wp::float32 var_1629;
    wp::float32 var_1630;
    wp::float32 var_1631;
    wp::float32 var_1632;
    wp::float32 var_1633;
    const wp::float32 var_1634 = 2.0;
    wp::float32 var_1635;
    wp::float32 var_1636;
    wp::float32 var_1637;
    wp::float32 var_1638;
    wp::float32 var_1639;
    wp::float32 var_1640;
    wp::float32 var_1641;
    wp::float32 var_1642;
    wp::float32 var_1643;
    wp::float32 var_1644;
    const wp::float32 var_1645 = 1.0;
    wp::float32 var_1646;
    wp::float32 var_1647;
    wp::float32 var_1648;
    wp::float32 var_1649;
    wp::float32 var_1650;
    wp::float32 var_1651;
    wp::float32 var_1652;
    wp::float32 var_1653;
    wp::float32 var_1654;
    wp::float32 var_1655;
    wp::int32 var_1656;
    wp::float32 var_1657;
    wp::int32 var_1658;
    const wp::int32 var_1659 = 0;
    bool var_1660;
    const wp::int32 var_1661 = 3;
    wp::int32 var_1662;
    const wp::int32 var_1663 = 1;
    bool var_1664;
    bool var_1665;
    wp::int32 var_1666;
    const wp::int32 var_1667 = 7;
    const wp::int32 var_1668 = 1;
    wp::int32 var_1669;
    wp::int32 var_1670;
    wp::int32 var_1671;
    const wp::int32 var_1672 = 1;
    bool var_1673;
    const wp::int32 var_1674 = 2;
    bool var_1675;
    const wp::int32 var_1676 = 4;
    bool var_1677;
    bool var_1678;
    const wp::int32 var_1679 = 0;
    bool var_1680;
    const wp::int32 var_1681 = 1;
    const wp::int32 var_1682 = 2;
    wp::int32 var_1683;
    wp::int32 var_1684;
    const wp::int32 var_1685 = 1;
    bool var_1686;
    const wp::int32 var_1687 = 2;
    const wp::int32 var_1688 = 0;
    wp::int32 var_1689;
    wp::int32 var_1690;
    const wp::int32 var_1691 = 2;
    bool var_1692;
    const wp::int32 var_1693 = 0;
    const wp::int32 var_1694 = 1;
    wp::int32 var_1695;
    wp::int32 var_1696;
    wp::int32 var_1697;
    wp::float32 var_1698;
    wp::float32 var_1699;
    wp::float32 var_1700;
    wp::float32 var_1701;
    bool var_1702;
    wp::int32 var_1703;
    wp::int32 var_1704;
    const wp::int32 var_1705 = 3;
    wp::int32 var_1706;
    wp::int32 var_1707;
    const wp::int32 var_1708 = 1;
    wp::int32 var_1709;
    wp::int32 var_1710;
    const wp::int32 var_1711 = 1;
    const wp::float32 var_1712 = 1.0;
    wp::float32 var_1713;
    wp::float32 var_1714;
    wp::int32 var_1715;
    const wp::int32 var_1716 = 1;
    const wp::int32 var_1717 = -1;
    const wp::float32 var_1718 = 1.0;
    wp::float32 var_1719;
    wp::float32 var_1720;
    wp::int32 var_1721;
    const wp::float32 var_1722 = 2.0;
    wp::float32 var_1723;
    wp::float32 var_1724;
    wp::float32 var_1725;
    wp::float32 var_1726;
    wp::float32 var_1727;
    wp::float32 var_1728;
    const wp::int32 var_1729 = 1;
    wp::int32 var_1730;
    wp::int32 var_1731;
    const wp::int32 var_1732 = 0;
    bool var_1733;
    const wp::int32 var_1734 = 1;
    wp::int32 var_1735;
    wp::int32 var_1736;
    const wp::int32 var_1737 = 0;
    bool var_1738;
    bool var_1739;
    const wp::float32 var_1740 = 1.0;
    wp::float32 var_1741;
    const wp::float32 var_1742 = 1.0;
    wp::float32 var_1743;
    wp::float32 var_1744;
    wp::float32 var_1745;
    wp::float32 var_1746;
    wp::float32 var_1747;
    wp::float32 var_1748;
    wp::float32 var_1749;
    wp::float32 var_1750;
    wp::float32 var_1751;
    wp::float32 var_1752;
    wp::float32 var_1753;
    wp::int32 var_1754;
    wp::int32 var_1755;
    wp::int32 var_1756;
    wp::int32 var_1757;
    wp::float32 var_1758;
    wp::int32 var_1759;
    wp::int32 var_1760;
    wp::int32 var_1761;
    wp::int32 var_1762;
    wp::int32 var_1763;
    const wp::int32 var_1764 = 0;
    bool var_1765;
    const wp::int32 var_1766 = 1;
    const wp::int32 var_1767 = -1;
    bool var_1768;
    const wp::int32 var_1769 = 3;
    const wp::int32 var_1770 = -3;
    bool var_1771;
    const wp::int32 var_1772 = 1;
    const wp::int32 var_1773 = 1;
    const wp::int32 var_1774 = -1;
    wp::int32 var_1775;
    const wp::float32 var_1776 = 2.0;
    wp::float32 var_1777;
    wp::vec_t<3, wp::float32> var_1778;
    wp::vec_t<3, wp::float32> var_1779;
    const wp::int32 var_1780 = 0;
    bool var_1781;
    wp::float32 var_1782;
    wp::float32 var_1783;
    wp::float32 var_1784;
    wp::float32 var_1785;
    wp::float32 var_1786;
    wp::float32 var_1787;
    wp::float32 var_1788;
    const wp::int32 var_1789 = 0;
    bool var_1790;
    bool var_1791;
    bool var_1792;
    wp::float32 var_1793;
    wp::float32 var_1794;
    wp::float32 var_1795;
    wp::float32 var_1796;
    wp::float32 var_1797;
    wp::float32 var_1798;
    wp::float32 var_1799;
    const wp::int32 var_1800 = 0;
    bool var_1801;
    bool var_1802;
    bool var_1803;
    wp::float32 var_1804;
    wp::float32 var_1805;
    wp::float32 var_1806;
    wp::float32 var_1807;
    const wp::int32 var_1808 = 1;
    bool var_1809;
    wp::float32 var_1810;
    wp::float32 var_1811;
    wp::float32 var_1812;
    wp::float32 var_1813;
    wp::float32 var_1814;
    wp::float32 var_1815;
    wp::float32 var_1816;
    const wp::int32 var_1817 = 0;
    bool var_1818;
    bool var_1819;
    bool var_1820;
    wp::float32 var_1821;
    wp::float32 var_1822;
    wp::float32 var_1823;
    wp::float32 var_1824;
    wp::float32 var_1825;
    wp::float32 var_1826;
    wp::float32 var_1827;
    const wp::int32 var_1828 = 0;
    bool var_1829;
    bool var_1830;
    bool var_1831;
    wp::float32 var_1832;
    wp::float32 var_1833;
    wp::float32 var_1834;
    wp::float32 var_1835;
    wp::float32 var_1836;
    const wp::int32 var_1837 = 2;
    bool var_1838;
    wp::float32 var_1839;
    wp::float32 var_1840;
    wp::float32 var_1841;
    wp::float32 var_1842;
    wp::float32 var_1843;
    wp::float32 var_1844;
    wp::float32 var_1845;
    const wp::int32 var_1846 = 0;
    bool var_1847;
    bool var_1848;
    bool var_1849;
    wp::float32 var_1850;
    wp::float32 var_1851;
    wp::float32 var_1852;
    wp::float32 var_1853;
    wp::float32 var_1854;
    wp::float32 var_1855;
    wp::float32 var_1856;
    const wp::int32 var_1857 = 0;
    bool var_1858;
    bool var_1859;
    bool var_1860;
    wp::float32 var_1861;
    wp::float32 var_1862;
    wp::float32 var_1863;
    wp::float32 var_1864;
    wp::float32 var_1865;
    wp::float32 var_1866;
    wp::float32 var_1867;
    wp::int32 var_1868;
    wp::float32 var_1869;
    wp::int32 var_1870;
    wp::float32 var_1871;
    wp::int32 var_1872;
    wp::float32 var_1873;
    wp::int32 var_1874;
    wp::float32 var_1875;
    wp::int32 var_1876;
    wp::float32 var_1877;
    wp::int32 var_1878;
    wp::float32 var_1879;
    wp::int32 var_1880;
    wp::float32 var_1881;
    wp::int32 var_1882;
    wp::int32 var_1883;
    wp::int32 var_1884;
    wp::int32 var_1885;
    wp::int32 var_1886;
    wp::vec_t<3, wp::float32> var_1887;
    wp::vec_t<3, wp::float32> var_1888;
    wp::vec_t<3, wp::float32> var_1889;
    wp::vec_t<3, wp::float32> var_1890;
    wp::float32 var_1891;
    wp::vec_t<3, wp::float32> var_1892;
    wp::vec_t<3, wp::float32> var_1893;
    const wp::int32 var_1894 = 3;
    const wp::int32 var_1895 = -3;
    bool var_1896;
    wp::float32 var_1897;
    wp::vec_t<3, wp::float32> var_1898;
    wp::vec_t<3, wp::float32> var_1899;
    wp::vec_t<3, wp::float32> var_1900;
    wp::vec_t<3, wp::float32> var_1901;
    wp::float32 var_1902;
    wp::vec_t<3, wp::float32> var_1903;
    wp::vec_t<3, wp::float32> var_1904;
    wp::float32 var_1905;
    wp::vec_t<3, wp::float32> var_1906;
    wp::vec_t<3, wp::float32> var_1907;
    wp::float32 var_1908;
    wp::vec_t<3, wp::float32> var_1909;
    wp::vec_t<3, wp::float32> var_1910;
    wp::vec_t<2, wp::float32> var_1911;
    const wp::int32 var_1912 = 0;
    wp::float32 var_1913;
    const wp::int32 var_1914 = 1;
    wp::float32 var_1915;
    const wp::int32 var_1916 = 2;
    wp::float32 var_1917;
    const wp::int32 var_1918 = 0;
    wp::float32 var_1919;
    const wp::int32 var_1920 = 1;
    wp::float32 var_1921;
    const wp::int32 var_1922 = 2;
    wp::float32 var_1923;
    wp::mat_t<2, 3, wp::float32> var_1924;
    const wp::int32 var_1925 = 0;
    wp::float32 var_1926;
    const wp::int32 var_1927 = 1;
    wp::float32 var_1928;
    const wp::int32 var_1929 = 2;
    wp::float32 var_1930;
    const wp::int32 var_1931 = 0;
    wp::float32 var_1932;
    const wp::int32 var_1933 = 1;
    wp::float32 var_1934;
    const wp::int32 var_1935 = 2;
    wp::float32 var_1936;
    wp::mat_t<2, 3, wp::float32> var_1937;
    //---------
    // forward
    // def capsule_box(                                                                       <L 1158>
    // boxmatT = wp.transpose(box_rot)                                                        <L 1185>
    var_0 = wp::transpose(var_box_rot);
    // pos = boxmatT @ (capsule_pos - box_pos)                                                <L 1186>
    var_1 = wp::sub(var_capsule_pos, var_box_pos);
    var_2 = wp::mul(var_0, var_1);
    // axis = boxmatT @ capsule_axis                                                          <L 1187>
    var_3 = wp::mul(var_0, var_capsule_axis);
    // halfaxis = axis * capsule_half_length  # halfaxis is the capsule direction             <L 1188>
    var_4 = wp::mul(var_3, var_capsule_half_length);
    // axisdir = wp.int32(halfaxis[0] > 0.0) + 2 * wp.int32(halfaxis[1] > 0.0) + 4 * wp.int32(halfaxis[2] > 0.0)       <L 1189>
    var_6 = wp::extract(var_4, var_5);
    var_8 = (var_6 > var_7);
    var_9 = wp::int32(var_8);
    var_12 = wp::extract(var_4, var_11);
    var_14 = (var_12 > var_13);
    var_15 = wp::int32(var_14);
    var_16 = wp::mul(var_10, var_15);
    var_17 = wp::add(var_9, var_16);
    var_20 = wp::extract(var_4, var_19);
    var_22 = (var_20 > var_21);
    var_23 = wp::int32(var_22);
    var_24 = wp::mul(var_18, var_23);
    var_25 = wp::add(var_17, var_24);
    // bestdist = wp.float32(1.0e32)                                                          <L 1192>
    var_27 = wp::float32(var_26);
    // bestsegmentpos = wp.float32(-12)                                                       <L 1193>
    var_30 = wp::float32(var_29);
    // cltype = wp.int32(-4)                                                                  <L 1202>
    var_33 = wp::int32(var_32);
    // clface = wp.int32(-12)                                                                 <L 1207>
    var_36 = wp::int32(var_35);
    // for i in range(-1, 2, 2):                                                              <L 1210>
    var_40 = wp::range(var_37, var_38, var_39);
    start_for_0:;
        if (iter_cmp(var_40) == 0) goto end_for_0;
        var_41 = wp::iter_next(var_40);
        // axisTip = pos + wp.float32(i) * halfaxis                                           <L 1211>
        var_42 = wp::float32(var_41);
        var_43 = wp::mul(var_42, var_4);
        var_44 = wp::add(var_2, var_43);
        // boxPoint = wp.vec3(axisTip)                                                        <L 1212>
        var_45 = wp::vec_t<3, wp::float32>(var_44);
        // n_out = wp.int32(0)                                                                <L 1214>
        var_47 = wp::int32(var_46);
        // ax_out = wp.int32(-1)                                                              <L 1215>
        var_50 = wp::int32(var_49);
        // for j in range(3):                                                                 <L 1217>
        // if boxPoint[j] < -box_size[j]:                                                     <L 1218>
        var_52 = wp::extract(var_45, var_51);
        var_53 = wp::extract(var_box_size, var_51);
        var_54 = wp::neg(var_53);
        var_55 = (var_52 < var_54);
        if (var_55) {
            // n_out += 1                                                                     <L 1219>
            var_57 = wp::add(var_47, var_56);
            // ax_out = j                                                                     <L 1220>
            var_58 = wp::copy(var_51);
            // boxPoint[j] = -box_size[j]                                                     <L 1221>
            var_59 = wp::extract(var_box_size, var_51);
            var_60 = wp::neg(var_59);
            wp::assign_inplace(var_45, var_51, var_60);
        }
        var_61 = wp::where(var_55, var_57, var_47);
        var_62 = wp::where(var_55, var_58, var_50);
        if (!var_55) {
            // elif boxPoint[j] > box_size[j]:                                                <L 1222>
            var_63 = wp::extract(var_45, var_51);
            var_64 = wp::extract(var_box_size, var_51);
            var_65 = (var_63 > var_64);
            if (var_65) {
                // n_out += 1                                                                 <L 1223>
                var_67 = wp::add(var_61, var_66);
                // ax_out = j                                                                 <L 1224>
                var_68 = wp::copy(var_51);
                // boxPoint[j] = box_size[j]                                                  <L 1225>
                var_69 = wp::extract(var_box_size, var_51);
                wp::assign_inplace(var_45, var_51, var_69);
            }
            var_70 = wp::where(var_65, var_67, var_61);
            var_71 = wp::where(var_65, var_68, var_62);
        }
        var_72 = wp::where(var_55, var_61, var_70);
        var_73 = wp::where(var_55, var_62, var_71);
        // if boxPoint[j] < -box_size[j]:                                                     <L 1218>
        var_75 = wp::extract(var_45, var_74);
        var_76 = wp::extract(var_box_size, var_74);
        var_77 = wp::neg(var_76);
        var_78 = (var_75 < var_77);
        if (var_78) {
            // n_out += 1                                                                     <L 1219>
            var_80 = wp::add(var_72, var_79);
            // ax_out = j                                                                     <L 1220>
            var_81 = wp::copy(var_74);
            // boxPoint[j] = -box_size[j]                                                     <L 1221>
            var_82 = wp::extract(var_box_size, var_74);
            var_83 = wp::neg(var_82);
            wp::assign_inplace(var_45, var_74, var_83);
        }
        var_84 = wp::where(var_78, var_80, var_72);
        var_85 = wp::where(var_78, var_81, var_73);
        if (!var_78) {
            // elif boxPoint[j] > box_size[j]:                                                <L 1222>
            var_86 = wp::extract(var_45, var_74);
            var_87 = wp::extract(var_box_size, var_74);
            var_88 = (var_86 > var_87);
            if (var_88) {
                // n_out += 1                                                                 <L 1223>
                var_90 = wp::add(var_84, var_89);
                // ax_out = j                                                                 <L 1224>
                var_91 = wp::copy(var_74);
                // boxPoint[j] = box_size[j]                                                  <L 1225>
                var_92 = wp::extract(var_box_size, var_74);
                wp::assign_inplace(var_45, var_74, var_92);
            }
            var_93 = wp::where(var_88, var_90, var_84);
            var_94 = wp::where(var_88, var_91, var_85);
        }
        var_95 = wp::where(var_78, var_84, var_93);
        var_96 = wp::where(var_78, var_85, var_94);
        // if boxPoint[j] < -box_size[j]:                                                     <L 1218>
        var_98 = wp::extract(var_45, var_97);
        var_99 = wp::extract(var_box_size, var_97);
        var_100 = wp::neg(var_99);
        var_101 = (var_98 < var_100);
        if (var_101) {
            // n_out += 1                                                                     <L 1219>
            var_103 = wp::add(var_95, var_102);
            // ax_out = j                                                                     <L 1220>
            var_104 = wp::copy(var_97);
            // boxPoint[j] = -box_size[j]                                                     <L 1221>
            var_105 = wp::extract(var_box_size, var_97);
            var_106 = wp::neg(var_105);
            wp::assign_inplace(var_45, var_97, var_106);
        }
        var_107 = wp::where(var_101, var_103, var_95);
        var_108 = wp::where(var_101, var_104, var_96);
        if (!var_101) {
            // elif boxPoint[j] > box_size[j]:                                                <L 1222>
            var_109 = wp::extract(var_45, var_97);
            var_110 = wp::extract(var_box_size, var_97);
            var_111 = (var_109 > var_110);
            if (var_111) {
                // n_out += 1                                                                 <L 1223>
                var_113 = wp::add(var_107, var_112);
                // ax_out = j                                                                 <L 1224>
                var_114 = wp::copy(var_97);
                // boxPoint[j] = box_size[j]                                                  <L 1225>
                var_115 = wp::extract(var_box_size, var_97);
                wp::assign_inplace(var_45, var_97, var_115);
            }
            var_116 = wp::where(var_111, var_113, var_107);
            var_117 = wp::where(var_111, var_114, var_108);
        }
        var_118 = wp::where(var_101, var_107, var_116);
        var_119 = wp::where(var_101, var_108, var_117);
        // if n_out > 1:                                                                      <L 1227>
        var_121 = (var_118 > var_120);
        if (var_121) {
            // continue                                                                       <L 1228>
            goto start_for_0;
        }
        // dist = wp.length_sq(boxPoint - axisTip)                                            <L 1230>
        var_122 = wp::sub(var_45, var_44);
        var_123 = wp::length_sq(var_122);
        // if dist < bestdist:                                                                <L 1232>
        var_124 = (var_123 < var_27);
        if (var_124) {
            // bestdist = dist                                                                <L 1233>
            var_125 = wp::copy(var_123);
            // bestsegmentpos = wp.float32(i)                                                 <L 1234>
            var_126 = wp::float32(var_41);
            // cltype = -2 + i                                                                <L 1235>
            var_129 = wp::add(var_128, var_41);
            // clface = ax_out                                                                <L 1236>
            var_130 = wp::copy(var_119);
        }
        var_131 = wp::where(var_124, var_125, var_27);
        var_132 = wp::where(var_124, var_126, var_30);
        var_133 = wp::where(var_124, var_129, var_33);
        var_134 = wp::where(var_124, var_130, var_36);
        wp::assign(var_27, var_131);
        wp::assign(var_30, var_132);
        wp::assign(var_33, var_133);
        wp::assign(var_36, var_134);
        goto start_for_0;
    end_for_0:;
    // clcorner = wp.int32(-123)  # which corner is the closest                               <L 1239>
    var_137 = wp::int32(var_136);
    // cledge = wp.int32(-123)  # which axis                                                  <L 1240>
    var_140 = wp::int32(var_139);
    // bestboxpos = wp.float32(0.0)                                                           <L 1241>
    var_142 = wp::float32(var_141);
    // for i in range(8):                                                                     <L 1243>
    // for j in range(3):                                                                     <L 1244>
    var_145 = wp::range(var_144);
    start_for_2:;
        if (iter_cmp(var_145) == 0) goto end_for_2;
        var_146 = wp::iter_next(var_145);
        // if i & (1 << j) != 0:                                                              <L 1245>
        var_148 = wp::lshift(var_147, var_146);
        var_149 = wp::bit_and(var_143, var_148);
        var_151 = (var_149 != var_150);
        if (var_151) {
            // continue                                                                       <L 1246>
            goto start_for_2;
        }
        // c2 = wp.int32(-123)                                                                <L 1248>
        var_154 = wp::int32(var_153);
        // box_pt = wp.cw_mul(                                                                <L 1251>
        // wp.vec3(                                                                           <L 1252>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1253>
        var_156 = wp::bit_and(var_143, var_155);
        var_160 = wp::where(var_156, var_157, var_159);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1254>
        var_162 = wp::bit_and(var_143, var_161);
        var_166 = wp::where(var_162, var_163, var_165);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1255>
        var_168 = wp::bit_and(var_143, var_167);
        var_172 = wp::where(var_168, var_169, var_171);
        var_173 = wp::vec_t<3, wp::float32>(var_160, var_166, var_172);
        // box_size,                                                                          <L 1257>
        var_174 = wp::cw_mul(var_173, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1259>
        wp::assign_inplace(var_174, var_146, var_175);
        // dif = box_pt - pos                                                                 <L 1262>
        var_176 = wp::sub(var_174, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1264>
        var_177 = wp::extract(var_box_size, var_146);
        var_178 = wp::neg(var_177);
        var_179 = wp::extract(var_176, var_146);
        var_180 = wp::mul(var_178, var_179);
        // v = wp.dot(halfaxis, dif)                                                          <L 1265>
        var_181 = wp::dot(var_4, var_176);
        // ma = box_size[j] * box_size[j]                                                     <L 1266>
        var_182 = wp::extract(var_box_size, var_146);
        var_183 = wp::extract(var_box_size, var_146);
        var_184 = wp::mul(var_182, var_183);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1267>
        var_185 = wp::extract(var_box_size, var_146);
        var_186 = wp::neg(var_185);
        var_187 = wp::extract(var_4, var_146);
        var_188 = wp::mul(var_186, var_187);
        // mc = capsule_half_length * capsule_half_length                                     <L 1268>
        var_189 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1269>
        var_190 = wp::mul(var_184, var_189);
        var_191 = wp::mul(var_188, var_188);
        var_192 = wp::sub(var_190, var_191);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1270>
        var_193 = wp::abs(var_192);
        var_195 = (var_193 < var_194);
        if (var_195) {
            // continue                                                                       <L 1271>
            goto start_for_2;
        }
        // idet = 1.0 / det                                                                   <L 1273>
        var_197 = wp::div(var_196, var_192);
        // x1 = wp.float32((mc * u - mb * v) * idet)                                          <L 1276>
        var_198 = wp::mul(var_189, var_180);
        var_199 = wp::mul(var_188, var_181);
        var_200 = wp::sub(var_198, var_199);
        var_201 = wp::mul(var_200, var_197);
        var_202 = wp::float32(var_201);
        // x2 = wp.float32((ma * v - mb * u) * idet)                                          <L 1277>
        var_203 = wp::mul(var_184, var_181);
        var_204 = wp::mul(var_188, var_180);
        var_205 = wp::sub(var_203, var_204);
        var_206 = wp::mul(var_205, var_197);
        var_207 = wp::float32(var_206);
        // s1 = wp.int32(1)                                                                   <L 1279>
        var_209 = wp::int32(var_208);
        // s2 = wp.int32(1)                                                                   <L 1280>
        var_211 = wp::int32(var_210);
        // if x1 > 1:                                                                         <L 1282>
        var_213 = (var_202 > var_212);
        if (var_213) {
            // x1 = 1.0                                                                       <L 1283>
            // s1 = 2                                                                         <L 1284>
            // x2 = safe_div(v - mb, mc)                                                      <L 1285>
            var_216 = wp::sub(var_181, var_188);
            var_217 = safe_div_1(var_216, var_189);
        }
        var_218 = wp::where(var_213, var_214, var_202);
        var_219 = wp::where(var_213, var_217, var_207);
        var_220 = wp::where(var_213, var_215, var_209);
        if (!var_213) {
            // elif x1 < -1:                                                                  <L 1286>
            var_223 = (var_218 < var_222);
            if (var_223) {
                // x1 = -1.0                                                                  <L 1287>
                // s1 = 0                                                                     <L 1288>
                // x2 = safe_div(v + mb, mc)                                                  <L 1289>
                var_227 = wp::add(var_181, var_188);
                var_228 = safe_div_1(var_227, var_189);
            }
            var_229 = wp::where(var_223, var_225, var_218);
            var_230 = wp::where(var_223, var_228, var_219);
            var_231 = wp::where(var_223, var_226, var_220);
        }
        var_232 = wp::where(var_213, var_218, var_229);
        var_233 = wp::where(var_213, var_219, var_230);
        var_234 = wp::where(var_213, var_220, var_231);
        // x2_over = x2 > 1.0                                                                 <L 1291>
        var_236 = (var_233 > var_235);
        // if x2_over or x2 < -1.0:                                                           <L 1292>
        var_239 = (var_233 < var_238);
        var_240 = var_236 || var_239;
        if (var_240) {
            // if x2_over:                                                                    <L 1293>
            if (var_236) {
                // x2 = 1.0                                                                   <L 1294>
                // s2 = 2                                                                     <L 1295>
                // x1 = safe_div(u - mb, ma)                                                  <L 1296>
                var_243 = wp::sub(var_180, var_188);
                var_244 = safe_div_1(var_243, var_184);
            }
            var_245 = wp::where(var_236, var_244, var_232);
            var_246 = wp::where(var_236, var_241, var_233);
            var_247 = wp::where(var_236, var_242, var_211);
            if (!var_236) {
                // x2 = -1.0                                                                  <L 1298>
                // s2 = 0                                                                     <L 1299>
                // x1 = safe_div(u + mb, ma)                                                  <L 1300>
                var_251 = wp::add(var_180, var_188);
                var_252 = safe_div_1(var_251, var_184);
            }
            var_253 = wp::where(var_236, var_245, var_252);
            var_254 = wp::where(var_236, var_246, var_249);
            var_255 = wp::where(var_236, var_247, var_250);
            // if x1 > 1:                                                                     <L 1302>
            var_257 = (var_253 > var_256);
            if (var_257) {
                // x1 = 1.0                                                                   <L 1303>
                // s1 = 2                                                                     <L 1304>
            }
            var_260 = wp::where(var_257, var_258, var_253);
            var_261 = wp::where(var_257, var_259, var_234);
            if (!var_257) {
                // elif x1 < -1:                                                              <L 1305>
                var_264 = (var_260 < var_263);
                if (var_264) {
                    // x1 = -1.0                                                              <L 1306>
                    // s1 = 0                                                                 <L 1307>
                }
                var_268 = wp::where(var_264, var_266, var_260);
                var_269 = wp::where(var_264, var_267, var_261);
            }
            var_270 = wp::where(var_257, var_260, var_268);
            var_271 = wp::where(var_257, var_261, var_269);
        }
        var_272 = wp::where(var_240, var_270, var_232);
        var_273 = wp::where(var_240, var_254, var_233);
        var_274 = wp::where(var_240, var_271, var_234);
        var_275 = wp::where(var_240, var_255, var_211);
        // dif -= halfaxis * x2                                                               <L 1309>
        var_276 = wp::mul(var_4, var_273);
        var_277 = wp::sub(var_176, var_276);
        // dif[j] += box_size[j] * x1                                                         <L 1310>
        var_278 = wp::extract(var_box_size, var_146);
        var_279 = wp::mul(var_278, var_272);
        wp::add_inplace(var_277, var_146, var_279);
        // ct = s1 * 3 + s2                                                                   <L 1313>
        var_281 = wp::mul(var_274, var_280);
        var_282 = wp::add(var_281, var_275);
        // dif_sq = wp.length_sq(dif)                                                         <L 1315>
        var_283 = wp::length_sq(var_277);
        // if dif_sq < bestdist - MJ_MINVAL:                                                  <L 1316>
        var_284 = wp::sub(var_27, var_194);
        var_285 = (var_283 < var_284);
        if (var_285) {
            // bestdist = dif_sq                                                              <L 1317>
            var_286 = wp::copy(var_283);
            // bestsegmentpos = x2                                                            <L 1318>
            var_287 = wp::copy(var_273);
            // bestboxpos = x1                                                                <L 1319>
            var_288 = wp::copy(var_272);
            // c2 = ct // 6                                                                   <L 1321>
            var_290 = wp::floordiv(var_282, var_289);
            // clcorner = i + (1 << j) * c2  # index of closest box corner                    <L 1323>
            var_292 = wp::lshift(var_291, var_146);
            var_293 = wp::mul(var_292, var_290);
            var_294 = wp::add(var_143, var_293);
            // cledge = j  # axis index of closest box edge                                   <L 1324>
            var_295 = wp::copy(var_146);
            // cltype = ct  # encoded collision configuration                                 <L 1325>
            var_296 = wp::copy(var_282);
        }
        var_297 = wp::where(var_285, var_286, var_27);
        var_298 = wp::where(var_285, var_287, var_30);
        var_299 = wp::where(var_285, var_296, var_33);
        var_300 = wp::where(var_285, var_294, var_137);
        var_301 = wp::where(var_285, var_295, var_140);
        var_302 = wp::where(var_285, var_288, var_142);
        var_303 = wp::where(var_285, var_290, var_154);
        wp::assign(var_27, var_297);
        wp::assign(var_30, var_298);
        wp::assign(var_33, var_299);
        wp::assign(var_137, var_300);
        wp::assign(var_140, var_301);
        wp::assign(var_142, var_302);
        goto start_for_2;
    end_for_2:;
    // for j in range(3):                                                                     <L 1244>
    var_306 = wp::range(var_305);
    start_for_4:;
        if (iter_cmp(var_306) == 0) goto end_for_4;
        var_307 = wp::iter_next(var_306);
        // if i & (1 << j) != 0:                                                              <L 1245>
        var_309 = wp::lshift(var_308, var_307);
        var_310 = wp::bit_and(var_304, var_309);
        var_312 = (var_310 != var_311);
        if (var_312) {
            // continue                                                                       <L 1246>
            goto start_for_4;
        }
        // c2 = wp.int32(-123)                                                                <L 1248>
        var_315 = wp::int32(var_314);
        // box_pt = wp.cw_mul(                                                                <L 1251>
        // wp.vec3(                                                                           <L 1252>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1253>
        var_317 = wp::bit_and(var_304, var_316);
        var_321 = wp::where(var_317, var_318, var_320);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1254>
        var_323 = wp::bit_and(var_304, var_322);
        var_327 = wp::where(var_323, var_324, var_326);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1255>
        var_329 = wp::bit_and(var_304, var_328);
        var_333 = wp::where(var_329, var_330, var_332);
        var_334 = wp::vec_t<3, wp::float32>(var_321, var_327, var_333);
        // box_size,                                                                          <L 1257>
        var_335 = wp::cw_mul(var_334, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1259>
        wp::assign_inplace(var_335, var_307, var_336);
        // dif = box_pt - pos                                                                 <L 1262>
        var_337 = wp::sub(var_335, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1264>
        var_338 = wp::extract(var_box_size, var_307);
        var_339 = wp::neg(var_338);
        var_340 = wp::extract(var_337, var_307);
        var_341 = wp::mul(var_339, var_340);
        // v = wp.dot(halfaxis, dif)                                                          <L 1265>
        var_342 = wp::dot(var_4, var_337);
        // ma = box_size[j] * box_size[j]                                                     <L 1266>
        var_343 = wp::extract(var_box_size, var_307);
        var_344 = wp::extract(var_box_size, var_307);
        var_345 = wp::mul(var_343, var_344);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1267>
        var_346 = wp::extract(var_box_size, var_307);
        var_347 = wp::neg(var_346);
        var_348 = wp::extract(var_4, var_307);
        var_349 = wp::mul(var_347, var_348);
        // mc = capsule_half_length * capsule_half_length                                     <L 1268>
        var_350 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1269>
        var_351 = wp::mul(var_345, var_350);
        var_352 = wp::mul(var_349, var_349);
        var_353 = wp::sub(var_351, var_352);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1270>
        var_354 = wp::abs(var_353);
        var_355 = (var_354 < var_194);
        if (var_355) {
            // continue                                                                       <L 1271>
            wp::assign(var_303, var_315);
            wp::assign(var_174, var_335);
            wp::assign(var_277, var_337);
            wp::assign(var_180, var_341);
            wp::assign(var_181, var_342);
            wp::assign(var_184, var_345);
            wp::assign(var_188, var_349);
            wp::assign(var_189, var_350);
            wp::assign(var_192, var_353);
            goto start_for_4;
        }
        var_356 = wp::where(var_355, var_303, var_315);
        var_357 = wp::where(var_355, var_174, var_335);
        var_358 = wp::where(var_355, var_277, var_337);
        var_359 = wp::where(var_355, var_180, var_341);
        var_360 = wp::where(var_355, var_181, var_342);
        var_361 = wp::where(var_355, var_184, var_345);
        var_362 = wp::where(var_355, var_188, var_349);
        var_363 = wp::where(var_355, var_189, var_350);
        var_364 = wp::where(var_355, var_192, var_353);
        // idet = 1.0 / det                                                                   <L 1273>
        var_366 = wp::div(var_365, var_364);
        // x1 = wp.float32((mc * u - mb * v) * idet)                                          <L 1276>
        var_367 = wp::mul(var_363, var_359);
        var_368 = wp::mul(var_362, var_360);
        var_369 = wp::sub(var_367, var_368);
        var_370 = wp::mul(var_369, var_366);
        var_371 = wp::float32(var_370);
        // x2 = wp.float32((ma * v - mb * u) * idet)                                          <L 1277>
        var_372 = wp::mul(var_361, var_360);
        var_373 = wp::mul(var_362, var_359);
        var_374 = wp::sub(var_372, var_373);
        var_375 = wp::mul(var_374, var_366);
        var_376 = wp::float32(var_375);
        // s1 = wp.int32(1)                                                                   <L 1279>
        var_378 = wp::int32(var_377);
        // s2 = wp.int32(1)                                                                   <L 1280>
        var_380 = wp::int32(var_379);
        // if x1 > 1:                                                                         <L 1282>
        var_382 = (var_371 > var_381);
        if (var_382) {
            // x1 = 1.0                                                                       <L 1283>
            // s1 = 2                                                                         <L 1284>
            // x2 = safe_div(v - mb, mc)                                                      <L 1285>
            var_385 = wp::sub(var_360, var_362);
            var_386 = safe_div_1(var_385, var_363);
        }
        var_387 = wp::where(var_382, var_383, var_371);
        var_388 = wp::where(var_382, var_386, var_376);
        var_389 = wp::where(var_382, var_384, var_378);
        if (!var_382) {
            // elif x1 < -1:                                                                  <L 1286>
            var_392 = (var_387 < var_391);
            if (var_392) {
                // x1 = -1.0                                                                  <L 1287>
                // s1 = 0                                                                     <L 1288>
                // x2 = safe_div(v + mb, mc)                                                  <L 1289>
                var_396 = wp::add(var_360, var_362);
                var_397 = safe_div_1(var_396, var_363);
            }
            var_398 = wp::where(var_392, var_394, var_387);
            var_399 = wp::where(var_392, var_397, var_388);
            var_400 = wp::where(var_392, var_395, var_389);
        }
        var_401 = wp::where(var_382, var_387, var_398);
        var_402 = wp::where(var_382, var_388, var_399);
        var_403 = wp::where(var_382, var_389, var_400);
        // x2_over = x2 > 1.0                                                                 <L 1291>
        var_405 = (var_402 > var_404);
        // if x2_over or x2 < -1.0:                                                           <L 1292>
        var_408 = (var_402 < var_407);
        var_409 = var_405 || var_408;
        if (var_409) {
            // if x2_over:                                                                    <L 1293>
            if (var_405) {
                // x2 = 1.0                                                                   <L 1294>
                // s2 = 2                                                                     <L 1295>
                // x1 = safe_div(u - mb, ma)                                                  <L 1296>
                var_412 = wp::sub(var_359, var_362);
                var_413 = safe_div_1(var_412, var_361);
            }
            var_414 = wp::where(var_405, var_413, var_401);
            var_415 = wp::where(var_405, var_410, var_402);
            var_416 = wp::where(var_405, var_411, var_380);
            if (!var_405) {
                // x2 = -1.0                                                                  <L 1298>
                // s2 = 0                                                                     <L 1299>
                // x1 = safe_div(u + mb, ma)                                                  <L 1300>
                var_420 = wp::add(var_359, var_362);
                var_421 = safe_div_1(var_420, var_361);
            }
            var_422 = wp::where(var_405, var_414, var_421);
            var_423 = wp::where(var_405, var_415, var_418);
            var_424 = wp::where(var_405, var_416, var_419);
            // if x1 > 1:                                                                     <L 1302>
            var_426 = (var_422 > var_425);
            if (var_426) {
                // x1 = 1.0                                                                   <L 1303>
                // s1 = 2                                                                     <L 1304>
            }
            var_429 = wp::where(var_426, var_427, var_422);
            var_430 = wp::where(var_426, var_428, var_403);
            if (!var_426) {
                // elif x1 < -1:                                                              <L 1305>
                var_433 = (var_429 < var_432);
                if (var_433) {
                    // x1 = -1.0                                                              <L 1306>
                    // s1 = 0                                                                 <L 1307>
                }
                var_437 = wp::where(var_433, var_435, var_429);
                var_438 = wp::where(var_433, var_436, var_430);
            }
            var_439 = wp::where(var_426, var_429, var_437);
            var_440 = wp::where(var_426, var_430, var_438);
        }
        var_441 = wp::where(var_409, var_439, var_401);
        var_442 = wp::where(var_409, var_423, var_402);
        var_443 = wp::where(var_409, var_440, var_403);
        var_444 = wp::where(var_409, var_424, var_380);
        // dif -= halfaxis * x2                                                               <L 1309>
        var_445 = wp::mul(var_4, var_442);
        var_446 = wp::sub(var_358, var_445);
        // dif[j] += box_size[j] * x1                                                         <L 1310>
        var_447 = wp::extract(var_box_size, var_307);
        var_448 = wp::mul(var_447, var_441);
        wp::add_inplace(var_446, var_307, var_448);
        // ct = s1 * 3 + s2                                                                   <L 1313>
        var_450 = wp::mul(var_443, var_449);
        var_451 = wp::add(var_450, var_444);
        // dif_sq = wp.length_sq(dif)                                                         <L 1315>
        var_452 = wp::length_sq(var_446);
        // if dif_sq < bestdist - MJ_MINVAL:                                                  <L 1316>
        var_453 = wp::sub(var_27, var_194);
        var_454 = (var_452 < var_453);
        if (var_454) {
            // bestdist = dif_sq                                                              <L 1317>
            var_455 = wp::copy(var_452);
            // bestsegmentpos = x2                                                            <L 1318>
            var_456 = wp::copy(var_442);
            // bestboxpos = x1                                                                <L 1319>
            var_457 = wp::copy(var_441);
            // c2 = ct // 6                                                                   <L 1321>
            var_459 = wp::floordiv(var_451, var_458);
            // clcorner = i + (1 << j) * c2  # index of closest box corner                    <L 1323>
            var_461 = wp::lshift(var_460, var_307);
            var_462 = wp::mul(var_461, var_459);
            var_463 = wp::add(var_304, var_462);
            // cledge = j  # axis index of closest box edge                                   <L 1324>
            var_464 = wp::copy(var_307);
            // cltype = ct  # encoded collision configuration                                 <L 1325>
            var_465 = wp::copy(var_451);
        }
        var_466 = wp::where(var_454, var_455, var_27);
        var_467 = wp::where(var_454, var_456, var_30);
        var_468 = wp::where(var_454, var_465, var_33);
        var_469 = wp::where(var_454, var_463, var_137);
        var_470 = wp::where(var_454, var_464, var_140);
        var_471 = wp::where(var_454, var_457, var_142);
        var_472 = wp::where(var_454, var_459, var_356);
        wp::assign(var_27, var_466);
        wp::assign(var_30, var_467);
        wp::assign(var_33, var_468);
        wp::assign(var_137, var_469);
        wp::assign(var_140, var_470);
        wp::assign(var_142, var_471);
        wp::assign(var_303, var_472);
        wp::assign(var_174, var_357);
        wp::assign(var_277, var_446);
        wp::assign(var_180, var_359);
        wp::assign(var_181, var_360);
        wp::assign(var_184, var_361);
        wp::assign(var_188, var_362);
        wp::assign(var_189, var_363);
        wp::assign(var_192, var_364);
        wp::assign(var_197, var_366);
        wp::assign(var_272, var_441);
        wp::assign(var_273, var_442);
        wp::assign(var_274, var_443);
        wp::assign(var_275, var_444);
        wp::assign(var_236, var_405);
        wp::assign(var_282, var_451);
        wp::assign(var_283, var_452);
        goto start_for_4;
    end_for_4:;
    // for j in range(3):                                                                     <L 1244>
    var_475 = wp::range(var_474);
    start_for_6:;
        if (iter_cmp(var_475) == 0) goto end_for_6;
        var_476 = wp::iter_next(var_475);
        // if i & (1 << j) != 0:                                                              <L 1245>
        var_478 = wp::lshift(var_477, var_476);
        var_479 = wp::bit_and(var_473, var_478);
        var_481 = (var_479 != var_480);
        if (var_481) {
            // continue                                                                       <L 1246>
            goto start_for_6;
        }
        // c2 = wp.int32(-123)                                                                <L 1248>
        var_484 = wp::int32(var_483);
        // box_pt = wp.cw_mul(                                                                <L 1251>
        // wp.vec3(                                                                           <L 1252>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1253>
        var_486 = wp::bit_and(var_473, var_485);
        var_490 = wp::where(var_486, var_487, var_489);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1254>
        var_492 = wp::bit_and(var_473, var_491);
        var_496 = wp::where(var_492, var_493, var_495);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1255>
        var_498 = wp::bit_and(var_473, var_497);
        var_502 = wp::where(var_498, var_499, var_501);
        var_503 = wp::vec_t<3, wp::float32>(var_490, var_496, var_502);
        // box_size,                                                                          <L 1257>
        var_504 = wp::cw_mul(var_503, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1259>
        wp::assign_inplace(var_504, var_476, var_505);
        // dif = box_pt - pos                                                                 <L 1262>
        var_506 = wp::sub(var_504, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1264>
        var_507 = wp::extract(var_box_size, var_476);
        var_508 = wp::neg(var_507);
        var_509 = wp::extract(var_506, var_476);
        var_510 = wp::mul(var_508, var_509);
        // v = wp.dot(halfaxis, dif)                                                          <L 1265>
        var_511 = wp::dot(var_4, var_506);
        // ma = box_size[j] * box_size[j]                                                     <L 1266>
        var_512 = wp::extract(var_box_size, var_476);
        var_513 = wp::extract(var_box_size, var_476);
        var_514 = wp::mul(var_512, var_513);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1267>
        var_515 = wp::extract(var_box_size, var_476);
        var_516 = wp::neg(var_515);
        var_517 = wp::extract(var_4, var_476);
        var_518 = wp::mul(var_516, var_517);
        // mc = capsule_half_length * capsule_half_length                                     <L 1268>
        var_519 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1269>
        var_520 = wp::mul(var_514, var_519);
        var_521 = wp::mul(var_518, var_518);
        var_522 = wp::sub(var_520, var_521);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1270>
        var_523 = wp::abs(var_522);
        var_524 = (var_523 < var_194);
        if (var_524) {
            // continue                                                                       <L 1271>
            wp::assign(var_303, var_484);
            wp::assign(var_174, var_504);
            wp::assign(var_277, var_506);
            wp::assign(var_180, var_510);
            wp::assign(var_181, var_511);
            wp::assign(var_184, var_514);
            wp::assign(var_188, var_518);
            wp::assign(var_189, var_519);
            wp::assign(var_192, var_522);
            goto start_for_6;
        }
        var_525 = wp::where(var_524, var_303, var_484);
        var_526 = wp::where(var_524, var_174, var_504);
        var_527 = wp::where(var_524, var_277, var_506);
        var_528 = wp::where(var_524, var_180, var_510);
        var_529 = wp::where(var_524, var_181, var_511);
        var_530 = wp::where(var_524, var_184, var_514);
        var_531 = wp::where(var_524, var_188, var_518);
        var_532 = wp::where(var_524, var_189, var_519);
        var_533 = wp::where(var_524, var_192, var_522);
        // idet = 1.0 / det                                                                   <L 1273>
        var_535 = wp::div(var_534, var_533);
        // x1 = wp.float32((mc * u - mb * v) * idet)                                          <L 1276>
        var_536 = wp::mul(var_532, var_528);
        var_537 = wp::mul(var_531, var_529);
        var_538 = wp::sub(var_536, var_537);
        var_539 = wp::mul(var_538, var_535);
        var_540 = wp::float32(var_539);
        // x2 = wp.float32((ma * v - mb * u) * idet)                                          <L 1277>
        var_541 = wp::mul(var_530, var_529);
        var_542 = wp::mul(var_531, var_528);
        var_543 = wp::sub(var_541, var_542);
        var_544 = wp::mul(var_543, var_535);
        var_545 = wp::float32(var_544);
        // s1 = wp.int32(1)                                                                   <L 1279>
        var_547 = wp::int32(var_546);
        // s2 = wp.int32(1)                                                                   <L 1280>
        var_549 = wp::int32(var_548);
        // if x1 > 1:                                                                         <L 1282>
        var_551 = (var_540 > var_550);
        if (var_551) {
            // x1 = 1.0                                                                       <L 1283>
            // s1 = 2                                                                         <L 1284>
            // x2 = safe_div(v - mb, mc)                                                      <L 1285>
            var_554 = wp::sub(var_529, var_531);
            var_555 = safe_div_1(var_554, var_532);
        }
        var_556 = wp::where(var_551, var_552, var_540);
        var_557 = wp::where(var_551, var_555, var_545);
        var_558 = wp::where(var_551, var_553, var_547);
        if (!var_551) {
            // elif x1 < -1:                                                                  <L 1286>
            var_561 = (var_556 < var_560);
            if (var_561) {
                // x1 = -1.0                                                                  <L 1287>
                // s1 = 0                                                                     <L 1288>
                // x2 = safe_div(v + mb, mc)                                                  <L 1289>
                var_565 = wp::add(var_529, var_531);
                var_566 = safe_div_1(var_565, var_532);
            }
            var_567 = wp::where(var_561, var_563, var_556);
            var_568 = wp::where(var_561, var_566, var_557);
            var_569 = wp::where(var_561, var_564, var_558);
        }
        var_570 = wp::where(var_551, var_556, var_567);
        var_571 = wp::where(var_551, var_557, var_568);
        var_572 = wp::where(var_551, var_558, var_569);
        // x2_over = x2 > 1.0                                                                 <L 1291>
        var_574 = (var_571 > var_573);
        // if x2_over or x2 < -1.0:                                                           <L 1292>
        var_577 = (var_571 < var_576);
        var_578 = var_574 || var_577;
        if (var_578) {
            // if x2_over:                                                                    <L 1293>
            if (var_574) {
                // x2 = 1.0                                                                   <L 1294>
                // s2 = 2                                                                     <L 1295>
                // x1 = safe_div(u - mb, ma)                                                  <L 1296>
                var_581 = wp::sub(var_528, var_531);
                var_582 = safe_div_1(var_581, var_530);
            }
            var_583 = wp::where(var_574, var_582, var_570);
            var_584 = wp::where(var_574, var_579, var_571);
            var_585 = wp::where(var_574, var_580, var_549);
            if (!var_574) {
                // x2 = -1.0                                                                  <L 1298>
                // s2 = 0                                                                     <L 1299>
                // x1 = safe_div(u + mb, ma)                                                  <L 1300>
                var_589 = wp::add(var_528, var_531);
                var_590 = safe_div_1(var_589, var_530);
            }
            var_591 = wp::where(var_574, var_583, var_590);
            var_592 = wp::where(var_574, var_584, var_587);
            var_593 = wp::where(var_574, var_585, var_588);
            // if x1 > 1:                                                                     <L 1302>
            var_595 = (var_591 > var_594);
            if (var_595) {
                // x1 = 1.0                                                                   <L 1303>
                // s1 = 2                                                                     <L 1304>
            }
            var_598 = wp::where(var_595, var_596, var_591);
            var_599 = wp::where(var_595, var_597, var_572);
            if (!var_595) {
                // elif x1 < -1:                                                              <L 1305>
                var_602 = (var_598 < var_601);
                if (var_602) {
                    // x1 = -1.0                                                              <L 1306>
                    // s1 = 0                                                                 <L 1307>
                }
                var_606 = wp::where(var_602, var_604, var_598);
                var_607 = wp::where(var_602, var_605, var_599);
            }
            var_608 = wp::where(var_595, var_598, var_606);
            var_609 = wp::where(var_595, var_599, var_607);
        }
        var_610 = wp::where(var_578, var_608, var_570);
        var_611 = wp::where(var_578, var_592, var_571);
        var_612 = wp::where(var_578, var_609, var_572);
        var_613 = wp::where(var_578, var_593, var_549);
        // dif -= halfaxis * x2                                                               <L 1309>
        var_614 = wp::mul(var_4, var_611);
        var_615 = wp::sub(var_527, var_614);
        // dif[j] += box_size[j] * x1                                                         <L 1310>
        var_616 = wp::extract(var_box_size, var_476);
        var_617 = wp::mul(var_616, var_610);
        wp::add_inplace(var_615, var_476, var_617);
        // ct = s1 * 3 + s2                                                                   <L 1313>
        var_619 = wp::mul(var_612, var_618);
        var_620 = wp::add(var_619, var_613);
        // dif_sq = wp.length_sq(dif)                                                         <L 1315>
        var_621 = wp::length_sq(var_615);
        // if dif_sq < bestdist - MJ_MINVAL:                                                  <L 1316>
        var_622 = wp::sub(var_27, var_194);
        var_623 = (var_621 < var_622);
        if (var_623) {
            // bestdist = dif_sq                                                              <L 1317>
            var_624 = wp::copy(var_621);
            // bestsegmentpos = x2                                                            <L 1318>
            var_625 = wp::copy(var_611);
            // bestboxpos = x1                                                                <L 1319>
            var_626 = wp::copy(var_610);
            // c2 = ct // 6                                                                   <L 1321>
            var_628 = wp::floordiv(var_620, var_627);
            // clcorner = i + (1 << j) * c2  # index of closest box corner                    <L 1323>
            var_630 = wp::lshift(var_629, var_476);
            var_631 = wp::mul(var_630, var_628);
            var_632 = wp::add(var_473, var_631);
            // cledge = j  # axis index of closest box edge                                   <L 1324>
            var_633 = wp::copy(var_476);
            // cltype = ct  # encoded collision configuration                                 <L 1325>
            var_634 = wp::copy(var_620);
        }
        var_635 = wp::where(var_623, var_624, var_27);
        var_636 = wp::where(var_623, var_625, var_30);
        var_637 = wp::where(var_623, var_634, var_33);
        var_638 = wp::where(var_623, var_632, var_137);
        var_639 = wp::where(var_623, var_633, var_140);
        var_640 = wp::where(var_623, var_626, var_142);
        var_641 = wp::where(var_623, var_628, var_525);
        wp::assign(var_27, var_635);
        wp::assign(var_30, var_636);
        wp::assign(var_33, var_637);
        wp::assign(var_137, var_638);
        wp::assign(var_140, var_639);
        wp::assign(var_142, var_640);
        wp::assign(var_303, var_641);
        wp::assign(var_174, var_526);
        wp::assign(var_277, var_615);
        wp::assign(var_180, var_528);
        wp::assign(var_181, var_529);
        wp::assign(var_184, var_530);
        wp::assign(var_188, var_531);
        wp::assign(var_189, var_532);
        wp::assign(var_192, var_533);
        wp::assign(var_197, var_535);
        wp::assign(var_272, var_610);
        wp::assign(var_273, var_611);
        wp::assign(var_274, var_612);
        wp::assign(var_275, var_613);
        wp::assign(var_236, var_574);
        wp::assign(var_282, var_620);
        wp::assign(var_283, var_621);
        goto start_for_6;
    end_for_6:;
    // for j in range(3):                                                                     <L 1244>
    var_644 = wp::range(var_643);
    start_for_8:;
        if (iter_cmp(var_644) == 0) goto end_for_8;
        var_645 = wp::iter_next(var_644);
        // if i & (1 << j) != 0:                                                              <L 1245>
        var_647 = wp::lshift(var_646, var_645);
        var_648 = wp::bit_and(var_642, var_647);
        var_650 = (var_648 != var_649);
        if (var_650) {
            // continue                                                                       <L 1246>
            goto start_for_8;
        }
        // c2 = wp.int32(-123)                                                                <L 1248>
        var_653 = wp::int32(var_652);
        // box_pt = wp.cw_mul(                                                                <L 1251>
        // wp.vec3(                                                                           <L 1252>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1253>
        var_655 = wp::bit_and(var_642, var_654);
        var_659 = wp::where(var_655, var_656, var_658);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1254>
        var_661 = wp::bit_and(var_642, var_660);
        var_665 = wp::where(var_661, var_662, var_664);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1255>
        var_667 = wp::bit_and(var_642, var_666);
        var_671 = wp::where(var_667, var_668, var_670);
        var_672 = wp::vec_t<3, wp::float32>(var_659, var_665, var_671);
        // box_size,                                                                          <L 1257>
        var_673 = wp::cw_mul(var_672, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1259>
        wp::assign_inplace(var_673, var_645, var_674);
        // dif = box_pt - pos                                                                 <L 1262>
        var_675 = wp::sub(var_673, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1264>
        var_676 = wp::extract(var_box_size, var_645);
        var_677 = wp::neg(var_676);
        var_678 = wp::extract(var_675, var_645);
        var_679 = wp::mul(var_677, var_678);
        // v = wp.dot(halfaxis, dif)                                                          <L 1265>
        var_680 = wp::dot(var_4, var_675);
        // ma = box_size[j] * box_size[j]                                                     <L 1266>
        var_681 = wp::extract(var_box_size, var_645);
        var_682 = wp::extract(var_box_size, var_645);
        var_683 = wp::mul(var_681, var_682);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1267>
        var_684 = wp::extract(var_box_size, var_645);
        var_685 = wp::neg(var_684);
        var_686 = wp::extract(var_4, var_645);
        var_687 = wp::mul(var_685, var_686);
        // mc = capsule_half_length * capsule_half_length                                     <L 1268>
        var_688 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1269>
        var_689 = wp::mul(var_683, var_688);
        var_690 = wp::mul(var_687, var_687);
        var_691 = wp::sub(var_689, var_690);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1270>
        var_692 = wp::abs(var_691);
        var_693 = (var_692 < var_194);
        if (var_693) {
            // continue                                                                       <L 1271>
            wp::assign(var_303, var_653);
            wp::assign(var_174, var_673);
            wp::assign(var_277, var_675);
            wp::assign(var_180, var_679);
            wp::assign(var_181, var_680);
            wp::assign(var_184, var_683);
            wp::assign(var_188, var_687);
            wp::assign(var_189, var_688);
            wp::assign(var_192, var_691);
            goto start_for_8;
        }
        var_694 = wp::where(var_693, var_303, var_653);
        var_695 = wp::where(var_693, var_174, var_673);
        var_696 = wp::where(var_693, var_277, var_675);
        var_697 = wp::where(var_693, var_180, var_679);
        var_698 = wp::where(var_693, var_181, var_680);
        var_699 = wp::where(var_693, var_184, var_683);
        var_700 = wp::where(var_693, var_188, var_687);
        var_701 = wp::where(var_693, var_189, var_688);
        var_702 = wp::where(var_693, var_192, var_691);
        // idet = 1.0 / det                                                                   <L 1273>
        var_704 = wp::div(var_703, var_702);
        // x1 = wp.float32((mc * u - mb * v) * idet)                                          <L 1276>
        var_705 = wp::mul(var_701, var_697);
        var_706 = wp::mul(var_700, var_698);
        var_707 = wp::sub(var_705, var_706);
        var_708 = wp::mul(var_707, var_704);
        var_709 = wp::float32(var_708);
        // x2 = wp.float32((ma * v - mb * u) * idet)                                          <L 1277>
        var_710 = wp::mul(var_699, var_698);
        var_711 = wp::mul(var_700, var_697);
        var_712 = wp::sub(var_710, var_711);
        var_713 = wp::mul(var_712, var_704);
        var_714 = wp::float32(var_713);
        // s1 = wp.int32(1)                                                                   <L 1279>
        var_716 = wp::int32(var_715);
        // s2 = wp.int32(1)                                                                   <L 1280>
        var_718 = wp::int32(var_717);
        // if x1 > 1:                                                                         <L 1282>
        var_720 = (var_709 > var_719);
        if (var_720) {
            // x1 = 1.0                                                                       <L 1283>
            // s1 = 2                                                                         <L 1284>
            // x2 = safe_div(v - mb, mc)                                                      <L 1285>
            var_723 = wp::sub(var_698, var_700);
            var_724 = safe_div_1(var_723, var_701);
        }
        var_725 = wp::where(var_720, var_721, var_709);
        var_726 = wp::where(var_720, var_724, var_714);
        var_727 = wp::where(var_720, var_722, var_716);
        if (!var_720) {
            // elif x1 < -1:                                                                  <L 1286>
            var_730 = (var_725 < var_729);
            if (var_730) {
                // x1 = -1.0                                                                  <L 1287>
                // s1 = 0                                                                     <L 1288>
                // x2 = safe_div(v + mb, mc)                                                  <L 1289>
                var_734 = wp::add(var_698, var_700);
                var_735 = safe_div_1(var_734, var_701);
            }
            var_736 = wp::where(var_730, var_732, var_725);
            var_737 = wp::where(var_730, var_735, var_726);
            var_738 = wp::where(var_730, var_733, var_727);
        }
        var_739 = wp::where(var_720, var_725, var_736);
        var_740 = wp::where(var_720, var_726, var_737);
        var_741 = wp::where(var_720, var_727, var_738);
        // x2_over = x2 > 1.0                                                                 <L 1291>
        var_743 = (var_740 > var_742);
        // if x2_over or x2 < -1.0:                                                           <L 1292>
        var_746 = (var_740 < var_745);
        var_747 = var_743 || var_746;
        if (var_747) {
            // if x2_over:                                                                    <L 1293>
            if (var_743) {
                // x2 = 1.0                                                                   <L 1294>
                // s2 = 2                                                                     <L 1295>
                // x1 = safe_div(u - mb, ma)                                                  <L 1296>
                var_750 = wp::sub(var_697, var_700);
                var_751 = safe_div_1(var_750, var_699);
            }
            var_752 = wp::where(var_743, var_751, var_739);
            var_753 = wp::where(var_743, var_748, var_740);
            var_754 = wp::where(var_743, var_749, var_718);
            if (!var_743) {
                // x2 = -1.0                                                                  <L 1298>
                // s2 = 0                                                                     <L 1299>
                // x1 = safe_div(u + mb, ma)                                                  <L 1300>
                var_758 = wp::add(var_697, var_700);
                var_759 = safe_div_1(var_758, var_699);
            }
            var_760 = wp::where(var_743, var_752, var_759);
            var_761 = wp::where(var_743, var_753, var_756);
            var_762 = wp::where(var_743, var_754, var_757);
            // if x1 > 1:                                                                     <L 1302>
            var_764 = (var_760 > var_763);
            if (var_764) {
                // x1 = 1.0                                                                   <L 1303>
                // s1 = 2                                                                     <L 1304>
            }
            var_767 = wp::where(var_764, var_765, var_760);
            var_768 = wp::where(var_764, var_766, var_741);
            if (!var_764) {
                // elif x1 < -1:                                                              <L 1305>
                var_771 = (var_767 < var_770);
                if (var_771) {
                    // x1 = -1.0                                                              <L 1306>
                    // s1 = 0                                                                 <L 1307>
                }
                var_775 = wp::where(var_771, var_773, var_767);
                var_776 = wp::where(var_771, var_774, var_768);
            }
            var_777 = wp::where(var_764, var_767, var_775);
            var_778 = wp::where(var_764, var_768, var_776);
        }
        var_779 = wp::where(var_747, var_777, var_739);
        var_780 = wp::where(var_747, var_761, var_740);
        var_781 = wp::where(var_747, var_778, var_741);
        var_782 = wp::where(var_747, var_762, var_718);
        // dif -= halfaxis * x2                                                               <L 1309>
        var_783 = wp::mul(var_4, var_780);
        var_784 = wp::sub(var_696, var_783);
        // dif[j] += box_size[j] * x1                                                         <L 1310>
        var_785 = wp::extract(var_box_size, var_645);
        var_786 = wp::mul(var_785, var_779);
        wp::add_inplace(var_784, var_645, var_786);
        // ct = s1 * 3 + s2                                                                   <L 1313>
        var_788 = wp::mul(var_781, var_787);
        var_789 = wp::add(var_788, var_782);
        // dif_sq = wp.length_sq(dif)                                                         <L 1315>
        var_790 = wp::length_sq(var_784);
        // if dif_sq < bestdist - MJ_MINVAL:                                                  <L 1316>
        var_791 = wp::sub(var_27, var_194);
        var_792 = (var_790 < var_791);
        if (var_792) {
            // bestdist = dif_sq                                                              <L 1317>
            var_793 = wp::copy(var_790);
            // bestsegmentpos = x2                                                            <L 1318>
            var_794 = wp::copy(var_780);
            // bestboxpos = x1                                                                <L 1319>
            var_795 = wp::copy(var_779);
            // c2 = ct // 6                                                                   <L 1321>
            var_797 = wp::floordiv(var_789, var_796);
            // clcorner = i + (1 << j) * c2  # index of closest box corner                    <L 1323>
            var_799 = wp::lshift(var_798, var_645);
            var_800 = wp::mul(var_799, var_797);
            var_801 = wp::add(var_642, var_800);
            // cledge = j  # axis index of closest box edge                                   <L 1324>
            var_802 = wp::copy(var_645);
            // cltype = ct  # encoded collision configuration                                 <L 1325>
            var_803 = wp::copy(var_789);
        }
        var_804 = wp::where(var_792, var_793, var_27);
        var_805 = wp::where(var_792, var_794, var_30);
        var_806 = wp::where(var_792, var_803, var_33);
        var_807 = wp::where(var_792, var_801, var_137);
        var_808 = wp::where(var_792, var_802, var_140);
        var_809 = wp::where(var_792, var_795, var_142);
        var_810 = wp::where(var_792, var_797, var_694);
        wp::assign(var_27, var_804);
        wp::assign(var_30, var_805);
        wp::assign(var_33, var_806);
        wp::assign(var_137, var_807);
        wp::assign(var_140, var_808);
        wp::assign(var_142, var_809);
        wp::assign(var_303, var_810);
        wp::assign(var_174, var_695);
        wp::assign(var_277, var_784);
        wp::assign(var_180, var_697);
        wp::assign(var_181, var_698);
        wp::assign(var_184, var_699);
        wp::assign(var_188, var_700);
        wp::assign(var_189, var_701);
        wp::assign(var_192, var_702);
        wp::assign(var_197, var_704);
        wp::assign(var_272, var_779);
        wp::assign(var_273, var_780);
        wp::assign(var_274, var_781);
        wp::assign(var_275, var_782);
        wp::assign(var_236, var_743);
        wp::assign(var_282, var_789);
        wp::assign(var_283, var_790);
        goto start_for_8;
    end_for_8:;
    // for j in range(3):                                                                     <L 1244>
    var_813 = wp::range(var_812);
    start_for_10:;
        if (iter_cmp(var_813) == 0) goto end_for_10;
        var_814 = wp::iter_next(var_813);
        // if i & (1 << j) != 0:                                                              <L 1245>
        var_816 = wp::lshift(var_815, var_814);
        var_817 = wp::bit_and(var_811, var_816);
        var_819 = (var_817 != var_818);
        if (var_819) {
            // continue                                                                       <L 1246>
            goto start_for_10;
        }
        // c2 = wp.int32(-123)                                                                <L 1248>
        var_822 = wp::int32(var_821);
        // box_pt = wp.cw_mul(                                                                <L 1251>
        // wp.vec3(                                                                           <L 1252>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1253>
        var_824 = wp::bit_and(var_811, var_823);
        var_828 = wp::where(var_824, var_825, var_827);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1254>
        var_830 = wp::bit_and(var_811, var_829);
        var_834 = wp::where(var_830, var_831, var_833);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1255>
        var_836 = wp::bit_and(var_811, var_835);
        var_840 = wp::where(var_836, var_837, var_839);
        var_841 = wp::vec_t<3, wp::float32>(var_828, var_834, var_840);
        // box_size,                                                                          <L 1257>
        var_842 = wp::cw_mul(var_841, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1259>
        wp::assign_inplace(var_842, var_814, var_843);
        // dif = box_pt - pos                                                                 <L 1262>
        var_844 = wp::sub(var_842, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1264>
        var_845 = wp::extract(var_box_size, var_814);
        var_846 = wp::neg(var_845);
        var_847 = wp::extract(var_844, var_814);
        var_848 = wp::mul(var_846, var_847);
        // v = wp.dot(halfaxis, dif)                                                          <L 1265>
        var_849 = wp::dot(var_4, var_844);
        // ma = box_size[j] * box_size[j]                                                     <L 1266>
        var_850 = wp::extract(var_box_size, var_814);
        var_851 = wp::extract(var_box_size, var_814);
        var_852 = wp::mul(var_850, var_851);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1267>
        var_853 = wp::extract(var_box_size, var_814);
        var_854 = wp::neg(var_853);
        var_855 = wp::extract(var_4, var_814);
        var_856 = wp::mul(var_854, var_855);
        // mc = capsule_half_length * capsule_half_length                                     <L 1268>
        var_857 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1269>
        var_858 = wp::mul(var_852, var_857);
        var_859 = wp::mul(var_856, var_856);
        var_860 = wp::sub(var_858, var_859);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1270>
        var_861 = wp::abs(var_860);
        var_862 = (var_861 < var_194);
        if (var_862) {
            // continue                                                                       <L 1271>
            wp::assign(var_303, var_822);
            wp::assign(var_174, var_842);
            wp::assign(var_277, var_844);
            wp::assign(var_180, var_848);
            wp::assign(var_181, var_849);
            wp::assign(var_184, var_852);
            wp::assign(var_188, var_856);
            wp::assign(var_189, var_857);
            wp::assign(var_192, var_860);
            goto start_for_10;
        }
        var_863 = wp::where(var_862, var_303, var_822);
        var_864 = wp::where(var_862, var_174, var_842);
        var_865 = wp::where(var_862, var_277, var_844);
        var_866 = wp::where(var_862, var_180, var_848);
        var_867 = wp::where(var_862, var_181, var_849);
        var_868 = wp::where(var_862, var_184, var_852);
        var_869 = wp::where(var_862, var_188, var_856);
        var_870 = wp::where(var_862, var_189, var_857);
        var_871 = wp::where(var_862, var_192, var_860);
        // idet = 1.0 / det                                                                   <L 1273>
        var_873 = wp::div(var_872, var_871);
        // x1 = wp.float32((mc * u - mb * v) * idet)                                          <L 1276>
        var_874 = wp::mul(var_870, var_866);
        var_875 = wp::mul(var_869, var_867);
        var_876 = wp::sub(var_874, var_875);
        var_877 = wp::mul(var_876, var_873);
        var_878 = wp::float32(var_877);
        // x2 = wp.float32((ma * v - mb * u) * idet)                                          <L 1277>
        var_879 = wp::mul(var_868, var_867);
        var_880 = wp::mul(var_869, var_866);
        var_881 = wp::sub(var_879, var_880);
        var_882 = wp::mul(var_881, var_873);
        var_883 = wp::float32(var_882);
        // s1 = wp.int32(1)                                                                   <L 1279>
        var_885 = wp::int32(var_884);
        // s2 = wp.int32(1)                                                                   <L 1280>
        var_887 = wp::int32(var_886);
        // if x1 > 1:                                                                         <L 1282>
        var_889 = (var_878 > var_888);
        if (var_889) {
            // x1 = 1.0                                                                       <L 1283>
            // s1 = 2                                                                         <L 1284>
            // x2 = safe_div(v - mb, mc)                                                      <L 1285>
            var_892 = wp::sub(var_867, var_869);
            var_893 = safe_div_1(var_892, var_870);
        }
        var_894 = wp::where(var_889, var_890, var_878);
        var_895 = wp::where(var_889, var_893, var_883);
        var_896 = wp::where(var_889, var_891, var_885);
        if (!var_889) {
            // elif x1 < -1:                                                                  <L 1286>
            var_899 = (var_894 < var_898);
            if (var_899) {
                // x1 = -1.0                                                                  <L 1287>
                // s1 = 0                                                                     <L 1288>
                // x2 = safe_div(v + mb, mc)                                                  <L 1289>
                var_903 = wp::add(var_867, var_869);
                var_904 = safe_div_1(var_903, var_870);
            }
            var_905 = wp::where(var_899, var_901, var_894);
            var_906 = wp::where(var_899, var_904, var_895);
            var_907 = wp::where(var_899, var_902, var_896);
        }
        var_908 = wp::where(var_889, var_894, var_905);
        var_909 = wp::where(var_889, var_895, var_906);
        var_910 = wp::where(var_889, var_896, var_907);
        // x2_over = x2 > 1.0                                                                 <L 1291>
        var_912 = (var_909 > var_911);
        // if x2_over or x2 < -1.0:                                                           <L 1292>
        var_915 = (var_909 < var_914);
        var_916 = var_912 || var_915;
        if (var_916) {
            // if x2_over:                                                                    <L 1293>
            if (var_912) {
                // x2 = 1.0                                                                   <L 1294>
                // s2 = 2                                                                     <L 1295>
                // x1 = safe_div(u - mb, ma)                                                  <L 1296>
                var_919 = wp::sub(var_866, var_869);
                var_920 = safe_div_1(var_919, var_868);
            }
            var_921 = wp::where(var_912, var_920, var_908);
            var_922 = wp::where(var_912, var_917, var_909);
            var_923 = wp::where(var_912, var_918, var_887);
            if (!var_912) {
                // x2 = -1.0                                                                  <L 1298>
                // s2 = 0                                                                     <L 1299>
                // x1 = safe_div(u + mb, ma)                                                  <L 1300>
                var_927 = wp::add(var_866, var_869);
                var_928 = safe_div_1(var_927, var_868);
            }
            var_929 = wp::where(var_912, var_921, var_928);
            var_930 = wp::where(var_912, var_922, var_925);
            var_931 = wp::where(var_912, var_923, var_926);
            // if x1 > 1:                                                                     <L 1302>
            var_933 = (var_929 > var_932);
            if (var_933) {
                // x1 = 1.0                                                                   <L 1303>
                // s1 = 2                                                                     <L 1304>
            }
            var_936 = wp::where(var_933, var_934, var_929);
            var_937 = wp::where(var_933, var_935, var_910);
            if (!var_933) {
                // elif x1 < -1:                                                              <L 1305>
                var_940 = (var_936 < var_939);
                if (var_940) {
                    // x1 = -1.0                                                              <L 1306>
                    // s1 = 0                                                                 <L 1307>
                }
                var_944 = wp::where(var_940, var_942, var_936);
                var_945 = wp::where(var_940, var_943, var_937);
            }
            var_946 = wp::where(var_933, var_936, var_944);
            var_947 = wp::where(var_933, var_937, var_945);
        }
        var_948 = wp::where(var_916, var_946, var_908);
        var_949 = wp::where(var_916, var_930, var_909);
        var_950 = wp::where(var_916, var_947, var_910);
        var_951 = wp::where(var_916, var_931, var_887);
        // dif -= halfaxis * x2                                                               <L 1309>
        var_952 = wp::mul(var_4, var_949);
        var_953 = wp::sub(var_865, var_952);
        // dif[j] += box_size[j] * x1                                                         <L 1310>
        var_954 = wp::extract(var_box_size, var_814);
        var_955 = wp::mul(var_954, var_948);
        wp::add_inplace(var_953, var_814, var_955);
        // ct = s1 * 3 + s2                                                                   <L 1313>
        var_957 = wp::mul(var_950, var_956);
        var_958 = wp::add(var_957, var_951);
        // dif_sq = wp.length_sq(dif)                                                         <L 1315>
        var_959 = wp::length_sq(var_953);
        // if dif_sq < bestdist - MJ_MINVAL:                                                  <L 1316>
        var_960 = wp::sub(var_27, var_194);
        var_961 = (var_959 < var_960);
        if (var_961) {
            // bestdist = dif_sq                                                              <L 1317>
            var_962 = wp::copy(var_959);
            // bestsegmentpos = x2                                                            <L 1318>
            var_963 = wp::copy(var_949);
            // bestboxpos = x1                                                                <L 1319>
            var_964 = wp::copy(var_948);
            // c2 = ct // 6                                                                   <L 1321>
            var_966 = wp::floordiv(var_958, var_965);
            // clcorner = i + (1 << j) * c2  # index of closest box corner                    <L 1323>
            var_968 = wp::lshift(var_967, var_814);
            var_969 = wp::mul(var_968, var_966);
            var_970 = wp::add(var_811, var_969);
            // cledge = j  # axis index of closest box edge                                   <L 1324>
            var_971 = wp::copy(var_814);
            // cltype = ct  # encoded collision configuration                                 <L 1325>
            var_972 = wp::copy(var_958);
        }
        var_973 = wp::where(var_961, var_962, var_27);
        var_974 = wp::where(var_961, var_963, var_30);
        var_975 = wp::where(var_961, var_972, var_33);
        var_976 = wp::where(var_961, var_970, var_137);
        var_977 = wp::where(var_961, var_971, var_140);
        var_978 = wp::where(var_961, var_964, var_142);
        var_979 = wp::where(var_961, var_966, var_863);
        wp::assign(var_27, var_973);
        wp::assign(var_30, var_974);
        wp::assign(var_33, var_975);
        wp::assign(var_137, var_976);
        wp::assign(var_140, var_977);
        wp::assign(var_142, var_978);
        wp::assign(var_303, var_979);
        wp::assign(var_174, var_864);
        wp::assign(var_277, var_953);
        wp::assign(var_180, var_866);
        wp::assign(var_181, var_867);
        wp::assign(var_184, var_868);
        wp::assign(var_188, var_869);
        wp::assign(var_189, var_870);
        wp::assign(var_192, var_871);
        wp::assign(var_197, var_873);
        wp::assign(var_272, var_948);
        wp::assign(var_273, var_949);
        wp::assign(var_274, var_950);
        wp::assign(var_275, var_951);
        wp::assign(var_236, var_912);
        wp::assign(var_282, var_958);
        wp::assign(var_283, var_959);
        goto start_for_10;
    end_for_10:;
    // for j in range(3):                                                                     <L 1244>
    var_982 = wp::range(var_981);
    start_for_12:;
        if (iter_cmp(var_982) == 0) goto end_for_12;
        var_983 = wp::iter_next(var_982);
        // if i & (1 << j) != 0:                                                              <L 1245>
        var_985 = wp::lshift(var_984, var_983);
        var_986 = wp::bit_and(var_980, var_985);
        var_988 = (var_986 != var_987);
        if (var_988) {
            // continue                                                                       <L 1246>
            goto start_for_12;
        }
        // c2 = wp.int32(-123)                                                                <L 1248>
        var_991 = wp::int32(var_990);
        // box_pt = wp.cw_mul(                                                                <L 1251>
        // wp.vec3(                                                                           <L 1252>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1253>
        var_993 = wp::bit_and(var_980, var_992);
        var_997 = wp::where(var_993, var_994, var_996);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1254>
        var_999 = wp::bit_and(var_980, var_998);
        var_1003 = wp::where(var_999, var_1000, var_1002);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1255>
        var_1005 = wp::bit_and(var_980, var_1004);
        var_1009 = wp::where(var_1005, var_1006, var_1008);
        var_1010 = wp::vec_t<3, wp::float32>(var_997, var_1003, var_1009);
        // box_size,                                                                          <L 1257>
        var_1011 = wp::cw_mul(var_1010, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1259>
        wp::assign_inplace(var_1011, var_983, var_1012);
        // dif = box_pt - pos                                                                 <L 1262>
        var_1013 = wp::sub(var_1011, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1264>
        var_1014 = wp::extract(var_box_size, var_983);
        var_1015 = wp::neg(var_1014);
        var_1016 = wp::extract(var_1013, var_983);
        var_1017 = wp::mul(var_1015, var_1016);
        // v = wp.dot(halfaxis, dif)                                                          <L 1265>
        var_1018 = wp::dot(var_4, var_1013);
        // ma = box_size[j] * box_size[j]                                                     <L 1266>
        var_1019 = wp::extract(var_box_size, var_983);
        var_1020 = wp::extract(var_box_size, var_983);
        var_1021 = wp::mul(var_1019, var_1020);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1267>
        var_1022 = wp::extract(var_box_size, var_983);
        var_1023 = wp::neg(var_1022);
        var_1024 = wp::extract(var_4, var_983);
        var_1025 = wp::mul(var_1023, var_1024);
        // mc = capsule_half_length * capsule_half_length                                     <L 1268>
        var_1026 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1269>
        var_1027 = wp::mul(var_1021, var_1026);
        var_1028 = wp::mul(var_1025, var_1025);
        var_1029 = wp::sub(var_1027, var_1028);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1270>
        var_1030 = wp::abs(var_1029);
        var_1031 = (var_1030 < var_194);
        if (var_1031) {
            // continue                                                                       <L 1271>
            wp::assign(var_303, var_991);
            wp::assign(var_174, var_1011);
            wp::assign(var_277, var_1013);
            wp::assign(var_180, var_1017);
            wp::assign(var_181, var_1018);
            wp::assign(var_184, var_1021);
            wp::assign(var_188, var_1025);
            wp::assign(var_189, var_1026);
            wp::assign(var_192, var_1029);
            goto start_for_12;
        }
        var_1032 = wp::where(var_1031, var_303, var_991);
        var_1033 = wp::where(var_1031, var_174, var_1011);
        var_1034 = wp::where(var_1031, var_277, var_1013);
        var_1035 = wp::where(var_1031, var_180, var_1017);
        var_1036 = wp::where(var_1031, var_181, var_1018);
        var_1037 = wp::where(var_1031, var_184, var_1021);
        var_1038 = wp::where(var_1031, var_188, var_1025);
        var_1039 = wp::where(var_1031, var_189, var_1026);
        var_1040 = wp::where(var_1031, var_192, var_1029);
        // idet = 1.0 / det                                                                   <L 1273>
        var_1042 = wp::div(var_1041, var_1040);
        // x1 = wp.float32((mc * u - mb * v) * idet)                                          <L 1276>
        var_1043 = wp::mul(var_1039, var_1035);
        var_1044 = wp::mul(var_1038, var_1036);
        var_1045 = wp::sub(var_1043, var_1044);
        var_1046 = wp::mul(var_1045, var_1042);
        var_1047 = wp::float32(var_1046);
        // x2 = wp.float32((ma * v - mb * u) * idet)                                          <L 1277>
        var_1048 = wp::mul(var_1037, var_1036);
        var_1049 = wp::mul(var_1038, var_1035);
        var_1050 = wp::sub(var_1048, var_1049);
        var_1051 = wp::mul(var_1050, var_1042);
        var_1052 = wp::float32(var_1051);
        // s1 = wp.int32(1)                                                                   <L 1279>
        var_1054 = wp::int32(var_1053);
        // s2 = wp.int32(1)                                                                   <L 1280>
        var_1056 = wp::int32(var_1055);
        // if x1 > 1:                                                                         <L 1282>
        var_1058 = (var_1047 > var_1057);
        if (var_1058) {
            // x1 = 1.0                                                                       <L 1283>
            // s1 = 2                                                                         <L 1284>
            // x2 = safe_div(v - mb, mc)                                                      <L 1285>
            var_1061 = wp::sub(var_1036, var_1038);
            var_1062 = safe_div_1(var_1061, var_1039);
        }
        var_1063 = wp::where(var_1058, var_1059, var_1047);
        var_1064 = wp::where(var_1058, var_1062, var_1052);
        var_1065 = wp::where(var_1058, var_1060, var_1054);
        if (!var_1058) {
            // elif x1 < -1:                                                                  <L 1286>
            var_1068 = (var_1063 < var_1067);
            if (var_1068) {
                // x1 = -1.0                                                                  <L 1287>
                // s1 = 0                                                                     <L 1288>
                // x2 = safe_div(v + mb, mc)                                                  <L 1289>
                var_1072 = wp::add(var_1036, var_1038);
                var_1073 = safe_div_1(var_1072, var_1039);
            }
            var_1074 = wp::where(var_1068, var_1070, var_1063);
            var_1075 = wp::where(var_1068, var_1073, var_1064);
            var_1076 = wp::where(var_1068, var_1071, var_1065);
        }
        var_1077 = wp::where(var_1058, var_1063, var_1074);
        var_1078 = wp::where(var_1058, var_1064, var_1075);
        var_1079 = wp::where(var_1058, var_1065, var_1076);
        // x2_over = x2 > 1.0                                                                 <L 1291>
        var_1081 = (var_1078 > var_1080);
        // if x2_over or x2 < -1.0:                                                           <L 1292>
        var_1084 = (var_1078 < var_1083);
        var_1085 = var_1081 || var_1084;
        if (var_1085) {
            // if x2_over:                                                                    <L 1293>
            if (var_1081) {
                // x2 = 1.0                                                                   <L 1294>
                // s2 = 2                                                                     <L 1295>
                // x1 = safe_div(u - mb, ma)                                                  <L 1296>
                var_1088 = wp::sub(var_1035, var_1038);
                var_1089 = safe_div_1(var_1088, var_1037);
            }
            var_1090 = wp::where(var_1081, var_1089, var_1077);
            var_1091 = wp::where(var_1081, var_1086, var_1078);
            var_1092 = wp::where(var_1081, var_1087, var_1056);
            if (!var_1081) {
                // x2 = -1.0                                                                  <L 1298>
                // s2 = 0                                                                     <L 1299>
                // x1 = safe_div(u + mb, ma)                                                  <L 1300>
                var_1096 = wp::add(var_1035, var_1038);
                var_1097 = safe_div_1(var_1096, var_1037);
            }
            var_1098 = wp::where(var_1081, var_1090, var_1097);
            var_1099 = wp::where(var_1081, var_1091, var_1094);
            var_1100 = wp::where(var_1081, var_1092, var_1095);
            // if x1 > 1:                                                                     <L 1302>
            var_1102 = (var_1098 > var_1101);
            if (var_1102) {
                // x1 = 1.0                                                                   <L 1303>
                // s1 = 2                                                                     <L 1304>
            }
            var_1105 = wp::where(var_1102, var_1103, var_1098);
            var_1106 = wp::where(var_1102, var_1104, var_1079);
            if (!var_1102) {
                // elif x1 < -1:                                                              <L 1305>
                var_1109 = (var_1105 < var_1108);
                if (var_1109) {
                    // x1 = -1.0                                                              <L 1306>
                    // s1 = 0                                                                 <L 1307>
                }
                var_1113 = wp::where(var_1109, var_1111, var_1105);
                var_1114 = wp::where(var_1109, var_1112, var_1106);
            }
            var_1115 = wp::where(var_1102, var_1105, var_1113);
            var_1116 = wp::where(var_1102, var_1106, var_1114);
        }
        var_1117 = wp::where(var_1085, var_1115, var_1077);
        var_1118 = wp::where(var_1085, var_1099, var_1078);
        var_1119 = wp::where(var_1085, var_1116, var_1079);
        var_1120 = wp::where(var_1085, var_1100, var_1056);
        // dif -= halfaxis * x2                                                               <L 1309>
        var_1121 = wp::mul(var_4, var_1118);
        var_1122 = wp::sub(var_1034, var_1121);
        // dif[j] += box_size[j] * x1                                                         <L 1310>
        var_1123 = wp::extract(var_box_size, var_983);
        var_1124 = wp::mul(var_1123, var_1117);
        wp::add_inplace(var_1122, var_983, var_1124);
        // ct = s1 * 3 + s2                                                                   <L 1313>
        var_1126 = wp::mul(var_1119, var_1125);
        var_1127 = wp::add(var_1126, var_1120);
        // dif_sq = wp.length_sq(dif)                                                         <L 1315>
        var_1128 = wp::length_sq(var_1122);
        // if dif_sq < bestdist - MJ_MINVAL:                                                  <L 1316>
        var_1129 = wp::sub(var_27, var_194);
        var_1130 = (var_1128 < var_1129);
        if (var_1130) {
            // bestdist = dif_sq                                                              <L 1317>
            var_1131 = wp::copy(var_1128);
            // bestsegmentpos = x2                                                            <L 1318>
            var_1132 = wp::copy(var_1118);
            // bestboxpos = x1                                                                <L 1319>
            var_1133 = wp::copy(var_1117);
            // c2 = ct // 6                                                                   <L 1321>
            var_1135 = wp::floordiv(var_1127, var_1134);
            // clcorner = i + (1 << j) * c2  # index of closest box corner                    <L 1323>
            var_1137 = wp::lshift(var_1136, var_983);
            var_1138 = wp::mul(var_1137, var_1135);
            var_1139 = wp::add(var_980, var_1138);
            // cledge = j  # axis index of closest box edge                                   <L 1324>
            var_1140 = wp::copy(var_983);
            // cltype = ct  # encoded collision configuration                                 <L 1325>
            var_1141 = wp::copy(var_1127);
        }
        var_1142 = wp::where(var_1130, var_1131, var_27);
        var_1143 = wp::where(var_1130, var_1132, var_30);
        var_1144 = wp::where(var_1130, var_1141, var_33);
        var_1145 = wp::where(var_1130, var_1139, var_137);
        var_1146 = wp::where(var_1130, var_1140, var_140);
        var_1147 = wp::where(var_1130, var_1133, var_142);
        var_1148 = wp::where(var_1130, var_1135, var_1032);
        wp::assign(var_27, var_1142);
        wp::assign(var_30, var_1143);
        wp::assign(var_33, var_1144);
        wp::assign(var_137, var_1145);
        wp::assign(var_140, var_1146);
        wp::assign(var_142, var_1147);
        wp::assign(var_303, var_1148);
        wp::assign(var_174, var_1033);
        wp::assign(var_277, var_1122);
        wp::assign(var_180, var_1035);
        wp::assign(var_181, var_1036);
        wp::assign(var_184, var_1037);
        wp::assign(var_188, var_1038);
        wp::assign(var_189, var_1039);
        wp::assign(var_192, var_1040);
        wp::assign(var_197, var_1042);
        wp::assign(var_272, var_1117);
        wp::assign(var_273, var_1118);
        wp::assign(var_274, var_1119);
        wp::assign(var_275, var_1120);
        wp::assign(var_236, var_1081);
        wp::assign(var_282, var_1127);
        wp::assign(var_283, var_1128);
        goto start_for_12;
    end_for_12:;
    // for j in range(3):                                                                     <L 1244>
    var_1151 = wp::range(var_1150);
    start_for_14:;
        if (iter_cmp(var_1151) == 0) goto end_for_14;
        var_1152 = wp::iter_next(var_1151);
        // if i & (1 << j) != 0:                                                              <L 1245>
        var_1154 = wp::lshift(var_1153, var_1152);
        var_1155 = wp::bit_and(var_1149, var_1154);
        var_1157 = (var_1155 != var_1156);
        if (var_1157) {
            // continue                                                                       <L 1246>
            goto start_for_14;
        }
        // c2 = wp.int32(-123)                                                                <L 1248>
        var_1160 = wp::int32(var_1159);
        // box_pt = wp.cw_mul(                                                                <L 1251>
        // wp.vec3(                                                                           <L 1252>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1253>
        var_1162 = wp::bit_and(var_1149, var_1161);
        var_1166 = wp::where(var_1162, var_1163, var_1165);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1254>
        var_1168 = wp::bit_and(var_1149, var_1167);
        var_1172 = wp::where(var_1168, var_1169, var_1171);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1255>
        var_1174 = wp::bit_and(var_1149, var_1173);
        var_1178 = wp::where(var_1174, var_1175, var_1177);
        var_1179 = wp::vec_t<3, wp::float32>(var_1166, var_1172, var_1178);
        // box_size,                                                                          <L 1257>
        var_1180 = wp::cw_mul(var_1179, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1259>
        wp::assign_inplace(var_1180, var_1152, var_1181);
        // dif = box_pt - pos                                                                 <L 1262>
        var_1182 = wp::sub(var_1180, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1264>
        var_1183 = wp::extract(var_box_size, var_1152);
        var_1184 = wp::neg(var_1183);
        var_1185 = wp::extract(var_1182, var_1152);
        var_1186 = wp::mul(var_1184, var_1185);
        // v = wp.dot(halfaxis, dif)                                                          <L 1265>
        var_1187 = wp::dot(var_4, var_1182);
        // ma = box_size[j] * box_size[j]                                                     <L 1266>
        var_1188 = wp::extract(var_box_size, var_1152);
        var_1189 = wp::extract(var_box_size, var_1152);
        var_1190 = wp::mul(var_1188, var_1189);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1267>
        var_1191 = wp::extract(var_box_size, var_1152);
        var_1192 = wp::neg(var_1191);
        var_1193 = wp::extract(var_4, var_1152);
        var_1194 = wp::mul(var_1192, var_1193);
        // mc = capsule_half_length * capsule_half_length                                     <L 1268>
        var_1195 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1269>
        var_1196 = wp::mul(var_1190, var_1195);
        var_1197 = wp::mul(var_1194, var_1194);
        var_1198 = wp::sub(var_1196, var_1197);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1270>
        var_1199 = wp::abs(var_1198);
        var_1200 = (var_1199 < var_194);
        if (var_1200) {
            // continue                                                                       <L 1271>
            wp::assign(var_303, var_1160);
            wp::assign(var_174, var_1180);
            wp::assign(var_277, var_1182);
            wp::assign(var_180, var_1186);
            wp::assign(var_181, var_1187);
            wp::assign(var_184, var_1190);
            wp::assign(var_188, var_1194);
            wp::assign(var_189, var_1195);
            wp::assign(var_192, var_1198);
            goto start_for_14;
        }
        var_1201 = wp::where(var_1200, var_303, var_1160);
        var_1202 = wp::where(var_1200, var_174, var_1180);
        var_1203 = wp::where(var_1200, var_277, var_1182);
        var_1204 = wp::where(var_1200, var_180, var_1186);
        var_1205 = wp::where(var_1200, var_181, var_1187);
        var_1206 = wp::where(var_1200, var_184, var_1190);
        var_1207 = wp::where(var_1200, var_188, var_1194);
        var_1208 = wp::where(var_1200, var_189, var_1195);
        var_1209 = wp::where(var_1200, var_192, var_1198);
        // idet = 1.0 / det                                                                   <L 1273>
        var_1211 = wp::div(var_1210, var_1209);
        // x1 = wp.float32((mc * u - mb * v) * idet)                                          <L 1276>
        var_1212 = wp::mul(var_1208, var_1204);
        var_1213 = wp::mul(var_1207, var_1205);
        var_1214 = wp::sub(var_1212, var_1213);
        var_1215 = wp::mul(var_1214, var_1211);
        var_1216 = wp::float32(var_1215);
        // x2 = wp.float32((ma * v - mb * u) * idet)                                          <L 1277>
        var_1217 = wp::mul(var_1206, var_1205);
        var_1218 = wp::mul(var_1207, var_1204);
        var_1219 = wp::sub(var_1217, var_1218);
        var_1220 = wp::mul(var_1219, var_1211);
        var_1221 = wp::float32(var_1220);
        // s1 = wp.int32(1)                                                                   <L 1279>
        var_1223 = wp::int32(var_1222);
        // s2 = wp.int32(1)                                                                   <L 1280>
        var_1225 = wp::int32(var_1224);
        // if x1 > 1:                                                                         <L 1282>
        var_1227 = (var_1216 > var_1226);
        if (var_1227) {
            // x1 = 1.0                                                                       <L 1283>
            // s1 = 2                                                                         <L 1284>
            // x2 = safe_div(v - mb, mc)                                                      <L 1285>
            var_1230 = wp::sub(var_1205, var_1207);
            var_1231 = safe_div_1(var_1230, var_1208);
        }
        var_1232 = wp::where(var_1227, var_1228, var_1216);
        var_1233 = wp::where(var_1227, var_1231, var_1221);
        var_1234 = wp::where(var_1227, var_1229, var_1223);
        if (!var_1227) {
            // elif x1 < -1:                                                                  <L 1286>
            var_1237 = (var_1232 < var_1236);
            if (var_1237) {
                // x1 = -1.0                                                                  <L 1287>
                // s1 = 0                                                                     <L 1288>
                // x2 = safe_div(v + mb, mc)                                                  <L 1289>
                var_1241 = wp::add(var_1205, var_1207);
                var_1242 = safe_div_1(var_1241, var_1208);
            }
            var_1243 = wp::where(var_1237, var_1239, var_1232);
            var_1244 = wp::where(var_1237, var_1242, var_1233);
            var_1245 = wp::where(var_1237, var_1240, var_1234);
        }
        var_1246 = wp::where(var_1227, var_1232, var_1243);
        var_1247 = wp::where(var_1227, var_1233, var_1244);
        var_1248 = wp::where(var_1227, var_1234, var_1245);
        // x2_over = x2 > 1.0                                                                 <L 1291>
        var_1250 = (var_1247 > var_1249);
        // if x2_over or x2 < -1.0:                                                           <L 1292>
        var_1253 = (var_1247 < var_1252);
        var_1254 = var_1250 || var_1253;
        if (var_1254) {
            // if x2_over:                                                                    <L 1293>
            if (var_1250) {
                // x2 = 1.0                                                                   <L 1294>
                // s2 = 2                                                                     <L 1295>
                // x1 = safe_div(u - mb, ma)                                                  <L 1296>
                var_1257 = wp::sub(var_1204, var_1207);
                var_1258 = safe_div_1(var_1257, var_1206);
            }
            var_1259 = wp::where(var_1250, var_1258, var_1246);
            var_1260 = wp::where(var_1250, var_1255, var_1247);
            var_1261 = wp::where(var_1250, var_1256, var_1225);
            if (!var_1250) {
                // x2 = -1.0                                                                  <L 1298>
                // s2 = 0                                                                     <L 1299>
                // x1 = safe_div(u + mb, ma)                                                  <L 1300>
                var_1265 = wp::add(var_1204, var_1207);
                var_1266 = safe_div_1(var_1265, var_1206);
            }
            var_1267 = wp::where(var_1250, var_1259, var_1266);
            var_1268 = wp::where(var_1250, var_1260, var_1263);
            var_1269 = wp::where(var_1250, var_1261, var_1264);
            // if x1 > 1:                                                                     <L 1302>
            var_1271 = (var_1267 > var_1270);
            if (var_1271) {
                // x1 = 1.0                                                                   <L 1303>
                // s1 = 2                                                                     <L 1304>
            }
            var_1274 = wp::where(var_1271, var_1272, var_1267);
            var_1275 = wp::where(var_1271, var_1273, var_1248);
            if (!var_1271) {
                // elif x1 < -1:                                                              <L 1305>
                var_1278 = (var_1274 < var_1277);
                if (var_1278) {
                    // x1 = -1.0                                                              <L 1306>
                    // s1 = 0                                                                 <L 1307>
                }
                var_1282 = wp::where(var_1278, var_1280, var_1274);
                var_1283 = wp::where(var_1278, var_1281, var_1275);
            }
            var_1284 = wp::where(var_1271, var_1274, var_1282);
            var_1285 = wp::where(var_1271, var_1275, var_1283);
        }
        var_1286 = wp::where(var_1254, var_1284, var_1246);
        var_1287 = wp::where(var_1254, var_1268, var_1247);
        var_1288 = wp::where(var_1254, var_1285, var_1248);
        var_1289 = wp::where(var_1254, var_1269, var_1225);
        // dif -= halfaxis * x2                                                               <L 1309>
        var_1290 = wp::mul(var_4, var_1287);
        var_1291 = wp::sub(var_1203, var_1290);
        // dif[j] += box_size[j] * x1                                                         <L 1310>
        var_1292 = wp::extract(var_box_size, var_1152);
        var_1293 = wp::mul(var_1292, var_1286);
        wp::add_inplace(var_1291, var_1152, var_1293);
        // ct = s1 * 3 + s2                                                                   <L 1313>
        var_1295 = wp::mul(var_1288, var_1294);
        var_1296 = wp::add(var_1295, var_1289);
        // dif_sq = wp.length_sq(dif)                                                         <L 1315>
        var_1297 = wp::length_sq(var_1291);
        // if dif_sq < bestdist - MJ_MINVAL:                                                  <L 1316>
        var_1298 = wp::sub(var_27, var_194);
        var_1299 = (var_1297 < var_1298);
        if (var_1299) {
            // bestdist = dif_sq                                                              <L 1317>
            var_1300 = wp::copy(var_1297);
            // bestsegmentpos = x2                                                            <L 1318>
            var_1301 = wp::copy(var_1287);
            // bestboxpos = x1                                                                <L 1319>
            var_1302 = wp::copy(var_1286);
            // c2 = ct // 6                                                                   <L 1321>
            var_1304 = wp::floordiv(var_1296, var_1303);
            // clcorner = i + (1 << j) * c2  # index of closest box corner                    <L 1323>
            var_1306 = wp::lshift(var_1305, var_1152);
            var_1307 = wp::mul(var_1306, var_1304);
            var_1308 = wp::add(var_1149, var_1307);
            // cledge = j  # axis index of closest box edge                                   <L 1324>
            var_1309 = wp::copy(var_1152);
            // cltype = ct  # encoded collision configuration                                 <L 1325>
            var_1310 = wp::copy(var_1296);
        }
        var_1311 = wp::where(var_1299, var_1300, var_27);
        var_1312 = wp::where(var_1299, var_1301, var_30);
        var_1313 = wp::where(var_1299, var_1310, var_33);
        var_1314 = wp::where(var_1299, var_1308, var_137);
        var_1315 = wp::where(var_1299, var_1309, var_140);
        var_1316 = wp::where(var_1299, var_1302, var_142);
        var_1317 = wp::where(var_1299, var_1304, var_1201);
        wp::assign(var_27, var_1311);
        wp::assign(var_30, var_1312);
        wp::assign(var_33, var_1313);
        wp::assign(var_137, var_1314);
        wp::assign(var_140, var_1315);
        wp::assign(var_142, var_1316);
        wp::assign(var_303, var_1317);
        wp::assign(var_174, var_1202);
        wp::assign(var_277, var_1291);
        wp::assign(var_180, var_1204);
        wp::assign(var_181, var_1205);
        wp::assign(var_184, var_1206);
        wp::assign(var_188, var_1207);
        wp::assign(var_189, var_1208);
        wp::assign(var_192, var_1209);
        wp::assign(var_197, var_1211);
        wp::assign(var_272, var_1286);
        wp::assign(var_273, var_1287);
        wp::assign(var_274, var_1288);
        wp::assign(var_275, var_1289);
        wp::assign(var_236, var_1250);
        wp::assign(var_282, var_1296);
        wp::assign(var_283, var_1297);
        goto start_for_14;
    end_for_14:;
    // for j in range(3):                                                                     <L 1244>
    var_1320 = wp::range(var_1319);
    start_for_16:;
        if (iter_cmp(var_1320) == 0) goto end_for_16;
        var_1321 = wp::iter_next(var_1320);
        // if i & (1 << j) != 0:                                                              <L 1245>
        var_1323 = wp::lshift(var_1322, var_1321);
        var_1324 = wp::bit_and(var_1318, var_1323);
        var_1326 = (var_1324 != var_1325);
        if (var_1326) {
            // continue                                                                       <L 1246>
            goto start_for_16;
        }
        // c2 = wp.int32(-123)                                                                <L 1248>
        var_1329 = wp::int32(var_1328);
        // box_pt = wp.cw_mul(                                                                <L 1251>
        // wp.vec3(                                                                           <L 1252>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1253>
        var_1331 = wp::bit_and(var_1318, var_1330);
        var_1335 = wp::where(var_1331, var_1332, var_1334);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1254>
        var_1337 = wp::bit_and(var_1318, var_1336);
        var_1341 = wp::where(var_1337, var_1338, var_1340);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1255>
        var_1343 = wp::bit_and(var_1318, var_1342);
        var_1347 = wp::where(var_1343, var_1344, var_1346);
        var_1348 = wp::vec_t<3, wp::float32>(var_1335, var_1341, var_1347);
        // box_size,                                                                          <L 1257>
        var_1349 = wp::cw_mul(var_1348, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1259>
        wp::assign_inplace(var_1349, var_1321, var_1350);
        // dif = box_pt - pos                                                                 <L 1262>
        var_1351 = wp::sub(var_1349, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1264>
        var_1352 = wp::extract(var_box_size, var_1321);
        var_1353 = wp::neg(var_1352);
        var_1354 = wp::extract(var_1351, var_1321);
        var_1355 = wp::mul(var_1353, var_1354);
        // v = wp.dot(halfaxis, dif)                                                          <L 1265>
        var_1356 = wp::dot(var_4, var_1351);
        // ma = box_size[j] * box_size[j]                                                     <L 1266>
        var_1357 = wp::extract(var_box_size, var_1321);
        var_1358 = wp::extract(var_box_size, var_1321);
        var_1359 = wp::mul(var_1357, var_1358);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1267>
        var_1360 = wp::extract(var_box_size, var_1321);
        var_1361 = wp::neg(var_1360);
        var_1362 = wp::extract(var_4, var_1321);
        var_1363 = wp::mul(var_1361, var_1362);
        // mc = capsule_half_length * capsule_half_length                                     <L 1268>
        var_1364 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1269>
        var_1365 = wp::mul(var_1359, var_1364);
        var_1366 = wp::mul(var_1363, var_1363);
        var_1367 = wp::sub(var_1365, var_1366);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1270>
        var_1368 = wp::abs(var_1367);
        var_1369 = (var_1368 < var_194);
        if (var_1369) {
            // continue                                                                       <L 1271>
            wp::assign(var_303, var_1329);
            wp::assign(var_174, var_1349);
            wp::assign(var_277, var_1351);
            wp::assign(var_180, var_1355);
            wp::assign(var_181, var_1356);
            wp::assign(var_184, var_1359);
            wp::assign(var_188, var_1363);
            wp::assign(var_189, var_1364);
            wp::assign(var_192, var_1367);
            goto start_for_16;
        }
        var_1370 = wp::where(var_1369, var_303, var_1329);
        var_1371 = wp::where(var_1369, var_174, var_1349);
        var_1372 = wp::where(var_1369, var_277, var_1351);
        var_1373 = wp::where(var_1369, var_180, var_1355);
        var_1374 = wp::where(var_1369, var_181, var_1356);
        var_1375 = wp::where(var_1369, var_184, var_1359);
        var_1376 = wp::where(var_1369, var_188, var_1363);
        var_1377 = wp::where(var_1369, var_189, var_1364);
        var_1378 = wp::where(var_1369, var_192, var_1367);
        // idet = 1.0 / det                                                                   <L 1273>
        var_1380 = wp::div(var_1379, var_1378);
        // x1 = wp.float32((mc * u - mb * v) * idet)                                          <L 1276>
        var_1381 = wp::mul(var_1377, var_1373);
        var_1382 = wp::mul(var_1376, var_1374);
        var_1383 = wp::sub(var_1381, var_1382);
        var_1384 = wp::mul(var_1383, var_1380);
        var_1385 = wp::float32(var_1384);
        // x2 = wp.float32((ma * v - mb * u) * idet)                                          <L 1277>
        var_1386 = wp::mul(var_1375, var_1374);
        var_1387 = wp::mul(var_1376, var_1373);
        var_1388 = wp::sub(var_1386, var_1387);
        var_1389 = wp::mul(var_1388, var_1380);
        var_1390 = wp::float32(var_1389);
        // s1 = wp.int32(1)                                                                   <L 1279>
        var_1392 = wp::int32(var_1391);
        // s2 = wp.int32(1)                                                                   <L 1280>
        var_1394 = wp::int32(var_1393);
        // if x1 > 1:                                                                         <L 1282>
        var_1396 = (var_1385 > var_1395);
        if (var_1396) {
            // x1 = 1.0                                                                       <L 1283>
            // s1 = 2                                                                         <L 1284>
            // x2 = safe_div(v - mb, mc)                                                      <L 1285>
            var_1399 = wp::sub(var_1374, var_1376);
            var_1400 = safe_div_1(var_1399, var_1377);
        }
        var_1401 = wp::where(var_1396, var_1397, var_1385);
        var_1402 = wp::where(var_1396, var_1400, var_1390);
        var_1403 = wp::where(var_1396, var_1398, var_1392);
        if (!var_1396) {
            // elif x1 < -1:                                                                  <L 1286>
            var_1406 = (var_1401 < var_1405);
            if (var_1406) {
                // x1 = -1.0                                                                  <L 1287>
                // s1 = 0                                                                     <L 1288>
                // x2 = safe_div(v + mb, mc)                                                  <L 1289>
                var_1410 = wp::add(var_1374, var_1376);
                var_1411 = safe_div_1(var_1410, var_1377);
            }
            var_1412 = wp::where(var_1406, var_1408, var_1401);
            var_1413 = wp::where(var_1406, var_1411, var_1402);
            var_1414 = wp::where(var_1406, var_1409, var_1403);
        }
        var_1415 = wp::where(var_1396, var_1401, var_1412);
        var_1416 = wp::where(var_1396, var_1402, var_1413);
        var_1417 = wp::where(var_1396, var_1403, var_1414);
        // x2_over = x2 > 1.0                                                                 <L 1291>
        var_1419 = (var_1416 > var_1418);
        // if x2_over or x2 < -1.0:                                                           <L 1292>
        var_1422 = (var_1416 < var_1421);
        var_1423 = var_1419 || var_1422;
        if (var_1423) {
            // if x2_over:                                                                    <L 1293>
            if (var_1419) {
                // x2 = 1.0                                                                   <L 1294>
                // s2 = 2                                                                     <L 1295>
                // x1 = safe_div(u - mb, ma)                                                  <L 1296>
                var_1426 = wp::sub(var_1373, var_1376);
                var_1427 = safe_div_1(var_1426, var_1375);
            }
            var_1428 = wp::where(var_1419, var_1427, var_1415);
            var_1429 = wp::where(var_1419, var_1424, var_1416);
            var_1430 = wp::where(var_1419, var_1425, var_1394);
            if (!var_1419) {
                // x2 = -1.0                                                                  <L 1298>
                // s2 = 0                                                                     <L 1299>
                // x1 = safe_div(u + mb, ma)                                                  <L 1300>
                var_1434 = wp::add(var_1373, var_1376);
                var_1435 = safe_div_1(var_1434, var_1375);
            }
            var_1436 = wp::where(var_1419, var_1428, var_1435);
            var_1437 = wp::where(var_1419, var_1429, var_1432);
            var_1438 = wp::where(var_1419, var_1430, var_1433);
            // if x1 > 1:                                                                     <L 1302>
            var_1440 = (var_1436 > var_1439);
            if (var_1440) {
                // x1 = 1.0                                                                   <L 1303>
                // s1 = 2                                                                     <L 1304>
            }
            var_1443 = wp::where(var_1440, var_1441, var_1436);
            var_1444 = wp::where(var_1440, var_1442, var_1417);
            if (!var_1440) {
                // elif x1 < -1:                                                              <L 1305>
                var_1447 = (var_1443 < var_1446);
                if (var_1447) {
                    // x1 = -1.0                                                              <L 1306>
                    // s1 = 0                                                                 <L 1307>
                }
                var_1451 = wp::where(var_1447, var_1449, var_1443);
                var_1452 = wp::where(var_1447, var_1450, var_1444);
            }
            var_1453 = wp::where(var_1440, var_1443, var_1451);
            var_1454 = wp::where(var_1440, var_1444, var_1452);
        }
        var_1455 = wp::where(var_1423, var_1453, var_1415);
        var_1456 = wp::where(var_1423, var_1437, var_1416);
        var_1457 = wp::where(var_1423, var_1454, var_1417);
        var_1458 = wp::where(var_1423, var_1438, var_1394);
        // dif -= halfaxis * x2                                                               <L 1309>
        var_1459 = wp::mul(var_4, var_1456);
        var_1460 = wp::sub(var_1372, var_1459);
        // dif[j] += box_size[j] * x1                                                         <L 1310>
        var_1461 = wp::extract(var_box_size, var_1321);
        var_1462 = wp::mul(var_1461, var_1455);
        wp::add_inplace(var_1460, var_1321, var_1462);
        // ct = s1 * 3 + s2                                                                   <L 1313>
        var_1464 = wp::mul(var_1457, var_1463);
        var_1465 = wp::add(var_1464, var_1458);
        // dif_sq = wp.length_sq(dif)                                                         <L 1315>
        var_1466 = wp::length_sq(var_1460);
        // if dif_sq < bestdist - MJ_MINVAL:                                                  <L 1316>
        var_1467 = wp::sub(var_27, var_194);
        var_1468 = (var_1466 < var_1467);
        if (var_1468) {
            // bestdist = dif_sq                                                              <L 1317>
            var_1469 = wp::copy(var_1466);
            // bestsegmentpos = x2                                                            <L 1318>
            var_1470 = wp::copy(var_1456);
            // bestboxpos = x1                                                                <L 1319>
            var_1471 = wp::copy(var_1455);
            // c2 = ct // 6                                                                   <L 1321>
            var_1473 = wp::floordiv(var_1465, var_1472);
            // clcorner = i + (1 << j) * c2  # index of closest box corner                    <L 1323>
            var_1475 = wp::lshift(var_1474, var_1321);
            var_1476 = wp::mul(var_1475, var_1473);
            var_1477 = wp::add(var_1318, var_1476);
            // cledge = j  # axis index of closest box edge                                   <L 1324>
            var_1478 = wp::copy(var_1321);
            // cltype = ct  # encoded collision configuration                                 <L 1325>
            var_1479 = wp::copy(var_1465);
        }
        var_1480 = wp::where(var_1468, var_1469, var_27);
        var_1481 = wp::where(var_1468, var_1470, var_30);
        var_1482 = wp::where(var_1468, var_1479, var_33);
        var_1483 = wp::where(var_1468, var_1477, var_137);
        var_1484 = wp::where(var_1468, var_1478, var_140);
        var_1485 = wp::where(var_1468, var_1471, var_142);
        var_1486 = wp::where(var_1468, var_1473, var_1370);
        wp::assign(var_27, var_1480);
        wp::assign(var_30, var_1481);
        wp::assign(var_33, var_1482);
        wp::assign(var_137, var_1483);
        wp::assign(var_140, var_1484);
        wp::assign(var_142, var_1485);
        wp::assign(var_303, var_1486);
        wp::assign(var_174, var_1371);
        wp::assign(var_277, var_1460);
        wp::assign(var_180, var_1373);
        wp::assign(var_181, var_1374);
        wp::assign(var_184, var_1375);
        wp::assign(var_188, var_1376);
        wp::assign(var_189, var_1377);
        wp::assign(var_192, var_1378);
        wp::assign(var_197, var_1380);
        wp::assign(var_272, var_1455);
        wp::assign(var_273, var_1456);
        wp::assign(var_274, var_1457);
        wp::assign(var_275, var_1458);
        wp::assign(var_236, var_1419);
        wp::assign(var_282, var_1465);
        wp::assign(var_283, var_1466);
        goto start_for_16;
    end_for_16:;
    // best = wp.float32(0.0)                                                                 <L 1327>
    var_1488 = wp::float32(var_1487);
    // p = wp.vec2(pos.x, pos.y)                                                              <L 1329>
    var_1490 = wp::extract(var_2, var_1489);
    var_1492 = wp::extract(var_2, var_1491);
    var_1493 = wp::vec_t<2, wp::float32>(var_1490, var_1492);
    // dd = wp.vec2(halfaxis.x, halfaxis.y)                                                   <L 1330>
    var_1495 = wp::extract(var_4, var_1494);
    var_1497 = wp::extract(var_4, var_1496);
    var_1498 = wp::vec_t<2, wp::float32>(var_1495, var_1497);
    // s = wp.vec2(box_size[0], box_size[1])                                                  <L 1331>
    var_1500 = wp::extract(var_box_size, var_1499);
    var_1502 = wp::extract(var_box_size, var_1501);
    var_1503 = wp::vec_t<2, wp::float32>(var_1500, var_1502);
    // secondpos = wp.float32(-4.0)                                                           <L 1332>
    var_1506 = wp::float32(var_1505);
    // uu = dd.x * s.y                                                                        <L 1334>
    var_1508 = wp::extract(var_1498, var_1507);
    var_1510 = wp::extract(var_1503, var_1509);
    var_1511 = wp::mul(var_1508, var_1510);
    // vv = dd.y * s.x                                                                        <L 1335>
    var_1513 = wp::extract(var_1498, var_1512);
    var_1515 = wp::extract(var_1503, var_1514);
    var_1516 = wp::mul(var_1513, var_1515);
    // w_neg = dd.x * p.y - dd.y * p.x < 0                                                    <L 1336>
    var_1518 = wp::extract(var_1498, var_1517);
    var_1520 = wp::extract(var_1493, var_1519);
    var_1521 = wp::mul(var_1518, var_1520);
    var_1523 = wp::extract(var_1498, var_1522);
    var_1525 = wp::extract(var_1493, var_1524);
    var_1526 = wp::mul(var_1523, var_1525);
    var_1527 = wp::sub(var_1521, var_1526);
    var_1529 = (var_1527 < var_1528);
    // best = wp.float32(-1.0)                                                                <L 1338>
    var_1532 = wp::float32(var_1531);
    // ee1 = uu - vv                                                                          <L 1340>
    var_1533 = wp::sub(var_1511, var_1516);
    // ee2 = uu + vv                                                                          <L 1341>
    var_1534 = wp::add(var_1511, var_1516);
    // if wp.abs(ee1) > best:                                                                 <L 1343>
    var_1535 = wp::abs(var_1533);
    var_1536 = (var_1535 > var_1532);
    if (var_1536) {
        // best = wp.abs(ee1)                                                                 <L 1344>
        var_1537 = wp::abs(var_1533);
        // c1 = wp.where((ee1 < 0) == w_neg, 0, 3)                                            <L 1345>
        var_1539 = (var_1533 < var_1538);
        var_1540 = (var_1539 == var_1529);
        var_1543 = wp::where(var_1540, var_1541, var_1542);
    }
    var_1544 = wp::where(var_1536, var_1537, var_1532);
    // if wp.abs(ee2) > best:                                                                 <L 1347>
    var_1545 = wp::abs(var_1534);
    var_1546 = (var_1545 > var_1544);
    if (var_1546) {
        // best = wp.abs(ee2)                                                                 <L 1348>
        var_1547 = wp::abs(var_1534);
        // c1 = wp.where((ee2 > 0) == w_neg, 1, 2)                                            <L 1349>
        var_1549 = (var_1534 > var_1548);
        var_1550 = (var_1549 == var_1529);
        var_1553 = wp::where(var_1550, var_1551, var_1552);
    }
    var_1554 = wp::where(var_1546, var_1547, var_1544);
    var_1555 = wp::where(var_1546, var_1553, var_1543);
    // if cltype == -4:  # invalid type                                                       <L 1351>
    var_1558 = (var_33 == var_1557);
    if (var_1558) {
        // return wp.vec2(MJ_MAXVAL), mat23f(), mat23f()                                      <L 1352>
        var_1560 = wp::vec_t<2, wp::float32>(var_1559);
        var_1561 = wp::mat_t<2, 3, wp::float32>();
        var_1562 = wp::mat_t<2, 3, wp::float32>();
        ret_0 = var_1560;
        ret_1 = var_1561;
        ret_2 = var_1562;
        return;
    }
    // if cltype >= 0 and cltype // 3 != 1:  # closest to a corner of the box                 <L 1354>
    var_1564 = (var_33 >= var_1563);
    var_1566 = wp::floordiv(var_33, var_1565);
    var_1568 = (var_1566 != var_1567);
    var_1569 = var_1564 && var_1568;
    if (var_1569) {
        // c1 = axisdir ^ clcorner                                                            <L 1355>
        var_1570 = wp::bit_xor(var_25, var_137);
        // if c1 != 0 and c1 != 7:  # create second contact point                             <L 1360>
        var_1572 = (var_1570 != var_1571);
        var_1574 = (var_1570 != var_1573);
        var_1575 = var_1572 && var_1574;
        if (var_1575) {
            // if c1 == 1 or c1 == 2 or c1 == 4:                                              <L 1361>
            var_1577 = (var_1570 == var_1576);
            var_1579 = (var_1570 == var_1578);
            var_1581 = (var_1570 == var_1580);
            var_1582 = var_1577 || var_1579 || var_1581;
            if (var_1582) {
                // mul = 1                                                                    <L 1362>
            }
            if (!var_1582) {
                // mul = -1                                                                   <L 1364>
                // c1 = 7 - c1                                                                <L 1365>
                var_1587 = wp::sub(var_1586, var_1570);
            }
            var_1588 = wp::where(var_1582, var_1570, var_1587);
            var_1589 = wp::where(var_1582, var_1583, var_1585);
            // if c1 == 1:                                                                    <L 1370>
            var_1591 = (var_1588 == var_1590);
            if (var_1591) {
                // ax = 0                                                                     <L 1371>
                // ax1 = 1                                                                    <L 1372>
                // ax2 = 2                                                                    <L 1373>
            }
            if (!var_1591) {
                // elif c1 == 2:                                                              <L 1374>
                var_1596 = (var_1588 == var_1595);
                if (var_1596) {
                    // ax = 1                                                                 <L 1375>
                    // ax1 = 2                                                                <L 1376>
                    // ax2 = 0                                                                <L 1377>
                }
                var_1600 = wp::where(var_1596, var_1597, var_1592);
                var_1601 = wp::where(var_1596, var_1598, var_1593);
                var_1602 = wp::where(var_1596, var_1599, var_1594);
                if (!var_1596) {
                    // elif c1 == 4:                                                          <L 1378>
                    var_1604 = (var_1588 == var_1603);
                    if (var_1604) {
                        // ax = 2                                                             <L 1379>
                        // ax1 = 0                                                            <L 1380>
                        // ax2 = 1                                                            <L 1381>
                    }
                    var_1608 = wp::where(var_1604, var_1605, var_1600);
                    var_1609 = wp::where(var_1604, var_1606, var_1601);
                    var_1610 = wp::where(var_1604, var_1607, var_1602);
                }
                var_1611 = wp::where(var_1596, var_1600, var_1608);
                var_1612 = wp::where(var_1596, var_1601, var_1609);
                var_1613 = wp::where(var_1596, var_1602, var_1610);
            }
            var_1614 = wp::where(var_1591, var_1592, var_1611);
            var_1615 = wp::where(var_1591, var_1593, var_1612);
            var_1616 = wp::where(var_1591, var_1594, var_1613);
            // if axis[ax] * axis[ax] > 0.5:  # second point along the edge of the box        <L 1383>
            var_1617 = wp::extract(var_3, var_1614);
            var_1618 = wp::extract(var_3, var_1614);
            var_1619 = wp::mul(var_1617, var_1618);
            var_1621 = (var_1619 > var_1620);
            if (var_1621) {
                // m = 2.0 * safe_div(box_size[ax], wp.abs(halfaxis[ax]))                     <L 1384>
                var_1623 = wp::extract(var_box_size, var_1614);
                var_1624 = wp::extract(var_4, var_1614);
                var_1625 = wp::abs(var_1624);
                var_1626 = safe_div_1(var_1623, var_1625);
                var_1627 = wp::mul(var_1622, var_1626);
                // secondpos = min(1.0 - wp.float32(mul) * bestsegmentpos, m)                 <L 1385>
                var_1629 = wp::float32(var_1589);
                var_1630 = wp::mul(var_1629, var_30);
                var_1631 = wp::sub(var_1628, var_1630);
                var_1632 = wp::min(var_1631, var_1627);
            }
            var_1633 = wp::where(var_1621, var_1632, var_1506);
            if (!var_1621) {
                // m = 2.0 * min(                                                             <L 1388>
                // safe_div(box_size[ax1], wp.abs(halfaxis[ax1])),                            <L 1389>
                var_1635 = wp::extract(var_box_size, var_1615);
                var_1636 = wp::extract(var_4, var_1615);
                var_1637 = wp::abs(var_1636);
                var_1638 = safe_div_1(var_1635, var_1637);
                // safe_div(box_size[ax2], wp.abs(halfaxis[ax2])),                            <L 1390>
                var_1639 = wp::extract(var_box_size, var_1616);
                var_1640 = wp::extract(var_4, var_1616);
                var_1641 = wp::abs(var_1640);
                var_1642 = safe_div_1(var_1639, var_1641);
                var_1643 = wp::min(var_1638, var_1642);
                var_1644 = wp::mul(var_1634, var_1643);
                // secondpos = -min(1.0 + wp.float32(mul) * bestsegmentpos, m)                <L 1392>
                var_1646 = wp::float32(var_1589);
                var_1647 = wp::mul(var_1646, var_30);
                var_1648 = wp::add(var_1645, var_1647);
                var_1649 = wp::min(var_1648, var_1644);
                var_1650 = wp::neg(var_1649);
            }
            var_1651 = wp::where(var_1621, var_1633, var_1650);
            var_1652 = wp::where(var_1621, var_1627, var_1644);
            // secondpos *= wp.float32(mul)                                                   <L 1393>
            var_1653 = wp::float32(var_1589);
            var_1654 = wp::mul(var_1651, var_1653);
        }
        var_1655 = wp::where(var_1575, var_1654, var_1506);
        var_1656 = wp::where(var_1575, var_1588, var_1570);
    }
    var_1657 = wp::where(var_1569, var_1655, var_1506);
    var_1658 = wp::where(var_1569, var_1656, var_1555);
    if (!var_1569) {
        // elif cltype >= 0 and cltype // 3 == 1:  # we are on box's edge                     <L 1395>
        var_1660 = (var_33 >= var_1659);
        var_1662 = wp::floordiv(var_33, var_1661);
        var_1664 = (var_1662 == var_1663);
        var_1665 = var_1660 && var_1664;
        if (var_1665) {
            // c1 = axisdir ^ clcorner                                                        <L 1400>
            var_1666 = wp::bit_xor(var_25, var_137);
            // c1 &= 7 - (1 << cledge)  # mask out edge axis to determine configuration       <L 1401>
            var_1669 = wp::lshift(var_1668, var_140);
            var_1670 = wp::sub(var_1667, var_1669);
            var_1671 = wp::bit_and(var_1666, var_1670);
            // if c1 == 1 or c1 == 2 or c1 == 4:  # create second contact point               <L 1403>
            var_1673 = (var_1671 == var_1672);
            var_1675 = (var_1671 == var_1674);
            var_1677 = (var_1671 == var_1676);
            var_1678 = var_1673 || var_1675 || var_1677;
            if (var_1678) {
                // if cledge == 0:                                                            <L 1404>
                var_1680 = (var_140 == var_1679);
                if (var_1680) {
                    // ax1 = 1                                                                <L 1405>
                    // ax2 = 2                                                                <L 1406>
                }
                var_1683 = wp::where(var_1680, var_1681, var_1615);
                var_1684 = wp::where(var_1680, var_1682, var_1616);
                // if cledge == 1:                                                            <L 1407>
                var_1686 = (var_140 == var_1685);
                if (var_1686) {
                    // ax1 = 2                                                                <L 1408>
                    // ax2 = 0                                                                <L 1409>
                }
                var_1689 = wp::where(var_1686, var_1687, var_1683);
                var_1690 = wp::where(var_1686, var_1688, var_1684);
                // if cledge == 2:                                                            <L 1410>
                var_1692 = (var_140 == var_1691);
                if (var_1692) {
                    // ax1 = 0                                                                <L 1411>
                    // ax2 = 1                                                                <L 1412>
                }
                var_1695 = wp::where(var_1692, var_1693, var_1689);
                var_1696 = wp::where(var_1692, var_1694, var_1690);
                // ax = cledge                                                                <L 1413>
                var_1697 = wp::copy(var_140);
                // if wp.abs(axis[ax1]) > wp.abs(axis[ax2]):                                  <L 1416>
                var_1698 = wp::extract(var_3, var_1695);
                var_1699 = wp::abs(var_1698);
                var_1700 = wp::extract(var_3, var_1696);
                var_1701 = wp::abs(var_1700);
                var_1702 = (var_1699 > var_1701);
                if (var_1702) {
                    // ax1 = ax2                                                              <L 1417>
                    var_1703 = wp::copy(var_1696);
                }
                var_1704 = wp::where(var_1702, var_1703, var_1695);
                // ax2 = 3 - ax - ax1                                                         <L 1418>
                var_1706 = wp::sub(var_1705, var_1697);
                var_1707 = wp::sub(var_1706, var_1704);
                // if c1 & (1 << ax2):                                                        <L 1421>
                var_1709 = wp::lshift(var_1708, var_1707);
                var_1710 = wp::bit_and(var_1671, var_1709);
                if (var_1710) {
                    // mul = 1                                                                <L 1422>
                    // secondpos = 1.0 - bestsegmentpos                                       <L 1423>
                    var_1713 = wp::sub(var_1712, var_30);
                }
                var_1714 = wp::where(var_1710, var_1713, var_1657);
                var_1715 = wp::where(var_1710, var_1711, var_1589);
                if (!var_1710) {
                    // mul = -1                                                               <L 1425>
                    // secondpos = 1.0 + bestsegmentpos                                       <L 1426>
                    var_1719 = wp::add(var_1718, var_30);
                }
                var_1720 = wp::where(var_1710, var_1714, var_1719);
                var_1721 = wp::where(var_1710, var_1715, var_1717);
                // e1 = 2.0 * safe_div(box_size[ax2], wp.abs(halfaxis[ax2]))                  <L 1431>
                var_1723 = wp::extract(var_box_size, var_1707);
                var_1724 = wp::extract(var_4, var_1707);
                var_1725 = wp::abs(var_1724);
                var_1726 = safe_div_1(var_1723, var_1725);
                var_1727 = wp::mul(var_1722, var_1726);
                // secondpos = min(e1, secondpos)                                             <L 1432>
                var_1728 = wp::min(var_1727, var_1720);
                // if ((axisdir & (1 << ax)) != 0) == ((c1 & (1 << ax2)) != 0):               <L 1434>
                var_1730 = wp::lshift(var_1729, var_1697);
                var_1731 = wp::bit_and(var_25, var_1730);
                var_1733 = (var_1731 != var_1732);
                var_1735 = wp::lshift(var_1734, var_1707);
                var_1736 = wp::bit_and(var_1671, var_1735);
                var_1738 = (var_1736 != var_1737);
                var_1739 = (var_1733 == var_1738);
                if (var_1739) {
                    // e2 = 1.0 - bestboxpos                                                  <L 1435>
                    var_1741 = wp::sub(var_1740, var_142);
                }
                if (!var_1739) {
                    // e2 = 1.0 + bestboxpos                                                  <L 1437>
                    var_1743 = wp::add(var_1742, var_142);
                }
                var_1744 = wp::where(var_1739, var_1741, var_1743);
                // e1 = box_size[ax] * safe_div(e2, wp.abs(halfaxis[ax]))                     <L 1439>
                var_1745 = wp::extract(var_box_size, var_1697);
                var_1746 = wp::extract(var_4, var_1697);
                var_1747 = wp::abs(var_1746);
                var_1748 = safe_div_1(var_1744, var_1747);
                var_1749 = wp::mul(var_1745, var_1748);
                // secondpos = min(e1, secondpos)                                             <L 1441>
                var_1750 = wp::min(var_1749, var_1728);
                // secondpos *= wp.float32(mul)                                               <L 1442>
                var_1751 = wp::float32(var_1721);
                var_1752 = wp::mul(var_1750, var_1751);
            }
            var_1753 = wp::where(var_1678, var_1752, var_1657);
            var_1754 = wp::where(var_1678, var_1721, var_1589);
            var_1755 = wp::where(var_1678, var_1697, var_1614);
            var_1756 = wp::where(var_1678, var_1704, var_1615);
            var_1757 = wp::where(var_1678, var_1707, var_1616);
        }
        var_1758 = wp::where(var_1665, var_1753, var_1657);
        var_1759 = wp::where(var_1665, var_1671, var_1658);
        var_1760 = wp::where(var_1665, var_1754, var_1589);
        var_1761 = wp::where(var_1665, var_1755, var_1614);
        var_1762 = wp::where(var_1665, var_1756, var_1615);
        var_1763 = wp::where(var_1665, var_1757, var_1616);
        if (!var_1665) {
            // elif cltype < 0:                                                               <L 1444>
            var_1765 = (var_33 < var_1764);
            if (var_1765) {
                // if clface != -1:  # create second contact point                            <L 1450>
                var_1768 = (var_36 != var_1767);
                if (var_1768) {
                    // mul = wp.where(cltype == -3, 1, -1)                                    <L 1451>
                    var_1771 = (var_33 == var_1770);
                    var_1775 = wp::where(var_1771, var_1772, var_1774);
                    // secondpos = 2.0                                                        <L 1452>
                    // tmp1 = pos - halfaxis * wp.float32(mul)                                <L 1454>
                    var_1777 = wp::float32(var_1775);
                    var_1778 = wp::mul(var_4, var_1777);
                    var_1779 = wp::sub(var_2, var_1778);
                    // for i in range(3):                                                     <L 1456>
                    // if i != clface:                                                        <L 1457>
                    var_1781 = (var_1780 != var_36);
                    if (var_1781) {
                        // ha_r = safe_div(wp.float32(mul), halfaxis[i])                      <L 1458>
                        var_1782 = wp::float32(var_1775);
                        var_1783 = wp::extract(var_4, var_1780);
                        var_1784 = safe_div_1(var_1782, var_1783);
                        // e1 = (box_size[i] - tmp1[i]) * ha_r                                <L 1459>
                        var_1785 = wp::extract(var_box_size, var_1780);
                        var_1786 = wp::extract(var_1779, var_1780);
                        var_1787 = wp::sub(var_1785, var_1786);
                        var_1788 = wp::mul(var_1787, var_1784);
                        // if 0 < e1 and e1 < secondpos:                                      <L 1460>
                        var_1790 = (var_1789 < var_1788);
                        var_1791 = (var_1788 < var_1776);
                        var_1792 = var_1790 && var_1791;
                        if (var_1792) {
                            // secondpos = e1                                                 <L 1461>
                            var_1793 = wp::copy(var_1788);
                        }
                        var_1794 = wp::where(var_1792, var_1793, var_1776);
                        // e1 = (-box_size[i] - tmp1[i]) * ha_r                               <L 1463>
                        var_1795 = wp::extract(var_box_size, var_1780);
                        var_1796 = wp::neg(var_1795);
                        var_1797 = wp::extract(var_1779, var_1780);
                        var_1798 = wp::sub(var_1796, var_1797);
                        var_1799 = wp::mul(var_1798, var_1784);
                        // if 0 < e1 and e1 < secondpos:                                      <L 1464>
                        var_1801 = (var_1800 < var_1799);
                        var_1802 = (var_1799 < var_1794);
                        var_1803 = var_1801 && var_1802;
                        if (var_1803) {
                            // secondpos = e1                                                 <L 1465>
                            var_1804 = wp::copy(var_1799);
                        }
                        var_1805 = wp::where(var_1803, var_1804, var_1794);
                    }
                    var_1806 = wp::where(var_1781, var_1805, var_1776);
                    var_1807 = wp::where(var_1781, var_1799, var_1749);
                    // if i != clface:                                                        <L 1457>
                    var_1809 = (var_1808 != var_36);
                    if (var_1809) {
                        // ha_r = safe_div(wp.float32(mul), halfaxis[i])                      <L 1458>
                        var_1810 = wp::float32(var_1775);
                        var_1811 = wp::extract(var_4, var_1808);
                        var_1812 = safe_div_1(var_1810, var_1811);
                        // e1 = (box_size[i] - tmp1[i]) * ha_r                                <L 1459>
                        var_1813 = wp::extract(var_box_size, var_1808);
                        var_1814 = wp::extract(var_1779, var_1808);
                        var_1815 = wp::sub(var_1813, var_1814);
                        var_1816 = wp::mul(var_1815, var_1812);
                        // if 0 < e1 and e1 < secondpos:                                      <L 1460>
                        var_1818 = (var_1817 < var_1816);
                        var_1819 = (var_1816 < var_1806);
                        var_1820 = var_1818 && var_1819;
                        if (var_1820) {
                            // secondpos = e1                                                 <L 1461>
                            var_1821 = wp::copy(var_1816);
                        }
                        var_1822 = wp::where(var_1820, var_1821, var_1806);
                        // e1 = (-box_size[i] - tmp1[i]) * ha_r                               <L 1463>
                        var_1823 = wp::extract(var_box_size, var_1808);
                        var_1824 = wp::neg(var_1823);
                        var_1825 = wp::extract(var_1779, var_1808);
                        var_1826 = wp::sub(var_1824, var_1825);
                        var_1827 = wp::mul(var_1826, var_1812);
                        // if 0 < e1 and e1 < secondpos:                                      <L 1464>
                        var_1829 = (var_1828 < var_1827);
                        var_1830 = (var_1827 < var_1822);
                        var_1831 = var_1829 && var_1830;
                        if (var_1831) {
                            // secondpos = e1                                                 <L 1465>
                            var_1832 = wp::copy(var_1827);
                        }
                        var_1833 = wp::where(var_1831, var_1832, var_1822);
                    }
                    var_1834 = wp::where(var_1809, var_1833, var_1806);
                    var_1835 = wp::where(var_1809, var_1827, var_1807);
                    var_1836 = wp::where(var_1809, var_1812, var_1784);
                    // if i != clface:                                                        <L 1457>
                    var_1838 = (var_1837 != var_36);
                    if (var_1838) {
                        // ha_r = safe_div(wp.float32(mul), halfaxis[i])                      <L 1458>
                        var_1839 = wp::float32(var_1775);
                        var_1840 = wp::extract(var_4, var_1837);
                        var_1841 = safe_div_1(var_1839, var_1840);
                        // e1 = (box_size[i] - tmp1[i]) * ha_r                                <L 1459>
                        var_1842 = wp::extract(var_box_size, var_1837);
                        var_1843 = wp::extract(var_1779, var_1837);
                        var_1844 = wp::sub(var_1842, var_1843);
                        var_1845 = wp::mul(var_1844, var_1841);
                        // if 0 < e1 and e1 < secondpos:                                      <L 1460>
                        var_1847 = (var_1846 < var_1845);
                        var_1848 = (var_1845 < var_1834);
                        var_1849 = var_1847 && var_1848;
                        if (var_1849) {
                            // secondpos = e1                                                 <L 1461>
                            var_1850 = wp::copy(var_1845);
                        }
                        var_1851 = wp::where(var_1849, var_1850, var_1834);
                        // e1 = (-box_size[i] - tmp1[i]) * ha_r                               <L 1463>
                        var_1852 = wp::extract(var_box_size, var_1837);
                        var_1853 = wp::neg(var_1852);
                        var_1854 = wp::extract(var_1779, var_1837);
                        var_1855 = wp::sub(var_1853, var_1854);
                        var_1856 = wp::mul(var_1855, var_1841);
                        // if 0 < e1 and e1 < secondpos:                                      <L 1464>
                        var_1858 = (var_1857 < var_1856);
                        var_1859 = (var_1856 < var_1851);
                        var_1860 = var_1858 && var_1859;
                        if (var_1860) {
                            // secondpos = e1                                                 <L 1465>
                            var_1861 = wp::copy(var_1856);
                        }
                        var_1862 = wp::where(var_1860, var_1861, var_1851);
                    }
                    var_1863 = wp::where(var_1838, var_1862, var_1834);
                    var_1864 = wp::where(var_1838, var_1856, var_1835);
                    var_1865 = wp::where(var_1838, var_1841, var_1836);
                    // secondpos *= wp.float32(mul)                                           <L 1467>
                    var_1866 = wp::float32(var_1775);
                    var_1867 = wp::mul(var_1863, var_1866);
                }
                var_1868 = wp::where(var_1768, var_1837, var_1318);
                var_1869 = wp::where(var_1768, var_1867, var_1758);
                var_1870 = wp::where(var_1768, var_1775, var_1760);
                var_1871 = wp::where(var_1768, var_1864, var_1749);
            }
            var_1872 = wp::where(var_1765, var_1868, var_1318);
            var_1873 = wp::where(var_1765, var_1869, var_1758);
            var_1874 = wp::where(var_1765, var_1870, var_1760);
            var_1875 = wp::where(var_1765, var_1871, var_1749);
        }
        var_1876 = wp::where(var_1665, var_1318, var_1872);
        var_1877 = wp::where(var_1665, var_1758, var_1873);
        var_1878 = wp::where(var_1665, var_1760, var_1874);
        var_1879 = wp::where(var_1665, var_1749, var_1875);
    }
    var_1880 = wp::where(var_1569, var_1318, var_1876);
    var_1881 = wp::where(var_1569, var_1657, var_1877);
    var_1882 = wp::where(var_1569, var_1658, var_1759);
    var_1883 = wp::where(var_1569, var_1589, var_1878);
    var_1884 = wp::where(var_1569, var_1614, var_1761);
    var_1885 = wp::where(var_1569, var_1615, var_1762);
    var_1886 = wp::where(var_1569, var_1616, var_1763);
    // s1_pos_l = pos + halfaxis * bestsegmentpos                                             <L 1470>
    var_1887 = wp::mul(var_4, var_30);
    var_1888 = wp::add(var_2, var_1887);
    // s1_pos_g = box_rot @ s1_pos_l + box_pos                                                <L 1471>
    var_1889 = wp::mul(var_box_rot, var_1888);
    var_1890 = wp::add(var_1889, var_box_pos);
    // dist1, pos1, normal1 = sphere_box(s1_pos_g, capsule_radius, box_pos, box_rot, box_size)       <L 1474>
    sphere_box_0(var_1890, var_capsule_radius, var_box_pos, var_box_rot, var_box_size, var_1891, var_1892, var_1893);
    // if secondpos > -3:  # secondpos was modified                                           <L 1476>
    var_1896 = (var_1881 > var_1895);
    if (var_1896) {
        // s2_pos_l = pos + halfaxis * (secondpos + bestsegmentpos)                           <L 1477>
        var_1897 = wp::add(var_1881, var_30);
        var_1898 = wp::mul(var_4, var_1897);
        var_1899 = wp::add(var_2, var_1898);
        // s2_pos_g = box_rot @ s2_pos_l + box_pos                                            <L 1478>
        var_1900 = wp::mul(var_box_rot, var_1899);
        var_1901 = wp::add(var_1900, var_box_pos);
        // dist2, pos2, normal2 = sphere_box(s2_pos_g, capsule_radius, box_pos, box_rot, box_size)       <L 1481>
        sphere_box_0(var_1901, var_capsule_radius, var_box_pos, var_box_rot, var_box_size, var_1902, var_1903, var_1904);
    }
    if (!var_1896) {
        // dist2 = MJ_MAXVAL                                                                  <L 1483>
        var_1905 = wp::copy(var_1559);
        // pos2 = wp.vec3()                                                                   <L 1484>
        var_1906 = wp::vec_t<3, wp::float32>();
        // normal2 = wp.vec3()                                                                <L 1485>
        var_1907 = wp::vec_t<3, wp::float32>();
    }
    var_1908 = wp::where(var_1896, var_1902, var_1905);
    var_1909 = wp::where(var_1896, var_1903, var_1906);
    var_1910 = wp::where(var_1896, var_1904, var_1907);
    // return (                                                                               <L 1487>
    // wp.vec2(dist1, dist2),                                                                 <L 1488>
    var_1911 = wp::vec_t<2, wp::float32>(var_1891, var_1908);
    // mat23f(pos1[0], pos1[1], pos1[2], pos2[0], pos2[1], pos2[2]),                          <L 1489>
    var_1913 = wp::extract(var_1892, var_1912);
    var_1915 = wp::extract(var_1892, var_1914);
    var_1917 = wp::extract(var_1892, var_1916);
    var_1919 = wp::extract(var_1909, var_1918);
    var_1921 = wp::extract(var_1909, var_1920);
    var_1923 = wp::extract(var_1909, var_1922);
    var_1924 = wp::mat_t<2, 3, wp::float32>({var_1913, var_1915, var_1917, var_1919, var_1921, var_1923});
    // mat23f(normal1[0], normal1[1], normal1[2], normal2[0], normal2[1], normal2[2]),        <L 1490>
    var_1926 = wp::extract(var_1893, var_1925);
    var_1928 = wp::extract(var_1893, var_1927);
    var_1930 = wp::extract(var_1893, var_1929);
    var_1932 = wp::extract(var_1910, var_1931);
    var_1934 = wp::extract(var_1910, var_1933);
    var_1936 = wp::extract(var_1910, var_1935);
    var_1937 = wp::mat_t<2, 3, wp::float32>({var_1926, var_1928, var_1930, var_1932, var_1934, var_1936});
    ret_0 = var_1911;
    ret_1 = var_1924;
    ret_2 = var_1937;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_primitive.py:1116
static CUDA_CALLABLE void capsule_box_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_3242f8a8 var_cap,
    Geom_3242f8a8 var_box,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
    wp::array_t<wp::float32> var_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_out,
    wp::array_t<wp::float32> var_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_solimp_out,
    wp::array_t<wp::int32> var_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::int32> var_nacon_out)
{
    //---------
    // primal vars
    wp::mat_t<3, 3, wp::float32>* var_0;
    const wp::int32 var_1 = 0;
    const wp::int32 var_2 = 2;
    wp::float32 var_3;
    wp::mat_t<3, 3, wp::float32> var_4;
    wp::mat_t<3, 3, wp::float32>* var_5;
    const wp::int32 var_6 = 1;
    const wp::int32 var_7 = 2;
    wp::float32 var_8;
    wp::mat_t<3, 3, wp::float32> var_9;
    wp::mat_t<3, 3, wp::float32>* var_10;
    const wp::int32 var_11 = 2;
    const wp::int32 var_12 = 2;
    wp::float32 var_13;
    wp::mat_t<3, 3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32>* var_16;
    wp::vec_t<3, wp::float32>* var_17;
    const wp::int32 var_18 = 0;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32>* var_21;
    const wp::int32 var_22 = 1;
    wp::float32 var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32>* var_25;
    wp::mat_t<3, 3, wp::float32>* var_26;
    wp::vec_t<3, wp::float32>* var_27;
    wp::vec_t<2, wp::float32> var_28;
    wp::mat_t<2, 3, wp::float32> var_29;
    wp::mat_t<2, 3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::mat_t<3, 3, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    const wp::int32 var_35 = 0;
    wp::float32 var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::mat_t<3, 3, wp::float32> var_39;
    wp::int32 var_40;
    const wp::int32 var_41 = 1;
    wp::float32 var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::mat_t<3, 3, wp::float32> var_45;
    wp::int32 var_46;
    //---------
    // forward
    // def capsule_box_wrapper(                                                               <L 1117>
    // axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])                            <L 1152>
    var_0 = &(var_cap.rot);
    var_4 = wp::load(var_0);
    var_3 = wp::extract(var_4, var_1, var_2);
    var_5 = &(var_cap.rot);
    var_9 = wp::load(var_5);
    var_8 = wp::extract(var_9, var_6, var_7);
    var_10 = &(var_cap.rot);
    var_14 = wp::load(var_10);
    var_13 = wp::extract(var_14, var_11, var_12);
    var_15 = wp::vec_t<3, wp::float32>(var_3, var_8, var_13);
    // dist, pos, normal = capsule_box(                                                       <L 1155>
    // cap.pos,                                                                               <L 1156>
    var_16 = &(var_cap.pos);
    // axis,                                                                                  <L 1157>
    // cap.size[0],  # capsule radius                                                         <L 1158>
    var_17 = &(var_cap.size);
    var_20 = wp::load(var_17);
    var_19 = wp::extract(var_20, var_18);
    // cap.size[1],  # capsule half length                                                    <L 1159>
    var_21 = &(var_cap.size);
    var_24 = wp::load(var_21);
    var_23 = wp::extract(var_24, var_22);
    // box.pos,                                                                               <L 1160>
    var_25 = &(var_box.pos);
    // box.rot,                                                                               <L 1161>
    var_26 = &(var_box.rot);
    // box.size,                                                                              <L 1162>
    var_27 = &(var_box.size);
    var_31 = wp::load(var_16);
    var_32 = wp::load(var_25);
    var_33 = wp::load(var_26);
    var_34 = wp::load(var_27);
    capsule_box_0(var_31, var_15, var_19, var_23, var_32, var_33, var_34, var_28, var_29, var_30);
    // for i in range(2):                                                                     <L 1166>
    // write_contact(                                                                         <L 1167>
    // naconmax_in,                                                                           <L 1168>
    // i,                                                                                     <L 1169>
    // dist[i],                                                                               <L 1170>
    var_36 = wp::extract(var_28, var_35);
    // pos[i],                                                                                <L 1171>
    var_37 = wp::extract(var_29, var_35);
    // make_frame(normal[i]),                                                                 <L 1172>
    var_38 = wp::extract(var_30, var_35);
    var_39 = make_frame_0(var_38);
    // margin,                                                                                <L 1173>
    // gap,                                                                                   <L 1174>
    // condim,                                                                                <L 1175>
    // friction,                                                                              <L 1176>
    // solref,                                                                                <L 1177>
    // solreffriction,                                                                        <L 1178>
    // solimp,                                                                                <L 1179>
    // geoms,                                                                                 <L 1180>
    // pairid,                                                                                <L 1181>
    // worldid,                                                                               <L 1182>
    // contact_dist_out,                                                                      <L 1183>
    // contact_pos_out,                                                                       <L 1184>
    // contact_frame_out,                                                                     <L 1185>
    // contact_includemargin_out,                                                             <L 1186>
    // contact_friction_out,                                                                  <L 1187>
    // contact_solref_out,                                                                    <L 1188>
    // contact_solreffriction_out,                                                            <L 1189>
    // contact_solimp_out,                                                                    <L 1190>
    // contact_dim_out,                                                                       <L 1191>
    // contact_geom_out,                                                                      <L 1192>
    // contact_efc_address_out,                                                               <L 1193>
    // contact_worldid_out,                                                                   <L 1194>
    // contact_type_out,                                                                      <L 1195>
    // contact_geomcollisionid_out,                                                           <L 1196>
    // nacon_out,                                                                             <L 1197>
    var_40 = write_contact_0(var_naconmax_in, var_35, var_36, var_37, var_39, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
    // write_contact(                                                                         <L 1167>
    // naconmax_in,                                                                           <L 1168>
    // i,                                                                                     <L 1169>
    // dist[i],                                                                               <L 1170>
    var_42 = wp::extract(var_28, var_41);
    // pos[i],                                                                                <L 1171>
    var_43 = wp::extract(var_29, var_41);
    // make_frame(normal[i]),                                                                 <L 1172>
    var_44 = wp::extract(var_30, var_41);
    var_45 = make_frame_0(var_44);
    // margin,                                                                                <L 1173>
    // gap,                                                                                   <L 1174>
    // condim,                                                                                <L 1175>
    // friction,                                                                              <L 1176>
    // solref,                                                                                <L 1177>
    // solreffriction,                                                                        <L 1178>
    // solimp,                                                                                <L 1179>
    // geoms,                                                                                 <L 1180>
    // pairid,                                                                                <L 1181>
    // worldid,                                                                               <L 1182>
    // contact_dist_out,                                                                      <L 1183>
    // contact_pos_out,                                                                       <L 1184>
    // contact_frame_out,                                                                     <L 1185>
    // contact_includemargin_out,                                                             <L 1186>
    // contact_friction_out,                                                                  <L 1187>
    // contact_solref_out,                                                                    <L 1188>
    // contact_solreffriction_out,                                                            <L 1189>
    // contact_solimp_out,                                                                    <L 1190>
    // contact_dim_out,                                                                       <L 1191>
    // contact_geom_out,                                                                      <L 1192>
    // contact_efc_address_out,                                                               <L 1193>
    // contact_worldid_out,                                                                   <L 1194>
    // contact_type_out,                                                                      <L 1195>
    // contact_geomcollisionid_out,                                                           <L 1196>
    // nacon_out,                                                                             <L 1197>
    var_46 = write_contact_0(var_naconmax_in, var_41, var_42, var_43, var_45, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_core.py:235
static CUDA_CALLABLE void adj_contact_params_0(
    wp::array_t<wp::int32> var_geom_condim,
    wp::array_t<wp::int32> var_geom_priority,
    wp::array_t<wp::float32> var_geom_solmix,
    wp::array_t<wp::vec_t<2, wp::float32>> var_geom_solref,
    wp::array_t<wp::vec_t<5, wp::float32>> var_geom_solimp,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_friction,
    wp::array_t<wp::float32> var_geom_margin,
    wp::array_t<wp::float32> var_geom_gap,
    wp::array_t<wp::int32> var_pair_dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_pair_solref,
    wp::array_t<wp::vec_t<2, wp::float32>> var_pair_solreffriction,
    wp::array_t<wp::vec_t<5, wp::float32>> var_pair_solimp,
    wp::array_t<wp::float32> var_pair_margin,
    wp::array_t<wp::float32> var_pair_gap,
    wp::array_t<wp::vec_t<5, wp::float32>> var_pair_friction,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pair_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pairid_in,
    wp::int32 var_cid,
    wp::int32 var_worldid,
    wp::vec_t<2, wp::int32> & ret_0,
    wp::float32 & ret_1,
    wp::float32 & ret_2,
    wp::int32 & ret_3,
    wp::vec_t<5, wp::float32> & ret_4,
    wp::vec_t<2, wp::float32> & ret_5,
    wp::vec_t<2, wp::float32> & ret_6,
    wp::vec_t<5, wp::float32> & ret_7,
    wp::array_t<wp::int32> & adj_geom_condim,
    wp::array_t<wp::int32> & adj_geom_priority,
    wp::array_t<wp::float32> & adj_geom_solmix,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_geom_solref,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_geom_solimp,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_friction,
    wp::array_t<wp::float32> & adj_geom_margin,
    wp::array_t<wp::float32> & adj_geom_gap,
    wp::array_t<wp::int32> & adj_pair_dim,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_pair_solref,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_pair_solreffriction,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_pair_solimp,
    wp::array_t<wp::float32> & adj_pair_margin,
    wp::array_t<wp::float32> & adj_pair_gap,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_pair_friction,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_collision_pair_in,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_collision_pairid_in,
    wp::int32 & adj_cid,
    wp::int32 & adj_worldid,
    wp::vec_t<2, wp::int32> & adj_ret_0,
    wp::float32 & adj_ret_1,
    wp::float32 & adj_ret_2,
    wp::int32 & adj_ret_3,
    wp::vec_t<5, wp::float32> & adj_ret_4,
    wp::vec_t<2, wp::float32> & adj_ret_5,
    wp::vec_t<2, wp::float32> & adj_ret_6,
    wp::vec_t<5, wp::float32> & adj_ret_7)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_core.py:65
static CUDA_CALLABLE void adj_geom_collision_pair_0(
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::int32> var_geom_dataid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::int32> var_mesh_vertadr,
    wp::array_t<wp::int32> var_mesh_vertnum,
    wp::array_t<wp::int32> var_mesh_graphadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_vert,
    wp::array_t<wp::int32> var_mesh_graph,
    wp::array_t<wp::int32> var_mesh_polynum,
    wp::array_t<wp::int32> var_mesh_polyadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_polynormal,
    wp::array_t<wp::int32> var_mesh_polyvertadr,
    wp::array_t<wp::int32> var_mesh_polyvertnum,
    wp::array_t<wp::int32> var_mesh_polyvert,
    wp::array_t<wp::int32> var_mesh_polymapadr,
    wp::array_t<wp::int32> var_mesh_polymapnum,
    wp::array_t<wp::int32> var_mesh_polymap,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::int32 var_worldid,
    Geom_3242f8a8 & ret_0,
    Geom_3242f8a8 & ret_1,
    wp::array_t<wp::int32> & adj_geom_type,
    wp::array_t<wp::int32> & adj_geom_dataid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_size,
    wp::array_t<wp::int32> & adj_mesh_vertadr,
    wp::array_t<wp::int32> & adj_mesh_vertnum,
    wp::array_t<wp::int32> & adj_mesh_graphadr,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_mesh_vert,
    wp::array_t<wp::int32> & adj_mesh_graph,
    wp::array_t<wp::int32> & adj_mesh_polynum,
    wp::array_t<wp::int32> & adj_mesh_polyadr,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_mesh_polynormal,
    wp::array_t<wp::int32> & adj_mesh_polyvertadr,
    wp::array_t<wp::int32> & adj_mesh_polyvertnum,
    wp::array_t<wp::int32> & adj_mesh_polyvert,
    wp::array_t<wp::int32> & adj_mesh_polymapadr,
    wp::array_t<wp::int32> & adj_mesh_polymapnum,
    wp::array_t<wp::int32> & adj_mesh_polymap,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_geom_xmat_in,
    wp::vec_t<2, wp::int32> & adj_geoms,
    wp::int32 & adj_worldid,
    Geom_3242f8a8 & adj_ret_0,
    Geom_3242f8a8 & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_primitive_core.py:114
static CUDA_CALLABLE void adj_sphere_sphere_0(
    wp::vec_t<3, wp::float32> var_pos1,
    wp::float32 var_radius1,
    wp::vec_t<3, wp::float32> var_pos2,
    wp::float32 var_radius2,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_pos1,
    wp::float32 & adj_radius1,
    wp::vec_t<3, wp::float32> & adj_pos2,
    wp::float32 & adj_radius2,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:202
static CUDA_CALLABLE void adj_orthogonals_0(
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_a,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:246
static CUDA_CALLABLE void adj_make_frame_0(
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> & adj_a,
    wp::mat_t<3, 3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_core.py:159
static CUDA_CALLABLE void adj_write_contact_0(
    wp::int32 var_naconmax_in,
    wp::int32 var_id_,
    wp::float32 var_dist_in,
    wp::vec_t<3, wp::float32> var_pos_in,
    wp::mat_t<3, 3, wp::float32> var_frame_in,
    wp::float32 var_margin_in,
    wp::float32 var_gap_in,
    wp::int32 var_condim_in,
    wp::vec_t<5, wp::float32> var_friction_in,
    wp::vec_t<2, wp::float32> var_solref_in,
    wp::vec_t<2, wp::float32> var_solreffriction_in,
    wp::vec_t<5, wp::float32> var_solimp_in,
    wp::vec_t<2, wp::int32> var_geoms_in,
    wp::vec_t<2, wp::int32> var_pairid_in,
    wp::int32 var_worldid_in,
    wp::array_t<wp::float32> var_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_out,
    wp::array_t<wp::float32> var_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_solimp_out,
    wp::array_t<wp::int32> var_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::int32> var_nacon_out,
    wp::int32 & adj_naconmax_in,
    wp::int32 & adj_id_,
    wp::float32 & adj_dist_in,
    wp::vec_t<3, wp::float32> & adj_pos_in,
    wp::mat_t<3, 3, wp::float32> & adj_frame_in,
    wp::float32 & adj_margin_in,
    wp::float32 & adj_gap_in,
    wp::int32 & adj_condim_in,
    wp::vec_t<5, wp::float32> & adj_friction_in,
    wp::vec_t<2, wp::float32> & adj_solref_in,
    wp::vec_t<2, wp::float32> & adj_solreffriction_in,
    wp::vec_t<5, wp::float32> & adj_solimp_in,
    wp::vec_t<2, wp::int32> & adj_geoms_in,
    wp::vec_t<2, wp::int32> & adj_pairid_in,
    wp::int32 & adj_worldid_in,
    wp::array_t<wp::float32> & adj_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_contact_frame_out,
    wp::array_t<wp::float32> & adj_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_solimp_out,
    wp::array_t<wp::int32> & adj_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_contact_geom_out,
    wp::array_t<wp::int32> & adj_contact_efc_address_out,
    wp::array_t<wp::int32> & adj_contact_worldid_out,
    wp::array_t<wp::int32> & adj_contact_type_out,
    wp::array_t<wp::int32> & adj_contact_geomcollisionid_out,
    wp::array_t<wp::int32> & adj_nacon_out,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_primitive.py:352
static CUDA_CALLABLE void adj_sphere_sphere_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_3242f8a8 var_sphere1,
    Geom_3242f8a8 var_sphere2,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
    wp::array_t<wp::float32> var_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_out,
    wp::array_t<wp::float32> var_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_solimp_out,
    wp::array_t<wp::int32> var_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::int32> var_nacon_out,
    wp::int32 & adj_naconmax_in,
    Geom_3242f8a8 & adj_sphere1,
    Geom_3242f8a8 & adj_sphere2,
    wp::int32 & adj_worldid,
    wp::float32 & adj_margin,
    wp::float32 & adj_gap,
    wp::int32 & adj_condim,
    wp::vec_t<5, wp::float32> & adj_friction,
    wp::vec_t<2, wp::float32> & adj_solref,
    wp::vec_t<2, wp::float32> & adj_solreffriction,
    wp::vec_t<5, wp::float32> & adj_solimp,
    wp::vec_t<2, wp::int32> & adj_geoms,
    wp::vec_t<2, wp::int32> & adj_pairid,
    wp::array_t<wp::float32> & adj_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_contact_frame_out,
    wp::array_t<wp::float32> & adj_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_solimp_out,
    wp::array_t<wp::int32> & adj_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_contact_geom_out,
    wp::array_t<wp::int32> & adj_contact_efc_address_out,
    wp::array_t<wp::int32> & adj_contact_worldid_out,
    wp::array_t<wp::int32> & adj_contact_type_out,
    wp::array_t<wp::int32> & adj_contact_geomcollisionid_out,
    wp::array_t<wp::int32> & adj_nacon_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_primitive_core.py:40
static CUDA_CALLABLE void adj_closest_segment_point_1(
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> var_b,
    wp::vec_t<3, wp::float32> var_pt,
    wp::vec_t<3, wp::float32> & adj_a,
    wp::vec_t<3, wp::float32> & adj_b,
    wp::vec_t<3, wp::float32> & adj_pt,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_primitive_core.py:146
static CUDA_CALLABLE void adj_sphere_capsule_0(
    wp::vec_t<3, wp::float32> var_sphere_pos,
    wp::float32 var_sphere_radius,
    wp::vec_t<3, wp::float32> var_capsule_pos,
    wp::vec_t<3, wp::float32> var_capsule_axis,
    wp::float32 var_capsule_radius,
    wp::float32 var_capsule_half_length,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_sphere_pos,
    wp::float32 & adj_sphere_radius,
    wp::vec_t<3, wp::float32> & adj_capsule_pos,
    wp::vec_t<3, wp::float32> & adj_capsule_axis,
    wp::float32 & adj_capsule_radius,
    wp::float32 & adj_capsule_half_length,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_primitive.py:423
static CUDA_CALLABLE void adj_sphere_capsule_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_3242f8a8 var_sphere,
    Geom_3242f8a8 var_cap,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
    wp::array_t<wp::float32> var_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_out,
    wp::array_t<wp::float32> var_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_solimp_out,
    wp::array_t<wp::int32> var_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::int32> var_nacon_out,
    wp::int32 & adj_naconmax_in,
    Geom_3242f8a8 & adj_sphere,
    Geom_3242f8a8 & adj_cap,
    wp::int32 & adj_worldid,
    wp::float32 & adj_margin,
    wp::float32 & adj_gap,
    wp::int32 & adj_condim,
    wp::vec_t<5, wp::float32> & adj_friction,
    wp::vec_t<2, wp::float32> & adj_solref,
    wp::vec_t<2, wp::float32> & adj_solreffriction,
    wp::vec_t<5, wp::float32> & adj_solimp,
    wp::vec_t<2, wp::int32> & adj_geoms,
    wp::vec_t<2, wp::int32> & adj_pairid,
    wp::array_t<wp::float32> & adj_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_contact_frame_out,
    wp::array_t<wp::float32> & adj_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_solimp_out,
    wp::array_t<wp::int32> & adj_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_contact_geom_out,
    wp::array_t<wp::int32> & adj_contact_efc_address_out,
    wp::array_t<wp::int32> & adj_contact_worldid_out,
    wp::array_t<wp::int32> & adj_contact_type_out,
    wp::array_t<wp::int32> & adj_contact_geomcollisionid_out,
    wp::array_t<wp::int32> & adj_nacon_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_primitive_core.py:106
static CUDA_CALLABLE void adj_plane_sphere_0(
    wp::vec_t<3, wp::float32> var_plane_normal,
    wp::vec_t<3, wp::float32> var_plane_pos,
    wp::vec_t<3, wp::float32> var_sphere_pos,
    wp::float32 var_sphere_radius,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_plane_normal,
    wp::vec_t<3, wp::float32> & adj_plane_pos,
    wp::vec_t<3, wp::float32> & adj_sphere_pos,
    wp::float32 & adj_sphere_radius,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_primitive_core.py:0
static CUDA_CALLABLE void adj_safe_div_1(
    wp::float32 var_x,
    wp::float32 var_y,
    wp::float32 & adj_x,
    wp::float32 & adj_y,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_primitive_core.py:446
static CUDA_CALLABLE void adj_sphere_cylinder_0(
    wp::vec_t<3, wp::float32> var_sphere_pos,
    wp::float32 var_sphere_radius,
    wp::vec_t<3, wp::float32> var_cylinder_pos,
    wp::vec_t<3, wp::float32> var_cylinder_axis,
    wp::float32 var_cylinder_radius,
    wp::float32 var_cylinder_half_height,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_sphere_pos,
    wp::float32 & adj_sphere_radius,
    wp::vec_t<3, wp::float32> & adj_cylinder_pos,
    wp::vec_t<3, wp::float32> & adj_cylinder_axis,
    wp::float32 & adj_cylinder_radius,
    wp::float32 & adj_cylinder_half_height,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_primitive.py:882
static CUDA_CALLABLE void adj_sphere_cylinder_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_3242f8a8 var_sphere,
    Geom_3242f8a8 var_cylinder,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
    wp::array_t<wp::float32> var_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_out,
    wp::array_t<wp::float32> var_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_solimp_out,
    wp::array_t<wp::int32> var_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::int32> var_nacon_out,
    wp::int32 & adj_naconmax_in,
    Geom_3242f8a8 & adj_sphere,
    Geom_3242f8a8 & adj_cylinder,
    wp::int32 & adj_worldid,
    wp::float32 & adj_margin,
    wp::float32 & adj_gap,
    wp::int32 & adj_condim,
    wp::vec_t<5, wp::float32> & adj_friction,
    wp::vec_t<2, wp::float32> & adj_solref,
    wp::vec_t<2, wp::float32> & adj_solreffriction,
    wp::vec_t<5, wp::float32> & adj_solimp,
    wp::vec_t<2, wp::int32> & adj_geoms,
    wp::vec_t<2, wp::int32> & adj_pairid,
    wp::array_t<wp::float32> & adj_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_contact_frame_out,
    wp::array_t<wp::float32> & adj_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_solimp_out,
    wp::array_t<wp::int32> & adj_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_contact_geom_out,
    wp::array_t<wp::int32> & adj_contact_efc_address_out,
    wp::array_t<wp::int32> & adj_contact_worldid_out,
    wp::array_t<wp::int32> & adj_contact_type_out,
    wp::array_t<wp::int32> & adj_contact_geomcollisionid_out,
    wp::array_t<wp::int32> & adj_nacon_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_primitive_core.py:0
static CUDA_CALLABLE void adj_normalize_with_norm_1(
    wp::vec_t<3, wp::float32> var_x,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::float32 & ret_1,
    wp::vec_t<3, wp::float32> & adj_x,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::float32 & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_primitive_core.py:1102
static CUDA_CALLABLE void adj_sphere_box_0(
    wp::vec_t<3, wp::float32> var_sphere_pos,
    wp::float32 var_sphere_radius,
    wp::vec_t<3, wp::float32> var_box_pos,
    wp::mat_t<3, 3, wp::float32> var_box_rot,
    wp::vec_t<3, wp::float32> var_box_size,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_sphere_pos,
    wp::float32 & adj_sphere_radius,
    wp::vec_t<3, wp::float32> & adj_box_pos,
    wp::mat_t<3, 3, wp::float32> & adj_box_rot,
    wp::vec_t<3, wp::float32> & adj_box_size,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_primitive.py:1046
static CUDA_CALLABLE void adj_sphere_box_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_3242f8a8 var_sphere,
    Geom_3242f8a8 var_box,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
    wp::array_t<wp::float32> var_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_out,
    wp::array_t<wp::float32> var_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_solimp_out,
    wp::array_t<wp::int32> var_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::int32> var_nacon_out,
    wp::int32 & adj_naconmax_in,
    Geom_3242f8a8 & adj_sphere,
    Geom_3242f8a8 & adj_box,
    wp::int32 & adj_worldid,
    wp::float32 & adj_margin,
    wp::float32 & adj_gap,
    wp::int32 & adj_condim,
    wp::vec_t<5, wp::float32> & adj_friction,
    wp::vec_t<2, wp::float32> & adj_solref,
    wp::vec_t<2, wp::float32> & adj_solreffriction,
    wp::vec_t<5, wp::float32> & adj_solimp,
    wp::vec_t<2, wp::int32> & adj_geoms,
    wp::vec_t<2, wp::int32> & adj_pairid,
    wp::array_t<wp::float32> & adj_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_contact_frame_out,
    wp::array_t<wp::float32> & adj_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_solimp_out,
    wp::array_t<wp::int32> & adj_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_contact_geom_out,
    wp::array_t<wp::int32> & adj_contact_efc_address_out,
    wp::array_t<wp::int32> & adj_contact_worldid_out,
    wp::array_t<wp::int32> & adj_contact_type_out,
    wp::array_t<wp::int32> & adj_contact_geomcollisionid_out,
    wp::array_t<wp::int32> & adj_nacon_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_primitive_core.py:181
static CUDA_CALLABLE void adj_capsule_capsule_0(
    wp::vec_t<3, wp::float32> var_cap1_pos,
    wp::vec_t<3, wp::float32> var_cap1_axis,
    wp::float32 var_cap1_radius,
    wp::float32 var_cap1_half_length,
    wp::vec_t<3, wp::float32> var_cap2_pos,
    wp::vec_t<3, wp::float32> var_cap2_axis,
    wp::float32 var_cap2_radius,
    wp::float32 var_cap2_half_length,
    wp::float32 var_margin,
    wp::vec_t<2, wp::float32> & ret_0,
    wp::mat_t<2, 3, wp::float32> & ret_1,
    wp::mat_t<2, 3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_cap1_pos,
    wp::vec_t<3, wp::float32> & adj_cap1_axis,
    wp::float32 & adj_cap1_radius,
    wp::float32 & adj_cap1_half_length,
    wp::vec_t<3, wp::float32> & adj_cap2_pos,
    wp::vec_t<3, wp::float32> & adj_cap2_axis,
    wp::float32 & adj_cap2_radius,
    wp::float32 & adj_cap2_half_length,
    wp::float32 & adj_margin,
    wp::vec_t<2, wp::float32> & adj_ret_0,
    wp::mat_t<2, 3, wp::float32> & adj_ret_1,
    wp::mat_t<2, 3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_primitive.py:497
static CUDA_CALLABLE void adj_capsule_capsule_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_3242f8a8 var_cap1,
    Geom_3242f8a8 var_cap2,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
    wp::array_t<wp::float32> var_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_out,
    wp::array_t<wp::float32> var_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_solimp_out,
    wp::array_t<wp::int32> var_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::int32> var_nacon_out,
    wp::int32 & adj_naconmax_in,
    Geom_3242f8a8 & adj_cap1,
    Geom_3242f8a8 & adj_cap2,
    wp::int32 & adj_worldid,
    wp::float32 & adj_margin,
    wp::float32 & adj_gap,
    wp::int32 & adj_condim,
    wp::vec_t<5, wp::float32> & adj_friction,
    wp::vec_t<2, wp::float32> & adj_solref,
    wp::vec_t<2, wp::float32> & adj_solreffriction,
    wp::vec_t<5, wp::float32> & adj_solimp,
    wp::vec_t<2, wp::int32> & adj_geoms,
    wp::vec_t<2, wp::int32> & adj_pairid,
    wp::array_t<wp::float32> & adj_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_contact_frame_out,
    wp::array_t<wp::float32> & adj_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_solimp_out,
    wp::array_t<wp::int32> & adj_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_contact_geom_out,
    wp::array_t<wp::int32> & adj_contact_efc_address_out,
    wp::array_t<wp::int32> & adj_contact_worldid_out,
    wp::array_t<wp::int32> & adj_contact_type_out,
    wp::array_t<wp::int32> & adj_contact_geomcollisionid_out,
    wp::array_t<wp::int32> & adj_nacon_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_primitive_core.py:1157
static CUDA_CALLABLE void adj_capsule_box_0(
    wp::vec_t<3, wp::float32> var_capsule_pos,
    wp::vec_t<3, wp::float32> var_capsule_axis,
    wp::float32 var_capsule_radius,
    wp::float32 var_capsule_half_length,
    wp::vec_t<3, wp::float32> var_box_pos,
    wp::mat_t<3, 3, wp::float32> var_box_rot,
    wp::vec_t<3, wp::float32> var_box_size,
    wp::vec_t<2, wp::float32> & ret_0,
    wp::mat_t<2, 3, wp::float32> & ret_1,
    wp::mat_t<2, 3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_capsule_pos,
    wp::vec_t<3, wp::float32> & adj_capsule_axis,
    wp::float32 & adj_capsule_radius,
    wp::float32 & adj_capsule_half_length,
    wp::vec_t<3, wp::float32> & adj_box_pos,
    wp::mat_t<3, 3, wp::float32> & adj_box_rot,
    wp::vec_t<3, wp::float32> & adj_box_size,
    wp::vec_t<2, wp::float32> & adj_ret_0,
    wp::mat_t<2, 3, wp::float32> & adj_ret_1,
    wp::mat_t<2, 3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_primitive.py:1116
static CUDA_CALLABLE void adj_capsule_box_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_3242f8a8 var_cap,
    Geom_3242f8a8 var_box,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
    wp::array_t<wp::float32> var_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_out,
    wp::array_t<wp::float32> var_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_solimp_out,
    wp::array_t<wp::int32> var_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::int32> var_nacon_out,
    wp::int32 & adj_naconmax_in,
    Geom_3242f8a8 & adj_cap,
    Geom_3242f8a8 & adj_box,
    wp::int32 & adj_worldid,
    wp::float32 & adj_margin,
    wp::float32 & adj_gap,
    wp::int32 & adj_condim,
    wp::vec_t<5, wp::float32> & adj_friction,
    wp::vec_t<2, wp::float32> & adj_solref,
    wp::vec_t<2, wp::float32> & adj_solreffriction,
    wp::vec_t<5, wp::float32> & adj_solimp,
    wp::vec_t<2, wp::int32> & adj_geoms,
    wp::vec_t<2, wp::int32> & adj_pairid,
    wp::array_t<wp::float32> & adj_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_contact_frame_out,
    wp::array_t<wp::float32> & adj_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_solimp_out,
    wp::array_t<wp::int32> & adj_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_contact_geom_out,
    wp::array_t<wp::int32> & adj_contact_efc_address_out,
    wp::array_t<wp::int32> & adj_contact_worldid_out,
    wp::array_t<wp::int32> & adj_contact_type_out,
    wp::array_t<wp::int32> & adj_contact_geomcollisionid_out,
    wp::array_t<wp::int32> & adj_nacon_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void _primitive_narrowphase__locals__primitive_narrowphase_1e795a93_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::int32> var_geom_condim,
    wp::array_t<wp::int32> var_geom_dataid,
    wp::array_t<wp::int32> var_geom_priority,
    wp::array_t<wp::float32> var_geom_solmix,
    wp::array_t<wp::vec_t<2, wp::float32>> var_geom_solref,
    wp::array_t<wp::vec_t<5, wp::float32>> var_geom_solimp,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_friction,
    wp::array_t<wp::float32> var_geom_margin,
    wp::array_t<wp::float32> var_geom_gap,
    wp::array_t<wp::int32> var_mesh_vertadr,
    wp::array_t<wp::int32> var_mesh_vertnum,
    wp::array_t<wp::int32> var_mesh_graphadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_vert,
    wp::array_t<wp::int32> var_mesh_graph,
    wp::array_t<wp::int32> var_mesh_polynum,
    wp::array_t<wp::int32> var_mesh_polyadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_polynormal,
    wp::array_t<wp::int32> var_mesh_polyvertadr,
    wp::array_t<wp::int32> var_mesh_polyvertnum,
    wp::array_t<wp::int32> var_mesh_polyvert,
    wp::array_t<wp::int32> var_mesh_polymapadr,
    wp::array_t<wp::int32> var_mesh_polymapnum,
    wp::array_t<wp::int32> var_mesh_polymap,
    wp::array_t<wp::int32> var_pair_dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_pair_solref,
    wp::array_t<wp::vec_t<2, wp::float32>> var_pair_solreffriction,
    wp::array_t<wp::vec_t<5, wp::float32>> var_pair_solimp,
    wp::array_t<wp::float32> var_pair_margin,
    wp::array_t<wp::float32> var_pair_gap,
    wp::array_t<wp::vec_t<5, wp::float32>> var_pair_friction,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::int32 var_naconmax_in,
    wp::array_t<wp::int32> var_ncollision_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pair_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pairid_in,
    wp::array_t<wp::int32> var_collision_worldid_in,
    wp::array_t<wp::float32> var_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_out,
    wp::array_t<wp::float32> var_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_solimp_out,
    wp::array_t<wp::int32> var_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::int32> var_nacon_out)
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
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::vec_t<2, wp::int32> var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::int32 var_14;
        wp::vec_t<5, wp::float32> var_15;
        wp::vec_t<2, wp::float32> var_16;
        wp::vec_t<2, wp::float32> var_17;
        wp::vec_t<5, wp::float32> var_18;
        Geom_3242f8a8 var_19;
        Geom_3242f8a8 var_20;
        const wp::int32 var_21 = 0;
        const wp::int32 var_22 = 2;
        const wp::int32 var_23 = 2;
        const wp::int32 var_24 = 0;
        wp::int32 var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        const wp::int32 var_29 = 1;
        wp::int32 var_30;
        wp::int32* var_31;
        wp::int32 var_32;
        wp::int32 var_33;
        bool var_34;
        bool var_35;
        bool var_36;
        wp::vec_t<2, wp::int32>* var_37;
        wp::vec_t<2, wp::int32> var_38;
        const wp::int32 var_39 = 1;
        const wp::int32 var_40 = 2;
        const wp::int32 var_41 = 3;
        const wp::int32 var_42 = 0;
        wp::int32 var_43;
        wp::int32* var_44;
        wp::int32 var_45;
        wp::int32 var_46;
        const wp::int32 var_47 = 1;
        wp::int32 var_48;
        wp::int32* var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        bool var_52;
        bool var_53;
        bool var_54;
        wp::vec_t<2, wp::int32>* var_55;
        wp::vec_t<2, wp::int32> var_56;
        const wp::int32 var_57 = 2;
        const wp::int32 var_58 = 2;
        const wp::int32 var_59 = 5;
        const wp::int32 var_60 = 0;
        wp::int32 var_61;
        wp::int32* var_62;
        wp::int32 var_63;
        wp::int32 var_64;
        const wp::int32 var_65 = 1;
        wp::int32 var_66;
        wp::int32* var_67;
        wp::int32 var_68;
        wp::int32 var_69;
        bool var_70;
        bool var_71;
        bool var_72;
        wp::vec_t<2, wp::int32>* var_73;
        wp::vec_t<2, wp::int32> var_74;
        const wp::int32 var_75 = 3;
        const wp::int32 var_76 = 2;
        const wp::int32 var_77 = 6;
        const wp::int32 var_78 = 0;
        wp::int32 var_79;
        wp::int32* var_80;
        wp::int32 var_81;
        wp::int32 var_82;
        const wp::int32 var_83 = 1;
        wp::int32 var_84;
        wp::int32* var_85;
        wp::int32 var_86;
        wp::int32 var_87;
        bool var_88;
        bool var_89;
        bool var_90;
        wp::vec_t<2, wp::int32>* var_91;
        wp::vec_t<2, wp::int32> var_92;
        const wp::int32 var_93 = 4;
        const wp::int32 var_94 = 3;
        const wp::int32 var_95 = 3;
        const wp::int32 var_96 = 0;
        wp::int32 var_97;
        wp::int32* var_98;
        wp::int32 var_99;
        wp::int32 var_100;
        const wp::int32 var_101 = 1;
        wp::int32 var_102;
        wp::int32* var_103;
        wp::int32 var_104;
        wp::int32 var_105;
        bool var_106;
        bool var_107;
        bool var_108;
        wp::vec_t<2, wp::int32>* var_109;
        wp::vec_t<2, wp::int32> var_110;
        const wp::int32 var_111 = 5;
        const wp::int32 var_112 = 3;
        const wp::int32 var_113 = 6;
        const wp::int32 var_114 = 0;
        wp::int32 var_115;
        wp::int32* var_116;
        wp::int32 var_117;
        wp::int32 var_118;
        const wp::int32 var_119 = 1;
        wp::int32 var_120;
        wp::int32* var_121;
        wp::int32 var_122;
        wp::int32 var_123;
        bool var_124;
        bool var_125;
        bool var_126;
        wp::vec_t<2, wp::int32>* var_127;
        wp::vec_t<2, wp::int32> var_128;
        //---------
        // forward
        // def primitive_narrowphase(                                                             <L 1303>
        // tid = wp.tid()                                                                         <L 1363>
        var_0 = builtin_tid1d();
        // if tid >= ncollision_in[0]:                                                            <L 1365>
        var_2 = wp::address(var_ncollision_in, var_1);
        var_4 = wp::load(var_2);
        var_3 = (var_0 >= var_4);
        if (var_3) {
            // return                                                                             <L 1366>
            continue;
        }
        // geoms = collision_pair_in[tid]                                                         <L 1368>
        var_5 = wp::address(var_collision_pair_in, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // worldid = collision_worldid_in[tid]                                                    <L 1369>
        var_8 = wp::address(var_collision_worldid_in, var_0);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // _, margin, gap, condim, friction, solref, solreffriction, solimp = contact_params(       <L 1371>
        // geom_condim,                                                                           <L 1372>
        // geom_priority,                                                                         <L 1373>
        // geom_solmix,                                                                           <L 1374>
        // geom_solref,                                                                           <L 1375>
        // geom_solimp,                                                                           <L 1376>
        // geom_friction,                                                                         <L 1377>
        // geom_margin,                                                                           <L 1378>
        // geom_gap,                                                                              <L 1379>
        // pair_dim,                                                                              <L 1380>
        // pair_solref,                                                                           <L 1381>
        // pair_solreffriction,                                                                   <L 1382>
        // pair_solimp,                                                                           <L 1383>
        // pair_margin,                                                                           <L 1384>
        // pair_gap,                                                                              <L 1385>
        // pair_friction,                                                                         <L 1386>
        // collision_pair_in,                                                                     <L 1387>
        // collision_pairid_in,                                                                   <L 1388>
        // tid,                                                                                   <L 1389>
        // worldid,                                                                               <L 1390>
        contact_params_0(var_geom_condim, var_geom_priority, var_geom_solmix, var_geom_solref, var_geom_solimp, var_geom_friction, var_geom_margin, var_geom_gap, var_pair_dim, var_pair_solref, var_pair_solreffriction, var_pair_solimp, var_pair_margin, var_pair_gap, var_pair_friction, var_collision_pair_in, var_collision_pairid_in, var_0, var_9, var_11, var_12, var_13, var_14, var_15, var_16, var_17, var_18);
        // geom1, geom2 = geom_collision_pair(                                                    <L 1393>
        // geom_type,                                                                             <L 1394>
        // geom_dataid,                                                                           <L 1395>
        // geom_size,                                                                             <L 1396>
        // mesh_vertadr,                                                                          <L 1397>
        // mesh_vertnum,                                                                          <L 1398>
        // mesh_graphadr,                                                                         <L 1399>
        // mesh_vert,                                                                             <L 1400>
        // mesh_graph,                                                                            <L 1401>
        // mesh_polynum,                                                                          <L 1402>
        // mesh_polyadr,                                                                          <L 1403>
        // mesh_polynormal,                                                                       <L 1404>
        // mesh_polyvertadr,                                                                      <L 1405>
        // mesh_polyvertnum,                                                                      <L 1406>
        // mesh_polyvert,                                                                         <L 1407>
        // mesh_polymapadr,                                                                       <L 1408>
        // mesh_polymapnum,                                                                       <L 1409>
        // mesh_polymap,                                                                          <L 1410>
        // geom_xpos_in,                                                                          <L 1411>
        // geom_xmat_in,                                                                          <L 1412>
        // geoms,                                                                                 <L 1413>
        // worldid,                                                                               <L 1414>
        geom_collision_pair_0(var_geom_type, var_geom_dataid, var_geom_size, var_mesh_vertadr, var_mesh_vertnum, var_mesh_graphadr, var_mesh_vert, var_mesh_graph, var_mesh_polynum, var_mesh_polyadr, var_mesh_polynormal, var_mesh_polyvertadr, var_mesh_polyvertnum, var_mesh_polyvert, var_mesh_polymapadr, var_mesh_polymapnum, var_mesh_polymap, var_geom_xpos_in, var_geom_xmat_in, var_6, var_9, var_19, var_20);
        // for i in range(wp.static(len(primitive_collisions_func))):                             <L 1417>
        // collision_type1 = wp.static(primitive_collisions_types[i][0])                          <L 1418>
        // collision_type2 = wp.static(primitive_collisions_types[i][1])                          <L 1419>
        // type1 = geom_type[geoms[0]]                                                            <L 1420>
        var_25 = wp::extract(var_6, var_24);
        var_26 = wp::address(var_geom_type, var_25);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // type2 = geom_type[geoms[1]]                                                            <L 1421>
        var_30 = wp::extract(var_6, var_29);
        var_31 = wp::address(var_geom_type, var_30);
        var_33 = wp::load(var_31);
        var_32 = wp::copy(var_33);
        // if collision_type1 == type1 and collision_type2 == type2:                              <L 1422>
        var_34 = (var_22 == var_27);
        var_35 = (var_23 == var_32);
        var_36 = var_34 && var_35;
        if (var_36) {
            // wp.static(primitive_collisions_func[i])(                                           <L 1423>
            // naconmax_in,                                                                       <L 1424>
            // geom1,                                                                             <L 1425>
            // geom2,                                                                             <L 1426>
            // worldid,                                                                           <L 1427>
            // margin,                                                                            <L 1428>
            // gap,                                                                               <L 1429>
            // condim,                                                                            <L 1430>
            // friction,                                                                          <L 1431>
            // solref,                                                                            <L 1432>
            // solreffriction,                                                                    <L 1433>
            // solimp,                                                                            <L 1434>
            // geoms,                                                                             <L 1435>
            // collision_pairid_in[tid],                                                          <L 1436>
            var_37 = wp::address(var_collision_pairid_in, var_0);
            // contact_dist_out,                                                                  <L 1437>
            // contact_pos_out,                                                                   <L 1438>
            // contact_frame_out,                                                                 <L 1439>
            // contact_includemargin_out,                                                         <L 1440>
            // contact_friction_out,                                                              <L 1441>
            // contact_solref_out,                                                                <L 1442>
            // contact_solreffriction_out,                                                        <L 1443>
            // contact_solimp_out,                                                                <L 1444>
            // contact_dim_out,                                                                   <L 1445>
            // contact_geom_out,                                                                  <L 1446>
            // contact_efc_address_out,                                                           <L 1447>
            // contact_worldid_out,                                                               <L 1448>
            // contact_type_out,                                                                  <L 1449>
            // contact_geomcollisionid_out,                                                       <L 1450>
            // nacon_out,                                                                         <L 1451>
            var_38 = wp::load(var_37);
            sphere_sphere_wrapper_0(var_naconmax_in, var_19, var_20, var_9, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_6, var_38, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
        }
        // collision_type1 = wp.static(primitive_collisions_types[i][0])                          <L 1418>
        // collision_type2 = wp.static(primitive_collisions_types[i][1])                          <L 1419>
        // type1 = geom_type[geoms[0]]                                                            <L 1420>
        var_43 = wp::extract(var_6, var_42);
        var_44 = wp::address(var_geom_type, var_43);
        var_46 = wp::load(var_44);
        var_45 = wp::copy(var_46);
        // type2 = geom_type[geoms[1]]                                                            <L 1421>
        var_48 = wp::extract(var_6, var_47);
        var_49 = wp::address(var_geom_type, var_48);
        var_51 = wp::load(var_49);
        var_50 = wp::copy(var_51);
        // if collision_type1 == type1 and collision_type2 == type2:                              <L 1422>
        var_52 = (var_40 == var_45);
        var_53 = (var_41 == var_50);
        var_54 = var_52 && var_53;
        if (var_54) {
            // wp.static(primitive_collisions_func[i])(                                           <L 1423>
            // naconmax_in,                                                                       <L 1424>
            // geom1,                                                                             <L 1425>
            // geom2,                                                                             <L 1426>
            // worldid,                                                                           <L 1427>
            // margin,                                                                            <L 1428>
            // gap,                                                                               <L 1429>
            // condim,                                                                            <L 1430>
            // friction,                                                                          <L 1431>
            // solref,                                                                            <L 1432>
            // solreffriction,                                                                    <L 1433>
            // solimp,                                                                            <L 1434>
            // geoms,                                                                             <L 1435>
            // collision_pairid_in[tid],                                                          <L 1436>
            var_55 = wp::address(var_collision_pairid_in, var_0);
            // contact_dist_out,                                                                  <L 1437>
            // contact_pos_out,                                                                   <L 1438>
            // contact_frame_out,                                                                 <L 1439>
            // contact_includemargin_out,                                                         <L 1440>
            // contact_friction_out,                                                              <L 1441>
            // contact_solref_out,                                                                <L 1442>
            // contact_solreffriction_out,                                                        <L 1443>
            // contact_solimp_out,                                                                <L 1444>
            // contact_dim_out,                                                                   <L 1445>
            // contact_geom_out,                                                                  <L 1446>
            // contact_efc_address_out,                                                           <L 1447>
            // contact_worldid_out,                                                               <L 1448>
            // contact_type_out,                                                                  <L 1449>
            // contact_geomcollisionid_out,                                                       <L 1450>
            // nacon_out,                                                                         <L 1451>
            var_56 = wp::load(var_55);
            sphere_capsule_wrapper_0(var_naconmax_in, var_19, var_20, var_9, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_6, var_56, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
        }
        // collision_type1 = wp.static(primitive_collisions_types[i][0])                          <L 1418>
        // collision_type2 = wp.static(primitive_collisions_types[i][1])                          <L 1419>
        // type1 = geom_type[geoms[0]]                                                            <L 1420>
        var_61 = wp::extract(var_6, var_60);
        var_62 = wp::address(var_geom_type, var_61);
        var_64 = wp::load(var_62);
        var_63 = wp::copy(var_64);
        // type2 = geom_type[geoms[1]]                                                            <L 1421>
        var_66 = wp::extract(var_6, var_65);
        var_67 = wp::address(var_geom_type, var_66);
        var_69 = wp::load(var_67);
        var_68 = wp::copy(var_69);
        // if collision_type1 == type1 and collision_type2 == type2:                              <L 1422>
        var_70 = (var_58 == var_63);
        var_71 = (var_59 == var_68);
        var_72 = var_70 && var_71;
        if (var_72) {
            // wp.static(primitive_collisions_func[i])(                                           <L 1423>
            // naconmax_in,                                                                       <L 1424>
            // geom1,                                                                             <L 1425>
            // geom2,                                                                             <L 1426>
            // worldid,                                                                           <L 1427>
            // margin,                                                                            <L 1428>
            // gap,                                                                               <L 1429>
            // condim,                                                                            <L 1430>
            // friction,                                                                          <L 1431>
            // solref,                                                                            <L 1432>
            // solreffriction,                                                                    <L 1433>
            // solimp,                                                                            <L 1434>
            // geoms,                                                                             <L 1435>
            // collision_pairid_in[tid],                                                          <L 1436>
            var_73 = wp::address(var_collision_pairid_in, var_0);
            // contact_dist_out,                                                                  <L 1437>
            // contact_pos_out,                                                                   <L 1438>
            // contact_frame_out,                                                                 <L 1439>
            // contact_includemargin_out,                                                         <L 1440>
            // contact_friction_out,                                                              <L 1441>
            // contact_solref_out,                                                                <L 1442>
            // contact_solreffriction_out,                                                        <L 1443>
            // contact_solimp_out,                                                                <L 1444>
            // contact_dim_out,                                                                   <L 1445>
            // contact_geom_out,                                                                  <L 1446>
            // contact_efc_address_out,                                                           <L 1447>
            // contact_worldid_out,                                                               <L 1448>
            // contact_type_out,                                                                  <L 1449>
            // contact_geomcollisionid_out,                                                       <L 1450>
            // nacon_out,                                                                         <L 1451>
            var_74 = wp::load(var_73);
            sphere_cylinder_wrapper_0(var_naconmax_in, var_19, var_20, var_9, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_6, var_74, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
        }
        // collision_type1 = wp.static(primitive_collisions_types[i][0])                          <L 1418>
        // collision_type2 = wp.static(primitive_collisions_types[i][1])                          <L 1419>
        // type1 = geom_type[geoms[0]]                                                            <L 1420>
        var_79 = wp::extract(var_6, var_78);
        var_80 = wp::address(var_geom_type, var_79);
        var_82 = wp::load(var_80);
        var_81 = wp::copy(var_82);
        // type2 = geom_type[geoms[1]]                                                            <L 1421>
        var_84 = wp::extract(var_6, var_83);
        var_85 = wp::address(var_geom_type, var_84);
        var_87 = wp::load(var_85);
        var_86 = wp::copy(var_87);
        // if collision_type1 == type1 and collision_type2 == type2:                              <L 1422>
        var_88 = (var_76 == var_81);
        var_89 = (var_77 == var_86);
        var_90 = var_88 && var_89;
        if (var_90) {
            // wp.static(primitive_collisions_func[i])(                                           <L 1423>
            // naconmax_in,                                                                       <L 1424>
            // geom1,                                                                             <L 1425>
            // geom2,                                                                             <L 1426>
            // worldid,                                                                           <L 1427>
            // margin,                                                                            <L 1428>
            // gap,                                                                               <L 1429>
            // condim,                                                                            <L 1430>
            // friction,                                                                          <L 1431>
            // solref,                                                                            <L 1432>
            // solreffriction,                                                                    <L 1433>
            // solimp,                                                                            <L 1434>
            // geoms,                                                                             <L 1435>
            // collision_pairid_in[tid],                                                          <L 1436>
            var_91 = wp::address(var_collision_pairid_in, var_0);
            // contact_dist_out,                                                                  <L 1437>
            // contact_pos_out,                                                                   <L 1438>
            // contact_frame_out,                                                                 <L 1439>
            // contact_includemargin_out,                                                         <L 1440>
            // contact_friction_out,                                                              <L 1441>
            // contact_solref_out,                                                                <L 1442>
            // contact_solreffriction_out,                                                        <L 1443>
            // contact_solimp_out,                                                                <L 1444>
            // contact_dim_out,                                                                   <L 1445>
            // contact_geom_out,                                                                  <L 1446>
            // contact_efc_address_out,                                                           <L 1447>
            // contact_worldid_out,                                                               <L 1448>
            // contact_type_out,                                                                  <L 1449>
            // contact_geomcollisionid_out,                                                       <L 1450>
            // nacon_out,                                                                         <L 1451>
            var_92 = wp::load(var_91);
            sphere_box_wrapper_0(var_naconmax_in, var_19, var_20, var_9, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_6, var_92, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
        }
        // collision_type1 = wp.static(primitive_collisions_types[i][0])                          <L 1418>
        // collision_type2 = wp.static(primitive_collisions_types[i][1])                          <L 1419>
        // type1 = geom_type[geoms[0]]                                                            <L 1420>
        var_97 = wp::extract(var_6, var_96);
        var_98 = wp::address(var_geom_type, var_97);
        var_100 = wp::load(var_98);
        var_99 = wp::copy(var_100);
        // type2 = geom_type[geoms[1]]                                                            <L 1421>
        var_102 = wp::extract(var_6, var_101);
        var_103 = wp::address(var_geom_type, var_102);
        var_105 = wp::load(var_103);
        var_104 = wp::copy(var_105);
        // if collision_type1 == type1 and collision_type2 == type2:                              <L 1422>
        var_106 = (var_94 == var_99);
        var_107 = (var_95 == var_104);
        var_108 = var_106 && var_107;
        if (var_108) {
            // wp.static(primitive_collisions_func[i])(                                           <L 1423>
            // naconmax_in,                                                                       <L 1424>
            // geom1,                                                                             <L 1425>
            // geom2,                                                                             <L 1426>
            // worldid,                                                                           <L 1427>
            // margin,                                                                            <L 1428>
            // gap,                                                                               <L 1429>
            // condim,                                                                            <L 1430>
            // friction,                                                                          <L 1431>
            // solref,                                                                            <L 1432>
            // solreffriction,                                                                    <L 1433>
            // solimp,                                                                            <L 1434>
            // geoms,                                                                             <L 1435>
            // collision_pairid_in[tid],                                                          <L 1436>
            var_109 = wp::address(var_collision_pairid_in, var_0);
            // contact_dist_out,                                                                  <L 1437>
            // contact_pos_out,                                                                   <L 1438>
            // contact_frame_out,                                                                 <L 1439>
            // contact_includemargin_out,                                                         <L 1440>
            // contact_friction_out,                                                              <L 1441>
            // contact_solref_out,                                                                <L 1442>
            // contact_solreffriction_out,                                                        <L 1443>
            // contact_solimp_out,                                                                <L 1444>
            // contact_dim_out,                                                                   <L 1445>
            // contact_geom_out,                                                                  <L 1446>
            // contact_efc_address_out,                                                           <L 1447>
            // contact_worldid_out,                                                               <L 1448>
            // contact_type_out,                                                                  <L 1449>
            // contact_geomcollisionid_out,                                                       <L 1450>
            // nacon_out,                                                                         <L 1451>
            var_110 = wp::load(var_109);
            capsule_capsule_wrapper_0(var_naconmax_in, var_19, var_20, var_9, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_6, var_110, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
        }
        // collision_type1 = wp.static(primitive_collisions_types[i][0])                          <L 1418>
        // collision_type2 = wp.static(primitive_collisions_types[i][1])                          <L 1419>
        // type1 = geom_type[geoms[0]]                                                            <L 1420>
        var_115 = wp::extract(var_6, var_114);
        var_116 = wp::address(var_geom_type, var_115);
        var_118 = wp::load(var_116);
        var_117 = wp::copy(var_118);
        // type2 = geom_type[geoms[1]]                                                            <L 1421>
        var_120 = wp::extract(var_6, var_119);
        var_121 = wp::address(var_geom_type, var_120);
        var_123 = wp::load(var_121);
        var_122 = wp::copy(var_123);
        // if collision_type1 == type1 and collision_type2 == type2:                              <L 1422>
        var_124 = (var_112 == var_117);
        var_125 = (var_113 == var_122);
        var_126 = var_124 && var_125;
        if (var_126) {
            // wp.static(primitive_collisions_func[i])(                                           <L 1423>
            // naconmax_in,                                                                       <L 1424>
            // geom1,                                                                             <L 1425>
            // geom2,                                                                             <L 1426>
            // worldid,                                                                           <L 1427>
            // margin,                                                                            <L 1428>
            // gap,                                                                               <L 1429>
            // condim,                                                                            <L 1430>
            // friction,                                                                          <L 1431>
            // solref,                                                                            <L 1432>
            // solreffriction,                                                                    <L 1433>
            // solimp,                                                                            <L 1434>
            // geoms,                                                                             <L 1435>
            // collision_pairid_in[tid],                                                          <L 1436>
            var_127 = wp::address(var_collision_pairid_in, var_0);
            // contact_dist_out,                                                                  <L 1437>
            // contact_pos_out,                                                                   <L 1438>
            // contact_frame_out,                                                                 <L 1439>
            // contact_includemargin_out,                                                         <L 1440>
            // contact_friction_out,                                                              <L 1441>
            // contact_solref_out,                                                                <L 1442>
            // contact_solreffriction_out,                                                        <L 1443>
            // contact_solimp_out,                                                                <L 1444>
            // contact_dim_out,                                                                   <L 1445>
            // contact_geom_out,                                                                  <L 1446>
            // contact_efc_address_out,                                                           <L 1447>
            // contact_worldid_out,                                                               <L 1448>
            // contact_type_out,                                                                  <L 1449>
            // contact_geomcollisionid_out,                                                       <L 1450>
            // nacon_out,                                                                         <L 1451>
            var_128 = wp::load(var_127);
            capsule_box_wrapper_0(var_naconmax_in, var_19, var_20, var_9, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_6, var_128, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
        }
    }
}

