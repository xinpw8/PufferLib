
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



struct SupportPoint_e82efc60
{
    wp::vec_t<3, wp::float32> point;
    wp::int32 cached_index;
    wp::int32 vertex_index;


    SupportPoint_e82efc60() = default;
    CUDA_CALLABLE SupportPoint_e82efc60(wp::vec_t<3, wp::float32> const& point,
    wp::int32 const& cached_index = {},
    wp::int32 const& vertex_index = {})
        : point{point}
        , cached_index{cached_index}
        , vertex_index{vertex_index}

    {
    }

    CUDA_CALLABLE SupportPoint_e82efc60& operator += (const SupportPoint_e82efc60& rhs)
    {    point += rhs.point;
    cached_index += rhs.cached_index;
    vertex_index += rhs.vertex_index;

        return *this;}

};

static CUDA_CALLABLE void adj_SupportPoint_e82efc60(wp::vec_t<3, wp::float32> const&,
    wp::int32 const&,
    wp::int32 const&,
    wp::vec_t<3, wp::float32> & adj_point,
    wp::int32 & adj_cached_index,
    wp::int32 & adj_vertex_index,
    SupportPoint_e82efc60 & adj_ret)
{
    adj_point += adj_ret.point;
    adj_cached_index += adj_ret.cached_index;
    adj_vertex_index += adj_ret.vertex_index;
}

// Required when compiling adjoints.
CUDA_CALLABLE SupportPoint_e82efc60 add(const SupportPoint_e82efc60& a, const SupportPoint_e82efc60& b)
{
    return SupportPoint_e82efc60();
}

CUDA_CALLABLE void adj_atomic_add(SupportPoint_e82efc60* p, SupportPoint_e82efc60 t)
{
    wp::adj_atomic_add(&p->point, t.point);
    wp::adj_atomic_add(&p->cached_index, t.cached_index);
    wp::adj_atomic_add(&p->vertex_index, t.vertex_index);
}



struct GJKResult_0220ee01
{
    wp::float32 dist;
    wp::vec_t<3, wp::float32> x1;
    wp::vec_t<3, wp::float32> x2;
    wp::int32 dim;
    wp::mat_t<4, 3, wp::float32> simplex;
    wp::mat_t<4, 3, wp::float32> simplex1;
    wp::mat_t<4, 3, wp::float32> simplex2;
    wp::vec_t<4, wp::int32> simplex_index1;
    wp::vec_t<4, wp::int32> simplex_index2;


    GJKResult_0220ee01() = default;
    CUDA_CALLABLE GJKResult_0220ee01(wp::float32 const& dist,
    wp::vec_t<3, wp::float32> const& x1 = {},
    wp::vec_t<3, wp::float32> const& x2 = {},
    wp::int32 const& dim = {},
    wp::mat_t<4, 3, wp::float32> const& simplex = {},
    wp::mat_t<4, 3, wp::float32> const& simplex1 = {},
    wp::mat_t<4, 3, wp::float32> const& simplex2 = {},
    wp::vec_t<4, wp::int32> const& simplex_index1 = {},
    wp::vec_t<4, wp::int32> const& simplex_index2 = {})
        : dist{dist}
        , x1{x1}
        , x2{x2}
        , dim{dim}
        , simplex{simplex}
        , simplex1{simplex1}
        , simplex2{simplex2}
        , simplex_index1{simplex_index1}
        , simplex_index2{simplex_index2}

    {
    }

    CUDA_CALLABLE GJKResult_0220ee01& operator += (const GJKResult_0220ee01& rhs)
    {    dist += rhs.dist;
    x1 += rhs.x1;
    x2 += rhs.x2;
    dim += rhs.dim;
    simplex += rhs.simplex;
    simplex1 += rhs.simplex1;
    simplex2 += rhs.simplex2;
    simplex_index1 += rhs.simplex_index1;
    simplex_index2 += rhs.simplex_index2;

        return *this;}

};

static CUDA_CALLABLE void adj_GJKResult_0220ee01(wp::float32 const&,
    wp::vec_t<3, wp::float32> const&,
    wp::vec_t<3, wp::float32> const&,
    wp::int32 const&,
    wp::mat_t<4, 3, wp::float32> const&,
    wp::mat_t<4, 3, wp::float32> const&,
    wp::mat_t<4, 3, wp::float32> const&,
    wp::vec_t<4, wp::int32> const&,
    wp::vec_t<4, wp::int32> const&,
    wp::float32 & adj_dist,
    wp::vec_t<3, wp::float32> & adj_x1,
    wp::vec_t<3, wp::float32> & adj_x2,
    wp::int32 & adj_dim,
    wp::mat_t<4, 3, wp::float32> & adj_simplex,
    wp::mat_t<4, 3, wp::float32> & adj_simplex1,
    wp::mat_t<4, 3, wp::float32> & adj_simplex2,
    wp::vec_t<4, wp::int32> & adj_simplex_index1,
    wp::vec_t<4, wp::int32> & adj_simplex_index2,
    GJKResult_0220ee01 & adj_ret)
{
    adj_dist += adj_ret.dist;
    adj_x1 += adj_ret.x1;
    adj_x2 += adj_ret.x2;
    adj_dim += adj_ret.dim;
    adj_simplex += adj_ret.simplex;
    adj_simplex1 += adj_ret.simplex1;
    adj_simplex2 += adj_ret.simplex2;
    adj_simplex_index1 += adj_ret.simplex_index1;
    adj_simplex_index2 += adj_ret.simplex_index2;
}

// Required when compiling adjoints.
CUDA_CALLABLE GJKResult_0220ee01 add(const GJKResult_0220ee01& a, const GJKResult_0220ee01& b)
{
    return GJKResult_0220ee01();
}

CUDA_CALLABLE void adj_atomic_add(GJKResult_0220ee01* p, GJKResult_0220ee01 t)
{
    wp::adj_atomic_add(&p->dist, t.dist);
    wp::adj_atomic_add(&p->x1, t.x1);
    wp::adj_atomic_add(&p->x2, t.x2);
    wp::adj_atomic_add(&p->dim, t.dim);
    wp::adj_atomic_add(&p->simplex, t.simplex);
    wp::adj_atomic_add(&p->simplex1, t.simplex1);
    wp::adj_atomic_add(&p->simplex2, t.simplex2);
    wp::adj_atomic_add(&p->simplex_index1, t.simplex_index1);
    wp::adj_atomic_add(&p->simplex_index2, t.simplex_index2);
}



struct Polytope_9ab93ade
{
    wp::int32 status;
    wp::array_t<wp::vec_t<3, wp::float32>> vert;
    wp::array_t<wp::int32> vert_index;
    wp::int32 nvert;
    wp::array_t<wp::int32> face;
    wp::array_t<wp::vec_t<3, wp::float32>> face_pr;
    wp::array_t<wp::float32> face_norm2;
    wp::int32 nface;
    wp::array_t<wp::int32> horizon;
    wp::int32 nhorizon;


    Polytope_9ab93ade() = default;
    CUDA_CALLABLE Polytope_9ab93ade(wp::int32 const& status,
    wp::array_t<wp::vec_t<3, wp::float32>> const& vert = {},
    wp::array_t<wp::int32> const& vert_index = {},
    wp::int32 const& nvert = {},
    wp::array_t<wp::int32> const& face = {},
    wp::array_t<wp::vec_t<3, wp::float32>> const& face_pr = {},
    wp::array_t<wp::float32> const& face_norm2 = {},
    wp::int32 const& nface = {},
    wp::array_t<wp::int32> const& horizon = {},
    wp::int32 const& nhorizon = {})
        : status{status}
        , vert{vert}
        , vert_index{vert_index}
        , nvert{nvert}
        , face{face}
        , face_pr{face_pr}
        , face_norm2{face_norm2}
        , nface{nface}
        , horizon{horizon}
        , nhorizon{nhorizon}

    {
    }

    CUDA_CALLABLE Polytope_9ab93ade& operator += (const Polytope_9ab93ade& rhs)
    {    status += rhs.status;
    nvert += rhs.nvert;
    nface += rhs.nface;
    nhorizon += rhs.nhorizon;

        return *this;}

};

static CUDA_CALLABLE void adj_Polytope_9ab93ade(wp::int32 const&,
    wp::array_t<wp::vec_t<3, wp::float32>> const&,
    wp::array_t<wp::int32> const&,
    wp::int32 const&,
    wp::array_t<wp::int32> const&,
    wp::array_t<wp::vec_t<3, wp::float32>> const&,
    wp::array_t<wp::float32> const&,
    wp::int32 const&,
    wp::array_t<wp::int32> const&,
    wp::int32 const&,
    wp::int32 & adj_status,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_vert,
    wp::array_t<wp::int32> & adj_vert_index,
    wp::int32 & adj_nvert,
    wp::array_t<wp::int32> & adj_face,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_face_pr,
    wp::array_t<wp::float32> & adj_face_norm2,
    wp::int32 & adj_nface,
    wp::array_t<wp::int32> & adj_horizon,
    wp::int32 & adj_nhorizon,
    Polytope_9ab93ade & adj_ret)
{
    adj_status += adj_ret.status;
    adj_vert = adj_ret.vert;
    adj_vert_index = adj_ret.vert_index;
    adj_nvert += adj_ret.nvert;
    adj_face = adj_ret.face;
    adj_face_pr = adj_ret.face_pr;
    adj_face_norm2 = adj_ret.face_norm2;
    adj_nface += adj_ret.nface;
    adj_horizon = adj_ret.horizon;
    adj_nhorizon += adj_ret.nhorizon;
}

// Required when compiling adjoints.
CUDA_CALLABLE Polytope_9ab93ade add(const Polytope_9ab93ade& a, const Polytope_9ab93ade& b)
{
    return Polytope_9ab93ade();
}

CUDA_CALLABLE void adj_atomic_add(Polytope_9ab93ade* p, Polytope_9ab93ade t)
{
    wp::adj_atomic_add(&p->status, t.status);
    wp::adj_atomic_add(&p->vert, t.vert);
    wp::adj_atomic_add(&p->vert_index, t.vert_index);
    wp::adj_atomic_add(&p->nvert, t.nvert);
    wp::adj_atomic_add(&p->face, t.face);
    wp::adj_atomic_add(&p->face_pr, t.face_pr);
    wp::adj_atomic_add(&p->face_norm2, t.face_norm2);
    wp::adj_atomic_add(&p->nface, t.nface);
    wp::adj_atomic_add(&p->horizon, t.horizon);
    wp::adj_atomic_add(&p->nhorizon, t.nhorizon);
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:90
static CUDA_CALLABLE bool _discrete_geoms_0(
    wp::int32 var_g1,
    wp::int32 var_g2)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 7;
    bool var_1;
    const wp::int32 var_2 = 6;
    bool var_3;
    const wp::int32 var_4 = 1;
    bool var_5;
    bool var_6;
    const wp::int32 var_7 = 7;
    bool var_8;
    const wp::int32 var_9 = 6;
    bool var_10;
    const wp::int32 var_11 = 1;
    bool var_12;
    bool var_13;
    bool var_14;
    //---------
    // forward
    // def _discrete_geoms(g1: int, g2: int) -> bool:                                         <L 91>
    // return (g1 == GeomType.MESH or g1 == GeomType.BOX or g1 == GeomType.HFIELD) and (       <L 92>
    var_1 = (var_g1 == var_0);
    var_3 = (var_g1 == var_2);
    var_5 = (var_g1 == var_4);
    var_6 = var_1 || var_3 || var_5;
    // g2 == GeomType.MESH or g2 == GeomType.BOX or g2 == GeomType.HFIELD                     <L 93>
    var_8 = (var_g2 == var_7);
    var_10 = (var_g2 == var_9);
    var_12 = (var_g2 == var_11);
    var_13 = var_8 || var_10 || var_12;
    var_14 = var_6 && var_13;
    return var_14;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:97
static CUDA_CALLABLE SupportPoint_e82efc60 support_0(
    Geom_3242f8a8 var_geom,
    wp::int32 var_geomtype,
    wp::vec_t<3, wp::float32> var_dir)
{
    //---------
    // primal vars
    SupportPoint_e82efc60 var_0;
    const wp::int32 var_1 = 1;
    const wp::int32 var_2 = -1;
    wp::int32* var_3;
    const wp::int32 var_4 = 1;
    const wp::int32 var_5 = -1;
    wp::int32* var_6;
    const wp::int32 var_7 = 2;
    bool var_8;
    wp::vec_t<3, wp::float32>* var_9;
    wp::vec_t<3, wp::float32>* var_10;
    const wp::int32 var_11 = 0;
    wp::float32 var_12;
    wp::vec_t<3, wp::float32> var_13;
    const wp::float32 var_14 = 0.5;
    wp::float32* var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32>* var_22;
    wp::mat_t<3, 3, wp::float32>* var_23;
    wp::mat_t<3, 3, wp::float32> var_24;
    wp::mat_t<3, 3, wp::float32> var_25;
    wp::vec_t<3, wp::float32> var_26;
    const wp::int32 var_27 = 6;
    bool var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32>* var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::mat_t<3, 3, wp::float32>* var_33;
    wp::vec_t<3, wp::float32> var_34;
    wp::mat_t<3, 3, wp::float32> var_35;
    wp::vec_t<3, wp::float32>* var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::vec_t<3, wp::float32>* var_39;
    const wp::int32 var_40 = 0;
    wp::float32 var_41;
    const wp::float32 var_42 = 0.0;
    bool var_43;
    const wp::int32 var_44 = 1;
    const wp::int32 var_45 = 0;
    wp::int32 var_46;
    wp::int32* var_47;
    const wp::int32 var_48 = 1;
    wp::float32 var_49;
    const wp::float32 var_50 = 0.0;
    bool var_51;
    const wp::int32 var_52 = 2;
    const wp::int32 var_53 = 0;
    wp::int32 var_54;
    wp::int32* var_55;
    const wp::int32 var_56 = 1;
    wp::float32 var_57;
    const wp::float32 var_58 = 0.0;
    bool var_59;
    const wp::int32 var_60 = 2;
    const wp::int32 var_61 = 0;
    wp::int32 var_62;
    wp::int32 var_63;
    wp::int32 var_64;
    wp::int32* var_65;
    const wp::int32 var_66 = 2;
    wp::float32 var_67;
    const wp::float32 var_68 = 0.0;
    bool var_69;
    const wp::int32 var_70 = 4;
    const wp::int32 var_71 = 0;
    wp::int32 var_72;
    wp::int32* var_73;
    const wp::int32 var_74 = 2;
    wp::float32 var_75;
    const wp::float32 var_76 = 0.0;
    bool var_77;
    const wp::int32 var_78 = 4;
    const wp::int32 var_79 = 0;
    wp::int32 var_80;
    wp::int32 var_81;
    wp::int32 var_82;
    wp::int32* var_83;
    const wp::int32 var_84 = 3;
    bool var_85;
    wp::vec_t<3, wp::float32>* var_86;
    const wp::int32 var_87 = 0;
    wp::float32 var_88;
    wp::vec_t<3, wp::float32> var_89;
    wp::vec_t<3, wp::float32> var_90;
    const wp::int32 var_91 = 2;
    wp::float32 var_92;
    wp::float32 var_93;
    wp::vec_t<3, wp::float32>* var_94;
    const wp::int32 var_95 = 1;
    wp::float32 var_96;
    wp::vec_t<3, wp::float32> var_97;
    wp::float32 var_98;
    const wp::int32 var_99 = 2;
    wp::mat_t<3, 3, wp::float32>* var_100;
    wp::vec_t<3, wp::float32> var_101;
    wp::mat_t<3, 3, wp::float32> var_102;
    wp::vec_t<3, wp::float32>* var_103;
    wp::vec_t<3, wp::float32> var_104;
    wp::vec_t<3, wp::float32> var_105;
    wp::vec_t<3, wp::float32>* var_106;
    wp::vec_t<3, wp::float32> var_107;
    const wp::int32 var_108 = 4;
    bool var_109;
    wp::vec_t<3, wp::float32>* var_110;
    wp::vec_t<3, wp::float32> var_111;
    wp::vec_t<3, wp::float32> var_112;
    wp::vec_t<3, wp::float32> var_113;
    wp::vec_t<3, wp::float32>* var_114;
    wp::vec_t<3, wp::float32> var_115;
    wp::vec_t<3, wp::float32> var_116;
    wp::mat_t<3, 3, wp::float32>* var_117;
    wp::vec_t<3, wp::float32> var_118;
    wp::mat_t<3, 3, wp::float32> var_119;
    wp::vec_t<3, wp::float32>* var_120;
    wp::vec_t<3, wp::float32> var_121;
    wp::vec_t<3, wp::float32> var_122;
    wp::vec_t<3, wp::float32>* var_123;
    wp::vec_t<3, wp::float32> var_124;
    const wp::int32 var_125 = 5;
    bool var_126;
    const wp::float32 var_127 = 0.0;
    const wp::float32 var_128 = 0.0;
    const wp::float32 var_129 = 0.0;
    wp::vec_t<3, wp::float32> var_130;
    const wp::int32 var_131 = 0;
    wp::float32 var_132;
    const wp::int32 var_133 = 0;
    wp::float32 var_134;
    wp::float32 var_135;
    const wp::int32 var_136 = 1;
    wp::float32 var_137;
    const wp::int32 var_138 = 1;
    wp::float32 var_139;
    wp::float32 var_140;
    wp::float32 var_141;
    wp::float32 var_142;
    const wp::float32 var_143 = 1e-15;
    bool var_144;
    wp::vec_t<3, wp::float32>* var_145;
    const wp::int32 var_146 = 0;
    wp::float32 var_147;
    wp::vec_t<3, wp::float32> var_148;
    wp::float32 var_149;
    const wp::int32 var_150 = 0;
    wp::float32 var_151;
    wp::float32 var_152;
    const wp::int32 var_153 = 0;
    const wp::int32 var_154 = 1;
    wp::float32 var_155;
    wp::float32 var_156;
    const wp::int32 var_157 = 1;
    const wp::int32 var_158 = 2;
    wp::float32 var_159;
    wp::float32 var_160;
    wp::vec_t<3, wp::float32>* var_161;
    const wp::int32 var_162 = 1;
    wp::float32 var_163;
    wp::vec_t<3, wp::float32> var_164;
    wp::float32 var_165;
    const wp::int32 var_166 = 2;
    wp::mat_t<3, 3, wp::float32>* var_167;
    wp::vec_t<3, wp::float32> var_168;
    wp::mat_t<3, 3, wp::float32> var_169;
    wp::vec_t<3, wp::float32>* var_170;
    wp::vec_t<3, wp::float32> var_171;
    wp::vec_t<3, wp::float32> var_172;
    wp::vec_t<3, wp::float32>* var_173;
    wp::vec_t<3, wp::float32> var_174;
    const wp::int32 var_175 = 7;
    bool var_176;
    const wp::float32 var_177 = -1e+30;
    wp::float32 var_178;
    wp::int32* var_179;
    const wp::int32 var_180 = 1;
    const wp::int32 var_181 = -1;
    bool var_182;
    wp::int32 var_183;
    wp::int32* var_184;
    const wp::int32 var_185 = 10;
    bool var_186;
    wp::int32 var_187;
    bool var_188;
    wp::int32* var_189;
    const wp::int32 var_190 = 1;
    const wp::int32 var_191 = -1;
    bool var_192;
    wp::int32 var_193;
    wp::int32* var_194;
    wp::int32* var_195;
    wp::int32 var_196;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_197;
    wp::int32* var_198;
    wp::vec_t<3, wp::float32>* var_199;
    wp::array_t<wp::vec_t<3, wp::float32>> var_200;
    wp::int32 var_201;
    wp::float32 var_202;
    wp::vec_t<3, wp::float32> var_203;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_204;
    wp::int32* var_205;
    wp::vec_t<3, wp::float32>* var_206;
    wp::array_t<wp::vec_t<3, wp::float32>> var_207;
    wp::int32 var_208;
    wp::vec_t<3, wp::float32>* var_209;
    wp::vec_t<3, wp::float32> var_210;
    wp::float32 var_211;
    wp::int32* var_212;
    wp::range_t var_213;
    wp::int32 var_214;
    wp::int32 var_215;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_216;
    wp::int32* var_217;
    wp::int32 var_218;
    wp::int32 var_219;
    wp::vec_t<3, wp::float32>* var_220;
    wp::array_t<wp::vec_t<3, wp::float32>> var_221;
    wp::vec_t<3, wp::float32> var_222;
    wp::vec_t<3, wp::float32> var_223;
    wp::float32 var_224;
    bool var_225;
    wp::float32 var_226;
    wp::vec_t<3, wp::float32>* var_227;
    wp::int32* var_228;
    wp::int32 var_229;
    wp::int32 var_230;
    wp::int32* var_231;
    wp::float32 var_232;
    wp::int32* var_233;
    wp::int32* var_234;
    wp::int32 var_235;
    wp::int32 var_236;
    wp::int32 var_237;
    wp::int32* var_238;
    wp::float32 var_239;
    wp::array_t<wp::int32>* var_240;
    wp::int32* var_241;
    wp::int32* var_242;
    wp::array_t<wp::int32> var_243;
    wp::int32 var_244;
    wp::int32 var_245;
    wp::int32 var_246;
    wp::int32* var_247;
    const wp::int32 var_248 = 2;
    wp::int32 var_249;
    wp::int32 var_250;
    wp::int32* var_251;
    const wp::int32 var_252 = 2;
    wp::int32 var_253;
    wp::int32 var_254;
    wp::int32 var_255;
    wp::int32* var_256;
    const wp::int32 var_257 = 2;
    wp::int32 var_258;
    wp::int32 var_259;
    const wp::int32 var_260 = 2;
    wp::int32 var_261;
    wp::int32 var_262;
    const wp::int32 var_263 = 1;
    const wp::int32 var_264 = -1;
    wp::int32 var_265;
    wp::int32* var_266;
    const wp::int32 var_267 = 1;
    const wp::int32 var_268 = -1;
    bool var_269;
    wp::int32 var_270;
    wp::int32* var_271;
    const wp::int32 var_272 = 0;
    wp::int32 var_273;
    wp::int32 var_274;
    bool var_275;
    wp::int32 var_276;
    wp::array_t<wp::int32>* var_277;
    wp::int32 var_278;
    wp::int32* var_279;
    wp::array_t<wp::int32> var_280;
    wp::int32 var_281;
    wp::int32 var_282;
    wp::array_t<wp::int32>* var_283;
    wp::int32 var_284;
    wp::int32* var_285;
    wp::array_t<wp::int32> var_286;
    wp::int32 var_287;
    wp::int32 var_288;
    const wp::int32 var_289 = 0;
    bool var_290;
    wp::array_t<wp::int32>* var_291;
    wp::int32 var_292;
    wp::int32* var_293;
    wp::array_t<wp::int32> var_294;
    wp::int32 var_295;
    wp::int32 var_296;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_297;
    wp::int32* var_298;
    wp::int32 var_299;
    wp::int32 var_300;
    wp::vec_t<3, wp::float32>* var_301;
    wp::array_t<wp::vec_t<3, wp::float32>> var_302;
    wp::float32 var_303;
    wp::vec_t<3, wp::float32> var_304;
    bool var_305;
    wp::int32 var_306;
    bool var_307;
    wp::float32 var_308;
    const wp::int32 var_309 = 1;
    wp::int32 var_310;
    wp::array_t<wp::int32>* var_311;
    wp::int32 var_312;
    wp::int32* var_313;
    wp::array_t<wp::int32> var_314;
    wp::int32 var_315;
    wp::int32 var_316;
    wp::int32* var_317;
    wp::array_t<wp::int32>* var_318;
    wp::int32 var_319;
    wp::int32* var_320;
    wp::array_t<wp::int32> var_321;
    wp::int32* var_322;
    wp::int32 var_323;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_324;
    wp::int32* var_325;
    wp::int32* var_326;
    wp::int32 var_327;
    wp::int32 var_328;
    wp::int32 var_329;
    wp::vec_t<3, wp::float32>* var_330;
    wp::array_t<wp::vec_t<3, wp::float32>> var_331;
    wp::vec_t<3, wp::float32>* var_332;
    wp::vec_t<3, wp::float32> var_333;
    wp::mat_t<3, 3, wp::float32>* var_334;
    wp::vec_t<3, wp::float32>* var_335;
    wp::vec_t<3, wp::float32> var_336;
    wp::mat_t<3, 3, wp::float32> var_337;
    wp::vec_t<3, wp::float32> var_338;
    wp::vec_t<3, wp::float32>* var_339;
    wp::vec_t<3, wp::float32> var_340;
    wp::vec_t<3, wp::float32> var_341;
    wp::vec_t<3, wp::float32>* var_342;
    const wp::int32 var_343 = 1;
    bool var_344;
    wp::float32 var_345;
    const wp::int32 var_346 = 2;
    wp::float32 var_347;
    const wp::float32 var_348 = 0.0;
    bool var_349;
    const wp::int32 var_350 = 2;
    const wp::int32 var_351 = -2;
    const wp::int32 var_352 = 3;
    const wp::int32 var_353 = -3;
    wp::int32 var_354;
    wp::int32* var_355;
    const wp::int32 var_356 = 0;
    wp::mat_t<6, 3, wp::float32>* var_357;
    wp::vec_t<3, wp::float32> var_358;
    wp::mat_t<6, 3, wp::float32> var_359;
    wp::float32 var_360;
    bool var_361;
    wp::float32 var_362;
    wp::vec_t<3, wp::float32>* var_363;
    wp::float32 var_364;
    const wp::int32 var_365 = 1;
    wp::mat_t<6, 3, wp::float32>* var_366;
    wp::vec_t<3, wp::float32> var_367;
    wp::mat_t<6, 3, wp::float32> var_368;
    wp::float32 var_369;
    bool var_370;
    wp::float32 var_371;
    wp::vec_t<3, wp::float32>* var_372;
    wp::float32 var_373;
    const wp::int32 var_374 = 2;
    wp::mat_t<6, 3, wp::float32>* var_375;
    wp::vec_t<3, wp::float32> var_376;
    wp::mat_t<6, 3, wp::float32> var_377;
    wp::float32 var_378;
    bool var_379;
    wp::float32 var_380;
    wp::vec_t<3, wp::float32>* var_381;
    wp::float32 var_382;
    const wp::int32 var_383 = 3;
    wp::mat_t<6, 3, wp::float32>* var_384;
    wp::vec_t<3, wp::float32> var_385;
    wp::mat_t<6, 3, wp::float32> var_386;
    wp::float32 var_387;
    bool var_388;
    wp::float32 var_389;
    wp::vec_t<3, wp::float32>* var_390;
    wp::float32 var_391;
    const wp::int32 var_392 = 4;
    wp::mat_t<6, 3, wp::float32>* var_393;
    wp::vec_t<3, wp::float32> var_394;
    wp::mat_t<6, 3, wp::float32> var_395;
    wp::float32 var_396;
    bool var_397;
    wp::float32 var_398;
    wp::vec_t<3, wp::float32>* var_399;
    wp::float32 var_400;
    const wp::int32 var_401 = 5;
    wp::mat_t<6, 3, wp::float32>* var_402;
    wp::vec_t<3, wp::float32> var_403;
    wp::mat_t<6, 3, wp::float32> var_404;
    wp::float32 var_405;
    bool var_406;
    wp::float32 var_407;
    wp::vec_t<3, wp::float32>* var_408;
    wp::float32 var_409;
    wp::float32 var_410;
    wp::int32 var_411;
    wp::vec_t<3, wp::float32> var_412;
    wp::float32 var_413;
    wp::float32 var_414;
    wp::int32 var_415;
    wp::vec_t<3, wp::float32> var_416;
    wp::float32 var_417;
    wp::vec_t<3, wp::float32> var_418;
    wp::vec_t<3, wp::float32> var_419;
    wp::vec_t<3, wp::float32> var_420;
    wp::float32* var_421;
    const wp::float32 var_422 = 0.0;
    bool var_423;
    wp::float32 var_424;
    const wp::float32 var_425 = 0.5;
    wp::float32* var_426;
    wp::float32 var_427;
    wp::float32 var_428;
    wp::vec_t<3, wp::float32> var_429;
    wp::vec_t<3, wp::float32>* var_430;
    const wp::float32 var_431 = 0.5;
    wp::float32* var_432;
    wp::float32 var_433;
    wp::float32 var_434;
    wp::vec_t<3, wp::float32> var_435;
    wp::vec_t<3, wp::float32> var_436;
    wp::vec_t<3, wp::float32> var_437;
    wp::vec_t<3, wp::float32>* var_438;
    //---------
    // forward
    // def support(geom: Geom, geomtype: int, dir: wp.vec3) -> SupportPoint:                  <L 98>
    // sp = SupportPoint()                                                                    <L 99>
    var_0 = SupportPoint_e82efc60();
    // sp.cached_index = -1                                                                   <L 100>
    var_3 = &(var_0.cached_index);
    wp::store(var_3, var_2);
    // sp.vertex_index = -1                                                                   <L 101>
    var_6 = &(var_0.vertex_index);
    wp::store(var_6, var_5);
    // if geomtype == GeomType.SPHERE:                                                        <L 102>
    var_8 = (var_geomtype == var_7);
    if (var_8) {
        // sp.point = geom.pos + (geom.size[0] + 0.5 * geom.margin) * dir                     <L 103>
        var_9 = &(var_geom.pos);
        var_10 = &(var_geom.size);
        var_13 = wp::load(var_10);
        var_12 = wp::extract(var_13, var_11);
        var_15 = &(var_geom.margin);
        var_17 = wp::load(var_15);
        var_16 = wp::mul(var_14, var_17);
        var_18 = wp::add(var_12, var_16);
        var_19 = wp::mul(var_18, var_dir);
        var_21 = wp::load(var_9);
        var_20 = wp::add(var_21, var_19);
        var_22 = &(var_0.point);
        wp::store(var_22, var_20);
        // return sp                                                                          <L 104>
        return var_0;
    }
    // local_dir = wp.transpose(geom.rot) @ dir                                               <L 106>
    var_23 = &(var_geom.rot);
    var_25 = wp::load(var_23);
    var_24 = wp::transpose(var_25);
    var_26 = wp::mul(var_24, var_dir);
    // if geomtype == GeomType.BOX:                                                           <L 107>
    var_28 = (var_geomtype == var_27);
    if (var_28) {
        // tmp = wp.sign(local_dir)                                                           <L 108>
        var_29 = wp::sign(var_26);
        // res = wp.cw_mul(tmp, geom.size)                                                    <L 109>
        var_30 = &(var_geom.size);
        var_32 = wp::load(var_30);
        var_31 = wp::cw_mul(var_29, var_32);
        // sp.point = geom.rot @ res + geom.pos                                               <L 110>
        var_33 = &(var_geom.rot);
        var_35 = wp::load(var_33);
        var_34 = wp::mul(var_35, var_31);
        var_36 = &(var_geom.pos);
        var_38 = wp::load(var_36);
        var_37 = wp::add(var_34, var_38);
        var_39 = &(var_0.point);
        wp::store(var_39, var_37);
        // sp.vertex_index = wp.where(tmp[0] > 0.0, 1, 0)                                     <L 111>
        var_41 = wp::extract(var_29, var_40);
        var_43 = (var_41 > var_42);
        var_46 = wp::where(var_43, var_44, var_45);
        var_47 = &(var_0.vertex_index);
        wp::store(var_47, var_46);
        // sp.vertex_index += wp.where(tmp[1] > 0.0, 2, 0)                                    <L 112>
        var_49 = wp::extract(var_29, var_48);
        var_51 = (var_49 > var_50);
        var_54 = wp::where(var_51, var_52, var_53);
        var_55 = &(var_0.vertex_index);
        var_57 = wp::extract(var_29, var_56);
        var_59 = (var_57 > var_58);
        var_62 = wp::where(var_59, var_60, var_61);
        var_64 = wp::load(var_55);
        var_63 = wp::add(var_64, var_62);
        var_65 = &(var_0.vertex_index);
        wp::store(var_65, var_63);
        // sp.vertex_index += wp.where(tmp[2] > 0.0, 4, 0)                                    <L 113>
        var_67 = wp::extract(var_29, var_66);
        var_69 = (var_67 > var_68);
        var_72 = wp::where(var_69, var_70, var_71);
        var_73 = &(var_0.vertex_index);
        var_75 = wp::extract(var_29, var_74);
        var_77 = (var_75 > var_76);
        var_80 = wp::where(var_77, var_78, var_79);
        var_82 = wp::load(var_73);
        var_81 = wp::add(var_82, var_80);
        var_83 = &(var_0.vertex_index);
        wp::store(var_83, var_81);
    }
    if (!var_28) {
        // elif geomtype == GeomType.CAPSULE:                                                 <L 114>
        var_85 = (var_geomtype == var_84);
        if (var_85) {
            // res = local_dir * geom.size[0]                                                 <L 115>
            var_86 = &(var_geom.size);
            var_89 = wp::load(var_86);
            var_88 = wp::extract(var_89, var_87);
            var_90 = wp::mul(var_26, var_88);
            // res[2] += wp.sign(local_dir[2]) * geom.size[1]                                 <L 117>
            var_92 = wp::extract(var_26, var_91);
            var_93 = wp::sign(var_92);
            var_94 = &(var_geom.size);
            var_97 = wp::load(var_94);
            var_96 = wp::extract(var_97, var_95);
            var_98 = wp::mul(var_93, var_96);
            wp::add_inplace(var_90, var_99, var_98);
            // sp.point = geom.rot @ res + geom.pos                                           <L 118>
            var_100 = &(var_geom.rot);
            var_102 = wp::load(var_100);
            var_101 = wp::mul(var_102, var_90);
            var_103 = &(var_geom.pos);
            var_105 = wp::load(var_103);
            var_104 = wp::add(var_101, var_105);
            var_106 = &(var_0.point);
            wp::store(var_106, var_104);
        }
        var_107 = wp::where(var_85, var_90, var_31);
        if (!var_85) {
            // elif geomtype == GeomType.ELLIPSOID:                                           <L 119>
            var_109 = (var_geomtype == var_108);
            if (var_109) {
                // res = wp.cw_mul(local_dir, geom.size)                                      <L 120>
                var_110 = &(var_geom.size);
                var_112 = wp::load(var_110);
                var_111 = wp::cw_mul(var_26, var_112);
                // res = wp.normalize(res)                                                    <L 121>
                var_113 = wp::normalize(var_111);
                // res = wp.cw_mul(res, geom.size)                                            <L 123>
                var_114 = &(var_geom.size);
                var_116 = wp::load(var_114);
                var_115 = wp::cw_mul(var_113, var_116);
                // sp.point = geom.rot @ res + geom.pos                                       <L 124>
                var_117 = &(var_geom.rot);
                var_119 = wp::load(var_117);
                var_118 = wp::mul(var_119, var_115);
                var_120 = &(var_geom.pos);
                var_122 = wp::load(var_120);
                var_121 = wp::add(var_118, var_122);
                var_123 = &(var_0.point);
                wp::store(var_123, var_121);
            }
            var_124 = wp::where(var_109, var_115, var_107);
            if (!var_109) {
                // elif geomtype == GeomType.CYLINDER:                                        <L 125>
                var_126 = (var_geomtype == var_125);
                if (var_126) {
                    // res = wp.vec3(0.0, 0.0, 0.0)                                           <L 126>
                    var_130 = wp::vec_t<3, wp::float32>(var_127, var_128, var_129);
                    // d = wp.sqrt(local_dir[0] * local_dir[0] + local_dir[1] * local_dir[1])       <L 128>
                    var_132 = wp::extract(var_26, var_131);
                    var_134 = wp::extract(var_26, var_133);
                    var_135 = wp::mul(var_132, var_134);
                    var_137 = wp::extract(var_26, var_136);
                    var_139 = wp::extract(var_26, var_138);
                    var_140 = wp::mul(var_137, var_139);
                    var_141 = wp::add(var_135, var_140);
                    var_142 = wp::sqrt(var_141);
                    // if d > MINVAL:                                                         <L 129>
                    var_144 = (var_142 > var_143);
                    if (var_144) {
                        // scl = geom.size[0] / d                                             <L 130>
                        var_145 = &(var_geom.size);
                        var_148 = wp::load(var_145);
                        var_147 = wp::extract(var_148, var_146);
                        var_149 = wp::div(var_147, var_142);
                        // res[0] = local_dir[0] * scl                                        <L 131>
                        var_151 = wp::extract(var_26, var_150);
                        var_152 = wp::mul(var_151, var_149);
                        wp::assign_inplace(var_130, var_153, var_152);
                        // res[1] = local_dir[1] * scl                                        <L 132>
                        var_155 = wp::extract(var_26, var_154);
                        var_156 = wp::mul(var_155, var_149);
                        wp::assign_inplace(var_130, var_157, var_156);
                    }
                    // res[2] = wp.sign(local_dir[2]) * geom.size[1]                          <L 134>
                    var_159 = wp::extract(var_26, var_158);
                    var_160 = wp::sign(var_159);
                    var_161 = &(var_geom.size);
                    var_164 = wp::load(var_161);
                    var_163 = wp::extract(var_164, var_162);
                    var_165 = wp::mul(var_160, var_163);
                    wp::assign_inplace(var_130, var_166, var_165);
                    // sp.point = geom.rot @ res + geom.pos                                   <L 135>
                    var_167 = &(var_geom.rot);
                    var_169 = wp::load(var_167);
                    var_168 = wp::mul(var_169, var_130);
                    var_170 = &(var_geom.pos);
                    var_172 = wp::load(var_170);
                    var_171 = wp::add(var_168, var_172);
                    var_173 = &(var_0.point);
                    wp::store(var_173, var_171);
                }
                var_174 = wp::where(var_126, var_130, var_124);
                if (!var_126) {
                    // elif geomtype == GeomType.MESH:                                        <L 136>
                    var_176 = (var_geomtype == var_175);
                    if (var_176) {
                        // max_dist = float(FLOAT_MIN)                                        <L 137>
                        var_178 = wp::float(var_177);
                        // if geom.graphadr == -1 or geom.vertnum < 10:                       <L 138>
                        var_179 = &(var_geom.graphadr);
                        var_183 = wp::load(var_179);
                        var_182 = (var_183 == var_181);
                        var_184 = &(var_geom.vertnum);
                        var_187 = wp::load(var_184);
                        var_186 = (var_187 < var_185);
                        var_188 = var_182 || var_186;
                        if (var_188) {
                            // if geom.index > -1:                                            <L 139>
                            var_189 = &(var_geom.index);
                            var_193 = wp::load(var_189);
                            var_192 = (var_193 > var_191);
                            if (var_192) {
                                // sp.cached_index = geom.index                               <L 140>
                                var_194 = &(var_geom.index);
                                var_195 = &(var_0.cached_index);
                                var_196 = wp::load(var_194);
                                wp::store(var_195, var_196);
                                // max_dist = wp.dot(geom.vert[geom.index], local_dir)        <L 141>
                                var_197 = &(var_geom.vert);
                                var_198 = &(var_geom.index);
                                var_200 = wp::load(var_197);
                                var_201 = wp::load(var_198);
                                var_199 = wp::address(var_200, var_201);
                                var_203 = wp::load(var_199);
                                var_202 = wp::dot(var_203, var_26);
                                // sp.point = geom.vert[geom.index]                           <L 142>
                                var_204 = &(var_geom.vert);
                                var_205 = &(var_geom.index);
                                var_207 = wp::load(var_204);
                                var_208 = wp::load(var_205);
                                var_206 = wp::address(var_207, var_208);
                                var_209 = &(var_0.point);
                                var_210 = wp::load(var_206);
                                wp::store(var_209, var_210);
                            }
                            var_211 = wp::where(var_192, var_202, var_178);
                            // for i in range(geom.vertnum):                                  <L 144>
                            var_212 = &(var_geom.vertnum);
                            var_214 = wp::load(var_212);
                            var_213 = wp::range(var_214);
                            start_for_1:;
                                if (iter_cmp(var_213) == 0) goto end_for_1;
                                var_215 = wp::iter_next(var_213);
                                // vert = geom.vert[geom.vertadr + i]                         <L 145>
                                var_216 = &(var_geom.vert);
                                var_217 = &(var_geom.vertadr);
                                var_219 = wp::load(var_217);
                                var_218 = wp::add(var_219, var_215);
                                var_221 = wp::load(var_216);
                                var_220 = wp::address(var_221, var_218);
                                var_223 = wp::load(var_220);
                                var_222 = wp::copy(var_223);
                                // dist = wp.dot(vert, local_dir)                             <L 146>
                                var_224 = wp::dot(var_222, var_26);
                                // if dist > max_dist:                                        <L 147>
                                var_225 = (var_224 > var_211);
                                if (var_225) {
                                    // max_dist = dist                                        <L 148>
                                    var_226 = wp::copy(var_224);
                                    // sp.point = vert                                        <L 149>
                                    var_227 = &(var_0.point);
                                    wp::store(var_227, var_222);
                                    // sp.cached_index = geom.vertadr + i                     <L 150>
                                    var_228 = &(var_geom.vertadr);
                                    var_230 = wp::load(var_228);
                                    var_229 = wp::add(var_230, var_215);
                                    var_231 = &(var_0.cached_index);
                                    wp::store(var_231, var_229);
                                }
                                var_232 = wp::where(var_225, var_226, var_211);
                                wp::assign(var_211, var_232);
                                goto start_for_1;
                            end_for_1:;
                            // sp.vertex_index = sp.cached_index - geom.vertadr               <L 151>
                            var_233 = &(var_0.cached_index);
                            var_234 = &(var_geom.vertadr);
                            var_236 = wp::load(var_233);
                            var_237 = wp::load(var_234);
                            var_235 = wp::sub(var_236, var_237);
                            var_238 = &(var_0.vertex_index);
                            wp::store(var_238, var_235);
                        }
                        var_239 = wp::where(var_188, var_211, var_178);
                        if (!var_188) {
                            // numvert = geom.graph[geom.graphadr]                            <L 153>
                            var_240 = &(var_geom.graph);
                            var_241 = &(var_geom.graphadr);
                            var_243 = wp::load(var_240);
                            var_244 = wp::load(var_241);
                            var_242 = wp::address(var_243, var_244);
                            var_246 = wp::load(var_242);
                            var_245 = wp::copy(var_246);
                            // vert_edgeadr = geom.graphadr + 2                               <L 154>
                            var_247 = &(var_geom.graphadr);
                            var_250 = wp::load(var_247);
                            var_249 = wp::add(var_250, var_248);
                            // vert_globalid = geom.graphadr + 2 + numvert                    <L 155>
                            var_251 = &(var_geom.graphadr);
                            var_254 = wp::load(var_251);
                            var_253 = wp::add(var_254, var_252);
                            var_255 = wp::add(var_253, var_245);
                            // edge_localid = geom.graphadr + 2 + 2 * numvert                 <L 156>
                            var_256 = &(var_geom.graphadr);
                            var_259 = wp::load(var_256);
                            var_258 = wp::add(var_259, var_257);
                            var_261 = wp::mul(var_260, var_245);
                            var_262 = wp::add(var_258, var_261);
                            // prev = int(-1)                                                 <L 157>
                            var_265 = wp::int(var_264);
                            // imax = wp.where(geom.index > -1, geom.index, 0)                <L 158>
                            var_266 = &(var_geom.index);
                            var_270 = wp::load(var_266);
                            var_269 = (var_270 > var_268);
                            var_271 = &(var_geom.index);
                            var_274 = wp::load(var_271);
                            var_273 = wp::where(var_269, var_274, var_272);
                            // while imax != prev:                                            <L 161>
    start_while_3:;
                            var_275 = (var_273 != var_265);
    if ((var_275) == false) goto end_while_3;
                                // prev = imax                                                <L 162>
                                var_276 = wp::copy(var_273);
                                // i = geom.graph[vert_edgeadr + imax]                        <L 163>
                                var_277 = &(var_geom.graph);
                                var_278 = wp::add(var_249, var_273);
                                var_280 = wp::load(var_277);
                                var_279 = wp::address(var_280, var_278);
                                var_282 = wp::load(var_279);
                                var_281 = wp::copy(var_282);
                                // subidx = geom.graph[edge_localid + i]                      <L 164>
                                var_283 = &(var_geom.graph);
                                var_284 = wp::add(var_262, var_281);
                                var_286 = wp::load(var_283);
                                var_285 = wp::address(var_286, var_284);
                                var_288 = wp::load(var_285);
                                var_287 = wp::copy(var_288);
                                // while subidx >= 0:                                         <L 165>
    start_while_5:;
                                var_290 = (var_287 >= var_289);
    if ((var_290) == false) goto end_while_5;
                                    // idx = geom.graph[vert_globalid + subidx]               <L 166>
                                    var_291 = &(var_geom.graph);
                                    var_292 = wp::add(var_255, var_287);
                                    var_294 = wp::load(var_291);
                                    var_293 = wp::address(var_294, var_292);
                                    var_296 = wp::load(var_293);
                                    var_295 = wp::copy(var_296);
                                    // dist = wp.dot(local_dir, geom.vert[geom.vertadr + idx])       <L 167>
                                    var_297 = &(var_geom.vert);
                                    var_298 = &(var_geom.vertadr);
                                    var_300 = wp::load(var_298);
                                    var_299 = wp::add(var_300, var_295);
                                    var_302 = wp::load(var_297);
                                    var_301 = wp::address(var_302, var_299);
                                    var_304 = wp::load(var_301);
                                    var_303 = wp::dot(var_26, var_304);
                                    // imax = wp.where(dist > max_dist, subidx, imax)         <L 168>
                                    var_305 = (var_303 > var_239);
                                    var_306 = wp::where(var_305, var_287, var_273);
                                    // max_dist = wp.where(dist > max_dist, dist, max_dist)       <L 169>
                                    var_307 = (var_303 > var_239);
                                    var_308 = wp::where(var_307, var_303, var_239);
                                    // i += 1                                                 <L 170>
                                    var_310 = wp::add(var_281, var_309);
                                    // subidx = geom.graph[edge_localid + i]                  <L 171>
                                    var_311 = &(var_geom.graph);
                                    var_312 = wp::add(var_262, var_310);
                                    var_314 = wp::load(var_311);
                                    var_313 = wp::address(var_314, var_312);
                                    var_316 = wp::load(var_313);
                                    var_315 = wp::copy(var_316);
                                    wp::assign(var_239, var_308);
                                    wp::assign(var_281, var_310);
                                    wp::assign(var_224, var_303);
                                    wp::assign(var_273, var_306);
                                    wp::assign(var_287, var_315);
    goto start_while_5;
    end_while_5:;
                                wp::assign(var_215, var_281);
                                wp::assign(var_265, var_276);
    goto start_while_3;
    end_while_3:;
                            // sp.cached_index = imax                                         <L 173>
                            var_317 = &(var_0.cached_index);
                            wp::store(var_317, var_273);
                            // sp.vertex_index = geom.graph[vert_globalid + imax]             <L 174>
                            var_318 = &(var_geom.graph);
                            var_319 = wp::add(var_255, var_273);
                            var_321 = wp::load(var_318);
                            var_320 = wp::address(var_321, var_319);
                            var_322 = &(var_0.vertex_index);
                            var_323 = wp::load(var_320);
                            wp::store(var_322, var_323);
                            // sp.point = geom.vert[geom.vertadr + sp.vertex_index]           <L 175>
                            var_324 = &(var_geom.vert);
                            var_325 = &(var_geom.vertadr);
                            var_326 = &(var_0.vertex_index);
                            var_328 = wp::load(var_325);
                            var_329 = wp::load(var_326);
                            var_327 = wp::add(var_328, var_329);
                            var_331 = wp::load(var_324);
                            var_330 = wp::address(var_331, var_327);
                            var_332 = &(var_0.point);
                            var_333 = wp::load(var_330);
                            wp::store(var_332, var_333);
                        }
                        // sp.point = geom.rot @ sp.point + geom.pos                          <L 177>
                        var_334 = &(var_geom.rot);
                        var_335 = &(var_0.point);
                        var_337 = wp::load(var_334);
                        var_338 = wp::load(var_335);
                        var_336 = wp::mul(var_337, var_338);
                        var_339 = &(var_geom.pos);
                        var_341 = wp::load(var_339);
                        var_340 = wp::add(var_336, var_341);
                        var_342 = &(var_0.point);
                        wp::store(var_342, var_340);
                    }
                    if (!var_176) {
                        // elif geomtype == GeomType.HFIELD:                                  <L 178>
                        var_344 = (var_geomtype == var_343);
                        if (var_344) {
                            // max_dist = float(FLOAT_MIN)                                    <L 179>
                            var_345 = wp::float(var_177);
                            // sp.vertex_index = wp.where(dir[2] < 0.0, -2, -3)               <L 181>
                            var_347 = wp::extract(var_dir, var_346);
                            var_349 = (var_347 < var_348);
                            var_354 = wp::where(var_349, var_351, var_353);
                            var_355 = &(var_0.vertex_index);
                            wp::store(var_355, var_354);
                            // for i in range(6):                                             <L 182>
                            // vert = geom.hfprism[i]                                         <L 183>
                            var_357 = &(var_geom.hfprism);
                            var_359 = wp::load(var_357);
                            var_358 = wp::extract(var_359, var_356);
                            // dist = wp.dot(vert, dir)                                       <L 184>
                            var_360 = wp::dot(var_358, var_dir);
                            // if dist > max_dist:                                            <L 185>
                            var_361 = (var_360 > var_345);
                            if (var_361) {
                                // max_dist = dist                                            <L 186>
                                var_362 = wp::copy(var_360);
                                // sp.point = vert                                            <L 187>
                                var_363 = &(var_0.point);
                                wp::store(var_363, var_358);
                            }
                            var_364 = wp::where(var_361, var_362, var_345);
                            // vert = geom.hfprism[i]                                         <L 183>
                            var_366 = &(var_geom.hfprism);
                            var_368 = wp::load(var_366);
                            var_367 = wp::extract(var_368, var_365);
                            // dist = wp.dot(vert, dir)                                       <L 184>
                            var_369 = wp::dot(var_367, var_dir);
                            // if dist > max_dist:                                            <L 185>
                            var_370 = (var_369 > var_364);
                            if (var_370) {
                                // max_dist = dist                                            <L 186>
                                var_371 = wp::copy(var_369);
                                // sp.point = vert                                            <L 187>
                                var_372 = &(var_0.point);
                                wp::store(var_372, var_367);
                            }
                            var_373 = wp::where(var_370, var_371, var_364);
                            // vert = geom.hfprism[i]                                         <L 183>
                            var_375 = &(var_geom.hfprism);
                            var_377 = wp::load(var_375);
                            var_376 = wp::extract(var_377, var_374);
                            // dist = wp.dot(vert, dir)                                       <L 184>
                            var_378 = wp::dot(var_376, var_dir);
                            // if dist > max_dist:                                            <L 185>
                            var_379 = (var_378 > var_373);
                            if (var_379) {
                                // max_dist = dist                                            <L 186>
                                var_380 = wp::copy(var_378);
                                // sp.point = vert                                            <L 187>
                                var_381 = &(var_0.point);
                                wp::store(var_381, var_376);
                            }
                            var_382 = wp::where(var_379, var_380, var_373);
                            // vert = geom.hfprism[i]                                         <L 183>
                            var_384 = &(var_geom.hfprism);
                            var_386 = wp::load(var_384);
                            var_385 = wp::extract(var_386, var_383);
                            // dist = wp.dot(vert, dir)                                       <L 184>
                            var_387 = wp::dot(var_385, var_dir);
                            // if dist > max_dist:                                            <L 185>
                            var_388 = (var_387 > var_382);
                            if (var_388) {
                                // max_dist = dist                                            <L 186>
                                var_389 = wp::copy(var_387);
                                // sp.point = vert                                            <L 187>
                                var_390 = &(var_0.point);
                                wp::store(var_390, var_385);
                            }
                            var_391 = wp::where(var_388, var_389, var_382);
                            // vert = geom.hfprism[i]                                         <L 183>
                            var_393 = &(var_geom.hfprism);
                            var_395 = wp::load(var_393);
                            var_394 = wp::extract(var_395, var_392);
                            // dist = wp.dot(vert, dir)                                       <L 184>
                            var_396 = wp::dot(var_394, var_dir);
                            // if dist > max_dist:                                            <L 185>
                            var_397 = (var_396 > var_391);
                            if (var_397) {
                                // max_dist = dist                                            <L 186>
                                var_398 = wp::copy(var_396);
                                // sp.point = vert                                            <L 187>
                                var_399 = &(var_0.point);
                                wp::store(var_399, var_394);
                            }
                            var_400 = wp::where(var_397, var_398, var_391);
                            // vert = geom.hfprism[i]                                         <L 183>
                            var_402 = &(var_geom.hfprism);
                            var_404 = wp::load(var_402);
                            var_403 = wp::extract(var_404, var_401);
                            // dist = wp.dot(vert, dir)                                       <L 184>
                            var_405 = wp::dot(var_403, var_dir);
                            // if dist > max_dist:                                            <L 185>
                            var_406 = (var_405 > var_400);
                            if (var_406) {
                                // max_dist = dist                                            <L 186>
                                var_407 = wp::copy(var_405);
                                // sp.point = vert                                            <L 187>
                                var_408 = &(var_0.point);
                                wp::store(var_408, var_403);
                            }
                            var_409 = wp::where(var_406, var_407, var_400);
                        }
                        var_410 = wp::where(var_344, var_409, var_239);
                        var_411 = wp::where(var_344, var_401, var_215);
                        var_412 = wp::where(var_344, var_403, var_222);
                        var_413 = wp::where(var_344, var_405, var_224);
                    }
                    var_414 = wp::where(var_176, var_239, var_410);
                    var_415 = wp::where(var_176, var_215, var_411);
                    var_416 = wp::where(var_176, var_222, var_412);
                    var_417 = wp::where(var_176, var_224, var_413);
                }
            }
            var_418 = wp::where(var_109, var_124, var_174);
        }
        var_419 = wp::where(var_85, var_107, var_418);
    }
    var_420 = wp::where(var_28, var_31, var_419);
    // if geom.margin > 0.0:                                                                  <L 189>
    var_421 = &(var_geom.margin);
    var_424 = wp::load(var_421);
    var_423 = (var_424 > var_422);
    if (var_423) {
        // sp.point += dir * (0.5 * geom.margin)                                              <L 190>
        var_426 = &(var_geom.margin);
        var_428 = wp::load(var_426);
        var_427 = wp::mul(var_425, var_428);
        var_429 = wp::mul(var_dir, var_427);
        var_430 = &(var_0.point);
        var_432 = &(var_geom.margin);
        var_434 = wp::load(var_432);
        var_433 = wp::mul(var_431, var_434);
        var_435 = wp::mul(var_dir, var_433);
        var_437 = wp::load(var_430);
        var_436 = wp::add(var_437, var_435);
        var_438 = &(var_0.point);
        wp::store(var_438, var_436);
    }
    // return sp                                                                              <L 191>
    return var_0;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:265
static CUDA_CALLABLE wp::float32 _det3_0(
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> var_v3)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::float32 var_1;
    //---------
    // forward
    // def _det3(v1: wp.vec3, v2: wp.vec3, v3: wp.vec3) -> float:                             <L 266>
    // return wp.dot(v1, wp.cross(v2, v3))                                                    <L 267>
    var_0 = wp::cross(var_v2, var_v3);
    var_1 = wp::dot(var_v1, var_0);
    return var_1;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:270
static CUDA_CALLABLE wp::int32 _same_sign_0(
    wp::float32 var_a,
    wp::float32 var_b)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    bool var_1;
    const wp::float32 var_2 = 0.0;
    bool var_3;
    bool var_4;
    const wp::int32 var_5 = 1;
    const wp::float32 var_6 = 0.0;
    bool var_7;
    const wp::float32 var_8 = 0.0;
    bool var_9;
    bool var_10;
    const wp::int32 var_11 = 1;
    const wp::int32 var_12 = -1;
    const wp::int32 var_13 = 0;
    //---------
    // forward
    // def _same_sign(a: float, b: float) -> int:                                             <L 271>
    // if a > 0.0 and b > 0.0:                                                                <L 272>
    var_1 = (var_a > var_0);
    var_3 = (var_b > var_2);
    var_4 = var_1 && var_3;
    if (var_4) {
        // return 1                                                                           <L 273>
        return var_5;
    }
    // if a < 0.0 and b < 0.0:                                                                <L 274>
    var_7 = (var_a < var_6);
    var_9 = (var_b < var_8);
    var_10 = var_7 && var_9;
    if (var_10) {
        // return -1                                                                          <L 275>
        return var_12;
    }
    // return 0                                                                               <L 276>
    return var_13;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:286
static CUDA_CALLABLE void _project_origin_plane_0(
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> var_v3,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::int32 & ret_1)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::float32 var_8 = 0.0;
    bool var_9;
    const wp::int32 var_10 = 1;
    const wp::float32 var_11 = 0.0;
    bool var_12;
    const wp::float32 var_13 = 1e-15;
    bool var_14;
    bool var_15;
    wp::float32 var_16;
    wp::vec_t<3, wp::float32> var_17;
    const wp::int32 var_18 = 0;
    wp::vec_t<3, wp::float32> var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    const wp::float32 var_22 = 0.0;
    bool var_23;
    const wp::int32 var_24 = 1;
    const wp::float32 var_25 = 0.0;
    bool var_26;
    bool var_27;
    bool var_28;
    wp::float32 var_29;
    wp::vec_t<3, wp::float32> var_30;
    const wp::int32 var_31 = 0;
    wp::vec_t<3, wp::float32> var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    wp::float32 var_35;
    wp::vec_t<3, wp::float32> var_36;
    const wp::int32 var_37 = 0;
    //---------
    // forward
    // def _project_origin_plane(v1: wp.vec3, v2: wp.vec3, v3: wp.vec3) -> Tuple[wp.vec3, int]:       <L 287>
    // z = wp.vec3(0.0)                                                                       <L 288>
    var_1 = wp::vec_t<3, wp::float32>(var_0);
    // diff21 = v2 - v1                                                                       <L 289>
    var_2 = wp::sub(var_v2, var_v1);
    // diff31 = v3 - v1                                                                       <L 290>
    var_3 = wp::sub(var_v3, var_v1);
    // diff32 = v3 - v2                                                                       <L 291>
    var_4 = wp::sub(var_v3, var_v2);
    // n = wp.cross(diff32, diff21)                                                           <L 294>
    var_5 = wp::cross(var_4, var_2);
    // nv = wp.dot(n, v2)                                                                     <L 295>
    var_6 = wp::dot(var_5, var_v2);
    // nn = wp.dot(n, n)                                                                      <L 296>
    var_7 = wp::dot(var_5, var_5);
    // if nn == 0.0:                                                                          <L 297>
    var_9 = (var_7 == var_8);
    if (var_9) {
        // return z, 1                                                                        <L 298>
        ret_0 = var_1;
        ret_1 = var_10;
        return;
    }
    // if nv != 0.0 and nn > MINVAL:                                                          <L 299>
    var_12 = (var_6 != var_11);
    var_14 = (var_7 > var_13);
    var_15 = var_12 && var_14;
    if (var_15) {
        // return (nv / nn) * n, 0                                                            <L 300>
        var_16 = wp::div(var_6, var_7);
        var_17 = wp::mul(var_16, var_5);
        ret_0 = var_17;
        ret_1 = var_18;
        return;
    }
    // n = wp.cross(diff21, diff31)                                                           <L 303>
    var_19 = wp::cross(var_2, var_3);
    // nv = wp.dot(n, v1)                                                                     <L 304>
    var_20 = wp::dot(var_19, var_v1);
    // nn = wp.dot(n, n)                                                                      <L 305>
    var_21 = wp::dot(var_19, var_19);
    // if nn == 0.0:                                                                          <L 306>
    var_23 = (var_21 == var_22);
    if (var_23) {
        // return z, 1                                                                        <L 307>
        ret_0 = var_1;
        ret_1 = var_24;
        return;
    }
    // if nv != 0.0 and nn > MINVAL:                                                          <L 308>
    var_26 = (var_20 != var_25);
    var_27 = (var_21 > var_13);
    var_28 = var_26 && var_27;
    if (var_28) {
        // return (nv / nn) * n, 0                                                            <L 309>
        var_29 = wp::div(var_20, var_21);
        var_30 = wp::mul(var_29, var_19);
        ret_0 = var_30;
        ret_1 = var_31;
        return;
    }
    // n = wp.cross(diff31, diff32)                                                           <L 312>
    var_32 = wp::cross(var_3, var_4);
    // nv = wp.dot(n, v3)                                                                     <L 313>
    var_33 = wp::dot(var_32, var_v3);
    // nn = wp.dot(n, n)                                                                      <L 314>
    var_34 = wp::dot(var_32, var_32);
    // return (nv / nn) * n, 0                                                                <L 315>
    var_35 = wp::div(var_33, var_34);
    var_36 = wp::mul(var_35, var_32);
    ret_0 = var_36;
    ret_1 = var_37;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:279
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _project_origin_line_0(
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    //---------
    // forward
    // def _project_origin_line(v1: wp.vec3, v2: wp.vec3) -> wp.vec3:                         <L 280>
    // diff = v2 - v1                                                                         <L 281>
    var_0 = wp::sub(var_v2, var_v1);
    // scl = -(wp.dot(v2, diff) / wp.dot(diff, diff))                                         <L 282>
    var_1 = wp::dot(var_v2, var_0);
    var_2 = wp::dot(var_0, var_0);
    var_3 = wp::div(var_1, var_2);
    var_4 = wp::neg(var_3);
    // return v2 + scl * diff                                                                 <L 283>
    var_5 = wp::mul(var_4, var_0);
    var_6 = wp::add(var_v2, var_5);
    return var_6;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:539
static CUDA_CALLABLE wp::vec_t<2, wp::float32> _S1D_0(
    wp::vec_t<3, wp::float32> var_s1,
    wp::vec_t<3, wp::float32> var_s2)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    const wp::float32 var_1 = 0.0;
    const wp::int32 var_2 = 0;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    bool var_9;
    wp::float32 var_10;
    wp::int32 var_11;
    wp::float32 var_12;
    wp::int32 var_13;
    const wp::int32 var_14 = 1;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    bool var_20;
    wp::float32 var_21;
    wp::int32 var_22;
    wp::float32 var_23;
    wp::int32 var_24;
    const wp::int32 var_25 = 2;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    bool var_31;
    wp::float32 var_32;
    wp::int32 var_33;
    wp::float32 var_34;
    wp::int32 var_35;
    wp::float32 var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::int32 var_42;
    wp::int32 var_43;
    bool var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    wp::vec_t<2, wp::float32> var_47;
    const wp::float32 var_48 = 0.0;
    const wp::float32 var_49 = 1.0;
    wp::vec_t<2, wp::float32> var_50;
    //---------
    // forward
    // def _S1D(s1: wp.vec3, s2: wp.vec3) -> wp.vec2:                                         <L 540>
    // p_o = _project_origin_line(s1, s2)                                                     <L 542>
    var_0 = _project_origin_line_0(var_s1, var_s2);
    // mu_max = 0.0                                                                           <L 545>
    // index = 0                                                                              <L 546>
    // for i in range(3):                                                                     <L 547>
    // mu = s1[i] - s2[i]                                                                     <L 548>
    var_4 = wp::extract(var_s1, var_3);
    var_5 = wp::extract(var_s2, var_3);
    var_6 = wp::sub(var_4, var_5);
    // if wp.abs(mu) >= wp.abs(mu_max):                                                       <L 549>
    var_7 = wp::abs(var_6);
    var_8 = wp::abs(var_1);
    var_9 = (var_7 >= var_8);
    if (var_9) {
        // mu_max = mu                                                                        <L 550>
        var_10 = wp::copy(var_6);
        // index = i                                                                          <L 551>
        var_11 = wp::copy(var_3);
    }
    var_12 = wp::where(var_9, var_10, var_1);
    var_13 = wp::where(var_9, var_11, var_2);
    // mu = s1[i] - s2[i]                                                                     <L 548>
    var_15 = wp::extract(var_s1, var_14);
    var_16 = wp::extract(var_s2, var_14);
    var_17 = wp::sub(var_15, var_16);
    // if wp.abs(mu) >= wp.abs(mu_max):                                                       <L 549>
    var_18 = wp::abs(var_17);
    var_19 = wp::abs(var_12);
    var_20 = (var_18 >= var_19);
    if (var_20) {
        // mu_max = mu                                                                        <L 550>
        var_21 = wp::copy(var_17);
        // index = i                                                                          <L 551>
        var_22 = wp::copy(var_14);
    }
    var_23 = wp::where(var_20, var_21, var_12);
    var_24 = wp::where(var_20, var_22, var_13);
    // mu = s1[i] - s2[i]                                                                     <L 548>
    var_26 = wp::extract(var_s1, var_25);
    var_27 = wp::extract(var_s2, var_25);
    var_28 = wp::sub(var_26, var_27);
    // if wp.abs(mu) >= wp.abs(mu_max):                                                       <L 549>
    var_29 = wp::abs(var_28);
    var_30 = wp::abs(var_23);
    var_31 = (var_29 >= var_30);
    if (var_31) {
        // mu_max = mu                                                                        <L 550>
        var_32 = wp::copy(var_28);
        // index = i                                                                          <L 551>
        var_33 = wp::copy(var_25);
    }
    var_34 = wp::where(var_31, var_32, var_23);
    var_35 = wp::where(var_31, var_33, var_24);
    // C1 = p_o[index] - s2[index]                                                            <L 553>
    var_36 = wp::extract(var_0, var_35);
    var_37 = wp::extract(var_s2, var_35);
    var_38 = wp::sub(var_36, var_37);
    // C2 = s1[index] - p_o[index]                                                            <L 554>
    var_39 = wp::extract(var_s1, var_35);
    var_40 = wp::extract(var_0, var_35);
    var_41 = wp::sub(var_39, var_40);
    // if _same_sign(mu_max, C1) and _same_sign(mu_max, C2):                                  <L 557>
    var_42 = _same_sign_0(var_34, var_38);
    var_43 = _same_sign_0(var_34, var_41);
    var_44 = var_42 && var_43;
    if (var_44) {
        // return wp.vec2(C1 / mu_max, C2 / mu_max)                                           <L 558>
        var_45 = wp::div(var_38, var_34);
        var_46 = wp::div(var_41, var_34);
        var_47 = wp::vec_t<2, wp::float32>(var_45, var_46);
        return var_47;
    }
    // return wp.vec2(0.0, 1.0)                                                               <L 559>
    var_50 = wp::vec_t<2, wp::float32>(var_48, var_49);
    return var_50;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:394
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _S2D_0(
    wp::vec_t<3, wp::float32> var_s1,
    wp::vec_t<3, wp::float32> var_s2,
    wp::vec_t<3, wp::float32> var_s3)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::int32 var_1;
    wp::vec_t<2, wp::float32> var_2;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    const wp::int32 var_5 = 1;
    wp::float32 var_6;
    const wp::float32 var_7 = 0.0;
    wp::vec_t<3, wp::float32> var_8;
    const wp::int32 var_9 = 1;
    wp::float32 var_10;
    const wp::int32 var_11 = 2;
    wp::float32 var_12;
    wp::float32 var_13;
    const wp::int32 var_14 = 2;
    wp::float32 var_15;
    const wp::int32 var_16 = 1;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    const wp::int32 var_20 = 1;
    wp::float32 var_21;
    const wp::int32 var_22 = 2;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    const wp::int32 var_26 = 2;
    wp::float32 var_27;
    const wp::int32 var_28 = 1;
    wp::float32 var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    const wp::int32 var_32 = 1;
    wp::float32 var_33;
    const wp::int32 var_34 = 2;
    wp::float32 var_35;
    wp::float32 var_36;
    wp::float32 var_37;
    const wp::int32 var_38 = 2;
    wp::float32 var_39;
    const wp::int32 var_40 = 1;
    wp::float32 var_41;
    wp::float32 var_42;
    wp::float32 var_43;
    const wp::int32 var_44 = 0;
    wp::float32 var_45;
    const wp::int32 var_46 = 2;
    wp::float32 var_47;
    wp::float32 var_48;
    const wp::int32 var_49 = 2;
    wp::float32 var_50;
    const wp::int32 var_51 = 0;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    const wp::int32 var_55 = 0;
    wp::float32 var_56;
    const wp::int32 var_57 = 2;
    wp::float32 var_58;
    wp::float32 var_59;
    wp::float32 var_60;
    const wp::int32 var_61 = 2;
    wp::float32 var_62;
    const wp::int32 var_63 = 0;
    wp::float32 var_64;
    wp::float32 var_65;
    wp::float32 var_66;
    const wp::int32 var_67 = 0;
    wp::float32 var_68;
    const wp::int32 var_69 = 2;
    wp::float32 var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    const wp::int32 var_73 = 2;
    wp::float32 var_74;
    const wp::int32 var_75 = 0;
    wp::float32 var_76;
    wp::float32 var_77;
    wp::float32 var_78;
    const wp::int32 var_79 = 0;
    wp::float32 var_80;
    const wp::int32 var_81 = 1;
    wp::float32 var_82;
    wp::float32 var_83;
    const wp::int32 var_84 = 1;
    wp::float32 var_85;
    const wp::int32 var_86 = 0;
    wp::float32 var_87;
    wp::float32 var_88;
    wp::float32 var_89;
    const wp::int32 var_90 = 0;
    wp::float32 var_91;
    const wp::int32 var_92 = 1;
    wp::float32 var_93;
    wp::float32 var_94;
    wp::float32 var_95;
    const wp::int32 var_96 = 1;
    wp::float32 var_97;
    const wp::int32 var_98 = 0;
    wp::float32 var_99;
    wp::float32 var_100;
    wp::float32 var_101;
    const wp::int32 var_102 = 0;
    wp::float32 var_103;
    const wp::int32 var_104 = 1;
    wp::float32 var_105;
    wp::float32 var_106;
    wp::float32 var_107;
    const wp::int32 var_108 = 1;
    wp::float32 var_109;
    const wp::int32 var_110 = 0;
    wp::float32 var_111;
    wp::float32 var_112;
    wp::float32 var_113;
    const wp::float32 var_114 = 0.0;
    const wp::float32 var_115 = 0.0;
    wp::vec_t<2, wp::float32> var_116;
    const wp::float32 var_117 = 0.0;
    wp::vec_t<2, wp::float32> var_118;
    const wp::float32 var_119 = 0.0;
    wp::vec_t<2, wp::float32> var_120;
    const wp::float32 var_121 = 0.0;
    wp::vec_t<2, wp::float32> var_122;
    wp::float32 var_123;
    wp::float32 var_124;
    wp::float32 var_125;
    bool var_126;
    bool var_127;
    bool var_128;
    wp::float32 var_129;
    const wp::int32 var_130 = 1;
    wp::float32 var_131;
    const wp::int32 var_132 = 0;
    const wp::int32 var_133 = 2;
    wp::float32 var_134;
    const wp::int32 var_135 = 1;
    const wp::int32 var_136 = 1;
    wp::float32 var_137;
    const wp::int32 var_138 = 0;
    const wp::int32 var_139 = 2;
    wp::float32 var_140;
    const wp::int32 var_141 = 1;
    const wp::int32 var_142 = 1;
    wp::float32 var_143;
    const wp::int32 var_144 = 0;
    const wp::int32 var_145 = 2;
    wp::float32 var_146;
    const wp::int32 var_147 = 1;
    const wp::int32 var_148 = 1;
    wp::float32 var_149;
    const wp::int32 var_150 = 0;
    const wp::int32 var_151 = 2;
    wp::float32 var_152;
    const wp::int32 var_153 = 1;
    wp::float32 var_154;
    bool var_155;
    wp::float32 var_156;
    const wp::int32 var_157 = 0;
    wp::float32 var_158;
    const wp::int32 var_159 = 0;
    const wp::int32 var_160 = 2;
    wp::float32 var_161;
    const wp::int32 var_162 = 1;
    const wp::int32 var_163 = 0;
    wp::float32 var_164;
    const wp::int32 var_165 = 0;
    const wp::int32 var_166 = 2;
    wp::float32 var_167;
    const wp::int32 var_168 = 1;
    const wp::int32 var_169 = 0;
    wp::float32 var_170;
    const wp::int32 var_171 = 0;
    const wp::int32 var_172 = 2;
    wp::float32 var_173;
    const wp::int32 var_174 = 1;
    const wp::int32 var_175 = 0;
    wp::float32 var_176;
    const wp::int32 var_177 = 0;
    const wp::int32 var_178 = 2;
    wp::float32 var_179;
    const wp::int32 var_180 = 1;
    wp::float32 var_181;
    wp::float32 var_182;
    const wp::int32 var_183 = 0;
    wp::float32 var_184;
    const wp::int32 var_185 = 0;
    const wp::int32 var_186 = 1;
    wp::float32 var_187;
    const wp::int32 var_188 = 1;
    const wp::int32 var_189 = 0;
    wp::float32 var_190;
    const wp::int32 var_191 = 0;
    const wp::int32 var_192 = 1;
    wp::float32 var_193;
    const wp::int32 var_194 = 1;
    const wp::int32 var_195 = 0;
    wp::float32 var_196;
    const wp::int32 var_197 = 0;
    const wp::int32 var_198 = 1;
    wp::float32 var_199;
    const wp::int32 var_200 = 1;
    const wp::int32 var_201 = 0;
    wp::float32 var_202;
    const wp::int32 var_203 = 0;
    const wp::int32 var_204 = 1;
    wp::float32 var_205;
    const wp::int32 var_206 = 1;
    wp::float32 var_207;
    wp::float32 var_208;
    const wp::int32 var_209 = 0;
    wp::float32 var_210;
    const wp::int32 var_211 = 1;
    wp::float32 var_212;
    wp::float32 var_213;
    const wp::int32 var_214 = 1;
    wp::float32 var_215;
    const wp::int32 var_216 = 0;
    wp::float32 var_217;
    wp::float32 var_218;
    wp::float32 var_219;
    const wp::int32 var_220 = 0;
    wp::float32 var_221;
    const wp::int32 var_222 = 1;
    wp::float32 var_223;
    wp::float32 var_224;
    wp::float32 var_225;
    const wp::int32 var_226 = 0;
    wp::float32 var_227;
    const wp::int32 var_228 = 1;
    wp::float32 var_229;
    wp::float32 var_230;
    wp::float32 var_231;
    const wp::int32 var_232 = 1;
    wp::float32 var_233;
    const wp::int32 var_234 = 0;
    wp::float32 var_235;
    wp::float32 var_236;
    wp::float32 var_237;
    const wp::int32 var_238 = 0;
    wp::float32 var_239;
    const wp::int32 var_240 = 1;
    wp::float32 var_241;
    wp::float32 var_242;
    wp::float32 var_243;
    const wp::int32 var_244 = 0;
    wp::float32 var_245;
    const wp::int32 var_246 = 1;
    wp::float32 var_247;
    wp::float32 var_248;
    const wp::int32 var_249 = 1;
    wp::float32 var_250;
    const wp::int32 var_251 = 0;
    wp::float32 var_252;
    wp::float32 var_253;
    wp::float32 var_254;
    const wp::int32 var_255 = 0;
    wp::float32 var_256;
    const wp::int32 var_257 = 1;
    wp::float32 var_258;
    wp::float32 var_259;
    wp::float32 var_260;
    const wp::int32 var_261 = 0;
    wp::float32 var_262;
    const wp::int32 var_263 = 1;
    wp::float32 var_264;
    wp::float32 var_265;
    wp::float32 var_266;
    const wp::int32 var_267 = 1;
    wp::float32 var_268;
    const wp::int32 var_269 = 0;
    wp::float32 var_270;
    wp::float32 var_271;
    wp::float32 var_272;
    const wp::int32 var_273 = 0;
    wp::float32 var_274;
    const wp::int32 var_275 = 1;
    wp::float32 var_276;
    wp::float32 var_277;
    wp::float32 var_278;
    const wp::int32 var_279 = 0;
    wp::float32 var_280;
    const wp::int32 var_281 = 1;
    wp::float32 var_282;
    wp::float32 var_283;
    const wp::int32 var_284 = 1;
    wp::float32 var_285;
    const wp::int32 var_286 = 0;
    wp::float32 var_287;
    wp::float32 var_288;
    wp::float32 var_289;
    const wp::int32 var_290 = 0;
    wp::float32 var_291;
    const wp::int32 var_292 = 1;
    wp::float32 var_293;
    wp::float32 var_294;
    wp::float32 var_295;
    const wp::int32 var_296 = 0;
    wp::float32 var_297;
    const wp::int32 var_298 = 1;
    wp::float32 var_299;
    wp::float32 var_300;
    wp::float32 var_301;
    const wp::int32 var_302 = 1;
    wp::float32 var_303;
    const wp::int32 var_304 = 0;
    wp::float32 var_305;
    wp::float32 var_306;
    wp::float32 var_307;
    const wp::int32 var_308 = 0;
    wp::float32 var_309;
    const wp::int32 var_310 = 1;
    wp::float32 var_311;
    wp::float32 var_312;
    wp::float32 var_313;
    wp::int32 var_314;
    wp::int32 var_315;
    wp::int32 var_316;
    bool var_317;
    wp::float32 var_318;
    wp::float32 var_319;
    wp::float32 var_320;
    wp::vec_t<3, wp::float32> var_321;
    const wp::float32 var_322 = 1e+30;
    wp::float32 var_323;
    const wp::float32 var_324 = 0.0;
    const wp::float32 var_325 = 0.0;
    const wp::float32 var_326 = 0.0;
    wp::vec_t<3, wp::float32> var_327;
    bool var_328;
    wp::vec_t<2, wp::float32> var_329;
    const wp::int32 var_330 = 0;
    wp::float32 var_331;
    wp::vec_t<3, wp::float32> var_332;
    const wp::int32 var_333 = 1;
    wp::float32 var_334;
    wp::vec_t<3, wp::float32> var_335;
    wp::vec_t<3, wp::float32> var_336;
    wp::float32 var_337;
    const wp::float32 var_338 = 0.0;
    const wp::int32 var_339 = 0;
    const wp::int32 var_340 = 0;
    wp::float32 var_341;
    const wp::int32 var_342 = 1;
    const wp::int32 var_343 = 1;
    wp::float32 var_344;
    const wp::int32 var_345 = 2;
    wp::float32 var_346;
    wp::float32 var_347;
    bool var_348;
    wp::vec_t<2, wp::float32> var_349;
    const wp::int32 var_350 = 0;
    wp::float32 var_351;
    wp::vec_t<3, wp::float32> var_352;
    const wp::int32 var_353 = 1;
    wp::float32 var_354;
    wp::vec_t<3, wp::float32> var_355;
    wp::vec_t<3, wp::float32> var_356;
    wp::float32 var_357;
    bool var_358;
    const wp::int32 var_359 = 0;
    wp::float32 var_360;
    const wp::int32 var_361 = 0;
    const wp::float32 var_362 = 0.0;
    const wp::int32 var_363 = 1;
    const wp::int32 var_364 = 1;
    wp::float32 var_365;
    const wp::int32 var_366 = 2;
    wp::float32 var_367;
    wp::float32 var_368;
    wp::float32 var_369;
    wp::vec_t<2, wp::float32> var_370;
    wp::vec_t<3, wp::float32> var_371;
    wp::float32 var_372;
    bool var_373;
    wp::vec_t<2, wp::float32> var_374;
    const wp::int32 var_375 = 0;
    wp::float32 var_376;
    wp::vec_t<3, wp::float32> var_377;
    const wp::int32 var_378 = 1;
    wp::float32 var_379;
    wp::vec_t<3, wp::float32> var_380;
    wp::vec_t<3, wp::float32> var_381;
    wp::float32 var_382;
    bool var_383;
    const wp::int32 var_384 = 0;
    wp::float32 var_385;
    const wp::int32 var_386 = 0;
    const wp::int32 var_387 = 1;
    wp::float32 var_388;
    const wp::int32 var_389 = 1;
    const wp::float32 var_390 = 0.0;
    const wp::int32 var_391 = 2;
    wp::vec_t<2, wp::float32> var_392;
    wp::vec_t<3, wp::float32> var_393;
    wp::float32 var_394;
    //---------
    // forward
    // def _S2D(s1: wp.vec3, s2: wp.vec3, s3: wp.vec3) -> wp.vec3:                            <L 395>
    // p_o, ret = _project_origin_plane(s1, s2, s3)                                           <L 397>
    _project_origin_plane_0(var_s1, var_s2, var_s3, var_0, var_1);
    // if ret:                                                                                <L 398>
    if (var_1) {
        // v = _S1D(s1, s2)                                                                   <L 399>
        var_2 = _S1D_0(var_s1, var_s2);
        // return wp.vec3(v[0], v[1], 0.0)                                                    <L 400>
        var_4 = wp::extract(var_2, var_3);
        var_6 = wp::extract(var_2, var_5);
        var_8 = wp::vec_t<3, wp::float32>(var_4, var_6, var_7);
        return var_8;
    }
    // M_14 = s2[1] * s3[2] - s2[2] * s3[1] - s1[1] * s3[2] + s1[2] * s3[1] + s1[1] * s2[2] - s1[2] * s2[1]       <L 407>
    var_10 = wp::extract(var_s2, var_9);
    var_12 = wp::extract(var_s3, var_11);
    var_13 = wp::mul(var_10, var_12);
    var_15 = wp::extract(var_s2, var_14);
    var_17 = wp::extract(var_s3, var_16);
    var_18 = wp::mul(var_15, var_17);
    var_19 = wp::sub(var_13, var_18);
    var_21 = wp::extract(var_s1, var_20);
    var_23 = wp::extract(var_s3, var_22);
    var_24 = wp::mul(var_21, var_23);
    var_25 = wp::sub(var_19, var_24);
    var_27 = wp::extract(var_s1, var_26);
    var_29 = wp::extract(var_s3, var_28);
    var_30 = wp::mul(var_27, var_29);
    var_31 = wp::add(var_25, var_30);
    var_33 = wp::extract(var_s1, var_32);
    var_35 = wp::extract(var_s2, var_34);
    var_36 = wp::mul(var_33, var_35);
    var_37 = wp::add(var_31, var_36);
    var_39 = wp::extract(var_s1, var_38);
    var_41 = wp::extract(var_s2, var_40);
    var_42 = wp::mul(var_39, var_41);
    var_43 = wp::sub(var_37, var_42);
    // M_24 = s2[0] * s3[2] - s2[2] * s3[0] - s1[0] * s3[2] + s1[2] * s3[0] + s1[0] * s2[2] - s1[2] * s2[0]       <L 408>
    var_45 = wp::extract(var_s2, var_44);
    var_47 = wp::extract(var_s3, var_46);
    var_48 = wp::mul(var_45, var_47);
    var_50 = wp::extract(var_s2, var_49);
    var_52 = wp::extract(var_s3, var_51);
    var_53 = wp::mul(var_50, var_52);
    var_54 = wp::sub(var_48, var_53);
    var_56 = wp::extract(var_s1, var_55);
    var_58 = wp::extract(var_s3, var_57);
    var_59 = wp::mul(var_56, var_58);
    var_60 = wp::sub(var_54, var_59);
    var_62 = wp::extract(var_s1, var_61);
    var_64 = wp::extract(var_s3, var_63);
    var_65 = wp::mul(var_62, var_64);
    var_66 = wp::add(var_60, var_65);
    var_68 = wp::extract(var_s1, var_67);
    var_70 = wp::extract(var_s2, var_69);
    var_71 = wp::mul(var_68, var_70);
    var_72 = wp::add(var_66, var_71);
    var_74 = wp::extract(var_s1, var_73);
    var_76 = wp::extract(var_s2, var_75);
    var_77 = wp::mul(var_74, var_76);
    var_78 = wp::sub(var_72, var_77);
    // M_34 = s2[0] * s3[1] - s2[1] * s3[0] - s1[0] * s3[1] + s1[1] * s3[0] + s1[0] * s2[1] - s1[1] * s2[0]       <L 409>
    var_80 = wp::extract(var_s2, var_79);
    var_82 = wp::extract(var_s3, var_81);
    var_83 = wp::mul(var_80, var_82);
    var_85 = wp::extract(var_s2, var_84);
    var_87 = wp::extract(var_s3, var_86);
    var_88 = wp::mul(var_85, var_87);
    var_89 = wp::sub(var_83, var_88);
    var_91 = wp::extract(var_s1, var_90);
    var_93 = wp::extract(var_s3, var_92);
    var_94 = wp::mul(var_91, var_93);
    var_95 = wp::sub(var_89, var_94);
    var_97 = wp::extract(var_s1, var_96);
    var_99 = wp::extract(var_s3, var_98);
    var_100 = wp::mul(var_97, var_99);
    var_101 = wp::add(var_95, var_100);
    var_103 = wp::extract(var_s1, var_102);
    var_105 = wp::extract(var_s2, var_104);
    var_106 = wp::mul(var_103, var_105);
    var_107 = wp::add(var_101, var_106);
    var_109 = wp::extract(var_s1, var_108);
    var_111 = wp::extract(var_s2, var_110);
    var_112 = wp::mul(var_109, var_111);
    var_113 = wp::sub(var_107, var_112);
    // M_max = 0.0                                                                            <L 412>
    // s1_2D = wp.vec2(0.0)                                                                   <L 413>
    var_116 = wp::vec_t<2, wp::float32>(var_115);
    // s2_2D = wp.vec2(0.0)                                                                   <L 414>
    var_118 = wp::vec_t<2, wp::float32>(var_117);
    // s3_2D = wp.vec2(0.0)                                                                   <L 415>
    var_120 = wp::vec_t<2, wp::float32>(var_119);
    // p_o_2D = wp.vec2(0.0)                                                                  <L 416>
    var_122 = wp::vec_t<2, wp::float32>(var_121);
    // mu1 = wp.abs(M_14)                                                                     <L 418>
    var_123 = wp::abs(var_43);
    // mu2 = wp.abs(M_24)                                                                     <L 419>
    var_124 = wp::abs(var_78);
    // mu3 = wp.abs(M_34)                                                                     <L 420>
    var_125 = wp::abs(var_113);
    // if mu1 >= mu2 and mu1 >= mu3:                                                          <L 422>
    var_126 = (var_123 >= var_124);
    var_127 = (var_123 >= var_125);
    var_128 = var_126 && var_127;
    if (var_128) {
        // M_max = M_14                                                                       <L 423>
        var_129 = wp::copy(var_43);
        // s1_2D[0] = s1[1]                                                                   <L 424>
        var_131 = wp::extract(var_s1, var_130);
        wp::assign_inplace(var_116, var_132, var_131);
        // s1_2D[1] = s1[2]                                                                   <L 425>
        var_134 = wp::extract(var_s1, var_133);
        wp::assign_inplace(var_116, var_135, var_134);
        // s2_2D[0] = s2[1]                                                                   <L 427>
        var_137 = wp::extract(var_s2, var_136);
        wp::assign_inplace(var_118, var_138, var_137);
        // s2_2D[1] = s2[2]                                                                   <L 428>
        var_140 = wp::extract(var_s2, var_139);
        wp::assign_inplace(var_118, var_141, var_140);
        // s3_2D[0] = s3[1]                                                                   <L 430>
        var_143 = wp::extract(var_s3, var_142);
        wp::assign_inplace(var_120, var_144, var_143);
        // s3_2D[1] = s3[2]                                                                   <L 431>
        var_146 = wp::extract(var_s3, var_145);
        wp::assign_inplace(var_120, var_147, var_146);
        // p_o_2D[0] = p_o[1]                                                                 <L 433>
        var_149 = wp::extract(var_0, var_148);
        wp::assign_inplace(var_122, var_150, var_149);
        // p_o_2D[1] = p_o[2]                                                                 <L 434>
        var_152 = wp::extract(var_0, var_151);
        wp::assign_inplace(var_122, var_153, var_152);
    }
    var_154 = wp::where(var_128, var_129, var_114);
    if (!var_128) {
        // elif mu2 >= mu3:                                                                   <L 435>
        var_155 = (var_124 >= var_125);
        if (var_155) {
            // M_max = M_24                                                                   <L 436>
            var_156 = wp::copy(var_78);
            // s1_2D[0] = s1[0]                                                               <L 437>
            var_158 = wp::extract(var_s1, var_157);
            wp::assign_inplace(var_116, var_159, var_158);
            // s1_2D[1] = s1[2]                                                               <L 438>
            var_161 = wp::extract(var_s1, var_160);
            wp::assign_inplace(var_116, var_162, var_161);
            // s2_2D[0] = s2[0]                                                               <L 440>
            var_164 = wp::extract(var_s2, var_163);
            wp::assign_inplace(var_118, var_165, var_164);
            // s2_2D[1] = s2[2]                                                               <L 441>
            var_167 = wp::extract(var_s2, var_166);
            wp::assign_inplace(var_118, var_168, var_167);
            // s3_2D[0] = s3[0]                                                               <L 443>
            var_170 = wp::extract(var_s3, var_169);
            wp::assign_inplace(var_120, var_171, var_170);
            // s3_2D[1] = s3[2]                                                               <L 444>
            var_173 = wp::extract(var_s3, var_172);
            wp::assign_inplace(var_120, var_174, var_173);
            // p_o_2D[0] = p_o[0]                                                             <L 446>
            var_176 = wp::extract(var_0, var_175);
            wp::assign_inplace(var_122, var_177, var_176);
            // p_o_2D[1] = p_o[2]                                                             <L 447>
            var_179 = wp::extract(var_0, var_178);
            wp::assign_inplace(var_122, var_180, var_179);
        }
        var_181 = wp::where(var_155, var_156, var_154);
        if (!var_155) {
            // M_max = M_34                                                                   <L 449>
            var_182 = wp::copy(var_113);
            // s1_2D[0] = s1[0]                                                               <L 450>
            var_184 = wp::extract(var_s1, var_183);
            wp::assign_inplace(var_116, var_185, var_184);
            // s1_2D[1] = s1[1]                                                               <L 451>
            var_187 = wp::extract(var_s1, var_186);
            wp::assign_inplace(var_116, var_188, var_187);
            // s2_2D[0] = s2[0]                                                               <L 453>
            var_190 = wp::extract(var_s2, var_189);
            wp::assign_inplace(var_118, var_191, var_190);
            // s2_2D[1] = s2[1]                                                               <L 454>
            var_193 = wp::extract(var_s2, var_192);
            wp::assign_inplace(var_118, var_194, var_193);
            // s3_2D[0] = s3[0]                                                               <L 456>
            var_196 = wp::extract(var_s3, var_195);
            wp::assign_inplace(var_120, var_197, var_196);
            // s3_2D[1] = s3[1]                                                               <L 457>
            var_199 = wp::extract(var_s3, var_198);
            wp::assign_inplace(var_120, var_200, var_199);
            // p_o_2D[0] = p_o[0]                                                             <L 459>
            var_202 = wp::extract(var_0, var_201);
            wp::assign_inplace(var_122, var_203, var_202);
            // p_o_2D[1] = p_o[1]                                                             <L 460>
            var_205 = wp::extract(var_0, var_204);
            wp::assign_inplace(var_122, var_206, var_205);
        }
        var_207 = wp::where(var_155, var_181, var_182);
    }
    var_208 = wp::where(var_128, var_154, var_207);
    // C31 = (                                                                                <L 468>
    // p_o_2D[0] * s2_2D[1]                                                                   <L 469>
    var_210 = wp::extract(var_122, var_209);
    var_212 = wp::extract(var_118, var_211);
    var_213 = wp::mul(var_210, var_212);
    // + p_o_2D[1] * s3_2D[0]                                                                 <L 470>
    var_215 = wp::extract(var_122, var_214);
    var_217 = wp::extract(var_120, var_216);
    var_218 = wp::mul(var_215, var_217);
    var_219 = wp::add(var_213, var_218);
    // + s2_2D[0] * s3_2D[1]                                                                  <L 471>
    var_221 = wp::extract(var_118, var_220);
    var_223 = wp::extract(var_120, var_222);
    var_224 = wp::mul(var_221, var_223);
    var_225 = wp::add(var_219, var_224);
    // - p_o_2D[0] * s3_2D[1]                                                                 <L 472>
    var_227 = wp::extract(var_122, var_226);
    var_229 = wp::extract(var_120, var_228);
    var_230 = wp::mul(var_227, var_229);
    var_231 = wp::sub(var_225, var_230);
    // - p_o_2D[1] * s2_2D[0]                                                                 <L 473>
    var_233 = wp::extract(var_122, var_232);
    var_235 = wp::extract(var_118, var_234);
    var_236 = wp::mul(var_233, var_235);
    var_237 = wp::sub(var_231, var_236);
    // - s3_2D[0] * s2_2D[1]                                                                  <L 474>
    var_239 = wp::extract(var_120, var_238);
    var_241 = wp::extract(var_118, var_240);
    var_242 = wp::mul(var_239, var_241);
    var_243 = wp::sub(var_237, var_242);
    // C32 = (                                                                                <L 478>
    // p_o_2D[0] * s3_2D[1]                                                                   <L 479>
    var_245 = wp::extract(var_122, var_244);
    var_247 = wp::extract(var_120, var_246);
    var_248 = wp::mul(var_245, var_247);
    // + p_o_2D[1] * s1_2D[0]                                                                 <L 480>
    var_250 = wp::extract(var_122, var_249);
    var_252 = wp::extract(var_116, var_251);
    var_253 = wp::mul(var_250, var_252);
    var_254 = wp::add(var_248, var_253);
    // + s3_2D[0] * s1_2D[1]                                                                  <L 481>
    var_256 = wp::extract(var_120, var_255);
    var_258 = wp::extract(var_116, var_257);
    var_259 = wp::mul(var_256, var_258);
    var_260 = wp::add(var_254, var_259);
    // - p_o_2D[0] * s1_2D[1]                                                                 <L 482>
    var_262 = wp::extract(var_122, var_261);
    var_264 = wp::extract(var_116, var_263);
    var_265 = wp::mul(var_262, var_264);
    var_266 = wp::sub(var_260, var_265);
    // - p_o_2D[1] * s3_2D[0]                                                                 <L 483>
    var_268 = wp::extract(var_122, var_267);
    var_270 = wp::extract(var_120, var_269);
    var_271 = wp::mul(var_268, var_270);
    var_272 = wp::sub(var_266, var_271);
    // - s1_2D[0] * s3_2D[1]                                                                  <L 484>
    var_274 = wp::extract(var_116, var_273);
    var_276 = wp::extract(var_120, var_275);
    var_277 = wp::mul(var_274, var_276);
    var_278 = wp::sub(var_272, var_277);
    // C33 = (                                                                                <L 488>
    // p_o_2D[0] * s1_2D[1]                                                                   <L 489>
    var_280 = wp::extract(var_122, var_279);
    var_282 = wp::extract(var_116, var_281);
    var_283 = wp::mul(var_280, var_282);
    // + p_o_2D[1] * s2_2D[0]                                                                 <L 490>
    var_285 = wp::extract(var_122, var_284);
    var_287 = wp::extract(var_118, var_286);
    var_288 = wp::mul(var_285, var_287);
    var_289 = wp::add(var_283, var_288);
    // + s1_2D[0] * s2_2D[1]                                                                  <L 491>
    var_291 = wp::extract(var_116, var_290);
    var_293 = wp::extract(var_118, var_292);
    var_294 = wp::mul(var_291, var_293);
    var_295 = wp::add(var_289, var_294);
    // - p_o_2D[0] * s2_2D[1]                                                                 <L 492>
    var_297 = wp::extract(var_122, var_296);
    var_299 = wp::extract(var_118, var_298);
    var_300 = wp::mul(var_297, var_299);
    var_301 = wp::sub(var_295, var_300);
    // - p_o_2D[1] * s1_2D[0]                                                                 <L 493>
    var_303 = wp::extract(var_122, var_302);
    var_305 = wp::extract(var_116, var_304);
    var_306 = wp::mul(var_303, var_305);
    var_307 = wp::sub(var_301, var_306);
    // - s2_2D[0] * s1_2D[1]                                                                  <L 494>
    var_309 = wp::extract(var_118, var_308);
    var_311 = wp::extract(var_116, var_310);
    var_312 = wp::mul(var_309, var_311);
    var_313 = wp::sub(var_307, var_312);
    // comp1 = _same_sign(M_max, C31)                                                         <L 497>
    var_314 = _same_sign_0(var_208, var_243);
    // comp2 = _same_sign(M_max, C32)                                                         <L 498>
    var_315 = _same_sign_0(var_208, var_278);
    // comp3 = _same_sign(M_max, C33)                                                         <L 499>
    var_316 = _same_sign_0(var_208, var_313);
    // if comp1 and comp2 and comp3:                                                          <L 502>
    var_317 = var_314 && var_315 && var_316;
    if (var_317) {
        // return wp.vec3(C31 / M_max, C32 / M_max, C33 / M_max)                              <L 503>
        var_318 = wp::div(var_243, var_208);
        var_319 = wp::div(var_278, var_208);
        var_320 = wp::div(var_313, var_208);
        var_321 = wp::vec_t<3, wp::float32>(var_318, var_319, var_320);
        return var_321;
    }
    // dmin = FLOAT_MAX                                                                       <L 506>
    var_323 = wp::copy(var_322);
    // coordinates = wp.vec3(0.0, 0.0, 0.0)                                                   <L 507>
    var_327 = wp::vec_t<3, wp::float32>(var_324, var_325, var_326);
    // if not comp1:                                                                          <L 509>
    var_328 = wp::unot(var_314);
    if (var_328) {
        // subcoord = _S1D(s2, s3)                                                            <L 510>
        var_329 = _S1D_0(var_s2, var_s3);
        // x = subcoord[0] * s2 + subcoord[1] * s3                                            <L 511>
        var_331 = wp::extract(var_329, var_330);
        var_332 = wp::mul(var_331, var_s2);
        var_334 = wp::extract(var_329, var_333);
        var_335 = wp::mul(var_334, var_s3);
        var_336 = wp::add(var_332, var_335);
        // d = wp.dot(x, x)                                                                   <L 512>
        var_337 = wp::dot(var_336, var_336);
        // coordinates[0] = 0.0                                                               <L 513>
        wp::assign_inplace(var_327, var_339, var_338);
        // coordinates[1] = subcoord[0]                                                       <L 514>
        var_341 = wp::extract(var_329, var_340);
        wp::assign_inplace(var_327, var_342, var_341);
        // coordinates[2] = subcoord[1]                                                       <L 515>
        var_344 = wp::extract(var_329, var_343);
        wp::assign_inplace(var_327, var_345, var_344);
        // dmin = d                                                                           <L 516>
        var_346 = wp::copy(var_337);
    }
    var_347 = wp::where(var_328, var_346, var_323);
    // if not comp2:                                                                          <L 518>
    var_348 = wp::unot(var_315);
    if (var_348) {
        // subcoord = _S1D(s1, s3)                                                            <L 519>
        var_349 = _S1D_0(var_s1, var_s3);
        // x = subcoord[0] * s1 + subcoord[1] * s3                                            <L 520>
        var_351 = wp::extract(var_349, var_350);
        var_352 = wp::mul(var_351, var_s1);
        var_354 = wp::extract(var_349, var_353);
        var_355 = wp::mul(var_354, var_s3);
        var_356 = wp::add(var_352, var_355);
        // d = wp.dot(x, x)                                                                   <L 521>
        var_357 = wp::dot(var_356, var_356);
        // if d < dmin:                                                                       <L 522>
        var_358 = (var_357 < var_347);
        if (var_358) {
            // coordinates[0] = subcoord[0]                                                   <L 523>
            var_360 = wp::extract(var_349, var_359);
            wp::assign_inplace(var_327, var_361, var_360);
            // coordinates[1] = 0.0                                                           <L 524>
            wp::assign_inplace(var_327, var_363, var_362);
            // coordinates[2] = subcoord[1]                                                   <L 525>
            var_365 = wp::extract(var_349, var_364);
            wp::assign_inplace(var_327, var_366, var_365);
            // dmin = d                                                                       <L 526>
            var_367 = wp::copy(var_357);
        }
        var_368 = wp::where(var_358, var_367, var_347);
    }
    var_369 = wp::where(var_348, var_368, var_347);
    var_370 = wp::where(var_348, var_349, var_329);
    var_371 = wp::where(var_348, var_356, var_336);
    var_372 = wp::where(var_348, var_357, var_337);
    // if not comp3:                                                                          <L 528>
    var_373 = wp::unot(var_316);
    if (var_373) {
        // subcoord = _S1D(s1, s2)                                                            <L 529>
        var_374 = _S1D_0(var_s1, var_s2);
        // x = subcoord[0] * s1 + subcoord[1] * s2                                            <L 530>
        var_376 = wp::extract(var_374, var_375);
        var_377 = wp::mul(var_376, var_s1);
        var_379 = wp::extract(var_374, var_378);
        var_380 = wp::mul(var_379, var_s2);
        var_381 = wp::add(var_377, var_380);
        // d = wp.dot(x, x)                                                                   <L 531>
        var_382 = wp::dot(var_381, var_381);
        // if d < dmin:                                                                       <L 532>
        var_383 = (var_382 < var_369);
        if (var_383) {
            // coordinates[0] = subcoord[0]                                                   <L 533>
            var_385 = wp::extract(var_374, var_384);
            wp::assign_inplace(var_327, var_386, var_385);
            // coordinates[1] = subcoord[1]                                                   <L 534>
            var_388 = wp::extract(var_374, var_387);
            wp::assign_inplace(var_327, var_389, var_388);
            // coordinates[2] = 0.0                                                           <L 535>
            wp::assign_inplace(var_327, var_391, var_390);
        }
    }
    var_392 = wp::where(var_373, var_374, var_370);
    var_393 = wp::where(var_373, var_381, var_371);
    var_394 = wp::where(var_373, var_382, var_372);
    // return coordinates                                                                     <L 536>
    return var_327;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:318
static CUDA_CALLABLE wp::vec_t<4, wp::float32> _S3D_0(
    wp::vec_t<3, wp::float32> var_s1,
    wp::vec_t<3, wp::float32> var_s2,
    wp::vec_t<3, wp::float32> var_s3,
    wp::vec_t<3, wp::float32> var_s4)
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
    wp::float32 var_8;
    wp::int32 var_9;
    wp::int32 var_10;
    wp::int32 var_11;
    wp::int32 var_12;
    bool var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::vec_t<4, wp::float32> var_18;
    const wp::float32 var_19 = 0.0;
    const wp::float32 var_20 = 0.0;
    const wp::float32 var_21 = 0.0;
    const wp::float32 var_22 = 0.0;
    wp::vec_t<4, wp::float32> var_23;
    const wp::float32 var_24 = 1e+30;
    wp::float32 var_25;
    bool var_26;
    wp::vec_t<3, wp::float32> var_27;
    const wp::int32 var_28 = 0;
    wp::float32 var_29;
    wp::vec_t<3, wp::float32> var_30;
    const wp::int32 var_31 = 1;
    wp::float32 var_32;
    wp::vec_t<3, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    const wp::int32 var_35 = 2;
    wp::float32 var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::float32 var_39;
    const wp::float32 var_40 = 0.0;
    const wp::int32 var_41 = 0;
    const wp::int32 var_42 = 0;
    wp::float32 var_43;
    const wp::int32 var_44 = 1;
    const wp::int32 var_45 = 1;
    wp::float32 var_46;
    const wp::int32 var_47 = 2;
    const wp::int32 var_48 = 2;
    wp::float32 var_49;
    const wp::int32 var_50 = 3;
    wp::float32 var_51;
    wp::float32 var_52;
    bool var_53;
    wp::vec_t<3, wp::float32> var_54;
    const wp::int32 var_55 = 0;
    wp::float32 var_56;
    wp::vec_t<3, wp::float32> var_57;
    const wp::int32 var_58 = 1;
    wp::float32 var_59;
    wp::vec_t<3, wp::float32> var_60;
    wp::vec_t<3, wp::float32> var_61;
    const wp::int32 var_62 = 2;
    wp::float32 var_63;
    wp::vec_t<3, wp::float32> var_64;
    wp::vec_t<3, wp::float32> var_65;
    wp::float32 var_66;
    bool var_67;
    const wp::int32 var_68 = 0;
    wp::float32 var_69;
    const wp::int32 var_70 = 0;
    const wp::float32 var_71 = 0.0;
    const wp::int32 var_72 = 1;
    const wp::int32 var_73 = 1;
    wp::float32 var_74;
    const wp::int32 var_75 = 2;
    const wp::int32 var_76 = 2;
    wp::float32 var_77;
    const wp::int32 var_78 = 3;
    wp::float32 var_79;
    wp::float32 var_80;
    wp::float32 var_81;
    wp::vec_t<3, wp::float32> var_82;
    wp::vec_t<3, wp::float32> var_83;
    wp::float32 var_84;
    bool var_85;
    wp::vec_t<3, wp::float32> var_86;
    const wp::int32 var_87 = 0;
    wp::float32 var_88;
    wp::vec_t<3, wp::float32> var_89;
    const wp::int32 var_90 = 1;
    wp::float32 var_91;
    wp::vec_t<3, wp::float32> var_92;
    wp::vec_t<3, wp::float32> var_93;
    const wp::int32 var_94 = 2;
    wp::float32 var_95;
    wp::vec_t<3, wp::float32> var_96;
    wp::vec_t<3, wp::float32> var_97;
    wp::float32 var_98;
    bool var_99;
    const wp::int32 var_100 = 0;
    wp::float32 var_101;
    const wp::int32 var_102 = 0;
    const wp::int32 var_103 = 1;
    wp::float32 var_104;
    const wp::int32 var_105 = 1;
    const wp::float32 var_106 = 0.0;
    const wp::int32 var_107 = 2;
    const wp::int32 var_108 = 2;
    wp::float32 var_109;
    const wp::int32 var_110 = 3;
    wp::float32 var_111;
    wp::float32 var_112;
    wp::float32 var_113;
    wp::vec_t<3, wp::float32> var_114;
    wp::vec_t<3, wp::float32> var_115;
    wp::float32 var_116;
    bool var_117;
    wp::vec_t<3, wp::float32> var_118;
    const wp::int32 var_119 = 0;
    wp::float32 var_120;
    wp::vec_t<3, wp::float32> var_121;
    const wp::int32 var_122 = 1;
    wp::float32 var_123;
    wp::vec_t<3, wp::float32> var_124;
    wp::vec_t<3, wp::float32> var_125;
    const wp::int32 var_126 = 2;
    wp::float32 var_127;
    wp::vec_t<3, wp::float32> var_128;
    wp::vec_t<3, wp::float32> var_129;
    wp::float32 var_130;
    bool var_131;
    const wp::int32 var_132 = 0;
    wp::float32 var_133;
    const wp::int32 var_134 = 0;
    const wp::int32 var_135 = 1;
    wp::float32 var_136;
    const wp::int32 var_137 = 1;
    const wp::int32 var_138 = 2;
    wp::float32 var_139;
    const wp::int32 var_140 = 2;
    const wp::float32 var_141 = 0.0;
    const wp::int32 var_142 = 3;
    wp::vec_t<3, wp::float32> var_143;
    wp::vec_t<3, wp::float32> var_144;
    wp::float32 var_145;
    //---------
    // forward
    // def _S3D(s1: wp.vec3, s2: wp.vec3, s3: wp.vec3, s4: wp.vec3) -> wp.vec4:               <L 319>
    // C41 = -_det3(s2, s3, s4)                                                               <L 328>
    var_0 = _det3_0(var_s2, var_s3, var_s4);
    var_1 = wp::neg(var_0);
    // C42 = _det3(s1, s3, s4)                                                                <L 329>
    var_2 = _det3_0(var_s1, var_s3, var_s4);
    // C43 = -_det3(s1, s2, s4)                                                               <L 330>
    var_3 = _det3_0(var_s1, var_s2, var_s4);
    var_4 = wp::neg(var_3);
    // C44 = _det3(s1, s2, s3)                                                                <L 331>
    var_5 = _det3_0(var_s1, var_s2, var_s3);
    // m_det = C41 + C42 + C43 + C44                                                          <L 335>
    var_6 = wp::add(var_1, var_2);
    var_7 = wp::add(var_6, var_4);
    var_8 = wp::add(var_7, var_5);
    // comp1 = _same_sign(m_det, C41)                                                         <L 337>
    var_9 = _same_sign_0(var_8, var_1);
    // comp2 = _same_sign(m_det, C42)                                                         <L 338>
    var_10 = _same_sign_0(var_8, var_2);
    // comp3 = _same_sign(m_det, C43)                                                         <L 339>
    var_11 = _same_sign_0(var_8, var_4);
    // comp4 = _same_sign(m_det, C44)                                                         <L 340>
    var_12 = _same_sign_0(var_8, var_5);
    // if comp1 and comp2 and comp3 and comp4:                                                <L 343>
    var_13 = var_9 && var_10 && var_11 && var_12;
    if (var_13) {
        // return wp.vec4(C41 / m_det, C42 / m_det, C43 / m_det, C44 / m_det)                 <L 344>
        var_14 = wp::div(var_1, var_8);
        var_15 = wp::div(var_2, var_8);
        var_16 = wp::div(var_4, var_8);
        var_17 = wp::div(var_5, var_8);
        var_18 = wp::vec_t<4, wp::float32>(var_14, var_15, var_16, var_17);
        return var_18;
    }
    // coordinates = wp.vec4(0.0, 0.0, 0.0, 0.0)                                              <L 347>
    var_23 = wp::vec_t<4, wp::float32>(var_19, var_20, var_21, var_22);
    // dmin = FLOAT_MAX                                                                       <L 348>
    var_25 = wp::copy(var_24);
    // if not comp1:                                                                          <L 350>
    var_26 = wp::unot(var_9);
    if (var_26) {
        // subcoord = _S2D(s2, s3, s4)                                                        <L 351>
        var_27 = _S2D_0(var_s2, var_s3, var_s4);
        // x = subcoord[0] * s2 + subcoord[1] * s3 + subcoord[2] * s4                         <L 352>
        var_29 = wp::extract(var_27, var_28);
        var_30 = wp::mul(var_29, var_s2);
        var_32 = wp::extract(var_27, var_31);
        var_33 = wp::mul(var_32, var_s3);
        var_34 = wp::add(var_30, var_33);
        var_36 = wp::extract(var_27, var_35);
        var_37 = wp::mul(var_36, var_s4);
        var_38 = wp::add(var_34, var_37);
        // d = wp.dot(x, x)                                                                   <L 353>
        var_39 = wp::dot(var_38, var_38);
        // coordinates[0] = 0.0                                                               <L 354>
        wp::assign_inplace(var_23, var_41, var_40);
        // coordinates[1] = subcoord[0]                                                       <L 355>
        var_43 = wp::extract(var_27, var_42);
        wp::assign_inplace(var_23, var_44, var_43);
        // coordinates[2] = subcoord[1]                                                       <L 356>
        var_46 = wp::extract(var_27, var_45);
        wp::assign_inplace(var_23, var_47, var_46);
        // coordinates[3] = subcoord[2]                                                       <L 357>
        var_49 = wp::extract(var_27, var_48);
        wp::assign_inplace(var_23, var_50, var_49);
        // dmin = d                                                                           <L 358>
        var_51 = wp::copy(var_39);
    }
    var_52 = wp::where(var_26, var_51, var_25);
    // if not comp2:                                                                          <L 360>
    var_53 = wp::unot(var_10);
    if (var_53) {
        // subcoord = _S2D(s1, s3, s4)                                                        <L 361>
        var_54 = _S2D_0(var_s1, var_s3, var_s4);
        // x = subcoord[0] * s1 + subcoord[1] * s3 + subcoord[2] * s4                         <L 362>
        var_56 = wp::extract(var_54, var_55);
        var_57 = wp::mul(var_56, var_s1);
        var_59 = wp::extract(var_54, var_58);
        var_60 = wp::mul(var_59, var_s3);
        var_61 = wp::add(var_57, var_60);
        var_63 = wp::extract(var_54, var_62);
        var_64 = wp::mul(var_63, var_s4);
        var_65 = wp::add(var_61, var_64);
        // d = wp.dot(x, x)                                                                   <L 363>
        var_66 = wp::dot(var_65, var_65);
        // if d < dmin:                                                                       <L 364>
        var_67 = (var_66 < var_52);
        if (var_67) {
            // coordinates[0] = subcoord[0]                                                   <L 365>
            var_69 = wp::extract(var_54, var_68);
            wp::assign_inplace(var_23, var_70, var_69);
            // coordinates[1] = 0.0                                                           <L 366>
            wp::assign_inplace(var_23, var_72, var_71);
            // coordinates[2] = subcoord[1]                                                   <L 367>
            var_74 = wp::extract(var_54, var_73);
            wp::assign_inplace(var_23, var_75, var_74);
            // coordinates[3] = subcoord[2]                                                   <L 368>
            var_77 = wp::extract(var_54, var_76);
            wp::assign_inplace(var_23, var_78, var_77);
            // dmin = d                                                                       <L 369>
            var_79 = wp::copy(var_66);
        }
        var_80 = wp::where(var_67, var_79, var_52);
    }
    var_81 = wp::where(var_53, var_80, var_52);
    var_82 = wp::where(var_53, var_54, var_27);
    var_83 = wp::where(var_53, var_65, var_38);
    var_84 = wp::where(var_53, var_66, var_39);
    // if not comp3:                                                                          <L 371>
    var_85 = wp::unot(var_11);
    if (var_85) {
        // subcoord = _S2D(s1, s2, s4)                                                        <L 372>
        var_86 = _S2D_0(var_s1, var_s2, var_s4);
        // x = subcoord[0] * s1 + subcoord[1] * s2 + subcoord[2] * s4                         <L 373>
        var_88 = wp::extract(var_86, var_87);
        var_89 = wp::mul(var_88, var_s1);
        var_91 = wp::extract(var_86, var_90);
        var_92 = wp::mul(var_91, var_s2);
        var_93 = wp::add(var_89, var_92);
        var_95 = wp::extract(var_86, var_94);
        var_96 = wp::mul(var_95, var_s4);
        var_97 = wp::add(var_93, var_96);
        // d = wp.dot(x, x)                                                                   <L 374>
        var_98 = wp::dot(var_97, var_97);
        // if d < dmin:                                                                       <L 375>
        var_99 = (var_98 < var_81);
        if (var_99) {
            // coordinates[0] = subcoord[0]                                                   <L 376>
            var_101 = wp::extract(var_86, var_100);
            wp::assign_inplace(var_23, var_102, var_101);
            // coordinates[1] = subcoord[1]                                                   <L 377>
            var_104 = wp::extract(var_86, var_103);
            wp::assign_inplace(var_23, var_105, var_104);
            // coordinates[2] = 0.0                                                           <L 378>
            wp::assign_inplace(var_23, var_107, var_106);
            // coordinates[3] = subcoord[2]                                                   <L 379>
            var_109 = wp::extract(var_86, var_108);
            wp::assign_inplace(var_23, var_110, var_109);
            // dmin = d                                                                       <L 380>
            var_111 = wp::copy(var_98);
        }
        var_112 = wp::where(var_99, var_111, var_81);
    }
    var_113 = wp::where(var_85, var_112, var_81);
    var_114 = wp::where(var_85, var_86, var_82);
    var_115 = wp::where(var_85, var_97, var_83);
    var_116 = wp::where(var_85, var_98, var_84);
    // if not comp4:                                                                          <L 382>
    var_117 = wp::unot(var_12);
    if (var_117) {
        // subcoord = _S2D(s1, s2, s3)                                                        <L 383>
        var_118 = _S2D_0(var_s1, var_s2, var_s3);
        // x = subcoord[0] * s1 + subcoord[1] * s2 + subcoord[2] * s3                         <L 384>
        var_120 = wp::extract(var_118, var_119);
        var_121 = wp::mul(var_120, var_s1);
        var_123 = wp::extract(var_118, var_122);
        var_124 = wp::mul(var_123, var_s2);
        var_125 = wp::add(var_121, var_124);
        var_127 = wp::extract(var_118, var_126);
        var_128 = wp::mul(var_127, var_s3);
        var_129 = wp::add(var_125, var_128);
        // d = wp.dot(x, x)                                                                   <L 385>
        var_130 = wp::dot(var_129, var_129);
        // if d < dmin:                                                                       <L 386>
        var_131 = (var_130 < var_113);
        if (var_131) {
            // coordinates[0] = subcoord[0]                                                   <L 387>
            var_133 = wp::extract(var_118, var_132);
            wp::assign_inplace(var_23, var_134, var_133);
            // coordinates[1] = subcoord[1]                                                   <L 388>
            var_136 = wp::extract(var_118, var_135);
            wp::assign_inplace(var_23, var_137, var_136);
            // coordinates[2] = subcoord[2]                                                   <L 389>
            var_139 = wp::extract(var_118, var_138);
            wp::assign_inplace(var_23, var_140, var_139);
            // coordinates[3] = 0.0                                                           <L 390>
            wp::assign_inplace(var_23, var_142, var_141);
        }
    }
    var_143 = wp::where(var_117, var_118, var_114);
    var_144 = wp::where(var_117, var_129, var_115);
    var_145 = wp::where(var_117, var_130, var_116);
    // return coordinates                                                                     <L 391>
    return var_23;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:252
static CUDA_CALLABLE wp::vec_t<4, wp::float32> _subdistance_0(
    wp::int32 var_n,
    wp::mat_t<4, 3, wp::float32> var_simplex)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 4;
    bool var_1;
    const wp::int32 var_2 = 0;
    wp::vec_t<3, wp::float32> var_3;
    const wp::int32 var_4 = 1;
    wp::vec_t<3, wp::float32> var_5;
    const wp::int32 var_6 = 2;
    wp::vec_t<3, wp::float32> var_7;
    const wp::int32 var_8 = 3;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<4, wp::float32> var_10;
    const wp::int32 var_11 = 3;
    bool var_12;
    const wp::int32 var_13 = 0;
    wp::vec_t<3, wp::float32> var_14;
    const wp::int32 var_15 = 1;
    wp::vec_t<3, wp::float32> var_16;
    const wp::int32 var_17 = 2;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    const wp::int32 var_20 = 0;
    wp::float32 var_21;
    const wp::int32 var_22 = 1;
    wp::float32 var_23;
    const wp::int32 var_24 = 2;
    wp::float32 var_25;
    const wp::float32 var_26 = 0.0;
    wp::vec_t<4, wp::float32> var_27;
    const wp::int32 var_28 = 2;
    bool var_29;
    const wp::int32 var_30 = 0;
    wp::vec_t<3, wp::float32> var_31;
    const wp::int32 var_32 = 1;
    wp::vec_t<3, wp::float32> var_33;
    wp::vec_t<2, wp::float32> var_34;
    const wp::int32 var_35 = 0;
    wp::float32 var_36;
    const wp::int32 var_37 = 1;
    wp::float32 var_38;
    const wp::float32 var_39 = 0.0;
    const wp::float32 var_40 = 0.0;
    wp::vec_t<4, wp::float32> var_41;
    const wp::float32 var_42 = 1.0;
    const wp::float32 var_43 = 0.0;
    const wp::float32 var_44 = 0.0;
    const wp::float32 var_45 = 0.0;
    wp::vec_t<4, wp::float32> var_46;
    //---------
    // forward
    // def _subdistance(n: int, simplex: mat43) -> wp.vec4:                                   <L 253>
    // if n == 4:                                                                             <L 254>
    var_1 = (var_n == var_0);
    if (var_1) {
        // return _S3D(simplex[0], simplex[1], simplex[2], simplex[3])                        <L 255>
        var_3 = wp::extract(var_simplex, var_2);
        var_5 = wp::extract(var_simplex, var_4);
        var_7 = wp::extract(var_simplex, var_6);
        var_9 = wp::extract(var_simplex, var_8);
        var_10 = _S3D_0(var_3, var_5, var_7, var_9);
        return var_10;
    }
    // if n == 3:                                                                             <L 256>
    var_12 = (var_n == var_11);
    if (var_12) {
        // coordinates3 = _S2D(simplex[0], simplex[1], simplex[2])                            <L 257>
        var_14 = wp::extract(var_simplex, var_13);
        var_16 = wp::extract(var_simplex, var_15);
        var_18 = wp::extract(var_simplex, var_17);
        var_19 = _S2D_0(var_14, var_16, var_18);
        // return wp.vec4(coordinates3[0], coordinates3[1], coordinates3[2], 0.0)             <L 258>
        var_21 = wp::extract(var_19, var_20);
        var_23 = wp::extract(var_19, var_22);
        var_25 = wp::extract(var_19, var_24);
        var_27 = wp::vec_t<4, wp::float32>(var_21, var_23, var_25, var_26);
        return var_27;
    }
    // if n == 2:                                                                             <L 259>
    var_29 = (var_n == var_28);
    if (var_29) {
        // coordinates2 = _S1D(simplex[0], simplex[1])                                        <L 260>
        var_31 = wp::extract(var_simplex, var_30);
        var_33 = wp::extract(var_simplex, var_32);
        var_34 = _S1D_0(var_31, var_33);
        // return wp.vec4(coordinates2[0], coordinates2[1], 0.0, 0.0)                         <L 261>
        var_36 = wp::extract(var_34, var_35);
        var_38 = wp::extract(var_34, var_37);
        var_41 = wp::vec_t<4, wp::float32>(var_36, var_38, var_39, var_40);
        return var_41;
    }
    // return wp.vec4(1.0, 0.0, 0.0, 0.0)                                                     <L 262>
    var_46 = wp::vec_t<4, wp::float32>(var_42, var_43, var_44, var_45);
    return var_46;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:233
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _linear_combine_0(
    wp::int32 var_n,
    wp::vec_t<4, wp::float32> var_coefs,
    wp::mat_t<4, 3, wp::float32> var_mat)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    wp::vec_t<3, wp::float32> var_1;
    const wp::int32 var_2 = 1;
    bool var_3;
    const wp::int32 var_4 = 0;
    wp::float32 var_5;
    const wp::int32 var_6 = 0;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    const wp::int32 var_10 = 2;
    bool var_11;
    const wp::int32 var_12 = 0;
    wp::float32 var_13;
    const wp::int32 var_14 = 0;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32> var_16;
    const wp::int32 var_17 = 1;
    wp::float32 var_18;
    const wp::int32 var_19 = 1;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    const wp::int32 var_24 = 3;
    bool var_25;
    const wp::int32 var_26 = 0;
    wp::float32 var_27;
    const wp::int32 var_28 = 0;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32> var_30;
    const wp::int32 var_31 = 1;
    wp::float32 var_32;
    const wp::int32 var_33 = 1;
    wp::vec_t<3, wp::float32> var_34;
    wp::vec_t<3, wp::float32> var_35;
    wp::vec_t<3, wp::float32> var_36;
    const wp::int32 var_37 = 2;
    wp::float32 var_38;
    const wp::int32 var_39 = 2;
    wp::vec_t<3, wp::float32> var_40;
    wp::vec_t<3, wp::float32> var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::vec_t<3, wp::float32> var_43;
    const wp::int32 var_44 = 0;
    wp::float32 var_45;
    const wp::int32 var_46 = 0;
    wp::vec_t<3, wp::float32> var_47;
    wp::vec_t<3, wp::float32> var_48;
    const wp::int32 var_49 = 1;
    wp::float32 var_50;
    const wp::int32 var_51 = 1;
    wp::vec_t<3, wp::float32> var_52;
    wp::vec_t<3, wp::float32> var_53;
    wp::vec_t<3, wp::float32> var_54;
    const wp::int32 var_55 = 2;
    wp::float32 var_56;
    const wp::int32 var_57 = 2;
    wp::vec_t<3, wp::float32> var_58;
    wp::vec_t<3, wp::float32> var_59;
    wp::vec_t<3, wp::float32> var_60;
    const wp::int32 var_61 = 3;
    wp::float32 var_62;
    const wp::int32 var_63 = 3;
    wp::vec_t<3, wp::float32> var_64;
    wp::vec_t<3, wp::float32> var_65;
    wp::vec_t<3, wp::float32> var_66;
    wp::vec_t<3, wp::float32> var_67;
    wp::vec_t<3, wp::float32> var_68;
    wp::vec_t<3, wp::float32> var_69;
    //---------
    // forward
    // def _linear_combine(n: int, coefs: wp.vec4, mat: mat43) -> wp.vec3:                    <L 234>
    // v = wp.vec3(0.0)                                                                       <L 235>
    var_1 = wp::vec_t<3, wp::float32>(var_0);
    // if n == 1:                                                                             <L 236>
    var_3 = (var_n == var_2);
    if (var_3) {
        // v = coefs[0] * mat[0]                                                              <L 237>
        var_5 = wp::extract(var_coefs, var_4);
        var_7 = wp::extract(var_mat, var_6);
        var_8 = wp::mul(var_5, var_7);
    }
    var_9 = wp::where(var_3, var_8, var_1);
    if (!var_3) {
        // elif n == 2:                                                                       <L 238>
        var_11 = (var_n == var_10);
        if (var_11) {
            // v = coefs[0] * mat[0] + coefs[1] * mat[1]                                      <L 239>
            var_13 = wp::extract(var_coefs, var_12);
            var_15 = wp::extract(var_mat, var_14);
            var_16 = wp::mul(var_13, var_15);
            var_18 = wp::extract(var_coefs, var_17);
            var_20 = wp::extract(var_mat, var_19);
            var_21 = wp::mul(var_18, var_20);
            var_22 = wp::add(var_16, var_21);
        }
        var_23 = wp::where(var_11, var_22, var_9);
        if (!var_11) {
            // elif n == 3:                                                                   <L 240>
            var_25 = (var_n == var_24);
            if (var_25) {
                // v = coefs[0] * mat[0] + coefs[1] * mat[1] + coefs[2] * mat[2]              <L 241>
                var_27 = wp::extract(var_coefs, var_26);
                var_29 = wp::extract(var_mat, var_28);
                var_30 = wp::mul(var_27, var_29);
                var_32 = wp::extract(var_coefs, var_31);
                var_34 = wp::extract(var_mat, var_33);
                var_35 = wp::mul(var_32, var_34);
                var_36 = wp::add(var_30, var_35);
                var_38 = wp::extract(var_coefs, var_37);
                var_40 = wp::extract(var_mat, var_39);
                var_41 = wp::mul(var_38, var_40);
                var_42 = wp::add(var_36, var_41);
            }
            var_43 = wp::where(var_25, var_42, var_23);
            if (!var_25) {
                // v = coefs[0] * mat[0] + coefs[1] * mat[1] + coefs[2] * mat[2] + coefs[3] * mat[3]       <L 243>
                var_45 = wp::extract(var_coefs, var_44);
                var_47 = wp::extract(var_mat, var_46);
                var_48 = wp::mul(var_45, var_47);
                var_50 = wp::extract(var_coefs, var_49);
                var_52 = wp::extract(var_mat, var_51);
                var_53 = wp::mul(var_50, var_52);
                var_54 = wp::add(var_48, var_53);
                var_56 = wp::extract(var_coefs, var_55);
                var_58 = wp::extract(var_mat, var_57);
                var_59 = wp::mul(var_56, var_58);
                var_60 = wp::add(var_54, var_59);
                var_62 = wp::extract(var_coefs, var_61);
                var_64 = wp::extract(var_mat, var_63);
                var_65 = wp::mul(var_62, var_64);
                var_66 = wp::add(var_60, var_65);
            }
            var_67 = wp::where(var_25, var_43, var_66);
        }
        var_68 = wp::where(var_11, var_23, var_67);
    }
    var_69 = wp::where(var_3, var_9, var_68);
    // return v                                                                               <L 244>
    return var_69;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:247
static CUDA_CALLABLE bool _almost_equal_0(
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    const wp::float32 var_6 = 1e-15;
    bool var_7;
    const wp::int32 var_8 = 1;
    wp::float32 var_9;
    const wp::int32 var_10 = 1;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    bool var_14;
    const wp::int32 var_15 = 2;
    wp::float32 var_16;
    const wp::int32 var_17 = 2;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    bool var_21;
    bool var_22;
    //---------
    // forward
    // def _almost_equal(v1: wp.vec3, v2: wp.vec3) -> bool:                                   <L 248>
    // return wp.abs(v1[0] - v2[0]) < MINVAL and wp.abs(v1[1] - v2[1]) < MINVAL and wp.abs(v1[2] - v2[2]) < MINVAL       <L 249>
    var_1 = wp::extract(var_v1, var_0);
    var_3 = wp::extract(var_v2, var_2);
    var_4 = wp::sub(var_1, var_3);
    var_5 = wp::abs(var_4);
    var_7 = (var_5 < var_6);
    var_9 = wp::extract(var_v1, var_8);
    var_11 = wp::extract(var_v2, var_10);
    var_12 = wp::sub(var_9, var_11);
    var_13 = wp::abs(var_12);
    var_14 = (var_13 < var_6);
    var_16 = wp::extract(var_v1, var_15);
    var_18 = wp::extract(var_v2, var_17);
    var_19 = wp::sub(var_16, var_18);
    var_20 = wp::abs(var_19);
    var_21 = (var_20 < var_6);
    var_22 = var_7 && var_14 && var_21;
    return var_22;
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:562
static CUDA_CALLABLE GJKResult_0220ee01 gjk_0(
    wp::float32 var_tolerance,
    wp::int32 var_gjk_iterations,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::vec_t<3, wp::float32> var_x1_0,
    wp::vec_t<3, wp::float32> var_x2_0,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2,
    wp::float32 var_cutoff,
    bool var_is_discrete)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::mat_t<4, 3, wp::float32> var_1;
    wp::mat_t<4, 3, wp::float32> var_2;
    wp::mat_t<4, 3, wp::float32> var_3;
    wp::vec_t<4, wp::int32> var_4;
    wp::vec_t<4, wp::int32> var_5;
    const wp::int32 var_6 = 0;
    wp::int32 var_7;
    wp::vec_t<4, wp::float32> var_8;
    wp::float32 var_9;
    const wp::float32 var_10 = 0.0;
    const wp::float32 var_11 = 0.5;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::vec_t<3, wp::float32> var_14;
    const wp::float32 var_15 = 1e+30;
    wp::float32 var_16;
    wp::range_t var_17;
    wp::int32 var_18;
    wp::float32 var_19;
    bool var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    bool var_23;
    bool var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    SupportPoint_e82efc60 var_29;
    wp::vec_t<3, wp::float32>* var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::int32* var_32;
    wp::int32* var_33;
    wp::int32 var_34;
    wp::int32* var_35;
    wp::int32 var_36;
    SupportPoint_e82efc60 var_37;
    wp::vec_t<3, wp::float32>* var_38;
    wp::vec_t<3, wp::float32> var_39;
    wp::int32* var_40;
    wp::int32* var_41;
    wp::int32 var_42;
    wp::int32* var_43;
    wp::int32 var_44;
    wp::vec_t<3, wp::float32> var_45;
    wp::vec_t<3, wp::float32> var_46;
    wp::vec_t<3, wp::float32> var_47;
    const wp::float32 var_48 = 0.0;
    bool var_49;
    wp::vec_t<3, wp::float32> var_50;
    wp::float32 var_51;
    const wp::float32 var_52 = 0.0;
    bool var_53;
    GJKResult_0220ee01 var_54;
    const wp::int32 var_55 = 0;
    wp::int32* var_56;
    wp::float32* var_57;
    bool var_58;
    wp::vec_t<3, wp::float32> var_59;
    wp::float32 var_60;
    wp::vec_t<3, wp::float32> var_61;
    wp::float32 var_62;
    const wp::float32 var_63 = 0.0;
    bool var_64;
    wp::float32 var_65;
    wp::float32 var_66;
    bool var_67;
    bool var_68;
    GJKResult_0220ee01 var_69;
    const wp::int32 var_70 = 0;
    wp::int32* var_71;
    wp::float32* var_72;
    GJKResult_0220ee01 var_73;
    GJKResult_0220ee01 var_74;
    GJKResult_0220ee01 var_75;
    wp::vec_t<3, wp::float32> var_76;
    wp::vec_t<3, wp::float32> var_77;
    wp::float32 var_78;
    bool var_79;
    wp::float32 var_80;
    const wp::int32 var_81 = 1;
    wp::int32 var_82;
    wp::vec_t<4, wp::float32> var_83;
    const wp::int32 var_84 = 0;
    wp::int32 var_85;
    const wp::int32 var_86 = 4;
    wp::range_t var_87;
    wp::int32 var_88;
    wp::float32 var_89;
    const wp::float32 var_90 = 0.0;
    bool var_91;
    wp::vec_t<3, wp::float32> var_92;
    wp::vec_t<3, wp::float32> var_93;
    wp::vec_t<3, wp::float32> var_94;
    wp::int32 var_95;
    wp::int32 var_96;
    wp::float32 var_97;
    const wp::int32 var_98 = 1;
    wp::int32 var_99;
    wp::int32 var_100;
    const wp::int32 var_101 = 1;
    bool var_102;
    wp::int32 var_103;
    wp::vec_t<4, wp::float32> var_104;
    wp::float32 var_105;
    wp::vec_t<3, wp::float32> var_106;
    bool var_107;
    wp::int32 var_108;
    wp::vec_t<4, wp::float32> var_109;
    wp::float32 var_110;
    wp::vec_t<3, wp::float32> var_111;
    const wp::int32 var_112 = 4;
    bool var_113;
    wp::int32 var_114;
    wp::vec_t<4, wp::float32> var_115;
    wp::vec_t<3, wp::float32> var_116;
    wp::float32 var_117;
    GJKResult_0220ee01 var_118;
    const wp::int32 var_119 = 0;
    bool var_120;
    wp::vec_t<3, wp::float32> var_121;
    wp::vec_t<3, wp::float32> var_122;
    wp::vec_t<3, wp::float32>* var_123;
    const wp::int32 var_124 = 0;
    bool var_125;
    wp::vec_t<3, wp::float32> var_126;
    wp::vec_t<3, wp::float32> var_127;
    wp::vec_t<3, wp::float32>* var_128;
    wp::float32 var_129;
    wp::float32* var_130;
    wp::int32* var_131;
    wp::mat_t<4, 3, wp::float32>* var_132;
    wp::mat_t<4, 3, wp::float32>* var_133;
    wp::vec_t<4, wp::int32>* var_134;
    wp::vec_t<4, wp::int32>* var_135;
    wp::mat_t<4, 3, wp::float32>* var_136;
    //---------
    // forward
    // def gjk(                                                                               <L 563>
    // cutoff2 = cutoff * cutoff                                                              <L 577>
    var_0 = wp::mul(var_cutoff, var_cutoff);
    // simplex = mat43()                                                                      <L 578>
    var_1 = wp::mat_t<4, 3, wp::float32>();
    // simplex1 = mat43()                                                                     <L 579>
    var_2 = wp::mat_t<4, 3, wp::float32>();
    // simplex2 = mat43()                                                                     <L 580>
    var_3 = wp::mat_t<4, 3, wp::float32>();
    // simplex_index1 = wp.vec4i()                                                            <L 581>
    var_4 = wp::vec_t<4, wp::int32>();
    // simplex_index2 = wp.vec4i()                                                            <L 582>
    var_5 = wp::vec_t<4, wp::int32>();
    // n = int(0)                                                                             <L 583>
    var_7 = wp::int(var_6);
    // coordinates = wp.vec4()  # barycentric coordinates                                     <L 584>
    var_8 = wp::vec_t<4, wp::float32>();
    // tol2 = tolerance * tolerance                                                           <L 585>
    var_9 = wp::mul(var_tolerance, var_tolerance);
    // epsilon = wp.where(is_discrete, 0.0, 0.5 * tol2)                                       <L 586>
    var_12 = wp::mul(var_11, var_9);
    var_13 = wp::where(var_is_discrete, var_10, var_12);
    // x_k = x1_0 - x2_0                                                                      <L 589>
    var_14 = wp::sub(var_x1_0, var_x2_0);
    // xnorm_old = FLOAT_MAX                                                                  <L 590>
    var_16 = wp::copy(var_15);
    // for _ in range(gjk_iterations):                                                        <L 592>
    var_17 = wp::range(var_gjk_iterations);
    start_for_0:;
        if (iter_cmp(var_17) == 0) goto end_for_0;
        var_18 = wp::iter_next(var_17);
        // xnorm = wp.dot(x_k, x_k)                                                           <L 593>
        var_19 = wp::dot(var_14, var_14);
        // if xnorm < tol2 or wp.abs(xnorm_old - xnorm) < tol2:                               <L 595>
        var_20 = (var_19 < var_9);
        var_21 = wp::sub(var_16, var_19);
        var_22 = wp::abs(var_21);
        var_23 = (var_22 < var_9);
        var_24 = var_20 || var_23;
        if (var_24) {
            // break                                                                          <L 596>
            goto end_for_0;
        }
        // xnorm_old = xnorm                                                                  <L 597>
        var_25 = wp::copy(var_19);
        // dir_neg = x_k / wp.sqrt(xnorm)                                                     <L 598>
        var_26 = wp::sqrt(var_19);
        var_27 = wp::div(var_14, var_26);
        // sp = support(geom1, geomtype1, -dir_neg)                                           <L 601>
        var_28 = wp::neg(var_27);
        var_29 = support_0(var_geom1, var_geomtype1, var_28);
        // simplex1[n] = sp.point                                                             <L 602>
        var_30 = &(var_29.point);
        var_31 = wp::load(var_30);
        wp::assign_inplace(var_2, var_7, var_31);
        // geom1.index = sp.cached_index                                                      <L 603>
        var_32 = &(var_29.cached_index);
        var_33 = &(var_geom1.index);
        var_34 = wp::load(var_32);
        wp::store(var_33, var_34);
        // simplex_index1[n] = sp.vertex_index                                                <L 604>
        var_35 = &(var_29.vertex_index);
        var_36 = wp::load(var_35);
        wp::assign_inplace(var_4, var_7, var_36);
        // sp = support(geom2, geomtype2, dir_neg)                                            <L 607>
        var_37 = support_0(var_geom2, var_geomtype2, var_27);
        // simplex2[n] = sp.point                                                             <L 608>
        var_38 = &(var_37.point);
        var_39 = wp::load(var_38);
        wp::assign_inplace(var_3, var_7, var_39);
        // geom2.index = sp.cached_index                                                      <L 609>
        var_40 = &(var_37.cached_index);
        var_41 = &(var_geom2.index);
        var_42 = wp::load(var_40);
        wp::store(var_41, var_42);
        // simplex_index2[n] = sp.vertex_index                                                <L 610>
        var_43 = &(var_37.vertex_index);
        var_44 = wp::load(var_43);
        wp::assign_inplace(var_5, var_7, var_44);
        // simplex[n] = simplex1[n] - simplex2[n]                                             <L 613>
        var_45 = wp::extract(var_2, var_7);
        var_46 = wp::extract(var_3, var_7);
        var_47 = wp::sub(var_45, var_46);
        wp::assign_inplace(var_1, var_7, var_47);
        // if cutoff == 0.0:                                                                  <L 615>
        var_49 = (var_cutoff == var_48);
        if (var_49) {
            // if wp.dot(x_k, simplex[n]) > 0.0:                                              <L 616>
            var_50 = wp::extract(var_1, var_7);
            var_51 = wp::dot(var_14, var_50);
            var_53 = (var_51 > var_52);
            if (var_53) {
                // result = GJKResult()                                                       <L 617>
                var_54 = GJKResult_0220ee01();
                // result.dim = 0                                                             <L 618>
                var_56 = &(var_54.dim);
                wp::store(var_56, var_55);
                // result.dist = FLOAT_MAX                                                    <L 619>
                var_57 = &(var_54.dist);
                wp::store(var_57, var_15);
                // return result                                                              <L 620>
                return var_54;
            }
        }
        if (!var_49) {
            // elif cutoff < FLOAT_MAX:                                                       <L 621>
            var_58 = (var_cutoff < var_15);
            if (var_58) {
                // vs = wp.dot(x_k, simplex[n])                                               <L 622>
                var_59 = wp::extract(var_1, var_7);
                var_60 = wp::dot(var_14, var_59);
                // if wp.dot(x_k, simplex[n]) > 0.0 and (vs * vs / xnorm) >= cutoff2:         <L 623>
                var_61 = wp::extract(var_1, var_7);
                var_62 = wp::dot(var_14, var_61);
                var_64 = (var_62 > var_63);
                var_65 = wp::mul(var_60, var_60);
                var_66 = wp::div(var_65, var_19);
                var_67 = (var_66 >= var_0);
                var_68 = var_64 && var_67;
                if (var_68) {
                    // result = GJKResult()                                                   <L 624>
                    var_69 = GJKResult_0220ee01();
                    // result.dim = 0                                                         <L 625>
                    var_71 = &(var_69.dim);
                    wp::store(var_71, var_70);
                    // result.dist = FLOAT_MAX                                                <L 626>
                    var_72 = &(var_69.dist);
                    wp::store(var_72, var_15);
                    // return result                                                          <L 627>
                    return var_69;
                }
                var_73 = wp::where(var_68, var_69, var_54);
            }
            var_74 = wp::where(var_58, var_73, var_54);
        }
        var_75 = wp::where(var_49, var_54, var_74);
        // if wp.dot(x_k, x_k - simplex[n]) < epsilon:                                        <L 631>
        var_76 = wp::extract(var_1, var_7);
        var_77 = wp::sub(var_14, var_76);
        var_78 = wp::dot(var_14, var_77);
        var_79 = (var_78 < var_13);
        if (var_79) {
            // break                                                                          <L 632>
            wp::assign(var_16, var_25);
            goto end_for_0;
        }
        var_80 = wp::where(var_79, var_16, var_25);
        // coordinates = _subdistance(n + 1, simplex)                                         <L 636>
        var_82 = wp::add(var_7, var_81);
        var_83 = _subdistance_0(var_82, var_1);
        // n = int(0)                                                                         <L 639>
        var_85 = wp::int(var_84);
        // for i in range(4):                                                                 <L 640>
        var_87 = wp::range(var_86);
        start_for_4:;
            if (iter_cmp(var_87) == 0) goto end_for_4;
            var_88 = wp::iter_next(var_87);
            // if coordinates[i] == 0.0:                                                      <L 641>
            var_89 = wp::extract(var_83, var_88);
            var_91 = (var_89 == var_90);
            if (var_91) {
                // continue                                                                   <L 642>
                goto start_for_4;
            }
            // simplex[n] = simplex[i]                                                        <L 644>
            var_92 = wp::extract(var_1, var_88);
            wp::assign_inplace(var_1, var_85, var_92);
            // simplex1[n] = simplex1[i]                                                      <L 645>
            var_93 = wp::extract(var_2, var_88);
            wp::assign_inplace(var_2, var_85, var_93);
            // simplex2[n] = simplex2[i]                                                      <L 646>
            var_94 = wp::extract(var_3, var_88);
            wp::assign_inplace(var_3, var_85, var_94);
            // simplex_index1[n] = simplex_index1[i]                                          <L 647>
            var_95 = wp::extract(var_4, var_88);
            wp::assign_inplace(var_4, var_85, var_95);
            // simplex_index2[n] = simplex_index2[i]                                          <L 648>
            var_96 = wp::extract(var_5, var_88);
            wp::assign_inplace(var_5, var_85, var_96);
            // coordinates[n] = coordinates[i]                                                <L 649>
            var_97 = wp::extract(var_83, var_88);
            wp::assign_inplace(var_83, var_85, var_97);
            // n += int(1)                                                                    <L 650>
            var_99 = wp::int(var_98);
            var_100 = wp::add(var_85, var_99);
            wp::assign(var_85, var_100);
            goto start_for_4;
        end_for_4:;
        // if n < 1:                                                                          <L 653>
        var_102 = (var_85 < var_101);
        if (var_102) {
            // break                                                                          <L 654>
            wp::assign(var_7, var_85);
            wp::assign(var_8, var_83);
            wp::assign(var_16, var_80);
            goto end_for_0;
        }
        var_103 = wp::where(var_102, var_7, var_85);
        var_104 = wp::where(var_102, var_8, var_83);
        var_105 = wp::where(var_102, var_16, var_80);
        // x_next = _linear_combine(n, coordinates, simplex)                                  <L 657>
        var_106 = _linear_combine_0(var_103, var_104, var_1);
        // if _almost_equal(x_next, x_k):                                                     <L 660>
        var_107 = _almost_equal_0(var_106, var_14);
        if (var_107) {
            // break                                                                          <L 661>
            wp::assign(var_7, var_103);
            wp::assign(var_8, var_104);
            wp::assign(var_16, var_105);
            goto end_for_0;
        }
        var_108 = wp::where(var_107, var_7, var_103);
        var_109 = wp::where(var_107, var_8, var_104);
        var_110 = wp::where(var_107, var_16, var_105);
        // x_k = x_next                                                                       <L 664>
        var_111 = wp::copy(var_106);
        // if n == 4:                                                                         <L 667>
        var_113 = (var_108 == var_112);
        if (var_113) {
            // break                                                                          <L 668>
            wp::assign(var_7, var_108);
            wp::assign(var_8, var_109);
            wp::assign(var_14, var_111);
            wp::assign(var_16, var_110);
            goto end_for_0;
        }
        var_114 = wp::where(var_113, var_7, var_108);
        var_115 = wp::where(var_113, var_8, var_109);
        var_116 = wp::where(var_113, var_14, var_111);
        var_117 = wp::where(var_113, var_16, var_110);
        wp::assign(var_7, var_114);
        wp::assign(var_8, var_115);
        wp::assign(var_14, var_116);
        wp::assign(var_16, var_117);
        goto start_for_0;
    end_for_0:;
    // result = GJKResult()                                                                   <L 670>
    var_118 = GJKResult_0220ee01();
    // result.x1 = wp.where(n == 0, x1_0, _linear_combine(n, coordinates, simplex1))          <L 675>
    var_120 = (var_7 == var_119);
    var_121 = _linear_combine_0(var_7, var_8, var_2);
    var_122 = wp::where(var_120, var_x1_0, var_121);
    var_123 = &(var_118.x1);
    wp::store(var_123, var_122);
    // result.x2 = wp.where(n == 0, x2_0, _linear_combine(n, coordinates, simplex2))          <L 676>
    var_125 = (var_7 == var_124);
    var_126 = _linear_combine_0(var_7, var_8, var_3);
    var_127 = wp::where(var_125, var_x2_0, var_126);
    var_128 = &(var_118.x2);
    wp::store(var_128, var_127);
    // result.dist = wp.norm_l2(x_k)                                                          <L 677>
    var_129 = norm_l2_0(var_14);
    var_130 = &(var_118.dist);
    wp::store(var_130, var_129);
    // result.dim = n                                                                         <L 679>
    var_131 = &(var_118.dim);
    wp::store(var_131, var_7);
    // result.simplex1 = simplex1                                                             <L 680>
    var_132 = &(var_118.simplex1);
    wp::store(var_132, var_2);
    // result.simplex2 = simplex2                                                             <L 681>
    var_133 = &(var_118.simplex2);
    wp::store(var_133, var_3);
    // result.simplex_index1 = simplex_index1                                                 <L 682>
    var_134 = &(var_118.simplex_index1);
    wp::store(var_134, var_4);
    // result.simplex_index2 = simplex_index2                                                 <L 683>
    var_135 = &(var_118.simplex_index2);
    wp::store(var_135, var_5);
    // result.simplex = simplex                                                               <L 684>
    var_136 = &(var_118.simplex);
    wp::store(var_136, var_1);
    // return result                                                                          <L 685>
    return var_118;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:701
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _tri_affine_coord_0(
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> var_v3,
    wp::vec_t<3, wp::float32> var_p)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::float32 var_1;
    const wp::int32 var_2 = 2;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::int32 var_5 = 2;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::int32 var_11 = 1;
    wp::float32 var_12;
    const wp::int32 var_13 = 2;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::int32 var_17 = 2;
    wp::float32 var_18;
    const wp::int32 var_19 = 1;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::int32 var_23 = 1;
    wp::float32 var_24;
    const wp::int32 var_25 = 2;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    const wp::int32 var_29 = 2;
    wp::float32 var_30;
    const wp::int32 var_31 = 1;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    const wp::int32 var_35 = 0;
    wp::float32 var_36;
    const wp::int32 var_37 = 2;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::int32 var_40 = 2;
    wp::float32 var_41;
    const wp::int32 var_42 = 0;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    const wp::int32 var_46 = 0;
    wp::float32 var_47;
    const wp::int32 var_48 = 2;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    const wp::int32 var_52 = 2;
    wp::float32 var_53;
    const wp::int32 var_54 = 0;
    wp::float32 var_55;
    wp::float32 var_56;
    wp::float32 var_57;
    const wp::int32 var_58 = 0;
    wp::float32 var_59;
    const wp::int32 var_60 = 2;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    const wp::int32 var_64 = 2;
    wp::float32 var_65;
    const wp::int32 var_66 = 0;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    const wp::int32 var_70 = 0;
    wp::float32 var_71;
    const wp::int32 var_72 = 1;
    wp::float32 var_73;
    wp::float32 var_74;
    const wp::int32 var_75 = 1;
    wp::float32 var_76;
    const wp::int32 var_77 = 0;
    wp::float32 var_78;
    wp::float32 var_79;
    wp::float32 var_80;
    const wp::int32 var_81 = 0;
    wp::float32 var_82;
    const wp::int32 var_83 = 1;
    wp::float32 var_84;
    wp::float32 var_85;
    wp::float32 var_86;
    const wp::int32 var_87 = 1;
    wp::float32 var_88;
    const wp::int32 var_89 = 0;
    wp::float32 var_90;
    wp::float32 var_91;
    wp::float32 var_92;
    const wp::int32 var_93 = 0;
    wp::float32 var_94;
    const wp::int32 var_95 = 1;
    wp::float32 var_96;
    wp::float32 var_97;
    wp::float32 var_98;
    const wp::int32 var_99 = 1;
    wp::float32 var_100;
    const wp::int32 var_101 = 0;
    wp::float32 var_102;
    wp::float32 var_103;
    wp::float32 var_104;
    const wp::float32 var_105 = 0.0;
    const wp::int32 var_106 = 0;
    const wp::int32 var_107 = 0;
    wp::float32 var_108;
    wp::float32 var_109;
    wp::float32 var_110;
    bool var_111;
    bool var_112;
    bool var_113;
    wp::float32 var_114;
    const wp::int32 var_115 = 1;
    const wp::int32 var_116 = 2;
    wp::float32 var_117;
    wp::int32 var_118;
    wp::int32 var_119;
    bool var_120;
    wp::float32 var_121;
    const wp::int32 var_122 = 0;
    const wp::int32 var_123 = 2;
    wp::float32 var_124;
    wp::int32 var_125;
    wp::int32 var_126;
    wp::float32 var_127;
    const wp::int32 var_128 = 0;
    const wp::int32 var_129 = 1;
    wp::float32 var_130;
    wp::int32 var_131;
    wp::int32 var_132;
    wp::float32 var_133;
    wp::int32 var_134;
    wp::int32 var_135;
    wp::float32 var_136;
    wp::float32 var_137;
    wp::float32 var_138;
    wp::float32 var_139;
    wp::float32 var_140;
    wp::float32 var_141;
    wp::float32 var_142;
    wp::float32 var_143;
    wp::float32 var_144;
    wp::float32 var_145;
    wp::float32 var_146;
    wp::float32 var_147;
    wp::float32 var_148;
    wp::float32 var_149;
    wp::float32 var_150;
    wp::float32 var_151;
    wp::float32 var_152;
    wp::float32 var_153;
    wp::float32 var_154;
    wp::float32 var_155;
    wp::float32 var_156;
    wp::float32 var_157;
    wp::float32 var_158;
    wp::float32 var_159;
    wp::float32 var_160;
    wp::float32 var_161;
    wp::float32 var_162;
    wp::float32 var_163;
    wp::float32 var_164;
    wp::float32 var_165;
    wp::float32 var_166;
    wp::float32 var_167;
    wp::float32 var_168;
    wp::float32 var_169;
    wp::float32 var_170;
    wp::float32 var_171;
    wp::float32 var_172;
    wp::float32 var_173;
    wp::float32 var_174;
    wp::float32 var_175;
    wp::float32 var_176;
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
    wp::float32 var_194;
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
    wp::float32 var_207;
    wp::vec_t<3, wp::float32> var_208;
    //---------
    // forward
    // def _tri_affine_coord(v1: wp.vec3, v2: wp.vec3, v3: wp.vec3, p: wp.vec3) -> wp.vec3:       <L 702>
    // M_14 = v2[1] * v3[2] - v2[2] * v3[1] - v1[1] * v3[2] + v1[2] * v3[1] + v1[1] * v2[2] - v1[2] * v2[1]       <L 704>
    var_1 = wp::extract(var_v2, var_0);
    var_3 = wp::extract(var_v3, var_2);
    var_4 = wp::mul(var_1, var_3);
    var_6 = wp::extract(var_v2, var_5);
    var_8 = wp::extract(var_v3, var_7);
    var_9 = wp::mul(var_6, var_8);
    var_10 = wp::sub(var_4, var_9);
    var_12 = wp::extract(var_v1, var_11);
    var_14 = wp::extract(var_v3, var_13);
    var_15 = wp::mul(var_12, var_14);
    var_16 = wp::sub(var_10, var_15);
    var_18 = wp::extract(var_v1, var_17);
    var_20 = wp::extract(var_v3, var_19);
    var_21 = wp::mul(var_18, var_20);
    var_22 = wp::add(var_16, var_21);
    var_24 = wp::extract(var_v1, var_23);
    var_26 = wp::extract(var_v2, var_25);
    var_27 = wp::mul(var_24, var_26);
    var_28 = wp::add(var_22, var_27);
    var_30 = wp::extract(var_v1, var_29);
    var_32 = wp::extract(var_v2, var_31);
    var_33 = wp::mul(var_30, var_32);
    var_34 = wp::sub(var_28, var_33);
    // M_24 = v2[0] * v3[2] - v2[2] * v3[0] - v1[0] * v3[2] + v1[2] * v3[0] + v1[0] * v2[2] - v1[2] * v2[0]       <L 705>
    var_36 = wp::extract(var_v2, var_35);
    var_38 = wp::extract(var_v3, var_37);
    var_39 = wp::mul(var_36, var_38);
    var_41 = wp::extract(var_v2, var_40);
    var_43 = wp::extract(var_v3, var_42);
    var_44 = wp::mul(var_41, var_43);
    var_45 = wp::sub(var_39, var_44);
    var_47 = wp::extract(var_v1, var_46);
    var_49 = wp::extract(var_v3, var_48);
    var_50 = wp::mul(var_47, var_49);
    var_51 = wp::sub(var_45, var_50);
    var_53 = wp::extract(var_v1, var_52);
    var_55 = wp::extract(var_v3, var_54);
    var_56 = wp::mul(var_53, var_55);
    var_57 = wp::add(var_51, var_56);
    var_59 = wp::extract(var_v1, var_58);
    var_61 = wp::extract(var_v2, var_60);
    var_62 = wp::mul(var_59, var_61);
    var_63 = wp::add(var_57, var_62);
    var_65 = wp::extract(var_v1, var_64);
    var_67 = wp::extract(var_v2, var_66);
    var_68 = wp::mul(var_65, var_67);
    var_69 = wp::sub(var_63, var_68);
    // M_34 = v2[0] * v3[1] - v2[1] * v3[0] - v1[0] * v3[1] + v1[1] * v3[0] + v1[0] * v2[1] - v1[1] * v2[0]       <L 706>
    var_71 = wp::extract(var_v2, var_70);
    var_73 = wp::extract(var_v3, var_72);
    var_74 = wp::mul(var_71, var_73);
    var_76 = wp::extract(var_v2, var_75);
    var_78 = wp::extract(var_v3, var_77);
    var_79 = wp::mul(var_76, var_78);
    var_80 = wp::sub(var_74, var_79);
    var_82 = wp::extract(var_v1, var_81);
    var_84 = wp::extract(var_v3, var_83);
    var_85 = wp::mul(var_82, var_84);
    var_86 = wp::sub(var_80, var_85);
    var_88 = wp::extract(var_v1, var_87);
    var_90 = wp::extract(var_v3, var_89);
    var_91 = wp::mul(var_88, var_90);
    var_92 = wp::add(var_86, var_91);
    var_94 = wp::extract(var_v1, var_93);
    var_96 = wp::extract(var_v2, var_95);
    var_97 = wp::mul(var_94, var_96);
    var_98 = wp::add(var_92, var_97);
    var_100 = wp::extract(var_v1, var_99);
    var_102 = wp::extract(var_v2, var_101);
    var_103 = wp::mul(var_100, var_102);
    var_104 = wp::sub(var_98, var_103);
    // M_max = 0.0                                                                            <L 710>
    // x = 0                                                                                  <L 711>
    // y = 0                                                                                  <L 712>
    // mu1 = wp.abs(M_14)                                                                     <L 714>
    var_108 = wp::abs(var_34);
    // mu2 = wp.abs(M_24)                                                                     <L 715>
    var_109 = wp::abs(var_69);
    // mu3 = wp.abs(M_34)                                                                     <L 716>
    var_110 = wp::abs(var_104);
    // if mu1 >= mu2 and mu1 >= mu3:                                                          <L 718>
    var_111 = (var_108 >= var_109);
    var_112 = (var_108 >= var_110);
    var_113 = var_111 && var_112;
    if (var_113) {
        // M_max = M_14                                                                       <L 719>
        var_114 = wp::copy(var_34);
        // x = 1                                                                              <L 720>
        // y = 2                                                                              <L 721>
    }
    var_117 = wp::where(var_113, var_114, var_105);
    var_118 = wp::where(var_113, var_115, var_106);
    var_119 = wp::where(var_113, var_116, var_107);
    if (!var_113) {
        // elif mu2 >= mu3:                                                                   <L 722>
        var_120 = (var_109 >= var_110);
        if (var_120) {
            // M_max = M_24                                                                   <L 723>
            var_121 = wp::copy(var_69);
            // x = 0                                                                          <L 724>
            // y = 2                                                                          <L 725>
        }
        var_124 = wp::where(var_120, var_121, var_117);
        var_125 = wp::where(var_120, var_122, var_118);
        var_126 = wp::where(var_120, var_123, var_119);
        if (!var_120) {
            // M_max = M_34                                                                   <L 727>
            var_127 = wp::copy(var_104);
            // x = 0                                                                          <L 728>
            // y = 1                                                                          <L 729>
        }
        var_130 = wp::where(var_120, var_124, var_127);
        var_131 = wp::where(var_120, var_125, var_128);
        var_132 = wp::where(var_120, var_126, var_129);
    }
    var_133 = wp::where(var_113, var_117, var_130);
    var_134 = wp::where(var_113, var_118, var_131);
    var_135 = wp::where(var_113, var_119, var_132);
    // C31 = p[x] * v2[y] + p[y] * v3[x] + v2[x] * v3[y] - p[x] * v3[y] - p[y] * v2[x] - v3[x] * v2[y]       <L 732>
    var_136 = wp::extract(var_p, var_134);
    var_137 = wp::extract(var_v2, var_135);
    var_138 = wp::mul(var_136, var_137);
    var_139 = wp::extract(var_p, var_135);
    var_140 = wp::extract(var_v3, var_134);
    var_141 = wp::mul(var_139, var_140);
    var_142 = wp::add(var_138, var_141);
    var_143 = wp::extract(var_v2, var_134);
    var_144 = wp::extract(var_v3, var_135);
    var_145 = wp::mul(var_143, var_144);
    var_146 = wp::add(var_142, var_145);
    var_147 = wp::extract(var_p, var_134);
    var_148 = wp::extract(var_v3, var_135);
    var_149 = wp::mul(var_147, var_148);
    var_150 = wp::sub(var_146, var_149);
    var_151 = wp::extract(var_p, var_135);
    var_152 = wp::extract(var_v2, var_134);
    var_153 = wp::mul(var_151, var_152);
    var_154 = wp::sub(var_150, var_153);
    var_155 = wp::extract(var_v3, var_134);
    var_156 = wp::extract(var_v2, var_135);
    var_157 = wp::mul(var_155, var_156);
    var_158 = wp::sub(var_154, var_157);
    // C32 = p[x] * v3[y] + p[y] * v1[x] + v3[x] * v1[y] - p[x] * v1[y] - p[y] * v3[x] - v1[x] * v3[y]       <L 735>
    var_159 = wp::extract(var_p, var_134);
    var_160 = wp::extract(var_v3, var_135);
    var_161 = wp::mul(var_159, var_160);
    var_162 = wp::extract(var_p, var_135);
    var_163 = wp::extract(var_v1, var_134);
    var_164 = wp::mul(var_162, var_163);
    var_165 = wp::add(var_161, var_164);
    var_166 = wp::extract(var_v3, var_134);
    var_167 = wp::extract(var_v1, var_135);
    var_168 = wp::mul(var_166, var_167);
    var_169 = wp::add(var_165, var_168);
    var_170 = wp::extract(var_p, var_134);
    var_171 = wp::extract(var_v1, var_135);
    var_172 = wp::mul(var_170, var_171);
    var_173 = wp::sub(var_169, var_172);
    var_174 = wp::extract(var_p, var_135);
    var_175 = wp::extract(var_v3, var_134);
    var_176 = wp::mul(var_174, var_175);
    var_177 = wp::sub(var_173, var_176);
    var_178 = wp::extract(var_v1, var_134);
    var_179 = wp::extract(var_v3, var_135);
    var_180 = wp::mul(var_178, var_179);
    var_181 = wp::sub(var_177, var_180);
    // C33 = p[x] * v1[y] + p[y] * v2[x] + v1[x] * v2[y] - p[x] * v2[y] - p[y] * v1[x] - v2[x] * v1[y]       <L 738>
    var_182 = wp::extract(var_p, var_134);
    var_183 = wp::extract(var_v1, var_135);
    var_184 = wp::mul(var_182, var_183);
    var_185 = wp::extract(var_p, var_135);
    var_186 = wp::extract(var_v2, var_134);
    var_187 = wp::mul(var_185, var_186);
    var_188 = wp::add(var_184, var_187);
    var_189 = wp::extract(var_v1, var_134);
    var_190 = wp::extract(var_v2, var_135);
    var_191 = wp::mul(var_189, var_190);
    var_192 = wp::add(var_188, var_191);
    var_193 = wp::extract(var_p, var_134);
    var_194 = wp::extract(var_v2, var_135);
    var_195 = wp::mul(var_193, var_194);
    var_196 = wp::sub(var_192, var_195);
    var_197 = wp::extract(var_p, var_135);
    var_198 = wp::extract(var_v1, var_134);
    var_199 = wp::mul(var_197, var_198);
    var_200 = wp::sub(var_196, var_199);
    var_201 = wp::extract(var_v2, var_134);
    var_202 = wp::extract(var_v1, var_135);
    var_203 = wp::mul(var_201, var_202);
    var_204 = wp::sub(var_200, var_203);
    // return wp.vec3(C31 / M_max, C32 / M_max, C33 / M_max)                                  <L 741>
    var_205 = wp::div(var_158, var_133);
    var_206 = wp::div(var_181, var_133);
    var_207 = wp::div(var_204, var_133);
    var_208 = wp::vec_t<3, wp::float32>(var_205, var_206, var_207);
    return var_208;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:2152
static CUDA_CALLABLE void _inflate_0(
    GJKResult_0220ee01 var_result,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2,
    wp::float32 var_margin1,
    wp::float32 var_margin2,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::float32* var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    wp::vec_t<3, wp::float32>* var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32>* var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    const wp::int32 var_9 = 1;
    bool var_10;
    wp::vec_t<4, wp::int32>* var_11;
    const wp::int32 var_12 = 0;
    wp::int32 var_13;
    wp::vec_t<4, wp::int32> var_14;
    const bool var_15 = false;
    bool var_16;
    wp::int32* var_17;
    wp::range_t var_18;
    wp::int32 var_19;
    wp::int32 var_20;
    wp::vec_t<4, wp::int32>* var_21;
    wp::int32 var_22;
    wp::vec_t<4, wp::int32> var_23;
    bool var_24;
    const bool var_25 = true;
    const wp::float32 var_26 = 0.0;
    const wp::float32 var_27 = 0.0;
    const wp::float32 var_28 = 1.0;
    wp::vec_t<3, wp::float32> var_29;
    SupportPoint_e82efc60 var_30;
    wp::vec_t<3, wp::float32>* var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::vec_t<3, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    wp::mat_t<6, 3, wp::float32>* var_35;
    const wp::int32 var_36 = 3;
    wp::vec_t<3, wp::float32> var_37;
    wp::mat_t<6, 3, wp::float32> var_38;
    wp::mat_t<6, 3, wp::float32>* var_39;
    const wp::int32 var_40 = 4;
    wp::vec_t<3, wp::float32> var_41;
    wp::mat_t<6, 3, wp::float32> var_42;
    wp::mat_t<6, 3, wp::float32>* var_43;
    const wp::int32 var_44 = 5;
    wp::vec_t<3, wp::float32> var_45;
    wp::mat_t<6, 3, wp::float32> var_46;
    wp::vec_t<3, wp::float32> var_47;
    const wp::int32 var_48 = 0;
    wp::float32 var_49;
    const wp::float32 var_50 = 0.0;
    bool var_51;
    const wp::int32 var_52 = 1;
    wp::float32 var_53;
    const wp::float32 var_54 = 0.0;
    bool var_55;
    const wp::int32 var_56 = 2;
    wp::float32 var_57;
    const wp::float32 var_58 = 0.0;
    bool var_59;
    bool var_60;
    const wp::int32 var_61 = 0;
    wp::float32 var_62;
    wp::vec_t<3, wp::float32> var_63;
    const wp::int32 var_64 = 1;
    wp::float32 var_65;
    wp::vec_t<3, wp::float32> var_66;
    wp::vec_t<3, wp::float32> var_67;
    const wp::int32 var_68 = 2;
    wp::float32 var_69;
    wp::vec_t<3, wp::float32> var_70;
    wp::vec_t<3, wp::float32> var_71;
    wp::vec_t<3, wp::float32> var_72;
    wp::vec_t<3, wp::float32> var_73;
    const wp::int32 var_74 = 1;
    wp::float32 var_75;
    const wp::float32 var_76 = 0.0;
    bool var_77;
    wp::vec_t<3, wp::float32> var_78;
    const wp::int32 var_79 = 0;
    wp::float32 var_80;
    const wp::float32 var_81 = 0.0;
    bool var_82;
    wp::vec_t<3, wp::float32> var_83;
    wp::vec_t<3, wp::float32> var_84;
    wp::float32 var_85;
    wp::vec_t<3, wp::float32> var_86;
    wp::vec_t<3, wp::float32> var_87;
    wp::vec_t<3, wp::float32> var_88;
    wp::vec_t<3, wp::float32> var_89;
    wp::float32 var_90;
    wp::float32 var_91;
    wp::float32 var_92;
    wp::vec_t<3, wp::float32> var_93;
    wp::vec_t<3, wp::float32> var_94;
    wp::float32 var_95;
    wp::vec_t<3, wp::float32> var_96;
    wp::vec_t<3, wp::float32> var_97;
    wp::vec_t<3, wp::float32> var_98;
    wp::vec_t<3, wp::float32> var_99;
    const wp::float32 var_100 = 0.0;
    bool var_101;
    wp::vec_t<3, wp::float32> var_102;
    wp::vec_t<3, wp::float32> var_103;
    wp::vec_t<3, wp::float32> var_104;
    const wp::float32 var_105 = 0.0;
    bool var_106;
    wp::vec_t<3, wp::float32> var_107;
    wp::vec_t<3, wp::float32> var_108;
    wp::vec_t<3, wp::float32> var_109;
    wp::float32 var_110;
    wp::float32 var_111;
    //---------
    // forward
    // def _inflate(                                                                          <L 2153>
    // dist = result.dist                                                                     <L 2156>
    var_0 = &(var_result.dist);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // x1 = result.x1                                                                         <L 2157>
    var_3 = &(var_result.x1);
    var_5 = wp::load(var_3);
    var_4 = wp::copy(var_5);
    // x2 = result.x2                                                                         <L 2158>
    var_6 = &(var_result.x2);
    var_8 = wp::load(var_6);
    var_7 = wp::copy(var_8);
    // if geomtype1 == GeomType.HFIELD:                                                       <L 2160>
    var_10 = (var_geomtype1 == var_9);
    if (var_10) {
        // v = result.simplex_index1[0]                                                       <L 2161>
        var_11 = &(var_result.simplex_index1);
        var_14 = wp::load(var_11);
        var_13 = wp::extract(var_14, var_12);
        // is_side = bool(False)                                                              <L 2162>
        var_16 = bool(var_15);
        // for i in range(result.dim):                                                        <L 2163>
        var_17 = &(var_result.dim);
        var_19 = wp::load(var_17);
        var_18 = wp::range(var_19);
        start_for_0:;
            if (iter_cmp(var_18) == 0) goto end_for_0;
            var_20 = wp::iter_next(var_18);
            // if result.simplex_index1[i] != v:                                              <L 2164>
            var_21 = &(var_result.simplex_index1);
            var_23 = wp::load(var_21);
            var_22 = wp::extract(var_23, var_20);
            var_24 = (var_22 != var_13);
            if (var_24) {
                // is_side = True                                                             <L 2165>
                // break                                                                      <L 2166>
                wp::assign(var_16, var_25);
                goto end_for_0;
            }
            goto start_for_0;
        end_for_0:;
        // if is_side:                                                                        <L 2168>
        if (var_16) {
            // n = wp.vec3(0.0, 0.0, 1.0)                                                     <L 2169>
            var_29 = wp::vec_t<3, wp::float32>(var_26, var_27, var_28);
            // sp = support(geom2, geomtype2, x2)                                             <L 2170>
            var_30 = support_0(var_geom2, var_geomtype2, var_7);
            // x2 = sp.point - margin2 * n                                                    <L 2171>
            var_31 = &(var_30.point);
            var_32 = wp::mul(var_margin2, var_29);
            var_34 = wp::load(var_31);
            var_33 = wp::sub(var_34, var_32);
            // a = geom1.hfprism[3]                                                           <L 2174>
            var_35 = &(var_geom1.hfprism);
            var_38 = wp::load(var_35);
            var_37 = wp::extract(var_38, var_36);
            // b = geom1.hfprism[4]                                                           <L 2175>
            var_39 = &(var_geom1.hfprism);
            var_42 = wp::load(var_39);
            var_41 = wp::extract(var_42, var_40);
            // c = geom1.hfprism[5]                                                           <L 2176>
            var_43 = &(var_geom1.hfprism);
            var_46 = wp::load(var_43);
            var_45 = wp::extract(var_46, var_44);
            // coordinates = _tri_affine_coord(a, b, c, x2)                                   <L 2178>
            var_47 = _tri_affine_coord_0(var_37, var_41, var_45, var_33);
            // if coordinates[0] > 0.0 and coordinates[1] > 0.0 and coordinates[2] > 0.0:       <L 2179>
            var_49 = wp::extract(var_47, var_48);
            var_51 = (var_49 > var_50);
            var_53 = wp::extract(var_47, var_52);
            var_55 = (var_53 > var_54);
            var_57 = wp::extract(var_47, var_56);
            var_59 = (var_57 > var_58);
            var_60 = var_51 && var_55 && var_59;
            if (var_60) {
                // x1 = coordinates[0] * a + coordinates[1] * b + coordinates[2] * c          <L 2180>
                var_62 = wp::extract(var_47, var_61);
                var_63 = wp::mul(var_62, var_37);
                var_65 = wp::extract(var_47, var_64);
                var_66 = wp::mul(var_65, var_41);
                var_67 = wp::add(var_63, var_66);
                var_69 = wp::extract(var_47, var_68);
                var_70 = wp::mul(var_69, var_45);
                var_71 = wp::add(var_67, var_70);
            }
            var_72 = wp::where(var_60, var_71, var_4);
            if (!var_60) {
                // p = c                                                                      <L 2182>
                var_73 = wp::copy(var_45);
                // p = wp.where(coordinates[1] > 0.0, b, p)                                   <L 2183>
                var_75 = wp::extract(var_47, var_74);
                var_77 = (var_75 > var_76);
                var_78 = wp::where(var_77, var_41, var_73);
                // p = wp.where(coordinates[0] > 0.0, a, p)                                   <L 2184>
                var_80 = wp::extract(var_47, var_79);
                var_82 = (var_80 > var_81);
                var_83 = wp::where(var_82, var_37, var_78);
                // x1 = x2 - wp.dot(x2 - p, n) * n                                            <L 2185>
                var_84 = wp::sub(var_33, var_83);
                var_85 = wp::dot(var_84, var_29);
                var_86 = wp::mul(var_85, var_29);
                var_87 = wp::sub(var_33, var_86);
            }
            var_88 = wp::where(var_60, var_72, var_87);
            // dist = -wp.norm_l2(x1 - x2)                                                    <L 2186>
            var_89 = wp::sub(var_88, var_33);
            var_90 = norm_l2_0(var_89);
            var_91 = wp::neg(var_90);
            // return dist, x1, x2                                                            <L 2187>
            ret_0 = var_91;
            ret_1 = var_88;
            ret_2 = var_33;
            return;
        }
        var_92 = wp::where(var_16, var_91, var_1);
        var_93 = wp::where(var_16, var_88, var_4);
        var_94 = wp::where(var_16, var_33, var_7);
    }
    var_95 = wp::where(var_10, var_92, var_1);
    var_96 = wp::where(var_10, var_93, var_4);
    var_97 = wp::where(var_10, var_94, var_7);
    // n = wp.normalize(x2 - x1)                                                              <L 2189>
    var_98 = wp::sub(var_97, var_96);
    var_99 = wp::normalize(var_98);
    // if margin1 > 0.0:                                                                      <L 2190>
    var_101 = (var_margin1 > var_100);
    if (var_101) {
        // x1 += margin1 * n                                                                  <L 2191>
        var_102 = wp::mul(var_margin1, var_99);
        var_103 = wp::add(var_96, var_102);
    }
    var_104 = wp::where(var_101, var_103, var_96);
    // if margin2 > 0.0:                                                                      <L 2193>
    var_106 = (var_margin2 > var_105);
    if (var_106) {
        // x2 -= margin2 * n                                                                  <L 2194>
        var_107 = wp::mul(var_margin2, var_99);
        var_108 = wp::sub(var_97, var_107);
    }
    var_109 = wp::where(var_106, var_108, var_97);
    // dist -= margin1 + margin2                                                              <L 2195>
    var_110 = wp::add(var_margin1, var_margin2);
    var_111 = wp::sub(var_95, var_110);
    // return dist, x1, x2                                                                    <L 2196>
    ret_0 = var_111;
    ret_1 = var_104;
    ret_2 = var_109;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:800
static CUDA_CALLABLE wp::mat_t<3, 3, wp::float32> _rotmat_0(
    wp::vec_t<3, wp::float32> var_axis)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::int32 var_1 = 0;
    wp::float32 var_2;
    wp::float32 var_3;
    const wp::int32 var_4 = 1;
    wp::float32 var_5;
    wp::float32 var_6;
    const wp::int32 var_7 = 2;
    wp::float32 var_8;
    wp::float32 var_9;
    const wp::float32 var_10 = 0.86602540378;
    const wp::float32 var_11 = 0.5;
    const wp::float32 var_12 = -0.5;
    wp::mat_t<3, 3, wp::float32> var_13;
    wp::float32 var_14;
    const wp::float32 var_15 = 1.0;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    const wp::int32 var_19 = 0;
    const wp::int32 var_20 = 0;
    wp::float32 var_21;
    const wp::float32 var_22 = 1.0;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    const wp::int32 var_27 = 0;
    const wp::int32 var_28 = 1;
    wp::float32 var_29;
    const wp::float32 var_30 = 1.0;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    const wp::int32 var_35 = 0;
    const wp::int32 var_36 = 2;
    wp::float32 var_37;
    const wp::float32 var_38 = 1.0;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    const wp::int32 var_43 = 1;
    const wp::int32 var_44 = 0;
    wp::float32 var_45;
    const wp::float32 var_46 = 1.0;
    wp::float32 var_47;
    wp::float32 var_48;
    wp::float32 var_49;
    const wp::int32 var_50 = 1;
    const wp::int32 var_51 = 1;
    wp::float32 var_52;
    const wp::float32 var_53 = 1.0;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::float32 var_56;
    wp::float32 var_57;
    const wp::int32 var_58 = 1;
    const wp::int32 var_59 = 2;
    wp::float32 var_60;
    const wp::float32 var_61 = 1.0;
    wp::float32 var_62;
    wp::float32 var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    const wp::int32 var_66 = 2;
    const wp::int32 var_67 = 0;
    wp::float32 var_68;
    const wp::float32 var_69 = 1.0;
    wp::float32 var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    wp::float32 var_73;
    const wp::int32 var_74 = 2;
    const wp::int32 var_75 = 1;
    wp::float32 var_76;
    const wp::float32 var_77 = 1.0;
    wp::float32 var_78;
    wp::float32 var_79;
    wp::float32 var_80;
    const wp::int32 var_81 = 2;
    const wp::int32 var_82 = 2;
    //---------
    // forward
    // def _rotmat(axis: wp.vec3) -> wp.mat33:                                                <L 801>
    // n = wp.norm_l2(axis)                                                                   <L 802>
    var_0 = norm_l2_0(var_axis);
    // u1 = axis[0] / n                                                                       <L 803>
    var_2 = wp::extract(var_axis, var_1);
    var_3 = wp::div(var_2, var_0);
    // u2 = axis[1] / n                                                                       <L 804>
    var_5 = wp::extract(var_axis, var_4);
    var_6 = wp::div(var_5, var_0);
    // u3 = axis[2] / n                                                                       <L 805>
    var_8 = wp::extract(var_axis, var_7);
    var_9 = wp::div(var_8, var_0);
    // sin = 0.86602540378  # sin(120 deg)                                                    <L 807>
    // cos = -0.5  # cos(120 deg)                                                             <L 808>
    // R = wp.mat33()                                                                         <L 809>
    var_13 = wp::mat_t<3, 3, wp::float32>();
    // R[0, 0] = cos + u1 * u1 * (1.0 - cos)                                                  <L 810>
    var_14 = wp::mul(var_3, var_3);
    var_16 = wp::sub(var_15, var_12);
    var_17 = wp::mul(var_14, var_16);
    var_18 = wp::add(var_12, var_17);
    wp::assign_inplace(var_13, var_19, var_20, var_18);
    // R[0, 1] = u1 * u2 * (1.0 - cos) - u3 * sin                                             <L 811>
    var_21 = wp::mul(var_3, var_6);
    var_23 = wp::sub(var_22, var_12);
    var_24 = wp::mul(var_21, var_23);
    var_25 = wp::mul(var_9, var_10);
    var_26 = wp::sub(var_24, var_25);
    wp::assign_inplace(var_13, var_27, var_28, var_26);
    // R[0, 2] = u1 * u3 * (1.0 - cos) + u2 * sin                                             <L 812>
    var_29 = wp::mul(var_3, var_9);
    var_31 = wp::sub(var_30, var_12);
    var_32 = wp::mul(var_29, var_31);
    var_33 = wp::mul(var_6, var_10);
    var_34 = wp::add(var_32, var_33);
    wp::assign_inplace(var_13, var_35, var_36, var_34);
    // R[1, 0] = u2 * u1 * (1.0 - cos) + u3 * sin                                             <L 813>
    var_37 = wp::mul(var_6, var_3);
    var_39 = wp::sub(var_38, var_12);
    var_40 = wp::mul(var_37, var_39);
    var_41 = wp::mul(var_9, var_10);
    var_42 = wp::add(var_40, var_41);
    wp::assign_inplace(var_13, var_43, var_44, var_42);
    // R[1, 1] = cos + u2 * u2 * (1.0 - cos)                                                  <L 814>
    var_45 = wp::mul(var_6, var_6);
    var_47 = wp::sub(var_46, var_12);
    var_48 = wp::mul(var_45, var_47);
    var_49 = wp::add(var_12, var_48);
    wp::assign_inplace(var_13, var_50, var_51, var_49);
    // R[1, 2] = u2 * u3 * (1.0 - cos) - u1 * sin                                             <L 815>
    var_52 = wp::mul(var_6, var_9);
    var_54 = wp::sub(var_53, var_12);
    var_55 = wp::mul(var_52, var_54);
    var_56 = wp::mul(var_3, var_10);
    var_57 = wp::sub(var_55, var_56);
    wp::assign_inplace(var_13, var_58, var_59, var_57);
    // R[2, 0] = u1 * u3 * (1.0 - cos) - u2 * sin                                             <L 816>
    var_60 = wp::mul(var_3, var_9);
    var_62 = wp::sub(var_61, var_12);
    var_63 = wp::mul(var_60, var_62);
    var_64 = wp::mul(var_6, var_10);
    var_65 = wp::sub(var_63, var_64);
    wp::assign_inplace(var_13, var_66, var_67, var_65);
    // R[2, 1] = u2 * u3 * (1.0 - cos) + u1 * sin                                             <L 817>
    var_68 = wp::mul(var_6, var_9);
    var_70 = wp::sub(var_69, var_12);
    var_71 = wp::mul(var_68, var_70);
    var_72 = wp::mul(var_3, var_10);
    var_73 = wp::add(var_71, var_72);
    wp::assign_inplace(var_13, var_74, var_75, var_73);
    // R[2, 2] = cos + u3 * u3 * (1.0 - cos)                                                  <L 818>
    var_76 = wp::mul(var_9, var_9);
    var_78 = wp::sub(var_77, var_12);
    var_79 = wp::mul(var_76, var_78);
    var_80 = wp::add(var_12, var_79);
    wp::assign_inplace(var_13, var_81, var_82, var_80);
    // return R                                                                               <L 819>
    return var_13;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:216
static CUDA_CALLABLE void _epa_support_0(
    Polytope_9ab93ade var_pt,
    wp::int32 var_idx,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geom1_type,
    wp::int32 var_geom2_type,
    wp::vec_t<3, wp::float32> var_dir,
    wp::int32 & ret_0,
    wp::int32 & ret_1)
{
    //---------
    // primal vars
    SupportPoint_e82efc60 var_0;
    wp::vec_t<3, wp::float32>* var_1;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_2;
    const wp::int32 var_3 = 2;
    wp::int32 var_4;
    wp::array_t<wp::vec_t<3, wp::float32>> var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::int32* var_7;
    wp::array_t<wp::int32>* var_8;
    const wp::int32 var_9 = 2;
    wp::int32 var_10;
    wp::array_t<wp::int32> var_11;
    wp::int32 var_12;
    wp::int32* var_13;
    wp::int32 var_14;
    wp::int32 var_15;
    wp::vec_t<3, wp::float32> var_16;
    SupportPoint_e82efc60 var_17;
    wp::vec_t<3, wp::float32>* var_18;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_19;
    const wp::int32 var_20 = 2;
    wp::int32 var_21;
    const wp::int32 var_22 = 1;
    wp::int32 var_23;
    wp::array_t<wp::vec_t<3, wp::float32>> var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::int32* var_26;
    wp::array_t<wp::int32>* var_27;
    const wp::int32 var_28 = 2;
    wp::int32 var_29;
    const wp::int32 var_30 = 1;
    wp::int32 var_31;
    wp::array_t<wp::int32> var_32;
    wp::int32 var_33;
    wp::int32* var_34;
    wp::int32 var_35;
    wp::int32 var_36;
    //---------
    // forward
    // def _epa_support(                                                                      <L 217>
    // sp = support(geom1, geom1_type, dir)                                                   <L 220>
    var_0 = support_0(var_geom1, var_geom1_type, var_dir);
    // pt.vert[2 * idx] = sp.point                                                            <L 221>
    var_1 = &(var_0.point);
    var_2 = &(var_pt.vert);
    var_4 = wp::mul(var_3, var_idx);
    var_5 = wp::load(var_2);
    var_6 = wp::load(var_1);
    wp::array_store(var_5, var_4, var_6);
    // pt.vert_index[2 * idx] = sp.vertex_index                                               <L 222>
    var_7 = &(var_0.vertex_index);
    var_8 = &(var_pt.vert_index);
    var_10 = wp::mul(var_9, var_idx);
    var_11 = wp::load(var_8);
    var_12 = wp::load(var_7);
    wp::array_store(var_11, var_10, var_12);
    // index1 = sp.cached_index                                                               <L 223>
    var_13 = &(var_0.cached_index);
    var_15 = wp::load(var_13);
    var_14 = wp::copy(var_15);
    // sp = support(geom2, geom2_type, -dir)                                                  <L 225>
    var_16 = wp::neg(var_dir);
    var_17 = support_0(var_geom2, var_geom2_type, var_16);
    // pt.vert[2 * idx + 1] = sp.point                                                        <L 226>
    var_18 = &(var_17.point);
    var_19 = &(var_pt.vert);
    var_21 = wp::mul(var_20, var_idx);
    var_23 = wp::add(var_21, var_22);
    var_24 = wp::load(var_19);
    var_25 = wp::load(var_18);
    wp::array_store(var_24, var_23, var_25);
    // pt.vert_index[2 * idx + 1] = sp.vertex_index                                           <L 227>
    var_26 = &(var_17.vertex_index);
    var_27 = &(var_pt.vert_index);
    var_29 = wp::mul(var_28, var_idx);
    var_31 = wp::add(var_29, var_30);
    var_32 = wp::load(var_27);
    var_33 = wp::load(var_26);
    wp::array_store(var_32, var_31, var_33);
    // index2 = sp.cached_index                                                               <L 228>
    var_34 = &(var_17.cached_index);
    var_36 = wp::load(var_34);
    var_35 = wp::copy(var_36);
    // return index1, index2                                                                  <L 230>
    ret_0 = var_14;
    ret_1 = var_35;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:194
static CUDA_CALLABLE wp::float32 _attach_face_0(
    Polytope_9ab93ade var_pt,
    wp::int32 var_idx,
    wp::int32 var_v1,
    wp::int32 var_v2,
    wp::int32 var_v3)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::array_t<wp::int32>* var_1;
    wp::shape_t* var_2;
    const wp::int32 var_3 = 0;
    wp::int32 var_4;
    wp::shape_t var_5;
    bool var_6;
    wp::int32 var_7;
    const wp::float32 var_8 = 0.0;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_9;
    const wp::int32 var_10 = 2;
    wp::int32 var_11;
    wp::vec_t<3, wp::float32>* var_12;
    wp::array_t<wp::vec_t<3, wp::float32>> var_13;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_14;
    const wp::int32 var_15 = 2;
    wp::int32 var_16;
    const wp::int32 var_17 = 1;
    wp::int32 var_18;
    wp::vec_t<3, wp::float32>* var_19;
    wp::array_t<wp::vec_t<3, wp::float32>> var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_24;
    const wp::int32 var_25 = 2;
    wp::int32 var_26;
    wp::vec_t<3, wp::float32>* var_27;
    wp::array_t<wp::vec_t<3, wp::float32>> var_28;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_29;
    const wp::int32 var_30 = 2;
    wp::int32 var_31;
    const wp::int32 var_32 = 1;
    wp::int32 var_33;
    wp::vec_t<3, wp::float32>* var_34;
    wp::array_t<wp::vec_t<3, wp::float32>> var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_39;
    const wp::int32 var_40 = 2;
    wp::int32 var_41;
    wp::vec_t<3, wp::float32>* var_42;
    wp::array_t<wp::vec_t<3, wp::float32>> var_43;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_44;
    const wp::int32 var_45 = 2;
    wp::int32 var_46;
    const wp::int32 var_47 = 1;
    wp::int32 var_48;
    wp::vec_t<3, wp::float32>* var_49;
    wp::array_t<wp::vec_t<3, wp::float32>> var_50;
    wp::vec_t<3, wp::float32> var_51;
    wp::vec_t<3, wp::float32> var_52;
    wp::vec_t<3, wp::float32> var_53;
    wp::vec_t<3, wp::float32> var_54;
    wp::int32 var_55;
    const wp::float32 var_56 = 0.0;
    const wp::int32 var_57 = 10;
    wp::int32 var_58;
    wp::int32 var_59;
    const wp::int32 var_60 = 20;
    wp::int32 var_61;
    wp::int32 var_62;
    wp::array_t<wp::int32>* var_63;
    wp::array_t<wp::int32> var_64;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_65;
    wp::array_t<wp::vec_t<3, wp::float32>> var_66;
    wp::float32 var_67;
    wp::array_t<wp::float32>* var_68;
    wp::array_t<wp::float32> var_69;
    wp::array_t<wp::float32>* var_70;
    wp::float32* var_71;
    wp::array_t<wp::float32> var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    //---------
    // forward
    // def _attach_face(pt: Polytope, idx: int, v1: int, v2: int, v3: int) -> float:          <L 195>
    // if pt.nface == pt.face.shape[0]:                                                       <L 197>
    var_0 = &(var_pt.nface);
    var_1 = &(var_pt.face);
    var_2 = &(var_1->shape);
    var_5 = wp::load(var_2);
    var_4 = wp::extract(var_5, var_3);
    var_7 = wp::load(var_0);
    var_6 = (var_7 == var_4);
    if (var_6) {
        // return 0.0                                                                         <L 198>
        return var_8;
    }
    // p1 = pt.vert[2 * v1] - pt.vert[2 * v1 + 1]                                             <L 201>
    var_9 = &(var_pt.vert);
    var_11 = wp::mul(var_10, var_v1);
    var_13 = wp::load(var_9);
    var_12 = wp::address(var_13, var_11);
    var_14 = &(var_pt.vert);
    var_16 = wp::mul(var_15, var_v1);
    var_18 = wp::add(var_16, var_17);
    var_20 = wp::load(var_14);
    var_19 = wp::address(var_20, var_18);
    var_22 = wp::load(var_12);
    var_23 = wp::load(var_19);
    var_21 = wp::sub(var_22, var_23);
    // p2 = pt.vert[2 * v2] - pt.vert[2 * v2 + 1]                                             <L 202>
    var_24 = &(var_pt.vert);
    var_26 = wp::mul(var_25, var_v2);
    var_28 = wp::load(var_24);
    var_27 = wp::address(var_28, var_26);
    var_29 = &(var_pt.vert);
    var_31 = wp::mul(var_30, var_v2);
    var_33 = wp::add(var_31, var_32);
    var_35 = wp::load(var_29);
    var_34 = wp::address(var_35, var_33);
    var_37 = wp::load(var_27);
    var_38 = wp::load(var_34);
    var_36 = wp::sub(var_37, var_38);
    // p3 = pt.vert[2 * v3] - pt.vert[2 * v3 + 1]                                             <L 203>
    var_39 = &(var_pt.vert);
    var_41 = wp::mul(var_40, var_v3);
    var_43 = wp::load(var_39);
    var_42 = wp::address(var_43, var_41);
    var_44 = &(var_pt.vert);
    var_46 = wp::mul(var_45, var_v3);
    var_48 = wp::add(var_46, var_47);
    var_50 = wp::load(var_44);
    var_49 = wp::address(var_50, var_48);
    var_52 = wp::load(var_42);
    var_53 = wp::load(var_49);
    var_51 = wp::sub(var_52, var_53);
    // r, ret = _project_origin_plane(p3, p2, p1)                                             <L 204>
    _project_origin_plane_0(var_51, var_36, var_21, var_54, var_55);
    // if ret:                                                                                <L 205>
    if (var_55) {
        // return 0.0                                                                         <L 206>
        return var_56;
    }
    // face = v1 + (v2 << 10) + (v3 << 20)                                                    <L 208>
    var_58 = wp::lshift(var_v2, var_57);
    var_59 = wp::add(var_v1, var_58);
    var_61 = wp::lshift(var_v3, var_60);
    var_62 = wp::add(var_59, var_61);
    // pt.face[idx] = face                                                                    <L 209>
    var_63 = &(var_pt.face);
    var_64 = wp::load(var_63);
    wp::array_store(var_64, var_idx, var_62);
    // pt.face_pr[idx] = r                                                                    <L 210>
    var_65 = &(var_pt.face_pr);
    var_66 = wp::load(var_65);
    wp::array_store(var_66, var_idx, var_54);
    // pt.face_norm2[idx] = wp.dot(r, r)                                                      <L 212>
    var_67 = wp::dot(var_54, var_54);
    var_68 = &(var_pt.face_norm2);
    var_69 = wp::load(var_68);
    wp::array_store(var_69, var_idx, var_67);
    // return pt.face_norm2[idx]                                                              <L 213>
    var_70 = &(var_pt.face_norm2);
    var_72 = wp::load(var_70);
    var_71 = wp::address(var_72, var_idx);
    var_74 = wp::load(var_71);
    var_73 = wp::copy(var_74);
    return var_73;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:761
static CUDA_CALLABLE GJKResult_0220ee01 _replace_simplex3_0(
    Polytope_9ab93ade var_pt,
    wp::int32 var_v1,
    wp::int32 var_v2,
    wp::int32 var_v3)
{
    //---------
    // primal vars
    GJKResult_0220ee01 var_0;
    wp::mat_t<4, 3, wp::float32> var_1;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_2;
    const wp::int32 var_3 = 2;
    wp::int32 var_4;
    wp::vec_t<3, wp::float32>* var_5;
    wp::array_t<wp::vec_t<3, wp::float32>> var_6;
    const wp::int32 var_7 = 0;
    wp::vec_t<3, wp::float32> var_8;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_9;
    const wp::int32 var_10 = 2;
    wp::int32 var_11;
    wp::vec_t<3, wp::float32>* var_12;
    wp::array_t<wp::vec_t<3, wp::float32>> var_13;
    const wp::int32 var_14 = 1;
    wp::vec_t<3, wp::float32> var_15;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_16;
    const wp::int32 var_17 = 2;
    wp::int32 var_18;
    wp::vec_t<3, wp::float32>* var_19;
    wp::array_t<wp::vec_t<3, wp::float32>> var_20;
    const wp::int32 var_21 = 2;
    wp::vec_t<3, wp::float32> var_22;
    wp::mat_t<4, 3, wp::float32> var_23;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_24;
    const wp::int32 var_25 = 2;
    wp::int32 var_26;
    const wp::int32 var_27 = 1;
    wp::int32 var_28;
    wp::vec_t<3, wp::float32>* var_29;
    wp::array_t<wp::vec_t<3, wp::float32>> var_30;
    const wp::int32 var_31 = 0;
    wp::vec_t<3, wp::float32> var_32;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_33;
    const wp::int32 var_34 = 2;
    wp::int32 var_35;
    const wp::int32 var_36 = 1;
    wp::int32 var_37;
    wp::vec_t<3, wp::float32>* var_38;
    wp::array_t<wp::vec_t<3, wp::float32>> var_39;
    const wp::int32 var_40 = 1;
    wp::vec_t<3, wp::float32> var_41;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_42;
    const wp::int32 var_43 = 2;
    wp::int32 var_44;
    const wp::int32 var_45 = 1;
    wp::int32 var_46;
    wp::vec_t<3, wp::float32>* var_47;
    wp::array_t<wp::vec_t<3, wp::float32>> var_48;
    const wp::int32 var_49 = 2;
    wp::vec_t<3, wp::float32> var_50;
    wp::mat_t<4, 3, wp::float32> var_51;
    const wp::int32 var_52 = 0;
    wp::vec_t<3, wp::float32> var_53;
    const wp::int32 var_54 = 0;
    wp::vec_t<3, wp::float32> var_55;
    wp::vec_t<3, wp::float32> var_56;
    const wp::int32 var_57 = 0;
    const wp::int32 var_58 = 1;
    wp::vec_t<3, wp::float32> var_59;
    const wp::int32 var_60 = 1;
    wp::vec_t<3, wp::float32> var_61;
    wp::vec_t<3, wp::float32> var_62;
    const wp::int32 var_63 = 1;
    const wp::int32 var_64 = 2;
    wp::vec_t<3, wp::float32> var_65;
    const wp::int32 var_66 = 2;
    wp::vec_t<3, wp::float32> var_67;
    wp::vec_t<3, wp::float32> var_68;
    const wp::int32 var_69 = 2;
    wp::vec_t<4, wp::int32> var_70;
    wp::array_t<wp::int32>* var_71;
    const wp::int32 var_72 = 2;
    wp::int32 var_73;
    wp::int32* var_74;
    wp::array_t<wp::int32> var_75;
    const wp::int32 var_76 = 0;
    wp::int32 var_77;
    wp::array_t<wp::int32>* var_78;
    const wp::int32 var_79 = 2;
    wp::int32 var_80;
    wp::int32* var_81;
    wp::array_t<wp::int32> var_82;
    const wp::int32 var_83 = 1;
    wp::int32 var_84;
    wp::array_t<wp::int32>* var_85;
    const wp::int32 var_86 = 2;
    wp::int32 var_87;
    wp::int32* var_88;
    wp::array_t<wp::int32> var_89;
    const wp::int32 var_90 = 2;
    wp::int32 var_91;
    wp::vec_t<4, wp::int32> var_92;
    wp::array_t<wp::int32>* var_93;
    const wp::int32 var_94 = 2;
    wp::int32 var_95;
    const wp::int32 var_96 = 1;
    wp::int32 var_97;
    wp::int32* var_98;
    wp::array_t<wp::int32> var_99;
    const wp::int32 var_100 = 0;
    wp::int32 var_101;
    wp::array_t<wp::int32>* var_102;
    const wp::int32 var_103 = 2;
    wp::int32 var_104;
    const wp::int32 var_105 = 1;
    wp::int32 var_106;
    wp::int32* var_107;
    wp::array_t<wp::int32> var_108;
    const wp::int32 var_109 = 1;
    wp::int32 var_110;
    wp::array_t<wp::int32>* var_111;
    const wp::int32 var_112 = 2;
    wp::int32 var_113;
    const wp::int32 var_114 = 1;
    wp::int32 var_115;
    wp::int32* var_116;
    wp::array_t<wp::int32> var_117;
    const wp::int32 var_118 = 2;
    wp::int32 var_119;
    wp::mat_t<4, 3, wp::float32>* var_120;
    wp::mat_t<4, 3, wp::float32>* var_121;
    wp::mat_t<4, 3, wp::float32>* var_122;
    wp::vec_t<4, wp::int32>* var_123;
    wp::vec_t<4, wp::int32>* var_124;
    //---------
    // forward
    // def _replace_simplex3(pt: Polytope, v1: int, v2: int, v3: int) -> GJKResult:           <L 762>
    // result = GJKResult()                                                                   <L 763>
    var_0 = GJKResult_0220ee01();
    // simplex1 = mat43()                                                                     <L 766>
    var_1 = wp::mat_t<4, 3, wp::float32>();
    // simplex1[0] = pt.vert[2 * v1]                                                          <L 767>
    var_2 = &(var_pt.vert);
    var_4 = wp::mul(var_3, var_v1);
    var_6 = wp::load(var_2);
    var_5 = wp::address(var_6, var_4);
    var_8 = wp::load(var_5);
    wp::assign_inplace(var_1, var_7, var_8);
    // simplex1[1] = pt.vert[2 * v2]                                                          <L 768>
    var_9 = &(var_pt.vert);
    var_11 = wp::mul(var_10, var_v2);
    var_13 = wp::load(var_9);
    var_12 = wp::address(var_13, var_11);
    var_15 = wp::load(var_12);
    wp::assign_inplace(var_1, var_14, var_15);
    // simplex1[2] = pt.vert[2 * v3]                                                          <L 769>
    var_16 = &(var_pt.vert);
    var_18 = wp::mul(var_17, var_v3);
    var_20 = wp::load(var_16);
    var_19 = wp::address(var_20, var_18);
    var_22 = wp::load(var_19);
    wp::assign_inplace(var_1, var_21, var_22);
    // simplex2 = mat43()                                                                     <L 771>
    var_23 = wp::mat_t<4, 3, wp::float32>();
    // simplex2[0] = pt.vert[2 * v1 + 1]                                                      <L 772>
    var_24 = &(var_pt.vert);
    var_26 = wp::mul(var_25, var_v1);
    var_28 = wp::add(var_26, var_27);
    var_30 = wp::load(var_24);
    var_29 = wp::address(var_30, var_28);
    var_32 = wp::load(var_29);
    wp::assign_inplace(var_23, var_31, var_32);
    // simplex2[1] = pt.vert[2 * v2 + 1]                                                      <L 773>
    var_33 = &(var_pt.vert);
    var_35 = wp::mul(var_34, var_v2);
    var_37 = wp::add(var_35, var_36);
    var_39 = wp::load(var_33);
    var_38 = wp::address(var_39, var_37);
    var_41 = wp::load(var_38);
    wp::assign_inplace(var_23, var_40, var_41);
    // simplex2[2] = pt.vert[2 * v3 + 1]                                                      <L 774>
    var_42 = &(var_pt.vert);
    var_44 = wp::mul(var_43, var_v3);
    var_46 = wp::add(var_44, var_45);
    var_48 = wp::load(var_42);
    var_47 = wp::address(var_48, var_46);
    var_50 = wp::load(var_47);
    wp::assign_inplace(var_23, var_49, var_50);
    // simplex = mat43()                                                                      <L 776>
    var_51 = wp::mat_t<4, 3, wp::float32>();
    // simplex[0] = simplex1[0] - simplex2[0]                                                 <L 777>
    var_53 = wp::extract(var_1, var_52);
    var_55 = wp::extract(var_23, var_54);
    var_56 = wp::sub(var_53, var_55);
    wp::assign_inplace(var_51, var_57, var_56);
    // simplex[1] = simplex1[1] - simplex2[1]                                                 <L 778>
    var_59 = wp::extract(var_1, var_58);
    var_61 = wp::extract(var_23, var_60);
    var_62 = wp::sub(var_59, var_61);
    wp::assign_inplace(var_51, var_63, var_62);
    // simplex[2] = simplex1[2] - simplex2[2]                                                 <L 779>
    var_65 = wp::extract(var_1, var_64);
    var_67 = wp::extract(var_23, var_66);
    var_68 = wp::sub(var_65, var_67);
    wp::assign_inplace(var_51, var_69, var_68);
    // simplex_index1 = wp.vec4i()                                                            <L 781>
    var_70 = wp::vec_t<4, wp::int32>();
    // simplex_index1[0] = pt.vert_index[2 * v1]                                              <L 782>
    var_71 = &(var_pt.vert_index);
    var_73 = wp::mul(var_72, var_v1);
    var_75 = wp::load(var_71);
    var_74 = wp::address(var_75, var_73);
    var_77 = wp::load(var_74);
    wp::assign_inplace(var_70, var_76, var_77);
    // simplex_index1[1] = pt.vert_index[2 * v2]                                              <L 783>
    var_78 = &(var_pt.vert_index);
    var_80 = wp::mul(var_79, var_v2);
    var_82 = wp::load(var_78);
    var_81 = wp::address(var_82, var_80);
    var_84 = wp::load(var_81);
    wp::assign_inplace(var_70, var_83, var_84);
    // simplex_index1[2] = pt.vert_index[2 * v3]                                              <L 784>
    var_85 = &(var_pt.vert_index);
    var_87 = wp::mul(var_86, var_v3);
    var_89 = wp::load(var_85);
    var_88 = wp::address(var_89, var_87);
    var_91 = wp::load(var_88);
    wp::assign_inplace(var_70, var_90, var_91);
    // simplex_index2 = wp.vec4i()                                                            <L 786>
    var_92 = wp::vec_t<4, wp::int32>();
    // simplex_index2[0] = pt.vert_index[2 * v1 + 1]                                          <L 787>
    var_93 = &(var_pt.vert_index);
    var_95 = wp::mul(var_94, var_v1);
    var_97 = wp::add(var_95, var_96);
    var_99 = wp::load(var_93);
    var_98 = wp::address(var_99, var_97);
    var_101 = wp::load(var_98);
    wp::assign_inplace(var_92, var_100, var_101);
    // simplex_index2[1] = pt.vert_index[2 * v2 + 1]                                          <L 788>
    var_102 = &(var_pt.vert_index);
    var_104 = wp::mul(var_103, var_v2);
    var_106 = wp::add(var_104, var_105);
    var_108 = wp::load(var_102);
    var_107 = wp::address(var_108, var_106);
    var_110 = wp::load(var_107);
    wp::assign_inplace(var_92, var_109, var_110);
    // simplex_index2[2] = pt.vert_index[2 * v3 + 1]                                          <L 789>
    var_111 = &(var_pt.vert_index);
    var_113 = wp::mul(var_112, var_v3);
    var_115 = wp::add(var_113, var_114);
    var_117 = wp::load(var_111);
    var_116 = wp::address(var_117, var_115);
    var_119 = wp::load(var_116);
    wp::assign_inplace(var_92, var_118, var_119);
    // result.simplex = simplex                                                               <L 791>
    var_120 = &(var_0.simplex);
    wp::store(var_120, var_51);
    // result.simplex1 = simplex1                                                             <L 792>
    var_121 = &(var_0.simplex1);
    wp::store(var_121, var_1);
    // result.simplex2 = simplex2                                                             <L 793>
    var_122 = &(var_0.simplex2);
    wp::store(var_122, var_23);
    // result.simplex_index1 = simplex_index1                                                 <L 794>
    var_123 = &(var_0.simplex_index1);
    wp::store(var_123, var_70);
    // result.simplex_index2 = simplex_index2                                                 <L 795>
    var_124 = &(var_0.simplex_index2);
    wp::store(var_124, var_92);
    // return result                                                                          <L 797>
    return var_0;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:822
static CUDA_CALLABLE wp::int32 _ray_triangle_0(
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> var_v3,
    wp::vec_t<3, wp::float32> var_v4,
    wp::vec_t<3, wp::float32> var_v5)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::float32 var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::float32 var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::float32 var_11;
    const wp::float32 var_12 = 0.0;
    bool var_13;
    const wp::float32 var_14 = 0.0;
    bool var_15;
    const wp::float32 var_16 = 0.0;
    bool var_17;
    bool var_18;
    const wp::int32 var_19 = 1;
    const wp::float32 var_20 = 0.0;
    bool var_21;
    const wp::float32 var_22 = 0.0;
    bool var_23;
    const wp::float32 var_24 = 0.0;
    bool var_25;
    bool var_26;
    const wp::int32 var_27 = 1;
    const wp::int32 var_28 = -1;
    const wp::int32 var_29 = 0;
    //---------
    // forward
    // def _ray_triangle(v1: wp.vec3, v2: wp.vec3, v3: wp.vec3, v4: wp.vec3, v5: wp.vec3) -> int:       <L 823>
    // vol1 = _det3(v3 - v1, v4 - v1, v2 - v1)                                                <L 824>
    var_0 = wp::sub(var_v3, var_v1);
    var_1 = wp::sub(var_v4, var_v1);
    var_2 = wp::sub(var_v2, var_v1);
    var_3 = _det3_0(var_0, var_1, var_2);
    // vol2 = _det3(v4 - v1, v5 - v1, v2 - v1)                                                <L 825>
    var_4 = wp::sub(var_v4, var_v1);
    var_5 = wp::sub(var_v5, var_v1);
    var_6 = wp::sub(var_v2, var_v1);
    var_7 = _det3_0(var_4, var_5, var_6);
    // vol3 = _det3(v5 - v1, v3 - v1, v2 - v1)                                                <L 826>
    var_8 = wp::sub(var_v5, var_v1);
    var_9 = wp::sub(var_v3, var_v1);
    var_10 = wp::sub(var_v2, var_v1);
    var_11 = _det3_0(var_8, var_9, var_10);
    // if vol1 >= 0.0 and vol2 >= 0.0 and vol3 >= 0.0:                                        <L 828>
    var_13 = (var_3 >= var_12);
    var_15 = (var_7 >= var_14);
    var_17 = (var_11 >= var_16);
    var_18 = var_13 && var_15 && var_17;
    if (var_18) {
        // return 1                                                                           <L 829>
        return var_19;
    }
    // if vol1 <= 0.0 and vol2 <= 0.0 and vol3 <= 0.0:                                        <L 830>
    var_21 = (var_3 <= var_20);
    var_23 = (var_7 <= var_22);
    var_25 = (var_11 <= var_24);
    var_26 = var_21 && var_23 && var_25;
    if (var_26) {
        // return -1                                                                          <L 831>
        return var_28;
    }
    // return 0                                                                               <L 832>
    return var_29;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:936
static CUDA_CALLABLE void _polytope2_0(
    Polytope_9ab93ade var_pt,
    wp::mat_t<4, 3, wp::float32> var_simplex,
    wp::mat_t<4, 3, wp::float32> var_simplex1,
    wp::mat_t<4, 3, wp::float32> var_simplex2,
    wp::vec_t<4, wp::int32> var_simplex_index1,
    wp::vec_t<4, wp::int32> var_simplex_index2,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2,
    Polytope_9ab93ade & ret_0,
    GJKResult_0220ee01 & ret_1)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::vec_t<3, wp::float32> var_1;
    const wp::int32 var_2 = 0;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    const wp::float32 var_5 = 1e+30;
    wp::float32 var_6;
    const wp::int32 var_7 = 0;
    const wp::int32 var_8 = 0;
    wp::float32 var_9;
    wp::float32 var_10;
    bool var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::int32 var_14;
    wp::float32 var_15;
    wp::int32 var_16;
    const wp::int32 var_17 = 1;
    wp::float32 var_18;
    wp::float32 var_19;
    bool var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::int32 var_23;
    wp::float32 var_24;
    wp::int32 var_25;
    const wp::int32 var_26 = 2;
    wp::float32 var_27;
    wp::float32 var_28;
    bool var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::int32 var_32;
    wp::float32 var_33;
    wp::int32 var_34;
    const wp::float32 var_35 = 0.0;
    const wp::float32 var_36 = 0.0;
    const wp::float32 var_37 = 0.0;
    wp::vec_t<3, wp::float32> var_38;
    const wp::float32 var_39 = 1.0;
    wp::vec_t<3, wp::float32> var_40;
    wp::mat_t<3, 3, wp::float32> var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::vec_t<3, wp::float32> var_43;
    const wp::int32 var_44 = 0;
    wp::vec_t<3, wp::float32> var_45;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_46;
    const wp::int32 var_47 = 0;
    wp::array_t<wp::vec_t<3, wp::float32>> var_48;
    const wp::int32 var_49 = 0;
    wp::vec_t<3, wp::float32> var_50;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_51;
    const wp::int32 var_52 = 1;
    wp::array_t<wp::vec_t<3, wp::float32>> var_53;
    const wp::int32 var_54 = 1;
    wp::vec_t<3, wp::float32> var_55;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_56;
    const wp::int32 var_57 = 2;
    wp::array_t<wp::vec_t<3, wp::float32>> var_58;
    const wp::int32 var_59 = 1;
    wp::vec_t<3, wp::float32> var_60;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_61;
    const wp::int32 var_62 = 3;
    wp::array_t<wp::vec_t<3, wp::float32>> var_63;
    const wp::int32 var_64 = 0;
    wp::int32 var_65;
    wp::array_t<wp::int32>* var_66;
    const wp::int32 var_67 = 0;
    wp::array_t<wp::int32> var_68;
    const wp::int32 var_69 = 0;
    wp::int32 var_70;
    wp::array_t<wp::int32>* var_71;
    const wp::int32 var_72 = 1;
    wp::array_t<wp::int32> var_73;
    const wp::int32 var_74 = 1;
    wp::int32 var_75;
    wp::array_t<wp::int32>* var_76;
    const wp::int32 var_77 = 2;
    wp::array_t<wp::int32> var_78;
    const wp::int32 var_79 = 1;
    wp::int32 var_80;
    wp::array_t<wp::int32>* var_81;
    const wp::int32 var_82 = 3;
    wp::array_t<wp::int32> var_83;
    const wp::int32 var_84 = 2;
    wp::float32 var_85;
    wp::vec_t<3, wp::float32> var_86;
    wp::int32 var_87;
    wp::int32 var_88;
    const wp::int32 var_89 = 3;
    wp::float32 var_90;
    wp::vec_t<3, wp::float32> var_91;
    wp::int32 var_92;
    wp::int32 var_93;
    const wp::int32 var_94 = 4;
    wp::float32 var_95;
    wp::vec_t<3, wp::float32> var_96;
    wp::int32 var_97;
    wp::int32 var_98;
    const wp::int32 var_99 = 0;
    const wp::int32 var_100 = 0;
    const wp::int32 var_101 = 2;
    const wp::int32 var_102 = 3;
    wp::float32 var_103;
    const wp::float32 var_104 = 1e-10;
    bool var_105;
    const wp::int32 var_106 = 1;
    const wp::int32 var_107 = -1;
    wp::int32* var_108;
    const wp::int32 var_109 = 0;
    const wp::int32 var_110 = 2;
    const wp::int32 var_111 = 3;
    GJKResult_0220ee01 var_112;
    const wp::int32 var_113 = 1;
    const wp::int32 var_114 = 0;
    const wp::int32 var_115 = 4;
    const wp::int32 var_116 = 2;
    wp::float32 var_117;
    bool var_118;
    const wp::int32 var_119 = 1;
    const wp::int32 var_120 = -1;
    wp::int32* var_121;
    const wp::int32 var_122 = 0;
    const wp::int32 var_123 = 4;
    const wp::int32 var_124 = 2;
    GJKResult_0220ee01 var_125;
    const wp::int32 var_126 = 2;
    const wp::int32 var_127 = 0;
    const wp::int32 var_128 = 3;
    const wp::int32 var_129 = 4;
    wp::float32 var_130;
    bool var_131;
    const wp::int32 var_132 = 1;
    const wp::int32 var_133 = -1;
    wp::int32* var_134;
    const wp::int32 var_135 = 0;
    const wp::int32 var_136 = 3;
    const wp::int32 var_137 = 4;
    GJKResult_0220ee01 var_138;
    const wp::int32 var_139 = 3;
    const wp::int32 var_140 = 1;
    const wp::int32 var_141 = 3;
    const wp::int32 var_142 = 2;
    wp::float32 var_143;
    bool var_144;
    const wp::int32 var_145 = 1;
    const wp::int32 var_146 = -1;
    wp::int32* var_147;
    const wp::int32 var_148 = 1;
    const wp::int32 var_149 = 3;
    const wp::int32 var_150 = 2;
    GJKResult_0220ee01 var_151;
    const wp::int32 var_152 = 4;
    const wp::int32 var_153 = 1;
    const wp::int32 var_154 = 2;
    const wp::int32 var_155 = 4;
    wp::float32 var_156;
    bool var_157;
    const wp::int32 var_158 = 1;
    const wp::int32 var_159 = -1;
    wp::int32* var_160;
    const wp::int32 var_161 = 1;
    const wp::int32 var_162 = 2;
    const wp::int32 var_163 = 4;
    GJKResult_0220ee01 var_164;
    const wp::int32 var_165 = 5;
    const wp::int32 var_166 = 1;
    const wp::int32 var_167 = 4;
    const wp::int32 var_168 = 3;
    wp::float32 var_169;
    bool var_170;
    const wp::int32 var_171 = 1;
    const wp::int32 var_172 = -1;
    wp::int32* var_173;
    const wp::int32 var_174 = 1;
    const wp::int32 var_175 = 4;
    const wp::int32 var_176 = 3;
    GJKResult_0220ee01 var_177;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_178;
    const wp::int32 var_179 = 4;
    wp::vec_t<3, wp::float32>* var_180;
    wp::array_t<wp::vec_t<3, wp::float32>> var_181;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_182;
    const wp::int32 var_183 = 5;
    wp::vec_t<3, wp::float32>* var_184;
    wp::array_t<wp::vec_t<3, wp::float32>> var_185;
    wp::vec_t<3, wp::float32> var_186;
    wp::vec_t<3, wp::float32> var_187;
    wp::vec_t<3, wp::float32> var_188;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_189;
    const wp::int32 var_190 = 6;
    wp::vec_t<3, wp::float32>* var_191;
    wp::array_t<wp::vec_t<3, wp::float32>> var_192;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_193;
    const wp::int32 var_194 = 7;
    wp::vec_t<3, wp::float32>* var_195;
    wp::array_t<wp::vec_t<3, wp::float32>> var_196;
    wp::vec_t<3, wp::float32> var_197;
    wp::vec_t<3, wp::float32> var_198;
    wp::vec_t<3, wp::float32> var_199;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_200;
    const wp::int32 var_201 = 8;
    wp::vec_t<3, wp::float32>* var_202;
    wp::array_t<wp::vec_t<3, wp::float32>> var_203;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_204;
    const wp::int32 var_205 = 9;
    wp::vec_t<3, wp::float32>* var_206;
    wp::array_t<wp::vec_t<3, wp::float32>> var_207;
    wp::vec_t<3, wp::float32> var_208;
    wp::vec_t<3, wp::float32> var_209;
    wp::vec_t<3, wp::float32> var_210;
    const wp::int32 var_211 = 0;
    wp::vec_t<3, wp::float32> var_212;
    const wp::int32 var_213 = 1;
    wp::vec_t<3, wp::float32> var_214;
    wp::int32 var_215;
    bool var_216;
    const wp::int32 var_217 = 1;
    wp::int32* var_218;
    GJKResult_0220ee01 var_219;
    const wp::int32 var_220 = 5;
    wp::int32* var_221;
    const wp::int32 var_222 = 6;
    wp::int32* var_223;
    const wp::int32 var_224 = 0;
    wp::int32* var_225;
    GJKResult_0220ee01 var_226;
    //---------
    // forward
    // def _polytope2(                                                                        <L 937>
    // diff = simplex[1] - simplex[0]                                                         <L 951>
    var_1 = wp::extract(var_simplex, var_0);
    var_3 = wp::extract(var_simplex, var_2);
    var_4 = wp::sub(var_1, var_3);
    // value = FLOAT_MAX                                                                      <L 954>
    var_6 = wp::copy(var_5);
    // index = 0                                                                              <L 955>
    // for i in range(3):                                                                     <L 956>
    // if wp.abs(diff[i]) < value:                                                            <L 957>
    var_9 = wp::extract(var_4, var_8);
    var_10 = wp::abs(var_9);
    var_11 = (var_10 < var_6);
    if (var_11) {
        // value = wp.abs(diff[i])                                                            <L 958>
        var_12 = wp::extract(var_4, var_8);
        var_13 = wp::abs(var_12);
        // index = i                                                                          <L 959>
        var_14 = wp::copy(var_8);
    }
    var_15 = wp::where(var_11, var_13, var_6);
    var_16 = wp::where(var_11, var_14, var_7);
    // if wp.abs(diff[i]) < value:                                                            <L 957>
    var_18 = wp::extract(var_4, var_17);
    var_19 = wp::abs(var_18);
    var_20 = (var_19 < var_15);
    if (var_20) {
        // value = wp.abs(diff[i])                                                            <L 958>
        var_21 = wp::extract(var_4, var_17);
        var_22 = wp::abs(var_21);
        // index = i                                                                          <L 959>
        var_23 = wp::copy(var_17);
    }
    var_24 = wp::where(var_20, var_22, var_15);
    var_25 = wp::where(var_20, var_23, var_16);
    // if wp.abs(diff[i]) < value:                                                            <L 957>
    var_27 = wp::extract(var_4, var_26);
    var_28 = wp::abs(var_27);
    var_29 = (var_28 < var_24);
    if (var_29) {
        // value = wp.abs(diff[i])                                                            <L 958>
        var_30 = wp::extract(var_4, var_26);
        var_31 = wp::abs(var_30);
        // index = i                                                                          <L 959>
        var_32 = wp::copy(var_26);
    }
    var_33 = wp::where(var_29, var_31, var_24);
    var_34 = wp::where(var_29, var_32, var_25);
    // e = wp.vec3(0.0, 0.0, 0.0)                                                             <L 962>
    var_38 = wp::vec_t<3, wp::float32>(var_35, var_36, var_37);
    // e[index] = 1.0                                                                         <L 963>
    wp::assign_inplace(var_38, var_34, var_39);
    // d1 = wp.cross(e, diff)                                                                 <L 964>
    var_40 = wp::cross(var_38, var_4);
    // R = _rotmat(diff)                                                                      <L 967>
    var_41 = _rotmat_0(var_4);
    // d2 = R @ d1                                                                            <L 968>
    var_42 = wp::mul(var_41, var_40);
    // d3 = R @ d2                                                                            <L 969>
    var_43 = wp::mul(var_41, var_42);
    // pt.vert[0] = simplex1[0]                                                               <L 972>
    var_45 = wp::extract(var_simplex1, var_44);
    var_46 = &(var_pt.vert);
    var_48 = wp::load(var_46);
    wp::array_store(var_48, var_47, var_45);
    // pt.vert[1] = simplex2[0]                                                               <L 973>
    var_50 = wp::extract(var_simplex2, var_49);
    var_51 = &(var_pt.vert);
    var_53 = wp::load(var_51);
    wp::array_store(var_53, var_52, var_50);
    // pt.vert[2] = simplex1[1]                                                               <L 974>
    var_55 = wp::extract(var_simplex1, var_54);
    var_56 = &(var_pt.vert);
    var_58 = wp::load(var_56);
    wp::array_store(var_58, var_57, var_55);
    // pt.vert[3] = simplex2[1]                                                               <L 975>
    var_60 = wp::extract(var_simplex2, var_59);
    var_61 = &(var_pt.vert);
    var_63 = wp::load(var_61);
    wp::array_store(var_63, var_62, var_60);
    // pt.vert_index[0] = simplex_index1[0]                                                   <L 977>
    var_65 = wp::extract(var_simplex_index1, var_64);
    var_66 = &(var_pt.vert_index);
    var_68 = wp::load(var_66);
    wp::array_store(var_68, var_67, var_65);
    // pt.vert_index[1] = simplex_index2[0]                                                   <L 978>
    var_70 = wp::extract(var_simplex_index2, var_69);
    var_71 = &(var_pt.vert_index);
    var_73 = wp::load(var_71);
    wp::array_store(var_73, var_72, var_70);
    // pt.vert_index[2] = simplex_index1[1]                                                   <L 979>
    var_75 = wp::extract(var_simplex_index1, var_74);
    var_76 = &(var_pt.vert_index);
    var_78 = wp::load(var_76);
    wp::array_store(var_78, var_77, var_75);
    // pt.vert_index[3] = simplex_index2[1]                                                   <L 980>
    var_80 = wp::extract(var_simplex_index2, var_79);
    var_81 = &(var_pt.vert_index);
    var_83 = wp::load(var_81);
    wp::array_store(var_83, var_82, var_80);
    // _epa_support(pt, 2, geom1, geom2, geomtype1, geomtype2, d1 / wp.norm_l2(d1))           <L 982>
    var_85 = norm_l2_0(var_40);
    var_86 = wp::div(var_40, var_85);
    _epa_support_0(var_pt, var_84, var_geom1, var_geom2, var_geomtype1, var_geomtype2, var_86, var_87, var_88);
    // _epa_support(pt, 3, geom1, geom2, geomtype1, geomtype2, d2 / wp.norm_l2(d2))           <L 983>
    var_90 = norm_l2_0(var_42);
    var_91 = wp::div(var_42, var_90);
    _epa_support_0(var_pt, var_89, var_geom1, var_geom2, var_geomtype1, var_geomtype2, var_91, var_92, var_93);
    // _epa_support(pt, 4, geom1, geom2, geomtype1, geomtype2, d3 / wp.norm_l2(d3))           <L 984>
    var_95 = norm_l2_0(var_43);
    var_96 = wp::div(var_43, var_95);
    _epa_support_0(var_pt, var_94, var_geom1, var_geom2, var_geomtype1, var_geomtype2, var_96, var_97, var_98);
    // if _attach_face(pt, 0, 0, 2, 3) < MIN_DIST:                                            <L 987>
    var_103 = _attach_face_0(var_pt, var_99, var_100, var_101, var_102);
    var_105 = (var_103 < var_104);
    if (var_105) {
        // pt.status = -1                                                                     <L 988>
        var_108 = &(var_pt.status);
        wp::store(var_108, var_107);
        // return pt, _replace_simplex3(pt, 0, 2, 3)                                          <L 989>
        var_112 = _replace_simplex3_0(var_pt, var_109, var_110, var_111);
        ret_0 = var_pt;
        ret_1 = var_112;
        return;
    }
    // if _attach_face(pt, 1, 0, 4, 2) < MIN_DIST:                                            <L 991>
    var_117 = _attach_face_0(var_pt, var_113, var_114, var_115, var_116);
    var_118 = (var_117 < var_104);
    if (var_118) {
        // pt.status = -1                                                                     <L 992>
        var_121 = &(var_pt.status);
        wp::store(var_121, var_120);
        // return pt, _replace_simplex3(pt, 0, 4, 2)                                          <L 993>
        var_125 = _replace_simplex3_0(var_pt, var_122, var_123, var_124);
        ret_0 = var_pt;
        ret_1 = var_125;
        return;
    }
    // if _attach_face(pt, 2, 0, 3, 4) < MIN_DIST:                                            <L 995>
    var_130 = _attach_face_0(var_pt, var_126, var_127, var_128, var_129);
    var_131 = (var_130 < var_104);
    if (var_131) {
        // pt.status = -1                                                                     <L 996>
        var_134 = &(var_pt.status);
        wp::store(var_134, var_133);
        // return pt, _replace_simplex3(pt, 0, 3, 4)                                          <L 997>
        var_138 = _replace_simplex3_0(var_pt, var_135, var_136, var_137);
        ret_0 = var_pt;
        ret_1 = var_138;
        return;
    }
    // if _attach_face(pt, 3, 1, 3, 2) < MIN_DIST:                                            <L 999>
    var_143 = _attach_face_0(var_pt, var_139, var_140, var_141, var_142);
    var_144 = (var_143 < var_104);
    if (var_144) {
        // pt.status = -1                                                                     <L 1000>
        var_147 = &(var_pt.status);
        wp::store(var_147, var_146);
        // return pt, _replace_simplex3(pt, 1, 3, 2)                                          <L 1001>
        var_151 = _replace_simplex3_0(var_pt, var_148, var_149, var_150);
        ret_0 = var_pt;
        ret_1 = var_151;
        return;
    }
    // if _attach_face(pt, 4, 1, 2, 4) < MIN_DIST:                                            <L 1003>
    var_156 = _attach_face_0(var_pt, var_152, var_153, var_154, var_155);
    var_157 = (var_156 < var_104);
    if (var_157) {
        // pt.status = -1                                                                     <L 1004>
        var_160 = &(var_pt.status);
        wp::store(var_160, var_159);
        // return pt, _replace_simplex3(pt, 1, 2, 4)                                          <L 1005>
        var_164 = _replace_simplex3_0(var_pt, var_161, var_162, var_163);
        ret_0 = var_pt;
        ret_1 = var_164;
        return;
    }
    // if _attach_face(pt, 5, 1, 4, 3) < MIN_DIST:                                            <L 1007>
    var_169 = _attach_face_0(var_pt, var_165, var_166, var_167, var_168);
    var_170 = (var_169 < var_104);
    if (var_170) {
        // pt.status = -1                                                                     <L 1008>
        var_173 = &(var_pt.status);
        wp::store(var_173, var_172);
        // return pt, _replace_simplex3(pt, 1, 4, 3)                                          <L 1009>
        var_177 = _replace_simplex3_0(var_pt, var_174, var_175, var_176);
        ret_0 = var_pt;
        ret_1 = var_177;
        return;
    }
    // v2 = pt.vert[4] - pt.vert[5]                                                           <L 1012>
    var_178 = &(var_pt.vert);
    var_181 = wp::load(var_178);
    var_180 = wp::address(var_181, var_179);
    var_182 = &(var_pt.vert);
    var_185 = wp::load(var_182);
    var_184 = wp::address(var_185, var_183);
    var_187 = wp::load(var_180);
    var_188 = wp::load(var_184);
    var_186 = wp::sub(var_187, var_188);
    // v3 = pt.vert[6] - pt.vert[7]                                                           <L 1013>
    var_189 = &(var_pt.vert);
    var_192 = wp::load(var_189);
    var_191 = wp::address(var_192, var_190);
    var_193 = &(var_pt.vert);
    var_196 = wp::load(var_193);
    var_195 = wp::address(var_196, var_194);
    var_198 = wp::load(var_191);
    var_199 = wp::load(var_195);
    var_197 = wp::sub(var_198, var_199);
    // v4 = pt.vert[8] - pt.vert[9]                                                           <L 1014>
    var_200 = &(var_pt.vert);
    var_203 = wp::load(var_200);
    var_202 = wp::address(var_203, var_201);
    var_204 = &(var_pt.vert);
    var_207 = wp::load(var_204);
    var_206 = wp::address(var_207, var_205);
    var_209 = wp::load(var_202);
    var_210 = wp::load(var_206);
    var_208 = wp::sub(var_209, var_210);
    // if not _ray_triangle(simplex[0], simplex[1], v2, v3, v4):                              <L 1015>
    var_212 = wp::extract(var_simplex, var_211);
    var_214 = wp::extract(var_simplex, var_213);
    var_215 = _ray_triangle_0(var_212, var_214, var_186, var_197, var_208);
    var_216 = wp::unot(var_215);
    if (var_216) {
        // pt.status = 1                                                                      <L 1016>
        var_218 = &(var_pt.status);
        wp::store(var_218, var_217);
        // return pt, GJKResult()                                                             <L 1017>
        var_219 = GJKResult_0220ee01();
        ret_0 = var_pt;
        ret_1 = var_219;
        return;
    }
    // pt.nvert = 5                                                                           <L 1020>
    var_221 = &(var_pt.nvert);
    wp::store(var_221, var_220);
    // pt.nface = 6                                                                           <L 1021>
    var_223 = &(var_pt.nface);
    wp::store(var_223, var_222);
    // pt.status = 0                                                                          <L 1022>
    var_225 = &(var_pt.status);
    wp::store(var_225, var_224);
    // return pt, GJKResult()                                                                 <L 1023>
    var_226 = GJKResult_0220ee01();
    ret_0 = var_pt;
    ret_1 = var_226;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:688
static CUDA_CALLABLE bool _same_side_0(
    wp::vec_t<3, wp::float32> var_p0,
    wp::vec_t<3, wp::float32> var_p1,
    wp::vec_t<3, wp::float32> var_p2,
    wp::vec_t<3, wp::float32> var_p3)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::float32 var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::float32 var_6;
    const wp::float32 var_7 = 0.0;
    bool var_8;
    const wp::float32 var_9 = 0.0;
    bool var_10;
    bool var_11;
    const wp::float32 var_12 = 0.0;
    bool var_13;
    const wp::float32 var_14 = 0.0;
    bool var_15;
    bool var_16;
    bool var_17;
    //---------
    // forward
    // def _same_side(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, p3: wp.vec3) -> bool:            <L 689>
    // n = wp.cross(p1 - p0, p2 - p0)                                                         <L 690>
    var_0 = wp::sub(var_p1, var_p0);
    var_1 = wp::sub(var_p2, var_p0);
    var_2 = wp::cross(var_0, var_1);
    // dot1 = wp.dot(n, p3 - p0)                                                              <L 691>
    var_3 = wp::sub(var_p3, var_p0);
    var_4 = wp::dot(var_2, var_3);
    // dot2 = wp.dot(n, -p0)                                                                  <L 692>
    var_5 = wp::neg(var_p0);
    var_6 = wp::dot(var_2, var_5);
    // return (dot1 > 0.0 and dot2 > 0.0) or (dot1 < 0.0 and dot2 < 0.0)                      <L 693>
    var_8 = (var_4 > var_7);
    var_10 = (var_6 > var_9);
    var_11 = var_8 && var_10;
    var_13 = (var_4 < var_12);
    var_15 = (var_6 < var_14);
    var_16 = var_13 && var_15;
    var_17 = var_11 || var_16;
    return var_17;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:696
static CUDA_CALLABLE bool _test_tetra_0(
    wp::vec_t<3, wp::float32> var_p0,
    wp::vec_t<3, wp::float32> var_p1,
    wp::vec_t<3, wp::float32> var_p2,
    wp::vec_t<3, wp::float32> var_p3)
{
    //---------
    // primal vars
    bool var_0;
    bool var_1;
    bool var_2;
    bool var_3;
    bool var_4;
    //---------
    // forward
    // def _test_tetra(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, p3: wp.vec3) -> bool:           <L 697>
    // return _same_side(p0, p1, p2, p3) and _same_side(p1, p2, p3, p0) and _same_side(p2, p3, p0, p1) and _same_side(p3, p0, p1, p2)       <L 698>
    var_0 = _same_side_0(var_p0, var_p1, var_p2, var_p3);
    var_1 = _same_side_0(var_p1, var_p2, var_p3, var_p0);
    var_2 = _same_side_0(var_p2, var_p3, var_p0, var_p1);
    var_3 = _same_side_0(var_p3, var_p0, var_p1, var_p2);
    var_4 = var_0 && var_1 && var_2 && var_3;
    return var_4;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:1114
static CUDA_CALLABLE void _polytope4_0(
    Polytope_9ab93ade var_pt,
    wp::mat_t<4, 3, wp::float32> var_simplex,
    wp::mat_t<4, 3, wp::float32> var_simplex1,
    wp::mat_t<4, 3, wp::float32> var_simplex2,
    wp::vec_t<4, wp::int32> var_simplex_index1,
    wp::vec_t<4, wp::int32> var_simplex_index2,
    Polytope_9ab93ade & ret_0,
    GJKResult_0220ee01 & ret_1)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::vec_t<3, wp::float32> var_1;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_2;
    const wp::int32 var_3 = 0;
    wp::array_t<wp::vec_t<3, wp::float32>> var_4;
    const wp::int32 var_5 = 0;
    wp::vec_t<3, wp::float32> var_6;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_7;
    const wp::int32 var_8 = 1;
    wp::array_t<wp::vec_t<3, wp::float32>> var_9;
    const wp::int32 var_10 = 1;
    wp::vec_t<3, wp::float32> var_11;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_12;
    const wp::int32 var_13 = 2;
    wp::array_t<wp::vec_t<3, wp::float32>> var_14;
    const wp::int32 var_15 = 1;
    wp::vec_t<3, wp::float32> var_16;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_17;
    const wp::int32 var_18 = 3;
    wp::array_t<wp::vec_t<3, wp::float32>> var_19;
    const wp::int32 var_20 = 2;
    wp::vec_t<3, wp::float32> var_21;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_22;
    const wp::int32 var_23 = 4;
    wp::array_t<wp::vec_t<3, wp::float32>> var_24;
    const wp::int32 var_25 = 2;
    wp::vec_t<3, wp::float32> var_26;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_27;
    const wp::int32 var_28 = 5;
    wp::array_t<wp::vec_t<3, wp::float32>> var_29;
    const wp::int32 var_30 = 3;
    wp::vec_t<3, wp::float32> var_31;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_32;
    const wp::int32 var_33 = 6;
    wp::array_t<wp::vec_t<3, wp::float32>> var_34;
    const wp::int32 var_35 = 3;
    wp::vec_t<3, wp::float32> var_36;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_37;
    const wp::int32 var_38 = 7;
    wp::array_t<wp::vec_t<3, wp::float32>> var_39;
    const wp::int32 var_40 = 0;
    wp::int32 var_41;
    wp::array_t<wp::int32>* var_42;
    const wp::int32 var_43 = 0;
    wp::array_t<wp::int32> var_44;
    const wp::int32 var_45 = 0;
    wp::int32 var_46;
    wp::array_t<wp::int32>* var_47;
    const wp::int32 var_48 = 1;
    wp::array_t<wp::int32> var_49;
    const wp::int32 var_50 = 1;
    wp::int32 var_51;
    wp::array_t<wp::int32>* var_52;
    const wp::int32 var_53 = 2;
    wp::array_t<wp::int32> var_54;
    const wp::int32 var_55 = 1;
    wp::int32 var_56;
    wp::array_t<wp::int32>* var_57;
    const wp::int32 var_58 = 3;
    wp::array_t<wp::int32> var_59;
    const wp::int32 var_60 = 2;
    wp::int32 var_61;
    wp::array_t<wp::int32>* var_62;
    const wp::int32 var_63 = 4;
    wp::array_t<wp::int32> var_64;
    const wp::int32 var_65 = 2;
    wp::int32 var_66;
    wp::array_t<wp::int32>* var_67;
    const wp::int32 var_68 = 5;
    wp::array_t<wp::int32> var_69;
    const wp::int32 var_70 = 3;
    wp::int32 var_71;
    wp::array_t<wp::int32>* var_72;
    const wp::int32 var_73 = 6;
    wp::array_t<wp::int32> var_74;
    const wp::int32 var_75 = 3;
    wp::int32 var_76;
    wp::array_t<wp::int32>* var_77;
    const wp::int32 var_78 = 7;
    wp::array_t<wp::int32> var_79;
    const wp::int32 var_80 = 0;
    const wp::int32 var_81 = 0;
    const wp::int32 var_82 = 1;
    const wp::int32 var_83 = 2;
    wp::float32 var_84;
    const wp::float32 var_85 = 1e-10;
    bool var_86;
    const wp::int32 var_87 = 1;
    const wp::int32 var_88 = -1;
    wp::int32* var_89;
    const wp::int32 var_90 = 0;
    const wp::int32 var_91 = 1;
    const wp::int32 var_92 = 2;
    GJKResult_0220ee01 var_93;
    const wp::int32 var_94 = 1;
    const wp::int32 var_95 = 0;
    const wp::int32 var_96 = 3;
    const wp::int32 var_97 = 1;
    wp::float32 var_98;
    bool var_99;
    const wp::int32 var_100 = 1;
    const wp::int32 var_101 = -1;
    wp::int32* var_102;
    const wp::int32 var_103 = 0;
    const wp::int32 var_104 = 3;
    const wp::int32 var_105 = 1;
    GJKResult_0220ee01 var_106;
    const wp::int32 var_107 = 2;
    const wp::int32 var_108 = 0;
    const wp::int32 var_109 = 2;
    const wp::int32 var_110 = 3;
    wp::float32 var_111;
    bool var_112;
    const wp::int32 var_113 = 1;
    const wp::int32 var_114 = -1;
    wp::int32* var_115;
    const wp::int32 var_116 = 0;
    const wp::int32 var_117 = 2;
    const wp::int32 var_118 = 3;
    GJKResult_0220ee01 var_119;
    const wp::int32 var_120 = 3;
    const wp::int32 var_121 = 3;
    const wp::int32 var_122 = 2;
    const wp::int32 var_123 = 1;
    wp::float32 var_124;
    bool var_125;
    const wp::int32 var_126 = 1;
    const wp::int32 var_127 = -1;
    wp::int32* var_128;
    const wp::int32 var_129 = 3;
    const wp::int32 var_130 = 2;
    const wp::int32 var_131 = 1;
    GJKResult_0220ee01 var_132;
    const wp::int32 var_133 = 0;
    wp::vec_t<3, wp::float32> var_134;
    const wp::int32 var_135 = 1;
    wp::vec_t<3, wp::float32> var_136;
    const wp::int32 var_137 = 2;
    wp::vec_t<3, wp::float32> var_138;
    const wp::int32 var_139 = 3;
    wp::vec_t<3, wp::float32> var_140;
    bool var_141;
    bool var_142;
    const wp::int32 var_143 = 12;
    wp::int32* var_144;
    GJKResult_0220ee01 var_145;
    const wp::int32 var_146 = 4;
    wp::int32* var_147;
    const wp::int32 var_148 = 4;
    wp::int32* var_149;
    const wp::int32 var_150 = 0;
    wp::int32* var_151;
    GJKResult_0220ee01 var_152;
    //---------
    // forward
    // def _polytope4(                                                                        <L 1115>
    // pt.vert[0] = simplex1[0]                                                               <L 1125>
    var_1 = wp::extract(var_simplex1, var_0);
    var_2 = &(var_pt.vert);
    var_4 = wp::load(var_2);
    wp::array_store(var_4, var_3, var_1);
    // pt.vert[1] = simplex2[0]                                                               <L 1126>
    var_6 = wp::extract(var_simplex2, var_5);
    var_7 = &(var_pt.vert);
    var_9 = wp::load(var_7);
    wp::array_store(var_9, var_8, var_6);
    // pt.vert[2] = simplex1[1]                                                               <L 1127>
    var_11 = wp::extract(var_simplex1, var_10);
    var_12 = &(var_pt.vert);
    var_14 = wp::load(var_12);
    wp::array_store(var_14, var_13, var_11);
    // pt.vert[3] = simplex2[1]                                                               <L 1128>
    var_16 = wp::extract(var_simplex2, var_15);
    var_17 = &(var_pt.vert);
    var_19 = wp::load(var_17);
    wp::array_store(var_19, var_18, var_16);
    // pt.vert[4] = simplex1[2]                                                               <L 1129>
    var_21 = wp::extract(var_simplex1, var_20);
    var_22 = &(var_pt.vert);
    var_24 = wp::load(var_22);
    wp::array_store(var_24, var_23, var_21);
    // pt.vert[5] = simplex2[2]                                                               <L 1130>
    var_26 = wp::extract(var_simplex2, var_25);
    var_27 = &(var_pt.vert);
    var_29 = wp::load(var_27);
    wp::array_store(var_29, var_28, var_26);
    // pt.vert[6] = simplex1[3]                                                               <L 1131>
    var_31 = wp::extract(var_simplex1, var_30);
    var_32 = &(var_pt.vert);
    var_34 = wp::load(var_32);
    wp::array_store(var_34, var_33, var_31);
    // pt.vert[7] = simplex2[3]                                                               <L 1132>
    var_36 = wp::extract(var_simplex2, var_35);
    var_37 = &(var_pt.vert);
    var_39 = wp::load(var_37);
    wp::array_store(var_39, var_38, var_36);
    // pt.vert_index[0] = simplex_index1[0]                                                   <L 1134>
    var_41 = wp::extract(var_simplex_index1, var_40);
    var_42 = &(var_pt.vert_index);
    var_44 = wp::load(var_42);
    wp::array_store(var_44, var_43, var_41);
    // pt.vert_index[1] = simplex_index2[0]                                                   <L 1135>
    var_46 = wp::extract(var_simplex_index2, var_45);
    var_47 = &(var_pt.vert_index);
    var_49 = wp::load(var_47);
    wp::array_store(var_49, var_48, var_46);
    // pt.vert_index[2] = simplex_index1[1]                                                   <L 1136>
    var_51 = wp::extract(var_simplex_index1, var_50);
    var_52 = &(var_pt.vert_index);
    var_54 = wp::load(var_52);
    wp::array_store(var_54, var_53, var_51);
    // pt.vert_index[3] = simplex_index2[1]                                                   <L 1137>
    var_56 = wp::extract(var_simplex_index2, var_55);
    var_57 = &(var_pt.vert_index);
    var_59 = wp::load(var_57);
    wp::array_store(var_59, var_58, var_56);
    // pt.vert_index[4] = simplex_index1[2]                                                   <L 1138>
    var_61 = wp::extract(var_simplex_index1, var_60);
    var_62 = &(var_pt.vert_index);
    var_64 = wp::load(var_62);
    wp::array_store(var_64, var_63, var_61);
    // pt.vert_index[5] = simplex_index2[2]                                                   <L 1139>
    var_66 = wp::extract(var_simplex_index2, var_65);
    var_67 = &(var_pt.vert_index);
    var_69 = wp::load(var_67);
    wp::array_store(var_69, var_68, var_66);
    // pt.vert_index[6] = simplex_index1[3]                                                   <L 1140>
    var_71 = wp::extract(var_simplex_index1, var_70);
    var_72 = &(var_pt.vert_index);
    var_74 = wp::load(var_72);
    wp::array_store(var_74, var_73, var_71);
    // pt.vert_index[7] = simplex_index2[3]                                                   <L 1141>
    var_76 = wp::extract(var_simplex_index2, var_75);
    var_77 = &(var_pt.vert_index);
    var_79 = wp::load(var_77);
    wp::array_store(var_79, var_78, var_76);
    // if _attach_face(pt, 0, 0, 1, 2) < MIN_DIST:                                            <L 1144>
    var_84 = _attach_face_0(var_pt, var_80, var_81, var_82, var_83);
    var_86 = (var_84 < var_85);
    if (var_86) {
        // pt.status = -1                                                                     <L 1145>
        var_89 = &(var_pt.status);
        wp::store(var_89, var_88);
        // return pt, _replace_simplex3(pt, 0, 1, 2)                                          <L 1146>
        var_93 = _replace_simplex3_0(var_pt, var_90, var_91, var_92);
        ret_0 = var_pt;
        ret_1 = var_93;
        return;
    }
    // if _attach_face(pt, 1, 0, 3, 1) < MIN_DIST:                                            <L 1148>
    var_98 = _attach_face_0(var_pt, var_94, var_95, var_96, var_97);
    var_99 = (var_98 < var_85);
    if (var_99) {
        // pt.status = -1                                                                     <L 1149>
        var_102 = &(var_pt.status);
        wp::store(var_102, var_101);
        // return pt, _replace_simplex3(pt, 0, 3, 1)                                          <L 1150>
        var_106 = _replace_simplex3_0(var_pt, var_103, var_104, var_105);
        ret_0 = var_pt;
        ret_1 = var_106;
        return;
    }
    // if _attach_face(pt, 2, 0, 2, 3) < MIN_DIST:                                            <L 1152>
    var_111 = _attach_face_0(var_pt, var_107, var_108, var_109, var_110);
    var_112 = (var_111 < var_85);
    if (var_112) {
        // pt.status = -1                                                                     <L 1153>
        var_115 = &(var_pt.status);
        wp::store(var_115, var_114);
        // return pt, _replace_simplex3(pt, 0, 2, 3)                                          <L 1154>
        var_119 = _replace_simplex3_0(var_pt, var_116, var_117, var_118);
        ret_0 = var_pt;
        ret_1 = var_119;
        return;
    }
    // if _attach_face(pt, 3, 3, 2, 1) < MIN_DIST:                                            <L 1156>
    var_124 = _attach_face_0(var_pt, var_120, var_121, var_122, var_123);
    var_125 = (var_124 < var_85);
    if (var_125) {
        // pt.status = -1                                                                     <L 1157>
        var_128 = &(var_pt.status);
        wp::store(var_128, var_127);
        // return pt, _replace_simplex3(pt, 3, 2, 1)                                          <L 1158>
        var_132 = _replace_simplex3_0(var_pt, var_129, var_130, var_131);
        ret_0 = var_pt;
        ret_1 = var_132;
        return;
    }
    // if not _test_tetra(simplex[0], simplex[1], simplex[2], simplex[3]):                    <L 1160>
    var_134 = wp::extract(var_simplex, var_133);
    var_136 = wp::extract(var_simplex, var_135);
    var_138 = wp::extract(var_simplex, var_137);
    var_140 = wp::extract(var_simplex, var_139);
    var_141 = _test_tetra_0(var_134, var_136, var_138, var_140);
    var_142 = wp::unot(var_141);
    if (var_142) {
        // pt.status = 12                                                                     <L 1161>
        var_144 = &(var_pt.status);
        wp::store(var_144, var_143);
        // return pt, GJKResult()                                                             <L 1162>
        var_145 = GJKResult_0220ee01();
        ret_0 = var_pt;
        ret_1 = var_145;
        return;
    }
    // pt.nvert = 4                                                                           <L 1165>
    var_147 = &(var_pt.nvert);
    wp::store(var_147, var_146);
    // pt.nface = 4                                                                           <L 1166>
    var_149 = &(var_pt.nface);
    wp::store(var_149, var_148);
    // pt.status = 0                                                                          <L 1167>
    var_151 = &(var_pt.status);
    wp::store(var_151, var_150);
    // return pt, GJKResult()                                                                 <L 1168>
    var_152 = GJKResult_0220ee01();
    ret_0 = var_pt;
    ret_1 = var_152;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:744
static CUDA_CALLABLE bool _tri_point_intersect_0(
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> var_v3,
    wp::vec_t<3, wp::float32> var_p)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    const wp::int32 var_1 = 0;
    wp::float32 var_2;
    const wp::int32 var_3 = 1;
    wp::float32 var_4;
    const wp::int32 var_5 = 2;
    wp::float32 var_6;
    const wp::float32 var_7 = 0.0;
    bool var_8;
    const wp::float32 var_9 = 0.0;
    bool var_10;
    const wp::float32 var_11 = 0.0;
    bool var_12;
    bool var_13;
    const bool var_14 = false;
    wp::vec_t<3, wp::float32> var_15;
    const wp::int32 var_16 = 0;
    wp::float32 var_17;
    wp::float32 var_18;
    const wp::int32 var_19 = 0;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::int32 var_23 = 0;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    const wp::int32 var_27 = 0;
    const wp::int32 var_28 = 1;
    wp::float32 var_29;
    wp::float32 var_30;
    const wp::int32 var_31 = 1;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    const wp::int32 var_35 = 1;
    wp::float32 var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    const wp::int32 var_39 = 1;
    const wp::int32 var_40 = 2;
    wp::float32 var_41;
    wp::float32 var_42;
    const wp::int32 var_43 = 2;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    const wp::int32 var_47 = 2;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    const wp::int32 var_51 = 2;
    wp::vec_t<3, wp::float32> var_52;
    wp::float32 var_53;
    const wp::float32 var_54 = 1e-15;
    bool var_55;
    //---------
    // forward
    // def _tri_point_intersect(v1: wp.vec3, v2: wp.vec3, v3: wp.vec3, p: wp.vec3) -> bool:       <L 745>
    // coordinates = _tri_affine_coord(v1, v2, v3, p)                                         <L 746>
    var_0 = _tri_affine_coord_0(var_v1, var_v2, var_v3, var_p);
    // l1 = coordinates[0]                                                                    <L 747>
    var_2 = wp::extract(var_0, var_1);
    // l2 = coordinates[1]                                                                    <L 748>
    var_4 = wp::extract(var_0, var_3);
    // l3 = coordinates[2]                                                                    <L 749>
    var_6 = wp::extract(var_0, var_5);
    // if l1 < 0.0 or l2 < 0.0 or l3 < 0.0:                                                   <L 751>
    var_8 = (var_2 < var_7);
    var_10 = (var_4 < var_9);
    var_12 = (var_6 < var_11);
    var_13 = var_8 || var_10 || var_12;
    if (var_13) {
        // return False                                                                       <L 752>
        return var_14;
    }
    // pr = wp.vec3()                                                                         <L 754>
    var_15 = wp::vec_t<3, wp::float32>();
    // pr[0] = v1[0] * l1 + v2[0] * l2 + v3[0] * l3                                           <L 755>
    var_17 = wp::extract(var_v1, var_16);
    var_18 = wp::mul(var_17, var_2);
    var_20 = wp::extract(var_v2, var_19);
    var_21 = wp::mul(var_20, var_4);
    var_22 = wp::add(var_18, var_21);
    var_24 = wp::extract(var_v3, var_23);
    var_25 = wp::mul(var_24, var_6);
    var_26 = wp::add(var_22, var_25);
    wp::assign_inplace(var_15, var_27, var_26);
    // pr[1] = v1[1] * l1 + v2[1] * l2 + v3[1] * l3                                           <L 756>
    var_29 = wp::extract(var_v1, var_28);
    var_30 = wp::mul(var_29, var_2);
    var_32 = wp::extract(var_v2, var_31);
    var_33 = wp::mul(var_32, var_4);
    var_34 = wp::add(var_30, var_33);
    var_36 = wp::extract(var_v3, var_35);
    var_37 = wp::mul(var_36, var_6);
    var_38 = wp::add(var_34, var_37);
    wp::assign_inplace(var_15, var_39, var_38);
    // pr[2] = v1[2] * l1 + v2[2] * l2 + v3[2] * l3                                           <L 757>
    var_41 = wp::extract(var_v1, var_40);
    var_42 = wp::mul(var_41, var_2);
    var_44 = wp::extract(var_v2, var_43);
    var_45 = wp::mul(var_44, var_4);
    var_46 = wp::add(var_42, var_45);
    var_48 = wp::extract(var_v3, var_47);
    var_49 = wp::mul(var_48, var_6);
    var_50 = wp::add(var_46, var_49);
    wp::assign_inplace(var_15, var_51, var_50);
    // return wp.norm_l2(pr - p) < MINVAL                                                     <L 758>
    var_52 = wp::sub(var_15, var_p);
    var_53 = norm_l2_0(var_52);
    var_55 = (var_53 < var_54);
    return var_55;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:1026
static CUDA_CALLABLE Polytope_9ab93ade _polytope3_0(
    Polytope_9ab93ade var_pt,
    wp::float32 var_dist,
    wp::mat_t<4, 3, wp::float32> var_simplex,
    wp::mat_t<4, 3, wp::float32> var_simplex1,
    wp::mat_t<4, 3, wp::float32> var_simplex2,
    wp::vec_t<4, wp::int32> var_simplex_index1,
    wp::vec_t<4, wp::int32> var_simplex_index2,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::vec_t<3, wp::float32> var_1;
    const wp::int32 var_2 = 0;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    const wp::int32 var_5 = 2;
    wp::vec_t<3, wp::float32> var_6;
    const wp::int32 var_7 = 0;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::float32 var_11;
    const wp::float32 var_12 = 1e-15;
    bool var_13;
    const wp::int32 var_14 = 2;
    wp::int32* var_15;
    const wp::int32 var_16 = 0;
    wp::vec_t<3, wp::float32> var_17;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_18;
    const wp::int32 var_19 = 0;
    wp::array_t<wp::vec_t<3, wp::float32>> var_20;
    const wp::int32 var_21 = 0;
    wp::vec_t<3, wp::float32> var_22;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_23;
    const wp::int32 var_24 = 1;
    wp::array_t<wp::vec_t<3, wp::float32>> var_25;
    const wp::int32 var_26 = 1;
    wp::vec_t<3, wp::float32> var_27;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_28;
    const wp::int32 var_29 = 2;
    wp::array_t<wp::vec_t<3, wp::float32>> var_30;
    const wp::int32 var_31 = 1;
    wp::vec_t<3, wp::float32> var_32;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_33;
    const wp::int32 var_34 = 3;
    wp::array_t<wp::vec_t<3, wp::float32>> var_35;
    const wp::int32 var_36 = 2;
    wp::vec_t<3, wp::float32> var_37;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_38;
    const wp::int32 var_39 = 4;
    wp::array_t<wp::vec_t<3, wp::float32>> var_40;
    const wp::int32 var_41 = 2;
    wp::vec_t<3, wp::float32> var_42;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_43;
    const wp::int32 var_44 = 5;
    wp::array_t<wp::vec_t<3, wp::float32>> var_45;
    const wp::int32 var_46 = 0;
    wp::int32 var_47;
    wp::array_t<wp::int32>* var_48;
    const wp::int32 var_49 = 0;
    wp::array_t<wp::int32> var_50;
    const wp::int32 var_51 = 0;
    wp::int32 var_52;
    wp::array_t<wp::int32>* var_53;
    const wp::int32 var_54 = 1;
    wp::array_t<wp::int32> var_55;
    const wp::int32 var_56 = 1;
    wp::int32 var_57;
    wp::array_t<wp::int32>* var_58;
    const wp::int32 var_59 = 2;
    wp::array_t<wp::int32> var_60;
    const wp::int32 var_61 = 1;
    wp::int32 var_62;
    wp::array_t<wp::int32>* var_63;
    const wp::int32 var_64 = 3;
    wp::array_t<wp::int32> var_65;
    const wp::int32 var_66 = 2;
    wp::int32 var_67;
    wp::array_t<wp::int32>* var_68;
    const wp::int32 var_69 = 4;
    wp::array_t<wp::int32> var_70;
    const wp::int32 var_71 = 2;
    wp::int32 var_72;
    wp::array_t<wp::int32>* var_73;
    const wp::int32 var_74 = 5;
    wp::array_t<wp::int32> var_75;
    const wp::int32 var_76 = 3;
    wp::vec_t<3, wp::float32> var_77;
    wp::int32 var_78;
    wp::int32 var_79;
    const wp::int32 var_80 = 4;
    wp::int32 var_81;
    wp::int32 var_82;
    const wp::int32 var_83 = 0;
    wp::vec_t<3, wp::float32> var_84;
    const wp::int32 var_85 = 1;
    wp::vec_t<3, wp::float32> var_86;
    const wp::int32 var_87 = 2;
    wp::vec_t<3, wp::float32> var_88;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_89;
    const wp::int32 var_90 = 6;
    wp::vec_t<3, wp::float32>* var_91;
    wp::array_t<wp::vec_t<3, wp::float32>> var_92;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_93;
    const wp::int32 var_94 = 7;
    wp::vec_t<3, wp::float32>* var_95;
    wp::array_t<wp::vec_t<3, wp::float32>> var_96;
    wp::vec_t<3, wp::float32> var_97;
    wp::vec_t<3, wp::float32> var_98;
    wp::vec_t<3, wp::float32> var_99;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_100;
    const wp::int32 var_101 = 8;
    wp::vec_t<3, wp::float32>* var_102;
    wp::array_t<wp::vec_t<3, wp::float32>> var_103;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_104;
    const wp::int32 var_105 = 9;
    wp::vec_t<3, wp::float32>* var_106;
    wp::array_t<wp::vec_t<3, wp::float32>> var_107;
    wp::vec_t<3, wp::float32> var_108;
    wp::vec_t<3, wp::float32> var_109;
    wp::vec_t<3, wp::float32> var_110;
    bool var_111;
    const wp::int32 var_112 = 3;
    wp::int32* var_113;
    bool var_114;
    const wp::int32 var_115 = 4;
    wp::int32* var_116;
    const wp::float32 var_117 = 1e-05;
    bool var_118;
    bool var_119;
    bool var_120;
    bool var_121;
    bool var_122;
    bool var_123;
    const wp::int32 var_124 = 5;
    wp::int32* var_125;
    const wp::int32 var_126 = 0;
    const wp::int32 var_127 = 4;
    const wp::int32 var_128 = 0;
    const wp::int32 var_129 = 1;
    wp::float32 var_130;
    const wp::float32 var_131 = 1e-10;
    bool var_132;
    const wp::int32 var_133 = 6;
    wp::int32* var_134;
    const wp::int32 var_135 = 1;
    const wp::int32 var_136 = 4;
    const wp::int32 var_137 = 2;
    const wp::int32 var_138 = 0;
    wp::float32 var_139;
    bool var_140;
    const wp::int32 var_141 = 7;
    wp::int32* var_142;
    const wp::int32 var_143 = 2;
    const wp::int32 var_144 = 4;
    const wp::int32 var_145 = 1;
    const wp::int32 var_146 = 2;
    wp::float32 var_147;
    bool var_148;
    const wp::int32 var_149 = 8;
    wp::int32* var_150;
    const wp::int32 var_151 = 3;
    const wp::int32 var_152 = 3;
    const wp::int32 var_153 = 1;
    const wp::int32 var_154 = 0;
    wp::float32 var_155;
    bool var_156;
    const wp::int32 var_157 = 9;
    wp::int32* var_158;
    const wp::int32 var_159 = 4;
    const wp::int32 var_160 = 3;
    const wp::int32 var_161 = 0;
    const wp::int32 var_162 = 2;
    wp::float32 var_163;
    bool var_164;
    const wp::int32 var_165 = 10;
    wp::int32* var_166;
    const wp::int32 var_167 = 5;
    const wp::int32 var_168 = 3;
    const wp::int32 var_169 = 2;
    const wp::int32 var_170 = 1;
    wp::float32 var_171;
    bool var_172;
    const wp::int32 var_173 = 11;
    wp::int32* var_174;
    const wp::int32 var_175 = 5;
    wp::int32* var_176;
    const wp::int32 var_177 = 6;
    wp::int32* var_178;
    const wp::int32 var_179 = 0;
    wp::int32* var_180;
    //---------
    // forward
    // def _polytope3(                                                                        <L 1027>
    // n = wp.cross(simplex[1] - simplex[0], simplex[2] - simplex[0])                         <L 1043>
    var_1 = wp::extract(var_simplex, var_0);
    var_3 = wp::extract(var_simplex, var_2);
    var_4 = wp::sub(var_1, var_3);
    var_6 = wp::extract(var_simplex, var_5);
    var_8 = wp::extract(var_simplex, var_7);
    var_9 = wp::sub(var_6, var_8);
    var_10 = wp::cross(var_4, var_9);
    // if wp.norm_l2(n) < MINVAL:                                                             <L 1044>
    var_11 = norm_l2_0(var_10);
    var_13 = (var_11 < var_12);
    if (var_13) {
        // pt.status = 2                                                                      <L 1045>
        var_15 = &(var_pt.status);
        wp::store(var_15, var_14);
        // return pt                                                                          <L 1046>
        return var_pt;
    }
    // pt.vert[0] = simplex1[0]                                                               <L 1048>
    var_17 = wp::extract(var_simplex1, var_16);
    var_18 = &(var_pt.vert);
    var_20 = wp::load(var_18);
    wp::array_store(var_20, var_19, var_17);
    // pt.vert[1] = simplex2[0]                                                               <L 1049>
    var_22 = wp::extract(var_simplex2, var_21);
    var_23 = &(var_pt.vert);
    var_25 = wp::load(var_23);
    wp::array_store(var_25, var_24, var_22);
    // pt.vert[2] = simplex1[1]                                                               <L 1050>
    var_27 = wp::extract(var_simplex1, var_26);
    var_28 = &(var_pt.vert);
    var_30 = wp::load(var_28);
    wp::array_store(var_30, var_29, var_27);
    // pt.vert[3] = simplex2[1]                                                               <L 1051>
    var_32 = wp::extract(var_simplex2, var_31);
    var_33 = &(var_pt.vert);
    var_35 = wp::load(var_33);
    wp::array_store(var_35, var_34, var_32);
    // pt.vert[4] = simplex1[2]                                                               <L 1052>
    var_37 = wp::extract(var_simplex1, var_36);
    var_38 = &(var_pt.vert);
    var_40 = wp::load(var_38);
    wp::array_store(var_40, var_39, var_37);
    // pt.vert[5] = simplex2[2]                                                               <L 1053>
    var_42 = wp::extract(var_simplex2, var_41);
    var_43 = &(var_pt.vert);
    var_45 = wp::load(var_43);
    wp::array_store(var_45, var_44, var_42);
    // pt.vert_index[0] = simplex_index1[0]                                                   <L 1055>
    var_47 = wp::extract(var_simplex_index1, var_46);
    var_48 = &(var_pt.vert_index);
    var_50 = wp::load(var_48);
    wp::array_store(var_50, var_49, var_47);
    // pt.vert_index[1] = simplex_index2[0]                                                   <L 1056>
    var_52 = wp::extract(var_simplex_index2, var_51);
    var_53 = &(var_pt.vert_index);
    var_55 = wp::load(var_53);
    wp::array_store(var_55, var_54, var_52);
    // pt.vert_index[2] = simplex_index1[1]                                                   <L 1057>
    var_57 = wp::extract(var_simplex_index1, var_56);
    var_58 = &(var_pt.vert_index);
    var_60 = wp::load(var_58);
    wp::array_store(var_60, var_59, var_57);
    // pt.vert_index[3] = simplex_index2[1]                                                   <L 1058>
    var_62 = wp::extract(var_simplex_index2, var_61);
    var_63 = &(var_pt.vert_index);
    var_65 = wp::load(var_63);
    wp::array_store(var_65, var_64, var_62);
    // pt.vert_index[4] = simplex_index1[2]                                                   <L 1059>
    var_67 = wp::extract(var_simplex_index1, var_66);
    var_68 = &(var_pt.vert_index);
    var_70 = wp::load(var_68);
    wp::array_store(var_70, var_69, var_67);
    // pt.vert_index[5] = simplex_index2[2]                                                   <L 1060>
    var_72 = wp::extract(var_simplex_index2, var_71);
    var_73 = &(var_pt.vert_index);
    var_75 = wp::load(var_73);
    wp::array_store(var_75, var_74, var_72);
    // _epa_support(pt, 3, geom1, geom2, geomtype1, geomtype2, -n)                            <L 1062>
    var_77 = wp::neg(var_10);
    _epa_support_0(var_pt, var_76, var_geom1, var_geom2, var_geomtype1, var_geomtype2, var_77, var_78, var_79);
    // _epa_support(pt, 4, geom1, geom2, geomtype1, geomtype2, n)                             <L 1063>
    _epa_support_0(var_pt, var_80, var_geom1, var_geom2, var_geomtype1, var_geomtype2, var_10, var_81, var_82);
    // v1 = simplex[0]                                                                        <L 1065>
    var_84 = wp::extract(var_simplex, var_83);
    // v2 = simplex[1]                                                                        <L 1066>
    var_86 = wp::extract(var_simplex, var_85);
    // v3 = simplex[2]                                                                        <L 1067>
    var_88 = wp::extract(var_simplex, var_87);
    // v4 = pt.vert[6] - pt.vert[7]                                                           <L 1068>
    var_89 = &(var_pt.vert);
    var_92 = wp::load(var_89);
    var_91 = wp::address(var_92, var_90);
    var_93 = &(var_pt.vert);
    var_96 = wp::load(var_93);
    var_95 = wp::address(var_96, var_94);
    var_98 = wp::load(var_91);
    var_99 = wp::load(var_95);
    var_97 = wp::sub(var_98, var_99);
    // v5 = pt.vert[8] - pt.vert[9]                                                           <L 1069>
    var_100 = &(var_pt.vert);
    var_103 = wp::load(var_100);
    var_102 = wp::address(var_103, var_101);
    var_104 = &(var_pt.vert);
    var_107 = wp::load(var_104);
    var_106 = wp::address(var_107, var_105);
    var_109 = wp::load(var_102);
    var_110 = wp::load(var_106);
    var_108 = wp::sub(var_109, var_110);
    // if _tri_point_intersect(v1, v2, v3, v4):                                               <L 1072>
    var_111 = _tri_point_intersect_0(var_84, var_86, var_88, var_97);
    if (var_111) {
        // pt.status = 3                                                                      <L 1073>
        var_113 = &(var_pt.status);
        wp::store(var_113, var_112);
        // return pt                                                                          <L 1074>
        return var_pt;
    }
    // if _tri_point_intersect(v1, v2, v3, v5):                                               <L 1077>
    var_114 = _tri_point_intersect_0(var_84, var_86, var_88, var_108);
    if (var_114) {
        // pt.status = 4                                                                      <L 1078>
        var_116 = &(var_pt.status);
        wp::store(var_116, var_115);
        // return pt                                                                          <L 1079>
        return var_pt;
    }
    // if dist > 1e-5 and not _test_tetra(v1, v2, v3, v4) and not _test_tetra(v1, v2, v3, v5):       <L 1083>
    var_118 = (var_dist > var_117);
    var_119 = _test_tetra_0(var_84, var_86, var_88, var_97);
    var_120 = wp::unot(var_119);
    var_121 = _test_tetra_0(var_84, var_86, var_88, var_108);
    var_122 = wp::unot(var_121);
    var_123 = var_118 && var_120 && var_122;
    if (var_123) {
        // pt.status = 5                                                                      <L 1084>
        var_125 = &(var_pt.status);
        wp::store(var_125, var_124);
        // return pt                                                                          <L 1085>
        return var_pt;
    }
    // if _attach_face(pt, 0, 4, 0, 1) < MIN_DIST:                                            <L 1088>
    var_130 = _attach_face_0(var_pt, var_126, var_127, var_128, var_129);
    var_132 = (var_130 < var_131);
    if (var_132) {
        // pt.status = 6                                                                      <L 1089>
        var_134 = &(var_pt.status);
        wp::store(var_134, var_133);
        // return pt                                                                          <L 1090>
        return var_pt;
    }
    // if _attach_face(pt, 1, 4, 2, 0) < MIN_DIST:                                            <L 1091>
    var_139 = _attach_face_0(var_pt, var_135, var_136, var_137, var_138);
    var_140 = (var_139 < var_131);
    if (var_140) {
        // pt.status = 7                                                                      <L 1092>
        var_142 = &(var_pt.status);
        wp::store(var_142, var_141);
        // return pt                                                                          <L 1093>
        return var_pt;
    }
    // if _attach_face(pt, 2, 4, 1, 2) < MIN_DIST:                                            <L 1094>
    var_147 = _attach_face_0(var_pt, var_143, var_144, var_145, var_146);
    var_148 = (var_147 < var_131);
    if (var_148) {
        // pt.status = 8                                                                      <L 1095>
        var_150 = &(var_pt.status);
        wp::store(var_150, var_149);
        // return pt                                                                          <L 1096>
        return var_pt;
    }
    // if _attach_face(pt, 3, 3, 1, 0) < MIN_DIST:                                            <L 1097>
    var_155 = _attach_face_0(var_pt, var_151, var_152, var_153, var_154);
    var_156 = (var_155 < var_131);
    if (var_156) {
        // pt.status = 9                                                                      <L 1098>
        var_158 = &(var_pt.status);
        wp::store(var_158, var_157);
        // return pt                                                                          <L 1099>
        return var_pt;
    }
    // if _attach_face(pt, 4, 3, 0, 2) < MIN_DIST:                                            <L 1100>
    var_163 = _attach_face_0(var_pt, var_159, var_160, var_161, var_162);
    var_164 = (var_163 < var_131);
    if (var_164) {
        // pt.status = 10                                                                     <L 1101>
        var_166 = &(var_pt.status);
        wp::store(var_166, var_165);
        // return pt                                                                          <L 1102>
        return var_pt;
    }
    // if _attach_face(pt, 5, 3, 2, 1) < MIN_DIST:                                            <L 1103>
    var_171 = _attach_face_0(var_pt, var_167, var_168, var_169, var_170);
    var_172 = (var_171 < var_131);
    if (var_172) {
        // pt.status = 11                                                                     <L 1104>
        var_174 = &(var_pt.status);
        wp::store(var_174, var_173);
        // return pt                                                                          <L 1105>
        return var_pt;
    }
    // pt.nvert = 5                                                                           <L 1108>
    var_176 = &(var_pt.nvert);
    wp::store(var_176, var_175);
    // pt.nface = 6                                                                           <L 1109>
    var_178 = &(var_pt.nface);
    wp::store(var_178, var_177);
    // pt.status = 0                                                                          <L 1110>
    var_180 = &(var_pt.status);
    wp::store(var_180, var_179);
    // return pt                                                                              <L 1111>
    return var_pt;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:1195
static CUDA_CALLABLE bool _is_invalid_face_0(
    wp::int32 var_face)
{
    //---------
    // primal vars
    wp::uint32 var_0;
    const wp::uint32 var_1 = 3221225472;
    wp::uint32 var_2;
    bool var_3;
    //---------
    // forward
    // def _is_invalid_face(face: int) -> bool:                                               <L 1196>
    // return bool(wp.uint32(face) & _FACE_INVALID_OR_DELETED_MASK)                           <L 1198>
    var_0 = wp::uint32(var_face);
    var_2 = wp::bit_and(var_0, var_1);
    var_3 = bool(var_2);
    return var_3;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:1177
static CUDA_CALLABLE wp::int32 _delete_face_0(
    wp::int32 var_face)
{
    //---------
    // primal vars
    wp::uint32 var_0;
    const wp::uint32 var_1 = 2147483648;
    wp::uint32 var_2;
    wp::int32 var_3;
    //---------
    // forward
    // def _delete_face(face: int) -> int:                                                    <L 1178>
    // return int(wp.uint32(face) | _FACE_DELETED_BIT)                                        <L 1180>
    var_0 = wp::uint32(var_face);
    var_2 = wp::bit_or(var_0, var_1);
    var_3 = wp::int(var_2);
    return var_3;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:1171
static CUDA_CALLABLE wp::vec_t<3, wp::int32> _get_face_verts_0(
    wp::int32 var_face)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1023;
    wp::int32 var_1;
    const wp::int32 var_2 = 10;
    wp::int32 var_3;
    const wp::int32 var_4 = 1023;
    wp::int32 var_5;
    const wp::int32 var_6 = 20;
    wp::int32 var_7;
    const wp::int32 var_8 = 1023;
    wp::int32 var_9;
    wp::vec_t<3, wp::int32> var_10;
    //---------
    // forward
    // def _get_face_verts(face: int) -> wp.vec3i:                                            <L 1172>
    // return wp.vec3i(face & 0x3FF, face >> 10 & 0x3FF, face >> 20 & 0x3FF)                  <L 1174>
    var_1 = wp::bit_and(var_face, var_0);
    var_3 = wp::rshift(var_face, var_2);
    var_5 = wp::bit_and(var_3, var_4);
    var_7 = wp::rshift(var_face, var_6);
    var_9 = wp::bit_and(var_7, var_8);
    var_10 = wp::vec_t<3, wp::int32>(var_1, var_5, var_9);
    return var_10;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:840
static CUDA_CALLABLE wp::int32 _add_edge_0(
    Polytope_9ab93ade var_pt,
    wp::int32 var_e1,
    wp::int32 var_e2)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    const wp::int32 var_3 = 0;
    bool var_4;
    const wp::int32 var_5 = 1;
    const wp::int32 var_6 = -1;
    wp::int32 var_7;
    const wp::int32 var_8 = 10;
    wp::int32 var_9;
    wp::int32 var_10;
    wp::int32 var_11;
    wp::range_t var_12;
    wp::int32 var_13;
    wp::array_t<wp::int32>* var_14;
    wp::int32* var_15;
    wp::array_t<wp::int32> var_16;
    bool var_17;
    wp::int32 var_18;
    wp::array_t<wp::int32>* var_19;
    const wp::int32 var_20 = 1;
    wp::int32 var_21;
    wp::int32* var_22;
    wp::array_t<wp::int32> var_23;
    wp::array_t<wp::int32>* var_24;
    wp::array_t<wp::int32> var_25;
    wp::int32 var_26;
    const wp::int32 var_27 = 1;
    wp::int32 var_28;
    wp::array_t<wp::int32>* var_29;
    wp::shape_t* var_30;
    const wp::int32 var_31 = 0;
    wp::int32 var_32;
    wp::shape_t var_33;
    bool var_34;
    const wp::int32 var_35 = 1;
    const wp::int32 var_36 = -1;
    wp::array_t<wp::int32>* var_37;
    wp::array_t<wp::int32> var_38;
    const wp::int32 var_39 = 1;
    wp::int32 var_40;
    //---------
    // forward
    // def _add_edge(pt: Polytope, e1: int, e2: int) -> int:                                  <L 841>
    // n = pt.nhorizon                                                                        <L 842>
    var_0 = &(var_pt.nhorizon);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // if n < 0:                                                                              <L 844>
    var_4 = (var_1 < var_3);
    if (var_4) {
        // return -1                                                                          <L 845>
        return var_6;
    }
    // edge = (wp.min(e1, e2) << 10) | wp.max(e1, e2)                                         <L 847>
    var_7 = wp::min(var_e1, var_e2);
    var_9 = wp::lshift(var_7, var_8);
    var_10 = wp::max(var_e1, var_e2);
    var_11 = wp::bit_or(var_9, var_10);
    // for i in range(n):                                                                     <L 849>
    var_12 = wp::range(var_1);
    start_for_1:;
        if (iter_cmp(var_12) == 0) goto end_for_1;
        var_13 = wp::iter_next(var_12);
        // if edge == pt.horizon[i]:                                                          <L 850>
        var_14 = &(var_pt.horizon);
        var_16 = wp::load(var_14);
        var_15 = wp::address(var_16, var_13);
        var_18 = wp::load(var_15);
        var_17 = (var_11 == var_18);
        if (var_17) {
            // pt.horizon[i] = pt.horizon[n - 1]                                              <L 851>
            var_19 = &(var_pt.horizon);
            var_21 = wp::sub(var_1, var_20);
            var_23 = wp::load(var_19);
            var_22 = wp::address(var_23, var_21);
            var_24 = &(var_pt.horizon);
            var_25 = wp::load(var_24);
            var_26 = wp::load(var_22);
            wp::array_store(var_25, var_13, var_26);
            // return n - 1                                                                   <L 852>
            var_28 = wp::sub(var_1, var_27);
            return var_28;
        }
        goto start_for_1;
    end_for_1:;
    // if n == pt.horizon.shape[0]:                                                           <L 855>
    var_29 = &(var_pt.horizon);
    var_30 = &(var_29->shape);
    var_33 = wp::load(var_30);
    var_32 = wp::extract(var_33, var_31);
    var_34 = (var_1 == var_32);
    if (var_34) {
        // return -1                                                                          <L 856>
        return var_36;
    }
    // pt.horizon[n] = edge                                                                   <L 858>
    var_37 = &(var_pt.horizon);
    var_38 = wp::load(var_37);
    wp::array_store(var_38, var_1, var_11);
    // return n + 1                                                                           <L 859>
    var_40 = wp::add(var_1, var_39);
    return var_40;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:1183
static CUDA_CALLABLE bool _is_face_deleted_0(
    wp::int32 var_face)
{
    //---------
    // primal vars
    wp::uint32 var_0;
    const wp::uint32 var_1 = 2147483648;
    wp::uint32 var_2;
    bool var_3;
    //---------
    // forward
    // def _is_face_deleted(face: int) -> bool:                                               <L 1184>
    // return bool(wp.uint32(face) & _FACE_DELETED_BIT)                                       <L 1186>
    var_0 = wp::uint32(var_face);
    var_2 = wp::bit_and(var_0, var_1);
    var_3 = bool(var_2);
    return var_3;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:835
static CUDA_CALLABLE wp::vec_t<2, wp::int32> _get_edge_0(
    wp::int32 var_edge)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1023;
    wp::int32 var_1;
    const wp::int32 var_2 = 10;
    wp::int32 var_3;
    const wp::int32 var_4 = 1023;
    wp::int32 var_5;
    wp::vec_t<2, wp::int32> var_6;
    //---------
    // forward
    // def _get_edge(edge: int) -> wp.vec2i:                                                  <L 836>
    // return wp.vec2i(edge & 0x3FF, (edge >> 10) & 0x3FF)                                    <L 837>
    var_1 = wp::bit_and(var_edge, var_0);
    var_3 = wp::rshift(var_edge, var_2);
    var_5 = wp::bit_and(var_3, var_4);
    var_6 = wp::vec_t<2, wp::int32>(var_1, var_5);
    return var_6;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:1189
static CUDA_CALLABLE wp::int32 _invalidate_face_0(
    wp::int32 var_face)
{
    //---------
    // primal vars
    wp::uint32 var_0;
    const wp::uint32 var_1 = 1073741824;
    wp::uint32 var_2;
    wp::int32 var_3;
    //---------
    // forward
    // def _invalidate_face(face: int) -> int:                                                <L 1190>
    // return int(wp.uint32(face) | _FACE_INVALID_BIT)                                        <L 1192>
    var_0 = wp::uint32(var_face);
    var_2 = wp::bit_or(var_0, var_1);
    var_3 = wp::int(var_2);
    return var_3;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:862
static CUDA_CALLABLE void _epa_witness_0(
    Polytope_9ab93ade var_pt,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2,
    wp::int32 var_face_idx,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::float32 & ret_2)
{
    //---------
    // primal vars
    wp::array_t<wp::int32>* var_0;
    wp::int32* var_1;
    wp::array_t<wp::int32> var_2;
    wp::vec_t<3, wp::int32> var_3;
    wp::int32 var_4;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_5;
    const wp::int32 var_6 = 2;
    const wp::int32 var_7 = 0;
    wp::int32 var_8;
    wp::int32 var_9;
    wp::vec_t<3, wp::float32>* var_10;
    wp::array_t<wp::vec_t<3, wp::float32>> var_11;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_12;
    const wp::int32 var_13 = 2;
    const wp::int32 var_14 = 0;
    wp::int32 var_15;
    wp::int32 var_16;
    const wp::int32 var_17 = 1;
    wp::int32 var_18;
    wp::vec_t<3, wp::float32>* var_19;
    wp::array_t<wp::vec_t<3, wp::float32>> var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_24;
    const wp::int32 var_25 = 2;
    const wp::int32 var_26 = 1;
    wp::int32 var_27;
    wp::int32 var_28;
    wp::vec_t<3, wp::float32>* var_29;
    wp::array_t<wp::vec_t<3, wp::float32>> var_30;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_31;
    const wp::int32 var_32 = 2;
    const wp::int32 var_33 = 1;
    wp::int32 var_34;
    wp::int32 var_35;
    const wp::int32 var_36 = 1;
    wp::int32 var_37;
    wp::vec_t<3, wp::float32>* var_38;
    wp::array_t<wp::vec_t<3, wp::float32>> var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::vec_t<3, wp::float32> var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_43;
    const wp::int32 var_44 = 2;
    const wp::int32 var_45 = 2;
    wp::int32 var_46;
    wp::int32 var_47;
    wp::vec_t<3, wp::float32>* var_48;
    wp::array_t<wp::vec_t<3, wp::float32>> var_49;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_50;
    const wp::int32 var_51 = 2;
    const wp::int32 var_52 = 2;
    wp::int32 var_53;
    wp::int32 var_54;
    const wp::int32 var_55 = 1;
    wp::int32 var_56;
    wp::vec_t<3, wp::float32>* var_57;
    wp::array_t<wp::vec_t<3, wp::float32>> var_58;
    wp::vec_t<3, wp::float32> var_59;
    wp::vec_t<3, wp::float32> var_60;
    wp::vec_t<3, wp::float32> var_61;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_62;
    wp::vec_t<3, wp::float32>* var_63;
    wp::array_t<wp::vec_t<3, wp::float32>> var_64;
    wp::vec_t<3, wp::float32> var_65;
    wp::vec_t<3, wp::float32> var_66;
    const wp::int32 var_67 = 0;
    wp::float32 var_68;
    const wp::int32 var_69 = 1;
    wp::float32 var_70;
    const wp::int32 var_71 = 2;
    wp::float32 var_72;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_73;
    const wp::int32 var_74 = 2;
    const wp::int32 var_75 = 0;
    wp::int32 var_76;
    wp::int32 var_77;
    const wp::int32 var_78 = 1;
    wp::int32 var_79;
    wp::vec_t<3, wp::float32>* var_80;
    wp::array_t<wp::vec_t<3, wp::float32>> var_81;
    wp::vec_t<3, wp::float32> var_82;
    wp::vec_t<3, wp::float32> var_83;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_84;
    const wp::int32 var_85 = 2;
    const wp::int32 var_86 = 1;
    wp::int32 var_87;
    wp::int32 var_88;
    const wp::int32 var_89 = 1;
    wp::int32 var_90;
    wp::vec_t<3, wp::float32>* var_91;
    wp::array_t<wp::vec_t<3, wp::float32>> var_92;
    wp::vec_t<3, wp::float32> var_93;
    wp::vec_t<3, wp::float32> var_94;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_95;
    const wp::int32 var_96 = 2;
    const wp::int32 var_97 = 2;
    wp::int32 var_98;
    wp::int32 var_99;
    const wp::int32 var_100 = 1;
    wp::int32 var_101;
    wp::vec_t<3, wp::float32>* var_102;
    wp::array_t<wp::vec_t<3, wp::float32>> var_103;
    wp::vec_t<3, wp::float32> var_104;
    wp::vec_t<3, wp::float32> var_105;
    wp::vec_t<3, wp::float32> var_106;
    const wp::int32 var_107 = 0;
    wp::float32 var_108;
    wp::float32 var_109;
    const wp::int32 var_110 = 0;
    wp::float32 var_111;
    wp::float32 var_112;
    wp::float32 var_113;
    const wp::int32 var_114 = 0;
    wp::float32 var_115;
    wp::float32 var_116;
    wp::float32 var_117;
    const wp::int32 var_118 = 0;
    const wp::int32 var_119 = 1;
    wp::float32 var_120;
    wp::float32 var_121;
    const wp::int32 var_122 = 1;
    wp::float32 var_123;
    wp::float32 var_124;
    wp::float32 var_125;
    const wp::int32 var_126 = 1;
    wp::float32 var_127;
    wp::float32 var_128;
    wp::float32 var_129;
    const wp::int32 var_130 = 1;
    const wp::int32 var_131 = 2;
    wp::float32 var_132;
    wp::float32 var_133;
    const wp::int32 var_134 = 2;
    wp::float32 var_135;
    wp::float32 var_136;
    wp::float32 var_137;
    const wp::int32 var_138 = 2;
    wp::float32 var_139;
    wp::float32 var_140;
    wp::float32 var_141;
    const wp::int32 var_142 = 2;
    wp::array_t<wp::int32>* var_143;
    const wp::int32 var_144 = 2;
    const wp::int32 var_145 = 0;
    wp::int32 var_146;
    wp::int32 var_147;
    wp::int32* var_148;
    wp::array_t<wp::int32> var_149;
    wp::int32 var_150;
    wp::int32 var_151;
    wp::array_t<wp::int32>* var_152;
    const wp::int32 var_153 = 2;
    const wp::int32 var_154 = 1;
    wp::int32 var_155;
    wp::int32 var_156;
    wp::int32* var_157;
    wp::array_t<wp::int32> var_158;
    wp::int32 var_159;
    wp::int32 var_160;
    wp::array_t<wp::int32>* var_161;
    const wp::int32 var_162 = 2;
    const wp::int32 var_163 = 2;
    wp::int32 var_164;
    wp::int32 var_165;
    wp::int32* var_166;
    wp::array_t<wp::int32> var_167;
    wp::int32 var_168;
    wp::int32 var_169;
    const wp::int32 var_170 = 1;
    bool var_171;
    bool var_172;
    bool var_173;
    bool var_174;
    bool var_175;
    const wp::float32 var_176 = 0.0;
    const wp::float32 var_177 = 0.0;
    const wp::float32 var_178 = 1.0;
    wp::vec_t<3, wp::float32> var_179;
    wp::mat_t<6, 3, wp::float32>* var_180;
    const wp::int32 var_181 = 3;
    wp::vec_t<3, wp::float32> var_182;
    wp::mat_t<6, 3, wp::float32> var_183;
    wp::mat_t<6, 3, wp::float32>* var_184;
    const wp::int32 var_185 = 4;
    wp::vec_t<3, wp::float32> var_186;
    wp::mat_t<6, 3, wp::float32> var_187;
    wp::mat_t<6, 3, wp::float32>* var_188;
    const wp::int32 var_189 = 5;
    wp::vec_t<3, wp::float32> var_190;
    wp::mat_t<6, 3, wp::float32> var_191;
    const wp::int32 var_192 = 3;
    bool var_193;
    const wp::int32 var_194 = 2;
    bool var_195;
    bool var_196;
    wp::vec_t<3, wp::float32>* var_197;
    const wp::int32 var_198 = 0;
    wp::float32 var_199;
    wp::vec_t<3, wp::float32> var_200;
    wp::float32* var_201;
    wp::float32 var_202;
    wp::float32 var_203;
    const wp::float32 var_204 = 0.0;
    wp::float32* var_205;
    const wp::float32 var_206 = 0.0;
    wp::vec_t<3, wp::float32>* var_207;
    const wp::int32 var_208 = 1;
    wp::float32 var_209;
    wp::vec_t<3, wp::float32> var_210;
    wp::vec_t<3, wp::float32>* var_211;
    const wp::int32 var_212 = 2;
    wp::float32 var_213;
    wp::vec_t<3, wp::float32> var_214;
    wp::vec_t<3, wp::float32> var_215;
    wp::vec_t<3, wp::float32>* var_216;
    SupportPoint_e82efc60 var_217;
    wp::vec_t<3, wp::float32>* var_218;
    const wp::float32 var_219 = 0.5;
    wp::float32 var_220;
    wp::float32 var_221;
    wp::vec_t<3, wp::float32> var_222;
    wp::vec_t<3, wp::float32> var_223;
    wp::vec_t<3, wp::float32> var_224;
    wp::vec_t<3, wp::float32>* var_225;
    const wp::int32 var_226 = 0;
    wp::float32* var_227;
    wp::float32* var_228;
    wp::vec_t<3, wp::float32> var_229;
    wp::vec_t<3, wp::float32> var_230;
    SupportPoint_e82efc60 var_231;
    wp::vec_t<3, wp::float32>* var_232;
    wp::vec_t<3, wp::float32> var_233;
    wp::vec_t<3, wp::float32> var_234;
    wp::vec_t<3, wp::float32> var_235;
    SupportPoint_e82efc60 var_236;
    wp::vec_t<3, wp::float32> var_237;
    const wp::int32 var_238 = 0;
    wp::float32 var_239;
    const wp::float32 var_240 = 0.0;
    bool var_241;
    const wp::int32 var_242 = 1;
    wp::float32 var_243;
    const wp::float32 var_244 = 0.0;
    bool var_245;
    const wp::int32 var_246 = 2;
    wp::float32 var_247;
    const wp::float32 var_248 = 0.0;
    bool var_249;
    bool var_250;
    const wp::int32 var_251 = 0;
    wp::float32 var_252;
    wp::vec_t<3, wp::float32> var_253;
    const wp::int32 var_254 = 1;
    wp::float32 var_255;
    wp::vec_t<3, wp::float32> var_256;
    wp::vec_t<3, wp::float32> var_257;
    const wp::int32 var_258 = 2;
    wp::float32 var_259;
    wp::vec_t<3, wp::float32> var_260;
    wp::vec_t<3, wp::float32> var_261;
    wp::vec_t<3, wp::float32> var_262;
    const wp::int32 var_263 = 1;
    wp::float32 var_264;
    const wp::int32 var_265 = 0;
    bool var_266;
    wp::vec_t<3, wp::float32> var_267;
    const wp::int32 var_268 = 0;
    wp::float32 var_269;
    const wp::int32 var_270 = 0;
    bool var_271;
    wp::vec_t<3, wp::float32> var_272;
    wp::vec_t<3, wp::float32> var_273;
    wp::float32 var_274;
    wp::vec_t<3, wp::float32> var_275;
    wp::vec_t<3, wp::float32> var_276;
    wp::vec_t<3, wp::float32> var_277;
    wp::vec_t<3, wp::float32> var_278;
    wp::float32 var_279;
    wp::float32 var_280;
    wp::vec_t<3, wp::float32> var_281;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_282;
    const wp::int32 var_283 = 2;
    const wp::int32 var_284 = 0;
    wp::int32 var_285;
    wp::int32 var_286;
    wp::vec_t<3, wp::float32>* var_287;
    wp::array_t<wp::vec_t<3, wp::float32>> var_288;
    wp::vec_t<3, wp::float32> var_289;
    wp::vec_t<3, wp::float32> var_290;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_291;
    const wp::int32 var_292 = 2;
    const wp::int32 var_293 = 1;
    wp::int32 var_294;
    wp::int32 var_295;
    wp::vec_t<3, wp::float32>* var_296;
    wp::array_t<wp::vec_t<3, wp::float32>> var_297;
    wp::vec_t<3, wp::float32> var_298;
    wp::vec_t<3, wp::float32> var_299;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_300;
    const wp::int32 var_301 = 2;
    const wp::int32 var_302 = 2;
    wp::int32 var_303;
    wp::int32 var_304;
    wp::vec_t<3, wp::float32>* var_305;
    wp::array_t<wp::vec_t<3, wp::float32>> var_306;
    wp::vec_t<3, wp::float32> var_307;
    wp::vec_t<3, wp::float32> var_308;
    wp::vec_t<3, wp::float32> var_309;
    const wp::int32 var_310 = 0;
    wp::float32 var_311;
    wp::float32 var_312;
    const wp::int32 var_313 = 0;
    wp::float32 var_314;
    wp::float32 var_315;
    wp::float32 var_316;
    const wp::int32 var_317 = 0;
    wp::float32 var_318;
    wp::float32 var_319;
    wp::float32 var_320;
    const wp::int32 var_321 = 0;
    const wp::int32 var_322 = 1;
    wp::float32 var_323;
    wp::float32 var_324;
    const wp::int32 var_325 = 1;
    wp::float32 var_326;
    wp::float32 var_327;
    wp::float32 var_328;
    const wp::int32 var_329 = 1;
    wp::float32 var_330;
    wp::float32 var_331;
    wp::float32 var_332;
    const wp::int32 var_333 = 1;
    const wp::int32 var_334 = 2;
    wp::float32 var_335;
    wp::float32 var_336;
    const wp::int32 var_337 = 2;
    wp::float32 var_338;
    wp::float32 var_339;
    wp::float32 var_340;
    const wp::int32 var_341 = 2;
    wp::float32 var_342;
    wp::float32 var_343;
    wp::float32 var_344;
    const wp::int32 var_345 = 2;
    wp::array_t<wp::float32>* var_346;
    wp::float32* var_347;
    wp::array_t<wp::float32> var_348;
    wp::float32 var_349;
    wp::float32 var_350;
    wp::float32 var_351;
    //---------
    // forward
    // def _epa_witness(                                                                      <L 863>
    // face = _get_face_verts(pt.face[face_idx])                                              <L 866>
    var_0 = &(var_pt.face);
    var_2 = wp::load(var_0);
    var_1 = wp::address(var_2, var_face_idx);
    var_4 = wp::load(var_1);
    var_3 = _get_face_verts_0(var_4);
    // v1 = pt.vert[2 * face[0]] - pt.vert[2 * face[0] + 1]                                   <L 868>
    var_5 = &(var_pt.vert);
    var_8 = wp::extract(var_3, var_7);
    var_9 = wp::mul(var_6, var_8);
    var_11 = wp::load(var_5);
    var_10 = wp::address(var_11, var_9);
    var_12 = &(var_pt.vert);
    var_15 = wp::extract(var_3, var_14);
    var_16 = wp::mul(var_13, var_15);
    var_18 = wp::add(var_16, var_17);
    var_20 = wp::load(var_12);
    var_19 = wp::address(var_20, var_18);
    var_22 = wp::load(var_10);
    var_23 = wp::load(var_19);
    var_21 = wp::sub(var_22, var_23);
    // v2 = pt.vert[2 * face[1]] - pt.vert[2 * face[1] + 1]                                   <L 869>
    var_24 = &(var_pt.vert);
    var_27 = wp::extract(var_3, var_26);
    var_28 = wp::mul(var_25, var_27);
    var_30 = wp::load(var_24);
    var_29 = wp::address(var_30, var_28);
    var_31 = &(var_pt.vert);
    var_34 = wp::extract(var_3, var_33);
    var_35 = wp::mul(var_32, var_34);
    var_37 = wp::add(var_35, var_36);
    var_39 = wp::load(var_31);
    var_38 = wp::address(var_39, var_37);
    var_41 = wp::load(var_29);
    var_42 = wp::load(var_38);
    var_40 = wp::sub(var_41, var_42);
    // v3 = pt.vert[2 * face[2]] - pt.vert[2 * face[2] + 1]                                   <L 870>
    var_43 = &(var_pt.vert);
    var_46 = wp::extract(var_3, var_45);
    var_47 = wp::mul(var_44, var_46);
    var_49 = wp::load(var_43);
    var_48 = wp::address(var_49, var_47);
    var_50 = &(var_pt.vert);
    var_53 = wp::extract(var_3, var_52);
    var_54 = wp::mul(var_51, var_53);
    var_56 = wp::add(var_54, var_55);
    var_58 = wp::load(var_50);
    var_57 = wp::address(var_58, var_56);
    var_60 = wp::load(var_48);
    var_61 = wp::load(var_57);
    var_59 = wp::sub(var_60, var_61);
    // coordinates = _tri_affine_coord(v1, v2, v3, pt.face_pr[face_idx])                      <L 872>
    var_62 = &(var_pt.face_pr);
    var_64 = wp::load(var_62);
    var_63 = wp::address(var_64, var_face_idx);
    var_66 = wp::load(var_63);
    var_65 = _tri_affine_coord_0(var_21, var_40, var_59, var_66);
    // l1 = coordinates[0]                                                                    <L 873>
    var_68 = wp::extract(var_65, var_67);
    // l2 = coordinates[1]                                                                    <L 874>
    var_70 = wp::extract(var_65, var_69);
    // l3 = coordinates[2]                                                                    <L 875>
    var_72 = wp::extract(var_65, var_71);
    // v1 = pt.vert[2 * face[0] + 1]                                                          <L 878>
    var_73 = &(var_pt.vert);
    var_76 = wp::extract(var_3, var_75);
    var_77 = wp::mul(var_74, var_76);
    var_79 = wp::add(var_77, var_78);
    var_81 = wp::load(var_73);
    var_80 = wp::address(var_81, var_79);
    var_83 = wp::load(var_80);
    var_82 = wp::copy(var_83);
    // v2 = pt.vert[2 * face[1] + 1]                                                          <L 879>
    var_84 = &(var_pt.vert);
    var_87 = wp::extract(var_3, var_86);
    var_88 = wp::mul(var_85, var_87);
    var_90 = wp::add(var_88, var_89);
    var_92 = wp::load(var_84);
    var_91 = wp::address(var_92, var_90);
    var_94 = wp::load(var_91);
    var_93 = wp::copy(var_94);
    // v3 = pt.vert[2 * face[2] + 1]                                                          <L 880>
    var_95 = &(var_pt.vert);
    var_98 = wp::extract(var_3, var_97);
    var_99 = wp::mul(var_96, var_98);
    var_101 = wp::add(var_99, var_100);
    var_103 = wp::load(var_95);
    var_102 = wp::address(var_103, var_101);
    var_105 = wp::load(var_102);
    var_104 = wp::copy(var_105);
    // x2 = wp.vec3()                                                                         <L 881>
    var_106 = wp::vec_t<3, wp::float32>();
    // x2[0] = v1[0] * l1 + v2[0] * l2 + v3[0] * l3                                           <L 882>
    var_108 = wp::extract(var_82, var_107);
    var_109 = wp::mul(var_108, var_68);
    var_111 = wp::extract(var_93, var_110);
    var_112 = wp::mul(var_111, var_70);
    var_113 = wp::add(var_109, var_112);
    var_115 = wp::extract(var_104, var_114);
    var_116 = wp::mul(var_115, var_72);
    var_117 = wp::add(var_113, var_116);
    wp::assign_inplace(var_106, var_118, var_117);
    // x2[1] = v1[1] * l1 + v2[1] * l2 + v3[1] * l3                                           <L 883>
    var_120 = wp::extract(var_82, var_119);
    var_121 = wp::mul(var_120, var_68);
    var_123 = wp::extract(var_93, var_122);
    var_124 = wp::mul(var_123, var_70);
    var_125 = wp::add(var_121, var_124);
    var_127 = wp::extract(var_104, var_126);
    var_128 = wp::mul(var_127, var_72);
    var_129 = wp::add(var_125, var_128);
    wp::assign_inplace(var_106, var_130, var_129);
    // x2[2] = v1[2] * l1 + v2[2] * l2 + v3[2] * l3                                           <L 884>
    var_132 = wp::extract(var_82, var_131);
    var_133 = wp::mul(var_132, var_68);
    var_135 = wp::extract(var_93, var_134);
    var_136 = wp::mul(var_135, var_70);
    var_137 = wp::add(var_133, var_136);
    var_139 = wp::extract(var_104, var_138);
    var_140 = wp::mul(var_139, var_72);
    var_141 = wp::add(var_137, var_140);
    wp::assign_inplace(var_106, var_142, var_141);
    // i1 = pt.vert_index[2 * face[0]]                                                        <L 887>
    var_143 = &(var_pt.vert_index);
    var_146 = wp::extract(var_3, var_145);
    var_147 = wp::mul(var_144, var_146);
    var_149 = wp::load(var_143);
    var_148 = wp::address(var_149, var_147);
    var_151 = wp::load(var_148);
    var_150 = wp::copy(var_151);
    // i2 = pt.vert_index[2 * face[1]]                                                        <L 888>
    var_152 = &(var_pt.vert_index);
    var_155 = wp::extract(var_3, var_154);
    var_156 = wp::mul(var_153, var_155);
    var_158 = wp::load(var_152);
    var_157 = wp::address(var_158, var_156);
    var_160 = wp::load(var_157);
    var_159 = wp::copy(var_160);
    // i3 = pt.vert_index[2 * face[2]]                                                        <L 889>
    var_161 = &(var_pt.vert_index);
    var_164 = wp::extract(var_3, var_163);
    var_165 = wp::mul(var_162, var_164);
    var_167 = wp::load(var_161);
    var_166 = wp::address(var_167, var_165);
    var_169 = wp::load(var_166);
    var_168 = wp::copy(var_169);
    // if geomtype1 == GeomType.HFIELD and (i1 != i2 or i1 != i3):                            <L 890>
    var_171 = (var_geomtype1 == var_170);
    var_172 = (var_150 != var_159);
    var_173 = (var_150 != var_168);
    var_174 = var_172 || var_173;
    var_175 = var_171 && var_174;
    if (var_175) {
        // n = wp.vec3(0.0, 0.0, 1.0)                                                         <L 892>
        var_179 = wp::vec_t<3, wp::float32>(var_176, var_177, var_178);
        // a = geom1.hfprism[3]                                                               <L 895>
        var_180 = &(var_geom1.hfprism);
        var_183 = wp::load(var_180);
        var_182 = wp::extract(var_183, var_181);
        // b = geom1.hfprism[4]                                                               <L 896>
        var_184 = &(var_geom1.hfprism);
        var_187 = wp::load(var_184);
        var_186 = wp::extract(var_187, var_185);
        // c = geom1.hfprism[5]                                                               <L 897>
        var_188 = &(var_geom1.hfprism);
        var_191 = wp::load(var_188);
        var_190 = wp::extract(var_191, var_189);
        // if geomtype2 == GeomType.CAPSULE or geomtype2 == GeomType.SPHERE:                  <L 900>
        var_193 = (var_geomtype2 == var_192);
        var_195 = (var_geomtype2 == var_194);
        var_196 = var_193 || var_195;
        if (var_196) {
            // radius = geom2.size[0]                                                         <L 901>
            var_197 = &(var_geom2.size);
            var_200 = wp::load(var_197);
            var_199 = wp::extract(var_200, var_198);
            // margin = geom2.margin                                                          <L 902>
            var_201 = &(var_geom2.margin);
            var_203 = wp::load(var_201);
            var_202 = wp::copy(var_203);
            // geom2.margin = 0.0                                                             <L 903>
            var_205 = &(var_geom2.margin);
            wp::store(var_205, var_204);
            // geom2.size = wp.vec3(0.0, geom2.size[1], geom2.size[2])                        <L 904>
            var_207 = &(var_geom2.size);
            var_210 = wp::load(var_207);
            var_209 = wp::extract(var_210, var_208);
            var_211 = &(var_geom2.size);
            var_214 = wp::load(var_211);
            var_213 = wp::extract(var_214, var_212);
            var_215 = wp::vec_t<3, wp::float32>(var_206, var_209, var_213);
            var_216 = &(var_geom2.size);
            wp::store(var_216, var_215);
            // sp = support(geom2, geomtype2, x2)                                             <L 905>
            var_217 = support_0(var_geom2, var_geomtype2, var_106);
            // x2 = sp.point - (0.5 * margin + radius) * n                                    <L 906>
            var_218 = &(var_217.point);
            var_220 = wp::mul(var_219, var_202);
            var_221 = wp::add(var_220, var_199);
            var_222 = wp::mul(var_221, var_179);
            var_224 = wp::load(var_218);
            var_223 = wp::sub(var_224, var_222);
            // geom2.size[0] = radius                                                         <L 907>
            var_225 = &(var_geom2.size);
            var_227 = wp::indexref(var_225, var_226);
            wp::store(var_227, var_199);
            // geom2.margin = margin                                                          <L 908>
            var_228 = &(var_geom2.margin);
            wp::store(var_228, var_202);
        }
        var_229 = wp::where(var_196, var_223, var_106);
        if (!var_196) {
            // x2 = wp.normalize(x2)                                                          <L 910>
            var_230 = wp::normalize(var_229);
            // sp = support(geom2, geomtype2, x2)                                             <L 911>
            var_231 = support_0(var_geom2, var_geomtype2, var_230);
            // x2 = sp.point                                                                  <L 912>
            var_232 = &(var_231.point);
            var_234 = wp::load(var_232);
            var_233 = wp::copy(var_234);
        }
        var_235 = wp::where(var_196, var_229, var_233);
        var_236 = wp::where(var_196, var_217, var_231);
        // coordinates2 = _tri_affine_coord(a, b, c, x2)                                      <L 914>
        var_237 = _tri_affine_coord_0(var_182, var_186, var_190, var_235);
        // if coordinates2[0] > 0.0 and coordinates2[1] > 0.0 and coordinates2[2] > 0.0:       <L 915>
        var_239 = wp::extract(var_237, var_238);
        var_241 = (var_239 > var_240);
        var_243 = wp::extract(var_237, var_242);
        var_245 = (var_243 > var_244);
        var_247 = wp::extract(var_237, var_246);
        var_249 = (var_247 > var_248);
        var_250 = var_241 && var_245 && var_249;
        if (var_250) {
            // x1 = coordinates2[0] * a + coordinates2[1] * b + coordinates2[2] * c           <L 916>
            var_252 = wp::extract(var_237, var_251);
            var_253 = wp::mul(var_252, var_182);
            var_255 = wp::extract(var_237, var_254);
            var_256 = wp::mul(var_255, var_186);
            var_257 = wp::add(var_253, var_256);
            var_259 = wp::extract(var_237, var_258);
            var_260 = wp::mul(var_259, var_190);
            var_261 = wp::add(var_257, var_260);
        }
        if (!var_250) {
            // p = c                                                                          <L 918>
            var_262 = wp::copy(var_190);
            // p = wp.where(coordinates2[1] > 0, b, p)                                        <L 919>
            var_264 = wp::extract(var_237, var_263);
            var_266 = (var_264 > var_265);
            var_267 = wp::where(var_266, var_186, var_262);
            // p = wp.where(coordinates2[0] > 0, a, p)                                        <L 920>
            var_269 = wp::extract(var_237, var_268);
            var_271 = (var_269 > var_270);
            var_272 = wp::where(var_271, var_182, var_267);
            // x1 = x2 - wp.dot(x2 - p, n) * n                                                <L 921>
            var_273 = wp::sub(var_235, var_272);
            var_274 = wp::dot(var_273, var_179);
            var_275 = wp::mul(var_274, var_179);
            var_276 = wp::sub(var_235, var_275);
        }
        var_277 = wp::where(var_250, var_261, var_276);
        // return x1, x2, -wp.norm_l2(x1 - x2)                                                <L 922>
        var_278 = wp::sub(var_277, var_235);
        var_279 = norm_l2_0(var_278);
        var_280 = wp::neg(var_279);
        ret_0 = var_277;
        ret_1 = var_235;
        ret_2 = var_280;
        return;
    }
    var_281 = wp::where(var_175, var_235, var_106);
    // v1 = pt.vert[2 * face[0]]                                                              <L 925>
    var_282 = &(var_pt.vert);
    var_285 = wp::extract(var_3, var_284);
    var_286 = wp::mul(var_283, var_285);
    var_288 = wp::load(var_282);
    var_287 = wp::address(var_288, var_286);
    var_290 = wp::load(var_287);
    var_289 = wp::copy(var_290);
    // v2 = pt.vert[2 * face[1]]                                                              <L 926>
    var_291 = &(var_pt.vert);
    var_294 = wp::extract(var_3, var_293);
    var_295 = wp::mul(var_292, var_294);
    var_297 = wp::load(var_291);
    var_296 = wp::address(var_297, var_295);
    var_299 = wp::load(var_296);
    var_298 = wp::copy(var_299);
    // v3 = pt.vert[2 * face[2]]                                                              <L 927>
    var_300 = &(var_pt.vert);
    var_303 = wp::extract(var_3, var_302);
    var_304 = wp::mul(var_301, var_303);
    var_306 = wp::load(var_300);
    var_305 = wp::address(var_306, var_304);
    var_308 = wp::load(var_305);
    var_307 = wp::copy(var_308);
    // x1 = wp.vec3()                                                                         <L 928>
    var_309 = wp::vec_t<3, wp::float32>();
    // x1[0] = v1[0] * l1 + v2[0] * l2 + v3[0] * l3                                           <L 929>
    var_311 = wp::extract(var_289, var_310);
    var_312 = wp::mul(var_311, var_68);
    var_314 = wp::extract(var_298, var_313);
    var_315 = wp::mul(var_314, var_70);
    var_316 = wp::add(var_312, var_315);
    var_318 = wp::extract(var_307, var_317);
    var_319 = wp::mul(var_318, var_72);
    var_320 = wp::add(var_316, var_319);
    wp::assign_inplace(var_309, var_321, var_320);
    // x1[1] = v1[1] * l1 + v2[1] * l2 + v3[1] * l3                                           <L 930>
    var_323 = wp::extract(var_289, var_322);
    var_324 = wp::mul(var_323, var_68);
    var_326 = wp::extract(var_298, var_325);
    var_327 = wp::mul(var_326, var_70);
    var_328 = wp::add(var_324, var_327);
    var_330 = wp::extract(var_307, var_329);
    var_331 = wp::mul(var_330, var_72);
    var_332 = wp::add(var_328, var_331);
    wp::assign_inplace(var_309, var_333, var_332);
    // x1[2] = v1[2] * l1 + v2[2] * l2 + v3[2] * l3                                           <L 931>
    var_335 = wp::extract(var_289, var_334);
    var_336 = wp::mul(var_335, var_68);
    var_338 = wp::extract(var_298, var_337);
    var_339 = wp::mul(var_338, var_70);
    var_340 = wp::add(var_336, var_339);
    var_342 = wp::extract(var_307, var_341);
    var_343 = wp::mul(var_342, var_72);
    var_344 = wp::add(var_340, var_343);
    wp::assign_inplace(var_309, var_345, var_344);
    // return x1, x2, -wp.sqrt(pt.face_norm2[face_idx])                                       <L 933>
    var_346 = &(var_pt.face_norm2);
    var_348 = wp::load(var_346);
    var_347 = wp::address(var_348, var_face_idx);
    var_350 = wp::load(var_347);
    var_349 = wp::sqrt(var_350);
    var_351 = wp::neg(var_349);
    ret_0 = var_309;
    ret_1 = var_281;
    ret_2 = var_351;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:1201
static CUDA_CALLABLE void _epa_0(
    wp::float32 var_tolerance,
    wp::int32 var_epa_iterations,
    Polytope_9ab93ade var_pt,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2,
    bool var_is_discrete,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::int32 & ret_3)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 1e+30;
    wp::float32 var_1;
    wp::float32 var_2;
    const wp::int32 var_3 = 1;
    const wp::int32 var_4 = -1;
    wp::int32 var_5;
    const wp::int32 var_6 = 1;
    const wp::int32 var_7 = -1;
    wp::int32 var_8;
    const wp::float32 var_9 = 1e-15;
    wp::float32 var_10;
    wp::int32* var_11;
    wp::int32 var_12;
    wp::int32 var_13;
    const wp::int32 var_14 = 1000;
    wp::int32 var_15;
    wp::range_t var_16;
    wp::int32 var_17;
    wp::int32 var_18;
    const wp::int32 var_19 = 1;
    const wp::int32 var_20 = -1;
    wp::int32 var_21;
    wp::float32 var_22;
    wp::int32* var_23;
    wp::range_t var_24;
    wp::int32 var_25;
    wp::int32 var_26;
    wp::array_t<wp::int32>* var_27;
    wp::int32* var_28;
    wp::array_t<wp::int32> var_29;
    bool var_30;
    wp::int32 var_31;
    bool var_32;
    wp::array_t<wp::float32>* var_33;
    wp::float32* var_34;
    wp::array_t<wp::float32> var_35;
    bool var_36;
    wp::float32 var_37;
    bool var_38;
    wp::int32 var_39;
    wp::array_t<wp::float32>* var_40;
    wp::float32* var_41;
    wp::array_t<wp::float32> var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::int32 var_45;
    wp::float32 var_46;
    bool var_47;
    const wp::int32 var_48 = 0;
    bool var_49;
    bool var_50;
    wp::int32 var_51;
    wp::int32 var_52;
    wp::int32 var_53;
    const wp::float32 var_54 = 0.0;
    bool var_55;
    wp::int32 var_56;
    wp::int32 var_57;
    wp::float32 var_58;
    wp::int32* var_59;
    wp::int32 var_60;
    wp::int32 var_61;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_62;
    wp::vec_t<3, wp::float32>* var_63;
    wp::array_t<wp::vec_t<3, wp::float32>> var_64;
    wp::vec_t<3, wp::float32> var_65;
    wp::vec_t<3, wp::float32> var_66;
    wp::int32 var_67;
    wp::int32 var_68;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_69;
    const wp::int32 var_70 = 2;
    wp::int32 var_71;
    wp::vec_t<3, wp::float32>* var_72;
    wp::array_t<wp::vec_t<3, wp::float32>> var_73;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_74;
    const wp::int32 var_75 = 2;
    wp::int32 var_76;
    const wp::int32 var_77 = 1;
    wp::int32 var_78;
    wp::vec_t<3, wp::float32>* var_79;
    wp::array_t<wp::vec_t<3, wp::float32>> var_80;
    wp::vec_t<3, wp::float32> var_81;
    wp::vec_t<3, wp::float32> var_82;
    wp::vec_t<3, wp::float32> var_83;
    wp::int32* var_84;
    wp::int32* var_85;
    const wp::int32 var_86 = 1;
    wp::int32* var_87;
    const wp::int32 var_88 = 1;
    wp::int32 var_89;
    wp::int32 var_90;
    wp::int32* var_91;
    wp::float32 var_92;
    bool var_93;
    wp::float32 var_94;
    wp::float32 var_95;
    wp::float32 var_96;
    wp::float32 var_97;
    wp::float32 var_98;
    bool var_99;
    wp::float32 var_100;
    wp::float32 var_101;
    wp::int32 var_102;
    wp::int32 var_103;
    const bool var_104 = false;
    bool var_105;
    wp::int32* var_106;
    const wp::int32 var_107 = 1;
    wp::int32 var_108;
    wp::int32 var_109;
    wp::range_t var_110;
    wp::int32 var_111;
    wp::array_t<wp::int32>* var_112;
    const wp::int32 var_113 = 2;
    wp::int32 var_114;
    wp::int32* var_115;
    wp::array_t<wp::int32> var_116;
    wp::array_t<wp::int32>* var_117;
    const wp::int32 var_118 = 2;
    wp::int32 var_119;
    wp::int32* var_120;
    wp::array_t<wp::int32> var_121;
    bool var_122;
    wp::int32 var_123;
    wp::int32 var_124;
    wp::array_t<wp::int32>* var_125;
    const wp::int32 var_126 = 2;
    wp::int32 var_127;
    const wp::int32 var_128 = 1;
    wp::int32 var_129;
    wp::int32* var_130;
    wp::array_t<wp::int32> var_131;
    wp::array_t<wp::int32>* var_132;
    const wp::int32 var_133 = 2;
    wp::int32 var_134;
    const wp::int32 var_135 = 1;
    wp::int32 var_136;
    wp::int32* var_137;
    wp::array_t<wp::int32> var_138;
    bool var_139;
    wp::int32 var_140;
    wp::int32 var_141;
    bool var_142;
    const bool var_143 = true;
    wp::float32 var_144;
    wp::float32 var_145;
    wp::int32 var_146;
    wp::int32 var_147;
    wp::float32 var_148;
    wp::float32 var_149;
    wp::int32 var_150;
    wp::int32 var_151;
    wp::int32 var_152;
    const wp::int32 var_153 = 1;
    wp::int32 var_154;
    wp::array_t<wp::int32>* var_155;
    wp::int32* var_156;
    wp::array_t<wp::int32> var_157;
    wp::int32 var_158;
    wp::int32 var_159;
    wp::array_t<wp::int32>* var_160;
    wp::array_t<wp::int32> var_161;
    wp::array_t<wp::int32>* var_162;
    wp::int32* var_163;
    wp::array_t<wp::int32> var_164;
    wp::vec_t<3, wp::int32> var_165;
    wp::int32 var_166;
    const wp::int32 var_167 = 0;
    wp::int32 var_168;
    const wp::int32 var_169 = 1;
    wp::int32 var_170;
    wp::int32 var_171;
    wp::int32* var_172;
    const wp::int32 var_173 = 1;
    wp::int32 var_174;
    const wp::int32 var_175 = 2;
    wp::int32 var_176;
    wp::int32 var_177;
    wp::int32* var_178;
    const wp::int32 var_179 = 2;
    wp::int32 var_180;
    const wp::int32 var_181 = 0;
    wp::int32 var_182;
    wp::int32 var_183;
    wp::int32* var_184;
    wp::int32* var_185;
    const wp::int32 var_186 = 1;
    const wp::int32 var_187 = -1;
    bool var_188;
    wp::int32 var_189;
    const wp::str var_190 = "Warning: EPA horizon = %d isn't large enough.\n";
    wp::array_t<wp::int32>* var_191;
    wp::shape_t* var_192;
    const wp::int32 var_193 = 0;
    wp::int32 var_194;
    wp::shape_t var_195;
    const wp::int32 var_196 = 1;
    const wp::int32 var_197 = -1;
    wp::float32 var_198;
    wp::float32 var_199;
    wp::int32 var_200;
    wp::int32 var_201;
    wp::int32 var_202;
    wp::int32* var_203;
    wp::range_t var_204;
    wp::int32 var_205;
    wp::int32 var_206;
    wp::array_t<wp::int32>* var_207;
    wp::int32* var_208;
    wp::array_t<wp::int32> var_209;
    bool var_210;
    wp::int32 var_211;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_212;
    wp::vec_t<3, wp::float32>* var_213;
    wp::array_t<wp::vec_t<3, wp::float32>> var_214;
    wp::float32 var_215;
    wp::vec_t<3, wp::float32> var_216;
    wp::array_t<wp::float32>* var_217;
    wp::float32* var_218;
    wp::array_t<wp::float32> var_219;
    wp::float32 var_220;
    wp::float32 var_221;
    const wp::float32 var_222 = 1e-10;
    bool var_223;
    wp::array_t<wp::int32>* var_224;
    wp::int32* var_225;
    wp::array_t<wp::int32> var_226;
    bool var_227;
    wp::int32 var_228;
    const wp::int32 var_229 = 1;
    wp::int32 var_230;
    wp::int32 var_231;
    wp::array_t<wp::int32>* var_232;
    wp::int32* var_233;
    wp::array_t<wp::int32> var_234;
    wp::int32 var_235;
    wp::int32 var_236;
    wp::array_t<wp::int32>* var_237;
    wp::array_t<wp::int32> var_238;
    wp::array_t<wp::int32>* var_239;
    wp::int32* var_240;
    wp::array_t<wp::int32> var_241;
    wp::vec_t<3, wp::int32> var_242;
    wp::int32 var_243;
    const wp::int32 var_244 = 0;
    wp::int32 var_245;
    const wp::int32 var_246 = 1;
    wp::int32 var_247;
    wp::int32 var_248;
    wp::int32* var_249;
    const wp::int32 var_250 = 1;
    wp::int32 var_251;
    const wp::int32 var_252 = 2;
    wp::int32 var_253;
    wp::int32 var_254;
    wp::int32* var_255;
    const wp::int32 var_256 = 2;
    wp::int32 var_257;
    const wp::int32 var_258 = 0;
    wp::int32 var_259;
    wp::int32 var_260;
    wp::int32* var_261;
    wp::int32* var_262;
    const wp::int32 var_263 = 1;
    const wp::int32 var_264 = -1;
    bool var_265;
    wp::int32 var_266;
    const wp::str var_267 = "Warning: EPA horizon = %d isn't large enough.\n";
    wp::array_t<wp::int32>* var_268;
    wp::shape_t* var_269;
    const wp::int32 var_270 = 0;
    wp::int32 var_271;
    wp::shape_t var_272;
    const wp::int32 var_273 = 1;
    const wp::int32 var_274 = -1;
    wp::int32 var_275;
    wp::vec_t<3, wp::int32> var_276;
    wp::int32 var_277;
    wp::vec_t<3, wp::int32> var_278;
    wp::int32* var_279;
    wp::range_t var_280;
    wp::int32 var_281;
    wp::int32 var_282;
    wp::array_t<wp::int32>* var_283;
    wp::int32* var_284;
    wp::array_t<wp::int32> var_285;
    wp::vec_t<2, wp::int32> var_286;
    wp::int32 var_287;
    wp::int32* var_288;
    const wp::int32 var_289 = 0;
    wp::int32 var_290;
    const wp::int32 var_291 = 1;
    wp::int32 var_292;
    wp::float32 var_293;
    wp::int32 var_294;
    const wp::float32 var_295 = 0.0;
    bool var_296;
    const wp::int32 var_297 = 1;
    const wp::int32 var_298 = -1;
    const wp::int32 var_299 = 1;
    wp::int32* var_300;
    const wp::int32 var_301 = 1;
    wp::int32 var_302;
    wp::int32 var_303;
    wp::int32* var_304;
    bool var_305;
    bool var_306;
    bool var_307;
    const wp::int32 var_308 = 1;
    wp::int32 var_309;
    wp::int32 var_310;
    wp::array_t<wp::int32>* var_311;
    wp::int32* var_312;
    const wp::int32 var_313 = 1;
    wp::int32 var_314;
    wp::int32 var_315;
    wp::int32* var_316;
    wp::array_t<wp::int32> var_317;
    wp::int32 var_318;
    wp::int32 var_319;
    wp::array_t<wp::int32>* var_320;
    wp::int32* var_321;
    const wp::int32 var_322 = 1;
    wp::int32 var_323;
    wp::int32 var_324;
    wp::array_t<wp::int32> var_325;
    const wp::int32 var_326 = 0;
    bool var_327;
    const wp::int32 var_328 = 1;
    const wp::int32 var_329 = -1;
    bool var_330;
    bool var_331;
    wp::float32 var_332;
    wp::float32 var_333;
    wp::int32 var_334;
    wp::int32 var_335;
    wp::int32 var_336;
    const wp::int32 var_337 = 0;
    wp::int32* var_338;
    const wp::int32 var_339 = 1;
    const wp::int32 var_340 = -1;
    bool var_341;
    wp::vec_t<3, wp::float32> var_342;
    wp::vec_t<3, wp::float32> var_343;
    wp::float32 var_344;
    const wp::float32 var_345 = 0.0;
    wp::vec_t<3, wp::float32> var_346;
    wp::vec_t<3, wp::float32> var_347;
    const wp::int32 var_348 = 1;
    const wp::int32 var_349 = -1;
    //---------
    // forward
    // def _epa(                                                                              <L 1202>
    // upper = FLOAT_MAX                                                                      <L 1214>
    var_1 = wp::copy(var_0);
    // upper2 = FLOAT_MAX                                                                     <L 1215>
    var_2 = wp::copy(var_0);
    // idx = int(-1)                                                                          <L 1216>
    var_5 = wp::int(var_4);
    // pidx = int(-1)                                                                         <L 1217>
    var_8 = wp::int(var_7);
    // epsilon = wp.where(is_discrete, 1e-15, tolerance)                                      <L 1218>
    var_10 = wp::where(var_is_discrete, var_9, var_tolerance);
    // nvalid = pt.nface  # number of potential faces for expanding the polytope              <L 1219>
    var_11 = &(var_pt.nface);
    var_13 = wp::load(var_11);
    var_12 = wp::copy(var_13);
    // epa_iterations = wp.min(epa_iterations, 1000)                                          <L 1224>
    var_15 = wp::min(var_epa_iterations, var_14);
    // for _ in range(epa_iterations):                                                        <L 1225>
    var_16 = wp::range(var_15);
    start_for_0:;
        if (iter_cmp(var_16) == 0) goto end_for_0;
        var_17 = wp::iter_next(var_16);
        // pidx = idx                                                                         <L 1226>
        var_18 = wp::copy(var_5);
        // idx = int(-1)                                                                      <L 1227>
        var_21 = wp::int(var_20);
        // lower2 = float(FLOAT_MAX)                                                          <L 1228>
        var_22 = wp::float(var_0);
        // for i in range(pt.nface):                                                          <L 1231>
        var_23 = &(var_pt.nface);
        var_25 = wp::load(var_23);
        var_24 = wp::range(var_25);
        start_for_2:;
            if (iter_cmp(var_24) == 0) goto end_for_2;
            var_26 = wp::iter_next(var_24);
            // if not _is_invalid_face(pt.face[i]) and pt.face_norm2[i] < lower2:             <L 1232>
            var_27 = &(var_pt.face);
            var_29 = wp::load(var_27);
            var_28 = wp::address(var_29, var_26);
            var_31 = wp::load(var_28);
            var_30 = _is_invalid_face_0(var_31);
            var_32 = wp::unot(var_30);
            var_33 = &(var_pt.face_norm2);
            var_35 = wp::load(var_33);
            var_34 = wp::address(var_35, var_26);
            var_37 = wp::load(var_34);
            var_36 = (var_37 < var_22);
            var_38 = var_32 && var_36;
            if (var_38) {
                // idx = i                                                                    <L 1233>
                var_39 = wp::copy(var_26);
                // lower2 = pt.face_norm2[i]                                                  <L 1234>
                var_40 = &(var_pt.face_norm2);
                var_42 = wp::load(var_40);
                var_41 = wp::address(var_42, var_26);
                var_44 = wp::load(var_41);
                var_43 = wp::copy(var_44);
            }
            var_45 = wp::where(var_38, var_39, var_21);
            var_46 = wp::where(var_38, var_43, var_22);
            wp::assign(var_21, var_45);
            wp::assign(var_22, var_46);
            goto start_for_2;
        end_for_2:;
        // if lower2 > upper2 or idx < 0:                                                     <L 1237>
        var_47 = (var_22 > var_2);
        var_49 = (var_21 < var_48);
        var_50 = var_47 || var_49;
        if (var_50) {
            // idx = pidx                                                                     <L 1238>
            var_51 = wp::copy(var_18);
            // break                                                                          <L 1239>
            wp::assign(var_5, var_51);
            wp::assign(var_8, var_18);
            goto end_for_0;
        }
        var_52 = wp::where(var_50, var_5, var_21);
        var_53 = wp::where(var_50, var_8, var_18);
        // if lower2 <= 0.0:                                                                  <L 1242>
        var_55 = (var_22 <= var_54);
        if (var_55) {
            // break                                                                          <L 1243>
            wp::assign(var_5, var_52);
            wp::assign(var_8, var_53);
            goto end_for_0;
        }
        var_56 = wp::where(var_55, var_5, var_52);
        var_57 = wp::where(var_55, var_8, var_53);
        // lower = wp.sqrt(lower2)                                                            <L 1246>
        var_58 = wp::sqrt(var_22);
        // wi = pt.nvert                                                                      <L 1247>
        var_59 = &(var_pt.nvert);
        var_61 = wp::load(var_59);
        var_60 = wp::copy(var_61);
        // face_pr_normalized = pt.face_pr[idx] / lower                                       <L 1248>
        var_62 = &(var_pt.face_pr);
        var_64 = wp::load(var_62);
        var_63 = wp::address(var_64, var_56);
        var_66 = wp::load(var_63);
        var_65 = wp::div(var_66, var_58);
        // i1, i2 = _epa_support(pt, wi, geom1, geom2, geomtype1, geomtype2, face_pr_normalized)       <L 1249>
        _epa_support_0(var_pt, var_60, var_geom1, var_geom2, var_geomtype1, var_geomtype2, var_65, var_67, var_68);
        // w = pt.vert[2 * wi] - pt.vert[2 * wi + 1]                                          <L 1250>
        var_69 = &(var_pt.vert);
        var_71 = wp::mul(var_70, var_60);
        var_73 = wp::load(var_69);
        var_72 = wp::address(var_73, var_71);
        var_74 = &(var_pt.vert);
        var_76 = wp::mul(var_75, var_60);
        var_78 = wp::add(var_76, var_77);
        var_80 = wp::load(var_74);
        var_79 = wp::address(var_80, var_78);
        var_82 = wp::load(var_72);
        var_83 = wp::load(var_79);
        var_81 = wp::sub(var_82, var_83);
        // geom1.index = i1                                                                   <L 1251>
        var_84 = &(var_geom1.index);
        wp::store(var_84, var_67);
        // geom2.index = i2                                                                   <L 1252>
        var_85 = &(var_geom2.index);
        wp::store(var_85, var_68);
        // pt.nvert += 1                                                                      <L 1253>
        var_87 = &(var_pt.nvert);
        var_90 = wp::load(var_87);
        var_89 = wp::add(var_90, var_88);
        var_91 = &(var_pt.nvert);
        wp::store(var_91, var_89);
        // upper_k = wp.dot(face_pr_normalized, w)                                            <L 1256>
        var_92 = wp::dot(var_65, var_81);
        // if upper_k < upper:                                                                <L 1257>
        var_93 = (var_92 < var_1);
        if (var_93) {
            // upper = upper_k                                                                <L 1258>
            var_94 = wp::copy(var_92);
            // upper2 = upper * upper                                                         <L 1259>
            var_95 = wp::mul(var_94, var_94);
        }
        var_96 = wp::where(var_93, var_94, var_1);
        var_97 = wp::where(var_93, var_95, var_2);
        // if upper - lower < epsilon:                                                        <L 1261>
        var_98 = wp::sub(var_96, var_58);
        var_99 = (var_98 < var_10);
        if (var_99) {
            // break                                                                          <L 1262>
            wp::assign(var_1, var_96);
            wp::assign(var_2, var_97);
            wp::assign(var_5, var_56);
            wp::assign(var_8, var_57);
            goto end_for_0;
        }
        var_100 = wp::where(var_99, var_1, var_96);
        var_101 = wp::where(var_99, var_2, var_97);
        var_102 = wp::where(var_99, var_5, var_56);
        var_103 = wp::where(var_99, var_8, var_57);
        // if is_discrete:                                                                    <L 1265>
        if (var_is_discrete) {
            // found_repeated = bool(False)                                                   <L 1266>
            var_105 = bool(var_104);
            // for i in range(pt.nvert - 1):                                                  <L 1267>
            var_106 = &(var_pt.nvert);
            var_109 = wp::load(var_106);
            var_108 = wp::sub(var_109, var_107);
            var_110 = wp::range(var_108);
            start_for_4:;
                if (iter_cmp(var_110) == 0) goto end_for_4;
                var_111 = wp::iter_next(var_110);
                // if pt.vert_index[2 * i] == pt.vert_index[2 * wi] and pt.vert_index[2 * i + 1] == pt.vert_index[2 * wi + 1]:       <L 1268>
                var_112 = &(var_pt.vert_index);
                var_114 = wp::mul(var_113, var_111);
                var_116 = wp::load(var_112);
                var_115 = wp::address(var_116, var_114);
                var_117 = &(var_pt.vert_index);
                var_119 = wp::mul(var_118, var_60);
                var_121 = wp::load(var_117);
                var_120 = wp::address(var_121, var_119);
                var_123 = wp::load(var_115);
                var_124 = wp::load(var_120);
                var_122 = (var_123 == var_124);
                var_125 = &(var_pt.vert_index);
                var_127 = wp::mul(var_126, var_111);
                var_129 = wp::add(var_127, var_128);
                var_131 = wp::load(var_125);
                var_130 = wp::address(var_131, var_129);
                var_132 = &(var_pt.vert_index);
                var_134 = wp::mul(var_133, var_60);
                var_136 = wp::add(var_134, var_135);
                var_138 = wp::load(var_132);
                var_137 = wp::address(var_138, var_136);
                var_140 = wp::load(var_130);
                var_141 = wp::load(var_137);
                var_139 = (var_140 == var_141);
                var_142 = var_122 && var_139;
                if (var_142) {
                    // found_repeated = True                                                  <L 1269>
                    // break                                                                  <L 1270>
                    wp::assign(var_105, var_143);
                    goto end_for_4;
                }
                goto start_for_4;
            end_for_4:;
            // if found_repeated:                                                             <L 1271>
            if (var_105) {
                // break                                                                      <L 1272>
                wp::assign(var_1, var_100);
                wp::assign(var_2, var_101);
                wp::assign(var_5, var_102);
                wp::assign(var_8, var_103);
                goto end_for_0;
            }
            var_144 = wp::where(var_105, var_1, var_100);
            var_145 = wp::where(var_105, var_2, var_101);
            var_146 = wp::where(var_105, var_5, var_102);
            var_147 = wp::where(var_105, var_8, var_103);
        }
        var_148 = wp::where(var_is_discrete, var_144, var_100);
        var_149 = wp::where(var_is_discrete, var_145, var_101);
        var_150 = wp::where(var_is_discrete, var_146, var_102);
        var_151 = wp::where(var_is_discrete, var_147, var_103);
        var_152 = wp::where(var_is_discrete, var_111, var_26);
        // nvalid -= 1                                                                        <L 1274>
        var_154 = wp::sub(var_12, var_153);
        // pt.face[idx] = _delete_face(pt.face[idx])                                          <L 1275>
        var_155 = &(var_pt.face);
        var_157 = wp::load(var_155);
        var_156 = wp::address(var_157, var_150);
        var_159 = wp::load(var_156);
        var_158 = _delete_face_0(var_159);
        var_160 = &(var_pt.face);
        var_161 = wp::load(var_160);
        wp::array_store(var_161, var_150, var_158);
        // face = _get_face_verts(pt.face[idx])                                               <L 1276>
        var_162 = &(var_pt.face);
        var_164 = wp::load(var_162);
        var_163 = wp::address(var_164, var_150);
        var_166 = wp::load(var_163);
        var_165 = _get_face_verts_0(var_166);
        // pt.nhorizon = _add_edge(pt, face[0], face[1])                                      <L 1277>
        var_168 = wp::extract(var_165, var_167);
        var_170 = wp::extract(var_165, var_169);
        var_171 = _add_edge_0(var_pt, var_168, var_170);
        var_172 = &(var_pt.nhorizon);
        wp::store(var_172, var_171);
        // pt.nhorizon = _add_edge(pt, face[1], face[2])                                      <L 1278>
        var_174 = wp::extract(var_165, var_173);
        var_176 = wp::extract(var_165, var_175);
        var_177 = _add_edge_0(var_pt, var_174, var_176);
        var_178 = &(var_pt.nhorizon);
        wp::store(var_178, var_177);
        // pt.nhorizon = _add_edge(pt, face[2], face[0])                                      <L 1279>
        var_180 = wp::extract(var_165, var_179);
        var_182 = wp::extract(var_165, var_181);
        var_183 = _add_edge_0(var_pt, var_180, var_182);
        var_184 = &(var_pt.nhorizon);
        wp::store(var_184, var_183);
        // if pt.nhorizon == -1:                                                              <L 1280>
        var_185 = &(var_pt.nhorizon);
        var_189 = wp::load(var_185);
        var_188 = (var_189 == var_187);
        if (var_188) {
            // wp.printf("Warning: EPA horizon = %d isn't large enough.\n", pt.horizon.shape[0])       <L 1281>
            var_191 = &(var_pt.horizon);
            var_192 = &(var_191->shape);
            var_195 = wp::load(var_192);
            var_194 = wp::extract(var_195, var_193);
            printf(var_190, var_194);
            // idx = -1                                                                       <L 1282>
            // break                                                                          <L 1283>
            wp::assign(var_1, var_148);
            wp::assign(var_2, var_149);
            wp::assign(var_5, var_197);
            wp::assign(var_8, var_151);
            wp::assign(var_12, var_154);
            goto end_for_0;
        }
        var_198 = wp::where(var_188, var_1, var_148);
        var_199 = wp::where(var_188, var_2, var_149);
        var_200 = wp::where(var_188, var_5, var_150);
        var_201 = wp::where(var_188, var_8, var_151);
        var_202 = wp::where(var_188, var_12, var_154);
        // for i in range(pt.nface):                                                          <L 1286>
        var_203 = &(var_pt.nface);
        var_205 = wp::load(var_203);
        var_204 = wp::range(var_205);
        start_for_6:;
            if (iter_cmp(var_204) == 0) goto end_for_6;
            var_206 = wp::iter_next(var_204);
            // if _is_face_deleted(pt.face[i]):                                               <L 1287>
            var_207 = &(var_pt.face);
            var_209 = wp::load(var_207);
            var_208 = wp::address(var_209, var_206);
            var_211 = wp::load(var_208);
            var_210 = _is_face_deleted_0(var_211);
            if (var_210) {
                // continue                                                                   <L 1288>
                goto start_for_6;
            }
            // if wp.dot(pt.face_pr[i], w) - pt.face_norm2[i] > 1e-10:                        <L 1290>
            var_212 = &(var_pt.face_pr);
            var_214 = wp::load(var_212);
            var_213 = wp::address(var_214, var_206);
            var_216 = wp::load(var_213);
            var_215 = wp::dot(var_216, var_81);
            var_217 = &(var_pt.face_norm2);
            var_219 = wp::load(var_217);
            var_218 = wp::address(var_219, var_206);
            var_221 = wp::load(var_218);
            var_220 = wp::sub(var_215, var_221);
            var_223 = (var_220 > var_222);
            if (var_223) {
                // nvalid = wp.where(_is_invalid_face(pt.face[i]), nvalid, nvalid - 1)        <L 1291>
                var_224 = &(var_pt.face);
                var_226 = wp::load(var_224);
                var_225 = wp::address(var_226, var_206);
                var_228 = wp::load(var_225);
                var_227 = _is_invalid_face_0(var_228);
                var_230 = wp::sub(var_202, var_229);
                var_231 = wp::where(var_227, var_202, var_230);
                // pt.face[i] = _delete_face(pt.face[i])                                      <L 1292>
                var_232 = &(var_pt.face);
                var_234 = wp::load(var_232);
                var_233 = wp::address(var_234, var_206);
                var_236 = wp::load(var_233);
                var_235 = _delete_face_0(var_236);
                var_237 = &(var_pt.face);
                var_238 = wp::load(var_237);
                wp::array_store(var_238, var_206, var_235);
                // face = _get_face_verts(pt.face[i])                                         <L 1293>
                var_239 = &(var_pt.face);
                var_241 = wp::load(var_239);
                var_240 = wp::address(var_241, var_206);
                var_243 = wp::load(var_240);
                var_242 = _get_face_verts_0(var_243);
                // pt.nhorizon = _add_edge(pt, face[0], face[1])                              <L 1294>
                var_245 = wp::extract(var_242, var_244);
                var_247 = wp::extract(var_242, var_246);
                var_248 = _add_edge_0(var_pt, var_245, var_247);
                var_249 = &(var_pt.nhorizon);
                wp::store(var_249, var_248);
                // pt.nhorizon = _add_edge(pt, face[1], face[2])                              <L 1295>
                var_251 = wp::extract(var_242, var_250);
                var_253 = wp::extract(var_242, var_252);
                var_254 = _add_edge_0(var_pt, var_251, var_253);
                var_255 = &(var_pt.nhorizon);
                wp::store(var_255, var_254);
                // pt.nhorizon = _add_edge(pt, face[2], face[0])                              <L 1296>
                var_257 = wp::extract(var_242, var_256);
                var_259 = wp::extract(var_242, var_258);
                var_260 = _add_edge_0(var_pt, var_257, var_259);
                var_261 = &(var_pt.nhorizon);
                wp::store(var_261, var_260);
                // if pt.nhorizon == -1:                                                      <L 1297>
                var_262 = &(var_pt.nhorizon);
                var_266 = wp::load(var_262);
                var_265 = (var_266 == var_264);
                if (var_265) {
                    // wp.printf("Warning: EPA horizon = %d isn't large enough.\n", pt.horizon.shape[0])       <L 1298>
                    var_268 = &(var_pt.horizon);
                    var_269 = &(var_268->shape);
                    var_272 = wp::load(var_269);
                    var_271 = wp::extract(var_272, var_270);
                    printf(var_267, var_271);
                    // idx = -1                                                               <L 1299>
                    // break                                                                  <L 1300>
                    wp::assign(var_200, var_274);
                    wp::assign(var_202, var_231);
                    wp::assign(var_165, var_242);
                    goto end_for_6;
                }
                var_275 = wp::where(var_265, var_202, var_231);
                var_276 = wp::where(var_265, var_165, var_242);
            }
            var_277 = wp::where(var_223, var_275, var_202);
            var_278 = wp::where(var_223, var_276, var_165);
            wp::assign(var_202, var_277);
            wp::assign(var_165, var_278);
            goto start_for_6;
        end_for_6:;
        // for i in range(pt.nhorizon):                                                       <L 1303>
        var_279 = &(var_pt.nhorizon);
        var_281 = wp::load(var_279);
        var_280 = wp::range(var_281);
        start_for_8:;
            if (iter_cmp(var_280) == 0) goto end_for_8;
            var_282 = wp::iter_next(var_280);
            // edge = _get_edge(pt.horizon[i])                                                <L 1304>
            var_283 = &(var_pt.horizon);
            var_285 = wp::load(var_283);
            var_284 = wp::address(var_285, var_282);
            var_287 = wp::load(var_284);
            var_286 = _get_edge_0(var_287);
            // dist2 = _attach_face(pt, pt.nface, wi, edge[0], edge[1])                       <L 1305>
            var_288 = &(var_pt.nface);
            var_290 = wp::extract(var_286, var_289);
            var_292 = wp::extract(var_286, var_291);
            var_294 = wp::load(var_288);
            var_293 = _attach_face_0(var_pt, var_294, var_60, var_290, var_292);
            // if dist2 == 0.0:                                                               <L 1306>
            var_296 = (var_293 == var_295);
            if (var_296) {
                // idx = -1                                                                   <L 1307>
                // break                                                                      <L 1308>
                wp::assign(var_200, var_298);
                goto end_for_8;
            }
            // pt.nface += 1                                                                  <L 1310>
            var_300 = &(var_pt.nface);
            var_303 = wp::load(var_300);
            var_302 = wp::add(var_303, var_301);
            var_304 = &(var_pt.nface);
            wp::store(var_304, var_302);
            // if dist2 >= lower2 and dist2 <= upper2:                                        <L 1312>
            var_305 = (var_293 >= var_22);
            var_306 = (var_293 <= var_199);
            var_307 = var_305 && var_306;
            if (var_307) {
                // nvalid += 1                                                                <L 1313>
                var_309 = wp::add(var_202, var_308);
            }
            var_310 = wp::where(var_307, var_309, var_202);
            if (!var_307) {
                // pt.face[pt.nface - 1] = _invalidate_face(pt.face[pt.nface - 1])            <L 1315>
                var_311 = &(var_pt.face);
                var_312 = &(var_pt.nface);
                var_315 = wp::load(var_312);
                var_314 = wp::sub(var_315, var_313);
                var_317 = wp::load(var_311);
                var_316 = wp::address(var_317, var_314);
                var_319 = wp::load(var_316);
                var_318 = _invalidate_face_0(var_319);
                var_320 = &(var_pt.face);
                var_321 = &(var_pt.nface);
                var_324 = wp::load(var_321);
                var_323 = wp::sub(var_324, var_322);
                var_325 = wp::load(var_320);
                wp::array_store(var_325, var_323, var_318);
            }
            wp::assign(var_202, var_310);
            goto start_for_8;
        end_for_8:;
        // if nvalid == 0 or idx == -1:                                                       <L 1318>
        var_327 = (var_202 == var_326);
        var_330 = (var_200 == var_329);
        var_331 = var_327 || var_330;
        if (var_331) {
            // break                                                                          <L 1319>
            wp::assign(var_1, var_198);
            wp::assign(var_2, var_199);
            wp::assign(var_5, var_200);
            wp::assign(var_8, var_201);
            wp::assign(var_12, var_202);
            goto end_for_0;
        }
        var_332 = wp::where(var_331, var_1, var_198);
        var_333 = wp::where(var_331, var_2, var_199);
        var_334 = wp::where(var_331, var_5, var_200);
        var_335 = wp::where(var_331, var_8, var_201);
        var_336 = wp::where(var_331, var_12, var_202);
        // pt.nhorizon = 0                                                                    <L 1322>
        var_338 = &(var_pt.nhorizon);
        wp::store(var_338, var_337);
        wp::assign(var_1, var_332);
        wp::assign(var_2, var_333);
        wp::assign(var_5, var_334);
        wp::assign(var_8, var_335);
        wp::assign(var_12, var_336);
        goto start_for_0;
    end_for_0:;
    // if idx > -1:                                                                           <L 1325>
    var_341 = (var_5 > var_340);
    if (var_341) {
        // x1, x2, dist = _epa_witness(pt, geom1, geom2, geomtype1, geomtype2, idx)           <L 1326>
        _epa_witness_0(var_pt, var_geom1, var_geom2, var_geomtype1, var_geomtype2, var_5, var_342, var_343, var_344);
        // return dist, x1, x2, idx                                                           <L 1327>
        ret_0 = var_344;
        ret_1 = var_342;
        ret_2 = var_343;
        ret_3 = var_5;
        return;
    }
    // return 0.0, wp.vec3(), wp.vec3(), -1                                                   <L 1328>
    var_346 = wp::vec_t<3, wp::float32>();
    var_347 = wp::vec_t<3, wp::float32>();
    ret_0 = var_345;
    ret_1 = var_346;
    ret_2 = var_347;
    ret_3 = var_349;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:2199
static CUDA_CALLABLE void ccd_0(
    wp::float32 var_tolerance,
    wp::float32 var_cutoff,
    wp::int32 var_gjk_iterations,
    wp::int32 var_epa_iterations,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2,
    wp::vec_t<3, wp::float32> var_x_1,
    wp::vec_t<3, wp::float32> var_x_2,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vert,
    wp::array_t<wp::int32> var_vert_index,
    wp::array_t<wp::int32> var_face,
    wp::array_t<wp::vec_t<3, wp::float32>> var_face_pr,
    wp::array_t<wp::float32> var_face_norm2,
    wp::array_t<wp::int32> var_horizon,
    wp::float32 & ret_0,
    wp::int32 & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & ret_3,
    wp::int32 & ret_4)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    const wp::float32 var_1 = 0.0;
    const wp::float32 var_2 = 0.0;
    const wp::float32 var_3 = 0.0;
    bool var_4;
    wp::float32* var_5;
    const wp::float32 var_6 = 0.0;
    bool var_7;
    wp::float32 var_8;
    wp::float32* var_9;
    const wp::float32 var_10 = 0.0;
    bool var_11;
    wp::float32 var_12;
    bool var_13;
    bool var_14;
    const wp::int32 var_15 = 2;
    bool var_16;
    const wp::int32 var_17 = 3;
    bool var_18;
    bool var_19;
    wp::vec_t<3, wp::float32>* var_20;
    const wp::int32 var_21 = 0;
    wp::float32 var_22;
    wp::vec_t<3, wp::float32> var_23;
    const wp::float32 var_24 = 0.5;
    wp::float32* var_25;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    const wp::float32 var_29 = 0.0;
    wp::float32* var_30;
    const wp::float32 var_31 = 0.0;
    wp::vec_t<3, wp::float32>* var_32;
    const wp::int32 var_33 = 1;
    wp::float32 var_34;
    wp::vec_t<3, wp::float32> var_35;
    wp::vec_t<3, wp::float32>* var_36;
    const wp::int32 var_37 = 2;
    wp::float32 var_38;
    wp::vec_t<3, wp::float32> var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::vec_t<3, wp::float32>* var_41;
    wp::float32 var_42;
    wp::float32 var_43;
    const wp::int32 var_44 = 2;
    bool var_45;
    const wp::int32 var_46 = 3;
    bool var_47;
    bool var_48;
    wp::vec_t<3, wp::float32>* var_49;
    const wp::int32 var_50 = 0;
    wp::float32 var_51;
    wp::vec_t<3, wp::float32> var_52;
    const wp::float32 var_53 = 0.5;
    wp::float32* var_54;
    wp::float32 var_55;
    wp::float32 var_56;
    wp::float32 var_57;
    const wp::float32 var_58 = 0.0;
    wp::float32* var_59;
    const wp::float32 var_60 = 0.0;
    wp::vec_t<3, wp::float32>* var_61;
    const wp::int32 var_62 = 1;
    wp::float32 var_63;
    wp::vec_t<3, wp::float32> var_64;
    wp::vec_t<3, wp::float32>* var_65;
    const wp::int32 var_66 = 2;
    wp::float32 var_67;
    wp::vec_t<3, wp::float32> var_68;
    wp::vec_t<3, wp::float32> var_69;
    wp::vec_t<3, wp::float32>* var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    wp::float32 var_73;
    const wp::float32 var_74 = 0.0;
    bool var_75;
    wp::float32 var_76;
    wp::float32 var_77;
    GJKResult_0220ee01 var_78;
    wp::float32* var_79;
    bool var_80;
    wp::float32 var_81;
    wp::float32* var_82;
    const wp::float32 var_83 = 1e+30;
    bool var_84;
    wp::float32 var_85;
    wp::float32* var_86;
    const wp::int32 var_87 = 1;
    wp::vec_t<3, wp::float32>* var_88;
    wp::vec_t<3, wp::float32>* var_89;
    const wp::int32 var_90 = 1;
    const wp::int32 var_91 = -1;
    wp::float32 var_92;
    wp::float32 var_93;
    wp::vec_t<3, wp::float32> var_94;
    wp::vec_t<3, wp::float32> var_95;
    wp::vec_t<3, wp::float32> var_96;
    wp::vec_t<3, wp::float32> var_97;
    wp::float32 var_98;
    wp::vec_t<3, wp::float32> var_99;
    wp::vec_t<3, wp::float32> var_100;
    const wp::int32 var_101 = 1;
    const wp::int32 var_102 = 1;
    const wp::int32 var_103 = -1;
    wp::float32 var_104;
    wp::float32* var_105;
    wp::vec_t<3, wp::float32>* var_106;
    const wp::int32 var_107 = 1;
    wp::float32 var_108;
    wp::vec_t<3, wp::float32> var_109;
    wp::vec_t<3, wp::float32>* var_110;
    const wp::int32 var_111 = 2;
    wp::float32 var_112;
    wp::vec_t<3, wp::float32> var_113;
    wp::vec_t<3, wp::float32> var_114;
    wp::vec_t<3, wp::float32>* var_115;
    wp::float32 var_116;
    wp::float32* var_117;
    wp::vec_t<3, wp::float32>* var_118;
    const wp::int32 var_119 = 1;
    wp::float32 var_120;
    wp::vec_t<3, wp::float32> var_121;
    wp::vec_t<3, wp::float32>* var_122;
    const wp::int32 var_123 = 2;
    wp::float32 var_124;
    wp::vec_t<3, wp::float32> var_125;
    wp::vec_t<3, wp::float32> var_126;
    wp::vec_t<3, wp::float32>* var_127;
    wp::float32 var_128;
    wp::float32 var_129;
    wp::float32 var_130;
    GJKResult_0220ee01 var_131;
    wp::float32* var_132;
    bool var_133;
    wp::float32 var_134;
    wp::int32* var_135;
    const wp::int32 var_136 = 2;
    bool var_137;
    wp::int32 var_138;
    bool var_139;
    wp::float32* var_140;
    const wp::int32 var_141 = 1;
    wp::vec_t<3, wp::float32>* var_142;
    wp::vec_t<3, wp::float32>* var_143;
    const wp::int32 var_144 = 1;
    const wp::int32 var_145 = -1;
    wp::float32 var_146;
    wp::float32 var_147;
    wp::vec_t<3, wp::float32> var_148;
    wp::vec_t<3, wp::float32> var_149;
    wp::vec_t<3, wp::float32> var_150;
    wp::vec_t<3, wp::float32> var_151;
    Polytope_9ab93ade var_152;
    const wp::int32 var_153 = 0;
    wp::int32* var_154;
    const wp::int32 var_155 = 0;
    wp::int32* var_156;
    const wp::int32 var_157 = 0;
    wp::int32* var_158;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_159;
    wp::array_t<wp::int32>* var_160;
    wp::array_t<wp::int32>* var_161;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_162;
    wp::array_t<wp::float32>* var_163;
    wp::array_t<wp::int32>* var_164;
    wp::int32* var_165;
    const wp::int32 var_166 = 2;
    bool var_167;
    wp::int32 var_168;
    wp::mat_t<4, 3, wp::float32>* var_169;
    wp::mat_t<4, 3, wp::float32>* var_170;
    wp::mat_t<4, 3, wp::float32>* var_171;
    wp::vec_t<4, wp::int32>* var_172;
    wp::vec_t<4, wp::int32>* var_173;
    Polytope_9ab93ade var_174;
    GJKResult_0220ee01 var_175;
    wp::mat_t<4, 3, wp::float32> var_176;
    wp::mat_t<4, 3, wp::float32> var_177;
    wp::mat_t<4, 3, wp::float32> var_178;
    wp::vec_t<4, wp::int32> var_179;
    wp::vec_t<4, wp::int32> var_180;
    wp::int32* var_181;
    const wp::int32 var_182 = 1;
    const wp::int32 var_183 = -1;
    bool var_184;
    wp::int32 var_185;
    wp::mat_t<4, 3, wp::float32>* var_186;
    wp::mat_t<4, 3, wp::float32>* var_187;
    wp::mat_t<4, 3, wp::float32> var_188;
    wp::mat_t<4, 3, wp::float32>* var_189;
    wp::mat_t<4, 3, wp::float32>* var_190;
    wp::mat_t<4, 3, wp::float32> var_191;
    wp::mat_t<4, 3, wp::float32>* var_192;
    wp::mat_t<4, 3, wp::float32>* var_193;
    wp::mat_t<4, 3, wp::float32> var_194;
    wp::vec_t<4, wp::int32>* var_195;
    wp::vec_t<4, wp::int32>* var_196;
    wp::vec_t<4, wp::int32> var_197;
    wp::vec_t<4, wp::int32>* var_198;
    wp::vec_t<4, wp::int32>* var_199;
    wp::vec_t<4, wp::int32> var_200;
    const wp::int32 var_201 = 3;
    wp::int32* var_202;
    Polytope_9ab93ade var_203;
    wp::int32* var_204;
    const wp::int32 var_205 = 4;
    bool var_206;
    wp::int32 var_207;
    wp::mat_t<4, 3, wp::float32>* var_208;
    wp::mat_t<4, 3, wp::float32>* var_209;
    wp::mat_t<4, 3, wp::float32>* var_210;
    wp::vec_t<4, wp::int32>* var_211;
    wp::vec_t<4, wp::int32>* var_212;
    Polytope_9ab93ade var_213;
    GJKResult_0220ee01 var_214;
    wp::mat_t<4, 3, wp::float32> var_215;
    wp::mat_t<4, 3, wp::float32> var_216;
    wp::mat_t<4, 3, wp::float32> var_217;
    wp::vec_t<4, wp::int32> var_218;
    wp::vec_t<4, wp::int32> var_219;
    wp::int32* var_220;
    const wp::int32 var_221 = 1;
    const wp::int32 var_222 = -1;
    bool var_223;
    wp::int32 var_224;
    wp::mat_t<4, 3, wp::float32>* var_225;
    wp::mat_t<4, 3, wp::float32>* var_226;
    wp::mat_t<4, 3, wp::float32> var_227;
    wp::mat_t<4, 3, wp::float32>* var_228;
    wp::mat_t<4, 3, wp::float32>* var_229;
    wp::mat_t<4, 3, wp::float32> var_230;
    wp::mat_t<4, 3, wp::float32>* var_231;
    wp::mat_t<4, 3, wp::float32>* var_232;
    wp::mat_t<4, 3, wp::float32> var_233;
    wp::vec_t<4, wp::int32>* var_234;
    wp::vec_t<4, wp::int32>* var_235;
    wp::vec_t<4, wp::int32> var_236;
    wp::vec_t<4, wp::int32>* var_237;
    wp::vec_t<4, wp::int32>* var_238;
    wp::vec_t<4, wp::int32> var_239;
    const wp::int32 var_240 = 3;
    wp::int32* var_241;
    Polytope_9ab93ade var_242;
    GJKResult_0220ee01 var_243;
    Polytope_9ab93ade var_244;
    GJKResult_0220ee01 var_245;
    wp::int32* var_246;
    const wp::int32 var_247 = 3;
    bool var_248;
    wp::int32 var_249;
    wp::float32* var_250;
    wp::mat_t<4, 3, wp::float32>* var_251;
    wp::mat_t<4, 3, wp::float32>* var_252;
    wp::mat_t<4, 3, wp::float32>* var_253;
    wp::vec_t<4, wp::int32>* var_254;
    wp::vec_t<4, wp::int32>* var_255;
    Polytope_9ab93ade var_256;
    wp::float32 var_257;
    wp::mat_t<4, 3, wp::float32> var_258;
    wp::mat_t<4, 3, wp::float32> var_259;
    wp::mat_t<4, 3, wp::float32> var_260;
    wp::vec_t<4, wp::int32> var_261;
    wp::vec_t<4, wp::int32> var_262;
    Polytope_9ab93ade var_263;
    wp::int32* var_264;
    wp::int32 var_265;
    wp::float32* var_266;
    const wp::int32 var_267 = 1;
    wp::vec_t<3, wp::float32>* var_268;
    wp::vec_t<3, wp::float32>* var_269;
    const wp::int32 var_270 = 1;
    const wp::int32 var_271 = -1;
    wp::float32 var_272;
    wp::float32 var_273;
    wp::vec_t<3, wp::float32> var_274;
    wp::vec_t<3, wp::float32> var_275;
    wp::vec_t<3, wp::float32> var_276;
    wp::vec_t<3, wp::float32> var_277;
    wp::int32 var_278;
    wp::float32 var_279;
    wp::vec_t<3, wp::float32> var_280;
    wp::vec_t<3, wp::float32> var_281;
    wp::int32 var_282;
    const wp::int32 var_283 = 1;
    const wp::int32 var_284 = -1;
    bool var_285;
    const wp::int32 var_286 = 0;
    wp::vec_t<3, wp::float32> var_287;
    wp::vec_t<3, wp::float32> var_288;
    const wp::int32 var_289 = 1;
    const wp::int32 var_290 = -1;
    wp::float32* var_291;
    const wp::float32 var_292 = 0.0;
    bool var_293;
    wp::float32 var_294;
    wp::float32* var_295;
    const wp::float32 var_296 = 0.0;
    bool var_297;
    wp::float32 var_298;
    bool var_299;
    const wp::int32 var_300 = 1;
    const wp::int32 var_301 = -1;
    wp::int32 var_302;
    const wp::int32 var_303 = 6;
    bool var_304;
    const wp::int32 var_305 = 7;
    bool var_306;
    bool var_307;
    const wp::int32 var_308 = 6;
    bool var_309;
    const wp::int32 var_310 = 7;
    bool var_311;
    bool var_312;
    bool var_313;
    const wp::int32 var_314 = 1;
    const wp::int32 var_315 = -1;
    wp::int32 var_316;
    const wp::int32 var_317 = 1;
    //---------
    // forward
    // def ccd(                                                                               <L 2200>
    // full_margin1 = 0.0                                                                     <L 2220>
    // full_margin2 = 0.0                                                                     <L 2221>
    // size1 = 0.0                                                                            <L 2222>
    // size2 = 0.0                                                                            <L 2223>
    // is_discrete = _discrete_geoms(geomtype1, geomtype2) and (geom1.margin == 0.0 and geom2.margin == 0.0)       <L 2226>
    var_4 = _discrete_geoms_0(var_geomtype1, var_geomtype2);
    var_5 = &(var_geom1.margin);
    var_8 = wp::load(var_5);
    var_7 = (var_8 == var_6);
    var_9 = &(var_geom2.margin);
    var_12 = wp::load(var_9);
    var_11 = (var_12 == var_10);
    var_13 = var_7 && var_11;
    var_14 = var_4 && var_13;
    // if geomtype1 == GeomType.SPHERE or geomtype1 == GeomType.CAPSULE:                      <L 2228>
    var_16 = (var_geomtype1 == var_15);
    var_18 = (var_geomtype1 == var_17);
    var_19 = var_16 || var_18;
    if (var_19) {
        // size1 = geom1.size[0]                                                              <L 2229>
        var_20 = &(var_geom1.size);
        var_23 = wp::load(var_20);
        var_22 = wp::extract(var_23, var_21);
        // full_margin1 = size1 + 0.5 * geom1.margin                                          <L 2230>
        var_25 = &(var_geom1.margin);
        var_27 = wp::load(var_25);
        var_26 = wp::mul(var_24, var_27);
        var_28 = wp::add(var_22, var_26);
        // geom1.margin = 0.0                                                                 <L 2231>
        var_30 = &(var_geom1.margin);
        wp::store(var_30, var_29);
        // geom1.size = wp.vec3(0.0, geom1.size[1], geom1.size[2])                            <L 2232>
        var_32 = &(var_geom1.size);
        var_35 = wp::load(var_32);
        var_34 = wp::extract(var_35, var_33);
        var_36 = &(var_geom1.size);
        var_39 = wp::load(var_36);
        var_38 = wp::extract(var_39, var_37);
        var_40 = wp::vec_t<3, wp::float32>(var_31, var_34, var_38);
        var_41 = &(var_geom1.size);
        wp::store(var_41, var_40);
    }
    var_42 = wp::where(var_19, var_28, var_0);
    var_43 = wp::where(var_19, var_22, var_2);
    // if geomtype2 == GeomType.SPHERE or geomtype2 == GeomType.CAPSULE:                      <L 2234>
    var_45 = (var_geomtype2 == var_44);
    var_47 = (var_geomtype2 == var_46);
    var_48 = var_45 || var_47;
    if (var_48) {
        // size2 = geom2.size[0]                                                              <L 2235>
        var_49 = &(var_geom2.size);
        var_52 = wp::load(var_49);
        var_51 = wp::extract(var_52, var_50);
        // full_margin2 = size2 + 0.5 * geom2.margin                                          <L 2236>
        var_54 = &(var_geom2.margin);
        var_56 = wp::load(var_54);
        var_55 = wp::mul(var_53, var_56);
        var_57 = wp::add(var_51, var_55);
        // geom2.margin = 0.0                                                                 <L 2237>
        var_59 = &(var_geom2.margin);
        wp::store(var_59, var_58);
        // geom2.size = wp.vec3(0.0, geom2.size[1], geom2.size[2])                            <L 2238>
        var_61 = &(var_geom2.size);
        var_64 = wp::load(var_61);
        var_63 = wp::extract(var_64, var_62);
        var_65 = &(var_geom2.size);
        var_68 = wp::load(var_65);
        var_67 = wp::extract(var_68, var_66);
        var_69 = wp::vec_t<3, wp::float32>(var_60, var_63, var_67);
        var_70 = &(var_geom2.size);
        wp::store(var_70, var_69);
    }
    var_71 = wp::where(var_48, var_57, var_1);
    var_72 = wp::where(var_48, var_51, var_3);
    // if size1 + size2 > 0.0:                                                                <L 2241>
    var_73 = wp::add(var_43, var_72);
    var_75 = (var_73 > var_74);
    if (var_75) {
        // cutoff += full_margin1 + full_margin2                                              <L 2242>
        var_76 = wp::add(var_42, var_71);
        var_77 = wp::add(var_cutoff, var_76);
        // result = gjk(tolerance, gjk_iterations, geom1, geom2, x_1, x_2, geomtype1, geomtype2, cutoff, is_discrete)       <L 2243>
        var_78 = gjk_0(var_tolerance, var_gjk_iterations, var_geom1, var_geom2, var_x_1, var_x_2, var_geomtype1, var_geomtype2, var_77, var_14);
        // if result.dist > tolerance:                                                        <L 2246>
        var_79 = &(var_78.dist);
        var_81 = wp::load(var_79);
        var_80 = (var_81 > var_tolerance);
        if (var_80) {
            // if result.dist == FLOAT_MAX:                                                   <L 2247>
            var_82 = &(var_78.dist);
            var_85 = wp::load(var_82);
            var_84 = (var_85 == var_83);
            if (var_84) {
                // return result.dist, 1, result.x1, result.x2, -1                            <L 2248>
                var_86 = &(var_78.dist);
                var_88 = &(var_78.x1);
                var_89 = &(var_78.x2);
                var_93 = wp::load(var_86);
                var_92 = wp::copy(var_93);
                var_95 = wp::load(var_88);
                var_94 = wp::copy(var_95);
                var_97 = wp::load(var_89);
                var_96 = wp::copy(var_97);
                ret_0 = var_92;
                ret_1 = var_87;
                ret_2 = var_94;
                ret_3 = var_96;
                ret_4 = var_91;
                return;
            }
            // dist, x1, x2 = _inflate(result, geom1, geom2, geomtype1, geomtype2, full_margin1, full_margin2)       <L 2249>
            _inflate_0(var_78, var_geom1, var_geom2, var_geomtype1, var_geomtype2, var_42, var_71, var_98, var_99, var_100);
            // return dist, 1, x1, x2, -1                                                     <L 2250>
            ret_0 = var_98;
            ret_1 = var_101;
            ret_2 = var_99;
            ret_3 = var_100;
            ret_4 = var_103;
            return;
        }
        // geom1.margin = full_margin1 - size1                                                <L 2253>
        var_104 = wp::sub(var_42, var_43);
        var_105 = &(var_geom1.margin);
        wp::store(var_105, var_104);
        // geom1.size = wp.vec3(size1, geom1.size[1], geom1.size[2])                          <L 2254>
        var_106 = &(var_geom1.size);
        var_109 = wp::load(var_106);
        var_108 = wp::extract(var_109, var_107);
        var_110 = &(var_geom1.size);
        var_113 = wp::load(var_110);
        var_112 = wp::extract(var_113, var_111);
        var_114 = wp::vec_t<3, wp::float32>(var_43, var_108, var_112);
        var_115 = &(var_geom1.size);
        wp::store(var_115, var_114);
        // geom2.margin = full_margin2 - size2                                                <L 2255>
        var_116 = wp::sub(var_71, var_72);
        var_117 = &(var_geom2.margin);
        wp::store(var_117, var_116);
        // geom2.size = wp.vec3(size2, geom2.size[1], geom2.size[2])                          <L 2256>
        var_118 = &(var_geom2.size);
        var_121 = wp::load(var_118);
        var_120 = wp::extract(var_121, var_119);
        var_122 = &(var_geom2.size);
        var_125 = wp::load(var_122);
        var_124 = wp::extract(var_125, var_123);
        var_126 = wp::vec_t<3, wp::float32>(var_72, var_120, var_124);
        var_127 = &(var_geom2.size);
        wp::store(var_127, var_126);
        // cutoff -= full_margin1 + full_margin2                                              <L 2257>
        var_128 = wp::add(var_42, var_71);
        var_129 = wp::sub(var_77, var_128);
    }
    var_130 = wp::where(var_75, var_129, var_cutoff);
    // result = gjk(tolerance, gjk_iterations, geom1, geom2, x_1, x_2, geomtype1, geomtype2, cutoff, is_discrete)       <L 2259>
    var_131 = gjk_0(var_tolerance, var_gjk_iterations, var_geom1, var_geom2, var_x_1, var_x_2, var_geomtype1, var_geomtype2, var_130, var_14);
    // if result.dist > tolerance or result.dim < 2:                                          <L 2262>
    var_132 = &(var_131.dist);
    var_134 = wp::load(var_132);
    var_133 = (var_134 > var_tolerance);
    var_135 = &(var_131.dim);
    var_138 = wp::load(var_135);
    var_137 = (var_138 < var_136);
    var_139 = var_133 || var_137;
    if (var_139) {
        // return result.dist, 1, result.x1, result.x2, -1                                    <L 2263>
        var_140 = &(var_131.dist);
        var_142 = &(var_131.x1);
        var_143 = &(var_131.x2);
        var_147 = wp::load(var_140);
        var_146 = wp::copy(var_147);
        var_149 = wp::load(var_142);
        var_148 = wp::copy(var_149);
        var_151 = wp::load(var_143);
        var_150 = wp::copy(var_151);
        ret_0 = var_146;
        ret_1 = var_141;
        ret_2 = var_148;
        ret_3 = var_150;
        ret_4 = var_145;
        return;
    }
    // pt = Polytope()                                                                        <L 2265>
    var_152 = Polytope_9ab93ade();
    // pt.nface = 0                                                                           <L 2266>
    var_154 = &(var_152.nface);
    wp::store(var_154, var_153);
    // pt.nvert = 0                                                                           <L 2267>
    var_156 = &(var_152.nvert);
    wp::store(var_156, var_155);
    // pt.nhorizon = 0                                                                        <L 2268>
    var_158 = &(var_152.nhorizon);
    wp::store(var_158, var_157);
    // pt.vert = vert                                                                         <L 2269>
    var_159 = &(var_152.vert);
    wp::store(var_159, var_vert);
    // pt.vert_index = vert_index                                                             <L 2270>
    var_160 = &(var_152.vert_index);
    wp::store(var_160, var_vert_index);
    // pt.face = face                                                                         <L 2271>
    var_161 = &(var_152.face);
    wp::store(var_161, var_face);
    // pt.face_pr = face_pr                                                                   <L 2272>
    var_162 = &(var_152.face_pr);
    wp::store(var_162, var_face_pr);
    // pt.face_norm2 = face_norm2                                                             <L 2273>
    var_163 = &(var_152.face_norm2);
    wp::store(var_163, var_face_norm2);
    // pt.horizon = horizon                                                                   <L 2274>
    var_164 = &(var_152.horizon);
    wp::store(var_164, var_horizon);
    // if result.dim == 2:                                                                    <L 2276>
    var_165 = &(var_131.dim);
    var_168 = wp::load(var_165);
    var_167 = (var_168 == var_166);
    if (var_167) {
        // pt, new_result = _polytope2(                                                       <L 2277>
        // pt,                                                                                <L 2278>
        // result.simplex,                                                                    <L 2279>
        var_169 = &(var_131.simplex);
        // result.simplex1,                                                                   <L 2280>
        var_170 = &(var_131.simplex1);
        // result.simplex2,                                                                   <L 2281>
        var_171 = &(var_131.simplex2);
        // result.simplex_index1,                                                             <L 2282>
        var_172 = &(var_131.simplex_index1);
        // result.simplex_index2,                                                             <L 2283>
        var_173 = &(var_131.simplex_index2);
        // geom1,                                                                             <L 2284>
        // geom2,                                                                             <L 2285>
        // geomtype1,                                                                         <L 2286>
        // geomtype2,                                                                         <L 2287>
        var_176 = wp::load(var_169);
        var_177 = wp::load(var_170);
        var_178 = wp::load(var_171);
        var_179 = wp::load(var_172);
        var_180 = wp::load(var_173);
        _polytope2_0(var_152, var_176, var_177, var_178, var_179, var_180, var_geom1, var_geom2, var_geomtype1, var_geomtype2, var_174, var_175);
        // if pt.status == -1:                                                                <L 2289>
        var_181 = &(var_174.status);
        var_185 = wp::load(var_181);
        var_184 = (var_185 == var_183);
        if (var_184) {
            // result.simplex = new_result.simplex                                            <L 2290>
            var_186 = &(var_175.simplex);
            var_187 = &(var_131.simplex);
            var_188 = wp::load(var_186);
            wp::store(var_187, var_188);
            // result.simplex1 = new_result.simplex1                                          <L 2291>
            var_189 = &(var_175.simplex1);
            var_190 = &(var_131.simplex1);
            var_191 = wp::load(var_189);
            wp::store(var_190, var_191);
            // result.simplex2 = new_result.simplex2                                          <L 2292>
            var_192 = &(var_175.simplex2);
            var_193 = &(var_131.simplex2);
            var_194 = wp::load(var_192);
            wp::store(var_193, var_194);
            // result.simplex_index1 = new_result.simplex_index1                              <L 2293>
            var_195 = &(var_175.simplex_index1);
            var_196 = &(var_131.simplex_index1);
            var_197 = wp::load(var_195);
            wp::store(var_196, var_197);
            // result.simplex_index2 = new_result.simplex_index2                              <L 2294>
            var_198 = &(var_175.simplex_index2);
            var_199 = &(var_131.simplex_index2);
            var_200 = wp::load(var_198);
            wp::store(var_199, var_200);
            // result.dim = 3                                                                 <L 2295>
            var_202 = &(var_131.dim);
            wp::store(var_202, var_201);
        }
    }
    var_203 = wp::where(var_167, var_174, var_152);
    if (!var_167) {
        // elif result.dim == 4:                                                              <L 2296>
        var_204 = &(var_131.dim);
        var_207 = wp::load(var_204);
        var_206 = (var_207 == var_205);
        if (var_206) {
            // pt, new_result = _polytope4(                                                   <L 2297>
            // pt,                                                                            <L 2298>
            // result.simplex,                                                                <L 2299>
            var_208 = &(var_131.simplex);
            // result.simplex1,                                                               <L 2300>
            var_209 = &(var_131.simplex1);
            // result.simplex2,                                                               <L 2301>
            var_210 = &(var_131.simplex2);
            // result.simplex_index1,                                                         <L 2302>
            var_211 = &(var_131.simplex_index1);
            // result.simplex_index2,                                                         <L 2303>
            var_212 = &(var_131.simplex_index2);
            var_215 = wp::load(var_208);
            var_216 = wp::load(var_209);
            var_217 = wp::load(var_210);
            var_218 = wp::load(var_211);
            var_219 = wp::load(var_212);
            _polytope4_0(var_203, var_215, var_216, var_217, var_218, var_219, var_213, var_214);
            // if pt.status == -1:                                                            <L 2305>
            var_220 = &(var_213.status);
            var_224 = wp::load(var_220);
            var_223 = (var_224 == var_222);
            if (var_223) {
                // result.simplex = new_result.simplex                                        <L 2306>
                var_225 = &(var_214.simplex);
                var_226 = &(var_131.simplex);
                var_227 = wp::load(var_225);
                wp::store(var_226, var_227);
                // result.simplex1 = new_result.simplex1                                      <L 2307>
                var_228 = &(var_214.simplex1);
                var_229 = &(var_131.simplex1);
                var_230 = wp::load(var_228);
                wp::store(var_229, var_230);
                // result.simplex2 = new_result.simplex2                                      <L 2308>
                var_231 = &(var_214.simplex2);
                var_232 = &(var_131.simplex2);
                var_233 = wp::load(var_231);
                wp::store(var_232, var_233);
                // result.simplex_index1 = new_result.simplex_index1                          <L 2309>
                var_234 = &(var_214.simplex_index1);
                var_235 = &(var_131.simplex_index1);
                var_236 = wp::load(var_234);
                wp::store(var_235, var_236);
                // result.simplex_index2 = new_result.simplex_index2                          <L 2310>
                var_237 = &(var_214.simplex_index2);
                var_238 = &(var_131.simplex_index2);
                var_239 = wp::load(var_237);
                wp::store(var_238, var_239);
                // result.dim = 3                                                             <L 2311>
                var_241 = &(var_131.dim);
                wp::store(var_241, var_240);
            }
        }
        var_242 = wp::where(var_206, var_213, var_203);
        var_243 = wp::where(var_206, var_214, var_175);
    }
    var_244 = wp::where(var_167, var_203, var_242);
    var_245 = wp::where(var_167, var_175, var_243);
    // if result.dim == 3:                                                                    <L 2314>
    var_246 = &(var_131.dim);
    var_249 = wp::load(var_246);
    var_248 = (var_249 == var_247);
    if (var_248) {
        // pt = _polytope3(                                                                   <L 2315>
        // pt,                                                                                <L 2316>
        // result.dist,                                                                       <L 2317>
        var_250 = &(var_131.dist);
        // result.simplex,                                                                    <L 2318>
        var_251 = &(var_131.simplex);
        // result.simplex1,                                                                   <L 2319>
        var_252 = &(var_131.simplex1);
        // result.simplex2,                                                                   <L 2320>
        var_253 = &(var_131.simplex2);
        // result.simplex_index1,                                                             <L 2321>
        var_254 = &(var_131.simplex_index1);
        // result.simplex_index2,                                                             <L 2322>
        var_255 = &(var_131.simplex_index2);
        // geom1,                                                                             <L 2323>
        // geom2,                                                                             <L 2324>
        // geomtype1,                                                                         <L 2325>
        // geomtype2,                                                                         <L 2326>
        var_257 = wp::load(var_250);
        var_258 = wp::load(var_251);
        var_259 = wp::load(var_252);
        var_260 = wp::load(var_253);
        var_261 = wp::load(var_254);
        var_262 = wp::load(var_255);
        var_256 = _polytope3_0(var_244, var_257, var_258, var_259, var_260, var_261, var_262, var_geom1, var_geom2, var_geomtype1, var_geomtype2);
    }
    var_263 = wp::where(var_248, var_256, var_244);
    // if pt.status:                                                                          <L 2330>
    var_264 = &(var_263.status);
    var_265 = wp::load(var_264);
    if (var_265) {
        // return result.dist, 1, result.x1, result.x2, -1                                    <L 2331>
        var_266 = &(var_131.dist);
        var_268 = &(var_131.x1);
        var_269 = &(var_131.x2);
        var_273 = wp::load(var_266);
        var_272 = wp::copy(var_273);
        var_275 = wp::load(var_268);
        var_274 = wp::copy(var_275);
        var_277 = wp::load(var_269);
        var_276 = wp::copy(var_277);
        ret_0 = var_272;
        ret_1 = var_267;
        ret_2 = var_274;
        ret_3 = var_276;
        ret_4 = var_271;
        return;
    }
    var_278 = wp::load(var_264);
    // dist, x1, x2, idx = _epa(tolerance, epa_iterations, pt, geom1, geom2, geomtype1, geomtype2, is_discrete)       <L 2333>
    _epa_0(var_tolerance, var_epa_iterations, var_263, var_geom1, var_geom2, var_geomtype1, var_geomtype2, var_14, var_279, var_280, var_281, var_282);
    // if idx == -1:                                                                          <L 2334>
    var_285 = (var_282 == var_284);
    if (var_285) {
        // return FLOAT_MAX, 0, wp.vec3(), wp.vec3(), -1                                      <L 2335>
        var_287 = wp::vec_t<3, wp::float32>();
        var_288 = wp::vec_t<3, wp::float32>();
        ret_0 = var_83;
        ret_1 = var_286;
        ret_2 = var_287;
        ret_3 = var_288;
        ret_4 = var_290;
        return;
    }
    // if geom1.margin != 0.0 or geom2.margin != 0.0:                                         <L 2338>
    var_291 = &(var_geom1.margin);
    var_294 = wp::load(var_291);
    var_293 = (var_294 != var_292);
    var_295 = &(var_geom2.margin);
    var_298 = wp::load(var_295);
    var_297 = (var_298 != var_296);
    var_299 = var_293 || var_297;
    if (var_299) {
        // idx = -1                                                                           <L 2339>
    }
    var_302 = wp::where(var_299, var_301, var_282);
    // if (geomtype1 != GeomType.BOX and geomtype1 != GeomType.MESH) or (geomtype2 != GeomType.BOX and geomtype2 != GeomType.MESH):       <L 2342>
    var_304 = (var_geomtype1 != var_303);
    var_306 = (var_geomtype1 != var_305);
    var_307 = var_304 && var_306;
    var_309 = (var_geomtype2 != var_308);
    var_311 = (var_geomtype2 != var_310);
    var_312 = var_309 && var_311;
    var_313 = var_307 || var_312;
    if (var_313) {
        // idx = -1                                                                           <L 2343>
    }
    var_316 = wp::where(var_313, var_315, var_302);
    // return dist, 1, x1, x2, idx                                                            <L 2345>
    ret_0 = var_279;
    ret_1 = var_317;
    ret_2 = var_280;
    ret_3 = var_281;
    ret_4 = var_316;
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_convex.py:711
static CUDA_CALLABLE wp::int32 ccd_kernel_builder__locals__eval_ccd_write_contact_1(
    wp::array_t<wp::float32> var_opt_ccd_tolerance,
    wp::int32 var_naconmax_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_epa_vert_in,
    wp::array_t<wp::int32> var_epa_vert_index_in,
    wp::array_t<wp::int32> var_epa_face_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_epa_pr_in,
    wp::array_t<wp::float32> var_epa_norm2_in,
    wp::array_t<wp::int32> var_epa_horizon_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_polygon_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_clipped_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_pnormal_in,
    wp::array_t<wp::float32> var_multiccd_pdist_in,
    wp::array_t<wp::int32> var_multiccd_idx1_in,
    wp::array_t<wp::int32> var_multiccd_idx2_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_n1_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_n2_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_endvert_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_face1_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_face2_in,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::int32 var_worldid,
    wp::int32 var_ccdid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::vec_t<3, wp::float32> var_x1,
    wp::vec_t<3, wp::float32> var_x2,
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
    wp::mat_t<4, 3, wp::float32> var_0;
    wp::mat_t<4, 3, wp::float32> var_1;
    wp::mat_t<4, 3, wp::float32> var_2;
    wp::float32* var_3;
    wp::float32* var_4;
    const wp::int32 var_5 = 1;
    wp::int32 var_6;
    const wp::int32 var_7 = 0;
    bool var_8;
    const wp::float32 var_9 = 1e+32;
    const wp::float32 var_10 = 0.0;
    wp::float32 var_11;
    wp::shape_t* var_12;
    const wp::int32 var_13 = 0;
    wp::int32 var_14;
    wp::shape_t var_15;
    wp::int32 var_16;
    wp::float32* var_17;
    const wp::int32 var_18 = 50;
    const wp::int32 var_19 = 50;
    const wp::int32 var_20 = 5;
    const wp::int32 var_21 = 5;
    wp::slice_t var_22;
    const wp::int32 var_23 = 0;
    wp::array_t<wp::vec_t<3, wp::float32>> var_24;
    wp::slice_t var_25;
    const wp::int32 var_26 = 0;
    wp::array_t<wp::int32> var_27;
    wp::slice_t var_28;
    const wp::int32 var_29 = 0;
    wp::array_t<wp::int32> var_30;
    wp::slice_t var_31;
    const wp::int32 var_32 = 0;
    wp::array_t<wp::vec_t<3, wp::float32>> var_33;
    wp::slice_t var_34;
    const wp::int32 var_35 = 0;
    wp::array_t<wp::float32> var_36;
    wp::slice_t var_37;
    const wp::int32 var_38 = 0;
    wp::array_t<wp::int32> var_39;
    wp::float32 var_40;
    wp::int32 var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::int32 var_44;
    wp::float32 var_45;
    const wp::float32 var_46 = 0.0;
    bool var_47;
    const wp::int32 var_48 = 1;
    wp::int32 var_49;
    const wp::int32 var_50 = 1;
    const wp::int32 var_51 = -1;
    bool var_52;
    bool var_53;
    const wp::int32 var_54 = 0;
    wp::float32 var_55;
    const wp::int32 var_56 = 0;
    const wp::int32 var_57 = 0;
    const bool var_58 = false;
    wp::range_t var_59;
    wp::int32 var_60;
    const wp::float32 var_61 = 0.5;
    wp::vec_t<3, wp::float32> var_62;
    wp::vec_t<3, wp::float32> var_63;
    wp::vec_t<3, wp::float32> var_64;
    wp::vec_t<3, wp::float32> var_65;
    const wp::int32 var_66 = 0;
    wp::vec_t<3, wp::float32> var_67;
    const wp::int32 var_68 = 0;
    wp::vec_t<3, wp::float32> var_69;
    wp::vec_t<3, wp::float32> var_70;
    wp::mat_t<3, 3, wp::float32> var_71;
    const wp::int32 var_72 = 1;
    wp::int32 var_73;
    const wp::int32 var_74 = 0;
    bool var_75;
    const wp::float32 var_76 = 1.0;
    const wp::float32 var_77 = -1.0;
    wp::mat_t<3, 3, wp::float32> var_78;
    const wp::int32 var_79 = 1;
    wp::int32 var_80;
    const wp::int32 var_81 = 0;
    wp::int32 var_82;
    wp::vec_t<2, wp::int32> var_83;
    wp::vec_t<2, wp::int32> var_84;
    wp::mat_t<3, 3, wp::float32> var_85;
    const wp::int32 var_86 = 0;
    wp::int32 var_87;
    wp::range_t var_88;
    wp::int32 var_89;
    wp::vec_t<3, wp::float32> var_90;
    wp::int32 var_91;
    wp::int32 var_92;
    //---------
    // forward
    // def eval_ccd_write_contact(                                                            <L 712>
    // points = mat43()                                                                       <L 767>
    var_0 = wp::mat_t<4, 3, wp::float32>();
    // witness1 = mat43()                                                                     <L 768>
    var_1 = wp::mat_t<4, 3, wp::float32>();
    // witness2 = mat43()                                                                     <L 769>
    var_2 = wp::mat_t<4, 3, wp::float32>();
    // geom1.margin = margin                                                                  <L 770>
    var_3 = &(var_geom1.margin);
    wp::store(var_3, var_margin);
    // geom2.margin = margin                                                                  <L 771>
    var_4 = &(var_geom2.margin);
    wp::store(var_4, var_margin);
    // is_collision_sensor = pairid[1] >= 0                                                   <L 772>
    var_6 = wp::extract(var_pairid, var_5);
    var_8 = (var_6 >= var_7);
    // if is_collision_sensor:                                                                <L 773>
    if (var_8) {
        // cutoff = 1.0e32                                                                    <L 774>
    }
    if (!var_8) {
        // cutoff = 0.0                                                                       <L 776>
    }
    var_11 = wp::where(var_8, var_9, var_10);
    // dist, ncollision, w1, w2, multiccd_idx = ccd(                                          <L 777>
    // opt_ccd_tolerance[worldid % opt_ccd_tolerance.shape[0]],                               <L 778>
    var_12 = &(var_opt_ccd_tolerance.shape);
    var_15 = wp::load(var_12);
    var_14 = wp::extract(var_15, var_13);
    var_16 = wp::mod(var_worldid, var_14);
    var_17 = wp::address(var_opt_ccd_tolerance, var_16);
    // cutoff,                                                                                <L 779>
    // gjk_iterations,                                                                        <L 780>
    // epa_iterations,                                                                        <L 781>
    // geom1,                                                                                 <L 782>
    // geom2,                                                                                 <L 783>
    // geomtype1,                                                                             <L 784>
    // geomtype2,                                                                             <L 785>
    // x1,                                                                                    <L 786>
    // x2,                                                                                    <L 787>
    // epa_vert_in[ccdid],                                                                    <L 788>
    var_22 = wp::slice_t(var_ccdid, var_ccdid, var_23);
    var_24 = wp::view(var_epa_vert_in, var_22);
    // epa_vert_index_in[ccdid],                                                              <L 789>
    var_25 = wp::slice_t(var_ccdid, var_ccdid, var_26);
    var_27 = wp::view(var_epa_vert_index_in, var_25);
    // epa_face_in[ccdid],                                                                    <L 790>
    var_28 = wp::slice_t(var_ccdid, var_ccdid, var_29);
    var_30 = wp::view(var_epa_face_in, var_28);
    // epa_pr_in[ccdid],                                                                      <L 791>
    var_31 = wp::slice_t(var_ccdid, var_ccdid, var_32);
    var_33 = wp::view(var_epa_pr_in, var_31);
    // epa_norm2_in[ccdid],                                                                   <L 792>
    var_34 = wp::slice_t(var_ccdid, var_ccdid, var_35);
    var_36 = wp::view(var_epa_norm2_in, var_34);
    // epa_horizon_in[ccdid],                                                                 <L 793>
    var_37 = wp::slice_t(var_ccdid, var_ccdid, var_38);
    var_39 = wp::view(var_epa_horizon_in, var_37);
    var_45 = wp::load(var_17);
    ccd_0(var_45, var_11, var_18, var_19, var_geom1, var_geom2, var_20, var_21, var_x1, var_x2, var_24, var_27, var_30, var_33, var_36, var_39, var_40, var_41, var_42, var_43, var_44);
    // if dist >= 0.0 and pairid[1] == -1:                                                    <L 796>
    var_47 = (var_40 >= var_46);
    var_49 = wp::extract(var_pairid, var_48);
    var_52 = (var_49 == var_51);
    var_53 = var_47 && var_52;
    if (var_53) {
        // return 0                                                                           <L 797>
        return var_54;
    }
    // dist += margin                                                                         <L 804>
    var_55 = wp::add(var_40, var_margin);
    // witness1[0] = w1                                                                       <L 806>
    wp::assign_inplace(var_1, var_56, var_42);
    // witness2[0] = w2                                                                       <L 807>
    wp::assign_inplace(var_2, var_57, var_43);
    // if wp.static(use_multiccd or (geomtype1 == GeomType.BOX and geomtype2 == GeomType.BOX)):       <L 809>
    // for i in range(ncollision):                                                            <L 844>
    var_59 = wp::range(var_41);
    start_for_1:;
        if (iter_cmp(var_59) == 0) goto end_for_1;
        var_60 = wp::iter_next(var_59);
        // points[i] = 0.5 * (witness1[i] + witness2[i])                                      <L 845>
        var_62 = wp::extract(var_1, var_60);
        var_63 = wp::extract(var_2, var_60);
        var_64 = wp::add(var_62, var_63);
        var_65 = wp::mul(var_61, var_64);
        wp::assign_inplace(var_0, var_60, var_65);
        goto start_for_1;
    end_for_1:;
    // normal = witness1[0] - witness2[0]                                                     <L 846>
    var_67 = wp::extract(var_1, var_66);
    var_69 = wp::extract(var_2, var_68);
    var_70 = wp::sub(var_67, var_69);
    // frame = make_frame(normal)                                                             <L 847>
    var_71 = make_frame_0(var_70);
    // if pairid[1] >= 0:                                                                     <L 850>
    var_73 = wp::extract(var_pairid, var_72);
    var_75 = (var_73 >= var_74);
    if (var_75) {
        // frame *= -1.0                                                                      <L 851>
        var_78 = wp::mul(var_71, var_77);
        // geoms = wp.vec2i(geoms[1], geoms[0])                                               <L 852>
        var_80 = wp::extract(var_geoms, var_79);
        var_82 = wp::extract(var_geoms, var_81);
        var_83 = wp::vec_t<2, wp::int32>(var_80, var_82);
    }
    var_84 = wp::where(var_75, var_83, var_geoms);
    var_85 = wp::where(var_75, var_78, var_71);
    // nactive = int(0)  # number of contacts contributing to the physics                     <L 854>
    var_87 = wp::int(var_86);
    // for i in range(ncollision):                                                            <L 855>
    var_88 = wp::range(var_41);
    start_for_3:;
        if (iter_cmp(var_88) == 0) goto end_for_3;
        var_89 = wp::iter_next(var_88);
        // active = write_contact(                                                            <L 856>
        // naconmax_in,                                                                       <L 857>
        // i,                                                                                 <L 858>
        // dist,                                                                              <L 859>
        // points[i],                                                                         <L 860>
        var_90 = wp::extract(var_0, var_89);
        // frame,                                                                             <L 861>
        // margin,                                                                            <L 862>
        // gap,                                                                               <L 863>
        // condim,                                                                            <L 864>
        // friction,                                                                          <L 865>
        // solref,                                                                            <L 866>
        // solreffriction,                                                                    <L 867>
        // solimp,                                                                            <L 868>
        // geoms,                                                                             <L 869>
        // pairid,                                                                            <L 870>
        // worldid,                                                                           <L 871>
        // contact_dist_out,                                                                  <L 872>
        // contact_pos_out,                                                                   <L 873>
        // contact_frame_out,                                                                 <L 874>
        // contact_includemargin_out,                                                         <L 875>
        // contact_friction_out,                                                              <L 876>
        // contact_solref_out,                                                                <L 877>
        // contact_solreffriction_out,                                                        <L 878>
        // contact_solimp_out,                                                                <L 879>
        // contact_dim_out,                                                                   <L 880>
        // contact_geom_out,                                                                  <L 881>
        // contact_efc_address_out,                                                           <L 882>
        // contact_worldid_out,                                                               <L 883>
        // contact_type_out,                                                                  <L 884>
        // contact_geomcollisionid_out,                                                       <L 885>
        // nacon_out,                                                                         <L 886>
        var_91 = write_contact_0(var_naconmax_in, var_89, var_55, var_90, var_85, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_84, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
        // nactive += active                                                                  <L 888>
        var_92 = wp::add(var_87, var_91);
        wp::assign(var_87, var_92);
        goto start_for_3;
    end_for_3:;
    // return nactive                                                                         <L 890>
    return var_87;
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:90
static CUDA_CALLABLE void adj__discrete_geoms_0(
    wp::int32 var_g1,
    wp::int32 var_g2,
    wp::int32 & adj_g1,
    wp::int32 & adj_g2,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:97
static CUDA_CALLABLE void adj_support_0(
    Geom_3242f8a8 var_geom,
    wp::int32 var_geomtype,
    wp::vec_t<3, wp::float32> var_dir,
    Geom_3242f8a8 & adj_geom,
    wp::int32 & adj_geomtype,
    wp::vec_t<3, wp::float32> & adj_dir,
    SupportPoint_e82efc60 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:265
static CUDA_CALLABLE void adj__det3_0(
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> var_v3,
    wp::vec_t<3, wp::float32> & adj_v1,
    wp::vec_t<3, wp::float32> & adj_v2,
    wp::vec_t<3, wp::float32> & adj_v3,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:270
static CUDA_CALLABLE void adj__same_sign_0(
    wp::float32 var_a,
    wp::float32 var_b,
    wp::float32 & adj_a,
    wp::float32 & adj_b,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:286
static CUDA_CALLABLE void adj__project_origin_plane_0(
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> var_v3,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::int32 & ret_1,
    wp::vec_t<3, wp::float32> & adj_v1,
    wp::vec_t<3, wp::float32> & adj_v2,
    wp::vec_t<3, wp::float32> & adj_v3,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::int32 & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:279
static CUDA_CALLABLE void adj__project_origin_line_0(
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> & adj_v1,
    wp::vec_t<3, wp::float32> & adj_v2,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:539
static CUDA_CALLABLE void adj__S1D_0(
    wp::vec_t<3, wp::float32> var_s1,
    wp::vec_t<3, wp::float32> var_s2,
    wp::vec_t<3, wp::float32> & adj_s1,
    wp::vec_t<3, wp::float32> & adj_s2,
    wp::vec_t<2, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:394
static CUDA_CALLABLE void adj__S2D_0(
    wp::vec_t<3, wp::float32> var_s1,
    wp::vec_t<3, wp::float32> var_s2,
    wp::vec_t<3, wp::float32> var_s3,
    wp::vec_t<3, wp::float32> & adj_s1,
    wp::vec_t<3, wp::float32> & adj_s2,
    wp::vec_t<3, wp::float32> & adj_s3,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:318
static CUDA_CALLABLE void adj__S3D_0(
    wp::vec_t<3, wp::float32> var_s1,
    wp::vec_t<3, wp::float32> var_s2,
    wp::vec_t<3, wp::float32> var_s3,
    wp::vec_t<3, wp::float32> var_s4,
    wp::vec_t<3, wp::float32> & adj_s1,
    wp::vec_t<3, wp::float32> & adj_s2,
    wp::vec_t<3, wp::float32> & adj_s3,
    wp::vec_t<3, wp::float32> & adj_s4,
    wp::vec_t<4, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:252
static CUDA_CALLABLE void adj__subdistance_0(
    wp::int32 var_n,
    wp::mat_t<4, 3, wp::float32> var_simplex,
    wp::int32 & adj_n,
    wp::mat_t<4, 3, wp::float32> & adj_simplex,
    wp::vec_t<4, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:233
static CUDA_CALLABLE void adj__linear_combine_0(
    wp::int32 var_n,
    wp::vec_t<4, wp::float32> var_coefs,
    wp::mat_t<4, 3, wp::float32> var_mat,
    wp::int32 & adj_n,
    wp::vec_t<4, wp::float32> & adj_coefs,
    wp::mat_t<4, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:247
static CUDA_CALLABLE void adj__almost_equal_0(
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> & adj_v1,
    wp::vec_t<3, wp::float32> & adj_v2,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/warp/_src/math.py:0
static CUDA_CALLABLE void adj_norm_l2_0(
    wp::vec_t<3, wp::float32> var_v,
    wp::vec_t<3, wp::float32> & adj_v,
    wp::float32 & adj_ret)
{
    //---------
    // primal vars
    wp::float32 var_0;
    //---------
    // dual vars
    wp::float32 adj_0 = {};
    //---------
    // forward
    // def norm_l2(v: Any) -> float:                                                          <L 1>
    // return wp.length(v)                                                                    <L 12>
    var_0 = wp::length(var_v);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_0 += adj_ret;
    wp::adj_length(var_v, var_0, adj_v, adj_0);
    // adj: return wp.length(v)                                                               <L 12>
    // adj: def norm_l2(v: Any) -> float:                                                     <L 1>
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:562
static CUDA_CALLABLE void adj_gjk_0(
    wp::float32 var_tolerance,
    wp::int32 var_gjk_iterations,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::vec_t<3, wp::float32> var_x1_0,
    wp::vec_t<3, wp::float32> var_x2_0,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2,
    wp::float32 var_cutoff,
    bool var_is_discrete,
    wp::float32 & adj_tolerance,
    wp::int32 & adj_gjk_iterations,
    Geom_3242f8a8 & adj_geom1,
    Geom_3242f8a8 & adj_geom2,
    wp::vec_t<3, wp::float32> & adj_x1_0,
    wp::vec_t<3, wp::float32> & adj_x2_0,
    wp::int32 & adj_geomtype1,
    wp::int32 & adj_geomtype2,
    wp::float32 & adj_cutoff,
    bool & adj_is_discrete,
    GJKResult_0220ee01 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:701
static CUDA_CALLABLE void adj__tri_affine_coord_0(
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> var_v3,
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> & adj_v1,
    wp::vec_t<3, wp::float32> & adj_v2,
    wp::vec_t<3, wp::float32> & adj_v3,
    wp::vec_t<3, wp::float32> & adj_p,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:2152
static CUDA_CALLABLE void adj__inflate_0(
    GJKResult_0220ee01 var_result,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2,
    wp::float32 var_margin1,
    wp::float32 var_margin2,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    GJKResult_0220ee01 & adj_result,
    Geom_3242f8a8 & adj_geom1,
    Geom_3242f8a8 & adj_geom2,
    wp::int32 & adj_geomtype1,
    wp::int32 & adj_geomtype2,
    wp::float32 & adj_margin1,
    wp::float32 & adj_margin2,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:800
static CUDA_CALLABLE void adj__rotmat_0(
    wp::vec_t<3, wp::float32> var_axis,
    wp::vec_t<3, wp::float32> & adj_axis,
    wp::mat_t<3, 3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:216
static CUDA_CALLABLE void adj__epa_support_0(
    Polytope_9ab93ade var_pt,
    wp::int32 var_idx,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geom1_type,
    wp::int32 var_geom2_type,
    wp::vec_t<3, wp::float32> var_dir,
    wp::int32 & ret_0,
    wp::int32 & ret_1,
    Polytope_9ab93ade & adj_pt,
    wp::int32 & adj_idx,
    Geom_3242f8a8 & adj_geom1,
    Geom_3242f8a8 & adj_geom2,
    wp::int32 & adj_geom1_type,
    wp::int32 & adj_geom2_type,
    wp::vec_t<3, wp::float32> & adj_dir,
    wp::int32 & adj_ret_0,
    wp::int32 & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:194
static CUDA_CALLABLE void adj__attach_face_0(
    Polytope_9ab93ade var_pt,
    wp::int32 var_idx,
    wp::int32 var_v1,
    wp::int32 var_v2,
    wp::int32 var_v3,
    Polytope_9ab93ade & adj_pt,
    wp::int32 & adj_idx,
    wp::int32 & adj_v1,
    wp::int32 & adj_v2,
    wp::int32 & adj_v3,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:761
static CUDA_CALLABLE void adj__replace_simplex3_0(
    Polytope_9ab93ade var_pt,
    wp::int32 var_v1,
    wp::int32 var_v2,
    wp::int32 var_v3,
    Polytope_9ab93ade & adj_pt,
    wp::int32 & adj_v1,
    wp::int32 & adj_v2,
    wp::int32 & adj_v3,
    GJKResult_0220ee01 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:822
static CUDA_CALLABLE void adj__ray_triangle_0(
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> var_v3,
    wp::vec_t<3, wp::float32> var_v4,
    wp::vec_t<3, wp::float32> var_v5,
    wp::vec_t<3, wp::float32> & adj_v1,
    wp::vec_t<3, wp::float32> & adj_v2,
    wp::vec_t<3, wp::float32> & adj_v3,
    wp::vec_t<3, wp::float32> & adj_v4,
    wp::vec_t<3, wp::float32> & adj_v5,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:936
static CUDA_CALLABLE void adj__polytope2_0(
    Polytope_9ab93ade var_pt,
    wp::mat_t<4, 3, wp::float32> var_simplex,
    wp::mat_t<4, 3, wp::float32> var_simplex1,
    wp::mat_t<4, 3, wp::float32> var_simplex2,
    wp::vec_t<4, wp::int32> var_simplex_index1,
    wp::vec_t<4, wp::int32> var_simplex_index2,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2,
    Polytope_9ab93ade & ret_0,
    GJKResult_0220ee01 & ret_1,
    Polytope_9ab93ade & adj_pt,
    wp::mat_t<4, 3, wp::float32> & adj_simplex,
    wp::mat_t<4, 3, wp::float32> & adj_simplex1,
    wp::mat_t<4, 3, wp::float32> & adj_simplex2,
    wp::vec_t<4, wp::int32> & adj_simplex_index1,
    wp::vec_t<4, wp::int32> & adj_simplex_index2,
    Geom_3242f8a8 & adj_geom1,
    Geom_3242f8a8 & adj_geom2,
    wp::int32 & adj_geomtype1,
    wp::int32 & adj_geomtype2,
    Polytope_9ab93ade & adj_ret_0,
    GJKResult_0220ee01 & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:688
static CUDA_CALLABLE void adj__same_side_0(
    wp::vec_t<3, wp::float32> var_p0,
    wp::vec_t<3, wp::float32> var_p1,
    wp::vec_t<3, wp::float32> var_p2,
    wp::vec_t<3, wp::float32> var_p3,
    wp::vec_t<3, wp::float32> & adj_p0,
    wp::vec_t<3, wp::float32> & adj_p1,
    wp::vec_t<3, wp::float32> & adj_p2,
    wp::vec_t<3, wp::float32> & adj_p3,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:696
static CUDA_CALLABLE void adj__test_tetra_0(
    wp::vec_t<3, wp::float32> var_p0,
    wp::vec_t<3, wp::float32> var_p1,
    wp::vec_t<3, wp::float32> var_p2,
    wp::vec_t<3, wp::float32> var_p3,
    wp::vec_t<3, wp::float32> & adj_p0,
    wp::vec_t<3, wp::float32> & adj_p1,
    wp::vec_t<3, wp::float32> & adj_p2,
    wp::vec_t<3, wp::float32> & adj_p3,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:1114
static CUDA_CALLABLE void adj__polytope4_0(
    Polytope_9ab93ade var_pt,
    wp::mat_t<4, 3, wp::float32> var_simplex,
    wp::mat_t<4, 3, wp::float32> var_simplex1,
    wp::mat_t<4, 3, wp::float32> var_simplex2,
    wp::vec_t<4, wp::int32> var_simplex_index1,
    wp::vec_t<4, wp::int32> var_simplex_index2,
    Polytope_9ab93ade & ret_0,
    GJKResult_0220ee01 & ret_1,
    Polytope_9ab93ade & adj_pt,
    wp::mat_t<4, 3, wp::float32> & adj_simplex,
    wp::mat_t<4, 3, wp::float32> & adj_simplex1,
    wp::mat_t<4, 3, wp::float32> & adj_simplex2,
    wp::vec_t<4, wp::int32> & adj_simplex_index1,
    wp::vec_t<4, wp::int32> & adj_simplex_index2,
    Polytope_9ab93ade & adj_ret_0,
    GJKResult_0220ee01 & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:744
static CUDA_CALLABLE void adj__tri_point_intersect_0(
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> var_v3,
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> & adj_v1,
    wp::vec_t<3, wp::float32> & adj_v2,
    wp::vec_t<3, wp::float32> & adj_v3,
    wp::vec_t<3, wp::float32> & adj_p,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:1026
static CUDA_CALLABLE void adj__polytope3_0(
    Polytope_9ab93ade var_pt,
    wp::float32 var_dist,
    wp::mat_t<4, 3, wp::float32> var_simplex,
    wp::mat_t<4, 3, wp::float32> var_simplex1,
    wp::mat_t<4, 3, wp::float32> var_simplex2,
    wp::vec_t<4, wp::int32> var_simplex_index1,
    wp::vec_t<4, wp::int32> var_simplex_index2,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2,
    Polytope_9ab93ade & adj_pt,
    wp::float32 & adj_dist,
    wp::mat_t<4, 3, wp::float32> & adj_simplex,
    wp::mat_t<4, 3, wp::float32> & adj_simplex1,
    wp::mat_t<4, 3, wp::float32> & adj_simplex2,
    wp::vec_t<4, wp::int32> & adj_simplex_index1,
    wp::vec_t<4, wp::int32> & adj_simplex_index2,
    Geom_3242f8a8 & adj_geom1,
    Geom_3242f8a8 & adj_geom2,
    wp::int32 & adj_geomtype1,
    wp::int32 & adj_geomtype2,
    Polytope_9ab93ade & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:1195
static CUDA_CALLABLE void adj__is_invalid_face_0(
    wp::int32 var_face,
    wp::int32 & adj_face,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:1177
static CUDA_CALLABLE void adj__delete_face_0(
    wp::int32 var_face,
    wp::int32 & adj_face,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:1171
static CUDA_CALLABLE void adj__get_face_verts_0(
    wp::int32 var_face,
    wp::int32 & adj_face,
    wp::vec_t<3, wp::int32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:840
static CUDA_CALLABLE void adj__add_edge_0(
    Polytope_9ab93ade var_pt,
    wp::int32 var_e1,
    wp::int32 var_e2,
    Polytope_9ab93ade & adj_pt,
    wp::int32 & adj_e1,
    wp::int32 & adj_e2,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:1183
static CUDA_CALLABLE void adj__is_face_deleted_0(
    wp::int32 var_face,
    wp::int32 & adj_face,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:835
static CUDA_CALLABLE void adj__get_edge_0(
    wp::int32 var_edge,
    wp::int32 & adj_edge,
    wp::vec_t<2, wp::int32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:1189
static CUDA_CALLABLE void adj__invalidate_face_0(
    wp::int32 var_face,
    wp::int32 & adj_face,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:862
static CUDA_CALLABLE void adj__epa_witness_0(
    Polytope_9ab93ade var_pt,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2,
    wp::int32 var_face_idx,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::float32 & ret_2,
    Polytope_9ab93ade & adj_pt,
    Geom_3242f8a8 & adj_geom1,
    Geom_3242f8a8 & adj_geom2,
    wp::int32 & adj_geomtype1,
    wp::int32 & adj_geomtype2,
    wp::int32 & adj_face_idx,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::float32 & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:1201
static CUDA_CALLABLE void adj__epa_0(
    wp::float32 var_tolerance,
    wp::int32 var_epa_iterations,
    Polytope_9ab93ade var_pt,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2,
    bool var_is_discrete,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::int32 & ret_3,
    wp::float32 & adj_tolerance,
    wp::int32 & adj_epa_iterations,
    Polytope_9ab93ade & adj_pt,
    Geom_3242f8a8 & adj_geom1,
    Geom_3242f8a8 & adj_geom2,
    wp::int32 & adj_geomtype1,
    wp::int32 & adj_geomtype2,
    bool & adj_is_discrete,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2,
    wp::int32 & adj_ret_3)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_gjk.py:2199
static CUDA_CALLABLE void adj_ccd_0(
    wp::float32 var_tolerance,
    wp::float32 var_cutoff,
    wp::int32 var_gjk_iterations,
    wp::int32 var_epa_iterations,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::int32 var_geomtype1,
    wp::int32 var_geomtype2,
    wp::vec_t<3, wp::float32> var_x_1,
    wp::vec_t<3, wp::float32> var_x_2,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vert,
    wp::array_t<wp::int32> var_vert_index,
    wp::array_t<wp::int32> var_face,
    wp::array_t<wp::vec_t<3, wp::float32>> var_face_pr,
    wp::array_t<wp::float32> var_face_norm2,
    wp::array_t<wp::int32> var_horizon,
    wp::float32 & ret_0,
    wp::int32 & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & ret_3,
    wp::int32 & ret_4,
    wp::float32 & adj_tolerance,
    wp::float32 & adj_cutoff,
    wp::int32 & adj_gjk_iterations,
    wp::int32 & adj_epa_iterations,
    Geom_3242f8a8 & adj_geom1,
    Geom_3242f8a8 & adj_geom2,
    wp::int32 & adj_geomtype1,
    wp::int32 & adj_geomtype2,
    wp::vec_t<3, wp::float32> & adj_x_1,
    wp::vec_t<3, wp::float32> & adj_x_2,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_vert,
    wp::array_t<wp::int32> & adj_vert_index,
    wp::array_t<wp::int32> & adj_face,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_face_pr,
    wp::array_t<wp::float32> & adj_face_norm2,
    wp::array_t<wp::int32> & adj_horizon,
    wp::float32 & adj_ret_0,
    wp::int32 & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2,
    wp::vec_t<3, wp::float32> & adj_ret_3,
    wp::int32 & adj_ret_4)
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_convex.py:711
static CUDA_CALLABLE void adj_ccd_kernel_builder__locals__eval_ccd_write_contact_1(
    wp::array_t<wp::float32> var_opt_ccd_tolerance,
    wp::int32 var_naconmax_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_epa_vert_in,
    wp::array_t<wp::int32> var_epa_vert_index_in,
    wp::array_t<wp::int32> var_epa_face_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_epa_pr_in,
    wp::array_t<wp::float32> var_epa_norm2_in,
    wp::array_t<wp::int32> var_epa_horizon_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_polygon_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_clipped_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_pnormal_in,
    wp::array_t<wp::float32> var_multiccd_pdist_in,
    wp::array_t<wp::int32> var_multiccd_idx1_in,
    wp::array_t<wp::int32> var_multiccd_idx2_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_n1_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_n2_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_endvert_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_face1_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_face2_in,
    Geom_3242f8a8 var_geom1,
    Geom_3242f8a8 var_geom2,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::int32 var_worldid,
    wp::int32 var_ccdid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::vec_t<3, wp::float32> var_x1,
    wp::vec_t<3, wp::float32> var_x2,
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
    wp::array_t<wp::float32> & adj_opt_ccd_tolerance,
    wp::int32 & adj_naconmax_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_epa_vert_in,
    wp::array_t<wp::int32> & adj_epa_vert_index_in,
    wp::array_t<wp::int32> & adj_epa_face_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_epa_pr_in,
    wp::array_t<wp::float32> & adj_epa_norm2_in,
    wp::array_t<wp::int32> & adj_epa_horizon_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_multiccd_polygon_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_multiccd_clipped_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_multiccd_pnormal_in,
    wp::array_t<wp::float32> & adj_multiccd_pdist_in,
    wp::array_t<wp::int32> & adj_multiccd_idx1_in,
    wp::array_t<wp::int32> & adj_multiccd_idx2_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_multiccd_n1_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_multiccd_n2_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_multiccd_endvert_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_multiccd_face1_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_multiccd_face2_in,
    Geom_3242f8a8 & adj_geom1,
    Geom_3242f8a8 & adj_geom2,
    wp::vec_t<2, wp::int32> & adj_geoms,
    wp::int32 & adj_worldid,
    wp::int32 & adj_ccdid,
    wp::float32 & adj_margin,
    wp::float32 & adj_gap,
    wp::int32 & adj_condim,
    wp::vec_t<5, wp::float32> & adj_friction,
    wp::vec_t<2, wp::float32> & adj_solref,
    wp::vec_t<2, wp::float32> & adj_solreffriction,
    wp::vec_t<5, wp::float32> & adj_solimp,
    wp::vec_t<3, wp::float32> & adj_x1,
    wp::vec_t<3, wp::float32> & adj_x2,
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
    wp::array_t<wp::int32> & adj_nacon_out,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void ccd_kernel_builder__locals__ccd_kernel_1711caa4_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_opt_ccd_tolerance,
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
    wp::int32 var_naccdmax_in,
    wp::array_t<wp::int32> var_ncollision_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pair_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pairid_in,
    wp::array_t<wp::int32> var_collision_worldid_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_epa_vert_in,
    wp::array_t<wp::int32> var_epa_vert_index_in,
    wp::array_t<wp::int32> var_epa_face_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_epa_pr_in,
    wp::array_t<wp::float32> var_epa_norm2_in,
    wp::array_t<wp::int32> var_epa_horizon_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_polygon_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_clipped_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_pnormal_in,
    wp::array_t<wp::float32> var_multiccd_pdist_in,
    wp::array_t<wp::int32> var_multiccd_idx1_in,
    wp::array_t<wp::int32> var_multiccd_idx2_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_n1_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_n2_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_endvert_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_face1_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_multiccd_face2_in,
    wp::array_t<wp::int32> var_nccd_in,
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
        const wp::int32 var_8 = 0;
        wp::int32 var_9;
        const wp::int32 var_10 = 1;
        wp::int32 var_11;
        wp::int32* var_12;
        const wp::int32 var_13 = 5;
        bool var_14;
        wp::int32 var_15;
        wp::int32* var_16;
        const wp::int32 var_17 = 5;
        bool var_18;
        wp::int32 var_19;
        bool var_20;
        const wp::int32 var_21 = 40;
        const wp::int32 var_22 = 1;
        wp::int32 var_23;
        bool var_24;
        const wp::str var_25 = "CCD overflow - please increase naccdmax to %u\n";
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::vec_t<2, wp::int32> var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        wp::int32 var_32;
        wp::vec_t<5, wp::float32> var_33;
        wp::vec_t<2, wp::float32> var_34;
        wp::vec_t<2, wp::float32> var_35;
        wp::vec_t<5, wp::float32> var_36;
        Geom_3242f8a8 var_37;
        Geom_3242f8a8 var_38;
        wp::vec_t<3, wp::float32>* var_39;
        wp::vec_t<3, wp::float32>* var_40;
        wp::vec_t<2, wp::int32>* var_41;
        wp::int32 var_42;
        wp::vec_t<3, wp::float32> var_43;
        wp::vec_t<3, wp::float32> var_44;
        wp::vec_t<2, wp::int32> var_45;
        //---------
        // forward
        // def ccd_kernel(                                                                        <L 894>
        // collisionid = wp.tid()                                                                 <L 974>
        var_0 = builtin_tid1d();
        // if collisionid >= ncollision_in[0]:                                                    <L 975>
        var_2 = wp::address(var_ncollision_in, var_1);
        var_4 = wp::load(var_2);
        var_3 = (var_0 >= var_4);
        if (var_3) {
            // return                                                                             <L 976>
            continue;
        }
        // geoms = collision_pair_in[collisionid]                                                 <L 978>
        var_5 = wp::address(var_collision_pair_in, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // g1 = geoms[0]                                                                          <L 979>
        var_9 = wp::extract(var_6, var_8);
        // g2 = geoms[1]                                                                          <L 980>
        var_11 = wp::extract(var_6, var_10);
        // if geom_type[g1] != geomtype1 or geom_type[g2] != geomtype2:                           <L 982>
        var_12 = wp::address(var_geom_type, var_9);
        var_15 = wp::load(var_12);
        var_14 = (var_15 != var_13);
        var_16 = wp::address(var_geom_type, var_11);
        var_19 = wp::load(var_16);
        var_18 = (var_19 != var_17);
        var_20 = var_14 || var_18;
        if (var_20) {
            // return                                                                             <L 983>
            continue;
        }
        // ccdid = wp.atomic_add(nccd_in, wp.static(geomgeomid), 1)                               <L 985>
        var_23 = wp::atomic_add(var_nccd_in, var_21, var_22);
        // if ccdid >= naccdmax_in:                                                               <L 986>
        var_24 = (var_23 >= var_naccdmax_in);
        if (var_24) {
            // wp.printf("CCD overflow - please increase naccdmax to %u\n", ccdid)                <L 987>
            printf(var_25, var_23);
            // return                                                                             <L 988>
            continue;
        }
        // worldid = collision_worldid_in[collisionid]                                            <L 990>
        var_26 = wp::address(var_collision_worldid_in, var_0);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // _, margin, gap, condim, friction, solref, solreffriction, solimp = contact_params(       <L 992>
        // geom_condim,                                                                           <L 993>
        // geom_priority,                                                                         <L 994>
        // geom_solmix,                                                                           <L 995>
        // geom_solref,                                                                           <L 996>
        // geom_solimp,                                                                           <L 997>
        // geom_friction,                                                                         <L 998>
        // geom_margin,                                                                           <L 999>
        // geom_gap,                                                                              <L 1000>
        // pair_dim,                                                                              <L 1001>
        // pair_solref,                                                                           <L 1002>
        // pair_solreffriction,                                                                   <L 1003>
        // pair_solimp,                                                                           <L 1004>
        // pair_margin,                                                                           <L 1005>
        // pair_gap,                                                                              <L 1006>
        // pair_friction,                                                                         <L 1007>
        // collision_pair_in,                                                                     <L 1008>
        // collision_pairid_in,                                                                   <L 1009>
        // collisionid,                                                                           <L 1010>
        // worldid,                                                                               <L 1011>
        contact_params_0(var_geom_condim, var_geom_priority, var_geom_solmix, var_geom_solref, var_geom_solimp, var_geom_friction, var_geom_margin, var_geom_gap, var_pair_dim, var_pair_solref, var_pair_solreffriction, var_pair_solimp, var_pair_margin, var_pair_gap, var_pair_friction, var_collision_pair_in, var_collision_pairid_in, var_0, var_27, var_29, var_30, var_31, var_32, var_33, var_34, var_35, var_36);
        // geom1, geom2 = geom_collision_pair(                                                    <L 1014>
        // geom_type,                                                                             <L 1015>
        // geom_dataid,                                                                           <L 1016>
        // geom_size,                                                                             <L 1017>
        // mesh_vertadr,                                                                          <L 1018>
        // mesh_vertnum,                                                                          <L 1019>
        // mesh_graphadr,                                                                         <L 1020>
        // mesh_vert,                                                                             <L 1021>
        // mesh_graph,                                                                            <L 1022>
        // mesh_polynum,                                                                          <L 1023>
        // mesh_polyadr,                                                                          <L 1024>
        // mesh_polynormal,                                                                       <L 1025>
        // mesh_polyvertadr,                                                                      <L 1026>
        // mesh_polyvertnum,                                                                      <L 1027>
        // mesh_polyvert,                                                                         <L 1028>
        // mesh_polymapadr,                                                                       <L 1029>
        // mesh_polymapnum,                                                                       <L 1030>
        // mesh_polymap,                                                                          <L 1031>
        // geom_xpos_in,                                                                          <L 1032>
        // geom_xmat_in,                                                                          <L 1033>
        // geoms,                                                                                 <L 1034>
        // worldid,                                                                               <L 1035>
        geom_collision_pair_0(var_geom_type, var_geom_dataid, var_geom_size, var_mesh_vertadr, var_mesh_vertnum, var_mesh_graphadr, var_mesh_vert, var_mesh_graph, var_mesh_polynum, var_mesh_polyadr, var_mesh_polynormal, var_mesh_polyvertadr, var_mesh_polyvertnum, var_mesh_polyvert, var_mesh_polymapadr, var_mesh_polymapnum, var_mesh_polymap, var_geom_xpos_in, var_geom_xmat_in, var_6, var_27, var_37, var_38);
        // eval_ccd_write_contact(                                                                <L 1038>
        // opt_ccd_tolerance,                                                                     <L 1039>
        // naconmax_in,                                                                           <L 1040>
        // epa_vert_in,                                                                           <L 1041>
        // epa_vert_index_in,                                                                     <L 1042>
        // epa_face_in,                                                                           <L 1043>
        // epa_pr_in,                                                                             <L 1044>
        // epa_norm2_in,                                                                          <L 1045>
        // epa_horizon_in,                                                                        <L 1046>
        // multiccd_polygon_in,                                                                   <L 1047>
        // multiccd_clipped_in,                                                                   <L 1048>
        // multiccd_pnormal_in,                                                                   <L 1049>
        // multiccd_pdist_in,                                                                     <L 1050>
        // multiccd_idx1_in,                                                                      <L 1051>
        // multiccd_idx2_in,                                                                      <L 1052>
        // multiccd_n1_in,                                                                        <L 1053>
        // multiccd_n2_in,                                                                        <L 1054>
        // multiccd_endvert_in,                                                                   <L 1055>
        // multiccd_face1_in,                                                                     <L 1056>
        // multiccd_face2_in,                                                                     <L 1057>
        // geom1,                                                                                 <L 1058>
        // geom2,                                                                                 <L 1059>
        // geoms,                                                                                 <L 1060>
        // worldid,                                                                               <L 1061>
        // ccdid,                                                                                 <L 1062>
        // margin,                                                                                <L 1063>
        // gap,                                                                                   <L 1064>
        // condim,                                                                                <L 1065>
        // friction,                                                                              <L 1066>
        // solref,                                                                                <L 1067>
        // solreffriction,                                                                        <L 1068>
        // solimp,                                                                                <L 1069>
        // geom1.pos,                                                                             <L 1070>
        var_39 = &(var_37.pos);
        // geom2.pos,                                                                             <L 1071>
        var_40 = &(var_38.pos);
        // collision_pairid_in[collisionid],                                                      <L 1072>
        var_41 = wp::address(var_collision_pairid_in, var_0);
        // contact_dist_out,                                                                      <L 1073>
        // contact_pos_out,                                                                       <L 1074>
        // contact_frame_out,                                                                     <L 1075>
        // contact_includemargin_out,                                                             <L 1076>
        // contact_friction_out,                                                                  <L 1077>
        // contact_solref_out,                                                                    <L 1078>
        // contact_solreffriction_out,                                                            <L 1079>
        // contact_solimp_out,                                                                    <L 1080>
        // contact_dim_out,                                                                       <L 1081>
        // contact_geom_out,                                                                      <L 1082>
        // contact_efc_address_out,                                                               <L 1083>
        // contact_worldid_out,                                                                   <L 1084>
        // contact_type_out,                                                                      <L 1085>
        // contact_geomcollisionid_out,                                                           <L 1086>
        // nacon_out,                                                                             <L 1087>
        var_43 = wp::load(var_39);
        var_44 = wp::load(var_40);
        var_45 = wp::load(var_41);
        var_42 = ccd_kernel_builder__locals__eval_ccd_write_contact_1(var_opt_ccd_tolerance, var_naconmax_in, var_epa_vert_in, var_epa_vert_index_in, var_epa_face_in, var_epa_pr_in, var_epa_norm2_in, var_epa_horizon_in, var_multiccd_polygon_in, var_multiccd_clipped_in, var_multiccd_pnormal_in, var_multiccd_pdist_in, var_multiccd_idx1_in, var_multiccd_idx2_in, var_multiccd_n1_in, var_multiccd_n2_in, var_multiccd_endvert_in, var_multiccd_face1_in, var_multiccd_face2_in, var_37, var_38, var_6, var_27, var_23, var_30, var_31, var_32, var_33, var_34, var_35, var_36, var_43, var_44, var_45, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_nacon_out);
    }
}

