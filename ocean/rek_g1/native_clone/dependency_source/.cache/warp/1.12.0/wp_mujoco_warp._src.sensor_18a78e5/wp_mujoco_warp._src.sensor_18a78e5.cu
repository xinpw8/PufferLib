
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


struct VolumeData_53ac1a2d
{
    wp::vec_t<3, wp::float32> center;
    wp::vec_t<3, wp::float32> half_size;
    wp::array_t<wp::vec_t<3, wp::float32>> oct_aabb;
    wp::array_t<wp::vec_t<8, wp::int32>> oct_child;
    wp::array_t<wp::vec_t<8, wp::float32>> oct_coeff;
    wp::int32 root;
    bool valid;


    VolumeData_53ac1a2d() = default;
    CUDA_CALLABLE VolumeData_53ac1a2d(wp::vec_t<3, wp::float32> const& center,
    wp::vec_t<3, wp::float32> const& half_size = {},
    wp::array_t<wp::vec_t<3, wp::float32>> const& oct_aabb = {},
    wp::array_t<wp::vec_t<8, wp::int32>> const& oct_child = {},
    wp::array_t<wp::vec_t<8, wp::float32>> const& oct_coeff = {},
    wp::int32 const& root = {},
    bool const& valid = {})
        : center{center}
        , half_size{half_size}
        , oct_aabb{oct_aabb}
        , oct_child{oct_child}
        , oct_coeff{oct_coeff}
        , root{root}
        , valid{valid}

    {
    }

    CUDA_CALLABLE VolumeData_53ac1a2d& operator += (const VolumeData_53ac1a2d& rhs)
    {    center += rhs.center;
    half_size += rhs.half_size;
    root += rhs.root;
    valid += rhs.valid;

        return *this;}

};

static CUDA_CALLABLE void adj_VolumeData_53ac1a2d(wp::vec_t<3, wp::float32> const&,
    wp::vec_t<3, wp::float32> const&,
    wp::array_t<wp::vec_t<3, wp::float32>> const&,
    wp::array_t<wp::vec_t<8, wp::int32>> const&,
    wp::array_t<wp::vec_t<8, wp::float32>> const&,
    wp::int32 const&,
    bool const&,
    wp::vec_t<3, wp::float32> & adj_center,
    wp::vec_t<3, wp::float32> & adj_half_size,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_oct_aabb,
    wp::array_t<wp::vec_t<8, wp::int32>> & adj_oct_child,
    wp::array_t<wp::vec_t<8, wp::float32>> & adj_oct_coeff,
    wp::int32 & adj_root,
    bool & adj_valid,
    VolumeData_53ac1a2d & adj_ret)
{
    adj_center += adj_ret.center;
    adj_half_size += adj_ret.half_size;
    adj_oct_aabb = adj_ret.oct_aabb;
    adj_oct_child = adj_ret.oct_child;
    adj_oct_coeff = adj_ret.oct_coeff;
    adj_root += adj_ret.root;
    adj_valid += adj_ret.valid;
}

// Required when compiling adjoints.
CUDA_CALLABLE VolumeData_53ac1a2d add(const VolumeData_53ac1a2d& a, const VolumeData_53ac1a2d& b)
{
    return VolumeData_53ac1a2d();
}

CUDA_CALLABLE void adj_atomic_add(VolumeData_53ac1a2d* p, VolumeData_53ac1a2d t)
{
    wp::adj_atomic_add(&p->center, t.center);
    wp::adj_atomic_add(&p->half_size, t.half_size);
    wp::adj_atomic_add(&p->oct_aabb, t.oct_aabb);
    wp::adj_atomic_add(&p->oct_child, t.oct_child);
    wp::adj_atomic_add(&p->oct_coeff, t.oct_coeff);
    wp::adj_atomic_add(&p->root, t.root);
    wp::adj_atomic_add(&p->valid, t.valid);
}



struct MeshData_52eaa0fa
{
    wp::int32 nmeshface;
    wp::array_t<wp::int32> mesh_vertadr;
    wp::array_t<wp::vec_t<3, wp::float32>> mesh_vert;
    wp::array_t<wp::int32> mesh_faceadr;
    wp::array_t<wp::vec_t<3, wp::int32>> mesh_face;
    wp::int32 data_id;
    wp::vec_t<3, wp::float32> pos;
    wp::mat_t<3, 3, wp::float32> mat;
    wp::vec_t<3, wp::float32> size;
    wp::vec_t<3, wp::float32> pnt;
    wp::vec_t<3, wp::float32> vec;
    bool valid;


    MeshData_52eaa0fa() = default;
    CUDA_CALLABLE MeshData_52eaa0fa(wp::int32 const& nmeshface,
    wp::array_t<wp::int32> const& mesh_vertadr = {},
    wp::array_t<wp::vec_t<3, wp::float32>> const& mesh_vert = {},
    wp::array_t<wp::int32> const& mesh_faceadr = {},
    wp::array_t<wp::vec_t<3, wp::int32>> const& mesh_face = {},
    wp::int32 const& data_id = {},
    wp::vec_t<3, wp::float32> const& pos = {},
    wp::mat_t<3, 3, wp::float32> const& mat = {},
    wp::vec_t<3, wp::float32> const& size = {},
    wp::vec_t<3, wp::float32> const& pnt = {},
    wp::vec_t<3, wp::float32> const& vec = {},
    bool const& valid = {})
        : nmeshface{nmeshface}
        , mesh_vertadr{mesh_vertadr}
        , mesh_vert{mesh_vert}
        , mesh_faceadr{mesh_faceadr}
        , mesh_face{mesh_face}
        , data_id{data_id}
        , pos{pos}
        , mat{mat}
        , size{size}
        , pnt{pnt}
        , vec{vec}
        , valid{valid}

    {
    }

    CUDA_CALLABLE MeshData_52eaa0fa& operator += (const MeshData_52eaa0fa& rhs)
    {    nmeshface += rhs.nmeshface;
    data_id += rhs.data_id;
    pos += rhs.pos;
    mat += rhs.mat;
    size += rhs.size;
    pnt += rhs.pnt;
    vec += rhs.vec;
    valid += rhs.valid;

        return *this;}

};

static CUDA_CALLABLE void adj_MeshData_52eaa0fa(wp::int32 const&,
    wp::array_t<wp::int32> const&,
    wp::array_t<wp::vec_t<3, wp::float32>> const&,
    wp::array_t<wp::int32> const&,
    wp::array_t<wp::vec_t<3, wp::int32>> const&,
    wp::int32 const&,
    wp::vec_t<3, wp::float32> const&,
    wp::mat_t<3, 3, wp::float32> const&,
    wp::vec_t<3, wp::float32> const&,
    wp::vec_t<3, wp::float32> const&,
    wp::vec_t<3, wp::float32> const&,
    bool const&,
    wp::int32 & adj_nmeshface,
    wp::array_t<wp::int32> & adj_mesh_vertadr,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_mesh_vert,
    wp::array_t<wp::int32> & adj_mesh_faceadr,
    wp::array_t<wp::vec_t<3, wp::int32>> & adj_mesh_face,
    wp::int32 & adj_data_id,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::vec_t<3, wp::float32> & adj_pnt,
    wp::vec_t<3, wp::float32> & adj_vec,
    bool & adj_valid,
    MeshData_52eaa0fa & adj_ret)
{
    adj_nmeshface += adj_ret.nmeshface;
    adj_mesh_vertadr = adj_ret.mesh_vertadr;
    adj_mesh_vert = adj_ret.mesh_vert;
    adj_mesh_faceadr = adj_ret.mesh_faceadr;
    adj_mesh_face = adj_ret.mesh_face;
    adj_data_id += adj_ret.data_id;
    adj_pos += adj_ret.pos;
    adj_mat += adj_ret.mat;
    adj_size += adj_ret.size;
    adj_pnt += adj_ret.pnt;
    adj_vec += adj_ret.vec;
    adj_valid += adj_ret.valid;
}

// Required when compiling adjoints.
CUDA_CALLABLE MeshData_52eaa0fa add(const MeshData_52eaa0fa& a, const MeshData_52eaa0fa& b)
{
    return MeshData_52eaa0fa();
}

CUDA_CALLABLE void adj_atomic_add(MeshData_52eaa0fa* p, MeshData_52eaa0fa t)
{
    wp::adj_atomic_add(&p->nmeshface, t.nmeshface);
    wp::adj_atomic_add(&p->mesh_vertadr, t.mesh_vertadr);
    wp::adj_atomic_add(&p->mesh_vert, t.mesh_vert);
    wp::adj_atomic_add(&p->mesh_faceadr, t.mesh_faceadr);
    wp::adj_atomic_add(&p->mesh_face, t.mesh_face);
    wp::adj_atomic_add(&p->data_id, t.data_id);
    wp::adj_atomic_add(&p->pos, t.pos);
    wp::adj_atomic_add(&p->mat, t.mat);
    wp::adj_atomic_add(&p->size, t.size);
    wp::adj_atomic_add(&p->pnt, t.pnt);
    wp::adj_atomic_add(&p->vec, t.vec);
    wp::adj_atomic_add(&p->valid, t.valid);
}



// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:113
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _magnetometer_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_opt_magnetic,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::shape_t* var_0;
    const wp::int32 var_1 = 0;
    wp::int32 var_2;
    wp::shape_t var_3;
    wp::int32 var_4;
    wp::vec_t<3, wp::float32>* var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::mat_t<3, 3, wp::float32>* var_8;
    wp::mat_t<3, 3, wp::float32> var_9;
    wp::mat_t<3, 3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    //---------
    // forward
    // def _magnetometer(                                                                     <L 114>
    // magnetic = opt_magnetic[worldid % opt_magnetic.shape[0]]                               <L 123>
    var_0 = &(var_opt_magnetic.shape);
    var_3 = wp::load(var_0);
    var_2 = wp::extract(var_3, var_1);
    var_4 = wp::mod(var_worldid, var_2);
    var_5 = wp::address(var_opt_magnetic, var_4);
    var_7 = wp::load(var_5);
    var_6 = wp::copy(var_7);
    // return wp.transpose(site_xmat_in[worldid, objid]) @ magnetic                           <L 124>
    var_8 = wp::address(var_site_xmat_in, var_worldid, var_objid);
    var_10 = wp::load(var_8);
    var_9 = wp::transpose(var_10);
    var_11 = wp::mul(var_9, var_6);
    return var_11;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:0
static CUDA_CALLABLE void _write_vector_0(
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::int32 var_sensorid,
    wp::int32 var_sensordim,
    wp::vec_t<3, wp::float32> var_sensor,
    wp::array_t<wp::float32> var_out)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    wp::float32* var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    const wp::float32 var_6 = 0.0;
    bool var_7;
    wp::int32* var_8;
    const wp::int32 var_9 = 41;
    const wp::int32 var_10 = 41;
    wp::int32 var_11;
    bool var_12;
    wp::int32 var_13;
    bool var_14;
    bool var_15;
    wp::int32* var_16;
    wp::int32 var_17;
    wp::int32 var_18;
    const wp::int32 var_19 = 0;
    bool var_20;
    wp::range_t var_21;
    wp::int32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::int32 var_26;
    const wp::int32 var_27 = 1;
    bool var_28;
    wp::range_t var_29;
    wp::int32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::int32 var_33;
    wp::int32 var_34;
    wp::int32 var_35;
    wp::range_t var_36;
    wp::int32 var_37;
    wp::float32 var_38;
    wp::int32 var_39;
    //---------
    // forward
    // def _write_vector(                                                                     <L 1>
    // adr = sensor_adr[sensorid]                                                             <L 14>
    var_0 = wp::address(var_sensor_adr, var_sensorid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // cutoff = sensor_cutoff[sensorid]                                                       <L 15>
    var_3 = wp::address(var_sensor_cutoff, var_sensorid);
    var_5 = wp::load(var_3);
    var_4 = wp::copy(var_5);
    // if cutoff > 0.0 and not (sensor_type[sensorid] == int(SensorType.GEOMFROMTO.value)):       <L 17>
    var_7 = (var_4 > var_6);
    var_8 = wp::address(var_sensor_type, var_sensorid);
    var_11 = wp::int(var_10);
    var_13 = wp::load(var_8);
    var_12 = (var_13 == var_11);
    var_14 = wp::unot(var_12);
    var_15 = var_7 && var_14;
    if (var_15) {
        // datatype = sensor_datatype[sensorid]                                               <L 18>
        var_16 = wp::address(var_sensor_datatype, var_sensorid);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // if datatype == DataType.REAL:                                                      <L 19>
        var_20 = (var_17 == var_19);
        if (var_20) {
            // for i in range(sensordim):                                                     <L 20>
            var_21 = wp::range(var_sensordim);
            start_for_0:;
                if (iter_cmp(var_21) == 0) goto end_for_0;
                var_22 = wp::iter_next(var_21);
                // out[adr + i] = wp.clamp(sensor[i], -cutoff, cutoff)                        <L 21>
                var_23 = wp::extract(var_sensor, var_22);
                var_24 = wp::neg(var_4);
                var_25 = wp::clamp(var_23, var_24, var_4);
                var_26 = wp::add(var_1, var_22);
                wp::array_store(var_out, var_26, var_25);
                goto start_for_0;
            end_for_0:;
            // return                                                                         <L 22>
            return;
        }
        if (!var_20) {
            // elif datatype == DataType.POSITIVE:                                            <L 23>
            var_28 = (var_17 == var_27);
            if (var_28) {
                // for i in range(sensordim):                                                 <L 24>
                var_29 = wp::range(var_sensordim);
                start_for_3:;
                    if (iter_cmp(var_29) == 0) goto end_for_3;
                    var_30 = wp::iter_next(var_29);
                    // out[adr + i] = wp.min(sensor[i], cutoff)                               <L 25>
                    var_31 = wp::extract(var_sensor, var_30);
                    var_32 = wp::min(var_31, var_4);
                    var_33 = wp::add(var_1, var_30);
                    wp::array_store(var_out, var_33, var_32);
                    goto start_for_3;
                end_for_3:;
                // return                                                                     <L 26>
                return;
            }
            var_34 = wp::where(var_28, var_30, var_22);
        }
        var_35 = wp::where(var_20, var_22, var_34);
    }
    // for i in range(sensordim):                                                             <L 28>
    var_36 = wp::range(var_sensordim);
    start_for_6:;
        if (iter_cmp(var_36) == 0) goto end_for_6;
        var_37 = wp::iter_next(var_36);
        // out[adr + i] = sensor[i]                                                           <L 29>
        var_38 = wp::extract(var_sensor, var_37);
        var_39 = wp::add(var_1, var_37);
        wp::array_store(var_out, var_39, var_38);
        goto start_for_6;
    end_for_6:;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:127
static CUDA_CALLABLE wp::vec_t<2, wp::float32> _cam_projection_0(
    wp::array_t<wp::float32> var_cam_fovy,
    wp::array_t<wp::vec_t<2, wp::int32>> var_cam_resolution,
    wp::array_t<wp::vec_t<2, wp::float32>> var_cam_sensorsize,
    wp::array_t<wp::vec_t<4, wp::float32>> var_cam_intrinsic,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::int32 var_refid)
{
    //---------
    // primal vars
    wp::vec_t<2, wp::float32>* var_0;
    wp::vec_t<2, wp::float32> var_1;
    wp::vec_t<2, wp::float32> var_2;
    wp::shape_t* var_3;
    const wp::int32 var_4 = 0;
    wp::int32 var_5;
    wp::shape_t var_6;
    wp::int32 var_7;
    wp::vec_t<4, wp::float32>* var_8;
    wp::vec_t<4, wp::float32> var_9;
    wp::vec_t<4, wp::float32> var_10;
    wp::shape_t* var_11;
    const wp::int32 var_12 = 0;
    wp::int32 var_13;
    wp::shape_t var_14;
    wp::int32 var_15;
    wp::float32* var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::vec_t<2, wp::int32>* var_19;
    wp::vec_t<2, wp::int32> var_20;
    wp::vec_t<2, wp::int32> var_21;
    wp::vec_t<3, wp::float32>* var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32>* var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::mat_t<3, 3, wp::float32>* var_28;
    wp::mat_t<3, 3, wp::float32> var_29;
    wp::mat_t<3, 3, wp::float32> var_30;
    const wp::float32 var_31 = 1.0;
    const wp::float32 var_32 = 0.0;
    const wp::float32 var_33 = 0.0;
    const wp::int32 var_34 = 0;
    wp::float32 var_35;
    wp::float32 var_36;
    const wp::float32 var_37 = 0.0;
    const wp::float32 var_38 = 1.0;
    const wp::float32 var_39 = 0.0;
    const wp::int32 var_40 = 1;
    wp::float32 var_41;
    wp::float32 var_42;
    const wp::float32 var_43 = 0.0;
    const wp::float32 var_44 = 0.0;
    const wp::float32 var_45 = 1.0;
    const wp::int32 var_46 = 2;
    wp::float32 var_47;
    wp::float32 var_48;
    const wp::float32 var_49 = 0.0;
    const wp::float32 var_50 = 0.0;
    const wp::float32 var_51 = 0.0;
    const wp::float32 var_52 = 1.0;
    wp::mat_t<4, 4, wp::float32> var_53;
    const wp::int32 var_54 = 0;
    const wp::int32 var_55 = 0;
    wp::float32 var_56;
    const wp::int32 var_57 = 1;
    const wp::int32 var_58 = 0;
    wp::float32 var_59;
    const wp::int32 var_60 = 2;
    const wp::int32 var_61 = 0;
    wp::float32 var_62;
    const wp::float32 var_63 = 0.0;
    const wp::int32 var_64 = 0;
    const wp::int32 var_65 = 1;
    wp::float32 var_66;
    const wp::int32 var_67 = 1;
    const wp::int32 var_68 = 1;
    wp::float32 var_69;
    const wp::int32 var_70 = 2;
    const wp::int32 var_71 = 1;
    wp::float32 var_72;
    const wp::float32 var_73 = 0.0;
    const wp::int32 var_74 = 0;
    const wp::int32 var_75 = 2;
    wp::float32 var_76;
    const wp::int32 var_77 = 1;
    const wp::int32 var_78 = 2;
    wp::float32 var_79;
    const wp::int32 var_80 = 2;
    const wp::int32 var_81 = 2;
    wp::float32 var_82;
    const wp::float32 var_83 = 0.0;
    const wp::float32 var_84 = 0.0;
    const wp::float32 var_85 = 0.0;
    const wp::float32 var_86 = 0.0;
    const wp::float32 var_87 = 1.0;
    wp::mat_t<4, 4, wp::float32> var_88;
    const wp::int32 var_89 = 0;
    wp::float32 var_90;
    const wp::float32 var_91 = 0.0;
    bool var_92;
    const wp::int32 var_93 = 1;
    wp::float32 var_94;
    const wp::float32 var_95 = 0.0;
    bool var_96;
    bool var_97;
    const wp::int32 var_98 = 0;
    wp::float32 var_99;
    const wp::int32 var_100 = 0;
    wp::float32 var_101;
    const wp::float32 var_102 = 1e-15;
    wp::float32 var_103;
    wp::float32 var_104;
    const wp::int32 var_105 = 0;
    wp::int32 var_106;
    wp::float32 var_107;
    wp::float32 var_108;
    const wp::int32 var_109 = 1;
    wp::float32 var_110;
    const wp::int32 var_111 = 1;
    wp::float32 var_112;
    wp::float32 var_113;
    wp::float32 var_114;
    const wp::int32 var_115 = 1;
    wp::int32 var_116;
    wp::float32 var_117;
    wp::float32 var_118;
    wp::float32 var_119;
    const wp::float32 var_120 = 0.0;
    const wp::float32 var_121 = 0.0;
    const wp::float32 var_122 = 0.0;
    const wp::float32 var_123 = 0.0;
    const wp::float32 var_124 = 0.0;
    const wp::float32 var_125 = 0.0;
    const wp::float32 var_126 = 0.0;
    const wp::float32 var_127 = 0.0;
    const wp::float32 var_128 = 1.0;
    const wp::float32 var_129 = 0.0;
    const wp::float32 var_130 = 0.0;
    const wp::float32 var_131 = 0.0;
    const wp::float32 var_132 = 0.0;
    const wp::float32 var_133 = 0.0;
    wp::mat_t<4, 4, wp::float32> var_134;
    const wp::float32 var_135 = 0.5;
    const wp::float32 var_136 = 0.008726646259971648;
    wp::float32 var_137;
    wp::float32 var_138;
    wp::float32 var_139;
    const wp::int32 var_140 = 1;
    wp::int32 var_141;
    wp::float32 var_142;
    wp::float32 var_143;
    wp::float32 var_144;
    const wp::float32 var_145 = 0.0;
    const wp::float32 var_146 = 0.0;
    const wp::float32 var_147 = 0.0;
    const wp::float32 var_148 = 0.0;
    const wp::float32 var_149 = 0.0;
    const wp::float32 var_150 = 0.0;
    const wp::float32 var_151 = 0.0;
    const wp::float32 var_152 = 0.0;
    const wp::float32 var_153 = 1.0;
    const wp::float32 var_154 = 0.0;
    const wp::float32 var_155 = 0.0;
    const wp::float32 var_156 = 0.0;
    const wp::float32 var_157 = 0.0;
    const wp::float32 var_158 = 0.0;
    wp::mat_t<4, 4, wp::float32> var_159;
    wp::mat_t<4, 4, wp::float32> var_160;
    const wp::float32 var_161 = 1.0;
    const wp::float32 var_162 = 0.0;
    const wp::float32 var_163 = 0.5;
    const wp::int32 var_164 = 0;
    wp::int32 var_165;
    wp::float32 var_166;
    wp::float32 var_167;
    const wp::float32 var_168 = 0.0;
    const wp::float32 var_169 = 0.0;
    const wp::float32 var_170 = 1.0;
    const wp::float32 var_171 = 0.5;
    const wp::int32 var_172 = 1;
    wp::int32 var_173;
    wp::float32 var_174;
    wp::float32 var_175;
    const wp::float32 var_176 = 0.0;
    const wp::float32 var_177 = 0.0;
    const wp::float32 var_178 = 0.0;
    const wp::float32 var_179 = 1.0;
    const wp::float32 var_180 = 0.0;
    const wp::float32 var_181 = 0.0;
    const wp::float32 var_182 = 0.0;
    const wp::float32 var_183 = 0.0;
    const wp::float32 var_184 = 0.0;
    wp::mat_t<4, 4, wp::float32> var_185;
    wp::mat_t<4, 4, wp::float32> var_186;
    wp::mat_t<4, 4, wp::float32> var_187;
    wp::mat_t<4, 4, wp::float32> var_188;
    const wp::int32 var_189 = 0;
    wp::float32 var_190;
    const wp::int32 var_191 = 1;
    wp::float32 var_192;
    const wp::int32 var_193 = 2;
    wp::float32 var_194;
    const wp::float32 var_195 = 1.0;
    wp::vec_t<4, wp::float32> var_196;
    wp::vec_t<4, wp::float32> var_197;
    const wp::int32 var_198 = 2;
    wp::float32 var_199;
    wp::float32 var_200;
    bool var_201;
    const wp::float32 var_202 = -1e-15;
    wp::float32 var_203;
    wp::float32 var_204;
    const wp::int32 var_205 = 0;
    wp::float32 var_206;
    const wp::int32 var_207 = 1;
    wp::float32 var_208;
    wp::vec_t<2, wp::float32> var_209;
    wp::vec_t<2, wp::float32> var_210;
    //---------
    // forward
    // def _cam_projection(                                                                   <L 128>
    // sensorsize = cam_sensorsize[refid]                                                     <L 143>
    var_0 = wp::address(var_cam_sensorsize, var_refid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // intrinsic = cam_intrinsic[worldid % cam_intrinsic.shape[0], refid]                     <L 144>
    var_3 = &(var_cam_intrinsic.shape);
    var_6 = wp::load(var_3);
    var_5 = wp::extract(var_6, var_4);
    var_7 = wp::mod(var_worldid, var_5);
    var_8 = wp::address(var_cam_intrinsic, var_7, var_refid);
    var_10 = wp::load(var_8);
    var_9 = wp::copy(var_10);
    // fovy = cam_fovy[worldid % cam_fovy.shape[0], refid]                                    <L 145>
    var_11 = &(var_cam_fovy.shape);
    var_14 = wp::load(var_11);
    var_13 = wp::extract(var_14, var_12);
    var_15 = wp::mod(var_worldid, var_13);
    var_16 = wp::address(var_cam_fovy, var_15, var_refid);
    var_18 = wp::load(var_16);
    var_17 = wp::copy(var_18);
    // res = cam_resolution[refid]                                                            <L 146>
    var_19 = wp::address(var_cam_resolution, var_refid);
    var_21 = wp::load(var_19);
    var_20 = wp::copy(var_21);
    // target_xpos = site_xpos_in[worldid, objid]                                             <L 148>
    var_22 = wp::address(var_site_xpos_in, var_worldid, var_objid);
    var_24 = wp::load(var_22);
    var_23 = wp::copy(var_24);
    // xpos = cam_xpos_in[worldid, refid]                                                     <L 149>
    var_25 = wp::address(var_cam_xpos_in, var_worldid, var_refid);
    var_27 = wp::load(var_25);
    var_26 = wp::copy(var_27);
    // xmat = cam_xmat_in[worldid, refid]                                                     <L 150>
    var_28 = wp::address(var_cam_xmat_in, var_worldid, var_refid);
    var_30 = wp::load(var_28);
    var_29 = wp::copy(var_30);
    // translation = wp.mat44f(1.0, 0.0, 0.0, -xpos[0], 0.0, 1.0, 0.0, -xpos[1], 0.0, 0.0, 1.0, -xpos[2], 0.0, 0.0, 0.0, 1.0)       <L 152>
    var_35 = wp::extract(var_26, var_34);
    var_36 = wp::neg(var_35);
    var_41 = wp::extract(var_26, var_40);
    var_42 = wp::neg(var_41);
    var_47 = wp::extract(var_26, var_46);
    var_48 = wp::neg(var_47);
    var_53 = wp::mat_t<4, 4, wp::float32>(var_31, var_32, var_33, var_36, var_37, var_38, var_39, var_42, var_43, var_44, var_45, var_48, var_49, var_50, var_51, var_52);
    // rotation = wp.mat44f(                                                                  <L 153>
    // xmat[0, 0], xmat[1, 0], xmat[2, 0], 0.0,                                               <L 154>
    var_56 = wp::extract(var_29, var_54, var_55);
    var_59 = wp::extract(var_29, var_57, var_58);
    var_62 = wp::extract(var_29, var_60, var_61);
    // xmat[0, 1], xmat[1, 1], xmat[2, 1], 0.0,                                               <L 155>
    var_66 = wp::extract(var_29, var_64, var_65);
    var_69 = wp::extract(var_29, var_67, var_68);
    var_72 = wp::extract(var_29, var_70, var_71);
    // xmat[0, 2], xmat[1, 2], xmat[2, 2], 0.0,                                               <L 156>
    var_76 = wp::extract(var_29, var_74, var_75);
    var_79 = wp::extract(var_29, var_77, var_78);
    var_82 = wp::extract(var_29, var_80, var_81);
    // 0.0, 0.0, 0.0, 1.0,                                                                    <L 157>
    var_88 = wp::mat_t<4, 4, wp::float32>(var_56, var_59, var_62, var_63, var_66, var_69, var_72, var_73, var_76, var_79, var_82, var_83, var_84, var_85, var_86, var_87);
    // if sensorsize[0] != 0.0 and sensorsize[1] != 0.0:                                      <L 161>
    var_90 = wp::extract(var_1, var_89);
    var_92 = (var_90 != var_91);
    var_94 = wp::extract(var_1, var_93);
    var_96 = (var_94 != var_95);
    var_97 = var_92 && var_96;
    if (var_97) {
        // fx = intrinsic[0] / (sensorsize[0] + MJ_MINVAL) * float(res[0])                    <L 162>
        var_99 = wp::extract(var_9, var_98);
        var_101 = wp::extract(var_1, var_100);
        var_103 = wp::add(var_101, var_102);
        var_104 = wp::div(var_99, var_103);
        var_106 = wp::extract(var_20, var_105);
        var_107 = wp::float(var_106);
        var_108 = wp::mul(var_104, var_107);
        // fy = intrinsic[1] / (sensorsize[1] + MJ_MINVAL) * float(res[1])                    <L 163>
        var_110 = wp::extract(var_9, var_109);
        var_112 = wp::extract(var_1, var_111);
        var_113 = wp::add(var_112, var_102);
        var_114 = wp::div(var_110, var_113);
        var_116 = wp::extract(var_20, var_115);
        var_117 = wp::float(var_116);
        var_118 = wp::mul(var_114, var_117);
        // focal = wp.mat44f(-fx, 0.0, 0.0, 0.0, 0.0, fy, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)       <L 164>
        var_119 = wp::neg(var_108);
        var_134 = wp::mat_t<4, 4, wp::float32>(var_119, var_120, var_121, var_122, var_123, var_118, var_124, var_125, var_126, var_127, var_128, var_129, var_130, var_131, var_132, var_133);
    }
    if (!var_97) {
        // f = 0.5 / wp.tan(fovy * wp.static(wp.pi / 360.0)) * float(res[1])                  <L 166>
        var_137 = wp::mul(var_17, var_136);
        var_138 = wp::tan(var_137);
        var_139 = wp::div(var_135, var_138);
        var_141 = wp::extract(var_20, var_140);
        var_142 = wp::float(var_141);
        var_143 = wp::mul(var_139, var_142);
        // focal = wp.mat44f(-f, 0.0, 0.0, 0.0, 0.0, f, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)       <L 167>
        var_144 = wp::neg(var_143);
        var_159 = wp::mat_t<4, 4, wp::float32>(var_144, var_145, var_146, var_147, var_148, var_143, var_149, var_150, var_151, var_152, var_153, var_154, var_155, var_156, var_157, var_158);
    }
    var_160 = wp::where(var_97, var_134, var_159);
    // image = wp.mat44f(                                                                     <L 170>
    // 1.0, 0.0, 0.5 * float(res[0]), 0.0, 0.0, 1.0, 0.5 * float(res[1]), 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0       <L 171>
    var_165 = wp::extract(var_20, var_164);
    var_166 = wp::float(var_165);
    var_167 = wp::mul(var_163, var_166);
    var_173 = wp::extract(var_20, var_172);
    var_174 = wp::float(var_173);
    var_175 = wp::mul(var_171, var_174);
    var_185 = wp::mat_t<4, 4, wp::float32>(var_161, var_162, var_167, var_168, var_169, var_170, var_175, var_176, var_177, var_178, var_179, var_180, var_181, var_182, var_183, var_184);
    // proj = image @ focal @ rotation @ translation                                          <L 176>
    var_186 = wp::mul(var_185, var_160);
    var_187 = wp::mul(var_186, var_88);
    var_188 = wp::mul(var_187, var_53);
    // pos_hom = wp.vec4(target_xpos[0], target_xpos[1], target_xpos[2], 1.0)                 <L 179>
    var_190 = wp::extract(var_23, var_189);
    var_192 = wp::extract(var_23, var_191);
    var_194 = wp::extract(var_23, var_193);
    var_196 = wp::vec_t<4, wp::float32>(var_190, var_192, var_194, var_195);
    // pixel_coord_hom = proj @ pos_hom                                                       <L 183>
    var_197 = wp::mul(var_188, var_196);
    // denom = pixel_coord_hom[2]                                                             <L 186>
    var_199 = wp::extract(var_197, var_198);
    // if wp.abs(denom) < MJ_MINVAL:                                                          <L 187>
    var_200 = wp::abs(var_199);
    var_201 = (var_200 < var_102);
    if (var_201) {
        // denom = wp.clamp(denom, -MJ_MINVAL, MJ_MINVAL)                                     <L 188>
        var_203 = wp::clamp(var_199, var_202, var_102);
    }
    var_204 = wp::where(var_201, var_203, var_199);
    // return wp.vec2f(pixel_coord_hom[0], pixel_coord_hom[1]) / denom                        <L 191>
    var_206 = wp::extract(var_197, var_205);
    var_208 = wp::extract(var_197, var_207);
    var_209 = wp::vec_t<2, wp::float32>(var_206, var_208);
    var_210 = wp::div(var_209, var_204);
    return var_210;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:0
static CUDA_CALLABLE void _write_vector_0(
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::int32 var_sensorid,
    wp::int32 var_sensordim,
    wp::vec_t<2, wp::float32> var_sensor,
    wp::array_t<wp::float32> var_out)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    wp::float32* var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    const wp::float32 var_6 = 0.0;
    bool var_7;
    wp::int32* var_8;
    const wp::int32 var_9 = 41;
    const wp::int32 var_10 = 41;
    wp::int32 var_11;
    bool var_12;
    wp::int32 var_13;
    bool var_14;
    bool var_15;
    wp::int32* var_16;
    wp::int32 var_17;
    wp::int32 var_18;
    const wp::int32 var_19 = 0;
    bool var_20;
    wp::range_t var_21;
    wp::int32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::int32 var_26;
    const wp::int32 var_27 = 1;
    bool var_28;
    wp::range_t var_29;
    wp::int32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::int32 var_33;
    wp::int32 var_34;
    wp::int32 var_35;
    wp::range_t var_36;
    wp::int32 var_37;
    wp::float32 var_38;
    wp::int32 var_39;
    //---------
    // forward
    // def _write_vector(                                                                     <L 1>
    // adr = sensor_adr[sensorid]                                                             <L 14>
    var_0 = wp::address(var_sensor_adr, var_sensorid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // cutoff = sensor_cutoff[sensorid]                                                       <L 15>
    var_3 = wp::address(var_sensor_cutoff, var_sensorid);
    var_5 = wp::load(var_3);
    var_4 = wp::copy(var_5);
    // if cutoff > 0.0 and not (sensor_type[sensorid] == int(SensorType.GEOMFROMTO.value)):       <L 17>
    var_7 = (var_4 > var_6);
    var_8 = wp::address(var_sensor_type, var_sensorid);
    var_11 = wp::int(var_10);
    var_13 = wp::load(var_8);
    var_12 = (var_13 == var_11);
    var_14 = wp::unot(var_12);
    var_15 = var_7 && var_14;
    if (var_15) {
        // datatype = sensor_datatype[sensorid]                                               <L 18>
        var_16 = wp::address(var_sensor_datatype, var_sensorid);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // if datatype == DataType.REAL:                                                      <L 19>
        var_20 = (var_17 == var_19);
        if (var_20) {
            // for i in range(sensordim):                                                     <L 20>
            var_21 = wp::range(var_sensordim);
            start_for_0:;
                if (iter_cmp(var_21) == 0) goto end_for_0;
                var_22 = wp::iter_next(var_21);
                // out[adr + i] = wp.clamp(sensor[i], -cutoff, cutoff)                        <L 21>
                var_23 = wp::extract(var_sensor, var_22);
                var_24 = wp::neg(var_4);
                var_25 = wp::clamp(var_23, var_24, var_4);
                var_26 = wp::add(var_1, var_22);
                wp::array_store(var_out, var_26, var_25);
                goto start_for_0;
            end_for_0:;
            // return                                                                         <L 22>
            return;
        }
        if (!var_20) {
            // elif datatype == DataType.POSITIVE:                                            <L 23>
            var_28 = (var_17 == var_27);
            if (var_28) {
                // for i in range(sensordim):                                                 <L 24>
                var_29 = wp::range(var_sensordim);
                start_for_3:;
                    if (iter_cmp(var_29) == 0) goto end_for_3;
                    var_30 = wp::iter_next(var_29);
                    // out[adr + i] = wp.min(sensor[i], cutoff)                               <L 25>
                    var_31 = wp::extract(var_sensor, var_30);
                    var_32 = wp::min(var_31, var_4);
                    var_33 = wp::add(var_1, var_30);
                    wp::array_store(var_out, var_33, var_32);
                    goto start_for_3;
                end_for_3:;
                // return                                                                     <L 26>
                return;
            }
            var_34 = wp::where(var_28, var_30, var_22);
        }
        var_35 = wp::where(var_20, var_22, var_34);
    }
    // for i in range(sensordim):                                                             <L 28>
    var_36 = wp::range(var_sensordim);
    start_for_6:;
        if (iter_cmp(var_36) == 0) goto end_for_6;
        var_37 = wp::iter_next(var_36);
        // out[adr + i] = sensor[i]                                                           <L 29>
        var_38 = wp::extract(var_sensor, var_37);
        var_39 = wp::add(var_1, var_37);
        wp::array_store(var_out, var_39, var_38);
        goto start_for_6;
    end_for_6:;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:0
static CUDA_CALLABLE void _write_scalar_0(
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::int32 var_sensorid,
    wp::float32 var_sensor,
    wp::array_t<wp::float32> var_out)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    wp::float32* var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    const wp::float32 var_6 = 0.0;
    bool var_7;
    wp::int32* var_8;
    const wp::int32 var_9 = 41;
    const wp::int32 var_10 = 41;
    wp::int32 var_11;
    bool var_12;
    wp::int32 var_13;
    bool var_14;
    bool var_15;
    wp::int32* var_16;
    wp::int32 var_17;
    wp::int32 var_18;
    const wp::int32 var_19 = 0;
    bool var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::int32 var_23 = 1;
    bool var_24;
    wp::float32 var_25;
    //---------
    // forward
    // def _write_scalar(                                                                     <L 1>
    // adr = sensor_adr[sensorid]                                                             <L 13>
    var_0 = wp::address(var_sensor_adr, var_sensorid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // cutoff = sensor_cutoff[sensorid]                                                       <L 14>
    var_3 = wp::address(var_sensor_cutoff, var_sensorid);
    var_5 = wp::load(var_3);
    var_4 = wp::copy(var_5);
    // if cutoff > 0.0 and not (sensor_type[sensorid] == int(SensorType.GEOMFROMTO.value)):       <L 16>
    var_7 = (var_4 > var_6);
    var_8 = wp::address(var_sensor_type, var_sensorid);
    var_11 = wp::int(var_10);
    var_13 = wp::load(var_8);
    var_12 = (var_13 == var_11);
    var_14 = wp::unot(var_12);
    var_15 = var_7 && var_14;
    if (var_15) {
        // datatype = sensor_datatype[sensorid]                                               <L 17>
        var_16 = wp::address(var_sensor_datatype, var_sensorid);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // if datatype == DataType.REAL:                                                      <L 18>
        var_20 = (var_17 == var_19);
        if (var_20) {
            // out[adr] = wp.clamp(sensor, -cutoff, cutoff)                                   <L 19>
            var_21 = wp::neg(var_4);
            var_22 = wp::clamp(var_sensor, var_21, var_4);
            wp::array_store(var_out, var_1, var_22);
            // return                                                                         <L 20>
            return;
        }
        if (!var_20) {
            // elif datatype == DataType.POSITIVE:                                            <L 21>
            var_24 = (var_17 == var_23);
            if (var_24) {
                // out[adr] = wp.min(sensor, cutoff)                                          <L 22>
                var_25 = wp::min(var_sensor, var_4);
                wp::array_store(var_out, var_1, var_25);
                // return                                                                     <L 23>
                return;
            }
        }
    }
    // out[adr] = sensor                                                                      <L 25>
    wp::array_store(var_out, var_1, var_sensor);
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:216
static CUDA_CALLABLE wp::float32 _joint_pos_0(
    wp::array_t<wp::int32> var_jnt_qposadr,
    wp::array_t<wp::float32> var_qpos_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::float32* var_1;
    wp::int32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    //---------
    // forward
    // def _joint_pos(jnt_qposadr: wp.array[int], qpos_in: wp.array2d[float], worldid: int, objid: int) -> float:       <L 217>
    // return qpos_in[worldid, jnt_qposadr[objid]]                                            <L 218>
    var_0 = wp::address(var_jnt_qposadr, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::address(var_qpos_in, var_worldid, var_2);
    var_4 = wp::load(var_1);
    var_3 = wp::copy(var_4);
    return var_3;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:221
static CUDA_CALLABLE wp::float32 _tendon_pos_0(
    wp::array_t<wp::float32> var_ten_length_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::float32* var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    //---------
    // forward
    // def _tendon_pos(ten_length_in: wp.array2d[float], worldid: int, objid: int) -> float:       <L 222>
    // return ten_length_in[worldid, objid]                                                   <L 223>
    var_0 = wp::address(var_ten_length_in, var_worldid, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    return var_1;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:226
static CUDA_CALLABLE wp::float32 _actuator_pos_0(
    wp::array_t<wp::float32> var_actuator_length_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::float32* var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    //---------
    // forward
    // def _actuator_pos(actuator_length_in: wp.array2d[float], worldid: int, objid: int) -> float:       <L 227>
    // return actuator_length_in[worldid, objid]                                              <L 228>
    var_0 = wp::address(var_actuator_length_in, var_worldid, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    return var_1;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:231
static CUDA_CALLABLE wp::quat_t<wp::float32> _ball_quat_0(
    wp::array_t<wp::int32> var_jnt_qposadr,
    wp::array_t<wp::float32> var_qpos_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    const wp::int32 var_3 = 0;
    wp::int32 var_4;
    wp::float32* var_5;
    const wp::int32 var_6 = 1;
    wp::int32 var_7;
    wp::float32* var_8;
    const wp::int32 var_9 = 2;
    wp::int32 var_10;
    wp::float32* var_11;
    const wp::int32 var_12 = 3;
    wp::int32 var_13;
    wp::float32* var_14;
    wp::quat_t<wp::float32> var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::quat_t<wp::float32> var_20;
    //---------
    // forward
    // def _ball_quat(jnt_qposadr: wp.array[int], qpos_in: wp.array2d[float], worldid: int, objid: int) -> wp.quat:       <L 232>
    // adr = jnt_qposadr[objid]                                                               <L 233>
    var_0 = wp::address(var_jnt_qposadr, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // quat = wp.quat(                                                                        <L 234>
    // qpos_in[worldid, adr + 0],                                                             <L 235>
    var_4 = wp::add(var_1, var_3);
    var_5 = wp::address(var_qpos_in, var_worldid, var_4);
    // qpos_in[worldid, adr + 1],                                                             <L 236>
    var_7 = wp::add(var_1, var_6);
    var_8 = wp::address(var_qpos_in, var_worldid, var_7);
    // qpos_in[worldid, adr + 2],                                                             <L 237>
    var_10 = wp::add(var_1, var_9);
    var_11 = wp::address(var_qpos_in, var_worldid, var_10);
    // qpos_in[worldid, adr + 3],                                                             <L 238>
    var_13 = wp::add(var_1, var_12);
    var_14 = wp::address(var_qpos_in, var_worldid, var_13);
    var_16 = wp::load(var_5);
    var_17 = wp::load(var_8);
    var_18 = wp::load(var_11);
    var_19 = wp::load(var_14);
    var_15 = wp::quat_t<wp::float32>(var_16, var_17, var_18, var_19);
    // return wp.normalize(quat)                                                              <L 240>
    var_20 = wp::normalize(var_15);
    return var_20;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:0
static CUDA_CALLABLE void _write_vector_0(
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::int32 var_sensorid,
    wp::int32 var_sensordim,
    wp::quat_t<wp::float32> var_sensor,
    wp::array_t<wp::float32> var_out)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    wp::float32* var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    const wp::float32 var_6 = 0.0;
    bool var_7;
    wp::int32* var_8;
    const wp::int32 var_9 = 41;
    const wp::int32 var_10 = 41;
    wp::int32 var_11;
    bool var_12;
    wp::int32 var_13;
    bool var_14;
    bool var_15;
    wp::int32* var_16;
    wp::int32 var_17;
    wp::int32 var_18;
    const wp::int32 var_19 = 0;
    bool var_20;
    wp::range_t var_21;
    wp::int32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::int32 var_26;
    const wp::int32 var_27 = 1;
    bool var_28;
    wp::range_t var_29;
    wp::int32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::int32 var_33;
    wp::int32 var_34;
    wp::int32 var_35;
    wp::range_t var_36;
    wp::int32 var_37;
    wp::float32 var_38;
    wp::int32 var_39;
    //---------
    // forward
    // def _write_vector(                                                                     <L 1>
    // adr = sensor_adr[sensorid]                                                             <L 14>
    var_0 = wp::address(var_sensor_adr, var_sensorid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // cutoff = sensor_cutoff[sensorid]                                                       <L 15>
    var_3 = wp::address(var_sensor_cutoff, var_sensorid);
    var_5 = wp::load(var_3);
    var_4 = wp::copy(var_5);
    // if cutoff > 0.0 and not (sensor_type[sensorid] == int(SensorType.GEOMFROMTO.value)):       <L 17>
    var_7 = (var_4 > var_6);
    var_8 = wp::address(var_sensor_type, var_sensorid);
    var_11 = wp::int(var_10);
    var_13 = wp::load(var_8);
    var_12 = (var_13 == var_11);
    var_14 = wp::unot(var_12);
    var_15 = var_7 && var_14;
    if (var_15) {
        // datatype = sensor_datatype[sensorid]                                               <L 18>
        var_16 = wp::address(var_sensor_datatype, var_sensorid);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // if datatype == DataType.REAL:                                                      <L 19>
        var_20 = (var_17 == var_19);
        if (var_20) {
            // for i in range(sensordim):                                                     <L 20>
            var_21 = wp::range(var_sensordim);
            start_for_0:;
                if (iter_cmp(var_21) == 0) goto end_for_0;
                var_22 = wp::iter_next(var_21);
                // out[adr + i] = wp.clamp(sensor[i], -cutoff, cutoff)                        <L 21>
                var_23 = wp::extract(var_sensor, var_22);
                var_24 = wp::neg(var_4);
                var_25 = wp::clamp(var_23, var_24, var_4);
                var_26 = wp::add(var_1, var_22);
                wp::array_store(var_out, var_26, var_25);
                goto start_for_0;
            end_for_0:;
            // return                                                                         <L 22>
            return;
        }
        if (!var_20) {
            // elif datatype == DataType.POSITIVE:                                            <L 23>
            var_28 = (var_17 == var_27);
            if (var_28) {
                // for i in range(sensordim):                                                 <L 24>
                var_29 = wp::range(var_sensordim);
                start_for_3:;
                    if (iter_cmp(var_29) == 0) goto end_for_3;
                    var_30 = wp::iter_next(var_29);
                    // out[adr + i] = wp.min(sensor[i], cutoff)                               <L 25>
                    var_31 = wp::extract(var_sensor, var_30);
                    var_32 = wp::min(var_31, var_4);
                    var_33 = wp::add(var_1, var_30);
                    wp::array_store(var_out, var_33, var_32);
                    goto start_for_3;
                end_for_3:;
                // return                                                                     <L 26>
                return;
            }
            var_34 = wp::where(var_28, var_30, var_22);
        }
        var_35 = wp::where(var_20, var_22, var_34);
    }
    // for i in range(sensordim):                                                             <L 28>
    var_36 = wp::range(var_sensordim);
    start_for_6:;
        if (iter_cmp(var_36) == 0) goto end_for_6;
        var_37 = wp::iter_next(var_36);
        // out[adr + i] = sensor[i]                                                           <L 29>
        var_38 = wp::extract(var_sensor, var_37);
        var_39 = wp::add(var_1, var_37);
        wp::array_store(var_out, var_39, var_38);
        goto start_for_6;
    end_for_6:;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:281
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _frame_pos_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::int32 var_objtype,
    wp::int32 var_refid,
    wp::int32 var_reftype)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    bool var_1;
    wp::vec_t<3, wp::float32>* var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    const wp::int32 var_5 = 2;
    bool var_6;
    wp::vec_t<3, wp::float32>* var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    const wp::int32 var_11 = 5;
    bool var_12;
    wp::vec_t<3, wp::float32>* var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32> var_16;
    const wp::int32 var_17 = 6;
    bool var_18;
    wp::vec_t<3, wp::float32>* var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    const wp::int32 var_23 = 7;
    bool var_24;
    wp::vec_t<3, wp::float32>* var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    const wp::float32 var_29 = 0.0;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::vec_t<3, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    wp::vec_t<3, wp::float32> var_35;
    const wp::int32 var_36 = 1;
    const wp::int32 var_37 = -1;
    bool var_38;
    const wp::int32 var_39 = 1;
    bool var_40;
    wp::vec_t<3, wp::float32>* var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::mat_t<3, 3, wp::float32>* var_44;
    wp::mat_t<3, 3, wp::float32> var_45;
    wp::mat_t<3, 3, wp::float32> var_46;
    const wp::int32 var_47 = 2;
    bool var_48;
    wp::vec_t<3, wp::float32>* var_49;
    wp::vec_t<3, wp::float32> var_50;
    wp::vec_t<3, wp::float32> var_51;
    wp::mat_t<3, 3, wp::float32>* var_52;
    wp::mat_t<3, 3, wp::float32> var_53;
    wp::mat_t<3, 3, wp::float32> var_54;
    wp::vec_t<3, wp::float32> var_55;
    wp::mat_t<3, 3, wp::float32> var_56;
    const wp::int32 var_57 = 5;
    bool var_58;
    wp::vec_t<3, wp::float32>* var_59;
    wp::vec_t<3, wp::float32> var_60;
    wp::vec_t<3, wp::float32> var_61;
    wp::mat_t<3, 3, wp::float32>* var_62;
    wp::mat_t<3, 3, wp::float32> var_63;
    wp::mat_t<3, 3, wp::float32> var_64;
    wp::vec_t<3, wp::float32> var_65;
    wp::mat_t<3, 3, wp::float32> var_66;
    const wp::int32 var_67 = 6;
    bool var_68;
    wp::vec_t<3, wp::float32>* var_69;
    wp::vec_t<3, wp::float32> var_70;
    wp::vec_t<3, wp::float32> var_71;
    wp::mat_t<3, 3, wp::float32>* var_72;
    wp::mat_t<3, 3, wp::float32> var_73;
    wp::mat_t<3, 3, wp::float32> var_74;
    wp::vec_t<3, wp::float32> var_75;
    wp::mat_t<3, 3, wp::float32> var_76;
    const wp::int32 var_77 = 7;
    bool var_78;
    wp::vec_t<3, wp::float32>* var_79;
    wp::vec_t<3, wp::float32> var_80;
    wp::vec_t<3, wp::float32> var_81;
    wp::mat_t<3, 3, wp::float32>* var_82;
    wp::mat_t<3, 3, wp::float32> var_83;
    wp::mat_t<3, 3, wp::float32> var_84;
    wp::vec_t<3, wp::float32> var_85;
    wp::mat_t<3, 3, wp::float32> var_86;
    const wp::float32 var_87 = 0.0;
    wp::vec_t<3, wp::float32> var_88;
    const wp::int32 var_89 = 3;
    wp::mat_t<3, 3, wp::float32> var_90;
    wp::vec_t<3, wp::float32> var_91;
    wp::mat_t<3, 3, wp::float32> var_92;
    wp::vec_t<3, wp::float32> var_93;
    wp::mat_t<3, 3, wp::float32> var_94;
    wp::vec_t<3, wp::float32> var_95;
    wp::mat_t<3, 3, wp::float32> var_96;
    wp::vec_t<3, wp::float32> var_97;
    wp::mat_t<3, 3, wp::float32> var_98;
    wp::vec_t<3, wp::float32> var_99;
    wp::mat_t<3, 3, wp::float32> var_100;
    wp::mat_t<3, 3, wp::float32> var_101;
    wp::vec_t<3, wp::float32> var_102;
    wp::vec_t<3, wp::float32> var_103;
    //---------
    // forward
    // def _frame_pos(                                                                        <L 282>
    // if objtype == ObjType.BODY:                                                            <L 301>
    var_1 = (var_objtype == var_0);
    if (var_1) {
        // xpos = xipos_in[worldid, objid]                                                    <L 302>
        var_2 = wp::address(var_xipos_in, var_worldid, var_objid);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
    }
    if (!var_1) {
        // elif objtype == ObjType.XBODY:                                                     <L 303>
        var_6 = (var_objtype == var_5);
        if (var_6) {
            // xpos = xpos_in[worldid, objid]                                                 <L 304>
            var_7 = wp::address(var_xpos_in, var_worldid, var_objid);
            var_9 = wp::load(var_7);
            var_8 = wp::copy(var_9);
        }
        var_10 = wp::where(var_6, var_8, var_3);
        if (!var_6) {
            // elif objtype == ObjType.GEOM:                                                  <L 305>
            var_12 = (var_objtype == var_11);
            if (var_12) {
                // xpos = geom_xpos_in[worldid, objid]                                        <L 306>
                var_13 = wp::address(var_geom_xpos_in, var_worldid, var_objid);
                var_15 = wp::load(var_13);
                var_14 = wp::copy(var_15);
            }
            var_16 = wp::where(var_12, var_14, var_10);
            if (!var_12) {
                // elif objtype == ObjType.SITE:                                              <L 307>
                var_18 = (var_objtype == var_17);
                if (var_18) {
                    // xpos = site_xpos_in[worldid, objid]                                    <L 308>
                    var_19 = wp::address(var_site_xpos_in, var_worldid, var_objid);
                    var_21 = wp::load(var_19);
                    var_20 = wp::copy(var_21);
                }
                var_22 = wp::where(var_18, var_20, var_16);
                if (!var_18) {
                    // elif objtype == ObjType.CAMERA:                                        <L 309>
                    var_24 = (var_objtype == var_23);
                    if (var_24) {
                        // xpos = cam_xpos_in[worldid, objid]                                 <L 310>
                        var_25 = wp::address(var_cam_xpos_in, var_worldid, var_objid);
                        var_27 = wp::load(var_25);
                        var_26 = wp::copy(var_27);
                    }
                    var_28 = wp::where(var_24, var_26, var_22);
                    if (!var_24) {
                        // xpos = wp.vec3(0.0)                                                <L 312>
                        var_30 = wp::vec_t<3, wp::float32>(var_29);
                    }
                    var_31 = wp::where(var_24, var_28, var_30);
                }
                var_32 = wp::where(var_18, var_22, var_31);
            }
            var_33 = wp::where(var_12, var_16, var_32);
        }
        var_34 = wp::where(var_6, var_10, var_33);
    }
    var_35 = wp::where(var_1, var_3, var_34);
    // if refid == -1:                                                                        <L 314>
    var_38 = (var_refid == var_37);
    if (var_38) {
        // return xpos                                                                        <L 315>
        return var_35;
    }
    // if reftype == ObjType.BODY:                                                            <L 317>
    var_40 = (var_reftype == var_39);
    if (var_40) {
        // xpos_ref = xipos_in[worldid, refid]                                                <L 318>
        var_41 = wp::address(var_xipos_in, var_worldid, var_refid);
        var_43 = wp::load(var_41);
        var_42 = wp::copy(var_43);
        // xmat_ref = ximat_in[worldid, refid]                                                <L 319>
        var_44 = wp::address(var_ximat_in, var_worldid, var_refid);
        var_46 = wp::load(var_44);
        var_45 = wp::copy(var_46);
    }
    if (!var_40) {
        // elif objtype == ObjType.XBODY:                                                     <L 320>
        var_48 = (var_objtype == var_47);
        if (var_48) {
            // xpos_ref = xpos_in[worldid, refid]                                             <L 321>
            var_49 = wp::address(var_xpos_in, var_worldid, var_refid);
            var_51 = wp::load(var_49);
            var_50 = wp::copy(var_51);
            // xmat_ref = xmat_in[worldid, refid]                                             <L 322>
            var_52 = wp::address(var_xmat_in, var_worldid, var_refid);
            var_54 = wp::load(var_52);
            var_53 = wp::copy(var_54);
        }
        var_55 = wp::where(var_48, var_50, var_42);
        var_56 = wp::where(var_48, var_53, var_45);
        if (!var_48) {
            // elif reftype == ObjType.GEOM:                                                  <L 323>
            var_58 = (var_reftype == var_57);
            if (var_58) {
                // xpos_ref = geom_xpos_in[worldid, refid]                                    <L 324>
                var_59 = wp::address(var_geom_xpos_in, var_worldid, var_refid);
                var_61 = wp::load(var_59);
                var_60 = wp::copy(var_61);
                // xmat_ref = geom_xmat_in[worldid, refid]                                    <L 325>
                var_62 = wp::address(var_geom_xmat_in, var_worldid, var_refid);
                var_64 = wp::load(var_62);
                var_63 = wp::copy(var_64);
            }
            var_65 = wp::where(var_58, var_60, var_55);
            var_66 = wp::where(var_58, var_63, var_56);
            if (!var_58) {
                // elif reftype == ObjType.SITE:                                              <L 326>
                var_68 = (var_reftype == var_67);
                if (var_68) {
                    // xpos_ref = site_xpos_in[worldid, refid]                                <L 327>
                    var_69 = wp::address(var_site_xpos_in, var_worldid, var_refid);
                    var_71 = wp::load(var_69);
                    var_70 = wp::copy(var_71);
                    // xmat_ref = site_xmat_in[worldid, refid]                                <L 328>
                    var_72 = wp::address(var_site_xmat_in, var_worldid, var_refid);
                    var_74 = wp::load(var_72);
                    var_73 = wp::copy(var_74);
                }
                var_75 = wp::where(var_68, var_70, var_65);
                var_76 = wp::where(var_68, var_73, var_66);
                if (!var_68) {
                    // elif reftype == ObjType.CAMERA:                                        <L 329>
                    var_78 = (var_reftype == var_77);
                    if (var_78) {
                        // xpos_ref = cam_xpos_in[worldid, refid]                             <L 330>
                        var_79 = wp::address(var_cam_xpos_in, var_worldid, var_refid);
                        var_81 = wp::load(var_79);
                        var_80 = wp::copy(var_81);
                        // xmat_ref = cam_xmat_in[worldid, refid]                             <L 331>
                        var_82 = wp::address(var_cam_xmat_in, var_worldid, var_refid);
                        var_84 = wp::load(var_82);
                        var_83 = wp::copy(var_84);
                    }
                    var_85 = wp::where(var_78, var_80, var_75);
                    var_86 = wp::where(var_78, var_83, var_76);
                    if (!var_78) {
                        // xpos_ref = wp.vec3(0.0)                                            <L 334>
                        var_88 = wp::vec_t<3, wp::float32>(var_87);
                        // xmat_ref = wp.identity(3, wp.float32)                              <L 335>
                        var_90 = wp::identity<3, wp::float32>();
                    }
                    var_91 = wp::where(var_78, var_85, var_88);
                    var_92 = wp::where(var_78, var_86, var_90);
                }
                var_93 = wp::where(var_68, var_75, var_91);
                var_94 = wp::where(var_68, var_76, var_92);
            }
            var_95 = wp::where(var_58, var_65, var_93);
            var_96 = wp::where(var_58, var_66, var_94);
        }
        var_97 = wp::where(var_48, var_55, var_95);
        var_98 = wp::where(var_48, var_56, var_96);
    }
    var_99 = wp::where(var_40, var_42, var_97);
    var_100 = wp::where(var_40, var_45, var_98);
    // return wp.transpose(xmat_ref) @ (xpos - xpos_ref)                                      <L 337>
    var_101 = wp::transpose(var_100);
    var_102 = wp::sub(var_35, var_99);
    var_103 = wp::mul(var_101, var_102);
    return var_103;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:340
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _frame_axis_0(
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::int32 var_objtype,
    wp::int32 var_refid,
    wp::int32 var_reftype,
    wp::int32 var_frame_axis)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    bool var_1;
    wp::mat_t<3, 3, wp::float32>* var_2;
    wp::mat_t<3, 3, wp::float32> var_3;
    wp::mat_t<3, 3, wp::float32> var_4;
    const wp::int32 var_5 = 0;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    const wp::int32 var_9 = 2;
    wp::float32 var_10;
    wp::vec_t<3, wp::float32> var_11;
    const wp::int32 var_12 = 2;
    bool var_13;
    wp::mat_t<3, 3, wp::float32>* var_14;
    wp::mat_t<3, 3, wp::float32> var_15;
    wp::mat_t<3, 3, wp::float32> var_16;
    const wp::int32 var_17 = 0;
    wp::float32 var_18;
    const wp::int32 var_19 = 1;
    wp::float32 var_20;
    const wp::int32 var_21 = 2;
    wp::float32 var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::mat_t<3, 3, wp::float32> var_24;
    wp::vec_t<3, wp::float32> var_25;
    const wp::int32 var_26 = 5;
    bool var_27;
    wp::mat_t<3, 3, wp::float32>* var_28;
    wp::mat_t<3, 3, wp::float32> var_29;
    wp::mat_t<3, 3, wp::float32> var_30;
    const wp::int32 var_31 = 0;
    wp::float32 var_32;
    const wp::int32 var_33 = 1;
    wp::float32 var_34;
    const wp::int32 var_35 = 2;
    wp::float32 var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::mat_t<3, 3, wp::float32> var_38;
    wp::vec_t<3, wp::float32> var_39;
    const wp::int32 var_40 = 6;
    bool var_41;
    wp::mat_t<3, 3, wp::float32>* var_42;
    wp::mat_t<3, 3, wp::float32> var_43;
    wp::mat_t<3, 3, wp::float32> var_44;
    const wp::int32 var_45 = 0;
    wp::float32 var_46;
    const wp::int32 var_47 = 1;
    wp::float32 var_48;
    const wp::int32 var_49 = 2;
    wp::float32 var_50;
    wp::vec_t<3, wp::float32> var_51;
    wp::mat_t<3, 3, wp::float32> var_52;
    wp::vec_t<3, wp::float32> var_53;
    const wp::int32 var_54 = 7;
    bool var_55;
    wp::mat_t<3, 3, wp::float32>* var_56;
    wp::mat_t<3, 3, wp::float32> var_57;
    wp::mat_t<3, 3, wp::float32> var_58;
    const wp::int32 var_59 = 0;
    wp::float32 var_60;
    const wp::int32 var_61 = 1;
    wp::float32 var_62;
    const wp::int32 var_63 = 2;
    wp::float32 var_64;
    wp::vec_t<3, wp::float32> var_65;
    wp::mat_t<3, 3, wp::float32> var_66;
    wp::vec_t<3, wp::float32> var_67;
    const wp::int32 var_68 = 0;
    wp::float32 var_69;
    const wp::int32 var_70 = 1;
    wp::float32 var_71;
    const wp::int32 var_72 = 2;
    wp::float32 var_73;
    wp::vec_t<3, wp::float32> var_74;
    wp::vec_t<3, wp::float32> var_75;
    wp::mat_t<3, 3, wp::float32> var_76;
    wp::vec_t<3, wp::float32> var_77;
    wp::mat_t<3, 3, wp::float32> var_78;
    wp::vec_t<3, wp::float32> var_79;
    wp::mat_t<3, 3, wp::float32> var_80;
    wp::vec_t<3, wp::float32> var_81;
    wp::mat_t<3, 3, wp::float32> var_82;
    wp::vec_t<3, wp::float32> var_83;
    const wp::int32 var_84 = 1;
    const wp::int32 var_85 = -1;
    bool var_86;
    const wp::int32 var_87 = 1;
    bool var_88;
    wp::mat_t<3, 3, wp::float32>* var_89;
    wp::mat_t<3, 3, wp::float32> var_90;
    wp::mat_t<3, 3, wp::float32> var_91;
    const wp::int32 var_92 = 2;
    bool var_93;
    wp::mat_t<3, 3, wp::float32>* var_94;
    wp::mat_t<3, 3, wp::float32> var_95;
    wp::mat_t<3, 3, wp::float32> var_96;
    wp::mat_t<3, 3, wp::float32> var_97;
    const wp::int32 var_98 = 5;
    bool var_99;
    wp::mat_t<3, 3, wp::float32>* var_100;
    wp::mat_t<3, 3, wp::float32> var_101;
    wp::mat_t<3, 3, wp::float32> var_102;
    wp::mat_t<3, 3, wp::float32> var_103;
    const wp::int32 var_104 = 6;
    bool var_105;
    wp::mat_t<3, 3, wp::float32>* var_106;
    wp::mat_t<3, 3, wp::float32> var_107;
    wp::mat_t<3, 3, wp::float32> var_108;
    wp::mat_t<3, 3, wp::float32> var_109;
    const wp::int32 var_110 = 7;
    bool var_111;
    wp::mat_t<3, 3, wp::float32>* var_112;
    wp::mat_t<3, 3, wp::float32> var_113;
    wp::mat_t<3, 3, wp::float32> var_114;
    wp::mat_t<3, 3, wp::float32> var_115;
    const wp::int32 var_116 = 3;
    wp::mat_t<3, 3, wp::float32> var_117;
    wp::mat_t<3, 3, wp::float32> var_118;
    wp::mat_t<3, 3, wp::float32> var_119;
    wp::mat_t<3, 3, wp::float32> var_120;
    wp::mat_t<3, 3, wp::float32> var_121;
    wp::mat_t<3, 3, wp::float32> var_122;
    wp::mat_t<3, 3, wp::float32> var_123;
    wp::vec_t<3, wp::float32> var_124;
    //---------
    // forward
    // def _frame_axis(                                                                       <L 341>
    // if objtype == ObjType.BODY:                                                            <L 356>
    var_1 = (var_objtype == var_0);
    if (var_1) {
        // xmat = ximat_in[worldid, objid]                                                    <L 357>
        var_2 = wp::address(var_ximat_in, var_worldid, var_objid);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // axis = wp.vec3(xmat[0, frame_axis], xmat[1, frame_axis], xmat[2, frame_axis])       <L 358>
        var_6 = wp::extract(var_3, var_5, var_frame_axis);
        var_8 = wp::extract(var_3, var_7, var_frame_axis);
        var_10 = wp::extract(var_3, var_9, var_frame_axis);
        var_11 = wp::vec_t<3, wp::float32>(var_6, var_8, var_10);
    }
    if (!var_1) {
        // elif objtype == ObjType.XBODY:                                                     <L 359>
        var_13 = (var_objtype == var_12);
        if (var_13) {
            // xmat = xmat_in[worldid, objid]                                                 <L 360>
            var_14 = wp::address(var_xmat_in, var_worldid, var_objid);
            var_16 = wp::load(var_14);
            var_15 = wp::copy(var_16);
            // axis = wp.vec3(xmat[0, frame_axis], xmat[1, frame_axis], xmat[2, frame_axis])       <L 361>
            var_18 = wp::extract(var_15, var_17, var_frame_axis);
            var_20 = wp::extract(var_15, var_19, var_frame_axis);
            var_22 = wp::extract(var_15, var_21, var_frame_axis);
            var_23 = wp::vec_t<3, wp::float32>(var_18, var_20, var_22);
        }
        var_24 = wp::where(var_13, var_15, var_3);
        var_25 = wp::where(var_13, var_23, var_11);
        if (!var_13) {
            // elif objtype == ObjType.GEOM:                                                  <L 362>
            var_27 = (var_objtype == var_26);
            if (var_27) {
                // xmat = geom_xmat_in[worldid, objid]                                        <L 363>
                var_28 = wp::address(var_geom_xmat_in, var_worldid, var_objid);
                var_30 = wp::load(var_28);
                var_29 = wp::copy(var_30);
                // axis = wp.vec3(xmat[0, frame_axis], xmat[1, frame_axis], xmat[2, frame_axis])       <L 364>
                var_32 = wp::extract(var_29, var_31, var_frame_axis);
                var_34 = wp::extract(var_29, var_33, var_frame_axis);
                var_36 = wp::extract(var_29, var_35, var_frame_axis);
                var_37 = wp::vec_t<3, wp::float32>(var_32, var_34, var_36);
            }
            var_38 = wp::where(var_27, var_29, var_24);
            var_39 = wp::where(var_27, var_37, var_25);
            if (!var_27) {
                // elif objtype == ObjType.SITE:                                              <L 365>
                var_41 = (var_objtype == var_40);
                if (var_41) {
                    // xmat = site_xmat_in[worldid, objid]                                    <L 366>
                    var_42 = wp::address(var_site_xmat_in, var_worldid, var_objid);
                    var_44 = wp::load(var_42);
                    var_43 = wp::copy(var_44);
                    // axis = wp.vec3(xmat[0, frame_axis], xmat[1, frame_axis], xmat[2, frame_axis])       <L 367>
                    var_46 = wp::extract(var_43, var_45, var_frame_axis);
                    var_48 = wp::extract(var_43, var_47, var_frame_axis);
                    var_50 = wp::extract(var_43, var_49, var_frame_axis);
                    var_51 = wp::vec_t<3, wp::float32>(var_46, var_48, var_50);
                }
                var_52 = wp::where(var_41, var_43, var_38);
                var_53 = wp::where(var_41, var_51, var_39);
                if (!var_41) {
                    // elif objtype == ObjType.CAMERA:                                        <L 368>
                    var_55 = (var_objtype == var_54);
                    if (var_55) {
                        // xmat = cam_xmat_in[worldid, objid]                                 <L 369>
                        var_56 = wp::address(var_cam_xmat_in, var_worldid, var_objid);
                        var_58 = wp::load(var_56);
                        var_57 = wp::copy(var_58);
                        // axis = wp.vec3(xmat[0, frame_axis], xmat[1, frame_axis], xmat[2, frame_axis])       <L 370>
                        var_60 = wp::extract(var_57, var_59, var_frame_axis);
                        var_62 = wp::extract(var_57, var_61, var_frame_axis);
                        var_64 = wp::extract(var_57, var_63, var_frame_axis);
                        var_65 = wp::vec_t<3, wp::float32>(var_60, var_62, var_64);
                    }
                    var_66 = wp::where(var_55, var_57, var_52);
                    var_67 = wp::where(var_55, var_65, var_53);
                    if (!var_55) {
                        // axis = wp.vec3(xmat[0, frame_axis], xmat[1, frame_axis], xmat[2, frame_axis])       <L 372>
                        var_69 = wp::extract(var_66, var_68, var_frame_axis);
                        var_71 = wp::extract(var_66, var_70, var_frame_axis);
                        var_73 = wp::extract(var_66, var_72, var_frame_axis);
                        var_74 = wp::vec_t<3, wp::float32>(var_69, var_71, var_73);
                    }
                    var_75 = wp::where(var_55, var_67, var_74);
                }
                var_76 = wp::where(var_41, var_52, var_66);
                var_77 = wp::where(var_41, var_53, var_75);
            }
            var_78 = wp::where(var_27, var_38, var_76);
            var_79 = wp::where(var_27, var_39, var_77);
        }
        var_80 = wp::where(var_13, var_24, var_78);
        var_81 = wp::where(var_13, var_25, var_79);
    }
    var_82 = wp::where(var_1, var_3, var_80);
    var_83 = wp::where(var_1, var_11, var_81);
    // if refid == -1:                                                                        <L 374>
    var_86 = (var_refid == var_85);
    if (var_86) {
        // return axis                                                                        <L 375>
        return var_83;
    }
    // if reftype == ObjType.BODY:                                                            <L 377>
    var_88 = (var_reftype == var_87);
    if (var_88) {
        // xmat_ref = ximat_in[worldid, refid]                                                <L 378>
        var_89 = wp::address(var_ximat_in, var_worldid, var_refid);
        var_91 = wp::load(var_89);
        var_90 = wp::copy(var_91);
    }
    if (!var_88) {
        // elif reftype == ObjType.XBODY:                                                     <L 379>
        var_93 = (var_reftype == var_92);
        if (var_93) {
            // xmat_ref = xmat_in[worldid, refid]                                             <L 380>
            var_94 = wp::address(var_xmat_in, var_worldid, var_refid);
            var_96 = wp::load(var_94);
            var_95 = wp::copy(var_96);
        }
        var_97 = wp::where(var_93, var_95, var_90);
        if (!var_93) {
            // elif reftype == ObjType.GEOM:                                                  <L 381>
            var_99 = (var_reftype == var_98);
            if (var_99) {
                // xmat_ref = geom_xmat_in[worldid, refid]                                    <L 382>
                var_100 = wp::address(var_geom_xmat_in, var_worldid, var_refid);
                var_102 = wp::load(var_100);
                var_101 = wp::copy(var_102);
            }
            var_103 = wp::where(var_99, var_101, var_97);
            if (!var_99) {
                // elif reftype == ObjType.SITE:                                              <L 383>
                var_105 = (var_reftype == var_104);
                if (var_105) {
                    // xmat_ref = site_xmat_in[worldid, refid]                                <L 384>
                    var_106 = wp::address(var_site_xmat_in, var_worldid, var_refid);
                    var_108 = wp::load(var_106);
                    var_107 = wp::copy(var_108);
                }
                var_109 = wp::where(var_105, var_107, var_103);
                if (!var_105) {
                    // elif reftype == ObjType.CAMERA:                                        <L 385>
                    var_111 = (var_reftype == var_110);
                    if (var_111) {
                        // xmat_ref = cam_xmat_in[worldid, refid]                             <L 386>
                        var_112 = wp::address(var_cam_xmat_in, var_worldid, var_refid);
                        var_114 = wp::load(var_112);
                        var_113 = wp::copy(var_114);
                    }
                    var_115 = wp::where(var_111, var_113, var_109);
                    if (!var_111) {
                        // xmat_ref = wp.identity(3, dtype=wp.float32)                        <L 388>
                        var_117 = wp::identity<3, wp::float32>();
                    }
                    var_118 = wp::where(var_111, var_115, var_117);
                }
                var_119 = wp::where(var_105, var_109, var_118);
            }
            var_120 = wp::where(var_99, var_103, var_119);
        }
        var_121 = wp::where(var_93, var_97, var_120);
    }
    var_122 = wp::where(var_88, var_90, var_121);
    // return wp.transpose(xmat_ref) @ axis                                                   <L 390>
    var_123 = wp::transpose(var_122);
    var_124 = wp::mul(var_123, var_83);
    return var_124;
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:393
static CUDA_CALLABLE wp::quat_t<wp::float32> _frame_quat_0(
    wp::array_t<wp::quat_t<wp::float32>> var_body_iquat,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> var_geom_quat,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> var_site_quat,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> var_cam_quat,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::int32 var_objtype,
    wp::int32 var_refid,
    wp::int32 var_reftype)
{
    //---------
    // primal vars
    wp::shape_t* var_0;
    const wp::int32 var_1 = 0;
    wp::int32 var_2;
    wp::shape_t var_3;
    wp::int32 var_4;
    wp::shape_t* var_5;
    const wp::int32 var_6 = 0;
    wp::int32 var_7;
    wp::shape_t var_8;
    wp::int32 var_9;
    wp::shape_t* var_10;
    const wp::int32 var_11 = 0;
    wp::int32 var_12;
    wp::shape_t var_13;
    wp::int32 var_14;
    wp::shape_t* var_15;
    const wp::int32 var_16 = 0;
    wp::int32 var_17;
    wp::shape_t var_18;
    wp::int32 var_19;
    const wp::int32 var_20 = 1;
    bool var_21;
    wp::quat_t<wp::float32>* var_22;
    wp::quat_t<wp::float32>* var_23;
    wp::quat_t<wp::float32> var_24;
    wp::quat_t<wp::float32> var_25;
    wp::quat_t<wp::float32> var_26;
    const wp::int32 var_27 = 2;
    bool var_28;
    wp::quat_t<wp::float32>* var_29;
    wp::quat_t<wp::float32> var_30;
    wp::quat_t<wp::float32> var_31;
    wp::quat_t<wp::float32> var_32;
    const wp::int32 var_33 = 5;
    bool var_34;
    wp::int32* var_35;
    wp::quat_t<wp::float32>* var_36;
    wp::int32 var_37;
    wp::quat_t<wp::float32>* var_38;
    wp::quat_t<wp::float32> var_39;
    wp::quat_t<wp::float32> var_40;
    wp::quat_t<wp::float32> var_41;
    wp::quat_t<wp::float32> var_42;
    const wp::int32 var_43 = 6;
    bool var_44;
    wp::int32* var_45;
    wp::quat_t<wp::float32>* var_46;
    wp::int32 var_47;
    wp::quat_t<wp::float32>* var_48;
    wp::quat_t<wp::float32> var_49;
    wp::quat_t<wp::float32> var_50;
    wp::quat_t<wp::float32> var_51;
    wp::quat_t<wp::float32> var_52;
    const wp::int32 var_53 = 7;
    bool var_54;
    wp::int32* var_55;
    wp::quat_t<wp::float32>* var_56;
    wp::int32 var_57;
    wp::quat_t<wp::float32>* var_58;
    wp::quat_t<wp::float32> var_59;
    wp::quat_t<wp::float32> var_60;
    wp::quat_t<wp::float32> var_61;
    wp::quat_t<wp::float32> var_62;
    const wp::float32 var_63 = 1.0;
    const wp::float32 var_64 = 0.0;
    const wp::float32 var_65 = 0.0;
    const wp::float32 var_66 = 0.0;
    wp::quat_t<wp::float32> var_67;
    wp::quat_t<wp::float32> var_68;
    wp::quat_t<wp::float32> var_69;
    wp::quat_t<wp::float32> var_70;
    wp::quat_t<wp::float32> var_71;
    wp::quat_t<wp::float32> var_72;
    const wp::int32 var_73 = 1;
    const wp::int32 var_74 = -1;
    bool var_75;
    const wp::int32 var_76 = 1;
    bool var_77;
    wp::quat_t<wp::float32>* var_78;
    wp::quat_t<wp::float32>* var_79;
    wp::quat_t<wp::float32> var_80;
    wp::quat_t<wp::float32> var_81;
    wp::quat_t<wp::float32> var_82;
    const wp::int32 var_83 = 2;
    bool var_84;
    wp::quat_t<wp::float32>* var_85;
    wp::quat_t<wp::float32> var_86;
    wp::quat_t<wp::float32> var_87;
    wp::quat_t<wp::float32> var_88;
    const wp::int32 var_89 = 5;
    bool var_90;
    wp::int32* var_91;
    wp::quat_t<wp::float32>* var_92;
    wp::int32 var_93;
    wp::quat_t<wp::float32>* var_94;
    wp::quat_t<wp::float32> var_95;
    wp::quat_t<wp::float32> var_96;
    wp::quat_t<wp::float32> var_97;
    wp::quat_t<wp::float32> var_98;
    const wp::int32 var_99 = 6;
    bool var_100;
    wp::int32* var_101;
    wp::quat_t<wp::float32>* var_102;
    wp::int32 var_103;
    wp::quat_t<wp::float32>* var_104;
    wp::quat_t<wp::float32> var_105;
    wp::quat_t<wp::float32> var_106;
    wp::quat_t<wp::float32> var_107;
    wp::quat_t<wp::float32> var_108;
    const wp::int32 var_109 = 7;
    bool var_110;
    wp::int32* var_111;
    wp::quat_t<wp::float32>* var_112;
    wp::int32 var_113;
    wp::quat_t<wp::float32>* var_114;
    wp::quat_t<wp::float32> var_115;
    wp::quat_t<wp::float32> var_116;
    wp::quat_t<wp::float32> var_117;
    wp::quat_t<wp::float32> var_118;
    const wp::float32 var_119 = 1.0;
    const wp::float32 var_120 = 0.0;
    const wp::float32 var_121 = 0.0;
    const wp::float32 var_122 = 0.0;
    wp::quat_t<wp::float32> var_123;
    wp::quat_t<wp::float32> var_124;
    wp::quat_t<wp::float32> var_125;
    wp::quat_t<wp::float32> var_126;
    wp::quat_t<wp::float32> var_127;
    wp::quat_t<wp::float32> var_128;
    wp::quat_t<wp::float32> var_129;
    wp::quat_t<wp::float32> var_130;
    //---------
    // forward
    // def _frame_quat(                                                                       <L 394>
    // body_iquat_id = worldid % body_iquat.shape[0]                                          <L 412>
    var_0 = &(var_body_iquat.shape);
    var_3 = wp::load(var_0);
    var_2 = wp::extract(var_3, var_1);
    var_4 = wp::mod(var_worldid, var_2);
    // geom_quat_id = worldid % geom_quat.shape[0]                                            <L 413>
    var_5 = &(var_geom_quat.shape);
    var_8 = wp::load(var_5);
    var_7 = wp::extract(var_8, var_6);
    var_9 = wp::mod(var_worldid, var_7);
    // site_quat_id = worldid % site_quat.shape[0]                                            <L 414>
    var_10 = &(var_site_quat.shape);
    var_13 = wp::load(var_10);
    var_12 = wp::extract(var_13, var_11);
    var_14 = wp::mod(var_worldid, var_12);
    // cam_quat_id = worldid % cam_quat.shape[0]                                              <L 415>
    var_15 = &(var_cam_quat.shape);
    var_18 = wp::load(var_15);
    var_17 = wp::extract(var_18, var_16);
    var_19 = wp::mod(var_worldid, var_17);
    // if objtype == ObjType.BODY:                                                            <L 416>
    var_21 = (var_objtype == var_20);
    if (var_21) {
        // quat = math.mul_quat(xquat_in[worldid, objid], body_iquat[body_iquat_id, objid])       <L 417>
        var_22 = wp::address(var_xquat_in, var_worldid, var_objid);
        var_23 = wp::address(var_body_iquat, var_4, var_objid);
        var_25 = wp::load(var_22);
        var_26 = wp::load(var_23);
        var_24 = mul_quat_0(var_25, var_26);
    }
    if (!var_21) {
        // elif objtype == ObjType.XBODY:                                                     <L 418>
        var_28 = (var_objtype == var_27);
        if (var_28) {
            // quat = xquat_in[worldid, objid]                                                <L 419>
            var_29 = wp::address(var_xquat_in, var_worldid, var_objid);
            var_31 = wp::load(var_29);
            var_30 = wp::copy(var_31);
        }
        var_32 = wp::where(var_28, var_30, var_24);
        if (!var_28) {
            // elif objtype == ObjType.GEOM:                                                  <L 420>
            var_34 = (var_objtype == var_33);
            if (var_34) {
                // quat = math.mul_quat(xquat_in[worldid, geom_bodyid[objid]], geom_quat[geom_quat_id, objid])       <L 421>
                var_35 = wp::address(var_geom_bodyid, var_objid);
                var_37 = wp::load(var_35);
                var_36 = wp::address(var_xquat_in, var_worldid, var_37);
                var_38 = wp::address(var_geom_quat, var_9, var_objid);
                var_40 = wp::load(var_36);
                var_41 = wp::load(var_38);
                var_39 = mul_quat_0(var_40, var_41);
            }
            var_42 = wp::where(var_34, var_39, var_32);
            if (!var_34) {
                // elif objtype == ObjType.SITE:                                              <L 422>
                var_44 = (var_objtype == var_43);
                if (var_44) {
                    // quat = math.mul_quat(xquat_in[worldid, site_bodyid[objid]], site_quat[site_quat_id, objid])       <L 423>
                    var_45 = wp::address(var_site_bodyid, var_objid);
                    var_47 = wp::load(var_45);
                    var_46 = wp::address(var_xquat_in, var_worldid, var_47);
                    var_48 = wp::address(var_site_quat, var_14, var_objid);
                    var_50 = wp::load(var_46);
                    var_51 = wp::load(var_48);
                    var_49 = mul_quat_0(var_50, var_51);
                }
                var_52 = wp::where(var_44, var_49, var_42);
                if (!var_44) {
                    // elif objtype == ObjType.CAMERA:                                        <L 424>
                    var_54 = (var_objtype == var_53);
                    if (var_54) {
                        // quat = math.mul_quat(xquat_in[worldid, cam_bodyid[objid]], cam_quat[cam_quat_id, objid])       <L 425>
                        var_55 = wp::address(var_cam_bodyid, var_objid);
                        var_57 = wp::load(var_55);
                        var_56 = wp::address(var_xquat_in, var_worldid, var_57);
                        var_58 = wp::address(var_cam_quat, var_19, var_objid);
                        var_60 = wp::load(var_56);
                        var_61 = wp::load(var_58);
                        var_59 = mul_quat_0(var_60, var_61);
                    }
                    var_62 = wp::where(var_54, var_59, var_52);
                    if (!var_54) {
                        // quat = wp.quat(1.0, 0.0, 0.0, 0.0)                                 <L 427>
                        var_67 = wp::quat_t<wp::float32>(var_63, var_64, var_65, var_66);
                    }
                    var_68 = wp::where(var_54, var_62, var_67);
                }
                var_69 = wp::where(var_44, var_52, var_68);
            }
            var_70 = wp::where(var_34, var_42, var_69);
        }
        var_71 = wp::where(var_28, var_32, var_70);
    }
    var_72 = wp::where(var_21, var_24, var_71);
    // if refid == -1:                                                                        <L 429>
    var_75 = (var_refid == var_74);
    if (var_75) {
        // return quat                                                                        <L 430>
        return var_72;
    }
    // if reftype == ObjType.BODY:                                                            <L 432>
    var_77 = (var_reftype == var_76);
    if (var_77) {
        // refquat = math.mul_quat(xquat_in[worldid, refid], body_iquat[body_iquat_id, refid])       <L 433>
        var_78 = wp::address(var_xquat_in, var_worldid, var_refid);
        var_79 = wp::address(var_body_iquat, var_4, var_refid);
        var_81 = wp::load(var_78);
        var_82 = wp::load(var_79);
        var_80 = mul_quat_0(var_81, var_82);
    }
    if (!var_77) {
        // elif reftype == ObjType.XBODY:                                                     <L 434>
        var_84 = (var_reftype == var_83);
        if (var_84) {
            // refquat = xquat_in[worldid, refid]                                             <L 435>
            var_85 = wp::address(var_xquat_in, var_worldid, var_refid);
            var_87 = wp::load(var_85);
            var_86 = wp::copy(var_87);
        }
        var_88 = wp::where(var_84, var_86, var_80);
        if (!var_84) {
            // elif reftype == ObjType.GEOM:                                                  <L 436>
            var_90 = (var_reftype == var_89);
            if (var_90) {
                // refquat = math.mul_quat(xquat_in[worldid, geom_bodyid[refid]], geom_quat[geom_quat_id, refid])       <L 437>
                var_91 = wp::address(var_geom_bodyid, var_refid);
                var_93 = wp::load(var_91);
                var_92 = wp::address(var_xquat_in, var_worldid, var_93);
                var_94 = wp::address(var_geom_quat, var_9, var_refid);
                var_96 = wp::load(var_92);
                var_97 = wp::load(var_94);
                var_95 = mul_quat_0(var_96, var_97);
            }
            var_98 = wp::where(var_90, var_95, var_88);
            if (!var_90) {
                // elif reftype == ObjType.SITE:                                              <L 438>
                var_100 = (var_reftype == var_99);
                if (var_100) {
                    // refquat = math.mul_quat(xquat_in[worldid, site_bodyid[refid]], site_quat[site_quat_id, refid])       <L 439>
                    var_101 = wp::address(var_site_bodyid, var_refid);
                    var_103 = wp::load(var_101);
                    var_102 = wp::address(var_xquat_in, var_worldid, var_103);
                    var_104 = wp::address(var_site_quat, var_14, var_refid);
                    var_106 = wp::load(var_102);
                    var_107 = wp::load(var_104);
                    var_105 = mul_quat_0(var_106, var_107);
                }
                var_108 = wp::where(var_100, var_105, var_98);
                if (!var_100) {
                    // elif reftype == ObjType.CAMERA:                                        <L 440>
                    var_110 = (var_reftype == var_109);
                    if (var_110) {
                        // refquat = math.mul_quat(xquat_in[worldid, cam_bodyid[refid]], cam_quat[cam_quat_id, refid])       <L 441>
                        var_111 = wp::address(var_cam_bodyid, var_refid);
                        var_113 = wp::load(var_111);
                        var_112 = wp::address(var_xquat_in, var_worldid, var_113);
                        var_114 = wp::address(var_cam_quat, var_19, var_refid);
                        var_116 = wp::load(var_112);
                        var_117 = wp::load(var_114);
                        var_115 = mul_quat_0(var_116, var_117);
                    }
                    var_118 = wp::where(var_110, var_115, var_108);
                    if (!var_110) {
                        // refquat = wp.quat(1.0, 0.0, 0.0, 0.0)                              <L 443>
                        var_123 = wp::quat_t<wp::float32>(var_119, var_120, var_121, var_122);
                    }
                    var_124 = wp::where(var_110, var_118, var_123);
                }
                var_125 = wp::where(var_100, var_108, var_124);
            }
            var_126 = wp::where(var_90, var_98, var_125);
        }
        var_127 = wp::where(var_84, var_88, var_126);
    }
    var_128 = wp::where(var_77, var_80, var_127);
    // return math.mul_quat(math.quat_inv(refquat), quat)                                     <L 445>
    var_129 = quat_inv_0(var_128);
    var_130 = mul_quat_0(var_129, var_72);
    return var_130;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:448
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _subtree_com_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32>* var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    //---------
    // forward
    // def _subtree_com(subtree_com_in: wp.array2d[wp.vec3], worldid: int, objid: int) -> wp.vec3:       <L 449>
    // return subtree_com_in[worldid, objid]                                                  <L 450>
    var_0 = wp::address(var_subtree_com_in, var_worldid, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    return var_1;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:322
static CUDA_CALLABLE wp::int32 upper_tri_index_0(
    wp::int32 var_n,
    wp::int32 var_i,
    wp::int32 var_j)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 2;
    wp::int32 var_1;
    wp::int32 var_2;
    const wp::int32 var_3 = 3;
    wp::int32 var_4;
    wp::int32 var_5;
    const wp::int32 var_6 = 2;
    wp::int32 var_7;
    wp::int32 var_8;
    const wp::int32 var_9 = 1;
    wp::int32 var_10;
    //---------
    // forward
    // def upper_tri_index(n: int, i: int, j: int) -> int:                                    <L 323>
    // return (i * (2 * n - i - 3)) // 2 + j - 1                                              <L 325>
    var_1 = wp::mul(var_0, var_n);
    var_2 = wp::sub(var_1, var_i);
    var_4 = wp::sub(var_2, var_3);
    var_5 = wp::mul(var_i, var_4);
    var_7 = wp::floordiv(var_5, var_6);
    var_8 = wp::add(var_7, var_j);
    var_10 = wp::sub(var_8, var_9);
    return var_10;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:0
static CUDA_CALLABLE void _write_vector_0(
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::int32 var_sensorid,
    wp::int32 var_sensordim,
    wp::vec_t<6, wp::float32> var_sensor,
    wp::array_t<wp::float32> var_out)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    wp::float32* var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    const wp::float32 var_6 = 0.0;
    bool var_7;
    wp::int32* var_8;
    const wp::int32 var_9 = 41;
    const wp::int32 var_10 = 41;
    wp::int32 var_11;
    bool var_12;
    wp::int32 var_13;
    bool var_14;
    bool var_15;
    wp::int32* var_16;
    wp::int32 var_17;
    wp::int32 var_18;
    const wp::int32 var_19 = 0;
    bool var_20;
    wp::range_t var_21;
    wp::int32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::int32 var_26;
    const wp::int32 var_27 = 1;
    bool var_28;
    wp::range_t var_29;
    wp::int32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::int32 var_33;
    wp::int32 var_34;
    wp::int32 var_35;
    wp::range_t var_36;
    wp::int32 var_37;
    wp::float32 var_38;
    wp::int32 var_39;
    //---------
    // forward
    // def _write_vector(                                                                     <L 1>
    // adr = sensor_adr[sensorid]                                                             <L 14>
    var_0 = wp::address(var_sensor_adr, var_sensorid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // cutoff = sensor_cutoff[sensorid]                                                       <L 15>
    var_3 = wp::address(var_sensor_cutoff, var_sensorid);
    var_5 = wp::load(var_3);
    var_4 = wp::copy(var_5);
    // if cutoff > 0.0 and not (sensor_type[sensorid] == int(SensorType.GEOMFROMTO.value)):       <L 17>
    var_7 = (var_4 > var_6);
    var_8 = wp::address(var_sensor_type, var_sensorid);
    var_11 = wp::int(var_10);
    var_13 = wp::load(var_8);
    var_12 = (var_13 == var_11);
    var_14 = wp::unot(var_12);
    var_15 = var_7 && var_14;
    if (var_15) {
        // datatype = sensor_datatype[sensorid]                                               <L 18>
        var_16 = wp::address(var_sensor_datatype, var_sensorid);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // if datatype == DataType.REAL:                                                      <L 19>
        var_20 = (var_17 == var_19);
        if (var_20) {
            // for i in range(sensordim):                                                     <L 20>
            var_21 = wp::range(var_sensordim);
            start_for_0:;
                if (iter_cmp(var_21) == 0) goto end_for_0;
                var_22 = wp::iter_next(var_21);
                // out[adr + i] = wp.clamp(sensor[i], -cutoff, cutoff)                        <L 21>
                var_23 = wp::extract(var_sensor, var_22);
                var_24 = wp::neg(var_4);
                var_25 = wp::clamp(var_23, var_24, var_4);
                var_26 = wp::add(var_1, var_22);
                wp::array_store(var_out, var_26, var_25);
                goto start_for_0;
            end_for_0:;
            // return                                                                         <L 22>
            return;
        }
        if (!var_20) {
            // elif datatype == DataType.POSITIVE:                                            <L 23>
            var_28 = (var_17 == var_27);
            if (var_28) {
                // for i in range(sensordim):                                                 <L 24>
                var_29 = wp::range(var_sensordim);
                start_for_3:;
                    if (iter_cmp(var_29) == 0) goto end_for_3;
                    var_30 = wp::iter_next(var_29);
                    // out[adr + i] = wp.min(sensor[i], cutoff)                               <L 25>
                    var_31 = wp::extract(var_sensor, var_30);
                    var_32 = wp::min(var_31, var_4);
                    var_33 = wp::add(var_1, var_30);
                    wp::array_store(var_out, var_33, var_32);
                    goto start_for_3;
                end_for_3:;
                // return                                                                     <L 26>
                return;
            }
            var_34 = wp::where(var_28, var_30, var_22);
        }
        var_35 = wp::where(var_20, var_22, var_34);
    }
    // for i in range(sensordim):                                                             <L 28>
    var_36 = wp::range(var_sensordim);
    start_for_6:;
        if (iter_cmp(var_36) == 0) goto end_for_6;
        var_37 = wp::iter_next(var_36);
        // out[adr + i] = sensor[i]                                                           <L 29>
        var_38 = wp::extract(var_sensor, var_37);
        var_39 = wp::add(var_1, var_37);
        wp::array_store(var_out, var_39, var_38);
        goto start_for_6;
    end_for_6:;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/util_misc.py:602
static CUDA_CALLABLE bool inside_geom_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::int32 var_geomtype,
    wp::vec_t<3, wp::float32> var_point)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    const wp::int32 var_1 = 2;
    bool var_2;
    wp::float32 var_3;
    const wp::int32 var_4 = 0;
    wp::float32 var_5;
    const wp::int32 var_6 = 0;
    wp::float32 var_7;
    wp::float32 var_8;
    bool var_9;
    wp::mat_t<3, 3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    const wp::int32 var_12 = 3;
    bool var_13;
    const wp::int32 var_14 = 2;
    wp::float32 var_15;
    const wp::int32 var_16 = 1;
    wp::float32 var_17;
    wp::float32 var_18;
    const wp::int32 var_19 = 1;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    const wp::int32 var_24 = 0;
    wp::float32 var_25;
    const wp::int32 var_26 = 0;
    wp::float32 var_27;
    wp::float32 var_28;
    const wp::int32 var_29 = 1;
    wp::float32 var_30;
    const wp::int32 var_31 = 1;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    wp::float32 var_35;
    const wp::int32 var_36 = 0;
    wp::float32 var_37;
    const wp::int32 var_38 = 0;
    wp::float32 var_39;
    wp::float32 var_40;
    bool var_41;
    const wp::int32 var_42 = 4;
    bool var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::float32 var_45;
    const wp::float32 var_46 = 1.0;
    bool var_47;
    const wp::int32 var_48 = 5;
    bool var_49;
    const wp::int32 var_50 = 2;
    wp::float32 var_51;
    wp::float32 var_52;
    const wp::int32 var_53 = 1;
    wp::float32 var_54;
    bool var_55;
    const wp::int32 var_56 = 0;
    wp::float32 var_57;
    const wp::int32 var_58 = 0;
    wp::float32 var_59;
    wp::float32 var_60;
    const wp::int32 var_61 = 1;
    wp::float32 var_62;
    const wp::int32 var_63 = 1;
    wp::float32 var_64;
    wp::float32 var_65;
    wp::float32 var_66;
    const wp::int32 var_67 = 0;
    wp::float32 var_68;
    const wp::int32 var_69 = 0;
    wp::float32 var_70;
    wp::float32 var_71;
    bool var_72;
    bool var_73;
    const wp::int32 var_74 = 6;
    bool var_75;
    const wp::int32 var_76 = 0;
    wp::float32 var_77;
    wp::float32 var_78;
    const wp::int32 var_79 = 0;
    wp::float32 var_80;
    bool var_81;
    const wp::int32 var_82 = 1;
    wp::float32 var_83;
    wp::float32 var_84;
    const wp::int32 var_85 = 1;
    wp::float32 var_86;
    bool var_87;
    const wp::int32 var_88 = 2;
    wp::float32 var_89;
    wp::float32 var_90;
    const wp::int32 var_91 = 2;
    wp::float32 var_92;
    bool var_93;
    bool var_94;
    const wp::int32 var_95 = 0;
    bool var_96;
    const wp::int32 var_97 = 2;
    wp::float32 var_98;
    const wp::float32 var_99 = 0.0;
    bool var_100;
    const bool var_101 = false;
    //---------
    // forward
    // def inside_geom(pos: wp.vec3, mat: wp.mat33, size: wp.vec3, geomtype: int, point: wp.vec3) -> bool:       <L 603>
    // vec = point - pos                                                                      <L 606>
    var_0 = wp::sub(var_point, var_pos);
    // if geomtype == GeomType.SPHERE:                                                        <L 609>
    var_2 = (var_geomtype == var_1);
    if (var_2) {
        // return wp.dot(vec, vec) < size[0] * size[0]                                        <L 610>
        var_3 = wp::dot(var_0, var_0);
        var_5 = wp::extract(var_size, var_4);
        var_7 = wp::extract(var_size, var_6);
        var_8 = wp::mul(var_5, var_7);
        var_9 = (var_3 < var_8);
        return var_9;
    }
    // plocal = wp.transpose(mat) @ vec                                                       <L 613>
    var_10 = wp::transpose(var_mat);
    var_11 = wp::mul(var_10, var_0);
    // if geomtype == GeomType.CAPSULE:                                                       <L 616>
    var_13 = (var_geomtype == var_12);
    if (var_13) {
        // z = plocal[2]                                                                      <L 617>
        var_15 = wp::extract(var_11, var_14);
        // z_clamped = wp.clamp(z, -size[1], size[1])                                         <L 618>
        var_17 = wp::extract(var_size, var_16);
        var_18 = wp::neg(var_17);
        var_20 = wp::extract(var_size, var_19);
        var_21 = wp::clamp(var_15, var_18, var_20);
        // z_dif = z - z_clamped                                                              <L 619>
        var_22 = wp::sub(var_15, var_21);
        // z_dist_sq = z_dif * z_dif                                                          <L 620>
        var_23 = wp::mul(var_22, var_22);
        // return plocal[0] * plocal[0] + plocal[1] * plocal[1] + z_dist_sq < size[0] * size[0]       <L 621>
        var_25 = wp::extract(var_11, var_24);
        var_27 = wp::extract(var_11, var_26);
        var_28 = wp::mul(var_25, var_27);
        var_30 = wp::extract(var_11, var_29);
        var_32 = wp::extract(var_11, var_31);
        var_33 = wp::mul(var_30, var_32);
        var_34 = wp::add(var_28, var_33);
        var_35 = wp::add(var_34, var_23);
        var_37 = wp::extract(var_size, var_36);
        var_39 = wp::extract(var_size, var_38);
        var_40 = wp::mul(var_37, var_39);
        var_41 = (var_35 < var_40);
        return var_41;
    }
    if (!var_13) {
        // elif geomtype == GeomType.ELLIPSOID:                                               <L 622>
        var_43 = (var_geomtype == var_42);
        if (var_43) {
            // plocalsize = wp.cw_div(plocal, size)                                           <L 623>
            var_44 = wp::cw_div(var_11, var_size);
            // return wp.dot(plocalsize, plocalsize) < 1.0                                    <L 624>
            var_45 = wp::dot(var_44, var_44);
            var_47 = (var_45 < var_46);
            return var_47;
        }
        if (!var_43) {
            // elif geomtype == GeomType.CYLINDER:                                            <L 625>
            var_49 = (var_geomtype == var_48);
            if (var_49) {
                // return (wp.abs(plocal[2]) < size[1]) and (plocal[0] * plocal[0] + plocal[1] * plocal[1] < size[0] * size[0])       <L 626>
                var_51 = wp::extract(var_11, var_50);
                var_52 = wp::abs(var_51);
                var_54 = wp::extract(var_size, var_53);
                var_55 = (var_52 < var_54);
                var_57 = wp::extract(var_11, var_56);
                var_59 = wp::extract(var_11, var_58);
                var_60 = wp::mul(var_57, var_59);
                var_62 = wp::extract(var_11, var_61);
                var_64 = wp::extract(var_11, var_63);
                var_65 = wp::mul(var_62, var_64);
                var_66 = wp::add(var_60, var_65);
                var_68 = wp::extract(var_size, var_67);
                var_70 = wp::extract(var_size, var_69);
                var_71 = wp::mul(var_68, var_70);
                var_72 = (var_66 < var_71);
                var_73 = var_55 && var_72;
                return var_73;
            }
            if (!var_49) {
                // elif geomtype == GeomType.BOX:                                             <L 627>
                var_75 = (var_geomtype == var_74);
                if (var_75) {
                    // return wp.abs(plocal[0]) < size[0] and wp.abs(plocal[1]) < size[1] and wp.abs(plocal[2]) < size[2]       <L 628>
                    var_77 = wp::extract(var_11, var_76);
                    var_78 = wp::abs(var_77);
                    var_80 = wp::extract(var_size, var_79);
                    var_81 = (var_78 < var_80);
                    var_83 = wp::extract(var_11, var_82);
                    var_84 = wp::abs(var_83);
                    var_86 = wp::extract(var_size, var_85);
                    var_87 = (var_84 < var_86);
                    var_89 = wp::extract(var_11, var_88);
                    var_90 = wp::abs(var_89);
                    var_92 = wp::extract(var_size, var_91);
                    var_93 = (var_90 < var_92);
                    var_94 = var_81 && var_87 && var_93;
                    return var_94;
                }
                if (!var_75) {
                    // elif geomtype == GeomType.PLANE:                                       <L 629>
                    var_96 = (var_geomtype == var_95);
                    if (var_96) {
                        // return plocal[2] < 0.0                                             <L 630>
                        var_98 = wp::extract(var_11, var_97);
                        var_100 = (var_98 < var_99);
                        return var_100;
                    }
                }
            }
        }
    }
    // return False                                                                           <L 632>
    return var_101;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:453
static CUDA_CALLABLE wp::float32 _clock_0(
    wp::array_t<wp::float32> var_time_in,
    wp::int32 var_worldid)
{
    //---------
    // primal vars
    wp::float32* var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    //---------
    // forward
    // def _clock(time_in: wp.array[float], worldid: int) -> float:                           <L 454>
    // return time_in[worldid]                                                                <L 455>
    var_0 = wp::address(var_time_in, var_worldid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    return var_1;
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:1450
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _accelerometer_0(
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cacc_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    wp::mat_t<3, 3, wp::float32>* var_3;
    wp::mat_t<3, 3, wp::float32> var_4;
    wp::mat_t<3, 3, wp::float32> var_5;
    wp::mat_t<3, 3, wp::float32> var_6;
    wp::vec_t<6, wp::float32>* var_7;
    wp::vec_t<6, wp::float32> var_8;
    wp::vec_t<6, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<6, wp::float32>* var_12;
    wp::vec_t<6, wp::float32> var_13;
    wp::vec_t<6, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::vec_t<3, wp::float32>* var_17;
    wp::int32* var_18;
    wp::vec_t<3, wp::float32>* var_19;
    wp::int32 var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    //---------
    // forward
    // def _accelerometer(                                                                    <L 1451>
    // bodyid = site_bodyid[objid]                                                            <L 1465>
    var_0 = wp::address(var_site_bodyid, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // rot = site_xmat_in[worldid, objid]                                                     <L 1466>
    var_3 = wp::address(var_site_xmat_in, var_worldid, var_objid);
    var_5 = wp::load(var_3);
    var_4 = wp::copy(var_5);
    // rotT = wp.transpose(rot)                                                               <L 1467>
    var_6 = wp::transpose(var_4);
    // cvel = cvel_in[worldid, bodyid]                                                        <L 1468>
    var_7 = wp::address(var_cvel_in, var_worldid, var_1);
    var_9 = wp::load(var_7);
    var_8 = wp::copy(var_9);
    // cvel_top = wp.spatial_top(cvel)                                                        <L 1469>
    var_10 = wp::spatial_top(var_8);
    // cvel_bottom = wp.spatial_bottom(cvel)                                                  <L 1470>
    var_11 = wp::spatial_bottom(var_8);
    // cacc = cacc_in[worldid, bodyid]                                                        <L 1471>
    var_12 = wp::address(var_cacc_in, var_worldid, var_1);
    var_14 = wp::load(var_12);
    var_13 = wp::copy(var_14);
    // cacc_top = wp.spatial_top(cacc)                                                        <L 1472>
    var_15 = wp::spatial_top(var_13);
    // cacc_bottom = wp.spatial_bottom(cacc)                                                  <L 1473>
    var_16 = wp::spatial_bottom(var_13);
    // dif = site_xpos_in[worldid, objid] - subtree_com_in[worldid, body_rootid[bodyid]]       <L 1474>
    var_17 = wp::address(var_site_xpos_in, var_worldid, var_objid);
    var_18 = wp::address(var_body_rootid, var_1);
    var_20 = wp::load(var_18);
    var_19 = wp::address(var_subtree_com_in, var_worldid, var_20);
    var_22 = wp::load(var_17);
    var_23 = wp::load(var_19);
    var_21 = wp::sub(var_22, var_23);
    // ang = rotT @ cvel_top                                                                  <L 1475>
    var_24 = wp::mul(var_6, var_10);
    // lin = rotT @ (cvel_bottom - wp.cross(dif, cvel_top))                                   <L 1476>
    var_25 = wp::cross(var_21, var_10);
    var_26 = wp::sub(var_11, var_25);
    var_27 = wp::mul(var_6, var_26);
    // acc = rotT @ (cacc_bottom - wp.cross(dif, cacc_top))                                   <L 1477>
    var_28 = wp::cross(var_21, var_15);
    var_29 = wp::sub(var_16, var_28);
    var_30 = wp::mul(var_6, var_29);
    // correction = wp.cross(ang, lin)                                                        <L 1478>
    var_31 = wp::cross(var_24, var_27);
    // return acc + correction                                                                <L 1479>
    var_32 = wp::add(var_30, var_31);
    return var_32;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:1482
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _force_0(
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_int_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    wp::vec_t<6, wp::float32>* var_3;
    wp::vec_t<6, wp::float32> var_4;
    wp::vec_t<6, wp::float32> var_5;
    wp::mat_t<3, 3, wp::float32>* var_6;
    wp::mat_t<3, 3, wp::float32> var_7;
    wp::mat_t<3, 3, wp::float32> var_8;
    wp::mat_t<3, 3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    //---------
    // forward
    // def _force(                                                                            <L 1483>
    // bodyid = site_bodyid[objid]                                                            <L 1493>
    var_0 = wp::address(var_site_bodyid, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // cfrc_int = cfrc_int_in[worldid, bodyid]                                                <L 1494>
    var_3 = wp::address(var_cfrc_int_in, var_worldid, var_1);
    var_5 = wp::load(var_3);
    var_4 = wp::copy(var_5);
    // site_xmat = site_xmat_in[worldid, objid]                                               <L 1495>
    var_6 = wp::address(var_site_xmat_in, var_worldid, var_objid);
    var_8 = wp::load(var_6);
    var_7 = wp::copy(var_8);
    // return wp.transpose(site_xmat) @ wp.spatial_bottom(cfrc_int)                           <L 1496>
    var_9 = wp::transpose(var_7);
    var_10 = wp::spatial_bottom(var_4);
    var_11 = wp::mul(var_9, var_10);
    return var_11;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:1499
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _torque_0(
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_int_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    wp::vec_t<6, wp::float32>* var_3;
    wp::vec_t<6, wp::float32> var_4;
    wp::vec_t<6, wp::float32> var_5;
    wp::mat_t<3, 3, wp::float32>* var_6;
    wp::mat_t<3, 3, wp::float32> var_7;
    wp::mat_t<3, 3, wp::float32> var_8;
    wp::vec_t<3, wp::float32>* var_9;
    wp::int32* var_10;
    wp::vec_t<3, wp::float32>* var_11;
    wp::int32 var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::mat_t<3, 3, wp::float32> var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    //---------
    // forward
    // def _torque(                                                                           <L 1500>
    // bodyid = site_bodyid[objid]                                                            <L 1513>
    var_0 = wp::address(var_site_bodyid, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // cfrc_int = cfrc_int_in[worldid, bodyid]                                                <L 1514>
    var_3 = wp::address(var_cfrc_int_in, var_worldid, var_1);
    var_5 = wp::load(var_3);
    var_4 = wp::copy(var_5);
    // site_xmat = site_xmat_in[worldid, objid]                                               <L 1515>
    var_6 = wp::address(var_site_xmat_in, var_worldid, var_objid);
    var_8 = wp::load(var_6);
    var_7 = wp::copy(var_8);
    // dif = site_xpos_in[worldid, objid] - subtree_com_in[worldid, body_rootid[bodyid]]       <L 1516>
    var_9 = wp::address(var_site_xpos_in, var_worldid, var_objid);
    var_10 = wp::address(var_body_rootid, var_1);
    var_12 = wp::load(var_10);
    var_11 = wp::address(var_subtree_com_in, var_worldid, var_12);
    var_14 = wp::load(var_9);
    var_15 = wp::load(var_11);
    var_13 = wp::sub(var_14, var_15);
    // return wp.transpose(site_xmat) @ (wp.spatial_top(cfrc_int) - wp.cross(dif, wp.spatial_bottom(cfrc_int)))       <L 1517>
    var_16 = wp::transpose(var_7);
    var_17 = wp::spatial_top(var_4);
    var_18 = wp::spatial_bottom(var_4);
    var_19 = wp::cross(var_13, var_18);
    var_20 = wp::sub(var_17, var_19);
    var_21 = wp::mul(var_16, var_20);
    return var_21;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:1520
static CUDA_CALLABLE wp::float32 _actuator_force_0(
    wp::array_t<wp::float32> var_actuator_force_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::float32* var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    //---------
    // forward
    // def _actuator_force(actuator_force_in: wp.array2d[float], worldid: int, objid: int) -> float:       <L 1521>
    // return actuator_force_in[worldid, objid]                                               <L 1522>
    var_0 = wp::address(var_actuator_force_in, var_worldid, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    return var_1;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:1525
static CUDA_CALLABLE wp::float32 _joint_actuator_force_0(
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::float32> var_qfrc_actuator_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::float32* var_1;
    wp::int32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    //---------
    // forward
    // def _joint_actuator_force(                                                             <L 1526>
    // return qfrc_actuator_in[worldid, jnt_dofadr[objid]]                                    <L 1535>
    var_0 = wp::address(var_jnt_dofadr, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::address(var_qfrc_actuator_in, var_worldid, var_2);
    var_4 = wp::load(var_1);
    var_3 = wp::copy(var_4);
    return var_3;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:1618
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _framelinacc_0(
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cacc_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::int32 var_objtype)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    bool var_1;
    wp::int32 var_2;
    wp::vec_t<3, wp::float32>* var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    const wp::int32 var_6 = 2;
    bool var_7;
    wp::int32 var_8;
    wp::vec_t<3, wp::float32>* var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::int32 var_12;
    wp::vec_t<3, wp::float32> var_13;
    const wp::int32 var_14 = 5;
    bool var_15;
    wp::int32* var_16;
    wp::int32 var_17;
    wp::int32 var_18;
    wp::vec_t<3, wp::float32>* var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::int32 var_22;
    wp::vec_t<3, wp::float32> var_23;
    const wp::int32 var_24 = 6;
    bool var_25;
    wp::int32* var_26;
    wp::int32 var_27;
    wp::int32 var_28;
    wp::vec_t<3, wp::float32>* var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::int32 var_32;
    wp::vec_t<3, wp::float32> var_33;
    const wp::int32 var_34 = 7;
    bool var_35;
    wp::int32* var_36;
    wp::int32 var_37;
    wp::int32 var_38;
    wp::vec_t<3, wp::float32>* var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::vec_t<3, wp::float32> var_41;
    wp::int32 var_42;
    wp::vec_t<3, wp::float32> var_43;
    const wp::int32 var_44 = 0;
    const wp::float32 var_45 = 0.0;
    wp::vec_t<3, wp::float32> var_46;
    wp::int32 var_47;
    wp::vec_t<3, wp::float32> var_48;
    wp::int32 var_49;
    wp::vec_t<3, wp::float32> var_50;
    wp::int32 var_51;
    wp::vec_t<3, wp::float32> var_52;
    wp::int32 var_53;
    wp::vec_t<3, wp::float32> var_54;
    wp::int32 var_55;
    wp::vec_t<3, wp::float32> var_56;
    wp::vec_t<6, wp::float32>* var_57;
    wp::vec_t<6, wp::float32> var_58;
    wp::vec_t<6, wp::float32> var_59;
    wp::vec_t<6, wp::float32>* var_60;
    wp::vec_t<6, wp::float32> var_61;
    wp::vec_t<6, wp::float32> var_62;
    wp::int32* var_63;
    wp::vec_t<3, wp::float32>* var_64;
    wp::int32 var_65;
    wp::vec_t<3, wp::float32> var_66;
    wp::vec_t<3, wp::float32> var_67;
    wp::vec_t<3, wp::float32> var_68;
    wp::vec_t<3, wp::float32> var_69;
    wp::vec_t<3, wp::float32> var_70;
    wp::vec_t<3, wp::float32> var_71;
    wp::vec_t<3, wp::float32> var_72;
    wp::vec_t<3, wp::float32> var_73;
    wp::vec_t<3, wp::float32> var_74;
    wp::vec_t<3, wp::float32> var_75;
    wp::vec_t<3, wp::float32> var_76;
    wp::vec_t<3, wp::float32> var_77;
    //---------
    // forward
    // def _framelinacc(                                                                      <L 1619>
    // if objtype == ObjType.BODY:                                                            <L 1639>
    var_1 = (var_objtype == var_0);
    if (var_1) {
        // bodyid = objid                                                                     <L 1640>
        var_2 = wp::copy(var_objid);
        // pos = xipos_in[worldid, objid]                                                     <L 1641>
        var_3 = wp::address(var_xipos_in, var_worldid, var_objid);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
    }
    if (!var_1) {
        // elif objtype == ObjType.XBODY:                                                     <L 1642>
        var_7 = (var_objtype == var_6);
        if (var_7) {
            // bodyid = objid                                                                 <L 1643>
            var_8 = wp::copy(var_objid);
            // pos = xpos_in[worldid, objid]                                                  <L 1644>
            var_9 = wp::address(var_xpos_in, var_worldid, var_objid);
            var_11 = wp::load(var_9);
            var_10 = wp::copy(var_11);
        }
        var_12 = wp::where(var_7, var_8, var_2);
        var_13 = wp::where(var_7, var_10, var_4);
        if (!var_7) {
            // elif objtype == ObjType.GEOM:                                                  <L 1645>
            var_15 = (var_objtype == var_14);
            if (var_15) {
                // bodyid = geom_bodyid[objid]                                                <L 1646>
                var_16 = wp::address(var_geom_bodyid, var_objid);
                var_18 = wp::load(var_16);
                var_17 = wp::copy(var_18);
                // pos = geom_xpos_in[worldid, objid]                                         <L 1647>
                var_19 = wp::address(var_geom_xpos_in, var_worldid, var_objid);
                var_21 = wp::load(var_19);
                var_20 = wp::copy(var_21);
            }
            var_22 = wp::where(var_15, var_17, var_12);
            var_23 = wp::where(var_15, var_20, var_13);
            if (!var_15) {
                // elif objtype == ObjType.SITE:                                              <L 1648>
                var_25 = (var_objtype == var_24);
                if (var_25) {
                    // bodyid = site_bodyid[objid]                                            <L 1649>
                    var_26 = wp::address(var_site_bodyid, var_objid);
                    var_28 = wp::load(var_26);
                    var_27 = wp::copy(var_28);
                    // pos = site_xpos_in[worldid, objid]                                     <L 1650>
                    var_29 = wp::address(var_site_xpos_in, var_worldid, var_objid);
                    var_31 = wp::load(var_29);
                    var_30 = wp::copy(var_31);
                }
                var_32 = wp::where(var_25, var_27, var_22);
                var_33 = wp::where(var_25, var_30, var_23);
                if (!var_25) {
                    // elif objtype == ObjType.CAMERA:                                        <L 1651>
                    var_35 = (var_objtype == var_34);
                    if (var_35) {
                        // bodyid = cam_bodyid[objid]                                         <L 1652>
                        var_36 = wp::address(var_cam_bodyid, var_objid);
                        var_38 = wp::load(var_36);
                        var_37 = wp::copy(var_38);
                        // pos = cam_xpos_in[worldid, objid]                                  <L 1653>
                        var_39 = wp::address(var_cam_xpos_in, var_worldid, var_objid);
                        var_41 = wp::load(var_39);
                        var_40 = wp::copy(var_41);
                    }
                    var_42 = wp::where(var_35, var_37, var_32);
                    var_43 = wp::where(var_35, var_40, var_33);
                    if (!var_35) {
                        // bodyid = 0                                                         <L 1655>
                        // pos = wp.vec3(0.0)                                                 <L 1656>
                        var_46 = wp::vec_t<3, wp::float32>(var_45);
                    }
                    var_47 = wp::where(var_35, var_42, var_44);
                    var_48 = wp::where(var_35, var_43, var_46);
                }
                var_49 = wp::where(var_25, var_32, var_47);
                var_50 = wp::where(var_25, var_33, var_48);
            }
            var_51 = wp::where(var_15, var_22, var_49);
            var_52 = wp::where(var_15, var_23, var_50);
        }
        var_53 = wp::where(var_7, var_12, var_51);
        var_54 = wp::where(var_7, var_13, var_52);
    }
    var_55 = wp::where(var_1, var_2, var_53);
    var_56 = wp::where(var_1, var_4, var_54);
    // cacc = cacc_in[worldid, bodyid]                                                        <L 1658>
    var_57 = wp::address(var_cacc_in, var_worldid, var_55);
    var_59 = wp::load(var_57);
    var_58 = wp::copy(var_59);
    // cvel = cvel_in[worldid, bodyid]                                                        <L 1659>
    var_60 = wp::address(var_cvel_in, var_worldid, var_55);
    var_62 = wp::load(var_60);
    var_61 = wp::copy(var_62);
    // offset = pos - subtree_com_in[worldid, body_rootid[bodyid]]                            <L 1660>
    var_63 = wp::address(var_body_rootid, var_55);
    var_65 = wp::load(var_63);
    var_64 = wp::address(var_subtree_com_in, var_worldid, var_65);
    var_67 = wp::load(var_64);
    var_66 = wp::sub(var_56, var_67);
    // ang = wp.spatial_top(cvel)                                                             <L 1661>
    var_68 = wp::spatial_top(var_61);
    // lin = wp.spatial_bottom(cvel) - wp.cross(offset, ang)                                  <L 1662>
    var_69 = wp::spatial_bottom(var_61);
    var_70 = wp::cross(var_66, var_68);
    var_71 = wp::sub(var_69, var_70);
    // acc = wp.spatial_bottom(cacc) - wp.cross(offset, wp.spatial_top(cacc))                 <L 1663>
    var_72 = wp::spatial_bottom(var_58);
    var_73 = wp::spatial_top(var_58);
    var_74 = wp::cross(var_66, var_73);
    var_75 = wp::sub(var_72, var_74);
    // correction = wp.cross(ang, lin)                                                        <L 1664>
    var_76 = wp::cross(var_68, var_71);
    // return acc + correction                                                                <L 1666>
    var_77 = wp::add(var_75, var_76);
    return var_77;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:1669
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _frameangacc_0(
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cacc_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::int32 var_objtype)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    bool var_1;
    const wp::int32 var_2 = 2;
    bool var_3;
    bool var_4;
    wp::int32 var_5;
    const wp::int32 var_6 = 5;
    bool var_7;
    wp::int32* var_8;
    wp::int32 var_9;
    wp::int32 var_10;
    wp::int32 var_11;
    const wp::int32 var_12 = 6;
    bool var_13;
    wp::int32* var_14;
    wp::int32 var_15;
    wp::int32 var_16;
    wp::int32 var_17;
    const wp::int32 var_18 = 7;
    bool var_19;
    wp::int32* var_20;
    wp::int32 var_21;
    wp::int32 var_22;
    wp::int32 var_23;
    const wp::int32 var_24 = 0;
    wp::int32 var_25;
    wp::int32 var_26;
    wp::int32 var_27;
    wp::int32 var_28;
    wp::vec_t<6, wp::float32>* var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<6, wp::float32> var_31;
    //---------
    // forward
    // def _frameangacc(                                                                      <L 1670>
    // if objtype == ObjType.BODY or objtype == ObjType.XBODY:                                <L 1682>
    var_1 = (var_objtype == var_0);
    var_3 = (var_objtype == var_2);
    var_4 = var_1 || var_3;
    if (var_4) {
        // bodyid = objid                                                                     <L 1683>
        var_5 = wp::copy(var_objid);
    }
    if (!var_4) {
        // elif objtype == ObjType.GEOM:                                                      <L 1684>
        var_7 = (var_objtype == var_6);
        if (var_7) {
            // bodyid = geom_bodyid[objid]                                                    <L 1685>
            var_8 = wp::address(var_geom_bodyid, var_objid);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
        }
        var_11 = wp::where(var_7, var_9, var_5);
        if (!var_7) {
            // elif objtype == ObjType.SITE:                                                  <L 1686>
            var_13 = (var_objtype == var_12);
            if (var_13) {
                // bodyid = site_bodyid[objid]                                                <L 1687>
                var_14 = wp::address(var_site_bodyid, var_objid);
                var_16 = wp::load(var_14);
                var_15 = wp::copy(var_16);
            }
            var_17 = wp::where(var_13, var_15, var_11);
            if (!var_13) {
                // elif objtype == ObjType.CAMERA:                                            <L 1688>
                var_19 = (var_objtype == var_18);
                if (var_19) {
                    // bodyid = cam_bodyid[objid]                                             <L 1689>
                    var_20 = wp::address(var_cam_bodyid, var_objid);
                    var_22 = wp::load(var_20);
                    var_21 = wp::copy(var_22);
                }
                var_23 = wp::where(var_19, var_21, var_17);
                if (!var_19) {
                    // bodyid = 0                                                             <L 1691>
                }
                var_25 = wp::where(var_19, var_23, var_24);
            }
            var_26 = wp::where(var_13, var_17, var_25);
        }
        var_27 = wp::where(var_7, var_11, var_26);
    }
    var_28 = wp::where(var_4, var_5, var_27);
    // return wp.spatial_top(cacc_in[worldid, bodyid])                                        <L 1693>
    var_29 = wp::address(var_cacc_in, var_worldid, var_28);
    var_31 = wp::load(var_29);
    var_30 = wp::spatial_top(var_31);
    return var_30;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:2255
static CUDA_CALLABLE bool _check_match_0(
    wp::array_t<wp::int32> var_body_parentid,
    wp::int32 var_body,
    wp::int32 var_geom,
    wp::int32 var_objtype,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    bool var_1;
    const bool var_2 = true;
    const wp::int32 var_3 = 6;
    bool var_4;
    const bool var_5 = true;
    const wp::int32 var_6 = 5;
    bool var_7;
    bool var_8;
    const wp::int32 var_9 = 1;
    bool var_10;
    bool var_11;
    const wp::int32 var_12 = 2;
    bool var_13;
    bool var_14;
    wp::int32* var_15;
    wp::int32 var_16;
    wp::int32 var_17;
    bool var_18;
    const bool var_19 = false;
    //---------
    // forward
    // def _check_match(body_parentid: wp.array[int], body: int, geom: int, objtype: int, objid: int) -> bool:       <L 2256>
    // if objtype == ObjType.UNKNOWN:                                                         <L 2258>
    var_1 = (var_objtype == var_0);
    if (var_1) {
        // return True                                                                        <L 2259>
        return var_2;
    }
    // if objtype == ObjType.SITE:                                                            <L 2260>
    var_4 = (var_objtype == var_3);
    if (var_4) {
        // return True  # already passed site filter test                                     <L 2261>
        return var_5;
    }
    // if objtype == ObjType.GEOM:                                                            <L 2262>
    var_7 = (var_objtype == var_6);
    if (var_7) {
        // return objid == geom                                                               <L 2263>
        var_8 = (var_objid == var_geom);
        return var_8;
    }
    // if objtype == ObjType.BODY:                                                            <L 2264>
    var_10 = (var_objtype == var_9);
    if (var_10) {
        // return objid == body                                                               <L 2265>
        var_11 = (var_objid == var_body);
        return var_11;
    }
    // if objtype == ObjType.XBODY:                                                           <L 2266>
    var_13 = (var_objtype == var_12);
    if (var_13) {
        // while body > objid:                                                                <L 2268>
    start_while_4:;
        var_14 = (var_body > var_objid);
    if ((var_14) == false) goto end_while_4;
            // body = body_parentid[body]                                                     <L 2269>
            var_15 = wp::address(var_body_parentid, var_body);
            var_17 = wp::load(var_15);
            var_16 = wp::copy(var_17);
            wp::assign(var_body, var_16);
    goto start_while_4;
    end_while_4:;
        // return body == objid                                                               <L 2270>
        var_18 = (var_body == var_objid);
        return var_18;
    }
    // return False                                                                           <L 2271>
    return var_19;
}


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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_sdf.py:80
static CUDA_CALLABLE void get_sdf_params_0(
    wp::array_t<wp::vec_t<8, wp::int32>> var_oct_child,
    wp::array_t<wp::vec_t<3, wp::float32>> var_oct_aabb,
    wp::array_t<wp::vec_t<8, wp::float32>> var_oct_coeff,
    wp::array_t<wp::int32> var_mesh_octadr,
    wp::array_t<wp::int32> var_plugin,
    wp::array_t<wp::vec_t<128, wp::float32>> var_plugin_attr,
    wp::int32 var_g_type,
    wp::vec_t<3, wp::float32> var_g_size,
    wp::int32 var_plugin_id,
    wp::int32 var_mesh_id,
    wp::vec_t<128, wp::float32> & ret_0,
    wp::int32 & ret_1,
    VolumeData_53ac1a2d & ret_2,
    MeshData_52eaa0fa & ret_3)
{
    //---------
    // primal vars
    wp::vec_t<128, wp::float32> var_0;
    const wp::int32 var_1 = 0;
    wp::float32 var_2;
    const wp::int32 var_3 = 0;
    const wp::int32 var_4 = 1;
    wp::float32 var_5;
    const wp::int32 var_6 = 1;
    const wp::int32 var_7 = 2;
    wp::float32 var_8;
    const wp::int32 var_9 = 2;
    const wp::int32 var_10 = 1;
    const wp::int32 var_11 = -1;
    VolumeData_53ac1a2d var_12;
    const wp::int32 var_13 = 8;
    bool var_14;
    const wp::int32 var_15 = 1;
    const wp::int32 var_16 = -1;
    bool var_17;
    bool var_18;
    wp::vec_t<128, wp::float32>* var_19;
    wp::vec_t<128, wp::float32> var_20;
    wp::vec_t<128, wp::float32> var_21;
    wp::int32* var_22;
    wp::int32 var_23;
    wp::int32 var_24;
    wp::vec_t<128, wp::float32> var_25;
    wp::int32 var_26;
    const wp::int32 var_27 = 8;
    bool var_28;
    const wp::int32 var_29 = 1;
    const wp::int32 var_30 = -1;
    bool var_31;
    bool var_32;
    wp::int32* var_33;
    wp::int32 var_34;
    wp::int32 var_35;
    const wp::int32 var_36 = 0;
    wp::vec_t<3, wp::float32>* var_37;
    wp::vec_t<3, wp::float32>* var_38;
    wp::vec_t<3, wp::float32> var_39;
    const wp::int32 var_40 = 1;
    wp::vec_t<3, wp::float32>* var_41;
    wp::vec_t<3, wp::float32>* var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::int32* var_44;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_45;
    wp::array_t<wp::vec_t<8, wp::int32>>* var_46;
    wp::array_t<wp::vec_t<8, wp::float32>>* var_47;
    const bool var_48 = true;
    bool* var_49;
    const wp::int32 var_50 = 7;
    bool var_51;
    const wp::int32 var_52 = 1;
    const wp::int32 var_53 = -1;
    bool var_54;
    wp::int32* var_55;
    const wp::int32 var_56 = 1;
    const wp::int32 var_57 = -1;
    bool var_58;
    wp::int32 var_59;
    bool var_60;
    wp::int32* var_61;
    wp::int32 var_62;
    wp::int32 var_63;
    const wp::int32 var_64 = 0;
    wp::vec_t<3, wp::float32>* var_65;
    wp::vec_t<3, wp::float32>* var_66;
    wp::vec_t<3, wp::float32> var_67;
    const wp::int32 var_68 = 1;
    wp::vec_t<3, wp::float32>* var_69;
    wp::vec_t<3, wp::float32>* var_70;
    wp::vec_t<3, wp::float32> var_71;
    wp::int32* var_72;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_73;
    wp::array_t<wp::vec_t<8, wp::int32>>* var_74;
    wp::array_t<wp::vec_t<8, wp::float32>>* var_75;
    const bool var_76 = true;
    bool* var_77;
    wp::int32 var_78;
    wp::int32 var_79;
    MeshData_52eaa0fa var_80;
    //---------
    // forward
    // def get_sdf_params(                                                                    <L 81>
    // attributes = vec_pluginattr()                                                          <L 96>
    var_0 = wp::vec_t<128, wp::float32>();
    // attributes[0] = g_size[0]                                                              <L 97>
    var_2 = wp::extract(var_g_size, var_1);
    wp::assign_inplace(var_0, var_3, var_2);
    // attributes[1] = g_size[1]                                                              <L 98>
    var_5 = wp::extract(var_g_size, var_4);
    wp::assign_inplace(var_0, var_6, var_5);
    // attributes[2] = g_size[2]                                                              <L 99>
    var_8 = wp::extract(var_g_size, var_7);
    wp::assign_inplace(var_0, var_9, var_8);
    // plugin_index = -1                                                                      <L 100>
    // volume_data = VolumeData()                                                             <L 101>
    var_12 = VolumeData_53ac1a2d();
    // if g_type == GeomType.SDF and plugin_id != -1:                                         <L 103>
    var_14 = (var_g_type == var_13);
    var_17 = (var_plugin_id != var_16);
    var_18 = var_14 && var_17;
    if (var_18) {
        // attributes = plugin_attr[plugin_id]                                                <L 104>
        var_19 = wp::address(var_plugin_attr, var_plugin_id);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // plugin_index = plugin[plugin_id]                                                   <L 105>
        var_22 = wp::address(var_plugin, var_plugin_id);
        var_24 = wp::load(var_22);
        var_23 = wp::copy(var_24);
    }
    var_25 = wp::where(var_18, var_20, var_0);
    var_26 = wp::where(var_18, var_23, var_11);
    if (!var_18) {
        // elif g_type == GeomType.SDF and mesh_id != -1:                                     <L 107>
        var_28 = (var_g_type == var_27);
        var_31 = (var_mesh_id != var_30);
        var_32 = var_28 && var_31;
        if (var_32) {
            // octadr = mesh_octadr[mesh_id]                                                  <L 108>
            var_33 = wp::address(var_mesh_octadr, var_mesh_id);
            var_35 = wp::load(var_33);
            var_34 = wp::copy(var_35);
            // volume_data.center = oct_aabb[octadr, 0]                                       <L 109>
            var_37 = wp::address(var_oct_aabb, var_34, var_36);
            var_38 = &(var_12.center);
            var_39 = wp::load(var_37);
            wp::store(var_38, var_39);
            // volume_data.half_size = oct_aabb[octadr, 1]                                    <L 110>
            var_41 = wp::address(var_oct_aabb, var_34, var_40);
            var_42 = &(var_12.half_size);
            var_43 = wp::load(var_41);
            wp::store(var_42, var_43);
            // volume_data.root = octadr                                                      <L 111>
            var_44 = &(var_12.root);
            wp::store(var_44, var_34);
            // volume_data.oct_aabb = oct_aabb                                                <L 112>
            var_45 = &(var_12.oct_aabb);
            wp::store(var_45, var_oct_aabb);
            // volume_data.oct_child = oct_child                                              <L 113>
            var_46 = &(var_12.oct_child);
            wp::store(var_46, var_oct_child);
            // volume_data.oct_coeff = oct_coeff                                              <L 114>
            var_47 = &(var_12.oct_coeff);
            wp::store(var_47, var_oct_coeff);
            // volume_data.valid = True                                                       <L 115>
            var_49 = &(var_12.valid);
            wp::store(var_49, var_48);
        }
        if (!var_32) {
            // elif g_type == GeomType.MESH and mesh_id != -1 and mesh_octadr[mesh_id] != -1:       <L 117>
            var_51 = (var_g_type == var_50);
            var_54 = (var_mesh_id != var_53);
            var_55 = wp::address(var_mesh_octadr, var_mesh_id);
            var_59 = wp::load(var_55);
            var_58 = (var_59 != var_57);
            var_60 = var_51 && var_54 && var_58;
            if (var_60) {
                // octadr = mesh_octadr[mesh_id]                                              <L 118>
                var_61 = wp::address(var_mesh_octadr, var_mesh_id);
                var_63 = wp::load(var_61);
                var_62 = wp::copy(var_63);
                // volume_data.center = oct_aabb[octadr, 0]                                   <L 119>
                var_65 = wp::address(var_oct_aabb, var_62, var_64);
                var_66 = &(var_12.center);
                var_67 = wp::load(var_65);
                wp::store(var_66, var_67);
                // volume_data.half_size = oct_aabb[octadr, 1]                                <L 120>
                var_69 = wp::address(var_oct_aabb, var_62, var_68);
                var_70 = &(var_12.half_size);
                var_71 = wp::load(var_69);
                wp::store(var_70, var_71);
                // volume_data.root = octadr                                                  <L 121>
                var_72 = &(var_12.root);
                wp::store(var_72, var_62);
                // volume_data.oct_aabb = oct_aabb                                            <L 122>
                var_73 = &(var_12.oct_aabb);
                wp::store(var_73, var_oct_aabb);
                // volume_data.oct_child = oct_child                                          <L 123>
                var_74 = &(var_12.oct_child);
                wp::store(var_74, var_oct_child);
                // volume_data.oct_coeff = oct_coeff                                          <L 124>
                var_75 = &(var_12.oct_coeff);
                wp::store(var_75, var_oct_coeff);
                // volume_data.valid = True                                                   <L 125>
                var_77 = &(var_12.valid);
                wp::store(var_77, var_76);
            }
            var_78 = wp::where(var_60, var_62, var_34);
        }
        var_79 = wp::where(var_32, var_34, var_78);
    }
    // return attributes, plugin_index, volume_data, MeshData()                               <L 127>
    var_80 = MeshData_52eaa0fa();
    ret_0 = var_25;
    ret_1 = var_26;
    ret_2 = var_12;
    ret_3 = var_80;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_sdf.py:157
static CUDA_CALLABLE wp::float32 sphere_0(
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> var_size)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::int32 var_1 = 0;
    wp::float32 var_2;
    wp::float32 var_3;
    //---------
    // forward
    // def sphere(p: wp.vec3, size: wp.vec3) -> float:                                        <L 158>
    // return wp.length(p) - size[0]                                                          <L 159>
    var_0 = wp::length(var_p);
    var_2 = wp::extract(var_size, var_1);
    var_3 = wp::sub(var_0, var_2);
    return var_3;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_sdf.py:147
static CUDA_CALLABLE wp::vec_t<3, wp::float32> radial_field_0(
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> var_x,
    wp::vec_t<3, wp::float32> var_size)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    wp::float32 var_5;
    const wp::int32 var_6 = 0;
    const wp::int32 var_7 = 0;
    wp::float32 var_8;
    const wp::int32 var_9 = 0;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    const wp::int32 var_13 = 0;
    const wp::int32 var_14 = 1;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::int32 var_17 = 1;
    const wp::int32 var_18 = 1;
    wp::float32 var_19;
    const wp::int32 var_20 = 1;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    const wp::int32 var_24 = 1;
    const wp::int32 var_25 = 2;
    wp::float32 var_26;
    wp::float32 var_27;
    const wp::int32 var_28 = 2;
    const wp::int32 var_29 = 2;
    wp::float32 var_30;
    const wp::int32 var_31 = 2;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    const wp::int32 var_35 = 2;
    //---------
    // forward
    // def radial_field(a: wp.vec3, x: wp.vec3, size: wp.vec3) -> wp.vec3:                    <L 148>
    // field = wp.cw_div(-size, a)                                                            <L 149>
    var_0 = wp::neg(var_size);
    var_1 = wp::cw_div(var_0, var_a);
    // field = wp.normalize(field)                                                            <L 150>
    var_2 = wp::normalize(var_1);
    // field[0] *= wp.sign(x[0])                                                              <L 151>
    var_4 = wp::extract(var_x, var_3);
    var_5 = wp::sign(var_4);
    var_8 = wp::extract(var_2, var_7);
    var_10 = wp::extract(var_x, var_9);
    var_11 = wp::sign(var_10);
    var_12 = wp::mul(var_8, var_11);
    wp::assign_inplace(var_2, var_13, var_12);
    // field[1] *= wp.sign(x[1])                                                              <L 152>
    var_15 = wp::extract(var_x, var_14);
    var_16 = wp::sign(var_15);
    var_19 = wp::extract(var_2, var_18);
    var_21 = wp::extract(var_x, var_20);
    var_22 = wp::sign(var_21);
    var_23 = wp::mul(var_19, var_22);
    wp::assign_inplace(var_2, var_24, var_23);
    // field[2] *= wp.sign(x[2])                                                              <L 153>
    var_26 = wp::extract(var_x, var_25);
    var_27 = wp::sign(var_26);
    var_30 = wp::extract(var_2, var_29);
    var_32 = wp::extract(var_x, var_31);
    var_33 = wp::sign(var_32);
    var_34 = wp::mul(var_30, var_33);
    wp::assign_inplace(var_2, var_35, var_34);
    // return field                                                                           <L 154>
    return var_2;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_sdf.py:162
static CUDA_CALLABLE wp::float32 box_0(
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> var_size)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    const wp::int32 var_4 = 0;
    bool var_5;
    const wp::int32 var_6 = 1;
    wp::float32 var_7;
    const wp::int32 var_8 = 0;
    bool var_9;
    const wp::int32 var_10 = 2;
    wp::float32 var_11;
    const wp::int32 var_12 = 0;
    bool var_13;
    bool var_14;
    const wp::float32 var_15 = 0.0;
    const wp::float32 var_16 = 0.0;
    const wp::float32 var_17 = 0.0;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    const wp::float32 var_22 = 0.0;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    //---------
    // forward
    // def box(p: wp.vec3, size: wp.vec3) -> float:                                           <L 163>
    // a = wp.abs(p) - size                                                                   <L 164>
    var_0 = wp::abs(var_p);
    var_1 = wp::sub(var_0, var_size);
    // if a[0] >= 0 or a[1] >= 0 or a[2] >= 0:                                                <L 165>
    var_3 = wp::extract(var_1, var_2);
    var_5 = (var_3 >= var_4);
    var_7 = wp::extract(var_1, var_6);
    var_9 = (var_7 >= var_8);
    var_11 = wp::extract(var_1, var_10);
    var_13 = (var_11 >= var_12);
    var_14 = var_5 || var_9 || var_13;
    if (var_14) {
        // z = wp.vec3(0.0, 0.0, 0.0)                                                         <L 166>
        var_18 = wp::vec_t<3, wp::float32>(var_15, var_16, var_17);
        // b = wp.max(a, z)                                                                   <L 167>
        var_19 = wp::max(var_1, var_18);
        // return wp.norm_l2(b) + wp.min(wp.max(a), 0.0)                                      <L 168>
        var_20 = norm_l2_0(var_19);
        var_21 = wp::max(var_1);
        var_23 = wp::min(var_21, var_22);
        var_24 = wp::add(var_20, var_23);
        return var_24;
    }
    // b = radial_field(a, p, size)                                                           <L 169>
    var_25 = radial_field_0(var_1, var_p, var_size);
    // t = -wp.cw_div(a, wp.abs(b))                                                           <L 170>
    var_26 = wp::abs(var_25);
    var_27 = wp::cw_div(var_1, var_26);
    var_28 = wp::neg(var_27);
    // return -wp.min(t) * wp.norm_l2(b)                                                      <L 171>
    var_29 = wp::min(var_28);
    var_30 = wp::neg(var_29);
    var_31 = norm_l2_0(var_25);
    var_32 = wp::mul(var_30, var_31);
    return var_32;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_sdf.py:174
static CUDA_CALLABLE wp::float32 ellipsoid_0(
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> var_size)
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
    const wp::int32 var_10 = 2;
    wp::float32 var_11;
    const wp::int32 var_12 = 2;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::float32 var_16;
    const wp::int32 var_17 = 0;
    wp::float32 var_18;
    const wp::int32 var_19 = 0;
    wp::float32 var_20;
    const wp::float32 var_21 = 2.0;
    wp::float32 var_22;
    wp::float32 var_23;
    const wp::int32 var_24 = 1;
    wp::float32 var_25;
    const wp::int32 var_26 = 1;
    wp::float32 var_27;
    const wp::float32 var_28 = 2.0;
    wp::float32 var_29;
    wp::float32 var_30;
    const wp::int32 var_31 = 2;
    wp::float32 var_32;
    const wp::int32 var_33 = 2;
    wp::float32 var_34;
    const wp::float32 var_35 = 2.0;
    wp::float32 var_36;
    wp::float32 var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::float32 var_39;
    const wp::float32 var_40 = 0.0;
    bool var_41;
    wp::float32 var_42;
    const wp::float32 var_43 = 1e-12;
    wp::float32 var_44;
    const wp::float32 var_45 = 1.0;
    wp::float32 var_46;
    wp::float32 var_47;
    wp::float32 var_48;
    //---------
    // forward
    // def ellipsoid(p: wp.vec3, size: wp.vec3) -> float:                                     <L 175>
    // scaled_p = wp.vec3(p[0] / size[0], p[1] / size[1], p[2] / size[2])                     <L 176>
    var_1 = wp::extract(var_p, var_0);
    var_3 = wp::extract(var_size, var_2);
    var_4 = wp::div(var_1, var_3);
    var_6 = wp::extract(var_p, var_5);
    var_8 = wp::extract(var_size, var_7);
    var_9 = wp::div(var_6, var_8);
    var_11 = wp::extract(var_p, var_10);
    var_13 = wp::extract(var_size, var_12);
    var_14 = wp::div(var_11, var_13);
    var_15 = wp::vec_t<3, wp::float32>(var_4, var_9, var_14);
    // k0 = wp.length(scaled_p)                                                               <L 177>
    var_16 = wp::length(var_15);
    // k1 = wp.length(wp.vec3(p[0] / (size[0] ** 2.0), p[1] / (size[1] ** 2.0), p[2] / (size[2] ** 2.0)))       <L 178>
    var_18 = wp::extract(var_p, var_17);
    var_20 = wp::extract(var_size, var_19);
    var_22 = wp::pow(var_20, var_21);
    var_23 = wp::div(var_18, var_22);
    var_25 = wp::extract(var_p, var_24);
    var_27 = wp::extract(var_size, var_26);
    var_29 = wp::pow(var_27, var_28);
    var_30 = wp::div(var_25, var_29);
    var_32 = wp::extract(var_p, var_31);
    var_34 = wp::extract(var_size, var_33);
    var_36 = wp::pow(var_34, var_35);
    var_37 = wp::div(var_32, var_36);
    var_38 = wp::vec_t<3, wp::float32>(var_23, var_30, var_37);
    var_39 = wp::length(var_38);
    // if k1 != 0.0:                                                                          <L 179>
    var_41 = (var_39 != var_40);
    if (var_41) {
        // denom = k1                                                                         <L 180>
        var_42 = wp::copy(var_39);
    }
    if (!var_41) {
        // denom = 1e-12                                                                      <L 182>
    }
    var_44 = wp::where(var_41, var_42, var_43);
    // return k0 * (k0 - 1.0) / denom                                                         <L 183>
    var_46 = wp::sub(var_16, var_45);
    var_47 = wp::mul(var_16, var_46);
    var_48 = wp::div(var_47, var_44);
    return var_48;
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/ray.py:105
static CUDA_CALLABLE void _ray_quad_0(
    wp::float32 var_a,
    wp::float32 var_b,
    wp::float32 var_c,
    wp::float32 & ret_0,
    wp::vec_t<2, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    const wp::float32 var_3 = 1e-15;
    bool var_4;
    const wp::float32 var_5 = 1.0;
    const wp::float32 var_6 = -1.0;
    const wp::float32 var_7 = 1.0;
    const wp::float32 var_8 = -1.0;
    const wp::float32 var_9 = 1.0;
    const wp::float32 var_10 = -1.0;
    wp::vec_t<2, wp::float32> var_11;
    wp::float32 var_12;
    const wp::float32 var_13 = 1.0;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::vec_t<2, wp::float32> var_21;
    const wp::float32 var_22 = 0.0;
    bool var_23;
    const wp::float32 var_24 = 0.0;
    bool var_25;
    const wp::float32 var_26 = 1.0;
    const wp::float32 var_27 = -1.0;
    //---------
    // forward
    // def _ray_quad(a: float, b: float, c: float) -> Tuple[float, wp.vec2]:                  <L 106>
    // det = b * b - a * c                                                                    <L 108>
    var_0 = wp::mul(var_b, var_b);
    var_1 = wp::mul(var_a, var_c);
    var_2 = wp::sub(var_0, var_1);
    // if det < MJ_MINVAL:                                                                    <L 109>
    var_4 = (var_2 < var_3);
    if (var_4) {
        // return -1.0, wp.vec2(-1.0, -1.0)                                                   <L 110>
        var_11 = wp::vec_t<2, wp::float32>(var_8, var_10);
        ret_0 = var_6;
        ret_1 = var_11;
        return;
    }
    // det = wp.sqrt(det)                                                                     <L 111>
    var_12 = wp::sqrt(var_2);
    // den = safe_div(1.0, a)                                                                 <L 114>
    var_14 = safe_div_0(var_13, var_a);
    // x0 = (-b - det) * den                                                                  <L 115>
    var_15 = wp::neg(var_b);
    var_16 = wp::sub(var_15, var_12);
    var_17 = wp::mul(var_16, var_14);
    // x1 = (-b + det) * den                                                                  <L 116>
    var_18 = wp::neg(var_b);
    var_19 = wp::add(var_18, var_12);
    var_20 = wp::mul(var_19, var_14);
    // x = wp.vec2(x0, x1)                                                                    <L 117>
    var_21 = wp::vec_t<2, wp::float32>(var_17, var_20);
    // if x0 >= 0.0:                                                                          <L 120>
    var_23 = (var_17 >= var_22);
    if (var_23) {
        // return x0, x                                                                       <L 121>
        ret_0 = var_17;
        ret_1 = var_21;
        return;
    }
    if (!var_23) {
        // elif x1 >= 0.0:                                                                    <L 122>
        var_25 = (var_20 >= var_24);
        if (var_25) {
            // return x1, x                                                                   <L 123>
            ret_0 = var_20;
            ret_1 = var_21;
            return;
        }
        if (!var_25) {
            // return -1.0, x                                                                 <L 125>
            ret_0 = var_27;
            ret_1 = var_21;
            return;
        }
    }
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/ray.py:211
static CUDA_CALLABLE void ray_sphere_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::float32 var_dist_sqr,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::vec_t<2, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    const wp::int32 var_8 = 0;
    bool var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    //---------
    // forward
    // def ray_sphere(pos: wp.vec3, dist_sqr: float, pnt: wp.vec3, vec: wp.vec3) -> Tuple[float, wp.vec3]:       <L 212>
    // dif = pnt - pos                                                                        <L 214>
    var_0 = wp::sub(var_pnt, var_pos);
    // a = wp.dot(vec, vec)                                                                   <L 216>
    var_1 = wp::dot(var_vec, var_vec);
    // b = wp.dot(vec, dif)                                                                   <L 217>
    var_2 = wp::dot(var_vec, var_0);
    // c = wp.dot(dif, dif) - dist_sqr                                                        <L 218>
    var_3 = wp::dot(var_0, var_0);
    var_4 = wp::sub(var_3, var_dist_sqr);
    // sol, _ = _ray_quad(a, b, c)                                                            <L 220>
    _ray_quad_0(var_1, var_2, var_4, var_5, var_6);
    // normal = wp.vec3()                                                                     <L 221>
    var_7 = wp::vec_t<3, wp::float32>();
    // if sol >= 0:                                                                           <L 222>
    var_9 = (var_5 >= var_8);
    if (var_9) {
        // s = pnt + vec * sol                                                                <L 223>
        var_10 = wp::mul(var_vec, var_5);
        var_11 = wp::add(var_pnt, var_10);
        // normal = wp.normalize(s - pos)                                                     <L 224>
        var_12 = wp::sub(var_11, var_pos);
        var_13 = wp::normalize(var_12);
    }
    var_14 = wp::where(var_9, var_13, var_7);
    // return sol, normal                                                                     <L 225>
    ret_0 = var_5;
    ret_1 = var_14;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/ray.py:32
static CUDA_CALLABLE void _ray_map_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::mat_t<3, 3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    //---------
    // forward
    // def _ray_map(pos: wp.vec3, mat: wp.mat33, pnt: wp.vec3, vec: wp.vec3) -> Tuple[wp.vec3, wp.vec3]:       <L 33>
    // matT = wp.transpose(mat)                                                               <L 45>
    var_0 = wp::transpose(var_mat);
    // lpnt = matT @ (pnt - pos)                                                              <L 46>
    var_1 = wp::sub(var_pnt, var_pos);
    var_2 = wp::mul(var_0, var_1);
    // lvec = matT @ vec                                                                      <L 47>
    var_3 = wp::mul(var_0, var_vec);
    // return lpnt, lvec                                                                      <L 49>
    ret_0 = var_2;
    ret_1 = var_3;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/ray.py:397
static CUDA_CALLABLE void ray_box_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 & ret_0,
    wp::vec_t<6, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 1.0;
    const wp::float32 var_1 = -1.0;
    const wp::float32 var_2 = 1.0;
    const wp::float32 var_3 = -1.0;
    const wp::float32 var_4 = 1.0;
    const wp::float32 var_5 = -1.0;
    const wp::float32 var_6 = 1.0;
    const wp::float32 var_7 = -1.0;
    const wp::float32 var_8 = 1.0;
    const wp::float32 var_9 = -1.0;
    const wp::float32 var_10 = 1.0;
    const wp::float32 var_11 = -1.0;
    wp::vec_t<6, wp::float32> var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::vec_t<3, wp::float32> var_15;
    const wp::int32 var_16 = 0;
    bool var_17;
    const wp::float32 var_18 = 1.0;
    const wp::float32 var_19 = -1.0;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    const wp::float32 var_23 = 1.0;
    const wp::float32 var_24 = -1.0;
    wp::float32 var_25;
    const wp::int32 var_26 = 1;
    const wp::int32 var_27 = -1;
    const wp::int32 var_28 = 1;
    const wp::int32 var_29 = -1;
    const wp::int32 var_30 = 0;
    wp::float32 var_31;
    wp::float32 var_32;
    const wp::float32 var_33 = 1e-15;
    bool var_34;
    const wp::int32 var_35 = -1;
    wp::float32 var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    const wp::float32 var_43 = 0.0;
    bool var_44;
    const wp::mat_t<3, 2, wp::int32> var_45 = wp::initializer_array<6,wp::int32>{1, 2, 0, 2, 0, 1};
    const wp::int32 var_46 = 0;
    wp::int32 var_47;
    const wp::int32 var_48 = 1;
    wp::int32 var_49;
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
    bool var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    bool var_63;
    bool var_64;
    const wp::float32 var_65 = 0.0;
    bool var_66;
    bool var_67;
    bool var_68;
    wp::float32 var_69;
    wp::int32 var_70;
    wp::int32 var_71;
    wp::float32 var_72;
    wp::int32 var_73;
    wp::int32 var_74;
    const wp::int32 var_75 = 2;
    wp::int32 var_76;
    const wp::int32 var_77 = 1;
    wp::int32 var_78;
    const wp::int32 var_79 = 2;
    wp::int32 var_80;
    wp::int32 var_81;
    wp::float32 var_82;
    wp::int32 var_83;
    wp::int32 var_84;
    wp::float32 var_85;
    wp::int32 var_86;
    wp::int32 var_87;
    const wp::int32 var_88 = 1;
    wp::float32 var_89;
    wp::float32 var_90;
    wp::float32 var_91;
    wp::float32 var_92;
    wp::float32 var_93;
    wp::float32 var_94;
    wp::float32 var_95;
    const wp::float32 var_96 = 0.0;
    bool var_97;
    const wp::int32 var_98 = 0;
    wp::int32 var_99;
    const wp::int32 var_100 = 1;
    wp::int32 var_101;
    wp::float32 var_102;
    wp::float32 var_103;
    wp::float32 var_104;
    wp::float32 var_105;
    wp::float32 var_106;
    wp::float32 var_107;
    wp::float32 var_108;
    wp::float32 var_109;
    wp::float32 var_110;
    wp::float32 var_111;
    bool var_112;
    wp::float32 var_113;
    wp::float32 var_114;
    bool var_115;
    bool var_116;
    const wp::float32 var_117 = 0.0;
    bool var_118;
    bool var_119;
    bool var_120;
    wp::float32 var_121;
    wp::int32 var_122;
    wp::int32 var_123;
    wp::float32 var_124;
    wp::int32 var_125;
    wp::int32 var_126;
    const wp::int32 var_127 = 2;
    wp::int32 var_128;
    const wp::int32 var_129 = 1;
    wp::int32 var_130;
    const wp::int32 var_131 = 2;
    wp::int32 var_132;
    wp::int32 var_133;
    wp::float32 var_134;
    wp::int32 var_135;
    wp::int32 var_136;
    wp::float32 var_137;
    wp::int32 var_138;
    wp::int32 var_139;
    wp::int32 var_140;
    wp::int32 var_141;
    wp::float32 var_142;
    wp::float32 var_143;
    wp::float32 var_144;
    wp::int32 var_145;
    wp::int32 var_146;
    const wp::int32 var_147 = 1;
    wp::float32 var_148;
    wp::float32 var_149;
    bool var_150;
    const wp::int32 var_151 = -1;
    wp::float32 var_152;
    wp::float32 var_153;
    wp::float32 var_154;
    wp::float32 var_155;
    wp::float32 var_156;
    wp::float32 var_157;
    wp::float32 var_158;
    const wp::float32 var_159 = 0.0;
    bool var_160;
    const wp::int32 var_161 = 0;
    wp::int32 var_162;
    const wp::int32 var_163 = 1;
    wp::int32 var_164;
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
    bool var_175;
    wp::float32 var_176;
    wp::float32 var_177;
    bool var_178;
    bool var_179;
    const wp::float32 var_180 = 0.0;
    bool var_181;
    bool var_182;
    bool var_183;
    wp::float32 var_184;
    wp::int32 var_185;
    wp::int32 var_186;
    wp::float32 var_187;
    wp::int32 var_188;
    wp::int32 var_189;
    const wp::int32 var_190 = 2;
    wp::int32 var_191;
    const wp::int32 var_192 = 1;
    wp::int32 var_193;
    const wp::int32 var_194 = 2;
    wp::int32 var_195;
    wp::int32 var_196;
    wp::float32 var_197;
    wp::int32 var_198;
    wp::int32 var_199;
    wp::float32 var_200;
    wp::int32 var_201;
    wp::int32 var_202;
    wp::int32 var_203;
    wp::int32 var_204;
    wp::float32 var_205;
    wp::float32 var_206;
    const wp::int32 var_207 = 1;
    wp::float32 var_208;
    wp::float32 var_209;
    wp::float32 var_210;
    wp::float32 var_211;
    wp::float32 var_212;
    wp::float32 var_213;
    wp::float32 var_214;
    const wp::float32 var_215 = 0.0;
    bool var_216;
    const wp::int32 var_217 = 0;
    wp::int32 var_218;
    const wp::int32 var_219 = 1;
    wp::int32 var_220;
    wp::float32 var_221;
    wp::float32 var_222;
    wp::float32 var_223;
    wp::float32 var_224;
    wp::float32 var_225;
    wp::float32 var_226;
    wp::float32 var_227;
    wp::float32 var_228;
    wp::float32 var_229;
    wp::float32 var_230;
    bool var_231;
    wp::float32 var_232;
    wp::float32 var_233;
    bool var_234;
    bool var_235;
    const wp::float32 var_236 = 0.0;
    bool var_237;
    bool var_238;
    bool var_239;
    wp::float32 var_240;
    wp::int32 var_241;
    wp::int32 var_242;
    wp::float32 var_243;
    wp::int32 var_244;
    wp::int32 var_245;
    const wp::int32 var_246 = 2;
    wp::int32 var_247;
    const wp::int32 var_248 = 1;
    wp::int32 var_249;
    const wp::int32 var_250 = 2;
    wp::int32 var_251;
    wp::int32 var_252;
    wp::float32 var_253;
    wp::int32 var_254;
    wp::int32 var_255;
    wp::float32 var_256;
    wp::int32 var_257;
    wp::int32 var_258;
    wp::int32 var_259;
    wp::int32 var_260;
    wp::float32 var_261;
    wp::float32 var_262;
    wp::float32 var_263;
    wp::int32 var_264;
    wp::int32 var_265;
    wp::int32 var_266;
    wp::float32 var_267;
    wp::int32 var_268;
    wp::int32 var_269;
    wp::float32 var_270;
    wp::float32 var_271;
    const wp::int32 var_272 = 2;
    wp::float32 var_273;
    wp::float32 var_274;
    bool var_275;
    const wp::int32 var_276 = -1;
    wp::float32 var_277;
    wp::float32 var_278;
    wp::float32 var_279;
    wp::float32 var_280;
    wp::float32 var_281;
    wp::float32 var_282;
    wp::float32 var_283;
    const wp::float32 var_284 = 0.0;
    bool var_285;
    const wp::int32 var_286 = 0;
    wp::int32 var_287;
    const wp::int32 var_288 = 1;
    wp::int32 var_289;
    wp::float32 var_290;
    wp::float32 var_291;
    wp::float32 var_292;
    wp::float32 var_293;
    wp::float32 var_294;
    wp::float32 var_295;
    wp::float32 var_296;
    wp::float32 var_297;
    wp::float32 var_298;
    wp::float32 var_299;
    bool var_300;
    wp::float32 var_301;
    wp::float32 var_302;
    bool var_303;
    bool var_304;
    const wp::float32 var_305 = 0.0;
    bool var_306;
    bool var_307;
    bool var_308;
    wp::float32 var_309;
    wp::int32 var_310;
    wp::int32 var_311;
    wp::float32 var_312;
    wp::int32 var_313;
    wp::int32 var_314;
    const wp::int32 var_315 = 2;
    wp::int32 var_316;
    const wp::int32 var_317 = 1;
    wp::int32 var_318;
    const wp::int32 var_319 = 2;
    wp::int32 var_320;
    wp::int32 var_321;
    wp::float32 var_322;
    wp::int32 var_323;
    wp::int32 var_324;
    wp::float32 var_325;
    wp::int32 var_326;
    wp::int32 var_327;
    wp::int32 var_328;
    wp::int32 var_329;
    wp::float32 var_330;
    wp::float32 var_331;
    const wp::int32 var_332 = 1;
    wp::float32 var_333;
    wp::float32 var_334;
    wp::float32 var_335;
    wp::float32 var_336;
    wp::float32 var_337;
    wp::float32 var_338;
    wp::float32 var_339;
    const wp::float32 var_340 = 0.0;
    bool var_341;
    const wp::int32 var_342 = 0;
    wp::int32 var_343;
    const wp::int32 var_344 = 1;
    wp::int32 var_345;
    wp::float32 var_346;
    wp::float32 var_347;
    wp::float32 var_348;
    wp::float32 var_349;
    wp::float32 var_350;
    wp::float32 var_351;
    wp::float32 var_352;
    wp::float32 var_353;
    wp::float32 var_354;
    wp::float32 var_355;
    bool var_356;
    wp::float32 var_357;
    wp::float32 var_358;
    bool var_359;
    bool var_360;
    const wp::float32 var_361 = 0.0;
    bool var_362;
    bool var_363;
    bool var_364;
    wp::float32 var_365;
    wp::int32 var_366;
    wp::int32 var_367;
    wp::float32 var_368;
    wp::int32 var_369;
    wp::int32 var_370;
    const wp::int32 var_371 = 2;
    wp::int32 var_372;
    const wp::int32 var_373 = 1;
    wp::int32 var_374;
    const wp::int32 var_375 = 2;
    wp::int32 var_376;
    wp::int32 var_377;
    wp::float32 var_378;
    wp::int32 var_379;
    wp::int32 var_380;
    wp::float32 var_381;
    wp::int32 var_382;
    wp::int32 var_383;
    wp::int32 var_384;
    wp::int32 var_385;
    wp::float32 var_386;
    wp::float32 var_387;
    wp::float32 var_388;
    wp::int32 var_389;
    wp::int32 var_390;
    wp::int32 var_391;
    wp::float32 var_392;
    wp::int32 var_393;
    wp::int32 var_394;
    wp::float32 var_395;
    wp::float32 var_396;
    wp::vec_t<3, wp::float32> var_397;
    const wp::int32 var_398 = 0;
    bool var_399;
    wp::float32 var_400;
    wp::vec_t<3, wp::float32> var_401;
    wp::vec_t<3, wp::float32> var_402;
    //---------
    // forward
    // def ray_box(pos: wp.vec3, mat: wp.mat33, size: wp.vec3, pnt: wp.vec3, vec: wp.vec3) -> Tuple[float, vec6, wp.vec3]:       <L 398>
    // all = vec6(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0)                                         <L 400>
    var_12 = wp::vec_t<6, wp::float32>({var_1, var_3, var_5, var_7, var_9, var_11});
    // ssz = wp.dot(size, size)                                                               <L 403>
    var_13 = wp::dot(var_size, var_size);
    // dist_sphere, _ = ray_sphere(pos, ssz, pnt, vec)                                        <L 404>
    ray_sphere_0(var_pos, var_13, var_pnt, var_vec, var_14, var_15);
    // if dist_sphere < 0:                                                                    <L 405>
    var_17 = (var_14 < var_16);
    if (var_17) {
        // return -1.0, all, wp.vec3()                                                        <L 406>
        var_20 = wp::vec_t<3, wp::float32>();
        ret_0 = var_19;
        ret_1 = var_12;
        ret_2 = var_20;
        return;
    }
    // lpnt, lvec = _ray_map(pos, mat, pnt, vec)                                              <L 409>
    _ray_map_0(var_pos, var_mat, var_pnt, var_vec, var_21, var_22);
    // x = float(-1.0)                                                                        <L 412>
    var_25 = wp::float(var_24);
    // face_side = -1                                                                         <L 413>
    // face_axis = -1                                                                         <L 414>
    // for i in range(3):                                                                     <L 417>
    // if wp.abs(lvec[i]) > MJ_MINVAL:                                                        <L 418>
    var_31 = wp::extract(var_22, var_30);
    var_32 = wp::abs(var_31);
    var_34 = (var_32 > var_33);
    if (var_34) {
        // for side in range(-1, 2, 2):                                                       <L 419>
        // sol = (float(side) * size[i] - lpnt[i]) / lvec[i]                                  <L 421>
        var_36 = wp::float(var_35);
        var_37 = wp::extract(var_size, var_30);
        var_38 = wp::mul(var_36, var_37);
        var_39 = wp::extract(var_21, var_30);
        var_40 = wp::sub(var_38, var_39);
        var_41 = wp::extract(var_22, var_30);
        var_42 = wp::div(var_40, var_41);
        // if sol >= 0.0:                                                                     <L 424>
        var_44 = (var_42 >= var_43);
        if (var_44) {
            // id0 = _IFACE[i][0]                                                             <L 425>
            var_47 = wp::extract(var_45, var_30, var_46);
            // id1 = _IFACE[i][1]                                                             <L 426>
            var_49 = wp::extract(var_45, var_30, var_48);
            // p0 = lpnt[id0] + sol * lvec[id0]                                               <L 429>
            var_50 = wp::extract(var_21, var_47);
            var_51 = wp::extract(var_22, var_47);
            var_52 = wp::mul(var_42, var_51);
            var_53 = wp::add(var_50, var_52);
            // p1 = lpnt[id1] + sol * lvec[id1]                                               <L 430>
            var_54 = wp::extract(var_21, var_49);
            var_55 = wp::extract(var_22, var_49);
            var_56 = wp::mul(var_42, var_55);
            var_57 = wp::add(var_54, var_56);
            // if (wp.abs(p0) <= size[id0]) and (wp.abs(p1) <= size[id1]):                    <L 433>
            var_58 = wp::abs(var_53);
            var_59 = wp::extract(var_size, var_47);
            var_60 = (var_58 <= var_59);
            var_61 = wp::abs(var_57);
            var_62 = wp::extract(var_size, var_49);
            var_63 = (var_61 <= var_62);
            var_64 = var_60 && var_63;
            if (var_64) {
                // if x < 0.0 or sol < x:                                                     <L 435>
                var_66 = (var_25 < var_65);
                var_67 = (var_42 < var_25);
                var_68 = var_66 || var_67;
                if (var_68) {
                    // x = sol                                                                <L 436>
                    var_69 = wp::copy(var_42);
                    // face_axis = i                                                          <L 437>
                    var_70 = wp::copy(var_30);
                    // face_side = side                                                       <L 438>
                    var_71 = wp::copy(var_35);
                }
                var_72 = wp::where(var_68, var_69, var_25);
                var_73 = wp::where(var_68, var_71, var_27);
                var_74 = wp::where(var_68, var_70, var_29);
                // all[2 * i + (side + 1) // 2] = sol                                         <L 441>
                var_76 = wp::mul(var_75, var_30);
                var_78 = wp::add(var_35, var_77);
                var_80 = wp::floordiv(var_78, var_79);
                var_81 = wp::add(var_76, var_80);
                wp::assign_inplace(var_12, var_81, var_42);
            }
            var_82 = wp::where(var_64, var_72, var_25);
            var_83 = wp::where(var_64, var_73, var_27);
            var_84 = wp::where(var_64, var_74, var_29);
        }
        var_85 = wp::where(var_44, var_82, var_25);
        var_86 = wp::where(var_44, var_83, var_27);
        var_87 = wp::where(var_44, var_84, var_29);
        // sol = (float(side) * size[i] - lpnt[i]) / lvec[i]                                  <L 421>
        var_89 = wp::float(var_88);
        var_90 = wp::extract(var_size, var_30);
        var_91 = wp::mul(var_89, var_90);
        var_92 = wp::extract(var_21, var_30);
        var_93 = wp::sub(var_91, var_92);
        var_94 = wp::extract(var_22, var_30);
        var_95 = wp::div(var_93, var_94);
        // if sol >= 0.0:                                                                     <L 424>
        var_97 = (var_95 >= var_96);
        if (var_97) {
            // id0 = _IFACE[i][0]                                                             <L 425>
            var_99 = wp::extract(var_45, var_30, var_98);
            // id1 = _IFACE[i][1]                                                             <L 426>
            var_101 = wp::extract(var_45, var_30, var_100);
            // p0 = lpnt[id0] + sol * lvec[id0]                                               <L 429>
            var_102 = wp::extract(var_21, var_99);
            var_103 = wp::extract(var_22, var_99);
            var_104 = wp::mul(var_95, var_103);
            var_105 = wp::add(var_102, var_104);
            // p1 = lpnt[id1] + sol * lvec[id1]                                               <L 430>
            var_106 = wp::extract(var_21, var_101);
            var_107 = wp::extract(var_22, var_101);
            var_108 = wp::mul(var_95, var_107);
            var_109 = wp::add(var_106, var_108);
            // if (wp.abs(p0) <= size[id0]) and (wp.abs(p1) <= size[id1]):                    <L 433>
            var_110 = wp::abs(var_105);
            var_111 = wp::extract(var_size, var_99);
            var_112 = (var_110 <= var_111);
            var_113 = wp::abs(var_109);
            var_114 = wp::extract(var_size, var_101);
            var_115 = (var_113 <= var_114);
            var_116 = var_112 && var_115;
            if (var_116) {
                // if x < 0.0 or sol < x:                                                     <L 435>
                var_118 = (var_85 < var_117);
                var_119 = (var_95 < var_85);
                var_120 = var_118 || var_119;
                if (var_120) {
                    // x = sol                                                                <L 436>
                    var_121 = wp::copy(var_95);
                    // face_axis = i                                                          <L 437>
                    var_122 = wp::copy(var_30);
                    // face_side = side                                                       <L 438>
                    var_123 = wp::copy(var_88);
                }
                var_124 = wp::where(var_120, var_121, var_85);
                var_125 = wp::where(var_120, var_123, var_86);
                var_126 = wp::where(var_120, var_122, var_87);
                // all[2 * i + (side + 1) // 2] = sol                                         <L 441>
                var_128 = wp::mul(var_127, var_30);
                var_130 = wp::add(var_88, var_129);
                var_132 = wp::floordiv(var_130, var_131);
                var_133 = wp::add(var_128, var_132);
                wp::assign_inplace(var_12, var_133, var_95);
            }
            var_134 = wp::where(var_116, var_124, var_85);
            var_135 = wp::where(var_116, var_125, var_86);
            var_136 = wp::where(var_116, var_126, var_87);
        }
        var_137 = wp::where(var_97, var_134, var_85);
        var_138 = wp::where(var_97, var_135, var_86);
        var_139 = wp::where(var_97, var_136, var_87);
        var_140 = wp::where(var_97, var_99, var_47);
        var_141 = wp::where(var_97, var_101, var_49);
        var_142 = wp::where(var_97, var_105, var_53);
        var_143 = wp::where(var_97, var_109, var_57);
    }
    var_144 = wp::where(var_34, var_137, var_25);
    var_145 = wp::where(var_34, var_138, var_27);
    var_146 = wp::where(var_34, var_139, var_29);
    // if wp.abs(lvec[i]) > MJ_MINVAL:                                                        <L 418>
    var_148 = wp::extract(var_22, var_147);
    var_149 = wp::abs(var_148);
    var_150 = (var_149 > var_33);
    if (var_150) {
        // for side in range(-1, 2, 2):                                                       <L 419>
        // sol = (float(side) * size[i] - lpnt[i]) / lvec[i]                                  <L 421>
        var_152 = wp::float(var_151);
        var_153 = wp::extract(var_size, var_147);
        var_154 = wp::mul(var_152, var_153);
        var_155 = wp::extract(var_21, var_147);
        var_156 = wp::sub(var_154, var_155);
        var_157 = wp::extract(var_22, var_147);
        var_158 = wp::div(var_156, var_157);
        // if sol >= 0.0:                                                                     <L 424>
        var_160 = (var_158 >= var_159);
        if (var_160) {
            // id0 = _IFACE[i][0]                                                             <L 425>
            var_162 = wp::extract(var_45, var_147, var_161);
            // id1 = _IFACE[i][1]                                                             <L 426>
            var_164 = wp::extract(var_45, var_147, var_163);
            // p0 = lpnt[id0] + sol * lvec[id0]                                               <L 429>
            var_165 = wp::extract(var_21, var_162);
            var_166 = wp::extract(var_22, var_162);
            var_167 = wp::mul(var_158, var_166);
            var_168 = wp::add(var_165, var_167);
            // p1 = lpnt[id1] + sol * lvec[id1]                                               <L 430>
            var_169 = wp::extract(var_21, var_164);
            var_170 = wp::extract(var_22, var_164);
            var_171 = wp::mul(var_158, var_170);
            var_172 = wp::add(var_169, var_171);
            // if (wp.abs(p0) <= size[id0]) and (wp.abs(p1) <= size[id1]):                    <L 433>
            var_173 = wp::abs(var_168);
            var_174 = wp::extract(var_size, var_162);
            var_175 = (var_173 <= var_174);
            var_176 = wp::abs(var_172);
            var_177 = wp::extract(var_size, var_164);
            var_178 = (var_176 <= var_177);
            var_179 = var_175 && var_178;
            if (var_179) {
                // if x < 0.0 or sol < x:                                                     <L 435>
                var_181 = (var_144 < var_180);
                var_182 = (var_158 < var_144);
                var_183 = var_181 || var_182;
                if (var_183) {
                    // x = sol                                                                <L 436>
                    var_184 = wp::copy(var_158);
                    // face_axis = i                                                          <L 437>
                    var_185 = wp::copy(var_147);
                    // face_side = side                                                       <L 438>
                    var_186 = wp::copy(var_151);
                }
                var_187 = wp::where(var_183, var_184, var_144);
                var_188 = wp::where(var_183, var_186, var_145);
                var_189 = wp::where(var_183, var_185, var_146);
                // all[2 * i + (side + 1) // 2] = sol                                         <L 441>
                var_191 = wp::mul(var_190, var_147);
                var_193 = wp::add(var_151, var_192);
                var_195 = wp::floordiv(var_193, var_194);
                var_196 = wp::add(var_191, var_195);
                wp::assign_inplace(var_12, var_196, var_158);
            }
            var_197 = wp::where(var_179, var_187, var_144);
            var_198 = wp::where(var_179, var_188, var_145);
            var_199 = wp::where(var_179, var_189, var_146);
        }
        var_200 = wp::where(var_160, var_197, var_144);
        var_201 = wp::where(var_160, var_198, var_145);
        var_202 = wp::where(var_160, var_199, var_146);
        var_203 = wp::where(var_160, var_162, var_140);
        var_204 = wp::where(var_160, var_164, var_141);
        var_205 = wp::where(var_160, var_168, var_142);
        var_206 = wp::where(var_160, var_172, var_143);
        // sol = (float(side) * size[i] - lpnt[i]) / lvec[i]                                  <L 421>
        var_208 = wp::float(var_207);
        var_209 = wp::extract(var_size, var_147);
        var_210 = wp::mul(var_208, var_209);
        var_211 = wp::extract(var_21, var_147);
        var_212 = wp::sub(var_210, var_211);
        var_213 = wp::extract(var_22, var_147);
        var_214 = wp::div(var_212, var_213);
        // if sol >= 0.0:                                                                     <L 424>
        var_216 = (var_214 >= var_215);
        if (var_216) {
            // id0 = _IFACE[i][0]                                                             <L 425>
            var_218 = wp::extract(var_45, var_147, var_217);
            // id1 = _IFACE[i][1]                                                             <L 426>
            var_220 = wp::extract(var_45, var_147, var_219);
            // p0 = lpnt[id0] + sol * lvec[id0]                                               <L 429>
            var_221 = wp::extract(var_21, var_218);
            var_222 = wp::extract(var_22, var_218);
            var_223 = wp::mul(var_214, var_222);
            var_224 = wp::add(var_221, var_223);
            // p1 = lpnt[id1] + sol * lvec[id1]                                               <L 430>
            var_225 = wp::extract(var_21, var_220);
            var_226 = wp::extract(var_22, var_220);
            var_227 = wp::mul(var_214, var_226);
            var_228 = wp::add(var_225, var_227);
            // if (wp.abs(p0) <= size[id0]) and (wp.abs(p1) <= size[id1]):                    <L 433>
            var_229 = wp::abs(var_224);
            var_230 = wp::extract(var_size, var_218);
            var_231 = (var_229 <= var_230);
            var_232 = wp::abs(var_228);
            var_233 = wp::extract(var_size, var_220);
            var_234 = (var_232 <= var_233);
            var_235 = var_231 && var_234;
            if (var_235) {
                // if x < 0.0 or sol < x:                                                     <L 435>
                var_237 = (var_200 < var_236);
                var_238 = (var_214 < var_200);
                var_239 = var_237 || var_238;
                if (var_239) {
                    // x = sol                                                                <L 436>
                    var_240 = wp::copy(var_214);
                    // face_axis = i                                                          <L 437>
                    var_241 = wp::copy(var_147);
                    // face_side = side                                                       <L 438>
                    var_242 = wp::copy(var_207);
                }
                var_243 = wp::where(var_239, var_240, var_200);
                var_244 = wp::where(var_239, var_242, var_201);
                var_245 = wp::where(var_239, var_241, var_202);
                // all[2 * i + (side + 1) // 2] = sol                                         <L 441>
                var_247 = wp::mul(var_246, var_147);
                var_249 = wp::add(var_207, var_248);
                var_251 = wp::floordiv(var_249, var_250);
                var_252 = wp::add(var_247, var_251);
                wp::assign_inplace(var_12, var_252, var_214);
            }
            var_253 = wp::where(var_235, var_243, var_200);
            var_254 = wp::where(var_235, var_244, var_201);
            var_255 = wp::where(var_235, var_245, var_202);
        }
        var_256 = wp::where(var_216, var_253, var_200);
        var_257 = wp::where(var_216, var_254, var_201);
        var_258 = wp::where(var_216, var_255, var_202);
        var_259 = wp::where(var_216, var_218, var_203);
        var_260 = wp::where(var_216, var_220, var_204);
        var_261 = wp::where(var_216, var_224, var_205);
        var_262 = wp::where(var_216, var_228, var_206);
    }
    var_263 = wp::where(var_150, var_256, var_144);
    var_264 = wp::where(var_150, var_257, var_145);
    var_265 = wp::where(var_150, var_258, var_146);
    var_266 = wp::where(var_150, var_207, var_88);
    var_267 = wp::where(var_150, var_214, var_95);
    var_268 = wp::where(var_150, var_259, var_140);
    var_269 = wp::where(var_150, var_260, var_141);
    var_270 = wp::where(var_150, var_261, var_142);
    var_271 = wp::where(var_150, var_262, var_143);
    // if wp.abs(lvec[i]) > MJ_MINVAL:                                                        <L 418>
    var_273 = wp::extract(var_22, var_272);
    var_274 = wp::abs(var_273);
    var_275 = (var_274 > var_33);
    if (var_275) {
        // for side in range(-1, 2, 2):                                                       <L 419>
        // sol = (float(side) * size[i] - lpnt[i]) / lvec[i]                                  <L 421>
        var_277 = wp::float(var_276);
        var_278 = wp::extract(var_size, var_272);
        var_279 = wp::mul(var_277, var_278);
        var_280 = wp::extract(var_21, var_272);
        var_281 = wp::sub(var_279, var_280);
        var_282 = wp::extract(var_22, var_272);
        var_283 = wp::div(var_281, var_282);
        // if sol >= 0.0:                                                                     <L 424>
        var_285 = (var_283 >= var_284);
        if (var_285) {
            // id0 = _IFACE[i][0]                                                             <L 425>
            var_287 = wp::extract(var_45, var_272, var_286);
            // id1 = _IFACE[i][1]                                                             <L 426>
            var_289 = wp::extract(var_45, var_272, var_288);
            // p0 = lpnt[id0] + sol * lvec[id0]                                               <L 429>
            var_290 = wp::extract(var_21, var_287);
            var_291 = wp::extract(var_22, var_287);
            var_292 = wp::mul(var_283, var_291);
            var_293 = wp::add(var_290, var_292);
            // p1 = lpnt[id1] + sol * lvec[id1]                                               <L 430>
            var_294 = wp::extract(var_21, var_289);
            var_295 = wp::extract(var_22, var_289);
            var_296 = wp::mul(var_283, var_295);
            var_297 = wp::add(var_294, var_296);
            // if (wp.abs(p0) <= size[id0]) and (wp.abs(p1) <= size[id1]):                    <L 433>
            var_298 = wp::abs(var_293);
            var_299 = wp::extract(var_size, var_287);
            var_300 = (var_298 <= var_299);
            var_301 = wp::abs(var_297);
            var_302 = wp::extract(var_size, var_289);
            var_303 = (var_301 <= var_302);
            var_304 = var_300 && var_303;
            if (var_304) {
                // if x < 0.0 or sol < x:                                                     <L 435>
                var_306 = (var_263 < var_305);
                var_307 = (var_283 < var_263);
                var_308 = var_306 || var_307;
                if (var_308) {
                    // x = sol                                                                <L 436>
                    var_309 = wp::copy(var_283);
                    // face_axis = i                                                          <L 437>
                    var_310 = wp::copy(var_272);
                    // face_side = side                                                       <L 438>
                    var_311 = wp::copy(var_276);
                }
                var_312 = wp::where(var_308, var_309, var_263);
                var_313 = wp::where(var_308, var_311, var_264);
                var_314 = wp::where(var_308, var_310, var_265);
                // all[2 * i + (side + 1) // 2] = sol                                         <L 441>
                var_316 = wp::mul(var_315, var_272);
                var_318 = wp::add(var_276, var_317);
                var_320 = wp::floordiv(var_318, var_319);
                var_321 = wp::add(var_316, var_320);
                wp::assign_inplace(var_12, var_321, var_283);
            }
            var_322 = wp::where(var_304, var_312, var_263);
            var_323 = wp::where(var_304, var_313, var_264);
            var_324 = wp::where(var_304, var_314, var_265);
        }
        var_325 = wp::where(var_285, var_322, var_263);
        var_326 = wp::where(var_285, var_323, var_264);
        var_327 = wp::where(var_285, var_324, var_265);
        var_328 = wp::where(var_285, var_287, var_268);
        var_329 = wp::where(var_285, var_289, var_269);
        var_330 = wp::where(var_285, var_293, var_270);
        var_331 = wp::where(var_285, var_297, var_271);
        // sol = (float(side) * size[i] - lpnt[i]) / lvec[i]                                  <L 421>
        var_333 = wp::float(var_332);
        var_334 = wp::extract(var_size, var_272);
        var_335 = wp::mul(var_333, var_334);
        var_336 = wp::extract(var_21, var_272);
        var_337 = wp::sub(var_335, var_336);
        var_338 = wp::extract(var_22, var_272);
        var_339 = wp::div(var_337, var_338);
        // if sol >= 0.0:                                                                     <L 424>
        var_341 = (var_339 >= var_340);
        if (var_341) {
            // id0 = _IFACE[i][0]                                                             <L 425>
            var_343 = wp::extract(var_45, var_272, var_342);
            // id1 = _IFACE[i][1]                                                             <L 426>
            var_345 = wp::extract(var_45, var_272, var_344);
            // p0 = lpnt[id0] + sol * lvec[id0]                                               <L 429>
            var_346 = wp::extract(var_21, var_343);
            var_347 = wp::extract(var_22, var_343);
            var_348 = wp::mul(var_339, var_347);
            var_349 = wp::add(var_346, var_348);
            // p1 = lpnt[id1] + sol * lvec[id1]                                               <L 430>
            var_350 = wp::extract(var_21, var_345);
            var_351 = wp::extract(var_22, var_345);
            var_352 = wp::mul(var_339, var_351);
            var_353 = wp::add(var_350, var_352);
            // if (wp.abs(p0) <= size[id0]) and (wp.abs(p1) <= size[id1]):                    <L 433>
            var_354 = wp::abs(var_349);
            var_355 = wp::extract(var_size, var_343);
            var_356 = (var_354 <= var_355);
            var_357 = wp::abs(var_353);
            var_358 = wp::extract(var_size, var_345);
            var_359 = (var_357 <= var_358);
            var_360 = var_356 && var_359;
            if (var_360) {
                // if x < 0.0 or sol < x:                                                     <L 435>
                var_362 = (var_325 < var_361);
                var_363 = (var_339 < var_325);
                var_364 = var_362 || var_363;
                if (var_364) {
                    // x = sol                                                                <L 436>
                    var_365 = wp::copy(var_339);
                    // face_axis = i                                                          <L 437>
                    var_366 = wp::copy(var_272);
                    // face_side = side                                                       <L 438>
                    var_367 = wp::copy(var_332);
                }
                var_368 = wp::where(var_364, var_365, var_325);
                var_369 = wp::where(var_364, var_367, var_326);
                var_370 = wp::where(var_364, var_366, var_327);
                // all[2 * i + (side + 1) // 2] = sol                                         <L 441>
                var_372 = wp::mul(var_371, var_272);
                var_374 = wp::add(var_332, var_373);
                var_376 = wp::floordiv(var_374, var_375);
                var_377 = wp::add(var_372, var_376);
                wp::assign_inplace(var_12, var_377, var_339);
            }
            var_378 = wp::where(var_360, var_368, var_325);
            var_379 = wp::where(var_360, var_369, var_326);
            var_380 = wp::where(var_360, var_370, var_327);
        }
        var_381 = wp::where(var_341, var_378, var_325);
        var_382 = wp::where(var_341, var_379, var_326);
        var_383 = wp::where(var_341, var_380, var_327);
        var_384 = wp::where(var_341, var_343, var_328);
        var_385 = wp::where(var_341, var_345, var_329);
        var_386 = wp::where(var_341, var_349, var_330);
        var_387 = wp::where(var_341, var_353, var_331);
    }
    var_388 = wp::where(var_275, var_381, var_263);
    var_389 = wp::where(var_275, var_382, var_264);
    var_390 = wp::where(var_275, var_383, var_265);
    var_391 = wp::where(var_275, var_332, var_266);
    var_392 = wp::where(var_275, var_339, var_267);
    var_393 = wp::where(var_275, var_384, var_268);
    var_394 = wp::where(var_275, var_385, var_269);
    var_395 = wp::where(var_275, var_386, var_270);
    var_396 = wp::where(var_275, var_387, var_271);
    // normal = wp.vec3()                                                                     <L 443>
    var_397 = wp::vec_t<3, wp::float32>();
    // if x >= 0:                                                                             <L 444>
    var_399 = (var_388 >= var_398);
    if (var_399) {
        // normal[face_axis] = float(face_side)                                               <L 445>
        var_400 = wp::float(var_389);
        wp::assign_inplace(var_397, var_390, var_400);
        // normal = mat @ normal                                                              <L 446>
        var_401 = wp::mul(var_mat, var_397);
    }
    var_402 = wp::where(var_399, var_401, var_397);
    // return x, all, normal                                                                  <L 448>
    ret_0 = var_388;
    ret_1 = var_12;
    ret_2 = var_402;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/ray.py:128
static CUDA_CALLABLE void _ray_triangle_1(
    wp::vec_t<3, wp::float32> var_v0,
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::vec_t<3, wp::float32> var_b0,
    wp::vec_t<3, wp::float32> var_b1,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    const wp::float32 var_9 = 0.0;
    bool var_10;
    const wp::float32 var_11 = 0.0;
    bool var_12;
    const wp::float32 var_13 = 0.0;
    bool var_14;
    bool var_15;
    const wp::float32 var_16 = 0.0;
    bool var_17;
    const wp::float32 var_18 = 0.0;
    bool var_19;
    const wp::float32 var_20 = 0.0;
    bool var_21;
    bool var_22;
    const wp::float32 var_23 = 0.0;
    bool var_24;
    const wp::float32 var_25 = 0.0;
    bool var_26;
    const wp::float32 var_27 = 0.0;
    bool var_28;
    bool var_29;
    const wp::float32 var_30 = 0.0;
    bool var_31;
    const wp::float32 var_32 = 0.0;
    bool var_33;
    const wp::float32 var_34 = 0.0;
    bool var_35;
    bool var_36;
    bool var_37;
    const wp::float32 var_38 = 1.0;
    const wp::float32 var_39 = -1.0;
    wp::vec_t<3, wp::float32> var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    wp::vec_t<2, wp::float32> var_47;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    const wp::float32 var_52 = 1e-15;
    bool var_53;
    const wp::float32 var_54 = 1.0;
    const wp::float32 var_55 = -1.0;
    wp::vec_t<3, wp::float32> var_56;
    const wp::int32 var_57 = 0;
    wp::float32 var_58;
    wp::float32 var_59;
    const wp::int32 var_60 = 1;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    const wp::int32 var_66 = 0;
    wp::float32 var_67;
    wp::float32 var_68;
    const wp::int32 var_69 = 1;
    wp::float32 var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    wp::float32 var_73;
    const wp::float32 var_74 = 0.0;
    bool var_75;
    const wp::float32 var_76 = 0.0;
    bool var_77;
    wp::float32 var_78;
    const wp::float32 var_79 = 1.0;
    bool var_80;
    bool var_81;
    const wp::float32 var_82 = 1.0;
    const wp::float32 var_83 = -1.0;
    wp::vec_t<3, wp::float32> var_84;
    wp::vec_t<3, wp::float32> var_85;
    wp::vec_t<3, wp::float32> var_86;
    wp::vec_t<3, wp::float32> var_87;
    wp::vec_t<3, wp::float32> var_88;
    wp::float32 var_89;
    wp::float32 var_90;
    bool var_91;
    const wp::float32 var_92 = 1.0;
    const wp::float32 var_93 = -1.0;
    wp::vec_t<3, wp::float32> var_94;
    wp::float32 var_95;
    wp::float32 var_96;
    wp::float32 var_97;
    const wp::float32 var_98 = 0.0;
    bool var_99;
    const wp::float32 var_100 = 1.0;
    const wp::float32 var_101 = -1.0;
    wp::float32 var_102;
    wp::vec_t<3, wp::float32> var_103;
    //---------
    // forward
    // def _ray_triangle(                                                                     <L 129>
    // dif0 = v0 - pnt                                                                        <L 133>
    var_0 = wp::sub(var_v0, var_pnt);
    // dif1 = v1 - pnt                                                                        <L 134>
    var_1 = wp::sub(var_v1, var_pnt);
    // dif2 = v2 - pnt                                                                        <L 135>
    var_2 = wp::sub(var_v2, var_pnt);
    // planar_00 = wp.dot(dif0, b0)                                                           <L 138>
    var_3 = wp::dot(var_0, var_b0);
    // planar_01 = wp.dot(dif0, b1)                                                           <L 139>
    var_4 = wp::dot(var_0, var_b1);
    // planar_10 = wp.dot(dif1, b0)                                                           <L 140>
    var_5 = wp::dot(var_1, var_b0);
    // planar_11 = wp.dot(dif1, b1)                                                           <L 141>
    var_6 = wp::dot(var_1, var_b1);
    // planar_20 = wp.dot(dif2, b0)                                                           <L 142>
    var_7 = wp::dot(var_2, var_b0);
    // planar_21 = wp.dot(dif2, b1)                                                           <L 143>
    var_8 = wp::dot(var_2, var_b1);
    // if (                                                                                   <L 146>
    // (planar_00 > 0.0 and planar_10 > 0.0 and planar_20 > 0.0)                              <L 147>
    var_10 = (var_3 > var_9);
    var_12 = (var_5 > var_11);
    var_14 = (var_7 > var_13);
    var_15 = var_10 && var_12 && var_14;
    // or (planar_00 < 0.0 and planar_10 < 0.0 and planar_20 < 0.0)                           <L 148>
    var_17 = (var_3 < var_16);
    var_19 = (var_5 < var_18);
    var_21 = (var_7 < var_20);
    var_22 = var_17 && var_19 && var_21;
    // or (planar_01 > 0.0 and planar_11 > 0.0 and planar_21 > 0.0)                           <L 149>
    var_24 = (var_4 > var_23);
    var_26 = (var_6 > var_25);
    var_28 = (var_8 > var_27);
    var_29 = var_24 && var_26 && var_28;
    // or (planar_01 < 0.0 and planar_11 < 0.0 and planar_21 < 0.0)                           <L 150>
    var_31 = (var_4 < var_30);
    var_33 = (var_6 < var_32);
    var_35 = (var_8 < var_34);
    var_36 = var_31 && var_33 && var_35;
    var_37 = var_15 || var_22 || var_29 || var_36;
    if (var_37) {
        // return -1.0, wp.vec3()                                                             <L 152>
        var_40 = wp::vec_t<3, wp::float32>();
        ret_0 = var_39;
        ret_1 = var_40;
        return;
    }
    // A00 = planar_00 - planar_20                                                            <L 156>
    var_41 = wp::sub(var_3, var_7);
    // A10 = planar_10 - planar_20                                                            <L 157>
    var_42 = wp::sub(var_5, var_7);
    // A01 = planar_01 - planar_21                                                            <L 158>
    var_43 = wp::sub(var_4, var_8);
    // A11 = planar_11 - planar_21                                                            <L 159>
    var_44 = wp::sub(var_6, var_8);
    // b = wp.vec2(-planar_20, -planar_21)                                                    <L 161>
    var_45 = wp::neg(var_7);
    var_46 = wp::neg(var_8);
    var_47 = wp::vec_t<2, wp::float32>(var_45, var_46);
    // det = A00 * A11 - A10 * A01                                                            <L 163>
    var_48 = wp::mul(var_41, var_44);
    var_49 = wp::mul(var_42, var_43);
    var_50 = wp::sub(var_48, var_49);
    // if wp.abs(det) < MJ_MINVAL:                                                            <L 164>
    var_51 = wp::abs(var_50);
    var_53 = (var_51 < var_52);
    if (var_53) {
        // return -1.0, wp.vec3()                                                             <L 165>
        var_56 = wp::vec_t<3, wp::float32>();
        ret_0 = var_55;
        ret_1 = var_56;
        return;
    }
    // t0 = (A11 * b[0] - A10 * b[1]) / det                                                   <L 167>
    var_58 = wp::extract(var_47, var_57);
    var_59 = wp::mul(var_44, var_58);
    var_61 = wp::extract(var_47, var_60);
    var_62 = wp::mul(var_42, var_61);
    var_63 = wp::sub(var_59, var_62);
    var_64 = wp::div(var_63, var_50);
    // t1 = (-A01 * b[0] + A00 * b[1]) / det                                                  <L 168>
    var_65 = wp::neg(var_43);
    var_67 = wp::extract(var_47, var_66);
    var_68 = wp::mul(var_65, var_67);
    var_70 = wp::extract(var_47, var_69);
    var_71 = wp::mul(var_41, var_70);
    var_72 = wp::add(var_68, var_71);
    var_73 = wp::div(var_72, var_50);
    // if t0 < 0.0 or t1 < 0.0 or t0 + t1 > 1.0:                                              <L 171>
    var_75 = (var_64 < var_74);
    var_77 = (var_73 < var_76);
    var_78 = wp::add(var_64, var_73);
    var_80 = (var_78 > var_79);
    var_81 = var_75 || var_77 || var_80;
    if (var_81) {
        // return -1.0, wp.vec3()                                                             <L 172>
        var_84 = wp::vec_t<3, wp::float32>();
        ret_0 = var_83;
        ret_1 = var_84;
        return;
    }
    // dif0 = v0 - v2                                                                         <L 175>
    var_85 = wp::sub(var_v0, var_v2);
    // dif1 = v1 - v2                                                                         <L 176>
    var_86 = wp::sub(var_v1, var_v2);
    // dif2 = pnt - v2                                                                        <L 177>
    var_87 = wp::sub(var_pnt, var_v2);
    // nrm = wp.cross(dif0, dif1)  # normal to triangle plane                                 <L 178>
    var_88 = wp::cross(var_85, var_86);
    // denom = wp.dot(vec, nrm)                                                               <L 179>
    var_89 = wp::dot(var_vec, var_88);
    // if wp.abs(denom) < MJ_MINVAL:                                                          <L 180>
    var_90 = wp::abs(var_89);
    var_91 = (var_90 < var_52);
    if (var_91) {
        // return -1.0, wp.vec3()                                                             <L 181>
        var_94 = wp::vec_t<3, wp::float32>();
        ret_0 = var_93;
        ret_1 = var_94;
        return;
    }
    // dist = -wp.dot(dif2, nrm) / denom                                                      <L 183>
    var_95 = wp::dot(var_87, var_88);
    var_96 = wp::neg(var_95);
    var_97 = wp::div(var_96, var_89);
    // return wp.where(dist >= 0.0, dist, -1.0), wp.normalize(nrm)                            <L 184>
    var_99 = (var_97 >= var_98);
    var_102 = wp::where(var_99, var_97, var_101);
    var_103 = wp::normalize(var_88);
    ret_0 = var_102;
    ret_1 = var_103;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/ray.py:622
static CUDA_CALLABLE void ray_mesh_0(
    wp::int32 var_nmeshface,
    wp::array_t<wp::int32> var_mesh_vertadr,
    wp::array_t<wp::int32> var_mesh_faceadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_vert,
    wp::array_t<wp::vec_t<3, wp::int32>> var_mesh_face,
    wp::int32 var_data_id,
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::vec_t<6, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    const wp::float32 var_3 = 0.0;
    bool var_4;
    const wp::float32 var_5 = 1.0;
    const wp::float32 var_6 = -1.0;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    const wp::int32 var_10 = 0;
    wp::float32 var_11;
    wp::float32 var_12;
    const wp::int32 var_13 = 1;
    wp::float32 var_14;
    wp::float32 var_15;
    bool var_16;
    const wp::int32 var_17 = 0;
    wp::float32 var_18;
    wp::float32 var_19;
    const wp::int32 var_20 = 2;
    wp::float32 var_21;
    wp::float32 var_22;
    bool var_23;
    const wp::float32 var_24 = 0.0;
    const wp::int32 var_25 = 2;
    wp::float32 var_26;
    const wp::int32 var_27 = 1;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::vec_t<3, wp::float32> var_30;
    const wp::int32 var_31 = 1;
    wp::float32 var_32;
    const wp::int32 var_33 = 0;
    wp::float32 var_34;
    wp::float32 var_35;
    const wp::float32 var_36 = 0.0;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    const wp::int32 var_39 = 1;
    wp::float32 var_40;
    wp::float32 var_41;
    const wp::int32 var_42 = 2;
    wp::float32 var_43;
    wp::float32 var_44;
    bool var_45;
    const wp::int32 var_46 = 2;
    wp::float32 var_47;
    wp::float32 var_48;
    const wp::float32 var_49 = 0.0;
    const wp::int32 var_50 = 0;
    wp::float32 var_51;
    wp::vec_t<3, wp::float32> var_52;
    wp::vec_t<3, wp::float32> var_53;
    const wp::int32 var_54 = 1;
    wp::float32 var_55;
    const wp::int32 var_56 = 0;
    wp::float32 var_57;
    wp::float32 var_58;
    const wp::float32 var_59 = 0.0;
    wp::vec_t<3, wp::float32> var_60;
    wp::vec_t<3, wp::float32> var_61;
    wp::vec_t<3, wp::float32> var_62;
    wp::vec_t<3, wp::float32> var_63;
    wp::vec_t<3, wp::float32> var_64;
    wp::vec_t<3, wp::float32> var_65;
    const wp::float32 var_66 = 1.0;
    const wp::float32 var_67 = -1.0;
    wp::float32 var_68;
    wp::vec_t<3, wp::float32> var_69;
    wp::int32* var_70;
    wp::int32 var_71;
    wp::int32 var_72;
    wp::int32* var_73;
    wp::int32 var_74;
    wp::int32 var_75;
    const wp::int32 var_76 = 1;
    wp::int32 var_77;
    wp::shape_t* var_78;
    const wp::int32 var_79 = 0;
    wp::int32 var_80;
    wp::shape_t var_81;
    bool var_82;
    const wp::int32 var_83 = 1;
    wp::int32 var_84;
    wp::int32* var_85;
    wp::int32 var_86;
    wp::int32 var_87;
    wp::int32 var_88;
    wp::int32 var_89;
    wp::range_t var_90;
    wp::int32 var_91;
    wp::vec_t<3, wp::int32>* var_92;
    wp::vec_t<3, wp::int32> var_93;
    wp::vec_t<3, wp::int32> var_94;
    const wp::int32 var_95 = 0;
    wp::int32 var_96;
    wp::int32 var_97;
    wp::vec_t<3, wp::float32>* var_98;
    wp::vec_t<3, wp::float32> var_99;
    wp::vec_t<3, wp::float32> var_100;
    const wp::int32 var_101 = 1;
    wp::int32 var_102;
    wp::int32 var_103;
    wp::vec_t<3, wp::float32>* var_104;
    wp::vec_t<3, wp::float32> var_105;
    wp::vec_t<3, wp::float32> var_106;
    const wp::int32 var_107 = 2;
    wp::int32 var_108;
    wp::int32 var_109;
    wp::vec_t<3, wp::float32>* var_110;
    wp::vec_t<3, wp::float32> var_111;
    wp::vec_t<3, wp::float32> var_112;
    wp::float32 var_113;
    wp::vec_t<3, wp::float32> var_114;
    const wp::int32 var_115 = 0;
    bool var_116;
    const wp::int32 var_117 = 0;
    bool var_118;
    bool var_119;
    bool var_120;
    bool var_121;
    wp::float32 var_122;
    wp::vec_t<3, wp::float32> var_123;
    wp::float32 var_124;
    wp::vec_t<3, wp::float32> var_125;
    wp::vec_t<3, wp::float32> var_126;
    //---------
    // forward
    // def ray_mesh(                                                                          <L 623>
    // dist_box, _all, _normal = ray_box(pos, mat, size, pnt, vec)                            <L 640>
    ray_box_0(var_pos, var_mat, var_size, var_pnt, var_vec, var_0, var_1, var_2);
    // if dist_box < 0.0:                                                                     <L 641>
    var_4 = (var_0 < var_3);
    if (var_4) {
        // return -1.0, wp.vec3()                                                             <L 642>
        var_7 = wp::vec_t<3, wp::float32>();
        ret_0 = var_6;
        ret_1 = var_7;
        return;
    }
    // pnt, vec = _ray_map(pos, mat, pnt, vec)                                                <L 644>
    _ray_map_0(var_pos, var_mat, var_pnt, var_vec, var_8, var_9);
    // if wp.abs(vec[0]) < wp.abs(vec[1]):                                                    <L 647>
    var_11 = wp::extract(var_9, var_10);
    var_12 = wp::abs(var_11);
    var_14 = wp::extract(var_9, var_13);
    var_15 = wp::abs(var_14);
    var_16 = (var_12 < var_15);
    if (var_16) {
        // if wp.abs(vec[0]) < wp.abs(vec[2]):                                                <L 648>
        var_18 = wp::extract(var_9, var_17);
        var_19 = wp::abs(var_18);
        var_21 = wp::extract(var_9, var_20);
        var_22 = wp::abs(var_21);
        var_23 = (var_19 < var_22);
        if (var_23) {
            // b0 = wp.vec3(0.0, vec[2], -vec[1])                                             <L 649>
            var_26 = wp::extract(var_9, var_25);
            var_28 = wp::extract(var_9, var_27);
            var_29 = wp::neg(var_28);
            var_30 = wp::vec_t<3, wp::float32>(var_24, var_26, var_29);
        }
        if (!var_23) {
            // b0 = wp.vec3(vec[1], -vec[0], 0.0)                                             <L 651>
            var_32 = wp::extract(var_9, var_31);
            var_34 = wp::extract(var_9, var_33);
            var_35 = wp::neg(var_34);
            var_37 = wp::vec_t<3, wp::float32>(var_32, var_35, var_36);
        }
        var_38 = wp::where(var_23, var_30, var_37);
    }
    if (!var_16) {
        // if wp.abs(vec[1]) < wp.abs(vec[2]):                                                <L 653>
        var_40 = wp::extract(var_9, var_39);
        var_41 = wp::abs(var_40);
        var_43 = wp::extract(var_9, var_42);
        var_44 = wp::abs(var_43);
        var_45 = (var_41 < var_44);
        if (var_45) {
            // b0 = wp.vec3(-vec[2], 0.0, vec[0])                                             <L 654>
            var_47 = wp::extract(var_9, var_46);
            var_48 = wp::neg(var_47);
            var_51 = wp::extract(var_9, var_50);
            var_52 = wp::vec_t<3, wp::float32>(var_48, var_49, var_51);
        }
        var_53 = wp::where(var_45, var_52, var_38);
        if (!var_45) {
            // b0 = wp.vec3(vec[1], -vec[0], 0.0)                                             <L 656>
            var_55 = wp::extract(var_9, var_54);
            var_57 = wp::extract(var_9, var_56);
            var_58 = wp::neg(var_57);
            var_60 = wp::vec_t<3, wp::float32>(var_55, var_58, var_59);
        }
        var_61 = wp::where(var_45, var_53, var_60);
    }
    var_62 = wp::where(var_16, var_38, var_61);
    // b0 = wp.normalize(b0)                                                                  <L 659>
    var_63 = wp::normalize(var_62);
    // b1 = wp.cross(vec, b0)                                                                 <L 662>
    var_64 = wp::cross(var_9, var_63);
    // b1 = wp.normalize(b1)                                                                  <L 663>
    var_65 = wp::normalize(var_64);
    // x = float(-1.0)                                                                        <L 665>
    var_68 = wp::float(var_67);
    // normal = wp.vec3()                                                                     <L 666>
    var_69 = wp::vec_t<3, wp::float32>();
    // vert_start = mesh_vertadr[data_id]                                                     <L 669>
    var_70 = wp::address(var_mesh_vertadr, var_data_id);
    var_72 = wp::load(var_70);
    var_71 = wp::copy(var_72);
    // face_start = mesh_faceadr[data_id]                                                     <L 672>
    var_73 = wp::address(var_mesh_faceadr, var_data_id);
    var_75 = wp::load(var_73);
    var_74 = wp::copy(var_75);
    // if data_id + 1 < mesh_faceadr.shape[0]:                                                <L 674>
    var_77 = wp::add(var_data_id, var_76);
    var_78 = &(var_mesh_faceadr.shape);
    var_81 = wp::load(var_78);
    var_80 = wp::extract(var_81, var_79);
    var_82 = (var_77 < var_80);
    if (var_82) {
        // face_end = mesh_faceadr[data_id + 1]                                               <L 675>
        var_84 = wp::add(var_data_id, var_83);
        var_85 = wp::address(var_mesh_faceadr, var_84);
        var_87 = wp::load(var_85);
        var_86 = wp::copy(var_87);
    }
    if (!var_82) {
        // face_end = nmeshface                                                               <L 677>
        var_88 = wp::copy(var_nmeshface);
    }
    var_89 = wp::where(var_82, var_86, var_88);
    // for i in range(face_start, face_end):                                                  <L 680>
    var_90 = wp::range(var_74, var_89);
    start_for_1:;
        if (iter_cmp(var_90) == 0) goto end_for_1;
        var_91 = wp::iter_next(var_90);
        // v_idx = mesh_face[i]                                                               <L 682>
        var_92 = wp::address(var_mesh_face, var_91);
        var_94 = wp::load(var_92);
        var_93 = wp::copy(var_94);
        // v0 = mesh_vert[vert_start + v_idx.x]                                               <L 685>
        var_96 = wp::extract(var_93, var_95);
        var_97 = wp::add(var_71, var_96);
        var_98 = wp::address(var_mesh_vert, var_97);
        var_100 = wp::load(var_98);
        var_99 = wp::copy(var_100);
        // v1 = mesh_vert[vert_start + v_idx.y]                                               <L 686>
        var_102 = wp::extract(var_93, var_101);
        var_103 = wp::add(var_71, var_102);
        var_104 = wp::address(var_mesh_vert, var_103);
        var_106 = wp::load(var_104);
        var_105 = wp::copy(var_106);
        // v2 = mesh_vert[vert_start + v_idx.z]                                               <L 687>
        var_108 = wp::extract(var_93, var_107);
        var_109 = wp::add(var_71, var_108);
        var_110 = wp::address(var_mesh_vert, var_109);
        var_112 = wp::load(var_110);
        var_111 = wp::copy(var_112);
        // dist, normal_tri = _ray_triangle(v0, v1, v2, pnt, vec, b0, b1)                     <L 690>
        _ray_triangle_1(var_99, var_105, var_111, var_8, var_9, var_63, var_65, var_113, var_114);
        // if dist >= 0 and (x < 0 or dist < x):                                              <L 691>
        var_116 = (var_113 >= var_115);
        var_118 = (var_68 < var_117);
        var_119 = (var_113 < var_68);
        var_120 = var_118 || var_119;
        var_121 = var_116 && var_120;
        if (var_121) {
            // x = dist                                                                       <L 692>
            var_122 = wp::copy(var_113);
            // normal = normal_tri                                                            <L 693>
            var_123 = wp::copy(var_114);
        }
        var_124 = wp::where(var_121, var_122, var_68);
        var_125 = wp::where(var_121, var_123, var_69);
        wp::assign(var_68, var_124);
        wp::assign(var_69, var_125);
        goto start_for_1;
    end_for_1:;
    // normal = mat @ normal                                                                  <L 695>
    var_126 = wp::mul(var_mat, var_69);
    // return x, normal                                                                       <L 697>
    ret_0 = var_68;
    ret_1 = var_126;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_sdf.py:325
static CUDA_CALLABLE void box_project_0(
    wp::vec_t<3, wp::float32> var_center,
    wp::vec_t<3, wp::float32> var_half_size,
    wp::vec_t<3, wp::float32> var_xyz,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    const wp::int32 var_1 = 0;
    wp::float32 var_2;
    wp::float32 var_3;
    const wp::int32 var_4 = 0;
    wp::float32 var_5;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    wp::float32 var_9;
    const wp::int32 var_10 = 1;
    wp::float32 var_11;
    wp::float32 var_12;
    const wp::int32 var_13 = 2;
    wp::float32 var_14;
    wp::float32 var_15;
    const wp::int32 var_16 = 2;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::vec_t<3, wp::float32> var_19;
    const wp::int32 var_20 = 0;
    wp::float32 var_21;
    const wp::float32 var_22 = 0.0;
    bool var_23;
    const wp::int32 var_24 = 1;
    wp::float32 var_25;
    const wp::float32 var_26 = 0.0;
    bool var_27;
    const wp::int32 var_28 = 2;
    wp::float32 var_29;
    const wp::float32 var_30 = 0.0;
    bool var_31;
    bool var_32;
    const wp::float32 var_33 = 0.0;
    const wp::float32 var_34 = 0.0;
    const wp::float32 var_35 = 0.0001;
    const wp::int32 var_36 = 0;
    wp::float32 var_37;
    const wp::int32 var_38 = 1;
    wp::float32 var_39;
    const wp::int32 var_40 = 2;
    wp::float32 var_41;
    wp::vec_t<3, wp::float32> var_42;
    const wp::int32 var_43 = 0;
    wp::float32 var_44;
    const wp::float32 var_45 = 0.0;
    bool var_46;
    const wp::int32 var_47 = 0;
    wp::float32 var_48;
    const wp::int32 var_49 = 0;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    const wp::int32 var_53 = 0;
    wp::float32 var_54;
    const wp::float32 var_55 = 0.0;
    bool var_56;
    const wp::int32 var_57 = 0;
    wp::float32 var_58;
    const wp::int32 var_59 = 0;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    const wp::int32 var_63 = 1;
    wp::float32 var_64;
    const wp::int32 var_65 = 2;
    wp::float32 var_66;
    wp::vec_t<3, wp::float32> var_67;
    wp::vec_t<3, wp::float32> var_68;
    const wp::int32 var_69 = 0;
    wp::float32 var_70;
    const wp::int32 var_71 = 0;
    wp::float32 var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    const wp::int32 var_75 = 1;
    wp::float32 var_76;
    const wp::int32 var_77 = 2;
    wp::float32 var_78;
    wp::vec_t<3, wp::float32> var_79;
    wp::vec_t<3, wp::float32> var_80;
    wp::float32 var_81;
    wp::vec_t<3, wp::float32> var_82;
    const wp::int32 var_83 = 1;
    wp::float32 var_84;
    const wp::float32 var_85 = 0.0;
    bool var_86;
    const wp::int32 var_87 = 1;
    wp::float32 var_88;
    const wp::int32 var_89 = 1;
    wp::float32 var_90;
    wp::float32 var_91;
    wp::float32 var_92;
    const wp::int32 var_93 = 1;
    wp::float32 var_94;
    const wp::float32 var_95 = 0.0;
    bool var_96;
    const wp::int32 var_97 = 0;
    wp::float32 var_98;
    const wp::int32 var_99 = 1;
    wp::float32 var_100;
    const wp::int32 var_101 = 1;
    wp::float32 var_102;
    wp::float32 var_103;
    wp::float32 var_104;
    const wp::int32 var_105 = 2;
    wp::float32 var_106;
    wp::vec_t<3, wp::float32> var_107;
    wp::vec_t<3, wp::float32> var_108;
    const wp::int32 var_109 = 0;
    wp::float32 var_110;
    const wp::int32 var_111 = 1;
    wp::float32 var_112;
    const wp::int32 var_113 = 1;
    wp::float32 var_114;
    wp::float32 var_115;
    wp::float32 var_116;
    const wp::int32 var_117 = 2;
    wp::float32 var_118;
    wp::vec_t<3, wp::float32> var_119;
    wp::vec_t<3, wp::float32> var_120;
    wp::float32 var_121;
    wp::vec_t<3, wp::float32> var_122;
    const wp::int32 var_123 = 2;
    wp::float32 var_124;
    const wp::float32 var_125 = 0.0;
    bool var_126;
    const wp::int32 var_127 = 2;
    wp::float32 var_128;
    const wp::int32 var_129 = 2;
    wp::float32 var_130;
    wp::float32 var_131;
    wp::float32 var_132;
    const wp::int32 var_133 = 2;
    wp::float32 var_134;
    const wp::float32 var_135 = 0.0;
    bool var_136;
    const wp::int32 var_137 = 0;
    wp::float32 var_138;
    const wp::int32 var_139 = 1;
    wp::float32 var_140;
    const wp::int32 var_141 = 2;
    wp::float32 var_142;
    const wp::int32 var_143 = 2;
    wp::float32 var_144;
    wp::float32 var_145;
    wp::float32 var_146;
    wp::vec_t<3, wp::float32> var_147;
    wp::vec_t<3, wp::float32> var_148;
    const wp::int32 var_149 = 0;
    wp::float32 var_150;
    const wp::int32 var_151 = 1;
    wp::float32 var_152;
    const wp::int32 var_153 = 2;
    wp::float32 var_154;
    const wp::int32 var_155 = 2;
    wp::float32 var_156;
    wp::float32 var_157;
    wp::float32 var_158;
    wp::vec_t<3, wp::float32> var_159;
    wp::vec_t<3, wp::float32> var_160;
    wp::float32 var_161;
    wp::vec_t<3, wp::float32> var_162;
    wp::float32 var_163;
    //---------
    // forward
    // def box_project(center: wp.vec3, half_size: wp.vec3, xyz: wp.vec3) -> Tuple[float, wp.vec3]:       <L 326>
    // r = xyz - center                                                                       <L 327>
    var_0 = wp::sub(var_xyz, var_center);
    // q = wp.vec3(wp.abs(r[0]) - half_size[0], wp.abs(r[1]) - half_size[1], wp.abs(r[2]) - half_size[2])       <L 328>
    var_2 = wp::extract(var_0, var_1);
    var_3 = wp::abs(var_2);
    var_5 = wp::extract(var_half_size, var_4);
    var_6 = wp::sub(var_3, var_5);
    var_8 = wp::extract(var_0, var_7);
    var_9 = wp::abs(var_8);
    var_11 = wp::extract(var_half_size, var_10);
    var_12 = wp::sub(var_9, var_11);
    var_14 = wp::extract(var_0, var_13);
    var_15 = wp::abs(var_14);
    var_17 = wp::extract(var_half_size, var_16);
    var_18 = wp::sub(var_15, var_17);
    var_19 = wp::vec_t<3, wp::float32>(var_6, var_12, var_18);
    // if q[0] <= 0.0 and q[1] <= 0.0 and q[2] <= 0.0:                                        <L 330>
    var_21 = wp::extract(var_19, var_20);
    var_23 = (var_21 <= var_22);
    var_25 = wp::extract(var_19, var_24);
    var_27 = (var_25 <= var_26);
    var_29 = wp::extract(var_19, var_28);
    var_31 = (var_29 <= var_30);
    var_32 = var_23 && var_27 && var_31;
    if (var_32) {
        // return 0.0, xyz                                                                    <L 331>
        ret_0 = var_33;
        ret_1 = var_xyz;
        return;
    }
    if (!var_32) {
        // dist_sqr = 0.0                                                                     <L 334>
        // eps = 1e-4                                                                         <L 335>
        // point = wp.vec3(xyz[0], xyz[1], xyz[2])                                            <L 336>
        var_37 = wp::extract(var_xyz, var_36);
        var_39 = wp::extract(var_xyz, var_38);
        var_41 = wp::extract(var_xyz, var_40);
        var_42 = wp::vec_t<3, wp::float32>(var_37, var_39, var_41);
        // if q[0] >= 0.0:                                                                    <L 338>
        var_44 = wp::extract(var_19, var_43);
        var_46 = (var_44 >= var_45);
        if (var_46) {
            // dist_sqr += q[0] * q[0]                                                        <L 339>
            var_48 = wp::extract(var_19, var_47);
            var_50 = wp::extract(var_19, var_49);
            var_51 = wp::mul(var_48, var_50);
            var_52 = wp::add(var_34, var_51);
            // if r[0] > 0.0:                                                                 <L 340>
            var_54 = wp::extract(var_0, var_53);
            var_56 = (var_54 > var_55);
            if (var_56) {
                // point = wp.vec3(point[0] - (q[0] + eps), point[1], point[2])               <L 341>
                var_58 = wp::extract(var_42, var_57);
                var_60 = wp::extract(var_19, var_59);
                var_61 = wp::add(var_60, var_35);
                var_62 = wp::sub(var_58, var_61);
                var_64 = wp::extract(var_42, var_63);
                var_66 = wp::extract(var_42, var_65);
                var_67 = wp::vec_t<3, wp::float32>(var_62, var_64, var_66);
            }
            var_68 = wp::where(var_56, var_67, var_42);
            if (!var_56) {
                // point = wp.vec3(point[0] + (q[0] + eps), point[1], point[2])               <L 343>
                var_70 = wp::extract(var_68, var_69);
                var_72 = wp::extract(var_19, var_71);
                var_73 = wp::add(var_72, var_35);
                var_74 = wp::add(var_70, var_73);
                var_76 = wp::extract(var_68, var_75);
                var_78 = wp::extract(var_68, var_77);
                var_79 = wp::vec_t<3, wp::float32>(var_74, var_76, var_78);
            }
            var_80 = wp::where(var_56, var_68, var_79);
        }
        var_81 = wp::where(var_46, var_52, var_34);
        var_82 = wp::where(var_46, var_80, var_42);
        // if q[1] >= 0.0:                                                                    <L 345>
        var_84 = wp::extract(var_19, var_83);
        var_86 = (var_84 >= var_85);
        if (var_86) {
            // dist_sqr += q[1] * q[1]                                                        <L 346>
            var_88 = wp::extract(var_19, var_87);
            var_90 = wp::extract(var_19, var_89);
            var_91 = wp::mul(var_88, var_90);
            var_92 = wp::add(var_81, var_91);
            // if r[1] > 0.0:                                                                 <L 347>
            var_94 = wp::extract(var_0, var_93);
            var_96 = (var_94 > var_95);
            if (var_96) {
                // point = wp.vec3(point[0], point[1] - (q[1] + eps), point[2])               <L 348>
                var_98 = wp::extract(var_82, var_97);
                var_100 = wp::extract(var_82, var_99);
                var_102 = wp::extract(var_19, var_101);
                var_103 = wp::add(var_102, var_35);
                var_104 = wp::sub(var_100, var_103);
                var_106 = wp::extract(var_82, var_105);
                var_107 = wp::vec_t<3, wp::float32>(var_98, var_104, var_106);
            }
            var_108 = wp::where(var_96, var_107, var_82);
            if (!var_96) {
                // point = wp.vec3(point[0], point[1] + (q[1] + eps), point[2])               <L 350>
                var_110 = wp::extract(var_108, var_109);
                var_112 = wp::extract(var_108, var_111);
                var_114 = wp::extract(var_19, var_113);
                var_115 = wp::add(var_114, var_35);
                var_116 = wp::add(var_112, var_115);
                var_118 = wp::extract(var_108, var_117);
                var_119 = wp::vec_t<3, wp::float32>(var_110, var_116, var_118);
            }
            var_120 = wp::where(var_96, var_108, var_119);
        }
        var_121 = wp::where(var_86, var_92, var_81);
        var_122 = wp::where(var_86, var_120, var_82);
        // if q[2] >= 0.0:                                                                    <L 352>
        var_124 = wp::extract(var_19, var_123);
        var_126 = (var_124 >= var_125);
        if (var_126) {
            // dist_sqr += q[2] * q[2]                                                        <L 353>
            var_128 = wp::extract(var_19, var_127);
            var_130 = wp::extract(var_19, var_129);
            var_131 = wp::mul(var_128, var_130);
            var_132 = wp::add(var_121, var_131);
            // if r[2] > 0.0:                                                                 <L 354>
            var_134 = wp::extract(var_0, var_133);
            var_136 = (var_134 > var_135);
            if (var_136) {
                // point = wp.vec3(point[0], point[1], point[2] - (q[2] + eps))               <L 355>
                var_138 = wp::extract(var_122, var_137);
                var_140 = wp::extract(var_122, var_139);
                var_142 = wp::extract(var_122, var_141);
                var_144 = wp::extract(var_19, var_143);
                var_145 = wp::add(var_144, var_35);
                var_146 = wp::sub(var_142, var_145);
                var_147 = wp::vec_t<3, wp::float32>(var_138, var_140, var_146);
            }
            var_148 = wp::where(var_136, var_147, var_122);
            if (!var_136) {
                // point = wp.vec3(point[0], point[1], point[2] + (q[2] + eps))               <L 357>
                var_150 = wp::extract(var_148, var_149);
                var_152 = wp::extract(var_148, var_151);
                var_154 = wp::extract(var_148, var_153);
                var_156 = wp::extract(var_19, var_155);
                var_157 = wp::add(var_156, var_35);
                var_158 = wp::add(var_154, var_157);
                var_159 = wp::vec_t<3, wp::float32>(var_150, var_152, var_158);
            }
            var_160 = wp::where(var_136, var_148, var_159);
        }
        var_161 = wp::where(var_126, var_132, var_121);
        var_162 = wp::where(var_126, var_160, var_122);
        // return wp.sqrt(dist_sqr), point                                                    <L 359>
        var_163 = wp::sqrt(var_161);
        ret_0 = var_163;
        ret_1 = var_162;
        return;
    }
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_sdf.py:253
static CUDA_CALLABLE void find_oct_0(
    wp::array_t<wp::vec_t<8, wp::int32>> var_oct_child,
    wp::array_t<wp::vec_t<3, wp::float32>> var_oct_aabb,
    wp::vec_t<3, wp::float32> var_p,
    bool var_grad,
    wp::int32 var_root,
    wp::int32 & ret_0,
    wp::tuple_t<wp::vec_t<8, wp::float32>, wp::vec_t<8, wp::float32>, wp::vec_t<8, wp::float32>> & ret_1)
{
    //---------
    // primal vars
    wp::int32 var_0;
    const wp::int32 var_1 = 100;
    wp::int32 var_2;
    const wp::float32 var_3 = 0.0;
    wp::vec_t<8, wp::float32> var_4;
    const wp::float32 var_5 = 0.0;
    wp::vec_t<8, wp::float32> var_6;
    const wp::float32 var_7 = 0.0;
    wp::vec_t<8, wp::float32> var_8;
    const wp::float32 var_9 = 1e-06;
    const wp::int32 var_10 = 0;
    bool var_11;
    const wp::int32 var_12 = 1;
    wp::int32 var_13;
    wp::int32 var_14;
    const wp::int32 var_15 = 1;
    const wp::int32 var_16 = -1;
    bool var_17;
    const wp::str var_18 = "ERROR: Invalid node number\n";
    const wp::int32 var_19 = 1;
    const wp::int32 var_20 = -1;
    wp::tuple_t<wp::vec_t<8, wp::float32>, wp::vec_t<8, wp::float32>, wp::vec_t<8, wp::float32>> var_21;
    const wp::int32 var_22 = 0;
    wp::vec_t<3, wp::float32>* var_23;
    const wp::int32 var_24 = 1;
    wp::vec_t<3, wp::float32>* var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    const wp::int32 var_29 = 0;
    wp::vec_t<3, wp::float32>* var_30;
    const wp::int32 var_31 = 1;
    wp::vec_t<3, wp::float32>* var_32;
    wp::vec_t<3, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    wp::vec_t<3, wp::float32> var_35;
    const wp::int32 var_36 = 0;
    wp::float32 var_37;
    wp::float32 var_38;
    const wp::int32 var_39 = 0;
    wp::float32 var_40;
    bool var_41;
    const wp::int32 var_42 = 0;
    wp::float32 var_43;
    wp::float32 var_44;
    const wp::int32 var_45 = 0;
    wp::float32 var_46;
    bool var_47;
    const wp::int32 var_48 = 1;
    wp::float32 var_49;
    wp::float32 var_50;
    const wp::int32 var_51 = 1;
    wp::float32 var_52;
    bool var_53;
    const wp::int32 var_54 = 1;
    wp::float32 var_55;
    wp::float32 var_56;
    const wp::int32 var_57 = 1;
    wp::float32 var_58;
    bool var_59;
    const wp::int32 var_60 = 2;
    wp::float32 var_61;
    wp::float32 var_62;
    const wp::int32 var_63 = 2;
    wp::float32 var_64;
    bool var_65;
    const wp::int32 var_66 = 2;
    wp::float32 var_67;
    wp::float32 var_68;
    const wp::int32 var_69 = 2;
    wp::float32 var_70;
    bool var_71;
    bool var_72;
    wp::int32 var_73;
    wp::vec_t<3, wp::float32> var_74;
    wp::vec_t<3, wp::float32> var_75;
    wp::vec_t<3, wp::float32> var_76;
    wp::vec_t<8, wp::int32>* var_77;
    const wp::int32 var_78 = 0;
    wp::int32 var_79;
    wp::vec_t<8, wp::int32> var_80;
    const wp::int32 var_81 = 1;
    const wp::int32 var_82 = -1;
    bool var_83;
    wp::vec_t<8, wp::int32>* var_84;
    const wp::int32 var_85 = 1;
    wp::int32 var_86;
    wp::vec_t<8, wp::int32> var_87;
    const wp::int32 var_88 = 1;
    const wp::int32 var_89 = -1;
    bool var_90;
    wp::vec_t<8, wp::int32>* var_91;
    const wp::int32 var_92 = 2;
    wp::int32 var_93;
    wp::vec_t<8, wp::int32> var_94;
    const wp::int32 var_95 = 1;
    const wp::int32 var_96 = -1;
    bool var_97;
    wp::vec_t<8, wp::int32>* var_98;
    const wp::int32 var_99 = 3;
    wp::int32 var_100;
    wp::vec_t<8, wp::int32> var_101;
    const wp::int32 var_102 = 1;
    const wp::int32 var_103 = -1;
    bool var_104;
    wp::vec_t<8, wp::int32>* var_105;
    const wp::int32 var_106 = 4;
    wp::int32 var_107;
    wp::vec_t<8, wp::int32> var_108;
    const wp::int32 var_109 = 1;
    const wp::int32 var_110 = -1;
    bool var_111;
    wp::vec_t<8, wp::int32>* var_112;
    const wp::int32 var_113 = 5;
    wp::int32 var_114;
    wp::vec_t<8, wp::int32> var_115;
    const wp::int32 var_116 = 1;
    const wp::int32 var_117 = -1;
    bool var_118;
    wp::vec_t<8, wp::int32>* var_119;
    const wp::int32 var_120 = 6;
    wp::int32 var_121;
    wp::vec_t<8, wp::int32> var_122;
    const wp::int32 var_123 = 1;
    const wp::int32 var_124 = -1;
    bool var_125;
    wp::vec_t<8, wp::int32>* var_126;
    const wp::int32 var_127 = 7;
    wp::int32 var_128;
    wp::vec_t<8, wp::int32> var_129;
    const wp::int32 var_130 = 1;
    const wp::int32 var_131 = -1;
    bool var_132;
    bool var_133;
    const wp::int32 var_134 = 0;
    bool var_135;
    const wp::int32 var_136 = 1;
    wp::int32 var_137;
    const wp::int32 var_138 = 0;
    wp::float32 var_139;
    const wp::float32 var_140 = 1.0;
    const wp::int32 var_141 = 0;
    wp::float32 var_142;
    wp::float32 var_143;
    wp::float32 var_144;
    const wp::int32 var_145 = 2;
    wp::int32 var_146;
    const wp::int32 var_147 = 1;
    wp::float32 var_148;
    const wp::float32 var_149 = 1.0;
    const wp::int32 var_150 = 1;
    wp::float32 var_151;
    wp::float32 var_152;
    wp::float32 var_153;
    wp::float32 var_154;
    const wp::int32 var_155 = 4;
    wp::int32 var_156;
    const wp::int32 var_157 = 2;
    wp::float32 var_158;
    const wp::float32 var_159 = 1.0;
    const wp::int32 var_160 = 2;
    wp::float32 var_161;
    wp::float32 var_162;
    wp::float32 var_163;
    wp::float32 var_164;
    const wp::int32 var_165 = 1;
    wp::int32 var_166;
    const wp::float32 var_167 = 1.0;
    const wp::float32 var_168 = 1.0;
    const wp::float32 var_169 = -1.0;
    wp::float32 var_170;
    const wp::int32 var_171 = 2;
    wp::int32 var_172;
    const wp::int32 var_173 = 1;
    wp::float32 var_174;
    const wp::float32 var_175 = 1.0;
    const wp::int32 var_176 = 1;
    wp::float32 var_177;
    wp::float32 var_178;
    wp::float32 var_179;
    wp::float32 var_180;
    const wp::int32 var_181 = 4;
    wp::int32 var_182;
    const wp::int32 var_183 = 2;
    wp::float32 var_184;
    const wp::float32 var_185 = 1.0;
    const wp::int32 var_186 = 2;
    wp::float32 var_187;
    wp::float32 var_188;
    wp::float32 var_189;
    wp::float32 var_190;
    const wp::int32 var_191 = 1;
    wp::int32 var_192;
    const wp::int32 var_193 = 0;
    wp::float32 var_194;
    const wp::float32 var_195 = 1.0;
    const wp::int32 var_196 = 0;
    wp::float32 var_197;
    wp::float32 var_198;
    wp::float32 var_199;
    const wp::int32 var_200 = 2;
    wp::int32 var_201;
    const wp::float32 var_202 = 1.0;
    const wp::float32 var_203 = 1.0;
    const wp::float32 var_204 = -1.0;
    wp::float32 var_205;
    wp::float32 var_206;
    const wp::int32 var_207 = 4;
    wp::int32 var_208;
    const wp::int32 var_209 = 2;
    wp::float32 var_210;
    const wp::float32 var_211 = 1.0;
    const wp::int32 var_212 = 2;
    wp::float32 var_213;
    wp::float32 var_214;
    wp::float32 var_215;
    wp::float32 var_216;
    const wp::int32 var_217 = 1;
    wp::int32 var_218;
    const wp::int32 var_219 = 0;
    wp::float32 var_220;
    const wp::float32 var_221 = 1.0;
    const wp::int32 var_222 = 0;
    wp::float32 var_223;
    wp::float32 var_224;
    wp::float32 var_225;
    const wp::int32 var_226 = 2;
    wp::int32 var_227;
    const wp::int32 var_228 = 1;
    wp::float32 var_229;
    const wp::float32 var_230 = 1.0;
    const wp::int32 var_231 = 1;
    wp::float32 var_232;
    wp::float32 var_233;
    wp::float32 var_234;
    wp::float32 var_235;
    const wp::int32 var_236 = 4;
    wp::int32 var_237;
    const wp::float32 var_238 = 1.0;
    const wp::float32 var_239 = 1.0;
    const wp::float32 var_240 = -1.0;
    wp::float32 var_241;
    wp::float32 var_242;
    const wp::int32 var_243 = 1;
    bool var_244;
    const wp::int32 var_245 = 1;
    wp::int32 var_246;
    const wp::int32 var_247 = 0;
    wp::float32 var_248;
    const wp::float32 var_249 = 1.0;
    const wp::int32 var_250 = 0;
    wp::float32 var_251;
    wp::float32 var_252;
    wp::float32 var_253;
    const wp::int32 var_254 = 2;
    wp::int32 var_255;
    const wp::int32 var_256 = 1;
    wp::float32 var_257;
    const wp::float32 var_258 = 1.0;
    const wp::int32 var_259 = 1;
    wp::float32 var_260;
    wp::float32 var_261;
    wp::float32 var_262;
    wp::float32 var_263;
    const wp::int32 var_264 = 4;
    wp::int32 var_265;
    const wp::int32 var_266 = 2;
    wp::float32 var_267;
    const wp::float32 var_268 = 1.0;
    const wp::int32 var_269 = 2;
    wp::float32 var_270;
    wp::float32 var_271;
    wp::float32 var_272;
    wp::float32 var_273;
    const wp::int32 var_274 = 1;
    wp::int32 var_275;
    const wp::float32 var_276 = 1.0;
    const wp::float32 var_277 = 1.0;
    const wp::float32 var_278 = -1.0;
    wp::float32 var_279;
    const wp::int32 var_280 = 2;
    wp::int32 var_281;
    const wp::int32 var_282 = 1;
    wp::float32 var_283;
    const wp::float32 var_284 = 1.0;
    const wp::int32 var_285 = 1;
    wp::float32 var_286;
    wp::float32 var_287;
    wp::float32 var_288;
    wp::float32 var_289;
    const wp::int32 var_290 = 4;
    wp::int32 var_291;
    const wp::int32 var_292 = 2;
    wp::float32 var_293;
    const wp::float32 var_294 = 1.0;
    const wp::int32 var_295 = 2;
    wp::float32 var_296;
    wp::float32 var_297;
    wp::float32 var_298;
    wp::float32 var_299;
    const wp::int32 var_300 = 1;
    wp::int32 var_301;
    const wp::int32 var_302 = 0;
    wp::float32 var_303;
    const wp::float32 var_304 = 1.0;
    const wp::int32 var_305 = 0;
    wp::float32 var_306;
    wp::float32 var_307;
    wp::float32 var_308;
    const wp::int32 var_309 = 2;
    wp::int32 var_310;
    const wp::float32 var_311 = 1.0;
    const wp::float32 var_312 = 1.0;
    const wp::float32 var_313 = -1.0;
    wp::float32 var_314;
    wp::float32 var_315;
    const wp::int32 var_316 = 4;
    wp::int32 var_317;
    const wp::int32 var_318 = 2;
    wp::float32 var_319;
    const wp::float32 var_320 = 1.0;
    const wp::int32 var_321 = 2;
    wp::float32 var_322;
    wp::float32 var_323;
    wp::float32 var_324;
    wp::float32 var_325;
    const wp::int32 var_326 = 1;
    wp::int32 var_327;
    const wp::int32 var_328 = 0;
    wp::float32 var_329;
    const wp::float32 var_330 = 1.0;
    const wp::int32 var_331 = 0;
    wp::float32 var_332;
    wp::float32 var_333;
    wp::float32 var_334;
    const wp::int32 var_335 = 2;
    wp::int32 var_336;
    const wp::int32 var_337 = 1;
    wp::float32 var_338;
    const wp::float32 var_339 = 1.0;
    const wp::int32 var_340 = 1;
    wp::float32 var_341;
    wp::float32 var_342;
    wp::float32 var_343;
    wp::float32 var_344;
    const wp::int32 var_345 = 4;
    wp::int32 var_346;
    const wp::float32 var_347 = 1.0;
    const wp::float32 var_348 = 1.0;
    const wp::float32 var_349 = -1.0;
    wp::float32 var_350;
    wp::float32 var_351;
    const wp::int32 var_352 = 2;
    bool var_353;
    const wp::int32 var_354 = 1;
    wp::int32 var_355;
    const wp::int32 var_356 = 0;
    wp::float32 var_357;
    const wp::float32 var_358 = 1.0;
    const wp::int32 var_359 = 0;
    wp::float32 var_360;
    wp::float32 var_361;
    wp::float32 var_362;
    const wp::int32 var_363 = 2;
    wp::int32 var_364;
    const wp::int32 var_365 = 1;
    wp::float32 var_366;
    const wp::float32 var_367 = 1.0;
    const wp::int32 var_368 = 1;
    wp::float32 var_369;
    wp::float32 var_370;
    wp::float32 var_371;
    wp::float32 var_372;
    const wp::int32 var_373 = 4;
    wp::int32 var_374;
    const wp::int32 var_375 = 2;
    wp::float32 var_376;
    const wp::float32 var_377 = 1.0;
    const wp::int32 var_378 = 2;
    wp::float32 var_379;
    wp::float32 var_380;
    wp::float32 var_381;
    wp::float32 var_382;
    const wp::int32 var_383 = 1;
    wp::int32 var_384;
    const wp::float32 var_385 = 1.0;
    const wp::float32 var_386 = 1.0;
    const wp::float32 var_387 = -1.0;
    wp::float32 var_388;
    const wp::int32 var_389 = 2;
    wp::int32 var_390;
    const wp::int32 var_391 = 1;
    wp::float32 var_392;
    const wp::float32 var_393 = 1.0;
    const wp::int32 var_394 = 1;
    wp::float32 var_395;
    wp::float32 var_396;
    wp::float32 var_397;
    wp::float32 var_398;
    const wp::int32 var_399 = 4;
    wp::int32 var_400;
    const wp::int32 var_401 = 2;
    wp::float32 var_402;
    const wp::float32 var_403 = 1.0;
    const wp::int32 var_404 = 2;
    wp::float32 var_405;
    wp::float32 var_406;
    wp::float32 var_407;
    wp::float32 var_408;
    const wp::int32 var_409 = 1;
    wp::int32 var_410;
    const wp::int32 var_411 = 0;
    wp::float32 var_412;
    const wp::float32 var_413 = 1.0;
    const wp::int32 var_414 = 0;
    wp::float32 var_415;
    wp::float32 var_416;
    wp::float32 var_417;
    const wp::int32 var_418 = 2;
    wp::int32 var_419;
    const wp::float32 var_420 = 1.0;
    const wp::float32 var_421 = 1.0;
    const wp::float32 var_422 = -1.0;
    wp::float32 var_423;
    wp::float32 var_424;
    const wp::int32 var_425 = 4;
    wp::int32 var_426;
    const wp::int32 var_427 = 2;
    wp::float32 var_428;
    const wp::float32 var_429 = 1.0;
    const wp::int32 var_430 = 2;
    wp::float32 var_431;
    wp::float32 var_432;
    wp::float32 var_433;
    wp::float32 var_434;
    const wp::int32 var_435 = 1;
    wp::int32 var_436;
    const wp::int32 var_437 = 0;
    wp::float32 var_438;
    const wp::float32 var_439 = 1.0;
    const wp::int32 var_440 = 0;
    wp::float32 var_441;
    wp::float32 var_442;
    wp::float32 var_443;
    const wp::int32 var_444 = 2;
    wp::int32 var_445;
    const wp::int32 var_446 = 1;
    wp::float32 var_447;
    const wp::float32 var_448 = 1.0;
    const wp::int32 var_449 = 1;
    wp::float32 var_450;
    wp::float32 var_451;
    wp::float32 var_452;
    wp::float32 var_453;
    const wp::int32 var_454 = 4;
    wp::int32 var_455;
    const wp::float32 var_456 = 1.0;
    const wp::float32 var_457 = 1.0;
    const wp::float32 var_458 = -1.0;
    wp::float32 var_459;
    wp::float32 var_460;
    const wp::int32 var_461 = 3;
    bool var_462;
    const wp::int32 var_463 = 1;
    wp::int32 var_464;
    const wp::int32 var_465 = 0;
    wp::float32 var_466;
    const wp::float32 var_467 = 1.0;
    const wp::int32 var_468 = 0;
    wp::float32 var_469;
    wp::float32 var_470;
    wp::float32 var_471;
    const wp::int32 var_472 = 2;
    wp::int32 var_473;
    const wp::int32 var_474 = 1;
    wp::float32 var_475;
    const wp::float32 var_476 = 1.0;
    const wp::int32 var_477 = 1;
    wp::float32 var_478;
    wp::float32 var_479;
    wp::float32 var_480;
    wp::float32 var_481;
    const wp::int32 var_482 = 4;
    wp::int32 var_483;
    const wp::int32 var_484 = 2;
    wp::float32 var_485;
    const wp::float32 var_486 = 1.0;
    const wp::int32 var_487 = 2;
    wp::float32 var_488;
    wp::float32 var_489;
    wp::float32 var_490;
    wp::float32 var_491;
    const wp::int32 var_492 = 1;
    wp::int32 var_493;
    const wp::float32 var_494 = 1.0;
    const wp::float32 var_495 = 1.0;
    const wp::float32 var_496 = -1.0;
    wp::float32 var_497;
    const wp::int32 var_498 = 2;
    wp::int32 var_499;
    const wp::int32 var_500 = 1;
    wp::float32 var_501;
    const wp::float32 var_502 = 1.0;
    const wp::int32 var_503 = 1;
    wp::float32 var_504;
    wp::float32 var_505;
    wp::float32 var_506;
    wp::float32 var_507;
    const wp::int32 var_508 = 4;
    wp::int32 var_509;
    const wp::int32 var_510 = 2;
    wp::float32 var_511;
    const wp::float32 var_512 = 1.0;
    const wp::int32 var_513 = 2;
    wp::float32 var_514;
    wp::float32 var_515;
    wp::float32 var_516;
    wp::float32 var_517;
    const wp::int32 var_518 = 1;
    wp::int32 var_519;
    const wp::int32 var_520 = 0;
    wp::float32 var_521;
    const wp::float32 var_522 = 1.0;
    const wp::int32 var_523 = 0;
    wp::float32 var_524;
    wp::float32 var_525;
    wp::float32 var_526;
    const wp::int32 var_527 = 2;
    wp::int32 var_528;
    const wp::float32 var_529 = 1.0;
    const wp::float32 var_530 = 1.0;
    const wp::float32 var_531 = -1.0;
    wp::float32 var_532;
    wp::float32 var_533;
    const wp::int32 var_534 = 4;
    wp::int32 var_535;
    const wp::int32 var_536 = 2;
    wp::float32 var_537;
    const wp::float32 var_538 = 1.0;
    const wp::int32 var_539 = 2;
    wp::float32 var_540;
    wp::float32 var_541;
    wp::float32 var_542;
    wp::float32 var_543;
    const wp::int32 var_544 = 1;
    wp::int32 var_545;
    const wp::int32 var_546 = 0;
    wp::float32 var_547;
    const wp::float32 var_548 = 1.0;
    const wp::int32 var_549 = 0;
    wp::float32 var_550;
    wp::float32 var_551;
    wp::float32 var_552;
    const wp::int32 var_553 = 2;
    wp::int32 var_554;
    const wp::int32 var_555 = 1;
    wp::float32 var_556;
    const wp::float32 var_557 = 1.0;
    const wp::int32 var_558 = 1;
    wp::float32 var_559;
    wp::float32 var_560;
    wp::float32 var_561;
    wp::float32 var_562;
    const wp::int32 var_563 = 4;
    wp::int32 var_564;
    const wp::float32 var_565 = 1.0;
    const wp::float32 var_566 = 1.0;
    const wp::float32 var_567 = -1.0;
    wp::float32 var_568;
    wp::float32 var_569;
    const wp::int32 var_570 = 4;
    bool var_571;
    const wp::int32 var_572 = 1;
    wp::int32 var_573;
    const wp::int32 var_574 = 0;
    wp::float32 var_575;
    const wp::float32 var_576 = 1.0;
    const wp::int32 var_577 = 0;
    wp::float32 var_578;
    wp::float32 var_579;
    wp::float32 var_580;
    const wp::int32 var_581 = 2;
    wp::int32 var_582;
    const wp::int32 var_583 = 1;
    wp::float32 var_584;
    const wp::float32 var_585 = 1.0;
    const wp::int32 var_586 = 1;
    wp::float32 var_587;
    wp::float32 var_588;
    wp::float32 var_589;
    wp::float32 var_590;
    const wp::int32 var_591 = 4;
    wp::int32 var_592;
    const wp::int32 var_593 = 2;
    wp::float32 var_594;
    const wp::float32 var_595 = 1.0;
    const wp::int32 var_596 = 2;
    wp::float32 var_597;
    wp::float32 var_598;
    wp::float32 var_599;
    wp::float32 var_600;
    const wp::int32 var_601 = 1;
    wp::int32 var_602;
    const wp::float32 var_603 = 1.0;
    const wp::float32 var_604 = 1.0;
    const wp::float32 var_605 = -1.0;
    wp::float32 var_606;
    const wp::int32 var_607 = 2;
    wp::int32 var_608;
    const wp::int32 var_609 = 1;
    wp::float32 var_610;
    const wp::float32 var_611 = 1.0;
    const wp::int32 var_612 = 1;
    wp::float32 var_613;
    wp::float32 var_614;
    wp::float32 var_615;
    wp::float32 var_616;
    const wp::int32 var_617 = 4;
    wp::int32 var_618;
    const wp::int32 var_619 = 2;
    wp::float32 var_620;
    const wp::float32 var_621 = 1.0;
    const wp::int32 var_622 = 2;
    wp::float32 var_623;
    wp::float32 var_624;
    wp::float32 var_625;
    wp::float32 var_626;
    const wp::int32 var_627 = 1;
    wp::int32 var_628;
    const wp::int32 var_629 = 0;
    wp::float32 var_630;
    const wp::float32 var_631 = 1.0;
    const wp::int32 var_632 = 0;
    wp::float32 var_633;
    wp::float32 var_634;
    wp::float32 var_635;
    const wp::int32 var_636 = 2;
    wp::int32 var_637;
    const wp::float32 var_638 = 1.0;
    const wp::float32 var_639 = 1.0;
    const wp::float32 var_640 = -1.0;
    wp::float32 var_641;
    wp::float32 var_642;
    const wp::int32 var_643 = 4;
    wp::int32 var_644;
    const wp::int32 var_645 = 2;
    wp::float32 var_646;
    const wp::float32 var_647 = 1.0;
    const wp::int32 var_648 = 2;
    wp::float32 var_649;
    wp::float32 var_650;
    wp::float32 var_651;
    wp::float32 var_652;
    const wp::int32 var_653 = 1;
    wp::int32 var_654;
    const wp::int32 var_655 = 0;
    wp::float32 var_656;
    const wp::float32 var_657 = 1.0;
    const wp::int32 var_658 = 0;
    wp::float32 var_659;
    wp::float32 var_660;
    wp::float32 var_661;
    const wp::int32 var_662 = 2;
    wp::int32 var_663;
    const wp::int32 var_664 = 1;
    wp::float32 var_665;
    const wp::float32 var_666 = 1.0;
    const wp::int32 var_667 = 1;
    wp::float32 var_668;
    wp::float32 var_669;
    wp::float32 var_670;
    wp::float32 var_671;
    const wp::int32 var_672 = 4;
    wp::int32 var_673;
    const wp::float32 var_674 = 1.0;
    const wp::float32 var_675 = 1.0;
    const wp::float32 var_676 = -1.0;
    wp::float32 var_677;
    wp::float32 var_678;
    const wp::int32 var_679 = 5;
    bool var_680;
    const wp::int32 var_681 = 1;
    wp::int32 var_682;
    const wp::int32 var_683 = 0;
    wp::float32 var_684;
    const wp::float32 var_685 = 1.0;
    const wp::int32 var_686 = 0;
    wp::float32 var_687;
    wp::float32 var_688;
    wp::float32 var_689;
    const wp::int32 var_690 = 2;
    wp::int32 var_691;
    const wp::int32 var_692 = 1;
    wp::float32 var_693;
    const wp::float32 var_694 = 1.0;
    const wp::int32 var_695 = 1;
    wp::float32 var_696;
    wp::float32 var_697;
    wp::float32 var_698;
    wp::float32 var_699;
    const wp::int32 var_700 = 4;
    wp::int32 var_701;
    const wp::int32 var_702 = 2;
    wp::float32 var_703;
    const wp::float32 var_704 = 1.0;
    const wp::int32 var_705 = 2;
    wp::float32 var_706;
    wp::float32 var_707;
    wp::float32 var_708;
    wp::float32 var_709;
    const wp::int32 var_710 = 1;
    wp::int32 var_711;
    const wp::float32 var_712 = 1.0;
    const wp::float32 var_713 = 1.0;
    const wp::float32 var_714 = -1.0;
    wp::float32 var_715;
    const wp::int32 var_716 = 2;
    wp::int32 var_717;
    const wp::int32 var_718 = 1;
    wp::float32 var_719;
    const wp::float32 var_720 = 1.0;
    const wp::int32 var_721 = 1;
    wp::float32 var_722;
    wp::float32 var_723;
    wp::float32 var_724;
    wp::float32 var_725;
    const wp::int32 var_726 = 4;
    wp::int32 var_727;
    const wp::int32 var_728 = 2;
    wp::float32 var_729;
    const wp::float32 var_730 = 1.0;
    const wp::int32 var_731 = 2;
    wp::float32 var_732;
    wp::float32 var_733;
    wp::float32 var_734;
    wp::float32 var_735;
    const wp::int32 var_736 = 1;
    wp::int32 var_737;
    const wp::int32 var_738 = 0;
    wp::float32 var_739;
    const wp::float32 var_740 = 1.0;
    const wp::int32 var_741 = 0;
    wp::float32 var_742;
    wp::float32 var_743;
    wp::float32 var_744;
    const wp::int32 var_745 = 2;
    wp::int32 var_746;
    const wp::float32 var_747 = 1.0;
    const wp::float32 var_748 = 1.0;
    const wp::float32 var_749 = -1.0;
    wp::float32 var_750;
    wp::float32 var_751;
    const wp::int32 var_752 = 4;
    wp::int32 var_753;
    const wp::int32 var_754 = 2;
    wp::float32 var_755;
    const wp::float32 var_756 = 1.0;
    const wp::int32 var_757 = 2;
    wp::float32 var_758;
    wp::float32 var_759;
    wp::float32 var_760;
    wp::float32 var_761;
    const wp::int32 var_762 = 1;
    wp::int32 var_763;
    const wp::int32 var_764 = 0;
    wp::float32 var_765;
    const wp::float32 var_766 = 1.0;
    const wp::int32 var_767 = 0;
    wp::float32 var_768;
    wp::float32 var_769;
    wp::float32 var_770;
    const wp::int32 var_771 = 2;
    wp::int32 var_772;
    const wp::int32 var_773 = 1;
    wp::float32 var_774;
    const wp::float32 var_775 = 1.0;
    const wp::int32 var_776 = 1;
    wp::float32 var_777;
    wp::float32 var_778;
    wp::float32 var_779;
    wp::float32 var_780;
    const wp::int32 var_781 = 4;
    wp::int32 var_782;
    const wp::float32 var_783 = 1.0;
    const wp::float32 var_784 = 1.0;
    const wp::float32 var_785 = -1.0;
    wp::float32 var_786;
    wp::float32 var_787;
    const wp::int32 var_788 = 6;
    bool var_789;
    const wp::int32 var_790 = 1;
    wp::int32 var_791;
    const wp::int32 var_792 = 0;
    wp::float32 var_793;
    const wp::float32 var_794 = 1.0;
    const wp::int32 var_795 = 0;
    wp::float32 var_796;
    wp::float32 var_797;
    wp::float32 var_798;
    const wp::int32 var_799 = 2;
    wp::int32 var_800;
    const wp::int32 var_801 = 1;
    wp::float32 var_802;
    const wp::float32 var_803 = 1.0;
    const wp::int32 var_804 = 1;
    wp::float32 var_805;
    wp::float32 var_806;
    wp::float32 var_807;
    wp::float32 var_808;
    const wp::int32 var_809 = 4;
    wp::int32 var_810;
    const wp::int32 var_811 = 2;
    wp::float32 var_812;
    const wp::float32 var_813 = 1.0;
    const wp::int32 var_814 = 2;
    wp::float32 var_815;
    wp::float32 var_816;
    wp::float32 var_817;
    wp::float32 var_818;
    const wp::int32 var_819 = 1;
    wp::int32 var_820;
    const wp::float32 var_821 = 1.0;
    const wp::float32 var_822 = 1.0;
    const wp::float32 var_823 = -1.0;
    wp::float32 var_824;
    const wp::int32 var_825 = 2;
    wp::int32 var_826;
    const wp::int32 var_827 = 1;
    wp::float32 var_828;
    const wp::float32 var_829 = 1.0;
    const wp::int32 var_830 = 1;
    wp::float32 var_831;
    wp::float32 var_832;
    wp::float32 var_833;
    wp::float32 var_834;
    const wp::int32 var_835 = 4;
    wp::int32 var_836;
    const wp::int32 var_837 = 2;
    wp::float32 var_838;
    const wp::float32 var_839 = 1.0;
    const wp::int32 var_840 = 2;
    wp::float32 var_841;
    wp::float32 var_842;
    wp::float32 var_843;
    wp::float32 var_844;
    const wp::int32 var_845 = 1;
    wp::int32 var_846;
    const wp::int32 var_847 = 0;
    wp::float32 var_848;
    const wp::float32 var_849 = 1.0;
    const wp::int32 var_850 = 0;
    wp::float32 var_851;
    wp::float32 var_852;
    wp::float32 var_853;
    const wp::int32 var_854 = 2;
    wp::int32 var_855;
    const wp::float32 var_856 = 1.0;
    const wp::float32 var_857 = 1.0;
    const wp::float32 var_858 = -1.0;
    wp::float32 var_859;
    wp::float32 var_860;
    const wp::int32 var_861 = 4;
    wp::int32 var_862;
    const wp::int32 var_863 = 2;
    wp::float32 var_864;
    const wp::float32 var_865 = 1.0;
    const wp::int32 var_866 = 2;
    wp::float32 var_867;
    wp::float32 var_868;
    wp::float32 var_869;
    wp::float32 var_870;
    const wp::int32 var_871 = 1;
    wp::int32 var_872;
    const wp::int32 var_873 = 0;
    wp::float32 var_874;
    const wp::float32 var_875 = 1.0;
    const wp::int32 var_876 = 0;
    wp::float32 var_877;
    wp::float32 var_878;
    wp::float32 var_879;
    const wp::int32 var_880 = 2;
    wp::int32 var_881;
    const wp::int32 var_882 = 1;
    wp::float32 var_883;
    const wp::float32 var_884 = 1.0;
    const wp::int32 var_885 = 1;
    wp::float32 var_886;
    wp::float32 var_887;
    wp::float32 var_888;
    wp::float32 var_889;
    const wp::int32 var_890 = 4;
    wp::int32 var_891;
    const wp::float32 var_892 = 1.0;
    const wp::float32 var_893 = 1.0;
    const wp::float32 var_894 = -1.0;
    wp::float32 var_895;
    wp::float32 var_896;
    const wp::int32 var_897 = 7;
    bool var_898;
    const wp::int32 var_899 = 1;
    wp::int32 var_900;
    const wp::int32 var_901 = 0;
    wp::float32 var_902;
    const wp::float32 var_903 = 1.0;
    const wp::int32 var_904 = 0;
    wp::float32 var_905;
    wp::float32 var_906;
    wp::float32 var_907;
    const wp::int32 var_908 = 2;
    wp::int32 var_909;
    const wp::int32 var_910 = 1;
    wp::float32 var_911;
    const wp::float32 var_912 = 1.0;
    const wp::int32 var_913 = 1;
    wp::float32 var_914;
    wp::float32 var_915;
    wp::float32 var_916;
    wp::float32 var_917;
    const wp::int32 var_918 = 4;
    wp::int32 var_919;
    const wp::int32 var_920 = 2;
    wp::float32 var_921;
    const wp::float32 var_922 = 1.0;
    const wp::int32 var_923 = 2;
    wp::float32 var_924;
    wp::float32 var_925;
    wp::float32 var_926;
    wp::float32 var_927;
    const wp::int32 var_928 = 1;
    wp::int32 var_929;
    const wp::float32 var_930 = 1.0;
    const wp::float32 var_931 = 1.0;
    const wp::float32 var_932 = -1.0;
    wp::float32 var_933;
    const wp::int32 var_934 = 2;
    wp::int32 var_935;
    const wp::int32 var_936 = 1;
    wp::float32 var_937;
    const wp::float32 var_938 = 1.0;
    const wp::int32 var_939 = 1;
    wp::float32 var_940;
    wp::float32 var_941;
    wp::float32 var_942;
    wp::float32 var_943;
    const wp::int32 var_944 = 4;
    wp::int32 var_945;
    const wp::int32 var_946 = 2;
    wp::float32 var_947;
    const wp::float32 var_948 = 1.0;
    const wp::int32 var_949 = 2;
    wp::float32 var_950;
    wp::float32 var_951;
    wp::float32 var_952;
    wp::float32 var_953;
    const wp::int32 var_954 = 1;
    wp::int32 var_955;
    const wp::int32 var_956 = 0;
    wp::float32 var_957;
    const wp::float32 var_958 = 1.0;
    const wp::int32 var_959 = 0;
    wp::float32 var_960;
    wp::float32 var_961;
    wp::float32 var_962;
    const wp::int32 var_963 = 2;
    wp::int32 var_964;
    const wp::float32 var_965 = 1.0;
    const wp::float32 var_966 = 1.0;
    const wp::float32 var_967 = -1.0;
    wp::float32 var_968;
    wp::float32 var_969;
    const wp::int32 var_970 = 4;
    wp::int32 var_971;
    const wp::int32 var_972 = 2;
    wp::float32 var_973;
    const wp::float32 var_974 = 1.0;
    const wp::int32 var_975 = 2;
    wp::float32 var_976;
    wp::float32 var_977;
    wp::float32 var_978;
    wp::float32 var_979;
    const wp::int32 var_980 = 1;
    wp::int32 var_981;
    const wp::int32 var_982 = 0;
    wp::float32 var_983;
    const wp::float32 var_984 = 1.0;
    const wp::int32 var_985 = 0;
    wp::float32 var_986;
    wp::float32 var_987;
    wp::float32 var_988;
    const wp::int32 var_989 = 2;
    wp::int32 var_990;
    const wp::int32 var_991 = 1;
    wp::float32 var_992;
    const wp::float32 var_993 = 1.0;
    const wp::int32 var_994 = 1;
    wp::float32 var_995;
    wp::float32 var_996;
    wp::float32 var_997;
    wp::float32 var_998;
    const wp::int32 var_999 = 4;
    wp::int32 var_1000;
    const wp::float32 var_1001 = 1.0;
    const wp::float32 var_1002 = 1.0;
    const wp::float32 var_1003 = -1.0;
    wp::float32 var_1004;
    wp::float32 var_1005;
    wp::tuple_t<wp::vec_t<8, wp::float32>, wp::vec_t<8, wp::float32>, wp::vec_t<8, wp::float32>> var_1006;
    const wp::int32 var_1007 = 0;
    wp::float32 var_1008;
    const wp::float32 var_1009 = 0.5;
    bool var_1010;
    const wp::int32 var_1011 = 0;
    const wp::int32 var_1012 = 1;
    wp::int32 var_1013;
    const wp::int32 var_1014 = 1;
    wp::float32 var_1015;
    const wp::float32 var_1016 = 0.5;
    bool var_1017;
    const wp::int32 var_1018 = 0;
    const wp::int32 var_1019 = 1;
    wp::int32 var_1020;
    const wp::int32 var_1021 = 2;
    wp::float32 var_1022;
    const wp::float32 var_1023 = 0.5;
    bool var_1024;
    const wp::int32 var_1025 = 0;
    const wp::int32 var_1026 = 1;
    wp::int32 var_1027;
    wp::vec_t<8, wp::int32>* var_1028;
    const wp::int32 var_1029 = 4;
    wp::int32 var_1030;
    const wp::int32 var_1031 = 2;
    wp::int32 var_1032;
    wp::int32 var_1033;
    wp::int32 var_1034;
    wp::int32 var_1035;
    wp::vec_t<8, wp::int32> var_1036;
    const wp::int32 var_1037 = 1;
    const wp::int32 var_1038 = -1;
    bool var_1039;
    wp::int32 var_1040;
    const wp::int32 var_1041 = 1;
    const wp::int32 var_1042 = -1;
    wp::int32 var_1043;
    const wp::str var_1044 = "ERROR: Node not found\n";
    const wp::int32 var_1045 = 1;
    const wp::int32 var_1046 = -1;
    wp::tuple_t<wp::vec_t<8, wp::float32>, wp::vec_t<8, wp::float32>, wp::vec_t<8, wp::float32>> var_1047;
    //---------
    // forward
    // def find_oct(                                                                          <L 254>
    // stack = root                                                                           <L 257>
    var_0 = wp::copy(var_root);
    // niter = int(100)                                                                       <L 258>
    var_2 = wp::int(var_1);
    // rx = vec8(0.0)                                                                         <L 259>
    var_4 = wp::vec_t<8, wp::float32>(var_3);
    // ry = vec8(0.0)                                                                         <L 260>
    var_6 = wp::vec_t<8, wp::float32>(var_5);
    // rz = vec8(0.0)                                                                         <L 261>
    var_8 = wp::vec_t<8, wp::float32>(var_7);
    // eps = 1e-6                                                                             <L 262>
    // while niter > 0:                                                                       <L 264>
    start_while_0:;
    var_11 = (var_2 > var_10);
    if ((var_11) == false) goto end_while_0;
        // niter -= 1                                                                         <L 265>
        var_13 = wp::sub(var_2, var_12);
        // node = stack                                                                       <L 266>
        var_14 = wp::copy(var_0);
        // if node == -1:                                                                     <L 268>
        var_17 = (var_14 == var_16);
        if (var_17) {
            // wp.printf("ERROR: Invalid node number\n")                                      <L 269>
            printf(var_18);
            // return -1, (rx, ry, rz)                                                        <L 270>
            var_21 = wp::tuple(var_4, var_6, var_8);
            ret_0 = var_20;
            ret_1 = var_21;
            return;
        }
        // vmin = oct_aabb[node, 0] - oct_aabb[node, 1]                                       <L 272>
        var_23 = wp::address(var_oct_aabb, var_14, var_22);
        var_25 = wp::address(var_oct_aabb, var_14, var_24);
        var_27 = wp::load(var_23);
        var_28 = wp::load(var_25);
        var_26 = wp::sub(var_27, var_28);
        // vmax = oct_aabb[node, 0] + oct_aabb[node, 1]                                       <L 273>
        var_30 = wp::address(var_oct_aabb, var_14, var_29);
        var_32 = wp::address(var_oct_aabb, var_14, var_31);
        var_34 = wp::load(var_30);
        var_35 = wp::load(var_32);
        var_33 = wp::add(var_34, var_35);
        // if (                                                                               <L 275>
        // p[0] + eps < vmin[0]                                                               <L 276>
        var_37 = wp::extract(var_p, var_36);
        var_38 = wp::add(var_37, var_9);
        var_40 = wp::extract(var_26, var_39);
        var_41 = (var_38 < var_40);
        // or p[0] - eps > vmax[0]                                                            <L 277>
        var_43 = wp::extract(var_p, var_42);
        var_44 = wp::sub(var_43, var_9);
        var_46 = wp::extract(var_33, var_45);
        var_47 = (var_44 > var_46);
        // or p[1] + eps < vmin[1]                                                            <L 278>
        var_49 = wp::extract(var_p, var_48);
        var_50 = wp::add(var_49, var_9);
        var_52 = wp::extract(var_26, var_51);
        var_53 = (var_50 < var_52);
        // or p[1] - eps > vmax[1]                                                            <L 279>
        var_55 = wp::extract(var_p, var_54);
        var_56 = wp::sub(var_55, var_9);
        var_58 = wp::extract(var_33, var_57);
        var_59 = (var_56 > var_58);
        // or p[2] + eps < vmin[2]                                                            <L 280>
        var_61 = wp::extract(var_p, var_60);
        var_62 = wp::add(var_61, var_9);
        var_64 = wp::extract(var_26, var_63);
        var_65 = (var_62 < var_64);
        // or p[2] - eps > vmax[2]                                                            <L 281>
        var_67 = wp::extract(var_p, var_66);
        var_68 = wp::sub(var_67, var_9);
        var_70 = wp::extract(var_33, var_69);
        var_71 = (var_68 > var_70);
        var_72 = var_41 || var_47 || var_53 || var_59 || var_65 || var_71;
        if (var_72) {
            // continue                                                                       <L 283>
            wp::assign(var_2, var_13);
            goto start_while_0;
        }
        var_73 = wp::where(var_72, var_2, var_13);
        // coord = wp.cw_div(p - vmin, vmax - vmin)                                           <L 285>
        var_74 = wp::sub(var_p, var_26);
        var_75 = wp::sub(var_33, var_26);
        var_76 = wp::cw_div(var_74, var_75);
        // child0 = oct_child[node][0]                                                        <L 289>
        var_77 = wp::address(var_oct_child, var_14);
        var_80 = wp::load(var_77);
        var_79 = wp::extract(var_80, var_78);
        // if (                                                                               <L 290>
        // child0 == -1                                                                       <L 291>
        var_83 = (var_79 == var_82);
        // and oct_child[node][1] == -1                                                       <L 292>
        var_84 = wp::address(var_oct_child, var_14);
        var_87 = wp::load(var_84);
        var_86 = wp::extract(var_87, var_85);
        var_90 = (var_86 == var_89);
        // and oct_child[node][2] == -1                                                       <L 293>
        var_91 = wp::address(var_oct_child, var_14);
        var_94 = wp::load(var_91);
        var_93 = wp::extract(var_94, var_92);
        var_97 = (var_93 == var_96);
        // and oct_child[node][3] == -1                                                       <L 294>
        var_98 = wp::address(var_oct_child, var_14);
        var_101 = wp::load(var_98);
        var_100 = wp::extract(var_101, var_99);
        var_104 = (var_100 == var_103);
        // and oct_child[node][4] == -1                                                       <L 295>
        var_105 = wp::address(var_oct_child, var_14);
        var_108 = wp::load(var_105);
        var_107 = wp::extract(var_108, var_106);
        var_111 = (var_107 == var_110);
        // and oct_child[node][5] == -1                                                       <L 296>
        var_112 = wp::address(var_oct_child, var_14);
        var_115 = wp::load(var_112);
        var_114 = wp::extract(var_115, var_113);
        var_118 = (var_114 == var_117);
        // and oct_child[node][6] == -1                                                       <L 297>
        var_119 = wp::address(var_oct_child, var_14);
        var_122 = wp::load(var_119);
        var_121 = wp::extract(var_122, var_120);
        var_125 = (var_121 == var_124);
        // and oct_child[node][7] == -1                                                       <L 298>
        var_126 = wp::address(var_oct_child, var_14);
        var_129 = wp::load(var_126);
        var_128 = wp::extract(var_129, var_127);
        var_132 = (var_128 == var_131);
        var_133 = var_83 && var_90 && var_97 && var_104 && var_111 && var_118 && var_125 && var_132;
        if (var_133) {
            // for j in range(8):                                                             <L 300>
            // if not grad:                                                                   <L 301>
            var_135 = wp::unot(var_grad);
            if (var_135) {
                // rx[j] = (                                                                  <L 302>
                // (coord[0] if j & 1 else 1.0 - coord[0])                                    <L 303>
                var_137 = wp::bit_and(var_134, var_136);
                if (var_137) {
                    var_139 = wp::extract(var_76, var_138);
                }
                if (!var_137) {
                    var_142 = wp::extract(var_76, var_141);
                    var_143 = wp::sub(var_140, var_142);
                }
                var_144 = wp::where(var_137, var_139, var_143);
                // * (coord[1] if j & 2 else 1.0 - coord[1])                                  <L 304>
                var_146 = wp::bit_and(var_134, var_145);
                if (var_146) {
                    var_148 = wp::extract(var_76, var_147);
                }
                if (!var_146) {
                    var_151 = wp::extract(var_76, var_150);
                    var_152 = wp::sub(var_149, var_151);
                }
                var_153 = wp::where(var_146, var_148, var_152);
                var_154 = wp::mul(var_144, var_153);
                // * (coord[2] if j & 4 else 1.0 - coord[2])                                  <L 305>
                var_156 = wp::bit_and(var_134, var_155);
                if (var_156) {
                    var_158 = wp::extract(var_76, var_157);
                }
                if (!var_156) {
                    var_161 = wp::extract(var_76, var_160);
                    var_162 = wp::sub(var_159, var_161);
                }
                var_163 = wp::where(var_156, var_158, var_162);
                var_164 = wp::mul(var_154, var_163);
                // rx[j] = (                                                                  <L 302>
                wp::assign_inplace(var_4, var_134, var_164);
            }
            if (!var_135) {
                // rx[j] = (1.0 if j & 1 else -1.0) * (coord[1] if j & 2 else 1.0 - coord[1]) * (coord[2] if j & 4 else 1.0 - coord[2])       <L 308>
                var_166 = wp::bit_and(var_134, var_165);
                if (var_166) {
                }
                if (!var_166) {
                }
                var_170 = wp::where(var_166, var_167, var_169);
                var_172 = wp::bit_and(var_134, var_171);
                if (var_172) {
                    var_174 = wp::extract(var_76, var_173);
                }
                if (!var_172) {
                    var_177 = wp::extract(var_76, var_176);
                    var_178 = wp::sub(var_175, var_177);
                }
                var_179 = wp::where(var_172, var_174, var_178);
                var_180 = wp::mul(var_170, var_179);
                var_182 = wp::bit_and(var_134, var_181);
                if (var_182) {
                    var_184 = wp::extract(var_76, var_183);
                }
                if (!var_182) {
                    var_187 = wp::extract(var_76, var_186);
                    var_188 = wp::sub(var_185, var_187);
                }
                var_189 = wp::where(var_182, var_184, var_188);
                var_190 = wp::mul(var_180, var_189);
                wp::assign_inplace(var_4, var_134, var_190);
                // ry[j] = (coord[0] if j & 1 else 1.0 - coord[0]) * (1.0 if j & 2 else -1.0) * (coord[2] if j & 4 else 1.0 - coord[2])       <L 309>
                var_192 = wp::bit_and(var_134, var_191);
                if (var_192) {
                    var_194 = wp::extract(var_76, var_193);
                }
                if (!var_192) {
                    var_197 = wp::extract(var_76, var_196);
                    var_198 = wp::sub(var_195, var_197);
                }
                var_199 = wp::where(var_192, var_194, var_198);
                var_201 = wp::bit_and(var_134, var_200);
                if (var_201) {
                }
                if (!var_201) {
                }
                var_205 = wp::where(var_201, var_202, var_204);
                var_206 = wp::mul(var_199, var_205);
                var_208 = wp::bit_and(var_134, var_207);
                if (var_208) {
                    var_210 = wp::extract(var_76, var_209);
                }
                if (!var_208) {
                    var_213 = wp::extract(var_76, var_212);
                    var_214 = wp::sub(var_211, var_213);
                }
                var_215 = wp::where(var_208, var_210, var_214);
                var_216 = wp::mul(var_206, var_215);
                wp::assign_inplace(var_6, var_134, var_216);
                // rz[j] = (coord[0] if j & 1 else 1.0 - coord[0]) * (coord[1] if j & 2 else 1.0 - coord[1]) * (1.0 if j & 4 else -1.0)       <L 310>
                var_218 = wp::bit_and(var_134, var_217);
                if (var_218) {
                    var_220 = wp::extract(var_76, var_219);
                }
                if (!var_218) {
                    var_223 = wp::extract(var_76, var_222);
                    var_224 = wp::sub(var_221, var_223);
                }
                var_225 = wp::where(var_218, var_220, var_224);
                var_227 = wp::bit_and(var_134, var_226);
                if (var_227) {
                    var_229 = wp::extract(var_76, var_228);
                }
                if (!var_227) {
                    var_232 = wp::extract(var_76, var_231);
                    var_233 = wp::sub(var_230, var_232);
                }
                var_234 = wp::where(var_227, var_229, var_233);
                var_235 = wp::mul(var_225, var_234);
                var_237 = wp::bit_and(var_134, var_236);
                if (var_237) {
                }
                if (!var_237) {
                }
                var_241 = wp::where(var_237, var_238, var_240);
                var_242 = wp::mul(var_235, var_241);
                wp::assign_inplace(var_8, var_134, var_242);
            }
            // if not grad:                                                                   <L 301>
            var_244 = wp::unot(var_grad);
            if (var_244) {
                // rx[j] = (                                                                  <L 302>
                // (coord[0] if j & 1 else 1.0 - coord[0])                                    <L 303>
                var_246 = wp::bit_and(var_243, var_245);
                if (var_246) {
                    var_248 = wp::extract(var_76, var_247);
                }
                if (!var_246) {
                    var_251 = wp::extract(var_76, var_250);
                    var_252 = wp::sub(var_249, var_251);
                }
                var_253 = wp::where(var_246, var_248, var_252);
                // * (coord[1] if j & 2 else 1.0 - coord[1])                                  <L 304>
                var_255 = wp::bit_and(var_243, var_254);
                if (var_255) {
                    var_257 = wp::extract(var_76, var_256);
                }
                if (!var_255) {
                    var_260 = wp::extract(var_76, var_259);
                    var_261 = wp::sub(var_258, var_260);
                }
                var_262 = wp::where(var_255, var_257, var_261);
                var_263 = wp::mul(var_253, var_262);
                // * (coord[2] if j & 4 else 1.0 - coord[2])                                  <L 305>
                var_265 = wp::bit_and(var_243, var_264);
                if (var_265) {
                    var_267 = wp::extract(var_76, var_266);
                }
                if (!var_265) {
                    var_270 = wp::extract(var_76, var_269);
                    var_271 = wp::sub(var_268, var_270);
                }
                var_272 = wp::where(var_265, var_267, var_271);
                var_273 = wp::mul(var_263, var_272);
                // rx[j] = (                                                                  <L 302>
                wp::assign_inplace(var_4, var_243, var_273);
            }
            if (!var_244) {
                // rx[j] = (1.0 if j & 1 else -1.0) * (coord[1] if j & 2 else 1.0 - coord[1]) * (coord[2] if j & 4 else 1.0 - coord[2])       <L 308>
                var_275 = wp::bit_and(var_243, var_274);
                if (var_275) {
                }
                if (!var_275) {
                }
                var_279 = wp::where(var_275, var_276, var_278);
                var_281 = wp::bit_and(var_243, var_280);
                if (var_281) {
                    var_283 = wp::extract(var_76, var_282);
                }
                if (!var_281) {
                    var_286 = wp::extract(var_76, var_285);
                    var_287 = wp::sub(var_284, var_286);
                }
                var_288 = wp::where(var_281, var_283, var_287);
                var_289 = wp::mul(var_279, var_288);
                var_291 = wp::bit_and(var_243, var_290);
                if (var_291) {
                    var_293 = wp::extract(var_76, var_292);
                }
                if (!var_291) {
                    var_296 = wp::extract(var_76, var_295);
                    var_297 = wp::sub(var_294, var_296);
                }
                var_298 = wp::where(var_291, var_293, var_297);
                var_299 = wp::mul(var_289, var_298);
                wp::assign_inplace(var_4, var_243, var_299);
                // ry[j] = (coord[0] if j & 1 else 1.0 - coord[0]) * (1.0 if j & 2 else -1.0) * (coord[2] if j & 4 else 1.0 - coord[2])       <L 309>
                var_301 = wp::bit_and(var_243, var_300);
                if (var_301) {
                    var_303 = wp::extract(var_76, var_302);
                }
                if (!var_301) {
                    var_306 = wp::extract(var_76, var_305);
                    var_307 = wp::sub(var_304, var_306);
                }
                var_308 = wp::where(var_301, var_303, var_307);
                var_310 = wp::bit_and(var_243, var_309);
                if (var_310) {
                }
                if (!var_310) {
                }
                var_314 = wp::where(var_310, var_311, var_313);
                var_315 = wp::mul(var_308, var_314);
                var_317 = wp::bit_and(var_243, var_316);
                if (var_317) {
                    var_319 = wp::extract(var_76, var_318);
                }
                if (!var_317) {
                    var_322 = wp::extract(var_76, var_321);
                    var_323 = wp::sub(var_320, var_322);
                }
                var_324 = wp::where(var_317, var_319, var_323);
                var_325 = wp::mul(var_315, var_324);
                wp::assign_inplace(var_6, var_243, var_325);
                // rz[j] = (coord[0] if j & 1 else 1.0 - coord[0]) * (coord[1] if j & 2 else 1.0 - coord[1]) * (1.0 if j & 4 else -1.0)       <L 310>
                var_327 = wp::bit_and(var_243, var_326);
                if (var_327) {
                    var_329 = wp::extract(var_76, var_328);
                }
                if (!var_327) {
                    var_332 = wp::extract(var_76, var_331);
                    var_333 = wp::sub(var_330, var_332);
                }
                var_334 = wp::where(var_327, var_329, var_333);
                var_336 = wp::bit_and(var_243, var_335);
                if (var_336) {
                    var_338 = wp::extract(var_76, var_337);
                }
                if (!var_336) {
                    var_341 = wp::extract(var_76, var_340);
                    var_342 = wp::sub(var_339, var_341);
                }
                var_343 = wp::where(var_336, var_338, var_342);
                var_344 = wp::mul(var_334, var_343);
                var_346 = wp::bit_and(var_243, var_345);
                if (var_346) {
                }
                if (!var_346) {
                }
                var_350 = wp::where(var_346, var_347, var_349);
                var_351 = wp::mul(var_344, var_350);
                wp::assign_inplace(var_8, var_243, var_351);
            }
            // if not grad:                                                                   <L 301>
            var_353 = wp::unot(var_grad);
            if (var_353) {
                // rx[j] = (                                                                  <L 302>
                // (coord[0] if j & 1 else 1.0 - coord[0])                                    <L 303>
                var_355 = wp::bit_and(var_352, var_354);
                if (var_355) {
                    var_357 = wp::extract(var_76, var_356);
                }
                if (!var_355) {
                    var_360 = wp::extract(var_76, var_359);
                    var_361 = wp::sub(var_358, var_360);
                }
                var_362 = wp::where(var_355, var_357, var_361);
                // * (coord[1] if j & 2 else 1.0 - coord[1])                                  <L 304>
                var_364 = wp::bit_and(var_352, var_363);
                if (var_364) {
                    var_366 = wp::extract(var_76, var_365);
                }
                if (!var_364) {
                    var_369 = wp::extract(var_76, var_368);
                    var_370 = wp::sub(var_367, var_369);
                }
                var_371 = wp::where(var_364, var_366, var_370);
                var_372 = wp::mul(var_362, var_371);
                // * (coord[2] if j & 4 else 1.0 - coord[2])                                  <L 305>
                var_374 = wp::bit_and(var_352, var_373);
                if (var_374) {
                    var_376 = wp::extract(var_76, var_375);
                }
                if (!var_374) {
                    var_379 = wp::extract(var_76, var_378);
                    var_380 = wp::sub(var_377, var_379);
                }
                var_381 = wp::where(var_374, var_376, var_380);
                var_382 = wp::mul(var_372, var_381);
                // rx[j] = (                                                                  <L 302>
                wp::assign_inplace(var_4, var_352, var_382);
            }
            if (!var_353) {
                // rx[j] = (1.0 if j & 1 else -1.0) * (coord[1] if j & 2 else 1.0 - coord[1]) * (coord[2] if j & 4 else 1.0 - coord[2])       <L 308>
                var_384 = wp::bit_and(var_352, var_383);
                if (var_384) {
                }
                if (!var_384) {
                }
                var_388 = wp::where(var_384, var_385, var_387);
                var_390 = wp::bit_and(var_352, var_389);
                if (var_390) {
                    var_392 = wp::extract(var_76, var_391);
                }
                if (!var_390) {
                    var_395 = wp::extract(var_76, var_394);
                    var_396 = wp::sub(var_393, var_395);
                }
                var_397 = wp::where(var_390, var_392, var_396);
                var_398 = wp::mul(var_388, var_397);
                var_400 = wp::bit_and(var_352, var_399);
                if (var_400) {
                    var_402 = wp::extract(var_76, var_401);
                }
                if (!var_400) {
                    var_405 = wp::extract(var_76, var_404);
                    var_406 = wp::sub(var_403, var_405);
                }
                var_407 = wp::where(var_400, var_402, var_406);
                var_408 = wp::mul(var_398, var_407);
                wp::assign_inplace(var_4, var_352, var_408);
                // ry[j] = (coord[0] if j & 1 else 1.0 - coord[0]) * (1.0 if j & 2 else -1.0) * (coord[2] if j & 4 else 1.0 - coord[2])       <L 309>
                var_410 = wp::bit_and(var_352, var_409);
                if (var_410) {
                    var_412 = wp::extract(var_76, var_411);
                }
                if (!var_410) {
                    var_415 = wp::extract(var_76, var_414);
                    var_416 = wp::sub(var_413, var_415);
                }
                var_417 = wp::where(var_410, var_412, var_416);
                var_419 = wp::bit_and(var_352, var_418);
                if (var_419) {
                }
                if (!var_419) {
                }
                var_423 = wp::where(var_419, var_420, var_422);
                var_424 = wp::mul(var_417, var_423);
                var_426 = wp::bit_and(var_352, var_425);
                if (var_426) {
                    var_428 = wp::extract(var_76, var_427);
                }
                if (!var_426) {
                    var_431 = wp::extract(var_76, var_430);
                    var_432 = wp::sub(var_429, var_431);
                }
                var_433 = wp::where(var_426, var_428, var_432);
                var_434 = wp::mul(var_424, var_433);
                wp::assign_inplace(var_6, var_352, var_434);
                // rz[j] = (coord[0] if j & 1 else 1.0 - coord[0]) * (coord[1] if j & 2 else 1.0 - coord[1]) * (1.0 if j & 4 else -1.0)       <L 310>
                var_436 = wp::bit_and(var_352, var_435);
                if (var_436) {
                    var_438 = wp::extract(var_76, var_437);
                }
                if (!var_436) {
                    var_441 = wp::extract(var_76, var_440);
                    var_442 = wp::sub(var_439, var_441);
                }
                var_443 = wp::where(var_436, var_438, var_442);
                var_445 = wp::bit_and(var_352, var_444);
                if (var_445) {
                    var_447 = wp::extract(var_76, var_446);
                }
                if (!var_445) {
                    var_450 = wp::extract(var_76, var_449);
                    var_451 = wp::sub(var_448, var_450);
                }
                var_452 = wp::where(var_445, var_447, var_451);
                var_453 = wp::mul(var_443, var_452);
                var_455 = wp::bit_and(var_352, var_454);
                if (var_455) {
                }
                if (!var_455) {
                }
                var_459 = wp::where(var_455, var_456, var_458);
                var_460 = wp::mul(var_453, var_459);
                wp::assign_inplace(var_8, var_352, var_460);
            }
            // if not grad:                                                                   <L 301>
            var_462 = wp::unot(var_grad);
            if (var_462) {
                // rx[j] = (                                                                  <L 302>
                // (coord[0] if j & 1 else 1.0 - coord[0])                                    <L 303>
                var_464 = wp::bit_and(var_461, var_463);
                if (var_464) {
                    var_466 = wp::extract(var_76, var_465);
                }
                if (!var_464) {
                    var_469 = wp::extract(var_76, var_468);
                    var_470 = wp::sub(var_467, var_469);
                }
                var_471 = wp::where(var_464, var_466, var_470);
                // * (coord[1] if j & 2 else 1.0 - coord[1])                                  <L 304>
                var_473 = wp::bit_and(var_461, var_472);
                if (var_473) {
                    var_475 = wp::extract(var_76, var_474);
                }
                if (!var_473) {
                    var_478 = wp::extract(var_76, var_477);
                    var_479 = wp::sub(var_476, var_478);
                }
                var_480 = wp::where(var_473, var_475, var_479);
                var_481 = wp::mul(var_471, var_480);
                // * (coord[2] if j & 4 else 1.0 - coord[2])                                  <L 305>
                var_483 = wp::bit_and(var_461, var_482);
                if (var_483) {
                    var_485 = wp::extract(var_76, var_484);
                }
                if (!var_483) {
                    var_488 = wp::extract(var_76, var_487);
                    var_489 = wp::sub(var_486, var_488);
                }
                var_490 = wp::where(var_483, var_485, var_489);
                var_491 = wp::mul(var_481, var_490);
                // rx[j] = (                                                                  <L 302>
                wp::assign_inplace(var_4, var_461, var_491);
            }
            if (!var_462) {
                // rx[j] = (1.0 if j & 1 else -1.0) * (coord[1] if j & 2 else 1.0 - coord[1]) * (coord[2] if j & 4 else 1.0 - coord[2])       <L 308>
                var_493 = wp::bit_and(var_461, var_492);
                if (var_493) {
                }
                if (!var_493) {
                }
                var_497 = wp::where(var_493, var_494, var_496);
                var_499 = wp::bit_and(var_461, var_498);
                if (var_499) {
                    var_501 = wp::extract(var_76, var_500);
                }
                if (!var_499) {
                    var_504 = wp::extract(var_76, var_503);
                    var_505 = wp::sub(var_502, var_504);
                }
                var_506 = wp::where(var_499, var_501, var_505);
                var_507 = wp::mul(var_497, var_506);
                var_509 = wp::bit_and(var_461, var_508);
                if (var_509) {
                    var_511 = wp::extract(var_76, var_510);
                }
                if (!var_509) {
                    var_514 = wp::extract(var_76, var_513);
                    var_515 = wp::sub(var_512, var_514);
                }
                var_516 = wp::where(var_509, var_511, var_515);
                var_517 = wp::mul(var_507, var_516);
                wp::assign_inplace(var_4, var_461, var_517);
                // ry[j] = (coord[0] if j & 1 else 1.0 - coord[0]) * (1.0 if j & 2 else -1.0) * (coord[2] if j & 4 else 1.0 - coord[2])       <L 309>
                var_519 = wp::bit_and(var_461, var_518);
                if (var_519) {
                    var_521 = wp::extract(var_76, var_520);
                }
                if (!var_519) {
                    var_524 = wp::extract(var_76, var_523);
                    var_525 = wp::sub(var_522, var_524);
                }
                var_526 = wp::where(var_519, var_521, var_525);
                var_528 = wp::bit_and(var_461, var_527);
                if (var_528) {
                }
                if (!var_528) {
                }
                var_532 = wp::where(var_528, var_529, var_531);
                var_533 = wp::mul(var_526, var_532);
                var_535 = wp::bit_and(var_461, var_534);
                if (var_535) {
                    var_537 = wp::extract(var_76, var_536);
                }
                if (!var_535) {
                    var_540 = wp::extract(var_76, var_539);
                    var_541 = wp::sub(var_538, var_540);
                }
                var_542 = wp::where(var_535, var_537, var_541);
                var_543 = wp::mul(var_533, var_542);
                wp::assign_inplace(var_6, var_461, var_543);
                // rz[j] = (coord[0] if j & 1 else 1.0 - coord[0]) * (coord[1] if j & 2 else 1.0 - coord[1]) * (1.0 if j & 4 else -1.0)       <L 310>
                var_545 = wp::bit_and(var_461, var_544);
                if (var_545) {
                    var_547 = wp::extract(var_76, var_546);
                }
                if (!var_545) {
                    var_550 = wp::extract(var_76, var_549);
                    var_551 = wp::sub(var_548, var_550);
                }
                var_552 = wp::where(var_545, var_547, var_551);
                var_554 = wp::bit_and(var_461, var_553);
                if (var_554) {
                    var_556 = wp::extract(var_76, var_555);
                }
                if (!var_554) {
                    var_559 = wp::extract(var_76, var_558);
                    var_560 = wp::sub(var_557, var_559);
                }
                var_561 = wp::where(var_554, var_556, var_560);
                var_562 = wp::mul(var_552, var_561);
                var_564 = wp::bit_and(var_461, var_563);
                if (var_564) {
                }
                if (!var_564) {
                }
                var_568 = wp::where(var_564, var_565, var_567);
                var_569 = wp::mul(var_562, var_568);
                wp::assign_inplace(var_8, var_461, var_569);
            }
            // if not grad:                                                                   <L 301>
            var_571 = wp::unot(var_grad);
            if (var_571) {
                // rx[j] = (                                                                  <L 302>
                // (coord[0] if j & 1 else 1.0 - coord[0])                                    <L 303>
                var_573 = wp::bit_and(var_570, var_572);
                if (var_573) {
                    var_575 = wp::extract(var_76, var_574);
                }
                if (!var_573) {
                    var_578 = wp::extract(var_76, var_577);
                    var_579 = wp::sub(var_576, var_578);
                }
                var_580 = wp::where(var_573, var_575, var_579);
                // * (coord[1] if j & 2 else 1.0 - coord[1])                                  <L 304>
                var_582 = wp::bit_and(var_570, var_581);
                if (var_582) {
                    var_584 = wp::extract(var_76, var_583);
                }
                if (!var_582) {
                    var_587 = wp::extract(var_76, var_586);
                    var_588 = wp::sub(var_585, var_587);
                }
                var_589 = wp::where(var_582, var_584, var_588);
                var_590 = wp::mul(var_580, var_589);
                // * (coord[2] if j & 4 else 1.0 - coord[2])                                  <L 305>
                var_592 = wp::bit_and(var_570, var_591);
                if (var_592) {
                    var_594 = wp::extract(var_76, var_593);
                }
                if (!var_592) {
                    var_597 = wp::extract(var_76, var_596);
                    var_598 = wp::sub(var_595, var_597);
                }
                var_599 = wp::where(var_592, var_594, var_598);
                var_600 = wp::mul(var_590, var_599);
                // rx[j] = (                                                                  <L 302>
                wp::assign_inplace(var_4, var_570, var_600);
            }
            if (!var_571) {
                // rx[j] = (1.0 if j & 1 else -1.0) * (coord[1] if j & 2 else 1.0 - coord[1]) * (coord[2] if j & 4 else 1.0 - coord[2])       <L 308>
                var_602 = wp::bit_and(var_570, var_601);
                if (var_602) {
                }
                if (!var_602) {
                }
                var_606 = wp::where(var_602, var_603, var_605);
                var_608 = wp::bit_and(var_570, var_607);
                if (var_608) {
                    var_610 = wp::extract(var_76, var_609);
                }
                if (!var_608) {
                    var_613 = wp::extract(var_76, var_612);
                    var_614 = wp::sub(var_611, var_613);
                }
                var_615 = wp::where(var_608, var_610, var_614);
                var_616 = wp::mul(var_606, var_615);
                var_618 = wp::bit_and(var_570, var_617);
                if (var_618) {
                    var_620 = wp::extract(var_76, var_619);
                }
                if (!var_618) {
                    var_623 = wp::extract(var_76, var_622);
                    var_624 = wp::sub(var_621, var_623);
                }
                var_625 = wp::where(var_618, var_620, var_624);
                var_626 = wp::mul(var_616, var_625);
                wp::assign_inplace(var_4, var_570, var_626);
                // ry[j] = (coord[0] if j & 1 else 1.0 - coord[0]) * (1.0 if j & 2 else -1.0) * (coord[2] if j & 4 else 1.0 - coord[2])       <L 309>
                var_628 = wp::bit_and(var_570, var_627);
                if (var_628) {
                    var_630 = wp::extract(var_76, var_629);
                }
                if (!var_628) {
                    var_633 = wp::extract(var_76, var_632);
                    var_634 = wp::sub(var_631, var_633);
                }
                var_635 = wp::where(var_628, var_630, var_634);
                var_637 = wp::bit_and(var_570, var_636);
                if (var_637) {
                }
                if (!var_637) {
                }
                var_641 = wp::where(var_637, var_638, var_640);
                var_642 = wp::mul(var_635, var_641);
                var_644 = wp::bit_and(var_570, var_643);
                if (var_644) {
                    var_646 = wp::extract(var_76, var_645);
                }
                if (!var_644) {
                    var_649 = wp::extract(var_76, var_648);
                    var_650 = wp::sub(var_647, var_649);
                }
                var_651 = wp::where(var_644, var_646, var_650);
                var_652 = wp::mul(var_642, var_651);
                wp::assign_inplace(var_6, var_570, var_652);
                // rz[j] = (coord[0] if j & 1 else 1.0 - coord[0]) * (coord[1] if j & 2 else 1.0 - coord[1]) * (1.0 if j & 4 else -1.0)       <L 310>
                var_654 = wp::bit_and(var_570, var_653);
                if (var_654) {
                    var_656 = wp::extract(var_76, var_655);
                }
                if (!var_654) {
                    var_659 = wp::extract(var_76, var_658);
                    var_660 = wp::sub(var_657, var_659);
                }
                var_661 = wp::where(var_654, var_656, var_660);
                var_663 = wp::bit_and(var_570, var_662);
                if (var_663) {
                    var_665 = wp::extract(var_76, var_664);
                }
                if (!var_663) {
                    var_668 = wp::extract(var_76, var_667);
                    var_669 = wp::sub(var_666, var_668);
                }
                var_670 = wp::where(var_663, var_665, var_669);
                var_671 = wp::mul(var_661, var_670);
                var_673 = wp::bit_and(var_570, var_672);
                if (var_673) {
                }
                if (!var_673) {
                }
                var_677 = wp::where(var_673, var_674, var_676);
                var_678 = wp::mul(var_671, var_677);
                wp::assign_inplace(var_8, var_570, var_678);
            }
            // if not grad:                                                                   <L 301>
            var_680 = wp::unot(var_grad);
            if (var_680) {
                // rx[j] = (                                                                  <L 302>
                // (coord[0] if j & 1 else 1.0 - coord[0])                                    <L 303>
                var_682 = wp::bit_and(var_679, var_681);
                if (var_682) {
                    var_684 = wp::extract(var_76, var_683);
                }
                if (!var_682) {
                    var_687 = wp::extract(var_76, var_686);
                    var_688 = wp::sub(var_685, var_687);
                }
                var_689 = wp::where(var_682, var_684, var_688);
                // * (coord[1] if j & 2 else 1.0 - coord[1])                                  <L 304>
                var_691 = wp::bit_and(var_679, var_690);
                if (var_691) {
                    var_693 = wp::extract(var_76, var_692);
                }
                if (!var_691) {
                    var_696 = wp::extract(var_76, var_695);
                    var_697 = wp::sub(var_694, var_696);
                }
                var_698 = wp::where(var_691, var_693, var_697);
                var_699 = wp::mul(var_689, var_698);
                // * (coord[2] if j & 4 else 1.0 - coord[2])                                  <L 305>
                var_701 = wp::bit_and(var_679, var_700);
                if (var_701) {
                    var_703 = wp::extract(var_76, var_702);
                }
                if (!var_701) {
                    var_706 = wp::extract(var_76, var_705);
                    var_707 = wp::sub(var_704, var_706);
                }
                var_708 = wp::where(var_701, var_703, var_707);
                var_709 = wp::mul(var_699, var_708);
                // rx[j] = (                                                                  <L 302>
                wp::assign_inplace(var_4, var_679, var_709);
            }
            if (!var_680) {
                // rx[j] = (1.0 if j & 1 else -1.0) * (coord[1] if j & 2 else 1.0 - coord[1]) * (coord[2] if j & 4 else 1.0 - coord[2])       <L 308>
                var_711 = wp::bit_and(var_679, var_710);
                if (var_711) {
                }
                if (!var_711) {
                }
                var_715 = wp::where(var_711, var_712, var_714);
                var_717 = wp::bit_and(var_679, var_716);
                if (var_717) {
                    var_719 = wp::extract(var_76, var_718);
                }
                if (!var_717) {
                    var_722 = wp::extract(var_76, var_721);
                    var_723 = wp::sub(var_720, var_722);
                }
                var_724 = wp::where(var_717, var_719, var_723);
                var_725 = wp::mul(var_715, var_724);
                var_727 = wp::bit_and(var_679, var_726);
                if (var_727) {
                    var_729 = wp::extract(var_76, var_728);
                }
                if (!var_727) {
                    var_732 = wp::extract(var_76, var_731);
                    var_733 = wp::sub(var_730, var_732);
                }
                var_734 = wp::where(var_727, var_729, var_733);
                var_735 = wp::mul(var_725, var_734);
                wp::assign_inplace(var_4, var_679, var_735);
                // ry[j] = (coord[0] if j & 1 else 1.0 - coord[0]) * (1.0 if j & 2 else -1.0) * (coord[2] if j & 4 else 1.0 - coord[2])       <L 309>
                var_737 = wp::bit_and(var_679, var_736);
                if (var_737) {
                    var_739 = wp::extract(var_76, var_738);
                }
                if (!var_737) {
                    var_742 = wp::extract(var_76, var_741);
                    var_743 = wp::sub(var_740, var_742);
                }
                var_744 = wp::where(var_737, var_739, var_743);
                var_746 = wp::bit_and(var_679, var_745);
                if (var_746) {
                }
                if (!var_746) {
                }
                var_750 = wp::where(var_746, var_747, var_749);
                var_751 = wp::mul(var_744, var_750);
                var_753 = wp::bit_and(var_679, var_752);
                if (var_753) {
                    var_755 = wp::extract(var_76, var_754);
                }
                if (!var_753) {
                    var_758 = wp::extract(var_76, var_757);
                    var_759 = wp::sub(var_756, var_758);
                }
                var_760 = wp::where(var_753, var_755, var_759);
                var_761 = wp::mul(var_751, var_760);
                wp::assign_inplace(var_6, var_679, var_761);
                // rz[j] = (coord[0] if j & 1 else 1.0 - coord[0]) * (coord[1] if j & 2 else 1.0 - coord[1]) * (1.0 if j & 4 else -1.0)       <L 310>
                var_763 = wp::bit_and(var_679, var_762);
                if (var_763) {
                    var_765 = wp::extract(var_76, var_764);
                }
                if (!var_763) {
                    var_768 = wp::extract(var_76, var_767);
                    var_769 = wp::sub(var_766, var_768);
                }
                var_770 = wp::where(var_763, var_765, var_769);
                var_772 = wp::bit_and(var_679, var_771);
                if (var_772) {
                    var_774 = wp::extract(var_76, var_773);
                }
                if (!var_772) {
                    var_777 = wp::extract(var_76, var_776);
                    var_778 = wp::sub(var_775, var_777);
                }
                var_779 = wp::where(var_772, var_774, var_778);
                var_780 = wp::mul(var_770, var_779);
                var_782 = wp::bit_and(var_679, var_781);
                if (var_782) {
                }
                if (!var_782) {
                }
                var_786 = wp::where(var_782, var_783, var_785);
                var_787 = wp::mul(var_780, var_786);
                wp::assign_inplace(var_8, var_679, var_787);
            }
            // if not grad:                                                                   <L 301>
            var_789 = wp::unot(var_grad);
            if (var_789) {
                // rx[j] = (                                                                  <L 302>
                // (coord[0] if j & 1 else 1.0 - coord[0])                                    <L 303>
                var_791 = wp::bit_and(var_788, var_790);
                if (var_791) {
                    var_793 = wp::extract(var_76, var_792);
                }
                if (!var_791) {
                    var_796 = wp::extract(var_76, var_795);
                    var_797 = wp::sub(var_794, var_796);
                }
                var_798 = wp::where(var_791, var_793, var_797);
                // * (coord[1] if j & 2 else 1.0 - coord[1])                                  <L 304>
                var_800 = wp::bit_and(var_788, var_799);
                if (var_800) {
                    var_802 = wp::extract(var_76, var_801);
                }
                if (!var_800) {
                    var_805 = wp::extract(var_76, var_804);
                    var_806 = wp::sub(var_803, var_805);
                }
                var_807 = wp::where(var_800, var_802, var_806);
                var_808 = wp::mul(var_798, var_807);
                // * (coord[2] if j & 4 else 1.0 - coord[2])                                  <L 305>
                var_810 = wp::bit_and(var_788, var_809);
                if (var_810) {
                    var_812 = wp::extract(var_76, var_811);
                }
                if (!var_810) {
                    var_815 = wp::extract(var_76, var_814);
                    var_816 = wp::sub(var_813, var_815);
                }
                var_817 = wp::where(var_810, var_812, var_816);
                var_818 = wp::mul(var_808, var_817);
                // rx[j] = (                                                                  <L 302>
                wp::assign_inplace(var_4, var_788, var_818);
            }
            if (!var_789) {
                // rx[j] = (1.0 if j & 1 else -1.0) * (coord[1] if j & 2 else 1.0 - coord[1]) * (coord[2] if j & 4 else 1.0 - coord[2])       <L 308>
                var_820 = wp::bit_and(var_788, var_819);
                if (var_820) {
                }
                if (!var_820) {
                }
                var_824 = wp::where(var_820, var_821, var_823);
                var_826 = wp::bit_and(var_788, var_825);
                if (var_826) {
                    var_828 = wp::extract(var_76, var_827);
                }
                if (!var_826) {
                    var_831 = wp::extract(var_76, var_830);
                    var_832 = wp::sub(var_829, var_831);
                }
                var_833 = wp::where(var_826, var_828, var_832);
                var_834 = wp::mul(var_824, var_833);
                var_836 = wp::bit_and(var_788, var_835);
                if (var_836) {
                    var_838 = wp::extract(var_76, var_837);
                }
                if (!var_836) {
                    var_841 = wp::extract(var_76, var_840);
                    var_842 = wp::sub(var_839, var_841);
                }
                var_843 = wp::where(var_836, var_838, var_842);
                var_844 = wp::mul(var_834, var_843);
                wp::assign_inplace(var_4, var_788, var_844);
                // ry[j] = (coord[0] if j & 1 else 1.0 - coord[0]) * (1.0 if j & 2 else -1.0) * (coord[2] if j & 4 else 1.0 - coord[2])       <L 309>
                var_846 = wp::bit_and(var_788, var_845);
                if (var_846) {
                    var_848 = wp::extract(var_76, var_847);
                }
                if (!var_846) {
                    var_851 = wp::extract(var_76, var_850);
                    var_852 = wp::sub(var_849, var_851);
                }
                var_853 = wp::where(var_846, var_848, var_852);
                var_855 = wp::bit_and(var_788, var_854);
                if (var_855) {
                }
                if (!var_855) {
                }
                var_859 = wp::where(var_855, var_856, var_858);
                var_860 = wp::mul(var_853, var_859);
                var_862 = wp::bit_and(var_788, var_861);
                if (var_862) {
                    var_864 = wp::extract(var_76, var_863);
                }
                if (!var_862) {
                    var_867 = wp::extract(var_76, var_866);
                    var_868 = wp::sub(var_865, var_867);
                }
                var_869 = wp::where(var_862, var_864, var_868);
                var_870 = wp::mul(var_860, var_869);
                wp::assign_inplace(var_6, var_788, var_870);
                // rz[j] = (coord[0] if j & 1 else 1.0 - coord[0]) * (coord[1] if j & 2 else 1.0 - coord[1]) * (1.0 if j & 4 else -1.0)       <L 310>
                var_872 = wp::bit_and(var_788, var_871);
                if (var_872) {
                    var_874 = wp::extract(var_76, var_873);
                }
                if (!var_872) {
                    var_877 = wp::extract(var_76, var_876);
                    var_878 = wp::sub(var_875, var_877);
                }
                var_879 = wp::where(var_872, var_874, var_878);
                var_881 = wp::bit_and(var_788, var_880);
                if (var_881) {
                    var_883 = wp::extract(var_76, var_882);
                }
                if (!var_881) {
                    var_886 = wp::extract(var_76, var_885);
                    var_887 = wp::sub(var_884, var_886);
                }
                var_888 = wp::where(var_881, var_883, var_887);
                var_889 = wp::mul(var_879, var_888);
                var_891 = wp::bit_and(var_788, var_890);
                if (var_891) {
                }
                if (!var_891) {
                }
                var_895 = wp::where(var_891, var_892, var_894);
                var_896 = wp::mul(var_889, var_895);
                wp::assign_inplace(var_8, var_788, var_896);
            }
            // if not grad:                                                                   <L 301>
            var_898 = wp::unot(var_grad);
            if (var_898) {
                // rx[j] = (                                                                  <L 302>
                // (coord[0] if j & 1 else 1.0 - coord[0])                                    <L 303>
                var_900 = wp::bit_and(var_897, var_899);
                if (var_900) {
                    var_902 = wp::extract(var_76, var_901);
                }
                if (!var_900) {
                    var_905 = wp::extract(var_76, var_904);
                    var_906 = wp::sub(var_903, var_905);
                }
                var_907 = wp::where(var_900, var_902, var_906);
                // * (coord[1] if j & 2 else 1.0 - coord[1])                                  <L 304>
                var_909 = wp::bit_and(var_897, var_908);
                if (var_909) {
                    var_911 = wp::extract(var_76, var_910);
                }
                if (!var_909) {
                    var_914 = wp::extract(var_76, var_913);
                    var_915 = wp::sub(var_912, var_914);
                }
                var_916 = wp::where(var_909, var_911, var_915);
                var_917 = wp::mul(var_907, var_916);
                // * (coord[2] if j & 4 else 1.0 - coord[2])                                  <L 305>
                var_919 = wp::bit_and(var_897, var_918);
                if (var_919) {
                    var_921 = wp::extract(var_76, var_920);
                }
                if (!var_919) {
                    var_924 = wp::extract(var_76, var_923);
                    var_925 = wp::sub(var_922, var_924);
                }
                var_926 = wp::where(var_919, var_921, var_925);
                var_927 = wp::mul(var_917, var_926);
                // rx[j] = (                                                                  <L 302>
                wp::assign_inplace(var_4, var_897, var_927);
            }
            if (!var_898) {
                // rx[j] = (1.0 if j & 1 else -1.0) * (coord[1] if j & 2 else 1.0 - coord[1]) * (coord[2] if j & 4 else 1.0 - coord[2])       <L 308>
                var_929 = wp::bit_and(var_897, var_928);
                if (var_929) {
                }
                if (!var_929) {
                }
                var_933 = wp::where(var_929, var_930, var_932);
                var_935 = wp::bit_and(var_897, var_934);
                if (var_935) {
                    var_937 = wp::extract(var_76, var_936);
                }
                if (!var_935) {
                    var_940 = wp::extract(var_76, var_939);
                    var_941 = wp::sub(var_938, var_940);
                }
                var_942 = wp::where(var_935, var_937, var_941);
                var_943 = wp::mul(var_933, var_942);
                var_945 = wp::bit_and(var_897, var_944);
                if (var_945) {
                    var_947 = wp::extract(var_76, var_946);
                }
                if (!var_945) {
                    var_950 = wp::extract(var_76, var_949);
                    var_951 = wp::sub(var_948, var_950);
                }
                var_952 = wp::where(var_945, var_947, var_951);
                var_953 = wp::mul(var_943, var_952);
                wp::assign_inplace(var_4, var_897, var_953);
                // ry[j] = (coord[0] if j & 1 else 1.0 - coord[0]) * (1.0 if j & 2 else -1.0) * (coord[2] if j & 4 else 1.0 - coord[2])       <L 309>
                var_955 = wp::bit_and(var_897, var_954);
                if (var_955) {
                    var_957 = wp::extract(var_76, var_956);
                }
                if (!var_955) {
                    var_960 = wp::extract(var_76, var_959);
                    var_961 = wp::sub(var_958, var_960);
                }
                var_962 = wp::where(var_955, var_957, var_961);
                var_964 = wp::bit_and(var_897, var_963);
                if (var_964) {
                }
                if (!var_964) {
                }
                var_968 = wp::where(var_964, var_965, var_967);
                var_969 = wp::mul(var_962, var_968);
                var_971 = wp::bit_and(var_897, var_970);
                if (var_971) {
                    var_973 = wp::extract(var_76, var_972);
                }
                if (!var_971) {
                    var_976 = wp::extract(var_76, var_975);
                    var_977 = wp::sub(var_974, var_976);
                }
                var_978 = wp::where(var_971, var_973, var_977);
                var_979 = wp::mul(var_969, var_978);
                wp::assign_inplace(var_6, var_897, var_979);
                // rz[j] = (coord[0] if j & 1 else 1.0 - coord[0]) * (coord[1] if j & 2 else 1.0 - coord[1]) * (1.0 if j & 4 else -1.0)       <L 310>
                var_981 = wp::bit_and(var_897, var_980);
                if (var_981) {
                    var_983 = wp::extract(var_76, var_982);
                }
                if (!var_981) {
                    var_986 = wp::extract(var_76, var_985);
                    var_987 = wp::sub(var_984, var_986);
                }
                var_988 = wp::where(var_981, var_983, var_987);
                var_990 = wp::bit_and(var_897, var_989);
                if (var_990) {
                    var_992 = wp::extract(var_76, var_991);
                }
                if (!var_990) {
                    var_995 = wp::extract(var_76, var_994);
                    var_996 = wp::sub(var_993, var_995);
                }
                var_997 = wp::where(var_990, var_992, var_996);
                var_998 = wp::mul(var_988, var_997);
                var_1000 = wp::bit_and(var_897, var_999);
                if (var_1000) {
                }
                if (!var_1000) {
                }
                var_1004 = wp::where(var_1000, var_1001, var_1003);
                var_1005 = wp::mul(var_998, var_1004);
                wp::assign_inplace(var_8, var_897, var_1005);
            }
            // return node, (rx, ry, rz)                                                      <L 311>
            var_1006 = wp::tuple(var_4, var_6, var_8);
            ret_0 = var_14;
            ret_1 = var_1006;
            return;
        }
        // x = 0 if coord[0] < 0.5 else 1                                                     <L 315>
        var_1008 = wp::extract(var_76, var_1007);
        var_1010 = (var_1008 < var_1009);
        if (var_1010) {
        }
        if (!var_1010) {
        }
        var_1013 = wp::where(var_1010, var_1011, var_1012);
        // y = 0 if coord[1] < 0.5 else 1                                                     <L 316>
        var_1015 = wp::extract(var_76, var_1014);
        var_1017 = (var_1015 < var_1016);
        if (var_1017) {
        }
        if (!var_1017) {
        }
        var_1020 = wp::where(var_1017, var_1018, var_1019);
        // z = 0 if coord[2] < 0.5 else 1                                                     <L 317>
        var_1022 = wp::extract(var_76, var_1021);
        var_1024 = (var_1022 < var_1023);
        if (var_1024) {
        }
        if (!var_1024) {
        }
        var_1027 = wp::where(var_1024, var_1025, var_1026);
        // child = oct_child[node][4 * z + 2 * y + x]                                         <L 318>
        var_1028 = wp::address(var_oct_child, var_14);
        var_1030 = wp::mul(var_1029, var_1027);
        var_1032 = wp::mul(var_1031, var_1020);
        var_1033 = wp::add(var_1030, var_1032);
        var_1034 = wp::add(var_1033, var_1013);
        var_1036 = wp::load(var_1028);
        var_1035 = wp::extract(var_1036, var_1034);
        // stack = child + root if child != -1 else -1                                        <L 319>
        var_1039 = (var_1035 != var_1038);
        if (var_1039) {
            var_1040 = wp::add(var_1035, var_root);
        }
        if (!var_1039) {
        }
        var_1043 = wp::where(var_1039, var_1040, var_1042);
        wp::assign(var_0, var_1043);
        wp::assign(var_2, var_73);
    goto start_while_0;
    end_while_0:;
    // wp.print("ERROR: Node not found\n")                                                    <L 321>
    wp::print(var_1044);
    // return -1, (rx, ry, rz)                                                                <L 322>
    var_1047 = wp::tuple(var_4, var_6, var_8);
    ret_0 = var_1046;
    ret_1 = var_1047;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_sdf.py:362
static CUDA_CALLABLE wp::float32 sample_volume_sdf_0(
    wp::vec_t<3, wp::float32> var_xyz,
    VolumeData_53ac1a2d var_volume_data)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32>* var_0;
    wp::vec_t<3, wp::float32>* var_1;
    wp::float32 var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::array_t<wp::vec_t<8, wp::int32>>* var_6;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_7;
    const bool var_8 = false;
    wp::int32* var_9;
    wp::int32 var_10;
    wp::tuple_t<wp::vec_t<8, wp::float32>, wp::vec_t<8, wp::float32>, wp::vec_t<8, wp::float32>> var_11;
    wp::array_t<wp::vec_t<8, wp::int32>> var_12;
    wp::array_t<wp::vec_t<3, wp::float32>> var_13;
    wp::int32 var_14;
    const wp::int32 var_15 = 0;
    wp::vec_t<8, wp::float32> var_16;
    wp::array_t<wp::vec_t<8, wp::float32>>* var_17;
    wp::vec_t<8, wp::float32>* var_18;
    wp::array_t<wp::vec_t<8, wp::float32>> var_19;
    wp::float32 var_20;
    wp::vec_t<8, wp::float32> var_21;
    wp::float32 var_22;
    //---------
    // forward
    // def sample_volume_sdf(xyz: wp.vec3, volume_data: VolumeData) -> float:                 <L 363>
    // dist0, point = box_project(volume_data.center, volume_data.half_size, xyz)             <L 364>
    var_0 = &(var_volume_data.center);
    var_1 = &(var_volume_data.half_size);
    var_4 = wp::load(var_0);
    var_5 = wp::load(var_1);
    box_project_0(var_4, var_5, var_xyz, var_2, var_3);
    // node, weights = find_oct(volume_data.oct_child, volume_data.oct_aabb, point, grad=False, root=volume_data.root)       <L 365>
    var_6 = &(var_volume_data.oct_child);
    var_7 = &(var_volume_data.oct_aabb);
    var_9 = &(var_volume_data.root);
    var_12 = wp::load(var_6);
    var_13 = wp::load(var_7);
    var_14 = wp::load(var_9);
    find_oct_0(var_12, var_13, var_3, var_8, var_14, var_10, var_11);
    // return dist0 + wp.dot(weights[0], volume_data.oct_coeff[node])                         <L 366>
    var_16 = wp::extract<0>(var_11);
    var_17 = &(var_volume_data.oct_coeff);
    var_19 = wp::load(var_17);
    var_18 = wp::address(var_19, var_10);
    var_21 = wp::load(var_18);
    var_20 = wp::dot(var_16, var_21);
    var_22 = wp::add(var_2, var_20);
    return var_22;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_sdf.py:233
static CUDA_CALLABLE wp::float32 user_sdf_0(
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<128, wp::float32> var_attr,
    wp::int32 var_sdf_type)
{
    //---------
    // primal vars
    const wp::str var_0 = "ERROR: user_sdf function must be implemented by user code\n";
    const wp::float32 var_1 = 0.0;
    //---------
    // forward
    // def user_sdf(p: wp.vec3, attr: vec_pluginattr, sdf_type: int) -> float:                <L 234>
    // wp.printf("ERROR: user_sdf function must be implemented by user code\n")               <L 239>
    printf(var_0);
    // return 0.0                                                                             <L 240>
    return var_1;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_sdf.py:389
static CUDA_CALLABLE wp::float32 sdf_0(
    wp::int32 var_type,
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<128, wp::float32> var_attr,
    wp::int32 var_sdf_type,
    VolumeData_53ac1a2d var_volume_data,
    MeshData_52eaa0fa var_mesh_data)
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
    const wp::int32 var_7 = 0;
    bool var_8;
    const wp::int32 var_9 = 2;
    wp::float32 var_10;
    const wp::int32 var_11 = 2;
    bool var_12;
    wp::float32 var_13;
    const wp::int32 var_14 = 6;
    bool var_15;
    wp::float32 var_16;
    const wp::int32 var_17 = 4;
    bool var_18;
    wp::float32 var_19;
    const wp::int32 var_20 = 7;
    bool var_21;
    bool* var_22;
    bool var_23;
    bool var_24;
    wp::vec_t<3, wp::float32>* var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32>* var_28;
    wp::int32* var_29;
    wp::array_t<wp::int32>* var_30;
    wp::array_t<wp::int32>* var_31;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_32;
    wp::array_t<wp::vec_t<3, wp::int32>>* var_33;
    wp::int32* var_34;
    wp::vec_t<3, wp::float32>* var_35;
    wp::mat_t<3, 3, wp::float32>* var_36;
    wp::vec_t<3, wp::float32>* var_37;
    wp::vec_t<3, wp::float32>* var_38;
    wp::vec_t<3, wp::float32>* var_39;
    wp::float32 var_40;
    wp::vec_t<3, wp::float32> var_41;
    wp::int32 var_42;
    wp::array_t<wp::int32> var_43;
    wp::array_t<wp::int32> var_44;
    wp::array_t<wp::vec_t<3, wp::float32>> var_45;
    wp::array_t<wp::vec_t<3, wp::int32>> var_46;
    wp::int32 var_47;
    wp::vec_t<3, wp::float32> var_48;
    wp::mat_t<3, 3, wp::float32> var_49;
    wp::vec_t<3, wp::float32> var_50;
    wp::vec_t<3, wp::float32> var_51;
    wp::vec_t<3, wp::float32> var_52;
    wp::float32 var_53;
    bool var_54;
    wp::int32* var_55;
    wp::array_t<wp::int32>* var_56;
    wp::array_t<wp::int32>* var_57;
    wp::array_t<wp::vec_t<3, wp::float32>>* var_58;
    wp::array_t<wp::vec_t<3, wp::int32>>* var_59;
    wp::int32* var_60;
    wp::vec_t<3, wp::float32>* var_61;
    wp::mat_t<3, 3, wp::float32>* var_62;
    wp::vec_t<3, wp::float32>* var_63;
    wp::vec_t<3, wp::float32>* var_64;
    wp::vec_t<3, wp::float32>* var_65;
    wp::vec_t<3, wp::float32> var_66;
    wp::vec_t<3, wp::float32> var_67;
    wp::float32 var_68;
    wp::vec_t<3, wp::float32> var_69;
    wp::int32 var_70;
    wp::array_t<wp::int32> var_71;
    wp::array_t<wp::int32> var_72;
    wp::array_t<wp::vec_t<3, wp::float32>> var_73;
    wp::array_t<wp::vec_t<3, wp::int32>> var_74;
    wp::int32 var_75;
    wp::vec_t<3, wp::float32> var_76;
    wp::mat_t<3, 3, wp::float32> var_77;
    wp::vec_t<3, wp::float32> var_78;
    wp::vec_t<3, wp::float32> var_79;
    wp::float32 var_80;
    wp::float32 var_81;
    wp::vec_t<3, wp::float32> var_82;
    const wp::int32 var_83 = 8;
    bool var_84;
    const wp::int32 var_85 = 1;
    const wp::int32 var_86 = -1;
    bool var_87;
    wp::float32 var_88;
    wp::float32 var_89;
    const wp::int32 var_90 = 7;
    bool var_91;
    bool* var_92;
    bool var_93;
    bool var_94;
    wp::float32 var_95;
    const wp::str var_96 = "ERROR: SDF type not implemented\n";
    const wp::float32 var_97 = 0.0;
    //---------
    // forward
    // def sdf(type: int, p: wp.vec3, attr: vec_pluginattr, sdf_type: int, volume_data: VolumeData, mesh_data: MeshData) -> float:       <L 390>
    // attr_vec3 = wp.vec3(attr[0], attr[1], attr[2])                                         <L 392>
    var_1 = wp::extract(var_attr, var_0);
    var_3 = wp::extract(var_attr, var_2);
    var_5 = wp::extract(var_attr, var_4);
    var_6 = wp::vec_t<3, wp::float32>(var_1, var_3, var_5);
    // if type == GeomType.PLANE:                                                             <L 393>
    var_8 = (var_type == var_7);
    if (var_8) {
        // return p[2]                                                                        <L 394>
        var_10 = wp::extract(var_p, var_9);
        return var_10;
    }
    if (!var_8) {
        // elif type == GeomType.SPHERE:                                                      <L 395>
        var_12 = (var_type == var_11);
        if (var_12) {
            // return sphere(p, attr_vec3)                                                    <L 396>
            var_13 = sphere_0(var_p, var_6);
            return var_13;
        }
        if (!var_12) {
            // elif type == GeomType.BOX:                                                     <L 397>
            var_15 = (var_type == var_14);
            if (var_15) {
                // return box(p, attr_vec3)                                                   <L 398>
                var_16 = box_0(var_p, var_6);
                return var_16;
            }
            if (!var_15) {
                // elif type == GeomType.ELLIPSOID:                                           <L 399>
                var_18 = (var_type == var_17);
                if (var_18) {
                    // return ellipsoid(p, attr_vec3)                                         <L 400>
                    var_19 = ellipsoid_0(var_p, var_6);
                    return var_19;
                }
                if (!var_18) {
                    // elif type == GeomType.MESH and mesh_data.valid:                        <L 401>
                    var_21 = (var_type == var_20);
                    var_22 = &(var_mesh_data.valid);
                    var_23 = wp::load(var_22);
                    var_24 = var_21 && var_23;
                    if (var_24) {
                        // mesh_data.pnt = p                                                  <L 402>
                        var_25 = &(var_mesh_data.pnt);
                        wp::store(var_25, var_p);
                        // mesh_data.vec = -wp.normalize(p)                                   <L 403>
                        var_26 = wp::normalize(var_p);
                        var_27 = wp::neg(var_26);
                        var_28 = &(var_mesh_data.vec);
                        wp::store(var_28, var_27);
                        // dist, normal = ray_mesh(                                           <L 404>
                        // mesh_data.nmeshface,                                               <L 405>
                        var_29 = &(var_mesh_data.nmeshface);
                        // mesh_data.mesh_vertadr,                                            <L 406>
                        var_30 = &(var_mesh_data.mesh_vertadr);
                        // mesh_data.mesh_faceadr,                                            <L 407>
                        var_31 = &(var_mesh_data.mesh_faceadr);
                        // mesh_data.mesh_vert,                                               <L 408>
                        var_32 = &(var_mesh_data.mesh_vert);
                        // mesh_data.mesh_face,                                               <L 409>
                        var_33 = &(var_mesh_data.mesh_face);
                        // mesh_data.data_id,                                                 <L 410>
                        var_34 = &(var_mesh_data.data_id);
                        // mesh_data.pos,                                                     <L 411>
                        var_35 = &(var_mesh_data.pos);
                        // mesh_data.mat,                                                     <L 412>
                        var_36 = &(var_mesh_data.mat);
                        // mesh_data.size,                                                    <L 413>
                        var_37 = &(var_mesh_data.size);
                        // mesh_data.pnt,                                                     <L 414>
                        var_38 = &(var_mesh_data.pnt);
                        // mesh_data.vec,                                                     <L 415>
                        var_39 = &(var_mesh_data.vec);
                        var_42 = wp::load(var_29);
                        var_43 = wp::load(var_30);
                        var_44 = wp::load(var_31);
                        var_45 = wp::load(var_32);
                        var_46 = wp::load(var_33);
                        var_47 = wp::load(var_34);
                        var_48 = wp::load(var_35);
                        var_49 = wp::load(var_36);
                        var_50 = wp::load(var_37);
                        var_51 = wp::load(var_38);
                        var_52 = wp::load(var_39);
                        ray_mesh_0(var_42, var_43, var_44, var_45, var_46, var_47, var_48, var_49, var_50, var_51, var_52, var_40, var_41);
                        // if dist > wp.norm_l2(p):                                           <L 417>
                        var_53 = norm_l2_0(var_p);
                        var_54 = (var_40 > var_53);
                        if (var_54) {
                            // dist, normal = ray_mesh(                                       <L 418>
                            // mesh_data.nmeshface,                                           <L 419>
                            var_55 = &(var_mesh_data.nmeshface);
                            // mesh_data.mesh_vertadr,                                        <L 420>
                            var_56 = &(var_mesh_data.mesh_vertadr);
                            // mesh_data.mesh_faceadr,                                        <L 421>
                            var_57 = &(var_mesh_data.mesh_faceadr);
                            // mesh_data.mesh_vert,                                           <L 422>
                            var_58 = &(var_mesh_data.mesh_vert);
                            // mesh_data.mesh_face,                                           <L 423>
                            var_59 = &(var_mesh_data.mesh_face);
                            // mesh_data.data_id,                                             <L 424>
                            var_60 = &(var_mesh_data.data_id);
                            // mesh_data.pos,                                                 <L 425>
                            var_61 = &(var_mesh_data.pos);
                            // mesh_data.mat,                                                 <L 426>
                            var_62 = &(var_mesh_data.mat);
                            // mesh_data.size,                                                <L 427>
                            var_63 = &(var_mesh_data.size);
                            // mesh_data.pnt,                                                 <L 428>
                            var_64 = &(var_mesh_data.pnt);
                            // -mesh_data.vec,                                                <L 429>
                            var_65 = &(var_mesh_data.vec);
                            var_67 = wp::load(var_65);
                            var_66 = wp::neg(var_67);
                            var_70 = wp::load(var_55);
                            var_71 = wp::load(var_56);
                            var_72 = wp::load(var_57);
                            var_73 = wp::load(var_58);
                            var_74 = wp::load(var_59);
                            var_75 = wp::load(var_60);
                            var_76 = wp::load(var_61);
                            var_77 = wp::load(var_62);
                            var_78 = wp::load(var_63);
                            var_79 = wp::load(var_64);
                            ray_mesh_0(var_70, var_71, var_72, var_73, var_74, var_75, var_76, var_77, var_78, var_79, var_66, var_68, var_69);
                            // return -dist                                                   <L 431>
                            var_80 = wp::neg(var_68);
                            return var_80;
                        }
                        var_81 = wp::where(var_54, var_68, var_40);
                        var_82 = wp::where(var_54, var_69, var_41);
                        // return dist                                                        <L 432>
                        return var_81;
                    }
                    if (!var_24) {
                        // elif type == GeomType.SDF:                                         <L 433>
                        var_84 = (var_type == var_83);
                        if (var_84) {
                            // if sdf_type == -1:                                             <L 434>
                            var_87 = (var_sdf_type == var_86);
                            if (var_87) {
                                // return sample_volume_sdf(p, volume_data)                   <L 435>
                                var_88 = sample_volume_sdf_0(var_p, var_volume_data);
                                return var_88;
                            }
                            if (!var_87) {
                                // return user_sdf(p, attr, sdf_type)                         <L 437>
                                var_89 = user_sdf_0(var_p, var_attr, var_sdf_type);
                                return var_89;
                            }
                        }
                        if (!var_84) {
                            // elif type == GeomType.MESH and volume_data.valid:              <L 438>
                            var_91 = (var_type == var_90);
                            var_92 = &(var_volume_data.valid);
                            var_93 = wp::load(var_92);
                            var_94 = var_91 && var_93;
                            if (var_94) {
                                // return sample_volume_sdf(p, volume_data)                   <L 439>
                                var_95 = sample_volume_sdf_0(var_p, var_volume_data);
                                return var_95;
                            }
                        }
                    }
                }
            }
        }
    }
    // wp.printf("ERROR: SDF type not implemented\n")                                         <L 440>
    printf(var_96);
    // return 0.0                                                                             <L 441>
    return var_97;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:2079
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _transform_spatial_0(
    wp::vec_t<6, wp::float32> var_vec,
    wp::vec_t<3, wp::float32> var_dif)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    //---------
    // forward
    // def _transform_spatial(vec: wp.spatial_vector, dif: wp.vec3) -> wp.vec3:               <L 2080>
    // return wp.spatial_bottom(vec) - wp.cross(dif, wp.spatial_top(vec))                     <L 2081>
    var_0 = wp::spatial_bottom(var_vec);
    var_1 = wp::spatial_top(var_vec);
    var_2 = wp::cross(var_dif, var_1);
    var_3 = wp::sub(var_0, var_2);
    return var_3;
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/ray.py:187
static CUDA_CALLABLE void ray_plane_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    const wp::int32 var_2 = 2;
    wp::float32 var_3;
    const wp::float32 var_4 = 1e-15;
    const wp::float32 var_5 = -1e-15;
    bool var_6;
    const wp::float32 var_7 = 1.0;
    const wp::float32 var_8 = -1.0;
    wp::vec_t<3, wp::float32> var_9;
    const wp::int32 var_10 = 2;
    wp::float32 var_11;
    wp::float32 var_12;
    const wp::int32 var_13 = 2;
    wp::float32 var_14;
    wp::float32 var_15;
    const wp::float32 var_16 = 0.0;
    bool var_17;
    const wp::float32 var_18 = 1.0;
    const wp::float32 var_19 = -1.0;
    wp::vec_t<3, wp::float32> var_20;
    const wp::int32 var_21 = 0;
    wp::float32 var_22;
    const wp::int32 var_23 = 0;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    const wp::int32 var_27 = 1;
    wp::float32 var_28;
    const wp::int32 var_29 = 1;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::vec_t<2, wp::float32> var_33;
    const wp::int32 var_34 = 0;
    wp::float32 var_35;
    const wp::float32 var_36 = 0.0;
    bool var_37;
    const wp::int32 var_38 = 0;
    wp::float32 var_39;
    wp::float32 var_40;
    const wp::int32 var_41 = 0;
    wp::float32 var_42;
    bool var_43;
    bool var_44;
    const wp::int32 var_45 = 1;
    wp::float32 var_46;
    const wp::float32 var_47 = 0.0;
    bool var_48;
    const wp::int32 var_49 = 1;
    wp::float32 var_50;
    wp::float32 var_51;
    const wp::int32 var_52 = 1;
    wp::float32 var_53;
    bool var_54;
    bool var_55;
    bool var_56;
    const wp::int32 var_57 = 0;
    const wp::int32 var_58 = 2;
    wp::float32 var_59;
    const wp::int32 var_60 = 1;
    const wp::int32 var_61 = 2;
    wp::float32 var_62;
    const wp::int32 var_63 = 2;
    const wp::int32 var_64 = 2;
    wp::float32 var_65;
    wp::vec_t<3, wp::float32> var_66;
    const wp::float32 var_67 = 1.0;
    const wp::float32 var_68 = -1.0;
    wp::vec_t<3, wp::float32> var_69;
    //---------
    // forward
    // def ray_plane(pos: wp.vec3, mat: wp.mat33, size: wp.vec3, pnt: wp.vec3, vec: wp.vec3) -> Tuple[float, wp.vec3]:       <L 188>
    // lpnt, lvec = _ray_map(pos, mat, pnt, vec)                                              <L 191>
    _ray_map_0(var_pos, var_mat, var_pnt, var_vec, var_0, var_1);
    // if lvec[2] > -MJ_MINVAL:                                                               <L 194>
    var_3 = wp::extract(var_1, var_2);
    var_6 = (var_3 > var_5);
    if (var_6) {
        // return -1.0, wp.vec3()                                                             <L 195>
        var_9 = wp::vec_t<3, wp::float32>();
        ret_0 = var_8;
        ret_1 = var_9;
        return;
    }
    // x = -lpnt[2] / lvec[2]                                                                 <L 198>
    var_11 = wp::extract(var_0, var_10);
    var_12 = wp::neg(var_11);
    var_14 = wp::extract(var_1, var_13);
    var_15 = wp::div(var_12, var_14);
    // if x < 0.0:                                                                            <L 199>
    var_17 = (var_15 < var_16);
    if (var_17) {
        // return -1.0, wp.vec3()                                                             <L 200>
        var_20 = wp::vec_t<3, wp::float32>();
        ret_0 = var_19;
        ret_1 = var_20;
        return;
    }
    // p = wp.vec2(lpnt[0] + x * lvec[0], lpnt[1] + x * lvec[1])                              <L 202>
    var_22 = wp::extract(var_0, var_21);
    var_24 = wp::extract(var_1, var_23);
    var_25 = wp::mul(var_15, var_24);
    var_26 = wp::add(var_22, var_25);
    var_28 = wp::extract(var_0, var_27);
    var_30 = wp::extract(var_1, var_29);
    var_31 = wp::mul(var_15, var_30);
    var_32 = wp::add(var_28, var_31);
    var_33 = wp::vec_t<2, wp::float32>(var_26, var_32);
    // if (size[0] <= 0.0 or wp.abs(p[0]) <= size[0]) and (size[1] <= 0.0 or wp.abs(p[1]) <= size[1]):       <L 205>
    var_35 = wp::extract(var_size, var_34);
    var_37 = (var_35 <= var_36);
    var_39 = wp::extract(var_33, var_38);
    var_40 = wp::abs(var_39);
    var_42 = wp::extract(var_size, var_41);
    var_43 = (var_40 <= var_42);
    var_44 = var_37 || var_43;
    var_46 = wp::extract(var_size, var_45);
    var_48 = (var_46 <= var_47);
    var_50 = wp::extract(var_33, var_49);
    var_51 = wp::abs(var_50);
    var_53 = wp::extract(var_size, var_52);
    var_54 = (var_51 <= var_53);
    var_55 = var_48 || var_54;
    var_56 = var_44 && var_55;
    if (var_56) {
        // return x, wp.vec3(mat[0, 2], mat[1, 2], mat[2, 2])                                 <L 206>
        var_59 = wp::extract(var_mat, var_57, var_58);
        var_62 = wp::extract(var_mat, var_60, var_61);
        var_65 = wp::extract(var_mat, var_63, var_64);
        var_66 = wp::vec_t<3, wp::float32>(var_59, var_62, var_65);
        ret_0 = var_15;
        ret_1 = var_66;
        return;
    }
    if (!var_56) {
        // return -1.0, wp.vec3()                                                             <L 208>
        var_69 = wp::vec_t<3, wp::float32>();
        ret_0 = var_68;
        ret_1 = var_69;
        return;
    }
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/ray.py:228
static CUDA_CALLABLE void ray_capsule_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::vec_t<3, wp::float32> var_7;
    const wp::int32 var_8 = 0;
    bool var_9;
    const wp::float32 var_10 = 1.0;
    const wp::float32 var_11 = -1.0;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    const wp::float32 var_15 = 1.0;
    const wp::float32 var_16 = -1.0;
    const wp::int32 var_17 = 0;
    wp::float32 var_18;
    const wp::int32 var_19 = 0;
    wp::float32 var_20;
    wp::float32 var_21;
    const wp::int32 var_22 = 0;
    wp::float32 var_23;
    const wp::int32 var_24 = 0;
    wp::float32 var_25;
    wp::float32 var_26;
    const wp::int32 var_27 = 1;
    wp::float32 var_28;
    const wp::int32 var_29 = 1;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    const wp::int32 var_33 = 0;
    wp::float32 var_34;
    const wp::int32 var_35 = 0;
    wp::float32 var_36;
    wp::float32 var_37;
    const wp::int32 var_38 = 1;
    wp::float32 var_39;
    const wp::int32 var_40 = 1;
    wp::float32 var_41;
    wp::float32 var_42;
    wp::float32 var_43;
    const wp::int32 var_44 = 0;
    wp::float32 var_45;
    const wp::int32 var_46 = 0;
    wp::float32 var_47;
    wp::float32 var_48;
    const wp::int32 var_49 = 1;
    wp::float32 var_50;
    const wp::int32 var_51 = 1;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::float32 var_56;
    wp::vec_t<2, wp::float32> var_57;
    const wp::int32 var_58 = 0;
    const wp::float32 var_59 = 0.0;
    bool var_60;
    const wp::int32 var_61 = 2;
    wp::float32 var_62;
    const wp::int32 var_63 = 2;
    wp::float32 var_64;
    wp::float32 var_65;
    wp::float32 var_66;
    wp::float32 var_67;
    const wp::int32 var_68 = 1;
    wp::float32 var_69;
    bool var_70;
    bool var_71;
    const wp::float32 var_72 = 0.0;
    bool var_73;
    bool var_74;
    bool var_75;
    wp::float32 var_76;
    wp::float32 var_77;
    wp::float32 var_78;
    const wp::int32 var_79 = 0;
    wp::float32 var_80;
    const wp::int32 var_81 = 1;
    wp::float32 var_82;
    const wp::int32 var_83 = 2;
    wp::float32 var_84;
    const wp::int32 var_85 = 1;
    wp::float32 var_86;
    wp::float32 var_87;
    wp::vec_t<3, wp::float32> var_88;
    const wp::int32 var_89 = 2;
    wp::float32 var_90;
    const wp::int32 var_91 = 2;
    wp::float32 var_92;
    wp::float32 var_93;
    wp::float32 var_94;
    wp::float32 var_95;
    wp::float32 var_96;
    wp::float32 var_97;
    wp::float32 var_98;
    wp::vec_t<2, wp::float32> var_99;
    const wp::int32 var_100 = 0;
    wp::float32 var_101;
    const wp::float32 var_102 = 0.0;
    bool var_103;
    const wp::int32 var_104 = 2;
    wp::float32 var_105;
    wp::float32 var_106;
    const wp::int32 var_107 = 2;
    wp::float32 var_108;
    wp::float32 var_109;
    wp::float32 var_110;
    const wp::int32 var_111 = 1;
    wp::float32 var_112;
    bool var_113;
    bool var_114;
    const wp::float32 var_115 = 0.0;
    bool var_116;
    wp::float32 var_117;
    bool var_118;
    bool var_119;
    wp::float32 var_120;
    const wp::int32 var_121 = 1;
    wp::float32 var_122;
    wp::int32 var_123;
    wp::float32 var_124;
    wp::int32 var_125;
    const wp::int32 var_126 = 1;
    wp::float32 var_127;
    const wp::float32 var_128 = 0.0;
    bool var_129;
    const wp::int32 var_130 = 2;
    wp::float32 var_131;
    wp::float32 var_132;
    const wp::int32 var_133 = 2;
    wp::float32 var_134;
    wp::float32 var_135;
    wp::float32 var_136;
    const wp::int32 var_137 = 1;
    wp::float32 var_138;
    bool var_139;
    bool var_140;
    const wp::float32 var_141 = 0.0;
    bool var_142;
    wp::float32 var_143;
    bool var_144;
    bool var_145;
    wp::float32 var_146;
    const wp::int32 var_147 = 1;
    wp::float32 var_148;
    wp::int32 var_149;
    wp::float32 var_150;
    wp::int32 var_151;
    const wp::int32 var_152 = 0;
    wp::float32 var_153;
    const wp::int32 var_154 = 1;
    wp::float32 var_155;
    const wp::int32 var_156 = 2;
    wp::float32 var_157;
    const wp::int32 var_158 = 1;
    wp::float32 var_159;
    wp::float32 var_160;
    wp::vec_t<3, wp::float32> var_161;
    wp::float32 var_162;
    wp::float32 var_163;
    wp::float32 var_164;
    wp::float32 var_165;
    wp::vec_t<2, wp::float32> var_166;
    const wp::int32 var_167 = 0;
    wp::float32 var_168;
    const wp::float32 var_169 = 0.0;
    bool var_170;
    const wp::int32 var_171 = 2;
    wp::float32 var_172;
    wp::float32 var_173;
    const wp::int32 var_174 = 2;
    wp::float32 var_175;
    wp::float32 var_176;
    wp::float32 var_177;
    const wp::int32 var_178 = 1;
    wp::float32 var_179;
    wp::float32 var_180;
    bool var_181;
    bool var_182;
    const wp::float32 var_183 = 0.0;
    bool var_184;
    wp::float32 var_185;
    bool var_186;
    bool var_187;
    wp::float32 var_188;
    const wp::int32 var_189 = 1;
    const wp::int32 var_190 = -1;
    wp::float32 var_191;
    wp::int32 var_192;
    wp::float32 var_193;
    wp::int32 var_194;
    const wp::int32 var_195 = 1;
    wp::float32 var_196;
    const wp::float32 var_197 = 0.0;
    bool var_198;
    const wp::int32 var_199 = 2;
    wp::float32 var_200;
    wp::float32 var_201;
    const wp::int32 var_202 = 2;
    wp::float32 var_203;
    wp::float32 var_204;
    wp::float32 var_205;
    const wp::int32 var_206 = 1;
    wp::float32 var_207;
    wp::float32 var_208;
    bool var_209;
    bool var_210;
    const wp::float32 var_211 = 0.0;
    bool var_212;
    wp::float32 var_213;
    bool var_214;
    bool var_215;
    wp::float32 var_216;
    const wp::int32 var_217 = 1;
    const wp::int32 var_218 = -1;
    wp::float32 var_219;
    wp::int32 var_220;
    wp::float32 var_221;
    wp::int32 var_222;
    wp::vec_t<3, wp::float32> var_223;
    const wp::int32 var_224 = 0;
    bool var_225;
    const wp::int32 var_226 = 0;
    wp::float32 var_227;
    const wp::int32 var_228 = 0;
    wp::float32 var_229;
    wp::float32 var_230;
    wp::float32 var_231;
    const wp::int32 var_232 = 0;
    const wp::int32 var_233 = 1;
    wp::float32 var_234;
    const wp::int32 var_235 = 1;
    wp::float32 var_236;
    wp::float32 var_237;
    wp::float32 var_238;
    const wp::int32 var_239 = 1;
    const wp::int32 var_240 = 0;
    bool var_241;
    const wp::float32 var_242 = 0.0;
    const wp::int32 var_243 = 2;
    const wp::int32 var_244 = 2;
    wp::float32 var_245;
    const wp::int32 var_246 = 2;
    wp::float32 var_247;
    wp::float32 var_248;
    wp::float32 var_249;
    const wp::int32 var_250 = 1;
    wp::float32 var_251;
    wp::float32 var_252;
    wp::float32 var_253;
    wp::float32 var_254;
    const wp::int32 var_255 = 2;
    wp::vec_t<3, wp::float32> var_256;
    wp::vec_t<3, wp::float32> var_257;
    wp::vec_t<3, wp::float32> var_258;
    //---------
    // forward
    // def ray_capsule(pos: wp.vec3, mat: wp.mat33, size: wp.vec3, pnt: wp.vec3, vec: wp.vec3) -> Tuple[float, wp.vec3]:       <L 229>
    // ssz = size[0] + size[1]                                                                <L 232>
    var_1 = wp::extract(var_size, var_0);
    var_3 = wp::extract(var_size, var_2);
    var_4 = wp::add(var_1, var_3);
    // dist_sphere, normal_sphere = ray_sphere(pos, ssz * ssz, pnt, vec)                      <L 233>
    var_5 = wp::mul(var_4, var_4);
    ray_sphere_0(var_pos, var_5, var_pnt, var_vec, var_6, var_7);
    // if dist_sphere < 0:                                                                    <L 234>
    var_9 = (var_6 < var_8);
    if (var_9) {
        // return -1.0, wp.vec3()                                                             <L 235>
        var_12 = wp::vec_t<3, wp::float32>();
        ret_0 = var_11;
        ret_1 = var_12;
        return;
    }
    // lpnt, lvec = _ray_map(pos, mat, pnt, vec)                                              <L 238>
    _ray_map_0(var_pos, var_mat, var_pnt, var_vec, var_13, var_14);
    // x = -1.0                                                                               <L 241>
    // sq_size0 = size[0] * size[0]                                                           <L 244>
    var_18 = wp::extract(var_size, var_17);
    var_20 = wp::extract(var_size, var_19);
    var_21 = wp::mul(var_18, var_20);
    // a = lvec[0] * lvec[0] + lvec[1] * lvec[1]                                              <L 245>
    var_23 = wp::extract(var_14, var_22);
    var_25 = wp::extract(var_14, var_24);
    var_26 = wp::mul(var_23, var_25);
    var_28 = wp::extract(var_14, var_27);
    var_30 = wp::extract(var_14, var_29);
    var_31 = wp::mul(var_28, var_30);
    var_32 = wp::add(var_26, var_31);
    // b = lvec[0] * lpnt[0] + lvec[1] * lpnt[1]                                              <L 246>
    var_34 = wp::extract(var_14, var_33);
    var_36 = wp::extract(var_13, var_35);
    var_37 = wp::mul(var_34, var_36);
    var_39 = wp::extract(var_14, var_38);
    var_41 = wp::extract(var_13, var_40);
    var_42 = wp::mul(var_39, var_41);
    var_43 = wp::add(var_37, var_42);
    // c = lpnt[0] * lpnt[0] + lpnt[1] * lpnt[1] - sq_size0                                   <L 247>
    var_45 = wp::extract(var_13, var_44);
    var_47 = wp::extract(var_13, var_46);
    var_48 = wp::mul(var_45, var_47);
    var_50 = wp::extract(var_13, var_49);
    var_52 = wp::extract(var_13, var_51);
    var_53 = wp::mul(var_50, var_52);
    var_54 = wp::add(var_48, var_53);
    var_55 = wp::sub(var_54, var_21);
    // sol, xx = _ray_quad(a, b, c)                                                           <L 250>
    _ray_quad_0(var_32, var_43, var_55, var_56, var_57);
    // part = 0  # -1: bottom, 0: cylinder, 1: top                                            <L 251>
    // if sol >= 0.0 and wp.abs(lpnt[2] + sol * lvec[2]) <= size[1]:                          <L 255>
    var_60 = (var_56 >= var_59);
    var_62 = wp::extract(var_13, var_61);
    var_64 = wp::extract(var_14, var_63);
    var_65 = wp::mul(var_56, var_64);
    var_66 = wp::add(var_62, var_65);
    var_67 = wp::abs(var_66);
    var_69 = wp::extract(var_size, var_68);
    var_70 = (var_67 <= var_69);
    var_71 = var_60 && var_70;
    if (var_71) {
        // if x < 0.0 or sol < x:                                                             <L 256>
        var_73 = (var_16 < var_72);
        var_74 = (var_56 < var_16);
        var_75 = var_73 || var_74;
        if (var_75) {
            // x = sol                                                                        <L 257>
            var_76 = wp::copy(var_56);
        }
        var_77 = wp::where(var_75, var_76, var_16);
    }
    var_78 = wp::where(var_71, var_77, var_16);
    // ldif = wp.vec3(lpnt[0], lpnt[1], lpnt[2] - size[1])                                    <L 260>
    var_80 = wp::extract(var_13, var_79);
    var_82 = wp::extract(var_13, var_81);
    var_84 = wp::extract(var_13, var_83);
    var_86 = wp::extract(var_size, var_85);
    var_87 = wp::sub(var_84, var_86);
    var_88 = wp::vec_t<3, wp::float32>(var_80, var_82, var_87);
    // a += lvec[2] * lvec[2]                                                                 <L 261>
    var_90 = wp::extract(var_14, var_89);
    var_92 = wp::extract(var_14, var_91);
    var_93 = wp::mul(var_90, var_92);
    var_94 = wp::add(var_32, var_93);
    // b = wp.dot(lvec, ldif)                                                                 <L 262>
    var_95 = wp::dot(var_14, var_88);
    // c = wp.dot(ldif, ldif) - sq_size0                                                      <L 263>
    var_96 = wp::dot(var_88, var_88);
    var_97 = wp::sub(var_96, var_21);
    // _, xx = _ray_quad(a, b, c)                                                             <L 264>
    _ray_quad_0(var_94, var_95, var_97, var_98, var_99);
    // for i in range(2):                                                                     <L 267>
    // if xx[i] >= 0.0 and lpnt[2] + xx[i] * lvec[2] >= size[1]:                              <L 268>
    var_101 = wp::extract(var_99, var_100);
    var_103 = (var_101 >= var_102);
    var_105 = wp::extract(var_13, var_104);
    var_106 = wp::extract(var_99, var_100);
    var_108 = wp::extract(var_14, var_107);
    var_109 = wp::mul(var_106, var_108);
    var_110 = wp::add(var_105, var_109);
    var_112 = wp::extract(var_size, var_111);
    var_113 = (var_110 >= var_112);
    var_114 = var_103 && var_113;
    if (var_114) {
        // if x < 0.0 or xx[i] < x:                                                           <L 269>
        var_116 = (var_78 < var_115);
        var_117 = wp::extract(var_99, var_100);
        var_118 = (var_117 < var_78);
        var_119 = var_116 || var_118;
        if (var_119) {
            // x = xx[i]                                                                      <L 270>
            var_120 = wp::extract(var_99, var_100);
            // part = 1                                                                       <L 271>
        }
        var_122 = wp::where(var_119, var_120, var_78);
        var_123 = wp::where(var_119, var_121, var_58);
    }
    var_124 = wp::where(var_114, var_122, var_78);
    var_125 = wp::where(var_114, var_123, var_58);
    // if xx[i] >= 0.0 and lpnt[2] + xx[i] * lvec[2] >= size[1]:                              <L 268>
    var_127 = wp::extract(var_99, var_126);
    var_129 = (var_127 >= var_128);
    var_131 = wp::extract(var_13, var_130);
    var_132 = wp::extract(var_99, var_126);
    var_134 = wp::extract(var_14, var_133);
    var_135 = wp::mul(var_132, var_134);
    var_136 = wp::add(var_131, var_135);
    var_138 = wp::extract(var_size, var_137);
    var_139 = (var_136 >= var_138);
    var_140 = var_129 && var_139;
    if (var_140) {
        // if x < 0.0 or xx[i] < x:                                                           <L 269>
        var_142 = (var_124 < var_141);
        var_143 = wp::extract(var_99, var_126);
        var_144 = (var_143 < var_124);
        var_145 = var_142 || var_144;
        if (var_145) {
            // x = xx[i]                                                                      <L 270>
            var_146 = wp::extract(var_99, var_126);
            // part = 1                                                                       <L 271>
        }
        var_148 = wp::where(var_145, var_146, var_124);
        var_149 = wp::where(var_145, var_147, var_125);
    }
    var_150 = wp::where(var_140, var_148, var_124);
    var_151 = wp::where(var_140, var_149, var_125);
    // ldif = wp.vec3(ldif[0], ldif[1], lpnt[2] + size[1])                                    <L 274>
    var_153 = wp::extract(var_88, var_152);
    var_155 = wp::extract(var_88, var_154);
    var_157 = wp::extract(var_13, var_156);
    var_159 = wp::extract(var_size, var_158);
    var_160 = wp::add(var_157, var_159);
    var_161 = wp::vec_t<3, wp::float32>(var_153, var_155, var_160);
    // b = wp.dot(lvec, ldif)                                                                 <L 275>
    var_162 = wp::dot(var_14, var_161);
    // c = wp.dot(ldif, ldif) - sq_size0                                                      <L 276>
    var_163 = wp::dot(var_161, var_161);
    var_164 = wp::sub(var_163, var_21);
    // _, xx = _ray_quad(a, b, c)                                                             <L 277>
    _ray_quad_0(var_94, var_162, var_164, var_165, var_166);
    // for i in range(2):                                                                     <L 280>
    // if xx[i] >= 0.0 and lpnt[2] + xx[i] * lvec[2] <= -size[1]:                             <L 281>
    var_168 = wp::extract(var_166, var_167);
    var_170 = (var_168 >= var_169);
    var_172 = wp::extract(var_13, var_171);
    var_173 = wp::extract(var_166, var_167);
    var_175 = wp::extract(var_14, var_174);
    var_176 = wp::mul(var_173, var_175);
    var_177 = wp::add(var_172, var_176);
    var_179 = wp::extract(var_size, var_178);
    var_180 = wp::neg(var_179);
    var_181 = (var_177 <= var_180);
    var_182 = var_170 && var_181;
    if (var_182) {
        // if x < 0.0 or xx[i] < x:                                                           <L 282>
        var_184 = (var_150 < var_183);
        var_185 = wp::extract(var_166, var_167);
        var_186 = (var_185 < var_150);
        var_187 = var_184 || var_186;
        if (var_187) {
            // x = xx[i]                                                                      <L 283>
            var_188 = wp::extract(var_166, var_167);
            // part = -1                                                                      <L 284>
        }
        var_191 = wp::where(var_187, var_188, var_150);
        var_192 = wp::where(var_187, var_190, var_151);
    }
    var_193 = wp::where(var_182, var_191, var_150);
    var_194 = wp::where(var_182, var_192, var_151);
    // if xx[i] >= 0.0 and lpnt[2] + xx[i] * lvec[2] <= -size[1]:                             <L 281>
    var_196 = wp::extract(var_166, var_195);
    var_198 = (var_196 >= var_197);
    var_200 = wp::extract(var_13, var_199);
    var_201 = wp::extract(var_166, var_195);
    var_203 = wp::extract(var_14, var_202);
    var_204 = wp::mul(var_201, var_203);
    var_205 = wp::add(var_200, var_204);
    var_207 = wp::extract(var_size, var_206);
    var_208 = wp::neg(var_207);
    var_209 = (var_205 <= var_208);
    var_210 = var_198 && var_209;
    if (var_210) {
        // if x < 0.0 or xx[i] < x:                                                           <L 282>
        var_212 = (var_193 < var_211);
        var_213 = wp::extract(var_166, var_195);
        var_214 = (var_213 < var_193);
        var_215 = var_212 || var_214;
        if (var_215) {
            // x = xx[i]                                                                      <L 283>
            var_216 = wp::extract(var_166, var_195);
            // part = -1                                                                      <L 284>
        }
        var_219 = wp::where(var_215, var_216, var_193);
        var_220 = wp::where(var_215, var_218, var_194);
    }
    var_221 = wp::where(var_210, var_219, var_193);
    var_222 = wp::where(var_210, var_220, var_194);
    // normal = wp.vec3()                                                                     <L 286>
    var_223 = wp::vec_t<3, wp::float32>();
    // if x >= 0:                                                                             <L 287>
    var_225 = (var_221 >= var_224);
    if (var_225) {
        // normal[0] = lpnt[0] + lvec[0] * x                                                  <L 288>
        var_227 = wp::extract(var_13, var_226);
        var_229 = wp::extract(var_14, var_228);
        var_230 = wp::mul(var_229, var_221);
        var_231 = wp::add(var_227, var_230);
        wp::assign_inplace(var_223, var_232, var_231);
        // normal[1] = lpnt[1] + lvec[1] * x                                                  <L 289>
        var_234 = wp::extract(var_13, var_233);
        var_236 = wp::extract(var_14, var_235);
        var_237 = wp::mul(var_236, var_221);
        var_238 = wp::add(var_234, var_237);
        wp::assign_inplace(var_223, var_239, var_238);
        // if part == 0:                                                                      <L 290>
        var_241 = (var_222 == var_240);
        if (var_241) {
            // normal[2] = 0.0                                                                <L 291>
            wp::assign_inplace(var_223, var_243, var_242);
        }
        if (!var_241) {
            // normal[2] = lpnt[2] + lvec[2] * x - size[1] * float(part)                      <L 293>
            var_245 = wp::extract(var_13, var_244);
            var_247 = wp::extract(var_14, var_246);
            var_248 = wp::mul(var_247, var_221);
            var_249 = wp::add(var_245, var_248);
            var_251 = wp::extract(var_size, var_250);
            var_252 = wp::float(var_222);
            var_253 = wp::mul(var_251, var_252);
            var_254 = wp::sub(var_249, var_253);
            wp::assign_inplace(var_223, var_255, var_254);
        }
        // normal = wp.normalize(normal)                                                      <L 296>
        var_256 = wp::normalize(var_223);
        // normal = mat @ normal                                                              <L 297>
        var_257 = wp::mul(var_mat, var_256);
    }
    var_258 = wp::where(var_225, var_257, var_223);
    // return x, normal                                                                       <L 299>
    ret_0 = var_221;
    ret_1 = var_258;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/ray.py:302
static CUDA_CALLABLE void ray_ellipsoid_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    const wp::float32 var_2 = 1.0;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    const wp::int32 var_5 = 0;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    const wp::float32 var_9 = 1.0;
    const wp::int32 var_10 = 1;
    wp::float32 var_11;
    const wp::int32 var_12 = 1;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    const wp::float32 var_16 = 1.0;
    const wp::int32 var_17 = 2;
    wp::float32 var_18;
    const wp::int32 var_19 = 2;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::float32 var_28;
    const wp::float32 var_29 = 1.0;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::vec_t<2, wp::float32> var_32;
    wp::vec_t<3, wp::float32> var_33;
    const wp::int32 var_34 = 0;
    bool var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::vec_t<3, wp::float32> var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::vec_t<3, wp::float32> var_41;
    //---------
    // forward
    // def ray_ellipsoid(pos: wp.vec3, mat: wp.mat33, size: wp.vec3, pnt: wp.vec3, vec: wp.vec3) -> Tuple[float, wp.vec3]:       <L 303>
    // lpnt, lvec = _ray_map(pos, mat, pnt, vec)                                              <L 306>
    _ray_map_0(var_pos, var_mat, var_pnt, var_vec, var_0, var_1);
    // s = wp.vec3(safe_div(1.0, size[0] * size[0]), safe_div(1.0, size[1] * size[1]), safe_div(1.0, size[2] * size[2]))       <L 309>
    var_4 = wp::extract(var_size, var_3);
    var_6 = wp::extract(var_size, var_5);
    var_7 = wp::mul(var_4, var_6);
    var_8 = safe_div_0(var_2, var_7);
    var_11 = wp::extract(var_size, var_10);
    var_13 = wp::extract(var_size, var_12);
    var_14 = wp::mul(var_11, var_13);
    var_15 = safe_div_0(var_9, var_14);
    var_18 = wp::extract(var_size, var_17);
    var_20 = wp::extract(var_size, var_19);
    var_21 = wp::mul(var_18, var_20);
    var_22 = safe_div_0(var_16, var_21);
    var_23 = wp::vec_t<3, wp::float32>(var_8, var_15, var_22);
    // slvec = wp.cw_mul(s, lvec)                                                             <L 312>
    var_24 = wp::cw_mul(var_23, var_1);
    // a = wp.dot(slvec, lvec)                                                                <L 313>
    var_25 = wp::dot(var_24, var_1);
    // b = wp.dot(slvec, lpnt)                                                                <L 314>
    var_26 = wp::dot(var_24, var_0);
    // c = wp.dot(wp.cw_mul(s, lpnt), lpnt) - 1.0                                             <L 315>
    var_27 = wp::cw_mul(var_23, var_0);
    var_28 = wp::dot(var_27, var_0);
    var_30 = wp::sub(var_28, var_29);
    // sol, _ = _ray_quad(a, b, c)                                                            <L 318>
    _ray_quad_0(var_25, var_26, var_30, var_31, var_32);
    // normal = wp.vec3()                                                                     <L 320>
    var_33 = wp::vec_t<3, wp::float32>();
    // if sol >= 0:                                                                           <L 321>
    var_35 = (var_31 >= var_34);
    if (var_35) {
        // l = lpnt + lvec * sol                                                              <L 323>
        var_36 = wp::mul(var_1, var_31);
        var_37 = wp::add(var_0, var_36);
        // normal = wp.cw_mul(s, l)                                                           <L 326>
        var_38 = wp::cw_mul(var_23, var_37);
        // normal = wp.normalize(normal)                                                      <L 327>
        var_39 = wp::normalize(var_38);
        // normal = mat @ normal                                                              <L 328>
        var_40 = wp::mul(var_mat, var_39);
    }
    var_41 = wp::where(var_35, var_40, var_33);
    // return sol, normal                                                                     <L 330>
    ret_0 = var_31;
    ret_1 = var_41;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/ray.py:333
static CUDA_CALLABLE void ray_cylinder_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
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
    wp::float32 var_11;
    wp::vec_t<3, wp::float32> var_12;
    const wp::int32 var_13 = 0;
    bool var_14;
    const wp::float32 var_15 = 1.0;
    const wp::float32 var_16 = -1.0;
    wp::vec_t<3, wp::float32> var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    const wp::float32 var_20 = 1.0;
    const wp::float32 var_21 = -1.0;
    const wp::int32 var_22 = 0;
    const wp::int32 var_23 = 2;
    wp::float32 var_24;
    wp::float32 var_25;
    const wp::float32 var_26 = 1e-15;
    bool var_27;
    const wp::int32 var_28 = -1;
    wp::float32 var_29;
    const wp::int32 var_30 = 1;
    wp::float32 var_31;
    wp::float32 var_32;
    const wp::int32 var_33 = 2;
    wp::float32 var_34;
    wp::float32 var_35;
    const wp::int32 var_36 = 2;
    wp::float32 var_37;
    wp::float32 var_38;
    const wp::float32 var_39 = 0.0;
    bool var_40;
    const wp::int32 var_41 = 0;
    wp::float32 var_42;
    const wp::int32 var_43 = 0;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    const wp::int32 var_47 = 1;
    wp::float32 var_48;
    const wp::int32 var_49 = 1;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    wp::vec_t<2, wp::float32> var_53;
    wp::float32 var_54;
    const wp::int32 var_55 = 0;
    wp::float32 var_56;
    const wp::int32 var_57 = 0;
    wp::float32 var_58;
    wp::float32 var_59;
    bool var_60;
    const wp::float32 var_61 = 0.0;
    bool var_62;
    bool var_63;
    bool var_64;
    wp::float32 var_65;
    wp::int32 var_66;
    wp::float32 var_67;
    wp::int32 var_68;
    wp::float32 var_69;
    wp::int32 var_70;
    wp::float32 var_71;
    wp::int32 var_72;
    const wp::int32 var_73 = 1;
    wp::float32 var_74;
    const wp::int32 var_75 = 1;
    wp::float32 var_76;
    wp::float32 var_77;
    const wp::int32 var_78 = 2;
    wp::float32 var_79;
    wp::float32 var_80;
    const wp::int32 var_81 = 2;
    wp::float32 var_82;
    wp::float32 var_83;
    const wp::float32 var_84 = 0.0;
    bool var_85;
    const wp::int32 var_86 = 0;
    wp::float32 var_87;
    const wp::int32 var_88 = 0;
    wp::float32 var_89;
    wp::float32 var_90;
    wp::float32 var_91;
    const wp::int32 var_92 = 1;
    wp::float32 var_93;
    const wp::int32 var_94 = 1;
    wp::float32 var_95;
    wp::float32 var_96;
    wp::float32 var_97;
    wp::vec_t<2, wp::float32> var_98;
    wp::float32 var_99;
    const wp::int32 var_100 = 0;
    wp::float32 var_101;
    const wp::int32 var_102 = 0;
    wp::float32 var_103;
    wp::float32 var_104;
    bool var_105;
    const wp::float32 var_106 = 0.0;
    bool var_107;
    bool var_108;
    bool var_109;
    wp::float32 var_110;
    wp::int32 var_111;
    wp::float32 var_112;
    wp::int32 var_113;
    wp::float32 var_114;
    wp::int32 var_115;
    wp::float32 var_116;
    wp::int32 var_117;
    wp::vec_t<2, wp::float32> var_118;
    wp::float32 var_119;
    wp::int32 var_120;
    const wp::int32 var_121 = 0;
    wp::float32 var_122;
    const wp::int32 var_123 = 0;
    wp::float32 var_124;
    wp::float32 var_125;
    const wp::int32 var_126 = 1;
    wp::float32 var_127;
    const wp::int32 var_128 = 1;
    wp::float32 var_129;
    wp::float32 var_130;
    wp::float32 var_131;
    const wp::int32 var_132 = 0;
    wp::float32 var_133;
    const wp::int32 var_134 = 0;
    wp::float32 var_135;
    wp::float32 var_136;
    const wp::int32 var_137 = 1;
    wp::float32 var_138;
    const wp::int32 var_139 = 1;
    wp::float32 var_140;
    wp::float32 var_141;
    wp::float32 var_142;
    const wp::int32 var_143 = 0;
    wp::float32 var_144;
    const wp::int32 var_145 = 0;
    wp::float32 var_146;
    wp::float32 var_147;
    const wp::int32 var_148 = 1;
    wp::float32 var_149;
    const wp::int32 var_150 = 1;
    wp::float32 var_151;
    wp::float32 var_152;
    wp::float32 var_153;
    const wp::int32 var_154 = 0;
    wp::float32 var_155;
    const wp::int32 var_156 = 0;
    wp::float32 var_157;
    wp::float32 var_158;
    wp::float32 var_159;
    wp::float32 var_160;
    wp::vec_t<2, wp::float32> var_161;
    const wp::float32 var_162 = 0.0;
    bool var_163;
    const wp::int32 var_164 = 2;
    wp::float32 var_165;
    const wp::int32 var_166 = 2;
    wp::float32 var_167;
    wp::float32 var_168;
    wp::float32 var_169;
    wp::float32 var_170;
    const wp::int32 var_171 = 1;
    wp::float32 var_172;
    bool var_173;
    bool var_174;
    const wp::float32 var_175 = 0.0;
    bool var_176;
    bool var_177;
    bool var_178;
    wp::float32 var_179;
    const wp::int32 var_180 = 0;
    wp::float32 var_181;
    wp::int32 var_182;
    wp::float32 var_183;
    wp::int32 var_184;
    wp::vec_t<3, wp::float32> var_185;
    const wp::int32 var_186 = 0;
    bool var_187;
    const wp::int32 var_188 = 0;
    bool var_189;
    wp::vec_t<3, wp::float32> var_190;
    wp::vec_t<3, wp::float32> var_191;
    const wp::float32 var_192 = 0.0;
    const wp::int32 var_193 = 2;
    wp::vec_t<3, wp::float32> var_194;
    wp::vec_t<3, wp::float32> var_195;
    const wp::float32 var_196 = 0.0;
    const wp::float32 var_197 = 0.0;
    wp::float32 var_198;
    wp::vec_t<3, wp::float32> var_199;
    wp::vec_t<3, wp::float32> var_200;
    wp::vec_t<3, wp::float32> var_201;
    wp::vec_t<3, wp::float32> var_202;
    //---------
    // forward
    // def ray_cylinder(pos: wp.vec3, mat: wp.mat33, size: wp.vec3, pnt: wp.vec3, vec: wp.vec3) -> Tuple[float, wp.vec3]:       <L 334>
    // ssz = size[0] * size[0] + size[1] * size[1]                                            <L 337>
    var_1 = wp::extract(var_size, var_0);
    var_3 = wp::extract(var_size, var_2);
    var_4 = wp::mul(var_1, var_3);
    var_6 = wp::extract(var_size, var_5);
    var_8 = wp::extract(var_size, var_7);
    var_9 = wp::mul(var_6, var_8);
    var_10 = wp::add(var_4, var_9);
    // dist_sphere, normal_sphere = ray_sphere(pos, ssz, pnt, vec)                            <L 338>
    ray_sphere_0(var_pos, var_10, var_pnt, var_vec, var_11, var_12);
    // if dist_sphere < 0:                                                                    <L 339>
    var_14 = (var_11 < var_13);
    if (var_14) {
        // return -1.0, wp.vec3()                                                             <L 340>
        var_17 = wp::vec_t<3, wp::float32>();
        ret_0 = var_16;
        ret_1 = var_17;
        return;
    }
    // lpnt, lvec = _ray_map(pos, mat, pnt, vec)                                              <L 343>
    _ray_map_0(var_pos, var_mat, var_pnt, var_vec, var_18, var_19);
    // x = -1.0                                                                               <L 346>
    // part = 0  # -1: bottom, 0: cylinder, 1: top                                            <L 347>
    // if wp.abs(lvec[2]) > MJ_MINVAL:                                                        <L 350>
    var_24 = wp::extract(var_19, var_23);
    var_25 = wp::abs(var_24);
    var_27 = (var_25 > var_26);
    if (var_27) {
        // for side in range(-1, 2, 2):                                                       <L 351>
        // sol = (float(side) * size[1] - lpnt[2]) / lvec[2]                                  <L 353>
        var_29 = wp::float(var_28);
        var_31 = wp::extract(var_size, var_30);
        var_32 = wp::mul(var_29, var_31);
        var_34 = wp::extract(var_18, var_33);
        var_35 = wp::sub(var_32, var_34);
        var_37 = wp::extract(var_19, var_36);
        var_38 = wp::div(var_35, var_37);
        // if sol >= 0.0:                                                                     <L 356>
        var_40 = (var_38 >= var_39);
        if (var_40) {
            // p = wp.vec2(lpnt[0] + sol * lvec[0], lpnt[1] + sol * lvec[1])                  <L 358>
            var_42 = wp::extract(var_18, var_41);
            var_44 = wp::extract(var_19, var_43);
            var_45 = wp::mul(var_38, var_44);
            var_46 = wp::add(var_42, var_45);
            var_48 = wp::extract(var_18, var_47);
            var_50 = wp::extract(var_19, var_49);
            var_51 = wp::mul(var_38, var_50);
            var_52 = wp::add(var_48, var_51);
            var_53 = wp::vec_t<2, wp::float32>(var_46, var_52);
            // if wp.dot(p, p) <= size[0] * size[0]:                                          <L 361>
            var_54 = wp::dot(var_53, var_53);
            var_56 = wp::extract(var_size, var_55);
            var_58 = wp::extract(var_size, var_57);
            var_59 = wp::mul(var_56, var_58);
            var_60 = (var_54 <= var_59);
            if (var_60) {
                // if x < 0.0 or sol < x:                                                     <L 362>
                var_62 = (var_21 < var_61);
                var_63 = (var_38 < var_21);
                var_64 = var_62 || var_63;
                if (var_64) {
                    // x = sol                                                                <L 363>
                    var_65 = wp::copy(var_38);
                    // part = side                                                            <L 364>
                    var_66 = wp::copy(var_28);
                }
                var_67 = wp::where(var_64, var_65, var_21);
                var_68 = wp::where(var_64, var_66, var_22);
            }
            var_69 = wp::where(var_60, var_67, var_21);
            var_70 = wp::where(var_60, var_68, var_22);
        }
        var_71 = wp::where(var_40, var_69, var_21);
        var_72 = wp::where(var_40, var_70, var_22);
        // sol = (float(side) * size[1] - lpnt[2]) / lvec[2]                                  <L 353>
        var_74 = wp::float(var_73);
        var_76 = wp::extract(var_size, var_75);
        var_77 = wp::mul(var_74, var_76);
        var_79 = wp::extract(var_18, var_78);
        var_80 = wp::sub(var_77, var_79);
        var_82 = wp::extract(var_19, var_81);
        var_83 = wp::div(var_80, var_82);
        // if sol >= 0.0:                                                                     <L 356>
        var_85 = (var_83 >= var_84);
        if (var_85) {
            // p = wp.vec2(lpnt[0] + sol * lvec[0], lpnt[1] + sol * lvec[1])                  <L 358>
            var_87 = wp::extract(var_18, var_86);
            var_89 = wp::extract(var_19, var_88);
            var_90 = wp::mul(var_83, var_89);
            var_91 = wp::add(var_87, var_90);
            var_93 = wp::extract(var_18, var_92);
            var_95 = wp::extract(var_19, var_94);
            var_96 = wp::mul(var_83, var_95);
            var_97 = wp::add(var_93, var_96);
            var_98 = wp::vec_t<2, wp::float32>(var_91, var_97);
            // if wp.dot(p, p) <= size[0] * size[0]:                                          <L 361>
            var_99 = wp::dot(var_98, var_98);
            var_101 = wp::extract(var_size, var_100);
            var_103 = wp::extract(var_size, var_102);
            var_104 = wp::mul(var_101, var_103);
            var_105 = (var_99 <= var_104);
            if (var_105) {
                // if x < 0.0 or sol < x:                                                     <L 362>
                var_107 = (var_71 < var_106);
                var_108 = (var_83 < var_71);
                var_109 = var_107 || var_108;
                if (var_109) {
                    // x = sol                                                                <L 363>
                    var_110 = wp::copy(var_83);
                    // part = side                                                            <L 364>
                    var_111 = wp::copy(var_73);
                }
                var_112 = wp::where(var_109, var_110, var_71);
                var_113 = wp::where(var_109, var_111, var_72);
            }
            var_114 = wp::where(var_105, var_112, var_71);
            var_115 = wp::where(var_105, var_113, var_72);
        }
        var_116 = wp::where(var_85, var_114, var_71);
        var_117 = wp::where(var_85, var_115, var_72);
        var_118 = wp::where(var_85, var_98, var_53);
    }
    var_119 = wp::where(var_27, var_116, var_21);
    var_120 = wp::where(var_27, var_117, var_22);
    // a = lvec[0] * lvec[0] + lvec[1] * lvec[1]                                              <L 367>
    var_122 = wp::extract(var_19, var_121);
    var_124 = wp::extract(var_19, var_123);
    var_125 = wp::mul(var_122, var_124);
    var_127 = wp::extract(var_19, var_126);
    var_129 = wp::extract(var_19, var_128);
    var_130 = wp::mul(var_127, var_129);
    var_131 = wp::add(var_125, var_130);
    // b = lvec[0] * lpnt[0] + lvec[1] * lpnt[1]                                              <L 368>
    var_133 = wp::extract(var_19, var_132);
    var_135 = wp::extract(var_18, var_134);
    var_136 = wp::mul(var_133, var_135);
    var_138 = wp::extract(var_19, var_137);
    var_140 = wp::extract(var_18, var_139);
    var_141 = wp::mul(var_138, var_140);
    var_142 = wp::add(var_136, var_141);
    // c = lpnt[0] * lpnt[0] + lpnt[1] * lpnt[1] - size[0] * size[0]                          <L 369>
    var_144 = wp::extract(var_18, var_143);
    var_146 = wp::extract(var_18, var_145);
    var_147 = wp::mul(var_144, var_146);
    var_149 = wp::extract(var_18, var_148);
    var_151 = wp::extract(var_18, var_150);
    var_152 = wp::mul(var_149, var_151);
    var_153 = wp::add(var_147, var_152);
    var_155 = wp::extract(var_size, var_154);
    var_157 = wp::extract(var_size, var_156);
    var_158 = wp::mul(var_155, var_157);
    var_159 = wp::sub(var_153, var_158);
    // sol, _ = _ray_quad(a, b, c)                                                            <L 372>
    _ray_quad_0(var_131, var_142, var_159, var_160, var_161);
    // if sol >= 0.0 and wp.abs(lpnt[2] + sol * lvec[2]) <= size[1]:                          <L 375>
    var_163 = (var_160 >= var_162);
    var_165 = wp::extract(var_18, var_164);
    var_167 = wp::extract(var_19, var_166);
    var_168 = wp::mul(var_160, var_167);
    var_169 = wp::add(var_165, var_168);
    var_170 = wp::abs(var_169);
    var_172 = wp::extract(var_size, var_171);
    var_173 = (var_170 <= var_172);
    var_174 = var_163 && var_173;
    if (var_174) {
        // if x < 0.0 or sol < x:                                                             <L 376>
        var_176 = (var_119 < var_175);
        var_177 = (var_160 < var_119);
        var_178 = var_176 || var_177;
        if (var_178) {
            // x = sol                                                                        <L 377>
            var_179 = wp::copy(var_160);
            // part = 0                                                                       <L 378>
        }
        var_181 = wp::where(var_178, var_179, var_119);
        var_182 = wp::where(var_178, var_180, var_120);
    }
    var_183 = wp::where(var_174, var_181, var_119);
    var_184 = wp::where(var_174, var_182, var_120);
    // normal = wp.vec3()                                                                     <L 380>
    var_185 = wp::vec_t<3, wp::float32>();
    // if x >= 0:                                                                             <L 381>
    var_187 = (var_183 >= var_186);
    if (var_187) {
        // if part == 0:                                                                      <L 382>
        var_189 = (var_184 == var_188);
        if (var_189) {
            // normal = lpnt + lvec * x                                                       <L 383>
            var_190 = wp::mul(var_19, var_183);
            var_191 = wp::add(var_18, var_190);
            // normal[2] = 0.0                                                                <L 384>
            wp::assign_inplace(var_191, var_193, var_192);
            // normal = wp.normalize(normal)                                                  <L 385>
            var_194 = wp::normalize(var_191);
        }
        var_195 = wp::where(var_189, var_194, var_185);
        if (!var_189) {
            // normal = wp.vec3(0.0, 0.0, float(part))                                        <L 387>
            var_198 = wp::float(var_184);
            var_199 = wp::vec_t<3, wp::float32>(var_196, var_197, var_198);
        }
        var_200 = wp::where(var_189, var_195, var_199);
        // normal = mat @ normal                                                              <L 389>
        var_201 = wp::mul(var_mat, var_200);
    }
    var_202 = wp::where(var_187, var_201, var_185);
    // return x, normal                                                                       <L 391>
    ret_0 = var_183;
    ret_1 = var_202;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/ray.py:799
static CUDA_CALLABLE void ray_geom_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::int32 var_geomtype,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    bool var_1;
    wp::float32 var_2;
    wp::vec_t<3, wp::float32> var_3;
    const wp::int32 var_4 = 2;
    bool var_5;
    const wp::int32 var_6 = 0;
    wp::float32 var_7;
    const wp::int32 var_8 = 0;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::vec_t<3, wp::float32> var_12;
    const wp::int32 var_13 = 3;
    bool var_14;
    wp::float32 var_15;
    wp::vec_t<3, wp::float32> var_16;
    const wp::int32 var_17 = 4;
    bool var_18;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    const wp::int32 var_21 = 5;
    bool var_22;
    wp::float32 var_23;
    wp::vec_t<3, wp::float32> var_24;
    const wp::int32 var_25 = 6;
    bool var_26;
    wp::float32 var_27;
    wp::vec_t<6, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    const wp::float32 var_30 = 1.0;
    const wp::float32 var_31 = -1.0;
    wp::vec_t<3, wp::float32> var_32;
    //---------
    // forward
    // def ray_geom(pos: wp.vec3, mat: wp.mat33, size: wp.vec3, pnt: wp.vec3, vec: wp.vec3, geomtype: int) -> Tuple[float, wp.vec3]:       <L 800>
    // if geomtype == GeomType.PLANE:                                                         <L 806>
    var_1 = (var_geomtype == var_0);
    if (var_1) {
        // return ray_plane(pos, mat, size, pnt, vec)                                         <L 807>
        ray_plane_0(var_pos, var_mat, var_size, var_pnt, var_vec, var_2, var_3);
        ret_0 = var_2;
        ret_1 = var_3;
        return;
    }
    if (!var_1) {
        // elif geomtype == GeomType.SPHERE:                                                  <L 808>
        var_5 = (var_geomtype == var_4);
        if (var_5) {
            // return ray_sphere(pos, size[0] * size[0], pnt, vec)                            <L 809>
            var_7 = wp::extract(var_size, var_6);
            var_9 = wp::extract(var_size, var_8);
            var_10 = wp::mul(var_7, var_9);
            ray_sphere_0(var_pos, var_10, var_pnt, var_vec, var_11, var_12);
            ret_0 = var_11;
            ret_1 = var_12;
            return;
        }
        if (!var_5) {
            // elif geomtype == GeomType.CAPSULE:                                             <L 810>
            var_14 = (var_geomtype == var_13);
            if (var_14) {
                // return ray_capsule(pos, mat, size, pnt, vec)                               <L 811>
                ray_capsule_0(var_pos, var_mat, var_size, var_pnt, var_vec, var_15, var_16);
                ret_0 = var_15;
                ret_1 = var_16;
                return;
            }
            if (!var_14) {
                // elif geomtype == GeomType.ELLIPSOID:                                       <L 812>
                var_18 = (var_geomtype == var_17);
                if (var_18) {
                    // return ray_ellipsoid(pos, mat, size, pnt, vec)                         <L 813>
                    ray_ellipsoid_0(var_pos, var_mat, var_size, var_pnt, var_vec, var_19, var_20);
                    ret_0 = var_19;
                    ret_1 = var_20;
                    return;
                }
                if (!var_18) {
                    // elif geomtype == GeomType.CYLINDER:                                    <L 814>
                    var_22 = (var_geomtype == var_21);
                    if (var_22) {
                        // return ray_cylinder(pos, mat, size, pnt, vec)                      <L 815>
                        ray_cylinder_0(var_pos, var_mat, var_size, var_pnt, var_vec, var_23, var_24);
                        ret_0 = var_23;
                        ret_1 = var_24;
                        return;
                    }
                    if (!var_22) {
                        // elif geomtype == GeomType.BOX:                                     <L 816>
                        var_26 = (var_geomtype == var_25);
                        if (var_26) {
                            // dist, _, normal = ray_box(pos, mat, size, pnt, vec)            <L 817>
                            ray_box_0(var_pos, var_mat, var_size, var_pnt, var_vec, var_27, var_28, var_29);
                            // return dist, normal                                            <L 818>
                            ret_0 = var_27;
                            ret_1 = var_29;
                            return;
                        }
                        if (!var_26) {
                            // return -1.0, wp.vec3()                                         <L 820>
                            var_32 = wp::vec_t<3, wp::float32>();
                            ret_0 = var_31;
                            ret_1 = var_32;
                            return;
                        }
                    }
                }
            }
        }
    }
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:908
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _velocimeter_0(
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    wp::vec_t<3, wp::float32>* var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::mat_t<3, 3, wp::float32>* var_6;
    wp::mat_t<3, 3, wp::float32> var_7;
    wp::mat_t<3, 3, wp::float32> var_8;
    wp::vec_t<6, wp::float32>* var_9;
    wp::vec_t<6, wp::float32> var_10;
    wp::vec_t<6, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::int32* var_14;
    wp::vec_t<3, wp::float32>* var_15;
    wp::int32 var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::mat_t<3, 3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    //---------
    // forward
    // def _velocimeter(                                                                      <L 909>
    // bodyid = site_bodyid[objid]                                                            <L 922>
    var_0 = wp::address(var_site_bodyid, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // pos = site_xpos_in[worldid, objid]                                                     <L 923>
    var_3 = wp::address(var_site_xpos_in, var_worldid, var_objid);
    var_5 = wp::load(var_3);
    var_4 = wp::copy(var_5);
    // rot = site_xmat_in[worldid, objid]                                                     <L 924>
    var_6 = wp::address(var_site_xmat_in, var_worldid, var_objid);
    var_8 = wp::load(var_6);
    var_7 = wp::copy(var_8);
    // cvel = cvel_in[worldid, bodyid]                                                        <L 925>
    var_9 = wp::address(var_cvel_in, var_worldid, var_1);
    var_11 = wp::load(var_9);
    var_10 = wp::copy(var_11);
    // ang = wp.spatial_top(cvel)                                                             <L 926>
    var_12 = wp::spatial_top(var_10);
    // lin = wp.spatial_bottom(cvel)                                                          <L 927>
    var_13 = wp::spatial_bottom(var_10);
    // subtree_com = subtree_com_in[worldid, body_rootid[bodyid]]                             <L 928>
    var_14 = wp::address(var_body_rootid, var_1);
    var_16 = wp::load(var_14);
    var_15 = wp::address(var_subtree_com_in, var_worldid, var_16);
    var_18 = wp::load(var_15);
    var_17 = wp::copy(var_18);
    // dif = pos - subtree_com                                                                <L 929>
    var_19 = wp::sub(var_4, var_17);
    // return wp.transpose(rot) @ (lin - wp.cross(dif, ang))                                  <L 930>
    var_20 = wp::transpose(var_7);
    var_21 = wp::cross(var_19, var_12);
    var_22 = wp::sub(var_13, var_21);
    var_23 = wp::mul(var_20, var_22);
    return var_23;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:933
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _gyro_0(
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    wp::mat_t<3, 3, wp::float32>* var_3;
    wp::mat_t<3, 3, wp::float32> var_4;
    wp::mat_t<3, 3, wp::float32> var_5;
    wp::vec_t<6, wp::float32>* var_6;
    wp::vec_t<6, wp::float32> var_7;
    wp::vec_t<6, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::mat_t<3, 3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    //---------
    // forward
    // def _gyro(                                                                             <L 934>
    // bodyid = site_bodyid[objid]                                                            <L 944>
    var_0 = wp::address(var_site_bodyid, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // rot = site_xmat_in[worldid, objid]                                                     <L 945>
    var_3 = wp::address(var_site_xmat_in, var_worldid, var_objid);
    var_5 = wp::load(var_3);
    var_4 = wp::copy(var_5);
    // cvel = cvel_in[worldid, bodyid]                                                        <L 946>
    var_6 = wp::address(var_cvel_in, var_worldid, var_1);
    var_8 = wp::load(var_6);
    var_7 = wp::copy(var_8);
    // ang = wp.spatial_top(cvel)                                                             <L 947>
    var_9 = wp::spatial_top(var_7);
    // return wp.transpose(rot) @ ang                                                         <L 948>
    var_10 = wp::transpose(var_4);
    var_11 = wp::mul(var_10, var_9);
    return var_11;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:951
static CUDA_CALLABLE wp::float32 _joint_vel_0(
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::float32> var_qvel_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::float32* var_1;
    wp::int32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    //---------
    // forward
    // def _joint_vel(jnt_dofadr: wp.array[int], qvel_in: wp.array2d[float], worldid: int, objid: int) -> float:       <L 952>
    // return qvel_in[worldid, jnt_dofadr[objid]]                                             <L 953>
    var_0 = wp::address(var_jnt_dofadr, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::address(var_qvel_in, var_worldid, var_2);
    var_4 = wp::load(var_1);
    var_3 = wp::copy(var_4);
    return var_3;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:956
static CUDA_CALLABLE wp::float32 _tendon_vel_0(
    wp::array_t<wp::float32> var_ten_velocity_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::float32* var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    //---------
    // forward
    // def _tendon_vel(ten_velocity_in: wp.array2d[float], worldid: int, objid: int) -> float:       <L 957>
    // return ten_velocity_in[worldid, objid]                                                 <L 958>
    var_0 = wp::address(var_ten_velocity_in, var_worldid, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    return var_1;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:961
static CUDA_CALLABLE wp::float32 _actuator_vel_0(
    wp::array_t<wp::float32> var_actuator_velocity_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::float32* var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    //---------
    // forward
    // def _actuator_vel(actuator_velocity_in: wp.array2d[float], worldid: int, objid: int) -> float:       <L 962>
    // return actuator_velocity_in[worldid, objid]                                            <L 963>
    var_0 = wp::address(var_actuator_velocity_in, var_worldid, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    return var_1;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:966
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _ball_ang_vel_0(
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::float32> var_qvel_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    const wp::int32 var_3 = 0;
    wp::int32 var_4;
    wp::float32* var_5;
    const wp::int32 var_6 = 1;
    wp::int32 var_7;
    wp::float32* var_8;
    const wp::int32 var_9 = 2;
    wp::int32 var_10;
    wp::float32* var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    //---------
    // forward
    // def _ball_ang_vel(jnt_dofadr: wp.array[int], qvel_in: wp.array2d[float], worldid: int, objid: int) -> wp.vec3:       <L 967>
    // adr = jnt_dofadr[objid]                                                                <L 968>
    var_0 = wp::address(var_jnt_dofadr, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // return wp.vec3(qvel_in[worldid, adr + 0], qvel_in[worldid, adr + 1], qvel_in[worldid, adr + 2])       <L 969>
    var_4 = wp::add(var_1, var_3);
    var_5 = wp::address(var_qvel_in, var_worldid, var_4);
    var_7 = wp::add(var_1, var_6);
    var_8 = wp::address(var_qvel_in, var_worldid, var_7);
    var_10 = wp::add(var_1, var_9);
    var_11 = wp::address(var_qvel_in, var_worldid, var_10);
    var_13 = wp::load(var_5);
    var_14 = wp::load(var_8);
    var_15 = wp::load(var_11);
    var_12 = wp::vec_t<3, wp::float32>(var_13, var_14, var_15);
    return var_12;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:1010
static CUDA_CALLABLE void _cvel_offset_0(
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::int32 var_worldid,
    wp::int32 var_objtype,
    wp::int32 var_objid,
    wp::vec_t<6, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    bool var_1;
    wp::vec_t<3, wp::float32>* var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::int32 var_5;
    const wp::int32 var_6 = 2;
    bool var_7;
    wp::vec_t<3, wp::float32>* var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::int32 var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::int32 var_13;
    const wp::int32 var_14 = 5;
    bool var_15;
    wp::vec_t<3, wp::float32>* var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::int32* var_19;
    wp::int32 var_20;
    wp::int32 var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::int32 var_23;
    const wp::int32 var_24 = 6;
    bool var_25;
    wp::vec_t<3, wp::float32>* var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::int32* var_29;
    wp::int32 var_30;
    wp::int32 var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::int32 var_33;
    const wp::int32 var_34 = 7;
    bool var_35;
    wp::vec_t<3, wp::float32>* var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::int32* var_39;
    wp::int32 var_40;
    wp::int32 var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::int32 var_43;
    const wp::float32 var_44 = 0.0;
    wp::vec_t<3, wp::float32> var_45;
    const wp::int32 var_46 = 0;
    wp::vec_t<3, wp::float32> var_47;
    wp::int32 var_48;
    wp::vec_t<3, wp::float32> var_49;
    wp::int32 var_50;
    wp::vec_t<3, wp::float32> var_51;
    wp::int32 var_52;
    wp::vec_t<3, wp::float32> var_53;
    wp::int32 var_54;
    wp::vec_t<3, wp::float32> var_55;
    wp::int32 var_56;
    wp::vec_t<6, wp::float32>* var_57;
    wp::int32* var_58;
    wp::vec_t<3, wp::float32>* var_59;
    wp::int32 var_60;
    wp::vec_t<3, wp::float32> var_61;
    wp::vec_t<3, wp::float32> var_62;
    wp::vec_t<6, wp::float32> var_63;
    wp::vec_t<6, wp::float32> var_64;
    //---------
    // forward
    // def _cvel_offset(                                                                      <L 1011>
    // if objtype == ObjType.BODY:                                                            <L 1030>
    var_1 = (var_objtype == var_0);
    if (var_1) {
        // pos = xipos_in[worldid, objid]                                                     <L 1031>
        var_2 = wp::address(var_xipos_in, var_worldid, var_objid);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // bodyid = objid                                                                     <L 1032>
        var_5 = wp::copy(var_objid);
    }
    if (!var_1) {
        // elif objtype == ObjType.XBODY:                                                     <L 1033>
        var_7 = (var_objtype == var_6);
        if (var_7) {
            // pos = xpos_in[worldid, objid]                                                  <L 1034>
            var_8 = wp::address(var_xpos_in, var_worldid, var_objid);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
            // bodyid = objid                                                                 <L 1035>
            var_11 = wp::copy(var_objid);
        }
        var_12 = wp::where(var_7, var_9, var_3);
        var_13 = wp::where(var_7, var_11, var_5);
        if (!var_7) {
            // elif objtype == ObjType.GEOM:                                                  <L 1036>
            var_15 = (var_objtype == var_14);
            if (var_15) {
                // pos = geom_xpos_in[worldid, objid]                                         <L 1037>
                var_16 = wp::address(var_geom_xpos_in, var_worldid, var_objid);
                var_18 = wp::load(var_16);
                var_17 = wp::copy(var_18);
                // bodyid = geom_bodyid[objid]                                                <L 1038>
                var_19 = wp::address(var_geom_bodyid, var_objid);
                var_21 = wp::load(var_19);
                var_20 = wp::copy(var_21);
            }
            var_22 = wp::where(var_15, var_17, var_12);
            var_23 = wp::where(var_15, var_20, var_13);
            if (!var_15) {
                // elif objtype == ObjType.SITE:                                              <L 1039>
                var_25 = (var_objtype == var_24);
                if (var_25) {
                    // pos = site_xpos_in[worldid, objid]                                     <L 1040>
                    var_26 = wp::address(var_site_xpos_in, var_worldid, var_objid);
                    var_28 = wp::load(var_26);
                    var_27 = wp::copy(var_28);
                    // bodyid = site_bodyid[objid]                                            <L 1041>
                    var_29 = wp::address(var_site_bodyid, var_objid);
                    var_31 = wp::load(var_29);
                    var_30 = wp::copy(var_31);
                }
                var_32 = wp::where(var_25, var_27, var_22);
                var_33 = wp::where(var_25, var_30, var_23);
                if (!var_25) {
                    // elif objtype == ObjType.CAMERA:                                        <L 1042>
                    var_35 = (var_objtype == var_34);
                    if (var_35) {
                        // pos = cam_xpos_in[worldid, objid]                                  <L 1043>
                        var_36 = wp::address(var_cam_xpos_in, var_worldid, var_objid);
                        var_38 = wp::load(var_36);
                        var_37 = wp::copy(var_38);
                        // bodyid = cam_bodyid[objid]                                         <L 1044>
                        var_39 = wp::address(var_cam_bodyid, var_objid);
                        var_41 = wp::load(var_39);
                        var_40 = wp::copy(var_41);
                    }
                    var_42 = wp::where(var_35, var_37, var_32);
                    var_43 = wp::where(var_35, var_40, var_33);
                    if (!var_35) {
                        // pos = wp.vec3(0.0)                                                 <L 1046>
                        var_45 = wp::vec_t<3, wp::float32>(var_44);
                        // bodyid = 0                                                         <L 1047>
                    }
                    var_47 = wp::where(var_35, var_42, var_45);
                    var_48 = wp::where(var_35, var_43, var_46);
                }
                var_49 = wp::where(var_25, var_32, var_47);
                var_50 = wp::where(var_25, var_33, var_48);
            }
            var_51 = wp::where(var_15, var_22, var_49);
            var_52 = wp::where(var_15, var_23, var_50);
        }
        var_53 = wp::where(var_7, var_12, var_51);
        var_54 = wp::where(var_7, var_13, var_52);
    }
    var_55 = wp::where(var_1, var_3, var_53);
    var_56 = wp::where(var_1, var_5, var_54);
    // return cvel_in[worldid, bodyid], pos - subtree_com_in[worldid, body_rootid[bodyid]]       <L 1049>
    var_57 = wp::address(var_cvel_in, var_worldid, var_56);
    var_58 = wp::address(var_body_rootid, var_56);
    var_60 = wp::load(var_58);
    var_59 = wp::address(var_subtree_com_in, var_worldid, var_60);
    var_62 = wp::load(var_59);
    var_61 = wp::sub(var_55, var_62);
    var_64 = wp::load(var_57);
    var_63 = wp::copy(var_64);
    ret_0 = var_63;
    ret_1 = var_61;
    return;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:1052
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _frame_linvel_0(
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::int32 var_objtype,
    wp::int32 var_refid,
    wp::int32 var_reftype)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    bool var_1;
    wp::vec_t<3, wp::float32>* var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    const wp::int32 var_5 = 2;
    bool var_6;
    wp::vec_t<3, wp::float32>* var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    const wp::int32 var_11 = 5;
    bool var_12;
    wp::vec_t<3, wp::float32>* var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32> var_16;
    const wp::int32 var_17 = 6;
    bool var_18;
    wp::vec_t<3, wp::float32>* var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    const wp::int32 var_23 = 7;
    bool var_24;
    wp::vec_t<3, wp::float32>* var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    const wp::float32 var_29 = 0.0;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::vec_t<3, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    wp::vec_t<3, wp::float32> var_35;
    const wp::int32 var_36 = 1;
    bool var_37;
    wp::vec_t<3, wp::float32>* var_38;
    wp::vec_t<3, wp::float32> var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::mat_t<3, 3, wp::float32>* var_41;
    wp::mat_t<3, 3, wp::float32> var_42;
    wp::mat_t<3, 3, wp::float32> var_43;
    const wp::int32 var_44 = 2;
    bool var_45;
    wp::vec_t<3, wp::float32>* var_46;
    wp::vec_t<3, wp::float32> var_47;
    wp::vec_t<3, wp::float32> var_48;
    wp::mat_t<3, 3, wp::float32>* var_49;
    wp::mat_t<3, 3, wp::float32> var_50;
    wp::mat_t<3, 3, wp::float32> var_51;
    wp::vec_t<3, wp::float32> var_52;
    wp::mat_t<3, 3, wp::float32> var_53;
    const wp::int32 var_54 = 5;
    bool var_55;
    wp::vec_t<3, wp::float32>* var_56;
    wp::vec_t<3, wp::float32> var_57;
    wp::vec_t<3, wp::float32> var_58;
    wp::mat_t<3, 3, wp::float32>* var_59;
    wp::mat_t<3, 3, wp::float32> var_60;
    wp::mat_t<3, 3, wp::float32> var_61;
    wp::vec_t<3, wp::float32> var_62;
    wp::mat_t<3, 3, wp::float32> var_63;
    const wp::int32 var_64 = 6;
    bool var_65;
    wp::vec_t<3, wp::float32>* var_66;
    wp::vec_t<3, wp::float32> var_67;
    wp::vec_t<3, wp::float32> var_68;
    wp::mat_t<3, 3, wp::float32>* var_69;
    wp::mat_t<3, 3, wp::float32> var_70;
    wp::mat_t<3, 3, wp::float32> var_71;
    wp::vec_t<3, wp::float32> var_72;
    wp::mat_t<3, 3, wp::float32> var_73;
    const wp::int32 var_74 = 7;
    bool var_75;
    wp::vec_t<3, wp::float32>* var_76;
    wp::vec_t<3, wp::float32> var_77;
    wp::vec_t<3, wp::float32> var_78;
    wp::mat_t<3, 3, wp::float32>* var_79;
    wp::mat_t<3, 3, wp::float32> var_80;
    wp::mat_t<3, 3, wp::float32> var_81;
    wp::vec_t<3, wp::float32> var_82;
    wp::mat_t<3, 3, wp::float32> var_83;
    const wp::float32 var_84 = 0.0;
    wp::vec_t<3, wp::float32> var_85;
    const wp::int32 var_86 = 3;
    wp::mat_t<3, 3, wp::float32> var_87;
    wp::vec_t<3, wp::float32> var_88;
    wp::mat_t<3, 3, wp::float32> var_89;
    wp::vec_t<3, wp::float32> var_90;
    wp::mat_t<3, 3, wp::float32> var_91;
    wp::vec_t<3, wp::float32> var_92;
    wp::mat_t<3, 3, wp::float32> var_93;
    wp::vec_t<3, wp::float32> var_94;
    wp::mat_t<3, 3, wp::float32> var_95;
    wp::vec_t<3, wp::float32> var_96;
    wp::mat_t<3, 3, wp::float32> var_97;
    wp::vec_t<6, wp::float32> var_98;
    wp::vec_t<3, wp::float32> var_99;
    wp::vec_t<6, wp::float32> var_100;
    wp::vec_t<3, wp::float32> var_101;
    wp::vec_t<3, wp::float32> var_102;
    wp::vec_t<3, wp::float32> var_103;
    wp::vec_t<3, wp::float32> var_104;
    wp::vec_t<3, wp::float32> var_105;
    wp::vec_t<3, wp::float32> var_106;
    const wp::int32 var_107 = 1;
    const wp::int32 var_108 = -1;
    bool var_109;
    wp::vec_t<3, wp::float32> var_110;
    wp::vec_t<3, wp::float32> var_111;
    wp::vec_t<3, wp::float32> var_112;
    wp::vec_t<3, wp::float32> var_113;
    wp::vec_t<3, wp::float32> var_114;
    wp::vec_t<3, wp::float32> var_115;
    wp::vec_t<3, wp::float32> var_116;
    wp::mat_t<3, 3, wp::float32> var_117;
    wp::vec_t<3, wp::float32> var_118;
    //---------
    // forward
    // def _frame_linvel(                                                                     <L 1053>
    // if objtype == ObjType.BODY:                                                            <L 1079>
    var_1 = (var_objtype == var_0);
    if (var_1) {
        // xpos = xipos_in[worldid, objid]                                                    <L 1080>
        var_2 = wp::address(var_xipos_in, var_worldid, var_objid);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
    }
    if (!var_1) {
        // elif objtype == ObjType.XBODY:                                                     <L 1081>
        var_6 = (var_objtype == var_5);
        if (var_6) {
            // xpos = xpos_in[worldid, objid]                                                 <L 1082>
            var_7 = wp::address(var_xpos_in, var_worldid, var_objid);
            var_9 = wp::load(var_7);
            var_8 = wp::copy(var_9);
        }
        var_10 = wp::where(var_6, var_8, var_3);
        if (!var_6) {
            // elif objtype == ObjType.GEOM:                                                  <L 1083>
            var_12 = (var_objtype == var_11);
            if (var_12) {
                // xpos = geom_xpos_in[worldid, objid]                                        <L 1084>
                var_13 = wp::address(var_geom_xpos_in, var_worldid, var_objid);
                var_15 = wp::load(var_13);
                var_14 = wp::copy(var_15);
            }
            var_16 = wp::where(var_12, var_14, var_10);
            if (!var_12) {
                // elif objtype == ObjType.SITE:                                              <L 1085>
                var_18 = (var_objtype == var_17);
                if (var_18) {
                    // xpos = site_xpos_in[worldid, objid]                                    <L 1086>
                    var_19 = wp::address(var_site_xpos_in, var_worldid, var_objid);
                    var_21 = wp::load(var_19);
                    var_20 = wp::copy(var_21);
                }
                var_22 = wp::where(var_18, var_20, var_16);
                if (!var_18) {
                    // elif objtype == ObjType.CAMERA:                                        <L 1087>
                    var_24 = (var_objtype == var_23);
                    if (var_24) {
                        // xpos = cam_xpos_in[worldid, objid]                                 <L 1088>
                        var_25 = wp::address(var_cam_xpos_in, var_worldid, var_objid);
                        var_27 = wp::load(var_25);
                        var_26 = wp::copy(var_27);
                    }
                    var_28 = wp::where(var_24, var_26, var_22);
                    if (!var_24) {
                        // xpos = wp.vec3(0.0)                                                <L 1090>
                        var_30 = wp::vec_t<3, wp::float32>(var_29);
                    }
                    var_31 = wp::where(var_24, var_28, var_30);
                }
                var_32 = wp::where(var_18, var_22, var_31);
            }
            var_33 = wp::where(var_12, var_16, var_32);
        }
        var_34 = wp::where(var_6, var_10, var_33);
    }
    var_35 = wp::where(var_1, var_3, var_34);
    // if reftype == ObjType.BODY:                                                            <L 1092>
    var_37 = (var_reftype == var_36);
    if (var_37) {
        // xposref = xipos_in[worldid, refid]                                                 <L 1093>
        var_38 = wp::address(var_xipos_in, var_worldid, var_refid);
        var_40 = wp::load(var_38);
        var_39 = wp::copy(var_40);
        // xmatref = ximat_in[worldid, refid]                                                 <L 1094>
        var_41 = wp::address(var_ximat_in, var_worldid, var_refid);
        var_43 = wp::load(var_41);
        var_42 = wp::copy(var_43);
    }
    if (!var_37) {
        // elif reftype == ObjType.XBODY:                                                     <L 1095>
        var_45 = (var_reftype == var_44);
        if (var_45) {
            // xposref = xpos_in[worldid, refid]                                              <L 1096>
            var_46 = wp::address(var_xpos_in, var_worldid, var_refid);
            var_48 = wp::load(var_46);
            var_47 = wp::copy(var_48);
            // xmatref = xmat_in[worldid, refid]                                              <L 1097>
            var_49 = wp::address(var_xmat_in, var_worldid, var_refid);
            var_51 = wp::load(var_49);
            var_50 = wp::copy(var_51);
        }
        var_52 = wp::where(var_45, var_47, var_39);
        var_53 = wp::where(var_45, var_50, var_42);
        if (!var_45) {
            // elif reftype == ObjType.GEOM:                                                  <L 1098>
            var_55 = (var_reftype == var_54);
            if (var_55) {
                // xposref = geom_xpos_in[worldid, refid]                                     <L 1099>
                var_56 = wp::address(var_geom_xpos_in, var_worldid, var_refid);
                var_58 = wp::load(var_56);
                var_57 = wp::copy(var_58);
                // xmatref = geom_xmat_in[worldid, refid]                                     <L 1100>
                var_59 = wp::address(var_geom_xmat_in, var_worldid, var_refid);
                var_61 = wp::load(var_59);
                var_60 = wp::copy(var_61);
            }
            var_62 = wp::where(var_55, var_57, var_52);
            var_63 = wp::where(var_55, var_60, var_53);
            if (!var_55) {
                // elif reftype == ObjType.SITE:                                              <L 1101>
                var_65 = (var_reftype == var_64);
                if (var_65) {
                    // xposref = site_xpos_in[worldid, refid]                                 <L 1102>
                    var_66 = wp::address(var_site_xpos_in, var_worldid, var_refid);
                    var_68 = wp::load(var_66);
                    var_67 = wp::copy(var_68);
                    // xmatref = site_xmat_in[worldid, refid]                                 <L 1103>
                    var_69 = wp::address(var_site_xmat_in, var_worldid, var_refid);
                    var_71 = wp::load(var_69);
                    var_70 = wp::copy(var_71);
                }
                var_72 = wp::where(var_65, var_67, var_62);
                var_73 = wp::where(var_65, var_70, var_63);
                if (!var_65) {
                    // elif reftype == ObjType.CAMERA:                                        <L 1104>
                    var_75 = (var_reftype == var_74);
                    if (var_75) {
                        // xposref = cam_xpos_in[worldid, refid]                              <L 1105>
                        var_76 = wp::address(var_cam_xpos_in, var_worldid, var_refid);
                        var_78 = wp::load(var_76);
                        var_77 = wp::copy(var_78);
                        // xmatref = cam_xmat_in[worldid, refid]                              <L 1106>
                        var_79 = wp::address(var_cam_xmat_in, var_worldid, var_refid);
                        var_81 = wp::load(var_79);
                        var_80 = wp::copy(var_81);
                    }
                    var_82 = wp::where(var_75, var_77, var_72);
                    var_83 = wp::where(var_75, var_80, var_73);
                    if (!var_75) {
                        // xposref = wp.vec3(0.0)                                             <L 1108>
                        var_85 = wp::vec_t<3, wp::float32>(var_84);
                        // xmatref = wp.identity(3, dtype=float)                              <L 1109>
                        var_87 = wp::identity<3, wp::float32>();
                    }
                    var_88 = wp::where(var_75, var_82, var_85);
                    var_89 = wp::where(var_75, var_83, var_87);
                }
                var_90 = wp::where(var_65, var_72, var_88);
                var_91 = wp::where(var_65, var_73, var_89);
            }
            var_92 = wp::where(var_55, var_62, var_90);
            var_93 = wp::where(var_55, var_63, var_91);
        }
        var_94 = wp::where(var_45, var_52, var_92);
        var_95 = wp::where(var_45, var_53, var_93);
    }
    var_96 = wp::where(var_37, var_39, var_94);
    var_97 = wp::where(var_37, var_42, var_95);
    // cvel, offset = _cvel_offset(                                                           <L 1111>
    // body_rootid,                                                                           <L 1112>
    // geom_bodyid,                                                                           <L 1113>
    // site_bodyid,                                                                           <L 1114>
    // cam_bodyid,                                                                            <L 1115>
    // xpos_in,                                                                               <L 1116>
    // xipos_in,                                                                              <L 1117>
    // geom_xpos_in,                                                                          <L 1118>
    // site_xpos_in,                                                                          <L 1119>
    // cam_xpos_in,                                                                           <L 1120>
    // subtree_com_in,                                                                        <L 1121>
    // cvel_in,                                                                               <L 1122>
    // worldid,                                                                               <L 1123>
    // objtype,                                                                               <L 1124>
    // objid,                                                                                 <L 1125>
    _cvel_offset_0(var_body_rootid, var_geom_bodyid, var_site_bodyid, var_cam_bodyid, var_xpos_in, var_xipos_in, var_geom_xpos_in, var_site_xpos_in, var_cam_xpos_in, var_subtree_com_in, var_cvel_in, var_worldid, var_objtype, var_objid, var_98, var_99);
    // cvelref, offsetref = _cvel_offset(                                                     <L 1127>
    // body_rootid,                                                                           <L 1128>
    // geom_bodyid,                                                                           <L 1129>
    // site_bodyid,                                                                           <L 1130>
    // cam_bodyid,                                                                            <L 1131>
    // xpos_in,                                                                               <L 1132>
    // xipos_in,                                                                              <L 1133>
    // geom_xpos_in,                                                                          <L 1134>
    // site_xpos_in,                                                                          <L 1135>
    // cam_xpos_in,                                                                           <L 1136>
    // subtree_com_in,                                                                        <L 1137>
    // cvel_in,                                                                               <L 1138>
    // worldid,                                                                               <L 1139>
    // reftype,                                                                               <L 1140>
    // refid,                                                                                 <L 1141>
    _cvel_offset_0(var_body_rootid, var_geom_bodyid, var_site_bodyid, var_cam_bodyid, var_xpos_in, var_xipos_in, var_geom_xpos_in, var_site_xpos_in, var_cam_xpos_in, var_subtree_com_in, var_cvel_in, var_worldid, var_reftype, var_refid, var_100, var_101);
    // clinvel = wp.spatial_bottom(cvel)                                                      <L 1143>
    var_102 = wp::spatial_bottom(var_98);
    // cangvel = wp.spatial_top(cvel)                                                         <L 1144>
    var_103 = wp::spatial_top(var_98);
    // cangvelref = wp.spatial_top(cvelref)                                                   <L 1145>
    var_104 = wp::spatial_top(var_100);
    // xlinvel = clinvel - wp.cross(offset, cangvel)                                          <L 1146>
    var_105 = wp::cross(var_99, var_103);
    var_106 = wp::sub(var_102, var_105);
    // if refid > -1:                                                                         <L 1148>
    var_109 = (var_refid > var_108);
    if (var_109) {
        // clinvelref = wp.spatial_bottom(cvelref)                                            <L 1149>
        var_110 = wp::spatial_bottom(var_100);
        // xlinvelref = clinvelref - wp.cross(offsetref, cangvelref)                          <L 1150>
        var_111 = wp::cross(var_101, var_104);
        var_112 = wp::sub(var_110, var_111);
        // rvec = xpos - xposref                                                              <L 1151>
        var_113 = wp::sub(var_35, var_96);
        // rel_vel = xlinvel - xlinvelref + wp.cross(rvec, cangvelref)                        <L 1152>
        var_114 = wp::sub(var_106, var_112);
        var_115 = wp::cross(var_113, var_104);
        var_116 = wp::add(var_114, var_115);
        // return wp.transpose(xmatref) @ rel_vel                                             <L 1153>
        var_117 = wp::transpose(var_97);
        var_118 = wp::mul(var_117, var_116);
        return var_118;
    }
    if (!var_109) {
        // return xlinvel                                                                     <L 1155>
        return var_106;
    }
    return {};
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:1158
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _frame_angvel_0(
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::int32 var_objtype,
    wp::int32 var_refid,
    wp::int32 var_reftype)
{
    //---------
    // primal vars
    wp::vec_t<6, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    const wp::int32 var_3 = 1;
    const wp::int32 var_4 = -1;
    bool var_5;
    const wp::int32 var_6 = 1;
    bool var_7;
    wp::mat_t<3, 3, wp::float32>* var_8;
    wp::mat_t<3, 3, wp::float32> var_9;
    wp::mat_t<3, 3, wp::float32> var_10;
    const wp::int32 var_11 = 2;
    bool var_12;
    wp::mat_t<3, 3, wp::float32>* var_13;
    wp::mat_t<3, 3, wp::float32> var_14;
    wp::mat_t<3, 3, wp::float32> var_15;
    wp::mat_t<3, 3, wp::float32> var_16;
    const wp::int32 var_17 = 5;
    bool var_18;
    wp::mat_t<3, 3, wp::float32>* var_19;
    wp::mat_t<3, 3, wp::float32> var_20;
    wp::mat_t<3, 3, wp::float32> var_21;
    wp::mat_t<3, 3, wp::float32> var_22;
    const wp::int32 var_23 = 6;
    bool var_24;
    wp::mat_t<3, 3, wp::float32>* var_25;
    wp::mat_t<3, 3, wp::float32> var_26;
    wp::mat_t<3, 3, wp::float32> var_27;
    wp::mat_t<3, 3, wp::float32> var_28;
    const wp::int32 var_29 = 7;
    bool var_30;
    wp::mat_t<3, 3, wp::float32>* var_31;
    wp::mat_t<3, 3, wp::float32> var_32;
    wp::mat_t<3, 3, wp::float32> var_33;
    wp::mat_t<3, 3, wp::float32> var_34;
    const wp::int32 var_35 = 3;
    wp::mat_t<3, 3, wp::float32> var_36;
    wp::mat_t<3, 3, wp::float32> var_37;
    wp::mat_t<3, 3, wp::float32> var_38;
    wp::mat_t<3, 3, wp::float32> var_39;
    wp::mat_t<3, 3, wp::float32> var_40;
    wp::mat_t<3, 3, wp::float32> var_41;
    wp::vec_t<6, wp::float32> var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::mat_t<3, 3, wp::float32> var_45;
    wp::vec_t<3, wp::float32> var_46;
    wp::vec_t<3, wp::float32> var_47;
    wp::vec_t<3, wp::float32> var_48;
    //---------
    // forward
    // def _frame_angvel(                                                                     <L 1159>
    // cvel, _ = _cvel_offset(                                                                <L 1185>
    // body_rootid,                                                                           <L 1186>
    // geom_bodyid,                                                                           <L 1187>
    // site_bodyid,                                                                           <L 1188>
    // cam_bodyid,                                                                            <L 1189>
    // xpos_in,                                                                               <L 1190>
    // xipos_in,                                                                              <L 1191>
    // geom_xpos_in,                                                                          <L 1192>
    // site_xpos_in,                                                                          <L 1193>
    // cam_xpos_in,                                                                           <L 1194>
    // subtree_com_in,                                                                        <L 1195>
    // cvel_in,                                                                               <L 1196>
    // worldid,                                                                               <L 1197>
    // objtype,                                                                               <L 1198>
    // objid,                                                                                 <L 1199>
    _cvel_offset_0(var_body_rootid, var_geom_bodyid, var_site_bodyid, var_cam_bodyid, var_xpos_in, var_xipos_in, var_geom_xpos_in, var_site_xpos_in, var_cam_xpos_in, var_subtree_com_in, var_cvel_in, var_worldid, var_objtype, var_objid, var_0, var_1);
    // cangvel = wp.spatial_top(cvel)                                                         <L 1201>
    var_2 = wp::spatial_top(var_0);
    // if refid > -1:                                                                         <L 1203>
    var_5 = (var_refid > var_4);
    if (var_5) {
        // if reftype == ObjType.BODY:                                                        <L 1204>
        var_7 = (var_reftype == var_6);
        if (var_7) {
            // xmatref = ximat_in[worldid, refid]                                             <L 1205>
            var_8 = wp::address(var_ximat_in, var_worldid, var_refid);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
        }
        if (!var_7) {
            // elif reftype == ObjType.XBODY:                                                 <L 1206>
            var_12 = (var_reftype == var_11);
            if (var_12) {
                // xmatref = xmat_in[worldid, refid]                                          <L 1207>
                var_13 = wp::address(var_xmat_in, var_worldid, var_refid);
                var_15 = wp::load(var_13);
                var_14 = wp::copy(var_15);
            }
            var_16 = wp::where(var_12, var_14, var_9);
            if (!var_12) {
                // elif reftype == ObjType.GEOM:                                              <L 1208>
                var_18 = (var_reftype == var_17);
                if (var_18) {
                    // xmatref = geom_xmat_in[worldid, refid]                                 <L 1209>
                    var_19 = wp::address(var_geom_xmat_in, var_worldid, var_refid);
                    var_21 = wp::load(var_19);
                    var_20 = wp::copy(var_21);
                }
                var_22 = wp::where(var_18, var_20, var_16);
                if (!var_18) {
                    // elif reftype == ObjType.SITE:                                          <L 1210>
                    var_24 = (var_reftype == var_23);
                    if (var_24) {
                        // xmatref = site_xmat_in[worldid, refid]                             <L 1211>
                        var_25 = wp::address(var_site_xmat_in, var_worldid, var_refid);
                        var_27 = wp::load(var_25);
                        var_26 = wp::copy(var_27);
                    }
                    var_28 = wp::where(var_24, var_26, var_22);
                    if (!var_24) {
                        // elif reftype == ObjType.CAMERA:                                    <L 1212>
                        var_30 = (var_reftype == var_29);
                        if (var_30) {
                            // xmatref = cam_xmat_in[worldid, refid]                          <L 1213>
                            var_31 = wp::address(var_cam_xmat_in, var_worldid, var_refid);
                            var_33 = wp::load(var_31);
                            var_32 = wp::copy(var_33);
                        }
                        var_34 = wp::where(var_30, var_32, var_28);
                        if (!var_30) {
                            // xmatref = wp.identity(3, dtype=float)                          <L 1215>
                            var_36 = wp::identity<3, wp::float32>();
                        }
                        var_37 = wp::where(var_30, var_34, var_36);
                    }
                    var_38 = wp::where(var_24, var_28, var_37);
                }
                var_39 = wp::where(var_18, var_22, var_38);
            }
            var_40 = wp::where(var_12, var_16, var_39);
        }
        var_41 = wp::where(var_7, var_9, var_40);
        // cvelref, _ = _cvel_offset(                                                         <L 1217>
        // body_rootid,                                                                       <L 1218>
        // geom_bodyid,                                                                       <L 1219>
        // site_bodyid,                                                                       <L 1220>
        // cam_bodyid,                                                                        <L 1221>
        // xpos_in,                                                                           <L 1222>
        // xipos_in,                                                                          <L 1223>
        // geom_xpos_in,                                                                      <L 1224>
        // site_xpos_in,                                                                      <L 1225>
        // cam_xpos_in,                                                                       <L 1226>
        // subtree_com_in,                                                                    <L 1227>
        // cvel_in,                                                                           <L 1228>
        // worldid,                                                                           <L 1229>
        // reftype,                                                                           <L 1230>
        // refid,                                                                             <L 1231>
        _cvel_offset_0(var_body_rootid, var_geom_bodyid, var_site_bodyid, var_cam_bodyid, var_xpos_in, var_xipos_in, var_geom_xpos_in, var_site_xpos_in, var_cam_xpos_in, var_subtree_com_in, var_cvel_in, var_worldid, var_reftype, var_refid, var_42, var_43);
        // cangvelref = wp.spatial_top(cvelref)                                               <L 1233>
        var_44 = wp::spatial_top(var_42);
        // return wp.transpose(xmatref) @ (cangvel - cangvelref)                              <L 1235>
        var_45 = wp::transpose(var_41);
        var_46 = wp::sub(var_2, var_44);
        var_47 = wp::mul(var_45, var_46);
        return var_47;
    }
    var_48 = wp::where(var_5, var_43, var_1);
    if (!var_5) {
        // return cangvel                                                                     <L 1237>
        return var_2;
    }
    return {};
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:1240
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _subtree_linvel_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_linvel_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32>* var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    //---------
    // forward
    // def _subtree_linvel(subtree_linvel_in: wp.array2d[wp.vec3], worldid: int, objid: int) -> wp.vec3:       <L 1241>
    // return subtree_linvel_in[worldid, objid]                                               <L 1242>
    var_0 = wp::address(var_subtree_linvel_in, var_worldid, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    return var_1;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:1245
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _subtree_angmom_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_angmom_in,
    wp::int32 var_worldid,
    wp::int32 var_objid)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32>* var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    //---------
    // forward
    // def _subtree_angmom(subtree_angmom_in: wp.array2d[wp.vec3], worldid: int, objid: int) -> wp.vec3:       <L 1246>
    // return subtree_angmom_in[worldid, objid]                                               <L 1247>
    var_0 = wp::address(var_subtree_angmom_in, var_worldid, var_objid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    return var_1;
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:113
static CUDA_CALLABLE void adj__magnetometer_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_opt_magnetic,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_opt_magnetic,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_site_xmat_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:0
static CUDA_CALLABLE void adj__write_vector_0(
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::int32 var_sensorid,
    wp::int32 var_sensordim,
    wp::vec_t<3, wp::float32> var_sensor,
    wp::array_t<wp::float32> var_out,
    wp::array_t<wp::int32> & adj_sensor_type,
    wp::array_t<wp::int32> & adj_sensor_datatype,
    wp::array_t<wp::int32> & adj_sensor_adr,
    wp::array_t<wp::float32> & adj_sensor_cutoff,
    wp::int32 & adj_sensorid,
    wp::int32 & adj_sensordim,
    wp::vec_t<3, wp::float32> & adj_sensor,
    wp::array_t<wp::float32> & adj_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:127
static CUDA_CALLABLE void adj__cam_projection_0(
    wp::array_t<wp::float32> var_cam_fovy,
    wp::array_t<wp::vec_t<2, wp::int32>> var_cam_resolution,
    wp::array_t<wp::vec_t<2, wp::float32>> var_cam_sensorsize,
    wp::array_t<wp::vec_t<4, wp::float32>> var_cam_intrinsic,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::int32 var_refid,
    wp::array_t<wp::float32> & adj_cam_fovy,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_cam_resolution,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_cam_sensorsize,
    wp::array_t<wp::vec_t<4, wp::float32>> & adj_cam_intrinsic,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_site_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_cam_xmat_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::int32 & adj_refid,
    wp::vec_t<2, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:0
static CUDA_CALLABLE void adj__write_vector_0(
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::int32 var_sensorid,
    wp::int32 var_sensordim,
    wp::vec_t<2, wp::float32> var_sensor,
    wp::array_t<wp::float32> var_out,
    wp::array_t<wp::int32> & adj_sensor_type,
    wp::array_t<wp::int32> & adj_sensor_datatype,
    wp::array_t<wp::int32> & adj_sensor_adr,
    wp::array_t<wp::float32> & adj_sensor_cutoff,
    wp::int32 & adj_sensorid,
    wp::int32 & adj_sensordim,
    wp::vec_t<2, wp::float32> & adj_sensor,
    wp::array_t<wp::float32> & adj_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:0
static CUDA_CALLABLE void adj__write_scalar_0(
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::int32 var_sensorid,
    wp::float32 var_sensor,
    wp::array_t<wp::float32> var_out,
    wp::array_t<wp::int32> & adj_sensor_type,
    wp::array_t<wp::int32> & adj_sensor_datatype,
    wp::array_t<wp::int32> & adj_sensor_adr,
    wp::array_t<wp::float32> & adj_sensor_cutoff,
    wp::int32 & adj_sensorid,
    wp::float32 & adj_sensor,
    wp::array_t<wp::float32> & adj_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:216
static CUDA_CALLABLE void adj__joint_pos_0(
    wp::array_t<wp::int32> var_jnt_qposadr,
    wp::array_t<wp::float32> var_qpos_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::int32> & adj_jnt_qposadr,
    wp::array_t<wp::float32> & adj_qpos_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:221
static CUDA_CALLABLE void adj__tendon_pos_0(
    wp::array_t<wp::float32> var_ten_length_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::float32> & adj_ten_length_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:226
static CUDA_CALLABLE void adj__actuator_pos_0(
    wp::array_t<wp::float32> var_actuator_length_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::float32> & adj_actuator_length_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:231
static CUDA_CALLABLE void adj__ball_quat_0(
    wp::array_t<wp::int32> var_jnt_qposadr,
    wp::array_t<wp::float32> var_qpos_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::int32> & adj_jnt_qposadr,
    wp::array_t<wp::float32> & adj_qpos_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:0
static CUDA_CALLABLE void adj__write_vector_0(
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::int32 var_sensorid,
    wp::int32 var_sensordim,
    wp::quat_t<wp::float32> var_sensor,
    wp::array_t<wp::float32> var_out,
    wp::array_t<wp::int32> & adj_sensor_type,
    wp::array_t<wp::int32> & adj_sensor_datatype,
    wp::array_t<wp::int32> & adj_sensor_adr,
    wp::array_t<wp::float32> & adj_sensor_cutoff,
    wp::int32 & adj_sensorid,
    wp::int32 & adj_sensordim,
    wp::quat_t<wp::float32> & adj_sensor,
    wp::array_t<wp::float32> & adj_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:281
static CUDA_CALLABLE void adj__frame_pos_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::int32 var_objtype,
    wp::int32 var_refid,
    wp::int32 var_reftype,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_cam_xmat_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::int32 & adj_objtype,
    wp::int32 & adj_refid,
    wp::int32 & adj_reftype,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:340
static CUDA_CALLABLE void adj__frame_axis_0(
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::int32 var_objtype,
    wp::int32 var_refid,
    wp::int32 var_reftype,
    wp::int32 var_frame_axis,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_xmat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_ximat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_geom_xmat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_site_xmat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_cam_xmat_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::int32 & adj_objtype,
    wp::int32 & adj_refid,
    wp::int32 & adj_reftype,
    wp::int32 & adj_frame_axis,
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:115
static CUDA_CALLABLE void adj_quat_inv_0(
    wp::quat_t<wp::float32> var_quat,
    wp::quat_t<wp::float32> & adj_quat,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:393
static CUDA_CALLABLE void adj__frame_quat_0(
    wp::array_t<wp::quat_t<wp::float32>> var_body_iquat,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> var_geom_quat,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> var_site_quat,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> var_cam_quat,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::int32 var_objtype,
    wp::int32 var_refid,
    wp::int32 var_reftype,
    wp::array_t<wp::quat_t<wp::float32>> & adj_body_iquat,
    wp::array_t<wp::int32> & adj_geom_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> & adj_geom_quat,
    wp::array_t<wp::int32> & adj_site_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> & adj_site_quat,
    wp::array_t<wp::int32> & adj_cam_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> & adj_cam_quat,
    wp::array_t<wp::quat_t<wp::float32>> & adj_xquat_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::int32 & adj_objtype,
    wp::int32 & adj_refid,
    wp::int32 & adj_reftype,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:448
static CUDA_CALLABLE void adj__subtree_com_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_subtree_com_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/math.py:322
static CUDA_CALLABLE void adj_upper_tri_index_0(
    wp::int32 var_n,
    wp::int32 var_i,
    wp::int32 var_j,
    wp::int32 & adj_n,
    wp::int32 & adj_i,
    wp::int32 & adj_j,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:0
static CUDA_CALLABLE void adj__write_vector_0(
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::int32 var_sensorid,
    wp::int32 var_sensordim,
    wp::vec_t<6, wp::float32> var_sensor,
    wp::array_t<wp::float32> var_out,
    wp::array_t<wp::int32> & adj_sensor_type,
    wp::array_t<wp::int32> & adj_sensor_datatype,
    wp::array_t<wp::int32> & adj_sensor_adr,
    wp::array_t<wp::float32> & adj_sensor_cutoff,
    wp::int32 & adj_sensorid,
    wp::int32 & adj_sensordim,
    wp::vec_t<6, wp::float32> & adj_sensor,
    wp::array_t<wp::float32> & adj_out)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/util_misc.py:602
static CUDA_CALLABLE void adj_inside_geom_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::int32 var_geomtype,
    wp::vec_t<3, wp::float32> var_point,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::int32 & adj_geomtype,
    wp::vec_t<3, wp::float32> & adj_point,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:453
static CUDA_CALLABLE void adj__clock_0(
    wp::array_t<wp::float32> var_time_in,
    wp::int32 var_worldid,
    wp::array_t<wp::float32> & adj_time_in,
    wp::int32 & adj_worldid,
    wp::float32 & adj_ret)
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:1450
static CUDA_CALLABLE void adj__accelerometer_0(
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cacc_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::int32> & adj_body_rootid,
    wp::array_t<wp::int32> & adj_site_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cacc_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:1482
static CUDA_CALLABLE void adj__force_0(
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_int_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::int32> & adj_site_bodyid,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_site_xmat_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cfrc_int_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:1499
static CUDA_CALLABLE void adj__torque_0(
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_int_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::int32> & adj_body_rootid,
    wp::array_t<wp::int32> & adj_site_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cfrc_int_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:1520
static CUDA_CALLABLE void adj__actuator_force_0(
    wp::array_t<wp::float32> var_actuator_force_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::float32> & adj_actuator_force_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:1525
static CUDA_CALLABLE void adj__joint_actuator_force_0(
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::float32> var_qfrc_actuator_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::int32> & adj_jnt_dofadr,
    wp::array_t<wp::float32> & adj_qfrc_actuator_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:1618
static CUDA_CALLABLE void adj__framelinacc_0(
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cacc_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::int32 var_objtype,
    wp::array_t<wp::int32> & adj_body_rootid,
    wp::array_t<wp::int32> & adj_geom_bodyid,
    wp::array_t<wp::int32> & adj_site_bodyid,
    wp::array_t<wp::int32> & adj_cam_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_site_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_cam_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cacc_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::int32 & adj_objtype,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:1669
static CUDA_CALLABLE void adj__frameangacc_0(
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cacc_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::int32 var_objtype,
    wp::array_t<wp::int32> & adj_geom_bodyid,
    wp::array_t<wp::int32> & adj_site_bodyid,
    wp::array_t<wp::int32> & adj_cam_bodyid,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cacc_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::int32 & adj_objtype,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:2255
static CUDA_CALLABLE void adj__check_match_0(
    wp::array_t<wp::int32> var_body_parentid,
    wp::int32 var_body,
    wp::int32 var_geom,
    wp::int32 var_objtype,
    wp::int32 var_objid,
    wp::array_t<wp::int32> & adj_body_parentid,
    wp::int32 & adj_body,
    wp::int32 & adj_geom,
    wp::int32 & adj_objtype,
    wp::int32 & adj_objid,
    bool & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_sdf.py:80
static CUDA_CALLABLE void adj_get_sdf_params_0(
    wp::array_t<wp::vec_t<8, wp::int32>> var_oct_child,
    wp::array_t<wp::vec_t<3, wp::float32>> var_oct_aabb,
    wp::array_t<wp::vec_t<8, wp::float32>> var_oct_coeff,
    wp::array_t<wp::int32> var_mesh_octadr,
    wp::array_t<wp::int32> var_plugin,
    wp::array_t<wp::vec_t<128, wp::float32>> var_plugin_attr,
    wp::int32 var_g_type,
    wp::vec_t<3, wp::float32> var_g_size,
    wp::int32 var_plugin_id,
    wp::int32 var_mesh_id,
    wp::vec_t<128, wp::float32> & ret_0,
    wp::int32 & ret_1,
    VolumeData_53ac1a2d & ret_2,
    MeshData_52eaa0fa & ret_3,
    wp::array_t<wp::vec_t<8, wp::int32>> & adj_oct_child,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_oct_aabb,
    wp::array_t<wp::vec_t<8, wp::float32>> & adj_oct_coeff,
    wp::array_t<wp::int32> & adj_mesh_octadr,
    wp::array_t<wp::int32> & adj_plugin,
    wp::array_t<wp::vec_t<128, wp::float32>> & adj_plugin_attr,
    wp::int32 & adj_g_type,
    wp::vec_t<3, wp::float32> & adj_g_size,
    wp::int32 & adj_plugin_id,
    wp::int32 & adj_mesh_id,
    wp::vec_t<128, wp::float32> & adj_ret_0,
    wp::int32 & adj_ret_1,
    VolumeData_53ac1a2d & adj_ret_2,
    MeshData_52eaa0fa & adj_ret_3)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_sdf.py:157
static CUDA_CALLABLE void adj_sphere_0(
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> & adj_p,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_sdf.py:147
static CUDA_CALLABLE void adj_radial_field_0(
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> var_x,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> & adj_a,
    wp::vec_t<3, wp::float32> & adj_x,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_sdf.py:162
static CUDA_CALLABLE void adj_box_0(
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> & adj_p,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_sdf.py:174
static CUDA_CALLABLE void adj_ellipsoid_0(
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> & adj_p,
    wp::vec_t<3, wp::float32> & adj_size,
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/ray.py:105
static CUDA_CALLABLE void adj__ray_quad_0(
    wp::float32 var_a,
    wp::float32 var_b,
    wp::float32 var_c,
    wp::float32 & ret_0,
    wp::vec_t<2, wp::float32> & ret_1,
    wp::float32 & adj_a,
    wp::float32 & adj_b,
    wp::float32 & adj_c,
    wp::float32 & adj_ret_0,
    wp::vec_t<2, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/ray.py:211
static CUDA_CALLABLE void adj_ray_sphere_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::float32 var_dist_sqr,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::float32 & adj_dist_sqr,
    wp::vec_t<3, wp::float32> & adj_pnt,
    wp::vec_t<3, wp::float32> & adj_vec,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/ray.py:32
static CUDA_CALLABLE void adj__ray_map_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_pnt,
    wp::vec_t<3, wp::float32> & adj_vec,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/ray.py:397
static CUDA_CALLABLE void adj_ray_box_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 & ret_0,
    wp::vec_t<6, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::vec_t<3, wp::float32> & adj_pnt,
    wp::vec_t<3, wp::float32> & adj_vec,
    wp::float32 & adj_ret_0,
    wp::vec_t<6, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/ray.py:128
static CUDA_CALLABLE void adj__ray_triangle_1(
    wp::vec_t<3, wp::float32> var_v0,
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::vec_t<3, wp::float32> var_b0,
    wp::vec_t<3, wp::float32> var_b1,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_v0,
    wp::vec_t<3, wp::float32> & adj_v1,
    wp::vec_t<3, wp::float32> & adj_v2,
    wp::vec_t<3, wp::float32> & adj_pnt,
    wp::vec_t<3, wp::float32> & adj_vec,
    wp::vec_t<3, wp::float32> & adj_b0,
    wp::vec_t<3, wp::float32> & adj_b1,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/ray.py:622
static CUDA_CALLABLE void adj_ray_mesh_0(
    wp::int32 var_nmeshface,
    wp::array_t<wp::int32> var_mesh_vertadr,
    wp::array_t<wp::int32> var_mesh_faceadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_vert,
    wp::array_t<wp::vec_t<3, wp::int32>> var_mesh_face,
    wp::int32 var_data_id,
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::int32 & adj_nmeshface,
    wp::array_t<wp::int32> & adj_mesh_vertadr,
    wp::array_t<wp::int32> & adj_mesh_faceadr,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_mesh_vert,
    wp::array_t<wp::vec_t<3, wp::int32>> & adj_mesh_face,
    wp::int32 & adj_data_id,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::vec_t<3, wp::float32> & adj_pnt,
    wp::vec_t<3, wp::float32> & adj_vec,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_sdf.py:325
static CUDA_CALLABLE void adj_box_project_0(
    wp::vec_t<3, wp::float32> var_center,
    wp::vec_t<3, wp::float32> var_half_size,
    wp::vec_t<3, wp::float32> var_xyz,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_center,
    wp::vec_t<3, wp::float32> & adj_half_size,
    wp::vec_t<3, wp::float32> & adj_xyz,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_sdf.py:253
static CUDA_CALLABLE void adj_find_oct_0(
    wp::array_t<wp::vec_t<8, wp::int32>> var_oct_child,
    wp::array_t<wp::vec_t<3, wp::float32>> var_oct_aabb,
    wp::vec_t<3, wp::float32> var_p,
    bool var_grad,
    wp::int32 var_root,
    wp::int32 & ret_0,
    wp::tuple_t<wp::vec_t<8, wp::float32>, wp::vec_t<8, wp::float32>, wp::vec_t<8, wp::float32>> & ret_1,
    wp::array_t<wp::vec_t<8, wp::int32>> & adj_oct_child,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_oct_aabb,
    wp::vec_t<3, wp::float32> & adj_p,
    bool & adj_grad,
    wp::int32 & adj_root,
    wp::int32 & adj_ret_0,
    wp::tuple_t<wp::vec_t<8, wp::float32>, wp::vec_t<8, wp::float32>, wp::vec_t<8, wp::float32>> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_sdf.py:362
static CUDA_CALLABLE void adj_sample_volume_sdf_0(
    wp::vec_t<3, wp::float32> var_xyz,
    VolumeData_53ac1a2d var_volume_data,
    wp::vec_t<3, wp::float32> & adj_xyz,
    VolumeData_53ac1a2d & adj_volume_data,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_sdf.py:233
static CUDA_CALLABLE void adj_user_sdf_0(
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<128, wp::float32> var_attr,
    wp::int32 var_sdf_type,
    wp::vec_t<3, wp::float32> & adj_p,
    wp::vec_t<128, wp::float32> & adj_attr,
    wp::int32 & adj_sdf_type,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/collision_sdf.py:389
static CUDA_CALLABLE void adj_sdf_0(
    wp::int32 var_type,
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<128, wp::float32> var_attr,
    wp::int32 var_sdf_type,
    VolumeData_53ac1a2d var_volume_data,
    MeshData_52eaa0fa var_mesh_data,
    wp::int32 & adj_type,
    wp::vec_t<3, wp::float32> & adj_p,
    wp::vec_t<128, wp::float32> & adj_attr,
    wp::int32 & adj_sdf_type,
    VolumeData_53ac1a2d & adj_volume_data,
    MeshData_52eaa0fa & adj_mesh_data,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:2079
static CUDA_CALLABLE void adj__transform_spatial_0(
    wp::vec_t<6, wp::float32> var_vec,
    wp::vec_t<3, wp::float32> var_dif,
    wp::vec_t<6, wp::float32> & adj_vec,
    wp::vec_t<3, wp::float32> & adj_dif,
    wp::vec_t<3, wp::float32> & adj_ret)
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


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/ray.py:187
static CUDA_CALLABLE void adj_ray_plane_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::vec_t<3, wp::float32> & adj_pnt,
    wp::vec_t<3, wp::float32> & adj_vec,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/ray.py:228
static CUDA_CALLABLE void adj_ray_capsule_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::vec_t<3, wp::float32> & adj_pnt,
    wp::vec_t<3, wp::float32> & adj_vec,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/ray.py:302
static CUDA_CALLABLE void adj_ray_ellipsoid_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::vec_t<3, wp::float32> & adj_pnt,
    wp::vec_t<3, wp::float32> & adj_vec,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/ray.py:333
static CUDA_CALLABLE void adj_ray_cylinder_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::vec_t<3, wp::float32> & adj_pnt,
    wp::vec_t<3, wp::float32> & adj_vec,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/ray.py:799
static CUDA_CALLABLE void adj_ray_geom_0(
    wp::vec_t<3, wp::float32> var_pos,
    wp::mat_t<3, 3, wp::float32> var_mat,
    wp::vec_t<3, wp::float32> var_size,
    wp::vec_t<3, wp::float32> var_pnt,
    wp::vec_t<3, wp::float32> var_vec,
    wp::int32 var_geomtype,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_mat,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::vec_t<3, wp::float32> & adj_pnt,
    wp::vec_t<3, wp::float32> & adj_vec,
    wp::int32 & adj_geomtype,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:908
static CUDA_CALLABLE void adj__velocimeter_0(
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::int32> & adj_body_rootid,
    wp::array_t<wp::int32> & adj_site_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cvel_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:933
static CUDA_CALLABLE void adj__gyro_0(
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::int32> & adj_site_bodyid,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_site_xmat_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cvel_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:951
static CUDA_CALLABLE void adj__joint_vel_0(
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::float32> var_qvel_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::int32> & adj_jnt_dofadr,
    wp::array_t<wp::float32> & adj_qvel_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:956
static CUDA_CALLABLE void adj__tendon_vel_0(
    wp::array_t<wp::float32> var_ten_velocity_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::float32> & adj_ten_velocity_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:961
static CUDA_CALLABLE void adj__actuator_vel_0(
    wp::array_t<wp::float32> var_actuator_velocity_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::float32> & adj_actuator_velocity_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:966
static CUDA_CALLABLE void adj__ball_ang_vel_0(
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::float32> var_qvel_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::int32> & adj_jnt_dofadr,
    wp::array_t<wp::float32> & adj_qvel_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:1010
static CUDA_CALLABLE void adj__cvel_offset_0(
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::int32 var_worldid,
    wp::int32 var_objtype,
    wp::int32 var_objid,
    wp::vec_t<6, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::array_t<wp::int32> & adj_body_rootid,
    wp::array_t<wp::int32> & adj_geom_bodyid,
    wp::array_t<wp::int32> & adj_site_bodyid,
    wp::array_t<wp::int32> & adj_cam_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_site_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_cam_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cvel_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objtype,
    wp::int32 & adj_objid,
    wp::vec_t<6, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:1052
static CUDA_CALLABLE void adj__frame_linvel_0(
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::int32 var_objtype,
    wp::int32 var_refid,
    wp::int32 var_reftype,
    wp::array_t<wp::int32> & adj_body_rootid,
    wp::array_t<wp::int32> & adj_geom_bodyid,
    wp::array_t<wp::int32> & adj_site_bodyid,
    wp::array_t<wp::int32> & adj_cam_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_cam_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cvel_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::int32 & adj_objtype,
    wp::int32 & adj_refid,
    wp::int32 & adj_reftype,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:1158
static CUDA_CALLABLE void adj__frame_angvel_0(
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::int32 var_objtype,
    wp::int32 var_refid,
    wp::int32 var_reftype,
    wp::array_t<wp::int32> & adj_body_rootid,
    wp::array_t<wp::int32> & adj_geom_bodyid,
    wp::array_t<wp::int32> & adj_site_bodyid,
    wp::array_t<wp::int32> & adj_cam_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_cam_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cvel_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::int32 & adj_objtype,
    wp::int32 & adj_refid,
    wp::int32 & adj_reftype,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:1240
static CUDA_CALLABLE void adj__subtree_linvel_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_linvel_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_subtree_linvel_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/spark-advantage/rek-training/gpu-runtime-20260910/deps/mujoco_warp/_src/sensor.py:1245
static CUDA_CALLABLE void adj__subtree_angmom_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_angmom_in,
    wp::int32 var_worldid,
    wp::int32 var_objid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_subtree_angmom_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_objid,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void _energy_pos_zero_5de26636_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_energy_out)
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
        wp::vec_t<2, wp::float32>* var_2;
        const wp::int32 var_3 = 0;
        wp::float32* var_4;
        //---------
        // forward
        // def _energy_pos_zero(                                                                  <L 2701>
        // worldid = wp.tid()                                                                     <L 2705>
        var_0 = builtin_tid1d();
        // energy_out[worldid][0] = 0.0                                                           <L 2706>
        var_2 = wp::address(var_energy_out, var_0);
        var_4 = wp::indexref(var_2, var_3);
        wp::store(var_4, var_1);
    }
}



extern "C" __global__ void _sensor_pos_b2db0eec_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_ngeom,
    wp::array_t<wp::vec_t<3, wp::float32>> var_opt_magnetic,
    wp::array_t<wp::int32> var_body_geomnum,
    wp::array_t<wp::int32> var_body_geomadr,
    wp::array_t<wp::quat_t<wp::float32>> var_body_iquat,
    wp::array_t<wp::int32> var_jnt_qposadr,
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> var_geom_quat,
    wp::array_t<wp::int32> var_site_type,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_size,
    wp::array_t<wp::quat_t<wp::float32>> var_site_quat,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::quat_t<wp::float32>> var_cam_quat,
    wp::array_t<wp::float32> var_cam_fovy,
    wp::array_t<wp::vec_t<2, wp::int32>> var_cam_resolution,
    wp::array_t<wp::vec_t<2, wp::float32>> var_cam_sensorsize,
    wp::array_t<wp::vec_t<4, wp::float32>> var_cam_intrinsic,
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_objtype,
    wp::array_t<wp::int32> var_sensor_objid,
    wp::array_t<wp::int32> var_sensor_reftype,
    wp::array_t<wp::int32> var_sensor_refid,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::array_t<wp::vec_t<2, wp::int32>> var_nxn_pairid,
    wp::array_t<wp::int32> var_sensor_pos_adr,
    wp::array_t<wp::int32> var_rangefinder_sensor_adr,
    wp::array_t<wp::float32> var_time_in,
    wp::array_t<wp::vec_t<2, wp::float32>> var_energy_in,
    wp::array_t<wp::float32> var_qpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::float32> var_ten_length_in,
    wp::array_t<wp::float32> var_actuator_length_in,
    wp::array_t<wp::float32> var_rangefinder_dist_in,
    wp::array_t<wp::float32> var_sensor_collision_in,
    wp::array_t<wp::float32> var_sensordata_out)
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
        wp::slice_t var_11;
        const wp::int32 var_12 = 0;
        wp::array_t<wp::float32> var_13;
        const wp::int32 var_14 = 6;
        bool var_15;
        wp::vec_t<3, wp::float32> var_16;
        const wp::int32 var_17 = 3;
        const wp::int32 var_18 = 8;
        bool var_19;
        wp::int32* var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        wp::vec_t<2, wp::float32> var_23;
        const wp::int32 var_24 = 2;
        const wp::int32 var_25 = 7;
        bool var_26;
        wp::int32* var_27;
        wp::float32* var_28;
        wp::int32 var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        const wp::int32 var_32 = 9;
        bool var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        const wp::int32 var_36 = 11;
        bool var_37;
        wp::float32 var_38;
        wp::float32 var_39;
        const wp::int32 var_40 = 13;
        bool var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        const wp::int32 var_44 = 18;
        bool var_45;
        wp::quat_t<wp::float32> var_46;
        const wp::int32 var_47 = 4;
        const wp::int32 var_48 = 26;
        bool var_49;
        wp::int32* var_50;
        wp::int32 var_51;
        wp::int32 var_52;
        wp::int32* var_53;
        wp::int32 var_54;
        wp::int32 var_55;
        wp::int32* var_56;
        wp::int32 var_57;
        wp::int32 var_58;
        wp::vec_t<3, wp::float32> var_59;
        const wp::int32 var_60 = 3;
        wp::vec_t<3, wp::float32> var_61;
        wp::int32 var_62;
        const wp::int32 var_63 = 28;
        bool var_64;
        const wp::int32 var_65 = 29;
        bool var_66;
        const wp::int32 var_67 = 30;
        bool var_68;
        bool var_69;
        wp::int32* var_70;
        wp::int32 var_71;
        wp::int32 var_72;
        wp::int32* var_73;
        wp::int32 var_74;
        wp::int32 var_75;
        wp::int32* var_76;
        wp::int32 var_77;
        wp::int32 var_78;
        const wp::int32 var_79 = 28;
        bool var_80;
        const wp::int32 var_81 = 0;
        const wp::int32 var_82 = 29;
        bool var_83;
        const wp::int32 var_84 = 1;
        wp::int32 var_85;
        const wp::int32 var_86 = 30;
        bool var_87;
        const wp::int32 var_88 = 2;
        wp::int32 var_89;
        wp::int32 var_90;
        wp::int32 var_91;
        wp::vec_t<3, wp::float32> var_92;
        const wp::int32 var_93 = 3;
        wp::vec_t<3, wp::float32> var_94;
        wp::int32 var_95;
        wp::int32 var_96;
        wp::int32 var_97;
        const wp::int32 var_98 = 27;
        bool var_99;
        wp::int32* var_100;
        wp::int32 var_101;
        wp::int32 var_102;
        wp::int32* var_103;
        wp::int32 var_104;
        wp::int32 var_105;
        wp::int32* var_106;
        wp::int32 var_107;
        wp::int32 var_108;
        wp::quat_t<wp::float32> var_109;
        const wp::int32 var_110 = 4;
        wp::int32 var_111;
        wp::quat_t<wp::float32> var_112;
        wp::int32 var_113;
        wp::int32 var_114;
        const wp::int32 var_115 = 35;
        bool var_116;
        wp::vec_t<3, wp::float32> var_117;
        const wp::int32 var_118 = 3;
        wp::vec_t<3, wp::float32> var_119;
        const wp::int32 var_120 = 39;
        bool var_121;
        const wp::int32 var_122 = 40;
        bool var_123;
        const wp::int32 var_124 = 41;
        bool var_125;
        bool var_126;
        wp::int32* var_127;
        wp::int32 var_128;
        wp::int32 var_129;
        wp::int32* var_130;
        wp::int32 var_131;
        wp::int32 var_132;
        wp::int32* var_133;
        wp::int32 var_134;
        wp::int32 var_135;
        wp::int32* var_136;
        wp::int32 var_137;
        wp::int32 var_138;
        wp::float32* var_139;
        wp::float32 var_140;
        wp::float32 var_141;
        const wp::float32 var_142 = 0.0;
        const wp::float32 var_143 = 0.0;
        const wp::float32 var_144 = 0.0;
        const wp::float32 var_145 = 0.0;
        const wp::float32 var_146 = 0.0;
        const wp::float32 var_147 = 0.0;
        wp::vec_t<6, wp::float32> var_148;
        const bool var_149 = false;
        bool var_150;
        const wp::int32 var_151 = 1;
        const wp::int32 var_152 = 1;
        wp::int32 var_153;
        bool var_154;
        wp::int32* var_155;
        wp::int32 var_156;
        wp::int32 var_157;
        wp::int32* var_158;
        wp::int32 var_159;
        wp::int32 var_160;
        const wp::int32 var_161 = 1;
        wp::int32 var_162;
        wp::int32 var_163;
        wp::int32 var_164;
        const wp::int32 var_165 = 1;
        const wp::int32 var_166 = 1;
        wp::int32 var_167;
        bool var_168;
        wp::int32* var_169;
        wp::int32 var_170;
        wp::int32 var_171;
        wp::int32* var_172;
        wp::int32 var_173;
        wp::int32 var_174;
        const wp::int32 var_175 = 1;
        wp::int32 var_176;
        wp::int32 var_177;
        wp::int32 var_178;
        wp::range_t var_179;
        wp::int32 var_180;
        wp::int32 var_181;
        wp::range_t var_182;
        wp::int32 var_183;
        wp::int32 var_184;
        bool var_185;
        wp::int32 var_186;
        wp::int32 var_187;
        wp::int32 var_188;
        wp::vec_t<2, wp::int32>* var_189;
        const wp::int32 var_190 = 1;
        wp::int32 var_191;
        wp::vec_t<2, wp::int32> var_192;
        const wp::int32 var_193 = 0;
        const wp::int32 var_194 = 0;
        wp::float32* var_195;
        wp::float32 var_196;
        wp::float32 var_197;
        bool var_198;
        wp::float32 var_199;
        const wp::int32 var_200 = 40;
        bool var_201;
        const wp::int32 var_202 = 41;
        bool var_203;
        bool var_204;
        const wp::int32 var_205 = 1;
        wp::float32* var_206;
        const wp::int32 var_207 = 2;
        wp::float32* var_208;
        const wp::int32 var_209 = 3;
        wp::float32* var_210;
        const wp::int32 var_211 = 4;
        wp::float32* var_212;
        const wp::int32 var_213 = 5;
        wp::float32* var_214;
        const wp::int32 var_215 = 6;
        wp::float32* var_216;
        wp::vec_t<6, wp::float32> var_217;
        wp::float32 var_218;
        wp::float32 var_219;
        wp::float32 var_220;
        wp::float32 var_221;
        wp::float32 var_222;
        wp::float32 var_223;
        wp::vec_t<6, wp::float32> var_224;
        wp::int32* var_225;
        wp::int32* var_226;
        bool var_227;
        wp::int32 var_228;
        wp::int32 var_229;
        const bool var_230 = true;
        bool var_231;
        wp::int32* var_232;
        wp::int32* var_233;
        bool var_234;
        wp::int32 var_235;
        wp::int32 var_236;
        bool var_237;
        bool var_238;
        const bool var_239 = false;
        bool var_240;
        bool var_241;
        wp::float32 var_242;
        wp::vec_t<6, wp::float32> var_243;
        bool var_244;
        const wp::int32 var_245 = 1;
        const wp::int32 var_246 = 0;
        wp::float32* var_247;
        wp::float32 var_248;
        wp::float32 var_249;
        bool var_250;
        wp::float32 var_251;
        const wp::int32 var_252 = 40;
        bool var_253;
        const wp::int32 var_254 = 41;
        bool var_255;
        bool var_256;
        const wp::int32 var_257 = 1;
        wp::float32* var_258;
        const wp::int32 var_259 = 2;
        wp::float32* var_260;
        const wp::int32 var_261 = 3;
        wp::float32* var_262;
        const wp::int32 var_263 = 4;
        wp::float32* var_264;
        const wp::int32 var_265 = 5;
        wp::float32* var_266;
        const wp::int32 var_267 = 6;
        wp::float32* var_268;
        wp::vec_t<6, wp::float32> var_269;
        wp::float32 var_270;
        wp::float32 var_271;
        wp::float32 var_272;
        wp::float32 var_273;
        wp::float32 var_274;
        wp::float32 var_275;
        wp::vec_t<6, wp::float32> var_276;
        wp::int32* var_277;
        wp::int32* var_278;
        bool var_279;
        wp::int32 var_280;
        wp::int32 var_281;
        const bool var_282 = true;
        bool var_283;
        wp::int32* var_284;
        wp::int32* var_285;
        bool var_286;
        wp::int32 var_287;
        wp::int32 var_288;
        bool var_289;
        bool var_290;
        const bool var_291 = false;
        bool var_292;
        bool var_293;
        wp::float32 var_294;
        wp::vec_t<6, wp::float32> var_295;
        bool var_296;
        const wp::int32 var_297 = 2;
        const wp::int32 var_298 = 0;
        wp::float32* var_299;
        wp::float32 var_300;
        wp::float32 var_301;
        bool var_302;
        wp::float32 var_303;
        const wp::int32 var_304 = 40;
        bool var_305;
        const wp::int32 var_306 = 41;
        bool var_307;
        bool var_308;
        const wp::int32 var_309 = 1;
        wp::float32* var_310;
        const wp::int32 var_311 = 2;
        wp::float32* var_312;
        const wp::int32 var_313 = 3;
        wp::float32* var_314;
        const wp::int32 var_315 = 4;
        wp::float32* var_316;
        const wp::int32 var_317 = 5;
        wp::float32* var_318;
        const wp::int32 var_319 = 6;
        wp::float32* var_320;
        wp::vec_t<6, wp::float32> var_321;
        wp::float32 var_322;
        wp::float32 var_323;
        wp::float32 var_324;
        wp::float32 var_325;
        wp::float32 var_326;
        wp::float32 var_327;
        wp::vec_t<6, wp::float32> var_328;
        wp::int32* var_329;
        wp::int32* var_330;
        bool var_331;
        wp::int32 var_332;
        wp::int32 var_333;
        const bool var_334 = true;
        bool var_335;
        wp::int32* var_336;
        wp::int32* var_337;
        bool var_338;
        wp::int32 var_339;
        wp::int32 var_340;
        bool var_341;
        bool var_342;
        const bool var_343 = false;
        bool var_344;
        bool var_345;
        wp::float32 var_346;
        wp::vec_t<6, wp::float32> var_347;
        bool var_348;
        const wp::int32 var_349 = 3;
        const wp::int32 var_350 = 0;
        wp::float32* var_351;
        wp::float32 var_352;
        wp::float32 var_353;
        bool var_354;
        wp::float32 var_355;
        const wp::int32 var_356 = 40;
        bool var_357;
        const wp::int32 var_358 = 41;
        bool var_359;
        bool var_360;
        const wp::int32 var_361 = 1;
        wp::float32* var_362;
        const wp::int32 var_363 = 2;
        wp::float32* var_364;
        const wp::int32 var_365 = 3;
        wp::float32* var_366;
        const wp::int32 var_367 = 4;
        wp::float32* var_368;
        const wp::int32 var_369 = 5;
        wp::float32* var_370;
        const wp::int32 var_371 = 6;
        wp::float32* var_372;
        wp::vec_t<6, wp::float32> var_373;
        wp::float32 var_374;
        wp::float32 var_375;
        wp::float32 var_376;
        wp::float32 var_377;
        wp::float32 var_378;
        wp::float32 var_379;
        wp::vec_t<6, wp::float32> var_380;
        wp::int32* var_381;
        wp::int32* var_382;
        bool var_383;
        wp::int32 var_384;
        wp::int32 var_385;
        const bool var_386 = true;
        bool var_387;
        wp::int32* var_388;
        wp::int32* var_389;
        bool var_390;
        wp::int32 var_391;
        wp::int32 var_392;
        bool var_393;
        bool var_394;
        const bool var_395 = false;
        bool var_396;
        bool var_397;
        wp::float32 var_398;
        wp::vec_t<6, wp::float32> var_399;
        bool var_400;
        const wp::int32 var_401 = 4;
        const wp::int32 var_402 = 0;
        wp::float32* var_403;
        wp::float32 var_404;
        wp::float32 var_405;
        bool var_406;
        wp::float32 var_407;
        const wp::int32 var_408 = 40;
        bool var_409;
        const wp::int32 var_410 = 41;
        bool var_411;
        bool var_412;
        const wp::int32 var_413 = 1;
        wp::float32* var_414;
        const wp::int32 var_415 = 2;
        wp::float32* var_416;
        const wp::int32 var_417 = 3;
        wp::float32* var_418;
        const wp::int32 var_419 = 4;
        wp::float32* var_420;
        const wp::int32 var_421 = 5;
        wp::float32* var_422;
        const wp::int32 var_423 = 6;
        wp::float32* var_424;
        wp::vec_t<6, wp::float32> var_425;
        wp::float32 var_426;
        wp::float32 var_427;
        wp::float32 var_428;
        wp::float32 var_429;
        wp::float32 var_430;
        wp::float32 var_431;
        wp::vec_t<6, wp::float32> var_432;
        wp::int32* var_433;
        wp::int32* var_434;
        bool var_435;
        wp::int32 var_436;
        wp::int32 var_437;
        const bool var_438 = true;
        bool var_439;
        wp::int32* var_440;
        wp::int32* var_441;
        bool var_442;
        wp::int32 var_443;
        wp::int32 var_444;
        bool var_445;
        bool var_446;
        const bool var_447 = false;
        bool var_448;
        bool var_449;
        wp::float32 var_450;
        wp::vec_t<6, wp::float32> var_451;
        bool var_452;
        const wp::int32 var_453 = 5;
        const wp::int32 var_454 = 0;
        wp::float32* var_455;
        wp::float32 var_456;
        wp::float32 var_457;
        bool var_458;
        wp::float32 var_459;
        const wp::int32 var_460 = 40;
        bool var_461;
        const wp::int32 var_462 = 41;
        bool var_463;
        bool var_464;
        const wp::int32 var_465 = 1;
        wp::float32* var_466;
        const wp::int32 var_467 = 2;
        wp::float32* var_468;
        const wp::int32 var_469 = 3;
        wp::float32* var_470;
        const wp::int32 var_471 = 4;
        wp::float32* var_472;
        const wp::int32 var_473 = 5;
        wp::float32* var_474;
        const wp::int32 var_475 = 6;
        wp::float32* var_476;
        wp::vec_t<6, wp::float32> var_477;
        wp::float32 var_478;
        wp::float32 var_479;
        wp::float32 var_480;
        wp::float32 var_481;
        wp::float32 var_482;
        wp::float32 var_483;
        wp::vec_t<6, wp::float32> var_484;
        wp::int32* var_485;
        wp::int32* var_486;
        bool var_487;
        wp::int32 var_488;
        wp::int32 var_489;
        const bool var_490 = true;
        bool var_491;
        wp::int32* var_492;
        wp::int32* var_493;
        bool var_494;
        wp::int32 var_495;
        wp::int32 var_496;
        bool var_497;
        bool var_498;
        const bool var_499 = false;
        bool var_500;
        bool var_501;
        wp::float32 var_502;
        wp::vec_t<6, wp::float32> var_503;
        bool var_504;
        const wp::int32 var_505 = 6;
        const wp::int32 var_506 = 0;
        wp::float32* var_507;
        wp::float32 var_508;
        wp::float32 var_509;
        bool var_510;
        wp::float32 var_511;
        const wp::int32 var_512 = 40;
        bool var_513;
        const wp::int32 var_514 = 41;
        bool var_515;
        bool var_516;
        const wp::int32 var_517 = 1;
        wp::float32* var_518;
        const wp::int32 var_519 = 2;
        wp::float32* var_520;
        const wp::int32 var_521 = 3;
        wp::float32* var_522;
        const wp::int32 var_523 = 4;
        wp::float32* var_524;
        const wp::int32 var_525 = 5;
        wp::float32* var_526;
        const wp::int32 var_527 = 6;
        wp::float32* var_528;
        wp::vec_t<6, wp::float32> var_529;
        wp::float32 var_530;
        wp::float32 var_531;
        wp::float32 var_532;
        wp::float32 var_533;
        wp::float32 var_534;
        wp::float32 var_535;
        wp::vec_t<6, wp::float32> var_536;
        wp::int32* var_537;
        wp::int32* var_538;
        bool var_539;
        wp::int32 var_540;
        wp::int32 var_541;
        const bool var_542 = true;
        bool var_543;
        wp::int32* var_544;
        wp::int32* var_545;
        bool var_546;
        wp::int32 var_547;
        wp::int32 var_548;
        bool var_549;
        bool var_550;
        const bool var_551 = false;
        bool var_552;
        bool var_553;
        wp::float32 var_554;
        wp::vec_t<6, wp::float32> var_555;
        bool var_556;
        const wp::int32 var_557 = 7;
        const wp::int32 var_558 = 0;
        wp::float32* var_559;
        wp::float32 var_560;
        wp::float32 var_561;
        bool var_562;
        wp::float32 var_563;
        const wp::int32 var_564 = 40;
        bool var_565;
        const wp::int32 var_566 = 41;
        bool var_567;
        bool var_568;
        const wp::int32 var_569 = 1;
        wp::float32* var_570;
        const wp::int32 var_571 = 2;
        wp::float32* var_572;
        const wp::int32 var_573 = 3;
        wp::float32* var_574;
        const wp::int32 var_575 = 4;
        wp::float32* var_576;
        const wp::int32 var_577 = 5;
        wp::float32* var_578;
        const wp::int32 var_579 = 6;
        wp::float32* var_580;
        wp::vec_t<6, wp::float32> var_581;
        wp::float32 var_582;
        wp::float32 var_583;
        wp::float32 var_584;
        wp::float32 var_585;
        wp::float32 var_586;
        wp::float32 var_587;
        wp::vec_t<6, wp::float32> var_588;
        wp::int32* var_589;
        wp::int32* var_590;
        bool var_591;
        wp::int32 var_592;
        wp::int32 var_593;
        const bool var_594 = true;
        bool var_595;
        wp::int32* var_596;
        wp::int32* var_597;
        bool var_598;
        wp::int32 var_599;
        wp::int32 var_600;
        bool var_601;
        bool var_602;
        const bool var_603 = false;
        bool var_604;
        bool var_605;
        wp::float32 var_606;
        wp::vec_t<6, wp::float32> var_607;
        bool var_608;
        const wp::int32 var_609 = 39;
        const wp::int32 var_610 = 39;
        wp::int32 var_611;
        bool var_612;
        const wp::int32 var_613 = 40;
        const wp::int32 var_614 = 40;
        wp::int32 var_615;
        bool var_616;
        wp::float32* var_617;
        bool var_618;
        wp::float32 var_619;
        const wp::int32 var_620 = 3;
        wp::float32 var_621;
        const wp::int32 var_622 = 0;
        wp::float32 var_623;
        wp::float32 var_624;
        const wp::int32 var_625 = 4;
        wp::float32 var_626;
        const wp::int32 var_627 = 1;
        wp::float32 var_628;
        wp::float32 var_629;
        const wp::int32 var_630 = 5;
        wp::float32 var_631;
        const wp::int32 var_632 = 2;
        wp::float32 var_633;
        wp::float32 var_634;
        wp::vec_t<3, wp::float32> var_635;
        wp::vec_t<3, wp::float32> var_636;
        const wp::float32 var_637 = 1.0;
        const wp::float32 var_638 = -1.0;
        wp::vec_t<3, wp::float32> var_639;
        wp::vec_t<3, wp::float32> var_640;
        const wp::float32 var_641 = 0.0;
        const wp::float32 var_642 = 0.0;
        const wp::float32 var_643 = 0.0;
        wp::vec_t<3, wp::float32> var_644;
        wp::vec_t<3, wp::float32> var_645;
        const wp::int32 var_646 = 3;
        const wp::int32 var_647 = 41;
        const wp::int32 var_648 = 41;
        wp::int32 var_649;
        bool var_650;
        wp::float32* var_651;
        bool var_652;
        wp::float32 var_653;
        const wp::int32 var_654 = 3;
        wp::float32 var_655;
        const wp::int32 var_656 = 4;
        wp::float32 var_657;
        const wp::int32 var_658 = 5;
        wp::float32 var_659;
        const wp::int32 var_660 = 0;
        wp::float32 var_661;
        const wp::int32 var_662 = 1;
        wp::float32 var_663;
        const wp::int32 var_664 = 2;
        wp::float32 var_665;
        wp::vec_t<6, wp::float32> var_666;
        wp::vec_t<6, wp::float32> var_667;
        wp::vec_t<6, wp::float32> var_668;
        const wp::float32 var_669 = 0.0;
        const wp::float32 var_670 = 0.0;
        const wp::float32 var_671 = 0.0;
        const wp::float32 var_672 = 0.0;
        const wp::float32 var_673 = 0.0;
        const wp::float32 var_674 = 0.0;
        wp::vec_t<6, wp::float32> var_675;
        wp::vec_t<6, wp::float32> var_676;
        const wp::int32 var_677 = 6;
        wp::int32 var_678;
        wp::int32 var_679;
        wp::int32 var_680;
        wp::int32 var_681;
        const wp::int32 var_682 = 38;
        bool var_683;
        wp::int32* var_684;
        wp::int32 var_685;
        wp::int32 var_686;
        const wp::int32 var_687 = 2;
        bool var_688;
        wp::vec_t<3, wp::float32>* var_689;
        wp::vec_t<3, wp::float32> var_690;
        wp::vec_t<3, wp::float32> var_691;
        const wp::int32 var_692 = 1;
        bool var_693;
        wp::vec_t<3, wp::float32>* var_694;
        wp::vec_t<3, wp::float32> var_695;
        wp::vec_t<3, wp::float32> var_696;
        wp::vec_t<3, wp::float32> var_697;
        const wp::int32 var_698 = 5;
        bool var_699;
        wp::vec_t<3, wp::float32>* var_700;
        wp::vec_t<3, wp::float32> var_701;
        wp::vec_t<3, wp::float32> var_702;
        wp::vec_t<3, wp::float32> var_703;
        const wp::int32 var_704 = 6;
        bool var_705;
        wp::vec_t<3, wp::float32>* var_706;
        wp::vec_t<3, wp::float32> var_707;
        wp::vec_t<3, wp::float32> var_708;
        wp::vec_t<3, wp::float32> var_709;
        const wp::int32 var_710 = 7;
        bool var_711;
        wp::vec_t<3, wp::float32>* var_712;
        wp::vec_t<3, wp::float32> var_713;
        wp::vec_t<3, wp::float32> var_714;
        wp::vec_t<3, wp::float32> var_715;
        wp::vec_t<3, wp::float32> var_716;
        wp::vec_t<3, wp::float32> var_717;
        wp::vec_t<3, wp::float32> var_718;
        wp::vec_t<3, wp::float32> var_719;
        wp::int32* var_720;
        wp::int32 var_721;
        wp::int32 var_722;
        wp::vec_t<3, wp::float32>* var_723;
        wp::mat_t<3, 3, wp::float32>* var_724;
        wp::vec_t<3, wp::float32>* var_725;
        wp::int32* var_726;
        bool var_727;
        wp::vec_t<3, wp::float32> var_728;
        wp::mat_t<3, 3, wp::float32> var_729;
        wp::vec_t<3, wp::float32> var_730;
        wp::int32 var_731;
        wp::float32 var_732;
        wp::int32 var_733;
        wp::int32 var_734;
        const wp::int32 var_735 = 43;
        bool var_736;
        wp::vec_t<2, wp::float32>* var_737;
        const wp::int32 var_738 = 0;
        wp::float32 var_739;
        wp::vec_t<2, wp::float32> var_740;
        wp::float32 var_741;
        const wp::int32 var_742 = 44;
        bool var_743;
        wp::vec_t<2, wp::float32>* var_744;
        const wp::int32 var_745 = 1;
        wp::float32 var_746;
        wp::vec_t<2, wp::float32> var_747;
        wp::float32 var_748;
        const wp::int32 var_749 = 45;
        bool var_750;
        wp::float32 var_751;
        wp::float32 var_752;
        wp::float32 var_753;
        wp::float32 var_754;
        wp::float32 var_755;
        wp::int32 var_756;
        wp::float32 var_757;
        wp::int32 var_758;
        wp::int32 var_759;
        wp::int32 var_760;
        wp::float32 var_761;
        wp::int32 var_762;
        wp::int32 var_763;
        wp::int32 var_764;
        wp::vec_t<3, wp::float32> var_765;
        wp::int32 var_766;
        wp::float32 var_767;
        wp::int32 var_768;
        wp::int32 var_769;
        wp::int32 var_770;
        wp::vec_t<3, wp::float32> var_771;
        wp::int32 var_772;
        wp::float32 var_773;
        wp::quat_t<wp::float32> var_774;
        wp::int32 var_775;
        wp::int32 var_776;
        wp::int32 var_777;
        wp::vec_t<3, wp::float32> var_778;
        wp::int32 var_779;
        wp::float32 var_780;
        wp::quat_t<wp::float32> var_781;
        wp::int32 var_782;
        wp::int32 var_783;
        wp::int32 var_784;
        wp::vec_t<3, wp::float32> var_785;
        wp::int32 var_786;
        wp::float32 var_787;
        wp::quat_t<wp::float32> var_788;
        wp::int32 var_789;
        wp::vec_t<3, wp::float32> var_790;
        wp::int32 var_791;
        wp::float32 var_792;
        wp::int32 var_793;
        wp::vec_t<3, wp::float32> var_794;
        wp::int32 var_795;
        wp::float32 var_796;
        wp::int32 var_797;
        wp::vec_t<3, wp::float32> var_798;
        wp::int32 var_799;
        wp::float32 var_800;
        wp::int32 var_801;
        wp::vec_t<3, wp::float32> var_802;
        wp::int32 var_803;
        wp::float32 var_804;
        wp::int32 var_805;
        wp::vec_t<3, wp::float32> var_806;
        wp::int32 var_807;
        wp::int32 var_808;
        wp::vec_t<3, wp::float32> var_809;
        //---------
        // forward
        // def _sensor_pos(                                                                       <L 459>
        // worldid, posid = wp.tid()                                                              <L 515>
        builtin_tid2d(var_0, var_1);
        // sensorid = sensor_pos_adr[posid]                                                       <L 516>
        var_2 = wp::address(var_sensor_pos_adr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // sensortype = sensor_type[sensorid]                                                     <L 517>
        var_5 = wp::address(var_sensor_type, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // objid = sensor_objid[sensorid]                                                         <L 518>
        var_8 = wp::address(var_sensor_objid, var_3);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // out = sensordata_out[worldid]                                                          <L 519>
        var_11 = wp::slice_t(var_0, var_0, var_12);
        var_13 = wp::view(var_sensordata_out, var_11);
        // if sensortype == SensorType.MAGNETOMETER:                                              <L 521>
        var_15 = (var_6 == var_14);
        if (var_15) {
            // vec3 = _magnetometer(opt_magnetic, site_xmat_in, worldid, objid)                   <L 522>
            var_16 = _magnetometer_0(var_opt_magnetic, var_site_xmat_in, var_0, var_9);
            // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)       <L 523>
            _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_17, var_16, var_13);
        }
        if (!var_15) {
            // elif sensortype == SensorType.CAMPROJECTION:                                       <L 524>
            var_19 = (var_6 == var_18);
            if (var_19) {
                // refid = sensor_refid[sensorid]                                                 <L 525>
                var_20 = wp::address(var_sensor_refid, var_3);
                var_22 = wp::load(var_20);
                var_21 = wp::copy(var_22);
                // vec2 = _cam_projection(                                                        <L 526>
                // cam_fovy, cam_resolution, cam_sensorsize, cam_intrinsic, site_xpos_in, cam_xpos_in, cam_xmat_in, worldid, objid, refid       <L 527>
                var_23 = _cam_projection_0(var_cam_fovy, var_cam_resolution, var_cam_sensorsize, var_cam_intrinsic, var_site_xpos_in, var_cam_xpos_in, var_cam_xmat_in, var_0, var_9, var_21);
                // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 2, vec2, out)       <L 529>
                _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_24, var_23, var_13);
            }
            if (!var_19) {
                // elif sensortype == SensorType.RANGEFINDER:                                     <L 530>
                var_26 = (var_6 == var_25);
                if (var_26) {
                    // val = rangefinder_dist_in[worldid, rangefinder_sensor_adr[sensorid]]       <L 531>
                    var_27 = wp::address(var_rangefinder_sensor_adr, var_3);
                    var_29 = wp::load(var_27);
                    var_28 = wp::address(var_rangefinder_dist_in, var_0, var_29);
                    var_31 = wp::load(var_28);
                    var_30 = wp::copy(var_31);
                    // _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)       <L 532>
                    _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_30, var_13);
                }
                if (!var_26) {
                    // elif sensortype == SensorType.JOINTPOS:                                    <L 533>
                    var_33 = (var_6 == var_32);
                    if (var_33) {
                        // val = _joint_pos(jnt_qposadr, qpos_in, worldid, objid)                 <L 534>
                        var_34 = _joint_pos_0(var_jnt_qposadr, var_qpos_in, var_0, var_9);
                        // _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)       <L 535>
                        _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_34, var_13);
                    }
                    var_35 = wp::where(var_33, var_34, var_30);
                    if (!var_33) {
                        // elif sensortype == SensorType.TENDONPOS:                               <L 536>
                        var_37 = (var_6 == var_36);
                        if (var_37) {
                            // val = _tendon_pos(ten_length_in, worldid, objid)                   <L 537>
                            var_38 = _tendon_pos_0(var_ten_length_in, var_0, var_9);
                            // _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)       <L 538>
                            _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_38, var_13);
                        }
                        var_39 = wp::where(var_37, var_38, var_35);
                        if (!var_37) {
                            // elif sensortype == SensorType.ACTUATORPOS:                         <L 539>
                            var_41 = (var_6 == var_40);
                            if (var_41) {
                                // val = _actuator_pos(actuator_length_in, worldid, objid)        <L 540>
                                var_42 = _actuator_pos_0(var_actuator_length_in, var_0, var_9);
                                // _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)       <L 541>
                                _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_42, var_13);
                            }
                            var_43 = wp::where(var_41, var_42, var_39);
                            if (!var_41) {
                                // elif sensortype == SensorType.BALLQUAT:                        <L 542>
                                var_45 = (var_6 == var_44);
                                if (var_45) {
                                    // quat = _ball_quat(jnt_qposadr, qpos_in, worldid, objid)       <L 543>
                                    var_46 = _ball_quat_0(var_jnt_qposadr, var_qpos_in, var_0, var_9);
                                    // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 4, quat, out)       <L 544>
                                    _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_47, var_46, var_13);
                                }
                                if (!var_45) {
                                    // elif sensortype == SensorType.FRAMEPOS:                    <L 545>
                                    var_49 = (var_6 == var_48);
                                    if (var_49) {
                                        // objtype = sensor_objtype[sensorid]                     <L 546>
                                        var_50 = wp::address(var_sensor_objtype, var_3);
                                        var_52 = wp::load(var_50);
                                        var_51 = wp::copy(var_52);
                                        // refid = sensor_refid[sensorid]                         <L 547>
                                        var_53 = wp::address(var_sensor_refid, var_3);
                                        var_55 = wp::load(var_53);
                                        var_54 = wp::copy(var_55);
                                        // reftype = sensor_reftype[sensorid]                     <L 548>
                                        var_56 = wp::address(var_sensor_reftype, var_3);
                                        var_58 = wp::load(var_56);
                                        var_57 = wp::copy(var_58);
                                        // vec3 = _frame_pos(                                     <L 549>
                                        // xpos_in,                                               <L 550>
                                        // xmat_in,                                               <L 551>
                                        // xipos_in,                                              <L 552>
                                        // ximat_in,                                              <L 553>
                                        // geom_xpos_in,                                          <L 554>
                                        // geom_xmat_in,                                          <L 555>
                                        // site_xpos_in,                                          <L 556>
                                        // site_xmat_in,                                          <L 557>
                                        // cam_xpos_in,                                           <L 558>
                                        // cam_xmat_in,                                           <L 559>
                                        // worldid,                                               <L 560>
                                        // objid,                                                 <L 561>
                                        // objtype,                                               <L 562>
                                        // refid,                                                 <L 563>
                                        // reftype,                                               <L 564>
                                        var_59 = _frame_pos_0(var_xpos_in, var_xmat_in, var_xipos_in, var_ximat_in, var_geom_xpos_in, var_geom_xmat_in, var_site_xpos_in, var_site_xmat_in, var_cam_xpos_in, var_cam_xmat_in, var_0, var_9, var_51, var_54, var_57);
                                        // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)       <L 566>
                                        _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_60, var_59, var_13);
                                    }
                                    var_61 = wp::where(var_49, var_59, var_16);
                                    var_62 = wp::where(var_49, var_54, var_21);
                                    if (!var_49) {
                                        // elif sensortype == SensorType.FRAMEXAXIS or sensortype == SensorType.FRAMEYAXIS or sensortype == SensorType.FRAMEZAXIS:       <L 567>
                                        var_64 = (var_6 == var_63);
                                        var_66 = (var_6 == var_65);
                                        var_68 = (var_6 == var_67);
                                        var_69 = var_64 || var_66 || var_68;
                                        if (var_69) {
                                            // objtype = sensor_objtype[sensorid]                 <L 568>
                                            var_70 = wp::address(var_sensor_objtype, var_3);
                                            var_72 = wp::load(var_70);
                                            var_71 = wp::copy(var_72);
                                            // refid = sensor_refid[sensorid]                     <L 569>
                                            var_73 = wp::address(var_sensor_refid, var_3);
                                            var_75 = wp::load(var_73);
                                            var_74 = wp::copy(var_75);
                                            // reftype = sensor_reftype[sensorid]                 <L 570>
                                            var_76 = wp::address(var_sensor_reftype, var_3);
                                            var_78 = wp::load(var_76);
                                            var_77 = wp::copy(var_78);
                                            // if sensortype == SensorType.FRAMEXAXIS:            <L 571>
                                            var_80 = (var_6 == var_79);
                                            if (var_80) {
                                                // axis = 0                                       <L 572>
                                            }
                                            if (!var_80) {
                                                // elif sensortype == SensorType.FRAMEYAXIS:       <L 573>
                                                var_83 = (var_6 == var_82);
                                                if (var_83) {
                                                    // axis = 1                                   <L 574>
                                                }
                                                var_85 = wp::where(var_83, var_84, var_81);
                                                if (!var_83) {
                                                    // elif sensortype == SensorType.FRAMEZAXIS:       <L 575>
                                                    var_87 = (var_6 == var_86);
                                                    if (var_87) {
                                                        // axis = 2                               <L 576>
                                                    }
                                                    var_89 = wp::where(var_87, var_88, var_85);
                                                }
                                                var_90 = wp::where(var_83, var_85, var_89);
                                            }
                                            var_91 = wp::where(var_80, var_81, var_90);
                                            // vec3 = _frame_axis(                                <L 577>
                                            // xmat_in, ximat_in, geom_xmat_in, site_xmat_in, cam_xmat_in, worldid, objid, objtype, refid, reftype, axis       <L 578>
                                            var_92 = _frame_axis_0(var_xmat_in, var_ximat_in, var_geom_xmat_in, var_site_xmat_in, var_cam_xmat_in, var_0, var_9, var_71, var_74, var_77, var_91);
                                            // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)       <L 580>
                                            _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_93, var_92, var_13);
                                        }
                                        var_94 = wp::where(var_69, var_92, var_61);
                                        var_95 = wp::where(var_69, var_74, var_62);
                                        var_96 = wp::where(var_69, var_71, var_51);
                                        var_97 = wp::where(var_69, var_77, var_57);
                                        if (!var_69) {
                                            // elif sensortype == SensorType.FRAMEQUAT:           <L 581>
                                            var_99 = (var_6 == var_98);
                                            if (var_99) {
                                                // objtype = sensor_objtype[sensorid]             <L 582>
                                                var_100 = wp::address(var_sensor_objtype, var_3);
                                                var_102 = wp::load(var_100);
                                                var_101 = wp::copy(var_102);
                                                // refid = sensor_refid[sensorid]                 <L 583>
                                                var_103 = wp::address(var_sensor_refid, var_3);
                                                var_105 = wp::load(var_103);
                                                var_104 = wp::copy(var_105);
                                                // reftype = sensor_reftype[sensorid]             <L 584>
                                                var_106 = wp::address(var_sensor_reftype, var_3);
                                                var_108 = wp::load(var_106);
                                                var_107 = wp::copy(var_108);
                                                // quat = _frame_quat(                            <L 585>
                                                // body_iquat,                                    <L 586>
                                                // geom_bodyid,                                   <L 587>
                                                // geom_quat,                                     <L 588>
                                                // site_bodyid,                                   <L 589>
                                                // site_quat,                                     <L 590>
                                                // cam_bodyid,                                    <L 591>
                                                // cam_quat,                                      <L 592>
                                                // xquat_in,                                      <L 593>
                                                // worldid,                                       <L 594>
                                                // objid,                                         <L 595>
                                                // objtype,                                       <L 596>
                                                // refid,                                         <L 597>
                                                // reftype,                                       <L 598>
                                                var_109 = _frame_quat_0(var_body_iquat, var_geom_bodyid, var_geom_quat, var_site_bodyid, var_site_quat, var_cam_bodyid, var_cam_quat, var_xquat_in, var_0, var_9, var_101, var_104, var_107);
                                                // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 4, quat, out)       <L 600>
                                                _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_110, var_109, var_13);
                                            }
                                            var_111 = wp::where(var_99, var_104, var_95);
                                            var_112 = wp::where(var_99, var_109, var_46);
                                            var_113 = wp::where(var_99, var_101, var_96);
                                            var_114 = wp::where(var_99, var_107, var_97);
                                            if (!var_99) {
                                                // elif sensortype == SensorType.SUBTREECOM:       <L 601>
                                                var_116 = (var_6 == var_115);
                                                if (var_116) {
                                                    // vec3 = _subtree_com(subtree_com_in, worldid, objid)       <L 602>
                                                    var_117 = _subtree_com_0(var_subtree_com_in, var_0, var_9);
                                                    // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)       <L 603>
                                                    _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_118, var_117, var_13);
                                                }
                                                var_119 = wp::where(var_116, var_117, var_94);
                                                if (!var_116) {
                                                    // elif sensortype == SensorType.GEOMDIST or sensortype == SensorType.GEOMNORMAL or sensortype == SensorType.GEOMFROMTO:       <L 604>
                                                    var_121 = (var_6 == var_120);
                                                    var_123 = (var_6 == var_122);
                                                    var_125 = (var_6 == var_124);
                                                    var_126 = var_121 || var_123 || var_125;
                                                    if (var_126) {
                                                        // objtype = sensor_objtype[sensorid]       <L 605>
                                                        var_127 = wp::address(var_sensor_objtype, var_3);
                                                        var_129 = wp::load(var_127);
                                                        var_128 = wp::copy(var_129);
                                                        // objid = sensor_objid[sensorid]         <L 606>
                                                        var_130 = wp::address(var_sensor_objid, var_3);
                                                        var_132 = wp::load(var_130);
                                                        var_131 = wp::copy(var_132);
                                                        // reftype = sensor_reftype[sensorid]       <L 607>
                                                        var_133 = wp::address(var_sensor_reftype, var_3);
                                                        var_135 = wp::load(var_133);
                                                        var_134 = wp::copy(var_135);
                                                        // refid = sensor_refid[sensorid]         <L 608>
                                                        var_136 = wp::address(var_sensor_refid, var_3);
                                                        var_138 = wp::load(var_136);
                                                        var_137 = wp::copy(var_138);
                                                        // dist = float(sensor_cutoff[sensorid])       <L 611>
                                                        var_139 = wp::address(var_sensor_cutoff, var_3);
                                                        var_141 = wp::load(var_139);
                                                        var_140 = wp::float(var_141);
                                                        // pnts = vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)       <L 612>
                                                        var_148 = wp::vec_t<6, wp::float32>({var_142, var_143, var_144, var_145, var_146, var_147});
                                                        // flip = bool(False)                     <L 613>
                                                        var_150 = bool(var_149);
                                                        // if objtype == int(ObjType.BODY.value):       <L 616>
                                                        var_153 = wp::int(var_152);
                                                        var_154 = (var_128 == var_153);
                                                        if (var_154) {
                                                            // n1 = body_geomnum[objid]           <L 617>
                                                            var_155 = wp::address(var_body_geomnum, var_131);
                                                            var_157 = wp::load(var_155);
                                                            var_156 = wp::copy(var_157);
                                                            // id1 = body_geomadr[objid]          <L 618>
                                                            var_158 = wp::address(var_body_geomadr, var_131);
                                                            var_160 = wp::load(var_158);
                                                            var_159 = wp::copy(var_160);
                                                        }
                                                        if (!var_154) {
                                                            // n1 = 1                             <L 620>
                                                            // id1 = objid                        <L 621>
                                                            var_162 = wp::copy(var_131);
                                                        }
                                                        var_163 = wp::where(var_154, var_156, var_161);
                                                        var_164 = wp::where(var_154, var_159, var_162);
                                                        // if reftype == int(ObjType.BODY.value):       <L 622>
                                                        var_167 = wp::int(var_166);
                                                        var_168 = (var_134 == var_167);
                                                        if (var_168) {
                                                            // n2 = body_geomnum[refid]           <L 623>
                                                            var_169 = wp::address(var_body_geomnum, var_137);
                                                            var_171 = wp::load(var_169);
                                                            var_170 = wp::copy(var_171);
                                                            // id2 = body_geomadr[refid]          <L 624>
                                                            var_172 = wp::address(var_body_geomadr, var_137);
                                                            var_174 = wp::load(var_172);
                                                            var_173 = wp::copy(var_174);
                                                        }
                                                        if (!var_168) {
                                                            // n2 = 1                             <L 626>
                                                            // id2 = refid                        <L 627>
                                                            var_176 = wp::copy(var_137);
                                                        }
                                                        var_177 = wp::where(var_168, var_170, var_175);
                                                        var_178 = wp::where(var_168, var_173, var_176);
                                                        // for geom1 in range(n1):                <L 629>
                                                        var_179 = wp::range(var_163);
                                                        start_for_0:;
                                                            if (iter_cmp(var_179) == 0) goto end_for_0;
                                                            var_180 = wp::iter_next(var_179);
                                                            // geomid1 = id1 + geom1              <L 630>
                                                            var_181 = wp::add(var_164, var_180);
                                                            // for geom2 in range(n2):            <L 631>
                                                            var_182 = wp::range(var_177);
                                                            start_for_2:;
                                                                if (iter_cmp(var_182) == 0) goto end_for_2;
                                                                var_183 = wp::iter_next(var_182);
                                                                // geomid2 = id2 + geom2          <L 632>
                                                                var_184 = wp::add(var_178, var_183);
                                                                // if geomid1 <= geomid2:         <L 634>
                                                                var_185 = (var_181 <= var_184);
                                                                if (var_185) {
                                                                    // pairid = math.upper_tri_index(ngeom, geomid1, geomid2)       <L 635>
                                                                    var_186 = upper_tri_index_0(var_ngeom, var_181, var_184);
                                                                }
                                                                if (!var_185) {
                                                                    // pairid = math.upper_tri_index(ngeom, geomid2, geomid1)       <L 637>
                                                                    var_187 = upper_tri_index_0(var_ngeom, var_184, var_181);
                                                                }
                                                                var_188 = wp::where(var_185, var_186, var_187);
                                                                // collisionid = nxn_pairid[pairid][1]       <L 638>
                                                                var_189 = wp::address(var_nxn_pairid, var_188);
                                                                var_192 = wp::load(var_189);
                                                                var_191 = wp::extract(var_192, var_190);
                                                                // for i in range(8):             <L 640>
                                                                // dist_new = sensor_collision_in[worldid, collisionid, i, 0]       <L 641>
                                                                var_195 = wp::address(var_sensor_collision_in, var_0, var_191, var_193, var_194);
                                                                var_197 = wp::load(var_195);
                                                                var_196 = wp::copy(var_197);
                                                                // if dist_new < dist:            <L 643>
                                                                var_198 = (var_196 < var_140);
                                                                if (var_198) {
                                                                    // dist = dist_new            <L 644>
                                                                    var_199 = wp::copy(var_196);
                                                                    // if sensortype == SensorType.GEOMNORMAL or sensortype == SensorType.GEOMFROMTO:       <L 646>
                                                                    var_201 = (var_6 == var_200);
                                                                    var_203 = (var_6 == var_202);
                                                                    var_204 = var_201 || var_203;
                                                                    if (var_204) {
                                                                        // pnts = vec6(           <L 647>
                                                                        // sensor_collision_in[worldid, collisionid, i, 1],       <L 648>
                                                                        var_206 = wp::address(var_sensor_collision_in, var_0, var_191, var_193, var_205);
                                                                        // sensor_collision_in[worldid, collisionid, i, 2],       <L 649>
                                                                        var_208 = wp::address(var_sensor_collision_in, var_0, var_191, var_193, var_207);
                                                                        // sensor_collision_in[worldid, collisionid, i, 3],       <L 650>
                                                                        var_210 = wp::address(var_sensor_collision_in, var_0, var_191, var_193, var_209);
                                                                        // sensor_collision_in[worldid, collisionid, i, 4],       <L 651>
                                                                        var_212 = wp::address(var_sensor_collision_in, var_0, var_191, var_193, var_211);
                                                                        // sensor_collision_in[worldid, collisionid, i, 5],       <L 652>
                                                                        var_214 = wp::address(var_sensor_collision_in, var_0, var_191, var_193, var_213);
                                                                        // sensor_collision_in[worldid, collisionid, i, 6],       <L 653>
                                                                        var_216 = wp::address(var_sensor_collision_in, var_0, var_191, var_193, var_215);
                                                                        var_218 = wp::load(var_206);
                                                                        var_219 = wp::load(var_208);
                                                                        var_220 = wp::load(var_210);
                                                                        var_221 = wp::load(var_212);
                                                                        var_222 = wp::load(var_214);
                                                                        var_223 = wp::load(var_216);
                                                                        var_217 = wp::vec_t<6, wp::float32>({var_218, var_219, var_220, var_221, var_222, var_223});
                                                                    }
                                                                    var_224 = wp::where(var_204, var_217, var_148);
                                                                    // if geom_type[geomid1] > geom_type[geomid2]:       <L 656>
                                                                    var_225 = wp::address(var_geom_type, var_181);
                                                                    var_226 = wp::address(var_geom_type, var_184);
                                                                    var_228 = wp::load(var_225);
                                                                    var_229 = wp::load(var_226);
                                                                    var_227 = (var_228 > var_229);
                                                                    if (var_227) {
                                                                        // flip = True            <L 657>
                                                                    }
                                                                    var_231 = wp::where(var_227, var_230, var_150);
                                                                    if (!var_227) {
                                                                        // elif geom_type[geomid1] == geom_type[geomid2]:       <L 658>
                                                                        var_232 = wp::address(var_geom_type, var_181);
                                                                        var_233 = wp::address(var_geom_type, var_184);
                                                                        var_235 = wp::load(var_232);
                                                                        var_236 = wp::load(var_233);
                                                                        var_234 = (var_235 == var_236);
                                                                        if (var_234) {
                                                                            // flip = geomid1 > geomid2       <L 659>
                                                                            var_237 = (var_181 > var_184);
                                                                        }
                                                                        var_238 = wp::where(var_234, var_237, var_231);
                                                                        if (!var_234) {
                                                                            // flip = False       <L 661>
                                                                        }
                                                                        var_240 = wp::where(var_234, var_238, var_239);
                                                                    }
                                                                    var_241 = wp::where(var_227, var_231, var_240);
                                                                }
                                                                var_242 = wp::where(var_198, var_199, var_140);
                                                                var_243 = wp::where(var_198, var_224, var_148);
                                                                var_244 = wp::where(var_198, var_241, var_150);
                                                                // dist_new = sensor_collision_in[worldid, collisionid, i, 0]       <L 641>
                                                                var_247 = wp::address(var_sensor_collision_in, var_0, var_191, var_245, var_246);
                                                                var_249 = wp::load(var_247);
                                                                var_248 = wp::copy(var_249);
                                                                // if dist_new < dist:            <L 643>
                                                                var_250 = (var_248 < var_242);
                                                                if (var_250) {
                                                                    // dist = dist_new            <L 644>
                                                                    var_251 = wp::copy(var_248);
                                                                    // if sensortype == SensorType.GEOMNORMAL or sensortype == SensorType.GEOMFROMTO:       <L 646>
                                                                    var_253 = (var_6 == var_252);
                                                                    var_255 = (var_6 == var_254);
                                                                    var_256 = var_253 || var_255;
                                                                    if (var_256) {
                                                                        // pnts = vec6(           <L 647>
                                                                        // sensor_collision_in[worldid, collisionid, i, 1],       <L 648>
                                                                        var_258 = wp::address(var_sensor_collision_in, var_0, var_191, var_245, var_257);
                                                                        // sensor_collision_in[worldid, collisionid, i, 2],       <L 649>
                                                                        var_260 = wp::address(var_sensor_collision_in, var_0, var_191, var_245, var_259);
                                                                        // sensor_collision_in[worldid, collisionid, i, 3],       <L 650>
                                                                        var_262 = wp::address(var_sensor_collision_in, var_0, var_191, var_245, var_261);
                                                                        // sensor_collision_in[worldid, collisionid, i, 4],       <L 651>
                                                                        var_264 = wp::address(var_sensor_collision_in, var_0, var_191, var_245, var_263);
                                                                        // sensor_collision_in[worldid, collisionid, i, 5],       <L 652>
                                                                        var_266 = wp::address(var_sensor_collision_in, var_0, var_191, var_245, var_265);
                                                                        // sensor_collision_in[worldid, collisionid, i, 6],       <L 653>
                                                                        var_268 = wp::address(var_sensor_collision_in, var_0, var_191, var_245, var_267);
                                                                        var_270 = wp::load(var_258);
                                                                        var_271 = wp::load(var_260);
                                                                        var_272 = wp::load(var_262);
                                                                        var_273 = wp::load(var_264);
                                                                        var_274 = wp::load(var_266);
                                                                        var_275 = wp::load(var_268);
                                                                        var_269 = wp::vec_t<6, wp::float32>({var_270, var_271, var_272, var_273, var_274, var_275});
                                                                    }
                                                                    var_276 = wp::where(var_256, var_269, var_243);
                                                                    // if geom_type[geomid1] > geom_type[geomid2]:       <L 656>
                                                                    var_277 = wp::address(var_geom_type, var_181);
                                                                    var_278 = wp::address(var_geom_type, var_184);
                                                                    var_280 = wp::load(var_277);
                                                                    var_281 = wp::load(var_278);
                                                                    var_279 = (var_280 > var_281);
                                                                    if (var_279) {
                                                                        // flip = True            <L 657>
                                                                    }
                                                                    var_283 = wp::where(var_279, var_282, var_244);
                                                                    if (!var_279) {
                                                                        // elif geom_type[geomid1] == geom_type[geomid2]:       <L 658>
                                                                        var_284 = wp::address(var_geom_type, var_181);
                                                                        var_285 = wp::address(var_geom_type, var_184);
                                                                        var_287 = wp::load(var_284);
                                                                        var_288 = wp::load(var_285);
                                                                        var_286 = (var_287 == var_288);
                                                                        if (var_286) {
                                                                            // flip = geomid1 > geomid2       <L 659>
                                                                            var_289 = (var_181 > var_184);
                                                                        }
                                                                        var_290 = wp::where(var_286, var_289, var_283);
                                                                        if (!var_286) {
                                                                            // flip = False       <L 661>
                                                                        }
                                                                        var_292 = wp::where(var_286, var_290, var_291);
                                                                    }
                                                                    var_293 = wp::where(var_279, var_283, var_292);
                                                                }
                                                                var_294 = wp::where(var_250, var_251, var_242);
                                                                var_295 = wp::where(var_250, var_276, var_243);
                                                                var_296 = wp::where(var_250, var_293, var_244);
                                                                // dist_new = sensor_collision_in[worldid, collisionid, i, 0]       <L 641>
                                                                var_299 = wp::address(var_sensor_collision_in, var_0, var_191, var_297, var_298);
                                                                var_301 = wp::load(var_299);
                                                                var_300 = wp::copy(var_301);
                                                                // if dist_new < dist:            <L 643>
                                                                var_302 = (var_300 < var_294);
                                                                if (var_302) {
                                                                    // dist = dist_new            <L 644>
                                                                    var_303 = wp::copy(var_300);
                                                                    // if sensortype == SensorType.GEOMNORMAL or sensortype == SensorType.GEOMFROMTO:       <L 646>
                                                                    var_305 = (var_6 == var_304);
                                                                    var_307 = (var_6 == var_306);
                                                                    var_308 = var_305 || var_307;
                                                                    if (var_308) {
                                                                        // pnts = vec6(           <L 647>
                                                                        // sensor_collision_in[worldid, collisionid, i, 1],       <L 648>
                                                                        var_310 = wp::address(var_sensor_collision_in, var_0, var_191, var_297, var_309);
                                                                        // sensor_collision_in[worldid, collisionid, i, 2],       <L 649>
                                                                        var_312 = wp::address(var_sensor_collision_in, var_0, var_191, var_297, var_311);
                                                                        // sensor_collision_in[worldid, collisionid, i, 3],       <L 650>
                                                                        var_314 = wp::address(var_sensor_collision_in, var_0, var_191, var_297, var_313);
                                                                        // sensor_collision_in[worldid, collisionid, i, 4],       <L 651>
                                                                        var_316 = wp::address(var_sensor_collision_in, var_0, var_191, var_297, var_315);
                                                                        // sensor_collision_in[worldid, collisionid, i, 5],       <L 652>
                                                                        var_318 = wp::address(var_sensor_collision_in, var_0, var_191, var_297, var_317);
                                                                        // sensor_collision_in[worldid, collisionid, i, 6],       <L 653>
                                                                        var_320 = wp::address(var_sensor_collision_in, var_0, var_191, var_297, var_319);
                                                                        var_322 = wp::load(var_310);
                                                                        var_323 = wp::load(var_312);
                                                                        var_324 = wp::load(var_314);
                                                                        var_325 = wp::load(var_316);
                                                                        var_326 = wp::load(var_318);
                                                                        var_327 = wp::load(var_320);
                                                                        var_321 = wp::vec_t<6, wp::float32>({var_322, var_323, var_324, var_325, var_326, var_327});
                                                                    }
                                                                    var_328 = wp::where(var_308, var_321, var_295);
                                                                    // if geom_type[geomid1] > geom_type[geomid2]:       <L 656>
                                                                    var_329 = wp::address(var_geom_type, var_181);
                                                                    var_330 = wp::address(var_geom_type, var_184);
                                                                    var_332 = wp::load(var_329);
                                                                    var_333 = wp::load(var_330);
                                                                    var_331 = (var_332 > var_333);
                                                                    if (var_331) {
                                                                        // flip = True            <L 657>
                                                                    }
                                                                    var_335 = wp::where(var_331, var_334, var_296);
                                                                    if (!var_331) {
                                                                        // elif geom_type[geomid1] == geom_type[geomid2]:       <L 658>
                                                                        var_336 = wp::address(var_geom_type, var_181);
                                                                        var_337 = wp::address(var_geom_type, var_184);
                                                                        var_339 = wp::load(var_336);
                                                                        var_340 = wp::load(var_337);
                                                                        var_338 = (var_339 == var_340);
                                                                        if (var_338) {
                                                                            // flip = geomid1 > geomid2       <L 659>
                                                                            var_341 = (var_181 > var_184);
                                                                        }
                                                                        var_342 = wp::where(var_338, var_341, var_335);
                                                                        if (!var_338) {
                                                                            // flip = False       <L 661>
                                                                        }
                                                                        var_344 = wp::where(var_338, var_342, var_343);
                                                                    }
                                                                    var_345 = wp::where(var_331, var_335, var_344);
                                                                }
                                                                var_346 = wp::where(var_302, var_303, var_294);
                                                                var_347 = wp::where(var_302, var_328, var_295);
                                                                var_348 = wp::where(var_302, var_345, var_296);
                                                                // dist_new = sensor_collision_in[worldid, collisionid, i, 0]       <L 641>
                                                                var_351 = wp::address(var_sensor_collision_in, var_0, var_191, var_349, var_350);
                                                                var_353 = wp::load(var_351);
                                                                var_352 = wp::copy(var_353);
                                                                // if dist_new < dist:            <L 643>
                                                                var_354 = (var_352 < var_346);
                                                                if (var_354) {
                                                                    // dist = dist_new            <L 644>
                                                                    var_355 = wp::copy(var_352);
                                                                    // if sensortype == SensorType.GEOMNORMAL or sensortype == SensorType.GEOMFROMTO:       <L 646>
                                                                    var_357 = (var_6 == var_356);
                                                                    var_359 = (var_6 == var_358);
                                                                    var_360 = var_357 || var_359;
                                                                    if (var_360) {
                                                                        // pnts = vec6(           <L 647>
                                                                        // sensor_collision_in[worldid, collisionid, i, 1],       <L 648>
                                                                        var_362 = wp::address(var_sensor_collision_in, var_0, var_191, var_349, var_361);
                                                                        // sensor_collision_in[worldid, collisionid, i, 2],       <L 649>
                                                                        var_364 = wp::address(var_sensor_collision_in, var_0, var_191, var_349, var_363);
                                                                        // sensor_collision_in[worldid, collisionid, i, 3],       <L 650>
                                                                        var_366 = wp::address(var_sensor_collision_in, var_0, var_191, var_349, var_365);
                                                                        // sensor_collision_in[worldid, collisionid, i, 4],       <L 651>
                                                                        var_368 = wp::address(var_sensor_collision_in, var_0, var_191, var_349, var_367);
                                                                        // sensor_collision_in[worldid, collisionid, i, 5],       <L 652>
                                                                        var_370 = wp::address(var_sensor_collision_in, var_0, var_191, var_349, var_369);
                                                                        // sensor_collision_in[worldid, collisionid, i, 6],       <L 653>
                                                                        var_372 = wp::address(var_sensor_collision_in, var_0, var_191, var_349, var_371);
                                                                        var_374 = wp::load(var_362);
                                                                        var_375 = wp::load(var_364);
                                                                        var_376 = wp::load(var_366);
                                                                        var_377 = wp::load(var_368);
                                                                        var_378 = wp::load(var_370);
                                                                        var_379 = wp::load(var_372);
                                                                        var_373 = wp::vec_t<6, wp::float32>({var_374, var_375, var_376, var_377, var_378, var_379});
                                                                    }
                                                                    var_380 = wp::where(var_360, var_373, var_347);
                                                                    // if geom_type[geomid1] > geom_type[geomid2]:       <L 656>
                                                                    var_381 = wp::address(var_geom_type, var_181);
                                                                    var_382 = wp::address(var_geom_type, var_184);
                                                                    var_384 = wp::load(var_381);
                                                                    var_385 = wp::load(var_382);
                                                                    var_383 = (var_384 > var_385);
                                                                    if (var_383) {
                                                                        // flip = True            <L 657>
                                                                    }
                                                                    var_387 = wp::where(var_383, var_386, var_348);
                                                                    if (!var_383) {
                                                                        // elif geom_type[geomid1] == geom_type[geomid2]:       <L 658>
                                                                        var_388 = wp::address(var_geom_type, var_181);
                                                                        var_389 = wp::address(var_geom_type, var_184);
                                                                        var_391 = wp::load(var_388);
                                                                        var_392 = wp::load(var_389);
                                                                        var_390 = (var_391 == var_392);
                                                                        if (var_390) {
                                                                            // flip = geomid1 > geomid2       <L 659>
                                                                            var_393 = (var_181 > var_184);
                                                                        }
                                                                        var_394 = wp::where(var_390, var_393, var_387);
                                                                        if (!var_390) {
                                                                            // flip = False       <L 661>
                                                                        }
                                                                        var_396 = wp::where(var_390, var_394, var_395);
                                                                    }
                                                                    var_397 = wp::where(var_383, var_387, var_396);
                                                                }
                                                                var_398 = wp::where(var_354, var_355, var_346);
                                                                var_399 = wp::where(var_354, var_380, var_347);
                                                                var_400 = wp::where(var_354, var_397, var_348);
                                                                // dist_new = sensor_collision_in[worldid, collisionid, i, 0]       <L 641>
                                                                var_403 = wp::address(var_sensor_collision_in, var_0, var_191, var_401, var_402);
                                                                var_405 = wp::load(var_403);
                                                                var_404 = wp::copy(var_405);
                                                                // if dist_new < dist:            <L 643>
                                                                var_406 = (var_404 < var_398);
                                                                if (var_406) {
                                                                    // dist = dist_new            <L 644>
                                                                    var_407 = wp::copy(var_404);
                                                                    // if sensortype == SensorType.GEOMNORMAL or sensortype == SensorType.GEOMFROMTO:       <L 646>
                                                                    var_409 = (var_6 == var_408);
                                                                    var_411 = (var_6 == var_410);
                                                                    var_412 = var_409 || var_411;
                                                                    if (var_412) {
                                                                        // pnts = vec6(           <L 647>
                                                                        // sensor_collision_in[worldid, collisionid, i, 1],       <L 648>
                                                                        var_414 = wp::address(var_sensor_collision_in, var_0, var_191, var_401, var_413);
                                                                        // sensor_collision_in[worldid, collisionid, i, 2],       <L 649>
                                                                        var_416 = wp::address(var_sensor_collision_in, var_0, var_191, var_401, var_415);
                                                                        // sensor_collision_in[worldid, collisionid, i, 3],       <L 650>
                                                                        var_418 = wp::address(var_sensor_collision_in, var_0, var_191, var_401, var_417);
                                                                        // sensor_collision_in[worldid, collisionid, i, 4],       <L 651>
                                                                        var_420 = wp::address(var_sensor_collision_in, var_0, var_191, var_401, var_419);
                                                                        // sensor_collision_in[worldid, collisionid, i, 5],       <L 652>
                                                                        var_422 = wp::address(var_sensor_collision_in, var_0, var_191, var_401, var_421);
                                                                        // sensor_collision_in[worldid, collisionid, i, 6],       <L 653>
                                                                        var_424 = wp::address(var_sensor_collision_in, var_0, var_191, var_401, var_423);
                                                                        var_426 = wp::load(var_414);
                                                                        var_427 = wp::load(var_416);
                                                                        var_428 = wp::load(var_418);
                                                                        var_429 = wp::load(var_420);
                                                                        var_430 = wp::load(var_422);
                                                                        var_431 = wp::load(var_424);
                                                                        var_425 = wp::vec_t<6, wp::float32>({var_426, var_427, var_428, var_429, var_430, var_431});
                                                                    }
                                                                    var_432 = wp::where(var_412, var_425, var_399);
                                                                    // if geom_type[geomid1] > geom_type[geomid2]:       <L 656>
                                                                    var_433 = wp::address(var_geom_type, var_181);
                                                                    var_434 = wp::address(var_geom_type, var_184);
                                                                    var_436 = wp::load(var_433);
                                                                    var_437 = wp::load(var_434);
                                                                    var_435 = (var_436 > var_437);
                                                                    if (var_435) {
                                                                        // flip = True            <L 657>
                                                                    }
                                                                    var_439 = wp::where(var_435, var_438, var_400);
                                                                    if (!var_435) {
                                                                        // elif geom_type[geomid1] == geom_type[geomid2]:       <L 658>
                                                                        var_440 = wp::address(var_geom_type, var_181);
                                                                        var_441 = wp::address(var_geom_type, var_184);
                                                                        var_443 = wp::load(var_440);
                                                                        var_444 = wp::load(var_441);
                                                                        var_442 = (var_443 == var_444);
                                                                        if (var_442) {
                                                                            // flip = geomid1 > geomid2       <L 659>
                                                                            var_445 = (var_181 > var_184);
                                                                        }
                                                                        var_446 = wp::where(var_442, var_445, var_439);
                                                                        if (!var_442) {
                                                                            // flip = False       <L 661>
                                                                        }
                                                                        var_448 = wp::where(var_442, var_446, var_447);
                                                                    }
                                                                    var_449 = wp::where(var_435, var_439, var_448);
                                                                }
                                                                var_450 = wp::where(var_406, var_407, var_398);
                                                                var_451 = wp::where(var_406, var_432, var_399);
                                                                var_452 = wp::where(var_406, var_449, var_400);
                                                                // dist_new = sensor_collision_in[worldid, collisionid, i, 0]       <L 641>
                                                                var_455 = wp::address(var_sensor_collision_in, var_0, var_191, var_453, var_454);
                                                                var_457 = wp::load(var_455);
                                                                var_456 = wp::copy(var_457);
                                                                // if dist_new < dist:            <L 643>
                                                                var_458 = (var_456 < var_450);
                                                                if (var_458) {
                                                                    // dist = dist_new            <L 644>
                                                                    var_459 = wp::copy(var_456);
                                                                    // if sensortype == SensorType.GEOMNORMAL or sensortype == SensorType.GEOMFROMTO:       <L 646>
                                                                    var_461 = (var_6 == var_460);
                                                                    var_463 = (var_6 == var_462);
                                                                    var_464 = var_461 || var_463;
                                                                    if (var_464) {
                                                                        // pnts = vec6(           <L 647>
                                                                        // sensor_collision_in[worldid, collisionid, i, 1],       <L 648>
                                                                        var_466 = wp::address(var_sensor_collision_in, var_0, var_191, var_453, var_465);
                                                                        // sensor_collision_in[worldid, collisionid, i, 2],       <L 649>
                                                                        var_468 = wp::address(var_sensor_collision_in, var_0, var_191, var_453, var_467);
                                                                        // sensor_collision_in[worldid, collisionid, i, 3],       <L 650>
                                                                        var_470 = wp::address(var_sensor_collision_in, var_0, var_191, var_453, var_469);
                                                                        // sensor_collision_in[worldid, collisionid, i, 4],       <L 651>
                                                                        var_472 = wp::address(var_sensor_collision_in, var_0, var_191, var_453, var_471);
                                                                        // sensor_collision_in[worldid, collisionid, i, 5],       <L 652>
                                                                        var_474 = wp::address(var_sensor_collision_in, var_0, var_191, var_453, var_473);
                                                                        // sensor_collision_in[worldid, collisionid, i, 6],       <L 653>
                                                                        var_476 = wp::address(var_sensor_collision_in, var_0, var_191, var_453, var_475);
                                                                        var_478 = wp::load(var_466);
                                                                        var_479 = wp::load(var_468);
                                                                        var_480 = wp::load(var_470);
                                                                        var_481 = wp::load(var_472);
                                                                        var_482 = wp::load(var_474);
                                                                        var_483 = wp::load(var_476);
                                                                        var_477 = wp::vec_t<6, wp::float32>({var_478, var_479, var_480, var_481, var_482, var_483});
                                                                    }
                                                                    var_484 = wp::where(var_464, var_477, var_451);
                                                                    // if geom_type[geomid1] > geom_type[geomid2]:       <L 656>
                                                                    var_485 = wp::address(var_geom_type, var_181);
                                                                    var_486 = wp::address(var_geom_type, var_184);
                                                                    var_488 = wp::load(var_485);
                                                                    var_489 = wp::load(var_486);
                                                                    var_487 = (var_488 > var_489);
                                                                    if (var_487) {
                                                                        // flip = True            <L 657>
                                                                    }
                                                                    var_491 = wp::where(var_487, var_490, var_452);
                                                                    if (!var_487) {
                                                                        // elif geom_type[geomid1] == geom_type[geomid2]:       <L 658>
                                                                        var_492 = wp::address(var_geom_type, var_181);
                                                                        var_493 = wp::address(var_geom_type, var_184);
                                                                        var_495 = wp::load(var_492);
                                                                        var_496 = wp::load(var_493);
                                                                        var_494 = (var_495 == var_496);
                                                                        if (var_494) {
                                                                            // flip = geomid1 > geomid2       <L 659>
                                                                            var_497 = (var_181 > var_184);
                                                                        }
                                                                        var_498 = wp::where(var_494, var_497, var_491);
                                                                        if (!var_494) {
                                                                            // flip = False       <L 661>
                                                                        }
                                                                        var_500 = wp::where(var_494, var_498, var_499);
                                                                    }
                                                                    var_501 = wp::where(var_487, var_491, var_500);
                                                                }
                                                                var_502 = wp::where(var_458, var_459, var_450);
                                                                var_503 = wp::where(var_458, var_484, var_451);
                                                                var_504 = wp::where(var_458, var_501, var_452);
                                                                // dist_new = sensor_collision_in[worldid, collisionid, i, 0]       <L 641>
                                                                var_507 = wp::address(var_sensor_collision_in, var_0, var_191, var_505, var_506);
                                                                var_509 = wp::load(var_507);
                                                                var_508 = wp::copy(var_509);
                                                                // if dist_new < dist:            <L 643>
                                                                var_510 = (var_508 < var_502);
                                                                if (var_510) {
                                                                    // dist = dist_new            <L 644>
                                                                    var_511 = wp::copy(var_508);
                                                                    // if sensortype == SensorType.GEOMNORMAL or sensortype == SensorType.GEOMFROMTO:       <L 646>
                                                                    var_513 = (var_6 == var_512);
                                                                    var_515 = (var_6 == var_514);
                                                                    var_516 = var_513 || var_515;
                                                                    if (var_516) {
                                                                        // pnts = vec6(           <L 647>
                                                                        // sensor_collision_in[worldid, collisionid, i, 1],       <L 648>
                                                                        var_518 = wp::address(var_sensor_collision_in, var_0, var_191, var_505, var_517);
                                                                        // sensor_collision_in[worldid, collisionid, i, 2],       <L 649>
                                                                        var_520 = wp::address(var_sensor_collision_in, var_0, var_191, var_505, var_519);
                                                                        // sensor_collision_in[worldid, collisionid, i, 3],       <L 650>
                                                                        var_522 = wp::address(var_sensor_collision_in, var_0, var_191, var_505, var_521);
                                                                        // sensor_collision_in[worldid, collisionid, i, 4],       <L 651>
                                                                        var_524 = wp::address(var_sensor_collision_in, var_0, var_191, var_505, var_523);
                                                                        // sensor_collision_in[worldid, collisionid, i, 5],       <L 652>
                                                                        var_526 = wp::address(var_sensor_collision_in, var_0, var_191, var_505, var_525);
                                                                        // sensor_collision_in[worldid, collisionid, i, 6],       <L 653>
                                                                        var_528 = wp::address(var_sensor_collision_in, var_0, var_191, var_505, var_527);
                                                                        var_530 = wp::load(var_518);
                                                                        var_531 = wp::load(var_520);
                                                                        var_532 = wp::load(var_522);
                                                                        var_533 = wp::load(var_524);
                                                                        var_534 = wp::load(var_526);
                                                                        var_535 = wp::load(var_528);
                                                                        var_529 = wp::vec_t<6, wp::float32>({var_530, var_531, var_532, var_533, var_534, var_535});
                                                                    }
                                                                    var_536 = wp::where(var_516, var_529, var_503);
                                                                    // if geom_type[geomid1] > geom_type[geomid2]:       <L 656>
                                                                    var_537 = wp::address(var_geom_type, var_181);
                                                                    var_538 = wp::address(var_geom_type, var_184);
                                                                    var_540 = wp::load(var_537);
                                                                    var_541 = wp::load(var_538);
                                                                    var_539 = (var_540 > var_541);
                                                                    if (var_539) {
                                                                        // flip = True            <L 657>
                                                                    }
                                                                    var_543 = wp::where(var_539, var_542, var_504);
                                                                    if (!var_539) {
                                                                        // elif geom_type[geomid1] == geom_type[geomid2]:       <L 658>
                                                                        var_544 = wp::address(var_geom_type, var_181);
                                                                        var_545 = wp::address(var_geom_type, var_184);
                                                                        var_547 = wp::load(var_544);
                                                                        var_548 = wp::load(var_545);
                                                                        var_546 = (var_547 == var_548);
                                                                        if (var_546) {
                                                                            // flip = geomid1 > geomid2       <L 659>
                                                                            var_549 = (var_181 > var_184);
                                                                        }
                                                                        var_550 = wp::where(var_546, var_549, var_543);
                                                                        if (!var_546) {
                                                                            // flip = False       <L 661>
                                                                        }
                                                                        var_552 = wp::where(var_546, var_550, var_551);
                                                                    }
                                                                    var_553 = wp::where(var_539, var_543, var_552);
                                                                }
                                                                var_554 = wp::where(var_510, var_511, var_502);
                                                                var_555 = wp::where(var_510, var_536, var_503);
                                                                var_556 = wp::where(var_510, var_553, var_504);
                                                                // dist_new = sensor_collision_in[worldid, collisionid, i, 0]       <L 641>
                                                                var_559 = wp::address(var_sensor_collision_in, var_0, var_191, var_557, var_558);
                                                                var_561 = wp::load(var_559);
                                                                var_560 = wp::copy(var_561);
                                                                // if dist_new < dist:            <L 643>
                                                                var_562 = (var_560 < var_554);
                                                                if (var_562) {
                                                                    // dist = dist_new            <L 644>
                                                                    var_563 = wp::copy(var_560);
                                                                    // if sensortype == SensorType.GEOMNORMAL or sensortype == SensorType.GEOMFROMTO:       <L 646>
                                                                    var_565 = (var_6 == var_564);
                                                                    var_567 = (var_6 == var_566);
                                                                    var_568 = var_565 || var_567;
                                                                    if (var_568) {
                                                                        // pnts = vec6(           <L 647>
                                                                        // sensor_collision_in[worldid, collisionid, i, 1],       <L 648>
                                                                        var_570 = wp::address(var_sensor_collision_in, var_0, var_191, var_557, var_569);
                                                                        // sensor_collision_in[worldid, collisionid, i, 2],       <L 649>
                                                                        var_572 = wp::address(var_sensor_collision_in, var_0, var_191, var_557, var_571);
                                                                        // sensor_collision_in[worldid, collisionid, i, 3],       <L 650>
                                                                        var_574 = wp::address(var_sensor_collision_in, var_0, var_191, var_557, var_573);
                                                                        // sensor_collision_in[worldid, collisionid, i, 4],       <L 651>
                                                                        var_576 = wp::address(var_sensor_collision_in, var_0, var_191, var_557, var_575);
                                                                        // sensor_collision_in[worldid, collisionid, i, 5],       <L 652>
                                                                        var_578 = wp::address(var_sensor_collision_in, var_0, var_191, var_557, var_577);
                                                                        // sensor_collision_in[worldid, collisionid, i, 6],       <L 653>
                                                                        var_580 = wp::address(var_sensor_collision_in, var_0, var_191, var_557, var_579);
                                                                        var_582 = wp::load(var_570);
                                                                        var_583 = wp::load(var_572);
                                                                        var_584 = wp::load(var_574);
                                                                        var_585 = wp::load(var_576);
                                                                        var_586 = wp::load(var_578);
                                                                        var_587 = wp::load(var_580);
                                                                        var_581 = wp::vec_t<6, wp::float32>({var_582, var_583, var_584, var_585, var_586, var_587});
                                                                    }
                                                                    var_588 = wp::where(var_568, var_581, var_555);
                                                                    // if geom_type[geomid1] > geom_type[geomid2]:       <L 656>
                                                                    var_589 = wp::address(var_geom_type, var_181);
                                                                    var_590 = wp::address(var_geom_type, var_184);
                                                                    var_592 = wp::load(var_589);
                                                                    var_593 = wp::load(var_590);
                                                                    var_591 = (var_592 > var_593);
                                                                    if (var_591) {
                                                                        // flip = True            <L 657>
                                                                    }
                                                                    var_595 = wp::where(var_591, var_594, var_556);
                                                                    if (!var_591) {
                                                                        // elif geom_type[geomid1] == geom_type[geomid2]:       <L 658>
                                                                        var_596 = wp::address(var_geom_type, var_181);
                                                                        var_597 = wp::address(var_geom_type, var_184);
                                                                        var_599 = wp::load(var_596);
                                                                        var_600 = wp::load(var_597);
                                                                        var_598 = (var_599 == var_600);
                                                                        if (var_598) {
                                                                            // flip = geomid1 > geomid2       <L 659>
                                                                            var_601 = (var_181 > var_184);
                                                                        }
                                                                        var_602 = wp::where(var_598, var_601, var_595);
                                                                        if (!var_598) {
                                                                            // flip = False       <L 661>
                                                                        }
                                                                        var_604 = wp::where(var_598, var_602, var_603);
                                                                    }
                                                                    var_605 = wp::where(var_591, var_595, var_604);
                                                                }
                                                                var_606 = wp::where(var_562, var_563, var_554);
                                                                var_607 = wp::where(var_562, var_588, var_555);
                                                                var_608 = wp::where(var_562, var_605, var_556);
                                                                wp::assign(var_140, var_606);
                                                                wp::assign(var_148, var_607);
                                                                wp::assign(var_150, var_608);
                                                                goto start_for_2;
                                                            end_for_2:;
                                                            goto start_for_0;
                                                        end_for_0:;
                                                        // if sensortype == int(SensorType.GEOMDIST.value):       <L 662>
                                                        var_611 = wp::int(var_610);
                                                        var_612 = (var_6 == var_611);
                                                        if (var_612) {
                                                            // _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, dist, out)       <L 663>
                                                            _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_140, var_13);
                                                        }
                                                        if (!var_612) {
                                                            // elif sensortype == int(SensorType.GEOMNORMAL.value):       <L 664>
                                                            var_615 = wp::int(var_614);
                                                            var_616 = (var_6 == var_615);
                                                            if (var_616) {
                                                                // if dist <= sensor_cutoff[sensorid]:       <L 665>
                                                                var_617 = wp::address(var_sensor_cutoff, var_3);
                                                                var_619 = wp::load(var_617);
                                                                var_618 = (var_140 <= var_619);
                                                                if (var_618) {
                                                                    // normal = wp.normalize(wp.vec3(pnts[3] - pnts[0], pnts[4] - pnts[1], pnts[5] - pnts[2]))       <L 666>
                                                                    var_621 = wp::extract(var_148, var_620);
                                                                    var_623 = wp::extract(var_148, var_622);
                                                                    var_624 = wp::sub(var_621, var_623);
                                                                    var_626 = wp::extract(var_148, var_625);
                                                                    var_628 = wp::extract(var_148, var_627);
                                                                    var_629 = wp::sub(var_626, var_628);
                                                                    var_631 = wp::extract(var_148, var_630);
                                                                    var_633 = wp::extract(var_148, var_632);
                                                                    var_634 = wp::sub(var_631, var_633);
                                                                    var_635 = wp::vec_t<3, wp::float32>(var_624, var_629, var_634);
                                                                    var_636 = wp::normalize(var_635);
                                                                    // if flip:                   <L 667>
                                                                    if (var_150) {
                                                                        // normal *= -1.0         <L 668>
                                                                        var_639 = wp::mul(var_636, var_638);
                                                                    }
                                                                    var_640 = wp::where(var_150, var_639, var_636);
                                                                }
                                                                if (!var_618) {
                                                                    // normal = wp.vec3(0.0, 0.0, 0.0)       <L 670>
                                                                    var_644 = wp::vec_t<3, wp::float32>(var_641, var_642, var_643);
                                                                }
                                                                var_645 = wp::where(var_618, var_640, var_644);
                                                                // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, normal, out)       <L 671>
                                                                _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_646, var_645, var_13);
                                                            }
                                                            if (!var_616) {
                                                                // elif sensortype == int(SensorType.GEOMFROMTO.value):       <L 672>
                                                                var_649 = wp::int(var_648);
                                                                var_650 = (var_6 == var_649);
                                                                if (var_650) {
                                                                    // if dist <= sensor_cutoff[sensorid]:       <L 673>
                                                                    var_651 = wp::address(var_sensor_cutoff, var_3);
                                                                    var_653 = wp::load(var_651);
                                                                    var_652 = (var_140 <= var_653);
                                                                    if (var_652) {
                                                                        // if flip:               <L 674>
                                                                        if (var_150) {
                                                                            // fromto = vec6(pnts[3], pnts[4], pnts[5], pnts[0], pnts[1], pnts[2])       <L 675>
                                                                            var_655 = wp::extract(var_148, var_654);
                                                                            var_657 = wp::extract(var_148, var_656);
                                                                            var_659 = wp::extract(var_148, var_658);
                                                                            var_661 = wp::extract(var_148, var_660);
                                                                            var_663 = wp::extract(var_148, var_662);
                                                                            var_665 = wp::extract(var_148, var_664);
                                                                            var_666 = wp::vec_t<6, wp::float32>({var_655, var_657, var_659, var_661, var_663, var_665});
                                                                        }
                                                                        if (!var_150) {
                                                                            // fromto = pnts       <L 677>
                                                                            var_667 = wp::copy(var_148);
                                                                        }
                                                                        var_668 = wp::where(var_150, var_666, var_667);
                                                                    }
                                                                    if (!var_652) {
                                                                        // fromto = vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)       <L 679>
                                                                        var_675 = wp::vec_t<6, wp::float32>({var_669, var_670, var_671, var_672, var_673, var_674});
                                                                    }
                                                                    var_676 = wp::where(var_652, var_668, var_675);
                                                                    // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 6, fromto, out)       <L 680>
                                                                    _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_677, var_676, var_13);
                                                                }
                                                            }
                                                        }
                                                    }
                                                    var_678 = wp::where(var_126, var_131, var_9);
                                                    var_679 = wp::where(var_126, var_137, var_111);
                                                    var_680 = wp::where(var_126, var_128, var_113);
                                                    var_681 = wp::where(var_126, var_134, var_114);
                                                    if (!var_126) {
                                                        // elif sensortype == SensorType.INSIDESITE:       <L 681>
                                                        var_683 = (var_6 == var_682);
                                                        if (var_683) {
                                                            // objtype = sensor_objtype[sensorid]       <L 682>
                                                            var_684 = wp::address(var_sensor_objtype, var_3);
                                                            var_686 = wp::load(var_684);
                                                            var_685 = wp::copy(var_686);
                                                            // if objtype == ObjType.XBODY:       <L 683>
                                                            var_688 = (var_685 == var_687);
                                                            if (var_688) {
                                                                // xpos = xpos_in[worldid, objid]       <L 684>
                                                                var_689 = wp::address(var_xpos_in, var_0, var_678);
                                                                var_691 = wp::load(var_689);
                                                                var_690 = wp::copy(var_691);
                                                            }
                                                            if (!var_688) {
                                                                // elif objtype == ObjType.BODY:       <L 685>
                                                                var_693 = (var_685 == var_692);
                                                                if (var_693) {
                                                                    // xpos = xipos_in[worldid, objid]       <L 686>
                                                                    var_694 = wp::address(var_xipos_in, var_0, var_678);
                                                                    var_696 = wp::load(var_694);
                                                                    var_695 = wp::copy(var_696);
                                                                }
                                                                var_697 = wp::where(var_693, var_695, var_690);
                                                                if (!var_693) {
                                                                    // elif objtype == ObjType.GEOM:       <L 687>
                                                                    var_699 = (var_685 == var_698);
                                                                    if (var_699) {
                                                                        // xpos = geom_xpos_in[worldid, objid]       <L 688>
                                                                        var_700 = wp::address(var_geom_xpos_in, var_0, var_678);
                                                                        var_702 = wp::load(var_700);
                                                                        var_701 = wp::copy(var_702);
                                                                    }
                                                                    var_703 = wp::where(var_699, var_701, var_697);
                                                                    if (!var_699) {
                                                                        // elif objtype == ObjType.SITE:       <L 689>
                                                                        var_705 = (var_685 == var_704);
                                                                        if (var_705) {
                                                                            // xpos = site_xpos_in[worldid, objid]       <L 690>
                                                                            var_706 = wp::address(var_site_xpos_in, var_0, var_678);
                                                                            var_708 = wp::load(var_706);
                                                                            var_707 = wp::copy(var_708);
                                                                        }
                                                                        var_709 = wp::where(var_705, var_707, var_703);
                                                                        if (!var_705) {
                                                                            // elif objtype == ObjType.CAMERA:       <L 691>
                                                                            var_711 = (var_685 == var_710);
                                                                            if (var_711) {
                                                                                // xpos = cam_xpos_in[worldid, objid]       <L 692>
                                                                                var_712 = wp::address(var_cam_xpos_in, var_0, var_678);
                                                                                var_714 = wp::load(var_712);
                                                                                var_713 = wp::copy(var_714);
                                                                            }
                                                                            var_715 = wp::where(var_711, var_713, var_709);
                                                                            if (!var_711) {
                                                                                // return  # should not occur       <L 694>
                                                                                continue;
                                                                            }
                                                                        }
                                                                        var_716 = wp::where(var_705, var_709, var_715);
                                                                    }
                                                                    var_717 = wp::where(var_699, var_703, var_716);
                                                                }
                                                                var_718 = wp::where(var_693, var_697, var_717);
                                                            }
                                                            var_719 = wp::where(var_688, var_690, var_718);
                                                            // refid = sensor_refid[sensorid]       <L 695>
                                                            var_720 = wp::address(var_sensor_refid, var_3);
                                                            var_722 = wp::load(var_720);
                                                            var_721 = wp::copy(var_722);
                                                            // val_bool = inside_geom(site_xpos_in[worldid, refid], site_xmat_in[worldid, refid], site_size[refid], site_type[refid], xpos)       <L 696>
                                                            var_723 = wp::address(var_site_xpos_in, var_0, var_721);
                                                            var_724 = wp::address(var_site_xmat_in, var_0, var_721);
                                                            var_725 = wp::address(var_site_size, var_721);
                                                            var_726 = wp::address(var_site_type, var_721);
                                                            var_728 = wp::load(var_723);
                                                            var_729 = wp::load(var_724);
                                                            var_730 = wp::load(var_725);
                                                            var_731 = wp::load(var_726);
                                                            var_727 = inside_geom_0(var_728, var_729, var_730, var_731, var_719);
                                                            // _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, float(val_bool), out)       <L 697>
                                                            var_732 = wp::float(var_727);
                                                            _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_732, var_13);
                                                        }
                                                        var_733 = wp::where(var_683, var_721, var_679);
                                                        var_734 = wp::where(var_683, var_685, var_680);
                                                        if (!var_683) {
                                                            // elif sensortype == SensorType.E_POTENTIAL:       <L 698>
                                                            var_736 = (var_6 == var_735);
                                                            if (var_736) {
                                                                // val = energy_in[worldid][0]       <L 699>
                                                                var_737 = wp::address(var_energy_in, var_0);
                                                                var_740 = wp::load(var_737);
                                                                var_739 = wp::extract(var_740, var_738);
                                                                // _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)       <L 700>
                                                                _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_739, var_13);
                                                            }
                                                            var_741 = wp::where(var_736, var_739, var_43);
                                                            if (!var_736) {
                                                                // elif sensortype == SensorType.E_KINETIC:       <L 701>
                                                                var_743 = (var_6 == var_742);
                                                                if (var_743) {
                                                                    // val = energy_in[worldid][1]       <L 702>
                                                                    var_744 = wp::address(var_energy_in, var_0);
                                                                    var_747 = wp::load(var_744);
                                                                    var_746 = wp::extract(var_747, var_745);
                                                                    // _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)       <L 703>
                                                                    _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_746, var_13);
                                                                }
                                                                var_748 = wp::where(var_743, var_746, var_741);
                                                                if (!var_743) {
                                                                    // elif sensortype == SensorType.CLOCK:       <L 704>
                                                                    var_750 = (var_6 == var_749);
                                                                    if (var_750) {
                                                                        // val = _clock(time_in, worldid)       <L 705>
                                                                        var_751 = _clock_0(var_time_in, var_0);
                                                                        // _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)       <L 706>
                                                                        _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_751, var_13);
                                                                    }
                                                                    var_752 = wp::where(var_750, var_751, var_748);
                                                                }
                                                                var_753 = wp::where(var_743, var_748, var_752);
                                                            }
                                                            var_754 = wp::where(var_736, var_741, var_753);
                                                        }
                                                        var_755 = wp::where(var_683, var_43, var_754);
                                                    }
                                                    var_756 = wp::where(var_126, var_679, var_733);
                                                    var_757 = wp::where(var_126, var_43, var_755);
                                                    var_758 = wp::where(var_126, var_680, var_734);
                                                }
                                                var_759 = wp::where(var_116, var_9, var_678);
                                                var_760 = wp::where(var_116, var_111, var_756);
                                                var_761 = wp::where(var_116, var_43, var_757);
                                                var_762 = wp::where(var_116, var_113, var_758);
                                                var_763 = wp::where(var_116, var_114, var_681);
                                            }
                                            var_764 = wp::where(var_99, var_9, var_759);
                                            var_765 = wp::where(var_99, var_94, var_119);
                                            var_766 = wp::where(var_99, var_111, var_760);
                                            var_767 = wp::where(var_99, var_43, var_761);
                                            var_768 = wp::where(var_99, var_113, var_762);
                                            var_769 = wp::where(var_99, var_114, var_763);
                                        }
                                        var_770 = wp::where(var_69, var_9, var_764);
                                        var_771 = wp::where(var_69, var_94, var_765);
                                        var_772 = wp::where(var_69, var_95, var_766);
                                        var_773 = wp::where(var_69, var_43, var_767);
                                        var_774 = wp::where(var_69, var_46, var_112);
                                        var_775 = wp::where(var_69, var_96, var_768);
                                        var_776 = wp::where(var_69, var_97, var_769);
                                    }
                                    var_777 = wp::where(var_49, var_9, var_770);
                                    var_778 = wp::where(var_49, var_61, var_771);
                                    var_779 = wp::where(var_49, var_62, var_772);
                                    var_780 = wp::where(var_49, var_43, var_773);
                                    var_781 = wp::where(var_49, var_46, var_774);
                                    var_782 = wp::where(var_49, var_51, var_775);
                                    var_783 = wp::where(var_49, var_57, var_776);
                                }
                                var_784 = wp::where(var_45, var_9, var_777);
                                var_785 = wp::where(var_45, var_16, var_778);
                                var_786 = wp::where(var_45, var_21, var_779);
                                var_787 = wp::where(var_45, var_43, var_780);
                                var_788 = wp::where(var_45, var_46, var_781);
                            }
                            var_789 = wp::where(var_41, var_9, var_784);
                            var_790 = wp::where(var_41, var_16, var_785);
                            var_791 = wp::where(var_41, var_21, var_786);
                            var_792 = wp::where(var_41, var_43, var_787);
                        }
                        var_793 = wp::where(var_37, var_9, var_789);
                        var_794 = wp::where(var_37, var_16, var_790);
                        var_795 = wp::where(var_37, var_21, var_791);
                        var_796 = wp::where(var_37, var_39, var_792);
                    }
                    var_797 = wp::where(var_33, var_9, var_793);
                    var_798 = wp::where(var_33, var_16, var_794);
                    var_799 = wp::where(var_33, var_21, var_795);
                    var_800 = wp::where(var_33, var_35, var_796);
                }
                var_801 = wp::where(var_26, var_9, var_797);
                var_802 = wp::where(var_26, var_16, var_798);
                var_803 = wp::where(var_26, var_21, var_799);
                var_804 = wp::where(var_26, var_30, var_800);
            }
            var_805 = wp::where(var_19, var_9, var_801);
            var_806 = wp::where(var_19, var_16, var_802);
            var_807 = wp::where(var_19, var_21, var_803);
        }
        var_808 = wp::where(var_15, var_9, var_805);
        var_809 = wp::where(var_15, var_16, var_806);
    }
}



extern "C" __global__ void _tendon_actuator_force_a7ed58b7_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_actuator_trntype,
    wp::array_t<wp::vec_t<2, wp::int32>> var_actuator_trnid,
    wp::array_t<wp::int32> var_sensor_objid,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::int32> var_sensor_tendonactfrc_adr,
    wp::array_t<wp::float32> var_actuator_force_in,
    wp::array_t<wp::float32> var_sensordata_out)
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
        wp::int32* var_6;
        const wp::int32 var_7 = 3;
        bool var_8;
        wp::int32 var_9;
        wp::vec_t<2, wp::int32>* var_10;
        const wp::int32 var_11 = 0;
        wp::int32 var_12;
        wp::vec_t<2, wp::int32> var_13;
        wp::int32* var_14;
        bool var_15;
        wp::int32 var_16;
        bool var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::float32* var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        //---------
        // forward
        // def _tendon_actuator_force(                                                            <L 1539>
        // worldid, tenactfrcid, actid = wp.tid()                                                 <L 1551>
        builtin_tid3d(var_0, var_1, var_2);
        // sensorid = sensor_tendonactfrc_adr[tenactfrcid]                                        <L 1552>
        var_3 = wp::address(var_sensor_tendonactfrc_adr, var_1);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // if actuator_trntype[actid] == TrnType.TENDON and actuator_trnid[actid][0] == sensor_objid[sensorid]:       <L 1554>
        var_6 = wp::address(var_actuator_trntype, var_2);
        var_9 = wp::load(var_6);
        var_8 = (var_9 == var_7);
        var_10 = wp::address(var_actuator_trnid, var_2);
        var_13 = wp::load(var_10);
        var_12 = wp::extract(var_13, var_11);
        var_14 = wp::address(var_sensor_objid, var_4);
        var_16 = wp::load(var_14);
        var_15 = (var_12 == var_16);
        var_17 = var_8 && var_15;
        if (var_17) {
            // adr = sensor_adr[sensorid]                                                         <L 1555>
            var_18 = wp::address(var_sensor_adr, var_4);
            var_20 = wp::load(var_18);
            var_19 = wp::copy(var_20);
            // sensordata_out[worldid, adr] += actuator_force_in[worldid, actid]                  <L 1556>
            var_21 = wp::address(var_actuator_force_in, var_0, var_2);
            var_23 = wp::load(var_21);
            var_22 = wp::atomic_add(var_sensordata_out, var_0, var_19, var_23);
        }
    }
}



extern "C" __global__ void _sensor_acc_683a4930_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_opt_cone,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_objtype,
    wp::array_t<wp::int32> var_sensor_objid,
    wp::array_t<wp::int32> var_sensor_intprm,
    wp::array_t<wp::int32> var_sensor_dim,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::array_t<wp::int32> var_sensor_acc_adr,
    wp::array_t<wp::int32> var_sensor_adr_to_contact_adr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::float32> var_actuator_force_in,
    wp::array_t<wp::float32> var_qfrc_actuator_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cacc_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cfrc_int_in,
    wp::array_t<wp::float32> var_contact_dist_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::float32> var_efc_force_in,
    wp::int32 var_njmax_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::int32> var_sensor_contact_nmatch_in,
    wp::array_t<wp::int32> var_sensor_contact_matchid_in,
    wp::array_t<wp::float32> var_sensor_contact_direction_in,
    wp::array_t<wp::float32> var_sensordata_out)
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
        wp::slice_t var_11;
        const wp::int32 var_12 = 0;
        wp::array_t<wp::float32> var_13;
        const wp::int32 var_14 = 42;
        bool var_15;
        const wp::int32 var_16 = 0;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::int32* var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        const wp::int32 var_26 = 1;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        const bool var_30 = false;
        const bool var_31 = false;
        const bool var_32 = false;
        const bool var_33 = false;
        const bool var_34 = false;
        const bool var_35 = false;
        const bool var_36 = false;
        const wp::int32 var_37 = 0;
        wp::int32 var_38;
        const wp::int32 var_39 = 0;
        const wp::int32 var_40 = 1;
        wp::int32 var_41;
        wp::int32 var_42;
        const wp::int32 var_43 = 0;
        bool var_44;
        const bool var_45 = true;
        const wp::int32 var_46 = 1;
        wp::int32 var_47;
        bool var_48;
        wp::int32 var_49;
        const wp::int32 var_50 = 1;
        bool var_51;
        const bool var_52 = true;
        const wp::int32 var_53 = 3;
        wp::int32 var_54;
        bool var_55;
        wp::int32 var_56;
        const wp::int32 var_57 = 2;
        bool var_58;
        const bool var_59 = true;
        const wp::int32 var_60 = 3;
        wp::int32 var_61;
        bool var_62;
        wp::int32 var_63;
        const wp::int32 var_64 = 3;
        bool var_65;
        const bool var_66 = true;
        const wp::int32 var_67 = 1;
        wp::int32 var_68;
        bool var_69;
        wp::int32 var_70;
        const wp::int32 var_71 = 4;
        bool var_72;
        const bool var_73 = true;
        const wp::int32 var_74 = 3;
        wp::int32 var_75;
        bool var_76;
        wp::int32 var_77;
        const wp::int32 var_78 = 5;
        bool var_79;
        const bool var_80 = true;
        const wp::int32 var_81 = 3;
        wp::int32 var_82;
        bool var_83;
        wp::int32 var_84;
        const wp::int32 var_85 = 6;
        bool var_86;
        const bool var_87 = true;
        const wp::int32 var_88 = 3;
        wp::int32 var_89;
        bool var_90;
        wp::int32 var_91;
        bool var_92;
        wp::int32 var_93;
        bool var_94;
        bool var_95;
        wp::int32 var_96;
        bool var_97;
        bool var_98;
        bool var_99;
        wp::int32 var_100;
        bool var_101;
        bool var_102;
        bool var_103;
        bool var_104;
        wp::int32 var_105;
        bool var_106;
        bool var_107;
        bool var_108;
        bool var_109;
        bool var_110;
        wp::int32 var_111;
        bool var_112;
        bool var_113;
        bool var_114;
        bool var_115;
        bool var_116;
        bool var_117;
        wp::int32 var_118;
        bool var_119;
        bool var_120;
        bool var_121;
        bool var_122;
        bool var_123;
        bool var_124;
        bool var_125;
        wp::int32 var_126;
        const wp::int32 var_127 = 1;
        const wp::int32 var_128 = 1;
        wp::int32 var_129;
        wp::int32 var_130;
        const wp::int32 var_131 = 0;
        bool var_132;
        const bool var_133 = true;
        const wp::int32 var_134 = 1;
        wp::int32 var_135;
        bool var_136;
        wp::int32 var_137;
        const wp::int32 var_138 = 1;
        bool var_139;
        const bool var_140 = true;
        const wp::int32 var_141 = 3;
        wp::int32 var_142;
        bool var_143;
        wp::int32 var_144;
        const wp::int32 var_145 = 2;
        bool var_146;
        const bool var_147 = true;
        const wp::int32 var_148 = 3;
        wp::int32 var_149;
        bool var_150;
        wp::int32 var_151;
        const wp::int32 var_152 = 3;
        bool var_153;
        const bool var_154 = true;
        const wp::int32 var_155 = 1;
        wp::int32 var_156;
        bool var_157;
        wp::int32 var_158;
        const wp::int32 var_159 = 4;
        bool var_160;
        const bool var_161 = true;
        const wp::int32 var_162 = 3;
        wp::int32 var_163;
        bool var_164;
        wp::int32 var_165;
        const wp::int32 var_166 = 5;
        bool var_167;
        const bool var_168 = true;
        const wp::int32 var_169 = 3;
        wp::int32 var_170;
        bool var_171;
        wp::int32 var_172;
        const wp::int32 var_173 = 6;
        bool var_174;
        const bool var_175 = true;
        const wp::int32 var_176 = 3;
        wp::int32 var_177;
        bool var_178;
        wp::int32 var_179;
        bool var_180;
        wp::int32 var_181;
        bool var_182;
        bool var_183;
        wp::int32 var_184;
        bool var_185;
        bool var_186;
        bool var_187;
        wp::int32 var_188;
        bool var_189;
        bool var_190;
        bool var_191;
        bool var_192;
        wp::int32 var_193;
        bool var_194;
        bool var_195;
        bool var_196;
        bool var_197;
        bool var_198;
        wp::int32 var_199;
        bool var_200;
        bool var_201;
        bool var_202;
        bool var_203;
        bool var_204;
        bool var_205;
        wp::int32 var_206;
        bool var_207;
        bool var_208;
        bool var_209;
        bool var_210;
        bool var_211;
        bool var_212;
        bool var_213;
        wp::int32 var_214;
        const wp::int32 var_215 = 2;
        const wp::int32 var_216 = 1;
        wp::int32 var_217;
        wp::int32 var_218;
        const wp::int32 var_219 = 0;
        bool var_220;
        const bool var_221 = true;
        const wp::int32 var_222 = 1;
        wp::int32 var_223;
        bool var_224;
        wp::int32 var_225;
        const wp::int32 var_226 = 1;
        bool var_227;
        const bool var_228 = true;
        const wp::int32 var_229 = 3;
        wp::int32 var_230;
        bool var_231;
        wp::int32 var_232;
        const wp::int32 var_233 = 2;
        bool var_234;
        const bool var_235 = true;
        const wp::int32 var_236 = 3;
        wp::int32 var_237;
        bool var_238;
        wp::int32 var_239;
        const wp::int32 var_240 = 3;
        bool var_241;
        const bool var_242 = true;
        const wp::int32 var_243 = 1;
        wp::int32 var_244;
        bool var_245;
        wp::int32 var_246;
        const wp::int32 var_247 = 4;
        bool var_248;
        const bool var_249 = true;
        const wp::int32 var_250 = 3;
        wp::int32 var_251;
        bool var_252;
        wp::int32 var_253;
        const wp::int32 var_254 = 5;
        bool var_255;
        const bool var_256 = true;
        const wp::int32 var_257 = 3;
        wp::int32 var_258;
        bool var_259;
        wp::int32 var_260;
        const wp::int32 var_261 = 6;
        bool var_262;
        const bool var_263 = true;
        const wp::int32 var_264 = 3;
        wp::int32 var_265;
        bool var_266;
        wp::int32 var_267;
        bool var_268;
        wp::int32 var_269;
        bool var_270;
        bool var_271;
        wp::int32 var_272;
        bool var_273;
        bool var_274;
        bool var_275;
        wp::int32 var_276;
        bool var_277;
        bool var_278;
        bool var_279;
        bool var_280;
        wp::int32 var_281;
        bool var_282;
        bool var_283;
        bool var_284;
        bool var_285;
        bool var_286;
        wp::int32 var_287;
        bool var_288;
        bool var_289;
        bool var_290;
        bool var_291;
        bool var_292;
        bool var_293;
        wp::int32 var_294;
        bool var_295;
        bool var_296;
        bool var_297;
        bool var_298;
        bool var_299;
        bool var_300;
        bool var_301;
        wp::int32 var_302;
        const wp::int32 var_303 = 3;
        const wp::int32 var_304 = 1;
        wp::int32 var_305;
        wp::int32 var_306;
        const wp::int32 var_307 = 0;
        bool var_308;
        const bool var_309 = true;
        const wp::int32 var_310 = 1;
        wp::int32 var_311;
        bool var_312;
        wp::int32 var_313;
        const wp::int32 var_314 = 1;
        bool var_315;
        const bool var_316 = true;
        const wp::int32 var_317 = 3;
        wp::int32 var_318;
        bool var_319;
        wp::int32 var_320;
        const wp::int32 var_321 = 2;
        bool var_322;
        const bool var_323 = true;
        const wp::int32 var_324 = 3;
        wp::int32 var_325;
        bool var_326;
        wp::int32 var_327;
        const wp::int32 var_328 = 3;
        bool var_329;
        const bool var_330 = true;
        const wp::int32 var_331 = 1;
        wp::int32 var_332;
        bool var_333;
        wp::int32 var_334;
        const wp::int32 var_335 = 4;
        bool var_336;
        const bool var_337 = true;
        const wp::int32 var_338 = 3;
        wp::int32 var_339;
        bool var_340;
        wp::int32 var_341;
        const wp::int32 var_342 = 5;
        bool var_343;
        const bool var_344 = true;
        const wp::int32 var_345 = 3;
        wp::int32 var_346;
        bool var_347;
        wp::int32 var_348;
        const wp::int32 var_349 = 6;
        bool var_350;
        const bool var_351 = true;
        const wp::int32 var_352 = 3;
        wp::int32 var_353;
        bool var_354;
        wp::int32 var_355;
        bool var_356;
        wp::int32 var_357;
        bool var_358;
        bool var_359;
        wp::int32 var_360;
        bool var_361;
        bool var_362;
        bool var_363;
        wp::int32 var_364;
        bool var_365;
        bool var_366;
        bool var_367;
        bool var_368;
        wp::int32 var_369;
        bool var_370;
        bool var_371;
        bool var_372;
        bool var_373;
        bool var_374;
        wp::int32 var_375;
        bool var_376;
        bool var_377;
        bool var_378;
        bool var_379;
        bool var_380;
        bool var_381;
        wp::int32 var_382;
        bool var_383;
        bool var_384;
        bool var_385;
        bool var_386;
        bool var_387;
        bool var_388;
        bool var_389;
        wp::int32 var_390;
        const wp::int32 var_391 = 4;
        const wp::int32 var_392 = 1;
        wp::int32 var_393;
        wp::int32 var_394;
        const wp::int32 var_395 = 0;
        bool var_396;
        const bool var_397 = true;
        const wp::int32 var_398 = 1;
        wp::int32 var_399;
        bool var_400;
        wp::int32 var_401;
        const wp::int32 var_402 = 1;
        bool var_403;
        const bool var_404 = true;
        const wp::int32 var_405 = 3;
        wp::int32 var_406;
        bool var_407;
        wp::int32 var_408;
        const wp::int32 var_409 = 2;
        bool var_410;
        const bool var_411 = true;
        const wp::int32 var_412 = 3;
        wp::int32 var_413;
        bool var_414;
        wp::int32 var_415;
        const wp::int32 var_416 = 3;
        bool var_417;
        const bool var_418 = true;
        const wp::int32 var_419 = 1;
        wp::int32 var_420;
        bool var_421;
        wp::int32 var_422;
        const wp::int32 var_423 = 4;
        bool var_424;
        const bool var_425 = true;
        const wp::int32 var_426 = 3;
        wp::int32 var_427;
        bool var_428;
        wp::int32 var_429;
        const wp::int32 var_430 = 5;
        bool var_431;
        const bool var_432 = true;
        const wp::int32 var_433 = 3;
        wp::int32 var_434;
        bool var_435;
        wp::int32 var_436;
        const wp::int32 var_437 = 6;
        bool var_438;
        const bool var_439 = true;
        const wp::int32 var_440 = 3;
        wp::int32 var_441;
        bool var_442;
        wp::int32 var_443;
        bool var_444;
        wp::int32 var_445;
        bool var_446;
        bool var_447;
        wp::int32 var_448;
        bool var_449;
        bool var_450;
        bool var_451;
        wp::int32 var_452;
        bool var_453;
        bool var_454;
        bool var_455;
        bool var_456;
        wp::int32 var_457;
        bool var_458;
        bool var_459;
        bool var_460;
        bool var_461;
        bool var_462;
        wp::int32 var_463;
        bool var_464;
        bool var_465;
        bool var_466;
        bool var_467;
        bool var_468;
        bool var_469;
        wp::int32 var_470;
        bool var_471;
        bool var_472;
        bool var_473;
        bool var_474;
        bool var_475;
        bool var_476;
        bool var_477;
        wp::int32 var_478;
        const wp::int32 var_479 = 5;
        const wp::int32 var_480 = 1;
        wp::int32 var_481;
        wp::int32 var_482;
        const wp::int32 var_483 = 0;
        bool var_484;
        const bool var_485 = true;
        const wp::int32 var_486 = 1;
        wp::int32 var_487;
        bool var_488;
        wp::int32 var_489;
        const wp::int32 var_490 = 1;
        bool var_491;
        const bool var_492 = true;
        const wp::int32 var_493 = 3;
        wp::int32 var_494;
        bool var_495;
        wp::int32 var_496;
        const wp::int32 var_497 = 2;
        bool var_498;
        const bool var_499 = true;
        const wp::int32 var_500 = 3;
        wp::int32 var_501;
        bool var_502;
        wp::int32 var_503;
        const wp::int32 var_504 = 3;
        bool var_505;
        const bool var_506 = true;
        const wp::int32 var_507 = 1;
        wp::int32 var_508;
        bool var_509;
        wp::int32 var_510;
        const wp::int32 var_511 = 4;
        bool var_512;
        const bool var_513 = true;
        const wp::int32 var_514 = 3;
        wp::int32 var_515;
        bool var_516;
        wp::int32 var_517;
        const wp::int32 var_518 = 5;
        bool var_519;
        const bool var_520 = true;
        const wp::int32 var_521 = 3;
        wp::int32 var_522;
        bool var_523;
        wp::int32 var_524;
        const wp::int32 var_525 = 6;
        bool var_526;
        const bool var_527 = true;
        const wp::int32 var_528 = 3;
        wp::int32 var_529;
        bool var_530;
        wp::int32 var_531;
        bool var_532;
        wp::int32 var_533;
        bool var_534;
        bool var_535;
        wp::int32 var_536;
        bool var_537;
        bool var_538;
        bool var_539;
        wp::int32 var_540;
        bool var_541;
        bool var_542;
        bool var_543;
        bool var_544;
        wp::int32 var_545;
        bool var_546;
        bool var_547;
        bool var_548;
        bool var_549;
        bool var_550;
        wp::int32 var_551;
        bool var_552;
        bool var_553;
        bool var_554;
        bool var_555;
        bool var_556;
        bool var_557;
        wp::int32 var_558;
        bool var_559;
        bool var_560;
        bool var_561;
        bool var_562;
        bool var_563;
        bool var_564;
        bool var_565;
        wp::int32 var_566;
        const wp::int32 var_567 = 6;
        const wp::int32 var_568 = 1;
        wp::int32 var_569;
        wp::int32 var_570;
        const wp::int32 var_571 = 0;
        bool var_572;
        const bool var_573 = true;
        const wp::int32 var_574 = 1;
        wp::int32 var_575;
        bool var_576;
        wp::int32 var_577;
        const wp::int32 var_578 = 1;
        bool var_579;
        const bool var_580 = true;
        const wp::int32 var_581 = 3;
        wp::int32 var_582;
        bool var_583;
        wp::int32 var_584;
        const wp::int32 var_585 = 2;
        bool var_586;
        const bool var_587 = true;
        const wp::int32 var_588 = 3;
        wp::int32 var_589;
        bool var_590;
        wp::int32 var_591;
        const wp::int32 var_592 = 3;
        bool var_593;
        const bool var_594 = true;
        const wp::int32 var_595 = 1;
        wp::int32 var_596;
        bool var_597;
        wp::int32 var_598;
        const wp::int32 var_599 = 4;
        bool var_600;
        const bool var_601 = true;
        const wp::int32 var_602 = 3;
        wp::int32 var_603;
        bool var_604;
        wp::int32 var_605;
        const wp::int32 var_606 = 5;
        bool var_607;
        const bool var_608 = true;
        const wp::int32 var_609 = 3;
        wp::int32 var_610;
        bool var_611;
        wp::int32 var_612;
        const wp::int32 var_613 = 6;
        bool var_614;
        const bool var_615 = true;
        const wp::int32 var_616 = 3;
        wp::int32 var_617;
        bool var_618;
        wp::int32 var_619;
        bool var_620;
        wp::int32 var_621;
        bool var_622;
        bool var_623;
        wp::int32 var_624;
        bool var_625;
        bool var_626;
        bool var_627;
        wp::int32 var_628;
        bool var_629;
        bool var_630;
        bool var_631;
        bool var_632;
        wp::int32 var_633;
        bool var_634;
        bool var_635;
        bool var_636;
        bool var_637;
        bool var_638;
        wp::int32 var_639;
        bool var_640;
        bool var_641;
        bool var_642;
        bool var_643;
        bool var_644;
        bool var_645;
        wp::int32 var_646;
        bool var_647;
        bool var_648;
        bool var_649;
        bool var_650;
        bool var_651;
        bool var_652;
        bool var_653;
        wp::int32 var_654;
        wp::int32 var_655;
        wp::int32* var_656;
        wp::int32 var_657;
        wp::int32 var_658;
        wp::int32* var_659;
        wp::int32 var_660;
        wp::int32 var_661;
        wp::int32* var_662;
        wp::int32 var_663;
        wp::int32 var_664;
        const wp::int32 var_665 = 3;
        bool var_666;
        const wp::float32 var_667 = 0.0;
        wp::vec_t<3, wp::float32> var_668;
        const wp::float32 var_669 = 0.0;
        wp::vec_t<3, wp::float32> var_670;
        const wp::float32 var_671 = 0.0;
        wp::vec_t<3, wp::float32> var_672;
        const wp::float32 var_673 = 0.0;
        wp::float32 var_674;
        wp::range_t var_675;
        wp::int32 var_676;
        wp::int32* var_677;
        wp::int32 var_678;
        wp::int32 var_679;
        wp::float32* var_680;
        wp::float32 var_681;
        wp::float32 var_682;
        const bool var_683 = false;
        wp::vec_t<6, wp::float32> var_684;
        wp::vec_t<3, wp::float32> var_685;
        wp::float32 var_686;
        wp::vec_t<3, wp::float32>* var_687;
        wp::vec_t<3, wp::float32> var_688;
        wp::vec_t<3, wp::float32> var_689;
        wp::vec_t<3, wp::float32> var_690;
        wp::vec_t<3, wp::float32> var_691;
        wp::float32 var_692;
        wp::vec_t<6, wp::float32> var_693;
        wp::vec_t<3, wp::float32> var_694;
        wp::vec_t<3, wp::float32> var_695;
        wp::mat_t<3, 3, wp::float32>* var_696;
        wp::mat_t<3, 3, wp::float32> var_697;
        wp::mat_t<3, 3, wp::float32> var_698;
        wp::mat_t<3, 3, wp::float32> var_699;
        wp::vec_t<3, wp::float32> var_700;
        wp::vec_t<3, wp::float32> var_701;
        wp::vec_t<3, wp::float32> var_702;
        wp::vec_t<3, wp::float32> var_703;
        wp::vec_t<3, wp::float32> var_704;
        wp::vec_t<3, wp::float32> var_705;
        const wp::float32 var_706 = 1e-15;
        wp::float32 var_707;
        wp::vec_t<3, wp::float32> var_708;
        wp::vec_t<3, wp::float32> var_709;
        wp::vec_t<3, wp::float32> var_710;
        wp::int32 var_711;
        wp::float32 var_712;
        const wp::int32 var_713 = 1;
        wp::int32 var_714;
        wp::int32 var_715;
        const wp::int32 var_716 = 0;
        wp::float32 var_717;
        const wp::int32 var_718 = 0;
        wp::int32 var_719;
        const wp::int32 var_720 = 1;
        wp::float32 var_721;
        const wp::int32 var_722 = 1;
        wp::int32 var_723;
        const wp::int32 var_724 = 2;
        wp::float32 var_725;
        const wp::int32 var_726 = 2;
        wp::int32 var_727;
        const wp::int32 var_728 = 3;
        wp::int32 var_729;
        wp::int32 var_730;
        const wp::int32 var_731 = 0;
        wp::float32 var_732;
        const wp::int32 var_733 = 0;
        wp::int32 var_734;
        const wp::int32 var_735 = 1;
        wp::float32 var_736;
        const wp::int32 var_737 = 1;
        wp::int32 var_738;
        const wp::int32 var_739 = 2;
        wp::float32 var_740;
        const wp::int32 var_741 = 2;
        wp::int32 var_742;
        const wp::int32 var_743 = 3;
        wp::int32 var_744;
        wp::int32 var_745;
        const wp::float32 var_746 = 0.0;
        const wp::int32 var_747 = 1;
        wp::int32 var_748;
        wp::int32 var_749;
        const wp::int32 var_750 = 0;
        wp::float32 var_751;
        const wp::int32 var_752 = 0;
        wp::int32 var_753;
        const wp::int32 var_754 = 1;
        wp::float32 var_755;
        const wp::int32 var_756 = 1;
        wp::int32 var_757;
        const wp::int32 var_758 = 2;
        wp::float32 var_759;
        const wp::int32 var_760 = 2;
        wp::int32 var_761;
        const wp::int32 var_762 = 3;
        wp::int32 var_763;
        wp::int32 var_764;
        const wp::float32 var_765 = 1.0;
        const wp::int32 var_766 = 0;
        wp::int32 var_767;
        const wp::float32 var_768 = 0.0;
        const wp::int32 var_769 = 1;
        wp::int32 var_770;
        const wp::float32 var_771 = 0.0;
        const wp::int32 var_772 = 2;
        wp::int32 var_773;
        const wp::int32 var_774 = 3;
        wp::int32 var_775;
        wp::int32 var_776;
        const wp::float32 var_777 = 0.0;
        const wp::int32 var_778 = 0;
        wp::int32 var_779;
        const wp::float32 var_780 = 1.0;
        const wp::int32 var_781 = 1;
        wp::int32 var_782;
        const wp::float32 var_783 = 0.0;
        const wp::int32 var_784 = 2;
        wp::int32 var_785;
        wp::int32 var_786;
        wp::int32 var_787;
        wp::range_t var_788;
        wp::int32 var_789;
        wp::int32* var_790;
        wp::int32 var_791;
        wp::int32 var_792;
        wp::float32* var_793;
        wp::float32 var_794;
        wp::float32 var_795;
        wp::int32 var_796;
        wp::int32 var_797;
        wp::float32 var_798;
        const wp::int32 var_799 = 1;
        wp::int32 var_800;
        wp::int32 var_801;
        bool var_802;
        const bool var_803 = false;
        wp::vec_t<6, wp::float32> var_804;
        wp::vec_t<6, wp::float32> var_805;
        const wp::int32 var_806 = 0;
        wp::float32 var_807;
        const wp::int32 var_808 = 0;
        wp::int32 var_809;
        const wp::int32 var_810 = 1;
        wp::float32 var_811;
        const wp::int32 var_812 = 1;
        wp::int32 var_813;
        const wp::int32 var_814 = 2;
        wp::float32 var_815;
        wp::float32 var_816;
        const wp::int32 var_817 = 2;
        wp::int32 var_818;
        const wp::int32 var_819 = 3;
        wp::int32 var_820;
        wp::int32 var_821;
        const wp::int32 var_822 = 3;
        wp::float32 var_823;
        const wp::int32 var_824 = 0;
        wp::int32 var_825;
        const wp::int32 var_826 = 4;
        wp::float32 var_827;
        const wp::int32 var_828 = 1;
        wp::int32 var_829;
        const wp::int32 var_830 = 5;
        wp::float32 var_831;
        wp::float32 var_832;
        const wp::int32 var_833 = 2;
        wp::int32 var_834;
        const wp::int32 var_835 = 3;
        wp::int32 var_836;
        wp::int32 var_837;
        wp::float32* var_838;
        wp::float32 var_839;
        const wp::int32 var_840 = 1;
        wp::int32 var_841;
        wp::int32 var_842;
        wp::vec_t<3, wp::float32>* var_843;
        wp::vec_t<3, wp::float32> var_844;
        wp::vec_t<3, wp::float32> var_845;
        const wp::int32 var_846 = 0;
        wp::float32 var_847;
        const wp::int32 var_848 = 0;
        wp::int32 var_849;
        const wp::int32 var_850 = 1;
        wp::float32 var_851;
        const wp::int32 var_852 = 1;
        wp::int32 var_853;
        const wp::int32 var_854 = 2;
        wp::float32 var_855;
        const wp::int32 var_856 = 2;
        wp::int32 var_857;
        const wp::int32 var_858 = 3;
        wp::int32 var_859;
        wp::vec_t<3, wp::float32> var_860;
        wp::int32 var_861;
        wp::mat_t<3, 3, wp::float32>* var_862;
        const wp::int32 var_863 = 0;
        wp::vec_t<3, wp::float32> var_864;
        wp::mat_t<3, 3, wp::float32> var_865;
        const wp::int32 var_866 = 0;
        wp::float32 var_867;
        wp::float32 var_868;
        const wp::int32 var_869 = 0;
        wp::int32 var_870;
        const wp::int32 var_871 = 1;
        wp::float32 var_872;
        wp::float32 var_873;
        const wp::int32 var_874 = 1;
        wp::int32 var_875;
        const wp::int32 var_876 = 2;
        wp::float32 var_877;
        wp::float32 var_878;
        const wp::int32 var_879 = 2;
        wp::int32 var_880;
        const wp::int32 var_881 = 3;
        wp::int32 var_882;
        wp::int32 var_883;
        wp::mat_t<3, 3, wp::float32>* var_884;
        const wp::int32 var_885 = 1;
        wp::vec_t<3, wp::float32> var_886;
        wp::mat_t<3, 3, wp::float32> var_887;
        const wp::int32 var_888 = 0;
        wp::float32 var_889;
        wp::float32 var_890;
        const wp::int32 var_891 = 0;
        wp::int32 var_892;
        const wp::int32 var_893 = 1;
        wp::float32 var_894;
        wp::float32 var_895;
        const wp::int32 var_896 = 1;
        wp::int32 var_897;
        const wp::int32 var_898 = 2;
        wp::float32 var_899;
        wp::float32 var_900;
        const wp::int32 var_901 = 2;
        wp::int32 var_902;
        wp::range_t var_903;
        wp::int32 var_904;
        wp::range_t var_905;
        wp::int32 var_906;
        const wp::float32 var_907 = 0.0;
        wp::int32 var_908;
        wp::int32 var_909;
        wp::int32 var_910;
        wp::int32 var_911;
        const wp::int32 var_912 = 1;
        bool var_913;
        wp::vec_t<3, wp::float32> var_914;
        const wp::int32 var_915 = 3;
        const wp::int32 var_916 = 4;
        bool var_917;
        wp::vec_t<3, wp::float32> var_918;
        const wp::int32 var_919 = 3;
        wp::vec_t<3, wp::float32> var_920;
        const wp::int32 var_921 = 5;
        bool var_922;
        wp::vec_t<3, wp::float32> var_923;
        const wp::int32 var_924 = 3;
        wp::vec_t<3, wp::float32> var_925;
        const wp::int32 var_926 = 15;
        bool var_927;
        wp::float32 var_928;
        const wp::int32 var_929 = 16;
        bool var_930;
        wp::float32 var_931;
        wp::float32 var_932;
        const wp::int32 var_933 = 33;
        bool var_934;
        wp::int32* var_935;
        wp::int32 var_936;
        wp::int32 var_937;
        wp::vec_t<3, wp::float32> var_938;
        const wp::int32 var_939 = 3;
        wp::int32 var_940;
        wp::vec_t<3, wp::float32> var_941;
        const wp::int32 var_942 = 34;
        bool var_943;
        wp::int32* var_944;
        wp::int32 var_945;
        wp::int32 var_946;
        wp::vec_t<3, wp::float32> var_947;
        const wp::int32 var_948 = 3;
        wp::int32 var_949;
        wp::vec_t<3, wp::float32> var_950;
        wp::int32 var_951;
        wp::vec_t<3, wp::float32> var_952;
        wp::int32 var_953;
        wp::vec_t<3, wp::float32> var_954;
        wp::int32 var_955;
        wp::vec_t<3, wp::float32> var_956;
        wp::float32 var_957;
        wp::int32 var_958;
        wp::vec_t<3, wp::float32> var_959;
        wp::int32 var_960;
        wp::vec_t<3, wp::float32> var_961;
        wp::int32 var_962;
        wp::vec_t<3, wp::float32> var_963;
        wp::int32 var_964;
        //---------
        // forward
        // def _sensor_acc(                                                                       <L 1697>
        // worldid, accid = wp.tid()                                                              <L 1744>
        builtin_tid2d(var_0, var_1);
        // sensorid = sensor_acc_adr[accid]                                                       <L 1745>
        var_2 = wp::address(var_sensor_acc_adr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // sensortype = sensor_type[sensorid]                                                     <L 1746>
        var_5 = wp::address(var_sensor_type, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // objid = sensor_objid[sensorid]                                                         <L 1747>
        var_8 = wp::address(var_sensor_objid, var_3);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // out = sensordata_out[worldid]                                                          <L 1748>
        var_11 = wp::slice_t(var_0, var_0, var_12);
        var_13 = wp::view(var_sensordata_out, var_11);
        // if sensortype == SensorType.CONTACT:                                                   <L 1750>
        var_15 = (var_6 == var_14);
        if (var_15) {
            // dataspec = sensor_intprm[sensorid, 0]                                              <L 1751>
            var_17 = wp::address(var_sensor_intprm, var_3, var_16);
            var_19 = wp::load(var_17);
            var_18 = wp::copy(var_19);
            // dim = sensor_dim[sensorid]                                                         <L 1752>
            var_20 = wp::address(var_sensor_dim, var_3);
            var_22 = wp::load(var_20);
            var_21 = wp::copy(var_22);
            // objtype = sensor_objtype[sensorid]                                                 <L 1753>
            var_23 = wp::address(var_sensor_objtype, var_3);
            var_25 = wp::load(var_23);
            var_24 = wp::copy(var_25);
            // reduce = sensor_intprm[sensorid, 1]                                                <L 1754>
            var_27 = wp::address(var_sensor_intprm, var_3, var_26);
            var_29 = wp::load(var_27);
            var_28 = wp::copy(var_29);
            // found = False                                                                      <L 1758>
            // force = False                                                                      <L 1759>
            // torque = False                                                                     <L 1760>
            // dist = False                                                                       <L 1761>
            // pos = False                                                                        <L 1762>
            // normal = False                                                                     <L 1763>
            // tangent = False                                                                    <L 1764>
            // size = int(0)                                                                      <L 1766>
            var_38 = wp::int(var_37);
            // for i in range(7):                                                                 <L 1767>
            // if dataspec & (1 << i):                                                            <L 1768>
            var_41 = wp::lshift(var_40, var_39);
            var_42 = wp::bit_and(var_18, var_41);
            if (var_42) {
                // if i == 0:                                                                     <L 1769>
                var_44 = (var_39 == var_43);
                if (var_44) {
                    // found = True                                                               <L 1770>
                    // size += 1                                                                  <L 1771>
                    var_47 = wp::add(var_38, var_46);
                }
                var_48 = wp::where(var_44, var_45, var_30);
                var_49 = wp::where(var_44, var_47, var_38);
                if (!var_44) {
                    // elif i == 1:                                                               <L 1772>
                    var_51 = (var_39 == var_50);
                    if (var_51) {
                        // force = True                                                           <L 1773>
                        // size += 3                                                              <L 1774>
                        var_54 = wp::add(var_49, var_53);
                    }
                    var_55 = wp::where(var_51, var_52, var_31);
                    var_56 = wp::where(var_51, var_54, var_49);
                    if (!var_51) {
                        // elif i == 2:                                                           <L 1775>
                        var_58 = (var_39 == var_57);
                        if (var_58) {
                            // torque = True                                                      <L 1776>
                            // size += 3                                                          <L 1777>
                            var_61 = wp::add(var_56, var_60);
                        }
                        var_62 = wp::where(var_58, var_59, var_32);
                        var_63 = wp::where(var_58, var_61, var_56);
                        if (!var_58) {
                            // elif i == 3:                                                       <L 1778>
                            var_65 = (var_39 == var_64);
                            if (var_65) {
                                // dist = True                                                    <L 1779>
                                // size += 1                                                      <L 1780>
                                var_68 = wp::add(var_63, var_67);
                            }
                            var_69 = wp::where(var_65, var_66, var_33);
                            var_70 = wp::where(var_65, var_68, var_63);
                            if (!var_65) {
                                // elif i == 4:                                                   <L 1781>
                                var_72 = (var_39 == var_71);
                                if (var_72) {
                                    // pos = True                                                 <L 1782>
                                    // size += 3                                                  <L 1783>
                                    var_75 = wp::add(var_70, var_74);
                                }
                                var_76 = wp::where(var_72, var_73, var_34);
                                var_77 = wp::where(var_72, var_75, var_70);
                                if (!var_72) {
                                    // elif i == 5:                                               <L 1784>
                                    var_79 = (var_39 == var_78);
                                    if (var_79) {
                                        // normal = True                                          <L 1785>
                                        // size += 3                                              <L 1786>
                                        var_82 = wp::add(var_77, var_81);
                                    }
                                    var_83 = wp::where(var_79, var_80, var_35);
                                    var_84 = wp::where(var_79, var_82, var_77);
                                    if (!var_79) {
                                        // elif i == 6:                                           <L 1787>
                                        var_86 = (var_39 == var_85);
                                        if (var_86) {
                                            // tangent = True                                     <L 1788>
                                            // size += 3                                          <L 1789>
                                            var_89 = wp::add(var_84, var_88);
                                        }
                                        var_90 = wp::where(var_86, var_87, var_36);
                                        var_91 = wp::where(var_86, var_89, var_84);
                                    }
                                    var_92 = wp::where(var_79, var_36, var_90);
                                    var_93 = wp::where(var_79, var_84, var_91);
                                }
                                var_94 = wp::where(var_72, var_35, var_83);
                                var_95 = wp::where(var_72, var_36, var_92);
                                var_96 = wp::where(var_72, var_77, var_93);
                            }
                            var_97 = wp::where(var_65, var_34, var_76);
                            var_98 = wp::where(var_65, var_35, var_94);
                            var_99 = wp::where(var_65, var_36, var_95);
                            var_100 = wp::where(var_65, var_70, var_96);
                        }
                        var_101 = wp::where(var_58, var_33, var_69);
                        var_102 = wp::where(var_58, var_34, var_97);
                        var_103 = wp::where(var_58, var_35, var_98);
                        var_104 = wp::where(var_58, var_36, var_99);
                        var_105 = wp::where(var_58, var_63, var_100);
                    }
                    var_106 = wp::where(var_51, var_32, var_62);
                    var_107 = wp::where(var_51, var_33, var_101);
                    var_108 = wp::where(var_51, var_34, var_102);
                    var_109 = wp::where(var_51, var_35, var_103);
                    var_110 = wp::where(var_51, var_36, var_104);
                    var_111 = wp::where(var_51, var_56, var_105);
                }
                var_112 = wp::where(var_44, var_31, var_55);
                var_113 = wp::where(var_44, var_32, var_106);
                var_114 = wp::where(var_44, var_33, var_107);
                var_115 = wp::where(var_44, var_34, var_108);
                var_116 = wp::where(var_44, var_35, var_109);
                var_117 = wp::where(var_44, var_36, var_110);
                var_118 = wp::where(var_44, var_49, var_111);
            }
            var_119 = wp::where(var_42, var_48, var_30);
            var_120 = wp::where(var_42, var_112, var_31);
            var_121 = wp::where(var_42, var_113, var_32);
            var_122 = wp::where(var_42, var_114, var_33);
            var_123 = wp::where(var_42, var_115, var_34);
            var_124 = wp::where(var_42, var_116, var_35);
            var_125 = wp::where(var_42, var_117, var_36);
            var_126 = wp::where(var_42, var_118, var_38);
            // if dataspec & (1 << i):                                                            <L 1768>
            var_129 = wp::lshift(var_128, var_127);
            var_130 = wp::bit_and(var_18, var_129);
            if (var_130) {
                // if i == 0:                                                                     <L 1769>
                var_132 = (var_127 == var_131);
                if (var_132) {
                    // found = True                                                               <L 1770>
                    // size += 1                                                                  <L 1771>
                    var_135 = wp::add(var_126, var_134);
                }
                var_136 = wp::where(var_132, var_133, var_119);
                var_137 = wp::where(var_132, var_135, var_126);
                if (!var_132) {
                    // elif i == 1:                                                               <L 1772>
                    var_139 = (var_127 == var_138);
                    if (var_139) {
                        // force = True                                                           <L 1773>
                        // size += 3                                                              <L 1774>
                        var_142 = wp::add(var_137, var_141);
                    }
                    var_143 = wp::where(var_139, var_140, var_120);
                    var_144 = wp::where(var_139, var_142, var_137);
                    if (!var_139) {
                        // elif i == 2:                                                           <L 1775>
                        var_146 = (var_127 == var_145);
                        if (var_146) {
                            // torque = True                                                      <L 1776>
                            // size += 3                                                          <L 1777>
                            var_149 = wp::add(var_144, var_148);
                        }
                        var_150 = wp::where(var_146, var_147, var_121);
                        var_151 = wp::where(var_146, var_149, var_144);
                        if (!var_146) {
                            // elif i == 3:                                                       <L 1778>
                            var_153 = (var_127 == var_152);
                            if (var_153) {
                                // dist = True                                                    <L 1779>
                                // size += 1                                                      <L 1780>
                                var_156 = wp::add(var_151, var_155);
                            }
                            var_157 = wp::where(var_153, var_154, var_122);
                            var_158 = wp::where(var_153, var_156, var_151);
                            if (!var_153) {
                                // elif i == 4:                                                   <L 1781>
                                var_160 = (var_127 == var_159);
                                if (var_160) {
                                    // pos = True                                                 <L 1782>
                                    // size += 3                                                  <L 1783>
                                    var_163 = wp::add(var_158, var_162);
                                }
                                var_164 = wp::where(var_160, var_161, var_123);
                                var_165 = wp::where(var_160, var_163, var_158);
                                if (!var_160) {
                                    // elif i == 5:                                               <L 1784>
                                    var_167 = (var_127 == var_166);
                                    if (var_167) {
                                        // normal = True                                          <L 1785>
                                        // size += 3                                              <L 1786>
                                        var_170 = wp::add(var_165, var_169);
                                    }
                                    var_171 = wp::where(var_167, var_168, var_124);
                                    var_172 = wp::where(var_167, var_170, var_165);
                                    if (!var_167) {
                                        // elif i == 6:                                           <L 1787>
                                        var_174 = (var_127 == var_173);
                                        if (var_174) {
                                            // tangent = True                                     <L 1788>
                                            // size += 3                                          <L 1789>
                                            var_177 = wp::add(var_172, var_176);
                                        }
                                        var_178 = wp::where(var_174, var_175, var_125);
                                        var_179 = wp::where(var_174, var_177, var_172);
                                    }
                                    var_180 = wp::where(var_167, var_125, var_178);
                                    var_181 = wp::where(var_167, var_172, var_179);
                                }
                                var_182 = wp::where(var_160, var_124, var_171);
                                var_183 = wp::where(var_160, var_125, var_180);
                                var_184 = wp::where(var_160, var_165, var_181);
                            }
                            var_185 = wp::where(var_153, var_123, var_164);
                            var_186 = wp::where(var_153, var_124, var_182);
                            var_187 = wp::where(var_153, var_125, var_183);
                            var_188 = wp::where(var_153, var_158, var_184);
                        }
                        var_189 = wp::where(var_146, var_122, var_157);
                        var_190 = wp::where(var_146, var_123, var_185);
                        var_191 = wp::where(var_146, var_124, var_186);
                        var_192 = wp::where(var_146, var_125, var_187);
                        var_193 = wp::where(var_146, var_151, var_188);
                    }
                    var_194 = wp::where(var_139, var_121, var_150);
                    var_195 = wp::where(var_139, var_122, var_189);
                    var_196 = wp::where(var_139, var_123, var_190);
                    var_197 = wp::where(var_139, var_124, var_191);
                    var_198 = wp::where(var_139, var_125, var_192);
                    var_199 = wp::where(var_139, var_144, var_193);
                }
                var_200 = wp::where(var_132, var_120, var_143);
                var_201 = wp::where(var_132, var_121, var_194);
                var_202 = wp::where(var_132, var_122, var_195);
                var_203 = wp::where(var_132, var_123, var_196);
                var_204 = wp::where(var_132, var_124, var_197);
                var_205 = wp::where(var_132, var_125, var_198);
                var_206 = wp::where(var_132, var_137, var_199);
            }
            var_207 = wp::where(var_130, var_136, var_119);
            var_208 = wp::where(var_130, var_200, var_120);
            var_209 = wp::where(var_130, var_201, var_121);
            var_210 = wp::where(var_130, var_202, var_122);
            var_211 = wp::where(var_130, var_203, var_123);
            var_212 = wp::where(var_130, var_204, var_124);
            var_213 = wp::where(var_130, var_205, var_125);
            var_214 = wp::where(var_130, var_206, var_126);
            // if dataspec & (1 << i):                                                            <L 1768>
            var_217 = wp::lshift(var_216, var_215);
            var_218 = wp::bit_and(var_18, var_217);
            if (var_218) {
                // if i == 0:                                                                     <L 1769>
                var_220 = (var_215 == var_219);
                if (var_220) {
                    // found = True                                                               <L 1770>
                    // size += 1                                                                  <L 1771>
                    var_223 = wp::add(var_214, var_222);
                }
                var_224 = wp::where(var_220, var_221, var_207);
                var_225 = wp::where(var_220, var_223, var_214);
                if (!var_220) {
                    // elif i == 1:                                                               <L 1772>
                    var_227 = (var_215 == var_226);
                    if (var_227) {
                        // force = True                                                           <L 1773>
                        // size += 3                                                              <L 1774>
                        var_230 = wp::add(var_225, var_229);
                    }
                    var_231 = wp::where(var_227, var_228, var_208);
                    var_232 = wp::where(var_227, var_230, var_225);
                    if (!var_227) {
                        // elif i == 2:                                                           <L 1775>
                        var_234 = (var_215 == var_233);
                        if (var_234) {
                            // torque = True                                                      <L 1776>
                            // size += 3                                                          <L 1777>
                            var_237 = wp::add(var_232, var_236);
                        }
                        var_238 = wp::where(var_234, var_235, var_209);
                        var_239 = wp::where(var_234, var_237, var_232);
                        if (!var_234) {
                            // elif i == 3:                                                       <L 1778>
                            var_241 = (var_215 == var_240);
                            if (var_241) {
                                // dist = True                                                    <L 1779>
                                // size += 1                                                      <L 1780>
                                var_244 = wp::add(var_239, var_243);
                            }
                            var_245 = wp::where(var_241, var_242, var_210);
                            var_246 = wp::where(var_241, var_244, var_239);
                            if (!var_241) {
                                // elif i == 4:                                                   <L 1781>
                                var_248 = (var_215 == var_247);
                                if (var_248) {
                                    // pos = True                                                 <L 1782>
                                    // size += 3                                                  <L 1783>
                                    var_251 = wp::add(var_246, var_250);
                                }
                                var_252 = wp::where(var_248, var_249, var_211);
                                var_253 = wp::where(var_248, var_251, var_246);
                                if (!var_248) {
                                    // elif i == 5:                                               <L 1784>
                                    var_255 = (var_215 == var_254);
                                    if (var_255) {
                                        // normal = True                                          <L 1785>
                                        // size += 3                                              <L 1786>
                                        var_258 = wp::add(var_253, var_257);
                                    }
                                    var_259 = wp::where(var_255, var_256, var_212);
                                    var_260 = wp::where(var_255, var_258, var_253);
                                    if (!var_255) {
                                        // elif i == 6:                                           <L 1787>
                                        var_262 = (var_215 == var_261);
                                        if (var_262) {
                                            // tangent = True                                     <L 1788>
                                            // size += 3                                          <L 1789>
                                            var_265 = wp::add(var_260, var_264);
                                        }
                                        var_266 = wp::where(var_262, var_263, var_213);
                                        var_267 = wp::where(var_262, var_265, var_260);
                                    }
                                    var_268 = wp::where(var_255, var_213, var_266);
                                    var_269 = wp::where(var_255, var_260, var_267);
                                }
                                var_270 = wp::where(var_248, var_212, var_259);
                                var_271 = wp::where(var_248, var_213, var_268);
                                var_272 = wp::where(var_248, var_253, var_269);
                            }
                            var_273 = wp::where(var_241, var_211, var_252);
                            var_274 = wp::where(var_241, var_212, var_270);
                            var_275 = wp::where(var_241, var_213, var_271);
                            var_276 = wp::where(var_241, var_246, var_272);
                        }
                        var_277 = wp::where(var_234, var_210, var_245);
                        var_278 = wp::where(var_234, var_211, var_273);
                        var_279 = wp::where(var_234, var_212, var_274);
                        var_280 = wp::where(var_234, var_213, var_275);
                        var_281 = wp::where(var_234, var_239, var_276);
                    }
                    var_282 = wp::where(var_227, var_209, var_238);
                    var_283 = wp::where(var_227, var_210, var_277);
                    var_284 = wp::where(var_227, var_211, var_278);
                    var_285 = wp::where(var_227, var_212, var_279);
                    var_286 = wp::where(var_227, var_213, var_280);
                    var_287 = wp::where(var_227, var_232, var_281);
                }
                var_288 = wp::where(var_220, var_208, var_231);
                var_289 = wp::where(var_220, var_209, var_282);
                var_290 = wp::where(var_220, var_210, var_283);
                var_291 = wp::where(var_220, var_211, var_284);
                var_292 = wp::where(var_220, var_212, var_285);
                var_293 = wp::where(var_220, var_213, var_286);
                var_294 = wp::where(var_220, var_225, var_287);
            }
            var_295 = wp::where(var_218, var_224, var_207);
            var_296 = wp::where(var_218, var_288, var_208);
            var_297 = wp::where(var_218, var_289, var_209);
            var_298 = wp::where(var_218, var_290, var_210);
            var_299 = wp::where(var_218, var_291, var_211);
            var_300 = wp::where(var_218, var_292, var_212);
            var_301 = wp::where(var_218, var_293, var_213);
            var_302 = wp::where(var_218, var_294, var_214);
            // if dataspec & (1 << i):                                                            <L 1768>
            var_305 = wp::lshift(var_304, var_303);
            var_306 = wp::bit_and(var_18, var_305);
            if (var_306) {
                // if i == 0:                                                                     <L 1769>
                var_308 = (var_303 == var_307);
                if (var_308) {
                    // found = True                                                               <L 1770>
                    // size += 1                                                                  <L 1771>
                    var_311 = wp::add(var_302, var_310);
                }
                var_312 = wp::where(var_308, var_309, var_295);
                var_313 = wp::where(var_308, var_311, var_302);
                if (!var_308) {
                    // elif i == 1:                                                               <L 1772>
                    var_315 = (var_303 == var_314);
                    if (var_315) {
                        // force = True                                                           <L 1773>
                        // size += 3                                                              <L 1774>
                        var_318 = wp::add(var_313, var_317);
                    }
                    var_319 = wp::where(var_315, var_316, var_296);
                    var_320 = wp::where(var_315, var_318, var_313);
                    if (!var_315) {
                        // elif i == 2:                                                           <L 1775>
                        var_322 = (var_303 == var_321);
                        if (var_322) {
                            // torque = True                                                      <L 1776>
                            // size += 3                                                          <L 1777>
                            var_325 = wp::add(var_320, var_324);
                        }
                        var_326 = wp::where(var_322, var_323, var_297);
                        var_327 = wp::where(var_322, var_325, var_320);
                        if (!var_322) {
                            // elif i == 3:                                                       <L 1778>
                            var_329 = (var_303 == var_328);
                            if (var_329) {
                                // dist = True                                                    <L 1779>
                                // size += 1                                                      <L 1780>
                                var_332 = wp::add(var_327, var_331);
                            }
                            var_333 = wp::where(var_329, var_330, var_298);
                            var_334 = wp::where(var_329, var_332, var_327);
                            if (!var_329) {
                                // elif i == 4:                                                   <L 1781>
                                var_336 = (var_303 == var_335);
                                if (var_336) {
                                    // pos = True                                                 <L 1782>
                                    // size += 3                                                  <L 1783>
                                    var_339 = wp::add(var_334, var_338);
                                }
                                var_340 = wp::where(var_336, var_337, var_299);
                                var_341 = wp::where(var_336, var_339, var_334);
                                if (!var_336) {
                                    // elif i == 5:                                               <L 1784>
                                    var_343 = (var_303 == var_342);
                                    if (var_343) {
                                        // normal = True                                          <L 1785>
                                        // size += 3                                              <L 1786>
                                        var_346 = wp::add(var_341, var_345);
                                    }
                                    var_347 = wp::where(var_343, var_344, var_300);
                                    var_348 = wp::where(var_343, var_346, var_341);
                                    if (!var_343) {
                                        // elif i == 6:                                           <L 1787>
                                        var_350 = (var_303 == var_349);
                                        if (var_350) {
                                            // tangent = True                                     <L 1788>
                                            // size += 3                                          <L 1789>
                                            var_353 = wp::add(var_348, var_352);
                                        }
                                        var_354 = wp::where(var_350, var_351, var_301);
                                        var_355 = wp::where(var_350, var_353, var_348);
                                    }
                                    var_356 = wp::where(var_343, var_301, var_354);
                                    var_357 = wp::where(var_343, var_348, var_355);
                                }
                                var_358 = wp::where(var_336, var_300, var_347);
                                var_359 = wp::where(var_336, var_301, var_356);
                                var_360 = wp::where(var_336, var_341, var_357);
                            }
                            var_361 = wp::where(var_329, var_299, var_340);
                            var_362 = wp::where(var_329, var_300, var_358);
                            var_363 = wp::where(var_329, var_301, var_359);
                            var_364 = wp::where(var_329, var_334, var_360);
                        }
                        var_365 = wp::where(var_322, var_298, var_333);
                        var_366 = wp::where(var_322, var_299, var_361);
                        var_367 = wp::where(var_322, var_300, var_362);
                        var_368 = wp::where(var_322, var_301, var_363);
                        var_369 = wp::where(var_322, var_327, var_364);
                    }
                    var_370 = wp::where(var_315, var_297, var_326);
                    var_371 = wp::where(var_315, var_298, var_365);
                    var_372 = wp::where(var_315, var_299, var_366);
                    var_373 = wp::where(var_315, var_300, var_367);
                    var_374 = wp::where(var_315, var_301, var_368);
                    var_375 = wp::where(var_315, var_320, var_369);
                }
                var_376 = wp::where(var_308, var_296, var_319);
                var_377 = wp::where(var_308, var_297, var_370);
                var_378 = wp::where(var_308, var_298, var_371);
                var_379 = wp::where(var_308, var_299, var_372);
                var_380 = wp::where(var_308, var_300, var_373);
                var_381 = wp::where(var_308, var_301, var_374);
                var_382 = wp::where(var_308, var_313, var_375);
            }
            var_383 = wp::where(var_306, var_312, var_295);
            var_384 = wp::where(var_306, var_376, var_296);
            var_385 = wp::where(var_306, var_377, var_297);
            var_386 = wp::where(var_306, var_378, var_298);
            var_387 = wp::where(var_306, var_379, var_299);
            var_388 = wp::where(var_306, var_380, var_300);
            var_389 = wp::where(var_306, var_381, var_301);
            var_390 = wp::where(var_306, var_382, var_302);
            // if dataspec & (1 << i):                                                            <L 1768>
            var_393 = wp::lshift(var_392, var_391);
            var_394 = wp::bit_and(var_18, var_393);
            if (var_394) {
                // if i == 0:                                                                     <L 1769>
                var_396 = (var_391 == var_395);
                if (var_396) {
                    // found = True                                                               <L 1770>
                    // size += 1                                                                  <L 1771>
                    var_399 = wp::add(var_390, var_398);
                }
                var_400 = wp::where(var_396, var_397, var_383);
                var_401 = wp::where(var_396, var_399, var_390);
                if (!var_396) {
                    // elif i == 1:                                                               <L 1772>
                    var_403 = (var_391 == var_402);
                    if (var_403) {
                        // force = True                                                           <L 1773>
                        // size += 3                                                              <L 1774>
                        var_406 = wp::add(var_401, var_405);
                    }
                    var_407 = wp::where(var_403, var_404, var_384);
                    var_408 = wp::where(var_403, var_406, var_401);
                    if (!var_403) {
                        // elif i == 2:                                                           <L 1775>
                        var_410 = (var_391 == var_409);
                        if (var_410) {
                            // torque = True                                                      <L 1776>
                            // size += 3                                                          <L 1777>
                            var_413 = wp::add(var_408, var_412);
                        }
                        var_414 = wp::where(var_410, var_411, var_385);
                        var_415 = wp::where(var_410, var_413, var_408);
                        if (!var_410) {
                            // elif i == 3:                                                       <L 1778>
                            var_417 = (var_391 == var_416);
                            if (var_417) {
                                // dist = True                                                    <L 1779>
                                // size += 1                                                      <L 1780>
                                var_420 = wp::add(var_415, var_419);
                            }
                            var_421 = wp::where(var_417, var_418, var_386);
                            var_422 = wp::where(var_417, var_420, var_415);
                            if (!var_417) {
                                // elif i == 4:                                                   <L 1781>
                                var_424 = (var_391 == var_423);
                                if (var_424) {
                                    // pos = True                                                 <L 1782>
                                    // size += 3                                                  <L 1783>
                                    var_427 = wp::add(var_422, var_426);
                                }
                                var_428 = wp::where(var_424, var_425, var_387);
                                var_429 = wp::where(var_424, var_427, var_422);
                                if (!var_424) {
                                    // elif i == 5:                                               <L 1784>
                                    var_431 = (var_391 == var_430);
                                    if (var_431) {
                                        // normal = True                                          <L 1785>
                                        // size += 3                                              <L 1786>
                                        var_434 = wp::add(var_429, var_433);
                                    }
                                    var_435 = wp::where(var_431, var_432, var_388);
                                    var_436 = wp::where(var_431, var_434, var_429);
                                    if (!var_431) {
                                        // elif i == 6:                                           <L 1787>
                                        var_438 = (var_391 == var_437);
                                        if (var_438) {
                                            // tangent = True                                     <L 1788>
                                            // size += 3                                          <L 1789>
                                            var_441 = wp::add(var_436, var_440);
                                        }
                                        var_442 = wp::where(var_438, var_439, var_389);
                                        var_443 = wp::where(var_438, var_441, var_436);
                                    }
                                    var_444 = wp::where(var_431, var_389, var_442);
                                    var_445 = wp::where(var_431, var_436, var_443);
                                }
                                var_446 = wp::where(var_424, var_388, var_435);
                                var_447 = wp::where(var_424, var_389, var_444);
                                var_448 = wp::where(var_424, var_429, var_445);
                            }
                            var_449 = wp::where(var_417, var_387, var_428);
                            var_450 = wp::where(var_417, var_388, var_446);
                            var_451 = wp::where(var_417, var_389, var_447);
                            var_452 = wp::where(var_417, var_422, var_448);
                        }
                        var_453 = wp::where(var_410, var_386, var_421);
                        var_454 = wp::where(var_410, var_387, var_449);
                        var_455 = wp::where(var_410, var_388, var_450);
                        var_456 = wp::where(var_410, var_389, var_451);
                        var_457 = wp::where(var_410, var_415, var_452);
                    }
                    var_458 = wp::where(var_403, var_385, var_414);
                    var_459 = wp::where(var_403, var_386, var_453);
                    var_460 = wp::where(var_403, var_387, var_454);
                    var_461 = wp::where(var_403, var_388, var_455);
                    var_462 = wp::where(var_403, var_389, var_456);
                    var_463 = wp::where(var_403, var_408, var_457);
                }
                var_464 = wp::where(var_396, var_384, var_407);
                var_465 = wp::where(var_396, var_385, var_458);
                var_466 = wp::where(var_396, var_386, var_459);
                var_467 = wp::where(var_396, var_387, var_460);
                var_468 = wp::where(var_396, var_388, var_461);
                var_469 = wp::where(var_396, var_389, var_462);
                var_470 = wp::where(var_396, var_401, var_463);
            }
            var_471 = wp::where(var_394, var_400, var_383);
            var_472 = wp::where(var_394, var_464, var_384);
            var_473 = wp::where(var_394, var_465, var_385);
            var_474 = wp::where(var_394, var_466, var_386);
            var_475 = wp::where(var_394, var_467, var_387);
            var_476 = wp::where(var_394, var_468, var_388);
            var_477 = wp::where(var_394, var_469, var_389);
            var_478 = wp::where(var_394, var_470, var_390);
            // if dataspec & (1 << i):                                                            <L 1768>
            var_481 = wp::lshift(var_480, var_479);
            var_482 = wp::bit_and(var_18, var_481);
            if (var_482) {
                // if i == 0:                                                                     <L 1769>
                var_484 = (var_479 == var_483);
                if (var_484) {
                    // found = True                                                               <L 1770>
                    // size += 1                                                                  <L 1771>
                    var_487 = wp::add(var_478, var_486);
                }
                var_488 = wp::where(var_484, var_485, var_471);
                var_489 = wp::where(var_484, var_487, var_478);
                if (!var_484) {
                    // elif i == 1:                                                               <L 1772>
                    var_491 = (var_479 == var_490);
                    if (var_491) {
                        // force = True                                                           <L 1773>
                        // size += 3                                                              <L 1774>
                        var_494 = wp::add(var_489, var_493);
                    }
                    var_495 = wp::where(var_491, var_492, var_472);
                    var_496 = wp::where(var_491, var_494, var_489);
                    if (!var_491) {
                        // elif i == 2:                                                           <L 1775>
                        var_498 = (var_479 == var_497);
                        if (var_498) {
                            // torque = True                                                      <L 1776>
                            // size += 3                                                          <L 1777>
                            var_501 = wp::add(var_496, var_500);
                        }
                        var_502 = wp::where(var_498, var_499, var_473);
                        var_503 = wp::where(var_498, var_501, var_496);
                        if (!var_498) {
                            // elif i == 3:                                                       <L 1778>
                            var_505 = (var_479 == var_504);
                            if (var_505) {
                                // dist = True                                                    <L 1779>
                                // size += 1                                                      <L 1780>
                                var_508 = wp::add(var_503, var_507);
                            }
                            var_509 = wp::where(var_505, var_506, var_474);
                            var_510 = wp::where(var_505, var_508, var_503);
                            if (!var_505) {
                                // elif i == 4:                                                   <L 1781>
                                var_512 = (var_479 == var_511);
                                if (var_512) {
                                    // pos = True                                                 <L 1782>
                                    // size += 3                                                  <L 1783>
                                    var_515 = wp::add(var_510, var_514);
                                }
                                var_516 = wp::where(var_512, var_513, var_475);
                                var_517 = wp::where(var_512, var_515, var_510);
                                if (!var_512) {
                                    // elif i == 5:                                               <L 1784>
                                    var_519 = (var_479 == var_518);
                                    if (var_519) {
                                        // normal = True                                          <L 1785>
                                        // size += 3                                              <L 1786>
                                        var_522 = wp::add(var_517, var_521);
                                    }
                                    var_523 = wp::where(var_519, var_520, var_476);
                                    var_524 = wp::where(var_519, var_522, var_517);
                                    if (!var_519) {
                                        // elif i == 6:                                           <L 1787>
                                        var_526 = (var_479 == var_525);
                                        if (var_526) {
                                            // tangent = True                                     <L 1788>
                                            // size += 3                                          <L 1789>
                                            var_529 = wp::add(var_524, var_528);
                                        }
                                        var_530 = wp::where(var_526, var_527, var_477);
                                        var_531 = wp::where(var_526, var_529, var_524);
                                    }
                                    var_532 = wp::where(var_519, var_477, var_530);
                                    var_533 = wp::where(var_519, var_524, var_531);
                                }
                                var_534 = wp::where(var_512, var_476, var_523);
                                var_535 = wp::where(var_512, var_477, var_532);
                                var_536 = wp::where(var_512, var_517, var_533);
                            }
                            var_537 = wp::where(var_505, var_475, var_516);
                            var_538 = wp::where(var_505, var_476, var_534);
                            var_539 = wp::where(var_505, var_477, var_535);
                            var_540 = wp::where(var_505, var_510, var_536);
                        }
                        var_541 = wp::where(var_498, var_474, var_509);
                        var_542 = wp::where(var_498, var_475, var_537);
                        var_543 = wp::where(var_498, var_476, var_538);
                        var_544 = wp::where(var_498, var_477, var_539);
                        var_545 = wp::where(var_498, var_503, var_540);
                    }
                    var_546 = wp::where(var_491, var_473, var_502);
                    var_547 = wp::where(var_491, var_474, var_541);
                    var_548 = wp::where(var_491, var_475, var_542);
                    var_549 = wp::where(var_491, var_476, var_543);
                    var_550 = wp::where(var_491, var_477, var_544);
                    var_551 = wp::where(var_491, var_496, var_545);
                }
                var_552 = wp::where(var_484, var_472, var_495);
                var_553 = wp::where(var_484, var_473, var_546);
                var_554 = wp::where(var_484, var_474, var_547);
                var_555 = wp::where(var_484, var_475, var_548);
                var_556 = wp::where(var_484, var_476, var_549);
                var_557 = wp::where(var_484, var_477, var_550);
                var_558 = wp::where(var_484, var_489, var_551);
            }
            var_559 = wp::where(var_482, var_488, var_471);
            var_560 = wp::where(var_482, var_552, var_472);
            var_561 = wp::where(var_482, var_553, var_473);
            var_562 = wp::where(var_482, var_554, var_474);
            var_563 = wp::where(var_482, var_555, var_475);
            var_564 = wp::where(var_482, var_556, var_476);
            var_565 = wp::where(var_482, var_557, var_477);
            var_566 = wp::where(var_482, var_558, var_478);
            // if dataspec & (1 << i):                                                            <L 1768>
            var_569 = wp::lshift(var_568, var_567);
            var_570 = wp::bit_and(var_18, var_569);
            if (var_570) {
                // if i == 0:                                                                     <L 1769>
                var_572 = (var_567 == var_571);
                if (var_572) {
                    // found = True                                                               <L 1770>
                    // size += 1                                                                  <L 1771>
                    var_575 = wp::add(var_566, var_574);
                }
                var_576 = wp::where(var_572, var_573, var_559);
                var_577 = wp::where(var_572, var_575, var_566);
                if (!var_572) {
                    // elif i == 1:                                                               <L 1772>
                    var_579 = (var_567 == var_578);
                    if (var_579) {
                        // force = True                                                           <L 1773>
                        // size += 3                                                              <L 1774>
                        var_582 = wp::add(var_577, var_581);
                    }
                    var_583 = wp::where(var_579, var_580, var_560);
                    var_584 = wp::where(var_579, var_582, var_577);
                    if (!var_579) {
                        // elif i == 2:                                                           <L 1775>
                        var_586 = (var_567 == var_585);
                        if (var_586) {
                            // torque = True                                                      <L 1776>
                            // size += 3                                                          <L 1777>
                            var_589 = wp::add(var_584, var_588);
                        }
                        var_590 = wp::where(var_586, var_587, var_561);
                        var_591 = wp::where(var_586, var_589, var_584);
                        if (!var_586) {
                            // elif i == 3:                                                       <L 1778>
                            var_593 = (var_567 == var_592);
                            if (var_593) {
                                // dist = True                                                    <L 1779>
                                // size += 1                                                      <L 1780>
                                var_596 = wp::add(var_591, var_595);
                            }
                            var_597 = wp::where(var_593, var_594, var_562);
                            var_598 = wp::where(var_593, var_596, var_591);
                            if (!var_593) {
                                // elif i == 4:                                                   <L 1781>
                                var_600 = (var_567 == var_599);
                                if (var_600) {
                                    // pos = True                                                 <L 1782>
                                    // size += 3                                                  <L 1783>
                                    var_603 = wp::add(var_598, var_602);
                                }
                                var_604 = wp::where(var_600, var_601, var_563);
                                var_605 = wp::where(var_600, var_603, var_598);
                                if (!var_600) {
                                    // elif i == 5:                                               <L 1784>
                                    var_607 = (var_567 == var_606);
                                    if (var_607) {
                                        // normal = True                                          <L 1785>
                                        // size += 3                                              <L 1786>
                                        var_610 = wp::add(var_605, var_609);
                                    }
                                    var_611 = wp::where(var_607, var_608, var_564);
                                    var_612 = wp::where(var_607, var_610, var_605);
                                    if (!var_607) {
                                        // elif i == 6:                                           <L 1787>
                                        var_614 = (var_567 == var_613);
                                        if (var_614) {
                                            // tangent = True                                     <L 1788>
                                            // size += 3                                          <L 1789>
                                            var_617 = wp::add(var_612, var_616);
                                        }
                                        var_618 = wp::where(var_614, var_615, var_565);
                                        var_619 = wp::where(var_614, var_617, var_612);
                                    }
                                    var_620 = wp::where(var_607, var_565, var_618);
                                    var_621 = wp::where(var_607, var_612, var_619);
                                }
                                var_622 = wp::where(var_600, var_564, var_611);
                                var_623 = wp::where(var_600, var_565, var_620);
                                var_624 = wp::where(var_600, var_605, var_621);
                            }
                            var_625 = wp::where(var_593, var_563, var_604);
                            var_626 = wp::where(var_593, var_564, var_622);
                            var_627 = wp::where(var_593, var_565, var_623);
                            var_628 = wp::where(var_593, var_598, var_624);
                        }
                        var_629 = wp::where(var_586, var_562, var_597);
                        var_630 = wp::where(var_586, var_563, var_625);
                        var_631 = wp::where(var_586, var_564, var_626);
                        var_632 = wp::where(var_586, var_565, var_627);
                        var_633 = wp::where(var_586, var_591, var_628);
                    }
                    var_634 = wp::where(var_579, var_561, var_590);
                    var_635 = wp::where(var_579, var_562, var_629);
                    var_636 = wp::where(var_579, var_563, var_630);
                    var_637 = wp::where(var_579, var_564, var_631);
                    var_638 = wp::where(var_579, var_565, var_632);
                    var_639 = wp::where(var_579, var_584, var_633);
                }
                var_640 = wp::where(var_572, var_560, var_583);
                var_641 = wp::where(var_572, var_561, var_634);
                var_642 = wp::where(var_572, var_562, var_635);
                var_643 = wp::where(var_572, var_563, var_636);
                var_644 = wp::where(var_572, var_564, var_637);
                var_645 = wp::where(var_572, var_565, var_638);
                var_646 = wp::where(var_572, var_577, var_639);
            }
            var_647 = wp::where(var_570, var_576, var_559);
            var_648 = wp::where(var_570, var_640, var_560);
            var_649 = wp::where(var_570, var_641, var_561);
            var_650 = wp::where(var_570, var_642, var_562);
            var_651 = wp::where(var_570, var_643, var_563);
            var_652 = wp::where(var_570, var_644, var_564);
            var_653 = wp::where(var_570, var_645, var_565);
            var_654 = wp::where(var_570, var_646, var_566);
            // num = dim // size  # number of slots                                               <L 1791>
            var_655 = wp::floordiv(var_21, var_654);
            // adr = sensor_adr[sensorid]                                                         <L 1793>
            var_656 = wp::address(var_sensor_adr, var_3);
            var_658 = wp::load(var_656);
            var_657 = wp::copy(var_658);
            // contactsensorid = sensor_adr_to_contact_adr[sensorid]                              <L 1794>
            var_659 = wp::address(var_sensor_adr_to_contact_adr, var_3);
            var_661 = wp::load(var_659);
            var_660 = wp::copy(var_661);
            // nmatch = sensor_contact_nmatch_in[worldid, contactsensorid]                        <L 1795>
            var_662 = wp::address(var_sensor_contact_nmatch_in, var_0, var_660);
            var_664 = wp::load(var_662);
            var_663 = wp::copy(var_664);
            // if reduce == 3:  # netforce                                                        <L 1797>
            var_666 = (var_28 == var_665);
            if (var_666) {
                // net_pos = wp.vec3(0.0)                                                         <L 1800>
                var_668 = wp::vec_t<3, wp::float32>(var_667);
                // net_force = wp.vec3(0.0)                                                       <L 1801>
                var_670 = wp::vec_t<3, wp::float32>(var_669);
                // net_torque = wp.vec3(0.0)                                                      <L 1802>
                var_672 = wp::vec_t<3, wp::float32>(var_671);
                // total_force_magnitude = float(0.0)                                             <L 1803>
                var_674 = wp::float(var_673);
                // for i in range(nmatch):                                                        <L 1805>
                var_675 = wp::range(var_663);
                start_for_0:;
                    if (iter_cmp(var_675) == 0) goto end_for_0;
                    var_676 = wp::iter_next(var_675);
                    // cid = sensor_contact_matchid_in[worldid, contactsensorid, i]               <L 1806>
                    var_677 = wp::address(var_sensor_contact_matchid_in, var_0, var_660, var_676);
                    var_679 = wp::load(var_677);
                    var_678 = wp::copy(var_679);
                    // dir = sensor_contact_direction_in[worldid, contactsensorid, i]             <L 1807>
                    var_680 = wp::address(var_sensor_contact_direction_in, var_0, var_660, var_676);
                    var_682 = wp::load(var_680);
                    var_681 = wp::copy(var_682);
                    // contact_forcetorque = support.contact_force_fn(                            <L 1809>
                    // opt_cone,                                                                  <L 1810>
                    // contact_frame_in,                                                          <L 1811>
                    // contact_friction_in,                                                       <L 1812>
                    // contact_dim_in,                                                            <L 1813>
                    // contact_efc_address_in,                                                    <L 1814>
                    // efc_force_in,                                                              <L 1815>
                    // njmax_in,                                                                  <L 1816>
                    // nacon_in,                                                                  <L 1817>
                    // worldid,                                                                   <L 1818>
                    // cid,                                                                       <L 1819>
                    // False,                                                                     <L 1820>
                    var_684 = contact_force_fn_0(var_opt_cone, var_contact_frame_in, var_contact_friction_in, var_contact_dim_in, var_contact_efc_address_in, var_efc_force_in, var_njmax_in, var_nacon_in, var_0, var_678, var_683);
                    // weight = wp.norm_l2(wp.spatial_top(contact_forcetorque))                   <L 1824>
                    var_685 = wp::spatial_top(var_684);
                    var_686 = norm_l2_0(var_685);
                    // contact_pos = contact_pos_in[cid]                                          <L 1825>
                    var_687 = wp::address(var_contact_pos_in, var_678);
                    var_689 = wp::load(var_687);
                    var_688 = wp::copy(var_689);
                    // net_pos += weight * contact_pos                                            <L 1826>
                    var_690 = wp::mul(var_686, var_688);
                    var_691 = wp::add(var_668, var_690);
                    // total_force_magnitude += weight                                            <L 1827>
                    var_692 = wp::add(var_674, var_686);
                    // contact_forcetorque *= dir                                                 <L 1830>
                    var_693 = wp::mul(var_684, var_681);
                    // force_local = wp.spatial_top(contact_forcetorque)                          <L 1831>
                    var_694 = wp::spatial_top(var_693);
                    // torque_local = wp.spatial_bottom(contact_forcetorque)                      <L 1832>
                    var_695 = wp::spatial_bottom(var_693);
                    // frame = contact_frame_in[cid]                                              <L 1834>
                    var_696 = wp::address(var_contact_frame_in, var_678);
                    var_698 = wp::load(var_696);
                    var_697 = wp::copy(var_698);
                    // frameT = wp.transpose(frame)                                               <L 1835>
                    var_699 = wp::transpose(var_697);
                    // force_global = frameT @ force_local                                        <L 1837>
                    var_700 = wp::mul(var_699, var_694);
                    // torque_global = frameT @ torque_local                                      <L 1838>
                    var_701 = wp::mul(var_699, var_695);
                    // net_force += force_global                                                  <L 1841>
                    var_702 = wp::add(var_670, var_700);
                    // net_torque += torque_global                                                <L 1842>
                    var_703 = wp::add(var_672, var_701);
                    // net_torque += wp.cross(contact_pos, force_global)                          <L 1844>
                    var_704 = wp::cross(var_688, var_700);
                    var_705 = wp::add(var_703, var_704);
                    wp::assign(var_668, var_691);
                    wp::assign(var_670, var_702);
                    wp::assign(var_672, var_705);
                    wp::assign(var_674, var_692);
                    goto start_for_0;
                end_for_0:;
                // net_pos /= wp.max(total_force_magnitude, MJ_MINVAL)                            <L 1847>
                var_707 = wp::max(var_674, var_706);
                var_708 = wp::div(var_668, var_707);
                // net_torque -= wp.cross(net_pos, net_force)                                     <L 1851>
                var_709 = wp::cross(var_708, var_670);
                var_710 = wp::sub(var_672, var_709);
                // adr_slot = adr                                                                 <L 1853>
                var_711 = wp::copy(var_657);
                // if found:                                                                      <L 1855>
                if (var_647) {
                    // out[adr_slot] = float(nmatch)                                              <L 1856>
                    var_712 = wp::float(var_663);
                    wp::array_store(var_13, var_711, var_712);
                    // adr_slot += 1                                                              <L 1857>
                    var_714 = wp::add(var_711, var_713);
                }
                var_715 = wp::where(var_647, var_714, var_711);
                // if force:                                                                      <L 1858>
                if (var_648) {
                    // out[adr_slot + 0] = net_force[0]                                           <L 1859>
                    var_717 = wp::extract(var_670, var_716);
                    var_719 = wp::add(var_715, var_718);
                    wp::array_store(var_13, var_719, var_717);
                    // out[adr_slot + 1] = net_force[1]                                           <L 1860>
                    var_721 = wp::extract(var_670, var_720);
                    var_723 = wp::add(var_715, var_722);
                    wp::array_store(var_13, var_723, var_721);
                    // out[adr_slot + 2] = net_force[2]                                           <L 1861>
                    var_725 = wp::extract(var_670, var_724);
                    var_727 = wp::add(var_715, var_726);
                    wp::array_store(var_13, var_727, var_725);
                    // adr_slot += 3                                                              <L 1862>
                    var_729 = wp::add(var_715, var_728);
                }
                var_730 = wp::where(var_648, var_729, var_715);
                // if torque:                                                                     <L 1863>
                if (var_649) {
                    // out[adr_slot + 0] = net_torque[0]                                          <L 1864>
                    var_732 = wp::extract(var_710, var_731);
                    var_734 = wp::add(var_730, var_733);
                    wp::array_store(var_13, var_734, var_732);
                    // out[adr_slot + 1] = net_torque[1]                                          <L 1865>
                    var_736 = wp::extract(var_710, var_735);
                    var_738 = wp::add(var_730, var_737);
                    wp::array_store(var_13, var_738, var_736);
                    // out[adr_slot + 2] = net_torque[2]                                          <L 1866>
                    var_740 = wp::extract(var_710, var_739);
                    var_742 = wp::add(var_730, var_741);
                    wp::array_store(var_13, var_742, var_740);
                    // adr_slot += 3                                                              <L 1867>
                    var_744 = wp::add(var_730, var_743);
                }
                var_745 = wp::where(var_649, var_744, var_730);
                // if dist:                                                                       <L 1868>
                if (var_650) {
                    // out[adr_slot] = 0.0                                                        <L 1869>
                    wp::array_store(var_13, var_745, var_746);
                    // adr_slot += 1                                                              <L 1870>
                    var_748 = wp::add(var_745, var_747);
                }
                var_749 = wp::where(var_650, var_748, var_745);
                // if pos:                                                                        <L 1871>
                if (var_651) {
                    // out[adr_slot + 0] = net_pos[0]                                             <L 1872>
                    var_751 = wp::extract(var_708, var_750);
                    var_753 = wp::add(var_749, var_752);
                    wp::array_store(var_13, var_753, var_751);
                    // out[adr_slot + 1] = net_pos[1]                                             <L 1873>
                    var_755 = wp::extract(var_708, var_754);
                    var_757 = wp::add(var_749, var_756);
                    wp::array_store(var_13, var_757, var_755);
                    // out[adr_slot + 2] = net_pos[2]                                             <L 1874>
                    var_759 = wp::extract(var_708, var_758);
                    var_761 = wp::add(var_749, var_760);
                    wp::array_store(var_13, var_761, var_759);
                    // adr_slot += 3                                                              <L 1875>
                    var_763 = wp::add(var_749, var_762);
                }
                var_764 = wp::where(var_651, var_763, var_749);
                // if normal:                                                                     <L 1876>
                if (var_652) {
                    // out[adr_slot + 0] = 1.0                                                    <L 1877>
                    var_767 = wp::add(var_764, var_766);
                    wp::array_store(var_13, var_767, var_765);
                    // out[adr_slot + 1] = 0.0                                                    <L 1878>
                    var_770 = wp::add(var_764, var_769);
                    wp::array_store(var_13, var_770, var_768);
                    // out[adr_slot + 2] = 0.0                                                    <L 1879>
                    var_773 = wp::add(var_764, var_772);
                    wp::array_store(var_13, var_773, var_771);
                    // adr_slot += 3                                                              <L 1880>
                    var_775 = wp::add(var_764, var_774);
                }
                var_776 = wp::where(var_652, var_775, var_764);
                // if tangent:                                                                    <L 1881>
                if (var_653) {
                    // out[adr_slot + 0] = 0.0                                                    <L 1882>
                    var_779 = wp::add(var_776, var_778);
                    wp::array_store(var_13, var_779, var_777);
                    // out[adr_slot + 1] = 1.0                                                    <L 1883>
                    var_782 = wp::add(var_776, var_781);
                    wp::array_store(var_13, var_782, var_780);
                    // out[adr_slot + 2] = 0.0                                                    <L 1884>
                    var_785 = wp::add(var_776, var_784);
                    wp::array_store(var_13, var_785, var_783);
                }
            }
            var_786 = wp::where(var_666, var_676, var_567);
            if (!var_666) {
                // nslots = wp.min(nmatch, num)                                                   <L 1886>
                var_787 = wp::min(var_663, var_655);
                // for i in range(nslots):                                                        <L 1887>
                var_788 = wp::range(var_787);
                start_for_2:;
                    if (iter_cmp(var_788) == 0) goto end_for_2;
                    var_789 = wp::iter_next(var_788);
                    // cid = sensor_contact_matchid_in[worldid, contactsensorid, i]               <L 1889>
                    var_790 = wp::address(var_sensor_contact_matchid_in, var_0, var_660, var_789);
                    var_792 = wp::load(var_790);
                    var_791 = wp::copy(var_792);
                    // dir = sensor_contact_direction_in[worldid, contactsensorid, i]             <L 1892>
                    var_793 = wp::address(var_sensor_contact_direction_in, var_0, var_660, var_789);
                    var_795 = wp::load(var_793);
                    var_794 = wp::copy(var_795);
                    // adr_slot = adr + i * size                                                  <L 1894>
                    var_796 = wp::mul(var_789, var_654);
                    var_797 = wp::add(var_657, var_796);
                    // if found:                                                                  <L 1896>
                    if (var_647) {
                        // out[adr_slot] = float(nmatch)                                          <L 1897>
                        var_798 = wp::float(var_663);
                        wp::array_store(var_13, var_797, var_798);
                        // adr_slot += 1                                                          <L 1898>
                        var_800 = wp::add(var_797, var_799);
                    }
                    var_801 = wp::where(var_647, var_800, var_797);
                    // if force or torque:                                                        <L 1899>
                    var_802 = var_648 || var_649;
                    if (var_802) {
                        // contact_forcetorque = support.contact_force_fn(                        <L 1900>
                        // opt_cone,                                                              <L 1901>
                        // contact_frame_in,                                                      <L 1902>
                        // contact_friction_in,                                                   <L 1903>
                        // contact_dim_in,                                                        <L 1904>
                        // contact_efc_address_in,                                                <L 1905>
                        // efc_force_in,                                                          <L 1906>
                        // njmax_in,                                                              <L 1907>
                        // nacon_in,                                                              <L 1908>
                        // worldid,                                                               <L 1909>
                        // cid,                                                                   <L 1910>
                        // False,                                                                 <L 1911>
                        var_804 = contact_force_fn_0(var_opt_cone, var_contact_frame_in, var_contact_friction_in, var_contact_dim_in, var_contact_efc_address_in, var_efc_force_in, var_njmax_in, var_nacon_in, var_0, var_791, var_803);
                    }
                    var_805 = wp::where(var_802, var_804, var_693);
                    // if force:                                                                  <L 1913>
                    if (var_648) {
                        // out[adr_slot + 0] = contact_forcetorque[0]                             <L 1914>
                        var_807 = wp::extract(var_805, var_806);
                        var_809 = wp::add(var_801, var_808);
                        wp::array_store(var_13, var_809, var_807);
                        // out[adr_slot + 1] = contact_forcetorque[1]                             <L 1915>
                        var_811 = wp::extract(var_805, var_810);
                        var_813 = wp::add(var_801, var_812);
                        wp::array_store(var_13, var_813, var_811);
                        // out[adr_slot + 2] = dir * contact_forcetorque[2]                       <L 1916>
                        var_815 = wp::extract(var_805, var_814);
                        var_816 = wp::mul(var_794, var_815);
                        var_818 = wp::add(var_801, var_817);
                        wp::array_store(var_13, var_818, var_816);
                        // adr_slot += 3                                                          <L 1917>
                        var_820 = wp::add(var_801, var_819);
                    }
                    var_821 = wp::where(var_648, var_820, var_801);
                    // if torque:                                                                 <L 1918>
                    if (var_649) {
                        // out[adr_slot + 0] = contact_forcetorque[3]                             <L 1919>
                        var_823 = wp::extract(var_805, var_822);
                        var_825 = wp::add(var_821, var_824);
                        wp::array_store(var_13, var_825, var_823);
                        // out[adr_slot + 1] = contact_forcetorque[4]                             <L 1920>
                        var_827 = wp::extract(var_805, var_826);
                        var_829 = wp::add(var_821, var_828);
                        wp::array_store(var_13, var_829, var_827);
                        // out[adr_slot + 2] = dir * contact_forcetorque[5]                       <L 1921>
                        var_831 = wp::extract(var_805, var_830);
                        var_832 = wp::mul(var_794, var_831);
                        var_834 = wp::add(var_821, var_833);
                        wp::array_store(var_13, var_834, var_832);
                        // adr_slot += 3                                                          <L 1922>
                        var_836 = wp::add(var_821, var_835);
                    }
                    var_837 = wp::where(var_649, var_836, var_821);
                    // if dist:                                                                   <L 1923>
                    if (var_650) {
                        // out[adr_slot] = contact_dist_in[cid]                                   <L 1924>
                        var_838 = wp::address(var_contact_dist_in, var_791);
                        var_839 = wp::load(var_838);
                        wp::array_store(var_13, var_837, var_839);
                        // adr_slot += 1                                                          <L 1925>
                        var_841 = wp::add(var_837, var_840);
                    }
                    var_842 = wp::where(var_650, var_841, var_837);
                    // if pos:                                                                    <L 1926>
                    if (var_651) {
                        // contact_pos = contact_pos_in[cid]                                      <L 1927>
                        var_843 = wp::address(var_contact_pos_in, var_791);
                        var_845 = wp::load(var_843);
                        var_844 = wp::copy(var_845);
                        // out[adr_slot + 0] = contact_pos[0]                                     <L 1928>
                        var_847 = wp::extract(var_844, var_846);
                        var_849 = wp::add(var_842, var_848);
                        wp::array_store(var_13, var_849, var_847);
                        // out[adr_slot + 1] = contact_pos[1]                                     <L 1929>
                        var_851 = wp::extract(var_844, var_850);
                        var_853 = wp::add(var_842, var_852);
                        wp::array_store(var_13, var_853, var_851);
                        // out[adr_slot + 2] = contact_pos[2]                                     <L 1930>
                        var_855 = wp::extract(var_844, var_854);
                        var_857 = wp::add(var_842, var_856);
                        wp::array_store(var_13, var_857, var_855);
                        // adr_slot += 3                                                          <L 1931>
                        var_859 = wp::add(var_842, var_858);
                    }
                    var_860 = wp::where(var_651, var_844, var_688);
                    var_861 = wp::where(var_651, var_859, var_842);
                    // if normal:                                                                 <L 1932>
                    if (var_652) {
                        // contact_normal = contact_frame_in[cid][0]                              <L 1933>
                        var_862 = wp::address(var_contact_frame_in, var_791);
                        var_865 = wp::load(var_862);
                        var_864 = wp::extract(var_865, var_863);
                        // out[adr_slot + 0] = dir * contact_normal[0]                            <L 1934>
                        var_867 = wp::extract(var_864, var_866);
                        var_868 = wp::mul(var_794, var_867);
                        var_870 = wp::add(var_861, var_869);
                        wp::array_store(var_13, var_870, var_868);
                        // out[adr_slot + 1] = dir * contact_normal[1]                            <L 1935>
                        var_872 = wp::extract(var_864, var_871);
                        var_873 = wp::mul(var_794, var_872);
                        var_875 = wp::add(var_861, var_874);
                        wp::array_store(var_13, var_875, var_873);
                        // out[adr_slot + 2] = dir * contact_normal[2]                            <L 1936>
                        var_877 = wp::extract(var_864, var_876);
                        var_878 = wp::mul(var_794, var_877);
                        var_880 = wp::add(var_861, var_879);
                        wp::array_store(var_13, var_880, var_878);
                        // adr_slot += 3                                                          <L 1937>
                        var_882 = wp::add(var_861, var_881);
                    }
                    var_883 = wp::where(var_652, var_882, var_861);
                    // if tangent:                                                                <L 1938>
                    if (var_653) {
                        // contact_tangent = contact_frame_in[cid][1]                             <L 1939>
                        var_884 = wp::address(var_contact_frame_in, var_791);
                        var_887 = wp::load(var_884);
                        var_886 = wp::extract(var_887, var_885);
                        // out[adr_slot + 0] = dir * contact_tangent[0]                           <L 1940>
                        var_889 = wp::extract(var_886, var_888);
                        var_890 = wp::mul(var_794, var_889);
                        var_892 = wp::add(var_883, var_891);
                        wp::array_store(var_13, var_892, var_890);
                        // out[adr_slot + 1] = dir * contact_tangent[1]                           <L 1941>
                        var_894 = wp::extract(var_886, var_893);
                        var_895 = wp::mul(var_794, var_894);
                        var_897 = wp::add(var_883, var_896);
                        wp::array_store(var_13, var_897, var_895);
                        // out[adr_slot + 2] = dir * contact_tangent[2]                           <L 1942>
                        var_899 = wp::extract(var_886, var_898);
                        var_900 = wp::mul(var_794, var_899);
                        var_902 = wp::add(var_883, var_901);
                        wp::array_store(var_13, var_902, var_900);
                    }
                    wp::assign(var_678, var_791);
                    wp::assign(var_681, var_794);
                    wp::assign(var_693, var_805);
                    wp::assign(var_688, var_860);
                    wp::assign(var_776, var_883);
                    goto start_for_2;
                end_for_2:;
                // for i in range(nmatch, num):                                                   <L 1945>
                var_903 = wp::range(var_663, var_655);
                start_for_4:;
                    if (iter_cmp(var_903) == 0) goto end_for_4;
                    var_904 = wp::iter_next(var_903);
                    // for j in range(size):                                                      <L 1946>
                    var_905 = wp::range(var_654);
                    start_for_6:;
                        if (iter_cmp(var_905) == 0) goto end_for_6;
                        var_906 = wp::iter_next(var_905);
                        // out[adr + i * size + j] = 0.0                                          <L 1947>
                        var_908 = wp::mul(var_904, var_654);
                        var_909 = wp::add(var_657, var_908);
                        var_910 = wp::add(var_909, var_906);
                        wp::array_store(var_13, var_910, var_907);
                        goto start_for_6;
                    end_for_6:;
                    goto start_for_4;
                end_for_4:;
            }
            var_911 = wp::where(var_666, var_786, var_904);
        }
        if (!var_15) {
            // elif sensortype == SensorType.ACCELEROMETER:                                       <L 1949>
            var_913 = (var_6 == var_912);
            if (var_913) {
                // vec3 = _accelerometer(                                                         <L 1950>
                // body_rootid, site_bodyid, site_xpos_in, site_xmat_in, subtree_com_in, cvel_in, cacc_in, worldid, objid       <L 1951>
                var_914 = _accelerometer_0(var_body_rootid, var_site_bodyid, var_site_xpos_in, var_site_xmat_in, var_subtree_com_in, var_cvel_in, var_cacc_in, var_0, var_9);
                // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)       <L 1953>
                _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_915, var_914, var_13);
            }
            if (!var_913) {
                // elif sensortype == SensorType.FORCE:                                           <L 1954>
                var_917 = (var_6 == var_916);
                if (var_917) {
                    // vec3 = _force(site_bodyid, site_xmat_in, cfrc_int_in, worldid, objid)       <L 1955>
                    var_918 = _force_0(var_site_bodyid, var_site_xmat_in, var_cfrc_int_in, var_0, var_9);
                    // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)       <L 1956>
                    _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_919, var_918, var_13);
                }
                var_920 = wp::where(var_917, var_918, var_914);
                if (!var_917) {
                    // elif sensortype == SensorType.TORQUE:                                      <L 1957>
                    var_922 = (var_6 == var_921);
                    if (var_922) {
                        // vec3 = _torque(body_rootid, site_bodyid, site_xpos_in, site_xmat_in, subtree_com_in, cfrc_int_in, worldid, objid)       <L 1958>
                        var_923 = _torque_0(var_body_rootid, var_site_bodyid, var_site_xpos_in, var_site_xmat_in, var_subtree_com_in, var_cfrc_int_in, var_0, var_9);
                        // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)       <L 1959>
                        _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_924, var_923, var_13);
                    }
                    var_925 = wp::where(var_922, var_923, var_920);
                    if (!var_922) {
                        // elif sensortype == SensorType.ACTUATORFRC:                             <L 1960>
                        var_927 = (var_6 == var_926);
                        if (var_927) {
                            // val = _actuator_force(actuator_force_in, worldid, objid)           <L 1961>
                            var_928 = _actuator_force_0(var_actuator_force_in, var_0, var_9);
                            // _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)       <L 1962>
                            _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_928, var_13);
                        }
                        if (!var_927) {
                            // elif sensortype == SensorType.JOINTACTFRC:                         <L 1963>
                            var_930 = (var_6 == var_929);
                            if (var_930) {
                                // val = _joint_actuator_force(jnt_dofadr, qfrc_actuator_in, worldid, objid)       <L 1964>
                                var_931 = _joint_actuator_force_0(var_jnt_dofadr, var_qfrc_actuator_in, var_0, var_9);
                                // _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)       <L 1965>
                                _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_931, var_13);
                            }
                            var_932 = wp::where(var_930, var_931, var_928);
                            if (!var_930) {
                                // elif sensortype == SensorType.FRAMELINACC:                     <L 1966>
                                var_934 = (var_6 == var_933);
                                if (var_934) {
                                    // objtype = sensor_objtype[sensorid]                         <L 1967>
                                    var_935 = wp::address(var_sensor_objtype, var_3);
                                    var_937 = wp::load(var_935);
                                    var_936 = wp::copy(var_937);
                                    // vec3 = _framelinacc(                                       <L 1968>
                                    // body_rootid,                                               <L 1969>
                                    // geom_bodyid,                                               <L 1970>
                                    // site_bodyid,                                               <L 1971>
                                    // cam_bodyid,                                                <L 1972>
                                    // xpos_in,                                                   <L 1973>
                                    // xipos_in,                                                  <L 1974>
                                    // geom_xpos_in,                                              <L 1975>
                                    // site_xpos_in,                                              <L 1976>
                                    // cam_xpos_in,                                               <L 1977>
                                    // subtree_com_in,                                            <L 1978>
                                    // cvel_in,                                                   <L 1979>
                                    // cacc_in,                                                   <L 1980>
                                    // worldid,                                                   <L 1981>
                                    // objid,                                                     <L 1982>
                                    // objtype,                                                   <L 1983>
                                    var_938 = _framelinacc_0(var_body_rootid, var_geom_bodyid, var_site_bodyid, var_cam_bodyid, var_xpos_in, var_xipos_in, var_geom_xpos_in, var_site_xpos_in, var_cam_xpos_in, var_subtree_com_in, var_cvel_in, var_cacc_in, var_0, var_9, var_936);
                                    // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)       <L 1985>
                                    _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_939, var_938, var_13);
                                }
                                var_940 = wp::where(var_934, var_936, var_24);
                                var_941 = wp::where(var_934, var_938, var_925);
                                if (!var_934) {
                                    // elif sensortype == SensorType.FRAMEANGACC:                 <L 1986>
                                    var_943 = (var_6 == var_942);
                                    if (var_943) {
                                        // objtype = sensor_objtype[sensorid]                     <L 1987>
                                        var_944 = wp::address(var_sensor_objtype, var_3);
                                        var_946 = wp::load(var_944);
                                        var_945 = wp::copy(var_946);
                                        // vec3 = _frameangacc(                                   <L 1988>
                                        // geom_bodyid,                                           <L 1989>
                                        // site_bodyid,                                           <L 1990>
                                        // cam_bodyid,                                            <L 1991>
                                        // cacc_in,                                               <L 1992>
                                        // worldid,                                               <L 1993>
                                        // objid,                                                 <L 1994>
                                        // objtype,                                               <L 1995>
                                        var_947 = _frameangacc_0(var_geom_bodyid, var_site_bodyid, var_cam_bodyid, var_cacc_in, var_0, var_9, var_945);
                                        // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)       <L 1997>
                                        _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_948, var_947, var_13);
                                    }
                                    var_949 = wp::where(var_943, var_945, var_940);
                                    var_950 = wp::where(var_943, var_947, var_941);
                                }
                                var_951 = wp::where(var_934, var_940, var_949);
                                var_952 = wp::where(var_934, var_941, var_950);
                            }
                            var_953 = wp::where(var_930, var_24, var_951);
                            var_954 = wp::where(var_930, var_925, var_952);
                        }
                        var_955 = wp::where(var_927, var_24, var_953);
                        var_956 = wp::where(var_927, var_925, var_954);
                        var_957 = wp::where(var_927, var_928, var_932);
                    }
                    var_958 = wp::where(var_922, var_24, var_955);
                    var_959 = wp::where(var_922, var_925, var_956);
                }
                var_960 = wp::where(var_917, var_24, var_958);
                var_961 = wp::where(var_917, var_920, var_959);
            }
            var_962 = wp::where(var_913, var_24, var_960);
            var_963 = wp::where(var_913, var_914, var_961);
        }
        var_964 = wp::where(var_15, var_24, var_962);
    }
}



extern "C" __global__ void _contact_match_00419bb0_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_opt_cone,
    wp::int32 var_opt_contact_sensor_maxmatch,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_site_type,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_size,
    wp::array_t<wp::int32> var_sensor_objtype,
    wp::array_t<wp::int32> var_sensor_objid,
    wp::array_t<wp::int32> var_sensor_reftype,
    wp::array_t<wp::int32> var_sensor_refid,
    wp::array_t<wp::int32> var_sensor_intprm,
    wp::array_t<wp::int32> var_sensor_contact_adr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::float32> var_contact_dist_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::int32> var_contact_worldid_in,
    wp::array_t<wp::int32> var_contact_type_in,
    wp::array_t<wp::float32> var_efc_force_in,
    wp::int32 var_njmax_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::int32> var_sensor_contact_nmatch_out,
    wp::array_t<wp::int32> var_sensor_contact_matchid_out,
    wp::array_t<wp::float32> var_sensor_contact_criteria_out,
    wp::array_t<wp::float32> var_sensor_contact_direction_out)
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
        wp::int32* var_6;
        bool var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        const wp::int32 var_10 = 1;
        wp::int32 var_11;
        wp::int32 var_12;
        bool var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::int32* var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        const wp::int32 var_26 = 1;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        const wp::int32 var_33 = 6;
        bool var_34;
        wp::vec_t<3, wp::float32>* var_35;
        wp::mat_t<3, 3, wp::float32>* var_36;
        wp::vec_t<3, wp::float32>* var_37;
        wp::int32* var_38;
        wp::vec_t<3, wp::float32>* var_39;
        bool var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::mat_t<3, 3, wp::float32> var_42;
        wp::vec_t<3, wp::float32> var_43;
        wp::int32 var_44;
        wp::vec_t<3, wp::float32> var_45;
        bool var_46;
        const wp::int32 var_47 = 0;
        bool var_48;
        const wp::int32 var_49 = 0;
        bool var_50;
        bool var_51;
        const wp::float32 var_52 = 1.0;
        wp::vec_t<2, wp::int32>* var_53;
        wp::vec_t<2, wp::int32> var_54;
        wp::vec_t<2, wp::int32> var_55;
        const wp::int32 var_56 = 0;
        wp::int32 var_57;
        const wp::int32 var_58 = 1;
        wp::int32 var_59;
        wp::int32* var_60;
        wp::int32 var_61;
        wp::int32 var_62;
        wp::int32* var_63;
        wp::int32 var_64;
        wp::int32 var_65;
        bool var_66;
        bool var_67;
        bool var_68;
        bool var_69;
        bool var_70;
        bool var_71;
        bool var_72;
        bool var_73;
        bool var_74;
        bool var_75;
        const wp::float32 var_76 = 1.0;
        const wp::int32 var_77 = 0;
        bool var_78;
        const wp::int32 var_79 = 0;
        bool var_80;
        bool var_81;
        bool var_82;
        bool var_83;
        bool var_84;
        bool var_85;
        bool var_86;
        bool var_87;
        bool var_88;
        const wp::float32 var_89 = 1.0;
        const wp::float32 var_90 = -1.0;
        wp::float32 var_91;
        wp::float32 var_92;
        const wp::int32 var_93 = 0;
        bool var_94;
        bool var_95;
        const wp::float32 var_96 = 1.0;
        const wp::float32 var_97 = -1.0;
        wp::float32 var_98;
        wp::float32 var_99;
        const wp::int32 var_100 = 0;
        bool var_101;
        bool var_102;
        const wp::float32 var_103 = 1.0;
        const wp::float32 var_104 = -1.0;
        wp::float32 var_105;
        wp::float32 var_106;
        wp::float32 var_107;
        wp::float32 var_108;
        wp::float32 var_109;
        wp::slice_t var_110;
        const wp::int32 var_111 = 0;
        wp::array_t<wp::int32> var_112;
        const wp::int32 var_113 = 1;
        wp::int32 var_114;
        bool var_115;
        const wp::str var_116 = "contact match overflow: please increase Option.contact_sensor_maxmatch to %u\n";
        const wp::int32 var_117 = 1;
        bool var_118;
        wp::float32* var_119;
        wp::float32 var_120;
        const wp::int32 var_121 = 2;
        bool var_122;
        const bool var_123 = false;
        wp::vec_t<6, wp::float32> var_124;
        const wp::int32 var_125 = 0;
        wp::float32 var_126;
        const wp::int32 var_127 = 0;
        wp::float32 var_128;
        wp::float32 var_129;
        const wp::int32 var_130 = 1;
        wp::float32 var_131;
        const wp::int32 var_132 = 1;
        wp::float32 var_133;
        wp::float32 var_134;
        wp::float32 var_135;
        const wp::int32 var_136 = 2;
        wp::float32 var_137;
        const wp::int32 var_138 = 2;
        wp::float32 var_139;
        wp::float32 var_140;
        wp::float32 var_141;
        wp::float32 var_142;
        //---------
        // forward
        // def _contact_match(                                                                    <L 2275>
        // contactsensorid, contactid = wp.tid()                                                  <L 2310>
        builtin_tid2d(var_0, var_1);
        // sensorid = sensor_contact_adr[contactsensorid]                                         <L 2311>
        var_2 = wp::address(var_sensor_contact_adr, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if contactid >= nacon_in[0]:                                                           <L 2313>
        var_6 = wp::address(var_nacon_in, var_5);
        var_8 = wp::load(var_6);
        var_7 = (var_1 >= var_8);
        if (var_7) {
            // return                                                                             <L 2314>
            continue;
        }
        // if not contact_type_in[contactid] & ContactType.CONSTRAINT:                            <L 2316>
        var_9 = wp::address(var_contact_type_in, var_1);
        var_12 = wp::load(var_9);
        var_11 = wp::bit_and(var_12, var_10);
        var_13 = wp::unot(var_11);
        if (var_13) {
            // return                                                                             <L 2317>
            continue;
        }
        // objtype = sensor_objtype[sensorid]                                                     <L 2320>
        var_14 = wp::address(var_sensor_objtype, var_3);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // objid = sensor_objid[sensorid]                                                         <L 2321>
        var_17 = wp::address(var_sensor_objid, var_3);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // reftype = sensor_reftype[sensorid]                                                     <L 2322>
        var_20 = wp::address(var_sensor_reftype, var_3);
        var_22 = wp::load(var_20);
        var_21 = wp::copy(var_22);
        // refid = sensor_refid[sensorid]                                                         <L 2323>
        var_23 = wp::address(var_sensor_refid, var_3);
        var_25 = wp::load(var_23);
        var_24 = wp::copy(var_25);
        // reduce = sensor_intprm[sensorid, 1]                                                    <L 2324>
        var_27 = wp::address(var_sensor_intprm, var_3, var_26);
        var_29 = wp::load(var_27);
        var_28 = wp::copy(var_29);
        // worldid = contact_worldid_in[contactid]                                                <L 2326>
        var_30 = wp::address(var_contact_worldid_in, var_1);
        var_32 = wp::load(var_30);
        var_31 = wp::copy(var_32);
        // if objtype == ObjType.SITE:                                                            <L 2329>
        var_34 = (var_15 == var_33);
        if (var_34) {
            // if not inside_geom(                                                                <L 2330>
            // site_xpos_in[worldid, objid], site_xmat_in[worldid, objid], site_size[objid], site_type[objid], contact_pos_in[contactid]       <L 2331>
            var_35 = wp::address(var_site_xpos_in, var_31, var_18);
            var_36 = wp::address(var_site_xmat_in, var_31, var_18);
            var_37 = wp::address(var_site_size, var_18);
            var_38 = wp::address(var_site_type, var_18);
            var_39 = wp::address(var_contact_pos_in, var_1);
            var_41 = wp::load(var_35);
            var_42 = wp::load(var_36);
            var_43 = wp::load(var_37);
            var_44 = wp::load(var_38);
            var_45 = wp::load(var_39);
            var_40 = inside_geom_0(var_41, var_42, var_43, var_44, var_45);
            var_46 = wp::unot(var_40);
            if (var_46) {
                // return                                                                         <L 2333>
                continue;
            }
        }
        // if objtype == ObjType.UNKNOWN and reftype == ObjType.UNKNOWN:                          <L 2336>
        var_48 = (var_15 == var_47);
        var_50 = (var_21 == var_49);
        var_51 = var_48 && var_50;
        if (var_51) {
            // dir = 1.0                                                                          <L 2337>
        }
        if (!var_51) {
            // geom = contact_geom_in[contactid]                                                  <L 2340>
            var_53 = wp::address(var_contact_geom_in, var_1);
            var_55 = wp::load(var_53);
            var_54 = wp::copy(var_55);
            // geom1 = geom[0]                                                                    <L 2341>
            var_57 = wp::extract(var_54, var_56);
            // geom2 = geom[1]                                                                    <L 2342>
            var_59 = wp::extract(var_54, var_58);
            // body1 = geom_bodyid[geom1]                                                         <L 2343>
            var_60 = wp::address(var_geom_bodyid, var_57);
            var_62 = wp::load(var_60);
            var_61 = wp::copy(var_62);
            // body2 = geom_bodyid[geom2]                                                         <L 2344>
            var_63 = wp::address(var_geom_bodyid, var_59);
            var_65 = wp::load(var_63);
            var_64 = wp::copy(var_65);
            // match11 = _check_match(body_parentid, body1, geom1, objtype, objid)                <L 2347>
            var_66 = _check_match_0(var_body_parentid, var_61, var_57, var_15, var_18);
            // match12 = _check_match(body_parentid, body2, geom2, objtype, objid)                <L 2348>
            var_67 = _check_match_0(var_body_parentid, var_64, var_59, var_15, var_18);
            // match21 = _check_match(body_parentid, body1, geom1, reftype, refid)                <L 2349>
            var_68 = _check_match_0(var_body_parentid, var_61, var_57, var_21, var_24);
            // match22 = _check_match(body_parentid, body2, geom2, reftype, refid)                <L 2350>
            var_69 = _check_match_0(var_body_parentid, var_64, var_59, var_21, var_24);
            // if not match11 and not match12:                                                    <L 2353>
            var_70 = wp::unot(var_66);
            var_71 = wp::unot(var_67);
            var_72 = var_70 && var_71;
            if (var_72) {
                // return                                                                         <L 2354>
                continue;
            }
            // if not match21 and not match22:                                                    <L 2355>
            var_73 = wp::unot(var_68);
            var_74 = wp::unot(var_69);
            var_75 = var_73 && var_74;
            if (var_75) {
                // return                                                                         <L 2356>
                continue;
            }
            // dir = 1.0                                                                          <L 2359>
            // if objtype != ObjType.UNKNOWN and reftype != ObjType.UNKNOWN:                      <L 2360>
            var_78 = (var_15 != var_77);
            var_80 = (var_21 != var_79);
            var_81 = var_78 && var_80;
            if (var_81) {
                // order_regular = match11 and match22                                            <L 2362>
                var_82 = var_66 && var_69;
                // order_reverse = match12 and match21                                            <L 2363>
                var_83 = var_67 && var_68;
                // if not order_regular and not order_reverse:                                    <L 2364>
                var_84 = wp::unot(var_82);
                var_85 = wp::unot(var_83);
                var_86 = var_84 && var_85;
                if (var_86) {
                    // return                                                                     <L 2365>
                    continue;
                }
                // if order_reverse and not order_regular:                                        <L 2366>
                var_87 = wp::unot(var_82);
                var_88 = var_83 && var_87;
                if (var_88) {
                    // dir = -1.0                                                                 <L 2367>
                }
                var_91 = wp::where(var_88, var_90, var_76);
            }
            var_92 = wp::where(var_81, var_91, var_76);
            if (!var_81) {
                // elif objtype != ObjType.UNKNOWN:                                               <L 2368>
                var_94 = (var_15 != var_93);
                if (var_94) {
                    // if not match11:                                                            <L 2369>
                    var_95 = wp::unot(var_66);
                    if (var_95) {
                        // dir = -1.0                                                             <L 2370>
                    }
                    var_98 = wp::where(var_95, var_97, var_92);
                }
                var_99 = wp::where(var_94, var_98, var_92);
                if (!var_94) {
                    // elif reftype != ObjType.UNKNOWN:                                           <L 2371>
                    var_101 = (var_21 != var_100);
                    if (var_101) {
                        // if not match22:                                                        <L 2372>
                        var_102 = wp::unot(var_69);
                        if (var_102) {
                            // dir = -1.0                                                         <L 2373>
                        }
                        var_105 = wp::where(var_102, var_104, var_99);
                    }
                    var_106 = wp::where(var_101, var_105, var_99);
                }
                var_107 = wp::where(var_94, var_99, var_106);
            }
            var_108 = wp::where(var_81, var_92, var_107);
        }
        var_109 = wp::where(var_51, var_52, var_108);
        // contactmatchid = wp.atomic_add(sensor_contact_nmatch_out[worldid], contactsensorid, 1)       <L 2375>
        var_110 = wp::slice_t(var_31, var_31, var_111);
        var_112 = wp::view(var_sensor_contact_nmatch_out, var_110);
        var_114 = wp::atomic_add(var_112, var_0, var_113);
        // if contactmatchid >= opt_contact_sensor_maxmatch:                                      <L 2377>
        var_115 = (var_114 >= var_opt_contact_sensor_maxmatch);
        if (var_115) {
            // wp.printf("contact match overflow: please increase Option.contact_sensor_maxmatch to %u\n", contactmatchid)       <L 2379>
            printf(var_116, var_114);
            // return                                                                             <L 2380>
            continue;
        }
        // sensor_contact_matchid_out[worldid, contactsensorid, contactmatchid] = contactid       <L 2382>
        wp::array_store(var_sensor_contact_matchid_out, var_31, var_0, var_114, var_1);
        // if reduce == 1:  # mindist                                                             <L 2384>
        var_118 = (var_28 == var_117);
        if (var_118) {
            // sensor_contact_criteria_out[worldid, contactsensorid, contactmatchid] = contact_dist_in[contactid]       <L 2385>
            var_119 = wp::address(var_contact_dist_in, var_1);
            var_120 = wp::load(var_119);
            wp::array_store(var_sensor_contact_criteria_out, var_31, var_0, var_114, var_120);
        }
        if (!var_118) {
            // elif reduce == 2:  # maxforce                                                      <L 2386>
            var_122 = (var_28 == var_121);
            if (var_122) {
                // contact_force = support.contact_force_fn(                                      <L 2387>
                // opt_cone,                                                                      <L 2388>
                // contact_frame_in,                                                              <L 2389>
                // contact_friction_in,                                                           <L 2390>
                // contact_dim_in,                                                                <L 2391>
                // contact_efc_address_in,                                                        <L 2392>
                // efc_force_in,                                                                  <L 2393>
                // njmax_in,                                                                      <L 2394>
                // nacon_in,                                                                      <L 2395>
                // worldid,                                                                       <L 2396>
                // contactid,                                                                     <L 2397>
                // False,                                                                         <L 2398>
                var_124 = contact_force_fn_0(var_opt_cone, var_contact_frame_in, var_contact_friction_in, var_contact_dim_in, var_contact_efc_address_in, var_efc_force_in, var_njmax_in, var_nacon_in, var_31, var_1, var_123);
                // force_magnitude = (                                                            <L 2400>
                // contact_force[0] * contact_force[0] + contact_force[1] * contact_force[1] + contact_force[2] * contact_force[2]       <L 2401>
                var_126 = wp::extract(var_124, var_125);
                var_128 = wp::extract(var_124, var_127);
                var_129 = wp::mul(var_126, var_128);
                var_131 = wp::extract(var_124, var_130);
                var_133 = wp::extract(var_124, var_132);
                var_134 = wp::mul(var_131, var_133);
                var_135 = wp::add(var_129, var_134);
                var_137 = wp::extract(var_124, var_136);
                var_139 = wp::extract(var_124, var_138);
                var_140 = wp::mul(var_137, var_139);
                var_141 = wp::add(var_135, var_140);
                // sensor_contact_criteria_out[worldid, contactsensorid, contactmatchid] = -force_magnitude       <L 2403>
                var_142 = wp::neg(var_141);
                wp::array_store(var_sensor_contact_criteria_out, var_31, var_0, var_114, var_142);
            }
        }
        // sensor_contact_direction_out[worldid, contactsensorid, contactmatchid] = dir           <L 2406>
        wp::array_store(var_sensor_contact_direction_out, var_31, var_0, var_114, var_109);
    }
}



extern "C" __global__ void _energy_pos_gravity_2e5e6dd5_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_opt_gravity,
    wp::array_t<wp::float32> var_body_mass,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<2, wp::float32>> var_energy_out)
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
        wp::vec_t<3, wp::float32>* var_7;
        wp::vec_t<3, wp::float32> var_8;
        wp::vec_t<3, wp::float32> var_9;
        const wp::int32 var_10 = 1;
        wp::int32 var_11;
        wp::shape_t* var_12;
        const wp::int32 var_13 = 0;
        wp::int32 var_14;
        wp::shape_t var_15;
        wp::int32 var_16;
        wp::float32* var_17;
        wp::vec_t<3, wp::float32>* var_18;
        wp::float32 var_19;
        wp::vec_t<3, wp::float32> var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        const wp::float32 var_23 = 0.0;
        wp::vec_t<2, wp::float32> var_24;
        wp::vec_t<2, wp::float32> var_25;
        //---------
        // forward
        // def _energy_pos_gravity(                                                               <L 2710>
        // worldid, bodyid = wp.tid()                                                             <L 2719>
        builtin_tid2d(var_0, var_1);
        // gravity = opt_gravity[worldid % opt_gravity.shape[0]]                                  <L 2720>
        var_2 = &(var_opt_gravity.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_opt_gravity, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // bodyid += 1  # skip world body                                                         <L 2721>
        var_11 = wp::add(var_1, var_10);
        // energy = wp.vec2(                                                                      <L 2723>
        // body_mass[worldid % body_mass.shape[0], bodyid] * wp.dot(gravity, xipos_in[worldid, bodyid]),       <L 2724>
        var_12 = &(var_body_mass.shape);
        var_15 = wp::load(var_12);
        var_14 = wp::extract(var_15, var_13);
        var_16 = wp::mod(var_0, var_14);
        var_17 = wp::address(var_body_mass, var_16, var_11);
        var_18 = wp::address(var_xipos_in, var_0, var_11);
        var_20 = wp::load(var_18);
        var_19 = wp::dot(var_8, var_20);
        var_22 = wp::load(var_17);
        var_21 = wp::mul(var_22, var_19);
        // 0.0,                                                                                   <L 2725>
        var_24 = wp::vec_t<2, wp::float32>(var_21, var_23);
        // wp.atomic_sub(energy_out, worldid, energy)                                             <L 2728>
        var_25 = wp::atomic_sub(var_energy_out, var_0, var_24);
    }
}



extern "C" __global__ void _sensor_rangefinder_init_49b66287_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_sensor_objid,
    wp::array_t<wp::int32> var_sensor_rangefinder_adr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_pnt_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vec_out)
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
        wp::vec_t<3, wp::float32>* var_8;
        wp::vec_t<3, wp::float32> var_9;
        wp::vec_t<3, wp::float32> var_10;
        wp::mat_t<3, 3, wp::float32>* var_11;
        wp::mat_t<3, 3, wp::float32> var_12;
        wp::mat_t<3, 3, wp::float32> var_13;
        const wp::int32 var_14 = 0;
        const wp::int32 var_15 = 2;
        wp::float32 var_16;
        const wp::int32 var_17 = 1;
        const wp::int32 var_18 = 2;
        wp::float32 var_19;
        const wp::int32 var_20 = 2;
        const wp::int32 var_21 = 2;
        wp::float32 var_22;
        wp::vec_t<3, wp::float32> var_23;
        //---------
        // forward
        // def _sensor_rangefinder_init(                                                          <L 195>
        // worldid, rfid = wp.tid()                                                               <L 206>
        builtin_tid2d(var_0, var_1);
        // sensorid = sensor_rangefinder_adr[rfid]                                                <L 207>
        var_2 = wp::address(var_sensor_rangefinder_adr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // objid = sensor_objid[sensorid]                                                         <L 208>
        var_5 = wp::address(var_sensor_objid, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // site_xpos = site_xpos_in[worldid, objid]                                               <L 209>
        var_8 = wp::address(var_site_xpos_in, var_0, var_6);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // site_xmat = site_xmat_in[worldid, objid]                                               <L 210>
        var_11 = wp::address(var_site_xmat_in, var_0, var_6);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // pnt_out[worldid, rfid] = site_xpos                                                     <L 212>
        wp::array_store(var_pnt_out, var_0, var_1, var_9);
        // vec_out[worldid, rfid] = wp.vec3(site_xmat[0, 2], site_xmat[1, 2], site_xmat[2, 2])       <L 213>
        var_16 = wp::extract(var_12, var_14, var_15);
        var_19 = wp::extract(var_12, var_17, var_18);
        var_22 = wp::extract(var_12, var_20, var_21);
        var_23 = wp::vec_t<3, wp::float32>(var_16, var_19, var_22);
        wp::array_store(var_vec_out, var_0, var_1, var_23);
    }
}



extern "C" __global__ void _sensor_collision_bade267e_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_ngeom,
    wp::array_t<wp::vec_t<2, wp::int32>> var_nxn_pairid,
    wp::array_t<wp::float32> var_contact_dist_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_in,
    wp::array_t<wp::int32> var_contact_worldid_in,
    wp::array_t<wp::int32> var_contact_type_in,
    wp::array_t<wp::int32> var_contact_geomcollisionid_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::float32> var_sensor_collision_out)
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
        wp::int32* var_5;
        const wp::int32 var_6 = 2;
        wp::int32 var_7;
        wp::int32 var_8;
        bool var_9;
        wp::vec_t<2, wp::int32>* var_10;
        wp::vec_t<2, wp::int32> var_11;
        wp::vec_t<2, wp::int32> var_12;
        const wp::int32 var_13 = 0;
        wp::int32 var_14;
        const wp::int32 var_15 = 1;
        wp::int32 var_16;
        bool var_17;
        const wp::int32 var_18 = 0;
        wp::int32 var_19;
        const wp::int32 var_20 = 1;
        wp::int32 var_21;
        wp::int32 var_22;
        const wp::int32 var_23 = 1;
        wp::int32 var_24;
        const wp::int32 var_25 = 0;
        wp::int32 var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::int32* var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        wp::vec_t<2, wp::int32>* var_32;
        const wp::int32 var_33 = 1;
        wp::int32 var_34;
        wp::vec_t<2, wp::int32> var_35;
        wp::int32* var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        wp::float32* var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        wp::vec_t<3, wp::float32>* var_42;
        wp::vec_t<3, wp::float32> var_43;
        wp::vec_t<3, wp::float32> var_44;
        wp::mat_t<3, 3, wp::float32>* var_45;
        wp::mat_t<3, 3, wp::float32> var_46;
        wp::mat_t<3, 3, wp::float32> var_47;
        const wp::int32 var_48 = 0;
        const wp::int32 var_49 = 0;
        wp::float32 var_50;
        const wp::int32 var_51 = 0;
        const wp::int32 var_52 = 1;
        wp::float32 var_53;
        const wp::int32 var_54 = 0;
        const wp::int32 var_55 = 2;
        wp::float32 var_56;
        wp::vec_t<3, wp::float32> var_57;
        const wp::float32 var_58 = 0.5;
        wp::float32 var_59;
        wp::vec_t<3, wp::float32> var_60;
        wp::vec_t<3, wp::float32> var_61;
        const wp::float32 var_62 = 0.5;
        wp::float32 var_63;
        wp::vec_t<3, wp::float32> var_64;
        wp::vec_t<3, wp::float32> var_65;
        const wp::int32 var_66 = 0;
        const wp::int32 var_67 = 0;
        wp::float32 var_68;
        const wp::int32 var_69 = 1;
        const wp::int32 var_70 = 1;
        wp::float32 var_71;
        const wp::int32 var_72 = 2;
        const wp::int32 var_73 = 2;
        wp::float32 var_74;
        const wp::int32 var_75 = 3;
        const wp::int32 var_76 = 0;
        wp::float32 var_77;
        const wp::int32 var_78 = 4;
        const wp::int32 var_79 = 1;
        wp::float32 var_80;
        const wp::int32 var_81 = 5;
        const wp::int32 var_82 = 2;
        wp::float32 var_83;
        const wp::int32 var_84 = 6;
        //---------
        // forward
        // def _sensor_collision(                                                                 <L 710>
        // conid = wp.tid()                                                                       <L 726>
        var_0 = builtin_tid1d();
        // if conid >= nacon_in[0]:                                                               <L 728>
        var_2 = wp::address(var_nacon_in, var_1);
        var_4 = wp::load(var_2);
        var_3 = (var_0 >= var_4);
        if (var_3) {
            // return                                                                             <L 729>
            continue;
        }
        // if not contact_type_in[conid] & ContactType.SENSOR:                                    <L 731>
        var_5 = wp::address(var_contact_type_in, var_0);
        var_8 = wp::load(var_5);
        var_7 = wp::bit_and(var_8, var_6);
        var_9 = wp::unot(var_7);
        if (var_9) {
            // return                                                                             <L 732>
            continue;
        }
        // geom = contact_geom_in[conid]                                                          <L 734>
        var_10 = wp::address(var_contact_geom_in, var_0);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // if geom[0] <= geom[1]:                                                                 <L 735>
        var_14 = wp::extract(var_11, var_13);
        var_16 = wp::extract(var_11, var_15);
        var_17 = (var_14 <= var_16);
        if (var_17) {
            // pairid = math.upper_tri_index(ngeom, geom[0], geom[1])                             <L 736>
            var_19 = wp::extract(var_11, var_18);
            var_21 = wp::extract(var_11, var_20);
            var_22 = upper_tri_index_0(var_ngeom, var_19, var_21);
        }
        if (!var_17) {
            // pairid = math.upper_tri_index(ngeom, geom[1], geom[0])                             <L 738>
            var_24 = wp::extract(var_11, var_23);
            var_26 = wp::extract(var_11, var_25);
            var_27 = upper_tri_index_0(var_ngeom, var_24, var_26);
        }
        var_28 = wp::where(var_17, var_22, var_27);
        // worldid = contact_worldid_in[conid]                                                    <L 740>
        var_29 = wp::address(var_contact_worldid_in, var_0);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // collisionid = nxn_pairid[pairid][1]                                                    <L 741>
        var_32 = wp::address(var_nxn_pairid, var_28);
        var_35 = wp::load(var_32);
        var_34 = wp::extract(var_35, var_33);
        // geomcollisionid = contact_geomcollisionid_in[conid]                                    <L 742>
        var_36 = wp::address(var_contact_geomcollisionid_in, var_0);
        var_38 = wp::load(var_36);
        var_37 = wp::copy(var_38);
        // dist = contact_dist_in[conid]                                                          <L 744>
        var_39 = wp::address(var_contact_dist_in, var_0);
        var_41 = wp::load(var_39);
        var_40 = wp::copy(var_41);
        // pos = contact_pos_in[conid]                                                            <L 745>
        var_42 = wp::address(var_contact_pos_in, var_0);
        var_44 = wp::load(var_42);
        var_43 = wp::copy(var_44);
        // frame = contact_frame_in[conid]                                                        <L 746>
        var_45 = wp::address(var_contact_frame_in, var_0);
        var_47 = wp::load(var_45);
        var_46 = wp::copy(var_47);
        // normal = wp.vec3(frame[0, 0], frame[0, 1], frame[0, 2])                                <L 747>
        var_50 = wp::extract(var_46, var_48, var_49);
        var_53 = wp::extract(var_46, var_51, var_52);
        var_56 = wp::extract(var_46, var_54, var_55);
        var_57 = wp::vec_t<3, wp::float32>(var_50, var_53, var_56);
        // pnt1 = pos - 0.5 * dist * normal                                                       <L 748>
        var_59 = wp::mul(var_58, var_40);
        var_60 = wp::mul(var_59, var_57);
        var_61 = wp::sub(var_43, var_60);
        // pnt2 = pos + 0.5 * dist * normal                                                       <L 749>
        var_63 = wp::mul(var_62, var_40);
        var_64 = wp::mul(var_63, var_57);
        var_65 = wp::add(var_43, var_64);
        // sensor_collision_out[worldid, collisionid, geomcollisionid, 0] = dist                  <L 751>
        wp::array_store(var_sensor_collision_out, var_30, var_34, var_37, var_66, var_40);
        // sensor_collision_out[worldid, collisionid, geomcollisionid, 1] = pnt1[0]               <L 752>
        var_68 = wp::extract(var_61, var_67);
        wp::array_store(var_sensor_collision_out, var_30, var_34, var_37, var_69, var_68);
        // sensor_collision_out[worldid, collisionid, geomcollisionid, 2] = pnt1[1]               <L 753>
        var_71 = wp::extract(var_61, var_70);
        wp::array_store(var_sensor_collision_out, var_30, var_34, var_37, var_72, var_71);
        // sensor_collision_out[worldid, collisionid, geomcollisionid, 3] = pnt1[2]               <L 754>
        var_74 = wp::extract(var_61, var_73);
        wp::array_store(var_sensor_collision_out, var_30, var_34, var_37, var_75, var_74);
        // sensor_collision_out[worldid, collisionid, geomcollisionid, 4] = pnt2[0]               <L 755>
        var_77 = wp::extract(var_65, var_76);
        wp::array_store(var_sensor_collision_out, var_30, var_34, var_37, var_78, var_77);
        // sensor_collision_out[worldid, collisionid, geomcollisionid, 5] = pnt2[1]               <L 756>
        var_80 = wp::extract(var_65, var_79);
        wp::array_store(var_sensor_collision_out, var_30, var_34, var_37, var_81, var_80);
        // sensor_collision_out[worldid, collisionid, geomcollisionid, 6] = pnt2[2]               <L 757>
        var_83 = wp::extract(var_65, var_82);
        wp::array_store(var_sensor_collision_out, var_30, var_34, var_37, var_84, var_83);
    }
}



extern "C" __global__ void _sensor_tactile_c542357d_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_weldid,
    wp::array_t<wp::vec_t<8, wp::int32>> var_oct_child,
    wp::array_t<wp::vec_t<3, wp::float32>> var_oct_aabb,
    wp::array_t<wp::vec_t<8, wp::float32>> var_oct_coeff,
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_geom_dataid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::int32> var_mesh_vertadr,
    wp::array_t<wp::int32> var_mesh_vertnum,
    wp::array_t<wp::int32> var_mesh_octadr,
    wp::array_t<wp::int32> var_mesh_normaladr,
    wp::array_t<wp::int32> var_mesh_normalnum,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_vert,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_normal,
    wp::array_t<wp::quat_t<wp::float32>> var_mesh_quat,
    wp::array_t<wp::int32> var_sensor_objid,
    wp::array_t<wp::int32> var_sensor_refid,
    wp::array_t<wp::int32> var_sensor_dim,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::int32> var_plugin,
    wp::array_t<wp::vec_t<128, wp::float32>> var_plugin_attr,
    wp::array_t<wp::int32> var_geom_plugin_index,
    wp::array_t<wp::int32> var_taxel_vertadr,
    wp::array_t<wp::int32> var_taxel_sensorid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::int32> var_weld_geom_count_in,
    wp::array_t<wp::int32> var_weld_geom_list_in,
    wp::array_t<wp::float32> var_sensordata_out)
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
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        const wp::int32 var_20 = 0;
        bool var_21;
        wp::int32* var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::vec_t<3, wp::float32>* var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::vec_t<3, wp::float32> var_32;
        wp::mat_t<3, 3, wp::float32>* var_33;
        wp::vec_t<3, wp::float32> var_34;
        wp::mat_t<3, 3, wp::float32> var_35;
        wp::vec_t<3, wp::float32>* var_36;
        wp::vec_t<3, wp::float32> var_37;
        wp::vec_t<3, wp::float32> var_38;
        wp::int32* var_39;
        const wp::int32 var_40 = 3;
        wp::int32* var_41;
        wp::int32 var_42;
        wp::int32 var_43;
        bool var_44;
        wp::int32 var_45;
        const wp::int32 var_46 = 3;
        const wp::int32 var_47 = 1;
        wp::int32 var_48;
        wp::int32* var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        wp::int32 var_52;
        wp::quat_t<wp::float32>* var_53;
        wp::quat_t<wp::float32> var_54;
        wp::quat_t<wp::float32> var_55;
        wp::vec_t<3, wp::float32>* var_56;
        wp::vec_t<3, wp::float32> var_57;
        wp::vec_t<3, wp::float32> var_58;
        const wp::float32 var_59 = 0.0;
        const wp::float32 var_60 = 0.0;
        const wp::float32 var_61 = 0.0;
        wp::vec_t<3, wp::float32> var_62;
        const wp::float32 var_63 = 0.0;
        const wp::float32 var_64 = 0.0;
        const wp::float32 var_65 = 0.0;
        wp::vec_t<3, wp::float32> var_66;
        const wp::int32 var_67 = 1;
        wp::int32 var_68;
        wp::vec_t<3, wp::float32>* var_69;
        wp::vec_t<3, wp::float32> var_70;
        wp::vec_t<3, wp::float32> var_71;
        const wp::int32 var_72 = 2;
        wp::int32 var_73;
        wp::vec_t<3, wp::float32>* var_74;
        wp::vec_t<3, wp::float32> var_75;
        wp::vec_t<3, wp::float32> var_76;
        wp::vec_t<3, wp::float32> var_77;
        wp::vec_t<3, wp::float32> var_78;
        const wp::int32 var_79 = 50;
        wp::range_t var_80;
        wp::int32 var_81;
        bool var_82;
        wp::int32* var_83;
        wp::int32 var_84;
        wp::int32 var_85;
        const wp::int32 var_86 = 0;
        bool var_87;
        const wp::int32 var_88 = 0;
        wp::int32 var_89;
        wp::range_t var_90;
        wp::int32 var_91;
        wp::int32* var_92;
        bool var_93;
        wp::int32 var_94;
        const wp::int32 var_95 = 1;
        wp::int32 var_96;
        const wp::int32 var_97 = 1;
        wp::int32 var_98;
        bool var_99;
        wp::int32* var_100;
        wp::int32 var_101;
        wp::int32 var_102;
        wp::vec_t<3, wp::float32>* var_103;
        wp::vec_t<3, wp::float32> var_104;
        wp::vec_t<3, wp::float32> var_105;
        wp::mat_t<3, 3, wp::float32>* var_106;
        wp::mat_t<3, 3, wp::float32> var_107;
        wp::mat_t<3, 3, wp::float32> var_108;
        wp::vec_t<3, wp::float32> var_109;
        wp::int32* var_110;
        wp::int32 var_111;
        wp::int32 var_112;
        wp::int32* var_113;
        wp::int32 var_114;
        wp::int32 var_115;
        wp::shape_t* var_116;
        const wp::int32 var_117 = 0;
        wp::int32 var_118;
        wp::shape_t var_119;
        wp::int32 var_120;
        wp::vec_t<3, wp::float32>* var_121;
        wp::shape_t* var_122;
        const wp::int32 var_123 = 0;
        wp::int32 var_124;
        wp::shape_t var_125;
        wp::int32 var_126;
        wp::int32* var_127;
        wp::vec_t<128, wp::float32> var_128;
        wp::int32 var_129;
        VolumeData_53ac1a2d var_130;
        MeshData_52eaa0fa var_131;
        wp::vec_t<3, wp::float32> var_132;
        wp::int32 var_133;
        wp::float32 var_134;
        const wp::float32 var_135 = 0.0;
        wp::float32 var_136;
        const wp::float32 var_137 = 0.0;
        bool var_138;
        wp::vec_t<6, wp::float32>* var_139;
        wp::int32* var_140;
        wp::vec_t<3, wp::float32>* var_141;
        wp::int32 var_142;
        wp::vec_t<3, wp::float32> var_143;
        wp::vec_t<3, wp::float32> var_144;
        wp::vec_t<3, wp::float32> var_145;
        wp::vec_t<6, wp::float32> var_146;
        wp::vec_t<6, wp::float32>* var_147;
        wp::vec_t<3, wp::float32>* var_148;
        wp::int32* var_149;
        wp::vec_t<3, wp::float32>* var_150;
        wp::int32 var_151;
        wp::vec_t<3, wp::float32> var_152;
        wp::vec_t<3, wp::float32> var_153;
        wp::vec_t<3, wp::float32> var_154;
        wp::vec_t<3, wp::float32> var_155;
        wp::vec_t<6, wp::float32> var_156;
        wp::vec_t<3, wp::float32> var_157;
        const wp::float32 var_158 = 0.05;
        wp::float32 var_159;
        const wp::float32 var_160 = 1e-15;
        wp::float32 var_161;
        wp::float32 var_162;
        wp::vec_t<3, wp::float32> var_163;
        const wp::float32 var_164 = 0.0;
        const wp::float32 var_165 = 0.0;
        const wp::float32 var_166 = 0.0;
        wp::vec_t<3, wp::float32> var_167;
        wp::float32 var_168;
        const wp::int32 var_169 = 0;
        wp::float32 var_170;
        wp::float32 var_171;
        const wp::int32 var_172 = 1;
        wp::float32 var_173;
        wp::float32 var_174;
        const wp::int32 var_175 = 2;
        wp::int32* var_176;
        const wp::int32 var_177 = 3;
        wp::int32 var_178;
        wp::int32 var_179;
        wp::int32* var_180;
        const wp::int32 var_181 = 0;
        wp::int32 var_182;
        wp::int32 var_183;
        wp::int32 var_184;
        wp::int32 var_185;
        const wp::int32 var_186 = 0;
        wp::float32 var_187;
        wp::float32 var_188;
        wp::int32* var_189;
        const wp::int32 var_190 = 1;
        wp::int32 var_191;
        wp::int32 var_192;
        wp::int32 var_193;
        wp::int32 var_194;
        const wp::int32 var_195 = 1;
        wp::float32 var_196;
        wp::float32 var_197;
        wp::int32* var_198;
        const wp::int32 var_199 = 2;
        wp::int32 var_200;
        wp::int32 var_201;
        wp::int32 var_202;
        wp::int32 var_203;
        const wp::int32 var_204 = 2;
        wp::float32 var_205;
        wp::float32 var_206;
        //---------
        // forward
        // def _sensor_tactile(                                                                   <L 2122>
        // worldid, taxelid = wp.tid()                                                            <L 2161>
        builtin_tid2d(var_0, var_1);
        // sensor_id = taxel_sensorid[taxelid]                                                    <L 2163>
        var_2 = wp::address(var_taxel_sensorid, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // mesh_id = sensor_objid[sensor_id]                                                      <L 2164>
        var_5 = wp::address(var_sensor_objid, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // geom_id = sensor_refid[sensor_id]                                                      <L 2165>
        var_8 = wp::address(var_sensor_refid, var_3);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // parent_body = geom_bodyid[geom_id]                                                     <L 2166>
        var_11 = wp::address(var_geom_bodyid, var_9);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // parent_weld = body_weldid[parent_body]                                                 <L 2167>
        var_14 = wp::address(var_body_weldid, var_12);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // geom_count = weld_geom_count_in[worldid, parent_weld]                                  <L 2169>
        var_17 = wp::address(var_weld_geom_count_in, var_0, var_15);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // if geom_count == 0:                                                                    <L 2170>
        var_21 = (var_18 == var_20);
        if (var_21) {
            // return                                                                             <L 2171>
            continue;
        }
        // vertid = taxel_vertadr[taxelid] - mesh_vertadr[mesh_id]                                <L 2174>
        var_22 = wp::address(var_taxel_vertadr, var_1);
        var_23 = wp::address(var_mesh_vertadr, var_6);
        var_25 = wp::load(var_22);
        var_26 = wp::load(var_23);
        var_24 = wp::sub(var_25, var_26);
        // pos = mesh_vert[vertid + mesh_vertadr[mesh_id]]                                        <L 2175>
        var_27 = wp::address(var_mesh_vertadr, var_6);
        var_29 = wp::load(var_27);
        var_28 = wp::add(var_24, var_29);
        var_30 = wp::address(var_mesh_vert, var_28);
        var_32 = wp::load(var_30);
        var_31 = wp::copy(var_32);
        // xpos = geom_xmat_in[worldid, geom_id] @ pos                                            <L 2178>
        var_33 = wp::address(var_geom_xmat_in, var_0, var_9);
        var_35 = wp::load(var_33);
        var_34 = wp::mul(var_35, var_31);
        // xpos += geom_xpos_in[worldid, geom_id]                                                 <L 2179>
        var_36 = wp::address(var_geom_xpos_in, var_0, var_9);
        var_38 = wp::load(var_36);
        var_37 = wp::add(var_34, var_38);
        // has_frame = mesh_normalnum[mesh_id] == 3 * mesh_vertnum[mesh_id]                       <L 2181>
        var_39 = wp::address(var_mesh_normalnum, var_6);
        var_41 = wp::address(var_mesh_vertnum, var_6);
        var_43 = wp::load(var_41);
        var_42 = wp::mul(var_40, var_43);
        var_45 = wp::load(var_39);
        var_44 = (var_45 == var_42);
        // normal_stride = 3 if has_frame else 1                                                  <L 2182>
        if (var_44) {
        }
        if (!var_44) {
        }
        var_48 = wp::where(var_44, var_46, var_47);
        // offset = mesh_normaladr[mesh_id] + normal_stride * vertid                              <L 2183>
        var_49 = wp::address(var_mesh_normaladr, var_6);
        var_50 = wp::mul(var_48, var_24);
        var_52 = wp::load(var_49);
        var_51 = wp::add(var_52, var_50);
        // quat = mesh_quat[mesh_id]                                                              <L 2184>
        var_53 = wp::address(var_mesh_quat, var_6);
        var_55 = wp::load(var_53);
        var_54 = wp::copy(var_55);
        // normal = math.rot_vec_quat(mesh_normal[offset], quat)                                  <L 2185>
        var_56 = wp::address(var_mesh_normal, var_51);
        var_58 = wp::load(var_56);
        var_57 = rot_vec_quat_0(var_58, var_54);
        // tang1 = wp.vec3(0.0, 0.0, 0.0)                                                         <L 2186>
        var_62 = wp::vec_t<3, wp::float32>(var_59, var_60, var_61);
        // tang2 = wp.vec3(0.0, 0.0, 0.0)                                                         <L 2187>
        var_66 = wp::vec_t<3, wp::float32>(var_63, var_64, var_65);
        // if has_frame:                                                                          <L 2188>
        if (var_44) {
            // tang1 = math.rot_vec_quat(mesh_normal[offset + 1], quat)                           <L 2189>
            var_68 = wp::add(var_51, var_67);
            var_69 = wp::address(var_mesh_normal, var_68);
            var_71 = wp::load(var_69);
            var_70 = rot_vec_quat_0(var_71, var_54);
            // tang2 = math.rot_vec_quat(mesh_normal[offset + 2], quat)                           <L 2190>
            var_73 = wp::add(var_51, var_72);
            var_74 = wp::address(var_mesh_normal, var_73);
            var_76 = wp::load(var_74);
            var_75 = rot_vec_quat_0(var_76, var_54);
        }
        var_77 = wp::where(var_44, var_70, var_62);
        var_78 = wp::where(var_44, var_75, var_66);
        // for g in range(MJ_MAXCONPAIR):                                                         <L 2192>
        var_80 = wp::range(var_79);
        start_for_1:;
            if (iter_cmp(var_80) == 0) goto end_for_1;
            var_81 = wp::iter_next(var_80);
            // if g >= geom_count:                                                                <L 2193>
            var_82 = (var_81 >= var_18);
            if (var_82) {
                // break                                                                          <L 2194>
                goto end_for_1;
            }
            // geom = weld_geom_list_in[worldid, parent_weld, g]                                  <L 2196>
            var_83 = wp::address(var_weld_geom_list_in, var_0, var_15, var_81);
            var_85 = wp::load(var_83);
            var_84 = wp::copy(var_85);
            // if geom < 0:                                                                       <L 2197>
            var_87 = (var_84 < var_86);
            if (var_87) {
                // continue                                                                       <L 2198>
                goto start_for_1;
            }
            // is_dup = int(0)                                                                    <L 2200>
            var_89 = wp::int(var_88);
            // for j in range(g):                                                                 <L 2201>
            var_90 = wp::range(var_81);
            start_for_3:;
                if (iter_cmp(var_90) == 0) goto end_for_3;
                var_91 = wp::iter_next(var_90);
                // if weld_geom_list_in[worldid, parent_weld, j] == geom:                         <L 2202>
                var_92 = wp::address(var_weld_geom_list_in, var_0, var_15, var_91);
                var_94 = wp::load(var_92);
                var_93 = (var_94 == var_84);
                if (var_93) {
                    // is_dup = int(1)                                                            <L 2203>
                    var_96 = wp::int(var_95);
                    // break                                                                      <L 2204>
                    wp::assign(var_89, var_96);
                    goto end_for_3;
                }
                goto start_for_3;
            end_for_3:;
            // if is_dup == int(1):                                                               <L 2205>
            var_98 = wp::int(var_97);
            var_99 = (var_89 == var_98);
            if (var_99) {
                // continue                                                                       <L 2206>
                goto start_for_1;
            }
            // body = geom_bodyid[geom]                                                           <L 2208>
            var_100 = wp::address(var_geom_bodyid, var_84);
            var_102 = wp::load(var_100);
            var_101 = wp::copy(var_102);
            // tmp = xpos - geom_xpos_in[worldid, geom]                                           <L 2210>
            var_103 = wp::address(var_geom_xpos_in, var_0, var_84);
            var_105 = wp::load(var_103);
            var_104 = wp::sub(var_37, var_105);
            // lpos = wp.transpose(geom_xmat_in[worldid, geom]) @ tmp                             <L 2211>
            var_106 = wp::address(var_geom_xmat_in, var_0, var_84);
            var_108 = wp::load(var_106);
            var_107 = wp::transpose(var_108);
            var_109 = wp::mul(var_107, var_104);
            // plugin_id = geom_plugin_index[geom]                                                <L 2213>
            var_110 = wp::address(var_geom_plugin_index, var_84);
            var_112 = wp::load(var_110);
            var_111 = wp::copy(var_112);
            // contact_type = geom_type[geom]                                                     <L 2214>
            var_113 = wp::address(var_geom_type, var_84);
            var_115 = wp::load(var_113);
            var_114 = wp::copy(var_115);
            // plugin_attributes, plugin_index, volume_data, mesh_data = get_sdf_params(          <L 2216>
            // oct_child,                                                                         <L 2217>
            // oct_aabb,                                                                          <L 2218>
            // oct_coeff,                                                                         <L 2219>
            // mesh_octadr,                                                                       <L 2220>
            // plugin,                                                                            <L 2221>
            // plugin_attr,                                                                       <L 2222>
            // contact_type,                                                                      <L 2223>
            // geom_size[worldid % geom_size.shape[0], geom],                                     <L 2224>
            var_116 = &(var_geom_size.shape);
            var_119 = wp::load(var_116);
            var_118 = wp::extract(var_119, var_117);
            var_120 = wp::mod(var_0, var_118);
            var_121 = wp::address(var_geom_size, var_120, var_84);
            // plugin_id,                                                                         <L 2225>
            // geom_dataid[worldid % geom_dataid.shape[0], geom],                                 <L 2226>
            var_122 = &(var_geom_dataid.shape);
            var_125 = wp::load(var_122);
            var_124 = wp::extract(var_125, var_123);
            var_126 = wp::mod(var_0, var_124);
            var_127 = wp::address(var_geom_dataid, var_126, var_84);
            var_132 = wp::load(var_121);
            var_133 = wp::load(var_127);
            get_sdf_params_0(var_oct_child, var_oct_aabb, var_oct_coeff, var_mesh_octadr, var_plugin, var_plugin_attr, var_114, var_132, var_111, var_133, var_128, var_129, var_130, var_131);
            // depth = wp.min(sdf(contact_type, lpos, plugin_attributes, plugin_index, volume_data, mesh_data), 0.0)       <L 2229>
            var_134 = sdf_0(var_114, var_109, var_128, var_129, var_130, var_131);
            var_136 = wp::min(var_134, var_135);
            // if depth >= 0.0:                                                                   <L 2230>
            var_138 = (var_136 >= var_137);
            if (var_138) {
                // continue                                                                       <L 2231>
                goto start_for_1;
            }
            // vel_sensor = _transform_spatial(cvel_in[worldid, parent_weld], xpos - subtree_com_in[worldid, body_rootid[parent_weld]])       <L 2233>
            var_139 = wp::address(var_cvel_in, var_0, var_15);
            var_140 = wp::address(var_body_rootid, var_15);
            var_142 = wp::load(var_140);
            var_141 = wp::address(var_subtree_com_in, var_0, var_142);
            var_144 = wp::load(var_141);
            var_143 = wp::sub(var_37, var_144);
            var_146 = wp::load(var_139);
            var_145 = _transform_spatial_0(var_146, var_143);
            // vel_other = _transform_spatial(                                                    <L 2234>
            // cvel_in[worldid, body], geom_xpos_in[worldid, geom] - subtree_com_in[worldid, body_rootid[body]]       <L 2235>
            var_147 = wp::address(var_cvel_in, var_0, var_101);
            var_148 = wp::address(var_geom_xpos_in, var_0, var_84);
            var_149 = wp::address(var_body_rootid, var_101);
            var_151 = wp::load(var_149);
            var_150 = wp::address(var_subtree_com_in, var_0, var_151);
            var_153 = wp::load(var_148);
            var_154 = wp::load(var_150);
            var_152 = wp::sub(var_153, var_154);
            var_156 = wp::load(var_147);
            var_155 = _transform_spatial_0(var_156, var_152);
            // vel_rel = vel_sensor - vel_other                                                   <L 2237>
            var_157 = wp::sub(var_145, var_155);
            // kMaxDepth = 0.05                                                                   <L 2239>
            // pressure = depth / wp.max(kMaxDepth - depth, MJ_MINVAL)                            <L 2240>
            var_159 = wp::sub(var_158, var_136);
            var_161 = wp::max(var_159, var_160);
            var_162 = wp::div(var_136, var_161);
            // force = wp.mul(normal, pressure)                                                   <L 2241>
            var_163 = wp::mul(var_57, var_162);
            // forceT = wp.vec3(0.0, 0.0, 0.0)                                                    <L 2243>
            var_167 = wp::vec_t<3, wp::float32>(var_164, var_165, var_166);
            // forceT[0] = wp.dot(force, normal)                                                  <L 2244>
            var_168 = wp::dot(var_163, var_57);
            wp::assign_inplace(var_167, var_169, var_168);
            // if has_frame:                                                                      <L 2245>
            if (var_44) {
                // forceT[1] = wp.abs(wp.dot(vel_rel, tang1))                                     <L 2246>
                var_170 = wp::dot(var_157, var_77);
                var_171 = wp::abs(var_170);
                wp::assign_inplace(var_167, var_172, var_171);
                // forceT[2] = wp.abs(wp.dot(vel_rel, tang2))                                     <L 2247>
                var_173 = wp::dot(var_157, var_78);
                var_174 = wp::abs(var_173);
                wp::assign_inplace(var_167, var_175, var_174);
            }
            // dim = sensor_dim[sensor_id] // 3                                                   <L 2249>
            var_176 = wp::address(var_sensor_dim, var_3);
            var_179 = wp::load(var_176);
            var_178 = wp::floordiv(var_179, var_177);
            // wp.atomic_add(sensordata_out, worldid, sensor_adr[sensor_id] + 0 * dim + vertid, forceT[0])       <L 2250>
            var_180 = wp::address(var_sensor_adr, var_3);
            var_182 = wp::mul(var_181, var_178);
            var_184 = wp::load(var_180);
            var_183 = wp::add(var_184, var_182);
            var_185 = wp::add(var_183, var_24);
            var_187 = wp::extract(var_167, var_186);
            var_188 = wp::atomic_add(var_sensordata_out, var_0, var_185, var_187);
            // wp.atomic_add(sensordata_out, worldid, sensor_adr[sensor_id] + 1 * dim + vertid, forceT[1])       <L 2251>
            var_189 = wp::address(var_sensor_adr, var_3);
            var_191 = wp::mul(var_190, var_178);
            var_193 = wp::load(var_189);
            var_192 = wp::add(var_193, var_191);
            var_194 = wp::add(var_192, var_24);
            var_196 = wp::extract(var_167, var_195);
            var_197 = wp::atomic_add(var_sensordata_out, var_0, var_194, var_196);
            // wp.atomic_add(sensordata_out, worldid, sensor_adr[sensor_id] + 2 * dim + vertid, forceT[2])       <L 2252>
            var_198 = wp::address(var_sensor_adr, var_3);
            var_200 = wp::mul(var_199, var_178);
            var_202 = wp::load(var_198);
            var_201 = wp::add(var_202, var_200);
            var_203 = wp::add(var_201, var_24);
            var_205 = wp::extract(var_167, var_204);
            var_206 = wp::atomic_add(var_sensordata_out, var_0, var_203, var_205);
            goto start_for_1;
        end_for_1:;
    }
}



extern "C" __global__ void _energy_pos_passive_joint_2dc3513e_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_qpos_spring,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_jnt_qposadr,
    wp::array_t<wp::float32> var_jnt_stiffness,
    wp::array_t<wp::float32> var_qpos_in,
    wp::array_t<wp::vec_t<2, wp::float32>> var_energy_out)
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
        wp::int32 var_16;
        wp::int32 var_17;
        wp::shape_t* var_18;
        const wp::int32 var_19 = 0;
        wp::int32 var_20;
        wp::shape_t var_21;
        wp::int32 var_22;
        const wp::int32 var_23 = 0;
        bool var_24;
        const wp::int32 var_25 = 0;
        wp::int32 var_26;
        wp::float32* var_27;
        const wp::int32 var_28 = 0;
        wp::int32 var_29;
        wp::float32* var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        const wp::int32 var_34 = 1;
        wp::int32 var_35;
        wp::float32* var_36;
        const wp::int32 var_37 = 1;
        wp::int32 var_38;
        wp::float32* var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        const wp::int32 var_43 = 2;
        wp::int32 var_44;
        wp::float32* var_45;
        const wp::int32 var_46 = 2;
        wp::int32 var_47;
        wp::float32* var_48;
        wp::float32 var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        wp::vec_t<3, wp::float32> var_52;
        const wp::int32 var_53 = 3;
        wp::int32 var_54;
        wp::float32* var_55;
        const wp::int32 var_56 = 4;
        wp::int32 var_57;
        wp::float32* var_58;
        const wp::int32 var_59 = 5;
        wp::int32 var_60;
        wp::float32* var_61;
        const wp::int32 var_62 = 6;
        wp::int32 var_63;
        wp::float32* var_64;
        wp::quat_t<wp::float32> var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        wp::float32 var_68;
        wp::float32 var_69;
        wp::quat_t<wp::float32> var_70;
        const wp::int32 var_71 = 3;
        wp::int32 var_72;
        wp::float32* var_73;
        const wp::int32 var_74 = 4;
        wp::int32 var_75;
        wp::float32* var_76;
        const wp::int32 var_77 = 5;
        wp::int32 var_78;
        wp::float32* var_79;
        const wp::int32 var_80 = 6;
        wp::int32 var_81;
        wp::float32* var_82;
        wp::quat_t<wp::float32> var_83;
        wp::float32 var_84;
        wp::float32 var_85;
        wp::float32 var_86;
        wp::float32 var_87;
        wp::vec_t<3, wp::float32> var_88;
        const wp::float32 var_89 = 0.5;
        wp::float32 var_90;
        wp::float32 var_91;
        wp::float32 var_92;
        wp::float32 var_93;
        wp::float32 var_94;
        const wp::float32 var_95 = 0.0;
        wp::vec_t<2, wp::float32> var_96;
        wp::vec_t<2, wp::float32> var_97;
        const wp::int32 var_98 = 1;
        bool var_99;
        const wp::int32 var_100 = 0;
        wp::int32 var_101;
        wp::float32* var_102;
        const wp::int32 var_103 = 1;
        wp::int32 var_104;
        wp::float32* var_105;
        const wp::int32 var_106 = 2;
        wp::int32 var_107;
        wp::float32* var_108;
        const wp::int32 var_109 = 3;
        wp::int32 var_110;
        wp::float32* var_111;
        wp::quat_t<wp::float32> var_112;
        wp::float32 var_113;
        wp::float32 var_114;
        wp::float32 var_115;
        wp::float32 var_116;
        wp::quat_t<wp::float32> var_117;
        const wp::int32 var_118 = 0;
        wp::int32 var_119;
        wp::float32* var_120;
        const wp::int32 var_121 = 1;
        wp::int32 var_122;
        wp::float32* var_123;
        const wp::int32 var_124 = 2;
        wp::int32 var_125;
        wp::float32* var_126;
        const wp::int32 var_127 = 3;
        wp::int32 var_128;
        wp::float32* var_129;
        wp::quat_t<wp::float32> var_130;
        wp::float32 var_131;
        wp::float32 var_132;
        wp::float32 var_133;
        wp::float32 var_134;
        wp::vec_t<3, wp::float32> var_135;
        const wp::float32 var_136 = 0.5;
        wp::float32 var_137;
        wp::float32 var_138;
        wp::float32 var_139;
        const wp::float32 var_140 = 0.0;
        wp::vec_t<2, wp::float32> var_141;
        wp::vec_t<2, wp::float32> var_142;
        wp::quat_t<wp::float32> var_143;
        wp::vec_t<2, wp::float32> var_144;
        const wp::int32 var_145 = 2;
        bool var_146;
        const wp::int32 var_147 = 3;
        bool var_148;
        bool var_149;
        wp::float32* var_150;
        wp::float32* var_151;
        wp::float32 var_152;
        wp::float32 var_153;
        wp::float32 var_154;
        const wp::float32 var_155 = 0.5;
        wp::float32 var_156;
        wp::float32 var_157;
        wp::float32 var_158;
        const wp::float32 var_159 = 0.0;
        wp::vec_t<2, wp::float32> var_160;
        wp::vec_t<2, wp::float32> var_161;
        wp::vec_t<2, wp::float32> var_162;
        wp::vec_t<2, wp::float32> var_163;
        wp::quat_t<wp::float32> var_164;
        wp::vec_t<2, wp::float32> var_165;
        //---------
        // forward
        // def _energy_pos_passive_joint(                                                         <L 2732>
        // worldid, jntid = wp.tid()                                                              <L 2743>
        builtin_tid2d(var_0, var_1);
        // jnt_stiffness_id = worldid % jnt_stiffness.shape[0]                                    <L 2744>
        var_2 = &(var_jnt_stiffness.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // stiffness = jnt_stiffness[jnt_stiffness_id, jntid]                                     <L 2745>
        var_7 = wp::address(var_jnt_stiffness, var_6, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // if stiffness == 0.0:                                                                   <L 2747>
        var_11 = (var_8 == var_10);
        if (var_11) {
            // return                                                                             <L 2748>
            continue;
        }
        // padr = jnt_qposadr[jntid]                                                              <L 2750>
        var_12 = wp::address(var_jnt_qposadr, var_1);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // jnttype = jnt_type[jntid]                                                              <L 2751>
        var_15 = wp::address(var_jnt_type, var_1);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // qpos_spring_id = worldid % qpos_spring.shape[0]                                        <L 2752>
        var_18 = &(var_qpos_spring.shape);
        var_21 = wp::load(var_18);
        var_20 = wp::extract(var_21, var_19);
        var_22 = wp::mod(var_0, var_20);
        // if jnttype == JointType.FREE:                                                          <L 2754>
        var_24 = (var_16 == var_23);
        if (var_24) {
            // dif0 = wp.vec3(                                                                    <L 2755>
            // qpos_in[worldid, padr + 0] - qpos_spring[qpos_spring_id, padr + 0],                <L 2756>
            var_26 = wp::add(var_13, var_25);
            var_27 = wp::address(var_qpos_in, var_0, var_26);
            var_29 = wp::add(var_13, var_28);
            var_30 = wp::address(var_qpos_spring, var_22, var_29);
            var_32 = wp::load(var_27);
            var_33 = wp::load(var_30);
            var_31 = wp::sub(var_32, var_33);
            // qpos_in[worldid, padr + 1] - qpos_spring[qpos_spring_id, padr + 1],                <L 2757>
            var_35 = wp::add(var_13, var_34);
            var_36 = wp::address(var_qpos_in, var_0, var_35);
            var_38 = wp::add(var_13, var_37);
            var_39 = wp::address(var_qpos_spring, var_22, var_38);
            var_41 = wp::load(var_36);
            var_42 = wp::load(var_39);
            var_40 = wp::sub(var_41, var_42);
            // qpos_in[worldid, padr + 2] - qpos_spring[qpos_spring_id, padr + 2],                <L 2758>
            var_44 = wp::add(var_13, var_43);
            var_45 = wp::address(var_qpos_in, var_0, var_44);
            var_47 = wp::add(var_13, var_46);
            var_48 = wp::address(var_qpos_spring, var_22, var_47);
            var_50 = wp::load(var_45);
            var_51 = wp::load(var_48);
            var_49 = wp::sub(var_50, var_51);
            var_52 = wp::vec_t<3, wp::float32>(var_31, var_40, var_49);
            // quat1 = wp.quat(                                                                   <L 2762>
            // qpos_in[worldid, padr + 3],                                                        <L 2763>
            var_54 = wp::add(var_13, var_53);
            var_55 = wp::address(var_qpos_in, var_0, var_54);
            // qpos_in[worldid, padr + 4],                                                        <L 2764>
            var_57 = wp::add(var_13, var_56);
            var_58 = wp::address(var_qpos_in, var_0, var_57);
            // qpos_in[worldid, padr + 5],                                                        <L 2765>
            var_60 = wp::add(var_13, var_59);
            var_61 = wp::address(var_qpos_in, var_0, var_60);
            // qpos_in[worldid, padr + 6],                                                        <L 2766>
            var_63 = wp::add(var_13, var_62);
            var_64 = wp::address(var_qpos_in, var_0, var_63);
            var_66 = wp::load(var_55);
            var_67 = wp::load(var_58);
            var_68 = wp::load(var_61);
            var_69 = wp::load(var_64);
            var_65 = wp::quat_t<wp::float32>(var_66, var_67, var_68, var_69);
            // quat1 = wp.normalize(quat1)                                                        <L 2768>
            var_70 = wp::normalize(var_65);
            // quat_spring = wp.quat(                                                             <L 2770>
            // qpos_spring[qpos_spring_id, padr + 3],                                             <L 2771>
            var_72 = wp::add(var_13, var_71);
            var_73 = wp::address(var_qpos_spring, var_22, var_72);
            // qpos_spring[qpos_spring_id, padr + 4],                                             <L 2772>
            var_75 = wp::add(var_13, var_74);
            var_76 = wp::address(var_qpos_spring, var_22, var_75);
            // qpos_spring[qpos_spring_id, padr + 5],                                             <L 2773>
            var_78 = wp::add(var_13, var_77);
            var_79 = wp::address(var_qpos_spring, var_22, var_78);
            // qpos_spring[qpos_spring_id, padr + 6],                                             <L 2774>
            var_81 = wp::add(var_13, var_80);
            var_82 = wp::address(var_qpos_spring, var_22, var_81);
            var_84 = wp::load(var_73);
            var_85 = wp::load(var_76);
            var_86 = wp::load(var_79);
            var_87 = wp::load(var_82);
            var_83 = wp::quat_t<wp::float32>(var_84, var_85, var_86, var_87);
            // dif1 = math.quat_sub(quat1, quat_spring)                                           <L 2777>
            var_88 = quat_sub_0(var_70, var_83);
            // energy = wp.vec2(                                                                  <L 2779>
            // 0.5 * stiffness * (wp.dot(dif0, dif0) + wp.dot(dif1, dif1)),                       <L 2780>
            var_90 = wp::mul(var_89, var_8);
            var_91 = wp::dot(var_52, var_52);
            var_92 = wp::dot(var_88, var_88);
            var_93 = wp::add(var_91, var_92);
            var_94 = wp::mul(var_90, var_93);
            // 0.0,                                                                               <L 2781>
            var_96 = wp::vec_t<2, wp::float32>(var_94, var_95);
            // wp.atomic_add(energy_out, worldid, energy)                                         <L 2784>
            var_97 = wp::atomic_add(var_energy_out, var_0, var_96);
        }
        if (!var_24) {
            // elif jnttype == JointType.BALL:                                                    <L 2786>
            var_99 = (var_16 == var_98);
            if (var_99) {
                // quat = wp.quat(                                                                <L 2787>
                // qpos_in[worldid, padr + 0],                                                    <L 2788>
                var_101 = wp::add(var_13, var_100);
                var_102 = wp::address(var_qpos_in, var_0, var_101);
                // qpos_in[worldid, padr + 1],                                                    <L 2789>
                var_104 = wp::add(var_13, var_103);
                var_105 = wp::address(var_qpos_in, var_0, var_104);
                // qpos_in[worldid, padr + 2],                                                    <L 2790>
                var_107 = wp::add(var_13, var_106);
                var_108 = wp::address(var_qpos_in, var_0, var_107);
                // qpos_in[worldid, padr + 3],                                                    <L 2791>
                var_110 = wp::add(var_13, var_109);
                var_111 = wp::address(var_qpos_in, var_0, var_110);
                var_113 = wp::load(var_102);
                var_114 = wp::load(var_105);
                var_115 = wp::load(var_108);
                var_116 = wp::load(var_111);
                var_112 = wp::quat_t<wp::float32>(var_113, var_114, var_115, var_116);
                // quat = wp.normalize(quat)                                                      <L 2793>
                var_117 = wp::normalize(var_112);
                // quat_spring = wp.quat(                                                         <L 2795>
                // qpos_spring[qpos_spring_id, padr + 0],                                         <L 2796>
                var_119 = wp::add(var_13, var_118);
                var_120 = wp::address(var_qpos_spring, var_22, var_119);
                // qpos_spring[qpos_spring_id, padr + 1],                                         <L 2797>
                var_122 = wp::add(var_13, var_121);
                var_123 = wp::address(var_qpos_spring, var_22, var_122);
                // qpos_spring[qpos_spring_id, padr + 2],                                         <L 2798>
                var_125 = wp::add(var_13, var_124);
                var_126 = wp::address(var_qpos_spring, var_22, var_125);
                // qpos_spring[qpos_spring_id, padr + 3],                                         <L 2799>
                var_128 = wp::add(var_13, var_127);
                var_129 = wp::address(var_qpos_spring, var_22, var_128);
                var_131 = wp::load(var_120);
                var_132 = wp::load(var_123);
                var_133 = wp::load(var_126);
                var_134 = wp::load(var_129);
                var_130 = wp::quat_t<wp::float32>(var_131, var_132, var_133, var_134);
                // dif = math.quat_sub(quat, quat_spring)                                         <L 2802>
                var_135 = quat_sub_0(var_117, var_130);
                // energy = wp.vec2(                                                              <L 2803>
                // 0.5 * stiffness * wp.dot(dif, dif),                                            <L 2804>
                var_137 = wp::mul(var_136, var_8);
                var_138 = wp::dot(var_135, var_135);
                var_139 = wp::mul(var_137, var_138);
                // 0.0,                                                                           <L 2805>
                var_141 = wp::vec_t<2, wp::float32>(var_139, var_140);
                // wp.atomic_add(energy_out, worldid, energy)                                     <L 2807>
                var_142 = wp::atomic_add(var_energy_out, var_0, var_141);
            }
            var_143 = wp::where(var_99, var_130, var_83);
            var_144 = wp::where(var_99, var_141, var_96);
            if (!var_99) {
                // elif jnttype == JointType.SLIDE or jnttype == JointType.HINGE:                 <L 2808>
                var_146 = (var_16 == var_145);
                var_148 = (var_16 == var_147);
                var_149 = var_146 || var_148;
                if (var_149) {
                    // dif_ = qpos_in[worldid, padr] - qpos_spring[qpos_spring_id, padr]          <L 2809>
                    var_150 = wp::address(var_qpos_in, var_0, var_13);
                    var_151 = wp::address(var_qpos_spring, var_22, var_13);
                    var_153 = wp::load(var_150);
                    var_154 = wp::load(var_151);
                    var_152 = wp::sub(var_153, var_154);
                    // energy = wp.vec2(                                                          <L 2810>
                    // 0.5 * stiffness * dif_ * dif_,                                             <L 2811>
                    var_156 = wp::mul(var_155, var_8);
                    var_157 = wp::mul(var_156, var_152);
                    var_158 = wp::mul(var_157, var_152);
                    // 0.0,                                                                       <L 2812>
                    var_160 = wp::vec_t<2, wp::float32>(var_158, var_159);
                    // wp.atomic_add(energy_out, worldid, energy)                                 <L 2814>
                    var_161 = wp::atomic_add(var_energy_out, var_0, var_160);
                }
                var_162 = wp::where(var_149, var_160, var_144);
            }
            var_163 = wp::where(var_99, var_144, var_162);
        }
        var_164 = wp::where(var_24, var_83, var_143);
        var_165 = wp::where(var_24, var_96, var_163);
    }
}



extern "C" __global__ void _limit_pos_b1ff6d76_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_objid,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::array_t<wp::int32> var_sensor_limitpos_adr,
    wp::array_t<wp::int32> var_ne_in,
    wp::array_t<wp::int32> var_nf_in,
    wp::array_t<wp::int32> var_nl_in,
    wp::array_t<wp::int32> var_efc_type_in,
    wp::array_t<wp::int32> var_efc_id_in,
    wp::array_t<wp::float32> var_efc_pos_in,
    wp::array_t<wp::float32> var_efc_margin_in,
    wp::array_t<wp::float32> var_sensordata_out)
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
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        bool var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        bool var_16;
        bool var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32* var_21;
        wp::int32* var_22;
        bool var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        const wp::int32 var_29 = 3;
        bool var_30;
        const wp::int32 var_31 = 4;
        bool var_32;
        bool var_33;
        wp::float32* var_34;
        wp::float32* var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        wp::float32 var_38;
        wp::slice_t var_39;
        const wp::int32 var_40 = 0;
        wp::array_t<wp::float32> var_41;
        //---------
        // forward
        // def _limit_pos(                                                                        <L 244>
        // worldid, efcid, limitposid = wp.tid()                                                  <L 263>
        builtin_tid3d(var_0, var_1, var_2);
        // ne = ne_in[worldid]                                                                    <L 265>
        var_3 = wp::address(var_ne_in, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // nf = nf_in[worldid]                                                                    <L 266>
        var_6 = wp::address(var_nf_in, var_0);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // nl = nl_in[worldid]                                                                    <L 267>
        var_9 = wp::address(var_nl_in, var_0);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // if efcid < ne + nf or efcid >= ne + nf + nl:                                           <L 270>
        var_12 = wp::add(var_4, var_7);
        var_13 = (var_1 < var_12);
        var_14 = wp::add(var_4, var_7);
        var_15 = wp::add(var_14, var_10);
        var_16 = (var_1 >= var_15);
        var_17 = var_13 || var_16;
        if (var_17) {
            // return                                                                             <L 271>
            continue;
        }
        // sensorid = sensor_limitpos_adr[limitposid]                                             <L 273>
        var_18 = wp::address(var_sensor_limitpos_adr, var_2);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // if efc_id_in[worldid, efcid] == sensor_objid[sensorid]:                                <L 274>
        var_21 = wp::address(var_efc_id_in, var_0, var_1);
        var_22 = wp::address(var_sensor_objid, var_19);
        var_24 = wp::load(var_21);
        var_25 = wp::load(var_22);
        var_23 = (var_24 == var_25);
        if (var_23) {
            // efc_type = efc_type_in[worldid, efcid]                                             <L 275>
            var_26 = wp::address(var_efc_type_in, var_0, var_1);
            var_28 = wp::load(var_26);
            var_27 = wp::copy(var_28);
            // if efc_type == ConstraintType.LIMIT_JOINT or efc_type == ConstraintType.LIMIT_TENDON:       <L 276>
            var_30 = (var_27 == var_29);
            var_32 = (var_27 == var_31);
            var_33 = var_30 || var_32;
            if (var_33) {
                // val = efc_pos_in[worldid, efcid] - efc_margin_in[worldid, efcid]               <L 277>
                var_34 = wp::address(var_efc_pos_in, var_0, var_1);
                var_35 = wp::address(var_efc_margin_in, var_0, var_1);
                var_37 = wp::load(var_34);
                var_38 = wp::load(var_35);
                var_36 = wp::sub(var_37, var_38);
                // _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, sensordata_out[worldid])       <L 278>
                var_39 = wp::slice_t(var_0, var_0, var_40);
                var_41 = wp::view(var_sensordata_out, var_39);
                _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_19, var_36, var_41);
            }
        }
    }
}



extern "C" __global__ void _limit_vel_ca76d496_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_objid,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::array_t<wp::int32> var_sensor_limitvel_adr,
    wp::array_t<wp::int32> var_ne_in,
    wp::array_t<wp::int32> var_nf_in,
    wp::array_t<wp::int32> var_nl_in,
    wp::array_t<wp::int32> var_efc_type_in,
    wp::array_t<wp::int32> var_efc_id_in,
    wp::array_t<wp::float32> var_efc_vel_in,
    wp::array_t<wp::float32> var_sensordata_out)
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
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        bool var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        bool var_16;
        bool var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32* var_21;
        wp::int32* var_22;
        bool var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        const wp::int32 var_29 = 3;
        bool var_30;
        const wp::int32 var_31 = 4;
        bool var_32;
        bool var_33;
        wp::float32* var_34;
        wp::slice_t var_35;
        const wp::int32 var_36 = 0;
        wp::array_t<wp::float32> var_37;
        wp::float32 var_38;
        //---------
        // forward
        // def _limit_vel(                                                                        <L 973>
        // worldid, efcid, limitvelid = wp.tid()                                                  <L 991>
        builtin_tid3d(var_0, var_1, var_2);
        // ne = ne_in[worldid]                                                                    <L 993>
        var_3 = wp::address(var_ne_in, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // nf = nf_in[worldid]                                                                    <L 994>
        var_6 = wp::address(var_nf_in, var_0);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // nl = nl_in[worldid]                                                                    <L 995>
        var_9 = wp::address(var_nl_in, var_0);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // if efcid < ne + nf or efcid >= ne + nf + nl:                                           <L 998>
        var_12 = wp::add(var_4, var_7);
        var_13 = (var_1 < var_12);
        var_14 = wp::add(var_4, var_7);
        var_15 = wp::add(var_14, var_10);
        var_16 = (var_1 >= var_15);
        var_17 = var_13 || var_16;
        if (var_17) {
            // return                                                                             <L 999>
            continue;
        }
        // sensorid = sensor_limitvel_adr[limitvelid]                                             <L 1001>
        var_18 = wp::address(var_sensor_limitvel_adr, var_2);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // if efc_id_in[worldid, efcid] == sensor_objid[sensorid]:                                <L 1002>
        var_21 = wp::address(var_efc_id_in, var_0, var_1);
        var_22 = wp::address(var_sensor_objid, var_19);
        var_24 = wp::load(var_21);
        var_25 = wp::load(var_22);
        var_23 = (var_24 == var_25);
        if (var_23) {
            // efc_type = efc_type_in[worldid, efcid]                                             <L 1003>
            var_26 = wp::address(var_efc_type_in, var_0, var_1);
            var_28 = wp::load(var_26);
            var_27 = wp::copy(var_28);
            // if efc_type == ConstraintType.LIMIT_JOINT or efc_type == ConstraintType.LIMIT_TENDON:       <L 1004>
            var_30 = (var_27 == var_29);
            var_32 = (var_27 == var_31);
            var_33 = var_30 || var_32;
            if (var_33) {
                // _write_scalar(                                                                 <L 1005>
                // sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, efc_vel_in[worldid, efcid], sensordata_out[worldid]       <L 1006>
                var_34 = wp::address(var_efc_vel_in, var_0, var_1);
                var_35 = wp::slice_t(var_0, var_0, var_36);
                var_37 = wp::view(var_sensordata_out, var_35);
                var_38 = wp::load(var_34);
                _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_19, var_38, var_37);
            }
        }
    }
}



extern "C" __global__ void _limit_frc_be4f0b30_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_objid,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::array_t<wp::int32> var_sensor_limitfrc_adr,
    wp::array_t<wp::int32> var_ne_in,
    wp::array_t<wp::int32> var_nf_in,
    wp::array_t<wp::int32> var_nl_in,
    wp::array_t<wp::int32> var_efc_type_in,
    wp::array_t<wp::int32> var_efc_id_in,
    wp::array_t<wp::float32> var_efc_force_in,
    wp::array_t<wp::float32> var_sensordata_out)
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
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        bool var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        bool var_16;
        bool var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32* var_21;
        wp::int32* var_22;
        bool var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        const wp::int32 var_29 = 3;
        bool var_30;
        const wp::int32 var_31 = 4;
        bool var_32;
        bool var_33;
        wp::float32* var_34;
        wp::slice_t var_35;
        const wp::int32 var_36 = 0;
        wp::array_t<wp::float32> var_37;
        wp::float32 var_38;
        //---------
        // forward
        // def _limit_frc(                                                                        <L 1581>
        // worldid, efcid, limitfrcid = wp.tid()                                                  <L 1599>
        builtin_tid3d(var_0, var_1, var_2);
        // ne = ne_in[worldid]                                                                    <L 1601>
        var_3 = wp::address(var_ne_in, var_0);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // nf = nf_in[worldid]                                                                    <L 1602>
        var_6 = wp::address(var_nf_in, var_0);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // nl = nl_in[worldid]                                                                    <L 1603>
        var_9 = wp::address(var_nl_in, var_0);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // if efcid < ne + nf or efcid >= ne + nf + nl:                                           <L 1606>
        var_12 = wp::add(var_4, var_7);
        var_13 = (var_1 < var_12);
        var_14 = wp::add(var_4, var_7);
        var_15 = wp::add(var_14, var_10);
        var_16 = (var_1 >= var_15);
        var_17 = var_13 || var_16;
        if (var_17) {
            // return                                                                             <L 1607>
            continue;
        }
        // sensorid = sensor_limitfrc_adr[limitfrcid]                                             <L 1609>
        var_18 = wp::address(var_sensor_limitfrc_adr, var_2);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // if efc_id_in[worldid, efcid] == sensor_objid[sensorid]:                                <L 1610>
        var_21 = wp::address(var_efc_id_in, var_0, var_1);
        var_22 = wp::address(var_sensor_objid, var_19);
        var_24 = wp::load(var_21);
        var_25 = wp::load(var_22);
        var_23 = (var_24 == var_25);
        if (var_23) {
            // efc_type = efc_type_in[worldid, efcid]                                             <L 1611>
            var_26 = wp::address(var_efc_type_in, var_0, var_1);
            var_28 = wp::load(var_26);
            var_27 = wp::copy(var_28);
            // if efc_type == ConstraintType.LIMIT_JOINT or efc_type == ConstraintType.LIMIT_TENDON:       <L 1612>
            var_30 = (var_27 == var_29);
            var_32 = (var_27 == var_31);
            var_33 = var_30 || var_32;
            if (var_33) {
                // _write_scalar(                                                                 <L 1613>
                // sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, efc_force_in[worldid, efcid], sensordata_out[worldid]       <L 1614>
                var_34 = wp::address(var_efc_force_in, var_0, var_1);
                var_35 = wp::slice_t(var_0, var_0, var_36);
                var_37 = wp::view(var_sensordata_out, var_35);
                var_38 = wp::load(var_34);
                _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_19, var_38, var_37);
            }
        }
    }
}



extern "C" __global__ void _sensor_touch_5d4feeec_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_opt_cone,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_site_type,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_size,
    wp::array_t<wp::int32> var_sensor_objid,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::int32> var_sensor_touch_adr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::int32> var_contact_worldid_in,
    wp::array_t<wp::float32> var_efc_force_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::float32> var_sensordata_out)
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
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::vec_t<2, wp::int32>* var_15;
        wp::vec_t<2, wp::int32> var_16;
        wp::vec_t<2, wp::int32> var_17;
        const wp::int32 var_18 = 0;
        wp::int32 var_19;
        wp::int32* var_20;
        const wp::int32 var_21 = 1;
        wp::int32 var_22;
        wp::int32* var_23;
        wp::vec_t<2, wp::int32> var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        const wp::int32 var_30 = 0;
        wp::int32* var_31;
        wp::int32 var_32;
        wp::int32 var_33;
        const wp::int32 var_34 = 0;
        bool var_35;
        const wp::int32 var_36 = 0;
        wp::int32 var_37;
        bool var_38;
        const wp::int32 var_39 = 1;
        wp::int32 var_40;
        bool var_41;
        bool var_42;
        bool var_43;
        wp::float32* var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        const wp::int32 var_47 = 0;
        bool var_48;
        wp::int32* var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        const wp::int32 var_52 = 2;
        const wp::int32 var_53 = 1;
        wp::int32 var_54;
        wp::int32 var_55;
        const wp::int32 var_56 = 1;
        wp::range_t var_57;
        wp::int32 var_58;
        wp::int32* var_59;
        wp::float32* var_60;
        wp::int32 var_61;
        wp::float32 var_62;
        wp::float32 var_63;
        const wp::float32 var_64 = 0.0;
        bool var_65;
        wp::mat_t<3, 3, wp::float32>* var_66;
        wp::mat_t<3, 3, wp::float32> var_67;
        wp::mat_t<3, 3, wp::float32> var_68;
        const wp::int32 var_69 = 0;
        const wp::int32 var_70 = 0;
        wp::float32 var_71;
        const wp::int32 var_72 = 0;
        const wp::int32 var_73 = 1;
        wp::float32 var_74;
        const wp::int32 var_75 = 0;
        const wp::int32 var_76 = 2;
        wp::float32 var_77;
        wp::vec_t<3, wp::float32> var_78;
        wp::vec_t<3, wp::float32> var_79;
        wp::vec_t<3, wp::float32> var_80;
        wp::float32 var_81;
        const wp::int32 var_82 = 1;
        wp::int32 var_83;
        bool var_84;
        wp::vec_t<3, wp::float32> var_85;
        wp::vec_t<3, wp::float32> var_86;
        wp::vec_t<3, wp::float32>* var_87;
        wp::mat_t<3, 3, wp::float32>* var_88;
        wp::vec_t<3, wp::float32>* var_89;
        wp::vec_t<3, wp::float32>* var_90;
        wp::int32* var_91;
        wp::float32 var_92;
        wp::vec_t<3, wp::float32> var_93;
        wp::vec_t<3, wp::float32> var_94;
        wp::mat_t<3, 3, wp::float32> var_95;
        wp::vec_t<3, wp::float32> var_96;
        wp::vec_t<3, wp::float32> var_97;
        wp::int32 var_98;
        const wp::float32 var_99 = 0.0;
        bool var_100;
        wp::int32* var_101;
        wp::int32 var_102;
        wp::int32 var_103;
        wp::slice_t var_104;
        const wp::int32 var_105 = 0;
        wp::array_t<wp::float32> var_106;
        wp::float32 var_107;
        //---------
        // forward
        // def _sensor_touch(                                                                     <L 2001>
        // conid, sensortouchadrid = wp.tid()                                                     <L 2025>
        builtin_tid2d(var_0, var_1);
        // if conid >= nacon_in[0]:                                                               <L 2027>
        var_3 = wp::address(var_nacon_in, var_2);
        var_5 = wp::load(var_3);
        var_4 = (var_0 >= var_5);
        if (var_4) {
            // return                                                                             <L 2028>
            continue;
        }
        // sensorid = sensor_touch_adr[sensortouchadrid]                                          <L 2030>
        var_6 = wp::address(var_sensor_touch_adr, var_1);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // objid = sensor_objid[sensorid]                                                         <L 2032>
        var_9 = wp::address(var_sensor_objid, var_7);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // bodyid = site_bodyid[objid]                                                            <L 2033>
        var_12 = wp::address(var_site_bodyid, var_10);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // geom = contact_geom_in[conid]                                                          <L 2038>
        var_15 = wp::address(var_contact_geom_in, var_0);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // conbody = wp.vec2i(geom_bodyid[geom[0]], geom_bodyid[geom[1]])                         <L 2039>
        var_19 = wp::extract(var_16, var_18);
        var_20 = wp::address(var_geom_bodyid, var_19);
        var_22 = wp::extract(var_16, var_21);
        var_23 = wp::address(var_geom_bodyid, var_22);
        var_25 = wp::load(var_20);
        var_26 = wp::load(var_23);
        var_24 = wp::vec_t<2, wp::int32>(var_25, var_26);
        // worldid = contact_worldid_in[conid]                                                    <L 2042>
        var_27 = wp::address(var_contact_worldid_in, var_0);
        var_29 = wp::load(var_27);
        var_28 = wp::copy(var_29);
        // efc_address0 = contact_efc_address_in[conid, 0]                                        <L 2043>
        var_31 = wp::address(var_contact_efc_address_in, var_0, var_30);
        var_33 = wp::load(var_31);
        var_32 = wp::copy(var_33);
        // if efc_address0 >= 0 and (bodyid == conbody[0] or bodyid == conbody[1]):               <L 2044>
        var_35 = (var_32 >= var_34);
        var_37 = wp::extract(var_24, var_36);
        var_38 = (var_13 == var_37);
        var_40 = wp::extract(var_24, var_39);
        var_41 = (var_13 == var_40);
        var_42 = var_38 || var_41;
        var_43 = var_35 && var_42;
        if (var_43) {
            // normalforce = efc_force_in[worldid, efc_address0]                                  <L 2046>
            var_44 = wp::address(var_efc_force_in, var_28, var_32);
            var_46 = wp::load(var_44);
            var_45 = wp::copy(var_46);
            // if opt_cone == ConeType.PYRAMIDAL:                                                 <L 2048>
            var_48 = (var_opt_cone == var_47);
            if (var_48) {
                // dim = contact_dim_in[conid]                                                    <L 2049>
                var_49 = wp::address(var_contact_dim_in, var_0);
                var_51 = wp::load(var_49);
                var_50 = wp::copy(var_51);
                // for i in range(1, 2 * (dim - 1)):                                              <L 2050>
                var_54 = wp::sub(var_50, var_53);
                var_55 = wp::mul(var_52, var_54);
                var_57 = wp::range(var_56, var_55);
                start_for_1:;
                    if (iter_cmp(var_57) == 0) goto end_for_1;
                    var_58 = wp::iter_next(var_57);
                    // normalforce += efc_force_in[worldid, contact_efc_address_in[conid, i]]       <L 2051>
                    var_59 = wp::address(var_contact_efc_address_in, var_0, var_58);
                    var_61 = wp::load(var_59);
                    var_60 = wp::address(var_efc_force_in, var_28, var_61);
                    var_63 = wp::load(var_60);
                    var_62 = wp::add(var_45, var_63);
                    wp::assign(var_45, var_62);
                    goto start_for_1;
                end_for_1:;
            }
            // if normalforce <= 0.0:                                                             <L 2053>
            var_65 = (var_45 <= var_64);
            if (var_65) {
                // return                                                                         <L 2054>
                continue;
            }
            // frame = contact_frame_in[conid]                                                    <L 2057>
            var_66 = wp::address(var_contact_frame_in, var_0);
            var_68 = wp::load(var_66);
            var_67 = wp::copy(var_68);
            // conray = wp.vec3(frame[0, 0], frame[0, 1], frame[0, 2]) * normalforce              <L 2058>
            var_71 = wp::extract(var_67, var_69, var_70);
            var_74 = wp::extract(var_67, var_72, var_73);
            var_77 = wp::extract(var_67, var_75, var_76);
            var_78 = wp::vec_t<3, wp::float32>(var_71, var_74, var_77);
            var_79 = wp::mul(var_78, var_45);
            // conray, _ = math.normalize_with_norm(conray)                                       <L 2059>
            normalize_with_norm_0(var_79, var_80, var_81);
            // if bodyid == conbody[1]:                                                           <L 2062>
            var_83 = wp::extract(var_24, var_82);
            var_84 = (var_13 == var_83);
            if (var_84) {
                // conray = -conray                                                               <L 2063>
                var_85 = wp::neg(var_80);
            }
            var_86 = wp::where(var_84, var_85, var_80);
            // dist, normal = ray.ray_geom(                                                       <L 2066>
            // site_xpos_in[worldid, objid],                                                      <L 2067>
            var_87 = wp::address(var_site_xpos_in, var_28, var_10);
            // site_xmat_in[worldid, objid],                                                      <L 2068>
            var_88 = wp::address(var_site_xmat_in, var_28, var_10);
            // site_size[objid],                                                                  <L 2069>
            var_89 = wp::address(var_site_size, var_10);
            // contact_pos_in[conid],                                                             <L 2070>
            var_90 = wp::address(var_contact_pos_in, var_0);
            // conray,                                                                            <L 2071>
            // site_type[objid],                                                                  <L 2072>
            var_91 = wp::address(var_site_type, var_10);
            var_94 = wp::load(var_87);
            var_95 = wp::load(var_88);
            var_96 = wp::load(var_89);
            var_97 = wp::load(var_90);
            var_98 = wp::load(var_91);
            ray_geom_0(var_94, var_95, var_96, var_97, var_86, var_98, var_92, var_93);
            // if dist >= 0.0:                                                                    <L 2074>
            var_100 = (var_92 >= var_99);
            if (var_100) {
                // adr = sensor_adr[sensorid]                                                     <L 2075>
                var_101 = wp::address(var_sensor_adr, var_7);
                var_103 = wp::load(var_101);
                var_102 = wp::copy(var_103);
                // wp.atomic_add(sensordata_out[worldid], adr, normalforce)                       <L 2076>
                var_104 = wp::slice_t(var_28, var_28, var_105);
                var_106 = wp::view(var_sensordata_out, var_104);
                var_107 = wp::atomic_add(var_106, var_102, var_45);
            }
        }
    }
}



extern "C" __global__ void _preprocess_tactile_contacts_0f2dba1d_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_body_weldid,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_in,
    wp::array_t<wp::int32> var_contact_worldid_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::int32> var_weld_geom_count_out,
    wp::array_t<wp::int32> var_weld_geom_list_out)
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
        wp::int32 var_3;
        wp::int32 var_4;
        bool var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::vec_t<2, wp::int32>* var_9;
        wp::vec_t<2, wp::int32> var_10;
        wp::vec_t<2, wp::int32> var_11;
        const wp::int32 var_12 = 0;
        wp::int32 var_13;
        wp::int32* var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        const wp::int32 var_19 = 1;
        wp::int32 var_20;
        wp::int32* var_21;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        const wp::int32 var_26 = 0;
        wp::int32 var_27;
        const wp::int32 var_28 = 1;
        wp::int32 var_29;
        const wp::int32 var_30 = 0;
        const wp::int32 var_31 = 0;
        bool var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        wp::slice_t var_39;
        const wp::int32 var_40 = 0;
        wp::array_t<wp::int32> var_41;
        const wp::int32 var_42 = 1;
        wp::int32 var_43;
        const wp::int32 var_44 = 50;
        bool var_45;
        const wp::int32 var_46 = 1;
        const wp::int32 var_47 = 0;
        bool var_48;
        wp::int32 var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        wp::int32 var_52;
        wp::int32 var_53;
        wp::int32 var_54;
        wp::int32 var_55;
        wp::int32 var_56;
        wp::slice_t var_57;
        const wp::int32 var_58 = 0;
        wp::array_t<wp::int32> var_59;
        const wp::int32 var_60 = 1;
        wp::int32 var_61;
        bool var_62;
        //---------
        // forward
        // def _preprocess_tactile_contacts(                                                      <L 2085>
        // conid = wp.tid()                                                                       <L 2097>
        var_0 = builtin_tid1d();
        // ncon = nacon_in[0]                                                                     <L 2098>
        var_2 = wp::address(var_nacon_in, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if conid >= ncon:                                                                      <L 2099>
        var_5 = (var_0 >= var_3);
        if (var_5) {
            // return                                                                             <L 2100>
            continue;
        }
        // worldid = contact_worldid_in[conid]                                                    <L 2101>
        var_6 = wp::address(var_contact_worldid_in, var_0);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // contact_geom = contact_geom_in[conid]                                                  <L 2102>
        var_9 = wp::address(var_contact_geom_in, var_0);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // weld1 = body_weldid[geom_bodyid[contact_geom[0]]]                                      <L 2103>
        var_13 = wp::extract(var_10, var_12);
        var_14 = wp::address(var_geom_bodyid, var_13);
        var_16 = wp::load(var_14);
        var_15 = wp::address(var_body_weldid, var_16);
        var_18 = wp::load(var_15);
        var_17 = wp::copy(var_18);
        // weld2 = body_weldid[geom_bodyid[contact_geom[1]]]                                      <L 2104>
        var_20 = wp::extract(var_10, var_19);
        var_21 = wp::address(var_geom_bodyid, var_20);
        var_23 = wp::load(var_21);
        var_22 = wp::address(var_body_weldid, var_23);
        var_25 = wp::load(var_22);
        var_24 = wp::copy(var_25);
        // geom1 = contact_geom[0]                                                                <L 2105>
        var_27 = wp::extract(var_10, var_26);
        // geom2 = contact_geom[1]                                                                <L 2106>
        var_29 = wp::extract(var_10, var_28);
        // for side in range(2):                                                                  <L 2108>
        // if side == 0:                                                                          <L 2109>
        var_32 = (var_30 == var_31);
        if (var_32) {
            // weld = weld1                                                                       <L 2110>
            var_33 = wp::copy(var_17);
            // geom = geom2                                                                       <L 2111>
            var_34 = wp::copy(var_29);
        }
        if (!var_32) {
            // weld = weld2                                                                       <L 2113>
            var_35 = wp::copy(var_24);
            // geom = geom1                                                                       <L 2114>
            var_36 = wp::copy(var_27);
        }
        var_37 = wp::where(var_32, var_33, var_35);
        var_38 = wp::where(var_32, var_34, var_36);
        // idx = wp.atomic_add(weld_geom_count_out[worldid], weld, 1)                             <L 2116>
        var_39 = wp::slice_t(var_7, var_7, var_40);
        var_41 = wp::view(var_weld_geom_count_out, var_39);
        var_43 = wp::atomic_add(var_41, var_37, var_42);
        // if idx < MJ_MAXCONPAIR:                                                                <L 2117>
        var_45 = (var_43 < var_44);
        if (var_45) {
            // weld_geom_list_out[worldid, weld, idx] = geom                                      <L 2118>
            wp::array_store(var_weld_geom_list_out, var_7, var_37, var_43, var_38);
        }
        // if side == 0:                                                                          <L 2109>
        var_48 = (var_46 == var_47);
        if (var_48) {
            // weld = weld1                                                                       <L 2110>
            var_49 = wp::copy(var_17);
            // geom = geom2                                                                       <L 2111>
            var_50 = wp::copy(var_29);
        }
        var_51 = wp::where(var_48, var_49, var_37);
        var_52 = wp::where(var_48, var_50, var_38);
        if (!var_48) {
            // weld = weld2                                                                       <L 2113>
            var_53 = wp::copy(var_24);
            // geom = geom1                                                                       <L 2114>
            var_54 = wp::copy(var_27);
        }
        var_55 = wp::where(var_48, var_51, var_53);
        var_56 = wp::where(var_48, var_52, var_54);
        // idx = wp.atomic_add(weld_geom_count_out[worldid], weld, 1)                             <L 2116>
        var_57 = wp::slice_t(var_7, var_7, var_58);
        var_59 = wp::view(var_weld_geom_count_out, var_57);
        var_61 = wp::atomic_add(var_59, var_55, var_60);
        // if idx < MJ_MAXCONPAIR:                                                                <L 2117>
        var_62 = (var_61 < var_44);
        if (var_62) {
            // weld_geom_list_out[worldid, weld, idx] = geom                                      <L 2118>
            wp::array_store(var_weld_geom_list_out, var_7, var_55, var_61, var_56);
        }
    }
}



extern "C" __global__ void _energy_pos_passive_tendon_fcd8b5af_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_tendon_stiffness,
    wp::array_t<wp::vec_t<2, wp::float32>> var_tendon_lengthspring,
    wp::array_t<wp::float32> var_ten_length_in,
    wp::array_t<wp::vec_t<2, wp::float32>> var_energy_out)
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
        const wp::int32 var_25 = 1;
        wp::float32 var_26;
        bool var_27;
        wp::float32 var_28;
        bool var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        const wp::float32 var_32 = 0.0;
        wp::float32 var_33;
        wp::float32 var_34;
        const wp::float32 var_35 = 0.5;
        wp::float32 var_36;
        wp::float32 var_37;
        wp::float32 var_38;
        const wp::float32 var_39 = 0.0;
        wp::vec_t<2, wp::float32> var_40;
        wp::vec_t<2, wp::float32> var_41;
        //---------
        // forward
        // def _energy_pos_passive_tendon(                                                        <L 2818>
        // worldid, tenid = wp.tid()                                                              <L 2827>
        builtin_tid2d(var_0, var_1);
        // tendon_stiffness_id = worldid % tendon_stiffness.shape[0]                              <L 2829>
        var_2 = &(var_tendon_stiffness.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // stiffness = tendon_stiffness[tendon_stiffness_id, tenid]                               <L 2830>
        var_7 = wp::address(var_tendon_stiffness, var_6, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // if stiffness == 0.0:                                                                   <L 2832>
        var_11 = (var_8 == var_10);
        if (var_11) {
            // return                                                                             <L 2833>
            continue;
        }
        // length = ten_length_in[worldid, tenid]                                                 <L 2835>
        var_12 = wp::address(var_ten_length_in, var_0, var_1);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // tendon_lengthspring_id = worldid % tendon_lengthspring.shape[0]                        <L 2838>
        var_15 = &(var_tendon_lengthspring.shape);
        var_18 = wp::load(var_15);
        var_17 = wp::extract(var_18, var_16);
        var_19 = wp::mod(var_0, var_17);
        // lengthspring = tendon_lengthspring[tendon_lengthspring_id, tenid]                      <L 2839>
        var_20 = wp::address(var_tendon_lengthspring, var_19, var_1);
        var_22 = wp::load(var_20);
        var_21 = wp::copy(var_22);
        // lower = lengthspring[0]                                                                <L 2840>
        var_24 = wp::extract(var_21, var_23);
        // upper = lengthspring[1]                                                                <L 2841>
        var_26 = wp::extract(var_21, var_25);
        // if length > upper:                                                                     <L 2843>
        var_27 = (var_13 > var_26);
        if (var_27) {
            // displacement = upper - length                                                      <L 2844>
            var_28 = wp::sub(var_26, var_13);
        }
        if (!var_27) {
            // elif length < lower:                                                               <L 2845>
            var_29 = (var_13 < var_24);
            if (var_29) {
                // displacement = lower - length                                                  <L 2846>
                var_30 = wp::sub(var_24, var_13);
            }
            var_31 = wp::where(var_29, var_30, var_28);
            if (!var_29) {
                // displacement = 0.0                                                             <L 2848>
            }
            var_33 = wp::where(var_29, var_31, var_32);
        }
        var_34 = wp::where(var_27, var_28, var_33);
        // energy = wp.vec2(0.5 * stiffness * displacement * displacement, 0.0)                   <L 2850>
        var_36 = wp::mul(var_35, var_8);
        var_37 = wp::mul(var_36, var_34);
        var_38 = wp::mul(var_37, var_34);
        var_40 = wp::vec_t<2, wp::float32>(var_38, var_39);
        // wp.atomic_add(energy_out, worldid, energy)                                             <L 2851>
        var_41 = wp::atomic_add(var_energy_out, var_0, var_40);
    }
}



extern "C" __global__ void _tendon_actuator_force_cutoff_1d17a2e3_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::array_t<wp::int32> var_sensor_tendonactfrc_adr,
    wp::array_t<wp::float32> var_sensordata_in,
    wp::array_t<wp::float32> var_sensordata_out)
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
        wp::slice_t var_11;
        const wp::int32 var_12 = 0;
        wp::array_t<wp::float32> var_13;
        //---------
        // forward
        // def _tendon_actuator_force_cutoff(                                                     <L 1560>
        // worldid, tenactfrcid = wp.tid()                                                        <L 1572>
        builtin_tid2d(var_0, var_1);
        // sensorid = sensor_tendonactfrc_adr[tenactfrcid]                                        <L 1573>
        var_2 = wp::address(var_sensor_tendonactfrc_adr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // adr = sensor_adr[sensorid]                                                             <L 1574>
        var_5 = wp::address(var_sensor_adr, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // val = sensordata_in[worldid, adr]                                                      <L 1575>
        var_8 = wp::address(var_sensordata_in, var_0, var_6);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, sensordata_out[worldid])       <L 1577>
        var_11 = wp::slice_t(var_0, var_0, var_12);
        var_13 = wp::view(var_sensordata_out, var_11);
        _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_9, var_13);
    }
}



extern "C" __global__ void _sensor_vel_e2a1adc2_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_site_bodyid,
    wp::array_t<wp::int32> var_cam_bodyid,
    wp::array_t<wp::int32> var_sensor_type,
    wp::array_t<wp::int32> var_sensor_datatype,
    wp::array_t<wp::int32> var_sensor_objtype,
    wp::array_t<wp::int32> var_sensor_objid,
    wp::array_t<wp::int32> var_sensor_reftype,
    wp::array_t<wp::int32> var_sensor_refid,
    wp::array_t<wp::int32> var_sensor_adr,
    wp::array_t<wp::float32> var_sensor_cutoff,
    wp::array_t<wp::int32> var_sensor_vel_adr,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_site_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_site_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cam_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cam_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::float32> var_ten_velocity_in,
    wp::array_t<wp::float32> var_actuator_velocity_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_linvel_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_angmom_in,
    wp::array_t<wp::float32> var_sensordata_out)
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
        wp::slice_t var_11;
        const wp::int32 var_12 = 0;
        wp::array_t<wp::float32> var_13;
        const wp::int32 var_14 = 2;
        bool var_15;
        wp::vec_t<3, wp::float32> var_16;
        const wp::int32 var_17 = 3;
        const wp::int32 var_18 = 3;
        bool var_19;
        wp::vec_t<3, wp::float32> var_20;
        const wp::int32 var_21 = 3;
        wp::vec_t<3, wp::float32> var_22;
        const wp::int32 var_23 = 10;
        bool var_24;
        wp::float32 var_25;
        const wp::int32 var_26 = 12;
        bool var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        const wp::int32 var_30 = 14;
        bool var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        const wp::int32 var_34 = 19;
        bool var_35;
        wp::vec_t<3, wp::float32> var_36;
        const wp::int32 var_37 = 3;
        wp::vec_t<3, wp::float32> var_38;
        const wp::int32 var_39 = 31;
        bool var_40;
        wp::int32* var_41;
        wp::int32 var_42;
        wp::int32 var_43;
        wp::int32* var_44;
        wp::int32 var_45;
        wp::int32 var_46;
        wp::int32* var_47;
        wp::int32 var_48;
        wp::int32 var_49;
        wp::vec_t<3, wp::float32> var_50;
        const wp::int32 var_51 = 3;
        const wp::int32 var_52 = 32;
        bool var_53;
        wp::int32* var_54;
        wp::int32 var_55;
        wp::int32 var_56;
        wp::int32* var_57;
        wp::int32 var_58;
        wp::int32 var_59;
        wp::int32* var_60;
        wp::int32 var_61;
        wp::int32 var_62;
        wp::vec_t<3, wp::float32> var_63;
        const wp::int32 var_64 = 3;
        wp::int32 var_65;
        wp::int32 var_66;
        wp::int32 var_67;
        const wp::int32 var_68 = 36;
        bool var_69;
        wp::vec_t<3, wp::float32> var_70;
        const wp::int32 var_71 = 3;
        wp::vec_t<3, wp::float32> var_72;
        const wp::int32 var_73 = 37;
        bool var_74;
        wp::vec_t<3, wp::float32> var_75;
        const wp::int32 var_76 = 3;
        wp::vec_t<3, wp::float32> var_77;
        wp::vec_t<3, wp::float32> var_78;
        wp::vec_t<3, wp::float32> var_79;
        wp::vec_t<3, wp::float32> var_80;
        wp::int32 var_81;
        wp::int32 var_82;
        wp::int32 var_83;
        wp::vec_t<3, wp::float32> var_84;
        wp::vec_t<3, wp::float32> var_85;
        wp::vec_t<3, wp::float32> var_86;
        wp::float32 var_87;
        wp::vec_t<3, wp::float32> var_88;
        wp::float32 var_89;
        wp::vec_t<3, wp::float32> var_90;
        wp::vec_t<3, wp::float32> var_91;
        //---------
        // forward
        // def _sensor_vel(                                                                       <L 1251>
        // worldid, velid = wp.tid()                                                              <L 1288>
        builtin_tid2d(var_0, var_1);
        // sensorid = sensor_vel_adr[velid]                                                       <L 1289>
        var_2 = wp::address(var_sensor_vel_adr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // sensortype = sensor_type[sensorid]                                                     <L 1290>
        var_5 = wp::address(var_sensor_type, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // objid = sensor_objid[sensorid]                                                         <L 1291>
        var_8 = wp::address(var_sensor_objid, var_3);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // out = sensordata_out[worldid]                                                          <L 1292>
        var_11 = wp::slice_t(var_0, var_0, var_12);
        var_13 = wp::view(var_sensordata_out, var_11);
        // if sensortype == SensorType.VELOCIMETER:                                               <L 1294>
        var_15 = (var_6 == var_14);
        if (var_15) {
            // vec3 = _velocimeter(body_rootid, site_bodyid, site_xpos_in, site_xmat_in, subtree_com_in, cvel_in, worldid, objid)       <L 1295>
            var_16 = _velocimeter_0(var_body_rootid, var_site_bodyid, var_site_xpos_in, var_site_xmat_in, var_subtree_com_in, var_cvel_in, var_0, var_9);
            // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)       <L 1296>
            _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_17, var_16, var_13);
        }
        if (!var_15) {
            // elif sensortype == SensorType.GYRO:                                                <L 1297>
            var_19 = (var_6 == var_18);
            if (var_19) {
                // vec3 = _gyro(site_bodyid, site_xmat_in, cvel_in, worldid, objid)               <L 1298>
                var_20 = _gyro_0(var_site_bodyid, var_site_xmat_in, var_cvel_in, var_0, var_9);
                // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)       <L 1299>
                _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_21, var_20, var_13);
            }
            var_22 = wp::where(var_19, var_20, var_16);
            if (!var_19) {
                // elif sensortype == SensorType.JOINTVEL:                                        <L 1300>
                var_24 = (var_6 == var_23);
                if (var_24) {
                    // val = _joint_vel(jnt_dofadr, qvel_in, worldid, objid)                      <L 1301>
                    var_25 = _joint_vel_0(var_jnt_dofadr, var_qvel_in, var_0, var_9);
                    // _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)       <L 1302>
                    _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_25, var_13);
                }
                if (!var_24) {
                    // elif sensortype == SensorType.TENDONVEL:                                   <L 1303>
                    var_27 = (var_6 == var_26);
                    if (var_27) {
                        // val = _tendon_vel(ten_velocity_in, worldid, objid)                     <L 1304>
                        var_28 = _tendon_vel_0(var_ten_velocity_in, var_0, var_9);
                        // _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)       <L 1305>
                        _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_28, var_13);
                    }
                    var_29 = wp::where(var_27, var_28, var_25);
                    if (!var_27) {
                        // elif sensortype == SensorType.ACTUATORVEL:                             <L 1306>
                        var_31 = (var_6 == var_30);
                        if (var_31) {
                            // val = _actuator_vel(actuator_velocity_in, worldid, objid)          <L 1307>
                            var_32 = _actuator_vel_0(var_actuator_velocity_in, var_0, var_9);
                            // _write_scalar(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)       <L 1308>
                            _write_scalar_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_32, var_13);
                        }
                        var_33 = wp::where(var_31, var_32, var_29);
                        if (!var_31) {
                            // elif sensortype == SensorType.BALLANGVEL:                          <L 1309>
                            var_35 = (var_6 == var_34);
                            if (var_35) {
                                // vec3 = _ball_ang_vel(jnt_dofadr, qvel_in, worldid, objid)       <L 1310>
                                var_36 = _ball_ang_vel_0(var_jnt_dofadr, var_qvel_in, var_0, var_9);
                                // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)       <L 1311>
                                _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_37, var_36, var_13);
                            }
                            var_38 = wp::where(var_35, var_36, var_22);
                            if (!var_35) {
                                // elif sensortype == SensorType.FRAMELINVEL:                     <L 1312>
                                var_40 = (var_6 == var_39);
                                if (var_40) {
                                    // objtype = sensor_objtype[sensorid]                         <L 1313>
                                    var_41 = wp::address(var_sensor_objtype, var_3);
                                    var_43 = wp::load(var_41);
                                    var_42 = wp::copy(var_43);
                                    // refid = sensor_refid[sensorid]                             <L 1314>
                                    var_44 = wp::address(var_sensor_refid, var_3);
                                    var_46 = wp::load(var_44);
                                    var_45 = wp::copy(var_46);
                                    // reftype = sensor_reftype[sensorid]                         <L 1315>
                                    var_47 = wp::address(var_sensor_reftype, var_3);
                                    var_49 = wp::load(var_47);
                                    var_48 = wp::copy(var_49);
                                    // frame_linvel = _frame_linvel(                              <L 1316>
                                    // body_rootid,                                               <L 1317>
                                    // geom_bodyid,                                               <L 1318>
                                    // site_bodyid,                                               <L 1319>
                                    // cam_bodyid,                                                <L 1320>
                                    // xpos_in,                                                   <L 1321>
                                    // xmat_in,                                                   <L 1322>
                                    // xipos_in,                                                  <L 1323>
                                    // ximat_in,                                                  <L 1324>
                                    // geom_xpos_in,                                              <L 1325>
                                    // geom_xmat_in,                                              <L 1326>
                                    // site_xpos_in,                                              <L 1327>
                                    // site_xmat_in,                                              <L 1328>
                                    // cam_xpos_in,                                               <L 1329>
                                    // cam_xmat_in,                                               <L 1330>
                                    // subtree_com_in,                                            <L 1331>
                                    // cvel_in,                                                   <L 1332>
                                    // worldid,                                                   <L 1333>
                                    // objid,                                                     <L 1334>
                                    // objtype,                                                   <L 1335>
                                    // refid,                                                     <L 1336>
                                    // reftype,                                                   <L 1337>
                                    var_50 = _frame_linvel_0(var_body_rootid, var_geom_bodyid, var_site_bodyid, var_cam_bodyid, var_xpos_in, var_xmat_in, var_xipos_in, var_ximat_in, var_geom_xpos_in, var_geom_xmat_in, var_site_xpos_in, var_site_xmat_in, var_cam_xpos_in, var_cam_xmat_in, var_subtree_com_in, var_cvel_in, var_0, var_9, var_42, var_45, var_48);
                                    // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, frame_linvel, out)       <L 1339>
                                    _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_51, var_50, var_13);
                                }
                                if (!var_40) {
                                    // elif sensortype == SensorType.FRAMEANGVEL:                 <L 1340>
                                    var_53 = (var_6 == var_52);
                                    if (var_53) {
                                        // objtype = sensor_objtype[sensorid]                     <L 1341>
                                        var_54 = wp::address(var_sensor_objtype, var_3);
                                        var_56 = wp::load(var_54);
                                        var_55 = wp::copy(var_56);
                                        // refid = sensor_refid[sensorid]                         <L 1342>
                                        var_57 = wp::address(var_sensor_refid, var_3);
                                        var_59 = wp::load(var_57);
                                        var_58 = wp::copy(var_59);
                                        // reftype = sensor_reftype[sensorid]                     <L 1343>
                                        var_60 = wp::address(var_sensor_reftype, var_3);
                                        var_62 = wp::load(var_60);
                                        var_61 = wp::copy(var_62);
                                        // frame_angvel = _frame_angvel(                          <L 1344>
                                        // body_rootid,                                           <L 1345>
                                        // geom_bodyid,                                           <L 1346>
                                        // site_bodyid,                                           <L 1347>
                                        // cam_bodyid,                                            <L 1348>
                                        // xpos_in,                                               <L 1349>
                                        // xmat_in,                                               <L 1350>
                                        // xipos_in,                                              <L 1351>
                                        // ximat_in,                                              <L 1352>
                                        // geom_xpos_in,                                          <L 1353>
                                        // geom_xmat_in,                                          <L 1354>
                                        // site_xpos_in,                                          <L 1355>
                                        // site_xmat_in,                                          <L 1356>
                                        // cam_xpos_in,                                           <L 1357>
                                        // cam_xmat_in,                                           <L 1358>
                                        // subtree_com_in,                                        <L 1359>
                                        // cvel_in,                                               <L 1360>
                                        // worldid,                                               <L 1361>
                                        // objid,                                                 <L 1362>
                                        // objtype,                                               <L 1363>
                                        // refid,                                                 <L 1364>
                                        // reftype,                                               <L 1365>
                                        var_63 = _frame_angvel_0(var_body_rootid, var_geom_bodyid, var_site_bodyid, var_cam_bodyid, var_xpos_in, var_xmat_in, var_xipos_in, var_ximat_in, var_geom_xpos_in, var_geom_xmat_in, var_site_xpos_in, var_site_xmat_in, var_cam_xpos_in, var_cam_xmat_in, var_subtree_com_in, var_cvel_in, var_0, var_9, var_55, var_58, var_61);
                                        // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, frame_angvel, out)       <L 1367>
                                        _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_64, var_63, var_13);
                                    }
                                    var_65 = wp::where(var_53, var_55, var_42);
                                    var_66 = wp::where(var_53, var_58, var_45);
                                    var_67 = wp::where(var_53, var_61, var_48);
                                    if (!var_53) {
                                        // elif sensortype == SensorType.SUBTREELINVEL:           <L 1368>
                                        var_69 = (var_6 == var_68);
                                        if (var_69) {
                                            // vec3 = _subtree_linvel(subtree_linvel_in, worldid, objid)       <L 1369>
                                            var_70 = _subtree_linvel_0(var_subtree_linvel_in, var_0, var_9);
                                            // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)       <L 1370>
                                            _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_71, var_70, var_13);
                                        }
                                        var_72 = wp::where(var_69, var_70, var_38);
                                        if (!var_69) {
                                            // elif sensortype == SensorType.SUBTREEANGMOM:       <L 1371>
                                            var_74 = (var_6 == var_73);
                                            if (var_74) {
                                                // vec3 = _subtree_angmom(subtree_angmom_in, worldid, objid)       <L 1372>
                                                var_75 = _subtree_angmom_0(var_subtree_angmom_in, var_0, var_9);
                                                // _write_vector(sensor_type, sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)       <L 1373>
                                                _write_vector_0(var_sensor_type, var_sensor_datatype, var_sensor_adr, var_sensor_cutoff, var_3, var_76, var_75, var_13);
                                            }
                                            var_77 = wp::where(var_74, var_75, var_72);
                                        }
                                        var_78 = wp::where(var_69, var_72, var_77);
                                    }
                                    var_79 = wp::where(var_53, var_38, var_78);
                                }
                                var_80 = wp::where(var_40, var_38, var_79);
                                var_81 = wp::where(var_40, var_42, var_65);
                                var_82 = wp::where(var_40, var_45, var_66);
                                var_83 = wp::where(var_40, var_48, var_67);
                            }
                            var_84 = wp::where(var_35, var_38, var_80);
                        }
                        var_85 = wp::where(var_31, var_22, var_84);
                    }
                    var_86 = wp::where(var_27, var_22, var_85);
                    var_87 = wp::where(var_27, var_29, var_33);
                }
                var_88 = wp::where(var_24, var_22, var_86);
                var_89 = wp::where(var_24, var_25, var_87);
            }
            var_90 = wp::where(var_19, var_22, var_88);
        }
        var_91 = wp::where(var_15, var_16, var_90);
    }
}

