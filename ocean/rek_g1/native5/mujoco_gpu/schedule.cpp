#include "schedule.h"
#include <stdexcept>
#include <utility>

namespace rek_mjgpu {
namespace {
constexpr const char* SMOOTH="wp_mujoco_warp._src.smooth_26bbc02";
constexpr const char* FORWARD="wp_mujoco_warp._src.forward_16fbac8";
constexpr const char* PASSIVE="wp_mujoco_warp._src.passive_8bfe11c";
constexpr const char* DERIVATIVE="wp_mujoco_warp._src.derivative_7ba27fe";
constexpr const char* SUPPORT="wp_mujoco_warp._src.support_dd3f924";
constexpr const char* SPARSE_SOLVE="wp__solve_LD_sparse_fused__locals__kernel_8d11feb4_ba43376";

void kernel(ScheduleSpec& out,const char* module,const char* entry,
            std::initializer_list<int> dimensions,std::initializer_list<ScheduleParameter> parameters,
            int block_dim=0,int shared_bytes=0){
    const auto bounds=launch_bounds(dimensions);
    if(bounds.size==0)return; // Same zero-work launch omission as Warp.
    if(block_dim<0||block_dim>1024)throw std::runtime_error("Invalid native MuJoCo kernel block dimension");
    ScheduleNode node;node.kind=ScheduleNode::Kind::Kernel;node.module=module;node.entry_prefix=entry;
    node.bounds=bounds;node.block_dim=block_dim;node.shared_bytes=shared_bytes;node.parameters=parameters;
    out.push_back(std::move(node));
}
void copy(ScheduleSpec& out,const char* dst,const char* src){
    ScheduleNode node;node.kind=ScheduleNode::Kind::Copy;node.destination=dst;node.source=src;out.push_back(std::move(node));
}
void zero(ScheduleSpec& out,const char* dst){
    ScheduleNode node;node.kind=ScheduleNode::Kind::Zero;node.destination=dst;out.push_back(std::move(node));
}
int count(const ModelData& data,const char* name){
    const int value=data.integer(name);if(value<0)throw std::runtime_error(std::string("Negative model dimension: ")+name);return value;
}
int one_dimensional_count(const ModelData& data,const std::string& name){
    const auto value=data.array(name);
    if(value.ndim!=1||value.shape[0]<0)throw std::runtime_error("Expected one-dimensional model tuple array: "+name);
    return value.shape[0];
}
void require_zero(const ModelData& data,std::initializer_list<const char*> names,const char* stage){
    for(auto name:names)if(data.integer(name)!=0)
        throw std::runtime_error(std::string(stage)+" specialization does not support nonzero "+name);
}
void scratch(ModelData& data,const char* name,std::initializer_list<int> shape,Element element=Element::F32){
    if(!data.contains(name)){data.allocate(name,std::vector<int>(shape),element);return;}
    const auto array=data.array(name);
    if(array.ndim!=int(shape.size()))throw std::runtime_error(std::string("Scratch rank mismatch: ")+name);
    int i=0;for(int size:shape)if(array.shape[i++]!=size)throw std::runtime_error(std::string("Scratch shape mismatch: ")+name);
}
} // namespace

void append_kinematics_com(ScheduleSpec& out,const ModelData& data){
    // smooth.py:358 kinematics, then smooth.py:602 com_pos. Parameter order
    // includes inputs followed by outputs exactly as the generated CUDA ABI.
    const int worlds=count(data,"d.nworld"),branches=count(data,"m.nbranch");
    const int bodies=count(data,"m.nbody"),geoms=count(data,"m.ngeom"),sites=count(data,"m.nsite");
    const int joints=count(data,"m.njnt");
    kernel(out,SMOOTH,"_kinematics_branch_",{worlds,branches},{
        "m.qpos0","m.body_parentid","m.body_mocapid","m.body_jntnum","m.body_jntadr",
        "m.body_pos","m.body_quat","m.jnt_type","m.jnt_qposadr","m.jnt_pos","m.jnt_axis",
        "m.body_branches","m.body_branch_start","d.qpos","d.mocap_pos","d.mocap_quat",
        "d.xpos","d.xquat","d.xanchor","d.xaxis"});
    kernel(out,SMOOTH,"_compute_body_matrices_",{worlds,bodies},{"d.xquat","d.xmat"});
    kernel(out,SMOOTH,"_compute_body_inertial_frames_",{worlds,bodies},{
        "m.body_ipos","m.body_iquat","d.xpos","d.xquat","d.xipos","d.ximat"});
    kernel(out,SMOOTH,"_geom_local_to_global_",{worlds,geoms},{
        "m.body_rootid","m.body_weldid","m.body_mocapid","m.geom_bodyid","m.geom_pos","m.geom_quat",
        "d.xpos","d.xquat","d.geom_xpos","d.geom_xmat"});
    kernel(out,SMOOTH,"_site_local_to_global_",{worlds,sites},{
        "m.site_bodyid","m.site_pos","m.site_quat","d.xpos","d.xquat","d.site_xpos","d.site_xmat"});

    kernel(out,SMOOTH,"_subtree_com_init_",{worlds,bodies},{"m.body_mass","d.xipos","d.subtree_com"});
    for(int level=count(data,"m.body_tree.count")-1;level>=0;--level){
        const auto tree="m.body_tree."+std::to_string(level);
        kernel(out,SMOOTH,"_subtree_com_acc_",{worlds,one_dimensional_count(data,tree)},
            {"m.body_parentid","d.subtree_com",tree,"d.subtree_com"});
    }
    kernel(out,SMOOTH,"_subtree_div_",{worlds,bodies},{"m.body_subtreemass","d.subtree_com","d.subtree_com"});
    kernel(out,SMOOTH,"_cinert_",{worlds,bodies},{
        "m.body_rootid","m.body_mass","m.body_inertia","d.xipos","d.ximat","d.subtree_com","d.cinert"});
    kernel(out,SMOOTH,"_cdof_",{worlds,joints},{
        "m.body_rootid","m.jnt_type","m.jnt_dofadr","m.jnt_bodyid","d.xmat","d.xanchor","d.xaxis",
        "d.subtree_com","d.cdof"});
}

void append_crb(ScheduleSpec& out,const ModelData& data){
    // smooth.py:889. This does not factor qM, matching forward's
    // fwd_position(..., factorize=False); factorization belongs to acceleration.
    const int worlds=count(data,"d.nworld"),nv=count(data,"m.nv");
    copy(out,"d.crb","d.cinert");
    for(int level=count(data,"m.body_tree.count")-1;level>=0;--level){
        const auto tree="m.body_tree."+std::to_string(level);
        kernel(out,SMOOTH,"_crb_accumulate_",{worlds,one_dimensional_count(data,tree)},
            {"m.body_parentid","d.crb",tree,"d.crb"});
    }
    zero(out,"d.qM");
    if(data.integer("m.is_sparse"))kernel(out,SMOOTH,"_qM_sparse_",{worlds,nv},{
        "m.dof_bodyid","m.dof_parentid","m.dof_Madr","m.dof_armature","d.cdof","d.crb","d.qM"});
    else kernel(out,SMOOTH,"_qM_dense_",{worlds,nv},{
        "m.dof_bodyid","m.dof_parentid","m.dof_armature","d.cdof","d.crb","d.qM"});
}

void append_camlight(ScheduleSpec& out,const ModelData& data){
    const int worlds=count(data,"d.nworld");
    kernel(out,SMOOTH,"_cam_local_to_global_",{worlds,count(data,"m.ncam")},{
        "m.cam_mode","m.cam_bodyid","m.cam_targetbodyid","m.cam_pos","m.cam_quat","m.cam_poscom0",
        "m.cam_pos0","m.cam_mat0","d.xpos","d.xquat","d.subtree_com","d.cam_xpos","d.cam_xmat"});
    kernel(out,SMOOTH,"_light_local_to_global_",{worlds,count(data,"m.nlight")},{
        "m.light_mode","m.light_bodyid","m.light_targetbodyid","m.light_pos","m.light_dir","m.light_poscom0",
        "m.light_pos0","m.light_dir0","d.xpos","d.xquat","d.subtree_com","d.light_xpos","d.light_xdir"});
}

void append_transmission(ScheduleSpec& out,ModelData& data){
    // The actual REK model has joint transmissions and no tendons/body actuators.
    require_zero(data,{"m.ntendon","m.nacttrnbody"},"Transmission");
    const int worlds=count(data,"d.nworld");
    scratch(data,"scratch.moment_nnz",{worlds},Element::I32);zero(out,"scratch.moment_nnz");
    kernel(out,SMOOTH,"_transmission_28153805_",{worlds,count(data,"m.nu")},{
        data.integer("m.nv"),"m.body_parentid","m.body_rootid","m.body_weldid","m.body_dofnum","m.body_dofadr",
        "m.jnt_type","m.jnt_qposadr","m.jnt_dofadr","m.dof_bodyid","m.dof_parentid","m.site_bodyid","m.site_quat",
        "m.ten_J_rownnz","m.ten_J_rowadr","m.ten_J_colind","m.actuator_trntype","m.actuator_trnid","m.actuator_gear",
        "m.actuator_cranklength","d.qpos","d.xquat","d.site_xpos","d.site_xmat","d.subtree_com","d.cdof",
        "d.ten_J","d.ten_length","scratch.moment_nnz","d.actuator_length","d.moment_rownnz","d.moment_rowadr",
        "d.moment_colind","d.actuator_moment"});
}

void append_spatial_velocity(ScheduleSpec& out,const ModelData& data){
    const int worlds=count(data,"d.nworld");
    kernel(out,SMOOTH,"_comvel_root_",{worlds,6},{"d.cvel"});
    kernel(out,SMOOTH,"_comvel_branch_",{worlds,count(data,"m.nbranch")},{
        "m.body_parentid","m.body_jntnum","m.body_jntadr","m.body_dofadr","m.jnt_type","m.body_branches",
        "m.body_branch_start","d.qvel","d.cdof","d.cvel","d.cdof_dot"});
}

void append_velocity_forces(ScheduleSpec& out,const ModelData& data){
    // forward.py:593 fwd_velocity and its passive/RNE dependency chain.
    require_zero(data,{"m.ntendon","m.nflex","m.has_fluid"},"Velocity/passive");
    const int worlds=count(data,"d.nworld"),nv=count(data,"m.nv"),bodies=count(data,"m.nbody");
    const int flags=data.integer("m.opt.disableflags");
    kernel(out,FORWARD,"_actuator_velocity_",{worlds,count(data,"m.nu")},{
        "d.qvel","d.moment_rownnz","d.moment_rowadr","d.moment_colind","d.actuator_moment","d.actuator_velocity"});
    append_spatial_velocity(out,data);
    const bool spring_disabled=(flags & mjDSBL_SPRING)!=0,damper_disabled=(flags & mjDSBL_DAMPER)!=0;
    if(spring_disabled&&damper_disabled){
        for(auto field:{"d.qfrc_spring","d.qfrc_damper","d.qfrc_gravcomp","d.qfrc_fluid","d.qfrc_passive"})zero(out,field);
    }else{
        kernel(out,PASSIVE,"_spring_damper_dof_passive_",{worlds,count(data,"m.njnt")},{
            flags,"m.qpos_spring","m.jnt_type","m.jnt_qposadr","m.jnt_dofadr","m.jnt_stiffness","m.dof_damping",
            "d.qpos","d.qvel","d.qfrc_spring","d.qfrc_damper"});
        const bool gravcomp=data.integer("m.ngravcomp")!=0&&!(flags & mjDSBL_GRAVITY);
        if(gravcomp){
            zero(out,"d.qfrc_gravcomp");
            kernel(out,PASSIVE,"_gravity_force_",{worlds,bodies-1,nv},{
                "m.opt.gravity","m.body_parentid","m.body_rootid","m.body_mass","m.body_gravcomp","m.dof_bodyid",
                "d.xipos","d.subtree_com","d.cdof","d.qfrc_gravcomp"});
        }
        kernel(out,PASSIVE,"_qfrc_passive_",{worlds,nv},{
            "m.jnt_actgravcomp","m.dof_jntid",false,"d.qfrc_spring","d.qfrc_damper","d.qfrc_gravcomp","d.qfrc_fluid",
            gravcomp,"d.qfrc_passive"});
    }
    if(flags & mjDSBL_GRAVITY)zero(out,"d.cacc");
    else kernel(out,SMOOTH,"_cacc_world_",{worlds},{"m.opt.gravity","d.cacc"});
    kernel(out,SMOOTH,"_cacc_branch_",{worlds,count(data,"m.nbranch")},{
        "m.body_parentid","m.body_dofnum","m.body_dofadr","m.body_branches","m.body_branch_start",
        "d.qvel","d.qacc","d.cdof","d.cdof_dot",false,"d.cacc"});
    kernel(out,SMOOTH,"_cfrc_e3a9fd02_",{worlds,bodies-1},{"d.cinert","d.cvel","d.cacc","d.cfrc_ext",false,"d.cfrc_int"});
    for(int level=count(data,"m.body_tree.count")-1;level>=0;--level){
        const auto tree="m.body_tree."+std::to_string(level);
        kernel(out,SMOOTH,"_cfrc_backward_",{worlds,one_dimensional_count(data,tree)},
            {"m.body_parentid","d.cfrc_int",tree,"d.cfrc_int"});
    }
    kernel(out,SMOOTH,"_qfrc_bias_",{worlds,nv},{"m.dof_bodyid","d.cdof","d.cfrc_int","d.qfrc_bias"});
}

void append_actuation(ScheduleSpec& out,const ModelData& data){
    require_zero(data,{"m.ntendon"},"Actuation");
    const int worlds=count(data,"d.nworld"),nu=count(data,"m.nu"),flags=data.integer("m.opt.disableflags");
    if(!nu||(flags & mjDSBL_ACTUATION)){zero(out,"d.act_dot");zero(out,"d.qfrc_actuator");return;}
    kernel(out,FORWARD,"_actuator_force_",{worlds,nu},{
        data.integer("m.na"),"m.opt.timestep","m.actuator_dyntype","m.actuator_gaintype","m.actuator_biastype",
        "m.actuator_actadr","m.actuator_actnum","m.actuator_ctrllimited","m.actuator_forcelimited","m.actuator_actlimited",
        "m.actuator_dynprm","m.actuator_gainprm","m.actuator_biasprm","m.actuator_actearly","m.actuator_ctrlrange",
        "m.actuator_forcerange","m.actuator_actrange","m.actuator_acc0","m.actuator_lengthrange","d.act","d.ctrl",
        "d.actuator_length","d.actuator_velocity",int32_t(flags & mjDSBL_CLAMPCTRL),"d.act_dot","d.actuator_force"});
    zero(out,"d.qfrc_actuator");
    kernel(out,FORWARD,"_qfrc_actuator_6de5c51c_",{worlds,nu},{
        "d.moment_rownnz","d.moment_rowadr","d.moment_colind","d.actuator_moment","d.actuator_force","d.qfrc_actuator"});
    kernel(out,FORWARD,"_qfrc_actuator_gravcomp_limits_",{worlds,count(data,"m.nv")},{
        data.integer("m.ngravcomp"),"m.jnt_actfrclimited","m.jnt_actgravcomp","m.jnt_actfrcrange","m.dof_jntid",
        "d.qfrc_gravcomp","d.qfrc_actuator","d.qfrc_actuator"});
}

void append_smooth_acceleration(ScheduleSpec& out,const ModelData& data){
    const int worlds=count(data,"d.nworld"),nv=count(data,"m.nv");
    kernel(out,FORWARD,"_qfrc_smooth_",{worlds,nv},{
        "d.qfrc_applied","d.qfrc_bias","d.qfrc_passive","d.qfrc_actuator","d.qfrc_smooth"});
    kernel(out,SUPPORT,"_apply_ft_",{worlds,nv},{
        data.integer("m.nbody"),"m.body_parentid","m.body_rootid","m.dof_bodyid","d.xipos","d.subtree_com",
        "d.cdof","d.xfrc_applied",true,"d.qfrc_smooth"});
    append_sparse_factor_solve(out,data,"d.qM","d.qLD","d.qLDiagInv","d.qacc_smooth","d.qfrc_smooth");
}

void append_implicitfast_integration(ScheduleSpec& out,ModelData& data){
    if(data.integer("m.opt.integrator")!=mjINT_IMPLICITFAST)throw std::runtime_error("Integration schedule requires implicitfast");
    if(!data.integer("m.is_sparse"))throw std::runtime_error("Current implicitfast schedule requires sparse storage");
    require_zero(data,{"m.ntendon"},"Implicitfast");
    const int worlds=count(data,"d.nworld"),nv=count(data,"m.nv"),nu=count(data,"m.nu"),flags=data.integer("m.opt.disableflags");
    const char* acceleration="d.qacc";
    if((~flags & (mjDSBL_ACTUATION|mjDSBL_SPRING|mjDSBL_DAMPER))!=0){
        scratch(data,"scratch.implicit.qDeriv",{worlds,1,count(data,"m.nM")});
        scratch(data,"scratch.implicit.qLD",{worlds,1,count(data,"m.nC")});
        scratch(data,"scratch.implicit.qLDiagInv",{worlds,nv});
        scratch(data,"scratch.implicit.qacc",{worlds,nv});
        scratch(data,"scratch.implicit.actuator_vel",{worlds,nu});
        zero(out,"scratch.implicit.qDeriv");
        if(nu&&!(flags & mjDSBL_ACTUATION)){
            kernel(out,DERIVATIVE,"_qderiv_actuator_passive_vel_",{worlds,nu},{
                "m.opt.timestep","m.actuator_dyntype","m.actuator_gaintype","m.actuator_biastype","m.actuator_actadr",
                "m.actuator_actnum","m.actuator_forcelimited","m.actuator_actlimited","m.actuator_dynprm",
                "m.actuator_gainprm","m.actuator_biasprm","m.actuator_actearly","m.actuator_forcerange",
                "m.actuator_actrange","d.act","d.ctrl","d.act_dot","d.actuator_force","scratch.implicit.actuator_vel"});
            kernel(out,DERIVATIVE,"_qderiv_actuator_passive_actuation_sparse_",{worlds,nu},{
                "m.M_rownnz","m.M_rowadr","d.moment_rownnz","d.moment_rowadr","d.moment_colind",
                "d.actuator_moment","scratch.implicit.actuator_vel","m.qM_fullm_j","scratch.implicit.qDeriv"});
        }
        kernel(out,DERIVATIVE,"_qderiv_actuator_passive_7ab7826a_",{worlds,one_dimensional_count(data,"m.qM_fullm_i")},{
            "m.opt.timestep",flags,"m.dof_damping",true,"d.qM","m.qM_fullm_i","m.qM_fullm_j",
            "scratch.implicit.qDeriv","scratch.implicit.qDeriv"});
        append_sparse_factor_solve(out,data,"scratch.implicit.qDeriv","scratch.implicit.qLD",
            "scratch.implicit.qLDiagInv","scratch.implicit.qacc","d.efc.Ma");
        acceleration="scratch.implicit.qacc";
    }
    // No activation state exists in the current model. Preserve the full
    // activation update when its dimension is nonzero for compatible models.
    if(data.integer("m.na"))kernel(out,FORWARD,"_next_activation_",{worlds,nu},{
        "m.opt.timestep","m.actuator_dyntype","m.actuator_actadr","m.actuator_actnum","m.actuator_actlimited",
        "m.actuator_dynprm","m.actuator_actrange","d.act","d.act_dot",1.f,true,"d.act"});
    kernel(out,FORWARD,"_next_velocity_",{worlds,nv},{"m.opt.timestep","d.qvel",acceleration,1.f,"d.qvel"});
    kernel(out,FORWARD,"_next_position_",{worlds,count(data,"m.njnt")},{
        "m.opt.timestep","m.jnt_type","m.jnt_qposadr","m.jnt_dofadr","d.qpos","d.qvel",1.f,"d.qpos"});
    kernel(out,FORWARD,"_next_time_",{worlds},{
        "m.opt.timestep",true,"d.nefc","d.time","d.efc.J_rownnz","d.efc.J_rowadr",worlds,
        data.integer("d.naconmax"),data.integer("d.njmax"),data.integer("d.njmax_nnz"),"d.nacon","d.ncollision","d.time"});
    copy(out,"d.qacc_warmstart","d.qacc");
}

void append_sparse_factor_solve(ScheduleSpec& out,const ModelData& data,const std::string& matrix,
        const std::string& factor,const std::string& diagonal_inverse,const std::string& output,const std::string& rhs){
    // smooth.py:1056 _factor_i_sparse and :2757 _solve_LD_sparse. The cached
    // fused solve specializes nv and tree-level count; caller must select the
    // model-matching cached module through its catalog and validate constants.
    if(!data.integer("m.is_sparse"))throw std::runtime_error("Sparse factor-solve requires sparse model storage");
    const int worlds=count(data,"d.nworld"),nv=count(data,"m.nv"),nc=count(data,"m.nC");
    if(nv!=70||count(data,"m.qLD_updates.count")!=15)
        throw std::runtime_error("Cached sparse solve specialization requires nv=70 and 15 update levels");
    kernel(out,SMOOTH,"_copy_CSR_",{worlds,nc},{"m.mapM2M",matrix,factor});
    for(int level=count(data,"m.qLD_updates.count")-1;level>=0;--level){
        const auto updates="m.qLD_updates."+std::to_string(level);
        kernel(out,SMOOTH,"_qLD_acc_",{worlds,one_dimensional_count(data,updates)},
            {"m.M_rownnz","m.M_rowadr",updates,factor,factor});
    }
    kernel(out,SMOOTH,"_qLDiag_div_",{worlds,nv},{"m.M_rownnz","m.M_rowadr",factor,diagonal_inverse});
    const int block=count(data,"m.block_dim.solve_LD_sparse_fused");
    if(block!=64)throw std::runtime_error("Cached sparse solve module requires 64-thread blocks");
    kernel(out,SPARSE_SOLVE,"_solve_LD_sparse_fused__locals__kernel_",{worlds,block},
        {factor,diagonal_inverse,"m.qLD_all_updates","m.qLD_level_offsets",rhs,output},block);
}

ScheduleSpec build_kinematics_com(const ModelData& data){
    ScheduleSpec result;append_kinematics_com(result,data);return result;
}

} // namespace rek_mjgpu
