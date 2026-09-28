// Constraint launch order and argument lists adapted from MuJoCo-Warp,
// Copyright 2025 The Newton Developers, Apache License 2.0.
#include "constraint_schedule.h"
#include <algorithm>
#include <stdexcept>

namespace rek_mjgpu {
namespace {
void launch(ScheduleSpec& out,const char* entry,std::initializer_list<int> shape,
            std::vector<ScheduleParameter> parameters){
    ScheduleNode node;node.module="wp_mujoco_warp._src.constraint_c567367";
    node.entry_prefix=entry;node.bounds=launch_bounds(shape);node.parameters=std::move(parameters);
    out.push_back(std::move(node));
}
void outputs(std::vector<ScheduleParameter>& args){
    for(const char* name:{"d.efc.type","d.efc.id","d.efc.J_rownnz","d.efc.J_rowadr",
        "d.efc.J_colind","d.efc.J","d.efc.pos","d.efc.margin","d.efc.D",
        "d.efc.vel","d.efc.aref","d.efc.frictionloss","constraint.efc_nnz"})args.emplace_back(name);
}
std::vector<int> counts(const ModelData& data,const char* field){
    const auto& storage=data.arrays().at(field);
    if(storage.element!=Element::I32||storage.components!=1)throw std::runtime_error("Constraint counter type mismatch");
    std::vector<int> result(storage.bytes/sizeof(int));
    if(storage.bytes&&cuMemcpyDtoH(result.data(),reinterpret_cast<CUdeviceptr>(storage.view.data),storage.bytes)!=CUDA_SUCCESS)
        throw std::runtime_error("Constraint counter GPU download failed");
    return result;
}
}
void append_constraints(ScheduleSpec& out,ModelData& data){
    if(data.integer("m.neq")||data.integer("m.ntendon")||data.integer("m.nflex")||
       data.array("m.jnt_limited_ball_adr").shape[0])
        throw std::runtime_error("Native constraint schedule does not cover equality/tendon/flex/ball-limit models");
    if(data.integer("m.opt.cone")!=mjCONE_ELLIPTIC||!data.integer("m.is_sparse"))
        throw std::runtime_error("Native constraints require sparse elliptic model");
    const int worlds=data.integer("d.nworld"),nv=data.integer("m.nv"),flags=data.integer("m.opt.disableflags");
    const int njmax=data.integer("d.njmax"),nnz=data.integer("d.njmax_nnz");
    if(!data.contains("constraint.efc_nnz"))data.allocate("constraint.efc_nnz",{worlds},Element::I32);
    launch(out,"_zero_constraint_counts_",{worlds},{"d.ne","d.nf","d.nl","d.nefc","constraint.efc_nnz"});
    if(flags&mjDSBL_CONSTRAINT)return;
    if(!(flags&mjDSBL_FRICTIONLOSS)){
        std::vector<ScheduleParameter> args={nv,"m.opt.timestep",flags,"m.dof_solref","m.dof_solimp",
            "m.dof_frictionloss","m.dof_invweight0",true,"d.qvel",njmax,nnz,"d.nf","d.nefc"};
        outputs(args);launch(out,"_friction_dof_",{worlds,nv},std::move(args));
    }
    if(!(flags&mjDSBL_LIMIT)){
        std::vector<ScheduleParameter> args={nv,"m.opt.timestep",flags,"m.jnt_qposadr","m.jnt_dofadr",
            "m.jnt_solref","m.jnt_solimp","m.jnt_range","m.jnt_margin","m.dof_invweight0",true,
            "m.jnt_limited_slide_hinge_adr","d.qpos","d.qvel",njmax,nnz,"d.nl","d.nefc"};
        outputs(args);launch(out,"_limit_slide_hinge_",{worlds,data.array("m.jnt_limited_slide_hinge_adr").shape[0]},std::move(args));
    }
    if(!(flags&mjDSBL_CONTACT)){
        std::vector<ScheduleParameter> args={nv,"m.opt.timestep",flags,"m.opt.impratio_invsqrt",
            "m.body_parentid","m.body_rootid","m.body_weldid","m.body_dofnum","m.body_dofadr",
            "m.body_invweight0","m.dof_bodyid","m.dof_parentid","m.geom_bodyid","m.flex_vertadr",
            "m.flex_vertbodyid",true,"d.qvel","d.subtree_com","d.cdof",njmax,nnz,"d.nacon",
            "d.contact.dist","d.contact.dim","d.contact.includemargin","d.contact.worldid",
            "d.contact.geom","d.contact.flex","d.contact.vert","d.contact.pos","d.contact.frame",
            "d.contact.friction","d.contact.solref","d.contact.solreffriction","d.contact.solimp",
            "d.contact.type","d.nefc","d.contact.efc_address"};
        outputs(args);launch(out,"_contact_elliptic_",{data.integer("d.naconmax"),data.integer("m.nmaxcondim")},std::move(args));
    }
}
ConstraintCapacityReport inspect_constraint_capacity(const ModelData& data,CUstream stream){
    if(cuStreamSynchronize(stream)!=CUDA_SUCCESS)throw std::runtime_error("Constraint diagnostic synchronization failed");
    ConstraintCapacityReport result;result.collisions=counts(data,"d.ncollision").at(0);result.contacts=counts(data,"d.nacon").at(0);
    const auto rows=counts(data,"d.nefc"),nonzeros=counts(data,"constraint.efc_nnz");
    result.overflow=result.collisions<0||result.contacts<0||result.collisions>data.integer("d.naconmax")||result.contacts>data.integer("d.naconmax");
    for(int world=0;world<data.integer("d.nworld");world++){
        result.max_rows=std::max(result.max_rows,rows.at(world));result.max_nonzeros=std::max(result.max_nonzeros,nonzeros.at(world));
        if(rows[world]<0||rows[world]>data.integer("d.njmax")||nonzeros[world]<0||nonzeros[world]>data.integer("d.njmax_nnz")){
            result.overflow=true;if(result.first_invalid_world<0)result.first_invalid_world=world;
        }
    }
    return result;
}
}
