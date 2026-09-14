#include "collision_schedule.h"
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace rek_mjgpu {
namespace {
constexpr const char* DRIVER="wp_mujoco_warp._src.collision_driver_88f7770";
constexpr const char* NXN="wp__nxn_broadphase__locals__kernel_c763f4cb_c763f4c";
constexpr const char* PRIMITIVE="wp__primitive_narrowphase__locals__primitive_narrowphase_aae36309_aae3630";
struct ConvexPair {int first,second;const char* module;};
// Verified compiled constants in ccd_kernel entry and eval_ccd_write_contact.
// The ordering is the installed MJ_COLLISION_TABLE's ordering.
constexpr ConvexPair convex_pairs[]={
    {mjGEOM_CAPSULE,mjGEOM_CYLINDER,"wp_ccd_kernel_builder__locals__ccd_kernel_d288f67b_d288f67"},
    {mjGEOM_CYLINDER,mjGEOM_CYLINDER,"wp_ccd_kernel_builder__locals__ccd_kernel_895b360e_895b360"},
    {mjGEOM_CYLINDER,mjGEOM_BOX,"wp_ccd_kernel_builder__locals__ccd_kernel_5c6dd5d1_5c6dd5d"},
    {mjGEOM_BOX,mjGEOM_BOX,"wp_ccd_kernel_builder__locals__ccd_kernel_156d2fb1_156d2fb"}
};
constexpr int primitive_pairs[][2]={
    {mjGEOM_SPHERE,mjGEOM_SPHERE},{mjGEOM_SPHERE,mjGEOM_CAPSULE},
    {mjGEOM_SPHERE,mjGEOM_CYLINDER},{mjGEOM_SPHERE,mjGEOM_BOX},
    {mjGEOM_CAPSULE,mjGEOM_CAPSULE},{mjGEOM_CAPSULE,mjGEOM_BOX}
};
int pair_index(int first,int second){return first*(2*10-first-1)/2+second;}
void require(bool value,const std::string& message){if(!value)throw std::runtime_error("Native collision: "+message);}
void scratch(ModelData& data,const char* name,std::initializer_list<int> shape,Element element=Element::F32,int components=1){
    if(!data.contains(name)){data.allocate(name,std::vector<int>(shape),element,components);return;}
    const auto& storage=data.arrays().at(name);
    require(storage.element==element&&storage.components==components,"scratch type mismatch "+std::string(name));
    require(storage.view.ndim==int(shape.size()),"scratch rank mismatch "+std::string(name));
    int index=0;for(int dimension:shape)require(storage.view.shape[index++]==dimension,"scratch shape mismatch "+std::string(name));
}
void zero(ScheduleSpec& out,const char* name){
    ScheduleNode node;node.kind=ScheduleNode::Kind::Zero;node.destination=name;out.push_back(std::move(node));
}
void launch(ScheduleSpec& out,const char* module,const char* prefix,std::initializer_list<int> dimensions,
            std::vector<ScheduleParameter> parameters){
    const auto bounds=launch_bounds(dimensions);if(!bounds.size)return;
    ScheduleNode node;node.module=module;node.entry_prefix=prefix;node.bounds=bounds;
    node.parameters=std::move(parameters);out.push_back(std::move(node));
}
void append_contact_outputs(std::vector<ScheduleParameter>& parameters){
    for(const char* name:{"d.contact.dist","d.contact.pos","d.contact.frame","d.contact.includemargin",
        "d.contact.friction","d.contact.solref","d.contact.solreffriction","d.contact.solimp",
        "d.contact.dim","d.contact.geom","d.contact.efc_address","d.contact.worldid",
        "d.contact.type","d.contact.geomcollisionid","d.nacon"})parameters.emplace_back(name);
}
std::vector<ScheduleParameter> convex_parameters(const ModelData& data){
    // collision_convex.py:1262, inputs followed by contact outputs.
    std::vector<ScheduleParameter> result={
        "m.opt.ccd_tolerance","m.geom_type","m.geom_condim","m.geom_dataid","m.geom_priority",
        "m.geom_solmix","m.geom_solref","m.geom_solimp","m.geom_size","m.geom_friction",
        "m.geom_margin","m.geom_gap","m.mesh_vertadr","m.mesh_vertnum","m.mesh_graphadr",
        "m.mesh_vert","m.mesh_graph","m.mesh_polynum","m.mesh_polyadr","m.mesh_polynormal",
        "m.mesh_polyvertadr","m.mesh_polyvertnum","m.mesh_polyvert","m.mesh_polymapadr",
        "m.mesh_polymapnum","m.mesh_polymap","m.pair_dim","m.pair_solref","m.pair_solreffriction",
        "m.pair_solimp","m.pair_margin","m.pair_gap","m.pair_friction","d.geom_xpos","d.geom_xmat",
        data.integer("d.naconmax"),data.integer("d.naccdmax"),"d.ncollision",
        "collision.pair","collision.pairid","collision.worldid",
        "collision.epa_vert","collision.epa_vert_index","collision.epa_face","collision.epa_pr",
        "collision.epa_norm2","collision.epa_horizon","collision.multiccd_polygon","collision.multiccd_clipped",
        "collision.multiccd_pnormal","collision.multiccd_pdist","collision.multiccd_idx1","collision.multiccd_idx2",
        "collision.multiccd_n1","collision.multiccd_n2","collision.multiccd_endvert",
        "collision.multiccd_face1","collision.multiccd_face2","collision.nccd"
    };
    append_contact_outputs(result);return result;
}
}

void append_collision(ScheduleSpec& out,ModelData& data){
    const int capacity=data.integer("d.naconmax"),ccd_capacity=data.integer("d.naccdmax");
    const int worlds=data.integer("d.nworld"),flags=data.integer("m.opt.disableflags");
    require(capacity>=0&&ccd_capacity>=0&&worlds>0,"invalid world/contact capacity");
    if(!capacity||(flags&(mjDSBL_CONSTRAINT|mjDSBL_CONTACT))){zero(out,"d.nacon");return;}
    require(data.integer("m.opt.broadphase")==0&&data.integer("m.opt.broadphase_filter")==11,
        "cached broadphase requires NXN and PLANE|SPHERE|OBB filtering");
    for(const char* name:{"m.geom_aabb","m.geom_rbound","m.geom_margin"})
        require(data.array(name).shape[0]==1,"cached broadphase requires one shared model array: "+std::string(name));
    for(const char* name:{"m.nmesh","m.nhfield","m.nflex","m.has_sdf_geom"})
        require(data.integer(name)==0,"unsupported nonzero feature "+std::string(name));
    require(!(flags&mjDSBL_NATIVECCD),"cached box-box kernel requires native CCD enabled");
    require(!(data.integer("m.opt.enableflags")&mjENBL_MULTICCD),"cached convex kernels require MULTICCD disabled");
    const auto& counts=data.geometry_pair_type_counts();require(counts.size()==55,"geometry-pair counts must use ten-type triangle");
    std::vector<bool> supported(55,false);
    for(const auto& pair:primitive_pairs)supported[pair_index(pair[0],pair[1])]=true;
    int convex_count=0,boxbox_count=counts[pair_index(mjGEOM_BOX,mjGEOM_BOX)];
    for(const auto& pair:convex_pairs){supported[pair_index(pair.first,pair.second)]=true;convex_count+=counts[pair_index(pair.first,pair.second)];}
    for(std::size_t index=0;index<counts.size();index++){
        require(counts[index]>=0,"negative geometry pair count");
        require(counts[index]==0||supported[index],"missing cached narrowphase for geometry-pair index "+std::to_string(index));
    }
    scratch(data,"collision.pair",{capacity},Element::I32,2);
    scratch(data,"collision.pairid",{capacity},Element::I32,2);
    scratch(data,"collision.worldid",{capacity},Element::I32);
    launch(out,DRIVER,"_zero_nacon_ncollision_",{1},{"d.nacon","d.ncollision"});
    launch(out,NXN,"_nxn_broadphase__locals__kernel_",{worlds,data.array("m.nxn_geom_pair_filtered").shape[0]},{
        "m.geom_type","m.geom_aabb","m.geom_rbound","m.geom_margin","m.nxn_geom_pair_filtered",
        "m.nxn_pairid_filtered","d.geom_xpos","d.geom_xmat",capacity,"d.ncollision",
        "collision.pair","collision.pairid","collision.worldid"});
    if(convex_count){
        // All four cached variants embed GJK=50/EPA=50. The installed host
        // switches EPA to16 for an exclusively box-box model; reject that
        // different specialization rather than pretending this cache matches.
        require(data.integer("m.opt.ccd_iterations")==50&&convex_count!=boxbox_count,
            "cached convex kernels require GJK/EPA iteration specialization 50/50");
        require(ccd_capacity>0,"nonzero convex pairs require CCD workspace");
        constexpr int iterations=50,max_epa_faces=5,max_epa_horizon=24;
        const int polygon=boxbox_count?4:0,mesh_degree=boxbox_count?3:0;
        scratch(data,"collision.nccd",{55},Element::I32);zero(out,"collision.nccd");
        scratch(data,"collision.epa_vert",{ccd_capacity,10+2*iterations},Element::F32,3);
        scratch(data,"collision.epa_vert_index",{ccd_capacity,10+2*iterations},Element::I32);
        scratch(data,"collision.epa_face",{ccd_capacity,6+max_epa_faces*iterations},Element::I32);
        scratch(data,"collision.epa_pr",{ccd_capacity,6+max_epa_faces*iterations},Element::F32,3);
        scratch(data,"collision.epa_norm2",{ccd_capacity,6+max_epa_faces*iterations});
        scratch(data,"collision.epa_horizon",{ccd_capacity,max_epa_horizon},Element::I32);
        scratch(data,"collision.multiccd_polygon",{ccd_capacity,2*polygon},Element::F32,3);
        scratch(data,"collision.multiccd_clipped",{ccd_capacity,2*polygon},Element::F32,3);
        scratch(data,"collision.multiccd_pnormal",{ccd_capacity,polygon},Element::F32,3);
        scratch(data,"collision.multiccd_pdist",{ccd_capacity,polygon});
        scratch(data,"collision.multiccd_idx1",{ccd_capacity,mesh_degree},Element::I32);
        scratch(data,"collision.multiccd_idx2",{ccd_capacity,mesh_degree},Element::I32);
        scratch(data,"collision.multiccd_n1",{ccd_capacity,mesh_degree},Element::F32,3);
        scratch(data,"collision.multiccd_n2",{ccd_capacity,mesh_degree},Element::F32,3);
        scratch(data,"collision.multiccd_endvert",{ccd_capacity,mesh_degree},Element::F32,3);
        scratch(data,"collision.multiccd_face1",{ccd_capacity,polygon},Element::F32,3);
        scratch(data,"collision.multiccd_face2",{ccd_capacity,polygon},Element::F32,3);
        for(const auto& pair:convex_pairs)if(counts[pair_index(pair.first,pair.second)])
            launch(out,pair.module,"ccd_kernel_builder__locals__ccd_kernel_",{capacity},convex_parameters(data));
    }
    // Primitive dispatch is one cached, internally type-dispatched kernel.
    std::vector<ScheduleParameter> primitive={
        "m.geom_type","m.geom_condim","m.geom_dataid","m.geom_priority","m.geom_solmix","m.geom_solref",
        "m.geom_solimp","m.geom_size","m.geom_friction","m.geom_margin","m.geom_gap","m.mesh_vertadr",
        "m.mesh_vertnum","m.mesh_graphadr","m.mesh_vert","m.mesh_graph","m.mesh_polynum","m.mesh_polyadr",
        "m.mesh_polynormal","m.mesh_polyvertadr","m.mesh_polyvertnum","m.mesh_polyvert","m.mesh_polymapadr",
        "m.mesh_polymapnum","m.mesh_polymap","m.pair_dim","m.pair_solref","m.pair_solreffriction",
        "m.pair_solimp","m.pair_margin","m.pair_gap","m.pair_friction","d.geom_xpos","d.geom_xmat",capacity,
        "d.ncollision","collision.pair","collision.pairid","collision.worldid"
    };
    append_contact_outputs(primitive);
    launch(out,PRIMITIVE,"_primitive_narrowphase__locals__primitive_narrowphase_",{capacity},std::move(primitive));
}
}
