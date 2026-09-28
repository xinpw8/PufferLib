#include "model_data.h"
#include <mujoco/mjxmacro.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace rek_mjgpu {
namespace {
void check(CUresult result,const char* where){if(result!=CUDA_SUCCESS){const char* error=nullptr;cuGetErrorString(result,&error);throw std::runtime_error(std::string(where)+": "+(error?error:"CUDA error"));}}
void require(bool value,const std::string& what){if(!value)throw std::runtime_error("MuJoCo GPU model/data: "+what);}
template<class T> std::vector<double> numbers(const T* values,std::size_t count){
    std::vector<double> result(count);for(std::size_t i=0;i<count;i++)result[i]=double(values[i]);return result;
}
struct Spec {const char* name;const char* dimensions;Element element;int components;};
const Spec specs[]={
#include "model_fields.inc"
};
std::vector<std::string> split(const std::string& value){std::vector<std::string> out;std::stringstream in(value);std::string part;while(std::getline(in,part,','))out.push_back(part);return out;}
std::size_t width(Element e){return e==Element::U8?1:4;}
}

ModelData::ModelData(const mjModel* model,const ModelDataConfig& config){
    try{build(model,config);}catch(...){for(auto& entry:arrays_)if(entry.second.owns_memory&&entry.second.view.data)cuMemFree(reinterpret_cast<CUdeviceptr>(entry.second.view.data));throw;}
}
ModelData::~ModelData(){for(auto& entry:arrays_)if(entry.second.owns_memory&&entry.second.view.data)cuMemFree(reinterpret_cast<CUdeviceptr>(entry.second.view.data));}
WarpArray ModelData::array(const std::string& name)const{return arrays_.at(name).view;}
const WarpArray& ModelData::array_ref(const std::string& name)const{return arrays_.at(name).view;}
const int& ModelData::integer_ref(const std::string& name)const{return integers_.at(name);}
const float& ModelData::scalar_ref(const std::string& name)const{return scalars_.at(name);}
int ModelData::integer(const std::string& name)const{return integers_.at(name);}
float ModelData::scalar(const std::string& name)const{return scalars_.at(name);}
std::size_t ModelData::byte_size(const std::string& name)const{return arrays_.at(name).bytes;}
bool ModelData::contains(const std::string& name)const{return arrays_.count(name)||integers_.count(name)||scalars_.count(name);}
WarpArray ModelData::allocate(const std::string& name,const std::vector<int>& shape,Element element,int components,bool broadcast){
    require(!arrays_.count(name),"duplicate array "+name);require(!shape.empty()&&shape.size()<=4&&components>0,"invalid shape "+name);
    ArrayStorage storage;storage.element=element;storage.components=components;storage.view.ndim=int(shape.size());
    std::size_t stride=width(element)*components;
    for(int i=int(shape.size())-1;i>=0;i--){require(shape[i]>=0&&stride<=INT32_MAX,"invalid byte stride "+name);storage.view.shape[i]=shape[i];storage.view.strides[i]=int(stride);require(shape[i]==0||stride<=SIZE_MAX/std::size_t(shape[i]),"array size overflow");stride*=shape[i];}
    storage.bytes=stride;if(broadcast)storage.view.strides[0]=0;
    if(storage.bytes){CUdeviceptr address;check(cuMemAlloc(&address,storage.bytes),"allocate model/data");storage.view.data=reinterpret_cast<void*>(address);try{check(cuMemsetD8(address,0,storage.bytes),"zero model/data");}catch(...){cuMemFree(address);throw;}}
    arrays_.emplace(name,storage);return storage.view;
}
WarpArray ModelData::reshape_alias(const std::string& name,const std::string& source,const std::vector<int>& shape){
    require(!arrays_.count(name),"duplicate alias "+name);require(!shape.empty()&&shape.size()<=4,"invalid alias rank "+name);
    const auto& original=arrays_.at(source);ArrayStorage alias=original;alias.owns_memory=false;
    std::size_t stride=width(original.element)*original.components;
    for(int i=original.view.ndim-1;i>=0;i--){require(original.view.strides[i]==int(stride),"alias source is not contiguous "+source);stride*=original.view.shape[i];}
    alias.view={};alias.view.data=original.view.data;alias.view.grad=original.view.grad;alias.view.ndim=int(shape.size());
    stride=width(original.element)*original.components;
    for(int i=int(shape.size())-1;i>=0;i--){require(shape[i]>=0&&stride<=INT32_MAX,"invalid alias byte stride "+name);alias.view.shape[i]=shape[i];alias.view.strides[i]=int(stride);require(shape[i]==0||stride<=SIZE_MAX/std::size_t(shape[i]),"alias size overflow");stride*=shape[i];}
    require(stride==original.bytes,"alias byte size differs "+name);arrays_.emplace(name,alias);return alias.view;
}
void ModelData::upload(const std::string& name,const void* data,std::size_t bytes){auto& entry=arrays_.at(name);require(bytes==entry.bytes,"upload size "+name);if(bytes)check(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(entry.view.data),data,bytes),"upload model/data");}
void ModelData::zero(const std::string& name,CUstream stream){auto& entry=arrays_.at(name);if(entry.bytes)check(cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(entry.view.data),0,entry.bytes,stream),"zero model/data");}
void ModelData::set_initial_state(const float* qp,const float* qv){
    require(qp!=nullptr,"missing initial qpos");const int worlds=integer("d.nworld"),nq=integer("m.nq"),nv=integer("m.nv");
    std::vector<float> pos(std::size_t(worlds)*nq),vel(std::size_t(worlds)*nv);
    for(int w=0;w<worlds;w++){std::copy(qp,qp+nq,pos.begin()+std::size_t(w)*nq);if(qv)std::copy(qv,qv+nv,vel.begin()+std::size_t(w)*nv);}
    for(float x:pos)require(std::isfinite(x),"nonfinite initial position");for(float x:vel)require(std::isfinite(x),"nonfinite initial velocity");
    upload("d.qpos",pos.data(),pos.size()*4);upload("d.qvel",vel.data(),vel.size()*4);
}

void ModelData::build(const mjModel* m,const ModelDataConfig& config){
    require(m&&config.nworld>0&&config.nconmax>=0&&config.njmax>0,"invalid model/capacity");
    require(m->nmesh==0&&m->nhfield==0&&m->nflex==0&&m->ntendon==0&&m->neq==0&&m->nplugin==0&&m->nmocap==0&&m->nsensor==0,
            "initial native integration supports rigid primitive models without sensors/tendons/equalities/plugins/mocap");
    require(m->opt.integrator==mjINT_IMPLICITFAST&&m->opt.solver==mjSOL_NEWTON&&m->opt.cone==mjCONE_ELLIPTIC,"requires pinned implicitfast/Newton/elliptic configuration");
    require(m->opt.noslip_iterations==0&&m->opt.viscosity==0&&m->opt.density==0,"unsupported fluid/noslip option");
    for(int i=0;i<m->nbody;i++)require(m->body_plugin[i]==-1,"body plugin");
    for(int i=0;i<m->nu;i++)require(m->actuator_plugin[i]==-1,"actuator plugin");
    const bool sparse=m->opt.jacobian==mjJAC_AUTO?m->nv>32:m->opt.jacobian==mjJAC_SPARSE;
    require(sparse,"initial native integration requires sparse model");
    std::map<std::string,std::vector<double>> source;
#define X(name) const int name=int(m->name); integers_["m." #name]=name;
    MJMODEL_SIZES
#undef X
#define X(type,name,rows,cols) source["m." #name]=numbers(m->name,std::size_t(rows)*std::size_t(cols));
    MJMODEL_POINTERS
#undef X
#define X(type,name,count) source["m.opt." #name]=numbers(&m->opt.name,count); scalars_["m.opt." #name]=float(m->opt.name); if constexpr(std::is_integral<type>::value)integers_["m.opt." #name]=int(m->opt.name);
#define XVEC(type,name,count) source["m.opt." #name]=numbers(m->opt.name,count);
    MJOPTION_FIELDS
#undef XVEC
#undef X
    source["m.opt.tolerance"]={std::max(double(m->opt.tolerance),1e-6)};
    scalars_["m.opt.tolerance"]=float(source["m.opt.tolerance"][0]);
    source["m.opt.impratio_invsqrt"]={1/std::sqrt(std::max(double(m->opt.impratio),double(mjMINVAL)))};
    source["m.stat.meaninertia"]={m->stat.meaninertia};
    integers_["m.opt.ls_parallel"]=0;
    int numeric=mj_name2id(m,mjOBJ_NUMERIC,"ls_parallel");if(numeric>=0)integers_["m.opt.ls_parallel"]=m->numeric_data[m->numeric_adr[numeric]]==1;
    integers_["m.opt.broadphase"]=0;integers_["m.opt.broadphase_filter"]=11;
    integers_["m.opt.graph_conditional"]=1;integers_["m.opt.run_collision_detection"]=1;integers_["m.opt.contact_sensor_maxmatch"]=64;
    scalars_["m.opt.ls_parallel_min_step"]=1e-6f;
    // Installed MuJoCo-Warp 3.7.0.1 BlockDim specialization defaults. Cached
    // tiled kernels require the matching launch width, including on SM 12.1.
    for(const auto& block:std::vector<std::pair<const char*,int>>{
        {"segmented_sort",128},{"euler_dense",32},{"actuator_velocity",32},
        {"ray",64},{"contact_sort",64},{"energy_vel_kinetic",32},
        {"cholesky_factorize",32},{"cholesky_solve",32},{"cholesky_factorize_solve",32},
        {"solve_LD_sparse_fused",64},{"update_gradient_cholesky",64},
        {"update_gradient_cholesky_blocked",32},{"update_gradient_JTDAJ_sparse",64},
        {"update_gradient_JTDAJ_dense",96},{"linesearch_iterative",32},{"qderiv_actuator_dense",32}})
        integers_[std::string("m.block_dim.")+block.first]=block.second;
    integers_["m.is_sparse"]=sparse;integers_["m.nv_pad"]=(nv+15)/16*16;
    integers_["m.nmaxcondim"]=0;for(int i=0;i<ngeom;i++)integers_["m.nmaxcondim"]=std::max(integer("m.nmaxcondim"),m->geom_condim[i]);
    for(int i=0;i<npair;i++)integers_["m.nmaxcondim"]=std::max(integer("m.nmaxcondim"),m->pair_dim[i]);
    integers_["m.nmaxpyramid"]=std::max(1,2*(integer("m.nmaxcondim")-1));
    for(const char* field:{"nacttrnbody","nsensorcollision","nsensortaxel","nsensorcontact","nrangefinder","nmaxpolygon","nmaxmeshdeg","has_fluid","has_sdf_geom","max_ten_J_rownnz","sensor_e_potential","sensor_e_kinetic","sensor_subtree_vel","sensor_rne_postconstraint"})integers_[std::string("m.")+field]=0;
    for(int i=0;i<nu;i++)require(m->actuator_trntype[i]!=mjTRN_BODY,"body actuator transmission unsupported");
    require(m->opt.wind[0]==0&&m->opt.wind[1]==0&&m->opt.wind[2]==0,"wind unsupported");
    require(std::int64_t(config.nworld)*config.nconmax<=INT32_MAX,"contact capacity overflow");
    integers_["d.nworld"]=config.nworld;integers_["d.naconmax"]=config.nworld*config.nconmax;integers_["d.naccdmax"]=integer("d.naconmax");
    integers_["d.njmax"]=config.njmax;integers_["d.njmax_pad"]=(config.njmax+15)/16*16;
    integers_["d.njmax_nnz"]=config.njmax_nnz>0?config.njmax_nnz:config.njmax*nv;
    std::map<int,std::vector<double>> levels;std::vector<int> depth(nbody,-1),children(nbody);
    for(int i=0;i<nbody;i++){depth[i]=depth[m->body_parentid[i]]+1;levels[depth[i]].push_back(i);if(i)children[m->body_parentid[i]]++;}
    integers_["m.body_tree.count"]=int(levels.size());
    std::vector<double> branch,branch_start{0};int branch_count=0;
    for(int i=1;i<nbody;i++)if(!children[i]){std::vector<int> chain;for(int b=i;b;b=m->body_parentid[b])chain.push_back(b);for(auto b=chain.rbegin();b!=chain.rend();b++)branch.push_back(*b);branch_start.push_back(branch.size());branch_count++;}
    integers_["m.nbranch"]=branch_count;source["m.body_branches"]=branch;source["m.body_branch_start"]=branch_start;
    for(const auto& level:levels)source["m.body_tree."+std::to_string(level.first)]=level.second;
    source["m.body_fluid_ellipsoid"]=std::vector<double>(nbody,0);source["m.geom_plugin_index"]=std::vector<double>(ngeom,-1);
    for(int j=0;j<njnt;j++)if(m->jnt_limited[j]){if(m->jnt_type[j]==mjJNT_HINGE||m->jnt_type[j]==mjJNT_SLIDE)source["m.jnt_limited_slide_hinge_adr"].push_back(j);else if(m->jnt_type[j]==mjJNT_BALL)source["m.jnt_limited_ball_adr"].push_back(j);}
    for(int i=0;i<nv;i++)for(int j=0;j<=i;j++){source["m.dof_tri_row"].push_back(i);source["m.dof_tri_col"].push_back(j);}
    std::set<int> exclusions(m->exclude_signature,m->exclude_signature+nexclude);pair_type_counts_.assign(55,0);
    for(int g1=0;g1<ngeom;g1++)for(int g2=g1+1;g2<ngeom;g2++){
        const int b1=m->geom_bodyid[g1],b2=m->geom_bodyid[g2],w1=m->body_weldid[b1],w2=m->body_weldid[b2];
        bool parent=!(m->opt.disableflags&mjDSBL_FILTERPARENT)&&w1&&w2&&(w1==m->body_weldid[m->body_parentid[w2]]||w2==m->body_weldid[m->body_parentid[w1]]);
        bool include=((m->geom_contype[g1]&m->geom_conaffinity[g2])||(m->geom_contype[g2]&m->geom_conaffinity[g1]))&&w1!=w2&&!parent&&!exclusions.count((b1<<16)+b2);
        int pid=include?-1:-2;for(int p=0;p<npair;p++)if((m->pair_geom1[p]==g1&&m->pair_geom2[p]==g2)||(m->pair_geom1[p]==g2&&m->pair_geom2[p]==g1)){pid=p;include=true;}
        auto& pairs=source["m.nxn_geom_pair"];pairs.push_back(g1);pairs.push_back(g2);auto& ids=source["m.nxn_pairid"];ids.push_back(pid);ids.push_back(-1);
        if(include){auto& pairs_f=source["m.nxn_geom_pair_filtered"];pairs_f.push_back(g1);pairs_f.push_back(g2);auto& ids_f=source["m.nxn_pairid_filtered"];ids_f.push_back(pid);ids_f.push_back(-1);int a=m->geom_type[g1],b=m->geom_type[g2];if(a>b)std::swap(a,b);int index=a*(2*10-a-1)/2+b;require(index<int(pair_type_counts_.size()),"geometry type outside Warp enumeration");pair_type_counts_[index]++;}
    }
    std::map<int,std::vector<double>> tiles,updates;std::vector<int> corners,dof_depth(nv,-1);
    for(int i=0;i<nv;i++)if(m->dof_parentid[i]==-1)corners.push_back(i);
    for(std::size_t i=0;i<corners.size();i++){int size=(i+1==corners.size()?nv:corners[i+1])-corners[i];tiles[size].push_back(corners[i]);}
    integers_["m.qM_tiles.count"]=int(tiles.size());int ti=0;for(auto& tile:tiles){std::string key="m.qM_tiles."+std::to_string(ti++);source[key+".adr"]=tile.second;integers_[key+".size"]=tile.first;}
    for(int k=0;k<nv;k++){if(m->M_rownnz[k]==1)continue;dof_depth[k]=(m->dof_parentid[k]>=0?dof_depth[m->dof_parentid[k]]:-1)+1;int adr=m->M_rowadr[k]+m->M_rownnz[k]-2;for(int i=m->dof_parentid[k];i>=0;i=m->dof_parentid[i],adr--){auto& row=updates[dof_depth[i]];row.push_back(i);row.push_back(k);row.push_back(adr);}}
    integers_["m.qLD_updates.count"]=int(updates.size());ti=0;source["m.qLD_level_offsets"]={0};
    for(auto& level:updates){source["m.qLD_updates."+std::to_string(ti++)]=level.second;auto& all=source["m.qLD_all_updates"];all.insert(all.end(),level.second.begin(),level.second.end());source["m.qLD_level_offsets"].push_back(all.size()/3);}
    if(source["m.qLD_all_updates"].empty())source["m.qLD_all_updates"]={0,0,0};
    std::vector<std::vector<std::pair<int,int>>> rows(nv);
    for(int i=0;i<nv;i++){rows[i].push_back({i,m->dof_Madr[i]});int adr=m->dof_Madr[i]+1;for(int j=m->dof_parentid[i];j>=0;j=m->dof_parentid[j],adr++){rows[i].push_back({j,adr});rows[j].push_back({i,adr});}for(int j=i;j>=0;j=m->dof_parentid[j]){source["m.qM_fullm_i"].push_back(i);source["m.qM_fullm_j"].push_back(j);}}
    source["m.qM_mulm_rowadr"]={0};for(auto& row:rows){for(auto [col,adr]:row){source["m.qM_mulm_col"].push_back(col);source["m.qM_mulm_madr"].push_back(adr);}source["m.qM_mulm_rowadr"].push_back(source["m.qM_mulm_col"].size());}
    auto send=[&](const std::string& key,const std::vector<int>& shape,Element element,int components,const std::vector<double>* values,bool broadcast=false){
        allocate(key,shape,element,components,broadcast);auto bytes=byte_size(key);if(!values)return;require(values->size()*width(element)==bytes,"source size differs for "+key);
        if(element==Element::F32){std::vector<float> v(values->begin(),values->end());upload(key,v.data(),bytes);}else if(element==Element::I32){std::vector<int> v;for(double x:*values){require(x>=INT32_MIN&&x<=INT32_MAX&&x==std::trunc(x),"integer conversion "+key);v.push_back(int(x));}upload(key,v.data(),bytes);}else{std::vector<unsigned char> v;for(double x:*values){require(x==0||x==1,"boolean conversion "+key);v.push_back(x!=0);}upload(key,v.data(),bytes);}
    };
    for(const Spec& spec:specs){std::string key=spec.name;std::vector<int> shape;bool broadcast=false;for(const auto& dim:split(spec.dimensions)){if(dim=="*"){shape.push_back(1);broadcast=shape.size()==1;}else if(std::isdigit(dim[0]))shape.push_back(std::stoi(dim));else if(integers_.count("d."+dim))shape.push_back(integer("d."+dim));else shape.push_back(integer("m."+dim));}
        if(key.rfind("m.",0)==0){auto it=source.find(key);if(it==source.end()){std::size_t count=1;for(int n:shape)count*=n;require(count==0,"missing model field "+key);source[key]={};it=source.find(key);}send(key,shape,spec.element,spec.components,&it->second,broadcast);}else send(key,shape,spec.element,spec.components,nullptr,broadcast);
    }
    for(const auto& item:source)if(!contains(item.first)){
        const auto& key=item.first;if(key.rfind("m.body_tree.",0)==0||key.rfind("m.qLD_updates.",0)==0||key.rfind("m.qM_tiles.",0)==0||key.rfind("m.qM_",0)==0||key.rfind("m.qLD_",0)==0||key.rfind("m.nxn_",0)==0||key=="m.body_branches"||key=="m.body_branch_start"||key=="m.jnt_limited_slide_hinge_adr"||key=="m.jnt_limited_ball_adr"||key=="m.dof_tri_row"||key=="m.dof_tri_col"){
            int components=key.rfind("m.nxn_",0)==0?2:(key.rfind("m.qLD_updates.",0)==0||key=="m.qLD_all_updates"?3:1);send(key,{int(item.second.size()/components)},Element::I32,components,&item.second);
        }
    }
    // These variable-length address lists are exactly empty for rejected/absent features.
    for(const char* name:{"jnt_limited_ball_adr","eq_connect_adr","eq_wld_adr","eq_jnt_adr","eq_ten_adr","eq_flex_adr","tendon_jnt_adr","tendon_site_pair_adr","tendon_geom_adr","tendon_limited_adr","ten_wrapnum_site","wrap_jnt_adr","wrap_site_adr","wrap_site_pair_adr","wrap_geom_adr","actuator_trntype_body_adr","sensor_pos_adr","sensor_limitpos_adr","sensor_vel_adr","sensor_limitvel_adr","sensor_acc_adr","sensor_rangefinder_adr","rangefinder_sensor_adr","sensor_collision_start_adr","sensor_touch_adr","sensor_limitfrc_adr","sensor_tendonactfrc_adr","taxel_sensorid"})if(!contains(std::string("m.")+name))allocate(std::string("m.")+name,{0},Element::I32);
    std::vector<double> one_zero{0};send("m.ten_wrapadr_site",{1},Element::I32,1,&one_zero);
    allocate("d.qM",{config.nworld,1,nM});allocate("d.qLD",{config.nworld,1,nC});
    allocate("d.efc.J_rownnz",{config.nworld,config.njmax},Element::I32);allocate("d.efc.J_rowadr",{config.nworld,config.njmax},Element::I32);
    allocate("d.efc.J_colind",{config.nworld,1,integer("d.njmax_nnz")},Element::I32);allocate("d.efc.J",{config.nworld,1,integer("d.njmax_nnz")});
    std::vector<float> initial(m->qpos0,m->qpos0+nq);set_initial_state(initial.data());
    // Constants for world-attached geometry. Dynamic transforms are computed by
    // the GPU kinematics phase, never by a CPU forward/step call.
    std::vector<float> quats(std::size_t(config.nworld)*nbody*4),matrices(std::size_t(config.nworld)*nbody*9);
    std::vector<float> geompos(std::size_t(config.nworld)*ngeom*3),geommat(std::size_t(config.nworld)*ngeom*9);
    for(int w=0;w<config.nworld;w++){for(int b=0;b<nbody;b++){quats[(w*nbody+b)*4]=1;for(int k=0;k<3;k++)matrices[(w*nbody+b)*9+k*4]=1;}
        for(int g=0;g<ngeom;g++)if(m->body_weldid[m->geom_bodyid[g]]==0){require(m->geom_bodyid[g]==0,"static geom attached to non-world body needs constant pose composition");mjtNum rotation[9];mju_quat2Mat(rotation,m->geom_quat+4*g);for(int k=0;k<3;k++)geompos[(w*ngeom+g)*3+k]=float(m->geom_pos[g*3+k]);for(int k=0;k<9;k++)geommat[(w*ngeom+g)*9+k]=float(rotation[k]);}}
    upload("d.xquat",quats.data(),quats.size()*4);upload("d.xmat",matrices.data(),matrices.size()*4);upload("d.ximat",matrices.data(),matrices.size()*4);
    upload("d.geom_xpos",geompos.data(),geompos.size()*4);upload("d.geom_xmat",geommat.data(),geommat.size()*4);
}
}
