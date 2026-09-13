#include "physics.cuh"
#include "../../../vendor/cJSON.h"
#include <openssl/evp.h>
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>

// Compile the exact baseline solver in this translation unit. Nothing in the
// native5 environment header imports B3 declarations or its global symbols.
#ifndef B3_ART_CONTACTS
#define B3_ART_CONTACTS 0
#endif
#if B3_ART_CONTACTS != 0
#error "native5 physics requires the approved independent-contact baseline"
#endif
namespace rek5_native {
#include "../puffysics_prototype/puffysics_semantic_native.cu"
}

namespace rek5 {
namespace {

using Json = const cJSON*;
using JsonOwner = std::unique_ptr<cJSON, decltype(&cJSON_Delete)>;
using DataOwner = std::unique_ptr<mjData, decltype(&mj_deleteData)>;
using Pose = std::array<double,7>;

void require(bool condition, const std::string& description) {
    if (!condition) throw std::runtime_error("native5 physics: " + description);
}

void cuda_check(cudaError_t result, const char* operation) {
    require(result == cudaSuccess, std::string(operation) + ": " + cudaGetErrorString(result));
}

void native_check(int result, const char* operation) {
    require(result == 0, std::string(operation) + ": " + rek5_native::rp_last_error());
}

Json member(Json object, const char* key) {
    require(cJSON_IsObject(object), std::string("object required for ") + key);
    Json value = cJSON_GetObjectItemCaseSensitive(object, key);
    require(value != nullptr, std::string("missing export field ") + key);
    return value;
}

Json item(Json array, int index) {
    require(cJSON_IsArray(array), "array required");
    Json value = cJSON_GetArrayItem(array, index);
    require(value != nullptr, "array index outside export");
    return value;
}

int count(Json array) {
    require(cJSON_IsArray(array), "array required");
    return cJSON_GetArraySize(array);
}

double number(Json value) {
    require(cJSON_IsNumber(value) && std::isfinite(value->valuedouble), "finite number required");
    return value->valuedouble;
}

double number(Json object, const char* key) { return number(member(object,key)); }

int integer(Json value) {
    const double value_double = number(value);
    require(value_double >= std::numeric_limits<int>::min()
            && value_double <= std::numeric_limits<int>::max()
            && std::trunc(value_double) == value_double, "integer required");
    return static_cast<int>(value_double);
}

int integer(Json object, const char* key) { return integer(member(object,key)); }

bool boolean(Json object, const char* key) {
    Json value = member(object,key);
    require(cJSON_IsBool(value), std::string("Boolean required for ") + key);
    return cJSON_IsTrue(value);
}

std::string string(Json value) {
    require(cJSON_IsString(value) && value->valuestring, "string required");
    return value->valuestring;
}

std::string string(Json object, const char* key) { return string(member(object,key)); }

std::vector<double> numbers(Json array, int size) {
    require(count(array) == size, "export array dimension mismatch");
    std::vector<double> output(size);
    for (int i=0;i<size;++i) output[i] = number(item(array,i));
    return output;
}

std::vector<double> numbers(Json object, const char* key, int size) {
    return numbers(member(object,key),size);
}

void append(std::vector<float>& output, double value) {
    const float packed = static_cast<float>(value);
    require(std::isfinite(packed), "export value exceeds finite float32 range");
    output.push_back(packed);
}

template<typename Container>
void append(std::vector<float>& output, const Container& values) {
    for (double value : values) append(output,value);
}

Pose pose(Json object) {
    auto p = numbers(object,"position_xyz",3);
    auto q = numbers(object,"quaternion_xyzw",4);
    double norm=0;
    for (double value:q) norm+=value*value;
    require(std::abs(norm-1.0) <= 1e-7, "export quaternion must be unit length");
    return {p[0],p[1],p[2],q[0],q[1],q[2],q[3]};
}

std::array<double,4> multiply_xyzw(const double* a, const double* b) {
    return {a[3]*b[0]+b[3]*a[0]+a[1]*b[2]-a[2]*b[1],
            a[3]*b[1]+b[3]*a[1]+a[2]*b[0]-a[0]*b[2],
            a[3]*b[2]+b[3]*a[2]+a[0]*b[1]-a[1]*b[0],
            a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2]};
}

std::array<double,9> quat_matrix(const double* q) {
    const double x=q[0], y=q[1], z=q[2], w=q[3];
    return {1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
            2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
            2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)};
}

std::array<double,3> local_position(const Pose& owner, const mjtNum* position) {
    const auto rotation=quat_matrix(owner.data()+3);
    std::array<double,3> output{};
    for (int c=0;c<3;++c)
        for (int r=0;r<3;++r) output[c]+=rotation[r*3+c]*(position[r]-owner[r]);
    return output;
}

std::array<double,4> local_rotation(const Pose& owner, const mjtNum* rotation) {
    const auto parent=quat_matrix(owner.data()+3);
    mjtNum local[9]={}, q[4];
    for (int r=0;r<3;++r)
        for (int c=0;c<3;++c)
            for (int k=0;k<3;++k) local[r*3+c]+=parent[k*3+r]*rotation[k*3+c];
    mju_mat2Quat(q,local);
    // matrix_quat in the existing exporter chooses the hemisphere w >= 0.
    if (q[0]<0) for (mjtNum& value:q) value=-value;
    return {q[1],q[2],q[3],q[0]};
}

std::string read_file(const char* path) {
    require(path && *path, "model/export path is empty");
    std::ifstream input(path,std::ios::binary|std::ios::ate);
    require(input.good(),std::string("cannot read ")+path);
    const std::streamoff length=input.tellg();
    require(length>0 && length<=16*1024*1024,"model/export file size outside bound");
    std::string output(static_cast<size_t>(length),'\0');
    input.seekg(0);
    input.read(&output[0],length);
    require(input.good(),std::string("incomplete read of ")+path);
    return output;
}

std::string sha256(const std::string& bytes) {
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int size=0;
    require(EVP_Digest(bytes.data(),bytes.size(),digest,&size,EVP_sha256(),nullptr)==1
            && size==32,"SHA-256 calculation failed");
    static const char hex[]="0123456789abcdef";
    std::string output(64,'0');
    for (int i=0;i<32;++i) { output[i*2]=hex[digest[i]>>4]; output[i*2+1]=hex[digest[i]&15]; }
    return output;
}

void ordered(Json values, int expected, const char* label) {
    require(count(values)==expected,std::string(label)+" count mismatch");
    for (int i=0;i<expected;++i)
        require(integer(item(values,i),"id")==i,std::string(label)+" IDs must be dense and ordered");
}

void pack_export(Physics& output, Json root) {
    require(string(root,"schema")=="rek.puffysics.compiled_model_adapter.v1","adapter schema mismatch");
    require(string(root,"coordinate_system")=="right_handed_z_up","world frame mismatch");
    require(string(root,"quaternion_order")=="xyzw","quaternion order mismatch");
    Json counts=member(root,"counts");
    const char* names[]={"source_bodies","rigid_bodies","shapes","hinges","free_bases","actuators","nq","nv"};
    const int expected[]={63,60,91,58,2,58,72,70};
    for (int i=0;i<8;++i) require(integer(counts,names[i])==expected[i],std::string("count mismatch: ")+names[i]);
    require(number(root,"runtime_timestep_seconds")==0.002,"physics timestep mismatch");
    const auto gravity=numbers(root,"gravity_xyz_m_s2",3);
    require(float(gravity[0])==0 && float(gravity[1])==0 && float(gravity[2])==float(-9.81),"gravity mismatch");
    for (double velocity:numbers(root,"initial_qvel",70)) require(velocity==0,"initial velocity must be zero");
    for (const char* field:{"source_exclude_signature","source_pair_geom1","source_pair_geom2"})
        require(count(member(root,field))==0,std::string("unsupported nonempty ")+field);
    require(string(member(root,"initial_pose"),"kind")=="semantic_idle_frame_zero_clipped","initial pose mismatch");
    output.assets_sha256=string(member(root,"initial_pose"),"manifest_sha256");
    append(output.initial_qpos,numbers(root,"initial_qpos",72));
    append(output.model_qpos0,numbers(root,"model_qpos0",72));
    Json bodies=member(root,"bodies"), shapes=member(root,"shapes");
    Json hinges=member(root,"hinges"), actuators=member(root,"actuators");
    ordered(bodies,60,"body"); ordered(shapes,91,"shape");
    ordered(hinges,58,"hinge"); ordered(actuators,58,"actuator");
    for (int i=0;i<60;++i) {
        Json body=item(bodies,i);
        append(output.packed_bodies,pose(member(body,"world_pose")));
        const double mass=number(body,"mass_kg");
        require(mass>0,"body mass must be positive"); append(output.packed_bodies,mass);
        const auto inertia=numbers(body,"principal_inertia_kg_m2",3);
        for (double value:inertia) require(value>0,"principal inertia must be positive");
        append(output.packed_bodies,inertia);
    }
    for (int i=0;i<91;++i) {
        Json shape=item(shapes,i);
        const std::string kind=string(shape,"kind");
        const int type=kind=="sphere"?0:kind=="capsule"?1:kind=="box"?2:kind=="cylinder"?3:-1;
        require(type>=0,"unsupported shape kind");
        require(integer(shape,"condim")==3,"contact dimension must be 3");
        if (type==1 || type==3) require(string(shape,"target_long_axis")=="y","native shape long axis mismatch");
        const int rigid=integer(shape,"rigid_body_id");
        require(rigid>=-1 && rigid<60,"shape rigid body outside packed world");
        const int contype=integer(shape,"contype"), affinity=integer(shape,"conaffinity");
        require(contype>=0 && affinity>=0 && double(float(contype))==contype
                && double(float(affinity))==affinity,"collision bits must round-trip through float32");
        append(output.packed_shapes,double(rigid)); append(output.packed_shapes,double(type));
        append(output.packed_shapes,pose(member(shape,"target_local_pose")));
        Json dimensions=member(shape,"dimensions");
        if (type==2) append(output.packed_shapes,numbers(dimensions,"half_extents_xyz",3));
        else {
            append(output.packed_shapes,number(dimensions,"radius_m"));
            Json half=cJSON_GetObjectItemCaseSensitive(dimensions,"half_length_m");
            append(output.packed_shapes,half?number(half):0.0); append(output.packed_shapes,0.0);
        }
        append(output.packed_shapes,numbers(shape,"friction",3)[0]);
        append(output.packed_shapes,double(contype)); append(output.packed_shapes,double(affinity));
        append(output.packed_shapes,0.0); append(output.packed_shapes,0.0);
    }
    for (int i=0;i<58;++i) {
        Json joint=item(hinges,i), actuator=item(actuators,i);
        require(integer(actuator,"hinge_id")==i && integer(actuator,"source_joint_id")==integer(joint,"source_joint_id"),
                "actuator/hinge order mismatch");
        require(integer(actuator,"dyntype")==0 && integer(actuator,"gaintype")==0 && integer(actuator,"biastype")==1,
                "unsupported actuator dynamics/gain/bias");
        const auto gear=numbers(actuator,"gear",6);
        require(gear==std::vector<double>({1,0,0,0,0,0}),"unsupported actuator gear");
        require(boolean(actuator,"forcelimited") && !boolean(actuator,"ctrllimited"),"actuator limit flags mismatch");
        const auto gain=numbers(actuator,"gainprm",mjNGAIN), bias=numbers(actuator,"biasprm",mjNBIAS);
        require(gain[0]>0 && bias[2]<=0 && bias[0]==0 && bias[1]==-gain[0],"unsupported affine position PD");
        for (int k=1;k<mjNGAIN;++k) require(gain[k]==0,"unsupported extra gain coefficient");
        for (int k=3;k<mjNBIAS;++k) require(bias[k]==0,"unsupported extra bias coefficient");
        const auto force=numbers(actuator,"forcerange",2);
        require(force[1]>0 && force[0]==-force[1],"force limits must be positive and symmetric");
        require(boolean(joint,"limited") && number(joint,"stiffness_Nm_per_rad")==0,
                "hinges must be limited without passive springs");
        const int parent=integer(joint,"parent_rigid_body_id"), child=integer(joint,"child_rigid_body_id");
        require(parent>=-1 && parent<60 && child>=0 && child<60,"hinge body outside packed world");
        const int qi=integer(joint,"qposadr"), vi=integer(joint,"dofadr");
        require(qi>=0 && qi<72 && vi>=0 && vi<70,"hinge generalized index outside state");
        const auto limits=numbers(joint,"limits_relative_to_initial_rad",2);
        require(limits[0]<=limits[1],"reversed hinge limits");
        append(output.packed_joints,double(parent)); append(output.packed_joints,double(child));
        append(output.packed_joints,double(qi)); append(output.packed_joints,double(vi));
        append(output.packed_joints,numbers(joint,"anchor_parent_xyz",3));
        append(output.packed_joints,numbers(joint,"anchor_child_xyz",3));
        append(output.packed_joints,numbers(joint,"axis_parent_xyz",3));
        append(output.packed_joints,number(joint,"mujoco_initial_angle_rad"));
        append(output.packed_joints,limits);
        append(output.packed_joints,gain[0]); append(output.packed_joints,-bias[2]);
        append(output.packed_joints,force[1]); append(output.packed_joints,number(joint,"armature_kg_m2"));
        append(output.packed_joints,number(joint,"damping_Nm_s_per_rad"));
        append(output.packed_joints,number(joint,"frictionloss_Nm"));
    }
    Json bases=member(root,"free_bases");
    require(count(bases)==2,"free root count mismatch");
    std::array<Json,2> roots={item(bases,0),item(bases,1)};
    if (integer(roots[0],"qposadr")>integer(roots[1],"qposadr")) std::swap(roots[0],roots[1]);
    for (int r=0;r<2;++r) {
        Json base=roots[r];
        require(integer(base,"qposadr")==r*36 && integer(base,"dofadr")==r*35,"free root addresses mismatch");
        const int rigid=integer(base,"rigid_body_id");
        require(rigid>=0 && rigid<60,"free root body outside packed world");
        Pose local=pose(member(base,"original_link_in_rigid"));
        const Pose world=pose(member(item(bodies,rigid),"world_pose"));
        auto composed=multiply_xyzw(world.data()+3,local.data()+3);
        const auto expected_pose=numbers(base,"initial_qpos_xyz_wxyz",7);
        const std::array<double,4> expected_q={expected_pose[4],expected_pose[5],expected_pose[6],expected_pose[3]};
        double dot=0;
        for (int k=0;k<4;++k) dot+=composed[k]*expected_q[k];
        if (dot<0) for (int k=0;k<4;++k) { local[k+3]=-local[k+3]; composed[k]=-composed[k]; }
        for (int k=0;k<4;++k) require(std::abs(composed[k]-expected_q[k])<=1e-7,"signed free root quaternion mismatch");
        append(output.packed_roots,double(rigid)); append(output.packed_roots,double(r*36));
        append(output.packed_roots,double(r*35)); append(output.packed_roots,local);
        append(output.packed_roots,0.0); append(output.packed_roots,0.0);
    }
    require(output.packed_bodies.size()==60*11 && output.packed_shapes.size()==91*17
            && output.packed_joints.size()==58*22 && output.packed_roots.size()==2*12,"packed ABI layout mismatch");
}

void build_frame_maps(Physics& output, Json root) {
    mjModel* model=output.model;
    require(model && model->nq==72 && model->nv==70 && model->nu==58
            && model->nbody==63 && model->ngeom==91 && model->njnt==60,"compiled model dimensions mismatch");
    require(model->nmesh==0 && model->nhfield==0 && model->nflex==0 && model->ntendon==0
            && model->neq==0 && model->nplugin==0 && model->nmocap==0,"unsupported compiled model feature");
    const auto qpos0=numbers(root,"model_qpos0",72);
    for (int i=0;i<72;++i) require(qpos0[i]==model->qpos0[i],"compiled model qpos0 differs from export");
    const auto gravity=numbers(root,"gravity_xyz_m_s2",3);
    for (int i=0;i<3;++i) require(gravity[i]==model->opt.gravity[i],"compiled model gravity differs from export");
    model->opt.timestep=0.002;
    Json hinges=member(root,"hinges"), actuators=member(root,"actuators");
    for (int i=0;i<58;++i) {
        Json actuator=item(actuators,i), joint=item(hinges,i);
        const int ji=model->actuator_trnid[i*2];
        require(ji>=0 && ji<model->njnt && model->actuator_trntype[i]==mjTRN_JOINT
                && model->jnt_type[ji]==mjJNT_HINGE && integer(joint,"source_joint_id")==ji,
                "compiled actuator joint mapping mismatch");
        output.actuator_ids[i]=i;
        output.joint_qpos[i]=model->jnt_qposadr[ji]; output.joint_qvel[i]=model->jnt_dofadr[ji];
        require(output.joint_qpos[i]==integer(joint,"qposadr") && output.joint_qvel[i]==integer(joint,"dofadr"),
                "compiled joint state index differs from export");
        const auto gain=numbers(actuator,"gainprm",mjNGAIN), bias=numbers(actuator,"biasprm",mjNBIAS);
        const auto force=numbers(actuator,"forcerange",2), gear=numbers(actuator,"gear",6);
        std::copy(gain.begin(),gain.end(),model->actuator_gainprm+i*mjNGAIN);
        std::copy(bias.begin(),bias.end(),model->actuator_biasprm+i*mjNBIAS);
        std::copy(force.begin(),force.end(),model->actuator_forcerange+i*2);
        model->actuator_dyntype[i]=mjDYN_NONE; model->actuator_gaintype[i]=mjGAIN_FIXED;
        model->actuator_biastype[i]=mjBIAS_AFFINE; model->actuator_forcelimited[i]=1;
        model->actuator_ctrllimited[i]=0;
        for (int k=0;k<6;++k) require(gear[k]==model->actuator_gear[i*6+k],"compiled actuator gear differs from export");
        append(output.joint_limits,model->jnt_range[ji*2]); append(output.joint_limits,model->jnt_range[ji*2+1]);
    }
    const char* names[]={"player__pelvis_3266","opponent__pelvis_3266"};
    for (int side=0;side<2;++side) {
        output.root_bodies[side]=mj_name2id(model,mjOBJ_BODY,names[side]);
        require(output.root_bodies[side]>0,"missing fighter root body");
        for (int k=0;k<4;++k) output.initial_heading_wxyz[side*4+k]=float(model->qpos0[side*36+3+k]);
    }
    DataOwner source(mj_makeData(model),mj_deleteData);
    require(bool(source),"mj_makeData failed");
    const auto initial=numbers(root,"initial_qpos",72);
    std::copy(initial.begin(),initial.end(),source->qpos);
    mj_kinematics(model,source.get());
    std::array<int,63> source_rigid{}, owner{};
    std::array<Pose,61> poses{}; poses[0]={0,0,0,0,0,0,1};
    Json bodies=member(root,"bodies");
    for (int i=0;i<60;++i) {
        Json body=item(bodies,i);
        const int source_id=integer(body,"source_body_id");
        require(source_id>0 && source_id<63 && source_rigid[source_id]==0,"invalid source rigid body mapping");
        source_rigid[source_id]=i+1; poses[i+1]=pose(member(body,"world_pose"));
        require(model->body_mass[source_id]==number(body,"mass_kg"),"compiled mass differs from export");
        const auto inertia=numbers(body,"principal_inertia_kg_m2",3);
        for (int k=0;k<3;++k) require(inertia[k]==model->body_inertia[source_id*3+k],"compiled inertia differs from export");
    }
    const auto exported_owner=numbers(root,"source_body_to_rigid_body",63);
    for (int b=0;b<63;++b) {
        const int parent=model->body_parentid[b];
        require(!b || (parent>=0 && parent<b),"source body parents must precede children");
        owner[b]=source_rigid[b]?source_rigid[b]:owner[parent];
        require(exported_owner[b]==owner[b]-1,"compiled rigid owner differs from export");
        const Pose& rigid=poses[owner[b]];
        append(output.host_body_map,double(owner[b])); append(output.host_body_map,double(parent));
        append(output.host_body_map,double(model->body_rootid[b])); append(output.host_body_map,model->body_mass[b]);
        append(output.host_body_map,local_position(rigid,source->xpos+b*3));
        append(output.host_body_map,local_rotation(rigid,source->xmat+b*9));
        append(output.host_body_map,local_position(rigid,source->xipos+b*3));
        append(output.host_body_map,local_rotation(rigid,source->ximat+b*9));
    }
    std::array<bool,61> visited{}; visited[0]=true;
    for (int r=0;r<2;++r) {
        const int rigid=int(output.packed_roots[r*12])+1;
        require(!visited[rigid],"duplicate free root"); visited[rigid]=true;
    }
    for (int i=0;i<58;++i) {
        Json joint=item(hinges,i);
        const int parent=integer(joint,"parent_rigid_body_id")+1, child=integer(joint,"child_rigid_body_id")+1;
        require(visited[parent] && !visited[child],"native reset requires parent-before-child hinge order");
        visited[child]=true;
    }
    require(std::all_of(visited.begin(),visited.end(),[](bool value){return value;}),"roots/hinges do not span all bodies");
    Json shapes=member(root,"shapes");
    for (int g=0;g<91;++g) {
        Json shape=item(shapes,g);
        require(integer(shape,"source_geom_id")==g,"geometry ID order differs from source");
        require(integer(shape,"source_body_id")==model->geom_bodyid[g],"geometry source body mismatch");
        append(output.host_geom_map,double(integer(shape,"rigid_body_id")+1));
        append(output.host_geom_map,pose(member(shape,"source_local_pose")));
        output.geom_bodyid.push_back(model->geom_bodyid[g]); output.geom_type.push_back(model->geom_type[g]);
        output.geom_contype.push_back(model->geom_contype[g]); output.geom_conaffinity.push_back(model->geom_conaffinity[g]);
        for (int k=0;k<3;++k) append(output.geom_size,model->geom_size[g*3+k]);
    }
    output.data.bodies=63; output.data.geoms=91;
    require(output.host_body_map.size()==63*18 && output.host_geom_map.size()==91*8,"semantic frame map layout mismatch");
}

template<typename T>
void allocate(Physics& physics, T*& output, size_t count) {
    require(count>0 && count<=std::numeric_limits<size_t>::max()/sizeof(T),"GPU buffer size overflow");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&output),count*sizeof(T)),"allocate semantic field");
    physics.allocations.push_back(output);
    cuda_check(cudaMemsetAsync(output,0,count*sizeof(T),physics.stream),"initialize semantic field");
}

rek5_native::RpsDescriptor native_descriptor(const PhysicsDescriptor& descriptor) {
    static_assert(sizeof(PhysicsDescriptor)==sizeof(rek5_native::RpsDescriptor),"semantic descriptor ABI size");
    static_assert(std::is_standard_layout<PhysicsDescriptor>::value,"semantic descriptor layout");
    static_assert(offsetof(PhysicsDescriptor,qpos)==offsetof(rek5_native::RpsDescriptor,qpos),"semantic descriptor pointer offset");
    static_assert(offsetof(PhysicsDescriptor,pre_centers)==offsetof(rek5_native::RpsDescriptor,pre_centers),"semantic descriptor tail offset");
    rek5_native::RpsDescriptor native;
    std::memcpy(&native,&descriptor,sizeof(native));
    return native;
}

__global__ void finite_fields_kernel(PhysicsDescriptor descriptor,int* result) {
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const float* fields[]={descriptor.qpos,descriptor.qvel,descriptor.xpos,
        descriptor.xipos,descriptor.ximat,descriptor.com,descriptor.cvel,
        descriptor.geom_xpos,descriptor.geom_xmat};
    const size_t a=descriptor.arenas,b=descriptor.bodies,g=descriptor.geoms;
    const size_t lengths[]={a*72,a*70,a*b*3,a*b*3,a*b*9,a*b*3,a*b*6,a*g*3,a*g*9};
    for (int field=0;field<9;++field)
        if (index<lengths[field] && !isfinite(fields[field][index])) atomicOr(result,1<<field);
}

} // namespace

Physics* physics_load_model(const char* xml_path, const char* export_json_path) {
    auto output=std::unique_ptr<Physics,decltype(&physics_close)>(new Physics,physics_close);
    const auto xml=read_file(xml_path), exported=read_file(export_json_path);
    const char* parsed_end=nullptr;
    JsonOwner root(cJSON_ParseWithOpts(exported.c_str(),&parsed_end,1),cJSON_Delete);
    require(bool(root),"invalid model export JSON");
    output->model_sha256=sha256(xml); output->export_sha256=sha256(exported);
    require(string(member(root.get(),"source"),"model_sha256")==output->model_sha256,"model SHA-256 differs from export source");
    pack_export(*output,root.get());
    char error[1024]={};
    output->model=mj_loadXML(xml_path,nullptr,error,sizeof(error));
    require(output->model!=nullptr,std::string("MuJoCo model compilation failed: ")+error);
    build_frame_maps(*output,root.get());
    return output.release();
}

Physics* physics_create(const char* xml_path, const char* export_json_path, int arenas, cudaStream_t stream) {
    require(arenas>0 && arenas<=std::numeric_limits<int>::max()/128,"arena count outside contact capacity bound");
    auto output=std::unique_ptr<Physics,decltype(&physics_close)>(physics_load_model(xml_path,export_json_path),physics_close);
    output->stream=stream;
    PhysicsDescriptor& d=output->data;
    d.arenas=arenas; d.capacity=arenas*128;
    const size_t a=arenas, b=d.bodies, g=d.geoms, c=d.capacity;
    output->allocations.reserve(28);
    allocate(*output,d.qpos,a*72); allocate(*output,d.qvel,a*70); allocate(*output,output->ctrl,a*58);
    allocate(*output,d.base,a*2*4); allocate(*output,d.angular,a*2*3); allocate(*output,d.time,a);
    allocate(*output,d.xpos,a*b*3); allocate(*output,d.xquat,a*b*4); allocate(*output,d.xmat,a*b*9);
    allocate(*output,d.xipos,a*b*3); allocate(*output,d.ximat,a*b*9); allocate(*output,d.com,a*b*3);
    allocate(*output,d.cvel,a*b*6); allocate(*output,d.geom_xpos,a*g*3); allocate(*output,d.geom_xmat,a*g*9);
    allocate(*output,d.contact_geom,c*2); allocate(*output,d.contact_world,c); allocate(*output,d.nacon,1);
    allocate(*output,d.contact_dist,c); allocate(*output,d.contact_pos,c*3); allocate(*output,d.contact_frame,c*9);
    allocate(*output,d.counts,a); allocate(*output,d.offsets,a);
    allocate(*output,d.body_map,b*18); allocate(*output,d.geom_map,g*8); allocate(*output,d.pre_centers,a*61*3);
    allocate(*output,output->finite_status,1);
    cuda_check(cudaMemcpyAsync(d.body_map,output->host_body_map.data(),b*18*sizeof(float),cudaMemcpyHostToDevice,stream),"upload body map");
    cuda_check(cudaMemcpyAsync(d.geom_map,output->host_geom_map.data(),g*8*sizeof(float),cudaMemcpyHostToDevice,stream),"upload geometry map");
    output->native_handle=rek5_native::rp_create(arenas,output->packed_bodies.data(),60,
            output->packed_shapes.data(),91,output->packed_joints.data(),58,output->packed_roots.data(),2,0);
    require(output->native_handle!=nullptr,std::string("native create failed: ")+rek5_native::rp_last_error());
    // The included rp_create uploads its immutable world on the default stream.
    // Establish startup ordering before resetting on a caller-owned stream.
    cuda_check(cudaStreamSynchronize(nullptr),"complete native world upload");
    output->stats=static_cast<rek5_native::RpHandle*>(output->native_handle)->stats;
    native_check(rek5_native::rp_reset(output->native_handle,d.qpos,d.qvel,d.base,d.angular,d.time,stream),"native initial reset");
    physics_refresh(output.get());
    cuda_check(cudaStreamSynchronize(stream),"complete physics initialization");
    return output.release();
}

void physics_step(Physics* physics, const float* device_ctrl) {
    require(physics && physics->native_handle && device_ctrl,"closed physics or null control buffer");
    const auto descriptor=native_descriptor(physics->data);
    native_check(rek5_native::rps_step_controls(physics->native_handle,&descriptor,device_ctrl,physics->stream),"semantic step");
}

void physics_forward_selected(Physics* physics, const uint8_t* device_mask) {
    require(physics && physics->native_handle && device_mask,"closed physics or null reset mask");
    const auto descriptor=native_descriptor(physics->data);
    native_check(rek5_native::rps_forward_selected(physics->native_handle,&descriptor,device_mask,physics->stream),"semantic masked forward");
}

void physics_refresh(Physics* physics) {
    require(physics && physics->native_handle,"closed physics");
    const auto descriptor=native_descriptor(physics->data);
    native_check(rek5_native::rps_refresh(physics->native_handle,&descriptor,physics->stream),"semantic refresh");
}

std::vector<int> physics_stats(Physics* physics) {
    require(physics && physics->native_handle,"closed physics");
    cuda_check(cudaStreamSynchronize(physics->stream),"physics reporting boundary");
    std::vector<int> stats(size_t(physics->data.arenas)*4);
    native_check(rek5_native::rp_get_stats(physics->native_handle,stats.data()),"download physics statistics");
    return stats;
}

void physics_check_status(Physics* physics) {
    const auto stats=physics_stats(physics);
    for (int a=0;a<physics->data.arenas;++a) {
        require(stats[a*4]<128,"native contact capacity reached in arena "+std::to_string(a));
        require(!stats[a*4+1],"nonfinite physics state in arena "+std::to_string(a));
        require(!stats[a*4+2],"persistent collision/solver failure in arena "+std::to_string(a)
                +", code "+std::to_string(stats[a*4+2]));
    }
    int contacts=0;
    cuda_check(cudaMemcpy(&contacts,physics->data.nacon,sizeof(contacts),cudaMemcpyDeviceToHost),"download contact count");
    require(contacts>=0 && contacts<=physics->data.capacity,"semantic contact capacity exceeded");
    cuda_check(cudaMemsetAsync(physics->finite_status,0,sizeof(int),physics->stream),"clear semantic finite status");
    const size_t count=size_t(physics->data.arenas)*physics->data.geoms*9;
    finite_fields_kernel<<<(count+255)/256,256,0,physics->stream>>>(physics->data,physics->finite_status);
    cuda_check(cudaGetLastError(),"semantic finite scan");
    int finite_status=0;
    cuda_check(cudaMemcpyAsync(&finite_status,physics->finite_status,sizeof(int),cudaMemcpyDeviceToHost,physics->stream),"download semantic finite status");
    cuda_check(cudaStreamSynchronize(physics->stream),"semantic finite reporting boundary");
    require(finite_status==0,"nonfinite semantic export field mask "+std::to_string(finite_status));
}

void physics_close(Physics* physics) noexcept {
    if (!physics) return;
    if (physics->native_handle || !physics->allocations.empty()) cudaStreamSynchronize(physics->stream);
    if (physics->native_handle) rek5_native::rp_destroy(physics->native_handle);
    for (void* allocation:physics->allocations) cudaFree(allocation);
    if (physics->model) mj_deleteModel(physics->model);
    delete physics;
}

} // namespace rek5
