#include "fast_assets.h"
#include "../../../vendor/cJSON.h"
#include <mujoco/mujoco.h>
#include <openssl/evp.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>

extern "C" {
#include "../g1_strike_catalog.h"
}
#include "recovered_contact_rules.cuh"

namespace {
void require(bool ok,const std::string& message) { if(!ok)throw std::runtime_error("fast assets: "+message); }
using Json=std::unique_ptr<cJSON,decltype(&cJSON_Delete)>;
const cJSON* field(const cJSON* o,const char* k) { auto* v=cJSON_GetObjectItemCaseSensitive(o,k);require(v!=nullptr,std::string("missing ")+k);return v; }
double number(const cJSON* v) { require(cJSON_IsNumber(v)&&std::isfinite(v->valuedouble),"expected finite number");return v->valuedouble; }
int integer(const cJSON* v) { double x=number(v);require(x>=INT32_MIN&&x<=INT32_MAX&&x==std::floor(x),"invalid int32");return int(x); }
std::string string(const cJSON* v) {require(cJSON_IsString(v)&&v->valuestring,"expected string");return v->valuestring;}
std::string filename(const char* root,const std::string& name) {
    require(root&&*root&&!name.empty()&&name!="."&&name!=".."&&name.find_first_of("/\\")==std::string::npos,"invalid bundle path");
    return std::string(root)+"/"+name;
}
std::vector<unsigned char> bytes(const std::string& path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);require(bool(f),"cannot open "+path);
    auto n=f.tellg();require(n>=0&&uint64_t(n)<(uint64_t(1)<<32),"invalid asset size");
    std::vector<unsigned char> b(size_t(n),0);f.seekg(0);if(n)f.read(reinterpret_cast<char*>(b.data()),n);require(bool(f),"incomplete read");return b;
}
std::string sha(const std::vector<unsigned char>& b) {
    unsigned char digest[32];unsigned n=0;require(EVP_Digest(b.data(),b.size(),digest,&n,EVP_sha256(),nullptr)==1&&n==32,"SHA256 failed");
    const char* h="0123456789abcdef";std::string s(64,'0');for(int i=0;i<32;i++){s[2*i]=h[digest[i]>>4];s[2*i+1]=h[digest[i]&15];}return s;
}
Json json(const std::vector<unsigned char>& b) {
    std::string text(b.begin(),b.end());Json j(cJSON_ParseWithLengthOpts(text.c_str(),text.size()+1,nullptr,1),cJSON_Delete);
    require(j&&cJSON_IsObject(j.get()),"invalid JSON");return j;
}
struct Array {std::vector<int> shape;std::vector<float> v;};
struct Clip {int id,count;float fps;const Array *q,*root,*xyz;};
float heading(const float* q) {return std::atan2(2*(q[1]*q[2]+q[0]*q[3]),1-2*(q[2]*q[2]+q[3]*q[3]));}
void rotate_yaw(float* q,float a) {
    float c=std::cos(a/2),s=std::sin(a/2),w=q[0],x=q[1],y=q[2],z=q[3];
    q[0]=c*w-s*z;q[1]=c*x-s*y;q[2]=s*x+c*y;q[3]=s*w+c*z;
}
void slerp(const float* a,const float* b,float t,float* out) {
    double dot=0;for(int k=0;k<4;k++)dot+=double(a[k])*b[k];double sign=dot<0?-1:1;dot=std::clamp(dot*sign,0.0,1.0);
    double x=1-t,y=t;if(dot<.9995){double angle=std::acos(dot),den=std::sin(angle);x=std::sin((1-t)*angle)/den;y=std::sin(t*angle)/den;}
    double n=0;for(int k=0;k<4;k++){out[k]=float(x*a[k]+y*sign*b[k]);n+=double(out[k])*out[k];}
    require(n>.9&&n<1.1,"invalid interpolated quaternion");for(int k=0;k<4;k++)out[k]/=float(std::sqrt(n));
}
int identity(const mjModel* m,int kind,const std::string& name) {int id=mj_name2id(m,kind,name.c_str());require(id>=0,"missing model identity "+name);return id;}
std::vector<int> body_geoms(const mjModel* m,const char* body) {
    int id=identity(m,mjOBJ_BODY,std::string("player__")+body);std::vector<int> g;
    for(int k=0;k<m->ngeom;k++)if(m->geom_bodyid[k]==id)g.push_back(k);
    require(!g.empty(),"body has no geometry");return g;
}
double bounding_radius(const mjModel* m,int g) {
    const mjtNum* s=m->geom_size+g*3;switch(m->geom_type[g]) {
    case mjGEOM_SPHERE:return s[0];
    case mjGEOM_CAPSULE:return s[0]+s[1];
    case mjGEOM_CYLINDER:return std::hypot(s[0],s[1]);
    case mjGEOM_BOX:case mjGEOM_ELLIPSOID:return std::sqrt(s[0]*s[0]+s[1]*s[1]+s[2]*s[2]);
    default:throw std::runtime_error("fast assets: unsupported proxy geometry");}
}
void sphere(const mjModel* m,const mjData* d,const std::vector<int>& geoms,float* center,float& radius) {
    double lo[3]={INFINITY,INFINITY,INFINITY},hi[3]={-INFINITY,-INFINITY,-INFINITY};
    for(int g:geoms){double r=bounding_radius(m,g);for(int k=0;k<3;k++){lo[k]=std::min(lo[k],d->geom_xpos[g*3+k]-r);hi[k]=std::max(hi[k],d->geom_xpos[g*3+k]+r);}}
    for(int k=0;k<3;k++)center[k]=float((lo[k]+hi[k])/2);
    double bound=0;for(int g:geoms){double d2=0;for(int k=0;k<3;k++)d2+=std::pow(d->geom_xpos[g*3+k]-center[k],2);bound=std::max(bound,std::sqrt(d2)+bounding_radius(m,g));}
    require(std::isfinite(bound)&&bound>0&&bound<2,"invalid proxy radius");radius=float(bound);
}
int primitive_kind(const mjModel* m,int geom) {
    switch(m->geom_type[geom]) {
    case mjGEOM_SPHERE:return rek5_primitive::Sphere;
    case mjGEOM_CAPSULE:return rek5_primitive::Capsule;
    case mjGEOM_BOX:return rek5_primitive::Box;
    default:throw std::runtime_error("fast assets: unsupported native contact primitive");
    }
}
void validate_primitive(const mjModel* m,int geom,int expected_kind) {
    const char* name=mj_id2name(m,mjOBJ_GEOM,geom);require(name&&*name,"unnamed native contact primitive");
    require(primitive_kind(m,geom)==expected_kind,std::string("native contact primitive type mismatch: ")+name);
    const mjtNum* size=m->geom_size+3*geom;
    for(int k=0;k<3;k++)require(std::isfinite(size[k])&&size[k]>=0&&size[k]<=std::numeric_limits<float>::max(),std::string("invalid native primitive dimension: ")+name);
    require(size[0]>0,std::string("zero native primitive radius or half-size: ")+name);
    if(expected_kind==rek5_primitive::Capsule)require(size[1]>0,std::string("zero native capsule half-segment: ")+name);
    if(expected_kind==rek5_primitive::Box)require(size[1]>0&&size[2]>0,std::string("zero native box half-size: ")+name);
}
void primitive(const mjModel* m,const mjData* d,int geom,rek5_primitive::Shape& shape) {
    shape.kind=primitive_kind(m,geom);
    for(int k=0;k<3;k++){
        shape.center[k]=float(d->geom_xpos[geom*3+k]);
        shape.size[k]=float(m->geom_size[geom*3+k]);
        require(std::isfinite(shape.center[k])&&std::isfinite(shape.size[k]),"nonfinite baked native primitive");
    }
    for(int k=0;k<9;k++){
        shape.axes[k]=float(d->geom_xmat[geom*9+k]);
        require(std::isfinite(shape.axes[k]),"nonfinite baked native primitive orientation");
    }
}
std::string quoted_geom_name(const mjModel* m,int geom) {
    Json value(cJSON_CreateString(mj_id2name(m,mjOBJ_GEOM,geom)),cJSON_Delete);require(bool(value),"cannot allocate primitive provenance name");
    std::unique_ptr<char,decltype(&cJSON_free)> text(cJSON_PrintUnformatted(value.get()),cJSON_free);
    require(bool(text),"cannot encode primitive provenance name");return text.get();
}
}

FastAssets load_fast_assets(const RekNative5Config& config) {
    require(config.model_path&&config.assets_path&&config.motion_features_path,"required paths missing");
    const std::uint16_t endian=1;require(*reinterpret_cast<const unsigned char*>(&endian)==1,"float32_le needs a little-endian host");
    FastAssets out;auto mb=bytes(filename(config.assets_path,"semantic_duel_assets_manifest.json"));out.manifest_sha256=sha(mb);auto manifest=json(mb);
    require(string(field(manifest.get(),"schema"))=="rek.g1_semantic_duel_assets.v1","unsupported semantic manifest");
    const auto* catalog=rek_g1_current_build_strike_catalog();
    require(rek_g1_validate_strike_catalog(catalog),"invalid recovered strike catalog");
    const auto* build=cJSON_GetObjectItemCaseSensitive(manifest.get(),"build_fingerprint");
    const auto* contracts=cJSON_GetObjectItemCaseSensitive(manifest.get(),"source_contracts");
    const auto* contract=contracts?cJSON_GetObjectItemCaseSensitive(contracts,"route_contract_file_sha256"):nullptr;
    out.recovered_catalog_compatible=build&&contract&&cJSON_IsString(build)&&cJSON_IsString(contract)&&
        string(build)==catalog->build_fingerprint&&string(contract)==catalog->source_sha256;
    std::copy(catalog->impact_events,catalog->impact_events+catalog->impact_event_count,out.impact_events.begin());
    for(size_t i=0;i<catalog->count;i++){const auto& e=catalog->entries[i];out.impact_offsets[e.route_id]=e.impact_event_offset;out.impact_counts[e.route_id]=e.impact_event_count;}
    out.recovered_hit_config=rek5_recovered::rek_g1_current_build_hit_detector_config();
    auto fb=bytes(filename(config.motion_features_path,"foot_features_manifest.json"));out.features_sha256=sha(fb);auto features=json(fb);
    require(string(field(features.get(),"asset_manifest_sha256"))==out.manifest_sha256,"feature manifest mismatch");
    std::map<std::string,Array> arrays;const cJSON* files=field(manifest.get(),"files");require(cJSON_IsObject(files),"files must be object");
    for(auto* r=files->child;r;r=r->next){
        require(r->string,"unnamed asset");auto b=bytes(filename(config.assets_path,r->string));
        require(double(b.size())==number(field(r,"bytes"))&&sha(b)==string(field(r,"sha256")),"asset identity mismatch");
        auto* dtype=cJSON_GetObjectItemCaseSensitive(r,"dtype");if(!dtype||string(dtype)!="float32_le")continue;
        Array a;size_t n=1;auto* shape=field(r,"shape");require(cJSON_IsArray(shape),"shape must be array");
        for(auto* d=shape->child;d;d=d->next){int count=integer(d);require(count>0&&n<=SIZE_MAX/size_t(count),"invalid array shape");a.shape.push_back(count);n*=size_t(count);}
        require(n<=SIZE_MAX/sizeof(float)&&n*sizeof(float)==b.size(),"array byte count mismatch");a.v.resize(n);std::memcpy(a.v.data(),b.data(),b.size());
        for(float v:a.v)require(std::isfinite(v),"nonfinite clip array");
        arrays.emplace(r->string,std::move(a));
    }
    std::map<int,Clip> clips;auto* clip_json=field(manifest.get(),"clips");require(cJSON_IsArray(clip_json),"clips must be array");
    double largest_xy_span=0;
    for(auto* r=clip_json->child;r;r=r->next){
        auto* cf=field(r,"files");Clip c{integer(field(r,"npz_path_id")),integer(field(r,"frames")),float(number(field(r,"fps"))),
            &arrays.at(string(field(cf,"mujoco_joint_order"))),&arrays.at(string(field(cf,"wxyz"))),&arrays.at(string(field(cf,"xyz_m")))};
        require(c.count>0&&c.fps>0&&c.q->shape==std::vector<int>({c.count,29})&&c.root->shape==std::vector<int>({c.count,4})&&c.xyz->shape==std::vector<int>({c.count,3}),"invalid clip shape");
        for(int f=0;f<c.count;f++){double norm=0;for(int k=0;k<4;k++)norm+=double(c.root->v[f*4+k])*c.root->v[f*4+k];require(std::abs(std::sqrt(norm)-1)<1e-4,"nonunit source root quaternion");}
        for(int k=0;k<2;k++){float lo=INFINITY,hi=-INFINITY;for(int f=0;f<c.count;f++){lo=std::min(lo,c.xyz->v[f*3+k]);hi=std::max(hi,c.xyz->v[f*3+k]);}largest_xy_span=std::max(largest_xy_span,double(hi-lo));}
        require(clips.emplace(c.id,c).second,"duplicate clip ID");
    }
    require(largest_xy_span<1e-4,"root XY is no longer stationary; update compact root contract");
    char error[2048]={};std::unique_ptr<mjModel,decltype(&mj_deleteModel)> model(mj_loadXML(config.model_path,nullptr,error,sizeof(error)),mj_deleteModel);
    require(bool(model),std::string("model compilation failed: ")+error);auto* m=model.get();
    require(m->nq==72&&m->nv==70&&m->nu==58,"model dimensions mismatch");out.model_sha256=sha(bytes(config.model_path));
    std::unique_ptr<mjData,decltype(&mj_deleteData)> data(mj_makeData(m),mj_deleteData);require(bool(data),"model data allocation failed");auto* d=data.get();
    for(int k=0;k<72;k++)out.initial_qpos[k]=float(m->qpos0[k]);
    for(int side=0;side<2;side++){
        for(int k=0;k<2;k++)out.spawn_xy[side][k]=float(m->qpos0[side*36+k]);
        out.initial_heading[side]=heading(out.initial_qpos+side*36+3);
        for(int j=0;j<29;j++){int joint=m->actuator_trnid[(side*29+j)*2];require(joint>=0&&m->jnt_type[joint]==mjJNT_HINGE,"actuator is not a hinge");out.qindices[side][j]=m->jnt_qposadr[joint];out.vindices[side][j]=m->jnt_dofadr[joint];}
    }
    int floor=identity(m,mjOBJ_GEOM,"arena_Collider_Floor_Rektagon");require(m->geom_type[floor]==mjGEOM_BOX&&m->geom_bodyid[floor]==0,"unsupported arena floor");
    out.floor_height=float(m->geom_pos[floor*3+2]+m->geom_size[floor*3+2]);
    for(int k=0;k<2;k++)out.arena_half_extent[k]=float(m->geom_size[floor*3+k]);
    const std::vector<int> strikes[]={body_geoms(m,"left_ankle_roll_link_3045"),body_geoms(m,"right_ankle_roll_link_3090"),
        body_geoms(m,"left_wrist_yaw_link_3467"),body_geoms(m,"right_wrist_yaw_link_3293"),body_geoms(m,"left_knee_link_3106"),body_geoms(m,"right_knee_link_3429")};
    const std::vector<int> targets[]={body_geoms(m,"pelvis_3266"),{identity(m,mjOBJ_GEOM,"player__mjgeom_3285")},{identity(m,mjOBJ_GEOM,"player__mjgeom_3064")}};
    const int strike_group_counts[6]={4,4,1,1,1,1};
    const int strike_group_kinds[6]={rek5_primitive::Sphere,rek5_primitive::Sphere,rek5_primitive::Box,rek5_primitive::Box,rek5_primitive::Capsule,rek5_primitive::Capsule};
    const int target_kinds[3]={rek5_primitive::Box,rek5_primitive::Box,rek5_primitive::Capsule};
    std::array<int,12> strike_geoms{};std::array<int,3> target_geoms{};int strike_count=0;
    for(int limb=0;limb<6;limb++){
        require(strikes[limb].size()==size_t(strike_group_counts[limb]),"native striker group count mismatch");
        for(int geom:strikes[limb]){
            require(strike_count<int(strike_geoms.size())&&out.strike_limb[strike_count]==limb,"native striker limb mapping mismatch");
            validate_primitive(m,geom,strike_group_kinds[limb]);strike_geoms[strike_count++]=geom;
        }
    }
    require(strike_count==int(strike_geoms.size()),"expected 12 native striker primitives");
    for(int target=0;target<3;target++){
        require(targets[target].size()==1,"expected one native primitive per target");
        target_geoms[target]=targets[target][0];validate_primitive(m,target_geoms[target],target_kinds[target]);
    }
    const int mirror_indices[29]={6,7,8,9,10,11,0,1,2,3,4,5,12,13,14,22,23,24,25,26,27,28,15,16,17,18,19,20,21};
    const int mirror_negate[29]={0,1,1,0,0,1,0,1,1,0,0,1,1,1,0,0,1,1,0,1,0,1,0,1,1,0,1,0,1};
    auto* routes=field(manifest.get(),"routes");require(cJSON_IsArray(routes)&&cJSON_GetArraySize(routes)==24,"expected 24 routes");
    int move_route[17];std::fill(move_route,move_route+17,-1);
    for(int ri=0;ri<24;ri++){
        auto* r=cJSON_GetArrayItem(routes,ri);require(integer(field(r,"route_id"))==ri,"route order mismatch");auto* rc=field(r,"config");
        auto& route=out.routes[ri];const auto& c=clips.at(integer(field(r,"npz_path_id")));route.source_clip_id=c.id;
        route.start_frame=std::clamp(integer(field(rc,"start_frame")),0,c.count-1);int end=integer(field(rc,"end_frame"));route.end_frame=end<0?c.count-1:std::clamp(end,route.start_frame,c.count-1);
        route.playback_speed=float(number(field(rc,"playback_speed")));require(route.playback_speed!=0,"zero playback speed unsupported");route.loop=integer(field(rc,"loop"));require(route.loop==0||route.loop==1,"invalid loop flag");
        route.blend_in_seconds=float(number(field(rc,"blend_in_seconds")));route.blend_out_seconds=float(number(field(rc,"blend_out_seconds")));route.yaw_blend=float(number(field(rc,"yaw_blend")));
        require(route.blend_in_seconds>=0&&route.blend_out_seconds>=0&&route.yaw_blend>=0&&route.yaw_blend<=1,"invalid blending values");
        int mirror=integer(field(rc,"mirror"));require(mirror==0||mirror==1,"invalid mirror flag");auto* move=field(r,"runtime_move_index");route.move=cJSON_IsNull(move)?-1:integer(move);
        const auto& expected=rek_g1_native_static_motion_routes()->routes[ri];
        out.recovered_catalog_compatible=out.recovered_catalog_compatible&&c.id==expected.npz_path_id&&c.count==int(expected.asset_frames)&&c.fps==expected.asset_fps&&
            route.playback_speed==expected.playback_speed&&integer(field(rc,"start_frame"))==expected.start_frame&&integer(field(rc,"end_frame"))==expected.end_frame&&
            route.blend_in_seconds==expected.blend_in_seconds&&route.blend_out_seconds==expected.blend_out_seconds&&route.yaw_blend==expected.yaw_blend&&
            mirror==expected.mirror&&route.loop==expected.loop&&(ri<7?route.move<0:route.move==expected.runtime_move_index);
        route.offset=int(out.frames.size());route.fps=50;
        if(route.move>=0){require(route.move<17&&move_route[route.move]<0,"duplicate move");move_route[route.move]=ri;require(config.move_duration_ticks[route.move]>0&&config.move_duration_ticks[route.move]<100000,"invalid explicit move duration");out.move_duration_ticks[route.move]=config.move_duration_ticks[route.move];route.count=int(config.move_duration_ticks[route.move])+1;}
        else route.count=std::max(2,int(std::ceil((route.end_frame-route.start_frame+1)*50.0/(c.fps*std::abs(route.playback_speed)))));
        float initial_yaw=heading(c.root->v.data());
        for(int tick=0;tick<route.count;tick++){
            double cursor=(route.playback_speed>0?route.start_frame:route.end_frame)+tick*double(c.fps)*route.playback_speed/50.0;
            if(route.loop){double span=route.end_frame-route.start_frame+1;cursor=route.start_frame+std::fmod(std::fmod(cursor-route.start_frame,span)+span,span);}else cursor=std::clamp(cursor,double(route.start_frame),double(route.end_frame));
            int a=int(std::floor(cursor)),b=a+1<=route.end_frame?a+1:route.loop?route.start_frame:route.end_frame;float t=float(cursor-a);FastFrame frame{};
            for(int j=0;j<29;j++){int k=mirror?mirror_indices[j]:j;float v=c.q->v[a*29+k]+t*(c.q->v[b*29+k]-c.q->v[a*29+k]);frame.q[j]=mirror&&mirror_negate[j]?-v:v;}
            slerp(c.root->v.data()+a*4,c.root->v.data()+b*4,t,frame.root_wxyz);rotate_yaw(frame.root_wxyz,-initial_yaw);
            if(mirror){frame.root_wxyz[1]=-frame.root_wxyz[1];frame.root_wxyz[3]=-frame.root_wxyz[3];}
            frame.clip_yaw=heading(frame.root_wxyz);rotate_yaw(frame.root_wxyz,-frame.clip_yaw*route.yaw_blend);
            frame.root_z=out.floor_height+c.xyz->v[a*3+2]+t*(c.xyz->v[b*3+2]-c.xyz->v[a*3+2]);
            std::copy(m->qpos0,m->qpos0+72,d->qpos);d->qpos[0]=d->qpos[1]=0;d->qpos[2]=frame.root_z;
            for(int k=0;k<4;k++)d->qpos[3+k]=frame.root_wxyz[k];
            for(int j=0;j<29;j++)d->qpos[out.qindices[0][j]]=frame.q[j];
            mj_kinematics(m,d); // Forward geometry only; no physics step/solver.
            for(int k=0;k<6;k++)sphere(m,d,strikes[k],frame.strike_xyz[k],frame.strike_radius[k]);
            for(int k=0;k<3;k++)sphere(m,d,targets[k],frame.target_xyz[k],frame.target_radius[k]);
            for(int k=0;k<12;k++)primitive(m,d,strike_geoms[k],frame.strike_shapes[k]);
            for(int k=0;k<3;k++)primitive(m,d,target_geoms[k],frame.target_shapes[k]);
            for(float v:frame.q)require(std::isfinite(v),"nonfinite baked pose");
            out.frames.push_back(frame);
        }
    }
    const int held_routes[16]={-1,0,1,2,3,4,5,6,1,1,2,2,3,3,4,4};std::copy(held_routes,held_routes+16,out.action_to_route.begin());
    const int move_order[17]={6,7,8,9,0,1,2,3,4,5,10,11,12,13,14,15,16};
    for(int i=0;i<17;i++){require(move_route[i]>=0,"missing discrete move");out.action_to_route[16+i]=move_route[move_order[i]];}
    const FastFrame& idle=out.frames.at(out.routes[0].offset);
    for(int side=0;side<2;side++){out.initial_qpos[side*36+2]=idle.root_z;for(int j=0;j<29;j++)out.initial_qpos[out.qindices[side][j]]=idle.q[j];}
    std::ostringstream info;info.precision(10);info<<"{\"schema\":\"rek.fast_assets.v1\",\"classification\":\"approximate_kinematic_candidate\",\"rek_parity_claim\":false,\"model_sha256\":\""<<out.model_sha256<<"\",\"asset_manifest_sha256\":\""<<out.manifest_sha256<<"\",\"features_manifest_sha256\":\""<<out.features_sha256<<"\",\"routes\":24,\"source_clips\":"<<clips.size()<<",\"baked_frames\":"<<out.frames.size()<<",\"frame_bytes\":"<<sizeof(FastFrame)<<",\"sample_hz\":50,\"max_source_root_xy_span_m\":"<<largest_xy_span<<",\"root_translation_model\":\"external_explicit_approximation\",\"root_height_source\":\"clip_xyz_m\",\"collision_proxy\":\"bounding_sphere_union_of_named_model_geoms\",\"offline_fk\":\"mj_kinematics\",\"cpu_physics_steps\":0,\"retained_native_primitives\":{\"legacy_sphere_fields_preserved\":true,\"axes_layout\":\"row_major_local_axes_in_columns\",\"size_semantics\":\"sphere_radius_capsule_radius_halfsegment_box_halfsizes\",\"kind_enum\":{\"sphere\":0,\"capsule\":1,\"box\":2},\"strikers\":[";
    for(int k=0;k<12;k++){if(k)info<<',';info<<"{\"geom\":"<<quoted_geom_name(m,strike_geoms[k])<<",\"kind\":"<<primitive_kind(m,strike_geoms[k])<<",\"limb\":"<<out.strike_limb[k]<<'}';}
    info<<"],\"targets\":[";
    for(int k=0;k<3;k++){if(k)info<<',';info<<"{\"geom\":"<<quoted_geom_name(m,target_geoms[k])<<",\"kind\":"<<primitive_kind(m,target_geoms[k])<<",\"target\":"<<k<<'}';}
    info<<"]}}";out.provenance_json=info.str();
    return out;
}
