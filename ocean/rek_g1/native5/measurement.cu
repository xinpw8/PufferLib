#include "measurement.cuh"
#include <cub/cub.cuh>
#include <algorithm>
#include <array>
#include <cfloat>
#include <map>
#include <memory>
#include <set>
#include "../gpu_combat_measurement_fused.cu"
#undef KERNEL
#undef DEVICE
#undef HOST_DEVICE
#undef EACH
#undef LAUNCH
#undef STATUS

namespace rek5 {
namespace {
struct BodyTag { const char* name; int zone, part, side, slot, geoms; };
const BodyTag tags[] = {
 {"pelvis_3266",3,0,-1,-1,1},
 {"left_hip_pitch_link_3457",12,0,-1,-1,1},
 {"left_hip_roll_link_3425",12,0,-1,-1,1},
 {"left_hip_yaw_link_2943",12,0,-1,-1,1},
 {"left_knee_link_3106",14,8,0,4,1},
 {"left_ankle_pitch_link_3033",16,0,-1,-1,1},
 {"left_ankle_roll_link_3045",16,2,0,2,4},
 {"right_hip_pitch_link_3469",13,0,-1,-1,1},
 {"right_hip_roll_link_3345",13,0,-1,-1,1},
 {"right_hip_yaw_link_3191",13,0,-1,-1,1},
 {"right_knee_link_3429",15,8,1,5,1},
 {"right_ankle_pitch_link_3173",17,0,-1,-1,1},
 {"right_ankle_roll_link_3090",17,2,1,3,4},
 {"waist_yaw_link_3359",0,0,-1,-1,1},
 {"waist_roll_link_2894",0,0,-1,-1,1},
 {"torso_link_3347",2,0,-1,-1,2},
 {"left_shoulder_pitch_link_3161",4,0,-1,-1,1},
 {"left_shoulder_roll_link_3360",4,0,-1,-1,1},
 {"left_shoulder_yaw_link_3168",4,0,-1,-1,1},
 {"left_elbow_link_3284",6,0,-1,-1,1},
 {"left_wrist_roll_link_3032",8,0,-1,-1,1},
 {"left_wrist_pitch_link_2914",8,0,-1,-1,1},
 {"left_wrist_yaw_link_3467",8,1,0,0,1},
 {"right_shoulder_pitch_link_3391",5,0,-1,-1,1},
 {"right_shoulder_roll_link_3426",5,0,-1,-1,1},
 {"right_shoulder_yaw_link_3107",5,0,-1,-1,1},
 {"right_elbow_link_3322",7,0,-1,-1,1},
 {"right_wrist_roll_link_3331",9,0,-1,-1,1},
 {"right_wrist_pitch_link_3282",9,0,-1,-1,1},
 {"right_wrist_yaw_link_3293",9,1,1,1,1},
};
int identity(const mjModel* m, int type, const std::string& name) {
    int id=mj_name2id(m,type,name.c_str());
    if(id<0) throw std::runtime_error("Missing pinned measurement identity: "+name);
    return id;
}
__device__ float3 rotate(const float* q, float3 v) {
    float3 t=make_float3(2*(q[2]*v.z-q[3]*v.y),2*(q[3]*v.x-q[1]*v.z),2*(q[1]*v.y-q[2]*v.x));
    return make_float3(v.x+q[0]*t.x+(q[2]*t.z-q[3]*t.y),
                      v.y+q[0]*t.y+(q[3]*t.x-q[1]*t.z),
                      v.z+q[0]*t.z+(q[1]*t.y-q[2]*t.x));
}
__device__ bool unit_root(const float* raw,float* q) {
    float n2=0; bool ok=true;
    for(int i=0;i<4;i++){n2+=raw[i]*raw[i];ok=ok&&isfinite(raw[i]);}
    ok=ok&&isfinite(n2)&&n2>0;
    float n=ok?sqrtf(n2):1;
    for(int i=0;i<4;i++)q[i]=raw[i]/n;
    return ok;
}
__device__ bool floor_height(PhysicsDescriptor p,int arena,int floor,float half,float& height) {
    const float* pos=p.geom_xpos+(arena*p.geoms+floor)*3;
    const float* mat=p.geom_xmat+(arena*p.geoms+floor)*9;
    bool ok=true;
    for(int i=0;i<3;i++)ok=ok&&isfinite(pos[i]);
    for(int i=0;i<9;i++)ok=ok&&isfinite(mat[i]);
    height=pos[2]+fabsf(mat[8])*half;
    return ok&&fabsf(mat[6])<=1e-10f&&fabsf(mat[7])<=1e-10f
        &&fabsf(fabsf(mat[8])-1)<=1e-10f&&isfinite(height);
}
struct FallView {
    PhysicsDescriptor p; Measurement d;
    int roots[2],left[2],right[2];float half;
    float* upright;float* standing;uint8_t* calibrated;
    float* floats;int64_t* integers;uint8_t* valid;
};
FallView fall_view(CombatMeasurement* m) {
    FallView v{};v.p=m->physics->data;v.d=*static_cast<Measurement*>(m->descriptor);
    for(int i=0;i<2;i++){v.roots[i]=m->roots[i];v.left[i]=m->left[i];v.right[i]=m->right[i];}
    v.half=m->floor_half_height;v.upright=m->upright;v.standing=m->standing;v.calibrated=m->calibrated;
    v.floats=m->fall_floats;v.integers=m->fall_integers;v.valid=m->fall_valid;return v;
}
__global__ void calibrate(FallView v) {
    int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=v.p.arenas*2)return;
    int a=row/2,s=row%2;float q[4],height;
    bool ok=unit_root(v.p.qpos+a*72+s*36+3,q);
    for(int i=1;i<4;i++)q[i]=-q[i];
    float3 up=rotate(q,make_float3(0,0,1));
    v.upright[row*3]=up.x;v.upright[row*3+1]=up.y;v.upright[row*3+2]=up.z;
    ok= floor_height(v.p,a,v.d.floor,v.half,height)&&ok;
    const float* pelvis=v.p.xpos+(a*v.p.bodies+v.roots[s])*3;
    for(int i=0;i<3;i++)ok=ok&&isfinite(pelvis[i]);
    float standing=pelvis[2]-height;
    v.standing[row]=standing;v.calibrated[row]=ok&&isfinite(standing)&&standing>DBL_EPSILON;
}
__global__ void all_calibrated(FallView v) {
    // Called only at reset. A failed calibration invalidates all fighter rows.
    for(int row=threadIdx.x+blockIdx.x*blockDim.x;row<v.p.arenas*2;row+=blockDim.x*gridDim.x)
        if(!v.calibrated[row]) atomicExch(v.d.world_bad,1);
}
__global__ void commit_calibrated(FallView v) {
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<v.p.arenas*2&&*v.d.world_bad)v.calibrated[row]=0;
}
__global__ void sample_fall(FallView v,const uint8_t* selected) {
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=v.p.arenas*2||(selected&&!selected[row]))return;
    int a=row/2,s=row%2;float q[4],height;
    bool ok=unit_root(v.p.qpos+a*72+s*36+3,q);
    float3 up=rotate(q,make_float3(v.upright[row*3],v.upright[row*3+1],v.upright[row*3+2]));
    float n2=up.x*up.x+up.y*up.y+up.z*up.z;
    float dot=up.z/(n2>0?sqrtf(n2):1);
    dot=fminf(1,fmaxf(-1,dot));
    float tilt=acosf(dot)*float(180.0/3.14159265358979323846);
    ok=floor_height(v.p,a,v.d.floor,v.half,height)&&ok;
    const float* pelvis=v.p.xpos+(a*v.p.bodies+v.roots[s])*3;
    for(int i=0;i<3;i++)ok=ok&&isfinite(pelvis[i]);
    float standing=v.standing[row],ratio=(pelvis[2]-height)/standing;
    bool left=v.d.floor_contact[a*v.p.bodies+v.left[s]],right=v.d.floor_contact[a*v.p.bodies+v.right[s]];
    int nonfoot=0;
    for(int b=0;b<v.p.bodies;b++) if(v.d.owner[b]==s&&b!=v.left[s]&&b!=v.right[s]&&v.d.floor_contact[a*v.p.bodies+b])nonfoot++;
    float* f=v.floats+row*5;f[0]=tilt;f[1]=ratio;f[2]=.002f;f[3]=height;f[4]=standing;
    int64_t* i=v.integers+row*7;i[0]=1;i[1]=!(left||right);i[2]=left||right;i[3]=nonfoot;i[4]=0;i[5]=left;i[6]=right;
    v.valid[row]=ok&&v.calibrated[row]&&v.d.fall_valid[a]&&isfinite(n2)&&n2>0
        &&isfinite(standing)&&standing>DBL_EPSILON&&isfinite(tilt)&&isfinite(ratio);
}
__global__ void body_velocity(Measurement d,float* velocity) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=d.arenas*d.bodies)return;
    int a=i/d.bodies,b=i%d.bodies;const float* p=d.xipos+i*3;
    const float* com=d.com+(a*d.bodies+d.body_root[b])*3;
    const float* v=d.cvel+i*6;float x=p[0]-com[0],y=p[1]-com[1],z=p[2]-com[2];
    velocity[i*3]=v[3]+(v[1]*z-v[2]*y);
    velocity[i*3+1]=v[4]+(v[2]*x-v[0]*z);
    velocity[i*3+2]=v[5]+(v[0]*y-v[1]*x);
}
__global__ void speeds(Measurement d,float* speed) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=2*d.capacity)return;
    const float* f=d.floats+13*i;
    float x=f[6]-f[9],y=f[7]-f[10],z=f[8]-f[11];speed[i]=sqrtf(x*x+y*y+z*z);
}
__global__ void sequence(int64_t* out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=i;}
__global__ void clear_pairs(Measurement d,const uint8_t* mask) {
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<int64_t(d.arenas)*d.geoms*d.geoms;i+=int64_t(blockDim.x)*gridDim.x)
        if(mask[i/(d.geoms*d.geoms)])d.previous[i]=0;
}
}

CombatMeasurement::~CombatMeasurement(){delete static_cast<Measurement*>(descriptor);}
CombatMeasurement* measurement_create(Physics* p) {
    auto m=std::make_unique<CombatMeasurement>();m->physics=p;
    auto d=new Measurement{};m->descriptor=d;auto& store=m->storage;
    const auto& v=p->data;const mjModel* model=p->model;
    if(model->nbody!=63||model->ngeom!=91||model->nq!=72||model->nv!=70||model->nu!=58
       ||!std::isfinite(model->opt.timestep)||fabs(model->opt.timestep-.002)>1e-15)
        throw std::runtime_error("Measurement model differs from pinned duel");
    std::vector<int64_t> owner(63,-1),zone(91),body_zone(63),part(63),side(63,-1),slot(63,-1),body_root(63),geom_body(91);
    for(int b=0;b<63;b++)body_root[b]=model->body_rootid[b];
    for(int g=0;g<91;g++)geom_body[g]=model->geom_bodyid[g];
    std::set<int> seen;
    for(int s=0;s<2;s++) {
        std::string prefix=s?"opponent__":"player__";
        int root=identity(model,mjOBJ_BODY,prefix+tags[0].name);m->roots[s]=root;
        int rootjoint=model->body_jntadr[root];
        if(model->body_jntnum[root]<1||rootjoint<0||model->jnt_type[rootjoint]!=mjJNT_FREE
           ||model->jnt_qposadr[rootjoint]!=s*36||model->jnt_dofadr[rootjoint]!=s*35)
            throw std::runtime_error("Root free-joint layout mismatch");
        int count=0;
        for(int b=1;b<63;b++)for(int k=b;k>0;k=model->body_parentid[k])if(k==root){
            if(owner[b]>=0)throw std::runtime_error("Fighter ownership overlap");owner[b]=s;count++;break;
        }
        if(count!=30)throw std::runtime_error("Fighter body count mismatch");
        for(auto tag:tags){
            int b=identity(model,mjOBJ_BODY,prefix+tag.name);
            if(!seen.insert(b).second||owner[b]!=s||std::count(geom_body.begin(),geom_body.end(),b)!=tag.geoms)
                throw std::runtime_error("Pinned body geometry map mismatch");
            body_zone[b]=tag.zone;part[b]=tag.part;side[b]=tag.side;slot[b]=tag.slot;
        }
        m->left[s]=identity(model,mjOBJ_BODY,prefix+tags[6].name);
        m->right[s]=identity(model,mjOBJ_BODY,prefix+tags[12].name);
        std::map<std::pair<int,int>,int> masks;
        for(int g=0;g<91;g++)if(owner[geom_body[g]]==s)masks[{model->geom_contype[g],model->geom_conaffinity[g]}]++;
        std::map<std::pair<int,int>,int> wanted=s?std::map<std::pair<int,int>,int>{{{9,5},12},{{9,6},25}}
            :std::map<std::pair<int,int>,int>{{{5,9},12},{{5,10},25}};
        if(masks!=wanted)throw std::runtime_error("Pinned fighter contact masks mismatch");
    }
    for(int g=0;g<91;g++)zone[g]=body_zone[geom_body[g]];
    for(int s=0;s<2;s++){
        std::string prefix=s?"opponent__":"player__";
        int h=identity(model,mjOBJ_GEOM,prefix+"mjgeom_3064"),t=identity(model,mjOBJ_GEOM,prefix+"mjgeom_3285");
        int torso=identity(model,mjOBJ_BODY,prefix+tags[15].name);
        if(h==t||geom_body[h]!=torso||geom_body[t]!=torso)throw std::runtime_error("Head/torso geometry mismatch");
        zone[h]=1;
    }
    int floor=identity(model,mjOBJ_GEOM,"arena_Collider_Floor_Rektagon");
    if(model->geom_type[floor]!=mjGEOM_BOX||geom_body[floor]!=0)throw std::runtime_error("Pinned floor mismatch");
    for(int k=0;k<3;k++)if(!std::isfinite(model->geom_size[3*floor+k])||model->geom_size[3*floor+k]<=0)
        throw std::runtime_error("Invalid floor extents");
    m->floor_half_height=float(model->geom_size[3*floor+2]);
    d->arenas=v.arenas;d->bodies=v.bodies;d->geoms=v.geoms;d->capacity=v.capacity;d->floor=floor;
    d->geom=v.contact_geom;d->world=v.contact_world;d->nacon=v.nacon;d->dist=v.contact_dist;
    d->pos=v.contact_pos;d->frame=v.contact_frame;d->time=v.time;d->xpos=v.xpos;d->xipos=v.xipos;d->com=v.com;d->cvel=v.cvel;
    d->geom_body=store.upload(geom_body);d->body_root=store.upload(body_root);d->owner=store.upload(owner);d->zone=store.upload(zone);
    d->part=store.upload(part);d->side=store.upload(side);d->slot=store.upload(slot);
    size_t a=v.arenas,b=v.bodies,c=v.capacity,pairs=a*size_t(v.geoms)*v.geoms,rows=2*a;
    d->previous=store.alloc<uint8_t>(pairs);d->expected=store.alloc<int64_t>(a);d->first=store.alloc<int32_t>(pairs);
    d->floor_count=store.alloc<int32_t>(a*b);d->fall_bad=store.alloc<int32_t>(a);d->hit_bad=store.alloc<int32_t>(a);
    d->candidate_bad=store.alloc<int32_t>(a);d->world_bad=store.alloc<int32_t>(1);d->capacity_overflow=store.alloc<uint8_t>(1);
    d->fall_valid=store.alloc<uint8_t>(a);d->floor_contact=store.alloc<uint8_t>(a*b);d->base_valid=store.alloc<uint8_t>(a);
    m->scan_valid=d->scan_valid=store.alloc<uint8_t>(a);m->hit_integers=d->integers=store.alloc<int64_t>(2*c*12);
    m->hit_floats=d->floats=store.alloc<float>(2*c*13);m->candidate_valid=d->candidate_valid=store.alloc<uint8_t>(2*c);
    d->keys=store.alloc<int64_t>(2*c);m->counts=d->counts=store.alloc<int64_t>(a);
    d->velocity=store.alloc<float>(a*b*3);d->speed=store.alloc<float>(2*c);
    m->order=store.alloc<int64_t>(2*c);m->offsets=store.alloc<int64_t>(a);
    m->indices=store.alloc<int64_t>(2*c);m->sorted_keys=store.alloc<int64_t>(2*c);
    m->upright=store.alloc<float>(rows*3);m->standing=store.alloc<float>(rows);m->calibrated=store.alloc<uint8_t>(rows);
    m->fall_floats=store.alloc<float>(rows*5);m->fall_integers=store.alloc<int64_t>(rows*7);m->fall_valid=store.alloc<uint8_t>(rows);
    sequence<<<(2*c+127)/128,128,0,p->stream>>>(m->indices,int(2*c));
    cuda_check(cub::DeviceRadixSort::SortPairs(nullptr,m->sort_bytes,d->keys,m->sorted_keys,m->indices,m->order,int(2*c),0,64,p->stream));
    m->sort_temp=store.alloc<uint8_t>(m->sort_bytes);
    cuda_check(cub::DeviceScan::ExclusiveSum(nullptr,m->scan_bytes,m->counts,m->offsets,int(a),p->stream));
    m->scan_temp=store.alloc<uint8_t>(m->scan_bytes);
    cuda_check(cudaGetLastError());return m.release();
}
void measurement_reset(CombatMeasurement* m,cudaStream_t s) {
    auto d=static_cast<Measurement*>(m->descriptor);int rows=d->arenas*2;
    cuda_check(cudaMemsetAsync(d->previous,0,size_t(d->arenas)*d->geoms*d->geoms,s));
    cuda_check(cudaMemsetAsync(d->expected,0,size_t(d->arenas)*sizeof(int64_t),s));
    cuda_check(cudaMemsetAsync(d->world_bad,0,sizeof(int32_t),s));
    auto v=fall_view(m);calibrate<<<(rows+127)/128,128,0,s>>>(v);
    all_calibrated<<<(rows+127)/128,128,0,s>>>(v);commit_calibrated<<<(rows+127)/128,128,0,s>>>(v);
    cuda_check(cudaGetLastError());
}
void measurement_clear_contacts(CombatMeasurement* m,const uint8_t* mask,cudaStream_t s) {
    auto d=*static_cast<Measurement*>(m->descriptor);
    clear_pairs<<<std::min(4096,(d.arenas*d.geoms*d.geoms+127)/128),128,0,s>>>(d,mask);cuda_check(cudaGetLastError());
}
void measurement_sample(CombatMeasurement* m,int substep,cudaStream_t s) {
    auto d=static_cast<Measurement*>(m->descriptor);int rows=d->arenas*2;
    cuda_check(static_cast<cudaError_t>(rek_measurement_facts(d,s)));
    sample_fall<<<(rows+127)/128,128,0,s>>>(fall_view(m),nullptr);
    body_velocity<<<(d->arenas*d->bodies+127)/128,128,0,s>>>(*d,const_cast<float*>(d->velocity));
    cuda_check(static_cast<cudaError_t>(rek_measurement_hits_prepare(d,substep,s)));
    speeds<<<(2*d->capacity+127)/128,128,0,s>>>(*d,const_cast<float*>(d->speed));
    cuda_check(static_cast<cudaError_t>(rek_measurement_hits_finish(d,substep,s)));
    cuda_check(cub::DeviceRadixSort::SortPairs(m->sort_temp,m->sort_bytes,d->keys,m->sorted_keys,m->indices,m->order,2*d->capacity,0,64,s));
    cuda_check(cub::DeviceScan::ExclusiveSum(m->scan_temp,m->scan_bytes,m->counts,m->offsets,d->arenas,s));
    cuda_check(cudaGetLastError());
}
void measurement_sample_reset_fall(CombatMeasurement* m,const uint8_t* rows,cudaStream_t s) {
    auto d=static_cast<Measurement*>(m->descriptor);
    cuda_check(static_cast<cudaError_t>(rek_measurement_facts(d,s)));
    sample_fall<<<(d->arenas*2+127)/128,128,0,s>>>(fall_view(m),rows);cuda_check(cudaGetLastError());
}
}
