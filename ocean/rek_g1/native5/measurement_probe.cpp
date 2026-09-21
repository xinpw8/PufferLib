// Native differential fixtures. The unchanged fused kernels execute on the
// CPU as a facts/history oracle; new velocity/fall math is checked separately.
#include "measurement.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>
#define rek_measurement_descriptor_size oracle_descriptor_size
#define rek_measurement_facts oracle_facts
#define rek_measurement_hits_prepare oracle_hits_prepare
#define rek_measurement_hits_finish oracle_hits_finish
namespace oracle {
#include "../gpu_combat_measurement_fused.cu"
}
#undef rek_measurement_descriptor_size
#undef rek_measurement_facts
#undef rek_measurement_hits_prepare
#undef rek_measurement_hits_finish
#undef KERNEL
#undef DEVICE
#undef HOST_DEVICE
#undef EACH
#undef LAUNCH
#undef STATUS

static void check(bool condition,const char* message) {
    if(!condition) throw std::runtime_error(message);
}
static void ck(cudaError_t status) { check(status==cudaSuccess,cudaGetErrorString(status)); }
template<class T> static std::vector<T> get(const T* device,size_t count) {
    std::vector<T> output(count);ck(cudaMemcpy(output.data(),device,count*sizeof(T),cudaMemcpyDeviceToHost));return output;
}
template<class T> static void put(T* device,const std::vector<T>& input) {
    ck(cudaMemcpy(device,input.data(),input.size()*sizeof(T),cudaMemcpyHostToDevice));
}
template<class T> static void same(const T* device,const T* expected,size_t count,const char* label) {
    auto actual=get(device,count);
    if(std::memcmp(actual.data(),expected,count*sizeof(T))) throw std::runtime_error(label);
}
struct Reference {
    oracle::Measurement d{};
    oracle::Measurement device{};
    std::vector<void*> allocations;
    explicit Reference(rek5::CombatMeasurement* measurement) {
        std::memcpy(&device,measurement->descriptor,sizeof(device));d=device;
        const size_t a=d.arenas,b=d.bodies,c=d.capacity,pairs=a*d.geoms*d.geoms;
#define C(field,n) d.field=clone(d.field,n)
        C(geom,2*c);C(world,c);C(nacon,1);C(dist,c);C(pos,3*c);C(frame,9*c);C(time,a);
        C(xpos,a*b*3);C(xipos,a*b*3);C(com,a*b*3);C(cvel,a*b*6);
        C(geom_body,d.geoms);C(body_root,b);C(owner,b);C(zone,d.geoms);C(part,b);C(side,b);C(slot,b);
        C(previous,pairs);C(expected,a);C(first,pairs);C(floor_count,a*b);C(fall_bad,a);C(hit_bad,a);
        C(candidate_bad,a);C(world_bad,1);C(capacity_overflow,1);C(fall_valid,a);C(floor_contact,a*b);
        C(base_valid,a);C(scan_valid,a);C(integers,2*c*12);C(floats,2*c*13);C(candidate_valid,2*c);
        C(keys,2*c);C(counts,a);C(velocity,a*b*3);C(speed,2*c);
#undef C
    }
    template<class T> T* clone(T* device_values,size_t count) {
        using Value=typename std::remove_const<T>::type;
        Value* output=static_cast<Value*>(std::malloc(count*sizeof(Value)));check(output,"host allocation failed");
        allocations.push_back(output);ck(cudaMemcpy(output,device_values,count*sizeof(Value),cudaMemcpyDeviceToHost));return output;
    }
    ~Reference(){for(void* p:allocations)std::free(p);}
    void read() {
        const size_t a=d.arenas,b=d.bodies,c=d.capacity;
#define R(field,n) ck(cudaMemcpy(const_cast<typename std::remove_const<typename std::remove_pointer<decltype(d.field)>::type>::type*>(d.field),device.field,(n)*sizeof(*d.field),cudaMemcpyDeviceToHost))
        R(geom,2*c);R(world,c);R(nacon,1);R(dist,c);R(pos,3*c);R(frame,9*c);R(time,a);
        R(xpos,a*b*3);R(xipos,a*b*3);R(com,a*b*3);R(cvel,a*b*6);
#undef R
    }
    void reset() {
        std::memset(d.previous,0,size_t(d.arenas)*d.geoms*d.geoms);
        std::memset(d.expected,0,size_t(d.arenas)*sizeof(int64_t));
    }
    void sample(int substep) {
        check(oracle::oracle_facts(&d,nullptr)==0,"oracle facts failed");
        float* velocity=const_cast<float*>(d.velocity);
        for(int a=0;a<d.arenas;a++)for(int b=0;b<d.bodies;b++) {
            const int offset=a*d.bodies+b;
            // Recovered native contact contract: unshifted linear cvel.
            const float* cv=d.cvel+offset*6;
            velocity[offset*3]=cv[3];velocity[offset*3+1]=cv[4];velocity[offset*3+2]=cv[5];
        }
        check(oracle::oracle_hits_prepare(&d,substep,nullptr)==0,"oracle hits prepare failed");
        float* speed=const_cast<float*>(d.speed);
        for(int i=0;i<2*d.capacity;i++) {
            const float* f=d.floats+13*i;
            float x=f[6]-f[9],y=f[7]-f[10],z=f[8]-f[11];speed[i]=std::sqrt(x*x+y*y+z*z);
        }
        check(oracle::oracle_hits_finish(&d,substep,nullptr)==0,"oracle hits finish failed");
    }
    void compare(rek5::CombatMeasurement* m) {
        size_t a=d.arenas,b=d.bodies,c=d.capacity;
        same(device.previous,d.previous,a*d.geoms*d.geoms,"pair history mismatch");
        same(device.expected,d.expected,a,"substep sequence mismatch");
        same(device.floor_contact,d.floor_contact,a*b,"floor body facts mismatch");
        same(device.fall_valid,d.fall_valid,a,"floor validity mismatch");
        same(device.scan_valid,d.scan_valid,a,"hit scan validity mismatch");
        same(device.counts,d.counts,a,"candidate counts mismatch");
        same(device.candidate_valid,d.candidate_valid,2*c,"candidate validity mismatch");
        same(device.capacity_overflow,d.capacity_overflow,1,"capacity overflow mismatch");
        auto integers=get(device.integers,2*c*12),keys=get(device.keys,2*c);
        auto floats=get(device.floats,2*c*13);
        for(size_t i=0;i<2*c;i++)if(d.candidate_valid[i]) {
            check(std::memcmp(integers.data()+12*i,d.integers+12*i,12*sizeof(int64_t))==0,"candidate integer payload mismatch");
            check(keys[i]==d.keys[i],"candidate key mismatch");
            for(int k=0;k<13;k++)check(std::abs(floats[13*i+k]-d.floats[13*i+k])<=1e-6f,"candidate float payload mismatch");
        }
        std::vector<int64_t> order(2*c);std::iota(order.begin(),order.end(),int64_t(0));
        std::sort(order.begin(),order.end(),[&](int64_t x,int64_t y){return d.keys[x]<d.keys[y];});
        size_t total=0;std::vector<int64_t> offsets(a);
        for(size_t i=0;i<a;i++){offsets[i]=total;total+=d.counts[i];}
        same(m->order,order.data(),total,"CUB directed candidate order mismatch");
        same(m->offsets,offsets.data(),a,"exclusive candidate offsets mismatch");
    }
};

static int geom_for(const mjModel* model,int body) {
    for(int g=0;g<model->ngeom;g++)if(model->geom_bodyid[g]==body)return g;
    throw std::runtime_error("body geometry missing");
}
static void contacts(rek5::Physics* p,const std::vector<std::array<int,3>>& entries) {
    const size_t c=p->data.capacity;
    std::vector<int> geom(c*2,-999),world(c,-999);
    std::vector<float> dist(c,std::numeric_limits<float>::quiet_NaN()),pos(c*3,0),frame(c*9,0);
    for(size_t i=0;i<entries.size();i++) {
        world[i]=entries[i][0];geom[2*i]=entries[i][1];geom[2*i+1]=entries[i][2];dist[i]=-.001f;
        frame[9*i]=frame[9*i+4]=frame[9*i+8]=1;
    }
    put(p->data.contact_geom,geom);put(p->data.contact_world,world);put(p->data.contact_dist,dist);
    put(p->data.contact_pos,pos);put(p->data.contact_frame,frame);put(p->data.nacon,std::vector<int>{int(entries.size())});
}

int main(int argc,char** argv) {
    try {
        check(argc==3,"usage: measurement_probe XML EXPORT");
        using PhysicsOwner=std::unique_ptr<rek5::Physics,decltype(&rek5::physics_close)>;
        PhysicsOwner p(rek5::physics_create(argv[1],argv[2],2,nullptr),rek5::physics_close);
        std::unique_ptr<rek5::CombatMeasurement> m(rek5::measurement_create(p.get()));
        ck(cudaDeviceSynchronize());Reference reference(m.get());
        int samples=0;
        auto sample=[&](int step){reference.read();reference.sample(step);rek5::measurement_sample(m.get(),step,nullptr);ck(cudaDeviceSynchronize());reference.compare(m.get());samples++;};
        auto reset=[&]{rek5::measurement_reset(m.get(),nullptr);reference.reset();ck(cudaDeviceSynchronize());};
        reset();sample(0);
        auto calibrated=get(m->calibrated,4),valid=get(m->fall_valid,4);
        auto fall=get(m->fall_floats,20);
        for(int row=0;row<4;row++) {
            check(calibrated[row]&&valid[row],"real initial reset calibration invalid");
            check(std::abs(fall[row*5])<.03f && std::abs(fall[row*5+1]-1)<1e-6,"real reset tilt/height mismatch");
        }
        auto qpos=get(p->data.qpos,144);std::vector<float> ctrl(116);
        for(int a=0;a<2;a++)for(int j=0;j<58;j++)ctrl[a*58+j]=qpos[a*72+p->joint_qpos[j]];
        put(p->ctrl,ctrl);rek5::physics_step(p.get(),p->ctrl);rek5::physics_check_status(p.get());sample(1);
        const int floor=mj_name2id(p->model,mjOBJ_GEOM,"arena_Collider_Floor_Rektagon");
        const int hand0=geom_for(p->model,mj_name2id(p->model,mjOBJ_BODY,"player__left_wrist_yaw_link_3467"));
        const int hand1=geom_for(p->model,mj_name2id(p->model,mjOBJ_BODY,"opponent__right_wrist_yaw_link_3293"));
        const int head1=mj_name2id(p->model,mjOBJ_GEOM,"opponent__mjgeom_3064");
        const int foot0=geom_for(p->model,m->left[0]),pelvis0=geom_for(p->model,m->roots[0]);
        std::vector<float> cvel(2*63*6),xipos(2*63*3),com(2*63*3);
        for(size_t i=0;i<cvel.size();i++)cvel[i]=float(int(i%17)-8)*.125f;
        for(size_t i=0;i<xipos.size();i++){xipos[i]=float(int(i%13)-6)*.25f;com[i]=float(int(i%7)-3)*.125f;}
        put(p->data.cvel,cvel);put(p->data.xipos,xipos);put(p->data.com,com);
        reset();
        const std::vector<std::array<int,3>> burst={{1,hand1,hand0},{0,hand0,head1},{0,head1,hand0},{0,floor,foot0},{0,foot0,floor},{0,floor,pelvis0}};
        contacts(p.get(),burst);sample(0);
        auto counts=get(m->counts,2),ints=get(m->fall_integers,28);
        check(counts[0]==1&&counts[1]==2,"directed duplicate fixture candidate counts incorrect");
        check(ints[1]==0&&ints[2]==1&&ints[3]==1&&ints[5]==1&&ints[6]==0,"floor body deduplication incorrect");
        sample(1);counts=get(m->counts,2);check(counts[0]==0&&counts[1]==0,"persistent contact was re-entered");
        contacts(p.get(),{});sample(2);contacts(p.get(),burst);sample(3);
        uint8_t* selected=nullptr;ck(cudaMalloc(&selected,4));put(selected,std::vector<uint8_t>{1,0,0,0});
        rek5::measurement_clear_contacts(m.get(),selected,nullptr);
        std::memset(reference.d.previous,0,size_t(reference.d.geoms)*reference.d.geoms);sample(4);
        counts=get(m->counts,2);check(counts[0]==1&&counts[1]==0,"selected contact history reset incorrect");

        // Invalid input transactions must retain history and sequence exactly.
        for(int mode=0;mode<9;mode++) {
            reset();contacts(p.get(),{{0,hand0,head1},{1,hand1,hand0}});
            if(mode==0)put(p->data.contact_world,std::vector<int>{-1});
            if(mode==1)put(p->data.contact_geom,std::vector<int>{999});
            if(mode==2)put(p->data.contact_geom,std::vector<int>{hand0,hand0});
            if(mode==3)put(p->data.contact_dist,std::vector<float>{std::numeric_limits<float>::quiet_NaN()});
            if(mode==4)put(p->data.contact_pos,std::vector<float>{std::numeric_limits<float>::infinity()});
            if(mode==5)put(p->data.contact_frame,std::vector<float>{std::numeric_limits<float>::quiet_NaN()});
            if(mode==6)put(p->data.nacon,std::vector<int>{-1});
            if(mode==7)put(p->data.nacon,std::vector<int>{p->data.capacity+1});
            if(mode==8)contacts(p.get(),{});
            sample(0);
        }
        // Selective fall refresh preserves all unselected row facts.
        reset();contacts(p.get(),{});sample(0);fall=get(m->fall_floats,20);ints=get(m->fall_integers,28);valid=get(m->fall_valid,4);
        const float half=std::sqrt(.5f);qpos=get(p->data.qpos,144);qpos[3]=half;qpos[4]=half;qpos[5]=qpos[6]=0;
        put(p->data.qpos,qpos);rek5::measurement_sample_reset_fall(m.get(),selected,nullptr);ck(cudaDeviceSynchronize());
        auto tilted=get(m->fall_floats,20);check(std::abs(tilted[0]-90)<.001f,"90 degree fall tilt fixture incorrect");
        check(std::memcmp(tilted.data()+5,fall.data()+5,15*sizeof(float))==0,"selective fall refresh changed unselected floats");
        same(m->fall_integers+7,ints.data()+7,21,"selective fall refresh changed unselected integers");
        same(m->fall_valid+1,valid.data()+1,3,"selective fall refresh changed unselected validity");
        // One corrupt calibration invalidates every row, as row_valid.all does.
        qpos[3]=qpos[4]=qpos[5]=qpos[6]=0;put(p->data.qpos,qpos);reset();calibrated=get(m->calibrated,4);
        for(uint8_t value:calibrated)check(!value,"global calibration validity mismatch");
        // The real constructor must reject NaN timestep metadata.
        p->model->opt.timestep=std::numeric_limits<double>::quiet_NaN();bool rejected=false;
        try{std::unique_ptr<rek5::CombatMeasurement> invalid(rek5::measurement_create(p.get()));}
        catch(const std::runtime_error&){rejected=true;}check(rejected,"NaN model timestep accepted");p->model->opt.timestep=.002;
        ck(cudaFree(selected));
        std::printf("{\"passed\":true,\"differential_samples\":%d,\"real_reset_calibration\":true,\"real_physics_contacts\":true,\"directed_order_and_speeds\":true,\"history_transactions\":true,\"selective_fall_refresh\":true,\"invalid_calibration\":true,\"nan_timestep_rejected\":true}\n",samples);
        return 0;
    }catch(const std::exception& error){std::fprintf(stderr,"%s\n",error.what());return 1;}
}
