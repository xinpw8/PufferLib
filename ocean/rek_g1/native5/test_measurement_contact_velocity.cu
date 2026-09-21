// Direct MuJoCo FK/comVel reference plus production contact packing/scoring.
// Synthetic contacts do not claim physical contact or controller parity.
#include "measurement.cu"
#include "recovered_contact_rules.cuh"
#include <cstdio>
#include <cstring>
#include <limits>

static void require(bool value,const char* text){if(!value)throw std::runtime_error(text);}
template<class T> static std::vector<T> read_device(const T* pointer,size_t count){
    std::vector<T> result(count);rek5::cuda_check(cudaMemcpy(result.data(),pointer,count*sizeof(T),cudaMemcpyDeviceToHost));return result;
}
struct ScoreResult {int processed,accepted;float points;};
__global__ void score_candidate(Measurement d,ScoreResult* result){
    if(threadIdx.x||blockIdx.x)return;
    *result={};if(!d.scan_valid[0]||!d.candidate_valid[0])return;
    RekG1HitContact contact{};const float* f=d.floats;
    for(int k=0;k<3;k++){
        contact.striker_body_position_world[k]=f[k];contact.target_body_position_world[k]=f[3+k];
        contact.striker_body_linear_velocity_world[k]=f[6+k];contact.target_body_linear_velocity_world[k]=f[9+k];
    }
    RekG1ImpactEvent impact{0,.1f,.1f,1,REK_G1_AIM_LIMB_LEFT_UPPER_BODY};
    contact.strike_intent={&impact,1,0,50,1,1,1,0};
    contact.relative_speed_mps=f[12];contact.time_seconds=1;
    contact.striker_part=REK_G1_BODY_PART_HAND;contact.striker_side=REK_G1_HAND_LEFT;
    contact.target_zone=REK_G1_BODY_ZONE_TORSO;contact.striker_fighter=0;contact.target_fighter=1;
    contact.striker_body_slot=0;contact.is_enter=contact.round_active=contact.striker_upright=contact.target_upright=contact.target_standing=1;
    RekG1HitDetectorState state{};RekG1HitResult output{};
    auto config=rek5_recovered::rek_g1_current_build_hit_detector_config();
    result->processed=rek5_recovered::rek_g1_hit_detector_process(&state,&config,&contact,&output);
    result->accepted=output.score_accepted;result->points=output.points_awarded;
}
static void contact_case(int mode){
    rek5::DeviceStorage s;Measurement d{};d.arenas=1;d.bodies=3;d.geoms=3;d.capacity=1;d.floor=0;
    d.geom=s.upload(std::vector<int32_t>{1,2});d.world=s.upload(std::vector<int32_t>{0});d.nacon=s.upload(std::vector<int32_t>{1});
    d.dist=s.upload(std::vector<float>{-.001f});d.pos=s.upload(std::vector<float>{0,0,0});d.frame=s.upload(std::vector<float>{1,0,0,0,1,0,0,0,1});d.time=s.upload(std::vector<float>{1});
    d.xpos=s.upload(std::vector<float>{0,0,0,0,0,1,1,0,1});
    d.xipos=s.upload(std::vector<float>{0,0,0,1,0,1,1,0,1});d.com=s.alloc<float>(9);
    const float fast=rek5_recovered::rek_g1_current_build_hit_detector_config().speed_threshold_mps+1;
    std::vector<float> cv(18);cv[8]=fast;
    if(mode==1){cv[8]=-fast;cv[10]=fast;}
    if(mode==2){cv[10]=fast;cv[16]=fast;}
    if(mode==3)cv[16]=fast;
    if(mode==4)cv[10]=std::numeric_limits<float>::quiet_NaN();
    d.cvel=s.upload(cv);d.geom_body=s.upload(std::vector<int64_t>{0,1,2});d.body_root=s.upload(std::vector<int64_t>{0,0,0});
    d.owner=s.upload(std::vector<int64_t>{-1,0,1});d.zone=s.upload(std::vector<int64_t>{0,8,2});
    d.part=s.upload(std::vector<int64_t>{0,1,0});d.side=s.upload(std::vector<int64_t>{-1,0,-1});d.slot=s.upload(std::vector<int64_t>{-1,0,-1});
    d.previous=s.alloc<uint8_t>(9);d.expected=s.alloc<int64_t>(1);d.first=s.alloc<int32_t>(9);d.floor_count=s.alloc<int32_t>(3);
    d.fall_bad=s.alloc<int32_t>(1);d.hit_bad=s.alloc<int32_t>(1);d.candidate_bad=s.alloc<int32_t>(1);d.world_bad=s.alloc<int32_t>(1);
    d.capacity_overflow=s.alloc<uint8_t>(1);d.fall_valid=s.alloc<uint8_t>(1);d.floor_contact=s.alloc<uint8_t>(3);d.base_valid=s.alloc<uint8_t>(1);d.scan_valid=s.alloc<uint8_t>(1);
    d.integers=s.alloc<int64_t>(24);d.floats=s.alloc<float>(26);d.candidate_valid=s.alloc<uint8_t>(2);d.keys=s.alloc<int64_t>(2);d.counts=s.alloc<int64_t>(1);
    auto* velocities=s.alloc<float>(9);auto* speeds=s.alloc<float>(2);d.velocity=velocities;d.speed=speeds;
    rek5::body_velocity<<<1,32>>>(d,velocities);
    rek5::cuda_check(static_cast<cudaError_t>(rek_measurement_facts(&d,nullptr)));
    rek5::cuda_check(static_cast<cudaError_t>(rek_measurement_hits_prepare(&d,0,nullptr)));
    rek5::speeds<<<1,32>>>(d,speeds);
    rek5::cuda_check(static_cast<cudaError_t>(rek_measurement_hits_finish(&d,0,nullptr)));
    auto* result=s.alloc<ScoreResult>(1);score_candidate<<<1,1>>>(d,result);
    rek5::cuda_check(cudaGetLastError());auto actual=read_device(result,1)[0];
    const bool invalid=mode==4,accepted=mode==1||mode==3;
    require(read_device(d.scan_valid,1)[0]==!invalid,"candidate validity changed");
    require(actual.processed==!invalid&&actual.accepted==accepted,"raw cvel scoring result mismatch");
    require(actual.points==(accepted?1:0),"scoring points mismatch");
    if(!invalid){auto packed=read_device(d.floats,26);require(packed[12]==(accepted?fast:0),"packed raw relative speed mismatch");}
}
int main(int argc,char** argv){try{
    require(argc==2||(argc==3&&std::string(argv[2])=="--cpu-only"),"usage: test_measurement_contact_velocity MODEL_XML [--cpu-only]");
    const bool cpu_only=argc==3;char error[1024]{};
    std::unique_ptr<mjModel,decltype(&mj_deleteModel)> model(mj_loadXML(argv[1],nullptr,error,sizeof(error)),mj_deleteModel);
    require(bool(model),error);require(model->nq==72&&model->nv==70,"pinned full-pose layout mismatch");
    std::unique_ptr<mjData,decltype(&mj_deleteData)> data(mj_makeData(model.get()),mj_deleteData);
    require(bool(data),"MuJoCo data allocation failed");
    double max_shift_difference=0,max_cast_error=0;size_t components=0;
    for(int scenario=0;scenario<12;scenario++){
        mj_resetData(model.get(),data.get());
        for(int side=0;side<2;side++){
            const int q=side*36,v=side*35;const double tilt=.05*scenario*(side?1:-1);
            data->qpos[q+3]=std::cos(tilt/2);data->qpos[q+4]=std::sin(tilt/2);data->qpos[q+5]=data->qpos[q+6]=0;
            data->qvel[v]=scenario*.1;data->qvel[v+1]=-.2*scenario;
            data->qvel[v+5]=scenario?(side?-3:4):0;
            for(int j=6;j<35;j++)data->qvel[v+j]=scenario>1?std::sin(double(j+scenario))*.5:0;
        }
        mj_kinematics(model.get(),data.get());mj_comPos(model.get(),data.get());mj_comVel(model.get(),data.get());
        std::vector<float> cv(model->nbody*6),expected(model->nbody*3);
        for(int b=0;b<model->nbody;b++){
            for(int k=0;k<6;k++)cv[6*b+k]=float(data->cvel[6*b+k]);
            const double* raw=data->cvel+6*b;const double* com=data->subtree_com+3*model->body_rootid[b];
            const double* point=data->xipos+3*b;double offset[3]={point[0]-com[0],point[1]-com[1],point[2]-com[2]};
            const double shift[3]={raw[1]*offset[2]-raw[2]*offset[1],raw[2]*offset[0]-raw[0]*offset[2],raw[0]*offset[1]-raw[1]*offset[0]};
            for(int k=0;k<3;k++){
                expected[b*3+k]=float(raw[3+k]);max_shift_difference=std::max(max_shift_difference,std::abs(shift[k]));
                max_cast_error=std::max(max_cast_error,std::abs(double(expected[b*3+k])-raw[3+k]));components++;
            }
        }
        if(!cpu_only){
            rek5::DeviceStorage storage;Measurement d{};d.arenas=1;d.bodies=model->nbody;d.cvel=storage.upload(cv);
            // Null point/COM pointers also prove this production kernel does not
            // depend on the old shifted-velocity inputs.
            auto* velocity=storage.alloc<float>(expected.size());
            rek5::body_velocity<<<(model->nbody+127)/128,128>>>(d,velocity);rek5::cuda_check(cudaGetLastError());
            auto actual=read_device(velocity,expected.size());
            require(std::memcmp(actual.data(),expected.data(),expected.size()*sizeof(float))==0,"direct MuJoCo raw cvel copy differs");
        }
    }
    require(max_shift_difference>.1,"rotating-body fixture did not distinguish quantities");
    if(!cpu_only)for(int mode=0;mode<5;mode++)contact_case(mode);
    std::printf("{\"test\":\"physical_contact_raw_cvel\",\"passed\":true,\"cpu_only\":%s,\"scenarios\":12,\"components\":%zu,\"max_legacy_shift_difference_m_s\":%.17g,\"max_cpu_float_cast_error_m_s\":%.17g,\"gpu_raw_copy_bitwise_equal\":%s,\"contact_scoring_cases\":%d,\"physics_steps\":0,\"controller_inference\":false}\n",cpu_only?"true":"false",components,max_shift_difference,max_cast_error,cpu_only?"false":"true",cpu_only?0:5);
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
