// Native, Python-free validation of the pinned physics adapter. Default mode
// uses CPU kinematics only. --gpu additionally tests a two-arena CUDA batch.
#include "physics.cuh"
#include <openssl/evp.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <limits>
#include <stdexcept>

static void check(bool value,const char* message) {
    if (!value) throw std::runtime_error(message);
}
static void ck(cudaError_t value) { check(value==cudaSuccess,cudaGetErrorString(value)); }
static std::string digest(const std::vector<float>& values) {
    unsigned char hash[EVP_MAX_MD_SIZE]; unsigned int size=0;
    check(EVP_Digest(values.data(),values.size()*sizeof(float),hash,&size,EVP_sha256(),nullptr)==1 && size==32,"SHA256 failed");
    char text[65];
    for (int i=0;i<32;++i) std::snprintf(text+i*2,3,"%02x",hash[i]);
    return text;
}
static std::vector<float> download(const float* values,size_t count) {
    std::vector<float> output(count);
    ck(cudaMemcpy(output.data(),values,count*sizeof(float),cudaMemcpyDeviceToHost));
    return output;
}
static float difference(const std::vector<float>& left,const std::vector<float>& right) {
    check(left.size()==right.size(),"difference size mismatch");
    float maximum=0;
    for (size_t i=0;i<left.size();++i) {
        check(std::isfinite(left[i]) && std::isfinite(right[i]),"nonfinite field");
        maximum=std::max(maximum,std::abs(left[i]-right[i]));
    }
    return maximum;
}

int main(int argc,char** argv) {
    try {
        check(argc==3 || (argc==4 && std::strcmp(argv[3],"--gpu")==0),"usage: physics_probe XML EXPORT [--gpu]");
        const bool gpu=argc==4;
        cudaStream_t stream=nullptr;
        if (gpu) ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        using Owner=std::unique_ptr<rek5::Physics,decltype(&rek5::physics_close)>;
        Owner physics(gpu?rek5::physics_create(argv[1],argv[2],2,stream)
                         :rek5::physics_load_model(argv[1],argv[2]),rek5::physics_close);
        const char* names[]={"bodies","shapes","joints","roots"};
        const std::vector<float>* packed[]={&physics->packed_bodies,&physics->packed_shapes,&physics->packed_joints,&physics->packed_roots};
        const char* expected[]={"35862eac1c0d36c3313d570397d75ae987642463b5feb0698eaab999130e7c08",
            "d9f33b962b11c795a3da3c0f0e9c264cbd5320a1570a62771fbad2b33698a889",
            "25c498005b43326f6a3dceafed6a6e921ae9b21b3943ab953f85b36d23f02f9b",
            "f2508f9ab746f01c76c63131d66971848027d6dc2db7b7953cfe01c9a3236d53"};
        std::printf("{\"gpu\":%s,\"packed_sha256\":{",gpu?"true":"false");
        for (int i=0;i<4;++i) {
            const auto actual=digest(*packed[i]);
            check(actual==expected[i],"packed export differs from existing capsule baseline");
            std::printf("%s\"%s\":\"%s\"",i?",":"",names[i],actual.c_str());
        }
        // Reconstruct both link and inertial frames from the actual float ABI.
        std::unique_ptr<mjData,decltype(&mj_deleteData)> source(mj_makeData(physics->model),mj_deleteData);
        check(bool(source),"mj_makeData failed");
        for (int i=0;i<72;++i) source->qpos[i]=physics->initial_qpos[i];
        mj_kinematics(physics->model,source.get());
        double position_error=0,rotation_error=0;
        for (int b=0;b<63;++b) {
            const float* map=physics->host_body_map.data()+b*18;
            const int owner=int(map[0]);
            const float identity[11]={0,0,0,0,0,0,1,0,0,0,0};
            const float* rigid=owner?physics->packed_bodies.data()+(owner-1)*11:identity;
            mjtNum world_q[4]={rigid[6],rigid[3],rigid[4],rigid[5]};
            for (int frame=0;frame<2;++frame) {
                const float* local=map+(frame?11:4);
                mjtNum local_position[3]={local[0],local[1],local[2]}, rotated[3];
                mju_rotVecQuat(rotated,local_position,world_q);
                const mjtNum* target_position=(frame?source->xipos:source->xpos)+b*3;
                for (int k=0;k<3;++k) position_error=std::max(position_error,std::abs(rotated[k]+rigid[k]-target_position[k]));
                mjtNum local_q[4]={local[6],local[3],local[4],local[5]}, composed[4], matrix[9];
                mju_mulQuat(composed,world_q,local_q); mju_normalize4(composed); mju_quat2Mat(matrix,composed);
                const mjtNum* target_rotation=(frame?source->ximat:source->xmat)+b*9;
                for (int k=0;k<9;++k) rotation_error=std::max(rotation_error,std::abs(matrix[k]-target_rotation[k]));
            }
        }
        check(position_error<2e-6 && rotation_error<2e-6,"source frame reconstruction differs");
        std::printf("},\"source_position_max_error_m\":%.9g,\"source_rotation_max_error\":%.9g",position_error,rotation_error);
        if (gpu) {
            rek5::physics_check_status(physics.get());
            auto& d=physics->data;
            const auto initial=download(d.qpos,144);
            std::vector<float> expected_state=physics->initial_qpos;
            expected_state.insert(expected_state.end(),physics->initial_qpos.begin(),physics->initial_qpos.end());
            const float initial_error=difference(initial,expected_state);
            check(initial_error<2e-6,"initial GPU generalized state differs");
            std::vector<float> controls(116);
            for (int a=0;a<2;++a) for (int j=0;j<58;++j) controls[a*58+j]=initial[a*72+physics->joint_qpos[j]];
            ck(cudaMemcpy(physics->ctrl,controls.data(),controls.size()*sizeof(float),cudaMemcpyHostToDevice));
            for (int step=0;step<20;++step) rek5::physics_step(physics.get(),physics->ctrl);
            rek5::physics_check_status(physics.get());
            const auto stepped=download(d.qpos,144), velocities=download(d.qvel,140);
            check(std::memcmp(stepped.data(),stepped.data()+72,72*sizeof(float))==0,"duplicate arenas diverged");
            const auto body_before=download(d.xpos,2*63*3);
            auto reset=stepped;
            std::copy(initial.begin(),initial.begin()+72,reset.begin());
            auto reset_velocities=velocities;
            std::fill(reset_velocities.begin(),reset_velocities.begin()+70,0.0f);
            ck(cudaMemcpy(d.qpos,reset.data(),reset.size()*sizeof(float),cudaMemcpyHostToDevice));
            ck(cudaMemcpy(d.qvel,reset_velocities.data(),reset_velocities.size()*sizeof(float),cudaMemcpyHostToDevice));
            uint8_t* mask=nullptr;
            ck(cudaMalloc(&mask,2)); const uint8_t selected[2]={1,0};
            ck(cudaMemcpy(mask,selected,2,cudaMemcpyHostToDevice));
            rek5::physics_forward_selected(physics.get(),mask);
            rek5::physics_check_status(physics.get());
            const auto after=download(d.qpos,144), after_velocities=download(d.qvel,140);
            const auto body_after=download(d.xpos,2*63*3);
            check(std::memcmp(after.data()+72,stepped.data()+72,72*sizeof(float))==0,"masked forward changed unselected qpos");
            check(std::memcmp(after_velocities.data()+70,velocities.data()+70,70*sizeof(float))==0,"masked forward changed unselected qvel");
            check(std::memcmp(body_after.data()+63*3,body_before.data()+63*3,63*3*sizeof(float))==0,"masked forward changed unselected body fields");
            check(difference(std::vector<float>(after.begin(),after.begin()+72),physics->initial_qpos)<3e-6,"masked reset pose mismatch");
            for (int step=0;step<20;++step) rek5::physics_step(physics.get(),physics->ctrl);
            rek5::physics_check_status(physics.get());
            const uint8_t none[2]={0,0};
            ck(cudaMemcpy(mask,none,2,cudaMemcpyHostToDevice));
            cudaGraph_t graph=nullptr;
            cudaGraphExec_t executable=nullptr;
            ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
            rek5::physics_step(physics.get(),physics->ctrl);
            rek5::physics_forward_selected(physics.get(),mask);
            ck(cudaStreamEndCapture(stream,&graph));
            ck(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
            ck(cudaGraphLaunch(executable,stream));
            rek5::physics_check_status(physics.get());
            ck(cudaGraphExecDestroy(executable)); ck(cudaGraphDestroy(graph));

            // Derived fields must fail the same reporting check as the facade.
            const float invalid=std::numeric_limits<float>::quiet_NaN();
            const auto valid_xpos=download(d.xpos,1);
            ck(cudaMemcpy(d.xpos,&invalid,sizeof(float),cudaMemcpyHostToDevice));
            bool rejected_derived=false;
            try { rek5::physics_check_status(physics.get()); }
            catch (const std::runtime_error&) { rejected_derived=true; }
            check(rejected_derived,"nonfinite derived body state was accepted");
            ck(cudaMemcpy(d.xpos,valid_xpos.data(),sizeof(float),cudaMemcpyHostToDevice));

            // Resetting a failed arena must never clear its cumulative failure.
            ck(cudaMemcpy(d.qpos,&invalid,sizeof(float),cudaMemcpyHostToDevice));
            ck(cudaMemcpy(mask,selected,2,cudaMemcpyHostToDevice));
            rek5::physics_forward_selected(physics.get(),mask);
            bool rejected_nan=false;
            try { rek5::physics_check_status(physics.get()); }
            catch (const std::runtime_error&) { rejected_nan=true; }
            check(rejected_nan,"nonfinite generalized state was accepted");
            ck(cudaMemcpy(d.qpos,initial.data(),72*sizeof(float),cudaMemcpyHostToDevice));
            ck(cudaMemsetAsync(d.qvel,0,70*sizeof(float),stream));
            rek5::physics_forward_selected(physics.get(),mask);
            const auto persistent=rek5::physics_stats(physics.get());
            check(persistent[1]!=0,"masked reset cleared persistent nonfinite failure");
            ck(cudaFree(mask));
            std::printf(",\"initial_gpu_max_error\":%.9g,\"gpu_steps\":41,\"masked_reset_isolated\":true,\"graph_capture_passed\":true,\"nonfinite_rejected\":true,\"failure_persists_after_reset\":true",initial_error);
        }
        std::printf(",\"passed\":true}\n");
        physics.reset();
        if (gpu) ck(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr,"%s\n",error.what()); return 1;
    }
}
