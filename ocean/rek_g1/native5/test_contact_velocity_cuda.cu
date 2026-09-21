#include "contact_velocity.h"
#include "device_storage.cuh"
#include <algorithm>
#include <cstdio>
#include <vector>

struct CvelCase {
    FastBodyVelocityFrame frame;
    float yaw[2],vx[2],vy[2],omega[2];
    int limb,target,moving[2];
};
struct CvelAnswer {rek_contact_velocity::Linear a,b;float speed;};
__global__ void compose_cases(const CvelCase* cases,CvelAnswer* out,int count) {
    const int index=int(blockIdx.x*blockDim.x+threadIdx.x);if(index>=count)return;
    const auto& in=cases[index];auto& result=out[index];
    result.a=rek_contact_velocity::compose(in.frame,0,rek_contact_velocity::limb_slot(in.limb),in.yaw[0],in.vx[0],in.vy[0],in.omega[0],in.moving[0]);
    result.b=rek_contact_velocity::compose(in.frame,1,rek_contact_velocity::target_slot(in.target),in.yaw[1],in.vx[1],in.vy[1],in.omega[1],in.moving[1]);
    result.speed=rek_contact_velocity::relative_speed(result.a,result.b);
}
int main(){try{
    constexpr int count=6*9*4*5;std::vector<CvelCase> input(count);std::vector<CvelAnswer> expected(count),actual(count);
    for(int index=0;index<count;index++) {
        auto& in=input[index];in.limb=index%6;in.target=(index/6)%9;const int motion=(index/54)%4,scenario=index/216;
        for(int side=0;side<2;side++) {
            in.moving[side]=(motion>>side)&1;
            in.yaw[side]=float(scenario-2)*.63f*(side?-1:1);in.vx[side]=float(scenario-2)*.17f;in.vy[side]=float(side?-.23:.37);in.omega[side]=float(scenario-2)*.41f;
            in.frame.root_com[side][0]=side?-.13f:.21f;in.frame.root_com[side][1]=side?.27f:-.19f;in.frame.root_com[side][2]=.67f;
            for(int slot=0;slot<14;slot++)for(int k=0;k<3;k++)in.frame.linear[side][slot][k]=float((slot*11+k*7+side*3)%19-9)*.63f;
        }
        expected[index].a=rek_contact_velocity::compose(in.frame,0,rek_contact_velocity::limb_slot(in.limb),in.yaw[0],in.vx[0],in.vy[0],in.omega[0],in.moving[0]);
        expected[index].b=rek_contact_velocity::compose(in.frame,1,rek_contact_velocity::target_slot(in.target),in.yaw[1],in.vx[1],in.vy[1],in.omega[1],in.moving[1]);
        expected[index].speed=rek_contact_velocity::relative_speed(expected[index].a,expected[index].b);
    }
    rek5::DeviceStorage storage;auto* cases=storage.upload(input);auto* result=storage.alloc<CvelAnswer>(count);
    compose_cases<<<(count+127)/128,128>>>(cases,result,count);rek5::cuda_check(cudaGetLastError());
    rek5::cuda_check(cudaMemcpy(actual.data(),result,count*sizeof(CvelAnswer),cudaMemcpyDeviceToHost));
    float max_component=0,max_speed=0;
    for(int i=0;i<count;i++) {
        const float e[]={expected[i].a.x,expected[i].a.y,expected[i].a.z,expected[i].b.x,expected[i].b.y,expected[i].b.z};
        const float a[]={actual[i].a.x,actual[i].a.y,actual[i].a.z,actual[i].b.x,actual[i].b.y,actual[i].b.z};
        for(int k=0;k<6;k++){if(!std::isfinite(a[k]))throw std::runtime_error("nonfinite CUDA cvel");max_component=std::max(max_component,std::abs(e[k]-a[k]));}
        if(!std::isfinite(actual[i].speed))throw std::runtime_error("nonfinite CUDA relative speed");
        max_speed=std::max(max_speed,std::abs(expected[i].speed-actual[i].speed));
    }
    std::printf("{\"test\":\"contact_velocity_cuda_composition\",\"cases\":%d,\"max_component_error_m_s\":%.9g,\"max_speed_error_m_s\":%.9g,\"bound_m_s\":1e-5,\"passed\":%s}\n",count,max_component,max_speed,(max_component<1e-5f&&max_speed<1e-5f)?"true":"false");
    return max_component<1e-5f&&max_speed<1e-5f?0:1;
}catch(const std::exception& error){std::fprintf(stderr,"%s\n",error.what());return 1;}}
