#include "primitive_contacts.cuh"
#include "primitive_motion.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
using rek5_primitive::Shape;
struct Case {
    Shape old_a,a,old_b,b;
    float yaw,x,y;
    int pair;
};
struct Result { int base,reverse,world,interpolated,sampled,sampled_reverse; };

__host__ __device__ Result evaluate(const Case& c) {
    using namespace rek5_primitive;
    Result result{};
    result.base=overlap(c.a,c.b);
    result.reverse=overlap(c.b,c.a);
    result.world=overlap(world_shape(c.a,c.x,c.y,c.yaw),world_shape(c.b,c.x,c.y,c.yaw));
    result.interpolated=overlap(interpolate_shape(c.old_a,c.a,.375f),interpolate_shape(c.old_b,c.b,.375f));
    result.sampled=sampled_overlap(c.old_a,c.a,c.old_b,c.b,4);
    result.sampled_reverse=sampled_overlap(c.old_b,c.b,c.old_a,c.a,4);
    return result;
}

__global__ void evaluate_kernel(const Case* cases,Result* results,int count) {
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<count) results[index]=evaluate(cases[index]);
}

void checked(cudaError_t status,const char* operation) {
    if(status!=cudaSuccess) {
        std::fprintf(stderr,"%s: %s\n",operation,cudaGetErrorString(status));
        throw std::runtime_error(operation);
    }
}

Shape random_shape(int kind,std::mt19937& rng) {
    std::uniform_real_distribution<float> position(-2,2),size(.05f,.65f),angle(-3.1f,3.1f);
    Shape s{kind,{position(rng),position(rng),position(rng)},{},{size(rng),size(rng),size(rng)}};
    const float yaw=angle(rng),pitch=angle(rng),roll=angle(rng);
    const float c=std::cos(yaw),d=std::sin(yaw),e=std::cos(pitch),f=std::sin(pitch);
    const float g=std::cos(roll),h=std::sin(roll);
    const float rotation[9]={c*e,c*f*h-d*g,c*f*g+d*h,
                             d*e,d*f*h+c*g,d*f*g-c*h,-f,e*h,e*g};
    for(int i=0;i<9;i++) s.axes[i]=rotation[i];
    return s;
}

Shape resized(Shape s,float delta) {
    // Only a test selection filter. Production overlap has zero extra margin.
    // Increasing capsule radius alone encloses the unchanged capsule segment.
    const int dimensions=s.kind==rek5_primitive::Box ? 3 : 1;
    for(int i=0;i<dimensions;i++) s.size[i]+=delta;
    return s;
}
bool away_from_boundary(const Shape& a,const Shape& b) {
    constexpr float filter=.0002f;
    return rek5_primitive::overlap(resized(a,-filter),resized(b,-filter))
        ==rek5_primitive::overlap(resized(a,filter),resized(b,filter));
}
bool usable(const Case& c) {
    using namespace rek5_primitive;
    if(!away_from_boundary(c.a,c.b)) return false;
    if(!away_from_boundary(world_shape(c.a,c.x,c.y,c.yaw),world_shape(c.b,c.x,c.y,c.yaw))) return false;
    if(!away_from_boundary(interpolate_shape(c.old_a,c.a,.375f),interpolate_shape(c.old_b,c.b,.375f))) return false;
    for(int i=1;i<=4;i++) {
        const float t=i*.25f;
        if(!away_from_boundary(interpolate_shape(c.old_a,c.a,t),interpolate_shape(c.old_b,c.b,t))) return false;
    }
    return true;
}
bool same(const Result& a,const Result& b) {
    return a.base==b.base && a.reverse==b.reverse && a.world==b.world
        && a.interpolated==b.interpolated && a.sampled==b.sampled
        && a.sampled_reverse==b.sampled_reverse;
}
} // namespace

int main() {
    Case* device_cases=nullptr;
    Result* device_results=nullptr;
    try {
        constexpr int per_pair=2048,total=6*per_pair;
        const int pairs[6][2]={{0,0},{0,1},{0,2},{1,1},{1,2},{2,2}};
        std::mt19937 rng(987461);
        std::uniform_real_distribution<float> offset(-.3f,.3f),angle(-3.1f,3.1f);
        std::vector<Case> cases;
        std::vector<Result> host;
        cases.reserve(total);host.reserve(total);
        int excluded=0,positive[6]={};
        for(int pair=0;pair<6;pair++) {
            for(int i=0;i<per_pair;) {
                Case c{};c.pair=pair;
                c.a=random_shape(pairs[pair][0],rng);c.b=random_shape(pairs[pair][1],rng);
                if(i%2==0) for(int k=0;k<3;k++) c.b.center[k]=c.a.center[k]+offset(rng);
                c.old_a=random_shape(c.a.kind,rng);c.old_b=random_shape(c.b.kind,rng);
                for(int k=0;k<3;k++) {c.old_a.size[k]=c.a.size[k];c.old_b.size[k]=c.b.size[k];}
                c.yaw=angle(rng);c.x=1.25f;c.y=-2.5f;
                if(!usable(c)) {++excluded;continue;}
                const Result reference=evaluate(c);
                if(reference.base!=reference.reverse || reference.base!=reference.world || reference.sampled!=reference.sampled_reverse)
                    throw std::runtime_error("host invariance failed away from selected boundary");
                positive[pair]+=reference.base;
                cases.push_back(c);host.push_back(reference);++i;
            }
        }
        checked(cudaSetDevice(0),"cudaSetDevice");
        checked(cudaMalloc(reinterpret_cast<void**>(&device_cases),cases.size()*sizeof(Case)),"cudaMalloc cases");
        checked(cudaMalloc(reinterpret_cast<void**>(&device_results),host.size()*sizeof(Result)),"cudaMalloc results");
        checked(cudaMemcpy(device_cases,cases.data(),cases.size()*sizeof(Case),cudaMemcpyHostToDevice),"cudaMemcpy cases");
        evaluate_kernel<<<(total+63)/64,64>>>(device_cases,device_results,total);
        checked(cudaGetLastError(),"evaluate_kernel launch");
        checked(cudaDeviceSynchronize(),"evaluate_kernel synchronize");
        std::vector<Result> gpu(total);
        checked(cudaMemcpy(gpu.data(),device_results,gpu.size()*sizeof(Result),cudaMemcpyDeviceToHost),"cudaMemcpy results");
        checked(cudaFree(device_results),"cudaFree results");device_results=nullptr;
        checked(cudaFree(device_cases),"cudaFree cases");device_cases=nullptr;
        int mismatches=0,symmetry=0,invariance=0;
        for(int i=0;i<total;i++) {
            if(!same(host[i],gpu[i])) {
                if(mismatches<8) std::fprintf(stderr,"CPU/GPU mismatch case=%d pair=%d host=%d,%d,%d,%d,%d,%d gpu=%d,%d,%d,%d,%d,%d\n",
                    i,cases[i].pair,host[i].base,host[i].reverse,host[i].world,host[i].interpolated,host[i].sampled,host[i].sampled_reverse,
                    gpu[i].base,gpu[i].reverse,gpu[i].world,gpu[i].interpolated,gpu[i].sampled,gpu[i].sampled_reverse);
                ++mismatches;
            }
            symmetry+=gpu[i].base!=gpu[i].reverse || gpu[i].sampled!=gpu[i].sampled_reverse;
            invariance+=gpu[i].base!=gpu[i].world;
        }
        const bool passed=!mismatches&&!symmetry&&!invariance;
        std::printf("{\"event\":\"primitive_contacts_cuda\",\"passed\":%s,\"n\":%d,\"checks\":%d,\"cases_per_pair_type\":%d,\"pair_types\":[\"sphere_sphere\",\"sphere_capsule\",\"sphere_box\",\"capsule_capsule\",\"capsule_box\",\"box_box\"],\"positive_overlap_per_type\":[%d,%d,%d,%d,%d,%d],\"host_gpu_mismatches\":%d,\"symmetric_violations\":%d,\"transform_violations\":%d,\"excluded_boundary_cases\":%d,\"boundary_filter_m\":0.0002,\"production_contact_margin_m\":0,\"temporal_samples\":4,\"python_runtime\":false,\"physics_stepping\":false,\"dynamic_parity_claim\":false}\n",
            passed?"true":"false",total,total*6,per_pair,positive[0],positive[1],positive[2],positive[3],positive[4],positive[5],mismatches,symmetry,invariance,excluded);
        return passed?0:1;
    } catch(const std::exception& error) {
        if(device_results) cudaFree(device_results);
        if(device_cases) cudaFree(device_cases);
        std::fprintf(stderr,"primitive CUDA diagnostic failed: %s\n",error.what());
        std::puts("{\"event\":\"primitive_contacts_cuda\",\"passed\":false,\"execution_error\":true}");
        return 2;
    }
}
