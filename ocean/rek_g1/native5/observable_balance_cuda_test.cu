#include <cuda_runtime.h>
#define main observable_balance_cpu_regression_main
#include "observable_balance_test.cpp"
#undef main

// Small synthetic projection-only execution. No physics, game or policy.
__global__ void project_snapshots(const ob::Snapshot* now,const ob::Snapshot* previous,
        float* observations,int* statuses,int count){
    const int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<count)statuses[row]=int(ob::project(now[row],row%7==0?nullptr:previous+row,observations+row*223));
}
int main(){
    constexpr int count=64;
    auto cuda_check=[](cudaError_t status){if(status!=cudaSuccess){std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(status));std::exit(2);}};
    ob::Snapshot *current=nullptr,*previous=nullptr;float* gpu=nullptr;int* statuses=nullptr;
    cuda_check(cudaMallocManaged(&current,count*sizeof(ob::Snapshot)));
    cuda_check(cudaMallocManaged(&previous,count*sizeof(ob::Snapshot)));
    cuda_check(cudaMallocManaged(&gpu,count*223*sizeof(float)));
    cuda_check(cudaMallocManaged(&statuses,count*sizeof(int)));
    std::array<std::array<float,223>,count> cpu{};
    for(int i=0;i<count;i++){
        previous[i]=fixture();previous[i].actor_slot=i%2;current[i]=previous[i];
        current[i].sample_seconds+=i%5==0?.3:.02;
        if(i%11==0)current[i].round_key++;
        current[i].referee_available=i%3!=0;current[i].count_mask=unsigned(i%4);
        current[i].points[i%2]=i%6;
        for(int side=0;side<2;side++){
            for(int k=0;k<3;k++){
                previous[i].fighter[side].root_xyz[k]=random(-4,4);
                current[i].fighter[side].root_xyz[k]=previous[i].fighter[side].root_xyz[k]+random(-.04f,.04f);
            }
            rotation(previous[i].fighter[side],euler(random(-3,3),random(-1,1),random(-1,1)));
            rotation(current[i].fighter[side],euler(random(-3,3),random(-1,1),random(-1,1)));
            if(i%13==0)rotation(current[i].fighter[side],euler(0,ob::kPi/2));
            if(i%2)for(float&value:current[i].fighter[side].root_wxyz)value=-value;
            for(int j=0;j<29;j++){
                previous[i].fighter[side].projected_joints[j]=random(-3,3);
                current[i].fighter[side].projected_joints[j]=previous[i].fighter[side].projected_joints[j]+random(-.1f,.1f);
            }
            if(i%9==0){current[i].fighter[side].joint_pose_available=0;current[i].fighter[side].projected_joints[0]=NAN;}
        }
        check(ob::project(current[i],i%7==0?nullptr:previous+i,cpu[i].data())==ob::kOk,"CPU synthetic projection");
    }
    project_snapshots<<<1,64>>>(current,previous,gpu,statuses,count);
    cuda_check(cudaGetLastError());cuda_check(cudaDeviceSynchronize());
    double maximum=0;int compared=0;
    for(int i=0;i<count;i++){
        check(statuses[i]==ob::kOk,"CUDA projection status");
        for(int k=0;k<223;k++){
            const double difference=std::abs(double(cpu[i][k])-gpu[i*223+k]);maximum=std::max(maximum,difference);
            near(gpu[i*223+k],cpu[i][k],2e-5,"CPU/CUDA projection equivalence");compared++;
        }
    }
    cuda_check(cudaFree(current));cuda_check(cudaFree(previous));cuda_check(cudaFree(gpu));cuda_check(cudaFree(statuses));
    std::printf("PASS observable_balance_v1 CUDA snapshots=%d compared_features=%d max_cpu_cuda_difference=%.9g no_physics_no_training\n",count,compared,maximum);
}
