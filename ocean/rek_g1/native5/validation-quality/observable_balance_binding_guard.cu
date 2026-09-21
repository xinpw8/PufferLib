// Reporting-boundary test instrumentation. Never runs inside captured rollout.
#include "runtime_api.h"
#include "observable_balance.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
namespace {
RekNative5Buffers observed_buffers{};
RekNative5Runtime* observed_runtime=nullptr;
int observed_arenas=0,checks=0;
void require_guard(bool value){if(!value){std::fputs("observable learner binding check failed\n",stderr);std::abort();}}
}
extern "C" RekNative5Runtime* __real_rek_native5_create(const RekNative5Config*,const RekNative5Buffers*,cudaStream_t);
extern "C" int __real_rek_native5_check_status(RekNative5Runtime*,cudaStream_t);
extern "C" RekNative5Runtime* __wrap_rek_native5_create(const RekNative5Config* config,const RekNative5Buffers* buffers,cudaStream_t stream){
    auto* runtime=__real_rek_native5_create(config,buffers,stream);
    if(runtime){observed_buffers=*buffers;observed_runtime=runtime;observed_arenas=config->arenas;}
    return runtime;
}
extern "C" int __wrap_rek_native5_check_status(RekNative5Runtime* runtime,cudaStream_t stream){
    const int status=__real_rek_native5_check_status(runtime,stream);if(status)return status;
    require_guard(runtime==observed_runtime&&observed_arenas>0);
    cudaStreamCaptureStatus capture;require_guard(cudaStreamIsCapturing(stream,&capture)==cudaSuccess&&capture==cudaStreamCaptureStatusNone);
    float* encoded=nullptr;require_guard(cudaMalloc(&encoded,size_t(observed_arenas)*446*sizeof(float))==cudaSuccess);
    require_guard(rek_native5_encode_fighter_observations(runtime,encoded,stream)==0);
    std::vector<float> learner(size_t(observed_arenas)*223),both(size_t(observed_arenas)*446);
    require_guard(cudaMemcpyAsync(learner.data(),observed_buffers.observations,learner.size()*sizeof(float),cudaMemcpyDeviceToHost,stream)==cudaSuccess);
    require_guard(cudaMemcpyAsync(both.data(),encoded,both.size()*sizeof(float),cudaMemcpyDeviceToHost,stream)==cudaSuccess);
    require_guard(cudaStreamSynchronize(stream)==cudaSuccess);
    for(int arena=0;arena<observed_arenas;arena++){
        const float* row=learner.data()+arena*223;
        require_guard(std::memcmp(row,both.data()+arena*446,223*sizeof(float))==0);
        require_guard(row[202]==1&&row[74]==0&&row[160]==0&&row[75]==0&&row[161]==0);
        for(int k=0;k<223;k++){
            require_guard(std::isfinite(row[k]));
            if(!rek_observable_balance::structurally_available(k))require_guard(row[k]==0);
        }
    }
    require_guard(cudaFree(encoded)==cudaSuccess);
    std::printf("observable_balance_binding_check={\"check\":%d,\"arenas\":%d,\"learner_equals_projected_export\":true,\"excluded_columns_zero\":true,\"joint_pose_available\":false,\"finite\":true}\n",++checks,observed_arenas);
    return 0;
}
