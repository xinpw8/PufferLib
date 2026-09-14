#include "model_data.h"
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>
int main(int argc,char** argv){try{
    if(argc!=2)throw std::runtime_error("Usage: model-data-probe MODEL.xml");
    char error[1024]={};std::unique_ptr<mjModel,decltype(&mj_deleteModel)> model(mj_loadXML(argv[1],nullptr,error,sizeof(error)),mj_deleteModel);
    if(!model)throw std::runtime_error(error);model->opt.timestep=.002;
    CUdevice dev;CUcontext context;if(cuInit(0)!=CUDA_SUCCESS||cuDeviceGet(&dev,0)!=CUDA_SUCCESS||cuDevicePrimaryCtxRetain(&context,dev)!=CUDA_SUCCESS||cuCtxSetCurrent(context)!=CUDA_SUCCESS)throw std::runtime_error("CUDA initialization");
    {rek_mjgpu::ModelDataConfig config;rek_mjgpu::ModelData data(model.get(),config);std::size_t bytes=0;for(const auto& field:data.arrays())bytes+=field.second.bytes;
    std::vector<float> qpos(model->nq);if(cuMemcpyDtoH(qpos.data(),reinterpret_cast<CUdeviceptr>(data.array("d.qpos").data),qpos.size()*4)!=CUDA_SUCCESS)throw std::runtime_error("qpos download");
    for(int i=0;i<model->nq;i++)if(qpos[i]!=float(model->qpos0[i]))throw std::runtime_error("qpos0 packing mismatch");
    std::printf("{\"status\":\"passed\",\"array_fields\":%zu,\"device_bytes\":%zu,\"nq\":%d,\"nv\":%d,\"nu\":%d,\"nbody\":%d,\"ngeom\":%d,\"njnt\":%d,\"ntree\":%d,\"nM\":%d,\"nC\":%d,\"nsite\":%d,\"nsensor\":%d,\"body_tree_levels\":%d,\"qLD_levels\":%d,\"filtered_pairs\":%d,\"integrator\":%d,\"solver\":%d,\"cone\":%d,\"jacobian\":%d,\"iterations\":%d,\"ls_iterations\":%d,\"ccd_iterations\":%d,\"disableflags\":%d,\"enableflags\":%d,\"cpu_physics_steps\":0,\"gpu_physics_steps\":0,\"python_invocations\":0}\n",
    data.arrays().size(),bytes,int(model->nq),int(model->nv),int(model->nu),int(model->nbody),int(model->ngeom),int(model->njnt),int(model->ntree),int(model->nM),int(model->nC),int(model->nsite),int(model->nsensor),data.integer("m.body_tree.count"),data.integer("m.qLD_updates.count"),data.array("m.nxn_geom_pair_filtered").shape[0],model->opt.integrator,model->opt.solver,model->opt.cone,model->opt.jacobian,model->opt.iterations,model->opt.ls_iterations,model->opt.ccd_iterations,model->opt.disableflags,model->opt.enableflags);
    std::printf("{\"geom_pair_type_counts\":[");bool comma=false;for(int count:data.geometry_pair_type_counts()){std::printf("%s%d",comma?",":"",count);comma=true;}std::printf("]}\n");
    }cuDevicePrimaryCtxRelease(dev);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"model-data-probe: %s\n",e.what());return 1;}}
