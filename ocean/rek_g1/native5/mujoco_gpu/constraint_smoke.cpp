#include "constraint_schedule.h"
#include "kernel_program.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
void check(CUresult result,const char* where){if(result!=CUDA_SUCCESS)throw std::runtime_error(where);}
template<class T>std::vector<T> read(const rek_mjgpu::ModelData& data,const char* name){
    std::vector<T> values(data.byte_size(name)/sizeof(T));
    if(!values.empty())check(cuMemcpyDtoH(values.data(),reinterpret_cast<CUdeviceptr>(data.array(name).data),data.byte_size(name)),name);
    return values;
}
template<class T>void upload(rek_mjgpu::ModelData& data,const char* name,const std::vector<T>& values){data.upload(name,values.data(),values.size()*sizeof(T));}
void finite_rows(const rek_mjgpu::ModelData& data){
    const auto rows=read<int>(data,"d.nefc");
    for(const char* field:{"d.efc.pos","d.efc.margin","d.efc.D","d.efc.vel","d.efc.aref"}){
        const auto values=read<float>(data,field);const auto a=data.array(field);
        for(int w=0;w<data.integer("d.nworld");w++)for(int j=0;j<rows[w];j++)
            if(!std::isfinite(values[(w*a.strides[0]+j*a.strides[1])/4]))throw std::runtime_error("Nonfinite active constraint row");
    }
}
}
int main(int argc,char** argv){try{
    if(argc!=3)throw std::runtime_error("Usage: constraint-smoke MODEL.xml CATALOG.json");
    check(cuInit(0),"cuInit");CUdevice dev;CUcontext context;CUstream stream;
    check(cuDeviceGet(&dev,0),"cuDeviceGet");check(cuDevicePrimaryCtxRetain(&context,dev),"context retain");
    check(cuCtxSetCurrent(context),"set context");check(cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING),"stream create");
    char error[1024]={};std::unique_ptr<mjModel,decltype(&mj_deleteModel)> model(mj_loadXML(argv[1],nullptr,error,sizeof(error)),mj_deleteModel);
    if(!model)throw std::runtime_error(error);model->opt.timestep=.002;
    {rek_mjgpu::ModelDataConfig config;config.nworld=4;rek_mjgpu::ModelData data(model.get(),config);
    auto spec=rek_mjgpu::build_kinematics_com(data);rek_mjgpu::append_constraints(spec,data);
    rek_mjgpu::KernelProgram program(data,argv[2],spec);
    auto run=[&](){program.launch(stream);check(cuStreamSynchronize(stream),"constraint launch");finite_rows(data);auto capacity=rek_mjgpu::inspect_constraint_capacity(data,stream);if(capacity.overflow)throw std::runtime_error("Constraint capacity overflow");return read<int>(data,"d.nefc");};
    const auto baseline=run();int hinge=-1;
    for(int j=0;j<model->njnt;j++)if(model->jnt_type[j]==mjJNT_HINGE&&model->jnt_limited[j]){
        const float q=model->qpos0[model->jnt_qposadr[j]];
        if(q>model->jnt_range[2*j]+model->jnt_margin[j]&&q<model->jnt_range[2*j+1]-model->jnt_margin[j]){hinge=j;break;}}
    if(hinge<0)throw std::runtime_error("No initially inactive hinge limit for fixture");
    std::vector<float> qpos(model->qpos0,model->qpos0+model->nq);
    qpos[model->jnt_qposadr[hinge]]=float(model->jnt_range[2*hinge+1]+model->jnt_margin[hinge]+.1);
    data.set_initial_state(qpos.data());const auto limited=run();
    for(int w=0;w<config.nworld;w++)if(limited[w]!=baseline[w]+1)throw std::runtime_error("Hinge fixture did not add exactly one constraint row");
    int fixed=-1,moving=-1;
    for(int g=0;g<model->ngeom;g++){if(model->geom_bodyid[g]==0&&fixed<0)fixed=g;if(model->body_dofnum[model->geom_bodyid[g]]>0&&moving<0)moving=g;}
    if(fixed<0||moving<0)throw std::runtime_error("Missing fixture geometry");
    const int cap=data.integer("d.naconmax");std::vector<int> worlds(cap),dim(cap),geom(2*cap),type(cap),addresses(cap*data.integer("m.nmaxpyramid"),-1);
    std::vector<float> dist(cap),pos(cap*3),frame(cap*9),friction(cap*5),solref(cap*2),solimp(cap*5);
    for(int w=0;w<config.nworld;w++){
        worlds[w]=w;dim[w]=3;geom[2*w]=fixed;geom[2*w+1]=moving;type[w]=1;dist[w]=-.01f;
        frame[9*w+2]=1;frame[9*w+3]=1;frame[9*w+7]=1;
        for(int k=0;k<2;k++)solref[2*w+k]=float(model->geom_solref[2*fixed+k]);
        for(int k=0;k<5;k++)solimp[5*w+k]=float(model->geom_solimp[5*fixed+k]);
        friction[5*w]=friction[5*w+1]=float(model->geom_friction[3*fixed]);
        friction[5*w+2]=float(model->geom_friction[3*fixed+1]);friction[5*w+3]=friction[5*w+4]=float(model->geom_friction[3*fixed+2]);
    }
    upload(data,"d.contact.worldid",worlds);upload(data,"d.contact.dim",dim);upload(data,"d.contact.geom",geom);upload(data,"d.contact.type",type);
    upload(data,"d.contact.efc_address",addresses);upload(data,"d.contact.dist",dist);upload(data,"d.contact.pos",pos);upload(data,"d.contact.frame",frame);
    upload(data,"d.contact.friction",friction);upload(data,"d.contact.solref",solref);upload(data,"d.contact.solimp",solimp);upload(data,"d.nacon",std::vector<int>{config.nworld});
    const auto contacted=run();for(int w=0;w<config.nworld;w++)if(contacted[w]!=limited[w]+3)throw std::runtime_error("Synthetic elliptic contact did not add exactly three rows");
    // Check alias layout without allocating a second device buffer.
    data.allocate("test.alias_source",{4,80});const auto alias=data.reshape_alias("test.alias", "test.alias_source",{4,80,1});
    if(alias.data!=data.array("test.alias_source").data||alias.shape[2]!=1||data.byte_size("test.alias")!=1280)throw std::runtime_error("Alias storage mismatch");
    const auto report=rek_mjgpu::inspect_constraint_capacity(data,stream);
    std::printf("{\"status\":\"passed\",\"test\":\"native_gpu_constraint_unit_fixtures\",\"worlds\":4,\"nodes\":%zu,\"baseline_rows\":%d,\"hinge_rows\":%d,\"synthetic_contact_rows\":%d,\"max_rows\":%d,\"max_nonzeros\":%d,\"overflow\":false,\"zero_copy_alias_verified\":true,\"native_mujoco_cuda_kernels\":true,\"collision_detection_tested\":false,\"full_physics_step\":false,\"cpu_physics_steps\":0,\"python_invocations\":0}\n",program.nodes(),baseline[0],limited[0],contacted[0],report.max_rows,report.max_nonzeros);
    }check(cuStreamDestroy(stream),"destroy stream");check(cuDevicePrimaryCtxRelease(dev),"release context");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"constraint-smoke: %s\n",e.what());return 1;}}
