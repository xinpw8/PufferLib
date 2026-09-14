// Included inside rek5's private implementation namespace after CUDA helpers.
// Original MuJoCo-Warp kernels and CUDA conditional graphs execute physics.
// These small kernels only export native state, validate it, and clear selected
// reset histories. They do not implement replacement dynamics.
struct NativeMujocoGpuAux {
    int* nefc=nullptr;int* ne=nullptr;int* nf=nullptr;int* nl=nullptr;int* nnz=nullptr;
    int* ncollision=nullptr;int* nccd=nullptr;int* global_bad=nullptr;
    int* rownnz=nullptr;int* rowadr=nullptr;int* state=nullptr;int* solver_niter=nullptr;
    float* qacc=nullptr;float* warmstart=nullptr;float* smooth_acc=nullptr;
    float* constraint_force=nullptr;float* force=nullptr;float* Ma=nullptr;
    int njmax=0,njmax_pad=0,njmax_nnz=0,naccdmax=0;
};
struct NativeMujocoGpu {
    std::unique_ptr<rek_mjgpu::ModelData> data;
    std::unique_ptr<rek_mjgpu::NativeStep> step;
    std::unique_ptr<rek_mjgpu::DeviceIf> reset_if;
    NativeMujocoGpuAux aux;
};

__global__ void mujoco_gpu_contact_counts(PhysicsDescriptor d,int* bad){
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=d.capacity||id>=*d.nacon)return;
    const int world=d.contact_world[id];
    if(world<0||world>=d.arenas){atomicOr(bad,8);return;}
    atomicAdd(d.counts+world,1);
}
__global__ void mujoco_gpu_export_stats(PhysicsDescriptor d,NativeMujocoGpuAux aux,int* stats){
    const int a=blockIdx.x,lane=threadIdx.x;if(a>=d.arenas)return;
    if(lane==0){
        for(int s=0;s<2;s++){
            for(int k=0;k<4;k++)d.base[a*8+s*4+k]=d.qpos[a*72+s*36+3+k];
            for(int k=0;k<3;k++)d.angular[a*6+s*3+k]=d.qvel[a*70+s*35+3+k];
        }
        // Native MuJoCo contacts are globally pooled and may interleave worlds.
        // A contiguous per-world offset does not exist; measurement uses worldid.
        d.offsets[a]=-1;atomicMax(stats+a*4,d.counts[a]);
        int failures=*aux.global_bad;
        if(*d.nacon<0||*d.nacon>d.capacity||*aux.ncollision<0||*aux.ncollision>d.capacity)failures|=1;
        if(aux.nefc[a]<0||aux.nefc[a]>aux.njmax||aux.nnz[a]<0||aux.nnz[a]>aux.njmax_nnz)failures|=2;
        for(int k=0;k<55;k++)if(aux.nccd[k]<0||aux.nccd[k]>aux.naccdmax)failures|=4;
        if(failures)atomicOr(stats+a*4+2,failures);
    }
    const float* fields[]={d.qpos+a*72,d.qvel+a*70,d.time+a,d.xpos+a*d.bodies*3,d.xquat+a*d.bodies*4,
        d.xmat+a*d.bodies*9,d.xipos+a*d.bodies*3,d.ximat+a*d.bodies*9,d.com+a*d.bodies*3,d.cvel+a*d.bodies*6,
        d.geom_xpos+a*d.geoms*3,d.geom_xmat+a*d.geoms*9,aux.qacc+a*70,aux.warmstart+a*70,aux.smooth_acc+a*70};
    const int widths[]={72,70,1,d.bodies*3,d.bodies*4,d.bodies*9,d.bodies*3,d.bodies*9,d.bodies*3,d.bodies*6,d.geoms*3,d.geoms*9,70,70,70};
    bool finite=true;for(int f=0;f<15;f++)for(int k=lane;k<widths[f];k+=blockDim.x)finite=finite&&isfinite(fields[f][k]);
    if(!finite)atomicOr(stats+a*4+1,1);
}
__global__ void mujoco_gpu_clear_selected(NativeMujocoGpuAux aux,const uint8_t* mask,int arenas){
    const int a=blockIdx.x,lane=threadIdx.x;if(a>=arenas||!mask[a])return;
    for(int k=lane;k<70;k+=blockDim.x){
        const int i=a*70+k;aux.qacc[i]=0;aux.warmstart[i]=0;aux.smooth_acc[i]=0;aux.constraint_force[i]=0;aux.Ma[i]=0;
    }
    for(int k=lane;k<aux.njmax;k+=blockDim.x){
        const int i=a*aux.njmax+k;aux.force[i]=0;aux.rownnz[i]=0;aux.rowadr[i]=0;
    }
    for(int k=lane;k<aux.njmax_pad;k+=blockDim.x)aux.state[a*aux.njmax_pad+k]=0;
    if(lane==0){aux.nefc[a]=aux.ne[a]=aux.nf[a]=aux.nl[a]=aux.nnz[a]=aux.solver_niter[a]=0;}
}
void mujoco_gpu_collect(Physics* p){
    auto* gpu=static_cast<NativeMujocoGpu*>(p->mujoco_gpu);auto& d=p->data;
    cuda_check(cudaMemsetAsync(d.counts,0,d.arenas*sizeof(int),p->stream),"clear native MuJoCo contact histogram");
    cuda_check(cudaMemsetAsync(gpu->aux.global_bad,0,sizeof(int),p->stream),"clear native MuJoCo contact diagnostic");
    mujoco_gpu_contact_counts<<<(d.capacity+255)/256,256,0,p->stream>>>(d,gpu->aux.global_bad);
    mujoco_gpu_export_stats<<<d.arenas,128,0,p->stream>>>(d,gpu->aux,p->stats);
    cuda_check(cudaGetLastError(),"native MuJoCo state export and validation");
}
void mujoco_gpu_refresh(Physics* p){
    auto* gpu=static_cast<NativeMujocoGpu*>(p->mujoco_gpu);
    gpu->step->refresh(reinterpret_cast<CUstream>(p->stream));mujoco_gpu_collect(p);
}
void mujoco_gpu_step(Physics* p,const float* controls){
    require(controls!=nullptr,"null native MuJoCo controls");
    if(controls!=p->ctrl)cuda_check(cudaMemcpyAsync(p->ctrl,controls,size_t(p->data.arenas)*58*sizeof(float),cudaMemcpyDeviceToDevice,p->stream),"copy native MuJoCo controls on device");
    auto* gpu=static_cast<NativeMujocoGpu*>(p->mujoco_gpu);
    gpu->step->step(reinterpret_cast<CUstream>(p->stream));mujoco_gpu_refresh(p);
}
void mujoco_gpu_forward_selected(Physics* p,const uint8_t* mask){
    require(mask!=nullptr,"null native MuJoCo reset mask");auto* gpu=static_cast<NativeMujocoGpu*>(p->mujoco_gpu);
    gpu->reset_if->execute(reinterpret_cast<CUstream>(p->stream),mask,p->data.arenas,[=](CUstream target){
        const cudaStream_t previous=p->stream;p->stream=reinterpret_cast<cudaStream_t>(target);
        try{
            mujoco_gpu_clear_selected<<<p->data.arenas,128,0,p->stream>>>(gpu->aux,mask,p->data.arenas);
            cuda_check(cudaGetLastError(),"clear selected native MuJoCo warmstarts");
            // Reset invalidates old contact locations as well as warmstarts.
            // The GPU mask-any condition skips this work for empty reset masks.
            gpu->step->refresh_contacts(target);mujoco_gpu_collect(p);
        }catch(...){p->stream=previous;throw;}
        p->stream=previous;
    });
}
void mujoco_gpu_create(Physics* p){
    auto required=[](const char* name){const char* value=std::getenv(name);require(value&&*value,std::string("missing ")+name);return std::string(value);};
    const std::string catalog=required("REK_MUJOCO_KERNEL_CATALOG"),ptx=required("REK_MUJOCO_CONDITIONAL_PTX"),hash=required("REK_MUJOCO_CONDITIONAL_SHA256");
    auto gpu=std::make_unique<NativeMujocoGpu>();rek_mjgpu::ModelDataConfig config;
    config.nworld=p->data.arenas;config.nconmax=512;config.njmax=512;
    gpu->data=std::make_unique<rek_mjgpu::ModelData>(p->model,config);gpu->data->set_initial_state(p->initial_qpos.data());
    gpu->step=std::make_unique<rek_mjgpu::NativeStep>(*gpu->data,catalog,ptx,hash);
    gpu->reset_if=std::make_unique<rek_mjgpu::DeviceIf>(ptx,hash);
    auto floats=[&](const char* name){return static_cast<float*>(gpu->data->array(name).data);};
    auto integers=[&](const char* name){return static_cast<int*>(gpu->data->array(name).data);};
    auto& d=p->data;
    d.qpos=floats("d.qpos");d.qvel=floats("d.qvel");d.time=floats("d.time");p->ctrl=floats("d.ctrl");
    d.xpos=floats("d.xpos");d.xquat=floats("d.xquat");d.xmat=floats("d.xmat");d.xipos=floats("d.xipos");d.ximat=floats("d.ximat");d.com=floats("d.subtree_com");d.cvel=floats("d.cvel");
    d.geom_xpos=floats("d.geom_xpos");d.geom_xmat=floats("d.geom_xmat");
    d.contact_geom=integers("d.contact.geom");d.contact_world=integers("d.contact.worldid");d.nacon=integers("d.nacon");
    d.contact_dist=floats("d.contact.dist");d.contact_pos=floats("d.contact.pos");d.contact_frame=floats("d.contact.frame");
    d.body_map=nullptr;d.geom_map=nullptr;d.pre_centers=nullptr;
    auto& a=gpu->aux;a.nefc=integers("d.nefc");a.ne=integers("d.ne");a.nf=integers("d.nf");a.nl=integers("d.nl");a.nnz=integers("constraint.efc_nnz");
    a.ncollision=integers("d.ncollision");a.nccd=integers("collision.nccd");
    a.rownnz=integers("d.efc.J_rownnz");a.rowadr=integers("d.efc.J_rowadr");a.state=integers("d.efc.state");a.solver_niter=integers("d.solver_niter");
    a.qacc=floats("d.qacc");a.warmstart=floats("d.qacc_warmstart");a.smooth_acc=floats("d.qacc_smooth");a.constraint_force=floats("d.qfrc_constraint");a.force=floats("d.efc.force");a.Ma=floats("d.efc.Ma");
    a.njmax=config.njmax;a.njmax_pad=gpu->data->integer("d.njmax_pad");a.njmax_nnz=gpu->data->integer("d.njmax_nnz");a.naccdmax=gpu->data->integer("d.naccdmax");
    allocate(*p,a.global_bad,1);allocate(*p,p->stats,size_t(d.arenas)*4);
    p->mujoco_gpu=gpu.release();mujoco_gpu_refresh(p);cuda_check(cudaStreamSynchronize(p->stream),"complete native MuJoCo CUDA startup");
    std::fprintf(stderr,"physics_backend=mujoco_cuda cpu_physics=0 python_runtime=0 arenas=%d contact_pool=%d constraint_rows_per_arena=%d native_phase_nodes=%zu\n",
        d.arenas,d.capacity,config.njmax,static_cast<NativeMujocoGpu*>(p->mujoco_gpu)->step->phase_nodes());
}
