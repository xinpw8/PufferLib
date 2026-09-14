// Included inside physics.cu's private namespace. Viewer-only host execution
// of the same Puffysics equations, with persistent pinned transfer buffers.
struct CpuPuffysicsEvaluation {
    struct Transfer {void* host;void* device;size_t bytes;};
    rek5_native::RpHandle handle{};
    rek5_native::RpsDescriptor data{};
    rek5_native::B3World initial;
    std::vector<rek5_native::B3World> worlds;
    std::vector<Transfer> transfers;
    std::vector<float> controls;
    std::vector<uint8_t> mask;
    ~CpuPuffysicsEvaluation(){for(const auto& v:transfers)cudaFreeHost(v.host);}
    template<typename T> void mirror(T*& field,T* device,size_t count){
        size_t bytes=count*sizeof(T);cuda_check(cudaMallocHost(reinterpret_cast<void**>(&field),bytes),"allocate CPU Puffysics pinned mirror");
        transfers.push_back({field,device,bytes});
        cuda_check(cudaMemcpy(field,device,bytes,cudaMemcpyDeviceToHost),"initialize CPU Puffysics mirror");
    }
};

void cpu_puffysics_create(Physics* p){
    using namespace rek5_native;
    auto c=std::make_unique<CpuPuffysicsEvaluation>();auto& d=c->data;const auto& g=p->data;
    c->handle=*static_cast<RpHandle*>(p->native_handle);
    cuda_check(cudaMemcpy(&c->initial,c->handle.initial,sizeof(B3World),cudaMemcpyDeviceToHost),"copy original Puffysics world to viewer");
    c->worlds.assign(g.arenas,c->initial);c->handle.worlds=c->worlds.data();c->handle.initial=&c->initial;
    p->evaluation_stats.resize(size_t(g.arenas)*4);c->handle.stats=p->evaluation_stats.data();
    c->controls.resize(size_t(g.arenas)*58);c->mask.resize(g.arenas);
    d=native_descriptor(g);size_t a=g.arenas,b=g.bodies,n=g.geoms,k=g.capacity;
    c->mirror(d.qpos,g.qpos,a*72);c->mirror(d.qvel,g.qvel,a*70);
    c->mirror(d.base,g.base,a*8);c->mirror(d.angular,g.angular,a*6);c->mirror(d.time,g.time,a);
    c->mirror(d.xpos,g.xpos,a*b*3);c->mirror(d.xquat,g.xquat,a*b*4);c->mirror(d.xmat,g.xmat,a*b*9);
    c->mirror(d.xipos,g.xipos,a*b*3);c->mirror(d.ximat,g.ximat,a*b*9);c->mirror(d.com,g.com,a*b*3);
    c->mirror(d.cvel,g.cvel,a*b*6);c->mirror(d.geom_xpos,g.geom_xpos,a*n*3);c->mirror(d.geom_xmat,g.geom_xmat,a*n*9);
    c->mirror(d.contact_geom,g.contact_geom,k*2);c->mirror(d.contact_world,g.contact_world,k);c->mirror(d.nacon,g.nacon,1);
    c->mirror(d.contact_dist,g.contact_dist,k);c->mirror(d.contact_pos,g.contact_pos,k*3);c->mirror(d.contact_frame,g.contact_frame,k*9);
    c->mirror(d.counts,g.counts,a);c->mirror(d.offsets,g.offsets,a);c->mirror(d.pre_centers,g.pre_centers,a*61*3);
    d.body_map=p->host_body_map.data();d.geom_map=p->host_geom_map.data();
    p->cpu_evaluation=true;p->puffysics_evaluation=c.release();
    std::fprintf(stderr,"physics_backend=puffysics_cpu_eval cpu_physics=1 training_backend=0 arenas=%d\n",g.arenas);
}

void cpu_puffysics_export(Physics* p){
    using namespace rek5_native;
    auto* c=static_cast<CpuPuffysicsEvaluation*>(p->puffysics_evaluation);auto& d=c->data;
    int total=0;
    for(int a=0;a<d.arenas;a++){
        rps_body_fields(c->handle.worlds+a,a,d);d.offsets[a]=total;total+=d.counts[a];
    }
    *d.nacon=total;
    for(int a=0;a<d.arenas;a++)rps_contacts_one(c->handle,d,a);
    for(const auto& t:c->transfers)cuda_check(cudaMemcpyAsync(t.device,t.host,t.bytes,cudaMemcpyHostToDevice,p->stream),"upload CPU Puffysics viewer state");
    cuda_check(cudaMemcpyAsync(p->stats,p->evaluation_stats.data(),p->evaluation_stats.size()*sizeof(int),cudaMemcpyHostToDevice,p->stream),"upload CPU Puffysics status");
}

void cpu_puffysics_step(Physics* p,const float* device_ctrl){
    using namespace rek5_native;
    evaluation_boundary(p);auto* c=static_cast<CpuPuffysicsEvaluation*>(p->puffysics_evaluation);auto& d=c->data;
    cuda_check(cudaMemcpy(c->controls.data(),device_ctrl,c->controls.size()*sizeof(float),cudaMemcpyDeviceToHost),"download CPU Puffysics controls");
    for(float value:c->controls)require(std::isfinite(value),"nonfinite CPU Puffysics control");
    for(int a=0;a<d.arenas;a++){
        B3World* w=c->handle.worlds+a;int* stats=c->handle.stats+a*4;
        if(stats[1]||stats[2]){
            std::fprintf(stderr,"puffysics_failure arena=%d time=%.9g mode=%d max_contacts=%d nonfinite=%d solver_status=%d limit_impulses=%d\n",
                a,d.time[a],c->handle.mode,stats[0],stats[1],stats[2],stats[3]);
            std::fprintf(stderr,"collision_meta");
            for(int k=0;k<7;k++)std::fprintf(stderr," %d",w->collision_failure_meta[k]);
            std::fprintf(stderr,"\nqpos");
            for(int k=0;k<72;k++)std::fprintf(stderr," %.9g",d.qpos[a*72+k]);
            std::fprintf(stderr,"\nqvel");
            for(int k=0;k<70;k++)std::fprintf(stderr," %.9g",d.qvel[a*70+k]);
            std::fprintf(stderr,"\n");
            require(false,"previous CPU Puffysics failure arena="+std::to_string(a)+
                " time="+std::to_string(d.time[a])+" nonfinite="+std::to_string(stats[1])+
                " solver_status="+std::to_string(stats[2]));
        }
        for(int b=0;b<61;b++)rps_store(d.pre_centers+(a*61+b)*3,w->bodies[b].center);
        if(c->handle.stabilization&1)rp_cold_start_joints(w);
        rp_forces(w,c->controls.data()+a*58);
#if B3_ART_CONTACTS
        if(c->handle.mode==1)require(rp_art_step(w,stats),"articulated CPU Puffysics solver failure");
        else
#endif
        b3_step(w,.002f,1);
        stats[2]|=w->collision_status;d.time[a]+=.002f;
        rp_gather(w,a,d.qpos,d.qvel,d.base,d.angular,c->handle.stats);
    }
    cpu_puffysics_export(p);
}

void cpu_puffysics_forward(Physics* p,const uint8_t* mask){
    using namespace rek5_native;
    evaluation_boundary(p);auto* c=static_cast<CpuPuffysicsEvaluation*>(p->puffysics_evaluation);auto& d=c->data;
    cuda_check(cudaMemcpy(c->mask.data(),mask,c->mask.size(),cudaMemcpyDeviceToHost),"download CPU Puffysics reset mask");
    if(std::none_of(c->mask.begin(),c->mask.end(),[](uint8_t value){return value!=0;}))return;
    cuda_check(cudaMemcpy(d.qpos,p->data.qpos,d.arenas*72*sizeof(float),cudaMemcpyDeviceToHost),"download CPU Puffysics reset qpos");
    cuda_check(cudaMemcpy(d.qvel,p->data.qvel,d.arenas*70*sizeof(float),cudaMemcpyDeviceToHost),"download CPU Puffysics reset qvel");
    cuda_check(cudaMemcpy(d.time,p->data.time,d.arenas*sizeof(float),cudaMemcpyDeviceToHost),"download CPU Puffysics reset clock");
    for(int a=0;a<d.arenas;a++)if(c->mask[a])rps_forward_one(c->handle,d,a);
    cpu_puffysics_export(p);
}
