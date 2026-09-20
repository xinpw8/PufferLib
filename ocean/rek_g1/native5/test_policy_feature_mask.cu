// Synthetic CUDA publication tests. No claims about authentic dynamics.
#include "fast_runtime.cu"

__global__ void mask_fixture(Parameters* p,FastFrame* frame){
    if(threadIdx.x||blockIdx.x)return;
    *p={};p->round_seconds=.06f;p->opponent_mode=1;p->move_speed=1;
    p->yaw_speed=1;p->yaw_ramp=.5f;p->brake_rate=3;p->settle_speed=.01f;
    p->spawn[0][0]=-2;p->spawn[1][0]=2;
    p->half_extent[0]=p->half_extent[1]=10;
    for(int i=0;i<24;i++){p->routes[i].count=1;p->routes[i].loop=1;}
    for(int side=0;side<2;side++)for(int j=0;j<29;j++){
        p->qindices[side][j]=side*36+7+j;p->vindices[side][j]=side*35+6+j;
    }
    *frame={};frame->root_z=1;frame->root_wxyz[0]=1;
}
template<class T>T* mask_managed(size_t count,std::vector<void*>& memory){
    T* p=nullptr;rek5::cuda_check(cudaMallocManaged(&p,count*sizeof(T)));
    rek5::cuda_check(cudaMemset(p,0,count*sizeof(T)));memory.push_back(p);return p;
}
View mask_view(const Parameters* p,const FastFrame* frames,std::vector<void*>& memory){
    View v{};v.arenas=1;v.p=p;v.frames=frames;
    v.state=mask_managed<Arena>(1,memory);v.rounds=mask_managed<RekNative5RoundResult>(1,memory);
    v.raw=mask_managed<float>(446,memory);v.qpos=mask_managed<float>(72,memory);
    v.qvel=mask_managed<float>(70,memory);v.actions=mask_managed<float>(2,memory);
    v.rewards=mask_managed<float>(2,memory);v.terminals=mask_managed<float>(2,memory);
    v.masks=mask_managed<uint8_t>(66,memory);v.learner_masks=mask_managed<uint8_t>(33,memory);
    v.out.observations=mask_managed<float>(223,memory);v.out.actions=mask_managed<float>(1,memory);
    v.out.rewards=mask_managed<float>(1,memory);v.out.terminals=mask_managed<float>(1,memory);
    v.out.logs=mask_managed<RekNative5Log>(1,memory);v.out.log_stride_bytes=sizeof(RekNative5Log);
    return v;
}
int main(int argc,char** argv){try{
    if(argc>2)throw std::runtime_error("usage: test-policy-feature-mask [223-byte-mask]");
    const auto omitted=rek_policy_features::load(nullptr);
    if(omitted.enabled||!omitted.sha256.empty()||std::count(omitted.values.begin(),omitted.values.end(),1)!=223)
        throw std::runtime_error("omitted mask changed default");
    auto selected=argc==2?rek_policy_features::load(argv[1]):rek_policy_features::Mask{};
    if(argc==1)for(int i=176;i<=183;i++)selected.values[i]=0;
    std::vector<void*> memory;
    auto* p=mask_managed<Parameters>(1,memory);auto* frame=mask_managed<FastFrame>(1,memory);
    mask_fixture<<<1,1>>>(p,frame);rek5::cuda_check(cudaDeviceSynchronize());
    View baseline=mask_view(p,frame,memory),all_ones=mask_view(p,frame,memory),masked=mask_view(p,frame,memory);
    auto* ones=mask_managed<uint8_t>(223,memory);auto* bytes=mask_managed<uint8_t>(223,memory);
    std::fill(ones,ones+223,uint8_t(1));all_ones.policy_feature_mask=ones;masked.policy_feature_mask=bytes;
    auto* encoded_base=mask_managed<float>(446,memory);auto* encoded_ones=mask_managed<float>(446,memory);
    auto* encoded_masked=mask_managed<float>(446,memory);
    unsigned checks=0,terminal_observations=0;
    auto check=[&](bool ok,const char* reason){++checks;if(!ok)throw std::runtime_error(reason);};
    auto equal_bytes=[&](const void* a,const void* b,size_t size,const char* why){check(!std::memcmp(a,b,size),why);};
    cudaStream_t stream;rek5::cuda_check(cudaStreamCreate(&stream));
    for(int pattern=0;pattern<2;pattern++){
        for(int i=0;i<223;i++)bytes[i]=pattern?uint8_t(i%2):selected.values[i];
        for(View v:{baseline,all_ones,masked})fast_reset<<<1,32,0,stream>>>(v);
        rek5::cuda_check(cudaStreamSynchronize(stream));
        auto validate=[&](){
            encode_rows<<<2,256,0,stream>>>(baseline,encoded_base);
            encode_rows<<<2,256,0,stream>>>(all_ones,encoded_ones);
            encode_rows<<<2,256,0,stream>>>(masked,encoded_masked);
            rek5::cuda_check(cudaGetLastError());rek5::cuda_check(cudaStreamSynchronize(stream));
            for(int i=0;i<223;i++){
                check(all_ones.out.observations[i]==baseline.out.observations[i],"all-one learner output differs from omitted");
                check(masked.out.observations[i]==(bytes[i]?baseline.out.observations[i]:0.f),"learner fused mask incorrect");
            }
            for(int i=0;i<446;i++){
                check(encoded_ones[i]==encoded_base[i],"all-one actor output differs from omitted");
                check(encoded_masked[i]==(bytes[i%223]?encoded_base[i]:0.f),"actor-row mask incorrect");
                check(masked.raw[i]==baseline.raw[i],"raw diagnostic was masked");
            }
            for(View other:{all_ones,masked}){
                equal_bytes(other.state,baseline.state,sizeof(Arena),"mask changed dynamics");
                equal_bytes(other.rounds,baseline.rounds,sizeof(RekNative5RoundResult),"mask changed referee result");
                equal_bytes(other.masks,baseline.masks,66,"mask changed action legality");
                equal_bytes(other.learner_masks,baseline.learner_masks,33,"mask changed learner legality");
                equal_bytes(other.qpos,baseline.qpos,72*sizeof(float),"mask changed physical positions");
                equal_bytes(other.qvel,baseline.qvel,70*sizeof(float),"mask changed physical velocities");
                equal_bytes(other.rewards,baseline.rewards,2*sizeof(float),"mask changed rewards");
                equal_bytes(other.terminals,baseline.terminals,2*sizeof(float),"mask changed terminal flags");
                equal_bytes(other.out.logs,baseline.out.logs,sizeof(RekNative5Log),"mask changed metrics");
            }
            if(baseline.out.terminals[0]){
                ++terminal_observations;
                check(masked.state[0].tick==0,"terminal mask path missed autoreset");
                check(masked.out.observations[189]==(bytes[189]?1.f:0.f)*(.06f/120.f),"masked terminal output is not fresh initial observation");
            }
        };
        validate();
        for(int action:{2,0,6,7,1,2,0,0,1}){
            for(View v:{baseline,all_ones,masked})v.out.actions[0]=float(action);
            rek5::cuda_check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            for(View v:{baseline,all_ones,masked})fast_step<<<1,32,0,stream>>>(v,true);
            cudaGraph_t graph;cudaGraphExec_t executable;
            rek5::cuda_check(cudaStreamEndCapture(stream,&graph));
            rek5::cuda_check(cudaGraphInstantiate(&executable,graph,0));
            rek5::cuda_check(cudaGraphLaunch(executable,stream));rek5::cuda_check(cudaStreamSynchronize(stream));
            validate();rek5::cuda_check(cudaGraphExecDestroy(executable));rek5::cuda_check(cudaGraphDestroy(graph));
        }
    }
    check(terminal_observations==6,"missing terminal autoreset coverage");
    rek5::cuda_check(cudaStreamDestroy(stream));for(void* allocation:memory)rek5::cuda_check(cudaFree(allocation));
    std::printf("{\"passed\":true,\"checks\":%u,\"cuda_graph_steps\":18,\"terminal_autoresets\":%u,\"policy_rows\":2,\"mask_sha256\":\"%s\",\"raw_diagnostics_unchanged\":true}\n",checks,terminal_observations,selected.sha256.c_str());
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
