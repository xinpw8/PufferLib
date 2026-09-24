#define main pinned_worker_main
#include "live_policy_worker.cu"
#undef main

int main(int argc,char**argv){try{
    require(argc==5,"usage_checkpoint_sha_seed_allones_mask");
    char* end=nullptr;const auto seed=std::strtoull(argv[3],&end,10);require(end&&!*end,"invalid_seed");
    Engine engine(argv[1],argv[2],seed,false,argv[4],false);
    for(auto v:engine.features.values)require(v==1,"full_live_allones_required");
    auto ready=response("drift_ready");str(ready.get(),"checkpoint_sha256",engine.sha);str(ready.get(),"observation_schema",rek_balance8::kSchema);str(ready.get(),"feature_mask_sha256",engine.features.sha256);number(ready.get(),"seed",double(seed));emit(ready.get());
    ProtocolState state;std::string line;bool overlong=false;
    while(read_line(line,overlong)){
        require(!overlong,"line_too_long");const auto r=parse(line,rek_balance8::kSchema);state.check(r);
        if(r.kind==Request::Close){state.accept(r);break;}
        const bool clear=state.new_round(r)||r.kind==Request::Reset||r.terminal;
        if(clear)engine.reset();
        if(r.kind==Request::Step&&!r.terminal){
            float ms=0;const int chosen=engine.infer(r,ms);std::array<float,34> logits{};
            cuda_ok(cudaMemcpyAsync(logits.data(),rek_native_policy_logits(engine.policy.get()),sizeof(logits),cudaMemcpyDeviceToHost,engine.stream));cuda_ok(cudaStreamSynchronize(engine.stream));
            auto row=response("drift_row",&r);number(row.get(),"action",chosen);boolean(row.get(),"recurrent_reset",clear||state.reset_pending);
            auto* values=cJSON_AddArrayToObject(row.get(),"logits");for(float x:logits){require(std::isfinite(x),"nonfinite_logit");cJSON_AddItemToArray(values,cJSON_CreateNumber(x));}emit(row.get());state.reset_pending=false;
        }else state.reset_pending=true;
        state.accept(r);
    }
    return 0;
}catch(const std::exception&e){std::cerr<<"offline_drift_error: "<<e.what()<<'\n';return 2;}}
