// Offline frozen behavior replay. Never connects to REK or steps an environment.
#include "authentic_trajectory.h"
#include "owned_yaw_trajectory.h"
#include "native_policy.h"
#include "device_storage.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <filesystem>
#include <memory>
#ifndef REK_AUTHENTIC_NATIVE_OBJECT_SHA256
#define REK_AUTHENTIC_NATIVE_OBJECT_SHA256 "unbound"
#endif

namespace {
using rek_authentic::require;
void policy_ok(int status) { if(status) throw std::runtime_error(rek_native_policy_error()); }
// Identical reduction order/intrinsics to native_policy.cu::sample. The logits
// are the exact FP32 diagnostic copies of the BF16 inference decoder values.
__global__ void behavior_logprob(const float* logits,const uint8_t* mask,int action,float* out) {
    float maximum=-INFINITY,total=0;
    for(int k=0;k<33;++k) {
        const float x=mask[k]?logits[k]:-1e4f;
        if(x>maximum) { total*=__expf(maximum-x); maximum=x; }
        total+=__expf(x-maximum);
    }
    out[0]=logits[action]-(maximum+__logf(total)); out[1]=logits[33];
}
template<class T> void put(std::vector<unsigned char>& b,size_t offset,T value) {
    std::memcpy(b.data()+offset,&value,sizeof(T));
}
void put_hash(std::vector<unsigned char>& b,size_t offset,const std::string& value) {
    require(value.size()==64,"SHA256 length mismatch");
    for(size_t i=0;i<32;++i) {
        unsigned x=0; require(std::sscanf(value.substr(i*2,2).c_str(),"%2x",&x)==1,"invalid SHA256"); b[offset+i]=x;
    }
}
void publish(const char* path,const void* data,size_t size) {
    FILE* f=std::fopen(path,"wbx"); require(f!=nullptr,"output exists or cannot be created");
    const size_t count=std::fwrite(data,1,size,f); const int status=std::fclose(f);
    require(count==size && status==0,"behavior output write failed");
}
}
int main(int argc,char** argv) {
    try {
        require(argc==5||argc==6||argc==7,"Usage: replay-authentic-behavior DATA CHECKPOINT SHA256 NEW_REPLAY_BINARY [IDENTITY_JSON EXACT_WORKER | --observation-schema=rek.native5.scaled_polar_xy.owned_yaw_v2]");
        const bool owned_yaw=argc==6;
        require(!owned_yaw || std::string(argv[5])==std::string("--observation-schema=")+rek_owned_yaw::kSchema,"unknown replay observation schema option");
        const auto data=owned_yaw?rek_owned_yaw_trajectory::load(argv[1]):rek_authentic::load(argv[1]);
        const bool identity=data.format_version==3;
        require(identity==(argc==7),"v3 requires exact identity JSON and worker executable");
        if(identity) {
            const auto identity_bytes=rek_authentic::read_file(argv[5]), worker_bytes=rek_authentic::read_file(argv[6]);
            require(rek_authentic::sha256(identity_bytes.data(),identity_bytes.size())==data.identity_sha256,
                "behavior identity JSON hash mismatch");
            require(rek_authentic::sha256(worker_bytes.data(),worker_bytes.size())==data.worker_sha256,
                "behavior worker executable hash mismatch");
            require(data.native_object_sha256==REK_AUTHENTIC_NATIVE_OBJECT_SHA256,"replay native-policy object hash mismatch");
            require(data.checkpoint_sha256==argv[3],"dataset checkpoint identity mismatch");
        }
        const std::string summary_path=std::string(argv[4])+".json";
        require(!std::filesystem::exists(argv[4]) && !std::filesystem::exists(summary_path),"replay output exists");
        rek5::cuda_check(cudaSetDevice(0)); cudaStream_t stream; rek5::cuda_check(cudaStreamCreate(&stream));
        rek5::DeviceStorage storage;
        float* observation=storage.alloc<float>(223); auto* mask=storage.alloc<uint8_t>(33);
        float* terminal=storage.alloc<float>(1); float* action=storage.alloc<float>(1); float* stats=storage.alloc<float>(2);
        RekNativePolicyConfig config{}; config.abi_version=REK_NATIVE_POLICY_ABI;
        config.checkpoint_path=argv[2]; config.expected_sha256=argv[3]; config.hidden_size=256; config.num_layers=2;
        config.batch=1; config.precision=REK_NATIVE_POLICY_BF16; config.seed=data.rows.front().behavior_seed;
        std::unique_ptr<RekNativePolicy,decltype(&rek_native_policy_destroy)> policy(rek_native_policy_create(&config,stream),rek_native_policy_destroy);
        require(bool(policy),rek_native_policy_error());
        const std::string checkpoint=rek_native_policy_sha256(policy.get()); require(checkpoint==argv[3],"checkpoint SHA mismatch");
        std::vector<unsigned char> output(rek_authentic::REPLAY_HEADER_BYTES+data.rows.size()*rek_authentic::REPLAY_ROW_BYTES,0);
        std::memcpy(output.data(),identity?"REKBR003":owned_yaw?"REKBR002":"REKBR001",8); put<uint32_t>(output,8,identity?3:owned_yaw?2:1); put<uint32_t>(output,12,data.rows.size());
        put<uint32_t>(output,16,rek_authentic::REPLAY_ROW_BYTES); put_hash(output,24,data.digest);
        put_hash(output,56,checkpoint); put<uint64_t>(output,88,identity?0:73);
        if(identity) put_hash(output,96,data.identity_sha256);
        size_t rounds=0,matching=0; double minimum_logprob=0,maximum_logprob=-INFINITY;
        for(size_t i=0;i<data.rows.size();++i) {
            const auto& row=data.rows[i];
            if(i==0 || row.sequence!=data.rows[i-1].sequence) {
                // A new sequence was a new OS worker, including fresh Philox RNG.
                if(identity && i>0) {
                    policy.reset(); config.seed=row.behavior_seed;
                    policy.reset(rek_native_policy_create(&config,stream)); require(bool(policy),rek_native_policy_error());
                }
                policy_ok(rek_native_policy_reset(policy.get(),stream)); ++rounds;
            } else if(row.reset) {
                // Explicit within-worker resets must not restart sampler RNG.
                policy_ok(rek_native_policy_reset_recurrent(policy.get(),stream));
            }
            std::array<uint8_t,33> host_mask{}; for(int j=0;j<33;++j) host_mask[j]=uint8_t(row.support[j]);
            std::array<float,223> host_observation{};
            for(int j=0;j<223;++j) host_observation[j]=data.feature_mask[j]?row.obs[j]:0;
            rek5::cuda_check(cudaMemcpyAsync(observation,host_observation.data(),223*sizeof(float),cudaMemcpyHostToDevice,stream));
            rek5::cuda_check(cudaMemcpyAsync(mask,host_mask.data(),33,cudaMemcpyHostToDevice,stream));
            policy_ok(rek_native_policy_step_rows(policy.get(),observation,mask,terminal,action,0,1,0,stream));
            behavior_logprob<<<1,1,0,stream>>>(rek_native_policy_logits(policy.get()),mask,row.action,stats);
            float selected=-1,host_stats[2]; std::array<float,34> logits{};
            rek5::cuda_check(cudaMemcpyAsync(&selected,action,sizeof(float),cudaMemcpyDeviceToHost,stream));
            rek5::cuda_check(cudaMemcpyAsync(host_stats,stats,sizeof(host_stats),cudaMemcpyDeviceToHost,stream));
            rek5::cuda_check(cudaMemcpyAsync(logits.data(),rek_native_policy_logits(policy.get()),34*sizeof(float),cudaMemcpyDeviceToHost,stream));
            rek5::cuda_check(cudaStreamSynchronize(stream)); policy_ok(rek_native_policy_check_status(policy.get(),stream));
            if(selected!=row.action) {
                std::fprintf(stderr,"behavior_action_mismatch row=%zu sequence=%u source_seq=%u saved=%d replayed=%.9g\n",
                    i,row.sequence,row.source_seq,row.action,selected);
                throw std::runtime_error("frozen behavior did not reproduce recorded sampled action; no replay published");
            }
            ++matching; const size_t offset=rek_authentic::REPLAY_HEADER_BYTES+i*rek_authentic::REPLAY_ROW_BYTES;
            put<uint32_t>(output,offset,i); put<uint32_t>(output,offset+4,row.action);
            put<float>(output,offset+8,host_stats[0]); put<float>(output,offset+12,host_stats[1]);
            for(int j=0;j<34;++j) put<float>(output,offset+16+4*j,logits[j]);
            minimum_logprob=std::min(minimum_logprob,double(host_stats[0])); maximum_logprob=std::max(maximum_logprob,double(host_stats[0]));
        }
        const auto checked=owned_yaw?rek_owned_yaw_trajectory::decode_replay(output,data):rek_authentic::decode_replay(output,data); (void)checked;
        const std::string digest=rek_authentic::sha256(output.data(),output.size());
        char report[3000]; const int count=std::snprintf(report,sizeof(report),
            "{\"schema\":\"%s\",%s\"verification_passed\":true,\"rows\":%zu,\"rounds\":%zu,\"matching_sampled_actions\":%zu,\"mismatches\":0,\"dataset_sha256\":\"%s\",\"checkpoint_sha256\":\"%s\",\"replay_sha256\":\"%s\",\"seed_per_new_worker\":73,\"precision\":\"bf16\",\"input_feature_mask\":\"original_unmasked\",\"logprob_precision\":\"float32_native_sampler_reduction\",\"minimum_logprob\":%.9g,\"maximum_logprob\":%.9g,\"terminal_race_requests_retained\":true,\"game_connection\":false,\"training_performed\":false}\n",
            identity?"rek.authentic_behavior_replay.v3":owned_yaw?"rek.authentic_behavior_replay.owned_yaw_v2":"rek.authentic_behavior_replay.v1",
            owned_yaw?"\"observation_schema\":\"rek.native5.scaled_polar_xy.owned_yaw_v2\",\"trajectory_format\":\"REKRL002\",\"replay_format\":\"REKBR002\",":"",
            data.rows.size(),rounds,matching,data.digest.c_str(),checkpoint.c_str(),digest.c_str(),minimum_logprob,maximum_logprob);
        require(count>0 && count<int(sizeof(report)),"replay report overflow");
        std::string report_text(report,count);
        if(identity) {
            const auto replace=[&](const std::string& before,const std::string& after) {
                const auto at=report_text.find(before); require(at!=std::string::npos,"missing replay report field");
                report_text.replace(at,before.size(),after);
            };
            replace("\"seed_per_new_worker\":73", "\"seed_per_new_worker\":\"recorded_per_sequence\"");
            replace("\"input_feature_mask\":\"original_unmasked\"", "\"feature_mask_sha256\":\""+
                rek_authentic::sha256(data.feature_mask.data(),data.feature_mask.size())+"\",\"identity_sha256\":\""+
                data.identity_sha256+"\",\"worker_sha256\":\""+data.worker_sha256+"\",\"native_object_sha256\":\""+data.native_object_sha256+"\"");
        }
        publish(argv[4],output.data(),output.size()); publish(summary_path.c_str(),report_text.data(),report_text.size());
        std::printf("%s",report_text.c_str()); policy.reset(); rek5::cuda_check(cudaStreamDestroy(stream)); return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"authentic_replay_error: %s\n",e.what()); return 2; }
}
