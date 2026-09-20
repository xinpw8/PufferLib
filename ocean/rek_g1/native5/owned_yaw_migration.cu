// Offline migration and exact dual-policy proof. No optimizer or environment.
#include "puffer5_bc_core.cuh"
#include "owned_yaw_trajectory.h"
#include "native_policy.h"
#include "device_storage.cuh"
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

namespace {
using rek_authentic::require;
constexpr int O=223,A=33,H=256,L=2;
struct Layout {size_t parameters=0,encoder_begin=0;};
Layout layout() {
    Arch arch=build_arch("rek_native5",O,H,L,A,false,128);Allocator params{};
    Weights w=weights_create(&arch,&params);
    auto* encoder=static_cast<EncoderWeights*>(w.encoder);
    auto* network=static_cast<MinGRUWeights*>(w.network);
    require(params.num_regs==4,"unexpected registered parameter count");
    const long dimensions[4][2]={{H,O},{A+1,H},{3*H,H},{3*H,H}};
    Layout out;int found=0;
    for(int i=0;i<params.num_regs;++i) {
        const auto& entry=params.regs[i];
        require(ndim(entry.shape)==2 && entry.shape[0]==dimensions[i][0] && entry.shape[1]==dimensions[i][1]
            && entry.elem_size==sizeof(precision_t) && (out.parameters*sizeof(precision_t))%16==0,
            "pinned native architecture layout changed");
        if(entry.data_ptr==reinterpret_cast<void**>(&encoder->weight.data)) {
            require(i==0,"encoder registration order changed");out.encoder_begin=out.parameters;++found;
        }
        out.parameters+=numel(entry.shape);
    }
    require(found==1 && out.parameters==size_t(params.total_elems) &&
        params.total_bytes==long(out.parameters*sizeof(precision_t)),"flat parameter layout mismatch");
    free(network->weights);free(w.network);free(w.decoder);free(w.encoder);free(params.regs);
    return out;
}
std::vector<float> load_weights(const char* file,const char* expected,Layout l) {
    const auto bytes=rek_authentic::read_file(file);
    require(bytes.size()==l.parameters*sizeof(float) && rek_authentic::sha256(bytes.data(),bytes.size())==expected,
        "checkpoint size or SHA mismatch");
    std::vector<float> out(l.parameters);std::memcpy(out.data(),bytes.data(),bytes.size());
    for(float x:out)require(std::isfinite(x),"nonfinite checkpoint parameter");return out;
}
size_t column_index(Layout l,int h) {return l.encoder_begin+size_t(h)*O+rek_owned_yaw::kColumn;}
void zero_column(std::vector<float>& weights,Layout l) {
    require(weights.size()==l.parameters,"migration checkpoint size mismatch");
    for(int h=0;h<H;++h)weights[column_index(l,h)]=0.f;
}
void verify_weights(const std::vector<float>& original,const std::vector<float>& candidate,Layout l) {
    auto expected=original;zero_column(expected,l);
    require(candidate.size()==expected.size() &&
        std::memcmp(expected.data(),candidate.data(),expected.size()*sizeof(float))==0,
        "migration differs from exactly zeroing encoder column187");
}
void publish(const char* path,const void* bytes,size_t size) {
    FILE* out=fopen(path,"wbx");require(out!=nullptr,"new output path required");
    const bool written=fwrite(bytes,1,size,out)==size;const int closed=fclose(out);
    require(written&&closed==0,"output write failed");
    const auto saved=rek_authentic::read_file(path);
    require(saved.size()==size&&std::memcmp(saved.data(),bytes,size)==0,"output readback mismatch");
}
template<class T>void put(std::vector<unsigned char>& b,size_t offset,T x){std::memcpy(b.data()+offset,&x,sizeof(x));}
void put_hash(std::vector<unsigned char>& b,size_t offset,const std::string& value) {
    require(value.size()==64,"hash length mismatch");
    for(size_t i=0;i<32;++i){unsigned x;require(std::sscanf(value.substr(i*2,2).c_str(),"%2x",&x)==1,"hash parse failed");b[offset+i]=x;}
}
void policy_ok(int status){if(status)throw std::runtime_error(rek_native_policy_error());}
__global__ void chosen_logprob(const float* logits,const uint8_t* mask,int action,float* out) {
    float maximum=-INFINITY,total=0;
    for(int k=0;k<A;++k) {
        const float x=mask[k]?logits[k]:-1e4f;
        if(x>maximum){total*=__expf(maximum-x);maximum=x;}total+=__expf(x-maximum);
    }
    out[0]=logits[action]-(maximum+__logf(total));
}
struct Forward {std::array<float,A+1> logits{};float action=0,logprob=0;};
struct Policy {
    rek5::DeviceStorage storage;
    std::unique_ptr<RekNativePolicy,decltype(&rek_native_policy_destroy)> policy{nullptr,rek_native_policy_destroy};
    float *obs=nullptr,*terminal=nullptr,*action=nullptr,*logprob=nullptr;uint8_t* mask=nullptr;
    Policy(const char* checkpoint,const char* digest) {
        obs=storage.alloc<float>(O);terminal=storage.alloc<float>(1);action=storage.alloc<float>(1);
        logprob=storage.alloc<float>(1);mask=storage.alloc<uint8_t>(A);
        RekNativePolicyConfig config{};config.abi_version=REK_NATIVE_POLICY_ABI;config.checkpoint_path=checkpoint;
        config.expected_sha256=digest;config.hidden_size=H;config.num_layers=L;config.batch=1;
        config.precision=REK_NATIVE_POLICY_BF16;config.seed=73;
        policy.reset(rek_native_policy_create(&config,0));require(bool(policy),rek_native_policy_error());
    }
    Forward step(const rek_authentic::Row& row,bool fresh) {
        if(fresh)policy_ok(rek_native_policy_reset(policy.get(),0));
        else if(row.reset)policy_ok(rek_native_policy_reset_recurrent(policy.get(),0));
        std::array<uint8_t,A> support{};for(int j=0;j<A;++j)support[j]=uint8_t(row.support[j]);
        rek5::cuda_check(cudaMemcpy(obs,row.obs.data(),O*sizeof(float),cudaMemcpyHostToDevice));
        rek5::cuda_check(cudaMemcpy(mask,support.data(),A,cudaMemcpyHostToDevice));
        policy_ok(rek_native_policy_step_rows(policy.get(),obs,mask,terminal,action,0,1,0,0));
        chosen_logprob<<<1,1>>>(rek_native_policy_logits(policy.get()),mask,row.action,logprob);
        rek5::cuda_check(cudaGetLastError());Forward out;
        rek5::cuda_check(cudaMemcpy(out.logits.data(),rek_native_policy_logits(policy.get()),(A+1)*sizeof(float),cudaMemcpyDeviceToHost));
        rek5::cuda_check(cudaMemcpy(&out.action,action,sizeof(float),cudaMemcpyDeviceToHost));
        rek5::cuda_check(cudaMemcpy(&out.logprob,logprob,sizeof(float),cudaMemcpyDeviceToHost));
        policy_ok(rek_native_policy_check_status(policy.get(),0));return out;
    }
};
void cpu_test() {
    const auto l=layout();require(l.parameters==459008&&l.encoder_begin==0,"unexpected native layout");
    std::vector<float> old(l.parameters);for(size_t i=0;i<old.size();++i)old[i]=float(int(i%127)-63)/64;
    auto fresh=old;zero_column(fresh,l);verify_weights(old,fresh,l);
    int columns=0;for(int h=0;h<H;++h){require(fresh[column_index(l,h)]==0,"column not zero");++columns;}
    for(size_t index:std::array<size_t,4>{0,size_t(O*H-1),size_t(O*H),l.parameters-1}) {
        auto broken=fresh;broken[index]+=1;bool rejected=false;
        try{verify_weights(old,broken,l);}catch(const std::exception&){rejected=true;}
        require(rejected,"non-column mutation accepted");
    }
    // A single valid closed row verifies the explicit format gate and byte guard.
    rek_authentic::Dataset d;d.feature_mask.fill(1);rek_authentic::Row row;
    row.reset=1;row.action=0;row.policy_weight=row.value_weight=1;row.applied=1;row.terminal_after=1;
    row.next_time=row.dt=.02;row.gamma=row.lambda=1;row.next_source_seq=1;row.support.fill(1);d.rows.push_back(row);
    std::vector<unsigned char> bytes(rek_authentic::HEADER_BYTES+rek_authentic::ROW_BYTES,0);
    std::memcpy(bytes.data(),"REKRL001",8);put<uint32_t>(bytes,8,1);put<uint32_t>(bytes,12,O);
    put<uint32_t>(bytes,16,A);put<uint32_t>(bytes,20,1);put<uint32_t>(bytes,24,rek_authentic::ROW_BYTES);put<uint32_t>(bytes,28,1);
    std::fill(bytes.begin()+32,bytes.begin()+255,1);const size_t p=rek_authentic::HEADER_BYTES;
    put<uint32_t>(bytes,p+8,1);put<float>(bytes,p+16,1);put<double>(bytes,p+1056,.02);put<double>(bytes,p+1064,.02);
    put<float>(bytes,p+1072,1);put<float>(bytes,p+1076,1);put<uint32_t>(bytes,p+1092,1);
    put<uint32_t>(bytes,p+1112,1);put<uint32_t>(bytes,p+1116,1);put<float>(bytes,p+1120,1);
    for(int j=0;j<A;++j)put<float>(bytes,p+924+j*4,1);
    put<float>(bytes,p+32+182*4,1);put<float>(bytes,p+32+183*4,1);
    auto v2=bytes;std::memcpy(v2.data(),"REKRL002",8);put<uint32_t>(v2,8,2);put<float>(v2,p+32+187*4,-1);
    rek_owned_yaw_trajectory::verify_column_only_upgrade(bytes,v2);
    bool rejected=false;try{rek_authentic::decode(v2);}catch(const std::exception&){rejected=true;}
    require(rejected,"legacy loader silently accepted v2");
    auto wrong=v2;put<float>(wrong,p+32+178*4,1);rejected=false;
    try{rek_owned_yaw_trajectory::verify_column_only_upgrade(bytes,wrong);}catch(const std::exception&){rejected=true;}
    require(rejected,"non187 data mutation accepted");
    std::cout<<"{\"cpu_self_test\":\"passed\",\"parameters\":"<<l.parameters<<",\"zeroed_column_weights\":"<<columns
        <<",\"v1_rejects_v2\":true,\"outside_column_byte_guards\":true,\"cuda_calls\":0}\n";
}
}
int main(int argc,char** argv) {
    try {
        if(argc==2&&std::string(argv[1])=="--cpu-self-test"){cpu_test();return 0;}
        const bool migrate=argc==5&&std::string(argv[1])=="--migrate";
        const bool replay=argc==9&&std::string(argv[1])=="--replay";
        require(migrate||replay,"usage: owned-yaw-migration --migrate ORIGINAL SHA NEW | --replay V1_DATA V2_DATA ORIGINAL OLD_SHA MIGRATED NEW_SHA NEW_REPLAY");
        const auto l=layout();
        if(migrate) {
            const auto old=load_weights(argv[2],argv[3],l);auto fresh=old;zero_column(fresh,l);verify_weights(old,fresh,l);
            publish(argv[4],fresh.data(),fresh.size()*sizeof(float));
            std::cout<<"{\"mode\":\"column187_zero_migration\",\"from_schema\":\""<<rek_owned_yaw::kLegacySchema
                <<"\",\"to_schema\":\""<<rek_owned_yaw::kSchema<<"\",\"input_sha256\":\""<<argv[3]
                <<"\",\"output_sha256\":\""<<rek_authentic::sha256(fresh.data(),fresh.size()*sizeof(float))
                <<"\",\"parameters\":"<<l.parameters<<",\"zeroed_column_weights\":256,\"other_parameter_bytes_identical\":true,\"cuda_calls\":0}\n";return 0;
        }
        require(!std::filesystem::exists(argv[8]),"new replay output required");
        const auto old_bytes=rek_authentic::read_file(argv[2]),fresh_bytes=rek_authentic::read_file(argv[3]);
        rek_owned_yaw_trajectory::verify_column_only_upgrade(old_bytes,fresh_bytes);
        const auto old_data=rek_authentic::decode(old_bytes),fresh_data=rek_owned_yaw_trajectory::decode(fresh_bytes);
        const auto old_weights=load_weights(argv[4],argv[5],l),fresh_weights=load_weights(argv[6],argv[7],l);
        verify_weights(old_weights,fresh_weights,l);
        Policy old_policy(argv[4],argv[5]),fresh_policy(argv[6],argv[7]);
        std::vector<unsigned char> output(rek_authentic::REPLAY_HEADER_BYTES+fresh_data.rows.size()*rek_authentic::REPLAY_ROW_BYTES,0);
        std::memcpy(output.data(),"REKBR002",8);put<uint32_t>(output,8,2);put<uint32_t>(output,12,fresh_data.rows.size());
        put<uint32_t>(output,16,rek_authentic::REPLAY_ROW_BYTES);put_hash(output,24,fresh_data.digest);put_hash(output,56,argv[7]);put<uint64_t>(output,88,73);
        size_t nonzero=0;
        for(size_t i=0;i<old_data.rows.size();++i) {
            const bool start=i==0||old_data.rows[i].sequence!=old_data.rows[i-1].sequence;
            const auto old=old_policy.step(old_data.rows[i],start),fresh=fresh_policy.step(fresh_data.rows[i],start);
            require(old.action==old_data.rows[i].action&&fresh.action==fresh_data.rows[i].action,"native sampled action differs from recorded behavior");
            require(std::memcmp(old.logits.data(),fresh.logits.data(),sizeof(old.logits))==0 &&
                std::memcmp(&old.logprob,&fresh.logprob,sizeof(float))==0 &&
                std::memcmp(&old.action,&fresh.action,sizeof(float))==0,"migration changed native34logits, chosenlogprob, or sampledaction");
            if(fresh_data.rows[i].obs[rek_owned_yaw::kColumn]!=0)++nonzero;
            const size_t p=rek_authentic::REPLAY_HEADER_BYTES+i*rek_authentic::REPLAY_ROW_BYTES;
            put<uint32_t>(output,p,i);put<uint32_t>(output,p+4,fresh_data.rows[i].action);
            put<float>(output,p+8,fresh.logprob);put<float>(output,p+12,fresh.logits[A]);
            std::memcpy(output.data()+p+16,fresh.logits.data(),sizeof(fresh.logits));
        }
        const auto checked=rek_owned_yaw_trajectory::decode_replay(output,fresh_data);(void)checked;
        publish(argv[8],output.data(),output.size());
        std::cout<<"{\"verification_passed\":true,\"rows\":"<<fresh_data.rows.size()<<",\"rounds\":"<<fresh_data.sequences.size()
            <<",\"nonzero_v2_rows\":"<<nonzero<<",\"all34logits_logprobs_sampledactions_bitwise_equal\":true,\"legacy187_positive_zero\":true,"
            "\"non187_data_bytes_unchanged\":true,\"other_parameter_bytes_unchanged\":true,\"seed_per_sequence\":73,\"precision\":\"bf16\","
            "\"old_dataset_sha256\":\""<<old_data.digest<<"\",\"new_dataset_sha256\":\""<<fresh_data.digest
            <<"\",\"actual_behavior_checkpoint_sha256\":\""<<argv[5]<<"\",\"compatible_migrated_checkpoint_sha256\":\""<<argv[7]
            <<"\",\"v2_replay_sha256\":\""<<rek_authentic::sha256(output.data(),output.size())
            <<"\",\"training_performed\":false,\"game_connection\":false}\n";return 0;
    }catch(const std::exception& e){std::cerr<<"owned_yaw_migration: "<<e.what()<<'\n';return 1;}
}
