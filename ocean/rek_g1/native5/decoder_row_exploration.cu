// Offline action-distribution intervention. No optimizer, environment, or replay output.
#include "puffer5_bc_core.cuh"
#include "authentic_trajectory.h"
#include "native_policy.h"
#include "device_storage.cuh"
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>

namespace {
using rek_authentic::require;
constexpr int O=223,A=33,H=256,L=2,KICK=17,DONOR=23;
constexpr double TARGET=.1;
struct Layout {size_t count=0,decoder=0;};
Layout layout() {
    Arch arch=build_arch("rek_native5",O,H,L,A,false,128);Allocator params{};
    Weights w=weights_create(&arch,&params);
    auto* decoder=static_cast<DecoderWeights*>(w.decoder);
    auto* network=static_cast<MinGRUWeights*>(w.network);
    require(params.num_regs==4&&!decoder->continuous,"unexpected bias-free discrete architecture");
    const long expected[4][2]={{H,O},{A+1,H},{3*H,H},{3*H,H}};
    Layout out;int found=0;
    for(int i=0;i<params.num_regs;++i) {
        const auto& e=params.regs[i];
        require(ndim(e.shape)==2&&e.shape[0]==expected[i][0]&&e.shape[1]==expected[i][1]
            &&e.elem_size==sizeof(precision_t)&&(out.count*sizeof(precision_t))%16==0,
            "pinned native architecture layout changed");
        if(e.data_ptr==reinterpret_cast<void**>(&decoder->weight.data)) {
            require(i==1,"decoder registration order changed");out.decoder=out.count;++found;
        }
        out.count+=numel(e.shape);
    }
    require(found==1&&out.count==size_t(params.total_elems)&&
        params.total_bytes==long(out.count*sizeof(precision_t)),"flat checkpoint layout mismatch");
    free(network->weights);free(w.network);free(w.decoder);free(w.encoder);free(params.regs);
    return out;
}
std::vector<float> weights(const char* path,const std::string& expected,Layout l) {
    const auto bytes=rek_authentic::read_file(path);
    require(bytes.size()==l.count*sizeof(float)&&rek_authentic::sha256(bytes.data(),bytes.size())==expected,
        "checkpoint size or SHA mismatch");
    std::vector<float> out(l.count);std::memcpy(out.data(),bytes.data(),bytes.size());
    for(float x:out)require(std::isfinite(x),"nonfinite checkpoint parameter");return out;
}
std::vector<float> blend(const std::vector<float>& original,Layout l,double alpha) {
    require(original.size()==l.count&&std::isfinite(alpha)&&alpha>=0&&alpha<=1,"invalid decoder blend");
    auto out=original;
    for(int h=0;h<H;++h) {
        const size_t k=l.decoder+KICK*H+h,d=l.decoder+DONOR*H+h;
        if(alpha==0)continue; // Preserve even signed zero at the identity endpoint.
        out[k]=alpha==1?original[d]:float((1-alpha)*double(original[k])+alpha*double(original[d]));
        require(std::isfinite(out[k]),"nonfinite blended parameter");
    }
    return out;
}
void verify_weights(const std::vector<float>& original,const std::vector<float>& candidate,Layout l,double alpha) {
    const auto expected=blend(original,l,alpha);
    require(candidate.size()==expected.size()&&
        std::memcmp(expected.data(),candidate.data(),expected.size()*sizeof(float))==0,
        "candidate differs from exactly blending decoder row17 toward23");
}
std::string digest(const std::vector<float>& x){return rek_authentic::sha256(x.data(),x.size()*sizeof(float));}
void publish(const char* path,const std::vector<float>& x) {
    FILE* f=std::fopen(path,"wbx");require(f!=nullptr,"new checkpoint output required");
    const size_t written=std::fwrite(x.data(),sizeof(float),x.size(),f);const int closed=std::fclose(f);
    require(written==x.size()&&closed==0,"checkpoint write failed");
    const auto saved=rek_authentic::read_file(path);
    require(saved.size()==x.size()*sizeof(float)&&std::memcmp(saved.data(),x.data(),saved.size())==0,
        "checkpoint readback mismatch");
}
std::array<double,A> probabilities(const std::array<double,A>& logits,const rek_authentic::Row& row) {
    double maximum=-INFINITY,total=0;std::array<double,A> out{};
    for(int k=0;k<A;++k) {require(std::isfinite(logits[k]),"nonfinite diagnostic logit");
        if(row.support[k])maximum=std::max(maximum,logits[k]);}
    require(std::isfinite(maximum),"no legal action");
    for(int k=0;k<A;++k)if(row.support[k]){out[k]=std::exp(logits[k]-maximum);total+=out[k];}
    for(auto& x:out)x/=total;return out;
}
struct Mass {
    size_t rows=0,legal=0,attack_legal=0;double kick=0,hook=0,attack=0;
    void add(const std::array<double,A>& logits,const rek_authentic::Row& row) {
        const auto p=probabilities(logits,row);++rows;legal+=row.support[KICK]!=0;
        bool available=false;for(int k=16;k<A;++k){attack+=p[k];available|=row.support[k]!=0;}
        attack_legal+=available;kick+=p[KICK];hook+=p[DONOR];
    }
    double share()const{require(attack>0,"no legal attack probability mass");return kick/attack;}
};
std::array<double,A> interpolated(const rek_authentic::ReplayRow& row,double alpha) {
    std::array<double,A> z{};for(int k=0;k<A;++k)z[k]=row.logits[k];
    z[KICK]=(1-alpha)*z[KICK]+alpha*z[DONOR];return z;
}
Mass prediction(const rek_authentic::Dataset& data,const rek_authentic::Replay& replay,double alpha) {
    Mass result;for(size_t i=0;i<data.rows.size();++i)result.add(interpolated(replay.rows[i],alpha),data.rows[i]);return result;
}
double calibrate(const rek_authentic::Dataset& data,const rek_authentic::Replay& replay) {
    require(prediction(data,replay,0).share()<TARGET&&prediction(data,replay,1).share()>TARGET,
        "target not bracketed by unchanged and full donor row");
    double lo=0,hi=1;
    for(int i=0;i<60;++i){const double mid=(lo+hi)/2;if(prediction(data,replay,mid).share()<TARGET)lo=mid;else hi=mid;}
    const double alpha=(lo+hi)/2;
    require(std::abs(prediction(data,replay,alpha).share()-TARGET)<1e-10,"blend target solve failed");return alpha;
}
void print_mass(const Mass& m) {
    std::cout<<"{\"rows\":"<<m.rows<<",\"kick_legal_rows\":"<<m.legal<<",\"attack_legal_rows\":"<<m.attack_legal
        <<",\"kick_probability_sum\":"<<m.kick<<",\"hook_probability_sum\":"<<m.hook<<",\"attack_probability_sum\":"<<m.attack
        <<",\"attack_mass_weighted_kick_probability\":"<<m.share()<<'}';
}
void policy_ok(int status){if(status)throw std::runtime_error(rek_native_policy_error());}
// Native reduction order, evaluated for the action actually sampled by each policy.
__global__ void sampled_logprob(const float* logits,const uint8_t* mask,const float* action,float* out) {
    float maximum=-INFINITY,total=0;
    for(int k=0;k<A;++k){const float x=mask[k]?logits[k]:-1e4f;
        if(x>maximum){total*=__expf(maximum-x);maximum=x;}total+=__expf(x-maximum);}
    const int selected=int(action[0]);out[0]=selected>=0&&selected<A?logits[selected]-(maximum+__logf(total)):NAN;
}
struct Forward {std::array<float,A+1> logits{};float action=0,logprob=0;};
struct Policy {
    rek5::DeviceStorage storage;
    std::unique_ptr<RekNativePolicy,decltype(&rek_native_policy_destroy)> policy{nullptr,rek_native_policy_destroy};
    float *obs=nullptr,*terminal=nullptr,*action=nullptr,*logprob=nullptr;uint8_t* mask=nullptr;
    Policy(const char* path,const std::string& sha) {
        obs=storage.alloc<float>(O);terminal=storage.alloc<float>(1);action=storage.alloc<float>(1);
        logprob=storage.alloc<float>(1);mask=storage.alloc<uint8_t>(A);
        RekNativePolicyConfig c{};c.abi_version=REK_NATIVE_POLICY_ABI;c.checkpoint_path=path;c.expected_sha256=sha.c_str();
        c.hidden_size=H;c.num_layers=L;c.batch=1;c.precision=REK_NATIVE_POLICY_BF16;c.seed=73;
        policy.reset(rek_native_policy_create(&c,0));require(bool(policy),rek_native_policy_error());
    }
    Forward step(const rek_authentic::Row& row,bool fresh) {
        if(fresh)policy_ok(rek_native_policy_reset(policy.get(),0));
        else if(row.reset)policy_ok(rek_native_policy_reset_recurrent(policy.get(),0));
        std::array<uint8_t,A> support{};for(int k=0;k<A;++k)support[k]=uint8_t(row.support[k]);
        rek5::cuda_check(cudaMemcpy(obs,row.obs.data(),O*sizeof(float),cudaMemcpyHostToDevice));
        rek5::cuda_check(cudaMemcpy(mask,support.data(),A,cudaMemcpyHostToDevice));
        policy_ok(rek_native_policy_step_rows(policy.get(),obs,mask,terminal,action,0,1,0,0));
        sampled_logprob<<<1,1>>>(rek_native_policy_logits(policy.get()),mask,action,logprob);
        rek5::cuda_check(cudaGetLastError());Forward out;
        rek5::cuda_check(cudaMemcpy(out.logits.data(),rek_native_policy_logits(policy.get()),sizeof(out.logits),cudaMemcpyDeviceToHost));
        rek5::cuda_check(cudaMemcpy(&out.action,action,sizeof(float),cudaMemcpyDeviceToHost));
        rek5::cuda_check(cudaMemcpy(&out.logprob,logprob,sizeof(float),cudaMemcpyDeviceToHost));
        policy_ok(rek_native_policy_check_status(policy.get(),0));
        require(std::isfinite(out.action)&&out.action>=0&&out.action<A&&out.action==int(out.action)
            &&row.support[int(out.action)]==1&&std::isfinite(out.logprob),"sampled action or logprob invalid");return out;
    }
};
void cpu_test() {
    const auto l=layout();require(l.count==459008&&l.decoder==57088,"registered layout fixture changed");
    std::vector<float> old(l.count);for(size_t i=0;i<old.size();++i)old[i]=float(int(i%127)-63)/64;
    old[0]=-0.f;const auto identity=blend(old,l,0);verify_weights(old,identity,l,0);
    require(std::memcmp(old.data(),identity.data(),old.size()*sizeof(float))==0,"alpha0 changed bytes");
    const auto donor=blend(old,l,1),half=blend(old,l,.5);verify_weights(old,half,l,.5);
    require(std::memcmp(donor.data()+l.decoder+KICK*H,old.data()+l.decoder+DONOR*H,H*sizeof(float))==0,"alpha1 not donor");
    size_t rejected=0;
    for(size_t index:std::array<size_t,7>{0,l.decoder+KICK*H-1,l.decoder+KICK*H,l.decoder+(KICK+1)*H,
            l.decoder+DONOR*H,l.decoder+A*H,l.count-1}) {
        auto bad=half;bad[index]+=1;try{verify_weights(old,bad,l,.5);}catch(const std::exception&){++rejected;}
    }
    for(double a:std::array<double,4>{-.1,1.1,INFINITY,NAN}) {
        try{blend(old,l,a);}catch(const std::exception&){++rejected;}
    }
    require(rejected==11,"mutation or invalid-alpha test did not reject");
    rek_authentic::Row row;row.support.fill(1);std::array<double,A> z{};
    const auto p=probabilities(z,row);double sum=0;for(double x:p)sum+=x;
    require(std::abs(sum-1)<1e-14&&std::abs(p[KICK]-1./A)<1e-14,"softmax fixture failed");
    row.support[KICK]=0;z[KICK]=10000;
    require(probabilities(z,row)[KICK]==0,"masked kick acquired probability");
    rek_authentic::Dataset d;row.support.fill(1);d.rows.push_back(row);
    rek_authentic::Replay r;r.rows.resize(1);r.rows[0].logits[KICK]=-8;r.rows[0].logits[DONOR]=2;
    const double alpha=calibrate(d,r);require(alpha>0&&alpha<1&&std::abs(prediction(d,r,alpha).share()-.1)<1e-10,"target fixture failed");
    std::cout<<"{\"cpu_self_test\":\"passed\",\"parameters\":"<<l.count
        <<",\"decoder_float_offset\":"<<l.decoder<<",\"bias_present\":false,\"negative_checks\":"<<rejected
        <<",\"identity_donor_midpoint_and_mask_tests\":true,\"cuda_calls\":0}\n";
}
}
int main(int argc,char** argv) {
    try {
        std::cout<<std::setprecision(17);
        if(argc==2&&std::string(argv[1])=="--cpu-self-test"){cpu_test();return 0;}
        const bool fit=argc==7&&std::string(argv[1])=="--calibrate";
        const bool verify=argc==8&&std::string(argv[1])=="--verify";
        require(fit||verify,"usage: decoder-row-exploration --calibrate DATA REPLAY ORIGINAL SHA NEW_CHECKPOINT | --verify DATA REPLAY ORIGINAL SHA CANDIDATE ALPHA");
        const auto l=layout();
        const auto data=rek_authentic::load(argv[2]);const auto replay=rek_authentic::load_replay(argv[3],data);
        require(replay.checkpoint_sha256==argv[5],"replay belongs to a different checkpoint");
        const auto original=weights(argv[4],argv[5],l);
        if(fit) {
            require(!std::filesystem::exists(argv[6]),"new checkpoint output required");
            const double alpha=calibrate(data,replay);const auto candidate=blend(original,l,alpha);
            verify_weights(original,candidate,l,alpha);publish(argv[6],candidate);
            std::cout<<"{\"mode\":\"cpu_recorded_logit_calibration\",\"alpha\":"<<alpha<<",\"target\":"<<TARGET
                <<",\"input_checkpoint_sha256\":\""<<argv[5]<<"\",\"candidate_checkpoint_sha256\":\""<<digest(candidate)
                <<"\",\"dataset_sha256\":\""<<data.digest<<"\",\"modified_row\":17,\"donor_row\":23,\"row_parameters\":256,"
                "\"other_458752_parameter_bytes_identical\":true,\"value_unchanged\":true,\"cuda_calls\":0,\"grid\":[";
            bool first=true;for(double a:std::array<double,6>{0,.25,.5,.75,1,alpha}) {
                if(!first)std::cout<<',';first=false;std::cout<<"{\"alpha\":"<<a<<",\"prediction\":";
                print_mass(prediction(data,replay,a));std::cout<<'}';
            }
            std::cout<<"],\"limitation\":\"state-dependent row interpolation; recorded-history calibration; native BF16 verification still required\"}\n";return 0;
        }
        size_t consumed=0;const double alpha=std::stod(argv[7],&consumed);
        require(consumed==std::strlen(argv[7]),"invalid alpha suffix");
        const auto candidate_bytes=rek_authentic::read_file(argv[6]);const auto sha=rek_authentic::sha256(candidate_bytes.data(),candidate_bytes.size());
        const auto candidate=weights(argv[6],sha,l);verify_weights(original,candidate,l,alpha);
        Policy old_policy(argv[4],argv[5]),new_policy(argv[6],sha);
        Mass old_mass,new_mass;size_t changed=0,kicks=0,hooks=0;double lp_min=0,lp_max=-INFINITY,max_logit_prediction_error=0;
        for(size_t i=0;i<data.rows.size();++i) {
            const auto& row=data.rows[i];const auto& saved=replay.rows[i];const bool fresh=i==0||row.sequence!=data.rows[i-1].sequence;
            const auto old=old_policy.step(row,fresh),now=new_policy.step(row,fresh);
            require(old.action==row.action&&std::memcmp(old.logits.data(),saved.logits.data(),sizeof(old.logits))==0
                &&std::memcmp(&old.logprob,&saved.old_logprob,sizeof(float))==0,"original native behavior no longer exact");
            for(int k=0;k<=A;++k)if(k!=KICK)require(std::memcmp(&old.logits[k],&now.logits[k],sizeof(float))==0,
                "intervention changed a non17 native logit or value");
            std::array<double,A> before{},after{};for(int k=0;k<A;++k){before[k]=old.logits[k];after[k]=now.logits[k];}
            old_mass.add(before,row);new_mass.add(after,row);changed+=old.action!=now.action;kicks+=now.action==KICK;hooks+=now.action==DONOR;
            lp_min=std::min(lp_min,double(now.logprob));lp_max=std::max(lp_max,double(now.logprob));
            max_logit_prediction_error=std::max(max_logit_prediction_error,std::abs(after[KICK]-((1-alpha)*before[KICK]+alpha*before[DONOR])));
        }
        std::cout<<"{\"mode\":\"counterfactual_fixed_history_native_verification\",\"verification_passed\":true,\"alpha\":"<<alpha
            <<",\"rows\":"<<data.rows.size()<<",\"rounds\":"<<data.sequences.size()<<",\"precision\":\"bf16\",\"seed_per_sequence\":73,"
            "\"original_actions_all34logits_logprobs_bitwise_exact\":true,\"candidate_non17_logits_including_value_bitwise_equal\":true,"
            "\"other_458752_parameter_bytes_identical\":true,\"candidate_samples_all_legal\":true,\"changed_counterfactual_samples\":"<<changed
            <<",\"counterfactual_kick_samples\":"<<kicks<<",\"counterfactual_hook_samples\":"<<hooks
            <<",\"candidate_sampled_logprob_min\":"<<lp_min<<",\"candidate_sampled_logprob_max\":"<<lp_max
            <<",\"max_kick_logit_error_vs_cpu_interpolation\":"<<max_logit_prediction_error<<",\"baseline\":";print_mass(old_mass);
        std::cout<<",\"candidate\":";print_mass(new_mass);
        std::cout<<",\"input_checkpoint_sha256\":\""<<argv[5]<<"\",\"candidate_checkpoint_sha256\":\""<<sha
            <<"\",\"dataset_sha256\":\""<<data.digest<<"\",\"behavior_replay_written\":false,\"training_performed\":false,"
            "\"game_connection\":false,\"limitation\":\"new samples are conditional on original recorded histories; no new behavior trajectory or efficacy claim\"}\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<"decoder_row_exploration: "<<e.what()<<'\n';return 1;}
}
