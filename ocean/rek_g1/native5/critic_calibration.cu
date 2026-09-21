// Offline critic experiment. No environment, optimizer, or actor mutation.
#include "puffer5_bc_core.cuh"
#include "authentic_trajectory.h"
#include "device_storage.cuh"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {
using rek_authentic::require;
constexpr int O=223, A=33, H=256, L=2;
struct Layout { size_t count=0, value_begin=0, value_end=0; };

// Derive the location from the actual pinned architecture registration. This
// allocates CPU metadata only; it never calls alloc_create or a CUDA API.
Layout value_layout() {
    Arch arch=build_arch("rek_native5",O,H,L,A,false,128);
    Allocator params{};
    Weights w=weights_create(&arch,&params);
    auto* decoder=static_cast<DecoderWeights*>(w.decoder);
    auto* network=static_cast<MinGRUWeights*>(w.network);
    require(params.num_regs==4 && !decoder->continuous, "unexpected discrete architecture registration");
    const long expected[4][2]={{H,O},{A+1,H},{3*H,H},{3*H,H}};
    Layout layout; int found=0;
    for(int i=0;i<params.num_regs;++i) {
        const auto& entry=params.regs[i];
        require(ndim(entry.shape)==2 && entry.shape[0]==expected[i][0] &&
            entry.shape[1]==expected[i][1] && entry.elem_size==sizeof(precision_t),
            "unexpected registered weight shape");
        require((layout.count*sizeof(precision_t))%16==0, "checkpoint has unexpected allocator padding");
        if(entry.data_ptr==reinterpret_cast<void**>(&decoder->weight.data)) {
            require(i==1 && decoder->weight.shape[0]==A+1 && decoder->weight.shape[1]==H,
                "decoder registration order changed");
            layout.value_begin=layout.count+size_t(A)*H;
            layout.value_end=layout.value_begin+H;
            ++found;
        }
        layout.count+=size_t(numel(entry.shape));
    }
    require(found==1 && layout.count==size_t(params.total_elems) &&
        params.total_bytes==long(layout.count*sizeof(precision_t)), "flat checkpoint layout mismatch");
    free(network->weights); free(w.network); free(w.decoder); free(w.encoder); free(params.regs);
    return layout;
}

void verify_value_only(const std::vector<float>& before,const std::vector<float>& after,Layout l) {
    require(before.size()==l.count && after.size()==l.count, "checkpoint shape mismatch");
    require(std::memcmp(before.data(),after.data(),l.value_begin*sizeof(float))==0 &&
        std::memcmp(before.data()+l.value_end,after.data()+l.value_end,
            (l.count-l.value_end)*sizeof(float))==0, "non-value parameter changed");
    for(float v:after) require(std::isfinite(v), "nonfinite calibrated weight");
}

double fixed_scale_argument(const std::string& argument) {
    size_t consumed=0;
    const double scale=std::stod(argument,&consumed);
    require(consumed==argument.size() && std::isfinite(scale) && scale>=0,
        "fixed scale must be a finite nonnegative number");
    return scale;
}
void verify_expected_sha(const std::string& digest) {
    require(digest.size()==64 && std::all_of(digest.begin(),digest.end(),[](char c) {
        return (c>='0' && c<='9') || (c>='a' && c<='f');
    }), "expected SHA must contain 64 lowercase hexadecimal characters");
}
std::vector<float> fixed_scale_weights(const std::vector<float>& original,Layout layout,double scale) {
    require(std::isfinite(scale) && scale>=0, "fixed scale must be finite and nonnegative");
    require(original.size()==layout.count, "checkpoint shape mismatch");
    for(float v:original) require(std::isfinite(v), "nonfinite input checkpoint");
    auto changed=original;
    for(size_t i=layout.value_begin;i<layout.value_end;++i)
        changed[i]=float(double(original[i])*scale);
    verify_value_only(original,changed,layout);
    return changed;
}
void fixed_scale_checkpoint(const char* input,const char* expected_sha,const char* scale_text,const char* output) {
    const double scale=fixed_scale_argument(scale_text);
    verify_expected_sha(expected_sha);
    require(!std::filesystem::exists(output), "output checkpoint already exists");
    const auto layout=value_layout();
    const auto bytes=rek_authentic::read_file(input);
    require(bytes.size()==layout.count*sizeof(float), "checkpoint size differs from registered architecture");
    const auto digest=rek_authentic::sha256(bytes.data(),bytes.size());
    require(digest==expected_sha, "checkpoint binding differs from supplied SHA");
    std::vector<float> weights(layout.count);
    std::memcpy(weights.data(),bytes.data(),bytes.size());
    const auto changed=fixed_scale_weights(weights,layout,scale);
    const auto changed_digest=rek_authentic::sha256(changed.data(),bytes.size());
    FILE* out=fopen(output,"wbx");require(out!=nullptr,"exclusive checkpoint create failed");
    const bool wrote=fwrite(changed.data(),sizeof(float),changed.size(),out)==changed.size();
    const int closed=fclose(out);require(wrote && closed==0,"fixed-scale checkpoint write failed");
    const auto saved=rek_authentic::read_file(output);
    require(saved.size()==bytes.size() && std::memcmp(saved.data(),changed.data(),saved.size())==0,
        "fixed-scale checkpoint readback failed");
    std::cout<<"{\"mode\":\"fixed_value_head_scale\",\"scale\":"<<scale
        <<",\"input_checkpoint_sha256\":\""<<digest<<"\",\"output_checkpoint_sha256\":\""<<changed_digest
        <<"\",\"parameter_count\":"<<layout.count<<",\"value_begin\":"<<layout.value_begin
        <<",\"value_end_exclusive\":"<<layout.value_end<<",\"outside_value_parameters_bitwise_equal\":true,"
        "\"bias_present\":false,\"cuda_calls\":0}\n";
}

struct Step { double reward, gamma, value; int terminal; };
struct Span { int begin, end; };
struct Fit {
    double scale=0, affine_a=0, affine_b=0, train_mean=0;
    int train_count=0, affine_available=0, error=0;
};
struct Metrics {
    int count=0;
    double target_mean=0, raw_mean=0, target_min=0, target_max=0;
    // Unchanged, through-origin scalar, train-mean constant, diagnostic affine,
    // and optional actual native replay of the transformed checkpoint.
    double mse[5]{}, bias[5]{};
};
__global__ void complete_returns(const Step* rows,const Span* spans,int count,double* target) {
    int s=blockIdx.x*blockDim.x+threadIdx.x;
    if(s>=count) return;
    double next=0;
    for(int i=spans[s].end-1;i>=spans[s].begin;--i) {
        next=rows[i].reward+(rows[i].terminal?0:rows[i].gamma*next);
        target[i]=next;
    }
}
__global__ void fit_training(const Step* rows,const double* target,const Span* spans,Fit* out) {
    if(blockIdx.x || threadIdx.x) return;
    Fit f{}; double sx=0,sy=0,sxx=0,sxy=0;
    // Explicit sequence split, independent of the all-zero policy-training split.
    for(int s=0;s<3;s+=2) for(int i=spans[s].begin;i<spans[s].end;++i) {
        double x=rows[i].value,y=target[i];
        sx+=x;sy+=y;sxx+=x*x;sxy+=x*y;++f.train_count;
    }
    if(!f.train_count || !(sxx>0) || !isfinite(sxx) || !isfinite(sxy)) { f.error=1;*out=f;return; }
    f.scale=sxy/sxx;f.train_mean=sy/f.train_count;
    const double mean_x=sx/f.train_count;
    double xx=0,xy=0;
    for(int s=0;s<3;s+=2) for(int i=spans[s].begin;i<spans[s].end;++i) {
        const double x=rows[i].value-mean_x,y=target[i]-f.train_mean;
        xx+=x*x;xy+=x*y;
    }
    if(xx>0 && isfinite(xx) && isfinite(xy)) {
        f.affine_a=xy/xx;f.affine_b=f.train_mean-f.affine_a*mean_x;
        f.affine_available=isfinite(f.affine_a)&&isfinite(f.affine_b);
    }
    if(!isfinite(f.scale)||!isfinite(f.train_mean)) f.error=2;
    *out=f;
}
__global__ void score_splits(const Step* rows,const double* target,const Span* spans,
        const Fit* fit,const float* realized,Metrics* result) {
    const int group=blockIdx.x;
    if(threadIdx.x || group>=3) return;
    Metrics m{};m.target_min=INFINITY;m.target_max=-INFINITY;
    const Fit f=*fit;
    for(int s=0;s<3;++s) {
        if((group==0 && s==1)||(group==1 && s!=1)) continue;
        for(int i=spans[s].begin;i<spans[s].end;++i) {
            double x=rows[i].value,y=target[i];
            const double pred[5]={x,f.scale*x,f.train_mean,f.affine_a*x+f.affine_b,realized?double(realized[i]):0};
            ++m.count;m.target_mean+=y;m.raw_mean+=x;
            m.target_min=fmin(m.target_min,y);m.target_max=fmax(m.target_max,y);
            for(int j=0;j<5;++j) { double e=pred[j]-y;m.mse[j]+=e*e;m.bias[j]+=e; }
        }
    }
    m.target_mean/=m.count;m.raw_mean/=m.count;
    for(int j=0;j<5;++j) { m.mse[j]/=m.count;m.bias[j]/=m.count; }
    result[group]=m;
}
__global__ void scale_value(float* weights,size_t begin,size_t end,const Fit* fit) {
    const size_t i=begin+size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<end) weights[i]=float(double(weights[i])*fit->scale);
}
struct Result { Fit fit; std::array<Metrics,3> metrics; std::vector<double> target; };
struct Calculation {
    rek5::DeviceStorage storage;
    Step* rows=nullptr;Span* spans=nullptr;double* target=nullptr;Fit* fit=nullptr;Metrics* metrics=nullptr;
    Calculation(const std::vector<Step>& r,const std::vector<Span>& s,const std::vector<float>& realized={}) {
        require(s.size()==3 && !r.empty(), "calibration requires exactly three complete sequences");
        rows=storage.upload(r);spans=storage.upload(s);target=storage.alloc<double>(r.size());
        fit=storage.alloc<Fit>(1);metrics=storage.alloc<Metrics>(3);
        float* actual=realized.empty()?nullptr:storage.upload(realized);
        complete_returns<<<1,32>>>(rows,spans,3,target);
        fit_training<<<1,1>>>(rows,target,spans,fit);
        score_splits<<<3,1>>>(rows,target,spans,fit,actual,metrics);
        rek5::cuda_check(cudaGetLastError());
    }
    Result download(size_t n) {
        Result r;r.target.resize(n);
        rek5::cuda_check(cudaMemcpy(&r.fit,fit,sizeof(Fit),cudaMemcpyDeviceToHost));
        rek5::cuda_check(cudaMemcpy(r.metrics.data(),metrics,3*sizeof(Metrics),cudaMemcpyDeviceToHost));
        rek5::cuda_check(cudaMemcpy(r.target.data(),target,n*sizeof(double),cudaMemcpyDeviceToHost));
        require(r.fit.error==0, "CUDA calibration fit is singular or nonfinite");
        for(double v:r.target) require(std::isfinite(v), "nonfinite complete return");
        return r;
    }
};
std::vector<Step> steps(const rek_authentic::Dataset& d,const rek_authentic::Replay& replay) {
    std::vector<Step> out;
    for(size_t i=0;i<d.rows.size();++i) {
        const auto& r=d.rows[i];
        out.push_back({double(r.reward),double(r.gamma),double(replay.rows[i].old_value),int(r.terminal_after)});
    }
    return out;
}
std::vector<Span> spans(const rek_authentic::Dataset& d) {
    require(d.sequences.size()==3, "expected iteration2 three-sequence dataset");
    std::vector<Span> out;
    for(size_t i=0;i<3;++i) {
        const auto& s=d.sequences[i];
        require(s.sequence==i, "calibration split requires sequence IDs 0,1,2 in order");
        out.push_back({int(s.begin),int(s.end)});
    }
    return out;
}
void print_result(const Result& r,bool realized) {
    const auto& f=r.fit;
    std::cout<<"\"fit\":{\"kind\":\"through_origin\",\"train_sequences\":[0,2],\"heldout_sequences\":[1],"
        "\"row_weighting\":\"uniform_all_value_rows\",\"count\":"<<f.train_count<<",\"scale\":"<<f.scale
        <<",\"train_mean\":"<<f.train_mean<<",\"diagnostic_affine_available\":"<<(f.affine_available?"true":"false")
        <<",\"diagnostic_affine_a\":"<<f.affine_a<<",\"diagnostic_affine_b\":"<<f.affine_b<<"},\"metrics\":{";
    const char* names[]={"train","heldout","all"};
    const char* predictors[]={"raw_value","ideal_scaled_value","train_mean_constant","diagnostic_affine","actual_scaled_checkpoint"};
    for(int group=0;group<3;++group) {
        if(group) std::cout<<',';
        const auto& m=r.metrics[group];
        std::cout<<'"'<<names[group]<<"\":{\"count\":"<<m.count<<",\"target_mean\":"<<m.target_mean
            <<",\"raw_mean\":"<<m.raw_mean<<",\"target_min\":"<<m.target_min<<",\"target_max\":"<<m.target_max;
        for(int j=0;j<5;++j) {
            if((j==3&&!f.affine_available)||(j==4&&!realized)) continue;
            require(std::isfinite(m.mse[j])&&std::isfinite(m.bias[j]), "nonfinite metric");
            std::cout<<",\""<<predictors[j]<<"\":{\"mse\":"<<m.mse[j]<<",\"bias\":"<<m.bias[j]<<'}';
        }
        std::cout<<'}';
    }
    std::cout<<'}';
}
void cpu_self_test() {
    const auto l=value_layout();
    require(l.count==459008 && l.value_begin==65536 && l.value_end==65792, "pinned layout test failed");
    std::vector<float> a(l.count),b;
    for(size_t i=0;i<a.size();++i) a[i]=float(int(i%257)-128)/128;
    b=a;b[l.value_begin]=123;verify_value_only(a,b,l);
    for(size_t i:std::array<size_t,4>{0,l.value_begin-1,l.value_end,l.count-1}) {
        b=a;b[i]+=1;bool rejected=false;
        try { verify_value_only(a,b,l); } catch(const std::exception&) { rejected=true; }
        require(rejected, "non-value mutation was not rejected");
    }
    // Test the same host-only transform used by --fixed-scale. Signed zero in
    // the untouched actor range must survive byte-for-byte as well.
    a[0]=-0.0f;
    for(double scale:std::array<double,4>{0,.01,1,2}) {
        b=fixed_scale_weights(a,l,scale);
        verify_value_only(a,b,l);
        for(size_t i=l.value_begin;i<l.value_end;++i)
            require(b[i]==float(double(a[i])*scale), "fixed value scale test failed");
        require(std::memcmp(a.data(),b.data(),l.value_begin*sizeof(float))==0,
            "fixed scale changed actor bytes");
    }
    for(double scale:std::array<double,3>{-1,INFINITY,NAN}) {
        bool rejected=false;
        try { fixed_scale_weights(a,l,scale); } catch(const std::exception&) { rejected=true; }
        require(rejected, "invalid fixed scale was not rejected");
    }
    for(const char* scale:{"-0.01","nan","inf","1x","","1e999"}) {
        bool rejected=false;
        try { fixed_scale_argument(scale); } catch(const std::exception&) { rejected=true; }
        require(rejected, "invalid fixed scale argument was not rejected");
    }
    require(fixed_scale_argument("0.01")==.01, "fixed scale argument test failed");
    verify_expected_sha(std::string(64,'a'));
    for(const auto& digest:std::array<std::string,3>{std::string(63,'a'),std::string(65,'a'),std::string(64,'g')}) {
        bool rejected=false;
        try { verify_expected_sha(digest); } catch(const std::exception&) { rejected=true; }
        require(rejected, "invalid expected SHA was not rejected");
    }
    b=a;b[l.value_begin]=std::numeric_limits<float>::max();
    bool overflow_rejected=false;
    try { fixed_scale_weights(b,l,2); } catch(const std::exception&) { overflow_rejected=true; }
    require(overflow_rejected, "nonfinite scaled weight was not rejected");
    b=a;b[0]=NAN;
    bool input_rejected=false;
    try { fixed_scale_weights(b,l,.01); } catch(const std::exception&) { input_rejected=true; }
    require(input_rejected, "nonfinite input weight was not rejected");
    std::cout<<"{\"cpu_self_test\":\"passed\",\"parameters\":"<<l.count<<",\"value_begin\":"<<l.value_begin
        <<",\"value_end_exclusive\":"<<l.value_end<<",\"bias_present\":false,\"cuda_calls\":0,"
        "\"fixed_scale_and_actor_preservation\":true,\"invalid_scale_sha_and_nonfinite_rejected\":true}\n";
}
void gpu_self_test() {
    // Closed-form targets [3,5], [-1,-2], [7,9]. Held-out values deliberately
    // conflict with the training relation y=2*x+1 and must not enter fitting.
    std::vector<Step> rows={{.5,.5,1,0},{5,.5,2,1},{0,.5,100,0},{-2,.5,200,1},
        {2.5,.5,3,0},{9,.5,4,1}};
    std::vector<Span> bounds={{0,2},{2,4},{4,6}};
    Calculation c(rows,bounds);const auto r=c.download(rows.size());
    const double expected[]={3,5,-1,-2,7,9};
    for(size_t i=0;i<rows.size();++i) require(r.target[i]==expected[i], "CUDA complete-return fixture failed");
    require(std::abs(r.fit.scale-7.0/3)<1e-12 && r.fit.train_mean==6 && r.fit.train_count==4 &&
        r.fit.affine_available && r.fit.affine_a==2 && r.fit.affine_b==1, "CUDA training-only fit fixture failed");
    require(r.metrics[0].mse[3]==0 && r.metrics[0].mse[2]==5 && r.metrics[1].count==2,
        "CUDA calibration metric fixture failed");
    const auto l=value_layout();std::vector<float> original(l.count,1.25f),changed(l.count);
    auto* device=c.storage.upload(original);
    scale_value<<<1,H>>>(device,l.value_begin,l.value_end,c.fit);
    rek5::cuda_check(cudaGetLastError());
    rek5::cuda_check(cudaMemcpy(changed.data(),device,l.count*sizeof(float),cudaMemcpyDeviceToHost));
    verify_value_only(original,changed,l);
    require(changed[l.value_begin]==float(1.25*r.fit.scale), "CUDA value scale fixture failed");
    for(auto& row:rows) row.value=0;
    Calculation singular(rows,bounds);bool rejected=false;
    try { singular.download(rows.size()); } catch(const std::exception&) { rejected=true; }
    require(rejected, "singular CUDA scalar fit was not rejected");
    std::cout<<"{\"gpu_self_test\":\"passed\",\"recurrence_fit_holdout_metrics_and_value_only\":true}\n";
}
} // namespace

int main(int argc,char** argv) {
    try {
        std::cout<<std::setprecision(17);
        if(argc==2 && std::string(argv[1])=="--cpu-self-test") { cpu_self_test();return 0; }
        if(argc==2 && std::string(argv[1])=="--gpu-self-test") { gpu_self_test();return 0; }
        if(argc==6 && std::string(argv[1])=="--fixed-scale") {
            fixed_scale_checkpoint(argv[2],argv[3],argv[4],argv[5]);return 0;
        }
        const bool verify=argc==5 && std::string(argv[1])=="--verify";
        require(verify || argc==6,"usage: critic-calibration DATA OLD_REPLAY CHECKPOINT EXPECTED_SHA NEW_CHECKPOINT | --fixed-scale CHECKPOINT SHA SCALE NEW_CHECKPOINT | --verify DATA OLD_REPLAY NEW_REPLAY | --cpu-self-test | --gpu-self-test");
        const char* data_path=argv[verify?2:1];const char* replay_path=argv[verify?3:2];
        const auto data=rek_authentic::load(data_path);
        const auto old_bytes=rek_authentic::read_file(replay_path);
        const auto replay=rek_authentic::decode_replay(old_bytes,data);
        const auto bounds=spans(data);const auto rows=steps(data,replay);
        if(verify) {
            const auto fresh_bytes=rek_authentic::read_file(argv[4]);
            const auto fresh=rek_authentic::decode_replay(fresh_bytes,data);
            require(fresh.checkpoint_sha256!=replay.checkpoint_sha256, "verification requires the new calibrated checkpoint replay");
            std::vector<float> actual;actual.reserve(rows.size());
            for(size_t i=0;i<rows.size();++i) {
                const auto& a=replay.rows[i];const auto& b=fresh.rows[i];
                require(a.action==b.action && std::memcmp(&a.old_logprob,&b.old_logprob,sizeof(float))==0 &&
                    std::memcmp(a.logits.data(),b.logits.data(),A*sizeof(float))==0,
                    "calibration changed sampled action, chosen logprob, or actor logits");
                actual.push_back(b.old_value);
            }
            Calculation c(rows,bounds,actual);const auto result=c.download(rows.size());
            std::cout<<"{\"mode\":\"native_replay_verification\",\"dataset_sha256\":\""<<data.digest
                <<"\",\"old_checkpoint_sha256\":\""<<replay.checkpoint_sha256<<"\",\"new_checkpoint_sha256\":\""
                <<fresh.checkpoint_sha256<<"\",\"old_replay_sha256\":\""<<rek_authentic::sha256(old_bytes.data(),old_bytes.size())
                <<"\",\"new_replay_sha256\":\""<<rek_authentic::sha256(fresh_bytes.data(),fresh_bytes.size())
                <<"\",\"actor_logits_logprobs_actions_bitwise_equal_rows\":"<<rows.size()<<',';
            print_result(result,true);std::cout<<"}\n";return 0;
        }
        const auto layout=value_layout();
        require(!std::filesystem::exists(argv[5]), "output checkpoint already exists");
        const auto bytes=rek_authentic::read_file(argv[3]);
        require(bytes.size()==layout.count*sizeof(float), "checkpoint size differs from registered architecture");
        const auto digest=rek_authentic::sha256(bytes.data(),bytes.size());
        require(digest==argv[4] && digest==replay.checkpoint_sha256, "checkpoint binding differs from supplied SHA or replay");
        std::vector<float> weights(layout.count),changed(layout.count);
        std::memcpy(weights.data(),bytes.data(),bytes.size());
        for(float v:weights) require(std::isfinite(v), "nonfinite input checkpoint");
        Calculation c(rows,bounds);const auto result=c.download(rows.size());
        auto* device=c.storage.upload(weights);
        scale_value<<<1,H>>>(device,layout.value_begin,layout.value_end,c.fit);
        rek5::cuda_check(cudaGetLastError());
        rek5::cuda_check(cudaMemcpy(changed.data(),device,bytes.size(),cudaMemcpyDeviceToHost));
        verify_value_only(weights,changed,layout);
        const auto changed_digest=rek_authentic::sha256(changed.data(),bytes.size());
        FILE* out=fopen(argv[5],"wbx");require(out!=nullptr,"exclusive checkpoint create failed");
        const bool wrote=fwrite(changed.data(),sizeof(float),changed.size(),out)==changed.size();
        const int closed=fclose(out);require(wrote && closed==0,"calibrated checkpoint write failed");
        const auto saved=rek_authentic::read_file(argv[5]);
        require(saved.size()==bytes.size() && std::memcmp(saved.data(),changed.data(),saved.size())==0,
            "calibrated checkpoint readback failed");
        std::cout<<"{\"mode\":\"scalar_value_head_calibration\",\"dataset_sha256\":\""<<data.digest
            <<"\",\"replay_sha256\":\""<<rek_authentic::sha256(old_bytes.data(),old_bytes.size())
            <<"\",\"input_checkpoint_sha256\":\""<<digest<<"\",\"output_checkpoint_sha256\":\""<<changed_digest
            <<"\",\"parameter_count\":"<<layout.count<<",\"value_begin\":"<<layout.value_begin
            <<",\"value_end_exclusive\":"<<layout.value_end<<",\"outside_value_parameters_bitwise_equal\":true,"
            "\"native_actor_replay_required_before_use\":true,\"affine_intercept_representable\":false,";
        print_result(result,false);std::cout<<"}\n";return 0;
    } catch(const std::exception& e) {
        std::cerr<<"critic calibration: "<<e.what()<<'\n';return 1;
    }
}
