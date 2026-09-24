#pragma once
#include "authentic_trajectory.h"

namespace rek_crossfit_baseline {
constexpr size_t HeaderBytes=160, RowBytes=32;
struct Baseline {
    std::string sha256, protocol_sha256;
    std::vector<float> returns, prediction, training_mean, residual;
};
inline void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
inline bool sha_string(const std::string& s){
    return s.size()==64 && std::all_of(s.begin(),s.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');});
}
inline std::vector<float> complete_mc_reference(const rek_authentic::Dataset& data){
    std::vector<float> out(data.rows.size());
    for(const auto& seq:data.sequences){
        double next=0;
        for(size_t i=seq.end;i>seq.begin;--i){
            const auto& row=data.rows[i-1];
            next=double(row.reward)+(row.terminal_after?0:double(row.gamma)*next);
            out[i-1]=float(next);require(std::isfinite(out[i-1]),"nonfinite baseline MC reference");
        }
    }
    return out;
}
#ifdef __CUDACC__
__host__ __device__
#endif
inline float subtract(float mc_return,float state_prediction){return mc_return-state_prediction;}
inline Baseline decode(const std::vector<unsigned char>& bytes,const rek_authentic::Dataset& data,
        const std::string& expected_sha,const std::string& expected_protocol){
    require(sha_string(expected_sha)&&sha_string(expected_protocol),"baseline SHA must be 64 lowercase hex characters");
    require(data.format_version==5,"cross-fitted baseline requires V5 score-delta dataset");
    require(bytes.size()>=HeaderBytes,"short baseline header");
    const auto scalar=[](const unsigned char* p){return rek_authentic::scalar<uint32_t>(p);};
    require(std::memcmp(bytes.data(),"REKSB001",8)==0 && scalar(&bytes[8])==1 && scalar(&bytes[12])==data.rows.size()
        && scalar(&bytes[16])==rek_authentic::OBS && scalar(&bytes[20])==data.sequences.size()
        && scalar(&bytes[24])==RowBytes && scalar(&bytes[28])==0,"baseline schema/shape mismatch");
    require(bytes.size()==HeaderBytes+data.rows.size()*RowBytes,"baseline byte count mismatch");
    Baseline out;out.sha256=rek_authentic::sha256(bytes.data(),bytes.size());
    require(out.sha256==expected_sha,"baseline artifact SHA mismatch");
    require(std::string(reinterpret_cast<const char*>(&bytes[32]),64)==data.digest,"baseline source dataset mismatch");
    out.protocol_sha256=std::string(reinterpret_cast<const char*>(&bytes[96]),64);
    require(out.protocol_sha256==expected_protocol,"baseline frozen protocol mismatch");
    const auto expected_returns=complete_mc_reference(data);
    for(size_t i=0;i<data.rows.size();i++){
        const auto* p=bytes.data()+HeaderBytes+i*RowBytes;const auto& row=data.rows[i];
        require(scalar(p)==i && scalar(p+4)==row.sequence && scalar(p+8)==row.source_seq
            && scalar(p+12)==uint32_t(row.policy_weight==1),"baseline row ordering/identity/eligibility mismatch");
        const float target=rek_authentic::scalar<float>(p+16),pred=rek_authentic::scalar<float>(p+20),
            mean=rek_authentic::scalar<float>(p+24),residual=rek_authentic::scalar<float>(p+28);
        require(std::isfinite(target)&&std::isfinite(pred)&&std::isfinite(mean)&&std::isfinite(residual),"nonfinite baseline row");
        require(target==expected_returns[i],"baseline row return differs from unchanged MC recurrence");
        require(residual==subtract(target,pred),"baseline row residual mismatch");
        out.returns.push_back(target);out.prediction.push_back(pred);out.training_mean.push_back(mean);out.residual.push_back(residual);
    }
    return out;
}
inline Baseline load(const char* path,const rek_authentic::Dataset& data,const std::string& expected_sha,
        const std::string& expected_protocol){return decode(rek_authentic::read_file(path),data,expected_sha,expected_protocol);}
} // namespace rek_crossfit_baseline
