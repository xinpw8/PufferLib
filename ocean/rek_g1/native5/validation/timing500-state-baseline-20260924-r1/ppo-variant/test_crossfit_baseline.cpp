#include "crossfit_baseline.h"
#include <iostream>
#include <limits>
using rek_crossfit_baseline::require;
namespace {
int checks=0;
template<class T>void write(std::vector<unsigned char>& b,size_t offset,T v){std::memcpy(b.data()+offset,&v,sizeof(v));}
std::string hash(const std::vector<unsigned char>& b){return rek_authentic::sha256(b.data(),b.size());}
rek_authentic::Dataset fixture(){
    rek_authentic::Dataset d;d.format_version=5;d.digest=std::string(64,'d');d.rows.resize(3);d.sequences.push_back({0,3,9,0});
    for(int i=0;i<3;i++){auto& r=d.rows[i];r.sequence=9;r.source_seq=10+i;r.gamma=.5f;r.lambda=1;r.policy_weight=i<2?1:0;r.terminal_after=i==2;r.reward=i==2?1:0;}
    return d;
}
std::vector<unsigned char> artifact(const rek_authentic::Dataset& d){
    std::vector<unsigned char> b(160+d.rows.size()*32,0);std::memcpy(b.data(),"REKSB001",8);
    for(auto entry:std::vector<std::pair<size_t,uint32_t>>{{8,1},{12,uint32_t(d.rows.size())},{16,223},{20,uint32_t(d.sequences.size())},{24,32}})write(b,entry.first,entry.second);
    std::memcpy(b.data()+32,d.digest.data(),64);const std::string protocol(64,'b');std::memcpy(b.data()+96,protocol.data(),64);
    const auto ret=rek_crossfit_baseline::complete_mc_reference(d);
    for(size_t i=0;i<d.rows.size();i++){
        const size_t p=160+i*32;write(b,p,uint32_t(i));write(b,p+4,d.rows[i].sequence);write(b,p+8,d.rows[i].source_seq);write(b,p+12,uint32_t(d.rows[i].policy_weight==1));
        write(b,p+16,ret[i]);write(b,p+20,.125f);write(b,p+24,.25f);write(b,p+28,ret[i]-.125f);
    }return b;
}
template<class F>void rejects(F fn){bool rejected=false;try{fn();}catch(const std::exception&){rejected=true;}require(rejected,"expected bad baseline rejection");++checks;}
void tests(){
    const auto d=fixture();const auto b=artifact(d);const auto sha=hash(b);const std::string protocol(64,'b');
    const auto good=rek_crossfit_baseline::decode(b,d,sha,protocol);
    require(good.returns==std::vector<float>({.25f,.5f,1.f})&&good.residual==std::vector<float>({.125f,.375f,.875f}),"complete MC/excluded-terminal/subtraction mismatch");++checks;
    const auto ret=rek_crossfit_baseline::complete_mc_reference(d);
    for(size_t i=0;i<ret.size();i++)require(rek_crossfit_baseline::subtract(ret[i],0)==ret[i],"zero baseline changed targets");++checks;
    require(d.rows.back().policy_weight==0 && good.returns.front()==.25f,"actor-zero terminal lost from recurrence");++checks;
    for(size_t offset:{size_t(0),size_t(8),size_t(12),size_t(16),size_t(20),size_t(24),size_t(28),size_t(32),size_t(96),size_t(160),size_t(164),size_t(168),size_t(172)}){
        auto bad=b;bad[offset]^=1;rejects([&]{rek_crossfit_baseline::decode(bad,d,hash(bad),protocol);});
    }
    auto short_file=b;short_file.pop_back();rejects([&]{rek_crossfit_baseline::decode(short_file,d,hash(short_file),protocol);});
    auto long_file=b;long_file.push_back(0);rejects([&]{rek_crossfit_baseline::decode(long_file,d,hash(long_file),protocol);});
    rejects([&]{rek_crossfit_baseline::decode(b,d,std::string(64,'0'),protocol);});
    rejects([&]{rek_crossfit_baseline::decode(b,d,sha,std::string(64,'a'));});
    rejects([&]{rek_crossfit_baseline::decode(b,d,"short",protocol);});
    auto wrong_format=d;wrong_format.format_version=3;rejects([&]{rek_crossfit_baseline::decode(b,wrong_format,sha,protocol);});
    for(size_t offset:{size_t(176),size_t(180),size_t(184),size_t(188)}){
        auto bad=b;write(bad,offset,std::numeric_limits<float>::quiet_NaN());rejects([&]{rek_crossfit_baseline::decode(bad,d,hash(bad),protocol);});
    }
    auto wrong_return=b;write(wrong_return,176,.5f);write(wrong_return,188,.375f);rejects([&]{rek_crossfit_baseline::decode(wrong_return,d,hash(wrong_return),protocol);});
    auto wrong_residual=b;write(wrong_residual,188,1.f);rejects([&]{rek_crossfit_baseline::decode(wrong_residual,d,hash(wrong_residual),protocol);});
    auto wrong_excluded=b;write(wrong_excluded,160+2*32+12,uint32_t(1));rejects([&]{rek_crossfit_baseline::decode(wrong_excluded,d,hash(wrong_excluded),protocol);});
    std::cout<<"{\"cpu_checks_passed\":true,\"checks\":"<<checks<<",\"gpu_used\":false}\n";
}
}
int main(int argc,char** argv){try{
    tests();
    if(argc==6){
        const auto d=rek_authentic::load(argv[1]);require(d.digest==argv[2],"actual dataset SHA mismatch");
        const auto b=rek_crossfit_baseline::load(argv[3],d,argv[4],argv[5]);
        require(b.returns.size()==14957&&d.sequences.size()==5,"actual fixed shape mismatch");
        size_t eligible=0;for(const auto& r:d.rows)eligible+=r.policy_weight==1;
        require(eligible==14952,"actual eligibility mismatch");
        std::cout<<"{\"actual_artifact_validated\":true,\"rows\":"<<b.returns.size()<<",\"eligible_rows\":"<<eligible<<",\"every_mc_return_exact\":true,\"gpu_used\":false}\n";
    }else require(argc==1,"usage: test-crossfit [DATA DATA_SHA BASELINE BASELINE_SHA PROTOCOL_SHA]");
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
