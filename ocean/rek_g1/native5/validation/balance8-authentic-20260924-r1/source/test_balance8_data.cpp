#include "authentic_trajectory.h"
#include "balance8_observation.h"
#include <cstdio>
int main(int argc,char** argv){try{
    using namespace rek_authentic;require(argc==3,"derived and original paths required");
    const auto bytes=read_file(argv[1]);const auto d=decode(bytes),old=load(argv[2]);
    require(d.format_version==4&&old.format_version==3&&d.rows.size()==28715&&d.sequences.size()==5,"derived shape");
    require(d.checkpoint_sha256==old.checkpoint_sha256&&d.derived_teacher_sha256!=d.checkpoint_sha256&&d.original_dataset_sha256==old.digest,"distinct recorded and derived identities");
    for(size_t i=0;i<d.rows.size();i++){
        require(d.rows[i].behavior_seed==old.rows[i].behavior_seed,"seed changed");
        for(int c=0;c<223;c++)if(!rek_balance8::column(c))require(std::memcmp(&d.rows[i].obs[c],&old.rows[i].obs[c],4)==0,"protected column changed");
    }
    for(size_t offset:{size_t(8),size_t(32),DERIVED_HEADER_BYTES+ROW_BYTES+20}){
        auto broken=bytes;broken[offset]^=2;bool rejected=false;try{decode(broken);}catch(...){rejected=true;}require(rejected,"invalid derived mutation accepted");
    }
    auto broken=bytes;std::fill(broken.begin()+384,broken.begin()+416,0);bool rejected=false;try{decode(broken);}catch(...){rejected=true;}require(rejected,"missing teacher accepted");
    std::puts("{\"balance8_dataset_cpu_tests\":\"passed\",\"rows\":28715,\"rounds\":5,\"protected_cells\":6173725,\"mutation_rejections\":4}");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 2;}}
