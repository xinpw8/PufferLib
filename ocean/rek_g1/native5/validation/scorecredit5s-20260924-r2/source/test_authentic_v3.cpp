#include "authentic_trajectory.h"
#include <cstdio>
using namespace rek_authentic;
int main(int argc,char** argv) {
    try {
        require(argc==2,"Usage: test-authentic-v3 DATA");
        const auto bytes=read_file(argv[1]); const auto d=decode(bytes);
        require(d.format_version==3 && d.sequences.size()==5,"expected five v3 sequences");
        size_t checks=1;
        for(size_t i=0;i<d.sequences.size();++i) {
            const auto& s=d.sequences[i];
            for(size_t j=s.begin;j<s.end;++j) require(d.rows[j].behavior_seed==901+i,"unexpected behavior seed");
            ++checks;
        }
        auto edited=bytes;
        edited[IDENTITY_HEADER_BYTES+ROW_BYTES+20]^=1;
        bool failed=false;try{decode(edited);}catch(const std::exception&){failed=true;}
        require(failed,"within-round seed mutation accepted");++checks;
        edited=bytes;edited[32]=2;failed=false;try{decode(edited);}catch(const std::exception&){failed=true;}
        require(failed,"nonbinary feature mask accepted");++checks;
        std::printf("authentic_v3_cpu_checks=%zu rows=%zu rounds=%zu sha256=%s passed\n",checks,d.rows.size(),d.sequences.size(),d.digest.c_str());
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"authentic_v3_test: %s\n",e.what());return 2;}
}
