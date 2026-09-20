#include "authentic_trajectory.h"
#include <cstdio>
#include <functional>
using namespace rek_authentic;
int main(int argc,char** argv) {
    try {
        if(argc==2) {
            const auto d=load(argv[1]); size_t applied=0; for(const auto& r:d.rows) applied+=r.applied;
            std::printf("authentic_dataset_cpu_pass rows=%zu rounds=%zu applied=%zu rejected=%zu sha256=%s\n",
                d.rows.size(),d.sequences.size(),applied,d.rows.size()-applied,d.digest.c_str()); return 0;
        }
        Dataset base; base.feature_mask.fill(1);
        for(int s=0;s<2;++s) for(int t=0;t<2;++t) {
            Row r; r.sequence=s; r.reset=t==0; r.action=2; r.support.fill(1);
            r.applied=t==0; r.policy_weight=r.applied; r.value_weight=1;
            r.time=t*.02; r.next_time=(t+1)*.02; r.dt=.02; r.gamma=.999f; r.lambda=.995f;
            r.source_seq=2+t; r.next_source_seq=3+t; r.terminal_after=t==1;
            r.outcome=t==1?1.f:0.f; r.reward=r.outcome; base.rows.push_back(r);
        }
        validate(base,2); int checks=1;
        const auto reject=[&](const std::function<void(Dataset&)>& edit) {
            auto d=base; edit(d); bool failed=false; try{validate(d,2);}catch(const std::exception&){failed=true;}
            require(failed,"malformed dataset was accepted"); ++checks;
        };
        reject([](auto& d){d.feature_mask[180]=0;});
        reject([](auto& d){d.rows[0].support[2]=0;});
        reject([](auto& d){d.rows[1].reset=1;});
        reject([](auto& d){d.rows[2].sequence=0;});
        reject([](auto& d){d.rows[1].time=.03;});
        reject([](auto& d){d.rows[0].applied=0;d.rows[0].policy_weight=0;});
        reject([](auto& d){d.rows[1].policy_weight=1;});
        reject([](auto& d){d.rows[1].reward=0;});
        reject([](auto& d){d.rows[1].terminal_after=0;});
        reject([](auto& d){d.rows[0].obs[1]=NAN;});
        reject([](auto& d){d.rows[0].next_own_points=1;});
        reject([](auto& d){d.rows[0].gamma=0;});
        reject([](auto& d){d.rows[3].terminal_after=0;});
        reject([](auto& d){d.rows[0].split=1;});
        std::printf("authentic_dataset_cpu_checks=%d passed\n",checks); return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"authentic_dataset_test: %s\n",e.what()); return 2; }
}
