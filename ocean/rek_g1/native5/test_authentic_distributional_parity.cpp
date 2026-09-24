#include "authentic_distributional_parity.h"
#include <cstdio>
#include <stdexcept>
#include <functional>
using namespace rek_authentic_parity;
int main() {
    const DistributionalEvidence observed{{.125,.00048828125,.0528736217504,0,0,0,.2},
        3.54386016848e-7,.000488246700609,3.86362154335e-5};
    int checks=0; const auto verify=[&](bool x){if(!x)throw std::runtime_error("distributional gate test failed");++checks;};
    verify(accepts_distributional(observed));verify(!accepts(observed.numerical,true));
    const auto reject=[&](const std::function<void(DistributionalEvidence&)>& edit){auto e=observed;edit(e);verify(!accepts_distributional(e));};
    reject([](auto& e){e.numerical.sequential_logit_error=1e-12;});
    reject([](auto& e){e.numerical.sequential_value_error=1e-12;});
    reject([](auto& e){e.numerical.initial_clipped_fraction=1e-12;});
    reject([](auto& e){e.numerical.batch_ratio_error=.2;});
    reject([](auto& e){e.numerical.clip=0;});
    reject([](auto& e){e.mean_legal_kl=1.0001e-5;});
    reject([](auto& e){e.maximum_legal_kl=1.0001e-3;});
    reject([](auto& e){e.relative_absolute_surrogate_error=1.0001e-3;});
    reject([](auto& e){e.mean_legal_kl=NAN;});reject([](auto& e){e.maximum_legal_kl=INFINITY;});
    reject([](auto& e){e.relative_absolute_surrogate_error=-1;});
    reject([](auto& e){e.numerical.batch_logit_error=NAN;});
    reject([](auto& e){e.maximum_legal_kl=.00726243340089;});
    auto boundary=observed;boundary.mean_legal_kl=kMeanLegalKlLimit;boundary.maximum_legal_kl=kMaximumLegalKlLimit;
    boundary.relative_absolute_surrogate_error=kRelativeSurrogateLimit;verify(accepts_distributional(boundary));
    std::printf("distributional_parity_cpu_checks=%d passed\n",checks);
}
