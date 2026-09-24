#pragma once
#include "authentic_parity.h"
namespace rek_authentic_parity {
constexpr double kMeanLegalKlLimit=1e-5;
constexpr double kMaximumLegalKlLimit=1e-3;
constexpr double kRelativeSurrogateLimit=1e-3;
struct DistributionalEvidence {
    Evidence numerical;
    double mean_legal_kl, maximum_legal_kl, relative_absolute_surrogate_error;
};
inline bool accepts_distributional(const DistributionalEvidence& e) {
    const auto& n=e.numerical;
    for(double x:{n.batch_logit_error,n.batch_value_error,n.batch_ratio_error,
            n.sequential_logit_error,n.sequential_value_error,n.initial_clipped_fraction,n.clip,
            e.mean_legal_kl,e.maximum_legal_kl,e.relative_absolute_surrogate_error})
        if(!std::isfinite(x)||x<0) return false;
    return n.clip>0 && n.sequential_logit_error==0 && n.sequential_value_error==0
        && n.initial_clipped_fraction==0 && n.batch_ratio_error<n.clip
        && e.mean_legal_kl<=kMeanLegalKlLimit && e.maximum_legal_kl<=kMaximumLegalKlLimit
        && e.relative_absolute_surrogate_error<=kRelativeSurrogateLimit;
}
}
