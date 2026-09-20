#pragma once
#include <cmath>
#include <initializer_list>
namespace rek_authentic_parity {
struct Evidence {
    double batch_logit_error, batch_value_error, batch_ratio_error;
    double sequential_logit_error, sequential_value_error;
    double initial_clipped_fraction, clip;
};
inline bool accepts(const Evidence& e, bool allow_bf16_batch) {
    for (double x : {e.batch_logit_error, e.batch_value_error, e.batch_ratio_error,
            e.sequential_logit_error, e.sequential_value_error, e.initial_clipped_fraction, e.clip})
        if (!std::isfinite(x) || x < 0) return false;
    if (!allow_bf16_batch)
        return e.batch_logit_error <= 1e-4 && e.batch_value_error <= 1e-4 && e.batch_ratio_error <= 1e-4;
    return e.clip > 0 && e.sequential_logit_error == 0 && e.sequential_value_error == 0
        && e.batch_ratio_error <= .1 * e.clip && e.initial_clipped_fraction == 0;
}
} // namespace rek_authentic_parity
