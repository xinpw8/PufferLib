#include "authentic_parity.h"
#include <initializer_list>
#include <iostream>
#include <limits>
#include <stdexcept>
int main() {
    using namespace rek_authentic_parity;
    int checks = 0;
    auto check = [&](bool ok) { if (!ok) throw std::runtime_error("parity acceptance mismatch"); ++checks; };
    Evidence measured{.125, .0625, .0157485825198, 0, 0, 0, .2};
    check(!accepts(measured, false)); check(accepts(measured, true));
    auto e = measured; e.batch_ratio_error = .0200001; check(!accepts(e, true));
    e = measured; e.batch_ratio_error = .02; check(accepts(e, true));
    e = measured; e.sequential_logit_error = 1e-9; check(!accepts(e, true));
    e = measured; e.sequential_value_error = 1e-9; check(!accepts(e, true));
    e = measured; e.initial_clipped_fraction = 1e-9; check(!accepts(e, true));
    e = measured; e.clip = 0; check(!accepts(e, true));
    e = measured; e.batch_ratio_error = std::numeric_limits<double>::quiet_NaN(); check(!accepts(e, true));
    e = measured; e.batch_value_error = std::numeric_limits<double>::infinity(); check(!accepts(e, true));
    e = measured; e.sequential_logit_error = -1; check(!accepts(e, true));
    e = {}; check(accepts(e, false));
    e.batch_ratio_error = .0001; check(accepts(e, false));
    e.batch_ratio_error = .00010001; check(!accepts(e, false));
    std::cout << "{\"authentic_parity_checks\":" << checks << ",\"passed\":true}\n";
}
