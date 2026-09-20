#include "authentic_gae.h"
#include <iostream>
#include <functional>
using namespace rek_authentic_gae;
int main() {
    int checks = 0;
    auto near = [&](double a, double b) { if (std::abs(a-b)>1e-6) throw std::runtime_error("GAE reference mismatch"); ++checks; };
    const std::vector<Step> s = {{.2,.9,.8,.3,false},{-.1,.7,.6,.2,false},{1,.8,.5,.1,true}};
    const auto got = compute(s);
    near(got.advantages[2], .9);
    near(got.advantages[1], -.1+.7*.1-.2+.7*.6*.9);
    near(got.advantages[0], .2+.9*.2-.3+.9*.8*got.advantages[1]);
    for (size_t i=0;i<s.size();++i) near(got.returns[i],s[i].old_value+got.advantages[i]);
    auto monte=s; for (auto& x:monte) x.lambda=1;
    const auto mc=compute(monte); near(mc.returns[0],.2+.9*(-.1+.7)); near(mc.returns[1],-.1+.7); near(mc.returns[2],1);
    auto zero_mc = monte; for (auto& x : zero_mc) x.old_value = 0;
    const auto zero = compute(zero_mc);
    for (size_t i = 0; i < zero_mc.size(); ++i) {
        near(zero.returns[i], mc.returns[i]); near(zero.advantages[i], zero.returns[i]);
    }
    // Lambda=1 alone still subtracts the old critic; the zero-baseline control must not.
    near(zero.advantages[0] - mc.advantages[0], s[0].old_value);
    auto td=s; for (auto& x:td) x.lambda=0;
    const auto one=compute(td); near(one.advantages[0],.2+.9*.2-.3); near(one.advantages[1],-.1+.7*.1-.2);
    auto boundary=s; boundary[0].terminal_after=true;
    near(compute(boundary).returns[0],.2);
    auto rejects=[&](const std::function<void()>& f) { bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}if(!rejected)throw std::runtime_error("invalid GAE accepted");++checks; };
    rejects([&]{compute({});});
    rejects([&]{auto a=s;a.back().terminal_after=false;compute(a);});
    rejects([&]{auto a=s;a[0].gamma=-1;compute(a);});
    rejects([&]{auto a=s;a[0].lambda=1.1;compute(a);});
    // Variable elapsed time: discount products preserve elapsed-time composition.
    const double dt1=.013,dt2=.041;
    near(std::pow(.99,dt1/.02)*std::pow(.99,dt2/.02),std::pow(.99,(dt1+dt2)/.02));
    std::cout<<"{\"authentic_gae_checks\":"<<checks<<",\"passed\":true}\n";
}
