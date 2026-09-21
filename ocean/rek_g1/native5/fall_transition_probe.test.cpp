#define main fall_transition_probe_entry
#include "fall_transition_probe.cpp"
#undef main
int main() {
  Data d{1,0,{}};
  for(int i=0;i<100;i++)d.rows.push_back({2,i,0,0,{double(i<5)}, {double(i<5),0}});
  Model m;m.name="constant_fixture";m.target=0;m.w={logit(.05)};
  const auto constant=measure(m,d,2,true);
  require(std::abs(constant.brier-.0475)<1e-12,"constant_brier");
  require(std::abs(constant.ap-.05)<1e-12,"constant_average_precision");
  require(std::abs(constant.auc-.5)<1e-12,"constant_auc");
  require(constant.bins.size()==1&&constant.ece<1e-12,"constant_calibration_ties");
  m.columns={0};m.mean={0};m.scale={1};m.w={-5,10};
  const auto ranked=measure(m,d,2,true);
  require(std::abs(ranked.ap-1)<1e-12&&std::abs(ranked.auc-1)<1e-12,"perfect_ranking");
  std::cout<<"fall_transition_probe_metric_tests_passed\n";
}
