// Native CPU diagnostic fit. No policy training or game/runtime connection.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct Row { int split, group, capture; float t; std::vector<double> x; std::array<double,2> y; };
struct Data { int features, state_features; std::vector<Row> rows; };
static void require(bool ok,const char* why) { if(!ok) throw std::runtime_error(why); }
static double sigmoid(double z) { return z>=0 ? 1/(1+std::exp(-z)) : std::exp(z)/(1+std::exp(z)); }
static double logit(double p) { p=std::clamp(p,1e-9,1-1e-9); return std::log(p/(1-p)); }
static double logloss(double y,double p) { p=std::clamp(p,1e-12,1-1e-12);return -y*std::log(p)-(1-y)*std::log1p(-p); }
static Data load(const char* file) {
  std::ifstream f(file,std::ios::binary);require(bool(f),"data_open_failed");char magic[8];f.read(magic,8);
  require(std::string(magic,7)=="REKFAL1","bad_magic");uint32_t h[4];f.read(reinterpret_cast<char*>(h),16);
  require(h[0]==1 && h[1]>0 && h[1]<256 && h[2]<h[1] && h[3]>0 && h[3]<10000000,"bad_header");
  Data d{int(h[1]),int(h[2]),{}};d.rows.reserve(h[3]);std::vector<float> b(4+h[1]+2);
  for(uint32_t i=0;i<h[3];i++) {
    f.read(reinterpret_cast<char*>(b.data()),b.size()*4);require(bool(f),"truncated_data");
    for(auto x:b)require(std::isfinite(x),"nonfinite_data");
    require(b[0]>=0&&b[0]<=2&&b[0]==int(b[0]),"invalid_split");
    Row r{int(b[0]),int(b[1]),int(b[2]),b[3],{}, {b[4+h[1]],b[5+h[1]]}};
    require((r.y[0]==0||r.y[0]==1)&&(r.y[1]==0||r.y[1]==1),"invalid_label");
    r.x.assign(b.begin()+4,b.begin()+4+h[1]);d.rows.push_back(std::move(r));
  }
  require(f.peek()==EOF,"trailing_data");
  std::unordered_map<int,int> groups;
  for(const auto&r:d.rows){auto [it,inserted]=groups.emplace(r.group,r.split);require(inserted||it->second==r.split,"process_split_leakage");}
  return d;
}
struct Model { std::string name; std::vector<int> columns; std::vector<double> mean,scale,w; double offset=0; int target; };
static double raw_score(const Model& m,const Row& r) {
  double z=m.w[0];for(size_t j=0;j<m.columns.size();j++) z+=m.w[j+1]*(r.x[m.columns[j]]-m.mean[j])/m.scale[j];return z;
}
static Model fit(const Data& d,std::string name,int target) {
  Model m;m.name=name;m.target=target;
  for(int j=0;j<d.features;j++)if(name=="state_action" || (name=="state_only"&&j<d.state_features) ||
    (name=="action_only"&&j>=d.state_features))m.columns.push_back(j);
  const size_t p=m.columns.size();m.mean.assign(p,0);m.scale.assign(p,0);int n=0,positive=0;
  for(const auto&r:d.rows)if(r.split==0){n++;positive+=r.y[target];for(size_t j=0;j<p;j++)m.mean[j]+=r.x[m.columns[j]];}
  require(n>0&&positive>0&&positive<n,"degenerate_training_labels");
  for(auto&x:m.mean)x/=n;
  for(const auto&r:d.rows)if(r.split==0)for(size_t j=0;j<p;j++)m.scale[j]+=std::pow(r.x[m.columns[j]]-m.mean[j],2)/n;
  for(auto&x:m.scale)x=std::max(std::sqrt(x),1e-6);
  m.w.assign(p+1,0);m.w[0]=logit(double(positive)/n);
  if(p==0)return m;
  std::vector<std::vector<double>> xs;std::vector<double> ys;
  for(const auto&r:d.rows)if(r.split==0){std::vector<double>x{1};for(size_t j=0;j<p;j++)x.push_back((r.x[m.columns[j]]-m.mean[j])/m.scale[j]);xs.push_back(std::move(x));ys.push_back(r.y[target]);}
  std::vector<double> first(p+1,0),second(p+1,0),gradient(p+1,0);
  // Fixed settings, declared before the held-out results. Natural prevalence.
  constexpr double lambda=.001,learning_rate=.03;constexpr int iterations=600;
  for(int iter=1;iter<=iterations;iter++) {
    std::fill(gradient.begin(),gradient.end(),0);
    for(size_t i=0;i<xs.size();i++){
      const auto&x=xs[i];const double error=sigmoid(std::inner_product(x.begin(),x.end(),m.w.begin(),0.0))-ys[i];
      for(size_t j=0;j<=p;j++)gradient[j]+=error*x[j]/n;
    }
    for(size_t j=0;j<=p;j++){
      if(j)gradient[j]+=lambda*m.w[j];
      first[j]=.9*first[j]+.1*gradient[j];second[j]=.999*second[j]+.001*gradient[j]*gradient[j];
      m.w[j]-=learning_rate*(first[j]/(1-std::pow(.9,iter)))/(std::sqrt(second[j]/(1-std::pow(.999,iter)))+1e-8);
    }
  }
  return m;
}
static void calibrate(Model& m,const Data& d) {
  // Single intercept correction fit on calibration processes, never final test.
  double low=-15,high=15;int count=0,positive=0;
  for(const auto&r:d.rows)if(r.split==1){count++;positive+=r.y[m.target];}
  require(count>0&&positive>0&&positive<count,"degenerate_calibration_labels");
  for(int i=0;i<80;i++){
    const double mid=(low+high)/2;double predicted=0;
    for(const auto&r:d.rows)if(r.split==1)predicted+=sigmoid(raw_score(m,r)+mid);
    if(predicted>positive)high=mid;else low=mid;
  }
  m.offset=(low+high)/2;
}
struct Metric { int n=0,positive=0;double brier=0,loss=0,mean=0,ap=0,auc=0,ece=0; std::vector<std::array<double,4>> bins; };
static Metric measure(const Model&m,const Data&d,int split,bool calibrated) {
  Metric a;std::vector<std::pair<double,double>> scored;
  for(const auto&r:d.rows)if(r.split==split){double p=sigmoid(raw_score(m,r)+(calibrated?m.offset:0)),y=r.y[m.target];
    a.n++;a.positive+=y;a.brier+=(p-y)*(p-y);a.loss+=logloss(y,p);a.mean+=p;scored.emplace_back(p,y);}
  require(a.n>0,"empty_evaluation");a.brier/=a.n;a.loss/=a.n;a.mean/=a.n;
  std::sort(scored.begin(),scored.end(),[](auto a,auto b){return a.first>b.first;});
  double tp=0,fp=0,area=0,previous_tp=0,previous_fp=0;
  for(size_t i=0;i<scored.size();) {
    size_t end=i;double positives=0;while(end<scored.size()&&scored[end].first==scored[i].first){positives+=scored[end].second;end++;}
    tp+=positives;fp+=(end-i)-positives;
    if(a.positive)a.ap+=(positives/a.positive)*(tp/(tp+fp));
    area+=(fp-previous_fp)*(tp+previous_tp)/2;previous_tp=tp;previous_fp=fp;i=end;
  }
  a.auc=a.positive&&a.positive<a.n?area/(a.positive*(a.n-a.positive)):0;
  size_t begin=0;
  for(size_t b=0;b<10&&begin<scored.size();b++) {
    size_t end=std::max(begin,(b+1)*scored.size()/10);
    while(end<scored.size() && end>0 && scored[end].first==scored[end-1].first)end++;
    double p=0,y=0;
    for(size_t i=begin;i<end;i++){p+=scored[i].first;y+=scored[i].second;}
    if(end>begin){a.bins.push_back({double(end-begin),p/(end-begin),y/(end-begin),y});a.ece+=std::abs(p-y)/a.n;}
    begin=end;
  }
  return a;
}
static void metric_json(std::ostream&o,const Metric&m) {
  o<<"{\"rows\":"<<m.n<<",\"positive\":"<<m.positive<<",\"prevalence\":"<<double(m.positive)/m.n
   <<",\"mean_predicted_probability\":"<<m.mean<<",\"log_loss\":"<<m.loss<<",\"brier\":"<<m.brier
   <<",\"average_precision\":"<<m.ap<<",\"roc_auc\":"<<m.auc<<",\"tie_aware_equal_count_ece10\":"<<m.ece<<",\"calibration_bins\":[";
  for(size_t i=0;i<m.bins.size();i++){if(i)o<<',';const auto&b=m.bins[i];o<<"{\"n\":"<<b[0]<<",\"predicted\":"<<b[1]<<",\"observed\":"<<b[2]<<",\"positive\":"<<b[3]<<'}';}o<<"]}";
}
static void array_json(std::ostream&o,const std::vector<double>&v) {o<<'[';for(size_t j=0;j<v.size();j++){if(j)o<<',';o<<v[j];}o<<']';}
int main(int argc,char**argv)try {
  require(argc==4,"usage_fall_transition_probe_DATA_NEW_REPORT_NEW_PRIVATE_WEIGHTS");
  require(!std::ifstream(argv[2])&&!std::ifstream(argv[3]),"output_exists");
  const auto d=load(argv[1]);std::vector<Model> models;
  for(int target=0;target<2;target++)for(const std::string name:{"prevalence","action_only","state_only","state_action"}) {
    auto m=fit(d,name,target);calibrate(m,d);models.push_back(std::move(m));
  }
  std::ofstream out(argv[2]),weights(argv[3]);require(bool(out)&&bool(weights),"output_open_failed");out<<std::setprecision(12);weights<<std::setprecision(17);
  out<<"{\"schema\":\"rek.received_fall_onset_probe.v1\",\"native_cpp_cpu\":true,\"features\":"<<d.features
     <<",\"training\":{\"algorithm\":\"logistic_regression_full_batch_adam\",\"iterations\":600,\"learning_rate\":0.03,\"l2\":0.001,\"class_weights\":[1,1],\"oversampling\":false,\"scaler\":\"training_only_population_mean_stddev\"},"
     <<"\"calibration\":\"intercept_only_on_separate_chronological_process_groups\",\"models\":[";
  weights<<"{\"schema\":\"rek.received_fall_onset_weights.v1\",\"runtime_enabled\":false,\"models\":[";
  for(size_t i=0;i<models.size();i++){
    const auto&m=models[i];if(i){out<<',';weights<<',';}
    out<<"{\"target\":"<<m.target<<",\"name\":\""<<m.name<<"\",\"calibration_intercept\":"<<m.offset<<",\"test_uncalibrated\":";
    metric_json(out,measure(m,d,2,false));out<<",\"test_calibrated\":";metric_json(out,measure(m,d,2,true));
    out<<",\"calibration_calibrated\":";metric_json(out,measure(m,d,1,true));out<<'}';
    weights<<"{\"target\":"<<m.target<<",\"name\":\""<<m.name<<"\",\"columns\":[";
    for(size_t j=0;j<m.columns.size();j++){if(j)weights<<',';weights<<m.columns[j];}
    weights<<"],\"mean\":";array_json(weights,m.mean);weights<<",\"scale\":";array_json(weights,m.scale);weights<<",\"weights\":";array_json(weights,m.w);weights<<",\"calibration_intercept\":"<<m.offset<<'}';
  }
  out<<"],\"runtime_enabled\":false,\"causal_transition_validated\":false,\"limits\":[\"received_onset_forecast_not_server_physics\",\"no_identified_person_holdout\",\"no_executed_action_or_contact_labels\",\"small_number_of_independent_positive_events\",\"no_policy_rollout_validation\"]}\n";
  weights<<"]}\n";
  std::cout<<"{\"rows\":"<<d.rows.size()<<",\"models\":"<<models.size()<<",\"runtime_enabled\":false}\n";
  return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
