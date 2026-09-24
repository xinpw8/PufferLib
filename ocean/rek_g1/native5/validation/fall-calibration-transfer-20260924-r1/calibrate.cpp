#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

static void need(bool x,const char* why){if(!x)throw std::runtime_error(why);}
static double sigmoid(double z){return z>=0?1/(1+std::exp(-z)):std::exp(z)/(1+std::exp(z));}
static double softplus(double z){return std::max(z,0.0)+std::log1p(std::exp(-std::abs(z)));}
static double logit(double p){need(p>0&&p<1,"bad_prevalence");return std::log(p)-std::log1p(-p);}
template<class T>static T read(std::ifstream& f){T x{};f.read(reinterpret_cast<char*>(&x),sizeof(x));need(bool(f),"truncated_input");return x;}
struct Model{std::vector<uint32_t> columns;std::vector<double> mean,scale,w;double old_offset;};
struct Row{uint32_t split,group,checkpoint,round;std::array<double,2> z,y;};
struct Data{std::array<Model,2> models;std::vector<Row> cal,test;};
static Data load(const char* file){
 std::ifstream f(file,std::ios::binary);need(bool(f),"input_open");char magic[8];f.read(magic,8);need(std::string(magic,7)=="REKCAL1","bad_magic");
 auto version=read<uint32_t>(f),features=read<uint32_t>(f),nc=read<uint32_t>(f),nt=read<uint32_t>(f);need(version==1&&features==67&&nc==2208&&nt==3693,"header_mismatch");Data d;
 for(auto& m:d.models){auto n=read<uint32_t>(f);need(n==42,"model_shape");for(uint32_t k=0;k<n;k++){auto c=read<uint32_t>(f);need(c<features,"column_range");m.columns.push_back(c);}for(auto* v:{&m.mean,&m.scale})for(uint32_t k=0;k<n;k++)v->push_back(read<double>(f));for(uint32_t k=0;k<=n;k++)m.w.push_back(read<double>(f));m.old_offset=read<double>(f);for(auto x:m.scale)need(std::isfinite(x)&&x>0,"invalid_scale");}
 std::map<uint32_t,uint32_t> groups,checkpoints;
 for(uint32_t i=0;i<nc+nt;i++){Row r{};r.split=read<uint32_t>(f);r.group=read<uint32_t>(f);r.checkpoint=read<uint32_t>(f);r.round=read<uint32_t>(f);need(r.split<=1,"split_range");std::vector<double>x(features);for(auto& v:x){v=read<double>(f);need(std::isfinite(v),"nonfinite_feature");}for(auto& y:r.y){y=read<double>(f);need(y==0||y==1,"binary_label");}for(int t=0;t<2;t++){const auto&m=d.models[t];r.z[t]=m.w[0];for(size_t k=0;k<m.columns.size();k++)r.z[t]+=m.w[k+1]*(x[m.columns[k]]-m.mean[k])/m.scale[k];need(std::isfinite(r.z[t]),"nonfinite_logit");}for(auto* g:{&groups,&checkpoints}){const auto key=g==&groups?r.group:r.checkpoint;auto [it,inserted]=g->emplace(key,r.split);need(inserted||it->second==r.split,"group_split_overlap");}(r.split?d.test:d.cal).push_back(r);}
 need(f.peek()==EOF&&d.cal.size()==nc&&d.test.size()==nt,"dataset_length");return d;
}
constexpr double Ridge=.001;
struct Deriv{double loss=0,ga=0,gb=0,haa=0,hab=0,hbb=0,mean=0;};
static Deriv objective(const std::vector<Row>& rows,int t,double a,double b){Deriv d;for(const auto&r:rows){const double z=r.z[t],v=a*z+b,p=sigmoid(v),e=p-r.y[t],h=p*(1-p);d.loss+=softplus(v)-r.y[t]*v;d.ga+=e*z;d.gb+=e;d.haa+=h*z*z;d.hab+=h*z;d.hbb+=h;d.mean+=p;}const double n=rows.size();d.loss=d.loss/n+Ridge*a*a/2;d.ga=d.ga/n+Ridge*a;d.gb/=n;d.haa=d.haa/n+Ridge;d.hab/=n;d.hbb/=n;d.mean/=n;return d;}
struct Fit{double a=1,b=0,prevalence=0;int iterations=0;bool converged=false;Deriv d;};
static Fit fit(const std::vector<Row>& rows,int t){
 Fit f;double mz=0;for(const auto&r:rows){f.prevalence+=r.y[t];mz+=r.z[t];}f.prevalence/=rows.size();mz/=rows.size();const double boundary=logit(f.prevalence);auto atzero=objective(rows,t,0,boundary);
 if(atzero.ga>=0){f.a=0;f.b=boundary;f.converged=std::abs(atzero.gb)<=1e-10;f.d=atzero;return f;}
 f.b=boundary-mz;
 for(int iter=0;iter<200;iter++){f.iterations=iter+1;f.d=objective(rows,t,f.a,f.b);if(std::max(std::abs(f.d.ga),std::abs(f.d.gb))<=1e-10){f.converged=true;break;}const double det=f.d.haa*f.d.hbb-f.d.hab*f.d.hab;need(det>0&&std::isfinite(det),"singular_hessian");const double da=-(f.d.hbb*f.d.ga-f.d.hab*f.d.gb)/det,db=-(-f.d.hab*f.d.ga+f.d.haa*f.d.gb)/det;need(f.d.ga*da+f.d.gb*db<0,"non_descent_direction");double step=1;if(da<0)step=std::min(step,.99*f.a/(-da));bool accepted=false;for(int ls=0;ls<80;ls++){const double a=f.a+step*da,b=f.b+step*db;auto next=objective(rows,t,a,b);if(a>=0&&next.loss<=f.d.loss+1e-4*step*(f.d.ga*da+f.d.gb*db)){f.a=a;f.b=b;accepted=true;break;}step*=.5;}need(accepted,"line_search_failed");}
 f.d=objective(rows,t,f.a,f.b);if(std::max(std::abs(f.d.ga),std::abs(f.d.gb))<=1e-10)f.converged=true;need(f.converged,"calibration_not_converged");need(std::abs(f.d.mean-f.prevalence)<1e-8,"intercept_mean_constraint");return f;
}
struct Prediction{double score,p,loss,y;uint32_t group,checkpoint;};
static std::vector<Prediction> predict(const std::vector<Row>& rows,int t,double a,double b){std::vector<Prediction> out;out.reserve(rows.size());for(const auto&r:rows){double z=a*r.z[t]+b;out.push_back({z,sigmoid(z),softplus(z)-r.y[t]*z,r.y[t],r.group,r.checkpoint});}return out;}
struct Metric{size_t n=0;double positives=0,brier=0,loss=0,mean=0,ap=NAN;};
static Metric metric(const std::vector<Prediction>& v){Metric m;m.n=v.size();need(m.n>0,"empty_metric");std::vector<std::pair<double,double>> ranked;ranked.reserve(m.n);for(const auto&r:v){m.positives+=r.y;m.brier+=(r.p-r.y)*(r.p-r.y);m.loss+=r.loss;m.mean+=r.p;ranked.emplace_back(r.score,r.y);}m.brier/=m.n;m.loss/=m.n;m.mean/=m.n;if(m.positives>0){m.ap=0;std::sort(ranked.begin(),ranked.end(),[](const auto&a,const auto&b){return a.first>b.first;});double tp=0;for(size_t i=0;i<ranked.size();){size_t j=i;double add=0;while(j<ranked.size()&&ranked[j].first==ranked[i].first){add+=ranked[j].second;j++;}tp+=add;m.ap+=(add/m.positives)*(tp/double(j));i=j;}}return m;}
static void number(std::ostream&o,double x){if(std::isfinite(x))o<<x;else o<<"null";}
static void metric_json(std::ostream&o,const Metric&m){o<<"{\"windows\":"<<m.n<<",\"positive\":"<<m.positives<<",\"prevalence\":"<<m.positives/m.n<<",\"mean_prediction\":"<<m.mean<<",\"brier\":"<<m.brier<<",\"log_loss\":"<<m.loss<<",\"average_precision\":";number(o,m.ap);o<<'}';}
static uint32_t random32(uint32_t& s){s^=s<<13;s^=s>>17;s^=s<<5;return s;}
static double quantile(std::vector<double> a,double q){a.erase(std::remove_if(a.begin(),a.end(),[](double x){return !std::isfinite(x);}),a.end());if(a.empty())return NAN;std::sort(a.begin(),a.end());return a[size_t(std::floor((a.size()-1)*q))];}
static void interval(std::ostream&o,const std::vector<double>&v){o<<"{\"finite_replicates\":"<<std::count_if(v.begin(),v.end(),[](double x){return std::isfinite(x);})<<",\"lower\":";number(o,quantile(v,.025));o<<",\"upper\":";number(o,quantile(v,.975));o<<'}';}
struct Bootstrap{std::array<std::vector<double>,3> calibrated,baseline;std::vector<double> brier_difference,loss_difference;};
static Bootstrap bootstrap(const std::vector<Prediction>& cal,const std::vector<Prediction>& base,const std::vector<std::vector<uint32_t>>& draws){
 std::map<uint32_t,std::vector<size_t>> indices;for(size_t i=0;i<cal.size();i++)indices[cal[i].group].push_back(i);Bootstrap out;for(const auto&draw:draws){std::vector<Prediction>a,b;for(auto group:draw)for(auto i:indices.at(group)){a.push_back(cal[i]);b.push_back(base[i]);}const auto ma=metric(a),mb=metric(b);const double va[]={ma.brier,ma.loss,ma.ap},vb[]={mb.brier,mb.loss,mb.ap};for(int k=0;k<3;k++){out.calibrated[k].push_back(va[k]);out.baseline[k].push_back(vb[k]);}out.brier_difference.push_back(ma.brier-mb.brier);out.loss_difference.push_back(ma.loss-mb.loss);}return out;
}
static void selftest(){
 need(sigmoid(1000)==1&&sigmoid(-1000)==0&&std::isfinite(softplus(1000)),"stable_math_test");
 std::vector<Prediction> v{{0,.5,softplus(0),0,1,1},{0,.5,softplus(0),1,2,1}};auto m=metric(v);need(m.brier==.25&&m.ap==.5,"tied_metric_test");v[1].score=1;need(metric(v).ap==1,"perfect_ranking_test");v[1].y=0;need(std::isnan(metric(v).ap),"undefined_ap_test");
 std::vector<Row> r(200);for(size_t i=0;i<r.size();i++){r[i].z={double(i%20)-10,double(i%20)-10};r[i].y={double(i%11==0),double(i%11==0)};}const auto f=fit(r,0);need(f.converged&&f.a>=0&&std::abs(f.d.gb)<1e-9,"fit_test");for(auto&x:r)x.z[0]=-x.y[0];const auto bound=fit(r,0);need(bound.a==0&&bound.d.ga>=0,"boundary_kkt_test");
 const auto d=objective(r,0,.4,-2.0),plus=objective(r,0,.400001,-2),minus=objective(r,0,.399999,-2);need(std::abs(d.ga-(plus.loss-minus.loss)/.000002)<1e-8,"gradient_test");need(quantile({1,2,3,4},.975)==3,"quantile_test");std::cout<<"self_tests_passed\n";
}
int main(int argc,char**argv)try{
 if(argc==2&&std::string(argv[1])=="--self-test"){selftest();return 0;}
 need(argc==3,"usage_calibrate_DATA_NEW_REPORT");need(bool(std::ifstream("freeze.json")),"freeze_receipt_required");need(!std::ifstream(argv[2]),"output_exists");const auto d=load(argv[1]);std::array<Fit,2> fits{fit(d.cal,0),fit(d.cal,1)};
 std::vector<uint32_t> groups;for(const auto&r:d.test)if(std::find(groups.begin(),groups.end(),r.group)==groups.end())groups.push_back(r.group);need(groups.size()==17,"test_cluster_count");std::vector<std::vector<uint32_t>> draws(2000);uint32_t rng=73;for(auto&draw:draws)for(size_t i=0;i<groups.size();i++)draw.push_back(groups[random32(rng)%groups.size()]);
 std::ofstream o(argv[2]);need(bool(o),"output_open");o<<std::setprecision(17)<<"{\"schema\":\"rek.received_onset_calibration_transfer_result.v1\",\"runtime_enabled\":false,\"ridge\":"<<Ridge<<",\"bootstrap_replicates\":2000,\"bootstrap_seed\":73,\"test_processes\":17,\"conditional_on_fitted_calibrator\":true,\"targets\":[";bool favorable=true;
 for(int t=0;t<2;t++){if(t)o<<',';const auto&f=fits[t];std::array<std::vector<Prediction>,4>pred{predict(d.test,t,1,0),predict(d.test,t,1,d.models[t].old_offset),predict(d.test,t,f.a,f.b),predict(d.test,t,0,logit(f.prevalence))};const char* names[]={"raw_geometry","original_intercept","slope_intercept","calibration_prevalence"};std::array<Metric,4> metrics;for(int k=0;k<4;k++)metrics[k]=metric(pred[k]);const auto bs=bootstrap(pred[2],pred[3],draws);bool good=metrics[2].brier<metrics[3].brier&&quantile(bs.brier_difference,.975)<0&&metrics[2].loss<=metrics[3].loss;favorable=favorable&&good;size_t positive_groups=0;for(auto g:groups)if(std::any_of(d.test.begin(),d.test.end(),[&](const auto&r){return r.group==g&&r.y[t]>0;}))positive_groups++;
 o<<"{\"target\":"<<t<<",\"slope\":"<<f.a<<",\"intercept\":"<<f.b<<",\"calibration_prevalence\":"<<f.prevalence<<",\"calibration_mean_prediction\":"<<f.d.mean<<",\"iterations\":"<<f.iterations<<",\"converged\":true,\"objective\":"<<f.d.loss<<",\"slope_gradient\":"<<f.d.ga<<",\"intercept_gradient\":"<<f.d.gb<<",\"test_positive_processes\":"<<positive_groups<<",\"criterion_met\":"<<(good?"true":"false")<<",\"pooled\":{";for(int k=0;k<4;k++){if(k)o<<',';o<<'"'<<names[k]<<"\":";metric_json(o,metrics[k]);}o<<"},\"per_checkpoint\":[";std::map<uint32_t,int>checkpoint_ids;for(const auto&r:d.test)checkpoint_ids[r.checkpoint]++;bool first=true;for(auto [ck,count]:checkpoint_ids){(void)count;if(!first)o<<',';first=false;o<<"{\"checkpoint_index\":"<<ck<<",\"models\":{";for(int k=0;k<4;k++){std::vector<Prediction>subset;for(const auto&p:pred[k])if(p.checkpoint==ck)subset.push_back(p);if(k)o<<',';o<<'"'<<names[k]<<"\":";metric_json(o,metric(subset));}o<<"}}";}o<<"],\"bootstrap\":{";const char*metric_names[]={"brier","log_loss","average_precision"};for(int arm=0;arm<2;arm++){if(arm)o<<',';o<<'"'<<(arm?"calibration_prevalence":"slope_intercept")<<"\":{";for(int k=0;k<3;k++){if(k)o<<',';o<<'"'<<metric_names[k]<<"\":";interval(o,arm?bs.baseline[k]:bs.calibrated[k]);}o<<'}';}o<<",\"paired_brier_difference\":";interval(o,bs.brier_difference);o<<",\"paired_log_loss_difference\":";interval(o,bs.loss_difference);o<<"}}";
 }
 o<<"],\"favorable_calibration_criterion_met\":"<<(favorable?"true":"false")<<",\"physical_transition_validated\":false}\n";o.close();need(bool(o),"output_write");std::cout<<"single_frozen_fit_and_transfer_evaluation_completed\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
