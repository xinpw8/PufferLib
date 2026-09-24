// Private diagnostic only. No policy, replay, reward or simulator mutation.
#include "authentic_trajectory.h"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#ifndef REK_BASELINE_CPU_ONLY
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#endif

namespace {
constexpr int D=223, P=D+1;
constexpr double Ridge=.01;
constexpr const char* DataSha="eb6b1ae210b3b3911597f93a513ae32081f15e4871b29c1cb29e75829f718655";
constexpr const char* CheckpointSha="c2c4987b268996cd912fe35e6ba5f9b20a94d69fd93e4ef15b6c5fc71ee5e533";
constexpr const char* IdentitySha="5e65d45f54e444a47d5f87e10691715ac1b0bc82a44685dd015e0e9efdb333c0";
void need(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
struct Input {
 int n=0,d=D,folds=0;
 std::vector<float> x,reward,gamma;
 std::vector<int> episode,eligible,bounds;
};
Input input_from(const rek_authentic::Dataset& data){
 need(data.format_version==5 && data.rows.size()==14957 && data.sequences.size()==5,"wrong fixed dataset shape");
 need(data.digest==DataSha && data.checkpoint_sha256==CheckpointSha && data.identity_sha256==IdentitySha,"dataset identity mismatch");
 Input h;h.n=int(data.rows.size());h.folds=int(data.sequences.size());h.x.resize(h.n*D);h.reward.resize(h.n);h.gamma.resize(h.n);h.episode.resize(h.n);h.eligible.resize(h.n);
 for(int fold=0;fold<h.folds;fold++){
  const auto& s=data.sequences[fold];h.bounds.push_back(int(s.begin));h.bounds.push_back(int(s.end));
  int count=0;
  for(size_t i=s.begin;i<s.end;i++){
   const auto& r=data.rows[i];need(r.policy_weight==0||r.policy_weight==1,"nonbinary actor weight");
   // Features are ONLY the recorded pre-action observation vector. No action,
   // outcome, score delta, source sequence, time or fold ID is appended.
   std::copy(r.obs.begin(),r.obs.end(),h.x.begin()+i*D);
   h.reward[i]=r.reward;h.gamma[i]=r.gamma;h.episode[i]=fold;h.eligible[i]=r.policy_weight==1;count+=h.eligible[i];
  }
  need(count>1,"fold has insufficient eligible rows");
 }
 need(std::accumulate(h.eligible.begin(),h.eligible.end(),0)==14952,"unexpected actor exclusion count");return h;
}
std::vector<float> reference_returns(const Input& h){
 std::vector<float> out(h.n);
 for(int k=0;k<h.folds;k++){double g=0;for(int i=h.bounds[2*k+1]-1;i>=h.bounds[2*k];i--){g=double(h.reward[i])+double(h.gamma[i])*g;out[i]=float(g);}}
 return out;
}
struct Scale {std::vector<double> mean,sd;int n=0;double target_mean=0;};
Scale reference_scale(const Input& h,const std::vector<float>& y,int heldout){
 Scale s;s.mean.assign(h.d,0);s.sd.assign(h.d,0);
 for(int i=0;i<h.n;i++)if(h.eligible[i]&&h.episode[i]!=heldout){s.n++;s.target_mean+=y[i];for(int j=0;j<h.d;j++)s.mean[j]+=h.x[i*h.d+j];}
 need(s.n>0,"empty training fold");s.target_mean/=s.n;for(double& m:s.mean)m/=s.n;
 for(int i=0;i<h.n;i++)if(h.eligible[i]&&h.episode[i]!=heldout)for(int j=0;j<h.d;j++){const double e=h.x[i*h.d+j]-s.mean[j];s.sd[j]+=e*e;}
 for(double& v:s.sd)v=std::sqrt(v/s.n);return s;
}
struct Metric {double n=0,sum_error=0,sum_error2=0;};
std::array<Metric,3> reference_metrics(const Input& h,const std::vector<float>& y,const std::vector<float>& fitted,const std::vector<float>& constant,int fold){
 std::array<Metric,3> out{};
 for(int i=0;i<h.n;i++)if(h.eligible[i]&&(fold<0||h.episode[i]==fold))for(int k=0;k<3;k++){
  const double e=double(y[i])-(k==0?0:k==1?constant[i]:fitted[i]);out[k].n++;out[k].sum_error+=e;out[k].sum_error2+=e*e;
 }return out;
}
void emit_metrics(std::ostream& out,const std::array<Metric,3>& ms){
 const char* names[]={"zero","training_mean","cross_fitted_ridge"};out<<'{';
 for(int k=0;k<3;k++){if(k)out<<',';const auto& m=ms[k];need(m.n>0,"empty evaluation");const double mean=m.sum_error/m.n,mse=m.sum_error2/m.n;
  out<<'"'<<names[k]<<"\":{\"rows\":"<<int(m.n)<<",\"mse\":"<<mse<<",\"residual_mean\":"<<mean<<",\"residual_variance\":"<<std::max(0.,mse-mean*mean)<<'}';
 }out<<'}';
}
template<class T>void emit_array(std::ostream& out,const std::vector<T>& a){out<<'[';for(size_t i=0;i<a.size();i++){if(i)out<<',';need(std::isfinite(double(a[i])),"nonfinite output");out<<a[i];}out<<']';}
// Independent CPU solver is used ONLY for synthetic numerical checks.
std::vector<double> reference_solve(std::vector<double> a,std::vector<double> b,int p){
 for(int i=0;i<p;i++)for(int j=0;j<=i;j++){double v=a[i*p+j];for(int k=0;k<j;k++)v-=a[i*p+k]*a[j*p+k];if(i==j){need(v>0,"reference nonpositive matrix");a[i*p+j]=std::sqrt(v);}else a[i*p+j]=v/a[j*p+j];}
 for(int i=0;i<p;i++){for(int j=0;j<i;j++)b[i]-=a[i*p+j]*b[j];b[i]/=a[i*p+i];}
 for(int i=p-1;i>=0;i--){for(int j=i+1;j<p;j++)b[i]-=a[j*p+i]*b[j];b[i]/=a[i*p+i];}return b;
}
Input toy(){
 Input h;h.n=9;h.d=3;h.folds=3;h.bounds={0,3,3,6,6,9};
 h.episode={0,0,0,1,1,1,2,2,2};h.eligible={1,1,0,1,1,0,1,1,0};h.reward={0,.2f,1,0,-.2f,-1,0,.4f,1};h.gamma.assign(9,.9f);
 for(int i=0;i<9;i++){h.x.push_back(float(i));h.x.push_back(7);h.x.push_back(float(i%2));}return h;
}
void cpu_tests(){
 const auto h=toy();const auto y=reference_returns(h);need(std::abs(y[0]-.99)<1e-6,"excluded terminal reward lost");
 auto a=reference_scale(h,y,2);need(a.n==4&&a.sd[1]==0,"eligible scaler/constant feature failure");
 auto changed=h;for(int i=6;i<9;i++)for(int j=0;j<3;j++)changed.x[i*3+j]=10000.f;auto b=reference_scale(changed,y,2);
 need(a.mean==b.mean&&a.sd==b.sd&&a.target_mean==b.target_mean,"heldout changed training scaler");
 changed=h;for(int i:{2,5})for(int j=0;j<3;j++)changed.x[i*3+j]=-10000;auto c=reference_scale(changed,y,2);need(a.mean==c.mean&&a.sd==c.sd,"excluded row changed scaler");
 auto sol=reference_solve({2.,0.,0.,3.},{4.,9.},2);need(std::abs(sol[0]-2)<1e-12&&std::abs(sol[1]-3)<1e-12,"reference solver");
 std::vector<float> zero(9,0);auto metrics=reference_metrics(h,y,zero,zero,-1);need(metrics[0].n==6,"metrics did not exclude actor-zero rows");
 std::cout<<"{\"cpu_tests_passed\":true,\"checks\":6,\"gpu_used\":false}\n";
}
#ifndef REK_BASELINE_CPU_ONLY
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void cb(cublasStatus_t e){need(e==CUBLAS_STATUS_SUCCESS,"cuBLAS failure");}
void cs(cusolverStatus_t e){need(e==CUSOLVER_STATUS_SUCCESS,"cuSOLVER failure");}
template<class T>struct Device {
 T* p=nullptr;size_t n=0;explicit Device(size_t count):n(count){ck(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));}
 ~Device(){cudaFree(p);}Device(const Device&)=delete;Device& operator=(const Device&)=delete;
 void put(const std::vector<T>& v){need(v.size()==n,"device shape mismatch");ck(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
 std::vector<T> get()const{std::vector<T> v(n);ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
struct Handles {cublasHandle_t blas{};cusolverDnHandle_t solver{};Handles(){cb(cublasCreate(&blas));cb(cublasSetMathMode(blas,CUBLAS_PEDANTIC_MATH));cb(cublasSetAtomicsMode(blas,CUBLAS_ATOMICS_NOT_ALLOWED));cs(cusolverDnCreate(&solver));}~Handles(){cusolverDnDestroy(solver);cublasDestroy(blas);}};
__global__ void returns_cuda(const float* rewards,const float* gammas,const int* bounds,int folds,float* returns,int* status){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=folds)return;double g=0;
 for(int i=bounds[2*k+1]-1;i>=bounds[2*k];i--){g=double(rewards[i])+double(gammas[i])*g;returns[i]=float(g);if(!isfinite(returns[i]))atomicExch(status,1);}
}
__global__ void scale_cuda(const float* x,const float* returns,const int* episode,const int* eligible,int n,int d,int heldout,double* mean,double* sd,double* target_mean,int* status){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>d)return;double m=0,m2=0;int count=0;
 for(int i=0;i<n;i++)if(eligible[i]&&episode[i]!=heldout){const double v=j==d?returns[i]:x[i*d+j];count++;const double delta=v-m;m+=delta/count;m2+=delta*(v-m);}
 if(count==0){atomicExch(status,2);return;}if(j==d)*target_mean=m;else{mean[j]=m;sd[j]=sqrt(fmax(0.,m2/count));if(!isfinite(mean[j])||!isfinite(sd[j]))atomicExch(status,3);}
}
__global__ void design_cuda(const float* x,const float* returns,const int* episode,const int* eligible,int n,int d,int heldout,const double* mean,const double* sd,double* design,double* targets){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=n*(d+1))return;const int row=k%n,col=k/n;const bool train=eligible[row]&&episode[row]!=heldout;
 design[k]=!train?0:col==d?1:sd[col]==0?0:(double(x[row*d+col])-mean[col])/sd[col];
 if(col==0)targets[row]=train?returns[row]:0;
}
__global__ void ridge_cuda(double* a,int p,double ridge){int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<p-1)a[j*p+j]+=ridge;}
__global__ void predict_cuda(const float* x,const int* episode,int n,int d,int heldout,const double* mean,const double* sd,const double* weights,const double* target_mean,float* prediction,float* constant,int* status){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||episode[i]!=heldout)return;double v=weights[d];for(int j=0;j<d;j++)if(sd[j]!=0)v+=weights[j]*(double(x[i*d+j])-mean[j])/sd[j];prediction[i]=float(v);constant[i]=float(*target_mean);if(!isfinite(prediction[i])||!isfinite(constant[i]))atomicExch(status,4);
}
__global__ void metrics_cuda(const float* y,const float* prediction,const float* constant,const int* episode,const int* eligible,int n,int heldout,Metric* output){
 const int k=threadIdx.x;if(k>=3)return;Metric m{};for(int i=0;i<n;i++)if(eligible[i]&&(heldout<0||episode[i]==heldout)){const double e=double(y[i])-(k==0?0:k==1?constant[i]:prediction[i]);m.n++;m.sum_error+=e;m.sum_error2+=e*e;}output[k]=m;
}
__global__ void normal_residual_cuda(const double* a,const double* b,const double* w,int p,double* error){
 int j=threadIdx.x;if(j>=p)return;double v=-b[j],scale=fabs(b[j])+1;for(int k=0;k<p;k++){v+=a[j+k*p]*w[k];scale+=fabs(a[j+k*p]*w[k]);}error[j]=fabs(v)/scale;
}
struct GpuInput {
 Device<float> x,reward,gamma,y,prediction,constant;Device<int> episode,eligible,bounds,status;
 explicit GpuInput(const Input& h):x(h.x.size()),reward(h.n),gamma(h.n),y(h.n),prediction(h.n),constant(h.n),episode(h.n),eligible(h.n),bounds(h.bounds.size()),status(1){
  x.put(h.x);reward.put(h.reward);gamma.put(h.gamma);episode.put(h.episode);eligible.put(h.eligible);bounds.put(h.bounds);ck(cudaMemset(status.p,0,sizeof(int)));ck(cudaMemset(prediction.p,0,h.n*sizeof(float)));ck(cudaMemset(constant.p,0,h.n*sizeof(float)));
  returns_cuda<<<1,32>>>(reward.p,gamma.p,bounds.p,h.folds,y.p,status.p);ck(cudaGetLastError());ck(cudaDeviceSynchronize());need(status.get()[0]==0,"nonfinite GPU MC return");
 }
};
struct Fold {int train_n=0;std::vector<double> mean,sd,weights;double train_mean=0,scale_error=0,normal_error=0;std::array<Metric,3> metrics;};
Fold fit_fold(const Input& h,GpuInput& g,Handles& handles,int heldout,bool synthetic=false){
 const int p=h.d+1;Fold result;for(int i=0;i<h.n;i++)result.train_n+=h.eligible[i]&&h.episode[i]!=heldout;need(result.train_n>p||synthetic,"insufficient training rows");
 Device<double> means(h.d),sd(h.d),target_mean(1),design(size_t(h.n)*p),targets(h.n),a(p*p),b(p),original_a(p*p),original_b(p),residual(p);Device<int> info(1);
 scale_cuda<<<(p+127)/128,128>>>(g.x.p,g.y.p,g.episode.p,g.eligible.p,h.n,h.d,heldout,means.p,sd.p,target_mean.p,g.status.p);
 design_cuda<<<(h.n*p+255)/256,256>>>(g.x.p,g.y.p,g.episode.p,g.eligible.p,h.n,h.d,heldout,means.p,sd.p,design.p,targets.p);ck(cudaGetLastError());
 const double alpha=1./result.train_n,beta=0;
 cb(cublasDgemm(handles.blas,CUBLAS_OP_T,CUBLAS_OP_N,p,p,h.n,&alpha,design.p,h.n,design.p,h.n,&beta,a.p,p));
 cb(cublasDgemv(handles.blas,CUBLAS_OP_T,h.n,p,&alpha,design.p,h.n,targets.p,1,&beta,b.p,1));
 ridge_cuda<<<1,256>>>(a.p,p,Ridge);ck(cudaGetLastError());ck(cudaMemcpy(original_a.p,a.p,p*p*sizeof(double),cudaMemcpyDeviceToDevice));ck(cudaMemcpy(original_b.p,b.p,p*sizeof(double),cudaMemcpyDeviceToDevice));
 int workspace=0;cs(cusolverDnDpotrf_bufferSize(handles.solver,CUBLAS_FILL_MODE_LOWER,p,a.p,p,&workspace));Device<double> work(workspace);
 cs(cusolverDnDpotrf(handles.solver,CUBLAS_FILL_MODE_LOWER,p,a.p,p,work.p,workspace,info.p));need(info.get()[0]==0,"Cholesky factorization failed");
 cs(cusolverDnDpotrs(handles.solver,CUBLAS_FILL_MODE_LOWER,p,1,a.p,p,b.p,p,info.p));need(info.get()[0]==0,"Cholesky solve failed");
 normal_residual_cuda<<<1,256>>>(original_a.p,original_b.p,b.p,p,residual.p);
 predict_cuda<<<(h.n+255)/256,256>>>(g.x.p,g.episode.p,h.n,h.d,heldout,means.p,sd.p,b.p,target_mean.p,g.prediction.p,g.constant.p,g.status.p);ck(cudaGetLastError());ck(cudaDeviceSynchronize());need(g.status.get()[0]==0,"GPU fit/prediction failed");
 result.mean=means.get();result.sd=sd.get();result.weights=b.get();result.train_mean=target_mean.get()[0];const auto err=residual.get();result.normal_error=*std::max_element(err.begin(),err.end());need(result.normal_error<1e-10,"normal equation residual too large");
 // CPU scaler is an independent comparison only; GPU values drive the fit.
 const auto y=g.y.get();const auto reference=reference_scale(h,y,heldout);
 for(int j=0;j<h.d;j++){result.scale_error=std::max(result.scale_error,std::abs(reference.mean[j]-result.mean[j])/(1+std::abs(reference.mean[j])));result.scale_error=std::max(result.scale_error,std::abs(reference.sd[j]-result.sd[j])/(1+reference.sd[j]));if(reference.sd[j]==0)need(result.sd[j]==0,"constant feature not zero");}
 need(result.scale_error<1e-10&&std::abs(result.train_mean-reference.target_mean)<1e-10,"GPU scaler disagrees with CPU reference");
 if(synthetic){
  std::vector<double> gram(p*p,0),rhs(p,0);for(int i=0;i<h.n;i++)if(h.eligible[i]&&h.episode[i]!=heldout){std::vector<double> row(p,1);for(int j=0;j<h.d;j++)row[j]=reference.sd[j]==0?0:(h.x[i*h.d+j]-reference.mean[j])/reference.sd[j];for(int j=0;j<p;j++){rhs[j]+=row[j]*y[i]/reference.n;for(int k=0;k<p;k++)gram[j*p+k]+=row[j]*row[k]/reference.n;}}
  for(int j=0;j<p-1;j++)gram[j*p+j]+=Ridge;const auto ref=reference_solve(gram,rhs,p);for(int j=0;j<p;j++)need(std::abs(ref[j]-result.weights[j])<1e-9,"synthetic CUDA ridge differs from CPU reference");
 }
 Device<Metric> stats(3);metrics_cuda<<<1,3>>>(g.y.p,g.prediction.p,g.constant.p,g.episode.p,g.eligible.p,h.n,heldout,stats.p);ck(cudaGetLastError());auto ms=stats.get();std::copy(ms.begin(),ms.end(),result.metrics.begin());return result;
}
void check_metrics(const std::array<Metric,3>& a,const std::array<Metric,3>& b){for(int k=0;k<3;k++){need(a[k].n==b[k].n,"metric row count mismatch");need(std::abs(a[k].sum_error-b[k].sum_error)<1e-8&&std::abs(a[k].sum_error2-b[k].sum_error2)<1e-8,"GPU metrics disagree with CPU reference");}}
void gpu_tests(){const auto h=toy();GpuInput g(h);Handles handles;const auto cpu=reference_returns(h),gpu=g.y.get();for(int i=0;i<h.n;i++)need(std::abs(cpu[i]-gpu[i])<1e-6,"synthetic GPU return mismatch");for(int fold=0;fold<h.folds;fold++){const auto fit=fit_fold(h,g,handles,fold,true);check_metrics(fit.metrics,reference_metrics(h,gpu,g.prediction.get(),g.constant.get(),fold));}std::cout<<"{\"gpu_synthetic_reference_tests_passed\":true,\"folds\":3}\n";}
template<class T>void binary(std::ostream& out,T value){out.write(reinterpret_cast<const char*>(&value),sizeof(T));}
std::string file_hash(const std::filesystem::path& p){const auto b=rek_authentic::read_file(p.c_str());return rek_authentic::sha256(b.data(),b.size());}
void run(const char* data_file,const char* expected,const char* protocol_file,const char* protocol_sha,const char* output,const char* executable){
 need(std::string(expected)==DataSha,"unexpected CLI dataset pin");const auto data=rek_authentic::load(data_file);const auto h=input_from(data);
 need(file_hash(protocol_file)==protocol_sha,"protocol hash mismatch");need(!std::filesystem::exists(output),"output already exists");std::filesystem::create_directory(output);const std::filesystem::path dir(output);
 std::ofstream provenance(dir/"run-provenance.json");provenance<<"{\"dataset_sha256\":\""<<data.digest<<"\",\"protocol_sha256\":\""<<protocol_sha<<"\",\"binary_sha256\":\""<<file_hash(executable)<<"\",\"rows\":"<<h.n<<",\"folds\":5,\"policy_changed\":false,\"replay_changed\":false}\n";provenance.close();
 gpu_tests();GpuInput g(h);Handles handles;const auto y=g.y.get(),ref=reference_returns(h);double return_error=0;for(int i=0;i<h.n;i++)return_error=std::max(return_error,std::abs(double(y[i])-ref[i]));need(return_error<=1e-6,"actual CUDA returns differ from CPU recurrence");
 std::vector<Fold> fits;for(int fold=0;fold<h.folds;fold++){
  fits.push_back(fit_fold(h,g,handles,fold));const auto& f=fits.back();check_metrics(f.metrics,reference_metrics(h,y,g.prediction.get(),g.constant.get(),fold));
  std::ofstream weights(dir/("fold-"+std::to_string(fold)+".json"));weights<<std::setprecision(17)<<"{\"heldout_fold\":"<<fold<<",\"heldout_sequence\":"<<data.sequences[fold].sequence<<",\"train_eligible_rows\":"<<f.train_n<<",\"ridge\":0.01,\"intercept_penalized\":false,\"training_return_mean\":"<<f.train_mean<<",\"feature_mean\":";emit_array(weights,f.mean);weights<<",\"feature_population_sd\":";emit_array(weights,f.sd);weights<<",\"weights_slopes_then_intercept\":";emit_array(weights,f.weights);weights<<",\"heldout_metrics\":";emit_metrics(weights,f.metrics);weights<<"}\n";weights.close();need(bool(weights),"fold artifact write failed");
  std::cout<<"{\"fold_finished\":"<<fold<<",\"eligible_rows\":"<<int(f.metrics[0].n)<<"}\n"<<std::flush;
 }
 Device<Metric> pooled_device(3);metrics_cuda<<<1,3>>>(g.y.p,g.prediction.p,g.constant.p,g.episode.p,g.eligible.p,h.n,-1,pooled_device.p);ck(cudaGetLastError());const auto m=pooled_device.get();std::array<Metric,3> pooled;std::copy(m.begin(),m.end(),pooled.begin());const auto prediction=g.prediction.get(),constant=g.constant.get();check_metrics(pooled,reference_metrics(h,y,prediction,constant,-1));
 std::ofstream out(dir/"row-baselines.bin",std::ios::binary);out.write("REKSB001",8);for(uint32_t v:{1u,uint32_t(h.n),uint32_t(D),5u,32u,0u})binary(out,v);out.write(data.digest.data(),64);out.write(protocol_sha,64);
 for(int i=0;i<h.n;i++){binary(out,uint32_t(i));binary(out,uint32_t(data.rows[i].sequence));binary(out,uint32_t(data.rows[i].source_seq));binary(out,uint32_t(h.eligible[i]));binary(out,y[i]);binary(out,prediction[i]);binary(out,constant[i]);binary(out,float(y[i]-prediction[i]));}out.close();need(bool(out),"baseline artifact write failed");
 std::ofstream report(dir/"report.json");report<<std::setprecision(17)<<"{\"schema\":\"rek.cross_fitted_state_return_baseline.v1\",\"dataset_sha256\":\""<<data.digest<<"\",\"checkpoint_sha256\":\""<<data.checkpoint_sha256<<"\",\"protocol_sha256\":\""<<protocol_sha<<"\",\"rows\":"<<h.n<<",\"eligible_rows\":14952,\"features\":223,\"folds\":5,\"ridge\":0.01,\"cuda_fit\":true,\"cuda_mc_returns\":true,\"cuda_metrics\":true,\"cpu_reference_only\":true,\"max_return_reference_error\":"<<return_error<<",\"pooled_metrics\":";emit_metrics(report,pooled);report<<",\"per_episode\":[";
 for(int k=0;k<5;k++){if(k)report<<',';report<<"{\"heldout_fold\":"<<k<<",\"sequence\":"<<data.sequences[k].sequence<<",\"begin_row\":"<<data.sequences[k].begin<<",\"end_row_exclusive\":"<<data.sequences[k].end<<",\"scaler_relative_reference_error\":"<<fits[k].scale_error<<",\"normal_equation_relative_residual\":"<<fits[k].normal_error<<",\"metrics\":";emit_metrics(report,fits[k].metrics);report<<'}';}
 report<<"],\"row_baseline_sha256\":\""<<file_hash(dir/"row-baselines.bin")<<"\",\"artifact_header_bytes\":160,\"artifact_row_bytes\":32,\"artifact_order\":\"original_dataset_order\",\"policy_training\":false,\"baseline_enabled\":false,\"selection\":\"fixed ridge, no sweep, all five development folds reported\"}\n";report.close();need(bool(report),"report write failed");
 need(rek_authentic::load(data_file).digest==data.digest,"source dataset changed during diagnostic");
 std::cout<<"{\"completed\":true,\"baseline_enabled\":false,\"rows\":"<<h.n<<",\"output\":\""<<output<<"\"}\n";
}
#endif
}
int main(int argc,char** argv){try{
 if(argc==2&&std::string(argv[1])=="--cpu-test"){cpu_tests();return 0;}
 if(argc==4&&std::string(argv[1])=="--inspect"){need(std::string(argv[3])==DataSha,"wrong dataset pin");const auto d=rek_authentic::load(argv[2]);const auto h=input_from(d);std::cout<<"{\"input_validated\":true,\"rows\":"<<h.n<<",\"episodes\":"<<h.folds<<",\"eligible_rows\":14952,\"gpu_used\":false}\n";return 0;}
#ifndef REK_BASELINE_CPU_ONLY
 if(argc==2&&std::string(argv[1])=="--gpu-self-test"){gpu_tests();return 0;}
 if(argc==7&&std::string(argv[1])=="--run"){run(argv[2],argv[3],argv[4],argv[5],argv[6],argv[0]);return 0;}
#endif
 throw std::runtime_error("usage: --cpu-test | --inspect DATA SHA | --gpu-self-test | --run DATA SHA PROTOCOL PROTOCOL_SHA NEW_OUT");
 }catch(const std::exception& e){std::cerr<<"state_baseline_error: "<<e.what()<<'\n';return 2;}}
