// Offline client-pose geometry diagnostic. Never advances physics or infers a
// negative outcome from missing hit packets. Contact predicates execute on GPU.
#include "primitive_contacts.cuh"
#include "../../../vendor/cJSON.h"
#include <mujoco/mujoco.h>
#include <openssl/evp.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using rek5_primitive::Shape;
using Json=std::unique_ptr<cJSON,decltype(&cJSON_Delete)>;
constexpr const char* BONES[]={"pelvis","left_hip_pitch_link","left_hip_roll_link","left_hip_yaw_link",
 "left_knee_link","left_ankle_pitch_link","left_ankle_roll_link","right_hip_pitch_link",
 "right_hip_roll_link","right_hip_yaw_link","right_knee_link","right_ankle_pitch_link",
 "right_ankle_roll_link","waist_yaw_link","waist_roll_link","torso_link",
 "left_shoulder_pitch_link","left_shoulder_roll_link","left_shoulder_yaw_link","left_elbow_link",
 "left_wrist_roll_link","left_wrist_pitch_link","left_wrist_yaw_link","right_shoulder_pitch_link",
 "right_shoulder_roll_link","right_shoulder_yaw_link","right_elbow_link","right_wrist_roll_link",
 "right_wrist_pitch_link","right_wrist_yaw_link"};
constexpr int LIMB[12]={0,0,0,0,1,1,1,1,2,3,4,5};
void require(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
void checked(cudaError_t s){require(s==cudaSuccess,cudaGetErrorString(s));}
const cJSON* field(const cJSON* j,const char* name){const auto* p=cJSON_GetObjectItemCaseSensitive(j,name);require(p,std::string("missing ")+name);return p;}
double number(const cJSON* j){require(cJSON_IsNumber(j)&&std::isfinite(j->valuedouble),"expected finite number");return j->valuedouble;}
int integer(const cJSON* j){double v=number(j);require(v==std::floor(v)&&v>=0&&v<=2147483647.,"invalid nonnegative integer");return int(v);}
std::string text(const cJSON* j){require(cJSON_IsString(j)&&j->valuestring,"expected string");return j->valuestring;}
std::string sha(const std::string& path){
 std::ifstream f(path,std::ios::binary);require(bool(f),"cannot open hash input");
 std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(),EVP_MD_CTX_free);require(bool(ctx),"digest allocation");
 require(EVP_DigestInit_ex(ctx.get(),EVP_sha256(),nullptr)==1,"digest init");char b[65536];
 while(f){f.read(b,sizeof b);if(f.gcount())require(EVP_DigestUpdate(ctx.get(),b,size_t(f.gcount()))==1,"digest update");}
 require(f.eof(),"hash read failed");unsigned char d[32];unsigned n=0;require(EVP_DigestFinal_ex(ctx.get(),d,&n)==1&&n==32,"digest final");
 std::string out;const char* h="0123456789abcdef";for(auto v:d){out+=h[v>>4];out+=h[v&15];}return out;
}
template<int N>void values(const cJSON* j,double* out){require(cJSON_IsArray(j)&&cJSON_GetArraySize(j)==N,"array size mismatch");int i=0;for(auto* v=j->child;v;v=v->next)out[i++]=number(v);}
struct Pose{double time;int frame,seq;double xyz[30][3],wxyz[30][4];};
struct Packet{double time;int frame,tick,seq,scorer=-1,points=0,kick=-1;float xyz[3]{};};
struct Capture{std::vector<Pose> poses[2];std::vector<Packet> hits,scores;std::string digest;};
Capture read_capture(const char* path,const char* expected){
 Capture out;out.digest=sha(path);require(out.digest==expected,"capture hash mismatch");std::ifstream f(path);std::string line;bool header=false;
 while(std::getline(f,line)){
  Json j(cJSON_Parse(line.c_str()),cJSON_Delete);require(bool(j),"invalid capture JSON");const std::string event=text(field(j.get(),"event"));
  if(event=="capture_start"){
   require(!header,"duplicate capture header");header=true;
   require(text(field(j.get(),"schema"))=="rek.private_ai.protocol.v5","capture schema mismatch");
   auto* scope=field(j.get(),"scope");require(cJSON_IsTrue(field(scope,"allowed"))&&cJSON_IsTrue(field(scope,"opponent_is_ai"))&&cJSON_IsFalse(field(scope,"human_in_opponent_slot")),"capture scope mismatch");
  }else if(event=="raw_bone_packet"){
   require(header,"bone packet before header");int side=integer(field(j.get(),"fighter_slot"));require(side<2&&integer(field(j.get(),"network_index"))==side,"fighter identity mismatch");
   require(integer(field(j.get(),"bone_count"))==30,"expected G1 bone count");auto* names=field(j.get(),"bone_names");require(cJSON_GetArraySize(names)==30,"bone names count");
   for(int i=0;i<30;i++)require(text(cJSON_GetArrayItem(names,i))==BONES[i],"bone order mismatch");
   Pose p{};p.time=number(field(j.get(),"monotonic_receipt_time"));p.frame=integer(field(j.get(),"unity_frame"));p.seq=integer(field(j.get(),"raw_bone_packet_sequence"));
   double xyz[90],xyzw[120];values<90>(field(j.get(),"world_positions_xyz"),xyz);values<120>(field(j.get(),"world_rotations_xyzw"),xyzw);
   for(int i=0;i<30;i++){
    p.xyz[i][0]=xyz[3*i];p.xyz[i][1]=xyz[3*i+2];p.xyz[i][2]=xyz[3*i+1];
    p.wxyz[i][0]=-xyzw[4*i+3];p.wxyz[i][1]=xyzw[4*i];p.wxyz[i][2]=xyzw[4*i+2];p.wxyz[i][3]=xyzw[4*i+1];
    double norm=0;for(double v:p.wxyz[i])norm+=v*v;require(std::abs(norm-1)<.01,"nonunit bone quaternion");for(double& v:p.wxyz[i])v/=std::sqrt(norm);
   }
   require(out.poses[side].empty()||p.time>=out.poses[side].back().time,"nonmonotonic pose receipt");out.poses[side].push_back(p);
  }else if(event=="raw_hit_packet"||event=="raw_score_packet"){
   Packet p{};p.time=number(field(j.get(),"monotonic_receipt_time"));p.frame=integer(field(j.get(),"unity_frame"));p.tick=integer(field(j.get(),"client_fixed_tick_at_observation"));auto* d=field(j.get(),"decoded");
   if(event=="raw_hit_packet"){
    p.seq=integer(field(j.get(),"raw_hit_sequence"));p.kick=integer(field(d,"is_kick"));require(p.kick<=1,"invalid kick flag");double xyz[3];values<3>(field(d,"position_xyz"),xyz);
    p.xyz[0]=float(xyz[0]);p.xyz[1]=float(xyz[2]);p.xyz[2]=float(xyz[1]);out.hits.push_back(p);
   }else{
    p.seq=integer(field(j.get(),"raw_score_sequence"));p.scorer=integer(field(d,"fighter_index"));require(p.scorer<2,"score identity");p.points=integer(field(d,"points_awarded"));out.scores.push_back(p);
   }
  }
 }
 require(f.eof()&&header&&!out.poses[0].empty()&&!out.poses[1].empty(),"incomplete capture");return out;
}
struct Geom{int bone;Shape local;};
struct Model{
 std::unique_ptr<mjModel,decltype(&mj_deleteModel)> m{nullptr,mj_deleteModel};
 std::array<Geom,12> strikes[2];std::array<Geom,3> targets[2];int bodies[2][30]{};int bone_for_body[256]{};
 explicit Model(const char* path,const char* expected){
  require(sha(path)==expected,"model hash mismatch");char error[2048]{};m.reset(mj_loadXML(path,nullptr,error,sizeof error));require(bool(m),error);require(m->nbody<256&&m->nq==72&&m->nv==70,"expected two G1 model");std::fill(bone_for_body,bone_for_body+256,-1);
  for(int side=0;side<2;side++)for(int bone=0;bone<30;bone++){
   std::string prefix=std::string(side?"opponent__":"player__")+BONES[bone]+"_";int match=-1;
   for(int i=1;i<m->nbody;i++){const char* name=mj_id2name(m.get(),mjOBJ_BODY,i);if(!name)continue;std::string s=name;
    if(s.rfind(prefix,0)==0&&s.size()>prefix.size()&&s.find_first_not_of("0123456789",prefix.size())==std::string::npos){require(match<0,"ambiguous body mapping");match=i;}}
   require(match>=0,"body mapping missing");bodies[side][bone]=match;bone_for_body[match]=bone;
  }
  const int strike_bones[6]={6,12,22,29,4,10},counts[6]={4,4,1,1,1,1},kinds[6]={0,0,2,2,1,1};
  for(int side=0;side<2;side++){
   int index=0;
   for(int limb=0;limb<6;limb++){int count=0;for(int g=0;g<m->ngeom;g++)if(m->geom_bodyid[g]==bodies[side][strike_bones[limb]]){require(index<12,"excess striker geoms");strikes[side][index]=geom(g);require(strikes[side][index].local.kind==kinds[limb]&&LIMB[index]==limb,"striker kind/group mismatch");index++;count++;}require(count==counts[limb],"striker count mismatch");}
   const std::string prefix=side?"opponent__":"player__";const char* target_names[]={"mjgeom_3021","mjgeom_3285","mjgeom_3064"};
   for(int i=0;i<3;i++){int g=mj_name2id(m.get(),mjOBJ_GEOM,(prefix+target_names[i]).c_str());require(g>=0,"target missing");targets[side][i]=geom(g);require(targets[side][i].local.kind==(i==2?1:2),"target kind mismatch");}
  }
 }
 Geom geom(int g)const{
  Geom out{};out.bone=bone_for_body[m->geom_bodyid[g]];require(out.bone>=0,"unmapped geom body");auto& s=out.local;
  s.kind=m->geom_type[g]==mjGEOM_SPHERE?0:m->geom_type[g]==mjGEOM_CAPSULE?1:m->geom_type[g]==mjGEOM_BOX?2:-1;require(s.kind>=0,"unsupported geom kind");double matrix[9];mju_quat2Mat(matrix,m->geom_quat+4*g);
  for(int k=0;k<3;k++){s.center[k]=float(m->geom_pos[3*g+k]);s.size[k]=float(m->geom_size[3*g+k]);require(std::isfinite(s.size[k])&&s.size[k]>=0,"bad geometry size");}
  for(int k=0;k<9;k++)s.axes[k]=float(matrix[k]);return out;
 }
};
Shape world(const Geom& geom,const Pose& pose){
 Shape out=geom.local;double r[9];mju_quat2Mat(r,pose.wxyz[geom.bone]);
 for(int i=0;i<3;i++){
  double p=pose.xyz[geom.bone][i];for(int k=0;k<3;k++)p+=r[3*i+k]*geom.local.center[k];out.center[i]=float(p);
  for(int j=0;j<3;j++){double x=0;for(int k=0;k<3;k++)x+=r[3*i+k]*geom.local.axes[3*k+j];out.axes[3*i+j]=float(x);}
 }return out;
}
float radius(const Shape& s){return s.kind==0?s.size[0]:s.kind==1?s.size[0]+s.size[1]:std::sqrt(s.size[0]*s.size[0]+s.size[1]*s.size[1]+s.size[2]*s.size[2]);}
Shape sphere_union(const Shape* shapes,int first,int end){
 Shape out{};out.kind=0;float lo[3]={INFINITY,INFINITY,INFINITY},hi[3]={-INFINITY,-INFINITY,-INFINITY};
 for(int i=first;i<end;i++)for(int k=0;k<3;k++){float r=radius(shapes[i]);lo[k]=std::min(lo[k],shapes[i].center[k]-r);hi[k]=std::max(hi[k],shapes[i].center[k]+r);}
 for(int k=0;k<3;k++)out.center[k]=.5f*(lo[k]+hi[k]);
 for(int i=first;i<end;i++){float d2=0;for(int k=0;k<3;k++){float d=out.center[k]-shapes[i].center[k];d2+=d*d;}out.size[0]=std::max(out.size[0],std::sqrt(d2)+radius(shapes[i]));}return out;
}
struct Query{Shape strikers[12],targets[3],old_strikers[6],old_targets[3];float hit[3];int kick;};
struct Result{unsigned long long pairs,compatible_pairs;int old_pairs,old_compatible;float point_striker,point_target;};
__device__ float point_distance(const Shape& s,const float* point){
 using namespace rek5_primitive::detail;Vec p=make(point);
 if(s.kind==0)return fmaxf(0,sqrtf(dot(sub(p,make(s.center)),sub(p,make(s.center))))-s.size[0]);
 if(s.kind==1){Vec a,b;endpoints(s,a,b);return fmaxf(0,sqrtf(point_segment2(p,a,b))-s.size[0]);}
 return sqrtf(point_aabb2(local(s,p),s.size));
}
__global__ void contacts(const Query* queries,Result* results,int n){
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=n)return;const auto& q=queries[row];Result out{};out.point_striker=out.point_target=INFINITY;
 for(int i=0;i<12;i++){
  const int limb=i<4?0:i<8?1:i-6;bool compatible=(q.kick!=0)==(limb<2||limb>=4);
  if(compatible)out.point_striker=fminf(out.point_striker,point_distance(q.strikers[i],q.hit));
  for(int j=0;j<3;j++)if(rek5_primitive::overlap(q.strikers[i],q.targets[j])){unsigned long long bit=1ull<<(3*i+j);out.pairs|=bit;if(compatible)out.compatible_pairs|=bit;}
 }
 for(int j=0;j<3;j++)out.point_target=fminf(out.point_target,point_distance(q.targets[j],q.hit));
 for(int i=0;i<6;i++)for(int j=0;j<3;j++)if(rek5_primitive::overlap(q.old_strikers[i],q.old_targets[j])){out.old_pairs++;if((q.kick!=0)==(i<2||i>=4))out.old_compatible++;}
 results[row]=out;
}
const Pose* preceding(const std::vector<Pose>& poses,double time){auto it=std::upper_bound(poses.begin(),poses.end(),time,[](double t,const Pose& p){return t<p.time;});return it==poses.begin()?nullptr:&*--it;}
struct Joined{Packet score,hit;int round;};
struct QueryMeta{int event;double lag,ages[2];int seq[2],frames[2];};
void linkage(const Model& model,const Capture& capture,int round,FILE* output){
 for(int side=0;side<2;side++)for(int bone=1;bone<30;bone++){
  int body=model.bodies[side][bone],parent=model.bone_for_body[model.m->body_parentid[body]];require(parent>=0,"missing bone parent");double rest=0;for(int k=0;k<3;k++)rest+=std::pow(model.m->body_pos[3*body+k],2);rest=std::sqrt(rest);if(rest<1e-8)continue;
  double sum=0,lo=INFINITY,hi=0;for(const auto& pose:capture.poses[side]){double d=0;for(int k=0;k<3;k++)d+=std::pow(pose.xyz[bone][k]-pose.xyz[parent][k],2);double ratio=std::sqrt(d)/rest;sum+=ratio;lo=std::min(lo,ratio);hi=std::max(hi,ratio);}
  std::fprintf(output,"{\"event\":\"link_length_check\",\"round\":%d,\"side\":%d,\"bone\":\"%s\",\"model_length_m\":%.9g,\"mean_ratio\":%.9g,\"min_ratio\":%.9g,\"max_ratio\":%.9g,\"samples\":%zu}\n",round,side,BONES[bone],rest,sum/capture.poses[side].size(),lo,hi,capture.poses[side].size());
 }
}
}
int main(int argc,char** argv){
 try{
  require(argc==8,"Usage: pose-contact-replay MODEL MODEL_SHA ROUND1 ROUND1_SHA ROUND2 ROUND2_SHA NEW_RESULTS_JSONL");
  Model model(argv[1],argv[2]);Capture captures[2]={read_capture(argv[3],argv[4]),read_capture(argv[5],argv[6])};
  std::unique_ptr<FILE,decltype(&std::fclose)> output(std::fopen(argv[7],"wx"),std::fclose);require(bool(output),"output must be new");
  std::vector<Joined> events;int unpaired_hits[2]={},unpaired_scores[2]={};
  for(int round=0;round<2;round++){
   const auto& c=captures[round];std::set<int> scored;
   for(const auto& h:c.hits){std::vector<const Packet*> matched;int hit_count=0;
    for(const auto& other:c.hits)if(other.frame==h.frame&&other.tick==h.tick)hit_count++;
    for(const auto& s:c.scores)if(s.frame==h.frame&&s.tick==h.tick&&std::abs(s.time-h.time)<=.01)matched.push_back(&s);
    if(matched.size()!=1||hit_count!=1){unpaired_hits[round]++;continue;}
    const auto& s=*matched[0];require(s.points==(h.kick?2:1),"paired score/contact category disagreement");require(scored.insert(s.seq).second,"reused score");events.push_back({s,h,round+1});
   }
   unpaired_scores[round]=int(c.scores.size()-scored.size());linkage(model,c,round+1,output.get());
  }
  std::vector<Query> queries;std::vector<QueryMeta> meta;
  for(size_t event=0;event<events.size();event++){
   const auto& e=events[event];const auto& c=captures[e.round-1];
   // Fixed, predeclared receipt-time scan. No geometry-dependent lag fitting.
   for(int step=-30;step<=30;step++){
    double lag=step/120.0,t=e.score.time+lag;const Pose* p[2]={preceding(c.poses[0],t),preceding(c.poses[1],t)};
    if(!p[0]||!p[1]||t-p[0]->time>.075||t-p[1]->time>.075)continue;
    Query q{};q.kick=e.hit.kick;for(int k=0;k<3;k++)q.hit[k]=e.hit.xyz[k];int side=e.score.scorer;
    for(int i=0;i<12;i++)q.strikers[i]=world(model.strikes[side][i],*p[side]);
    for(int i=0;i<3;i++){q.targets[i]=world(model.targets[side^1][i],*p[side^1]);q.old_targets[i]=sphere_union(q.targets,i,i+1);}
    const int offsets[7]={0,4,8,9,10,11,12};for(int i=0;i<6;i++)q.old_strikers[i]=sphere_union(q.strikers,offsets[i],offsets[i+1]);
    queries.push_back(q);meta.push_back({int(event),lag,{t-p[0]->time,t-p[1]->time},{p[0]->seq,p[1]->seq},{p[0]->frame,p[1]->frame}});
   }
  }
  require(!queries.empty(),"no pose queries");Query* gpu_queries=nullptr;Result* gpu_results=nullptr;checked(cudaMalloc(&gpu_queries,queries.size()*sizeof(Query)));checked(cudaMalloc(&gpu_results,queries.size()*sizeof(Result)));
  checked(cudaMemcpy(gpu_queries,queries.data(),queries.size()*sizeof(Query),cudaMemcpyHostToDevice));contacts<<<(queries.size()+63)/64,64>>>(gpu_queries,gpu_results,int(queries.size()));checked(cudaGetLastError());checked(cudaDeviceSynchronize());
  std::vector<Result> results(queries.size());checked(cudaMemcpy(results.data(),gpu_results,results.size()*sizeof(Result),cudaMemcpyDeviceToHost));checked(cudaFree(gpu_results));checked(cudaFree(gpu_queries));
  for(size_t i=0;i<queries.size();i++){
   const auto& m=meta[i];const auto& e=events[m.event];const auto& r=results[i];
   std::fprintf(output.get(),"{\"event\":\"pose_contact_query\",\"round\":%d,\"hit_sequence\":%d,\"score_sequence\":%d,\"scorer\":%d,\"points\":%d,\"is_kick\":%d,\"receipt_lag_seconds\":%.9g,\"pose_sequences\":[%d,%d],\"pose_unity_frames\":[%d,%d],\"pose_age_seconds\":[%.9g,%.9g],\"primitive_pair_mask\":%llu,\"part_compatible_pair_mask\":%llu,\"legacy_pair_count\":%d,\"legacy_part_compatible_count\":%d,\"hit_point_striker_distance_m\":%.9g,\"hit_point_target_distance_m\":%.9g}\n",e.round,e.hit.seq,e.score.seq,e.score.scorer,e.score.points,e.hit.kick,m.lag,m.seq[0],m.seq[1],m.frames[0],m.frames[1],m.ages[0],m.ages[1],r.pairs,r.compatible_pairs,r.old_pairs,r.old_compatible,r.point_striker,r.point_target);
  }
  require(std::fflush(output.get())==0,"output flush failed");
  std::printf("{\"event\":\"pose_contact_replay\",\"model_sha256\":\"%s\",\"capture_sha256\":[\"%s\",\"%s\"],\"rounds\":2,\"paired_strikes\":%zu,\"queries\":%zu,\"unpaired_hits\":[%d,%d],\"unpaired_scores\":[%d,%d],\"pose_counts\":[[%zu,%zu],[%zu,%zu]],\"lag_scan_seconds\":[-0.25,0.25],\"lag_step_seconds\":0.008333333333333333,\"max_pose_age_seconds\":0.075,\"contact_execution\":\"CUDA_static_overlap\",\"cpu_physics\":false,\"python_runtime\":false,\"authoritative_contact_time\":false,\"negative_outcome_labels\":0,\"canned_motion_used\":false,\"round2_fit\":false,\"authentic_parity\":false}\n",argv[2],captures[0].digest.c_str(),captures[1].digest.c_str(),events.size(),queries.size(),unpaired_hits[0],unpaired_hits[1],unpaired_scores[0],unpaired_scores[1],captures[0].poses[0].size(),captures[0].poses[1].size(),captures[1].poses[0].size(),captures[1].poses[1].size());return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"pose contact replay failed: %s\n",e.what());return 1;}
}
