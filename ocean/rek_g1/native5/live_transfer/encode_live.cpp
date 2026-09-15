// Client observation projection only. This program never steps a simulator.
#include "../../../../vendor/cJSON.h"
#include <mujoco/mujoco.h>
#include <openssl/evp.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr double PI=3.14159265358979323846;
constexpr const char* PROJECTION="client_pose_projection_v1";
constexpr const char* OBS_SCHEMA="rek.native5.scaled_polar_xy.v1";
using Json=std::unique_ptr<cJSON,decltype(&cJSON_Delete)>;
using Vec=std::array<double,3>;
using Quat=std::array<double,4>; // MuJoCo WXYZ
const char* BONES[]={"pelvis","left_hip_pitch_link","left_hip_roll_link","left_hip_yaw_link",
 "left_knee_link","left_ankle_pitch_link","left_ankle_roll_link","right_hip_pitch_link",
 "right_hip_roll_link","right_hip_yaw_link","right_knee_link","right_ankle_pitch_link",
 "right_ankle_roll_link","waist_yaw_link","waist_roll_link","torso_link",
 "left_shoulder_pitch_link","left_shoulder_roll_link","left_shoulder_yaw_link","left_elbow_link",
 "left_wrist_roll_link","left_wrist_pitch_link","left_wrist_yaw_link","right_shoulder_pitch_link",
 "right_shoulder_roll_link","right_shoulder_yaw_link","right_elbow_link","right_wrist_roll_link",
 "right_wrist_pitch_link","right_wrist_yaw_link"};
constexpr int EFFECTORS[]={6,12,22,29,4,10};
// Exact selected V4 puffer_env.cu/eval_worker.cpp defaults, 50 Hz control ticks.
constexpr int MOVE_TICKS[]={35,27,31,45,32,45,157,145,158,139,134,138,73,75,68,71,103};
void require(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
const cJSON* get(const cJSON* o,const char* name){auto* p=cJSON_GetObjectItemCaseSensitive(o,name);require(p&&!cJSON_IsNull(p),std::string("unavailable:")+name);return p;}
const cJSON* optional(const cJSON* o,const char* name){auto* p=cJSON_GetObjectItemCaseSensitive(o,name);return p&&!cJSON_IsNull(p)?p:nullptr;}
double num(const cJSON* v){require(cJSON_IsNumber(v)&&std::isfinite(v->valuedouble),"expected_finite_number");return v->valuedouble;}
int integer(const cJSON* v,int lo,int hi){double x=num(v);require(x==std::floor(x)&&x>=lo&&x<=hi,"integer_out_of_range");return int(x);}
bool boolean(const cJSON* v){require(cJSON_IsBool(v),"expected_boolean");return cJSON_IsTrue(v);}
std::string str(const cJSON* v){require(cJSON_IsString(v)&&v->valuestring,"expected_string");return v->valuestring;}
std::uint64_t exact_uint(const cJSON* v){double x=num(v);require(x>=0&&x<=9007199254740991.&&x==std::floor(x),"unsafe_integer");return std::uint64_t(x);}
const cJSON* array(const cJSON* v,int size){require(cJSON_IsArray(v)&&cJSON_GetArraySize(v)==size,"array_length_mismatch");return v;}
template<size_t N>std::array<double,N> values(const cJSON* v){array(v,N);std::array<double,N> a{};for(size_t i=0;i<N;i++)a[i]=num(cJSON_GetArrayItem(v,int(i)));return a;}
Json object(){return Json(cJSON_CreateObject(),cJSON_Delete);}
void text(cJSON* j,const char* k,const std::string& value){cJSON_AddStringToObject(j,k,value.c_str());}
void number(cJSON* j,const char* k,double value){cJSON_AddNumberToObject(j,k,value);}
void flag(cJSON* j,const char* k,bool value){cJSON_AddBoolToObject(j,k,value);}
void emit(const cJSON* j){char* s=cJSON_PrintUnformatted(j);require(s,"JSON_print_failed");std::cout<<s<<'\n'<<std::flush;cJSON_free(s);}
template<size_t N>cJSON* json_array(const std::array<double,N>& a){auto* j=cJSON_CreateArray();for(double x:a)cJSON_AddItemToArray(j,cJSON_CreateNumber(x));return j;}
double wrap(double x){return std::remainder(x,2*PI);}
Quat normalize(Quat q){double n=0;for(double x:q)n+=x*x;require(n>.8&&n<1.2,"invalid_quaternion_norm");for(double& x:q)x/=std::sqrt(n);return q;}
Quat multiply(const Quat& a,const Quat& b){return {a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]};}
Quat conjugate(Quat q){for(int i=1;i<4;i++)q[i]=-q[i];return q;}
Quat unity_quat(const cJSON* v){auto q=values<4>(v);return normalize({-q[3],q[0],q[2],q[1]});}
Vec unity_vec(const cJSON* v){auto p=values<3>(v);return {p[0],p[2],p[1]};}
double yaw(const Quat& q){return std::atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]*q[2]+q[3]*q[3]));}
std::string sha_file(const std::string& path){std::ifstream f(path,std::ios::binary);require(bool(f),"model_read_failed");auto* ctx=EVP_MD_CTX_new();require(ctx,"digest_allocation_failed");EVP_DigestInit_ex(ctx,EVP_sha256(),nullptr);char b[16384];while(f){f.read(b,sizeof b);if(f.gcount())EVP_DigestUpdate(ctx,b,size_t(f.gcount()));}require(f.eof(),"model_read_incomplete");unsigned char d[32];unsigned n=0;EVP_DigestFinal_ex(ctx,d,&n);EVP_MD_CTX_free(ctx);require(n==32,"digest_failed");const char* h="0123456789abcdef";std::string s;for(auto x:d){s+=h[x>>4];s+=h[x&15];}return s;}

struct Joint {int bone;Quat rest;Vec axis;std::string name;};
struct Calibration {
 std::array<Joint,29> joints;std::string model_sha;
 explicit Calibration(const std::string& path){
  model_sha=sha_file(path);char error[2048]={};
  std::unique_ptr<mjModel,decltype(&mj_deleteModel)> m(mj_loadXML(path.c_str(),nullptr,error,sizeof error),mj_deleteModel);
  require(bool(m),std::string("model_compile_failed:")+error);require(m->nu==58&&m->nq==72&&m->nv==70,"expected_two_G1_model");
  std::array<bool,30> seen{};
  for(int i=0;i<29;i++){
   int j=m->actuator_trnid[2*i];require(j>=0&&m->jnt_type[j]==mjJNT_HINGE,"actuator_not_hinge");int b=m->jnt_bodyid[j];
   const char* name=mj_id2name(m.get(),mjOBJ_BODY,b);require(name,"unnamed_joint_body");std::string full=name;int bone=-1;
   for(int k=1;k<30;k++){std::string prefix=std::string("player__")+BONES[k]+"_";if(full.rfind(prefix,0)==0&&full.size()>prefix.size()&&full.find_first_not_of("0123456789",prefix.size())==std::string::npos){require(bone<0,"ambiguous_bone_name");bone=k;}}
   require(bone>0&&!seen[bone],"unmapped_or_duplicate_bone:"+full);seen[bone]=true;
   auto& out=joints[i];out.bone=bone;out.name=BONES[bone];double norm=0;
   for(int k=0;k<4;k++)out.rest[k]=m->body_quat[4*b+k];for(int k=0;k<3;k++){out.axis[k]=m->jnt_axis[3*j+k];norm+=out.axis[k]*out.axis[k];}
   require(std::abs(norm-1)<1e-6,"nonunit_joint_axis");out.rest=normalize(out.rest);
  }
 }
 std::pair<double,double> project(const Joint& j,const Quat& measured) const {
  Quat relative=multiply(conjugate(j.rest),measured);double along=0;for(int k=0;k<3;k++)along+=relative[k+1]*j.axis[k];
  double norm=std::hypot(relative[0],along);require(norm>1e-8,"singular_hinge_pose_projection");
  double off=0;for(int k=0;k<3;k++){double e=relative[k+1]-along*j.axis[k];off+=e*e;}
  return {wrap(2*std::atan2(along,relative[0])),2*std::asin(std::min(1.,std::sqrt(off)))};
 }
};
struct Pose {Vec root;Quat rotation;double heading;std::array<double,29> q;std::array<Vec,6> effectors;double max_residual=0;};
struct Sample {
 std::uint64_t seq,ticks,frequency;std::string round;int side,phase,round_result,winner,fight_result,fight_winner;
 double duration,remaining;bool active,stream,punching,settled,fallen[2],falling[2];double tilt[2];int floor[2];
 std::array<Pose,2> poses;std::array<int,2> clean_hits,falls;std::array<int,33> mask;
 Vec command;int move=-1,requested_move=-1,desired_action=-1;bool move_is_requested=false;
 bool visual=false,native_busy_known=false,native_busy=false,move_send_returned=false;
 std::uint64_t request_ticks=0;
};
Sample parse(const cJSON* j,const Calibration& c){
 require(str(get(j,"schema"))=="rek.g1_policy_source.v1","unsupported_source_schema");
 require(!boolean(get(j,"global_input_emitted")),"global_input_not_allowed");Sample s;
 s.seq=exact_uint(get(j,"observation_sequence"));s.round=str(get(j,"round_identity_sha256"));require(s.round.size()==64&&s.round.find_first_not_of("0123456789abcdef")==std::string::npos,"invalid_round_identity");
 s.side=integer(get(j,"local_slot"),0,1);s.phase=integer(get(j,"phase"),0,6);s.stream=boolean(get(j,"stream_active"));
 auto* clock=get(j,"clock");s.ticks=exact_uint(get(clock,"qpc_ticks"));s.frequency=exact_uint(get(clock,"qpc_frequency_hz"));require(s.frequency>0,"invalid_clock_frequency");
 auto* fighters=array(get(j,"fighters"),2);
 for(int side=0;side<2;side++){
  auto* f=cJSON_GetArrayItem(fighters,side);auto& p=s.poses[side];p.root=unity_vec(get(f,"root_position_xyz"));p.rotation=unity_quat(get(f,"root_rotation_xyzw"));p.heading=yaw(p.rotation);
  auto* names=array(get(f,"bone_names"),30);auto* rotations=array(get(f,"bone_local_rotations_xyzw"),30);auto* positions=array(get(f,"bone_world_positions_xyz"),30);
  for(int k=0;k<30;k++)require(str(cJSON_GetArrayItem(names,k))==BONES[k],"g1_bone_signature_mismatch");
  for(int k=0;k<29;k++){auto result=c.project(c.joints[k],unity_quat(cJSON_GetArrayItem(rotations,c.joints[k].bone)));p.q[k]=result.first;p.max_residual=std::max(p.max_residual,result.second);}
  for(int k=0;k<6;k++)p.effectors[k]=unity_vec(cJSON_GetArrayItem(positions,EFFECTORS[k]));
  s.fallen[side]=boolean(get(f,"fallen"));s.falling[side]=boolean(get(f,"falling"));s.tilt[side]=num(get(f,"tilt_angle"));s.floor[side]=integer(get(f,"floor_contact_count"),0,1000);
  if(side==s.side){s.visual=boolean(get(f,"visual_only"));auto* runner=get(f,"runner");if(auto* move=optional(runner,"current_move_index"))s.move=integer(move,0,16);}
 }
 auto* input=get(j,"input");s.punching=boolean(get(input,"punching"));s.command=values<3>(get(input,"velocity_command_xyz"));
 if(auto* move=optional(input,"requested_move_index"))s.requested_move=integer(move,0,16);
 if(auto* requested=optional(input,"requested_move_qpc_ticks"))s.request_ticks=exact_uint(requested);
 if(auto* sent=optional(input,"move_send_method_returned"))s.move_send_returned=boolean(sent);
 if(auto* desired=optional(input,"desired_action"))s.desired_action=integer(desired,0,15);
 if(auto* busy=optional(input,"action_busy")){s.native_busy_known=true;s.native_busy=boolean(busy);}
 else if(!s.visual){s.native_busy_known=true;s.native_busy=s.punching;}
 if(s.punching&&s.move<0&&s.requested_move>=0){s.move=s.requested_move;s.move_is_requested=true;}
 auto* settled=get(input,"native_transition_settled");s.settled=boolean(get(settled,"forward"))&&boolean(get(settled,"backward"))&&boolean(get(settled,"strafe_left"))&&boolean(get(settled,"strafe_right"));
 auto* r=get(j,"round");s.duration=num(get(r,"duration"));s.remaining=num(get(r,"time_remaining"));s.active=boolean(get(r,"active"));
 require(s.duration>0&&s.duration<=86400&&s.remaining>=0&&s.remaining<=s.duration+.1,"invalid_round_clock");
 auto* hits=array(get(r,"clean_hits"),2);auto* falls=array(get(r,"falls"),2);
 for(int k=0;k<2;k++){s.clean_hits[k]=integer(cJSON_GetArrayItem(hits,k),0,1000000);s.falls[k]=integer(cJSON_GetArrayItem(falls,k),0,1000000);}
 s.round_result=integer(get(r,"result_value"),0,4);s.winner=integer(get(r,"winner_index"),-1,1);
 auto* fight=get(j,"fight");s.fight_result=integer(get(fight,"result_value"),0,2);s.fight_winner=integer(get(fight,"winner_index"),-1,1);
 auto* mask=array(get(j,"action_mask"),33);int count=0;for(int k=0;k<33;k++){auto* v=cJSON_GetArrayItem(mask,k);s.mask[k]=cJSON_IsBool(v)?boolean(v):integer(v,0,1);count+=s.mask[k];}require(count>0,"empty_action_mask");
 return s;
}
struct Field {std::string kind,source;};
class Encoder {
 const Calibration& calibration;bool have_previous=false;Sample previous;std::uint64_t last_seq=0;bool have_seq=false;
 bool request_duration_projection;std::uint64_t canceled_request_ticks=0;
 std::array<double,2> hit_age{},hit_speed{};std::array<bool,2> hit_valid{};
 std::array<Field,223> inventory;
 void describe(int i,const char* kind,const std::string& source){inventory[i]={kind,source};}
 void constant(int i,double value,std::array<double,223>& obs){obs[i]=value;}
public:
 explicit Encoder(const Calibration& c,bool projected_busy=false):calibration(c),request_duration_projection(projected_busy){
  for(int side=0;side<2;side++){
   int b=86*side;std::string p=side?"opponent":"actor";
   for(int k=0;k<86;k++)describe(b+k,"structural_constant","V4 unmodeled feature fixed to zero; this does not assert a measured game zero");
   for(int k=0;k<3;k++)describe(b+k,"derived",p+".root_position_xyz with Unity (x,y,z) -> (x,z,y)");
   for(int k=3;k<7;k++)describe(b+k,"derived",p+".root_rotation_xyzw -> WXYZ (-w,x,z,y)");
   for(int k:{7,8,12})describe(b+k,"derived",p+" timestamped root pose finite difference, actor-local horizontal axes/yaw");
   for(int k=0;k<29;k++){describe(b+13+k,"derived",p+"."+c.joints[k].name+" client local quaternion hinge twist projection");describe(b+42+k,"derived",p+"."+c.joints[k].name+" unwrapped projected angle / observed QPC delta");}
   describe(b+71,"structural_constant","V4 down flag is always zero: no compact knockdown transition exists");describe(b+72,"structural_constant","V4 synthetic tilt is zero because compact down flag is always zero");describe(b+73,"derived",p+" measured root height");
   describe(b+77,"structural_constant","literal 2 in V4 entity_value, not an authentic floor-contact measurement");describe(b+79,"structural_constant","V4 down flag is always zero; observed falling remains diagnostic");
  }
  describe(86,"derived","horizontal measured root distance");describe(87,"derived","measured target bearing relative to measured actor yaw, / pi");
  for(int i=172;i<223;i++)describe(i,"structural_constant","V4 unused/default feature explicitly fixed to zero");
  describe(172,"derived","cos(measured actor yaw / 2)");describe(175,"derived","sin(measured actor yaw / 2)");
  for(int i=176;i<=178;i++)describe(i,"derived","owned desired_action held category, or observed velocity-command sign outside owned control; zero during declared busy projection");
  describe(179,"derived","exact runner move pointer -> route; acknowledged client requested move only if pointer unavailable; idle/locomotion from command");
  describe(180,"derived","route in [1,6]");describe(181,"derived","no translation command and any native transition not settled");
  describe(182,"derived",projected_busy?"candidate-request-duration busy: exact dispatched request QPC plus selected V4 duration; actual playback unavailable":"native action_busy where available");describe(183,"derived",inventory[182].source);
  describe(184,"measured","local_slot");describe(185,"derived","authentic RoundActive phase 1 -> candidate active phase 2; completed round -> candidate terminal phase 4");describe(186,"structural_constant","independent episode feature 1; not cumulative authentic round number");
  describe(188,"derived","round.duration / 120");describe(189,"derived","round.time_remaining / 120");
  describe(190,"derived","actor round.clean_hits count projected into candidate points slot; authentic scoreboard formula unknown");describe(191,"derived","opponent round.clean_hits count projected into candidate points slot; authentic scoreboard formula unknown");
  describe(192,"structural_constant","V4 falls always zero: no compact knockdown transition exists; raw authentic falls remain diagnostic");describe(193,"structural_constant","V4 falls always zero: no compact knockdown transition exists; raw authentic falls remain diagnostic");
  describe(194,"derived","terminal and native winner equals actor");describe(195,"derived","terminal and native winner equals opponent");
  for(int i=196;i<=201;i++)describe(i,"derived","observation-window hit proxy from clean-hit counter delta and opposite fighter maximum observed effector speed; authoritative contact attribution/history unknown");
  for(int i=202;i<=205;i++)describe(i,"structural_constant","V4 down-state slots always zero: no compact knockdown transition exists");
  describe(209,"structural_constant","V4 half-second reset duration feature .5; authentic duration not asserted");
  describe(210,"measured","round.result_value");describe(211,"measured","round.winner_index");describe(212,"derived","round.result_value == knockout enum 2");
  describe(213,"measured","fight.result_value");describe(214,"measured","fight.winner_index");
  describe(217,"derived","actor clean_hits increase since previous source sample");describe(218,"derived","opponent clean_hits increase since previous source sample");
  for(int i=221;i<=222;i++)describe(i,"derived","sum of both measured clean-hit counter increases since previous sample");
 }
 void reset(){have_previous=false;hit_age={};hit_speed={};hit_valid={};canceled_request_ticks=0;}
 Json manifest() const {
  auto j=object();text(j.get(),"event","projection_manifest");text(j.get(),"projection",PROJECTION);text(j.get(),"observation_schema",OBS_SCHEMA);text(j.get(),"model_sha256",calibration.model_sha);
  flag(j.get(),"candidate_physics_stepped",false);flag(j.get(),"authoritative_server_state",false);
  text(j.get(),"busy_projection",request_duration_projection?"dispatched_request_v4_duration":"native_busy_required");
  if(request_duration_projection){auto* durations=cJSON_AddArrayToObject(j.get(),"move_duration_ticks");for(int n:MOVE_TICKS)cJSON_AddItemToArray(durations,cJSON_CreateNumber(n));number(j.get(),"duration_control_hz",50);text(j.get(),"duration_source","selected V4 puffer_env.cu and eval_worker.cpp explicit default table; candidate duration, not measured server playback");}
  text(j.get(),"score_semantics","measured clean-hit counts projected into policy points slots; no inferred knockout points");
  text(j.get(),"history_semantics","last-hit proxy history starts at observation window, not an assertion of no earlier hits");
  text(j.get(),"pose_semantics","joint angles are client bone twist projections using recovered model axes/rest orientations; server joint values unavailable");
  auto* fields=cJSON_AddArrayToObject(j.get(),"fields");for(int i=0;i<223;i++){auto* f=cJSON_CreateObject();number(f,"index",i);text(f,"kind",inventory[i].kind);text(f,"source",inventory[i].source);cJSON_AddItemToArray(fields,f);}
  auto* unavailable=cJSON_AddArrayToObject(j.get(),"authoritative_unavailable");for(const char* s:{"server_joint_positions","server_joint_velocities","contact_impulse","contact_limb_attribution","last_hit_before_observation_window","scoreboard_formula","server_command_acceptance"})cJSON_AddItemToArray(unavailable,cJSON_CreateString(s));
  return j;
 }
 Json unavailable(const std::string& why){auto j=object();text(j.get(),"event","policy_observation");text(j.get(),"projection",PROJECTION);flag(j.get(),"ready",false);auto* a=cJSON_AddArrayToObject(j.get(),"unavailable");cJSON_AddItemToArray(a,cJSON_CreateString(why.c_str()));return j;}
 Json process(const cJSON* source){
  Sample s=parse(source,calibration);require(!have_seq||s.seq>last_seq,"nonmonotonic_observation_sequence");last_seq=s.seq;have_seq=true;
  bool terminal=s.round_result!=0&&!s.active;
  if(!have_previous||s.round!=previous.round||s.side!=previous.side||s.frequency!=previous.frequency){reset();previous=s;have_previous=true;if(!terminal)return unavailable("derivative_warmup");}
  require(s.ticks>=previous.ticks,"nonmonotonic_source_clock");double dt=double(s.ticks-previous.ticks)/double(s.frequency);
  if(!terminal&&(dt<=0||dt>.25)){reset();previous=s;have_previous=true;return unavailable("source_sample_interval_outside_(0,250ms]");}
  if(!terminal&&(!s.active||s.phase!=1)){reset();return unavailable("round_not_active");}
  bool busy=s.native_busy_known?s.native_busy:false;bool uses_duration=false;double request_age=0;
  if(request_duration_projection){
   uses_duration=true;busy=false;
   if(s.requested_move>=0){require(s.request_ticks>0&&s.request_ticks<=s.ticks,"invalid_requested_move_qpc_ticks");request_age=double(s.ticks-s.request_ticks)/double(s.frequency);
    if(s.fallen[s.side]||s.falls[s.side]>previous.falls[s.side])canceled_request_ticks=s.request_ticks;
    busy=s.move_send_returned&&s.request_ticks!=canceled_request_ticks&&request_age<double(MOVE_TICKS[s.requested_move])/50.;
    if(busy){s.move=s.requested_move;s.move_is_requested=true;}
   }
  }else if(!terminal&&!s.native_busy_known){previous=s;return unavailable("native_action_busy_unavailable_explicit_projection_required");}
  if(!terminal&&busy&&s.move<0){previous=s;return unavailable("active_move_route_unavailable");}
  std::array<double,223> obs;obs.fill(std::numeric_limits<double>::quiet_NaN());
  for(int side=0;side<2;side++){
   int b=side*86,index=s.side^side;const auto& p=s.poses[index];const auto& old=previous.poses[index];
   for(int k:{9,10,11,74,75,76,78,80,81,82,83,84,85})constant(b+k,0,obs);
   for(int k=0;k<3;k++)obs[b+k]=p.root[k];for(int k=0;k<4;k++)obs[b+3+k]=p.rotation[k];
   double vx=dt>0?(p.root[0]-old.root[0])/dt:0,vy=dt>0?(p.root[1]-old.root[1])/dt:0;
   obs[b+7]=std::cos(p.heading)*vx+std::sin(p.heading)*vy;obs[b+8]=-std::sin(p.heading)*vx+std::cos(p.heading)*vy;obs[b+12]=dt>0?wrap(p.heading-old.heading)/dt:0;
   for(int k=0;k<29;k++){obs[b+13+k]=p.q[k];obs[b+42+k]=dt>0?wrap(p.q[k]-old.q[k])/dt:0;}
   obs[b+71]=0;obs[b+72]=0;obs[b+73]=p.root[2];obs[b+77]=2;obs[b+79]=0;
  }
  const auto& p=s.poses[s.side];const auto& enemy=s.poses[s.side^1];double dx=enemy.root[0]-p.root[0],dy=enemy.root[1]-p.root[1];
  obs[86]=std::hypot(dx,dy);obs[87]=wrap(std::atan2(dy,dx)-p.heading)/PI;
  for(int i:{173,174,187,202,203,204,205,206,207,208,215,216,219,220})constant(i,0,obs);
  obs[172]=std::cos(.5*p.heading);obs[175]=std::sin(.5*p.heading);
  auto sign=[](double value){return double((value>0)-(value<0));};
  Vec held{sign(s.command[0]),sign(s.command[1]),sign(s.command[2])};
  if(s.desired_action>=0){held={0,0,0};switch(s.desired_action){case 2:held[0]=1;break;case 3:held[0]=-1;break;case 4:held[1]=1;break;case 5:held[1]=-1;break;case 6:held[2]=1;break;case 7:held[2]=-1;break;case 8:held={1,0,1};break;case 9:held={1,0,-1};break;case 10:held={-1,0,1};break;case 11:held={-1,0,-1};break;case 12:held={0,1,1};break;case 13:held={0,1,-1};break;case 14:held={0,-1,1};break;case 15:held={0,-1,-1};break;}}
  for(int i=0;i<3;i++)obs[176+i]=busy?0:held[i];
  int route=0;
  if(busy)route=s.move<6?11+s.move:s.move<10?s.move+1:s.move+7;
  else if(held[0]!=0)route=held[0]>0?1:2;else if(held[1]!=0)route=held[1]>0?3:4;else if(held[2]!=0)route=held[2]>0?5:6;
  obs[179]=route;obs[180]=route>0&&route<7;obs[181]=held[0]==0&&held[1]==0&&!s.settled;obs[182]=obs[183]=busy;
  obs[184]=s.side;obs[185]=terminal?4:2;obs[186]=1;obs[188]=s.duration/120;obs[189]=s.remaining/120;
  std::array<int,2> delta{};for(int k=0;k<2;k++){require(s.clean_hits[k]>=previous.clean_hits[k],"clean_hit_counter_decreased_within_round");delta[k]=s.clean_hits[k]-previous.clean_hits[k];}
  for(int k=0;k<2;k++){
   if(hit_valid[k])hit_age[k]=std::min(120.,hit_age[k]+dt);
   if(delta[k^1]>0){double speed=0;for(int e=0;e<6;e++){double d2=0;for(int axis=0;axis<3;axis++){double d=s.poses[k^1].effectors[e][axis]-previous.poses[k^1].effectors[e][axis];d2+=d*d;}if(dt>0)speed=std::max(speed,std::sqrt(d2)/dt);}hit_valid[k]=true;hit_age[k]=0;hit_speed[k]=speed;}
  }
  for(int relative=0;relative<2;relative++){int k=s.side^relative;obs[190+relative]=s.clean_hits[k];obs[192+relative]=0;obs[194+relative]=terminal&&s.winner==k;obs[196+relative]=hit_valid[k];obs[198+relative]=hit_age[k];obs[200+relative]=hit_speed[k];obs[202+relative]=0;obs[204+relative]=0;obs[217+relative]=delta[k];}
  obs[209]=.5;obs[210]=s.round_result;obs[211]=s.winner;obs[212]=s.round_result==2;obs[213]=s.fight_result;obs[214]=s.fight_winner;obs[221]=obs[222]=delta[0]+delta[1];
  for(int i=0;i<223;i++)require(std::isfinite(obs[i])&&std::abs(obs[i])<=std::numeric_limits<float>::max(),"unavailable_or_nonfinite_feature:"+std::to_string(i));
  auto out=object();text(out.get(),"event","policy_observation");flag(out.get(),"ready",true);text(out.get(),"projection",PROJECTION);
  auto* request=cJSON_AddObjectToObject(out.get(),"worker_request");text(request,"type","step");number(request,"seq",double(s.seq));text(request,"round_id",s.round);text(request,"observation_schema",OBS_SCHEMA);flag(request,"terminal",terminal);
  cJSON_AddItemToObject(request,"observation",json_array(obs));auto* mask=cJSON_AddArrayToObject(request,"mask");for(int k=0;k<33;k++){int m=s.mask[k];if(uses_duration&&busy&&k!=0&&k!=1&&k!=6&&k!=7)m=0;cJSON_AddItemToArray(mask,cJSON_CreateNumber(m));}
  auto* provenance=cJSON_AddObjectToObject(out.get(),"provenance");text(provenance,"model_sha256",calibration.model_sha);number(provenance,"source_qpc_ticks",double(s.ticks));number(provenance,"source_qpc_frequency_hz",double(s.frequency));number(provenance,"source_native_phase",s.phase);number(provenance,"observed_delta_seconds",dt);flag(provenance,"stream_active",s.stream);flag(provenance,"route_uses_acknowledged_client_request",busy&&s.move_is_requested);flag(provenance,"authoritative_server_state",false);flag(provenance,"candidate_physics_stepped",false);text(provenance,"last_hit_semantics","observation-window counter/effector-speed proxy; earlier history and actual contact velocity unavailable");
  text(provenance,"busy_projection",uses_duration?"dispatched_request_v4_duration":"native_controller_busy");flag(provenance,"projected_busy",busy);flag(provenance,"raw_local_punching",s.punching);if(s.native_busy_known)flag(provenance,"native_action_busy",s.native_busy);else cJSON_AddNullToObject(provenance,"native_action_busy");number(provenance,"requested_move_age_seconds",request_age);number(provenance,"requested_move_qpc_ticks",double(s.request_ticks));text(provenance,"server_playback_acceptance","unknown");
  number(provenance,"actor_max_off_axis_rotation_radians",s.poses[s.side].max_residual);number(provenance,"opponent_max_off_axis_rotation_radians",s.poses[s.side^1].max_residual);
  auto* measured=cJSON_AddArrayToObject(provenance,"observed_source_values_excluded_from_structural_features");
  for(int k=0;k<2;k++){auto* f=cJSON_CreateObject();number(f,"slot",k);flag(f,"fallen",s.fallen[k]);flag(f,"falling",s.falling[k]);number(f,"tilt_degrees",s.tilt[k]);number(f,"floor_contact_count",s.floor[k]);number(f,"falls",s.falls[k]);cJSON_AddItemToArray(measured,f);}
  previous=s;if(terminal)reset();return out;
 }
};

void self_test(const Calibration& c){
 double worst=0;int checks=0;for(const auto& j:c.joints)for(double a:{-2.,-.3,0.,.4,2.}){Quat twist{std::cos(a/2),j.axis[0]*std::sin(a/2),j.axis[1]*std::sin(a/2),j.axis[2]*std::sin(a/2)};auto result=c.project(j,multiply(j.rest,twist));worst=std::max(worst,std::abs(wrap(result.first-a)));require(worst<1e-10&&result.second<1e-7,"synthetic_hinge_projection_failed");checks++;}
 auto j=object();text(j.get(),"event","self_test");flag(j.get(),"ok",true);number(j.get(),"hinge_projection_cases",checks);number(j.get(),"maximum_angle_error_radians",worst);text(j.get(),"model_sha256",c.model_sha);flag(j.get(),"simulation_stepped",false);emit(j.get());
}
}

#ifndef REK_ENCODER_NO_MAIN
int main(int argc,char** argv){
 try{
  std::string model,projection,busy_projection;bool test=false;for(int i=1;i<argc;i++){std::string arg=argv[i];if(arg=="--model"&&i+1<argc)model=argv[++i];else if(arg=="--projection"&&i+1<argc)projection=argv[++i];else if(arg=="--busy-projection"&&i+1<argc)busy_projection=argv[++i];else if(arg=="--self-test")test=true;else throw std::runtime_error("Usage: encode-live --model PRIVATE_XML --projection client_pose_projection_v1 [--busy-projection dispatched_request_v4_duration] [--self-test]");}
  require(!model.empty()&&projection==PROJECTION,"explicit_model_and_projection_required");require(busy_projection.empty()||busy_projection=="dispatched_request_v4_duration","unsupported_busy_projection");Calibration calibration(model);Encoder encoder(calibration,!busy_projection.empty());auto manifest=encoder.manifest();emit(manifest.get());if(test){self_test(calibration);return 0;}
  std::string line;while(std::getline(std::cin,line)){
   try{
    require(line.size()<=1048576,"source_line_too_large");Json j(cJSON_ParseWithLengthOpts(line.c_str(),line.size()+1,nullptr,1),cJSON_Delete);require(j&&cJSON_IsObject(j.get()),"invalid_source_JSON");
    if(auto* type=optional(j.get(),"type")){if(str(type)=="reset"){encoder.reset();auto out=object();text(out.get(),"event","projection_reset");text(out.get(),"projection",PROJECTION);emit(out.get());continue;}if(str(type)=="close")break;}
    auto out=encoder.process(j.get());emit(out.get());
   }catch(const std::exception& e){encoder.reset();auto out=encoder.unavailable(e.what());emit(out.get());}
  }
  return 0;
 }catch(const std::exception& e){std::cerr<<"encode-live: "<<e.what()<<'\n';return 2;}
}
#endif
