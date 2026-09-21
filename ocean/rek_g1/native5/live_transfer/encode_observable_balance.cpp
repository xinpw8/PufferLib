// Offline/live client observation adapter only. No physics, policy or input dispatch.
#include "../../../../vendor/cJSON.h"
#include "../observable_balance.h"
#include "../action_cadence.h"
#include <openssl/evp.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <map>
#include <stdexcept>
#include <string>

namespace {
namespace balance=rek_observable_balance;
using Json=std::unique_ptr<cJSON,decltype(&cJSON_Delete)>;
constexpr const char* PROJECTION="client_pose_projection_v1";
// Existing declared client-request-duration gate, not measured server playback.
constexpr int MOVE_TICKS[]={35,27,31,45,32,45,157,145,158,139,134,138,73,75,68,71,103};
void need(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
const cJSON* raw(const cJSON* j,const char* key){return cJSON_GetObjectItemCaseSensitive(j,key);}
const cJSON* get(const cJSON* j,const char* key){auto* v=raw(j,key);need(v&&!cJSON_IsNull(v),std::string("unavailable:")+key);return v;}
const cJSON* optional(const cJSON* j,const char* key){auto* v=raw(j,key);return v&&!cJSON_IsNull(v)?v:nullptr;}
double number(const cJSON* j){need(cJSON_IsNumber(j)&&std::isfinite(j->valuedouble),"expected_finite_number");return j->valuedouble;}
int integer(const cJSON* j,int low,int high){double n=number(j);need(n>=low&&n<=high&&n==std::floor(n),"invalid_integer");return int(n);}
std::uint64_t tick(const cJSON* j){double n=number(j);need(n>0&&n<=9007199254740991.&&n==std::floor(n),"invalid_qpc_or_identity_integer");return std::uint64_t(n);}
bool boolean(const cJSON* j){need(cJSON_IsBool(j),"expected_boolean");return cJSON_IsTrue(j);}
std::string string(const cJSON* j){need(cJSON_IsString(j)&&j->valuestring,"expected_string");return j->valuestring;}
const cJSON* array(const cJSON* j,int n){need(cJSON_IsArray(j)&&cJSON_GetArraySize(j)==n,"array_length_mismatch");return j;}
bool digest_string(const std::string& s){return s.size()==64&&s.find_first_not_of("0123456789abcdef")==std::string::npos;}
void require_null(const cJSON* j,const char* key){need(cJSON_IsNull(raw(j,key)),std::string("expected_explicit_null:")+key);}
std::string sha(const void* data,size_t size){unsigned char out[32];unsigned n=0;need(EVP_Digest(data,size,out,&n,EVP_sha256(),nullptr)==1&&n==32,"sha256_failed");const char* hex="0123456789abcdef";std::string result;for(unsigned char b:out){result+=hex[b>>4];result+=hex[b&15];}return result;}
std::string file_sha(const std::string& path){std::ifstream f(path,std::ios::binary);need(bool(f),"model_read_failed");std::string bytes((std::istreambuf_iterator<char>(f)),{});need(!f.bad(),"model_read_failed");return sha(bytes.data(),bytes.size());}
Json object(){return Json(cJSON_CreateObject(),cJSON_Delete);}
void text(cJSON* j,const char* key,const std::string& value){cJSON_AddStringToObject(j,key,value.c_str());}
void num(cJSON* j,const char* key,double value){cJSON_AddNumberToObject(j,key,value);}
void flag(cJSON* j,const char* key,bool value){cJSON_AddBoolToObject(j,key,value);}
void emit(const cJSON* j){char* data=cJSON_PrintUnformatted(j);need(data,"json_print_failed");std::cout<<data<<'\n'<<std::flush;cJSON_free(data);}
template<class T,size_t N>cJSON* json_array(const std::array<T,N>& values){auto* out=cJSON_CreateArray();for(auto v:values)cJSON_AddItemToArray(out,cJSON_CreateNumber(double(v)));return out;}

struct Sample {
 balance::Snapshot snapshot{};std::uint64_t seq=0,ticks=0,frequency=0,request_ticks=0;
 std::string round;int phase=0,requested_move=-1,move=-1,desired=-1;
 bool stream=false,punching=false,native_busy_known=false,native_busy=false,sent=false,fallen[2]={};
 int falls[2]={};std::array<double,3> command{};std::array<int,33> mask{};
};
Sample parse_sample(const cJSON* source){
 need(string(get(source,"event"))=="g1_policy_state"&&string(get(source,"schema"))=="rek.g1_policy_source.v1","unsupported_source_schema");
 need(!boolean(get(source,"global_input_emitted")),"global_input_not_allowed");
 Sample s;s.seq=tick(get(source,"observation_sequence"));s.round=string(get(source,"round_identity_sha256"));need(digest_string(s.round),"invalid_round_identity");
 s.snapshot.actor_slot=integer(get(source,"local_slot"),0,1);s.phase=integer(get(source,"phase"),0,6);s.stream=boolean(get(source,"stream_active"));
 auto* clock=get(source,"clock");s.ticks=tick(get(clock,"qpc_ticks"));s.frequency=tick(get(clock,"qpc_frequency_hz"));
 auto* fighters=array(get(source,"fighters"),2);
 for(int slot=0;slot<2;slot++){
  auto* fighter=cJSON_GetArrayItem(fighters,slot);auto* xyz=array(get(fighter,"root_position_xyz"),3);auto* xyzw=array(get(fighter,"root_rotation_xyzw"),4);
  float p[3],q[4];for(int k=0;k<3;k++)p[k]=float(number(cJSON_GetArrayItem(xyz,k)));for(int k=0;k<4;k++)q[k]=float(number(cJSON_GetArrayItem(xyzw,k)));
  need(balance::from_unity_root(p,q,s.snapshot.fighter[slot]),"invalid_root_pose");
  s.snapshot.fighter[slot].joint_pose_available=0; // Physical/live joint correspondence is unverified.
  s.fallen[slot]=boolean(get(fighter,"fallen"));
  if(slot==s.snapshot.actor_slot){
   bool visual=boolean(get(fighter,"visual_only"));auto* runner=get(fighter,"runner");if(auto* move=optional(runner,"current_move_index"))s.move=integer(move,0,16);
   auto* input=get(source,"input");s.punching=boolean(get(input,"punching"));
   if(auto* busy=optional(input,"action_busy")){s.native_busy_known=true;s.native_busy=boolean(busy);}
   else if(!visual){s.native_busy_known=true;s.native_busy=s.punching;}
  }
 }
 auto* input=get(source,"input");auto* command=array(get(input,"velocity_command_xyz"),3);
 for(int k=0;k<3;k++)s.command[k]=number(cJSON_GetArrayItem(command,k));
 if(auto* value=optional(input,"requested_move_index"))s.requested_move=integer(value,0,16);
 if(auto* value=optional(input,"requested_move_qpc_ticks"))s.request_ticks=tick(value);
 if(auto* value=optional(input,"move_send_method_returned"))s.sent=boolean(value);
 if(auto* value=optional(input,"desired_action"))s.desired=integer(value,0,15);
 if(s.punching&&s.move<0&&s.requested_move>=0)s.move=s.requested_move;
 auto* round=get(source,"round");double duration=number(get(round,"duration")),remaining=number(get(round,"time_remaining"));
 need(duration>0&&duration<=86400&&remaining>=0&&remaining<=duration+.1,"invalid_round_clock");
 s.snapshot.round_duration_seconds=float(duration);s.snapshot.round_remaining_seconds=float(remaining);s.snapshot.round_active=boolean(get(round,"active"));
 s.snapshot.terminal=integer(get(round,"result_value"),0,4)!=0&&!s.snapshot.round_active;
 auto* points=array(get(round,"clean_hits"),2);auto* falls=array(get(round,"falls"),2);
 for(int k=0;k<2;k++){s.snapshot.points[k]=integer(cJSON_GetArrayItem(points,k),0,1000000);s.falls[k]=integer(cJSON_GetArrayItem(falls,k),0,1000000);}
 auto* mask=array(get(source,"action_mask"),33);int legal=0;
 for(int k=0;k<33;k++){auto* value=cJSON_GetArrayItem(mask,k);s.mask[k]=cJSON_IsBool(value)?boolean(value):integer(value,0,1);legal+=s.mask[k];}
 need(legal>0,"empty_source_action_mask");return s;
}

struct Referee {
 bool available=false;std::string reason,hash,round,transition,call_signature;
 std::uint64_t sequence=0,lifecycle=0,qpc=0,call_id=0;int unity_frame=0,count_mask=0;
 double unity_time=0,unity_unscaled=0,age=0;bool call_available=false,censored=false;
};
// Payload checks mirror validate_live_referee.cjs. The adapter checks the live
// bridge receipt, not an independent recorder file or current server execution.
Referee parse_referee(const cJSON* source,const Sample& s){
 auto* r=get(source,"referee");Referee out;out.round=s.round;
 need(string(get(r,"schema"))=="rek.g1_received_referee.v1","unsupported_referee_schema");out.available=boolean(get(r,"available"));out.reason=string(get(r,"reason"));need(!out.reason.empty(),"missing_referee_reason");
 const bool hooks=boolean(get(r,"observation_hooks_verified"));
 need(string(get(r,"source"))=="received_REK_FightState_33_byte_body"&&string(get(r,"provenance"))=="ApplyFightStateSnapshot_prefix_copy_postfix_client_mirror_verification"&&string(get(r,"authority_scope"))=="server_authored_packet_observed_on_client_not_server_current_state","referee_provenance_mismatch");
 need(number(get(r,"maximum_receipt_age_seconds"))==.5,"changed_receipt_freshness_budget");
 for(const char* key:{"server_tick","server_time","server_fight_epoch"})require_null(r,key);
 if(!out.available){
  for(const char* key:{"receipt_sequence","lifecycle","receipt_qpc_ticks","receipt_qpc_frequency_hz","receipt_unity_frame","receipt_unity_time","receipt_unity_unscaled_time","wire_body_sha256","wire_body_base64","count_mask","count_seconds","slot0_count_active","slot1_count_active","call_sequence","call_type","call_name","call_faller","call_points","call_observation_sequence","call_sequence_transition","call_history_censored","packet_phase","packet_round_number","packet_round_active","packet_round_redo","packet_round_knockout_occurred","packet_round_result"})require_null(r,key);
  need(!boolean(get(r,"call_available")),"unavailable_call_available");
  if(auto* age=optional(r,"receipt_age_seconds"))need(number(age)>=0,"invalid_unavailable_receipt_age");else require_null(r,"receipt_age_seconds");
  return out;
 }
 need(hooks&&out.reason=="received_snapshot_applied_and_bound","unverified_referee_hook");
 std::string encoded=string(get(r,"wire_body_base64"));out.hash=string(get(r,"wire_body_sha256"));need(digest_string(out.hash)&&encoded.size()==44&&encoded.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/")==std::string::npos,"invalid_wire_encoding");
 unsigned char b[36]={};need(EVP_DecodeBlock(b,reinterpret_cast<const unsigned char*>(encoded.data()),int(encoded.size()))==33,"invalid_wire_length");
 unsigned char canonical[45]={};need(EVP_EncodeBlock(canonical,b,33)==44&&encoded==reinterpret_cast<const char*>(canonical)&&sha(b,33)==out.hash,"wire_bytes_or_hash_mismatch");
 need(b[0]<=6&&b[2]<=1&&b[3]<=1&&b[12]<=1&&b[13]<=4&&b[25]<=3,"wire_schema_value_invalid");
 for(auto pair:{std::pair<const char*,int>{"packet_phase",0},{"packet_round_number",1},{"packet_round_result",13},{"count_mask",25},{"count_seconds",26},{"call_sequence",27}})need(integer(get(r,pair.first),0,255)==b[pair.second],std::string("decoded_byte_mismatch:")+pair.first);
 for(auto pair:{std::pair<const char*,int>{"packet_round_active",2},{"packet_round_redo",3},{"packet_round_knockout_occurred",12}})need(boolean(get(r,pair.first))==(b[pair.second]!=0),"decoded_boolean_mismatch");
 auto* round=get(source,"round");need(integer(get(round,"number"),0,255)==b[1]&&boolean(get(round,"redo"))==(b[3]!=0),"referee_source_round_or_redo_mismatch");
 need(boolean(get(r,"slot0_count_active"))==bool(b[25]&1)&&boolean(get(r,"slot1_count_active"))==bool(b[25]&2),"count_slot_bits_mismatch");out.count_mask=b[25];
 out.sequence=tick(get(r,"receipt_sequence"));out.lifecycle=tick(get(r,"lifecycle"));out.qpc=tick(get(r,"receipt_qpc_ticks"));need(tick(get(r,"receipt_qpc_frequency_hz"))==s.frequency,"receipt_clock_frequency_mismatch");
 need(s.ticks>=out.qpc,"receipt_from_future");out.age=double(s.ticks-out.qpc)/double(s.frequency);const double reported=number(get(r,"receipt_age_seconds"));
 need(out.age<=.5&&reported>=0&&reported<=.5&&std::abs(out.age-reported)<=1e-9,"receipt_freshness_or_age_mismatch");
 out.unity_frame=integer(get(r,"receipt_unity_frame"),0,2147483647);out.unity_time=number(get(r,"receipt_unity_time"));out.unity_unscaled=number(get(r,"receipt_unity_unscaled_time"));need(out.unity_time>=0&&out.unity_unscaled>=0,"invalid_receipt_unity_clock");
 out.transition=string(get(r,"call_sequence_transition"));need(out.transition=="empty_sequence"||out.transition=="initial_latched_call"||out.transition=="repeated_latched_call"||out.transition=="same_sequence_payload_changed_censored"||out.transition=="observed_255_to_1_wrap"||out.transition=="sequence_gap_censored"||out.transition=="sequence_decrease_censored"||out.transition=="changed_received_sequence","invalid_call_transition");
 out.call_available=boolean(get(r,"call_available"));
 if(!b[27]){need(!out.call_available&&out.transition=="empty_sequence","zero_call_not_empty");for(const char* key:{"call_type","call_name","call_faller","call_points","call_observation_sequence","call_history_censored"})require_null(r,key);}
 else{
  need(out.call_available&&integer(get(r,"call_type"),0,255)==b[28]&&integer(get(r,"call_faller"),-128,127)==int(static_cast<std::int8_t>(b[29]))&&integer(get(r,"call_points"),0,255)==b[30],"call_payload_mismatch");
  const char* names[]={"Slip","SlipEStop","Knockdown","BeatCount","Knockout","DoubleKnockdown","DoubleKnockout"};if(b[28]<7)need(string(get(r,"call_name"))==names[b[28]],"call_name_mismatch");else require_null(r,"call_name");
  out.call_id=tick(get(r,"call_observation_sequence"));out.censored=boolean(get(r,"call_history_censored"));need(out.transition!="empty_sequence","nonzero_call_empty_transition");
  if(out.transition=="initial_latched_call"||out.transition=="same_sequence_payload_changed_censored"||out.transition=="sequence_gap_censored"||out.transition=="sequence_decrease_censored")need(out.censored,"uncensored_call_discontinuity");
  if(out.transition=="changed_received_sequence"||out.transition=="observed_255_to_1_wrap")need(!out.censored,"censored_contiguous_call");
 }
 out.call_signature=std::to_string(b[27])+":"+std::to_string(b[28])+":"+std::to_string(b[29])+":"+std::to_string(b[30]);return out;
}

class ObservableEncoder {
 std::string model_hash,round,cadence_round;bool projected_busy,have_previous=false,have_sequence=false,have_referee=false;
 int stride;std::uint64_t last_sequence=0,round_key=0,ordinal=0,canceled_request=0;Sample previous{};Referee last_referee{};
 std::map<std::uint64_t,std::pair<std::string,bool>> call_identities;
 void bind_referee(const Referee& r){
  if(!r.available)return;
  if(have_referee){
   const auto& old=last_referee;need(r.sequence>=old.sequence&&r.lifecycle>=old.lifecycle&&r.qpc>=old.qpc,"receipt_order_regressed");
   if(r.sequence==old.sequence)need(r.lifecycle==old.lifecycle&&r.round==old.round&&r.qpc==old.qpc&&r.hash==old.hash&&r.unity_frame==old.unity_frame&&r.unity_time==old.unity_time&&r.unity_unscaled==old.unity_unscaled&&r.call_id==old.call_id&&r.transition==old.transition&&r.censored==old.censored,"receipt_sequence_reused_or_changed");
   if(r.round!=old.round)need(r.lifecycle>old.lifecycle,"referee_round_changed_without_lifecycle");
   if(r.call_available&&r.lifecycle==old.lifecycle&&r.sequence==old.sequence+1&&r.call_signature==old.call_signature)need(r.call_id==old.call_id&&r.transition=="repeated_latched_call"&&r.censored==old.censored,"repeated_call_identity_changed");
  }
  const bool same_lifecycle=have_referee&&r.lifecycle==last_referee.lifecycle;
  if(r.call_available&&same_lifecycle){auto found=call_identities.find(r.call_id);if(found!=call_identities.end())need(found->second==std::make_pair(r.call_signature,r.censored),"latched_call_identity_changed");}
  if(!same_lifecycle)call_identities.clear();
  if(r.call_available)call_identities.emplace(r.call_id,std::make_pair(r.call_signature,r.censored));
  last_referee=r;have_referee=true;
 }
public:
 ObservableEncoder(std::string hash,bool duration,int cadence):model_hash(std::move(hash)),projected_busy(duration),stride(cadence){need(stride==1||stride==5,"invalid_action_stride");}
 void reset(){have_previous=false;canceled_request=0;}
 void reset_cadence(){ordinal=0;cadence_round.clear();}
 Json unavailable(const std::string& reason){auto out=object();text(out.get(),"event","policy_observation");text(out.get(),"projection",PROJECTION);flag(out.get(),"ready",false);auto* reasons=cJSON_AddArrayToObject(out.get(),"unavailable");cJSON_AddItemToArray(reasons,cJSON_CreateString(reason.c_str()));return out;}
 Json manifest() const {
  auto out=object();text(out.get(),"event","projection_manifest");text(out.get(),"projection",PROJECTION);text(out.get(),"observation_schema",balance::kSchema);text(out.get(),"model_sha256",model_hash);
  text(out.get(),"model_usage","hash-only provenance; joint correspondence unavailable in both adapters");flag(out.get(),"candidate_physics_stepped",false);flag(out.get(),"authoritative_server_state",false);flag(out.get(),"legacy_checkpoint_compatible",false);
  text(out.get(),"busy_projection",projected_busy?"dispatched_request_v4_duration":"native_busy_required");if(projected_busy){auto* durations=cJSON_AddArrayToObject(out.get(),"move_duration_ticks");for(int n:MOVE_TICKS)cJSON_AddItemToArray(durations,cJSON_CreateNumber(n));num(out.get(),"duration_control_hz",50);}
  text(out.get(),"mask_semantics","source transport mask intersected with declared request-duration busy, held translation, and cadence gates; no server readiness or physical/live mask parity claim");
  text(out.get(),"history_semantics","actual preceding valid source observation; first/gap/nonpositive interval masked; same-round body displacement preserved");
  text(out.get(),"referee_semantics","fresh hash-bound received bridge receipt; source round/redo, slot bits, lifecycle/sequence validated; no independent recorder match or server-current-state claim");
  text(out.get(),"process_binding","authenticated bridge transport owns process binding; this source schema has no PID field; restart adapter for a new producer process");
  text(out.get(),"point_semantics","round.clean_hits is received awarded points, including referee awards; deltas have no inferred contact/fall cause");
  num(out.get(),"joint_pose_available",0);num(out.get(),"action_stride",stride);std::array<unsigned char,223> mask{};balance::feature_mask(mask.data());cJSON_AddItemToObject(out.get(),"structural_feature_mask",json_array(mask));
  auto* fields=cJSON_AddArrayToObject(out.get(),"fields");
  for(int column=0;column<223;column++){
   std::string kind="structural_padding",description="Unavailable in observable_balance.v1; numerical zero is padding, not an observed zero";
   if(column<172){const int local=column%86;const std::string fighter=column<86?"actor":"opponent";
    if(local<=2){kind="measured_transformed";description=fighter+" root position: Unity XYZ to common XZY";}
    else if(local<=6){kind="derived";description=fighter+" normalized sign-canonical root quaternion from Unity XYZW to common WXYZ (-w,x,z,y)";}
    else if(local<=9){kind="derived";description=fighter+" preceding-observation root finite difference; horizontal local +X/+Y or world vertical; history at203, heading at"+std::to_string(column-local+71);}
    else if(local==12){kind="derived";description=fighter+" wrapped horizontal +X heading difference / QPC interval; availability at"+std::to_string(column-local+76);}
    else if(local>=13&&local<=70){kind="unavailable_joint_correspondence";description=fighter+" projected joint angle/rate padding; joint_pose_available and joint_rate_available are explicitly0 in both adapters";}
    else if(local==71){kind="availability";description=fighter+" horizontal local +X heading is numerically observable";}
    else if(local==72){kind="derived";description=fighter+" root up-axis tilt / pi from normalized quaternion; no fall threshold";}
    else if(local==73){kind="measured_transformed";description=fighter+" absolute common-frame root height; no inferred floor or standing-height ratio";}
    else if(local==74||local==75){kind="availability";description=fighter+" joint pose/rate correspondence unavailable:0";}
    else if(local==76){kind="availability";description=fighter+" current and preceding horizontal headings and observation history are available";}
   }
   if(column==86){kind="derived";description="Horizontal measured root distance";}
   if(column==87){kind="derived";description="Actor-relative root bearing / pi; zero padding when heading unavailable or roots coincide";}
   if(column>=172&&column<=175){kind="derived";description="Actor horizontal-heading WXYZ quaternion; heading availability at71";}
   if(column==184){kind="measured";description="Absolute actor slot";}
   if(column==185){kind="derived";description="Observation phase4 at terminal,2 active,otherwise0";}
   if(column==188||column==189){kind="measured_scaled";description=column==188?"Observed round duration /120 seconds":"Observed remaining round time /120 seconds";}
   if(column==190||column==191){kind="measured";description=column==190?"Actor received awarded-point counter, including referee awards":"Opponent received awarded-point counter, including referee awards";}
   if(column==202){kind="availability";description="Validated fresh lifecycle-bound received referee count state is available";}
   if(column==203){kind="availability";description="Preceding same-round, same-perspective observation with positive QPC interval at most250ms is available";}
   if(column==204||column==205){kind="received";description=column==204?"Actor count-active bit; unavailable padding when202 is0":"Opponent count-active bit; unavailable padding when202 is0";}
   if(column==217||column==218){kind="derived";description=column==217?"Actor awarded-point delta over preceding observation; availability203, no cause inference":"Opponent awarded-point delta over preceding observation; availability203, no cause inference";}
   auto* field=cJSON_CreateObject();num(field,"index",column);text(field,"kind",kind);text(field,"source",description);flag(field,"structurally_available",mask[column]!=0);cJSON_AddItemToArray(fields,field);
  }
  return out;
 }
 Json process(const cJSON* source){
  Sample s=parse_sample(source);need(!have_sequence||s.seq>last_sequence,"nonmonotonic_observation_sequence");last_sequence=s.seq;have_sequence=true;
  Referee referee=parse_referee(source,s);bind_referee(referee);s.snapshot.referee_available=referee.available;s.snapshot.count_mask=referee.available?unsigned(referee.count_mask):0u;
  if(round!=s.round){round=s.round;round_key++;need(round_key!=0,"round_key_overflow");reset();}s.snapshot.round_key=round_key;
  const bool comparable=have_previous&&s.frequency==previous.frequency;
  const double dt=comparable?(s.ticks>=previous.ticks?double(s.ticks-previous.ticks):-double(previous.ticks-s.ticks))/double(s.frequency):0;
  s.snapshot.sample_seconds=comparable?previous.snapshot.sample_seconds+dt:0;
  if(!s.snapshot.terminal&&(!s.snapshot.round_active||s.phase!=1)){reset();return unavailable("round_not_active");}
  bool busy=s.native_busy_known?s.native_busy:false;double request_age=0;
  if(projected_busy){busy=false;if(s.requested_move>=0){need(s.request_ticks>0&&s.request_ticks<=s.ticks,"invalid_requested_move_qpc_ticks");request_age=double(s.ticks-s.request_ticks)/double(s.frequency);
   if(s.fallen[s.snapshot.actor_slot]||(have_previous&&s.falls[s.snapshot.actor_slot]>previous.falls[s.snapshot.actor_slot]))canceled_request=s.request_ticks;
   busy=s.sent&&s.request_ticks!=canceled_request&&request_age<double(MOVE_TICKS[s.requested_move])/50.;if(busy)s.move=s.requested_move;
  }}else if(!s.snapshot.terminal&&!s.native_busy_known){previous=s;have_previous=true;return unavailable("native_action_busy_unavailable_explicit_projection_required");}
  if(!s.snapshot.terminal&&busy&&s.move<0){previous=s;have_previous=true;return unavailable("active_move_route_unavailable");}
  std::array<float,223> obs{};auto status=balance::project(s.snapshot,comparable?&previous.snapshot:nullptr,obs.data());need(status==balance::kOk,"observable_balance_projection_status:"+std::to_string(int(status)));
  std::array<int,3> held{};for(int k=0;k<3;k++)held[k]=(s.command[k]>0)-(s.command[k]<0);
  if(s.desired>=0){held={0,0,0};switch(s.desired){case 2:held[0]=1;break;case 3:held[0]=-1;break;case 4:held[1]=1;break;case 5:held[1]=-1;break;case 6:held[2]=1;break;case 7:held[2]=-1;break;case 8:held={1,0,1};break;case 9:held={1,0,-1};break;case 10:held={-1,0,1};break;case 11:held={-1,0,-1};break;case 12:held={0,1,1};break;case 13:held={0,1,-1};break;case 14:held={0,-1,1};break;case 15:held={0,-1,-1};break;}}
  const bool translating=held[0]!=0||held[1]!=0;const std::uint64_t ready_ordinal=cadence_round==s.round?ordinal:0;
  if(!s.snapshot.terminal&&!rek_action_cadence::decision(stride,ready_ordinal))need(s.mask[0]!=0,"cadence_hold_not_source_legal");
  auto mask=s.mask;int legal=0;for(int k=0;k<33;k++){if(projected_busy&&busy&&k!=0&&k!=1&&k!=6&&k!=7)mask[k]=0;if(translating&&k>=16)mask[k]=0;if(!rek_action_cadence::permit(stride,ready_ordinal,k,true,s.snapshot.terminal))mask[k]=0;legal+=mask[k];}need(legal>0,"empty_projected_action_mask");
  auto out=object();text(out.get(),"event","policy_observation");text(out.get(),"projection",PROJECTION);flag(out.get(),"ready",true);
  auto* request=cJSON_AddObjectToObject(out.get(),"worker_request");text(request,"type","step");num(request,"seq",double(s.seq));text(request,"round_id",s.round);text(request,"observation_schema",balance::kSchema);flag(request,"terminal",s.snapshot.terminal);cJSON_AddItemToObject(request,"observation",json_array(obs));cJSON_AddItemToObject(request,"mask",json_array(mask));
  auto* p=cJSON_AddObjectToObject(out.get(),"provenance");text(p,"model_sha256",model_hash);num(p,"source_qpc_ticks",double(s.ticks));num(p,"source_qpc_frequency_hz",double(s.frequency));num(p,"source_native_phase",s.phase);num(p,"observed_delta_seconds",dt);flag(p,"history_available",obs[balance::kHistoryAvailable]!=0);num(p,"joint_pose_available",0);flag(p,"stream_active",s.stream);
  flag(p,"authoritative_server_state",false);flag(p,"candidate_physics_stepped",false);flag(p,"physics_parity_established",false);text(p,"server_playback_acceptance","unknown");text(p,"busy_projection",projected_busy?"dispatched_request_v4_duration":"native_controller_busy");flag(p,"projected_busy",busy);flag(p,"raw_local_punching",s.punching);if(s.native_busy_known)flag(p,"native_action_busy",s.native_busy);else cJSON_AddNullToObject(p,"native_action_busy");num(p,"requested_move_age_seconds",request_age);num(p,"requested_move_qpc_ticks",double(s.request_ticks));flag(p,"attack_mask_held_translation_blocked",translating);
  if(auto* source_mask=optional(source,"action_mask_source"))text(p,"source_action_mask_provenance",string(source_mask));else cJSON_AddNullToObject(p,"source_action_mask_provenance");
  text(p,"referee_status",referee.reason);flag(p,"received_referee_available",referee.available);flag(p,"independent_native_receipt_match_verified",false);if(referee.available){text(p,"received_wire_sha256",referee.hash);num(p,"received_lifecycle",double(referee.lifecycle));num(p,"received_receipt_sequence",double(referee.sequence));num(p,"received_age_seconds",referee.age);num(p,"received_absolute_count_mask",referee.count_mask);}
  if(stride!=1){auto* cadence=cJSON_AddObjectToObject(p,"action_cadence");text(cadence,"contract",rek_action_cadence::kContract);num(cadence,"stride",stride);num(cadence,"ready_ordinal",double(ready_ordinal));num(cadence,"phase",double(ready_ordinal%unsigned(stride)));flag(cadence,"decision_allowed",!s.snapshot.terminal&&rek_action_cadence::decision(stride,ready_ordinal));flag(cadence,"terminal_bypass",s.snapshot.terminal);}
  previous=s;have_previous=true;if(s.snapshot.terminal){reset();reset_cadence();}else{cadence_round=s.round;ordinal=ready_ordinal+1;}return out;
 }
};
}

#ifndef REK_OBSERVABLE_ENCODER_NO_MAIN
int main(int argc,char** argv){
 try{
  std::string model,projection,schema,busy;int stride=1;
  for(int i=1;i<argc;i++){std::string arg=argv[i];need(i+1<argc,"missing_cli_value");if(arg=="--model")model=argv[++i];else if(arg=="--projection")projection=argv[++i];else if(arg=="--observation-schema")schema=argv[++i];else if(arg=="--busy-projection")busy=argv[++i];else if(arg=="--action-stride")stride=rek_action_cadence::parse(argv[++i]);else throw std::runtime_error("unknown_cli_argument");}
  need(!model.empty()&&projection==PROJECTION&&schema==balance::kSchema,"explicit_model_projection_and_observable_balance_schema_required");need(busy.empty()||busy=="dispatched_request_v4_duration","unsupported_busy_projection");
  ObservableEncoder encoder(file_sha(model),!busy.empty(),stride);emit(encoder.manifest().get());std::string line;
  while(std::getline(std::cin,line)){
   try{need(line.size()<=1048576,"source_line_too_large");Json source(cJSON_ParseWithLengthOpts(line.c_str(),line.size()+1,nullptr,1),cJSON_Delete);need(source&&cJSON_IsObject(source.get()),"invalid_source_JSON");
    if(auto* type=optional(source.get(),"type")){std::string value=string(type);if(value=="close")break;if(value=="reset"){encoder.reset();encoder.reset_cadence();auto out=object();text(out.get(),"event","projection_reset");text(out.get(),"projection",PROJECTION);emit(out.get());continue;}}
    emit(encoder.process(source.get()).get());
   }catch(const std::exception& e){encoder.reset();emit(encoder.unavailable(e.what()).get());}
  }return 0;
 }catch(const std::exception& e){std::cerr<<"encode-observable-balance: "<<e.what()<<'\n';return 2;}
}
#endif
