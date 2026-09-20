#define REK_ENCODER_NO_MAIN
#include "encode_live.cpp"

namespace {
Json fixture(const Calibration& c,int seq,int ticks,int actor_hits=0,bool busy=false){
 auto j=object();text(j.get(),"schema","rek.g1_policy_source.v1");number(j.get(),"observation_sequence",seq);text(j.get(),"round_identity_sha256",std::string(64,'a'));number(j.get(),"local_slot",0);number(j.get(),"phase",1);flag(j.get(),"stream_active",true);flag(j.get(),"global_input_emitted",false);
 auto* clock=cJSON_AddObjectToObject(j.get(),"clock");number(clock,"qpc_ticks",ticks);number(clock,"qpc_frequency_hz",1000);
 auto* fighters=cJSON_AddArrayToObject(j.get(),"fighters");
 for(int side=0;side<2;side++){
  auto* f=cJSON_CreateObject();cJSON_AddItemToArray(fighters,f);cJSON_AddItemToObject(f,"root_position_xyz",json_array(std::array<double,3>{double(side)*2,.77,0}));cJSON_AddItemToObject(f,"root_rotation_xyzw",json_array(std::array<double,4>{0,0,0,-1}));
  flag(f,"fallen",false);flag(f,"falling",false);flag(f,"visual_only",false);number(f,"tilt_angle",0);number(f,"floor_contact_count",2);
  auto* names=cJSON_AddArrayToObject(f,"bone_names");auto* rotations=cJSON_AddArrayToObject(f,"bone_local_rotations_xyzw");auto* positions=cJSON_AddArrayToObject(f,"bone_world_positions_xyz");
  for(int k=0;k<30;k++){
   cJSON_AddItemToArray(names,cJSON_CreateString(BONES[k]));Quat q{1,0,0,0};for(const auto& joint:c.joints)if(joint.bone==k)q=joint.rest;
   cJSON_AddItemToArray(rotations,json_array(std::array<double,4>{q[1],q[3],q[2],-q[0]}));
   cJSON_AddItemToArray(positions,json_array(std::array<double,3>{double(side)*2+double(ticks-1000)/1000,.5,0}));
  }
  auto* runner=cJSON_AddObjectToObject(f,"runner");cJSON_AddNullToObject(runner,"current_move_index");
 }
 auto* input=cJSON_AddObjectToObject(j.get(),"input");flag(input,"punching",busy);cJSON_AddItemToObject(input,"velocity_command_xyz",json_array(std::array<double,3>{0,0,0}));if(busy)number(input,"requested_move_index",7);else cJSON_AddNullToObject(input,"requested_move_index");
 auto* settled=cJSON_AddObjectToObject(input,"native_transition_settled");for(const char* name:{"forward","backward","strafe_left","strafe_right"})flag(settled,name,true);
 auto* round=cJSON_AddObjectToObject(j.get(),"round");number(round,"duration",20);number(round,"time_remaining",20-double(ticks-1000)/1000);flag(round,"active",true);number(round,"result_value",0);number(round,"winner_index",-1);cJSON_AddItemToObject(round,"clean_hits",json_array(std::array<double,2>{double(actor_hits),0}));cJSON_AddItemToObject(round,"falls",json_array(std::array<double,2>{0,0}));
 auto* fight=cJSON_AddObjectToObject(j.get(),"fight");number(fight,"result_value",0);number(fight,"winner_index",-1);
 auto* mask=cJSON_AddArrayToObject(j.get(),"action_mask");for(int i=0;i<33;i++)cJSON_AddItemToArray(mask,cJSON_CreateBool(!busy||i==0||i==1));return j;
}
double feature(const cJSON* result,int i){return num(cJSON_GetArrayItem(get(get(result,"worker_request"),"observation"),i));}
void tests(const Calibration& c){
 int checks=0;auto check=[&](bool ok,const char* why){require(ok,why);checks++;};
 Encoder encoder(c);auto manifest=encoder.manifest();check(cJSON_GetArraySize(get(manifest.get(),"fields"))==223,"inventory_dimensions");
 auto a=fixture(c,1,1000);auto result=encoder.process(a.get());check(!boolean(get(result.get(),"ready"))&&!optional(result.get(),"worker_request"),"initial_warmup_has_no_action");
 auto b=fixture(c,2,1020);result=encoder.process(b.get());check(boolean(get(result.get(),"ready")),"second_sample_ready");
 check(cJSON_GetArraySize(get(get(result.get(),"worker_request"),"observation"))==223,"observation_dimensions");
 for(int i=0;i<223;i++)check(std::isfinite(feature(result.get(),i)),"all_features_finite");
 check(feature(result.get(),86)==2&&feature(result.get(),87)==0,"polar_geometry");check(std::abs(feature(result.get(),188)-20./120)<1e-12,"timer_encoding");check(feature(result.get(),186)==1,"episode_round_feature");check(feature(result.get(),185)==2,"native_active1_maps_candidate2");
 for(int i=13;i<71;i++)check(std::abs(feature(result.get(),i))<1e-12,"rest_pose_projected_zero");
 Encoder structural_encoder(c);auto measured=fixture(c,1,1000);auto* f=cJSON_GetArrayItem(cJSON_GetObjectItemCaseSensitive(measured.get(),"fighters"),0);cJSON_ReplaceItemInObjectCaseSensitive(f,"floor_contact_count",cJSON_CreateNumber(0));cJSON_ReplaceItemInObjectCaseSensitive(f,"tilt_angle",cJSON_CreateNumber(12));cJSON_ReplaceItemInObjectCaseSensitive(f,"fallen",cJSON_CreateTrue());structural_encoder.process(measured.get());cJSON_ReplaceItemInObjectCaseSensitive(measured.get(),"observation_sequence",cJSON_CreateNumber(2));cJSON_ReplaceItemInObjectCaseSensitive(cJSON_GetObjectItemCaseSensitive(measured.get(),"clock"),"qpc_ticks",cJSON_CreateNumber(1020));result=structural_encoder.process(measured.get());check(feature(result.get(),77)==2&&feature(result.get(),71)==0&&feature(result.get(),72)==0,"V4_structural_constants_preserved");auto* diagnostics=cJSON_GetArrayItem(get(get(result.get(),"provenance"),"observed_source_values_excluded_from_structural_features"),0);check(num(get(diagnostics,"floor_contact_count"))==0&&num(get(diagnostics,"tilt_degrees"))==12&&boolean(get(diagnostics,"fallen")),"real_excluded_values_retained");
 auto hit=fixture(c,3,1040,1);result=encoder.process(hit.get());check(feature(result.get(),190)==1&&feature(result.get(),217)==1,"clean_hit_projection");check(feature(result.get(),196)==0&&feature(result.get(),197)==1,"hit_recipient_attribution_proxy");check(std::abs(feature(result.get(),201)-1)<1e-12,"observed_effector_speed_proxy");
 auto busy=fixture(c,4,1060,1,true);result=encoder.process(busy.get());check(feature(result.get(),179)==8,"move7_route8");check(boolean(get(get(result.get(),"provenance"),"route_uses_acknowledged_client_request")),"requested_route_labeled");
 auto unknown=fixture(c,5,1080,1,true);cJSON_DeleteItemFromObjectCaseSensitive(cJSON_GetObjectItemCaseSensitive(unknown.get(),"input"),"requested_move_index");result=encoder.process(unknown.get());check(!boolean(get(result.get(),"ready"))&&!optional(result.get(),"worker_request"),"unknown_busy_route_no_fallback");
 auto missing=fixture(c,6,1100,1);cJSON_DeleteItemFromObjectCaseSensitive(cJSON_GetArrayItem(cJSON_GetObjectItemCaseSensitive(missing.get(),"fighters"),0),"bone_world_positions_xyz");bool rejected=false;try{encoder.process(missing.get());}catch(const std::exception&){rejected=true;}check(rejected,"missing_measurement_rejected");
 auto backwards=fixture(c,6,1050,1);rejected=false;try{encoder.process(backwards.get());}catch(const std::exception&){rejected=true;}check(rejected,"backwards_clock_rejected");
 Encoder terminal_encoder(c);auto end=fixture(c,1,1000);auto* round=cJSON_GetObjectItemCaseSensitive(end.get(),"round");cJSON_ReplaceItemInObjectCaseSensitive(round,"active",cJSON_CreateFalse());cJSON_ReplaceItemInObjectCaseSensitive(round,"result_value",cJSON_CreateNumber(1));cJSON_ReplaceItemInObjectCaseSensitive(round,"winner_index",cJSON_CreateNumber(0));result=terminal_encoder.process(end.get());check(boolean(get(get(result.get(),"worker_request"),"terminal")),"terminal_no_inference_ack");
 auto next=fixture(c,2,1020);cJSON_ReplaceItemInObjectCaseSensitive(next.get(),"round_identity_sha256",cJSON_CreateString(std::string(64,'b').c_str()));result=terminal_encoder.process(next.get());check(!boolean(get(result.get(),"ready")),"new_round_warmup");
 auto next2=fixture(c,3,1040);cJSON_ReplaceItemInObjectCaseSensitive(next2.get(),"round_identity_sha256",cJSON_CreateString(std::string(64,'b').c_str()));result=terminal_encoder.process(next2.get());check(boolean(get(result.get(),"ready"))&&feature(result.get(),197)==0,"new_round_clears_proxy_history");
 auto duplicate=fixture(c,3,1060);rejected=false;try{terminal_encoder.process(duplicate.get());}catch(const std::exception&){rejected=true;}check(rejected,"duplicate_sequence_rejected");
 auto long_gap=fixture(c,4,2000);cJSON_ReplaceItemInObjectCaseSensitive(long_gap.get(),"round_identity_sha256",cJSON_CreateString(std::string(64,'b').c_str()));result=terminal_encoder.process(long_gap.get());check(!boolean(get(result.get(),"ready")),"stale_interval_rewarms");
 Encoder locked(c,true);auto dispatched=[&](int seq,int ticks,bool sent,int request_tick=1000){auto r=fixture(c,seq,ticks);auto* f=cJSON_GetArrayItem(cJSON_GetObjectItemCaseSensitive(r.get(),"fighters"),0);cJSON_ReplaceItemInObjectCaseSensitive(f,"visual_only",cJSON_CreateTrue());auto* input=cJSON_GetObjectItemCaseSensitive(r.get(),"input");cJSON_AddNullToObject(input,"action_busy");cJSON_ReplaceItemInObjectCaseSensitive(input,"requested_move_index",cJSON_CreateNumber(7));number(input,"requested_move_qpc_ticks",request_tick);flag(input,"move_send_method_returned",sent);number(input,"desired_action",6);return r;};
 auto lock=dispatched(1,1000,false);locked.process(lock.get());lock=dispatched(2,1020,true);result=locked.process(lock.get());check(feature(result.get(),182)==1&&feature(result.get(),179)==8,"dispatched_duration_sets_projected_busy_and_route");check(cJSON_IsNull(get(result.get(),"provenance")->child)==false,"provenance_object_exists");check(cJSON_IsNull(cJSON_GetObjectItemCaseSensitive(get(result.get(),"provenance"),"native_action_busy")),"raw_busy_remains_unknown");check(!boolean(get(get(result.get(),"provenance"),"raw_local_punching")),"raw_punching_stays_false");
 auto* projected_mask=get(get(result.get(),"worker_request"),"mask");for(int k=0;k<33;k++)check(num(cJSON_GetArrayItem(projected_mask,k))==double(k==0||k==1||k==6||k==7),"busy_mask_matches_candidate");
 int seq=3;for(int ticks=1220;ticks<3920;ticks+=200){lock=dispatched(seq++,ticks,true);result=locked.process(lock.get());}
 lock=dispatched(seq++,3920,true);result=locked.process(lock.get());check(feature(result.get(),182)==0&&feature(result.get(),179)==5&&feature(result.get(),178)==1,"duration_expires_restores_held_yaw");
 lock=dispatched(seq++,3940,true,3930);result=locked.process(lock.get());check(feature(result.get(),182)==1,"same_move_new_request_restarts_projection");
 auto* death=cJSON_GetArrayItem(cJSON_GetObjectItemCaseSensitive(lock.get(),"fighters"),0);cJSON_ReplaceItemInObjectCaseSensitive(death,"fallen",cJSON_CreateTrue());cJSON_ReplaceItemInObjectCaseSensitive(lock.get(),"observation_sequence",cJSON_CreateNumber(seq++));cJSON_ReplaceItemInObjectCaseSensitive(cJSON_GetObjectItemCaseSensitive(lock.get(),"clock"),"qpc_ticks",cJSON_CreateNumber(3960));result=locked.process(lock.get());check(feature(result.get(),182)==0,"measured_death_cancels_request_lock");
 Encoder unknown_busy(c);auto unknown1=dispatched(1,1000,true);unknown_busy.process(unknown1.get());auto unknown2=dispatched(2,1020,true);result=unknown_busy.process(unknown2.get());check(!boolean(get(result.get(),"ready")),"unknown_native_busy_requires_opt_in");
 // A visual client's transient zero VelocityCommand is not a release of the
 // bridge-owned held input. Do not offer attacks that dispatch will reject.
 for(int test_category=0;test_category<16;test_category++){
  const int desired=(test_category+2)%16;
  Encoder held_encoder(c);auto first=fixture(c,1,1000);held_encoder.process(first.get());
  auto held_source=fixture(c,2,1020);auto* input=cJSON_GetObjectItemCaseSensitive(held_source.get(),"input");
  number(input,"desired_action",desired);result=held_encoder.process(held_source.get());
  const bool translation=(desired>=2&&desired<=5)||desired>=8;
  const auto* mask=get(get(result.get(),"worker_request"),"mask");
  for(int k=0;k<33;k++)check(num(cJSON_GetArrayItem(mask,k))==double(k<16||!translation),"held_translation_blocks_attacks_despite_zero_native_velocity");
  check(boolean(get(get(result.get(),"provenance"),"attack_mask_held_translation_blocked"))==translation,"held_mask_provenance");
 }
 // Mask intersection must not open a source-disallowed attack on release/yaw.
 Encoder release_encoder(c);auto first_release=fixture(c,1,1000);release_encoder.process(first_release.get());
 auto release=fixture(c,2,1020);number(cJSON_GetObjectItemCaseSensitive(release.get(),"input"),"desired_action",6);
 cJSON_ReplaceItemInArray(cJSON_GetObjectItemCaseSensitive(release.get(),"action_mask"),16,cJSON_CreateFalse());
 result=release_encoder.process(release.get());
 check(num(cJSON_GetArrayItem(get(get(result.get(),"worker_request"),"mask"),16))==0,"source_restriction_preserved");
 // V2 changes only the previously unused column. Desired action is owned
 // retained state: policy hold0 must not be substituted for a source category.
 for(int desired=1;desired<=15;desired++){
  Encoder legacy(c,true),v2(c,true,true);
  auto initial=dispatched(1,1000,false);legacy.process(initial.get());v2.process(initial.get());
  auto sample=dispatched(2,1020,true);
  cJSON_ReplaceItemInObjectCaseSensitive(cJSON_GetObjectItemCaseSensitive(sample.get(),"input"),"desired_action",cJSON_CreateNumber(desired));
  auto old=legacy.process(sample.get()),newer=v2.process(sample.get());
  for(int i=0;i<223;i++)check(feature(newer.get(),i)==(i==187?rek_owned_yaw::desired_yaw(desired):feature(old.get(),i)),"v2_changes_only_owned_yaw_column");
  check(feature(old.get(),187)==0&&feature(newer.get(),178)==0,"legacy_held_yaw_feature_remains_zero_during_busy");
  const auto* oldmask=get(get(old.get(),"worker_request"),"mask");const auto* newmask=get(get(newer.get(),"worker_request"),"mask");
  for(int i=0;i<33;i++)check(num(cJSON_GetArrayItem(oldmask,i))==num(cJSON_GetArrayItem(newmask,i)),"v2_masks_unchanged");
  check(str(get(get(newer.get(),"worker_request"),"observation_schema"))==rek_owned_yaw::kSchema,"v2_schema_published");
  // No new request between these two snapshots: retained desired state stays visible.
  cJSON_ReplaceItemInObjectCaseSensitive(sample.get(),"observation_sequence",cJSON_CreateNumber(3));
  cJSON_ReplaceItemInObjectCaseSensitive(cJSON_GetObjectItemCaseSensitive(sample.get(),"clock"),"qpc_ticks",cJSON_CreateNumber(1040));
  newer=v2.process(sample.get());check(feature(newer.get(),187)==rek_owned_yaw::desired_yaw(desired),"retained_desired_yaw_not_current_action");
 }
 // Outside projected busy, the old held-yaw field remains sufficient and the
 // additive column is zero. A retained snapshot corresponds to hold0; release1
 // is a changed owned source value, not a transient native command zero.
 for(int desired:{6,7}){
  Encoder v2(c,false,true);auto initial=fixture(c,1,1000);
  number(cJSON_GetObjectItemCaseSensitive(initial.get(),"input"),"desired_action",desired);v2.process(initial.get());
  auto sample=fixture(c,2,1020);number(cJSON_GetObjectItemCaseSensitive(sample.get(),"input"),"desired_action",desired);
  auto nonbusy=v2.process(sample.get());
  check(feature(nonbusy.get(),187)==0&&feature(nonbusy.get(),178)==rek_owned_yaw::desired_yaw(desired),"nonbusy_yaw_keeps_legacy_field_and_zero_pending");
 }
 Encoder transition_v2(c,true,true);auto retained=dispatched(1,1000,false);transition_v2.process(retained.get());
 retained=dispatched(2,1020,true);auto transition=transition_v2.process(retained.get());check(feature(transition.get(),187)==1,"busy_yaw_requested");
 retained=dispatched(3,1040,true);transition=transition_v2.process(retained.get());check(feature(transition.get(),187)==1,"hold_retains_owned_yaw");
 retained=dispatched(4,1060,true);cJSON_ReplaceItemInObjectCaseSensitive(cJSON_GetObjectItemCaseSensitive(retained.get(),"input"),"desired_action",cJSON_CreateNumber(1));
 transition=transition_v2.process(retained.get());check(feature(transition.get(),187)==0&&feature(transition.get(),182)==1,"release_clears_owned_yaw_without_ending_busy");
 for(int invalid:{-1,0,16}){
  Encoder v2(c,true,true);auto sample=dispatched(1,1000,false);auto* input=cJSON_GetObjectItemCaseSensitive(sample.get(),"input");
  if(invalid<0)cJSON_DeleteItemFromObjectCaseSensitive(input,"desired_action");else cJSON_ReplaceItemInObjectCaseSensitive(input,"desired_action",cJSON_CreateNumber(invalid));
  bool bad=false;try{v2.process(sample.get());}catch(const std::exception&){bad=true;}check(bad,"v2_unknown_owned_intent_rejected");
 }
 Encoder terminal_v2(c,true,true);auto terminal_source=fixture(c,1,1000);auto* terminal_round=cJSON_GetObjectItemCaseSensitive(terminal_source.get(),"round");
 cJSON_ReplaceItemInObjectCaseSensitive(terminal_round,"active",cJSON_CreateFalse());cJSON_ReplaceItemInObjectCaseSensitive(terminal_round,"result_value",cJSON_CreateNumber(1));
 cJSON_ReplaceItemInObjectCaseSensitive(terminal_source.get(),"stream_active",cJSON_CreateFalse());
 auto terminal_result=terminal_v2.process(terminal_source.get());check(feature(terminal_result.get(),187)==0,"terminal_without_owned_intent_is_zero_no_action");
 check(!rek_owned_yaw::enabled(nullptr)&&!rek_owned_yaw::enabled(rek_owned_yaw::kLegacySchema)&&rek_owned_yaw::enabled(rek_owned_yaw::kSchema),"explicit_schema_opt_in");
 bool bad_schema=false;try{rek_owned_yaw::enabled("unknown");}catch(const std::exception&){bad_schema=true;}check(bad_schema,"unknown_schema_rejected");
 // Cadence changes only the learner support, never the 223 feature values.
 auto serialized=[](const cJSON* j){char* p=cJSON_PrintUnformatted(j);require(p,"serialize_failed");std::string s=p;cJSON_free(p);return s;};
 auto cadence_ordinal=[](const cJSON* j){return exact_uint(get(get(get(j,"provenance"),"action_cadence"),"ready_ordinal"));};
 auto mask_value=[](const cJSON* j,int k){return num(cJSON_GetArrayItem(get(get(j,"worker_request"),"mask"),k));};
 check(rek_action_cadence::parse(nullptr)==1&&rek_action_cadence::parse("1")==1&&rek_action_cadence::parse("5")==5,"cadence_explicit_opt_in");
 for(const char* invalid:{"","0","2","-1","5.0"," 5","5junk"}){bool bad=false;try{rek_action_cadence::parse(invalid);}catch(const std::exception&){bad=true;}check(bad,"invalid_stride_rejected");}
 for(int tick=0;tick<151;tick++)for(int k=0;k<33;k++){
  check(rek_action_cadence::permit(1,tick,k),"stride1_unrestricted");
  check(rek_action_cadence::permit(5,tick,k)==(tick%5==0||k==0),"native_tick_cadence_contract");
  check(rek_action_cadence::permit(5,tick,k,false),"opponent_and_bot_unrestricted");
  check(rek_action_cadence::permit(5,tick,k,true,true),"terminal_mask_bypass");
 }
 Encoder ordinary(c),explicit_one(c,false,false,1),fifth(c,false,false,5);
 check(serialized(ordinary.manifest().get())==serialized(explicit_one.manifest().get()),"default_manifest_byte_identical_to_stride1");
 check(!optional(ordinary.manifest().get(),"action_cadence"),"default_manifest_has_no_new_fields");
 for(int row=0;row<18;row++){
  // Source IDs skip and QPC intervals jitter. Neither is the decision clock.
  auto sample=fixture(c,100+row*3,1000+row*23,0,row>=8&&row<12);
  if(row==6)cJSON_ReplaceItemInArray(cJSON_GetObjectItemCaseSensitive(sample.get(),"action_mask"),17,cJSON_CreateFalse());
  auto baseline=ordinary.process(sample.get()),one=explicit_one.process(sample.get()),five=fifth.process(sample.get());
  check(serialized(baseline.get())==serialized(one.get()),"explicit_stride1_byte_identical_all_outputs");
  if(!row){check(!boolean(get(five.get(),"ready")),"cadence_warmup_no_ready_row");continue;}
  const int ordinal=row-1;check(cadence_ordinal(five.get())==unsigned(ordinal),"ordinal_counts_only_ready_rows");
  for(int i=0;i<223;i++)check(feature(five.get(),i)==feature(baseline.get(),i),"cadence_features_unchanged");
  for(int k=0;k<33;k++)check(mask_value(five.get(),k)==mask_value(baseline.get(),k)*double(ordinal%5==0||k==0),"cadence_intersects_existing_mask_even_busy");
 }
 // Feature-only resets and unavailable-only paths do not reset the worker RNN.
 Encoder history(c,false,false,5);auto initial_history=fixture(c,1,1000);history.process(initial_history.get());
 auto h=fixture(c,2,1020);auto hr=history.process(h.get());check(cadence_ordinal(hr.get())==0,"initial_ready_phase_zero");
 h=fixture(c,3,1040,0,true);cJSON_DeleteItemFromObjectCaseSensitive(cJSON_GetObjectItemCaseSensitive(h.get(),"input"),"requested_move_index");
 hr=history.process(h.get());check(!boolean(get(hr.get(),"ready")),"unavailable_does_not_emit");
 h=fixture(c,4,1060);hr=history.process(h.get());check(cadence_ordinal(hr.get())==1,"unavailable_does_not_consume_or_reset_phase");
 h=fixture(c,5,2000);hr=history.process(h.get());check(!boolean(get(hr.get(),"ready")),"cadence_long_gap_rewarms_features");
 h=fixture(c,6,2020);hr=history.process(h.get());check(cadence_ordinal(hr.get())==2,"derivative_gap_preserves_worker_phase");
 history.reset();h=fixture(c,7,2040);hr=history.process(h.get());check(!boolean(get(hr.get(),"ready")),"existing_catch_reset_rewarms");
 h=fixture(c,8,2060);hr=history.process(h.get());check(cadence_ordinal(hr.get())==3,"feature_reset_preserves_phase");
 h=fixture(c,9,2080);cJSON_ReplaceItemInArray(cJSON_GetObjectItemCaseSensitive(h.get(),"action_mask"),0,cJSON_CreateFalse());
 bool nohold=false;try{history.process(h.get());}catch(const std::exception&){nohold=true;}check(nohold,"hold_only_row_never_invents_source_legality");
 history.reset();h=fixture(c,10,2100);history.process(h.get());h=fixture(c,11,2120);hr=history.process(h.get());check(cadence_ordinal(hr.get())==4,"failed_output_does_not_consume_phase");
 h=fixture(c,12,2140);hr=history.process(h.get());check(cadence_ordinal(hr.get())==5&&mask_value(hr.get(),17)==1,"fifth_interval_restores_legal_decisions");
 history.reset();history.reset_cadence();h=fixture(c,13,2160);history.process(h.get());h=fixture(c,14,2180);hr=history.process(h.get());check(cadence_ordinal(hr.get())==0,"explicit_worker_encoder_reset_clears_phase");
 auto changed_round=[&](int seq,int ticks){auto s=fixture(c,seq,ticks);cJSON_ReplaceItemInObjectCaseSensitive(s.get(),"round_identity_sha256",cJSON_CreateString(std::string(64,'c').c_str()));return s;};
 h=changed_round(15,2200);hr=history.process(h.get());check(!boolean(get(hr.get(),"ready")),"changed_round_derivative_warmup");
 h=changed_round(16,2220);hr=history.process(h.get());check(cadence_ordinal(hr.get())==0,"changed_worker_round_restarts_phase");
 h=changed_round(17,2240);auto* final_round=cJSON_GetObjectItemCaseSensitive(h.get(),"round");cJSON_ReplaceItemInObjectCaseSensitive(final_round,"active",cJSON_CreateFalse());cJSON_ReplaceItemInObjectCaseSensitive(final_round,"result_value",cJSON_CreateNumber(1));
 hr=history.process(h.get());check(cadence_ordinal(hr.get())==1&&mask_value(hr.get(),17)==1,"terminal_bypasses_hold_mask_and_does_not_infer");
 h=changed_round(18,2260);history.process(h.get());h=changed_round(19,2280);hr=history.process(h.get());check(cadence_ordinal(hr.get())==0,"terminal_clears_cadence_for_next_recurrent_history");
 auto out=object();text(out.get(),"event","encoder_tests");flag(out.get(),"ok",true);number(out.get(),"assertions",checks);flag(out.get(),"simulation_stepped",false);emit(out.get());
}
}
int main(int argc,char** argv){try{require(argc==2,"encoder-test PRIVATE_MODEL_XML");Calibration c(argv[1]);self_test(c);tests(c);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
