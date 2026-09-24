#include "authentic_trajectory.h"
#include <cstdio>
using namespace rek_authentic;
int main(int argc,char** argv){try{
 require(argc==3,"Usage: test-score-delta BASELINE_V3 SCORE_DELTA_V5");
 const auto old_bytes=read_file(argv[1]),bytes=read_file(argv[2]);const auto old=decode(old_bytes),data=decode(bytes);
 require(old.format_version==3&&data.format_version==5&&data.rows.size()==15078&&data.sequences.size()==5,"unexpected source shape");
 require(old.checkpoint_sha256==data.checkpoint_sha256&&old.worker_sha256==data.worker_sha256&&old.native_object_sha256==data.native_object_sha256,"behavior identity changed");
 size_t checks=0;
 for(size_t i=0;i<data.rows.size();++i){const size_t at=IDENTITY_HEADER_BYTES+i*ROW_BYTES;
  for(size_t j=0;j<ROW_BYTES;++j)if(!(j>=1072&&j<1076)&&!(j>=1080&&j<1084))require(old_bytes[at+j]==bytes[at+j],"protected recorded row byte changed");
  ++checks;
 }
 for(size_t i=0;i<data.sequences.size();++i){const auto& seq=data.sequences[i];double sum=0;size_t rejected=0;
  for(size_t j=seq.begin;j<seq.end;++j){const auto&r=data.rows[j];require(r.behavior_seed==901+i,"seed mismatch");sum+=r.reward;rejected+=r.policy_weight==0;}
  const auto&last=data.rows[seq.end-1];require(std::abs(sum-double(last.next_own_points-last.next_opponent_points)/5)<1e-6,"score sum mismatch");
  require(rejected==1&&last.policy_weight==0&&last.terminal_after,"terminal rejected actor weight changed");++checks;
 }
 for(size_t offset:{size_t(1072),size_t(1080)}){auto edited=bytes;const float invalid=.123f;std::memcpy(edited.data()+IDENTITY_HEADER_BYTES+offset,&invalid,4);bool rejected=false;try{decode(edited);}catch(const std::exception&){rejected=true;}require(rejected,"mutated contract accepted");++checks;}
 std::printf("{\"score_delta_cpu_checks\":%zu,\"rows\":%zu,\"passed\":true}\n",checks,data.rows.size());return 0;
}catch(const std::exception&e){std::fprintf(stderr,"score_delta_test: %s\n",e.what());return 2;}}
