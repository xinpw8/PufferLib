#include "contact_entry.h"
#include "recovered_contact_rules.cuh"
#include <cstdio>
#include <cstring>

namespace {
unsigned checks=0;
void check(bool ok,const char* why){++checks;if(!ok)throw std::runtime_error(why);}
void equal(const rek_contact_entry::State& a,const rek_contact_entry::State& b){
    check(a.words[0]==b.words[0]&&a.words[1]==b.words[1],"pair history differs");
}
unsigned rng=0x291ec31u;
float uniform(){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return float(rng&0xffffffu)/float(0x1000000u);}
rek5_primitive::Shape shape(int kind,float x,float y,float z,float turn){
    rek5_primitive::Shape s{};s.kind=kind;s.center[0]=x;s.center[1]=y;s.center[2]=z;
    float q[4]={cosf(turn*.5f),0,0,sinf(turn*.5f)};
    rek5_primitive::quaternion_matrix(q,s.axes);s.size[0]=.2f;s.size[1]=.17f;s.size[2]=.12f;return s;
}
}
int main(){try{
    using namespace rek_contact_entry;
    State reference{},optimized{};
    auto center=shape(rek5_primitive::Sphere,0,0,0,0),left=center,right=center;
    left.center[0]=-1;right.center[0]=1;
    // Idle crossing enters/exits within the tick, leaving no endpoint latch.
    auto crossed=sample(reference,107,left,right,center,center,8);
    unscored_endpoint(optimized,107,right,center);equal(reference,optimized);
    check(crossed.entered&&crossed.overlap_after_start,"idle transient fixture did not cross");
    auto r=sample(reference,107,right,left,center,center,8),o=sample(optimized,107,right,left,center,center,8);
    check(r.entered==o.entered&&r.overlap_after_start==o.overlap_after_start&&r.entered,"later active crossing changed");equal(reference,optimized);
    // Idle endpoint overlap suppresses a fabricated entry at attack start.
    sample(reference,0,left,center,center,center,8);unscored_endpoint(optimized,0,center,center);equal(reference,optimized);
    r=sample(reference,0,center,center,center,center,8);o=sample(optimized,0,center,center,center,center,8);
    check(!r.entered&&!o.entered&&r.overlap_after_start==o.overlap_after_start,"persistent idle contact became attack entry");
    sample(reference,0,center,left,center,center,8);unscored_endpoint(optimized,0,left,center);equal(reference,optimized);
    r=sample(reference,0,left,center,center,center,8);o=sample(optimized,0,left,center,center,center,8);
    check(r.entered&&o.entered,"separation and later active reentry changed");equal(reference,optimized);

    RekG1HitDetectorState score_ref{},score_opt{};
    auto config=rek5_recovered::rek_g1_current_build_hit_detector_config();
    {
        RekG1ImpactEvent event{.02f,.1f,.1f,1.f,REK_G1_AIM_LIMB_LEFT_LOWER_BODY};RekG1StrikeIntent intent{};
        intent.impact_events=&event;intent.impact_event_count=1;intent.clip_cursor_frames=50;intent.clip_fps=50;
        intent.move_id=1;intent.action_playing=intent.layer_active=1;
        int32_t apex=-1;float ramp=0;
        check(!rek5_recovered::embedded_strike_intent_apex(&intent,REK_G1_BODY_PART_FOOT,REK_G1_HAND_LEFT,config.apex_min_ramp,&apex,&ramp),"invalid-apex fixture is eligible");
        reference={};optimized={};
        r=sample(reference,0,left,center,center,center,8);unscored_endpoint(optimized,0,center,center);equal(reference,optimized);
        auto rejected=rek5_recovered::score(score_ref,config,intent,0,0,3.f,1.f);
        check(r.entered&&rejected.points==0&&rejected.reject==rek5_recovered::Apex,"invalid apex scored");
        check(!std::memcmp(&score_ref,&score_opt,sizeof(score_ref)),"invalid apex mutated score state");
        intent.clip_cursor_frames=1;
        check(rek5_recovered::embedded_strike_intent_apex(&intent,REK_G1_BODY_PART_FOOT,REK_G1_HAND_LEFT,config.apex_min_ramp,&apex,&ramp),"valid-apex fixture is ineligible");
        r=sample(reference,0,center,center,center,center,8);o=sample(optimized,0,center,center,center,center,8);
        check(!r.entered&&!o.entered,"apex activation fabricated contact entry");equal(reference,optimized);
        intent.clip_cursor_frames=50;
        sample(reference,0,center,left,center,center,8);unscored_endpoint(optimized,0,left,center);equal(reference,optimized);
        r=sample(reference,0,left,right,center,center,8);unscored_endpoint(optimized,0,right,center);
        check(r.entered&&reference.words[0]==0,"invalid-apex transient fixture did not enter then exit");equal(reference,optimized);
        intent.clip_cursor_frames=1;intent.move_id=2;
        r=sample(reference,0,right,center,center,center,8);o=sample(optimized,0,right,center,center,center,8);
        check(r.entered&&o.entered,"valid apex lost later reentry");equal(reference,optimized);
        auto sr=rek5_recovered::score(score_ref,config,intent,0,0,3.f,2.f),so=rek5_recovered::score(score_opt,config,intent,0,0,3.f,2.f);
        check(sr.points==2&&sr.points==so.points&&!std::memcmp(&score_ref,&score_opt,sizeof(score_ref)),"later apex-valid score changed");
    }
    unsigned active_entries=0,idle_transients=0,broad_rejects=0,score_points=0,apex_rejected=0;
    for(int tick=0;tick<256;tick++){
        if(tick%31==0){
            reference={};optimized={};score_ref={};score_opt={};
            // Unused bits must survive all updates to the 108 measured pairs.
            reference.words[1]=optimized.words[1]=0xa55a500000000000ull;
        }
        const bool active=tick%7>=3;const int samples=(tick%4==0?1:tick%4==1?2:tick%4==2?8:16);
        for(int pair=0;pair<Pairs;pair++){
            const int limb=pair%6;
            const RekG1AimLimb aim=limb<2||limb>=4?(limb%2?REK_G1_AIM_LIMB_RIGHT_LOWER_BODY:REK_G1_AIM_LIMB_LEFT_LOWER_BODY):(limb%2?REK_G1_AIM_LIMB_RIGHT_UPPER_BODY:REK_G1_AIM_LIMB_LEFT_UPPER_BODY);
            RekG1ImpactEvent event{.02f,.1f,.1f,1.f,aim};RekG1StrikeIntent intent{};
            intent.impact_events=&event;intent.impact_event_count=1;intent.clip_cursor_frames=tick%5==0?50.f:1.f;intent.clip_fps=50;
            intent.move_id=tick+1;intent.action_playing=intent.layer_active=1;
            const auto part=limb<2?REK_G1_BODY_PART_FOOT:limb<4?REK_G1_BODY_PART_HAND:REK_G1_BODY_PART_SHIN;
            const auto hand=limb%2?REK_G1_HAND_RIGHT:REK_G1_HAND_LEFT;
            int32_t apex=-1;float ramp=0;
            const bool can_score=active&&rek5_recovered::embedded_strike_intent_apex(&intent,part,hand,config.apex_min_ramp,&apex,&ramp);
            apex_rejected+=active&&!can_score;
            const int ka=pair%3,kb=(pair/3)%3;
            auto old_a=shape(ka,2*uniform()-1,2*uniform()-1,.2f*(uniform()-.5f),6*uniform());
            auto a=shape(ka,2*uniform()-1,2*uniform()-1,.2f*(uniform()-.5f),6*uniform());
            auto old_b=shape(kb,.4f*(uniform()-.5f),.4f*(uniform()-.5f),0,6*uniform());
            auto b=shape(kb,.4f*(uniform()-.5f),.4f*(uniform()-.5f),0,6*uniform());
            const bool broad=(tick+pair)%11!=0;
            Sampled expected{},actual{};
            if(!broad){update(reference,pair,false);update(optimized,pair,false);++broad_rejects;}
            else{
                expected=sample(reference,pair,old_a,a,old_b,b,samples);
                if(can_score)actual=sample(optimized,pair,old_a,a,old_b,b,samples);
                else unscored_endpoint(optimized,pair,a,b);
            }
            equal(reference,optimized);
            if(active){
                if(can_score)check(expected.entered==actual.entered&&expected.overlap_after_start==actual.overlap_after_start,"scorable flags differ after optimized unscored history");
                active_entries+=expected.entered;
                rek5_recovered::Result sr{},so{};
                if(expected.entered)sr=rek5_recovered::score(score_ref,config,intent,0,limb,3.f,tick*.02f);
                if(can_score&&actual.entered)so=rek5_recovered::score(score_opt,config,intent,0,limb,3.f,tick*.02f);
                check(sr.points==so.points,"later active score differs");
                if(can_score)check(sr.reject==so.reject&&sr.apex==so.apex,"scorable result differs");
                check(!std::memcmp(&score_ref,&score_opt,sizeof(score_ref)),"score cooldown/dedup state differs");score_points+=sr.points;
            }else if(expected.entered&&!(reference.words[pair/64]&(std::uint64_t(1)<<(pair%64))))++idle_transients;
        }
    }
    check(active_entries>0&&idle_transients>0&&broad_rejects>0&&score_points>0&&apex_rejected>0,"equivalence corpus missing required cases");
    std::printf("{\"test\":\"contact_unscored_history_cpu\",\"checks\":%u,\"pairs_per_tick\":108,\"ticks\":256,\"active_entries\":%u,\"idle_transients\":%u,\"broad_rejects\":%u,\"score_points\":%u,\"apex_rejected\":%u,\"passed\":true}\n",checks,active_entries,idle_transients,broad_rejects,score_points,apex_rejected);
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 2;}}
