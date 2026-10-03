// CPU tests for lite_falls.h and lite_falls_loader.h.
// g++ -std=c++17 -O2 -I. test_lite_falls.cpp ../../../vendor/cJSON.c -lcrypto -o test_lite_falls
// ./test_lite_falls lite_falls_smoke_v1.json
#include "lite_falls_loader.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

using namespace rek_lite_falls;
static int failures=0,checks=0;
static void check(bool ok,const char* what){checks++;if(!ok){failures++;std::printf("FAIL %s\n",what);}}
static void close_to(double a,double b,double tolerance,const char* what){check(std::fabs(a-b)<=tolerance,what);}

static void test_bins(){
    check(route_move(8)==7,"route 8 is native move 7, the HH left front kick");
    check(route_move(7)==6&&route_move(10)==9&&route_move(11)==0&&route_move(23)==16,"route registry order");
    check(route_move(0)==-1&&route_move(6)==-1&&route_move(24)==-1,"locomotion routes have no move");
    check(move_class(7)==2&&move_class(0)==1&&move_class(-1)==0,"move classes");
    check(move_bin(7,0.f,false,false)==28&&move_bin(7,.99f,false,false)==31,"phase bins");
    check(move_bin(7,1.f,false,false)==31&&move_bin(7,-.1f,false,false)==28,"phase clamps");
    check(move_bin(-1,0,true,true)==kTranslateBin&&move_bin(-1,0,false,true)==kYawBin&&move_bin(-1,0,false,false)==kIdleBin,"locomotion bins");
    const float edges[4]={.5f,1.f,1.5f,2.f};
    check(bucket(.49f,edges,4)==0&&bucket(.5f,edges,4)==1&&bucket(5.f,edges,4)==4,"bucket edges are lower-inclusive");
}

static void test_sampling(const Model& m){
    const float* q=m.delay_quantiles+Stuck*kQuantiles;
    check(sample_ticks(q,0.f)==int(std::ceil(q[0])),"u=0 gives the minimum quantile");
    check(sample_ticks(q,1.f)==int(std::ceil(q[kQuantiles-1])),"u=1 gives the maximum quantile");
    check(sample_ticks(q,.0625f)==int(std::ceil(.5f*(q[0]+q[1]))),"interpolates between quantiles");
    const float zeros[kQuantiles]={};
    check(sample_ticks(zeros,.5f)==1,"delays are at least one tick");
    check(sample_outcome(m,2,0.f)==Recover,"lowest u recovers");
    check(sample_outcome(m,2,.999f)==Stuck,"highest u is stuck");
    const float* p=m.outcome_probability+2*kOutcomes;
    check(sample_outcome(m,2,p[0]+.5f*p[1])==FallenQuick,"middle u falls quickly");
}

static void test_hazard(const Model& m){
    // Smoke model: kick in mid phase at close range is far riskier than idling.
    Features idle=features(m,-1,0,false,false,-1,0,false,false,3.f,0.f,false);
    Features kick=features(m,7,.4f,false,false,-1,0,false,false,.4f,0.f,false);
    Features struck=features(m,-1,0,false,false,-1,0,false,false,3.f,0.f,true);
    close_to(logit(m,idle),m.bias,1e-6,"idle far logit is the bias");
    check(hazard(m,kick)>100*hazard(m,idle),"close kick hazard dominates idle");
    check(hazard(m,struck)>hazard(m,idle),"received hit raises hazard");
    Features extreme=idle;Model big=m;big.bias=200;check(hazard(big,extreme)==1.f,"large logits saturate without overflow");
    big.bias=-200;check(hazard(big,extreme)==0.f,"large negative logits saturate");
    check(knockdown(true,1.4f,1.0f)&&!knockdown(true,1.6f,1.0f),"weak strike window is 1.5 s");
    check(knockdown(true,4.9f,6.5f)&&!knockdown(true,5.1f,6.5f),"strong strike window is 5 s");
    check(!knockdown(false,0.f,10.f),"no strike is a slip");
}

static void test_fall_paths(const Model& m){
    Features x=features(m,7,.4f,false,false,-1,0,false,false,.4f,0,false);
    FighterState f{};
    check(!try_onset(m,f,x,7,false,0,0,1.f,0,0),"u=1 never starts a fall");
    f.grace_ticks=1;check(!try_onset(m,f,x,7,false,0,0,0.f,0,0),"spawn grace blocks onset");
    check(!advance_fall(f)&&f.grace_ticks==0,"grace counts down while upright");
    check(try_onset(m,f,x,7,false,0,0,0.f,0.f,0.f),"u=0 starts a fall");
    check(f.phase==Falling&&(f.events&FallingStarted)&&f.outcome==Recover&&f.classification==Slip,"onset state");
    int guard=0;while(f.phase==Falling&&guard++<1000)advance_fall(f);
    check(f.phase==Upright&&(f.events&FallingCleared),"recover outcome returns upright");
    FighterState g{};
    check(try_onset(m,g,x,7,true,.2f,3.f,0.f,.999f,0.f),"struck onset");
    check(g.outcome==Stuck&&g.classification==Knockdown,"stuck knockdown");
    bool fallen=false;guard=0;while(!fallen&&guard++<1000)fallen=advance_fall(g);
    check(fallen&&g.phase==Fallen&&(g.events&BecameFallen),"stuck eventually becomes fallen");
    check(g.ticks==0&&!advance_fall(g)&&g.ticks==1,"fallen elapsed ticks advance");
    Model off=m;off.enabled=0;FighterState h{};
    check(!try_onset(off,h,x,7,false,0,0,0.f,0,0),"disabled model never falls");
}

static void test_referee(){
    ArenaState s{};int award[2];
    s.fighter[1].classification=Slip;s.fighter[1].phase=Fallen;
    on_fallen(s,1);
    check(s.referee.count_active[1]&&(s.referee.calls&CallSlip),"single slip count");
    for(int t=0;t<kCountTicks-1;t++)check(!advance_referee(s,award)&&award[0]==0&&award[1]==0,"count runs 3 s");
    check(advance_referee(s,award)&&award[0]==kKnockoutPoints&&award[1]==0,"expiry awards the opponent 5");
    check(!counting(s.referee)&&(s.referee.calls&CallKnockout),"expiry clears the count");

    ArenaState d{};d.fighter[0].classification=Knockdown;d.fighter[1].classification=Slip;
    on_fallen(d,0);for(int t=0;t<100;t++)advance_referee(d,award);
    on_fallen(d,1);
    check(d.referee.count_ticks==0&&(d.referee.calls&CallDoubleKnockdown),"double down restarts the count");
    for(int t=0;t<kCountTicks-1;t++)check(!advance_referee(d,award),"double count runs a fresh 3 s");
    check(advance_referee(d,award)&&award[0]==kKnockoutPoints&&award[1]==kKnockoutPoints,"double expiry awards both");

    // The reported edge case: fighter 0 is down but uncounted (stuck falling)
    // while fighter 1 is counted. Fighter 1's expiry pays fighter 0 and the
    // spawn reset clears fighter 0 before it is ever counted.
    ArenaState e{};
    e.fighter[0].phase=Falling;e.fighter[0].outcome=Stuck;e.fighter[0].resolve_ticks=400;
    e.fighter[1].phase=Fallen;e.fighter[1].classification=Slip;on_fallen(e,1);
    bool expired=false;int t=0;
    for(;t<kCountTicks&&!expired;t++){
        check(!advance_fall(e.fighter[0]),"stuck fighter stays uncounted");
        expired=advance_referee(e,award);
    }
    check(expired&&award[0]==kKnockoutPoints&&award[1]==0,"only the uncounted fighter scores 5");
    reset_to_spawn(e);
    check(upright(e.fighter[0])&&upright(e.fighter[1])&&!counting(e.referee),"spawn reset clears the stuck fighter");
    check(e.fighter[0].grace_ticks==kSpawnGraceTicks&&e.fighter[1].grace_ticks==kSpawnGraceTicks,"spawn reset sets the 2 s grace");
}

static std::string read(const char* path){std::ifstream f(path);std::stringstream s;s<<f.rdbuf();return s.str();}
static bool rejects(std::string text){try{parse(text);return false;}catch(const std::exception&){return true;}}
static std::string replace(std::string text,const std::string& from,const std::string& to){
    const size_t at=text.find(from);if(at==std::string::npos){std::printf("missing fixture text %s\n",from.c_str());std::exit(2);}
    return text.replace(at,from.size(),to);
}

int main(int argc,char** argv){
    if(argc!=2){std::fprintf(stderr,"usage: test_lite_falls MODEL.json\n");return 2;}
    const Loaded loaded=load(argv[1]);const Model& m=loaded.model;
    check(m.enabled==1&&loaded.file_sha256.size()==64&&!loaded.model_id.empty(),"model loads");
    test_bins();test_sampling(m);test_hazard(m);test_fall_paths(m);test_referee();
    const std::string text=read(argv[1]);
    check(!rejects(text),"fixture parses");
    check(rejects(replace(text,"rek.lite_falls.v1","rek.lite_falls.v0")),"rejects schema");
    check(rejects(replace(text,"\"phase_bins\": 4","\"phase_bins\": 5")),"rejects dimensions");
    check(rejects(replace(text,"\"distance_edges_m\": [","\"distance_edges_m\": [9, ")),"rejects edge count");
    check(rejects(replace(text,"\"outcome_probability\": [\n  [\n   0.85,","\"outcome_probability\": [\n  [\n   0.95,")),"rejects outcome probabilities not summing to one");
    check(rejects(replace(text,"\"delay_quantiles_ticks\": [\n  [\n   8.0,","\"delay_quantiles_ticks\": [\n  [\n   80.0,")),"rejects decreasing delay quantiles");
    check(rejects(replace(text,"\"bias\": -11.0","\"bias\": \"x\"")),"rejects nonnumeric weights");
    std::printf("%s: %d checks, %d failures\n",failures?"FAIL":"PASS",checks,failures);
    return failures?1:0;
}
