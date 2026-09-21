#include "contact_entry.h"
#include <cstdio>
int main(){try{
    unsigned checks=0;auto check=[&](bool ok,const char* why){checks++;if(!ok)throw std::runtime_error(why);};
    using namespace rek_contact_entry;
    check(parse(nullptr)==Mode::LegacyLimbUnion,"default changed");
    check(parse("legacy_limb_union_v1")==Mode::LegacyLimbUnion,"explicit legacy");
    check(parse("geom_pair_v1")==Mode::GeomPair,"opt in");
    bool rejected=false;try{parse("body_pair_v1");}catch(const std::invalid_argument&){rejected=true;}check(rejected,"unknown mode accepted");
    State state{};
    for(int p=0;p<Pairs;p++)check(update(state,p,true),"first entry missing, possibly64bit truncation");
    for(int p=0;p<Pairs;p++)check(!update(state,p,true),"persistent pair reentered");
    for(int p=0;p<Pairs;p++)check(!update(state,p,false),"exit became entry");
    for(int p=0;p<Pairs;p++)check(update(state,p,true),"reentry missing");
    state={};check(update(state,8*9+1,true),"torso box entry");
    check(update(state,8*9+2,true),"same-body distinct capsule must enter");
    check(!update(state,8*9+1,true),"other collider cleared history");
    state={};check(update(state,107,true),"round reset/last pair");
    rek5_primitive::Shape a{},b{};a.kind=b.kind=rek5_primitive::Sphere;
    a.size[0]=b.size[0]=.1f;a.axes[0]=a.axes[4]=a.axes[8]=1;b.axes[0]=b.axes[4]=b.axes[8]=1;
    auto before=a;before.center[0]=-1;a.center[0]=1;state={};
    auto result=sample(state,107,before,a,b,b,8);
    check(result.entered&&result.overlap_after_start,"sampled crossing absent");
    check(state.words[1]==0,"transient enter exit remained latched");
    result=sample(state,107,a,before,b,b,8);check(result.entered,"reverse transient crossing absent");
    before=a=b;a.size[0]=before.size[0]=b.size[0]=.1f;state={};update(state,0,true);
    result=sample(state,0,before,a,b,b,8);check(!result.entered&&result.overlap_after_start,"persistent sample reentered");
    // Move/intent/cooldown changes have no API for clearing pair state.
    result=sample(state,0,before,a,b,b,8);check(!result.entered,"new move must not reset pair");
    a.center[0]=1;result=sample(state,0,before,a,b,b,8);check(!result.entered,"exit only produced entry");
    result=sample(state,0,a,before,b,b,8);check(result.entered,"separation reentry denied");
    std::printf("{\"test\":\"contact_entry\",\"checks\":%u,\"pairs\":108,\"passed\":true}\n",checks);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 2;}}
