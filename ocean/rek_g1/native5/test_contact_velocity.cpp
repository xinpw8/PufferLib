#include "contact_velocity.h"
#include <cstdio>
int main(){try{
    using namespace rek_contact_velocity;unsigned checks=0;
    auto check=[&](bool value,const char* why){++checks;if(!value)throw std::runtime_error(why);};
    check(parse(nullptr)==Mode::LegacySphereProxy,"default mode changed");
    check(parse("legacy_sphere_proxy_v1")==Mode::LegacySphereProxy,"explicit default");
    check(parse("body_cvel_v1")==Mode::BodyCvel,"opt-in mode");
    for(const char* invalid:{"","cvel","body_cvel_v2","true"}) {
        bool rejected=false;try{parse(invalid);}catch(const std::invalid_argument&){rejected=true;}check(rejected,"unknown mode accepted");
    }
    for(int pairs=0;pairs<2;pairs++)for(int primitive=0;primitive<2;primitive++)for(int scoring=0;scoring<3;scoring++) {
        check(compatible(Mode::LegacySphereProxy,pairs,primitive,scoring),"legacy compatibility changed");
        check(compatible(Mode::BodyCvel,pairs,primitive,scoring)==bool(pairs&&primitive&&scoring==2),"unsupported contact mode accepted");
    }
    for(int limb=0;limb<6;limb++)check(limb_slot(limb)==limb,"striker slot");
    const int targets[]={6,7,7,8,9,10,11,12,13};
    for(int target=0;target<9;target++)check(target_slot(target)==targets[target],"target slot");
    FastBodyVelocityFrame frame{};frame.root_com[0][0]=.2f;frame.root_com[0][1]=.3f;
    frame.linear[0][0][0]=2;frame.linear[0][0][1]=-3;frame.linear[0][0][2]=.5f;
    const auto v=compose(frame,0,0,0,1,2,4,true);
    check(std::abs(v.x-1.8f)<1e-6f&&std::abs(v.y+.2f)<1e-6f&&v.z==.5f,"canonical composition");
    const auto held=compose(frame,0,0,0,1,2,4,false);
    check(std::abs(held.x+.2f)<1e-6f&&std::abs(held.y-2.8f)<1e-6f&&held.z==0,"held frame retained clip rate");
    check(relative_speed({1,2,3},{1,2,3})==0,"same velocity");
    check(relative_speed({1,2,3},{4,6,3})==5,"relative norm");
    frame.root_com[1][0]=-.4f;frame.linear[1][7][0]=9;
    check(compose(frame,1,target_slot(1),0,0,0,0,true).x==9,"opponent body slot");
    check(compose(frame,1,target_slot(2),0,0,0,0,true).x==9,"same torso body mismatch");
    std::printf("{\"test\":\"contact_velocity_helpers\",\"checks\":%u,\"passed\":true}\n",checks);return 0;
}catch(const std::exception& error){std::fprintf(stderr,"%s\n",error.what());return 1;}}
