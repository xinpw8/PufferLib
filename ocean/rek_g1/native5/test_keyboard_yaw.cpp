#include "keyboard_yaw.h"
#include "action_cadence.h"
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <stdexcept>

unsigned checks=0;
void check(bool ok,const char* why){checks++;if(!ok)throw std::runtime_error(why);}
bool same(float a,float b){return std::memcmp(&a,&b,sizeof(float))==0;}
// Independently transcribed valid-input C# expression. No physical omega state.
float reference(rek_keyboard_yaw::State& state,float raw,float dt,float ramp,float speed){
    if(ramp<=0||raw==0){state={0,0};return raw*speed;}
    float sign=raw>0?1:-1;
    float value=state.sign==sign?state.ramp:0;
    value=std::min(1.f,value+dt/ramp);
    state={value,sign};return value*sign*speed;
}
int main(){try{
    using namespace rek_keyboard_yaw;
    check(parse(nullptr)==Mode::LegacyVelocitySlew,"default legacy");
    check(parse("legacy_velocity_slew_v1")==Mode::LegacyVelocitySlew,"explicit legacy");
    check(parse("keyboard_reset_v1")==Mode::KeyboardReset,"explicit keyboard");
    for(const char* bad:{"","keyboard_reset","0","1"}){
        bool rejected=false;try{parse(bad);}catch(const std::invalid_argument&){rejected=true;}
        check(rejected,"unknown mode accepted");
    }
    State s{};
    check(same(advance(s,1,.02f,.5f,1),.04f),"first positive tick");
    for(int i=1;i<30;i++)advance(s,1,.02f,.5f,1);
    check(s.ramp==1&&s.sign==1,"sustained saturation");
    check(advance(s,0,.02f,.5f,1)==0&&s.ramp==0&&s.sign==0,"release resets immediately");
    check(same(advance(s,1,.02f,.5f,1),.04f),"repress starts at first tick");
    check(same(advance(s,-1,.02f,.5f,1),-.04f),"reversal starts new sign immediately");
    State before=s;check(same(advance(s,-1,0,.5f,1),-.04f)&&same(s.ramp,before.ramp),"zero dt retained sign");
    check(advance(s,1,0,.5f,1)==0&&s.sign==1,"zero dt reversal clears old ramp");
    check(advance(s,.25f,.02f,0,2)==.5f&&s.ramp==0&&s.sign==0,"nonpositive ramp direct branch");
    check(same(advance(s,.25f,.02f,.5f,1),.04f),"positive ramp uses sign not magnitude");
    uint32_t rng=419;State actual{},expected{};
    for(int i=0;i<20000;i++){
        rng=1664525u*rng+1013904223u;
        float raw=float(int(rng%3)-1),dt=i%7? .02f:0.f;
        float ramp=i%101?.5f:0.f,speed=i%2?1.f:1.8f;
        float a=advance(actual,raw,dt,ramp,speed),b=reference(expected,raw,dt,ramp,speed);
        check(same(a,b)&&same(actual.ramp,expected.ramp)&&same(actual.sign,expected.sign),"C# expression mismatch");
    }
    // Policy hold0 retains the owned category. A five-row mask must not pause
    // the 50 Hz command ramp; busy ticks reset it even with retained yaw.
    State cadence{},every{};int held=1;
    for(int tick=0;tick<50;tick++){
        int action=rek_action_cadence::decision(5,tick)?6:0;
        if(action>0)held=action;
        bool busy=tick>=10&&tick<20;
        float a=advance(cadence,busy?0.f:held==6?1.f:0.f,.02f,.5f,1);
        float b=advance(every,busy?0.f:1.f,.02f,.5f,1);
        check(same(a,b),"hold0 or cadence changed command clock");
        if(busy)check(cadence.ramp==0&&cadence.sign==0,"busy must reset ramp");
        if(tick==20)check(same(a,.04f),"postbusy starts fresh");
    }
    std::printf("{\"test\":\"keyboard_yaw_cpu\",\"checks\":%u,\"passed\":true,\"physical_parity\":false}\n",checks);
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 2;}}
