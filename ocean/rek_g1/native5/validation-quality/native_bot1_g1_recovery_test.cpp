#include "../native_bot1.cuh"
#include <cstdio>
#include <cstdlib>

namespace {
using namespace rek5_bot1;
int checks=0;
void require(bool condition,const char* label){
    ++checks;
    if(!condition){std::fprintf(stderr,"Bot1 G1 recovery fixture failed: %s\n",label);std::exit(2);}
}
bool near(float a,float b){return fabsf(a-b)<1e-6f;}
bool zero(Command c){return c.forward==0&&c.strafe==0&&c.yaw==0;}
struct NoRandom {
    int calls=0;
    float value(){++calls;return .5f;}
    int integer(int){++calls;return 0;}
};
Catalog catalog(){return {{1,1,1,2,2,1,3,3,4,4,1,2,1,1,1,1,1}};}
Input input(){return {2,45,.125f,2,2,true,false,true,true};}
RecoveryInput fallen(){return {true,false,false,false,false,0,3.f};}
void lifecycle(){
    State s{};activate(s,123);s.phase=Attacking;s.timer=1.25f;s.min_timer=.17f;
    s.reposition={-.1f,.2f,.3f};s.attempts=9;s.accepted=6;s.rejected=3;s.repositions=2;
    RecoveryState recovery{1.5f,true,true};
    require(!activate_g1(s,true)&&s.phase==Attacking&&s.timer==1.25f,"already active activation is inert");
    auto v=deactivate_g1(s,true);
    require(s.phase==Inactive&&v.write&&zero(v.command),"deactivate writes zero when sink present");
    require(s.timer==1.25f&&s.min_timer==.17f&&s.rng==123,"deactivate preserves timers and RNG");
    require(!deactivate_g1(s,true).write,"already inactive deactivation writes nothing");
    require(!activate_g1(s,false)&&s.phase==Inactive,"activation without sink rejected");
    require(activate_g1(s,true)&&s.phase==Settling&&s.timer==initial_delay,"round activation settling delay");
    require(s.min_timer==.17f&&s.rng==123&&s.reposition.strafe==.2f,"round activation preserves retained state");
    require(s.attempts==9&&s.accepted==6&&s.rejected==3&&s.repositions==2,"round activation preserves counters");
    require(recovery.fault_estop_timer==1.5f&&recovery.fault_estop_engaged&&recovery.straighten_issued,
        "lifecycle leaves recovery latches intact");
    v=deactivate_g1(s,false);
    require(s.phase==Inactive&&!v.write,"deactivate still changes phase without sink");
    auto i=input();auto own=fallen();auto c=catalog();NoRandom rng;
    auto d=update_g1(s,recovery,i,own,c,rng);
    require(s.timer==initial_delay&&!d.write_zero_velocity&&!d.toggle_estop&&d.special==NoSpecial,
        "inactive Update has no timer or recovery work");
    require(recovery.fault_estop_timer==1.5f&&rng.calls==0,"inactive Update preserves recovery and RNG");
}
void recovery_sequence(){
    auto i=input();auto own=fallen();auto c=catalog();NoRandom rng;
    for(int phase=Engaging;phase<=GivingRoom;phase++){
        State s{};activate(s,321);s.phase=phase;s.timer=.0625f;s.min_timer=.75f;
        RecoveryState recovery{};
        auto d=update_g1(s,recovery,i,own,c,rng);
        require(s.phase==phase&&near(s.timer,-.0625f)&&s.min_timer==.75f,"own fall preserves tactical phase and min timer");
        require(d.write_zero_velocity&&d.special==Dampen&&!d.toggle_estop,"undampened fall requests dampen");
        require(d.tactical.move==-1&&!d.tactical.clear_punching&&!d.tactical.unsupported_recovery,
            "own fall bypasses attack and cancellation without unsupported fallback");
        require(s.rng==321&&s.attempts==0,"own fall leaves RNG and attack counters unchanged");
    }
    require(rng.calls==0,"recovery uses no random draws");
    State s{};activate(s,99);s.phase=Attacking;s.timer=3;
    RecoveryState recovery{};
    auto d=update_g1(s,recovery,i,own,c,rng);
    auto result=g1_local_special_result(d.special,false,own);
    require(result.supported&&result.accepted&&result.enter_dampen,"local G1 accepts undampened request");
    recovery_special_result(recovery,d.special,result.accepted);
    require(!recovery.straighten_issued&&!own.dampened,"dampen result does not synthesize input or straighten success");
    own.dampened=true;
    for(int n=0;n<40;n++){
        d=update_g1(s,recovery,i,own,c,rng);
        result=g1_local_special_result(d.special,false,own);
        require(d.special==Straighten&&result.supported&&!result.accepted&&!result.enter_dampen,
            "current Sonic G1 repeats failed Straighten each Update");
        recovery_special_result(recovery,d.special,result.accepted);
    }
    require(!recovery.straighten_issued&&s.phase==Attacking&&s.timer<0,
        "failed Straighten adds no get-up delay or tactical phase change");
    own.fallen=false;own.dampened=false;
    d=update_g1(s,recovery,i,own,c,rng);
    require(d.tactical.clear_punching&&s.phase==Recovering&&s.timer==recovery_time,
        "post-reset tactical continuation consumes expired attack timer");
    require(s.rng==99&&rng.calls==0,"counted-body-reset continuation does not reseed");
}
void fault_cycle(){
    State s{};activate(s,1);s.phase=Engaging;
    RecoveryState recovery{};auto own=fallen();own.motor_shutdown_hold=true;
    auto i=input();auto c=catalog();NoRandom rng;
    for(int n=1;n<=24;n++){
        auto d=update_g1(s,recovery,i,own,c,rng);
        require(d.toggle_estop==(n==24),"fault engage request exactly at configured three seconds");
        require(d.special==Dampen,"fault cycle still drives recovery command in same Update");
        auto result=g1_local_special_result(d.special,false,own);
        require(result.supported&&!result.accepted&&!result.enter_dampen,"motor hold rejects dampen");
    }
    require(recovery.fault_estop_engaged&&recovery.fault_estop_timer==0,"engage resets fault timer");
    for(int n=1;n<=4;n++){
        auto d=update_g1(s,recovery,i,own,c,rng);
        require(d.toggle_estop==(n==4),"fault release request exactly at half second");
    }
    require(!recovery.fault_estop_engaged&&recovery.fault_estop_timer==0,"release resets fault timer");
    own.fault_estop_delay=.25f;
    require(!update_g1(s,recovery,i,own,c,rng).toggle_estop,"serialized fault delay supplied explicitly");
    require(update_g1(s,recovery,i,own,c,rng).toggle_estop,"explicit fault delay honored");
    own.motor_shutdown_hold=false;
    auto d=update_g1(s,recovery,i,own,c,rng);
    require(!d.toggle_estop&&!recovery.fault_estop_engaged&&recovery.fault_estop_timer==0,
        "cleared motor hold resets fault cycle without release request");
    own.motor_shutdown_hold=true;i.delta_seconds=8;
    d=update_g1(s,recovery,i,own,c,rng);
    require(d.toggle_estop&&recovery.fault_estop_engaged&&recovery.fault_estop_timer==0,
        "long Update toggles once and discards overshoot");
}
void result_and_guard_semantics(){
    auto own=fallen();
    for(int remote=0;remote<2;remote++)for(int hold=0;hold<2;hold++)for(int dampened=0;dampened<2;dampened++){
        own.motor_shutdown_hold=hold;own.dampened=dampened;
        auto r=g1_local_special_result(Dampen,remote,own);
        const bool accept=!remote&&!hold&&!dampened;
        require(r.supported&&r.accepted==accept&&r.enter_dampen==accept,"Dampen native guard truth table");
        r=g1_local_special_result(Straighten,remote,own);
        require(r.supported&&!r.accepted&&!r.enter_dampen,"G1 Straighten false across native guard combinations");
    }
    State s{};activate(s,1);s.phase=Attacking;s.timer=3;
    RecoveryState recovery{};own=fallen();own.dampened=true;
    auto i=input();auto c=catalog();NoRandom rng;
    recovery_special_result(recovery,Straighten,true);
    auto d=update_g1(s,recovery,i,own,c,rng);
    require(recovery.straighten_issued&&d.special==NoSpecial,"successful feedback waits for recovery-armed flag");
    own.recovery_armed=true;
    d=update_g1(s,recovery,i,own,c,rng);
    require(d.special==GetUpProne,"zero suggested orientation requests prone");
    own.suggested_get_up_orientation=1;
    d=update_g1(s,recovery,i,own,c,rng);
    require(d.special==GetUpSupine,"nonzero suggested orientation requests supine");
    require(!g1_local_special_result(GetUpProne,false,own).supported&&
        !g1_local_special_result(GetUpSupine,false,own).supported,
        "unexpected get-up path is explicit unsupported for current G1 assets");
    recovery_special_result(recovery,Dampen,false);
    recovery_special_result(recovery,GetUpSupine,false);
    require(recovery.straighten_issued,"non-Straighten results do not change latch");
    own.dampened=false;
    d=update_g1(s,recovery,i,own,c,rng);
    require(d.special==Dampen&&recovery.straighten_issued,"Dampen branch preserves existing straighten latch");
    own.fallen=false;own.runner_recovering=true;
    d=update_g1(s,recovery,i,own,c,rng);
    require(d.write_zero_velocity&&d.special==NoSpecial&&!recovery.straighten_issued,
        "upright runner recovery clears straighten latch and skips tactics");
    recovery.straighten_issued=true;recovery.fault_estop_timer=.75f;recovery.fault_estop_engaged=true;
    own.runner_recovering=false;i.punching=true;s.phase=Attacking;s.timer=3;
    d=update_g1(s,recovery,i,own,c,rng);
    require(!d.write_zero_velocity&&recovery.straighten_issued&&!recovery.fault_estop_engaged&&recovery.fault_estop_timer==0,
        "ordinary tactical branch clears only fault cycle, ignoring compact own_recovery flag");
    recovery_special_result(recovery,Straighten,false);
    require(!recovery.straighten_issued,"Straighten failure feedback explicitly clears latch");
}
void cancel_and_cadence(){
    State s{};activate(s,1);s.phase=Attacking;s.timer=0;
    RecoveryState recovery{};RecoveryInput own{};own.fault_estop_delay=3;
    auto i=input();auto c=catalog();NoRandom rng;
    auto d=update_g1(s,recovery,i,own,c,rng);
    require(d.tactical.clear_punching&&s.phase==Recovering&&s.timer==recovery_time,
        "expired playing attack requests cancellation");
    s.phase=Attacking;s.timer=2;i.punching=false;
    d=update_g1(s,recovery,i,own,c,rng);
    require(!d.tactical.clear_punching&&s.phase==Recovering&&s.timer==recovery_time,
        "completed attack enters recovery without redundant cancellation");
    s.phase=Engaging;s.timer=1;s.min_timer=1;
    require(zero(locomotion(s,i)),"legacy own-recovery locomotion remains zero");
    auto v=fixed_locomotion_g1(s,i,true);
    require(v.write&&v.command.forward>0,"native FixedUpdate does not add an own-fall gate");
    require(!fixed_locomotion_g1(s,i,false).write,"inactive input sink means no velocity write");
    own=fallen();
    Command applied{.2f,.3f,.4f};
    d=update_g1(s,recovery,i,own,c,rng);
    if(d.write_zero_velocity)applied={};
    require(zero(applied),"Update recovery explicitly writes zero");
    v=fixed_locomotion_g1(s,i,true);if(v.write)applied=v.command;
    require(applied.forward>0,"subsequent FixedUpdate overwrites command when sink active");
    d=update_g1(s,recovery,i,own,c,rng);if(d.write_zero_velocity)applied={};
    require(zero(applied),"subsequent Update overwrites FixedUpdate command");
    s.phase=Inactive;
    require(!fixed_locomotion_g1(s,i,true).write,"inactive brain skips FixedUpdate");
}
}
int main(){
    lifecycle();recovery_sequence();fault_cycle();result_and_guard_semantics();cancel_and_cadence();
    std::printf("{\"event\":\"native_bot1_g1_recovery_cpu_fixtures\",\"checks\":%d,\"failures\":0,\"physical_runtime_bound\":false,\"server_parity\":false}\n",checks);
}
