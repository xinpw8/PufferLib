#include "../native_bot1.cuh"
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

namespace {
using namespace rek5_bot1;
int checks=0;
void require(bool condition,const char* label){++checks;if(!condition){std::fprintf(stderr,"Bot1 fixture failed: %s\n",label);std::exit(2);}}
bool near(float a,float b){return fabsf(a-b)<1e-6f;}
struct FixedRandom {
    float draws[8];int cursor=0,ints=0;
    float value(){return draws[cursor++];}
    int integer(int){++ints;return 0;}
};
Catalog catalog(){return {{1,1,1,2,2,1,3,3,4,4,1,2,1,1,1,1,1}};}
}
int main(){
    auto c=catalog();State s{};activate(s,123);Input i{1,0,.02f,2,2,false,false,false,true};
    require(s.phase==Settling&&s.timer==initial_delay,"activation settling delay");
    FixedRandom r{{.5f}};s.timer=0;auto d=update(s,i,c,r);
    require(d.move==11,"facing-only attack beyond distance gate, preferred right punch");
    require(r.ints==16,"both full and preferred reservoirs consumed");
    attack_result(s,i,true);require(s.phase==Attacking&&s.timer==3&&s.accepted==1,"execution acceptance");
    i.punching=true;d=update(s,i,c,r);require(s.phase==Attacking&&!d.clear_punching,"punching waits");
    auto cmd=locomotion(s,i);require(cmd.forward==0&&cmd.strafe==0&&cmd.yaw==0,"attacking velocity zero");
    s.timer=0;d=update(s,i,c,r);require(d.clear_punching&&s.phase==Recovering&&s.timer==recovery_time,"timeout clears punching");
    cmd=locomotion(s,i);require(cmd.forward==0&&cmd.yaw==0,"recovery velocity zero");
    r={{0,.75f,.25f,.5f}};s.timer=0;update(s,i,c,r);
    require(s.phase==Repositioning&&s.repositions==1,"recovery probabilistic reposition");
    require(near(s.reposition.forward,-.15f)&&near(s.reposition.strafe,-.434f)&&near(s.reposition.yaw,.225f),"reposition continuous signs");
    require(near(s.timer,(reposition_min+reposition_max)*.5f),"random reposition duration");
    s.timer=0;update(s,i,c,r);require(s.phase==Engaging&&s.timer==max_engage,"reposition expiry engages");
    i.distance=2;i.angle_degrees=90;s.timer=0;s.min_timer=0;update(s,i,c,r);
    require(s.phase==Engaging&&s.timer==max_engage&&s.min_timer==min_footwork,"engage expiry restarts without attack");
    i.distance=stop+.15f;i.angle_degrees=0;engage(s,i);require(s.min_timer==0,"skip minimum footwork close facing");
    i.distance=stop+.3f;update(s,i,c,r);require(s.phase==Settling&&s.timer==settle_time,"inclusive engage stop+.3");
    i.distance=stop+.5f+.001f;i.angle_degrees=34;update(s,i,c,r);require(s.phase==Settling,"settle far but facing continues");
    i.angle_degrees=35;update(s,i,c,r);require(s.phase==Engaging,"settle far and not facing reengages");
    s.phase=Settling;s.timer=0;i.distance=stop+.3f;i.angle_degrees=35;d=update(s,i,c,r);
    require(s.phase==Engaging&&d.move<0,"expiry strict distance and strict facing boundaries");
    s.phase=Settling;s.timer=0;i.distance=.5f;i.angle_degrees=90;r={{0}};d=update(s,i,c,r);
    require(d.move==9,"distance-only attack, right kick");attack_result(s,i,false);
    require(s.phase==Engaging&&s.rejected==1,"execution rejection restarts engagement");
    s.phase=Settling;s.timer=0;i.round_elapsed=.1f;d=update(s,i,c,r);require(d.move<0&&s.phase==Engaging,"initial round gate");
    i.round_elapsed=2;i.opponent_down=true;update(s,i,c,r);require(s.phase==GivingRoom&&s.timer==1,"opponent down gives room");
    cmd=locomotion(s,i);require(cmd.forward==-.25f,"give-room backward command");
    i.opponent_down=false;s.timer=.5f;update(s,i,c,r);require(s.phase==GivingRoom,"upright grace retained");
    s.timer=0;update(s,i,c,r);require(s.phase==Engaging,"upright grace expiry");
    i.own_recovery=true;d=update(s,i,c,r);cmd=locomotion(s,i);
    require(d.unsupported_recovery&&cmd.forward==0&&cmd.yaw==0,"unsupported own recovery explicit zero");i.own_recovery=false;
    i.distance=.44f;i.angle_degrees=0;i.time_seconds=.5f;engage(s,i);cmd=locomotion(s,i);
    require(near(cmd.forward,.24f)&&near(cmd.strafe,-sinf(1.f)*.15f),"close-facing footwork overrides approach");
    require(facing_yaw(17.5f)==0&&near(facing_yaw(45),-1.5f)&&near(facing_yaw(-45),1.5f),"yaw dead zone and native sign");
    r={{.5f}};require(pick_attack(c,-1,r)==16,"left punch includes assigned emote16");
    Catalog sparse{};sparse.primary_limb[3]=2;r={{0}};require(pick_attack(sparse,-1,r)==3,"category and side fallback");
    State repeat{};activate(s,981);activate(repeat,981);Random ra{s.rng},rb{repeat.rng};int counts[17]={};
    for(int n=0;n<20000;n++){int a=pick_attack(c,-1,ra),b=pick_attack(c,-1,rb);require(a==b,"seeded stream reproducibility");counts[a]++;}
    int kicks=counts[6]+counts[7];require(kicks>4700&&kicks<5300,"kick probability measured");
    for(int move:{0,1,2,5,10,12,13,14,15,16})require(counts[move]>1300&&counts[move]<1700,"preferred punch reservoir uniformity");
    require(counts[3]+counts[4]+counts[8]+counts[9]+counts[11]==0,"opposite-side pool excluded when preferred exists");
    std::printf("{\"event\":\"native_bot1_cpu_fixtures\",\"checks\":%d,\"failures\":0,\"draws\":20000,\"kick_count\":%d,\"server_parity\":false}\n",checks,kicks);
}
